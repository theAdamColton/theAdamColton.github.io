// Compresses images so public/ can hold full quality originals while the site
// serves small files. Posts always reference the original filename.
//
// - build: images in dist/ are resized, recompressed and optionally converted
//   to `format`; references to converted files in the html are rewritten.
// - dev: the same compressed bytes are served at the original urls, so what
//   you see locally is what gets deployed.
//
// Results are cached by content hash, so only new or changed images take time.
import { mkdir, readdir, readFile, rm, stat, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";
import path from "node:path";
import sharp from "sharp";

const CACHE_DIR = "node_modules/.cache/compress-images";
// bump when changing the encoders below to invalidate the cache
const CACHE_VERSION = 1;
// keep the original unless compressing saves at least this much
const MIN_SAVINGS = 0.1;
const CONCURRENCY = 4;

const INPUT_EXTS = [".jpg", ".jpeg", ".png", ".webp", ".avif", ".gif"];
const MIME = {
	".jpg": "image/jpeg",
	".jpeg": "image/jpeg",
	".png": "image/png",
	".webp": "image/webp",
	".avif": "image/avif",
	".gif": "image/gif",
};

const encoders = {
	".jpg": (img, q) => img.jpeg({ quality: q, mozjpeg: true }),
	".jpeg": (img, q) => img.jpeg({ quality: q, mozjpeg: true }),
	".png": (img, q) => img.png({ palette: true, quality: Math.max(q, 90), effort: 10 }),
	".webp": (img, q) => img.webp({ quality: q, effort: 6 }),
	// avif quality numbers run lower than webp/jpeg for the same look
	".avif": (img, q) => img.avif({ quality: Math.round(q * 0.75) }),
	".gif": (img) => img.gif({ effort: 10, interFrameMaxError: 8 }),
};

/**
 * @param {object} [options]
 * @param {"webp" | "avif" | null} [options.format] convert images to this format;
 *   null keeps each image's own format. sharp can't write animated avif, so
 *   animated images become webp instead.
 * @param {number} [options.quality] 1-100, lower is smaller and blurrier
 * @param {number} [options.maxWidth] wider images are scaled down
 * @param {boolean} [options.dev] also serve compressed images from `astro dev`
 */
export default function compressImages({
	format = "webp",
	quality = 75,
	maxWidth = 1600,
	dev = true,
} = {}) {
	const settings = JSON.stringify([CACHE_VERSION, format, quality, maxWidth]);
	const inflight = new Map();
	let publicDir;

	// -> { data, ext }; ext is the original extension if it was left alone
	async function compressed(input, ext) {
		const hash = createHash("sha1").update(settings).update(input).digest("hex");
		if (!inflight.has(hash)) inflight.set(hash, lookupOrEncode(hash, input, ext));
		return inflight.get(hash);
	}

	async function lookupOrEncode(hash, input, ext) {
		await mkdir(CACHE_DIR, { recursive: true });
		for (const e of INPUT_EXTS) {
			const data = await readFile(path.join(CACHE_DIR, hash + e)).catch(() => null);
			if (data) return { hash, ext: e, data: e === ext && data.length === 0 ? input : data };
		}

		const { pages = 1 } = await sharp(input).metadata();
		let outExt = format ? `.${format}` : ext;
		if (outExt === ".avif" && pages > 1) outExt = ".webp";
		const img = sharp(input, { animated: true })
			// metadata is dropped, so bake in phone photos' EXIF rotation
			.rotate()
			.resize({ width: maxWidth, withoutEnlargement: true });
		let data = await encoders[outExt](img, quality).toBuffer();
		if (data.length > input.length * (1 - MIN_SAVINGS)) {
			[data, outExt] = [input, ext];
		}
		// an empty cache file means "keep the original"
		await writeFile(path.join(CACHE_DIR, hash + outExt), data === input ? "" : data);
		return { hash, ext: outExt, data };
	}

	return {
		name: "compress-images",
		hooks: {
			"astro:config:done": ({ config }) => {
				publicDir = fileURLToPath(config.publicDir);
			},

			"astro:server:setup": ({ server }) => {
				if (!dev) return;
				server.middlewares.use(async (req, res, next) => {
					try {
						const urlPath = decodeURIComponent(new URL(req.url, "http://x").pathname);
						const ext = path.extname(urlPath).toLowerCase();
						if (!INPUT_EXTS.includes(ext)) return next();
						const file = path.join(publicDir, urlPath);
						if (!file.startsWith(publicDir) || !(await stat(file).catch(() => null))) {
							return next();
						}
						const out = await compressed(await readFile(file), ext);
						res.setHeader("Content-Type", MIME[out.ext]);
						res.setHeader("Cache-Control", "no-cache");
						res.end(out.data);
					} catch (e) {
						next(e);
					}
				});
			},

			"astro:build:done": async ({ dir, logger }) => {
				const root = fileURLToPath(dir);
				const all = await readdir(root, { recursive: true });
				const images = all.filter((f) => INPUT_EXTS.includes(path.extname(f).toLowerCase()));
				// names already taken, by an original or by an earlier conversion
				const taken = new Set(all);
				const renamed = new Map(); // old path -> new path, relative to dist
				const used = new Set();
				let before = 0;
				let after = 0;
				const start = Date.now();

				const queue = [...images];
				await Promise.all(
					Array.from({ length: CONCURRENCY }, async () => {
						for (let rel; (rel = queue.shift()); ) {
							const file = path.join(root, rel);
							const ext = path.extname(rel).toLowerCase();
							try {
								const input = await readFile(file);
								const out = await compressed(input, ext);
								used.add(out.hash);
								before += input.length;
								after += out.data.length;
								if (out.data === input) continue;

								if (out.ext === ext) {
									await writeFile(file, out.data);
								} else {
									let newRel = rel.slice(0, -ext.length) + out.ext;
									// e.g. lenna.jpg and lenna.png both want lenna.webp
									if (taken.has(newRel)) newRel = rel + out.ext;
									taken.add(newRel);
									await writeFile(path.join(root, newRel), out.data);
									renamed.set(rel.split(path.sep).join("/"), newRel.split(path.sep).join("/"));
								}
							} catch (e) {
								logger.warn(`skipped ${rel}: ${e.message}`);
							}
						}
					}),
				);

				// point html at the converted files
				for (const rel of all.filter((f) => f.endsWith(".html"))) {
					const file = path.join(root, rel);
					const pageUrl = new URL(rel.split(path.sep).join("/"), "http://x/");
					const html = await readFile(file, "utf8");
					const rewritten = html.replace(
						/(\s(?:src|href|poster)=)(["'])(.*?)\2/g,
						(match, attr, quote, value) => {
							const url = URL.parse(value.replaceAll("&amp;", "&"), pageUrl);
							if (url?.host !== "x") return match;
							const target = decodeURIComponent(url.pathname).slice(1);
							const newTarget = renamed.get(target);
							if (!newTarget) return match;
							// swap e.g. ".png" for ".webp" (or ".png.webp")
							const oldExt = path.extname(target);
							const newSuffix = newTarget.slice(target.length - oldExt.length);
							const i = value.toLowerCase().lastIndexOf(oldExt.toLowerCase());
							return `${attr}${quote}${value.slice(0, i)}${newSuffix}${value.slice(i + oldExt.length)}${quote}`;
						},
					);
					if (rewritten !== html) await writeFile(file, rewritten);
				}

				// remove originals of converted images, unless a script or stylesheet
				// (which we don't rewrite) mentions them
				const texts = await Promise.all(
					all
						.filter((f) => /\.(js|css|json)$/.test(f))
						.map((f) => readFile(path.join(root, f), "utf8")),
				);
				for (const [oldRel] of renamed) {
					const name = path.basename(oldRel);
					if (texts.some((t) => t.includes(name) || t.includes(encodeURI(name)))) {
						logger.warn(`kept ${oldRel}: still referenced somewhere`);
						continue;
					}
					await rm(path.join(root, oldRel));
				}

				// drop cache entries for images that no longer exist or have changed
				for (const f of await readdir(CACHE_DIR)) {
					if (!used.has(path.parse(f).name)) await rm(path.join(CACHE_DIR, f));
				}

				const mb = (n) => (n / 1e6).toFixed(1);
				logger.info(
					`${images.length} images: ${mb(before)} MB -> ${mb(after)} MB, ` +
						`${renamed.size} converted to ${format}, ` +
						`in ${((Date.now() - start) / 1000).toFixed(1)}s`,
				);
			},
		},
	};
}
