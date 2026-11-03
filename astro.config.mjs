import { defineConfig } from "astro/config";
import mdx from "@astrojs/mdx";
import { unified } from "@astrojs/markdown-remark";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeCitation from "rehype-citation";
import remarkDetectMath from "./src/plugins/remark-detect-math.mjs";
import compressImages from "./src/integrations/compress-images.mjs";

export default defineConfig({
	integrations: [
		mdx(),
		// format: "webp" | "avif" | null (keep each image's format)
		// quality: 1-100; lower is smaller and blurrier
		compressImages({ format: "webp", quality: 75 }),
	],
	// /post-name -> post-name.html, matching the old SvelteKit output
	build: { format: "file" },
	trailingSlash: "never",
	markdown: {
		// build-time syntax highlighting, no client JS
		shikiConfig: { theme: "one-dark-pro" },
		processor: unified({
			smartypants: false,
			remarkPlugins: [remarkMath, remarkDetectMath],
			rehypePlugins: [
				rehypeKatex,
				// [@key] in a post resolves against src/refs.bib
				[rehypeCitation, { bibliography: "src/refs.bib", linkCitations: true }],
			],
		}),
	},
});
