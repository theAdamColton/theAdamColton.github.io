import { defineCollection } from "astro:content";
import { glob } from "astro/loaders";
import { z } from "astro/zod";

// Fields that control how an entry shows up as a card on the home page.
const card = {
	date: z.coerce.date(),
	section: z.enum(["projects", "adventures"]).default("projects"),
	// image/video path(s) under public/, shown next to the blurb
	cover: z
		.union([z.string(), z.array(z.string())])
		.optional()
		.transform((c) => (c === undefined ? [] : [c].flat())),
	coverCaption: z.string().optional(),
	// extra links shown on the card, e.g. { github: "https://..." }
	links: z.record(z.string(), z.string()).default({}),
};

// A blog post: src/content/posts/<slug>.mdx is served at /<slug> and
// automatically listed on the home page, using `description` as the blurb.
const posts = defineCollection({
	loader: glob({ pattern: "*.mdx", base: "./src/content/posts" }),
	schema: z.object({
		title: z.string(),
		description: z.string().optional(),
		// set when the post renders its own title instead of the default header
		customHeader: z.boolean().default(false),
		...card,
	}),
});

// A home page entry that isn't a post (a repo, paper, pdf, ...).
// The file body is the blurb.
const projects = defineCollection({
	loader: glob({ pattern: "*.{md,mdx}", base: "./src/content/projects" }),
	schema: z.object({
		title: z.string(),
		url: z.string(),
		...card,
	}),
});

export const collections = { posts, projects };
