// Scaffold a post: npm run new-post -- "My Post Title"
// Creates src/content/posts/<slug>.mdx, which also lists it on the home page.
import { existsSync, writeFileSync } from "node:fs";

const title = process.argv.slice(2).join(" ").trim();
if (!title) {
	console.error('usage: npm run new-post -- "My Post Title"');
	process.exit(1);
}

const slug = title
	.toLowerCase()
	.normalize("NFKD")
	.replace(/[\u0300-\u036f]/g, "")
	.replace(/[^a-z0-9]+/g, "-")
	.replace(/^-|-$/g, "");
const path = `src/content/posts/${slug}.mdx`;
if (existsSync(path)) {
	console.error(`${path} already exists`);
	process.exit(1);
}

const today = new Date().toISOString().slice(0, 10);
writeFileSync(
	path,
	`---
title: ${JSON.stringify(title)}
date: ${today}
# shown on the home page card and in search results
description: >-
  One or two sentences about the post.
# optional:
# section: adventures              # default: projects
# cover: ${slug}/cover.png         # image or video in public/, shown on the home page
# coverCaption: What the cover shows
# links:
#   github: https://github.com/theAdamColton/...
---

Write here. Media goes in public/${slug}/ and is referenced as "${slug}/file.png".

<MediaBox>
	<img src="${slug}/example.png" />
	A caption
</MediaBox>
`,
);
console.log(`created ${path}\nhttp://localhost:4321/${slug}`);
