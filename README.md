# theAdamColton.github.io

Astro site, deployed to GitHub Pages on push to `main`.

```sh
npm install
npm run dev        # http://localhost:4321
```

## Adding a post

```sh
npm run new-post -- "My Post Title"
```

This creates `src/content/posts/my-post-title.mdx`. Fill in the frontmatter
`description` (and optionally `cover`, `links`, `section`) and the post shows up
on the home page automatically, sorted by `date`. Put images/videos in
`public/my-post-title/`.

## Images

Put full quality images straight into `public/` and reference them by their
original filename; no need to resize or compress them. The build
(`src/integrations/compress-images.mjs`) scales them to at most 1600px wide and
converts them to WebP, rewriting the html to match. `npm run dev` serves the
same compressed images, so what you see locally is what gets deployed.

Format and quality are set in `astro.config.mjs`. Results are cached by content
hash, so only new or changed images take time.

## Adding a home page entry that isn't a post

Add a `.md` file to `src/content/projects/` with `title`, `url`, and `date`
frontmatter; the body is the blurb. See the existing files for examples.
