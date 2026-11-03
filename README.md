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

## Adding a home page entry that isn't a post

Add a `.md` file to `src/content/projects/` with `title`, `url`, and `date`
frontmatter; the body is the blurb. See the existing files for examples.
