# Juseong's Website

This site is now structured as a lightweight Jekyll site compatible with GitHub Pages.

## Adding a Blog Post

Create a new Markdown file in `_posts/` using the filename format `YYYY-MM-DD-title.md`.

Example:

```md
---
layout: post
title: "Post Title"
date: 2026-04-08 09:00:00 -0500
description: "Short summary"
---
Write your post here.
```

## Structure

- Homepage: `index.html`
- Layouts: `_layouts/`
- Shared includes: `_includes/`
- Main stylesheet: `assets/css/main.css`
- Blog index: `blog/index.html`
- Posts: `_posts/`

## Local Preview

If Jekyll is installed locally, run:

```bash
jekyll serve
```

Then open `http://127.0.0.1:4000`.
