# Storefront — small business web studio site

A single-page marketing website for **Storefront**, a web design &
development studio that builds websites for small businesses.

> "Storefront" is a placeholder brand — rename it, swap the colors, and drop in
> real contact details before going live. Every string lives in one file.

## What's here

- `index.html` — the entire, self-contained website. No build step, no
  dependencies, no external requests. Inline CSS and a small vanilla-JS block
  handle the theme toggle (light/dark), mobile menu, and scroll reveals.

## Preview it

Open the file directly:

```bash
open website/index.html          # macOS
xdg-open website/index.html      # Linux
```

Or serve it locally:

```bash
cd website && python3 -m http.server 8000
# then visit http://localhost:8000
```

## Sections

- **Hero** with a live browser-frame mockup of a sample client site
- **Services** — website builds, online stores, local SEO, booking, copy, care,
  and custom software
- **Custom software** — a dedicated feature panel (booking systems, portals,
  inventory, dashboards, automations, integrations)
- **How it works** — a four-step process from discovery call to launch
- **Pricing** — three build tiers, a custom-software callout, monthly care
  plans, and add-ons
- **FAQ** and a closing call-to-action

## Make it yours

Everything is editable in `index.html`:

| Change | Where |
|---|---|
| Brand name | Search for `Storefront` |
| Colors | The `:root` CSS variables near the top (`--accent`, `--amber`, …) |
| Prices | The `#pricing` section |
| Contact email / phone | Search for `hello@storefront.studio` and `tel:+` |

## Deploy

Because it's one static file, it hosts anywhere — Netlify, Vercel, GitHub
Pages, Cloudflare Pages, or any static web host. Drag the `website/` folder in,
or point the host at this repo's `website/` directory.
