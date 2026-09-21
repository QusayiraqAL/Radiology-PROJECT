// Builds the static site Vercel serves.
//
// There is one source of truth for the page - "Radiology Hub.html" at the repo root,
// the same file main.py hands out at "/" - and this script copies it to public/index.html
// rather than keeping a second copy in the tree. A second copy is a file that drifts:
// every fix to the served page would have to be made twice, and the one nobody remembered
// to update is the one the public URL shows.
//
// The only edit made on the way through is an optional API_BASE, which becomes
// window.__API_BASE__ and enters the page's candidate list ahead of location.origin.
// Set it when the API has a fixed public URL:
//
//     API_BASE=https://api.example.com npm run build
//
// Leave it unset and visitors point the page at their own server through the field in
// the API bar (or a ?api=<url> link), which is remembered per browser.
import { mkdir, readFile, writeFile, rm } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const SRC = join(root, 'Radiology Hub.html');
const OUT_DIR = join(root, 'public');
const OUT = join(OUT_DIR, 'index.html');

const apiBase = (process.env.API_BASE || '').trim().replace(/\/+$/, '');

let html = await readFile(SRC, 'utf8');

if (apiBase) {
  if (!/^https?:\/\//.test(apiBase)) {
    throw new Error(`API_BASE must be an absolute http(s) URL, got: ${apiBase}`);
  }
  // JSON.stringify escapes quotes and the "</script>" sequence is impossible in a URL,
  // so this is safe to drop into an inline script.
  const tag = `<script>window.__API_BASE__ = ${JSON.stringify(apiBase)};</script>`;
  const at = html.indexOf('</head>');
  if (at === -1) throw new Error('no </head> in the page - cannot inject API_BASE');
  html = html.slice(0, at) + '  ' + tag + '\n' + html.slice(at);
}

await rm(OUT_DIR, { recursive: true, force: true });
await mkdir(OUT_DIR, { recursive: true });
await writeFile(OUT, html, 'utf8');

console.log(`[build] ${SRC} -> ${OUT} (${(html.length / 1024).toFixed(1)} KB)`);
console.log(`[build] API_BASE: ${apiBase || '(unset - resolved in the browser)'}`);
