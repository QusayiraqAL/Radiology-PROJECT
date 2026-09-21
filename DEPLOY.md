# DEPLOY — putting the Hub on Vercel

This file covers one thing: getting the front-end onto a public URL, and pointing it at an
API that can actually answer. For what the models are and what they score, read
[`api/README.md`](api/README.md); for how they got those scores, read
[`TRAINING_LOG.md`](TRAINING_LOG.md).

## The split, and the measurement behind it

**The page goes on Vercel. The API cannot, and this is not a configuration problem.**

Vercel runs a serverless function in a bundle capped at **250 MB unzipped**. Measured on
this machine, what `main.py` needs to start:

| What | Size | Against the 250 MB cap |
|---|---|---|
| `torch` alone (`api/venv/Lib/site-packages/torch`) | **509 MB** | **2.0× over**, before any other package |
| All serving dependencies (`site-packages`) | **1.4 GB** | **5.7× over** |
| Trained weights (`api/models/`) | **3.4 GB** | **13.9× over** |
| **Total the API opens** | **~4.8 GB** | **19× over** |

`torch` on its own is twice the entire budget, so there is no subset of the weights, no
pruning of `models/`, and no lazy-loading trick that brings this under the line. The API
needs a host that gives it a filesystem and RAM — the local box already does it in under a
second per request (TRAINING_LOG step 33: 14/14 models loaded, 0.10–0.87 s per prediction).

So: **Vercel serves the page, some other host serves the API,** and the page is built to be
told where that host is.

## Deploying

Two routes. They produce the same site; pick one and stay on it.

### Route A - connect the GitHub repo (deploys on push)

The repo is already at `QusayiraqAL/Radiology-PROJECT`. On
[vercel.com/new](https://vercel.com/new): **Add New > Project > Import** that repo, leave
**Root Directory** at the repo root, and deploy. Vercel reads `vercel.json` and needs no
framework preset, no build command typed in the dashboard, and no environment variable
unless you are setting `API_BASE`.

After that, **every push to the connected branch builds and deploys itself.** Push a branch
other than the production one and Vercel builds it as a preview URL instead.

One thing to know about this route: Vercel clones the repo, so what it builds is what is
**committed**. Weights are gitignored (`*.pt`, 3.4 GB of them), which is what keeps the
clone small — and it is also why the API can never ride along on this route.

### Route B - deploy from this machine

```bash
vercel login          # opens a browser - interactive, has to be you
vercel link           # pick the scope, name the project
vercel deploy --prod  # and again for every deploy
```

Route B uploads from the working tree rather than from a commit, which is what
`.vercelignore` is guarding.

### What both routes run

`vercel.json` runs `npm run build`, which is [`scripts/build-site.mjs`](scripts/build-site.mjs):
it copies `Radiology Hub.html` to `public/index.html` and nothing else. There is no second
copy of the page in the tree, because a second copy is a file that drifts — the fix nobody
remembered to make twice is the one the public URL shows.

`.vercelignore` is an **allowlist**: deny `*`, then name the four files the build reads.
Without it a deploy uploads `api/models` and `api/venv` — 4.8 GB across the wire for a
150 KB page. Verified: the rules resolve to exactly `Radiology Hub.html`, `package.json`,
`vercel.json`, `scripts/build-site.mjs`.

### If the API has a fixed public URL

Route A: add `API_BASE` under **Project Settings > Environment Variables**, then redeploy.
Route B: `API_BASE=https://your-api.example.com vercel deploy --prod`.

Either way it becomes `window.__API_BASE__` in the page and is tried before the page falls
back to guessing. Leave it unset and visitors supply the address themselves.

## How the page finds the API

`resolveApi()` walks a candidate list at load and keeps the first whose `/health` answers:

| # | Candidate | When it wins |
|---|---|---|
| 1 | `?api=<url>` in the query string | a shareable link that carries the server with it |
| 2 | the value stored in `localStorage` | a return visit after anything below worked once |
| 3 | `window.__API_BASE__` | `API_BASE` was set at build time |
| 4 | `location.origin` | the API is serving the page itself (local, or a tunnel) |
| 5 | `http://127.0.0.1:8000` | a `file://` copy, or a browser on the server box |

The winner is written back to `localStorage`, so the address is typed once per browser.

**Why this replaced a one-line default.** The page used to set the field to
`location.origin` for any `http(s)` origin. That was right when the only two http hosts
were the API itself and a tunnel to it. On a static host it is wrong in a way the visitor
cannot diagnose: `your-project.vercel.app/health` is a 404, so the bar read *"offline — run
start_server.bat first"* — advice that only the person sitting at the server can act on,
shown to someone who may be on a phone in another country.

## Connecting a local API to the public page

The page is on `https://`, and a browser will not let an `https://` page call a plain
`http://` server on the open internet. Two ways across:

1. **A tunnel (works today, no certificate work).** `cloudflared tunnel --url http://localhost:8000`
   or `ngrok http 8000` gives the local API a public `https://` address. Paste it into the
   API field, or share `https://your-project.vercel.app/?api=https://your-tunnel.trycloudflare.com`.
   `main.py` already sends `Access-Control-Allow-Origin: *`, so nothing else is needed.
2. **A real host.** Any VM or container with ~5 GB of disk and a few GB of RAM runs
   `main.py` unchanged. Set `API_BASE` at build time and the page needs no configuration.

`http://127.0.0.1:8000` from the public page is **not** a reliable third option: Chrome
gates requests from a public origin to a local address behind Private Network Access, and
what that costs you depends on the browser and its version. Use a tunnel.

## What is live without an API

The page itself: the whole site, the model descriptions, the disclaimers, the navigation.
What needs the API is everything that reports a number — the accuracy cards, prediction,
the quiz, the atlas, the report grader — and those stay empty until `/health` answers.

They are **not** stubbed with baked-in copies of the numbers, deliberately. This project's
rule is that the only trustworthy source for a served metric is what `main.py` loads
(TRAINING_LOG, "the current state" table), and a snapshot committed next to the page is
exactly the drift that rule exists to prevent: step 7 and the session-2 status table both
record a hand-written number going stale against the server. An empty card is honest. A
confidently wrong one is not.
