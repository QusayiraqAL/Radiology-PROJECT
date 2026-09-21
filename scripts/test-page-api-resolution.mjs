// Drives the BUILT page (public/index.html) in a real browser and asserts what a reader
// actually sees, because the interesting failures here are not syntax errors - they are
// the page giving correct-looking advice to someone who cannot act on it.
//
//   npm run build && npm test
//
// Uses the Edge already on the machine rather than downloading a browser. Two of the four
// scenarios below failed on first run and are the reason onServerBox() ignores the API
// field and resolveApi() no longer falls back to location.origin.
//
import { chromium } from 'playwright-core';
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { networkInterfaces } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

// 127.0.0.1 is what the page treats as "the browser is on the server machine", so
// visiting the static host there tests the LOCAL branch, not the Vercel one. The remote
// case needs a hostname that is not loopback - the LAN address is the honest stand-in.
const LAN = Object.values(networkInterfaces()).flat()
  .filter(i => i && i.family === 'IPv4' && !i.internal).map(i => i.address)[0];
if (!LAN) { console.error('no non-loopback IPv4 - cannot simulate a remote host'); process.exit(2); }
console.log('remote host simulated as: ' + LAN);

const PAGE_DIR = process.argv[2] ||
  join(dirname(fileURLToPath(import.meta.url)), '..', 'public');
const EDGE = process.env.EDGE_PATH ||
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';

// Stand-in for the real API: answers /health and /models the way main.py does.
let apiHits = [];
const api = createServer((req, res) => {
  apiHits.push(req.url);
  res.setHeader('access-control-allow-origin', '*');
  if (req.url === '/health') {
    res.writeHead(200, {'content-type':'application/json'});
    return res.end(JSON.stringify({ status:'ok', device:'cpu',
      models:{ chest:true, pneumonia:true, brain:true, breast:true, derma:true, blood:true } }));
  }
  if (req.url === '/models') {
    res.writeHead(200, {'content-type':'application/json'});
    return res.end(JSON.stringify({ models:[
      { id:'brain', name_ar:'أورام الدماغ', metrics:{ accuracy:0.99 } },
      { id:'chest', name_ar:'أشعة الصدر', metrics:{ mean_auc:0.7525 } } ] }));
  }
  res.writeHead(404, {'content-type':'text/html'}); res.end('<h1>404</h1>');
});

// The static host. Everything but the page 404s as HTML - exactly Vercel.
const site = createServer(async (req, res) => {
  const p = req.url.split('?')[0];
  try {
    const b = await readFile(PAGE_DIR + (p === '/' ? '/index.html' : p));
    res.writeHead(200, {'content-type':'text/html; charset=utf-8'}); res.end(b);
  } catch { res.writeHead(404, {'content-type':'text/html'}); res.end('<h1>404</h1>'); }
});

await new Promise(r => api.listen(8123, r));
await new Promise(r => site.listen(8124, '0.0.0.0', r));

const browser = await chromium.launch({ executablePath: EDGE, headless: true });
const fail = [];
const check = (name, got, want) => {
  const ok = typeof want === 'function' ? want(got) : got === want;
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${name}\n        got: ${JSON.stringify(got)}`);
  if (!ok) fail.push(name);
};

async function run(label, url, clearStore) {
  console.log(`\n--- ${label} ---`);
  const ctx = await browser.newContext();
  const page = await ctx.newPage();
  const errs = [];
  page.on('pageerror', e => errs.push(e.message));
  await page.goto(url, { waitUntil: 'load' });
  await page.waitForFunction(() =>
    !document.getElementById('apiStatusText').textContent.includes('جارٍ'), null, { timeout: 20000 });
  const r = await page.evaluate(() => ({
    status: document.getElementById('apiStatusText').textContent.trim(),
    cls:    document.getElementById('apiStatus').className,
    field:  document.getElementById('apiUrl').value,
    step1:  (document.getElementById('realStep1')||{}).textContent?.trim().slice(0,60),
    metrics:(document.getElementById('metricsLoading')||{}).textContent?.trim().slice(0,45),
    stored: (()=>{try{return localStorage.getItem('radhub.apiBase')}catch(e){return 'ERR'}})(),
  }));
  await ctx.close();
  return { r, errs };
}

// --- A: static host, no API reachable -------------------------------------
{
  const { r, errs } = await run('A. static host, no API (the Vercel visitor today)',
    `http://${LAN}:8124/`);
  check('A no JS errors', errs, e => e.length === 0);
  check('A status is offline', r.cls, c => c.includes('off'));
  check('A message points at the FIELD, not start_server.bat',
        r.status, s => s.includes('اكتب عنوان الخادم') && !s.includes('start_server'));
  check('A step 1 rewritten away from start_server.bat',
        r.step1, s => s && !s.includes('start_server') && s.includes('عنوان الخادم'));
  check('A metrics placeholder rewritten',
        r.metrics, s => s && !s.includes('start_server'));
  check('A did NOT leave the static origin in the API field',
        r.field, f => !f.includes('8124'));
  check('A fell back to the documented local default',
        r.field, 'http://127.0.0.1:8000');
}

// --- B: static host + live API via ?api= -----------------------------------
{
  apiHits = [];
  const { r, errs } = await run('B. static host + live API via ?api=',
    `http://${LAN}:8124/?api=http://${LAN}:8123`);
  check('B no JS errors', errs, e => e.length === 0);
  check('B status is online', r.cls, c => c.includes('on'));
  check('B counted the ready models', r.status, s => s.includes('6') && s.includes('متصل'));
  check('B field settled on the API', r.field, `http://${LAN}:8123`);
  check('B remembered it in localStorage', r.stored, `http://${LAN}:8123`);
  check('B actually called /models', apiHits, h => h.includes('/models'));
}

// --- C: served from loopback, no API - the local branch must still say the .bat -------
{
  const { r, errs } = await run('C. loopback host, no API (someone at the server box)',
    'http://127.0.0.1:8124/');
  check('C no JS errors', errs, e => e.length === 0);
  check('C message still names start_server.bat',
        r.status, s => s.includes('start_server.bat'));
  check('C step 1 left alone', r.step1, s => s && s.includes('start_server'));
}

// --- D: a bad ?api= is kept in the field so it can be corrected -----------------------
{
  const { r } = await run('D. unreachable ?api= stays visible for editing',
    `http://${LAN}:8124/?api=http://${LAN}:8199`);
  check('D kept the address the reader supplied', r.field, `http://${LAN}:8199`);
}

await browser.close();
api.close(); site.close();
console.log(fail.length ? `\n${fail.length} FAILED: ${fail.join(', ')}` : '\nALL CHECKS PASSED');
process.exit(fail.length ? 1 : 0);
