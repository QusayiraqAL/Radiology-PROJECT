// Runs against the DEPLOYED page, not a local copy of it - the difference matters, because
// the two failures this caught are both properties of the deployment rather than of the
// file: a static host answering /health with an HTML 404, and the browser's local address
// space rule. Neither exists when the page is served from the machine the API is on.
//
//   npm run test:live            # or LIVE_URL=https://... npm run test:live
//
// The second scenario is expected to FAIL to connect. That is the assertion: it captures
// what the browser says when the deployed page is pointed at an API on a private address,
// and that message is quoted in DEPLOY.md.
import { chromium } from 'playwright-core';
import { createServer } from 'node:http';
import { networkInterfaces } from 'node:os';

const LIVE = (process.env.LIVE_URL || 'https://ai-powered-radiology-hub.vercel.app')
  .replace(/\/+$/, '');
const EDGE = process.env.EDGE_PATH ||
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
const LAN = Object.values(networkInterfaces()).flat()
  .filter(i => i && i.family === 'IPv4' && !i.internal).map(i => i.address)[0];
if (!LAN) { console.error('no non-loopback IPv4 - cannot run the address-space check'); process.exit(2); }

const api = createServer((req, res) => {
  res.setHeader('access-control-allow-origin', '*');
  if (req.url === '/health') {
    res.writeHead(200, {'content-type':'application/json'});
    return res.end(JSON.stringify({ status:'ok', device:'cpu', models:{ a:true, b:true } }));
  }
  res.writeHead(404); res.end();
});
await new Promise(r => api.listen(8123, '0.0.0.0', r));

const browser = await chromium.launch({ executablePath: EDGE, headless: true });
const fail = [];
const check = (n, got, want) => {
  const ok = typeof want === 'function' ? want(got) : got === want;
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${n}\n        ${JSON.stringify(got)}`);
  if (!ok) fail.push(n);
};

async function load(url) {
  const ctx = await browser.newContext();
  const page = await ctx.newPage();
  const errs = [], blocked = [];
  page.on('pageerror', e => errs.push(e.message));
  page.on('console', m => { if (m.type() === 'error') blocked.push(m.text().slice(0, 400)); });
  await page.goto(url, { waitUntil: 'load', timeout: 45000 });
  await page.waitForFunction(() =>
    !document.getElementById('apiStatusText').textContent.includes('جارٍ'), null, { timeout: 30000 });
  const r = await page.evaluate(() => ({
    status: document.getElementById('apiStatusText').textContent.trim(),
    cls:    document.getElementById('apiStatus').className,
    field:  document.getElementById('apiUrl').value,
    step1:  (document.getElementById('realStep1')||{}).textContent?.trim().slice(0,55),
    metrics:(document.getElementById('metricsLoading')||{}).textContent?.trim().slice(0,40),
    // does the page itself render? headline + nav, not just the API bar
    h1:     (document.querySelector('h1')||{}).textContent?.trim().slice(0,40),
    secs:   document.querySelectorAll('section').length,
  }));
  await ctx.close();
  return { r, errs, blocked };
}

console.log(`\n--- LIVE: ${LIVE} (no API configured) ---`);
{
  const { r, errs } = await load(LIVE + '/');
  check('page renders (sections present)', r.secs, n => n >= 5);
  check('headline rendered', r.h1, s => !!s && s.length > 3);
  check('no JS errors on the deployed page', errs, e => e.length === 0);
  check('status offline', r.cls, c => c.includes('off'));
  check('advice names the FIELD, not start_server.bat',
        r.status, s => s.includes('اكتب عنوان الخادم') && !s.includes('start_server'));
  check('step 1 rewritten for a remote reader',
        r.step1, s => s && !s.includes('start_server'));
  check('metrics placeholder rewritten', r.metrics, s => s && !s.includes('start_server'));
  check('field did NOT keep the vercel.app origin',
        r.field, f => !f.includes('vercel.app'));
  check('field fell back to the local default', r.field, 'http://127.0.0.1:8000');
}

console.log(`\n--- LIVE + plain-http API via ?api= (the mixed-content claim in DEPLOY.md) ---`);
{
  const { r, blocked } = await load(`${LIVE}/?api=http://${LAN}:8123`);
  check('https page could NOT reach the http API', r.cls, c => c.includes('off'));
  check('browser reported it as blocked', blocked,
        b => b.some(x => /Mixed Content|insecure|blocked by CORS|ERR_FAILED/i.test(x)));
  console.log('        browser said: ' + (blocked[0] || '(nothing)'));
}

await browser.close(); api.close();
console.log(fail.length ? `\n${fail.length} FAILED: ${fail.join(', ')}` : '\nALL LIVE CHECKS PASSED');
process.exit(fail.length ? 1 : 0);
