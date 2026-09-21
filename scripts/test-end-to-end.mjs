// The whole chain, end to end: deployed page -> tunnel -> local API -> real model.
// Nothing is stubbed here - it uploads a real MRI from samples/ and asserts on what the
// trained network actually returned.
//
//   npm run test:e2e -- https://<your-tunnel>.trycloudflare.com
//
// Needs start_server.bat running and a tunnel pointing at it (see DEPLOY.md). The tunnel
// URL is different every time a quick tunnel starts, so it is an argument, not a constant.
import { chromium } from 'playwright-core';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const LIVE = (process.env.LIVE_URL || 'https://ai-powered-radiology-hub.vercel.app').replace(/\/+$/, '');
const TUNNEL = (process.argv[2] || process.env.API_URL || '').replace(/\/+$/, '');
const IMG = join(ROOT, 'samples', 'brain_glioma.png');
const EDGE = process.env.EDGE_PATH ||
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';

if (!TUNNEL) {
  console.error('usage: npm run test:e2e -- https://<tunnel-url>   (or set API_URL)');
  process.exit(2);
}

const browser = await chromium.launch({ executablePath: EDGE, headless: true });
const ctx = await browser.newContext();
const page = await ctx.newPage();
const errs = [];
page.on('pageerror', e => errs.push(e.message));

await page.goto(`${LIVE}/?api=${TUNNEL}`, { waitUntil: 'load', timeout: 60000 });
await page.waitForFunction(() =>
  !document.getElementById('apiStatusText').textContent.includes('جارٍ'), null, { timeout: 40000 });

const bar = await page.evaluate(() => ({
  status: document.getElementById('apiStatusText').textContent.trim(),
  cls: document.getElementById('apiStatus').className,
  field: document.getElementById('apiUrl').value,
}));
console.log('API bar   :', bar.status);
console.log('field     :', bar.field);

// Did the real measured metrics land in the cards? loadModels() is async and fires after
// the health probe resolves, so waiting on the API bar is NOT waiting on this.
await page.waitForFunction(() => {
  const g = document.getElementById('metricsGrid');
  return g && !g.textContent.includes('start_server') && g.textContent.trim().length > 60;
}, null, { timeout: 45000 });
const cards = await page.evaluate(() =>
  [...document.querySelectorAll('#metricsGrid .metric-card, #metricsGrid > div')]
    .slice(0, 3).map(d => d.textContent.replace(/\s+/g, ' ').trim().slice(0, 62)));
console.log('cards     :', cards.length);
cards.forEach(c => console.log('   -', c));

// Now a real prediction: pick brain, upload a real MRI, press analyze.
await page.click('button[data-model="brain"]');
await page.setInputFiles('#fileInput', IMG);
await page.waitForFunction(() => !document.getElementById('realAnalyzeBtn').disabled,
  null, { timeout: 15000 });
const t0 = Date.now();
await page.click('#realAnalyzeBtn');
await page.waitForFunction(() => {
  const t = document.getElementById('realResults').textContent;
  return t && !t.includes('١.') && t.trim().length > 40;
}, null, { timeout: 90000 });
const ms = Date.now() - t0;

const out = await page.evaluate(() =>
  document.getElementById('realResults').textContent.replace(/\s+/g, ' ').trim().slice(0, 260));
console.log(`\nprediction (${(ms/1000).toFixed(1)}s round trip through the tunnel):`);
console.log(out);
console.log('\nJS errors :', errs.length ? errs : 'none');

const ok = bar.cls.includes('on') && /glioma|ورم|دبق/i.test(out) && errs.length === 0;
await browser.close();
console.log(ok ? '\nEND-TO-END PASS' : '\nEND-TO-END FAIL');
process.exit(ok ? 0 : 1);
