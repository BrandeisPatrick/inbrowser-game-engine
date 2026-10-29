// GPU benchmark in headed Google Chrome (ANGLE on Metal on macOS, vsync on): starts the Vite dev server in-process,
// opens /<page>/?bench=1&size=WxH in a window sized to the canvas plus browser chrome (viewport emulated at
// deviceScaleFactor 1, so the drawing buffer is exactly W x H), waits for window.__sample.ready, runs
// __sample.run(plan) (every setting on every pose: GPU-synchronised ms per frame and the pixel diff against the first
// setting), prints the table and writes bench/results/<YYYY-MM-DD>-<HHMM>-<page>-<WxH>-<label>.json. The browser stays
// open only for the run. Every page call runs under a watchdog (60 s + 250 ms per frame of the plan); a page that reloads
// after it was ready is waited out and the run retried once. Exit code 1 on errors (console errors, no harness, unknown
// pose / setting, watchdog).
//
//   node test/browser/bench.mjs --page=canyon-run --size=3024x1964 --poses=straight,chase --frames=24 \
//     --label=baseline --settings='[{"label":"a","set":{"postfx.aa":"fxaa"}}]'
// Options: --page (canyon-run), --size (3024x1964), --poses (all), --frames (24), --warmup (3), --label (baseline),
// --settings (JSON array of { label, set: { path: value } }; default: the current settings), --out (bench/results),
// --query (extra URL query, e.g. notraffic&nostream=1), --headless (debug only: no real GPU timing on CI),
// --config (another vite config).
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {
  benchSize, repoRoot, headedArgs, headlessArgs, parseArgs, parseSize, startVite, resolvePage, launchChrome,
  watchErrors, loadSample, evalStep, withTimeout, onSignals, formatTable, stamp,
} from './playwright.config.mjs';

const args = parseArgs();
const pageId = typeof args.page === 'string' ? args.page : 'canyon-run';
const size = parseSize(args.size || benchSize);
const label = typeof args.label === 'string' && args.label ? args.label : 'baseline';
const headless = !!args.headless;
const query = typeof args.query === 'string' && args.query ? `&${args.query.replace(/^[?&]+/, '')}` : '';
const intArg = (name, def, min) => {
  if (args[name] === undefined) return def;
  const v = Number(args[name]);
  if (!Number.isFinite(v) || v < min) { console.error(`[bench] --${name} must be a number >= ${min} (got '${args[name]}')`); process.exit(1); }
  return Math.floor(v);
};
const plan = {
  poses: typeof args.poses === 'string' && args.poses ? args.poses.split(',').map((s) => s.trim()).filter(Boolean) : [],
  frames: intArg('frames', 24, 1),
  warmup: intArg('warmup', 3, 0),
  settings: [],
};
if (typeof args.settings === 'string' && args.settings.trim()) {
  try { plan.settings = JSON.parse(args.settings); } catch (e) { console.error(`[bench] --settings is not JSON: ${e.message}`); process.exit(1); }
  if (!Array.isArray(plan.settings)) plan.settings = [plan.settings];
}
const outDir = path.resolve(repoRoot, typeof args.out === 'string' ? args.out : 'bench/results');
const safe = (s) => String(s).replace(/[^\w.-]+/g, '_');

let vite = null, browser = null, code = 0;
const cleanup = async () => {
  if (browser) { const b = browser; browser = null; await withTimeout(b.close(), 15000, 'browser close').catch(() => {}); }
  if (vite) { const v = vite; vite = null; await v.close().catch(() => {}); }
};
onSignals(cleanup);

try {
  vite = await startVite({ configFile: typeof args.config === 'string' ? args.config : undefined });
  const pagePath = await resolvePage(vite, pageId);
  if (!pagePath) throw new Error(`no page '${pageId}' on this server (root ${vite.server.config.root})`);
  const url = `${vite.base}${pagePath}?bench=1&size=${size.w}x${size.h}${query}`;
  const chromeH = 120; // tab strip, toolbar and the automation info bar
  browser = await launchChrome({ headless, args: [...(headless ? headlessArgs : headedArgs), `--window-size=${size.w},${size.h + chromeH}`] });
  const context = await browser.newContext({ viewport: { width: size.w, height: size.h }, deviceScaleFactor: 1 });
  const page = await context.newPage();
  const errors = watchErrors(page);
  console.log(`[bench] ${url}  (Chrome ${browser.version()}, ${headless ? 'headless' : 'headed'})`);
  const r = await loadSample(page, url, errors);
  if (r.status !== 'ready') throw new Error(`${pageId}: ${r.status === 'skipped' ? 'the page has no bench harness (window.__sample) yet' : r.note}${errors.length ? `\n  ${errors.join('\n  ')}` : ''}`);
  await page.bringToFront();
  const nPoses = plan.poses.length || await evalStep(page, errors, r, () => window.__sample.poses.length, undefined, { ms: 60000, what: 'poses' });
  const nSettings = plan.settings.length || 1;
  // watchdog: 60 s + 250 ms per rendered frame (warm, settle, capture, warmup and timed frames of every row)
  const runMs = 60000 + nPoses * nSettings * (plan.frames + plan.warmup + 8) * 250;
  console.log(`[bench] ${pageId} ready in ${(r.loadMs / 1000).toFixed(1)} s; running ${plan.poses.length ? plan.poses.join(', ') : `all ${nPoses} poses`} x ${nSettings} setting(s), ${plan.frames} frames (watchdog ${Math.round(runMs / 1000)} s)`);
  const res = await evalStep(page, errors, r, (p) => window.__sample.run(p), plan, { ms: runMs, what: 'run' });
  const result = {
    page: pageId, label, date: new Date().toISOString(), url, chrome: browser.version(), headless,
    platform: `${process.platform} ${os.arch()}`, cpu: (os.cpus()[0] || {}).model || '',
    renderer: res.renderer, size: res.size, frames: res.frames, warmup: res.warmup, runMs: res.ms, reloads: r.reloads,
    poses: [...new Set(res.rows.map((x) => x.pose))], settings: plan.settings, errors: errors.slice(), rows: res.rows,
  };
  fs.mkdirSync(outDir, { recursive: true });
  const file = path.join(outDir, `${stamp()}-${safe(pageId)}-${size.w}x${size.h}-${safe(label)}.json`);
  fs.writeFileSync(file, `${JSON.stringify(result, null, 2)}\n`);

  console.log(`\n${result.renderer} · ${res.size.w}x${res.size.h} @${res.size.pixelRatio} · ${res.frames} frames · ${res.ms} ms\n`);
  console.log(formatTable(res.rows, ['setting', 'pose', 'ms', 'fps', 'meanAbs', 'p99', 'max', 'pctOver2', 'pctOver8', 'drawCalls', 'triangles', 'hash']));
  // per setting: mean ms over the poses and the change against the first setting
  const bySet = new Map();
  for (const row of res.rows) { const a = bySet.get(row.setting) || []; a.push(row.ms); bySet.set(row.setting, a); }
  let ref = 0;
  const summary = [...bySet].map(([setting, ms], i) => {
    const mean = ms.reduce((a, b) => a + b, 0) / ms.length;
    if (i === 0) ref = mean;
    return { setting, 'mean ms': +mean.toFixed(3), 'vs first': i === 0 ? '-' : `${mean >= ref ? '+' : ''}${(mean - ref).toFixed(3)} ms (${(100 * (mean - ref) / ref).toFixed(1)}%)` };
  });
  const rel = path.relative(repoRoot, file);
  console.log(`\n${formatTable(summary)}\n\n[bench] wrote ${rel.startsWith('..') ? file : rel}`);
  if (errors.length) { code = 1; console.log(`\n${errors.length} error(s) during the run:\n  ${errors.join('\n  ')}`); }
} catch (e) {
  code = 1;
  console.error(`[bench] ${e.message}`);
} finally {
  await cleanup();
}
process.exit(code);
