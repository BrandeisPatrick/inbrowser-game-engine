// Shared configuration of the browser scripts (test/browser/smoke.mjs, bench.mjs). Plain Playwright library use, no
// test runner: both launch the installed Google Chrome (channel 'chrome'; `playwright install` is never needed) and
// serve the samples from a Vite dev server started in-process on a free port (vitePort, or the next free one).
//
// Constants: channel, vitePort, defaultSize, benchSize, readyTimeoutMs, headlessArgs, headedArgs, legacyGlobals,
// ignoreConsole.
// Helpers: repoRoot, parseArgs(argv), parseSize('WxH'), startVite({ port, configFile }), resolvePage(vite, id),
// launchChrome({ headless, args }), watchErrors(page), loadSample(page, url, errors) -> session, waitReady,
// ensureReady, evalStep(page, errors, session, fn, arg, { ms, what }), withTimeout(promise, ms, what),
// onSignals(cleanup), threeAddonImports(), formatTable(rows, cols), today(), stamp().
import { fileURLToPath } from 'node:url';
import fs from 'node:fs';
import path from 'node:path';

/** Repository root (two levels above this file). */
export const repoRoot = fileURLToPath(new URL('../../', import.meta.url));

/** Playwright browser channel: the system Google Chrome. */
export const channel = 'chrome';
/** Preferred dev-server port (the interactive dev server uses 5190; the next free port is taken when busy). */
export const vitePort = 5199;
/** Canvas size of the smoke test (CSS px at pixel ratio 1). */
export const defaultSize = '1280x720';
/** Default canvas of bench.mjs: the 14-inch MacBook Pro pixel count, the engine's measurement protocol. */
export const benchSize = '3024x1964';
/** How long a page may take to report window.__sample.ready (the city builds a procedural city, Canyon Run bakes impostors). */
export const readyTimeoutMs = 90000;

const mac = process.platform === 'darwin';
// keep timers and rAF running when the window is covered or in the background (Playwright adds some of these too)
const unthrottled = ['--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows'];

/**
 * Headless Chrome: ANGLE on Metal on macOS (the real GPU); elsewhere SwiftShader, which recent Chrome only allows for
 * WebGL with --enable-unsafe-swiftshader (CI machines have no GPU).
 */
export const headlessArgs = [
  ...(mac ? ['--use-angle=metal'] : ['--use-angle=swiftshader', '--enable-unsafe-swiftshader']),
  '--ignore-gpu-blocklist', '--enable-precise-memory-info', ...unthrottled,
];

/** Headed Chrome for GPU numbers: ANGLE on Metal on macOS, vsync and the frame-rate limit left alone. */
export const headedArgs = [...(mac ? ['--use-angle=metal'] : []), '--ignore-gpu-blocklist', '--enable-precise-memory-info', ...unthrottled];

/** Globals the samples set before the bench harness existed: a page with one of these but no __sample is SKIPPED. */
export const legacyGlobals = ['__canyon', '__city', '__character', '__sandbox'];

/** Console errors to ignore, matched against the message text and its source URL. */
export const ignoreConsole = [/\/favicon\.ico\b/];

/**
 * `--key=value` / `--flag` arguments -> { key: value | true, _: [positional] }.
 * @param {string[]} [argv]
 */
export function parseArgs(argv = process.argv.slice(2)) {
  const out = { _: [] };
  for (const a of argv) {
    const m = /^--([^=]+)(?:=([\s\S]*))?$/.exec(a);
    if (m) out[m[1]] = m[2] === undefined ? true : m[2];
    else out._.push(a);
  }
  return out;
}

/** 'WxH' -> { w, h } (throws on anything else). */
export function parseSize(s) {
  const m = /^(\d+)\s*[x*,]\s*(\d+)$/i.exec(String(s || ''));
  if (!m || !(+m[1] > 0) || !(+m[2] > 0)) throw new Error(`bad size '${s}' (want WxH, e.g. ${defaultSize})`);
  return { w: +m[1], h: +m[2] };
}

/**
 * Plain-text table of `rows` (objects) over `cols` (keys; default: the keys of the first row).
 * @param {object[]} rows
 * @param {string[]} [cols]
 */
export function formatTable(rows, cols) {
  if (!rows.length) return '(no rows)';
  cols = cols || Object.keys(rows[0]);
  const cell = (v) => (v === null || v === undefined ? '-' : String(v));
  const wd = cols.map((c) => Math.max(c.length, ...rows.map((r) => cell(r[c]).length)));
  const line = (vals) => vals.map((v, i) => v.padEnd(wd[i])).join('  ').trimEnd();
  return [line(cols), line(wd.map((n) => '-'.repeat(n))), ...rows.map((r) => line(cols.map((c) => cell(r[c]))))].join('\n');
}

/** Local date YYYY-MM-DD. */
export function today() {
  const d = new Date(), p = (v) => String(v).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`;
}

/** Local date and time YYYY-MM-DD-HHMM (result file names). */
export function stamp() {
  const d = new Date(), p = (v) => String(v).padStart(2, '0');
  return `${today()}-${p(d.getHours())}${p(d.getMinutes())}`;
}

/**
 * Race `promise` against a watchdog: rejects with "<what> timed out after N s" if it has not settled within `ms`.
 * @template T
 * @param {Promise<T>} promise
 * @param {number} ms
 * @param {string} [what]
 * @returns {Promise<T>}
 */
export function withTimeout(promise, ms, what = 'step') {
  let timer = null;
  const dog = new Promise((_, reject) => { timer = setTimeout(() => reject(new Error(`${what} timed out after ${Math.round(ms / 1000)} s`)), ms); });
  return Promise.race([promise, dog]).finally(() => clearTimeout(timer));
}

/** Run `cleanup` (closes the browser and Vite) and exit on SIGINT (130) or SIGTERM (143). */
export function onSignals(cleanup) {
  for (const [sig, code] of [['SIGINT', 130], ['SIGTERM', 143]]) {
    process.once(sig, () => { Promise.resolve().then(cleanup).catch(() => {}).finally(() => process.exit(code)); });
  }
}

/**
 * Every `three/addons/...` module imported by a .js / .html file under src/ and samples/. startVite pre-bundles them:
 * a deep import that Vite first meets at runtime makes it re-optimise its dependencies and reload the page.
 * @returns {string[]}
 */
export function threeAddonImports(dirs = ['src', 'samples']) {
  const found = new Set(), re = /['"](three\/addons\/[^'"]+)['"]/g;
  const walk = (dir) => {
    for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
      if (e.name === 'node_modules' || e.name.startsWith('.')) continue;
      const p = path.join(dir, e.name);
      if (e.isDirectory()) walk(p);
      else if (/\.(m?js|html)$/.test(e.name)) for (const m of fs.readFileSync(p, 'utf8').matchAll(re)) found.add(m[1]);
    }
  };
  for (const d of dirs) { const p = path.join(repoRoot, d); if (fs.existsSync(p)) walk(p); }
  return [...found].sort();
}

/**
 * Start the repository's Vite dev server (vite.config.js, or `configFile`) in-process on 127.0.0.1, without file
 * watching or HMR, with three and every three/addons module the code imports pre-bundled (no late re-optimise reload).
 * @param {{ port?: number, configFile?: string }} [o]
 * @returns {Promise<{ server: import('vite').ViteDevServer, base: string, port: number, close: () => Promise<void> }>}
 */
export async function startVite({ port = vitePort, configFile } = {}) {
  const { createServer } = await import('vite');
  const server = await createServer({
    configFile: configFile ? path.resolve(configFile) : path.join(repoRoot, 'vite.config.js'),
    logLevel: 'warn',
    clearScreen: false,
    server: { host: '127.0.0.1', port, strictPort: false, open: false, hmr: false, watch: null },
    optimizeDeps: { include: ['three', ...threeAddonImports()] },
  });
  await server.listen();
  const addr = server.httpServer && server.httpServer.address();
  const actual = addr && typeof addr === 'object' ? addr.port : port;
  const base = `http://127.0.0.1:${actual}`;
  return { server, base, port: actual, close: () => server.close() };
}

async function fetchHtml(url) {
  try {
    const r = await withTimeout(fetch(url, { headers: { accept: 'text/html' } }), 30000, `GET ${url}`);
    return { status: r.status, html: r.ok ? await r.text() : '' };
  } catch (e) { return { status: 0, html: '' }; }
}

/**
 * URL path of sample page `id` on this server, or null when it has none. Tries `/<id>/` and `/<id>` (a page must
 * answer 200 with HTML that differs from the SPA fallback served for unknown paths), then `/` when the Vite root
 * itself is the sample's folder (the single-sample layout, where Canyon Run is the root page).
 * @param {{ server: import('vite').ViteDevServer, base: string }} vite
 * @param {string} id
 * @returns {Promise<string|null>}
 */
export async function resolvePage({ server, base }, id) {
  const fallback = await fetchHtml(`${base}/__smoke-missing-${Date.now()}__/`);
  for (const p of [`/${id}/`, `/${id}`]) {
    const r = await fetchHtml(base + p);
    if (r.status === 200 && r.html && !(fallback.status === 200 && r.html === fallback.html)) return p;
  }
  return path.basename(path.resolve(server.config.root)) === id && fs.existsSync(path.join(server.config.root, 'index.html')) ? '/' : null;
}

/**
 * Launch the installed Google Chrome.
 * @param {{ headless?: boolean, args?: string[] }} [o]
 */
export async function launchChrome({ headless = true, args } = {}) {
  const { chromium } = await import('playwright');
  return chromium.launch({ channel, headless, args: args || (headless ? headlessArgs : headedArgs) });
}

/**
 * Collect console.error messages and uncaught page errors of `page` into the returned array (strings).
 * @param {import('playwright').Page} page
 * @returns {string[]}
 */
export function watchErrors(page) {
  const errors = [];
  page.on('console', (m) => {
    if (m.type() !== 'error') return;
    const loc = m.location(), where = loc && loc.url ? `${loc.url}:${loc.lineNumber}` : '';
    if (ignoreConsole.some((re) => re.test(m.text()) || re.test(where))) return;
    errors.push(`console.error: ${m.text()}${where ? `  (${where})` : ''}`);
  });
  page.on('pageerror', (e) => errors.push(`pageerror: ${e && e.stack ? e.stack.split('\n').slice(0, 3).join(' | ') : e}`));
  page.on('crash', () => errors.push('page crashed'));
  return errors;
}

/**
 * Poll the open page until window.__sample.ready. Resolves { status, loadMs, note }: 'ready'; 'skipped' (one of
 * legacyGlobals is set and no harness appeared within a 3 s grace period); 'failed' (an error was collected in
 * `errors`, nothing was ready within `timeout`, or __sample exists but never became ready).
 * @param {import('playwright').Page} page
 * @param {string[]} errors  from watchErrors(page)
 * @param {{ timeout?: number, t0?: number }} [o]
 */
export async function waitReady(page, errors, { timeout = readyTimeoutMs, t0 = Date.now() } = {}) {
  let legacySince = 0;
  for (;;) {
    if (errors.length) return { status: 'failed', loadMs: Date.now() - t0, note: `${errors.length} error(s) while loading` };
    let st = null;
    try {
      st = await withTimeout(page.evaluate((names) => ({
        sample: !!window.__sample, ready: !!(window.__sample && window.__sample.ready),
        legacy: names.filter((n) => window[n] !== undefined),
      }), legacyGlobals), 10000, 'ready poll');
    } catch { /* navigation in progress (a reload), or a busy main thread: poll again */ }
    const t = Date.now() - t0;
    if (st && st.ready) return { status: 'ready', loadMs: t, note: '' };
    if (st && !st.sample && st.legacy.length) {
      if (!legacySince) legacySince = Date.now();
      else if (Date.now() - legacySince > 3000) return { status: 'skipped', loadMs: t, note: `no window.__sample (loaded: ${st.legacy.join(', ')})` };
    }
    if (t > timeout) return { status: 'failed', loadMs: t, note: st && st.sample ? '__sample never became ready' : `nothing ready after ${Math.round(timeout / 1000)} s` };
    await new Promise((r) => setTimeout(r, 250));
  }
}

/**
 * Open a sample and wait until window.__sample.ready (waitReady). Resolves a session { status, loadMs, note, loads,
 * readyLoads, reloads } for ensureReady / evalStep: `loads` counts the page's load events, so a reload after ready
 * (e.g. Vite re-optimising) is noticed and waited out.
 * @param {import('playwright').Page} page
 * @param {string} url
 * @param {string[]} errors  from watchErrors(page)
 * @param {{ timeout?: number }} [o]
 */
export async function loadSample(page, url, errors, { timeout = readyTimeoutMs } = {}) {
  const t0 = Date.now();
  const session = { status: 'failed', loadMs: 0, note: '', loads: 0, readyLoads: -1, reloads: 0 };
  page.on('load', () => { session.loads++; });
  try {
    await page.goto(url, { waitUntil: 'load', timeout });
  } catch (e) {
    return Object.assign(session, { loadMs: Date.now() - t0, note: `goto: ${e.message.split('\n')[0]}` });
  }
  Object.assign(session, await waitReady(page, errors, { timeout, t0 }));
  if (session.status === 'ready') session.readyLoads = session.loads;
  return session;
}

/**
 * When the page has loaded again since it was ready (or `force`), wait for window.__sample.ready once more. Resolves
 * true after such a re-wait; throws when the page does not come back ready.
 * @param {import('playwright').Page} page
 * @param {string[]} errors
 * @param {{ loads: number, readyLoads: number, reloads: number }} session  from loadSample
 * @param {boolean} [force]
 */
export async function ensureReady(page, errors, session, force = false) {
  if (!force && session.loads === session.readyLoads) return false;
  await withTimeout(page.waitForLoadState('load'), readyTimeoutMs, 'reload').catch(() => {});
  const r = await waitReady(page, errors);
  if (r.status !== 'ready') throw new Error(`the page reloaded after it was ready and did not come back: ${r.note}`);
  session.readyLoads = session.loads; session.reloads++;
  return true;
}

/**
 * page.evaluate(fn, arg) under a watchdog of `ms`, after ensureReady(). A call whose execution context is destroyed
 * (the page navigated or reloaded meanwhile) re-waits for ready and is retried once.
 * @param {import('playwright').Page} page
 * @param {string[]} errors
 * @param {object} session  from loadSample
 * @param {Function} fn
 * @param {*} [arg]
 * @param {{ ms?: number, what?: string }} [o]
 */
export async function evalStep(page, errors, session, fn, arg, { ms = readyTimeoutMs, what = 'evaluate' } = {}) {
  for (let attempt = 0; ; attempt++) {
    await ensureReady(page, errors, session);
    try {
      return await withTimeout(page.evaluate(fn, arg), ms, what);
    } catch (e) {
      if (attempt > 0 || !/context was destroyed|Cannot find context|navigat/i.test(e.message)) throw e;
      await ensureReady(page, errors, session, true);
    }
  }
}

export default { channel, vitePort, defaultSize, benchSize, readyTimeoutMs, headlessArgs, headedArgs, legacyGlobals, ignoreConsole };
