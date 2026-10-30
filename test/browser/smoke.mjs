// Browser smoke test: every sample page boots in headless Chrome without console errors, and every bench pose renders
// a deterministic frame. Starts the Vite dev server in-process (free port from playwright.config.mjs), opens
// /<page>/?bench=1&size=1280x720&nofx=0 (plus PAGE_QUERY), waits for window.__sample.ready (90 s), then per pose:
// pose(id), settle(), two capture(4) calls (separate tasks) must hash equal (one retry: the first frame after a pose
// may still adapt), and info() is recorded. Every page step runs under a 90 s watchdog; a page that reloads after it
// was ready is waited out and the pose redone once. A page without a harness yet (it loads and sets a legacy global
// such as window.__canyon) or without a page on this server is SKIPPED, not failed. On macOS a SwiftShader renderer
// (no GPU) fails. Exit code 1 on any failure. `city-clustered` is the city page again, at night with every lamp in
// range lit through ClusteredLights (one pose, plus in-page checks: cluster lights in view, no failed program, no
// black frame), so a GLSL error in the clustered chunks shows up here.
//
//   node test/browser/smoke.mjs                        # canyon-run, city, character, sandbox, city-clustered, canyon-night, city-water,
//                                                      # sandbox-cloth / rope / soft / buoyancy / destructible
//   node test/browser/smoke.mjs city --headed --size=1600x900 [--config=other.vite.config.js]
import {
  defaultSize, repoRoot, legacyGlobals, readyTimeoutMs, parseArgs, parseSize, startVite, resolvePage, launchChrome,
  watchErrors, loadSample, evalStep, withTimeout, onSignals, formatTable,
} from './playwright.config.mjs';

// extra query per page: the city without traffic or tile streaming (async tile builds could land between captures)
const PAGE_QUERY = {
  city: 'notraffic&nostream=1', sandbox: 'scene=stack', 'city-clustered': 'notraffic&nostream=1&clustered&t=night',
  // 0.8.0: the canyon at night with the clustered lamp field and the GPU dust devils; the city's water refraction (needs a
  // single-sampled scene target: the G-buffer path, with FXAA rather than TAA, whose captures are never hash-stable) at the shore
  'canyon-night': 'lamps=1&gpuparticles=1&t=night', 'city-water': 'notraffic&nostream=1&water=refraction&gbuffer=1&aa=fxaa',
  // 0.9.0: the sandbox's particle / soft-body / buoyancy / fracture scenes (in-thread; two poses each)
  'sandbox-cloth': 'scene=cloth', 'sandbox-rope': 'scene=rope', 'sandbox-soft': 'scene=soft', 'sandbox-buoyancy': 'scene=buoyancy', 'sandbox-destructible': 'scene=destructible',
};
// pages that are another page with a different query, and the poses they run (default: all)
const PAGE_PATH = {
  'city-clustered': 'city', 'canyon-night': 'canyon-run', 'city-water': 'city',
  'sandbox-cloth': 'sandbox', 'sandbox-rope': 'sandbox', 'sandbox-soft': 'sandbox', 'sandbox-buoyancy': 'sandbox', 'sandbox-destructible': 'sandbox',
};
const PAGE_POSES = { 'city-clustered': ['street'], 'canyon-night': ['night-lamps', 'devils'], 'city-water': ['shore'] };
// in-page checks after each pose: a failure message or ''
const PAGE_CHECK = {
  'city-clustered': () => {
    const C = window.__city, w = C && C.world, cl = w && w.clustered;
    if (!cl) return 'world.clustered missing (fell back to the PointLight pool)';
    if (w.props.lightPool.mode !== 'clustered') return `lamp mode ${w.props.lightPool.mode}`;
    if (!(w.atmo.darkness > 0.3)) return `not night (darkness ${w.atmo.darkness})`;
    if (!(cl.count > 0 && cl.stats.lights > 0)) return `no cluster lights (count ${cl.count}, in view ${cl.stats.lights})`;
    const bad = C.engine.renderer.info.programs.filter((p) => p.diagnostics && !p.diagnostics.runnable).length;
    if (bad) return `${bad} program(s) failed to compile`;
    const cap = window.__sample.capture(4);
    let sum = 0;
    for (let i = 0; i < cap.data.length; i += 4) sum += Math.max(cap.data[i], cap.data[i + 1], cap.data[i + 2]);
    const mean = sum / (cap.data.length / 4) / (cap.scale * cap.scale);
    return mean < 1 ? `black frame (mean brightest channel ${mean.toFixed(2)} / 255)` : '';
  },
  'canyon-night': () => {
    const C = window.__canyon;
    if (!C || !C.lamps) return 'window.__canyon.lamps missing (lamp field not built)';
    if (!(C.atmo.darkness > 0.3)) return `not night (darkness ${C.atmo.darkness})`;
    if (!(C.lamps.clustered && C.lamps.clustered.count > 0)) return 'no cluster lights';
    if (!(C.devils && C.devils.gpu)) return 'devils are not on the GPU particle system';
    const bad = C.engine.renderer.info.programs.filter((p) => p.diagnostics && !p.diagnostics.runnable).length;
    return bad ? `${bad} program(s) failed to compile` : '';
  },
  'city-water': () => {
    const C = window.__city, w = C && C.world;
    if (!w || !w.sceneCopy) return 'world.sceneCopy missing (water refraction not built)';
    if (!w.sceneCopy.valid) return 'scene copy invalid (multisampled or depth-less scene target)';
    const bad = C.engine.renderer.info.programs.filter((p) => p.diagnostics && !p.diagnostics.runnable).length;
    return bad ? `${bad} program(s) failed to compile` : '';
  },
};
const STEP_MS = 90000;

const args = parseArgs();
const pages = args._.length ? args._ : args.pages ? String(args.pages).split(',') : ['canyon-run', 'city', 'character', 'sandbox', 'city-clustered', 'canyon-night', 'city-water', 'sandbox-cloth', 'sandbox-rope', 'sandbox-soft', 'sandbox-buoyancy', 'sandbox-destructible'];
const size = parseSize(args.size || defaultSize);
const headless = !args.headed;
const mac = process.platform === 'darwin';

const pageRows = [], poseRows = [], failures = [];
let vite = null, browser = null;
const cleanup = async () => {
  if (browser) { const b = browser; browser = null; await withTimeout(b.close(), 15000, 'browser close').catch(() => {}); }
  if (vite) { const v = vite; vite = null; await v.close().catch(() => {}); }
};
onSignals(cleanup);

const fmtSize = (s) => `${s.w}x${s.h}@${s.pixelRatio}`;
const checkRenderer = (id, renderer) => {
  if (mac && /SwiftShader/i.test(renderer || '')) failures.push(`${id}: renderer is SwiftShader (no GPU): ${renderer}`);
};

try {
  vite = await startVite({ configFile: typeof args.config === 'string' ? args.config : undefined });
  console.log(`[smoke] vite ${vite.base} (root ${vite.server.config.root.replace(repoRoot, '')}), Chrome ${headless ? 'headless' : 'headed'}, ${size.w}x${size.h}`);
  browser = await launchChrome({ headless });
  const context = await browser.newContext({ viewport: { width: size.w, height: size.h }, deviceScaleFactor: 1 });

  for (const id of pages) {
    const path = await resolvePage(vite, PAGE_PATH[id] || id);
    if (!path) { pageRows.push({ page: id, path: '-', status: 'SKIPPED', load_s: null, poses: 0, renderer: '', note: 'no such page on this server' }); continue; }
    const page = await context.newPage();
    const errors = watchErrors(page);
    const extra = PAGE_QUERY[id] ? `&${PAGE_QUERY[id]}` : '';
    const url = `${vite.base}${path}?bench=1&size=${size.w}x${size.h}&nofx=0${extra}`;
    const s = await loadSample(page, url, errors, { timeout: readyTimeoutMs });
    const row = { page: id, path: `${path}${extra ? ` (+${PAGE_QUERY[id]})` : ''}`, status: s.status.toUpperCase(), load_s: +(s.loadMs / 1000).toFixed(1), poses: 0, renderer: '', note: s.note };
    pageRows.push(row);
    const step = (fn, arg, what) => evalStep(page, errors, s, fn, arg, { ms: STEP_MS, what: `${id} ${what}` });
    try {
      if (s.status === 'ready') {
        const ids = (await step(() => window.__sample.poses.slice(), undefined, 'poses')).filter((p) => !PAGE_POSES[id] || PAGE_POSES[id].includes(p));
        row.poses = ids.length;
        if (!ids.length) {
          const info = await step(() => window.__sample.info(), undefined, 'info');
          row.renderer = info.renderer;
          poseRows.push({ page: id, pose: '(none)', hash: '-', stable: '-', draws: info.drawCalls, tris: info.triangles, size: fmtSize(info.size) });
        }
        for (const pose of ids) {
          const pr = { page: id, pose, hash: '', stable: '', draws: null, tris: null, size: '' };
          poseRows.push(pr);
          try {
            for (let attempt = 0; ; attempt++) {
              const reloads = s.reloads;
              await step((p) => { window.__sample.pose(p); }, pose, `pose ${pose}`);
              await step(() => (window.__sample.settle ? window.__sample.settle().then(() => true) : true), undefined, `settle ${pose}`);
              const cap = () => step(() => window.__sample.capture(4).hash, undefined, `capture ${pose}`);
              const a = await cap(), b = await cap();
              let ok = a === b, stable = ok ? 'yes' : '', hash = b;
              if (!ok) { const c = await cap(); ok = b === c; stable = ok ? 'retry' : `NO ${a}/${b}/${c}`; hash = c; }
              const info = await step(() => window.__sample.info(), undefined, `info ${pose}`);
              if (s.reloads !== reloads && attempt === 0) continue; // the page reloaded mid-pose: redo it on the new page
              Object.assign(pr, { hash, stable: s.reloads !== reloads ? `${stable} (reloaded)` : stable, draws: info.drawCalls, tris: info.triangles, size: fmtSize(info.size) });
              row.renderer = info.renderer;
              if (!ok) failures.push(`${id}/${pose}: captures differ (${stable})`);
              if (PAGE_CHECK[id]) { const msg = await step(PAGE_CHECK[id], undefined, `check ${pose}`); if (msg) failures.push(`${id}/${pose}: ${msg}`); }
              break;
            }
          } catch (e) {
            pr.stable = 'ERROR';
            failures.push(`${id}/${pose}: ${e.message.split('\n')[0]}`);
            if (/timed out/.test(e.message)) break; // a hung page: skip its remaining poses
          }
        }
        if (s.reloads) row.note = `${row.note ? `${row.note}; ` : ''}reloaded ${s.reloads}x after ready`;
      } else if (s.status === 'skipped') {
        // no harness yet: still report what the page rendered with
        const g = await withTimeout(page.evaluate((names) => {
          for (const n of names) {
            const e = window[n] && window[n].engine;
            if (e && e.renderer) { const gl = e.renderer.getContext(); return { renderer: e.profiler ? e.profiler.gpuRenderer : '', w: gl.drawingBufferWidth, h: gl.drawingBufferHeight }; }
          }
          return null;
        }, legacyGlobals), STEP_MS, `${id} legacy info`).catch(() => null);
        if (g) { row.renderer = g.renderer; row.note += `; canvas ${g.w}x${g.h}`; }
      }
      checkRenderer(id, row.renderer);
    } catch (e) {
      failures.push(`${id}: ${e.message.split('\n')[0]}`);
    }
    if (errors.length) failures.push(...errors.map((e) => `${id}: ${e}`));
    if (s.status === 'failed' && !errors.length) failures.push(`${id}: ${s.note}`);
    if (failures.some((f) => f.startsWith(`${id}:`) || f.startsWith(`${id}/`))) row.status = 'FAILED';
    await withTimeout(page.close({ runBeforeUnload: false }), 15000, `${id} close`).catch(() => {});
  }
} catch (e) {
  failures.push(`smoke: ${e.stack || e.message}`);
} finally {
  await cleanup();
}

console.log(`\n${formatTable(pageRows, ['page', 'path', 'status', 'load_s', 'poses', 'renderer', 'note'])}`);
if (poseRows.length) console.log(`\n${formatTable(poseRows, ['page', 'pose', 'hash', 'stable', 'draws', 'tris', 'size'])}`);
if (failures.length) {
  console.log(`\n${failures.length} failure(s):`);
  for (const f of failures) console.log(`  - ${f}`);
}
const failed = failures.length > 0;
console.log(`\n[smoke] ${failed ? 'FAILED' : 'OK'} (${pageRows.filter((r) => r.status === 'READY').length} ready, ${pageRows.filter((r) => r.status === 'SKIPPED').length} skipped)`);
process.exit(failed ? 1 : 0);
