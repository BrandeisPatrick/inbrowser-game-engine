// The 1.0 measurement sweep (ENGINE.md "Measured (1.0)"): every sample page, every pose, at 3024x1964 and 2560x1440,
// under the engine defaults and under the "everything on" settings, run in the desktop app's browser pane through
// window.__sample.run(plan) (src/core/Bench.js). Load a page with ?bench=1&size=WxH (plus the page query listed per
// plan), then from the console (Vite serves the repository under /@fs/):
//   const s = await import('/@fs/<repo>/bench/release-sweep.js'); const r = await s.run('canyon');
// r.result is the run(plan) table (ms per frame, pixel diff against the first setting) and r.file the name the
// result was saved under in bench/results/ (the receiver that wrote them during the 1.0 pass is not part of the
// package: any `fetch` of r.file / r.result to a local endpoint, or copy(JSON.stringify(r.result)), works).
// Settings are paths the sample registered (see each sample's createBenchHarness call); run() restores them
// between settings, so every row starts from the state the page loaded with.
const FX_MAX = { 'postfx.gbuffer': true, 'postfx.aa': 'taa', 'postfx.ssr': true, 'postfx.volumetric': true, 'postfx.dof': true, 'postfx.motionBlur': 1, 'postfx.ao.normals': 'gbuffer' };
const GB = { 'postfx.gbuffer': true, 'postfx.aa': 'taa' };
const CANYON_POSES = ['straight', 'chase', 'aerial', 'ahead'];
export const plans = {
  // /canyon-run/?bench=1&size=WxH
  canyon: { poses: CANYON_POSES, settings: [
    { label: 'default', set: {} },
    { label: 'gbuffer+taa', set: GB },
    { label: 'max', set: { ...FX_MAX, 'shadows.kernel': 'pcss' } },
  ] },
  // /canyon-run/?bench=1&size=WxH&lamps=1&t=night&gpuparticles=1
  'canyon-night': { poses: ['night-lamps', 'devils', 'chase'], settings: [
    { label: 'default (lamps, gpu particles)', set: {} },
    { label: 'max', set: { ...FX_MAX, 'shadows.kernel': 'pcss' } },
  ] },
  // /city/?bench=1&size=WxH (traffic and streaming on, as shipped)
  city: { poses: ['street', 'crossing', 'high', 'shore'], settings: [
    { label: 'default', set: {} },
    { label: 'gbuffer+taa', set: GB },
    { label: 'max', set: { ...FX_MAX, 'water.refraction': true } },
    { label: 'night max clustered', set: { ...FX_MAX, 'water.refraction': true, 'time': 'night', 'lights.mode': 'clustered' } },
  ] },
  // /sandbox/?bench=1&size=WxH&scene=<stack|bench1000|cloth|rope|soft|buoyancy|destructible>
  sandbox: { poses: ['overview', 'close'], settings: [
    { label: 'default', set: {} },
    { label: 'max', set: FX_MAX },
  ] },
  // /character/?bench=1&size=WxH
  character: { poses: ['idle', 'run', 'aerial', 'push', 'ragdoll'], settings: [
    { label: 'default', set: {} },
    { label: 'gbuffer+taa', set: GB },
    { label: 'max', set: { ...FX_MAX, 'shadows.kernel': 'pcss' } },
  ] },
};
const pad = (n) => String(n).padStart(2, '0');
/** Run one plan on the current page; label = plan name (+ '-<scene>' for the sandbox). Resolves { file, result }. */
export async function run(name, { frames = 24, warmup = 3, label = name } = {}) {
  const S = window.__sample;
  if (!S || !S.ready) throw new Error('window.__sample is not ready');
  const plan = plans[name];
  if (!plan) throw new Error(`unknown plan '${name}' (${Object.keys(plans).join(', ')})`);
  const missing = plan.poses.filter((p) => !S.poses.includes(p));
  if (missing.length) throw new Error(`poses missing on this page: ${missing.join(', ')}`);
  const t0 = performance.now();
  const result = await S.run({ poses: plan.poses, settings: plan.settings, frames, warmup });
  const d = new Date();
  const size = `${result.size.w}x${result.size.h}`;
  const file = `${d.toISOString().slice(0, 10)}-${pad(d.getHours())}${pad(d.getMinutes())}-${S.name}-${size}-p6-${label}.json`;
  const ua = navigator.userAgentData;
  const header = {
    page: S.name, label: `p6-${label}`, date: d.toISOString(), url: location.href,
    browser: 'claude-desktop-app in-app Chromium', chrome: ua && ua.brands ? (ua.brands.find((b) => /Chromium/.test(b.brand)) || {}).version : undefined,
    headless: false, platform: `${navigator.platform} ${ua ? ua.platform : ''}`.trim(), renderer: result.renderer,
    size: result.size, frames: result.frames, warmup: result.warmup, runMs: Math.round(performance.now() - t0),
    poses: plan.poses, settings: plan.settings, rows: result.rows,
  };
  return { file, result: header };
}
/** One line per row for a console / log brief. */
export function brief(r) {
  return r.result.rows.map((x) => `${x.pose.padEnd(12)} ${x.setting.padEnd(28)} ${x.ms.toFixed(2).padStart(6)} ms  diff ${x.meanAbs == null ? '  ref ' : x.meanAbs.toFixed(3)}  draws ${x.drawCalls}`).join('\n');
}
