// Procedural city harness (dev only, http://localhost:5190/city/): MapWorld with its built-in procedural city (no map
// data needed), the engine's post stack and a free-fly camera, so the world layer can be seen and measured in a
// browser without the game repository. Query: ?t=night|hours, ?traffic=N | ?notraffic, ?nostream=1, ?nofx=1,
// ?profile=1 (profiler table after N s), ?water=refraction (0.8.0: the bay sees the scene behind its surface; needs
// &gbuffer=1 or &aa=fxaa, pose `shore`), ?ssr=1 / ?volumetric=1 / ?motionblur=<s> / ?dof=1 / ?lut=warm / ?aonormals=gbuffer
// (0.8.0 post passes, samples/shared/fx-flags.js), F3 = profiler overlay. window.__city exposes everything for the console.
import * as THREE from 'three';
import { Engine, PostFX, FreeFlyRig, InputMap, LoadingScreen, createBenchHarness, readBenchParams } from '@vibe/engine';
import { MapWorld } from '@vibe/engine/world';
import { readFxFlags, aoFlags, loadLut, fxSettings } from '../shared/fx-flags.js';

const params = new URLSearchParams(location.search);
const NOFX = params.has('nofx') && params.get('nofx') !== '0';
const t = params.get('t');

const engine = new Engine({
  container: document.getElementById('app'), fixedHz: 120, maxSubsteps: 4, far: 6000, fov: 60, renderer: { antialias: NOFX },
  profile: params.get('profile') ? +params.get('profile') : false,
});
const { scene, camera, renderer } = engine;
const loading = new LoadingScreen({ title: 'PROCEDURAL CITY', label: 'Building the city…' });
const world = new MapWorld({
  scene, renderer, forceProcedural: true,
  timeOfDay: t == null ? 'golden' : Number.isFinite(+t) ? +t : t,
  traffic: params.has('notraffic') ? false : { count: +(params.get('traffic') || 150) },
  streaming: params.has('nostream') ? false : undefined,
  physics: true,
  // 0.7.0 knobs: ?clustered[&range=220] (every lamp in range lights the scene through ClusteredLights), ?cascades=2|3|4,
  // ?shadows=pcf|fast|pcss; the default path is untouched when the params are absent
  lights: params.has('clustered') ? { clustered: true, clusteredRange: +params.get('range') || 220 } : undefined,
  // 0.8.0: ?water=refraction (bay + ponds see the opaque scene behind their surface through a half-res SceneCopy)
  ...(params.get('water') === 'refraction' ? { water: { refraction: true } } : {}),
  ...(params.has('cascades') || params.has('shadows') ? { atmosphere: { shadows: { ...(params.has('cascades') ? { cascades: +params.get('cascades') || 2 } : {}), ...(params.has('shadows') ? { kernel: params.get('shadows') } : {}) } } } : {}),
});
engine.addSystem(world.system({ camera }), 'world');
const t0 = performance.now();
await engine.load((p, label) => loading.setProgress(p, label));
const loadMs = Math.round(performance.now() - t0);

let postfx = null;
if (!NOFX) {
  try {
    const { width, height } = engine.size;
    // 0.8.0 knobs (samples/shared/fx-flags.js): ?ssr=1, ?volumetric=1, ?motionblur=<s>, ?dof=1, ?lut=<name>, ?aonormals=gbuffer
    const fx = readFxFlags(params), lut = await loadLut(params);
    postfx = new PostFX(renderer, scene, camera, { atmosphere: world.atmo, width, height, pixelRatio: engine.pixelRatio, gbuffer: params.get('gbuffer') === '1', aa: params.get('aa') || 'auto', ao: aoFlags(params), ...fx, lut });
  } catch (e) { console.warn('[city] post-processing unavailable', e); world.atmo.setFogMode('scene'); }
}
engine.renderFn = (sc, cam, dt) => { if (postfx) postfx.render(dt); else { renderer.render(sc, cam); world.atmo.renderFlare(renderer, cam); } };
engine.onResize = (w, h, pr) => { if (postfx) postfx.setSize(w, h, pr); world.setSize(w, h, pr); };

// free-fly camera from the spawn, looking along the spawn heading (heading 0 = -Z, positive = left)
const sp = world.getSpawn();
const place = () => {
  camera.position.set(sp.position.x, sp.position.y + 2.2, sp.position.z);
  camera.lookAt(sp.position.x - Math.sin(sp.headingRad) * 60, sp.position.y + 2, sp.position.z - Math.cos(sp.headingRad) * 60);
  camera.updateMatrixWorld();
};
place();
const input = new InputMap({ pointerTarget: renderer.domElement });
const rig = new FreeFlyRig(camera, renderer.domElement, { input, speed: 20, collide: false, minY: 1.5, physics: world.physics });
engine.addSystem({ earlyUpdate: (dt) => input.update(dt), update: (dt) => rig.update(dt) }, 'camera');

// measurement harness (window.__sample; `?bench=1&size=WxH`): poses from the spawn along its heading; the world is
// updated past its throttles (culling, LOD partitions, light selection) and the time-dependent inputs are pinned
// (clouds, water, wind) so captures reproduce across page loads. Use ?notraffic&nostream=1 for exact comparisons.
const S = sp.position, sx = -Math.sin(sp.headingRad), sz = -Math.cos(sp.headingRad), MSAA0 = postfx ? postfx.msaaMaxPixels : Infinity;
// `shore` (water refraction): over the bay 8 m off the seawall, in the gap south of the pier nearest the spawn's z,
// 4.5 m above the water, looking north-north-east along the waterfront at that pier (its pilings, fender piles and
// warehouse 55 m ahead, the boulevard and the city on the left, open bay on the right), water in the lower half of the
// frame. From the map alone (pier rectangles + isLand), so it never depends on traffic; null without piers.
const SHORE = (() => {
  let best = null, bd = Infinity;
  for (const pr of (world.data && world.data.piers) || []) {
    if (!pr || !Array.isArray(pr.p) || pr.p.length < 3) continue;
    let x0 = Infinity, z0 = Infinity, z1 = -Infinity;
    for (const q of pr.p) { x0 = Math.min(x0, q[0]); z0 = Math.min(z0, q[1]); z1 = Math.max(z1, q[1]); }
    const d = Math.abs((z0 + z1) / 2 - S.z);
    if (d < bd) { bd = d; best = { x0, z1 }; }
  }
  if (!best) return null;
  const z = best.z1 + 55, wl = world.waterLevel;
  let x = best.x0 - 40; // on land west of the pier root, then east to the seawall
  for (let i = 0; i < 800 && world.isLand(x, z); i++) x += 0.5;
  return world.isLand(x, z) ? null : { x: x + 8, y: wl + 4.5, z, tx: x + 45, ty: wl + 0.5, tz: z - 115 };
})();
function copyKnob(apply) { // harness setters on world.sceneCopy (see 'water.copyScale')
  return (v) => {
    if (world.sceneCopy) { apply(world.sceneCopy, v); return; }
    console.warn('[city] water.copyScale / water.copyClear need world.sceneCopy: load with ?water=refraction or set water.refraction first');
  };
}
const sample = createBenchHarness(engine, {
  name: 'city',
  poses: {
    street: (cam) => { cam.position.set(S.x, S.y + 2.2, S.z); cam.up.set(0, 1, 0); cam.lookAt(S.x + sx * 60, S.y + 2, S.z + sz * 60); },
    crossing: (cam) => { cam.position.set(S.x + sx * 80, S.y + 2.2, S.z + sz * 80); cam.up.set(0, 1, 0); cam.lookAt(S.x + sx * 140, S.y + 2, S.z + sz * 140); },
    high: (cam) => { cam.position.set(S.x, S.y + 60, S.z); cam.up.set(0, 1, 0); cam.lookAt(S.x + sx * 100, S.y, S.z + sz * 100); },
    shore: (cam) => {
      if (!SHORE) { cam.position.set(S.x, S.y + 2.2, S.z); cam.up.set(0, 1, 0); cam.lookAt(S.x + sx * 60, S.y + 2, S.z + sz * 60); return; }
      cam.position.set(SHORE.x, SHORE.y, SHORE.z); cam.up.set(0, 1, 0); cam.lookAt(SHORE.tx, SHORE.ty, SHORE.tz);
    },
  },
  after: (cam) => {
    for (let i = 0; i < 3; i++) world.update(1, cam.position);
    const su = world.atmo.sky.material.uniforms; su.uTime.value = 0; su.uCloudOffset.value.set(0.13, 0.41);
    world.time = 0; if (world.mat && world.mat.water && world.mat.water.userData.u) world.mat.water.userData.u.uTime.value = 0;
  },
  quiet: () => {
    rig.deactivate(); if (postfx) { postfx.autoExposure = false; postfx.setTimeOfDay(world.atmo.darkness); }
    if (postfx && postfx.history) { postfx.history.reset(); for (let i = 0; i < 32; i++) postfx.render(0); } // TAA: converge from a clean history
  },
  settings: { // get + set so __sample.run() restores every knob between settings (not cumulative)
    'postfx.enabled': { get: () => !!(postfx && postfx.enabled), set: (v) => { if (postfx) postfx.setEnabled(!!v); } },
    'postfx.aa': { get: () => (postfx ? postfx.aa : null), set: (v) => { if (postfx) postfx.set('aa', v); } },
    ...fxSettings(() => postfx), // 0.8.0: postfx.ssr / volumetric / motionBlur / dof / lut.strength / ao.normals (+ their options)
    'postfx.gbuffer': { get: () => !!(postfx && postfx.gbuffer), set: (v) => { if (postfx) postfx.set('gbuffer', !!v); } },
    'lights.mode': { get: () => (world.props && world.props.lightPool ? world.props.lightPool.mode : null), set: (v) => world.setLightMode(v) },
    'postfx.ao': { get: () => !!(postfx && postfx.aoPass.enabled), set: (v) => { if (postfx) postfx.aoPass.enabled = !!v; } },
    'shadows.kernel': { get: () => world.atmo.shadowKernel, set: (v) => world.atmo.setShadowKernel(v) },
    'time': { get: () => world.atmo.timeOfDay, set: (v) => { world.atmo.setTimeOfDay(v); if (postfx) postfx.setTimeOfDay(world.atmo.darkness); } },
    // 0.8.0: bay / pond refraction through world.sceneCopy (created on first use; needs gbuffer or aa 'fxaa').
    // copyScale / copyClear act on world.sceneCopy, which exists once refraction has been on: load the page with
    // ?water=refraction (or set 'water.refraction' true before them); until then they warn and do nothing
    'water.refraction': { get: () => world.waterRefraction, set: (v) => world.setWaterRefraction(!!v) },
    'water.copyScale': { get: () => (world.sceneCopy ? world.sceneCopy.scale : 0.5), set: copyKnob((c, v) => { const s = +v; if (s > 0 && s <= 1) c.setScale(s); else console.warn(`[city] water.copyScale: ${v} is not in (0, 1]`); }) },
    'water.copyClear': { get: () => (world.sceneCopy ? world.sceneCopy.clear : true), set: copyKnob((c, v) => { c.clear = !!v; }) },
  },
});
const bp = readBenchParams(location.search); if (bp.size) sample.resize(bp.size.w, bp.size.h);
sample.setReady();
renderer.compile(scene, camera);
window.__city = { THREE, engine, world, postfx, atmo: world.atmo, rig, input, loadMs, place, stats: world.stats };
console.info(`[city] ready in ${loadMs} ms`, world.stats);
engine.start();
if (bp.enabled) sample.freeze(); // ?bench=1: no live ticks before the first pose, so captures reproduce across loads
// ?water=refraction: once frames have rendered, say why the scene copy is skipped, if it is (the resolved state, not
// the query: aa 'auto' is single-sampled by night or above msaaMaxPixels, ?nofx=1 renders to the canvas)
if (world.sceneCopy) setTimeout(() => {
  const c = world.sceneCopy;
  const hint = c.reason === 'msaa' || c.reason === 'canvas' ? `: use &gbuffer=1 or &aa=fxaa${NOFX ? ' without ?nofx' : ''}` : '';
  if (c.reason !== null) console.warn(`[city] ?water=refraction: the scene copy is skipped (${c.reason}), the water stays opaque${hint}`);
}, 2000);
loading.hide(300);
