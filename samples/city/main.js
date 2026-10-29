// Procedural city harness (dev only, http://localhost:5190/city/): MapWorld with its built-in procedural city (no map
// data needed), the engine's post stack and a free-fly camera, so the world layer can be seen and measured in a
// browser without the game repository. Query: ?t=night|hours, ?traffic=N | ?notraffic, ?nostream=1, ?nofx=1,
// ?profile=1 (profiler table after N s), F3 = profiler overlay. window.__city exposes everything for the console.
import * as THREE from 'three';
import { Engine, PostFX, FreeFlyRig, InputMap, LoadingScreen, createBenchHarness, readBenchParams } from '@vibe/engine';
import { MapWorld } from '@vibe/engine/world';

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
});
engine.addSystem(world.system({ camera }), 'world');
const t0 = performance.now();
await engine.load((p, label) => loading.setProgress(p, label));
const loadMs = Math.round(performance.now() - t0);

let postfx = null;
if (!NOFX) {
  try {
    const { width, height } = engine.size;
    postfx = new PostFX(renderer, scene, camera, { atmosphere: world.atmo, width, height, pixelRatio: engine.pixelRatio });
  } catch (e) { console.warn('[city] post-processing unavailable', e); world.atmo.setFogMode('scene'); }
}
engine.renderFn = (sc, cam, dt) => { if (postfx) postfx.render(dt); else { renderer.render(sc, cam); world.atmo.renderFlare(renderer, cam); } };
engine.onResize = (w, h, pr) => { if (postfx) postfx.setSize(w, h, pr); };

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
const sample = createBenchHarness(engine, {
  name: 'city',
  poses: {
    street: (cam) => { cam.position.set(S.x, S.y + 2.2, S.z); cam.up.set(0, 1, 0); cam.lookAt(S.x + sx * 60, S.y + 2, S.z + sz * 60); },
    crossing: (cam) => { cam.position.set(S.x + sx * 80, S.y + 2.2, S.z + sz * 80); cam.up.set(0, 1, 0); cam.lookAt(S.x + sx * 140, S.y + 2, S.z + sz * 140); },
    high: (cam) => { cam.position.set(S.x, S.y + 60, S.z); cam.up.set(0, 1, 0); cam.lookAt(S.x + sx * 100, S.y, S.z + sz * 100); },
  },
  after: (cam) => {
    for (let i = 0; i < 3; i++) world.update(1, cam.position);
    const su = world.atmo.sky.material.uniforms; su.uTime.value = 0; su.uCloudOffset.value.set(0.13, 0.41);
    world.time = 0; if (world.mat && world.mat.water && world.mat.water.userData.u) world.mat.water.userData.u.uTime.value = 0;
  },
  quiet: () => { rig.deactivate(); if (postfx) { postfx.autoExposure = false; postfx.setTimeOfDay(world.atmo.darkness); } },
  settings: {
    'postfx.enabled': (v) => { if (postfx) postfx.setEnabled(!!v); },
    'postfx.aa': (v) => { if (postfx) postfx.setMSAA(v === 'fxaa' ? 0 : v === 'msaa' ? 4 : 'auto', v === 'msaa' ? Infinity : MSAA0); },
    'postfx.ao': (v) => { if (postfx) postfx.aoPass.enabled = !!v; },
    'shadows.kernel': (v) => world.atmo.setShadowKernel(v),
    'time': (v) => { world.atmo.setTimeOfDay(v); if (postfx) postfx.setTimeOfDay(world.atmo.darkness); },
  },
});
const bp = readBenchParams(location.search); if (bp.size) sample.resize(bp.size.w, bp.size.h);
sample.setReady();
renderer.compile(scene, camera);
window.__city = { THREE, engine, world, postfx, atmo: world.atmo, rig, input, loadMs, place, stats: world.stats };
console.info(`[city] ready in ${loadMs} ms`, world.stats);
engine.start();
loading.hide(300);
