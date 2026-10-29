// Procedural city harness (dev only, http://localhost:5190/city): MapWorld with its built-in procedural city (no map
// data needed), the engine's post stack and a free-fly camera, so the world layer can be seen and measured in a
// browser without the game repository. Query: ?t=night|hours, ?traffic=N | ?notraffic, ?nostream=1, ?nofx=1,
// ?profile=1 (profiler table after N s), F3 = profiler overlay. window.__city exposes everything for the console.
import * as THREE from 'three';
import { Engine, PostFX, FreeFlyRig, InputMap, LoadingScreen } from '@vibe/engine';
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

renderer.compile(scene, camera);
window.__city = { THREE, engine, world, postfx, atmo: world.atmo, rig, input, loadMs, place, stats: world.stats };
console.info(`[city] ready in ${loadMs} ms`, world.stats);
engine.start();
loading.hide(300);
