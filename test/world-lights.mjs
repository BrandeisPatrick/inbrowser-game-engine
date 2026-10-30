// Clustered lamp lights (MapWorld `lights.clustered` -> Props -> NearestLights mode 'clustered'): the procedural city
// built headless with a recording ClusteredLights stand-in instance (the contract's methods, nothing rendered), then
// the default build (8-PointLight pool) for comparison. Run: node --import ./test/helpers/dom-shim.mjs test/world-lights.mjs
// MapWorld loads render/ClusteredLights.js (installClusteredChunks / syncClustered / applyClustered) for any
// `lights.clustered`; `clustered: true` also constructs the real class headless.
import * as THREE from 'three';
import { MapWorld } from '@vibe/engine/world';

let failures = 0;
const results = [];
const fmt = (x) => (typeof x === 'number' ? (Number.isInteger(x) ? String(x) : x.toFixed(4)) : String(x));
function check(name, value, lo, hi = lo, unit = '') {
  const ok = typeof value === 'number' ? Number.isFinite(value) && value >= lo && value <= hi : value === lo;
  if (!ok) failures++;
  results.push(`${ok ? 'PASS' : 'FAIL'}  ${name.padEnd(56)} ${fmt(value).padStart(10)} ${unit}   [${fmt(lo)}${hi !== lo ? ' .. ' + fmt(hi) : ''}]`);
  return ok;
}
for (const k of ['info', 'log']) console[k] = () => {}; // the build logs its stats
let noCamWarnings = 0; // MapWorld: clustered mode without atmo.camera warns once at night
{ const warn = console.warn; console.warn = (...a) => { if (/atmo\.camera/.test(String(a[0]))) noCamWarnings++; else warn(...a); }; }

/** Records what NearestLights / MapWorld do with a ClusteredLights (contract: setLight, count, update, setSize, enabled). */
class FakeClustered {
  constructor() { this.maxLights = 1024; this.enabled = true; this.lights = new Map(); this._count = 0; this.setLights = 0; this.counts = []; this.updates = 0; this.camera = null; this.sizes = []; }
  setLight(i, x, y, z, range, r, g, b, decay = 2) { this.setLights++; this.lights.set(i, { x, y, z, range, r, g, b, decay }); }
  setSpot() {}
  clearLight(i) { this.lights.delete(i); }
  get count() { return this._count; }
  set count(n) { this.counts.push(n); this._count = n; for (const i of [...this.lights.keys()]) if (i >= n) this.lights.delete(i); }
  update(camera) { this.updates++; this.camera = camera; this.setLightsAtUpdate = this.setLights; this.countAtUpdate = this._count; }
  setSize(w, h) { this.sizes.push([w, h]); }
  get stats() { return {}; }
  dispose() {}
}

const LAMP_Y = -0.35, RANGE = 220, FADE = 30, LAMP_COLOR = new THREE.Color(0xffc98a); // Props lamp pool values
const poolIntensity = (darkness) => (darkness > 0.3 ? 260 * Math.min(1, darkness * 1.3) : 0); // Props.update
const build = async (o) => {
  const world = new MapWorld({ scene: new THREE.Scene(), renderer: null, forceProcedural: true, traffic: false, streaming: false, ...o });
  await world.load(() => {});
  return world;
};
const pointLights = (world) => { let n = 0; world.scene.traverse((o) => { if (o.isPointLight) n++; }); return n; };
/** Lamps within RANGE of cam: Map key "x,y,z" -> distance (the positions NearestLights gives a light). */
function lampsInRange(world, cam) {
  const m = new Map(); let beyond = 0;
  for (const h of world.props.lampHeads) {
    const x = Math.fround(h[0]), y = Math.fround(h[1]) + LAMP_Y, z = Math.fround(h[2]);
    const d2 = (x - cam.x) ** 2 + (y - cam.y) ** 2 + (z - cam.z) ** 2; // as NearestLights
    if (d2 <= RANGE * RANGE) m.set(`${x},${y},${z}`, Math.sqrt(d2)); else beyond++;
  }
  return { m, beyond };
}
/** The fake's light list against the expected lamps: positions, range, decay, colour = linear lamp colour x intensity x edge fade. */
function compareList(fake, want, intensity) {
  let missing = 0, extra = 0, maxErr = 0, faded = 0, rangeDecayOk = true;
  const seen = new Set();
  for (let i = 0; i < fake.count; i++) {
    const L = fake.lights.get(i);
    if (!L) { missing++; continue; }
    const key = `${L.x},${L.y},${L.z}`, d = want.m.get(key);
    if (d === undefined) { extra++; continue; }
    seen.add(key);
    if (L.range !== 48 || L.decay !== 2) rangeDecayOk = false;
    const k = Math.min(1, (RANGE - d) / FADE); if (k < 1) faded++;
    for (const c of ['r', 'g', 'b']) maxErr = Math.max(maxErr, Math.abs(L[c] - LAMP_COLOR[c] * intensity * k) / (LAMP_COLOR[c] * intensity));
  }
  missing += want.m.size - seen.size;
  return { missing, extra, maxErr, faded, rangeDecayOk };
}

// =============================================================================================== clustered (fake)
const fake = new FakeClustered();
const W = await build({ lights: { clustered: fake } });
const LP = W.props.lightPool;
check('stats.lamps unchanged (world-build pin)', W.stats.lamps, 698);
check('world.clustered is the instance passed', W.clustered === fake ? 1 : 0, 1);
check('lamp pool in mode clustered', LP.mode, 'clustered');
check('no PointLights in the scene (one program per material)', pointLights(W), 0);
check('props.pointLights empty', W.props.pointLights.length, 0);

const sp = W.getSpawn().position, cam = sp.clone(); cam.y += 2;
// --- day: nothing pushed
W.setTimeOfDay('noon');
for (let i = 0; i < 10; i++) W.update(1 / 60, cam);
check('noon: darkness <= 0.3', W.atmo.darkness, 0, 0.3);
check('noon: setLight calls', fake.setLights, 0);
check('noon: cluster count', fake.count, 0);

// --- night: every lamp within 220 m, colour x intensity as the pool's PointLights
W.setTimeOfDay('night');
W.update(1 / 60, cam);
const dNight = W.atmo.darkness, INight = poolIntensity(dNight);
let want = lampsInRange(W, cam);
let cmp = compareList(fake, want, INight);
check('night: darkness > 0.3', dNight, 0.3001, 1);
check('night: lamps within 220 m of the spawn', want.m.size, 20, 698);
check('night: lamps beyond 220 m (not pushed)', want.beyond, 1, 698);
check('night: cluster count = lamps in range', fake.count, want.m.size);
check('night: lamps in range missing from the list', cmp.missing, 0);
check('night: lights not in range (or not lamps)', cmp.extra, 0);
check('night: range 48 m, decay 2', cmp.rangeDecayOk ? 1 : 0, 1);
check('night: rgb = linear 0xffc98a x intensity x fade, rel err', cmp.maxErr, 0, 1e-5);
check('night: lamps in the 30 m edge fade', cmp.faded, 1, 698);
check('night: lightPool.pushed = count', LP.pushed, fake.count);
check('night without atmo.camera: no clustered.update', fake.updates, 0);

// --- intensity scales with darkness exactly as the pool's: a twilight hour with 0.3 < darkness < 1 / 1.3
let hTw = null;
for (let h = 17; h <= 21 && hTw === null; h += 0.02) { W.setTimeOfDay(h); if (W.atmo.darkness > 0.4 && W.atmo.darkness < 0.7) hTw = h; }
W.setTimeOfDay(hTw); W.update(1 / 60, cam);
const ITw = poolIntensity(W.atmo.darkness);
cmp = compareList(fake, want, ITw);
check('twilight: intensity 260 * darkness * 1.3 (< 260)', ITw, 105, 259);
check('twilight: rgb follows the darkness, rel err', cmp.maxErr, 0, 1e-5);
check('twilight: same lamps', cmp.missing + cmp.extra, 0);
W.setTimeOfDay('night'); W.update(1 / 60, cam);

// --- throttle: < 4 m of camera travel, same intensity: no re-push; > 4 m: the list follows the camera
let n0 = fake.setLights;
const c2 = cam.clone(); c2.x += 1.5;
for (let i = 0; i < 30; i++) W.update(1 / 60, c2);
check('camera moved 1.5 m: setLight calls', fake.setLights - n0, 0);
const c3 = cam.clone(); c3.x += 150; c3.z -= 60;
W.update(1 / 60, c3);
want = lampsInRange(W, c3); cmp = compareList(fake, want, INight);
check('camera moved 160 m: re-pushed', fake.setLights - n0, 1, 1e9);
check('camera moved 160 m: count = lamps in range', fake.count, want.m.size);
check('camera moved 160 m: missing + extra', cmp.missing + cmp.extra, 0);

// --- clustered.update(atmo.camera) once per world.update, world.setSize -> drawing-buffer pixels
const camera = new THREE.PerspectiveCamera(60, 1.5, 0.5, 6000);
camera.position.copy(c3);
W.atmo.camera = camera;
const u0 = fake.updates;
for (let i = 0; i < 5; i++) W.update(1 / 60, c3);
check('clustered.update calls per world.update (5 frames)', fake.updates - u0, 5);
check('clustered.update gets atmo.camera', fake.camera === camera ? 1 : 0, 1);
// ordering: the frame's lamps are pushed before clustered.update clusters them (same world.update)
n0 = fake.setLights;
const c4 = c3.clone(); c4.z += 120; camera.position.copy(c4);
W.update(1 / 60, c4);
check('moved with a camera: lamps re-pushed this frame', fake.setLights - n0, 1, 1e9);
check('clustered.update after the push: setLight calls seen', fake.setLightsAtUpdate, fake.setLights);
check('clustered.update after the push: count seen', fake.countAtUpdate, fake.count);
check('no-camera warning at night: once', noCamWarnings, 1);
W.setSize(800, 450, 2);
check('world.setSize(800, 450, 2) -> setSize(1600, 900)', fake.sizes.length ? fake.sizes[fake.sizes.length - 1].join('x') : '', '1600x900');

// --- back to day: the list empties, nothing pushed
n0 = fake.setLights;
W.setTimeOfDay('noon');
for (let i = 0; i < 5; i++) W.update(1 / 60, c3);
check('noon again: cluster count', fake.count, 0);
check('noon again: setLight calls', fake.setLights - n0, 0);

// --- runtime switch: pool <-> clustered
W.atmo.camera = null; // no frustum: the pool picks the 8 lamps nearest the camera
W.setTimeOfDay('night');
const u1 = fake.updates;
check('setLightMode(pool)', W.setLightMode('pool'), 'pool');
check('pool: clustered.update(null) once (zeroes the shader light count)', fake.updates - u1 === 1 && fake.camera === null ? 1 : 0, 1);
n0 = fake.setLights;
check('pool: cluster emptied', fake.count, 0);
check('pool: clustered.enabled off', fake.enabled ? 1 : 0, 0);
for (let i = 0; i < 3; i++) W.update(1 / 60, cam);
const poolLit = W.props.pointLights.filter((l) => l.visible && l.intensity > 0) // snapshot: setMode zeroes them
  .map((l) => ({ x: l.position.x, y: l.position.y, z: l.position.z, r: l.color.r * l.intensity, g: l.color.g * l.intensity, b: l.color.b * l.intensity, distance: l.distance, decay: l.decay }));
check('pool: PointLights created', pointLights(W), 8);
check('pool: lit at night', poolLit.length, 8);
check('pool: nothing pushed', fake.setLights - n0, 0);
check('setLightMode(clustered)', W.setLightMode('clustered'), 'clustered');
check('clustered.enabled on', fake.enabled ? 1 : 0, 1);
W.update(1 / 60, cam);
want = lampsInRange(W, cam); cmp = compareList(fake, want, INight);
check('clustered again: pool lights hidden', W.props.pointLights.filter((l) => l.visible).length, 0);
check('clustered again: count = lamps in range', fake.count, want.m.size);
// every lamp the pool lit is in the cluster list with the pool light's colour x intensity (the same look per lamp)
let same = 0;
for (const l of poolLit) {
  for (let i = 0; i < fake.count; i++) {
    const L = fake.lights.get(i);
    if (L.x !== l.x || L.y !== l.y || L.z !== l.z) continue;
    const e = Math.max(...['r', 'g', 'b'].map((c) => Math.abs(L[c] - l[c]) / l[c]));
    if (e < 1e-5 && L.range === l.distance && L.decay === l.decay) same++;
  }
}
check('pool lights found in the list, same rgb x intensity', same, 8);

// =============================================================================================== default: the pool
const D = await build({});
check('default: world.clustered', D.clustered, null);
check('default: lamp pool mode', D.props.lightPool.mode, 'pool');
check('default: PointLights in the scene', pointLights(D), 8);
check('default: setLightMode is a no-op', D.setLightMode('clustered'), 'pool');
D.setTimeOfDay('noon');
for (let i = 0; i < 3; i++) D.update(1 / 60, cam);
check('default noon: PointLights visible', D.props.pointLights.filter((l) => l.visible).length, 0);
D.setTimeOfDay('night');
for (let i = 0; i < 3; i++) D.update(1 / 60, cam);
const dl = D.props.pointLights;
check('default night: PointLights visible', dl.filter((l) => l.visible).length, 8);
check('default night: intensity 260 * min(1, darkness * 1.3)', Math.max(...dl.map((l) => Math.abs(l.intensity - poolIntensity(D.atmo.darkness)))), 0, 1e-9);
check('default night: lamp colour, distance 48, decay 2', dl.every((l) => l.color.equals(LAMP_COLOR) && l.distance === 48 && l.decay === 2 && l.name === 'lamp-light') ? 1 : 0, 1);

// =============================================================================================== clustered: true
const T = await build({ lights: { clustered: true } });
const TC = T.clustered;
check('clustered true: world.clustered created', TC && typeof TC.setLight === 'function' && typeof TC.update === 'function' ? 1 : 0, 1);
check('clustered true: lamp pool mode', T.props.lightPool.mode, 'clustered');
T.setTimeOfDay('night');
T.update(1 / 60, cam);
check('clustered true: count = lamps in range', TC ? TC.count : -1, lampsInRange(T, cam).m.size);
const tCam = new THREE.PerspectiveCamera(60, 16 / 9, 0.5, 6000);
tCam.position.copy(cam); tCam.lookAt(cam.x, cam.y - 1, cam.z - 40); // looking down the street at the lamps around the spawn
T.atmo.camera = tCam;
T.update(1 / 60, cam);
const P2 = TC.uniforms.engineClParams2.value;
check('clustered true: lights in view after update (params2.w)', P2[3], 1, 698);
T.setLightMode('pool');
check('clustered true: setLightMode(pool) zeroes params2.w', P2[3], 0);

console.error(results.join('\n'));
console.error(failures ? `\n${failures} lights check(s) FAILED` : '\nAll lights checks passed.');
process.exit(failures ? 1 : 0);
