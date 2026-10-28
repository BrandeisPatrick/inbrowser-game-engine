// Headless build of the procedural city: the world layer (MapWorld + roads, buildings, props, traffic, physics)
// must build in Node without a renderer or DOM, deterministically. Run: node --import ./test/helpers/dom-shim.mjs test/world-build.mjs
// (npm test runs it after car-sim). The reference numbers pin the current build; an intended change updates them.
import * as THREE from 'three';
import { MapWorld } from '@vibe/engine/world';

let failures = 0;
const results = [];
const fmt = (x) => (typeof x === 'number' ? (Number.isInteger(x) ? String(x) : x.toFixed(4)) : String(x));
function check(name, value, lo, hi = lo, unit = '') {
  const ok = typeof value === 'number' ? Number.isFinite(value) && value >= lo && value <= hi : value === lo;
  if (!ok) failures++;
  results.push(`${ok ? 'PASS' : 'FAIL'}  ${name.padEnd(44)} ${fmt(value).padStart(10)} ${unit}   [${fmt(lo)}${hi !== lo ? ' .. ' + fmt(hi) : ''}]`);
  return ok;
}

for (const k of ['info', 'log']) console[k] = () => {}; // the build logs its stats; keep the check output readable
const t0 = performance.now();
const scene = new THREE.Scene();
const world = new MapWorld({ scene, renderer: null, forceProcedural: true, traffic: { count: 250, worker: false }, streaming: false, physics: true });
await world.load(() => {});
const loadMs = performance.now() - t0;
const s = world.stats;

// --- build ---------------------------------------------------------------------------------------------------
check('headless load time (ms)', loadMs, 0, 30000, 'ms');
check('source is the procedural city', s.source, 'procedural');
check('buildings', s.buildings, 1775);
check('roads', s.roads, 41);
check('bridges', s.bridges, 1);
check('colliders (buildings, piers, furniture, trees)', s.colliders, 7927);
check('traffic vehicles', s.vehicles, 250);
check('traffic runs in-thread headless', s.trafficSim, 'main');
check('static mesh tiles', s.chunkTiles, 55);
check('lamps', s.lamps, 698);
check('trees', s.trees, 1202);
check('physics world attached', world.physics ? 1 : 0, 1);

// --- queries -------------------------------------------------------------------------------------------------
const sp = world.getSpawn();
check('spawn at the origin (bake frame shifted)', Math.hypot(sp.position.x, sp.position.z), 0, 1e-6, 'm');
check('ground at the spawn (road lift)', world.getGroundHeight(sp.position.x, sp.position.z, 0), 0.05, 0.05, 'm');
const hit = world.physics.raycast(new THREE.Vector3(0, 50, 0), new THREE.Vector3(0, -1, 0), 200);
check('physics raycast down hits the ground', hit ? hit.distance : NaN, 49.9, 50.0, 'm');
let maxBoxes = 0; for (let i = 0; i < 50; i++) maxBoxes = Math.max(maxBoxes, world.queryColliders(i * 20 - 500, i * 15 - 400, 30).length);
check('collider queries find boxes around the district', maxBoxes, 1, 1e9);

// --- building shells sit at their own origin (front-to-back sorting), nothing left at the world origin ---------
let shells = 0, atOrigin = 0;
world.group.traverse((m) => { if (m.isMesh && /^buildings/.test(m.name)) { shells++; if (m.position.x === 0 && m.position.z === 0) atOrigin++; } });
check('building shell meshes', shells, 160);
check('building shells left at the origin', atOrigin, 0);

// --- traffic: 10 s in-thread, deterministic ------------------------------------------------------------------
const sim = world.traffic.sim;
for (let i = 0; i < 600; i++) sim.update(1 / 60);
const sum = sim.summary();
check('traffic: vehicles after 10 s', sum.count, 250);
check('traffic: moving after 10 s', sum.moving, 204);
check('traffic: mean speed (m/s), deterministic', sum.avgSpeed, 4.648989387833288, 4.648989387833288, 'm/s');
check('traffic: signal entries on red', sum.redEntries, 0);

// --- per-frame update at the spawn --------------------------------------------------------------------------
const cam = sp.position.clone(); cam.y += 2;
const t1 = performance.now();
for (let i = 0; i < 120; i++) world.update(1 / 60, cam);
check('world.update (traffic in-thread) ms / frame', (performance.now() - t1) / 120, 0, 20, 'ms');

console.error(results.join('\n'));
console.error(failures ? `\n${failures} world check(s) FAILED` : '\nAll world checks passed.');
process.exit(failures ? 1 : 0);
