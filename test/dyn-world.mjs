// Headless checks for the XPBD DynamicsWorld (src/physics/dynamics): resting, stacking, a 600-box pit with a
// determinism hash, restitution, friction, kinematic pushing, sleeping / waking, removal, contact events, a
// serialize / from round trip, the PhysicsWorld collider records, the bridge and the delegating queries, and allocation
// in step(). No browser, no WebGL.
// Run: npm test  (node test/dyn-world.mjs). Skips (exit 0) while the dynamics modules it needs are not all present.
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import v8 from 'node:v8';
import vm from 'node:vm';

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const NEED = ['state.js', 'shapes.js', 'broadphase.js', 'narrowphase.js', 'xpbd.js', 'joints.js', 'queries.js', 'debug.js', 'islands.js', 'DynamicsWorld.js'];
const missing = NEED.filter((f) => !existsSync(here(`../src/physics/dynamics/${f}`)));
if (missing.length) {
  console.log(`dyn-world: SKIPPED, dynamics modules missing: ${missing.join(', ')}`);
  process.exit(0);
}
const { DynamicsWorld } = await import('../src/physics/dynamics/DynamicsWorld.js');
const { box, sphere, capsule, convex, compound } = await import('../src/physics/dynamics/shapes.js');
const { WORLD_BODY, pairKey } = await import('../src/physics/dynamics/state.js');
const { JOINT } = await import('../src/physics/dynamics/joints.js');
const { PhysicsWorld } = await import('@vibe/engine/physics');
const { hashF64Array, hashU32, mulberry32 } = await import('../src/core/mathDet.js');

const DT = 1 / 120;
let failures = 0;
const results = [];
const fmt = (x) => (!Number.isFinite(x) ? String(x) : Number.isInteger(x) ? String(x) : Math.abs(x) < 1e-3 && x !== 0 ? x.toExponential(2) : x.toFixed(4)).padStart(12);
let _allocSink = null; // section 14's control stores its arrays here so V8 cannot elide them
function check(name, value, lo, hi, unit = '') {
  const v = typeof value === 'boolean' ? (value ? 1 : 0) : value;
  const ok = Number.isFinite(v) && v >= lo && v <= hi;
  if (!ok) failures++;
  results.push(`${ok ? 'PASS' : 'FAIL'}  ${name.padEnd(58)} ${fmt(v)} ${unit}   [${fmt(lo).trim()}${hi !== lo ? ' .. ' + fmt(hi).trim() : ''}]`);
  return ok;
}
const note = (s) => results.push(`NOTE  ${s}`);

function run(w, seconds, each) {
  const n = Math.round(seconds / DT);
  for (let i = 0; i < n; i++) { if (each) each(i); w.step(DT); }
}
function hashWorld(w) {
  const B = w.bodies, n = B.count;
  let h = hashF64Array(B.pos, 2166136261, 0, n * 3);
  h = hashF64Array(B.quat, h, 0, n * 4);
  h = hashF64Array(B.vel, h, 0, n * 3);
  h = hashF64Array(B.angVel, h, 0, n * 3);
  for (let i = 0; i < n; i++) h = hashU32(h, B.flags[i] | (B.alive[i] << 8));
  return h;
}
function allFinite(w) {
  const B = w.bodies;
  for (let i = 0; i < B.count; i++) {
    if (!B.alive[i]) continue;
    for (let k = 0; k < 3; k++) if (!Number.isFinite(B.pos[i * 3 + k]) || !Number.isFinite(B.vel[i * 3 + k]) || !Number.isFinite(B.angVel[i * 3 + k])) return false;
    for (let k = 0; k < 4; k++) if (!Number.isFinite(B.quat[i * 4 + k])) return false;
  }
  return true;
}
const P = (w, id) => { const o = id * 3, p = w.bodies.pos; return { x: p[o], y: p[o + 1], z: p[o + 2] }; };
const worldKey = (id) => pairKey(id, WORLD_BODY) >>> 0;
const qAxis = (ax, ay, az, deg) => { const h = (deg * Math.PI) / 360, s = Math.sin(h); return [ax * s, ay * s, az * s, Math.cos(h)]; };

// --- 1. a box dropped on the static ground rests at its half extent and sleeps; applyImpulse wakes it ----------
{
  const physics = new PhysicsWorld({ groundLevel: 0 });
  const w = new DynamicsWorld({ physics });
  const id = w.addBody({ shape: box(0.25, 0.25, 0.25), position: [0.3, 1.2, -0.2] });
  run(w, 2);
  check('drop: box rests at its half extent (y)', P(w, id).y, 0.25 - 1e-4, 0.25 + 1e-4, 'm');
  check('drop: box asleep after 2 s', !w.isAwake(id), 1, 1);
  check('drop: horizontal drift', Math.hypot(P(w, id).x - 0.3, P(w, id).z + 0.2), 0, 1e-3, 'm');
  const m = w.bodies.mass[id];
  w.applyImpulse(id, 0.3, P(w, id).y, -0.2, 0, 3 * m, 0);
  check('applyImpulse wakes a sleeping body', w.isAwake(id), 1, 1);
  run(w, 0.2);
  check('applyImpulse: the body flies up', P(w, id).y, 0.6, 2, 'm');
}

// --- 2. a 10-box stack is stable for 10 s (substeps 4, dt 1/120) -------------------------------------------------
for (const sleep of [undefined, false]) {
  const physics = new PhysicsWorld({ groundLevel: 0 });
  const w = new DynamicsWorld({ physics, substeps: 4, ...(sleep === false ? { sleep: false } : {}) });
  const ids = [];
  for (let i = 0; i < 10; i++) ids.push(w.addBody({ shape: box(0.25, 0.25, 0.25), position: [0, 0.25 + i * 0.5, 0] }));
  const top = ids[9];
  let maxDrift = 0;
  run(w, 10, () => { const p = P(w, top); const d = Math.hypot(p.x, p.z); if (d > maxDrift) maxDrift = d; });
  const label = sleep === false ? 'stack (sleep off)' : 'stack';
  check(`${label}: no NaN after 10 s`, allFinite(w), 1, 1);
  check(`${label}: top box horizontal drift (max over 10 s)`, maxDrift, 0, 1e-3, 'm');
  check(`${label}: top box height`, P(w, top).y, 4.75 - 0.01, 4.75 + 0.005, 'm');
  if (sleep !== false) check('stack: asleep after 10 s', w.stats.awake, 0, 0);
}

// --- 3. 600 boxes fall into a 20 m pit (static walls in the PhysicsWorld): sane, deterministic, timed -----------
const PIT = 600;
function buildPit(timing = false) {
  const physics = new PhysicsWorld({ groundLevel: 0 });
  physics.addStaticBox(-11, 0, -10, 6, -11, 11);   // x walls
  physics.addStaticBox(10, 0, 11, 6, -11, 11);
  physics.addStaticBox(-10, 0, 10, 6, -11, -10);   // z walls
  physics.addStaticBox(-10, 0, 10, 6, 10, 11);
  const w = new DynamicsWorld({ physics, maxBodies: 1024, maxContacts: 16384, timing });
  const rnd = mulberry32(7);
  for (let i = 0; i < PIT; i++) {
    const gx = i % 10, gz = ((i / 10) | 0) % 10, gy = (i / 100) | 0;
    const ax = rnd() - 0.5, ay = rnd() - 0.5, az = rnd() - 0.5, al = Math.hypot(ax, ay, az) || 1;
    w.addBody({
      shape: box(0.25, 0.25, 0.25),
      position: [-8.1 + gx * 1.8 + (rnd() - 0.5) * 0.4, 1 + gy * 1.1 + rnd() * 0.3, -8.1 + gz * 1.8 + (rnd() - 0.5) * 0.4],
      quaternion: qAxis(ax / al, ay / al, az / al, rnd() * 180),
    });
  }
  return w;
}
{
  const N1 = Math.round(2 / DT);
  const a = buildPit(true);
  let tPeak = 0, msPeak = 0;
  const t0 = performance.now();
  for (let i = 0; i < N1; i++) { const t = performance.now(); a.step(DT); const ms = performance.now() - t; if (ms > msPeak) { msPeak = ms; tPeak = i * DT; } }
  const msFall = (performance.now() - t0) / N1;
  const hA = hashWorld(a);
  const b = buildPit(false);
  for (let i = 0; i < N1; i++) b.step(DT);
  const hB = hashWorld(b);
  check('pit: determinism hash identical across two runs', hA === hB ? 1 : 0, 1, 1);
  results.push(`NOTE  pit: hash after 2 s 0x${hA.toString(16)} / 0x${hB.toString(16)} (the second run without timing)`);
  run(a, 4);
  const B = a.bodies;
  let minY = Infinity, inside = 0;
  for (let i = 0; i < B.count; i++) {
    const y = B.pos[i * 3 + 1], x = B.pos[i * 3], z = B.pos[i * 3 + 2];
    if (y < minY) minY = y;
    if (Math.abs(x) < 10 && Math.abs(z) < 10) inside++;
  }
  check('pit: no NaN after 6 s', allFinite(a), 1, 1);
  check('pit: every box above the floor (min centre y)', minY, 0.25 - 0.02, 10, 'm');
  check('pit: every box inside the walls', inside, PIT, PIT);
  // settled cost
  const nT = 120, t1 = performance.now();
  let br = 0, na = 0, so = 0;
  for (let i = 0; i < nT; i++) { a.step(DT); br += a.stats.msBroad; na += a.stats.msNarrow; so += a.stats.msSolve; }
  const msSettled = (performance.now() - t1) / nT;
  check('pit: step time once settled (Node, target <= 8)', msSettled, 0, 8, 'ms');
  note(`pit: falling phase ${msFall.toFixed(2)} ms/step avg (peak ${msPeak.toFixed(1)} ms at t=${tPeak.toFixed(2)} s); settled ${msSettled.toFixed(2)} ms/step ` +
    `(broad ${(br / nT).toFixed(2)}, narrow ${(na / nT).toFixed(2)}, solve ${(so / nT).toFixed(2)}); awake ${a.stats.awake}, contacts ${a.stats.contacts}, ` +
    `pairs ${a.stats.pairs}, islands ${a.stats.islands}`);
}

// --- 4. sphere bounce: restitution 0.9 returns >= 75 % of the drop height -----------------------------------------
{
  const physics = new PhysicsWorld({ groundLevel: 0 });
  const w = new DynamicsWorld({ physics });
  const id = w.addBody({ shape: sphere(0.5), position: [0, 2.5, 0], restitution: 0.9 });
  let phase = 0, peak = 0;
  run(w, 3, () => {
    const vy = w.bodies.vel[id * 3 + 1], y = P(w, id).y;
    if (phase === 0 && vy > 0.5) phase = 1;             // bounced
    if (phase === 1) { if (y > peak) peak = y; if (vy < 0) phase = 2; }
  });
  check('bounce: restitution 0.9, rebound / drop height', (peak - 0.5) / 2, 0.75, 0.9);
}

// --- 5. friction: a box on a 20 deg slope stays with mu 0.6 and slides with mu 0.2 -------------------------------
for (const mu of [0.6, 0.2]) {
  const w = new DynamicsWorld();
  const q = qAxis(0, 0, 1, 20);
  w.addBody({ type: 'static', shape: box(5, 0.25, 5), position: [0, 3, 0], quaternion: q, friction: mu });
  // on the slope's top face: local (0, 0.5, 0) rotated
  const s = Math.sin((20 * Math.PI) / 180), c = Math.cos((20 * Math.PI) / 180);
  const p0 = [-0.5 * s, 3 + 0.5 * c + 0.002, 0];
  const id = w.addBody({ shape: box(0.25, 0.25, 0.25), position: p0, quaternion: q, friction: mu });
  let maxW = 0;
  run(w, 2, () => { const a = w.bodies.angVel, m = Math.hypot(a[id * 3], a[id * 3 + 1], a[id * 3 + 2]); if (m > maxW) maxW = m; });
  const p = P(w, id), moved = Math.hypot(p.x - p0[0], p.y - p0[1], p.z - p0[2]);
  if (mu > 0.5) check('slope 20 deg, mu 0.6: the box stays', moved, 0, 0.01, 'm');
  else {
    check('slope 20 deg, mu 0.2: the box slides (2 s, ~3 m expected)', moved, 1.5, 4, 'm');
    check('slope 20 deg, mu 0.2: the sliding box does not spin (max |w|)', maxW, 0, 1e-3, 'rad/s');
  }
}

// --- 6. a kinematic box driven by targets pushes a dynamic box -----------------------------------------------------
{
  const physics = new PhysicsWorld({ groundLevel: 0 });
  const w = new DynamicsWorld({ physics });
  const pusher = w.addBody({ type: 'kinematic', shape: box(0.5, 0.5, 0.5), position: [-2, 0.5, 0] });
  const crate = w.addBody({ shape: box(0.25, 0.25, 0.25), position: [0, 0.25, 0] });
  run(w, 0.5); // the crate settles (and sleeps once still)
  let k = 0;
  run(w, 3, () => { k++; w.setKinematicTarget(pusher, -2 + k * DT, 0.5, 0, 0, 0, 0, 1); });
  check('kinematic: pusher lands exactly on its target', P(w, pusher).x, -2 + k * DT, -2 + k * DT, 'm');
  check('kinematic: crate pushed ahead of the pusher (x)', P(w, crate).x, 1 + 0.75 - 0.02, 1 + 0.75 + 0.1, 'm');
  check('kinematic: crate stays on the ground (y)', P(w, crate).y, 0.24, 0.26, 'm');
  run(w, 0.1);
  check('kinematic: stops without new targets', Math.abs(w.bodies.vel[pusher * 3]), 0, 0, 'm/s');
}

// --- 7. removeBody mid-run: the box above wakes and falls onto the one below; the id is reused --------------------
{
  const physics = new PhysicsWorld({ groundLevel: 0 });
  const w = new DynamicsWorld({ physics });
  const ids = [0, 1, 2].map((i) => w.addBody({ shape: box(0.25, 0.25, 0.25), position: [0, 0.25 + i * 0.5, 0] }));
  run(w, 3);
  check('remove: the stack sleeps first', w.stats.awake, 0, 0);
  w.removeBody(ids[1]);
  check('remove: the body above wakes at once', w.isAwake(ids[2]), 1, 1);
  run(w, 2);
  check('remove: it lands on the bottom box (y)', P(w, ids[2]).y, 0.75 - 0.01, 0.75 + 0.01, 'm');
  check('remove: no contact references the removed id', (() => { const C = w.contacts; for (let i = 0; i < C.count; i++) if (C.a[i] === ids[1] || C.b[i] === ids[1]) return 0; return 1; })(), 1, 1);
  const again = w.addBody({ shape: sphere(0.1), position: [3, 1, 0] });
  check('remove: the freed id is reused by the next addBody', again, ids[1], ids[1]);
}

// --- 8. began / ended events: once per landing, kept while asleep, ended when lifted ------------------------------
{
  const physics = new PhysicsWorld({ groundLevel: 0 });
  const w = new DynamicsWorld({ physics });
  const lower = w.addBody({ shape: box(0.5, 0.25, 0.5), position: [0, 0.25, 0], restitution: 0 });
  const upper = w.addBody({ shape: box(0.25, 0.25, 0.25), position: [0, 1.0, 0], restitution: 0 });
  const kWorld = worldKey(lower), kPair = pairKey(lower, upper);
  let beganW = 0, endedW = 0, beganP = 0, endedP = 0;
  const count = () => {
    for (let i = 0; i < w.began.count; i++) { if (w.began.keys[i] === kWorld) beganW++; if (w.began.keys[i] === kPair) beganP++; }
    for (let i = 0; i < w.ended.count; i++) { if (w.ended.keys[i] === kWorld) endedW++; if (w.ended.keys[i] === kPair) endedP++; }
  };
  for (let i = 0; i < Math.round(3 / DT); i++) { w.step(DT); count(); }
  check('events: lower box / ground began once', beganW, 1, 1);
  check('events: upper box landing on the lower one began once', beganP, 1, 1);
  check('events: nothing ended while resting and asleep', endedW + endedP, 0, 0);
  check('events: both asleep', w.stats.awake, 0, 0);
  w.setPose(upper, 0, 3, 0, 0, 0, 0, 1);
  w.step(DT); count();
  check('events: lifting the upper box ends its pair once', endedP, 1, 1);
}

// --- 9. serialize / from: both worlds continue bit-identically for 1 s --------------------------------------------
{
  const physics = new PhysicsWorld({ groundLevel: 0 });
  physics.addStaticBox(3, 0, 4, 2, -4, 4);
  const w = new DynamicsWorld({ physics });
  const rnd = mulberry32(3);
  const hull = convex([0.3, 0, 0, -0.3, 0, 0, 0, 0.35, 0, 0, -0.2, 0, 0, 0, 0.3, 0, 0, -0.3, 0.1, 0.1, 0.1]);
  const comp = compound([{ shape: box(0.3, 0.1, 0.1) }, { shape: sphere(0.15), position: [0.3, 0, 0] }, { shape: sphere(0.15), position: [-0.3, 0, 0] }]);
  for (let i = 0; i < 40; i++) {
    const shape = [box(0.2, 0.15, 0.25), sphere(0.2), capsule(0.12, 0.2), hull, comp][i % 5];
    w.addBody({ shape, position: [(rnd() - 0.5) * 4, 0.5 + i * 0.35, (rnd() - 0.5) * 4], quaternion: qAxis(0, 1, 0, rnd() * 90), friction: 0.3 + rnd() * 0.5 });
  }
  const kin = w.addBody({ type: 'kinematic', shape: box(0.4, 0.4, 0.4), position: [-3, 0.4, 0] });
  w.setVelocity(kin, 1, 0, 0, 0, 0.5, 0);
  w.removeBody(5);                       // a free-list entry in the snapshot
  // a hinged pendulum and a ball-jointed pair (joint state is part of the snapshot)
  const pend = w.addBody({ shape: box(0.1, 0.5, 0.1), position: [-6, 3, 2], quaternion: qAxis(0, 0, 1, 40) });
  w.addJoint({ type: JOINT.HINGE, bodyA: pend, bodyB: WORLD_BODY, anchorA: [0, 0.5, 0], axisA: [0, 0, 1] });
  const c0 = w.addBody({ shape: sphere(0.15), position: [-6, 3, -2] }), c1 = w.addBody({ shape: sphere(0.15), position: [-5.6, 3, -2] });
  w.addJoint({ type: JOINT.BALL, bodyA: c0, bodyB: WORLD_BODY, anchorA: [0, 0.2, 0] });
  w.addJoint({ type: JOINT.DISTANCE, bodyA: c0, bodyB: c1 });
  run(w, 0.5);
  const snap = w.serialize();
  const w2 = DynamicsWorld.from(structuredClone(snap), { physics });
  check('serialize: restored hash equals the original at the snapshot', hashWorld(w2) === hashWorld(w) ? 1 : 0, 1, 1);
  let evA = 0, evB = 0;
  run(w, 1, () => { evA += w.began.count * 3 + w.ended.count; });
  run(w2, 1, () => { evB += w2.began.count * 3 + w2.ended.count; });
  const id6 = w.addBody({ shape: sphere(0.1), position: [0, 5, 0] }), id6b = w2.addBody({ shape: sphere(0.1), position: [0, 5, 0] });
  check('serialize: original and restored hashes after 1 s (bit-identical)', hashWorld(w) === hashWorld(w2) ? 1 : 0, 1, 1);
  check('serialize: identical event streams', evA === evB ? 1 : 0, 1, 1);
  check('serialize: same next free id', id6 === id6b ? 1 : 0, 1, 1);
  check('serialize: time / stepCount carried over', w2.stepCount === w.stepCount && w2.time === w.time ? 1 : 0, 1, 1);
}

// --- 10. queryColliders: pooled records of nearby non-static bodies with their AABBs ------------------------------
{
  const w = new DynamicsWorld({ sleep: false, gravity: [0, 0, 0] });
  const a = w.addBody({ shape: box(0.5, 0.25, 0.75), position: [1, 2, 0], quaternion: qAxis(0, 1, 0, 30), linearDamping: 0 });
  const s = w.addBody({ shape: sphere(0.5), position: [8, 1, 0] });
  w.addBody({ type: 'static', shape: box(1, 1, 1), position: [0, 0, 0] });
  const k = w.addBody({ type: 'kinematic', shape: box(0.2, 0.2, 0.2), position: [0, 1, 1.5] });
  w.setVelocity(a, 0.5, 0, -0.25, 0, 0, 0);
  w.step(DT);
  const out = [];
  w.queryColliders(0, 0, 1.5, out);
  const ids = out.map((r) => r.id).join(',');
  check('queryColliders: finds the box and the kinematic, not the static or the far sphere', ids === `${a},${k}` ? 1 : 0, 1, 1);
  const r = out[0], A = w.bodies.aabb, o = a * 6;
  const exact = r.minX === A[o] && r.minY === A[o + 1] && r.minZ === A[o + 2] && r.maxX === A[o + 3] && r.maxY === A[o + 4] && r.maxZ === A[o + 5];
  check('queryColliders: record AABB = bodies.aabb', exact ? 1 : 0, 1, 1);
  // the rotated box's AABB half extent in x: 0.5 cos30 + 0.75 sin30 (+ margin)
  const hx = 0.5 * Math.cos(Math.PI / 6) + 0.75 * Math.sin(Math.PI / 6) + w.margin;
  check('queryColliders: rotated box AABB width', r.maxX - r.minX, 2 * hx - 1e-9, 2 * hx + 1e-9, 'm');
  check('queryColliders: record velocity (vx, vz)', r.vx * 10 + r.vz, 4.75 - 1e-9, 4.75 + 1e-9);
  check('queryColliders: kind', r.kind === 'rigid' ? 1 : 0, 1, 1);
  const out2 = [{ dummy: true }];
  w.queryColliders(8, 0, 0.6, out2);
  check('queryColliders: appends, and reuses the pooled record', out2.length === 2 && out2[1].id === s && out2[1] === w.queryColliders(8, 0, 0.6, [])[0] ? 1 : 0, 1, 1);
}

// --- 11. raycast: bodies and the static world, nearer wins ---------------------------------------------------------
{
  const physics = new PhysicsWorld({ groundLevel: 0 });
  const w = new DynamicsWorld({ physics });
  const id = w.addBody({ shape: box(0.5, 0.5, 0.5), position: [0, 0.5, 0] });
  w.step(DT);
  const h1 = w.raycast({ x: 0, y: 5, z: 0 }, { x: 0, y: -1, z: 0 });
  check('raycast: hits the box top (distance)', h1 ? h1.distance : NaN, 4 - 1e-3, 4 + 1e-3, 'm');
  check('raycast: body id', h1 ? h1.body : NaN, id, id);
  const h2 = w.raycast([3, 5, 0], [0, -1, 0]);
  check('raycast: misses the box, hits the ground (body -1)', h2 && h2.ground && h2.body === -1 ? h2.distance : NaN, 5 - 1e-3, 5 + 1e-3, 'm');
}

// --- 12. bridge, debug lines, shape cast, overlap: the world's bodies published to the PhysicsWorld are not read back
//         as static geometry (a box resting on the ground stays put), and the delegating queries answer ---------------
{
  const { attachToPhysicsWorld } = await import('../src/physics/dynamics/bridge.js');
  const physics = new PhysicsWorld({ groundLevel: 0 });
  const w = new DynamicsWorld({ physics });
  const id = w.addBody({ shape: box(0.25, 0.25, 0.25), position: [0, 0.25, 0] });
  const ball = w.addBody({ shape: sphere(0.2), position: [3, 0.2, 0] });
  attachToPhysicsWorld(w, physics);
  run(w, 1);
  check('bridge: a resting box ignores its own published AABB (y)', P(w, id).y, 0.25 - 1e-4, 0.25 + 1e-4, 'm');
  check('bridge: physics.queryColliders sees the bodies', physics.queryColliders(0, 0, 0.5, []).filter((r) => r.kind === 'rigid').length, 1, 1);
  const lines = new Float32Array(6 * 512);
  check('debugLines: segments written', w.debugLines(lines, 512), 12 + 48, 512);
  const hit = w.shapeCast(sphere(0.1), [3, 2, 0], [0, 0, 0, 1], [0, -1, 0], 10);
  check('shapeCast: a sphere swept down onto the ball (distance)', hit ? hit.distance : NaN, 1.5 - 2e-3, 1.5 + 2e-3, 'm');
  check('shapeCast: hit body', hit ? hit.body : NaN, ball, ball);
  const ids = new Int32Array(8);
  check('overlap: AABB query finds the box only', w.overlap(-0.5, 0, -0.5, 0.5, 1, 0.5, ids) === 1 && ids[0] === id ? 1 : 0, 1, 1);
  const r = w.raycast([0, 3, 0], [0, -1, 0], 10, { bodies: false });
  check('raycast: statics only (bodies: false) skips the published box record', r ? r.distance : NaN, 3 - 1e-3, 3 + 1e-3, 'm');
}

// --- 13. regressions: id reuse, queries between steps, jointed pairs -----------------------------------------------
{
  // a reused id must not inherit the static-query cache of its previous owner (a small sphere far from the wall)
  const physics = new PhysicsWorld({ groundLevel: 0 });
  const wall = physics.addStaticBox(2, 0, 3, 3, -3, 3);
  const w = new DynamicsWorld({ physics });
  const s0 = w.addBody({ shape: sphere(0.1), position: [0.8, 0.1, 0] });
  run(w, 0.1);
  w.removeBody(s0);
  const big = w.addBody({ shape: box(1.5, 1.5, 1.5), position: [0.8, 1.5, 0] });
  w.step(DT);
  let wallContacts = 0;
  for (let i = 0; i < w.contacts.count; i++) if (w.contacts.a[i] === big && w.contacts.b[i] === WORLD_BODY && w.contacts.staticIndex[i] === wall) wallContacts++;
  check('id reuse: a new body in a wall gets wall contacts (static-query cache reset)', big === s0 && wallContacts > 0 ? 1 : 0, 1, 1);
  run(w, 0.5);
  check('id reuse: ... and is pushed out of the wall (max x <= 2)', P(w, big).x + 1.5, 0, 2 + 2e-3, 'm');
}
{
  // a reused id must not inherit friction anchors (nor narrowphase warm starts): identical to a fresh world
  const make = () => new DynamicsWorld({ physics: new PhysicsWorld({ groundLevel: 0 }) });
  const w1 = make();
  const old = w1.addBody({ shape: box(0.25, 0.25, 0.25), position: [0, 0.25, 0] });
  run(w1, 0.3);                                    // resting, sticking: anchors cached
  w1.removeBody(old);
  const nb = w1.addBody({ shape: box(0.25, 0.25, 0.25), position: [0.1, 0.25, 0], quaternion: qAxis(0, 1, 0, 10) });
  w1.setVelocity(nb, 0.5, 0, 0, 0, 0, 0);
  const w2 = make();
  const fb = w2.addBody({ shape: box(0.25, 0.25, 0.25), position: [0.1, 0.25, 0], quaternion: qAxis(0, 1, 0, 10) });
  w2.setVelocity(fb, 0.5, 0, 0, 0, 0, 0);
  run(w1, 1); run(w2, 1);
  check('id reuse: same as a fresh world (no inherited anchors / warm starts)', nb === fb && hashWorld(w1) === hashWorld(w2) ? 1 : 0, 1, 1);
}
{
  // queries see bodies added / moved since the last step (the broadphase is synced on demand)
  const w = new DynamicsWorld({ gravity: [0, 0, 0] });
  w.step(DT);
  const id = w.addBody({ shape: box(0.5, 0.5, 0.5), position: [0, 0, 5] });
  const h1 = w.raycast([0, 0, 0], [0, 0, 1], 100);
  check('queries: raycast right after addBody', h1 && h1.body === id ? h1.distance : NaN, 4.5 - 1e-9, 4.5 + 1e-9, 'm');
  w.setPose(id, 0, 0, 9, 0, 0, 0, 1);
  const h2 = w.raycast([0, 0, 0], [0, 0, 1], 100);
  check('queries: raycast right after setPose', h2 && h2.body === id ? h2.distance : NaN, 8.5 - 1e-9, 8.5 + 1e-9, 'm');
  check('queries: pointInBody right after setPose', w.pointInBody(0, 0, 9), id, id);
  w.removeBody(id);
  check('queries: raycast right after removeBody misses', w.raycast([0, 0, 0], [0, 0, 1], 100, { statics: false }) === null ? 1 : 0, 1, 1);
}
{
  // two overlapping hinge-joined boxes do not collide (collideConnected false); a free overlapping pair does
  const w = new DynamicsWorld({ gravity: [0, 0, 0], sleep: false });
  const a = w.addBody({ shape: box(0.5, 0.5, 0.5), position: [0, 0, 0] });
  const b = w.addBody({ shape: box(0.5, 0.5, 0.5), position: [0.8, 0, 0] });
  const c = w.addBody({ shape: box(0.5, 0.5, 0.5), position: [0, 3, 0] });
  const d = w.addBody({ shape: box(0.5, 0.5, 0.5), position: [0.8, 3, 0] });
  w.addJoint({ type: JOINT.HINGE, bodyA: a, bodyB: b, anchorA: [0.4, 0, 0], axisA: [0, 0, 1] });
  w.step(DT);
  let ab = 0, cd = 0;
  for (let i = 0; i < w.contacts.count; i++) {
    const k = pairKey(w.contacts.a[i], w.contacts.b[i]);
    if (k === pairKey(a, b)) ab++; else if (k === pairKey(c, d)) cd++;
  }
  if (typeof w.joints.connected === 'function') check('joints: a hinge-joined overlapping pair has no contacts', ab, 0, 0);
  else note('joints: JointStore.connected not available yet: collideConnected not checked');
  check('joints: an unjointed overlapping pair has contacts', cd, 1, 8);
}

// --- 14. no allocation in step(): retained heap over 500 steps after warm-up, and the bytes allocated per step
//         (total, and those V8's sampling heap profiler attributes to DynamicsWorld.js / islands.js frames) ----------
{
  let gc = globalThis.gc;
  if (!gc) { try { v8.setFlagsFromString('--expose-gc'); gc = vm.runInNewContext('gc'); } catch { gc = null; } }
  const exact = v8.getHeapStatistics().total_allocated_bytes !== undefined;   // Node >= 26
  const allocated = exact ? () => v8.getHeapStatistics().total_allocated_bytes : () => process.memoryUsage().heapUsed;
  const physics = new PhysicsWorld({ groundLevel: 0 });
  physics.addStaticBox(-4, 0, -3, 2, -4, 4); physics.addStaticBox(3, 0, 4, 2, -4, 4);
  const w = new DynamicsWorld({ physics, sleep: false, maxBodies: 256 });
  const rnd = mulberry32(11);
  for (let i = 0; i < 80; i++) w.addBody({ shape: i % 3 ? box(0.2, 0.2, 0.2) : sphere(0.2), position: [(rnd() - 0.5) * 5, 0.3 + i * 0.25, (rnd() - 0.5) * 5] });
  const plat = w.addBody({ type: 'kinematic', shape: box(1, 0.1, 1), position: [0, 0.1, 0] });
  const c0 = w.addBody({ shape: sphere(0.15), position: [0, 4, 6] });
  w.addJoint({ type: JOINT.BALL, bodyA: c0, bodyB: WORLD_BODY, anchorA: [0.6, 0, 0] });
  let i = 0;
  const frame = () => { i++; const x = Math.sin(i * 0.01) * 2; w.setKinematicTarget(plat, x, 0.1, 0, 0, 0, 0, 1); w.step(DT); };
  for (let k = 0; k < 300; k++) frame();   // warm-up (JIT, pools, event sets)
  if (gc) gc();
  const h0 = process.memoryUsage().heapUsed;
  // bytes per step in blocks of 50 steps. A block's figure mixes real per-step garbage with V8 work that is not the
  // engine's: HeapNumber boxing in code V8 has not (re)optimised yet (the joint tree solver and the contact rows box a
  // few hundred bytes per step until then) and the code / feedback objects of functions it compiles in that block. Both
  // come and go: blocks read 0.2 - 60 KB per step on the M4, and on a shared 2-core CI runner, where the background
  // compiler is slower, the quietest of a fixed first 10 blocks once read 3.4 KB. Under load the quietest block
  // settles at a floor of ~2 KB on the M4 too, while the sampling heap profiler puts only a few hundred bytes per step
  // in the engine's own frames (the attribution check below): the floor is V8's, and the old 2 KB budget sat on it.
  // So the budget is 6 KB (about 2x the worst runner reading) and at least 10 blocks run, then more, up to MAX_BLOCKS,
  // until one is under it. Real garbage (an object or array per contact or per body and substep: ~10 KB per step and
  // up) is in every block and still fails: the control below allocates one array per contact and reads ~21 KB.
  const BUDGET = 6144, MIN_BLOCKS = 10, MAX_BLOCKS = 60;
  const blocks = [];
  let per = Infinity;
  for (let q = 0; q < MAX_BLOCKS && (q < MIN_BLOCKS || per > BUDGET); q++) {
    const b0 = allocated();
    for (let k = 0; k < 50; k++) frame();
    const v = (allocated() - b0) / 50;
    blocks.push(v);
    if (v < per) per = v;
  }
  if (gc) gc();
  const h1 = process.memoryUsage().heapUsed;
  check(`alloc: retained heap growth over ${blocks.length * 50} steps (gc before/after)`, (h1 - h0) / 1024, -Infinity, 64, 'KB');
  blocks.sort((x, y) => x - y);
  // the exact counter (Node >= 26) only: a heapUsed delta drops by whatever a GC inside the block frees, so a block
  // can read hundreds of KB below zero and the minimum means nothing (Node 22: -163 KB per step)
  if (exact) check(`alloc: bytes allocated per step in total, quietest of ${blocks.length} 50-step blocks (timing off)`, per, 0, BUDGET, 'B');
  else note('alloc: bytes per step not checked: total_allocated_bytes needs Node >= 26 (heapUsed deltas include GC)');
  // attribution: sample every allocation of 200 more steps, sum the bytes allocated in this world's own frames
  let own = NaN, top = '';
  try {
    const inspector = await import('node:inspector/promises');
    const s = new inspector.Session(); s.connect();
    await s.post('HeapProfiler.enable');
    await s.post('HeapProfiler.startSampling', { samplingInterval: 64, includeObjectsCollectedByMajorGC: true, includeObjectsCollectedByMinorGC: true });
    for (let k = 0; k < 200; k++) frame();
    const { profile } = await s.post('HeapProfiler.stopSampling');
    s.disconnect();
    const by = new Map(), fns = new Map();
    const walk = (nd) => {
      const file = nd.callFrame.url.split('/').pop() || '(native)';
      if (nd.selfSize) {
        by.set(file, (by.get(file) || 0) + nd.selfSize);
        if (file === 'DynamicsWorld.js' || file === 'islands.js') fns.set(nd.callFrame.functionName, (fns.get(nd.callFrame.functionName) || 0) + nd.selfSize);
      }
      for (const c of nd.children) walk(c);
    };
    walk(profile.head);
    own = ((by.get('DynamicsWorld.js') || 0) + (by.get('islands.js') || 0)) / 200;
    top = [...by.entries()].sort((x, y) => y[1] - x[1]).slice(0, 4).map(([f, b]) => `${f} ${(b / 200).toFixed(0)}`).join(', ');
    if (own > 64) top += '; own: ' + [...fns.entries()].sort((x, y) => y[1] - x[1]).map(([f, b]) => `${f} ${(b / 200).toFixed(0)}`).join(', ');
  } catch { /* no inspector */ }
  // the sampling profiler's per-file attribution of HeapNumber boxing varies with the V8 build (Node 26 arm64 ~130 B,
  // Node 22 x64 ~1100 B): a loose bound catches real per-step garbage (arrays, closures: tens of KB), not boxing
  if (Number.isFinite(own)) check('alloc: bytes per step allocated in DynamicsWorld.js / islands.js', own, 0, 4096, 'B');
  // control: the same world plus one 3-element array per contact per step (what a regression that builds a temporary
  // vector per contact allocates, ~70 B x ~290 contacts) never yields a block under the budget
  if (exact) {
    let ctl = Infinity;
    for (let q = 0; q < MIN_BLOCKS; q++) {
      const b0 = allocated();
      for (let k = 0; k < 50; k++) { frame(); for (let c = 0; c < w.contacts.count; c++) _allocSink = [c, c + 1, c + 2]; }
      const v = (allocated() - b0) / 50;
      if (v < ctl) ctl = v;
    }
    check(`alloc: control: an array per contact per step is detected (quietest of ${MIN_BLOCKS} blocks)`, ctl, BUDGET, Infinity, 'B');
  }
  note(`alloc: ${per.toFixed(0)} B allocated per step in total (median block ${blocks[blocks.length >> 1].toFixed(0)}, max ${blocks[blocks.length - 1].toFixed(0)}, ${blocks.length} blocks)` +
    `${exact ? '' : ' (heapUsed delta, approximate: Node < 26)'} with ${w.stats.contacts} contacts, ${w.stats.awake} awake, ` +
    `steps 300..${300 + blocks.length * 50}; sampled by file (B/step): ${top || 'n/a'}. ` +
    'A fresh world allocates ~30 KB/step (V8 boxing in narrowphase.flush -> ContactBuffer.add) until V8 optimises flush');
  if (!gc) note('no gc available: retained heap growth measured without gc');
}

// --- 15. rolling: a sphere and a lying capsule rolling on flat ground keep (not gain) their spin ---------------------
// (0.9.0 fix: curved shapes re-seat their contact anchor every pass; 0.6.0 - 0.8.0 carried it along as a material point
// and every normal impulse spun the body up: |w| 0.6 -> 3.4 (sphere r 0.3) / 44.6 (capsule r 0.05) in 5 s at mu 0)
{
  const S2 = Math.SQRT1_2;
  const roll = (shape, q, r, mu) => {
    const w = new DynamicsWorld({ physics: new PhysicsWorld({ groundLevel: 0 }), sleep: false });
    const id = w.addBody({ shape, position: [0, r, 0], quaternion: q, friction: mu, mass: 1 });
    w.setVelocity(id, 0, 0, 0.6 * r, 0.6, 0, 0);       // rolling about +x along +z: v = w r
    const W = w.bodies.angVel, V = w.bodies.vel;
    let prev = 0.6, inc = 0;
    run(w, 5, () => { const m = Math.hypot(W[id * 3], W[id * 3 + 1], W[id * 3 + 2]); if (m - prev > inc) inc = m - prev; prev = m; });
    const wm = Math.hypot(W[id * 3], W[id * 3 + 1], W[id * 3 + 2]), vm = Math.hypot(V[id * 3], V[id * 3 + 1], V[id * 3 + 2]);
    return { inc, wm, ratio: vm / (wm * r) };
  };
  for (const [label, shape, q, r] of [['sphere r 0.3', sphere(0.3), null, 0.3], ['lying capsule r 0.05', capsule(0.05, 0.2), [0, 0, -S2, S2], 0.05]]) {
    const f = roll(shape, q, r, 0), g = roll(shape, q, r, 0.6);
    check(`rolling ${label}, mu 0: |w| never increases (max step increase)`, f.inc, 0, 0, 'rad/s');
    check(`rolling ${label}, mu 0: |w| after 5 s (angular damping only)`, f.wm, 0.46, 0.48, 'rad/s');
    check(`rolling ${label}, mu 0.6: |w| never increases (max step increase)`, g.inc, 0, 0, 'rad/s');
    check(`rolling ${label}, mu 0.6: rolls without slip (|v| / (|w| r) after 5 s)`, g.ratio, 1 - 1e-3, 1 + 1e-3);
  }
  // a sphere rolling down a 10 deg slope (a static rotated box) accelerates at 5/7 g sin(theta) (0.8.0: 1.316 m/s^2)
  const th = 10 * Math.PI / 180, c = Math.cos(th), sn = Math.sin(th), r = 0.3;
  const ws = new DynamicsWorld({ sleep: false });
  ws.addBody({ type: 'static', shape: box(5, 0.5, 2), quaternion: [0, 0, Math.sin(th / 2), Math.cos(th / 2)], friction: 0.6 });
  const sp = ws.addBody({ shape: sphere(r), position: [2 * c - 0.5 * sn - r * sn, 2 * sn + 0.5 * c + r * c, 0], friction: 0.6, linearDamping: 0, angularDamping: 0 });
  const V = ws.bodies.vel, along = () => -(V[sp * 3] * c + V[sp * 3 + 1] * sn);   // down-slope speed
  let v1 = 0, v2 = 0;
  run(ws, 1.5, (i) => { if (i === 60) v1 = along(); if (i === 180 - 1) v2 = along(); });
  const acc = (v2 - v1) / (119 * DT), aRef = 5 / 7 * 9.81 * sn;
  check('rolling sphere down a 10 deg slope: a / (5/7 g sin theta)', acc / aRef, 1 - 1e-3, 1 + 1e-3);
  // a capsule pivoting across a static box edge: a cylinder contact keeps its axial coordinate through the step's
  // re-seats (re-seating along the full normal marched it toward a cap: 1.9 cm per step here)
  const wc = new DynamicsWorld({ sleep: false });
  const cap = wc.addBody({ shape: capsule(0.05, 0.3), position: [0.6, 1.05, 0], quaternion: [0, 0, -Math.SQRT1_2, Math.SQRT1_2] });
  wc.addBody({ type: 'static', shape: box(0.5, 0.5, 0.5), position: [0, 0.5, 0] });
  const y0 = new Float64Array(64);
  let drift = 0, nI = 0, seen = 0;
  const axialOf = (world, i) => (world.contacts.a[i] === cap ? world._la[i * 3 + 1] : world.contacts.b[i] === cap ? world._lb[i * 3 + 1] : NaN);
  wc.addParticleSolver({
    name: 'axial-probe',
    beginStep(world) { nI = Math.min(64, world.contacts.count); for (let i = 0; i < nI; i++) y0[i] = axialOf(world, i); },
    endStep(world) {
      for (let i = 0; i < nI; i++) if (Number.isFinite(y0[i]) && Math.abs(y0[i]) < 0.3 - 1e-9) { seen++; drift = Math.max(drift, Math.abs(axialOf(world, i) - y0[i])); }
    },
  });
  run(wc, 0.5);
  check('capsule pivoting on a box edge: cylinder contacts seen', seen, 1, Infinity);
  check('capsule pivoting on a box edge: axial drift of a contact anchor over a step', drift, 0, 1e-6, 'm');
  note(`capsule pivoting on a box edge: wz after 0.5 s ${wc.bodies.angVel[cap * 3 + 2].toFixed(3)} rad/s (previous re-seat: -4.9)`);
}

console.log(results.join('\n'));
console.log(failures ? `\n${failures} check(s) FAILED` : '\nAll dyn-world checks passed.');
process.exit(failures ? 1 : 0);
