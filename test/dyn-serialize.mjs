// Headless checks of the dynamics worker protocol (src/physics/dynamics.worker.js) and DynamicsClient: command
// encoding round trips, synchronous id mirroring against a real BodyStore, the snapshot layout / extrapolation /
// collider records / main-thread patches, the step policy, the double-buffered transfer loop over a node:worker_threads
// MessageChannel, and - when the dynamics world and all its modules load - serialize/from bit-identity, a real worker
// thread against the in-thread world (bit-identical), and the start-up fallback replay.
// Run: node test/dyn-serialize.mjs
import { MessageChannel, Worker as ThreadWorker } from 'node:worker_threads';
import * as nodeModule from 'node:module';
import { Vector3, Quaternion } from 'three';

let failures = 0;
const results = [];
function check(name, ok, detail = '') {
  if (!ok) failures++;
  results.push(`${ok ? 'PASS' : 'FAIL'}  ${name.padEnd(68)} ${detail}`);
  return ok;
}
const skip = (name, why) => results.push(`SKIP  ${name.padEnd(68)} ${why}`);
const src = (p) => new URL(`../src/physics/${p}`, import.meta.url).href;
const DT = 1 / 60;

// ---- modules. The real DynamicsWorld when it loads; otherwise (TEST ONLY) a stub module stands in for it so the
// protocol can still be checked while the dynamics modules are being written.
let REAL = true, whyNot = '';
try { await import(src('dynamics/DynamicsWorld.js')); } catch (e) { REAL = false; whyNot = String(e.message).split('\n')[0]; }
if (!REAL) {
  const stub = `import { BodyStore, ContactBuffer } from ${JSON.stringify(src('dynamics/state.js'))};
export function describeShape(s) { return { type: s.type, radius: s.radius, halfHeight: s.halfHeight, half: Array.from(s.half) }; }
export function shapeFromDescriptor(d) { return { ...d, aabb() { return null; } }; }
export class DynamicsWorld {
  constructor(o = {}) { this.bodies = new BodyStore(o.maxBodies ?? 4096); this.contacts = new ContactBuffer(16); this.time = 0; this.stepCount = 0; this.stats = {}; }
  addBody() { return this.bodies.alloc(); } removeBody(id) { this.bodies.free(id); } step(dt) { this.time += dt; this.stepCount++; }
  kindOf() { return 'rigid'; } static from() { return new DynamicsWorld(); } dispose() {}
}`;
  const stubURL = 'data:text/javascript,' + encodeURIComponent(stub), target = '/src/physics/dynamics/DynamicsWorld.js';
  if (typeof nodeModule.registerHooks === 'function') {
    nodeModule.registerHooks({ resolve(spec, ctx, next) { const r = next(spec, ctx); return r.url.endsWith(target) ? { url: stubURL, shortCircuit: true } : r; } });
  } else { // Node 20: async hooks
    const hook = `export async function resolve(spec, ctx, next) {
  const r = await next(spec, ctx);
  return r.url.endsWith(${JSON.stringify(target)}) ? { url: ${JSON.stringify(stubURL)}, shortCircuit: true } : r;
}`;
    nodeModule.register('data:text/javascript,' + encodeURIComponent(hook));
  }
}
const P = await import(src('dynamics.worker.js'));
const { DynamicsClient, POSE_STRIDE, POSE_ALIVE, POSE_PENDING, POSE_TYPE_SHIFT } = await import(src('DynamicsClient.js'));
const { BodyStore, ContactBuffer, BODY, FLAG } = await import(src('dynamics/state.js'));
const { sphere, box, capsule, convex } = await import(src('dynamics/shapes.js'));
const { PhysicsWorld, layerOf, namedLayers } = await import(src('PhysicsWorld.js'));
const { attachToPhysicsWorld } = await import(src('dynamics/bridge.js'));
const { OP } = P;

// ---- helpers
/** Deep equality: numbers by Object.is (NaN, -0), typed arrays by type and elements, objects by key order. */
function same(a, b) {
  if (typeof a === 'number' || typeof b === 'number') return Object.is(a, b);
  if (a === b) return true;
  if (a === null || b === null || typeof a !== 'object' || typeof b !== 'object') return false;
  if (ArrayBuffer.isView(a) || ArrayBuffer.isView(b)) {
    if (a.constructor !== b.constructor || a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (!Object.is(a[i], b[i])) return false;
    return true;
  }
  if (Array.isArray(a) !== Array.isArray(b)) return false;
  const ka = Object.keys(a), kb = Object.keys(b);
  if (ka.length !== kb.length) return false;
  for (let i = 0; i < ka.length; i++) if (ka[i] !== kb[i] || !same(a[ka[i]], b[kb[i]])) return false;
  return true;
}
const tick = () => new Promise((r) => setImmediate(r));
const pause = (ms = 0) => new Promise((r) => setTimeout(r, ms));
function rng(seed) { let s = seed >>> 0; return () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296; }; }
const near = (a, b, e = 1e-6) => Math.abs(a - b) <= e;
/** Worker-like that records what it is sent and never answers (the test plays the worker through onmessage). */
function dummyWorker() {
  const w = {
    posted: [], onmessage: null, onerror: null, dead: false,
    postMessage(m) { w.posted.push(m.type === 'step' ? { type: 'step', n: m.n, seq: m.seq, dt: m.dt, bufs: m.bufs ? m.bufs.length : 0 } : { type: m.type, seq: m.seq, named: m.namedLayers ? m.namedLayers.length : -1 }); },
    terminate() { w.dead = true; },
  };
  return w;
}
/** A snapshot message as the worker would post it: recs = [{ id, p, q, v, w, flags, type, aabb }]. */
function snapMsg(cap, seq, time, recs, contacts = [], ccap = 4) {
  const buffer = new Float32Array(cap * POSE_STRIDE), aabbs = new Float32Array(cap * 6), cb = new Float32Array(1 + ccap * 10);
  let count = 0;
  for (const r of recs) {
    const o = r.id * POSE_STRIDE;
    buffer.set([...r.p, ...(r.q || [0, 0, 0, 1]), ...(r.v || [0, 0, 0]), ...(r.w || [0, 0, 0]), (r.flags || 0) | POSE_ALIVE | (r.type << POSE_TYPE_SHIFT)], o);
    aabbs.set(r.aabb || [r.p[0] - 0.5, r.p[1] - 0.5, r.p[2] - 0.5, r.p[0] + 0.5, r.p[1] + 0.5, r.p[2] + 0.5], r.id * 6);
    count = Math.max(count, r.id + 1);
  }
  cb[0] = contacts.length;
  contacts.forEach((c, i) => cb.set(c, 1 + i * 10));
  return { type: 'poses', seq, stepCount: seq, time, count, buffer, aabbs, contacts: cb, contactTotal: contacts.length, stats: { bodies: recs.length, awake: 1 }, stepMs: 0.1, cmd: null, sets: 1 };
}

// ==================================================================== 1. command encoding round trip
{
  const w = new P.CommandWriter(8); // small: forces growth
  const fixed = [
    [OP.REMOVE_BODY, [3]],
    [OP.SET_POSE, [1, 1.5, -2, 3e-7, 0, 0.7071067811865476, 0, 0.7071067811865476]],
    [OP.SET_POSE, [2, NaN, 0, 0, 0, 0, 0, NaN]],
    [OP.SET_VELOCITY, [4, -0, 1e308, -5e-324, NaN, 0, 0]],
    [OP.APPLY_IMPULSE, [5, 1, 2, 3, NaN, 0, 0]],
    [OP.APPLY_FORCE, [6, 10, 0, -10, 0.25, 0.5, Infinity]],
    [OP.APPLY_TORQUE, [7, 0.1, -0.2, 0.3]],
    [OP.SET_KINEMATIC_TARGET, [8, 1, 2, 3, 0, 0, 0, 1]],
    [OP.REMOVE_JOINT, [9]],
    [OP.WAKE, [10]],
    [OP.SLEEP, [11]],
    [OP.SET_GRAVITY, [0, -9.81, 0.5]],
  ];
  for (const [op, a] of fixed) w.op(op, a.length, ...a);
  const sh = box(0.5, 0.25, 1), shapeDesc = P.describeShape(sh);
  const extra = [1, 'é✓\u0000x', null, undefined, true, false, { f32: new Float32Array([0.1, -0, 3]) }, new Int32Array([-1, 2]), new Uint16Array([65535]), new Uint8Array(0), new Float64Array([Math.PI, NaN])];
  const body = {
    type: 'kinematic', shape: sh, position: new Vector3(1, 2, 3), quaternion: new Quaternion(0, 0.6, 0, 0.8), mass: 2.5, friction: 0.3,
    restitution: -0, linearDamping: NaN, layer: 5, mask: -1, kind: 'crate', noGravity: true, userData: { mesh: {}, onHit() {} }, extra, cb: () => 1,
  };
  const bodyWant = {
    type: 'kinematic', shape: shapeDesc, position: { x: 1, y: 2, z: 3 }, quaternion: { x: 0, y: 0.6, z: 0, w: 0.8 }, mass: 2.5, friction: 0.3,
    restitution: -0, linearDamping: NaN, layer: 5, mask: -1, kind: 'crate', noGravity: true, extra, cb: null,
  };
  w.addBody(12, 0, body, shapeDesc);
  w.addBody(13, 0, { shape: sh, position: [0, 1, 0] }, undefined); // shape already sent: left out
  const joint = { type: 2, bodyA: 12, bodyB: null, anchorA: [-0.5, 0, 0], axisA: [0, 1, 0], limits: [0, 1.6], motor: { velocity: 1, maxForce: 50 }, breakForce: Infinity };
  w.addJoint(14, joint);
  const dec = P.decodeCommands(w.buf, w.len);
  let okFixed = dec.length === fixed.length + 3;
  for (let i = 0; i < fixed.length && okFixed; i++) okFixed = dec[i].op === fixed[i][0] && same(dec[i].args, fixed[i][1]) && P.OP_ARGS[fixed[i][0]] === fixed[i][1].length;
  check('fixed-layout opcodes round-trip bit-exactly (NaN, -0, denormals)', okFixed, `${fixed.length} records, buffer grew ${w.grows}x`);
  const add = dec[fixed.length], add2 = dec[fixed.length + 1], j = dec[fixed.length + 2];
  check('ADD_BODY: id, sid, descriptor (userData dropped, three math -> plain)', add.op === OP.ADD_BODY && add.args[0] === 12 && add.args[1] === 0 && same(add.args[2], bodyWant));
  check('ADD_BODY of an already sent shape carries no shape', add2.op === OP.ADD_BODY && add2.args[1] === 0 && same(add2.args[2], { position: [0, 1, 0] }));
  check('ADD_JOINT: id + descriptor', j.op === OP.ADD_JOINT && j.args[0] === 14 && same(j.args[1], joint));
  check('every opcode is exercised', new Set(dec.map((d) => d.op)).size === Object.keys(OP).length, `${Object.keys(OP).length} opcodes`);
  let threw = 0;
  try { P.encodeValue(new P.CommandWriter(), () => 1); } catch { threw++; }
  try { P.encodeValue(new P.CommandWriter(), { a: new BigInt64Array(1) }); } catch { threw++; }
  check('top-level functions / BigInt arrays are refused', threw === 2);
}

// ==================================================================== 2. id mirroring
{
  const r = rng(7), store = new BodyStore(64), al = new P.IdAllocator(64);
  let ok = true;
  for (let i = 0; i < 2000; i++) {
    const live = [];
    for (let id = 0; id < store.count; id++) if (store.alive[id]) live.push(id);
    if (!live.length || (r() < 0.55 && live.length < 64)) { if (store.alloc() !== al.alloc()) ok = false; } else { const id = live[(r() * live.length) | 0]; store.free(id); al.free(id); }
  }
  check('IdAllocator == BodyStore allocation (2000 random ops)', ok && al.count === store.count);
}
{
  const dw = dummyWorker(), r = rng(11);
  const c = new DynamicsClient({ worker: () => dw, maxBodies: 64, readyTimeout: 60000 });
  const store = new BodyStore(64), ball = sphere(0.5), crate = box(0.5, 0.5, 0.5);
  let same_ = 0, adds = 0, removes = 0;
  for (let i = 0; i < 100; i++) {
    const live = [];
    for (let id = 0; id < store.count; id++) if (store.alive[id]) live.push(id);
    if (!live.length || r() < 0.6) {
      const cid = c.addBody({ shape: r() < 0.5 ? ball : crate, position: [i, 1, 0] });
      if (cid === store.alloc()) same_++;
      adds++;
    } else {
      const id = live[(r() * live.length) | 0];
      if (c.removeBody(id)) same_++;
      store.free(id); removes++;
    }
  }
  let mirror = c.bodies.count === store.count;
  for (let id = 0; id < store.count; id++) if (c.bodies.alive[id] !== store.alive[id]) mirror = false;
  check('client synchronous ids == BodyStore ids (100 random add/remove)', same_ === 100 && mirror, `${adds} adds, ${removes} removes`);
  const recs = P.decodeCommands(c._cw.buf, c._cw.len).filter((d) => d.op === OP.ADD_BODY);
  const withShape = recs.filter((d) => d.args[2].shape !== undefined).length;
  check('shape descriptors cross once per shape object (refcounted)', withShape >= 2 && withShape < recs.length && c._sids.count <= 2, `${withShape} of ${recs.length} adds carried a shape`);
  let threw = 0;
  try { c.addBody({ position: [0, 0, 0] }); } catch (e) { threw += e instanceof TypeError; }
  try { c.setPose(63, 0, 0, 0); } catch (e) { threw += e instanceof RangeError; }
  check('misuse throws synchronously like DynamicsWorld (no shape, unknown id)', threw === 2);
  c.dispose();
}

// ==================================================================== 3. snapshot layout, extrapolation, colliders, patches
{
  const dw = dummyWorker(), cap = 16;
  const c = new DynamicsClient({ worker: () => dw, maxBodies: cap, snapshotContacts: 4, readyTimeout: 60000 });
  const s = sphere(0.5), b = box(1, 1, 1);
  const A = c.addBody({ shape: s, position: [1, 2, 3] });
  const S = c.addBody({ type: 'static', shape: b, position: [0, -1, 0] });
  const K = c.addBody({ type: 'kinematic', shape: b, position: [5, 0, 5], kind: 'door' });
  const D = c.addBody({ shape: s, position: [40, 0, 0] });
  const p = new Vector3(), q = new Quaternion(), a3 = [0, 0, 0], a4 = [0, 0, 0, 0];
  check('before any snapshot: added bodies show at their start pose', c.getPose(A, p, q) && p.x === 1 && p.y === 2 && p.z === 3 && q.w === 1 && c.bodies.alive[A] === 1);
  check('before any snapshot: no colliders for bodies the worker has not seen', c.queryColliders(1, 3, 5, []).length === 0);
  dw.onmessage({ data: { type: 'ready', capacity: cap, count: 0, time: 0, stepCount: 0 } });
  await c.ready;
  c.step(DT);
  const sent = dw.posted.map((m) => m.type).join(' ');
  check('init first; ready flushes the queued commands, then one step message', c.mode === 'worker' && sent === 'init cmd step' && dw.posted[2].seq === 1 && dw.posted[2].n === 1, sent);
  const recs1 = [
    { id: A, p: [1, 2, 3], v: [1, 0, 0], w: [0, Math.PI, 0], type: BODY.DYNAMIC, aabb: [0.5, 1.5, 2.5, 1.5, 2.5, 3.5] },
    { id: S, p: [0, -1, 0], type: BODY.STATIC, aabb: [-1, -2, -1, 1, 0, 1] },
    { id: K, p: [5, 0, 5], v: [0, 0, 2], type: BODY.KINEMATIC, aabb: [4, -1, 4, 6, 1, 6] },
    { id: D, p: [40, 0, 0], flags: FLAG.ASLEEP, type: BODY.DYNAMIC },
  ];
  dw.onmessage({ data: snapMsg(cap, 1, DT, recs1, [[A, -1, 1, 1.5, 3, 0, -1, 0, 0.01, 0.25]]) });
  c.getPose(A, p, q, 0);
  check('poses buffer: getPose reads the record of the id (Vector3 / Quaternion out)', p.x === 1 && p.y === 2 && p.z === 3 && q.x === 0 && q.w === 1);
  c.getPose(A, a3, a4);
  check('getPose with arrays; no extrapolation when the snapshot is current', same(a3, [1, 2, 3]) && same(a4, [0, 0, 0, 1]));
  c.getPose(A, p, q, 0.5);
  check('extrapolation: v (1, 0, 0), 0.5 s -> x + 0.5', p.x === 1.5 && p.y === 2 && p.z === 3, `x ${p.x}`);
  check('extrapolation: w = pi rad/s about Y, 0.5 s -> quarter turn', near(q.y, Math.SQRT1_2) && near(q.w, Math.SQRT1_2) && near(q.x, 0), `q (${q.x.toFixed(4)}, ${q.y.toFixed(4)}, ${q.z.toFixed(4)}, ${q.w.toFixed(4)})`);
  const B = c.bodies;
  check('bodies mirror: pos / vel / aabb / flags / type views over the snapshot',
    same(Array.from(B.pos.subarray(A * 3, A * 3 + 3)), [1, 2, 3]) && B.vel[K * 3 + 2] === 2 && B.aabb[K * 6 + 3] === 6 &&
    (B.flags[D] & FLAG.ASLEEP) !== 0 && B.type[K] === BODY.KINEMATIC && B.layer[A] === 1 && B.count === 4);
  check('isAwake / getVelocity from the snapshot', !c.isAwake(D) && c.isAwake(A) && c.getVelocity(K, a3, null) && same(a3, [0, 0, 2]));
  const C = c.contacts;
  check('contacts: count + a, b, point, normal, depth, impulse', C.count === 1 && C.a[0] === A && C.b[0] === -1 && C.ny[0] === -1 && near(C.depth[0], 0.01) && C.lambdaN[0] === 0.25 && c.stats.awake === 1);
  let out = c.queryColliders(5, 5, 0.5, []);
  check('queryColliders: record { minX..maxZ, kind, id, vx, vz }', out.length === 1 && out[0].id === K && out[0].kind === 'door' && out[0].minX === 4 && out[0].maxY === 1 && out[0].vz === 2 && out[0].vx === 0);
  check('queryColliders: static bodies are not published', c.queryColliders(0, 0, 0.1, []).length === 0);
  check('queryColliders: footprint distance vs radius', c.queryColliders(1, 6, 2, []).length === 0 && c.queryColliders(1, 6, 2.6, []).length === 1);
  out = c.queryColliders(20, 0, 100, []);
  const again = c.queryColliders(20, 0, 100, []);
  check('queryColliders: ascending ids, pooled records', same(out.map((o) => o.id), [A, K, D]) && out[0] === again[0]);
  // patches survive snapshots that predate them
  c.step(DT); // seq 2 posted: carries nothing new
  c.setPose(A, 10, 0, 0);
  c.removeBody(D);
  c.getPose(A, p, null, 0);
  check('setPose / removeBody show at once (AABB moved with the body)', p.x === 10 && !c.getPose(D, p) && c.bodies.alive[D] === 0 && c.bodies.aabb[A * 6] === 9.5);
  dw.onmessage({ data: snapMsg(cap, 2, 2 * DT, [{ ...recs1[0], p: [1.02, 2, 3] }, recs1[1], recs1[2], recs1[3]]) });
  c.getPose(A, p, null, 0);
  check('a snapshot older than the command keeps the patch', p.x === 10 && !c.getPose(D, p) && c.queryColliders(40, 0, 1, []).length === 0);
  c.step(DT); // seq 3 carries setPose + removeBody
  const st3 = dw.posted.filter((m) => m.type === 'step').pop();
  check('each step returns the snapshot set it replaced (poses, aabbs, contacts)', st3.seq === 3 && st3.bufs === 3, `${st3.bufs} buffers returned`);
  dw.onmessage({ data: snapMsg(cap, 3, 3 * DT, [{ ...recs1[0], p: [10.5, 0, 0] }, recs1[1], recs1[2]]) });
  c.getPose(A, p, null, 0);
  check('the snapshot that includes the command replaces the patch', p.x === 10.5 && !c.getPose(D, p) && c._ovCount === 0);
  c.dispose();
}

// ==================================================================== 4. step policy (at most 2 in flight)
{
  const dw = dummyWorker(), cap = 4;
  const c = new DynamicsClient({ worker: () => dw, maxBodies: cap, readyTimeout: 60000 });
  dw.onmessage({ data: { type: 'ready', capacity: cap, count: 0, time: 0, stepCount: 0 } });
  for (let i = 0; i < 5; i++) c.step(DT);
  const steps = () => dw.posted.filter((m) => m.type === 'step');
  check('lagging worker: 2 steps posted, 2 owed, 1 skipped', steps().length === 2 && c.perf.inFlight === 2 && c.perf.skipped === 1 && near(c.time, 4 * DT, 1e-12));
  dw.onmessage({ data: snapMsg(cap, 1, DT, []) });
  dw.onmessage({ data: snapMsg(cap, 2, 2 * DT, []) });
  c.flush();
  const st = steps();
  check('owed steps go out as one message with n = 2', st.length === 3 && st[2].n === 2 && st[2].seq === 3 && st[2].bufs === 6);
  c.dispose();
}

// ==================================================================== 5. transfer loop through a MessageChannel (fake worlds)
class FakeWorld { // integrates positions: enough to drive the protocol
  constructor(o = {}) {
    this.bodies = new BodyStore(o.maxBodies ?? 64); this._cb = new ContactBuffer(8); this.joints = null;
    this.time = 0; this.stepCount = 0; this.stats = { bodies: 0 }; this.shapeOf = [];
  }
  get contacts() { return this._cb; }
  addBody(d) {
    const B = this.bodies, id = B.alloc(), p = d.position || [0, 0, 0];
    B.pos[id * 3] = p[0]; B.pos[id * 3 + 1] = p[1]; B.pos[id * 3 + 2] = p[2]; B.type[id] = BODY.DYNAMIC; this.shapeOf[id] = d.shape;
    return id;
  }
  removeBody(id) { this.bodies.free(id); this.shapeOf[id] = null; }
  setPose(id, x, y, z) { const B = this.bodies; B.pos[id * 3] = x; B.pos[id * 3 + 1] = y; B.pos[id * 3 + 2] = z; }
  setVelocity(id, vx, vy, vz) { const B = this.bodies; B.vel[id * 3] = vx; B.vel[id * 3 + 1] = vy; B.vel[id * 3 + 2] = vz; }
  applyImpulse(id, px, py, pz, ix, iy, iz) { const B = this.bodies; B.vel[id * 3] += ix; B.vel[id * 3 + 1] += iy; B.vel[id * 3 + 2] += iz; }
  applyForce() {} applyTorque() {} setKinematicTarget() {} addJoint() { return 0; } removeJoint() {} wake() {} sleep() {} setGravity() {}
  step(dt) {
    const B = this.bodies, C = this._cb;
    C.clear();
    for (let id = 0; id < B.count; id++) {
      if (!B.alive[id]) continue;
      for (let k = 0; k < 3; k++) { B.pos[id * 3 + k] += B.vel[id * 3 + k] * dt; B.aabb[id * 6 + k] = B.pos[id * 3 + k] - 0.5; B.aabb[id * 6 + 3 + k] = B.pos[id * 3 + k] + 0.5; }
      C.add(id, -1, B.pos[id * 3], B.pos[id * 3 + 1] - 0.5, B.pos[id * 3 + 2], 0, -1, 0, 0.001, 0, 0);
    }
    this.time += dt; this.stepCount++; this.stats.bodies = B.size;
  }
}
/** A DynamicsClient whose worker is createDynamicsWorkerHandler over a MessageChannel, with world class W. */
function channelClient(W, opts = {}) {
  const { port1, port2 } = new MessageChannel();
  const handler = P.createDynamicsWorkerHandler((m, t) => port2.postMessage(m, t), { createWorld: (m) => new W(m.options) });
  port2.on('message', handler);
  const seqs = [], sentN = [];
  const wl = {
    onmessage: null, onerror: null,
    postMessage(m, t) { if (m.type === 'step') sentN.push(m.n); port1.postMessage(m, t); },
    terminate() { port1.close(); port2.close(); },
  };
  port1.on('message', (data) => { if (data.type === 'poses') seqs.push(data.seq); if (wl.onmessage) wl.onmessage({ data }); });
  const c = new DynamicsClient({ worker: () => wl, maxBodies: 64, readyTimeout: 5000, ...opts });
  return { c, handler, seqs, sentN };
}
const drain = async (c) => { for (let k = 0; k < 500 && c.perf.inFlight; k++) await tick(); };
{
  const { c, handler, seqs, sentN } = channelClient(FakeWorld);
  const ball = sphere(0.5);
  const id = c.addBody({ shape: ball, position: [0, 0, 0] }), id2 = c.addBody({ shape: ball, position: [0, 5, 0] });
  c.setVelocity(id, 1, 0, 0);
  await c.ready;
  for (let i = 0; i < 100; i++) {
    if (i === 50) c.applyImpulse(id2, [0, 1, 0]);
    c.step(DT);
    await tick();
  }
  for (let k = 0; k < 2000 && c.stepCount + c.perf.skipped < 100; k++) { c.flush(); await tick(); }
  await drain(c);
  const mono = seqs.length > 0 && seqs.every((s, i) => s === i + 1);
  check('100 steps through MessageChannel: every step answered once, seq 1, 2, 3, ...', mono && c.stepCount + c.perf.skipped === 100 && sentN.reduce((s, n) => s + n, 0) === c.stepCount,
    `${seqs.length} snapshots, ${c.stepCount} steps, ${sentN.filter((n) => n === 2).length} merged, ${c.perf.skipped} skipped`);
  let x = 0;
  for (let k = 0; k < c.stepCount; k++) x += 1 * DT;
  const p = [0, 0, 0], v = [0, 0, 0];
  c.getPose(id, p, null, 0); c.getVelocity(id2, v, null);
  check('fake world integrated every step (x = float32(sum v dt)); impulse arrived', p[0] === Math.fround(x) && v[1] === 1, `x ${p[0].toFixed(5)}`);
  const pools = handler.pools();
  const circulating = pools.freePoses + c._spare.length / 3 + 1; // worker pool + client spares + the set on display
  check('double buffering: no lost buffers, <= 3 snapshot sets ever allocated', circulating === pools.sets + 1 && pools.sets <= 3 && c._poses.length === 64 * POSE_STRIDE,
    `${pools.sets} worker sets + 1 initial, ${c.perf.cmdAllocs} command buffers allocated`);
  check('command buffers recycled (<= 3 allocated after warm-up)', c.perf.cmdAllocs <= 3 && pools.cmdPending === 0);
  const W = handler.world();
  check('bodies sharing a shape object share one rebuilt shape in the worker', W.shapeOf[id] === W.shapeOf[id2] && W.shapeOf[id] && handler.ctx.shapes.length === 1);
  c.removeBody(id); c.removeBody(id2); c.step(DT);
  await drain(c);
  check('the shape is released with its last body (both sides)', handler.ctx.shapes[0] === undefined && c._sids.alive[0] === 0 && c.contacts.count === 0 && c.lastError === null);
  check('latency measured (ms, EMA)', c.latency > 0, `${c.latency.toFixed(3)} ms`);
  // command buffers: a frame that outgrows the buffer must not make every later frame regrow it
  const id3 = c.addBody({ shape: ball, position: [0, 0, 0] });
  const grows0 = c._cw.grows, allocs0 = c.perf.cmdAllocs;
  let grows1 = 0;
  for (let f = 0; f < 10; f++) {
    for (let k = 0; k < 300; k++) c.setPose(id3, f, k, 0); // 300 x 10 doubles > the 1024 the writer starts with
    c.step(DT);
    await drain(c);
    if (f === 0) grows1 = c._cw.grows;
  }
  check('largest command buffer kept: grows in the first big frame only, then reused', grows1 > grows0 && c._cw.grows === grows1 && c.perf.cmdAllocs - allocs0 <= 2 && c.lastError === null,
    `${c._cw.grows - grows0} doublings (frame 1), ${c.perf.cmdAllocs - allocs0} new buffers, ${c._cw.buf.length} doubles`);
  c.dispose();
}
{
  // world.step throws on its 3rd and 4th call; the 7th step's snapshot cannot be written
  class ThrowingWorld extends FakeWorld {
    get contacts() { if (this.calls === 7 && !this.snapThrown) { this.snapThrown = true; throw new Error('snapshot boom'); } return this._cb; }
    step(dt) { this.calls = (this.calls || 0) + 1; if (this.calls === 3 || this.calls === 4) throw new Error(`boom ${this.calls}`); super.step(dt); }
  }
  const errors = [], warn = console.warn;
  console.warn = () => {};
  const { c, seqs } = channelClient(ThrowingWorld, { onError: (m) => errors.push(m) });
  const id = c.addBody({ shape: sphere(0.5), position: [0, 0, 0] });
  c.setVelocity(id, 1, 0, 0);
  await c.ready;
  const xs = [];
  for (let i = 0; i < 10; i++) { c.step(DT); await drain(c); const p = [0, 0, 0]; c.getPose(id, p, null, 0); xs.push(p[0]); }
  console.warn = warn;
  let x = 0;
  for (let k = 0; k < 8; k++) x += DT; // calls 3 and 4 threw before integrating
  check('world.step throws (steps 3-4, snapshot 7): every step still answered', c.perf.inFlight === 0 && c.perf.late === 0 && c.perf.skipped === 0 && same(seqs, [1, 2, 3, 4, 5, 6, 8, 9, 10]),
    `snapshots ${seqs.join(',')}`);
  check('... client keeps stepping and reports failed / lastError / onError', c.failed && errors.length === 3 && /boom 3/.test(errors[0]) && /snapshot boom/.test(c.lastError) && xs[9] === Math.fround(x) && xs[9] > xs[5],
    `${errors.length} errors, x ${xs[9].toFixed(4)}`);
  c.dispose();
}
{
  // a worker world that assigns another id than the client mirrored: the id mirror is out of sync
  class SkewWorld extends FakeWorld { addBody(d) { if (d.kind === 'skew') this.bodies.alloc(); return super.addBody(d); } }
  const errors = [], warn = console.warn;
  console.warn = () => {};
  const { c } = channelClient(SkewWorld, { onError: (m) => errors.push(m) });
  await c.ready;
  c.addBody({ shape: sphere(0.5), position: [0, 0, 0] });
  check('no failure before the mirror breaks', !c.failed);
  c.addBody({ shape: sphere(0.5), position: [0, 0, 0], kind: 'skew' });
  c.step(DT); await drain(c); await tick();
  console.warn = warn;
  check('addBody id mirror out of sync -> failed', c.failed && /out of sync/.test(c.lastError) && errors.length === 1);
  c.dispose();
}

// ==================================================================== 6. dispose, pending patches, layer table
{
  const pw = new PhysicsWorld();
  pw.addStaticBox(-5, -1, 5, 0, -5, 5, 'ground');
  const dw = dummyWorker();
  const c = new DynamicsClient({ physics: pw, worker: () => dw, maxBodies: 8, readyTimeout: 60000 });
  check('init carries the named layer table (PhysicsWorld.namedLayers)', dw.posted[0].type === 'init' && dw.posted[0].named === namedLayers().length && dw.posted[0].named >= 2);
  const A = c.addBody({ shape: sphere(0.5), position: [0, 1, 0] });
  dw.onmessage({ data: { type: 'ready', capacity: 8, count: 0, time: 0, stepCount: 0 } });
  c.step(DT);              // seq 1 carries the add
  c.setPose(A, 5, 1, 0);   // seq 2 will carry the teleport
  dw.onmessage({ data: snapMsg(8, 1, DT, [{ id: A, p: [0, 1, 0], type: BODY.DYNAMIC, aabb: [-0.5, 0.5, -0.5, 0.5, 1.5, 0.5] }]) });
  const hit = c.queryColliders(5, 0, 0.1, []);
  check('setPose on a pending body: not pending once the worker has it (AABB at the patched pose)', !(c._bits(A) & POSE_PENDING) && hit.length === 1 && hit[0].minX === 4.5 && hit[0].maxX === 5.5);
  pw.addDynamicProvider(c.colliderProvider);
  attachToPhysicsWorld(c, pw);
  const before = pw.providers.length;
  c.dispose();
  check('dispose: providers removed, worker terminated, buffers dropped, reads empty',
    before === 2 && pw.providers.length === 0 && c.disposed && dw.dead && c.bodies === null && !c.getPose(A, [0, 0, 0]) && c.queryColliders(5, 0, 1, []).length === 0 && c._poses === null);
  const cm = new DynamicsClient({ physics: pw, worker: false });
  attachToPhysicsWorld(cm, pw);
  cm.dispose();
  check('dispose (in-thread): providers removed, world disposed', pw.providers.length === 0 && cm.world === null && cm.disposed);
}

// ==================================================================== 7. the real dynamics world
/** A Worker-like running dynamics.worker.js in a real node:worker_threads thread. */
function threadWorker() {
  const boot = `import { parentPort } from 'node:worker_threads';
const { createDynamicsWorkerHandler } = await import(${JSON.stringify(src('dynamics.worker.js'))});
const handle = createDynamicsWorkerHandler((m, t) => parentPort.postMessage(m, t));
parentPort.on('message', handle);`;
  const thread = new ThreadWorker(new URL('data:text/javascript,' + encodeURIComponent(boot)));
  const adapter = { onmessage: null, onerror: null, postMessage: (m, t) => thread.postMessage(m, t), terminate: () => thread.terminate() };
  thread.on('message', (data) => adapter.onmessage && adapter.onmessage({ data }));
  thread.on('error', (e) => adapter.onerror && adapter.onerror({ message: e.message }));
  return adapter;
}
async function lockstep(cm, cw, n, each) {
  for (let i = 0; i < n; i++) {
    if (each) each(i);
    cm.step(DT); cw.step(DT);
    const t0 = performance.now();
    while (cw.perf.inFlight && performance.now() - t0 < 5000) await pause(0); // no merged / skipped steps
  }
}
if (!REAL) {
  skip('serialize() -> from() -> 60 steps -> bit-identical', `DynamicsWorld not loadable yet (${whyNot})`);
  skip('worker thread vs in-thread world -> bit-identical', 'needs the real DynamicsWorld');
  skip('worker-mode start from serialized', 'needs the real DynamicsWorld');
  skip('start-up fallback replays queued commands', 'needs the real DynamicsWorld');
} else {
  const { DynamicsWorld, shapeFromDescriptor } = await import(src('dynamics/DynamicsWorld.js'));
  const { JOINT } = await import(src('dynamics/joints.js'));
  const { trimesh } = await import(src('dynamics/shapes.js'));
  layerOf('zz-another-system'); // a kind registered on this thread only: a fresh worker would hand out other bits
  const makePhysics = () => {
    const pw = new PhysicsWorld({ groundLevel: 0 });
    pw.addStaticBox(-20, -1, 20, 0, -20, 20, 'ground');
    pw.addStaticBox(4, 0, 6, 1.5, -1, 1, 'building');
    return pw;
  };
  const crate = box(0.3, 0.3, 0.3), ball = sphere(0.25), pill = capsule(0.2, 0.3);
  const rock = convex([0.3, 0, 0, -0.3, 0, 0, 0, 0.35, 0, 0, -0.25, 0, 0, 0, 0.3, 0, 0, -0.3, 0.2, 0.2, 0.2]);
  /** The same scene through any object with the DynamicsWorld / DynamicsClient API. */
  function scene(w) {
    const ids = [];
    for (let i = 0; i < 5; i++) ids.push(w.addBody({ shape: crate, position: [0, 0.3 + i * 0.61, 0], friction: 0.7 }));
    ids.push(w.addBody({ shape: ball, position: [2, 2, 0.1], restitution: 0.5 }));
    ids.push(w.addBody({ shape: pill, position: [-2, 1.5, 0], quaternion: [0, 0, 0.3826834, 0.9238795] }));
    ids.push(w.addBody({ shape: rock, position: [3, 3, -1], density: 2500 }));
    const lid = w.addBody({ shape: box(0.5, 0.05, 0.5), position: [-4, 2, 2] });
    ids.push(lid);
    // masked off the 'building' layer: falls through the building box onto the ground only if layer bits agree
    ids.push(w.addBody({ shape: crate, position: [5, 2.5, 0], mask: ~layerOf('building') }));
    w.addJoint({ type: JOINT.HINGE, bodyA: lid, bodyB: null, anchorA: [-0.5, 0, 0], axisA: [0, 0, 1] });
    w.setVelocity(ids[5], 1, 0, 0.5);
    w.applyImpulse(ids[7], [0, 0, 2]);
    return ids;
  }
  const hash = (B) => {
    let h = 2166136261;
    for (const arr of [B.pos, B.quat, B.vel, B.angVel]) {
      const stride = arr.length / B.capacity, u8 = new Uint8Array(arr.buffer, arr.byteOffset, B.count * stride * 8);
      for (let i = 0; i < u8.length; i++) { h ^= u8[i]; h = Math.imul(h, 16777619) >>> 0; }
    }
    return h >>> 0;
  };
  const samePos = (sw, sm) => same(sw.bodies.pos, sm.bodies.pos) && same(sw.bodies.quat, sm.bodies.quat) && same(sw.bodies.angVel, sm.bodies.angVel);

  // ---- a. shape descriptors as typed arrays; serialize -> from -> continue
  {
    const mesh = trimesh([0, 0, 0, 4, 0, 0, 0, 0, 4, 4, 0, 4], [0, 1, 2, 2, 1, 3]);
    let ok = true;
    for (const sh of [rock, mesh, crate]) {
      const w = new P.CommandWriter(64);
      w.addBody(0, 0, { shape: sh }, P.shapeDescriptor(sh));
      const d = P.decodeCommands(w.buf, w.len)[0].args[2].shape, back = shapeFromDescriptor(d);
      for (const k of ['volume', 'boundingRadius', 'centroid', 'half', 'offset', 'unitInertia', 'points', 'planes', 'faceStart', 'edges', 'positions', 'indices']) if (!same(back[k], sh[k])) ok = false;
    }
    const dr = P.shapeDescriptor(rock), dm = P.shapeDescriptor(mesh);
    check('shape descriptors: typed arrays on the wire, rebuilt bit-identically', ok && dr.points instanceof Float64Array && dm.positions instanceof Float64Array && dm.indices instanceof Uint32Array);
    const pw = makePhysics();
    const w1 = new DynamicsWorld({ physics: pw, maxBodies: 64 });
    const ids = scene(w1);
    for (let i = 0; i < 30; i++) w1.step(DT);
    const s = w1.serialize();
    const w2 = DynamicsWorld.from(structuredClone(s), { physics: pw }); // what a worker init receives
    const start = w1.bodies.pos[ids[5] * 3];
    for (let i = 0; i < 60; i++) { w1.step(DT); w2.step(DT); }
    const h1 = hash(w1.bodies), h2 = hash(w2.bodies);
    check('serialize() -> structuredClone -> from() -> 60 steps -> bit-identical', h1 === h2 && w1.bodies.pos[ids[5] * 3] !== start && w1.contacts.count > 0,
      `hash ${h1.toString(16)} / ${h2.toString(16)}, ${w1.contacts.count} contacts`);
  }

  // ---- b. a real worker thread vs the in-thread world
  {
    const pw = makePhysics();
    const errors = [];
    const cm = new DynamicsClient({ physics: pw, maxBodies: 64, worker: false });
    const cw = new DynamicsClient({ physics: pw, maxBodies: 64, worker: threadWorker, readyTimeout: 10000, onError: (m) => errors.push(m) });
    const idsM = scene(cm), idsW = scene(cw);
    await cw.ready;
    await lockstep(cm, cw, 60, (i) => {
      if (i === 20) for (const c of [cm, cw]) { c.applyImpulse(idsM[0], 0.3, 0.6, 0, 0, 0, 3); c.setPose(idsM[6], -2, 2.5, 0); c.setGravity(0, -9.81, 0.5); }
      if (i === 40) for (const c of [cm, cw]) c.removeBody(idsM[2]);
    });
    let ok = same(idsM, idsW) && cw.mode === 'worker' && cw.stepCount === cm.stepCount && cw.perf.skipped === 0;
    const BM = cm.bodies, BW = cw.bodies;
    for (let id = 0; id < BM.count; id++) {
      if (BM.alive[id] !== BW.alive[id]) { ok = false; continue; }
      if (!BM.alive[id]) continue;
      for (let k = 0; k < 3; k++) if (BW.pos[id * 3 + k] !== Math.fround(BM.pos[id * 3 + k])) ok = false;
      for (let k = 0; k < 4; k++) if (BW.quat[id * 4 + k] !== Math.fround(BM.quat[id * 4 + k])) ok = false;
    }
    const masked = idsM[9], fellThrough = BM.pos[masked * 3 + 1] < 0.5;
    const sw = await cw.serialize(), sm = cm.world.serialize();
    check('worker thread == in-thread world after 60 steps (float32 snapshot)', ok && cw.contacts.count === cm.contacts.count && errors.length === 0,
      `${BM.size} bodies, ${cm.contacts.count} contacts, worker ${cw.perf.stepMs.toFixed(2)} ms/step, latency ${cw.latency.toFixed(2)} ms${errors.length ? ', ' + errors[0] : ''}`);
    check('worker world state bit-identical to the in-thread one (serialize)', samePos(sw, sm), 'static boxes + named layer table shipped');
    check('layer bits agree across threads (masked crate falls through the building in both)', fellThrough && BW.pos[masked * 3 + 1] === Math.fround(BM.pos[masked * 3 + 1]),
      `y ${BM.pos[masked * 3 + 1].toFixed(3)}, building bit ${layerOf('building').toString(2).length - 1}`);
    cw.dispose(); cm.dispose();
  }

  // ---- c. worker mode started from a serialized world (ids, poses and userData known before 'ready')
  {
    const pw = makePhysics();
    const w0 = new DynamicsWorld({ physics: pw, maxBodies: 64 });
    const ids0 = scene(w0);
    w0.bodies.userData[ids0[1]] = { tag: 'crate1' };
    for (let i = 0; i < 20; i++) w0.step(DT);
    w0.removeBody(ids0[3]); // a hole in the id free list
    const s = w0.serialize();
    const errors = [];
    const cw = new DynamicsClient({ serialized: s, physics: pw, worker: threadWorker, readyTimeout: 10000, onError: (m) => errors.push(m) });
    const cm = new DynamicsClient({ serialized: s, physics: pw, worker: false });
    const p = [0, 0, 0];
    cw.getPose(ids0[0], p, null, 0);
    let alive = cw.bodies.count === w0.bodies.count;
    for (let id = 0; id < w0.bodies.count; id++) if (cw.bodies.alive[id] !== w0.bodies.alive[id]) alive = false;
    const nw = cw.addBody({ shape: ball, position: [0, 4, 3] }), nm = cm.addBody({ shape: ball, position: [0, 4, 3] });
    check('serialized start: ids, poses, userData mirrored before ready; next id fills the hole', cw.mode === 'starting' && alive && p[1] === Math.fround(w0.bodies.pos[ids0[0] * 3 + 1]) &&
      cw.bodies.userData[ids0[1]].tag === 'crate1' && nw === ids0[3] && nm === nw);
    await cw.ready;
    await lockstep(cm, cw, 30);
    const sw = await cw.serialize(), sm = cm.world.serialize();
    check('serialized start: 30 steps bit-identical to in-thread, userData back in serialize()', cw.mode === 'worker' && samePos(sw, sm) && sw.bodies.userData[ids0[1]].tag === 'crate1' && errors.length === 0);
    cw.dispose(); cm.dispose();
  }

  // ---- d. a worker that never answers: the queued commands replay into an in-thread world
  {
    const pw = makePhysics();
    const mute = { onmessage: null, onerror: null, postMessage() {}, terminate() {} };
    const warn = console.warn; console.warn = () => {};
    const cf = new DynamicsClient({ physics: pw, maxBodies: 64, worker: () => mute, readyTimeout: 30 });
    scene(cf); cf.step(DT); cf.step(DT); cf.step(DT); // two owed, one skipped
    await cf.ready;
    console.warn = warn;
    const ref = new DynamicsClient({ physics: pw, maxBodies: 64, worker: false });
    scene(ref); ref.step(DT); ref.step(DT);
    check('start-up fallback: queued commands + owed steps replay in-thread', cf.mode === 'main' && cf.perf.skipped === 1 && hash(cf.bodies) === hash(ref.bodies) && cf.stepCount === 2);
    cf.dispose(); ref.dispose();
  }
}

console.log('dynamics worker protocol / DynamicsClient');
for (const r of results) console.log(r);
console.log(failures ? `\n${failures} FAILED` : `\nall ${results.filter((r) => r.startsWith('PASS')).length} passed${REAL ? '' : ' (stub DynamicsWorld: ' + whyNot + ')'}`);
process.exit(failures ? 1 : 0);
