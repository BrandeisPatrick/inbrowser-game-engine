// Main-thread front of the XPBD DynamicsWorld with one API whether the world runs in a Web Worker
// (dynamics.worker.js) or in this thread (`worker: false`, no Worker (Node), or a worker that failed to start), so
// games write one code path.
//
//   const dyn = new DynamicsClient({ physics, substeps: 4, worker: true });   // DynamicsWorld options + client options
//   const id = dyn.addBody({ shape: box(0.25, 0.25, 0.25), position: [0, 3, 0] }); // id known synchronously
//   attachToPhysicsWorld(dyn, physics);       // bridge.js reads dyn.bodies (a snapshot mirror in worker mode)
//   engine fixed step:  dyn.applyForce(id, 0, 50, 0); dyn.step(fixedDt);
//   each frame:         dyn.getPose(id, mesh.position, mesh.quaternion);
//
// Worker mode (protocol, buffer layouts, opcodes: header of dynamics.worker.js). Calls are encoded into a Float64
// command stream; step(dt) posts it ('cmd') and then a 'step' message; the worker applies the commands, steps and
// answers with a transferable snapshot (poses + AABBs + contacts), double-buffered like Traffic: the previous set
// rides back with the next step, command buffers come back with the snapshot, so nothing is allocated per frame once
// warm. Body ids are assigned here with BodyStore's rule (lowest free id first) and checked by the worker; joint ids
// are the client's own (same rule), mapped to the world's in the worker. Shapes cross once per shape object
// (shapeDescriptor -> shapeFromDescriptor, bit-identical); bodies sharing a shape here share it there.
//   Reads (getPose, getVelocity, isAwake, bodies.*, contacts, stats, queryColliders) come from the latest snapshot,
// one or two steps behind the calls made since. getPose extrapolates it by the snapshot velocities over
// `time - snapshotTime` (the steps requested but not answered yet; capped at maxExtrapolation, default 0.1 s). A body
// added, removed, teleported (setPose), given a velocity, woken or put to sleep is patched into the snapshot at once
// and re-applied to snapshots that predate the command, so it shows where it was put immediately; a just-added body
// has no AABB (no colliders) until the worker has seen it.
//   Step policy: at most 2 step messages in flight. A step() that finds 2 in flight is owed and goes out with the next
// step() that finds a free slot, as ONE message with n = 2 (both of the latest dt); a third step owed meanwhile is
// skipped (perf.skipped: sim time falls behind the caller's clock instead of the worker queueing up). Commands issued
// between two merged steps therefore apply before both of them.
//   Before the worker answers 'ready' only 'init' is posted: commands and up to 2 steps are held, so if the worker
// cannot start (no answer within readyTimeout, an error, uncloneable options) they replay into an in-thread world and
// nothing is lost.
//   Failures after 'ready': an error the worker reports (world.step threw - that step's snapshot still arrives, with
// `error` -, a command batch threw, the id mirror went out of sync) sets `failed` (sticky) and `lastError`, calls
// onError and warns once; stepping goes on (the world may have diverged from what this thread assumes). A worker that
// dies cannot be recovered (its state is gone): `failed` is set, steps become no-ops and the last snapshot stays.
//   The worker's world gets the PhysicsWorld passed as `physics` as a copy of its static boxes + groundLevel taken at
// construction (boxes added later, a heightfield (worker mode then falls back to in-thread, with a warning) and the
// PhysicsWorld's dynamic providers such as traffic stay on this thread). Raycasts / shape casts need the world:
// in-thread `dyn.world.raycast(...)`.
//
// Errors follow DynamicsWorld: misuse detectable here (unknown body id, bad addBody options, kinematic target on a
// non-kinematic body, invalid joint bodies) throws synchronously in both modes; what fails inside the worker is
// reported (above).
//
// Differences between the modes (the same code runs in both, but):
// - worker-mode poses, velocities and AABBs are float32 snapshot copies, so bridge collider records (and a car
//   colliding with them) are not bit-identical across modes, and reads lag the calls by one or two steps;
// - applyImpulse / applyForce / applyTorque / setKinematicTarget are not patched into the snapshot: velocity / pose
//   reads stay stale until the snapshot of the step that applied them arrives;
// - serialize() returns a Promise in both modes; addJoint returns -1 when maxJoints is reached instead of throwing;
// - worker mode: `joints` is null (debugLines draws no joints) and there is no raycast / shapeCast / overlap / began
//   / ended (they need the world: in-thread `dyn.world`).
//
// In-thread mode: calls go straight to the DynamicsWorld (fixed-layout commands through the same applier the worker
// uses, so argument semantics match); `bodies` / `contacts` / `stats` are the world's own stores, queryColliders is
// the world's, getPose extrapolates only when asked (`ahead`).
import { DynamicsWorld } from './dynamics/DynamicsWorld.js';
import { BODY, FLAG, SHAPE } from './dynamics/state.js';
import { writeDebugLines } from './dynamics/debug.js';
import { detachFromPhysicsWorld } from './dynamics/bridge.js';
import {
  POSE_STRIDE, POSE_ALIVE, POSE_PENDING, POSE_TYPE_SHIFT, CONTACT_STRIDE, DEFAULT_SNAPSHOT_CONTACTS, OP,
  CommandWriter, IdAllocator, applyCommands, applyOp, createApplyContext, writeSnapshot, mapExistingJoints,
  shapeDescriptor, staticWorldOf, namedLayers,
} from './dynamics.worker.js';

export { POSE_STRIDE, POSE_ALIVE, POSE_PENDING, POSE_TYPE_SHIFT, CONTACT_STRIDE, OP };

const READY_TIMEOUT = 5000;
const MAX_IN_FLIGHT = 2;
const TYPE_OF = Object.freeze({ static: BODY.STATIC, kinematic: BODY.KINEMATIC, dynamic: BODY.DYNAMIC });
/** Options the client reads itself (never part of the world options sent to the worker). */
const CLIENT_KEYS = ['worker', 'extrapolate', 'maxExtrapolation', 'serialized', 'onError', 'onSnapshot', 'readyTimeout', 'snapshotContacts', 'physics'];
let _warned = false;

/** Whether `worker: true` can be honoured here (a Worker constructor exists). */
export function dynamicsWorkerAvailable() { return typeof Worker !== 'undefined'; }

function worldOptionsOf(o) {
  const w = {};
  for (const k of Object.keys(o)) if (!CLIENT_KEYS.includes(k)) w[k] = o[k];
  return w;
}
/** Component i of [x, y, z(, w)] / {x, y, z(, w)}; `def` when absent. */
function comp(v, i, def) {
  if (v === null || v === undefined) return def;
  const x = Array.isArray(v) || ArrayBuffer.isView(v) ? v[i] : i === 0 ? v.x : i === 1 ? v.y : i === 2 ? v.z : v.w;
  return typeof x === 'number' ? x : def;
}
const _g = new Float64Array(7);
/** Pose arguments: (x, y, z[, qx, qy, qz, qw]) scalars or (position, quaternion?) vectors -> _g; NaN x / qw = keep. */
function poseArgs(a, b, c, d, e, f, g) {
  if (typeof a === 'number') {
    _g[0] = a; _g[1] = +b || 0; _g[2] = +c || 0;
    if (typeof d === 'number' && typeof g === 'number') { _g[3] = d; _g[4] = +e || 0; _g[5] = +f || 0; _g[6] = g; } else { _g[3] = 0; _g[4] = 0; _g[5] = 0; _g[6] = NaN; }
    return _g;
  }
  _g[0] = comp(a, 0, NaN); _g[1] = comp(a, 1, 0); _g[2] = comp(a, 2, 0);
  _g[3] = comp(b, 0, 0); _g[4] = comp(b, 1, 0); _g[5] = comp(b, 2, 0); _g[6] = comp(b, 3, NaN);
  return _g;
}
/**
 * Two 3-vectors: (x, y, z[, u, v, w]) scalars or (vecA[, vecB]) -> _g[0..5]. A missing first vector is 0 (or NaN x
 * when firstDef is NaN = keep); a missing second one has NaN u (= keep / centre of mass).
 */
function pairArgs(a, b, c, d, e, f, firstDef) {
  if (typeof a === 'number') {
    _g[0] = a; _g[1] = +b || 0; _g[2] = +c || 0;
    if (typeof d === 'number') { _g[3] = d; _g[4] = +e || 0; _g[5] = +f || 0; } else { _g[3] = NaN; _g[4] = 0; _g[5] = 0; }
    return _g;
  }
  _g[0] = comp(a, 0, firstDef); _g[1] = comp(a, 1, 0); _g[2] = comp(a, 2, 0);
  _g[3] = comp(b, 0, NaN); _g[4] = comp(b, 1, 0); _g[5] = comp(b, 2, 0);
  return _g;
}
function out3(o, x, y, z) {
  if (!o) return;
  if (Array.isArray(o) || ArrayBuffer.isView(o)) { o[0] = x; o[1] = y; o[2] = z; } else if (typeof o.set === 'function') o.set(x, y, z);
  else { o.x = x; o.y = y; o.z = z; }
}
function out4(o, x, y, z, w) {
  if (!o) return;
  if (Array.isArray(o) || ArrayBuffer.isView(o)) { o[0] = x; o[1] = y; o[2] = z; o[3] = w; } else if (typeof o.set === 'function') o.set(x, y, z, w);
  else { o.x = x; o.y = y; o.z = z; o.w = w; }
}
const _eq = new Float64Array(4);
/** q' = dq(w h) * q (world-space angular velocity) -> _eq. */
function advanceQuat(qx, qy, qz, qw, wx, wy, wz, h) {
  const wl = Math.sqrt(wx * wx + wy * wy + wz * wz);
  if (!(wl * Math.abs(h) > 1e-9)) { _eq[0] = qx; _eq[1] = qy; _eq[2] = qz; _eq[3] = qw; return _eq; }
  const half = 0.5 * wl * h, s = Math.sin(half) / wl, c = Math.cos(half);
  const ax = wx * s, ay = wy * s, az = wz * s;
  const x = c * qx + ax * qw + ay * qz - az * qy;
  const y = c * qy - ax * qz + ay * qw + az * qx;
  const z = c * qz + ax * qy - ay * qx + az * qw;
  const w = c * qw - ax * qx - ay * qy - az * qz;
  const inv = 1 / Math.sqrt(x * x + y * y + z * z + w * w);
  _eq[0] = x * inv; _eq[1] = y * inv; _eq[2] = z * inv; _eq[3] = w * inv;
  return _eq;
}

/**
 * Read-only BodyStore look-alike over the latest snapshot (worker mode): Float32 pos (3n), quat (4n), vel (3n),
 * angVel (3n), aabb (6n), Uint8 flags, refreshed in place (lazily, once per snapshot) when read; alive / type /
 * layer / mask / shape / userData / count are the client's own live state (a just-added body is alive at once).
 * Enough for bridge.js (attachToPhysicsWorld) and dynamics/debug.js (PhysicsDebug) to take a client.
 */
class SnapshotBodies {
  constructor(client, capacity) {
    this._c = client; this.capacity = capacity;
    this._pos = new Float32Array(3 * capacity); this._quat = new Float32Array(4 * capacity);
    this._vel = new Float32Array(3 * capacity); this._angVel = new Float32Array(3 * capacity);
    this._aabb = new Float32Array(6 * capacity); this._flags = new Uint8Array(capacity);
    this.type = new Uint8Array(capacity);
    this.layer = new Int32Array(capacity).fill(1); this.mask = new Int32Array(capacity).fill(-1);
    this.shape = new Array(capacity).fill(null);
    this.userData = new Array(capacity).fill(null);
  }
  get count() { return this._c._ids.count; }
  get alive() { return this._c._ids.alive; }
  get pos() { this._c._sync(); return this._pos; }
  get quat() { this._c._sync(); return this._quat; }
  get vel() { this._c._sync(); return this._vel; }
  get angVel() { this._c._sync(); return this._angVel; }
  get aabb() { this._c._sync(); return this._aabb; }
  get flags() { this._c._sync(); return this._flags; }
  get size() { const a = this.alive; let s = 0; for (let i = 0; i < this.count; i++) s += a[i]; return s; }
  isDynamic(id) { return this.alive[id] === 1 && this.type[id] === BODY.DYNAMIC; }
  isAwake(id) { return (this.flags[id] & FLAG.ASLEEP) === 0; }
}

/** ContactBuffer look-alike over the latest snapshot (worker mode): count, a, b, px..nz, depth, lambdaN; total. */
class SnapshotContacts {
  constructor(capacity) {
    this.capacity = capacity; this.count = 0; this.total = 0; this.overflow = 0;
    this.a = new Int32Array(capacity); this.b = new Int32Array(capacity);
    this.px = new Float32Array(capacity); this.py = new Float32Array(capacity); this.pz = new Float32Array(capacity);
    this.nx = new Float32Array(capacity); this.ny = new Float32Array(capacity); this.nz = new Float32Array(capacity);
    this.depth = new Float32Array(capacity); this.lambdaN = new Float32Array(capacity);
  }
}

export class DynamicsClient {
  /**
   * @param {object} [options]  DynamicsWorld options (physics, substeps, maxBodies, maxJoints, gravity, ...) plus:
   * @param {boolean|Function} [options.worker=true]  run the world in a Web Worker when `Worker` exists; a function
   *        returning a Worker-like ({ postMessage(msg, transfer), terminate(), onmessage, onerror }) is used instead
   *        of creating dynamics.worker.js (tests, custom bundling)
   * @param {object} [options.serialized]        start from DynamicsWorld#serialize() output
   * @param {boolean} [options.extrapolate=true]  getPose extrapolates the snapshot to the client's time (worker mode)
   * @param {number} [options.maxExtrapolation=0.1]  s
   * @param {number} [options.snapshotContacts=2048] contacts carried per snapshot
   * @param {number} [options.readyTimeout=5000]  ms before an unanswered worker is replaced by the in-thread world
   * @param {(message: string) => void} [options.onError]   errors reported by the worker
   * @param {(client: DynamicsClient) => void} [options.onSnapshot]  a snapshot arrived (worker mode)
   */
  constructor(options = {}) {
    const o = this.options = options || {};
    this.extrapolate = o.extrapolate !== false;
    this.maxExtrapolation = Number.isFinite(o.maxExtrapolation) ? o.maxExtrapolation : 0.1;
    this.onError = typeof o.onError === 'function' ? o.onError : null;
    this.onSnapshot = typeof o.onSnapshot === 'function' ? o.onSnapshot : null;
    /** 'worker' once the worker answered, 'main' when the world runs in this thread, 'starting' before. */
    this.mode = 'main';
    this.world = null; this.worker = null;
    /** Simulation time requested so far (s; skipped steps excluded) / steps of the latest state. */
    this.time = 0; this.stepCount = 0;
    this.lastError = null; this.failed = false; this.disposed = false;
    // latency: ms from step post to its snapshot (EMA); stepMs: worker ms per step message (EMA)
    this.perf = { latency: 0, stepMs: 0, snapshots: 0, late: 0, skipped: 0, inFlight: 0, cmdAllocs: 0, cmdBytes: 0 };
    this._physics = o.physics || null;
    this._ctx = createApplyContext();
    this._a = new Float64Array(8); // in-thread command args
    const maxJoints = Math.max(1, (o.maxJoints ?? 1024) | 0);
    this._jids = new IdAllocator(maxJoints);
    this._jointA = new Int32Array(maxJoints).fill(-1); this._jointB = new Int32Array(maxJoints).fill(-1);
    this._recs = [];
    this._serials = new Map(); this._serialId = 0;
    this._worldOptions = worldOptionsOf(o);
    // PhysicsWorld provider: bound, and named like bridge.js's so a DynamicsWorld's static pass skips it
    const query = DynamicsClient.prototype.queryColliders, self = this;
    this.queryColliders = this.colliderProvider = function dynamicsProvider(x, z, r, out) { return query.call(self, x, z, r, out); };

    let factory = null;
    const want = o.worker ?? true;
    if (typeof want === 'function') factory = want;
    else if (want && dynamicsWorkerAvailable()) {
      // keep this literal form: Vite detects it to bundle the worker and rewrite its URL in builds
      factory = () => new Worker(new URL('./dynamics.worker.js', import.meta.url), { type: 'module' });
    }
    let shipped = null;
    if (factory && this._physics) {
      try { shipped = staticWorldOf(this._physics); } catch (e) { factory = null; warnOnce(`worker mode off: ${e.message}`); }
    }
    if (!factory) { this._startMain(true); this.ready = Promise.resolve(this); return; }
    this.ready = this._startWorker(factory, shipped);
  }

  // ================================================================ start-up
  _makeWorld() {
    const s = this.options.serialized;
    return s ? DynamicsWorld.from(s, { physics: this._physics }) : new DynamicsWorld({ ...this._worldOptions, physics: this._physics });
  }
  _startMain(adoptJoints) {
    const W = this.world = this._makeWorld();
    this.capacity = W.bodies.capacity;
    if (adoptJoints) this._adoptJoints(W); else mapExistingJoints(W, this._ctx.jointMap);
    this.time = W.time || 0; this.stepCount = W.stepCount || 0;
    this.mode = 'main';
  }
  _adoptJoints(W) {
    const J = W.joints;
    if (!J || !J.alive || !(J.count >= 0)) return;
    this._jids.adopt(J.alive, J.count);
    mapExistingJoints(W, this._ctx.jointMap);
    for (let j = 0; j < J.count; j++) if (J.alive[j]) { this._jointA[j] = J.a[j]; this._jointB[j] = J.b[j]; }
  }

  _startWorker(factory, shipped) {
    const o = this.options;
    this.mode = 'starting';
    let cap = Math.max(1, (o.maxBodies ?? 4096) | 0), temp = null, serialized = null;
    if (o.serialized) {
      // the ids of a restored world must be known now: read them from a throw-away in-thread copy (once)
      temp = DynamicsWorld.from(o.serialized);
      cap = temp.bodies.capacity;
      const s = o.serialized;
      serialized = s.bodies ? { ...s, bodies: { ...s.bodies, userData: [] } } : s; // userData stays on this thread
    }
    this._ccap = Math.max(0, (o.snapshotContacts ?? DEFAULT_SNAPSHOT_CONTACTS) | 0);
    this._initSnapshot(cap);
    if (temp) {
      const B = temp.bodies, M = this._mirror;
      this._ids.adopt(B.alive, B.count);
      for (let i = 0; i < B.count; i++) {
        if (!B.alive[i]) continue;
        M.userData[i] = B.userData[i] ?? null; M.shape[i] = B.shape[i]; M.type[i] = B.type[i]; M.layer[i] = B.layer[i]; M.mask[i] = B.mask[i];
        this._kinds[i] = temp.kindOf ? temp.kindOf(i) : 'rigid';
      }
      this._snapCount = writeSnapshot(temp, this._poses, this._aabbs, this._cbuf).count;
      this._adoptJoints(temp);
      this.time = this._snapTime = temp.time || 0; this.stepCount = temp.stepCount || 0;
    }
    this._cw = new CommandWriter(1024);
    this._cmdSpare = []; this._spare = []; this._cmdMax = this._cw.buf.length; this._answered = 0;
    this._msg = { type: 'step', dt: 0, n: 0, seq: 0, bufs: null };
    this._cmdMsg = { type: 'cmd', seq: 0, buffer: null, length: 0 };
    this._tr = []; this._cmdTr = [null];
    this._postT = new Float64Array(8);
    this._seq = 0; this._snapSeq = 0; this._inFlight = 0; this._owed = 0; this._owedDt = 0;
    this._ready = false;
    return new Promise((resolve) => {
      this._resolveReady = () => { clearTimeout(this._timer); this._timer = 0; this._resolveReady = null; resolve(this); };
      let worker;
      try { worker = factory(); } catch (e) { this._fallback(e); return; }
      this.worker = worker;
      worker.onmessage = (ev) => { if (this.worker === worker) this._onMessage(ev.data); };
      worker.onerror = (ev) => { if (ev && ev.preventDefault) ev.preventDefault(); if (this.worker === worker) this._workerFailed(new Error((ev && ev.message) || 'worker error')); };
      worker.onmessageerror = () => { if (this.worker === worker) this._workerFailed(new Error('unreadable worker message')); };
      const ms = o.readyTimeout ?? READY_TIMEOUT;
      this._timer = setTimeout(() => this._fallback(new Error(`no answer within ${ms} ms`)), ms);
      try {
        worker.postMessage({ type: 'init', options: this._worldOptions, seed: o.seed, serialized, physics: shipped, namedLayers: namedLayers(), contactCapacity: this._ccap });
      } catch (e) { this._fallback(e); } // DataCloneError: options holding functions etc.
    });
  }

  /** Snapshot-side state for `cap` body ids (worker mode). */
  _initSnapshot(cap) {
    this.capacity = cap;
    this._ids = new IdAllocator(cap);
    this._sids = new IdAllocator(cap); this._shapeIds = new WeakMap(); this._sidShape = new Array(cap).fill(null);
    this._sidRefs = new Int32Array(cap); this._bodySid = new Int32Array(cap).fill(-1); this._addSeq = new Int32Array(cap);
    this._kinds = new Array(cap).fill('rigid');
    this._poses = new Float32Array(cap * POSE_STRIDE);
    this._aabbs = new Float32Array(cap * 6);
    this._cbuf = new Float32Array(1 + this._ccap * CONTACT_STRIDE);
    this._snapCount = 0; this._snapTime = this.time; this._contactTotal = 0; this._stats = {};
    this._ov = new Float32Array(cap * POSE_STRIDE); this._ovA = new Float32Array(cap * 6);
    this._ovSeq = new Int32Array(cap); this._ovList = new Int32Array(cap); this._ovCount = 0;
    this._mirror = new SnapshotBodies(this, cap);
    this._contacts = new SnapshotContacts(this._ccap);
    this._gen = 1; this._syncGen = 0; this._cgen = 0;
  }

  _fallback(err) {
    if (this.world || this.disposed) return;
    if (this.worker) { try { this.worker.terminate(); } catch { /* gone */ } this.worker = null; }
    warnOnce(`Web Worker unavailable, simulating on the main thread: ${err && err.message ? err.message : err}`);
    this._startMain(false); // the client's joint ids were adopted from the same state already
    // replay everything issued while the worker was starting (ids were assigned with the world's rule)
    if (this._cw.len) { applyCommands(this.world, this._cw.buf, this._cw.len, this._ctx); this._cw.reset(); this._flushCtxErrors(); }
    for (let i = 0; i < this._owed; i++) this.world.step(this._owedDt);
    this._owed = 0; this._ovCount = 0;
    this.time = this.world.time; this.stepCount = this.world.stepCount;
    if (this._resolveReady) this._resolveReady();
  }

  _workerFailed(err) {
    if (!this._ready) { this._fallback(err); return; }
    if (this.worker) { try { this.worker.terminate(); } catch { /* gone */ } this.worker = null; }
    this._inFlight = 0; this.perf.inFlight = 0;
    this._fail(`worker died (state lost, snapshot frozen): ${err && err.message ? err.message : err}`);
  }
  /** The worker reported a failure: sticky `failed`, `lastError`, onError, one warning. */
  _fail(message) {
    this.failed = true;
    this._error(message);
  }

  // ================================================================ messages
  _onMessage(m) {
    if (!m || this.disposed) return;
    if (m.type === 'poses') { this._onPoses(m); if (m.error) this._fail(`step ${m.seq}: ${m.error}`); return; }
    if (m.type === 'ready') {
      if (this._ready || this.world) return;
      if (m.capacity !== this.capacity) { this._fallback(new Error(`worker world holds ${m.capacity} bodies, the client expected ${this.capacity}`)); return; }
      this._ready = true; this.mode = 'worker';
      if (this._resolveReady) this._resolveReady();
      this.flush();
      return;
    }
    if (m.type === 'error') {
      if (!this._ready) { this._fallback(new Error(m.message)); return; }
      // a step the worker could not answer with a snapshot: this error is its answer
      if (m.phase === 'step' && m.seq > this._answered && m.seq <= this._seq) { this._answered = m.seq; this._inFlight = Math.max(0, this._inFlight - 1); this.perf.inFlight = this._inFlight; }
      this._fail(`${m.phase || 'worker'}${m.seq >= 0 ? ' ' + m.seq : ''}: ${m.message}`);
      return;
    }
    if (m.type === 'serialized') {
      const p = this._serials.get(m.id);
      if (!p) return;
      this._serials.delete(m.id);
      if (m.error) { p.reject(new Error(m.error)); return; }
      const d = m.data;
      if (d && d.bodies) d.bodies.userData = this._mirror.userData.slice(0, d.bodies.count);
      p.resolve(d);
    }
  }

  _onPoses(m) {
    const P = this.perf;
    if (m.seq > this._answered) { this._answered = m.seq; this._inFlight = Math.max(0, this._inFlight - 1); P.inFlight = this._inFlight; }
    // the set on display goes back to the worker with the next step
    this._spare.push(this._poses, this._aabbs, this._cbuf);
    this._poses = m.buffer; this._aabbs = m.aabbs; this._cbuf = m.contacts;
    this._snapSeq = m.seq; this._snapTime = m.time; this._snapCount = m.count; this.stepCount = m.stepCount;
    this._contactTotal = m.contactTotal; this._stats = m.stats || this._stats;
    // returned command buffers: keep only full-size ones, so the writer never regrows on a smaller one
    if (m.cmd) for (let i = 0; i < m.cmd.length; i++) if (m.cmd[i].length >= this._cmdMax) this._cmdSpare.push(m.cmd[i]);
    // patches the worker had not seen when it wrote this snapshot survive into it
    const S = this._poses, A = this._aabbs, R = this._ov, RA = this._ovA, seqOf = this._ovSeq, L = this._ovList;
    let k = 0;
    for (let i = 0; i < this._ovCount; i++) {
      const id = L[i];
      if (seqOf[id] <= m.seq) { seqOf[id] = 0; continue; }
      L[k++] = id;
      const o = id * POSE_STRIDE, a = id * 6;
      if ((R[o + 13] & POSE_PENDING) && m.seq >= this._addSeq[id] && (S[o + 13] & POSE_ALIVE)) {
        // the worker has the body now (a later patch, e.g. setPose, is still ahead of it): take its AABB, moved to
        // the patched position, and stop treating the body as pending
        const dx = R[o] - S[o], dy = R[o + 1] - S[o + 1], dz = R[o + 2] - S[o + 2];
        RA[a] = A[a] + dx; RA[a + 1] = A[a + 1] + dy; RA[a + 2] = A[a + 2] + dz; RA[a + 3] = A[a + 3] + dx; RA[a + 4] = A[a + 4] + dy; RA[a + 5] = A[a + 5] + dz;
        R[o + 13] = R[o + 13] & ~POSE_PENDING;
      }
      for (let j = 0; j < POSE_STRIDE; j++) S[o + j] = R[o + j];
      for (let j = 0; j < 6; j++) A[a + j] = RA[a + j];
    }
    this._ovCount = k;
    this._gen++;
    const lat = performance.now() - this._postT[m.seq & 7];
    P.latency = P.snapshots ? P.latency * 0.9 + lat * 0.1 : lat;
    P.stepMs = P.snapshots ? P.stepMs * 0.9 + m.stepMs * 0.1 : m.stepMs;
    P.snapshots++;
    if (this.onSnapshot) this.onSnapshot(this);
  }

  /** Post queued commands (and owed steps when a slot is free); step() calls it. Use it to send commands while paused. */
  flush() {
    if (this.world || !this._ready || !this.worker || this.disposed) return this;
    this._flushCmds();
    this._pump();
    return this;
  }

  _flushCmds() {
    const w = this._cw;
    if (!w.len) return;
    const m = this._cmdMsg, buf = w.buf, cap = buf.length;
    m.seq = this._seq + 1; m.buffer = buf; m.length = w.len;
    this.perf.cmdBytes = w.len * 8;
    this._cmdTr[0] = buf.buffer;
    this.worker.postMessage(m, this._cmdTr);
    m.buffer = null; this._cmdTr[0] = null;
    if (cap > this._cmdMax) this._cmdMax = cap;
    let next = this._cmdSpare.pop();
    while (next && next.length < this._cmdMax) next = this._cmdSpare.pop(); // spares from before the writer grew
    if (!next) { next = new Float64Array(this._cmdMax); this.perf.cmdAllocs++; }
    w.reset(next);
  }

  _pump() {
    if (!this._owed) return;
    if (this._inFlight >= MAX_IN_FLIGHT) { this.perf.late++; return; }
    const m = this._msg, tr = this._tr, sp = this._spare;
    m.dt = this._owedDt; m.n = this._owed; m.seq = ++this._seq;
    tr.length = 0;
    for (let i = 0; i < sp.length; i++) tr.push(sp[i].buffer);
    m.bufs = sp.length ? sp : null;
    this._postT[m.seq & 7] = performance.now();
    this.worker.postMessage(m, tr);
    sp.length = 0; tr.length = 0; m.bufs = null;
    this._owed = 0;
    this._inFlight++; this.perf.inFlight = this._inFlight;
  }

  // ================================================================ stepping
  /**
   * Advance the world by dt (s; the engine's fixed dt). In-thread: steps now. Worker: posts the commands issued so
   * far and a step message (see the step policy in the header); the snapshot arrives asynchronously.
   */
  step(dt) {
    if (!(dt > 0) || this.disposed) return this;
    if (this.world) {
      this.world.step(dt);
      this.time = this.world.time; this.stepCount = this.world.stepCount;
      return this;
    }
    if (!this.worker) { this._cw.reset(); return this; } // the worker died: nothing to step
    this._owedDt = dt;
    if (this._owed >= MAX_IN_FLIGHT) this.perf.skipped++; else { this._owed++; this.time += dt; }
    return this.flush();
  }

  // ================================================================ commands
  _alive(id) {
    if (this.disposed) return false;
    if (this.world) { const B = this.world.bodies; return id >= 0 && id < B.count && B.alive[id] === 1; }
    return id >= 0 && id < this._ids.count && this._ids.alive[id] === 1;
  }
  _check(id, who) {
    if (this.disposed) throw new Error(`DynamicsClient.${who}: disposed`);
    if (!this._alive(id)) throw new RangeError(`DynamicsClient.${who}: no body ${id}`);
  }
  /** One fixed-layout command: in-thread it runs now (through the worker's applier), else it is queued. */
  _cmd(op, n, a0, a1, a2, a3, a4, a5, a6, a7) {
    if (this.disposed) return true;
    if (this.world) {
      const a = this._a; a[0] = a0; a[1] = a1; a[2] = a2; a[3] = a3; a[4] = a4; a[5] = a5; a[6] = a6; a[7] = a7;
      applyOp(this.world, op, a, 0, this._ctx);
      return true;
    }
    this._cw.op(op, n, a0, a1, a2, a3, a4, a5, a6, a7);
    return false;
  }

  /**
   * Add a body (DynamicsWorld#addBody options; `shape` is a shapes.js shape; `userData` stays on this thread).
   * Returns its id synchronously; throws like DynamicsWorld.addBody on bad options or when maxBodies is reached.
   */
  addBody(desc = {}) {
    if (this.disposed) throw new Error('DynamicsClient.addBody: disposed');
    if (this.world) return this.world.addBody(desc);
    const shape = desc.shape;
    if (!shape || typeof shape.aabb !== 'function') throw new TypeError('DynamicsClient.addBody: a shapes.js shape is required');
    const t = typeof desc.type === 'number' ? desc.type : TYPE_OF[desc.type ?? 'dynamic'];
    if (t === undefined) throw new RangeError(`DynamicsClient.addBody: unknown type ${desc.type}`);
    if (t === BODY.DYNAMIC && shape.type === SHAPE.TRIMESH) throw new RangeError('DynamicsClient.addBody: trimesh shapes are static only');
    if (t === BODY.DYNAMIC) {
      const m = Number.isFinite(desc.mass) && desc.mass > 0 ? desc.mass : (desc.density ?? 1000) * shape.volume;
      if (!(m > 0) || !Number.isFinite(m)) throw new RangeError('DynamicsClient.addBody: a dynamic body needs a positive mass (or density and a shape with volume)');
    }
    const id = this._ids.alloc();
    if (id < 0) throw new RangeError(`DynamicsClient.addBody: maxBodies (${this.capacity}) reached`);
    // shape: the full descriptor the first time this object is sent, then its id only
    let sid = this._shapeIds.get(shape), sendShape;
    if (sid === undefined) {
      sid = this._sids.alloc();
      try { sendShape = shapeDescriptor(shape); } catch (e) { this._sids.free(sid); this._ids.free(id); throw e; }
      this._shapeIds.set(shape, sid); this._sidShape[sid] = shape; this._sidRefs[sid] = 0;
    }
    try { this._cw.addBody(id, sid, desc, sendShape); } catch (e) {
      if (!this._sidRefs[sid]) { this._shapeIds.delete(shape); this._sidShape[sid] = null; this._sids.free(sid); }
      this._ids.free(id);
      throw e;
    }
    this._sidRefs[sid]++; this._bodySid[id] = sid; this._addSeq[id] = this._seq + 1;
    const M = this._mirror;
    M.type[id] = t; M.shape[id] = shape; M.userData[id] = desc.userData ?? null;
    M.layer[id] = (desc.layer ?? 1) | 0; M.mask[id] = (desc.mask ?? -1) | 0;
    this._kinds[id] = desc.kind ?? 'rigid';
    // patch: the body shows at its start pose at once; no AABB until the worker computed one (an empty box)
    const o = this._ovBegin(id, true), R = this._ov, RA = this._ovA, i6 = id * 6;
    const p = desc.position, q = desc.quaternion;
    let qx = comp(q, 0, 0), qy = comp(q, 1, 0), qz = comp(q, 2, 0), qw = comp(q, 3, 1);
    const ql = Math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
    if (ql > 0) { qx /= ql; qy /= ql; qz /= ql; qw /= ql; } else { qx = 0; qy = 0; qz = 0; qw = 1; }
    R[o] = comp(p, 0, 0); R[o + 1] = comp(p, 1, 0); R[o + 2] = comp(p, 2, 0);
    R[o + 3] = qx; R[o + 4] = qy; R[o + 5] = qz; R[o + 6] = qw;
    R[o + 13] = POSE_ALIVE | POSE_PENDING | (t << POSE_TYPE_SHIFT) | (desc.noGravity ? FLAG.NO_GRAVITY : 0);
    RA[i6] = RA[i6 + 1] = RA[i6 + 2] = Infinity; RA[i6 + 3] = RA[i6 + 4] = RA[i6 + 5] = -Infinity;
    this._ovCommit(id);
    return id;
  }

  /** Remove a body (and the joints attached to it); its id is reused by a later addBody. False if there is none. */
  removeBody(id) {
    if (!this._alive(id)) return false;
    // DynamicsWorld.removeBody removes the attached joints: their client ids are released here too
    const J = this._jids;
    for (let j = 0; j < J.count; j++) if (J.alive[j] && (this._jointA[j] === id || this._jointB[j] === id)) J.free(j);
    if (this._cmd(OP.REMOVE_BODY, 1, id, 0, 0, 0, 0, 0, 0, 0)) return true;
    this._ids.free(id);
    const sid = this._bodySid[id];
    if (sid >= 0) {
      this._bodySid[id] = -1;
      if (--this._sidRefs[sid] <= 0) { this._sidRefs[sid] = 0; this._shapeIds.delete(this._sidShape[sid]); this._sidShape[sid] = null; this._sids.free(sid); }
    }
    this._mirror.userData[id] = null; this._mirror.shape[id] = null; this._kinds[id] = 'rigid';
    this._ovBegin(id, true);
    this._ovCommit(id);
    return true;
  }

  /** Teleport: (id, x, y, z[, qx, qy, qz, qw]) or (id, position, quaternion?) vectors; a missing part keeps its value. */
  setPose(id, a, b, c, d, e, f, g) {
    this._check(id, 'setPose');
    const P = poseArgs(a, b, c, d, e, f, g), x = P[0], y = P[1], z = P[2], qx = P[3], qy = P[4], qz = P[5], qw = P[6];
    if (this._cmd(OP.SET_POSE, 8, id, x, y, z, qx, qy, qz, qw)) return true;
    const o = this._ovBegin(id, false), R = this._ov;
    if (!Number.isNaN(x)) {
      const i = id * 6, RA = this._ovA, dx = x - R[o], dy = y - R[o + 1], dz = z - R[o + 2];
      RA[i] += dx; RA[i + 1] += dy; RA[i + 2] += dz; RA[i + 3] += dx; RA[i + 4] += dy; RA[i + 5] += dz;
      R[o] = x; R[o + 1] = y; R[o + 2] = z;
    }
    if (!Number.isNaN(qw)) {
      const l = Math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw) || 1;
      R[o + 3] = qx / l; R[o + 4] = qy / l; R[o + 5] = qz / l; R[o + 6] = qw / l;
    }
    R[o + 13] = R[o + 13] & ~FLAG.ASLEEP;
    this._ovCommit(id);
    return true;
  }

  /**
   * Velocity: (id, vx, vy, vz[, wx, wy, wz]) or (id, linear, angular?) vectors (angular: world space, rad/s); a missing
   * part keeps its value.
   */
  setVelocity(id, a, b, c, d, e, f) {
    this._check(id, 'setVelocity');
    const V = pairArgs(a, b, c, d, e, f, NaN), vx = V[0], vy = V[1], vz = V[2], wx = V[3], wy = V[4], wz = V[5];
    if (this._cmd(OP.SET_VELOCITY, 7, id, vx, vy, vz, wx, wy, wz, 0)) return true;
    if (this._mirror.type[id] === BODY.STATIC) return true;
    const o = this._ovBegin(id, false), R = this._ov;
    if (!Number.isNaN(vx)) { R[o + 7] = vx; R[o + 8] = vy; R[o + 9] = vz; }
    if (!Number.isNaN(wx)) { R[o + 10] = wx; R[o + 11] = wy; R[o + 12] = wz; }
    R[o + 13] = R[o + 13] & ~FLAG.ASLEEP;
    this._ovCommit(id);
    return true;
  }

  /** Impulse (N s) at the centre of mass or a world point: (id, impulse, point?) vectors or (id, px, py, pz, ix, iy, iz) like DynamicsWorld. */
  applyImpulse(id, a, b, c, d, e, f) {
    this._check(id, 'applyImpulse');
    if (typeof a === 'number') return this._cmd(OP.APPLY_IMPULSE, 7, id, +d || 0, +e || 0, +f || 0, a, +b || 0, +c || 0, 0), true; // scalar form: point first
    return this._cmd(OP.APPLY_IMPULSE, 7, id, comp(a, 0, 0), comp(a, 1, 0), comp(a, 2, 0), comp(b, 0, NaN), comp(b, 1, 0), comp(b, 2, 0), 0), true;
  }
  /** Force (N) for the next step: (id, fx, fy, fz[, px, py, pz]) or (id, force, point?); default at the centre of mass. */
  applyForce(id, a, b, c, d, e, f) {
    this._check(id, 'applyForce');
    const V = pairArgs(a, b, c, d, e, f, 0);
    return this._cmd(OP.APPLY_FORCE, 7, id, V[0], V[1], V[2], V[3], V[4], V[5], 0), true;
  }
  /** World-space torque (N m) for the next step: (id, tx, ty, tz) or (id, torque). */
  applyTorque(id, a, b, c) {
    this._check(id, 'applyTorque');
    const V = pairArgs(a, b, c, undefined, undefined, undefined, 0);
    return this._cmd(OP.APPLY_TORQUE, 4, id, V[0], V[1], V[2], 0, 0, 0, 0), true;
  }
  /** Pose a kinematic body reaches at the end of the next step; same argument forms as setPose. */
  setKinematicTarget(id, a, b, c, d, e, f, g) {
    this._check(id, 'setKinematicTarget');
    if (!this.world && this._mirror.type[id] !== BODY.KINEMATIC) throw new RangeError(`DynamicsClient.setKinematicTarget: body ${id} is not kinematic`);
    const P = poseArgs(a, b, c, d, e, f, g);
    return this._cmd(OP.SET_KINEMATIC_TARGET, 8, id, P[0], P[1], P[2], P[3], P[4], P[5], P[6]), true;
  }
  /** Wake a body (and its sleeping island). */
  wake(id) {
    if (!this._alive(id)) return false;
    if (this._cmd(OP.WAKE, 1, id, 0, 0, 0, 0, 0, 0, 0)) return true;
    const o = this._ovBegin(id, false); this._ov[o + 13] = this._ov[o + 13] & ~FLAG.ASLEEP; this._ovCommit(id);
    return true;
  }
  /** Put one dynamic body to sleep (velocities zeroed). */
  sleep(id) {
    if (!this._alive(id)) return false;
    if (this._cmd(OP.SLEEP, 1, id, 0, 0, 0, 0, 0, 0, 0)) return true;
    if (this._mirror.type[id] !== BODY.DYNAMIC) return true;
    const o = this._ovBegin(id, false), R = this._ov;
    R[o + 13] = R[o + 13] | FLAG.ASLEEP;
    R.fill(0, o + 7, o + 13);
    this._ovCommit(id);
    return true;
  }
  /** Gravity: (x, y, z) or one vector. */
  setGravity(x, y, z) {
    if (typeof x === 'object' && x !== null) { z = comp(x, 2, 0); y = comp(x, 1, 0); x = comp(x, 0, 0); }
    return this._cmd(OP.SET_GRAVITY, 3, +x || 0, +y || 0, +z || 0, 0, 0, 0, 0, 0), true;
  }

  /**
   * Add a joint (DynamicsWorld#addJoint / joints.js descriptor); returns the client's joint id synchronously
   * (-1 when full or, in-thread, when the world refuses it).
   */
  addJoint(desc = {}) {
    if (this.disposed) throw new Error('DynamicsClient.addJoint: disposed');
    const a = desc.bodyA ?? -1, b = desc.bodyB ?? -1; // joints.js: null / undefined = WORLD_BODY
    const ok = (k) => k === -1 || (Number.isInteger(k) && this._alive(k));
    if (!ok(a) || !ok(b) || a === b) throw new RangeError(`DynamicsClient.addJoint: invalid bodies ${a}, ${b}`);
    const id = this._jids.alloc();
    if (id < 0) { this._error('addJoint: maxJoints reached'); return -1; }
    if (this.world) {
      let jid;
      try { jid = this.world.addJoint(desc); } catch (e) { this._jids.free(id); throw e; }
      if (!(jid >= 0)) { this._jids.free(id); return -1; }
      this._ctx.jointMap[id] = jid;
    } else {
      try { this._cw.addJoint(id, desc); } catch (e) { this._jids.free(id); throw e; }
    }
    this._jointA[id] = typeof a === 'number' ? a : -1; this._jointB[id] = typeof b === 'number' ? b : -1;
    return id;
  }
  /** Remove a joint by the id addJoint returned. False if there is none (removed with one of its bodies). */
  removeJoint(id) {
    if (!this._jids.free(id)) return false;
    this._cmd(OP.REMOVE_JOINT, 1, id, 0, 0, 0, 0, 0, 0, 0);
    return true;
  }

  /** The body's `kind` (addBody option). */
  kindOf(id) { return this.world ? this.world.kindOf(id) : this._kinds[id]; }

  /** World state as DynamicsWorld#serialize() gives it (worker: after every command and step posted so far). */
  serialize() {
    if (this.world) return Promise.resolve(this.world.serialize());
    if (!this.worker || this.disposed) return Promise.reject(new Error('DynamicsClient: no world'));
    return this.ready.then(() => {
      if (this.world) return this.world.serialize();
      this.flush();
      const id = ++this._serialId;
      return new Promise((resolve, reject) => { this._serials.set(id, { resolve, reject }); this.worker.postMessage({ type: 'serialize', id }); });
    });
  }

  // ================================================================ snapshot patches (worker mode)
  /** Start / extend a main-thread patch of body `id`'s snapshot record; `fresh` starts empty. Returns its offset. */
  _ovBegin(id, fresh) {
    const o = id * POSE_STRIDE, a = id * 6, R = this._ov, RA = this._ovA;
    if (fresh) { R.fill(0, o, o + POSE_STRIDE); R[o + 6] = 1; RA.fill(0, a, a + 6); } else if (!this._ovSeq[id]) {
      if (id < this._snapCount) { R.set(this._poses.subarray(o, o + POSE_STRIDE), o); RA.set(this._aabbs.subarray(a, a + 6), a); } else { R.fill(0, o, o + POSE_STRIDE); RA.fill(0, a, a + 6); }
    }
    if (!this._ovSeq[id]) this._ovList[this._ovCount++] = id;
    this._ovSeq[id] = this._seq + 1; // the step that carries the command
    return o;
  }
  _ovCommit(id) {
    const o = id * POSE_STRIDE, a = id * 6;
    this._poses.set(this._ov.subarray(o, o + POSE_STRIDE), o);
    this._aabbs.set(this._ovA.subarray(a, a + 6), a);
    this._gen++;
  }
  /** Record bits of `id` in the current snapshot (0 = no body). */
  _bits(id) {
    if (!(id >= 0 && id < this.capacity) || this.disposed) return 0;
    if (id >= this._snapCount && !this._ovSeq[id]) return 0;
    return this._poses[id * POSE_STRIDE + 13];
  }
  /** Refresh the SnapshotBodies arrays from the snapshot (once per snapshot / patch). */
  _sync() {
    if (this._syncGen === this._gen) return;
    this._syncGen = this._gen;
    const M = this._mirror, S = this._poses, A = this._aabbs, n = this._ids.count;
    const pos = M._pos, quat = M._quat, vel = M._vel, ang = M._angVel, ab = M._aabb, flags = M._flags;
    for (let id = 0; id < n; id++) {
      const o = id * POSE_STRIDE, i3 = id * 3, i4 = id * 4, a = id * 6;
      const bits = this._bits(id);
      if (!(bits & POSE_ALIVE)) {
        pos.fill(0, i3, i3 + 3); vel.fill(0, i3, i3 + 3); ang.fill(0, i3, i3 + 3); quat.fill(0, i4, i4 + 3); quat[i4 + 3] = 1;
        ab.fill(0, a, a + 6); flags[id] = 0;
        continue;
      }
      pos[i3] = S[o]; pos[i3 + 1] = S[o + 1]; pos[i3 + 2] = S[o + 2];
      quat[i4] = S[o + 3]; quat[i4 + 1] = S[o + 4]; quat[i4 + 2] = S[o + 5]; quat[i4 + 3] = S[o + 6];
      vel[i3] = S[o + 7]; vel[i3 + 1] = S[o + 8]; vel[i3 + 2] = S[o + 9];
      ang[i3] = S[o + 10]; ang[i3 + 1] = S[o + 11]; ang[i3 + 2] = S[o + 12];
      for (let k = 0; k < 6; k++) ab[a + k] = A[a + k];
      flags[id] = bits & 255;
    }
  }

  // ================================================================ reads
  /**
   * The PhysicsWorld the dynamics world collides with (bridge.js sets it when unset). In-thread: the world's field.
   * Worker mode: kept here; the worker got a copy of its static boxes at construction.
   */
  get physics() { return this.world ? this.world.physics : this._physics; }
  set physics(v) { if (this.world) this.world.physics = v; else this._physics = v; }
  /** Body state: the world's BodyStore in-thread, a Float32 snapshot mirror in worker mode (same field names). */
  get bodies() { return this.world ? this.world.bodies : this._mirror; }
  /** Contacts of the last step: the world's ContactBuffer in-thread, a snapshot copy in worker mode (lambdaN = impulse). */
  get contacts() {
    if (this.world) return this.world.contacts;
    const C = this._contacts;
    if (!C) return null;
    if (this._cgen !== this._gen) {
      this._cgen = this._gen;
      const S = this._cbuf, m = Math.min(C.capacity, S.length ? S[0] | 0 : 0);
      for (let i = 0; i < m; i++) {
        const o = 1 + i * CONTACT_STRIDE;
        C.a[i] = S[o]; C.b[i] = S[o + 1]; C.px[i] = S[o + 2]; C.py[i] = S[o + 3]; C.pz[i] = S[o + 4];
        C.nx[i] = S[o + 5]; C.ny[i] = S[o + 6]; C.nz[i] = S[o + 7]; C.depth[i] = S[o + 8]; C.lambdaN[i] = S[o + 9];
      }
      C.count = m; C.total = this._contactTotal; C.overflow = Math.max(0, this._contactTotal - m);
    }
    return C;
  }
  /** The world's JointStore in-thread; null in worker mode (joints live in the worker). */
  get joints() { return this.world ? this.world.joints : null; }
  /** Debug line segments (dynamics/debug.js; worker mode: from the snapshot, no joints). Returns the count written. */
  debugLines(out, max, opts, colors) { return this.world ? this.world.debugLines(out, max, opts, colors) : writeDebugLines(this, out, max, opts, colors); }
  /** World stats (worker: as of the latest snapshot). */
  get stats() { return this.world ? this.world.stats : this._stats; }
  /** ms between posting a step and receiving its snapshot (EMA; 0 in-thread). */
  get latency() { return this.perf.latency; }
  /** Simulation time of the latest snapshot (in-thread: the world's time). */
  get snapshotTime() { return this.world ? this.world.time : this._snapTime; }
  /** Step seq of the latest snapshot (worker mode; in-thread: the world's step count). */
  get snapshotSeq() { return this.world ? this.world.stepCount : this._snapSeq; }

  /**
   * Pose of body `id` into outPos ({x,y,z} / Vector3 / array) and outQuat ({x,y,z,w} / Quaternion / array). Worker
   * mode: the latest snapshot advanced by its velocities over `ahead` s (default: the time since the snapshot when
   * `extrapolate`, capped at maxExtrapolation; just-added bodies are not extrapolated). In-thread: the current state
   * (advanced by `ahead` if given). Returns false (outputs untouched) when there is no such body.
   */
  getPose(id, outPos, outQuat, ahead) {
    let x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz, h;
    if (this.world) {
      if (!this._alive(id)) return false;
      const B = this.world.bodies, i3 = id * 3, i4 = id * 4;
      x = B.pos[i3]; y = B.pos[i3 + 1]; z = B.pos[i3 + 2]; qx = B.quat[i4]; qy = B.quat[i4 + 1]; qz = B.quat[i4 + 2]; qw = B.quat[i4 + 3];
      vx = B.vel[i3]; vy = B.vel[i3 + 1]; vz = B.vel[i3 + 2]; wx = B.angVel[i3]; wy = B.angVel[i3 + 1]; wz = B.angVel[i3 + 2];
      h = ahead || 0;
    } else {
      const bits = this._bits(id);
      if (!(bits & POSE_ALIVE)) return false;
      const S = this._poses, o = id * POSE_STRIDE;
      x = S[o]; y = S[o + 1]; z = S[o + 2]; qx = S[o + 3]; qy = S[o + 4]; qz = S[o + 5]; qw = S[o + 6];
      vx = S[o + 7]; vy = S[o + 8]; vz = S[o + 9]; wx = S[o + 10]; wy = S[o + 11]; wz = S[o + 12];
      if (ahead !== undefined) h = ahead;
      else if (!this.extrapolate || (bits & POSE_PENDING)) h = 0;
      else h = Math.min(this.maxExtrapolation, Math.max(0, this.time - this._snapTime));
    }
    if (h) {
      x += vx * h; y += vy * h; z += vz * h;
      const q = advanceQuat(qx, qy, qz, qw, wx, wy, wz, h);
      qx = q[0]; qy = q[1]; qz = q[2]; qw = q[3];
    }
    out3(outPos, x, y, z);
    out4(outQuat, qx, qy, qz, qw);
    return true;
  }

  /** Linear and angular (world) velocity of body `id` (snapshot values in worker mode). False when there is none. */
  getVelocity(id, outLinear, outAngular) {
    if (this.world) {
      if (!this._alive(id)) return false;
      const B = this.world.bodies, i3 = id * 3;
      out3(outLinear, B.vel[i3], B.vel[i3 + 1], B.vel[i3 + 2]); out3(outAngular, B.angVel[i3], B.angVel[i3 + 1], B.angVel[i3 + 2]);
      return true;
    }
    if (!(this._bits(id) & POSE_ALIVE)) return false;
    const S = this._poses, o = id * POSE_STRIDE;
    out3(outLinear, S[o + 7], S[o + 8], S[o + 9]); out3(outAngular, S[o + 10], S[o + 11], S[o + 12]);
    return true;
  }

  /** Whether body `id` exists and is not asleep. */
  isAwake(id) {
    if (this.world) return this.world.isAwake(id);
    const bits = this._bits(id);
    return (bits & POSE_ALIVE) !== 0 && (bits & FLAG.ASLEEP) === 0;
  }

  /**
   * PhysicsWorld dynamic provider (also `colliderProvider`; bound): APPENDS records of the alive non-static bodies
   * whose AABB footprint is within r of (x, z) to `out`: { minX, minY, maxX, maxY, minZ, maxZ, kind, id, vx, vz }
   * (pooled per body, read only), ascending id. Worker mode: from the snapshot (linear scan, like Traffic); in-thread:
   * DynamicsWorld#queryColliders.
   */
  queryColliders(x, z, r, out = []) {
    if (!out) out = [];
    if (this.disposed) return out;
    if (this.world) return this.world.queryColliders(x, z, r, out);
    if (!Number.isFinite(x) || !Number.isFinite(z) || this.disposed) return out;
    const rr = r > 0 ? r * r : 0, S = this._poses, A = this._aabbs, T = this._mirror.type;
    const n = Math.min(this._ids.count, this.capacity);
    for (let id = 0; id < n; id++) {
      const bits = this._bits(id);
      if (!(bits & POSE_ALIVE) || (bits & POSE_PENDING) || T[id] === BODY.STATIC) continue;
      const a = id * 6, minX = A[a], minZ = A[a + 2], maxX = A[a + 3], maxZ = A[a + 5];
      const dx = x < minX ? minX - x : x > maxX ? x - maxX : 0;
      const dz = z < minZ ? minZ - z : z > maxZ ? z - maxZ : 0;
      if (dx * dx + dz * dz > rr) continue;
      let rec = this._recs[id];
      if (rec === undefined) {
        while (this._recs.length < id) this._recs.push(undefined); // keep the pool packed (ids are dense)
        rec = this._recs[id] = { minX: 0, minY: 0, maxX: 0, maxY: 0, minZ: 0, maxZ: 0, kind: 'rigid', id, vx: 0, vz: 0 };
      }
      rec.minX = minX; rec.minY = A[a + 1]; rec.minZ = minZ; rec.maxX = maxX; rec.maxY = A[a + 4]; rec.maxZ = maxZ;
      rec.kind = this._kinds[id]; rec.vx = S[id * POSE_STRIDE + 7]; rec.vz = S[id * POSE_STRIDE + 9];
      out.push(rec);
    }
    return out;
  }

  // ================================================================ errors / teardown
  _flushCtxErrors() {
    const e = this._ctx.errors;
    if (!e.length) return;
    this._error(e.join('; '));
    e.length = 0;
  }
  _error(message) {
    this.lastError = message;
    if (this.onError) this.onError(message);
    if (!this._errWarned) { this._errWarned = true; console.warn('[DynamicsClient]', message); }
  }

  /**
   * Like DynamicsWorld#dispose: unpublish the bodies from the PhysicsWorld (colliderProvider and any bridge.js
   * provider attached to this client), terminate the worker or dispose the in-thread world, drop the buffers.
   * `disposed` is set first (bridge providers check it); reads return nothing afterwards.
   */
  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    const pw = this.physics;
    if (pw && typeof pw.removeDynamicProvider === 'function') { pw.removeDynamicProvider(this.colliderProvider); detachFromPhysicsWorld(this, pw); }
    if (this._timer) { clearTimeout(this._timer); this._timer = 0; }
    if (this.worker) { try { this.worker.terminate(); } catch { /* gone */ } this.worker = null; }
    for (const p of this._serials.values()) p.reject(new Error('DynamicsClient disposed'));
    this._serials.clear();
    if (this._resolveReady) this._resolveReady();
    if (this.world) { this.world.dispose(); this.world = null; }
    this._physics = null;
    this._poses = this._aabbs = this._cbuf = this._ov = this._ovA = null;
    this._spare = this._cmdSpare = this._cw = this._mirror = this._contacts = this._recs = null;
    this._inFlight = 0; this.perf.inFlight = 0;
  }
}

function warnOnce(msg) {
  if (_warned) return;
  _warned = true;
  console.warn('[DynamicsClient]', msg);
}
