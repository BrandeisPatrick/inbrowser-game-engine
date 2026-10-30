// Dynamics Web Worker + the wire protocol between it and DynamicsClient (module worker; DynamicsClient.js creates it
// with  new Worker(new URL('./dynamics.worker.js', import.meta.url), { type: 'module' })  -> bundled by Vite).
// The worker owns a DynamicsWorld and steps it; the main thread keeps a read-only snapshot of the poses.
//
// Messages, main -> worker:
//   { type: 'init', options, seed, serialized?, physics?, namedLayers, contactCapacity }
//        -> new DynamicsWorld(options) or DynamicsWorld.from(serialized); answers { type: 'ready', capacity, count, time,
//        stepCount }. `physics` = staticWorldOf(physicsWorld): the main thread's PhysicsWorld static boxes (same order,
//        so static contact indices match) + groundLevel / stepHeight / cell size, rebuilt here as a PhysicsWorld;
//        `namedLayers` = PhysicsWorld.namedLayers(), installed (setNamedLayers) first so layerOf agrees across threads
//   { type: 'cmd', seq, buffer: Float64Array (transferred), length }   command stream (below), applied on receipt, in
//        order, before step `seq`; the buffer comes back in the next 'poses' message (`cmd`)
//   { type: 'step', dt, n, seq, bufs: [poses, aabbs, contacts]* | null }   n (1 or 2) world.step(dt) calls; `bufs`
//        returns snapshot buffers the main thread no longer reads (transferred), so the worker reuses them
//   { type: 'serialize', id }  -> { type: 'serialized', id, data: world.serialize() }
// worker -> main:
//   { type: 'poses', seq, stepCount, time, count, buffer, aabbs, contacts, contactTotal, stats, stepMs, cmd, sets, error }
//        one answer per step message; buffer / aabbs / contacts / cmd are transferred (double-buffered: the main thread
//        hands the previous set back with its next step); `sets` = snapshot sets this worker has allocated. `error`
//        (null when fine): world.step threw; the remaining steps of that message were not run, the snapshot is the
//        state after the throw and the worker keeps going
//   { type: 'error', phase: 'init' | 'cmd' | 'step' | 'serialize', message, seq }   a throwing init / command batch /
//        message. phase 'step' = step `seq` could not be answered with a snapshot (it is that step's answer)
//
// Snapshot layouts (Float32Array, sized by the world's maxBodies so one buffer fits every snapshot):
//   buffer   POSE_STRIDE = 14 floats per body id: x y z  qx qy qz qw  vx vy vz  wx wy wz  bits
//            bits = FLAG.* (bits 0..7) | POSE_ALIVE (256) | type << 10 (BODY.*); 0 = no body. Ids < count are written.
//   aabbs    6 floats per body id: minX minY minZ maxX maxY maxZ (BodyStore.aabb order, margin included)
//   contacts [count, (a, b, px, py, pz, nx, ny, nz, depth, lambdaN) * count]  (CONTACT_STRIDE = 10; lambdaN = the
//            step's normal impulse, N s; at most contactCapacity contacts, `contactTotal` = how many the world had;
//            b = WORLD_BODY (-1) for the static world)
//
// Command stream (Float64Array): records [opcode, argCount, args...]; argCount lets a reader skip a record.
//   ADD_BODY      id, sid, desc...   id assigned on the main thread (BodyStore rule: lowest free id first; the worker
//                 checks world.addBody's id against it). desc = encodeValue(addBody options without userData); its
//                 `shape` is shapeDescriptor(shape) (DynamicsWorld's describeShape, long lists as typed arrays) the
//                 first time a shape object is sent (registered as shape `sid`, rebuilt bit-identically with
//                 shapeFromDescriptor) and absent afterwards (bodies sharing a shape
//                 object on the main thread share one here). Both sides refcount sids by body and drop a shape when
//                 its last body is removed.
//   REMOVE_BODY   id              (the joints attached to it go too, on both sides)
//   SET_POSE      id, x, y, z, qx, qy, qz, qw              NaN in x (position) or qw (rotation) = keep that part
//   SET_VELOCITY  id, vx, vy, vz, wx, wy, wz               NaN in vx / wx = keep that part
//   APPLY_IMPULSE id, ix, iy, iz, px, py, pz               point NaN = at the body origin (centre of mass)
//   APPLY_FORCE   id, fx, fy, fz, px, py, pz               point NaN = at the centre of mass
//   APPLY_TORQUE  id, tx, ty, tz
//   SET_KINEMATIC_TARGET id, x, y, z, qx, qy, qz, qw       NaN qw = keep the rotation
//   ADD_JOINT     id, desc...      id = the client's joint id (same allocation rule); mapped to the world's joint id
//   REMOVE_JOINT  id
//   WAKE id | SLEEP id | SET_GRAVITY gx, gy, gz
// Values (encodeValue / decodeValue): a tagged tree, so any plain descriptor round-trips exactly (numbers bit-exact,
// NaN / -0 included): UNDEF | NULL | NUM v | BOOL b | STR len chars... | ARR len values... | OBJ n (keylen keychars...
// value)* | TYPED kind len values... (Float64/32, Int32/16/8, Uint32/16/8/8Clamped) | FN (a function field: decodes
// as null). three.js Vector3 / Quaternion / Euler encode as plain { x, y, z (, w | order) }; other class instances
// as their own enumerable fields.
//
// The same module is imported by DynamicsClient on the main thread (codec, snapshot writer, command applier for the
// in-thread fallback), so it binds `self.onmessage` only inside a real worker scope (WorkerGlobalScope), never on a
// page. Node tests drive createDynamicsWorkerHandler(post) through node:worker_threads.
import { DynamicsWorld, describeShape, shapeFromDescriptor } from './dynamics/DynamicsWorld.js';
import { PhysicsWorld, namedLayers, setNamedLayers } from './PhysicsWorld.js';

/** Floats per body in the poses buffer. */
export const POSE_STRIDE = 14;
/** `bits` (float 13 of a pose record): the body exists in this snapshot. */
export const POSE_ALIVE = 256;
/** `bits`: set by the main thread on a record it patched ahead of the worker (just added: no AABB yet). */
export const POSE_PENDING = 512;
/** `bits >> POSE_TYPE_SHIFT & 3` = BODY.* type. */
export const POSE_TYPE_SHIFT = 10;
/** Floats per contact in the contacts buffer (after the leading count). */
export const CONTACT_STRIDE = 10;
/** Default contacts carried per snapshot. */
export const DEFAULT_SNAPSHOT_CONTACTS = 2048;

/** Command opcodes. */
export const OP = Object.freeze({
  ADD_BODY: 1, REMOVE_BODY: 2, SET_POSE: 3, SET_VELOCITY: 4, APPLY_IMPULSE: 5, APPLY_FORCE: 6, APPLY_TORQUE: 7,
  SET_KINEMATIC_TARGET: 8, ADD_JOINT: 9, REMOVE_JOINT: 10, WAKE: 11, SLEEP: 12, SET_GRAVITY: 13,
});
/** Fixed argument counts by opcode (-1 = variable: ids + encoded descriptor). */
export const OP_ARGS = Object.freeze({ 1: -1, 2: 1, 3: 8, 4: 7, 5: 7, 6: 7, 7: 4, 8: 8, 9: -1, 10: 1, 11: 1, 12: 1, 13: 3 });

// ------------------------------------------------------------------ value codec
const T_UNDEF = 0, T_NULL = 1, T_NUM = 2, T_BOOL = 3, T_STR = 4, T_ARR = 5, T_OBJ = 6, T_TYPED = 7, T_FN = 8;
const TYPED = [Float64Array, Float32Array, Int32Array, Uint32Array, Int16Array, Uint16Array, Int8Array, Uint8Array, Uint8ClampedArray];
const NO_KEY = {};

/** Growable Float64 command writer; the stream is buf[0 .. len). reset(buf) continues on a recycled buffer. */
export class CommandWriter {
  constructor(capacity = 1024) { this.buf = new Float64Array(capacity); this.len = 0; this.records = 0; this.grows = 0; }
  reset(buf) { if (buf) this.buf = buf; this.len = 0; this.records = 0; return this; }
  ensure(n) {
    if (this.len + n <= this.buf.length) return;
    let cap = Math.max(64, this.buf.length * 2);
    while (cap < this.len + n) cap *= 2;
    const b = new Float64Array(cap);
    b.set(this.buf.subarray(0, this.len));
    this.buf = b; this.grows++;
  }
  push(v) { this.ensure(1); this.buf[this.len++] = v; }
  /** One fixed-size record: opcode, n args (the first n of a0..a7). */
  op(code, n, a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0) {
    this.ensure(n + 2);
    const b = this.buf, o = this.len;
    b[o] = code; b[o + 1] = n;
    if (n > 0) b[o + 2] = a0; if (n > 1) b[o + 3] = a1; if (n > 2) b[o + 4] = a2; if (n > 3) b[o + 5] = a3;
    if (n > 4) b[o + 6] = a4; if (n > 5) b[o + 7] = a5; if (n > 6) b[o + 8] = a6; if (n > 7) b[o + 9] = a7;
    this.len = o + n + 2; this.records++;
  }
  /** ADD_BODY record: id, sid, desc without `userData`, its `shape` replaced by `shapeValue` (undefined = omitted). */
  addBody(id, sid, desc, shapeValue) {
    const start = this.len;
    this.ensure(4);
    this.buf[start] = OP.ADD_BODY; this.buf[start + 2] = id; this.buf[start + 3] = sid; this.len += 4;
    encodeObject(this, desc, 'userData', 'shape', shapeValue);
    this.buf[start + 1] = this.len - start - 2;
    this.records++;
  }
  /** ADD_JOINT record: id, desc. */
  addJoint(id, desc) {
    const start = this.len;
    this.ensure(3);
    this.buf[start] = OP.ADD_JOINT; this.buf[start + 2] = id; this.len += 3;
    encodeValue(this, desc);
    this.buf[start + 1] = this.len - start - 2;
    this.records++;
  }
}

/** Append the tagged encoding of `v` to writer `w` (see the header). A top-level function throws. */
export function encodeValue(w, v, top = true) {
  if (v === undefined) return w.push(T_UNDEF);
  if (v === null) return w.push(T_NULL);
  const t = typeof v;
  if (t === 'number') { w.ensure(2); w.buf[w.len++] = T_NUM; w.buf[w.len++] = v; return; }
  if (t === 'boolean') { w.ensure(2); w.buf[w.len++] = T_BOOL; w.buf[w.len++] = v ? 1 : 0; return; }
  if (t === 'string') { w.ensure(1); w.buf[w.len++] = T_STR; writeChars(w, v); return; }
  if (t === 'function' && !top) return w.push(T_FN);
  if (t !== 'object') throw new TypeError(`dynamics: cannot encode a ${t}`);
  if (ArrayBuffer.isView(v)) {
    const k = TYPED.indexOf(v.constructor);
    if (k < 0) throw new TypeError(`dynamics: cannot encode a ${v.constructor.name}`);
    w.ensure(3 + v.length); w.buf[w.len++] = T_TYPED; w.buf[w.len++] = k; w.buf[w.len++] = v.length;
    w.buf.set(v, w.len); w.len += v.length;
    return;
  }
  if (Array.isArray(v)) {
    w.ensure(2); w.buf[w.len++] = T_ARR; w.buf[w.len++] = v.length;
    for (let i = 0; i < v.length; i++) encodeValue(w, v[i], false);
    return;
  }
  // three.js math objects: private fields (_x) / flags; send the public values
  if (v.isQuaternion) return encodeObject(w, { x: v.x, y: v.y, z: v.z, w: v.w });
  if (v.isEuler) return encodeObject(w, { x: v.x, y: v.y, z: v.z, order: v.order });
  if (v.isVector3) return encodeObject(w, { x: v.x, y: v.y, z: v.z });
  if (v.isVector2) return encodeObject(w, { x: v.x, y: v.y });
  encodeObject(w, v);
}
/** Object `v` without key `skip`, with key `repl` (if present in v) encoded as `replValue` (undefined: left out). */
function encodeObject(w, v, skip = NO_KEY, repl = NO_KEY, replValue) {
  const keys = Object.keys(v);
  let n = 0;
  for (let i = 0; i < keys.length; i++) if (keys[i] !== skip && !(keys[i] === repl && replValue === undefined)) n++;
  w.ensure(2); w.buf[w.len++] = T_OBJ; w.buf[w.len++] = n;
  for (let i = 0; i < keys.length; i++) {
    const key = keys[i];
    if (key === skip) continue;
    if (key === repl) { if (replValue === undefined) continue; writeChars(w, key); encodeValue(w, replValue, false); continue; }
    writeChars(w, key);
    encodeValue(w, v[key], false);
  }
}
function writeChars(w, s) {
  w.ensure(1 + s.length); w.buf[w.len++] = s.length;
  for (let c = 0; c < s.length; c++) w.buf[w.len++] = s.charCodeAt(c);
}

/** Decode one value from reader { buf, pos } (advances pos). */
export function decodeValue(r) {
  const b = r.buf, tag = b[r.pos++];
  switch (tag) {
    case T_UNDEF: return undefined;
    case T_NULL: case T_FN: return null;
    case T_NUM: return b[r.pos++];
    case T_BOOL: return b[r.pos++] !== 0;
    case T_STR: return readChars(r);
    case T_TYPED: {
      const k = b[r.pos++], n = b[r.pos++];
      const a = new TYPED[k](n);
      for (let i = 0; i < n; i++) a[i] = b[r.pos + i];
      r.pos += n;
      return a;
    }
    case T_ARR: { const n = b[r.pos++]; const a = new Array(n); for (let i = 0; i < n; i++) a[i] = decodeValue(r); return a; }
    case T_OBJ: { const n = b[r.pos++], o = {}; for (let i = 0; i < n; i++) { const key = readChars(r); o[key] = decodeValue(r); } return o; }
    default: throw new Error(`dynamics: bad value tag ${tag} at ${r.pos - 1}`);
  }
}
function readChars(r) {
  const b = r.buf, n = b[r.pos++];
  let s = '';
  for (let i = 0; i < n; i++) s += String.fromCharCode(b[r.pos + i]);
  r.pos += n;
  return s;
}

/**
 * Decode a command stream into [{ op, args }] (args: plain array; ADD_BODY gives [id, sid, desc], ADD_JOINT [id,
 * desc]). Allocates: for tests and debugging; the worker applies records in place with applyCommands.
 */
export function decodeCommands(buf, len = buf.length) {
  const out = [], r = { buf, pos: 0 };
  let o = 0;
  while (o < len) {
    const op = buf[o], n = buf[o + 1];
    if (op === OP.ADD_BODY) { r.pos = o + 4; out.push({ op, args: [buf[o + 2], buf[o + 3], decodeValue(r)] }); }
    else if (op === OP.ADD_JOINT) { r.pos = o + 3; out.push({ op, args: [buf[o + 2], decodeValue(r)] }); }
    else out.push({ op, args: Array.from(buf.subarray(o + 2, o + 2 + n)) });
    o += n + 2;
  }
  return out;
}

// ------------------------------------------------------------------ id mirror
/**
 * Id allocator with BodyStore's rule (dynamics/state.js: lowest free id first, else `count++`), so the main thread can
 * hand out body ids synchronously that equal the ids the worker's world assigns. Also used for joint and shape ids.
 */
export class IdAllocator {
  constructor(capacity = 4096) {
    this.capacity = capacity; this.count = 0;
    this.alive = new Uint8Array(capacity);
    this._free = new Int32Array(capacity); this._freeCount = 0;
  }
  alloc() {
    let id;
    if (this._freeCount > 0) id = this._free[--this._freeCount];
    else if (this.count < this.capacity) id = this.count++;
    else return -1;
    this.alive[id] = 1;
    return id;
  }
  free(id) {
    if (!(id >= 0 && id < this.count) || !this.alive[id]) return false;
    this.alive[id] = 0;
    const f = this._free; let i = this._freeCount++;
    while (i > 0 && f[i - 1] < id) { f[i] = f[i - 1]; i--; } // descending: the pop in alloc() returns the smallest
    f[i] = id;
    return true;
  }
  /** Adopt an alive[] / count state (a deserialized world's store). */
  adopt(alive, count) {
    this.alive.fill(0); this._freeCount = 0; this.count = count;
    for (let id = count - 1; id >= 0; id--) { if (alive[id]) this.alive[id] = 1; else this._free[this._freeCount++] = id; }
  }
}

// ------------------------------------------------------------------ applying commands (the only place that calls the world)
/** Per-world command state: client -> world joint ids, the shape table, errors of the last batch. */
export function createApplyContext() {
  return { jointMap: [], shapes: [], shapeRefs: [], bodySid: [], errors: [], reader: { buf: null, pos: 0 } };
}

/** Apply records buf[0 .. len) to `world` in order; failures are collected in ctx.errors (the rest still applies). */
export function applyCommands(world, buf, len, ctx) {
  let o = 0;
  while (o < len) {
    const op = buf[o], n = buf[o + 1];
    try { applyOp(world, op, buf, o + 2, ctx); } catch (e) { ctx.errors.push(`op ${op}: ${(e && e.message) || e}`); }
    o += n + 2;
  }
}

/** Apply one record whose args start at a[o]. */
export function applyOp(world, op, a, o, ctx) {
  const B = world.bodies;
  switch (op) {
    case OP.ADD_BODY: {
      const want = a[o], sid = a[o + 1], r = ctx.reader;
      r.buf = a; r.pos = o + 2;
      const desc = decodeValue(r);
      r.buf = null;
      if (sid >= 0) {
        if (desc.shape) ctx.shapes[sid] = shapeFromDescriptor(desc.shape);
        desc.shape = ctx.shapes[sid];
        ctx.shapeRefs[sid] = (ctx.shapeRefs[sid] || 0) + 1;
      }
      ctx.bodySid[want] = sid;
      const got = world.addBody(desc);
      if (got !== want) throw new Error(`addBody id ${got} != client id ${want} (id mirror out of sync)`);
      return;
    }
    case OP.REMOVE_BODY: {
      const id = a[o], sid = ctx.bodySid[id];
      if (sid !== undefined && sid >= 0) {
        ctx.bodySid[id] = -1;
        if (--ctx.shapeRefs[sid] <= 0) { ctx.shapeRefs[sid] = 0; ctx.shapes[sid] = undefined; }
      }
      // world.removeBody drops the joints attached to the body: forget their client ids first
      const J = world.joints, map = ctx.jointMap;
      if (J) for (let c = 0; c < map.length; c++) { const j = map[c]; if (j >= 0 && J.alive[j] && (J.a[j] === id || J.b[j] === id)) map[c] = -1; }
      world.removeBody(id);
      return;
    }
    case OP.SET_POSE: {
      const id = a[o], i3 = id * 3, i4 = id * 4;
      if (!(id >= 0 && id < B.count && B.alive[id])) throw new RangeError(`setPose: no body ${id}`);
      const keepP = Number.isNaN(a[o + 1]), keepQ = Number.isNaN(a[o + 7]);
      world.setPose(id,
        keepP ? B.pos[i3] : a[o + 1], keepP ? B.pos[i3 + 1] : a[o + 2], keepP ? B.pos[i3 + 2] : a[o + 3],
        keepQ ? B.quat[i4] : a[o + 4], keepQ ? B.quat[i4 + 1] : a[o + 5], keepQ ? B.quat[i4 + 2] : a[o + 6], keepQ ? B.quat[i4 + 3] : a[o + 7]);
      return;
    }
    case OP.SET_VELOCITY: {
      const id = a[o], i3 = id * 3;
      if (!(id >= 0 && id < B.count && B.alive[id])) throw new RangeError(`setVelocity: no body ${id}`);
      const keepV = Number.isNaN(a[o + 1]), keepW = Number.isNaN(a[o + 4]);
      world.setVelocity(id,
        keepV ? B.vel[i3] : a[o + 1], keepV ? B.vel[i3 + 1] : a[o + 2], keepV ? B.vel[i3 + 2] : a[o + 3],
        keepW ? B.angVel[i3] : a[o + 4], keepW ? B.angVel[i3 + 1] : a[o + 5], keepW ? B.angVel[i3 + 2] : a[o + 6]);
      return;
    }
    case OP.APPLY_IMPULSE: {
      const id = a[o], i3 = id * 3;
      if (!(id >= 0 && id < B.count && B.alive[id])) throw new RangeError(`applyImpulse: no body ${id}`);
      const com = Number.isNaN(a[o + 4]);
      // DynamicsWorld's scalar form is (id, point, impulse)
      world.applyImpulse(id, com ? B.pos[i3] : a[o + 4], com ? B.pos[i3 + 1] : a[o + 5], com ? B.pos[i3 + 2] : a[o + 6], a[o + 1], a[o + 2], a[o + 3]);
      return;
    }
    case OP.APPLY_FORCE:
      if (Number.isNaN(a[o + 4])) world.applyForce(a[o], a[o + 1], a[o + 2], a[o + 3]);
      else world.applyForce(a[o], a[o + 1], a[o + 2], a[o + 3], a[o + 4], a[o + 5], a[o + 6]);
      return;
    case OP.APPLY_TORQUE: world.applyTorque(a[o], a[o + 1], a[o + 2], a[o + 3]); return;
    case OP.SET_KINEMATIC_TARGET:
      if (Number.isNaN(a[o + 7])) world.setKinematicTarget(a[o], a[o + 1], a[o + 2], a[o + 3]);
      else world.setKinematicTarget(a[o], a[o + 1], a[o + 2], a[o + 3], a[o + 4], a[o + 5], a[o + 6], a[o + 7]);
      return;
    case OP.ADD_JOINT: {
      const r = ctx.reader; r.buf = a; r.pos = o + 1;
      const cid = a[o], desc = decodeValue(r);
      r.buf = null;
      ctx.jointMap[cid] = -1;
      const jid = world.addJoint(desc);
      ctx.jointMap[cid] = jid >= 0 ? jid : -1;
      if (!(jid >= 0)) throw new Error(`addJoint failed (client joint ${cid})`);
      return;
    }
    case OP.REMOVE_JOINT: {
      const cid = a[o], jid = ctx.jointMap[cid];
      ctx.jointMap[cid] = -1;
      if (jid !== undefined && jid >= 0) world.removeJoint(jid); // -1: already gone with one of its bodies
      return;
    }
    case OP.WAKE: world.wake(a[o]); return;
    case OP.SLEEP: world.sleep(a[o]); return;
    case OP.SET_GRAVITY: world.setGravity(a[o], a[o + 1], a[o + 2]); return;
    default: throw new Error(`unknown opcode ${op}`);
  }
}

// ------------------------------------------------------------------ shapes / static world
/** Plain descriptor of a shapes.js shape (DynamicsWorld's own, rebuilt bit-identically by shapeFromDescriptor). */
export { describeShape };

/**
 * Structured-clone friendly copy of a PhysicsWorld's static part: boxes (grid order), their kinds, groundLevel,
 * stepHeight, cell size. Throws when the PhysicsWorld has a heightfield (a function: it cannot reach a worker).
 */
export function staticWorldOf(pw) {
  if (pw._hf || pw.heightfield) throw new Error('the PhysicsWorld has a heightfield, which cannot be sent to a worker');
  const list = pw.grid ? pw.grid.boxes : [], n = list.length;
  const boxes = new Float64Array(n * 6), kinds = new Array(n);
  for (let i = 0; i < n; i++) {
    const b = list[i], o = i * 6;
    boxes[o] = b.minX; boxes[o + 1] = b.minY; boxes[o + 2] = b.maxX; boxes[o + 3] = b.maxY; boxes[o + 4] = b.minZ; boxes[o + 5] = b.maxZ;
    kinds[i] = b.kind;
  }
  return { boxes, kinds, cellSize: pw.grid ? pw.grid.cell : 32, groundLevel: pw.groundLevel, stepHeight: pw.stepHeight };
}

/** Rebuild staticWorldOf()'s copy as a PhysicsWorld (install the main thread's namedLayers() first). */
export function physicsFromStatic(s) {
  const pw = new PhysicsWorld({ cellSize: s.cellSize, groundLevel: s.groundLevel, stepHeight: s.stepHeight });
  const B = s.boxes;
  for (let i = 0, n = B.length / 6; i < n; i++) { const o = i * 6; pw.addStaticBox(B[o], B[o + 1], B[o + 2], B[o + 3], B[o + 4], B[o + 5], s.kinds[i]); }
  return pw;
}

/** The main thread's named-kind layer table, for the init message. */
export { namedLayers };

/**
 * describeShape() with its long number lists (convex points, trimesh positions / indices, exact arrays) as typed
 * arrays: they encode as one TYPED run (one float per number instead of two) and decode without boxing; shapes.js
 * and shapeFromDescriptor take typed arrays, values are unchanged (Float64 / Uint32).
 */
export function shapeDescriptor(shape) { return packArrays(describeShape(shape)); }
function packArrays(v, key) {
  if (Array.isArray(v)) {
    if (v.length > 4 && v.every((x) => typeof x === 'number')) {
      if (key === 'indices' && v.every((x) => x >= 0 && x <= 0xffffffff && Math.floor(x) === x)) return Uint32Array.from(v);
      return Float64Array.from(v);
    }
    for (let i = 0; i < v.length; i++) v[i] = packArrays(v[i]);
    return v;
  }
  if (v && typeof v === 'object' && !ArrayBuffer.isView(v)) for (const k of Object.keys(v)) v[k] = packArrays(v[k], k);
  return v;
}

// ------------------------------------------------------------------ snapshot writer
/**
 * Write a world's state into snapshot buffers (layouts in the header). `poses` / `aabbs` must hold
 * world.bodies.count records; `contacts` may be null. Returns `out` = { count, contactTotal }.
 */
export function writeSnapshot(world, poses, aabbs, contacts, out = { count: 0, contactTotal: 0 }) {
  const B = world.bodies, n = B.count;
  const P = B.pos, Q = B.quat, V = B.vel, W = B.angVel, A = B.aabb, F = B.flags, T = B.type, L = B.alive;
  for (let id = 0; id < n; id++) {
    const o = id * POSE_STRIDE;
    if (!L[id]) { poses.fill(0, o, o + POSE_STRIDE); if (aabbs) aabbs.fill(0, id * 6, id * 6 + 6); continue; }
    const i3 = id * 3, i4 = id * 4;
    poses[o] = P[i3]; poses[o + 1] = P[i3 + 1]; poses[o + 2] = P[i3 + 2];
    poses[o + 3] = Q[i4]; poses[o + 4] = Q[i4 + 1]; poses[o + 5] = Q[i4 + 2]; poses[o + 6] = Q[i4 + 3];
    poses[o + 7] = V[i3]; poses[o + 8] = V[i3 + 1]; poses[o + 9] = V[i3 + 2];
    poses[o + 10] = W[i3]; poses[o + 11] = W[i3 + 1]; poses[o + 12] = W[i3 + 2];
    poses[o + 13] = (F[id] & 255) | POSE_ALIVE | (T[id] << POSE_TYPE_SHIFT);
    if (aabbs) { const a = id * 6; for (let k = 0; k < 6; k++) aabbs[a + k] = A[a + k]; }
  }
  out.count = n; out.contactTotal = 0;
  const C = world.contacts;
  if (contacts) {
    const cap = ((contacts.length - 1) / CONTACT_STRIDE) | 0, total = C ? C.count : 0, m = Math.min(total, cap);
    contacts[0] = m;
    for (let i = 0; i < m; i++) {
      const o = 1 + i * CONTACT_STRIDE;
      contacts[o] = C.a[i]; contacts[o + 1] = C.b[i];
      contacts[o + 2] = C.px[i]; contacts[o + 3] = C.py[i]; contacts[o + 4] = C.pz[i];
      contacts[o + 5] = C.nx[i]; contacts[o + 6] = C.ny[i]; contacts[o + 7] = C.nz[i];
      contacts[o + 8] = C.depth[i]; contacts[o + 9] = C.lambdaN ? C.lambdaN[i] : 0;
    }
    out.contactTotal = total;
  }
  return out;
}

/** Default world factory of the worker (init message -> DynamicsWorld). */
export function createWorldFrom(msg) {
  if (msg.namedLayers) setNamedLayers(msg.namedLayers); // before any layerOf: the same bits as the main thread
  const physics = msg.physics ? physicsFromStatic(msg.physics) : null;
  if (msg.serialized) return DynamicsWorld.from(msg.serialized, { physics });
  return new DynamicsWorld(physics ? { ...(msg.options || {}), physics } : (msg.options || {}));
}

/** Joints already in a world (a deserialized one): identity client -> world id map. Returns the joint count. */
export function mapExistingJoints(world, map) {
  const J = world.joints;
  if (!J || !J.alive || !(J.count >= 0)) return 0;
  for (let i = 0; i < J.count; i++) map[i] = J.alive[i] ? i : -1;
  return J.count;
}

// ------------------------------------------------------------------ worker message handler
/**
 * The worker's message handler, transport-agnostic: `post(msg, transferList)` sends to the main thread; the returned
 * function takes a message's data. Bound to `self` below in a browser worker; Node tests bind it to a MessagePort.
 * @param {(msg: object, transfer?: Transferable[]) => void} post
 * @param {{ createWorld?: (initMsg: object) => object }} [o]  world factory (default: createWorldFrom)
 */
export function createDynamicsWorkerHandler(post, o = {}) {
  const createWorld = o.createWorld || createWorldFrom;
  let world = null, cap = 0, ccap = 0, time = 0, stepCount = 0, sets = 0;
  const freeP = [], freeA = [], freeC = [];
  let cmdBack = [], cmdSent = []; // returned command buffers: collected in one array while the other is posted
  const ctx = createApplyContext();
  const snapOut = { count: 0, contactTotal: 0 };
  const msg = { type: 'poses', seq: 0, stepCount: 0, time: 0, count: 0, buffer: null, aabbs: null, contacts: null, contactTotal: 0, stats: null, stepMs: 0, cmd: null, sets: 0, error: null };
  const tr = [];
  const fail = (phase, seq, message) => post({ type: 'error', phase, message, seq });
  const flushErrors = (phase, seq) => {
    if (!ctx.errors.length) return;
    fail(phase, seq, ctx.errors.join('; '));
    ctx.errors.length = 0;
  };

  function step(m) {
    const bufs = m.bufs;
    if (bufs) {
      for (let i = 0; i + 2 < bufs.length; i += 3) {
        if (bufs[i] && bufs[i].length === cap * POSE_STRIDE) freeP.push(bufs[i]);
        if (bufs[i + 1] && bufs[i + 1].length === cap * 6) freeA.push(bufs[i + 1]);
        if (bufs[i + 2] && bufs[i + 2].length === 1 + ccap * CONTACT_STRIDE) freeC.push(bufs[i + 2]);
      }
    }
    const n = Math.max(0, m.n | 0), dt = m.dt;
    let error = null;
    const t0 = performance.now();
    try {
      for (let i = 0; i < n; i++) { world.step(dt); time += dt; stepCount++; }
    } catch (e) { error = `world.step threw: ${String((e && e.stack) || e)}`; }
    const stepMs = performance.now() - t0;
    let P = freeP.pop();
    if (!P) { P = new Float32Array(cap * POSE_STRIDE); sets++; }
    const A = freeA.pop() || new Float32Array(cap * 6);
    const C = freeC.pop() || new Float32Array(1 + ccap * CONTACT_STRIDE);
    try {
      writeSnapshot(world, P, A, C, snapOut);
    } catch (e) { // no snapshot for this step: keep the buffers, answer with an error
      freeP.push(P); freeA.push(A); freeC.push(C);
      fail('step', m.seq, `${error ? error + '; ' : ''}snapshot failed: ${String((e && e.stack) || e)}`);
      return;
    }
    if (Number.isFinite(world.time)) time = world.time;
    if (Number.isFinite(world.stepCount)) stepCount = world.stepCount;
    msg.seq = m.seq; msg.stepCount = stepCount; msg.time = time; msg.count = snapOut.count;
    msg.buffer = P; msg.aabbs = A; msg.contacts = C; msg.contactTotal = snapOut.contactTotal;
    msg.stats = world.stats || null; msg.stepMs = stepMs; msg.sets = sets; msg.error = error;
    tr.length = 0; tr.push(P.buffer, A.buffer, C.buffer);
    msg.cmd = null;
    if (cmdBack.length) {
      const sent = cmdBack; cmdBack = cmdSent; cmdSent = sent; cmdBack.length = 0;
      msg.cmd = sent;
      for (let i = 0; i < sent.length; i++) tr.push(sent[i].buffer);
    }
    post(msg, tr);
    msg.buffer = msg.aabbs = msg.contacts = msg.cmd = msg.stats = msg.error = null;
  }

  function onmessage(m) {
    if (!m || typeof m !== 'object') return;
    try {
      if (m.type === 'cmd') {
        if (!world) throw new Error('cmd before init');
        applyCommands(world, m.buffer, m.length, ctx);
        cmdBack.push(m.buffer);
        flushErrors('cmd', m.seq);
        return;
      }
      if (m.type === 'step') {
        if (!world) throw new Error('step before init');
        step(m);
        return;
      }
      if (m.type === 'init') {
        world = createWorld(m);
        cap = world.bodies.capacity;
        ccap = Math.max(0, (m.contactCapacity ?? DEFAULT_SNAPSHOT_CONTACTS) | 0);
        time = Number.isFinite(world.time) ? world.time : 0;
        stepCount = Number.isFinite(world.stepCount) ? world.stepCount : 0;
        mapExistingJoints(world, ctx.jointMap);
        post({ type: 'ready', capacity: cap, count: world.bodies.count, time, stepCount });
        return;
      }
      if (m.type === 'serialize') {
        let data = null, error = null;
        try { data = world ? world.serialize() : null; } catch (e) { error = String((e && e.stack) || e); }
        post({ type: 'serialized', id: m.id, data, error });
        return;
      }
    } catch (e) {
      fail(m.type, m.seq ?? -1, String((e && e.stack) || e));
    }
  }
  /** Diagnostics for tests: snapshot sets allocated and buffers waiting in the pools. */
  onmessage.pools = () => ({ sets, freePoses: freeP.length, freeAabbs: freeA.length, freeContacts: freeC.length, cmdPending: cmdBack.length });
  onmessage.world = () => world;
  onmessage.ctx = ctx;
  return onmessage;
}

// Browser module worker: bind to the worker global (never to a page's `self`: DynamicsClient imports this file).
if (typeof self !== 'undefined' && typeof WorkerGlobalScope !== 'undefined' && self instanceof WorkerGlobalScope) { // eslint-disable-line no-undef
  const handle = createDynamicsWorkerHandler((m, transfer) => self.postMessage(m, transfer || []));
  self.onmessage = (ev) => handle(ev.data);
}
