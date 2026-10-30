// Shared data layout of the dynamics world (pure JS, deterministic, Node-safe). Every dynamics module codes against
// these structures, so they are the contract between the broadphase, the narrowphase, the solver, the joints and the
// queries: structure-of-arrays Float64 state indexed by body id, a contact buffer the narrowphase fills and the
// solver consumes, and the few shared constants. Nothing here allocates after construction.
//
//   const bodies = new BodyStore(4096);  const id = bodies.alloc();  bodies.pos[id * 3 + 1] = 2;
//   contacts.clear(); contacts.add(a, b, px, py, pz, nx, ny, nz, depth, featureKey, pairKey(a, b));
//
// Conventions: three.js Y-up right-handed, SI units; quaternions stored x, y, z, w; angular velocity in WORLD space
// (rad/s); the body-frame inverse inertia tensor is a full 3x3 (row-major) so convex hulls need no diagonalisation;
// body ids are dense small integers reused through a free list, `alive[id]` says whether an id is in use and
// `count` is one past the highest id ever allocated (iterate `for (id = 0; id < count; id++) if (alive[id])`).
// Determinism: the step path uses only + - * / Math.sqrt and comparisons (see core/mathDet.js for angles) and never
// iterates a Set or Map; iteration order is always ascending body / contact / joint index.

/** Body types. */
export const BODY = Object.freeze({ STATIC: 0, KINEMATIC: 1, DYNAMIC: 2 });
/** `flags[id]` bits. */
export const FLAG = Object.freeze({ ASLEEP: 1, NO_GRAVITY: 2, CCD: 4, TOUCHED_STATIC: 8 });
/** Shape type ids (shapes.js builds the objects; see its header for their fields). */
export const SHAPE = Object.freeze({ SPHERE: 0, CAPSULE: 1, BOX: 2, CONVEX: 3, TRIMESH: 4, COMPOUND: 5 });
/** Body id that stands for the static world (PhysicsWorld boxes, heightfield) in a contact's `b`. */
export const WORLD_BODY = -1;
/** Marker property on the collider records a dynamics world publishes into a PhysicsWorld (`rec[DYN_RECORD] === true`),
 * so the solver's static-world pass can skip its own bodies whatever their `kind` and whichever provider published them. */
export const DYN_RECORD = Symbol.for('vibe.dynamics.record');
/** Hard cap on body ids (pair keys pack two ids into a uint32). */
export const MAX_BODIES = 65536;

/** Unordered pair key of two body ids (a, b in 0..65535); the smaller id goes first. */
export function pairKey(a, b) { return a < b ? a * 65536 + b : b * 65536 + a; }
export function pairA(key) { return (key / 65536) | 0; }
export function pairB(key) { return key % 65536; }

/**
 * Structure-of-arrays rigid-body state. All per-body arrays are indexed by body id; vectors are packed x, y, z at
 * `id * 3`, quaternions x, y, z, w at `id * 4`, 3x3 tensors row-major at `id * 9`, AABBs minX, minY, minZ, maxX, maxY,
 * maxZ at `id * 6`.
 */
export class BodyStore {
  constructor(capacity = 4096) {
    if (capacity > MAX_BODIES) throw new RangeError(`BodyStore: capacity ${capacity} > ${MAX_BODIES}`);
    const n = this.capacity = capacity;
    this.count = 0;           // one past the highest allocated id
    this.alive = new Uint8Array(n);
    this.type = new Uint8Array(n);        // BODY.*
    this.flags = new Uint8Array(n);       // FLAG.* bits
    this.pos = new Float64Array(3 * n);
    this.quat = new Float64Array(4 * n);
    this.prevPos = new Float64Array(3 * n);   // pose at the start of the current substep (XPBD)
    this.prevQuat = new Float64Array(4 * n);
    this.vel = new Float64Array(3 * n);
    this.angVel = new Float64Array(3 * n);    // world space
    this.force = new Float64Array(3 * n);     // per-step accumulators, cleared by the solver after the step
    this.torque = new Float64Array(3 * n);
    this.invMass = new Float64Array(n);       // 0 for static / kinematic
    this.invInertia = new Float64Array(9 * n); // body-frame inverse inertia tensor, row-major; zeros when static
    this.mass = new Float64Array(n);
    this.friction = new Float64Array(n);
    this.restitution = new Float64Array(n);
    this.linearDamping = new Float64Array(n);
    this.angularDamping = new Float64Array(n);
    this.layer = new Int32Array(n);           // collision layer bits of this body
    this.mask = new Int32Array(n);            // layers it collides with
    this.aabb = new Float64Array(6 * n);      // world AABB, refreshed by the solver each step
    this.boundingRadius = new Float64Array(n);
    this.sleepTime = new Float64Array(n);
    this.island = new Int32Array(n);
    this.kinTarget = new Float64Array(7 * n); // kinematic target pose (x y z qx qy qz qw) for the next step
    this.shape = new Array(n).fill(null);     // shape object (shapes.js)
    this.userData = new Array(n).fill(null);
    this._free = new Int32Array(n); this._freeCount = 0;
    for (let i = 0; i < n; i++) this.quat[i * 4 + 3] = 1;
  }
  /** Number of live bodies. */
  get size() { let s = 0; for (let i = 0; i < this.count; i++) s += this.alive[i]; return s; }
  /** Allocate an id (lowest free id first, so iteration order stays stable); -1 when full. */
  alloc() {
    let id;
    if (this._freeCount > 0) id = this._free[--this._freeCount];
    else if (this.count < this.capacity) id = this.count++;
    else return -1;
    this.alive[id] = 1; this.type[id] = BODY.DYNAMIC; this.flags[id] = 0;
    const p = id * 3, q = id * 4;
    this.pos.fill(0, p, p + 3); this.prevPos.fill(0, p, p + 3); this.vel.fill(0, p, p + 3); this.angVel.fill(0, p, p + 3);
    this.force.fill(0, p, p + 3); this.torque.fill(0, p, p + 3);
    this.quat[q] = 0; this.quat[q + 1] = 0; this.quat[q + 2] = 0; this.quat[q + 3] = 1;
    this.prevQuat.set(this.quat.subarray(q, q + 4), q);
    this.invMass[id] = 0; this.mass[id] = 0; this.invInertia.fill(0, id * 9, id * 9 + 9);
    this.friction[id] = 0.6; this.restitution[id] = 0.1; this.linearDamping[id] = 0.01; this.angularDamping[id] = 0.05;
    this.layer[id] = 1; this.mask[id] = -1; this.aabb.fill(0, id * 6, id * 6 + 6); this.boundingRadius[id] = 0;
    this.sleepTime[id] = 0; this.island[id] = -1; this.kinTarget.fill(0, id * 7, id * 7 + 7); this.kinTarget[id * 7 + 6] = 1;
    this.shape[id] = null; this.userData[id] = null;
    return id;
  }
  /** Release an id (its slot is reused by a later alloc; the free list is kept sorted descending so the lowest id comes back first). */
  free(id) {
    if (!this.alive[id]) return;
    this.alive[id] = 0; this.shape[id] = null; this.userData[id] = null;
    const f = this._free; let i = this._freeCount++;
    while (i > 0 && f[i - 1] < id) { f[i] = f[i - 1]; i--; } // descending: pop() returns the smallest id
    f[i] = id;
  }
  isDynamic(id) { return this.alive[id] === 1 && this.type[id] === BODY.DYNAMIC; }
  isAwake(id) { return (this.flags[id] & FLAG.ASLEEP) === 0; }
}

/**
 * Contact points for one step, structure-of-arrays. The narrowphase appends up to 4 points per body pair with the
 * normal ALWAYS pointing from body `a` to body `b` (for the static world `b === WORLD_BODY`, so a body resting on the
 * ground has the normal (0, -1, 0): from the body into the ground). `depth` > 0 is penetration; negative depth (down
 * to -margin) is a speculative
 * contact the solver may ignore until the bodies close in. `key` identifies the contact feature within its pair so
 * warm-start lambdas can be carried across steps by the narrowphase's persistent table; `lambdaN` / `lambdaT` hold
 * the accumulated impulses the solver writes back. `staticIndex` is the index of the static box in the query list
 * (or -1 for the heightfield) when `b === WORLD_BODY`.
 */
export class ContactBuffer {
  constructor(capacity = 16384) {
    const n = this.capacity = capacity;
    this.count = 0;
    this.a = new Int32Array(n); this.b = new Int32Array(n);
    this.px = new Float64Array(n); this.py = new Float64Array(n); this.pz = new Float64Array(n);
    this.nx = new Float64Array(n); this.ny = new Float64Array(n); this.nz = new Float64Array(n);
    this.depth = new Float64Array(n);
    this.key = new Uint32Array(n); this.pairKey = new Uint32Array(n);
    this.lambdaN = new Float64Array(n); this.lambdaT = new Float64Array(n);
    this.staticIndex = new Int32Array(n);
    this.overflow = 0; // contacts dropped because the buffer was full (diagnostic)
  }
  clear() { this.count = 0; this.overflow = 0; }
  /** Append a contact; returns its index or -1 when full. */
  add(a, b, px, py, pz, nx, ny, nz, depth, key, pairKey, staticIndex = -1) {
    const i = this.count;
    if (i >= this.capacity) { this.overflow++; return -1; }
    this.a[i] = a; this.b[i] = b; this.px[i] = px; this.py[i] = py; this.pz[i] = pz; this.nx[i] = nx; this.ny[i] = ny; this.nz[i] = nz;
    this.depth[i] = depth; this.key[i] = key >>> 0; this.pairKey[i] = pairKey >>> 0; this.lambdaN[i] = 0; this.lambdaT[i] = 0; this.staticIndex[i] = staticIndex;
    this.count = i + 1;
    return i;
  }
}

/** Fixed-size pose scratch helpers (x y z qx qy qz qw in a Float64Array(7)). */
export function writePose(out, o, x, y, z, qx, qy, qz, qw) { out[o] = x; out[o + 1] = y; out[o + 2] = z; out[o + 3] = qx; out[o + 4] = qy; out[o + 5] = qz; out[o + 6] = qw; }
