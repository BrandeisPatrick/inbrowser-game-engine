// XPBD particles of the dynamics world (pure JS, deterministic, Node-safe, no THREE): the particle store, the built-in
// particle constraint solver (distance / bending / attachment / tether) and the Cloth / Rope builders. Particles live
// inside DynamicsWorld's substeps (see "Particles" in DynamicsWorld.js): the world integrates them next to the rigid
// bodies and calls every registered particle solver inside its position and velocity passes, so particles and bodies
// exchange corrections within the same substep (two-way attachments here; particle-vs-rigid coupling, buoyancy and soft
// bodies are further solvers on the same hooks).
//
//   const world = new DynamicsWorld({ physics, maxParticles: 4096 });       // owns world.particles + world.constraints
//   const flag = Cloth.grid(world, { nx: 32, ny: 24, position: [0, 2, 0], pins: ['top-left', 'top-right'] });
//   const rope = Rope.chain(world, { from: [1, 3, 0], to: [1, 1, 0], pins: ['from', { end: 'to', body: crate, anchor: [0, 0.25, 0] }] });
//   engine fixed step: world.step(dt);
//   render: vertex k of the cloth mesh = world.particles.pos at flag.ids[k] * 3, indexed by flag.triangles()
//
// ParticleStore (structure of arrays, index = particle id; dense ids reused through a free list, lowest free id first,
// `alive[id]` says whether an id is in use and `count` is one past the highest id ever allocated, so iterate
// `for (id = 0; id < count; id++) if (alive[id])`):
//   alive Uint8, flags Uint8 (PFLAG.PINNED: invMass 0 and never integrated; PFLAG.NO_GRAVITY), group Int32 (a caller
//   tag: the cloth / rope / soft body a particle belongs to, -1 = none; newGroup() hands out fresh ones),
//   pos / prevPos / vel Float64 (3n; prevPos = the pose at the start of the current substep), invMass, radius
//   (collision radius, m), mass Float64 (n).
//   alloc() -> id (throws when full; mass 1 kg, radius 0, group -1, at the origin), free(id) (runs the free hooks, then
//   clears the slot), addFreeHook(fn(id)) / removeFreeHook(fn) (the world's ParticleConstraints registers its
//   detachParticle, so a freed id never keeps constraints and a reused id never inherits them; solvers that keep
//   particle ids register theirs), setPinned(id, on) (invMass 0 and velocity 0 / invMass restored from mass),
//   setMass(id, kg), newGroup() -> int, size (live particles, O(1)), serialize() -> plain object of typed-array
//   copies, restore(data).
//
// ParticleConstraints (the built-in particle solver, `world.constraints`, always particleSolvers[0]; SoA, index =
// constraint id, ids reused lowest first, `count` one past the highest id; arrays are replaced when a full store grows
// on an add, so re-read them after adding):
//   addDistance(a, b, restLength = current, compliance = 0) -> ci          |p_a - p_b| = rest
//   addBending(a, b, c, d, compliance = 0) -> ci                           the quick bending of Mueller 2007 /
//       Macklin: a distance constraint between the opposite vertices c and d of the two triangles (a, b, c) and
//       (a, b, d) that share the edge a-b, rest = their current distance. Cheap, linear, deterministic and good
//       enough for cloth: it resists folding about a-b through the c-d distance (it is weak near flat, where the
//       distance is first-order insensitive to the fold angle, so a flat cloth still buckles naturally). A rope's
//       bending is addBending(m, m, i, k) for the triple i - m - k.
//   addAttachment(p, body, ax, ay, az, compliance = 0) -> ci               particle p held at the body-local point
//       (ax, ay, az) of `body`, or at the world point for WORLD_BODY (-1). Two-way: C = |p - t| with t = x_b + R_b a;
//       the particle moves by its invMass, the body by its generalized inverse mass m^-1 + (r x n)^T I_w^-1 (r x n)
//       (xpbd.applyPositional maths: x_b -= m^-1 dLambda n, q_b rotated by -I_w^-1 (r x dLambda n)), so momentum is
//       exchanged. Static / kinematic / WORLD_BODY targets are immovable, and so is a body that is asleep at the start
//       of a step. Sleep (endStep, attachments and body tethers alike): an awake body's sleep timer restarts while the
//       constraint's particle moved more than sleep.linear * dt over the step, so a body tied to moving cloth stays
//       awake while a slack or resting one sleeps with its island; a sleeping body wakes when that particle moves
//       (e.g. a body tether's pin is moved) or when the constraint's force departs from its value while asleep by
//       more than 1e-3 N + 1 % (a yank on the rope; the baseline is taken in the first step spent wholly asleep).
//       DynamicsWorld.removeBody removes the attachments of that body (detachBody).
//   addTether(p, q, maxLength = current, compliance = 0) -> ci             (extension) one-sided |p_q - p_p| <=
//       maxLength (a long-range attachment, Kim, Chentanez, Mueller 2012): it only pulls. One Gauss-Seidel sweep per
//       substep stretches a hanging chain (0.7 % for the default 20-segment rope at 4 substeps of 1/60 s, its bending
//       pairs included; ~N^2 h^2 g / 2 without them); a tether from every particle to its nearest pinned particle,
//       solved after the links, removes that stretch for anything hanging from pins (a swinging, curved rope still
//       stretches dynamically: tethers only bound straight-line distances).
//   addBodyTether(p, body, ax, ay, az, maxLength = current, compliance = 0) -> ci   (extension) one-sided
//       |p - t| <= maxLength between particle p (typically pinned) and a body point, two-way like an attachment. A
//       light particle between a heavy body and its tether ping-pongs in Gauss-Seidel (a 10 kg box on a 0.5 kg rope
//       sags 13 cm); tying the body to the pin directly holds it at the rope length exactly. Note for break logic:
//       with a body tether the body's load goes through it, so the attachment's own `force` no longer reports the
//       body's full load (a hanging body reads ~0 on its attachment); sum the two.
//   remove(ci), detachBody(bodyId) -> removed count (attachments + body tethers), detachParticle(id) -> removed count
//   (every constraint naming the particle; O(1) when none does), reserve(n) -> first (the next n adds take
//   [first, first + n)) / endReserve(), count, size (live, O(1)), alive Uint8, type Uint8 (PCON.*), p0 / p1 Int32 (the particle
//   pair a constraint acts on: distance a b, bending c d, attachment / body tether p -1, tether p q), e0 / e1 Int32
//   (bending edge a b, else -1), body Int32 (attachment / body tether body, else -1), rest Float64 (attachment 0),
//   compliance Float64 (m/N; 0 = rigid), anchor Float64 (3n; body-local or world point), lambda Float64 (per substep,
//   reset at iteration 0), force Float64 (|lambda| / h^2 of the last substep, N; 0 when the constraint was skipped:
//   dead particle, coincident ends, both ends immovable), lastPos / restForce (sleep bookkeeping of attachments and
//   body tethers), serial Uint32 (stamp of the add that filled the slot: tells a reused id from its former owner).
//   Solve (solvePositions): one Gauss-Seidel sweep in ascending constraint index per call, XPBD with alpha~ =
//   compliance / h^2 (Macklin, Mueller, Chentanez 2016): dLambda = (-C - alpha~ lambda) / (w_a + w_b + alpha~),
//   p_a += w_a dLambda n, p_b -= w_b dLambda n; pinned particles have invMass 0. No velocity pass (damping is the
//   world's `particleDamping`): solveVelocities(world, h, it) is a no-op.
//
// Builders (both allocate their particles in ascending order, so ids are contiguous in a fresh store, and their
// constraints as one contiguous block; `ids` is the truth, `first` only names ids[0]; dispose() removes exactly the
// constraints it emitted, by id and serial, then frees the particles; a builder that throws leaves nothing behind).
// One pin per particle: plain pins (strings, { i, j }, 'from' / 'to') may overlap, but a pin with a `body` must be
// the only pin on its particle (Error otherwise). Pins: a pin with
// `body: WORLD_BODY` is a real pin (PINNED) at its `anchor` (the particle is moved there after the rest lengths are
// taken), not an attachment: tethers anchored on it then work fully. With `tethers` (default: stretch compliance 0)
// every unpinned particle gets a tether to its nearest pinned particle (rest distance; with no pin, to its nearest
// attached particle; none without either), and with pins every attachment to a DYNAMIC body gets a body tether from
// the nearest pinned particle (max = their rest distance + the anchor's offset from the particle).
//   Cloth.grid(world, { nx = 32, ny = 32, spacing = 0.05, position = [0, 0, 0] (the centre of the rest rectangle),
//     quaternion = [0, 0, 0, 1], mass = 1 (total), stretch = 0, shear = 1e-4, bend = 1e-5 (compliances, m/N; shear 0
//     with stretch 0 makes every cell a rigid square: a developable sheet of hinged plates that cannot drape doubly
//     curved), radius =
//     spacing * 0.4, pins = [] ('top-left' | 'top-right' | 'top' | 'left' | 'right' | 'bottom-left' |
//     'bottom-right' | 'bottom', { i, j } pinned in place, { i, j, body, anchor: [x, y, z] } attached to a body,
//     anchor default: where the particle is, in the body's frame; body WORLD_BODY: pinned at anchor), group = a
//     fresh one, tethers = stretch === 0 })
//     -> { first, count, nx, ny, ids: Int32Array, group, constraints: { first, count }, triangles(), indexOf(i, j),
//          dispose() }
//     Particle (i, j) (i along the local +X, j along the local +Y of `quaternion`; 'top' = j = ny - 1, 'left' = i = 0)
//     starts at position + R ((i - (nx - 1) / 2) spacing, (j - (ny - 1) / 2) spacing, 0), ids[j * nx + i].
//     Constraints: structural distance on both grid axes (stretch), both cell diagonals (shear), bending across every
//     interior grid edge for BOTH diagonal triangulations of the cells (the pair of the "/" split and the pair of the
//     "\" split per edge; the in-cell pairs are the shear diagonals themselves, so they are not repeated), tethers,
//     body tethers and the attachments, in that order,
//     then reordered so the left-right mirror image of every constraint (i -> nx - 1 - i) follows it directly: with
//     an even nx a constraint and its mirror share no particle and commute, so the Gauss-Seidel sweep is itself
//     mirror-symmetric and a symmetrically pinned cloth stays symmetric to the bit (both triangulations make the
//     bending set symmetric too; one triangulation drapes with a diagonal bias). triangles(): Uint32Array (allocated
//     once), 3 indices into `ids` per triangle, the "/" split, counter-clockwise seen from the rest plane's +normal
//     (local +Z). indexOf(i, j) -> particle id. dispose() frees the particles and the constraints.
//   Rope.chain(world, { from = [0, 0, 0], to = [0, -1, 0], segments = 20, mass = 0.5 (total), radius = 0.02,
//     stretch = 0, bend = 1e-4, pins = ['from'] ('from' | 'to' pinned in place, { end: 'from' | 'to', body, anchor }
//     attached; body WORLD_BODY: pinned at anchor), group, tethers = stretch === 0 })
//     -> { first, count, ids, group, constraints: { first, count }, length, dispose() }
//     segments + 1 particles evenly on from -> to; distance per segment (stretch), bending per triple (bend), the
//     tethers, the body tethers, the attachments; solved from the anchored end outward (descending when only 'to' is
//     anchored). An end both pinned and attached stays pinned (the attachment is dropped).
//
// Determinism: the solve path uses only + - * / Math.sqrt and comparisons, iterates ascending constraint / particle
// indices and never a Set or Map (the builders look a Map up at construction only). Allocation: none in the hooks; only
// integers and arrays cross function boundaries in the solve (h arrives as the world's boxed number and is only read).
import { BODY, FLAG, WORLD_BODY } from './state.js';
import { displaceBodyFrom } from './xpbd.js';

/** `ParticleStore.flags` bits. */
export const PFLAG = Object.freeze({ PINNED: 1, NO_GRAVITY: 2 });
/** ParticleConstraints types. */
export const PCON = Object.freeze({ DISTANCE: 0, BENDING: 1, ATTACHMENT: 2, TETHER: 3, BODY_TETHER: 4 });

const PINNED = PFLAG.PINNED;
const { DISTANCE, BENDING, ATTACHMENT, TETHER, BODY_TETHER } = PCON;
const DYN = BODY.DYNAMIC, ASLEEP = FLAG.ASLEEP;

// module scratch (the solve is not re-entrant)
const _sc = new Float64Array(2);   // 1 / h^2 of the running solve
const _c = new Float64Array(3);    // r x n of an attachment
const _i = new Float64Array(3);    // I_w^-1 (r x n)
const _d = new Float64Array(6);    // body displacement (dx dy dz wx wy wz)
const WAKE_FORCE = 1e-3, WAKE_REL = 0.01;   // N, relative: attachment force change that wakes a sleeping body

// D = I_w^-1 S = R I_b^-1 R^T S for a movable body (xpbd.worldInvInertiaApply with array arguments: a double passed to
// a call V8 does not inline is boxed, so only integers and arrays cross this boundary)
function invInertiaApply(B, id, S, D) {
  const vx = S[0], vy = S[1], vz = S[2], I = B.invInertia, k = id * 9;
  const i0 = I[k], i1 = I[k + 1], i2 = I[k + 2], i3 = I[k + 3], i4 = I[k + 4], i5 = I[k + 5], i6 = I[k + 6], i7 = I[k + 7], i8 = I[k + 8];
  if (i1 === 0 && i2 === 0 && i3 === 0 && i5 === 0 && i6 === 0 && i7 === 0 && i0 === i4 && i0 === i8) {
    D[0] = i0 * vx; D[1] = i0 * vy; D[2] = i0 * vz; return;   // isotropic: rotation-invariant
  }
  const q = B.quat, o = id * 4, x = q[o], y = q[o + 1], z = q[o + 2], w = q[o + 3];
  let tx = 2 * (z * vy - y * vz), ty = 2 * (x * vz - z * vx), tz = 2 * (y * vx - x * vy);
  const ux = vx + w * tx + (z * ty - y * tz), uy = vy + w * ty + (x * tz - z * tx), uz = vz + w * tz + (y * tx - x * ty);
  const mx = i0 * ux + i1 * uy + i2 * uz, my = i3 * ux + i4 * uy + i5 * uz, mz = i6 * ux + i7 * uy + i8 * uz;
  tx = 2 * (y * mz - z * my); ty = 2 * (z * mx - x * mz); tz = 2 * (x * my - y * mx);
  D[0] = mx + w * tx + (y * tz - z * ty);
  D[1] = my + w * ty + (z * tx - x * tz);
  D[2] = mz + w * tz + (x * ty - y * tx);
}

// typed-array view of a serialized array (typed array, plain array or a JSON'd typed array {0: .., 1: ..})
const asArray = (v) => (v == null ? [] : ArrayBuffer.isView(v) || Array.isArray(v) ? v : Object.values(v));
const vx = (v) => (v == null ? 0 : typeof v.x === 'number' ? v.x : v[0]);
const vy = (v) => (v == null ? 0 : typeof v.y === 'number' ? v.y : v[1]);
const vz = (v) => (v == null ? 0 : typeof v.z === 'number' ? v.z : v[2]);
const vw = (v) => (v == null ? 1 : typeof v.w === 'number' ? v.w : v[3]);

const PARTICLE_FIELDS = Object.freeze(['alive', 'flags', 'group', 'pos', 'prevPos', 'vel', 'invMass', 'radius', 'mass']);

/** Structure-of-arrays particle state (see the header). */
export class ParticleStore {
  /** @param {number} [capacity=4096] */
  constructor(capacity = 4096) {
    const n = this.capacity = Math.max(1, capacity | 0);
    this.count = 0;                           // one past the highest allocated id
    this.alive = new Uint8Array(n);
    this.flags = new Uint8Array(n);           // PFLAG.* bits
    this.group = new Int32Array(n).fill(-1);
    this.pos = new Float64Array(3 * n);
    this.prevPos = new Float64Array(3 * n);   // pose at the start of the current substep
    this.vel = new Float64Array(3 * n);
    this.invMass = new Float64Array(n);       // 0 when pinned
    this.radius = new Float64Array(n);        // collision radius (m)
    this.mass = new Float64Array(n);
    this.nextGroup = 0;                       // newGroup() counter
    this._free = new Int32Array(n); this._freeCount = 0;
    this._size = 0;
    this._freeHooks = [];                     // fn(id), called by free() before the slot is cleared
  }
  /** Number of live particles. */
  get size() { return this._size; }
  /**
   * Call fn(id) whenever a particle is freed, before its slot is cleared (ParticleConstraints registers its
   * detachParticle; other particle solvers that keep particle ids register theirs), in registration order.
   */
  addFreeHook(fn) { if (typeof fn === 'function' && !this._freeHooks.includes(fn)) this._freeHooks.push(fn); }
  removeFreeHook(fn) { const k = this._freeHooks.indexOf(fn); if (k >= 0) this._freeHooks.splice(k, 1); }
  /** Allocate an id (lowest free id first): 1 kg, radius 0, group -1, at rest at the origin. Throws when full. */
  alloc() {
    let id;
    if (this._freeCount > 0) id = this._free[--this._freeCount];
    else if (this.count < this.capacity) id = this.count++;
    else throw new RangeError(`ParticleStore.alloc: capacity (${this.capacity}) reached`);
    this._size++;
    this.alive[id] = 1; this.flags[id] = 0; this.group[id] = -1;
    const p = id * 3;
    this.pos.fill(0, p, p + 3); this.prevPos.fill(0, p, p + 3); this.vel.fill(0, p, p + 3);
    this.mass[id] = 1; this.invMass[id] = 1; this.radius[id] = 0;
    return id;
  }
  /**
   * Release an id: the free hooks run first (the world's constraints on it are removed: a reused id never inherits
   * them), then the slot is cleared and reused by a later alloc (the lowest free id comes back first).
   */
  free(id) {
    if (!(id >= 0 && id < this.count) || !this.alive[id]) return;
    const hooks = this._freeHooks;
    for (let k = 0; k < hooks.length; k++) hooks[k](id);
    this._size--;
    this.alive[id] = 0; this.flags[id] = 0; this.group[id] = -1;
    const p = id * 3;
    this.pos.fill(0, p, p + 3); this.prevPos.fill(0, p, p + 3); this.vel.fill(0, p, p + 3);
    this.mass[id] = 0; this.invMass[id] = 0; this.radius[id] = 0;
    const f = this._free; let i = this._freeCount++;
    while (i > 0 && f[i - 1] < id) { f[i] = f[i - 1]; i--; } // descending: pop() returns the smallest id
    f[i] = id;
  }
  /** Pin (invMass 0, velocity 0, never integrated) or release (invMass = 1 / mass) a particle. */
  setPinned(id, on) {
    if (on) {
      this.flags[id] |= PINNED; this.invMass[id] = 0;
      this.vel.fill(0, id * 3, id * 3 + 3);
    } else {
      this.flags[id] &= ~PINNED;
      const m = this.mass[id];
      this.invMass[id] = m > 0 ? 1 / m : 0;
    }
  }
  /** Set the mass (kg; > 0); a pinned particle keeps invMass 0 until released. */
  setMass(id, m) {
    this.mass[id] = m;
    if ((this.flags[id] & PINNED) === 0) this.invMass[id] = m > 0 ? 1 / m : 0;
  }
  /** A fresh group id (Cloth / Rope / soft bodies tag their particles with one). */
  newGroup() { return this.nextGroup++; }
  /** Plain snapshot (typed-array copies up to count). */
  serialize() {
    const n = this.count, out = { capacity: this.capacity, count: n, nextGroup: this.nextGroup, free: this._free.slice(0, this._freeCount) };
    for (const k of PARTICLE_FIELDS) { const a = this[k], s = a.length / this.capacity; out[k] = a.slice(0, n * s); }
    return out;
  }
  /** Restore serialize()'s snapshot (bit-identical). */
  restore(d) {
    const n = d.count | 0;
    if (n > this.capacity) throw new RangeError(`ParticleStore.restore: snapshot holds ${n} ids, capacity ${this.capacity}`);
    for (const k of PARTICLE_FIELDS) {
      const a = this[k];
      a.fill(k === 'group' ? -1 : 0);
      if (d[k] !== undefined) a.set(asArray(d[k]));
    }
    this.count = n;
    this.nextGroup = d.nextGroup | 0;
    const free = asArray(d.free);
    this._free.set(free); this._freeCount = free.length;
    let live = 0;
    for (let i = 0; i < n; i++) live += this.alive[i];
    this._size = live;
  }
}

const CONSTRAINT_FIELDS = Object.freeze(['alive', 'type', 'p0', 'p1', 'e0', 'e1', 'body', 'rest', 'compliance', 'anchor', 'lambda', 'force', 'lastPos', 'restForce', 'serial']);

/** The built-in particle solver: distance, bending, attachment and tether constraints (see the header). */
export class ParticleConstraints {
  /**
   * @param {ParticleStore} particles
   * @param {number} [capacity=1024]  initial constraint slots (doubles when full)
   * @param {object|null} [bodies=null]  the world's BodyStore (validates attachment bodies)
   */
  constructor(particles, capacity = 1024, bodies = null) {
    this.name = 'constraints';
    this.particles = particles;
    this.bodies = bodies;
    this.count = 0;            // one past the highest allocated id
    this._lowFree = 0;         // every id below is alive
    this._cursor = -1; this._cursorEnd = -1;  // reserve(): the next adds take [cursor, cursorEnd)
    this._attachments = 0;     // live attachment constraints (beginStep / endStep skip their scan without any)
    this._size = 0;            // live constraints
    this._nextSerial = 1;      // serial[] stamp of the next add (tells a reused id from the constraint that had it)
    this._refs = new Int32Array(particles.capacity); // per particle: live constraints that name it (p0 p1 e0 e1)
    this._alloc(Math.max(16, capacity | 0));
    this._onFree = (id) => { this.detachParticle(id); };
    particles.addFreeHook(this._onFree);   // a freed particle never keeps (or hands its reused id) constraints
  }
  _alloc(n) {
    this.capacity = n;
    this.alive = new Uint8Array(n);
    this.type = new Uint8Array(n);
    this.p0 = new Int32Array(n); this.p1 = new Int32Array(n);
    this.e0 = new Int32Array(n); this.e1 = new Int32Array(n);
    this.body = new Int32Array(n);
    this.rest = new Float64Array(n);
    this.compliance = new Float64Array(n);
    this.anchor = new Float64Array(3 * n);
    this.lambda = new Float64Array(n);
    this.force = new Float64Array(n);
    this.lastPos = new Float64Array(3 * n);  // attachment / body tether: its particle at the end of the last step
    this.restForce = new Float64Array(n);    // ... its force while the body sleeps (-1: not measured yet)
    this.serial = new Uint32Array(n);        // stamp of the add that created the constraint in this slot
    this._live = new Uint8Array(n);   // attachment: its body is movable this step (beginStep)
  }
  _grow(min) {
    let n = this.capacity * 2;
    while (n < min) n *= 2;
    const old = {};
    for (const k of CONSTRAINT_FIELDS) old[k] = this[k];
    const live = this._live;
    this._alloc(n);
    for (const k of CONSTRAINT_FIELDS) this[k].set(old[k]);
    this._live.set(live);
  }
  /** Number of live constraints. */
  get size() { return this._size; }
  // add `d` to the reference count of every distinct particle constraint c names
  _ref(c, d) {
    const r = this._refs, a = this.p0[c], b = this.p1[c], e = this.e0[c], f = this.e1[c];
    if (a >= 0) r[a] += d;
    if (b >= 0 && b !== a) r[b] += d;
    if (e >= 0 && e !== a && e !== b) r[e] += d;
    if (f >= 0 && f !== a && f !== b && f !== e) r[f] += d;
  }
  /** Stop a pending reserve() (the builders call it when they add fewer constraints than they reserved). */
  endReserve() { this._cursor = -1; this._cursorEnd = -1; }

  /** Reserve n contiguous free ids (the lowest such run): the next n adds take [first, first + n). Returns first. */
  reserve(n) {
    let start = this._lowFree, run = 0;
    for (let i = this._lowFree; i < this.count && run < n; i++) {
      if (this.alive[i]) { start = i + 1; run = 0; } else run++;
    }
    if (start + n > this.capacity) this._grow(start + n);
    if (n > 0) { this._cursor = start; this._cursorEnd = start + n; }
    return start;
  }
  _slot() {
    let id;
    if (this._cursor >= 0) {
      id = this._cursor++;
      if (this._cursor >= this._cursorEnd) { this._cursor = -1; this._cursorEnd = -1; }
    } else {
      id = this._lowFree;
      while (id < this.count && this.alive[id]) id++;
      if (id >= this.capacity) this._grow(id + 1);
      this._lowFree = id + 1;
    }
    if (id >= this.count) this.count = id + 1;
    this.alive[id] = 1;
    this._size++;
    this.serial[id] = this._nextSerial; this._nextSerial = (this._nextSerial + 1) >>> 0 || 1;
    this.restForce[id] = -1; this.lastPos[id * 3] = 0; this.lastPos[id * 3 + 1] = 0; this.lastPos[id * 3 + 2] = 0;
    this.p0[id] = -1; this.p1[id] = -1; this.e0[id] = -1; this.e1[id] = -1; this.body[id] = -1;
    this.rest[id] = 0; this.compliance[id] = 0; this.lambda[id] = 0; this.force[id] = 0; this._live[id] = 0;
    this.anchor[id * 3] = 0; this.anchor[id * 3 + 1] = 0; this.anchor[id * 3 + 2] = 0;
    return id;
  }
  _checkParticle(p, who) {
    const P = this.particles;
    if (!(p >= 0 && p < P.count) || !P.alive[p]) throw new RangeError(`ParticleConstraints.${who}: no particle ${p}`);
  }
  _dist(a, b) {
    const x = this.particles.pos, a3 = a * 3, b3 = b * 3;
    const dx = x[a3] - x[b3], dy = x[a3 + 1] - x[b3 + 1], dz = x[a3 + 2] - x[b3 + 2];
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  }
  _two(type, a, b, rest, compliance, who) {
    this._checkParticle(a, who); this._checkParticle(b, who);
    if (a === b) throw new RangeError(`ParticleConstraints.${who}: both ends are particle ${a}`);
    const c = this._slot();
    this.type[c] = type; this.p0[c] = a; this.p1[c] = b;
    this.rest[c] = rest === undefined || rest === null ? this._dist(a, b) : rest;
    this.compliance[c] = compliance > 0 ? compliance : 0;
    this._ref(c, 1);
    return c;
  }
  /** |p_a - p_b| = restLength (default: the current distance). Returns the constraint id. */
  addDistance(a, b, restLength, compliance = 0) { return this._two(DISTANCE, a, b, restLength, compliance, 'addDistance'); }
  /** Quick bending across the edge a-b: |p_c - p_d| = their current distance (see the header). */
  addBending(a, b, c, d, compliance = 0) {
    this._checkParticle(a, 'addBending'); this._checkParticle(b, 'addBending');
    const ci = this._two(BENDING, c, d, undefined, compliance, 'addBending');
    this._ref(ci, -1);
    this.e0[ci] = a; this.e1[ci] = b;
    this._ref(ci, 1);
    return ci;
  }
  /** One-sided |p_q - p_p| <= maxLength (default: the current distance): a long-range attachment (see the header). */
  addTether(p, q, maxLength, compliance = 0) { return this._two(TETHER, p, q, maxLength, compliance, 'addTether'); }
  /**
   * Hold particle p at the body-local point (ax, ay, az) of `body` (WORLD_BODY = -1: the world point). Two-way.
   * @returns {number} constraint id
   */
  addAttachment(p, body, ax, ay, az, compliance = 0) { return this._bodyConstraint(ATTACHMENT, p, body, ax, ay, az, 0, compliance, 'addAttachment'); }
  /**
   * One-sided |p - t| <= maxLength between particle p and the body-local point (ax, ay, az) of `body` (extension; the
   * builders tie a pinned particle to a dynamic body's attachment point with it). Two-way like an attachment.
   * @returns {number} constraint id
   */
  addBodyTether(p, body, ax, ay, az, maxLength, compliance = 0) {
    const c = this._bodyConstraint(BODY_TETHER, p, body, ax, ay, az, 0, compliance, 'addBodyTether');
    this.rest[c] = maxLength === undefined || maxLength === null ? this._bodyPointDist(c) : maxLength;
    return c;
  }
  _bodyConstraint(type, p, body, ax, ay, az, rest, compliance, who) {
    this._checkParticle(p, who);
    const b = body === null || body === undefined ? WORLD_BODY : body | 0;
    const B = this.bodies;
    if (b < WORLD_BODY || (b >= 0 && B && (!(b < B.count) || !B.alive[b]))) throw new RangeError(`ParticleConstraints.${who}: no body ${body}`);
    const c = this._slot();
    this.type[c] = type; this.p0[c] = p; this.body[c] = b; this.rest[c] = rest;
    this.anchor[c * 3] = +ax || 0; this.anchor[c * 3 + 1] = +ay || 0; this.anchor[c * 3 + 2] = +az || 0;
    this.compliance[c] = compliance > 0 ? compliance : 0;
    const x = this.particles.pos, p3 = p * 3, c3 = c * 3;
    this.lastPos[c3] = x[p3]; this.lastPos[c3 + 1] = x[p3 + 1]; this.lastPos[c3 + 2] = x[p3 + 2];
    this._attachments++;
    this._ref(c, 1);
    return c;
  }
  // current distance between the particle and the body point of constraint c (construction)
  _bodyPointDist(c) {
    const x = this.particles.pos, p3 = this.p0[c] * 3, A = this.anchor, c3 = c * 3, b = this.body[c], B = this.bodies;
    let tx = A[c3], ty = A[c3 + 1], tz = A[c3 + 2];
    if (b >= 0 && B) {
      const Q = B.quat, o4 = b * 4, o3 = b * 3, out = [0, 0, 0];
      rotateInto(Q[o4], Q[o4 + 1], Q[o4 + 2], Q[o4 + 3], tx, ty, tz, out, 0);
      tx = B.pos[o3] + out[0]; ty = B.pos[o3 + 1] + out[1]; tz = B.pos[o3 + 2] + out[2];
    }
    return Math.sqrt((x[p3] - tx) * (x[p3] - tx) + (x[p3 + 1] - ty) * (x[p3 + 1] - ty) + (x[p3 + 2] - tz) * (x[p3 + 2] - tz));
  }
  /** Remove a constraint (its id is reused lowest first). */
  remove(ci) {
    if (!(ci >= 0 && ci < this.count) || !this.alive[ci]) return;
    if (this.type[ci] === ATTACHMENT || this.type[ci] === BODY_TETHER) this._attachments--;
    this._ref(ci, -1);
    this._size--;
    this.alive[ci] = 0; this.lambda[ci] = 0; this.force[ci] = 0; this._live[ci] = 0;
    if (ci < this._lowFree) this._lowFree = ci;
  }
  /**
   * Remove every constraint that names particle `id` (acting pair or bending edge); ParticleStore.free calls it through
   * the free hook. O(1) when none does (per-particle reference counts), else one scan. Returns how many.
   */
  detachParticle(id) {
    if (!(id >= 0 && id < this._refs.length) || this._refs[id] === 0) return 0;
    let k = 0;
    for (let c = 0; c < this.count && this._refs[id] > 0; c++) {
      if (this.alive[c] && (this.p0[c] === id || this.p1[c] === id || this.e0[c] === id || this.e1[c] === id)) { this.remove(c); k++; }
    }
    return k;
  }
  /** Remove every attachment to body `id` (DynamicsWorld.removeBody calls it). Returns how many. */
  detachBody(id) {
    if (this._attachments === 0) return 0;
    let k = 0;
    for (let c = 0; c < this.count; c++) if (this.alive[c] && (this.type[c] === ATTACHMENT || this.type[c] === BODY_TETHER) && this.body[c] === id) { this.remove(c); k++; }
    return k;
  }

  // ---- solver hooks (DynamicsWorld calls them; see DynamicsWorld.js "Particles") ---------------------------------
  /** Which attachment / body-tether bodies move this step (awake dynamic bodies; a sleeping one is immovable). */
  beginStep(world, dt) {
    if (this._attachments === 0) return;
    const B = world.bodies, n = this.count, live = this._live, alive = this.alive, type = this.type, body = this.body;
    for (let c = 0; c < n; c++) {
      live[c] = 0;
      if (alive[c] === 0 || (type[c] !== ATTACHMENT && type[c] !== BODY_TETHER)) continue;
      const b = body[c];
      if (b < 0 || b >= B.count || B.alive[b] === 0 || B.type[b] !== DYN || !(B.invMass[b] > 0) || (B.flags[b] & ASLEEP) !== 0) continue;
      live[c] = 1;
    }
  }

  /** One Gauss-Seidel sweep over the constraints in ascending id (lambdas reset at iteration 0). */
  solvePositions(world, h, it) {
    const n = this.count;
    if (n === 0) return;
    const P = world.particles, pos = P.pos, wInv = P.invMass, palive = P.alive;
    const alive = this.alive, type = this.type, p0 = this.p0, p1 = this.p1, rest = this.rest, comp = this.compliance;
    const lam = this.lambda, force = this.force;
    if (it === 0) lam.fill(0, 0, n);
    const ih2 = 1 / (h * h);
    _sc[0] = ih2;
    for (let c = 0; c < n; c++) {
      if (alive[c] === 0) continue;
      const t = type[c];
      if (t === ATTACHMENT || t === BODY_TETHER) { this._attach(world, c, t); continue; }
      const a = p0[c], b = p1[c];
      if (palive[a] === 0 || palive[b] === 0) { force[c] = 0; continue; }
      const a3 = a * 3, b3 = b * 3;
      const dx = pos[a3] - pos[b3], dy = pos[a3 + 1] - pos[b3 + 1], dz = pos[a3 + 2] - pos[b3 + 2];
      const d2 = dx * dx + dy * dy + dz * dz;
      if (!(d2 > 0)) { force[c] = 0; continue; }       // coincident: no direction
      const len = Math.sqrt(d2), C = len - rest[c], l0 = lam[c];
      if (t === TETHER && C <= 0 && l0 === 0) { force[c] = 0; continue; } // slack tether
      const wa = wInv[a], wb = wInv[b], at = comp[c] * ih2, w = wa + wb + at;
      if (!(w > 0)) { force[c] = 0; continue; }         // both ends immovable
      let dl = (-C - at * l0) / w;
      if (t === TETHER && l0 + dl > 0) dl = -l0;       // a tether only pulls (lambda <= 0)
      const l1 = l0 + dl;
      lam[c] = l1;
      force[c] = (l1 < 0 ? -l1 : l1) * ih2;
      if (dl === 0) continue;
      const s = dl / len;                              // p_a += w_a dLambda n, n = d / len
      if (wa > 0) { const k = wa * s; pos[a3] += k * dx; pos[a3 + 1] += k * dy; pos[a3 + 2] += k * dz; }
      if (wb > 0) { const k = wb * s; pos[b3] -= k * dx; pos[b3 + 1] -= k * dy; pos[b3 + 2] -= k * dz; }
    }
  }

  // attachment c: C = |p - t|, t = x_b + R_b anchor (or the world point); body tether: C = |p - t| - rest, only while
  // > 0 (lambda <= 0); two-way with an awake dynamic body
  _attach(world, c, type) {
    const P = world.particles, pos = P.pos, p = this.p0[c];
    if (P.alive[p] === 0) { this.force[c] = 0; return; }
    const A = this.anchor, c3 = c * 3, p3 = p * 3, b = this.body[c];
    let tx = A[c3], ty = A[c3 + 1], tz = A[c3 + 2], rx = 0, ry = 0, rz = 0;
    const B = world.bodies;
    if (b >= 0) {
      if (b >= B.count || B.alive[b] === 0) { this.force[c] = 0; return; }
      const Q = B.quat, o4 = b * 4, qx = Q[o4], qy = Q[o4 + 1], qz = Q[o4 + 2], qw = Q[o4 + 3];
      const ux = 2 * (qy * tz - qz * ty), uy = 2 * (qz * tx - qx * tz), uz = 2 * (qx * ty - qy * tx);
      rx = tx + qw * ux + (qy * uz - qz * uy);
      ry = ty + qw * uy + (qz * ux - qx * uz);
      rz = tz + qw * uz + (qx * uy - qy * ux);
      const o3 = b * 3;
      tx = B.pos[o3] + rx; ty = B.pos[o3 + 1] + ry; tz = B.pos[o3 + 2] + rz;
    }
    const dx = pos[p3] - tx, dy = pos[p3 + 1] - ty, dz = pos[p3 + 2] - tz;
    const d2 = dx * dx + dy * dy + dz * dz;
    if (!(d2 > 0)) { this.force[c] = 0; return; }
    const len = Math.sqrt(d2), nx = dx / len, ny = dy / len, nz = dz / len;
    const live = this._live[c] === 1;
    let wb = 0, ix = 0, iy = 0, iz = 0;
    if (live) {
      // generalized inverse mass m^-1 + (r x n)^T I_w^-1 (r x n): one 3x3 apply
      _c[0] = ry * nz - rz * ny; _c[1] = rz * nx - rx * nz; _c[2] = rx * ny - ry * nx;
      invInertiaApply(B, b, _c, _i);
      ix = _i[0]; iy = _i[1]; iz = _i[2];
      wb = B.invMass[b] + _c[0] * ix + _c[1] * iy + _c[2] * iz;
    }
    const ih2 = _sc[0], wp = P.invMass[p], at = this.compliance[c] * ih2, w = wp + wb + at, l0 = this.lambda[c];
    const C = len - this.rest[c];
    if (type === BODY_TETHER && C <= 0 && l0 === 0) { this.force[c] = 0; return; } // slack
    if (!(w > 0)) { this.force[c] = 0; return; }       // both sides immovable
    let dl = (-C - at * l0) / w;
    if (type === BODY_TETHER && l0 + dl > 0) dl = -l0;
    const l1 = l0 + dl;
    this.lambda[c] = l1;
    this.force[c] = (l1 < 0 ? -l1 : l1) * ih2;
    if (wp > 0) { const k = wp * dl; pos[p3] += k * nx; pos[p3 + 1] += k * ny; pos[p3 + 2] += k * nz; }
    if (live) {
      const k = -B.invMass[b] * dl;
      _d[0] = k * nx; _d[1] = k * ny; _d[2] = k * nz;
      _d[3] = -dl * ix; _d[4] = -dl * iy; _d[5] = -dl * iz;
      displaceBodyFrom(B, b, _d, 0);
    }
  }

  /** No velocity pass (the world's particleDamping is the only velocity-level term). */
  solveVelocities(world, h, it) {}

  /**
   * Sleep of the bodies held by attachments / body tethers (see the header): an awake body's sleep timer restarts while
   * the constraint's particle moved more than sleep.linear * dt over the step; a sleeping body wakes when that particle
   * moves, or when the constraint's force departs from its value while asleep by more than 1e-3 N + 1 % (a tug).
   */
  endStep(world, dt) {
    if (this._attachments === 0) return;
    const sp = world.sleepParams, B = world.bodies, x = world.particles.pos, n = this.count;
    const alive = this.alive, type = this.type, body = this.body, p0 = this.p0, last = this.lastPos, rf = this.restForce;
    const lin = sp ? sp.linear * dt : 0, lin2 = lin * lin;
    for (let c = 0; c < n; c++) {
      if (alive[c] === 0 || (type[c] !== ATTACHMENT && type[c] !== BODY_TETHER)) continue;
      const p3 = p0[c] * 3, c3 = c * 3;
      const dx = x[p3] - last[c3], dy = x[p3 + 1] - last[c3 + 1], dz = x[p3 + 2] - last[c3 + 2];
      last[c3] = x[p3]; last[c3 + 1] = x[p3 + 1]; last[c3 + 2] = x[p3 + 2];
      const b = body[c];
      if (!sp || b < 0 || b >= B.count || B.alive[b] === 0 || B.type[b] !== DYN) continue;
      const moved = dx * dx + dy * dy + dz * dz > lin2;
      if ((B.flags[b] & ASLEEP) === 0) { rf[c] = -1; if (moved) B.sleepTime[b] = 0; continue; }
      if (this._live[c] === 1) { rf[c] = -1; continue; }   // fell asleep this step: measure from the next one
      if (moved) { rf[c] = -1; world.wake(b); continue; }
      const f = this.force[c], f0 = rf[c];
      if (f0 < 0) rf[c] = f;
      else if ((f > f0 ? f - f0 : f0 - f) > WAKE_FORCE + WAKE_REL * f0) { rf[c] = -1; world.wake(b); }
    }
  }

  /** Plain snapshot (typed-array copies up to count). */
  serialize() {
    const n = this.count, out = { capacity: this.capacity, count: n, lowFree: this._lowFree, nextSerial: this._nextSerial };
    for (const k of CONSTRAINT_FIELDS) { const a = this[k], s = a.length / this.capacity; out[k] = a.slice(0, n * s); }
    return out;
  }
  /** Restore serialize()'s snapshot (bit-identical). */
  restore(d) {
    const n = d.count | 0;
    if (n > this.capacity) this._grow(n);
    for (const k of CONSTRAINT_FIELDS) { this[k].fill(k === 'restForce' ? -1 : 0); if (d[k] !== undefined) this[k].set(asArray(d[k])); }
    this._live.fill(0);
    this.count = n;
    this._lowFree = Math.min(d.lowFree | 0, n);
    this._nextSerial = d.nextSerial > 0 ? d.nextSerial >>> 0 : 1;
    this._cursor = -1; this._cursorEnd = -1;
    let att = 0, live = 0;
    this._refs.fill(0);
    for (let c = 0; c < n; c++) {
      if (!this.alive[c]) continue;
      live++;
      this._ref(c, 1);
      if (this.type[c] === ATTACHMENT || this.type[c] === BODY_TETHER) att++;
    }
    this._attachments = att; this._size = live;
  }
}

// ---- builders -------------------------------------------------------------------------------------------------------

function needParticles(world, who) {
  if (!world || !world.particles || !world.constraints) throw new TypeError(`${who}: the world has no particles (new DynamicsWorld({ maxParticles }))`);
  return world.particles;
}
// q v q* into out[o..o+3] (construction only)
function rotateInto(qx, qy, qz, qw, x, y, z, out, o) {
  const tx = 2 * (qy * z - qz * y), ty = 2 * (qz * x - qx * z), tz = 2 * (qx * y - qy * x);
  out[o] = x + qw * tx + (qy * tz - qz * ty);
  out[o + 1] = y + qw * ty + (qz * tx - qx * tz);
  out[o + 2] = z + qw * tz + (qx * ty - qy * tx);
}
// body-local point of world point (x, y, z) for body b (WORLD_BODY / no BodyStore: the world point) -> out[0..3]
function bodyLocal(world, b, x, y, z, out) {
  if (b < 0) { out[0] = x; out[1] = y; out[2] = z; return; }
  const B = world.bodies;
  if (!(b < B.count) || !B.alive[b]) throw new RangeError(`attachment: no body ${b}`);
  const Q = B.quat, o4 = b * 4, P = B.pos, o3 = b * 3;
  rotateInto(-Q[o4], -Q[o4 + 1], -Q[o4 + 2], Q[o4 + 3], x - P[o3], y - P[o3 + 1], z - P[o3 + 2], out, 0);
}
function checkBody(world, b, who) {
  const B = world.bodies;
  if (b < WORLD_BODY || (b >= 0 && (!(b < B.count) || !B.alive[b]))) throw new RangeError(`${who}: no body ${b} to attach to`);
}
function bodyOf(pin) { return pin.body === undefined || pin.body === null ? WORLD_BODY : pin.body | 0; }

// Add the constraint specs [type, a, b, e0, e1, compliance, attach] (indices into ids) as one contiguous block; returns
// the constraint ids. On an error the constraints added so far are removed and the reservation is ended.
function emit(world, ids, specs, order, attach) {
  const K = world.constraints, cids = new Int32Array(order.length);
  let done = 0;
  try {
    K.reserve(order.length);
    for (; done < order.length; done++) cids[done] = emitOne(world, ids, specs[order[done]], attach);
  } catch (e) {
    for (let q = 0; q < done; q++) K.remove(cids[q]);
    throw e;
  } finally {
    K.endReserve();
  }
  return cids;
}
function emitOne(world, ids, s, attach) {
  const K = world.constraints, tmp = [0, 0, 0], t = s[0];
  {
    if (t === DISTANCE) return K.addDistance(ids[s[1]], ids[s[2]], undefined, s[5]);
    if (t === BENDING) return K.addBending(ids[s[3]], ids[s[4]], ids[s[1]], ids[s[2]], s[5]);
    if (t === TETHER) return K.addTether(ids[s[1]], ids[s[2]], undefined, s[5]);
    if (t === BODY_TETHER) {
      // pinned particle s[1] -> the body point of the attachment of particle s[2]: max = |pin - particle| + |particle - point|
      const at = attach[s[6]], id = ids[s[2]], pin = ids[s[1]];
      const off = anchorLocal(world, at, id, tmp);
      return K.addBodyTether(pin, at.body, tmp[0], tmp[1], tmp[2], distOf(world.particles, pin, id) + off, s[5]);
    }
    const at = attach[s[6]], id = ids[s[1]];
    anchorLocal(world, at, id, tmp);
    return K.addAttachment(id, at.body, tmp[0], tmp[1], tmp[2], s[5]);
  }
}
// body-local attachment point of pin record `at` for particle id -> out[0..3] (default: where the particle is); returns
// the distance between the particle and that point
function anchorLocal(world, at, id, out) {
  const P = world.particles, p3 = id * 3;
  if (!at.anchor) { bodyLocal(world, at.body, P.pos[p3], P.pos[p3 + 1], P.pos[p3 + 2], out); return 0; }
  out[0] = vx(at.anchor); out[1] = vy(at.anchor); out[2] = vz(at.anchor);
  const B = world.bodies, b = at.body, w = [out[0], out[1], out[2]];
  if (b >= 0) {
    const Q = B.quat, o4 = b * 4, o3 = b * 3;
    rotateInto(Q[o4], Q[o4 + 1], Q[o4 + 2], Q[o4 + 3], out[0], out[1], out[2], w, 0);
    w[0] += B.pos[o3]; w[1] += B.pos[o3 + 1]; w[2] += B.pos[o3 + 2];
  }
  return Math.sqrt((P.pos[p3] - w[0]) ** 2 + (P.pos[p3 + 1] - w[1]) ** 2 + (P.pos[p3 + 2] - w[2]) ** 2);
}
const isDynamicBody = (world, b) => b >= 0 && world.bodies.type[b] === BODY.DYNAMIC;
// move the particles of world-pinned pins onto their anchors (after the rest lengths were taken)
function applyMoves(world, ids, moves) {
  const P = world.particles;
  for (const [k, x, y, z] of moves) {
    const p3 = ids[k] * 3;
    P.pos[p3] = x; P.pos[p3 + 1] = y; P.pos[p3 + 2] = z;
    P.prevPos[p3] = x; P.prevPos[p3 + 1] = y; P.prevPos[p3 + 2] = z;
  }
}
// dispose(): removes exactly the constraints the builder emitted (still alive: one the caller removed may have been
// reused by a foreign add), then frees the particles
function makeDispose(world, ids, cids) {
  const K0 = world.constraints, serials = new Uint32Array(cids.length);
  for (let q = 0; q < cids.length; q++) serials[q] = K0.serial[cids[q]];
  let done = false;
  return () => {
    if (done) return;
    done = true;
    const K = world.constraints, P = world.particles;
    if (K) for (let q = 0; q < cids.length; q++) { const c = cids[q]; if (K.alive[c] && K.serial[c] === serials[q]) K.remove(c); }
    if (P) for (let k = 0; k < ids.length; k++) P.free(ids[k]);
  };
}
// emit + rollback of the builder's particles on an error
function emitOrFree(world, ids, specs, order, attach) {
  try { return emit(world, ids, specs, order, attach); } catch (e) {
    for (let k = 0; k < ids.length; k++) world.particles.free(ids[k]);
    throw e;
  }
}
function quatOf(q) {
  let x = vx(q), y = vy(q), z = vz(q), w = q ? vw(q) : 1;
  const l = Math.sqrt(x * x + y * y + z * z + w * w);
  if (l > 0) { x /= l; y /= l; z /= l; w /= l; } else { x = 0; y = 0; z = 0; w = 1; }
  return [x, y, z, w];
}

const CLOTH_PINS = Object.freeze(['top-left', 'top-right', 'top', 'left', 'right', 'bottom-left', 'bottom-right', 'bottom']);
// grid indices of a string pin
function clothPinIndices(pin, nx, ny) {
  const out = [];
  if (pin === 'top-left') out.push((ny - 1) * nx);
  else if (pin === 'top-right') out.push((ny - 1) * nx + nx - 1);
  else if (pin === 'bottom-left') out.push(0);
  else if (pin === 'bottom-right') out.push(nx - 1);
  else if (pin === 'top') for (let i = 0; i < nx; i++) out.push((ny - 1) * nx + i);
  else if (pin === 'bottom') for (let i = 0; i < nx; i++) out.push(i);
  else if (pin === 'left') for (let j = 0; j < ny; j++) out.push(j * nx);
  else for (let j = 0; j < ny; j++) out.push(j * nx + nx - 1);
  return out;
}

/** Cloth builder (see the header). */
export const Cloth = Object.freeze({
  /**
   * A rectangular nx x ny particle grid with structural, shear, bending (and tether / attachment) constraints.
   * @param {object} world  a DynamicsWorld with maxParticles > 0
   * @param {object} [o]  see the file header
   * @param {number[]|{x,y,z}} [o.position=[0,0,0]]  the CENTRE of the rest rectangle (not a corner)
   * @param {number[]|{x,y,z,w}} [o.quaternion=[0,0,0,1]]  rows along its local +X (i) and +Y (j), normal local +Z
   * @param {number} [o.stretch=0]  structural compliance (m/N; 0 = inextensible, tethers on)
   * @param {number} [o.shear=1e-4]  diagonal compliance (0 with stretch 0: rigid cells, a developable sheet)
   * @param {number} [o.bend=1e-5]  bending compliance
   * @param {Array<string|object>} [o.pins=[]]  one pin per particle when a pin has a `body`
   */
  grid(world, o = {}) {
    const P = needParticles(world, 'Cloth.grid');
    const { nx = 32, ny = 32, spacing = 0.05, position = [0, 0, 0], quaternion = [0, 0, 0, 1], mass = 1, stretch = 0, shear = 1e-4, bend = 1e-5, pins = [] } = o;
    const radius = o.radius ?? spacing * 0.4, tethers = o.tethers ?? stretch === 0;
    if (!(Number.isInteger(nx) && Number.isInteger(ny) && nx >= 2 && ny >= 2)) throw new RangeError('Cloth.grid: nx and ny must be integers >= 2');
    if (!(spacing > 0) || !(mass > 0)) throw new RangeError('Cloth.grid: spacing and mass must be > 0');
    const N = nx * ny;
    if (P.size + N > P.capacity) throw new RangeError(`Cloth.grid: ${N} particles do not fit (${P.capacity - P.size} free)`);
    // validate before allocating: plain pins (strings, { i, j }) may overlap; a body pin must be its particle's only pin
    const plain = new Uint8Array(N), bodyPin = new Uint8Array(N);
    for (const pin of pins) {
      if (typeof pin === 'string') {
        if (!CLOTH_PINS.includes(pin)) throw new RangeError(`Cloth.grid: unknown pin '${pin}'`);
        for (const k of clothPinIndices(pin, nx, ny)) plain[k] = 1;
        continue;
      }
      const i = pin && pin.i, j = pin && pin.j;
      if (!(Number.isInteger(i) && Number.isInteger(j) && i >= 0 && i < nx && j >= 0 && j < ny)) throw new RangeError('Cloth.grid: a pin needs integer { i, j } inside the grid');
      if (pin.body === undefined) { plain[j * nx + i] = 1; continue; }
      checkBody(world, bodyOf(pin), 'Cloth.grid');
      if (bodyPin[j * nx + i]) throw new Error(`Cloth.grid: particle (${i}, ${j}) has two body pins (one pin per particle)`);
      bodyPin[j * nx + i] = 1;
    }
    for (let k = 0; k < N; k++) if (bodyPin[k] && plain[k]) throw new Error(`Cloth.grid: particle (${k % nx}, ${(k / nx) | 0}) has a body pin and another pin (one pin per particle)`);
    const group = o.group ?? P.newGroup();
    const [qx, qy, qz, qw] = quatOf(quaternion), cx = vx(position), cy = vy(position), cz = vz(position);
    const pm = mass / N, ids = new Int32Array(N), r = new Float64Array(3);
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const id = P.alloc(), k = j * nx + i, p3 = id * 3;
        ids[k] = id;
        rotateInto(qx, qy, qz, qw, (i - (nx - 1) / 2) * spacing, (j - (ny - 1) / 2) * spacing, 0, r, 0);
        P.pos[p3] = cx + r[0]; P.pos[p3 + 1] = cy + r[1]; P.pos[p3 + 2] = cz + r[2];
        P.prevPos[p3] = P.pos[p3]; P.prevPos[p3 + 1] = P.pos[p3 + 1]; P.prevPos[p3 + 2] = P.pos[p3 + 2];
        P.setMass(id, pm); P.radius[id] = radius; P.group[id] = group;
      }
    }
    // pins: pinned in place, or attachments
    const pinned = new Uint8Array(N), attachAt = new Int32Array(N).fill(-1), attach = [], moves = [];
    const pinK = (k) => { pinned[k] = 1; P.setPinned(ids[k], true); };
    for (const pin of pins) {
      if (typeof pin === 'string') { for (const k of clothPinIndices(pin, nx, ny)) pinK(k); continue; }
      const k = pin.j * nx + pin.i, b = pin.body === undefined ? undefined : bodyOf(pin);
      if (b === undefined || b === WORLD_BODY) {   // pinned in place, or (world) pinned at the anchor point
        pinK(k);
        if (b === WORLD_BODY && pin.anchor) moves.push([k, vx(pin.anchor), vy(pin.anchor), vz(pin.anchor)]);
        continue;
      }
      attachAt[k] = attach.length; attach.push({ body: b, anchor: pin.anchor || null });
    }
    // constraint specs [type, a, b, e0, e1, compliance, attach] in grid indices
    const specs = [], K = (i, j) => j * nx + i;
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx - 1; i++) specs.push([DISTANCE, K(i, j), K(i + 1, j), -1, -1, stretch, -1]);
    for (let j = 0; j < ny - 1; j++) for (let i = 0; i < nx; i++) specs.push([DISTANCE, K(i, j), K(i, j + 1), -1, -1, stretch, -1]);
    for (let j = 0; j < ny - 1; j++) {
      for (let i = 0; i < nx - 1; i++) {
        specs.push([DISTANCE, K(i, j), K(i + 1, j + 1), -1, -1, shear, -1]);   // "/"
        specs.push([DISTANCE, K(i + 1, j), K(i, j + 1), -1, -1, shear, -1]);   // "\"
      }
    }
    // bending across interior edges, both triangulations: horizontal edge (i, j)-(i + 1, j), 0 < j < ny - 1
    for (let j = 1; j < ny - 1; j++) {
      for (let i = 0; i < nx - 1; i++) {
        specs.push([BENDING, K(i + 1, j + 1), K(i, j - 1), K(i, j), K(i + 1, j), bend, -1]);   // "/" split
        specs.push([BENDING, K(i, j + 1), K(i + 1, j - 1), K(i, j), K(i + 1, j), bend, -1]);   // "\" split
      }
    }
    // vertical edge (i, j)-(i, j + 1), 0 < i < nx - 1
    for (let j = 0; j < ny - 1; j++) {
      for (let i = 1; i < nx - 1; i++) {
        specs.push([BENDING, K(i - 1, j), K(i + 1, j + 1), K(i, j), K(i, j + 1), bend, -1]);
        specs.push([BENDING, K(i - 1, j + 1), K(i + 1, j), K(i, j), K(i, j + 1), bend, -1]);
      }
    }
    if (tethers) {
      // tether anchors: the pinned particles, or (none pinned) the attached ones
      const pinnedList = [], attachedList = [];
      for (let k = 0; k < N; k++) { if (pinned[k]) pinnedList.push(k); else if (attachAt[k] >= 0) attachedList.push(k); }
      const anchors = pinnedList.length ? pinnedList : attachedList;
      const nearest = (k) => {
        let best = -1, bd = Infinity;
        for (const a of anchors) { const d = distOf(P, ids[k], ids[a]); if (d < bd) { bd = d; best = a; } }
        return best;
      };
      if (anchors.length) for (let k = 0; k < N; k++) if (!pinned[k] && (pinnedList.length > 0 || attachAt[k] < 0)) specs.push([TETHER, k, nearest(k), -1, -1, 0, -1]);
      // a dynamic body held by the cloth is tied to the nearest pin directly (see addBodyTether)
      if (pinnedList.length) for (const k of attachedList) if (isDynamicBody(world, attach[attachAt[k]].body)) specs.push([BODY_TETHER, nearest(k), k, -1, -1, 0, attachAt[k]]);
    }
    for (let k = 0; k < N; k++) if (attachAt[k] >= 0) specs.push([ATTACHMENT, k, -1, -1, -1, 0, attachAt[k]]);
    const order = mirrorOrder(specs, nx);
    const cids = emitOrFree(world, ids, specs, order, attach), cFirst = cids.length ? cids[0] : 0, cCount = cids.length;
    applyMoves(world, ids, moves);
    let tris = null;
    return {
      first: ids[0], count: N, nx, ny, ids, group, constraints: { first: cFirst, count: cCount },
      /** Uint32Array of indices into `ids`, 3 per triangle, CCW seen from the rest plane's +normal (allocated once). */
      triangles() {
        if (tris) return tris;
        tris = new Uint32Array((nx - 1) * (ny - 1) * 6);
        let t = 0;
        for (let j = 0; j < ny - 1; j++) {
          for (let i = 0; i < nx - 1; i++) {
            const a = K(i, j), b = K(i + 1, j), c = K(i + 1, j + 1), d = K(i, j + 1);
            tris[t++] = a; tris[t++] = b; tris[t++] = c;
            tris[t++] = a; tris[t++] = c; tris[t++] = d;
          }
        }
        return tris;
      },
      /** Particle id of grid point (i, j). */
      indexOf(i, j) { return ids[j * nx + i]; },
      dispose: makeDispose(world, ids, cids),
    };
  },
});

function distOf(P, a, b) {
  const x = P.pos, a3 = a * 3, b3 = b * 3;
  const dx = x[a3] - x[b3], dy = x[a3 + 1] - x[b3 + 1], dz = x[a3 + 2] - x[b3 + 2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

// Emission order of the cloth's specs: natural order, each constraint directly followed by its left-right mirror image
// (i -> nx - 1 - i), so the Gauss-Seidel sweep is mirror-equivariant (see the header). Map used for lookups only.
function mirrorOrder(specs, nx) {
  const mi = (k) => { const i = k % nx; return k - i + (nx - 1 - i); };
  const key = (t, a, b) => (a < b ? `${t}:${a}:${b}` : `${t}:${b}:${a}`);
  const id = (k) => k;
  const keyOf = (s, m) => (s[0] === ATTACHMENT ? `${ATTACHMENT}:${m(s[1])}` : s[0] === BODY_TETHER ? `${BODY_TETHER}:${m(s[2])}` : key(s[0], m(s[1]), m(s[2])));
  const index = new Map();
  for (let q = 0; q < specs.length; q++) { const k = keyOf(specs[q], id); if (!index.has(k)) index.set(k, q); }
  const done = new Uint8Array(specs.length), order = [];
  for (let q = 0; q < specs.length; q++) {
    if (done[q]) continue;
    done[q] = 1; order.push(q);
    const k = keyOf(specs[q], mi);
    const m = index.get(k);
    if (m !== undefined && !done[m]) { done[m] = 1; order.push(m); }
  }
  return order;
}

/** Rope builder (see the header). */
export const Rope = Object.freeze({
  /**
   * A particle chain from `from` to `to` with distance, bending (and tether / attachment) constraints.
   * @param {object} world  a DynamicsWorld with maxParticles > 0
   * @param {object} [o]  see the file header
   */
  chain(world, o = {}) {
    const P = needParticles(world, 'Rope.chain');
    const { from = [0, 0, 0], to = [0, -1, 0], segments = 20, mass = 0.5, radius = 0.02, stretch = 0, bend = 1e-4, pins = ['from'] } = o;
    const tethers = o.tethers ?? stretch === 0;
    if (!(Number.isInteger(segments) && segments >= 1)) throw new RangeError('Rope.chain: segments must be an integer >= 1');
    if (!(mass > 0)) throw new RangeError('Rope.chain: mass must be > 0');
    const N = segments, n = N + 1;
    if (P.size + n > P.capacity) throw new RangeError(`Rope.chain: ${n} particles do not fit (${P.capacity - P.size} free)`);
    // validate before allocating: plain pins of one end may repeat; a body pin must be its end's only pin
    const plainEnd = [0, 0], bodyEnd = [0, 0];
    for (const pin of pins) {
      const end = typeof pin === 'string' ? pin : pin && pin.end;
      if (end !== 'from' && end !== 'to') throw new RangeError(`Rope.chain: unknown pin ${JSON.stringify(pin)}`);
      const e = end === 'from' ? 0 : 1;
      if (typeof pin === 'string' || pin.body === undefined) { plainEnd[e]++; continue; }
      checkBody(world, bodyOf(pin), 'Rope.chain');
      bodyEnd[e]++;
    }
    for (let e = 0; e < 2; e++) {
      if (bodyEnd[e] > 1 || (bodyEnd[e] > 0 && plainEnd[e] > 0)) throw new Error(`Rope.chain: the '${e === 0 ? 'from' : 'to'}' end has a body pin and another pin (one pin per particle)`);
    }
    const group = o.group ?? P.newGroup();
    const fx = vx(from), fy = vy(from), fz = vz(from), tx = vx(to), ty = vy(to), tz = vz(to);
    const pm = mass / n, ids = new Int32Array(n);
    for (let k = 0; k < n; k++) {
      const id = P.alloc(), p3 = id * 3, f = k / N;
      ids[k] = id;
      P.pos[p3] = fx + (tx - fx) * f; P.pos[p3 + 1] = fy + (ty - fy) * f; P.pos[p3 + 2] = fz + (tz - fz) * f;
      P.prevPos[p3] = P.pos[p3]; P.prevPos[p3 + 1] = P.pos[p3 + 1]; P.prevPos[p3 + 2] = P.pos[p3 + 2];
      P.setMass(id, pm); P.radius[id] = radius; P.group[id] = group;
    }
    const anchored = [false, false], pinnedEnd = [false, false], attach = [], attachEnd = [], moves = [];
    for (const pin of pins) {
      const end = typeof pin === 'string' ? pin : pin && pin.end;
      const e = end === 'from' ? 0 : 1, k = e === 0 ? 0 : N, b = typeof pin === 'string' || pin.body === undefined ? undefined : bodyOf(pin);
      if (b === undefined || b === WORLD_BODY) {   // pinned in place, or (world) pinned at the anchor point
        P.setPinned(ids[k], true); pinnedEnd[e] = true;
        if (b === WORLD_BODY && pin.anchor) moves.push([k, vx(pin.anchor), vy(pin.anchor), vz(pin.anchor)]);
      } else { attach.push({ body: b, anchor: pin.anchor || null }); attachEnd.push(k); }
      anchored[e] = true;
    }
    const up = !anchored[0] && anchored[1];   // solve from the anchored end outward
    const specs = [], at = (q) => (up ? N - q : q);
    for (let q = 0; q < N; q++) specs.push([DISTANCE, at(q), up ? at(q) - 1 : at(q) + 1, -1, -1, stretch, -1]);
    for (let q = 0; q + 2 <= N; q++) { const m = up ? at(q) - 1 : at(q) + 1; specs.push([BENDING, at(q), up ? m - 1 : m + 1, m, m, bend, -1]); }
    if (tethers && (anchored[0] || anchored[1])) {
      // tether anchors: the pinned ends, or (none pinned) the attached ones; nearest end first, 'from' on a tie
      const tie = pinnedEnd[0] || pinnedEnd[1] ? pinnedEnd : anchored;
      for (let q = 0; q < n; q++) {
        const k = at(q);
        if ((k === 0 && tie[0]) || (k === N && tie[1])) continue;
        const e = !tie[1] ? 0 : !tie[0] ? N : k <= N - k ? 0 : N;
        specs.push([TETHER, k, e, -1, -1, 0, -1]);
      }
      // a dynamic body held by an end is tied to a pinned end directly (see addBodyTether)
      for (let a = 0; a < attach.length; a++) {
        const k = attachEnd[a], other = k === 0 ? 1 : 0;
        if (pinnedEnd[other] && isDynamicBody(world, attach[a].body)) specs.push([BODY_TETHER, other === 0 ? 0 : N, k, -1, -1, 0, a]);
      }
    }
    for (let a = 0; a < attach.length; a++) specs.push([ATTACHMENT, attachEnd[a], -1, -1, -1, 0, a]);
    const order = specs.map((_, q) => q);
    const cids = emitOrFree(world, ids, specs, order, attach), cFirst = cids.length ? cids[0] : 0, cCount = cids.length;
    applyMoves(world, ids, moves);
    const K = world.constraints;
    let length = 0;
    for (let q = 0; q < cids.length; q++) if (K.type[cids[q]] === DISTANCE) length += K.rest[cids[q]];
    return { first: ids[0], count: n, ids, group, constraints: { first: cFirst, count: cCount }, length, dispose: makeDispose(world, ids, cids) };
  },
});
