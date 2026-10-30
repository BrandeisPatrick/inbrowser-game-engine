// Joints of the dynamics world (pure JS, deterministic, Node-safe, no THREE): a structure-of-arrays JointStore and the
// XPBD position / velocity passes over it (Müller et al. 2020, "Detailed Rigid Body Simulation with Extended Position
// Based Dynamics", sections 3.3-3.6; primitives in xpbd.js).
//
//   const J = new JointStore(1024);
//   const door = J.add({ type: JOINT.HINGE, bodyA: id, bodyB: WORLD_BODY, anchorA: [-0.5, 0, 0], axisA: [0, 1, 0],
//                        limits: [0, 1.6], motor: { velocity: 1, maxForce: 50 } }, bodies);
//   per substep:  integrate ... solveJointsPositions(J, B, h) ... velocityFromPositions ... solveJointsVelocities(J, B, h)
//   jointAngle(J, B, door) -> rad;  jointDistance(J, B, id) -> m;  J.force[id] (N), J.broken[id]
//
// Contract notes. anchorB / axisB (when not given) and refB (always) are frozen from the creation pose: the joint
// starts satisfied at angle 0. motorTarget NaN is the velocity-drive sentinel (a number makes the motor a position
// servo). force / torque are the last substep's |lambda| / h^2 (summed over that substep's position iterations).
// Velocity recovery is xpbd.js's tan form. The motor reads the start-of-substep pose from prevPos / prevQuat of
// EVERY body type: integrateBody saves them for dynamic bodies, the world must do so for kinematic and static ones
// (DynamicsWorld does, and on setPose). Contact pairs: J.connected(a, b) is true for the bodies of an alive, unbroken
// joint with collideConnected false; DynamicsWorld's pair path must skip those pairs (allocation-free lookup; the set
// is rebuilt by add / remove / breaking, or by J.rebuildConnected() after editing collideConnected / broken by hand).
// Supported envelope: mass ratios up to ~300 across a joint at 4 substeps of 1/120 s with 0.1 m links are solved
// exactly (gaps ~1e-7 m); at ~1000 the trust region keeps a whipping chain bounded (no blow-up; transient gaps of a
// few cm, recovering) but it is no longer accurate: use more substeps there.
//
// Descriptor (add). type: JOINT.*; bodyA / bodyB: body ids, or WORLD_BODY (-1, also null / undefined for bodyB): the
// world has the identity pose, so its anchor / axis are world coordinates. anchorA / anchorB: local attachment points
// (anchorA defaults to the origin; anchorB defaults to where anchor A is at creation, so the joint starts satisfied).
// axisA / axisB: local joint axes (axisA defaults to [0, 1, 0]; axisB defaults to axis A carried into B's frame by the
// creation pose, i.e. [0, 1, 0] when both bodies start with the same orientation). refA: optional local reference
// direction perpendicular to axisA (angle zero; default: local X projected off the axis, local Z for an X axis); refB
// is always derived from the creation pose, so every joint starts at angle 0. limits: [lo, hi] (hinge: radians of the
// twist angle; slider: metres along the axis; distance: [min, max] length). motor: { velocity, maxForce, target? }
// (hinge, ball, d6: about the axis; slider: along it; `target` turns the velocity drive into a position servo).
// compliance (m/N, rad/(N m); 0 = rigid), damping (1/s), breakForce (N) and breakTorque (N m) (default Infinity; the
// bound of Breaking applies to LOCK rows only: a rope / limit / motor row of a breakable joint still passes its full
// impulse in the substep it breaks; <= 0 breaks on any force), distance (default: the creation length), rope
// (distance: only pulls), swing: max | [lo, hi] (cone angle between the axes, rad), twist: [lo, hi] | max (rad), linear: [x, y, z] each true (locked), false (free) or [lo, hi] (m)
// (d6; default all locked), collideConnected (flag only; the narrowphase filter reads it).
//
// Fields (SoA, index = joint id; vectors packed x, y, z at id * 3): count, alive, type, a, b, anchorA, anchorB, axisA,
// axisB, refA, refB, lo, hi, linLo / linHi (3n), swingLo / swingHi, twistLo / twistHi, motorOn, motorVel, motorMax,
// motorTarget (NaN = velocity drive), compliance, damping, breakForce, breakTorque, broken, collideConnected,
// lambdaPos (3n), lambdaRot (3n), force, torque (|lambdaPos| / h^2, |lambdaRot| / h^2 of the last pass). Runtime edits
// of lo / hi (hinge, slider, distance), the motor fields, compliance, damping and the break thresholds take effect at
// the next pass.
//
// Frames. Every joint carries two orthonormal local frames, X = ref, Y = axis, Z = ref x axis. In world space (p =
// x + R anchor, a = R axis, r = R ref; A for body a, B for body b; d = pA - pB):
//   linear coordinates  s_k = d . e_k, e = (rB, aB, rB x aB)                 (B's frame: B is the parent)
//   swing               theta_s = atan2(|aB x aA|, aB . aA), axis f1 = (aB x aA) / |aB x aA|
//   twist               half axis f0 = (aA + aB) / |aA + aB|, r1 / r2 = rA / rB projected off f0,
//                       phi = atan2(f0 . (r2 x r1), r2 . r1)      (detAtan2; the angle of A relative to B about f0)
//   angular basis       (f0, f1, f2 = f0 x f1)
// Constraint sets (rows; "lock" rows are bilateral C = coordinate - target and use the joint compliance, "limit" rows
// are active only while violated, C = coordinate - clamp(coordinate, lo, hi), rigid and one-sided: a violated lower
// bound may only push (lambda >= 0), an upper bound only pull back (lambda <= 0), so when the coupled solve would make
// a limit row pull, the clamp step deactivates it (an active-set step) instead of holding the coordinate at the bound):
//   FIXED     s_0 = s_1 = s_2 = 0 (anchors coincide);  theta_s = 0 (rows f1 with C = theta_s, f2 with C = 0);  phi = 0
//   BALL      s_0 = s_1 = s_2 = 0;  optional swing / twist limits as D6
//   HINGE     s_k = 0;  theta_s = 0 (axis A aligned with axis B: the rotation vector theta_s f1 = atan2-exact
//             version of the paper's aB x aA);  lo <= phi <= hi;  motor on phi
//   SLIDER    s_0 = s_2 = 0 (no offset perpendicular to the axis);  lo <= s_1 <= hi;  theta_s = 0;  phi = 0;  motor on s_1
//   DISTANCE  lo <= |d| <= hi (lo = hi = distance; a rope has lo = -Infinity), scalar along n = d / |d|; a zero
//             length (hi <= 1e-9) is a ball joint (three lock rows); coinciding anchors with a positive length push
//             apart along the start-of-substep direction
//   D6        per linear axis: locked (s_k = 0), limited, or free;  swing: theta_s <= max (or locked when max <= 0) and
//             >= lo;  twist: lo <= phi <= hi
// Motor row (velocity drive): C = phi - (phi_prev + v h) (phi_prev measured on the substep's start poses, prevQuat /
// prevPos of dynamic bodies, the current pose of static / kinematic ones), or C = phi - target (servo); the target is
// clamped into [lo, hi] (at a limit the motor holds instead of pushing through it), angle errors are wrapped to
// (-pi, pi], |lambda| <= maxForce h^2 (N m s^2 or N s^2), rigid otherwise. A violated limit takes the row over.
//
// Solve. All rows of one joint form one vector constraint with Jacobian rows (linear row k: +e_k at lever r_a = pA -
// x_a on a, -e_k at r_b = pA - x_b on b, the exact Jacobian of d . e_k with e_k turning with B; angular row: +f on
// a, -f on b; DISTANCE: +-n at the two anchors). With W_i = diag(m_i^-1, I_w,i^-1), K = sum_i J_i W_i J_i^T and
// alpha~ = compliance / h^2 per row, one pass takes the XPBD step of ALL joints of a connected joint tree at once:
//   (K + alpha~) dLambda = -C - alpha~ lambda;   x_i += m_i^-1 p_i,  q_i += 1/2 [I_i^-1 tau_i, 0] (x) q_i
// (p_i, tau_i: the summed impulses of the rows touching body i). The joint-space K of a tree is sparse: rows couple
// only through a shared body, so eliminating joints leaf-first (each body's child joints, keeping its parent joint)
// creates no fill beyond the joints that meet at one body. The solver roots every tree at a body (union-find over the
// movable bodies, BFS from the lowest slot), eliminates bodies in reverse BFS order with a dense partial LDL^T of
// K restricted to the joints at that body (plus the Schur block each child joint brings from below), and back-
// substitutes root first: O(sum of deg^3) per pass, linear in chain length. Static / kinematic bodies and the world
// are not tree nodes, so anchoring a chain at both ends forms no loop. Why: a Gauss-Seidel sweep (the paper's per-
// constraint projection) stretches a hanging N-link chain by ~N^2 h^2 g / 2 per substep (8.5 mm for 20 links at h =
// 1/480), and its scalar projection along the violation direction leaves an O(|C|) residual whenever the point
// mobility m^-1 - [r]x I^-1 [r]x is anisotropic or a position correction twists a locked rotation (1e-5 m / rad of
// permanent wobble in a fixed joint); the tree solve satisfies every row to first order (1e-9 m / rad measured).
// Joints closing a loop (both bodies already in the same tree) are solved one at a time with the same 6-row block
// (LDL^T, 6 x 6) after the trees, and trees containing loops alternate with them LOOP_ROUNDS times (block Gauss-
// Seidel: a rigid triangle of ball joints keeps ~0.05 mm of loop error at h = 1/480; an exact Schur complement on
// the loop rows is the upgrade path). Bounded rows (motors) are solved, clamped to their budget, fixed and moved to
// the right-hand side, and the rest re-solved. Nonlinearity: a pass whose correction rotated a body by more than
// 1e-3 rad is re-linearised and solved again (at most twice more). Trust region: a linearised step that would turn
// some body by more than 0.5 rad (high mass ratios) is scaled down as a whole, multipliers included, and the pass is
// re-linearised up to 8 times. Rank-deficient pivots (redundant rows, e.g. two hinges on one axis) get a zero
// multiplier. Lambdas are accumulated per row slot AND kind (lock / limit / motor; a slot that changes kind within a
// substep restarts at 0), so with several position iterations a limit's share never eats a motor's budget; lambdaPos
// / lambdaRot keep the totals.
// lambda is reset at iteration 0 (one call per substep = the paper's single iteration; pass iteration > 0 for extra
// passes within a substep, which then keep accumulating for the compliance term).
// Breaking: the lock rows of a joint with a finite breakForce / breakTorque are BOUNDED, |lambda| <= breakForce h^2 per
// linear row (breakTorque h^2 per angular row), through the same clamp-and-re-solve path as motor budgets (clampTree,
// solveLocal), so in the substep a joint breaks it transmits at most breakForce h per row (sqrt(3) breakForce h on a
// diagonal) instead of whatever the rigid solve asked for. After its rows, force = |lambdaPos| / h^2 (N), torque =
// |lambdaRot| / h^2 (N m); a joint whose force exceeds breakForce (torque breakTorque) or one of whose rows was held at
// its bound gets broken = 1 and is skipped from then on. Why: without the bound a breaking joint first passed its full
// impulse on, so a struck body shared the hit with everything glued to it (a 30 kg ball at 20 m/s into a 40-piece
// glued wall broke joints across the whole wall and threw nothing; physics/dynamics/fracture.js). Caveats: a joint
// below its threshold is untouched (an unbreakable one, breakForce Infinity, bit for bit); a breakable joint yields at
// breakForce per axis in its breaking substep, and when several rows clamp at once the joint's constraint error is
// left partly uncorrected for that substep (it is broken afterwards anyway); a pass whose step the trust region cut
// (applyTree / solveLocal scale sc < 1) can leave a clamped row below its bound, so that substep yields without breaking
// and the joint breaks in the next one.
//
// Velocity pass (paper 3.6), joint by joint. Damping mu: dv = -v_rel min(mu h, 1) as one impulse along v_rel
// (relative point velocity at anchor A; DISTANCE: its component along d only) and dw = -w_rel min(mu h, 1) along
// w_rel (not for DISTANCE). Velocity motor: impulse driving dphi/dt = w_rel . f0 (slider: ds_1/dt) to the target
// velocity (0 while holding at a limit), clamped to the budget left by the position pass, maxForce h - |lambda_motor|
// / h (the motor kind's own lambda).
//
// Limits / caveats: angles are single-turn (phi in (-pi, pi]); twist is undefined when the axes are opposite (swing
// = pi), then only swing rows act. Sleeping and freed bodies are not checked: the world must wake jointed bodies
// together and remove the joints of a body it frees.
// Determinism: + - * / Math.sqrt, comparisons and detAtan2 (through jointAtan2, its bit-identical flat copy); joints,
// trees and bodies in an order fixed by the ids; module / workspace scratch only. Allocation: none after the first
// pass (the workspace is created then, and again only if the BodyStore is replaced by a larger one; the dense block
// pool grows by doubling if a body ever meets more joint rows than it has room for).
import { BODY, WORLD_BODY } from './state.js';
import { detAtan2 } from '../../core/mathDet.js';
import { rotate, rotateInverse, worldInvInertiaMatrix, displaceBodyFrom } from './xpbd.js';

/** Joint types. */
export const JOINT = Object.freeze({ FIXED: 0, BALL: 1, HINGE: 2, SLIDER: 3, DISTANCE: 4, D6: 5 });

const { FIXED, BALL, HINGE, SLIDER, DISTANCE, D6 } = JOINT;
const DYN = BODY.DYNAMIC;
const PI = 3.14159265358979323846, TWO_PI = 2 * PI;

function movable(B, id) { return id >= 0 && B.type[id] === DYN && B.invMass[id] > 0; }

/** Structure-of-arrays joint storage (see the header for the descriptor and the fields). */
export class JointStore {
  constructor(capacity = 1024) {
    const n = this.capacity = capacity;
    this.count = 0; // one past the highest allocated id
    this.alive = new Uint8Array(n);
    this.type = new Uint8Array(n);
    this.a = new Int32Array(n); this.b = new Int32Array(n);
    this.anchorA = new Float64Array(3 * n); this.anchorB = new Float64Array(3 * n);
    this.axisA = new Float64Array(3 * n); this.axisB = new Float64Array(3 * n);
    this.refA = new Float64Array(3 * n); this.refB = new Float64Array(3 * n);
    this.lo = new Float64Array(n); this.hi = new Float64Array(n);
    this.linLo = new Float64Array(3 * n); this.linHi = new Float64Array(3 * n);
    this.swingLo = new Float64Array(n); this.swingHi = new Float64Array(n);
    this.twistLo = new Float64Array(n); this.twistHi = new Float64Array(n);
    this.motorOn = new Uint8Array(n); this.motorVel = new Float64Array(n); this.motorMax = new Float64Array(n);
    this.motorTarget = new Float64Array(n);
    this.compliance = new Float64Array(n); this.damping = new Float64Array(n);
    this.breakForce = new Float64Array(n); this.breakTorque = new Float64Array(n);
    this.broken = new Uint8Array(n); this.collideConnected = new Uint8Array(n);
    this.lambdaPos = new Float64Array(3 * n); this.lambdaRot = new Float64Array(3 * n);
    this.force = new Float64Array(n); this.torque = new Float64Array(n);
    this._motorActive = new Uint8Array(n); // last position pass: 1 motor row drove, 2 it held at a limit, 0 inactive
    this._rowKind = new Uint8Array(6 * n);  // per row slot (j * 6 + m): the kind its lambda belongs to (K_*), 0 none
    this._rowLam = new Float64Array(6 * n); // per row slot: lambda accumulated by that kind in this substep
    this._ws = null;                        // solver workspace (created by the first solveJointsPositions)
    // set of pairKeys of unbroken joints with collideConnected false (open addressing, load <= 1/2)
    let bits = 3;
    while ((1 << bits) < 2 * n) bits++;
    this._connKeys = new Uint32Array(1 << bits).fill(EMPTY_KEY); this._connShift = 32 - bits;
    this._free = new Int32Array(n); this._freeCount = 0;
  }

  /** Number of live joints. */
  get size() { let s = 0; for (let i = 0; i < this.count; i++) s += this.alive[i]; return s; }

  /** Create a joint from a descriptor (header) at the bodies' current poses; returns its id, or -1 when full. */
  add(desc, B) {
    const type = desc.type;
    if (!(type === FIXED || type === BALL || type === HINGE || type === SLIDER || type === DISTANCE || type === D6)) {
      throw new TypeError(`JointStore.add: unknown joint type ${type}`);
    }
    const a = desc.bodyA == null ? WORLD_BODY : desc.bodyA, b = desc.bodyB == null ? WORLD_BODY : desc.bodyB;
    const ok = (id) => id === WORLD_BODY || (Number.isInteger(id) && id >= 0 && id < B.count && B.alive[id] === 1);
    if (!ok(a) || !ok(b) || a === b) throw new RangeError(`JointStore.add: invalid bodies ${a}, ${b}`);
    const axA = desc.axisA || Y3, axB = desc.axisB;
    const zero = (v) => !(Math.hypot(+v[0], +v[1], +v[2]) > 1e-12);
    if (zero(axA) || (axB && zero(axB))) throw new RangeError('JointStore.add: zero-length axis');
    let id;
    if (this._freeCount > 0) id = this._free[--this._freeCount];
    else if (this.count < this.capacity) id = this.count++;
    else return -1;
    const j3 = id * 3, v = _v;
    this.alive[id] = 1; this.type[id] = type; this.a[id] = a; this.b[id] = b;

    // anchors: anchor B defaults to anchor A's creation position
    const anA = desc.anchorA || ZERO3;
    this.anchorA[j3] = +anA[0]; this.anchorA[j3 + 1] = +anA[1]; this.anchorA[j3 + 2] = +anA[2];
    toWorldPoint(B, a, anA[0], anA[1], anA[2], v, 0);              // pA
    if (desc.anchorB) { this.anchorB[j3] = +desc.anchorB[0]; this.anchorB[j3 + 1] = +desc.anchorB[1]; this.anchorB[j3 + 2] = +desc.anchorB[2]; }
    else toLocalPoint(B, b, v[0], v[1], v[2], this.anchorB, j3);
    // axes: axis B defaults to axis A carried into B's frame
    normInto(+axA[0], +axA[1], +axA[2], this.axisA, j3);
    toWorldDir(B, a, this.axisA[j3], this.axisA[j3 + 1], this.axisA[j3 + 2], v, 0);
    if (axB) normInto(+axB[0], +axB[1], +axB[2], this.axisB, j3);
    else { toLocalDir(B, b, v[0], v[1], v[2], v, 3); normInto(v[3], v[4], v[5], this.axisB, j3); }
    // references: refA given or derived; refB = refA carried into B by the creation pose (angle 0 at creation)
    const rA = desc.refA;
    if (!(rA && orthoInto(+rA[0], +rA[1], +rA[2], this.axisA, j3, this.refA))) perpInto(this.axisA, j3, this.refA);
    toWorldDir(B, a, this.refA[j3], this.refA[j3 + 1], this.refA[j3 + 2], v, 0);
    toLocalDir(B, b, v[0], v[1], v[2], v, 3);
    if (!orthoInto(v[3], v[4], v[5], this.axisB, j3, this.refB)) perpInto(this.axisB, j3, this.refB);

    // limits per type
    const lim = desc.limits;
    this.lo[id] = lim ? +lim[0] : -Infinity; this.hi[id] = lim ? +lim[1] : Infinity;
    for (let k = 0; k < 3; k++) { this.linLo[j3 + k] = 0; this.linHi[j3 + k] = 0; } // locked
    this.swingLo[id] = -Infinity; this.swingHi[id] = 0;                               // locked
    this.twistLo[id] = 0; this.twistHi[id] = 0;                                       // locked
    if (type === BALL || type === D6) {
      parseRange(desc.swing, this.swingLo, this.swingHi, id, true);
      parseRange(desc.twist, this.twistLo, this.twistHi, id, false);
      if (type === D6 && desc.linear) {
        for (let k = 0; k < 3; k++) {
          const m = desc.linear[k];
          if (m === true) { this.linLo[j3 + k] = 0; this.linHi[j3 + k] = 0; }
          else if (Array.isArray(m)) { this.linLo[j3 + k] = +m[0]; this.linHi[j3 + k] = +m[1]; }
          else { this.linLo[j3 + k] = -Infinity; this.linHi[j3 + k] = Infinity; }
        }
      }
    } else if (type === HINGE) {
      this.twistLo[id] = this.lo[id]; this.twistHi[id] = this.hi[id];
    } else if (type === SLIDER) {
      this.linLo[j3 + 1] = this.lo[id]; this.linHi[j3 + 1] = this.hi[id];
    } else if (type === DISTANCE) {
      toWorldPoint(B, a, this.anchorA[j3], this.anchorA[j3 + 1], this.anchorA[j3 + 2], v, 0);
      toWorldPoint(B, b, this.anchorB[j3], this.anchorB[j3 + 1], this.anchorB[j3 + 2], v, 3);
      const dx = v[0] - v[3], dy = v[1] - v[4], dz = v[2] - v[5];
      const d = desc.distance != null ? +desc.distance : Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (!lim) { this.lo[id] = desc.rope ? -Infinity : d; this.hi[id] = d; }
      this.swingLo[id] = -Infinity; this.swingHi[id] = Infinity; this.twistLo[id] = -Infinity; this.twistHi[id] = Infinity;
      for (let k = 0; k < 3; k++) { this.linLo[j3 + k] = -Infinity; this.linHi[j3 + k] = Infinity; }
    }

    const mo = desc.motor;
    this.motorOn[id] = mo ? 1 : 0;
    this.motorVel[id] = mo && mo.velocity != null ? +mo.velocity : 0;
    this.motorMax[id] = mo && mo.maxForce != null ? +mo.maxForce : Infinity;
    this.motorTarget[id] = mo && mo.target != null ? +mo.target : NaN;
    this.compliance[id] = desc.compliance != null ? +desc.compliance : 0;
    this.damping[id] = desc.damping != null ? +desc.damping : 0;
    this.breakForce[id] = desc.breakForce != null ? +desc.breakForce : Infinity;
    this.breakTorque[id] = desc.breakTorque != null ? +desc.breakTorque : Infinity;
    this.broken[id] = 0; this.collideConnected[id] = desc.collideConnected ? 1 : 0;
    this.lambdaPos.fill(0, j3, j3 + 3); this.lambdaRot.fill(0, j3, j3 + 3);
    this.force[id] = 0; this.torque[id] = 0; this._motorActive[id] = 0;
    this._rowKind.fill(0, id * 6, id * 6 + 6); this._rowLam.fill(0, id * 6, id * 6 + 6);
    if (this.collideConnected[id] === 0) this._connInsert(a, b);
    return id;
  }

  /** Release a joint id (reused, lowest first, by a later add). */
  remove(id) {
    if (!(id >= 0 && id < this.count) || !this.alive[id]) return;
    this.alive[id] = 0;
    const f = this._free; let i = this._freeCount++;
    while (i > 0 && f[i - 1] < id) { f[i] = f[i - 1]; i--; } // descending: the smallest id is popped first
    f[i] = id;
    this.rebuildConnected();
  }

  /**
   * True when an alive, unbroken joint with collideConnected false links bodies a and b (either order; false for
   * WORLD_BODY): the contact pair path must skip such pairs. Allocation-free hash lookup.
   */
  connected(a, b) {
    if (a < 0 || b < 0 || a === b) return false;
    const key = a < b ? a * 65536 + b : b * 65536 + a, T = this._connKeys, mask = T.length - 1;
    let i = Math.imul(key | 0, 0x9e3779b1) >>> this._connShift;
    for (;;) {
      const k = T[i];
      if (k === key) return true;
      if (k === EMPTY_KEY) return false;
      i = (i + 1) & mask;
    }
  }

  /** Rebuild the connected() set (automatic on add / remove / breaking; call it after editing collideConnected or
   * broken by hand). */
  rebuildConnected() {
    this._connKeys.fill(EMPTY_KEY);
    for (let j = 0; j < this.count; j++) {
      if (this.alive[j] === 1 && this.broken[j] === 0 && this.collideConnected[j] === 0) this._connInsert(this.a[j], this.b[j]);
    }
  }

  _connInsert(a, b) {
    if (a < 0 || b < 0) return;
    const key = a < b ? a * 65536 + b : b * 65536 + a, T = this._connKeys, mask = T.length - 1;
    let i = Math.imul(key | 0, 0x9e3779b1) >>> this._connShift;
    while (T[i] !== EMPTY_KEY) { if (T[i] === key) return; i = (i + 1) & mask; }
    T[i] = key;
  }
}

const EMPTY_KEY = 0xffffffff; // never a pairKey: a !== b keeps keys below 65535 * 65536 + 65535

function parseRange(r, loArr, hiArr, id, isSwing) {
  if (r == null) { loArr[id] = -Infinity; hiArr[id] = Infinity; }
  else if (typeof r === 'number') { loArr[id] = isSwing ? -Infinity : -r; hiArr[id] = r; }
  else { loArr[id] = +r[0]; hiArr[id] = +r[1]; }
}

// ---- construction helpers (not in the step path) ---------------------------------------------------------------------
const ZERO3 = [0, 0, 0], Y3 = [0, 1, 0];
const _v = new Float64Array(6);

function toWorldPoint(B, id, x, y, z, out, o) {
  if (id < 0) { out[o] = x; out[o + 1] = y; out[o + 2] = z; return; }
  rotate(B.quat, id * 4, x, y, z, out, o);
  out[o] += B.pos[id * 3]; out[o + 1] += B.pos[id * 3 + 1]; out[o + 2] += B.pos[id * 3 + 2];
}
function toLocalPoint(B, id, x, y, z, out, o) {
  if (id < 0) { out[o] = x; out[o + 1] = y; out[o + 2] = z; return; }
  rotateInverse(B.quat, id * 4, x - B.pos[id * 3], y - B.pos[id * 3 + 1], z - B.pos[id * 3 + 2], out, o);
}
function toWorldDir(B, id, x, y, z, out, o) {
  if (id < 0) { out[o] = x; out[o + 1] = y; out[o + 2] = z; } else rotate(B.quat, id * 4, x, y, z, out, o);
}
function toLocalDir(B, id, x, y, z, out, o) {
  if (id < 0) { out[o] = x; out[o + 1] = y; out[o + 2] = z; } else rotateInverse(B.quat, id * 4, x, y, z, out, o);
}
function normInto(x, y, z, out, o) {
  const l = Math.sqrt(x * x + y * y + z * z);
  if (!(l > 1e-12)) return false;
  out[o] = x / l; out[o + 1] = y / l; out[o + 2] = z / l;
  return true;
}
// out[o..] = (v minus its component along unit axis[o..]) normalised; false when v is (nearly) parallel to the axis
function orthoInto(x, y, z, axis, o, out) {
  const ax = axis[o], ay = axis[o + 1], az = axis[o + 2], d = x * ax + y * ay + z * az;
  return normInto(x - d * ax, y - d * ay, z - d * az, out, o);
}
// default reference: local X projected off the axis (local Z when the axis is mostly X)
function perpInto(axis, o, out) {
  if (Math.abs(axis[o]) < 0.9) orthoInto(1, 0, 0, axis, o, out); else orthoInto(0, 0, 1, axis, o, out);
}

// ---- world-space joint frame (step path) ----------------------------------------------------------------------------
// _g layout: pA, pB, levers rA = pA - xA, rB = pB - xB, axis A / B, ref A / B (world), linear basis e0 e1 e2,
// angular basis f0 f1 f2, row levers, spare direction. Step-path helpers take only ints and arrays: V8 boxes a double
// passed to (or returned from) a call it does not inline, which would allocate in every substep.
const PA = 0, PB = 3, RA = 6, RB = 9, XA = 12, XB = 15, FA = 18, FB = 21, E0 = 24, E1 = 27, E2 = 30, F0 = 33, F1 = 36, F2 = 39;
const LA = 42, LB = 45, DV = 48; // levers of the rows being built (a, b) and a spare direction
const _g = new Float64Array(51);
const _s = new Float64Array(3); // swing angle, twist angle, twist valid (1 / 0)

// World point / direction of a local vector of body id (WORLD_BODY: identity). The rotation is written out (no call
// with double arguments, which V8 would box when it does not inline the callee).
function bodyPoint(B, id, prev, local, o, gp, gr) {
  const g = _g, lx = local[o], ly = local[o + 1], lz = local[o + 2];
  if (id < 0) { g[gp] = lx; g[gp + 1] = ly; g[gp + 2] = lz; g[gr] = 0; g[gr + 1] = 0; g[gr + 2] = 0; return; }
  const X = prev ? B.prevPos : B.pos, q = prev ? B.prevQuat : B.quat;
  const k = id * 4, x = q[k], y = q[k + 1], z = q[k + 2], w = q[k + 3];
  const tx = 2 * (y * lz - z * ly), ty = 2 * (z * lx - x * lz), tz = 2 * (x * ly - y * lx);
  const rx = lx + w * tx + (y * tz - z * ty), ry = ly + w * ty + (z * tx - x * tz), rz = lz + w * tz + (x * ty - y * tx);
  g[gr] = rx; g[gr + 1] = ry; g[gr + 2] = rz;
  g[gp] = X[id * 3] + rx; g[gp + 1] = X[id * 3 + 1] + ry; g[gp + 2] = X[id * 3 + 2] + rz;
}
function bodyDir(B, id, prev, local, o, gd) {
  const g = _g, lx = local[o], ly = local[o + 1], lz = local[o + 2];
  if (id < 0) { g[gd] = lx; g[gd + 1] = ly; g[gd + 2] = lz; return; }
  const q = prev ? B.prevQuat : B.quat;
  const k = id * 4, x = q[k], y = q[k + 1], z = q[k + 2], w = q[k + 3];
  const tx = 2 * (y * lz - z * ly), ty = 2 * (z * lx - x * lz), tz = 2 * (x * ly - y * lx);
  g[gd] = lx + w * tx + (y * tz - z * ty); g[gd + 1] = ly + w * ty + (z * tx - x * tz); g[gd + 2] = lz + w * tz + (x * ty - y * tx);
}

// World anchors, levers, axes, references and the linear basis of joint j (prev: start-of-substep poses, prevPos /
// prevQuat of every body type: integrateBody saves them for dynamic bodies, the world for kinematic / static ones).
function frame(J, j, B, prev) {
  const a = J.a[j], b = J.b[j], o = j * 3, g = _g;
  bodyPoint(B, a, prev, J.anchorA, o, PA, RA);
  bodyPoint(B, b, prev, J.anchorB, o, PB, RB);
  bodyDir(B, a, prev, J.axisA, o, XA); bodyDir(B, b, prev, J.axisB, o, XB);
  bodyDir(B, a, prev, J.refA, o, FA); bodyDir(B, b, prev, J.refB, o, FB);
  const ux = g[FB], uy = g[FB + 1], uz = g[FB + 2], vx = g[XB], vy = g[XB + 1], vz = g[XB + 2];
  g[E0] = ux; g[E0 + 1] = uy; g[E0 + 2] = uz;
  g[E1] = vx; g[E1 + 1] = vy; g[E1 + 2] = vz;
  g[E2] = uy * vz - uz * vy; g[E2 + 1] = uz * vx - ux * vz; g[E2 + 2] = ux * vy - uy * vx;
}

// _at[0] = detAtan2(_at[0], _at[1]), bit for bit: a flat copy of core/mathDet.js detAtan2 (same Cephes constants and
// operation order; test/dyn-joints.mjs compares the two on 1e5 inputs) that takes and returns its values through a
// typed array. Reason: V8 boxes a double passed to or returned from a call it does not inline, and detAtan2 ->
// satan -> xatan exceeds the inlining budget of a large caller, so calling it from the step path allocated ~64 B per
// angle. NaN / infinite arguments are left to detAtan2 itself.
const _at = new Float64Array(2);
const AT_PIO2 = 1.57079632679489661923, AT_PIO4 = 0.785398163397448309616, AT_PI = 3.14159265358979323846;
const AT_PI_LO = 1.2246467991473531772e-16, AT_MOREBITS = 6.123233995736765886130e-17, AT_T3P8 = 2.41421356237309504880;
const AT_P0 = -8.750608600031904122785e-1, AT_P1 = -1.615753718733365076637e1, AT_P2 = -7.500855792314704667340e1;
const AT_P3 = -1.228866684490136173410e2, AT_P4 = -6.485021904942025371773e1;
const AT_Q0 = 2.485846490142306297962e1, AT_Q1 = 1.650270098316988542046e2, AT_Q2 = 4.328810604912902668951e2;
const AT_Q3 = 4.853903996359136964868e2, AT_Q4 = 1.945506571482613964425e2;
function atan2s() {
  const y = _at[0], x = _at[1];
  if (!(y - y === 0 && x - x === 0)) { _at[0] = detAtan2(y, x); return; } // NaN or infinite
  if (y === 0) { _at[0] = x > 0 || (x === 0 && 1 / x > 0) ? y : (1 / y > 0 ? AT_PI : -AT_PI); return; }
  if (x === 0) { _at[0] = y > 0 ? AT_PIO2 : -AT_PIO2; return; }
  const ax = x < 0 ? -x : x, ay = y < 0 ? -y : y, t = ay / ax;
  const v = t <= 0.66 ? t : t > AT_T3P8 ? 1 / t : (t - 1) / (t + 1), z = v * v;
  const p = v * (z * ((((AT_P0 * z + AT_P1) * z + AT_P2) * z + AT_P3) * z + AT_P4) / (((((z + AT_Q0) * z + AT_Q1) * z + AT_Q2) * z + AT_Q3) * z + AT_Q4)) + v;
  const q = t <= 0.66 ? p : t > AT_T3P8 ? AT_PIO2 - p + AT_MOREBITS : AT_PIO4 + p + 0.5 * AT_MOREBITS;
  const r = x > 0 ? q : AT_PI - (q - AT_PI_LO);
  _at[0] = y > 0 ? r : -r;
}

/** The joint solver's flat detAtan2 (bit-identical to core/mathDet.js detAtan2; exported for the test). */
export function jointAtan2(y, x) { _at[0] = y; _at[1] = x; atan2s(); return _at[0]; }

// Swing / twist of the frame in _g -> _s, and the angular basis f0 f1 f2.
function angles() {
  const g = _g, s = _s;
  const ax = g[XA], ay = g[XA + 1], az = g[XA + 2], bx = g[XB], by = g[XB + 1], bz = g[XB + 2];
  const cx = by * az - bz * ay, cy = bz * ax - bx * az, cz = bx * ay - by * ax; // aB x aA
  const cl = Math.sqrt(cx * cx + cy * cy + cz * cz);
  _at[0] = cl; _at[1] = bx * ax + by * ay + bz * az; atan2s(); s[0] = _at[0];
  let hx = ax + bx, hy = ay + by, hz = az + bz;
  const hl = Math.sqrt(hx * hx + hy * hy + hz * hz);
  if (hl > 1e-9) { hx /= hl; hy /= hl; hz /= hl; s[2] = 1; } else { hx = ax; hy = ay; hz = az; s[2] = 0; }
  g[F0] = hx; g[F0 + 1] = hy; g[F0 + 2] = hz;
  // references projected off f0
  let d = g[FA] * hx + g[FA + 1] * hy + g[FA + 2] * hz;
  const r1x = g[FA] - d * hx, r1y = g[FA + 1] - d * hy, r1z = g[FA + 2] - d * hz;
  d = g[FB] * hx + g[FB + 1] * hy + g[FB + 2] * hz;
  const r2x = g[FB] - d * hx, r2y = g[FB + 1] - d * hy, r2z = g[FB + 2] - d * hz;
  if (s[2] === 1) {
    const tx = r2y * r1z - r2z * r1y, ty = r2z * r1x - r2x * r1z, tz = r2x * r1y - r2y * r1x;
    _at[0] = hx * tx + hy * ty + hz * tz; _at[1] = r2x * r1x + r2y * r1y + r2z * r1z; atan2s(); s[1] = _at[0];
  } else s[1] = 0;
  // swing axis f1 (any perpendicular when the axes are aligned or opposite)
  let fx, fy, fz;
  if (cl > 1e-12) { fx = cx / cl; fy = cy / cl; fz = cz / cl; }
  else {
    const l = Math.sqrt(r1x * r1x + r1y * r1y + r1z * r1z);
    if (l > 1e-12) { fx = r1x / l; fy = r1y / l; fz = r1z / l; }
    else { fx = g[FA]; fy = g[FA + 1]; fz = g[FA + 2]; }
  }
  g[F1] = fx; g[F1 + 1] = fy; g[F1 + 2] = fz;
  g[F2] = hy * fz - hz * fy; g[F2 + 1] = hz * fx - hx * fz; g[F2 + 2] = hx * fy - hy * fx;
}

// ---- rows of one joint ----------------------------------------------------------------------------------------------
// buildRows fills the 6 candidate rows of joint j for the current poses (linear k = 0..2 along e_k, angular 3..5
// about f0..f2; DISTANCE: row 0 along d): activity, C, alpha~, accumulated lambda, lambda bounds, and the Jacobian
// rows of body a (_JA: linear 3, angular 3) and body b (_JB, sign included).
const _act = new Uint8Array(6), _fix = new Uint8Array(6);
const _c = new Float64Array(6), _al = new Float64Array(6), _lam = new Float64Array(6);
const _lmin = new Float64Array(6), _lmax = new Float64Array(6), _x = new Float64Array(6), _y = new Float64Array(6);
const _JA = new Float64Array(36), _JB = new Float64Array(36), _WA = new Float64Array(36), _WB = new Float64Array(36);
const _K0 = new Float64Array(36), _K = new Float64Array(36), _D = new Float64Array(6), _iD = new Float64Array(6);
const _M = new Float64Array(18), _rhs = new Float64Array(6);
const _out = new Float64Array(2); // solveLocal / applyTree: largest |rotation|^2 applied; 1 when the trust region cut
const _dl = new Float64Array(12);  // solveLocal pose deltas of a and b

// Row kinds: a row slot's lambda is accumulated per kind and restarts at 0 when the slot changes kind (a limit's lambda
// must not count against a motor's budget, nor a motor's against a lock's compliance term).
const K_LOCK = 1, K_LIMIT = 2, K_MOTOR = 3;
// _lam[m] = this substep's lambda of row slot m of joint j for `kind`, and the bounds of that kind: limits one-sided
// (sign = +1: a lower bound may only push up, lambda >= 0; -1: an upper bound, lambda <= 0), locks +-breakForce h^2
// (linear) / +-breakTorque h^2 (angular) and motors +-mmax, both passed through _lmax[m] by the caller (a lock of an
// unbreakable joint gets +-Infinity, i.e. free).
function setLam(J, j, m, kind, sign) {
  const ko = j * 6 + m;
  if (J._rowKind[ko] !== kind) { J._rowKind[ko] = kind; J._rowLam[ko] = 0; }
  _lam[m] = J._rowLam[ko];
  if (kind === K_LIMIT) { _lmin[m] = sign > 0 ? 0 : -Infinity; _lmax[m] = sign > 0 ? Infinity : 0; }
  else { _lmin[m] = -_lmax[m]; }
}

// Jacobian row m along the unit direction _g[e..] at the row levers _g[LA..] (a) / _g[LB..] (b): (e, r_a x e) on a,
// -(e, r_b x e) on b.
function rowLin(m, e) {
  const g = _g, o = m * 6, ex = g[e], ey = g[e + 1], ez = g[e + 2];
  const rax = g[LA], ray = g[LA + 1], raz = g[LA + 2], rbx = g[LB], rby = g[LB + 1], rbz = g[LB + 2];
  _JA[o] = ex; _JA[o + 1] = ey; _JA[o + 2] = ez;
  _JA[o + 3] = ray * ez - raz * ey; _JA[o + 4] = raz * ex - rax * ez; _JA[o + 5] = rax * ey - ray * ex;
  _JB[o] = -ex; _JB[o + 1] = -ey; _JB[o + 2] = -ez;
  _JB[o + 3] = rbz * ey - rby * ez; _JB[o + 4] = rbx * ez - rbz * ex; _JB[o + 5] = rby * ex - rbx * ey;
}
// Angular Jacobian row m about the unit axis _g[f..]: (0, f) on a, (0, -f) on b.
function rowAng(m, f) {
  const g = _g, o = m * 6, fx = g[f], fy = g[f + 1], fz = g[f + 2];
  _JA[o] = 0; _JA[o + 1] = 0; _JA[o + 2] = 0; _JA[o + 3] = fx; _JA[o + 4] = fy; _JA[o + 5] = fz;
  _JB[o] = 0; _JB[o + 1] = 0; _JB[o + 2] = 0; _JB[o + 3] = -fx; _JB[o + 4] = -fy; _JB[o + 5] = -fz;
}
function dot6(a, ao, b, bo) {
  return a[ao] * b[bo] + a[ao + 1] * b[bo + 1] + a[ao + 2] * b[bo + 2] + a[ao + 3] * b[bo + 3] + a[ao + 4] * b[bo + 4] + a[ao + 5] * b[bo + 5];
}

/** Candidate rows of joint j (see the header) into the scratch above; returns the number of active rows. */
function buildRows(J, j, B, h) {
  const type = J.type[j], g = _g, s = _s, o = j * 3, at = J.compliance[j] / (h * h);
  // lock rows of a breakable joint transmit at most its break threshold per row and substep (Infinity: unbounded; a
  // threshold <= 0 means "break on any force": unbounded rows, so the first pass loads the joint and breaks it)
  const fmax = J.breakForce[j] > 0 ? J.breakForce[j] * h * h : Infinity, tmax = J.breakTorque[j] > 0 ? J.breakTorque[j] * h * h : Infinity;
  for (let m = 0; m < 6; m++) _act[m] = 0;
  J._motorActive[j] = 0;
  if (type === DISTANCE) {
    const lo = J.lo[j], hi = J.hi[j];
    bodyPoint(B, J.a[j], false, J.anchorA, o, PA, RA);
    bodyPoint(B, J.b[j], false, J.anchorB, o, PB, RB);
    let dx = g[PA] - g[PB], dy = g[PA + 1] - g[PB + 1], dz = g[PA + 2] - g[PB + 2];
    g[LA] = g[RA]; g[LA + 1] = g[RA + 1]; g[LA + 2] = g[RA + 2]; g[LB] = g[RB]; g[LB + 1] = g[RB + 1]; g[LB + 2] = g[RB + 2];
    if (hi <= 1e-9) {
      // zero length: the anchors coincide, a ball joint (three world-axis lock rows)
      for (let k = 0; k < 3; k++) {
        g[DV] = k === 0 ? 1 : 0; g[DV + 1] = k === 1 ? 1 : 0; g[DV + 2] = k === 2 ? 1 : 0;
        rowLin(k, DV);
        _act[k] = 1; _c[k] = k === 0 ? dx : k === 1 ? dy : dz; _al[k] = at; _lmax[k] = fmax; setLam(J, j, k, K_LOCK, 0);
      }
      return 3;
    }
    let L = Math.sqrt(dx * dx + dy * dy + dz * dz);
    const t = L < lo ? lo : L > hi ? hi : L;
    if (t === L && lo !== hi) return 0; // inside [lo, hi] (a slack rope)
    if (!(L > 1e-12)) {
      // anchors on top of each other but a positive length: push apart along the start-of-substep direction (world
      // up when that is degenerate too)
      bodyPoint(B, J.a[j], true, J.anchorA, o, PA, DV);
      bodyPoint(B, J.b[j], true, J.anchorB, o, PB, DV);
      dx = g[PA] - g[PB]; dy = g[PA + 1] - g[PB + 1]; dz = g[PA + 2] - g[PB + 2];
      const Lp = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (Lp > 1e-12) { dx /= Lp; dy /= Lp; dz /= Lp; } else { dx = 0; dy = 1; dz = 0; }
      L = 1; // (dx, dy, dz) is now the unit direction; the violation is -t
      g[DV] = dx; g[DV + 1] = dy; g[DV + 2] = dz;
      rowLin(0, DV);
      _act[0] = 1; _c[0] = -t; _al[0] = at; _lmax[0] = fmax;
      setLam(J, j, 0, lo === hi ? K_LOCK : K_LIMIT, 1);
      return 1;
    }
    g[DV] = dx / L; g[DV + 1] = dy / L; g[DV + 2] = dz / L;
    rowLin(0, DV);
    _act[0] = 1; _c[0] = L - t; _al[0] = at; _lmax[0] = fmax;
    // equality: a lock; a range or rope: one-sided (too long may only pull, too short only push)
    setLam(J, j, 0, lo === hi ? K_LOCK : K_LIMIT, L < lo ? 1 : -1);
    return 1;
  }
  // motor: which row it drives and its coordinate on the start-of-substep poses
  const motorRow = J.motorOn[j] === 1 && J.motorMax[j] > 0 ? (type === SLIDER ? 1 : type === FIXED ? -1 : 3) : -1;
  const target = J.motorTarget[j], servo = target === target;
  let prevCoord = 0;
  if (motorRow >= 0 && !servo) {
    frame(J, j, B, true);
    if (motorRow === 1) prevCoord = (g[PA] - g[PB]) * g[E1] + (g[PA + 1] - g[PB + 1]) * g[E1 + 1] + (g[PA + 2] - g[PB + 2]) * g[E1 + 2];
    else { angles(); prevCoord = s[1]; }
  }
  frame(J, j, B, false);
  // angles only when an angular row can be active (a free ball joint needs none)
  let tLo, tHi;
  if (type === HINGE) { tLo = J.lo[j]; tHi = J.hi[j]; } else { tLo = J.twistLo[j]; tHi = J.twistHi[j]; }
  const sLo = J.swingLo[j], sHi = J.swingHi[j];
  const needTwist = tLo > -Infinity || tHi < Infinity || motorRow === 3, needSwing = sHi < Infinity || sLo > -Infinity;
  if (needTwist || needSwing) angles();
  const vh = J.motorVel[j] * h, mmax = J.motorMax[j] * h * h;
  const dx = g[PA] - g[PB], dy = g[PA + 1] - g[PB + 1], dz = g[PA + 2] - g[PB + 2];
  // levers: a at anchor A; b at anchor A too (exact Jacobian of rows measured in B's turning frame)
  g[LA] = g[RA]; g[LA + 1] = g[RA + 1]; g[LA + 2] = g[RA + 2];
  g[LB] = g[RB] + dx; g[LB + 1] = g[RB + 1] + dy; g[LB + 2] = g[RB + 2] + dz;
  let n = 0;
  // linear rows in B's frame
  for (let k = 0; k < 3; k++) {
    const e = E0 + k * 3, sk = dx * g[e] + dy * g[e + 1] + dz * g[e + 2];
    let lo, hi;
    if (type === SLIDER && k === 1) { lo = J.lo[j]; hi = J.hi[j]; } else { lo = J.linLo[o + k]; hi = J.linHi[o + k]; }
    let cv, al = 0, motor = 0, kind = K_LOCK, sign = 0;
    if (lo === hi) { cv = sk - lo; al = at; }
    else if (sk < lo) { cv = sk - lo; kind = K_LIMIT; sign = 1; }
    else if (sk > hi) { cv = sk - hi; kind = K_LIMIT; sign = -1; }
    else if (motorRow === k) {
      // the motor's target never leaves [lo, hi]: at a limit it holds (motor = 2) instead of pushing through it
      let tg = servo ? target : prevCoord + vh;
      motor = 1; kind = K_MOTOR;
      if (tg > hi) { tg = hi; motor = 2; } else if (tg < lo) { tg = lo; motor = 2; }
      cv = sk - tg;
    } else continue;
    _act[k] = 1; _c[k] = cv; _al[k] = al; _lmax[k] = kind === K_LOCK ? fmax : mmax;
    setLam(J, j, k, kind, sign);
    if (motor > 0) J._motorActive[j] = motor;
    rowLin(k, e);
    n++;
  }
  // twist row (f0): lock, limit or motor; angle errors of locks and motors wrapped to (-pi, pi]
  if (needTwist && s[2] === 1) {
    const phi = s[1];
    let cv = 0, al = 0, motor = 0, on = true, wrap = true, kind = K_LOCK, sign = 0;
    if (tLo === tHi) { cv = phi - tLo; al = at; }
    else if (phi < tLo) { cv = phi - tLo; wrap = false; kind = K_LIMIT; sign = 1; }
    else if (phi > tHi) { cv = phi - tHi; wrap = false; kind = K_LIMIT; sign = -1; }
    else if (motorRow === 3) {
      let tg = servo ? target : prevCoord + vh;
      if (tg > PI) tg -= TWO_PI; else if (tg <= -PI) tg += TWO_PI;
      motor = 1; kind = K_MOTOR;
      if (tg > tHi) { tg = tHi; motor = 2; } else if (tg < tLo) { tg = tLo; motor = 2; } // hold at a limit
      cv = phi - tg;
    } else on = false;
    if (on) {
      if (wrap) {
        if (cv > PI) cv -= TWO_PI; else if (cv <= -PI) cv += TWO_PI;
        if (cv > PI) cv -= TWO_PI; else if (cv <= -PI) cv += TWO_PI;
      }
      _act[3] = 1; _c[3] = cv; _al[3] = al; _lmax[3] = kind === K_LOCK ? tmax : mmax;
      setLam(J, j, 3, kind, sign);
      if (motor > 0) J._motorActive[j] = motor;
      rowAng(3, F0);
      n++;
    }
  }
  // swing rows (f1, f2): lock both, or limit along f1
  if (!needSwing) return n;
  const swing = s[0];
  if (sHi <= 0) {
    _act[4] = 1; _c[4] = swing; _al[4] = at; _lmax[4] = tmax; setLam(J, j, 4, K_LOCK, 0);
    _act[5] = 1; _c[5] = 0; _al[5] = at; _lmax[5] = tmax; setLam(J, j, 5, K_LOCK, 0);
    rowAng(4, F1); rowAng(5, F2);
    n += 2;
  } else if (swing > sHi || swing < sLo) {
    _act[4] = 1; _c[4] = swing > sHi ? swing - sHi : swing - sLo; _al[4] = 0;
    setLam(J, j, 4, K_LIMIT, swing > sHi ? -1 : 1);
    rowAng(4, F1);
    n++;
  }
  return n;
}

// W J^T of the active scratch rows for body id (linear m^-1 J_lin, angular I_w^-1 J_ang; zero when not movable);
// _M[mo..mo+9] must hold the body's world inverse inertia.
function weightRows(B, id, Jr, Wr, mo) {
  const mv = movable(B, id), im = mv ? B.invMass[id] : 0, M = _M;
  for (let m = 0; m < 6; m++) {
    if (_act[m] === 0) continue;
    const o = m * 6, x = Jr[o + 3], y = Jr[o + 4], z = Jr[o + 5];
    Wr[o] = im * Jr[o]; Wr[o + 1] = im * Jr[o + 1]; Wr[o + 2] = im * Jr[o + 2];
    Wr[o + 3] = M[mo] * x + M[mo + 1] * y + M[mo + 2] * z;
    Wr[o + 4] = M[mo + 3] * x + M[mo + 4] * y + M[mo + 5] * z;
    Wr[o + 5] = M[mo + 6] * x + M[mo + 7] * y + M[mo + 8] * z;
  }
}

// ---- one joint on its own (loop-closing joints) ---------------------------------------------------------------------

// LDL^T solve of the 6 x 6 system _K (lower triangle overwritten) -> _y. Rank-deficient pivots get a zero solution.
function ldlSolve(rhs) {
  const K = _K, D = _D, iD = _iD, y = _y;
  for (let i = 0; i < 6; i++) {
    const ri = i * 6;
    for (let k = 0; k < i; k++) {
      const rk = k * 6;
      let s = K[ri + k];
      for (let m = 0; m < k; m++) s -= K[ri + m] * K[rk + m] * D[m];
      K[ri + k] = s * iD[k];
    }
    const d0 = K[ri + i];
    let d = d0;
    for (let m = 0; m < i; m++) d -= K[ri + m] * K[ri + m] * D[m];
    if (d0 > 0 && d > 1e-12 * d0) { D[i] = d; iD[i] = 1 / d; } else { D[i] = 0; iD[i] = 0; }
  }
  for (let i = 0; i < 6; i++) { let s = rhs[i]; for (let m = 0; m < i; m++) s -= K[i * 6 + m] * y[m]; y[i] = s; }
  for (let i = 0; i < 6; i++) y[i] *= iD[i];
  for (let i = 5; i >= 0; i--) { let s = y[i]; for (let m = i + 1; m < 6; m++) s -= K[m * 6 + i] * y[m]; y[i] = s; }
}

// Block-solve the active scratch rows of joint j alone and apply them; _out[0] = the largest |rotation|^2 applied.
function solveLocal(J, j, B) {
  const a = J.a[j], b = J.b[j];
  worldInvInertiaMatrix(B, a, _M, 0); worldInvInertiaMatrix(B, b, _M, 9);
  weightRows(B, a, _JA, _WA, 0); weightRows(B, b, _JB, _WB, 9);
  const K0 = _K0;
  for (let m = 0; m < 6; m++) {
    for (let k = m; k < 6; k++) {
      let val = 0;
      if (_act[m] === 1 && _act[k] === 1) {
        val = dot6(_JA, m * 6, _WA, k * 6) + dot6(_JB, m * 6, _WB, k * 6);
        if (m === k) val += _al[m];
      } else if (m === k) val = 1;
      K0[m * 6 + k] = val; K0[k * 6 + m] = val;
    }
    _fix[m] = 0; _x[m] = 0;
  }
  // solve; clamp bounded rows to their budget, fix them and re-solve the rest
  for (let pass = 0; pass < 6; pass++) {
    for (let m = 0; m < 6; m++) {
      const free = _act[m] === 1 && _fix[m] === 0;
      for (let k = 0; k < 6; k++) _K[m * 6 + k] = free && _act[k] === 1 && _fix[k] === 0 ? K0[m * 6 + k] : (m === k ? 1 : 0);
      if (free) {
        let r = -_c[m] - _al[m] * _lam[m];
        for (let k = 0; k < 6; k++) if (_fix[k] === 1) r -= K0[m * 6 + k] * _x[k];
        _rhs[m] = r;
      } else _rhs[m] = 0;
    }
    ldlSolve(_rhs);
    let clamped = false;
    for (let m = 0; m < 6; m++) {
      if (_act[m] === 0 || _fix[m] === 1) continue;
      _x[m] = _y[m];
      const t = _lam[m] + _x[m];
      if (t < _lmin[m]) { _x[m] = _lmin[m] - _lam[m]; _fix[m] = 1; clamped = true; }
      else if (t > _lmax[m]) { _x[m] = _lmax[m] - _lam[m]; _fix[m] = 1; clamped = true; }
    }
    if (!clamped) break;
  }
  // pose deltas: sum x_m W J_m^T per body
  _dl.fill(0);
  for (let m = 0; m < 6; m++) {
    const xm = _x[m];
    if (_act[m] === 0 || xm === 0) continue;
    const o = m * 6;
    for (let k = 0; k < 6; k++) { _dl[k] += xm * _WA[o + k]; _dl[6 + k] += xm * _WB[o + k]; }
  }
  const ra = _dl[3] * _dl[3] + _dl[4] * _dl[4] + _dl[5] * _dl[5], rb = _dl[9] * _dl[9] + _dl[10] * _dl[10] + _dl[11] * _dl[11];
  let r2 = ra > rb ? ra : rb;
  _out[1] = 0;
  if (r2 > TRUST2) { // trust region: scale the whole step so no body turns by more than TRUST
    const sc = TRUST / Math.sqrt(r2);
    for (let k = 0; k < 12; k++) _dl[k] *= sc;
    for (let m = 0; m < 6; m++) _x[m] *= sc;
    r2 = TRUST2; _out[1] = 1;
  }
  displaceBodyFrom(B, a, _dl, 0);
  displaceBodyFrom(B, b, _dl, 6);
  const o = j * 3, LP = J.lambdaPos, LR = J.lambdaRot;
  for (let m = 0; m < 3; m++) { LP[o + m] += _x[m]; LR[o + m] += _x[m + 3]; }
  for (let m = 0; m < 6; m++) if (_act[m] === 1) J._rowLam[j * 6 + m] += _x[m];
  _out[0] = r2;
}

// ---- joint trees: exact sparse solve --------------------------------------------------------------------------------
const TREE = 1, LOOP = 2;
const REFINE2 = 1e-6; // re-linearise a pass whose correction rotated a body by more than 1e-3 rad
const TRUST = 0.5, TRUST2 = TRUST * TRUST; // trust region: a pass rotates no body by more than 0.5 rad
const MAX_REPS = 8;   // re-linearisations per pass while the trust region is cutting the step (else 3)
const LOOP_ROUNDS = 4; // tree / loop alternations in a tree that contains loop-closing joints

/** Per-JointStore solver memory (created on the first pass; recreated when the BodyStore is larger). */
class Workspace {
  constructor(jcap, bcap) {
    this.jcap = jcap; this.bcap = bcap; this.pass = 0; this.no = 0;
    const r6 = 6 * jcap, r36 = 36 * jcap, sc = 2 * jcap + 1, vc = 12 * jcap;
    this.kind = new Uint8Array(jcap); this.nrow = new Uint8Array(jcap);
    this.sa = new Int32Array(jcap); this.sb = new Int32Array(jcap);       // slots of bodies a / b (-1: not movable)
    // compressed active rows of each joint (at j * 6 + r)
    this.src = new Uint8Array(r6); this.c = new Float64Array(r6); this.al = new Float64Array(r6); this.lam = new Float64Array(r6);
    this.lo = new Float64Array(r6); this.hi = new Float64Array(r6); this.fix = new Uint8Array(r6);
    this.x = new Float64Array(r6); this.off = new Float64Array(r6);
    this.JA = new Float64Array(r36); this.JB = new Float64Array(r36); this.WA = new Float64Array(r36); this.WB = new Float64Array(r36);
    this.Kacc = new Float64Array(r36); this.racc = new Float64Array(r6); // Schur block / rhs a joint brings from below
    // bodies -> slots
    this.stamp = new Int32Array(bcap); this.slotOf = new Int32Array(bcap); this.uf = new Int32Array(bcap);
    // slots: body, parent joint (-1 root, -2 unvisited), adjacency (CSR), BFS order, dense block per slot
    this.body = new Int32Array(sc); this.pj = new Int32Array(sc); this.deg = new Int32Array(sc);
    this.start = new Int32Array(sc + 1); this.cursor = new Int32Array(sc); this.adj = new Int32Array(2 * jcap);
    this.order = new Int32Array(sc); this.dim = new Int32Array(sc); this.dimC = new Int32Array(sc);
    this.comp = new Int32Array(sc); this.loopy = new Uint8Array(sc); this.only = 0; // tree id per slot; tree has loops
    this.disp = new Float64Array(6 * sc); // pose delta per slot (applyTree)
    this.moff = new Int32Array(sc); this.voff = new Int32Array(sc);
    this.rowJ = new Int32Array(vc); this.rowR = new Uint8Array(vc);
    this.z = new Float64Array(vc); this.iD = new Float64Array(vc); this.d0 = new Float64Array(vc);
    this.mat = new Float64Array(144 * jcap > 1024 ? 144 * jcap : 1024);
  }
}
function workspace(J, B) {
  let ws = J._ws;
  if (ws === null || ws.bcap < B.capacity || ws.jcap !== J.capacity) ws = J._ws = new Workspace(J.capacity, B.capacity);
  return ws;
}
// While ws.only === 1 the tree passes skip every tree without loop-closing joints.
function skipSlot(ws, X) { return ws.only === 1 && ws.loopy[ws.comp[X]] === 0; }
function skipJoint(ws, j) { return ws.only === 1 && ws.loopy[ws.comp[ws.sa[j] >= 0 ? ws.sa[j] : ws.sb[j]]] === 0; }
function find(uf, x) {
  while (uf[x] !== x) { const p = uf[uf[x]]; uf[x] = p; x = p; }
  return x;
}
function touch(ws, id, n, pass) {
  ws.stamp[id] = pass; ws.uf[id] = id; ws.slotOf[id] = n; ws.body[n] = id; ws.deg[n] = 0; ws.pj[n] = -2;
  return n + 1;
}

// Rows of every tree joint -> workspace (compressed to the active rows, with W J^T per body).
function loadTreeRows(J, B, ws, h) {
  for (let j = 0; j < J.count; j++) {
    if (ws.kind[j] !== TREE || skipJoint(ws, j)) continue;
    ws.nrow[j] = 0;
    if (buildRows(J, j, B, h) === 0) continue;
    const a = J.a[j], b = J.b[j];
    worldInvInertiaMatrix(B, a, _M, 0); worldInvInertiaMatrix(B, b, _M, 9);
    weightRows(B, a, _JA, _WA, 0); weightRows(B, b, _JB, _WB, 9);
    let r = 0;
    for (let m = 0; m < 6; m++) {
      if (_act[m] === 0) continue;
      const o = j * 6 + r, o6 = o * 6, m6 = m * 6;
      ws.src[o] = m; ws.c[o] = _c[m]; ws.al[o] = _al[m]; ws.lam[o] = _lam[m]; ws.lo[o] = _lmin[m]; ws.hi[o] = _lmax[m];
      ws.fix[o] = 0; ws.x[o] = 0; ws.off[o] = 0;
      for (let k = 0; k < 6; k++) {
        ws.JA[o6 + k] = _JA[m6 + k]; ws.JB[o6 + k] = _JB[m6 + k]; ws.WA[o6 + k] = _WA[m6 + k]; ws.WB[o6 + k] = _WB[m6 + k];
      }
      r++;
    }
    ws.nrow[j] = r;
  }
}

// Eliminate the rows of slot X's child joints (dense partial LDL^T of K restricted to the joints at X, plus what the
// child joints bring from below); the Schur complement left on X's parent joint goes to that joint's accumulators.
function eliminate(ws, X) {
  const q = ws.pj[X], n = ws.dim[X], A = ws.mat, mo = ws.moff[X], vo = ws.voff[X];
  const rowJ = ws.rowJ, rowR = ws.rowR, z = ws.z;
  let i = 0;
  for (let k = ws.start[X]; k < ws.start[X + 1]; k++) {
    const j = ws.adj[k];
    if (j === q) continue;
    for (let r = 0; r < ws.nrow[j]; r++) if (ws.fix[j * 6 + r] === 0) { rowJ[vo + i] = j; rowR[vo + i] = r; i++; }
  }
  const nc = i;
  ws.dimC[X] = nc;
  if (q >= 0) for (let r = 0; r < ws.nrow[q]; r++) if (ws.fix[q * 6 + r] === 0) { rowJ[vo + i] = q; rowR[vo + i] = r; i++; }
  // K at X (lower triangle) and the right-hand side
  for (let s = 0; s < n; s++) {
    const js = rowJ[vo + s], rs = rowR[vo + s], os = js * 6 + rs, Js = ws.sa[js] === X ? ws.JA : ws.JB, jo = os * 6;
    const row = mo + s * n;
    for (let t = 0; t <= s; t++) {
      const jt = rowJ[vo + t], ot = jt * 6 + rowR[vo + t], Wt = ws.sa[jt] === X ? ws.WA : ws.WB;
      let v = dot6(Js, jo, Wt, ot * 6);
      if (js === jt && js !== q) v += ws.Kacc[js * 36 + rs * 6 + rowR[vo + t]];
      A[row + t] = v;
    }
    if (js !== q) {
      A[row + s] += ws.al[os];
      z[vo + s] = -ws.c[os] - ws.al[os] * ws.lam[os] + ws.racc[os] + ws.off[os];
    } else z[vo + s] = 0;
    ws.d0[vo + s] = A[row + s];
  }
  // partial LDL^T over the nc child rows (right-looking; rank-deficient pivots are dropped)
  for (let p = 0; p < nc; p++) {
    const d = A[mo + p * n + p], d0 = ws.d0[vo + p];
    if (!(d0 > 0 && d > 1e-12 * d0)) {
      ws.iD[vo + p] = 0;
      for (let k = p + 1; k < n; k++) A[mo + k * n + p] = 0;
      continue;
    }
    const id = 1 / d;
    ws.iD[vo + p] = id;
    for (let k = p + 1; k < n; k++) A[mo + k * n + p] *= id;
    const zp = z[vo + p];
    for (let k = p + 1; k < n; k++) {
      const lk = A[mo + k * n + p];
      if (lk === 0) continue;
      const lkd = lk * d, rk = mo + k * n;
      for (let m = p + 1; m <= k; m++) A[rk + m] -= lkd * A[mo + m * n + p];
      z[vo + k] -= lk * zp;
    }
  }
  if (q >= 0) {
    for (let s = nc; s < n; s++) {
      const rs = rowR[vo + s];
      ws.racc[q * 6 + rs] = z[vo + s];
      for (let t = nc; t <= s; t++) {
        const rt = rowR[vo + t], v = A[mo + s * n + t];
        ws.Kacc[q * 36 + rs * 6 + rt] = v; ws.Kacc[q * 36 + rt * 6 + rs] = v;
      }
    }
  }
}

// Back substitution at slot X: its parent joint's solution is known; solve its child joints' rows.
function backSolve(ws, X) {
  const n = ws.dim[X], nc = ws.dimC[X], A = ws.mat, mo = ws.moff[X], vo = ws.voff[X], rowJ = ws.rowJ, rowR = ws.rowR;
  for (let p = nc - 1; p >= 0; p--) {
    let s = ws.z[vo + p] * ws.iD[vo + p];
    for (let k = p + 1; k < n; k++) {
      const l = A[mo + k * n + p];
      if (l !== 0) s -= l * ws.x[rowJ[vo + k] * 6 + rowR[vo + k]];
    }
    ws.x[rowJ[vo + p] * 6 + rowR[vo + p]] = s;
  }
}

function solveTree(J, ws, no) {
  // block sizes and offsets (elimination order = reverse BFS: children first)
  let mo = 0, vo = 0;
  for (let i = no - 1; i >= 0; i--) {
    const X = ws.order[i];
    if (skipSlot(ws, X)) continue;
    let n = 0;
    for (let k = ws.start[X]; k < ws.start[X + 1]; k++) {
      const j = ws.adj[k];
      for (let r = 0; r < ws.nrow[j]; r++) if (ws.fix[j * 6 + r] === 0) n++;
    }
    ws.dim[X] = n; ws.moff[X] = mo; ws.voff[X] = vo; mo += n * n; vo += n;
  }
  if (mo > ws.mat.length) ws.mat = new Float64Array(mo > 2 * ws.mat.length ? mo : 2 * ws.mat.length); // rare growth
  for (let j = 0; j < J.count; j++) {
    if (ws.kind[j] !== TREE || skipJoint(ws, j)) continue;
    ws.Kacc.fill(0, j * 36, j * 36 + 36); ws.racc.fill(0, j * 6, j * 6 + 6);
  }
  for (let i = no - 1; i >= 0; i--) if (!skipSlot(ws, ws.order[i])) eliminate(ws, ws.order[i]);
  for (let i = 0; i < no; i++) if (!skipSlot(ws, ws.order[i])) backSolve(ws, ws.order[i]);
}

// Clamp bounded rows (motors) that exceeded their budget: fix them at the bound and move their effect to the
// right-hand side of every row sharing a body. Returns true when a row was clamped (the tree is then re-solved).
function clampTree(J, ws) {
  let any = false;
  for (let j = 0; j < J.count; j++) {
    if (ws.kind[j] !== TREE || skipJoint(ws, j)) continue;
    for (let r = 0; r < ws.nrow[j]; r++) {
      const o = j * 6 + r;
      if (ws.fix[o] === 1) continue;
      const t = ws.lam[o] + ws.x[o];
      let xf;
      if (t < ws.lo[o]) xf = ws.lo[o] - ws.lam[o]; else if (t > ws.hi[o]) xf = ws.hi[o] - ws.lam[o]; else continue;
      ws.x[o] = xf; ws.fix[o] = 1; any = true;
      for (let side = 0; side < 2; side++) {
        const X = side === 0 ? ws.sa[j] : ws.sb[j];
        if (X < 0) continue;
        const Wr = side === 0 ? ws.WA : ws.WB;
        for (let k = ws.start[X]; k < ws.start[X + 1]; k++) {
          const jj = ws.adj[k], Js = ws.sa[jj] === X ? ws.JA : ws.JB;
          for (let rr = 0; rr < ws.nrow[jj]; rr++) {
            const os = jj * 6 + rr;
            if (ws.fix[os] === 0) ws.off[os] -= dot6(Js, os * 6, Wr, o * 6) * xf;
          }
        }
      }
    }
  }
  return any;
}

// Apply the solution: per body sum x W J^T over its joints' rows; accumulate the lambdas. _out[0] = max |rotation|^2.
function applyTree(J, B, ws, no) {
  let maxRot = 0;
  const D = ws.disp;
  for (let i = 0; i < no; i++) {
    const X = ws.order[i];
    if (skipSlot(ws, X)) continue;
    const d = X * 6;
    let dx = 0, dy = 0, dz = 0, wx = 0, wy = 0, wz = 0;
    for (let k = ws.start[X]; k < ws.start[X + 1]; k++) {
      const j = ws.adj[k], W = ws.sa[j] === X ? ws.WA : ws.WB;
      for (let r = 0; r < ws.nrow[j]; r++) {
        const o = j * 6 + r, xv = ws.x[o];
        if (xv === 0) continue;
        const w = o * 6;
        dx += xv * W[w]; dy += xv * W[w + 1]; dz += xv * W[w + 2]; wx += xv * W[w + 3]; wy += xv * W[w + 4]; wz += xv * W[w + 5];
      }
    }
    D[d] = dx; D[d + 1] = dy; D[d + 2] = dz; D[d + 3] = wx; D[d + 4] = wy; D[d + 5] = wz;
    const r2 = wx * wx + wy * wy + wz * wz;
    if (r2 > maxRot) maxRot = r2;
  }
  // trust region: a linearised step that would turn some body by more than TRUST is scaled down as a whole (pose
  // deltas and multipliers alike); the caller re-linearises and continues from there
  let sc = 1;
  _out[1] = 0;
  if (maxRot > TRUST2) { sc = TRUST / Math.sqrt(maxRot); maxRot = TRUST2; _out[1] = 1; }
  for (let i = 0; i < no; i++) {
    const X = ws.order[i];
    if (skipSlot(ws, X)) continue;
    if (sc !== 1) for (let k = X * 6; k < X * 6 + 6; k++) D[k] *= sc;
    displaceBodyFrom(B, ws.body[X], D, X * 6);
  }
  const LP = J.lambdaPos, LR = J.lambdaRot;
  for (let j = 0; j < J.count; j++) {
    if (ws.kind[j] !== TREE || skipJoint(ws, j)) continue;
    for (let r = 0; r < ws.nrow[j]; r++) {
      const o = j * 6 + r, m = ws.src[o], xv = ws.x[o] * sc;
      if (m < 3) LP[j * 3 + m] += xv; else LR[j * 3 + m - 3] += xv;
      J._rowLam[j * 6 + m] += xv;
    }
  }
  _out[0] = maxRot;
}

/**
 * One XPBD positional pass over all alive, unbroken joints (see the header: joint trees are solved exactly, loop-
 * closing joints one by one afterwards, in ascending id order). `iteration` 0 (default) resets the accumulated
 * lambdas first: call once per substep, or with 0, 1, 2 ... for extra passes within a substep.
 */
export function solveJointsPositions(J, B, h, iteration = 0) {
  const ws = workspace(J, B), LP = J.lambdaPos, LR = J.lambdaRot;
  let pass = ws.pass + 1;
  if (pass > 0x3fffffff) { ws.stamp.fill(0); pass = 1; }
  ws.pass = pass;
  // classify: union-find over the movable bodies; a joint closing a loop is solved on its own
  let ns = 0, nTree = 0, nLoop = 0;
  for (let j = 0; j < J.count; j++) {
    ws.kind[j] = 0;
    if (J.alive[j] === 0 || J.broken[j] === 1) continue;
    const o = j * 3;
    if (iteration === 0) {
      LP[o] = 0; LP[o + 1] = 0; LP[o + 2] = 0; LR[o] = 0; LR[o + 1] = 0; LR[o + 2] = 0;
      for (let k = j * 6; k < j * 6 + 6; k++) { J._rowKind[k] = 0; J._rowLam[k] = 0; }
    }
    const a = J.a[j], b = J.b[j], ma = movable(B, a), mb = movable(B, b);
    if (!ma && !mb) { J.force[j] = 0; J.torque[j] = 0; continue; }
    if (ma && ws.stamp[a] !== pass) ns = touch(ws, a, ns, pass);
    if (mb && ws.stamp[b] !== pass) ns = touch(ws, b, ns, pass);
    if (ma && mb) {
      const ra = find(ws.uf, a), rb = find(ws.uf, b);
      if (ra === rb) { ws.kind[j] = LOOP; nLoop++; continue; }
      ws.uf[ra] = rb;
    }
    ws.kind[j] = TREE; nTree++;
    const sa = ma ? ws.slotOf[a] : -1, sb = mb ? ws.slotOf[b] : -1;
    ws.sa[j] = sa; ws.sb[j] = sb;
    if (sa >= 0) ws.deg[sa]++;
    if (sb >= 0) ws.deg[sb]++;
  }
  if (nTree > 0) {
    // adjacency (CSR, ascending joint id) and a BFS order of every tree from its lowest slot
    ws.start[0] = 0;
    for (let s = 0; s < ns; s++) { ws.start[s + 1] = ws.start[s] + ws.deg[s]; ws.cursor[s] = ws.start[s]; }
    for (let j = 0; j < J.count; j++) {
      if (ws.kind[j] !== TREE) continue;
      if (ws.sa[j] >= 0) ws.adj[ws.cursor[ws.sa[j]]++] = j;
      if (ws.sb[j] >= 0) ws.adj[ws.cursor[ws.sb[j]]++] = j;
    }
    let no = 0, nc = 0;
    for (let s = 0; s < ns; s++) {
      if (ws.pj[s] !== -2 || ws.deg[s] === 0) continue;
      ws.pj[s] = -1; ws.order[no++] = s; ws.loopy[nc] = 0;
      for (let head = no - 1; head < no; head++) {
        const X = ws.order[head];
        ws.comp[X] = nc;
        for (let k = ws.start[X]; k < ws.start[X + 1]; k++) {
          const j = ws.adj[k];
          if (j === ws.pj[X]) continue;
          const C = ws.sa[j] === X ? ws.sb[j] : ws.sa[j];
          if (C >= 0) { ws.pj[C] = j; ws.order[no++] = C; }
        }
      }
      nc++;
    }
    for (let j = 0; nLoop > 0 && j < J.count; j++) {
      if (ws.kind[j] === LOOP) ws.loopy[ws.comp[ws.slotOf[movable(B, J.a[j]) ? J.a[j] : J.b[j]]]] = 1;
    }
    ws.no = no;
  } else ws.no = 0;
  // trees exactly, then the loop-closing joints one by one; with loops the two alternate (block Gauss-Seidel)
  const rounds = nLoop > 0 ? LOOP_ROUNDS : 1;
  for (let round = 0; round < rounds; round++) {
    ws.only = round > 0 ? 1 : 0; // later rounds: only the trees that contain loops
    if (nTree > 0) {
      for (let rep = 0; rep < MAX_REPS; rep++) {
        loadTreeRows(J, B, ws, h);
        for (let k = 0; k < 4; k++) { solveTree(J, ws, ws.no); if (!clampTree(J, ws)) break; }
        applyTree(J, B, ws, ws.no);
        if (_out[1] === 0 && (_out[0] <= REFINE2 || rep >= 2)) break;
      }
    }
    for (let j = 0; nLoop > 0 && j < J.count; j++) {
      if (ws.kind[j] !== LOOP) continue;
      for (let rep = 0; rep < MAX_REPS; rep++) {
        if (buildRows(J, j, B, h) === 0) break;
        solveLocal(J, j, B);
        if (_out[1] === 0 && (_out[0] <= REFINE2 || rep >= 2)) break;
      }
    }
  }
  ws.only = 0;
  const h2 = h * h, ih2 = 1 / h2;
  let broke = false;
  for (let j = 0; j < J.count; j++) {
    if (ws.kind[j] === 0) continue;
    const o = j * 3;
    const f = Math.sqrt(LP[o] * LP[o] + LP[o + 1] * LP[o + 1] + LP[o + 2] * LP[o + 2]) * ih2;
    const t = Math.sqrt(LR[o] * LR[o] + LR[o + 1] * LR[o + 1] + LR[o + 2] * LR[o + 2]) * ih2;
    J.force[j] = f; J.torque[j] = t;
    let brk = f > J.breakForce[j] || t > J.breakTorque[j];
    if (!brk && J.breakForce[j] < Infinity) brk = atBound(LP, o, J.breakForce[j] * h2); // a lock row held at its bound
    if (!brk && J.breakTorque[j] < Infinity) brk = atBound(LR, o, J.breakTorque[j] * h2);
    if (brk) { J.broken[j] = 1; broke = true; }
  }
  if (broke) J.rebuildConnected(); // a broken joint's bodies collide again
}

// true when some row lambda of the joint (L[o..o+2]) reached the bound lim > 0 (rounding slack 1e-9 relative)
function atBound(L, o, lim) {
  if (!(lim > 0)) return false;
  const b = lim * (1 - 1e-9);
  return (L[o] < 0 ? -L[o] : L[o]) >= b || (L[o + 1] < 0 ? -L[o + 1] : L[o + 1]) >= b || (L[o + 2] < 0 ? -L[o + 2] : L[o + 2]) >= b;
}

const _vp = new Float64Array(4); // velocity row: target, keep factor, impulse min, impulse max

// Velocity impulse along scratch row 0 (bodies a, b; _M holds their world inverse inertias) that drives J v to
// _vp[0] + _vp[1] (J v), clamped to [_vp[2], _vp[3]].
function velocityRow(B, a, b) {
  _act[0] = 1; _act[1] = 0; _act[2] = 0; _act[3] = 0; _act[4] = 0; _act[5] = 0;
  weightRows(B, a, _JA, _WA, 0); weightRows(B, b, _JB, _WB, 9);
  const k = dot6(_JA, 0, _WA, 0) + dot6(_JB, 0, _WB, 0);
  if (!(k > 0)) return;
  const v = B.vel, w = B.angVel;
  let jv = 0;
  if (a >= 0) { const p = a * 3; jv += _JA[0] * v[p] + _JA[1] * v[p + 1] + _JA[2] * v[p + 2] + _JA[3] * w[p] + _JA[4] * w[p + 1] + _JA[5] * w[p + 2]; }
  if (b >= 0) { const p = b * 3; jv += _JB[0] * v[p] + _JB[1] * v[p + 1] + _JB[2] * v[p + 2] + _JB[3] * w[p] + _JB[4] * w[p + 1] + _JB[5] * w[p + 2]; }
  let p = (_vp[0] + _vp[1] * jv - jv) / k;
  if (p < _vp[2]) p = _vp[2]; else if (p > _vp[3]) p = _vp[3];
  if (p === 0) return;
  if (movable(B, a)) {
    const q = a * 3;
    v[q] += p * _WA[0]; v[q + 1] += p * _WA[1]; v[q + 2] += p * _WA[2]; w[q] += p * _WA[3]; w[q + 1] += p * _WA[4]; w[q + 2] += p * _WA[5];
  }
  if (movable(B, b)) {
    const q = b * 3;
    v[q] += p * _WB[0]; v[q + 1] += p * _WB[1]; v[q + 2] += p * _WB[2]; w[q] += p * _WB[3]; w[q + 1] += p * _WB[4]; w[q + 2] += p * _WB[5];
  }
}

/** Velocity pass: joint damping and velocity motors (see the header). Call after velocityFromPositions. */
export function solveJointsVelocities(J, B, h) {
  const g = _g, v = B.vel, w = B.angVel;
  for (let j = 0; j < J.count; j++) {
    if (J.alive[j] === 0 || J.broken[j] === 1) continue;
    const a = J.a[j], b = J.b[j];
    if (!movable(B, a) && !movable(B, b)) continue;
    const type = J.type[j], mu = J.damping[j];
    const motor = J._motorActive[j] > 0 && J.motorOn[j] === 1 && !(J.motorTarget[j] === J.motorTarget[j]);
    if (!(mu > 0) && !motor) continue;
    frame(J, j, B, false);
    worldInvInertiaMatrix(B, a, _M, 0); worldInvInertiaMatrix(B, b, _M, 9);
    const dx = g[PA] - g[PB], dy = g[PA + 1] - g[PB + 1], dz = g[PA + 2] - g[PB + 2];
    // row levers: a at anchor A; b at anchor A (DISTANCE: at anchor B)
    g[LA] = g[RA]; g[LA + 1] = g[RA + 1]; g[LA + 2] = g[RA + 2];
    if (type === DISTANCE) { g[LB] = g[RB]; g[LB + 1] = g[RB + 1]; g[LB + 2] = g[RB + 2]; }
    else { g[LB] = g[RB] + dx; g[LB + 1] = g[RB + 1] + dy; g[LB + 2] = g[RB + 2] + dz; }
    if (mu > 0) {
      _vp[0] = 0; _vp[1] = mu * h < 1 ? 1 - mu * h : 0; _vp[2] = -Infinity; _vp[3] = Infinity;
      if (type === DISTANCE) {
        const L = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (L > 1e-12) { g[DV] = dx / L; g[DV + 1] = dy / L; g[DV + 2] = dz / L; rowLin(0, DV); velocityRow(B, a, b); }
      } else {
        // relative velocity of the anchor points, then of the rotations
        let vx = 0, vy = 0, vz = 0;
        if (a >= 0) {
          const p = a * 3, rx = g[LA], ry = g[LA + 1], rz = g[LA + 2];
          vx += v[p] + (w[p + 1] * rz - w[p + 2] * ry); vy += v[p + 1] + (w[p + 2] * rx - w[p] * rz); vz += v[p + 2] + (w[p] * ry - w[p + 1] * rx);
        }
        if (b >= 0) {
          const p = b * 3, rx = g[LB], ry = g[LB + 1], rz = g[LB + 2];
          vx -= v[p] + (w[p + 1] * rz - w[p + 2] * ry); vy -= v[p + 1] + (w[p + 2] * rx - w[p] * rz); vz -= v[p + 2] + (w[p] * ry - w[p + 1] * rx);
        }
        const vn = Math.sqrt(vx * vx + vy * vy + vz * vz);
        if (vn > 0) { g[DV] = vx / vn; g[DV + 1] = vy / vn; g[DV + 2] = vz / vn; rowLin(0, DV); velocityRow(B, a, b); }
        let wx = 0, wy = 0, wz = 0;
        if (a >= 0) { const p = a * 3; wx += w[p]; wy += w[p + 1]; wz += w[p + 2]; }
        if (b >= 0) { const p = b * 3; wx -= w[p]; wy -= w[p + 1]; wz -= w[p + 2]; }
        const wn = Math.sqrt(wx * wx + wy * wy + wz * wz);
        if (wn > 0) { g[DV] = wx / wn; g[DV + 1] = wy / wn; g[DV + 2] = wz / wn; rowAng(0, DV); velocityRow(B, a, b); }
      }
    }
    if (motor) {
      // budget left by the position pass: |lambda_motor| / h of maxForce h already spent
      const ko = j * 6 + (type === SLIDER ? 1 : 3), budget = J.motorMax[j] * h;
      const used = J._rowKind[ko] === K_MOTOR ? J._rowLam[ko] / h : 0;
      if (type === SLIDER) {
        _vp[0] = J._motorActive[j] === 2 ? 0 : J.motorVel[j]; _vp[1] = 0; _vp[2] = -budget - used; _vp[3] = budget - used;
        if (_vp[2] < _vp[3]) { rowLin(0, E1); velocityRow(B, a, b); }
      } else if (type !== FIXED) {
        angles();
        _vp[0] = J._motorActive[j] === 2 ? 0 : J.motorVel[j]; _vp[1] = 0; _vp[2] = -budget - used; _vp[3] = budget - used;
        if (_vp[2] < _vp[3] && _s[2] === 1) { rowAng(0, F0); velocityRow(B, a, b); }
      }
    }
  }
}

/** Twist angle (rad, in (-pi, pi]) of body A relative to body B about the joint axis (the hinge angle); 0 when the
 * axes are opposite. */
export function jointAngle(J, B, id) {
  frame(J, id, B, false);
  angles();
  return _s[1];
}

/** Swing angle (rad, [0, pi]) between the two joint axes. */
export function jointSwing(J, B, id) {
  frame(J, id, B, false);
  angles();
  return _s[0];
}

/** Slider: the signed position of anchor A along B's axis (m); any other type: |pA - pB| (DISTANCE: its length). */
export function jointDistance(J, B, id) {
  frame(J, id, B, false);
  const g = _g, dx = g[PA] - g[PB], dy = g[PA + 1] - g[PB + 1], dz = g[PA + 2] - g[PB + 2];
  if (J.type[id] === SLIDER) return dx * g[E1] + dy * g[E1 + 1] + dz * g[E1 + 2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

