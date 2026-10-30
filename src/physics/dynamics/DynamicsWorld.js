// Unified rigid-body dynamics world (pure JS, deterministic, Node-safe): the hub that owns the shared state
// (state.js BodyStore / ContactBuffer), the broadphase, the narrowphase, the joints and the islands, and steps them
// with substepped XPBD (Mueller, Macklin et al. 2020, "Detailed Rigid Body Simulation with Extended Position Based
// Dynamics"). The static world (level boxes, heightfield) lives in an optional engine PhysicsWorld.
//
//   const physics = new PhysicsWorld();                         // static boxes + ground (groundLevel 0)
//   const world = new DynamicsWorld({ physics, substeps: 4 });
//   const id = world.addBody({ shape: box(0.25, 0.25, 0.25), position: [0, 3, 0] });
//   engine fixed step: world.step(fixedDt);  world.getPose(id, mesh.position, mesh.quaternion);
//
// API
//   new DynamicsWorld({ gravity = [0, -9.81, 0], substeps = 4, maxBodies = 4096, maxContacts = 16384,
//     maxJoints = 1024, sleep = { linear: 0.05, angular: 0.05, time: 0.5 } | false, physics = null, staticMask = -1,
//     margin = 0.01, seed = 1, positionIterations = 1, velocityIterations = 1, broadphaseMargin = 0.1,
//     staticProviders = true, timing = false, maxParticles = 0, particleDamping = 0.01 })
//       physics: the PhysicsWorld holding the static world; null means no static world at all, not even a ground.
//       maxParticles > 0: the world owns `particles` (particles.js ParticleStore) and `constraints` (its
//       ParticleConstraints, the built-in particle solver); 0 (default): both are null and step() is the 0.8.0 one.
//       timing: fill stats.ms / msBroad / msNarrow / msSolve (performance.now(), narrowphase included); off, step()
//       reads no timer.
//   addBody({ type = 'dynamic' | 'static' | 'kinematic' (or the BODY.* number), shape, position = [0, 0, 0],
//     quaternion = [0, 0, 0, 1], mass, density = 1000, friction = 0.6, restitution = 0.1, linearDamping = 0.01,
//     angularDamping = 0.05, layer = 1, mask = -1, kind = 'rigid', userData = null, noGravity = false }) -> id
//       mass defaults to density * shape.volume; static and kinematic bodies get invMass 0 / invInertia 0.
//       position / quaternion: arrays or {x, y, z(, w)} objects. The body origin is the shape origin (shapes.js
//       centres spheres, capsules, boxes and convex hulls on their centre of mass; a compound whose `centroid` is
//       off the origin rotates about its origin). kindOf(id) returns the kind.
//   removeBody(id)                     wakes everything that touched it (its sleeping island, contact and joint
//                                      partners) and removes the joints attached to it. Ids are reused (lowest
//                                      first): a body added later with the same id starts clean (no static-query
//                                      cache, friction anchors or narrowphase warm starts of the old one, the last
//                                      when the narrowphase has evictBody), but an `ended` key reported in the step
//                                      after a removeBody + addBody names the id, i.e. reads as the new body's
//   setPose(id, x, y, z, qx, qy, qz, qw) | setPose(id, pos, quat?)       teleport (wakes); without a quaternion the
//                                      orientation is kept
//   getPose(id, outPos, outQuat)       out: {x, y, z(, w)} objects (THREE.Vector3 / Quaternion) or arrays
//   setVelocity(id, vx, vy, vz[, wx, wy, wz]) | setVelocity(id, v, w?)    (wakes; without w the angular velocity
//                                      is kept; on a kinematic body: moves it at that velocity every step until changed)
//   getVelocity(id, outV, outW)
//   applyImpulse(id, px, py, pz, ix, iy, iz) | applyImpulse(id, impulse, point?)   world point and impulse (wakes)
//   applyForce(id, fx, fy, fz[, px, py, pz]) | applyForce(id, force, point?)       accumulated until the end of
//   applyTorque(id, tx, ty, tz) | applyTorque(id, torque)                          the next step (wakes)
//   setKinematicTarget(id, x, y, z, qx, qy, qz, qw) | (id, pos, quat)   the kinematic body moves there over the next
//                                      step with the implied velocity (so it pushes dynamic bodies) and ends exactly on
//                                      the target; without a new target it stops at the start of the following step
//   setGravity(x, y, z) | setGravity(vec)   (wakes every dynamic body)
//   addJoint(desc) -> id, or -1 when the JointStore is full (JointStore.add; wakes both bodies); removeJoint(id);
//                                      joints (the JointStore). Jointed pairs do not collide unless the joint has
//                                      collideConnected (JointStore.connected(a, b), when the JointStore has it)
//   wake(id) (wakes its whole sleeping island); sleep(id) (that body alone); isAwake(id)
//   recollidePairs(ids, count) -> contacts added   (from a solver / plugin beginStep only) collide the pairs among
//                                      ids[0 .. count) that the joint filter skipped this step and are no longer
//                                      connected (glue broken in beginStep), and set their contacts up (see the method)
//   isActive(id)                       true when the body is integrated in the current / was in the last step (awake
//                                      dynamic or moving kinematic: the active set of stage 1, plus the bodies woken by
//                                      contacts in stage 2); particle solvers ask it instead of reading internals
//   step(dt)                           fixed dt (the Engine's fixedDt), substeps inside; no allocation once warm
//   raycast(origin, dir, maxDist = 1e4, opts = {}) -> { distance, point, normal, body, box, kind, ground } | null
//       bodies (opts.bodies !== false; queries.raycastBodies: opts.mask vs body layers, opts.filter(id)) and, when
//       opts.statics !== false and `physics` is set, physics.raycast (opts.staticMask, opts.staticFilter(box),
//       opts.ground, opts.dynamics: the PhysicsWorld's dynamic providers, opts.hfStep; records of dynamics bodies are
//       always filtered out); the nearer wins (body = -1 for a static hit). opts.out reuses a hit object
//   shapeCast(shape, pos, quat, dir, maxDist = 1e4, opts = {}) -> hit | null   (queries.shapeCastBodies; bodies only)
//   overlap(minX, minY, minZ, maxX, maxY, maxZ, outInt32, mask = -1) -> count;  pointInBody(x, y, z) -> id | -1
//       The queries see bodies added, removed or teleported since the last step: the broadphase is brought up to
//       date first when anything changed (pairs are a pure function of the BodyStore, so this changes no result).
//   queryColliders(x, z, r, out)       APPENDS pooled read-only records { minX, minY, maxX, maxY, minZ, maxZ, kind,
//                                      id, vx, vz, [DYN_RECORD]: true } (bodies.aabb, i.e. margin included; kind =
//                                      the body's kind, 'rigid' by default) of the alive non-static bodies within r
//                                      of (x, z) in XZ; `colliderProvider` is the same as a stable bound function
//                                      (marked fn[DYN_RECORD] = true). Attach with it OR with bridge.js, not both
//   getGroundHeight(x, z, y), surfaceAt(x, z, out)   passthrough to physics (0 / { mu: 1, crr: 1, tag: null } without)
//   debugLines(out, max, opts?, colors?) -> count   (debug.js writeDebugLines)
//   serialize() -> { version, desc, time, stepCount, bodies, shapes, joints, narrowphase, friction, events
//                    (, particles, solvers: [{ name, data }] when the world has particles) }
//                                      typed-array copies (structured-clone friendly, e.g. postMessage, so userData
//                                      must be structured-cloneable; a JSON round trip is accepted but loses -0, NaN
//                                      and Infinity). DynamicsWorld.from(s, { physics, solvers }) rebuilds a world that
//                                      continues bit-identically (shapes re-created from descriptors: describeShape /
//                                      shapeFromDescriptor). The current began / ended lists are not carried (the
//                                      touching-pair set is, so the next step's events are right)
//   dispose()
//   Fields: bodies, contacts (last step's contacts; lambdaN / lambdaT = normal / friction impulse over the step, N s),
//   broadphase, narrowphase, joints, islands, physics, gravity (Float64Array(3)), substeps, positionIterations,
//   velocityIterations, margin, staticMask, time, stepCount, stats (reused: bodies, awake, pairs, contacts, joints,
//   islands, ms, msBroad, msNarrow, msSolve, particles, constraints), began / ended ({ count, keys: Uint32Array }
//   pair keys, see Events); particles, constraints, particleSolvers, particleDamping (see Particles).
//
// Particles (0.9.0; particles.js). With maxParticles > 0:
//   particles: ParticleStore (SoA pos / prevPos / vel / invMass / radius / mass / flags / group, PFLAG.PINNED /
//     NO_GRAVITY); constraints: ParticleConstraints (distance / bending / attachment / tether / body tether), the
//     built-in solver; Cloth.grid / Rope.chain build on both.
//   addParticleSolver(solver) -> solver; removeParticleSolver(solver) -> bool; particleSolvers: the registered solvers
//     in run order (insertion order; the built-in constraints are always index 0 and cannot be removed). A solver is
//     { name: string (required: serialize / from match by it), beginStep(world, dt), solvePositions(world, h, it),
//     solveVelocities(world, h, it), endStep(world, dt), serialize() -> plain | null, restore(data) }, every method
//     but name optional, called with positional arguments (dt and h are the step's boxed numbers: read them, nothing is
//     allocated per call; `it` is the 0-based position / velocity iteration index within the substep, so a solver that
//     must act once per substep acts at it === 0; with velocityIterations 0 solveVelocities is never called). A world
//     without particles (maxParticles 0) accepts solvers too, as plugins (e.g. coupling.js Buoyancy on rigid bodies
//     alone): world.particles / world.constraints stay null and a plugin must tolerate that; the hooks run whenever
//     the world has particles or at least one solver.
//   particleAabb(group, out6) -> bool   AABB (minX minY minZ maxX maxY maxZ, radius included) of the alive particles
//                                      of `group`; false (out6 untouched) when there are none
//   The particles' place in step() (h = dt / substeps; `solvers` = particleSolvers in order; the solver calls happen
//   when the world has particles or solvers, the particle integrate / velocity / damping only with particles):
//     after stage 4 (contact setup): solvers.beginStep(world, dt)
//     each substep: rigid integrate loop; particles integrate (prevPos = pos for every alive particle; alive and not
//       PINNED: vel += h g unless NO_GRAVITY, pos += h vel); per position iteration it: contacts, joints,
//       solvers.solvePositions(world, h, it) (it 0 resets lambdas, as the joints do); rigid velocityFromPositions;
//       particles vel = (pos - prevPos) / h (alive, not pinned); per velocity iteration it: contacts, joints,
//       solvers.solveVelocities(world, h, it); rigid damping, then particle damping vel *= max(0, 1 - particleDamping h)
//     after stage 7: solvers.endStep(world, dt); stats.particles (alive) / stats.constraints (alive, built-in)
//   serialize() adds `particles` (ParticleStore.serialize) and desc.maxParticles / particleDamping (with particles) and
//   `solvers` ([{ name, data: serialize() or null }] in run order; with particles or plugins); from(s, { physics,
//   solvers }) rebuilds the store and the built-in
//   constraints bit-identically, then, in the snapshot's order, registers each instance of `solvers` whose `name`
//   matches an entry and calls its restore(data); instances matching no entry are registered after them, entries with
//   no instance are dropped, an instance listed twice (or the built-in) is registered once. A world with neither
//   particles nor plugins steps and serializes exactly as 0.8.0 (no new keys).
//   removeBody removes the attachments to the body. Particles never sleep; a body held by an attachment sleeps once it
//   and the attached particle rest, and wakes when that particle moves or the attachment's force changes (particles.js).
//
// Step pipeline (step(dt), h = dt / substeps)
//   1. kinematic targets: vel = (target - pos) / dt, angVel = axis * angle(target * conj(q)) / dt (detAtan2); a
//      target-driven body without a new target stops. Active set: awake dynamic and moving kinematic bodies; their
//      AABBs are refreshed (shape.aabb(pose, margin)).
//   2. broadphase.update; narrowphase.beginStep; collide every pair (ascending key) with an active body. A contact
//      that reaches a sleeping body wakes its island; the broadphase is then updated again and the pairs that just
//      became active are collided (until nothing wakes, at most MAX_WAKE_PASSES = 16 passes; an island woken in the
//      last pass integrates once without its contacts), so a woken stack has its contacts this very step.
//   3. static world, per active dynamic body (ascending id): the PhysicsWorld grid boxes near it (queryStatic within
//      boundingRadius + margin + 1 m slack, cached per body while it stays within the slack; ascending grid index),
//      the records of the PhysicsWorld's other dynamic providers (traffic, ...; providers marked fn[DYN_RECORD] or
//      named 'dynamicsProvider' (bridge.js) and records marked rec[DYN_RECORD] are dynamics bodies and skipped; the
//      staticIndex of a provider record is 0x40000000 + its list index, a stable feature key only while the providers
//      return their records in the same order), culled by AABB and by body mask & staticMask & layerOf(kind) -> collideStaticBox; then the ground
//      when LAYER.GROUND passes the masks: collideHeightfield with physics.groundHeight, or, for a PhysicsWorld
//      without a heightfield, a flat slab (top = groundLevel) through collideStaticBox (exact box paths). Each
//      narrowphase call's contacts form a manifold (split into chunks of at most MAX_BLOCK = 8 points).
//   4. contact setup: the contact point is split into the two surface points pA = p + n depth / 2, pB = p - n depth / 2
//      (n from a to b), stored as body-frame anchors (b = WORLD_BODY: world point), so the separation is re-evaluated
//      from the current poses in every substep (0.9.0: the anchor of a body whose top-level shape is a sphere or a
//      capsule is re-seated on its surface along n before every use, since a rolling contact is not a material point;
//      compounds keep material anchors; see _reseat): d = (pA - pB) . n > 0 is penetration; speculative contacts (depth < 0)
//      act only once the predicted motion closes them. Friction mu = sqrt(mu_a mu_b) (static world: mu_a),
//      restitution max(e_a, e_b) (static world: e_a). Static-friction anchors come from the anchor cache (below).
//   5. substeps: pre-solve normal velocities vn~ (relative anchor velocity along n, > 0 approaching); integrate
//      (xpbd.integrateBody, gravity unless FLAG.NO_GRAVITY; kinematic bodies move with their velocity and land exactly
//      on their target in the last substep); positionIterations x { contacts, forward then backward over the
//      manifolds; solveJointsPositions }; velocityFromPositions; velocityIterations x { contacts, forward then
//      backward; solveJointsVelocities }; damping (v *= max(0, 1 - linearDamping h), w likewise).
//      Position pass of a manifold, one coupled XPBD block (Mueller 2020 sec. 3.5 rows, compliance 0):
//        normal rows    u = -n at each penetrating point, rhs = d, dLambda_n >= 0 (active set: a pulling point leaves);
//        static friction at the patch centre: 2 tangential rows undoing the mean offset of the static-friction
//                       anchors + 1 twist row about n (patches of 2+ points), kept while |lambda_t| <= mu lambda_n
//                       (twist: <= mu lambda_n r_patch), otherwise the block is re-solved with the normal rows alone;
//        K = J M^-1 J^T over the rows, LDL^T with dependent rows dropped (4 corners of a face have rank 3), applied as
//        one displacement of each body.
//      Velocity pass of a manifold: Coulomb friction once for the patch (an impulse at the patch centre against the
//      tangential slip, min(|vt| / w_t, mu sum(lambda_n) / h - used), and a torsional one against the relative spin
//      about n, within mu sum(lambda_n) r_patch / h), then the normal rows as one block: the points that pushed in
//      this substep as equalities with the target normal velocity -e vn~ after an impact (vn~ > 2 |g| h, sec. 3.6)
//      else 0 (as the paper's update), the other points as push-only inequalities vn <= gap / h, so the block also
//      takes the friction's tilt torque (a box sliding on a slope does not spin).
//      Why blocks and symmetric sweeps: one Gauss-Seidel sweep over the 4 corners of a box face leaves a spurious
//      rotation (each corner's correction tilts the box onto the others; a flat landing hops and spins), and a sweep in
//      one direction only lets a tall stack with friction grow a tilt mode from round-off (10 boxes topple within 3 s);
//      the symmetric sweep (symmetric Gauss-Seidel) keeps it at rest.
//   6. contacts.lambdaN / lambdaT <- the step's impulses; narrowphase.endStep(contacts); the anchors of the manifolds
//      whose static friction held are stored in the anchor cache.
//   7. islands (islands.js) + sleeping; forces and torques cleared; AABBs of the moved bodies refreshed (so the bridge
//      and the queries see the new poses); began / ended events; stats.
//
// Static-friction anchors: while a manifold sticks, its points keep the body-frame anchors they had when it started
// sticking (AnchorCache, keyed by the narrowphase's (pairKey, feature key)), so static friction also pulls back the
// drift of earlier steps instead of only stopping the current substep's motion: no creep on slopes, and a pile of
// 1000 boxes comes to rest and sleeps instead of crawling.
//
// Sleeping: a body sleeps with its island (islands.js); it wakes when a contact links it to an awake body or a moving
// kinematic one (speculative contacts count: islands wake within `margin` of an approaching body), on applyImpulse / applyForce / applyTorque / setPose / setVelocity / addJoint / wake, or when a body
// touching it is removed (or a static / kinematic one is moved by setPose). Sleeping bodies are neither integrated
// nor collided (pairs of sleeping and non-moving bodies are skipped), so a settled pile costs only its broadphase.
//
// Events: `began` / `ended` list the pair keys that started / stopped touching this step: pairKey(a, b) for two
// bodies, (pairKey(a, WORLD_BODY) >>> 0) = 0xFFFF0000 + a for a body against the static world (all static boxes and the
// ground count as one partner). Pairs are kept in typed-array open-addressing sets (no Set / Map); a pair whose
// bodies are all asleep or non-moving keeps "touching" while they sleep (sleeping bodies produce no contacts).
//
// Conventions: three.js Y-up, SI units; quaternions x y z w; angular velocity in world space. Determinism: the step
// uses only + - * / Math.sqrt, comparisons and detAtan2 (core/mathDet.js), iterates ascending body / pair / manifold /
// contact / joint indices (plus the backward sweeps) and never iterates a Set or Map (the static-box index Map is only
// looked up). No randomness is needed (`seed` is kept in the serialized description). Allocation: step() creates no
// objects once warm (typed arrays, pooled records); the solver passes only integers between its functions (vectors
// live in typed arrays: V8 would box a double crossing a non-inlined call), and h / g / static box bounds go to the
// other modules as numbers boxed once. Growth on overflow (documented): the event sets and key lists, the candidate
// index buffer, the static-box tables when boxes are added, the joint-skipped pair list (sized for maxJoints pairs: a
// pair is skipped at most once per step, so it only grows past more connected pairs than joints). What remains per step is V8 boxing inside other modules'
// call interfaces (e.g. ContactBuffer.add from the narrowphase), short-lived and never retained. Timings in `stats`
// use performance.now() and never feed the simulation.
import { Vector3 } from 'three';
import { detAtan2 } from '../../core/mathDet.js';
import { LAYER, layerOf } from '../PhysicsWorld.js';
import { BodyStore, ContactBuffer, BODY, FLAG, SHAPE, WORLD_BODY, DYN_RECORD, pairKey, pairA, pairB } from './state.js';
import { sphere, capsule, box, convex, trimesh, compound } from './shapes.js';
import { Broadphase } from './broadphase.js';
import { Narrowphase } from './narrowphase.js';
import { integrateBody, velocityFromPositions, worldInvInertiaApply, isMovable } from './xpbd.js';
import { JointStore, solveJointsPositions, solveJointsVelocities, JOINT } from './joints.js';
import { Islands } from './islands.js';
import { raycastBodies, shapeCastBodies, overlapBodies, pointInBody } from './queries.js';
import { writeDebugLines } from './debug.js';
import { detachFromPhysicsWorld } from './bridge.js';
import { ParticleStore, ParticleConstraints, PFLAG } from './particles.js';

const STATIC = BODY.STATIC, KINEMATIC = BODY.KINEMATIC, DYNAMIC = BODY.DYNAMIC;
const ASLEEP = FLAG.ASLEEP, NO_GRAVITY = FLAG.NO_GRAVITY;
const P_PINNED = PFLAG.PINNED, P_NO_GRAVITY = PFLAG.NO_GRAVITY;
const TYPE_OF = Object.freeze({ static: STATIC, kinematic: KINEMATIC, dynamic: DYNAMIC });
const DEFAULT_SLEEP = Object.freeze({ linear: 0.05, angular: 0.05, time: 0.5 });
const EMPTY_OPTS = Object.freeze({});
const MAX_WAKE_PASSES = 16;
const SERIAL_VERSION = 1;

/** BodyStore arrays copied by serialize() (stride = length / capacity). */
const BODY_FIELDS = Object.freeze([
  'alive', 'type', 'flags', 'pos', 'quat', 'prevPos', 'prevQuat', 'vel', 'angVel', 'force', 'torque', 'invMass',
  'invInertia', 'mass', 'friction', 'restitution', 'linearDamping', 'angularDamping', 'layer', 'mask', 'aabb',
  'boundingRadius', 'sleepTime', 'island', 'kinTarget',
]);

const hasPerf = typeof performance !== 'undefined' && typeof performance.now === 'function';

// ---- module scratch (never escapes; the step is not re-entrant) --------------------------------------------------
const _ab = new Float64Array(6);   // shape AABB output
const _m9 = new Float64Array(9);   // inertia tensor
const _r = new Float64Array(3);    // rotated vector
const _va = new Float64Array(3);   // relative point velocity
const _w3 = new Float64Array(3);   // world inverse inertia * vector
const _sn = new Float64Array(3), _snb = new Float64Array(3);   // contact normal, world / body frame (_reseat)
const _ro = new Vector3(), _rd = new Vector3();
// manifold block solve scratch: up to MAX_BLOCK contact points + 3 friction rows (2 tangential + twist)
const MAX_BLOCK = 8, MAX_ROWS = MAX_BLOCK + 3;
const _bra = new Float64Array(3 * MAX_BLOCK), _brb = new Float64Array(3 * MAX_BLOCK); // world levers per point
const _bdp = new Float64Array(3 * MAX_BLOCK);                                         // static-friction anchor offsets
const _RU = new Float64Array(3 * MAX_ROWS);                                           // row: linear direction (on a)
const _RLa = new Float64Array(3 * MAX_ROWS), _RLb = new Float64Array(3 * MAX_ROWS);   // row: levers on a / b
const _RCa = new Float64Array(3 * MAX_ROWS), _RCb = new Float64Array(3 * MAX_ROWS);   // row: angular part (r x u, or n)
const _RTa = new Float64Array(3 * MAX_ROWS), _RTb = new Float64Array(3 * MAX_ROWS);   // I^-1 of the angular parts
const _Rrhs = new Float64Array(MAX_ROWS);
const _bK = new Float64Array(MAX_ROWS * MAX_ROWS);  // K over all rows of the manifold (row-major, stride MAX_ROWS)
const _bA = new Float64Array(MAX_ROWS * MAX_ROWS);  // active sub-block / its LDL^T factor
const _bD = new Float64Array(MAX_ROWS);
const _bd = new Float64Array(MAX_BLOCK);
const _brhs = new Float64Array(MAX_ROWS), _bx = new Float64Array(MAX_ROWS);
const _bidx = new Int32Array(MAX_ROWS);
const _bact = new Uint8Array(MAX_BLOCK);
const PIVOT_TOL = 1e-9;          // relative pivot below which a row is linearly dependent (dropped: x = 0)
const STATIC_SLACK = 1;          // m: extra radius of the static grid query (an empty answer is reused within it)
const GROUND_EXTENT = 1e5;       // m: half extent of the flat-ground slab
const GROUND_INDEX = 0x3fffffff; // staticIndex of the flat-ground slab
const SQ_CAP = 8;                // static boxes remembered per body by the query cache

// Solve K_s x = rhs for the m rows _bidx[0 .. m) of _bK (symmetric positive semi-definite) by LDL^T without
// reordering; a pivot <= PIVOT_TOL * max diagonal marks a dependent row, whose unknown is fixed at 0. -> _bx.
function _solveBlock(m) {
  const A = _bA, D = _bD, S = MAX_ROWS;
  let dmax = 0;
  for (let r = 0; r < m; r++) {
    const kr = _bidx[r] * S;
    for (let c = 0; c < m; c++) A[r * S + c] = _bK[kr + _bidx[c]];
    if (A[r * S + r] > dmax) dmax = A[r * S + r];
  }
  const tol = dmax * PIVOT_TOL;
  for (let j = 0; j < m; j++) {
    let dj = A[j * S + j];
    for (let k = 0; k < j; k++) { const l = A[j * S + k]; dj -= l * l * D[k]; }
    if (!(dj > tol)) {
      D[j] = 0;
      for (let i = j + 1; i < m; i++) A[i * S + j] = 0;
      continue;
    }
    D[j] = dj;
    for (let i = j + 1; i < m; i++) {
      let v = A[i * S + j];
      for (let k = 0; k < j; k++) v -= A[i * S + k] * A[j * S + k] * D[k];
      A[i * S + j] = v / dj;
    }
  }
  for (let i = 0; i < m; i++) {
    let v = _brhs[i];
    for (let k = 0; k < i; k++) v -= A[i * S + k] * _bx[k];
    _bx[i] = v;
  }
  for (let i = 0; i < m; i++) _bx[i] = D[i] > 0 ? _bx[i] / D[i] : 0;
  for (let i = m - 1; i >= 0; i--) {
    if (D[i] === 0) { _bx[i] = 0; continue; }
    let v = _bx[i];
    for (let k = i + 1; k < m; k++) v -= A[k * S + i] * _bx[k];
    _bx[i] = v;
  }
}

// Vector helpers take and return their vectors through typed arrays (offsets), never as double arguments or return
// values: V8 boxes a double that crosses a non-inlined call, and the solver runs these per contact per substep.
// out[oo..] = q V[vo..], q at Q[qo .. qo + 3]
function rotQ(Q, qo, V, vo, out, oo) {
  const qx = Q[qo], qy = Q[qo + 1], qz = Q[qo + 2], qw = Q[qo + 3], x = V[vo], y = V[vo + 1], z = V[vo + 2];
  const tx = 2 * (qy * z - qz * y), ty = 2 * (qz * x - qx * z), tz = 2 * (qx * y - qy * x);
  out[oo] = x + qw * tx + (qy * tz - qz * ty);
  out[oo + 1] = y + qw * ty + (qz * tx - qx * tz);
  out[oo + 2] = z + qw * tz + (qx * ty - qy * tx);
}
// out[oo..] = conj(q) V[vo..]
function invRotQ(Q, qo, V, vo, out, oo) {
  const qx = -Q[qo], qy = -Q[qo + 1], qz = -Q[qo + 2], qw = Q[qo + 3], x = V[vo], y = V[vo + 1], z = V[vo + 2];
  const tx = 2 * (qy * z - qz * y), ty = 2 * (qz * x - qx * z), tz = 2 * (qx * y - qy * x);
  out[oo] = x + qw * tx + (qy * tz - qz * ty);
  out[oo + 1] = y + qw * ty + (qz * tx - qx * tz);
  out[oo + 2] = z + qw * tz + (qx * ty - qy * tx);
}
// D[d0..] = I_w^-1 S[s0..] = R I_b^-1 R^T S of body id (zero unless it is a movable dynamic body; xpbd's
// worldInvInertiaApply with array arguments)
function invI(B, id, S, s0, D, d0) {
  if (!(id >= 0 && B.type[id] === DYNAMIC && B.invMass[id] > 0)) { D[d0] = 0; D[d0 + 1] = 0; D[d0 + 2] = 0; return; }
  const vx = S[s0], vy = S[s0 + 1], vz = S[s0 + 2];
  const I = B.invInertia, k = id * 9;
  const i0 = I[k], i1 = I[k + 1], i2 = I[k + 2], i3 = I[k + 3], i4 = I[k + 4], i5 = I[k + 5], i6 = I[k + 6], i7 = I[k + 7], i8 = I[k + 8];
  if (i1 === 0 && i2 === 0 && i3 === 0 && i5 === 0 && i6 === 0 && i7 === 0 && i0 === i4 && i0 === i8) {
    D[d0] = i0 * vx; D[d0 + 1] = i0 * vy; D[d0 + 2] = i0 * vz; return;
  }
  const q = B.quat, o = id * 4, x = q[o], y = q[o + 1], z = q[o + 2], w = q[o + 3];
  let tx = 2 * (z * vy - y * vz), ty = 2 * (x * vz - z * vx), tz = 2 * (y * vx - x * vy);
  const ux = vx + w * tx + (z * ty - y * tz), uy = vy + w * ty + (x * tz - z * tx), uz = vz + w * tz + (y * tx - x * ty);
  const mx = i0 * ux + i1 * uy + i2 * uz, my = i3 * ux + i4 * uy + i5 * uz, mz = i6 * ux + i7 * uy + i8 * uz;
  tx = 2 * (y * mz - z * my); ty = 2 * (z * mx - x * mz); tz = 2 * (x * my - y * mx);
  D[d0] = mx + w * tx + (y * tz - z * ty);
  D[d0 + 1] = my + w * ty + (z * tx - x * tz);
  D[d0 + 2] = mz + w * tz + (x * ty - y * tx);
}
// q += 1/2 [sign * R[r0..], 0] (x) q, normalised (xpbd's rotateBody)
function rotateBy(Q, o, R, r0, sign) {
  const hx = 0.5 * sign * R[r0], hy = 0.5 * sign * R[r0 + 1], hz = 0.5 * sign * R[r0 + 2];
  const qx = Q[o], qy = Q[o + 1], qz = Q[o + 2], qw = Q[o + 3];
  const x = qx + (hx * qw + hy * qz - hz * qy);
  const y = qy + (hy * qw + hz * qx - hx * qz);
  const z = qz + (hz * qw + hx * qy - hy * qx);
  const w = qw - (hx * qx + hy * qy + hz * qz);
  const l = 1 / Math.sqrt(x * x + y * y + z * z + w * w);
  Q[o] = x * l; Q[o + 1] = y * l; Q[o + 2] = z * l; Q[o + 3] = w * l;
}

// 3x3 row-major inverse (zeros when singular)
function invert3(m, out, o) {
  const a = m[0], b = m[1], c = m[2], d = m[3], e = m[4], f = m[5], g = m[6], h = m[7], k = m[8];
  const A = e * k - f * h, B = -(d * k - f * g), C = d * h - e * g;
  const det = a * A + b * B + c * C;
  if (!(det > 1e-300 || det < -1e-300) || !Number.isFinite(det)) { out.fill(0, o, o + 9); return; }
  const id = 1 / det;
  out[o] = A * id; out[o + 1] = -(b * k - c * h) * id; out[o + 2] = (b * f - c * e) * id;
  out[o + 3] = B * id; out[o + 4] = (a * k - c * g) * id; out[o + 5] = -(a * f - c * d) * id;
  out[o + 6] = C * id; out[o + 7] = -(a * h - b * g) * id; out[o + 8] = (a * e - b * d) * id;
}

const vx = (v) => (v == null ? 0 : typeof v.x === 'number' ? v.x : v[0]);
const vy = (v) => (v == null ? 0 : typeof v.y === 'number' ? v.y : v[1]);
const vz = (v) => (v == null ? 0 : typeof v.z === 'number' ? v.z : v[2]);
const vw = (v) => (v == null ? 1 : typeof v.w === 'number' ? v.w : v[3]);
function writeVec(out, x, y, z) {
  if (!out) return;
  if (typeof out.length === 'number') { out[0] = x; out[1] = y; out[2] = z; } else { out.x = x; out.y = y; out.z = z; }
}
function writeQuat(out, x, y, z, w) {
  if (!out) return;
  if (typeof out.length === 'number') { out[0] = x; out[1] = y; out[2] = z; out[3] = w; } else { out.x = x; out.y = y; out.z = z; out.w = w; }
}
const makeHit = () => ({ distance: 0, point: new Vector3(), normal: new Vector3(), body: -1, box: null, kind: null, ground: false });
const makeRecord = (id) => ({ minX: 0.5, minY: 0.5, maxX: 0.5, maxY: 0.5, minZ: 0.5, maxZ: 0.5, kind: 'rigid', id, vx: 0.5, vz: 0.5, [DYN_RECORD]: true });

// ---- shape descriptors (serialize / from) ------------------------------------------------------------------------
// Exact numeric state of a shape whose factory re-derives values (convex re-centring, compound normalisation):
// copied back after re-creation so a round trip is bit-identical. Face planes are copied back only when the rebuilt
// hull has the very same face topology (faceStart / faceVerts, canonical in shapes.js), so old planes can never be
// paired with different faces; otherwise the rebuilt planes stay.
const EXACT_ARRAYS = ['centroid', 'half', 'offset', 'unitInertia'];
const sameArray = (x, y) => { if (!x || !y || x.length !== y.length) return false; for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) return false; return true; };
function exactOf(s) {
  const e = { volume: s.volume, boundingRadius: s.boundingRadius };
  for (const k of EXACT_ARRAYS) if (s[k]) e[k] = Array.from(s[k]);
  if (s.planes && s.faceStart && s.faceVerts) { e.planes = Array.from(s.planes); e.faceStart = Array.from(s.faceStart); e.faceVerts = Array.from(s.faceVerts); }
  return e;
}
function patchExact(s, e) {
  if (!e) return s;
  if (typeof e.volume === 'number') s.volume = e.volume;
  if (typeof e.boundingRadius === 'number') s.boundingRadius = e.boundingRadius;
  for (const k of EXACT_ARRAYS) if (e[k] && s[k] && s[k].length === e[k].length) s[k].set(e[k]);
  if (e.planes && s.planes && s.planes.length === e.planes.length && sameArray(s.faceStart, e.faceStart) && sameArray(s.faceVerts, e.faceVerts)) s.planes.set(e.planes);
  return s;
}
/** Plain-object descriptor of a shapes.js shape (enough to rebuild it bit-identically with shapeFromDescriptor). */
export function describeShape(s) {
  switch (s.type) {
    case SHAPE.SPHERE: return { type: s.type, radius: s.radius };
    case SHAPE.CAPSULE: return { type: s.type, radius: s.radius, halfHeight: s.halfHeight };
    case SHAPE.BOX: return { type: s.type, half: [s.half[0], s.half[1], s.half[2]] };
    case SHAPE.CONVEX: return { type: s.type, points: Array.from(s.points), exact: exactOf(s) };
    case SHAPE.TRIMESH: return { type: s.type, positions: Array.from(s.positions), indices: Array.from(s.indices) };
    case SHAPE.COMPOUND: return {
      type: s.type, exact: exactOf(s),
      children: s.children.map((c) => ({ shape: describeShape(c.shape), position: Array.from(c.pos), quaternion: Array.from(c.quat) })),
    };
    default: throw new TypeError(`describeShape: unknown shape type ${s && s.type}`);
  }
}
/** Rebuild a shape from describeShape()'s descriptor. */
export function shapeFromDescriptor(d) {
  switch (d.type) {
    case SHAPE.SPHERE: return sphere(d.radius);
    case SHAPE.CAPSULE: return capsule(d.radius, d.halfHeight);
    case SHAPE.BOX: return box(d.half[0], d.half[1], d.half[2]);
    case SHAPE.CONVEX: {
      const s = convex(d.points, { hull: false });
      if (s.points.length === d.points.length) s.points.set(d.points); // undo the re-centring round-off
      return patchExact(s, d.exact);
    }
    case SHAPE.TRIMESH: return trimesh(d.positions, d.indices);
    case SHAPE.COMPOUND: {
      const s = compound(d.children.map((c) => ({ shape: shapeFromDescriptor(c.shape), position: c.position, quaternion: c.quaternion })));
      for (let i = 0; i < s.children.length; i++) { s.children[i].pos.set(d.children[i].position); s.children[i].quat.set(d.children[i].quaternion); }
      return patchExact(s, d.exact);
    }
    default: throw new TypeError(`shapeFromDescriptor: unknown shape type ${d && d.type}`);
  }
}

// Snapshot / restore of a store's own typed arrays and numbers (JointStore, ...).
function snapshotFields(obj) {
  const out = {};
  for (const k of Object.keys(obj)) {
    const v = obj[k];
    if (ArrayBuffer.isView(v)) out[k] = v.slice();
    else if (typeof v === 'number' || typeof v === 'boolean' || typeof v === 'string') out[k] = v;
    else if (Array.isArray(v)) out[k] = v.slice();
  }
  return out;
}
function restoreFields(obj, s) {
  for (const k of Object.keys(s)) {
    const cur = obj[k], v = s[k];
    if (ArrayBuffer.isView(cur)) {
      const arr = asArray(v);
      if (cur.length === arr.length) cur.set(arr);
      else obj[k] = new cur.constructor(arr);
    } else if (Array.isArray(cur) && Array.isArray(v)) obj[k] = v.slice();
    else if (typeof cur === 'number' || typeof cur === 'boolean' || typeof cur === 'string' || cur === undefined) obj[k] = v;
  }
}
// typed-array view of a serialized array (typed array, plain array or a JSON'd typed array {0: .., 1: ..})
function asArray(v) { return v == null ? [] : ArrayBuffer.isView(v) || Array.isArray(v) ? v : Object.values(v); }

// ---- pair-key set (typed-array open addressing; insertion-ordered key list) ---------------------------------------
// Keys are held as int32 (key | 0): a static-world key 0xFFFF0000 + a is a negative small integer then, and V8 does
// not box it when it crosses a call (a uint32 above 2^30 would be a heap number).
class PairSet {
  constructor(cap) { this._alloc(cap); }
  _alloc(cap) {
    this.cap = cap; this.count = 0;
    this.keys = new Int32Array(cap); this.ida = new Int32Array(cap); this.idb = new Int32Array(cap);
    let size = 16;
    while (size < cap * 2) size *= 2;
    this.mask = size - 1;
    this.slot = new Int32Array(size);        // entry index
    this.stamp = new Uint32Array(size);      // slot is in use when stamp === gen
    this.gen = 1;
  }
  clear() {
    this.count = 0;
    if (++this.gen >= 0xffffffff) { this.stamp.fill(0); this.gen = 1; }
  }
  _hash(key) {
    let h = Math.imul(key ^ (key >>> 16), 0x45d9f3b);
    h = Math.imul(h ^ (h >>> 16), 0x45d9f3b);
    return (h ^ (h >>> 16)) & this.mask;
  }
  has(key) {
    const st = this.stamp, sl = this.slot, keys = this.keys, gen = this.gen, m = this.mask;
    for (let s = this._hash(key); st[s] === gen; s = (s + 1) & m) if (keys[sl[s]] === key) return true;
    return false;
  }
  /** Insert; false when already present. Grows (allocates) when full. */
  add(key, a, b) {
    if (this.count >= this.cap) this._grow();
    const st = this.stamp, sl = this.slot, keys = this.keys, gen = this.gen, m = this.mask;
    let s = this._hash(key);
    for (; st[s] === gen; s = (s + 1) & m) if (keys[sl[s]] === key) return false;
    const i = this.count++;
    st[s] = gen; sl[s] = i; keys[i] = key; this.ida[i] = a; this.idb[i] = b;
    return true;
  }
  _grow() {
    const keys = this.keys, ida = this.ida, idb = this.idb, n = this.count;
    this._alloc(this.cap * 2);
    for (let i = 0; i < n; i++) this.add(keys[i], ida[i], idb[i]);
  }
}

// ---- persistent static-friction anchors ---------------------------------------------------------------------------
// While a manifold sticks, its points keep the body-frame anchors they had when sticking began (keyed by the
// narrowphase's (pairKey, feature key)), so static friction pulls back the drift of earlier steps instead of only
// stopping the current one: no creep on slopes, piles settle and sleep. Open addressing on two uint32 keys; entries
// whose stamp is not the latest generation are empty (O(1) eviction of everything not written in the last step).
class AnchorCache {
  constructor(cap) {
    let size = 16;
    while (size < cap * 2) size *= 2;
    this.mask = size - 1;
    this.pk = new Uint32Array(size); this.key = new Uint32Array(size); this.stamp = new Uint32Array(size);
    this.fa = new Float64Array(3 * size); this.fb = new Float64Array(3 * size);
    this.gen = 1;
  }
  _hash(pk, key) {
    let h = Math.imul(pk ^ (pk >>> 16), 0x45d9f3b) ^ key;
    h = Math.imul(h ^ (h >>> 16), 0x45d9f3b);
    return (h ^ (h >>> 16)) & this.mask;
  }
  /** Slot of (pk, key) written in the latest generation, or -1. */
  find(pk, key) {
    const m = this.mask, st = this.stamp, g = this.gen;
    for (let s = this._hash(pk, key), n = 0; st[s] === g && n <= m; s = (s + 1) & m, n++) if (this.pk[s] === pk && this.key[s] === key) return s;
    return -1;
  }
  /** Start a new generation (everything written before becomes empty for the writes that follow). */
  next() { if (++this.gen >= 0xffffffff) { this.stamp.fill(0); this.gen = 1; } }
  /** Slot to write (pk, key) into in the current generation (-1 when the table is full). */
  slot(pk, key) {
    const m = this.mask, st = this.stamp, g = this.gen;
    let s = this._hash(pk, key);
    for (let n = 0; n <= m; s = (s + 1) & m, n++) {
      if (st[s] !== g) { st[s] = g; this.pk[s] = pk; this.key[s] = key; return s; }
      if (this.pk[s] === pk && this.key[s] === key) return s;
    }
    return -1;
  }
  serialize() {
    const pk = [], key = [], fa = [], fb = [];
    for (let s = 0; s <= this.mask; s++) {
      if (this.stamp[s] !== this.gen) continue;
      pk.push(this.pk[s]); key.push(this.key[s]);
      for (let k = 0; k < 3; k++) { fa.push(this.fa[s * 3 + k]); fb.push(this.fb[s * 3 + k]); }
    }
    return { pk: Uint32Array.from(pk), key: Uint32Array.from(key), fa: Float64Array.from(fa), fb: Float64Array.from(fb) };
  }
  restore(d) {
    const pk = asArray(d.pk), key = asArray(d.key), fa = asArray(d.fa), fb = asArray(d.fb);
    this.next();
    for (let i = 0; i < pk.length; i++) {
      const s = this.slot(pk[i], key[i]);
      if (s < 0) break;
      for (let k = 0; k < 3; k++) { this.fa[s * 3 + k] = fa[i * 3 + k]; this.fb[s * 3 + k] = fb[i * 3 + k]; }
    }
  }
}

// append int32 key (as its uint32 value) to a { count, keys: Uint32Array } list (grows by doubling)
function pushKey(list, key) {
  if (list.count >= list.keys.length) { const k = new Uint32Array(list.keys.length * 2); k.set(list.keys); list.keys = k; }
  list.keys[list.count++] = key >>> 0;
}

export class DynamicsWorld {
  /**
   * @param {object} [o]
   * @param {number[]|{x,y,z}} [o.gravity=[0,-9.81,0]]
   * @param {number} [o.substeps=4]          XPBD substeps per step
   * @param {number} [o.maxBodies=4096]
   * @param {number} [o.maxContacts=16384]
   * @param {number} [o.maxJoints=1024]
   * @param {{linear:number, angular:number, time:number}|false} [o.sleep]
   * @param {import('../PhysicsWorld.js').PhysicsWorld|null} [o.physics=null]  static world (boxes, heightfield)
   * @param {number} [o.staticMask=-1]       layers of the static world the bodies collide with (LAYER / layerOf bits)
   * @param {number} [o.margin=0.01]         contact margin (speculative contacts, AABB inflation)
   * @param {number} [o.seed=1]
   * @param {number} [o.positionIterations=1]  contact sweeps per substep (each forward + backward) before the joints
   * @param {number} [o.velocityIterations=1]  likewise for the velocity pass
   * @param {number} [o.broadphaseMargin=0.1] fat-AABB slack of the broadphase tree (pairs do not depend on it)
   * @param {boolean} [o.staticProviders=true] also collide with the PhysicsWorld's dynamic providers (traffic, ...)
   * @param {boolean} [o.timing=false]   fill stats.ms / msBroad / msNarrow / msSolve (performance.now(), also in the
   *   narrowphase); off, step() calls no timer at all
   * @param {number} [o.maxParticles=0]   > 0: own a ParticleStore of that capacity + the built-in ParticleConstraints
   * @param {number} [o.particleDamping=0.01]  1/s (clamped at 0), particle velocity damping per substep (vel *= 1 - d h)
   */
  constructor(o = {}) {
    const {
      gravity = [0, -9.81, 0], substeps = 4, maxBodies = 4096, maxContacts = 16384, maxJoints = 1024,
      sleep = DEFAULT_SLEEP, physics = null, staticMask = -1, margin = 0.01, seed = 1,
      positionIterations = 1, velocityIterations = 1, broadphaseMargin = 0.1, staticProviders = true, timing = false,
    } = o;
    this.gravity = new Float64Array([vx(gravity), vy(gravity), vz(gravity)]);
    this.substeps = Math.max(1, substeps | 0);
    this.positionIterations = Math.max(1, positionIterations | 0);
    this.velocityIterations = Math.max(0, velocityIterations | 0);
    this.margin = margin;
    this.staticMask = staticMask | 0;
    this.seed = seed >>> 0;
    this.sleepParams = sleep ? { linear: sleep.linear ?? DEFAULT_SLEEP.linear, angular: sleep.angular ?? DEFAULT_SLEEP.angular, time: sleep.time ?? DEFAULT_SLEEP.time } : false;
    this.broadphaseMargin = broadphaseMargin;
    this.staticProviders = staticProviders !== false;
    this.timing = timing === true && hasPerf;
    this.maxBodies = maxBodies; this.maxContacts = maxContacts; this.maxJoints = maxJoints;

    this.bodies = new BodyStore(maxBodies);
    this.contacts = new ContactBuffer(maxContacts);
    this.broadphase = new Broadphase(maxBodies, { margin: broadphaseMargin });
    let table = 1024;
    while (table < maxContacts * 2) table *= 2;
    this.narrowphase = new Narrowphase({ maxContacts, margin, tableSize: table, timing: this.timing });
    this.joints = new JointStore(maxJoints);
    this.islands = new Islands(maxBodies);
    this.physics = physics;

    this.time = 0;
    this.stepCount = 0;
    this._bpDirty = true;                          // the broadphase lags bodies.aabb (see _syncBroadphase)
    this._dt = 0.5;                                // step / substep length of the running step (fields, not arguments)
    this._h = 0.5;
    this._restThreshold = 0.5;
    this._tagged = [0.5, 0.5, 0.5, 0.5, null];     // h, gx, gy, gz as boxed numbers (generic elements; see step)
    this.stats = { bodies: 0, awake: 0, pairs: 0, contacts: 0, joints: 0, islands: 0, ms: 0, msBroad: 0, msNarrow: 0, msSolve: 0 };
    this.began = { count: 0, keys: new Uint32Array(256) };
    this.ended = { count: 0, keys: new Uint32Array(256) };

    // per body
    this._kinPending = new Uint8Array(maxBodies);  // a kinematic target waits for the next step
    this._noAnchors = new Uint8Array(maxBodies);   // removed / added since the last step: skip its cached anchors
    this._kinMode = new Uint8Array(maxBodies);     // 1: target-driven (stops when targets stop)
    this._active = new Uint8Array(maxBodies);      // this step: awake dynamic or moving kinematic
    this._act0 = new Uint8Array(maxBodies);        // active when the current narrowphase pass started
    this._kind = new Array(maxBodies).fill('rigid');
    this._desc = new Array(maxBodies).fill(null);  // shape descriptor per body (serialize)
    this._records = new Array(maxBodies).fill(null);
    this._descCache = new WeakMap();               // shape object -> descriptor (shared shapes described once)
    // per contact (solver)
    const nc = maxContacts;
    this._la = new Float64Array(3 * nc);           // anchor on a (body frame)
    this._lb = new Float64Array(3 * nc);           // anchor on b (body frame; world point for WORLD_BODY)
    this._mu = new Float64Array(nc);
    this._e = new Float64Array(nc);
    this._vn0 = new Float64Array(nc);              // pre-solve normal velocity of the substep
    this._lamN = new Float64Array(nc);             // substep multipliers
    this._lamT = new Float64Array(nc);
    this._impN = new Float64Array(nc);             // step impulses (-> contacts.lambdaN / lambdaT)
    this._impT = new Float64Array(nc);
    this._mfJt = new Float64Array(nc);             // manifold: dynamic friction impulse used in the substep (budget)
    this._mfJtw = new Float64Array(nc);            // manifold: torsional friction impulse used in the substep
    this._mfStart = new Int32Array(nc + 1);        // manifold m = contacts [_mfStart[m], _mfStart[m + 1])
    this._mfT = new Float64Array(3 * nc);          // manifold static friction multipliers (t1, t2, twist) per substep
    this._mfStick = new Uint8Array(nc);            // manifold: static friction held in its latest solve
    this._fa = new Float64Array(3 * nc);           // static-friction anchor on a (body frame)
    this._fb = new Float64Array(3 * nc);           // ... on b (body frame; world point for WORLD_BODY)
    this._frictionCache = new AnchorCache(nc);
    this._mfCount = 0;
    // static world
    this._statics = [];
    this._staticIdx = new Map();                   // static box object -> grid index (looked up, never iterated)
    this._staticIndexed = 0;
    this._staticTagged = [];                       // grid index -> [minX, minY, minZ, maxX, maxY, maxZ] (boxed once)
    this._groundSlab = [0, 0, 0, 0, NaN, 0, null]; // flat ground as a static box (built on first use)
    this._sqx = new Float64Array(maxBodies);       // per body: where the last empty static query was made ...
    this._sqz = new Float64Array(maxBodies);
    this._sqr = new Float64Array(maxBodies);       // ... with what radius
    this._sqn = new Int32Array(maxBodies).fill(-1); // ... and the grid's box count then (-1: no cached answer)
    this._sqCnt = new Int32Array(maxBodies);       // ... the number of boxes it found
    this._sqIdx = new Int32Array(maxBodies * SQ_CAP); // ... and their grid indices (ascending)
    this._sIdx = new Int32Array(64);               // candidate grid indices of the current body (grows on overflow)
    this._staticGrid = null;
    this._inStaticPass = false;
    this._heightAt = (x, z, y) => this.physics.groundHeight(x, z, y);
    // events
    this._prevPairs = new PairSet(1024);
    this._curPairs = new PairSet(1024);
    // queries
    this._phHit = makeHit();
    this._phOpts = { mask: -1, filter: null, statics: true, dynamics: true, ground: true, out: null, hfStep: undefined };
    this._userStaticFilter = null;
    this._phOpts.filter = (b) => b[DYN_RECORD] !== true && (this._userStaticFilter === null || this._userStaticFilter(b) !== false);
    /** Stable bound queryColliders (register it with physics.addDynamicProvider to publish the bodies). */
    this.colliderProvider = (x, z, r, out) => this.queryColliders(x, z, r, out);
    this.colliderProvider[DYN_RECORD] = true;      // marks a dynamics provider: never called back as static geometry
    // particles (0.9.0): null / empty without maxParticles, and step() then never touches them
    const { maxParticles = 0, particleDamping = 0.01 } = o;
    this.maxParticles = maxParticles > 0 ? maxParticles | 0 : 0;
    this.particleDamping = particleDamping > 0 ? particleDamping : 0;
    /** @type {ParticleStore|null} */
    this.particles = this.maxParticles > 0 ? new ParticleStore(this.maxParticles) : null;
    /** @type {ParticleConstraints|null} */
    this.constraints = this.particles !== null ? new ParticleConstraints(this.particles, Math.max(64, this.maxParticles * 8), this.bodies) : null;
    /** @type {object[]} */
    this.particleSolvers = this.constraints !== null ? [this.constraints] : [];
    this._ptag = [0.5, null];                      // dt as a boxed number for the solver hooks (generic elements)
    this._jskip = new Int32Array(2 * Math.max(32, maxJoints)); // (a, b) pairs the joint filter skipped this step (grows on overflow)
    this._jskipCount = 0;
    this._mark = new Uint8Array(maxBodies);        // recollidePairs body marker (cleared after each call)
    this._inBeginStep = false;
    this.stats.particles = 0; this.stats.constraints = 0;
  }

  // ---- bodies ------------------------------------------------------------------------------------------------------
  /** Add a body; returns its id. See the header for the options. */
  addBody(o = {}) {
    const shape = o.shape;
    if (!shape || typeof shape.aabb !== 'function') throw new TypeError('DynamicsWorld.addBody: a shapes.js shape is required');
    const t = typeof o.type === 'number' ? o.type : TYPE_OF[o.type ?? 'dynamic'];
    if (t === undefined) throw new RangeError(`DynamicsWorld.addBody: unknown type ${o.type}`);
    if (t === DYNAMIC && shape.type === SHAPE.TRIMESH) throw new RangeError('DynamicsWorld.addBody: trimesh shapes are static only');
    const B = this.bodies;
    const id = B.alloc();
    if (id < 0) throw new RangeError(`DynamicsWorld.addBody: maxBodies (${B.capacity}) reached`);
    B.type[id] = t;
    const p = o.position, q = o.quaternion;
    const o3 = id * 3, o4 = id * 4;
    B.pos[o3] = vx(p); B.pos[o3 + 1] = vy(p); B.pos[o3 + 2] = vz(p);
    let qx = q ? vx(q) : 0, qy = q ? vy(q) : 0, qz = q ? vz(q) : 0, qw = q ? vw(q) : 1;
    const ql = Math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
    if (ql > 0 && ql !== 1) { qx /= ql; qy /= ql; qz /= ql; qw /= ql; } else if (!(ql > 0)) { qx = 0; qy = 0; qz = 0; qw = 1; }
    B.quat[o4] = qx; B.quat[o4 + 1] = qy; B.quat[o4 + 2] = qz; B.quat[o4 + 3] = qw;
    B.prevPos[o3] = B.pos[o3]; B.prevPos[o3 + 1] = B.pos[o3 + 1]; B.prevPos[o3 + 2] = B.pos[o3 + 2];
    B.prevQuat[o4] = qx; B.prevQuat[o4 + 1] = qy; B.prevQuat[o4 + 2] = qz; B.prevQuat[o4 + 3] = qw;
    if (t === DYNAMIC) {
      const m = Number.isFinite(o.mass) && o.mass > 0 ? o.mass : (o.density ?? 1000) * shape.volume;
      if (!(m > 0) || !Number.isFinite(m)) { B.free(id); throw new RangeError('DynamicsWorld.addBody: a dynamic body needs a positive mass (or density and a shape with volume)'); }
      B.mass[id] = m; B.invMass[id] = 1 / m;
      shape.inertia(m, _m9);
      invert3(_m9, B.invInertia, id * 9);
    }
    B.friction[id] = o.friction ?? 0.6;
    B.restitution[id] = o.restitution ?? 0.1;
    B.linearDamping[id] = o.linearDamping ?? 0.01;
    B.angularDamping[id] = o.angularDamping ?? 0.05;
    B.layer[id] = (o.layer ?? 1) | 0;
    B.mask[id] = (o.mask ?? -1) | 0;
    if (o.noGravity) B.flags[id] |= NO_GRAVITY;
    B.boundingRadius[id] = shape.boundingRadius;
    B.shape[id] = shape;
    B.userData[id] = o.userData ?? null;
    this._kind[id] = o.kind ?? 'rigid';
    let d = this._descCache.get(shape);
    if (!d) { d = describeShape(shape); this._descCache.set(shape, d); }
    this._desc[id] = d;
    if (!this._records[id]) this._records[id] = makeRecord(id);
    this._records[id].kind = this._kind[id];
    this._kinPending[id] = 0; this._kinMode[id] = 0;
    this._sqn[id] = -1;            // a reused id must not inherit the static-query cache ...
    this._noAnchors[id] = 1;       // ... nor last step's friction anchors (see removeBody)
    this.islands.reset(id);
    this._refreshAabb(id);
    this._bpDirty = true;
    return id;
  }

  /** Remove a body (wakes what touched it, removes its joints). */
  removeBody(id) {
    const B = this.bodies;
    if (!(id >= 0 && id < B.count) || !B.alive[id]) return;
    this.islands.wake(B, id);
    this._wakePartners(id);   // its support role ends
    // drop its contacts from the last step's buffer
    const C = this.contacts;
    let k = 0;
    for (let i = 0; i < C.count; i++) {
      if (C.a[i] === id || C.b[i] === id) continue;
      if (k !== i) this._moveContact(i, k);
      k++;
    }
    C.count = k;
    const J = this.joints;
    for (let j = 0; j < J.count; j++) {
      if (!J.alive[j] || (J.a[j] !== id && J.b[j] !== id)) continue;
      const other = J.a[j] === id ? J.b[j] : J.a[j];
      if (other >= 0) this.wake(other);
      J.remove(j);
    }
    this.broadphase.remove(id);
    if (typeof this.narrowphase.evictBody === 'function') this.narrowphase.evictBody(id); // warm-start entries of the id
    this._noAnchors[id] = 1;       // the anchor cache's entries of this id are ignored until the next step rewrites it
    this._sqn[id] = -1;
    this._bpDirty = true;
    this._kinPending[id] = 0; this._kinMode[id] = 0; this._active[id] = 0; this._act0[id] = 0;
    this._desc[id] = null; this._kind[id] = 'rigid';
    if (this.constraints !== null) this.constraints.detachBody(id); // particle attachments to it
    B.free(id);
    this.islands.reset(id);
  }

  _moveContact(i, k) {
    const C = this.contacts;
    C.a[k] = C.a[i]; C.b[k] = C.b[i]; C.px[k] = C.px[i]; C.py[k] = C.py[i]; C.pz[k] = C.pz[i];
    C.nx[k] = C.nx[i]; C.ny[k] = C.ny[i]; C.nz[k] = C.nz[i]; C.depth[k] = C.depth[i]; C.key[k] = C.key[i];
    C.pairKey[k] = C.pairKey[i]; C.lambdaN[k] = C.lambdaN[i]; C.lambdaT[k] = C.lambdaT[i]; C.staticIndex[k] = C.staticIndex[i];
  }

  _check(id, who) {
    if (!(id >= 0 && id < this.bodies.count) || !this.bodies.alive[id]) throw new RangeError(`DynamicsWorld.${who}: no body ${id}`);
  }

  /** Teleport a body: setPose(id, x, y, z, qx, qy, qz, qw) or setPose(id, pos, quat). Wakes it. */
  setPose(id, x, y, z, qx, qy, qz, qw) {
    this._check(id, 'setPose');
    if (typeof x !== 'number') {
      const p = x, q = y;
      x = vx(p); y = vy(p); z = vz(p);
      if (q) { qx = vx(q); qy = vy(q); qz = vz(q); qw = vw(q); }
    }
    const B = this.bodies, o3 = id * 3, o4 = id * 4;
    if (qx === undefined) { qx = B.quat[o4]; qy = B.quat[o4 + 1]; qz = B.quat[o4 + 2]; qw = B.quat[o4 + 3]; }
    const ql = Math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
    if (ql > 0 && ql !== 1) { qx /= ql; qy /= ql; qz /= ql; qw /= ql; } else if (!(ql > 0)) { qx = 0; qy = 0; qz = 0; qw = 1; }
    B.pos[o3] = x; B.pos[o3 + 1] = y; B.pos[o3 + 2] = z;
    B.quat[o4] = qx; B.quat[o4 + 1] = qy; B.quat[o4 + 2] = qz; B.quat[o4 + 3] = qw;
    B.prevPos[o3] = x; B.prevPos[o3 + 1] = y; B.prevPos[o3 + 2] = z;
    B.prevQuat[o4] = qx; B.prevQuat[o4 + 1] = qy; B.prevQuat[o4 + 2] = qz; B.prevQuat[o4 + 3] = qw;
    if (B.type[id] === KINEMATIC) this._kinPending[id] = 0;
    this._refreshAabb(id);
    this._bpDirty = true;
    this.wake(id);
    // a moved static / kinematic body changes the support of its (possibly sleeping) neighbours
    if (B.type[id] !== DYNAMIC) this._wakePartners(id);
  }

  /** Pose into outPos ({x,y,z} or array) and outQuat ({x,y,z,w} or array); either may be null. */
  getPose(id, outPos, outQuat) {
    const B = this.bodies, o3 = id * 3, o4 = id * 4;
    writeVec(outPos, B.pos[o3], B.pos[o3 + 1], B.pos[o3 + 2]);
    writeQuat(outQuat, B.quat[o4], B.quat[o4 + 1], B.quat[o4 + 2], B.quat[o4 + 3]);
  }

  /** setVelocity(id, vx, vy, vz[, wx, wy, wz]) or setVelocity(id, v, w?) (world-space angular velocity, kept when omitted). Wakes. */
  setVelocity(id, x, y, z, wx, wy, wz) {
    this._check(id, 'setVelocity');
    const B = this.bodies, o3 = id * 3;
    if (typeof x !== 'number') {
      const v = x, w = y;
      x = vx(v); y = vy(v); z = vz(v);
      if (w) { wx = vx(w); wy = vy(w); wz = vz(w); } else { wx = B.angVel[o3]; wy = B.angVel[o3 + 1]; wz = B.angVel[o3 + 2]; }
    }
    if (B.type[id] === STATIC) return;
    B.vel[o3] = x; B.vel[o3 + 1] = y; B.vel[o3 + 2] = z;
    if (typeof wx === 'number') { B.angVel[o3] = wx; B.angVel[o3 + 1] = wy ?? 0; B.angVel[o3 + 2] = wz ?? 0; } // else kept
    if (B.type[id] === KINEMATIC) { this._kinMode[id] = 0; this._kinPending[id] = 0; }
    this.wake(id);
  }

  /** Linear velocity into outV, angular (world) into outW ({x,y,z} or arrays; either may be null). */
  getVelocity(id, outV, outW) {
    const B = this.bodies, o3 = id * 3;
    writeVec(outV, B.vel[o3], B.vel[o3 + 1], B.vel[o3 + 2]);
    writeVec(outW, B.angVel[o3], B.angVel[o3 + 1], B.angVel[o3 + 2]);
  }

  /** applyImpulse(id, px, py, pz, ix, iy, iz) (world point, impulse N s) or applyImpulse(id, impulse, point?). Wakes. */
  applyImpulse(id, px, py, pz, ix, iy, iz) {
    this._check(id, 'applyImpulse');
    const B = this.bodies, o3 = id * 3;
    if (typeof px !== 'number') {
      const imp = px, at = py;
      ix = vx(imp); iy = vy(imp); iz = vz(imp);
      if (at) { px = vx(at); py = vy(at); pz = vz(at); } else { px = B.pos[o3]; py = B.pos[o3 + 1]; pz = B.pos[o3 + 2]; }
    }
    if (B.type[id] !== DYNAMIC) return;
    this.wake(id);
    const im = B.invMass[id];
    B.vel[o3] += ix * im; B.vel[o3 + 1] += iy * im; B.vel[o3 + 2] += iz * im;
    const rx = px - B.pos[o3], ry = py - B.pos[o3 + 1], rz = pz - B.pos[o3 + 2];
    worldInvInertiaApply(B, id, ry * iz - rz * iy, rz * ix - rx * iz, rx * iy - ry * ix, _w3);
    B.angVel[o3] += _w3[0]; B.angVel[o3 + 1] += _w3[1]; B.angVel[o3 + 2] += _w3[2];
  }

  /** applyForce(id, fx, fy, fz[, px, py, pz]) or applyForce(id, force, point?): accumulated for the next step. */
  applyForce(id, fx, fy, fz, px, py, pz) {
    this._check(id, 'applyForce');
    const B = this.bodies, o3 = id * 3;
    if (typeof fx !== 'number') {
      const f = fx, at = fy;
      fx = vx(f); fy = vy(f); fz = vz(f);
      if (at) { px = vx(at); py = vy(at); pz = vz(at); } else px = undefined;
    }
    if (B.type[id] !== DYNAMIC) return;
    B.force[o3] += fx; B.force[o3 + 1] += fy; B.force[o3 + 2] += fz;
    if (typeof px === 'number') {
      const rx = px - B.pos[o3], ry = py - B.pos[o3 + 1], rz = pz - B.pos[o3 + 2];
      B.torque[o3] += ry * fz - rz * fy; B.torque[o3 + 1] += rz * fx - rx * fz; B.torque[o3 + 2] += rx * fy - ry * fx;
    }
    if (fx !== 0 || fy !== 0 || fz !== 0) this.wake(id);
  }

  /** applyTorque(id, tx, ty, tz) or applyTorque(id, torque): world-space torque accumulated for the next step. */
  applyTorque(id, tx, ty, tz) {
    this._check(id, 'applyTorque');
    if (typeof tx !== 'number') { const t = tx; tx = vx(t); ty = vy(t); tz = vz(t); }
    const B = this.bodies, o3 = id * 3;
    if (B.type[id] !== DYNAMIC) return;
    B.torque[o3] += tx; B.torque[o3 + 1] += ty; B.torque[o3 + 2] += tz;
    if (tx !== 0 || ty !== 0 || tz !== 0) this.wake(id);
  }

  /** Kinematic target for the next step: (id, x, y, z, qx, qy, qz, qw) or (id, pos, quat). */
  setKinematicTarget(id, x, y, z, qx, qy, qz, qw) {
    this._check(id, 'setKinematicTarget');
    const B = this.bodies, o4 = id * 4;
    if (B.type[id] !== KINEMATIC) throw new RangeError(`DynamicsWorld.setKinematicTarget: body ${id} is not kinematic`);
    if (typeof x !== 'number') {
      const p = x, q = y;
      x = vx(p); y = vy(p); z = vz(p);
      if (q) { qx = vx(q); qy = vy(q); qz = vz(q); qw = vw(q); } else qx = undefined;
    }
    if (qx === undefined) { qx = B.quat[o4]; qy = B.quat[o4 + 1]; qz = B.quat[o4 + 2]; qw = B.quat[o4 + 3]; }
    const ql = Math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
    if (ql > 0 && ql !== 1) { qx /= ql; qy /= ql; qz /= ql; qw /= ql; } else if (!(ql > 0)) { qx = 0; qy = 0; qz = 0; qw = 1; }
    const t = B.kinTarget, o7 = id * 7;
    t[o7] = x; t[o7 + 1] = y; t[o7 + 2] = z; t[o7 + 3] = qx; t[o7 + 4] = qy; t[o7 + 5] = qz; t[o7 + 6] = qw;
    this._kinPending[id] = 1;
    this._kinMode[id] = 1;
  }

  /** setGravity(x, y, z) or setGravity(vec | array). */
  setGravity(x, y, z) {
    if (typeof x !== 'number') { const g = x; x = vx(g); y = vy(g); z = vz(g); }
    this.gravity[0] = x; this.gravity[1] = y; this.gravity[2] = z;
    // gravity changes the balance of resting bodies: wake everything
    const B = this.bodies;
    for (let i = 0; i < B.count; i++) if (B.alive[i] && B.type[i] === DYNAMIC) this.islands.wake(B, i);
  }

  /** The body's `kind` (addBody option). */
  kindOf(id) { return this._kind[id]; }

  // ---- joints ------------------------------------------------------------------------------------------------------
  /** Add a joint (joints.js descriptor); wakes both bodies. Returns its id. */
  addJoint(desc) {
    const J = this.joints;
    if (typeof desc.type === 'string') { // public API: 'fixed' | 'ball' | 'hinge' | 'slider' | 'distance' | 'd6'
      const t = JOINT[desc.type.toUpperCase()];
      if (t === undefined) throw new TypeError(`DynamicsWorld.addJoint: unknown joint type ${desc.type}`);
      desc = Object.assign({}, desc, { type: t });
    }
    const id = J.add(desc, this.bodies);
    if (id >= 0) { this.wake(J.a[id]); this.wake(J.b[id]); }
    return id;
  }
  /** Remove a joint (wakes both bodies). */
  removeJoint(id) {
    const J = this.joints;
    if (!(id >= 0 && id < J.count) || !J.alive[id]) return;
    this.wake(J.a[id]); this.wake(J.b[id]);
    J.remove(id);
  }

  // ---- sleeping ----------------------------------------------------------------------------------------------------
  /** Wake a body and its whole sleeping island. */
  wake(id) {
    const B = this.bodies;
    if (!(id >= 0 && id < B.count) || !B.alive[id]) return;
    B.sleepTime[id] = 0;
    this.islands.wake(B, id);
  }
  /** Put one dynamic body to sleep (velocities zeroed); contacts with awake bodies wake it again. */
  sleep(id) { this.islands.sleepBody(this.bodies, id); }
  /** True when the body is alive and not asleep. */
  isAwake(id) { const B = this.bodies; return id >= 0 && id < B.count && B.alive[id] === 1 && (B.flags[id] & ASLEEP) === 0; }
  /** True when the body is integrated this step (during a step) / was in the last one (between steps): see the header. */
  isActive(id) { const B = this.bodies; return id >= 0 && id < B.count && B.alive[id] === 1 && this._active[id] === 1; }

  // wake every body that touched `id` in the last step (the touching-pair set also holds sleeping pairs, which
  // produce no contacts)
  _wakePartners(id) {
    const p = this._prevPairs;
    for (let j = 0; j < p.count; j++) {
      const a = p.ida[j], b = p.idb[j];
      if (a === id && b >= 0) this.wake(b); else if (b === id) this.wake(a);
    }
  }

  // ---- step --------------------------------------------------------------------------------------------------------
  /** Advance by dt (fixed step; substeps inside). */
  step(dt) {
    if (!(dt > 0)) return;
    const timing = this.timing, t0 = timing ? performance.now() : 0;
    const B = this.bodies, C = this.contacts, J = this.joints, bp = this.broadphase, np = this.narrowphase;
    const n = B.count, alive = B.alive, type = B.type, flags = B.flags, act = this._active, act0 = this._act0;
    const st = this.stats;

    // 1. kinematic velocities, active set, AABBs
    this._dt = dt;
    for (let id = 0; id < n; id++) {
      act[id] = 0;
      if (!alive[id]) continue;
      const t = type[id];
      if (t === DYNAMIC) { if ((flags[id] & ASLEEP) === 0) act[id] = 1; }
      else if (t === KINEMATIC) {
        if (this._kinPending[id]) this._kinematicVelocity(id);
        else if (this._kinMode[id]) { this._kinMode[id] = 0; B.vel.fill(0, id * 3, id * 3 + 3); B.angVel.fill(0, id * 3, id * 3 + 3); }
        const o3 = id * 3, v = B.vel, w = B.angVel;
        if (v[o3] !== 0 || v[o3 + 1] !== 0 || v[o3 + 2] !== 0 || w[o3] !== 0 || w[o3 + 1] !== 0 || w[o3 + 2] !== 0) act[id] = 1;
        else if (this._kinPending[id]) { this._snapKinematic(id); this._kinPending[id] = 0; this._refreshAabb(id); } // target on the spot
      }
      if (act[id]) this._refreshAabb(id);
    }

    // 2. broadphase + narrowphase, waking what the active bodies touch
    this._bpDirty = true;                       // the active AABBs were refreshed
    let tb = timing ? performance.now() : 0;
    this._syncBroadphase();
    let msBroad = timing ? performance.now() - tb : 0;
    const tn = timing ? performance.now() : 0;
    let msLoopBroad = 0;
    np.beginStep();
    C.clear();
    this._mfCount = 0;
    this._jskipCount = 0;                       // pairs the joint filter skipped this step (recollidePairs)
    for (let id = 0; id < n; id++) act0[id] = act[id];
    this._collidePairs(false);
    let scanned = 0;
    for (let pass = 0; pass < MAX_WAKE_PASSES; pass++) {
      const woke = this._wakeFromContacts(scanned);
      scanned = C.count;
      if (woke === 0) break;
      tb = timing ? performance.now() : 0;
      bp.update(B);
      if (timing) { const dtb = performance.now() - tb; msBroad += dtb; msLoopBroad += dtb; }
      this._collidePairs(true);
      for (let id = 0; id < n; id++) act0[id] = act[id];
    }
    // 3. static world
    if (this.physics) {
      this._inStaticPass = true;
      for (let id = 0; id < n; id++) if (act[id] && type[id] === DYNAMIC) this._collideStatic(id);
      this._inStaticPass = false;
    }
    const msNarrow = timing ? performance.now() - tn - msLoopBroad : 0; // wake-pass broadphase updates count as broadphase
    const ts = timing ? performance.now() : 0;

    // 4. contact setup
    this._setupContacts();
    const PS = this.particles;                  // null: no particle integration
    const SV = PS !== null || this.particleSolvers.length > 0; // solver hooks (neither: the 0.8.0 step)
    if (SV) { this._ptag[0] = dt; this._beginParticleStep(); }

    // 5. substeps
    const ns = this.substeps, h = dt / ns, cc = C.count;
    const gx = this.gravity[0], gy = this.gravity[1], gz = this.gravity[2];
    this._h = h;
    // h and g as already-boxed numbers (a generic-elements array): passed as-is to integrateBody & co., they are not
    // re-boxed per body and substep the way a double read from a typed array would be
    const tg = this._tagged;
    tg[0] = h; tg[1] = gx; tg[2] = gy; tg[3] = gz;
    this._restThreshold = 2 * Math.sqrt(gx * gx + gy * gy + gz * gz) * h;
    const impN = this._impN, impT = this._impT, lamN = this._lamN, lamT = this._lamT;
    impN.fill(0, 0, cc); impT.fill(0, 0, cc);
    const hasJoints = J.count > 0;
    for (let s = 0; s < ns; s++) {
      const last = s === ns - 1;
      this._preSolveVelocities();
      lamN.fill(0, 0, cc); lamT.fill(0, 0, cc); this._mfT.fill(0, 0, 3 * this._mfCount);
      this._mfJt.fill(0, 0, this._mfCount); this._mfJtw.fill(0, 0, this._mfCount);
      for (let id = 0; id < n; id++) {
        if (!act[id]) continue;
        if (type[id] === DYNAMIC) {
          if ((flags[id] & NO_GRAVITY) !== 0) integrateBody(B, id, tg[0], 0, 0, 0);
          else integrateBody(B, id, tg[0], tg[1], tg[2], tg[3]);
        } else this._integrateKinematic(id, last);
      }
      if (PS !== null) this._integrateParticles();
      for (let it = 0; it < this.positionIterations; it++) {
        this._solveContactPositions();
        if (hasJoints) solveJointsPositions(J, B, tg[0], it);   // it 0 resets the joint lambdas
        if (SV) this._solveParticlePositions(it);
      }
      for (let id = 0; id < n; id++) if (act[id] && type[id] === DYNAMIC) velocityFromPositions(B, id, tg[0]);
      if (PS !== null) this._particleVelocities();
      for (let i = 0; i < cc; i++) { impN[i] += lamN[i] / h; impT[i] += lamT[i] / h; }
      for (let it = 0; it < this.velocityIterations; it++) {
        this._solveContactVelocities();
        if (hasJoints) solveJointsVelocities(J, B, tg[0]);
        if (SV) this._solveParticleVelocities(it);
      }
      this._damp();
      if (PS !== null) this._dampParticles();
    }

    // 6. contact impulses -> narrowphase, static-friction anchors -> cache
    for (let i = 0; i < cc; i++) { C.lambdaN[i] = impN[i]; C.lambdaT[i] = impT[i]; }
    np.endStep(C);
    this._storeFrictionAnchors();
    this._noAnchors.fill(0, 0, n);
    const msSolve = timing ? performance.now() - ts : 0;

    // 7. islands, sleeping, forces, AABBs, events, stats
    st.islands = this.islands.build(B, C, J);
    if (this.sleepParams) this.islands.updateSleep(B, C, J, dt, this.sleepParams);
    B.force.fill(0, 0, n * 3); B.torque.fill(0, 0, n * 3);
    let awake = 0, bodies = 0;
    for (let id = 0; id < n; id++) {
      if (!alive[id]) continue;
      bodies++;
      if (act[id]) { this._refreshAabb(id); this._bpDirty = true; }
      if (type[id] === DYNAMIC && (flags[id] & ASLEEP) === 0) awake++;
    }
    this._updateEvents();
    this.time += dt;
    this.stepCount++;
    let joints = 0;
    for (let j = 0; j < J.count; j++) if (J.alive[j]) joints++;
    st.bodies = bodies; st.awake = awake; st.pairs = bp.pairCount; st.contacts = C.count; st.joints = joints;
    st.msBroad = msBroad; st.msNarrow = msNarrow; st.msSolve = msSolve; st.ms = timing ? performance.now() - t0 : 0;
    if (SV) this._endParticleStep();
    if (PS !== null) { st.particles = PS.size; st.constraints = this.constraints.size; }
  }

  // ---- particles (particles.js; see "Particles" in the header) -----------------------------------------------------
  // The hooks pass only integers and the step's boxed numbers (this._ptag[0] = dt, this._tagged[0] = h) to the solvers,
  // and read h / g from fields and typed arrays, so nothing is boxed per call.

  /**
   * Register a particle solver / plugin (run after the solvers already registered). A world without particles accepts
   * plugins too (world.particles / constraints stay null: the plugin must tolerate that).
   * @param {object} solver  { name, beginStep?, solvePositions?, solveVelocities?, endStep?, serialize?, restore? }
   * @returns {object} the solver
   */
  addParticleSolver(solver) {
    if (!solver || typeof solver !== 'object') throw new TypeError('DynamicsWorld.addParticleSolver: a solver object is required');
    if (typeof solver.name !== 'string') throw new TypeError('DynamicsWorld.addParticleSolver: a solver needs a string `name` (serialize / from match solvers by it)');
    if (this.particleSolvers.includes(solver)) throw new RangeError('DynamicsWorld.addParticleSolver: solver already registered');
    this.particleSolvers.push(solver);
    return solver;
  }
  /** Unregister a particle solver (not the built-in constraints). Returns whether it was registered. */
  removeParticleSolver(solver) {
    if (solver !== null && solver === this.constraints) throw new RangeError('DynamicsWorld.removeParticleSolver: the built-in constraints solver cannot be removed');
    const k = this.particleSolvers.indexOf(solver);
    if (k < 0) return false;
    this.particleSolvers.splice(k, 1);
    return true;
  }
  /**
   * AABB of the alive particles of `group`, radius included, into out6 (minX, minY, minZ, maxX, maxY, maxZ).
   * @returns {boolean} false (out6 untouched) when the group has no alive particle
   */
  particleAabb(group, out6) {
    const P = this.particles;
    if (P === null) return false;
    const n = P.count, alive = P.alive, grp = P.group, x = P.pos, rad = P.radius;
    let found = false, x0 = 0, y0 = 0, z0 = 0, x1 = 0, y1 = 0, z1 = 0;
    for (let i = 0; i < n; i++) {
      if (alive[i] === 0 || grp[i] !== group) continue;
      const i3 = i * 3, r = rad[i], px = x[i3], py = x[i3 + 1], pz = x[i3 + 2];
      if (!found) { found = true; x0 = px - r; y0 = py - r; z0 = pz - r; x1 = px + r; y1 = py + r; z1 = pz + r; continue; }
      if (px - r < x0) x0 = px - r; if (py - r < y0) y0 = py - r; if (pz - r < z0) z0 = pz - r;
      if (px + r > x1) x1 = px + r; if (py + r > y1) y1 = py + r; if (pz + r > z1) z1 = pz + r;
    }
    if (!found) return false;
    out6[0] = x0; out6[1] = y0; out6[2] = z0; out6[3] = x1; out6[4] = y1; out6[5] = z1;
    return true;
  }
  _beginParticleStep() {
    const sv = this.particleSolvers, tp = this._ptag;
    this._inBeginStep = true;                    // recollidePairs is legal only here
    try {
      for (let k = 0; k < sv.length; k++) { const s = sv[k]; if (typeof s.beginStep === 'function') s.beginStep(this, tp[0]); }
    } finally { this._inBeginStep = false; }
  }
  _solveParticlePositions(it) {
    const sv = this.particleSolvers, tg = this._tagged;
    for (let k = 0; k < sv.length; k++) { const s = sv[k]; if (typeof s.solvePositions === 'function') s.solvePositions(this, tg[0], it); }
  }
  _solveParticleVelocities(it) {
    const sv = this.particleSolvers, tg = this._tagged;
    for (let k = 0; k < sv.length; k++) { const s = sv[k]; if (typeof s.solveVelocities === 'function') s.solveVelocities(this, tg[0], it); }
  }
  _endParticleStep() {
    const sv = this.particleSolvers, tp = this._ptag;
    for (let k = 0; k < sv.length; k++) { const s = sv[k]; if (typeof s.endStep === 'function') s.endStep(this, tp[0]); }
  }
  // prevPos = pos (every alive particle); alive and not pinned: vel += h g (unless NO_GRAVITY), pos += h vel
  _integrateParticles() {
    const P = this.particles, n = P.count, alive = P.alive, flags = P.flags, x = P.pos, x0 = P.prevPos, v = P.vel;
    const h = this._h, g = this.gravity, hgx = h * g[0], hgy = h * g[1], hgz = h * g[2];
    for (let i = 0; i < n; i++) {
      if (alive[i] === 0) continue;
      const i3 = i * 3, f = flags[i];
      x0[i3] = x[i3]; x0[i3 + 1] = x[i3 + 1]; x0[i3 + 2] = x[i3 + 2];
      if ((f & P_PINNED) !== 0) continue;
      if ((f & P_NO_GRAVITY) === 0) { v[i3] += hgx; v[i3 + 1] += hgy; v[i3 + 2] += hgz; }
      x[i3] += h * v[i3]; x[i3 + 1] += h * v[i3 + 1]; x[i3 + 2] += h * v[i3 + 2];
    }
  }
  // vel = (pos - prevPos) / h for alive, non-pinned particles
  _particleVelocities() {
    const P = this.particles, n = P.count, alive = P.alive, flags = P.flags, x = P.pos, x0 = P.prevPos, v = P.vel, h = this._h;
    for (let i = 0; i < n; i++) {
      if (alive[i] === 0 || (flags[i] & P_PINNED) !== 0) continue;
      const i3 = i * 3;
      v[i3] = (x[i3] - x0[i3]) / h; v[i3 + 1] = (x[i3 + 1] - x0[i3 + 1]) / h; v[i3 + 2] = (x[i3 + 2] - x0[i3 + 2]) / h;
    }
  }
  // vel *= max(0, 1 - particleDamping h) for alive, non-pinned particles
  _dampParticles() {
    let k = 1 - this.particleDamping * this._h;
    if (k < 0) k = 0;
    if (k === 1) return;
    const P = this.particles, n = P.count, alive = P.alive, flags = P.flags, v = P.vel;
    for (let i = 0; i < n; i++) {
      if (alive[i] === 0 || (flags[i] & P_PINNED) !== 0) continue;
      const i3 = i * 3;
      v[i3] *= k; v[i3 + 1] *= k; v[i3 + 2] *= k;
    }
  }

  // implied velocity of a kinematic target
  _kinematicVelocity(id) {
    const dt = this._dt, B = this.bodies, t = B.kinTarget, o7 = id * 7, o3 = id * 3, o4 = id * 4, P = B.pos, Q = B.quat;
    const inv = 1 / dt;
    B.vel[o3] = (t[o7] - P[o3]) * inv; B.vel[o3 + 1] = (t[o7 + 1] - P[o3 + 1]) * inv; B.vel[o3 + 2] = (t[o7 + 2] - P[o3 + 2]) * inv;
    // dq = target * conj(q)
    const ax = t[o7 + 3], ay = t[o7 + 4], az = t[o7 + 5], aw = t[o7 + 6];
    const bx = -Q[o4], by = -Q[o4 + 1], bz = -Q[o4 + 2], bw = Q[o4 + 3];
    let dx = aw * bx + ax * bw + ay * bz - az * by;
    let dy = aw * by - ax * bz + ay * bw + az * bx;
    let dz = aw * bz + ax * by - ay * bx + az * bw;
    let dw = aw * bw - ax * bx - ay * by - az * bz;
    if (dw < 0) { dx = -dx; dy = -dy; dz = -dz; dw = -dw; }
    const sn = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (sn < 1e-12) { B.angVel[o3] = 2 * dx * inv; B.angVel[o3 + 1] = 2 * dy * inv; B.angVel[o3 + 2] = 2 * dz * inv; }
    else {
      const k = (2 * detAtan2(sn, dw)) / sn * inv;
      B.angVel[o3] = dx * k; B.angVel[o3 + 1] = dy * k; B.angVel[o3 + 2] = dz * k;
    }
  }

  // move a kinematic body by one substep (exactly onto its target at the end of the step)
  _integrateKinematic(id, last) {
    const h = this._h, B = this.bodies, o3 = id * 3, o4 = id * 4, P = B.pos, Q = B.quat, PP = B.prevPos, PQ = B.prevQuat;
    PP[o3] = P[o3]; PP[o3 + 1] = P[o3 + 1]; PP[o3 + 2] = P[o3 + 2];
    PQ[o4] = Q[o4]; PQ[o4 + 1] = Q[o4 + 1]; PQ[o4 + 2] = Q[o4 + 2]; PQ[o4 + 3] = Q[o4 + 3];
    if (last && this._kinPending[id]) {
      this._snapKinematic(id);
      this._kinPending[id] = 0;
      return;
    }
    const v = B.vel, w = B.angVel;
    P[o3] += v[o3] * h; P[o3 + 1] += v[o3 + 1] * h; P[o3 + 2] += v[o3 + 2] * h;
    const wx = w[o3], wy = w[o3 + 1], wz = w[o3 + 2];
    if (wx === 0 && wy === 0 && wz === 0) return;
    const qx = Q[o4], qy = Q[o4 + 1], qz = Q[o4 + 2], qw = Q[o4 + 3], hh = 0.5 * h;
    let nx = qx + hh * (wx * qw + wy * qz - wz * qy);
    let ny = qy + hh * (wy * qw + wz * qx - wx * qz);
    let nz = qz + hh * (wz * qw + wx * qy - wy * qx);
    let nw = qw + hh * (-wx * qx - wy * qy - wz * qz);
    const l = Math.sqrt(nx * nx + ny * ny + nz * nz + nw * nw);
    nx /= l; ny /= l; nz /= l; nw /= l;
    Q[o4] = nx; Q[o4 + 1] = ny; Q[o4 + 2] = nz; Q[o4 + 3] = nw;
  }

  // bring the broadphase up to date with bodies.aabb when bodies were added, removed or moved since its last update
  // (queries read its tree; pairs are a pure function of the BodyStore, so an extra update changes no result)
  _syncBroadphase() {
    if (!this._bpDirty) return;
    this._bpDirty = false;
    this.broadphase.update(this.bodies);
  }

  _snapKinematic(id) {
    const B = this.bodies, t = B.kinTarget, o7 = id * 7, o3 = id * 3, o4 = id * 4, P = B.pos, Q = B.quat;
    P[o3] = t[o7]; P[o3 + 1] = t[o7 + 1]; P[o3 + 2] = t[o7 + 2];
    Q[o4] = t[o7 + 3]; Q[o4 + 1] = t[o7 + 4]; Q[o4 + 2] = t[o7 + 5]; Q[o4 + 3] = t[o7 + 6];
  }

  _refreshAabb(id) {
    const B = this.bodies, s = B.shape[id];
    if (!s) return;
    const o3 = id * 3, o4 = id * 4, P = B.pos, Q = B.quat;
    s.aabb(P[o3], P[o3 + 1], P[o3 + 2], Q[o4], Q[o4 + 1], Q[o4 + 2], Q[o4 + 3], _ab, this.margin);
    const A = B.aabb, o6 = id * 6;
    A[o6] = _ab[0]; A[o6 + 1] = _ab[1]; A[o6 + 2] = _ab[2]; A[o6 + 3] = _ab[3]; A[o6 + 4] = _ab[4]; A[o6 + 5] = _ab[5];
  }

  // Collide broadphase pairs. First pass: pairs with an active body. Wake passes (`fresh`): pairs whose bodies were
  // both inactive in the previous pass but now one is active (each pair is collided at most once per step).
  _collidePairs(fresh) {
    const B = this.bodies, C = this.contacts, np = this.narrowphase, bp = this.broadphase;
    const act = this._active, act0 = this._act0, pairs = bp.pairs, pc = bp.pairCount;
    const J = this.joints, jc = J.count > 0 && typeof J.connected === 'function';
    for (let k = 0; k < pc; k++) {
      const key = pairs[k], a = pairA(key), b = pairB(key);
      if (fresh) { if (act0[a] || act0[b] || (!act[a] && !act[b])) continue; }
      else if (!act[a] && !act[b]) continue;
      if (jc && J.connected(a, b)) {                // jointed pair without collideConnected: skipped, remembered
        let js = this._jskip, q = this._jskipCount * 2;
        if (q + 2 > js.length) { const g = new Int32Array(js.length * 2); g.set(js); js = this._jskip = g; } // growth
        js[q] = a; js[q + 1] = b; this._jskipCount++;
        continue;
      }
      const before = C.count;
      np.collide(B, a, b, C);
      if (C.count > before) this._addManifold(before);
    }
  }

  /**
   * From a particle solver / plugin's beginStep only (else it throws): collide this step's broadphase pairs (a, b)
   * whose bodies are both in ids[0 .. count) and that the joint filter skipped earlier in the step but are no longer
   * connected (a glue joint broken in beginStep), in ascending pair key, appending their contacts and setting them up
   * like stage 4 (manifolds continue from the current count), so freed pieces touch in the very step they are freed.
   * Pairs with a sleeping dynamic body, pairs already re-collided this step and still-connected pairs are left out.
   * Allocation-free (a body marker array, cleared after). Returns the number of contacts added (0: nothing changed).
   * @param {Int32Array|number[]} ids  body ids (caller-owned)
   * @param {number} count  how many of ids to use
   * @returns {number}
   */
  recollidePairs(ids, count) {
    if (!this._inBeginStep) throw new Error('DynamicsWorld.recollidePairs: only callable from a particle solver / plugin beginStep');
    if (!(count > 0) || this._jskipCount === 0) return 0;
    const J = this.joints;
    if (typeof J.connected !== 'function') return 0;
    const B = this.bodies, C = this.contacts, np = this.narrowphase, bp = this.broadphase, mark = this._mark, act = this._active;
    const n = B.count, type = B.type, flags = B.flags, js = this._jskip, nj = this._jskipCount;
    for (let q = 0; q < count; q++) { const id = ids[q]; if (id >= 0 && id < n) mark[id] = 1; }
    const start = C.count, mf0 = this._mfCount, pairs = bp.pairs, pc = bp.pairCount;
    for (let k = 0; k < pc; k++) {
      const key = pairs[k], a = pairA(key), b = pairB(key);
      if (mark[a] === 0 || mark[b] === 0 || (!act[a] && !act[b])) continue;
      if ((type[a] === DYNAMIC && (flags[a] & ASLEEP) !== 0) || (type[b] === DYNAMIC && (flags[b] & ASLEEP) !== 0)) continue;
      if (J.connected(a, b)) continue;
      let s = -1;
      for (let q = 0; q < nj; q++) if (js[q * 2] === a && js[q * 2 + 1] === b) { s = q; break; }
      if (s < 0) continue;                          // collided normally (not joint-skipped) or already re-collided
      js[s * 2] = -1;
      const before = C.count;
      np.collide(B, a, b, C);
      if (C.count > before) this._addManifold(before);
    }
    for (let q = 0; q < count; q++) { const id = ids[q]; if (id >= 0 && id < n) mark[id] = 0; }
    if (C.count > start) this._setupContacts(start, mf0);
    return C.count - start;
  }

  // Wake the sleeping bodies reached by contacts[from ..); returns the number of bodies woken.
  _wakeFromContacts(from) {
    const B = this.bodies, C = this.contacts, flags = B.flags, type = B.type;
    let woke = 0;
    for (let i = from; i < C.count; i++) {
      const a = C.a[i], b = C.b[i];
      if (type[a] === DYNAMIC && (flags[a] & ASLEEP) !== 0) woke += this.islands.wake(B, a);
      if (b >= 0 && type[b] === DYNAMIC && (flags[b] & ASLEEP) !== 0) woke += this.islands.wake(B, b);
    }
    if (woke === 0) return 0;
    const act = this._active, n = B.count, alive = B.alive;
    for (let id = 0; id < n; id++) {
      if (act[id] || !alive[id] || type[id] !== DYNAMIC || (flags[id] & ASLEEP) !== 0) continue;
      act[id] = 1;
      this._refreshAabb(id);
    }
    return woke;
  }

  // Adopt the PhysicsWorld's current grid: index its boxes (box object -> grid index, looked up and never iterated)
  // and box their bounds once into generic-element arrays (`_staticTagged`), forgetting a previous grid's state.
  _adoptGrid() {
    const grid = this.physics.grid, boxes = grid && grid.boxes;
    if (!boxes) return;
    if (grid !== this._staticGrid) {
      this._sqn.fill(-1); this._staticGrid = grid; this._staticIdx.clear(); this._staticIndexed = 0; this._staticTagged.length = 0;
    }
    for (let i = this._staticIndexed; i < boxes.length; i++) {
      const q = boxes[i];
      this._staticIdx.set(q, i);
      this._staticTagged[i] = [q.minX, q.minY, q.minZ, q.maxX, q.maxY, q.maxZ, null];
    }
    this._staticIndexed = boxes.length;
  }

  // Contacts of body a against the PhysicsWorld: the static grid boxes in ascending grid index, then the records of
  // the foreign dynamic providers (traffic, ...), then the ground. The grid is queried with STATIC_SLACK of extra
  // radius and the answer (up to SQ_CAP box indices, sorted) is cached per body and reused while the body stays
  // within the slack of where it asked and no box was added: a superset of the boxes a fresh query would return, and
  // the AABB cull below makes the result identical (ascending index order either way, so it is also independent of
  // the cache's history: serialize / from stay bit-identical).
  _collideStatic(a) {
    const ph = this.physics, B = this.bodies, C = this.contacts, np = this.narrowphase;
    const o3 = a * 3, x = B.pos[o3], z = B.pos[o3 + 2];
    const r = B.boundingRadius[a] + this.margin;
    const mask = B.mask[a] & this.staticMask;
    const out = this._statics;
    const A = B.aabb, o6 = a * 6;
    const minX = A[o6], minY = A[o6 + 1], minZ = A[o6 + 2], maxX = A[o6 + 3], maxY = A[o6 + 4], maxZ = A[o6 + 5];
    const grid = ph.grid, boxes = grid && grid.boxes;
    if (boxes && typeof ph.queryStatic === 'function') {
      if (grid !== this._staticGrid || boxes.length > this._staticIndexed) this._adoptGrid(); // adopt / index
      let idx = this._sIdx, n = 0;
      const dqx = x - this._sqx[a], dqz = z - this._sqz[a], c0 = a * SQ_CAP, room = this._sqr[a] - r;
      // reusable while the circle (x, z, r) lies inside the cached query circle and no box was added
      if (this._sqn[a] === boxes.length && room >= 0 && dqx * dqx + dqz * dqz <= room * room) {
        n = this._sqCnt[a];
        for (let q = 0; q < n; q++) idx[q] = this._sqIdx[c0 + q];
      } else {
        out.length = 0;
        ph.queryStatic(x, z, r + STATIC_SLACK, out);
        for (let j = 0; j < out.length; j++) {
          const k = this._staticIdx.get(out[j]);
          if (k === undefined) continue;
          if (n >= idx.length) { const g = new Int32Array(idx.length * 2); g.set(idx); idx = this._sIdx = g; } // growth
          let q = n++;                                   // insertion sort: ascending grid index
          while (q > 0 && idx[q - 1] > k) { idx[q] = idx[q - 1]; q--; }
          idx[q] = k;
        }
        out.length = 0;
        if (n <= SQ_CAP) {
          this._sqx[a] = x; this._sqz[a] = z; this._sqr[a] = r + STATIC_SLACK; this._sqn[a] = boxes.length; this._sqCnt[a] = n;
          for (let q = 0; q < n; q++) this._sqIdx[c0 + q] = idx[q];
        } else this._sqn[a] = -1;
      }
      for (let q = 0; q < n; q++) {
        const k = idx[q], bx = boxes[k];
        if (bx.maxY < minY || bx.minY > maxY || bx.maxX < minX || bx.minX > maxX || bx.maxZ < minZ || bx.minZ > maxZ) continue;
        if (mask !== -1 && (mask & layerOf(bx.kind)) === 0) continue;
        const t = this._staticTagged[k], before = C.count;
        np.collideStaticBox(B, a, t[0], t[1], t[2], t[3], t[4], t[5], k, C);   // pre-boxed bounds (see step)
        if (C.count > before) this._addManifold(before);
      }
    } else if (typeof ph.queryColliders === 'function') {
      out.length = 0;
      ph.queryColliders(x, z, r, out);                 // a grid-less world: its own query (providers included)
      this._collideRecords(a, 0, mask);
    }
    if (this.staticProviders && ph.providers && boxes) {
      out.length = 0;
      const pr = ph.providers;
      for (let i = 0; i < pr.length; i++) {
        const fn = pr[i];
        // dynamics providers publish bodies, not static geometry: marked functions, and bridge.js's by its documented name
        if (fn[DYN_RECORD] === true || fn.name === 'dynamicsProvider') continue;
        fn(x, z, r, out);
      }
      if (out.length) this._collideRecords(a, 0x40000000, mask);
    }
    out.length = 0;
    if ((mask & LAYER.GROUND) !== 0) {
      const before = C.count;
      if (ph.heightfield == null && typeof ph.groundLevel === 'number') {
        // flat ground (no heightfield): a static slab whose top is groundLevel, through the exact box paths
        let t = this._groundSlab;
        if (t[4] !== ph.groundLevel) t = this._groundSlab = [-GROUND_EXTENT, ph.groundLevel - 100, -GROUND_EXTENT, GROUND_EXTENT, ph.groundLevel, GROUND_EXTENT, null];
        np.collideStaticBox(B, a, t[0], t[1], t[2], t[3], t[4], t[5], GROUND_INDEX, C);
      } else np.collideHeightfield(B, a, this._heightAt, C);
      if (C.count > before) this._addManifold(before);
    }
  }

  // collide body a with the box records in this._statics (provider records: staticIndex = base + list index)
  _collideRecords(a, base, mask) {
    const B = this.bodies, C = this.contacts, np = this.narrowphase, out = this._statics, A = B.aabb, o6 = a * 6;
    for (let j = 0; j < out.length; j++) {
      const bx = out[j];
      if (bx[DYN_RECORD] === true) continue;          // a dynamics body's record (ours or another world's)
      if (mask !== -1 && (mask & layerOf(bx.kind)) === 0) continue;
      if (bx.maxY < A[o6 + 1] || bx.minY > A[o6 + 4] || bx.maxX < A[o6] || bx.minX > A[o6 + 3] || bx.maxZ < A[o6 + 2] || bx.minZ > A[o6 + 5]) continue;
      const before = C.count;
      np.collideStaticBox(B, a, bx.minX, bx.minY, bx.minZ, bx.maxX, bx.maxY, bx.maxZ, base + j, C);
      if (C.count > before) this._addManifold(before);
    }
  }

  // record the contacts [start, contacts.count) of one narrowphase call as manifold(s) of at most MAX_BLOCK points
  _addManifold(start) {
    const end = this.contacts.count, mf = this._mfStart;
    for (let i = start; i < end; i += MAX_BLOCK) mf[this._mfCount++] = i;
  }

  // anchors, friction and restitution of this step's contacts
  // contacts [start, count) and manifolds [mf0, _mfCount) (recollidePairs sets up only what it appended)
  _setupContacts(start = 0, mf0 = 0) {
    const B = this.bodies, C = this.contacts, P = B.pos, Q = B.quat, la = this._la, lb = this._lb;
    const fr = B.friction, re = B.restitution, mu = this._mu, e = this._e, fa = this._fa, fb = this._fb, fc = this._frictionCache;
    const noA = this._noAnchors;
    for (let i = start; i < C.count; i++) {
      const a = C.a[i], b = C.b[i], i3 = i * 3;
      const nx = C.nx[i], ny = C.ny[i], nz = C.nz[i], hd = 0.5 * C.depth[i];
      const px = C.px[i], py = C.py[i], pz = C.pz[i];
      const a3 = a * 3;
      _r[0] = px + nx * hd - P[a3]; _r[1] = py + ny * hd - P[a3 + 1]; _r[2] = pz + nz * hd - P[a3 + 2];
      invRotQ(Q, a * 4, _r, 0, la, i3);
      if (b >= 0) {
        const b3 = b * 3;
        _r[0] = px - nx * hd - P[b3]; _r[1] = py - ny * hd - P[b3 + 1]; _r[2] = pz - nz * hd - P[b3 + 2];
        invRotQ(Q, b * 4, _r, 0, lb, i3);
        mu[i] = Math.sqrt(fr[a] * fr[b]);
        e[i] = re[a] > re[b] ? re[a] : re[b];
      } else {
        lb[i3] = px - nx * hd; lb[i3 + 1] = py - ny * hd; lb[i3 + 2] = pz - nz * hd;
        mu[i] = fr[a];
        e[i] = re[a];
      }
      // static-friction anchors: kept from the last step while the manifold sticks, else the contact's own (a body
      // removed or added since the last step, i.e. a reused id, never inherits the previous owner's anchors)
      const slot = noA[a] || (b >= 0 && noA[b]) ? -1 : fc.find(C.pairKey[i], C.key[i]);
      if (slot >= 0) { const s3 = slot * 3; fa[i3] = fc.fa[s3]; fa[i3 + 1] = fc.fa[s3 + 1]; fa[i3 + 2] = fc.fa[s3 + 2]; fb[i3] = fc.fb[s3]; fb[i3 + 1] = fc.fb[s3 + 1]; fb[i3 + 2] = fc.fb[s3 + 2]; }
      else { fa[i3] = la[i3]; fa[i3 + 1] = la[i3 + 1]; fa[i3 + 2] = la[i3 + 2]; fb[i3] = lb[i3]; fb[i3 + 1] = lb[i3 + 1]; fb[i3 + 2] = lb[i3 + 2]; }
    }
    this._mfStart[this._mfCount] = C.count;
    this._mfStick.fill(0, mf0, this._mfCount);
  }

  // remember the anchors of the manifolds that stuck in their latest solve (for the next step's static friction)
  _storeFrictionAnchors() {
    const C = this.contacts, fc = this._frictionCache, fa = this._fa, fb = this._fb, mf = this._mfStart;
    fc.next();
    for (let m = 0; m < this._mfCount; m++) {
      if (!this._mfStick[m]) continue;
      for (let i = mf[m]; i < mf[m + 1]; i++) {
        const s = fc.slot(C.pairKey[i], C.key[i]);
        if (s < 0) return;
        const s3 = s * 3, i3 = i * 3;
        fc.fa[s3] = fa[i3]; fc.fa[s3 + 1] = fa[i3 + 1]; fc.fa[s3 + 2] = fa[i3 + 2];
        fc.fb[s3] = fb[i3]; fc.fb[s3 + 1] = fb[i3 + 1]; fc.fb[s3 + 2] = fb[i3 + 2];
      }
    }
  }

  // ---- contact solver -------------------------------------------------------------------------------------------
  // Only integers (contact / point / row indices) cross the function boundaries below; every vector and scalar
  // lives in the typed arrays (contact SoA, module scratch), so nothing is boxed per contact.

  // relative normal velocity (> 0 approaching) of every contact at the start of the substep
  _preSolveVelocities() {
    const C = this.contacts, vn0 = this._vn0;
    for (let i = 0; i < C.count; i++) {
      this._levers(i, 0);
      this._relVel(i, 0);
      vn0[i] = _va[0] * C.nx[i] + _va[1] * C.ny[i] + _va[2] * C.nz[i];
    }
  }

  // world levers of contact i at the current poses -> _bra / _brb (point slot c) and its separation along n
  // (penetration d > 0) -> _bd[c]
  _levers(i, c) {
    const B = this.bodies, C = this.contacts, P = B.pos, Q = B.quat, lb = this._lb;
    const a = C.a[i], b = C.b[i], i3 = i * 3, a3 = a * 3, c3 = c * 3;
    this._reseat(i, a, this._la, 1);
    if (b >= 0) this._reseat(i, b, lb, -1);
    rotQ(Q, a * 4, this._la, i3, _bra, c3);
    let pbx, pby, pbz;
    if (b >= 0) {
      rotQ(Q, b * 4, lb, i3, _brb, c3);
      const b3 = b * 3;
      pbx = P[b3] + _brb[c3]; pby = P[b3 + 1] + _brb[c3 + 1]; pbz = P[b3 + 2] + _brb[c3 + 2];
    } else {
      _brb[c3] = 0; _brb[c3 + 1] = 0; _brb[c3 + 2] = 0;
      pbx = lb[i3]; pby = lb[i3 + 1]; pbz = lb[i3 + 2];
    }
    _bd[c] = (P[a3] + _bra[c3] - pbx) * C.nx[i] + (P[a3 + 1] + _bra[c3 + 1] - pby) * C.ny[i] + (P[a3 + 2] + _bra[c3 + 2] - pbz) * C.nz[i];
  }

  // Curved shapes (sphere, capsule): their contact point is not a material point. A body-frame anchor carried along by
  // the rotation would sit r sin(theta) behind the geometric contact after rolling by theta, and every normal impulse
  // would then torque the body along its roll (0.6 -> 3.4 rad/s in 5 s for a 0.3 m sphere, 44.6 for a lying 0.05 m
  // capsule; 0.6.0 / 0.8.0 behaviour). So the anchor of contact i on body id (L = _la with sign +1: the surface toward
  // n; L = _lb with sign -1) is re-seated from the current pose before every use: axis point (the centre; a capsule's
  // segment point at the anchor's axial coordinate, clamped to +-halfHeight) + R^T n * its distance from that point
  // (a capsule's cylinder point along n without its axial part). Only a body whose TOP-LEVEL shape is a sphere or a
  // capsule is re-seated: a compound of spheres / capsules keeps the material anchors of 0.8.0.
  // The static-friction anchors (_fa / _fb) stay material points, which is right for rolling. Integers only cross.
  _reseat(i, id, L, sign) {
    const s = this.bodies.shape[id];
    if (s === null || (s.type !== SHAPE.SPHERE && s.type !== SHAPE.CAPSULE)) return;
    const C = this.contacts, i3 = i * 3;
    _sn[0] = C.nx[i]; _sn[1] = C.ny[i]; _sn[2] = C.nz[i];
    invRotQ(this.bodies.quat, id * 4, _sn, 0, _snb, 0);
    let ay = 0, nx = _snb[0], ny = _snb[1], nz = _snb[2];
    if (s.type === SHAPE.CAPSULE) {
      // a cap point re-seats along the full n (the cap is a sphere); a cylinder point along n's component perpendicular
      // to the axis, so its axial coordinate is a fixed point (re-seating along the full n would march it to a cap)
      const hh = s.halfHeight, y = L[i3 + 1];
      if (y >= hh) ay = hh;
      else if (y <= -hh) ay = -hh;
      else {
        ay = y;
        const p = Math.sqrt(nx * nx + nz * nz);
        if (!(p > 1e-9)) return;                     // n along the axis: no radial direction, keep the anchor
        nx /= p; ny = 0; nz /= p;
      }
    }
    const dx = L[i3], dy = L[i3 + 1] - ay, dz = L[i3 + 2];
    const len = sign * Math.sqrt(dx * dx + dy * dy + dz * dz);
    L[i3] = nx * len; L[i3 + 1] = ay + ny * len; L[i3 + 2] = nz * len;
  }

  // velocity of a's anchor minus b's anchor (levers of point slot c) -> _va
  _relVel(i, c) {
    const B = this.bodies, C = this.contacts, V = B.vel, W = B.angVel, a = C.a[i], b = C.b[i], c3 = c * 3, a3 = a * 3;
    const rx = _bra[c3], ry = _bra[c3 + 1], rz = _bra[c3 + 2];
    let vx = V[a3] + (W[a3 + 1] * rz - W[a3 + 2] * ry), vy = V[a3 + 1] + (W[a3 + 2] * rx - W[a3] * rz), vz = V[a3 + 2] + (W[a3] * ry - W[a3 + 1] * rx);
    if (b >= 0) {
      const b3 = b * 3, sx = _brb[c3], sy = _brb[c3 + 1], sz = _brb[c3 + 2];
      vx -= V[b3] + (W[b3 + 1] * sz - W[b3 + 2] * sy); vy -= V[b3 + 1] + (W[b3 + 2] * sx - W[b3] * sz); vz -= V[b3 + 2] + (W[b3] * sy - W[b3 + 1] * sx);
    }
    _va[0] = vx; _va[1] = vy; _va[2] = vz;
  }

  // static-friction offset of contact i: a's anchor minus b's anchor (world) -> _bdp (point slot c)
  _frictionOffset(i, c) {
    const B = this.bodies, C = this.contacts, P = B.pos, Q = B.quat, fb = this._fb;
    const a = C.a[i], b = C.b[i], i3 = i * 3, a3 = a * 3, c3 = c * 3;
    rotQ(Q, a * 4, this._fa, i3, _r, 0);
    let dx = P[a3] + _r[0], dy = P[a3 + 1] + _r[1], dz = P[a3 + 2] + _r[2];
    if (b >= 0) {
      const b3 = b * 3;
      rotQ(Q, b * 4, fb, i3, _r, 0);
      dx -= P[b3] + _r[0]; dy -= P[b3 + 1] + _r[1]; dz -= P[b3 + 2] + _r[2];
    } else { dx -= fb[i3]; dy -= fb[i3 + 1]; dz -= fb[i3 + 2]; }
    _bdp[c3] = dx; _bdp[c3 + 1] = dy; _bdp[c3 + 2] = dz;
  }

  // Block rows. Row r: linear direction _RU (on a; -_RU on b) at the levers _RLa / _RLb, or (_RU = 0) an angular row
  // about _RCa = _RCb = axis. _rowFinish computes the angular parts c = lever x u and I^-1 c of both bodies.
  _rowFinish(r, a, b) {
    const B = this.bodies, r3 = r * 3;
    const ux = _RU[r3], uy = _RU[r3 + 1], uz = _RU[r3 + 2];
    let x = _RLa[r3], y = _RLa[r3 + 1], z = _RLa[r3 + 2];
    _RCa[r3] = y * uz - z * uy; _RCa[r3 + 1] = z * ux - x * uz; _RCa[r3 + 2] = x * uy - y * ux;
    x = _RLb[r3]; y = _RLb[r3 + 1]; z = _RLb[r3 + 2];
    _RCb[r3] = y * uz - z * uy; _RCb[r3 + 1] = z * ux - x * uz; _RCb[r3 + 2] = x * uy - y * ux;
    invI(B, a, _RCa, r3, _RTa, r3);
    invI(B, b, _RCb, r3, _RTb, r3);
  }
  _rowAngularFinish(r, a, b) {
    const B = this.bodies, r3 = r * 3;
    _RU[r3] = 0; _RU[r3 + 1] = 0; _RU[r3 + 2] = 0;
    invI(B, a, _RCa, r3, _RTa, r3);
    invI(B, b, _RCb, r3, _RTb, r3);
  }
  // normal row of point slot c of contact i: u = sign * n at the point's levers
  _normalRow(c, i, sign, a, b) {
    const C = this.contacts, c3 = c * 3;
    _RU[c3] = sign * C.nx[i]; _RU[c3 + 1] = sign * C.ny[i]; _RU[c3 + 2] = sign * C.nz[i];
    _RLa[c3] = _bra[c3]; _RLa[c3 + 1] = _bra[c3 + 1]; _RLa[c3 + 2] = _bra[c3 + 2];
    _RLb[c3] = _brb[c3]; _RLb[c3 + 1] = _brb[c3 + 1]; _RLb[c3 + 2] = _brb[c3 + 2];
    this._rowFinish(c, a, b);
  }
  // K_rs = (m_a^-1 + m_b^-1) u_r . u_s + c_a,r . I_a^-1 c_a,s + c_b,r . I_b^-1 c_b,s over rows [0, nr)
  _buildK(nr, a, b) {
    const B = this.bodies, im = (isMovable(B, a) ? B.invMass[a] : 0) + (b >= 0 && isMovable(B, b) ? B.invMass[b] : 0);
    for (let r = 0; r < nr; r++) {
      const r3 = r * 3;
      for (let c = r; c < nr; c++) {
        const c3 = c * 3;
        const v = im * (_RU[r3] * _RU[c3] + _RU[r3 + 1] * _RU[c3 + 1] + _RU[r3 + 2] * _RU[c3 + 2]) +
          _RCa[r3] * _RTa[c3] + _RCa[r3 + 1] * _RTa[c3 + 1] + _RCa[r3 + 2] * _RTa[c3 + 2] +
          _RCb[r3] * _RTb[c3] + _RCb[r3 + 1] * _RTb[c3 + 1] + _RCb[r3 + 2] * _RTb[c3 + 2];
        _bK[r * MAX_ROWS + c] = v; _bK[c * MAX_ROWS + r] = v;
      }
    }
  }
  // Apply the solution _bx of the rows _bidx[0 .. m): a gets sum x u / m_a and the rotation sum x I_a^-1 c_a, b the
  // opposite; as a pose displacement (positions) or a velocity change (velocities).
  _applyRows(m, a, b, velocity) {
    const B = this.bodies;
    let lx = 0, ly = 0, lz = 0, ax = 0, ay = 0, az = 0, bx = 0, by = 0, bz = 0;
    for (let k = 0; k < m; k++) {
      const x = _bx[k];
      if (x === 0) continue;
      const r3 = _bidx[k] * 3;
      lx += x * _RU[r3]; ly += x * _RU[r3 + 1]; lz += x * _RU[r3 + 2];
      ax += x * _RTa[r3]; ay += x * _RTa[r3 + 1]; az += x * _RTa[r3 + 2];
      bx += x * _RTb[r3]; by += x * _RTb[r3 + 1]; bz += x * _RTb[r3 + 2];
    }
    const ima = isMovable(B, a) ? B.invMass[a] : 0, imb = b >= 0 && isMovable(B, b) ? B.invMass[b] : 0;
    const L = velocity ? B.vel : B.pos;
    if (ima > 0) {
      const o = a * 3;
      L[o] += ima * lx; L[o + 1] += ima * ly; L[o + 2] += ima * lz;
      if (velocity) { const w = B.angVel; w[o] += ax; w[o + 1] += ay; w[o + 2] += az; }
      else { _r[0] = ax; _r[1] = ay; _r[2] = az; rotateBy(B.quat, a * 4, _r, 0, 1); }
    }
    if (imb > 0) {
      const o = b * 3;
      L[o] -= imb * lx; L[o + 1] -= imb * ly; L[o + 2] -= imb * lz;
      if (velocity) { const w = B.angVel; w[o] -= bx; w[o + 1] -= by; w[o + 2] -= bz; }
      else { _r[0] = bx; _r[1] = by; _r[2] = bz; rotateBy(B.quat, b * 4, _r, 0, -1); }
    }
  }

  // Contact passes run manifold by manifold (the contacts one narrowphase call produced, split into chunks of at most
  // MAX_BLOCK points: one body pair, or one body and one static box / the ground). A manifold's normal rows (all its
  // points at once) and its patch static friction are solved as one block (`_blockPosition`), so a box face neither
  // tilts onto its own corners (a single Gauss-Seidel sweep over 4 corners leaves a spurious rotation that a flat
  // landing turns into a hop and a spin) nor wobbles under friction. Each iteration sweeps the manifolds forward and
  // then backward: a one-directional sweep makes the result depend on the pair order, and a tall stack with friction
  // then grows a tilt mode from round-off (10 boxes topple within 3 s); the symmetric sweep keeps it at rest.
  _solveContactPositions() {
    const nm = this._mfCount, mf = this._mfStart;
    for (let m = 0; m < nm; m++) this._blockPosition(m, mf[m], mf[m + 1]);
    for (let m = nm - 1; m >= 0; m--) this._blockPosition(m, mf[m], mf[m + 1]);
  }

  // Block position solve of manifold m = contacts [i0, i1), all rows coupled (Mueller 2020 sec. 3.5 per row):
  //   normal rows    (penetrating points) u = -n at the point's levers, rhs = d;  dLambda_n >= 0 (active set)
  //   static friction (mu > 0 and the manifold carries normal force) at the patch centre (mean of the levers): two
  //                  tangential rows u = t1, t2 whose rhs undoes the mean offset of the persistent friction anchors
  //                  (AnchorCache: kept while the manifold sticks, so the drift of earlier steps is pulled back too),
  //                  and a twist row about n (rhs: the least-squares rotation of those offsets) for patches of 2+
  //                  points; kept only while the accumulated |lambda_t| <= mu lambda_n (twist: <= mu lambda_n
  //                  r_patch), else the block is solved again without them (dynamic friction then acts in the
  //                  velocity pass).
  // K is built over all rows once; dependent rows (the 4th corner of a face: rank 3) are dropped by the pivoting.
  _blockPosition(m, i0, i1) {
    const C = this.contacts, k = i1 - i0, a = C.a[i0], b = C.b[i0], lamN = this._lamN, act = _bact;
    const nx = C.nx[i0], ny = C.ny[i0], nz = C.nz[i0], mu = this._mu[i0];
    let na = 0, lamSum = 0;
    let cax = 0, cay = 0, caz = 0, cbx = 0, cby = 0, cbz = 0, dpx = 0, dpy = 0, dpz = 0;
    for (let c = 0; c < k; c++) {
      const i = i0 + c, c3 = c * 3;
      this._levers(i, c);
      act[c] = _bd[c] > 0 ? 1 : 0;
      na += act[c];
      lamSum += lamN[i];
      this._normalRow(c, i, -1, a, b);
      _Rrhs[c] = _bd[c];
      if (mu > 0) {
        this._frictionOffset(i, c);
        cax += _bra[c3]; cay += _bra[c3 + 1]; caz += _bra[c3 + 2]; cbx += _brb[c3]; cby += _brb[c3 + 1]; cbz += _brb[c3 + 2];
        dpx += _bdp[c3]; dpy += _bdp[c3 + 1]; dpz += _bdp[c3 + 2];
      }
    }
    if (na === 0 && !(lamSum > 0)) return;
    let nr = k, rT = -1, rTw = -1, radius = 0;
    if (mu > 0) {
      const inv = 1 / k;
      cax *= inv; cay *= inv; caz *= inv; cbx *= inv; cby *= inv; cbz *= inv; dpx *= inv; dpy *= inv; dpz *= inv;
      // tangent basis of n: t1 = n x e (e the axis least aligned with n), t2 = n x t1
      let t1x, t1y, t1z;
      if ((nx < 0 ? -nx : nx) < 0.57735) { t1x = 0; t1y = nz; t1z = -ny; } else { t1x = -nz; t1y = 0; t1z = nx; }
      const tl = Math.sqrt(t1x * t1x + t1y * t1y + t1z * t1z);
      t1x /= tl; t1y /= tl; t1z /= tl;
      const t2x = ny * t1z - nz * t1y, t2y = nz * t1x - nx * t1z, t2z = nx * t1y - ny * t1x;
      rT = nr;
      for (let q = 0; q < 2; q++) {
        const r3 = nr * 3;
        if (q === 0) { _RU[r3] = t1x; _RU[r3 + 1] = t1y; _RU[r3 + 2] = t1z; } else { _RU[r3] = t2x; _RU[r3 + 1] = t2y; _RU[r3 + 2] = t2z; }
        _RLa[r3] = cax; _RLa[r3 + 1] = cay; _RLa[r3 + 2] = caz; _RLb[r3] = cbx; _RLb[r3 + 1] = cby; _RLb[r3 + 2] = cbz;
        this._rowFinish(nr, a, b);
        _Rrhs[nr] = -(dpx * _RU[r3] + dpy * _RU[r3 + 1] + dpz * _RU[r3 + 2]);
        nr++;
      }
      if (k > 1) {
        // twist about n: least-squares rotation of the anchor offsets e_c about the patch centre,
        // theta = sum (s_c x e_c) . n / sum |s_c|^2 with s_c the in-plane offset of point c from the centre
        let s2 = 0, tq = 0;
        for (let c = 0; c < k; c++) {
          const c3 = c * 3, ex = _bra[c3] - cax, ey = _bra[c3 + 1] - cay, ez = _bra[c3 + 2] - caz;
          const en = ex * nx + ey * ny + ez * nz;
          const fx = ex - en * nx, fy = ey - en * ny, fz = ez - en * nz;
          const l2 = fx * fx + fy * fy + fz * fz;
          radius += Math.sqrt(l2);
          s2 += l2;
          const ox = _bdp[c3], oy = _bdp[c3 + 1], oz = _bdp[c3 + 2];
          tq += (fy * oz - fz * oy) * nx + (fz * ox - fx * oz) * ny + (fx * oy - fy * ox) * nz;
        }
        radius /= k;
        if (radius > 1e-6) {
          rTw = nr;
          const r3 = nr * 3;
          _RCa[r3] = nx; _RCa[r3 + 1] = ny; _RCa[r3 + 2] = nz; _RCb[r3] = nx; _RCb[r3 + 1] = ny; _RCb[r3 + 2] = nz;
          this._rowAngularFinish(nr, a, b);
          _Rrhs[nr] = -tq / s2;
          nr++;
        }
      }
    }
    this._buildK(nr, a, b);
    let useF = rT >= 0;
    const T = this._mfT, m3 = m * 3;
    for (;;) {                       // each pass either ends or removes a row (a pulling point, or the friction rows)
      let mm = 0;
      for (let c = 0; c < k; c++) if (act[c]) { _bidx[mm] = c; _brhs[mm] = _Rrhs[c]; mm++; }
      if (useF) {
        _bidx[mm] = rT; _brhs[mm] = _Rrhs[rT]; mm++;
        _bidx[mm] = rT + 1; _brhs[mm] = _Rrhs[rT + 1]; mm++;
        if (rTw >= 0) { _bidx[mm] = rTw; _brhs[mm] = _Rrhs[rTw]; mm++; }
      }
      if (mm === 0) return;
      _solveBlock(mm);
      let worst = -1, wv = 0, sumN = lamSum;
      for (let r = 0; r < mm; r++) {
        if (_bidx[r] >= k) continue;
        if (_bx[r] < wv) { wv = _bx[r]; worst = _bidx[r]; }
        sumN += _bx[r];
      }
      if (worst >= 0) { act[worst] = 0; continue; }
      if (useF) {
        let r0 = 0;
        while (_bidx[r0] !== rT) r0++;
        const f1 = T[m3] + _bx[r0], f2 = T[m3 + 1] + _bx[r0 + 1], tw = rTw >= 0 ? T[m3 + 2] + _bx[r0 + 2] : 0;
        const lim = mu * sumN;
        if (f1 * f1 + f2 * f2 > lim * lim || (tw < 0 ? -tw : tw) > lim * radius) {
          useF = false;                                       // sliding: normal rows alone, from the full set again
          for (let c = 0; c < k; c++) act[c] = _bd[c] > 0 ? 1 : 0;
          continue;
        }
        T[m3] = f1; T[m3 + 1] = f2; if (rTw >= 0) T[m3 + 2] = tw;
      }
      break;
    }
    this._mfStick[m] = useF ? 1 : 0;
    let mm = 0;
    for (let c = 0; c < k; c++) if (act[c]) mm++;
    for (let r = 0; r < mm; r++) lamN[i0 + _bidx[r]] += _bx[r];
    if (useF) { this._lamT[i0] = Math.sqrt(T[m3] * T[m3] + T[m3 + 1] * T[m3 + 1]); mm += rTw >= 0 ? 3 : 2; }
    this._applyRows(mm, a, b, false);
  }

  // restitution + dynamic friction of the contacts that pushed in this substep: forward then backward over the
  // manifolds; per manifold the normal rows as one block, then Coulomb friction point by point
  _solveContactVelocities() {
    const nm = this._mfCount;
    for (let m = 0; m < nm; m++) this._velocityManifold(m);
    for (let m = nm - 1; m >= 0; m--) this._velocityManifold(m);
  }

  // Velocity pass of manifold m: the normal rows of the manifold's points as one block: the points that pushed in this
  // substep (lambda_n > 0) as equalities, K j = target - vn (the paper's restitution update: target -e vn~ after an
  // impact, vn~ > 2 |g| h, else 0), and the other points as inequalities that may only push, keeping a touching
  // point from approaching faster than its gap closes (vn <= gap / h); dependent points are dropped by the pivoting.
  // Coulomb friction (once for the patch) goes first, so the normal rows also cancel the tilt its impulse at the
  // contact plane gives the body (the real normal force shifts towards the leading edge, possibly onto a corner that
  // did not push in the position pass).
  _velocityManifold(m) {
    const i0 = this._mfStart[m], i1 = this._mfStart[m + 1];
    const C = this.contacts, k = i1 - i0, a = C.a[i0], b = C.b[i0], lamN = this._lamN, vn0 = this._vn0, e = this._e;
    const thr = this._restThreshold, ih = 1 / this._h, kind = _bact;
    let any = false;
    for (let i = i0; i < i1; i++) if (lamN[i] > 0) { any = true; break; }
    if (!any) return;
    for (let c = 0; c < k; c++) this._levers(i0 + c, c);
    this._patchFriction(m, i0, k, a, b);
    for (let c = 0; c < k; c++) {
      const i = i0 + c;
      this._normalRow(c, i, 1, a, b);
      this._relVel(i, c);
      const vn = _va[0] * C.nx[i] + _va[1] * C.ny[i] + _va[2] * C.nz[i];
      if (lamN[i] > 0) { kind[c] = 1; const v0 = vn0[i]; _Rrhs[c] = (v0 > thr ? -e[i] * v0 : 0) - vn; continue; }
      const gap = _bd[c] < 0 ? -_bd[c] * ih : 0;        // d < 0: separation
      if (vn > gap) { kind[c] = 2; _Rrhs[c] = gap - vn; } else kind[c] = 0;
    }
    this._buildK(k, a, b);
    let nr = 0;
    for (;;) {                       // each pass either ends or drops an inequality row that would pull
      nr = 0;
      for (let c = 0; c < k; c++) if (kind[c]) { _bidx[nr] = c; _brhs[nr] = _Rrhs[c]; nr++; }
      if (nr === 0) return;
      _solveBlock(nr);
      let worst = -1, wv = 0;
      for (let r = 0; r < nr; r++) if (kind[_bidx[r]] === 2 && _bx[r] > wv) { wv = _bx[r]; worst = _bidx[r]; }
      if (worst < 0) break;
      kind[worst] = 0;
    }
    for (let r = 0; r < nr; r++) this._impN[i0 + _bidx[r]] += _bx[r];
    this._applyRows(nr, a, b, true);
  }

  // Dynamic (Coulomb) friction of manifold m, once for the whole patch (mirroring the static rows): one impulse at the
  // patch centre (mean of the levers) against the tangential slip, at most mu sum(lambda_n) / h per substep, then one
  // angular impulse against the relative spin about n, at most mu sum(lambda_n) r_patch / h. Per-point friction would
  // act through the points whose lambda_n the block left non-zero (3 of a face's 4 corners) and spin a sliding box.
  _patchFriction(m, i0, k, a, b) {
    const mu = this._mu[i0], lamN = this._lamN, h = this._h;
    if (!(mu > 0)) return;
    let sumN = 0;
    for (let c = 0; c < k; c++) if (lamN[i0 + c] > 0) sumN += lamN[i0 + c];
    if (!(sumN > 0)) return;
    const B = this.bodies, C = this.contacts, V = B.vel, W = B.angVel, nx = C.nx[i0], ny = C.ny[i0], nz = C.nz[i0];
    let cax = 0, cay = 0, caz = 0, cbx = 0, cby = 0, cbz = 0;
    for (let c = 0; c < k; c++) {
      const c3 = c * 3;
      cax += _bra[c3]; cay += _bra[c3 + 1]; caz += _bra[c3 + 2]; cbx += _brb[c3]; cby += _brb[c3 + 1]; cbz += _brb[c3 + 2];
    }
    const inv = 1 / k;
    cax *= inv; cay *= inv; caz *= inv; cbx *= inv; cby *= inv; cbz *= inv;
    const im = (isMovable(B, a) ? B.invMass[a] : 0) + (b >= 0 && isMovable(B, b) ? B.invMass[b] : 0);
    const r = MAX_ROWS - 1, r3 = r * 3, a3 = a * 3;
    // relative velocity at the patch centre
    let vx = V[a3] + (W[a3 + 1] * caz - W[a3 + 2] * cay), vy = V[a3 + 1] + (W[a3 + 2] * cax - W[a3] * caz), vz = V[a3 + 2] + (W[a3] * cay - W[a3 + 1] * cax);
    if (b >= 0) {
      const b3 = b * 3;
      vx -= V[b3] + (W[b3 + 1] * cbz - W[b3 + 2] * cby); vy -= V[b3 + 1] + (W[b3 + 2] * cbx - W[b3] * cbz); vz -= V[b3 + 2] + (W[b3] * cby - W[b3 + 1] * cbx);
    }
    const vn = vx * nx + vy * ny + vz * nz;
    const tx = vx - vn * nx, ty = vy - vn * ny, tz = vz - vn * nz;
    const vt = Math.sqrt(tx * tx + ty * ty + tz * tz);
    const budget = mu * sumN / h - this._mfJt[m];
    if (vt > 1e-12 && budget > 0) {
      _RU[r3] = tx / vt; _RU[r3 + 1] = ty / vt; _RU[r3 + 2] = tz / vt;
      _RLa[r3] = cax; _RLa[r3 + 1] = cay; _RLa[r3 + 2] = caz; _RLb[r3] = cbx; _RLb[r3 + 1] = cby; _RLb[r3 + 2] = cbz;
      this._rowFinish(r, a, b);
      const wt = im + _RCa[r3] * _RTa[r3] + _RCa[r3 + 1] * _RTa[r3 + 1] + _RCa[r3 + 2] * _RTa[r3 + 2] +
        _RCb[r3] * _RTb[r3] + _RCb[r3 + 1] * _RTb[r3 + 1] + _RCb[r3 + 2] * _RTb[r3 + 2];
      if (wt > 0) {
        let jt = vt / wt;
        if (jt > budget) jt = budget;
        _bidx[0] = r; _bx[0] = -jt;
        this._applyRows(1, a, b, true);
        this._mfJt[m] += jt;
        this._impT[i0] += jt;
      }
    }
    if (k < 2) return;
    // torsion: relative spin about n, against the patch radius (mean in-plane distance of the points from the centre)
    let radius = 0;
    for (let c = 0; c < k; c++) {
      const c3 = c * 3, ex = _bra[c3] - cax, ey = _bra[c3 + 1] - cay, ez = _bra[c3 + 2] - caz;
      const en = ex * nx + ey * ny + ez * nz;
      const fx = ex - en * nx, fy = ey - en * ny, fz = ez - en * nz;
      radius += Math.sqrt(fx * fx + fy * fy + fz * fz);
    }
    radius *= inv;
    let wn = W[a3] * nx + W[a3 + 1] * ny + W[a3 + 2] * nz;
    if (b >= 0) wn -= W[b * 3] * nx + W[b * 3 + 1] * ny + W[b * 3 + 2] * nz;
    const tb = mu * sumN * radius / h - this._mfJtw[m];
    if (!(wn > 1e-12 || wn < -1e-12) || !(tb > 0)) return;
    _RCa[r3] = nx; _RCa[r3 + 1] = ny; _RCa[r3 + 2] = nz; _RCb[r3] = nx; _RCb[r3 + 1] = ny; _RCb[r3 + 2] = nz;
    this._rowAngularFinish(r, a, b);
    const ww = _RCa[r3] * _RTa[r3] + _RCa[r3 + 1] * _RTa[r3 + 1] + _RCa[r3 + 2] * _RTa[r3 + 2] +
      _RCb[r3] * _RTb[r3] + _RCb[r3 + 1] * _RTb[r3 + 1] + _RCb[r3 + 2] * _RTb[r3 + 2];
    if (!(ww > 0)) return;
    let j = (wn < 0 ? -wn : wn) / ww;
    if (j > tb) j = tb;
    _bidx[0] = r; _bx[0] = wn > 0 ? -j : j;
    this._applyRows(1, a, b, true);
    this._mfJtw[m] += j;
  }

  _damp() {
    const h = this._h, B = this.bodies, n = B.count, act = this._active, type = B.type, v = B.vel, w = B.angVel;
    const ld = B.linearDamping, ad = B.angularDamping;
    for (let id = 0; id < n; id++) {
      if (!act[id] || type[id] !== DYNAMIC) continue;
      let kl = 1 - ld[id] * h, ka = 1 - ad[id] * h;
      if (kl < 0) kl = 0;
      if (ka < 0) ka = 0;
      const o3 = id * 3;
      if (kl !== 1) { v[o3] *= kl; v[o3 + 1] *= kl; v[o3 + 2] *= kl; }
      if (ka !== 1) { w[o3] *= ka; w[o3 + 1] *= ka; w[o3 + 2] *= ka; }
    }
  }

  // began / ended pair keys from this step's contacts and the previous step's pairs
  _updateEvents() {
    const C = this.contacts, B = this.bodies, prev = this._prevPairs, cur = this._curPairs;
    cur.clear();
    let last = 0, have = false;
    for (let i = 0; i < C.count; i++) {
      const a = C.a[i], b = C.b[i];
      const key = b >= 0 ? pairKey(a, b) | 0 : a - 65536;   // int32 of pairKey(a, b) / of 0xFFFF0000 + a
      if (have && key === last) continue;
      last = key; have = true;
      cur.add(key, a, b);
    }
    // sleeping bodies produce no contacts: their pairs keep touching while every member stays inactive
    const alive = B.alive, act = this._active, flags = B.flags, type = B.type;
    for (let j = 0; j < prev.count; j++) {
      const key = prev.keys[j];
      if (cur.has(key)) continue;
      const a = prev.ida[j], b = prev.idb[j];
      if (!alive[a] || (b >= 0 && !alive[b])) continue;
      const movingA = type[a] === DYNAMIC ? (flags[a] & ASLEEP) === 0 : act[a] === 1;
      const movingB = b >= 0 && (type[b] === DYNAMIC ? (flags[b] & ASLEEP) === 0 : act[b] === 1);
      if (movingA || movingB) continue;
      cur.add(key, a, b);
    }
    const began = this.began, ended = this.ended;
    began.count = 0; ended.count = 0;
    for (let j = 0; j < cur.count; j++) if (!prev.has(cur.keys[j])) pushKey(began, cur.keys[j]);
    for (let j = 0; j < prev.count; j++) if (!cur.has(prev.keys[j])) pushKey(ended, prev.keys[j]);
    this._prevPairs = cur; this._curPairs = prev;
  }

  // ---- queries -----------------------------------------------------------------------------------------------------
  /**
   * Nearest ray hit among the bodies and (opts.statics !== false) the PhysicsWorld.
   * @param {{x,y,z}|number[]} origin @param {{x,y,z}|number[]} dir need not be unit
   * @param {number} [maxDist=1e4]
   * @param {object} [opts] mask, filter(id) (bodies); staticMask, staticFilter(box), ground, hfStep (physics);
   *   bodies / statics (default true); out (hit to reuse)
   * @returns {{distance, point, normal, body, box, kind, ground}|null}
   */
  raycast(origin, dir, maxDist = 1e4, opts = EMPTY_OPTS) {
    const o = opts || EMPTY_OPTS;
    const ox = vx(origin), oy = vy(origin), oz = vz(origin);
    let dx = vx(dir), dy = vy(dir), dz = vz(dir);
    const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (!(len > 1e-12)) return null;
    dx /= len; dy /= len; dz /= len;
    this._syncBroadphase();
    const hit = o.out || makeHit();
    let found = false, best = maxDist;
    if (o.bodies !== false && raycastBodies(this, ox, oy, oz, dx, dy, dz, maxDist, o, hit)) {
      found = true; best = hit.distance; hit.kind = this._kind[hit.body]; hit.box = null; hit.ground = false;
    }
    if (o.statics !== false && this.physics && typeof this.physics.raycast === 'function') {
      const po = this._phOpts;
      po.mask = o.staticMask ?? LAYER.ALL; po.ground = o.ground; po.hfStep = o.hfStep; po.out = this._phHit;
      po.dynamics = o.dynamics;
      this._userStaticFilter = typeof o.staticFilter === 'function' ? o.staticFilter : null;
      _ro.set(ox, oy, oz); _rd.set(dx, dy, dz);
      const ph = this.physics.raycast(_ro, _rd, best, po);
      this._userStaticFilter = null; po.out = null;
      if (ph && (!found || ph.distance < best)) {
        found = true;
        hit.distance = ph.distance; writeVec(hit.point, ph.point.x, ph.point.y, ph.point.z);
        writeVec(hit.normal, ph.normal.x, ph.normal.y, ph.normal.z);
        hit.body = -1; hit.box = ph.box; hit.kind = ph.kind; hit.ground = ph.ground;
      }
    }
    return found ? hit : null;
  }

  /**
   * Sweep `shape` from pose (pos, quat) along dir for up to maxDist against the bodies (queries.shapeCastBodies).
   * pos / quat / dir: {x,y,z(,w)} or arrays. Returns the hit ({ distance, point, normal, body, ... }) or null.
   */
  shapeCast(shape, pos, quat, dir, maxDist = 1e4, opts = EMPTY_OPTS) {
    const o = opts || EMPTY_OPTS;
    this._syncBroadphase();
    const hit = o.out || makeHit();
    const qx = quat ? vx(quat) : 0, qy = quat ? vy(quat) : 0, qz = quat ? vz(quat) : 0, qw = quat ? vw(quat) : 1;
    if (!shapeCastBodies(this, shape, vx(pos), vy(pos), vz(pos), qx, qy, qz, qw, vx(dir), vy(dir), vz(dir), maxDist, o, hit)) return null;
    hit.kind = this._kind[hit.body];
    return hit;
  }

  /** Bodies whose AABB overlaps the box (ids into outInt32); returns the count. */
  overlap(minX, minY, minZ, maxX, maxY, maxZ, outInt32, mask = -1) {
    this._syncBroadphase();
    return overlapBodies(this, minX, minY, minZ, maxX, maxY, maxZ, outInt32, mask);
  }

  /** Lowest body id whose shape contains the world point (queries.pointInBody), or -1. */
  pointInBody(x, y, z) {
    this._syncBroadphase();
    return pointInBody(this, x, y, z);
  }

  /**
   * PhysicsWorld provider contract: APPEND pooled read-only records of the alive non-static bodies whose AABB
   * footprint is within r of (x, z) (circle vs rectangle in XZ), ascending id. No allocation after warm-up.
   */
  queryColliders(x, z, r, out = []) {
    if (this._inStaticPass) return out; // never read our own bodies back as static geometry
    const B = this.bodies, n = B.count, A = B.aabb, V = B.vel, alive = B.alive, type = B.type;
    const rr = r > 0 ? r * r : 0;
    for (let id = 0; id < n; id++) {
      if (!alive[id] || type[id] === STATIC) continue;
      const o6 = id * 6;
      const minX = A[o6], minZ = A[o6 + 2], maxX = A[o6 + 3], maxZ = A[o6 + 5];
      const dx = x < minX ? minX - x : x > maxX ? x - maxX : 0;
      const dz = z < minZ ? minZ - z : z > maxZ ? z - maxZ : 0;
      if (dx * dx + dz * dz > rr) continue;
      const rec = this._records[id];
      rec.minX = minX; rec.minY = A[o6 + 1]; rec.minZ = minZ; rec.maxX = maxX; rec.maxY = A[o6 + 4]; rec.maxZ = maxZ;
      rec.vx = V[id * 3]; rec.vz = V[id * 3 + 2];
      out.push(rec);
    }
    return out;
  }

  /** Ground height from the PhysicsWorld (0 without one). */
  getGroundHeight(x, z, y) { return this.physics ? this.physics.groundHeight(x, z, y) : 0; }

  /** Surface multipliers from the PhysicsWorld (defaults without one). */
  surfaceAt(x, z, out = { mu: 1, crr: 1, tag: null }) {
    if (this.physics) return this.physics.surfaceAt(x, z, out);
    out.mu = 1; out.crr = 1; out.tag = null;
    return out;
  }

  /** Debug line segments (debug.js writeDebugLines); returns the number written. */
  debugLines(out, max, opts, colors) { return writeDebugLines(this, out, max, opts, colors); }

  // ---- serialization -----------------------------------------------------------------------------------------------
  /** Plain snapshot of the whole simulation state (typed-array copies). DynamicsWorld.from() continues it bit-identically. */
  serialize() {
    const B = this.bodies, n = B.count;
    const bodies = { count: n, free: B._free.slice(0, B._freeCount) };
    for (const k of BODY_FIELDS) { const arr = B[k], stride = arr.length / B.capacity; bodies[k] = arr.slice(0, n * stride); }
    bodies.kinPending = this._kinPending.slice(0, n);
    bodies.kinMode = this._kinMode.slice(0, n);
    bodies.noAnchors = this._noAnchors.slice(0, n);
    bodies.kind = this._kind.slice(0, n);
    bodies.userData = B.userData.slice(0, n);
    bodies.islandNext = this.islands.next.slice(0, n);
    const shapes = [], index = new Map(), shapeIndex = new Int32Array(n).fill(-1);
    for (let id = 0; id < n; id++) {
      const s = B.alive[id] ? B.shape[id] : null;
      if (!s) continue;
      let k = index.get(s);
      if (k === undefined) { k = shapes.length; shapes.push(this._desc[id] || describeShape(s)); index.set(s, k); }
      shapeIndex[id] = k;
    }
    bodies.shape = shapeIndex;
    const J = this.joints;
    const joints = typeof J.serialize === 'function' ? J.serialize() : snapshotFields(J);
    const np = this.narrowphase;
    const p = this._prevPairs;
    const out = {
      version: SERIAL_VERSION,
      desc: {
        gravity: Array.from(this.gravity), substeps: this.substeps, maxBodies: this.maxBodies, maxContacts: this.maxContacts,
        maxJoints: this.maxJoints, sleep: this.sleepParams ? { ...this.sleepParams } : false, staticMask: this.staticMask,
        margin: this.margin, seed: this.seed, positionIterations: this.positionIterations, velocityIterations: this.velocityIterations,
        broadphaseMargin: this.broadphaseMargin, staticProviders: this.staticProviders, timing: this.timing,
      },
      time: this.time, stepCount: this.stepCount,
      bodies, shapes, joints,
      narrowphase: typeof np.serialize === 'function' ? np.serialize() : null,
      friction: this._frictionCache.serialize(),
      events: { keys: p.keys.slice(0, p.count), a: p.ida.slice(0, p.count), b: p.idb.slice(0, p.count) },
    };
    if (this.particles !== null) {                 // a world without particles or plugins serializes exactly as 0.8.0
      out.desc.maxParticles = this.maxParticles;
      out.desc.particleDamping = this.particleDamping;
      out.particles = this.particles.serialize();
    }
    if (this.particles !== null || this.particleSolvers.length > 0) {
      out.solvers = this.particleSolvers.map((sv) => ({ name: sv.name, data: typeof sv.serialize === 'function' ? sv.serialize() : null }));
    }
    return out;
  }

  /**
   * Rebuild a world from serialize()'s snapshot.
   * @param {object} s
   * @param {{ physics?: object, solvers?: object[] }} [o]  the static world (not part of the snapshot); particle
   *   solver instances to register and restore by `name` (see "Particles" in the header)
   */
  static from(s, o = {}) {
    if (!s || !s.desc || !s.bodies) throw new TypeError('DynamicsWorld.from: not a serialized world');
    const w = new DynamicsWorld({ ...s.desc, physics: o.physics ?? null });
    const B = w.bodies, sb = s.bodies, n = sb.count;
    if (n > B.capacity) throw new RangeError('DynamicsWorld.from: snapshot has more bodies than maxBodies');
    for (const k of BODY_FIELDS) if (sb[k] !== undefined) B[k].set(asArray(sb[k]));
    B.count = n;
    const free = asArray(sb.free);
    B._free.set(free); B._freeCount = free.length;
    if (sb.kinPending !== undefined) w._kinPending.set(asArray(sb.kinPending));
    if (sb.kinMode !== undefined) w._kinMode.set(asArray(sb.kinMode));
    if (sb.noAnchors !== undefined) w._noAnchors.set(asArray(sb.noAnchors));
    if (sb.islandNext !== undefined) w.islands.next.set(asArray(sb.islandNext));
    const kinds = asArray(sb.kind), users = asArray(sb.userData), shapeIdx = asArray(sb.shape);
    const shapes = (s.shapes || []).map(shapeFromDescriptor);
    for (let id = 0; id < n; id++) {
      if (!B.alive[id]) continue;
      const k = shapeIdx[id];
      const shape = k >= 0 ? shapes[k] : null;
      B.shape[id] = shape;
      B.userData[id] = users[id] ?? null;
      w._kind[id] = kinds[id] ?? 'rigid';
      w._desc[id] = k >= 0 ? s.shapes[k] : null;
      if (shape) w._descCache.set(shape, s.shapes[k]);
      w._records[id] = makeRecord(id);
      w._records[id].kind = w._kind[id];
    }
    if (s.joints) { const J = w.joints; if (typeof J.restore === 'function') J.restore(s.joints, B); else restoreFields(J, s.joints); }
    if (s.narrowphase && typeof w.narrowphase.restore === 'function') w.narrowphase.restore(s.narrowphase);
    if (s.friction) w._frictionCache.restore(s.friction);
    if (s.events) {
      const keys = asArray(s.events.keys), ea = asArray(s.events.a), eb = asArray(s.events.b);
      for (let i = 0; i < keys.length; i++) w._prevPairs.add(keys[i] | 0, ea[i], eb[i]);
    }
    w.time = s.time || 0;
    w.stepCount = s.stepCount || 0;
    const given = o.solvers || [];
    if (s.particles && w.particles !== null) w.particles.restore(s.particles);
    if (s.solvers || given.length) w._restoreSolvers(s.solvers || [], given);
    return w;
  }

  // solvers of a snapshot: the built-in constraints (worlds with particles), then the given instances in the snapshot's
  // order, then the given instances that match no entry
  _restoreSolvers(entries, given) {
    const used = new Uint8Array(given.length);
    for (let e = 0; e < entries.length; e++) {
      const en = entries[e];
      if (!en) continue;
      if (e === 0 && this.constraints !== null && en.name === this.constraints.name) { if (en.data) this.constraints.restore(en.data); continue; }
      let k = -1;
      for (let q = 0; q < given.length; q++) if (!used[q] && this._newSolver(given[q]) && given[q].name === en.name) { k = q; break; }
      if (k < 0) continue;
      used[k] = 1;
      this.addParticleSolver(given[k]);
      if (typeof given[k].restore === 'function') given[k].restore(en.data ?? null);
    }
    for (let q = 0; q < given.length; q++) if (!used[q] && this._newSolver(given[q])) this.addParticleSolver(given[q]);
  }
  // a solver object not registered yet (the built-in one, repeats and non-objects are skipped)
  _newSolver(sv) { return !!sv && typeof sv === 'object' && !this.particleSolvers.includes(sv); }

  /** Release references (removes colliderProvider and a bridge.js provider of this world from the PhysicsWorld). */
  dispose() {
    if (this.physics && typeof this.physics.removeDynamicProvider === 'function') {
      this.physics.removeDynamicProvider(this.colliderProvider);
      detachFromPhysicsWorld(this, this.physics);
    }
    if (typeof this.broadphase.dispose === 'function') this.broadphase.dispose();
    if (typeof this.narrowphase.dispose === 'function') this.narrowphase.dispose();
    this.physics = null;
    this.bodies.shape.fill(null); this.bodies.userData.fill(null);
    this._records.fill(null); this._desc.fill(null); this._statics.length = 0; this._staticIdx.clear();
    this._staticTagged.length = 0; this._staticGrid = null; this._sqn.fill(-1);
    this.particleSolvers.length = this.constraints !== null ? 1 : 0; // drop the user solvers / plugins
  }
}
