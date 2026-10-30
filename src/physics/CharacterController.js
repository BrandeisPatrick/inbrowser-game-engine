// CharacterController: kinematic capsule character over the PhysicsWorld query API (pure JS, deterministic, Node-safe,
// THREE math only), optionally coupled to a DynamicsWorld (attachDynamics: pushes light bodies, rides moving ones).
//
//   const cc = new CharacterController({ radius, height, stepHeight, slopeLimit (deg), gravity, walk, run, accel, decel,
//     airControl, jumpSpeed, coyoteTime, jumpBuffer, snap, maxFall, pushVehicles });
//   cc.step(dt, moveX, moveZ, jump, sprint, world)  one fixed step (dt clamped to 1e-4 .. 1/30). (moveX, moveZ): desired
//     horizontal direction in WORLD space, length 0..1 (camera-relative already); jump: the jump BUTTON state (a press is
//     its rising edge, detected here, so a held button jumps once and never bunny-hops; passing a one-step edge works
//     too); sprint: bool; world: PhysicsWorld, legacy { getGroundHeight(x, z, y), queryColliders(x, z, r[, out]) } (it
//     may fill `out` or return its own array) or null (flat ground at y = 0).
//   cc.teleport(x, y, z) / cc.reset(x, y, z)  place the feet, zero the velocity; ground within `snap` below the feet is
//     taken on the next step without a landing event (a jump on that first step works).
//   crouch: cc.setHeight(h, world) -> true, or false (height unchanged) when the taller capsule would hit a box;
//     cc.canStand(h, world) -> bool tests it without changing anything. Shrinking always succeeds. A box over the
//     capsule centre (one that arrived there, or `height` assigned directly) never ejects it sideways: the height is
//     clamped to fit under it (auto-crouch, >= 2 radius) and hitCeiling is set; call setHeight(standing, world) each
//     step to stand back up once there is room.
//   cc.setRadius(r); cc.slopeLimit = deg (accessor; the cos / tan cache follows); other tuning fields are live.
//   cc.hold = true  frozen (zero velocity, no integration); rests on the ground when it is within `snap`, otherwise
//     stays where it is (in the air).
//   outputs (fields, reused): position (capsule BOTTOM = the feet), velocity, grounded (walkable contact), onSlope (contact
//     steeper than slopeLimit: sliding), groundNormal, groundKind ('ground' heightfield | box.kind | null in the air),
//     landedThisStep, landingSpeed (down speed at the last touchdown), jumpedThisStep, fallSpeed, speed (horizontal),
//     blocked (wall hit this step), wallNormal, steppedUp (climbed a step of >= 5 cm this step), hitCeiling,
//     airTime (s without ground contact).
//   dynamics: cc.attachDynamics(dyn, { pushMass = 120, pushForce = 600, mass = 80, ride = true, rideKinds = ['rigid'] })
//     -> cc; cc.detachDynamics(); cc.dynamics (the DynamicsWorld / in-thread DynamicsClient, or null). dyn must be the
//     world whose bodies reach `world` as collider records (bridge.js attachToPhysicsWorld, or dyn.colliderProvider).
//     Outputs: platformVelocity (Vector3: the ride body's velocity at the feet this step, 0 when none), rideBody (id or
//     -1), pushedBody (id of the body that took the largest push this step, or -1), pushImpulse (N s it took).
//     Step the character BEFORE dyn.step(dt): the feet are pre-moved with the ride body over the coming dynamics step,
//     and pushes are impulses applied at once. Without attachDynamics nothing below happens (bit-identical to 0.5.0).
//
// Dynamics coupling (attachDynamics; one way: the character is not a body, so bodies never push it):
//   records: a collider record with rec[DYN_RECORD] === true whose AABB is this world's current AABB for rec.id (so
//     another dynamics world's records, traffic and static boxes are never mistaken for it).
//   pushing: such a wall record (kind not 'vehicle', or any kind with pushVehicles) of a DYNAMIC body with
//     0 < mass <= pushMass is PUSHABLE (heavier, kinematic and static bodies, and bodies whose mass this thread cannot
//     read, i.e. a worker-mode client's, stay walls as before). On the first contact of a step the body takes
//     J = min(m v_n, pushForce dt) along the push normal (v_n: the character's velocity into it, in the body's frame)
//     at the closest point of its footprint, halfway up the vertical overlap of capsule and box (a low crate is not
//     tipped by a push at hip height), delivered as dyn.applyForce(id, J / dt along -n, point) over the coming step
//     (not one impulse: the solver bounds static friction per substep, so a one-substep impulse slid bodies whose
//     friction exceeds pushForce), and the character's velocity into it drops by J / mass. The push-out against a
//     pushable is reduced by how far the body moves away over the step so far (at its velocity after J), so the
//     character keeps walking into a light crate while it accelerates instead of stopping dead, and there is no slide
//     (velocity removal) unless the body is HELD: it kept less than half of the velocity change the last step's push
//     gave it (friction took it: steady pushing, a heavy body, more friction than pushForce, or a wall behind it). A
//     held body is followed at its pace: the velocity into it is removed relative to its velocity (as a vehicle's),
//     while it keeps taking the push. Up to 4 bodies a step take a push; the held test remembers only the one that took
//     the largest (pushedBody), so a second body pushed at once is never held (it yields, at most its motion per step).
//     `blocked` / wallNormal are set while a push-out remains (the character is faster than the body yields): pushing
//     reads as blocked, check pushedBody before treating it as a wall. A pushable whose top is within stepHeight of
//     the feet is a kerb first: a grounded character steps onto it when that gets further than pushing.
//   riding: when the support box the step ends on is such a record (any body type) and its kind is in rideKinds, the
//     character rides it: after its own motion the feet move by platformVelocity dt (the body's velocity at the feet,
//     angular part included) so they stay on it through the coming dyn.step, and at the start of the next step they are
//     re-placed exactly where the body's new pose carries them (rotation included, so speed changes and elevators do
//     not drift). While riding, `velocity` (and speed) is relative to the ride body; changing support changes frames
//     (landing on a platform subtracts its velocity, leaving it, by a jump or off an edge, adds it back, the vertical
//     part too when airborne), so a jump keeps the platform's velocity. The pre-move does not test walls (the next
//     step's push-out does). The ride ends, adding its velocity back as when leaving, when the body is gone or its id
//     now names another body (shape, type or kind changed, or the re-placed feet would jump more than
//     |platformVelocity| dt + 0.5 m). The character's weight is not applied to the bodies it stands on.
//
// Contact model (capsule: feet at position, segment from feet + r to feet + height - r, radius r):
//   ground: heightfield at the centre (feet height) + 4 samples at 0.7 r: groundNormal from finite differences with a
//     min-mod limiter per axis (the smaller one-sided slope, 0 at a kink), so slopes read exactly while kerbs, cliffs and
//     deck edges read as flat ground beside a step / wall. Non-vehicle box tops at or below the feet whose footprint the
//     capsule circle overlaps are floors (flat-bottom support). n.y >= cos(slopeLimit) is walkable; steeper ground
//     slides the character down (gravity's horizontal tangential part g n.y |n_h|, input only steers across the slope,
//     no uphill motion) and refuses jumps; walking onto it uphill from walkable ground is a wall.
//   vehicles ('vehicle' boxes) are never support: you cannot stand on a vehicle, and landing on one slides off sideways
//     (standing on / riding vehicles is reserved for Phase 5).
//   walls: boxes above the feet (+ every vehicle box): horizontal push-out by the closest point of the capsule segment
//     (rounded ends: effective radius sqrt(r^2 - dy^2)), up to 4 passes and at most r of push in all per sub-step
//     (sub-steps <= r / 2); the velocity into the wall is removed (tangential slide kept); a vehicle box responds in its own frame,
//     so its vx / vz shoves. Terrain is a wall (normal from its gradient) where the centre would rise more than
//     stepHeight, or the point r ahead more than max(stepHeight, r tan(slopeLimit)) above the feet.
//   steps: a grounded move blocked by a box whose top is <= feet + stepHeight is retried stepHeight higher (if the
//     raised capsule is free); it is kept when it gets further and lands on a support above the old feet.
//   ceilings: box bottoms above the (rounded) head stop upward motion; a box over the centre clamps the height (above).
//     Snapping: after ground contact (and no jump), a support up to `snap` below the feet keeps contact (down slopes,
//     off kerbs) instead of launching.
//   timing: in the air y += v dt - g dt^2 / 2 (exact for constant gravity: the jump apex does not depend on dt), then
//     v -= g dt, capped at maxFall; landingSpeed comes from energy (sqrt(v^2 + 2 g drop)); grounded / sliding contact
//     follows the surface (v.y = -(n.x v.x + n.z v.z) / n.y); coyote time and jump buffering as named.
// Determinism: the step path uses only + - * /, Math.sqrt, Math.ceil (sub-step count) and comparisons (no trig; slopeLimit
// is cached as cos / tan when set; the dynamics coupling adds only + - * / and comparisons: body poses are rotated with
// quaternion products); nothing in step() allocates (module scratch, reused outputs, the coupling reads the BodyStore
// arrays into typed scratch; V8 still boxes numbers passed to world queries it does not inline, as before). Instances
// share module scratch, so steps are not re-entrant.

import { Vector3 } from 'three';
import { BODY, DYN_RECORD } from './dynamics/state.js';

const DEG = Math.PI / 180;
const EPS = 1e-4;        // box tops within this of the feet are floors, not walls
const SKIN = 1e-6;       // push-out margin
const STEP_EVENT = 0.05; // minimum rise reported as steppedUp (m)
const EMPTY = [];

const num = (x) => (typeof x === 'number' && Number.isFinite(x) ? x : 0);
const opt = (x, d) => (typeof x === 'number' && x >= 0 && x < Infinity ? x : d); // non-negative finite option or default
const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);

// ---- scratch (no per-step allocation) ---------------------------------------------------------------------
// push slots of a move (attachDynamics): id, J, nx, nz, px, py, pz (contact point), bvx, bvz (body velocity after J),
// held (1: a wall in its frame)
const MAX_PUSH = 4, PS = 10;
const makeMove = () => ({ x: 0, z: 0, vx: 0, vz: 0, blocked: false, step: false, ceil: false, nx: 0, nz: 0,
  pushN: 0, pushFresh: false, push: new Float64Array(MAX_PUSH * PS) });
const _m1 = makeMove(), _m2 = makeMove();
let _supBox = null; // box that gave the last support (null: the heightfield)
// dynamics scratch: a body's state (x y z, qx qy qz qw, vx vy vz, wx wy wz) read straight from the BodyStore arrays, and
// double arguments handed to the coupling helpers (typed arrays: no number boxing across calls)
const _B = new Float64Array(13), _ARG = new Float64Array(10);
const _pv0 = new Vector3(); // ride body's velocity at the feet at the start of the step (0: none)
const _frc = { x: 0.5, y: 0.5, z: 0.5 }, _at = { x: 0.5, y: 0.5, z: 0.5 }; // push force and point (double fields)
const _xr = new Vector3(); // re-placed feet (ride)

export class CharacterController {
  /**
   * @param {object} [o]
   * @param {number} [o.radius=0.35]     capsule radius (m)
   * @param {number} [o.height=1.8]      capsule height, feet to head (m, >= 2 radius)
   * @param {number} [o.stepHeight=0.35] box tops / terrain rises up to this far above the feet are climbed
   * @param {number} [o.slopeLimit=50]   steepest walkable ground (degrees; cached as cos / tan)
   * @param {number} [o.gravity=9.81]    m/s^2
   * @param {number} [o.walk=2]          m/s
   * @param {number} [o.run=5.5]         m/s (sprint)
   * @param {number} [o.accel=20]        m/s^2 toward the input velocity on the ground
   * @param {number} [o.decel=30]        m/s^2 to rest when the input is released on the ground
   * @param {number} [o.airControl=0.3]  fraction of accel in the air / while sliding
   * @param {number} [o.jumpSpeed=5]     m/s take-off speed (apex jumpSpeed^2 / 2g)
   * @param {number} [o.coyoteTime=0.1]  s after leaving walkable ground a jump is still accepted
   * @param {number} [o.jumpBuffer=0.1]  s a jump press is remembered until it can be used
   * @param {number} [o.snap=0.3]        m below the feet the ground is followed instead of launching
   * @param {number} [o.maxFall=50]      m/s fall speed cap
   * @param {boolean} [o.pushVehicles=false] with attachDynamics: dynamics bodies of kind 'vehicle' can be pushed too
   *   (under pushMass); otherwise vehicle boxes are walls that shove the character
   */
  constructor(o = {}) {
    this.radius = Math.max(0.01, opt(o.radius, 0.35));
    this.height = Math.max(2 * this.radius, opt(o.height, 1.8));
    this.stepHeight = opt(o.stepHeight, 0.35);
    this.gravity = opt(o.gravity, 9.81);
    this.walk = opt(o.walk, 2);
    this.run = opt(o.run, 5.5);
    this.accel = opt(o.accel, 20);
    this.decel = opt(o.decel, 30);
    this.airControl = opt(o.airControl, 0.3);
    this.jumpSpeed = opt(o.jumpSpeed, 5);
    this.coyoteTime = opt(o.coyoteTime, 0.1);
    this.jumpBuffer = opt(o.jumpBuffer, 0.1);
    this.snap = opt(o.snap, 0.3);
    this.maxFall = opt(o.maxFall, 50);
    this.pushVehicles = !!o.pushVehicles;
    this.slopeLimit = opt(o.slopeLimit, 50);
    this.hold = false;                      // true: frozen in place (see step)
    this.position = new Vector3();          // capsule bottom (the feet)
    this.velocity = new Vector3();
    this.groundNormal = new Vector3(0, 1, 0);
    this.wallNormal = new Vector3();        // horizontal normal of the last wall contact this step (0 when not blocked)
    this._colOut = [];
    this._jumpHeld = false;                 // jump button state last step (edge detection)
    // dynamics coupling (attachDynamics); inert while `dynamics` is null
    this.dynamics = null;
    this.pushMass = 120;                    // kg: dynamic bodies up to this mass are pushed
    this.pushForce = 600;                   // N: the most the character applies to one body
    this.mass = 80;                         // kg: the character's mass (momentum given in a push)
    this.ride = true;                       // moving support bodies carry the character
    this.rideKinds = ['rigid'];             // body kinds that carry it
    this.platformVelocity = new Vector3();  // ride body's velocity at the feet this step (0 when none)
    this.rideBody = -1;                     // body id the character rides, -1 when none
    this.pushedBody = -1;                   // body that took the largest push this step, -1 when none
    this.pushImpulse = 0;                   // N s it took
    this._rideLocal = new Float64Array(3);  // feet in the ride body's frame at the end of the last step
    this._rideShape = null; this._rideType = -1; this._rideKind = null; // the ride body's identity (a reused id differs)
    // last step's pushed body (-1: none), the velocity change the push gave it and its velocity before the push (held
    // test): id, dvx, dvz, vx, vz (a typed array: writing numbers there never allocates)
    this._pushLast = new Float64Array(5);
    this.teleport(0, 0, 0);
  }

  /**
   * Couple the character to a dynamics world (one way): light dynamic bodies are pushed, moving support bodies carry
   * it (see the header). `dyn`: the DynamicsWorld or in-thread DynamicsClient whose bodies the step's `world` sees as
   * collider records (attachToPhysicsWorld). Options are stored as live fields.
   * @param {object} dyn
   * @param {object} [o]
   * @param {number} [o.pushMass=120]   kg: dynamic bodies up to this mass are pushed; heavier ones are walls
   * @param {number} [o.pushForce=600]  N: the most the character applies to a body (J <= pushForce dt per step)
   * @param {number} [o.mass=80]        kg: the character's mass (its velocity drops by J / mass in a push)
   * @param {boolean} [o.ride=true]     moving support bodies carry the character
   * @param {string[]} [o.rideKinds=['rigid']] body kinds that carry it
   * @returns {this}
   */
  attachDynamics(dyn, o = {}) {
    if (!dyn || !dyn.bodies || typeof dyn.applyForce !== 'function') {
      throw new TypeError('CharacterController.attachDynamics(dyn): a DynamicsWorld or DynamicsClient');
    }
    o = o || {};
    this.dynamics = dyn;
    this.pushMass = typeof o.pushMass === 'number' && o.pushMass >= 0 ? o.pushMass : 120; // Infinity: push everything
    this.pushForce = typeof o.pushForce === 'number' && o.pushForce >= 0 ? o.pushForce : 600; // Infinity: uncapped
    this.mass = Math.max(1e-3, opt(o.mass, 80));
    this.ride = o.ride !== false;
    this.rideKinds = Array.isArray(o.rideKinds) ? o.rideKinds.slice() : ['rigid'];
    this._resetDynamics();
    return this;
  }
  /** Undo attachDynamics: back to the plain kinematic controller (a ride's velocity is added back to `velocity`). */
  detachDynamics() { this._leaveRide(); this.dynamics = null; this._resetDynamics(); return this; }
  _resetDynamics() {
    this.rideBody = -1; this.platformVelocity.set(0, 0, 0);
    this.pushedBody = -1; this.pushImpulse = 0; this._pushLast[0] = -1;
  }

  /** Steepest walkable slope (degrees, 0..90); setting it updates the cos / tan the step path reads. */
  get slopeLimit() { return this._slopeLimit; }
  set slopeLimit(deg) {
    this._slopeLimit = clamp(num(deg), 0, 90);
    this._cosSlope = Math.cos(this._slopeLimit * DEG);
    this._tanSlope = this._slopeLimit < 90 ? Math.tan(this._slopeLimit * DEG) : Infinity;
  }
  /** Chainable form of `slopeLimit = deg`. */
  setSlopeLimit(deg) { this.slopeLimit = deg; return this; }
  /** Capsule radius (between steps). The height is kept >= 2 r. */
  setRadius(r) { this.radius = Math.max(0.01, opt(r, this.radius)); if (this.height < 2 * this.radius) this.height = 2 * this.radius; return this; }

  /** True when a capsule of `height` (default: the current one) fits at the current position (no box overlap). */
  canStand(height = this.height, world = null) {
    const p = this.position, r = this.radius;
    const h = Math.max(2 * r, opt(height, this.height));
    return !overlapsWall(queryColliders(world, p.x, p.z, r + this.stepHeight + 1, this._colOut), p.x, p.z, p.y, r, h);
  }
  /**
   * Capsule height, feet to head (between steps; crouch / stand). Clamped to >= 2 radius. Growing is refused (returns
   * false, height unchanged) when the taller capsule would overlap a box of `world`; returns true when applied.
   */
  setHeight(h, world = null) {
    const nh = Math.max(2 * this.radius, opt(h, this.height));
    if (nh > this.height && world && !this.canStand(nh, world)) return false;
    this.height = nh;
    return true;
  }

  /** Place the feet at (x, y, z) with zero velocity; ground within `snap` is taken on the next step (no landing). */
  teleport(x, y, z) {
    this.position.set(num(x), num(y), num(z));
    this.velocity.set(0, 0, 0);
    this.grounded = false;
    this.onSlope = false;
    this.groundNormal.set(0, 1, 0);
    this.groundKind = null;
    this.landedThisStep = false;
    this.landingSpeed = 0;
    this.jumpedThisStep = false;
    this.fallSpeed = 0;
    this.speed = 0;
    this.blocked = false;
    this.wallNormal.set(0, 0, 0);
    this.steppedUp = false;
    this.hitCeiling = false;
    this.airTime = 0;
    this._sinceGround = Infinity;   // s since walkable contact (coyote time)
    this._sinceJump = Infinity;     // s since the last unused jump press (jump buffer)
    this._fresh = true;             // first step after a teleport: settle on the ground, no landing event
    this._resetDynamics();          // a teleport leaves any ride body (at rest on the new support)
    return this;
  }
  /** Alias of teleport(). */
  reset(x, y, z) { return this.teleport(x, y, z); }

  // ------------------------------------------------------------------------
  /**
   * Advance one fixed step.
   * @param {number} dtIn     seconds (clamped to 1e-4 .. 1/30)
   * @param {number} moveX    desired horizontal direction, world X (|(moveX, moveZ)| 0..1, longer is normalised)
   * @param {number} moveZ    desired horizontal direction, world Z
   * @param {boolean} jump    jump button held (a press = rising edge; a one-step edge works too)
   * @param {boolean} sprint  run instead of walk
   * @param {object} [world]  PhysicsWorld, legacy world, or null (flat ground at y = 0)
   */
  step(dtIn, moveX, moveZ, jump, sprint, world = null) {
    const dt = clamp(num(dtIn), 1e-4, 1 / 30);
    const p = this.position, v = this.velocity, n = this.groundNormal, g = this.gravity;
    const r = this.radius > 0.01 ? this.radius : 0.01;
    this.landedThisStep = false; this.jumpedThisStep = false;
    this.blocked = false; this.steppedUp = false; this.hitCeiling = false;
    this.wallNormal.set(0, 0, 0);
    if (this.hold) return this._holdStep(r, world, jump);
    const dyn = this.dynamics, fresh = this._fresh;
    if (dyn !== null) { _ARG[9] = dt; this.pushedBody = -1; this.pushImpulse = 0; this._carry(); }

    const hs = Math.sqrt(v.x * v.x + v.z * v.z);
    const cols = queryColliders(world, p.x, p.z, r + this.stepHeight + 1 + hs * dt, this._colOut);
    if (this._fresh && this._settle(world, cols, r) && this.grounded) this._sinceGround = 0;
    let h = this._fitUnder(cols, p.x, p.z, p.y, r);
    const wasGrounded = this.grounded, wasContact = this.grounded || this.onSlope;
    const onx = n.x, ony = n.y, onz = n.z, oKind = this.groundKind;

    // ---- input ----------------------------------------------------------------
    let mx = num(moveX), mz = num(moveZ);
    const m2 = mx * mx + mz * mz;
    if (m2 > 1) { const l = Math.sqrt(m2); mx /= l; mz /= l; }
    const moving = m2 > 1e-8;
    const spd = sprint ? this.run : this.walk;
    const tx = mx * spd, tz = mz * spd;

    // ---- jump: rising edge, buffered, coyote time; never from a steep slope or with the head under a box -----
    const press = !!jump && !this._jumpHeld;
    this._jumpHeld = !!jump;
    this._sinceJump = press ? 0 : this._sinceJump + dt;
    let jumped = false;
    if (this._sinceJump <= this.jumpBuffer && !this.onSlope && !this.hitCeiling
      && (wasGrounded || this._sinceGround <= this.coyoteTime)) {
      jumped = true;
      this.jumpedThisStep = true;
      this._sinceJump = Infinity;
      this._sinceGround = Infinity;
    }

    // ---- horizontal velocity ---------------------------------------------------
    if (wasGrounded) {
      moveToward(v, tx, tz, (moving ? this.accel : this.decel) * dt);
    } else if (this.onSlope) {
      // steep: slide down (horizontal part of gravity along the surface); input only steers across the slope
      const nh = Math.sqrt(n.x * n.x + n.z * n.z);
      if (nh > 1e-9) {
        const dx = n.x / nh, dz = n.z / nh;               // downhill
        let vd = v.x * dx + v.z * dz;
        const td = tx * dx + tz * dz;
        v.x -= vd * dx; v.z -= vd * dz;
        moveToward(v, tx - td * dx, tz - td * dz, this.accel * this.airControl * dt);
        if (vd < 0) vd = 0;                               // no uphill motion on a steep slope
        vd += g * n.y * nh * dt;
        v.x += vd * dx; v.z += vd * dz;
      }
    } else if (moving) {
      moveToward(v, tx, tz, this.accel * this.airControl * dt);
    }

    // ---- vertical velocity: jump / follow the surface / gravity -------------------------------
    if (jumped) v.y = this.jumpSpeed;
    const follow = !jumped && wasContact;
    if (follow) { v.y = -(n.x * v.x + n.z * v.z) / n.y; if (v.y < -this.maxFall) v.y = -this.maxFall; }
    const vy0 = v.y;
    if (!follow) { v.y -= g * dt; if (v.y < -this.maxFall) v.y = -this.maxFall; }
    // contact: v dt; air: trapezoid (= v0 dt - g dt^2 / 2 below the cap)
    let dy = follow ? v.y * dt : (vy0 + v.y) * 0.5 * dt;

    // ---- horizontal move: walls, terrain, step up -------------------------------------
    const F = p.y, x0 = p.x, z0 = p.z;
    const g0 = groundAt(world, x0, z0, F);
    let mv = this._moveH(world, cols, F, g0, x0, z0, v.x, v.z, dt, r, h, _m1);
    if (wasGrounded && !jumped && mv.step) {
      const F2 = F + this.stepHeight;
      if (!overlapsWall(cols, x0, z0, F2, r, h)) {
        const m2r = this._moveH(world, cols, F2, g0, x0, z0, v.x, v.z, dt, r, h, _m2);
        const s2 = support(world, cols, m2r.x, m2r.z, F2, r, F2);
        const Dx = v.x, Dz = v.z;
        if (s2 > F + EPS && s2 <= F2 + EPS
          && (m2r.x - x0) * Dx + (m2r.z - z0) * Dz > (mv.x - x0) * Dx + (mv.z - z0) * Dz + 1e-12) {
          mv = m2r; p.y = s2; this.steppedUp = s2 - F >= STEP_EVENT;
        }
      }
    }
    p.x = mv.x; p.z = mv.z; v.x = mv.vx; v.z = mv.vz;
    if (dyn !== null) this._applyPushes(dyn, mv);
    if (mv.blocked) { this.blocked = true; this.wallNormal.set(mv.nx, 0, mv.nz); }
    if (mv.ceil) h = this._fitUnder(cols, p.x, p.z, p.y, r);
    if (this.hitCeiling && dy > 0) { dy = 0; v.y = 0; }

    // ---- vertical move: ceilings, ground contact, snapping ---------------------------------
    const yOld = p.y;
    let y = yOld + dy;
    if (dy > 0) {
      const r2 = r * r;
      for (let i = 0; i < cols.length; i++) {
        const b = cols[i];
        if (!b) continue;
        const d2 = rectDist2(b, p.x, p.z);
        if (d2 >= r2) continue;
        const top = h - r + Math.sqrt(r2 - d2);          // rounded head above the feet at this horizontal offset
        if (b.minY < yOld + top - EPS) continue;          // not above the head (a wall or a floor)
        const lim = b.minY - top;
        if (y > lim) { y = lim > yOld ? lim : yOld; v.y = 0; this.hitCeiling = true; }
      }
    }
    const refY = yOld > y ? yOld : y;
    const s = support(world, cols, p.x, p.z, yOld, r, refY);
    const sBox = _supBox;
    const reach = wasContact && !jumped ? this.snap : 0;
    const contact = y - s <= reach + 1e-6;
    if (contact) {
      p.y = s;
      if (sBox) { n.set(0, 1, 0); this.groundKind = sBox.kind !== undefined ? sBox.kind : 0; }
      else { hfNormal(world, p.x, p.z, s, r, n); this.groundKind = 'ground'; }
      let walkable = n.y >= this._cosSlope;
      if (!walkable && wasGrounded && !jumped && n.x * v.x + n.z * v.z < 0) {
        // walked uphill onto ground too steep to walk: it is a wall; stay on the walkable side and slide along it
        const nh = Math.sqrt(n.x * n.x + n.z * n.z), wx = n.x / nh, wz = n.z / nh;
        const vn = v.x * wx + v.z * wz;
        v.x -= vn * wx; v.z -= vn * wz;
        p.set(x0, F, z0);
        n.set(onx, ony, onz);
        this.groundKind = oKind;
        this.blocked = true; this.steppedUp = false;
        this.wallNormal.set(wx, 0, wz);
        walkable = true;
      }
      if (!wasContact && !this._fresh) {
        // touchdown speed from energy (exact for the ballistic step, independent of where in the step it landed)
        const e2 = vy0 * vy0 + 2 * g * (yOld - s);
        let ls = v.y < 0 && e2 > 0 ? Math.sqrt(e2) : 0;
        if (ls > this.maxFall) ls = this.maxFall;
        this.landedThisStep = true; this.landingSpeed = ls;
      }
      this.grounded = walkable;
      this.onSlope = !walkable;
      v.y = -(n.x * v.x + n.z * v.z) / n.y;
      if (v.y < -this.maxFall) v.y = -this.maxFall;
      this.airTime = 0;
    } else {
      p.y = y;
      this.grounded = false;
      this.onSlope = false;
      n.set(0, 1, 0);
      this.groundKind = null;
      this.airTime += dt;
    }
    if (dyn !== null) this._rideEnd(dyn, contact ? sBox : null, contact, fresh);
    this._sinceGround = this.grounded ? 0 : this._sinceGround + dt;
    this._fresh = false;
    this.speed = Math.sqrt(v.x * v.x + v.z * v.z);
    this.fallSpeed = v.y < 0 ? -v.y : 0;
    return this;
  }

  // Horizontal move from (x, z) with velocity (vx, vz) at feet height F, sub-stepped (<= r / 2 per sub-step): wall
  // push-out (<= r in all per sub-step) + slide, terrain walls. Fills and returns `out` ({ x, z, vx, vz, blocked, step: blocked by a
  // box low enough to step onto, ceil: the centre went under a box over the head, nx, nz: last wall normal; with
  // attachDynamics, push / pushN: the pushes this move would give, applied by _applyPushes once a move is chosen }).
  _moveH(world, cols, F, g0, x, z, vx, vz, dt, r, h, out) {
    out.blocked = false; out.step = false; out.ceil = false; out.nx = 0; out.nz = 0; out.pushN = 0;
    const dyn = this.dynamics;
    const L = Math.sqrt(vx * vx + vz * vz) * dt;
    let ns = L > 0.5 * r ? Math.ceil(L / (0.5 * r)) : 1;
    if (ns > 32) ns = 32;
    const sdt = dt / ns;
    const segA = F + r, segB = F + h - r, floorTop = F + EPS, stepTop = F + this.stepHeight + EPS;
    const overhead = F + this.stepHeight;
    const terrainLim = (F > g0 ? F : g0) + this.stepHeight;
    const frontRise = (this.stepHeight > r * this._tanSlope ? this.stepHeight : r * this._tanSlope) + EPS;
    const r2 = r * r;
    for (let k = 0; k < ns; k++) {
      const px = x, pz = z;
      x += vx * sdt; z += vz * sdt;
      let budget = r;                                   // total push-out per sub-step
      for (let pass = 0; pass < 4 && budget > 0; pass++) {
        let hit = false;
        for (let i = 0; i < cols.length; i++) {
          const b = cols[i];
          if (!b) continue;
          const veh = b.kind === 'vehicle';
          if (!veh && b.maxY <= floorTop) continue;      // a floor under the feet
          // closest point of the capsule segment to the box: vertical gap dy -> effective horizontal radius
          const dy = b.minY > segB ? b.minY - segB : b.maxY < segA ? segA - b.maxY : 0;
          if (dy >= r) continue;
          const re2 = r2 - dy * dy;
          const cx = x < b.minX ? b.minX : x > b.maxX ? b.maxX : x;
          const cz = z < b.minZ ? b.minZ : z > b.maxZ ? b.maxZ : z;
          let nx = x - cx, nz = z - cz;
          const d2 = nx * nx + nz * nz;
          if (d2 >= re2) continue;
          const re = Math.sqrt(re2);
          let push;
          if (d2 > 1e-18) { const d = Math.sqrt(d2); nx /= d; nz /= d; push = re - d; }
          else {
            // centre inside the footprint: a box over the head is a ceiling (the height is clamped, no sideways eject);
            // anything else is left through the nearest side
            if (b.minY > overhead) { out.ceil = true; continue; }
            let m = x - b.minX; nx = -1; nz = 0;
            let t = b.maxX - x; if (t < m) { m = t; nx = 1; }
            t = z - b.minZ; if (t < m) { m = t; nx = 0; nz = -1; }
            t = b.maxZ - z; if (t < m) { m = t; nx = 0; nz = 1; }
            push = m + re;
          }
          let pi = -1; // push slot of a pushable dynamics body (attachDynamics), -1: a wall
          if (dyn !== null && b[DYN_RECORD] === true && (!veh || this.pushVehicles)) {
            const A = _ARG;
            A[0] = F; A[1] = h; A[2] = cx; A[3] = cz; A[4] = nx; A[5] = nz; A[6] = vx; A[7] = vz; A[8] = dt;
            pi = this._pushSlot(dyn, b, out);
            if (pi >= 0) {
              const P = out.push, o = pi * PS;
              if (out.pushFresh) { const dv = P[o + 1] / this.mass; vx += dv * nx; vz += dv * nz; } // momentum given
              // the body moves away over the part of the step done so far (velocity after the push): no push-out for that
              const away = -(P[o + 7] * nx + P[o + 8] * nz);
              if (away > 0) push -= away * (k + 1) * sdt;
              if (!(push > 0)) continue;
            }
          }
          if (push > budget) push = budget;
          budget -= push;
          x += nx * (push + SKIN); z += nz * (push + SKIN);
          hit = true;
          out.blocked = true; out.nx = nx; out.nz = nz;
          if (!veh && b.maxY <= stepTop) out.step = true;
          if (pi >= 0 && out.push[pi * PS + 9] === 0) continue; // a pushable that is not held: no slide
          // slide: remove the velocity into the box, measured in the box's frame (a moving vehicle shoves; a held
          // pushable is followed at its velocity before the push)
          const own = veh || pi >= 0;
          const bvx = own ? num(b.vx) : 0, bvz = own ? num(b.vz) : 0;
          const vn = (vx - bvx) * nx + (vz - bvz) * nz;
          if (vn < 0) { vx -= vn * nx; vz -= vn * nz; }
        }
        if (!hit) break;
      }
      // terrain rising more than a step under the centre, or steeper than walkable at the capsule front (r ahead), is
      // a wall: undo this sub-step and slide along the terrain gradient there
      if (!world) continue;
      const ddx = x - px, ddz = z - pz, dl2 = ddx * ddx + ddz * ddz;
      const gc = groundAt(world, x, z, F);
      let wall = gc > terrainLim, wx = x, wz = z;
      if (!wall && dl2 > 1e-18) {
        const k2 = r / Math.sqrt(dl2);
        wx = x + ddx * k2; wz = z + ddz * k2;
        wall = groundAt(world, wx, wz, F) > (F > gc ? F : gc) + frontRise;
      }
      if (wall) {
        const e = 0.7 * r;
        let gx = groundAt(world, wx + e, wz, F) - groundAt(world, wx - e, wz, F);
        let gz = groundAt(world, wx, wz + e, F) - groundAt(world, wx, wz - e, F);
        let gl = gx * gx + gz * gz;
        if (!(gl > 1e-18)) { gx = ddx; gz = ddz; gl = dl2; }
        x = px; z = pz;
        if (gl > 1e-18) {
          gl = Math.sqrt(gl);
          const nx = -gx / gl, nz = -gz / gl;
          out.blocked = true; out.nx = nx; out.nz = nz;
          const vn = vx * nx + vz * nz;
          if (vn < 0) { vx -= vn * nx; vz -= vn * nz; }
        }
      }
    }
    out.x = x; out.z = z; out.vx = vx; out.vz = vz;
    return out;
  }

  // Boxes over the head whose footprint holds the capsule centre (bottom above feet + stepHeight, below the head) clamp
  // the height to fit under them (>= 2 r) and set hitCeiling. Returns the height to use this step.
  _fitUnder(cols, x, z, F, r) {
    let h = this.height > 2 * r ? this.height : 2 * r;
    const lo = F + this.stepHeight;
    for (let i = 0; i < cols.length; i++) {
      const b = cols[i];
      if (!b || b.minY <= lo || b.minY >= F + h || x < b.minX || x > b.maxX || z < b.minZ || z > b.maxZ) continue;
      h = b.minY - F > 2 * r ? b.minY - F : 2 * r;
      this.hitCeiling = true;
    }
    if (h < this.height) this.height = h;
    return h;
  }

  // Rest on the support when it is within `snap` below the feet (or above them); otherwise airborne in place.
  // Sets the ground outputs; returns true in contact.
  _settle(world, cols, r) {
    const p = this.position, n = this.groundNormal;
    const s = support(world, cols, p.x, p.z, p.y, r, p.y);
    if (p.y - s > this.snap + 1e-6) {
      this.grounded = false; this.onSlope = false; n.set(0, 1, 0); this.groundKind = null;
      return false;
    }
    p.y = s;
    if (_supBox) { n.set(0, 1, 0); this.groundKind = _supBox.kind !== undefined ? _supBox.kind : 0; }
    else { hfNormal(world, p.x, p.z, s, r, n); this.groundKind = 'ground'; }
    this.grounded = n.y >= this._cosSlope;
    this.onSlope = !this.grounded;
    return true;
  }

  // hold: frozen (no velocity, no integration); rests on the ground within `snap`, otherwise stays in the air.
  _holdStep(r, world, jump) {
    const p = this.position;
    this.velocity.set(0, 0, 0);
    this._resetDynamics(); // frozen in the world: no ride, no push
    this._jumpHeld = !!jump;
    if (this._settle(world, queryColliders(world, p.x, p.z, r + this.stepHeight + 1, this._colOut), r)) {
      this.airTime = 0;
      this._sinceGround = this.grounded ? 0 : Infinity;
    }
    this.speed = 0; this.fallSpeed = 0;
    this._sinceJump = Infinity;
    this._fresh = false;
    return this;
  }

  // ---- dynamics coupling (attachDynamics) ----------------------------------------------------------------------
  // Push slot of wall record b in move `out`: the existing one for its body, or a new one when b is a pushable body of
  // dyn (impulse, contact point, the body's velocity after it and the held test computed here, once per move;
  // out.pushFresh says it is new). -1: b is a wall.
  // Doubles come in _ARG: F, h, cx, cz (closest footprint point), nx, nz (push normal), vx, vz (character), dt.
  _pushSlot(dyn, b, out) {
    out.pushFresh = false;
    const A = _ARG, F = A[0], h = A[1], cx = A[2], cz = A[3], nx = A[4], nz = A[5], vx = A[6], vz = A[7], dt = A[8];
    const P = out.push, id = b.id;
    for (let i = 0; i < out.pushN; i++) if (P[i * PS] === id) return i;
    if (out.pushN >= MAX_PUSH) return -1;
    const B = dyn.bodies;
    if (!ownRecord(B, b) || B.type[id] !== BODY.DYNAMIC) return -1;
    const M = B.mass, m = M ? M[id] : NaN;
    if (!(m > 0 && m <= this.pushMass)) return -1;
    const bvx = num(b.vx), bvz = num(b.vz);
    const vn = (vx - bvx) * nx + (vz - bvz) * nz;      // < 0: closing in, in the body's frame
    let J = vn < 0 ? -vn * m : 0;
    const cap = this.pushForce * dt;
    if (J > cap) J = cap;
    // held: it kept less than half of the velocity change it was given last step (friction / a wall / its load took
    // the rest: steady pushing against friction, a pinned or heavy body), so the character moves at its pace
    let held = 0;
    const PL = this._pushLast;
    if (PL[0] === id) {
      const dx = PL[1], dz = PL[2], d2 = dx * dx + dz * dz;
      if (d2 > 1e-12 && (bvx - PL[3]) * dx + (bvz - PL[4]) * dz < 0.5 * d2) held = 1;
    }
    if (vn < 0 && !(J > 0)) held = 1; // closing in with nothing to push with (pushForce 0)
    const lo = b.minY > F ? b.minY : F, hi = b.maxY < F + h ? b.maxY : F + h; // vertical overlap of capsule and box
    const i = out.pushN++, o = i * PS;
    P[o] = id; P[o + 1] = J; P[o + 2] = nx; P[o + 3] = nz; P[o + 4] = cx; P[o + 5] = 0.5 * (lo + hi); P[o + 6] = cz;
    P[o + 7] = bvx - J / m * nx; P[o + 8] = bvz - J / m * nz; P[o + 9] = held;
    out.pushFresh = true;
    return i;
  }

  // Give the pushes of the chosen move (J along -n at the contact points, as a force over the coming step; dt in
  // _ARG[9]); outputs pushedBody / pushImpulse.
  _applyPushes(dyn, mv) {
    const P = mv.push, idt = 1 / _ARG[9];
    let best = -1, bestJ = 0;
    for (let i = 0; i < mv.pushN; i++) {
      const o = i * PS, J = P[o + 1];
      if (!(J > 0)) continue;
      _frc.x = -P[o + 2] * J * idt; _frc.y = 0; _frc.z = -P[o + 3] * J * idt; _at.x = P[o + 4]; _at.y = P[o + 5]; _at.z = P[o + 6];
      dyn.applyForce(P[o] | 0, _frc, _at); // the (id, force, point) form: no numbers boxed for the call
      if (J > bestJ) { best = o; bestJ = J; }
    }
    const PL = this._pushLast;
    if (best >= 0) {
      const id = P[best] | 0;   // ids are small integers: kept as Smis
      this.pushedBody = id; this.pushImpulse = bestJ;
      // for the held test: the velocity change the push gave it (J / m along -n) and its velocity before the push
      const dv = bestJ / dyn.bodies.mass[id];
      PL[0] = id; PL[1] = -P[best + 2] * dv; PL[2] = -P[best + 3] * dv; PL[3] = P[best + 7] - PL[1]; PL[4] = P[best + 8] - PL[2];
    } else PL[0] = -1;
  }

  // Start of a step: put the feet where the ride body's pose change since the last step carries them, and read its
  // velocity at the feet into _pv0 (0 when not riding).
  // The ride ends (velocity added back) when the id no longer names the body it was: removed, or reused by another
  // (shape object, type or kind differ), or the re-placement would move the feet more than |platformVelocity| dt +
  // 0.5 m. dt comes in _ARG[9].
  _carry() {
    const id = this.rideBody, dyn = this.dynamics;
    _pv0.set(0, 0, 0);
    if (id < 0) return;
    const B = dyn.bodies;
    if (!aliveBody(B, id) || B.shape[id] !== this._rideShape || B.type[id] !== this._rideType
      || (typeof dyn.kindOf === 'function' && dyn.kindOf(id) !== this._rideKind)) { this._leaveRide(true); return; }
    readBody(B, id);
    fromLocal(this._rideLocal, _xr);
    const p = this.position, pv = this.platformVelocity, dt = _ARG[9];
    const dx = _xr.x - p.x, dy = _xr.y - p.y, dz = _xr.z - p.z, lim = Math.sqrt(pv.x * pv.x + pv.y * pv.y + pv.z * pv.z) * dt + 0.5;
    if (dx * dx + dy * dy + dz * dz > lim * lim) { this._leaveRide(true); return; }
    p.x = _xr.x; p.y = _xr.y; p.z = _xr.z;
    pointVelocity(p, _pv0);
  }

  // End a ride outside a step's support change: the ride body's last velocity goes back into `velocity` (it was
  // relative to the body), as leaving it does. gone: the body is no longer there, so this step starts in the air (no
  // ground deceleration of the carried velocity; the vertical move finds whatever support is below).
  _leaveRide(gone = false) {
    if (this.rideBody < 0) return;
    const v = this.velocity, pv = this.platformVelocity;
    v.x += pv.x; v.y += pv.y; v.z += pv.z;
    this.rideBody = -1; pv.set(0, 0, 0);
    if (gone) { this.grounded = false; this.onSlope = false; }
  }

  // End of a step: the ride body (`rec`, the support taken this step, or null), frame change of the velocity when the
  // ride changes (it is relative to the ride body), the feet moved to where the coming dynamics step takes them (with
  // the ride body; or, having left one, with the one they had at the start of the step) and stored in the new ride
  // body's frame for the next _carry.
  // dt comes in _ARG[9].
  _rideEnd(dyn, rec, contact, fresh) {
    const p = this.position, v = this.velocity, pv = this.platformVelocity, prev = this.rideBody, dt = _ARG[9];
    const B = dyn.bodies;
    let id = -1;
    if (rec !== null && this.ride && rec[DYN_RECORD] === true && this.rideKinds.indexOf(rec.kind) >= 0 && ownRecord(B, rec)) id = rec.id;
    if (id < 0 && prev < 0) return; // no ride before or after: nothing moves (platformVelocity is already 0)
    if (id >= 0) { readBody(B, id); pointVelocity(p, pv); } else pv.set(0, 0, 0);
    if (fresh) _pv0.copy(pv);       // after a teleport the character starts at rest on its support
    v.x += _pv0.x - pv.x; v.z += _pv0.z - pv.z;
    if (!contact) v.y += _pv0.y;    // left into the air: the body's vertical velocity carries too
    // where the feet are once the coming dynamics step has moved the bodies: this step's own motion was in the frame of
    // the start-of-step ride body (+ its motion), on a support the feet go with it, in the air they keep the old ride's
    const ax = p.x + _pv0.x * dt, az = p.z + _pv0.z * dt, ay = p.y + (contact ? pv.y : _pv0.y) * dt;
    if (id >= 0) {
      // the spot on the ride body (at its current pose) that its motion over the coming step brings there
      p.x = ax - pv.x * dt; p.y = ay - pv.y * dt; p.z = az - pv.z * dt;
      toLocal(p, this._rideLocal);
      this._rideShape = B.shape[id]; this._rideType = B.type[id]; this._rideKind = rec.kind;
    }
    p.x = ax; p.y = ay; p.z = az;
    this.rideBody = id;
  }
}

// ---- helpers ------------------------------------------------------------------
// Move (v.x, v.z) toward (tx, tz) by at most d.
function moveToward(v, tx, tz, d) {
  const ex = tx - v.x, ez = tz - v.z, e2 = ex * ex + ez * ez;
  if (e2 <= d * d) { v.x = tx; v.z = tz; return; }
  const k = d / Math.sqrt(e2);
  v.x += ex * k; v.z += ez * k;
}

// Squared horizontal distance from (x, z) to box b's footprint (0 inside).
function rectDist2(b, x, z) {
  const ex = x < b.minX ? b.minX - x : x > b.maxX ? x - b.maxX : 0;
  const ez = z < b.minZ ? b.minZ - z : z > b.maxZ ? z - b.maxZ : 0;
  return ex * ex + ez * ez;
}

// Highest support under the capsule circle at (x, z) for feet height F: the heightfield at the centre, or a non-vehicle
// box top at or below the feet whose footprint the circle overlaps. Sets _supBox (null: the heightfield).
function support(world, cols, x, z, F, r, refY) {
  let s = groundAt(world, x, z, refY);
  _supBox = null;
  const ft = F + EPS, r2 = r * r;
  for (let i = 0; i < cols.length; i++) {
    const b = cols[i];
    if (!b || b.kind === 'vehicle' || b.maxY > ft || b.maxY < s) continue;
    if (rectDist2(b, x, z) >= r2) continue;
    s = b.maxY; _supBox = b;
  }
  return s;
}

// True when the capsule with feet at (x, F, z) and height h overlaps a wall box (anything not a floor under the feet).
function overlapsWall(cols, x, z, F, r, h) {
  const segA = F + r, segB = F + h - r, r2 = r * r;
  for (let i = 0; i < cols.length; i++) {
    const b = cols[i];
    if (!b || (b.kind !== 'vehicle' && b.maxY <= F + EPS)) continue;
    const dy = b.minY > segB ? b.minY - segB : b.maxY < segA ? segA - b.maxY : 0;
    if (dy >= r) continue;
    if (rectDist2(b, x, z) < r2 - dy * dy) return true;
  }
  return false;
}

// Heightfield normal at (x, z): centre + 4 samples at 0.7 r, per axis the smaller one-sided difference (min-mod; 0 when
// they disagree in sign). Exact on a plane; a step / cliff / deck edge on one side reads as the flat side (no trig).
function hfNormal(world, x, z, y, r, out) {
  if (!world) return out.set(0, 1, 0);
  const e = 0.7 * r, h0 = groundAt(world, x, z, y);
  const nx = -minmod(groundAt(world, x + e, z, y) - h0, h0 - groundAt(world, x - e, z, y)) / e;
  const nz = -minmod(groundAt(world, x, z + e, y) - h0, h0 - groundAt(world, x, z - e, y)) / e;
  const l = Math.sqrt(nx * nx + 1 + nz * nz);
  return out.set(nx / l, 1 / l, nz / l);
}

const minmod = (a, b) => (a * b <= 0 ? 0 : a > 0 ? (a < b ? a : b) : (a > b ? a : b));

// ---- dynamics helpers (attachDynamics) ----------------------------------------------------------------------------------
const aliveBody = (B, id) => id >= 0 && id < B.count && B.alive[id] === 1;
// rec is the current record of a body of this BodyStore: alive, and its footprint is the body's AABB right now (the
// bridge refreshes records from bodies.aabb, so a record of another world or a stale one does not match)
function ownRecord(B, rec) {
  const id = rec.id;
  if (!aliveBody(B, id)) return false;
  const A = B.aabb, o = id * 6;
  return A[o] === rec.minX && A[o + 3] === rec.maxX && A[o + 2] === rec.minZ && A[o + 5] === rec.maxZ;
}
// _B <- body id's pose and velocities, read from the BodyStore (or a client's snapshot mirror) arrays
function readBody(B, id) {
  const P = B.pos, Q = B.quat, V = B.vel, W = B.angVel, i3 = id * 3, i4 = id * 4;
  _B[0] = P[i3]; _B[1] = P[i3 + 1]; _B[2] = P[i3 + 2];
  _B[3] = Q[i4]; _B[4] = Q[i4 + 1]; _B[5] = Q[i4 + 2]; _B[6] = Q[i4 + 3];
  _B[7] = V[i3]; _B[8] = V[i3 + 1]; _B[9] = V[i3 + 2];
  _B[10] = W[i3]; _B[11] = W[i3 + 1]; _B[12] = W[i3 + 2];
}
// out (Vector3) = pose of _B applied to the body-frame point a[0..2] (q a + x; no trig)
function fromLocal(a, out) {
  const x = a[0], y = a[1], z = a[2], qx = _B[3], qy = _B[4], qz = _B[5], qw = _B[6];
  const tx = 2 * (qy * z - qz * y), ty = 2 * (qz * x - qx * z), tz = 2 * (qx * y - qy * x);
  out.x = x + qw * tx + qy * tz - qz * ty + _B[0];
  out.y = y + qw * ty + qz * tx - qx * tz + _B[1];
  out.z = z + qw * tz + qx * ty - qy * tx + _B[2];
}
// out[0..2] = world point p in the body frame of _B (conj(q) (p - x))
function toLocal(p, out) {
  const x = p.x - _B[0], y = p.y - _B[1], z = p.z - _B[2], qx = -_B[3], qy = -_B[4], qz = -_B[5], qw = _B[6];
  const tx = 2 * (qy * z - qz * y), ty = 2 * (qz * x - qx * z), tz = 2 * (qx * y - qy * x);
  out[0] = x + qw * tx + qy * tz - qz * ty;
  out[1] = y + qw * ty + qz * tx - qx * tz;
  out[2] = z + qw * tz + qx * ty - qy * tx;
}
// out (Vector3) = velocity of the body in _B at world point p (v + w x (p - x))
function pointVelocity(p, out) {
  const rx = p.x - _B[0], ry = p.y - _B[1], rz = p.z - _B[2], wx = _B[10], wy = _B[11], wz = _B[12];
  out.x = _B[7] + wy * rz - wz * ry; out.y = _B[8] + wz * rx - wx * rz; out.z = _B[9] + wx * ry - wy * rx;
}

// World adapters (as VehicleController): PhysicsWorld and legacy worlds share getGroundHeight / queryColliders;
// groundHeight / queryStatic are accepted for minimal custom worlds. Non-finite heights read as 0. A query may fill
// `out` (cleared first) and return nothing, or return its own array.
function groundAt(world, x, z, y) {
  if (!world) return 0;
  const h = typeof world.getGroundHeight === 'function' ? world.getGroundHeight(x, z, y)
    : typeof world.groundHeight === 'function' ? world.groundHeight(x, z, y) : 0;
  return typeof h === 'number' && Number.isFinite(h) ? h : 0;
}

function queryColliders(world, x, z, r, out) {
  if (!world) return EMPTY;
  const q = typeof world.queryColliders === 'function' ? 'queryColliders' : typeof world.queryStatic === 'function' ? 'queryStatic' : null;
  if (!q) return EMPTY;
  out.length = 0;
  const res = world[q](x, z, r, out);
  return res || out;
}
