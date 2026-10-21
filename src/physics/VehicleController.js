// VehicleController: arcade-realistic wheeled vehicle with an optional thruster flight package.
// Pure simulation (THREE math only, no scene graph, no DOM) so it runs in Node (tools/car-sim.mjs).
// Parameterised by a plain `spec` object; ROADSTER_SPEC reproduces the Roadster.
//
// Frame conventions (Three.js): Y up, vehicle forward = local -Z, right = local +X.
// `position` is the vehicle's reference point: centre of the footprint at tyre-contact height at the STATIC
// suspension load, so a vehicle at rest on flat ground has position.y === groundHeight. The body is sprung: it
// rides on four spring / damper units and rotates about its centre of gravity (cgHeight above the reference
// point), so under braking it dives, in corners it rolls, on a kerb or a slope break it compresses instead of
// snapping, and position.y moves within +-suspensionTravel of the ground. Wheels are massless and sit on their
// support surface within the travel: wheels[i].y is the contact point's local height relative to static (<0
// extended, >0 compressed), wheels[i].compression 0..travel, wheels[i].load the spring force (N).
// Heading 0 faces -Z; positive heading turns left (counter-clockwise seen from above), identical to
// Object3D.rotation.y.
//
// World: either a PhysicsWorld or any legacy object with getGroundHeight(x, z, y) / queryColliders(x, z, r)
// (groundHeight / queryStatic are accepted too). Colliders are AABBs {minX,minY,maxX,maxY,minZ,maxZ,kind};
// kind 'vehicle' boxes only shove a grounded vehicle sideways and may carry a velocity (vx, vz) that the
// contact response uses (a moving car shoves with its momentum). Optional surface hook: when the world has
// surfaceAt(x, z, out) (PhysicsWorld.setSurfaceProvider), it is asked once per wheel per step for { mu, crr }
// multipliers (default 1, 1; optional `tag`), which scale the tyre grip per axle and the rolling resistance.
//
// `hold = true` freezes the vehicle in place (zero velocity, no integration, still rests on the ground), e.g.
// for countdowns; reset() leaves it unchanged.
//
// Capabilities: spec.flight = false removes the thruster / flight package (no lift, nozzles, hover assist or tank
// use; thrusterLevel 1, thrust 0, mode always 'ground'). Flight has to be ARMED by a lift input (thrust >
// spec.flightArmThrust): a car that leaves a crest or a ramp without using its thrusters is simply airborne
// (ballistic, auto-levelled, mode stays 'ground', no hover assist); a lift input in the air arms flight at once.
// spec.airControl picks the airborne attitude model while armed: 'flight' (fly-by-wire, the Roadster), 'arcade'
// (small pitch / yaw authority, no roll from steer, auto-level), 'none' (auto-level only); a spec with
// flight: false defaults to 'arcade'.
//
// Model: bicycle-model tyres (simplified Pacejka, friction ellipse, aero downforce), per-wheel spring / damper
// suspension with anti-roll and bump stops (axle loads, load transfer, pitch and roll all emerge from it),
// speed-sensitive steering, motor force/power curve with limiter, ABS/TC caps, handbrake; flight: lift thrusters
// along body up, fly-by-wire attitude hold with auto-level, coordinated turns, hover assist, retro/forward
// nozzles, a pressure tank and a boost store; contacts: per-wheel support (ground or box tops), body-point
// surface clamp, OBB-vs-AABB SAT with sub-stepped sweeps, restitution/friction, damage, upside-down recovery.

import { Vector3, Quaternion } from 'three';
import { obbVsAabb, obbFromPose, sweepToContact, bounceVelocity } from './collision.js';

export const G = 9.81;
const RHO = 1.225;
const DEG = Math.PI / 180;

const deepFreeze = (o) => { for (const v of Object.values(o)) if (v && typeof v === 'object') deepFreeze(v); return Object.freeze(o); };

/** The Roadster: electric hypercar (tri-motor AWD) with a cold-gas thruster package. */
export const ROADSTER_SPEC = deepFreeze({
  // ---- chassis --------------------------------------------------------------------------------
  mass: 2000,
  length: 4.6, width: 2.0, height: 1.2,
  axleFront: -1.45,          // local z of front axle (CG at z = 0); 2.8 m wheelbase
  axleRear: 1.35,            // local z of rear axle
  halfTrack: 0.83,
  wheelRadius: 0.35,
  cgHeight: 0.45,            // centre of gravity above the reference point (the body rotates about it)
  yawInertia: 3800,          // kg m^2
  pitchInertia: 3600,        // kg m^2 (about the local x axis)
  rollInertia: 800,          // kg m^2 (about the local z axis)
  // contact geometry (local, metres). wheels: [FL, FR, RL, RR] contact points at the static load
  wheels: [[-0.83, 0, -1.45], [0.83, 0, -1.45], [-0.83, 0, 1.35], [0.83, 0, 1.35]],
  // body points kept above the support surface (bumper corners, shoulders, roof)
  bodyPoints: [
    [-0.97, 0.18, -2.28], [0.97, 0.18, -2.28], [-0.97, 0.18, 2.28], [0.97, 0.18, 2.28],
    [-0.97, 0.8, -2.2], [0.97, 0.8, -2.2], [-0.97, 0.85, 2.25], [0.97, 0.85, 2.25],
    [-0.6, 1.17, -0.35], [0.6, 1.17, -0.35], [-0.6, 1.17, 0.7], [0.6, 1.17, 0.7],
  ],
  // body box used against world AABBs: x +-0.98, y 0.12..1.17, z +-2.3
  obb: { centerY: 0.645, half: [0.98, 0.525, 2.3] },
  stepHeight: 0.35,          // box tops up to this far above a fully extended wheel count as support (kerbs, decks)
  normalClamp: 0.5,          // m: the ground-plane fit (tyre force plane) raises wheel supports to within this of the highest one
  colliderRadius: 6,         // collider query radius around the vehicle each step
  sweepStep: 0.4,            // max displacement per collision sub-step (anti-tunnelling)
  // ---- suspension (per wheel spring / damper; rate from the static sag, travel split by suspensionSag) -------
  suspensionTravel: 0.12,    // m total wheel travel
  suspensionSag: 0.35,       // fraction of the travel used at the static load (ride height sits here)
  suspensionDamping: 0.6,    // damping ratio of each corner
  antiRoll: 0.6,             // anti-roll bar stiffness per axle as a fraction of the spring rate
  // ---- tyres (simplified Pacejka: F = mu Fz sin(C atan(B alpha))) --------------------------------
  mu: 1.35, Bf: 14, Br: 16, C: 1.45,
  tyreLowSpeed: 2.0,         // slip-angle denominator floor (m/s)
  gripCap: 0.95,             // traction control / ABS: longitudinal force limit as a fraction of grip
  handbrakeGrip: 0.38,       // rear grip multiplier with the handbrake on
  handbrakeForce: 0.8,       // handbrake rear brake force, fraction of rear grip
  // ---- steering ---------------------------------------------------------------------------------
  maxSteer: 35 * DEG,
  steerBase: 0.10,           // speed-sensitive limit: base + gain * atan(L mu g / u^2) + slipGain * |beta|
  steerGain: 1.2,
  steerSlipGain: 0.9,
  airSteer: 0.15,            // wheel angle (fraction of input) while airborne
  // ---- powertrain -------------------------------------------------------------------------------
  motorPeakForce: 30000,     // N at the contact patch, below base speed
  motorPower: 1.0e6,         // W
  vMax: 113.5,               // m/s electronic limiter (~408 km/h, settles ~400)
  limiterRamp: 6,            // m/s over which the limiter fades drive out
  reverseMax: 14,
  reverseForce: 9000,        // N
  frontDrive: 0.3,
  // ---- brakes -----------------------------------------------------------------------------------
  brakeDecel: 12.5,          // m/s^2 demand at full pedal (ABS-limited by grip)
  // ---- aero -------------------------------------------------------------------------------------
  CdA: 0.3 * 1.9,            // Cd 0.3 x 1.9 m^2 frontal
  CdA_side: 3.4,
  CdA_top: 7.5,
  ClA: 0.7,                  // downforce coefficient * area
  downforceFront: 0.4,       // share of aero load applied at the front axle (diffuser + active rear wing)
  Crr: 0.011,
  // ---- cold-gas thrusters (SpaceX package) -------------------------------------------------------
  flight: true,              // false: no thruster / flight package at all (ground vehicle)
  flightArmThrust: 0.05,     // lift input that arms flight (hover assist, flight mode) until the next touchdown
  liftMaxG: 2.2,             // total lift along local up at full thrust
  liftEmptyG: 0.6,           // available when the tank is empty
  flareG: 1.5,               // ground-proximity reserve with an empty tank
  flareAltitude: 12,         // m: reserve available below this altitude ...
  flareSinkRate: 3,          // ... when sinking faster than this (m/s)
  rearThrustG: 1.0,          // flight forward thrust (rear nozzles)
  retroThrustG: 0.6,         // flight retro thrust (front nozzles)
  boostThrustG: 1.2,         // rear nozzles during boost on the ground
  boostPower: 1.3,           // motor power/force multiplier while boosting
  boostFlightMul: 1.3,       // forward-thrust multiplier while boosting in flight
  damagePowerLoss: 0.3,      // power/thrust lost at 100% damage
  // ---- fuel -------------------------------------------------------------------------------------
  tankFullThrustSeconds: 25,
  tankRechargeSeconds: 15,
  tankReserve: 0.08,         // tank fraction below which thrust authority fades
  boostSeconds: 4,
  boostRechargeSeconds: 12,
  boostMinLevel: 0.2,        // boost store needed to (re)start boosting
  // ---- flight control (fly-by-wire) -------------------------------------------------------------
  airControl: 'flight',      // airborne attitude while armed: 'flight' (fly-by-wire below) | 'arcade' | 'none'
  airPitchRate: 0.6, airYawRate: 0.6,            // rad/s at full stick with airControl 'arcade'
  autoLevelTau: 1.0,         // s, auto-level time constant with no stick input
  pitchRate: 1.4, rollRate: 2.0, yawRate: 1.2,   // rad/s at full stick
  levelPitchRateMax: 1.2, levelRollRateMax: 1.6, // rad/s caps while auto-levelling
  coordTurn: 0.8,            // coordinated-turn yaw from bank
  attitudeGain: 6,           // 1/s rate-loop gain
  maxAngAccel: 8,            // rad/s^2 at full authority
  authorityBase: 0.6, authorityTank: 0.4,        // control authority = base + tank * tankFrac
  // ---- hover assist (holds a gentle sink rate when airborne without lift input, flight armed) --------
  hoverAssist: true,
  hoverSinkMax: 6, hoverSinkBase: 1.2, hoverSinkPerMeter: 0.5, hoverGain: 2.5,
  // ---- landing / impacts ------------------------------------------------------------------------
  hardLandingSpeed: 8,
  hardLandingBounce: 0.3,
  landingDamageBase: 0.05, landingDamagePerMs: 0.04,
  wallRestitution: 0.2, wallFriction: 0.25,
  wallDamageSpeed: 3, wallDamagePerMs: 0.012,
  wallYawKick: 0.5,
  // ---- upside down ------------------------------------------------------------------------------
  roofFriction: 0.6,         // g of sliding friction on the roof
  autoRightDelay: 1.0,       // s before auto-righting
  autoRightTau: 0.35,
});

const GEOMETRY_DIMS = ['length', 'width', 'height'];
const WHEEL_DIMS = ['axleFront', 'axleRear', 'halfTrack'];

/**
 * Merge a partial spec over ROADSTER_SPEC. Contact geometry the caller did not give is derived from the
 * dimensions when those changed (wheels from axles/halfTrack; body points + OBB from length/width/height;
 * pitch / roll inertia from mass and dimensions).
 */
export function resolveVehicleSpec(spec = ROADSTER_SPEC) {
  if (spec === ROADSTER_SPEC) return ROADSTER_SPEC;
  const base = ROADSTER_SPEC;
  const s = { ...base, ...spec };
  const changed = (keys) => keys.some((k) => k in spec && spec[k] !== base[k]);
  if (!('wheels' in spec) && changed(WHEEL_DIMS)) {
    s.wheels = [[-s.halfTrack, 0, s.axleFront], [s.halfTrack, 0, s.axleFront], [-s.halfTrack, 0, s.axleRear], [s.halfTrack, 0, s.axleRear]];
  }
  if (changed(GEOMETRY_DIMS)) {
    const hw = s.width / 2, hl = s.length / 2, top = s.height - 0.03;
    if (!('bodyPoints' in spec)) {
      const x = hw - 0.03, z = hl - 0.02, mid = s.height * 0.68;
      s.bodyPoints = [
        [-x, 0.18, -z], [x, 0.18, -z], [-x, 0.18, z], [x, 0.18, z],
        [-x, mid, -z * 0.96], [x, mid, -z * 0.96], [-x, mid, z * 0.98], [x, mid, z * 0.98],
        [-x * 0.62, top, -z * 0.15], [x * 0.62, top, -z * 0.15], [-x * 0.62, top, z * 0.3], [x * 0.62, top, z * 0.3],
      ];
    }
    if (!('obb' in spec)) s.obb = { centerY: (0.12 + top) / 2, half: [hw - 0.02, (top - 0.12) / 2, hl] };
  }
  if (!('pitchInertia' in spec) && (changed(GEOMETRY_DIMS) || 'mass' in spec)) s.pitchInertia = s.mass * (s.length * s.length + s.height * s.height) / 12;
  if (!('rollInertia' in spec) && (changed(GEOMETRY_DIMS) || 'mass' in spec)) s.rollInertia = s.mass * (s.width * s.width + s.height * s.height) / 12;
  // a ground vehicle without its own air-control choice gets the arcade model, not the fly-by-wire
  if (spec.flight === false && !('airControl' in spec)) s.airControl = 'arcade';
  if (!Array.isArray(s.wheels) || s.wheels.length !== 4) throw new Error('VehicleController: spec.wheels must list 4 contact points [FL, FR, RL, RR]');
  return s;
}

// ---- legacy module-level exports (Roadster) -------------------------------------------------------------
const v3 = (a) => new Vector3(a[0], a[1], a[2]);
/** Roadster wheel contact points (local), [FL, FR, RL, RR]. */
export const WHEEL_LOCAL = ROADSTER_SPEC.wheels.map(v3);
/** Thruster indices (for visuals): 4 lift nozzles at the corners, 2 rear, 2 front (retro). */
export const THRUSTER = Object.freeze({ FL: 0, FR: 1, RL: 2, RR: 3, REAR_L: 4, REAR_R: 5, FRONT_L: 6, FRONT_R: 7 });

// ---- scratch objects (no per-step allocation; steps are not re-entrant) --------------------------------
const _fwd = new Vector3(), _up = new Vector3(), _right = new Vector3();
const _F = new Vector3(), _tau = new Vector3(), _t1 = new Vector3(), _t2 = new Vector3(), _t3 = new Vector3(), _t4 = new Vector3();
const _n = new Vector3(), _nAny = new Vector3(), _q1 = new Quaternion(), _q2 = new Quaternion();
const _prevPos = new Vector3(), _prevVel = new Vector3(), _cg = new Vector3();
const _ax = [new Vector3(), new Vector3(), new Vector3()];
const _ca = new Vector3();
const EMPTY = [];
const Y_AXIS = new Vector3(0, 1, 0);

const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
const num = (x) => (typeof x === 'number' && Number.isFinite(x) ? x : 0);
const mult = (x) => (typeof x === 'number' && x >= 0 && x < Infinity ? x : 1); // surface multiplier (bad -> 1)

export class VehicleController {
  /**
   * @param {object} [spec=ROADSTER_SPEC]  full or partial spec (merged over ROADSTER_SPEC)
   * @param {object} [options]  { hoverAssist } (overrides spec.hoverAssist)
   */
  constructor(spec = ROADSTER_SPEC, options = {}) {
    const S = this.spec = resolveVehicleSpec(spec || ROADSTER_SPEC);
    this.hoverAssist = (options.hoverAssist ?? S.hoverAssist) !== false;
    // local wheel contact points [FL, FR, RL, RR] at the static load, also carrying the per-wheel outputs
    this.wheels = S.wheels.map((w) => Object.assign(v3(w), { y: 0, load: 0, compression: 0, contact: true, mu: 1, crr: 1, tag: null }));
    this.hold = false;                       // true: frozen in place (countdowns); see step()
    this._surf = { mu: 1, crr: 1, tag: null };
    this.bodyPoints = S.bodyPoints.map(v3);  // local body points kept above the surface
    this._obbCY = S.obb.centerY;
    this._obbH = S.obb.half.slice();
    this._obbR = Math.hypot(...this._obbH);
    this._wheelW = this.wheels.map(() => new Vector3()); // world contact points (on the support while in contact)
    this._bodyW = this.bodyPoints.map(() => new Vector3());
    this._supports = new Float64Array(4);
    this._comp = new Float64Array(4);        // compression per wheel (m, 0..travel)
    this._compPrev = new Float64Array(4).fill(-1); // previous compression (-1: was not in contact)
    this._colOut = [];
    // suspension rates from the static corner loads and the sag (so the vehicle rests at position.y == ground)
    const a = -S.axleFront, b = S.axleRear, L = a + b;
    const travel = S.suspensionTravel > 0 ? S.suspensionTravel : 0.001;
    this._travel = travel;
    this._sag = clamp(S.suspensionSag ?? 0.35, 0.05, 0.95) * travel;
    this._kW = new Float64Array(4); this._dW = new Float64Array(4); this._static = new Float64Array(4);
    for (let i = 0; i < 4; i++) {
      const stat = Math.max(1e-6, (S.mass * G * 0.5 * (i < 2 ? b : a)) / L);
      this._static[i] = stat;
      this._kW[i] = stat / this._sag;
      this._dW[i] = 2 * (S.suspensionDamping ?? 0.6) * Math.sqrt(this._kW[i] * (stat / G));
    }
    this._kAR = [(S.antiRoll ?? 0) * (this._kW[0] + this._kW[1]) * 0.5, (S.antiRoll ?? 0) * (this._kW[2] + this._kW[3]) * 0.5];
    this.position = new Vector3();
    this.quaternion = new Quaternion();
    this.velocity = new Vector3();
    this.angularVelocity = new Vector3(); // local frame: x pitch (nose up +), y yaw (left +), z roll (left-bank +)

    // outputs
    this.thrusters = new Float32Array(8);
    this.localAccel = new Vector3();      // smoothed, body frame (x right, y up, z forward(+))
    this.groundNormal = new Vector3(0, 1, 0);
    this.wheelGaps = new Float32Array(4); // metres a wheel hangs above its support beyond full extension (0 in contact)
    this.wheelSpeed = 0;                  // surface speed of driven wheels, m/s
    this.reset(new Vector3(), 0, null);
  }

  /**
   * Place the vehicle on the ground at `position` facing `headingRad` (0 = -Z, + = left), resting on its
   * suspension at the static load with the body fitted to the local ground plane (no first-step snap on a hill).
   * Without a world, position.y is taken as the ground height.
   */
  reset(position, headingRad = 0, world = null) {
    const p = position || _t1.set(0, 0, 0);
    const gh = world ? groundAt(world, p.x, p.z, p.y) : (Number.isFinite(p.y) ? p.y : 0);
    this.position.set(p.x, gh, p.z);
    this._world = world;
    this._colliders = queryColliders(world, p.x, p.z, this.spec.colliderRadius, this._colOut) || EMPTY;
    this.velocity.set(0, 0, 0);
    this.angularVelocity.set(0, 0, 0);
    this._restOnSupports(headingRad || 0, !world);
    this.grounded = true;
    this.mode = 'ground';
    this.flightArmed = false;
    this.airTime = 0;
    this.steerAngle = 0;
    this.thrustCmd = 0;
    this.liftCmd = 0;
    this.tank = 1;
    this.boostLevel = 1;
    this.boosting = false;
    this._boostLatch = false;
    this.damage = 0;
    this.gForce = 1;
    this.wheelSlip = 0;
    this.wheelSpeed = 0;
    this.wheelSpin = 0;
    this.altitude = 0;
    this.upsideDown = false;
    this._upsideTimer = 0;
    this._righting = false;
    this.braking = false;
    this.reversing = false;
    this.impact = 0;         // largest impact speed in the last step (m/s)
    this.landingImpact = 0;  // vertical speed at last touchdown (m/s, positive)
    this.landedThisStep = false;
    this.bodyContact = false;
    this.bottomed = false;   // a wheel ran out of travel this step (bump stop)
    this.thrusters.fill(0);
    this.localAccel.set(0, 0, 0);
    this.speed = 0;
    this.forwardSpeed = 0;
    return this;
  }

  get heading() {
    _t1.set(0, 0, -1).applyQuaternion(this.quaternion);
    return Math.atan2(-_t1.x, -_t1.z);
  }

  // Rest the body on its wheel supports at heading `yaw`: the body plane is fitted to the supports under the
  // wheels (flat: no world, the supports are position.y) and the wheels sit at their static compression.
  _restOnSupports(yaw, flat) {
    const W = this.wheels, sup = this._supports, p = this.position, q = this.quaternion;
    q.setFromAxisAngle(Y_AXIS, yaw);
    for (let i = 0; i < 4; i++) {
      const w = _t2.copy(W[i]).applyQuaternion(q).add(p);
      sup[i] = flat ? p.y : this.supportAt(w.x, w.z, p.y + 0.35);
    }
    const n = this.groundNormal.copy(this._estimateNormal());
    q.setFromUnitVectors(Y_AXIS, n).multiply(_q1.setFromAxisAngle(Y_AXIS, yaw)).normalize();
    let y = 0;
    for (let i = 0; i < 4; i++) y += (sup[i] - _t2.copy(W[i]).applyQuaternion(q).y) * 0.25;
    p.y = y;
    for (let i = 0; i < 4; i++) {
      const w = W[i];
      this._comp[i] = this._compPrev[i] = this._sag;
      w.y = 0; w.compression = this._sag; w.contact = true; w.load = this._static[i];
      this._wheelW[i].copy(w).applyQuaternion(q).add(p);
      this.wheelGaps[i] = 0;
    }
  }

  /** Heading of the orientation with the ground tilt (rotation Y -> n) taken out: q = tilt(n) * yaw(heading). */
  _headingUnder(n) {
    _q2.setFromUnitVectors(Y_AXIS, n).invert().multiply(this.quaternion);
    _t1.set(0, 0, -1).applyQuaternion(_q2);
    return Math.atan2(-_t1.x, -_t1.z);
  }

  /** Thruster tank level 0..1 (always 1 without the flight package). */
  get thrusterLevel() { return this.spec.flight === false ? 1 : this.tank; }
  /** Current thrust fraction 0..1: max of the lift command and the rear nozzles (0 without the flight package). */
  get thrust() { return Math.max(this.liftCmd, this.thrusters[4], this.thrusters[5]); }

  /** Alias of step() (engine system naming). */
  update(dt, controls, world) { return this.step(dt, controls, world); }

  // ------------------------------------------------------------------------
  /**
   * Advance one fixed step.
   * @param {number} dtIn  seconds (clamped to 1e-4 .. 1/30)
   * @param {object} c     controls: throttle 0..1, brake 0..1, steer -1..1 (+right), thrust 0..1,
   *                       pitch -1..1 (+nose up), roll -1..1 (+bank right, defaults to steer), yaw -1..1 (+right),
   *                       boost, handbrake
   * @param {object} world PhysicsWorld or legacy world (or null: flat ground at y = 0)
   */
  step(dtIn, c = {}, world = null) {
    const S = this.spec;
    const dt = clamp(num(dtIn), 1e-4, 1 / 30);
    if (this.hold) return this._holdStep(dt, c, world);
    const m = S.mass;
    const p = this.position, v = this.velocity, q = this.quaternion;
    const canFly = S.flight !== false;
    const liftFull = S.liftMaxG * m * G;    // N at full lift (0 for a vehicle without lift thrusters)

    const throttle = clamp(num(c.throttle), 0, 1);
    const brake = clamp(num(c.brake), 0, 1);
    const steer = clamp(num(c.steer), -1, 1);       // +1 = right
    const thrustIn = clamp(num(c.thrust), 0, 1);
    const pitchIn = clamp(num(c.pitch), -1, 1);     // +1 = nose up
    const rollIn = clamp(num(c.roll ?? c.steer), -1, 1); // +1 = bank right
    const yawIn = clamp(num(c.yaw), -1, 1);         // +1 = yaw right
    const handbrake = !!c.handbrake;
    const boostReq = !!c.boost;

    _prevPos.copy(p); _prevVel.copy(v);
    this.impact = 0;
    this.landedThisStep = false;
    this.bottomed = false;

    const colliders = queryColliders(world, p.x, p.z, S.colliderRadius, this._colOut) || EMPTY;
    this._colliders = colliders;
    this._world = world;

    basis(q);
    // ---- wheel contacts: each wheel hangs from its attach point (top of the travel) along body down and sits
    //      on its support within the travel; beyond it the wheel is in the air
    const W = this.wheels, sup = this._supports, comp = this._comp;
    const travel = this._travel, sag = this._sag, upMax = travel - sag;
    const upY = Math.max(_up.y, 0.3);
    // tyre-radius sampling direction: the tyre rolls up a step over ~sqrt(2 R h) of travel, so the support is also
    // read a little ahead of / behind the contact point along the horizontal velocity (see _wheelSupport)
    const hs = Math.hypot(v.x, v.z);
    const sdx = hs > 0.5 ? v.x / hs : 0, sdz = hs > 0.5 ? v.z / hs : 0;
    let contacts = 0;
    for (let i = 0; i < 4; i++) {
      const w = W[i];
      const tw = _t1.set(w.x, upMax, w.z).applyQuaternion(q).add(p);       // attach point (full compression)
      const ex = tw.x - _up.x * travel, ez = tw.z - _up.z * travel, ey = tw.y - _up.y * travel; // fully extended contact
      let s = this._wheelSupport(ex, ez, ey, sdx, sdz);
      // re-read at the actual contact point (the attach point hangs along body up, not the vertical: on steep
      // ground the two differ by a few cm)
      const d0 = clamp((tw.y - s) / upY, 0, travel);
      if (Math.abs(_up.x) + Math.abs(_up.z) > 0.02) {
        const cx = tw.x - _up.x * d0, cz = tw.z - _up.z * d0;
        s = this._wheelSupport(cx, cz, ey, sdx, sdz);
      }
      sup[i] = s;
      const d = (tw.y - s) / upY;            // distance along body-down from the attach point to the support
      const cRaw = travel - d;
      const ci = clamp(cRaw, 0, travel);
      comp[i] = ci;
      const inContact = cRaw > 0;
      w.contact = inContact;
      w.compression = ci;
      w.y = ci - sag;
      this.wheelGaps[i] = inContact ? 0 : d - travel;
      this._wheelW[i].copy(_up).multiplyScalar(-clamp(d, 0, travel)).add(tw);
      if (inContact) contacts++; else this._compPrev[i] = -1;
    }
    this._querySurfaces(world);
    const upright = _up.y > 0.5;
    const gn = this.groundNormal.copy(this._estimateNormal());
    const vn = v.dot(gn);
    let touching = upright && contacts > 0;

    if (touching && !this.grounded) {
      // touchdown: the impact is the speed INTO the surface (a landing on a downslope keeps its downhill speed)
      const down = -vn;
      this.landingImpact = Math.max(0, down);
      this.landedThisStep = true;
      if (down > S.hardLandingSpeed) {
        // beyond what the suspension can take: damage, and the bump stops bounce the body (see _clampToSurface)
        this.damage = Math.min(1, this.damage + S.landingDamageBase + (down - S.hardLandingSpeed) * S.landingDamagePerMs);
        this.impact = Math.max(this.impact, down);
        touching = false;
        this.airTime = 0.11;               // stay airborne for the bounce
      }
    }
    this.grounded = touching;
    if (touching) this.airTime = 0; else this.airTime += dt;
    if (this.upsideDown) this.airTime = 0;
    // flight is armed by a lift input and disarmed by any touchdown; an unarmed airborne car is just airborne
    if (this.grounded || !canFly) this.flightArmed = false;
    if (canFly && thrustIn > (S.flightArmThrust ?? 0.05)) this.flightArmed = true;
    this.mode = canFly && this.flightArmed && this.airTime > 0.1 ? 'flight' : 'ground';
    const flying = this.mode === 'flight';

    // ---- energy stores ------------------------------------------------------
    if (!boostReq) this._boostLatch = false;
    const canBoost = this.boostLevel > (this.boosting ? 0 : S.boostMinLevel) && !this._boostLatch;
    this.boosting = boostReq && canBoost;
    if (this.boosting) {
      this.boostLevel = Math.max(0, this.boostLevel - dt / S.boostSeconds);
      if (this.boostLevel <= 0) this._boostLatch = true;
    } else {
      this.boostLevel = Math.min(1, this.boostLevel + dt / S.boostRechargeSeconds);
    }

    if (!canFly) this.tank = 1;
    const tankFrac = clamp(this.tank / S.tankReserve, 0, 1);
    const altitude = p.y - this.supportAt(p.x, p.z, p.y + 0.1);
    this.altitude = Math.max(0, altitude);
    let maxLiftG = S.liftEmptyG + (S.liftMaxG - S.liftEmptyG) * tankFrac;
    if (tankFrac < 1 && altitude < S.flareAltitude && v.y < -S.flareSinkRate) maxLiftG = Math.max(maxLiftG, S.flareG);
    const powerScale = 1 - S.damagePowerLoss * this.damage;

    // ---- lift command (fly-by-wire hover assist when airborne with flight armed) -------------
    let liftCmd = canFly ? thrustIn : 0;
    if (flying && this.hoverAssist && !this.upsideDown && liftFull > 0) {
      const vyTarget = -Math.min(S.hoverSinkMax, S.hoverSinkBase + altitude * S.hoverSinkPerMeter);
      const aReq = G + S.hoverGain * (vyTarget - v.y);
      const assist = clamp((m * aReq) / Math.max(_up.y, 0.35) / liftFull, 0, 1);
      liftCmd = Math.max(liftCmd, assist);
    }
    // physical lift force, capped by tank pressure
    let liftForce = Math.min(liftCmd * S.liftMaxG, maxLiftG) * m * G * powerScale;
    if (this.upsideDown || !(liftForce > 0)) liftForce = 0;
    this.thrustCmd = canFly ? thrustIn : 0;
    this.liftCmd = liftFull > 0 ? liftForce / liftFull : 0;

    // ---- common forces (world) and torques (local, about the CG) ---------------------------------
    _F.set(0, -m * G, 0);
    _tau.set(0, 0, 0);
    _F.addScaledVector(_up, liftForce);
    // aero in body frame
    const vf = v.dot(_fwd), vr = v.dot(_right), vu = v.dot(_up);
    const spd = v.length();
    const kq = 0.5 * RHO * spd;
    _F.addScaledVector(_fwd, -kq * S.CdA * vf);
    _F.addScaledVector(_right, -kq * S.CdA_side * vr);
    _F.addScaledVector(_up, -kq * S.CdA_top * vu);
    const downforce = 0.5 * RHO * S.ClA * vf * vf;
    _F.addScaledVector(_up, -downforce);
    // downforce split over the axles (at CG height): the share decides the pitch moment
    _tau.x += S.axleFront * downforce * S.downforceFront + S.axleRear * downforce * (1 - S.downforceFront);

    let rearThrust = 0, retroThrust = 0;
    if (this.boosting && canFly) rearThrust += S.boostThrustG * m * G;
    if (flying) {
      rearThrust += throttle * S.rearThrustG * m * G * (this.boosting ? S.boostFlightMul : 1) * this._tankFracSoft();
      retroThrust += brake * S.retroThrustG * m * G * this._tankFracSoft();
    }
    rearThrust *= powerScale; retroThrust *= powerScale;
    _F.addScaledVector(_fwd, rearThrust - retroThrust);

    // ---- suspension forces (springs + dampers + anti-roll along body up at the contact points) ----------
    if (this.grounded) this._suspension(dt);
    else for (let i = 0; i < 4; i++) { W[i].load = 0; }

    // ---- mode specific ---------------------------------------------------------
    let aa = _t3.set(0, 0, 0); // angular accel (local) for thruster visuals
    this.braking = false; this.reversing = false;
    this.wheelSlip = 0;

    if (this.grounded) {
      this._groundStep(dt, throttle, brake, steer, handbrake, powerScale, vf, vr);
    } else if (this.upsideDown) {
      this._upsideStep(dt);
    } else {
      aa = this._flightStep(dt, pitchIn, rollIn, yawIn, tankFrac, flying); // airborne attitude (auto-level only when unarmed)
      // wheels spin down in the air
      this.wheelSpeed *= Math.exp(-dt / 2.5);
      this.steerAngle += (-steer * S.airSteer - this.steerAngle) * Math.min(1, dt * 8);
    }

    // ---- integrate: translation of the CG, rotation about it ---------------------------------------
    v.addScaledVector(_F, dt / m);
    if (this.grounded) {
      const w = this.angularVelocity;
      w.x += (_tau.x / S.pitchInertia) * dt;
      w.z += (_tau.z / S.rollInertia) * dt;
      _cg.copy(_up).multiplyScalar(S.cgHeight).add(p);
      _cg.addScaledVector(v, dt);
      integrateLocal(q, w, dt);
      basis(q);
      p.copy(_cg).addScaledVector(_up, -S.cgHeight);
    } else {
      p.addScaledVector(v, dt);
    }

    // ---- never below the surface (wheels at the bump stops, body points), roof handling --------------
    this._clampToSurface(dt);

    // ---- colliders -------------------------------------------------------------
    if (colliders.length) this._collide(colliders);

    // ---- bookkeeping -----------------------------------------------------------------
    this._updateTank(dt, liftForce, rearThrust, retroThrust);
    this._thrusterVisuals(aa, rearThrust, retroThrust, flying);

    // proper acceleration (g-force) and local accel for suspension visuals
    _t1.subVectors(v, _prevVel).divideScalar(dt);
    _t2.copy(_t1); _t2.y += G;
    const gNow = _t2.length() / G;
    this.gForce += (gNow - this.gForce) * Math.min(1, dt / 0.08);
    basis(q);
    const k = Math.min(1, dt / 0.12);
    this.localAccel.x += (_t1.dot(_right) - this.localAccel.x) * k;
    this.localAccel.y += (_t1.dot(_up) - this.localAccel.y) * k;
    this.localAccel.z += (_t1.dot(_fwd) - this.localAccel.z) * k;

    this.speed = v.length();
    this.forwardSpeed = v.dot(_fwd);
    this.wheelSpin = (this.wheelSpin + (this.wheelSpeed / S.wheelRadius) * dt) % (Math.PI * 2000);
    return this;
  }

  // ---------------------------------------------------------------------------
  // Spring / damper / anti-roll force per wheel in contact, applied along body up at the contact point:
  // accumulates into _F (world) and _tau (local, about the CG) and writes wheels[i].load.
  _suspension(dt) {
    const S = this.spec, W = this.wheels, comp = this._comp, prev = this._compPrev;
    const kW = this._kW, dW = this._dW, kAR = this._kAR, cgH = S.cgHeight;
    for (let i = 0; i < 4; i++) {
      const w = W[i];
      if (!w.contact) { w.load = 0; continue; }
      const c = comp[i];
      const rate = prev[i] >= 0 ? clamp((c - prev[i]) / dt, -8, 8) : 0;
      prev[i] = c;
      const j = i ^ 1; // the other wheel of the axle
      const ar = W[j].contact ? kAR[i >> 1] * (c - comp[j]) : 0;
      const F = Math.max(0, kW[i] * c + dW[i] * rate + ar);
      w.load = F;
      _F.addScaledVector(_up, F);
      // r x (0, F, 0) with r = (x, y - cgH, z): (-z F, 0, x F)
      _tau.x -= w.z * F;
      _tau.z += w.x * F;
    }
  }

  _groundStep(dt, throttle, brake, steer, handbrake, powerScale, u, wLat) {
    const S = this.spec;
    const m = S.mass, q = this.quaternion, v = this.velocity, n = this.groundNormal;
    const a = -S.axleFront, b = S.axleRear, L = a + b;
    const Wh = this.wheels;

    // axle loads from the suspension (load transfer, downforce and slopes all arrive through the springs)
    const FzF = Wh[0].load + Wh[1].load, FzR = Wh[2].load + Wh[3].load, Fz = FzF + FzR;
    const spdNow = v.length();
    // grip per axle from the per-wheel surface multipliers (all 1 without a surface provider)
    const mu = S.mu * (Wh[0].mu + Wh[1].mu + Wh[2].mu + Wh[3].mu) * 0.25; // speed-sensitive steering limit
    const muF = S.mu * (Wh[0].mu + Wh[1].mu) * 0.5;
    const muRs = S.mu * (Wh[2].mu + Wh[3].mu) * 0.5;
    const muR = handbrake ? muRs * S.handbrakeGrip : muRs;

    // sideslip & speed-sensitive steering
    const absU = Math.abs(u);
    const beta = Math.atan2(wLat, Math.max(absU, 1));
    let dMax = S.maxSteer;
    if (absU > 1) dMax = Math.min(S.maxSteer, S.steerBase + S.steerGain * Math.atan((L * mu * G) / (absU * absU)) + Math.abs(beta) * S.steerSlipGain);
    const delta = -steer * dMax;           // positive = wheels turned left
    this.steerAngle = delta;

    const wy = this.angularVelocity.y;
    const cd = Math.cos(delta), sd = Math.sin(delta);
    // front axle, wheel frame
    const latFc = wLat - a * wy;
    const longF = u * cd - latFc * sd;
    const latF = u * sd + latFc * cd;
    // rear axle
    const latR = wLat + b * wy;
    const longR = u;
    const vLow = S.tyreLowSpeed;
    const alphaF = Math.atan2(latF, Math.max(Math.abs(longF), vLow));
    const alphaR = Math.atan2(latR, Math.max(Math.abs(longR), vLow));

    // ---- longitudinal demand -------------------------------------------------
    let drive = 0, brakeF = 0;
    const dir = u >= 0 ? 1 : -1;
    // reverse when (nearly) stopped and only the brake pedal is held
    const wantReverse = brake > 0.1 && throttle < 0.05 && u < 0.8;
    if (wantReverse && u < 0.3) {
      this.reversing = true;
      const lim = clamp((S.reverseMax + u) / 3, 0, 1);
      drive = -brake * S.reverseForce * lim * powerScale;
    } else {
      const boostMul = this.boosting ? S.boostPower : 1;
      const P = S.motorPower * boostMul * powerScale;
      const fMotor = Math.min(S.motorPeakForce * boostMul, P / Math.max(absU, 1));
      const limiter = clamp((S.vMax - u) / S.limiterRamp, 0, 1);
      drive = throttle * fMotor * limiter;
      if (u < -0.3 && throttle > 0) { drive = 0; brakeF = throttle * m * S.brakeDecel; }
      if (brake > 0) { brakeF = Math.max(brakeF, brake * m * S.brakeDecel); this.braking = true; }
    }
    // handbrake locks the rear (rear brake force)
    const hbF = handbrake ? S.handbrakeForce * muR * FzR : 0;

    // split per axle, traction control / ABS limit
    let FxF = drive * S.frontDrive, FxR = drive * (1 - S.frontDrive);
    const bF = brakeF * (Fz > 0 ? FzF / Fz : 0.6), bR = brakeF - bF;
    const stopCap = (m * absU) / dt; // brakes cannot reverse motion within a step
    FxF -= dir * Math.min(bF, stopCap * 0.5 + 1e-6);
    FxR -= dir * Math.min(bR + hbF, stopCap * 0.5 + 1e-6);
    const capF = S.gripCap * muF * FzF, capR = S.gripCap * muR * FzR;
    FxF = clamp(FxF, -capF, capF);
    FxR = clamp(FxR, -capR, capR);
    // rolling resistance (scaled by the mean surface crr multiplier)
    const crr = (Wh[0].crr + Wh[1].crr + Wh[2].crr + Wh[3].crr) * 0.25;
    const rr = Math.min(S.Crr * crr * Fz, stopCap * 0.2);
    FxR -= dir * rr * 0.5; FxF -= dir * rr * 0.5;

    // ---- lateral (combined slip via friction ellipse) --------------------------
    const availF = Math.sqrt(Math.max(0, 1 - (FxF / (muF * FzF + 1e-6)) ** 2));
    const availR = Math.sqrt(Math.max(0, 1 - (FxR / (muR * FzR + 1e-6)) ** 2));
    // low speed: fade lateral force to avoid jitter at standstill
    const fade = clamp((spdNow + 0.2) / 1.5, 0.15, 1);
    const FyF = -muF * FzF * pac(S.C, S.Bf, alphaF) * availF * fade;
    const FyR = -muR * FzR * pac(S.C, S.Br, alphaR) * availR * fade;

    // to body frame
    let FFwd = FxF * cd + FyF * sd + FxR;
    let FRight = -FxF * sd + FyF * cd + FyR;
    let torque = -a * (-FxF * sd + FyF * cd) + b * FyR;

    basis(q);
    // tyre force directions in the ground plane
    _t1.copy(_fwd).addScaledVector(n, -_fwd.dot(n)).normalize();
    _t2.copy(_right).addScaledVector(n, -_right.dot(n)).normalize();

    // parking hold: stationary with no drive request -> stay put (also on slopes): the tyres exactly cancel the
    // in-plane part of the other forces and the in-plane velocity bleeds off (the suspension keeps moving)
    if (throttle === 0 && !this.reversing && spdNow < 0.2 && Fz > 0) {
      const fn = _F.dot(n);
      _t4.copy(_F).addScaledVector(n, -fn);             // in-plane force to cancel
      FFwd = -_t4.dot(_t1); FRight = -_t4.dot(_t2); torque = 0;
      const vnP = v.dot(n);
      _t4.copy(v).addScaledVector(n, -vnP);
      v.addScaledVector(_t4, -(1 - Math.exp(-dt * 12)));
    }
    _F.addScaledVector(_t1, FFwd).addScaledVector(_t2, FRight);
    // tyre forces act at the contact patches, cgHeight below the CG: pitch from the longitudinal force (squat /
    // dive), roll from the lateral force (outward lean); the suspension resists both
    _tau.x += S.cgHeight * FFwd;
    _tau.z += S.cgHeight * FRight;

    // yaw
    const Iz = S.yawInertia;
    let wyNew = wy + (torque / Iz) * dt;
    if (spdNow < 0.3 && throttle === 0) wyNew *= Math.exp(-dt * 20);
    this.angularVelocity.y = wyNew;

    // wheel / slip outputs
    const slipLat = Math.max(Math.abs(alphaR) / 0.2, Math.abs(alphaF) / 0.24);
    const slipLong = handbrake && absU > 2 ? 0.8 : 0;
    this.wheelSlip = absU > 2 || Math.abs(wLat) > 1 ? clamp(Math.max(slipLat, slipLong) - 0.25, 0, 1) : 0;
    this.wheelSpeed = handbrake ? this.wheelSpeed * Math.exp(-dt * 10) : u;
    this._alphaF = alphaF; this._alphaR = alphaR;
  }

  _upsideStep(dt) {
    // sliding on the roof/side: strong friction, then auto-right after a delay
    const S = this.spec;
    const v = this.velocity;
    const hs = Math.hypot(v.x, v.z);
    if (hs > 1e-3) {
      const dec = Math.min(hs, S.roofFriction * G * dt);
      v.x -= (v.x / hs) * dec; v.z -= (v.z / hs) * dec;
    }
    this.angularVelocity.multiplyScalar(Math.exp(-dt * 6));
    this.wheelSpeed *= Math.exp(-dt / 1.5);
    this._upsideTimer += dt;
    if (this._upsideTimer > S.autoRightDelay) this._righting = true;
    if (this._righting) {
      const heading = this.heading;
      _q1.setFromAxisAngle(Y_AXIS, Number.isFinite(heading) ? heading : 0);
      this.quaternion.slerp(_q1, 1 - Math.exp(-dt / S.autoRightTau)).normalize();
      this.angularVelocity.set(0, 0, 0);
    } else {
      const q = this.quaternion; basis(q);
      // integrate residual tumble
      integrateLocal(q, this.angularVelocity, dt);
    }
  }

  /**
   * Airborne attitude (rate targets -> clamped angular acceleration). Unarmed (a jump): auto-level pitch and roll,
   * leave the yaw rate alone, no stick authority. Armed, by spec.airControl:
   * 'flight' fly-by-wire (stick rates, roll from steer, coordinated turns, auto-level without input),
   * 'arcade' small pitch / yaw authority (airPitchRate / airYawRate), roll always auto-levels,
   * 'none'   no attitude control: pitch / roll auto-level, the yaw rate is left alone.
   */
  _flightStep(dt, pitchIn, rollIn, yawIn, tankFrac, armed = true) {
    const S = this.spec;
    const q = this.quaternion, w = this.angularVelocity, v = this.velocity;
    basis(q);
    const theta = Math.asin(clamp(_fwd.y, -1, 1));       // nose up +
    const phi = Math.atan2(-_right.y, _up.y);             // right bank +
    const tau = S.autoLevelTau;
    const levelX = clamp(-theta / tau, -S.levelPitchRateMax, S.levelPitchRateMax);
    const levelZ = clamp(phi / tau, -S.levelRollRateMax, S.levelRollRateMax);
    const ac = armed ? S.airControl : 'none';
    let wxD, wyD, wzD;
    if (ac === 'arcade') {
      wxD = Math.abs(pitchIn) > 0.02 ? pitchIn * S.airPitchRate : levelX;
      wyD = -yawIn * S.airYawRate;
      wzD = levelZ;
    } else if (ac === 'none') {
      wxD = levelX; wyD = w.y * Math.exp(-dt * 0.5); wzD = levelZ;
    } else {
      wxD = Math.abs(pitchIn) > 0.02 ? pitchIn * S.pitchRate : levelX;
      wzD = Math.abs(rollIn) > 0.02 ? -rollIn * S.rollRate : levelZ;
      const hs = Math.max(Math.hypot(v.x, v.z), 12);
      const coord = -(G * Math.tan(clamp(phi, -1.2, 1.2)) / hs) * S.coordTurn * clamp(_up.y, 0, 1);
      wyD = -yawIn * S.yawRate + coord;
    }
    const auth = (armed ? 1 : 0.6) * (S.authorityBase + S.authorityTank * tankFrac);
    const K = S.attitudeGain, maxAA = S.maxAngAccel * auth;
    const aa = _t3.set(
      clamp(K * (wxD - w.x), -maxAA, maxAA),
      clamp(K * (wyD - w.y), -maxAA, maxAA),
      clamp(K * (wzD - w.z), -maxAA, maxAA),
    );
    w.addScaledVector(aa, dt);
    integrateLocal(q, w, dt);
    return aa;
  }

  // Wheels at their bump stops (travel exhausted) and body points never go below the support surface: push the
  // body up, stop the motion into the surface (a hard touchdown bounces), handle roof / side contact.
  _clampToSurface(dt) {
    const S = this.spec;
    const p = this.position, v = this.velocity, q = this.quaternion;
    const W = this.wheels, B = this.bodyPoints, nW = W.length;
    let pen = 0, wheelPen = 0;
    let upsideTouch = false;
    basis(q);
    const upY = Math.max(_up.y, 0.3), travel = this._travel, upMax = travel - this._sag;
    for (let i = 0; i < nW; i++) {
      // the wheel at full compression (attach point) must stay on or above its support; _wheelW ends on the
      // support within the travel (effects, surfaces)
      const w = W[i];
      const tw = this._wheelW[i].set(w.x, upMax, w.z).applyQuaternion(q).add(p);
      const s = this.supportAt(tw.x - _up.x * travel, tw.z - _up.z * travel, tw.y - _up.y * travel);
      if (s - tw.y > wheelPen) wheelPen = s - tw.y;
      tw.addScaledVector(_up, -clamp((tw.y - s) / upY, 0, travel));
    }
    pen = wheelPen;
    // body points: a surface counts as a floor for the body only when the wheels could step onto it (reference at
    // wheel level + step height); anything higher is an obstacle, not a floor (a low deck used to lift the car
    // 0.5-0.8 m by its bumper corner)
    const bodyRef = p.y + this._travel;
    for (let i = 0; i < B.length; i++) {
      const wp = this._bodyW[i].copy(B[i]).applyQuaternion(q).add(p);
      const s = this.supportAt(wp.x, wp.z, bodyRef);
      const d = s - wp.y;
      if (d > pen) pen = d;
      if (d > -0.03) upsideTouch = true;
    }
    this.bodyContact = upsideTouch;
    if (wheelPen > 0 && _up.y > 0.5) this.bottomed = true;
    if (pen > 0) {
      p.y += pen;
      for (let i = 0; i < nW; i++) this._wheelW[i].y += pen; // the contact points move with the body
      // remove only the velocity component INTO the surface (keeps the downhill / along-slope speed)
      const n = this.groundNormal;
      const vn = v.dot(n);
      if (vn < 0) {
        const down = -vn;
        if (!this.grounded && !this.landedThisStep) { // body-first touchdown (step() has not seen a wheel contact)
          this.landingImpact = down;
          this.landedThisStep = true;
          if (down > S.hardLandingSpeed) {
            this.damage = Math.min(1, this.damage + S.landingDamageBase + (down - S.hardLandingSpeed) * S.landingDamagePerMs);
            this.impact = Math.max(this.impact, down);
          }
        }
        v.addScaledVector(n, down);
        // a hard touchdown on the wheels bounces off the bump stops; otherwise the surface just stops the motion
        if (!this.grounded && down > S.hardLandingSpeed && _up.y > 0.5) v.addScaledVector(n, down * S.hardLandingBounce);
        // bottoming kills the body's pitch / roll rate into the stop as well
        if (this.grounded && wheelPen > 0) { this.angularVelocity.x *= 0.5; this.angularVelocity.z *= 0.5; }
      }
    }
    // roof / side contact -> upside-down state
    if (_up.y < 0.5 && (upsideTouch || pen > 0)) {
      if (!this.upsideDown) { this.upsideDown = true; this._upsideTimer = 0; this._righting = false; }
    } else if (this.upsideDown && _up.y > 0.95) {
      this.upsideDown = false; this._righting = false;
    } else if (this.upsideDown && !upsideTouch && pen <= 0 && !this._righting) {
      this.upsideDown = false; // tumbled off something
    }
    // body scraping while upright but not on wheels (e.g. nose-first)
    if (!this.grounded && !this.upsideDown && pen > wheelPen) {
      const hs = Math.hypot(v.x, v.z);
      if (hs > 1e-3) {
        const dec = Math.min(hs, 0.5 * G * dt);
        v.x -= (v.x / hs) * dec; v.z -= (v.z / hs) * dec;
      }
    }
  }

  // ------------------------------------------------------------------------
  _collide(colliders) {
    const S = this.spec;
    const p = this.position, v = this.velocity, m = S.mass;
    obbFromPose(p, this.quaternion, this._obbCY, _ax, null);
    // swept test for fast motion: stop at the first penetrating sub-position
    sweepToContact(p, _prevPos, S.sweepStep, () => this._anyPenetration(colliders));
    const e = S.wallRestitution, muW = S.wallFriction;
    for (let iter = 0; iter < 4; iter++) {
      let hit = false;
      for (let ci = 0; ci < colliders.length; ci++) {
        const b = colliders[ci];
        if (!b) continue;
        // a box low enough to be driven onto (kerb, step, low deck edge) is a support for the wheels, not a wall
        if (b.maxY <= p.y + S.stepHeight && b.kind !== 'vehicle') continue;
        const depth = this._obbVsAabb(b, _n);
        if (depth <= 0) continue;
        hit = true;
        p.addScaledVector(_n, depth + 0.002);
        // a moving box (traffic) hits with its own velocity: respond in its frame
        const bvx = num(b.vx), bvz = num(b.vz), moving = bvx !== 0 || bvz !== 0;
        if (moving) { v.x -= bvx; v.z -= bvz; }
        const impact = bounceVelocity(v, _n, e, muW);
        if (moving) { v.x += bvx; v.z += bvz; }
        if (impact > 0) {
          this.impact = Math.max(this.impact, impact);
          if (impact > S.wallDamageSpeed) this.damage = Math.min(1, this.damage + (impact - S.wallDamageSpeed) * S.wallDamagePerMs);
          // yaw kick from an off-centre hit: arm = closest point of the box to the vehicle centre
          if (Math.abs(_n.y) < 0.7) {
            const cx = clamp(p.x, b.minX, b.maxX) - p.x, cz = clamp(p.z, b.minZ, b.maxZ) - p.z;
            const J = (1 + e) * impact * m;
            const tq = (cz * _n.x - cx * _n.z) * J; // (r x J).y
            this.angularVelocity.y += clamp((tq / S.yawInertia) * S.wallYawKick, -2, 2);
          }
        }
      }
      if (!hit) break;
    }
  }

  _anyPenetration(colliders) {
    for (let ci = 0; ci < colliders.length; ci++) {
      const b = colliders[ci];
      if (b && this._obbVsAabb(b, _nAny) > 0) return true;
    }
    return false;
  }

  // Body OBB vs a world AABB (axes in _ax must be current). Returns penetration depth and the push
  // direction for the vehicle in outN.
  _obbVsAabb(b, outN) {
    const p = this.position;
    const ca = _ca.copy(_ax[1]).multiplyScalar(this._obbCY).add(p);
    return obbVsAabb(ca, _ax, this._obbH, this._obbR, b, outN, b.minY > p.y + 0.2, b.kind === 'vehicle' && this.grounded);
  }

  // Support under a rolling tyre at (x,z): the surface a wheel of radius R effectively sits on. A step edge (box
  // top or a heightfield kerb) within the tyre radius ahead / behind lifts the wheel along the tyre's circle,
  // s = top - R + sqrt(R^2 - dist^2), so climbing a kerb takes several steps instead of one. (dx,dz): unit
  // horizontal travel direction (0,0 at rest: centre sample only).
  _wheelSupport(x, z, refY, dx, dz) {
    const R = this.spec.wheelRadius || 0.35;
    const cs = this._colliders || EMPTY;
    const lim = refY + this.spec.stepHeight;
    let s = groundAt(this._world, x, z, refY);
    if (dx !== 0 || dz !== 0) {
      // heightfield steps (raster kerbs, deck edges): two samples at 0.6 R along the travel line
      const o = 0.6 * R, lift = R - Math.sqrt(R * R - o * o);
      const sa = groundAt(this._world, x + dx * o, z + dz * o, refY) - lift;
      const sb = groundAt(this._world, x - dx * o, z - dz * o, refY) - lift;
      if (sa > s && sa <= lim) s = sa;
      if (sb > s && sb <= lim) s = sb;
    }
    for (let i = 0; i < cs.length; i++) {
      const b = cs[i];
      if (!b || b.maxY > lim || b.maxY <= s) continue;
      const ex = x < b.minX ? b.minX - x : x > b.maxX ? x - b.maxX : 0;
      const ez = z < b.minZ ? b.minZ - z : z > b.maxZ ? z - b.maxZ : 0;
      if (ex === 0 && ez === 0) { s = b.maxY; continue; }
      const d2 = ex * ex + ez * ez;
      if (d2 >= R * R) continue;
      const se = b.maxY - R + Math.sqrt(R * R - d2);
      if (se > s) s = se;
    }
    return s;
  }

  // Highest walkable surface under (x,z) that is not above refY + step tolerance:
  // ground or the top of a collider box (rooftops, bridge decks, kerbs).
  supportAt(x, z, refY) {
    let s = groundAt(this._world, x, z, refY);
    const cs = this._colliders || EMPTY;
    const lim = refY + this.spec.stepHeight;
    for (let i = 0; i < cs.length; i++) {
      const b = cs[i];
      if (!b) continue;
      if (x > b.minX && x < b.maxX && z > b.minZ && z < b.maxZ && b.maxY <= lim && b.maxY > s) s = b.maxY;
    }
    return s;
  }

  _estimateNormal() {
    // plane through the four wheel support heights (clamped near the highest one)
    const S = this.spec, sup = this._supports;
    let hmax = -Infinity;
    for (let i = 0; i < 4; i++) hmax = Math.max(hmax, sup[i]);
    const nc = S.normalClamp ?? 0.15;
    const h = (i) => Math.max(sup[i], hmax - nc);
    const L = S.axleRear - S.axleFront, T = 2 * S.halfTrack;
    const sF = ((h(0) + h(1)) - (h(2) + h(3))) / (2 * L);
    const sR = ((h(1) + h(3)) - (h(0) + h(2))) / (2 * T);
    _t1.set(0, 0, -1).applyQuaternion(this.quaternion); _t1.y = 0;
    if (_t1.lengthSq() < 1e-6) _t1.set(0, 0, -1);
    _t1.normalize();
    _t2.set(-_t1.z, 0, _t1.x); // right (horizontal)
    _t1.y = sF; _t2.y = sR;
    return _n.crossVectors(_t2, _t1).normalize();
  }

  _tankFracSoft() {
    return 0.35 + 0.65 * clamp(this.tank / this.spec.tankReserve, 0, 1);
  }

  _updateTank(dt, liftForce, rearThrust, retroThrust) {
    const S = this.spec;
    if (S.flight === false) { this.tank = 1; return; }
    const full = S.liftMaxG * S.mass * G;
    const use = full > 0 ? (liftForce + rearThrust * (this.boosting && this.grounded ? 0 : 1) + retroThrust) / full : 0;
    if (use > 0.01) this.tank = Math.max(0, this.tank - (use * dt) / S.tankFullThrustSeconds);
    else if (this.grounded) this.tank = Math.min(1, this.tank + dt / S.tankRechargeSeconds);
  }

  _thrusterVisuals(aa, rearThrust, retroThrust, flying) {
    const S = this.spec;
    const t = this.thrusters;
    if (S.flight === false) { t.fill(0); return; }
    const base = this.liftCmd;
    const pa = clamp(aa.x / 8, -1, 1) * 0.35, ra = clamp(aa.z / 8, -1, 1) * 0.35, ya = clamp(aa.y / 8, -1, 1) * 0.5;
    const att = flying ? 1 : 0;
    t[0] = clamp(base + att * (pa - ra) * 0.5 + att * Math.max(0, pa - ra) * 0.5, 0, 1);
    t[1] = clamp(base + att * (pa + ra) * 0.5 + att * Math.max(0, pa + ra) * 0.5, 0, 1);
    t[2] = clamp(base + att * (-pa - ra) * 0.5 + att * Math.max(0, -pa - ra) * 0.5, 0, 1);
    t[3] = clamp(base + att * (-pa + ra) * 0.5 + att * Math.max(0, -pa + ra) * 0.5, 0, 1);
    const rear = clamp(rearThrust / (S.mass * G * 1.3), 0, 1);
    t[4] = clamp(rear + att * Math.max(0, -ya), 0, 1);
    t[5] = clamp(rear + att * Math.max(0, ya), 0, 1);
    const retro = S.retroThrustG > 0 ? clamp(retroThrust / (S.mass * G * S.retroThrustG), 0, 1) : 0;
    t[6] = clamp(retro + att * Math.max(0, -ya), 0, 1);
    t[7] = clamp(retro + att * Math.max(0, ya), 0, 1);
  }

  // Per-wheel surface multipliers from world.surfaceAt(x, z, out) at the wheel contact points (_wheelW).
  _querySurfaces(world) {
    const W = this.wheels;
    if (!world || typeof world.surfaceAt !== 'function') {
      for (let i = 0; i < 4; i++) { const w = W[i]; w.mu = 1; w.crr = 1; w.tag = null; }
      return;
    }
    const o = this._surf;
    for (let i = 0; i < 4; i++) {
      const wp = this._wheelW[i], w = W[i];
      o.mu = 1; o.crr = 1; o.tag = null;
      const r = world.surfaceAt(wp.x, wp.z, o) || o;
      w.mu = mult(r.mu); w.crr = mult(r.crr); w.tag = r.tag ?? null;
    }
  }

  // hold: frozen in place (no velocity, no integration), resting on the support surface with the heading kept.
  _holdStep(dt, c, world) {
    const S = this.spec, p = this.position, q = this.quaternion, W = this.wheels, sup = this._supports;
    this.velocity.set(0, 0, 0);
    this.angularVelocity.set(0, 0, 0);
    this.impact = 0; this.landedThisStep = false; this.bottomed = false;
    this._colliders = queryColliders(world, p.x, p.z, S.colliderRadius, this._colOut) || EMPTY;
    this._world = world;
    // heading measured under the current tilt (so holding on a side slope does not drift), then rest on the supports
    for (let i = 0; i < 4; i++) {
      const wp = _t1.copy(W[i]).applyQuaternion(q).add(p);
      sup[i] = this.supportAt(wp.x, wp.z, wp.y + 0.35);
    }
    const heading = this._headingUnder(this._estimateNormal());
    this._restOnSupports(Number.isFinite(heading) ? heading : 0, false);
    this._querySurfaces(world);
    this.grounded = true; this.mode = 'ground'; this.flightArmed = false; this.airTime = 0; this.altitude = 0;
    this.upsideDown = false; this._upsideTimer = 0; this._righting = false;
    this.bodyContact = false;
    this.boosting = false; this.braking = false; this.reversing = false;
    this.boostLevel = Math.min(1, this.boostLevel + dt / S.boostRechargeSeconds);
    this.tank = S.flight === false ? 1 : Math.min(1, this.tank + dt / S.tankRechargeSeconds);
    this.thrustCmd = 0; this.liftCmd = 0; this.thrusters.fill(0);
    this.wheelSlip = 0; this.wheelSpeed = 0; this.speed = 0; this.forwardSpeed = 0;
    this.steerAngle = -clamp(num(c && c.steer), -1, 1) * S.maxSteer; // wheels still turn (visual)
    const k = Math.min(1, dt / 0.12);
    this.localAccel.multiplyScalar(1 - k);
    this.gForce += (1 - this.gForce) * Math.min(1, dt / 0.08);
    return this;
  }
}

// ---- helpers ------------------------------------------------------------------
const pac = (C, B, alpha) => Math.sin(C * Math.atan(B * alpha));

function basis(q) {
  _fwd.set(0, 0, -1).applyQuaternion(q);
  _up.set(0, 1, 0).applyQuaternion(q);
  _right.set(1, 0, 0).applyQuaternion(q);
}

function integrateLocal(q, w, dt) {
  const ang = w.length() * dt;
  if (ang < 1e-9) return;
  _t1.copy(w).normalize();
  _q1.setFromAxisAngle(_t1, ang);
  q.multiply(_q1).normalize();
}

// World adapters: PhysicsWorld and legacy worlds share getGroundHeight/queryColliders; groundHeight/queryStatic
// are accepted for minimal custom worlds.
function groundAt(world, x, z, y) {
  if (!world) return 0;
  const h = typeof world.getGroundHeight === 'function' ? world.getGroundHeight(x, z, y)
    : typeof world.groundHeight === 'function' ? world.groundHeight(x, z, y) : 0;
  return typeof h === 'number' && Number.isFinite(h) ? h : 0;
}

function queryColliders(world, x, z, r, out) {
  if (!world) return null;
  if (typeof world.queryColliders === 'function') return world.queryColliders(x, z, r, out);
  if (typeof world.queryStatic === 'function') return world.queryStatic(x, z, r, out);
  return null;
}
