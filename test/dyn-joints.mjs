// Headless checks for the XPBD primitives (src/physics/dynamics/xpbd.js) and the joints (joints.js) of the dynamics
// world. No DynamicsWorld: the bodies are driven by a minimal substep loop (integrateBody -> solveJointsPositions ->
// velocityFromPositions -> solveJointsVelocities), 4 substeps per 1/120 s frame, gravity 9.81, 10 s runs; box
// inertia tensors are computed inline. Run: node test/dyn-joints.mjs (imports the modules by path).
import v8 from 'node:v8';
import vm from 'node:vm';
import { BodyStore, BODY, WORLD_BODY } from '../src/physics/dynamics/state.js';
import {
  integrateBody, velocityFromPositions, worldInertiaApply, worldInvInertiaApply, generalizedInverseMass, applyPositional,
  applyAngular, quatMul, quatConj, rotate, rotateInverse,
} from '../src/physics/dynamics/xpbd.js';
import {
  JointStore, JOINT, solveJointsPositions, solveJointsVelocities, jointAngle, jointDistance, jointSwing, jointAtan2,
} from '../src/physics/dynamics/joints.js';
import { hashF64Array, detAtan2, mulberry32 } from '../src/core/mathDet.js';

const G = 9.81, FRAME = 1 / 120, SUB = 4, H = FRAME / SUB, SECONDS = 10, STEPS = Math.round(SECONDS / H);
const DEG = Math.PI / 180;

let failures = 0;
const results = [];
function check(name, value, lo = 1, hi = lo, unit = '') {
  if (typeof value === 'boolean') value = value ? 1 : 0;
  const ok = Number.isFinite(value) && value >= lo && value <= hi;
  if (!ok) failures++;
  results.push(`${ok ? 'PASS' : 'FAIL'}  ${name.padEnd(60)} ${fmt(value)} ${unit}   [${fmt(lo).trim()}${hi !== lo ? ' .. ' + fmt(hi).trim() : ''}]`);
  return ok;
}
const fmt = (x) => (!Number.isFinite(x) ? String(x) : Number.isInteger(x) ? String(x) : Math.abs(x) < 1e-3 && x !== 0 ? x.toExponential(2) : x.toFixed(4)).padStart(11);
const note = (s) => results.push(`NOTE  ${s}`);

// ---- harness --------------------------------------------------------------------------------------------------------
/** Dynamic box body (full sizes sx, sy, sz), analytic tensor I = m / 12 diag(sy^2 + sz^2, sx^2 + sz^2, sx^2 + sy^2). */
function addBox(B, m, sx, sy, sz, x, y, z, q = null) {
  const id = B.alloc();
  B.type[id] = BODY.DYNAMIC; B.mass[id] = m; B.invMass[id] = 1 / m;
  const k = id * 9;
  B.invInertia[k] = 12 / (m * (sy * sy + sz * sz));
  B.invInertia[k + 4] = 12 / (m * (sx * sx + sz * sz));
  B.invInertia[k + 8] = 12 / (m * (sx * sx + sy * sy));
  B.pos[id * 3] = x; B.pos[id * 3 + 1] = y; B.pos[id * 3 + 2] = z;
  if (q) B.quat.set(q, id * 4);
  B.prevPos.set(B.pos.subarray(id * 3, id * 3 + 3), id * 3);
  B.prevQuat.set(B.quat.subarray(id * 4, id * 4 + 4), id * 4);
  return id;
}
function axisAngleQuat(ax, ay, az, ang) { // construction time only (Math.sin / cos allowed here)
  const l = Math.hypot(ax, ay, az), s = Math.sin(ang / 2) / l;
  return [ax * s, ay * s, az * s, Math.cos(ang / 2)];
}
const GY = -G; // one boxed constant: a `-g` computed per call would be a fresh heap number per body (harness allocation)
function substep(B, J, h = H, gy = GY) {
  for (let id = 0; id < B.count; id++) if (B.alive[id]) integrateBody(B, id, h, 0, gy, 0);
  solveJointsPositions(J, B, h);
  for (let id = 0; id < B.count; id++) if (B.alive[id]) velocityFromPositions(B, id, h);
  solveJointsVelocities(J, B, h);
}
/** Run `steps` substeps; onStep(t) after each (t = end time of the substep). */
function run(B, J, steps, onStep) {
  for (let i = 0; i < steps; i++) { substep(B, J); if (onStep) onStep((i + 1) * H); }
}
const _w = new Float64Array(3), _u = new Float64Array(3), _q = new Float64Array(8);
function finiteState(B) {
  for (let i = 0; i < B.count * 3; i++) if (!Number.isFinite(B.pos[i]) || !Number.isFinite(B.vel[i]) || !Number.isFinite(B.angVel[i])) return false;
  for (let i = 0; i < B.count * 4; i++) if (!Number.isFinite(B.quat[i])) return false;
  return true;
}
/** Pose of body b in body a's frame: position (into out[0..3]) and rotation quaternion (into out[3..7]). */
function relPose(B, a, b, out) {
  rotateInverse(B.quat, a * 4, B.pos[b * 3] - B.pos[a * 3], B.pos[b * 3 + 1] - B.pos[a * 3 + 1], B.pos[b * 3 + 2] - B.pos[a * 3 + 2], out, 0);
  quatConj(_q, 0, B.quat, a * 4); quatMul(_q, 4, _q, 0, B.quat, b * 4);
  out[3] = _q[4]; out[4] = _q[5]; out[5] = _q[6]; out[6] = _q[7];
}
/** Angle (rad) between two unit quaternions given as arrays at offsets. */
function quatAngle(p, po, q, qo) {
  const d = Math.abs(p[po] * q[qo] + p[po + 1] * q[qo + 1] + p[po + 2] * q[qo + 2] + p[po + 3] * q[qo + 3]);
  // 2 atan2(|sin|, |cos|) of the half angle, accurate near 0
  quatConj(_q, 0, p, po); quatMul(_q, 4, q, qo, _q, 0);
  const s = Math.hypot(_q[4], _q[5], _q[6]);
  return 2 * detAtan2(s, d);
}

// ================================================================================== 1. integrator round trip, |L|
{
  const B = new BodyStore(8);
  const id = addBox(B, 2, 1, 2, 3, 0.3, 1.2, -0.7, axisAngleQuat(1, 2, 3, 0.8));
  B.vel.set([1.5, -2, 0.5], id * 3); B.angVel.set([0.7, -1.3, 2.1], id * 3);
  let errV = 0, errW = 0;
  for (let i = 0; i < 200; i++) {
    integrateBody(B, id, H, 0, -G, 0);
    const v = Array.from(B.vel.subarray(0, 3)), w = Array.from(B.angVel.subarray(0, 3));
    velocityFromPositions(B, id, H);
    for (let k = 0; k < 3; k++) { errV = Math.max(errV, Math.abs(B.vel[k] - v[k])); errW = Math.max(errW, Math.abs(B.angVel[k] - w[k])); }
  }
  check('round trip: |v - v(x, x_prev)| max over 200 substeps', errV, 0, 1e-12, 'm/s');
  check('round trip: |w - w(q, q_prev)| max (tumbling 1x2x3 box)', errW, 0, 1e-12, 'rad/s');

  // a tumbling asymmetric box (spin near the unstable intermediate axis): |L| and rotational energy
  const C = new BodyStore(4);
  const c = addBox(C, 3, 0.4, 1, 2.2, 0, 0, 0, axisAngleQuat(0.3, 1, -0.4, 1.1));
  C.angVel.set([0.4, 6, 0.3], c * 3); // rad/s, mostly about the intermediate axis -> tumbles
  const momentum = () => { worldInertiaApply(C, c, C.angVel[0], C.angVel[1], C.angVel[2], _w); return Math.hypot(_w[0], _w[1], _w[2]); };
  const energy = () => { worldInertiaApply(C, c, C.angVel[0], C.angVel[1], C.angVel[2], _w); return 0.5 * (_w[0] * C.angVel[0] + _w[1] * C.angVel[1] + _w[2] * C.angVel[2]); };
  worldInertiaApply(C, c, C.angVel[0], C.angVel[1], C.angVel[2], _u);
  const L0 = Math.hypot(_u[0], _u[1], _u[2]), E0 = energy(), Lv0 = Array.from(_u);
  let dL = 0, dE = 0, wMin = Infinity, wMax = 0;
  for (let i = 0; i < 1000; i++) {
    integrateBody(C, c, H, 0, 0, 0); velocityFromPositions(C, c, H);
    dL = Math.max(dL, Math.abs(momentum() / L0 - 1)); dE = Math.max(dE, Math.abs(energy() / E0 - 1));
    rotateInverse(C.quat, c * 4, C.angVel[0], C.angVel[1], C.angVel[2], _u, 0);
    wMin = Math.min(wMin, Math.abs(_u[1])); wMax = Math.max(wMax, Math.abs(_u[1]));
  }
  worldInertiaApply(C, c, C.angVel[0], C.angVel[1], C.angVel[2], _u);
  const dirErr = Math.acos(Math.min(1, (_u[0] * Lv0[0] + _u[1] * Lv0[1] + _u[2] * Lv0[2]) / (Math.hypot(_u[0], _u[1], _u[2]) * L0)));
  check('gyroscopic: |L| relative drift, 1000 substeps (tumbling box)', dL, 0, 1e-6);
  check('gyroscopic: rotational energy relative drift, 1000 substeps', dE, 0, 1e-6);
  check('gyroscopic term active: body-frame w_y swings (min/max)', wMin / wMax, 0, 0.9);
  note(`world L direction error after 1000 substeps ${dirErr.toExponential(2)} rad (first-order precession error, O(h^2) per step)`);

  // primitives: positional correction of a static-vs-dynamic pair closes c, angular correction undoes the rotation
  const D = new BodyStore(4);
  const d = addBox(D, 1, 1, 1, 1, 0, 0, 0);
  const lam = applyPositional(D, d, WORLD_BODY, 0, 1, 0, 0.01, 0, 0, 0, 0, 0, 0, 0, H, 0);
  check('applyPositional: centre point, c = 0.01 along +y -> y = -0.01', D.pos[1], -0.01 - 1e-15, -0.01 + 1e-15, 'm');
  check('applyPositional: dLambda = -c / w (w = 1/m)', lam, -0.01 - 1e-15, -0.01 + 1e-15);
  const w = generalizedInverseMass(D, d, 0.5, 0, 0, 0, 1, 0);
  worldInvInertiaApply(D, d, 0, 0, 1, _w);
  check('generalizedInverseMass: 1/m + (r x n) I^-1 (r x n)', w, 1 + 0.25 * _w[2] - 1e-12, 1 + 0.25 * _w[2] + 1e-12);
  applyAngular(D, d, WORLD_BODY, 0, 0, 0.002, 0, H, 0);
  const qz = D.quat[2];
  check('applyAngular: rotation vector +0.002 z undone (q_z = -0.001)', qz, -0.001 - 1e-9, -0.001 + 1e-9);
}

// ================================================================================== 2. simple pendulum (ball joint)
{
  const B = new BodyStore(4), J = new JointStore(4);
  const L = 1, th0 = 10 * DEG, px = 0, py = 2;
  const bob = addBox(B, 1, 0.05, 0.05, 0.05, px + L * Math.sin(th0), py - L * Math.cos(th0), 0);
  J.add({ type: JOINT.BALL, bodyA: bob, bodyB: WORLD_BODY, anchorA: [-L * Math.sin(th0), L * Math.cos(th0), 0] }, B);
  const cross = [];
  let xPrev = B.pos[0] - px;
  run(B, J, STEPS, (t) => {
    const x = B.pos[0] - px;
    if (xPrev < 0 && x >= 0) cross.push(t - H * x / (x - xPrev)); // upward zero crossing, linearly interpolated
    xPrev = x;
  });
  const T = (cross[cross.length - 1] - cross[0]) / (cross.length - 1), T0 = 2 * Math.PI * Math.sqrt(L / G);
  check('pendulum: period / 2 pi sqrt(L/g) (1 m, 10 deg)', T / T0, 0.99, 1.01);
  note(`pendulum period ${T.toFixed(5)} s over ${cross.length - 1} periods; 2 pi sqrt(L/g) = ${T0.toFixed(5)} s; with the 10 deg ` +
    `amplitude term (1 + theta^2/16) and the bob's own inertia the exact value is ${(T0 * (1 + th0 * th0 / 16) * Math.sqrt(1 + (0.05 * 0.05 / 6) / (L * L))).toFixed(5)} s`);
  check('pendulum: anchor gap after 10 s', jointDistance(J, B, 0), 0, 1e-9, 'm');
}

// ================================================================================== 3. hinge limit under gravity
{
  const B = new BodyStore(4), J = new JointStore(4);
  const lever = addBox(B, 1, 1, 0.1, 0.1, 0.5, 1, 0); // pivot at (0, 1, 0), lever along +x
  const lo = -20 * DEG, hi = 20 * DEG;
  const j = J.add({ type: JOINT.HINGE, bodyA: lever, bodyB: WORLD_BODY, anchorA: [-0.5, 0, 0], axisA: [0, 0, 1], limits: [lo, hi] }, B);
  let minPhi = Infinity, gap = 0;
  run(B, J, STEPS, () => { minPhi = Math.min(minPhi, jointAngle(J, B, j)); gap = Math.max(gap, jointDistance(J, B, j)); });
  check('hinge limit: lowest angle - lo (gravity pushes past -20 deg)', (minPhi - lo) / DEG, -0.5, 0.5, 'deg');
  check('hinge limit: resting angle after 10 s - lo', (jointAngle(J, B, j) - lo) / DEG, -0.5, 0.5, 'deg');
  check('hinge limit: max anchor gap', gap, 0, 1e-6, 'm');
  check('hinge limit: axis misalignment after 10 s', jointSwing(J, B, j), 0, 1e-6, 'rad');
}

// ================================================================================== 4. hinge motor
{
  // a 1 m, 2 kg lever on a horizontal hinge: gravity torque up to m g L/2 = 9.81 N m opposes the motor half a turn
  const mk = (motor) => {
    const B = new BodyStore(4), J = new JointStore(4);
    const lever = addBox(B, 2, 1, 0.1, 0.1, 0.5, 1, 0);
    const j = J.add({ type: JOINT.HINGE, bodyA: lever, bodyB: WORLD_BODY, anchorA: [-0.5, 0, 0], axisA: [0, 0, 1], motor }, B);
    return { B, J, lever, j };
  };
  const load = 2 * G * 0.5;
  {
    const { B, J, lever } = mk({ velocity: 2, maxForce: 3 * load });
    let tReach = NaN, lo = Infinity, hi = -Infinity;
    run(B, J, STEPS, (t) => {
      const w = B.angVel[lever * 3 + 2];
      if (Number.isNaN(tReach) && Math.abs(w / 2 - 1) <= 0.05) tReach = t;
      if (t > 0.5) { lo = Math.min(lo, w); hi = Math.max(hi, w); }
    });
    check('hinge motor: time to reach 2 rad/s within 5 %', tReach, 0, 0.5, 's');
    check('hinge motor: slowest w after 0.5 s (lifting the load) / target', lo / 2, 0.95, 1.05);
    check('hinge motor: fastest w after 0.5 s (load helping) / target', hi / 2, 0.95, 1.05);
  }
  {
    const { B, J, j } = mk({ velocity: 0, maxForce: 3 * load });
    const phi0 = jointAngle(J, B, j);
    let drift = 0;
    run(B, J, STEPS, () => { drift = Math.max(drift, Math.abs(jointAngle(J, B, j) - phi0)); });
    check('hinge motor: velocity 0 holds the horizontal lever (max drift)', drift / DEG, 0, 0.01, 'deg');
    check('hinge motor: reported torque = m g L/2', J.torque[j] / load, 0.99, 1.01);
    check('hinge motor: reported force = m g', J.force[j] / (2 * G), 0.99, 1.01);
  }
  {
    const { B, J, j } = mk({ velocity: 0, maxForce: 0.5 * load });
    run(B, J, 120);
    check('hinge motor: maxForce below the load slips (angle after 0.25 s)', -jointAngle(J, B, j) / DEG, 5, 90, 'deg');
  }
}

// ================================================================================== 5. 20-link ball-joint chain
function buildChain(B, J, n = 20, len = 0.1, m = 0.1, y = 5) {
  const ids = [];
  for (let i = 0; i < n; i++) {
    const id = addBox(B, m, len, 0.02, 0.02, len * (i + 0.5), y, 0);
    J.add({ type: JOINT.BALL, bodyA: id, bodyB: i === 0 ? WORLD_BODY : ids[i - 1], anchorA: [-len / 2, 0, 0] }, B);
    ids.push(id);
  }
  return ids;
}
{
  const B = new BodyStore(32), J = new JointStore(32);
  buildChain(B, J);
  const gapSum = () => { let s = 0; for (let j = 0; j < J.count; j++) s += jointDistance(J, B, j); return s; };
  let maxGap = 0;
  run(B, J, STEPS, () => { maxGap = Math.max(maxGap, gapSum()); });
  check('chain: total length drift (sum of joint gaps) after 10 s', gapSum() * 1000, 0, 1, 'mm');
  check('chain: no NaN', finiteState(B));
  note(`chain: largest total gap during the fall from horizontal ${(maxGap * 1000).toFixed(3)} mm; tip at y = ${B.pos[19 * 3 + 1].toFixed(3)} m`);
}

// ================================================================================== 6. fixed joint
{
  const B = new BodyStore(4), J = new JointStore(4);
  // two different boxes welded at A's origin; the pair hangs from a ball joint and swings in 3D
  const a = addBox(B, 1, 0.4, 0.2, 0.2, 0.5, 3, 0, axisAngleQuat(0, 0, 1, 0.3));
  const b = addBox(B, 3, 0.4, 0.3, 0.6, 0.85, 3.1, 0.05, axisAngleQuat(1, 1, 0, 0.4));
  J.add({ type: JOINT.BALL, bodyA: a, bodyB: WORLD_BODY, anchorA: [-0.5, 0.2, 0] }, B);
  const f = J.add({ type: JOINT.FIXED, bodyA: b, bodyB: a }, B);
  B.vel.set([0, 0, 2], a * 3); B.vel.set([0, 0, 2], b * 3);
  const p0 = new Float64Array(7), p = new Float64Array(7);
  relPose(B, a, b, p0);
  let dPos = 0, dRot = 0, ext = 0;
  run(B, J, STEPS, () => {
    relPose(B, a, b, p);
    dPos = Math.max(dPos, Math.hypot(p[0] - p0[0], p[1] - p0[1], p[2] - p0[2]));
    dRot = Math.max(dRot, quatAngle(p, 3, p0, 3));
    ext = Math.max(ext, Math.abs(B.pos[a * 3 + 2]));
  });
  check('fixed: relative position drift, max over 10 s swinging', dPos, 0, 1e-6, 'm');
  check('fixed: relative rotation drift, max over 10 s swinging', dRot, 0, 1e-6, 'rad');
  check('fixed: the pair actually swings (max |z|)', ext, 0.2, 10, 'm');
  note(`fixed: drift ${dPos.toExponential(2)} m / ${dRot.toExponential(2)} rad; joint force at the end ${J.force[f].toFixed(2)} N`);
}

// ================================================================================== 7. slider
{
  const B = new BodyStore(4), J = new JointStore(4);
  const box = addBox(B, 1, 0.2, 0.3, 0.2, 0, 2, 0);
  const s2 = Math.SQRT1_2, lo = -1, hi = 0.5;
  const j = J.add({ type: JOINT.SLIDER, bodyA: box, bodyB: WORLD_BODY, anchorA: [0.1, 0.15, -0.05], axisA: [s2, s2, 0], limits: [lo, hi] }, B);
  B.vel.set([0, 0, 1.5], box * 3);    // sideways kick: must not leave the axis
  B.angVel.set([1, 2, -1], box * 3);  // and a spin: must not rotate
  const q0 = Array.from(B.quat.subarray(box * 4, box * 4 + 4));
  let perp = 0, minS = Infinity, rot = 0;
  run(B, J, STEPS, () => {
    const s = jointDistance(J, B, j);
    // perpendicular offset of anchor A from B's axis line through anchor B (world axis = (1, 1, 0)/sqrt 2)
    rotate(B.quat, box * 4, 0.1, 0.15, -0.05, _w, 0);
    const ax = B.pos[0] + _w[0], ay = B.pos[1] + _w[1], az = B.pos[2] + _w[2];
    const bx = 0.1, by = 2.15, bz = -0.05, dx = ax - bx, dy = ay - by, dz = az - bz, along = (dx + dy) * s2;
    perp = Math.max(perp, Math.hypot(dx - along * s2, dy - along * s2, dz));
    minS = Math.min(minS, s);
    rot = Math.max(rot, quatAngle(B.quat, box * 4, q0, 0));
  });
  check('slider: perpendicular drift of the anchor, max over 10 s', perp, 0, 1e-6, 'm');
  check('slider: lowest position - lower limit (gravity along the axis)', minS - lo, -1e-6, 1e-6, 'm');
  check('slider: resting at the lower limit after 10 s', jointDistance(J, B, j) - lo, -1e-6, 1e-6, 'm');
  check('slider: rotation drift, max over 10 s', rot, 0, 1e-6, 'rad');
}

// ================================================================================== 8. distance rope
{
  const B = new BodyStore(4), J = new JointStore(4);
  const box = addBox(B, 1, 0.1, 0.1, 0.1, 0.3, 4.6, 0);
  const j = J.add({ type: JOINT.DISTANCE, bodyA: box, bodyB: WORLD_BODY, anchorB: [0, 5, 0], distance: 1, rope: true }, B);
  let tTaut = NaN, freeErr = 0, over = 0, slackAgain = 0, tautAgain = NaN, kicked = false;
  const y0 = 4.6;
  run(B, J, STEPS, (t) => {
    const L = jointDistance(J, B, j);
    if (Number.isNaN(tTaut)) {
      if (J.force[j] > 0) tTaut = t;
      else { const n = Math.round(t / H); freeErr = Math.max(freeErr, Math.abs(B.pos[1] - (y0 - G * H * H * n * (n + 1) / 2))); }
    }
    over = Math.max(over, L - 1);
    if (!kicked && t >= 4) { kicked = true; B.vel[1] += 7; }          // throw it up: the rope must go slack
    if (kicked && Number.isNaN(tautAgain)) {
      if (L < 0.9 && J.force[j] === 0) slackAgain++;
      if (slackAgain > 0 && J.force[j] > 0) tautAgain = t;
    }
  });
  check('rope: slack phase is exact free fall (max |y - y_free|)', freeErr, 0, 1e-12, 'm');
  check('rope: tightens at |d| = 1 (time)', tTaut, 0.2, 0.6, 's');
  check('rope: never longer than 1 m (max excess)', over, 0, 1e-9, 'm');
  check('rope: slackens after an upward throw (substeps with |d| < 0.9, no force)', slackAgain, 50, Infinity);
  check('rope: tightens again', tautAgain, 4, 10, 's');
}

// ================================================================================== 9. d6 swing cone and twist limit
{
  const B = new BodyStore(4), J = new JointStore(4);
  const box = addBox(B, 1, 0.1, 0.6, 0.1, 0, 4.7, 0); // hangs from its top end
  const cone = 30 * DEG, tw = 10 * DEG;
  const j = J.add({ type: JOINT.D6, bodyA: box, bodyB: WORLD_BODY, anchorA: [0, 0.3, 0], linear: [true, true, true], swing: cone, twist: [-tw, tw] }, B);
  B.vel.set([4, 0, 2], box * 3); B.angVel.set([0, 6, 0], box * 3); // swing hard and spin about the long axis
  let maxSwing = 0, maxTwist = 0, gap = 0;
  run(B, J, STEPS, () => {
    maxSwing = Math.max(maxSwing, jointSwing(J, B, j)); maxTwist = Math.max(maxTwist, Math.abs(jointAngle(J, B, j)));
    gap = Math.max(gap, jointDistance(J, B, j));
  });
  check('d6: max swing - 30 deg cone', (maxSwing - cone) / DEG, -1, 1, 'deg');
  check('d6: max |twist| - 10 deg limit', (maxTwist - tw) / DEG, -1, 1, 'deg');
  check('d6: linear lock (max anchor gap)', gap, 0, 1e-6, 'm');
}

// ================================================================================== 10. compliance = spring
{
  const B = new BodyStore(4), J = new JointStore(4);
  const m = 2, k = 1000;
  const box = addBox(B, m, 0.1, 0.1, 0.1, 0, 4, 0);
  const j = J.add({ type: JOINT.DISTANCE, bodyA: box, bodyB: WORLD_BODY, anchorB: [0, 5, 0], distance: 1, compliance: 1 / k, damping: 4 }, B);
  let maxExt = 0;
  run(B, J, STEPS, () => { maxExt = Math.max(maxExt, jointDistance(J, B, j) - 1); });
  const ext = jointDistance(J, B, j) - 1, expect = m * G / k;
  check('spring: static extension / (m g / k), k = 1/compliance = 1000 N/m', ext / expect, 0.98, 1.02);
  check('spring: overshoot from rest (dynamic peak ~ 2 m g / k)', maxExt / expect, 1.5, 2.05);
  check('spring: reported force = m g', J.force[j] / (m * G), 0.98, 1.02);
}

// ================================================================================== 11. breakForce
{
  const B = new BodyStore(4), J = new JointStore(4);
  const box = addBox(B, 1, 0.2, 0.2, 0.2, 0, 4, 0);
  const j = J.add({ type: JOINT.BALL, bodyA: box, bodyB: WORLD_BODY, anchorA: [0, 0.5, 0], breakForce: 15 }, B);
  run(B, J, 480);
  check('break: holds 9.81 N under breakForce 15 N (broken)', J.broken[j], 0, 0);
  check('break: reported force while holding = m g', J.force[j] / G, 0.99, 1.01);
  B.force[box * 3 + 1] = -10; // 19.81 N > 15 N
  let tBreak = NaN;
  run(B, J, 480, (t) => { if (Number.isNaN(tBreak) && J.broken[j]) tBreak = t; });
  check('break: overloaded joint breaks (broken flag)', J.broken[j], 1, 1);
  check('break: within one substep of the overload', tBreak, 0, H * 1.5, 's');
  check('break: body falls free afterwards (y)', B.pos[box * 3 + 1], -Infinity, 0);
}

// ================================================================================== 12. more topologies, slider motor
{
  // the solver's flat atan2 against core/mathDet.js detAtan2: bit-identical (all quadrants, 1e-300 .. 1e300, specials)
  const rnd = mulberry32(7);
  let diff = 0;
  const specials = [0, -0, 1, -1, 1e-300, -1e-300, 1e300, Infinity, -Infinity, NaN, 0.66, 2.414213562373095];
  for (const y of specials) for (const x of specials) if (!Object.is(jointAtan2(y, x), detAtan2(y, x))) diff++;
  for (let i = 0; i < 100000; i++) {
    const y = (rnd() - 0.5) * 10 ** (rnd() * 600 - 300) * (i % 3 === 0 ? 1e-300 : 1), x = (rnd() - 0.5) * 10 ** (rnd() * 20 - 10);
    if (!Object.is(jointAtan2(y, x), detAtan2(y, x))) diff++;
  }
  check('jointAtan2 == detAtan2 bit for bit (1e5 random + specials)', diff, 0, 0);
}
{
  // a closed loop: three rods joined into a triangle by ball joints, one corner hung from the world (the third joint
  // closes the loop and is solved on its own after the tree)
  const B = new BodyStore(8), J = new JointStore(8);
  const L = 0.6, hgt = L * Math.sqrt(3) / 2, top = [0, 3, 0];
  const p0 = [0, 3], p1 = [-L / 2, 3 - hgt], p2 = [L / 2, 3 - hgt];
  const rod = (pa, pb) => {
    const cx = (pa[0] + pb[0]) / 2, cy = (pa[1] + pb[1]) / 2, ang = Math.atan2(pb[1] - pa[1], pb[0] - pa[0]);
    return addBox(B, 0.5, L, 0.04, 0.04, cx, cy, 0, axisAngleQuat(0, 0, 1, ang));
  };
  const r0 = rod(p0, p1), r1 = rod(p1, p2), r2 = rod(p2, p0);
  const corner = (ra, rb, p) => J.add({ type: JOINT.BALL, bodyA: ra, bodyB: rb, anchorA: localOf(B, ra, p[0], p[1], 0) }, B);
  function localOf(B, id, x, y, z) { rotateInverse(B.quat, id * 4, x - B.pos[id * 3], y - B.pos[id * 3 + 1], z - B.pos[id * 3 + 2], _w, 0); return [_w[0], _w[1], _w[2]]; }
  J.add({ type: JOINT.BALL, bodyA: r0, bodyB: WORLD_BODY, anchorA: localOf(B, r0, top[0], top[1], 0) }, B);
  corner(r0, r1, p1); corner(r1, r2, p2); corner(r2, r0, p0);
  B.vel.set([0, 0, 1.5], r1 * 3); // swing it out of plane
  let gap = 0, gap0 = 0;
  run(B, J, STEPS, (t) => {
    for (let j = 0; j < J.count; j++) { const d = jointDistance(J, B, j); if (t < 1) gap0 = Math.max(gap0, d); else gap = Math.max(gap, d); }
  });
  check('loop: triangle of ball joints, max joint gap 1..10 s', gap, 0, 1e-4, 'm');
  check('loop: no NaN', finiteState(B));
  note(`loop: largest gap in the first second (after the 1.5 m/s kick) ${(gap0 * 1000).toFixed(3)} mm, then ${(gap * 1000).toFixed(4)} mm`);
}
{
  // a door on two hinges with the same axis (redundant rows): it must stay level and keep spinning freely
  const B = new BodyStore(4), J = new JointStore(4);
  const door = addBox(B, 20, 1, 2, 0.05, 0.5, 1, 0);
  J.add({ type: JOINT.HINGE, bodyA: door, bodyB: WORLD_BODY, anchorA: [-0.5, 0.9, 0], axisA: [0, 1, 0] }, B);
  J.add({ type: JOINT.HINGE, bodyA: door, bodyB: WORLD_BODY, anchorA: [-0.5, -0.9, 0], axisA: [0, 1, 0] }, B);
  B.angVel[door * 3 + 1] = 1; B.vel[door * 3 + 2] = -0.5; // spinning about the hinge line x = 0
  let gap = 0;
  run(B, J, STEPS, () => { gap = Math.max(gap, jointDistance(J, B, 0), jointDistance(J, B, 1)); });
  check('two hinges: max anchor gap (redundant constraints)', gap, 0, 1e-6, 'm');
  // XPBD's projection loses ~0.4 (w h)^2 of the speed per substep on a circular path: ~1 % over 10 s here
  check('two hinges: door keeps spinning (w_y after 10 s)', B.angVel[door * 3 + 1], 0.98, 1.0, 'rad/s');
  check('two hinges: no sag (door centre y)', B.pos[door * 3 + 1], 1 - 1e-6, 1 + 1e-6, 'm');
}
{
  // a jointed pendulum hanging from a rotated static body (not the world): anchors in a static frame
  const B = new BodyStore(4), J = new JointStore(4);
  const base = B.alloc();
  B.type[base] = BODY.STATIC; B.pos.set([2, 3, 1], base * 3); B.quat.set(axisAngleQuat(1, 2, 0.5, 0.9), base * 4);
  B.prevPos.set(B.pos.subarray(base * 3, base * 3 + 3), base * 3); B.prevQuat.set(B.quat.subarray(base * 4, base * 4 + 4), base * 4);
  const bob = addBox(B, 1, 0.1, 0.1, 0.1, 2.6, 3, 1);
  const j = J.add({ type: JOINT.BALL, bodyA: bob, bodyB: base, anchorA: [-0.6, 0, 0] }, B);
  let gap = 0;
  run(B, J, STEPS, () => { gap = Math.max(gap, jointDistance(J, B, j)); });
  check('static parent: anchor gap, max over 10 s', gap, 0, 1e-6, 'm');
  check('static parent: base did not move', Math.hypot(B.pos[base * 3] - 2, B.pos[base * 3 + 1] - 3, B.pos[base * 3 + 2] - 1), 0, 0);
}
{
  // slider motor: drives the carriage at 0.5 m/s along a horizontal rail, then stops at the upper limit
  const B = new BodyStore(4), J = new JointStore(4);
  const cart = addBox(B, 2, 0.3, 0.2, 0.2, 0, 1, 0);
  const j = J.add({ type: JOINT.SLIDER, bodyA: cart, bodyB: WORLD_BODY, axisA: [1, 0, 0], limits: [-1, 2], motor: { velocity: 0.5, maxForce: 50 } }, B);
  let vLo = Infinity, vHi = -Infinity, vPark = 0;
  run(B, J, STEPS, (t) => {
    const v = B.vel[cart * 3];
    if (t > 0.5 && t < 3.5) { vLo = Math.min(vLo, v); vHi = Math.max(vHi, v); }
    if (t > 5) vPark = Math.max(vPark, Math.abs(v));
  });
  check('slider motor: speed 0.5..3.5 s / 0.5 m/s (min)', vLo / 0.5, 0.95, 1.05);
  check('slider motor: speed 0.5..3.5 s / 0.5 m/s (max)', vHi / 0.5, 0.95, 1.05);
  check('slider motor: parked at the upper limit after 10 s', jointDistance(J, B, j) - 2, -1e-6, 1e-6, 'm');
  check('slider motor: no chatter against the limit (max |v| 5..10 s)', vPark, 0, 1e-9, 'm/s');
}

// ================================================================================== 13. review fixes: regressions
/** Substep with `iters` joint position passes (iteration 0, 1, ... as DynamicsWorld's positionIterations). */
function substepIter(B, J, iters) {
  for (let id = 0; id < B.count; id++) if (B.alive[id]) integrateBody(B, id, H, 0, GY, 0);
  for (let it = 0; it < iters; it++) solveJointsPositions(J, B, H, it);
  for (let id = 0; id < B.count; id++) if (B.alive[id]) velocityFromPositions(B, id, H);
  solveJointsVelocities(J, B, H);
}
{
  // mass ratio 1000 (10 links of 0.1 kg, 100 kg bob) falling from horizontal: the trust region keeps it bounded
  // (before: total gap 2.5 m, |w| 1.4e6 rad/s); exact up to ratio ~300, transient centimetre gaps at ~1000
  const B = new BodyStore(16), J = new JointStore(16);
  const ids = buildChain(B, J, 10, 0.1, 0.1, 5);
  const bob = addBox(B, 100, 0.1, 0.1, 0.1, 1.05, 5, 0);
  J.add({ type: JOINT.BALL, bodyA: bob, bodyB: ids[9], anchorA: [-0.05, 0, 0] }, B);
  let wMax = 0, gMax = 0, mm = 0;
  run(B, J, STEPS, () => {
    let gsum = 0;
    for (let j = 0; j < J.count; j++) gsum += jointDistance(J, B, j);
    for (let id = 0; id < B.count; id++) wMax = Math.max(wMax, Math.hypot(B.angVel[id * 3], B.angVel[id * 3 + 1], B.angVel[id * 3 + 2]));
    gMax = Math.max(gMax, gsum); if (gsum < 1e-3) mm++;
  });
  check('mass ratio 1000: no blow-up (max |w| of any body)', wMax, 0, 1000, 'rad/s');
  check('mass ratio 1000: largest total gap (transient)', gMax, 0, 0.2, 'm');
  check('mass ratio 1000: recovers (substeps with total gap < 1 mm)', mm / STEPS, 0.5, 1);
  check('mass ratio 1000: no NaN', finiteState(B));
  note(`mass ratio 1000: max total gap ${(gMax * 100).toFixed(1)} cm, under 1 mm in ${(100 * mm / STEPS).toFixed(0)} % of substeps (outside the exact envelope, ratio <~ 300)`);
}
{
  // one-sided limits: a taut rope must not push. A hangs on a 1 m rope, B (ball-jointed above A) is thrown up and
  // drags A along, so the rope goes slack (a two-sided rope row in the same tree solve held A at exactly 1 m)
  const B = new BodyStore(4), J = new JointStore(4);
  const a = addBox(B, 1, 0.1, 0.1, 0.1, 0, 4, 0), b = addBox(B, 1, 0.1, 0.1, 0.1, 0, 4.3, 0);
  const rope = J.add({ type: JOINT.DISTANCE, bodyA: a, bodyB: WORLD_BODY, anchorB: [0, 5, 0], distance: 1, rope: true }, B);
  J.add({ type: JOINT.BALL, bodyA: b, bodyB: a, anchorA: [0, -0.15, 0] }, B);
  B.vel[b * 3 + 1] = 4;
  let minL = Infinity;
  run(B, J, 96, () => { minL = Math.min(minL, jointDistance(J, B, rope)); });
  check('one-sided: rope goes slack when the coupled body is thrown up', minL, 0, 0.95, 'm');
}
{
  // positionIterations > 1: lambdas are accumulated per row kind, so the limit's share does not count against the
  // motor budget when the twist row turns from limit (iteration 0) into motor (iterations 1, 2)
  const B = new BodyStore(4), J = new JointStore(4);
  const lever = addBox(B, 2, 1, 0.1, 0.1, 0.5, 1, 0);
  const j = J.add({ type: JOINT.HINGE, bodyA: lever, bodyB: WORLD_BODY, anchorA: [-0.5, 0, 0], axisA: [0, 0, 1], limits: [0, 1], motor: { velocity: 1, maxForce: 30 } }, B);
  B.angVel[lever * 3 + 2] = -0.5;
  integrateBody(B, lever, H, 0, GY, 0);                       // turning down: below the lower limit 0
  solveJointsPositions(J, B, H, 0);
  const kind0 = J._rowKind[j * 6 + 3], lim = J.lambdaRot[j * 3];
  solveJointsPositions(J, B, H, 1); solveJointsPositions(J, B, H, 2);
  const kind2 = J._rowKind[j * 6 + 3], motorLam = J._rowLam[j * 6 + 3], total = J.lambdaRot[j * 3];
  check('iterations: twist row is a limit at iteration 0, a motor after', kind0 * 10 + kind2, 23, 23);
  check('iterations: motor lambda excludes the limit share', Math.abs(motorLam - (total - lim)), 0, 1e-18);
  check('iterations: limit lambda one-sided (pushes up, >= 0)', lim, 0, Infinity);
  // and whole runs with 3 passes per substep stay exact: chain gaps, hinge motor speed
  const C = new BodyStore(32), K = new JointStore(32);
  buildChain(C, K, 20, 0.1, 0.1, 5);
  const m = addBox(C, 2, 1, 0.1, 0.1, 3.5, 1, 0);
  const jm = K.add({ type: JOINT.HINGE, bodyA: m, bodyB: WORLD_BODY, anchorA: [-0.5, 0, 0], axisA: [0, 0, 1], motor: { velocity: 2, maxForce: 30 } }, C);
  let gap = 0, wLo = Infinity, wHi = -Infinity;
  for (let i = 0; i < STEPS; i++) {
    substepIter(C, K, 3);
    let gs = 0; for (let q = 0; q < 20; q++) gs += jointDistance(K, C, q);
    gap = Math.max(gap, gs);
    if (i > 240) { const w = C.angVel[m * 3 + 2]; wLo = Math.min(wLo, w); wHi = Math.max(wHi, w); }
  }
  check('iterations: 3 passes per substep, chain total gap max', gap * 1000, 0, 1, 'mm');
  check('iterations: 3 passes, hinge motor speed min / max vs 2 rad/s', Math.max(Math.abs(wLo / 2 - 1), Math.abs(wHi / 2 - 1)), 0, 0.05);
  check('iterations: motor joint force finite', K.force[jm], 0, 1e4, 'N');
}
{
  // velocity motor on a moving kinematic parent: the relative speed is the motor's (start poses from prevQuat of
  // every body type; the world saves them for kinematic bodies, as this harness does)
  const B = new BodyStore(4), J = new JointStore(4);
  const car = B.alloc();
  B.type[car] = BODY.KINEMATIC; B.pos.set([0, 2, 0], car * 3); B.prevPos.set([0, 2, 0], car * 3);
  const wheel = addBox(B, 1, 0.4, 0.4, 0.1, 0, 2, 0);
  const j = J.add({ type: JOINT.HINGE, bodyA: wheel, bodyB: car, axisA: [0, 0, 1], motor: { velocity: 2, maxForce: 100 } }, B);
  const wk = 1, dq = axisAngleQuat(0, 0, 1, wk * H);       // the carrier turns at 1 rad/s about z
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < STEPS; i++) {
    B.prevQuat.set(B.quat.subarray(car * 4, car * 4 + 4), car * 4);
    quatMul(B.quat, car * 4, dq, 0, B.quat, car * 4);
    B.angVel[car * 3 + 2] = wk;
    substep(B, J);
    if (i > 240) { const rel = B.angVel[wheel * 3 + 2] - wk; lo = Math.min(lo, rel); hi = Math.max(hi, rel); }
  }
  check('kinematic parent: relative motor speed min / 2 rad/s', lo / 2, 0.95, 1.05);
  check('kinematic parent: relative motor speed max / 2 rad/s', hi / 2, 0.95, 1.05);
  check('kinematic parent: hinge angle tracks (anchor gap)', jointDistance(J, B, j), 0, 1e-6, 'm');
}
{
  // collideConnected: J.connected(a, b) for the contact pair filter
  const B = new BodyStore(8), J = new JointStore(8);
  const p = addBox(B, 1, 0.1, 0.1, 0.1, 0, 3, 0), q = addBox(B, 1, 0.1, 0.1, 0.1, 0.2, 3, 0), r = addBox(B, 1, 0.1, 0.1, 0.1, 0.4, 3, 0);
  const j1 = J.add({ type: JOINT.BALL, bodyA: q, bodyB: p, anchorA: [-0.1, 0, 0] }, B);
  const j2 = J.add({ type: JOINT.BALL, bodyA: r, bodyB: q, anchorA: [-0.1, 0, 0], collideConnected: true }, B);
  J.add({ type: JOINT.BALL, bodyA: p, bodyB: WORLD_BODY }, B);
  const before = J.connected(p, q) && J.connected(q, p) && !J.connected(q, r) && !J.connected(p, r) && !J.connected(p, WORLD_BODY);
  J.remove(j1);
  const afterRemove = !J.connected(p, q);
  const j3 = J.add({ type: JOINT.BALL, bodyA: q, bodyB: p, anchorA: [-0.1, 0, 0], breakForce: 0.01 }, B);
  run(B, J, 4);
  const afterBreak = J.broken[j3] === 1 && !J.connected(p, q);
  check('connected(): pairs of collideConnected-false joints, either order', before);
  check('connected(): dropped on remove and on break', afterRemove && afterBreak);
  const lookups = (n) => { let hits = 0; for (let i = 0; i < n; i++) if (J.connected(i & 7, (i >> 3) & 7)) hits++; return hits; };
  lookups(1e6); // warm-up (a top-level loop would run unoptimised and box)
  const a0 = v8.getHeapStatistics().total_allocated_bytes;
  lookups(1e6);
  // one boxed double per lookup would be 16 B; V8's counter shows ~0.1-0.3 B per lookup of background noise
  check('connected(): bytes counted per lookup (V8 noise only)', (v8.getHeapStatistics().total_allocated_bytes - a0) / 1e6, 0, 2, 'B');
  void j2;
}
{
  // a zero-length distance joint acts as a ball joint
  const B = new BodyStore(4), J = new JointStore(4);
  const bob = addBox(B, 1, 0.1, 0.1, 0.1, 0.7, 3, 0);
  const j = J.add({ type: JOINT.DISTANCE, bodyA: bob, bodyB: WORLD_BODY, anchorA: [-0.7, 0, 0], anchorB: [0, 3, 0], distance: 0 }, B);
  let gap = 0;
  run(B, J, STEPS, () => { gap = Math.max(gap, jointDistance(J, B, j)); });
  check('distance 0: behaves as a ball joint (max anchor gap)', gap, 0, 1e-6, 'm');
  check('distance 0: the bob swings (y dropped)', 3 - B.pos[bob * 3 + 1], 0.1, 1.4, 'm');
}

// ================================================================================== 14. determinism and allocation
function scene() {
  const B = new BodyStore(64), J = new JointStore(64);
  buildChain(B, J, 20, 0.1, 0.1, 5);
  const a = addBox(B, 1, 0.4, 0.2, 0.2, 3.5, 3, 0, axisAngleQuat(0, 0, 1, 0.3));
  const b = addBox(B, 3, 0.4, 0.3, 0.6, 3.85, 3.1, 0.05, axisAngleQuat(1, 1, 0, 0.4));
  J.add({ type: JOINT.BALL, bodyA: a, bodyB: WORLD_BODY, anchorA: [-0.5, 0.2, 0] }, B);
  J.add({ type: JOINT.FIXED, bodyA: b, bodyB: a }, B);
  const h = addBox(B, 2, 1, 0.1, 0.1, 6.5, 1, 0);
  J.add({ type: JOINT.HINGE, bodyA: h, bodyB: WORLD_BODY, anchorA: [-0.5, 0, 0], axisA: [0, 0, 1], limits: [-1, 2], motor: { velocity: 2, maxForce: 30 }, damping: 0.5 }, B);
  const s = addBox(B, 1, 0.2, 0.3, 0.2, 9, 2, 0);
  J.add({ type: JOINT.SLIDER, bodyA: s, bodyB: WORLD_BODY, anchorA: [0.1, 0.15, -0.05], axisA: [1, 1, 0], limits: [-1, 0.5], motor: { velocity: 0.5, maxForce: 20 } }, B);
  const r = addBox(B, 1, 0.1, 0.1, 0.1, 12.3, 4.6, 0);
  J.add({ type: JOINT.DISTANCE, bodyA: r, bodyB: WORLD_BODY, anchorB: [12, 5, 0], distance: 1, rope: true, damping: 1 }, B);
  const d = addBox(B, 1, 0.1, 0.6, 0.1, 15, 4.7, 0);
  J.add({ type: JOINT.D6, bodyA: d, bodyB: WORLD_BODY, anchorA: [0, 0.3, 0], swing: 0.5, twist: [-0.2, 0.2], linear: [true, [-0.1, 0.1], true] }, B);
  B.vel.set([4, 0, 2], d * 3); B.angVel.set([0, 6, 0], d * 3); B.vel.set([0, 0, 2], a * 3); B.vel.set([0, 0, 2], b * 3);
  return { B, J };
}
const stateHash = (B) => hashF64Array(B.quat, hashF64Array(B.pos, undefined, 0, B.count * 3), 0, B.count * 4);
{
  const r1 = scene(), r2 = scene();
  run(r1.B, r1.J, STEPS); run(r2.B, r2.J, STEPS);
  const s1 = stateHash(r1.B), s2 = stateHash(r2.B);
  check('determinism: two identical 10 s runs hash-equal (pos + quat)', s1 === s2);
  check('determinism: scene state finite', finiteState(r1.B));
  note(`state hash ${s1.toString(16)}`);

  let gc = globalThis.gc;
  if (!gc) { try { v8.setFlagsFromString('--expose-gc'); gc = vm.runInNewContext('gc'); } catch { gc = null; } }
  const exact = v8.getHeapStatistics().total_allocated_bytes !== undefined; // Node >= 26
  const { B, J } = scene();
  const N = 4800, alloc = () => (exact ? v8.getHeapStatistics().total_allocated_bytes : 0);
  run(B, J, 3 * N); // warm-up: JIT tiers settle (their code and feedback are allocated here, once)
  performance.now(); alloc(); // both initialise lazily on first use: keep that out of the measured window
  if (gc) gc();
  const h0 = process.memoryUsage().heapUsed;
  run(B, J, N);
  if (gc) gc();
  const h1 = process.memoryUsage().heapUsed;
  run(B, J, N);
  if (gc) gc();
  const h2 = process.memoryUsage().heapUsed;
  run(B, J, N); // settle after the forced gc, then count allocation and time over one more run
  const a0 = alloc(), t0 = performance.now();
  run(B, J, N);
  const ms = performance.now() - t0, a1 = alloc();
  check('alloc: retained heap growth, run 1 (4800 substeps, gc before/after)', (h1 - h0) / 1024, -Infinity, 4, 'KB');
  check('alloc: retained heap growth, run 2', (h2 - h1) / 1024, -Infinity, 4, 'KB');
  // V8's allocation counter shows 30-90 B/substep of background noise here even though the sampling heap profiler
  // attributes ~0 B to the engine; one boxed double per joint and substep would already be 25 x 16 = 400 B
  if (exact) check('alloc: bytes counted per substep (26 bodies, 25 joints; V8 noise ~30-90)', (a1 - a0) / N, 0, 128, 'B');
  else note('alloc: Node < 26, total_allocated_bytes unavailable; only the retained heap growth is checked');
  if (!gc) note('alloc: no gc available: retained heap growth measured without gc');
  note(`perf: ${(ms / N * 1000).toFixed(1)} us per substep for 26 bodies / 25 joints (${(ms / N / 25 * 1000).toFixed(2)} us per joint incl. integration)`);
}

console.log(results.join('\n'));
console.log(failures ? `\n${failures} check(s) FAILED` : '\nAll dynamics joint checks passed.');
process.exit(failures ? 1 : 0);
