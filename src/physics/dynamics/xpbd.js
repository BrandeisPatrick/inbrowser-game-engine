// XPBD primitives of the dynamics world (pure JS, deterministic, Node-safe, no THREE): substep integration, velocity
// recovery, generalized inverse masses and the positional / angular corrections that the contact solver and the joints
// (joints.js) are built from. Reference: Müller, Macklin, Chentanez, Jeschke, Kim 2020, "Detailed Rigid Body Simulation
// with Extended Position Based Dynamics" (SCA 2020), sections 3.1-3.5 and Algorithm 2.
//
//   for (let s = 0; s < substeps; s++) {                  // h = dt / substeps
//     for (id ...) integrateBody(B, id, h, 0, -9.81, 0);   // save the pose, predict x and q from v and w
//     ... contacts: lambdaN += applyPositional(B, a, b, nx, ny, nz, depth, rax, ..., rbz, 0, h, lambdaN);
//     solveJointsPositions(J, B, h);                       // joints.js
//     for (id ...) velocityFromPositions(B, id, h);        // v and w from the pose change
//     ... velocity pass: restitution / friction (pointVelocity, applyVelocityImpulse), solveJointsVelocities
//   }
//
// API. Every function takes a BodyStore `B` (state.js) and body ids; vectors are passed as scalar arguments and
// written to caller-owned arrays; nothing allocates (but V8 boxes a computed double passed to a call it does not
// inline: hot callers keep such arguments already boxed or pass only integers and arrays, as joints.js does with
// worldInvInertiaMatrix and displaceBodyFrom). An id < 0 (WORLD_BODY) means "no body". Only a BODY.DYNAMIC
// body with invMass > 0 is "movable": anything else (static, kinematic, WORLD_BODY) has zero generalized inverse mass
// and is never moved by a correction or an impulse (`isMovable`).
//   integrateBody(B, id, h, gx, gy, gz)                     movable bodies only (no-op otherwise)
//   velocityFromPositions(B, id, h)                         movable bodies only (kinematic w / v are the world's job)
//   generalizedInverseMass(B, id, rx, ry, rz, nx, ny, nz) -> w      angularInverseMass(B, id, nx, ny, nz) -> w
//   applyPositional(B, a, b, nx, ny, nz, c, rax, ray, raz, rbx, rby, rbz, compliance, h, lambda) -> dLambda
//   applyAngular(B, a, b, qx, qy, qz, compliance, h, lambda) -> dLambda
//   applyPositionalImpulse(B, a, b, px, py, pz, rax, ray, raz, rbx, rby, rbz)   position-level p (kg m), +a / -b
//   applyAngularImpulse(B, a, b, px, py, pz)                                      position-level (kg m^2 rad), +a / -b
//   applyVelocityImpulse(B, a, b, px, py, pz, rax, ray, raz, rbx, rby, rbz)     velocity-level p (N s), +a / -b
//   applyAngularVelocityImpulse(B, a, b, px, py, pz)                              velocity-level (N m s), +a / -b
//   displaceBody(B, id, dx, dy, dz, wx, wy, wz)       x += d; q += 1/2 [w, 0] (x) q, normalised (movable bodies only)
//   displaceBodyFrom(B, id, d, o)                     the same with (dx dy dz wx wy wz) read from d[o .. o + 6]
//   pointVelocity(B, id, rx, ry, rz, out3)            v + w x r (zero for WORLD_BODY; stored v / w otherwise)
//   worldInvInertiaApply(B, id, vx, vy, vz, out3)     I_w^-1 v = R I_b^-1 R^T v (zero when not movable)
//   worldInertiaApply(B, id, vx, vy, vz, out3)        I_w v = R I_b R^T v (zero when not movable or I_b^-1 singular)
//   worldInvInertiaMatrix(B, id, out9, o = 0)         R I_b^-1 R^T, row-major (zeros when not movable)
//   quatMul(out, oo, a, ao, b, bo)  out = a (x) b      quatConj(out, oo, a, ao)      quatNormalize(q, o)
//   rotate(q, qo, vx, vy, vz, out, oo)  out = q v q*   rotateInverse(q, qo, vx, vy, vz, out, oo)  out = q* v q
//   isMovable(B, id)
// `r` is always the WORLD lever arm from the body origin (its centre of mass) to the constraint point.
//
// Integration (paper Algorithm 2). x_prev = x; v += h (g + f m^-1) (g omitted for FLAG.NO_GRAVITY); x += h v;
// q_prev = q; w += h I_w^-1 (tau - w_m x I_w w_m); q += 1/2 h [w, 0] (x) q; q = q / |q|. Damping is the solver's.
// Gyroscopic term: w_m is evaluated at the step midpoint, w_m = (w + w')/2 (implicit midpoint; the explicit choice
// w_m = w is the paper's). Both are first-order accurate, but explicit Euler grows |L| by the factor
// sqrt(1 + h^2 |w x L|^2 / |L|^2) every step (|L'|^2 = |L|^2 + h^2 |w x L|^2 since w x L is perpendicular to L): 0.9 %
// over 1000 substeps of the test's box tumbling at 6 rad/s (h = 1/480), while the midpoint rule conserves both
// quadratic invariants of Euler's equations, |L| and the rotational energy, exactly (1e-13 measured). It is solved in
// the body frame of q (where I_b is constant) by fixed-point iteration, w'_b = w_b + h I_b^-1 (tau_b - w_m x I_b w_m),
// starting from the explicit value and stopping at a relative change of 1e-13 (~7 iterations at 6 rad/s, at most 8;
// a diverging iteration, h |w| > ~1, falls back to the explicit value). The quaternion step is a rotation about w',
// which leaves w' unchanged, so |L| = |I_b R^T w| carries over exactly between steps. Bodies with an isotropic tensor
// (spheres, cubes) skip all of it (w x I w = 0).
//
// Velocity recovery (tan form). v = (x - x_prev) / h; dq = q (x) q_prev*; w = 2 dq.xyz / (h dq.w), |dq.w| clamped
// to >= 0.25 (~150 deg per substep). Dividing by dq.w (and so
// flipping the sign when dq.w < 0, the shorter of the two equivalent rotations) makes this the exact inverse of the
// integrator: its normalised update q' ~ q + 1/2 h [w, 0] (x) q gives dq ~ (1/2 h w, 1), so integrate -> recover
// returns w to rounding (1e-13 rad/s). The paper's 2 dq.xyz / h (= 2 sin(theta / 2) / h) is low by the factor
// cos(theta / 2) = (1 + h^2 |w|^2 / 4)^(-1/2), 2e-6 at w = 2 rad/s, h = 1/480.
//
// Generalized inverse mass of body i at lever r along unit n: w_i = m_i^-1 + (r x n)^T I_w,i^-1 (r x n); about a
// unit axis n (angular): w_i = n^T I_w,i^-1 n. I_w^-1 = R I_b^-1 R^T with R from q.
//
// Positional correction (paper eq. 2-5). The constraint is C = (p_a - p_b) . n = c (the signed violation along n,
// p_a / p_b the constraint points on a / b); the correction drives it to zero:
//   alpha~ = compliance / h^2;  dLambda = (-c - alpha~ lambda) / (w_a + w_b + alpha~);  p = dLambda n
//   x_a += p m_a^-1;  q_a += 1/2 [I_a^-1 (r_a x p), 0] (x) q_a;  x_b -= p m_b^-1;  q_b -= 1/2 [I_b^-1 (r_b x p), 0] (x) q_b
// (quaternions renormalised). To first order C changes by -c (w_a + w_b) / (w_a + w_b + alpha~). With dLambda < 0, a
// moves along -n and b along +n. Contacts: normal n from a to b and penetration d > 0 give c = d. The caller
// accumulates lambda (lambda += dLambda); the constraint force is lambda n / h^2 (N).
// Angular correction (paper eq. 11-12). (qx, qy, qz) = theta n is the rotation of a relative to b that must be undone
// (axis times angle, world space):
//   dLambda = (-theta - alpha~ lambda) / (w_a + w_b + alpha~);  q_a += 1/2 [I_a^-1 (dLambda n), 0] (x) q_a;  q_b -= ...
// which rotates a by -theta w_a / w and b by +theta w_b / w about n (alpha~ = 0); the torque is lambda n / h^2 (N m).
// Velocity impulses are the same with velocities: v_a += p m_a^-1, w_a += I_a^-1 (r_a x p), the opposite for b.
//
// Determinism: + - * / Math.sqrt and comparisons only (no trig at all in this file); fixed iteration order; module
// scratch is fully rewritten before it is read.
import { BODY, FLAG } from './state.js';

const DYN = BODY.DYNAMIC;
const GYRO_ITERS = 8;
const _t = new Float64Array(3); // scratch for I_w^-1 products

// ---- quaternion / vector helpers (typed arrays with offsets, alias-safe) -------------------------------------------

/** out[oo..] = a[ao..] (x) b[bo..] (Hamilton product, x y z w). */
export function quatMul(out, oo, a, ao, b, bo) {
  const ax = a[ao], ay = a[ao + 1], az = a[ao + 2], aw = a[ao + 3];
  const bx = b[bo], by = b[bo + 1], bz = b[bo + 2], bw = b[bo + 3];
  out[oo] = aw * bx + ax * bw + ay * bz - az * by;
  out[oo + 1] = aw * by + ay * bw + az * bx - ax * bz;
  out[oo + 2] = aw * bz + az * bw + ax * by - ay * bx;
  out[oo + 3] = aw * bw - ax * bx - ay * by - az * bz;
}

/** out[oo..] = conjugate of a[ao..]. */
export function quatConj(out, oo, a, ao) {
  out[oo] = -a[ao]; out[oo + 1] = -a[ao + 1]; out[oo + 2] = -a[ao + 2]; out[oo + 3] = a[ao + 3];
}

/** Normalise q[o..o+4] in place (a zero quaternion becomes the identity). */
export function quatNormalize(q, o = 0) {
  const x = q[o], y = q[o + 1], z = q[o + 2], w = q[o + 3];
  const l2 = x * x + y * y + z * z + w * w;
  if (l2 > 0) {
    const s = 1 / Math.sqrt(l2);
    q[o] = x * s; q[o + 1] = y * s; q[o + 2] = z * s; q[o + 3] = w * s;
  } else { q[o] = 0; q[o + 1] = 0; q[o + 2] = 0; q[o + 3] = 1; }
}

/** out[oo..oo+3] = q v q* (q a unit quaternion at q[qo..]). */
export function rotate(q, qo, vx, vy, vz, out, oo = 0) {
  const x = q[qo], y = q[qo + 1], z = q[qo + 2], w = q[qo + 3];
  const tx = 2 * (y * vz - z * vy), ty = 2 * (z * vx - x * vz), tz = 2 * (x * vy - y * vx);
  out[oo] = vx + w * tx + (y * tz - z * ty);
  out[oo + 1] = vy + w * ty + (z * tx - x * tz);
  out[oo + 2] = vz + w * tz + (x * ty - y * tx);
}

/** out[oo..oo+3] = q* v q (the inverse rotation: world -> body frame). */
export function rotateInverse(q, qo, vx, vy, vz, out, oo = 0) {
  const x = -q[qo], y = -q[qo + 1], z = -q[qo + 2], w = q[qo + 3];
  const tx = 2 * (y * vz - z * vy), ty = 2 * (z * vx - x * vz), tz = 2 * (x * vy - y * vx);
  out[oo] = vx + w * tx + (y * tz - z * ty);
  out[oo + 1] = vy + w * ty + (z * tx - x * tz);
  out[oo + 2] = vz + w * tz + (x * ty - y * tx);
}

/** True when `id` is a BODY.DYNAMIC body with invMass > 0 (the only bodies corrections and impulses move). */
export function isMovable(B, id) { return id >= 0 && B.type[id] === DYN && B.invMass[id] > 0; }

// ---- inertia --------------------------------------------------------------------------------------------------------

/** out3 = I_w^-1 v = R I_b^-1 R^T v for body id; zero when the body is not movable. */
export function worldInvInertiaApply(B, id, vx, vy, vz, out3) {
  if (!(id >= 0 && B.type[id] === DYN && B.invMass[id] > 0)) { out3[0] = 0; out3[1] = 0; out3[2] = 0; return; }
  const I = B.invInertia, k = id * 9;
  const i0 = I[k], i1 = I[k + 1], i2 = I[k + 2], i3 = I[k + 3], i4 = I[k + 4], i5 = I[k + 5], i6 = I[k + 6], i7 = I[k + 7], i8 = I[k + 8];
  if (i1 === 0 && i2 === 0 && i3 === 0 && i5 === 0 && i6 === 0 && i7 === 0 && i0 === i4 && i0 === i8) {
    out3[0] = i0 * vx; out3[1] = i0 * vy; out3[2] = i0 * vz; return; // isotropic: rotation-invariant
  }
  const q = B.quat, o = id * 4, x = q[o], y = q[o + 1], z = q[o + 2], w = q[o + 3];
  // u = R^T v (rotate by the conjugate: -x, -y, -z)
  let tx = 2 * (z * vy - y * vz), ty = 2 * (x * vz - z * vx), tz = 2 * (y * vx - x * vy);
  const ux = vx + w * tx + (z * ty - y * tz), uy = vy + w * ty + (x * tz - z * tx), uz = vz + w * tz + (y * tx - x * ty);
  const mx = i0 * ux + i1 * uy + i2 * uz, my = i3 * ux + i4 * uy + i5 * uz, mz = i6 * ux + i7 * uy + i8 * uz;
  tx = 2 * (y * mz - z * my); ty = 2 * (z * mx - x * mz); tz = 2 * (x * my - y * mx);
  out3[0] = mx + w * tx + (y * tz - z * ty);
  out3[1] = my + w * ty + (z * tx - x * tz);
  out3[2] = mz + w * tz + (x * ty - y * tx);
}

/** out3 = I_w v = R I_b R^T v (I_b = inverse of the stored body-frame tensor); zero when not movable or singular. */
export function worldInertiaApply(B, id, vx, vy, vz, out3) {
  out3[0] = 0; out3[1] = 0; out3[2] = 0;
  if (!(id >= 0 && B.type[id] === DYN && B.invMass[id] > 0)) return;
  const I = B.invInertia, k = id * 9;
  const i0 = I[k], i1 = I[k + 1], i2 = I[k + 2], i3 = I[k + 3], i4 = I[k + 4], i5 = I[k + 5], i6 = I[k + 6], i7 = I[k + 7], i8 = I[k + 8];
  const c0 = i4 * i8 - i5 * i7, c1 = i5 * i6 - i3 * i8, c2 = i3 * i7 - i4 * i6;
  const det = i0 * c0 + i1 * c1 + i2 * c2;
  if (!(det > 0) || det === Infinity) return;
  const s = 1 / det;
  const I0 = c0 * s, I1 = (i2 * i7 - i1 * i8) * s, I2 = (i1 * i5 - i2 * i4) * s;
  const I3 = c1 * s, I4 = (i0 * i8 - i2 * i6) * s, I5 = (i2 * i3 - i0 * i5) * s;
  const I6 = c2 * s, I7 = (i1 * i6 - i0 * i7) * s, I8 = (i0 * i4 - i1 * i3) * s;
  const q = B.quat, o = id * 4, x = q[o], y = q[o + 1], z = q[o + 2], w = q[o + 3];
  let tx = 2 * (z * vy - y * vz), ty = 2 * (x * vz - z * vx), tz = 2 * (y * vx - x * vy);
  const ux = vx + w * tx + (z * ty - y * tz), uy = vy + w * ty + (x * tz - z * tx), uz = vz + w * tz + (y * tx - x * ty);
  const mx = I0 * ux + I1 * uy + I2 * uz, my = I3 * ux + I4 * uy + I5 * uz, mz = I6 * ux + I7 * uy + I8 * uz;
  tx = 2 * (y * mz - z * my); ty = 2 * (z * mx - x * mz); tz = 2 * (x * my - y * mx);
  out3[0] = mx + w * tx + (y * tz - z * ty);
  out3[1] = my + w * ty + (z * tx - x * tz);
  out3[2] = mz + w * tz + (x * ty - y * tx);
}

/** out9[o..o+9] = R I_b^-1 R^T (row-major, symmetric); zeros when the body is not movable. */
export function worldInvInertiaMatrix(B, id, out9, o = 0) {
  if (!(id >= 0 && B.type[id] === DYN && B.invMass[id] > 0)) { out9.fill(0, o, o + 9); return; }
  const I = B.invInertia, k = id * 9;
  const a0 = I[k], a1 = I[k + 1], a2 = I[k + 2], a3 = I[k + 3], a4 = I[k + 4], a5 = I[k + 5], a6 = I[k + 6], a7 = I[k + 7], a8 = I[k + 8];
  if (a1 === 0 && a2 === 0 && a3 === 0 && a5 === 0 && a6 === 0 && a7 === 0 && a0 === a4 && a0 === a8) {
    out9[o] = a0; out9[o + 1] = 0; out9[o + 2] = 0; out9[o + 3] = 0; out9[o + 4] = a0; out9[o + 5] = 0;
    out9[o + 6] = 0; out9[o + 7] = 0; out9[o + 8] = a0;
    return;
  }
  const q = B.quat, qo = id * 4, x = q[qo], y = q[qo + 1], z = q[qo + 2], w = q[qo + 3];
  const r0 = 1 - 2 * (y * y + z * z), r1 = 2 * (x * y - z * w), r2 = 2 * (x * z + y * w);
  const r3 = 2 * (x * y + z * w), r4 = 1 - 2 * (x * x + z * z), r5 = 2 * (y * z - x * w);
  const r6 = 2 * (x * z - y * w), r7 = 2 * (y * z + x * w), r8 = 1 - 2 * (x * x + y * y);
  // T = R A
  const t0 = r0 * a0 + r1 * a3 + r2 * a6, t1 = r0 * a1 + r1 * a4 + r2 * a7, t2 = r0 * a2 + r1 * a5 + r2 * a8;
  const t3 = r3 * a0 + r4 * a3 + r5 * a6, t4 = r3 * a1 + r4 * a4 + r5 * a7, t5 = r3 * a2 + r4 * a5 + r5 * a8;
  const t6 = r6 * a0 + r7 * a3 + r8 * a6, t7 = r6 * a1 + r7 * a4 + r8 * a7, t8 = r6 * a2 + r7 * a5 + r8 * a8;
  // M = T R^T (symmetric: compute the upper triangle)
  const m0 = t0 * r0 + t1 * r1 + t2 * r2, m1 = t0 * r3 + t1 * r4 + t2 * r5, m2 = t0 * r6 + t1 * r7 + t2 * r8;
  const m4 = t3 * r3 + t4 * r4 + t5 * r5, m5 = t3 * r6 + t4 * r7 + t5 * r8, m8 = t6 * r6 + t7 * r7 + t8 * r8;
  out9[o] = m0; out9[o + 1] = m1; out9[o + 2] = m2; out9[o + 3] = m1; out9[o + 4] = m4; out9[o + 5] = m5;
  out9[o + 6] = m2; out9[o + 7] = m5; out9[o + 8] = m8;
}

/** w = m^-1 + (r x n)^T I_w^-1 (r x n): inverse mass of body id at world lever r along unit n (0 when not movable). */
export function generalizedInverseMass(B, id, rx, ry, rz, nx, ny, nz) {
  if (!(id >= 0 && B.type[id] === DYN && B.invMass[id] > 0)) return 0;
  const cx = ry * nz - rz * ny, cy = rz * nx - rx * nz, cz = rx * ny - ry * nx;
  worldInvInertiaApply(B, id, cx, cy, cz, _t);
  return B.invMass[id] + cx * _t[0] + cy * _t[1] + cz * _t[2];
}

/** w = n^T I_w^-1 n: rotational inverse mass of body id about unit axis n (0 when not movable). */
export function angularInverseMass(B, id, nx, ny, nz) {
  if (!(id >= 0 && B.type[id] === DYN && B.invMass[id] > 0)) return 0;
  worldInvInertiaApply(B, id, nx, ny, nz, _t);
  return nx * _t[0] + ny * _t[1] + nz * _t[2];
}

// ---- integration ----------------------------------------------------------------------------------------------------

// q += 1/2 [w, 0] (x) q for the world rotation vector (wx, wy, wz), then normalise.
function rotateBody(q, o, wx, wy, wz) {
  const hx = 0.5 * wx, hy = 0.5 * wy, hz = 0.5 * wz;
  const qx = q[o], qy = q[o + 1], qz = q[o + 2], qw = q[o + 3];
  const x = qx + (hx * qw + hy * qz - hz * qy);
  const y = qy + (hy * qw + hz * qx - hx * qz);
  const z = qz + (hz * qw + hx * qy - hy * qx);
  const w = qw - (hx * qx + hy * qy + hz * qz);
  const s = 1 / Math.sqrt(x * x + y * y + z * z + w * w);
  q[o] = x * s; q[o + 1] = y * s; q[o + 2] = z * s; q[o + 3] = w * s;
}

// angVel += h I_w^-1 (tau - w_m x I_w w_m), w_m the midpoint angular velocity (see header).
function integrateAngularVelocity(B, id, h) {
  const p = id * 3, k = id * 9, I = B.invInertia, av = B.angVel, tq = B.torque;
  const i0 = I[k], i1 = I[k + 1], i2 = I[k + 2], i3 = I[k + 3], i4 = I[k + 4], i5 = I[k + 5], i6 = I[k + 6], i7 = I[k + 7], i8 = I[k + 8];
  const wx = av[p], wy = av[p + 1], wz = av[p + 2], tx = tq[p], ty = tq[p + 1], tz = tq[p + 2];
  if (i1 === 0 && i2 === 0 && i3 === 0 && i5 === 0 && i6 === 0 && i7 === 0 && i0 === i4 && i0 === i8) {
    av[p] = wx + h * i0 * tx; av[p + 1] = wy + h * i0 * ty; av[p + 2] = wz + h * i0 * tz; // w x I w = 0
    return;
  }
  const q = B.quat, o = id * 4, x = q[o], y = q[o + 1], z = q[o + 2], w = q[o + 3];
  // body-frame w and tau: R^T v
  let sx = 2 * (z * wy - y * wz), sy = 2 * (x * wz - z * wx), sz = 2 * (y * wx - x * wy);
  const bx = wx + w * sx + (z * sy - y * sz), by = wy + w * sy + (x * sz - z * sx), bz = wz + w * sz + (y * sx - x * sy);
  sx = 2 * (z * ty - y * tz); sy = 2 * (x * tz - z * tx); sz = 2 * (y * tx - x * ty);
  const ux = tx + w * sx + (z * sy - y * sz), uy = ty + w * sy + (x * sz - z * sx), uz = tz + w * sz + (y * sx - x * sy);
  // explicit start: d = h I^-1 (tau - w x I w) (with I = inverse of the stored tensor; skipped when singular)
  const c0 = i4 * i8 - i5 * i7, c1 = i5 * i6 - i3 * i8, c2 = i3 * i7 - i4 * i6;
  const det = i0 * c0 + i1 * c1 + i2 * c2;
  let dx, dy, dz;
  if (!(det > 0) || det === Infinity || (bx === 0 && by === 0 && bz === 0)) {
    dx = h * (i0 * ux + i1 * uy + i2 * uz); dy = h * (i3 * ux + i4 * uy + i5 * uz); dz = h * (i6 * ux + i7 * uy + i8 * uz);
  } else {
    const s = 1 / det;
    const I0 = c0 * s, I1 = (i2 * i7 - i1 * i8) * s, I2 = (i1 * i5 - i2 * i4) * s;
    const I3 = c1 * s, I4 = (i0 * i8 - i2 * i6) * s, I5 = (i2 * i3 - i0 * i5) * s;
    const I6 = c2 * s, I7 = (i1 * i6 - i0 * i7) * s, I8 = (i0 * i4 - i1 * i3) * s;
    dx = 0; dy = 0; dz = 0;
    let ex0 = 0, ey0 = 0, ez0 = 0, prevErr = Infinity;
    for (let it = 0; it < GYRO_ITERS; it++) {
      const mx = bx + 0.5 * dx, my = by + 0.5 * dy, mz = bz + 0.5 * dz;
      const lx = I0 * mx + I1 * my + I2 * mz, ly = I3 * mx + I4 * my + I5 * mz, lz = I6 * mx + I7 * my + I8 * mz;
      const gx = ux - (my * lz - mz * ly), gy = uy - (mz * lx - mx * lz), gz = uz - (mx * ly - my * lx);
      const nx = h * (i0 * gx + i1 * gy + i2 * gz), ny = h * (i3 * gx + i4 * gy + i5 * gz), nz = h * (i6 * gx + i7 * gy + i8 * gz);
      if (it === 0) { ex0 = nx; ey0 = ny; ez0 = nz; }
      const ex = nx - dx, ey = ny - dy, ez = nz - dz, err = ex * ex + ey * ey + ez * ez;
      if (it > 1 && err > prevErr) { dx = ex0; dy = ey0; dz = ez0; break; } // diverging (h |w| too large): explicit
      dx = nx; dy = ny; dz = nz; prevErr = err;
      if (err <= 1e-26 * (nx * nx + ny * ny + nz * nz)) break;
    }
  }
  const cx = bx + dx, cy = by + dy, cz = bz + dz;
  // back to world: R c
  sx = 2 * (y * cz - z * cy); sy = 2 * (z * cx - x * cz); sz = 2 * (x * cy - y * cx);
  av[p] = cx + w * sx + (y * sz - z * sy);
  av[p + 1] = cy + w * sy + (z * sx - x * sz);
  av[p + 2] = cz + w * sz + (x * sy - y * sx);
}

/**
 * Predict step of a movable body (no-op otherwise): saves prevPos / prevQuat, then v += h (g + f m^-1) (no g with
 * FLAG.NO_GRAVITY), x += h v, w += h I_w^-1 (tau - w_m x I_w w_m), q += 1/2 h [w, 0] (x) q, normalised.
 */
export function integrateBody(B, id, h, gx, gy, gz) {
  if (B.type[id] !== DYN) return;
  const im = B.invMass[id];
  if (!(im > 0)) return;
  if (B.flags[id] & FLAG.NO_GRAVITY) { gx = 0; gy = 0; gz = 0; }
  const p = id * 3, pos = B.pos, prev = B.prevPos, v = B.vel, f = B.force;
  prev[p] = pos[p]; prev[p + 1] = pos[p + 1]; prev[p + 2] = pos[p + 2];
  v[p] += h * (gx + f[p] * im); v[p + 1] += h * (gy + f[p + 1] * im); v[p + 2] += h * (gz + f[p + 2] * im);
  pos[p] += h * v[p]; pos[p + 1] += h * v[p + 1]; pos[p + 2] += h * v[p + 2];
  const o = id * 4, q = B.quat, pq = B.prevQuat;
  pq[o] = q[o]; pq[o + 1] = q[o + 1]; pq[o + 2] = q[o + 2]; pq[o + 3] = q[o + 3];
  integrateAngularVelocity(B, id, h);
  const w = B.angVel;
  rotateBody(q, o, h * w[p], h * w[p + 1], h * w[p + 2]);
}

/** v = (x - x_prev) / h; w = 2 dq.xyz / (h dq.w) with dq = q (x) q_prev* (movable bodies only). */
export function velocityFromPositions(B, id, h) {
  if (B.type[id] !== DYN || !(B.invMass[id] > 0)) return;
  const p = id * 3, pos = B.pos, prev = B.prevPos, v = B.vel;
  v[p] = (pos[p] - prev[p]) / h; v[p + 1] = (pos[p + 1] - prev[p + 1]) / h; v[p + 2] = (pos[p + 2] - prev[p + 2]) / h;
  const o = id * 4, q = B.quat, pq = B.prevQuat;
  const qx = q[o], qy = q[o + 1], qz = q[o + 2], qw = q[o + 3];
  const px = pq[o], py = pq[o + 1], pz = pq[o + 2], pw = pq[o + 3];
  const dx = qx * pw - qw * px - qy * pz + qz * py;
  const dy = qy * pw - qw * py - qz * px + qx * pz;
  const dz = qz * pw - qw * pz - qx * py + qy * px;
  let dw = qw * pw + qx * px + qy * py + qz * pz;
  // |dw| >= 0.25: a pose change beyond ~150 deg in one substep (a solver blow-up, never motion) is read as 150 deg,
  // which caps the recovered |w| at 2 tan(75 deg) / h ~ 7.5 / h instead of letting it run to 2e9 / h
  if (dw >= 0 && dw < 0.25) dw = 0.25; else if (dw < 0 && dw > -0.25) dw = -0.25;
  const s = 2 / (h * dw), w = B.angVel;
  w[p] = dx * s; w[p + 1] = dy * s; w[p + 2] = dz * s;
}

// ---- corrections and impulses ---------------------------------------------------------------------------------------

/** Pose delta of a movable body (no-op otherwise): x += (dx, dy, dz); q += 1/2 [w, 0] (x) q for the small world
 * rotation vector w, then normalised (the update every correction uses). */
export function displaceBody(B, id, dx, dy, dz, wx, wy, wz) {
  if (!(id >= 0 && B.type[id] === DYN && B.invMass[id] > 0)) return;
  const p = id * 3, pos = B.pos;
  pos[p] += dx; pos[p + 1] += dy; pos[p + 2] += dz;
  rotateBody(B.quat, id * 4, wx, wy, wz);
}

/** displaceBody with the delta (dx dy dz wx wy wz) read from d[o .. o + 6] (no double crosses the call). */
export function displaceBodyFrom(B, id, d, o) {
  if (!(id >= 0 && B.type[id] === DYN && B.invMass[id] > 0)) return;
  const p = id * 3, pos = B.pos;
  pos[p] += d[o]; pos[p + 1] += d[o + 1]; pos[p + 2] += d[o + 2];
  const q = B.quat, k = id * 4, hx = 0.5 * d[o + 3], hy = 0.5 * d[o + 4], hz = 0.5 * d[o + 5];
  const qx = q[k], qy = q[k + 1], qz = q[k + 2], qw = q[k + 3];
  const x = qx + (hx * qw + hy * qz - hz * qy), y = qy + (hy * qw + hz * qx - hx * qz);
  const z = qz + (hz * qw + hx * qy - hy * qx), w = qw - (hx * qx + hy * qy + hz * qz);
  const s = 1 / Math.sqrt(x * x + y * y + z * z + w * w);
  q[k] = x * s; q[k + 1] = y * s; q[k + 2] = z * s; q[k + 3] = w * s;
}

/** Position-level impulse p at world levers: x_a += p m_a^-1, q_a rotated by I_a^-1 (r_a x p); the opposite for b. */
export function applyPositionalImpulse(B, a, b, px, py, pz, rax, ray, raz, rbx, rby, rbz) {
  if (a >= 0 && B.type[a] === DYN && B.invMass[a] > 0) {
    const im = B.invMass[a], p3 = a * 3, pos = B.pos;
    pos[p3] += px * im; pos[p3 + 1] += py * im; pos[p3 + 2] += pz * im;
    worldInvInertiaApply(B, a, ray * pz - raz * py, raz * px - rax * pz, rax * py - ray * px, _t);
    rotateBody(B.quat, a * 4, _t[0], _t[1], _t[2]);
  }
  if (b >= 0 && B.type[b] === DYN && B.invMass[b] > 0) {
    const im = B.invMass[b], p3 = b * 3, pos = B.pos;
    pos[p3] -= px * im; pos[p3 + 1] -= py * im; pos[p3 + 2] -= pz * im;
    worldInvInertiaApply(B, b, rby * pz - rbz * py, rbz * px - rbx * pz, rbx * py - rby * px, _t);
    rotateBody(B.quat, b * 4, -_t[0], -_t[1], -_t[2]);
  }
}

/** Position-level angular impulse p: q_a rotated by I_a^-1 p, q_b by -I_b^-1 p. */
export function applyAngularImpulse(B, a, b, px, py, pz) {
  if (a >= 0 && B.type[a] === DYN && B.invMass[a] > 0) {
    worldInvInertiaApply(B, a, px, py, pz, _t);
    rotateBody(B.quat, a * 4, _t[0], _t[1], _t[2]);
  }
  if (b >= 0 && B.type[b] === DYN && B.invMass[b] > 0) {
    worldInvInertiaApply(B, b, px, py, pz, _t);
    rotateBody(B.quat, b * 4, -_t[0], -_t[1], -_t[2]);
  }
}

/**
 * XPBD positional correction of C = (p_a - p_b) . n = c (unit n, world levers r_a / r_b, accumulated lambda). Returns
 * dLambda; the caller accumulates it. See the header for the equations and the sign convention.
 */
export function applyPositional(B, a, b, nx, ny, nz, c, rax, ray, raz, rbx, rby, rbz, compliance, h, lambda) {
  const wa = generalizedInverseMass(B, a, rax, ray, raz, nx, ny, nz);
  const wb = generalizedInverseMass(B, b, rbx, rby, rbz, nx, ny, nz);
  const w = wa + wb;
  if (!(w > 0)) return 0;
  const at = compliance / (h * h);
  const dl = (-c - at * lambda) / (w + at);
  applyPositionalImpulse(B, a, b, dl * nx, dl * ny, dl * nz, rax, ray, raz, rbx, rby, rbz);
  return dl;
}

/**
 * XPBD angular correction for the rotation vector (qx, qy, qz) = theta n of a relative to b that must be undone.
 * Returns dLambda (accumulate it); see the header.
 */
export function applyAngular(B, a, b, qx, qy, qz, compliance, h, lambda) {
  const t2 = qx * qx + qy * qy + qz * qz;
  if (!(t2 > 0)) return 0;
  const th = Math.sqrt(t2), nx = qx / th, ny = qy / th, nz = qz / th;
  const w = angularInverseMass(B, a, nx, ny, nz) + angularInverseMass(B, b, nx, ny, nz);
  if (!(w > 0)) return 0;
  const at = compliance / (h * h);
  const dl = (-th - at * lambda) / (w + at);
  applyAngularImpulse(B, a, b, dl * nx, dl * ny, dl * nz);
  return dl;
}

/** Velocity-level impulse p at world levers: v_a += p m_a^-1, w_a += I_a^-1 (r_a x p); the opposite for b. */
export function applyVelocityImpulse(B, a, b, px, py, pz, rax, ray, raz, rbx, rby, rbz) {
  if (a >= 0 && B.type[a] === DYN && B.invMass[a] > 0) {
    const im = B.invMass[a], p3 = a * 3, v = B.vel, w = B.angVel;
    v[p3] += px * im; v[p3 + 1] += py * im; v[p3 + 2] += pz * im;
    worldInvInertiaApply(B, a, ray * pz - raz * py, raz * px - rax * pz, rax * py - ray * px, _t);
    w[p3] += _t[0]; w[p3 + 1] += _t[1]; w[p3 + 2] += _t[2];
  }
  if (b >= 0 && B.type[b] === DYN && B.invMass[b] > 0) {
    const im = B.invMass[b], p3 = b * 3, v = B.vel, w = B.angVel;
    v[p3] -= px * im; v[p3 + 1] -= py * im; v[p3 + 2] -= pz * im;
    worldInvInertiaApply(B, b, rby * pz - rbz * py, rbz * px - rbx * pz, rbx * py - rby * px, _t);
    w[p3] -= _t[0]; w[p3 + 1] -= _t[1]; w[p3 + 2] -= _t[2];
  }
}

/** Velocity-level angular impulse p: w_a += I_a^-1 p, w_b -= I_b^-1 p. */
export function applyAngularVelocityImpulse(B, a, b, px, py, pz) {
  if (a >= 0 && B.type[a] === DYN && B.invMass[a] > 0) {
    const p3 = a * 3, w = B.angVel;
    worldInvInertiaApply(B, a, px, py, pz, _t);
    w[p3] += _t[0]; w[p3 + 1] += _t[1]; w[p3 + 2] += _t[2];
  }
  if (b >= 0 && B.type[b] === DYN && B.invMass[b] > 0) {
    const p3 = b * 3, w = B.angVel;
    worldInvInertiaApply(B, b, px, py, pz, _t);
    w[p3] -= _t[0]; w[p3 + 1] -= _t[1]; w[p3 + 2] -= _t[2];
  }
}

/** out3 = v + w x r for body id (zero for WORLD_BODY; static / kinematic bodies report their stored v / w). */
export function pointVelocity(B, id, rx, ry, rz, out3) {
  if (id < 0) { out3[0] = 0; out3[1] = 0; out3[2] = 0; return; }
  const p = id * 3, v = B.vel, w = B.angVel, wx = w[p], wy = w[p + 1], wz = w[p + 2];
  out3[0] = v[p] + (wy * rz - wz * ry);
  out3[1] = v[p + 1] + (wz * rx - wx * rz);
  out3[2] = v[p + 2] + (wx * ry - wy * rx);
}
