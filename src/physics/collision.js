// Pure collision helpers on THREE math (Node-safe, no scene graph, no allocation per call).
//   obbVsAabb        separating-axis test between an oriented box and an axis-aligned box (15 axes),
//                    returns penetration depth + minimum-translation direction for the OBB
//   sweepToContact   sub-stepped sweep: move a body back to the first penetrating sub-position
//   bounceVelocity   restitution + Coulomb friction response against a contact normal
//   rayAabb          slab test (optionally inflated by a radius)
//   sphereCastAabb   exact swept sphere vs box (faces, edges, corners)
//   closestPointOnAabb
// Boxes are plain objects { minX, minY, maxX, maxY, minZ, maxZ, kind? } as stored by ColliderGrid.
import { Vector3 } from 'three';

// ---- OBB vs AABB (SAT) ---------------------------------------------------------------------------------
const _sat = { best: 0, depth: 0, out: null, allowDown: false, horizOnly: false, Tx: 0, Ty: 0, Tz: 0, hbx: 0, hby: 0, hbz: 0, ax: null, h0: 0, h1: 0, h2: 0 };

// one SAT axis; returns false when the boxes are separated along it
function satAxis(Lx, Ly, Lz, bias) {
  const len = Math.hypot(Lx, Ly, Lz);
  if (len < 1e-6) return true;
  Lx /= len; Ly /= len; Lz /= len;
  const ax = _sat.ax;
  const rA = _sat.h0 * Math.abs(ax[0].x * Lx + ax[0].y * Ly + ax[0].z * Lz)
    + _sat.h1 * Math.abs(ax[1].x * Lx + ax[1].y * Ly + ax[1].z * Lz)
    + _sat.h2 * Math.abs(ax[2].x * Lx + ax[2].y * Ly + ax[2].z * Lz);
  const rB = _sat.hbx * Math.abs(Lx) + _sat.hby * Math.abs(Ly) + _sat.hbz * Math.abs(Lz);
  const d = _sat.Tx * Lx + _sat.Ty * Ly + _sat.Tz * Lz;
  const overlap = rA + rB - Math.abs(d);
  if (overlap <= 0) return false;
  const sg = d >= 0 ? 1 : -1;
  if (!_sat.allowDown && sg * Ly < -0.5) return true; // never push the body down (into the ground)
  if (_sat.horizOnly && Math.abs(Ly) > 0.5) return true; // sideways-only response (e.g. shoved by traffic)
  if (overlap * bias < _sat.best) { _sat.best = overlap * bias; _sat.depth = overlap; _sat.out.set(Lx * sg, Ly * sg, Lz * sg); }
  return true;
}

/**
 * Separating-axis test between an oriented box and an axis-aligned box.
 * @param {Vector3} center   OBB centre (world)
 * @param {Vector3[]} axes   OBB unit axes [x, y, z] (world)
 * @param {number[]} half    OBB half extents [hx, hy, hz]
 * @param {number} radius    OBB bounding-sphere radius (hypot of half extents), for the quick reject
 * @param {object} b         AABB { minX, minY, maxX, maxY, minZ, maxZ }
 * @param {Vector3} outN     receives the push direction for the OBB (minimum translation axis)
 * @param {boolean} [allowDown=true]  allow push directions pointing down
 * @param {boolean} [horizOnly=false] ignore near-vertical push directions
 * @returns {number} penetration depth (> 0 when overlapping), 0 otherwise. Face axes are preferred over
 *   edge-edge axes (bias 1 / 1.02 / 1.1) so resting contacts resolve along box faces.
 */
export function obbVsAabb(center, axes, half, radius, b, outN, allowDown = true, horizOnly = false) {
  const hbx = (b.maxX - b.minX) * 0.5, hby = (b.maxY - b.minY) * 0.5, hbz = (b.maxZ - b.minZ) * 0.5;
  // quick reject with the OBB bounding sphere
  const Tx = center.x - (b.minX + hbx), Ty = center.y - (b.minY + hby), Tz = center.z - (b.minZ + hbz);
  if (Math.abs(Tx) > hbx + radius || Math.abs(Ty) > hby + radius || Math.abs(Tz) > hbz + radius) return 0;
  _sat.best = Infinity; _sat.depth = 0; _sat.out = outN;
  _sat.allowDown = allowDown; _sat.horizOnly = horizOnly;
  _sat.Tx = Tx; _sat.Ty = Ty; _sat.Tz = Tz; _sat.hbx = hbx; _sat.hby = hby; _sat.hbz = hbz;
  _sat.ax = axes; _sat.h0 = half[0]; _sat.h1 = half[1]; _sat.h2 = half[2];
  if (!satAxis(1, 0, 0, 1) || !satAxis(0, 1, 0, 1) || !satAxis(0, 0, 1, 1)) return 0;
  for (let i = 0; i < 3; i++) if (!satAxis(axes[i].x, axes[i].y, axes[i].z, 1.02)) return 0;
  for (let i = 0; i < 3; i++) {
    const u = axes[i];
    if (!satAxis(0, -u.z, u.y, 1.1) || !satAxis(u.z, 0, -u.x, 1.1) || !satAxis(-u.y, u.x, 0, 1.1)) return 0;
  }
  return _sat.best === Infinity ? 0 : _sat.depth;
}

/** OBB axes + centre from a pose: centre = position + up * centerY (body-frame box resting on the reference point). */
export function obbFromPose(position, quaternion, centerY, outAxes, outCenter) {
  outAxes[0].set(1, 0, 0).applyQuaternion(quaternion);
  outAxes[1].set(0, 1, 0).applyQuaternion(quaternion);
  outAxes[2].set(0, 0, 1).applyQuaternion(quaternion);
  if (outCenter) outCenter.copy(outAxes[1]).multiplyScalar(centerY).add(position);
  return outAxes;
}

// ---- sweeps ---------------------------------------------------------------------------------------------
const _sweepEnd = new Vector3();
/**
 * Sub-stepped sweep for fast bodies: when `pos` moved more than `maxStep` since `from`, walk from `from` to
 * the current `pos` in equal sub-steps and stop at the first sub-position where `penetrates()` is true
 * (`pos` is updated in place at each sub-step, so `penetrates` reads it). Returns true when stopped early.
 */
export function sweepToContact(pos, from, maxStep, penetrates) {
  const disp = pos.distanceTo(from);
  if (!(disp > maxStep)) return false;
  const n = Math.ceil(disp / maxStep);
  _sweepEnd.copy(pos);
  for (let k = 1; k <= n; k++) {
    pos.lerpVectors(from, _sweepEnd, k / n);
    if (penetrates()) return true;
  }
  return false;
}

// ---- contact response -----------------------------------------------------------------------------------
const _vt = new Vector3();
/**
 * Reflect the normal component of `v` (restitution `e`) and apply Coulomb friction `mu` to the tangential
 * component, for a contact with unit normal `n` (pointing out of the obstacle). Returns the impact speed
 * (closing speed along n), 0 when already separating (v unchanged).
 */
export function bounceVelocity(v, n, e, mu) {
  const vn = v.dot(n);
  if (vn >= 0) return 0;
  const impact = -vn;
  v.addScaledVector(n, -(1 + e) * vn);
  _vt.copy(v).addScaledVector(n, -v.dot(n));
  const vt = _vt.length();
  if (vt > 1e-4) v.addScaledVector(_vt, -Math.min(vt, mu * (1 + e) * impact) / vt);
  return impact;
}

// ---- rays -----------------------------------------------------------------------------------------------
/**
 * Ray vs AABB slab test. Direction (dx,dy,dz) should be unit length; distances are along it.
 * `pad` inflates the box (sphere casts: Minkowski sum with square corners - conservative at edges).
 * Returns the entry distance in [0, maxT] or Infinity; writes the entry-face normal into `outN` when given.
 * A ray starting inside the box hits at 0 with normal = -direction.
 */
let _slabExit = 0;
export function rayAabb(ox, oy, oz, dx, dy, dz, b, maxT, pad = 0, outN = null) {
  let tmin = 0, tmax = maxT, axis = -1, sign = 0;
  // x
  if (Math.abs(dx) < 1e-12) { if (ox < b.minX - pad || ox > b.maxX + pad) return Infinity; }
  else {
    const inv = 1 / dx;
    let t1 = (b.minX - pad - ox) * inv, t2 = (b.maxX + pad - ox) * inv, s = -1;
    if (t1 > t2) { const t = t1; t1 = t2; t2 = t; s = 1; }
    if (t1 > tmin) { tmin = t1; axis = 0; sign = s; }
    if (t2 < tmax) tmax = t2;
    if (tmin > tmax) return Infinity;
  }
  // y
  if (Math.abs(dy) < 1e-12) { if (oy < b.minY - pad || oy > b.maxY + pad) return Infinity; }
  else {
    const inv = 1 / dy;
    let t1 = (b.minY - pad - oy) * inv, t2 = (b.maxY + pad - oy) * inv, s = -1;
    if (t1 > t2) { const t = t1; t1 = t2; t2 = t; s = 1; }
    if (t1 > tmin) { tmin = t1; axis = 1; sign = s; }
    if (t2 < tmax) tmax = t2;
    if (tmin > tmax) return Infinity;
  }
  // z
  if (Math.abs(dz) < 1e-12) { if (oz < b.minZ - pad || oz > b.maxZ + pad) return Infinity; }
  else {
    const inv = 1 / dz;
    let t1 = (b.minZ - pad - oz) * inv, t2 = (b.maxZ + pad - oz) * inv, s = -1;
    if (t1 > t2) { const t = t1; t1 = t2; t2 = t; s = 1; }
    if (t1 > tmin) { tmin = t1; axis = 2; sign = s; }
    if (t2 < tmax) tmax = t2;
    if (tmin > tmax) return Infinity;
  }
  _slabExit = tmax;
  if (outN) {
    if (axis < 0) outN.set(-dx, -dy, -dz);
    else outN.set(axis === 0 ? sign : 0, axis === 1 ? sign : 0, axis === 2 ? sign : 0);
  }
  return tmin;
}

// distance from (x,y,z) to box b (0 inside)
function distToAabb(b, x, y, z) {
  const ex = x < b.minX ? b.minX - x : x > b.maxX ? x - b.maxX : 0;
  const ey = y < b.minY ? b.minY - y : y > b.maxY ? y - b.maxY : 0;
  const ez = z < b.minZ ? b.minZ - z : z > b.maxZ ? z - b.maxZ : 0;
  return Math.sqrt(ex * ex + ey * ey + ez * ez);
}

/**
 * Exact sphere cast against an AABB (rounded Minkowski sum: faces, edges and corners).
 * Slab test on the box inflated by r gives [t0, t1]; the distance from the moving centre to the box is convex
 * along the ray, so the first contact is found by a golden-section minimum + bisection inside that interval.
 * Returns the centre travel distance in [0, maxT] or Infinity; writes the contact normal into `outN`
 * (centre minus closest box point; -direction when the centre starts inside the box).
 */
export function sphereCastAabb(ox, oy, oz, dx, dy, dz, b, maxT, r, outN = null) {
  const t0 = rayAabb(ox, oy, oz, dx, dy, dz, b, maxT, r, null);
  if (t0 === Infinity) return Infinity;
  const f = (t) => distToAabb(b, ox + dx * t, oy + dy * t, oz + dz * t) - r;
  let th = t0;
  if (f(t0) > 1e-7) {
    // edge/corner region: the inflated box's square corners over-estimate; search [t0, exit of inflated box]
    const t1 = _slabExit;
    const gr = 0.6180339887498949;
    let a = t0, c = t1, x1 = c - gr * (c - a), x2 = a + gr * (c - a), f1 = f(x1), f2 = f(x2);
    for (let k = 0; k < 48 && c - a > 1e-6; k++) {
      if (f1 < f2) { c = x2; x2 = x1; f2 = f1; x1 = c - gr * (c - a); f1 = f(x1); }
      else { a = x1; x1 = x2; f1 = f2; x2 = a + gr * (c - a); f2 = f(x2); }
    }
    const tm = (a + c) * 0.5;
    if (f(tm) > 1e-7) return Infinity;
    let lo = t0, hi = tm;
    for (let k = 0; k < 48 && hi - lo > 1e-7; k++) { const m = (lo + hi) * 0.5; if (f(m) > 0) lo = m; else hi = m; }
    th = hi;
  }
  if (outN) {
    const cx = ox + dx * th, cy = oy + dy * th, cz = oz + dz * th;
    const nx = cx - (cx < b.minX ? b.minX : cx > b.maxX ? b.maxX : cx);
    const ny = cy - (cy < b.minY ? b.minY : cy > b.maxY ? b.maxY : cy);
    const nz = cz - (cz < b.minZ ? b.minZ : cz > b.maxZ ? b.maxZ : cz);
    const l = Math.sqrt(nx * nx + ny * ny + nz * nz);
    if (l > 1e-9) outN.set(nx / l, ny / l, nz / l); else outN.set(-dx, -dy, -dz);
  }
  return th;
}

/** Closest point of box `b` to point `p` (written to `out`). */
export function closestPointOnAabb(b, p, out) {
  return out.set(
    p.x < b.minX ? b.minX : p.x > b.maxX ? b.maxX : p.x,
    p.y < b.minY ? b.minY : p.y > b.maxY ? b.maxY : p.y,
    p.z < b.minZ ? b.minZ : p.z > b.maxZ ? b.maxZ : p.z,
  );
}

/** True when two AABBs overlap (touching counts as overlap). */
export function aabbOverlap(a, b) {
  return a.minX <= b.maxX && a.maxX >= b.minX && a.minY <= b.maxY && a.maxY >= b.minY && a.minZ <= b.maxZ && a.maxZ >= b.minZ;
}
