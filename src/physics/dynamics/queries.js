// Scene queries against the dynamics bodies (pure JS, deterministic, Node-safe, allocation-free after warm-up).
// DynamicsWorld delegates to these; they read only the shared state (state.js BodyStore) and the shape fields of the
// shapes.js contract, so they also run against hand-built worlds { bodies, broadphase? }.
//
//   raycastBodies(world, ox, oy, oz, dx, dy, dz, maxDist, opts, outHit) -> boolean
//       nearest exact hit: broadphase.raycast candidates, then ray vs shape in the body frame: sphere (quadratic),
//       capsule (cylinder + caps), box (slab), convex (clip against the half-space planes), trimesh (Moller-Trumbore,
//       front faces only, near-first BVH traversal with an explicit stack), compound (children). opts: mask (vs
//       bodies.layer), filter(id).
//   shapeCastBodies(world, shape, px, py, pz, qx, qy, qz, qw, dx, dy, dz, maxDist, opts, outHit) -> boolean
//       sweep a shape (sphere / capsule / box / convex / compound of those) along a direction: conservative
//       advancement on the GJK distance (cores + radius margins, so spheres and capsules are exact) against each
//       candidate's convex pieces (trimesh: per triangle). Bodies are treated as frozen at their current pose.
//   overlapBodies(world, minX, minY, minZ, maxX, maxY, maxZ, outInt32, mask = -1) -> true count (tight AABB check;
//       at most outInt32.length ids written)
//   pointInBody(world, x, y, z) -> lowest id whose shape contains the point, or -1
//
// Hit records: { distance, point, normal, body, box: null, kind: 'rigid', ground: false } written into the caller's
// object (point / normal may be THREE.Vector3 or plain {x, y, z}: only .x/.y/.z are assigned). `distance` is along
// the normalised direction; a ray (or cast) that starts inside a shape hits at 0 with normal = -direction (the
// PhysicsWorld / rayAabb convention), except a trimesh: only its front faces count, so a ray starting inside a closed
// trimesh misses it (pointInBody uses crossing parity instead). Normals point out of the hit body. Ties go to the
// lower body id.
// Determinism: only + - * / Math.sqrt and comparisons; results never depend on the broadphase's candidate order
// (rays break exact ties by id, casts walk their candidates in ascending id order).
// Candidates: world.broadphase (query / raycast, which see the bodies as of its last update()); without a broadphase
// every alive body is tested (hand-built worlds, tests). On a DynamicsWorld the broadphase is refreshed in step(),
// so these queries do NOT see a body added, removed or moved (setPose) since the last step() (until DynamicsWorld
// refreshes it lazily before a query). Casts cull with
// a box sweep of the cast shape's bounds against bodies.aabb (broadphase mode) and against each triangle's AABB.
// Cost (Node 26, Apple M4, 300 bodies + 800-triangle mesh): ray ~0.9 us; 5 m capsule cast ~4 us, 50 m ~14 us.
import { SHAPE } from './state.js';

const EMPTY_OPTS = Object.freeze({});
const KIND = 'rigid';
const MAX_DIST = 1e5;

// ---- quaternion helpers (module scratch outputs) ---------------------------------------------------------------
let _rx = 0, _ry = 0, _rz = 0;
// v' = q v q^-1  (t = 2 q.xyz x v; v' = v + w t + q.xyz x t)
function rot(qx, qy, qz, qw, x, y, z) {
  const tx = 2 * (qy * z - qz * y), ty = 2 * (qz * x - qx * z), tz = 2 * (qx * y - qy * x);
  _rx = x + qw * tx + (qy * tz - qz * ty);
  _ry = y + qw * ty + (qz * tx - qx * tz);
  _rz = z + qw * tz + (qx * ty - qy * tx);
}
function invRot(qx, qy, qz, qw, x, y, z) { rot(-qx, -qy, -qz, qw, x, y, z); }
let _mx = 0, _my = 0, _mz = 0, _mw = 1;
// Hamilton product a * b
function qmul(ax, ay, az, aw, bx, by, bz, bw) {
  _mx = aw * bx + ax * bw + ay * bz - az * by;
  _my = aw * by - ax * bz + ay * bw + az * bx;
  _mz = aw * bz + ax * by - ay * bx + az * bw;
  _mw = aw * bw - ax * bx - ay * by - az * bz;
}

// ---- growable scratch ------------------------------------------------------------------------------------------
let _cand = new Int32Array(256);
function candBuf(bodies) {
  const n = bodies.capacity || bodies.count;
  if (_cand.length < n) _cand = new Int32Array(n);
  return _cand;
}
// insertion sort of ids[0..n) (candidate lists are short)
function sortIds(ids, n) {
  for (let i = 1; i < n; i++) {
    const v = ids[i];
    let j = i - 1;
    while (j >= 0 && ids[j] > v) { ids[j + 1] = ids[j]; j--; }
    ids[j + 1] = v;
  }
}
let _tri = new Int32Array(256);
function triBuf(shape) {
  const n = (shape.indices.length / 3) | 0;
  if (_tri.length < n) _tri = new Int32Array(n);
  return _tri;
}

// Local-frame bounds of a trimesh (minX, minY, minZ, maxX, maxY, maxZ), computed once per shape object.
const _bounds = new WeakMap();
/** Local AABB of a trimesh shape as Float64Array(6) (min xyz, max xyz); cached per shape object. */
export function meshBounds(shape) {
  let b = _bounds.get(shape);
  if (b) return b;
  b = new Float64Array(6);
  const P = shape.positions;
  if (!P || P.length < 3) { _bounds.set(shape, b); return b; }
  b[0] = b[3] = P[0]; b[1] = b[4] = P[1]; b[2] = b[5] = P[2];
  for (let i = 3; i + 2 < P.length; i += 3) {
    const x = P[i], y = P[i + 1], z = P[i + 2];
    if (x < b[0]) b[0] = x; if (x > b[3]) b[3] = x;
    if (y < b[1]) b[1] = y; if (y > b[4]) b[4] = y;
    if (z < b[2]) b[2] = z; if (z > b[5]) b[5] = z;
  }
  _bounds.set(shape, b);
  return b;
}

// ==== rays ======================================================================================================
// Ray vs shape in the shape's local frame, unit direction. Returns t in [0, maxT] or Infinity; the local hit normal
// goes to _nx, _ny, _nz.
let _nx = 0, _ny = 0, _nz = 0;

function raySphere(cx, cy, cz, r, ox, oy, oz, dx, dy, dz, maxT) {
  const mx = ox - cx, my = oy - cy, mz = oz - cz;
  const c = mx * mx + my * my + mz * mz - r * r;
  if (c <= 0) { _nx = -dx; _ny = -dy; _nz = -dz; return 0; }
  const b = mx * dx + my * dy + mz * dz;
  if (b >= 0) return Infinity;
  const disc = b * b - c;
  if (disc < 0) return Infinity;
  const t = -b - Math.sqrt(disc);
  if (t > maxT) return Infinity;
  const ir = 1 / r;
  _nx = (mx + dx * t) * ir; _ny = (my + dy * t) * ir; _nz = (mz + dz * t) * ir;
  return t;
}

function rayCapsule(h, r, ox, oy, oz, dx, dy, dz, maxT) {
  const cy = oy < -h ? -h : oy > h ? h : oy, ey = oy - cy;
  if (ox * ox + ey * ey + oz * oz <= r * r) { _nx = -dx; _ny = -dy; _nz = -dz; return 0; }
  let best = Infinity, bx = 0, by = 0, bz = 0;
  // lateral surface of the finite cylinder (entries through the end discs are inside a cap sphere: found below)
  const a = dx * dx + dz * dz;
  if (a > 1e-24) {
    const b = ox * dx + oz * dz, c = ox * ox + oz * oz - r * r, disc = b * b - a * c;
    if (disc >= 0) {
      const t = (-b - Math.sqrt(disc)) / a;
      if (t >= 0 && t <= maxT) {
        const y = oy + dy * t;
        if (y >= -h && y <= h) { best = t; bx = (ox + dx * t) / r; by = 0; bz = (oz + dz * t) / r; }
      }
    }
  }
  let t = raySphere(0, h, 0, r, ox, oy, oz, dx, dy, dz, maxT);
  if (t < best) { best = t; bx = _nx; by = _ny; bz = _nz; }
  t = raySphere(0, -h, 0, r, ox, oy, oz, dx, dy, dz, maxT);
  if (t < best) { best = t; bx = _nx; by = _ny; bz = _nz; }
  _nx = bx; _ny = by; _nz = bz;
  return best;
}

function rayBox(hx, hy, hz, ox, oy, oz, dx, dy, dz, maxT) {
  let tmin = 0, tmax = maxT, axis = -1, sign = 0;
  if (dx < 1e-15 && dx > -1e-15) { if (ox < -hx || ox > hx) return Infinity; }
  else {
    const inv = 1 / dx;
    let t1 = (-hx - ox) * inv, t2 = (hx - ox) * inv, s = -1;
    if (t1 > t2) { const tt = t1; t1 = t2; t2 = tt; s = 1; }
    if (t1 > tmin) { tmin = t1; axis = 0; sign = s; }
    if (t2 < tmax) tmax = t2;
    if (tmin > tmax) return Infinity;
  }
  if (dy < 1e-15 && dy > -1e-15) { if (oy < -hy || oy > hy) return Infinity; }
  else {
    const inv = 1 / dy;
    let t1 = (-hy - oy) * inv, t2 = (hy - oy) * inv, s = -1;
    if (t1 > t2) { const tt = t1; t1 = t2; t2 = tt; s = 1; }
    if (t1 > tmin) { tmin = t1; axis = 1; sign = s; }
    if (t2 < tmax) tmax = t2;
    if (tmin > tmax) return Infinity;
  }
  if (dz < 1e-15 && dz > -1e-15) { if (oz < -hz || oz > hz) return Infinity; }
  else {
    const inv = 1 / dz;
    let t1 = (-hz - oz) * inv, t2 = (hz - oz) * inv, s = -1;
    if (t1 > t2) { const tt = t1; t1 = t2; t2 = tt; s = 1; }
    if (t1 > tmin) { tmin = t1; axis = 2; sign = s; }
    if (t2 < tmax) tmax = t2;
    if (tmin > tmax) return Infinity;
  }
  if (axis < 0) { _nx = -dx; _ny = -dy; _nz = -dz; }
  else { _nx = axis === 0 ? sign : 0; _ny = axis === 1 ? sign : 0; _nz = axis === 2 ? sign : 0; }
  return tmin;
}

// clip the ray against the outward half-spaces n.p + d <= 0
function rayConvex(planes, ox, oy, oz, dx, dy, dz, maxT) {
  let tEnter = 0, tExit = maxT, enter = -1;
  for (let i = 0; i + 3 < planes.length; i += 4) {
    const nx = planes[i], ny = planes[i + 1], nz = planes[i + 2];
    const dist = nx * ox + ny * oy + nz * oz + planes[i + 3];
    const den = nx * dx + ny * dy + nz * dz;
    if (den < 1e-15 && den > -1e-15) { if (dist > 0) return Infinity; continue; }
    const t = -dist / den;
    if (den < 0) { if (t > tEnter) { tEnter = t; enter = i; } }
    else if (t < tExit) tExit = t;
    if (tEnter > tExit) return Infinity;
  }
  if (enter < 0) { _nx = -dx; _ny = -dy; _nz = -dz; return 0; }
  _nx = planes[enter]; _ny = planes[enter + 1]; _nz = planes[enter + 2];
  return tEnter;
}

// Moller-Trumbore on triangle tri; front faces only (counter-clockwise seen from the ray origin) unless twoSided.
// Writes the unit face normal (e1 x e2) to _nx.. on a hit.
function rayTriangle(P, I, tri, ox, oy, oz, dx, dy, dz, maxT, twoSided) {
  const i0 = I[tri * 3] * 3, i1 = I[tri * 3 + 1] * 3, i2 = I[tri * 3 + 2] * 3;
  const ax = P[i0], ay = P[i0 + 1], az = P[i0 + 2];
  const e1x = P[i1] - ax, e1y = P[i1 + 1] - ay, e1z = P[i1 + 2] - az;
  const e2x = P[i2] - ax, e2y = P[i2 + 1] - ay, e2z = P[i2 + 2] - az;
  const px = dy * e2z - dz * e2y, py = dz * e2x - dx * e2z, pz = dx * e2y - dy * e2x;
  const det = e1x * px + e1y * py + e1z * pz;
  if (twoSided ? det === 0 : !(det > 0)) return Infinity;
  const inv = 1 / det;
  const sx = ox - ax, sy = oy - ay, sz = oz - az;
  const u = (sx * px + sy * py + sz * pz) * inv;
  if (u < 0 || u > 1) return Infinity;
  const qx = sy * e1z - sz * e1y, qy = sz * e1x - sx * e1z, qz = sx * e1y - sy * e1x;
  const v = (dx * qx + dy * qy + dz * qz) * inv;
  if (v < 0 || u + v > 1) return Infinity;
  const t = (e2x * qx + e2y * qy + e2z * qz) * inv;
  if (t < 0 || t > maxT) return Infinity;
  let nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
  const l = Math.sqrt(nx * nx + ny * ny + nz * nz);
  if (l > 0) { nx /= l; ny /= l; nz /= l; }
  _nx = nx; _ny = ny; _nz = nz;
  return t;
}

// Entry distance of the ray into node nd's box (Infinity on a miss or beyond maxT). The inverse direction has its
// zero components replaced by +-1e300, so no 0 * Infinity NaN arises.
let _stack = new Int32Array(64);
function nodeEntry(lo, hi, nd, ox, oy, oz, ix, iy, iz, maxT) {
  const k = nd * 3;
  let t1 = (lo[k] - ox) * ix, t2 = (hi[k] - ox) * ix;
  let tmin = t1 < t2 ? t1 : t2, tmax = t1 < t2 ? t2 : t1;
  t1 = (lo[k + 1] - oy) * iy; t2 = (hi[k + 1] - oy) * iy;
  if (t1 > t2) { const t = t1; t1 = t2; t2 = t; }
  if (t1 > tmin) tmin = t1; if (t2 < tmax) tmax = t2;
  t1 = (lo[k + 2] - oz) * iz; t2 = (hi[k + 2] - oz) * iz;
  if (t1 > t2) { const t = t1; t1 = t2; t2 = t; }
  if (t1 > tmin) tmin = t1; if (t2 < tmax) tmax = t2;
  if (tmin < 0) tmin = 0;
  return tmin <= tmax && tmin <= maxT ? tmin : Infinity;
}

// Ray vs trimesh: near-first traversal of the shape's BVH (shapes.js node arrays) with an explicit stack, pruning
// nodes that start beyond the best hit; brute force over the triangles when there is no BVH.
function rayTrimesh(shape, ox, oy, oz, dx, dy, dz, maxT) {
  const P = shape.positions, I = shape.indices;
  if (!P || !I) return Infinity;
  let best = maxT, bnx = 0, bny = 0, bnz = 0, hit = false;
  const bvh = shape.bvh;
  if (!bvh || !bvh.nodeMin || !(bvh.nodeCount > 0)) {
    const nTri = (I.length / 3) | 0;
    for (let k = 0; k < nTri; k++) {
      const t = rayTriangle(P, I, k, ox, oy, oz, dx, dy, dz, best, false);
      if (t <= best) { best = t; bnx = _nx; bny = _ny; bnz = _nz; hit = true; }
    }
  } else {
    const lo = bvh.nodeMin, hi = bvh.nodeMax, L = bvh.left, R = bvh.right, TS = bvh.triStart, TC = bvh.triCount, TO = bvh.triOrder;
    const ix = dx > 1e-15 || dx < -1e-15 ? 1 / dx : dx < 0 ? -1e300 : 1e300;
    const iy = dy > 1e-15 || dy < -1e-15 ? 1 / dy : dy < 0 ? -1e300 : 1e300;
    const iz = dz > 1e-15 || dz < -1e-15 ? 1 / dz : dz < 0 ? -1e300 : 1e300;
    let stack = _stack, sp = 0;
    if (nodeEntry(lo, hi, 0, ox, oy, oz, ix, iy, iz, best) !== Infinity) stack[sp++] = 0;
    while (sp > 0) {
      const nd = stack[--sp];
      if (L[nd] < 0) {
        for (let k = TS[nd], e = TS[nd] + TC[nd]; k < e; k++) {
          const t = rayTriangle(P, I, TO[k], ox, oy, oz, dx, dy, dz, best, false);
          if (t <= best) { best = t; bnx = _nx; bny = _ny; bnz = _nz; hit = true; }
        }
        continue;
      }
      const a = L[nd], b = R[nd];
      const ta = nodeEntry(lo, hi, a, ox, oy, oz, ix, iy, iz, best), tb = nodeEntry(lo, hi, b, ox, oy, oz, ix, iy, iz, best);
      if (sp + 2 > stack.length) { const g = new Int32Array(stack.length * 2); g.set(stack); stack = _stack = g; }
      // push the farther child first so the nearer one is popped next
      if (ta <= tb) { if (tb !== Infinity) stack[sp++] = b; if (ta !== Infinity) stack[sp++] = a; }
      else { if (ta !== Infinity) stack[sp++] = a; if (tb !== Infinity) stack[sp++] = b; }
    }
  }
  if (!hit) return Infinity;
  _nx = bnx; _ny = bny; _nz = bnz;
  return best;
}

function rayShape(shape, ox, oy, oz, dx, dy, dz, maxT) {
  switch (shape.type) {
    case SHAPE.SPHERE: return raySphere(0, 0, 0, shape.radius, ox, oy, oz, dx, dy, dz, maxT);
    case SHAPE.CAPSULE: return rayCapsule(shape.halfHeight, shape.radius, ox, oy, oz, dx, dy, dz, maxT);
    case SHAPE.BOX: { const h = shape.half; return rayBox(h[0], h[1], h[2], ox, oy, oz, dx, dy, dz, maxT); }
    case SHAPE.CONVEX: return rayConvex(shape.planes, ox, oy, oz, dx, dy, dz, maxT);
    case SHAPE.TRIMESH: return rayTrimesh(shape, ox, oy, oz, dx, dy, dz, maxT);
    case SHAPE.COMPOUND: {
      const ch = shape.children;
      let best = maxT, hit = false, bnx = 0, bny = 0, bnz = 0;
      for (let i = 0; i < ch.length; i++) {
        const c = ch[i], cp = c.pos, cq = c.quat;
        const qx = cq[0], qy = cq[1], qz = cq[2], qw = cq[3];
        invRot(qx, qy, qz, qw, ox - cp[0], oy - cp[1], oz - cp[2]);
        const lox = _rx, loy = _ry, loz = _rz;
        invRot(qx, qy, qz, qw, dx, dy, dz);
        const t = rayShape(c.shape, lox, loy, loz, _rx, _ry, _rz, best);
        if (t <= best) { rot(qx, qy, qz, qw, _nx, _ny, _nz); best = t; bnx = _rx; bny = _ry; bnz = _rz; hit = true; }
      }
      if (!hit) return Infinity;
      _nx = bnx; _ny = bny; _nz = bnz;
      return best;
    }
    default: return Infinity;
  }
}

/** Ray vs one body (world frame, unit direction): distance or Infinity; the world normal goes to the scratch. */
function rayBody(B, id, shape, ox, oy, oz, dx, dy, dz, maxT) {
  const p = id * 3, q = id * 4, P = B.pos, Q = B.quat;
  const qx = Q[q], qy = Q[q + 1], qz = Q[q + 2], qw = Q[q + 3];
  invRot(qx, qy, qz, qw, ox - P[p], oy - P[p + 1], oz - P[p + 2]);
  const lox = _rx, loy = _ry, loz = _rz;
  invRot(qx, qy, qz, qw, dx, dy, dz);
  const t = rayShape(shape, lox, loy, loz, _rx, _ry, _rz, maxT);
  if (t !== Infinity) { rot(qx, qy, qz, qw, _nx, _ny, _nz); _nx = _rx; _ny = _ry; _nz = _rz; }
  return t;
}

function writeHit(out, distance, x, y, z, nx, ny, nz, id) {
  out.distance = distance;
  const p = out.point || (out.point = { x: 0, y: 0, z: 0 });
  const n = out.normal || (out.normal = { x: 0, y: 0, z: 0 });
  p.x = x; p.y = y; p.z = z;
  n.x = nx; n.y = ny; n.z = nz;
  out.body = id; out.box = null; out.kind = KIND; out.ground = false;
}

/**
 * Nearest exact ray hit among the alive bodies.
 * @param {object} world   DynamicsWorld (or { bodies, broadphase? })
 * @param {number} dx      direction (need not be unit)
 * @param {number} maxDist distance along the normalised direction
 * @param {object|null|undefined} opts  mask (bit mask tested against bodies.layer), filter(id) => false skips the body
 * @param {object} outHit  receives { distance, point, normal, body, box: null, kind: 'rigid', ground: false }
 * @returns {boolean} true on a hit (outHit untouched otherwise)
 */
export function raycastBodies(world, ox, oy, oz, dx, dy, dz, maxDist, opts, outHit) {
  const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
  if (!(len > 1e-12) || !Number.isFinite(ox + oy + oz)) return false;
  dx /= len; dy /= len; dz /= len;
  if (!(maxDist > 0)) return false;
  if (!(maxDist <= MAX_DIST)) maxDist = MAX_DIST;
  const o = opts || EMPTY_OPTS, mask = o.mask ?? -1, filter = o.filter || null;
  const B = world.bodies, bp = world.broadphase;
  let n, cand = null;
  if (bp && typeof bp.raycast === 'function') { cand = candBuf(B); n = bp.raycast(ox, oy, oz, dx, dy, dz, maxDist, cand, mask); }
  else n = B.count;
  let best = maxDist, bestId = -1, bnx = 0, bny = 0, bnz = 0;
  for (let i = 0; i < n; i++) {
    const id = cand ? cand[i] : i;
    if (id < 0 || id >= B.count || !B.alive[id]) continue;
    if (mask !== -1 && (B.layer[id] & mask) === 0) continue;
    const shape = B.shape[id];
    if (!shape || (filter && !filter(id))) continue;
    const t = rayBody(B, id, shape, ox, oy, oz, dx, dy, dz, best);
    if (t < best || (t === best && (bestId < 0 || id < bestId))) { best = t; bestId = id; bnx = _nx; bny = _ny; bnz = _nz; }
  }
  if (bestId < 0) return false;
  writeHit(outHit, best, ox + dx * best, oy + dy * best, oz + dz * best, bnx, bny, bnz, bestId);
  return true;
}

// ==== containment ===============================================================================================
function pointInShape(shape, x, y, z) {
  switch (shape.type) {
    case SHAPE.SPHERE: return x * x + y * y + z * z <= shape.radius * shape.radius;
    case SHAPE.CAPSULE: {
      const h = shape.halfHeight, r = shape.radius, cy = y < -h ? -h : y > h ? h : y, ey = y - cy;
      return x * x + ey * ey + z * z <= r * r;
    }
    case SHAPE.BOX: {
      const h = shape.half;
      return x >= -h[0] && x <= h[0] && y >= -h[1] && y <= h[1] && z >= -h[2] && z <= h[2];
    }
    case SHAPE.CONVEX: {
      const pl = shape.planes;
      for (let i = 0; i + 3 < pl.length; i += 4) if (pl[i] * x + pl[i + 1] * y + pl[i + 2] * z + pl[i + 3] > 0) return false;
      return true;
    }
    case SHAPE.TRIMESH: {
      // closed meshes: crossing parity along a fixed oblique direction (both faces count)
      const b = meshBounds(shape);
      if (x < b[0] || x > b[3] || y < b[1] || y > b[4] || z < b[2] || z > b[5]) return false;
      const P = shape.positions, I = shape.indices, nTri = (I.length / 3) | 0;
      const dx = 0.5773190, dy = 0.5773603, dz = 0.5773710;
      let crossings = 0;
      for (let k = 0; k < nTri; k++) if (rayTriangle(P, I, k, x, y, z, dx, dy, dz, Infinity, true) !== Infinity) crossings++;
      return (crossings & 1) === 1;
    }
    case SHAPE.COMPOUND: {
      const ch = shape.children;
      for (let i = 0; i < ch.length; i++) {
        const c = ch[i], cp = c.pos, cq = c.quat;
        invRot(cq[0], cq[1], cq[2], cq[3], x - cp[0], y - cp[1], z - cp[2]);
        if (pointInShape(c.shape, _rx, _ry, _rz)) return true;
      }
      return false;
    }
    default: return false;
  }
}

/** Lowest alive body id whose shape contains (x, y, z) (surface included), or -1. */
export function pointInBody(world, x, y, z) {
  const B = world.bodies, bp = world.broadphase;
  let n, cand = null;
  if (bp && typeof bp.query === 'function') { cand = candBuf(B); n = bp.query(x, y, z, x, y, z, cand, -1); }
  else n = B.count;
  let best = -1;
  for (let i = 0; i < n; i++) {
    const id = cand ? cand[i] : i;
    if (id < 0 || id >= B.count || !B.alive[id] || (best >= 0 && id >= best)) continue;
    const shape = B.shape[id];
    if (!shape) continue;
    const p = id * 3, q = id * 4, P = B.pos, Q = B.quat;
    invRot(Q[q], Q[q + 1], Q[q + 2], Q[q + 3], x - P[p], y - P[p + 1], z - P[p + 2]);
    if (pointInShape(shape, _rx, _ry, _rz)) best = id;
  }
  return best;
}

/**
 * Bodies whose world AABB overlaps the box: broadphase.query candidates (or every alive body), then a tight check
 * against bodies.aabb and the layer mask. Writes up to outInt32.length ids (ascending) and returns the TRUE number
 * of overlapping bodies (more than were written when outInt32 is too short).
 */
export function overlapBodies(world, minX, minY, minZ, maxX, maxY, maxZ, outInt32, mask = -1) {
  const B = world.bodies, A = B.aabb, bp = world.broadphase, cap = outInt32.length;
  if (!bp || typeof bp.query !== 'function') { // brute force: every alive body, ascending id
    let total = 0;
    for (let id = 0; id < B.count; id++) {
      if (!B.alive[id] || (mask !== -1 && (B.layer[id] & mask) === 0)) continue;
      const o = id * 6;
      if (A[o] > maxX || A[o + 3] < minX || A[o + 1] > maxY || A[o + 4] < minY || A[o + 2] > maxZ || A[o + 5] < minZ) continue;
      if (total < cap) outInt32[total] = id;
      total++;
    }
    return total;
  }
  const n = bp.query(minX, minY, minZ, maxX, maxY, maxZ, outInt32, mask);
  let k = 0;
  for (let i = 0; i < n; i++) {
    const id = outInt32[i];
    if (id < 0 || id >= B.count || !B.alive[id]) continue;
    if (mask !== -1 && (B.layer[id] & mask) === 0) continue;
    const o = id * 6;
    if (A[o] > maxX || A[o + 3] < minX || A[o + 1] > maxY || A[o + 4] < minY || A[o + 2] > maxZ || A[o + 5] < minZ) continue;
    outInt32[k++] = id;
  }
  // a truncated broadphase result (bp.queryTotal > n) still reports the true count
  const total = typeof bp.queryTotal === 'number' && bp.queryTotal > n ? bp.queryTotal : n;
  return k + (total - n);
}

// ==== shape casts: GJK distance + conservative advancement ======================================================
// A convex piece is a core (point, segment, box, point cloud or world-space triangle) at a pose, plus a margin
// (sphere / capsule radius). GJK runs on the cores; the surface distance is core distance - margins.
const CORE_POINT = 0, CORE_SEGMENT = 1, CORE_BOX = 2, CORE_POINTS = 3, CORE_TRI = 4;
function makeDesc() {
  return { core: 0, margin: 0, h: 0, hx: 0, hy: 0, hz: 0, pts: null, tri: new Float64Array(9), px: 0, py: 0, pz: 0, qx: 0, qy: 0, qz: 0, qw: 1 };
}
const _A = makeDesc(), _Bd = makeDesc();

// false when the shape is not a convex primitive (trimesh / compound / unknown)
function setDesc(d, shape, px, py, pz, qx, qy, qz, qw) {
  d.px = px; d.py = py; d.pz = pz; d.qx = qx; d.qy = qy; d.qz = qz; d.qw = qw;
  switch (shape.type) {
    case SHAPE.SPHERE: d.core = CORE_POINT; d.margin = shape.radius; return true;
    case SHAPE.CAPSULE: d.core = CORE_SEGMENT; d.h = shape.halfHeight; d.margin = shape.radius; return true;
    case SHAPE.BOX: d.core = CORE_BOX; d.hx = shape.half[0]; d.hy = shape.half[1]; d.hz = shape.half[2]; d.margin = 0; return true;
    case SHAPE.CONVEX: d.core = CORE_POINTS; d.pts = shape.points; d.margin = 0; return true;
    default: return false;
  }
}

// core support point in world direction (dx, dy, dz) -> _sx, _sy, _sz
let _sx = 0, _sy = 0, _sz = 0;
function support(d, dx, dy, dz) {
  if (d.core === CORE_TRI) {
    const T = d.tri;
    let bi = 0, bd = T[0] * dx + T[1] * dy + T[2] * dz;
    const d1 = T[3] * dx + T[4] * dy + T[5] * dz; if (d1 > bd) { bd = d1; bi = 3; }
    const d2 = T[6] * dx + T[7] * dy + T[8] * dz; if (d2 > bd) bi = 6;
    _sx = T[bi]; _sy = T[bi + 1]; _sz = T[bi + 2];
    return;
  }
  invRot(d.qx, d.qy, d.qz, d.qw, dx, dy, dz);
  const lx = _rx, ly = _ry, lz = _rz;
  let sx = 0, sy = 0, sz = 0;
  switch (d.core) {
    case CORE_SEGMENT: sy = ly >= 0 ? d.h : -d.h; break;
    case CORE_BOX: sx = lx >= 0 ? d.hx : -d.hx; sy = ly >= 0 ? d.hy : -d.hy; sz = lz >= 0 ? d.hz : -d.hz; break;
    case CORE_POINTS: {
      const P = d.pts;
      let bd = -Infinity;
      for (let i = 0; i + 2 < P.length; i += 3) {
        const v = P[i] * lx + P[i + 1] * ly + P[i + 2] * lz;
        if (v > bd) { bd = v; sx = P[i]; sy = P[i + 1]; sz = P[i + 2]; }
      }
      break;
    }
    default: break; // point core: the origin
  }
  rot(d.qx, d.qy, d.qz, d.qw, sx, sy, sz);
  _sx = d.px + _rx; _sy = d.py + _ry; _sz = d.pz + _rz;
}

// World AABB of a convex piece (core extents in the axis directions, grown by the margin) -> _ab
const _ab = new Float64Array(6);
function descAabb(d) {
  const m = d.margin;
  support(d, 1, 0, 0); _ab[3] = _sx + m; support(d, -1, 0, 0); _ab[0] = _sx - m;
  support(d, 0, 1, 0); _ab[4] = _sy + m; support(d, 0, -1, 0); _ab[1] = _sy - m;
  support(d, 0, 0, 1); _ab[5] = _sz + m; support(d, 0, 0, -1); _ab[2] = _sz - m;
}

// ---- GJK distance (Minkowski difference A - B; simplex of up to 4 points with their A / B supports) -------------
const _W = new Float64Array(12), _PA = new Float64Array(12), _PB = new Float64Array(12);
const _lam = new Float64Array(4), _lamT = new Float64Array(4), _lamS = new Float64Array(4);
let _sn = 0;
let _vx = 0, _vy = 0, _vz = 0;               // closest point of the simplex to the origin
let _pax = 0, _pay = 0, _paz = 0, _pbx = 0, _pby = 0, _pbz = 0; // closest core points

// closest point of segment (i, j) to the origin: weights into _lam, returns the used-vertex bit mask
function closestSeg(i, j) {
  const ax = _W[i * 3], ay = _W[i * 3 + 1], az = _W[i * 3 + 2];
  const abx = _W[j * 3] - ax, aby = _W[j * 3 + 1] - ay, abz = _W[j * 3 + 2] - az;
  const ll = abx * abx + aby * aby + abz * abz;
  const t = ll > 0 ? -(ax * abx + ay * aby + az * abz) / ll : 0;
  if (t <= 0) { _lam[i] = 1; _lam[j] = 0; return 1 << i; }
  if (t >= 1) { _lam[i] = 0; _lam[j] = 1; return 1 << j; }
  _lam[i] = 1 - t; _lam[j] = t;
  return (1 << i) | (1 << j);
}

// closest point of triangle (i, j, k) to the origin (Ericson, Real-Time Collision Detection 5.1.5)
function closestTri(i, j, k) {
  const ax = _W[i * 3], ay = _W[i * 3 + 1], az = _W[i * 3 + 2];
  const bx = _W[j * 3], by = _W[j * 3 + 1], bz = _W[j * 3 + 2];
  const cx = _W[k * 3], cy = _W[k * 3 + 1], cz = _W[k * 3 + 2];
  const abx = bx - ax, aby = by - ay, abz = bz - az, acx = cx - ax, acy = cy - ay, acz = cz - az;
  const d1 = -(abx * ax + aby * ay + abz * az), d2 = -(acx * ax + acy * ay + acz * az);
  _lam[i] = 0; _lam[j] = 0; _lam[k] = 0;
  if (d1 <= 0 && d2 <= 0) { _lam[i] = 1; return 1 << i; }
  const d3 = -(abx * bx + aby * by + abz * bz), d4 = -(acx * bx + acy * by + acz * bz);
  if (d3 >= 0 && d4 <= d3) { _lam[j] = 1; return 1 << j; }
  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) { const v = d1 / (d1 - d3); _lam[i] = 1 - v; _lam[j] = v; return (1 << i) | (1 << j); }
  const d5 = -(abx * cx + aby * cy + abz * cz), d6 = -(acx * cx + acy * cy + acz * cz);
  if (d6 >= 0 && d5 <= d6) { _lam[k] = 1; return 1 << k; }
  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) { const w = d2 / (d2 - d6); _lam[i] = 1 - w; _lam[k] = w; return (1 << i) | (1 << k); }
  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
    const w = (d4 - d3) / ((d4 - d3) + (d5 - d6)); _lam[j] = 1 - w; _lam[k] = w; return (1 << j) | (1 << k);
  }
  const den = va + vb + vc;
  if (!(den > 0)) { // degenerate (collinear) triangle: best of its three edges
    let bm = closestSeg(i, j); _lam[k] = 0;
    let bd = lamDist2(bm);
    _lamS.set(_lam);
    let m = closestSeg(j, k); _lam[i] = 0;
    let dd = lamDist2(m);
    if (dd < bd) { bd = dd; bm = m; _lamS.set(_lam); }
    m = closestSeg(i, k); _lam[j] = 0;
    dd = lamDist2(m);
    if (dd < bd) { bm = m; _lamS.set(_lam); }
    _lam.set(_lamS);
    return bm;
  }
  const inv = 1 / den, v = vb * inv, w = vc * inv;
  _lam[i] = 1 - v - w; _lam[j] = v; _lam[k] = w;
  return (1 << i) | (1 << j) | (1 << k);
}

function lamDist2(mask) {
  let x = 0, y = 0, z = 0;
  for (let i = 0; i < 4; i++) if (mask & (1 << i)) { x += _lam[i] * _W[i * 3]; y += _lam[i] * _W[i * 3 + 1]; z += _lam[i] * _W[i * 3 + 2]; }
  return x * x + y * y + z * z;
}

// sign of the origin vs the plane (a, b, c) relative to the fourth point d: true when the origin is on the far side
function outsideFace(a, b, c, d) {
  const ax = _W[a * 3], ay = _W[a * 3 + 1], az = _W[a * 3 + 2];
  const abx = _W[b * 3] - ax, aby = _W[b * 3 + 1] - ay, abz = _W[b * 3 + 2] - az;
  const acx = _W[c * 3] - ax, acy = _W[c * 3 + 1] - ay, acz = _W[c * 3 + 2] - az;
  const nx = aby * acz - abz * acy, ny = abz * acx - abx * acz, nz = abx * acy - aby * acx;
  const sp = -(ax * nx + ay * ny + az * nz);
  const sd = (_W[d * 3] - ax) * nx + (_W[d * 3 + 1] - ay) * ny + (_W[d * 3 + 2] - az) * nz;
  if (sd * sd <= 1e-24 * (nx * nx + ny * ny + nz * nz)) return true; // flat tetrahedron: test every face
  return sp * sd < 0;
}

// closest point of the tetrahedron 0..3; -1 when the origin is inside
function closestTet() {
  let best = Infinity, bm = -1;
  if (outsideFace(0, 1, 2, 3)) { const m = closestTri(0, 1, 2); _lam[3] = 0; const d = lamDist2(m); if (d < best) { best = d; bm = m; _lamT.set(_lam); } }
  if (outsideFace(0, 1, 3, 2)) { const m = closestTri(0, 1, 3); _lam[2] = 0; const d = lamDist2(m); if (d < best) { best = d; bm = m; _lamT.set(_lam); } }
  if (outsideFace(0, 2, 3, 1)) { const m = closestTri(0, 2, 3); _lam[1] = 0; const d = lamDist2(m); if (d < best) { best = d; bm = m; _lamT.set(_lam); } }
  if (outsideFace(1, 2, 3, 0)) { const m = closestTri(1, 2, 3); _lam[0] = 0; const d = lamDist2(m); if (d < best) { best = d; bm = m; _lamT.set(_lam); } }
  if (bm >= 0) _lam.set(_lamT);
  return bm;
}

// reduce the simplex to the vertices carrying its closest point; false when the origin is enclosed
function reduceSimplex() {
  let mask;
  if (_sn === 1) { _lam[0] = 1; mask = 1; }
  else if (_sn === 2) mask = closestSeg(0, 1);
  else if (_sn === 3) mask = closestTri(0, 1, 2);
  else mask = closestTet();
  if (mask < 0) return false;
  let k = 0;
  _vx = 0; _vy = 0; _vz = 0;
  for (let i = 0; i < _sn; i++) {
    if (!(mask & (1 << i))) continue;
    if (k !== i) {
      for (let c = 0; c < 3; c++) { _W[k * 3 + c] = _W[i * 3 + c]; _PA[k * 3 + c] = _PA[i * 3 + c]; _PB[k * 3 + c] = _PB[i * 3 + c]; }
      _lam[k] = _lam[i];
    }
    _vx += _lam[k] * _W[k * 3]; _vy += _lam[k] * _W[k * 3 + 1]; _vz += _lam[k] * _W[k * 3 + 2];
    k++;
  }
  _sn = k;
  return true;
}

// Distance between the cores of A and B (0 when they overlap); closest core points in _pa*, _pb*.
function gjk(A, B) {
  let dx = A.px - B.px, dy = A.py - B.py, dz = A.pz - B.pz;
  if (dx * dx + dy * dy + dz * dz < 1e-24) { dx = 1; dy = 0; dz = 0; }
  support(A, -dx, -dy, -dz); const ax = _sx, ay = _sy, az = _sz;
  support(B, dx, dy, dz);
  _W[0] = ax - _sx; _W[1] = ay - _sy; _W[2] = az - _sz;
  _PA[0] = ax; _PA[1] = ay; _PA[2] = az; _PB[0] = _sx; _PB[1] = _sy; _PB[2] = _sz;
  _sn = 1; _lam[0] = 1; _vx = _W[0]; _vy = _W[1]; _vz = _W[2];
  let vv = _vx * _vx + _vy * _vy + _vz * _vz;
  for (let it = 0; it < 64 && vv > 1e-24; it++) {
    support(A, -_vx, -_vy, -_vz); const sax = _sx, say = _sy, saz = _sz;
    support(B, _vx, _vy, _vz);
    const wx = sax - _sx, wy = say - _sy, wz = saz - _sz;
    if (vv - (_vx * wx + _vy * wy + _vz * wz) <= 1e-12 * vv) break; // no further progress possible
    let dup = false;
    for (let i = 0; i < _sn; i++) if (_W[i * 3] === wx && _W[i * 3 + 1] === wy && _W[i * 3 + 2] === wz) { dup = true; break; }
    if (dup) break;
    const k = _sn * 3;
    _W[k] = wx; _W[k + 1] = wy; _W[k + 2] = wz;
    _PA[k] = sax; _PA[k + 1] = say; _PA[k + 2] = saz; _PB[k] = _sx; _PB[k + 1] = _sy; _PB[k + 2] = _sz;
    _sn++;
    if (!reduceSimplex()) { vv = 0; break; }
    const nv = _vx * _vx + _vy * _vy + _vz * _vz;
    if (nv >= vv) { vv = nv; break; }
    vv = nv;
  }
  if (!(vv > 1e-24)) return 0;
  _pax = 0; _pay = 0; _paz = 0; _pbx = 0; _pby = 0; _pbz = 0;
  for (let i = 0; i < _sn; i++) {
    const l = _lam[i], k = i * 3;
    _pax += l * _PA[k]; _pay += l * _PA[k + 1]; _paz += l * _PA[k + 2];
    _pbx += l * _PB[k]; _pby += l * _PB[k + 1]; _pbz += l * _PB[k + 2];
  }
  return Math.sqrt(vv);
}

// Conservative advancement of piece A (moving along unit _cd*) against static piece B, up to maxT. On a hit returns
// the travel and writes the contact point on B (_cpx..) and the normal out of B (_cnx..); Infinity otherwise.
let _cdx = 0, _cdy = 0, _cdz = 0;
let _cpx = 0, _cpy = 0, _cpz = 0, _cnx = 0, _cny = 0, _cnz = 0;
const CA_TOL = 1e-9;
function castPieces(A, B, maxT) {
  const x0 = A.px, y0 = A.py, z0 = A.pz, m = A.margin + B.margin;
  _cpx = _cpy = _cpz = _cnx = _cny = _cnz = NaN; // never report a previous cast's contact
  let t = 0, result = Infinity;
  let pnx = 0, pny = 0, pnz = 0, pbx = 0, pby = 0, pbz = 0; // last iteration's normal (A -> B) and core point on B
  for (let it = 0; it < 64; it++) {
    A.px = x0 + _cdx * t; A.py = y0 + _cdy * t; A.pz = z0 + _cdz * t;
    const dc = gjk(A, B);
    if (dc === 0) {
      if (t === 0) { // cores overlap at the start
        _cnx = -_cdx; _cny = -_cdy; _cnz = -_cdz; _cpx = A.px; _cpy = A.py; _cpz = A.pz; result = 0;
      } else { // margin-less cores advanced exactly into touch: the contact is the previous iteration's
        _cnx = -pnx; _cny = -pny; _cnz = -pnz;
        _cpx = pbx - pnx * B.margin; _cpy = pby - pny * B.margin; _cpz = pbz - pnz * B.margin;
        result = t;
      }
      break;
    }
    const inv = 1 / dc, nx = (_pbx - _pax) * inv, ny = (_pby - _pay) * inv, nz = (_pbz - _paz) * inv; // A -> B
    pnx = nx; pny = ny; pnz = nz; pbx = _pbx; pby = _pby; pbz = _pbz;
    const dist = dc - m;
    if (dist <= CA_TOL) {
      _cnx = -nx; _cny = -ny; _cnz = -nz;
      _cpx = _pbx - nx * B.margin; _cpy = _pby - ny * B.margin; _cpz = _pbz - nz * B.margin;
      result = t;
      break;
    }
    const dn = _cdx * nx + _cdy * ny + _cdz * nz;
    if (!(dn > 1e-12)) break; // separating (or sliding past): never closes
    t += dist / dn;
    if (t > maxT) break;
  }
  A.px = x0; A.py = y0; A.pz = z0;
  return result;
}

// local AABB (in the frame p, q) of a world AABB: centre to local, extents through |R^T|
const _lb = new Float64Array(6);
function worldAabbToLocal(a, px, py, pz, qx, qy, qz, qw) {
  const cx = (a[0] + a[3]) * 0.5, cy = (a[1] + a[4]) * 0.5, cz = (a[2] + a[5]) * 0.5;
  const ex = (a[3] - a[0]) * 0.5, ey = (a[4] - a[1]) * 0.5, ez = (a[5] - a[2]) * 0.5;
  invRot(qx, qy, qz, qw, cx - px, cy - py, cz - pz);
  const lx = _rx, ly = _ry, lz = _rz;
  // images of the world axes in the local frame: |R^T| e
  invRot(qx, qy, qz, qw, 1, 0, 0); const r0x = _rx, r0y = _ry, r0z = _rz;
  invRot(qx, qy, qz, qw, 0, 1, 0); const r1x = _rx, r1y = _ry, r1z = _rz;
  invRot(qx, qy, qz, qw, 0, 0, 1); const r2x = _rx, r2y = _ry, r2z = _rz;
  const abs = Math.abs;
  const hx = abs(r0x) * ex + abs(r1x) * ey + abs(r2x) * ez;
  const hy = abs(r0y) * ex + abs(r1y) * ey + abs(r2y) * ez;
  const hz = abs(r0z) * ex + abs(r1z) * ey + abs(r2z) * ez;
  _lb[0] = lx - hx; _lb[1] = ly - hy; _lb[2] = lz - hz; _lb[3] = lx + hx; _lb[4] = ly + hy; _lb[5] = lz + hz;
}

// Entry travel of a box (centre c, half extents e) moving along unit d into the box [min, max]: the ray from c
// against [min - e, max + e]. Infinity when it never gets there within maxT. Conservative cull for the casts.
function sweepBoxEntry(cx, cy, cz, ex, ey, ez, dx, dy, dz, minX, minY, minZ, maxX, maxY, maxZ, maxT) {
  let t0 = 0, t1 = maxT;
  let lo = minX - ex - 1e-9, hi = maxX + ex + 1e-9;
  if (dx < 1e-15 && dx > -1e-15) { if (cx < lo || cx > hi) return Infinity; }
  else { let a = (lo - cx) / dx, b = (hi - cx) / dx; if (a > b) { const t = a; a = b; b = t; } if (a > t0) t0 = a; if (b < t1) t1 = b; if (t0 > t1) return Infinity; }
  lo = minY - ey - 1e-9; hi = maxY + ey + 1e-9;
  if (dy < 1e-15 && dy > -1e-15) { if (cy < lo || cy > hi) return Infinity; }
  else { let a = (lo - cy) / dy, b = (hi - cy) / dy; if (a > b) { const t = a; a = b; b = t; } if (a > t0) t0 = a; if (b < t1) t1 = b; if (t0 > t1) return Infinity; }
  lo = minZ - ez - 1e-9; hi = maxZ + ez + 1e-9;
  if (dz < 1e-15 && dz > -1e-15) { if (cz < lo || cz > hi) return Infinity; }
  else { let a = (lo - cz) / dz, b = (hi - cz) / dz; if (a > b) { const t = a; a = b; b = t; } if (a > t0) t0 = a; if (b < t1) t1 = b; if (t0 > t1) return Infinity; }
  return t0;
}

// convex piece A (already in _A) against the triangles of a trimesh at pose (p, q): BVH query with A's swept box in
// the mesh frame, then per triangle a box-sweep cull against the current best before the GJK advancement.
function castVsTrimesh(shape, px, py, pz, qx, qy, qz, qw, best) {
  const P = shape.positions, I = shape.indices;
  if (!P || !I) return best;
  descAabb(_A);
  worldAabbToLocal(_ab, px, py, pz, qx, qy, qz, qw);
  invRot(qx, qy, qz, qw, _cdx, _cdy, _cdz);
  const ldx = _rx, ldy = _ry, ldz = _rz, L = _lb;
  const cx = (L[0] + L[3]) * 0.5, cy = (L[1] + L[4]) * 0.5, cz = (L[2] + L[5]) * 0.5;
  const ex = (L[3] - L[0]) * 0.5, ey = (L[4] - L[1]) * 0.5, ez = (L[5] - L[2]) * 0.5;
  const sx = ldx * best, sy = ldy * best, sz = ldz * best, pad = 1e-6;
  let n, list = null;
  if (typeof shape.queryAabb === 'function') { // shapes.js: BVH query in the mesh frame
    list = triBuf(shape);
    n = shape.queryAabb(L[0] + (sx < 0 ? sx : 0) - pad, L[1] + (sy < 0 ? sy : 0) - pad, L[2] + (sz < 0 ? sz : 0) - pad,
      L[3] + (sx > 0 ? sx : 0) + pad, L[4] + (sy > 0 ? sy : 0) + pad, L[5] + (sz > 0 ? sz : 0) + pad, list);
  } else n = (I.length / 3) | 0;
  const T = _Bd.tri;
  _Bd.core = CORE_TRI; _Bd.margin = 0; _Bd.qx = 0; _Bd.qy = 0; _Bd.qz = 0; _Bd.qw = 1;
  for (let k = 0; k < n; k++) {
    const tri = list ? list[k] : k;
    let mnx = Infinity, mny = Infinity, mnz = Infinity, mxx = -Infinity, mxy = -Infinity, mxz = -Infinity;
    for (let v = 0; v < 3; v++) {
      const o = I[tri * 3 + v] * 3, x = P[o], y = P[o + 1], z = P[o + 2];
      if (x < mnx) mnx = x; if (x > mxx) mxx = x; if (y < mny) mny = y; if (y > mxy) mxy = y; if (z < mnz) mnz = z; if (z > mxz) mxz = z;
    }
    if (sweepBoxEntry(cx, cy, cz, ex, ey, ez, ldx, ldy, ldz, mnx, mny, mnz, mxx, mxy, mxz, best) === Infinity) continue;
    let tx = 0, ty = 0, tz = 0;
    for (let v = 0; v < 3; v++) {
      const o = I[tri * 3 + v] * 3;
      rot(qx, qy, qz, qw, P[o], P[o + 1], P[o + 2]);
      T[v * 3] = px + _rx; T[v * 3 + 1] = py + _ry; T[v * 3 + 2] = pz + _rz;
      tx += T[v * 3]; ty += T[v * 3 + 1]; tz += T[v * 3 + 2];
    }
    _Bd.px = tx / 3; _Bd.py = ty / 3; _Bd.pz = tz / 3; // only seeds GJK's first direction
    const t = castPieces(_A, _Bd, best);
    if (t < best) { best = t; _hx = _cpx; _hy = _cpy; _hz = _cpz; _hnx = _cnx; _hny = _cny; _hnz = _cnz; }
  }
  return best;
}

// best hit so far of the current cast
let _hx = 0, _hy = 0, _hz = 0, _hnx = 0, _hny = 0, _hnz = 0;

// moving shape a (pose pa, qa) against static shape b (pose pb, qb); returns the improved best (or best)
function castShapes(a, apx, apy, apz, aqx, aqy, aqz, aqw, b, bpx, bpy, bpz, bqx, bqy, bqz, bqw, best) {
  if (a.type === SHAPE.COMPOUND) {
    const ch = a.children;
    for (let i = 0; i < ch.length; i++) {
      const c = ch[i], cp = c.pos, cq = c.quat;
      rot(aqx, aqy, aqz, aqw, cp[0], cp[1], cp[2]);
      const cx = apx + _rx, cy = apy + _ry, cz = apz + _rz;
      qmul(aqx, aqy, aqz, aqw, cq[0], cq[1], cq[2], cq[3]);
      best = castShapes(c.shape, cx, cy, cz, _mx, _my, _mz, _mw, b, bpx, bpy, bpz, bqx, bqy, bqz, bqw, best);
    }
    return best;
  }
  if (b.type === SHAPE.COMPOUND) {
    const ch = b.children;
    for (let i = 0; i < ch.length; i++) {
      const c = ch[i], cp = c.pos, cq = c.quat;
      rot(bqx, bqy, bqz, bqw, cp[0], cp[1], cp[2]);
      const cx = bpx + _rx, cy = bpy + _ry, cz = bpz + _rz;
      qmul(bqx, bqy, bqz, bqw, cq[0], cq[1], cq[2], cq[3]);
      best = castShapes(a, apx, apy, apz, aqx, aqy, aqz, aqw, c.shape, cx, cy, cz, _mx, _my, _mz, _mw, best);
    }
    return best;
  }
  if (!setDesc(_A, a, apx, apy, apz, aqx, aqy, aqz, aqw)) return best; // trimesh (or unknown) cast shapes: unsupported
  if (b.type === SHAPE.TRIMESH) return castVsTrimesh(b, bpx, bpy, bpz, bqx, bqy, bqz, bqw, best);
  if (!setDesc(_Bd, b, bpx, bpy, bpz, bqx, bqy, bqz, bqw)) return best;
  const t = castPieces(_A, _Bd, best);
  if (t < best) { best = t; _hx = _cpx; _hy = _cpy; _hz = _cpz; _hnx = _cnx; _hny = _cny; _hnz = _cnz; }
  return best;
}

// world AABB of a (possibly compound) cast shape at a pose, accumulated into _cab
const _cab = new Float64Array(6);
function castShapeAabb(shape, px, py, pz, qx, qy, qz, qw) {
  if (shape.type === SHAPE.COMPOUND) {
    const ch = shape.children;
    for (let i = 0; i < ch.length; i++) {
      const c = ch[i], cp = c.pos, cq = c.quat;
      rot(qx, qy, qz, qw, cp[0], cp[1], cp[2]);
      const cx = px + _rx, cy = py + _ry, cz = pz + _rz;
      qmul(qx, qy, qz, qw, cq[0], cq[1], cq[2], cq[3]);
      castShapeAabb(c.shape, cx, cy, cz, _mx, _my, _mz, _mw);
    }
    return;
  }
  if (!setDesc(_A, shape, px, py, pz, qx, qy, qz, qw)) return;
  descAabb(_A);
  for (let i = 0; i < 3; i++) { if (_ab[i] < _cab[i]) _cab[i] = _ab[i]; if (_ab[i + 3] > _cab[i + 3]) _cab[i + 3] = _ab[i + 3]; }
}

/**
 * Sweep `shape` from pose (p, q) along (dx, dy, dz) (need not be unit) for up to maxDist; first contact among the
 * alive bodies (frozen at their current poses). Cast shapes: sphere, capsule, box, convex, or a compound of them.
 * opts / outHit as raycastBodies; `distance` is the travel of the shape, `point` the contact point on the hit body,
 * `normal` points out of the hit body (towards the cast shape). A cast that starts overlapping hits at 0.
 */
export function shapeCastBodies(world, shape, px, py, pz, qx, qy, qz, qw, dx, dy, dz, maxDist, opts, outHit) {
  const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
  if (!shape || !(len > 1e-12) || !Number.isFinite(px + py + pz)) return false;
  if (!(maxDist > 0)) return false;
  if (!(maxDist <= MAX_DIST)) maxDist = MAX_DIST;
  _cdx = dx / len; _cdy = dy / len; _cdz = dz / len;
  const o = opts || EMPTY_OPTS, mask = o.mask ?? -1, filter = o.filter || null;
  const B = world.bodies, bp = world.broadphase;
  // swept bounds of the cast shape
  _cab[0] = _cab[1] = _cab[2] = Infinity; _cab[3] = _cab[4] = _cab[5] = -Infinity;
  castShapeAabb(shape, px, py, pz, qx, qy, qz, qw);
  if (!(_cab[0] <= _cab[3])) return false;
  let n, cand = null;
  if (bp && typeof bp.query === 'function') {
    const ex = _cdx * maxDist, ey = _cdy * maxDist, ez = _cdz * maxDist;
    cand = candBuf(B);
    n = bp.query(_cab[0] + (ex < 0 ? ex : 0), _cab[1] + (ey < 0 ? ey : 0), _cab[2] + (ez < 0 ? ez : 0),
      _cab[3] + (ex > 0 ? ex : 0), _cab[4] + (ey > 0 ? ey : 0), _cab[5] + (ez > 0 ? ez : 0), cand, mask);
    sortIds(cand, n); // ascending, so a tie keeps the lower id whatever the broadphase order
  } else n = B.count;
  let best = maxDist, bestId = -1, bx = 0, by = 0, bz = 0, bnx = 0, bny = 0, bnz = 0;
  const P = B.pos, Q = B.quat, A = B.aabb;
  const cx = (_cab[0] + _cab[3]) * 0.5, cy = (_cab[1] + _cab[4]) * 0.5, cz = (_cab[2] + _cab[5]) * 0.5;
  const ex = (_cab[3] - _cab[0]) * 0.5, ey = (_cab[4] - _cab[1]) * 0.5, ez = (_cab[5] - _cab[2]) * 0.5;
  for (let i = 0; i < n; i++) {
    const id = cand ? cand[i] : i;
    if (id < 0 || id >= B.count || !B.alive[id]) continue;
    if (mask !== -1 && (B.layer[id] & mask) === 0) continue;
    const s = B.shape[id];
    if (!s || (filter && !filter(id))) continue;
    // with a broadphase, bodies.aabb is current: skip bodies the swept bounds reach only beyond the best hit
    if (cand) { const o = id * 6; if (sweepBoxEntry(cx, cy, cz, ex, ey, ez, _cdx, _cdy, _cdz, A[o], A[o + 1], A[o + 2], A[o + 3], A[o + 4], A[o + 5], best) === Infinity) continue; }
    const p3 = id * 3, q4 = id * 4;
    const t = castShapes(shape, px, py, pz, qx, qy, qz, qw, s, P[p3], P[p3 + 1], P[p3 + 2], Q[q4], Q[q4 + 1], Q[q4 + 2], Q[q4 + 3], best);
    if (t < best) { best = t; bestId = id; bx = _hx; by = _hy; bz = _hz; bnx = _hnx; bny = _hny; bnz = _hnz; }
  }
  if (bestId < 0) return false;
  writeHit(outHit, best, bx, by, bz, bnx, bny, bnz, bestId);
  return true;
}
