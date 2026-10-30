// Collision shapes of the dynamics world (pure JS, deterministic, Node-safe): factories for sphere, capsule, box,
// convex hull, static triangle mesh and compound shapes. Each factory returns a plain object carrying its mass
// properties and the allocation-free geometric queries the broadphase, the narrowphase (GJK / SAT) and the solver
// need. A shape is defined in its own frame, which is the body frame: a body pose (position p, unit quaternion q)
// maps a local point x to the world point p + q x.
//
//   const s = box(0.5, 0.25, 1);
//   s.inertia(mass, out9);                                   // body-frame inertia tensor about the body origin
//   s.aabb(px, py, pz, qx, qy, qz, qw, out6, margin);         // world AABB minX minY minZ maxX maxY maxZ
//   s.support(dx, dy, dz, px, py, pz, qx, qy, qz, qw, out3);  // farthest world point along a world direction
//   s.project(ax, ay, az, px, py, pz, qx, qy, qz, qw, out2);  // [min, max] of x . a over the shape (SAT)
//
// Factories (construction time: they allocate and may use three's ConvexHull and Math.PI):
//   sphere(radius)                  centred at the origin
//   capsule(radius, halfHeight)     axis = local Y: segment (0, -halfHeight, 0)..(0, halfHeight, 0) swept by radius;
//                                   total height 2 * halfHeight + 2 * radius
//   box(hx, hy, hz)                 half extents; also carries the convex topology fields below, filled from a constant
//                                   table (8 corners, corner i = (i & 1 ? +hx : -hx, i & 2 ? +hy : -hy, i & 4 ? +hz :
//                                   -hz); 6 faces, planes +-e_i with d = -h_i; 12 edges) so SAT code for hulls accepts
//                                   boxes; identical to convex() of those corners with { hull: false }
//   convex(points, { hull = true }) points: Float64Array / number[] of x y z triples, or Vector3[] (>= 4, not
//                                   coplanar). Builds the hull with three/addons ConvexHull and merges adjacent
//                                   triangles whose normals agree within ~0.008 deg and whose vertices lie within
//                                   1e-6 x the hull's size of each other's plane (absorbs float32 quantisation of a
//                                   mesh modelled around its own origin) into polygon faces. `points` holds the hull's vertices in input order; `hull: false`
//                                   says the points already are the hull's vertices: all are kept (point i = input i;
//                                   topology is still derived from them). The topology is canonical (see faceVerts),
//                                   so convex(h.points, { hull: false }) reproduces h's faceStart / faceVerts / edges
//                                   / plane order exactly. The hull is RE-CENTRED at its centre of volume: local =
//                                   input + offset (offset = -centroid of the input), so a visual built from the input
//                                   points gets position = offset
//   trimesh(positions, indices)     STATIC ONLY triangle mesh (positions: Float32Array / Float64Array / number[];
//                                   indices: Uint32Array / Uint16Array / number[]), copied to Float64Array /
//                                   Uint32Array; a BVH (median split on the longest centroid axis, leaves of <= 4
//                                   triangles) is built at creation. Not re-centred
//   compound(children)              children: [{ shape, position: [x, y, z] | Vector3, quaternion: [x, y, z, w] |
//                                   Quaternion }] (both optional), stored as children: [{ shape, pos: Float64Array(3),
//                                   quat: Float64Array(4) (normalised) }]. Mass is shared by volume (uniform density);
//                                   NOT re-centred: `centroid` reports the resulting centre of mass in the body frame
//
// Fields (every shape has the full set, in the same order, so all shapes share one hidden class):
//   type             SHAPE.* (state.js); SHAPE_NAMES[type] is its name
//   volume           m^3 (trimesh: volume of its local AABB)
//   boundingRadius   max distance of any point of the shape from the shape origin (exact for every type)
//   centroid         Float64Array(3): centre of volume in the shape frame; 0 for sphere / capsule / box / convex
//                    (re-centred); the AABB centre for a trimesh; the combined centre of mass for a compound
//   radius           sphere, capsule (0 otherwise)
//   halfHeight       capsule (0 otherwise)
//   half             Float64Array(3): box half extents (zeros otherwise)
//   margin           0; reserved (aabb takes its margin as an argument)
//   offset           Float64Array(3): convex re-centring translation (zeros otherwise)
//   points           Float64Array(3n): local vertices (convex, box; null otherwise)
//   planes           Float64Array(4m): face planes nx ny nz d, unit outward normal, n . x + d <= 0 inside
//   faceStart        Int32Array(m + 1): face f has vertex indices faceVerts[faceStart[f] .. faceStart[f + 1])
//   faceVerts        Int32Array: vertex indices per face, counter-clockwise seen from outside. Canonical order: each
//                    loop starts at its smallest vertex index, faces sorted lexicographically by their sequences
//                    (planes follow the same face order)
//   edges            Int32Array(2e): unique undirected edges (vertex index pairs, smaller index first), sorted
//   vertexCount, faceCount, edgeCount   n, m, e (convex, box); trimesh: vertexCount = positions.length / 3
//   positions, indices, triCount   trimesh geometry (Float64Array, Uint32Array, number of triangles)
//   bvh              trimesh: { nodeCount, nodeMin: Float64Array(3k), nodeMax: Float64Array(3k), left: Int32Array(k),
//                    right: Int32Array(k), triStart: Int32Array(k), triCount: Int32Array(k), triOrder: Uint32Array }
//                    node 0 is the root; a leaf has left = right = -1 and owns triangles triOrder[triStart ..
//                    triStart + triCount) (1..4 of them); an internal node has triCount 0, and nodes are stored depth
//                    first, so its left child is ALWAYS node + 1 (right is the node after the left subtree)
//   children         compound children (above)
//   unitInertia      Float64Array(9): inertia tensor per unit mass about the shape origin (what inertia() scales)
//
// Methods (call them as methods: they use `this`). Deterministic (+ - * / Math.sqrt and comparisons only) and
// allocation-free: module-level scratch, so they are not re-entrant (single-threaded use; a compound's queries
// re-enter its children safely). Poses use unit quaternions.
//   inertia(mass, out9, aboutCentroid = false) -> out9   body-frame inertia TENSOR (not inverse), row-major 3x3,
//       about the shape origin (aboutCentroid: about `centroid` instead, i.e. minus the parallel-axis term; the same
//       thing except for compounds and trimeshes). Analytic for sphere / capsule / box; polyhedron integrals
//       (divergence theorem over the faces) for convex; parallel-axis sum of the children for compounds; trimesh:
//       a solid box filling its local AABB (it is static, so this only matters if someone gives it a mass).
//   aabb(px, py, pz, qx, qy, qz, qw, out6, margin = 0) -> out6   world AABB inflated by margin. Exact for sphere,
//       capsule, box (rotated extents) and convex (loop over the points); trimesh: the rotated BVH root box
//       (conservative unless the pose is axis aligned); compound: union of the children's.
//   support(dx, dy, dz, px, py, pz, qx, qy, qz, qw, out3) -> out3   world point of the shape farthest along the
//       WORLD direction d (any length; for d = 0 some point of the shape). Sphere: centre + r d^; capsule: segment
//       end + r d^; box: corner by sign per axis (+ on ties); convex: max-dot vertex (brute force, first on ties);
//       compound: max over children; trimesh: max over ALL vertices (O(n): for broadphase-sized meshes only).
//   localSupport(dx, dy, dz, out3) -> out3   the same in the shape frame (full shape, radius included).
//   project(ax, ay, az, px, py, pz, qx, qy, qz, qw, out2) -> out2   [min, max] of x . a over the shape at the pose
//       (a not normalised: the values scale with |a|). Exact for every type; trimesh loops over all vertices.
//   queryAabb(minX, minY, minZ, maxX, maxY, maxZ, outTriIndices) -> count   trimesh only (null on the others): the
//       triangles whose own AABB overlaps (touching counts) the box given in the MESH frame, written to the Int32Array
//       in BVH order; returns the number written, at most outTriIndices.length (a full array means it may have been
//       truncated).
// Implementation note (V8): a call that is not inlined passes doubles boxed as heap numbers, so the public methods
// are the same small functions for every shape type: they store their scalar arguments in module scratch and call
// the type's core (`_support`, `_local`, `_aabb`, `_project`: (shape, out) -> out), which takes no double
// arguments (their helpers are small and inline). A call site that sees several shape types therefore still calls
// one function (inlinable), and nothing below it boxes; compounds reach their children's cores the same way. Measured
// in test/dyn-shapes.mjs: 0 GCs over 1e5 support + aabb + project per type and through one mixed call site.
//
// Helpers: transformPoint(px, py, pz, qx, qy, qz, qw, x, y, z, out3), rotateVector(qx, qy, qz, qw, x, y, z, out3),
// inverseRotateVector(qx, qy, qz, qw, x, y, z, out3), inertiaOfSphere(mass, radius, out9),
// inertiaOfBox(mass, hx, hy, hz, out9), inertiaOfCapsule(mass, radius, halfHeight, out9) (out9 optional: allocated when
// omitted), SHAPE_NAMES.
import { Vector3 } from 'three';
import { ConvexHull } from 'three/addons/math/ConvexHull.js';
import { SHAPE } from './state.js';

/** Shape names indexed by SHAPE id. */
export const SHAPE_NAMES = Object.freeze(['sphere', 'capsule', 'box', 'convex', 'trimesh', 'compound']);

const LEAF_TRIS = 4;          // max triangles per BVH leaf
const STACK_SIZE = 128;       // BVH traversal stack (median split: depth ~ log2(triangles / 2), so 128 is ample)
const COPLANAR_DOT = 1 - 1e-8; // adjacent hull triangles whose unit normals agree this closely (~0.008 deg) ...
const COPLANAR_GAP = 1e-6;     // ... and whose vertices lie this close (x the hull's AABB size) to each other's plane form one face

// ---- module scratch (never escapes) --------------------------------------------------------------------------
// The public methods write their arguments here; the cores read them.
const _P = new Float64Array(8);    // pose px py pz qx qy qz qw, then the aabb margin
const _D = new Float64Array(3);    // world direction (support) / axis (project)
const _L = new Float64Array(3);    // direction in the shape frame
const _Q = new Float64Array(6);    // queryAabb box
const _R = new Float64Array(9);    // rotation matrix of the pose in _P
const _cs = new Float64Array(3);   // compound: child support point
const _cb = new Float64Array(6);   // compound: child AABB
const _cp = new Float64Array(2);   // compound: child projection
const _stack = new Int32Array(STACK_SIZE);
const _I = new Float64Array(9);    // construction: child inertia

const abs = (x) => (x < 0 ? -x : x);

// ---- pose helpers (public, scalar arguments) --------------------------------------------------------------------
/** out = q v (unit quaternion q = qx qy qz qw). Returns out (the inputs are scalars, so out may be anything). */
export function rotateVector(qx, qy, qz, qw, x, y, z, out) {
  const tx = 2 * (qy * z - qz * y), ty = 2 * (qz * x - qx * z), tz = 2 * (qx * y - qy * x);
  out[0] = x + qw * tx + (qy * tz - qz * ty);
  out[1] = y + qw * ty + (qz * tx - qx * tz);
  out[2] = z + qw * tz + (qx * ty - qy * tx);
  return out;
}

/** out = conj(q) v: a world vector expressed in the frame rotated by q. */
export function inverseRotateVector(qx, qy, qz, qw, x, y, z, out) { return rotateVector(-qx, -qy, -qz, qw, x, y, z, out); }

/** out = p + q v: a local point of a body at pose (p, q) in world space. */
export function transformPoint(px, py, pz, qx, qy, qz, qw, x, y, z, out) {
  rotateVector(qx, qy, qz, qw, x, y, z, out);
  out[0] += px; out[1] += py; out[2] += pz;
  return out;
}

// ---- scratch helpers (no double arguments) -----------------------------------------------------------------------
// _R = row-major rotation matrix of the quaternion in _P (world = R local).
function poseMatrix() {
  const qx = _P[3], qy = _P[4], qz = _P[5], qw = _P[6], m = _R;
  const x2 = qx + qx, y2 = qy + qy, z2 = qz + qz;
  const xx = qx * x2, xy = qx * y2, xz = qx * z2, yy = qy * y2, yz = qy * z2, zz = qz * z2;
  const wx = qw * x2, wy = qw * y2, wz = qw * z2;
  m[0] = 1 - (yy + zz); m[1] = xy - wz; m[2] = xz + wy;
  m[3] = xy + wz; m[4] = 1 - (xx + zz); m[5] = yz - wx;
  m[6] = xz - wy; m[7] = yz + wx; m[8] = 1 - (xx + yy);
  return m;
}
// _L = conj(q) _D (the world direction in the shape frame).
function dirToLocal() {
  const qx = -_P[3], qy = -_P[4], qz = -_P[5], qw = _P[6], x = _D[0], y = _D[1], z = _D[2];
  const tx = 2 * (qy * z - qz * y), ty = 2 * (qz * x - qx * z), tz = 2 * (qx * y - qy * x);
  _L[0] = x + qw * tx + (qy * tz - qz * ty);
  _L[1] = y + qw * ty + (qz * tx - qx * tz);
  _L[2] = z + qw * tz + (qx * ty - qy * tx);
}
// out = p + q out (a local point to world, pose from _P).
function localToWorld(out) {
  const qx = _P[3], qy = _P[4], qz = _P[5], qw = _P[6], x = out[0], y = out[1], z = out[2];
  const tx = 2 * (qy * z - qz * y), ty = 2 * (qz * x - qx * z), tz = 2 * (qx * y - qy * x);
  out[0] = _P[0] + x + qw * tx + (qy * tz - qz * ty);
  out[1] = _P[1] + y + qw * ty + (qz * tx - qx * tz);
  out[2] = _P[2] + z + qw * tz + (qx * ty - qy * tx);
  return out;
}
function writeAabb(out, cx, cy, cz, ex, ey, ez) {
  out[0] = cx - ex; out[1] = cy - ey; out[2] = cz - ez; out[3] = cx + ex; out[4] = cy + ey; out[5] = cz + ez;
  return out;
}

// ---- public methods (shared by every shape type) -------------------------------------------------------------------
function supportMethod(dx, dy, dz, px, py, pz, qx, qy, qz, qw, out) {
  _D[0] = dx; _D[1] = dy; _D[2] = dz;
  _P[0] = px; _P[1] = py; _P[2] = pz; _P[3] = qx; _P[4] = qy; _P[5] = qz; _P[6] = qw;
  return this._support(this, out);
}
function localSupportMethod(dx, dy, dz, out) {
  _L[0] = dx; _L[1] = dy; _L[2] = dz;
  return this._local(this, out);
}
function aabbMethod(px, py, pz, qx, qy, qz, qw, out, margin = 0) {
  _P[0] = px; _P[1] = py; _P[2] = pz; _P[3] = qx; _P[4] = qy; _P[5] = qz; _P[6] = qw; _P[7] = margin;
  return this._aabb(this, out);
}
function projectMethod(ax, ay, az, px, py, pz, qx, qy, qz, qw, out) {
  _D[0] = ax; _D[1] = ay; _D[2] = az;
  _P[0] = px; _P[1] = py; _P[2] = pz; _P[3] = qx; _P[4] = qy; _P[5] = qz; _P[6] = qw;
  return this._project(this, out);
}
function queryAabbMethod(minX, minY, minZ, maxX, maxY, maxZ, out) {
  _Q[0] = minX; _Q[1] = minY; _Q[2] = minZ; _Q[3] = maxX; _Q[4] = maxY; _Q[5] = maxZ;
  return queryTrimesh(this, out);
}
function shapeInertia(mass, out, aboutCentroid = false) {
  const u = this.unitInertia;
  for (let i = 0; i < 9; i++) out[i] = u[i] * mass;
  if (aboutCentroid) {
    const c = this.centroid, cx = c[0], cy = c[1], cz = c[2], c2 = cx * cx + cy * cy + cz * cz;
    out[0] -= mass * (c2 - cx * cx); out[4] -= mass * (c2 - cy * cy); out[8] -= mass * (c2 - cz * cz);
    const xy = mass * cx * cy, xz = mass * cx * cz, yz = mass * cy * cz;
    out[1] += xy; out[3] += xy; out[2] += xz; out[6] += xz; out[5] += yz; out[7] += yz;
  }
  return out;
}

// Generic world support core: local support of the direction rotated into the shape frame, mapped back to world.
function supportGeneric(s, out) {
  dirToLocal();
  s._local(s, out);
  return localToWorld(out);
}

function makeShape(type, cores) {
  return {
    type, volume: 0, boundingRadius: 0, centroid: new Float64Array(3),
    radius: 0, halfHeight: 0, half: new Float64Array(3), margin: 0, offset: new Float64Array(3),
    points: null, planes: null, faceStart: null, faceVerts: null, edges: null, vertexCount: 0, faceCount: 0, edgeCount: 0,
    positions: null, indices: null, triCount: 0, bvh: null, children: null,
    unitInertia: new Float64Array(9),
    inertia: shapeInertia, aabb: aabbMethod, support: supportMethod, localSupport: localSupportMethod, project: projectMethod,
    queryAabb: type === SHAPE.TRIMESH ? queryAabbMethod : null,
    _support: cores.support || supportGeneric, _local: cores.local, _aabb: cores.aabb, _project: cores.project,
  };
}

function checkPositive(who, name, v, allowZero = false) {
  if (!(Number.isFinite(v) && (v > 0 || (allowZero && v === 0)))) throw new RangeError(`${who}: ${name} must be ${allowZero ? '>= 0' : '> 0'} (got ${v})`);
}

// ---- analytic inertia ---------------------------------------------------------------------------------------------
function setDiag(out, a, b, c) {
  out[0] = a; out[1] = 0; out[2] = 0; out[3] = 0; out[4] = b; out[5] = 0; out[6] = 0; out[7] = 0; out[8] = c;
  return out;
}

/** Solid sphere inertia tensor about its centre (2/5 m r^2 on the diagonal). */
export function inertiaOfSphere(mass, radius, out = new Float64Array(9)) {
  const k = 0.4 * mass * radius * radius;
  return setDiag(out, k, k, k);
}

/** Solid box inertia tensor about its centre, from half extents. */
export function inertiaOfBox(mass, hx, hy, hz, out = new Float64Array(9)) {
  const x2 = hx * hx, y2 = hy * hy, z2 = hz * hz, k = mass / 3;
  return setDiag(out, k * (y2 + z2), k * (x2 + z2), k * (x2 + y2));
}

/**
 * Solid capsule (axis Y) inertia tensor about its centre: a cylinder of height 2 * halfHeight plus two hemispheres,
 * mass shared by volume; each hemisphere's transverse term is 83/320 m r^2 about its own centre of mass (3r/8 from
 * its flat face) moved to the capsule centre by the parallel-axis rule.
 */
export function inertiaOfCapsule(mass, radius, halfHeight, out = new Float64Array(9)) {
  const r = radius, h = 2 * halfHeight, r2 = r * r;
  const mc = mass * h / (h + 4 * r / 3); // cylinder share (pi r^2 h of pi r^2 h + 4/3 pi r^3)
  const ms = mass - mc;                  // both hemispheres
  const iy = mc * r2 / 2 + ms * 0.4 * r2;
  const ix = mc * (h * h / 12 + r2 / 4) + ms * (0.4 * r2 + h * h / 4 + 3 * h * r / 8);
  return setDiag(out, ix, iy, ix);
}

// ---- sphere ---------------------------------------------------------------------------------------------------------
const SPHERE_CORES = {
  support(s, out) {
    const dx = _D[0], dy = _D[1], dz = _D[2], l2 = dx * dx + dy * dy + dz * dz, k = l2 > 0 ? s.radius / Math.sqrt(l2) : 0;
    out[0] = _P[0] + dx * k; out[1] = _P[1] + dy * k; out[2] = _P[2] + dz * k;
    return out;
  },
  local(s, out) {
    const dx = _L[0], dy = _L[1], dz = _L[2], l2 = dx * dx + dy * dy + dz * dz, k = l2 > 0 ? s.radius / Math.sqrt(l2) : 0;
    out[0] = dx * k; out[1] = dy * k; out[2] = dz * k;
    return out;
  },
  aabb(s, out) {
    const e = s.radius + _P[7];
    return writeAabb(out, _P[0], _P[1], _P[2], e, e, e);
  },
  project(s, out) {
    const ax = _D[0], ay = _D[1], az = _D[2];
    const c = _P[0] * ax + _P[1] * ay + _P[2] * az, e = s.radius * Math.sqrt(ax * ax + ay * ay + az * az);
    out[0] = c - e; out[1] = c + e;
    return out;
  },
};

/** Sphere of `radius` centred at the shape origin. */
export function sphere(radius) {
  checkPositive('sphere', 'radius', radius);
  const s = makeShape(SHAPE.SPHERE, SPHERE_CORES);
  s.radius = radius;
  s.volume = 4 / 3 * Math.PI * radius * radius * radius;
  s.boundingRadius = radius;
  inertiaOfSphere(1, radius, s.unitInertia);
  return s;
}

// ---- capsule --------------------------------------------------------------------------------------------------------
const CAPSULE_CORES = {
  local(s, out) {
    const dx = _L[0], dy = _L[1], dz = _L[2], l2 = dx * dx + dy * dy + dz * dz, k = l2 > 0 ? s.radius / Math.sqrt(l2) : 0;
    out[0] = dx * k; out[1] = (dy >= 0 ? s.halfHeight : -s.halfHeight) + dy * k; out[2] = dz * k;
    return out;
  },
  aabb(s, out) {
    // world axis = column 1 of the rotation matrix
    const qx = _P[3], qy = _P[4], qz = _P[5], qw = _P[6], hh = s.halfHeight, e = s.radius + _P[7];
    const ux = 2 * (qx * qy - qz * qw), uy = 1 - 2 * (qx * qx + qz * qz), uz = 2 * (qy * qz + qx * qw);
    return writeAabb(out, _P[0], _P[1], _P[2], abs(ux) * hh + e, abs(uy) * hh + e, abs(uz) * hh + e);
  },
  project(s, out) {
    const qx = _P[3], qy = _P[4], qz = _P[5], qw = _P[6], ax = _D[0], ay = _D[1], az = _D[2];
    const ux = 2 * (qx * qy - qz * qw), uy = 1 - 2 * (qx * qx + qz * qz), uz = 2 * (qy * qz + qx * qw);
    const c = _P[0] * ax + _P[1] * ay + _P[2] * az;
    const e = abs(ux * ax + uy * ay + uz * az) * s.halfHeight + s.radius * Math.sqrt(ax * ax + ay * ay + az * az);
    out[0] = c - e; out[1] = c + e;
    return out;
  },
};

/** Capsule along local Y: the segment (0, +-halfHeight, 0) swept by `radius`. */
export function capsule(radius, halfHeight) {
  checkPositive('capsule', 'radius', radius);
  checkPositive('capsule', 'halfHeight', halfHeight, true);
  const s = makeShape(SHAPE.CAPSULE, CAPSULE_CORES);
  s.radius = radius; s.halfHeight = halfHeight;
  s.volume = Math.PI * radius * radius * (2 * halfHeight) + 4 / 3 * Math.PI * radius * radius * radius;
  s.boundingRadius = halfHeight + radius;
  inertiaOfCapsule(1, radius, halfHeight, s.unitInertia);
  return s;
}

// ---- box ------------------------------------------------------------------------------------------------------------
const BOX_CORES = {
  local(s, out) {
    const h = s.half;
    out[0] = _L[0] >= 0 ? h[0] : -h[0]; out[1] = _L[1] >= 0 ? h[1] : -h[1]; out[2] = _L[2] >= 0 ? h[2] : -h[2];
    return out;
  },
  aabb(s, out) {
    const m = poseMatrix(), h = s.half, hx = h[0], hy = h[1], hz = h[2], g = _P[7];
    return writeAabb(out, _P[0], _P[1], _P[2], abs(m[0]) * hx + abs(m[1]) * hy + abs(m[2]) * hz + g,
      abs(m[3]) * hx + abs(m[4]) * hy + abs(m[5]) * hz + g, abs(m[6]) * hx + abs(m[7]) * hy + abs(m[8]) * hz + g);
  },
  project(s, out) {
    const m = poseMatrix(), h = s.half, ax = _D[0], ay = _D[1], az = _D[2];
    const c = _P[0] * ax + _P[1] * ay + _P[2] * az;
    // local axis components = columns of the rotation matrix dotted with a
    const e = abs(m[0] * ax + m[3] * ay + m[6] * az) * h[0] + abs(m[1] * ax + m[4] * ay + m[7] * az) * h[1] +
      abs(m[2] * ax + m[5] * ay + m[8] * az) * h[2];
    out[0] = c - e; out[1] = c + e;
    return out;
  },
};

// Box topology over corner i = sign bits of i (x: bit 0, y: bit 1, z: bit 2), already canonical: faces CCW from
// outside, each starting at its smallest corner, sorted; BOX_PLANES gives each face's normal axis (0..2) and sign.
const BOX_LOOPS = [[0, 1, 5, 4], [0, 2, 3, 1], [0, 4, 6, 2], [1, 3, 7, 5], [2, 6, 7, 3], [4, 5, 7, 6]];
const BOX_PLANES = [[1, -1], [2, -1], [0, -1], [0, 1], [1, 1], [2, 1]];
const BOX_TOPOLOGY = canonicalTopology(BOX_LOOPS);

/** Box with half extents hx, hy, hz centred at the shape origin (also carries hull topology: see the header). */
export function box(hx, hy, hz) {
  checkPositive('box', 'hx', hx); checkPositive('box', 'hy', hy); checkPositive('box', 'hz', hz);
  const s = makeShape(SHAPE.BOX, BOX_CORES);
  const h = s.half;
  h[0] = hx; h[1] = hy; h[2] = hz;
  s.volume = 8 * hx * hy * hz;
  s.boundingRadius = Math.sqrt(hx * hx + hy * hy + hz * hz);
  inertiaOfBox(1, hx, hy, hz, s.unitInertia);
  const p = s.points = new Float64Array(24);
  for (let i = 0; i < 8; i++) { p[i * 3] = i & 1 ? hx : -hx; p[i * 3 + 1] = i & 2 ? hy : -hy; p[i * 3 + 2] = i & 4 ? hz : -hz; }
  s.faceStart = BOX_TOPOLOGY.faceStart.slice(); s.faceVerts = BOX_TOPOLOGY.faceVerts.slice(); s.edges = BOX_TOPOLOGY.edges.slice();
  s.vertexCount = 8; s.faceCount = 6; s.edgeCount = 12;
  const pl = s.planes = new Float64Array(24);
  for (let f = 0; f < 6; f++) { const [axis, sign] = BOX_PLANES[f]; pl[f * 4 + axis] = sign; pl[f * 4 + 3] = -h[axis]; }
  return s;
}

// ---- point-set cores (convex: s.points; trimesh support / project: s.positions) ----------------------------------------
// max-dot vertex of a packed xyz array along _L (first one on ties)
function maxDotLocal(p, out) {
  const dx = _L[0], dy = _L[1], dz = _L[2];
  let best = -Infinity, bi = 0;
  for (let i = 0, n = p.length; i < n; i += 3) {
    const d = p[i] * dx + p[i + 1] * dy + p[i + 2] * dz;
    if (d > best) { best = d; bi = i; }
  }
  out[0] = p[bi]; out[1] = p[bi + 1]; out[2] = p[bi + 2];
  return out;
}
// [min, max] of x . _D over a packed xyz array at the pose in _P
function projectPoints(p, out) {
  dirToLocal();
  const lx = _L[0], ly = _L[1], lz = _L[2];
  let lo = Infinity, hi = -Infinity;
  for (let i = 0, n = p.length; i < n; i += 3) {
    const d = p[i] * lx + p[i + 1] * ly + p[i + 2] * lz;
    if (d < lo) lo = d; if (d > hi) hi = d;
  }
  const c = _P[0] * _D[0] + _P[1] * _D[1] + _P[2] * _D[2];
  out[0] = c + lo; out[1] = c + hi;
  return out;
}
const CONVEX_CORES = {
  local(s, out) { return maxDotLocal(s.points, out); },
  aabb(s, out) {
    const m = poseMatrix();
    const m0 = m[0], m1 = m[1], m2 = m[2], m3 = m[3], m4 = m[4], m5 = m[5], m6 = m[6], m7 = m[7], m8 = m[8];
    const p = s.points, n = p.length;
    let minX = Infinity, minY = Infinity, minZ = Infinity, maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (let i = 0; i < n; i += 3) {
      const x = p[i], y = p[i + 1], z = p[i + 2];
      const wx = m0 * x + m1 * y + m2 * z, wy = m3 * x + m4 * y + m5 * z, wz = m6 * x + m7 * y + m8 * z;
      if (wx < minX) minX = wx; if (wx > maxX) maxX = wx;
      if (wy < minY) minY = wy; if (wy > maxY) maxY = wy;
      if (wz < minZ) minZ = wz; if (wz > maxZ) maxZ = wz;
    }
    const g = _P[7];
    out[0] = _P[0] + minX - g; out[1] = _P[1] + minY - g; out[2] = _P[2] + minZ - g;
    out[3] = _P[0] + maxX + g; out[4] = _P[1] + maxY + g; out[5] = _P[2] + maxZ + g;
    return out;
  },
  project(s, out) { return projectPoints(s.points, out); },
};

function toVectors(points) {
  const out = [];
  if (points.length && typeof points[0] === 'object') {
    for (let i = 0; i < points.length; i++) out.push(new Vector3(points[i].x, points[i].y, points[i].z));
  } else {
    for (let i = 0; i + 2 < points.length; i += 3) out.push(new Vector3(points[i], points[i + 1], points[i + 2]));
  }
  for (let i = 0; i < out.length; i++) {
    const v = out[i];
    if (!(Number.isFinite(v.x) && Number.isFinite(v.y) && Number.isFinite(v.z))) throw new RangeError(`convex: point ${i} is not finite`);
  }
  return out;
}

/**
 * Hull topology of a point set: three's quickhull triangles, nearly coplanar neighbours merged into polygon faces
 * (union-find over twin half-edges), each face's boundary walked counter-clockwise from outside, then put in
 * canonical form (canonicalTopology). Vertices are numbered in input order: keepOrder keeps every input point (vertex
 * i = vecs[i]), otherwise only the hull's vertices. Planes are filled by setTopology.
 */
function hullTopology(vecs, keepOrder) {
  const hull = new ConvexHull().setFromPoints(vecs);
  const faces = hull.faces, nf = faces.length;
  if (nf < 4) throw new RangeError('convex: degenerate point set (no 3D hull)');
  const used = new Set();
  for (let f = 0; f < nf; f++) {
    let e = faces[f].edge;
    for (let k = 0; k < 3; k++, e = e.next) used.add(e.head().point);
  }
  const index = new Map(), verts = [];
  for (let i = 0; i < vecs.length; i++) if (keepOrder || used.has(vecs[i])) { index.set(vecs[i], verts.length); verts.push(vecs[i]); }
  const fid = new Map();
  for (let f = 0; f < nf; f++) fid.set(faces[f], f);
  // union nearly coplanar neighbours. The test compares the two groups' root triangles (not the neighbours), so a
  // face cannot drift along a chain of slightly bent triangles: normals within COPLANAR_DOT and each root's vertices
  // within COPLANAR_GAP * size of the other root's plane. size = the largest AABB side of the points: translation
  // invariant, so the re-centred points (a hull: false rebuild) make the same decisions as the original input
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < verts.length; i++) {
    const v = verts[i];
    lo[0] = Math.min(lo[0], v.x); lo[1] = Math.min(lo[1], v.y); lo[2] = Math.min(lo[2], v.z);
    hi[0] = Math.max(hi[0], v.x); hi[1] = Math.max(hi[1], v.y); hi[2] = Math.max(hi[2], v.z);
  }
  const gapTol = COPLANAR_GAP * Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
  const gap = (F, G) => {
    let m = 0, e = G.edge;
    for (let k = 0; k < 3; k++, e = e.next) m = Math.max(m, Math.abs(F.normal.dot(e.head().point) - F.constant));
    return m;
  };
  const parent = new Int32Array(nf);
  for (let f = 0; f < nf; f++) parent[f] = f;
  const find = (f) => { while (parent[f] !== f) { parent[f] = parent[parent[f]]; f = parent[f]; } return f; };
  for (let f = 0; f < nf; f++) {
    let e = faces[f].edge;
    for (let k = 0; k < 3; k++, e = e.next) {
      const g = fid.get(e.twin.face);
      if (g < f) continue;
      const a = find(f), b = find(g), A = faces[a], B = faces[b];
      if (a !== b && A.normal.dot(B.normal) > COPLANAR_DOT && gap(A, B) <= gapTol && gap(B, A) <= gapTol) {
        parent[a > b ? a : b] = a < b ? a : b; // smaller root wins
      }
    }
  }
  const loops = [];
  const seen = new Uint8Array(nf);
  for (let f = 0; f < nf; f++) {
    const root = find(f);
    if (seen[root]) continue;
    seen[root] = 1;
    // boundary half-edges of this group: tail -> head, keyed by tail
    const next = new Map();
    let first = -1;
    for (let g = f; g < nf; g++) {
      if (find(g) !== root) continue;
      let e = faces[g].edge;
      for (let k = 0; k < 3; k++, e = e.next) {
        if (find(fid.get(e.twin.face)) === root) continue;
        const t = index.get(e.tail().point), h = index.get(e.head().point);
        next.set(t, h);
        if (first < 0) first = t;
      }
    }
    const loop = [];
    let v = first;
    do {
      loop.push(v);
      v = next.get(v);
      if (loop.length > verts.length) throw new Error('convex: non-manifold face boundary');
    } while (v !== first);
    loops.push(loop);
  }
  const pts = new Float64Array(verts.length * 3);
  for (let i = 0; i < verts.length; i++) { pts[i * 3] = verts[i].x; pts[i * 3 + 1] = verts[i].y; pts[i * 3 + 2] = verts[i].z; }
  const t = canonicalTopology(loops);
  t.points = pts;
  return t;
}

/**
 * Canonical topology of face loops (vertex indices, CCW from outside): each loop rotated to start at its smallest
 * index (orientation kept), faces sorted lexicographically by their sequences (two faces never share a directed edge,
 * so no ties), edges (smaller index first) sorted. It depends only on the faces' vertex cycles, not on the order in
 * which quickhull created its triangles, so rebuilding a hull from its own points reproduces the same arrays.
 */
function canonicalTopology(loops) {
  const canon = loops.map((l) => {
    let m = 0;
    for (let i = 1; i < l.length; i++) if (l[i] < l[m]) m = i;
    return l.slice(m).concat(l.slice(0, m));
  });
  canon.sort((a, b) => {
    for (let i = 0; i < a.length && i < b.length; i++) if (a[i] !== b[i]) return a[i] - b[i];
    return a.length - b.length;
  });
  const faceStart = [0], faceVerts = [], edges = [];
  for (const l of canon) {
    for (let i = 0; i < l.length; i++) {
      const v = l[i], w = l[i + 1 < l.length ? i + 1 : 0];
      faceVerts.push(v);
      if (v < w) edges.push([v, w]); // each undirected edge appears once in this direction
    }
    faceStart.push(faceVerts.length);
  }
  edges.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  return { faceStart: Int32Array.from(faceStart), faceVerts: Int32Array.from(faceVerts), edges: Int32Array.from(edges.flat()) };
}

// Store topology on a shape and compute its planes (Newell normal of each face loop; d from the loop's mean).
function setTopology(s, t) {
  s.points = t.points; s.faceStart = t.faceStart; s.faceVerts = t.faceVerts; s.edges = t.edges;
  s.vertexCount = t.points.length / 3; s.faceCount = t.faceStart.length - 1; s.edgeCount = t.edges.length / 2;
  computePlanes(s);
}
function computePlanes(s) {
  const p = s.points, fs = s.faceStart, fv = s.faceVerts, m = s.faceCount;
  const planes = s.planes = new Float64Array(4 * m);
  for (let f = 0; f < m; f++) {
    const a = fs[f], b = fs[f + 1];
    let nx = 0, ny = 0, nz = 0, cx = 0, cy = 0, cz = 0;
    for (let k = a; k < b; k++) {
      const i = fv[k] * 3, j = fv[k + 1 < b ? k + 1 : a] * 3;
      nx += (p[i + 1] - p[j + 1]) * (p[i + 2] + p[j + 2]);
      ny += (p[i + 2] - p[j + 2]) * (p[i] + p[j]);
      nz += (p[i] - p[j]) * (p[i + 1] + p[j + 1]);
      cx += p[i]; cy += p[i + 1]; cz += p[i + 2];
    }
    const l = Math.sqrt(nx * nx + ny * ny + nz * nz), k = b - a;
    nx /= l; ny /= l; nz /= l;
    planes[f * 4] = nx; planes[f * 4 + 1] = ny; planes[f * 4 + 2] = nz;
    planes[f * 4 + 3] = -(nx * cx + ny * cy + nz * cz) / k;
  }
}

// Volume and centre of volume of a closed polyhedron (fan tetrahedra from the mean of its face vertices).
function polyVolumeCentroid(p, fs, fv, outC) {
  let rx = 0, ry = 0, rz = 0;
  const nv = fv.length;
  for (let k = 0; k < nv; k++) { rx += p[fv[k] * 3]; ry += p[fv[k] * 3 + 1]; rz += p[fv[k] * 3 + 2]; }
  rx /= nv; ry /= nv; rz /= nv;
  let vol = 0, cx = 0, cy = 0, cz = 0;
  for (let f = 0, m = fs.length - 1; f < m; f++) {
    const a = fv[fs[f]] * 3;
    const ax = p[a] - rx, ay = p[a + 1] - ry, az = p[a + 2] - rz;
    for (let k = fs[f] + 1; k + 1 < fs[f + 1]; k++) {
      const b = fv[k] * 3, c = fv[k + 1] * 3;
      const bx = p[b] - rx, by = p[b + 1] - ry, bz = p[b + 2] - rz, qx = p[c] - rx, qy = p[c + 1] - ry, qz = p[c + 2] - rz;
      const v = (ax * (by * qz - bz * qy) + ay * (bz * qx - bx * qz) + az * (bx * qy - by * qx)) / 6;
      vol += v; cx += v * (ax + bx + qx) / 4; cy += v * (ay + by + qy) / 4; cz += v * (az + bz + qz) / 4;
    }
  }
  outC[0] = rx + cx / vol; outC[1] = ry + cy / vol; outC[2] = rz + cz / vol;
  return vol;
}

// Unit-mass inertia about the origin of a closed polyhedron of volume `vol`: covariance C = sum over the tetrahedra
// (0, a, b, c) of det / 120 (s s^T + a a^T + b b^T + c c^T), s = a + b + c; I = trace(C) E - C.
function polyUnitInertia(p, fs, fv, vol, out) {
  let xx = 0, yy = 0, zz = 0, xy = 0, xz = 0, yz = 0;
  for (let f = 0, m = fs.length - 1; f < m; f++) {
    const a = fv[fs[f]] * 3, ax = p[a], ay = p[a + 1], az = p[a + 2];
    for (let k = fs[f] + 1; k + 1 < fs[f + 1]; k++) {
      const b = fv[k] * 3, c = fv[k + 1] * 3;
      const bx = p[b], by = p[b + 1], bz = p[b + 2], cx = p[c], cy = p[c + 1], cz = p[c + 2];
      const det = ax * (by * cz - bz * cy) + ay * (bz * cx - bx * cz) + az * (bx * cy - by * cx);
      const sx = ax + bx + cx, sy = ay + by + cy, sz = az + bz + cz;
      xx += det * (sx * sx + ax * ax + bx * bx + cx * cx);
      yy += det * (sy * sy + ay * ay + by * by + cy * cy);
      zz += det * (sz * sz + az * az + bz * bz + cz * cz);
      xy += det * (sx * sy + ax * ay + bx * by + cx * cy);
      xz += det * (sx * sz + ax * az + bx * bz + cx * cz);
      yz += det * (sy * sz + ay * az + by * bz + cy * cz);
    }
  }
  const k = 1 / (120 * vol);
  xx *= k; yy *= k; zz *= k; xy *= k; xz *= k; yz *= k;
  out[0] = yy + zz; out[1] = -xy; out[2] = -xz;
  out[3] = -xy; out[4] = xx + zz; out[5] = -yz;
  out[6] = -xz; out[7] = -yz; out[8] = xx + yy;
  return out;
}

/**
 * Convex hull of `points` (see the header for the options and the topology fields). Re-centred at its centre of
 * volume: local point = input point + `offset` (offset = -centroid of the input), so a visual mesh built from the input
 * points must be placed at `offset` in the body frame.
 */
export function convex(points, { hull = true } = {}) {
  const vecs = toVectors(points);
  if (vecs.length < 4) throw new RangeError(`convex: need at least 4 points (got ${vecs.length})`);
  const s = makeShape(SHAPE.CONVEX, CONVEX_CORES);
  const t = hullTopology(vecs, !hull);
  const p = t.points, n = p.length;
  let ext = 0;
  for (let i = 0; i < n; i++) ext = Math.max(ext, Math.abs(p[i]));
  const c = new Float64Array(3);
  const vol = polyVolumeCentroid(p, t.faceStart, t.faceVerts, c);
  if (!(vol > 1e-12 * ext * ext * ext)) throw new RangeError('convex: degenerate point set (zero volume)');
  for (let i = 0; i < n; i += 3) { p[i] -= c[0]; p[i + 1] -= c[1]; p[i + 2] -= c[2]; }
  s.offset[0] = -c[0]; s.offset[1] = -c[1]; s.offset[2] = -c[2];
  setTopology(s, t);
  s.volume = vol;
  polyUnitInertia(p, s.faceStart, s.faceVerts, vol, s.unitInertia);
  let r2 = 0;
  for (let i = 0; i < n; i += 3) r2 = Math.max(r2, p[i] * p[i] + p[i + 1] * p[i + 1] + p[i + 2] * p[i + 2]);
  s.boundingRadius = Math.sqrt(r2);
  return s;
}

// ---- triangle mesh ----------------------------------------------------------------------------------------------------
const TRIMESH_CORES = {
  local(s, out) { return maxDotLocal(s.positions, out); },
  aabb(s, out) {
    const b = s.bvh, lo = b.nodeMin, hi = b.nodeMax;
    const cx = (lo[0] + hi[0]) * 0.5, cy = (lo[1] + hi[1]) * 0.5, cz = (lo[2] + hi[2]) * 0.5;
    const hx = (hi[0] - lo[0]) * 0.5, hy = (hi[1] - lo[1]) * 0.5, hz = (hi[2] - lo[2]) * 0.5;
    const m = poseMatrix(), g = _P[7];
    return writeAabb(out, _P[0] + m[0] * cx + m[1] * cy + m[2] * cz, _P[1] + m[3] * cx + m[4] * cy + m[5] * cz,
      _P[2] + m[6] * cx + m[7] * cy + m[8] * cz, abs(m[0]) * hx + abs(m[1]) * hy + abs(m[2]) * hz + g,
      abs(m[3]) * hx + abs(m[4]) * hy + abs(m[5]) * hz + g, abs(m[6]) * hx + abs(m[7]) * hy + abs(m[8]) * hz + g);
  },
  project(s, out) { return projectPoints(s.positions, out); },
};

// Triangles whose AABB overlaps the box in _Q (iterative traversal with the module stack).
function queryTrimesh(s, out) {
  const b = s.bvh, nMin = b.nodeMin, nMax = b.nodeMax, left = b.left, right = b.right, order = b.triOrder;
  const tStart = b.triStart, tCount = b.triCount, pos = s.positions, idx = s.indices;
  const minX = _Q[0], minY = _Q[1], minZ = _Q[2], maxX = _Q[3], maxY = _Q[4], maxZ = _Q[5];
  const cap = out.length;
  let count = 0, sp = 0;
  _stack[sp++] = 0;
  while (sp > 0) {
    const node = _stack[--sp], o = node * 3;
    if (nMin[o] > maxX || nMax[o] < minX || nMin[o + 1] > maxY || nMax[o + 1] < minY || nMin[o + 2] > maxZ || nMax[o + 2] < minZ) continue;
    if (left[node] >= 0) { _stack[sp++] = right[node]; _stack[sp++] = left[node]; continue; }
    for (let k = tStart[node], e = k + tCount[node]; k < e; k++) {
      const t = order[k], i0 = idx[t * 3] * 3, i1 = idx[t * 3 + 1] * 3, i2 = idx[t * 3 + 2] * 3;
      let a = pos[i0], c = pos[i1], d = pos[i2];
      if ((a < c ? (a < d ? a : d) : (c < d ? c : d)) > maxX || (a > c ? (a > d ? a : d) : (c > d ? c : d)) < minX) continue;
      a = pos[i0 + 1]; c = pos[i1 + 1]; d = pos[i2 + 1];
      if ((a < c ? (a < d ? a : d) : (c < d ? c : d)) > maxY || (a > c ? (a > d ? a : d) : (c > d ? c : d)) < minY) continue;
      a = pos[i0 + 2]; c = pos[i1 + 2]; d = pos[i2 + 2];
      if ((a < c ? (a < d ? a : d) : (c < d ? c : d)) > maxZ || (a > c ? (a > d ? a : d) : (c > d ? c : d)) < minZ) continue;
      if (count >= cap) return count;
      out[count++] = t;
    }
  }
  return count;
}

// Deterministic quickselect: reorder order[lo..hi] (inclusive) so that order[k] holds the triangle of rank k by
// (centroid[axis], triangle id), smaller ranks before it and larger after.
function selectNth(order, cen, axis, lo, hi, k) {
  const less = (u, v) => { const a = cen[u * 3 + axis], b = cen[v * 3 + axis]; return a < b || (a === b && u < v); };
  while (hi > lo) {
    let a = order[lo], b = order[(lo + hi) >> 1], c = order[hi], t;
    if (less(b, a)) { t = a; a = b; b = t; }
    if (less(c, b)) { b = c; if (less(b, a)) b = a; }
    const pivot = b;
    let i = lo, j = hi;
    while (i <= j) {
      while (less(order[i], pivot)) i++;
      while (less(pivot, order[j])) j--;
      if (i <= j) { t = order[i]; order[i] = order[j]; order[j] = t; i++; j--; }
    }
    if (k <= j) hi = j;
    else if (k >= i) lo = i;
    else return;
  }
}

function buildBvh(pos, idx, nt) {
  const cen = new Float64Array(3 * nt), tmin = new Float64Array(3 * nt), tmax = new Float64Array(3 * nt);
  for (let t = 0; t < nt; t++) {
    for (let a = 0; a < 3; a++) {
      const v0 = pos[idx[t * 3] * 3 + a], v1 = pos[idx[t * 3 + 1] * 3 + a], v2 = pos[idx[t * 3 + 2] * 3 + a];
      tmin[t * 3 + a] = Math.min(v0, v1, v2); tmax[t * 3 + a] = Math.max(v0, v1, v2);
      cen[t * 3 + a] = (v0 + v1 + v2) / 3;
    }
  }
  const order = new Uint32Array(nt);
  for (let t = 0; t < nt; t++) order[t] = t;
  const cap = 2 * nt + 1;
  const nodeMin = new Float64Array(3 * cap), nodeMax = new Float64Array(3 * cap);
  const left = new Int32Array(cap), right = new Int32Array(cap), triStart = new Int32Array(cap), triCount = new Int32Array(cap);
  let nodeCount = 0;
  const build = (start, end) => {
    const node = nodeCount++, o = node * 3;
    let x0 = Infinity, y0 = Infinity, z0 = Infinity, x1 = -Infinity, y1 = -Infinity, z1 = -Infinity;
    let cx0 = Infinity, cy0 = Infinity, cz0 = Infinity, cx1 = -Infinity, cy1 = -Infinity, cz1 = -Infinity;
    for (let i = start; i < end; i++) {
      const t = order[i] * 3;
      x0 = Math.min(x0, tmin[t]); y0 = Math.min(y0, tmin[t + 1]); z0 = Math.min(z0, tmin[t + 2]);
      x1 = Math.max(x1, tmax[t]); y1 = Math.max(y1, tmax[t + 1]); z1 = Math.max(z1, tmax[t + 2]);
      cx0 = Math.min(cx0, cen[t]); cy0 = Math.min(cy0, cen[t + 1]); cz0 = Math.min(cz0, cen[t + 2]);
      cx1 = Math.max(cx1, cen[t]); cy1 = Math.max(cy1, cen[t + 1]); cz1 = Math.max(cz1, cen[t + 2]);
    }
    nodeMin[o] = x0; nodeMin[o + 1] = y0; nodeMin[o + 2] = z0; nodeMax[o] = x1; nodeMax[o + 1] = y1; nodeMax[o + 2] = z1;
    triStart[node] = start;
    if (end - start <= LEAF_TRIS) { left[node] = right[node] = -1; triCount[node] = end - start; return node; }
    const ex = cx1 - cx0, ey = cy1 - cy0, ez = cz1 - cz0;
    const axis = ex >= ey && ex >= ez ? 0 : ey >= ez ? 1 : 2;
    const mid = (start + end) >> 1;
    selectNth(order, cen, axis, start, end - 1, mid);
    triCount[node] = 0;
    left[node] = build(start, mid);
    right[node] = build(mid, end);
    return node;
  };
  build(0, nt);
  const k = nodeCount;
  return {
    nodeCount: k, nodeMin: nodeMin.slice(0, 3 * k), nodeMax: nodeMax.slice(0, 3 * k), left: left.slice(0, k), right: right.slice(0, k),
    triStart: triStart.slice(0, k), triCount: triCount.slice(0, k), triOrder: order,
  };
}

/** Static triangle mesh with a BVH (see the header). Mass properties: a solid box filling its local AABB. */
export function trimesh(positions, indices) {
  const pos = new Float64Array(positions), idx = new Uint32Array(indices);
  const nv = (pos.length / 3) | 0, nt = (idx.length / 3) | 0;
  if (nt < 1) throw new RangeError('trimesh: needs at least one triangle');
  for (let i = 0; i < pos.length; i++) if (!Number.isFinite(pos[i])) throw new RangeError(`trimesh: position ${i} is not finite`);
  for (let i = 0; i < nt * 3; i++) if (idx[i] >= nv) throw new RangeError(`trimesh: index ${idx[i]} >= vertex count ${nv}`);
  const s = makeShape(SHAPE.TRIMESH, TRIMESH_CORES);
  s.positions = pos; s.indices = nt * 3 === idx.length ? idx : idx.slice(0, nt * 3); s.triCount = nt; s.vertexCount = nv;
  s.bvh = buildBvh(pos, s.indices, nt);
  const lo = s.bvh.nodeMin, hi = s.bvh.nodeMax;
  const hx = (hi[0] - lo[0]) / 2, hy = (hi[1] - lo[1]) / 2, hz = (hi[2] - lo[2]) / 2;
  const c = s.centroid;
  c[0] = (lo[0] + hi[0]) / 2; c[1] = (lo[1] + hi[1]) / 2; c[2] = (lo[2] + hi[2]) / 2;
  s.volume = 8 * hx * hy * hz;
  inertiaOfBox(1, hx, hy, hz, s.unitInertia);
  addParallelAxis(s.unitInertia, 1, c[0], c[1], c[2]);
  let r2 = 0;
  for (let i = 0; i < nv * 3; i += 3) r2 = Math.max(r2, pos[i] * pos[i] + pos[i + 1] * pos[i + 1] + pos[i + 2] * pos[i + 2]);
  s.boundingRadius = Math.sqrt(r2);
  return s;
}

// io += m (|d|^2 E - d d^T): moves an inertia tensor from the centre of mass to a point at -d from it.
function addParallelAxis(io, m, dx, dy, dz) {
  const d2 = dx * dx + dy * dy + dz * dz;
  io[0] += m * (d2 - dx * dx); io[4] += m * (d2 - dy * dy); io[8] += m * (d2 - dz * dz);
  io[1] -= m * dx * dy; io[3] -= m * dx * dy; io[2] -= m * dx * dz; io[6] -= m * dx * dz; io[5] -= m * dy * dz; io[7] -= m * dy * dz;
}

// ---- compound ---------------------------------------------------------------------------------------------------------
// The compound cores read their own inputs from scratch into locals first, then write each child's composed pose /
// direction into the same scratch and call the child's core (children never touch the parent's locals).
// _P[0..6] = parent pose (px..qw locals) composed with child (pos, quat): position p + q pos, rotation q * quat.
function setChildPose(px, py, pz, qx, qy, qz, qw, c) {
  const cp = c.pos, cq = c.quat, x = cp[0], y = cp[1], z = cp[2], bx = cq[0], by = cq[1], bz = cq[2], bw = cq[3];
  const tx = 2 * (qy * z - qz * y), ty = 2 * (qz * x - qx * z), tz = 2 * (qx * y - qy * x);
  _P[0] = px + x + qw * tx + (qy * tz - qz * ty);
  _P[1] = py + y + qw * ty + (qz * tx - qx * tz);
  _P[2] = pz + z + qw * tz + (qx * ty - qy * tx);
  _P[3] = qw * bx + qx * bw + qy * bz - qz * by;
  _P[4] = qw * by - qx * bz + qy * bw + qz * bx;
  _P[5] = qw * bz + qx * by - qy * bx + qz * bw;
  _P[6] = qw * bw - qx * bx - qy * by - qz * bz;
}
const COMPOUND_CORES = {
  local(s, out) {
    const ch = s.children, dx = _L[0], dy = _L[1], dz = _L[2];
    let best = -Infinity, sx = 0, sy = 0, sz = 0;
    for (let i = 0; i < ch.length; i++) {
      const c = ch[i], cp = c.pos, q = c.quat, bx = q[0], by = q[1], bz = q[2], bw = q[3];
      // child-frame direction conj(quat) d
      let tx = 2 * (bz * dy - by * dz), ty = 2 * (bx * dz - bz * dx), tz = 2 * (by * dx - bx * dy);
      _L[0] = dx + bw * tx + (bz * ty - by * tz);
      _L[1] = dy + bw * ty + (bx * tz - bz * tx);
      _L[2] = dz + bw * tz + (by * tx - bx * ty);
      const cs = c.shape;
      cs._local(cs, _cs);
      // back to the compound frame: pos + quat cs
      const x = _cs[0], y = _cs[1], z = _cs[2];
      tx = 2 * (by * z - bz * y); ty = 2 * (bz * x - bx * z); tz = 2 * (bx * y - by * x);
      const wx = cp[0] + x + bw * tx + (by * tz - bz * ty);
      const wy = cp[1] + y + bw * ty + (bz * tx - bx * tz);
      const wz = cp[2] + z + bw * tz + (bx * ty - by * tx);
      const d = wx * dx + wy * dy + wz * dz;
      if (d > best) { best = d; sx = wx; sy = wy; sz = wz; }
    }
    out[0] = sx; out[1] = sy; out[2] = sz;
    return out;
  },

  aabb(s, out) {
    const ch = s.children, px = _P[0], py = _P[1], pz = _P[2], qx = _P[3], qy = _P[4], qz = _P[5], qw = _P[6], g = _P[7];
    let minX = Infinity, minY = Infinity, minZ = Infinity, maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (let i = 0; i < ch.length; i++) {
      const c = ch[i], cs = c.shape;
      setChildPose(px, py, pz, qx, qy, qz, qw, c);
      _P[7] = 0;
      cs._aabb(cs, _cb);
      if (_cb[0] < minX) minX = _cb[0]; if (_cb[1] < minY) minY = _cb[1]; if (_cb[2] < minZ) minZ = _cb[2];
      if (_cb[3] > maxX) maxX = _cb[3]; if (_cb[4] > maxY) maxY = _cb[4]; if (_cb[5] > maxZ) maxZ = _cb[5];
    }
    out[0] = minX - g; out[1] = minY - g; out[2] = minZ - g; out[3] = maxX + g; out[4] = maxY + g; out[5] = maxZ + g;
    return out;
  },
  project(s, out) {
    const ch = s.children, px = _P[0], py = _P[1], pz = _P[2], qx = _P[3], qy = _P[4], qz = _P[5], qw = _P[6];
    const ax = _D[0], ay = _D[1], az = _D[2];
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < ch.length; i++) {
      const c = ch[i], cs = c.shape;
      setChildPose(px, py, pz, qx, qy, qz, qw, c);
      _D[0] = ax; _D[1] = ay; _D[2] = az;
      cs._project(cs, _cp);
      if (_cp[0] < lo) lo = _cp[0]; if (_cp[1] > hi) hi = _cp[1];
    }
    out[0] = lo; out[1] = hi;
    return out;
  },
};

// Exact max distance from the origin of a shape placed at pose (p, q) (construction time).
function farthest(s, px, py, pz, qx, qy, qz, qw) {
  const v = new Float64Array(3);
  const dist = (x, y, z) => { transformPoint(px, py, pz, qx, qy, qz, qw, x, y, z, v); return Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); };
  const scan = (p) => { let r = 0; for (let i = 0; i < p.length; i += 3) r = Math.max(r, dist(p[i], p[i + 1], p[i + 2])); return r; };
  switch (s.type) {
    case SHAPE.SPHERE: return dist(0, 0, 0) + s.radius;
    case SHAPE.CAPSULE: return Math.max(dist(0, s.halfHeight, 0), dist(0, -s.halfHeight, 0)) + s.radius;
    case SHAPE.BOX: case SHAPE.CONVEX: return scan(s.points);
    case SHAPE.TRIMESH: return scan(s.positions);
    default: {
      let r = 0;
      for (const c of s.children) {
        const [cx, cy, cz] = c.pos, [bx, by, bz, bw] = c.quat;
        transformPoint(px, py, pz, qx, qy, qz, qw, cx, cy, cz, v);
        r = Math.max(r, farthest(c.shape, v[0], v[1], v[2],
          qw * bx + qx * bw + qy * bz - qz * by, qw * by - qx * bz + qy * bw + qz * bx,
          qw * bz + qx * by - qy * bx + qz * bw, qw * bw - qx * bx - qy * by - qz * bz));
      }
      return r;
    }
  }
}

/** Compound of child shapes placed in the body frame (see the header). */
export function compound(children) {
  if (!Array.isArray(children) || children.length === 0) throw new RangeError('compound: needs a non-empty children array');
  const s = makeShape(SHAPE.COMPOUND, COMPOUND_CORES);
  const list = s.children = [];
  for (const ch of children) {
    if (!ch || !ch.shape || typeof ch.shape._local !== 'function') throw new TypeError('compound: each child needs a shape');
    const p = ch.position, q = ch.quaternion;
    const pos = new Float64Array(3), quat = new Float64Array(4);
    if (p) { pos[0] = p.x ?? p[0]; pos[1] = p.y ?? p[1]; pos[2] = p.z ?? p[2]; }
    if (q) { quat[0] = q.x ?? q[0]; quat[1] = q.y ?? q[1]; quat[2] = q.z ?? q[2]; quat[3] = q.w ?? q[3]; } else quat[3] = 1;
    const ql = Math.sqrt(quat[0] * quat[0] + quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3]);
    if (!(ql > 0) || !pos.every(Number.isFinite)) throw new RangeError('compound: child pose is not finite / quaternion is zero');
    for (let i = 0; i < 4; i++) quat[i] /= ql;
    list.push({ shape: ch.shape, pos, quat });
  }
  // volume-weighted centre of mass (equal weights if every child is flat)
  let vol = 0;
  for (const c of list) vol += c.shape.volume;
  const w = (c) => (vol > 0 ? c.shape.volume / vol : 1 / list.length);
  const wc = list.map((c) => {
    const v = transformPoint(c.pos[0], c.pos[1], c.pos[2], c.quat[0], c.quat[1], c.quat[2], c.quat[3], c.shape.centroid[0], c.shape.centroid[1], c.shape.centroid[2], new Float64Array(3));
    s.centroid[0] += w(c) * v[0]; s.centroid[1] += w(c) * v[1]; s.centroid[2] += w(c) * v[2];
    return v;
  });
  s.volume = vol;
  // unit-mass inertia about the origin: each child's tensor about its own centre of mass, rotated (R I R^T), then moved
  // to the origin (parallel axis)
  const u = s.unitInertia, R = new Float64Array(9);
  list.forEach((c, k) => {
    const m = w(c);
    c.shape.inertia(m, _I, true);
    for (let j = 0; j < 3; j++) {
      const e = [0, 0, 0];
      e[j] = 1;
      const col = rotateVector(c.quat[0], c.quat[1], c.quat[2], c.quat[3], e[0], e[1], e[2], new Float64Array(3));
      R[j] = col[0]; R[3 + j] = col[1]; R[6 + j] = col[2];
    }
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        let sum = 0;
        for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) sum += R[i * 3 + a] * _I[a * 3 + b] * R[j * 3 + b];
        u[i * 3 + j] += sum;
      }
    }
    addParallelAxis(u, m, wc[k][0], wc[k][1], wc[k][2]);
  });
  s.boundingRadius = farthest(s, 0, 0, 0, 0, 0, 0, 1);
  return s;
}
