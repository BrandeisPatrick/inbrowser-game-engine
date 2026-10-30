// Headless checks for the dynamics collision shapes (src/physics/dynamics/shapes.js): analytic and integrated mass
// properties, hull topology, support / aabb / project against brute force, the trimesh BVH against a linear scan, and
// that the per-step queries do not allocate. No browser, no WebGL.
// Run: node test/dyn-shapes.mjs  (imports the modules by path until the physics index exports them)
import v8 from 'node:v8';
import vm from 'node:vm';
import { Vector3, Quaternion } from 'three';
import { mulberry32 } from '../src/core/mathDet.js';
import { SHAPE } from '../src/physics/dynamics/state.js';
import {
  sphere, capsule, box, convex, trimesh, compound, SHAPE_NAMES,
  transformPoint, rotateVector, inverseRotateVector, inertiaOfBox, inertiaOfSphere, inertiaOfCapsule,
} from '../src/physics/dynamics/shapes.js';

let failures = 0;
const results = [];
function check(name, value, lo = 1, hi = lo, unit = '') {
  if (typeof value === 'boolean') value = value ? 1 : 0;
  const ok = Number.isFinite(value) && value >= lo && value <= hi;
  if (!ok) failures++;
  results.push(`${ok ? 'PASS' : 'FAIL'}  ${name.padEnd(60)} ${fmt(value)} ${unit}   [${fmt(lo).trim()}${hi !== lo ? ' .. ' + fmt(hi).trim() : ''}]`);
  return ok;
}
const fmt = (x) => (!Number.isFinite(x) ? String(x) : Number.isInteger(x) ? String(x) : Math.abs(x) < 1e-3 ? x.toExponential(2) : x.toFixed(3)).padStart(10);

// Bytes allocated and GCs run while fn() executes (same helper as test/core.mjs).
performance.now(); process.memoryUsage();
function measureAlloc(fn) {
  const prof = new v8.GCProfiler();
  prof.start();
  const h0 = process.memoryUsage().heapUsed;
  const t0 = performance.now();
  fn();
  const ms = performance.now() - t0;
  const h1 = process.memoryUsage().heapUsed;
  const gcs = prof.stop().statistics.length;
  return { bytes: h1 - h0, gcs, ms };
}

// ---- helpers --------------------------------------------------------------------------------------------------------
const rnd = mulberry32(0x5eed5);
const range = (a, b) => a + (b - a) * rnd();
function randUnit(out = new Float64Array(3)) {
  const z = 2 * rnd() - 1, phi = 2 * Math.PI * rnd(), r = Math.sqrt(1 - z * z);
  out[0] = r * Math.cos(phi); out[1] = r * Math.sin(phi); out[2] = z;
  return out;
}
function randQuat() { // Shoemake: uniform over SO(3)
  const u1 = rnd(), u2 = 2 * Math.PI * rnd(), u3 = 2 * Math.PI * rnd(), a = Math.sqrt(1 - u1), b = Math.sqrt(u1);
  return [a * Math.sin(u2), a * Math.cos(u2), b * Math.sin(u3), b * Math.cos(u3)];
}
const randPose = () => [range(-20, 20), range(-20, 20), range(-20, 20), ...randQuat()];
const maxDiff = (a, b) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };
const quatMul = (a, b) => [
  a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1], a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
  a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3], a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]];
// pose = [px, py, pz, qx, qy, qz, qw]; local point -> world
const xf = (pose, x, y, z) => Array.from(transformPoint(...pose, x, y, z, new Float64Array(3)));
const childPose = (pose, c) => [...xf(pose, c.pos[0], c.pos[1], c.pos[2]), ...quatMul(pose.slice(3), Array.from(c.quat))];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const diag = (a, b, c) => [a, 0, 0, 0, b, 0, 0, 0, c];

// Independent brute-force support (vertex scans / closed forms; never calls the shape's own support).
function bruteSupport(s, d, pose) {
  const n = Math.hypot(d[0], d[1], d[2]), u = [d[0] / n, d[1] / n, d[2] / n];
  const scan = (p) => {
    let best = -Infinity, bp = null;
    for (let i = 0; i < p.length; i += 3) { const w = xf(pose, p[i], p[i + 1], p[i + 2]), v = dot(w, d); if (v > best) { best = v; bp = w; } }
    return bp;
  };
  switch (s.type) {
    case SHAPE.SPHERE: return [pose[0] + s.radius * u[0], pose[1] + s.radius * u[1], pose[2] + s.radius * u[2]];
    case SHAPE.CAPSULE: {
      const a = xf(pose, 0, s.halfHeight, 0), b = xf(pose, 0, -s.halfHeight, 0), e = dot(a, d) >= dot(b, d) ? a : b;
      return [e[0] + s.radius * u[0], e[1] + s.radius * u[1], e[2] + s.radius * u[2]];
    }
    case SHAPE.BOX: {
      const h = s.half, c = [];
      for (let i = 0; i < 8; i++) c.push(i & 1 ? h[0] : -h[0], i & 2 ? h[1] : -h[1], i & 4 ? h[2] : -h[2]);
      return scan(c);
    }
    case SHAPE.CONVEX: return scan(s.points);
    case SHAPE.TRIMESH: return scan(s.positions);
    default: {
      let best = -Infinity, bp = null;
      for (const c of s.children) { const w = bruteSupport(c.shape, d, childPose(pose, c)), v = dot(w, d); if (v > best) { best = v; bp = w; } }
      return bp;
    }
  }
}

// A random point on the shape's surface (or inside it, for capsules) at a pose.
function samplePoint(s, pose) {
  const u = randUnit();
  switch (s.type) {
    case SHAPE.SPHERE: return xf(pose, s.radius * u[0], s.radius * u[1], s.radius * u[2]);
    case SHAPE.CAPSULE: { const t = range(-s.halfHeight, s.halfHeight); return xf(pose, s.radius * u[0], t + s.radius * u[1], s.radius * u[2]); }
    case SHAPE.BOX: {
      const h = s.half, p = [range(-h[0], h[0]), range(-h[1], h[1]), range(-h[2], h[2])], a = (rnd() * 3) | 0;
      p[a] = rnd() < 0.5 ? -h[a] : h[a];
      return xf(pose, p[0], p[1], p[2]);
    }
    case SHAPE.CONVEX: {
      const f = (rnd() * s.faceCount) | 0, a = s.faceStart[f], n = s.faceStart[f + 1] - a, k = 1 + ((rnd() * (n - 2)) | 0);
      return xf(pose, ...triPoint(s.points, s.faceVerts[a], s.faceVerts[a + k], s.faceVerts[a + k + 1]));
    }
    case SHAPE.TRIMESH: {
      const t = (rnd() * s.triCount) | 0, ix = s.indices;
      return xf(pose, ...triPoint(s.positions, ix[t * 3], ix[t * 3 + 1], ix[t * 3 + 2]));
    }
    default: { const c = s.children[(rnd() * s.children.length) | 0]; return samplePoint(c.shape, childPose(pose, c)); }
  }
}
function triPoint(p, i, j, k) {
  let a = rnd(), b = rnd();
  if (a + b > 1) { a = 1 - a; b = 1 - b; }
  const c = 1 - a - b, out = [];
  for (let e = 0; e < 3; e++) out.push(c * p[i * 3 + e] + a * p[j * 3 + e] + b * p[k * 3 + e]);
  return out;
}

// Solid of revolution about Y (radius R(y)), integrated with 5-point Gauss-Legendre per piece: exact for the
// piecewise-polynomial integrands of spheres and capsules. Returns { volume, ix, iy } for the given mass.
const GL = [[-0.9061798459386640, 0.2369268850561891], [-0.5384693101056831, 0.4786286704993665], [0, 0.5688888888888889],
  [0.5384693101056831, 0.4786286704993665], [0.9061798459386640, 0.2369268850561891]];
function revolve(mass, pieces) {
  let vol = 0, ix = 0, iy = 0;
  for (const [a, b, R2] of pieces) {
    for (const [x, w] of GL) {
      const y = (a + b) / 2 + (b - a) / 2 * x, r2 = R2(y), ww = w * (b - a) / 2;
      vol += ww * Math.PI * r2; ix += ww * Math.PI * r2 * (r2 / 4 + y * y); iy += ww * Math.PI * r2 * r2 / 2;
    }
  }
  return { volume: vol, ix: ix * mass / vol, iy: iy * mass / vol };
}
const capsulePieces = (r, hh) => [
  [-hh - r, -hh, (y) => r * r - (y + hh) * (y + hh)], [-hh, hh, () => r * r], [hh, hh + r, (y) => r * r - (y - hh) * (y - hh)]];

// ================================================================================================= analytic inertia
{
  const out = new Float64Array(9);
  const s = sphere(0.7), sr = revolve(3.2, [[-0.7, 0.7, (y) => 0.49 - y * y]]);
  check('sphere inertia vs 2/5 m r^2', maxDiff(s.inertia(3.2, out), diag(0.4 * 3.2 * 0.49, 0.4 * 3.2 * 0.49, 0.4 * 3.2 * 0.49)), 0, 1e-9);
  check('sphere inertia vs integrated solid of revolution', maxDiff(out, diag(sr.ix, sr.iy, sr.ix)), 0, 1e-9);
  check('sphere volume', Math.abs(s.volume - sr.volume), 0, 1e-9);
  const b = box(0.5, 0.25, 1), m = 5, W = 1, H = 0.5, D = 2;
  check('box inertia vs m (b^2 + c^2) / 12', maxDiff(b.inertia(m, out), diag(m * (H * H + D * D) / 12, m * (W * W + D * D) / 12, m * (W * W + H * H) / 12)), 0, 1e-9);
  check('box volume / boundingRadius', Math.abs(b.volume - 1) + Math.abs(b.boundingRadius - Math.hypot(0.5, 0.25, 1)), 0, 1e-12);
  for (const [r, hh] of [[0.3, 0.8], [0.5, 0.05], [1.2, 2.5]]) {
    const c = capsule(r, hh), ref = revolve(4, capsulePieces(r, hh));
    check(`capsule(${r}, ${hh}) inertia vs integrated (Gauss-Legendre)`, maxDiff(c.inertia(4, out), diag(ref.ix, ref.iy, ref.ix)), 0, 1e-9);
    check(`capsule(${r}, ${hh}) volume vs integrated`, Math.abs(c.volume - ref.volume), 0, 1e-9);
  }
  check('inertiaOf* helpers match the shapes',
    maxDiff(inertiaOfSphere(2, 0.7), sphere(0.7).inertia(2, out)) + maxDiff(inertiaOfBox(2, 1, 2, 3), box(1, 2, 3).inertia(2, out)) +
    maxDiff(inertiaOfCapsule(2, 0.3, 0.8), capsule(0.3, 0.8).inertia(2, out)), 0, 0);
  check('SHAPE_NAMES / type ids', SHAPE_NAMES[sphere(1).type] === 'sphere' && SHAPE_NAMES[capsule(1, 1).type] === 'capsule' &&
    SHAPE_NAMES[box(1, 1, 1).type] === 'box' && SHAPE_NAMES[SHAPE.COMPOUND] === 'compound');
}

// ================================================================================================= convex hull topology + mass
function topologyErrors(s) {
  const p = s.points, pl = s.planes;
  let outside = 0, offPlane = 0, cw = 0;
  for (let f = 0; f < s.faceCount; f++) {
    const n = [pl[f * 4], pl[f * 4 + 1], pl[f * 4 + 2]], d = pl[f * 4 + 3];
    for (let i = 0; i < p.length; i += 3) outside = Math.max(outside, n[0] * p[i] + n[1] * p[i + 1] + n[2] * p[i + 2] + d);
    const a = s.faceStart[f], b = s.faceStart[f + 1];
    for (let k = a; k < b; k++) { const v = s.faceVerts[k] * 3; offPlane = Math.max(offPlane, Math.abs(n[0] * p[v] + n[1] * p[v + 1] + n[2] * p[v + 2] + d)); }
    // CCW from outside: summed fan-triangle cross products point along the outward normal
    const v0 = s.faceVerts[a] * 3;
    let cx = 0, cy = 0, cz = 0;
    for (let k = a + 1; k + 1 < b; k++) {
      const v1 = s.faceVerts[k] * 3, v2 = s.faceVerts[k + 1] * 3;
      const e1 = [p[v1] - p[v0], p[v1 + 1] - p[v0 + 1], p[v1 + 2] - p[v0 + 2]], e2 = [p[v2] - p[v0], p[v2 + 1] - p[v0 + 1], p[v2 + 2] - p[v0 + 2]];
      cx += e1[1] * e2[2] - e1[2] * e2[1]; cy += e1[2] * e2[0] - e1[0] * e2[2]; cz += e1[0] * e2[1] - e1[1] * e2[0];
    }
    if (cx * n[0] + cy * n[1] + cz * n[2] <= 0) cw++;
  }
  // every undirected edge appears once and borders exactly two faces
  const use = new Map();
  for (let f = 0; f < s.faceCount; f++) {
    for (let k = s.faceStart[f]; k < s.faceStart[f + 1]; k++) {
      const a = s.faceVerts[k], b = s.faceVerts[k + 1 < s.faceStart[f + 1] ? k + 1 : s.faceStart[f]], key = Math.min(a, b) * 1e6 + Math.max(a, b);
      use.set(key, (use.get(key) || 0) + 1);
    }
  }
  let edgeBad = use.size !== s.edgeCount ? 1 : 0;
  for (const v of use.values()) if (v !== 2) edgeBad++;
  for (let e = 0; e < s.edgeCount; e++) if (use.get(s.edges[2 * e] * 1e6 + s.edges[2 * e + 1]) !== 2) edgeBad++;
  return { outside, offPlane, cw, edgeBad, euler: s.vertexCount - s.edgeCount + s.faceCount };
}
{
  const hx = 0.6, hy = 0.35, hz = 1.1, T = [1, 2, -3], out = new Float64Array(9);
  const corners = [];
  for (let i = 0; i < 8; i++) corners.push((i & 1 ? hx : -hx) + T[0], (i & 2 ? hy : -hy) + T[1], (i & 4 ? hz : -hz) + T[2]);
  const h = convex(corners);
  check('hull(box corners): 8 points', h.vertexCount, 8);
  check('hull(box corners): 6 faces (coplanar triangles merged)', h.faceCount, 6);
  check('hull(box corners): 12 edges', h.edgeCount, 12);
  check('hull(box corners): inertia vs box (1e-6)', maxDiff(h.inertia(7, out), inertiaOfBox(7, hx, hy, hz)), 0, 1e-6);
  check('hull(box corners): volume', Math.abs(h.volume - 8 * hx * hy * hz), 0, 1e-12);
  check('hull(box corners): offset = -translation, centroid 0', maxDiff(h.offset, [-T[0], -T[1], -T[2]]) + maxDiff(h.centroid, [0, 0, 0]), 0, 1e-12);
  check('hull(box corners): boundingRadius', Math.abs(h.boundingRadius - Math.hypot(hx, hy, hz)), 0, 1e-12);
  const te = topologyErrors(h);
  check('hull(box corners): points inside every plane', te.outside, -Infinity, 1e-12);
  check('hull(box corners): face vertices on their plane', te.offPlane, 0, 1e-12);
  check('hull(box corners): faces CCW from outside / manifold edges', te.cw + te.edgeBad, 0);
  check('hull(box corners): Euler V - E + F', te.euler, 2);
  const faceSizes = Array.from({ length: h.faceCount }, (_, f) => h.faceStart[f + 1] - h.faceStart[f]);
  check('hull(box corners): every face a quad', faceSizes.every((n) => n === 4));
  // interior and face-centre points are dropped; Vector3 input
  const extra = [];
  for (let i = 0; i < 24; i += 3) extra.push(new Vector3(corners[i], corners[i + 1], corners[i + 2]));
  extra.push(new Vector3(T[0], T[1], T[2]), new Vector3(T[0] + hx, T[1], T[2]), new Vector3(T[0] + 0.1, T[1] - 0.1, T[2] + 0.2));
  const h2 = convex(extra);
  check('hull(corners + interior + face centre, Vector3[]): 8/6/12', h2.vertexCount === 8 && h2.faceCount === 6 && h2.edgeCount === 12);
  // rotated box corners: full tensor R I R^T
  const q = new Quaternion(...randQuat()), rc = [], R = new Float64Array(9);
  for (let i = 0; i < 8; i++) { const v = new Vector3(i & 1 ? hx : -hx, i & 2 ? hy : -hy, i & 4 ? hz : -hz).applyQuaternion(q); rc.push(v.x, v.y, v.z); }
  for (let j = 0; j < 3; j++) { const e = [0, 0, 0]; e[j] = 1; const c = rotateVector(q.x, q.y, q.z, q.w, ...e, new Float64Array(3)); R[j] = c[0]; R[3 + j] = c[1]; R[6 + j] = c[2]; }
  const Ib = inertiaOfBox(7, hx, hy, hz), Irot = new Float64Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) { let s = 0; for (let a = 0; a < 3; a++) s += R[i * 3 + a] * Ib[a * 3 + a] * R[j * 3 + a]; Irot[i * 3 + j] = s; }
  check('hull(rotated box corners): inertia vs R I R^T (1e-6)', maxDiff(convex(rc).inertia(7, out), Irot), 0, 1e-6);
  // box() carries the same topology with corner i = sign bits of i
  const b = box(hx, hy, hz), tb = topologyErrors(b);
  check('box(): 8 points / 6 faces / 12 edges, Euler 2', b.vertexCount === 8 && b.faceCount === 6 && b.edgeCount === 12 && tb.euler === 2);
  check('box(): corner order + planes + CCW', maxDiff(b.points.subarray(21, 24), [hx, hy, hz]) + Math.max(0, tb.outside) + tb.offPlane + tb.cw + tb.edgeBad, 0, 1e-12);
}
{
  // 500 random points inside the unit sphere. The hull of 500 points uniform in the ball misses ~21 % of its volume
  // (checked below for information), so these are drawn in the outer shell r in [0.995, 1]: the hull then holds ~97 %
  const pts = new Float64Array(1500), u = new Float64Array(3);
  for (let i = 0; i < 500; i++) { randUnit(u); const r = range(0.995, 1); pts[i * 3] = r * u[0]; pts[i * 3 + 1] = r * u[1]; pts[i * 3 + 2] = r * u[2]; }
  const t0 = performance.now();
  const h = convex(pts);
  const ms = performance.now() - t0;
  const V = 4 / 3 * Math.PI;
  check('hull(500 points in unit sphere): volume within 5 %', h.volume / V, 0.95, 1.0);
  const te = topologyErrors(h);
  check('hull(500 points): Euler 2, manifold, CCW', te.euler === 2 && te.edgeBad === 0 && te.cw === 0);
  check('hull(500 points): points inside every plane', te.outside, -Infinity, 1e-12);
  check('hull(500 points): inertia ~ 2/5 m r^2 (sphere-like)', h.inertia(1, new Float64Array(9))[4] / 0.4, 0.9, 1.0);
  check('hull(500 points): build time', ms, 0, 500, 'ms');
  // uniform interior points: what the hull loses (reported, loose)
  const pu = new Float64Array(1500);
  for (let i = 0; i < 500; i++) { randUnit(u); const r = Math.cbrt(rnd()); pu[i * 3] = r * u[0]; pu[i * 3 + 1] = r * u[1]; pu[i * 3 + 2] = r * u[2]; }
  check('hull(500 uniform interior points): volume / sphere (info)', convex(pu).volume / V, 0.7, 1.0);
  // hull: false keeps the given points and their order
  const hp = Array.from(h.points), hf = convex(hp, { hull: false });
  check('hull: false on hull points: same volume / faces / order', Math.abs(hf.volume - h.volume) < 1e-12 && hf.faceCount === h.faceCount &&
    hf.vertexCount === h.vertexCount && maxDiff(hf.points, hp) < 1e-12);
  let threw = 0;
  try { convex([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0]); } catch { threw = 1; }
  check('coplanar points throw', threw, 1);
}

// ================================================================================================= canonical topology
// Topology must be a pure function of the faces, so convex(h.points, { hull: false }) (what a serializer rebuilds)
// reproduces faceStart / faceVerts / edges and the plane order exactly.
const eqArr = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);
function canonicalErrors(s) {
  let bad = 0;
  for (let f = 0; f < s.faceCount; f++) {
    const a = s.faceStart[f], b = s.faceStart[f + 1];
    for (let k = a + 1; k < b; k++) if (s.faceVerts[k] <= s.faceVerts[a]) bad++; // loop starts at its smallest vertex
    if (f > 0) { // faces sorted lexicographically
      const p = s.faceStart[f - 1], pl = a - p, cl = b - a;
      let cmp = 0;
      for (let k = 0; k < Math.min(pl, cl) && cmp === 0; k++) cmp = s.faceVerts[p + k] - s.faceVerts[a + k];
      if (cmp > 0 || (cmp === 0 && pl >= cl)) bad++;
    }
  }
  for (let e = 0; e < s.edgeCount; e++) {
    if (s.edges[2 * e] >= s.edges[2 * e + 1]) bad++;
    if (e > 0 && (s.edges[2 * e - 2] > s.edges[2 * e] || (s.edges[2 * e - 2] === s.edges[2 * e] && s.edges[2 * e - 1] >= s.edges[2 * e + 1]))) bad++;
  }
  return bad;
}
{
  let mismatch = 0, planeErr = 0, notCanonical = 0, orderBad = 0, faces = 0;
  const clouds = [];
  for (let c = 0; c < 100; c++) {
    const n = 6 + ((rnd() * 200) | 0), sx = range(0.2, 5), sy = range(0.2, 5), sz = range(0.2, 5), o = [range(-50, 50), range(-50, 50), range(-50, 50)];
    const pts = new Float64Array(n * 3), u = new Float64Array(3);
    for (let i = 0; i < n; i++) {
      randUnit(u); const r = c % 2 ? Math.cbrt(rnd()) : range(0.9, 1);
      pts[i * 3] = o[0] + sx * r * u[0]; pts[i * 3 + 1] = o[1] + sy * r * u[1]; pts[i * 3 + 2] = o[2] + sz * r * u[2];
    }
    clouds.push(pts);
  }
  for (let c = 0; c < 20; c++) { // rotated, translated boxes (float64)
    const q = new Quaternion(...randQuat()), h = [range(0.1, 3), range(0.1, 3), range(0.1, 3)], o = [range(-20, 20), range(-20, 20), range(-20, 20)], pts = [];
    for (let i = 0; i < 8; i++) { const v = new Vector3(i & 1 ? h[0] : -h[0], i & 2 ? h[1] : -h[1], i & 4 ? h[2] : -h[2]).applyQuaternion(q); pts.push(v.x + o[0], v.y + o[1], v.z + o[2]); }
    clouds.push(pts);
  }
  for (const pts of clouds) {
    const h = convex(pts), r = convex(Array.from(h.points), { hull: false });
    faces += h.faceCount;
    if (!eqArr(h.faceStart, r.faceStart) || !eqArr(h.faceVerts, r.faceVerts) || !eqArr(h.edges, r.edges) || h.planes.length !== r.planes.length) mismatch++;
    else planeErr = Math.max(planeErr, maxDiff(h.planes, r.planes));
    notCanonical += canonicalErrors(h);
    // hull vertices keep the input order
    let last = -1;
    for (let i = 0; i < h.vertexCount; i++) {
      let k = -1;
      for (let j = 0; j < pts.length / 3 && k < 0; j++) if (Math.abs(pts[j * 3] + h.offset[0] - h.points[i * 3]) + Math.abs(pts[j * 3 + 1] + h.offset[1] - h.points[i * 3 + 1]) + Math.abs(pts[j * 3 + 2] + h.offset[2] - h.points[i * 3 + 2]) < 1e-9) k = j;
      if (k <= last) orderBad++;
      last = k;
    }
  }
  check('hull: false rebuild == original topology (100 clouds + 20 rotated boxes)', mismatch, 0);
  check('hull: false rebuild: planes in the same order and equal', planeErr, 0, 1e-12);
  check('topology canonical (loops start at min, faces / edges sorted)', notCanonical, 0);
  check('hull vertices in input order', orderBad, 0);
  check('canonical topology test covers faces', faces, 1000, Infinity);
  // float32 (glTF-quantised) rotated boxes, modelled around their own origin (pivot within the box), still merge into
  // 6 quads (the merge tolerance scales with the hull's size, not its distance from the origin)
  let f32bad = 0, f32out = 0;
  for (let c = 0; c < 20; c++) {
    const q = new Quaternion(...randQuat()), h = [range(0.1, 3), range(0.1, 3), range(0.1, 3)], o = [range(-h[0], h[0]), range(-h[1], h[1]), range(-h[2], h[2])];
    const pts = new Float32Array(24);
    for (let i = 0; i < 8; i++) { const v = new Vector3(i & 1 ? h[0] : -h[0], i & 2 ? h[1] : -h[1], i & 4 ? h[2] : -h[2]).applyQuaternion(q); pts[i * 3] = v.x + o[0]; pts[i * 3 + 1] = v.y + o[1]; pts[i * 3 + 2] = v.z + o[2]; }
    const hb = convex(pts);
    if (hb.faceCount !== 6 || hb.edgeCount !== 12 || hb.vertexCount !== 8) f32bad++;
    f32out = Math.max(f32out, topologyErrors(hb).outside);
  }
  check('float32 rotated boxes: 6 faces / 12 edges / 8 points (20 poses)', f32bad, 0);
  check('float32 rotated boxes: points within 1e-6 of every plane', f32out, -Infinity, 1e-6);
  // box() table == convex() of its corners (bit order) with hull: false; planes exactly +-e_i, d = -h_i
  const bh = [0.6, 0.35, 1.1], b = box(...bh), corners = [];
  for (let i = 0; i < 8; i++) corners.push(i & 1 ? bh[0] : -bh[0], i & 2 ? bh[1] : -bh[1], i & 4 ? bh[2] : -bh[2]);
  const bc = convex(corners, { hull: false });
  check('box() topology == convex(corners, { hull: false })', eqArr(b.faceStart, bc.faceStart) && eqArr(b.faceVerts, bc.faceVerts) && eqArr(b.edges, bc.edges) &&
    maxDiff(b.points, bc.points) === 0 && maxDiff(b.planes, bc.planes) < 1e-15);
  let planeExact = 0;
  for (let f = 0; f < 6; f++) {
    const n = [b.planes[f * 4], b.planes[f * 4 + 1], b.planes[f * 4 + 2]], axis = n.findIndex((v) => v !== 0);
    if (n.filter((v) => v !== 0).length !== 1 || Math.abs(n[axis]) !== 1 || b.planes[f * 4 + 3] !== -bh[axis]) planeExact++;
  }
  check('box() planes exactly +-e_i with d = -h_i', planeExact, 0);
  check('box() arrays are per shape (not shared)', box(1, 1, 1).faceVerts !== b.faceVerts && box(1, 1, 1).edges !== b.edges);
}

// ================================================================================================= compound
{
  const out = new Float64Array(9), M = 6;
  const a = box(0.5, 0.25, 0.4), b = box(0.3, 0.6, 0.2), pa = [1.2, 0.3, -0.5], pb = [-0.7, -0.4, 0.9];
  const c = compound([{ shape: a, position: pa }, { shape: b, position: new Vector3(...pb), quaternion: [0, 0, 0, 1] }]);
  const V = a.volume + b.volume, ma = M * a.volume / V, mb = M * b.volume / V;
  const ref = new Array(9).fill(0);
  for (const [m, h, d] of [[ma, [0.5, 0.25, 0.4], pa], [mb, [0.3, 0.6, 0.2], pb]]) {
    const ib = inertiaOfBox(m, ...h), d2 = dot(d, d);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) ref[i * 3 + j] += ib[i * 3 + j] + m * ((i === j ? d2 : 0) - d[i] * d[j]);
  }
  check('compound(2 offset boxes): inertia vs parallel-axis', maxDiff(c.inertia(M, out), ref), 0, 1e-9);
  const com = [0, 1, 2].map((i) => (ma * pa[i] + mb * pb[i]) / M);
  check('compound(2 offset boxes): centroid', maxDiff(c.centroid, com), 0, 1e-12);
  check('compound(2 offset boxes): volume', Math.abs(c.volume - V), 0, 1e-12);
  const refC = ref.slice(), cc = dot(com, com);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) refC[i * 3 + j] -= M * ((i === j ? cc : 0) - com[i] * com[j]);
  check('compound: inertia aboutCentroid', maxDiff(c.inertia(M, out, true), refC), 0, 1e-9);
  // a child rotated 90 deg about Y equals a box with swapped x / z extents
  const s2 = Math.SQRT1_2;
  const r1 = compound([{ shape: box(0.5, 0.25, 0.4), position: [0.3, 1, 0], quaternion: new Quaternion(0, s2, 0, s2) }]);
  const r2 = compound([{ shape: box(0.4, 0.25, 0.5), position: [0.3, 1, 0] }]);
  check('compound: rotated child == swapped box', maxDiff(r1.inertia(3, out), r2.inertia(3, new Float64Array(9))), 0, 1e-9);
  // nested compound == flat compound
  const n1 = compound([{ shape: compound([{ shape: a, position: pa }]), position: [0, 0, 0] }, { shape: b, position: pb }]);
  check('compound: nested == flat (inertia, centroid)', maxDiff(n1.inertia(M, out), ref) + maxDiff(n1.centroid, com), 0, 1e-9);
  check('compound: boundingRadius exact (box corners)', Math.abs(c.boundingRadius - Math.max(
    Math.hypot(1.7, 0.55, 0.9), Math.hypot(1.0, 1.0, 1.1))), 0, 1e-12);
}

// ================================================================================================= support / aabb / project
const hullPts = [];
for (let i = 0; i < 60; i++) { const u = randUnit(); const r = range(0.5, 1.5); hullPts.push(u[0] * r * 1.4, u[1] * r, u[2] * r * 0.7 + 0.4); }
function randomMesh(nt, spread, size) {
  const pos = new Float32Array(nt * 9), idx = new Uint32Array(nt * 3);
  for (let t = 0; t < nt; t++) {
    const cx = range(0, spread), cy = range(0, spread * 0.3), cz = range(0, spread);
    for (let k = 0; k < 3; k++) { pos[t * 9 + k * 3] = cx + range(-size, size); pos[t * 9 + k * 3 + 1] = cy + range(-size, size); pos[t * 9 + k * 3 + 2] = cz + range(-size, size); idx[t * 3 + k] = t * 3 + k; }
  }
  return { pos, idx };
}
const smallMesh = randomMesh(40, 6, 1);
const SHAPES = {
  sphere: sphere(0.8),
  capsule: capsule(0.35, 0.9),
  box: box(0.7, 0.3, 1.2),
  convex: convex(hullPts),
  trimesh: trimesh(smallMesh.pos, smallMesh.idx),
  compound: compound([
    { shape: box(0.5, 0.2, 0.3), position: [1, 0, 0], quaternion: randQuat() },
    { shape: sphere(0.4), position: [-1, 0.5, 0] },
    { shape: capsule(0.2, 0.5), position: [0, -0.8, 0.3], quaternion: randQuat() },
    { shape: convex(hullPts), position: [0, 1, -1], quaternion: randQuat() },
  ]),
};
{
  const out3 = new Float64Array(3), out6 = new Float64Array(6), out2 = new Float64Array(2), loc = new Float64Array(3);
  for (const [name, s] of Object.entries(SHAPES)) {
    let supErr = 0, locErr = 0, projErr = 0, outside = 0, radErr = 0;
    for (let i = 0; i < 200; i++) {
      const pose = randPose(), d = randUnit().map((v) => v * range(0.1, 10));
      s.support(d[0], d[1], d[2], ...pose, out3);
      supErr = Math.max(supErr, maxDiff(out3, bruteSupport(s, d, pose)));
      // localSupport == support at the identity pose
      s.localSupport(d[0], d[1], d[2], loc);
      locErr = Math.max(locErr, maxDiff(loc, s.support(d[0], d[1], d[2], 0, 0, 0, 0, 0, 0, 1, out3)));
      // project == support-based [min, max]
      s.project(d[0], d[1], d[2], ...pose, out2);
      const hi = dot(s.support(d[0], d[1], d[2], ...pose, out3), d), lo = dot(s.support(-d[0], -d[1], -d[2], ...pose, out3), d);
      projErr = Math.max(projErr, Math.abs(out2[0] - lo), Math.abs(out2[1] - hi));
    }
    for (let i = 0; i < 20; i++) {
      const pose = randPose();
      s.aabb(...pose, out6, 0);
      for (let k = 0; k < 50; k++) {
        const p = samplePoint(s, pose);
        outside = Math.max(outside, out6[0] - p[0], out6[1] - p[1], out6[2] - p[2], p[0] - out6[3], p[1] - out6[4], p[2] - out6[5]);
        radErr = Math.max(radErr, Math.hypot(p[0] - pose[0], p[1] - pose[1], p[2] - pose[2]) - s.boundingRadius);
      }
    }
    check(`${name}: support vs brute force (200 dirs x poses)`, supErr, 0, 1e-9);
    check(`${name}: localSupport == support at identity`, locErr, 0, 1e-12);
    check(`${name}: project vs support-based min/max`, projErr, 0, 1e-9);
    check(`${name}: aabb contains 1000 sampled points`, outside, -Infinity, 1e-9);
    check(`${name}: points within boundingRadius`, radErr, -Infinity, 1e-9);
  }
  // tightness: exact AABBs equal the extremes of the vertices / analytic bounds
  let boxTight = 0, hullTight = 0, capTight = 0, compTight = 0, margin = 0;
  const bx = SHAPES.box, hx = SHAPES.convex, cp = SHAPES.capsule, cm = SHAPES.compound;
  for (let i = 0; i < 200; i++) {
    const pose = randPose();
    const ext = (pts) => { const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity]; for (const p of pts) for (let a = 0; a < 3; a++) { lo[a] = Math.min(lo[a], p[a]); hi[a] = Math.max(hi[a], p[a]); } return [...lo, ...hi]; };
    const corners = [];
    for (let c = 0; c < 8; c++) corners.push(xf(pose, c & 1 ? 0.7 : -0.7, c & 2 ? 0.3 : -0.3, c & 4 ? 1.2 : -1.2));
    boxTight = Math.max(boxTight, maxDiff(bx.aabb(...pose, out6), ext(corners)));
    const hp = [];
    for (let k = 0; k < hx.points.length; k += 3) hp.push(xf(pose, hx.points[k], hx.points[k + 1], hx.points[k + 2]));
    hullTight = Math.max(hullTight, maxDiff(hx.aabb(...pose, out6), ext(hp)));
    const e0 = xf(pose, 0, 0.9, 0), e1 = xf(pose, 0, -0.9, 0), r = 0.35;
    capTight = Math.max(capTight, maxDiff(cp.aabb(...pose, out6), [...[0, 1, 2].map((a) => Math.min(e0[a], e1[a]) - r), ...[0, 1, 2].map((a) => Math.max(e0[a], e1[a]) + r)]));
    // compound: union of the children's AABBs; each axis bound is attained by the support in that axis direction
    cm.aabb(...pose, out6);
    const sup = [];
    for (let a = 0; a < 6; a++) { const d = [0, 0, 0]; d[a % 3] = a < 3 ? -1 : 1; sup.push(bruteSupport(cm, d, pose)[a % 3]); }
    compTight = Math.max(compTight, maxDiff(out6, sup));
    for (const s of Object.values(SHAPES)) {
      const m0 = Array.from(s.aabb(...pose, out6, 0)), m1 = Array.from(s.aabb(...pose, out6, 0.25));
      margin = Math.max(margin, maxDiff(m1, [m0[0] - 0.25, m0[1] - 0.25, m0[2] - 0.25, m0[3] + 0.25, m0[4] + 0.25, m0[5] + 0.25]));
    }
  }
  check('box aabb tight (rotated extents == corner min/max)', boxTight, 0, 1e-9);
  check('convex aabb tight (== transformed point min/max)', hullTight, 0, 1e-9);
  check('capsule aabb tight (== end-sphere bounds)', capTight, 0, 1e-9);
  check('compound aabb tight (== axis supports)', compTight, 0, 1e-9);
  check('aabb margin inflates every side (all 6 types, compound too)', margin, 0, 1e-12);
  // pose helpers
  const q = randQuat(), v = [0.3, -1.2, 2.5], w = rotateVector(...q, ...v, new Float64Array(3)), back = inverseRotateVector(...q, ...w, new Float64Array(3));
  const tq = new Vector3(...v).applyQuaternion(new Quaternion(...q));
  check('rotateVector vs three / inverseRotateVector round trip', maxDiff(w, [tq.x, tq.y, tq.z]) + maxDiff(back, v), 0, 1e-12);
  const zs = SHAPES.sphere.support(0, 0, 0, 1, 2, 3, 0, 0, 0, 1, out3), zb = SHAPES.box.support(0, 0, 0, 0, 0, 0, 0, 0, 0, 1, new Float64Array(3));
  check('zero direction: finite point of the shape', maxDiff(zs, [1, 2, 3]) + maxDiff(zb, [0.7, 0.3, 1.2]), 0, 0);
}

// ================================================================================================= trimesh BVH
{
  const { pos, idx } = randomMesh(2000, 100, 3);
  const t0 = performance.now();
  const tm = trimesh(pos, idx);
  const buildMs = performance.now() - t0;
  const b = tm.bvh;
  let leafBad = 0, boundsBad = 0, leaves = 0, leafTris = 0;
  for (let n = 0; n < b.nodeCount; n++) {
    if (b.left[n] < 0) { leaves++; leafTris += b.triCount[n]; if (b.triCount[n] < 1 || b.triCount[n] > 4 || b.right[n] >= 0) leafBad++; continue; }
    if (b.triCount[n] !== 0 || b.left[n] !== n + 1) leafBad++;
    for (const c of [b.left[n], b.right[n]]) for (let a = 0; a < 3; a++) if (b.nodeMin[c * 3 + a] < b.nodeMin[n * 3 + a] || b.nodeMax[c * 3 + a] > b.nodeMax[n * 3 + a]) boundsBad++;
  }
  const perm = new Uint8Array(2000);
  for (const t of b.triOrder) perm[t]++;
  check('bvh: leaves hold 1..4 triangles, internal nodes none', leafBad, 0);
  check('bvh: children inside parent bounds', boundsBad, 0);
  check('bvh: triOrder is a permutation covered by leaves', perm.every((c) => c === 1) && leafTris === 2000);
  const p64 = tm.positions, ix = tm.indices, out = new Int32Array(2000);
  let mismatch = 0, total = 0, tQuery = 0, tBrute = 0;
  for (let i = 0; i < 200; i++) {
    const c = [range(-10, 110), range(-10, 40), range(-10, 110)], h = [range(0.5, 20), range(0.5, 20), range(0.5, 20)];
    const lo = [c[0] - h[0], c[1] - h[1], c[2] - h[2]], hi = [c[0] + h[0], c[1] + h[1], c[2] + h[2]];
    let t = performance.now();
    const n = tm.queryAabb(lo[0], lo[1], lo[2], hi[0], hi[1], hi[2], out);
    tQuery += performance.now() - t;
    t = performance.now();
    const brute = [];
    for (let tri = 0; tri < 2000; tri++) {
      let hit = true;
      for (let a = 0; a < 3 && hit; a++) {
        const v0 = p64[ix[tri * 3] * 3 + a], v1 = p64[ix[tri * 3 + 1] * 3 + a], v2 = p64[ix[tri * 3 + 2] * 3 + a];
        if (Math.min(v0, v1, v2) > hi[a] || Math.max(v0, v1, v2) < lo[a]) hit = false;
      }
      if (hit) brute.push(tri);
    }
    tBrute += performance.now() - t;
    const got = Array.from(out.subarray(0, n)).sort((x, y) => x - y);
    if (got.length !== brute.length || got.some((v, k) => v !== brute[k])) mismatch++;
    total += n;
  }
  check('bvh queryAabb == brute force (200 boxes, 2000 tris)', mismatch, 0);
  check('bvh queries found triangles (non-trivial test)', total, 200, Infinity);
  check('bvh build (2000 tris)', buildMs, 0, 200, 'ms');
  results.push(`info  bvh: ${b.nodeCount} nodes, ${leaves} leaves; 200 queries ${tQuery.toFixed(2)} ms (brute force ${tBrute.toFixed(2)} ms), ${(total / 200).toFixed(1)} tris/query`);
  const small = new Int32Array(3);
  check('bvh queryAabb truncates at out.length', tm.queryAabb(-1e9, -1e9, -1e9, 1e9, 1e9, 1e9, small), 3);
  const two = trimesh([0, 0, 0, 1, 0, 0, 0, 0, 1, 2, 0, 0, 3, 0, 0, 2, 0, 1], new Uint16Array([0, 1, 2, 3, 4, 5]));
  check('trimesh: Uint16 indices, queries pick the right triangle', two.triCount === 2 && two.queryAabb(1.5, -0.1, 0.2, 2.5, 0.1, 0.4, out) === 1 &&
    out[0] === 1 && two.queryAabb(-1, -1, -1, 0.5, 1, 0.5, out) === 1 && out[0] === 0 && two.queryAabb(5, 5, 5, 6, 6, 6, out) === 0);
  const t16 = trimesh([0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1], new Uint16Array([0, 1, 2, 1, 3, 2]));
  check('trimesh: inertia = AABB box moved to the origin', maxDiff(t16.inertia(1, new Float64Array(9)), [
    (0 + 0.25) / 3 + 0.25, -0, -0.25, -0, (0.25 + 0.25) / 3 + 0.5, 0, -0.25, 0, 0.25 / 3 + 0.25]), 0, 1e-12);
}

// ================================================================================================= allocation + timing
// V8 boxes double arguments into heap numbers at call sites it does not inline; shapes.js therefore gives every shape
// the same small public methods (scalars -> module scratch -> per-type core taking no doubles). Each shape is measured
// with its own runner (new Function: its own feedback), then all six shapes go through ONE shared call site, which
// must not allocate either. Runners are warmed up with many short calls so the measured call enters optimized code (an
// interpreted loop boxes every double until on-stack replacement kicks in), a full GC runs before each measurement so
// leftover garbage cannot trigger a scavenge inside it, and the best of up to 3 repetitions counts (a concurrent JIT
// job finishing late can leave one repetition partly interpreted; per-call allocation would show in all of them).
{
  v8.setFlagsFromString('--expose-gc');
  const gc = vm.runInNewContext('gc');
  const N = 1e5, dirs = new Float64Array(N * 3), poses = new Float64Array(N * 7), u = new Float64Array(3);
  for (let i = 0; i < N; i++) {
    randUnit(u); dirs.set(u, i * 3);
    const p = randPose();
    for (let k = 0; k < 7; k++) poses[i * 7 + k] = p[k];
  }
  const out3 = new Float64Array(3), out6 = new Float64Array(6), out2 = new Float64Array(2);
  const CALL = {
    support: 's.support(dirs[d], dirs[d + 1], dirs[d + 2], poses[p], poses[p + 1], poses[p + 2], poses[p + 3], poses[p + 4], poses[p + 5], poses[p + 6], out3); acc += out3[0];',
    aabb: 's.aabb(poses[p], poses[p + 1], poses[p + 2], poses[p + 3], poses[p + 4], poses[p + 5], poses[p + 6], out6, 0.01); acc += out6[0];',
    project: 's.project(dirs[d], dirs[d + 1], dirs[d + 2], poses[p], poses[p + 1], poses[p + 2], poses[p + 3], poses[p + 4], poses[p + 5], poses[p + 6], out2); acc += out2[0];',
  };
  const makeRunner = (...calls) => new Function('list', 'n', 'dirs', 'poses', 'out3', 'out6', 'out2',
    `let acc = 0; for (let j = 0; j < list.length; j++) { const s = list[j]; for (let i = 0; i < n; i++) { const d = i * 3, p = i * 7; ${calls.map((c) => CALL[c]).join(' ')} } } return acc;`);
  const measure = (run, list) => {
    for (let k = 0; k < 400; k++) run(list, 200, dirs, poses, out3, out6, out2); // many short calls: the function entry gets optimized
    let best = null;
    for (let rep = 0; rep < 3 && !(best && best.gcs === 0); rep++) {
      gc();
      const m = measureAlloc(() => run(list, N, dirs, poses, out3, out6, out2));
      if (!best || m.gcs < best.gcs) best = m;
    }
    return best;
  };
  const timing = [];
  for (const [name, s] of Object.entries(SHAPES)) {
    const m = measure(makeRunner('support', 'aabb'), [s]), mp = measure(makeRunner('project'), [s]);
    timing.push(`${name} ${(m.ms * 1e6 / N).toFixed(0)} + ${(mp.ms * 1e6 / N).toFixed(0)} ns`);
    check(`${name}: 1e5 x (support + aabb): heap growth`, m.bytes / 1024, -Infinity, 1024, 'KB');
    check(`${name}: 1e5 x (support + aabb), 1e5 x project: GCs`, m.gcs + mp.gcs, 0);
  }
  const mm = measure(makeRunner('support', 'aabb', 'project'), Object.values(SHAPES));
  check('one call site, 6 shapes x 1e5 x 3 queries: heap growth', mm.bytes / 1024, -Infinity, 1024, 'KB');
  check('one call site, 6 shapes x 1e5 x 3 queries: GCs', mm.gcs, 0);
  results.push(`info  ns per call, support + aabb | project (one shape per call site): ${timing.join(', ')}`);
  results.push(`info  (convex: ${SHAPES.convex.vertexCount} verts, trimesh: ${SHAPES.trimesh.vertexCount} verts, compound: 4 children); shared call site: ${(mm.ms * 1e6 / (6 * N)).toFixed(0)} ns per support + aabb + project`);
}

console.log(results.join('\n'));
console.log(failures ? `\n${failures} FAILED` : '\nall dynamics shape checks passed');
process.exit(failures ? 1 : 0);
