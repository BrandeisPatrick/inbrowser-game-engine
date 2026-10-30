// Headless checks for the dynamics queries, the PhysicsWorld bridge and the debug lines (no browser, no WebGL).
// Run: node test/dyn-bridge.mjs (npm test runs it too).
// Part 1 (always): exact ray / shape-cast / overlap / containment math against hand-built shape objects carrying the
// shapes.js contract fields, debug line counts per shape type, the bridge provider on a BodyStore stub with a real
// PhysicsWorld + VehicleController (determinism gate, car stopped by a crate), PhysicsDebug.
// Part 2 (when shapes.js / DynamicsWorld.js import): the same against the real shapes and DynamicsWorld.
import './helpers/dom-shim.mjs';
import { Vector3, Object3D, Quaternion, Matrix4 } from 'three';
import { VehicleController, ROADSTER_SPEC, PhysicsWorld, LAYER } from '@vibe/engine/physics';
import { BodyStore, ContactBuffer, BODY, FLAG, SHAPE, DYN_RECORD } from '../src/physics/dynamics/state.js';
import { raycastBodies, shapeCastBodies, overlapBodies, pointInBody } from '../src/physics/dynamics/queries.js';
import { writeDebugLines, DEBUG_LINES } from '../src/physics/dynamics/debug.js';
import { attachToPhysicsWorld, detachFromPhysicsWorld, syncKinematicFromObject } from '../src/physics/dynamics/bridge.js';
import { PhysicsDebug } from '../src/render/PhysicsDebug.js';

const DT = 1 / 120;
let failures = 0;
const results = [];
function check(name, value, lo, hi, unit = '') {
  const ok = Number.isFinite(value) && value >= lo && value <= hi;
  if (!ok) failures++;
  results.push(`${ok ? 'PASS' : 'FAIL'}  ${name.padEnd(64)} ${fmt(value)} ${unit}   [${lo} .. ${hi}]`);
  return ok;
}
const fmt = (x) => (Number.isFinite(x) ? (Math.abs(x) < 1e-3 && x !== 0 ? x.toExponential(2) : x.toFixed(4)).padStart(10) : String(x).padStart(10));
const near = (name, v, want, tol = 1e-9, unit = '') => check(name, v, want - tol, want + tol, unit);
const yes = (name, b) => check(name, b ? 1 : 0, 1, 1);
const ctl = (o = {}) => ({ throttle: 0, brake: 0, steer: 0, thrust: 0, pitch: 0, roll: 0, yaw: 0, boost: false, handbrake: false, ...o });
const S = Math.SQRT1_2;
const Q_ID = [0, 0, 0, 1];
const qAxis = (ax, ay, az, a) => { const s = Math.sin(a / 2); return [ax * s, ay * s, az * s, Math.cos(a / 2)]; };

// ---- hand-built shapes (shapes.js contract fields only) -----------------------------------------------------------
const sphereShape = (r) => ({ type: SHAPE.SPHERE, radius: r });
const capsuleShape = (r, h) => ({ type: SHAPE.CAPSULE, radius: r, halfHeight: h });
const boxShape = (x, y, z) => ({ type: SHAPE.BOX, half: Float64Array.of(x, y, z) });
function hullBoxShape(hx, hy, hz) { // convex hull of a box: corners (bit order as shapes.js), 6 planes, 12 edges
  const points = new Float64Array(24);
  for (let i = 0; i < 8; i++) { points[i * 3] = i & 1 ? hx : -hx; points[i * 3 + 1] = i & 2 ? hy : -hy; points[i * 3 + 2] = i & 4 ? hz : -hz; }
  const planes = Float64Array.of(1, 0, 0, -hx, -1, 0, 0, -hx, 0, 1, 0, -hy, 0, -1, 0, -hy, 0, 0, 1, -hz, 0, 0, -1, -hz);
  const edges = Int32Array.of(0, 1, 2, 3, 4, 5, 6, 7, 0, 2, 1, 3, 4, 6, 5, 7, 0, 4, 1, 5, 2, 6, 3, 7);
  return { type: SHAPE.CONVEX, points, planes, edges };
}
// n x n quads on [-s, s]^2 at y = 0, counter-clockwise seen from +Y (front faces up)
function gridShape(n = 4, s = 2, withBvh = false) {
  const positions = [], indices = [];
  for (let k = 0; k <= n; k++) for (let i = 0; i <= n; i++) positions.push(-s + (2 * s * i) / n, 0, -s + (2 * s * k) / n);
  for (let k = 0; k < n; k++) for (let i = 0; i < n; i++) {
    const a = k * (n + 1) + i, b = a + 1, c = a + n + 1, d = c + 1;
    indices.push(a, c, b, b, c, d);
  }
  const shape = { type: SHAPE.TRIMESH, positions: Float64Array.from(positions), indices: Uint32Array.from(indices), bvh: null };
  if (withBvh) shape.bvh = stubBvh(shape.positions, shape.indices);
  return shape;
}
// two-level BVH stub in the shapes.js layout: root over all triangles, two leaves split at x = 0
function stubBvh(P, I) {
  const nt = I.length / 3, left = [], right = [];
  for (let t = 0; t < nt; t++) { let cx = 0; for (let v = 0; v < 3; v++) cx += P[I[t * 3 + v] * 3]; (cx < 0 ? left : right).push(t); }
  const nodeMin = new Float64Array(9).fill(Infinity), nodeMax = new Float64Array(9).fill(-Infinity);
  const grow = (nd, tris) => { for (const t of tris) for (let v = 0; v < 3; v++) for (let a = 0; a < 3; a++) { const x = P[I[t * 3 + v] * 3 + a]; nodeMin[nd * 3 + a] = Math.min(nodeMin[nd * 3 + a], x); nodeMax[nd * 3 + a] = Math.max(nodeMax[nd * 3 + a], x); } };
  grow(0, [...left, ...right]); grow(1, left); grow(2, right);
  return {
    nodeCount: 3, nodeMin, nodeMax, left: Int32Array.of(1, -1, -1), right: Int32Array.of(2, -1, -1),
    triStart: Int32Array.of(0, 0, left.length), triCount: Int32Array.of(0, left.length, right.length), triOrder: Uint32Array.from([...left, ...right]),
  };
}
const compoundShape = (children) => ({ type: SHAPE.COMPOUND, children: children.map((c) => ({ shape: c.shape, pos: Float64Array.from(c.pos || [0, 0, 0]), quat: Float64Array.from(c.quat || Q_ID) })) });

// ---- hand-built world --------------------------------------------------------------------------------------------
function stubWorld(cap = 64) { return { bodies: new BodyStore(cap), contacts: new ContactBuffer(64), joints: null, broadphase: null }; }
function add(w, shape, x, y, z, q = Q_ID, type = BODY.DYNAMIC, r = 3) {
  const B = w.bodies, id = B.alloc();
  B.type[id] = type; B.shape[id] = shape;
  B.pos[id * 3] = x; B.pos[id * 3 + 1] = y; B.pos[id * 3 + 2] = z;
  for (let k = 0; k < 4; k++) B.quat[id * 4 + k] = q[k];
  setAabb(B, id, x - r, y - r, z - r, x + r, y + r, z + r);
  return id;
}
function setAabb(B, id, ...v) { for (let k = 0; k < 6; k++) B.aabb[id * 6 + k] = v[k]; }
const hit = () => ({ distance: 0, point: new Vector3(), normal: new Vector3(), body: -1, box: null, kind: null, ground: false });
const H = hit();
const ray = (w, o, d, max = 100, opts) => (raycastBodies(w, o[0], o[1], o[2], d[0], d[1], d[2], max, opts, H) ? H : null);
const hd = (h) => (h ? h.distance : NaN);
const nErr = (h, n) => (h ? Math.hypot(h.normal.x - n[0], h.normal.y - n[1], h.normal.z - n[2]) : NaN);
const pErr = (h, p) => (h ? Math.hypot(h.point.x - p[0], h.point.y - p[1], h.point.z - p[2]) : NaN);

// --- 1. ray vs sphere ------------------------------------------------------------------------------------------------
{
  const w = stubWorld();
  const id = add(w, sphereShape(0.5), 1, 2, 3);
  const h = ray(w, [1, 2, -10], [0, 0, 5]); // non-unit direction
  near('ray-sphere: head-on distance 12.5', hd(h), 12.5);
  near('ray-sphere: normal (0,0,-1)', nErr(h, [0, 0, -1]), 0);
  near('ray-sphere: point (1,2,2.5)', pErr(h, [1, 2, 2.5]), 0);
  check('ray-sphere: body id, kind rigid, box null', h && h.body === id && h.kind === 'rigid' && h.box === null && h.ground === false ? 1 : 0, 1, 1);
  const h2 = ray(w, [1.3, 2, -10], [0, 0, 1]);
  near('ray-sphere: offset 0.3 -> 13 - 0.4', hd(h2), 12.6);
  near('ray-sphere: offset normal (0.6,0,-0.8)', nErr(h2, [0.6, 0, -0.8]), 0);
  yes('ray-sphere: miss beside', !ray(w, [1.51, 2, -10], [0, 0, 1]));
  yes('ray-sphere: maxDist short of it', !ray(w, [1, 2, -10], [0, 0, 1], 12.4));
  const h3 = ray(w, [1, 2.1, 3], [0, 1, 0]);
  near('ray-sphere: starting inside hits at 0, normal -dir', hd(h3) + nErr(h3, [0, -1, 0]), 0);
}

// --- 2. ray vs box (body frame slab) ---------------------------------------------------------------------------------
{
  const w = stubWorld();
  add(w, boxShape(1, 0.5, 2), 5, 0, 0, [0, S, 0, S]); // 90 deg about Y: world x half-extent 2
  const h = ray(w, [0, 0, 0], [1, 0, 0]);
  near('ray-box: rotated 90 deg, face at x = 3', hd(h), 3);
  near('ray-box: normal (-1,0,0)', nErr(h, [-1, 0, 0]), 0);
  const w2 = stubWorld();
  add(w2, boxShape(1, 1, 1), 5, 0, 0, qAxis(0, 1, 0, Math.PI / 4));
  near('ray-box: 45 deg about Y, vertical edge at 5 - sqrt2', hd(ray(w2, [0, 0.3, 0], [1, 0, 0])), 5 - Math.SQRT2);
  const h3 = ray(w, [5, 3, 0.4], [0, -2, 0]);
  near('ray-box: from above onto top face y = 0.5', hd(h3) + nErr(h3, [0, 1, 0]), 2.5);
}

// --- 3. ray vs capsule ------------------------------------------------------------------------------------------------
{
  const w = stubWorld();
  add(w, capsuleShape(0.5, 1), 0, 0, 0, [0, 0, S, S]); // axis along world X, caps at x = +-1
  const h1 = ray(w, [0.7, 5, 0], [0, -1, 0]);
  near('ray-capsule: lateral surface from above', hd(h1), 4.5);
  near('ray-capsule: lateral normal (0,1,0)', nErr(h1, [0, 1, 0]), 0);
  const h2 = ray(w, [5, 0, 0], [-1, 0, 0]);
  near('ray-capsule: cap tip at x = 1.5', hd(h2), 3.5);
  const h3 = ray(w, [5, 0.3, 0], [-1, 0, 0]);
  near('ray-capsule: cap off-axis 0.3 -> x = 1.4', hd(h3), 3.6);
  near('ray-capsule: cap normal (0.8,0.6,0)', nErr(h3, [0.8, 0.6, 0]), 0);
  yes('ray-capsule: miss above', !ray(w, [5, 0.51, 0], [-1, 0, 0]));
}

// --- 4. ray vs convex (box hull) == ray vs box --------------------------------------------------------------------------
{
  const w = stubWorld();
  add(w, hullBoxShape(1, 1, 1), 0, 0, 0);
  const h = ray(w, [5, 0.2, 0.3], [-1, 0, 0]);
  near('ray-convex: face x = 1 at 4', hd(h), 4);
  near('ray-convex: normal (1,0,0)', nErr(h, [1, 0, 0]), 0);
  // random rays: hull vs box shape at the same rotated pose
  const q = qAxis(0.36, 0.48, 0.8, 0.9);
  const wh = stubWorld(), wb = stubWorld();
  add(wh, hullBoxShape(1, 0.6, 1.4), 0.5, 1, -0.5, q); add(wb, boxShape(1, 0.6, 1.4), 0.5, 1, -0.5, q);
  let seed = 7, maxDt = 0, maxDn = 0, hits = 0, mismatch = 0;
  const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
  const Hb = hit();
  for (let i = 0; i < 400; i++) {
    const o = [rnd() * 10 - 5, rnd() * 10 - 5, rnd() * 10 - 5], tgt = [rnd() * 2 - 0.5, rnd() * 2, rnd() * 2 - 1.5];
    const d = [tgt[0] - o[0], tgt[1] - o[1], tgt[2] - o[2]];
    const a = raycastBodies(wh, o[0], o[1], o[2], d[0], d[1], d[2], 100, null, H);
    const b = raycastBodies(wb, o[0], o[1], o[2], d[0], d[1], d[2], 100, null, Hb);
    if (a !== b) { mismatch++; continue; }
    if (!a) continue;
    hits++;
    maxDt = Math.max(maxDt, Math.abs(H.distance - Hb.distance));
    maxDn = Math.max(maxDn, H.normal.distanceTo(Hb.normal));
  }
  check('ray-convex vs ray-box: 400 random rays, hit/miss agree', mismatch, 0, 0);
  check('ray-convex vs ray-box: hits', hits, 100, 400);
  near('ray-convex vs ray-box: max |dt|', maxDt, 0);
  near('ray-convex vs ray-box: max |dn|', maxDn, 0, 1e-9);
}

// --- 5. ray vs trimesh (front faces only), brute force and BVH stub -------------------------------------------------
{
  for (const withBvh of [false, true]) {
    const tag = withBvh ? 'bvh stub' : 'brute force';
    const w = stubWorld();
    add(w, gridShape(4, 2, withBvh), 0, 1, 0, Q_ID, BODY.STATIC);
    const h = ray(w, [0.3, 4, -0.7], [0, -1, 0]);
    near(`ray-trimesh (${tag}): down onto the grid at y = 1`, hd(h), 3);
    near(`ray-trimesh (${tag}): normal (0,1,0)`, nErr(h, [0, 1, 0]), 0);
    yes(`ray-trimesh (${tag}): from below (back face) misses`, !ray(w, [0.3, -4, -0.7], [0, 1, 0]));
    yes(`ray-trimesh (${tag}): outside the grid misses`, !ray(w, [2.5, 4, 0], [0, -1, 0]));
    const th = Math.PI / 6, wt = stubWorld();
    add(wt, gridShape(4, 2, withBvh), 0, 1, 0, qAxis(1, 0, 0, th), BODY.STATIC);
    near(`ray-trimesh (${tag}): grid tilted 30 deg about X`, hd(ray(wt, [0.3, 4, -0.7], [0, -1, 0])), 3 - 0.7 * Math.tan(th));
    // oblique ray hits the same point the plane formula gives
    const ho = ray(w, [-3, 3, 1], [1.2, -1, -0.4]);
    const l = Math.hypot(1.2, 1, 0.4);
    near(`ray-trimesh (${tag}): oblique ray (plane intersection)`, hd(ho), 2 * l);
  }
  // brute force == stub BVH over many rays (shared vertices / edges included)
  const wa = stubWorld(), wb = stubWorld(), Hb = hit();
  const q = qAxis(0.2, 0.3, 0.93, 0.4);
  add(wa, gridShape(4, 2, false), 0, 1, 0, q, BODY.STATIC); add(wb, gridShape(4, 2, true), 0, 1, 0, q, BODY.STATIC);
  let bad = 0, hits = 0;
  for (let i = 0; i < 300; i++) {
    const o = [Math.sin(i * 1.7) * 3, 5 + Math.cos(i * 0.3), Math.cos(i * 2.3) * 3], d = [Math.sin(i) * 0.4, -1, Math.cos(i * 0.7) * 0.4];
    const a = raycastBodies(wa, o[0], o[1], o[2], d[0], d[1], d[2], 50, null, H);
    const b = raycastBodies(wb, o[0], o[1], o[2], d[0], d[1], d[2], 50, null, Hb);
    if (a !== b || (a && Math.abs(H.distance - Hb.distance) > 1e-12)) bad++;
    if (a) hits++;
  }
  check('ray-trimesh: brute force == BVH stub (300 rays)', bad, 0, 0);
  check('ray-trimesh: of which hits', hits, 50, 300);
}

// --- 6. compound, nearest body, mask, filter -------------------------------------------------------------------------
{
  const w = stubWorld();
  const c = add(w, compoundShape([{ shape: sphereShape(0.5), pos: [2, 0, 0] }, { shape: sphereShape(0.5), pos: [-2, 0, 0] }]), 0, 0, 5, [0, S, 0, S]);
  const h = ray(w, [0, 0, 0], [0, 0, 1]);
  near('ray-compound: nearer child (world z = 3) at 2.5', hd(h), 2.5);
  near('ray-compound: normal (0,0,-1)', nErr(h, [0, 0, -1]), 0);
  check('ray-compound: body id', h ? h.body : NaN, c, c);
  const w2 = stubWorld();
  const a = add(w2, sphereShape(0.5), 0, 0, 10); const b = add(w2, boxShape(0.5, 0.5, 0.5), 0, 0, 5);
  w2.bodies.layer[a] = 1; w2.bodies.layer[b] = 2;
  check('ray: nearest of two bodies (box at 4.5)', ray(w2, [0, 0, 0], [0, 0, 1])?.body ?? NaN, b, b);
  check('ray: mask skips the box (layer 2)', ray(w2, [0, 0, 0], [0, 0, 1], 100, { mask: 1 })?.body ?? NaN, a, a);
  check('ray: filter skips the box', ray(w2, [0, 0, 0], [0, 0, 1], 100, { filter: (id) => id !== b })?.body ?? NaN, a, a);
  w2.bodies.free(b);
  check('ray: freed body is not hit', ray(w2, [0, 0, 0], [0, 0, 1])?.body ?? NaN, a, a);
  // plain {x,y,z} hit objects work too
  const plain = {};
  yes('ray: plain hit object filled', raycastBodies(w2, 0, 0, 0, 0, 0, 1, 100, null, plain) && plain.point.z === 9.5 && plain.normal.z === -1);
}

// --- 6b. FIRST shape casts of the process: margin-less cores that land exactly touching still report the contact --------
{
  const cases = [
    ['box -> box along +X', boxShape(0.5, 0.5, 0.5), [5, 0, 0], [1, 0, 0], [-1, 0, 0], [4, NaN, NaN]],
    ['box -> box along -Y', boxShape(0.5, 0.5, 0.5), [0, -5, 0], [0, -1, 0], [0, 1, 0], [NaN, -4, NaN]],
    ['box -> box along +Z', boxShape(0.5, 0.5, 0.5), [0, 0, 5], [0, 0, 1], [0, 0, -1], [NaN, NaN, 4]],
    ['convex hull -> box along +X', hullBoxShape(0.5, 0.5, 0.5), [5, 0, 0], [1, 0, 0], [-1, 0, 0], [4, NaN, NaN]],
  ];
  for (const [name, shape, at, dir, n, p] of cases) {
    const w = stubWorld();
    add(w, boxShape(1, 1, 1), at[0], at[1], at[2]);
    const h = hit();
    const ok = shapeCastBodies(w, shape, 0, 0, 0, 0, 0, 0, 1, dir[0], dir[1], dir[2], 100, null, h);
    near(`first casts: ${name}, travel 3.5`, ok ? h.distance : NaN, 3.5, 1e-7);
    near(`first casts: ${name}, face normal`, ok ? Math.hypot(h.normal.x - n[0], h.normal.y - n[1], h.normal.z - n[2]) : NaN, 0, 1e-9);
    const k = Number.isNaN(p[0]) ? (Number.isNaN(p[1]) ? 'z' : 'y') : 'x', want = p['xyz'.indexOf(k)];
    near(`first casts: ${name}, point on the hit face`, ok ? h.point[k] : NaN, want, 1e-7);
  }
}

// --- 7. shape casts (GJK + conservative advancement) -------------------------------------------------------------------
{
  const TOL = 1e-7;
  const cast = (w, shape, p, q, d, max = 100, opts) => (shapeCastBodies(w, shape, p[0], p[1], p[2], q[0], q[1], q[2], q[3], d[0], d[1], d[2], max, opts, H) ? H : null);
  const w = stubWorld();
  const bx = add(w, boxShape(1, 1, 1), 5, 0, 0);
  let h = cast(w, sphereShape(0.5), [0, 0, 0], Q_ID, [2, 0, 0]);
  near('cast sphere -> box face: travel 3.5', hd(h), 3.5, TOL);
  near('cast sphere -> box face: normal (-1,0,0)', nErr(h, [-1, 0, 0]), 0, 1e-6);
  near('cast sphere -> box face: point on the box (4,0,0)', pErr(h, [4, 0, 0]), 0, 1e-6);
  check('cast sphere -> box: body id', h ? h.body : NaN, bx, bx);
  h = cast(w, boxShape(0.5, 0.5, 0.5), [0, 0.2, 0.1], Q_ID, [1, 0, 0]);
  near('cast box -> box face-face: travel 3.5', hd(h), 3.5, TOL);
  h = cast(w, boxShape(0.5, 0.5, 0.5), [0, 0, 0], qAxis(0, 0, 1, Math.PI / 4), [1, 0, 0]);
  near('cast box (45 deg about Z, edge first) -> box: 4 - sqrt(0.5)', hd(h), 4 - S, TOL);
  h = cast(w, hullBoxShape(0.5, 0.5, 0.5), [0, 0, 0], Q_ID, [1, 0, 0]);
  near('cast convex hull -> box: travel 3.5', hd(h), 3.5, TOL);
  h = cast(w, capsuleShape(0.25, 0.5), [0, 0, 0], [0, 0, S, S], [1, 0, 0]);
  near('cast capsule (axis along X) -> box: 4 - 0.75', hd(h), 3.25, TOL);
  const w2 = stubWorld();
  add(w2, sphereShape(1), 5, 0, 0);
  h = cast(w2, sphereShape(0.5), [0, 0.3, 0], Q_ID, [1, 0, 0]);
  near('cast sphere -> sphere off-axis: 5 - sqrt(2.16)', hd(h), 5 - Math.sqrt(2.16), TOL);
  const nx = -Math.sqrt(2.16) / 1.5, ny = 0.3 / 1.5;
  near('cast sphere -> sphere: normal out of the hit sphere', nErr(h, [nx, ny, 0]), 0, 1e-6);
  yes('cast sphere moving away: no hit', !cast(w2, sphereShape(0.5), [0, 0, 0], Q_ID, [-1, 0, 0]));
  yes('cast sphere passing beside: no hit', !cast(w2, sphereShape(0.5), [0, 1.6, 0], Q_ID, [1, 0, 0]));
  yes('cast stops at maxDist', !cast(w2, sphereShape(0.5), [0, 0, 0], Q_ID, [1, 0, 0], 3.4));
  h = cast(w2, sphereShape(0.5), [4.2, 0, 0], Q_ID, [1, 0, 0]);
  near('cast starting in overlap hits at 0', hd(h), 0);
  // trimesh grid (y = 1) under a capsule / tilted box, brute force and BVH stub
  for (const withBvh of [false, true]) {
    const wg = stubWorld();
    add(wg, gridShape(4, 2, withBvh), 0, 1, 0, Q_ID, BODY.STATIC);
    const tag = withBvh ? 'bvh stub' : 'brute';
    h = cast(wg, capsuleShape(0.25, 0.5), [0.3, 5, -0.7], Q_ID, [0, -1, 0]);
    near(`cast capsule -> trimesh (${tag}): 5 - 0.75 - 1`, hd(h), 3.25, TOL);
    near(`cast capsule -> trimesh (${tag}): normal (0,1,0)`, nErr(h, [0, 1, 0]), 0, 1e-6);
    h = cast(wg, boxShape(0.5, 0.5, 0.5), [0.2, 5, 0.1], qAxis(0, 0, 1, Math.PI / 4), [0, -1, 0]);
    near(`cast box corner-down -> trimesh (${tag}): 4 - sqrt(0.5)`, hd(h), 4 - S, TOL);
  }
  // compound cast shape: the lower child leads
  const cs = compoundShape([{ shape: sphereShape(0.25), pos: [0, -1, 0] }, { shape: boxShape(0.5, 0.5, 0.5), pos: [0, 1, 0] }]);
  const wg = stubWorld();
  add(wg, gridShape(4, 2, true), 0, 0, 0, Q_ID, BODY.STATIC);
  near('cast compound (sphere child lowest) -> trimesh: 5 - 1.25', hd(cast(wg, cs, [0, 5, 0], Q_ID, [0, -1, 0])), 3.75, TOL);
}

// --- 8. overlap and containment --------------------------------------------------------------------------------------------
{
  const w = stubWorld();
  const a = add(w, sphereShape(1), 0, 0, 0, Q_ID, BODY.DYNAMIC, 1);
  const b = add(w, boxShape(1, 1, 1), 5, 0, 0, qAxis(0, 1, 0, 0.3), BODY.DYNAMIC, 1.5);
  const c = add(w, capsuleShape(0.5, 1), 0, 0, 5, Q_ID, BODY.DYNAMIC, 1.5);
  w.bodies.layer[c] = 4;
  const out = new Int32Array(16);
  let n = overlapBodies(w, -0.5, -0.5, -0.5, 4, 0.5, 0.5, out);
  check('overlapBodies: box spanning two bodies', n === 2 && out[0] === a && out[1] === b ? 1 : 0, 1, 1);
  n = overlapBodies(w, -10, -10, -10, 10, 10, 10, out, 4);
  check('overlapBodies: layer mask', n === 1 && out[0] === c ? 1 : 0, 1, 1);
  check('overlapBodies: empty region', overlapBodies(w, 20, 20, 20, 21, 21, 21, out), 0, 0);
  const one = new Int32Array(1);
  check('overlapBodies (brute force): short out -> true count, lowest id written', overlapBodies(w, -10, -10, -10, 10, 10, 10, one) === 3 && one[0] === a ? 1 : 0, 1, 1);
  check('overlapBodies (brute force): no truncation before the overlap test', overlapBodies(w, 4, -1, -1, 6, 1, 1, one) === 1 && one[0] === b ? 1 : 0, 1, 1);
  check('pointInBody: inside the sphere', pointInBody(w, 0.5, 0.5, 0.5), a, a);
  check('pointInBody: inside the rotated box corner region', pointInBody(w, 5.9, 0.9, 0), b, b);
  check('pointInBody: inside the capsule cap', pointInBody(w, 0, 1.4, 5), c, c);
  check('pointInBody: outside the capsule cap corner', pointInBody(w, 0.45, 1.45, 5), -1, -1);
  check('pointInBody: in the AABB but outside the sphere', pointInBody(w, 0.8, 0.8, 0), -1, -1);
  const wc = stubWorld();
  const cx = add(wc, hullBoxShape(1, 1, 1), 0, 0, 0);
  check('pointInBody: convex hull inside / outside', pointInBody(wc, 0.99, -0.99, 0.5) === cx && pointInBody(wc, 1.01, 0, 0) === -1 ? 1 : 0, 1, 1);
}

// --- 9. debug lines ---------------------------------------------------------------------------------------------------------
{
  const out = new Float32Array(6 * 4096), col = new Float32Array(6 * 4096);
  const count = (shape, opts) => { const w = stubWorld(); add(w, shape, 1, 2, 3, qAxis(0, 1, 0, 0.5)); return writeDebugLines(w, out, 4096, opts, col); };
  check('debug: sphere = 3 x 16', count(sphereShape(0.5)), 48, 48);
  check('debug: capsule = 2 rings + 4 lines + 4 half circles', count(capsuleShape(0.5, 1)), 68, 68);
  check('debug: box = 12 edges', count(boxShape(1, 1, 1)), 12, 12);
  check('debug: convex = its 12 edges', count(hullBoxShape(1, 1, 1)), 12, 12);
  check('debug: convex without edges / faces = its bounding box', count({ type: SHAPE.CONVEX, points: hullBoxShape(1, 0.5, 2).points }), 12, 12);
  check('debug: trimesh = its AABB (12)', count(gridShape(4, 2)), 12, 12);
  check('debug: compound sphere + box = 60', count(compoundShape([{ shape: sphereShape(0.5), pos: [1, 0, 0] }, { shape: boxShape(1, 1, 1) }])), 60, 60);
  check('debug: DEBUG_LINES table matches', DEBUG_LINES.sphere + DEBUG_LINES.capsule + DEBUG_LINES.box, 128, 128);
  check('debug: + aabbs', count(boxShape(1, 1, 1), { aabbs: true }), 24, 24);
  check('debug: bodies off', count(boxShape(1, 1, 1), { bodies: false }), 0, 0);
  // sphere geometry: first great circle starts at pos + q * (r, 0, 0)
  { const w = stubWorld(); add(w, sphereShape(2), 1, 2, 3, [0, S, 0, S]); writeDebugLines(w, out, 10, undefined, col);
    near('debug: sphere first vertex = p + q (r,0,0)', Math.hypot(out[0] - 1, out[1] - 2, out[2] - (3 - 2)), 0, 1e-6); }
  // contacts, joints, colours by state, the max budget
  const w = stubWorld();
  const s = add(w, boxShape(1, 1, 1), 0, 0, 0, Q_ID, BODY.STATIC);
  const k = add(w, boxShape(1, 1, 1), 3, 0, 0, Q_ID, BODY.KINEMATIC);
  const z = add(w, boxShape(1, 1, 1), 6, 0, 0); w.bodies.flags[z] |= FLAG.ASLEEP;
  const d = add(w, boxShape(1, 1, 1), 9, 0, 0);
  w.contacts.add(d, -1, 9, -1, 0, 0, -1, 0, 0.01, 0, 0); w.contacts.add(z, d, 7, 0, 0, 1, 0, 0, 0.01, 1, 0);
  w.joints = { count: 2, alive: Uint8Array.of(1, 1), type: Uint8Array.of(0, 0), a: Int32Array.of(k, d), b: Int32Array.of(-1, z), anchorA: Float64Array.of(0, 1, 0, 0, 0, 0), anchorB: Float64Array.of(3, 5, 0, 0, 0, 0), axisA: new Float64Array(6), broken: Uint8Array.of(0, 1) };
  const n = writeDebugLines(w, out, 4096, undefined, col);
  check('debug: 4 boxes + 2 contacts + 1 live joint (7)', n, 4 * 12 + 2 + 7, 4 * 12 + 2 + 7);
  const rgb = (line) => [col[line * 6], col[line * 6 + 1], col[line * 6 + 2]].map((v) => Math.round(v * 100) / 100).join(',');
  yes('debug: static blue, kinematic yellow, asleep grey, awake green', rgb(0) === '0.3,0.55,1' && rgb(12) === '1,0.85,0.2' && rgb(24) === '0.5,0.5,0.5' && rgb(36) === '0.25,0.95,0.35');
  const ci = 48 * 6; // first contact line
  near('debug: contact normal line 0.2 m', Math.hypot(out[ci + 3] - out[ci], out[ci + 4] - out[ci + 1], out[ci + 5] - out[ci + 2]), 0.2, 1e-6);
  const ji = 50 * 6; // first joint cross line at body k's anchor (3, 1, 0)
  near('debug: joint cross centred on the world anchor', Math.hypot((out[ji] + out[ji + 3]) / 2 - 3, (out[ji + 1] + out[ji + 4]) / 2 - 1, (out[ji + 2] + out[ji + 5]) / 2), 0, 1e-6);
  check('debug: stops at max', writeDebugLines(w, out, 10, undefined, col), 10, 10);
  check('debug: capped by the array size', writeDebugLines(w, new Float32Array(6 * 5), 1000), 5, 5);
  let finite = true; for (let i = 0; i < n * 6; i++) if (!Number.isFinite(out[i]) || !Number.isFinite(col[i])) finite = false;
  yes('debug: all written floats finite', finite);
}

// --- 10. bridge: provider records, kinematic sync ----------------------------------------------------------------------------
{
  const dyn = { bodies: new BodyStore(32), physics: null, setKinematicTarget(id, ...p) { this.last = [id, ...p]; } };
  const pw = new PhysicsWorld();
  const provider = attachToPhysicsWorld(dyn, pw);
  check('bridge: provider registered, dyn.physics set', pw.providers.includes(provider) && dyn.physics === pw ? 1 : 0, 1, 1);
  const B = dyn.bodies;
  const st = B.alloc(); B.type[st] = BODY.STATIC; setAabb(B, st, -1, 0, -1, 1, 1, 1);
  const dy = B.alloc(); setAabb(B, dy, 2, 0, -0.5, 3, 1, 0.5); B.vel[dy * 3] = 1.5; B.vel[dy * 3 + 2] = -2;
  const kn = B.alloc(); B.type[kn] = BODY.KINEMATIC; setAabb(B, kn, -3, 0, -0.5, -2, 1, 0.5);
  const far = B.alloc(); setAabb(B, far, 50, 0, 50, 51, 1, 51);
  const out = [];
  pw.queryDynamic(0, 0, 3, out);
  check('bridge: non-static bodies within r, ascending', out.length === 2 && out[0].id === dy && out[1].id === kn ? 1 : 0, 1, 1);
  const r0 = out[0];
  check('bridge: record fields', r0.kind === 'rigid' && r0.minX === 2 && r0.maxY === 1 && r0.minZ === -0.5 && r0.vx === 1.5 && r0.vz === -2 ? 1 : 0, 1, 1);
  setAabb(B, dy, 2.5, 0, -0.5, 3.5, 1, 0.5);
  pw.queryDynamic(0, 0, 3, out);
  check('bridge: pooled record reused and refreshed', out[0] === r0 && r0.minX === 2.5 ? 1 : 0, 1, 1);
  check('bridge: XZ circle rule (corner just outside r)', pw.queryDynamic(0, 0, 2.4, out).length, 1, 1);
  const all = pw.queryColliders(0, 0, 3, []);
  check('bridge: queryColliders includes the records', all.length, 2, 2);
  const filt = attachToPhysicsWorld(dyn, pw, { filter: (id) => id !== kn });
  check('bridge: re-attach replaces the provider; filter excludes the proxy', !pw.providers.includes(provider) && pw.providers.length === 1 && pw.queryDynamic(0, 0, 3, out).length === 1 ? 1 : 0, 1, 1);
  const rh = pw.raycast(new Vector3(0, 0.5, 0), new Vector3(1, 0, 0), 10);
  check('bridge: PhysicsWorld.raycast hits the record (kind rigid, id)', rh && rh.kind === 'rigid' && rh.box.id === dy ? rh.distance : NaN, 2.5, 2.5, 'm');
  yes('bridge: detach removes it', detachFromPhysicsWorld(dyn, pw) && !pw.providers.includes(filt) && pw.queryDynamic(0, 0, 3, out).length === 0);
  check('bridge: records carry the DYN_RECORD marker', r0[DYN_RECORD] === true ? 1 : 0, 1, 1);
  const kinds = ['crate', 'vehicle'];
  dyn.kindOf = (id) => kinds[id % 2];
  attachToPhysicsWorld(dyn, pw);
  pw.queryDynamic(0, 0, 3, out);
  check('bridge: kind = dyn.kindOf(id)', out[0].kind === kinds[dy % 2] && out[1].kind === kinds[kn % 2] ? 1 : 0, 1, 1);
  dyn.disposed = true;
  check('bridge: disposed world publishes nothing', pw.queryDynamic(0, 0, 3, out).length, 0, 0);
  dyn.disposed = false; delete dyn.kindOf;
  const before = pw.queryDynamic(0, 0, 3, out)[0];
  B.free(dy); const again = B.alloc(); setAabb(B, again, 2, 0, -0.5, 3, 1, 0.5);
  check('bridge: pooled record identity survives removeBody (documented)', again === dy && pw.queryDynamic(0, 0, 3, out)[0] === before ? 1 : 0, 1, 1);
  detachFromPhysicsWorld(dyn, pw);
  // kinematic sync from a scaled, parented Object3D
  const parent = new Object3D(); parent.position.set(1, 2, 3); parent.quaternion.setFromAxisAngle(new Vector3(0, 1, 0), 2.5); parent.scale.set(2, 2, 2);
  const child = new Object3D(); child.position.set(0.5, -1, 4); child.quaternion.setFromAxisAngle(new Vector3(1, 2, -0.5).normalize(), 2.9);
  parent.add(child);
  let worst = 0;
  for (const ang of [0, 1, 2.2, 3.1, -2.7]) {
    child.quaternion.setFromAxisAngle(new Vector3(1, 2, -0.5).normalize(), ang);
    syncKinematicFromObject(dyn, 7, child);
    const p = new Vector3(), q = new Quaternion(), sc = new Vector3();
    child.matrixWorld.decompose(p, q, sc);
    const [id, x, y, z, qx, qy, qz, qw] = dyn.last;
    const sgn = qx * q.x + qy * q.y + qz * q.z + qw * q.w < 0 ? -1 : 1;
    worst = Math.max(worst, id === 7 ? Math.hypot(x - p.x, y - p.y, z - p.z) + Math.hypot(sgn * qx - q.x, sgn * qy - q.y, sgn * qz - q.z, sgn * qw - q.w) : 1);
  }
  near('syncKinematicFromObject: world pose (5 angles, scaled parent)', worst, 0, 1e-12);
}

// --- 11. bridge + car: determinism gate and the crate (BodyStore stub) -------------------------------------------------------
function carRun(pw, seconds, controls, v0 = 0, onStep) {
  const car = new VehicleController(ROADSTER_SPEC);
  car.reset(new Vector3(0, 0, 0), -Math.PI / 2, pw); // heading -pi/2: drives along +X
  car.velocity.set(v0, 0, 0);
  const n = Math.round(seconds / DT);
  for (let i = 0; i < n; i++) { car.step(DT, controls, pw); if (onStep) onStep(car); }
  return car;
}
function gateWorld() { // car-sim section 11 setup: a PhysicsWorld with a wall
  const pw = new PhysicsWorld();
  pw.addStaticBox(-20, 0, 20, 10, -42, -40, 1);
  return pw;
}
{
  const drive = ctl({ throttle: 0.6, steer: 0.15 });
  const plain = carRun(gateWorld(), 5, drive);
  const pwB = gateWorld();
  attachToPhysicsWorld({ bodies: new BodyStore(16), physics: null }, pwB);
  const bridged = carRun(pwB, 5, drive);
  near('gate (stub): 5 s drive, bridge with 0 bodies == no bridge', plain.position.distanceTo(bridged.position) + plain.velocity.distanceTo(bridged.velocity), 0, 1e-12, 'm');
  // a 1 m crate 10 m ahead (dynamic body: published as a moving collider; not stepped here)
  const crateRun = (attach) => {
    const pw = new PhysicsWorld(), dyn = { bodies: new BodyStore(16), physics: null };
    const B = dyn.bodies, id = B.alloc();
    B.pos[id * 3] = 10.5; B.pos[id * 3 + 1] = 0.5; setAabb(B, id, 10, 0, -0.5, 11, 1, 0.5);
    if (attach) attachToPhysicsWorld(dyn, pw);
    let front = -Infinity, impact = 0;
    const car = carRun(pw, 3, ctl({ throttle: 0.3 }), 10, (c) => { front = Math.max(front, c.position.x + 2.3); impact = Math.max(impact, c.impact); });
    return { car, front, impact };
  };
  const hitRun = crateRun(true), freeRun = crateRun(false);
  check('crate (stub): front bumper never past the crate face (x = 10)', hitRun.front - 10, -2, 0.05, 'm');
  check('crate (stub): final x short of the crate face', 10 - (hitRun.car.position.x + 2.3), 0, 5, 'm');
  check('crate (stub): impact registered', hitRun.impact, 1, 30, 'm/s');
  check('crate (stub): control run without the bridge drives through', freeRun.car.position.x, 11, 1e3, 'm');
}

// --- 12. PhysicsDebug (render layer) -------------------------------------------------------------------------------------------
{
  const w = stubWorld();
  add(w, sphereShape(0.5), 0, 1, 0); add(w, boxShape(1, 1, 1), 3, 1, 0, Q_ID, BODY.STATIC);
  w.contacts.add(0, -1, 0, 0.5, 0, 0, -1, 0, 0.01, 0, 0);
  const dbg = new PhysicsDebug(w, { maxLines: 256, aabbs: true });
  const n = dbg.update();
  check('PhysicsDebug: update() count = sphere + box + 2 aabbs + contact', n, 48 + 12 + 24 + 1, 48 + 12 + 24 + 1);
  const g = dbg.geometry, pos = g.getAttribute('position'), col = g.getAttribute('color');
  check('PhysicsDebug: drawRange = count * 2 vertices', g.drawRange.start === 0 ? g.drawRange.count : NaN, n * 2, n * 2);
  check('PhysicsDebug: upload range covers the written floats', pos.updateRanges.length === 1 && pos.updateRanges[0].count === n * 6 && col.updateRanges[0].count === n * 6 ? 1 : 0, 1, 1);
  yes('PhysicsDebug: line material (vertex colours, no depth test, on top, unculled)', dbg.isLineSegments && dbg.material.vertexColors && !dbg.material.depthTest && dbg.frustumCulled === false && dbg.renderOrder >= 100);
  dbg.enabled = false;
  w.contacts.clear();
  check('PhysicsDebug: disabled -> update() skips (count unchanged)', dbg.update(), n, n);
  dbg.enabled = true;
  check('PhysicsDebug: re-enabled -> one line fewer', dbg.update(), n - 1, n - 1);
  dbg.dispose();
}

// --- 13. integration: real shapes.js + DynamicsWorld ------------------------------------------------------------------------
let shapesMod = null, dynMod = null;
try { shapesMod = await import('../src/physics/dynamics/shapes.js'); } catch (e) { results.push(`SKIP  shapes.js not importable yet (${e.message.split('\n')[0]})`); }
try { dynMod = await import('../src/physics/dynamics/DynamicsWorld.js'); } catch (e) { results.push(`SKIP  DynamicsWorld.js not importable yet (${e.message.split('\n')[0]})`); }

if (shapesMod) {
  const { sphere, capsule, box, convex, trimesh, compound } = shapesMod;
  // real shapes: queries agree with the hand-built ones
  const grid = gridShape(8, 3);
  const real = trimesh(grid.positions, grid.indices);
  const wa = stubWorld(), wb = stubWorld(), Hb = hit();
  const q = qAxis(0.3, 0.1, 0.95, 0.5);
  add(wa, grid, 0, 1, 0, q, BODY.STATIC); add(wb, real, 0, 1, 0, q, BODY.STATIC);
  let bad = 0, hits = 0;
  for (let i = 0; i < 500; i++) {
    const o = [Math.sin(i * 1.3) * 4, 6 + Math.cos(i * 0.2), Math.cos(i * 2.1) * 4], d = [Math.sin(i * 0.9) * 0.5, -1, Math.cos(i * 0.4) * 0.5];
    const a = raycastBodies(wa, o[0], o[1], o[2], d[0], d[1], d[2], 50, null, H);
    const b = raycastBodies(wb, o[0], o[1], o[2], d[0], d[1], d[2], 50, null, Hb);
    if (a !== b || (a && Math.abs(H.distance - Hb.distance) > 1e-12)) bad++;
    if (a) hits++;
  }
  check('shapes.js trimesh BVH ray == brute force (500 rays)', bad, 0, 0);
  check('shapes.js trimesh: of which hits', hits, 100, 500);
  const w = stubWorld();
  add(w, convex(hullBoxShape(1, 1, 1).points), 0, 0, 0);
  near('shapes.js convex(box points): ray at face x = 1', hd(ray(w, [5, 0.2, 0.3], [-1, 0, 0])), 4);
  const wc = stubWorld();
  add(wc, compound([{ shape: sphere(0.5), position: [2, 0, 0] }, { shape: capsule(0.25, 0.5), position: [-2, 0, 0] }, { shape: box(0.5, 0.5, 0.5), position: [0, 2, 0] }]), 0, 0, 0);
  near('shapes.js compound: ray hits the box child top', hd(ray(wc, [0, 10, 0], [0, -1, 0])), 7.5);
  near('shapes.js sphere cast -> compound capsule child (top 0.75)', hd(shapeCastBodies(wc, sphere(0.25), -2, 10, 0, 0, 0, 0, 1, 0, -1, 0, 50, null, H) ? H : null), 10 - 0.25 - 0.75, 1e-7);
  // closed cube trimesh: outward front faces, containment by crossing parity
  const cp = [], ci = [];
  for (let i = 0; i < 8; i++) cp.push(i & 1 ? 1 : -1, i & 2 ? 1 : -1, i & 4 ? 1 : -1);
  for (const [a, b, c, d] of [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]) {
    for (const [u, v, x] of [[a, b, c], [a, c, d]]) {
      const P = (k) => cp.slice(k * 3, k * 3 + 3), A = P(u), Bv = P(v), C = P(x);
      const e1 = Bv.map((q, k) => q - A[k]), e2 = C.map((q, k) => q - A[k]);
      const n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
      const out = n[0] * (A[0] + Bv[0] + C[0]) + n[1] * (A[1] + Bv[1] + C[1]) + n[2] * (A[2] + Bv[2] + C[2]) > 0;
      ci.push(...(out ? [u, v, x] : [u, x, v]));
    }
  }
  const wcube = stubWorld();
  const cube = add(wcube, trimesh(cp, ci), 3, 0, 0, qAxis(0, 1, 0, 0.4), BODY.STATIC);
  const hc = ray(wcube, [-5, 0.2, 0.1], [1, 0, 0]);
  check('shapes.js closed cube trimesh: outside ray hits its outward face', hc && hc.body === cube && hc.normal.x < -0.9 ? 1 : 0, 1, 1);
  yes('shapes.js closed cube trimesh: ray from inside misses (back faces)', !ray(wcube, [3, 0, 0], [1, 0, 0]));
  check('shapes.js closed cube trimesh: pointInBody parity inside / outside', pointInBody(wcube, 3.5, 0.5, 0.2) === cube && pointInBody(wcube, 4.6, 0, 0) === -1 ? 1 : 0, 1, 1);
}

// broadphase candidate paths (ascending ids from the real Broadphase) give the same answers as brute force
let bpMod = null;
try { bpMod = await import('../src/physics/dynamics/broadphase.js'); } catch (e) { results.push(`SKIP  broadphase.js not importable yet (${e.message.split('\n')[0]})`); }
if (bpMod && typeof bpMod.Broadphase === 'function') {
  const w = stubWorld(256), wb = stubWorld(256);
  let seed = 11;
  const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
  for (let i = 0; i < 150; i++) {
    const kind = i % 4, x = rnd() * 40 - 20, y = rnd() * 6, z = rnd() * 40 - 20;
    const ax = rnd() - 0.5, ay = rnd() - 0.5, az = rnd() - 0.5, al = Math.hypot(ax, ay, az) || 1, q = qAxis(ax / al, ay / al, az / al, rnd() * 6);
    const shape = kind === 0 ? sphereShape(0.3 + rnd()) : kind === 1 ? capsuleShape(0.3, 0.2 + rnd()) : kind === 2 ? boxShape(0.3 + rnd(), 0.3 + rnd(), 0.3 + rnd()) : hullBoxShape(0.5, 0.4, 0.6);
    const r = 0.3 + 2.3; // conservative AABB half size for every shape above
    for (const ww of [w, wb]) { const id = add(ww, shape, x, y, z, q, BODY.DYNAMIC, r); ww.bodies.layer[id] = 1 << (i % 3); }
  }
  w.broadphase = new bpMod.Broadphase(256);
  w.broadphase.update(w.bodies);
  const Hb = hit(), ids = new Int32Array(256), idsB = new Int32Array(256);
  let rayBad = 0, castBad = 0, ovBad = 0, ptBad = 0, rayHits = 0, castHits = 0;
  for (let i = 0; i < 300; i++) {
    const o = [rnd() * 50 - 25, rnd() * 8 - 1, rnd() * 50 - 25], d = [rnd() - 0.5, rnd() * 0.4 - 0.2, rnd() - 0.5], mask = i % 5 === 0 ? 2 : -1;
    const a = raycastBodies(w, o[0], o[1], o[2], d[0], d[1], d[2], 60, { mask }, H), b = raycastBodies(wb, o[0], o[1], o[2], d[0], d[1], d[2], 60, { mask }, Hb);
    if (a !== b || (a && (H.body !== Hb.body || H.distance !== Hb.distance))) rayBad++;
    if (a) rayHits++;
    const cs = i % 2 ? sphereShape(0.4) : boxShape(0.3, 0.5, 0.2);
    const ca = shapeCastBodies(w, cs, o[0], o[1], o[2], 0, 0, 0, 1, d[0], d[1], d[2], 60, { mask }, H), cb = shapeCastBodies(wb, cs, o[0], o[1], o[2], 0, 0, 0, 1, d[0], d[1], d[2], 60, { mask }, Hb);
    if (ca !== cb || (ca && (H.body !== Hb.body || H.distance !== Hb.distance))) castBad++;
    if (ca) castHits++;
    const na = overlapBodies(w, o[0] - 3, o[1] - 3, o[2] - 3, o[0] + 3, o[1] + 3, o[2] + 3, ids, mask);
    const nb = overlapBodies(wb, o[0] - 3, o[1] - 3, o[2] - 3, o[0] + 3, o[1] + 3, o[2] + 3, idsB, mask);
    if (na !== nb || ids.subarray(0, na).some((v, k) => v !== idsB[k])) ovBad++;
    if (pointInBody(w, o[0] * 0.8, o[1], o[2] * 0.8) !== pointInBody(wb, o[0] * 0.8, o[1], o[2] * 0.8)) ptBad++;
  }
  check('broadphase path == brute force: raycastBodies (300, masks)', rayBad, 0, 0);
  check('broadphase path == brute force: shapeCastBodies (300)', castBad, 0, 0);
  check('broadphase path == brute force: overlapBodies (300)', ovBad, 0, 0);
  check('broadphase path == brute force: pointInBody (300)', ptBad, 0, 0);
  check('broadphase path: ray / cast hit counts', Math.min(rayHits, castHits), 20, 300);
}


if (dynMod && shapesMod && typeof dynMod.DynamicsWorld === 'function') {
  const { DynamicsWorld } = dynMod;
  const { box } = shapesMod;
  // determinism gate: same run with a bridged, empty DynamicsWorld
  const drive = ctl({ throttle: 0.6, steer: 0.15 });
  const plain = carRun(gateWorld(), 5, drive);
  const pw = gateWorld();
  const dyn = new DynamicsWorld({ physics: pw });
  attachToPhysicsWorld(dyn, pw);
  const bridged = carRun(pw, 5, drive);
  near('gate (DynamicsWorld): bridge with 0 bodies == no bridge', plain.position.distanceTo(bridged.position) + plain.velocity.distanceTo(bridged.velocity), 0, 1e-12, 'm');
  // a 1 m crate (dynamic body) 10 m ahead of the car, a static box behind it; the dynamics world is not stepped
  const pw2 = new PhysicsWorld();
  pw2.addStaticBox(14, 0, 15, 3, -2, 2, 2);
  const dyn2 = new DynamicsWorld({ physics: pw2 });
  const crate = dyn2.addBody({ shape: box(0.5, 0.5, 0.5), position: [10.5, 0.5, 0], mass: 20 });
  attachToPhysicsWorld(dyn2, pw2);
  const B = dyn2.bodies;
  const recs = pw2.queryDynamic(10.5, 0, 1, []);
  const tagged = dyn2.addBody({ shape: box(0.25, 0.25, 0.25), position: [0, 0.25, 30], mass: 5, kind: 'barrel' });
  const tr = pw2.queryDynamic(0, 30, 1, []);
  check('DynamicsWorld + bridge: record kind = the body kind option, DYN_RECORD set', recs.length === 1 && recs[0].kind === 'rigid' && tr.length === 1 && tr[0].id === tagged && tr[0].kind === 'barrel' && tr[0][DYN_RECORD] === true ? 1 : 0, 1, 1);
  dyn2.removeBody(tagged);
  check('DynamicsWorld crate: AABB published at addBody (minX, margin included)', B.aabb[crate * 6], 9.98, 10, 'm');
  let front = -Infinity, impact = 0;
  const car = carRun(pw2, 3, ctl({ throttle: 0.3 }), 10, (c) => { front = Math.max(front, c.position.x + 2.3); impact = Math.max(impact, c.impact); });
  check('DynamicsWorld crate: car front never past the crate face', front - 10, -2, 0.05, 'm');
  check('DynamicsWorld crate: final x short of the crate face', 10 - (car.position.x + 2.3), 0, 5, 'm');
  check('DynamicsWorld crate: impact registered', impact, 1, 30, 'm/s');
  // queries go through the broadphase, which DynamicsWorld updates in step(): note whether a fresh body is visible
  const pre = raycastBodies(dyn2, 0, 0.5, 0, 1, 0, 0, 100, null, hit());
  if (!pre) results.push('NOTE  broadphase-backed queries miss a body until the first step() after addBody (DynamicsWorld updates the broadphase only in step)');
  // the bridged world steps cleanly: the crate rests (its own records are not read back as static boxes)
  for (let i = 0; i < 120; i++) dyn2.step(1 / 60);
  const cx = B.pos[crate * 3], cy = B.pos[crate * 3 + 1];
  check('DynamicsWorld + bridge: crate rests after 2 s (|dx| + |dy - 0.5|)', Math.abs(cx - 10.5) + Math.abs(cy - 0.5), 0, 0.02, 'm');
  // raycast through the world: the crate (body id) is nearer than the static box behind it
  const Hd = hit();
  const okB = raycastBodies(dyn2, 0, 0.5, 0, 1, 0, 0, 100, null, Hd);
  check('raycastBodies(DynamicsWorld): the crate face', okB && Hd.body === crate ? Hd.distance : NaN, cx - 0.5 - 1e-9, cx - 0.5 + 1e-9, 'm');
  const r = dyn2.raycast(new Vector3(0, 0.5, 0), new Vector3(1, 0, 0), 100);
  check('DynamicsWorld.raycast: nearest is the crate (body id), not the static box', r && r.body === crate ? r.distance : NaN, cx - 0.5 - 1e-9, cx - 0.5 + 1e-9, 'm');
  const rs = dyn2.raycast(new Vector3(0, 0.5, 0), new Vector3(1, 0, 0), 100, { bodies: false });
  check('DynamicsWorld.raycast (bodies off): static box at 14 m, not the bridged record', rs && rs.body === -1 && rs.box && rs.box.kind !== 'rigid' ? rs.distance : NaN, 14 - 1e-9, 14 + 1e-9, 'm');
  const ids = new Int32Array(8);
  const n = overlapBodies(dyn2, 9, 0, -1, 12, 2, 1, ids);
  check('overlapBodies(DynamicsWorld): returns the crate', n === 1 && ids[0] === crate ? 1 : 0, 1, 1);
  check('pointInBody(DynamicsWorld): crate centre', pointInBody(dyn2, cx, cy, 0), crate, crate);
  const sc = dyn2.shapeCast(shapesMod.sphere(0.25), [cx, 5, 0], null, [0, -1, 0], 10);
  check('DynamicsWorld.shapeCast: sphere dropped onto the crate top', sc && sc.body === crate ? sc.distance : NaN, 5 - 0.25 - (cy + 0.5) - 1e-6, 5 - 0.25 - (cy + 0.5) + 1e-6, 'm');
  // kinematic proxy driven from an Object3D lands exactly on the target after a step
  const kin = dyn2.addBody({ type: 'kinematic', shape: box(1, 0.5, 2), position: [0, 0.5, -20] });
  const proxy = new Object3D(); proxy.position.set(1, 0.6, -19); proxy.quaternion.setFromAxisAngle(new Vector3(0, 1, 0), 0.3);
  syncKinematicFromObject(dyn2, kin, proxy);
  dyn2.step(1 / 60);
  const kq = proxy.quaternion;
  check('syncKinematicFromObject(DynamicsWorld): body on the Object3D pose after a step', Math.hypot(B.pos[kin * 3] - 1, B.pos[kin * 3 + 1] - 0.6, B.pos[kin * 3 + 2] + 19) + Math.hypot(B.quat[kin * 4] - kq.x, B.quat[kin * 4 + 1] - kq.y, B.quat[kin * 4 + 2] - kq.z, B.quat[kin * 4 + 3] - kq.w), 0, 1e-9);
  const dbg = new PhysicsDebug(dyn2, { maxLines: 1024 });
  const ln = dbg.update();
  check('PhysicsDebug(DynamicsWorld): update() writes count * 2 vertices', dbg.geometry.drawRange.count === ln * 2 ? ln : NaN, 24, 1024);
  dbg.dispose();
} else if (shapesMod) results.push('SKIP  DynamicsWorld integration (module missing): the lead re-runs this suite');

console.log(results.join('\n'));
console.log(failures ? `\n${failures} check(s) FAILED` : '\nAll dynamics query / bridge / debug checks passed.');
process.exit(failures ? 1 : 0);
