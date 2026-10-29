// Headless checks for the dynamics narrowphase (src/physics/dynamics/narrowphase.js): every pair type against
// analytic depth / normal / point, speculative contacts inside the margin and none beyond it, box-box manifolds (face
// and edge), GJK / EPA against the analytic box paths, warm-start keys under jitter, heightfield, static boxes,
// trimesh one-sidedness and merging, compounds, and that collide() does not allocate. No browser, no WebGL.
// Run: node test/dyn-narrowphase.mjs  (imports the modules by path until the physics index exports them)
import v8 from 'node:v8';
import { mulberry32, hashF64Array } from '../src/core/mathDet.js';
import { BodyStore, ContactBuffer, WORLD_BODY, pairKey } from '../src/physics/dynamics/state.js';
import { sphere, capsule, box, convex, trimesh, compound } from '../src/physics/dynamics/shapes.js';
import { Narrowphase } from '../src/physics/dynamics/narrowphase.js';

let failures = 0;
const results = [];
function check(name, value, lo = 1, hi = lo, unit = '') {
  if (typeof value === 'boolean') value = value ? 1 : 0;
  const ok = Number.isFinite(value) && value >= lo && value <= hi;
  if (!ok) failures++;
  results.push(`${ok ? 'PASS' : 'FAIL'}  ${name.padEnd(64)} ${fmt(value)} ${unit}   [${fmt(lo).trim()}${hi !== lo ? ' .. ' + fmt(hi).trim() : ''}]`);
  return ok;
}
const fmt = (x) => (!Number.isFinite(x) ? String(x) : Number.isInteger(x) ? String(x) : Math.abs(x) < 1e-3 ? x.toExponential(2) : x.toFixed(4)).padStart(10);
const err = (name, v, e, tol) => check(`${name}`, Math.abs(v - e), 0, tol);

// ---- fixtures ---------------------------------------------------------------------------------------------------------
const MARGIN = 0.01;
const B = new BodyStore(256);
const C = new ContactBuffer(8192);
const np = new Narrowphase({ margin: MARGIN });
const Q0 = [0, 0, 0, 1];
function quat(ax, ay, az, ang) {
  const l = Math.hypot(ax, ay, az), s = Math.sin(ang / 2) / l;
  return [ax * s, ay * s, az * s, Math.cos(ang / 2)];
}
function qmul(a, b) {
  return [a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1], a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
    a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3], a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]];
}
function rot(q, v) {
  const [x, y, z, w] = q, tx = 2 * (y * v[2] - z * v[1]), ty = 2 * (z * v[0] - x * v[2]), tz = 2 * (x * v[1] - y * v[0]);
  return [v[0] + w * tx + (y * tz - z * ty), v[1] + w * ty + (z * tx - x * tz), v[2] + w * tz + (x * ty - y * tx)];
}
function body(shape, p = [0, 0, 0], q = Q0) { const id = B.alloc(); B.shape[id] = shape; pose(id, p, q); return id; }
function pose(id, p, q = Q0) {
  B.pos[id * 3] = p[0]; B.pos[id * 3 + 1] = p[1]; B.pos[id * 3 + 2] = p[2];
  B.quat[id * 4] = q[0]; B.quat[id * 4 + 1] = q[1]; B.quat[id * 4 + 2] = q[2]; B.quat[id * 4 + 3] = q[3];
}
function list(from = 0) {
  const out = [];
  for (let i = from; i < C.count; i++) {
    out.push({ p: [C.px[i], C.py[i], C.pz[i]], n: [C.nx[i], C.ny[i], C.nz[i]], d: C.depth[i], key: C.key[i], a: C.a[i], b: C.b[i], pk: C.pairKey[i], si: C.staticIndex[i], ln: C.lambdaN[i], lt: C.lambdaT[i] });
  }
  return out;
}
function hit(a, b) { C.clear(); np.collide(B, a, b, C); return list(); }
const dist3 = (u, v) => Math.hypot(u[0] - v[0], u[1] - v[1], u[2] - v[2]);
const nlen = (n) => Math.hypot(n[0], n[1], n[2]);
// max over contacts of A of the distance to the nearest point of B (and vice versa), normals and depths compared too
function matchSets(A, Bs) {
  let e = 0;
  if (A.length !== Bs.length) return Infinity;
  for (const c of A) {
    let best = Infinity, bc = null;
    for (const d of Bs) { const x = dist3(c.p, d.p); if (x < best) { best = x; bc = d; } }
    e = Math.max(e, best, dist3(c.n, bc.n), Math.abs(c.d - bc.d));
  }
  return e;
}
function hasPoint(cs, p, tol) { return cs.some((c) => dist3(c.p, p) < tol); }
function allNormal(cs, n, tol) { return cs.length > 0 && cs.every((c) => dist3(c.n, n) < tol); }
function allDepth(cs, d, tol) { return cs.length > 0 && cs.every((c) => Math.abs(c.d - d) < tol); }

// ---- 1. sphere-sphere -----------------------------------------------------------------------------------------------------
{
  const a = body(sphere(0.5), [0, 0, 0]), b = body(sphere(0.5), [0.9, 0, 0]);
  let c = hit(a, b);
  check('sphere-sphere: 1 contact', c.length, 1);
  err('sphere-sphere: depth 0.1', c[0].d, 0.1, 1e-9);
  check('sphere-sphere: normal (1,0,0) a->b', dist3(c[0].n, [1, 0, 0]), 0, 1e-9);
  check('sphere-sphere: point midpoint (0.45,0,0)', dist3(c[0].p, [0.45, 0, 0]), 0, 1e-9);
  check('sphere-sphere: pairKey / a / b', c[0].pk === pairKey(a, b) && c[0].a === a && c[0].b === b);
  c = hit(b, a);
  check('sphere-sphere reversed: normal (-1,0,0)', dist3(c[0].n, [-1, 0, 0]), 0, 1e-9);
  pose(b, [0.3, 0.6, -0.5]);
  c = hit(a, b);
  const d = Math.hypot(0.3, 0.6, 0.5);
  err('sphere-sphere diagonal: depth', c[0].d, 1 - d, 1e-9);
  check('sphere-sphere diagonal: normal', dist3(c[0].n, [0.3 / d, 0.6 / d, -0.5 / d]), 0, 1e-9);
  pose(b, [1.005, 0, 0]);
  c = hit(a, b);
  check('sphere-sphere: separated 5 mm -> speculative', c.length === 1 && Math.abs(c[0].d + 0.005) < 1e-9);
  pose(b, [1.02, 0, 0]);
  check('sphere-sphere: separated 20 mm -> none', hit(a, b).length, 0);
  B.free(a); B.free(b);
}

// ---- 2. sphere-capsule ------------------------------------------------------------------------------------------------------
{
  const cap = body(capsule(0.3, 1), [0, 0, 0]), s = body(sphere(0.5), [0.7, 0.5, 0]);
  let c = hit(s, cap);
  check('sphere-capsule side: 1 contact', c.length, 1);
  err('sphere-capsule side: depth 0.1', c[0].d, 0.1, 1e-9);
  check('sphere-capsule side: normal (-1,0,0)', dist3(c[0].n, [-1, 0, 0]), 0, 1e-9);
  check('sphere-capsule side: point (0.25,0.5,0)', dist3(c[0].p, [0.25, 0.5, 0]), 0, 1e-9);
  pose(s, [0, 1.7, 0]);
  c = hit(cap, s);
  err('capsule-sphere cap: depth 0.1', c[0].d, 0.1, 1e-9);
  check('capsule-sphere cap: normal (0,1,0) capsule->sphere', dist3(c[0].n, [0, 1, 0]), 0, 1e-9);
  pose(cap, [1, 2, 3], quat(0, 0, 1, Math.PI / 2)); // axis along -x
  pose(s, [1 + 0.4, 2 - 0.75, 3]);
  c = hit(s, cap);
  err('sphere-capsule rotated: depth 0.05', c[0].d, 0.05, 1e-9);
  check('sphere-capsule rotated: normal (0,1,0)', dist3(c[0].n, [0, 1, 0]), 0, 1e-9);
  pose(s, [1.4, 2 - 0.805, 3]);
  c = hit(s, cap);
  check('sphere-capsule: speculative depth -0.005', c.length === 1 && Math.abs(c[0].d + 0.005) < 1e-9);
  pose(s, [1.4, 2 - 0.82, 3]);
  check('sphere-capsule: beyond margin -> none', hit(s, cap).length, 0);
  B.free(cap); B.free(s);
}

// ---- 3. capsule-capsule -----------------------------------------------------------------------------------------------------
{
  const A = body(capsule(0.3, 1), [0, 0, 0]), Bc = body(capsule(0.3, 1), [0, 0, 0.5], quat(0, 0, 1, Math.PI / 2));
  let c = hit(A, Bc);
  check('capsule-capsule crossed: 1 contact', c.length, 1);
  err('capsule-capsule crossed: depth 0.1', c[0].d, 0.1, 1e-9);
  check('capsule-capsule crossed: normal (0,0,1)', dist3(c[0].n, [0, 0, 1]), 0, 1e-9);
  check('capsule-capsule crossed: point (0,0,0.25)', dist3(c[0].p, [0, 0, 0.25]), 0, 1e-9);
  pose(Bc, [0.55, 0.5, 0]);
  c = hit(A, Bc);
  check('capsule-capsule parallel: 2 contacts', c.length, 2);
  check('capsule-capsule parallel: depth 0.05 both', allDepth(c, 0.05, 1e-9));
  check('capsule-capsule parallel: ends of overlap y=-0.5, y=1', hasPoint(c, [0.275, -0.5, 0], 1e-9) && hasPoint(c, [0.275, 1, 0], 1e-9));
  check('capsule-capsule parallel: normal (1,0,0)', allNormal(c, [1, 0, 0], 1e-9));
  check('capsule-capsule parallel: distinct keys', c.length === 2 && c[0].key !== c[1].key);
  pose(Bc, [0, 0, 0.605], quat(0, 0, 1, Math.PI / 2));
  c = hit(A, Bc);
  check('capsule-capsule: speculative -0.005', c.length === 1 && Math.abs(c[0].d + 0.005) < 1e-9);
  pose(Bc, [0, 0, 0.62], quat(0, 0, 1, Math.PI / 2));
  check('capsule-capsule: beyond margin -> none', hit(A, Bc).length, 0);
  B.free(A); B.free(Bc);
}

// ---- 4. sphere-box -----------------------------------------------------------------------------------------------------------
{
  const bx = body(box(0.5, 0.5, 0.5)), s = body(sphere(0.5), [0.2, 0.95, -0.1]);
  let c = hit(s, bx);
  check('sphere-box face: 1 contact', c.length, 1);
  err('sphere-box face: depth 0.05', c[0].d, 0.05, 1e-9);
  check('sphere-box face: normal (0,-1,0) sphere->box', dist3(c[0].n, [0, -1, 0]), 0, 1e-9);
  check('sphere-box face: point (0.2,0.475,-0.1)', dist3(c[0].p, [0.2, 0.475, -0.1]), 0, 1e-9);
  pose(s, [0.8, 0.8, 0]);
  c = hit(s, bx);
  err('sphere-box edge: depth 0.5 - sqrt(0.18)', c[0].d, 0.5 - Math.sqrt(0.18), 1e-9);
  check('sphere-box edge: normal (-1,-1,0)/sqrt2', dist3(c[0].n, [-Math.SQRT1_2, -Math.SQRT1_2, 0]), 0, 1e-9);
  pose(s, [0.75, 0.75, 0.75]);
  c = hit(s, bx);
  err('sphere-box corner: depth 0.5 - sqrt(0.1875)', c[0].d, 0.5 - Math.sqrt(0.1875), 1e-9);
  pose(s, [0.1, 0.3, 0]);
  c = hit(s, bx);
  err('sphere-box centre inside: depth 0.7', c[0].d, 0.7, 1e-9);
  check('sphere-box centre inside: normal (0,-1,0)', dist3(c[0].n, [0, -1, 0]), 0, 1e-9);
  check('sphere-box centre inside: point (0.1,0.15,0)', dist3(c[0].p, [0.1, 0.15, 0]), 0, 1e-9);
  const q = quat(0.3, 1, -0.2, 0.7), bp = [3, -1, 2];
  pose(bx, bp, q);
  const lp = [0.2, 0.95, -0.1], wp = rot(q, lp);
  pose(s, [bp[0] + wp[0], bp[1] + wp[1], bp[2] + wp[2]]);
  c = hit(bx, s);
  err('box-sphere rotated face: depth 0.05', c[0].d, 0.05, 1e-9);
  check('box-sphere rotated face: normal = box +y axis (box->sphere)', dist3(c[0].n, rot(q, [0, 1, 0])), 0, 1e-9);
  pose(bx, [0, 0, 0]);
  pose(s, [0.2, 1.005, -0.1]);
  c = hit(s, bx);
  check('sphere-box: speculative -0.005', c.length === 1 && Math.abs(c[0].d + 0.005) < 1e-9);
  pose(s, [0.2, 1.02, -0.1]);
  check('sphere-box: beyond margin -> none', hit(s, bx).length, 0);
  B.free(bx); B.free(s);
}

// ---- 5. capsule-box -------------------------------------------------------------------------------------------------------------
{
  const bx = body(box(1, 0.5, 1)), qz = quat(0, 0, 1, Math.PI / 2);
  const cp = body(capsule(0.2, 0.5), [0.1, 0.69, 0.2], qz);
  let c = hit(cp, bx);
  check('capsule-box lying: 2 contacts', c.length, 2);
  check('capsule-box lying: depth 0.01 both', allDepth(c, 0.01, 1e-9));
  check('capsule-box lying: normal (0,-1,0)', allNormal(c, [0, -1, 0], 1e-9));
  check('capsule-box lying: points at both ends', hasPoint(c, [-0.4, 0.495, 0.2], 1e-9) && hasPoint(c, [0.6, 0.495, 0.2], 1e-9));
  pose(cp, [0.1, 0.705, 0.2], qz);
  c = hit(cp, bx);
  check('capsule-box lying: speculative 2 x -0.005', c.length === 2 && allDepth(c, -0.005, 1e-9));
  pose(cp, [0.1, 0.72, 0.2], qz);
  check('capsule-box lying: beyond margin -> none', hit(cp, bx).length, 0);
  pose(cp, [0.95, 0.69, 0.2], qz); // overhanging the +x edge: clipped at the face boundary
  c = hit(cp, bx);
  check('capsule-box overhang: 2 contacts, one at x = 1', c.length === 2 && hasPoint(c, [1, 0.495, 0.2], 1e-9) && hasPoint(c, [0.45, 0.495, 0.2], 1e-9));
  pose(cp, [0.3, 1.19, 0]);
  c = hit(cp, bx);
  check('capsule-box standing: 1 contact', c.length, 1);
  err('capsule-box standing: depth 0.01', c[0].d, 0.01, 1e-9);
  check('capsule-box standing: point (0.3,0.495,0)', dist3(c[0].p, [0.3, 0.495, 0]), 0, 1e-9);
  pose(cp, [1.1, 0.6, 0], quat(1, 0, 0, Math.PI / 2)); // along z, beside the +x top edge
  c = hit(cp, bx);
  check('capsule-box edge: 1 contact', c.length, 1);
  err('capsule-box edge: depth 0.2 - sqrt(0.02)', c[0].d, 0.2 - Math.sqrt(0.02), 1e-9);
  check('capsule-box edge: normal (-1,-1,0)/sqrt2', dist3(c[0].n, [-Math.SQRT1_2, -Math.SQRT1_2, 0]), 0, 1e-9);
  pose(cp, [0.3, 0.6, 0]); // core inside the box
  c = hit(cp, bx);
  check('capsule-box core inside: 1 contact', c.length, 1);
  err('capsule-box core inside: depth 0.6', c[0].d, 0.6, 1e-9);
  check('capsule-box core inside: normal (0,-1,0), point (0.3,0.2,0)', dist3(c[0].n, [0, -1, 0]) + dist3(c[0].p, [0.3, 0.2, 0]), 0, 1e-9);
  const q = quat(1, 2, 3, 0.4), bp = [-2, 5, 1];
  pose(bx, bp, q);
  const w0 = rot(q, [0.1, 0.69, 0.2]);
  pose(cp, [bp[0] + w0[0], bp[1] + w0[1], bp[2] + w0[2]], qmul(q, qz));
  c = hit(bx, cp);
  check('box-capsule rotated lying: 2 contacts depth 0.01, normal +y_box', c.length === 2 && allDepth(c, 0.01, 1e-9) && allNormal(c, rot(q, [0, 1, 0]), 1e-9));
  B.free(bx); B.free(cp);
}

// ---- 6. box-box --------------------------------------------------------------------------------------------------------------
{
  const A = body(box(0.5, 0.5, 0.5)), Bb = body(box(0.5, 0.5, 0.5), [0.2, 0.99, 0.1]);
  let c = hit(A, Bb);
  check('box-box face: 4 contacts', c.length, 4);
  check('box-box face: depth = overlap 0.01 (1e-9)', allDepth(c, 0.01, 1e-9));
  check('box-box face: normal (0,1,0)', allNormal(c, [0, 1, 0], 1e-12));
  check('box-box face: points = overlap rectangle corners', [[-0.3, -0.4], [0.5, -0.4], [0.5, 0.5], [-0.3, 0.5]].every(([x, z]) => hasPoint(c, [x, 0.495, z], 1e-9)));
  c = hit(Bb, A);
  check('box-box face reversed: 4 contacts, normal (0,-1,0)', c.length === 4 && allNormal(c, [0, -1, 0], 1e-12) && allDepth(c, 0.01, 1e-9));
  const small = body(box(0.3, 0.3, 0.3), [0.1, 0.79, 0]);
  c = hit(A, small);
  check('box-box small on big: 4 corners of the small box', c.length === 4 && [[-0.2, -0.3], [0.4, -0.3], [0.4, 0.3], [-0.2, 0.3]].every(([x, z]) => hasPoint(c, [x, 0.495, z], 1e-9)) && allDepth(c, 0.01, 1e-9));
  pose(Bb, [0.2, 1.005, 0.1]);
  c = hit(A, Bb);
  check('box-box: speculative 4 x -0.005', c.length === 4 && allDepth(c, -0.005, 1e-9));
  pose(Bb, [0.2, 1.02, 0.1]);
  check('box-box: beyond margin -> none', hit(A, Bb).length, 0);
  // edge-edge: A rotated 45 deg about z (top edge along z), B rotated 45 deg about x (bottom edge along x)
  const r2 = Math.SQRT2 / 2;
  pose(A, [0, 0, 0], quat(0, 0, 1, Math.PI / 4));
  pose(Bb, [0, 2 * r2 - 0.01, 0], quat(1, 0, 0, Math.PI / 4));
  c = hit(A, Bb);
  check('box-box edge-edge: 1 contact', c.length, 1);
  err('box-box edge-edge: depth 0.01', c[0].d, 0.01, 1e-9);
  check('box-box edge-edge: normal (0,1,0)', dist3(c[0].n, [0, 1, 0]), 0, 1e-9);
  check('box-box edge-edge: point (0, r2 - 0.005, 0)', dist3(c[0].p, [0, r2 - 0.005, 0]), 0, 1e-9);
  // tilted box resting on an edge: two points
  const th = 10 * Math.PI / 180, lowest = 0.5 * (Math.cos(th) + Math.sin(th));
  pose(A, [0, 0, 0]);
  pose(Bb, [0, 0.5 + lowest - 0.01, 0], quat(0, 0, 1, th));
  c = hit(A, Bb);
  check('box-box tilted on edge: 2 contacts depth 0.01', c.length === 2 && allDepth(c, 0.01, 1e-9) && allNormal(c, [0, 1, 0], 1e-9));
  B.free(A); B.free(Bb); B.free(small);
}

// ---- 7. GJK / EPA against the analytic box paths ------------------------------------------------------------------------------
{
  const corners = (hx, hy, hz) => { const p = []; for (let i = 0; i < 8; i++) p.push(i & 1 ? hx : -hx, i & 2 ? hy : -hy, i & 4 ? hz : -hz); return p; };
  const hull5 = convex(corners(0.5, 0.5, 0.5));
  check('convex(box corners): 6 faces, 8 vertices', hull5.faceCount === 6 && hull5.vertexCount === 8);
  const A = body(box(0.5, 0.5, 0.5)), Bb = body(box(0.5, 0.5, 0.5)), H = body(hull5);
  const cfgs = [
    ['face offset', [0.2, 0.99, 0.1], Q0],
    ['face speculative', [0.2, 1.004, 0.1], Q0],
    ['tilted on edge', [0.1, 0.5 + 0.5 * (Math.cos(0.2) + Math.sin(0.2)) - 0.01, -0.05], quat(0, 0, 1, 0.2)],
    ['tilted on corner', [0.05, 0.5 + 0.4, 0.02], qmul(quat(0, 0, 1, 0.3), quat(1, 0, 0, 0.4))],
    ['yawed 30 deg', [0.3, 0.995, -0.2], quat(0, 1, 0, Math.PI / 6)],
    ['edge-edge', null, null],
  ];
  for (const [name, p, q] of cfgs) {
    let pa = [0, 0, 0], qa = Q0, pb = p, qb = q;
    if (!p) { qa = quat(0, 0, 1, Math.PI / 4); pb = [0.02, Math.SQRT2 - 0.01, -0.03]; qb = quat(1, 0, 0, Math.PI / 4); }
    if (name === 'tilted on corner') { // lower it until the lowest corner penetrates 0.01
      let low = Infinity;
      for (let i = 0; i < 8; i++) low = Math.min(low, rot(qb, [i & 1 ? 0.5 : -0.5, i & 2 ? 0.5 : -0.5, i & 4 ? 0.5 : -0.5])[1]);
      pb = [p[0], 0.5 - low - 0.01, p[2]];
    }
    pose(A, pa, qa); pose(Bb, pb, qb); pose(H, pb, qb);
    const ref = hit(A, Bb), got = hit(A, H), rev = hit(H, A);
    check(`hull vs box == box-box (${name}), ${ref.length} pts`, ref.length > 0 ? matchSets(got, ref) : Infinity, 0, 1e-6);
    check(`box vs hull reversed (${name}): negated normals`, matchSets(rev.map((c) => ({ ...c, n: c.n.map((v) => -v) })), ref), 0, 1e-6);
  }
  // sphere vs hull (GJK / EPA) == sphere vs box (analytic), random poses around a rotated 0.5 x 0.3 x 0.4 box
  const hb = body(box(0.5, 0.3, 0.4)), hh = body(convex(corners(0.5, 0.3, 0.4))), s = body(sphere(0.35));
  const rnd = mulberry32(7);
  const qh = quat(0.2, 1, 0.4, 1.1), ph = [1, 2, 3];
  pose(hb, ph, qh); pose(hh, ph, qh);
  let worst = 0, n = 0, inside = 0, spec = 0;
  for (let k = 0; k < 400; k++) {
    const lp = [(rnd() * 2 - 1) * 0.9, (rnd() * 2 - 1) * 0.7, (rnd() * 2 - 1) * 0.8];
    const w = rot(qh, lp);
    pose(s, [ph[0] + w[0], ph[1] + w[1], ph[2] + w[2]]);
    const ref = hit(s, hb), got = hit(s, hh);
    if (ref.length !== got.length) { worst = Infinity; break; }
    if (ref.length === 0) continue;
    n++;
    if (Math.abs(lp[0]) < 0.5 && Math.abs(lp[1]) < 0.3 && Math.abs(lp[2]) < 0.4) inside++;
    if (ref[0].d < 0) spec++;
    worst = Math.max(worst, matchSets(got, ref));
  }
  check(`sphere-hull (GJK/EPA) == sphere-box: ${n} contacts (${inside} centre inside, ${spec} spec.)`, worst, 0, 1e-6);
  // capsule vs hull == capsule vs box (deterministic configurations)
  const cb = body(box(1, 0.5, 1)), ch = body(convex(corners(1, 0.5, 1))), cp = body(capsule(0.2, 0.5));
  const capCfgs = [
    ['lying', [0.1, 0.69, 0.2], quat(0, 0, 1, Math.PI / 2)],
    ['standing', [0.3, 1.19, 0], Q0],
    ['tilted 30 deg', [0.2, 0.5 + 0.2 + 0.5 * Math.cos(Math.PI / 6) - 0.01, 0.1], quat(0, 0, 1, Math.PI / 6)],
    ['core inside', [0.3, 0.6, 0], Q0],
    ['beside edge', [1.1, 0.6, 0.05], quat(1, 0.2, 0, Math.PI / 2)],
    ['speculative lying', [0.1, 0.705, 0.2], quat(0, 0, 1, Math.PI / 2)],
  ];
  for (const [name, p, q] of capCfgs) {
    pose(cp, p, q);
    const ref = hit(cp, cb), got = hit(cp, ch);
    check(`capsule-hull (GJK/EPA) == capsule-box (${name}), ${ref.length} pts`, ref.length > 0 ? matchSets(got, ref) : Infinity, 0, 1e-6);
  }
  // hull vs hull: octahedron vertex-down on a box hull; box hull on box hull
  const oct = body(convex([1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1]), [0.1, 0.5 + 1 - 0.01, 0.2]);
  let c = hit(oct, ch);
  check('hull-hull vertex on face: 1 contact depth 0.01, normal (0,-1,0)', c.length === 1 && Math.abs(c[0].d - 0.01) < 1e-9 && dist3(c[0].n, [0, -1, 0]) < 1e-9 && dist3(c[0].p, [0.1, 0.495, 0.2]) < 1e-9);
  pose(oct, [0.1, 0.5 + Math.SQRT1_2 - 0.01, 0.2], quat(0, 0, 1, Math.PI / 4));
  c = hit(oct, ch);
  check('hull-hull edge on face: 2 contacts depth 0.01', c.length === 2 && allDepth(c, 0.01, 1e-9) && allNormal(c, [0, -1, 0], 1e-9));
  pose(H, [0.2, 0.5 + 0.5 - 0.01, 0.1]);
  c = hit(ch, H);
  check('hull-hull face: 4 contacts depth 0.01, normal (0,1,0)', c.length === 4 && allDepth(c, 0.01, 1e-9) && allNormal(c, [0, 1, 0], 1e-9));
  pose(oct, [3, 0.5 + 1 - 0.3, 0.2]); // deep penetration of a vertex beside the box hull: EPA
  pose(ch, [3, 0, 0]);
  c = hit(oct, ch);
  check('hull-hull deep (EPA): 1 contact depth 0.3', c.length === 1 && Math.abs(c[0].d - 0.3) < 1e-6 && dist3(c[0].n, [0, -1, 0]) < 1e-6);
  for (const id of [A, Bb, H, hb, hh, s, cb, ch, cp, oct]) B.free(id);
}

// ---- 8. warm start: stable keys under 1 mm jitter ----------------------------------------------------------------------------
{
  const base = body(box(1, 0.5, 1)), top = body(box(0.4, 0.4, 0.4), [0.1, 0.895, -0.2]);
  const rnd = mulberry32(42);
  np.beginStep(); C.clear(); np.collide(B, base, top, C);
  const keys0 = list().map((c) => c.key).sort();
  for (let i = 0; i < C.count; i++) { C.lambdaN[i] = 1 + i; C.lambdaT[i] = 0.5 * i; }
  let want = new Map(list().map((c) => [c.key, [c.ln, c.lt]]));
  np.endStep(C);
  let sameKeys = true, carried = true, steps = 0;
  for (let k = 0; k < 50; k++) {
    const j = () => (rnd() * 2 - 1) * 0.001;
    pose(top, [0.1 + j(), 0.895 + j(), -0.2 + j()], quat(rnd() - 0.5, 1, rnd() - 0.5, (rnd() * 2 - 1) * 0.001));
    np.beginStep(); C.clear(); np.collide(B, base, top, C);
    const cs = list();
    if (cs.map((c) => c.key).sort().join() !== keys0.join()) sameKeys = false;
    for (const c of cs) { const w = want.get(c.key); if (!w || w[0] !== c.ln || w[1] !== c.lt) carried = false; }
    for (let i = 0; i < C.count; i++) { C.lambdaN[i] = 1 + k + i * 0.25; C.lambdaT[i] = k - i; }
    want = new Map(list().map((c) => [c.key, [c.ln, c.lt]]));
    np.endStep(C);
    steps++;
  }
  check(`warm start: same 4-key set over ${steps} jittered steps (1 mm, 1 mrad)`, sameKeys && keys0.length === 4);
  check('warm start: lambdaN / lambdaT carried endStep -> beginStep -> collide', carried);
  check('warm start: cache holds this step\'s 4 entries', np.cached, 4);
  pose(top, [0.1, 5, -0.2]);
  np.beginStep(); C.clear(); np.collide(B, base, top, C); np.endStep(C); // a step without contact evicts
  pose(top, [0.1, 0.895, -0.2]);
  np.beginStep(); C.clear(); np.collide(B, base, top, C);
  check('warm start: entries evicted after a step without contact', list().every((c) => c.ln === 0 && c.lt === 0) && C.count === 4);
  for (let i = 0; i < C.count; i++) C.lambdaN[i] = 3 + i;
  np.endStep(C);
  const snap = np.serialize(), np2 = new Narrowphase({ margin: MARGIN });
  np2.restore(JSON.parse(JSON.stringify(snap)));
  np2.beginStep(); C.clear(); np2.collide(B, base, top, C);
  check('serialize / restore keeps the warm-start lambdas', list().map((c) => c.ln).sort().join() === '3,4,5,6');
  B.free(base); B.free(top);
}

// ---- 9. heightfield -----------------------------------------------------------------------------------------------------------------
const flat = (x, z) => 0;
const T10 = Math.tan(10 * Math.PI / 180);
const slope = (x, z) => x * T10;
function ground(a, h) { C.clear(); np.collideHeightfield(B, a, h, C); return list(); }
{
  const s = body(sphere(0.5), [1, 0.49, 2]);
  let c = ground(s, flat);
  check('heightfield sphere on y=0: 1 contact', c.length, 1);
  err('heightfield sphere on y=0: depth 0.01', c[0].d, 0.01, 1e-9);
  check('heightfield sphere on y=0: normal (0,-1,0), point (1,-0.005,2)', dist3(c[0].n, [0, -1, 0]) + dist3(c[0].p, [1, -0.005, 2]), 0, 1e-9);
  check('heightfield: b = WORLD_BODY, staticIndex -1, world pairKey', c[0].b === WORLD_BODY && c[0].si === -1 && c[0].pk === 0xffff0000 + s);
  const N = [-Math.sin(10 * Math.PI / 180), Math.cos(10 * Math.PI / 180), 0], G = [2, 2 * T10, -1];
  pose(s, [G[0] + N[0] * 0.49, G[1] + N[1] * 0.49, G[2]]);
  c = ground(s, slope);
  check('heightfield sphere on 10 deg slope: 1 contact', c.length, 1);
  err('heightfield 10 deg slope: depth 0.01', c[0].d, 0.01, 1e-9);
  check('heightfield 10 deg slope: normal = -N (into the slope)', dist3(c[0].n, [-N[0], -N[1], -N[2]]), 0, 1e-9);
  check('heightfield 10 deg slope: point at the contact', dist3(c[0].p, [G[0] - N[0] * 0.005, G[1] - N[1] * 0.005, G[2]]), 0, 1e-9);
  pose(s, [1, 0.515, 2]);
  c = ground(s, flat);
  check('heightfield sphere: speculative -0.015 beyond margin -> none', c.length, 0);
  pose(s, [1, 0.505, 2]);
  c = ground(s, flat);
  check('heightfield sphere: speculative -0.005', c.length === 1 && Math.abs(c[0].d + 0.005) < 1e-9);
  const bx = body(box(0.5, 0.25, 0.5), [0, 0.24, 0]);
  c = ground(bx, flat);
  check('heightfield box: 4 corner contacts depth 0.01', c.length === 4 && allDepth(c, 0.01, 1e-9) && allNormal(c, [0, -1, 0], 1e-12));
  const ang = 10 * Math.PI / 180, qb = quat(0, 0, 1, ang), off = rot(qb, [0, -0.25, 0]);
  pose(bx, [3 - off[0] + N[0] * -0.01, 3 * T10 - off[1] + N[1] * -0.01, 0], qb);
  c = ground(bx, slope);
  check('heightfield box aligned on 10 deg slope: 4 contacts depth 0.01 (1e-9)', c.length === 4 && allDepth(c, 0.01, 1e-9) && allNormal(c, [-N[0], -N[1], -N[2]], 1e-9));
  const cp = body(capsule(0.2, 0.5), [0, 0.19, 0], quat(0, 0, 1, Math.PI / 2));
  c = ground(cp, flat);
  check('heightfield capsule lying: 2 contacts depth 0.01', c.length === 2 && allDepth(c, 0.01, 1e-9));
  const cmp = body(compound([{ shape: sphere(0.3), position: [-1, 0, 0] }, { shape: sphere(0.3), position: [1, 0, 0] }]), [0, 0.29, 0]);
  c = ground(cmp, flat);
  check('heightfield compound (2 spheres): 2 contacts, distinct keys', c.length === 2 && c[0].key !== c[1].key && allDepth(c, 0.01, 1e-9));
  B.free(s); B.free(bx); B.free(cp); B.free(cmp);
}

// ---- 10. static boxes ---------------------------------------------------------------------------------------------------------------
{
  const s = body(sphere(0.5), [0, 1.49, 0]);
  C.clear();
  np.collideStaticBox(B, s, -1, -1, -1, 1, 1, 1, 7, C);
  let c = list();
  check('static box: sphere on top -> 1 contact', c.length, 1);
  err('static box: depth 0.01', c[0].d, 0.01, 1e-9);
  check('static box: normal (0,-1,0) into the box, point (0,0.995,0)', dist3(c[0].n, [0, -1, 0]) + dist3(c[0].p, [0, 0.995, 0]), 0, 1e-9);
  check('static box: b = WORLD_BODY, staticIndex 7, pairKey 0xFFFF0000 + a', c[0].b === WORLD_BODY && c[0].si === 7 && c[0].pk === 0xffff0000 + s);
  const k7 = c[0].key;
  C.clear();
  np.collideStaticBox(B, s, -1, -1, -1, 1, 1, 1, 8, C);
  check('static box: key differs per staticIndex', C.count === 1 && C.key[0] !== k7);
  const bx = body(box(0.3, 0.3, 0.3), [0.2, 1.29, 0.1]);
  C.clear();
  np.collideStaticBox(B, bx, -1, -1, -1, 1, 1, 1, 3, C);
  c = list();
  check('static box: box on top -> 4 contacts depth 0.01, normal (0,-1,0)', c.length === 4 && allDepth(c, 0.01, 1e-9) && allNormal(c, [0, -1, 0], 1e-12));
  const cp = body(capsule(0.2, 0.5), [0.1, 1.19, 0.2], quat(0, 0, 1, Math.PI / 2));
  C.clear();
  np.collideStaticBox(B, cp, -1, -1, -1, 1, 1, 1, 3, C);
  c = list();
  check('static box: capsule lying on top -> 2 contacts depth 0.01', c.length === 2 && allDepth(c, 0.01, 1e-9) && allNormal(c, [0, -1, 0], 1e-12));
  const corners = []; for (let i = 0; i < 8; i++) corners.push(i & 1 ? 0.3 : -0.3, i & 2 ? 0.3 : -0.3, i & 4 ? 0.3 : -0.3);
  const hx = body(convex(corners), [0.2, 1.29, 0.1], quat(0, 1, 0, 0.3));
  C.clear();
  np.collideStaticBox(B, hx, -1, -1, -1, 1, 1, 1, 3, C);
  c = list();
  check('static box: hull resting on top -> 4 contacts depth 0.01', c.length === 4 && allDepth(c, 0.01, 1e-9) && allNormal(c, [0, -1, 0], 1e-9));
  B.free(cp); B.free(hx);
  pose(s, [0, 1.52, 0]);
  C.clear();
  np.collideStaticBox(B, s, -1, -1, -1, 1, 1, 1, 7, C);
  check('static box: beyond margin -> none', C.count, 0);
  B.free(s); B.free(bx);
}

// ---- 11. trimesh -------------------------------------------------------------------------------------------------------------------
function grid(n, size) {
  const pos = [], idx = [];
  for (let j = 0; j <= n; j++) for (let i = 0; i <= n; i++) pos.push(-size / 2 + (i * size) / n, 0, -size / 2 + (j * size) / n);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const a = j * (n + 1) + i, b = a + 1, c = a + n + 1, d = c + 1;
      idx.push(a, c, b, b, c, d); // counter-clockwise seen from +y
    }
  }
  return trimesh(pos, idx);
}
{
  const mesh = body(grid(20, 20)), s = body(sphere(0.5), [0.3, 0.49, 0.6]);
  let c = hit(s, mesh);
  check('trimesh sphere in a triangle: 1 contact', c.length, 1);
  err('trimesh sphere: depth 0.01 (1e-6)', c[0].d, 0.01, 1e-6);
  check('trimesh sphere: normal (0,-1,0), point (0.3,-0.005,0.6)', dist3(c[0].n, [0, -1, 0]) + dist3(c[0].p, [0.3, -0.005, 0.6]), 0, 1e-9);
  pose(s, [2, 0.49, 3]);
  c = hit(s, mesh);
  check('trimesh sphere on a grid vertex: 1 contact (merged)', c.length === 1 && Math.abs(c[0].d - 0.01) < 1e-6);
  pose(s, [2.5, 0.49, 3]);
  c = hit(s, mesh);
  check('trimesh sphere on an edge: 1 contact (merged)', c.length === 1 && Math.abs(c[0].d - 0.01) < 1e-6);
  pose(s, [2.3, 0.49, 3.05]);
  c = hit(s, mesh);
  check('trimesh sphere near an edge: 1 contact (ghost edge removed)', c.length === 1 && Math.abs(c[0].d - 0.01) < 1e-6 && dist3(c[0].n, [0, -1, 0]) < 1e-9);
  pose(s, [0.3, -0.49, 0.6]);
  check('trimesh sphere below the mesh: no back-face contact', hit(s, mesh).length, 0);
  pose(s, [0.3, -0.1, 0.6]);
  check('trimesh sphere centre just below the plane: none', hit(s, mesh).length, 0);
  pose(s, [0.3, 0.505, 0.6]);
  c = hit(s, mesh);
  check('trimesh sphere: speculative -0.005', c.length === 1 && Math.abs(c[0].d + 0.005) < 1e-6);
  pose(s, [0.3, 0.52, 0.6]);
  check('trimesh sphere: beyond margin -> none', hit(s, mesh).length, 0);
  pose(s, [0.3, 0.49, 0.6]);
  c = hit(mesh, s);
  check('trimesh first: normal (0,1,0) mesh->sphere', c.length === 1 && dist3(c[0].n, [0, 1, 0]) < 1e-9);
  const bx = body(box(0.4, 0.25, 0.4), [0.25, 0.24, 0.35]);
  c = hit(bx, mesh);
  check(`trimesh box: ${c.length} contacts (4..8) depth 0.01, normal (0,-1,0)`, c.length >= 4 && c.length <= 8 && allDepth(c, 0.01, 1e-9) && allNormal(c, [0, -1, 0], 1e-9));
  check('trimesh box: the 4 bottom corners are kept', [[-0.15, -0.05], [0.65, -0.05], [0.65, 0.75], [-0.15, 0.75]].every(([x, z]) => hasPoint(c, [x, -0.005, z], 1e-9)));
  const cp = body(capsule(0.2, 0.5), [0.25, 0.19, 0.35], quat(0, 0, 1, Math.PI / 2));
  c = hit(cp, mesh);
  check(`trimesh capsule lying: ${c.length} contacts incl. both ends, depth 0.01`, c.length >= 2 && allDepth(c, 0.01, 1e-9) && hasPoint(c, [-0.25, -0.005, 0.35], 1e-9) && hasPoint(c, [0.75, -0.005, 0.35], 1e-9));
  const hx = body(convex([1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1]), [0.3, 0.99, 0.6]);
  c = hit(hx, mesh);
  check('trimesh hull vertex: 1 contact depth 0.01', c.length === 1 && Math.abs(c[0].d - 0.01) < 1e-9);
  // posed mesh: tilted 10 deg about x and lifted; a sphere along the tilted normal
  const q = quat(1, 0, 0, 10 * Math.PI / 180), nrm = rot(q, [0, 1, 0]), gp = rot(q, [0.3, 0, 0.6]);
  pose(mesh, [5, 1, -2], q);
  pose(s, [5 + gp[0] + nrm[0] * 0.49, 1 + gp[1] + nrm[1] * 0.49, -2 + gp[2] + nrm[2] * 0.49]);
  c = hit(s, mesh);
  check('trimesh posed (10 deg tilt): 1 contact depth 0.01, normal -N', c.length === 1 && Math.abs(c[0].d - 0.01) < 1e-9 && dist3(c[0].n, nrm.map((v) => -v)) < 1e-9);
  const mesh2 = body(grid(2, 2));
  check('trimesh vs trimesh: none', hit(mesh, mesh2).length, 0);
  // sliding sweeps over the flat grid (face interiors, edges, vertices): no ghost or missing contacts anywhere
  pose(mesh, [0, 0, 0]);
  let bad = 0, total = 0;
  for (let i = 0; i <= 400; i++) {
    const x = -1.5 + i * 0.0075, z = 0.37 + i * 0.0049;
    pose(s, [x, 0.49, z]);
    const cs = hit(s, mesh);
    total++;
    if (cs.length !== 1 || Math.abs(cs[0].d - 0.01) > 1e-9 || dist3(cs[0].n, [0, -1, 0]) > 1e-9) bad++;
  }
  pose(s, [1, 0.49, 1]); // exactly on a grid vertex
  if (hit(s, mesh).length !== 1) bad++;
  check(`trimesh sphere sweep: exactly 1 contact (depth 0.01, normal -y) at ${total + 1} positions`, bad, 0);
  bad = 0;
  for (let i = 0; i <= 200; i++) {
    const yaw = i * 0.031;
    pose(bx, [-1 + i * 0.011, 0.24, 0.5 + i * 0.007], quat(0, 1, 0, yaw));
    const cs = hit(bx, mesh);
    if (cs.length < 3 || cs.length > 8 || !allDepth(cs, 0.01, 1e-9) || !allNormal(cs, [0, -1, 0], 1e-9)) bad++;
    pose(cp, [-1 + i * 0.011, 0.19, 0.5 + i * 0.007], qmul(quat(0, 1, 0, yaw), quat(0, 0, 1, Math.PI / 2)));
    const ks = hit(cp, mesh);
    if (ks.length < 2 || !allDepth(ks, 0.01, 1e-9) || !allNormal(ks, [0, -1, 0], 1e-9)) bad++;
  }
  check('trimesh box / capsule sweeps (201 poses, yawing): all contacts depth 0.01, normal -y', bad, 0);
  // bumpy mesh (random heights, slopes up to ~30 deg): every shape type at random poses near the surface
  {
    const n = 16, pos = [], idx = [], rh = mulberry32(5), hgt = [];
    for (let j = 0; j <= n; j++) for (let i = 0; i <= n; i++) { const h = (rh() * 2 - 1) * 0.3; hgt.push(h); pos.push(-8 + i, h, -8 + j); }
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) { const a = j * (n + 1) + i, b2 = a + 1, c2 = a + n + 1, d = c2 + 1; idx.push(a, c2, b2, b2, c2, d); }
    const bumpy = body(trimesh(pos, idx), [100, 0, 0]);
    const cs4 = []; for (let i = 0; i < 8; i++) cs4.push(i & 1 ? 0.3 : -0.3, i & 2 ? 0.2 : -0.2, i & 4 ? 0.25 : -0.25);
    const kinds = [sphere(0.3), capsule(0.2, 0.4), box(0.3, 0.2, 0.25), convex(cs4), compound([{ shape: sphere(0.2), position: [-0.3, 0, 0] }, { shape: box(0.15, 0.15, 0.15), position: [0.3, 0, 0] }])];
    const r = mulberry32(17);
    let nBad = 0, nC = 0, nPairs = 0, nUp = 0;
    for (let k = 0; k < 500; k++) {
      const x = (r() * 2 - 1) * 6, z = (r() * 2 - 1) * 6;
      const id = body(kinds[k % kinds.length], [100 + x, 0.1 + r() * 0.4, z], quat(r() - 0.5, r() - 0.5, r() - 0.5, r() * 6));
      const cs = hit(id, bumpy);
      if (cs.length) nPairs++;
      nC += cs.length;
      for (const c of cs) {
        if (!(Math.abs(nlen(c.n) - 1) < 1e-9) || !(c.n[1] < 0.5) || !(c.d > -MARGIN - 1e-12 && c.d < 1)) nBad++;
        if (c.n[1] >= 0) nUp++; // a body spawned below nearby terrain can touch an edge above its centre
      }
      if (cs.length > 8) nBad++;
      B.free(id);
    }
    check(`trimesh bumpy: ${nC} contacts on ${nPairs} of 500 random bodies: unit normals, sane depths, <= 8 per pair`, nBad, 0);
    check('trimesh bumpy: normals point into the mesh (n.y < 0) for > 99 %', nUp / nC, 0, 0.01);
    B.free(bumpy);
  }
  for (const id of [mesh, s, bx, cp, hx, mesh2]) B.free(id);
}

// ---- 12. compounds -----------------------------------------------------------------------------------------------------------------
{
  const dumb = compound([{ shape: sphere(0.3), position: [-0.5, 0, 0] }, { shape: sphere(0.3), position: [0.5, 0, 0] }]);
  const cmp = body(dumb, [0, 0.79, 0]), bx = body(box(1, 0.5, 1));
  let c = hit(cmp, bx);
  check('compound (2 spheres) on box: 2 contacts depth 0.01, distinct keys', c.length === 2 && allDepth(c, 0.01, 1e-9) && c[0].key !== c[1].key && allNormal(c, [0, -1, 0], 1e-12));
  pose(cmp, [0, 0.79, 0], quat(0, 1, 0, Math.PI / 2));
  c = hit(bx, cmp);
  check('box vs rotated compound: points at z = +-0.5, normal (0,1,0)', c.length === 2 && hasPoint(c, [0, 0.495, 0.5], 1e-9) && hasPoint(c, [0, 0.495, -0.5], 1e-9) && allNormal(c, [0, 1, 0], 1e-12));
  // a box + capsule compound against the dumbbell: only the left sphere touches the capsule child, from the side
  const cmp2 = body(compound([{ shape: box(0.2, 0.2, 0.2), position: [0, 0, 0] }, { shape: capsule(0.1, 0.3), position: [0, 0.5, 0] }]), [10, 0, 0]);
  pose(cmp, [10 + 0.5 + 0.39, 0.5, 0]);
  c = hit(cmp2, cmp);
  check('compound vs compound: 1 contact (capsule child vs left sphere)', c.length === 1 && Math.abs(c[0].d - 0.01) < 1e-9 && dist3(c[0].n, [1, 0, 0]) < 1e-9);
  pose(cmp, [10 + 0.5 + 0.39, 0.5, 0], quat(0, 1, 0, 1e-3));
  const c2 = hit(cmp2, cmp);
  check('compound vs compound: key stable under a small rotation', c2.length === 1 && c2[0].key === c[0].key);
  const mesh = body(grid(4, 8));
  pose(cmp, [0.3, 0.29, 0.2]);
  c = hit(cmp, mesh);
  check('compound vs trimesh: 2 contacts depth 0.01', c.length === 2 && allDepth(c, 0.01, 1e-6));
  for (const id of [cmp, bx, cmp2, mesh]) B.free(id);
}

// ---- 12b. review regressions ------------------------------------------------------------------------------------------------
{
  // (1) mesh overlap resolved by EPA: a step (top y = 0.3, riser at x = 0 facing +x, floor y = 0), shared vertices
  const h = 0.3;
  const stepPos = [-2, h, -2, 0, h, -2, -2, h, 2, 0, h, 2, 0, 0, -2, 0, 0, 2, 2, 0, -2, 2, 0, 2];
  const stepIdx = [0, 2, 1, 1, 2, 3, 4, 1, 5, 5, 1, 3, 4, 5, 6, 6, 5, 7];
  const step = body(trimesh(stepPos, stepIdx));
  const bx = body(box(0.5, 0.2, 0.5));
  let onlyTop = true, cnt = 0;
  for (const off of [0, 0.05, 0.1, 0.15, 0.2, 0.25]) { // centre 0 .. 25 cm past the edge (0.1: 60 % overhang)
    pose(bx, [off, h + 0.2 - 0.005, 0.1]);
    const c = hit(bx, step);
    cnt += c.length;
    if (!c.length || !c.every((x) => dist3(x.n, [0, -1, 0]) < 1e-9 && Math.abs(x.d - 0.005) < 1e-9 && x.p[0] < 1e-9)) onlyTop = false;
  }
  check(`trimesh step: box overhanging the edge by 0..25 cm -> only top-face contacts (${cnt}), depth 5 mm`, onlyTop);
  pose(bx, [0.1, h + 0.2 - 0.005, 0.1], quat(0, 1, 0, 0.3));
  check('trimesh step: yawed overhanging box -> only top-face contacts', hit(bx, step).every((x) => dist3(x.n, [0, -1, 0]) < 1e-9 && x.d < 0.006));
  const ridge = body(trimesh([-1, -1, -2, -1, -1, 2, 0, 0, -2, 0, 0, 2, 1, -1, -2, 1, -1, 2], [0, 1, 2, 2, 1, 3, 2, 3, 4, 4, 3, 5]));
  pose(bx, [0, 0.2 - 0.005, 0.1]);
  let c = hit(bx, ridge);
  check(`trimesh 90 deg ridge: box straddling it -> ${c.length} contacts, all shallower than 2 cm`, c.length > 0 && c.every((x) => x.d < 0.02 && x.d > -MARGIN && Math.abs(nlen(x.n) - 1) < 1e-9));
  // (6) mesh edge keys from the sorted vertex pair: a sphere on the ridge apex, jittered across it
  const s = body(sphere(0.4));
  const keys = new Set();
  const rnd = mulberry32(3);
  let one = true;
  for (let k = 0; k < 20; k++) {
    pose(s, [(rnd() * 2 - 1) * 0.001, 0.395 + (rnd() * 2 - 1) * 0.0005, 0.3 + (rnd() * 2 - 1) * 0.001]);
    c = hit(s, ridge);
    if (c.length !== 1) one = false;
    for (const x of c) keys.add(x.key);
  }
  check('trimesh ridge apex: 1 edge contact, same key under 1 mm jitter (20 poses)', one && keys.size === 1);
  // (7) separated corner-to-corner boxes: empty face clip -> exact GJK point
  const b1 = body(box(0.5, 0.5, 0.5), [20, 0, 0]), b2 = body(box(0.5, 0.5, 0.5), [21.003, 1.003, 0]);
  c = hit(b1, b2);
  check('box-box corner-to-corner 4.2 mm apart: 1 speculative contact via GJK', c.length === 1 && Math.abs(c[0].d + 0.003 * Math.SQRT2) < 1e-9 && dist3(c[0].n, [Math.SQRT1_2, Math.SQRT1_2, 0]) < 1e-9);
  // (7) compound nested deeper than the pose stack: capped, counted, no throw
  let deep = sphere(0.3);
  for (let k = 0; k < 70; k++) deep = compound([{ shape: deep }]);
  const dp = body(deep, [30, 0.29, 0]), fl = body(box(5, 0.5, 5), [30, -0.5, 0]);
  np.beginStep();
  C.clear(); np.collide(B, dp, fl, C);
  check('compound nested 70 deep: no throw, stats.capped counts the pose-stack limit', np.stats.capped, 1, Infinity);
  for (const id of [step, bx, ridge, s, b1, b2, dp, fl]) B.free(id);
}
{
  // (2) direct SoA writes: one overflow count per dropped contact, stale lambdas overwritten
  const small = new ContactBuffer(3), np3 = new Narrowphase({ margin: MARGIN, maxContacts: 100 });
  const a = body(box(0.5, 0.5, 0.5)), b = body(box(0.5, 0.5, 0.5), [0.2, 0.99, 0.1]);
  small.lambdaN.fill(NaN); small.lambdaT.fill(NaN);
  const n = np3.collide(B, a, b, small);
  check('flush: buffer of 3 gets 3 of 4 contacts, overflow 1, lambdas 0', n === 3 && small.count === 3 && small.overflow === 1 && [0, 1, 2].every((i) => small.lambdaN[i] === 0 && small.lambdaT[i] === 0));
  check('flush: key / pairKey stored as uint32', small.pairKey[0] === pairKey(a, b) && small.key[0] >= 0);
  // (3) timing off by default
  check('timing defaults to false (stats.ms stays 0)', np3.timing === false && np3.stats.ms === 0);
  // (4) evictBody: a small, crowded table; the evicted body's entries go, the others survive, a reused id starts cold
  const npE = new Narrowphase({ margin: MARGIN, tableSize: 32 });
  const base = body(box(3, 0.5, 3)), tops = [];
  for (let k = 0; k < 3; k++) tops.push(body(box(0.3, 0.3, 0.3), [-2 + k * 2, 0.795, 0]));
  npE.beginStep(); C.clear();
  for (const t of tops) npE.collide(B, base, t, C);
  for (let i = 0; i < C.count; i++) { C.lambdaN[i] = 1 + i; C.lambdaT[i] = 0.5 + i; }
  npE.endStep(C);
  const victim = tops[1], dropped = npE.evictBody(victim);
  npE.beginStep(); C.clear();
  for (const t of tops) if (t !== victim) npE.collide(B, base, t, C);
  const survivors = list();
  B.free(victim);
  const again = body(box(0.3, 0.3, 0.3), [0, 0.795, 0]);
  const reused = again === victim;
  C.clear(); npE.collide(B, base, again, C);
  const fresh = list();
  check(`evictBody: ${dropped} entries dropped, the other 8 still warm, reused id ${reused ? 'cold' : '?'}`,
    dropped === 4 && reused && survivors.length === 8 && survivors.every((x) => x.ln > 0) && fresh.length === 4 && fresh.every((x) => x.ln === 0 && x.lt === 0));
  check('evictBody: cache holds the 8 remaining entries', npE.cached, 8);
  for (const id of [a, b, base, again, ...tops.filter((t) => t !== victim)]) B.free(id);
}
{
  // (5) heightfield stencil: 3 heightAt calls per sample, the centre reused under a sphere; hfStep / hfCell option
  let calls = 0, maxOff = 0;
  const counting = (x, z) => { calls++; maxOff = Math.max(maxOff, Math.abs(x - Math.round(x * 4) / 4)); return 0; };
  const s = body(sphere(0.5), [1, 0.49, 2]), bx = body(box(0.5, 0.25, 0.5), [4, 0.24, 0]);
  C.clear(); calls = 0; np.collideHeightfield(B, s, counting, C);
  check('heightfield sphere on flat ground: 3 heightAt calls (centre reused)', calls, 3);
  C.clear(); calls = 0; np.collideHeightfield(B, bx, counting, C);
  check('heightfield box: 3 + 3 x 4 heightAt calls (only vertices below the centre plane)', calls === 15 && C.count === 4);
  const npc = new Narrowphase({ margin: MARGIN, hfCell: 2 });
  let steps = 0;
  const probe = (x, z) => { if (x !== 1 && x - 1 > steps) steps = x - 1; return 0; };
  C.clear(); npc.collideHeightfield(B, s, probe, C);
  check('heightfield hfCell: 2 -> finite differences over 2 m', npc.hfStep === 2 && steps === 2);
  B.free(s); B.free(bx);
}

// ---- 13. edge cases: exact touching (EPA blow-up), coincident centres, intersecting axes ---------------------------------------
{
  const corners = []; for (let i = 0; i < 8; i++) corners.push(i & 1 ? 0.5 : -0.5, i & 2 ? 0.5 : -0.5, i & 4 ? 0.5 : -0.5);
  const h1 = body(convex(corners)), h2 = body(convex(corners), [0.2, 1, 0.1]);
  let c = hit(h1, h2);
  check('hull-hull exactly touching: 4 contacts depth 0', c.length === 4 && allDepth(c, 0, 1e-9) && allNormal(c, [0, 1, 0], 1e-9));
  pose(h2, [0, 1, 0]);
  c = hit(h1, h2);
  check('hull-hull exactly coincident faces: 4 contacts depth 0', c.length === 4 && allDepth(c, 0, 1e-9) && allNormal(c, [0, 1, 0], 1e-9));
  const s = body(sphere(0.25), [0.1, 0.5, 0.2]);
  c = hit(s, h1);
  check('sphere centre on a hull face: depth = r, normal (0,-1,0)', c.length === 1 && Math.abs(c[0].d - 0.25) < 1e-9 && dist3(c[0].n, [0, -1, 0]) < 1e-9);
  pose(s, [0, 0, 0]);
  c = hit(s, h1);
  check('sphere centre at the hull centre: depth = r + 0.5, unit normal', c.length === 1 && Math.abs(c[0].d - 0.75) < 1e-9 && Math.abs(nlen(c[0].n) - 1) < 1e-12);
  const s2 = body(sphere(0.5));
  c = hit(s, s2);
  check('sphere-sphere coincident centres: depth = r1 + r2, unit normal', c.length === 1 && Math.abs(c[0].d - 0.75) < 1e-12 && Math.abs(nlen(c[0].n) - 1) < 1e-12);
  const k1 = body(capsule(0.3, 1)), k2 = body(capsule(0.3, 1), [0, 0, 0], quat(0, 0, 1, Math.PI / 2));
  c = hit(k1, k2);
  check('capsule-capsule intersecting axes: depth 0.6, normal _|_ both axes', c.length === 1 && Math.abs(c[0].d - 0.6) < 1e-12 && Math.abs(Math.abs(c[0].n[2]) - 1) < 1e-12);
  const bx = body(box(0.5, 0.5, 0.5)), cp = body(capsule(0.2, 0.3));
  c = hit(cp, bx);
  check('capsule core fully inside a box: 1 contact, finite, unit normal', c.length >= 1 && c.every((x) => Number.isFinite(x.d) && Math.abs(nlen(x.n) - 1) < 1e-12));
  for (const id of [h1, h2, s, s2, k1, k2, bx, cp]) B.free(id);
}

// ---- 14. determinism: a random scene twice, bit-identical contacts --------------------------------------------------------------
{
  const corners = []; for (let i = 0; i < 8; i++) corners.push(i & 1 ? 0.4 : -0.4, i & 2 ? 0.3 : -0.3, i & 4 ? 0.5 : -0.5);
  const shapes = [sphere(0.4), capsule(0.25, 0.5), box(0.4, 0.3, 0.5), convex(corners), convex([1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1]),
    compound([{ shape: sphere(0.3), position: [-0.4, 0, 0] }, { shape: box(0.2, 0.2, 0.2), position: [0.4, 0, 0] }])];
  const mesh = body(grid(8, 8), [0, -0.2, 0]);
  const ids = [];
  const rnd = mulberry32(99);
  for (let i = 0; i < 40; i++) {
    ids.push(body(shapes[i % shapes.length], [(rnd() * 2 - 1) * 2, rnd() * 1.2, (rnd() * 2 - 1) * 2], quat(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5, rnd() * 6)));
  }
  const pass = () => {
    const n2 = new Narrowphase({ margin: MARGIN });
    C.clear();
    for (let i = 0; i < ids.length; i++) {
      for (let j = i + 1; j < ids.length; j++) n2.collide(B, ids[i], ids[j], C);
      n2.collide(B, ids[i], mesh, C);
      n2.collideStaticBox(B, ids[i], -3, -2, -3, 3, -0.1, 3, 5, C);
      n2.collideHeightfield(B, ids[i], (x, z) => 0.1 * x - 0.05 * z, C);
    }
    let h = hashF64Array(C.px, 2166136261, 0, C.count);
    for (const a of [C.py, C.pz, C.nx, C.ny, C.nz, C.depth, C.key, C.pairKey, C.a, C.b]) h = hashF64Array(a, h, 0, C.count);
    let bad = 0;
    for (let i = 0; i < C.count; i++) if (!(Math.abs(Math.hypot(C.nx[i], C.ny[i], C.nz[i]) - 1) < 1e-9) || !Number.isFinite(C.depth[i]) || C.depth[i] < -MARGIN - 1e-12) bad++;
    return [h, C.count, bad, n2.stats];
  };
  const [h1, n1, bad1, st] = pass(), [h2, n2c] = pass();
  check(`determinism: random scene (${n1} contacts, ${st.gjk} GJK, ${st.epa} EPA, ${st.sat} SAT) twice, same hash`, h1 === h2 && n1 === n2c && n1 > 0);
  check('random scene: unit normals, finite depths >= -margin', bad1, 0);
  for (const id of ids) B.free(id);
  B.free(mesh);
}

// ---- 15. allocation and timing per pair type --------------------------------------------------------------------------------------
{
  const corners = []; for (let i = 0; i < 8; i++) corners.push(i & 1 ? 0.5 : -0.5, i & 2 ? 0.5 : -0.5, i & 4 ? 0.5 : -0.5);
  const big = body(box(1, 0.5, 1)), hull = body(convex(corners), [0, 0, 0]), mesh = body(grid(20, 20), [0, -10, 0]);
  const pairs = [
    ['sphere-sphere', body(sphere(0.5), [20, 0, 0]), body(sphere(0.5), [20.9, 0, 0])],
    ['sphere-capsule', body(sphere(0.5), [30.7, 0.5, 0]), body(capsule(0.3, 1), [30, 0, 0])],
    ['capsule-capsule', body(capsule(0.3, 1), [40, 0, 0]), body(capsule(0.3, 1), [40.55, 0.5, 0])],
    ['sphere-box', body(sphere(0.5), [0.2, 0.99, 0.1]), big],
    ['capsule-box', body(capsule(0.2, 0.5), [0.1, 0.69, 0.2], quat(0, 0, 1, Math.PI / 2)), big],
    ['box-box face', body(box(0.4, 0.4, 0.4), [0.1, 0.895, -0.2]), big],
    ['box-box edge', body(box(0.5, 0.5, 0.5), [50, 0, 0], quat(0, 0, 1, Math.PI / 4)), body(box(0.5, 0.5, 0.5), [50, Math.SQRT2 - 0.01, 0], quat(1, 0, 0, Math.PI / 4))],
    ['sphere-hull (GJK)', body(sphere(0.5), [0.2, 0.99, 0.1]), hull],
    ['sphere-hull (EPA)', body(sphere(0.5), [0.1, 0.3, 0]), hull],
    ['capsule-hull', body(capsule(0.2, 0.5), [0.1, 0.69, 0.2], quat(0, 0, 1, Math.PI / 2)), hull],
    ['box-hull face', body(box(0.4, 0.4, 0.4), [0.1, 0.895, -0.2]), hull],
    ['hull-hull face', body(convex(corners), [0.2, 0.99, 0.1]), hull],
    ['sphere-trimesh', body(sphere(0.5), [0.3, -9.51, 0.6]), mesh],
    ['box-trimesh', body(box(0.4, 0.25, 0.4), [0.25, -9.76, 0.35]), mesh],
    ['compound-box', body(compound([{ shape: sphere(0.3), position: [-0.5, 0, 0] }, { shape: box(0.2, 0.2, 0.2), position: [0.5, 0, 0] }]), [0, 0.79, 0]), big],
  ];
  const sb = body(box(0.3, 0.3, 0.3), [60.2, 1.29, 0.1]), hf = body(box(0.3, 0.3, 0.3), [70, 0.29, 0]);
  const tests = pairs.map(([name, a, b]) => [name, () => np.collide(B, a, b, C)]);
  tests.push(['static box (box)', () => np.collideStaticBox(B, sb, 59, -1, -1, 61, 1, 1, 3, C)]);
  tests.push(['heightfield (box)', () => np.collideHeightfield(B, hf, flat, C)]);
  // Every call site sees every shape type before anything is measured (warm-up interleaved, so the optimised code is
  // the final polymorphic one); then blocks of N calls per type, heapUsed before / after (byte-exact while no GC runs:
  // a block with a GC is discarded). process.memoryUsage() itself costs ~256 B; one boxed double per call would be
  // 16 B x N. (v8 total_allocated_bytes is too coarse here: it moves in ~256 KB allocation-buffer steps.)
  const N = 10000;
  const loop = (fn, n) => { for (let i = 0; i < n; i++) { C.clear(); fn(); } };
  for (let r = 0; r < 4; r++) for (const [, fn] of tests) loop(fn, 5000);
  performance.now(); process.memoryUsage();
  const lines = [];
  for (const [name, fn] of tests) {
    C.clear(); fn();
    const pts = C.count;
    let minB = Infinity, minMs = Infinity;
    for (let k = 0; k < 5; k++) {
      const prof = new v8.GCProfiler();
      prof.start();
      const h0 = process.memoryUsage().heapUsed, t0 = performance.now();
      loop(fn, N);
      const ms = performance.now() - t0, h1 = process.memoryUsage().heapUsed;
      const gcs = prof.stop().statistics.length;
      if (ms < minMs) minMs = ms;
      if (gcs === 0 && h1 - h0 < minB) minB = h1 - h0;
    }
    const heap = Number.isFinite(minB) ? `${String(minB).padStart(8)} B / ${N} calls` : '   (GC in every block: the heightAt(x, z, y) callback receives boxed doubles)';
    lines.push(`      ${name.padEnd(20)} ${String(pts).padStart(2)} pts  ${(minMs * 1000 / N).toFixed(2).padStart(7)} us/call  ${heap}`);
    if (name.startsWith('heightfield')) continue;
    check(`alloc: ${name}, heap growth over ${N} calls`, minB, 0, 1024, 'B');
  }
  results.push('INFO  per pair type (after interleaved warm-up; best of 5 blocks):\n' + lines.join('\n'));
}

console.log(results.join('\n'));
console.log(failures ? `\n${failures} check(s) FAILED` : '\nAll narrowphase checks passed.');
process.exit(failures ? 1 : 0);
