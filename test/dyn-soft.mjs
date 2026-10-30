// Headless checks for the tetrahedral soft bodies (src/physics/dynamics/softbody.js) on the dynamics world's particle
// store: lattice topology (tetrahedralize), volume preservation of a resting / squeezed cube, compliance and pressure,
// barycentric skinning, determinism across a serialize / from round trip, allocation and step cost. 60 Hz frames, the
// world's 4 substeps, gravity 9.81. Run: node test/dyn-soft.mjs (imports the modules by path).
import v8 from 'node:v8';
import vm from 'node:vm';
import { DynamicsWorld } from '../src/physics/dynamics/DynamicsWorld.js';
import { SoftBody, VolumeConstraints, tetrahedralize } from '../src/physics/dynamics/softbody.js';
import { hashF64Array } from '../src/core/mathDet.js';

const DT = 1 / 60;

let failures = 0;
const results = [];
function check(name, value, lo = 1, hi = lo, unit = '') {
  if (typeof value === 'boolean') value = value ? 1 : 0;
  const ok = Number.isFinite(value) && value >= lo && value <= hi;
  if (!ok) failures++;
  results.push(`${ok ? 'PASS' : 'FAIL'}  ${name.padEnd(64)} ${fmt(value)} ${unit}   [${fmt(lo).trim()}${hi !== lo ? ' .. ' + fmt(hi).trim() : ''}]`);
  return ok;
}
const fmt = (x) => (!Number.isFinite(x) ? String(x) : Number.isInteger(x) ? String(x) : Math.abs(x) < 1e-3 && x !== 0 ? x.toExponential(2) : x.toFixed(4)).padStart(11);
const note = (s) => results.push(`NOTE  ${s}`);

// ---- helpers --------------------------------------------------------------------------------------------------------
function signedVolume(P, a, b, c, d) {
  const ax = P[a * 3], ay = P[a * 3 + 1], az = P[a * 3 + 2];
  const ux = P[b * 3] - ax, uy = P[b * 3 + 1] - ay, uz = P[b * 3 + 2] - az;
  const vx = P[c * 3] - ax, vy = P[c * 3 + 1] - ay, vz = P[c * 3 + 2] - az;
  const wx = P[d * 3] - ax, wy = P[d * 3 + 1] - ay, wz = P[d * 3 + 2] - az;
  return (ux * (vy * wz - vz * wy) + uy * (vz * wx - vx * wz) + uz * (vx * wy - vy * wx)) / 6;
}
/** Barycentric coordinates of p in tet t of a topology (local points). */
function baryOf(T, t, p) {
  const A = T.points, q = [0, 1, 2, 3].map((k) => T.tets[t * 4 + k]);
  const v = q.map((i) => [A[i * 3], A[i * 3 + 1], A[i * 3 + 2]]);
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const det3 = (a, b, c) => a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
  const e1 = sub(v[1], v[0]), e2 = sub(v[2], v[0]), e3 = sub(v[3], v[0]), r = sub(p, v[0]), D = det3(e1, e2, e3);
  const l1 = det3(r, e2, e3) / D, l2 = det3(e1, r, e3) / D, l3 = det3(e1, e2, r) / D;
  return [1 - l1 - l2 - l3, l1, l2, l3];
}
const minBary = (T, t, p) => Math.min(...baryOf(T, t, p));
const maxAbsBary = (T, t, p) => Math.max(...baryOf(T, t, p).map(Math.abs));
/** Distance from p to triangle (a, b, c) of a packed point array: the plane projection when inside, else the edges. */
function pointTriangleDistance(p, A, a, b, c) {
  const P = [a, b, c].map((i) => [A[i * 3], A[i * 3 + 1], A[i * 3 + 2]]);
  const sub = (u, v) => [u[0] - v[0], u[1] - v[1], u[2] - v[2]], dot = (u, v) => u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
  const cross = (u, v) => [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
  const n = cross(sub(P[1], P[0]), sub(P[2], P[0])), nn = dot(n, n), h = dot(sub(p, P[0]), n) / nn;
  const q = [p[0] - h * n[0], p[1] - h * n[1], p[2] - h * n[2]];
  let inside = true;
  for (let k = 0; k < 3; k++) if (dot(cross(sub(P[(k + 1) % 3], P[k]), sub(q, P[k])), n) < 0) inside = false;
  if (inside) return Math.abs(h) * Math.sqrt(nn);
  let best = Infinity;
  for (let k = 0; k < 3; k++) {
    const u = P[k], d = sub(P[(k + 1) % 3], u), s = Math.min(1, Math.max(0, dot(sub(p, u), d) / dot(d, d)));
    best = Math.min(best, Math.hypot(p[0] - u[0] - s * d[0], p[1] - u[1] - s * d[1], p[2] - u[2] - s * d[2]));
  }
  return best;
}
/** Topology checks on (points, tets, surface): orientation, volume, outward normals, closed + consistently wound. */
function meshStats(P, tets, m, surface, k, cx = 0, cy = 0, cz = 0) {
  let minVol = Infinity, vol = 0;
  for (let t = 0; t < m; t++) {
    const v = signedVolume(P, tets[t * 4], tets[t * 4 + 1], tets[t * 4 + 2], tets[t * 4 + 3]);
    minVol = Math.min(minVol, v); vol += v;
  }
  let minOut = Infinity;
  const directed = new Map();
  for (let f = 0; f < k; f++) {
    const a = surface[f * 3], b = surface[f * 3 + 1], c = surface[f * 3 + 2];
    const ux = P[b * 3] - P[a * 3], uy = P[b * 3 + 1] - P[a * 3 + 1], uz = P[b * 3 + 2] - P[a * 3 + 2];
    const vx = P[c * 3] - P[a * 3], vy = P[c * 3 + 1] - P[a * 3 + 1], vz = P[c * 3 + 2] - P[a * 3 + 2];
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const mx = (P[a * 3] + P[b * 3] + P[c * 3]) / 3 - cx, my = (P[a * 3 + 1] + P[b * 3 + 1] + P[c * 3 + 1]) / 3 - cy;
    const mz = (P[a * 3 + 2] + P[b * 3 + 2] + P[c * 3 + 2]) / 3 - cz;
    minOut = Math.min(minOut, (nx * mx + ny * my + nz * mz) / Math.hypot(nx, ny, nz) / Math.hypot(mx, my, mz));
    for (const [p, q] of [[a, b], [b, c], [c, a]]) directed.set(`${p},${q}`, (directed.get(`${p},${q}`) || 0) + 1);
  }
  // closed: every undirected surface edge in exactly 2 triangles; consistent: once in each direction
  const und = new Map();
  let badDir = 0;
  for (const [key, n] of directed) {
    const [p, q] = key.split(',').map(Number);
    und.set(p < q ? `${p},${q}` : `${q},${p}`, (und.get(p < q ? `${p},${q}` : `${q},${p}`) || 0) + n);
    if (n !== 1 || directed.get(`${q},${p}`) !== 1) badDir++;
  }
  let badShare = 0;
  for (const n of und.values()) if (n !== 2) badShare++;
  const sv = new Set(surface.slice(0, 3 * k));
  return { minVol, vol, minOut, badShare, badDir, surfEdges: und.size, surfVerts: sv.size };
}
/** Faces of the tets: how many distinct faces, how many shared by more than 2 tets. */
function faceStats(tets, m) {
  const cnt = new Map();
  for (let t = 0; t < m; t++) {
    const q = [tets[t * 4], tets[t * 4 + 1], tets[t * 4 + 2], tets[t * 4 + 3]];
    for (let s = 0; s < 4; s++) {
      const f = q.filter((_, i) => i !== s).sort((x, y) => x - y).join(',');
      cnt.set(f, (cnt.get(f) || 0) + 1);
    }
  }
  let over = 0, once = 0;
  for (const n of cnt.values()) { if (n > 2) over++; if (n === 1) once++; }
  return { distinct: cnt.size, over, once };
}
/** Structural equality of plain objects, arrays and typed arrays (same type and elements). */
function deepEqual(a, b) {
  if (a === b) return true;
  if (a === null || b === null || typeof a !== 'object' || typeof b !== 'object') return Object.is(a, b);
  if (ArrayBuffer.isView(a)) return ArrayBuffer.isView(b) && a.constructor === b.constructor && a.length === b.length && a.every((v, i) => Object.is(v, b[i]));
  if (Array.isArray(a) !== Array.isArray(b)) return false;
  const ka = Object.keys(a), kb = Object.keys(b);
  return ka.length === kb.length && ka.every((k) => deepEqual(a[k], b[k]));
}
function maxSpeed(w, soft) {
  const v = w.particles.vel;
  let m = 0;
  for (let i = 0; i < soft.count; i++) { const p = soft.ids[i] * 3; m = Math.max(m, Math.hypot(v[p], v[p + 1], v[p + 2])); }
  return m;
}
function finite(w) {
  const P = w.particles;
  for (let i = 0; i < P.pos.length; i++) if (!Number.isFinite(P.pos[i]) || !Number.isFinite(P.vel[i])) return false;
  return true;
}
/** Axis extents of the soft body's particles: [minX, minY, minZ, maxX, maxY, maxZ]. */
function extents(w, soft) {
  const x = w.particles.pos, e = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
  for (let i = 0; i < soft.count; i++) {
    const p = soft.ids[i] * 3;
    for (let k = 0; k < 3; k++) { e[k] = Math.min(e[k], x[p + k]); e[k + 3] = Math.max(e[k + 3], x[p + k]); }
  }
  return e;
}
/** Pin the particles of `soft` whose rest (local) y is at the bottom; returns their ids. */
function pinLayer(w, soft, yLocal) {
  const out = [], L = soft.topology.points;
  for (let i = 0; i < soft.count; i++) if (Math.abs(L[i * 3 + 1] - yLocal) < 1e-9) { w.particles.setPinned(soft.ids[i], true); out.push(soft.ids[i]); }
  return out;
}
/** Move pinned particles to y = y0 + dy (pos and prevPos, zero velocity). */
function placeY(w, ids, y) {
  const P = w.particles;
  for (const id of ids) { P.pos[id * 3 + 1] = y; P.prevPos[id * 3 + 1] = y; P.vel[id * 3] = 0; P.vel[id * 3 + 1] = 0; P.vel[id * 3 + 2] = 0; }
}
/** Relative deviation of each tet's volume from its rest: max |V / V0 - 1|. */
function maxTetDeviation(w, soft) {
  const x = w.particles.pos, T = soft.tets, L = soft.topology.points, lt = soft.topology.tets;
  let m = 0;
  for (let t = 0; t < soft.tetCount; t++) {
    const v = signedVolume(x, T[t * 4], T[t * 4 + 1], T[t * 4 + 2], T[t * 4 + 3]);
    const v0 = signedVolume(L, lt[t * 4], lt[t * 4 + 1], lt[t * 4 + 2], lt[t * 4 + 3]);
    m = Math.max(m, Math.abs(v / v0 - 1));
  }
  return m;
}
/**
 * Squeeze a 1 m cube (4 x 4 x 4 cells, resting at y in [0, 1]): bottom layer pinned, top layer pinned and pressed
 * down by 20 % over 0.5 s, then held for 2.5 s. Returns the world, the body and the volume record.
 */
function squeeze(opts) {
  const w = new DynamicsWorld({ maxParticles: 512, particleDamping: 0.01 });
  const s = SoftBody.lattice(w, { size: [1, 1, 1], resolution: 4, position: [0, 0.5, 0], mass: 10, ...opts });
  pinLayer(w, s, -0.5);
  const top = pinLayer(w, s, 0.5);
  let maxDev = 0, minRatio = Infinity;
  const ramp = 30, hold = 150;
  for (let f = 1; f <= ramp + hold; f++) {
    if (f <= ramp) placeY(w, top, 1 - 0.2 * f / ramp);
    w.step(DT);
    const r = s.volume() / s.restVolume;
    maxDev = Math.max(maxDev, Math.abs(r - 1)); minRatio = Math.min(minRatio, r);
  }
  return { w, s, maxDev, minRatio, end: s.volume() / s.restVolume };
}

// ================================================================================== 1. tetrahedralize: 2 x 2 x 2 box
{
  const topo = tetrahedralize([2, 2, 2], null, [2, 2, 2]);
  const { points, tets, tetCount, surface, surfaceCount, edges, edgeCount } = topo;
  check('box 2x2x2: 27 points', topo.pointCount, 27);
  check('box 2x2x2: 48 tets (6 per cell)', tetCount, 48);
  const st = meshStats(points, tets, tetCount, surface, surfaceCount);
  check('box 2x2x2: every tet positively oriented (min signed volume)', st.minVol, 1 / 6 - 1e-12, 1 / 6 + 1e-12, 'm^3');
  check('box 2x2x2: total volume = 8 (box volume)', st.vol, 8 - 1e-9, 8 + 1e-9, 'm^3');
  check('box 2x2x2: 48 surface triangles (6 faces x 4 cell faces x 2)', surfaceCount, 48);
  check('box 2x2x2: every surface normal points outward (min cos)', st.minOut, 1e-6, 1);
  check('box 2x2x2: each surface edge shared by exactly 2 triangles', st.badShare, 0);
  check('box 2x2x2: surface consistently wound (each directed edge once)', st.badDir, 0);
  check('box 2x2x2: surface Euler characteristic V - E + F = 2', st.surfVerts - st.surfEdges + surfaceCount, 2);
  check('box 2x2x2: 26 surface vertices (all but the centre)', st.surfVerts, 26);
  const fs = faceStats(tets, tetCount);
  check('box 2x2x2: conforming: no face shared by 3+ tets', fs.over, 0);
  check('box 2x2x2: 98 unique edges (54 axis + 36 face + 8 body diagonals)', edgeCount, 98);
  let sortedUnique = 1;
  for (let e = 0; e < edgeCount; e++) {
    if (!(edges[e * 2] < edges[e * 2 + 1])) sortedUnique = 0;
    if (e > 0 && !(edges[e * 2 - 2] < edges[e * 2] || (edges[e * 2 - 2] === edges[e * 2] && edges[e * 2 - 1] < edges[e * 2 + 1]))) sortedUnique = 0;
  }
  check('box 2x2x2: edges a < b, ascending, unique', sortedUnique);
  let lenOk = 1;
  for (let e = 0; e < edgeCount; e++) {
    const a = edges[e * 2], b = edges[e * 2 + 1];
    const l2 = (points[a * 3] - points[b * 3]) ** 2 + (points[a * 3 + 1] - points[b * 3 + 1]) ** 2 + (points[a * 3 + 2] - points[b * 3 + 2]) ** 2;
    if (!(Math.abs(l2 - 1) < 1e-12 || Math.abs(l2 - 2) < 1e-12 || Math.abs(l2 - 3) < 1e-12)) lenOk = 0;
  }
  check('box 2x2x2: edges are cell edges / face / body diagonals only', lenOk);
  const again = tetrahedralize([2, 2, 2], null, [2, 2, 2]);
  check('box 2x2x2: deterministic (second call identical)',
    hashF64Array(Float64Array.from(again.tets)) === hashF64Array(Float64Array.from(tets)) &&
    hashF64Array(Float64Array.from(again.surface)) === hashF64Array(Float64Array.from(surface)) && hashF64Array(again.points) === hashF64Array(points));
  // non-cubic cells and counts
  const nc = tetrahedralize([3, 1, 2], null, [1, 2, 3]);
  const sn = meshStats(nc.points, nc.tets, nc.tetCount, nc.surface, nc.surfaceCount);
  check('box 3x1x2 cells over 1 x 2 x 3 m: total volume 6', sn.vol, 6 - 1e-9, 6 + 1e-9, 'm^3');
  check('box 3x1x2: all tets positive, surface closed and outward', sn.minVol > 0 && sn.badShare === 0 && sn.badDir === 0 && sn.minOut > 0);
  check('box 3x1x2: surface = 2 (3 1 + 3 2 + 1 2) x 2 triangles', nc.surfaceCount, 44);
}

// ================================================================================== 2. sphere lattice
{
  const R = 0.5;
  const topo = tetrahedralize(4, (x, y, z) => x * x + y * y + z * z <= R * R, 2 * R);
  check('sphere r 0.5, 4 cells: corner cells dropped', topo.cellTet[0] === -1 && topo.cellTet[3] === -1 && topo.cellTet[63] === -1);
  check('sphere: 32 cells kept (no corner, no edge cells) -> 192 tets', topo.tetCount, 192);
  const used = new Uint8Array(topo.pointCount);
  for (const p of topo.tets) used[p] = 1;
  check('sphere: points compacted (every point used by a tet)', used.reduce((a, b) => a + b, 0), topo.pointCount);
  check('sphere: 81 points after dropping (27 inner + 6 x 9 outer face points)', topo.pointCount, 81);
  const st = meshStats(topo.points, topo.tets, topo.tetCount, topo.surface, topo.surfaceCount);
  check('sphere: volume = 32 cells x 0.25^3', st.vol, 0.5 - 1e-12, 0.5 + 1e-12, 'm^3');
  check('sphere: surface closed, consistently wound, positive tets', st.badShare === 0 && st.badDir === 0 && st.minVol > 0);
  const big = tetrahedralize(8, (x, y, z) => x * x + y * y + z * z <= R * R, 2 * R);
  const sb = meshStats(big.points, big.tets, big.tetCount, big.surface, big.surfaceCount);
  check('sphere 8 cells: volume within 10 % of 4/3 pi r^3', sb.vol / (4 / 3 * Math.PI * R ** 3), 0.9, 1.1);
  check('sphere 8 cells: surface closed and consistently wound', sb.badShare === 0 && sb.badDir === 0);
}

// ================================================================================== 3. SoftBody.lattice in a world
{
  const w = new DynamicsWorld({ maxParticles: 1024, particleDamping: 0.01 });
  const q = [0.2, -0.3, 0.1, 0.9]; const ql = Math.hypot(...q);
  const s = SoftBody.lattice(w, { size: [1, 1, 1], resolution: 4, position: [1, 2, 3], quaternion: q.map((v) => v / ql), mass: 10 });
  check('lattice 4x4x4: 125 particles', s.count, 125);
  check('lattice: 384 tets, 604 edges, 192 surface triangles', s.tetCount === 384 && s.edgeCount === 604 && s.surfaceCount === 192);
  check('lattice: 98 surface vertices (125 - 27 interior)', s.surfaceVertices.length, 98);
  const P = w.particles;
  let ok = 1;
  const seen = new Set();
  for (const id of s.ids) { if (!P.alive[id] || seen.has(id)) ok = 0; seen.add(id); }
  check('lattice: particle ids alive and unique; first = ids[0]', ok && s.first === s.ids[0]);
  check('lattice: particle mass = 10 / 125, invMass = 12.5', P.mass[s.ids[7]] === 10 / 125 && P.invMass[s.ids[7]] === 12.5);
  check('lattice: particle radius = cell * 0.4', P.radius[s.ids[0]], 0.1 - 1e-12, 0.1 + 1e-12, 'm');
  check('lattice: rest volume = 1 m^3 (rotated)', s.restVolume, 1 - 1e-12, 1 + 1e-12, 'm^3');
  check('lattice: volume() = rest volume at creation', Math.abs(s.volume() - s.restVolume), 0, 1e-12, 'm^3');
  const c = s.centroid([0, 0, 0]);
  check('lattice: centroid = position', Math.hypot(c[0] - 1, c[1] - 2, c[2] - 3), 0, 1e-12, 'm');
  const cv = s.centroid({ x: 0, y: 0, z: 0 });
  check('lattice: centroid({x, y, z}) works', cv.x === c[0] && cv.y === c[1] && cv.z === c[2]);
  const st = meshStats(P.pos, s.tets, s.tetCount, s.surface, s.surfaceCount, 1, 2, 3);
  check('lattice: rotated tets positive, surface outward (particle ids)', st.minVol > 0 && st.minOut > 0 && st.badDir === 0);
  check('lattice: one VolumeConstraints registered (find(world))', s.solver instanceof VolumeConstraints && VolumeConstraints.find(w) === s.solver &&
    w.particleSolvers.filter((x) => x instanceof VolumeConstraints).length === 1);
  check('lattice: 384 tet constraints in the solver', s.solver.size, 384);
  const s2 = SoftBody.lattice(w, { shape: 'sphere', radius: 0.5, resolution: 4, position: [5, 1, 0] });
  check('second body: shares the solver, gets its own group', s2.solver === s.solver && s2.group !== s.group);
  check('sphere body: restVolume 0.5, 81 particles', Math.abs(s2.restVolume - 0.5) < 1e-12 && s2.count === 81);
  check('edges: one distance constraint per edge of both bodies', w.constraints.size, s.edgeCount + s2.edgeCount);
  check('edges: rest length = creation length', Math.abs(w.constraints.rest[s.edgeConstraints[5]] - Math.hypot(
    P.pos[s.edges[10] * 3] - P.pos[s.edges[11] * 3], P.pos[s.edges[10] * 3 + 1] - P.pos[s.edges[11] * 3 + 1],
    P.pos[s.edges[10] * 3 + 2] - P.pos[s.edges[11] * 3 + 2])), 0, 1e-15, 'm');
  s2.dispose();
  let freed = 1;
  for (const id of s2.ids) if (P.alive[id]) freed = 0;
  check('dispose: particles freed, tets removed, solver kept for the cube', freed && s.solver.size === 384 && VolumeConstraints.find(w) === s.solver);
  check('dispose: its edge constraints removed', w.constraints.size, s.edgeCount);
  s.dispose();
  check('dispose of the last body: solver unregistered, no constraint left',
    VolumeConstraints.find(w) === null && s.solver.isEmpty() && !w.particleSolvers.includes(s.solver) && w.constraints.size === 0);
  let threw = false;
  try { SoftBody.lattice(new DynamicsWorld({ maxParticles: 64 }), { resolution: 4 }); } catch (e) { threw = e instanceof RangeError; }
  check('lattice: a full particle store throws RangeError', threw);
}

// ================================================================================= 4. resting cube on pinned particles
/** 10 s resting on the pinned bottom layer: max |V / V0 - 1| over every step, final max |v|, top sag, finite. */
function rest(opts) {
  const w = new DynamicsWorld({ maxParticles: 512, particleDamping: 0.01 });
  const s = SoftBody.lattice(w, { size: [1, 1, 1], resolution: 4, position: [0, 0.5, 0], mass: 10, ...opts });
  pinLayer(w, s, -0.5);
  let maxDev = 0, ok = true;
  for (let f = 0; f < 600; f++) {
    w.step(DT);
    maxDev = Math.max(maxDev, Math.abs(s.volume() / s.restVolume - 1));
    if ((f & 15) === 0 && !finite(w)) ok = false;
  }
  return { maxDev, ok: ok && finite(w), v: maxSpeed(w, s), sag: 1 - extents(w, s)[4] };
}
for (const [label, opts] of [['default, edge 1e-4', {}], ['soft edges 1e-3', { edge: 1e-3 }]]) {
  const r = rest(opts), c = rest({ ...opts, volume: false });
  check(`rest (${label}): volume within 5e-4 of rest at every step (10 s)`, r.maxDev, 0, 5e-4);
  check(`rest (${label}): control without tets (volume: false) deviates 4x more`, c.maxDev / r.maxDev, 4, Infinity);
  check(`rest (${label}): no NaN`, r.ok);
  check(`rest (${label}): settled, max |v| at the end`, r.v, 0, 0.05, 'm/s');
  check(`rest (${label}): top stays near y = 1 (sag)`, r.sag, -1e-3, 0.05, 'm');
  note(`rest (${label}): max volume deviation ${r.maxDev.toExponential(2)} (without tets ${c.maxDev.toExponential(2)}), top sag ${r.sag.toExponential(2)} m`);
}

// ================================================================================== 5. squeeze: bulge, volume
{
  const r = squeeze({ edge: 1e-2, volume: 0 });
  const e = extents(r.w, r.s);
  check('squeeze: bulges sideways in x (extent)', e[3] - e[0], 1.03, 2, 'm');
  check('squeeze: bulges sideways in z (extent)', e[5] - e[2], 1.03, 2, 'm');
  check('squeeze: volume within 1 % at every step', r.maxDev, 0, 0.01);
  check('squeeze: no NaN', finite(r.w));
  note(`squeeze (edge 1e-2, volume 0): x extent ${(e[3] - e[0]).toFixed(4)}, z ${(e[5] - e[2]).toFixed(4)}, volume ratio end ${r.end.toFixed(5)}, max dev ${r.maxDev.toExponential(2)}`);
  const soft = squeeze({ edge: 1e-2, volume: 1e-3 });
  check('compliance: volume 1e-3 loses more volume than 0', (1 - soft.end) - (1 - r.end), 1e-3, 1);
  check('compliance: tet force = lambda / h^2 (Pa) positive under compression', r.s.solver.force.reduce((a, b) => Math.max(a, b), 0), 1, Infinity, 'Pa');
  const def = squeeze({}), truss = squeeze({ edge: 0 }), defP = squeeze({ pressure: 0 });
  check('squeeze: default body + rigid pressure (0) keeps the volume to 0.2 %', defP.maxDev, 0, 2e-3);
  note(`compliance: volume ratio at the end: volume 0 -> ${r.end.toFixed(5)}, volume 1e-3 -> ${soft.end.toFixed(5)}; rigid tets ` +
    `keep ${def.end.toFixed(4)} with the default edges (1e-4), ${truss.end.toFixed(4)} with a rigid truss (edge 0), ` +
    `${defP.end.toFixed(5)} with pressure: 0 (one Gauss-Seidel pass per substep: use pressure)`);
}

// ================================================================================== 6. pressure (global volume)
{
  const free = squeeze({ edge: 1e-3, volume: false });
  const pr = squeeze({ edge: 1e-3, volume: false, pressure: 0 });
  const prSoft = squeeze({ edge: 1e-3, volume: 1e-2, pressure: 0 });
  check('pressure 0 (rigid): total volume within 0.1 % at every step', pr.maxDev, 0, 1e-3);
  check('pressure: individual tets deform (max |V_t / V0 - 1|)', maxTetDeviation(pr.w, pr.s), 0.05, Infinity);
  check('pressure: without it the squeezed cube loses more volume', (1 - free.end) - Math.abs(1 - pr.end), 0.02, 1);
  check('pressure + soft tets: total volume within 0.1 %', prSoft.maxDev, 0, 1e-3);
  check('pressure: no NaN', finite(pr.w) && finite(prSoft.w));
  const rec = pr.s.solver.pressure[pr.s.pressureConstraint];
  check('pressure: force (Pa) positive under compression', rec.state[1], 1, Infinity, 'Pa');
  note(`pressure: end volume ratios: edges only ${free.end.toFixed(4)}, + pressure ${pr.end.toFixed(5)}, + soft tets ${prSoft.end.toFixed(5)}; tet deviation ${maxTetDeviation(pr.w, pr.s).toFixed(3)}`);
  // option semantics: false = off, a number (0 included) = the compliance; negative -> 0, non-finite throws
  const w = new DynamicsWorld({ maxParticles: 1024 });
  const off = SoftBody.lattice(w, { resolution: 2 });
  const on0 = SoftBody.lattice(w, { resolution: 2, pressure: 0 });
  const neg = SoftBody.lattice(w, { resolution: 2, pressure: -1, volume: -2 });
  check('options: pressure defaults to off, pressure: 0 is a rigid constraint', off.pressureConstraint === -1 && on0.pressureConstraint >= 0 &&
    on0.solver.pressure[on0.pressureConstraint].compliance === 0);
  check('options: negative compliances are stored as 0', neg.solver.pressure[neg.pressureConstraint].compliance === 0 &&
    neg.solver.compliance[neg.tetConstraints[0]] === 0);
  let threw = 0;
  for (const bad of [{ volume: Infinity }, { pressure: NaN }, { edge: false }, { volume: '1e-3' }]) {
    try { SoftBody.lattice(w, { resolution: 2, ...bad }); } catch (e) { if (e instanceof RangeError) threw++; }
  }
  check('options: non-finite / non-number compliances throw RangeError (4 cases)', threw, 4);
  try { on0.solver.addTet(0, 1, 2, 3, 1, Infinity); } catch (e) { threw += e instanceof RangeError ? 1 : 0; }
  check('options: VolumeConstraints.addTet with an infinite compliance throws', threw, 5);
  const w2 = new DynamicsWorld({ maxParticles: 64 });
  const bare = SoftBody.lattice(w2, { resolution: 2, volume: false });
  check('options: volume false + no pressure registers no solver', bare.solver === null && w2.particleSolvers.length === 1 &&
    VolumeConstraints.find(w2) === null);
  bare.dispose();
  check('options: disposing a solver-less body works', w2.particles.size === 0 && w2.constraints.size === 0);
}

// ================================================================================== 6b. freed particles
// A particle freed under a live body (no dispose) is invMass 0 at the origin: unless the solver skips its tets (tet and
// pressure passes alike) it drags them there. Soft edges (1e-1) so the volume terms dominate; the corner particle goes.
for (const [label, opts] of [['tets', { volume: 0 }], ['pressure only', { volume: false, pressure: 0 }]]) {
  const w = new DynamicsWorld({ maxParticles: 256, particleDamping: 0.01 });
  const s = SoftBody.lattice(w, { size: [0.5, 0.5, 0.5], resolution: 3, position: [0, 2, 0], mass: 2, edge: 1e-1, ...opts });
  const cid = s.ids[0], P = w.particles;
  P.free(cid);
  let maxR = 0;
  for (let f = 0; f < 60; f++) {
    w.step(DT);
    let cx = 0, cy = 0, cz = 0, n = 0;
    for (const id of s.ids) if (P.alive[id]) { cx += P.pos[id * 3]; cy += P.pos[id * 3 + 1]; cz += P.pos[id * 3 + 2]; n++; }
    cx /= n; cy /= n; cz /= n;
    for (const id of s.ids) if (P.alive[id]) maxR = Math.max(maxR, Math.hypot(P.pos[id * 3] - cx, P.pos[id * 3 + 1] - cy, P.pos[id * 3 + 2] - cz));
  }
  // (without the alive test: 0.92 m for the tets, 0.47 m for the pressure; the half-diagonal is 0.433 m)
  check(`freed particle (${label}): the body keeps its size (max distance from its centre)`, maxR, 0, 0.45, 'm');
  check(`freed particle (${label}): no NaN, the freed slot untouched at the origin`, finite(w) && P.pos[cid * 3] === 0 && P.pos[cid * 3 + 1] === 0);
  if (label === 'pressure only') {
    let v = 0, v0 = 0;
    for (let t = 0; t < s.tetCount; t++) {
      const k = t * 4;
      if (!P.alive[s.tets[k]] || !P.alive[s.tets[k + 1]] || !P.alive[s.tets[k + 2]] || !P.alive[s.tets[k + 3]]) continue;
      v += signedVolume(P.pos, s.tets[k], s.tets[k + 1], s.tets[k + 2], s.tets[k + 3]);
      v0 += s.restVolume / s.tetCount;
    }
    check('freed particle (pressure only): the alive tets keep their rest volume', Math.abs(v / v0 - 1), 0, 1e-3);
  }
}

// ================================================================================== 7. skinning
{
  const w = new DynamicsWorld({ maxParticles: 512, particleDamping: 0.01 });
  const q = [0, Math.sin(0.3), 0, Math.cos(0.3)];
  const s = SoftBody.lattice(w, { size: [1, 1, 1], resolution: 4, position: [0, 0.5, 0], quaternion: q, mass: 10, edge: 1e-3 });
  const L = s.topology.points, lt = s.topology.tets, t = 77;
  const cen = [0, 1, 2].map((k) => (L[lt[t * 4] * 3 + k] + L[lt[t * 4 + 1] * 3 + k] + L[lt[t * 4 + 2] * 3 + k] + L[lt[t * 4 + 3] * 3 + k]) / 4);
  const mesh = new Float32Array([...cen, 1.3, 0.2, -0.1, 0.5, 0.5, 0.5, -0.2, 0.1, 0.33, 0.7, 0.7, 0.7]);
  const mesh64 = Float64Array.from(mesh);
  mesh64.set(cen, 0);
  const sk = SoftBody.skin(s, mesh64);
  check('skin: centroid vertex bound to its tet', sk.tet[0], t);
  check('skin: centroid weights 1/4', Math.max(...[0, 1, 2, 3].map((k) => Math.abs(sk.weights[k] - 0.25))), 0, 1e-12);
  const out = new Float64Array(mesh64.length);
  sk.update(out);
  // at rest the skin reproduces the rigidly placed rest mesh (inside and extrapolated vertices alike)
  let rigidErr = 0;
  const tmp = [0, 0, 0];
  for (let i = 0; i < 5; i++) {
    const [x, y, z] = [mesh64[i * 3], mesh64[i * 3 + 1], mesh64[i * 3 + 2]];
    const c = Math.cos(0.6), sn = Math.sin(0.6); // rotation about y by 2 * 0.3
    tmp[0] = c * x + sn * z; tmp[1] = y + 0.5; tmp[2] = -sn * x + c * z;
    rigidErr = Math.max(rigidErr, Math.hypot(out[i * 3] - tmp[0], out[i * 3 + 1] - tmp[1], out[i * 3 + 2] - tmp[2]));
  }
  check('skin: at rest = rest mesh placed by position / quaternion (5 vertices)', rigidErr, 0, 1e-12, 'm');
  check('skin: outside vertex (1.3, 0.2, -0.1) bound, weights finite', sk.weights.slice(4, 8).every(Number.isFinite) && sk.tet[1] >= 0);
  check('skin: weights of every vertex sum to 1', Math.max(...[0, 1, 2, 3, 4].map((i) => Math.abs(sk.weights[i * 4] + sk.weights[i * 4 + 1] + sk.weights[i * 4 + 2] + sk.weights[i * 4 + 3] - 1))), 0, 1e-12);
  // deform: pin the bottom, let it sag and swing for 1 s with a shove
  pinLayer(w, s, -0.5);
  for (let i = 0; i < s.count; i++) w.particles.vel[s.ids[i] * 3] = 0.5 * s.topology.points[i * 3 + 1];
  for (let f = 0; f < 60; f++) w.step(DT);
  sk.update(out);
  const X = w.particles.pos, T = s.tets;
  const tc = [0, 1, 2].map((k) => (X[T[t * 4] * 3 + k] + X[T[t * 4 + 1] * 3 + k] + X[T[t * 4 + 2] * 3 + k] + X[T[t * 4 + 3] * 3 + k]) / 4);
  check('skin: centroid vertex tracks the deformed tet centroid', Math.hypot(out[0] - tc[0], out[1] - tc[1], out[2] - tc[2]), 0, 1e-9, 'm');
  check('skin: deformed output finite (outside vertex too)', out.every(Number.isFinite));
  check('skin: deformation happened (tet volume changed)', maxTetDeviation(w, s), 1e-4, Infinity);
  const out32 = new Float32Array(mesh.length);
  sk.update(out32);
  check('skin: Float32Array output agrees', Math.abs(out32[0] - out[0]) + Math.abs(out32[4] - out[4]), 0, 1e-6);
  const skc = SoftBody.skin(s, mesh64, { clamp: true });
  check('skin { clamp }: outside vertex weights >= 0, sum 1', Math.min(...skc.weights.slice(4, 8)) >= 0 && Math.abs(skc.weights[4] + skc.weights[5] + skc.weights[6] + skc.weights[7] - 1) < 1e-12);
  // a round mesh over the voxel sphere: vertices in the dropped corners are bound to the nearest surface tet
  const w2 = new DynamicsWorld({ maxParticles: 256 });
  const sp = SoftBody.lattice(w2, { shape: 'sphere', radius: 0.5, resolution: 4 });
  const ring = [];
  for (let i = 0; i < 26; i++) { const a = i * 0.7, b = i * 1.3; ring.push(0.5 * Math.cos(a) * Math.sin(b), 0.5 * Math.cos(b), 0.5 * Math.sin(a) * Math.sin(b)); }
  ring.push(0.2887, 0.2887, 0.2887);
  const sks = SoftBody.skin(sp, Float32Array.from(ring));
  const o2 = sks.update(new Float64Array(ring.length));
  let e2 = 0;
  for (let i = 0; i < ring.length; i++) e2 = Math.max(e2, Math.abs(o2[i] - Math.fround(ring[i])));
  check('skin: sphere mesh over the voxel sphere reproduced at rest', e2, 0, 1e-9, 'm');
  // least extrapolation: a denser round mesh (r 0.55, a little outside the 0.5 sphere); for every vertex outside the
  // lattice, the chosen tet's sum |w| (the lever arm of a tet perturbation) is at most that of the nearest surface
  // triangle's own tet (found here independently)
  const ST = sp.topology, SA = ST.points, dense = [];
  for (let i = 0; i < 400; i++) { const z = 1 - (2 * i + 1) / 400, r = Math.sqrt(1 - z * z), a = i * 2.399963; dense.push(0.55 * r * Math.cos(a), 0.55 * z, 0.55 * r * Math.sin(a)); }
  const skd = SoftBody.skin(sp, Float64Array.from(dense));
  let worse = 0, better = 0, outside = 0, wMax = 0, wMaxOwn = 0;
  for (let i = 0; i < 400; i++) {
    const p = [dense[i * 3], dense[i * 3 + 1], dense[i * 3 + 2]];
    if (minBary(ST, skd.tet[i], p) >= -1e-9) continue; // inside the lattice
    outside++;
    let bd = Infinity, own = -1;
    for (let k = 0; k < ST.surfaceCount; k++) {
      const d = pointTriangleDistance(p, SA, ST.surface[k * 3], ST.surface[k * 3 + 1], ST.surface[k * 3 + 2]);
      if (d < bd - 1e-12) { bd = d; own = ST.surfaceTet[k]; }
    }
    const l1 = (t) => baryOf(ST, t, p).reduce((acc, x) => acc + Math.abs(x), 0), mc = l1(skd.tet[i]), mo = l1(own);
    if (mc > mo + 1e-12) worse++; else if (mc < mo - 1e-12) better++;
    wMax = Math.max(wMax, [0, 1, 2, 3].reduce((acc, k) => acc + Math.abs(skd.weights[i * 4 + k]), 0));
    wMaxOwn = Math.max(wMaxOwn, baryOf(ST, own, p).reduce((acc, x) => acc + Math.abs(x), 0));
  }
  check('skin: outside vertices never bound to a more extrapolating tet than the triangle\'s own', worse, 0);
  check('skin: and to a less extrapolating one for some (edge / corner vertices)', better, 1, Infinity);
  check('skin: dense round mesh reproduced at rest', (() => { const o = skd.update(new Float64Array(1200)); let e = 0; for (let i = 0; i < 1200; i++) e = Math.max(e, Math.abs(o[i] - dense[i])); return e; })(), 0, 1e-12, 'm');
  note(`skin: ${outside} of 400 sphere vertices outside the 4-cell voxel sphere, ${better} rebound; max sum |w| (the strain lever arm) ${wMax.toFixed(3)} (triangle's own tet: ${wMaxOwn.toFixed(3)})`);
}

// ================================================================================== 8. determinism, serialize / from
function detScene() {
  const w = new DynamicsWorld({ maxParticles: 512, particleDamping: 0.02 });
  const s = SoftBody.lattice(w, { size: [1, 0.6, 0.8], resolution: [4, 3, 3], position: [0, 2, 0], quaternion: [0.1, 0.2, 0, 0.9747], mass: 6, edge: 1e-4, volume: 1e-5, pressure: 1e-8 });
  const L = s.topology.points;
  for (let i = 0; i < s.count; i++) if (L[i * 3] < -0.49 && L[i * 3 + 1] > 0.29) w.particles.setPinned(s.ids[i], true); // one top edge
  for (let i = 0; i < s.count; i++) w.particles.vel[s.ids[i] * 3 + 2] = 1.5;
  return { w, s };
}
const posHash = (w) => hashF64Array(w.particles.pos);
const SOFT_GOLDEN = 0x18bfdba3; // positions after 300 steps of detScene (+ - * / sqrt only: platform-independent)
{
  const a = detScene(), b = detScene();
  for (let f = 0; f < 300; f++) { a.w.step(DT); b.w.step(DT); }
  const h1 = posHash(a.w), h2 = posHash(b.w);
  check('determinism: two identical 300-step runs hash-equal (positions)', h1 === h2);
  check('determinism: scene finite and moving', finite(a.w) && maxSpeed(a.w, a.s) > 0.01);
  check('golden: soft-body hash after 300 steps', h1 === SOFT_GOLDEN);
  note(`soft golden ${h1.toString(16)} (pinned ${SOFT_GOLDEN.toString(16)})`);
  const c = detScene();
  for (let f = 0; f < 150; f++) c.w.step(DT);
  const snap = c.w.serialize();
  const solver = new VolumeConstraints();
  const d = DynamicsWorld.from(structuredClone(snap), { solvers: [solver] });
  for (let f = 0; f < 150; f++) { c.w.step(DT); d.step(DT); }
  check('serialize / from: original continues to the golden hash', posHash(c.w), h1, h1);
  check('serialize / from: restored world continues to the golden hash', posHash(d), h1, h1);
  // pressure slots survive a round trip at their indices: [A, removed, B] restores as [A, null, B]
  const S1 = new VolumeConstraints(), T4 = Uint32Array.from([0, 1, 2, 3, 1, 2, 3, 4]);
  S1.addPressure(T4, 1, 0); S1.addPressure(T4, 2, 1e-6); S1.addPressure(T4, 3, 0, [1, 2]);
  S1.removePressure(1);
  S1.pressure[2].state[0] = 0.5;
  const x1 = S1.serialize(), S2 = new VolumeConstraints();
  S2.restore(structuredClone(x1));
  check('serialize / restore: a removed pressure slot before a live one keeps the indices',
    S2.pressure.length === 3 && S2.pressure[1] === null && S2.pressure[2].rest === 3 && S2.pressure[2].state[0] === 0.5 && S2.pressure[2].restTet[1] === 2);
  check('serialize / restore: serialize(restore(x)) deep-equals x', deepEqual(S2.serialize(), x1));
  check('serialize / restore: the next addPressure reuses the free slot', S2.addPressure(T4, 4, 0), 1);
  check('serialize / from: restored solver has the tets and the pressure', solver.size === a.s.tetCount && solver.pressure.filter(Boolean).length === 1);
}

// ================================================================================== 9. allocation and step cost
{
  let gc = globalThis.gc;
  if (!gc) { try { v8.setFlagsFromString('--expose-gc'); gc = vm.runInNewContext('gc'); } catch { gc = null; } }
  const make = (n) => {
    const w = new DynamicsWorld({ maxParticles: (n + 1) ** 3 + 8, particleDamping: 0.01 });
    const s = SoftBody.lattice(w, { size: [1, 1, 1], resolution: n, position: [0, 0.5, 0], mass: 10, edge: 1e-4, pressure: 1e-8 });
    pinLayer(w, s, -0.5);
    for (let i = 0; i < s.count; i++) w.particles.vel[s.ids[i] * 3] = 0.3 * s.topology.points[i * 3 + 1];
    return { w, s };
  };
  const { w, s } = make(6);
  for (let f = 0; f < 600; f++) w.step(DT); // warm-up (JIT tiers)
  performance.now(); process.memoryUsage();
  // heap growth while fn() runs, Infinity when a GC ran inside (the round is void); best of 3 rounds
  const grow = (fn) => {
    if (gc) gc();
    const prof = new v8.GCProfiler();
    prof.start();
    const h0 = process.memoryUsage().heapUsed;
    fn();
    const h1 = process.memoryUsage().heapUsed;
    return prof.stop().statistics.length ? Infinity : h1 - h0;
  };
  const best = (fn) => { let b = Infinity; for (let r = 0; r < 3; r++) b = Math.min(b, grow(fn)); return b; };
  let gcs = 0;
  {
    if (gc) gc(); // start with an empty young generation (the checks above leave garbage behind)
    const prof = new v8.GCProfiler();
    prof.start();
    for (let f = 0; f < 500; f++) w.step(DT);
    gcs = prof.stop().statistics.length;
  }
  check('alloc: GCs during 500 steps of a 6x6x6 cube', gcs, 0, 0);
  if (gc) gc();
  const r0 = process.memoryUsage().heapUsed;
  for (let f = 0; f < 500; f++) w.step(DT);
  if (gc) gc();
  const retained = process.memoryUsage().heapUsed - r0;
  check('alloc: retained heap over 500 steps (gc before / after)', retained / 1024, -Infinity, 1, 'KB');
  // what the soft body adds to a step: the same world with the soft solver unregistered (edges only) as the control,
  // since DynamicsWorld.step itself boxes a few doubles per step (short-lived, never retained)
  // the solver alone: position passes straight into it, and the per-frame helpers
  const S = s.solver, H = DT / 4;
  const passes = (n) => { for (let i = 0; i < n; i++) S.solvePositions(w, H, i & 3); };
  passes(8000);
  // heapUsed carries a constant few-KB offset between rounds (accounting granularity, not per call): bound the total
  // well below one heap number per call (4000 x 16 B = 64 KB) and the slope between 4000 and 20 000 calls
  const p4 = best(() => passes(4000)), p20 = best(() => passes(20000));
  check('alloc: VolumeConstraints.solvePositions x 4000, heap growth (best of 3)', p4, -Infinity, 8192, 'B');
  check('alloc: VolumeConstraints.solvePositions, bytes per call (slope 4k -> 20k)', Math.max(0, p20 - p4) / 16000, 0, 0.5, 'B');
  const sk = SoftBody.skin(s, Float64Array.from(s.topology.points)), out = new Float32Array(s.count * 3), c3 = new Float64Array(3);
  const vsum = new Float64Array(1); // a closure `let` would box every sum (harness allocation)
  s.centroid([0, 0, 0]); s.centroid({ x: 0, y: 0, z: 0 }); sk.update(new Float64Array(s.count * 3)); // every receiver kind
  const helpers = (n) => { for (let i = 0; i < n; i++) { s.centroid(c3); sk.update(out); } };
  const vols = (n) => { let v = 0; for (let i = 0; i < n; i++) v += s.volume(); vsum[0] = v; };
  helpers(5000); vols(5000);
  check('alloc: centroid() / skin.update x 5000, heap growth (best of 3)', best(() => helpers(5000)), -Infinity, 512, 'B');
  // a double returned from a call V8 does not inline is a 16 B heap number: that is all volume() may cost
  check('alloc: volume() x 5000, bytes per call (the boxed return at most)', best(() => vols(5000)) / 5000, -Infinity, 32, 'B');
  check('alloc: skin of the lattice points reproduces the particles', Math.abs(out[3] - w.particles.pos[s.ids[1] * 3]), 0, 1e-6);
  // what the soft body adds to a step: V8's sampling heap profiler, every allocation of 500 more steps attributed to
  // the file of its frame (DynamicsWorld.step itself boxes a few doubles per step: short-lived, never retained)
  let own = NaN, top = '';
  try {
    const inspector = await import('node:inspector/promises');
    const ses = new inspector.Session(); ses.connect();
    await ses.post('HeapProfiler.enable');
    await ses.post('HeapProfiler.startSampling', { samplingInterval: 32, includeObjectsCollectedByMajorGC: true, includeObjectsCollectedByMinorGC: true });
    for (let f = 0; f < 500; f++) w.step(DT);
    const { profile } = await ses.post('HeapProfiler.stopSampling');
    ses.disconnect();
    const by = new Map();
    const walk = (nd) => {
      const file = nd.callFrame.url.split('/').pop() || '(native)';
      if (nd.selfSize) by.set(file, (by.get(file) || 0) + nd.selfSize);
      for (const c of nd.children) walk(c);
    };
    walk(profile.head);
    own = (by.get('softbody.js') || 0) / 500;
    top = [...by.entries()].sort((x, y) => y[1] - x[1]).slice(0, 3).map(([f, n]) => `${f} ${(n / 500).toFixed(1)}`).join(', ');
  } catch { /* no inspector */ }
  // one boxed double per solver call would be 4 x 16 = 64 B/step; a one-off (tier-up) of ~3 KB reads up to ~7 B/step
  if (Number.isFinite(own)) check('alloc: bytes per step allocated in softbody.js frames (sampled)', own, 0, 16, 'B');
  note(`alloc: 500 steps of the 6x6x6 cube: ${gcs} GCs, ${retained} B retained; sampled B/step by file: ${top || 'n/a'}`);
  // step cost (60 Hz frame, 4 substeps)
  const cost = (n) => {
    const r = make(n);
    for (let f = 0; f < 120; f++) r.w.step(DT);
    const t0 = performance.now();
    for (let f = 0; f < 300; f++) r.w.step(DT);
    return { ms: (performance.now() - t0) / 300, s: r.s };
  };
  const c6 = cost(6), c8 = cost(8);
  note(`perf: 6x6x6 cube (${c6.s.count} particles, ${c6.s.edgeCount} edges, ${c6.s.tetCount} tets + pressure): ${c6.ms.toFixed(3)} ms per step (60 Hz x 4 substeps)`);
  note(`perf: 8x8x8 cube (${c8.s.count} particles, ${c8.s.edgeCount} edges, ${c8.s.tetCount} tets + pressure): ${c8.ms.toFixed(3)} ms per step`);
  if (!gc) note('alloc: no gc available: heap growth measured without a gc first');
}

console.log(results.join('\n'));
const passed = results.filter((r) => r.startsWith('PASS')).length;
console.log(failures ? `\n${failures} check(s) FAILED` : `\nall ${passed} checks passed`);
process.exit(failures ? 1 : 0);
