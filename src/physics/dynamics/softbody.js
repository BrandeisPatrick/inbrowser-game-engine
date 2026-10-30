// Tetrahedral soft bodies on the dynamics world's particle store (pure JS, deterministic, Node-safe, no THREE): a
// regular lattice cut into tetrahedra (tetrahedralize), the XPBD tet-volume and global-volume ("pressure") constraints
// that keep it from collapsing (VolumeConstraints, one particle solver per world), the soft-body handle that builds a
// lattice into the world's ParticleStore / ParticleConstraints (SoftBody.lattice) and the barycentric skin that
// drives an arbitrary render mesh from the tets (SoftBody.skin). References: Macklin, Mueller, Chentanez 2016, "XPBD:
// Position-Based Simulation of Compliant Constrained Dynamics" (compliance, lambdas); Mueller's "Ten Minute Physics"
// soft-body tutorial (the tet volume constraint with Macklin's gradients, edge + volume constraints on a tet mesh).
//
//   const world = new DynamicsWorld({ maxParticles: 4096 });
//   const jelly = SoftBody.lattice(world, { shape: 'box', size: [1, 1, 1], resolution: 4, position: [0, 2, 0],
//                                          mass: 10, edge: 1e-4, volume: 0, pressure: 0 });
//   const skin = SoftBody.skin(jelly, meshRestPositions);         // mesh in the lattice's local (rest) frame
//   engine fixed step: world.step(dt);  render: skin.update(geometry.attributes.position.array); needsUpdate = true
//   jelly.volume() / jelly.restVolume;  jelly.dispose();
//
// API
//   tetrahedralize(cells, inside = null, size = cells) -> topology (plain typed arrays, pure, deterministic)
//       cells: [nx, ny, nz] (or one number) cells per axis; size: [sx, sy, sz] full extents of the lattice box (or one
//       number), centred on the origin, so the cell is (sx / nx, sy / ny, sz / nz); inside(x, y, z) -> bool: the cell
//       is kept when its CENTRE passes (null keeps every cell). Every kept cube cell is split into the same 6 tets (the
//       Freudenthal / Kuhn split along the cell's main diagonal, corner (0, 0, 0) -> (1, 1, 1): every cube face is cut
//       along the diagonal through its own lowest corner in both cells that share it, so the lattice is conforming);
//       points no kept cell uses are dropped and the rest renumbered in ascending lattice order (i fastest, then j, k).
//       Returns { points: Float64Array(3p) (local coordinates), pointCount, tets: Uint32Array(4m) (point indices,
//       signed volume (p1 - p0) . ((p2 - p0) x (p3 - p0)) / 6 > 0), tetCount, edges: Uint32Array(2e) (unique, a < b,
//       ascending), edgeCount, surface: Uint32Array(3k) (boundary triangles: tet faces no other tet shares, counter-
//       clockwise seen from outside, in tet order), surfaceCount, surfaceTet: Uint32Array(k) (the tet of each surface
//       triangle), cellTet: Int32Array(nx ny nz) (first tet of each cell, -1 for a dropped cell; a cell's 6 tets are
//       consecutive), cells, size, origin (the lattice's min corner), cellSize }. The surface is closed, edge-manifold
//       and consistently oriented by construction; it is also vertex-manifold when no two kept cells meet only at a
//       corner or along an edge (true for boxes and for convex `inside` tests such as the voxel spheres).
//   SoftBody.lattice(world, { shape = 'box' | 'sphere', size = [1, 1, 1] (box full extents), radius = 0.5 (sphere),
//     resolution = [4, 4, 4] | 4 (cells per axis; a sphere spans 2 radius), position = [0, 0, 0], quaternion = [0, 0,
//     0, 1], mass = 10 (total; every particle gets mass / count), edge = 1e-4 (compliance of the edge distance
//     constraints, m/N; 0 makes the edges a rigid, over-constrained truss that fights rigid tets in the one
//     Gauss-Seidel pass per substep: volume loss and jitter under load), volume = 0 (compliance of the per-tet volume
//     constraints, m^3/Pa = m^5/N, 0 = rigid; false: no per-tet constraints), pressure = false (false = off; a
//     number, 0 included: the compliance (m^3/Pa) of one global volume constraint over all the body's tets, which
//     keeps the total volume while single tets deform), particleRadius = cell * 0.4 (collision radius; for a box
//     `radius` is accepted as an alias, for a sphere `radius` is the sphere's), group = world.particles.newGroup() })
//     -> SoftBody
//       Compliances: negative values are stored as 0 (as particles.js), a non-number / non-finite one throws.
//       The sphere keeps the cells of the [-r, r]^3 lattice whose centre lies within r (a voxel sphere: skin a round
//       mesh on it). position / quaternion place the lattice's local frame (arrays or {x, y, z(, w)}). Edges go to
//       world.constraints.addDistance (rest = creation length), tets to the world's VolumeConstraints (rest = creation
//       volume). Throws a RangeError when the particle store cannot hold the lattice (nothing is left allocated).
//   SoftBody fields: world, first (ids[0]), count, ids: Int32Array (lattice point -> particle id), group, tets:
//     Uint32Array(4m), tetCount, edges: Uint32Array(2e), edgeCount, surface: Uint32Array(3k), surfaceCount,
//     surfaceVertices: Int32Array (unique boundary particle ids, ascending), restVolume (sum of the rest tet volumes,
//     m^3), edgeConstraints: Int32Array (constraint index per edge), tetConstraints: Int32Array (tet index in the
//     solver per tet, -1 when volume is off), pressureConstraint (-1 when off), solver (the world's VolumeConstraints;
//     null when the body has neither tets nor pressure, and then none is created), topology (tetrahedralize's
//     result: local rest frame, point indices). Particle ids are the store's (ascending lattice order as allocated).
//     volume() -> current total signed volume (m^3; allocates nothing itself, V8 boxes the returned double when it does
//     not inline the call); centroid(out3) -> out3 (mean particle position, every particle has the same mass; out3 a
//     Float64Array, an array or {x, y, z}, each kind with its own store site); dispose() frees the particles and
//     removes the edge, tet and pressure constraints; the shared solver is unregistered when it holds none anymore.
//   SoftBody.skin(soft, positions, { clamp = false } = {}) -> SoftSkin { vertexCount, tet: Int32Array(v) (index into
//     soft.tets), ids: Uint32Array(4v) (particle ids), weights: Float64Array(4v), update(out) }
//       positions: Float32Array / Float64Array (3v), the mesh's rest positions in the lattice's LOCAL frame (the frame
//       position / quaternion place; the lattice is centred on its origin). Each vertex is bound to the tet that
//       contains it (its cell is found directly from the lattice coordinates, then the cell's 6 tets) or, outside the
//       lattice (a round mesh over a voxel sphere), to the least extrapolating tet among those around the nearest
//       surface triangle's vertices: the smallest sum |w| (ties: the largest minimum weight). sum |w| bounds how much
//       a perturbation of the tet's vertices is amplified at the vertex, and the triangle's own tet is a candidate, so
//       the choice is never worse than it. The largest minimum weight alone (Mueller's choice) picks the same tet when
//       one weight is negative but, with two, raised the worst sum |w| of a round mesh over the voxel sphere from 1.83
//       to 1.97 (4 cells; 6 cells: 1.60 to 1.91).
//       Weights are the vertex's barycentric coordinates in its tet; outside the lattice they extrapolate (some are
//       negative: the vertex keeps its offset from the tet under any affine motion, and update() at rest returns the
//       rest mesh exactly). Lever arm: at a distance d outside the lattice the weights grow like 1 + d / cell (up to
//       ~3 for a mesh well outside a coarse voxel sphere's corners), so a tet strain e moves the vertex by about
//       e (1 + d / cell) cells: keep the mesh within about a cell of the lattice (more resolution) or pass
//       { clamp: true }, which clamps the weights to >= 0 and renormalises (the vertex is projected onto its tet).
//       update(out): out[3i..3i+3] = sum_k weights[4i + k] pos(ids[4i + k]), world space, allocation-free; out is any
//       array-like of 3v numbers (Float32Array for a BufferAttribute, Float64Array to keep full precision).
//   VolumeConstraints (the particle solver; one per world, shared by every soft body). VolumeConstraints.find(world)
//     -> the first VolumeConstraints in world.particleSolvers or null (e.g. the one DynamicsWorld.from(s, { solvers:
//     [new VolumeConstraints()] }) registered and restored); VolumeConstraints.of(world, capacity = 1024) -> find(),
//     or a new one it registers (world.addParticleSolver). Both scan the solver list: construction time only.
//     new VolumeConstraints(world = null, { capacity = 1024 }) builds an unregistered one; `world` is not retained.
//     Name 'softbody.volume' (the key of its serialized state in the world snapshot).
//     addTet(a, b, c, d, restVolume, compliance) -> ti (storage grows by doubling: no hard cap); remove(ti);
//     addPressure(tets: Uint32Array(4m) particle ids, restVolume, compliance, restTets = restVolume / m each) -> pi
//     (the lowest free slot); removePressure(pi) (the slot stays null until reused); compliances: negative -> 0,
//     Infinity throws; isEmpty(); serialize() -> plain typed-array copies | restore(data) (the solver's whole state,
//     lambdas included, every pressure record at its snapshot index).
//     Fields (SoA, index = tet id): count (one past the highest id), alive, tet (Uint32Array(4n), particle ids),
//     restVolume, compliance, lambda, force (lambda / h^2 of the last position pass, Pa: > 0 compressed, pushing out),
//     pressure (array of { tets, restTet, particles, rest, compliance, state: Float64Array(2) [lambda, force] } or
//     null for a removed slot).
//   Snapshots: DynamicsWorld.serialize / from carry the particles, the edge constraints and this solver's state (pass
//     a new VolumeConstraints() in from's `solvers`), not the SoftBody / SoftSkin handles (as with Cloth / Rope): after
//     from() the bodies continue bit-identically, but a handle has to be rebuilt by the caller if it needs one.
//
// Solve (solvePositions(world, h, iteration), called by the world in every position iteration of every substep after
// its built-in particle constraints; iteration 0 resets the lambdas). Per alive tet, ascending (Gauss-Seidel), the XPBD
// volume constraint C = V - V0 with V = (p1 - p0) . ((p2 - p0) x (p3 - p0)) / 6; the gradient of each vertex is 1/6 of
// the cross product of two edges of the opposite face (grad_1 = e2 x e3 / 6, grad_2 = e3 x e1 / 6, grad_3 = e1 x e2 / 6
// with e_k = p_k - p0, grad_0 = (p3 - p1) x (p2 - p1) / 6):
//   alpha~ = compliance / h^2;  dLambda = (-C - alpha~ lambda) / (sum_i w_i |grad_i|^2 + alpha~);
//   p_i += w_i dLambda grad_i
// (w_i = invMass: 0 for a pinned particle, ParticleStore.setPinned). An inverted tet (V < 0) is pushed back out the
// same way. A tet with a freed vertex (particles.alive 0) is skipped, as particles.js skips dead ends. Then each global
// volume constraint once: C = sum V_t - sum V0_t over its tets whose vertices are all alive, grad_i = sum over those
// tets touching i, the same update applied to all its particles at once (one scalar constraint, exact for it).
// Envelope: one Gauss-Seidel pass per substep does not converge a stiff mix. When a 4^3 cube is squeezed by 20 %
// (test/dyn-soft.mjs) rigid tets keep the volume to 0.7 % with edge compliance 1e-2 but lose 3 % at 1e-3, 6.7 % at the
// default 1e-4 and 8.3 % with rigid edges (0, which also jitters); a rigid pressure constraint (pressure: 0) holds the
// default body to 0.08 % and soft-edged ones to 0.02 %. Use pressure for incompressibility, tets for local shape. At
// rest under gravity the tets keep the volume to 3e-4 (edges alone: 2e-3 at the default compliance, 1e-2 at 1e-3).
//
// Determinism: the solve uses + - * / and comparisons only (no Math.sqrt); tets, pressure constraints and particles in
// ascending index order. Allocation: none in the solve (typed arrays; scalars stay inside one function); the lattice,
// the skin binding and the storage growth allocate when they are created.
import { rotate } from './xpbd.js';

const SIXTH = 1 / 6;
// the 6 tets of a cube cell: paths 0 -> 0 + e_a -> 0 + e_a + e_b -> (1, 1, 1) for the axis permutations (a, b, c)
const PERMS = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
// outward faces of a positively oriented tet (p0 p1 p2 p3), counter-clockwise seen from outside
const FACES = [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]];

// ---- small construction-time helpers --------------------------------------------------------------------------------
function vec3(v, d) {
  if (v == null) return [d[0], d[1], d[2]];
  if (typeof v === 'number') return [v, v, v];
  if (typeof v.x === 'number') return [+v.x, +v.y, +v.z];
  return [+v[0], +v[1], +v[2]];
}
function quat4(q) {
  if (q == null) return [0, 0, 0, 1];
  const x = typeof q.x === 'number' ? +q.x : +q[0], y = typeof q.x === 'number' ? +q.y : +q[1];
  const z = typeof q.x === 'number' ? +q.z : +q[2], w = typeof q.x === 'number' ? +q.w : +q[3];
  const l = Math.sqrt(x * x + y * y + z * z + w * w);
  return l > 0 ? [x / l, y / l, z / l, w / l] : [0, 0, 0, 1];
}
function asArr(v) { return v == null ? [] : ArrayBuffer.isView(v) || Array.isArray(v) ? v : Object.values(v); }
/** A lattice compliance option: false (allowFalse) = off, else a finite number; negative -> 0 (as particles.js). */
function complianceOpt(c, name, allowFalse) {
  if (c === false && allowFalse) return false;
  if (typeof c !== 'number' || !Number.isFinite(c)) {
    throw new RangeError(`SoftBody.lattice: ${name} must be ${allowFalse ? 'false or ' : ''}a finite compliance, got ${c}`);
  }
  return c > 0 ? c : 0;
}
/** A compliance as stored: negative / NaN -> 0 (as particles.js); Infinity throws (alpha~ lambda would be NaN). */
function compliancePos(c, who) {
  if (c === Infinity) throw new RangeError(`VolumeConstraints.${who}: infinite compliance (remove the constraint instead)`);
  return c > 0 ? +c : 0;
}
/** 6 x the signed volume of (a, b, c, d) read from a packed xyz array. */
function vol6(P, a, b, c, d) {
  const a3 = a * 3, b3 = b * 3, c3 = c * 3, d3 = d * 3, x0 = P[a3], y0 = P[a3 + 1], z0 = P[a3 + 2];
  const e1x = P[b3] - x0, e1y = P[b3 + 1] - y0, e1z = P[b3 + 2] - z0;
  const e2x = P[c3] - x0, e2y = P[c3 + 1] - y0, e2z = P[c3 + 2] - z0;
  const e3x = P[d3] - x0, e3y = P[d3 + 1] - y0, e3z = P[d3 + 2] - z0;
  return e1x * (e2y * e3z - e2z * e3y) + e1y * (e2z * e3x - e2x * e3z) + e1z * (e2x * e3y - e2y * e3x);
}

// ---- lattice topology -----------------------------------------------------------------------------------------------
/**
 * Tetrahedralize a regular lattice of nx x ny x nz cube cells (6 tets per kept cell); see the header.
 * @param {number|number[]} cells  cells per axis
 * @param {((x: number, y: number, z: number) => boolean)|null} [inside]  cell-centre test (local coordinates)
 * @param {number|number[]|null} [size]  full extents of the lattice box (default: 1 per cell)
 */
export function tetrahedralize(cells, inside = null, size = null) {
  const [fx, fy, fz] = vec3(cells, [4, 4, 4]);
  const nx = Math.floor(fx), ny = Math.floor(fy), nz = Math.floor(fz);
  if (!(nx >= 1 && ny >= 1 && nz >= 1)) throw new RangeError(`tetrahedralize: bad cell counts ${cells}`);
  const [sx, sy, sz] = vec3(size, [nx, ny, nz]);
  if (!(sx > 0 && sy > 0 && sz > 0)) throw new RangeError(`tetrahedralize: bad size ${size}`);
  const cx = sx / nx, cy = sy / ny, cz = sz / nz, ox = -0.5 * sx, oy = -0.5 * sy, oz = -0.5 * sz;
  const px = nx + 1, py = ny + 1, pz = nz + 1, NP = px * py * pz, NC = nx * ny * nz;
  // kept cells
  const keep = new Uint8Array(NC);
  let kept = 0;
  for (let k = 0, c = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++, c++) {
    const ok = inside ? !!inside(ox + (i + 0.5) * cx, oy + (j + 0.5) * cy, oz + (k + 0.5) * cz) : true;
    if (ok) { keep[c] = 1; kept++; }
  }
  // full-lattice point coordinates (point index i + px (j + py k))
  const full = new Float64Array(3 * NP);
  for (let k = 0, p = 0; k < pz; k++) for (let j = 0; j < py; j++) for (let i = 0; i < px; i++, p++) {
    full[p * 3] = ox + i * cx; full[p * 3 + 1] = oy + j * cy; full[p * 3 + 2] = oz + k * cz;
  }
  // tets on full-lattice indices
  const step = [1, px, px * py];
  const m = kept * 6, tets = new Uint32Array(4 * m), cellTet = new Int32Array(NC).fill(-1);
  let t = 0;
  for (let k = 0, c = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++, c++) {
    if (!keep[c]) continue;
    cellTet[c] = t;
    const v0 = i + px * (j + py * k), v7 = v0 + step[0] + step[1] + step[2];
    for (let s = 0; s < 6; s++, t++) {
      const pr = PERMS[s], v1 = v0 + step[pr[0]], v2 = v1 + step[pr[1]];
      const o = t * 4;
      tets[o] = v0; tets[o + 1] = v1; tets[o + 2] = v2; tets[o + 3] = v7;
      if (vol6(full, v0, v1, v2, v7) < 0) { tets[o + 2] = v7; tets[o + 3] = v2; } // odd permutation: flip
    }
  }
  // drop unused points, renumber ascending
  const remap = new Int32Array(NP).fill(-1);
  for (let q = 0; q < 4 * m; q++) remap[tets[q]] = 0;
  let P = 0;
  for (let p = 0; p < NP; p++) if (remap[p] === 0) remap[p] = P++;
  const points = new Float64Array(3 * P);
  for (let p = 0; p < NP; p++) {
    const r = remap[p];
    if (r >= 0) { points[r * 3] = full[p * 3]; points[r * 3 + 1] = full[p * 3 + 1]; points[r * 3 + 2] = full[p * 3 + 2]; }
  }
  for (let q = 0; q < 4 * m; q++) tets[q] = remap[tets[q]];
  // unique edges: key a * P + b (a < b), sorted
  const keys = new Float64Array(6 * m);
  for (let tt = 0, e = 0; tt < m; tt++) {
    const o = tt * 4;
    for (let u = 0; u < 3; u++) for (let v = u + 1; v < 4; v++, e++) {
      const a = tets[o + u], b = tets[o + v];
      keys[e] = a < b ? a * P + b : b * P + a;
    }
  }
  keys.sort();
  let E = 0;
  for (let e = 0; e < keys.length; e++) if (e === 0 || keys[e] !== keys[e - 1]) keys[E++] = keys[e];
  const edges = new Uint32Array(2 * E);
  for (let e = 0; e < E; e++) { const b = keys[e] % P; edges[e * 2] = (keys[e] - b) / P; edges[e * 2 + 1] = b; }
  // boundary faces: faces whose sorted triple occurs once
  const F = 4 * m, srt = new Uint32Array(3 * F), order = new Uint32Array(F);
  for (let f = 0; f < F; f++) {
    const o = (f >> 2) * 4, fc = FACES[f & 3];
    let a = tets[o + fc[0]], b = tets[o + fc[1]], c = tets[o + fc[2]], x;
    if (a > b) { x = a; a = b; b = x; }
    if (b > c) { x = b; b = c; c = x; }
    if (a > b) { x = a; a = b; b = x; }
    srt[f * 3] = a; srt[f * 3 + 1] = b; srt[f * 3 + 2] = c; order[f] = f;
  }
  order.sort((f, g) => (srt[f * 3] - srt[g * 3]) || (srt[f * 3 + 1] - srt[g * 3 + 1]) || (srt[f * 3 + 2] - srt[g * 3 + 2]) || (f - g));
  const once = new Uint8Array(F);
  let K = 0;
  for (let s = 0; s < F;) {
    let e = s + 1;
    const f = order[s];
    while (e < F && srt[order[e] * 3] === srt[f * 3] && srt[order[e] * 3 + 1] === srt[f * 3 + 1] && srt[order[e] * 3 + 2] === srt[f * 3 + 2]) e++;
    if (e - s === 1) { once[f] = 1; K++; }
    s = e;
  }
  const surface = new Uint32Array(3 * K), surfaceTet = new Uint32Array(K);
  for (let f = 0, k = 0; f < F; f++) {
    if (!once[f]) continue;
    const o = (f >> 2) * 4, fc = FACES[f & 3];
    surface[k * 3] = tets[o + fc[0]]; surface[k * 3 + 1] = tets[o + fc[1]]; surface[k * 3 + 2] = tets[o + fc[2]];
    surfaceTet[k++] = f >> 2;
  }
  return {
    points, pointCount: P, tets, tetCount: m, edges, edgeCount: E, surface, surfaceCount: K, surfaceTet, cellTet,
    cells: [nx, ny, nz], size: [sx, sy, sz], origin: [ox, oy, oz], cellSize: [cx, cy, cz],
  };
}

// ---- the solver -----------------------------------------------------------------------------------------------------
/** XPBD tet-volume + global-volume constraints over the world's particles (a particle solver; see the header). */
export class VolumeConstraints {
  /**
   * @param {object|null} [world]  unused (kept for symmetry with VolumeConstraints.of; not retained)
   * @param {{ capacity?: number }} [o]
   */
  constructor(world = null, { capacity = 1024 } = {}) {
    this.name = 'softbody.volume';
    this.count = 0;
    this.pressure = [];
    this._alloc(Math.max(1, Math.floor(capacity)));
  }

  /** The world's registered VolumeConstraints (the first in world.particleSolvers), or null. Construction time only. */
  static find(world) {
    const sv = world.particleSolvers;
    for (let i = 0; i < sv.length; i++) if (sv[i] instanceof VolumeConstraints) return sv[i];
    return null;
  }

  /** The world's shared solver (find), created and registered when there is none. Construction time only. */
  static of(world, capacity = 1024) {
    let s = VolumeConstraints.find(world);
    if (!s) { s = new VolumeConstraints(null, { capacity }); world.addParticleSolver(s); }
    return s;
  }

  _alloc(n) {
    const old = this.capacity ? this : null;
    this.capacity = n;
    const grow = (a, len) => { const b = new a.constructor(len); if (old) b.set(a.subarray(0, Math.min(a.length, len))); return b; };
    if (old) {
      this.alive = grow(this.alive, n); this.tet = grow(this.tet, 4 * n); this.restVolume = grow(this.restVolume, n);
      this.compliance = grow(this.compliance, n); this.lambda = grow(this.lambda, n); this.force = grow(this.force, n);
      this._free = grow(this._free, n);
    } else {
      this.alive = new Uint8Array(n); this.tet = new Uint32Array(4 * n); this.restVolume = new Float64Array(n);
      this.compliance = new Float64Array(n); this.lambda = new Float64Array(n); this.force = new Float64Array(n);
      this._free = new Int32Array(n); this._freeCount = 0;
    }
  }

  /** Number of live tet constraints. */
  get size() { let s = 0; for (let i = 0; i < this.count; i++) s += this.alive[i]; return s; }

  /** True when no tet and no pressure constraint is alive. */
  isEmpty() {
    for (let i = 0; i < this.count; i++) if (this.alive[i]) return false;
    for (let i = 0; i < this.pressure.length; i++) if (this.pressure[i]) return false;
    return true;
  }

  /** Add a tet volume constraint over particles a b c d (positive orientation); returns its index. */
  addTet(a, b, c, d, restVolume, compliance = 0) {
    let id;
    if (this._freeCount > 0) id = this._free[--this._freeCount];
    else {
      if (this.count >= this.capacity) this._alloc(this.capacity * 2);
      id = this.count++;
    }
    const o = id * 4;
    this.alive[id] = 1; this.tet[o] = a; this.tet[o + 1] = b; this.tet[o + 2] = c; this.tet[o + 3] = d;
    this.restVolume[id] = +restVolume; this.compliance[id] = compliancePos(compliance, 'addTet');
    this.lambda[id] = 0; this.force[id] = 0;
    return id;
  }

  /** Remove tet constraint ti (its slot is reused, lowest first). */
  remove(ti) {
    if (!(ti >= 0 && ti < this.count) || !this.alive[ti]) return;
    this.alive[ti] = 0; this.lambda[ti] = 0; this.force[ti] = 0;
    const f = this._free;
    let i = this._freeCount++;
    while (i > 0 && f[i - 1] < ti) { f[i] = f[i - 1]; i--; } // descending: the lowest id comes back first
    f[i] = ti;
  }

  /**
   * Add a global volume constraint over the tets (Uint32Array(4m) of particle ids); returns its index (the lowest free
   * slot). restTets: the rest volume of each tet (default restVolume / m each): a tet with a freed vertex drops out of
   * the sum and its rest share out of restVolume.
   */
  addPressure(tets, restVolume, compliance = 0, restTets = null) {
    const rec = this._pressureRecord(tets, restVolume, compliance, restTets);
    let pi = this.pressure.indexOf(null);
    if (pi < 0) { pi = this.pressure.length; this.pressure.push(rec); } else this.pressure[pi] = rec;
    return pi;
  }

  _pressureRecord(tets, restVolume, compliance, restTets) {
    const T = Uint32Array.from(tets), m = T.length >> 2, ids = Array.from(new Set(T)).sort((x, y) => x - y);
    const particles = Uint32Array.from(ids), local = new Uint32Array(T.length);
    const at = new Map();
    for (let i = 0; i < ids.length; i++) at.set(ids[i], i);
    for (let i = 0; i < T.length; i++) local[i] = at.get(T[i]);
    const restTet = restTets ? Float64Array.from(restTets) : new Float64Array(m).fill(m > 0 ? +restVolume / m : 0);
    if (restTet.length !== m) throw new RangeError(`VolumeConstraints.addPressure: ${restTet.length} rest volumes for ${m} tets`);
    return {
      tets: T, particles, local, grad: new Float64Array(3 * particles.length), restTet,
      rest: +restVolume, compliance: compliancePos(compliance, 'addPressure'), state: new Float64Array(2),
    };
  }

  /** Remove global volume constraint pi. */
  removePressure(pi) { if (pi >= 0 && pi < this.pressure.length) this.pressure[pi] = null; }

  /** Particle-solver hook: the XPBD position pass (see the header). */
  solvePositions(world, h, iteration) {
    const Pt = world.particles, x = Pt.pos, im = Pt.invMass, pa = Pt.alive;
    const alive = this.alive, T = this.tet, V0 = this.restVolume, cpl = this.compliance, lam = this.lambda, F = this.force;
    const ih2 = 1 / (h * h), reset = iteration === 0;
    for (let t = 0, n = this.count; t < n; t++) {
      if (alive[t] === 0) continue;
      if (reset) lam[t] = 0;
      const k = t * 4, i0 = T[k], i1 = T[k + 1], i2 = T[k + 2], i3 = T[k + 3];
      if (pa[i0] === 0 || pa[i1] === 0 || pa[i2] === 0 || pa[i3] === 0) continue; // a freed vertex: the tet is gone
      const w0 = im[i0], w1 = im[i1], w2 = im[i2], w3 = im[i3];
      if (w0 + w1 + w2 + w3 === 0) continue;
      const a = i0 * 3, b = i1 * 3, c = i2 * 3, d = i3 * 3;
      const x0 = x[a], y0 = x[a + 1], z0 = x[a + 2], x1 = x[b], y1 = x[b + 1], z1 = x[b + 2];
      const x2 = x[c], y2 = x[c + 1], z2 = x[c + 2], x3 = x[d], y3 = x[d + 1], z3 = x[d + 2];
      const e1x = x1 - x0, e1y = y1 - y0, e1z = z1 - z0, e2x = x2 - x0, e2y = y2 - y0, e2z = z2 - z0;
      const e3x = x3 - x0, e3y = y3 - y0, e3z = z3 - z0;
      // grad_1 = e2 x e3 / 6, grad_2 = e3 x e1 / 6, grad_3 = e1 x e2 / 6, grad_0 = (p3 - p1) x (p2 - p1) / 6
      const g1x = (e2y * e3z - e2z * e3y) * SIXTH, g1y = (e2z * e3x - e2x * e3z) * SIXTH, g1z = (e2x * e3y - e2y * e3x) * SIXTH;
      const g2x = (e3y * e1z - e3z * e1y) * SIXTH, g2y = (e3z * e1x - e3x * e1z) * SIXTH, g2z = (e3x * e1y - e3y * e1x) * SIXTH;
      const g3x = (e1y * e2z - e1z * e2y) * SIXTH, g3y = (e1z * e2x - e1x * e2z) * SIXTH, g3z = (e1x * e2y - e1y * e2x) * SIXTH;
      const f1x = x3 - x1, f1y = y3 - y1, f1z = z3 - z1, f2x = x2 - x1, f2y = y2 - y1, f2z = z2 - z1;
      const g0x = (f1y * f2z - f1z * f2y) * SIXTH, g0y = (f1z * f2x - f1x * f2z) * SIXTH, g0z = (f1x * f2y - f1y * f2x) * SIXTH;
      const w = w0 * (g0x * g0x + g0y * g0y + g0z * g0z) + w1 * (g1x * g1x + g1y * g1y + g1z * g1z) +
        w2 * (g2x * g2x + g2y * g2y + g2z * g2z) + w3 * (g3x * g3x + g3y * g3y + g3z * g3z);
      const at = cpl[t] * ih2;
      if (!(w + at > 0)) continue; // collapsed tet with rigid compliance: no direction to push
      const C = (e1x * g1x + e1y * g1y + e1z * g1z) - V0[t];
      const l = lam[t], dl = (-C - at * l) / (w + at);
      lam[t] = l + dl; F[t] = (l + dl) * ih2;
      let s = w0 * dl; x[a] += s * g0x; x[a + 1] += s * g0y; x[a + 2] += s * g0z;
      s = w1 * dl; x[b] += s * g1x; x[b + 1] += s * g1y; x[b + 2] += s * g1z;
      s = w2 * dl; x[c] += s * g2x; x[c + 1] += s * g2y; x[c + 2] += s * g2z;
      s = w3 * dl; x[d] += s * g3x; x[d + 1] += s * g3y; x[d + 2] += s * g3z;
    }
    const PR = this.pressure;
    for (let r = 0; r < PR.length; r++) {
      const rec = PR[r];
      if (rec === null) continue;
      const st = rec.state;
      if (reset) st[0] = 0;
      const Tt = rec.tets, L = rec.local, G = rec.grad, ids = rec.particles, np = ids.length, m4 = Tt.length, RT = rec.restTet;
      G.fill(0);
      let V6 = 0, gone = 0;
      for (let k = 0; k < m4; k += 4) {
        const i0 = Tt[k], i1 = Tt[k + 1], i2 = Tt[k + 2], i3 = Tt[k + 3];
        if (pa[i0] === 0 || pa[i1] === 0 || pa[i2] === 0 || pa[i3] === 0) { gone += RT[k >> 2]; continue; }
        const a = i0 * 3, b = i1 * 3, c = i2 * 3, d = i3 * 3;
        const x0 = x[a], y0 = x[a + 1], z0 = x[a + 2];
        const e1x = x[b] - x0, e1y = x[b + 1] - y0, e1z = x[b + 2] - z0;
        const e2x = x[c] - x0, e2y = x[c + 1] - y0, e2z = x[c + 2] - z0;
        const e3x = x[d] - x0, e3y = x[d + 1] - y0, e3z = x[d + 2] - z0;
        const g1x = e2y * e3z - e2z * e3y, g1y = e2z * e3x - e2x * e3z, g1z = e2x * e3y - e2y * e3x;
        const g2x = e3y * e1z - e3z * e1y, g2y = e3z * e1x - e3x * e1z, g2z = e3x * e1y - e3y * e1x;
        const g3x = e1y * e2z - e1z * e2y, g3y = e1z * e2x - e1x * e2z, g3z = e1x * e2y - e1y * e2x;
        const f1x = x[d] - x[b], f1y = x[d + 1] - x[b + 1], f1z = x[d + 2] - x[b + 2];
        const f2x = x[c] - x[b], f2y = x[c + 1] - x[b + 1], f2z = x[c + 2] - x[b + 2];
        V6 += e1x * g1x + e1y * g1y + e1z * g1z;
        let o = L[k] * 3;
        G[o] += f1y * f2z - f1z * f2y; G[o + 1] += f1z * f2x - f1x * f2z; G[o + 2] += f1x * f2y - f1y * f2x;
        o = L[k + 1] * 3; G[o] += g1x; G[o + 1] += g1y; G[o + 2] += g1z;
        o = L[k + 2] * 3; G[o] += g2x; G[o + 1] += g2y; G[o + 2] += g2z;
        o = L[k + 3] * 3; G[o] += g3x; G[o + 1] += g3y; G[o + 2] += g3z;
      }
      // G holds 6 x the gradients
      let w = 0;
      for (let i = 0; i < np; i++) {
        const o = i * 3, gx = G[o], gy = G[o + 1], gz = G[o + 2];
        w += im[ids[i]] * (gx * gx + gy * gy + gz * gz);
      }
      w *= SIXTH * SIXTH;
      const at = rec.compliance * ih2;
      if (!(w + at > 0)) continue;
      const C = V6 * SIXTH - (rec.rest - gone), l = st[0], dl = (-C - at * l) / (w + at);
      st[0] = l + dl; st[1] = (l + dl) * ih2;
      const sdl = dl * SIXTH;
      for (let i = 0; i < np; i++) {
        const p = ids[i], s = im[p] * sdl, o = i * 3, q = p * 3;
        x[q] += s * G[o]; x[q + 1] += s * G[o + 1]; x[q + 2] += s * G[o + 2];
      }
    }
  }

  /** Plain snapshot (typed-array copies) of the whole solver state. */
  serialize() {
    const n = this.count;
    return {
      capacity: this.capacity, count: n, free: this._free.slice(0, this._freeCount),
      alive: this.alive.slice(0, n), tet: this.tet.slice(0, 4 * n), restVolume: this.restVolume.slice(0, n),
      compliance: this.compliance.slice(0, n), lambda: this.lambda.slice(0, n), force: this.force.slice(0, n),
      pressure: this.pressure.map((r) => (r ? {
        tets: r.tets.slice(), restTet: r.restTet.slice(), rest: r.rest, compliance: r.compliance, state: r.state.slice(),
      } : null)),
    };
  }

  /** Restore serialize()'s snapshot (replaces everything this solver holds). */
  restore(data) {
    if (!data) return;
    const n = data.count | 0;
    this.capacity = 0;
    this._alloc(Math.max(1, data.capacity | 0, n));
    this.count = n;
    this.alive.set(asArr(data.alive)); this.tet.set(asArr(data.tet)); this.restVolume.set(asArr(data.restVolume));
    this.compliance.set(asArr(data.compliance)); this.lambda.set(asArr(data.lambda)); this.force.set(asArr(data.force));
    const free = asArr(data.free);
    this._free.set(free); this._freeCount = free.length;
    // every record at its snapshot index (removed slots stay null, so the indices the soft bodies hold stay valid)
    this.pressure = [];
    for (const r of data.pressure || []) {
      if (!r) { this.pressure.push(null); continue; }
      const rec = this._pressureRecord(asArr(r.tets), r.rest, r.compliance, r.restTet == null ? null : asArr(r.restTet));
      rec.state.set(asArr(r.state));
      this.pressure.push(rec);
    }
  }
}

// ---- soft bodies ----------------------------------------------------------------------------------------------------
/** A tetrahedral soft body living in a DynamicsWorld's particle store (create with SoftBody.lattice). */
export class SoftBody {
  /** @private use SoftBody.lattice */
  constructor(world, topo, ids, group, solver) {
    this.world = world;
    this.topology = topo;
    this.ids = ids;
    this.first = ids.length ? ids[0] : -1;
    this.count = ids.length;
    this.group = group;
    this.solver = solver;
    const map = (src) => { const out = new Uint32Array(src.length); for (let i = 0; i < src.length; i++) out[i] = ids[src[i]]; return out; };
    this.tets = map(topo.tets); this.tetCount = topo.tetCount;
    this.edges = map(topo.edges); this.edgeCount = topo.edgeCount;
    this.surface = map(topo.surface); this.surfaceCount = topo.surfaceCount;
    const onSurf = new Uint8Array(topo.pointCount);
    for (let i = 0; i < topo.surface.length; i++) onSurf[topo.surface[i]] = 1;
    const sv = [];
    for (let i = 0; i < topo.pointCount; i++) if (onSurf[i]) sv.push(ids[i]);
    this.surfaceVertices = Int32Array.from(sv.sort((x, y) => x - y));
    this.edgeConstraints = new Int32Array(topo.edgeCount).fill(-1);
    this.tetConstraints = new Int32Array(topo.tetCount).fill(-1);
    this.pressureConstraint = -1;
    this.restVolume = 0;
    this.disposed = false;
  }

  /**
   * Build a lattice soft body into `world` (the header lists every option). Compliances: `edge` (m/N, default 1e-4),
   * `volume` (m^3/Pa per tet, default 0 = rigid, false = no tets), `pressure` (m^3/Pa, one global volume constraint,
   * default false = off, 0 = rigid); negative values are stored as 0, a non-finite one throws.
   * @param {object} world  a DynamicsWorld with particles (maxParticles > 0)
   * @param {{ shape?: 'box'|'sphere', size?: number|number[], radius?: number, resolution?: number|number[],
   *   position?: number[]|object, quaternion?: number[]|object, mass?: number, edge?: number, volume?: number|false,
   *   pressure?: number|false, particleRadius?: number, group?: number }} [o]
   * @returns {SoftBody}
   */
  static lattice(world, o = {}) {
    const P = world.particles, C = world.constraints;
    if (!P || !C) throw new TypeError('SoftBody.lattice: the world has no particle store (maxParticles)');
    const shape = o.shape ?? 'box';
    let res, size, inside = null;
    if (shape === 'sphere') {
      const R = o.radius ?? 0.5;
      if (!(R > 0)) throw new RangeError(`SoftBody.lattice: bad sphere radius ${R}`);
      res = vec3(o.resolution, [4, 4, 4]); size = [2 * R, 2 * R, 2 * R];
      const R2 = R * R;
      inside = (x, y, z) => x * x + y * y + z * z <= R2;
    } else if (shape === 'box') {
      res = vec3(o.resolution, [4, 4, 4]); size = vec3(o.size, [1, 1, 1]);
    } else throw new TypeError(`SoftBody.lattice: unknown shape '${shape}'`);
    const topo = tetrahedralize(res, inside, size);
    const n = topo.pointCount;
    if (n === 0) throw new RangeError('SoftBody.lattice: the shape keeps no cell (resolution too low?)');
    const cell = Math.min(topo.cellSize[0], topo.cellSize[1], topo.cellSize[2]);
    const pr = o.particleRadius ?? (shape === 'box' ? o.radius : undefined) ?? cell * 0.4;
    const mass = o.mass ?? 10;
    if (!(mass > 0)) throw new RangeError(`SoftBody.lattice: bad mass ${mass}`);
    const edgeC = complianceOpt(o.edge ?? 1e-4, 'edge', false), volC = complianceOpt(o.volume ?? 0, 'volume', true);
    const pressC = complianceOpt(o.pressure ?? false, 'pressure', true);
    const pos = vec3(o.position, [0, 0, 0]), q = quat4(o.quaternion);
    // particles (ascending, so contiguous in a fresh store); nothing stays allocated when the store is full
    const ids = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      let id = -1;
      try { id = P.alloc(); } catch { id = -1; }
      if (id < 0) {
        for (let j = 0; j < i; j++) P.free(ids[j]);
        throw new RangeError(`SoftBody.lattice: particle store full (${n} particles needed, ${i} free)`);
      }
      ids[i] = id;
    }
    const group = o.group ?? P.newGroup();
    const pm = mass / n, tmp = new Float64Array(3), L = topo.points;
    for (let i = 0; i < n; i++) {
      const id = ids[i], p3 = id * 3;
      rotate(q, 0, L[i * 3], L[i * 3 + 1], L[i * 3 + 2], tmp, 0);
      P.pos[p3] = pos[0] + tmp[0]; P.pos[p3 + 1] = pos[1] + tmp[1]; P.pos[p3 + 2] = pos[2] + tmp[2];
      P.prevPos[p3] = P.pos[p3]; P.prevPos[p3 + 1] = P.pos[p3 + 1]; P.prevPos[p3 + 2] = P.pos[p3 + 2];
      P.vel[p3] = 0; P.vel[p3 + 1] = 0; P.vel[p3 + 2] = 0;
      P.setMass(id, pm); P.radius[id] = pr; P.group[id] = group;
    }
    // the shared solver exists only once some soft body has a tet or pressure constraint
    const solver = volC !== false || pressC !== false ? VolumeConstraints.of(world) : null;
    const sb = new SoftBody(world, topo, ids, group, solver);
    const X = P.pos;
    // edges (one contiguous block of constraint ids)
    C.reserve(sb.edgeCount);
    for (let e = 0; e < sb.edgeCount; e++) {
      const a = sb.edges[e * 2], b = sb.edges[e * 2 + 1];
      const dx = X[a * 3] - X[b * 3], dy = X[a * 3 + 1] - X[b * 3 + 1], dz = X[a * 3 + 2] - X[b * 3 + 2];
      sb.edgeConstraints[e] = C.addDistance(a, b, Math.sqrt(dx * dx + dy * dy + dz * dz), edgeC);
    }
    // tets
    let rest = 0;
    const restTets = new Float64Array(sb.tetCount);
    for (let t = 0; t < sb.tetCount; t++) {
      const k = t * 4, a = sb.tets[k], b = sb.tets[k + 1], c = sb.tets[k + 2], d = sb.tets[k + 3];
      const v = vol6(X, a, b, c, d) * SIXTH;
      rest += v; restTets[t] = v;
      if (volC !== false) sb.tetConstraints[t] = solver.addTet(a, b, c, d, v, volC);
    }
    sb.restVolume = rest;
    if (pressC !== false) sb.pressureConstraint = solver.addPressure(sb.tets, rest, pressC, restTets);
    return sb;
  }

  /** Current total signed volume of the tets (m^3); allocation-free. */
  volume() {
    const x = this.world.particles.pos, T = this.tets;
    let v6 = 0;
    for (let k = 0, m = this.tetCount * 4; k < m; k += 4) {
      const a = T[k] * 3, b = T[k + 1] * 3, c = T[k + 2] * 3, d = T[k + 3] * 3;
      const x0 = x[a], y0 = x[a + 1], z0 = x[a + 2];
      const e1x = x[b] - x0, e1y = x[b + 1] - y0, e1z = x[b + 2] - z0;
      const e2x = x[c] - x0, e2y = x[c + 1] - y0, e2z = x[c + 2] - z0;
      const e3x = x[d] - x0, e3y = x[d + 1] - y0, e3z = x[d + 2] - z0;
      v6 += e1x * (e2y * e3z - e2z * e3y) + e1y * (e2z * e3x - e2x * e3z) + e1z * (e2x * e3y - e2y * e3x);
    }
    return v6 * SIXTH;
  }

  /** Mean particle position (every particle has the same mass) into out3 (array or {x, y, z}); returns out3. */
  centroid(out3) {
    const x = this.world.particles.pos, ids = this.ids, n = this.count;
    let sx = 0, sy = 0, sz = 0;
    for (let i = 0; i < n; i++) { const p = ids[i] * 3; sx += x[p]; sy += x[p + 1]; sz += x[p + 2]; }
    const s = n > 0 ? 1 / n : 0;
    // one store site per receiver kind: a site that sees several kinds stores through a generic IC, which boxes doubles
    if (out3 instanceof Float64Array) { out3[0] = sx * s; out3[1] = sy * s; out3[2] = sz * s; }
    else if (typeof out3.x === 'number') { out3.x = sx * s; out3.y = sy * s; out3.z = sz * s; }
    else { out3[0] = sx * s; out3[1] = sy * s; out3[2] = sz * s; }
    return out3;
  }

  /** Free the particles and remove every constraint of this body (the shared solver goes when it is empty). */
  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    const w = this.world, C = w.constraints, S = this.solver;
    for (let e = 0; e < this.edgeConstraints.length; e++) if (this.edgeConstraints[e] >= 0) C.remove(this.edgeConstraints[e]);
    if (S) {
      for (let t = 0; t < this.tetConstraints.length; t++) if (this.tetConstraints[t] >= 0) S.remove(this.tetConstraints[t]);
      if (this.pressureConstraint >= 0) S.removePressure(this.pressureConstraint);
    }
    for (let i = 0; i < this.count; i++) w.particles.free(this.ids[i]);
    this.edgeConstraints.fill(-1); this.tetConstraints.fill(-1); this.pressureConstraint = -1;
    if (S && S.isEmpty()) w.removeParticleSolver(S);
  }

  /**
   * Bind a render mesh to the tets (see the header).
   * @param {SoftBody} soft
   * @param {ArrayLike<number>} positions  rest positions (3v) in the lattice's local frame
   * @param {{ clamp?: boolean }} [o]
   * @returns {SoftSkin}
   */
  static skin(soft, positions, o = {}) { return new SoftSkin(soft, positions, o.clamp === true); }
}

// closest point on triangle (a, b, c) to p (Ericson, Real-Time Collision Detection 5.1.5) -> squared distance
function triDist2(px, py, pz, A, a, b, c) {
  const ax = A[a * 3], ay = A[a * 3 + 1], az = A[a * 3 + 2];
  const abx = A[b * 3] - ax, aby = A[b * 3 + 1] - ay, abz = A[b * 3 + 2] - az;
  const acx = A[c * 3] - ax, acy = A[c * 3 + 1] - ay, acz = A[c * 3 + 2] - az;
  const apx = px - ax, apy = py - ay, apz = pz - az;
  const d1 = abx * apx + aby * apy + abz * apz, d2 = acx * apx + acy * apy + acz * apz;
  let qx, qy, qz;
  const bpx = px - A[b * 3], bpy = py - A[b * 3 + 1], bpz = pz - A[b * 3 + 2];
  const d3 = abx * bpx + aby * bpy + abz * bpz, d4 = acx * bpx + acy * bpy + acz * bpz;
  const cpx = px - A[c * 3], cpy = py - A[c * 3 + 1], cpz = pz - A[c * 3 + 2];
  const d5 = abx * cpx + aby * cpy + abz * cpz, d6 = acx * cpx + acy * cpy + acz * cpz;
  const vc = d1 * d4 - d3 * d2, vb = d5 * d2 - d1 * d6, va = d3 * d6 - d5 * d4;
  if (d1 <= 0 && d2 <= 0) { qx = ax; qy = ay; qz = az; }
  else if (d3 >= 0 && d4 <= d3) { qx = ax + abx; qy = ay + aby; qz = az + abz; }
  else if (vc <= 0 && d1 >= 0 && d3 <= 0) { const v = d1 / (d1 - d3); qx = ax + v * abx; qy = ay + v * aby; qz = az + v * abz; }
  else if (d6 >= 0 && d5 <= d6) { qx = ax + acx; qy = ay + acy; qz = az + acz; }
  else if (vb <= 0 && d2 >= 0 && d6 <= 0) { const w = d2 / (d2 - d6); qx = ax + w * acx; qy = ay + w * acy; qz = az + w * acz; }
  else if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
    const w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    qx = A[b * 3] + w * (A[c * 3] - A[b * 3]); qy = A[b * 3 + 1] + w * (A[c * 3 + 1] - A[b * 3 + 1]); qz = A[b * 3 + 2] + w * (A[c * 3 + 2] - A[b * 3 + 2]);
  } else {
    const den = 1 / (va + vb + vc), v = vb * den, w = vc * den;
    qx = ax + abx * v + acx * w; qy = ay + aby * v + acy * w; qz = az + abz * v + acz * w;
  }
  const dx = px - qx, dy = py - qy, dz = pz - qz;
  return dx * dx + dy * dy + dz * dz;
}

// barycentric coordinates of p in tet t of the topology (local points) into out[o..o+4]; returns the smallest
function bary(topo, t, px, py, pz, out, o) {
  const A = topo.points, T = topo.tets, k = t * 4, a = T[k] * 3, b = T[k + 1] * 3, c = T[k + 2] * 3, d = T[k + 3] * 3;
  const x0 = A[a], y0 = A[a + 1], z0 = A[a + 2];
  const e1x = A[b] - x0, e1y = A[b + 1] - y0, e1z = A[b + 2] - z0;
  const e2x = A[c] - x0, e2y = A[c + 1] - y0, e2z = A[c + 2] - z0;
  const e3x = A[d] - x0, e3y = A[d + 1] - y0, e3z = A[d + 2] - z0;
  const rx = px - x0, ry = py - y0, rz = pz - z0;
  const c23x = e2y * e3z - e2z * e3y, c23y = e2z * e3x - e2x * e3z, c23z = e2x * e3y - e2y * e3x;
  const c31x = e3y * e1z - e3z * e1y, c31y = e3z * e1x - e3x * e1z, c31z = e3x * e1y - e3y * e1x;
  const c12x = e1y * e2z - e1z * e2y, c12y = e1z * e2x - e1x * e2z, c12z = e1x * e2y - e1y * e2x;
  const det = e1x * c23x + e1y * c23y + e1z * c23z, inv = 1 / det;
  const l1 = (rx * c23x + ry * c23y + rz * c23z) * inv, l2 = (rx * c31x + ry * c31y + rz * c31z) * inv;
  const l3 = (rx * c12x + ry * c12y + rz * c12z) * inv, l0 = 1 - l1 - l2 - l3;
  out[o] = l0; out[o + 1] = l1; out[o + 2] = l2; out[o + 3] = l3;
  return Math.min(l0, l1, l2, l3);
}

/** Barycentric binding of a render mesh to a soft body's tets (SoftBody.skin). */
export class SoftSkin {
  constructor(soft, positions, clamp = false) {
    const topo = soft.topology, v = Math.floor(positions.length / 3);
    this.soft = soft;
    this.vertexCount = v;
    this.tet = new Int32Array(v);
    this.ids = new Uint32Array(4 * v);
    this.weights = new Float64Array(4 * v);
    const [nx, ny, nz] = topo.cells, [ox, oy, oz] = topo.origin, [cx, cy, cz] = topo.cellSize;
    const tmp = new Float64Array(4), S = topo.surface, K = topo.surfaceCount, TT = topo.tets, m = topo.tetCount;
    // point -> incident tets (CSR, ascending tet index)
    const start = new Int32Array(topo.pointCount + 1), inc = new Int32Array(4 * m);
    for (let q = 0; q < 4 * m; q++) start[TT[q] + 1]++;
    for (let p = 0; p < topo.pointCount; p++) start[p + 1] += start[p];
    const fillAt = start.slice(0, topo.pointCount);
    for (let t = 0; t < m; t++) for (let k = 0; k < 4; k++) inc[fillAt[TT[t * 4 + k]]++] = t;
    for (let i = 0; i < v; i++) {
      const px = +positions[i * 3], py = +positions[i * 3 + 1], pz = +positions[i * 3 + 2];
      let best = -1;
      // the cell under the point, then its 6 tets
      const fi = Math.floor((px - ox) / cx), fj = Math.floor((py - oy) / cy), fk = Math.floor((pz - oz) / cz);
      const ci = Math.min(nx - 1, Math.max(0, fi)), cj = Math.min(ny - 1, Math.max(0, fj)), ck = Math.min(nz - 1, Math.max(0, fk));
      const t0 = (fi >= -1 && fi <= nx && fj >= -1 && fj <= ny && fk >= -1 && fk <= nz) ? topo.cellTet[ci + nx * (cj + ny * ck)] : -1;
      if (t0 >= 0) {
        let bm = -Infinity;
        for (let t = t0; t < t0 + 6; t++) { const m = bary(topo, t, px, py, pz, tmp, 0); if (m > bm) { bm = m; best = t; } }
        if (bm < -1e-9) best = -1;
      }
      if (best < 0) { // outside the lattice: the nearest surface triangle, then the least extrapolating tet around it
        let bd = Infinity, tri = -1;
        for (let k = 0; k < K; k++) {
          const d2 = triDist2(px, py, pz, topo.points, S[k * 3], S[k * 3 + 1], S[k * 3 + 2]);
          if (d2 < bd) { bd = d2; tri = k; }
        }
        if (tri >= 0) {
          // smallest sum |w| (= 1 - 2 sum of the negative weights: the factor by which a perturbation of the tet's
          // vertices reaches the vertex), ties to the largest minimum weight; first found wins a full tie
          best = topo.surfaceTet[tri];
          let bm = bary(topo, best, px, py, pz, tmp, 0), bl = Math.abs(tmp[0]) + Math.abs(tmp[1]) + Math.abs(tmp[2]) + Math.abs(tmp[3]);
          for (let c = 0; c < 3; c++) {
            const p = S[tri * 3 + c];
            for (let j = start[p]; j < start[p + 1]; j++) {
              const t = inc[j], mb = bary(topo, t, px, py, pz, tmp, 0);
              const l1 = Math.abs(tmp[0]) + Math.abs(tmp[1]) + Math.abs(tmp[2]) + Math.abs(tmp[3]);
              if (l1 < bl || (l1 === bl && mb > bm)) { bl = l1; bm = mb; best = t; }
            }
          }
        }
      }
      if (best < 0) best = 0;
      bary(topo, best, px, py, pz, tmp, 0);
      if (clamp) {
        let s = 0;
        for (let k = 0; k < 4; k++) { if (!(tmp[k] > 0)) tmp[k] = 0; s += tmp[k]; }
        if (s > 0) for (let k = 0; k < 4; k++) tmp[k] /= s; else tmp.fill(0.25);
      }
      this.tet[i] = best;
      for (let k = 0; k < 4; k++) { this.ids[i * 4 + k] = soft.tets[best * 4 + k]; this.weights[i * 4 + k] = tmp[k]; }
    }
  }

  /** World-space skinned positions from the current particle positions into out (3v); allocation-free. */
  update(out) {
    if (out instanceof Float32Array) return this._update32(out);
    const x = this.soft.world.particles.pos, I = this.ids, W = this.weights;
    for (let i = 0, n = this.vertexCount; i < n; i++) {
      const k = i * 4, a = I[k] * 3, b = I[k + 1] * 3, c = I[k + 2] * 3, d = I[k + 3] * 3;
      const w0 = W[k], w1 = W[k + 1], w2 = W[k + 2], w3 = W[k + 3], o = i * 3;
      out[o] = w0 * x[a] + w1 * x[b] + w2 * x[c] + w3 * x[d];
      out[o + 1] = w0 * x[a + 1] + w1 * x[b + 1] + w2 * x[c + 1] + w3 * x[d + 1];
      out[o + 2] = w0 * x[a + 2] + w1 * x[b + 2] + w2 * x[c + 2] + w3 * x[d + 2];
    }
    return out;
  }

  // the Float32Array (BufferAttribute) case with its own monomorphic store sites
  _update32(out) {
    const x = this.soft.world.particles.pos, I = this.ids, W = this.weights;
    for (let i = 0, n = this.vertexCount; i < n; i++) {
      const k = i * 4, a = I[k] * 3, b = I[k + 1] * 3, c = I[k + 2] * 3, d = I[k + 3] * 3;
      const w0 = W[k], w1 = W[k + 1], w2 = W[k + 2], w3 = W[k + 3], o = i * 3;
      out[o] = w0 * x[a] + w1 * x[b] + w2 * x[c] + w3 * x[d];
      out[o + 1] = w0 * x[a + 1] + w1 * x[b + 1] + w2 * x[c + 1] + w3 * x[d + 1];
      out[o + 2] = w0 * x[a + 2] + w1 * x[b + 2] + w2 * x[c + 2] + w3 * x[d + 2];
    }
    return out;
  }
}
