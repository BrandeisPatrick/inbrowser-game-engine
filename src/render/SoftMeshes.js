// Render-layer meshes for the particle objects of the dynamics world (physics/dynamics/particles.js ParticleStore:
// pos / alive / group / radius, Float64 positions, 3 per particle): cloth grids, rope tubes, soft-body surfaces and
// plain particle spheres. Each one allocates its BufferAttributes (or InstancedPools) once and rewrites them in place
// (Float32 positions, normals, needsUpdate): update() allocates nothing and never rebuilds a geometry.
//
//   const cm = new ClothMesh(world.particles, cloth, material);   scene.add(cm.mesh);
//   after each fixed step: cm.capture();   per frame: cm.update(alpha);   cm.dispose();
//
// Render interpolation (as render/Fracture.js FractureMesh and the sandbox's BodyPools): capture() after each fixed
// step keeps the previous and the current vertex positions (previous <- current, current <- the store; the first
// capture sets both); update(alpha) draws previous + (current - previous) * alpha. Before the first capture() update()
// draws the store's positions as they are (alpha ignored).
//
// - ClothMesh(store, cloth, material): one indexed grid (cloth.nx x cloth.ny vertices, vertex i + j * nx = particle
//   cloth.indexOf(i, j), uv (i / (nx - 1), j / (ny - 1)), two triangles per cell). Smooth normals from an
//   allocation-free area-weighted pass over the index (not computeVertexNormals: it allocates temporaries per call).
//   Meant for a double-sided material (three flips the normal on back faces).
// - SoftBodyMesh(store, soft, material, { inflate, geometry }): an indexed mesh over soft.surface (boundary
//   triangles as particle ids, counter-clockwise seen from outside) whose vertices are the surface particles
//   (soft.surfaceVertices, or the distinct ids of the surface), smooth normals recomputed in place; `inflate` (m,
//   default: the particle radius of the body's first surface particle) pushes every vertex out along its normal so the
//   drawn surface is where the particles collide (the lattice points sit one particle radius inside it).
//   With `geometry` (a BufferGeometry in the lattice's LOCAL rest frame, e.g. a SphereGeometry over a voxel sphere) the
//   mesh draws that geometry skinned to the tets instead (physics SoftBody.skin: barycentric weights, linear in the
//   particle positions, so interpolating the skinned vertices is exact); vertices at equal rest positions (UV seams)
//   share one normal. inflate defaults to 0 then.
// - RopeMesh(store, rope, { radius, radialSegments = 6, material, joints = true, pairs = false }): a tube as instanced
//   unit cylinders (one InstancedPool, one per segment, oriented from particle k to k + 1 with the shortest-arc rotation
//   from +Y: q = (d.z, 0, -d.x, 1 + d.y) normalised, no lookAt) plus, with `joints`, one sphere of the same radius
//   at every particle (a second pool) so bends read as a continuous tube. `mesh` is a Group holding both pools.
//   pairs: the segments are (ids[0], ids[1]), (ids[2], ids[3]), ... instead of a chain (rungs, cross ties); `rope`
//   may then be any { ids, count }.
// - ParticleMesh(store, { group, radiusScale = 1, material, color, segments = 12, cullRadius = 0.5 }): every alive
//   particle (of `group`, when given) as a sphere of its radius (InstancedPool: debug view, ropes without a tube); the
//   instance count follows the store.
// Bounds: the surface meshes keep their own bounding sphere / box from the positions they wrote, the instanced ones
// their pools' (InstancedPool.updateBounds), so frustum and shadow culling stay right without frustumCulled = false.
// Materials passed in stay the caller's (dispose() frees geometry and pools only); a default material made here
// (no `material` given) is disposed with the mesh. Like the dynamics world these read, they assume the in-thread
// world (DynamicsClient worker mode has no particle store).
import * as THREE from 'three';
import { InstancedPool } from './InstancedPool.js';
import { SoftBody } from '../physics/dynamics/softbody.js';

const _m = new THREE.Matrix4(), _p = new THREE.Vector3(), _q = new THREE.Quaternion(), _s = new THREE.Vector3();

/** out[3v..] <- the store's current position of particle ids[v]. */
function readParticles(pos, ids, n, out) {
  for (let v = 0; v < n; v++) { const k = ids[v] * 3, o = v * 3; out[o] = pos[k]; out[o + 1] = pos[k + 1]; out[o + 2] = pos[k + 2]; }
}
/** out <- a + (b - a) * alpha. */
function lerp(a, b, alpha, out) {
  if (alpha >= 1) { out.set(b); return; }
  for (let i = 0, n = out.length; i < n; i++) out[i] = a[i] + (b[i] - a[i]) * alpha;
}
/**
 * Area-weighted smooth vertex normals of the indexed triangles, (b - a) x (c - a), normalised (degenerate: +Y).
 * `index` may map duplicated vertices to one canonical vertex (weld): then copy[] (vertex -> canonical, or null)
 * spreads the canonical normals back.
 */
function vertexNormals(p, index, nrm, weld) {
  nrm.fill(0);
  for (let t = 0, e = index.length; t < e; t += 3) {
    const a = index[t] * 3, b = index[t + 1] * 3, c = index[t + 2] * 3;
    const abx = p[b] - p[a], aby = p[b + 1] - p[a + 1], abz = p[b + 2] - p[a + 2];
    const acx = p[c] - p[a], acy = p[c + 1] - p[a + 1], acz = p[c + 2] - p[a + 2];
    const nx = aby * acz - abz * acy, ny = abz * acx - abx * acz, nz = abx * acy - aby * acx;
    nrm[a] += nx; nrm[a + 1] += ny; nrm[a + 2] += nz;
    nrm[b] += nx; nrm[b + 1] += ny; nrm[b + 2] += nz;
    nrm[c] += nx; nrm[c + 1] += ny; nrm[c + 2] += nz;
  }
  for (let i = 0, e = nrm.length; i < e; i += 3) {
    const x = nrm[i], y = nrm[i + 1], z = nrm[i + 2], l = Math.sqrt(x * x + y * y + z * z);
    if (l > 1e-20) { nrm[i] = x / l; nrm[i + 1] = y / l; nrm[i + 2] = z / l; } else { nrm[i] = 0; nrm[i + 1] = 1; nrm[i + 2] = 0; }
  }
  if (weld !== null) {
    for (let v = 0, n = weld.length; v < n; v++) {
      const w = weld[v];
      if (w !== v) { nrm[v * 3] = nrm[w * 3]; nrm[v * 3 + 1] = nrm[w * 3 + 1]; nrm[v * 3 + 2] = nrm[w * 3 + 2]; }
    }
  }
}
/** Bounding box + sphere of `geo` from the packed positions, grown by `pad` (reused objects, allocation-free). */
function applyBounds(geo, p, pad) {
  let x0 = Infinity, y0 = Infinity, z0 = Infinity, x1 = -Infinity, y1 = -Infinity, z1 = -Infinity;
  for (let i = 0, n = p.length; i < n; i += 3) {
    const x = p[i], y = p[i + 1], z = p[i + 2];
    if (x < x0) x0 = x; if (y < y0) y0 = y; if (z < z0) z0 = z;
    if (x > x1) x1 = x; if (y > y1) y1 = y; if (z > z1) z1 = z;
  }
  const bb = geo.boundingBox, bs = geo.boundingSphere;
  if (!(x0 <= x1)) { bb.makeEmpty(); bs.center.set(0, 0, 0); bs.radius = -1; return; }
  bb.min.set(x0 - pad, y0 - pad, z0 - pad); bb.max.set(x1 + pad, y1 + pad, z1 + pad);
  bs.center.set((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2);
  const dx = x1 - x0, dy = y1 - y0, dz = z1 - z0;
  bs.radius = Math.sqrt(dx * dx + dy * dy + dz * dz) / 2 + pad;
}
/** Particle ids of an object: its `ids` array, else first .. first + count - 1. */
function idsOf(obj) {
  if (obj.ids && obj.ids.length >= obj.count) return obj.ids;
  const ids = new Uint32Array(obj.count);
  for (let k = 0; k < obj.count; k++) ids[k] = obj.first + k;
  return ids;
}
const indexArray = (n, vertices) => (vertices > 65535 ? new Uint32Array(n) : new Uint16Array(n));

/** Shared state of the indexed surface meshes (cloth, soft body): `read(out)` fills the current vertex positions. */
class SurfaceMesh {
  constructor(vertexCount, index, uv, material, name, read, o = {}) {
    this.count = vertexCount;
    this._read = read;
    this.positions = new Float32Array(vertexCount * 3);
    this.normals = new Float32Array(vertexCount * 3);
    this.prev = null; this.cur = null;   // Float32Array(3n) each, after the first capture()
    this.inflate = o.inflate ?? 0;       // m along the vertex normal
    this.weld = o.weld ?? null;          // Int32Array vertex -> canonical vertex (index is welded), or null
    this.index = o.normalIndex ?? index; // the index the normals are accumulated over (welded when weld is set)
    const geo = this.geometry = new THREE.BufferGeometry();
    this.positionAttr = new THREE.BufferAttribute(this.positions, 3).setUsage(THREE.DynamicDrawUsage);
    this.normalAttr = new THREE.BufferAttribute(this.normals, 3).setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute('position', this.positionAttr);
    geo.setAttribute('normal', this.normalAttr);
    if (uv) geo.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
    geo.setIndex(new THREE.BufferAttribute(index, 1));
    geo.boundingBox = new THREE.Box3();
    geo.boundingSphere = new THREE.Sphere();
    this.mesh = new THREE.Mesh(geo, material);
    this.mesh.name = name;
    this.mesh.castShadow = true; this.mesh.receiveShadow = true;
    this.update(1);
  }
  /** After each fixed step: previous <- current, current <- the store (the first call sets both). */
  capture() {
    if (this.cur === null) { this.cur = new Float32Array(this.count * 3); this._read(this.cur); this.prev = this.cur.slice(); return; }
    const t = this.prev; this.prev = this.cur; this.cur = t;
    this._read(this.cur);
  }
  /** Per frame: positions (interpolated by alpha, inflated), normals, bounds. */
  update(alpha = 1) {
    const p = this.positions, nrm = this.normals;
    if (this.cur === null) this._read(p); else lerp(this.prev, this.cur, alpha, p);
    vertexNormals(p, this.index, nrm, this.weld);
    const f = this.inflate;
    if (f !== 0) for (let i = 0, n = p.length; i < n; i++) p[i] += nrm[i] * f;
    this.positionAttr.needsUpdate = true;
    this.normalAttr.needsUpdate = true;
    applyBounds(this.geometry, p, 0);
  }
  dispose() {
    if (this.mesh.parent) this.mesh.parent.remove(this.mesh);
    this.geometry.dispose();
  }
}

// ------------------------------------------------------------------ cloth
export class ClothMesh extends SurfaceMesh {
  /**
   * @param {object} store  ParticleStore (world.particles)
   * @param {object} cloth  Cloth.grid result ({ nx, ny, indexOf(i, j) | first })
   * @param {THREE.Material} material  double-sided (THREE.DoubleSide) lit material
   */
  constructor(store, cloth, material) {
    const nx = cloth.nx, ny = cloth.ny, n = nx * ny;
    const ids = new Uint32Array(n), uv = new Float32Array(n * 2);
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const v = i + j * nx;
        ids[v] = typeof cloth.indexOf === 'function' ? cloth.indexOf(i, j) : cloth.first + v;
        uv[v * 2] = nx > 1 ? i / (nx - 1) : 0; uv[v * 2 + 1] = ny > 1 ? j / (ny - 1) : 0;
      }
    }
    const index = indexArray((nx - 1) * (ny - 1) * 6, n);
    let t = 0;
    for (let j = 0; j < ny - 1; j++) {
      for (let i = 0; i < nx - 1; i++) {
        const a = i + j * nx, b = a + 1, c = a + nx, d = c + 1;
        index[t++] = a; index[t++] = b; index[t++] = d;
        index[t++] = a; index[t++] = d; index[t++] = c;
      }
    }
    super(n, index, uv, material, 'clothMesh', (out) => readParticles(store.pos, ids, n, out));
    this.store = store; this.cloth = cloth; this.ids = ids;
    this.nx = nx; this.ny = ny;
  }
}

// ------------------------------------------------------------------ soft body
export class SoftBodyMesh extends SurfaceMesh {
  /**
   * @param {object} store  ParticleStore (world.particles)
   * @param {object} soft  SoftBody.lattice result ({ surface: Uint32Array(3k) particle ids, surfaceVertices })
   * @param {THREE.Material} material
   * @param {object} [o]  inflate (m), geometry (BufferGeometry in the lattice's local rest frame: skinned instead)
   */
  constructor(store, soft, material, o = {}) {
    if (o.geometry) {
      const g = o.geometry, rest = g.getAttribute('position'), nv = rest.count;
      const restArr = new Float64Array(nv * 3);
      for (let v = 0; v < nv; v++) { restArr[v * 3] = rest.getX(v); restArr[v * 3 + 1] = rest.getY(v); restArr[v * 3 + 2] = rest.getZ(v); }
      const skin = SoftBody.skin(soft, restArr);
      let index;
      if (g.index) { index = indexArray(g.index.count, nv); for (let k = 0; k < g.index.count; k++) index[k] = g.index.getX(k); }
      else { index = indexArray(nv, nv); for (let k = 0; k < nv; k++) index[k] = k; }
      // weld equal rest positions (UV seams, poles) so they share one normal
      const weld = new Int32Array(nv), first = new Map();
      for (let v = 0; v < nv; v++) {
        const key = `${restArr[v * 3].toFixed(5)},${restArr[v * 3 + 1].toFixed(5)},${restArr[v * 3 + 2].toFixed(5)}`;
        const w = first.get(key);
        if (w === undefined) { first.set(key, v); weld[v] = v; } else weld[v] = w;
      }
      const normalIndex = indexArray(index.length, nv);
      for (let k = 0; k < index.length; k++) normalIndex[k] = weld[index[k]];
      const uvAttr = g.getAttribute('uv');
      let uv = null;
      if (uvAttr) { uv = new Float32Array(nv * 2); for (let v = 0; v < nv; v++) { uv[v * 2] = uvAttr.getX(v); uv[v * 2 + 1] = uvAttr.getY(v); } }
      super(nv, index, uv, material, 'softBodyMesh', (out) => skin.update(out), { inflate: o.inflate ?? 0, weld, normalIndex });
      this.skin = skin;
    } else {
      const surf = soft.surface;
      let verts = soft.surfaceVertices;
      if (!verts || !verts.length) { // the distinct particle ids of the surface, ascending
        const seen = new Map();
        for (let k = 0; k < surf.length; k++) seen.set(surf[k], 1);
        verts = Uint32Array.from(seen.keys()).sort();
      }
      const ids = Uint32Array.from(verts), nv = ids.length;
      const local = new Map();           // particle id -> vertex index (build time only)
      for (let v = 0; v < nv; v++) local.set(ids[v], v);
      const index = indexArray(surf.length, nv);
      for (let k = 0; k < surf.length; k++) {
        const v = local.get(surf[k]);
        if (v === undefined) throw new Error(`SoftBodyMesh: surface particle ${surf[k]} is not in surfaceVertices`);
        index[k] = v;
      }
      const inflate = o.inflate ?? (nv > 0 && store.radius ? store.radius[ids[0]] : 0);
      super(nv, index, null, material, 'softBodyMesh', (out) => readParticles(store.pos, ids, nv, out), { inflate });
      this.ids = ids;
      this.skin = null;
    }
    this.store = store; this.soft = soft;
  }
}

// ------------------------------------------------------------------ rope
export class RopeMesh {
  /**
   * @param {object} store  ParticleStore (world.particles)
   * @param {object} rope  Rope.chain result ({ ids | first, count, length })
   * @param {object} [o]  radius (m, default 0.03), radialSegments (6), material, joints (true: a sphere at every
   *   particle), castShadow (true), pairs (false: segments k -> k + 1; true: 2k -> 2k + 1)
   */
  constructor(store, rope, o = {}) {
    this.store = store;
    this.rope = rope;
    this.ids = idsOf(rope);
    this.count = this.ids.length;
    this.radius = o.radius ?? 0.03;
    const rs = Math.max(3, o.radialSegments ?? 6), joints = o.joints ?? true, shadow = o.castShadow !== false;
    this._ownMaterial = !o.material;
    this.material = o.material || new THREE.MeshStandardMaterial({ color: 0xc9b48a, roughness: 0.85, metalness: 0 });
    this.pairs = !!o.pairs;
    const segs = this.pairs ? this.count >> 1 : Math.max(0, this.count - 1);
    this.segments = segs;
    // one instance reaches half a segment (+ the radius) from its centre; the cull radius takes a whole laid-out
    // segment (+ the radius), so segments stretched up to twice their rest length stay inside it
    const segLen = segs > 0 && rope.length > 0 ? rope.length / segs : 0.5;
    this._cyl = new THREE.CylinderGeometry(1, 1, 1, rs, 1, joints); // open ends under the joint spheres
    this.pool = new InstancedPool(this._cyl, this.material, { capacity: Math.max(1, segs), castShadow: shadow, cullRadius: this.radius + segLen, name: 'ropeMesh.segments' });
    this.pool.setCount(segs);
    this.mesh = new THREE.Group();
    this.mesh.name = 'ropeMesh';
    this.mesh.add(this.pool.mesh);
    this.jointPool = null; this._sph = null;
    if (joints) {
      this._sph = new THREE.SphereGeometry(1, rs, Math.max(3, (rs >> 1) + 1));
      this.jointPool = new InstancedPool(this._sph, this.material, { capacity: Math.max(1, this.count), castShadow: shadow, cullRadius: this.radius, name: 'ropeMesh.joints' });
      this.jointPool.setCount(this.count);
      this.mesh.add(this.jointPool.mesh);
    }
    this.points = new Float32Array(this.count * 3);
    this.prev = null; this.cur = null;
    this.update(1);
  }
  /** After each fixed step: previous <- current, current <- the store (the first call sets both). */
  capture() {
    if (this.cur === null) { this.cur = new Float32Array(this.count * 3); readParticles(this.store.pos, this.ids, this.count, this.cur); this.prev = this.cur.slice(); return; }
    const t = this.prev; this.prev = this.cur; this.cur = t;
    readParticles(this.store.pos, this.ids, this.count, this.cur);
  }
  /** Per frame: segment and joint instances from the (interpolated) particle positions. */
  update(alpha = 1) {
    const p = this.points, r = this.radius, n = this.count;
    if (this.cur === null) readParticles(this.store.pos, this.ids, n, p); else lerp(this.prev, this.cur, alpha, p);
    const P = this.pool, segs = this.segments, step = this.pairs ? 6 : 3;
    for (let k = 0; k < segs; k++) {
      const a = k * step, b = a + 3;
      let dx = p[b] - p[a], dy = p[b + 1] - p[a + 1], dz = p[b + 2] - p[a + 2];
      const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (len > 1e-9) { dx /= len; dy /= len; dz /= len; } else { dx = 0; dy = 1; dz = 0; }
      // shortest arc +Y -> d: (cross(Y, d), 1 + dot(Y, d)) = (dz, 0, -dx, 1 + dy); d = -Y: half turn about X
      if (dy < -0.999999) _q.set(1, 0, 0, 0);
      else { const w = 1 + dy, l = Math.sqrt(dz * dz + dx * dx + w * w); _q.set(dz / l, 0, -dx / l, w / l); }
      _p.set((p[a] + p[b]) / 2, (p[a + 1] + p[b + 1]) / 2, (p[a + 2] + p[b + 2]) / 2);
      _m.compose(_p, _q, _s.set(r, len, r));
      P.setMatrix(k, _m);
    }
    if (segs > 0) P.updateBounds();
    P.mesh.visible = segs > 0;
    const J = this.jointPool;
    if (J !== null) {
      for (let k = 0; k < n; k++) J.setTRS(k, p[k * 3], p[k * 3 + 1], p[k * 3 + 2], 0, r);
      if (n > 0) J.updateBounds();
      J.mesh.visible = n > 0;
    }
  }
  dispose() {
    if (this.mesh.parent) this.mesh.parent.remove(this.mesh);
    this.pool.dispose(); this._cyl.dispose();
    if (this.jointPool !== null) { this.jointPool.dispose(); this._sph.dispose(); }
    if (this._ownMaterial) this.material.dispose();
  }
}

// ------------------------------------------------------------------ plain particles
export class ParticleMesh {
  /**
   * @param {object} store  ParticleStore (world.particles: alive, group, pos, radius, count)
   * @param {object} [o]  group (only that group; default every alive particle), radiusScale (1), material, color
   *   (default material's colour), segments (12: sphere width segments), castShadow (true), cullRadius (0.5: the
   *   largest drawn sphere radius)
   */
  constructor(store, o = {}) {
    this.store = store;
    this.group = o.group ?? null;
    this.radiusScale = o.radiusScale ?? 1;
    this._ownMaterial = !o.material;
    this.material = o.material || new THREE.MeshStandardMaterial({ color: o.color ?? 0xe6dccb, roughness: 0.6, metalness: 0 });
    const seg = Math.max(4, o.segments ?? 12);
    this._sph = new THREE.SphereGeometry(1, seg, Math.max(3, (seg >> 1) + 1));
    this.pool = new InstancedPool(this._sph, this.material, { capacity: 64, castShadow: o.castShadow !== false, cullRadius: o.cullRadius ?? 0.5, name: 'particleMesh' });
    this.mesh = this.pool.mesh;
    this.prev = null; this.cur = null;   // Float32Array(3 * size) each, by particle index, after capture()
    this.captured = null;                // Uint8Array: 1 = prev / cur hold this particle (captured last time)
    this.update(1);
  }
  _in(p) { const s = this.store; return s.alive[p] && (this.group === null || s.group[p] === this.group); }
  /** After each fixed step: previous <- current, current <- the store, per particle (new ones: both). */
  capture() {
    const s = this.store, n = s.count, pos = s.pos;
    if (this.cur === null || this.captured.length < n) { // sized to the store's capacity once (its high-water mark)
      const size = Math.max(64, n, (s.alive && s.alive.length) || 0);
      this.prev = new Float32Array(size * 3); this.cur = new Float32Array(size * 3); this.captured = new Uint8Array(size);
    }
    const prev = this.prev, cur = this.cur, got = this.captured;
    for (let p = 0; p < n; p++) {
      if (!this._in(p)) { got[p] = 0; continue; }
      const k = p * 3;
      if (got[p]) { prev[k] = cur[k]; prev[k + 1] = cur[k + 1]; prev[k + 2] = cur[k + 2]; }
      else { prev[k] = pos[k]; prev[k + 1] = pos[k + 1]; prev[k + 2] = pos[k + 2]; }
      cur[k] = pos[k]; cur[k + 1] = pos[k + 1]; cur[k + 2] = pos[k + 2]; got[p] = 1;
    }
    for (let p = n, e = got.length; p < e; p++) got[p] = 0;
  }
  update(alpha = 1) {
    const s = this.store, n = s.count, pos = s.pos, rad = s.radius, P = this.pool;
    let m = 0;
    for (let p = 0; p < n; p++) if (this._in(p)) m++;
    P.setCount(m);
    const prev = this.prev, cur = this.cur, got = this.captured, rs = this.radiusScale;
    let i = 0;
    for (let p = 0; p < n && i < m; p++) {
      if (!this._in(p)) continue;
      const k = p * 3;
      let x = pos[k], y = pos[k + 1], z = pos[k + 2];
      if (cur !== null && p < got.length && got[p]) { x = prev[k] + (cur[k] - prev[k]) * alpha; y = prev[k + 1] + (cur[k + 1] - prev[k + 1]) * alpha; z = prev[k + 2] + (cur[k + 2] - prev[k + 2]) * alpha; }
      P.setTRS(i++, x, y, z, 0, rad[p] * rs);
    }
    if (m > 0) P.updateBounds();
    P.mesh.visible = m > 0;
  }
  dispose() {
    if (this.mesh.parent) this.mesh.parent.remove(this.mesh);
    this.pool.dispose(); this._sph.dispose();
    if (this._ownMaterial) this.material.dispose();
  }
}
