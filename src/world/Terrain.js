// Terrain: real elevation heightfield + the ground surface the world is built on + polygon draping.
//
// `Terrain` loads a terrain.json (contract below; baked by engine/tools/bake-terrain.mjs from AWS Terrarium tiles):
// heights in MSL metres, stored in the city.json BAKE frame. MapWorld shifts all map data by
// `world.offset` so spawn is (0,0); call `setOffset(world.offset)` and query in WORLD coordinates.
// Water stays a flat plane at the world's water level (-0.6 by default), so terrain y == metres above sea level.
//   terrain.json: { origin: [x0, z0], cell, nx, nz, h: Int16 decimetres row-major (z rows, then x),
//                   outer?: { origin, cell, nx, nz, h } coarse grid, source?, datum?, frame? }
//
// `GroundSurface` is the height field the visible ground is actually built from: a nested lattice
// (5 m in the detailed district, 10 m over the filler ring, 20 m to the horizon) initialised from the
// terrain, with ground-level road corridors flattened to their smoothed centre-line profile (flat
// cross-slope, gentle intersections). It is sampled with the SAME two-triangles-per-cell interpolation
// the ground mesh uses, so heightAt() is exactly the rendered ground. Once attached
// (`terrain.attachSurface`) Terrain.heightAt returns the surface; Terrain.rawHeightAt the plain DEM.
import * as THREE from 'three';
import { clamp } from './geom.js';

const smooth01 = (t) => (t <= 0 ? 0 : t >= 1 ? 1 : t * t * (3 - 2 * t));

export class Terrain {
  constructor() {
    this.ready = false;
    this.offset = { x: 0, z: 0 };
    this.fine = null;   // {x0,z0,cell,nx,nz,h:Float32Array (metres)}
    this.outer = null;
    this.surface = null;
    this.info = null;
  }

  /** Fetch terrain.json. Never throws: on failure the terrain stays flat (heightAt -> 0). */
  async load(url) {
    try {
      if (!url) {
        const base = (import.meta.env && import.meta.env.BASE_URL) || '/';
        url = `${base}map/terrain.json`;
      }
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      this.setData(await res.json());
    } catch (e) {
      console.warn('[Terrain] terrain.json unavailable, using flat ground:', e.message || e);
      this.ready = false;
    }
    return this;
  }

  setData(j) {
    const grid = (g) => {
      if (!g || !Array.isArray(g.h) || !(g.nx > 1) || !(g.nz > 1) || g.h.length !== g.nx * g.nz) return null;
      const h = new Float32Array(g.h.length);
      for (let i = 0; i < h.length; i++) h[i] = g.h[i] / 10;
      return { x0: g.origin[0], z0: g.origin[1], cell: g.cell, nx: g.nx, nz: g.nz, h, x1: g.origin[0] + (g.nx - 1) * g.cell, z1: g.origin[1] + (g.nz - 1) * g.cell };
    };
    this.fine = grid(j);
    this.outer = grid(j.outer);
    this.info = { source: j.source, datum: j.datum, frame: j.frame };
    this.ready = !!(this.fine || this.outer);
  }

  setOffset(o) { this.offset = { x: (o && o.x) || 0, z: (o && o.z) || 0 }; }
  attachSurface(s) { this.surface = s; }

  /** DEM height (MSL metres, bilinear, clamped at the edges) at WORLD (x,z). 0 when not loaded. */
  rawHeightAt(x, z) {
    if (!this.ready) return 0;
    const bx = x + this.offset.x, bz = z + this.offset.z;
    const F = this.fine, O = this.outer;
    if (F && bx >= F.x0 && bx <= F.x1 && bz >= F.z0 && bz <= F.z1) {
      const hf = bilinear(F, bx, bz);
      if (!O) return hf;
      // blend into the coarse grid over the last 150 m so there is no seam at the fine-grid edge
      const e = Math.min(bx - F.x0, F.x1 - bx, bz - F.z0, F.z1 - bz);
      if (e >= 150) return hf;
      return bilinear(O, bx, bz) + (hf - bilinear(O, bx, bz)) * smooth01(e / 150);
    }
    if (O) return bilinear(O, bx, bz);
    return bilinear(F, bx, bz);
  }

  /** Ground height at WORLD (x,z): the built ground surface when attached, else the DEM; 0 if not loaded. */
  heightAt(x, z) {
    if (!Number.isFinite(x) || !Number.isFinite(z)) return 0;
    if (this.surface) return this.surface.heightAt(x, z);
    return this.rawHeightAt(x, z);
  }

  /** Unit normal of heightAt at (x,z) written to out ({x,y,z} or THREE.Vector3). */
  normalAt(x, z, out = { x: 0, y: 1, z: 0 }) {
    const e = this.surface ? this.surface.cellAt(x, z) : 4;
    const dx = (this.heightAt(x + e, z) - this.heightAt(x - e, z)) / (2 * e);
    const dz = (this.heightAt(x, z + e) - this.heightAt(x, z - e)) / (2 * e);
    const l = Math.hypot(dx, 1, dz);
    out.x = -dx / l; out.y = 1 / l; out.z = -dz / l;
    return out;
  }
}

function bilinear(G, x, z) {
  const fx = (x - G.x0) / G.cell, fz = (z - G.z0) / G.cell;
  const i = clamp(Math.floor(fx), 0, G.nx - 2), k = clamp(Math.floor(fz), 0, G.nz - 2);
  const tx = clamp(fx - i, 0, 1), tz = clamp(fz - k, 0, 1);
  const h = G.h, r = k * G.nx + i;
  const a = h[r], b = h[r + 1], c = h[r + G.nx], d = h[r + G.nx + 1];
  return (a + (b - a) * tx) * (1 - tz) + (c + (d - c) * tx) * tz;
}

// ====================================================================== ground surface
/**
 * Nested lattice height field. levels: [{minX,minZ,maxX,maxZ,cell}] finest first; each cell must be
 * twice the previous one. Rects are snapped outward to the next coarser cell so every finer level's
 * boundary lies on coarser lattice lines (no T-junction cracks: boundary vertices are made linear).
 * Each cell is split along its (x0,z0)-(x1,z1) diagonal.
 */
export class GroundSurface {
  constructor(terrain, levels, { minHeight = 0 } = {}) {
    this.terrain = terrain;
    this.minHeight = minHeight;
    const L = levels.map((l) => ({ ...l }));
    for (let i = 0; i < L.length; i++) {
      const snap = i + 1 < L.length ? L[i + 1].cell : L[i].cell;
      const l = L[i];
      if (i > 0) { // contain the finer level
        const f = L[i - 1];
        l.minX = Math.min(l.minX, f.x0); l.minZ = Math.min(l.minZ, f.z0); l.maxX = Math.max(l.maxX, f.x1); l.maxZ = Math.max(l.maxZ, f.z1);
      }
      l.x0 = Math.floor(l.minX / snap) * snap; l.z0 = Math.floor(l.minZ / snap) * snap;
      l.x1 = Math.ceil(l.maxX / snap) * snap; l.z1 = Math.ceil(l.maxZ / snap) * snap;
      l.nx = Math.round((l.x1 - l.x0) / l.cell) + 1; l.nz = Math.round((l.z1 - l.z0) / l.cell) + 1;
      l.h = new Float32Array(l.nx * l.nz);
      l.index = i;
    }
    this.levels = L;
  }

  /** Terrain height used before corridors (clamped so land never dips to the water). */
  baseAt(x, z) { return Math.max(this.minHeight, this.terrain.rawHeightAt(x, z)); }

  init() {
    for (const l of this.levels) {
      for (let k = 0; k < l.nz; k++) for (let i = 0; i < l.nx; i++) l.h[k * l.nx + i] = this.baseAt(l.x0 + i * l.cell, l.z0 + k * l.cell);
    }
  }

  /**
   * Flatten ground-level road corridors. list: [{pts:[[x,z],...], hs:[h...], core, blend}]. Vertices
   * within `core` metres of a centre line take the line's height (weighted average where corridors
   * overlap); the weight fades to 0 over the next `blend` metres.
   */
  applyCorridors(list) {
    for (const l of this.levels) {
      const N = l.nx * l.nz;
      const sw = new Float32Array(N), swh = new Float32Array(N), mw = new Float32Array(N);
      const c = l.cell;
      for (const C of list) {
        const P = C.pts, H = C.hs, core = C.core, R = C.core + C.blend;
        for (let s = 0; s + 1 < P.length; s++) {
          const ax = P[s][0], az = P[s][1], bx = P[s + 1][0], bz = P[s + 1][1];
          const dx = bx - ax, dz = bz - az, L2 = dx * dx + dz * dz;
          if (L2 < 1e-6) continue;
          const i0 = Math.max(0, Math.ceil((Math.min(ax, bx) - R - l.x0) / c)), i1 = Math.min(l.nx - 1, Math.floor((Math.max(ax, bx) + R - l.x0) / c));
          const k0 = Math.max(0, Math.ceil((Math.min(az, bz) - R - l.z0) / c)), k1 = Math.min(l.nz - 1, Math.floor((Math.max(az, bz) + R - l.z0) / c));
          if (i0 > i1 || k0 > k1) continue;
          const ha = H[s], hb = H[s + 1];
          for (let k = k0; k <= k1; k++) {
            const z = l.z0 + k * c;
            for (let i = i0; i <= i1; i++) {
              const x = l.x0 + i * c;
              let t = ((x - ax) * dx + (z - az) * dz) / L2;
              t = t < 0 ? 0 : t > 1 ? 1 : t;
              const px = ax + dx * t - x, pz = az + dz * t - z;
              const d = Math.sqrt(px * px + pz * pz);
              if (d >= R) continue;
              const w = d <= core ? 1 : 1 - smooth01((d - core) / C.blend);
              const j = k * l.nx + i;
              sw[j] += w; swh[j] += w * (ha + (hb - ha) * t);
              if (w > mw[j]) mw[j] = w;
            }
          }
        }
      }
      for (let j = 0; j < N; j++) {
        if (!(sw[j] > 0)) continue;
        const road = swh[j] / sw[j];
        l.h[j] = Math.max(this.minHeight, l.h[j] + (road - l.h[j]) * mw[j]);
      }
    }
  }

  /** Make each finer level's boundary vertices linear between the coarser lattice points (crack-free). */
  finalize() {
    // coarse -> fine, so a finer level copies boundary values that are already final
    for (let li = this.levels.length - 2; li >= 0; li--) {
      const l = this.levels[li], C = this.levels[li + 1];
      const at = (i, k) => k * l.nx + i;
      // even boundary vertices coincide with coarse vertices: copy them, odd ones = average
      const fix = (i, k) => {
        const x = l.x0 + i * l.cell, z = l.z0 + k * l.cell;
        const ci = Math.round((x - C.x0) / C.cell), ck = Math.round((z - C.z0) / C.cell);
        if (Math.abs(C.x0 + ci * C.cell - x) < 1e-6 && Math.abs(C.z0 + ck * C.cell - z) < 1e-6) l.h[at(i, k)] = C.h[ck * C.nx + ci];
      };
      for (let i = 0; i < l.nx; i++) { fix(i, 0); fix(i, l.nz - 1); }
      for (let k = 0; k < l.nz; k++) { fix(0, k); fix(l.nx - 1, k); }
      for (let i = 1; i < l.nx - 1; i += 2) for (const k of [0, l.nz - 1]) l.h[at(i, k)] = (l.h[at(i - 1, k)] + l.h[at(i + 1, k)]) / 2;
      for (let k = 1; k < l.nz - 1; k += 2) for (const i of [0, l.nx - 1]) l.h[at(i, k)] = (l.h[at(i, k - 1)] + l.h[at(i, k + 1)]) / 2;
    }
  }

  levelAt(x, z) {
    const L = this.levels;
    for (let i = 0; i < L.length; i++) { const l = L[i]; if (x >= l.x0 && x <= l.x1 && z >= l.z0 && z <= l.z1) return l; }
    return L[L.length - 1];
  }
  cellAt(x, z) { return this.levelAt(x, z).cell; }

  /** Height of the ground mesh at (x,z) (triangle interpolation identical to the mesh). */
  heightAt(x, z) { return sampleLevel(this.levelAt(x, z), x, z); }

  /** Smooth normal (central differences over one lattice cell). */
  normalAt(x, z, out = [0, 1, 0]) {
    const e = this.cellAt(x, z);
    const dx = (this.heightAt(x + e, z) - this.heightAt(x - e, z)) / (2 * e);
    const dz = (this.heightAt(x, z + e) - this.heightAt(x, z - e)) / (2 * e);
    const l = Math.hypot(dx, 1, dz);
    out[0] = -dx / l; out[1] = 1 / l; out[2] = -dz / l;
    return out;
  }
}

export function sampleLevel(l, x, z) {
  const fx = (x - l.x0) / l.cell, fz = (z - l.z0) / l.cell;
  const i = clamp(Math.floor(fx), 0, l.nx - 2), k = clamp(Math.floor(fz), 0, l.nz - 2);
  const tx = clamp(fx - i, 0, 1), tz = clamp(fz - k, 0, 1);
  const h = l.h, r = k * l.nx + i;
  const h00 = h[r], h10 = h[r + 1], h01 = h[r + l.nx], h11 = h[r + l.nx + 1];
  return tx >= tz ? h00 + (h10 - h00) * tx + (h11 - h10) * tz : h00 + (h11 - h01) * tx + (h01 - h00) * tz;
}

// ====================================================================== draping
// Drape flat map polygons (land, piers, parks, plazas, parking lots, ponds) onto the GroundSurface
// lattice. Cells fully inside a polygon become the lattice's own two triangles (shared vertices);
// cells crossed by the polygon outline are clipped against each of their two triangles
// (Sutherland-Hodgman, polygon vs convex triangle) and re-triangulated, so every emitted triangle
// lies in exactly one planar ground triangle: draped layers are coplanar with the ground mesh.

const TILE = 16; // cells per pre-clip tile (keeps the per-cell clipping input small)

/**
 * @param {GroundSurface} surface
 * @param {Array<[number,number]>} outer  ring [x,z]
 * @param {Array<Array<[number,number]>>} holes
 * @param {number} yOff  added to the ground height
 * @param {{vert:(x:number,y:number,z:number,n:number[])=>number, tri:(a:number,b:number,c:number)=>void}} sink
 *        tri receives indices already wound counter-clockwise seen from above (+Y normal).
 */
export function drapePolygon(surface, outer, holes, yOff, sink) {
  if (!outer || outer.length < 3) return 0;
  holes = (holes || []).filter((h) => h && h.length >= 3);
  const rings = [outer, ...holes];
  let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
  for (const [x, z] of outer) { if (x < minX) minX = x; if (x > maxX) maxX = x; if (z < minZ) minZ = z; if (z > maxZ) maxZ = z; }
  const P = []; // vertex positions emitted (for winding)
  const nrm = [0, 1, 0];
  let tris = 0;
  const emit = (x, y, z) => {
    surface.normalAt(x, z, nrm);
    const idx = sink.vert(x, y, z, nrm);
    P[idx] = [x, z];
    return idx;
  };
  const triUp = (a, b, c) => {
    const pa = P[a], pb = P[b], pc = P[c];
    const cy = (pb[1] - pa[1]) * (pc[0] - pa[0]) - (pb[0] - pa[0]) * (pc[1] - pa[1]);
    if (Math.abs(cy) < 1e-9) return;
    if (cy > 0) sink.tri(a, b, c); else sink.tri(a, c, b);
    tris++;
  };

  const L = surface.levels;
  for (let li = 0; li < L.length; li++) {
    const l = L[li], f = li > 0 ? L[li - 1] : null, c = l.cell;
    const i0 = Math.max(0, Math.floor((minX - l.x0) / c)), i1 = Math.min(l.nx - 2, Math.floor((maxX - l.x0) / c));
    const k0 = Math.max(0, Math.floor((minZ - l.z0) / c)), k1 = Math.min(l.nz - 2, Math.floor((maxZ - l.z0) / c));
    if (i0 > i1 || k0 > k1) continue;
    const W = i1 - i0 + 1;
    const inFiner = (i, k) => {
      if (!f) return false;
      const x = l.x0 + (i + 0.5) * c, z = l.z0 + (k + 0.5) * c;
      return x > f.x0 && x < f.x1 && z > f.z0 && z < f.z1;
    };
    // cells touched by the outline
    const edge = new Set();
    for (const r of rings) {
      for (let a = 0; a < r.length; a++) {
        const p = r[a], q = r[(a + 1) % r.length];
        const len = Math.hypot(q[0] - p[0], q[1] - p[1]);
        const n = Math.max(1, Math.ceil(len / (c * 0.2)));
        for (let s = 0; s <= n; s++) {
          const x = p[0] + ((q[0] - p[0]) * s) / n, z = p[1] + ((q[1] - p[1]) * s) / n;
          const ci = Math.floor((x - l.x0) / c), ck = Math.floor((z - l.z0) / c);
          for (let di = -1; di <= 0; di++) for (let dk = -1; dk <= 0; dk++) {
            // a point on a cell border touches both neighbours
            const ii = ci + di, kk = ck + dk;
            if (di && Math.abs(x - (l.x0 + ci * c)) > 1e-6) continue;
            if (dk && Math.abs(z - (l.z0 + ck * c)) > 1e-6) continue;
            if (ii >= i0 && ii <= i1 && kk >= k0 && kk <= k1) edge.add((kk - k0) * W + (ii - i0));
          }
        }
      }
    }
    // lattice vertices of full cells (shared)
    const vmap = new Map();
    const latV = (i, k) => {
      const key = k * l.nx + i;
      let v = vmap.get(key);
      if (v === undefined) { v = emit(l.x0 + i * c, l.h[key] + yOff, l.z0 + k * c); vmap.set(key, v); }
      return v;
    };
    const tileCache = new Map();
    const xs = [];
    for (let k = k0; k <= k1; k++) {
      // scanline at the row centre for the inside test of untouched cells
      const zc = l.z0 + (k + 0.5) * c;
      xs.length = 0;
      for (const r of rings) for (let a = 0, b = r.length - 1; a < r.length; b = a++) {
        const za = r[a][1], zb = r[b][1];
        if ((za > zc) !== (zb > zc)) xs.push(r[a][0] + ((zc - za) / (zb - za)) * (r[b][0] - r[a][0]));
      }
      xs.sort((p, q) => p - q);
      let ptr = 0;
      for (let i = i0; i <= i1; i++) {
        const xc = l.x0 + (i + 0.5) * c;
        while (ptr < xs.length && xs[ptr] < xc) ptr++;
        if (inFiner(i, k)) continue;
        if (edge.has((k - k0) * W + (i - i0))) {
          clipCell(l, i, k);
        } else if (ptr & 1) {
          const v00 = latV(i, k), v10 = latV(i + 1, k), v11 = latV(i + 1, k + 1), v01 = latV(i, k + 1);
          triUp(v00, v10, v11); triUp(v00, v11, v01);
        }
      }
    }

    function tileRings(ti, tk) {
      const key = tk * 100000 + ti;
      let t = tileCache.get(key);
      if (t) return t;
      const x0 = l.x0 + ti * TILE * c, z0 = l.z0 + tk * TILE * c, x1 = x0 + TILE * c, z1 = z0 + TILE * c;
      const sq = [[x0, z0], [x1, z0], [x1, z1], [x0, z1]];
      t = rings.map((r) => clipConvex(r, sq));
      tileCache.set(key, t);
      return t;
    }
    function clipCell(l, i, k) {
      const x0 = l.x0 + i * c, z0 = l.z0 + k * c, x1 = x0 + c, z1 = z0 + c;
      const tr = tileRings(Math.floor(i / TILE), Math.floor(k / TILE));
      if (!tr[0] || tr[0].length < 3) return;
      for (const T of [[[x0, z0], [x1, z0], [x1, z1]], [[x0, z0], [x1, z1], [x0, z1]]]) {
        const o = clipConvex(tr[0], T);
        const ao = Math.abs(area(o));
        if (ao < 1e-4) continue;
        const hs = [];
        let skip = false;
        for (let h = 1; h < tr.length; h++) {
          const hc = clipConvex(tr[h], T);
          const ah = Math.abs(area(hc));
          if (ah < 1e-4) continue;
          if (ah >= ao - 1e-3) { skip = true; break; }
          hs.push(hc);
        }
        if (skip) continue;
        const cont = o.map((p) => new THREE.Vector2(p[0], p[1]));
        const hv = hs.map((h) => h.map((p) => new THREE.Vector2(p[0], p[1])));
        let faces;
        try { faces = THREE.ShapeUtils.triangulateShape(cont, hv); } catch { continue; }
        if (!faces || !faces.length) continue;
        const all = cont.concat(...hv);
        const ids = all.map((p) => emit(p.x, sampleLevel(l, p.x, p.y) + yOff, p.y));
        for (const fc of faces) if (fc[0] < ids.length && fc[1] < ids.length && fc[2] < ids.length) triUp(ids[fc[0]], ids[fc[1]], ids[fc[2]]);
      }
    }
  }
  return tris;
}

/** Sutherland-Hodgman: clip ring (any shape) by a convex polygon. Returns ring (may be empty). */
export function clipConvex(ring, clip) {
  if (!ring || ring.length < 3) return [];
  const s = area(clip) > 0 ? 1 : -1;
  let out = ring;
  for (let e = 0; e < clip.length && out.length; e++) {
    const p = clip[e], q = clip[(e + 1) % clip.length];
    const ex = q[0] - p[0], ez = q[1] - p[1];
    const side = (v) => s * (ex * (v[1] - p[1]) - ez * (v[0] - p[0]));
    const inp = out;
    out = [];
    for (let a = 0; a < inp.length; a++) {
      const A = inp[a], B = inp[(a + 1) % inp.length];
      const sa = side(A), sb = side(B);
      if (sa >= 0) out.push(A);
      if ((sa >= 0) !== (sb >= 0)) {
        const t = sa / (sa - sb);
        out.push([A[0] + (B[0] - A[0]) * t, A[1] + (B[1] - A[1]) * t]);
      }
    }
  }
  // drop consecutive duplicates
  const res = [];
  for (const p of out) {
    const l = res[res.length - 1];
    if (!l || Math.abs(l[0] - p[0]) > 1e-7 || Math.abs(l[1] - p[1]) > 1e-7) res.push(p);
  }
  while (res.length > 1 && Math.abs(res[0][0] - res[res.length - 1][0]) < 1e-7 && Math.abs(res[0][1] - res[res.length - 1][1]) < 1e-7) res.pop();
  return res.length >= 3 ? res : [];
}

function area(r) {
  let a = 0;
  for (let i = 0, n = r.length; i < n; i++) { const p = r[i], q = r[(i + 1) % n]; a += p[0] * q[1] - q[0] * p[1]; }
  return a / 2;
}

/**
 * Points along segment a->b split wherever it crosses a lattice line of the finest level
 * (x = n*c, z = n*c, x - z = n*c), so a wall top sampled at these points follows the ground exactly.
 */
export function latticeSplit(surface, a, b) {
  const c = surface.levels[0].cell;
  const ts = [0, 1];
  const dx = b[0] - a[0], dz = b[1] - a[1];
  const fam = (v0, dv) => {
    if (Math.abs(dv) < 1e-9) return;
    const lo = Math.min(v0, v0 + dv), hi = Math.max(v0, v0 + dv);
    for (let n = Math.ceil(lo / c); n * c <= hi; n++) { const t = (n * c - v0) / dv; if (t > 1e-6 && t < 1 - 1e-6) ts.push(t); }
  };
  fam(a[0], dx); fam(a[1], dz); fam(a[0] - a[1], dx - dz);
  ts.sort((p, q) => p - q);
  const out = [];
  for (const t of ts) { if (out.length && t - out[out.length - 1][2] < 1e-5) continue; out.push([a[0] + dx * t, a[1] + dz * t, t]); }
  return out;
}
