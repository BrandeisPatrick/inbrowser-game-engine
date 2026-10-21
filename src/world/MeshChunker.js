// Spatial chunking of merged static meshes. A city merged into a few huge meshes is cheap on draw calls but can
// never be culled: every pass (main view AND each shadow cascade) shades all of it. chunkMesh() splits a merged
// mesh into tiles on an XZ grid (a triangle goes to the tile of its centroid; tile bounds are computed from the
// real vertices, so culling stays exact), giving three.js per-tile frustum culling in every pass for the price
// of one draw call per visible tile. Optional per-location tile size (small tiles where the camera usually is,
// large ones toward the horizon). Pure geometry: no material, shader or behaviour changes.
//
//   import { chunkMesh } from '../engine/src/world/MeshChunker.js';
//   const group = chunkMesh(mesh, { cell: 320 });                 // Group of tile meshes (same name/material/flags)
//   const group = chunkMesh(mesh, { cellAt: (x, z) => Math.hypot(x, z) < 1500 ? 320 : 1280 });
//   parent.remove(mesh); parent.add(group);                       // or replaceWithChunks(mesh, opts)
import * as THREE from 'three';

/**
 * Split a BufferGeometry into tiles. Non-indexed geometry is treated as a triangle soup; geometry groups and
 * interleaved attributes are not supported (returns null). Degenerate triangles (repeated indices or three
 * identical positions, e.g. faces collapsed by a later edit) are dropped.
 * @param {THREE.BufferGeometry} geo
 * @param {object} [o]
 * @param {number} [o.cell=256]  tile size (m)
 * @param {(x:number, z:number) => number} [o.cellAt]  tile size at a location (overrides `cell`; sizes should be
 *        powers of two times a base so tiles nest)
 * @param {number} [o.minTris=0]  tiles with fewer triangles are merged into one "rest" tile
 * @param {Uint8Array} [o.triClass]  per source triangle class 0..255 (default geo.userData.triClass): each tile
 *        stores its triangles ordered by class and geometry.userData.classStarts[c] = first index of class c, so a
 *        pass can draw a class prefix with drawRange (e.g. shadowDrawRange)
 * @returns {{geometry: THREE.BufferGeometry, tris: number, key: string}[] | null}
 */
export function chunkGeometry(geo, o = {}) {
  if (!geo || !geo.attributes.position || (geo.groups && geo.groups.length > 1)) return null;
  for (const k in geo.attributes) if (geo.attributes[k].isInterleavedBufferAttribute) return null;
  const P = geo.attributes.position.array, I = geo.index ? geo.index.array : null;
  const nTri = Math.floor((I ? I.length : P.length / 3) / 3);
  const cell = o.cell || 256, cellAt = o.cellAt || null;
  const cls = o.triClass || (geo.userData && geo.userData.triClass) || null;
  const keyOf = new Float64Array(nTri);
  const counts = new Map();
  let kept = 0, lastKey = NaN, lastN = 0; // merged runs: consecutive triangles are mostly in the same tile
  for (let t = 0; t < nTri; t++) {
    const a = I ? I[t * 3] : t * 3, b = I ? I[t * 3 + 1] : t * 3 + 1, c = I ? I[t * 3 + 2] : t * 3 + 2;
    const ax = P[a * 3], ay = P[a * 3 + 1], az = P[a * 3 + 2], bx = P[b * 3], by = P[b * 3 + 1], bz = P[b * 3 + 2], cx = P[c * 3], cy = P[c * 3 + 1], cz = P[c * 3 + 2];
    if (a === b || b === c || a === c || (ax === bx && ay === by && az === bz && ax === cx && ay === cy && az === cz)) { keyOf[t] = NaN; continue; }
    const x = (ax + bx + cx) / 3, z = (az + bz + cz) / 3;
    const s = cellAt ? cellAt(x, z) : cell;
    const i = Math.floor(x / s), k = Math.floor(z / s);
    const key = (s * 4096 + (i + 2048)) * 4096 + (k + 2048); // exact in a double for s < 2^20, |i|,|k| < 2048
    keyOf[t] = key;
    if (key === lastKey) lastN++;
    else { if (lastN) counts.set(lastKey, (counts.get(lastKey) || 0) + lastN); lastKey = key; lastN = 1; }
    kept++;
  }
  if (lastN) counts.set(lastKey, (counts.get(lastKey) || 0) + lastN);
  if (!kept) return [];
  // small tiles -> one shared rest tile
  const REST = -1;
  if (o.minTris > 0) {
    let rest = 0;
    for (const [k, n] of [...counts]) if (n < o.minTris) { counts.delete(k); rest += n; }
    if (rest) counts.set(REST, rest);
  }
  const keys = [...counts.keys()];
  const slot = new Map(keys.map((k, j) => [k, j]));
  // triangle lists per tile (stable order)
  const start = new Int32Array(keys.length + 1);
  keys.forEach((k, j) => { start[j + 1] = start[j] + counts.get(k); });
  const fill = start.slice(0, keys.length), order = new Int32Array(kept);
  const nCls = cls ? 1 + cls.reduce((m, c) => (c > m ? c : m), 0) : 1;
  for (let c = 0; c < nCls; c++) { // class-major within each tile (one pass without classes)
    let pk = NaN, pj = -1;
    for (let t = 0; t < nTri; t++) {
      const k = keyOf[t]; if (k !== k || (cls && cls[t] !== c)) continue; // NaN = dropped
      if (k !== pk) { pk = k; pj = slot.get(k); if (pj === undefined) pj = slot.get(REST); }
      order[fill[pj]++] = t;
    }
  }
  const nV = P.length / 3;
  const remap = new Int32Array(nV), stamp = new Int32Array(nV).fill(-1);
  const names = Object.keys(geo.attributes);
  const out = [];
  let maxT = 0; for (let j = 0; j < keys.length; j++) maxT = Math.max(maxT, start[j + 1] - start[j]);
  const src = new Int32Array(maxT * 3); // tile vertex -> source vertex
  for (let j = 0; j < keys.length; j++) {
    const t0 = start[j], t1 = start[j + 1];
    // vertex remap for this tile
    let nv = 0;
    const idx = new Uint32Array((t1 - t0) * 3);
    for (let q = t0, w = 0; q < t1; q++) {
      const t = order[q];
      for (let e = 0; e < 3; e++) {
        const v = I ? I[t * 3 + e] : t * 3 + e;
        if (stamp[v] !== j) { stamp[v] = j; remap[v] = nv; src[nv++] = v; }
        idx[w++] = remap[v];
      }
    }
    const g = new THREE.BufferGeometry();
    for (const name of names) {
      const A = geo.attributes[name], sz = A.itemSize, arr = A.array;
      const dst = new arr.constructor(nv * sz);
      if (sz === 3) for (let v = 0; v < nv; v++) { const s0 = src[v] * 3, d0 = v * 3; dst[d0] = arr[s0]; dst[d0 + 1] = arr[s0 + 1]; dst[d0 + 2] = arr[s0 + 2]; }
      else if (sz === 2) for (let v = 0; v < nv; v++) { const s0 = src[v] * 2, d0 = v * 2; dst[d0] = arr[s0]; dst[d0 + 1] = arr[s0 + 1]; }
      else if (sz === 1) for (let v = 0; v < nv; v++) dst[v] = arr[src[v]];
      else for (let v = 0; v < nv; v++) { const s0 = src[v] * sz, d0 = v * sz; for (let c = 0; c < sz; c++) dst[d0 + c] = arr[s0 + c]; }
      const B = new THREE.BufferAttribute(dst, sz, A.normalized);
      B.name = A.name; B.usage = A.usage;
      g.setAttribute(name, B);
    }
    g.setIndex(new THREE.BufferAttribute(nv > 65535 ? idx : Uint16Array.from(idx), 1));
    g.computeBoundingBox(); g.computeBoundingSphere();
    const { triClass, ...ud } = geo.userData || {}; void triClass;
    g.userData = ud;
    if (cls) { // first index of each class in this tile
      const cs = new Array(nCls + 1).fill(0);
      for (let q = t0; q < t1; q++) cs[cls[order[q]] + 1] += 3;
      for (let c = 1; c <= nCls; c++) cs[c] += cs[c - 1];
      g.userData.classStarts = cs;
    }
    out.push({ geometry: g, tris: t1 - t0, key: keys[j] === REST ? 'rest' : String(keys[j]) });
  }
  return out;
}

/**
 * Tile a static mesh. Returns a Group (named like the mesh) of tile meshes sharing the mesh's material(s),
 * custom depth / distance materials, shadow flags, render order, layers, matrix and userData; or the mesh itself
 * when it cannot be split (or would yield a single tile).
 * @param {THREE.Mesh} mesh
 * @param {object} [o] chunkGeometry options + dispose (default true: dispose the source geometry)
 */
export function chunkMesh(mesh, o = {}) {
  if (!mesh || !mesh.isMesh || mesh.isInstancedMesh || mesh.isBatchedMesh || mesh.isSkinnedMesh) return mesh;
  const tiles = chunkGeometry(mesh.geometry, o);
  if (!tiles || tiles.length < 2) return mesh;
  const g = new THREE.Group();
  g.name = mesh.name;
  g.userData = { ...mesh.userData, chunked: tiles.length };
  mesh.updateMatrix();
  g.matrix.copy(mesh.matrix); g.matrix.decompose(g.position, g.quaternion, g.scale);
  g.matrixAutoUpdate = mesh.matrixAutoUpdate;
  for (const t of tiles) {
    const m = new THREE.Mesh(t.geometry, mesh.material);
    m.name = mesh.name;
    for (const k of ['customDepthMaterial', 'customDistanceMaterial', 'castShadow', 'receiveShadow', 'renderOrder', 'frustumCulled', 'visible']) m[k] = mesh[k];
    m.layers.mask = mesh.layers.mask;
    m.onBeforeRender = mesh.onBeforeRender; m.onAfterRender = mesh.onAfterRender;
    m.onBeforeShadow = mesh.onBeforeShadow; m.onAfterShadow = mesh.onAfterShadow;
    m.userData = mesh.userData;
    m.matrixAutoUpdate = false; m.updateMatrix();
    g.add(m);
  }
  if (o.dispose !== false) mesh.geometry.dispose();
  return g;
}

/** chunkMesh() + swap the result into the mesh's parent at the same position in the child list. */
export function replaceWithChunks(mesh, o = {}) {
  const parent = mesh && mesh.parent;
  const r = chunkMesh(mesh, o);
  if (r === mesh || !parent) return r;
  const i = parent.children.indexOf(mesh);
  parent.remove(mesh);
  parent.add(r);
  // keep the original draw / traversal order
  const c = parent.children; c.splice(c.indexOf(r), 1); c.splice(Math.max(0, i), 0, r);
  return r;
}

/** Sum of triangles / tiles of a chunked group (debug). */
export function chunkStats(obj) {
  let tiles = 0, tris = 0;
  obj.traverse((m) => { if (m.isMesh && m.geometry) { tiles++; tris += (m.geometry.index ? m.geometry.index.count : m.geometry.attributes.position.count) / 3; } });
  return { tiles, tris };
}
