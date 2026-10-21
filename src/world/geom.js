// Small deterministic helpers for map data: string hash, polygon math (area, point-in-polygon, bounds, centroid,
// ring sanitising) and polyline length. Seeded RNG / lattice noise / scalar easing live in ../render/noise.js and
// are re-exported here so world modules have one import. Bit-exact with the game's former src/world/util.js.
export { mulberry32, hash2, pnoise, pfbm, clamp, lerp, smoothstep } from '../render/noise.js';

/** FNV-1a string hash -> [0, 1). */
export function hashStr(s) {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
  return (h >>> 0) / 4294967296;
}

/** Signed area of a ring [[x, z], ...] (positive = counter-clockwise in x/z). */
export function signedArea(p) {
  let a = 0;
  for (let i = 0, n = p.length; i < n; i++) {
    const q = p[i], s = p[(i + 1) % n];
    a += q[0] * s[1] - s[0] * q[1];
  }
  return a / 2;
}

export function pointInRing(x, z, ring) {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const xi = ring[i][0], zi = ring[i][1], xj = ring[j][0], zj = ring[j][1];
    if ((zi > z) !== (zj > z) && x < ((xj - xi) * (z - zi)) / (zj - zi) + xi) inside = !inside;
  }
  return inside;
}

/**
 * Exact, faster pointInRing for big rings queried many times (coastlines): edges bucketed into z rows, so a query
 * scans only the edges whose z span reaches its row. Same crossing predicate and arithmetic as pointInRing (edges
 * outside the row can never toggle the parity), so `idx.contains(x, z) === pointInRing(x, z, ring)` for every point.
 */
export class RingIndex {
  constructor(ring, rowH = 0) {
    const n = ring.length;
    let z0 = Infinity, z1 = -Infinity;
    for (const p of ring) { if (p[1] < z0) z0 = p[1]; if (p[1] > z1) z1 = p[1]; }
    this.ring = ring; this.z0 = z0; this.z1 = z1;
    this.h = rowH > 0 ? rowH : Math.max(4, (z1 - z0) / Math.max(1, Math.round(n / 4)));
    const rows = this.rows = Math.max(1, Math.ceil((z1 - z0) / this.h) + 1);
    const cnt = new Uint32Array(rows + 1);
    const span = (i, j) => [Math.floor((Math.min(ring[i][1], ring[j][1]) - z0) / this.h), Math.floor((Math.max(ring[i][1], ring[j][1]) - z0) / this.h)];
    for (let i = 0, j = n - 1; i < n; j = i++) { const [a, b] = span(i, j); for (let r = a; r <= b; r++) cnt[r + 1]++; }
    for (let r = 0; r < rows; r++) cnt[r + 1] += cnt[r];
    const fill = cnt.slice(0, rows), E = this.e = new Float64Array(cnt[rows] * 4);
    for (let i = 0, j = n - 1; i < n; j = i++) {
      const [a, b] = span(i, j);
      for (let r = a; r <= b; r++) { const k = fill[r]++ * 4; E[k] = ring[i][0]; E[k + 1] = ring[i][1]; E[k + 2] = ring[j][0]; E[k + 3] = ring[j][1]; }
    }
    this.start = cnt;
  }
  contains(x, z) {
    if (!(z >= this.z0 && z <= this.z1)) return false; // no edge spans z: even crossings (as pointInRing)
    const r = Math.floor((z - this.z0) / this.h), E = this.e;
    let inside = false;
    for (let k = this.start[r] * 4, e = this.start[r + 1] * 4; k < e; k += 4) {
      const xi = E[k], zi = E[k + 1], xj = E[k + 2], zj = E[k + 3];
      if ((zi > z) !== (zj > z) && x < ((xj - xi) * (z - zi)) / (zj - zi) + xi) inside = !inside;
    }
    return inside;
  }
}

export function pointInPoly(x, z, outer, holes) {
  if (!pointInRing(x, z, outer)) return false;
  if (holes) for (const h of holes) if (pointInRing(x, z, h)) return false;
  return true;
}

export function ringBounds(ring) {
  let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
  for (const [x, z] of ring) {
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
  }
  return { minX, maxX, minZ, maxZ };
}

export function centroid(ring) {
  let a = 0, cx = 0, cz = 0;
  for (let i = 0, n = ring.length; i < n; i++) {
    const [x0, z0] = ring[i], [x1, z1] = ring[(i + 1) % n];
    const f = x0 * z1 - x1 * z0;
    a += f; cx += (x0 + x1) * f; cz += (z0 + z1) * f;
  }
  if (Math.abs(a) < 1e-9) {
    let sx = 0, sz = 0;
    for (const [x, z] of ring) { sx += x; sz += z; }
    return [sx / ring.length, sz / ring.length];
  }
  return [cx / (3 * a), cz / (3 * a)];
}

/** Remove consecutive duplicates / closing duplicate / non-finite points. Returns null if degenerate. */
export function sanitizeRing(ring, minLen = 0.05) {
  if (!Array.isArray(ring)) return null;
  const out = [];
  for (const p of ring) {
    if (!p || !isFinite(p[0]) || !isFinite(p[1])) continue;
    const l = out[out.length - 1];
    if (l && Math.abs(l[0] - p[0]) < minLen && Math.abs(l[1] - p[1]) < minLen) continue;
    out.push([p[0], p[1]]);
  }
  while (out.length > 1) {
    const a = out[0], b = out[out.length - 1];
    if (Math.abs(a[0] - b[0]) < minLen && Math.abs(a[1] - b[1]) < minLen) out.pop(); else break;
  }
  if (out.length < 3) return null;
  if (Math.abs(signedArea(out)) < 0.5) return null;
  return out;
}

export function polylineLength(p) {
  let L = 0;
  for (let i = 1; i < p.length; i++) L += Math.hypot(p[i][0] - p[i - 1][0], p[i][1] - p[i - 1][1]);
  return L;
}
