// Generic spatial acceleration structures (no THREE, no DOM; runs in Node):
//   ColliderGrid  uniform-grid hash of axis-aligned boxes {minX,minY,maxX,maxY,minZ,maxZ,kind} + DDA ray traversal
//   polygonToBoxes  footprint polygon (with holes) -> merged axis-aligned strips
//   SegmentHash   hash of 2D segments with payload (road/bridge surfaces, nearest-road queries)
//   Raster        byte mask over a rectangle with polygon scanline fill
//   MinHeap       binary min-heap keyed by number

const KEY_OFF = 2048;
const cellKey = (i, k) => (i + KEY_OFF) * 4096 + (k + KEY_OFF);

/** Uniform-grid spatial hash of axis-aligned boxes {minX,minY,maxX,maxY,minZ,maxZ}. */
export class ColliderGrid {
  constructor(cellSize = 32) {
    this.cell = cellSize;
    this.boxes = [];
    this.cells = new Map();
    this._stamp = new Uint32Array(1024);
    this._q = 0;
  }
  add(minX, minY, maxX, maxY, minZ, maxZ, kind = 0) {
    if (!(maxX > minX) || !(maxZ > minZ) || !(maxY > minY)) return -1;
    if (![minX, minY, maxX, maxY, minZ, maxZ].every(Number.isFinite)) return -1;
    const idx = this.boxes.length;
    this.boxes.push({ minX, minY, maxX, maxY, minZ, maxZ, kind });
    const c = this.cell;
    const i0 = Math.floor(minX / c), i1 = Math.floor(maxX / c), k0 = Math.floor(minZ / c), k1 = Math.floor(maxZ / c);
    for (let i = i0; i <= i1; i++) for (let k = k0; k <= k1; k++) {
      const key = cellKey(i, k);
      let a = this.cells.get(key);
      if (!a) { a = []; this.cells.set(key, a); }
      a.push(idx);
    }
    if (this._stamp.length < this.boxes.length) {
      const s = new Uint32Array(this._stamp.length * 2); s.set(this._stamp); this._stamp = s;
    }
    return idx;
  }
  /** Boxes whose XZ footprint is within `radius` of (x,z). Returned objects are shared: do not mutate. */
  query(x, z, radius, out = []) {
    out.length = 0;
    if (!Number.isFinite(x) || !Number.isFinite(z)) return out;
    const r = Math.max(0, radius || 0);
    const c = this.cell;
    const i0 = Math.floor((x - r) / c), i1 = Math.floor((x + r) / c);
    const k0 = Math.floor((z - r) / c), k1 = Math.floor((z + r) / c);
    if ((i1 - i0 + 1) * (k1 - k0 + 1) > 4096) return out; // absurd radius guard
    const q = ++this._q;
    if (q === 0xffffffff) { this._stamp.fill(0); this._q = 1; }
    const st = this._stamp, boxes = this.boxes;
    for (let i = i0; i <= i1; i++) for (let k = k0; k <= k1; k++) {
      const a = this.cells.get(cellKey(i, k));
      if (!a) continue;
      for (let j = 0; j < a.length; j++) {
        const bi = a[j];
        if (st[bi] === this._q) continue;
        st[bi] = this._q;
        const b = boxes[bi];
        // circle vs rectangle in XZ
        const dx = x < b.minX ? b.minX - x : x > b.maxX ? x - b.maxX : 0;
        const dz = z < b.minZ ? b.minZ - z : z > b.maxZ ? z - b.maxZ : 0;
        if (dx * dx + dz * dz <= r * r) out.push(b);
      }
    }
    return out;
  }
  /**
   * Walk the grid cells crossed by the XZ projection of a ray, near to far (Amanatides-Woo DDA), calling
   * `visit(box, best)` once per box. `dx,dz` are the XZ components of the ray's 3D unit direction, so all
   * distances are along the 3D ray. `visit` returns a hit distance (or Infinity); traversal stops once the
   * next cell starts beyond the best hit. Returns the best hit distance (maxDist when nothing was hit).
   */
  raycast(ox, oz, dx, dz, maxDist, visit) {
    if (!Number.isFinite(ox) || !Number.isFinite(oz) || !(maxDist > 0)) return maxDist;
    const c = this.cell;
    let i = Math.floor(ox / c), k = Math.floor(oz / c);
    const si = dx > 1e-12 ? 1 : dx < -1e-12 ? -1 : 0, sk = dz > 1e-12 ? 1 : dz < -1e-12 ? -1 : 0;
    const tdi = si ? c / Math.abs(dx) : Infinity, tdk = sk ? c / Math.abs(dz) : Infinity;
    let tmi = si > 0 ? ((i + 1) * c - ox) / dx : si < 0 ? (i * c - ox) / dx : Infinity;
    let tmk = sk > 0 ? ((k + 1) * c - oz) / dz : sk < 0 ? (k * c - oz) / dz : Infinity;
    const q = ++this._q;
    if (q === 0xffffffff) { this._stamp.fill(0); this._q = 1; }
    const st = this._stamp, boxes = this.boxes, stamp = this._q;
    let best = maxDist, tCell = 0;
    for (let guard = 0; guard < 8192 && tCell <= best; guard++) {
      const a = this.cells.get(cellKey(i, k));
      if (a) {
        for (let j = 0; j < a.length; j++) {
          const bi = a[j];
          if (st[bi] === stamp) continue;
          st[bi] = stamp;
          const t = visit(boxes[bi], best);
          if (t < best) best = t;
        }
      }
      if (tmi < tmk) { tCell = tmi; tmi += tdi; i += si; } else { tCell = tmk; tmk += tdk; k += sk; }
      if (tCell === Infinity) break;
    }
    return best;
  }
}

/** Decompose a polygon footprint (with holes) into axis-aligned boxes (scanline strips, merged). */
export function polygonToBoxes(outer, holes, strip = 1.5) {
  let minZ = Infinity, maxZ = -Infinity, minX = Infinity, maxX = -Infinity;
  for (const [x, z] of outer) { if (z < minZ) minZ = z; if (z > maxZ) maxZ = z; if (x < minX) minX = x; if (x > maxX) maxX = x; }
  if (!(maxZ > minZ) || !(maxX > minX)) return [];
  const rings = holes && holes.length ? [outer, ...holes] : [outer];
  // axis-aligned rectangle fast path
  if (outer.length === 4 && (!holes || !holes.length)) {
    let aligned = true;
    for (let i = 0; i < 4; i++) {
      const a = outer[i], b = outer[(i + 1) % 4];
      if (Math.abs(a[0] - b[0]) > 0.3 && Math.abs(a[1] - b[1]) > 0.3) { aligned = false; break; }
    }
    if (aligned) return [[minX, minZ, maxX, maxZ]];
  }
  const n = Math.max(1, Math.ceil((maxZ - minZ) / strip));
  const h = (maxZ - minZ) / n;
  const rows = [];
  const xs = [];
  for (let s = 0; s < n; s++) {
    const z0 = minZ + s * h, z1 = z0 + h;
    const intervals = scan(rings, z0 + h * 0.5, xs); // mid scanline: +-strip/2*slope error on slanted walls
    rows.push({ z0, z1, iv: intervals });
  }
  // merge vertically
  const out = [];
  let open = [];
  for (const row of rows) {
    const next = [];
    for (const iv of row.iv) {
      const m = open.find((o) => Math.abs(o[0] - iv[0]) < 0.35 && Math.abs(o[2] - iv[1]) < 0.35 && !o.used);
      if (m) { m.used = true; m[3] = row.z1; next.push(m); }
      else next.push(Object.assign([iv[0], row.z0, iv[1], row.z1], { used: true }));
    }
    for (const o of open) if (!o.used) out.push([o[0], o[1], o[2], o[3]]);
    for (const o of next) o.used = false;
    open = next;
  }
  for (const o of open) out.push([o[0], o[1], o[2], o[3]]);
  return out.filter((b) => b[2] - b[0] > 0.05 && b[3] - b[1] > 0.05);
}
function scan(rings, z, xs) {
  xs.length = 0;
  for (const r of rings) {
    for (let i = 0, j = r.length - 1; i < r.length; j = i++) {
      const zi = r[i][1], zj = r[j][1];
      if ((zi > z) !== (zj > z)) xs.push(r[i][0] + ((z - zi) / (zj - zi)) * (r[j][0] - r[i][0]));
    }
  }
  xs.sort((a, b) => a - b);
  const iv = [];
  for (let i = 0; i + 1 < xs.length; i += 2) if (xs[i + 1] - xs[i] > 0.05) iv.push([xs[i], xs[i + 1]]);
  return iv;
}

/** Hash of 2D line segments with payload; used for road/bridge surfaces and nearest-road queries. */
export class SegmentHash {
  constructor(cellSize = 16) {
    this.cell = cellSize;
    this.segs = []; // {ax,az,bx,bz,ha,hb,hw,road}
    this.cells = new Map();
    this._stamp = new Uint32Array(1024);
    this._q = 0;
  }
  add(seg) {
    const idx = this.segs.length;
    this.segs.push(seg);
    const c = this.cell, m = seg.hw + 1;
    const i0 = Math.floor((Math.min(seg.ax, seg.bx) - m) / c), i1 = Math.floor((Math.max(seg.ax, seg.bx) + m) / c);
    const k0 = Math.floor((Math.min(seg.az, seg.bz) - m) / c), k1 = Math.floor((Math.max(seg.az, seg.bz) + m) / c);
    for (let i = i0; i <= i1; i++) for (let k = k0; k <= k1; k++) {
      const key = cellKey(i, k);
      let a = this.cells.get(key);
      if (!a) { a = []; this.cells.set(key, a); }
      a.push(idx);
    }
    if (this._stamp.length < this.segs.length) { const s = new Uint32Array(this._stamp.length * 2); s.set(this._stamp); this._stamp = s; }
    return idx;
  }
  /** Calls fn(seg, t, lateralSigned, dist) for every segment whose ribbon (hw + pad) contains (x,z). */
  forEachNear(x, z, pad, fn) {
    if (!Number.isFinite(x) || !Number.isFinite(z)) return;
    const c = this.cell;
    const i0 = Math.floor((x - pad) / c), i1 = Math.floor((x + pad) / c);
    const k0 = Math.floor((z - pad) / c), k1 = Math.floor((z + pad) / c);
    const q = ++this._q;
    if (q === 0xffffffff) { this._stamp.fill(0); this._q = 1; }
    for (let i = i0; i <= i1; i++) for (let k = k0; k <= k1; k++) {
      const a = this.cells.get(cellKey(i, k));
      if (!a) continue;
      for (let j = 0; j < a.length; j++) {
        const si = a[j];
        if (this._stamp[si] === this._q) continue;
        this._stamp[si] = this._q;
        const s = this.segs[si];
        const dx = s.bx - s.ax, dz = s.bz - s.az, L2 = dx * dx + dz * dz;
        if (L2 < 1e-6) continue;
        const L = Math.sqrt(L2);
        let t = ((x - s.ax) * dx + (z - s.az) * dz) / L2;
        const ext = (s.ext || 0) / L;
        if (t < -ext || t > 1 + ext) continue;
        const px = s.ax + dx * t, pz = s.az + dz * t;
        const lat = ((x - s.ax) * -dz + (z - s.az) * dx) / L; // +ve to the right of a->b (x right, z down)
        if (Math.abs(lat) > s.hw + pad) continue;
        fn(s, t < 0 ? 0 : t > 1 ? 1 : t, lat, Math.hypot(x - px, z - pz));
      }
    }
  }
}

/** Byte raster over a rectangle with polygon scanline fill (cell-centre sampling). */
export class Raster {
  constructor(minX, minZ, maxX, maxZ, res) {
    this.minX = minX; this.minZ = minZ; this.res = res;
    this.w = Math.max(1, Math.ceil((maxX - minX) / res));
    this.h = Math.max(1, Math.ceil((maxZ - minZ) / res));
    this.data = new Uint8Array(this.w * this.h);
  }
  idx(x, z) {
    const i = Math.floor((x - this.minX) / this.res), k = Math.floor((z - this.minZ) / this.res);
    if (i < 0 || k < 0 || i >= this.w || k >= this.h) return -1;
    return k * this.w + i;
  }
  get(x, z) { const i = this.idx(x, z); return i < 0 ? 0 : this.data[i]; }
  /** OR `bits` into every cell whose centre is inside the polygon (rings: outer + holes, even-odd). */
  fillPolygon(rings, bits, mode = 'or') {
    let minZ = Infinity, maxZ = -Infinity;
    for (const r of rings) for (const p of r) { if (p[1] < minZ) minZ = p[1]; if (p[1] > maxZ) maxZ = p[1]; }
    const k0 = Math.max(0, Math.floor((minZ - this.minZ) / this.res)), k1 = Math.min(this.h - 1, Math.ceil((maxZ - this.minZ) / this.res));
    const xs = [];
    for (let k = k0; k <= k1; k++) {
      const z = this.minZ + (k + 0.5) * this.res;
      xs.length = 0;
      for (const r of rings) for (let i = 0, j = r.length - 1; i < r.length; j = i++) {
        const zi = r[i][1], zj = r[j][1];
        if ((zi > z) !== (zj > z)) xs.push(r[i][0] + ((z - zi) / (zj - zi)) * (r[j][0] - r[i][0]));
      }
      if (xs.length < 2) continue;
      xs.sort((a, b) => a - b);
      const row = k * this.w;
      for (let q = 0; q + 1 < xs.length; q += 2) {
        const i0 = Math.max(0, Math.ceil((xs[q] - this.minX) / this.res - 0.5));
        const i1 = Math.min(this.w - 1, Math.floor((xs[q + 1] - this.minX) / this.res - 0.5));
        if (mode === 'or') for (let i = i0; i <= i1; i++) this.data[row + i] |= bits;
        else if (mode === 'clear') for (let i = i0; i <= i1; i++) this.data[row + i] &= ~bits;
        else for (let i = i0; i <= i1; i++) this.data[row + i] = bits;
      }
    }
  }
  /** True if any cell within radius r of (x,z) has any of `bits`. */
  any(x, z, r, bits) {
    const i0 = Math.floor((x - r - this.minX) / this.res), i1 = Math.floor((x + r - this.minX) / this.res);
    const k0 = Math.floor((z - r - this.minZ) / this.res), k1 = Math.floor((z + r - this.minZ) / this.res);
    for (let k = Math.max(0, k0); k <= Math.min(this.h - 1, k1); k++)
      for (let i = Math.max(0, i0); i <= Math.min(this.w - 1, i1); i++) if (this.data[k * this.w + i] & bits) return true;
    return false;
  }
}

/** Minimal binary min-heap keyed by number. */
export class MinHeap {
  constructor() { this.k = []; this.v = []; }
  get size() { return this.k.length; }
  push(key, val) {
    const k = this.k, v = this.v; let i = k.length; k.push(key); v.push(val);
    while (i > 0) { const p = (i - 1) >> 1; if (k[p] <= k[i]) break; [k[p], k[i]] = [k[i], k[p]]; [v[p], v[i]] = [v[i], v[p]]; i = p; }
  }
  pop() {
    const k = this.k, v = this.v; const top = v[0], tk = k[0];
    const lk = k.pop(), lv = v.pop();
    if (k.length) {
      k[0] = lk; v[0] = lv; let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1; let m = i;
        if (l < k.length && k[l] < k[m]) m = l;
        if (r < k.length && k[r] < k[m]) m = r;
        if (m === i) break;
        [k[m], k[i]] = [k[i], k[m]]; [v[m], v[i]] = [v[i], v[m]]; i = m;
      }
    }
    this.lastKey = tk;
    return top;
  }
}
