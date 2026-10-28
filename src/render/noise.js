// Deterministic helpers for procedural content: seeded RNG, integer hash, tileable value noise / fBm (bit-exact
// with the game's src/world/util.js versions so textures stay identical), non-repeating seeded simplex noise
// (2D / 3D, fBm, ridged) and scalar easing. Pure JS, allocation-free after construction, Node-safe.
//
//   const n = new Simplex(seed);  n.noise(x, y); n.noise3(x, y, z); n.fbm(x, y, 4); n.ridged(x, y, 4);
//   simplex2(x, y, seed); simplex3(x, y, z, seed);   // same, with a per-seed instance cached internally

/** Seeded PRNG (mulberry32): returns () => [0, 1). */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Integer lattice hash -> [0, 1). */
export function hash2(x, y) {
  let h = (Math.imul(x | 0, 374761393) + Math.imul(y | 0, 668265263)) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
}

/** Periodic value noise (tileable with integer period), smooth-step interpolated, [0, 1]. */
export function pnoise(x, y, period, seed = 0) {
  const xi = Math.floor(x), yi = Math.floor(y);
  const xf = x - xi, yf = y - yi;
  const u = xf * xf * (3 - 2 * xf), v = yf * yf * (3 - 2 * yf);
  // lattice corners wrapped to the period (inline: this runs a few million times per procedural texture)
  const x0 = ((xi % period) + period) % period, x1 = (((xi + 1) % period) + period) % period;
  const y0 = ((yi % period) + period) % period, y1 = (((yi + 1) % period) + period) % period;
  const sx = seed * 131;
  const a = hash2(x0 + sx, y0), b = hash2(x1 + sx, y0);
  const c = hash2(x0 + sx, y1), d = hash2(x1 + sx, y1);
  return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
}

/** Tileable fBm over the unit square (x, y in 0..1), base period in lattice cells. */
export function pfbm(x, y, basePeriod, octaves, seed = 0, gain = 0.5) {
  let amp = 1, sum = 0, norm = 0, p = basePeriod, f = basePeriod;
  for (let o = 0; o < octaves; o++) {
    sum += amp * pnoise(x * f, y * f, p, seed + o * 7);
    norm += amp; amp *= gain; p *= 2; f *= 2;
  }
  return sum / norm;
}

export const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
export const lerp = (a, b, t) => a + (b - a) * t;
export const smoothstep = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };

// ------------------------------------------------------------------ simplex noise (non-repeating, seeded)
const F2 = 0.5 * (Math.sqrt(3) - 1), G2 = (3 - Math.sqrt(3)) / 6;
const F3 = 1 / 3, G3 = 1 / 6;
// 2D gradients (12; axis + diagonal directions) and the 12 cube-edge gradients for 3D
const GX = new Float32Array([1, -1, 1, -1, 1, -1, 0, 0, 0.7071, -0.7071, 0.7071, -0.7071]);
const GY = new Float32Array([0, 0, 0, 0, 1, 1, -1, -1, 0.7071, 0.7071, -0.7071, -0.7071]);
const G3X = new Int8Array([1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0]);
const G3Y = new Int8Array([1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1]);
const G3Z = new Int8Array([0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1]);

/** Seeded simplex noise (2D / 3D) in [-1, 1] with fBm helpers. The permutation comes from mulberry32(seed). */
export class Simplex {
  constructor(seed = 1) {
    this.seed = seed;
    const r = mulberry32(seed), p = new Uint8Array(256);
    for (let i = 0; i < 256; i++) p[i] = i;
    for (let i = 255; i > 0; i--) { const j = Math.floor(r() * (i + 1)); const t = p[i]; p[i] = p[j]; p[j] = t; }
    this.perm = new Uint8Array(512); this.g = new Uint8Array(512);
    for (let i = 0; i < 512; i++) { this.perm[i] = p[i & 255]; this.g[i] = this.perm[i] % 12; }
  }

  /** 2D simplex noise in [-1, 1]. */
  noise(x, y) {
    const perm = this.perm, g = this.g;
    const s = (x + y) * F2, i = Math.floor(x + s), j = Math.floor(y + s);
    const t = (i + j) * G2, x0 = x - (i - t), y0 = y - (j - t);
    const i1 = x0 > y0 ? 1 : 0, j1 = 1 - i1;
    const x1 = x0 - i1 + G2, y1 = y0 - j1 + G2, x2 = x0 - 1 + 2 * G2, y2 = y0 - 1 + 2 * G2;
    const ii = i & 255, jj = j & 255;
    let n = 0, tt;
    tt = 0.5 - x0 * x0 - y0 * y0;
    if (tt > 0) { const q = g[ii + perm[jj]]; tt *= tt; n += tt * tt * (GX[q] * x0 + GY[q] * y0); }
    tt = 0.5 - x1 * x1 - y1 * y1;
    if (tt > 0) { const q = g[ii + i1 + perm[jj + j1]]; tt *= tt; n += tt * tt * (GX[q] * x1 + GY[q] * y1); }
    tt = 0.5 - x2 * x2 - y2 * y2;
    if (tt > 0) { const q = g[ii + 1 + perm[jj + 1]]; tt *= tt; n += tt * tt * (GX[q] * x2 + GY[q] * y2); }
    return 70 * n;
  }
  /** Alias of noise(). */
  noise2(x, y) { return this.noise(x, y); }

  /** 3D simplex noise in about [-1, 1]. */
  noise3(x, y, z) {
    const perm = this.perm, g = this.g;
    const s = (x + y + z) * F3, i = Math.floor(x + s), j = Math.floor(y + s), k = Math.floor(z + s);
    const t = (i + j + k) * G3, x0 = x - (i - t), y0 = y - (j - t), z0 = z - (k - t);
    let i1, j1, k1, i2, j2, k2; // simplex corner offsets (second and third corner)
    if (x0 >= y0) {
      if (y0 >= z0) { i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 1; k2 = 0; }
      else if (x0 >= z0) { i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 0; k2 = 1; }
      else { i1 = 0; j1 = 0; k1 = 1; i2 = 1; j2 = 0; k2 = 1; }
    } else if (y0 < z0) { i1 = 0; j1 = 0; k1 = 1; i2 = 0; j2 = 1; k2 = 1; }
    else if (x0 < z0) { i1 = 0; j1 = 1; k1 = 0; i2 = 0; j2 = 1; k2 = 1; }
    else { i1 = 0; j1 = 1; k1 = 0; i2 = 1; j2 = 1; k2 = 0; }
    const x1 = x0 - i1 + G3, y1 = y0 - j1 + G3, z1 = z0 - k1 + G3;
    const x2 = x0 - i2 + 2 * G3, y2 = y0 - j2 + 2 * G3, z2 = z0 - k2 + 2 * G3;
    const x3 = x0 - 1 + 3 * G3, y3 = y0 - 1 + 3 * G3, z3 = z0 - 1 + 3 * G3;
    const ii = i & 255, jj = j & 255, kk = k & 255;
    let n = 0, tt, q;
    tt = 0.6 - x0 * x0 - y0 * y0 - z0 * z0;
    if (tt > 0) { q = g[ii + perm[jj + perm[kk]]]; tt *= tt; n += tt * tt * (G3X[q] * x0 + G3Y[q] * y0 + G3Z[q] * z0); }
    tt = 0.6 - x1 * x1 - y1 * y1 - z1 * z1;
    if (tt > 0) { q = g[ii + i1 + perm[jj + j1 + perm[kk + k1]]]; tt *= tt; n += tt * tt * (G3X[q] * x1 + G3Y[q] * y1 + G3Z[q] * z1); }
    tt = 0.6 - x2 * x2 - y2 * y2 - z2 * z2;
    if (tt > 0) { q = g[ii + i2 + perm[jj + j2 + perm[kk + k2]]]; tt *= tt; n += tt * tt * (G3X[q] * x2 + G3Y[q] * y2 + G3Z[q] * z2); }
    tt = 0.6 - x3 * x3 - y3 * y3 - z3 * z3;
    if (tt > 0) { q = g[ii + 1 + perm[jj + 1 + perm[kk + 1]]]; tt *= tt; n += tt * tt * (G3X[q] * x3 + G3Y[q] * y3 + G3Z[q] * z3); }
    return 32 * n;
  }

  /** 2D fBm in about [-1, 1]. */
  fbm(x, y, octaves = 4, lac = 2, gain = 0.5) {
    let a = 1, f = 1, s = 0, nrm = 0;
    for (let o = 0; o < octaves; o++) { s += a * this.noise(x * f + o * 17.3, y * f - o * 9.1); nrm += a; a *= gain; f *= lac; }
    return s / nrm;
  }
  /** 3D fBm in about [-1, 1]. */
  fbm3(x, y, z, octaves = 4, lac = 2, gain = 0.5) {
    let a = 1, f = 1, s = 0, nrm = 0;
    for (let o = 0; o < octaves; o++) { s += a * this.noise3(x * f + o * 17.3, y * f - o * 9.1, z * f + o * 5.3); nrm += a; a *= gain; f *= lac; }
    return s / nrm;
  }
  /** 2D ridged fBm in [0, 1] (sharp crests). */
  ridged(x, y, octaves = 4) {
    let a = 1, f = 1, s = 0, nrm = 0;
    for (let o = 0; o < octaves; o++) { const n = 1 - Math.abs(this.noise(x * f + o * 5.7, y * f + o * 3.1)); s += a * n * n; nrm += a; a *= 0.5; f *= 2; }
    return s / nrm;
  }
}

const _simplex = new Map();
let _lastSeed = NaN, _lastSimplex = null;
function simplexFor(seed) {
  if (seed === _lastSeed) return _lastSimplex;
  let s = _simplex.get(seed);
  if (!s) { s = new Simplex(seed); _simplex.set(seed, s); }
  _lastSeed = seed; _lastSimplex = s;
  return s;
}
/** 2D simplex noise in [-1, 1] for a seed (instances cached per seed). */
export function simplex2(x, y, seed = 1) { return simplexFor(seed).noise(x, y); }
/** 3D simplex noise in about [-1, 1] for a seed (instances cached per seed). */
export function simplex3(x, y, z, seed = 1) { return simplexFor(seed).noise3(x, y, z); }
