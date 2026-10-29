// Procedural PBR fallback layers for the texture arrays (TextureSets.makeTextureArray): the tileable material painter
// and the RGBA8 packing of one array layer. Pure JS on noise.js (no three.js, no DOM) so the same code runs on the
// main thread (Node / tests / fallback) and in proceduralTextures.worker.js; both paths produce the same bytes.
//
//   proceduralPbr(kind, size, seed)           -> { alb, hgt, rgh } float maps (re-exported by TextureSets.js)
//   proceduralTexels(kind, P, seed, tint)     -> { pa, pn }: P x P RGBA8 texels of an array layer (the noise-bound
//                                                part, ~20-40 ms at 256 px on an M-series core; what the workers run)
//   tileTexels(pa, pn, P, S, A, N, o)          nearest-tiles them into the S x S layer at byte offset o of A / N
//                                                (~1 ms at 1024 px; done on the main thread, into the shared arrays)
import { pfbm, hash2 } from './noise.js';

/**
 * Fallback material painter: linear albedo (rgb 0..1), height (0..1) and roughness per pixel, tileable.
 * kinds: brick, panel, stucco, siding, corrugated, slate, grass, plain, fabric; anything else = rough concrete.
 */
export function proceduralPbr(kind, S, seed) {
  const alb = new Float32Array(S * S * 3), hgt = new Float32Array(S * S), rgh = new Float32Array(S * S);
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    const u = x / S, v = y / S, i = y * S + x;
    const n = pfbm(u, v, 8, 4, seed), f = hash2(x * 3 + seed, y * 7);
    let a = 0.72 + (n - 0.5) * 0.25 + (f - 0.5) * 0.08, h = n * 0.3, r = 0.85;
    let c = [a, a, a];
    if (kind === 'brick') {
      const rows = 12, row = Math.floor(v * rows), bw = 1 / 5, off = (row % 2) * bw * 0.5;
      const bx = (u + off) / bw, fx = bx - Math.floor(bx), fy = v * rows - row;
      const mort = fx < 0.05 || fy < 0.12;
      const k = hash2(Math.floor(bx) + 31, row + seed);
      c = mort ? [0.62, 0.6, 0.57] : [0.55 + k * 0.15, 0.27 + k * 0.08, 0.2 + k * 0.05];
      h = mort ? 0.1 : 0.7 + n * 0.2; r = mort ? 0.95 : 0.8;
    } else if (kind === 'panel') {
      const j = (u * 2) % 1 < 0.01 || (v * 2) % 1 < 0.01;
      a = 0.7 + (n - 0.5) * 0.12 + (f - 0.5) * 0.12; c = [a, a, a]; h = j ? 0 : 0.6 + n * 0.1; r = 0.7;
    } else if (kind === 'stucco') {
      a = 0.78 + (n - 0.5) * 0.1 + (f - 0.5) * 0.1; c = [a, a, a]; h = 0.5 + f * 0.3 + n * 0.2; r = 0.9;
    } else if (kind === 'siding') {
      const bd = (v * 8) % 1; a = 0.78 + (n - 0.5) * 0.06 - (bd < 0.08 ? 0.25 : 0); c = [a, a, a]; h = bd; r = 0.7;
    } else if (kind === 'corrugated') {
      const s = 0.5 + 0.5 * Math.sin(u * Math.PI * 2 * 14); a = 0.6 + s * 0.2; c = [a, a, a]; h = s; r = 0.45;
    } else if (kind === 'slate') {
      const rows = 10, row = Math.floor(v * rows), off = (row % 2) * 0.05;
      const fx = ((u + off) * 10) % 1, fy = v * rows - row, k = hash2(Math.floor((u + off) * 10), row);
      a = 0.32 + k * 0.08 - (fy > 0.85 ? 0.1 : 0) - (fx < 0.04 ? 0.08 : 0); c = [a * 0.95, a * 0.97, a]; h = fy; r = 0.75;
    } else if (kind === 'grass') {
      c = [0.18 + n * 0.12, 0.34 + n * 0.16 + f * 0.08, 0.1 + n * 0.05]; h = f; r = 0.95;
    } else if (kind === 'plain') {
      a = 0.8 + (n - 0.5) * 0.04; c = [a, a, a]; h = 0.5; r = 0.55;
    } else if (kind === 'fabric') {
      const w = ((Math.floor(u * 160) + Math.floor(v * 160)) % 2) * 0.06; a = 0.75 + w + (n - 0.5) * 0.05; c = [a, a, a]; h = w * 8; r = 0.9;
    }
    alb[i * 3] = c[0]; alb[i * 3 + 1] = c[1]; alb[i * 3 + 2] = c[2]; hgt[i] = h; rgh[i] = r;
  }
  return { alb, hgt, rgh };
}

/**
 * Texels of a procedural array layer at P x P (makeTextureArray: 256 px, tiled up to the array size by tileTexels):
 * pa = sRGB albedo (tintable sets normalised to the photo sets' neutral mean) + AO 255 in alpha, pn = tangent normal xy
 * from the height map's central differences + roughness in b. Fresh buffers, so a worker can transfer them.
 */
export function proceduralTexels(kind, P, seed, tint) {
  const pr = proceduralPbr(kind, P, seed);
  if (tint) { // same neutral mean as the normalised photo sets
    const m = [0, 0, 0];
    for (let i = 0; i < P * P; i++) { m[0] += pr.alb[i * 3]; m[1] += pr.alb[i * 3 + 1]; m[2] += pr.alb[i * 3 + 2]; }
    const k = m.map((v) => 0.58 / Math.max(0.02, v / (P * P)));
    for (let i = 0; i < P * P; i++) for (let c = 0; c < 3; c++) pr.alb[i * 3 + c] *= k[c];
  }
  // texel values once per source texel (P x P), then nearest-tiled into the S x S layer (tileTexels)
  const PA = new Uint8Array(P * P * 4), PN = new Uint8Array(P * P * 4);
  for (let py = 0; py < P; py++) for (let px = 0; px < P; px++) {
    const i = py * P + px, d = i * 4;
    PA[d] = Math.min(255, Math.pow(pr.alb[i * 3], 1 / 2.2) * 255); PA[d + 1] = Math.min(255, Math.pow(pr.alb[i * 3 + 1], 1 / 2.2) * 255); PA[d + 2] = Math.min(255, Math.pow(pr.alb[i * 3 + 2], 1 / 2.2) * 255); PA[d + 3] = 255;
    const xm = (px + P - 1) % P, xp = (px + 1) % P, ym = (py + P - 1) % P, yp = (py + 1) % P;
    const dx = (pr.hgt[py * P + xp] - pr.hgt[py * P + xm]) * 2.5, dy = (pr.hgt[yp * P + px] - pr.hgt[ym * P + px]) * 2.5;
    const ln = Math.hypot(dx, dy, 1);
    PN[d] = (-dx / ln * 0.5 + 0.5) * 255; PN[d + 1] = (-dy / ln * 0.5 + 0.5) * 255; PN[d + 2] = pr.rgh[i] * 255; PN[d + 3] = 255;
  }
  return { pa: PA, pn: PN };
}

/**
 * Nearest-tile P x P RGBA8 texels (proceduralTexels) into the S x S layer that starts at byte offset o of the RGBA8
 * arrays A / N: whole texels at a time (32-bit views), repeated rows copied once.
 */
export function tileTexels(PA, PN, P, S, A, N, o) {
  const row = S * 4;
  const xs = new Int32Array(S); for (let x = 0; x < S; x++) xs[x] = Math.floor((x * P) / S);
  const A32 = new Uint32Array(A.buffer, A.byteOffset, A.byteLength >> 2), N32 = new Uint32Array(N.buffer, N.byteOffset, N.byteLength >> 2);
  const PA32 = new Uint32Array(PA.buffer, PA.byteOffset, PA.byteLength >> 2), PN32 = new Uint32Array(PN.buffer, PN.byteOffset, PN.byteLength >> 2);
  let lastPy = -1, lastRow = 0;
  for (let y = 0; y < S; y++) {
    const py = Math.floor((y * P) / S), d0 = o + y * row;
    if (py === lastPy) { A.copyWithin(d0, lastRow, lastRow + row); N.copyWithin(d0, lastRow, lastRow + row); continue; }
    const s0 = py * P, d4 = d0 >> 2;
    for (let x = 0; x < S; x++) { const sx = s0 + xs[x]; A32[d4 + x] = PA32[sx]; N32[d4 + x] = PN32[sx]; }
    lastPy = py; lastRow = d0;
  }
}
