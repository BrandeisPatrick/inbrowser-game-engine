// CPU texture post-processing for alpha-tested atlases (foliage, impostors): transparent-texel colour fill /
// dilation, mip chains built in linear space with alpha-weighted colour and per-region coverage-preserving alpha
// (so alpha-tested cut-outs do not thin out with distance), and DataTextures with an explicit mip chain.
//   fillTransparent(rgba, W, x, y, w, h, thr)   dilateRGB(rgba, W, x, y, w, h)
//   buildMipChain(rgba, W, H, regions, srgb)    -> [{ data, width, height }]
//   dataTextureFromMips(levels, colorSpace)     -> THREE.DataTexture
import * as THREE from 'three';

const S2L = new Float32Array(256);
for (let i = 0; i < 256; i++) { const c = i / 255; S2L[i] = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); }
const L2S = new Uint8Array(4096);
for (let i = 0; i < 4096; i++) { const c = i / 4095; L2S[i] = Math.round(255 * (c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055)); }

/** Fill transparent texels with nearby opaque colour (mean fill + 1-texel dilation) inside rect. */
export function fillTransparent(d, W, rx, ry, rw, rh, thr = 12) {
  let sr = 0, sg = 0, sb = 0, n = 0;
  const x1 = Math.min(W, rx + rw), y1 = ry + rh;
  for (let y = ry; y < y1; y++) for (let x = rx; x < x1; x++) {
    const i = (y * W + x) * 4;
    if (d[i + 3] > 127) { sr += d[i]; sg += d[i + 1]; sb += d[i + 2]; n++; }
  }
  const mr = n ? sr / n : 60, mg = n ? sg / n : 75, mb = n ? sb / n : 40;
  for (let y = ry; y < y1; y++) for (let x = rx; x < x1; x++) {
    const i = (y * W + x) * 4;
    if (d[i + 3] < thr) { d[i] = mr; d[i + 1] = mg; d[i + 2] = mb; d[i + 3] = 0; }
  }
  // one dilation pass so bilinear filtering at the alpha edge only mixes leaf colours
  const W4 = W * 4;
  for (let y = ry + 1; y < y1 - 1; y++) for (let x = rx + 1; x < x1 - 1; x++) {
    const i = (y * W + x) * 4;
    if (d[i + 3] !== 0) continue;
    let j = i - 4;
    if (d[j + 3] > 127 || d[(j = i + 4) + 3] > 127 || d[(j = i - W4) + 3] > 127 || d[(j = i + W4) + 3] > 127) { d[i] = d[j]; d[i + 1] = d[j + 1]; d[i + 2] = d[j + 2]; }
  }
}

function coverage(d, W, r, thr = 128) {
  let c = 0, n = 0;
  for (let y = r.y; y < r.y + r.h; y++) for (let x = r.x; x < r.x + r.w; x++) { n++; if (d[(y * W + x) * 4 + 3] >= thr) c++; }
  return n ? c / n : 0;
}

/**
 * Mip chain [{data,width,height}] from RGBA8 level 0. Colour is averaged in linear space (srgb=true) weighted by
 * alpha; alpha in each `regions` rect (level-0 pixel coords, cov=true) is rescaled so the fraction of texels that
 * pass alphaTest 0.5 stays what it was at level 0.
 */
export function buildMipChain(data, W, H, regions = [], srgb = true) {
  const levels = [{ data, width: W, height: H }];
  const targets = regions.map((r) => (r.cov ? coverage(data, W, r) : -1));
  let src = data, w = W, h = H, lvl = 0;
  const hist = new Uint32Array(256);
  while (w > 1 || h > 1) {
    const nw = Math.max(1, w >> 1), nh = Math.max(1, h >> 1);
    const dst = new Uint8Array(nw * nh * 4);
    for (let y = 0; y < nh; y++) {
      const y0 = Math.min(h - 1, y * 2), y1 = Math.min(h - 1, y * 2 + 1);
      for (let x = 0; x < nw; x++) {
        const x0 = Math.min(w - 1, x * 2), x1 = Math.min(w - 1, x * 2 + 1);
        const i0 = (y0 * w + x0) * 4, i1 = (y0 * w + x1) * 4, i2 = (y1 * w + x0) * 4, i3 = (y1 * w + x1) * 4;
        const a0 = src[i0 + 3], a1 = src[i1 + 3], a2 = src[i2 + 3], a3 = src[i3 + 3];
        const as = a0 + a1 + a2 + a3;
        const o = (y * nw + x) * 4;
        if (srgb) {
          if (as > 0) for (let c = 0; c < 3; c++) dst[o + c] = L2S[Math.min(4095, (((S2L[src[i0 + c]] * a0 + S2L[src[i1 + c]] * a1 + S2L[src[i2 + c]] * a2 + S2L[src[i3 + c]] * a3) / as) * 4095 + 0.5) | 0)];
          else for (let c = 0; c < 3; c++) dst[o + c] = L2S[Math.min(4095, (((S2L[src[i0 + c]] + S2L[src[i1 + c]] + S2L[src[i2 + c]] + S2L[src[i3 + c]]) * 0.25) * 4095 + 0.5) | 0)];
        } else if (as > 0) for (let c = 0; c < 3; c++) dst[o + c] = (src[i0 + c] * a0 + src[i1 + c] * a1 + src[i2 + c] * a2 + src[i3 + c] * a3) / as + 0.5;
        else for (let c = 0; c < 3; c++) dst[o + c] = (src[i0 + c] + src[i1 + c] + src[i2 + c] + src[i3 + c]) * 0.25 + 0.5;
        dst[o + 3] = (as + 2) >> 2;
      }
    }
    lvl++;
    const f = 1 << lvl;
    regions.forEach((r, k) => {
      if (targets[k] < 0) return;
      const rx = Math.floor(r.x / f), ry = Math.floor(r.y / f), rx1 = Math.min(nw, Math.ceil((r.x + r.w) / f)), ry1 = Math.min(nh, Math.ceil((r.y + r.h) / f));
      if (rx1 - rx < 2 || ry1 - ry < 2) return;
      hist.fill(0); let n = 0;
      for (let y = ry; y < ry1; y++) for (let x = rx; x < rx1; x++) { hist[dst[(y * nw + x) * 4 + 3]]++; n++; }
      let acc = 0, t = 255;
      for (; t > 1; t--) { acc += hist[t]; if (acc / n >= targets[k]) break; }
      const s = Math.min(6, Math.max(0.6, 127.5 / t));
      if (Math.abs(s - 1) < 0.02) return;
      for (let y = ry; y < ry1; y++) for (let x = rx; x < rx1; x++) { const i = (y * nw + x) * 4 + 3; dst[i] = Math.min(255, dst[i] * s + 0.5); }
    });
    levels.push({ data: dst, width: nw, height: nh });
    src = dst; w = nw; h = nh;
  }
  return levels;
}

/** RGBA8 DataTexture with an explicit mip chain. */
export function dataTextureFromMips(levels, colorSpace) {
  const l0 = levels[0];
  const t = new THREE.DataTexture(l0.data, l0.width, l0.height, THREE.RGBAFormat, THREE.UnsignedByteType);
  t.mipmaps = levels;
  t.generateMipmaps = false;
  t.minFilter = THREE.LinearMipmapLinearFilter;
  t.magFilter = THREE.LinearFilter;
  t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
  t.colorSpace = colorSpace;
  t.flipY = false;
  t.unpackAlignment = 4;
  t.needsUpdate = true;
  return t;
}


/** Copy RGB from an opaque 4-neighbour into fully transparent texels (alpha becomes 1/255) inside a rect. */
export function dilateRGB(d, W, rx, ry, rw, rh) {
  const mark = [], W4 = W * 4;
  for (let y = ry + 1; y < ry + rh - 1; y++) for (let x = rx + 1; x < rx + rw - 1; x++) {
    const i = (y * W + x) * 4;
    if (d[i + 3] > 0) continue;
    let j = i - 4;
    if (d[j + 3] > 0 || d[(j = i + 4) + 3] > 0 || d[(j = i - W4) + 3] > 0 || d[(j = i + W4) + 3] > 0) mark.push(i, j);
  }
  for (let k = 0; k < mark.length; k += 2) { const i = mark[k], j = mark[k + 1]; d[i] = d[j]; d[i + 1] = d[j + 1]; d[i + 2] = d[j + 2]; d[i + 3] = 1; }
}
