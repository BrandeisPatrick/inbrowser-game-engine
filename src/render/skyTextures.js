// Procedural textures for the sky: a tileable cloud-noise atlas (generated once in JS, no assets) and
// small lens-flare sprites drawn on canvases. Used by Atmosphere; no external assets.
import * as THREE from 'three';

function mulberry(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makePerm(seed) {
  const r = mulberry(seed), p = new Uint8Array(256);
  for (let i = 0; i < 256; i++) p[i] = i;
  for (let i = 255; i > 0; i--) { const j = Math.floor(r() * (i + 1)); const t = p[i]; p[i] = p[j]; p[j] = t; }
  const pp = new Uint8Array(512);
  for (let i = 0; i < 512; i++) pp[i] = p[i & 255];
  return pp;
}

const GX = [1, -1, 1, -1, 1, -1, 0, 0], GY = [1, 1, -1, -1, 0, 0, 1, -1];
const fade = (t) => t * t * t * (t * (t * 6 - 15) + 10);

/** Periodic 2D gradient noise in [-1, 1] (period p lattice cells, p <= 256). */
function pnoise(x, y, p, perm) {
  const X = Math.floor(x), Y = Math.floor(y), xf = x - X, yf = y - Y;
  const X0 = ((X % p) + p) % p, Y0 = ((Y % p) + p) % p, X1 = (X0 + 1) % p, Y1 = (Y0 + 1) % p;
  const h00 = perm[perm[X0] + Y0] & 7, h10 = perm[perm[X1] + Y0] & 7, h01 = perm[perm[X0] + Y1] & 7, h11 = perm[perm[X1] + Y1] & 7;
  const n00 = GX[h00] * xf + GY[h00] * yf, n10 = GX[h10] * (xf - 1) + GY[h10] * yf;
  const n01 = GX[h01] * xf + GY[h01] * (yf - 1), n11 = GX[h11] * (xf - 1) + GY[h11] * (yf - 1);
  const u = fade(xf), v = fade(yf);
  const a = n00 + (n10 - n00) * u, b = n01 + (n11 - n01) * u;
  return (a + (b - a) * v) * 0.7071;
}

function fbmChannel(size, baseFreq, octaves, seed, billow, out, ch, equalize = false) {
  const perm = makePerm(seed), vals = new Float32Array(size * size);
  let mn = Infinity, mx = -Infinity;
  for (let j = 0; j < size; j++) {
    for (let i = 0; i < size; i++) {
      let f = baseFreq, a = 1, s = 0;
      const u = i / size, v = j / size;
      for (let o = 0; o < octaves; o++) {
        let n = pnoise(u * f, v * f, f, perm);
        if (billow) n = 0.6 * n + 0.4 * (0.5 - Math.abs(n)); // part billow: puffy tops without ridges
        s += n * a; f *= 2; a *= 0.52;
      }
      vals[j * size + i] = s;
      if (s < mn) mn = s; if (s > mx) mx = s;
    }
  }
  const N = size * size;
  if (equalize) { // rank-normalise: a coverage threshold c then covers ~c of the sky
    const sorted = Float32Array.from(vals).sort(), q = new Float32Array(256);
    for (let k = 0; k < 256; k++) q[k] = sorted[Math.min(N - 1, Math.floor(((k + 1) / 256) * N))];
    for (let i = 0; i < N; i++) {
      const v = vals[i];
      let lo = 0, hi = 255;
      while (lo < hi) { const m = (lo + hi) >> 1; if (q[m] < v) lo = m + 1; else hi = m; }
      out[i * 4 + ch] = lo;
    }
    return;
  }
  const k = 255 / (mx - mn || 1);
  for (let i = 0; i < N; i++) out[i * 4 + ch] = Math.round((vals[i] - mn) * k);
}

/**
 * Tileable RGBA noise atlas: R = puffy cumulus fbm, G = soft fbm for cirrus (stretched in the shader),
 * B = fine detail (cloud erosion, moon maria), A = 255. Returns { texture, data, size }.
 */
export function makeCloudTexture(size = 512) {
  const data = new Uint8Array(size * size * 4);
  fbmChannel(size, 4, 6, 1337, true, data, 0, true);
  fbmChannel(size, 3, 5, 4242, false, data, 1, true);
  fbmChannel(size, 16, 3, 99, false, data, 2);
  for (let i = 0; i < size * size; i++) data[i * 4 + 3] = 255;
  const tex = new THREE.DataTexture(data, size, size, THREE.RGBAFormat, THREE.UnsignedByteType);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.magFilter = THREE.LinearFilter;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.generateMipmaps = true;
  tex.anisotropy = 4;
  tex.colorSpace = THREE.NoColorSpace;
  tex.needsUpdate = true;
  tex.name = 'cloudNoise';
  return { texture: tex, data, size };
}

/** Bilinear CPU sample of one channel of the atlas at repeat-wrapped uv (0..1 per tile). */
export function sampleCloud(atlas, u, v, ch = 0) {
  const s = atlas.size, d = atlas.data;
  const x = (u - Math.floor(u)) * s - 0.5, y = (v - Math.floor(v)) * s - 0.5;
  const x0 = Math.floor(x), y0 = Math.floor(y), fx = x - x0, fy = y - y0;
  const w = (a) => ((a % s) + s) % s;
  const at = (i, j) => d[(w(j) * s + w(i)) * 4 + ch] / 255;
  const a = at(x0, y0) + (at(x0 + 1, y0) - at(x0, y0)) * fx;
  const b = at(x0, y0 + 1) + (at(x0 + 1, y0 + 1) - at(x0, y0 + 1)) * fx;
  return a + (b - a) * fy;
}

function canvasTex(size, draw) {
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const g = c.getContext('2d');
  draw(g, size);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

/** Small procedural lens-flare sprites: core glow with faint streaks, soft hexagon ghost, thin ring, soft dot. */
export function makeFlareTextures() {
  const glow = canvasTex(256, (g, s) => {
    const c = s / 2;
    let gr = g.createRadialGradient(c, c, 0, c, c, c);
    gr.addColorStop(0, 'rgba(255,255,255,1)');
    gr.addColorStop(0.06, 'rgba(255,250,235,0.85)');
    gr.addColorStop(0.2, 'rgba(255,225,180,0.25)');
    gr.addColorStop(0.55, 'rgba(255,200,150,0.05)');
    gr.addColorStop(1, 'rgba(255,190,140,0)');
    g.fillStyle = gr; g.fillRect(0, 0, s, s);
    g.globalCompositeOperation = 'lighter';
    // faint diffraction streaks
    for (let k = 0; k < 6; k++) {
      g.save(); g.translate(c, c); g.rotate((k / 6) * Math.PI + 0.26);
      const lg = g.createLinearGradient(-c, 0, c, 0);
      lg.addColorStop(0, 'rgba(255,240,220,0)'); lg.addColorStop(0.5, 'rgba(255,240,220,0.18)'); lg.addColorStop(1, 'rgba(255,240,220,0)');
      g.fillStyle = lg; g.fillRect(-c, -0.8, s, 1.6);
      g.restore();
    }
  });
  const hex = canvasTex(128, (g, s) => {
    const c = s / 2, r = s * 0.44;
    g.beginPath();
    for (let k = 0; k < 6; k++) { const a = (k / 6) * Math.PI * 2 + Math.PI / 6; g[k ? 'lineTo' : 'moveTo'](c + Math.cos(a) * r, c + Math.sin(a) * r); }
    g.closePath();
    const gr = g.createRadialGradient(c, c, 0, c, c, r);
    gr.addColorStop(0, 'rgba(255,255,255,0.35)'); gr.addColorStop(0.8, 'rgba(255,255,255,0.55)'); gr.addColorStop(1, 'rgba(255,255,255,0.8)');
    g.fillStyle = gr; g.filter = 'blur(1.5px)'; g.fill();
  });
  const ring = canvasTex(256, (g, s) => {
    const c = s / 2;
    const gr = g.createRadialGradient(c, c, s * 0.36, c, c, s * 0.48);
    gr.addColorStop(0, 'rgba(255,255,255,0)');
    gr.addColorStop(0.35, 'rgba(255,120,80,0.35)');
    gr.addColorStop(0.55, 'rgba(160,255,170,0.35)');
    gr.addColorStop(0.75, 'rgba(120,150,255,0.35)');
    gr.addColorStop(1, 'rgba(255,255,255,0)');
    g.fillStyle = gr; g.fillRect(0, 0, s, s);
  });
  const dot = canvasTex(64, (g, s) => {
    const c = s / 2, gr = g.createRadialGradient(c, c, 0, c, c, c);
    gr.addColorStop(0, 'rgba(255,255,255,0.9)'); gr.addColorStop(0.5, 'rgba(255,255,255,0.35)'); gr.addColorStop(1, 'rgba(255,255,255,0)');
    g.fillStyle = gr; g.fillRect(0, 0, s, s);
  });
  return { glow, hex, ring, dot };
}
