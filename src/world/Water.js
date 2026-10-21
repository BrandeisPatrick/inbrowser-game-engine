// Bay / harbour water (open water plane at the world's water level + still ponds).
//
// - Geometry: one camera-following polar grid (dense near the camera, coarse to 30 km) snapped to 0.5 m and
//   placed in onBeforeRender, so it works for any camera and needs no per-frame CPU work. Within ~300 m it
//   carries a small Gerstner swell (7 waves, faded out per wave before the grid gets too coarse for it).
// - Shading: patched MeshStandardMaterial (so shadows, the sun / moon, street-lamp point lights, the PMREM
//   sky environment, fog and tone mapping all behave like the rest of the scene, with or without post-FX).
//   Four scrolling normal-map octaves (two wind directions, 2.9 m .. 61 m tiles) modulated by large "cat's paw"
//   wind patches (610 m / 1.9 km noise) so no tile repeat is visible from 2 m to the horizon; roughness grows
//   with distance (the sun glitter widens into a streak); a sparkle term is added for every direct light
//   (sun, moon, lamps) using a sharper normal; the sky reflection keeps its reflection vector above the
//   horizon; F0 = 0.02 (water). Body colour: estuary green-grey (San Francisco Bay tuning), lighter and greener near the shore.
// - Shore: a signed distance-to-shore field (metres; negative under land / piers) baked at load from the land
//   and pier polygons (exact segment distance near the edge, EDT elsewhere): two textures, 2.5 m texels over
//   the detailed district and 18 m over the 9 km map. Drives foam + wet line along seawalls and piers, the
//   near-shore tint and darker water under piers.
// - Horizon: the far end of the disk fades into the (tone-mapped) sky colour in the view direction, sampled from
//   the same PMREM environment the sky box is rendered from, so the water meets the sky box seamlessly at the
//   camera far plane.
// Per frame: uTime / uNight uniforms only (World.update -> Water.update).
import * as THREE from 'three';
import { pfbm, mulberry32, clamp } from './geom.js';
import { Raster } from '../physics/spatial.js';

const G = 9.81;
// Prevailing westerly: waves run toward the east-north-east (+x east, -z north)
const WIND = Math.atan2(-0.36, 0.93);

/** Gerstner swell. Each: wavelength m, amplitude m, direction offset rad, steepness Q, phase. */
export const WAVES = [
  [42, 0.07, -0.12, 0.6, 0.0],
  [27, 0.06, 0.28, 0.65, 1.7],
  [17, 0.045, -0.46, 0.7, 4.1],
  [11, 0.032, 0.7, 0.7, 2.3],
  [7.1, 0.022, -0.2, 0.65, 5.2],
  [4.6, 0.014, 1.05, 0.6, 0.9],
  [3.1, 0.009, -0.9, 0.55, 3.3],
].map(([L, a, off, q, ph]) => {
  const k = (2 * Math.PI) / L, ang = WIND + off;
  // the polar grid spacing is up to ~6.5 % of the radius: fade each wave out before the grid undersamples it
  const fade0 = Math.min(220, L * 3.0);
  return { L, k, a, dx: Math.cos(ang), dz: Math.sin(ang), w: Math.sqrt(G * k), q, ph, fade0, fade1: fade0 * 1.5 };
});
export const NWAVES = WAVES.length;

/** Vertical wave displacement (m, relative to the still level) at (x,z), time t, camera (cx,cz) - mirrors the vertex shader. */
export function waveHeight(x, z, t, cx, cz) {
  const d = Math.hypot(x - cx, z - cz);
  let y = 0;
  for (const W of WAVES) {
    if (d >= W.fade1) continue;
    const f = d <= W.fade0 ? 1 : 1 - smooth((d - W.fade0) / (W.fade1 - W.fade0));
    y += W.a * f * Math.sin(W.k * (W.dx * x + W.dz * z) - W.w * t + W.ph);
  }
  return y;
}
/** Approximate wave slope (dh/dx, dh/dz) at (x,z) for boat pitch / roll. */
export function waveSlope(x, z, t, cx, cz, out = [0, 0]) {
  const d = Math.hypot(x - cx, z - cz);
  let sx = 0, sz = 0;
  for (const W of WAVES) {
    if (d >= W.fade1) continue;
    const f = d <= W.fade0 ? 1 : 1 - smooth((d - W.fade0) / (W.fade1 - W.fade0));
    const c = W.a * f * W.k * Math.cos(W.k * (W.dx * x + W.dz * z) - W.w * t + W.ph);
    sx += c * W.dx; sz += c * W.dz;
  }
  out[0] = sx; out[1] = sz;
  return out;
}
const smooth = (t) => (t <= 0 ? 0 : t >= 1 ? 1 : t * t * (3 - 2 * t));

/** Uniforms + GLSL for the swell, shared with the boat wakes (boats.js). */
export function waveUniforms() {
  return {
    uWaveA: { value: WAVES.map((W) => new THREE.Vector4(W.dx, W.dz, W.k, W.a)) },
    uWaveB: { value: WAVES.map((W) => new THREE.Vector4(W.w, W.q, W.ph, W.fade0)) },
  };
}
export const WAVE_GLSL = /* glsl */`
#define NWAVES ${NWAVES}
uniform vec4 uWaveA[NWAVES]; // dir.x, dir.z, k, amplitude
uniform vec4 uWaveB[NWAVES]; // omega, steepness, phase, fade start
// Gerstner displacement at world xz (dist = horizontal distance to the camera); n = surface normal (world)
vec3 gerstner(vec2 p, float dist, float t, out vec3 n) {
  vec3 d = vec3(0.0);
  n = vec3(0.0, 1.0, 0.0);
  for (int i = 0; i < NWAVES; i++) {
    vec4 A = uWaveA[i]; vec4 B = uWaveB[i];
    float a = A.w * (1.0 - smoothstep(B.w, B.w * 1.5, dist));
    if (a <= 0.0) continue;
    float th = A.z * dot(A.xy, p) - B.x * t + B.z;
    float c = cos(th), s = sin(th);
    d.xz += B.y * a * A.xy * c;
    d.y += a * s;
    float wa = A.z * a;
    n.x -= A.x * wa * c;
    n.z -= A.y * wa * c;
    n.y -= B.y * wa * s;
  }
  n = normalize(n);
  return d;
}
`;

// ====================================================================== textures
/** Tileable water normal map: sum of integer-wavenumber waves biased to the +u direction (wind), 512 px. */
export function makeWaterNormalMap(renderer, S = 512) {
  const rnd = mulberry32(9071);
  const waves = [];
  for (let i = 0; i < 46; i++) {
    const mag = 2 + Math.floor(Math.pow(rnd(), 1.6) * 44);            // cycles per tile
    let ang = (rnd() - 0.5) * 2.2;                                     // +-63 deg around the wind
    if (rnd() < 0.18) ang += Math.PI;                                  // a little back-scatter
    let kx = Math.round(Math.cos(ang) * mag), ky = Math.round(Math.sin(ang) * mag);
    if (!kx && !ky) kx = 2;
    const km = Math.hypot(kx, ky);
    waves.push({ kx, ky, a: Math.pow(km, -1.18) * (0.75 + rnd() * 0.5), ph: rnd() * Math.PI * 2 });
  }
  const gx = new Float32Array(S * S), gy = new Float32Array(S * S);
  const TWO_PI = Math.PI * 2;
  for (const w of waves) {
    const ax = w.a * TWO_PI * w.kx, ay = w.a * TWO_PI * w.ky;
    // separable phase: sin(a+b) via row / column tables
    const cu = new Float32Array(S), su = new Float32Array(S), cv = new Float32Array(S), sv = new Float32Array(S);
    for (let i = 0; i < S; i++) { const t = (TWO_PI * w.kx * i) / S + w.ph; cu[i] = Math.cos(t); su[i] = Math.sin(t); }
    for (let j = 0; j < S; j++) { const t = (TWO_PI * w.ky * j) / S; cv[j] = Math.cos(t); sv[j] = Math.sin(t); }
    for (let j = 0; j < S; j++) {
      const c2 = cv[j], s2 = sv[j], row = j * S;
      for (let i = 0; i < S; i++) {
        const s = su[i] * c2 + cu[i] * s2; // sin(t1 + t2)
        gx[row + i] -= ax * s; gy[row + i] -= ay * s;
      }
    }
  }
  // normalise slope magnitude
  let m2 = 0;
  for (let i = 0; i < S * S; i++) m2 += gx[i] * gx[i] + gy[i] * gy[i];
  const k = 0.38 / Math.sqrt(m2 / (S * S));
  const data = new Uint8Array(S * S * 4);
  for (let i = 0; i < S * S; i++) {
    const x = gx[i] * k, y = gy[i] * k, l = Math.hypot(x, y, 1);
    data[i * 4] = (-x / l * 0.5 + 0.5) * 255; data[i * 4 + 1] = (-y / l * 0.5 + 0.5) * 255; data[i * 4 + 2] = (1 / l * 0.5 + 0.5) * 255; data[i * 4 + 3] = 255;
  }
  const t = new THREE.DataTexture(data, S, S, THREE.RGBAFormat);
  return finishTex(t, renderer, false);
}

/** Tileable noise: R = fbm (4 cells), G = fbm (8), B = fbm (16), A = foam cells. 256 px, linear. */
export function makeNoiseTexture(renderer, S = 256) {
  const data = new Uint8Array(S * S * 4);
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    const u = x / S, v = y / S, o = (y * S + x) * 4;
    const a = pfbm(u, v, 4, 4, 301), b = pfbm(u, v, 8, 4, 311), c = pfbm(u, v, 16, 3, 321);
    const d = pfbm(u, v, 32, 2, 331);
    const cells = Math.pow(clamp(1 - Math.abs(d - 0.5) * 4.2, 0, 1), 1.5) * 0.6 + c * 0.4;
    data[o] = a * 255; data[o + 1] = b * 255; data[o + 2] = c * 255; data[o + 3] = clamp(cells, 0, 1) * 255;
  }
  const t = new THREE.DataTexture(data, S, S, THREE.RGBAFormat);
  return finishTex(t, renderer, false);
}

function finishTex(t, renderer, srgb) {
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
  t.magFilter = THREE.LinearFilter;
  t.minFilter = THREE.LinearMipmapLinearFilter;
  t.generateMipmaps = true;
  t.anisotropy = renderer ? Math.min(8, renderer.capabilities.getMaxAnisotropy()) : 4;
  t.needsUpdate = true;
  return t;
}

// ====================================================================== shoreline distance field
/** 1-D squared Euclidean distance transform (Felzenszwalb & Huttenlocher). */
function edt1d(f, n, d, v, z) {
  let k = 0; v[0] = 0; z[0] = -1e20; z[1] = 1e20;
  for (let q = 1; q < n; q++) {
    let s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
    while (s <= z[k]) { k--; s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]); }
    k++; v[k] = q; z[k] = s; z[k + 1] = 1e20;
  }
  k = 0;
  for (let q = 0; q < n; q++) { while (z[k + 1] < q) k++; const dq = q - v[k]; d[q] = dq * dq + f[v[k]]; }
}
/** Squared distance (cells) to the nearest cell where mask == target. */
function edt2d(mask, W, H, target) {
  const INF = 1e20, out = new Float32Array(W * H);
  const n = Math.max(W, H), f = new Float64Array(n), d = new Float64Array(n), v = new Int32Array(n), z = new Float64Array(n + 1);
  for (let i = 0; i < W; i++) { // columns
    for (let k = 0; k < H; k++) f[k] = mask[k * W + i] === target ? 0 : INF;
    edt1d(f, H, d, v, z);
    for (let k = 0; k < H; k++) out[k * W + i] = d[k];
  }
  for (let k = 0; k < H; k++) { // rows
    const r = k * W;
    for (let i = 0; i < W; i++) f[i] = out[r + i];
    edt1d(f, W, d, v, z);
    for (let i = 0; i < W; i++) out[r + i] = d[i];
  }
  return out;
}

/** Shore segments (land + pier polygon edges, minus the artificial 9 km box cut). */
export function shoreSegments(world) {
  const OB = world.outerBox, e = 2;
  const onBox = (a, b) => (Math.abs(a[0] - OB.minX) < e && Math.abs(b[0] - OB.minX) < e) || (Math.abs(a[0] - OB.maxX) < e && Math.abs(b[0] - OB.maxX) < e)
    || (Math.abs(a[1] - OB.minZ) < e && Math.abs(b[1] - OB.minZ) < e) || (Math.abs(a[1] - OB.maxZ) < e && Math.abs(b[1] - OB.maxZ) < e);
  const segs = [];
  for (const L of [...world.landPolys, ...(world.pierPolys || [])]) {
    const r = L.ring;
    for (let i = 0; i < r.length; i++) {
      const a = r[i], b = r[(i + 1) % r.length];
      if (onBox(a, b)) continue;
      segs.push([a[0], a[1], b[0], b[1]]);
    }
  }
  return segs;
}

function bakeField(world, segs, R, outerClamp) {
  const n = R.n, cell = R.size / n;
  const ras = new Raster(R.minX, R.minZ, R.minX + R.size - cell * 0.01, R.minZ + R.size - cell * 0.01, cell);
  const W = ras.w, H = ras.h;
  for (const L of [...world.landPolys, ...(world.pierPolys || [])]) ras.fillPolygon([L.ring], 1, 'or');
  const land = ras.data;
  if (outerClamp) { // outside the 9 km square: copy the nearest edge cell (the land polygons are cut there)
    const OB = world.outerBox;
    const i0 = clamp(Math.ceil((OB.minX - R.minX) / cell), 0, W - 1), i1 = clamp(Math.floor((OB.maxX - R.minX) / cell) - 1, 0, W - 1);
    const k0 = clamp(Math.ceil((OB.minZ - R.minZ) / cell), 0, H - 1), k1 = clamp(Math.floor((OB.maxZ - R.minZ) / cell) - 1, 0, H - 1);
    for (let k = 0; k < H; k++) for (let i = 0; i < W; i++) {
      if (i >= i0 && i <= i1 && k >= k0 && k <= k1) continue;
      land[k * W + i] = land[clamp(k, k0, k1) * W + clamp(i, i0, i1)];
    }
  }
  const dL = edt2d(land, W, H, 1), dW = edt2d(land, W, H, 0);
  const sd = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) sd[i] = land[i] ? -(Math.sqrt(dW[i]) - 0.5) * cell : (Math.sqrt(dL[i]) - 0.5) * cell;
  // exact distance to the polygon edges near the shore (no staircase along diagonal walls)
  const band = cell * 4, B = Math.max(band * 2, cell * 8);
  const hash = new Map();
  const key = (i, k) => i * 73856093 ^ k * 19349663;
  segs.forEach((s, si) => {
    const i0 = Math.floor((Math.min(s[0], s[2]) - band) / B), i1 = Math.floor((Math.max(s[0], s[2]) + band) / B);
    const k0 = Math.floor((Math.min(s[1], s[3]) - band) / B), k1 = Math.floor((Math.max(s[1], s[3]) + band) / B);
    for (let i = i0; i <= i1; i++) for (let k = k0; k <= k1; k++) { const kk = key(i, k); let a = hash.get(kk); if (!a) hash.set(kk, (a = [])); a.push(si); }
  });
  for (let k = 0; k < H; k++) for (let i = 0; i < W; i++) {
    const j = k * W + i;
    if (Math.abs(sd[j]) > band) continue;
    const x = R.minX + (i + 0.5) * cell, z = R.minZ + (k + 0.5) * cell;
    const list = hash.get(key(Math.floor(x / B), Math.floor(z / B)));
    if (!list) continue;
    let best = Infinity;
    for (const si of list) {
      const s = segs[si], dx = s[2] - s[0], dz = s[3] - s[1], L2 = dx * dx + dz * dz;
      let t = L2 > 0 ? ((x - s[0]) * dx + (z - s[1]) * dz) / L2 : 0;
      t = t < 0 ? 0 : t > 1 ? 1 : t;
      const px = s[0] + dx * t - x, pz = s[1] + dz * t - z, d2 = px * px + pz * pz;
      if (d2 < best) best = d2;
    }
    if (best < Infinity) { const d = Math.sqrt(best); sd[j] = land[j] ? -d : d; }
  }
  return { sd, W, H, cell };
}

/**
 * Bake the signed shoreline distance textures. fine: RG8 over the detailed district (R: -8..56 m @0.25,
 * G: -32..223 m @1); coarse: R8 over the 9 km map (-64..446 m @2).
 */
export function buildShoreField(world, renderer) {
  const t0 = performance.now();
  const segs = shoreSegments(world);
  const b = world.bounds, OB = world.outerBox;
  const fs = Math.max(b.maxX - b.minX, b.maxZ - b.minZ) + 360;
  const fineR = { minX: (b.minX + b.maxX) / 2 - fs / 2, minZ: (b.minZ + b.maxZ) / 2 - fs / 2, size: fs, n: 1024 };
  const cs = Math.max(OB.maxX - OB.minX, OB.maxZ - OB.minZ) + 400;
  const coarseR = { minX: (OB.minX + OB.maxX) / 2 - cs / 2, minZ: (OB.minZ + OB.maxZ) / 2 - cs / 2, size: cs, n: 512 };
  const F = bakeField(world, segs, fineR, false);
  const C = bakeField(world, segs, coarseR, true);
  const fd = new Uint8Array(F.W * F.H * 2);
  for (let i = 0; i < F.W * F.H; i++) {
    fd[i * 2] = clamp(Math.round(((F.sd[i] + 8) / 64) * 255), 0, 255);
    fd[i * 2 + 1] = clamp(Math.round(F.sd[i] + 32), 0, 255);
  }
  const cd = new Uint8Array(C.W * C.H);
  for (let i = 0; i < C.W * C.H; i++) cd[i] = clamp(Math.round((C.sd[i] + 64) / 2), 0, 255);
  const mk = (data, w, h, fmt) => {
    const t = new THREE.DataTexture(data, w, h, fmt, THREE.UnsignedByteType);
    t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
    t.magFilter = THREE.LinearFilter; t.minFilter = THREE.LinearMipmapLinearFilter; t.generateMipmaps = true;
    t.colorSpace = THREE.NoColorSpace; t.needsUpdate = true;
    return t;
  };
  const field = {
    fine: { tex: mk(fd, F.W, F.H, THREE.RGFormat), rect: new THREE.Vector4(fineR.minX, fineR.minZ, 1 / (F.W * F.cell), 1 / (F.H * F.cell)), R: fineR, F },
    coarse: { tex: mk(cd, C.W, C.H, THREE.RedFormat), rect: new THREE.Vector4(coarseR.minX, coarseR.minZ, 1 / (C.W * C.cell), 1 / (C.H * C.cell)), R: coarseR, F: C },
    segs,
    ms: Math.round(performance.now() - t0),
  };
  /** CPU lookup of the same field (metres; + over water). */
  field.distanceAt = (x, z) => {
    const sample = (L) => {
      const i = Math.floor((x - L.R.minX) / L.F.cell), k = Math.floor((z - L.R.minZ) / L.F.cell);
      if (i < 0 || k < 0 || i >= L.F.W || k >= L.F.H) return null;
      return L.F.sd[k * L.F.W + i];
    };
    const f = sample(field.fine);
    if (f !== null && Math.abs(f) < 200) return f;
    const c = sample(field.coarse);
    return c === null ? 500 : c;
  };
  return field;
}

// ====================================================================== material
const WATER_PARS_V = /* glsl */`
uniform float uTime;
varying vec3 vWPos;
varying vec3 vGN;
${WAVE_GLSL}
`;

const WATER_PARS_F = /* glsl */`
uniform float uTime;
uniform float uNight;
uniform float uFar;
uniform float uChop;
uniform sampler2D uNrm;
uniform sampler2D uNoise;
uniform sampler2D uShoreF;
uniform sampler2D uShoreC;
uniform vec4 uFineRect;
uniform vec4 uCoarseRect;
uniform vec3 uDeep;
uniform vec3 uShallow;
uniform vec3 uFoamCol;
uniform vec3 uGlowCol;
uniform vec4 uPiers[16]; // nearest columns / bridge piers standing in the water: centre xz, half size (local)
uniform vec2 uPierR[16]; // their orientation (cos, sin)
uniform float uPierN;
varying vec3 vWPos;
varying vec3 vGN;

vec3 wN;          // final world normal
vec3 wMeanN;      // smooth normal (glitter envelope)
vec3 wSparkN;     // sharper normal for the sparkle term (view space, set later)
float wSpark;
float wRough;
float wFoam;
vec3 wCol;
vec3 wGlow;
float wHorizon;

vec2 wOctave(vec2 p, float scale, float ang, float speed, float w) {
  float c = cos(ang), s = sin(ang);
  vec2 q = vec2(c * p.x - s * p.y, s * p.x + c * p.y) / scale + vec2(speed * uTime / scale, 0.0);
  vec3 n = texture2D(uNrm, q).xyz * 2.0 - 1.0;
  vec2 g = -n.xy / max(n.z, 0.25);
  return vec2(c * g.x + s * g.y, -s * g.x + c * g.y) * w;
}
float wShoreSD(vec2 p) {
#ifdef WATER_POND
  return 200.0;
#else
  vec2 cu = (p - uCoarseRect.xy) * uCoarseRect.zw;
  float dc = texture2D(uShoreC, cu).r * 510.0 - 64.0;
  vec2 fu = (p - uFineRect.xy) * uFineRect.zw;
  vec2 e = min(fu, 1.0 - fu);
  float inF = smoothstep(0.0, 0.015, min(e.x, e.y));
  if (inF <= 0.0 || dc > 260.0) return dc;
  vec2 f = texture2D(uShoreF, fu).rg;
  float dn = f.r * 64.0 - 8.0, dm = f.g * 255.0 - 32.0;
  float df = dn < 54.0 ? dn : dm;
  df = dm > 220.0 ? max(df, dc) : df;
  return mix(dc, df, inF);
#endif
}
`;

// computes wN, wCol, wRough, wFoam, wGlow, wHorizon (replaces map_fragment)
const WATER_MAIN_F = /* glsl */`
{
  vec3 toCam = cameraPosition - vWPos;
  float dist = length(toCam);
  vec3 Vw = toCam / max(dist, 1e-3);
  vec2 p = vWPos.xz;
  float hd = length(toCam.xz);
  // large-scale wind patches (cat's paws): modulate chop + colour, break any tile repeat
  vec4 pn = texture2D(uNoise, p / 1900.0 + vec2(0.0011, 0.0004) * uTime);
  float pa = pn.r, pb = pn.b; // ~475 m and ~120 m patches from one fetch
  float patchv = pa * 0.6 + pb * 0.4;
  float wind = clamp(0.35 + 1.5 * (patchv - 0.28), 0.25, 1.5);
  // wind rows / slicks: long smooth streaks aligned with the wind (very visible from the air)
  vec2 wd = vec2(0.93, -0.36);
  vec2 ws = vec2(dot(p, wd) / 1400.0, dot(p, vec2(-wd.y, wd.x)) / 170.0);
  float slick = smoothstep(0.56, 0.74, texture2D(uNoise, ws + vec2(-0.0006 * uTime, 0.0)).b * 0.75 + pb * 0.35);
  wind *= 1.0 - 0.62 * slick;
  float fHi = 1.0 - smoothstep(60.0, 700.0, hd);
  float fMid = 1.0 - smoothstep(500.0, 3500.0, hd);
  vec2 g = vec2(0.0);
  g += wOctave(p, 61.0, 0.31, 2.2, 0.50);
  vec2 gMean = g * uChop * wind;
  g += wOctave(p, 23.0, -0.45, 1.5, 0.46) * mix(0.55, 1.0, fMid);
  g += wOctave(p, 8.3, 0.92, 1.0, 0.40) * mix(0.3, 1.0, fHi);
  vec2 gFine = fHi > 0.0 ? wOctave(p, 2.9, -1.17, 0.65, 0.36) * fHi : vec2(0.0);
  g += gFine;
  g *= uChop * wind;
  vec2 gg = -vGN.xz / max(vGN.y, 0.3);
  wN = normalize(vec3(-(g.x + gg.x), 1.0, -(g.y + gg.y)));
  wMeanN = normalize(vec3(-(gMean.x + gg.x) * 0.6, 1.0, -(gMean.y + gg.y) * 0.6));
  // sparkle normal: exaggerated fine chop
  float fSp = 1.0 - smoothstep(40.0, 420.0, hd);
  vec2 g2 = g + gFine * (2.2 * fSp * uChop * wind);
  wSparkN = normalize(vec3(-(g2.x + gg.x), 1.0, -(g2.y + gg.y)));
  wSpark = fSp * (0.6 + 0.6 * wind);

  // shoreline distance (m): + over water, - under land / piers
  float sd = wShoreSD(p);
  float aa = max(fwidth(sd), 0.02);
  float nearShore = 1.0 - smoothstep(0.0, 170.0, sd);
  vec3 col = mix(uDeep, uShallow, nearShore * nearShore * 0.85 + 0.15 * nearShore);
  col *= 0.9 + 0.22 * patchv;
  // sediment / plankton plumes on the kilometre scale
  col = mix(col, col * vec3(1.16, 1.1, 0.84), smoothstep(0.5, 0.78, pa) * 0.7); // sediment / plankton plumes
  // under piers / decks: shaded, almost black
  float under = smoothstep(0.3, -5.0, sd);
  col *= 1.0 - 0.85 * under;

  // piers / bridge columns in the water (box distance), merged into the foam distance only
  float sdF = sd;
#ifndef WATER_POND
  for (int i = 0; i < 16; i++) {
    if (float(i) >= uPierN) break;
    vec4 b = uPiers[i]; vec2 r = uPierR[i];
    vec2 d = p - b.xy;
    vec2 q = abs(vec2(d.x * r.x + d.y * r.y, -d.x * r.y + d.y * r.x)) - b.zw;
    sdF = min(sdF, length(max(q, 0.0)) + min(max(q.x, q.y), 0.0));
  }
#endif
  // foam: contact line + lapping bands at walls / piers (only evaluated near them)
  float foam = 0.0;
  if (sdF < 7.0) {
    float fn = texture2D(uNoise, p / 7.0 + vec2(0.021, -0.013) * uTime).a;
    float fn2 = texture2D(uNoise, p / 2.3 + vec2(-0.05, 0.03) * uTime).b;
    float fsd = sdF + (fn - 0.5) * 0.8;
    float contact = 1.0 - smoothstep(0.0, 0.7 + aa * 1.5, fsd);
    float lap = sin(sdF * 2.1 - uTime * 1.35 + fn * 4.0) * 0.5 + 0.5;
    float bands = smoothstep(0.62, 0.95, lap) * (1.0 - smoothstep(0.4, 4.5, sdF));
    foam = contact * (0.55 + 0.45 * fn2) + bands * (0.25 + 0.6 * fn2) * fn;
    foam *= step(-0.6, sd) * (1.0 - smoothstep(1.0, 8.0, aa)); // not under decks; fades out when sub-pixel
  }
  wFoam = clamp(foam, 0.0, 1.0);
  wCol = mix(col, uFoamCol, wFoam);
  wN = normalize(mix(wN, vec3(0.0, 1.0, 0.0), wFoam * 0.7));
  wSpark *= 1.0 - wFoam;

  // roughness: glassy near, wider sun streak far (sub-pixel waves), foam is matte
  wRough = mix(0.045, 0.2, smoothstep(30.0, 2600.0, hd)) * mix(0.85, 1.25, clamp(wind - 0.5, 0.0, 1.0)) * (1.0 - 0.45 * slick);
  wRough = mix(wRough, 0.75, wFoam);

  // night: lit windows / street lights reflected where the reflected ray runs low toward the shore (probe the
  // shoreline field along the reflection); the wave normals scatter it into shimmering columns
  wGlow = vec3(0.0);
#ifndef WATER_POND
  if (uNight > 0.12) {
    vec3 rv = reflect(-Vw, wN);
    float elev = smoothstep(0.32, 0.03, rv.y);
    vec2 rh = normalize(rv.xz + 1e-5);
    float s1 = wShoreSD(p + rh * 90.0), s2 = wShoreSD(p + rh * 260.0);
    float land = smoothstep(4.0, -25.0, min(s1, s2 + 40.0));
    float fres = pow(1.0 - clamp(dot(wN, Vw), 0.0, 1.0), 3.0);
    float patchy = 0.4 + 0.6 * texture2D(uNoise, (p + rh * 200.0) / 330.0).g;
    wGlow = uGlowCol * (smoothstep(0.12, 0.5, uNight) * uNight * land * elev * (0.1 + 1.4 * fres) * patchy * (1.0 - under) * (1.0 - smoothstep(1800.0, 4500.0, hd)));
  }
#endif

  // far plane: fade into the sky seen at the horizon
#ifdef WATER_POND
  wHorizon = 0.0;
#else
  wHorizon = smoothstep(0.62 * uFar, 0.96 * uFar, hd);
#endif
}
`;

const SPARKLE_PARS = /* glsl */`
vec3 wSparkNv;
vec3 wMeanNv;
float waterSparkle(vec3 L, vec3 V) {
  vec3 H = normalize(L + V);
  // glints only inside the glitter path: where the mean (large-scale) surface can reflect the light at all
  float env = smoothstep(0.965, 0.995, dot(wMeanNv, H));
  float d = max(dot(wSparkNv, H), 0.0);
  return wSpark * env * (pow(d, 2500.0) * 60.0 + pow(d, 400.0) * 0.8) * saturate(dot(wSparkNv, L));
}
`;

/**
 * Patched MeshStandardMaterial for the bay (pond = true: still water for fountains / ponds, no swell or shore).
 * @param {{normalMap:THREE.Texture, noiseMap:THREE.Texture, shore?:object, pond?:boolean}} o
 */
export function createWaterMaterial({ normalMap, noiseMap, shore = null, pond = false }) {
  const m = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.06, metalness: 0, envMapIntensity: 1 });
  m.name = pond ? 'pond' : 'bay';
  const u = {
    uTime: { value: 0 }, uNight: { value: 0 }, uFar: { value: 6000 }, uChop: { value: pond ? 0.3 : 0.62 },
    uNrm: { value: normalMap }, uNoise: { value: noiseMap },
    uShoreF: { value: shore ? shore.fine.tex : null }, uShoreC: { value: shore ? shore.coarse.tex : null },
    uFineRect: { value: shore ? shore.fine.rect : new THREE.Vector4(0, 0, 1, 1) },
    uCoarseRect: { value: shore ? shore.coarse.rect : new THREE.Vector4(0, 0, 1, 1) },
    // estuary water: murky green-grey-blue offshore, lighter / greener (sediment) along the shore
    uDeep: { value: new THREE.Color().setRGB(0.082, 0.13, 0.122, THREE.SRGBColorSpace) },
    uShallow: { value: new THREE.Color().setRGB(0.135, 0.185, 0.14, THREE.SRGBColorSpace) },
    uFoamCol: { value: new THREE.Color().setRGB(0.78, 0.8, 0.78, THREE.SRGBColorSpace) },
    uGlowCol: { value: new THREE.Color(1.0, 0.66, 0.36).multiplyScalar(0.3) },
    uPiers: { value: Array.from({ length: 16 }, () => new THREE.Vector4(1e6, 1e6, 0, 0)) }, uPierN: { value: 0 },
    uPierR: { value: Array.from({ length: 16 }, () => new THREE.Vector2(1, 0)) },
    ...waveUniforms(),
  };
  if (pond) m.defines = { WATER_POND: '' };
  m.userData.u = u;
  m.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, u);
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', `#include <common>\n${WATER_PARS_V}`)
      .replace('#include <beginnormal_vertex>', /* glsl */`
vec3 objectNormal = vec3(0.0, 1.0, 0.0);
vec3 wDisp = vec3(0.0);
{
  vec3 wp0 = (modelMatrix * vec4(position, 1.0)).xyz;
#ifndef WATER_POND
  float hdist = length(wp0.xz - cameraPosition.xz);
  wDisp = gerstner(wp0.xz, hdist, uTime, objectNormal);
#endif
  vWPos = wp0 + wDisp;
  vGN = objectNormal;
}`)
      .replace('#include <begin_vertex>', 'vec3 transformed = vec3( position ) + wDisp;');
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', `#include <common>\n${WATER_PARS_F}`)
      .replace('#include <map_fragment>', WATER_MAIN_F)
      .replace('#include <color_fragment>', 'diffuseColor.rgb = wCol;')
      .replace('#include <roughnessmap_fragment>', 'float roughnessFactor = wRough;')
      .replace('#include <normal_fragment_maps>', 'normal = normalize( ( viewMatrix * vec4( wN, 0.0 ) ).xyz );\nwSparkNv = normalize( ( viewMatrix * vec4( wSparkN, 0.0 ) ).xyz );\nwMeanNv = normalize( ( viewMatrix * vec4( wMeanN, 0.0 ) ).xyz );')
      .replace('#include <emissivemap_fragment>', '#include <emissivemap_fragment>\ntotalEmissiveRadiance += wGlow;')
      .replace('#include <lights_physical_fragment>', THREE.ShaderChunk.lights_physical_fragment.replace('vec3( 0.04 )', 'vec3( 0.02 )'))
      .replace('#include <lights_physical_pars_fragment>', SPARKLE_PARS + THREE.ShaderChunk.lights_physical_pars_fragment.replace(
        'reflectedLight.directSpecular += irradiance * BRDF_GGX( directLight.direction, geometryViewDir, geometryNormal, material );',
        'reflectedLight.directSpecular += irradiance * BRDF_GGX( directLight.direction, geometryViewDir, geometryNormal, material );\n\treflectedLight.directSpecular += directLight.color * waterSparkle( directLight.direction, geometryViewDir );'))
      .replace('#include <lights_fragment_maps>', /* glsl */`
#if defined( RE_IndirectDiffuse ) && defined( USE_ENVMAP ) && defined( STANDARD ) && defined( ENVMAP_TYPE_CUBE_UV )
  iblIrradiance += getIBLIrradiance( geometryNormal );
#endif
#if defined( USE_ENVMAP ) && defined( RE_IndirectSpecular )
  {
    vec3 rv = inverseTransformDirection( reflect( - geometryViewDir, geometryNormal ), viewMatrix );
    rv.y = abs( rv.y ) * 0.9 + 0.02; // never reflect the sky below the horizon
    rv = normalize( mix( normalize( rv ), inverseTransformDirection( geometryNormal, viewMatrix ), material.roughness * material.roughness ) );
    radiance += textureCubeUV( envMap, envMapRotation * rv, material.roughness ).rgb * envMapIntensity;
  }
#endif`)
      .replace('#include <fog_fragment>', /* glsl */`#include <fog_fragment>
#ifndef WATER_POND
if ( wHorizon > 0.0 ) {
#ifdef USE_ENVMAP
  vec3 hz = textureCubeUV( envMap, envMapRotation * normalize( vWPos - cameraPosition ), 0.0 ).rgb;
  #ifdef TONE_MAPPING
  hz = toneMapping( hz );
  #endif
  hz = linearToOutputTexel( vec4( hz, 1.0 ) ).rgb;
#else
  vec3 hz = gl_FragColor.rgb;
  #ifdef USE_FOG
  hz = fogColor;
  #endif
#endif
  gl_FragColor.rgb = mix( gl_FragColor.rgb, hz, wHorizon );
}
#endif`);
  };
  m.customProgramCacheKey = () => (pond ? 'roadster-pond-3' : 'roadster-bay-3');
  return m;
}

// ====================================================================== geometry
/**
 * Stitched polar grid: 48 segments to 10 m, 96 to 80 m, 192 beyond (crack-free transition strips where the
 * segment count doubles), radial steps ~6.5 % / 3.3 % of the radius to 420 m, then 20 % to 30 km. No micro
 * triangles at the centre, roughly uniform screen-space density near the camera. ~19k vertices.
 */
function makeDiskGeometry() {
  const zones = [{ r1: 10, n: 48, k: 0.065, min: 0.45 }, { r1: 80, n: 96, k: 0.065, min: 0 }, { r1: 420, n: 192, k: 0.033, min: 0 }, { r1: 30000, n: 192, k: 0.2, min: 0 }];
  const rings = []; // {r, n}
  let r = 0.5, zi = 0;
  rings.push({ r: 0, n: 1 });
  while (r < 30000) {
    while (zi < zones.length - 1 && r >= zones[zi].r1) zi++;
    const z = zones[zi];
    rings.push({ r, n: z.n });
    r += Math.max(z.min, r * z.k);
  }
  rings.push({ r: 30000, n: 192 });
  const pos = [], idx = [];
  const start = [];
  for (const R of rings) {
    start.push(pos.length / 3);
    if (R.n === 1) { pos.push(0, 0, 0); continue; }
    for (let s = 0; s < R.n; s++) { const a = (s / R.n) * Math.PI * 2; pos.push(Math.cos(a) * R.r, 0, Math.sin(a) * R.r); }
  }
  for (let i = 1; i < rings.length; i++) {
    const A = rings[i - 1], B = rings[i], a0 = start[i - 1], b0 = start[i];
    if (A.n === 1) { for (let s = 0; s < B.n; s++) idx.push(a0, b0 + ((s + 1) % B.n), b0 + s); continue; }
    if (B.n === A.n) {
      for (let s = 0; s < A.n; s++) { const s1 = (s + 1) % A.n; idx.push(a0 + s, a0 + s1, b0 + s1, a0 + s, b0 + s1, b0 + s); }
    } else { // B.n = 2 A.n: transition strip
      for (let s = 0; s < A.n; s++) {
        const s1 = (s + 1) % A.n, c = b0 + 2 * s, m = b0 + 2 * s + 1, d = b0 + ((2 * s + 2) % B.n);
        idx.push(a0 + s, m, c, a0 + s, a0 + s1, m, a0 + s1, d, m);
      }
    }
  }
  const nor = new Float32Array(pos.length);
  for (let i = 1; i < nor.length; i += 3) nor[i] = 1;
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
  g.setIndex(pos.length / 3 > 65535 ? new THREE.Uint32BufferAttribute(idx, 1) : new THREE.Uint16BufferAttribute(idx, 1));
  g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 30000);
  return g;
}

/** The bay surface: camera-following polar grid with the water material. */
export class Water {
  /** @param {{material:THREE.Material, level:number}} o */
  constructor({ material, level }) {
    this.material = material;
    this.level = level;
    this.mesh = null;
  }
  build() {
    const mesh = new THREE.Mesh(makeDiskGeometry(), this.material);
    mesh.name = 'bay';
    mesh.frustumCulled = false;
    mesh.receiveShadow = true;
    mesh.castShadow = false;
    mesh.renderOrder = 1; // after the opaque city: hidden water fragments are rejected by depth
    mesh.matrixAutoUpdate = false;
    mesh.matrixWorldAutoUpdate = false;
    const u = this.material.userData.u, lvl = this.level, step = 0.5;
    mesh.onBeforeRender = (renderer, scene, cam) => {
      const e = cam.matrixWorld.elements;
      mesh.matrixWorld.makeTranslation(Math.round(e[12] / step) * step, lvl, Math.round(e[14] / step) * step);
      if (cam.far) u.uFar.value = cam.far;
    };
    this.mesh = mesh;
    return mesh;
  }
  /** @param {number} time seconds @param {number} night 0..1 */
  update(time, night) {
    const u = this.material.userData.u;
    u.uTime.value = time;
    u.uNight.value = night;
  }
}
