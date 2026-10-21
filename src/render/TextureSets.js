// PBR texture sets: a manifest-driven loader for photo-scanned material sets (diff / nor_gl / rough / ao JPEGs,
// one folder per set id, e.g. Poly Haven CC0 downloads; optional GPU-compressed .ktx2 twins written by
// engine/tools/compress-textures.mjs) with procedural canvas fallbacks for every map, so a
// missing file (offline, assets not fetched) never breaks a game. Also the generic tileable canvas painters
// (asphalt, concrete paving, grass, ground, water normals) and DataArrayTexture helpers.
//
//   manifest.json: { sets: { <role>: { id, size: [w, h] (metres), maps: { ... }, ktx2?, ktx2Maps? } } }   (optional)
//
// API
//   textureBase() / basisPath()                    default `${BASE_URL}textures/` / `${BASE_URL}basis/`
//   loadTextureManifest(base?)                     -> manifest | null (never throws)
//   loadTextureSet(id, { renderer, base, manifest, maps?, ktx2? })
//                                                  -> { map, normalMap, roughnessMap, aoMap?, size } | null
//        (KTX2 CompressedTextures when the manifest set has `ktx2: true` and the GPU has ASTC / BC7 / ETC2 / BC1,
//         else JPG; see "KTX2" below)
//   makeTextureArray(renderer, sets, { size, base, manifest, onProgress, ktx2? })
//        sets: [{ role, id|null, size (m), tint (bool), proc (fallback kind) }]
//        -> { albedo, normal, sizes, tintable, layers, loaded, wanted }
//        albedo (sRGB DataArrayTexture): rgb = albedo (tintable sets normalised to neutral grey), a = AO
//        normal (linear):                rg = tangent normal xy (OpenGL), b = roughness
//   TextureSets.stats / .options / .configure(o)   per-set path (ktx2 | jpg | mixed | procedural), bytes, decode ms,
//                                                  GPU bytes; one `[TextureSets]` summary line per load wave
//   ktx2Support(renderer)                          -> { loader (shared KTX2Loader), formats, ready } | null
//   ?ktx2=0 (or ?nokx) in the page URL forces the JPG path (A/B comparisons)
//   proceduralPbr(kind, size, seed)               -> { alb, hgt, rgh } float maps (fallback painter)
//   makeAsphaltTexture / makePavingTexture (= makeConcreteTexture) / makeGroundTexture / makeGrassTexture /
//   makeWaterNormalTexture (renderer)             -> tileable CanvasTexture
//   makeCanvas, paintCanvas, finishTexture, makeDataArrayTexture        low-level helpers
import * as THREE from 'three';
import { KTX2Loader } from 'three/addons/loaders/KTX2Loader.js';
import { mulberry32, pfbm, hash2 } from './noise.js';

// ------------------------------------------------------------------ low-level helpers
/** Canvas element of w x h. */
export function makeCanvas(w, h = w) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  return c;
}
/** 2D context that is cheap to read back. */
export function ctx2d(c) { return c.getContext('2d', { willReadFrequently: true }); }

/**
 * Standard settings for a tileable colour / data texture: repeat wrap, trilinear + anisotropic filtering,
 * colour space. `srgb` false for data maps (normals, roughness).
 */
export function finishTexture(tex, renderer, srgb = true, repeat = true, maxAnisotropy = 8) {
  if (repeat) tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
  tex.anisotropy = renderer ? Math.min(maxAnisotropy, renderer.capabilities.getMaxAnisotropy()) : 4;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.generateMipmaps = true;
  tex.needsUpdate = true;
  return tex;
}

/** Fill a canvas with fn(x, y) -> [r, g, b] (0..255); returns its 2D context for further drawing. */
export function paintCanvas(c, fn) {
  const g = ctx2d(c);
  const img = g.createImageData(c.width, c.height);
  const d = img.data;
  for (let y = 0; y < c.height; y++) for (let x = 0; x < c.width; x++) {
    const o = (y * c.width + x) * 4;
    const v = fn(x, y);
    d[o] = v[0]; d[o + 1] = v[1]; d[o + 2] = v[2]; d[o + 3] = 255;
  }
  g.putImageData(img, 0, 0);
  return g;
}

/** Release the CPU copy of a DataTexture / DataArrayTexture once it is on the GPU. */
export function freeAfterUpload(t) { t.onUpdate = () => { t.image.data = null; t.onUpdate = null; }; return t; }

/** RGBA8 DataArrayTexture (size x size x layers), repeat-wrapped, mipmapped, max anisotropy. */
export function makeDataArrayTexture(data, size, layers, srgb, renderer, { free = true } = {}) {
  const t = new THREE.DataArrayTexture(data, size, size, layers);
  t.format = THREE.RGBAFormat; t.type = THREE.UnsignedByteType;
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.minFilter = THREE.LinearMipmapLinearFilter; t.magFilter = THREE.LinearFilter;
  t.generateMipmaps = true;
  t.anisotropy = renderer ? renderer.capabilities.getMaxAnisotropy() : 4;
  t.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
  t.needsUpdate = true;
  return free ? freeAfterUpload(t) : t;
}

// ------------------------------------------------------------------ procedural tileable painters
/** Asphalt with aggregate, patch repairs and cracks (1024 px; ~8 m world tile). */
export function makeAsphaltTexture(renderer) {
  const S = 1024, c = makeCanvas(S);
  const rnd = mulberry32(11);
  const g = paintCanvas(c, (x, y) => {
    const u = x / S, v = y / S;
    const big = pfbm(u, v, 4, 4, 1);
    const fine = hash2(x * 3 + 1, y * 7 + 3);
    const grain = pfbm(u, v, 64, 2, 5);
    let l = 46 + big * 22 + (fine - 0.5) * 26 + (grain - 0.5) * 14;
    if (fine > 0.985) l += 40; // light aggregate stones
    return [l, l * 1.0, l * 1.02];
  });
  // patch repairs and cracks
  g.globalAlpha = 0.25;
  for (let i = 0; i < 7; i++) {
    g.fillStyle = rnd() < 0.5 ? '#1c1c1d' : '#5a5a5a';
    const w = 80 + rnd() * 220, h = 60 + rnd() * 160, x = rnd() * S, y = rnd() * S;
    g.fillRect(x, y, w, h);
    if (x + w > S) g.fillRect(x - S, y, w, h);
    if (y + h > S) g.fillRect(x, y - S, w, h);
  }
  g.globalAlpha = 0.55; g.strokeStyle = '#141414'; g.lineWidth = 1.2;
  for (let i = 0; i < 22; i++) {
    let x = rnd() * S, y = rnd() * S; g.beginPath(); g.moveTo(x, y);
    for (let k = 0; k < 14; k++) { x += (rnd() - 0.5) * 34; y += (rnd() - 0.5) * 34; g.lineTo(x, y); }
    g.stroke();
  }
  g.globalAlpha = 1;
  return finishTexture(new THREE.CanvasTexture(c), renderer);
}

/** Concrete paving slabs, 4 x 4 per tile with joints (512 px; ~2 m tile). */
export function makePavingTexture(renderer) {
  const S = 512, c = makeCanvas(S);
  const g = paintCanvas(c, (x, y) => {
    const u = x / S, v = y / S;
    const n = pfbm(u, v, 8, 4, 21);
    const f = hash2(x + 9, y * 3);
    const slab = hash2(Math.floor(x / 128), Math.floor(y / 128) + 77);
    let l = 168 + n * 30 + (f - 0.5) * 14 + (slab - 0.5) * 16;
    const jx = x % 128, jy = y % 128;
    if (jx < 2 || jy < 2) l -= 52;
    else if (jx < 4 || jy < 4) l -= 12;
    return [l, l * 0.985, l * 0.95];
  });
  g.globalAlpha = 1;
  return finishTexture(new THREE.CanvasTexture(c), renderer);
}
export const makeConcreteTexture = makePavingTexture;

/** Generic weathered ground with faint large slabs (1024 px; ~6 m tile). */
export function makeGroundTexture(renderer) {
  const S = 1024, c = makeCanvas(S);
  paintCanvas(c, (x, y) => {
    const u = x / S, v = y / S;
    const n = pfbm(u, v, 3, 5, 31);
    const f = hash2(x * 5, y * 11 + 1);
    const slabs = (x % 256 < 2 || y % 256 < 2) ? -30 : 0;
    const l = 118 + n * 40 + (f - 0.5) * 18 + slabs;
    return [l * 1.0, l * 0.98, l * 0.93];
  });
  return finishTexture(new THREE.CanvasTexture(c), renderer);
}

/** Grass (512 px; ~4 m tile). */
export function makeGrassTexture(renderer) {
  const S = 512, c = makeCanvas(S);
  paintCanvas(c, (x, y) => {
    const u = x / S, v = y / S;
    const n = pfbm(u, v, 4, 5, 41);
    const m = pfbm(u, v, 16, 3, 43);
    const f = hash2(x * 13, y * 17);
    const l = 0.75 + n * 0.45 + (f - 0.5) * 0.35;
    return [52 * l + m * 18, 92 * l + m * 20, 34 * l];
  });
  return finishTexture(new THREE.CanvasTexture(c), renderer);
}

/** Tileable small-wave water normal map (512 px, linear). */
export function makeWaterNormalTexture(renderer) {
  const S = 512;
  const hgt = new Float32Array(S * S);
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    const u = x / S, v = y / S;
    hgt[y * S + x] = pfbm(u, v, 6, 5, 51, 0.55) + 0.35 * pfbm(u, v, 24, 3, 53);
  }
  const c = makeCanvas(S);
  paintCanvas(c, (x, y) => {
    const xm = (x + S - 1) % S, xp = (x + 1) % S, ym = (y + S - 1) % S, yp = (y + 1) % S;
    const dx = (hgt[y * S + xp] - hgt[y * S + xm]) * 6;
    const dy = (hgt[yp * S + x] - hgt[ym * S + x]) * 6;
    const l = Math.hypot(dx, dy, 1);
    return [(-dx / l * 0.5 + 0.5) * 255, (-dy / l * 0.5 + 0.5) * 255, (1 / l * 0.5 + 0.5) * 255];
  });
  return finishTexture(new THREE.CanvasTexture(c), renderer, false);
}

// ------------------------------------------------------------------ procedural PBR fallback
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

// ------------------------------------------------------------------ manifest + loading
/** Default texture root: `${import.meta.env.BASE_URL}textures/` under Vite, else `/textures/`. */
export function textureBase() {
  const b = (typeof import.meta !== 'undefined' && import.meta.env && import.meta.env.BASE_URL) || '/';
  return `${b}textures/`;
}
/** Default Basis transcoder path (basis_transcoder.js + .wasm, copied from three by compress-textures.mjs). */
export function basisPath() {
  if (TextureSets.options.basisPath) return TextureSets.options.basisPath;
  const b = (typeof import.meta !== 'undefined' && import.meta.env && import.meta.env.BASE_URL) || '/';
  return `${b}basis/`;
}

/** Fetch `${base}manifest.json`; null when missing or offline (never throws). */
export async function loadTextureManifest(base = textureBase()) {
  try { const r = await fetch(`${base}manifest.json`); if (r.ok) return await r.json(); } catch { /* offline */ }
  return null;
}

async function loadBitmap(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`HTTP ${r.status} ${url}`);
  const blob = await r.blob();
  const t = performance.now();
  const bmp = await createImageBitmap(blob, { colorSpaceConversion: 'none', premultiplyAlpha: 'none', imageOrientation: 'none' });
  return { bmp, bytes: blob.size, ms: performance.now() - t };
}
function pixelsOf(bmp, S, scratch) {
  scratch.g.clearRect(0, 0, S, S);
  scratch.g.drawImage(bmp, 0, 0, S, S);
  return scratch.g.getImageData(0, 0, S, S).data;
}

// map slot -> [jpg file, sRGB, manifest key]
const MAP_FILES = { map: ['diff.jpg', true, 'Diffuse'], normalMap: ['nor_gl.jpg', false, 'nor_gl'], roughnessMap: ['rough.jpg', false, 'Rough'], aoMap: ['ao.jpg', false, 'AO'] };
const ARRAY_MAPS = ['map', 'normalMap', 'roughnessMap', 'aoMap'];

// ------------------------------------------------------------------ KTX2 (Basis Universal) path + stats
// engine/tools/compress-textures.mjs writes <id>/{diff,nor_gl,rough,ao}.ktx2 (Y-flipped, so the data matches the JPGs
// uploaded with flipY) and lists them per set in the manifest: `ktx2Maps: { Diffuse|nor_gl|Rough|AO: { file, bytes,
// srcBytes, codec, ... } }` (+ `ktx2: true` when every map has one; the loader goes by the per-map entries).
// loadTextureSet turns them into CompressedTextures through one shared KTX2Loader (transcoded off-thread to ASTC / BC7
// / ETC2 / BC1 for this GPU). makeTextureArray packs several maps per texel (albedo + AO, normal xy + roughness) and
// normalises tintable albedos, which compressed blocks cannot do, so it transcodes to RGBA8 in its own small worker
// pool and packs exactly like the JPG path: smaller downloads, same RGBA8 arrays (see ENGINE.md for the trade-off).
// Every failure (no renderer, no compressed format, transcoder missing or not WASM, file missing / corrupt, transcode
// timeout) falls back to the JPG, then to the procedural painter. `?ktx2=0` (or `?nokx`) forces the JPG path,
// `TextureSets.options` changes the defaults.

/** Loader options + per-set statistics (which path each set used, bytes downloaded, decode time, GPU bytes). */
export const TextureSets = {
  options: {
    ktx2: true,            // use .ktx2 when the manifest lists it (URL ?ktx2=0 / ?nokx overrides)
    arrayKtx2: 'auto',     // texture arrays: 'auto' = per map, whichever of .ktx2 / .jpg is smaller to download
                           // (both end up RGBA8 in the array), true = always .ktx2, false = always .jpg
    basisPath: null,       // transcoder folder (default `${BASE_URL}basis/`)
    workers: 3,            // RGBA transcode workers for the array path (KTX2Loader keeps its own pool of 4)
    log: true,             // one summary console line per load wave
  },
  stats: null,
  /** Merge loader options (see above). */
  configure(o = {}) { Object.assign(this.options, o); return this; },
  /** Clear the statistics. */
  resetStats() { this.stats = freshStats(); return this.stats; },
};
function freshStats() {
  return {
    ktx2: null,            // { formats: [...] } once a renderer was probed, false when KTX2 is unavailable
    // `${consumer}:${id}` (consumer 'set' = loadTextureSet, 'array' = a makeTextureArray layer) ->
    //   { id, consumer, path: 'ktx2' | 'jpg' | 'mixed' | 'procedural' | 'missing', maps: { map: 'ktx2' | 'jpg' | 'missing', ... },
    //     bytes (downloaded; per map in mapBytes), ms (wall time from the call to ready), decodeMs (per-map decode / transcode time after the
    //     download, summed; includes worker queueing), packMs (array layers: main-thread readback + packing),
    //     gpuBytes (textures this set owns; the arrays' RGBA8 storage is added to totals.gpuBytes), formats }
    sets: {},
    totals: { sets: 0, ktx2: 0, jpg: 0, mixed: 0, procedural: 0, missing: 0, bytes: 0, ktx2Bytes: 0, jpgBytes: 0, gpuBytes: 0, decodeMs: 0, packMs: 0, wallMs: 0 },
  };
}
TextureSets.resetStats();

const FORMAT_NAMES = {
  [THREE.RGBA_ASTC_4x4_Format]: 'ASTC 4x4', [THREE.RGBA_BPTC_Format]: 'BC7', [THREE.RGB_ETC2_Format]: 'ETC2',
  [THREE.RGBA_ETC2_EAC_Format]: 'ETC2 EAC', [THREE.RGB_ETC1_Format]: 'ETC1', [THREE.RGBA_S3TC_DXT1_Format]: 'BC1',
  [THREE.RGBA_S3TC_DXT5_Format]: 'BC3', [THREE.RGB_PVRTC_4BPPV1_Format]: 'PVRTC', [THREE.RGBA_PVRTC_4BPPV1_Format]: 'PVRTC',
  [THREE.RGBAFormat]: 'RGBA8',
};
const MIP = 4 / 3; // full mip chain overhead

/** URL switch: ?ktx2=0 / ?nokx / ?noktx2 -> false, ?ktx2=1 -> true, else null. */
function urlKtx2() {
  try {
    const q = new URLSearchParams(globalThis.location ? globalThis.location.search : '');
    if (q.has('nokx') || q.has('noktx2') || q.get('ktx2') === '0') return false;
    if (q.get('ktx2') === '1') return true;
  } catch { /* no location */ }
  return null;
}
function ktx2Wanted(opt) {
  const u = urlKtx2();
  if (u !== null) return u;
  return opt !== undefined ? !!opt : !!TextureSets.options.ktx2;
}

// --- stats bookkeeping (a "wave" = loads that overlap in time; one summary line per wave)
let _pending = 0, _waveStart = 0, _logTimer = 0;
function beginStat(consumer, id) {
  if (_pending++ === 0 && !_logTimer) _waveStart = performance.now();
  if (_logTimer) { clearTimeout(_logTimer); _logTimer = 0; }
  return { id, consumer, path: 'missing', maps: {}, bytes: 0, ms: 0, decodeMs: 0, packMs: 0, gpuBytes: 0, formats: [], t0: performance.now() };
}
function noteMap(st, slot, path, bytes = 0, decodeMs = 0, format = null) {
  st.maps[slot] = path; st.bytes += bytes; st.decodeMs += decodeMs;
  if (format && !st.formats.includes(format)) st.formats.push(format);
}
function endStat(st, path) {
  const vals = Object.values(st.maps).filter((v) => v !== 'missing');
  st.path = path || (vals.length === 0 ? 'missing' : vals.every((v) => v === vals[0]) ? vals[0] : 'mixed');
  st.ms = Math.round(performance.now() - st.t0); st.decodeMs = Math.round(st.decodeMs); st.packMs = Math.round(st.packMs);
  delete st.t0;
  const S = TextureSets.stats, T = S.totals, key = `${st.consumer}:${st.id}`;
  if (!S.sets[key]) T.sets++;
  S.sets[key] = st;
  T[st.path] = (T[st.path] || 0) + 1;
  T.bytes += st.bytes; T.gpuBytes += st.gpuBytes; T.decodeMs += st.decodeMs; T.packMs += st.packMs;
  for (const [slot, p] of Object.entries(st.maps)) if (p === 'ktx2' || p === 'jpg') T[p + 'Bytes'] += st.mapBytes ? st.mapBytes[slot] || 0 : 0;
  if (--_pending === 0) _logTimer = setTimeout(logSummary, 300);
}
function logSummary() {
  _logTimer = 0;
  const S = TextureSets.stats, T = S.totals;
  T.wallMs = Math.round(performance.now() - 300 - _waveStart);
  if (!TextureSets.options.log) return;
  const fm = new Set(); for (const s of Object.values(S.sets)) for (const f of s.formats) fm.add(f);
  const by = ['ktx2', 'jpg', 'mixed', 'procedural', 'missing'].filter((k) => T[k]).map((k) => `${k} ${T[k]}`).join(' / ');
  console.info(`[TextureSets] ${T.sets} sets: ${by} · ${(T.bytes / 1048576).toFixed(1)} MB downloaded (ktx2 ${(T.ktx2Bytes / 1048576).toFixed(1)}, jpg ${(T.jpgBytes / 1048576).toFixed(1)}) · ~${(T.gpuBytes / 1048576).toFixed(0)} MB GPU · ready in ${T.wallMs} ms (main-thread packing ${T.packMs} ms) · ${[...fm].join(', ') || 'no textures'}${S.ktx2 === false ? ' (KTX2 off)' : ''}`);
}
function trackBytes(st, slot, bytes) { (st.mapBytes || (st.mapBytes = {}))[slot] = bytes; }

// --- shared KTX2Loader (compressed textures for loadTextureSet)
let _kx;
/**
 * The shared KTX2Loader for `renderer` (created and probed on first call, synchronously, so callers can start their
 * downloads right away), or null when KTX2 cannot help: no renderer, or no compressed texture format (the loader would
 * transcode to RGBA8, and the JPG is as good). `ready` resolves to false when the transcoder cannot be fetched, after
 * which loads use the JPGs. Never throws.
 * @returns {{loader: KTX2Loader, formats: string[], ready: Promise<boolean>, failed: boolean}|null}
 */
export function ktx2Support(renderer) {
  if (!renderer) return null;
  if (_kx !== undefined) return _kx;
  try {
    const loader = new KTX2Loader().setTranscoderPath(basisPath());
    loader.detectSupport(renderer);
    const c = loader.workerConfig;
    const formats = [['astcSupported', 'ASTC'], ['bptcSupported', 'BC7'], ['etc2Supported', 'ETC2'], ['etc1Supported', 'ETC1'], ['dxtSupported', 'BC1/BC3'], ['pvrtcSupported', 'PVRTC']].filter(([k]) => c[k]).map(([, n]) => n);
    if (!formats.length) { TextureSets.stats.ktx2 = false; return (_kx = null); }
    const kx = { loader, formats, failed: false, ready: null };
    // fetch the transcoder now (in parallel with the texture downloads); a missing /basis/ means JPG, quietly
    // (and check the WASM magic: a dev server's SPA fallback answers missing files with index.html, which would
    // leave KTX2Loader's workers dead and every parse() pending)
    kx.ready = loader.init().then(() => {
      const b = new Uint8Array(loader.transcoderBinary || new ArrayBuffer(0), 0, 4);
      if (b.length < 4 || b[0] !== 0 || b[1] !== 0x61 || b[2] !== 0x73 || b[3] !== 0x6d) throw new Error('basis_transcoder.wasm is not a WASM file');
      return true;
    }).catch((e) => {
      kx.failed = true; TextureSets.stats.ktx2 = false;
      console.warn(`[TextureSets] KTX2 transcoder unavailable at ${basisPath()} (${e && e.message ? e.message : e}); using JPG`);
      return false;
    });
    TextureSets.stats.ktx2 = { formats };
    return (_kx = kx);
  } catch (e) {
    console.warn(`[TextureSets] KTX2 unavailable (${e && e.message ? e.message : e}); using JPG`);
    TextureSets.stats.ktx2 = false;
    return (_kx = null);
  }
}
let _warned = 0;
function warnFallback(url, e) { if (_warned++ < 4) console.warn(`[TextureSets] ${url}: ${e && e.message ? e.message : e}; falling back to JPG`); }

const withTimeout = (p, ms, msg) => new Promise((res, rej) => { const t = setTimeout(() => rej(new Error(msg)), ms); p.then((v) => { clearTimeout(t); res(v); }, (e) => { clearTimeout(t); rej(e); }); });
const fetchBuffer = (url) => fetch(url).then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.arrayBuffer(); });
async function loadKtx2Texture(kx, bufP, url, st, slot) {
  try {
    const buf = await bufP;
    if (!(await kx.ready)) return null;
    const bytes = buf.byteLength, t = performance.now();
    const tex = await withTimeout(new Promise((res, rej) => kx.loader.parse(buf, res, rej)), 20000, 'transcode timed out');
    const fmt = FORMAT_NAMES[tex.format] || String(tex.format);
    let gpu = 0; for (const m of tex.mipmaps || []) gpu += m.data ? m.data.byteLength : 0;
    st.gpuBytes += gpu; trackBytes(st, slot, bytes);
    noteMap(st, slot, 'ktx2', bytes, performance.now() - t, fmt);
    return tex;
  } catch (e) { if (!kx.failed) warnFallback(url, e); return null; }
}
async function loadJpgTexture(url, srgb, st, slot) {
  try {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const blob = await r.blob(), t = performance.now();
    const obj = URL.createObjectURL(blob);
    const tex = await new THREE.TextureLoader().loadAsync(obj).finally(() => URL.revokeObjectURL(obj));
    const w = tex.image.width || 0, h = tex.image.height || 0;
    st.gpuBytes += w * h * 4 * MIP; trackBytes(st, slot, blob.size);
    noteMap(st, slot, 'jpg', blob.size, performance.now() - t, 'RGBA8');
    return tex;
  } catch { noteMap(st, slot, 'missing'); return null; }
}

// --- RGBA8 transcode worker pool (texture-array path): basis_transcoder.js + a tiny message loop in a blob worker
// that loads the transcoder itself (importScripts + fetch), so the pool starts without waiting for the main thread
function rgbaWorkerMain() {
  /* global BASIS, importScripts */
  let M = null;
  const queue = [];
  let ready = false, failed = null;
  const run = (m) => {
    if (failed) { self.postMessage({ id: m.id, error: failed }); return; }
    let k = null;
    try {
      k = new M.KTX2File(new Uint8Array(m.buffer));
      if (!k.isValid() || !k.startTranscoding()) throw new Error('invalid or unsupported .ktx2');
      // the level that is exactly size x size, else the smallest level covering it, else the base level
      const n = k.getLevels(); let lv = 0;
      for (let i = 0; i < n; i++) {
        const inf = k.getImageLevelInfo(i, 0, 0);
        if (inf.origWidth === m.size && inf.origHeight === m.size) { lv = i; break; }
        if (inf.origWidth >= m.size && inf.origHeight >= m.size) lv = i;
      }
      const inf = k.getImageLevelInfo(lv, 0, 0);
      const t = performance.now();
      const dst = new Uint8Array(k.getImageTranscodedSizeInBytes(lv, 0, 0, 13));
      if (!k.transcodeImage(dst, lv, 0, 0, 13 /* RGBA32 */, 0, -1, -1)) throw new Error('transcode failed');
      self.postMessage({ id: m.id, data: dst, width: inf.origWidth, height: inf.origHeight, ms: performance.now() - t }, [dst.buffer]);
    } catch (err) {
      self.postMessage({ id: m.id, error: String(err && err.message ? err.message : err) });
    } finally { if (k) { k.close(); k.delete(); } }
  };
  self.onmessage = (e) => { if (ready || failed) run(e.data); else queue.push(e.data); };
  const fail = (e) => { failed = String(e && e.message ? e.message : e); self.postMessage({ type: 'failed', error: failed }); for (const m of queue.splice(0)) run(m); };
  // fetch + validate both files first: a dev server's SPA fallback answers 200 with index.html for missing files
  const get = (f, type) => fetch(self.BASIS_PATH + f).then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status} ${f}`); return r[type](); });
  Promise.all([get('basis_transcoder.js', 'text'), get('basis_transcoder.wasm', 'arrayBuffer')])
    .then(([js, wasmBinary]) => {
      const b = new Uint8Array(wasmBinary, 0, Math.min(4, wasmBinary.byteLength));
      if (/^\s*</.test(js) || b[0] !== 0 || b[1] !== 0x61 || b[2] !== 0x73 || b[3] !== 0x6d) throw new Error('transcoder files are not JS / WASM');
      importScripts(URL.createObjectURL(new Blob([js], { type: 'text/javascript' })));
      return new Promise((res, rej) => { M = { wasmBinary, onRuntimeInitialized: res, onAbort: (w) => rej(new Error(`transcoder aborted: ${w}`)) }; BASIS(M); });
    })
    .then(() => { M.initializeBasis(); ready = true; for (const m of queue.splice(0)) run(m); })
    .catch(fail);
}
let _rgba = null, _rgbaFailed = false, _rgbaWarned = false;
function warnRgba(msg) { if (!_rgbaWarned) console.warn(`[TextureSets] KTX2 transcoder unavailable for texture arrays (${msg}); using JPG`); _rgbaWarned = true; }
function rgbaPool() {
  if (_rgba) return _rgba;
  const abs = new URL(basisPath(), globalThis.location ? globalThis.location.href : undefined).href;
  const fn = rgbaWorkerMain.toString();
  const url = URL.createObjectURL(new Blob([`self.BASIS_PATH = ${JSON.stringify(abs)};\n`, fn.substring(fn.indexOf('{') + 1, fn.lastIndexOf('}'))], { type: 'text/javascript' }));
  const n = Math.max(1, TextureSets.options.workers | 0), workers = [], jobs = new Map(), queue = [], busy = new Array(n).fill(0);
  let seq = 0, idle = 0;
  // one job in flight per worker, lowest `prio` first: the array packs layers in order, so layer 0's maps go first
  const pump = () => {
    for (let i = 0; i < n && queue.length; i++) {
      if (busy[i]) continue;
      let b = 0; for (let k = 1; k < queue.length; k++) if (queue[k].prio < queue[b].prio) b = k;
      const j = queue.splice(b, 1)[0];
      busy[i] = j.id; workers[i].postMessage({ id: j.id, buffer: j.buffer, size: j.size }, [j.buffer]);
    }
  };
  // workers (and their WASM heaps) go away 10 s after the last job; the next load starts a fresh pool
  const retire = () => { idle = 0; if (jobs.size) return; for (const w of workers) w.terminate(); URL.revokeObjectURL(url); if (_rgba === pool) _rgba = null; };
  const failAll = (msg) => {
    if (!_rgbaFailed && jobs.size) warnRgba(msg); // silent when nothing was waiting (speculative boot)
    _rgbaFailed = msg || 'transcoder unavailable';
    queue.length = 0;
    for (const [id, j] of jobs) { jobs.delete(id); j.rej(new Error(msg)); }
  };
  for (let i = 0; i < n; i++) {
    const w = new Worker(url);
    w.onmessage = (e) => {
      const d = e.data;
      if (d.type === 'failed') return failAll(d.error);
      if (busy[i] === d.id) busy[i] = 0;
      const j = jobs.get(d.id);
      if (j) { jobs.delete(d.id); d.error ? j.rej(new Error(d.error)) : j.res(d); }
      pump();
      if (!jobs.size && !idle) idle = setTimeout(retire, 10000);
    };
    w.onerror = (e) => { e.preventDefault && e.preventDefault(); failAll(e.message || 'transcode worker failed'); };
    workers.push(w);
  }
  const pool = {
    transcode(buffer, size, prio = 0) {
      if (_rgbaFailed) return Promise.reject(new Error('transcoder unavailable'));
      if (idle) { clearTimeout(idle); idle = 0; }
      return new Promise((res, rej) => { const id = ++seq; jobs.set(id, { res, rej }); queue.push({ id, buffer, size, prio }); pump(); });
    },
  };
  return (_rgba = pool);
}

/** Bilinear RGBA8 resample (half-pixel centres, clamped edges, like canvas drawImage) to S x S. */
function resampleRGBA(src, sw, sh, S) {
  const out = new Uint8Array(S * S * 4), fx = sw / S, fy = sh / S;
  for (let y = 0; y < S; y++) {
    const v = Math.min(sh - 1, Math.max(0, (y + 0.5) * fy - 0.5)), y0 = Math.floor(v), y1 = Math.min(sh - 1, y0 + 1), ty = v - y0;
    for (let x = 0; x < S; x++) {
      const u = Math.min(sw - 1, Math.max(0, (x + 0.5) * fx - 0.5)), x0 = Math.floor(u), x1 = Math.min(sw - 1, x0 + 1), tx = u - x0;
      const a = (y0 * sw + x0) * 4, b = (y0 * sw + x1) * 4, c = (y1 * sw + x0) * 4, d = (y1 * sw + x1) * 4, o = (y * S + x) * 4;
      for (let k = 0; k < 4; k++) {
        const top = src[a + k] + (src[b + k] - src[a + k]) * tx, bot = src[c + k] + (src[d + k] - src[c + k]) * tx;
        out[o + k] = Math.round(top + (bot - top) * ty);
      }
    }
  }
  return out;
}
/** Transcode a fetched .ktx2 (its S x S level) to RGBA8. Rows come bottom-up (compress-textures writes "ru"). */
async function loadKtx2Rgba(bufP, S, prio = 0) {
  const buf = await bufP, bytes = buf.byteLength;
  const res = await withTimeout(rgbaPool().transcode(buf, S, prio), 20000, 'transcode timed out');
  let data = res.data;
  if (res.width !== S || res.height !== S) data = resampleRGBA(data, res.width, res.height, S);
  return { data, up: true, bytes, ms: res.ms };
}

/**
 * Load one set as plain THREE textures: `.ktx2` CompressedTextures when the manifest lists them and the renderer has a
 * compressed format (see KTX2 notes above), otherwise browser-decoded JPGs (TextureLoader, flipY). Repeat-wrapped, max
 * anisotropy, sRGB albedo / linear data. `maps`: which of map / normalMap / roughnessMap / aoMap to fetch. With a
 * manifest, a set id it does not list resolves to null without fetching. Returns null when the albedo is missing.
 * `ktx2: false` forces the JPGs for this call. Never throws.
 */
export async function loadTextureSet(id, { renderer = null, base = textureBase(), manifest = null, maps = ['map', 'normalMap', 'roughnessMap'], defaultSize = 2, ktx2 } = {}) {
  const m = manifest && manifest.sets && Object.values(manifest.sets).find((e) => e.id === id);
  if (manifest && !m) return null;
  const st = beginStat('set', id);
  const kx = m && m.ktx2Maps && ktx2Wanted(ktx2) ? ktx2Support(renderer) : null;
  const finish = (t, srgb) => {
    t.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    t.anisotropy = renderer ? renderer.capabilities.getMaxAnisotropy() : 4;
    t.needsUpdate = true;
    return t;
  };
  const one = async (k) => {
    const [jpg, srgb, key] = MAP_FILES[k];
    const info = kx && !kx.failed && m.ktx2Maps[key];
    const url = info ? `${base}${id}/${info.file}` : null;
    const bufP = url ? fetchBuffer(url) : null; // download starts now, transcoder or not
    if (bufP) bufP.catch(() => {});
    let t = bufP ? await loadKtx2Texture(kx, bufP, url, st, k) : null;
    if (!t) t = await loadJpgTexture(`${base}${id}/${jpg}`, srgb, st, k);
    return t ? finish(t, srgb) : null;
  };
  const got = await Promise.all(maps.map(one));
  const out = {};
  maps.forEach((k, i) => { out[k] = got[i]; });
  endStat(st);
  if (!out.map) return null;
  out.size = m && m.size ? m.size[0] : defaultSize;
  return out;
}

/**
 * Pack material sets into two DataArrayTextures (see header). Photo sets are fetched + decoded in parallel
 * (JPG decode is off-thread via createImageBitmap, KTX2 transcode in a worker pool) and packed sequentially; missing
 * sets fall back to proceduralPbr at 256 px tiled up to `size`. Tintable sets are normalised to a neutral mean so a
 * per-instance tint reaches real paint albedos. With KTX2 each map comes from whichever of `.ktx2` / `.jpg`
 * TextureSets.options.arrayKtx2 picks (default: the smaller download); both are packed into the same RGBA8 layers,
 * so the shaders that sample the arrays see the same layout either way. Never throws for missing files.
 */
export async function makeTextureArray(renderer, sets, { size = 1024, base = textureBase(), manifest, onProgress, ktx2 } = {}) {
  if (manifest === undefined) {
    // boot the transcode workers while the manifest loads (they retire after 10 s idle if it lists no .ktx2)
    if (ktx2Wanted(ktx2) && TextureSets.options.arrayKtx2 !== false && !_rgbaFailed && typeof Worker !== 'undefined') rgbaPool();
    manifest = await loadTextureManifest(base);
  }
  sets = sets.map((s) => {
    const m = manifest && manifest.sets && manifest.sets[s.role];
    return { ...s, size: m && m.size ? m.size[0] : s.size, maps: m ? m.maps || {} : null, kx: m && m.ktx2Maps ? m.ktx2Maps : null };
  });
  const S = size, L = sets.length;
  const A = new Uint8Array(S * S * 4 * L), N = new Uint8Array(S * S * 4 * L);
  const c = makeCanvas(S), scratch = { g: ctx2d(c) };
  let loaded = 0, done = 0;
  const want = sets.filter((s) => s.id).length;
  const useKx = ktx2Wanted(ktx2), mode = TextureSets.options.arrayKtx2;
  const pickKtx2 = (info) => !!info && (mode === true || (mode === 'auto' && info.bytes <= (info.srcBytes || Infinity)));
  // start the transcode workers now: they fetch the transcoder themselves while the maps download
  if (useKx && !_rgbaFailed && sets.some((s) => s.id && s.kx && ARRAY_MAPS.some((k) => pickKtx2(s.kx[MAP_FILES[k][2]])))) rgbaPool();
  const stats = [];
  const fetchSet = (s, l) => {
    if (!s.id || (manifest && !s.maps)) return Promise.resolve(null);
    const st = stats[l] = beginStat('array', s.id);
    const get = async (slot) => {
      const [jpg, , key] = MAP_FILES[slot];
      const info = useKx && s.kx ? s.kx[key] : null;
      if (pickKtx2(info)) {
        if (_rgbaFailed) warnRgba(_rgbaFailed);
        else {
          const url = `${base}${s.id}/${info.file}`;
          try { const r = await loadKtx2Rgba(fetchBuffer(url), S, l); trackBytes(st, slot, r.bytes); noteMap(st, slot, 'ktx2', r.bytes, r.ms, 'RGBA8 array'); return r; } catch (e) { if (_rgbaFailed) warnRgba(_rgbaFailed); else warnFallback(url, e); }
        }
      }
      try { const r = await loadBitmap(`${base}${s.id}/${jpg}`); trackBytes(st, slot, r.bytes); noteMap(st, slot, 'jpg', r.bytes, r.ms, 'RGBA8 array'); return r; } catch { noteMap(st, slot, 'missing'); return null; }
    };
    return Promise.all(ARRAY_MAPS.map(get)).then((r) => { onProgress && onProgress(++done / Math.max(1, want)); return r; });
  };
  const jobs = sets.map(fetchSet);
  // a decoded map as { d: RGBA8 S x S, up: rows bottom-up (KTX2) or top-down (canvas readback of a JPG) }
  const pix = (r) => (!r ? null : r.data ? { d: r.data, up: true } : { d: pixelsOf(r.bmp, S, scratch), up: false });
  for (let l = 0; l < L; l++) {
    const s = sets[l];
    let imgs = null;
    try { imgs = await jobs[l]; } catch { imgs = null; }
    const o = l * S * S * 4, row = S * 4;
    if (imgs && imgs[0] && imgs[1]) {
      const t = performance.now();
      const dif = pix(imgs[0]);
      // albedo (+ optional neutralisation for tintable sets)
      let mr = 1, mg = 1, mb = 1;
      if (s.tint) {
        let sr = 0, sg = 0, sb = 0; const step = 7, dd = dif.d;
        for (let i = 0; i < S * S; i += step) { sr += dd[i * 4]; sg += dd[i * 4 + 1]; sb += dd[i * 4 + 2]; }
        const cnt = Math.ceil((S * S) / step);
        const target = 200; // neutral mean ~0.58 linear -> lets tints (<=1) reach real paint albedos
        mr = target / Math.max(8, sr / cnt); mg = target / Math.max(8, sg / cnt); mb = target / Math.max(8, sb / cnt);
      }
      const ao = pix(imgs[3]);
      // texture row y (0 = bottom, v = 0) <- source row: bottom-up sources copy straight, canvas rows flip
      const srow = (p, y) => (p.up ? y : S - 1 - y) * row;
      for (let y = 0; y < S; y++) {
        const dst = o + y * row, sd = srow(dif, y), sa = ao ? srow(ao, y) : 0, D = dif.d, Ad = ao ? ao.d : null;
        for (let x = 0; x < row; x += 4) {
          A[dst + x] = Math.min(255, D[sd + x] * mr); A[dst + x + 1] = Math.min(255, D[sd + x + 1] * mg); A[dst + x + 2] = Math.min(255, D[sd + x + 2] * mb);
          A[dst + x + 3] = Ad ? Ad[sa + x] : 255;
        }
      }
      const nor = pix(imgs[1]);
      const rough = pix(imgs[2]);
      for (let y = 0; y < S; y++) {
        const dst = o + y * row, sn = srow(nor, y), sr = rough ? srow(rough, y) : 0, Nd = nor.d, Rd = rough ? rough.d : null;
        for (let x = 0; x < row; x += 4) {
          N[dst + x] = Nd[sn + x]; N[dst + x + 1] = Nd[sn + x + 1]; N[dst + x + 2] = Rd ? Rd[sr + x] : 200; N[dst + x + 3] = 255;
        }
      }
      for (const b of imgs) if (b && b.bmp && b.bmp.close) b.bmp.close();
      if (stats[l]) stats[l].packMs += performance.now() - t; // canvas readback + packing
      loaded++;
    } else {
      // procedural fallback at 256 px, nearest-tiled up to S
      const P = 256, pr = proceduralPbr(s.proc, P, 97 + l * 13);
      if (s.tint) { // same neutral mean as the normalised photo sets
        const m = [0, 0, 0];
        for (let i = 0; i < P * P; i++) { m[0] += pr.alb[i * 3]; m[1] += pr.alb[i * 3 + 1]; m[2] += pr.alb[i * 3 + 2]; }
        const k = m.map((v) => 0.58 / Math.max(0.02, v / (P * P)));
        for (let i = 0; i < P * P; i++) for (let c = 0; c < 3; c++) pr.alb[i * 3 + c] *= k[c];
      }
      // texel values once per source texel (P x P), then nearest-tiled into the S x S layer
      const PA = new Uint8Array(P * P * 4), PN = new Uint8Array(P * P * 4);
      for (let py = 0; py < P; py++) for (let px = 0; px < P; px++) {
        const i = py * P + px, d = i * 4;
        PA[d] = Math.min(255, Math.pow(pr.alb[i * 3], 1 / 2.2) * 255); PA[d + 1] = Math.min(255, Math.pow(pr.alb[i * 3 + 1], 1 / 2.2) * 255); PA[d + 2] = Math.min(255, Math.pow(pr.alb[i * 3 + 2], 1 / 2.2) * 255); PA[d + 3] = 255;
        const xm = (px + P - 1) % P, xp = (px + 1) % P, ym = (py + P - 1) % P, yp = (py + 1) % P;
        const dx = (pr.hgt[py * P + xp] - pr.hgt[py * P + xm]) * 2.5, dy = (pr.hgt[yp * P + px] - pr.hgt[ym * P + px]) * 2.5;
        const ln = Math.hypot(dx, dy, 1);
        PN[d] = (-dx / ln * 0.5 + 0.5) * 255; PN[d + 1] = (-dy / ln * 0.5 + 0.5) * 255; PN[d + 2] = pr.rgh[i] * 255; PN[d + 3] = 255;
      }
      const xs = new Int32Array(S); for (let x = 0; x < S; x++) xs[x] = Math.floor((x * P) / S) * 4;
      let lastPy = -1, lastRow = 0;
      for (let y = 0; y < S; y++) {
        const py = Math.floor((y * P) / S), d0 = o + y * row;
        if (py === lastPy) { A.copyWithin(d0, lastRow, lastRow + row); N.copyWithin(d0, lastRow, lastRow + row); continue; }
        const s0 = py * P * 4;
        for (let x = 0; x < S; x++) {
          const sx = s0 + xs[x], d = d0 + x * 4;
          A[d] = PA[sx]; A[d + 1] = PA[sx + 1]; A[d + 2] = PA[sx + 2]; A[d + 3] = PA[sx + 3];
          N[d] = PN[sx]; N[d + 1] = PN[sx + 1]; N[d + 2] = PN[sx + 2]; N[d + 3] = PN[sx + 3];
        }
        lastPy = py; lastRow = d0;
      }
    }
    if (stats[l]) endStat(stats[l], imgs && imgs[0] && imgs[1] ? undefined : 'procedural');
    else { const st = beginStat('array', `(${s.role})`); endStat(st, 'procedural'); } // id-less layer: always procedural
  }
  const albedo = makeDataArrayTexture(A, S, L, true, renderer), normal = makeDataArrayTexture(N, S, L, false, renderer);
  // GPU memory of the two RGBA8 arrays (+ mips), credited to the wave's totals
  TextureSets.stats.totals.gpuBytes += 2 * S * S * 4 * L * MIP;
  return { albedo, normal, sizes: sets.map((s) => s.size), tintable: sets.map((s) => (s.tint ? 1 : 0)), layers: L, loaded, wanted: want };
}
