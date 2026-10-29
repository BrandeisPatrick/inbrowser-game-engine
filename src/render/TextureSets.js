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
//   makeTextureArray(renderer, sets, { size, base, manifest, onProgress, ktx2?, free? })
//        sets: [{ role, id|null, size (m), tint (bool), proc (fallback kind) }]
//        -> { albedo, normal, sizes, tintable, layers, loaded, wanted }
//        albedo (sRGB DataArrayTexture): rgb = albedo (tintable sets normalised to neutral grey), a = AO
//        normal (linear):                rg = tangent normal xy (OpenGL), b = roughness
//        (procedural layers generated in a worker pool, see "Procedural layers" below; `free: false` keeps the CPU
//         copies albedo.image.data / normal.image.data after the GPU upload, e.g. to hash them)
//   TextureSets.stats / .options / .configure(o)   per-set path (ktx2 | jpg | mixed | procedural), bytes, decode ms,
//                                                  main-thread packing ms, GPU bytes; one `[TextureSets]` summary
//                                                  line per load wave
//   ktx2Support(renderer)                          -> { loader (shared KTX2Loader), formats, ready } | null
//   ?ktx2=0 (or ?nokx) in the page URL forces the JPG path (A/B comparisons)
//   ?procworkers=0 generates procedural array layers on the main thread (A/B of their main-thread cost; =N caps the
//   pool at N workers); overrides TextureSets.options.proceduralWorkers
//   proceduralPbr(kind, size, seed)               -> { alb, hgt, rgh } float maps (fallback painter; lives in
//                                                  proceduralPbr.js with the layer packing, re-exported here)
//   makeAsphaltTexture / makePavingTexture (= makeConcreteTexture) / makeGroundTexture / makeGrassTexture /
//   makeWaterNormalTexture (renderer)             -> tileable CanvasTexture
//   makeCanvas, paintCanvas, finishTexture, makeDataArrayTexture        low-level helpers
//
// Procedural layers off the main thread. An array layer without photo maps (id-less sets, roles the manifest does not
// list, sets whose diff / normal map failed) used to be generated in the packing loop: proceduralTexels (256 px of fBm
// noise, tint normalisation, sRGB / normal packing) costs ~20-40 ms per layer on an M-series core, 64-132 ms for
// Canyon Run's three, on the thread that is the bottleneck during load. It now runs in a small pool of module workers
// (proceduralTextures.worker.js; up to TextureSets.options.proceduralWorkers, spawned on demand, one job each, lowest
// layer first, retired after 10 s idle). makeTextureArray dispatches a layer as soon as it is known to be procedural
// (id-less layers before the manifest is even fetched, the others once the manifest / their fetches tell), so the
// noise overlaps the downloads and whatever the caller does meanwhile; when the loop reaches the layer it only
// nearest-tiles the returned 256 px texels into its slice of the arrays (~1 ms at 1024 px, tileTexels). Both threads
// run the same module code (proceduralPbr.js), so the bytes equal the in-thread path's, which still runs without
// `Worker` (Node, the headless tests), with `proceduralWorkers: false`, and for every layer once a worker failed to
// load / errored / did not answer within 10 s (one console warning, then in-thread for the rest of the page).
// stats: packMs of a procedural layer = its main-thread cost (whole generation in-thread, or dispatch + tiling),
// procIn = 'worker' | 'main', workerMs = generation time in the worker; totals.procMs / workerMs / procWorker.
import * as THREE from 'three';
import { KTX2Loader } from 'three/addons/loaders/KTX2Loader.js';
import { mulberry32, pfbm, hash2 } from './noise.js';
import { proceduralPbr, proceduralTexels, tileTexels } from './proceduralPbr.js';

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

/**
 * Fill a canvas with fn(x, y, out) -> [r, g, b] (0..255); returns its 2D context for further drawing. `out` is a
 * reusable 3-array the painter may fill and return instead of allocating one per texel (a megapixel of short-lived
 * arrays otherwise, most of a painter's time in the GC).
 */
export function paintCanvas(c, fn) {
  const g = ctx2d(c);
  const img = g.createImageData(c.width, c.height);
  const d = img.data, out = [0, 0, 0], W = c.width, H = c.height;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const o = (y * W + x) * 4;
    const v = fn(x, y, out) || out;
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
  const g = paintCanvas(c, (x, y, o) => {
    const u = x / S, v = y / S;
    const big = pfbm(u, v, 4, 4, 1);
    const fine = hash2(x * 3 + 1, y * 7 + 3);
    const grain = pfbm(u, v, 64, 2, 5);
    let l = 46 + big * 22 + (fine - 0.5) * 26 + (grain - 0.5) * 14;
    if (fine > 0.985) l += 40; // light aggregate stones
    o[0] = l; o[1] = l * 1.0; o[2] = l * 1.02; return o;
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
  const g = paintCanvas(c, (x, y, o) => {
    const u = x / S, v = y / S;
    const n = pfbm(u, v, 8, 4, 21);
    const f = hash2(x + 9, y * 3);
    const slab = hash2(Math.floor(x / 128), Math.floor(y / 128) + 77);
    let l = 168 + n * 30 + (f - 0.5) * 14 + (slab - 0.5) * 16;
    const jx = x % 128, jy = y % 128;
    if (jx < 2 || jy < 2) l -= 52;
    else if (jx < 4 || jy < 4) l -= 12;
    o[0] = l; o[1] = l * 0.985; o[2] = l * 0.95; return o;
  });
  g.globalAlpha = 1;
  return finishTexture(new THREE.CanvasTexture(c), renderer);
}
export const makeConcreteTexture = makePavingTexture;

/** Generic weathered ground with faint large slabs (1024 px; ~6 m tile). */
export function makeGroundTexture(renderer) {
  const S = 1024, c = makeCanvas(S);
  paintCanvas(c, (x, y, o) => {
    const u = x / S, v = y / S;
    const n = pfbm(u, v, 3, 5, 31);
    const f = hash2(x * 5, y * 11 + 1);
    const slabs = (x % 256 < 2 || y % 256 < 2) ? -30 : 0;
    const l = 118 + n * 40 + (f - 0.5) * 18 + slabs;
    o[0] = l * 1.0; o[1] = l * 0.98; o[2] = l * 0.93; return o;
  });
  return finishTexture(new THREE.CanvasTexture(c), renderer);
}

/** Grass (512 px; ~4 m tile). */
export function makeGrassTexture(renderer) {
  const S = 512, c = makeCanvas(S);
  paintCanvas(c, (x, y, o) => {
    const u = x / S, v = y / S;
    const n = pfbm(u, v, 4, 5, 41);
    const m = pfbm(u, v, 16, 3, 43);
    const f = hash2(x * 13, y * 17);
    const l = 0.75 + n * 0.45 + (f - 0.5) * 0.35;
    o[0] = 52 * l + m * 18; o[1] = 92 * l + m * 20; o[2] = 34 * l; return o;
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
  paintCanvas(c, (x, y, o) => {
    const xm = (x + S - 1) % S, xp = (x + 1) % S, ym = (y + S - 1) % S, yp = (y + 1) % S;
    const dx = (hgt[y * S + xp] - hgt[y * S + xm]) * 6;
    const dy = (hgt[yp * S + x] - hgt[ym * S + x]) * 6;
    const l = Math.hypot(dx, dy, 1);
    o[0] = (-dx / l * 0.5 + 0.5) * 255; o[1] = (-dy / l * 0.5 + 0.5) * 255; o[2] = (1 / l * 0.5 + 0.5) * 255; return o;
  });
  return finishTexture(new THREE.CanvasTexture(c), renderer, false);
}

// ------------------------------------------------------------------ procedural PBR fallback
// proceduralPbr (the fallback material painter) and the array-layer packing (proceduralTexels / tileTexels) live in
// proceduralPbr.js: pure JS, shared with proceduralTextures.worker.js. Re-exported for existing importers.
export { proceduralPbr };

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
    proceduralWorkers: true, // texture arrays: procedural layers generated in module workers (see header), true = up
                           // to 4 (hardwareConcurrency - 1), a number = at most that many, false = on the main thread
                           // as before. On by default although it is new behaviour, because only *where* the noise
                           // runs changes: same module code and packing, byte-identical layers (content hash); no
                           // Worker (Node / tests) or a failing worker = the previous in-thread path. URL
                           // ?procworkers=0 / =N overrides it (A/B in the browser without code changes)
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
    //     download, summed; includes worker queueing), packMs (array layers: main-thread readback + packing, 0.1 ms;
    //     procedural layers: the whole generation in-thread, or the dispatch + tiling of a worker's texels),
    //     procIn: 'worker' | 'main' + workerMs (procedural layers only: where the noise ran, its time in the worker),
    //     gpuBytes (textures this set owns; the arrays' RGBA8 storage is added to totals.gpuBytes), formats }
    sets: {},
    // procMs = the procedural layers' share of packMs (main thread), workerMs = their generation time in workers,
    // procWorker = how many of them were generated in workers
    totals: { sets: 0, ktx2: 0, jpg: 0, mixed: 0, procedural: 0, missing: 0, bytes: 0, ktx2Bytes: 0, jpgBytes: 0, gpuBytes: 0, decodeMs: 0, packMs: 0, procMs: 0, workerMs: 0, procWorker: 0, wallMs: 0 },
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
const r1 = (v) => Math.round(v * 10) / 10; // ms fields at 0.1 ms (procedural layers on the worker path cost ~1 ms)
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
  st.ms = Math.round(performance.now() - st.t0); st.decodeMs = Math.round(st.decodeMs); st.packMs = r1(st.packMs);
  delete st.t0;
  const S = TextureSets.stats, T = S.totals, key = `${st.consumer}:${st.id}`;
  if (!S.sets[key]) T.sets++;
  S.sets[key] = st;
  T[st.path] = (T[st.path] || 0) + 1;
  T.bytes += st.bytes; T.gpuBytes += st.gpuBytes; T.decodeMs += st.decodeMs; T.packMs = r1(T.packMs + st.packMs);
  if (st.procIn) { T.procMs = r1(T.procMs + st.packMs); T.workerMs = r1(T.workerMs + st.workerMs); if (st.procIn === 'worker') T.procWorker++; }
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
  // procedural layers: their main-thread share of the packing, and where the noise ran (e.g. "procedural 3 layers in
  // workers: 2.6 ms main-thread, 95 ms off-thread"; with proceduralWorkers off "procedural 3 layers: 71.4 ms main-thread")
  const np = T.procedural, nw = T.procWorker;
  const proc = np ? `; procedural ${np} layer${np > 1 ? 's' : ''}${nw === np ? ' in workers' : nw ? ` (${nw} in workers)` : ''}: ${T.procMs.toFixed(1)} ms main-thread${nw ? `, ${T.workerMs.toFixed(0)} ms off-thread` : ''}` : '';
  console.info(`[TextureSets] ${T.sets} sets: ${by} · ${(T.bytes / 1048576).toFixed(1)} MB downloaded (ktx2 ${(T.ktx2Bytes / 1048576).toFixed(1)}, jpg ${(T.jpgBytes / 1048576).toFixed(1)}) · ~${(T.gpuBytes / 1048576).toFixed(0)} MB GPU · ready in ${T.wallMs} ms (main-thread packing ${T.packMs.toFixed(1)} ms${proc}) · ${[...fm].join(', ') || 'no textures'}${S.ktx2 === false ? ' (KTX2 off)' : ''}`);
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

// --- procedural layer worker pool (texture-array path; see "Procedural layers" in the header): module workers running
// proceduralTexels, spawned on demand up to the option's count, one job in flight per worker, lowest layer first (the
// packing loop places layers in order). A worker that fails to load, errors or does not answer within PROC_TIMEOUT ms
// fails the pool for the rest of the page: every pending job rejects and those layers are generated in-thread (same
// bytes, only main-thread time lost). Workers retire 10 s after the last job, like rgbaPool's.
const PROC_PX = 256, PROC_TIMEOUT = 10000; // procedural layer resolution (nearest-tiled up to the array size); ms
let _proc = null, _procFailed = null, _procWarned = false;
function warnProc(msg) { if (!_procWarned) console.warn(`[TextureSets] procedural texture workers unavailable (${msg}); generating procedural layers on the main thread`); _procWarned = true; }
/** URL switch: ?procworkers=0 -> in-thread, ?procworkers=N -> at most N workers, else null (the option decides). */
function urlProcWorkers() {
  try {
    const v = new URLSearchParams(globalThis.location ? globalThis.location.search : '').get('procworkers');
    if (v !== null && v !== '') return Math.max(0, +v | 0);
  } catch { /* no location */ }
  return null;
}
/** Workers for procedural layers; 0 = generate in-thread (no Worker: Node / tests; option off; the pool failed). */
function procWorkerCount() {
  const u = urlProcWorkers(), o = u !== null ? u : TextureSets.options.proceduralWorkers;
  if (!o || _procFailed || typeof Worker === 'undefined') return 0;
  if (o === true) return Math.max(1, Math.min(4, ((globalThis.navigator && globalThis.navigator.hardwareConcurrency) || 4) - 1));
  return Math.max(0, o | 0);
}
function procPool(n) {
  if (_proc) return _proc;
  const workers = [], busy = [], timers = [], jobs = new Map(), queue = [];
  let seq = 0, idle = 0, dead = false;
  const retire = () => { idle = 0; if (jobs.size) return; for (const w of workers) w.terminate(); if (_proc === pool) _proc = null; };
  const failAll = (msg) => {
    if (dead) return; // every worker of a broken pool reports it (e.g. the module 404s); act once
    dead = true;
    warnProc(msg);
    _procFailed = msg || 'worker failed';
    for (const t of timers) clearTimeout(t);
    if (idle) { clearTimeout(idle); idle = 0; }
    for (const w of workers) { try { w.terminate(); } catch { /* gone */ } }
    queue.length = 0;
    for (const [id, j] of jobs) { jobs.delete(id); j.rej(new Error(_procFailed)); }
    if (_proc === pool) _proc = null;
  };
  const spawn = () => {
    const i = workers.length;
    // keep this literal form: it is what Vite detects to bundle the worker (+ proceduralPbr.js / noise.js) and rewrite
    // its URL in builds; served as a module worker in dev
    const w = new Worker(new URL('./proceduralTextures.worker.js', import.meta.url), { type: 'module' });
    w.onmessage = (e) => {
      const d = e.data;
      if (d.error) return failAll(d.error); // deterministic code: what throws for one layer throws for all of them
      clearTimeout(timers[i]); timers[i] = 0;
      if (busy[i] === d.id) busy[i] = 0;
      const j = jobs.get(d.id);
      if (j) { jobs.delete(d.id); j.res(d); }
      pump();
      if (!jobs.size && !idle) idle = setTimeout(retire, 10000);
    };
    w.onerror = (e) => { if (e && e.preventDefault) e.preventDefault(); failAll((e && e.message) || 'worker failed to load'); };
    w.onmessageerror = () => failAll('unreadable worker message');
    workers.push(w); busy.push(0); timers.push(0);
  };
  const pump = () => {
    while (queue.length) {
      let i = busy.indexOf(0);
      if (i < 0) { if (workers.length >= n) return; spawn(); i = workers.length - 1; }
      let b = 0; for (let k = 1; k < queue.length; k++) if (queue[k].prio < queue[b].prio) b = k;
      const j = queue.splice(b, 1)[0];
      busy[i] = j.msg.id;
      timers[i] = setTimeout(() => failAll(`no answer within ${PROC_TIMEOUT / 1000} s`), PROC_TIMEOUT);
      workers[i].postMessage(j.msg);
    }
  };
  const pool = {
    /** P x P texels of one procedural layer (proceduralTexels) -> Promise<{ pa, pn, ms }>; rejects when the pool fails. */
    run(kind, P, seed, tint, prio = 0) {
      if (idle) { clearTimeout(idle); idle = 0; }
      return new Promise((res, rej) => {
        const id = ++seq;
        jobs.set(id, { res, rej }); queue.push({ prio, msg: { id, kind, P, seed, tint } });
        try { pump(); } catch (e) { failAll(e && e.message ? e.message : String(e)); } // new Worker threw (CSP, file://)
      });
    },
  };
  return (_proc = pool);
}
/** Dispatch one procedural layer -> Promise<{ pa, pn, ms }>, or null when it is to be generated in-thread. */
function procJob(kind, seed, tint, prio) {
  const n = procWorkerCount();
  if (!n) return null;
  const job = procPool(n).run(kind, PROC_PX, seed, tint, prio);
  job.catch(() => {}); // handled where the layer is placed (in-thread fallback); no unhandled rejection meanwhile
  return job;
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
 * sets fall back to proceduralPbr at 256 px tiled up to `size`, generated in the procedural worker pool from the
 * moment a layer is known to need it (TextureSets.options.proceduralWorkers; same bytes as in-thread). Tintable sets
 * are normalised to a neutral mean so a per-instance tint reaches real paint albedos. With KTX2 each map comes from
 * whichever of `.ktx2` / `.jpg` TextureSets.options.arrayKtx2 picks (default: the smaller download); both are packed
 * into the same RGBA8 layers, so the shaders that sample the arrays see the same layout either way. `free: false`
 * keeps albedo.image.data / normal.image.data after the GPU upload (default: freed). Never throws for missing files.
 */
export async function makeTextureArray(renderer, sets, { size = 1024, base = textureBase(), manifest, onProgress, ktx2, free = true } = {}) {
  // procedural layers go to the worker pool as soon as they are known: id-less sets now (before the manifest), roles
  // the manifest lacks and photo sets whose diff / normal failed below. proc[l] = { job (null = in-thread), ms: the
  // main-thread cost of dispatching it }; the seed depends only on the layer index, the texels on kind / seed / tint
  const L = sets.length, proc = new Array(L).fill(null);
  const procSeed = (l) => 97 + l * 13;
  const dispatch = (l) => {
    if (!proc[l]) { const t = performance.now(), s = sets[l], job = procJob(s.proc, procSeed(l), !!s.tint, l); proc[l] = { job, ms: performance.now() - t }; }
    return proc[l];
  };
  for (let l = 0; l < L; l++) if (!sets[l].id) dispatch(l);
  if (manifest === undefined) {
    // boot the transcode workers while the manifest loads (they retire after 10 s idle if it lists no .ktx2)
    if (ktx2Wanted(ktx2) && TextureSets.options.arrayKtx2 !== false && !_rgbaFailed && typeof Worker !== 'undefined') rgbaPool();
    manifest = await loadTextureManifest(base);
  }
  sets = sets.map((s) => {
    const m = manifest && manifest.sets && manifest.sets[s.role];
    return { ...s, size: m && m.size ? m.size[0] : s.size, maps: m ? m.maps || {} : null, kx: m && m.ktx2Maps ? m.ktx2Maps : null };
  });
  const S = size;
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
  const isPhoto = (imgs) => !!(imgs && imgs[0] && imgs[1]); // else the layer is procedural
  sets.forEach((s, l) => {
    if (!s.id || (manifest && !s.maps)) dispatch(l);
    else jobs[l].then((imgs) => { if (!isPhoto(imgs)) dispatch(l); }, () => dispatch(l));
  });
  // a decoded map as { d: RGBA8 S x S, up: rows bottom-up (KTX2) or top-down (canvas readback of a JPG) }
  const pix = (r) => (!r ? null : r.data ? { d: r.data, up: true } : { d: pixelsOf(r.bmp, S, scratch), up: false });
  for (let l = 0; l < L; l++) {
    const s = sets[l];
    let imgs = null;
    try { imgs = await jobs[l]; } catch { imgs = null; }
    const o = l * S * S * 4, row = S * 4;
    if (isPhoto(imgs)) {
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
      // procedural fallback at 256 px, nearest-tiled up to S: the worker's texels (dispatched earlier, normally done by
      // now), else generated here; only the tiling (and the dispatch) is main-thread time on the worker path
      const d = dispatch(l);
      let r = null;
      if (d.job) { try { r = await d.job; } catch (e) { warnProc(e && e.message ? e.message : e); } }
      if (r && !(r.pa && r.pa.length === PROC_PX * PROC_PX * 4 && r.pn && r.pn.length === PROC_PX * PROC_PX * 4)) r = null;
      const t = performance.now();
      const px = r || proceduralTexels(s.proc, PROC_PX, procSeed(l), s.tint);
      tileTexels(px.pa, px.pn, PROC_PX, S, A, N, o);
      const st = stats[l] || (stats[l] = beginStat('array', `(${s.role})`)); // id-less / unlisted layer: always procedural
      st.packMs += d.ms + performance.now() - t;
      st.procIn = r ? 'worker' : 'main'; st.workerMs = r ? r1(r.ms) : 0;
    }
    endStat(stats[l], isPhoto(imgs) ? undefined : 'procedural');
  }
  const albedo = makeDataArrayTexture(A, S, L, true, renderer, { free }), normal = makeDataArrayTexture(N, S, L, false, renderer, { free });
  // GPU memory of the two RGBA8 arrays (+ mips), credited to the wave's totals
  TextureSets.stats.totals.gpuBytes += 2 * S * S * 4 * L * MIP;
  return { albedo, normal, sizes: sets.map((s) => s.size), tintable: sets.map((s) => (s.tint ? 1 : 0)), layers: L, loaded, wanted: want };
}
