#!/usr/bin/env node
// Bake the DISTANT relief around a map square (any lat / lon; default: the San Francisco Bay Area) into a
// farland.json (no npm deps).
//
//   node engine/tools/bake-farland.mjs --city public/map/city.json --out public/map/farland.json --cache .terrain-cache
//   node engine/tools/bake-farland.mjs --dry-run   (grid + tile count + paths; no network, no files)   --help
//   export async function bakeFarland(opts)        (same options, camelCase; used by the game's tools/bake-farland.mjs)
//
// The detailed terrain (bake-terrain.mjs) stops at the map square; beyond it the world only has the water plane.
// This grid (--cell m cells over +---half m) is drawn by engine/src/world/Shore.js `buildFarLand` as a hazy
// silhouette ring (depth-compressed to fit inside the camera far plane). Same source + projection as bake-terrain.mjs:
// AWS / Mapzen Terrarium tiles (USGS 3DEP / NED), elevation = R*256 + G + B/256 - 32768.
// Output frame = city.json BAKE frame (x = east, z = SOUTH, metres around CENTER); the World shifts it
// by world.offset. Heights are integer metres, water clamped to -5.

import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { runCli } from './cli.mjs';

/** CLI flags / bakeFarland options (defaults: the Roadster San Francisco Bay Area bake). */
export const SPEC = {
  city: { type: 'string', def: 'public/map/city.json', help: 'city.json (centre when --lat / --lon are not given)' },
  lat: { type: 'number', help: 'centre latitude (default: city.json center)' },
  lon: { type: 'number', help: 'centre longitude (default: city.json center)' },
  half: { type: 'number', def: 24000, help: 'half extent of the grid (m)' },
  cell: { type: 'number', def: 250, help: 'grid cell (m)' },
  zoom: { type: 'number', def: 11, help: 'Terrarium zoom' },
  sea: { type: 'number', def: -5, help: 'water clamp (m)' },
  out: { type: 'string', def: 'public/map/farland.json', help: 'output farland.json' },
  cache: { type: 'string', def: '.terrain-cache', help: 'tile cache dir (shared with bake-terrain)' },
  refresh: { type: 'bool', def: false, help: 're-download tiles' },
  'dry-run': { type: 'bool', def: false, help: 'print the plan only (no network, no files)' },
};

// configuration (set by configure(); read by the functions below at call time)
let ROOT, OUT, CITY, CACHE, REFRESH, CENTER, HALF, CELL, ZOOM, SEA, COS0;
function configure(o = {}) {
  const d = (k) => (o[k] !== undefined ? o[k] : SPEC[k.replace(/[A-Z]/g, (c) => '-' + c.toLowerCase())].def);
  ROOT = process.cwd();
  OUT = path.resolve(d('out')); CACHE = path.resolve(d('cache')); REFRESH = !!d('refresh');
  CITY = d('city') ? path.resolve(d('city')) : null;
  const city = CITY && fs.existsSync(CITY) ? JSON.parse(fs.readFileSync(CITY, 'utf8')) : {};
  const c0 = city.center || { lat: 37.792, lon: -122.3965 };
  CENTER = { lat: o.lat ?? c0.lat, lon: o.lon ?? c0.lon };
  HALF = d('half'); CELL = d('cell'); ZOOM = d('zoom'); SEA = d('sea');
  COS0 = Math.cos(CENTER.lat * DEG);
}

const R = 6378137, DEG = Math.PI / 180;
const unprojLat = (z) => CENTER.lat - z / (DEG * R);
const unprojLon = (x) => CENTER.lon + x / (DEG * R * COS0);

function decodePNG(buf) {
  const sig = [137, 80, 78, 71, 13, 10, 26, 10];
  for (let i = 0; i < 8; i++) if (buf[i] !== sig[i]) throw new Error('not a PNG');
  let off = 8, width = 0, height = 0, depth = 0, ctype = 0, interlace = 0, palette = null;
  const idat = [];
  while (off < buf.length) {
    const len = buf.readUInt32BE(off), type = buf.toString('ascii', off + 4, off + 8);
    const data = buf.subarray(off + 8, off + 8 + len);
    if (type === 'IHDR') { width = data.readUInt32BE(0); height = data.readUInt32BE(4); depth = data[8]; ctype = data[9]; interlace = data[12]; }
    else if (type === 'PLTE') palette = data;
    else if (type === 'IDAT') idat.push(data);
    else if (type === 'IEND') break;
    off += 12 + len;
  }
  if (depth !== 8 || interlace) throw new Error(`unsupported PNG depth ${depth} interlace ${interlace}`);
  const bpp = { 2: 3, 6: 4, 3: 1, 0: 1, 4: 2 }[ctype];
  if (!bpp) throw new Error(`unsupported PNG colour type ${ctype}`);
  const raw = zlib.inflateSync(Buffer.concat(idat));
  const stride = width * bpp;
  const px = Buffer.alloc(stride * height);
  let prev = Buffer.alloc(stride);
  for (let y = 0; y < height; y++) {
    const ft = raw[y * (stride + 1)];
    const line = raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1));
    const cur = px.subarray(y * stride, (y + 1) * stride);
    for (let i = 0; i < stride; i++) {
      const a = i >= bpp ? cur[i - bpp] : 0, b = prev[i], c = i >= bpp ? prev[i - bpp] : 0;
      let v = line[i];
      if (ft === 1) v += a; else if (ft === 2) v += b; else if (ft === 3) v += (a + b) >> 1;
      else if (ft === 4) { const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c); v += pa <= pb && pa <= pc ? a : pb <= pc ? b : c; }
      cur[i] = v & 255;
    }
    prev = cur;
  }
  const rgb = new Uint8Array(width * height * 3);
  for (let i = 0; i < width * height; i++) {
    if (ctype === 2 || ctype === 6) { rgb[i * 3] = px[i * bpp]; rgb[i * 3 + 1] = px[i * bpp + 1]; rgb[i * 3 + 2] = px[i * bpp + 2]; }
    else if (ctype === 3) { const k = px[i] * 3; rgb[i * 3] = palette[k]; rgb[i * 3 + 1] = palette[k + 1]; rgb[i * 3 + 2] = palette[k + 2]; }
    else { rgb[i * 3] = rgb[i * 3 + 1] = rgb[i * 3 + 2] = px[i * bpp]; }
  }
  return { width, height, rgb };
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const tiles = new Map();
let downloaded = 0;
async function fetchTile(z, x, y) {
  const k = `${z}/${x}/${y}`;
  if (tiles.has(k)) return tiles.get(k);
  const file = path.join(CACHE, `terrarium_${z}_${x}_${y}.png`);
  let buf = null;
  if (!REFRESH && fs.existsSync(file)) buf = fs.readFileSync(file);
  else {
    const url = `https://s3.amazonaws.com/elevation-tiles-prod/terrarium/${z}/${x}/${y}.png`;
    for (let attempt = 0; attempt < 4 && !buf; attempt++) {
      try {
        const res = await fetch(url, { headers: { 'User-Agent': 'roadster-terrain-baker/1.0' } });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        buf = Buffer.from(await res.arrayBuffer());
        fs.writeFileSync(file, buf);
        downloaded++;
      } catch (e) { console.warn(`  tile ${k} attempt ${attempt + 1}: ${e.message}`); await sleep(1000 * (attempt + 1)); }
    }
    if (!buf) throw new Error(`tile ${k} unavailable`);
  }
  const img = decodePNG(buf);
  const h = new Float32Array(img.width * img.height);
  for (let i = 0; i < h.length; i++) h[i] = img.rgb[i * 3] * 256 + img.rgb[i * 3 + 1] + img.rgb[i * 3 + 2] / 256 - 32768;
  const t = { h };
  tiles.set(k, t);
  return t;
}
const tileXY = (lat, lon, z) => {
  const n = 2 ** z, la = lat * DEG;
  return [((lon + 180) / 360) * n, ((1 - Math.log(Math.tan(la) + 1 / Math.cos(la)) / Math.PI) / 2) * n];
};
function demAt(lat, lon, z) {
  const [fx, fy] = tileXY(lat, lon, z);
  const X = fx * 256 - 0.5, Y = fy * 256 - 0.5;
  const x0 = Math.floor(X), y0 = Math.floor(Y), tx = X - x0, ty = Y - y0;
  const px = (gx, gy) => {
    const t = tiles.get(`${z}/${Math.floor(gx / 256)}/${Math.floor(gy / 256)}`);
    return t ? t.h[(((gy % 256) + 256) % 256) * 256 + (((gx % 256) + 256) % 256)] : NaN;
  };
  const a = px(x0, y0), b = px(x0 + 1, y0), c = px(x0, y0 + 1), d = px(x0 + 1, y0 + 1);
  return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty;
}

/** Bake farland.json. opts: SPEC keys in camelCase (city, lat, lon, half, cell, zoom, sea, out, cache, refresh, dryRun). */
export async function bakeFarland(opts = {}) {
  configure(opts);
  const x0 = -HALF, z0 = -HALF, n = Math.round((2 * HALF) / CELL) + 1;
  const corners = [[x0, z0], [-x0, -z0]].map(([x, z]) => tileXY(unprojLat(z), unprojLon(x), ZOOM));
  const tx0 = Math.floor(Math.min(corners[0][0], corners[1][0])) - 1, tx1 = Math.floor(Math.max(corners[0][0], corners[1][0])) + 1;
  const ty0 = Math.floor(Math.min(corners[0][1], corners[1][1])) - 1, ty1 = Math.floor(Math.max(corners[0][1], corners[1][1])) + 1;
  const list = [];
  for (let x = tx0; x <= tx1; x++) for (let y = ty0; y <= ty1; y++) list.push([x, y]);
  console.log(`farland: ${n} x ${n} nodes @ ${CELL} m, z${ZOOM}: ${list.length} tiles`);
  if (opts.dryRun) {
    console.log(`  centre ${CENTER.lat}, ${CENTER.lon} (city ${CITY || '-'}), +-${HALF} m, sea clamp ${SEA} m\n  out ${OUT}\n  cache ${CACHE}${REFRESH ? ' (refresh)' : ''}\n  dry run: no requests made, nothing written`);
    return { dryRun: true, out: OUT, cache: CACHE, tiles: list.length };
  }
  fs.mkdirSync(CACHE, { recursive: true });
  for (let i = 0; i < list.length; i += 6) await Promise.all(list.slice(i, i + 6).map(([x, y]) => fetchTile(ZOOM, x, y)));
  const h = new Array(n * n);
  const S = 4; // box filter over the cell
  let land = 0, maxH = -1e9;
  for (let k = 0; k < n; k++) for (let i = 0; i < n; i++) {
    let sum = 0, cnt = 0, mx = -1e9;
    for (let a = 0; a < S; a++) for (let b = 0; b < S; b++) {
      const x = x0 + (i + (a + 0.5) / S - 0.5) * CELL, z = z0 + (k + (b + 0.5) / S - 0.5) * CELL;
      const v = demAt(unprojLat(z), unprojLon(x), ZOOM);
      if (Number.isFinite(v)) { sum += v; cnt++; if (v > mx) mx = v; }
    }
    // keep ridgelines: bias the box mean toward the cell maximum on hills
    let v = cnt ? sum / cnt : SEA;
    if (v > 20) v = v * 0.7 + mx * 0.3;
    const hv = Math.max(SEA, Math.round(v));
    h[k * n + i] = hv;
    if (hv > 0) land++;
    if (hv > maxH) maxH = hv;
  }
  const out = {
    v: 1, source: 'AWS Terrain Tiles (Mapzen Terrarium, USGS 3DEP/NED) z11, 4x4 box filter',
    generated: new Date().toISOString(), frame: 'bake (city.json coordinates; World applies world.offset)',
    center: CENTER, units: 'm', origin: [x0, z0], cell: CELL, nx: n, nz: n, sea: SEA, h,
  };
  fs.writeFileSync(OUT, JSON.stringify(out));
  console.log(`wrote ${path.relative(ROOT, OUT)}: ${(fs.statSync(OUT).size / 1024).toFixed(0)} KB, land ${(100 * land / (n * n)).toFixed(1)} %, max ${maxH} m, ${downloaded} tiles downloaded`);
  return { out: OUT };
}

runCli(import.meta.url, 'engine/tools/bake-farland.mjs', 'bake the distant relief ring (Terrarium tiles) into a farland.json', SPEC, bakeFarland);
