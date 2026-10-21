#!/usr/bin/env node
// Bake real elevation for a city.json's area (any lat / lon; default: San Francisco) into a terrain.json
// (the Terrain contract, engine/src/world/Terrain.js). No npm deps, Node >= 18.
//
//   node engine/tools/bake-terrain.mjs --city public/map/city.json --out public/map/terrain.json --cache .terrain-cache
//   node engine/tools/bake-terrain.mjs --dry-run   (grids + tile counts + paths; no network, no files)   --help
//   export async function bakeTerrain(opts)        (same options, camelCase; used by the game's tools/bake-terrain.mjs)
//
// Source: AWS / Mapzen "Terrarium" elevation tiles (USGS 3DEP/NED in the US, ~10 m native), decoded
// with a tiny built-in PNG decoder:  elevation = R*256 + G + B/256 - 32768  (metres, ~MSL / NAVD88).
// Fallback when that host is unreachable: OpenTopoData ned10m point queries on a coarser grid.
//
// Output frame = the city.json BAKE frame (metres around CENTER, x = east, z = SOUTH), i.e. BEFORE the
// world shifts everything by `world.offset` so that spawn is (0,0). engine Terrain applies
// that offset when sampling. Two node-centred grids, row-major (z rows, then x), Int16 decimetres:
//   fine : --fine rect, 10 m cells (the playable area; SF default: the filler ring + Yerba Buena Island)
//   outer: 40 m cells over the whole --outer land square (horizon hills)
// Bay / negative values are clamped to -2 m. Land that the DEM reports at/below sea level (piers,
// coastal pixel bleed) is filled from the nearest real land, and water nodes next to the coast copy
// the nearest land height so bilinear sampling never drags the shoreline down.

import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { runCli } from './cli.mjs';

/** CLI flags / bakeTerrain options (defaults: the Roadster San Francisco bake). */
export const SPEC = {
  city: { type: 'string', def: 'public/map/city.json', help: 'city.json (land / pier rings for the coast fill; centre / outer when not given)' },
  lat: { type: 'number', help: 'centre latitude (default: city.json center)' },
  lon: { type: 'number', help: 'centre longitude (default: city.json center)' },
  outer: { type: 'number', help: 'land square edge, m (default: city.json outer or 9000)' },
  fine: { type: 'numbers', def: [-2600, -2900, 3500, 2500], help: 'fine grid rect x0,z0,x1,z1 (bake frame m)' },
  'fine-cell': { type: 'number', def: 10, help: 'fine grid cell (m)' },
  'fine-zoom': { type: 'number', def: 15, help: 'Terrarium zoom of the fine grid' },
  'outer-cell': { type: 'number', def: 40, help: 'outer grid cell (m)' },
  'outer-zoom': { type: 'number', def: 13, help: 'Terrarium zoom of the outer grid' },
  'bay-min': { type: 'number', def: -2, help: 'water / negative heights are clamped to this (m)' },
  checks: { type: 'string', help: 'JSON file [[name, lat, lon], ...] printed as a height check at the end' },
  out: { type: 'string', def: 'public/map/terrain.json', help: 'output terrain.json' },
  cache: { type: 'string', def: '.terrain-cache', help: 'tile cache dir' },
  refresh: { type: 'bool', def: false, help: 're-download tiles' },
  'dry-run': { type: 'bool', def: false, help: 'print the plan only (no network, no files)' },
};

// configuration (set by configure(); read by the functions below at call time)
let ROOT, OUT, CITY, CACHE, REFRESH, city, CENTER, OUTER, FINE, COARSE, BAY_MIN, COS0, landRings;
function configure(o = {}) {
  const d = (k) => (o[k] !== undefined ? o[k] : SPEC[k.replace(/[A-Z]/g, (c) => '-' + c.toLowerCase())].def);
  ROOT = process.cwd();
  OUT = path.resolve(d('out')); CACHE = path.resolve(d('cache')); REFRESH = !!d('refresh');
  CITY = d('city') ? path.resolve(d('city')) : null;
  city = CITY && fs.existsSync(CITY) ? JSON.parse(fs.readFileSync(CITY, 'utf8')) : {};
  const c0 = city.center || { lat: 37.792, lon: -122.3965 };
  CENTER = { lat: o.lat ?? c0.lat, lon: o.lon ?? c0.lon };
  OUTER = o.outer ?? (city.outer || 9000);
  // fine grid (SF default: filler ring = city half 1100 + 1000 m ring, + margin, and Yerba Buena Island / bridge approach)
  const f = d('fine');
  FINE = { x0: f[0], z0: f[1], x1: f[2], z1: f[3], cell: d('fineCell'), zoom: d('fineZoom') };
  COARSE = { x0: -OUTER / 2 - 80, z0: -OUTER / 2 - 80, x1: OUTER / 2 + 80, z1: OUTER / 2 + 80, cell: d('outerCell'), zoom: d('outerZoom') };
  BAY_MIN = d('bayMin');
  COS0 = Math.cos(CENTER.lat * DEG);
  landRings = [...(city.land || []), ...(city.piers || [])].map((q) => q && q.p).filter((p) => Array.isArray(p) && p.length > 2);
}

// ---------------------------------------------------------------- projection (same as bake-map.mjs)
const R = 6378137, DEG = Math.PI / 180;
const proj = (lat, lon) => [(lon - CENTER.lon) * DEG * R * COS0, -(lat - CENTER.lat) * DEG * R];
const unprojLat = (z) => CENTER.lat - z / (DEG * R);
const unprojLon = (x) => CENTER.lon + x / (DEG * R * COS0);

// ---------------------------------------------------------------- minimal PNG decoder
function decodePNG(buf) {
  const sig = [137, 80, 78, 71, 13, 10, 26, 10];
  for (let i = 0; i < 8; i++) if (buf[i] !== sig[i]) throw new Error('not a PNG');
  let off = 8, width = 0, height = 0, depth = 0, ctype = 0, interlace = 0, palette = null;
  const idat = [];
  while (off < buf.length) {
    const len = buf.readUInt32BE(off), type = buf.toString('ascii', off + 4, off + 8);
    const data = buf.subarray(off + 8, off + 8 + len);
    if (type === 'IHDR') {
      width = data.readUInt32BE(0); height = data.readUInt32BE(4); depth = data[8]; ctype = data[9]; interlace = data[12];
    } else if (type === 'PLTE') palette = data;
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
      switch (ft) {
        case 0: break;
        case 1: v += a; break;
        case 2: v += b; break;
        case 3: v += (a + b) >> 1; break;
        case 4: { const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c); v += pa <= pb && pa <= pc ? a : pb <= pc ? b : c; break; }
        default: throw new Error(`bad PNG filter ${ft}`);
      }
      cur[i] = v & 255;
    }
    prev = cur;
  }
  // -> RGB
  const rgb = new Uint8Array(width * height * 3);
  for (let i = 0; i < width * height; i++) {
    if (ctype === 2 || ctype === 6) { rgb[i * 3] = px[i * bpp]; rgb[i * 3 + 1] = px[i * bpp + 1]; rgb[i * 3 + 2] = px[i * bpp + 2]; }
    else if (ctype === 3) { const k = px[i] * 3; rgb[i * 3] = palette[k]; rgb[i * 3 + 1] = palette[k + 1]; rgb[i * 3 + 2] = palette[k + 2]; }
    else { rgb[i * 3] = rgb[i * 3 + 1] = rgb[i * 3 + 2] = px[i * bpp]; }
  }
  return { width, height, rgb };
}

// ---------------------------------------------------------------- Terrarium tiles
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const tiles = new Map();
let downloaded = 0, cached = 0;
async function fetchTile(z, x, y) {
  const k = `${z}/${x}/${y}`;
  if (tiles.has(k)) return tiles.get(k);
  const file = path.join(CACHE, `terrarium_${z}_${x}_${y}.png`);
  let buf = null;
  if (!REFRESH && fs.existsSync(file)) { buf = fs.readFileSync(file); cached++; }
  else {
    const url = `https://s3.amazonaws.com/elevation-tiles-prod/terrarium/${z}/${x}/${y}.png`;
    for (let attempt = 0; attempt < 4 && !buf; attempt++) {
      try {
        const res = await fetch(url, { headers: { 'User-Agent': 'roadster-terrain-baker/1.0' } });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        buf = Buffer.from(await res.arrayBuffer());
        fs.writeFileSync(file, buf);
        downloaded++;
      } catch (e) {
        console.warn(`  tile ${k} attempt ${attempt + 1} failed: ${e.message}`);
        await sleep(1000 * (attempt + 1));
      }
    }
    if (!buf) throw new Error(`tile ${k} unavailable`);
  }
  const img = decodePNG(buf);
  const h = new Float32Array(img.width * img.height);
  for (let i = 0; i < h.length; i++) h[i] = img.rgb[i * 3] * 256 + img.rgb[i * 3 + 1] + img.rgb[i * 3 + 2] / 256 - 32768;
  const t = { w: img.width, hgt: img.height, h };
  tiles.set(k, t);
  return t;
}
const tileXY = (lat, lon, z) => {
  const n = 2 ** z, la = lat * DEG;
  return [((lon + 180) / 360) * n, ((1 - Math.log(Math.tan(la) + 1 / Math.cos(la)) / Math.PI) / 2) * n];
};
async function prefetch(g) {
  const corners = [[g.x0, g.z0], [g.x1, g.z1]].map(([x, z]) => tileXY(unprojLat(z), unprojLon(x), g.zoom));
  const tx0 = Math.floor(Math.min(corners[0][0], corners[1][0])), tx1 = Math.floor(Math.max(corners[0][0], corners[1][0]));
  const ty0 = Math.floor(Math.min(corners[0][1], corners[1][1])), ty1 = Math.floor(Math.max(corners[0][1], corners[1][1]));
  const list = [];
  for (let x = tx0 - 1; x <= tx1 + 1; x++) for (let y = ty0 - 1; y <= ty1 + 1; y++) list.push([x, y]);
  console.log(`  z${g.zoom}: ${list.length} tiles`);
  for (let i = 0; i < list.length; i += 6) await Promise.all(list.slice(i, i + 6).map(([x, y]) => fetchTile(g.zoom, x, y)));
}
// bilinear DEM sample (pixel centres) at lat/lon, zoom z; tiles must be prefetched
function demAt(lat, lon, z) {
  const [fx, fy] = tileXY(lat, lon, z);
  const X = fx * 256 - 0.5, Y = fy * 256 - 0.5;
  const x0 = Math.floor(X), y0 = Math.floor(Y), tx = X - x0, ty = Y - y0;
  const px = (gx, gy) => {
    const t = tiles.get(`${z}/${Math.floor(gx / 256)}/${Math.floor(gy / 256)}`);
    if (!t) return NaN;
    return t.h[(((gy % 256) + 256) % 256) * 256 + (((gx % 256) + 256) % 256)];
  };
  const a = px(x0, y0), b = px(x0 + 1, y0), c = px(x0, y0 + 1), d = px(x0 + 1, y0 + 1);
  return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty;
}

// ---------------------------------------------------------------- OpenTopoData fallback
async function opentopoGrid(g, nx, nz, step) {
  console.log(`  OpenTopoData ned10m fallback, ${Math.ceil(nx / step)}x${Math.ceil(nz / step)} points`);
  const pts = [];
  for (let k = 0; k < nz; k += step) for (let i = 0; i < nx; i += step) pts.push([i, k]);
  const vals = new Map();
  for (let p = 0; p < pts.length; p += 100) {
    const chunk = pts.slice(p, p + 100);
    const loc = chunk.map(([i, k]) => `${unprojLat(g.z0 + k * g.cell).toFixed(6)},${unprojLon(g.x0 + i * g.cell).toFixed(6)}`).join('|');
    const res = await fetch(`https://api.opentopodata.org/v1/ned10m?locations=${loc}`);
    const j = await res.json();
    j.results.forEach((r, q) => vals.set(`${chunk[q][0]},${chunk[q][1]}`, r.elevation ?? BAY_MIN));
    await sleep(1100);
  }
  const out = new Float32Array(nx * nz);
  for (let k = 0; k < nz; k++) for (let i = 0; i < nx; i++) {
    const i0 = Math.floor(i / step) * step, k0 = Math.floor(k / step) * step;
    const i1 = Math.min(i0 + step, Math.floor((nx - 1) / step) * step), k1 = Math.min(k0 + step, Math.floor((nz - 1) / step) * step);
    const tx = i1 > i0 ? (i - i0) / (i1 - i0) : 0, tz = k1 > k0 ? (k - k0) / (k1 - k0) : 0;
    const v = (a, b) => vals.get(`${a},${b}`) ?? BAY_MIN;
    out[k * nx + i] = (v(i0, k0) * (1 - tx) + v(i1, k0) * tx) * (1 - tz) + (v(i0, k1) * (1 - tx) + v(i1, k1) * tx) * tz;
  }
  return out;
}

// ---------------------------------------------------------------- land mask (city.json land + piers, bake frame)
function landMask(g, nx, nz) {
  const m = new Uint8Array(nx * nz);
  for (let k = 0; k < nz; k++) {
    const z = g.z0 + k * g.cell;
    // union of rings (piers overlap land): even-odd per ring, OR-ed together
    for (const r of landRings) {
      const rx = [];
      for (let i = 0, j = r.length - 1; i < r.length; j = i++) {
        const zi = r[i][1], zj = r[j][1];
        if ((zi > z) !== (zj > z)) rx.push(r[i][0] + ((z - zi) / (zj - zi)) * (r[j][0] - r[i][0]));
      }
      rx.sort((a, b) => a - b);
      for (let q = 0; q + 1 < rx.length; q += 2) {
        const i0 = Math.max(0, Math.ceil((rx[q] - g.x0) / g.cell)), i1 = Math.min(nx - 1, Math.floor((rx[q + 1] - g.x0) / g.cell));
        for (let i = i0; i <= i1; i++) m[k * nx + i] = 1;
      }
    }
  }
  return m;
}

// ---------------------------------------------------------------- grid bake
async function bakeGrid(g, label) {
  const nx = Math.round((g.x1 - g.x0) / g.cell) + 1, nz = Math.round((g.z1 - g.z0) / g.cell) + 1;
  console.log(`${label}: ${nx} x ${nz} nodes @ ${g.cell} m, origin (${g.x0}, ${g.z0})`);
  let h;
  let source = 'terrarium';
  try {
    await prefetch(g);
    h = new Float32Array(nx * nz);
    const S = 3; // 3x3 supersampling: box filter over the cell (removes pixel noise, keeps hills)
    for (let k = 0; k < nz; k++) for (let i = 0; i < nx; i++) {
      let sum = 0, n = 0;
      for (let a = 0; a < S; a++) for (let b = 0; b < S; b++) {
        const x = g.x0 + (i + (a + 0.5) / S - 0.5) * g.cell, z = g.z0 + (k + (b + 0.5) / S - 0.5) * g.cell;
        const v = demAt(unprojLat(z), unprojLon(x), g.zoom);
        if (Number.isFinite(v)) { sum += v; n++; }
      }
      h[k * nx + i] = n ? sum / n : BAY_MIN;
    }
  } catch (e) {
    console.warn(`  Terrarium failed (${e.message}); falling back to OpenTopoData`);
    source = 'opentopodata-ned10m';
    h = await opentopoGrid(g, nx, nz, Math.max(1, Math.round(40 / g.cell)));
  }
  // land / water treatment
  const land = landMask(g, nx, nz);
  const N = nx * nz;
  const known = new Uint8Array(N); // 1 = trusted land value
  for (let i = 0; i < N; i++) {
    if (h[i] < BAY_MIN) h[i] = BAY_MIN;
    if (land[i] && h[i] >= 0.5) known[i] = 1;
  }
  // multi-source BFS: unknown land nodes and coastal water nodes copy the nearest trusted land value
  const dist = new Int32Array(N).fill(-1), q = new Int32Array(N);
  let qh = 0, qt = 0;
  for (let i = 0; i < N; i++) if (known[i]) { dist[i] = 0; q[qt++] = i; }
  const COAST_N = Math.max(3, Math.round(40 / g.cell));
  const LAND_FILL_N = Math.round(1500 / g.cell);
  while (qh < qt) {
    const c = q[qh++], ci = c % nx, ck = (c / nx) | 0, d = dist[c] + 1;
    for (const [di, dk] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
      const ii = ci + di, kk = ck + dk;
      if (ii < 0 || kk < 0 || ii >= nx || kk >= nz) continue;
      const j = kk * nx + ii;
      if (dist[j] >= 0) continue;
      if (land[j] ? d > LAND_FILL_N : d > COAST_N) continue;
      dist[j] = d; h[j] = h[c]; q[qt++] = j;
    }
  }
  for (let i = 0; i < N; i++) {
    if (dist[i] < 0) h[i] = land[i] ? Math.max(h[i], 0.5) : BAY_MIN;
  }
  // light 3x3 binomial smoothing on land/coastal nodes (bay stays flat)
  const sm = Float32Array.from(h);
  for (let k = 1; k < nz - 1; k++) for (let i = 1; i < nx - 1; i++) {
    const c = k * nx + i;
    if (dist[c] < 0) continue;
    let s = 0, w = 0;
    for (let dk = -1; dk <= 1; dk++) for (let di = -1; di <= 1; di++) {
      const j = c + dk * nx + di;
      if (dist[j] < 0) continue;
      const ww = (di ? 1 : 2) * (dk ? 1 : 2);
      s += h[j] * ww; w += ww;
    }
    sm[c] = s / w;
  }
  const q16 = new Array(N);
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < N; i++) { const v = Math.max(-3276, Math.min(3276, sm[i])); q16[i] = Math.round(v * 10); mn = Math.min(mn, v); mx = Math.max(mx, v); }
  console.log(`  ${label}: min ${mn.toFixed(1)} m  max ${mx.toFixed(1)} m  (${source})`);
  return { origin: [g.x0, g.z0], cell: g.cell, nx, nz, h: q16, source, _f: sm };
}

function sampleGrid(G, x, z) {
  const fx = (x - G.origin[0]) / G.cell, fz = (z - G.origin[1]) / G.cell;
  const i = Math.max(0, Math.min(G.nx - 2, Math.floor(fx))), k = Math.max(0, Math.min(G.nz - 2, Math.floor(fz)));
  const tx = Math.max(0, Math.min(1, fx - i)), tz = Math.max(0, Math.min(1, fz - k));
  const v = (a, b) => G.h[b * G.nx + a] / 10;
  return (v(i, k) * (1 - tx) + v(i + 1, k) * tx) * (1 - tz) + (v(i, k + 1) * (1 - tx) + v(i + 1, k + 1) * tx) * tz;
}

// ---------------------------------------------------------------- main
function tileCount(g) {
  const corners = [[g.x0, g.z0], [g.x1, g.z1]].map(([x, z]) => tileXY(unprojLat(z), unprojLon(x), g.zoom));
  const nx = Math.floor(Math.max(corners[0][0], corners[1][0])) - Math.floor(Math.min(corners[0][0], corners[1][0])) + 3;
  const ny = Math.floor(Math.max(corners[0][1], corners[1][1])) - Math.floor(Math.min(corners[0][1], corners[1][1])) + 3;
  return nx * ny;
}
/** Bake terrain.json. opts: SPEC keys in camelCase (city, lat, lon, outer, fine, fineCell, ..., checks (file or array), out, cache, refresh, dryRun). */
export async function bakeTerrain(opts = {}) {
  configure(opts);
  const checks = Array.isArray(opts.checks) ? opts.checks : opts.checks ? JSON.parse(fs.readFileSync(opts.checks, 'utf8')) : [];
  if (opts.dryRun) {
    console.log(`terrain plan: centre ${CENTER.lat}, ${CENTER.lon} (city ${CITY || '-'}${city.center ? '' : ', no center in it'}), outer square ${OUTER} m, ${landRings.length} land / pier rings`);
    for (const [label, g] of [['fine', FINE], ['outer', COARSE]]) {
      const nx = Math.round((g.x1 - g.x0) / g.cell) + 1, nz = Math.round((g.z1 - g.z0) / g.cell) + 1;
      console.log(`  ${label}: ${nx} x ${nz} nodes @ ${g.cell} m, origin (${g.x0}, ${g.z0}), z${g.zoom}: ${tileCount(g)} tiles`);
    }
    console.log(`  bay min ${BAY_MIN} m, ${checks.length} height checks\n  out ${OUT}\n  cache ${CACHE}${REFRESH ? ' (refresh)' : ''}\n  dry run: no requests made, nothing written`);
    return { dryRun: true, out: OUT, cache: CACHE };
  }
  fs.mkdirSync(CACHE, { recursive: true });
  const t0 = Date.now();
  const fine = await bakeGrid(FINE, 'fine');
  const outer = await bakeGrid(COARSE, 'outer');
  console.log(`  tiles: ${downloaded} downloaded, ${cached} from cache`);
  delete fine._f; delete outer._f;
  const out = {
    v: 1,
    source: fine.source === 'terrarium' ? 'AWS Terrain Tiles (Mapzen Terrarium, USGS 3DEP/NED) z15 fine / z13 outer' : fine.source,
    generated: new Date().toISOString(),
    frame: 'bake (city.json coordinates; World applies world.offset)',
    center: CENTER,
    datum: 'MSL metres',
    units: 'h = Int16 decimetres, row-major (z rows, then x); node (i,k) at origin + [i,k]*cell',
    origin: fine.origin, cell: fine.cell, nx: fine.nx, nz: fine.nz, h: fine.h,
    outer: { origin: outer.origin, cell: outer.cell, nx: outer.nx, nz: outer.nz, h: outer.h },
  };
  const json = JSON.stringify(out);
  fs.writeFileSync(OUT, json);
  console.log(`wrote ${path.relative(ROOT, OUT)}  ${(json.length / 1024 / 1024).toFixed(2)} MB  in ${((Date.now() - t0) / 1000).toFixed(1)} s`);

  // height check at known places (opts.checks: [[name, lat, lon], ...])
  for (const [name, lat, lon] of checks) {
    const [x, z] = proj(lat, lon);
    const inFine = x >= fine.origin[0] && z >= fine.origin[1] && x <= fine.origin[0] + (fine.nx - 1) * fine.cell && z <= fine.origin[1] + (fine.nz - 1) * fine.cell;
    const v = inFine ? sampleGrid(fine, x, z) : sampleGrid(outer, x, z);
    console.log(`  ${name.padEnd(46)} x ${x.toFixed(0).padStart(6)} z ${z.toFixed(0).padStart(6)}  ${v.toFixed(1).padStart(6)} m ${inFine ? '' : '(outer)'}`);
  }
  return { out: OUT };
}

runCli(import.meta.url, 'engine/tools/bake-terrain.mjs', 'bake Terrarium elevation tiles into a terrain.json', SPEC, bakeTerrain);
