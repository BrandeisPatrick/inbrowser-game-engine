#!/usr/bin/env node
// Bake real OpenStreetMap data for a city district (any lat / lon; default: downtown San Francisco) into a
// city.json (the MapWorld map contract, engine/src/world/MapData.js).
//
//   node engine/tools/bake-map.mjs --lat 37.792 --lon -122.3965 --size 2200 --outer 9000 --out public/map/city.json
//   node engine/tools/bake-map.mjs --dry-run   (print the plan: bboxes, Overpass jobs, paths; no network, no files)
//   node engine/tools/bake-map.mjs --help      (all flags; cached raw responses in --cache are reused unless --refresh)
//   export async function bakeMap(opts)        (same options, camelCase; used by the game's tools/bake-map.mjs)
//
// Data source: Overpass API (several mirrors, retried with exponential backoff, chunked by feature
// type and, for buildings, spatially). If every Overpass mirror keeps failing, falls back to the OSM
// editing API (/api/0.6/map.json in small tiles) which returns the same primitives.
//
// Output coordinate frame: metres around CENTER, x = east, z = SOUTH (so Three.js -Z is north).
// All coordinates rounded to 0.1 m. Consumer: engine/src/world/MapData.js + MapWorld.js.

import fs from 'node:fs';
import path from 'node:path';
import { runCli } from './cli.mjs';

/** CLI flags / bakeMap options (defaults: the Roadster San Francisco bake). */
export const SPEC = {
  lat: { type: 'number', def: 37.792, help: 'centre latitude' },
  lon: { type: 'number', def: -122.3965, help: 'centre longitude' },
  size: { type: 'number', def: 2200, help: 'detailed city square edge (m)' },
  outer: { type: 'number', def: 9000, help: 'coastline / land square edge (m)' },
  'bridge-ext': { type: 'numbers', def: [3200, 2200], help: 'extra bbox east,north of the city for motorway bridges (0,0 = none)' },
  out: { type: 'string', def: 'public/map/city.json', help: 'output city.json' },
  cache: { type: 'string', def: '.osm-cache', help: 'raw response cache dir' },
  ua: { type: 'string', def: 'roadster-flying-car-map-baker/1.0 (one-off offline bake for a three.js game)', help: 'HTTP User-Agent' },
  refresh: { type: 'bool', def: false, help: 'ignore the cache' },
  'dry-run': { type: 'bool', def: false, help: 'print the plan only (no network, no files)' },
};

// configuration (set by configure() from the options; read by the functions below at call time)
let ROOT, OUT, CACHE, REFRESH, CENTER, SIZE, OUTER, BRIDGE_EXT, UA, COS0, H, CITY, CITY_BB, OH, OUTER_BB, BRIDGE_BB;
function configure(o = {}) {
  const d = (k) => (o[k] !== undefined ? o[k] : SPEC[k.replace(/[A-Z]/g, (c) => '-' + c.toLowerCase())].def);
  ROOT = process.cwd();
  OUT = path.resolve(d('out')); CACHE = path.resolve(d('cache')); REFRESH = !!d('refresh');
  CENTER = { lat: d('lat'), lon: d('lon') };
  SIZE = d('size'); OUTER = d('outer');
  const be = d('bridgeExt'); BRIDGE_EXT = { east: be[0] || 0, north: be[1] || 0 };
  UA = d('ua');
  COS0 = Math.cos(CENTER.lat * DEG);
  H = SIZE / 2;
  CITY = { minX: -H, maxX: H, minZ: -H, maxZ: H };
  CITY_BB = bboxFor(-H, -H, H, H);
  OH = OUTER / 2;
  OUTER_BB = bboxFor(-OH, -OH, OH, OH);
  BRIDGE_BB = bboxFor(H - 200, -BRIDGE_EXT.north, H + BRIDGE_EXT.east, H);
}

const MIRRORS = [
  'https://overpass-api.de/api/interpreter',
  'https://overpass.kumi.systems/api/interpreter',
  'https://overpass.private.coffee/api/interpreter',
  'https://maps.mail.ru/osm/tools/overpass/api/interpreter',
];

// ---------------------------------------------------------------- projection
const R = 6378137;
const DEG = Math.PI / 180;
const proj = (lat, lon) => [(lon - CENTER.lon) * DEG * R * COS0, -(lat - CENTER.lat) * DEG * R];
const unprojLat = (z) => CENTER.lat - z / (DEG * R);
const unprojLon = (x) => CENTER.lon + x / (DEG * R * COS0);
function bboxFor(minX, minZ, maxX, maxZ) { // returns overpass (s,w,n,e)
  return [unprojLat(maxZ), unprojLon(minX), unprojLat(minZ), unprojLon(maxX)].map((v) => +v.toFixed(6));
}
const bbStr = (b) => b.join(',');

// ---------------------------------------------------------------- fetching
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function overpass(name, body, { maxRounds = 7 } = {}) {
  let qh = 0; for (let i = 0; i < body.length; i++) qh = (Math.imul(qh, 31) + body.charCodeAt(i)) | 0;
  const cacheFile = path.join(CACHE, `${name}_${(qh >>> 0).toString(36)}.json`);
  if (!REFRESH && fs.existsSync(cacheFile)) {
    const j = JSON.parse(fs.readFileSync(cacheFile, 'utf8'));
    console.log(`  [cache] ${name}: ${j.elements.length} elements`);
    return j;
  }
  const q = `[out:json][timeout:180][maxsize:536870912];\n${body}`;
  let wait = 4000;
  for (let round = 0; round < maxRounds; round++) {
    for (const url of MIRRORS) {
      const t0 = Date.now();
      try {
        const ctl = new AbortController();
        const timer = setTimeout(() => ctl.abort(), 200000);
        const res = await fetch(url, {
          method: 'POST',
          headers: { 'User-Agent': UA, 'Content-Type': 'application/x-www-form-urlencoded', Accept: 'application/json' },
          body: 'data=' + encodeURIComponent(q),
          signal: ctl.signal,
        });
        clearTimeout(timer);
        const text = await res.text();
        if (!res.ok) throw new Error(`HTTP ${res.status}: ${text.slice(0, 80).replace(/\s+/g, ' ')}`);
        let j;
        try { j = JSON.parse(text); } catch { throw new Error('non-JSON response: ' + text.slice(0, 80).replace(/\s+/g, ' ')); }
        if (j.remark && /error|timed out|out of memory/i.test(j.remark)) throw new Error('remark: ' + j.remark);
        if (!Array.isArray(j.elements)) throw new Error('no elements');
        fs.writeFileSync(cacheFile, JSON.stringify(j));
        console.log(`  ${name}: ${j.elements.length} elements from ${new URL(url).host} in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
        return j;
      } catch (e) {
        console.log(`  ${name}: ${new URL(url).host} failed (${e.name === 'AbortError' ? 'timeout' : e.message})`);
      }
    }
    console.log(`  ${name}: round ${round + 1}/${maxRounds} failed, waiting ${(wait / 1000) | 0}s`);
    await sleep(wait);
    wait = Math.min(wait * 2, 90000);
  }
  return null;
}

// ---- Fallback: OSM editing API in small tiles, converted to Overpass `out geom` shaped elements.
async function osmApiGet(url, cacheName) {
  const cacheFile = cacheName && path.join(CACHE, cacheName + '.json');
  if (cacheFile && !REFRESH && fs.existsSync(cacheFile)) return JSON.parse(fs.readFileSync(cacheFile, 'utf8'));
  let wait = 3000;
  for (let i = 0; i < 6; i++) {
    try {
      const res = await fetch(url, { headers: { 'User-Agent': UA, Accept: 'application/json' } });
      const text = await res.text();
      if (!res.ok) throw new Error(`HTTP ${res.status} ${text.slice(0, 80)}`);
      const j = JSON.parse(text);
      if (cacheFile) fs.writeFileSync(cacheFile, text);
      return j;
    } catch (e) {
      console.log(`  osm api ${url.slice(0, 90)} failed (${e.message}); retry in ${wait / 1000}s`);
      await sleep(wait); wait *= 2;
    }
  }
  return null;
}

async function osmApiFallback() {
  console.log('All Overpass mirrors failed -> falling back to the OSM API (tiled map calls).');
  const N = 8; // 8x8 tiles of ~212 m
  const nodes = new Map(), ways = new Map(), rels = new Map();
  const ingest = (j) => {
    for (const e of j.elements) {
      if (e.type === 'node') nodes.set(e.id, e);
      else if (e.type === 'way') ways.set(e.id, e);
      else if (e.type === 'relation') rels.set(e.id, e);
    }
  };
  for (let i = 0; i < N; i++) for (let k = 0; k < N; k++) {
    const x0 = -H + (SIZE * i) / N, x1 = -H + (SIZE * (i + 1)) / N;
    const z0 = -H + (SIZE * k) / N, z1 = -H + (SIZE * (k + 1)) / N;
    const [s, w, n, e] = bboxFor(x0, z0, x1, z1);
    const j = await osmApiGet(`https://api.openstreetmap.org/api/0.6/map.json?bbox=${w},${s},${e},${n}`, `osmapi_${i}_${k}`);
    if (!j) throw new Error('OSM API tile failed');
    ingest(j);
    process.stdout.write(`\r  tiles ${i * N + k + 1}/${N * N}  nodes ${nodes.size} ways ${ways.size} rels ${rels.size}   `);
    await sleep(400);
  }
  console.log();
  // complete building / water multipolygon relations and coastline ways
  for (const r of rels.values()) {
    const t = r.tags || {};
    if (t.type !== 'multipolygon') continue;
    if (!(t.building || t['building:part'] || t.natural === 'water' || t.leisure || t.amenity === 'parking' || t.man_made === 'pier')) continue;
    if (r.members.every((m) => m.type !== 'way' || (ways.has(m.ref) && ways.get(m.ref).nodes.every((id) => nodes.has(id))))) continue;
    const j = await osmApiGet(`https://api.openstreetmap.org/api/0.6/relation/${r.id}/full.json`, `osmapi_rel_${r.id}`);
    if (j) ingest(j);
    await sleep(300);
  }
  for (const w of [...ways.values()]) {
    if ((w.tags || {}).natural !== 'coastline') continue;
    if (w.nodes.every((id) => nodes.has(id))) continue;
    const j = await osmApiGet(`https://api.openstreetmap.org/api/0.6/way/${w.id}/full.json`, `osmapi_way_${w.id}`);
    if (j) ingest(j);
    await sleep(300);
  }
  const geomOf = (w) => w.nodes.map((id) => nodes.get(id)).filter(Boolean).map((n) => ({ lat: n.lat, lon: n.lon }));
  const elements = [];
  for (const n of nodes.values()) if (n.tags) elements.push(n);
  for (const w of ways.values()) elements.push({ type: 'way', id: w.id, tags: w.tags || {}, nodes: w.nodes, geometry: geomOf(w) });
  for (const r of rels.values()) {
    elements.push({
      type: 'relation', id: r.id, tags: r.tags || {},
      members: r.members.map((m) => (m.type === 'way' && ways.has(m.ref) ? { type: 'way', ref: m.ref, role: m.role, geometry: geomOf(ways.get(m.ref)) } : { type: m.type, ref: m.ref, role: m.role })),
    });
  }
  return elements;
}

// ---------------------------------------------------------------- geometry helpers
const r1 = (v) => Math.round(v * 10) / 10;
function signedArea(p) {
  let a = 0;
  for (let i = 0, n = p.length; i < n; i++) { const q = p[i], s = p[(i + 1) % n]; a += q[0] * s[1] - s[0] * q[1]; }
  return a / 2;
}
function dpSimplify(pts, tol, keep) { // open polyline Douglas-Peucker; keep: optional Set of indices to force
  if (pts.length <= 2) return pts.map((_, i) => i);
  const mark = new Uint8Array(pts.length); mark[0] = mark[pts.length - 1] = 1;
  if (keep) for (const i of keep) mark[i] = 1;
  const stack = [];
  // split at forced indices first
  let prev = 0;
  for (let i = 1; i < pts.length; i++) if (mark[i]) { stack.push([prev, i]); prev = i; }
  const tol2 = tol * tol;
  while (stack.length) {
    const [a, b] = stack.pop();
    if (b <= a + 1) continue;
    const [ax, az] = pts[a], [bx, bz] = pts[b];
    const dx = bx - ax, dz = bz - az, L2 = dx * dx + dz * dz;
    let best = -1, bi = -1;
    for (let i = a + 1; i < b; i++) {
      const [px, pz] = pts[i];
      let t = L2 > 0 ? ((px - ax) * dx + (pz - az) * dz) / L2 : 0; t = Math.max(0, Math.min(1, t));
      const ex = ax + t * dx - px, ez = az + t * dz - pz, d = ex * ex + ez * ez;
      if (d > best) { best = d; bi = i; }
    }
    if (best > tol2) { mark[bi] = 1; stack.push([a, bi], [bi, b]); }
  }
  const out = [];
  for (let i = 0; i < pts.length; i++) if (mark[i]) out.push(i);
  return out;
}
function cleanRing(ring, tol, minArea) {
  let p = ring.map(([x, z]) => [x, z]);
  if (p.length > 1) { const a = p[0], b = p[p.length - 1]; if (Math.abs(a[0] - b[0]) < 1e-6 && Math.abs(a[1] - b[1]) < 1e-6) p.pop(); }
  p = p.filter((q, i) => { const n = p[(i + 1) % p.length]; return Math.hypot(q[0] - n[0], q[1] - n[1]) > 0.05; });
  if (p.length < 3) return null;
  if (tol > 0 && p.length > 4) {
    const closed = p.concat([p[0]]);
    // start the simplification at the point farthest from p[0] too, so the ring is split in two halves
    let far = 0, fd = -1;
    for (let i = 0; i < p.length; i++) { const d = (p[i][0] - p[0][0]) ** 2 + (p[i][1] - p[0][1]) ** 2; if (d > fd) { fd = d; far = i; } }
    const idx = dpSimplify(closed, tol, new Set([far]));
    p = idx.slice(0, -1).map((i) => closed[i]);
  }
  // remove (near) collinear points
  for (let pass = 0; pass < 2 && p.length > 3; pass++) {
    p = p.filter((q, i) => {
      const a = p[(i + p.length - 1) % p.length], b = p[(i + 1) % p.length];
      const cr = (q[0] - a[0]) * (b[1] - a[1]) - (q[1] - a[1]) * (b[0] - a[0]);
      const L = Math.hypot(b[0] - a[0], b[1] - a[1]);
      return L < 1e-6 ? false : Math.abs(cr) / L > 0.04;
    });
  }
  if (p.length < 3) return null;
  const A = signedArea(p);
  if (!isFinite(A) || Math.abs(A) < minArea) return null;
  return p.map(([x, z]) => [r1(x), r1(z)]);
}
function pip(x, z, ring) {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, zi] = ring[i], [xj, zj] = ring[j];
    if ((zi > z) !== (zj > z) && x < ((xj - xi) * (z - zi)) / (zj - zi) + xi) inside = !inside;
  }
  return inside;
}
function centroid(ring) {
  let a = 0, cx = 0, cz = 0;
  for (let i = 0, n = ring.length; i < n; i++) {
    const [x0, z0] = ring[i], [x1, z1] = ring[(i + 1) % n];
    const f = x0 * z1 - x1 * z0; a += f; cx += (x0 + x1) * f; cz += (z0 + z1) * f;
  }
  if (Math.abs(a) < 1e-9) { let sx = 0, sz = 0; for (const [x, z] of ring) { sx += x; sz += z; } return [sx / ring.length, sz / ring.length]; }
  return [cx / (3 * a), cz / (3 * a)];
}
const inRect = (x, z, r, m = 0) => x >= r.minX - m && x <= r.maxX + m && z >= r.minZ - m && z <= r.maxZ + m;

// Clip a polyline (array of [x,z] with parallel flags array) to a rectangle. Returns pieces [{pts, flags, inStart, inEnd}]
function clipPolyline(pts, flags, rc) {
  const pieces = [];
  let cur = null;
  const inside = (p) => p[0] >= rc.minX && p[0] <= rc.maxX && p[1] >= rc.minZ && p[1] <= rc.maxZ;
  for (let i = 0; i < pts.length - 1; i++) {
    const a = pts[i], b = pts[i + 1];
    // Liang-Barsky
    let t0 = 0, t1 = 1; const dx = b[0] - a[0], dz = b[1] - a[1];
    const P = [-dx, dx, -dz, dz], Q = [a[0] - rc.minX, rc.maxX - a[0], a[1] - rc.minZ, rc.maxZ - a[1]];
    let ok = true;
    for (let k = 0; k < 4; k++) {
      if (P[k] === 0) { if (Q[k] < 0) { ok = false; break; } }
      else { const t = Q[k] / P[k]; if (P[k] < 0) { if (t > t1) { ok = false; break; } if (t > t0) t0 = t; } else { if (t < t0) { ok = false; break; } if (t < t1) t1 = t; } }
    }
    if (!ok) { if (cur) { pieces.push(cur); cur = null; } continue; }
    const pa = t0 > 0 ? [a[0] + dx * t0, a[1] + dz * t0] : a;
    const pb = t1 < 1 ? [a[0] + dx * t1, a[1] + dz * t1] : b;
    if (!cur) cur = { pts: [pa], flags: [t0 > 0 ? 0 : flags[i]], startClipped: t0 > 0 || !inside(a), endClipped: false, i0: i };
    cur.pts.push(pb); cur.flags.push(t1 < 1 ? 0 : flags[i + 1]);
    cur.endClipped = t1 < 1;
    cur.i1 = i + 1;
    if (t1 < 1) { pieces.push(cur); cur = null; }
  }
  if (cur) pieces.push(cur);
  return pieces.filter((p) => p.pts.length >= 2);
}

// ---------------------------------------------------------------- tag helpers
function parseLen(v) {
  if (v == null) return NaN;
  const s = String(v).trim().replace(',', '.');
  let m = s.match(/^(-?\d+(?:\.\d+)?)\s*'\s*(?:(\d+(?:\.\d+)?)\s*")?$/);
  if (m) return parseFloat(m[1]) * 0.3048 + (m[2] ? parseFloat(m[2]) * 0.0254 : 0);
  m = s.match(/^(-?\d+(?:\.\d+)?)\s*(m|meters?|metres?|ft|feet|foot)?\b/i);
  if (!m) return NaN;
  const n = parseFloat(m[1]);
  return m[2] && /^f/i.test(m[2]) ? n * 0.3048 : n;
}
function hash32(n) { n = (n ^ 61) ^ (n >>> 16); n = (n + (n << 3)) | 0; n ^= n >>> 4; n = Math.imul(n, 0x27d4eb2d); n ^= n >>> 15; return (n >>> 0) / 4294967296; }

function buildingHeight(tags, id, area) {
  let h = parseLen(tags.height);
  let lv = parseFloat(tags['building:levels']);
  if (!isFinite(lv) || lv <= 0) lv = 0;
  if (!(h > 0)) {
    if (lv > 0) h = lv * 3.2 + 1;
    else {
      const t = tags.building || tags['building:part'] || 'yes';
      const rnd = hash32(id | 0);
      switch (t) {
        case 'house': case 'detached': case 'semidetached_house': case 'terrace': h = 7 + rnd * 3; break;
        case 'residential': case 'apartments': h = 10 + rnd * 8; break;
        case 'garage': case 'garages': case 'shed': case 'hut': case 'kiosk': case 'roof': case 'carport': h = 3 + rnd; break;
        case 'retail': case 'supermarket': h = 6 + rnd * 4; break;
        case 'industrial': case 'warehouse': h = 9 + rnd * 4; break;
        case 'church': case 'cathedral': h = 15 + rnd * 8; break;
        case 'parking': h = 12 + rnd * 8; break;
        case 'hotel': h = 25 + rnd * 25; break;
        case 'office': case 'commercial': h = 20 + rnd * 20; break;
        case 'train_station': case 'transportation': h = 12 + rnd * 6; break;
        default: h = area > 1500 ? 20 + rnd * 25 : area > 400 ? 12 + rnd * 18 : 8 + rnd * 8;
      }
    }
  }
  return { h, lv };
}

const DRIVABLE = new Set(['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'unclassified', 'residential', 'living_street',
  'motorway_link', 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link', 'service', 'road', 'busway']);
const MAJOR = new Set([...DRIVABLE].filter((c) => c !== 'service' && c !== 'busway'));
const MINOR_KEEP = new Set(['pedestrian', 'footway', 'path', 'cycleway', 'track', 'steps']);
const isYes = (v) => v != null && v !== 'no' && v !== '0' && v !== 'false';

// ---------------------------------------------------------------- multipolygon assembly
function joinRings(segs) { // segs: arrays of [x,z]; returns closed rings (without repeated last point)
  const rings = [];
  const open = segs.filter((s) => s.length >= 2).map((s) => s.slice());
  const same = (a, b) => Math.abs(a[0] - b[0]) < 0.01 && Math.abs(a[1] - b[1]) < 0.01;
  while (open.length) {
    let cur = open.shift();
    let guard = 0;
    while (!same(cur[0], cur[cur.length - 1]) && guard++ < 10000) {
      const end = cur[cur.length - 1];
      let k = open.findIndex((s) => same(s[0], end) || same(s[s.length - 1], end));
      if (k < 0) break;
      let s = open.splice(k, 1)[0];
      if (!same(s[0], end)) s = s.reverse();
      cur = cur.concat(s.slice(1));
    }
    if (cur.length >= 4) rings.push(cur); // unclosed rings are closed implicitly
  }
  return rings;
}
function elementPolygons(e) { // returns [{outer, holes:[...]}] in projected coords
  if (e.type === 'way') {
    if (!e.geometry || e.geometry.length < 4) return [];
    const g = e.geometry.filter(Boolean);
    const a = g[0], b = g[g.length - 1];
    if (a.lat !== b.lat || a.lon !== b.lon) return [];
    return [{ outer: g.map((n) => proj(n.lat, n.lon)), holes: [] }];
  }
  if (e.type === 'relation') {
    const outers = [], inners = [];
    for (const m of e.members || []) {
      if (m.type !== 'way' || !m.geometry) continue;
      const pts = m.geometry.filter(Boolean).map((n) => proj(n.lat, n.lon));
      if (m.role === 'inner') inners.push(pts); else outers.push(pts);
    }
    const O = joinRings(outers), I = joinRings(inners);
    const polys = O.map((o) => ({ outer: o, holes: [] }));
    for (const h of I) {
      const c = h[0];
      const host = polys.find((p) => pip(c[0], c[1], p.outer));
      if (host) host.holes.push(h);
    }
    return polys;
  }
  return [];
}

// ---------------------------------------------------------------- coastline -> land polygons
function coastToLand(coastWays, rc) {
  // merge into chains
  const chains = joinOpenChains(coastWays);
  // reverse: OSM coastline has land on the LEFT; after reversing, land is on the RIGHT, which in the
  // x-right/z-down (map seen from above) frame means land rings run clockwise.
  const pieces = [], rings = [];
  for (const ch of chains) {
    const c = ch.slice().reverse();
    const closed = Math.hypot(c[0][0] - c[c.length - 1][0], c[0][1] - c[c.length - 1][1]) < 0.01;
    if (closed && c.every((p) => inRect(p[0], p[1], rc))) { rings.push(c.slice(0, -1)); continue; }
    for (const pc of clipPolyline(c, c.map(() => 0), rc)) pieces.push(snapPiece(pc.pts, rc));
  }
  const W = rc.maxX - rc.minX, Hh = rc.maxZ - rc.minZ, P = 2 * (W + Hh);
  const per = ([x, z]) => {
    const e = 1e-6;
    if (Math.abs(z - rc.minZ) < e) return x - rc.minX;
    if (Math.abs(x - rc.maxX) < e) return W + (z - rc.minZ);
    if (Math.abs(z - rc.maxZ) < e) return W + Hh + (rc.maxX - x);
    return 2 * W + Hh + (rc.maxZ - z);
  };
  const corners = [[rc.maxX, rc.minZ, W], [rc.maxX, rc.maxZ, W + Hh], [rc.minX, rc.maxZ, 2 * W + Hh], [rc.minX, rc.minZ, P]];
  const polys = [];
  const used = new Array(pieces.length).fill(false);
  for (let s = 0; s < pieces.length; s++) {
    if (used[s]) continue;
    const poly = [];
    let k = s, guard = 0;
    while (guard++ < 1000) {
      used[k] = true;
      poly.push(...pieces[k]);
      const pOut = per(pieces[k][pieces[k].length - 1]);
      // next entry clockwise
      let best = -1, bd = Infinity;
      for (let j = 0; j < pieces.length; j++) {
        if (used[j] && j !== s) continue;
        let d = per(pieces[j][0]) - pOut; if (d < 0) d += P;
        if (d < bd) { bd = d; best = j; }
      }
      if (best < 0) break;
      // corners passed between pOut and pOut+bd
      const cs = [];
      for (const [cx, cz, cp] of corners) {
        for (const base of [0, P]) { let d = cp + base - pOut; if (d > 1e-6 && d < bd - 1e-6) cs.push([d, [cx, cz]]); }
      }
      cs.sort((a, b) => a[0] - b[0]);
      for (const [, c] of cs) poly.push(c);
      if (best === s) break;
      k = best;
    }
    polys.push(poly);
  }
  if (!pieces.length) polys.push([[rc.minX, rc.minZ], [rc.maxX, rc.minZ], [rc.maxX, rc.maxZ], [rc.minX, rc.maxZ]]);
  for (const r of rings) if (signedArea(r) > 0) polys.push(r); // clockwise on screen == positive shoelace in x/z(down)
  return polys;
}
function snapPiece(pts, rc) { // make sure first/last point lie exactly on the rect boundary
  const snap = ([x, z]) => {
    const d = [Math.abs(x - rc.minX), Math.abs(rc.maxX - x), Math.abs(z - rc.minZ), Math.abs(rc.maxZ - z)];
    const k = d.indexOf(Math.min(...d));
    return k === 0 ? [rc.minX, z] : k === 1 ? [rc.maxX, z] : k === 2 ? [x, rc.minZ] : [x, rc.maxZ];
  };
  const p = pts.slice(); p[0] = snap(p[0]); p[p.length - 1] = snap(p[p.length - 1]);
  return p;
}
function joinOpenChains(segs) {
  const out = [];
  const open = segs.map((s) => s.slice());
  const same = (a, b) => Math.abs(a[0] - b[0]) < 0.01 && Math.abs(a[1] - b[1]) < 0.01;
  while (open.length) {
    let cur = open.shift();
    let changed = true;
    while (changed) {
      changed = false;
      for (let k = 0; k < open.length; k++) {
        const s = open[k];
        if (same(s[0], cur[cur.length - 1])) { cur = cur.concat(s.slice(1)); open.splice(k, 1); changed = true; break; }
        if (same(s[s.length - 1], cur[0])) { cur = s.concat(cur.slice(1)); open.splice(k, 1); changed = true; break; }
      }
    }
    out.push(cur);
  }
  return out;
}

// ---------------------------------------------------------------- main
/** Bake city.json. opts: SPEC keys in camelCase (lat, lon, size, outer, bridgeExt, out, cache, ua, refresh, dryRun). */
export async function bakeMap(opts = {}) {
  configure(opts);
  console.log(`Baking ${SIZE} m square around ${CENTER.lat},${CENTER.lon}  bbox(s,w,n,e)=${bbStr(CITY_BB)}`);
  const halves = [];
  for (let i = 0; i < 2; i++) for (let k = 0; k < 2; k++) halves.push(bboxFor(-H + i * H, -H + k * H, i * H, -H + (k + 1) * H));
  const jobs = [
    ...halves.map((bb, i) => [`buildings_${i}`, `(way["building"](${bbStr(bb)});relation["building"](${bbStr(bb)}););out body geom;`]),
    ['parts', `(way["building:part"](${bbStr(CITY_BB)});relation["building:part"](${bbStr(CITY_BB)}););out body geom;`],
    ['roads', `(way["highway"](${bbStr(CITY_BB)}););out body geom;`],
    ...(BRIDGE_EXT.east > 0 || BRIDGE_EXT.north > 0 ? [['bridge_ext', `(way["highway"~"^motorway"]["bridge"](${bbStr(BRIDGE_BB)}););out body geom;`]] : []),
    ['points', `(node["highway"~"^(traffic_signals|street_lamp)$"](${bbStr(CITY_BB)});node["natural"="tree"](${bbStr(CITY_BB)}););out body;`],
    ['water', `(way["natural"="water"](${bbStr(CITY_BB)});relation["natural"="water"](${bbStr(CITY_BB)});way["man_made"~"^(pier|breakwater|groyne)$"](${bbStr(CITY_BB)});relation["man_made"="pier"](${bbStr(CITY_BB)}););out body geom;`],
    ['coast', `(way["natural"="coastline"](${bbStr(OUTER_BB)}););out body geom;`],
    ['green', `(way["leisure"~"^(park|garden|playground|pitch|dog_park|common)$"](${bbStr(CITY_BB)});relation["leisure"~"^(park|garden)$"](${bbStr(CITY_BB)});way["landuse"~"^(grass|recreation_ground|village_green|forest|flowerbed|meadow)$"](${bbStr(CITY_BB)});way["natural"~"^(wood|scrub|grassland)$"](${bbStr(CITY_BB)}););out body geom;`],
    ['rail', `(way["railway"~"^(rail|light_rail|tram|narrow_gauge|funicular|subway)$"](${bbStr(CITY_BB)}););out body geom;`],
    ['parking', `(way["amenity"="parking"](${bbStr(CITY_BB)});relation["amenity"="parking"](${bbStr(CITY_BB)});way["place"="square"](${bbStr(CITY_BB)}););out body geom;`],
  ];
  if (opts.dryRun) {
    console.log(`  centre ${CENTER.lat}, ${CENTER.lon}  city ${SIZE} m  outer ${OUTER} m  bridge ext east ${BRIDGE_EXT.east} north ${BRIDGE_EXT.north} m`);
    console.log(`  bbox(s,w,n,e) city ${bbStr(CITY_BB)} | outer ${bbStr(OUTER_BB)}${BRIDGE_EXT.east > 0 || BRIDGE_EXT.north > 0 ? ` | bridge ${bbStr(BRIDGE_BB)}` : ''}`);
    console.log(`  Overpass jobs (${jobs.length}): ${jobs.map((j) => j[0]).join(', ')}`);
    console.log(`  out ${OUT}\n  cache ${CACHE}${REFRESH ? ' (refresh)' : ''}\n  user agent ${UA}\n  dry run: no requests made, nothing written`);
    return { dryRun: true, jobs: jobs.map((j) => j[0]), out: OUT, cache: CACHE };
  }
  fs.mkdirSync(CACHE, { recursive: true });
  let elements = [];
  let failed = false;
  for (const [name, q] of jobs) {
    const j = await overpass(name, q);
    if (!j) { failed = true; console.log(`Overpass failed for ${name}`); break; }
    elements.push(...j.elements);
  }
  let source = 'overpass';
  if (failed) { elements = await osmApiFallback(); source = 'osm-api'; }

  // de-duplicate (building chunks overlap at the seams)
  const seen = new Set();
  elements = elements.filter((e) => { const k = e.type[0] + e.id; if (seen.has(k)) return false; seen.add(k); return true; });
  console.log(`Processing ${elements.length} unique elements (${source})`);

  const out = {
    v: 1, source, generated: new Date().toISOString(),
    center: CENTER, size: SIZE, outer: OUTER,
    bounds: { minX: -H, maxX: H, minZ: -H, maxZ: H },
    buildings: [], roads: [], water: [], land: [], piers: [], parks: [], parking: [], plazas: [], rail: [],
    trees: [], signals: [], lamps: [],
  };

  // ---------- buildings & parts
  const bOutlines = [], bParts = [];
  for (const e of elements) {
    const t = e.tags || {};
    if (e.type === 'node') continue;
    const isPart = !!t['building:part'] && t['building:part'] !== 'no';
    const isB = !!t.building && t.building !== 'no' && !isPart;
    if (!isPart && !isB) continue;
    if (t.location === 'underground' || (parseFloat(t.layer) < 0 && !t.height)) continue;
    for (const poly of elementPolygons(e)) {
      const outer = cleanRing(poly.outer, 0.25, isPart ? 4 : 10);
      if (!outer) continue;
      const area = Math.abs(signedArea(outer));
      const c = centroid(outer);
      if (!inRect(c[0], c[1], CITY)) continue;
      const holes = poly.holes.map((h) => cleanRing(h, 0.25, 4)).filter(Boolean);
      (isPart ? bParts : bOutlines).push({ e, t, outer, holes, area, c });
    }
  }
  // outlines that contain building:parts are replaced by their parts (Simple 3D Buildings rule)
  const cell = 50, grid = new Map();
  const key = (i, k) => i * 100003 + k;
  bOutlines.forEach((b, idx) => {
    let x0 = Infinity, x1 = -Infinity, z0 = Infinity, z1 = -Infinity;
    for (const [x, z] of b.outer) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); z0 = Math.min(z0, z); z1 = Math.max(z1, z); }
    b.bb = [x0, z0, x1, z1];
    for (let i = Math.floor(x0 / cell); i <= Math.floor(x1 / cell); i++) for (let k = Math.floor(z0 / cell); k <= Math.floor(z1 / cell); k++) {
      const kk = key(i, k); if (!grid.has(kk)) grid.set(kk, []); grid.get(kk).push(idx);
    }
  });
  const hasParts = new Set();
  const partParent = new Map();
  for (const p of bParts) {
    const cands = grid.get(key(Math.floor(p.c[0] / cell), Math.floor(p.c[1] / cell))) || [];
    for (const idx of cands) {
      const b = bOutlines[idx];
      if (p.c[0] < b.bb[0] || p.c[0] > b.bb[2] || p.c[1] < b.bb[1] || p.c[1] > b.bb[3]) continue;
      if (pip(p.c[0], p.c[1], b.outer)) { hasParts.add(idx); partParent.set(p, b); break; }
    }
  }
  const partCount = new Map();
  for (const [, par] of partParent) partCount.set(par, (partCount.get(par) || 0) + 1);
  const pushB = (b, parent) => {
    const t = b.t;
    const area = b.area;
    // a part without height/levels inherits the outline's height only when it is (nearly) the whole building
    const inherit = parent && !t.height && !t['building:levels'] && (partCount.get(parent) === 1 || area > 0.6 * parent.area);
    let { h, lv } = buildingHeight(inherit ? { ...parent.t, ...t } : t, b.e.id, area);
    if (parent && !inherit && !t.height && !t['building:levels']) {
      const ph = buildingHeight(parent.t, parent.e.id, parent.area).h;
      h = Math.min(h, ph);
    }
    let mh = parseLen(t.min_height);
    if (!(mh > 0)) { const ml = parseFloat(t['building:min_level']); mh = ml > 0 ? ml * 3.2 + (ml > 0 ? 1 : 0) : 0; }
    if (mh >= h - 0.5) { if (mh > 0 && !t.height) h = mh + 3.2; else mh = 0; }
    const btype = t.building || (parent && parent.t.building);
    if (btype === 'roof' || btype === 'carport') { if (!(mh > 0)) mh = Math.max(0, h - 1); }       // open canopies
    if (btype === 'bridge' && !(mh > 0)) { h = Math.max(h, 9); mh = h - 4; }                       // skybridges
    const type = (t['building:part'] && t['building:part'] !== 'yes' ? t['building:part'] : null) || (parent ? parent.t.building : t.building) || 'yes';
    const o = { p: b.outer, h: r1(Math.min(h, 400)) };
    if (b.holes.length) o.hl = b.holes;
    if (mh > 0) o.mh = r1(mh);
    if (lv) o.lv = lv;
    o.t = type === 'yes' && (t.amenity || t.shop || t.office) ? (t.office ? 'office' : 'commercial') : type;
    const rs = t['roof:shape'] || (parent && parent.t['roof:shape']);
    if (rs && rs !== 'flat') o.r = rs;
    const rh = parseLen(t['roof:height']); if (rh > 0) o.rh = r1(rh);
    const n = t.name || (parent && parent.t.name); if (n) o.n = n;
    const col = t['building:colour'] || t.colour || (parent && parent.t['building:colour']); if (col) o.c = col;
    const mat = t['building:material'] || (parent && parent.t['building:material']); if (mat) o.m = mat;
    const rc = t['roof:colour']; if (rc) o.rc = rc;
    if (parent) o.part = 1;
    out.buildings.push(o);
  };
  bOutlines.forEach((b, idx) => { if (!hasParts.has(idx)) pushB(b, null); });
  for (const p of bParts) pushB(p, partParent.get(p) || null);
  console.log(`  buildings: ${bOutlines.length} outlines (${hasParts.size} replaced by parts), ${bParts.length} parts -> ${out.buildings.length}`);

  // ---------- roads
  const roadWays = elements.filter((e) => e.type === 'way' && e.tags && e.tags.highway && e.geometry && e.geometry.length >= 2);
  const deg = new Map(), anyNonBridge = new Set();
  for (const w of roadWays) {
    const t = w.tags;
    const tunnel = isYes(t.tunnel) || isYes(t.covered) || parseFloat(t.layer) < 0 || t.indoor === 'yes';
    if (!tunnel && !isYes(t.bridge) && w.nodes) for (const id of w.nodes) anyNonBridge.add(id);
    if (!MAJOR.has(t.highway) || tunnel || !w.nodes) continue;
    const n = w.nodes, closed = n[0] === n[n.length - 1];
    n.forEach((id, i) => {
      const end = !closed && (i === 0 || i === n.length - 1);
      if (closed && i === n.length - 1) return;
      deg.set(id, (deg.get(id) || 0) + (end ? 1 : 2));
    });
  }
  let roadCount = 0;
  const seenWay = new Set();
  for (const w of roadWays) {
    if (seenWay.has(w.id)) continue; seenWay.add(w.id);
    const t = w.tags; const cls = t.highway;
    if (t.area === 'yes' || (cls === 'pedestrian' && w.nodes && w.nodes[0] === w.nodes[w.nodes.length - 1] && t.area !== 'no')) {
      if (cls === 'pedestrian' || cls === 'footway') {
        for (const poly of elementPolygons(w)) { const r = cleanRing(poly.outer, 0.3, 20); if (r) out.plazas.push({ p: r }); }
      }
      continue;
    }
    if (!DRIVABLE.has(cls) && !MINOR_KEEP.has(cls)) continue;
    if (t.footway === 'sidewalk' || t.footway === 'crossing' || t.cycleway === 'crossing' || t.path === 'sidewalk') continue;
    if (isYes(t.tunnel) || isYes(t.covered) || parseFloat(t.layer) < 0 || t.indoor === 'yes' || t.level && parseFloat(t.level) < 0) continue;
    const bridge = isYes(t.bridge) ? 1 : 0;
    if (bridge && !DRIVABLE.has(cls)) continue; // skybridges / footbridges are skipped
    let pts = w.geometry.map((g) => proj(g.lat, g.lon));
    let flags = (w.nodes || []).map((id) => ((deg.get(id) || 0) >= 3 ? 1 : 0));
    if (flags.length !== pts.length) flags = pts.map(() => 0);
    let grounded = [0, 0];
    if (bridge && w.nodes) grounded = [anyNonBridge.has(w.nodes[0]) ? 1 : 0, anyNonBridge.has(w.nodes[w.nodes.length - 1]) ? 1 : 0];
    let oneway = isYes(t.oneway) && t.oneway !== '-1' ? 1 : 0;
    if (t.oneway === '-1' || t.oneway === 'reverse') { pts.reverse(); flags.reverse(); grounded.reverse(); oneway = 1; }
    if ((cls === 'motorway' || cls === 'motorway_link' || t.junction === 'roundabout') && t.oneway !== 'no') oneway = 1;
    let lanes = parseInt(t.lanes, 10); if (!(lanes > 0)) lanes = 0; lanes = Math.min(lanes, 10);
    let width = parseLen(t.width); if (!(width > 1)) width = 0;
    const layer = parseInt(t.layer, 10) || 0;
    const isExt = !w.geometry.some((g) => { const [x, z] = proj(g.lat, g.lon); return inRect(x, z, CITY); });
    const rect = isExt || (bridge && /^motorway/.test(cls)) ? { minX: -OH, maxX: OH, minZ: -OH, maxZ: OH } : CITY;
    const pieces = clipPolyline(pts, flags, rect);
    for (const pc of pieces) {
      const keep = new Set(); pc.flags.forEach((f, i) => { if (f) keep.add(i); });
      const idx = dpSimplify(pc.pts, 0.35, keep);
      const p = idx.map((i) => [r1(pc.pts[i][0]), r1(pc.pts[i][1])]);
      const fl = idx.map((i) => pc.flags[i]);
      // drop duplicate consecutive points
      const P = [], F = [];
      p.forEach((q, i) => { if (!P.length || Math.hypot(q[0] - P[P.length - 1][0], q[1] - P[P.length - 1][1]) > 0.05) { P.push(q); F.push(fl[i]); } else if (fl[i]) F[F.length - 1] = 1; });
      if (P.length < 2) continue;
      let len = 0; for (let i = 1; i < P.length; i++) len += Math.hypot(P[i][0] - P[i - 1][0], P[i][1] - P[i - 1][1]);
      if (len < 2) continue;
      const o = { p: P, c: cls };
      if (lanes) o.l = lanes;
      if (oneway) o.o = 1;
      if (width) o.w = r1(width);
      if (bridge) { o.b = 1; o.g = [pc.startClipped || pc.i0 > 0 ? 0 : grounded[0], pc.endClipped || pc.i1 < pts.length - 1 ? 0 : grounded[1]]; }
      if (layer) o.ly = layer;
      if (t.name) o.n = t.name;
      if (t.service) o.sv = t.service;
      const x = []; F.forEach((f, i) => { if (f) x.push(i); });
      if (x.length) o.x = x;
      out.roads.push(o); roadCount++;
    }
  }
  console.log(`  roads: ${roadCount} polylines (${out.roads.filter((r) => r.b).length} bridge pieces), ${out.plazas.length} pedestrian plazas`);

  // ---------- points
  for (const e of elements) {
    if (e.type !== 'node' || !e.tags) continue;
    const [x, z] = proj(e.lat, e.lon);
    if (!inRect(x, z, CITY)) continue;
    if (e.tags.highway === 'traffic_signals') out.signals.push([r1(x), r1(z)]);
    else if (e.tags.highway === 'street_lamp') out.lamps.push([r1(x), r1(z)]);
    else if (e.tags.natural === 'tree') out.trees.push([r1(x), r1(z)]);
  }
  console.log(`  points: ${out.signals.length} signals, ${out.lamps.length} lamps, ${out.trees.length} trees`);

  // ---------- areas
  const polyOut = (e, tol, minA, rect = CITY) => {
    const res = [];
    for (const poly of elementPolygons(e)) {
      const outer = cleanRing(poly.outer, tol, minA); if (!outer) continue;
      const c = centroid(outer);
      if (!outer.some(([x, z]) => inRect(x, z, rect)) && !inRect(c[0], c[1], rect)) continue;
      const o = { p: outer };
      const holes = poly.holes.map((h) => cleanRing(h, tol, 4)).filter(Boolean);
      if (holes.length) o.hl = holes;
      res.push(o);
    }
    return res;
  };
  const coastWays = [];
  for (const e of elements) {
    const t = e.tags || {};
    if (e.type === 'node') continue;
    if (t.natural === 'coastline' && e.type === 'way' && e.geometry) { coastWays.push(e.geometry.filter(Boolean).map((g) => proj(g.lat, g.lon))); continue; }
    if (t.natural === 'water' || t.waterway === 'riverbank') { out.water.push(...polyOut(e, 0.5, 4)); continue; }
    if (/^(pier|breakwater|groyne)$/.test(t.man_made || '')) {
      const polys = polyOut(e, 0.3, 8);
      if (polys.length) out.piers.push(...polys);
      else if (e.type === 'way' && e.geometry) { // linear pier -> buffer it
        const w = parseLen(t.width) > 1 ? parseLen(t.width) : 5;
        const pts = e.geometry.map((g) => proj(g.lat, g.lon));
        const L = [], Rr = [];
        for (let i = 0; i < pts.length; i++) {
          const a = pts[Math.max(0, i - 1)], b = pts[Math.min(pts.length - 1, i + 1)];
          let dx = b[0] - a[0], dz = b[1] - a[1]; const l = Math.hypot(dx, dz) || 1; dx /= l; dz /= l;
          L.push([pts[i][0] + dz * w / 2, pts[i][1] - dx * w / 2]); Rr.push([pts[i][0] - dz * w / 2, pts[i][1] + dx * w / 2]);
        }
        const ring = cleanRing(L.concat(Rr.reverse()), 0.2, 8);
        if (ring && ring.some(([x, z]) => inRect(x, z, CITY))) out.piers.push({ p: ring });
      }
      continue;
    }
    if (t.leisure || t.landuse || (t.natural && /^(wood|scrub|grassland)$/.test(t.natural))) {
      if (t.building) continue;
      for (const o of polyOut(e, 0.4, 15)) { o.k = t.leisure || t.landuse || t.natural; out.parks.push(o); }
      continue;
    }
    if (t.amenity === 'parking') {
      if (t.parking === 'underground' || t.parking === 'multi-storey' || t.building || t.location === 'underground') continue;
      out.parking.push(...polyOut(e, 0.4, 30));
      continue;
    }
    if (t.place === 'square') { out.plazas.push(...polyOut(e, 0.4, 30)); continue; }
  }
  const OR = { minX: -OH, maxX: OH, minZ: -OH, maxZ: OH };
  if (coastWays.length) {
    const land = coastToLand(coastWays, OR);
    for (const ring of land) { const r = cleanRing(ring, 2.0, 200); if (r) out.land.push({ p: r }); }
  }
  console.log(`  areas: ${out.water.length} water, ${out.land.length} land (from ${coastWays.length} coastline ways), ${out.piers.length} piers, ${out.parks.length} parks, ${out.parking.length} parking, ${out.plazas.length} plazas`);

  // ---------- rail
  for (const e of elements) {
    const t = e.tags || {};
    if (e.type !== 'way' || !t.railway || !e.geometry) continue;
    if (!/^(rail|light_rail|tram|narrow_gauge|funicular)$/.test(t.railway)) continue;
    if (isYes(t.tunnel) || parseFloat(t.layer) < 0 || t.location === 'underground') continue;
    const pts = e.geometry.map((g) => proj(g.lat, g.lon));
    for (const pc of clipPolyline(pts, pts.map(() => 0), CITY)) {
      const idx = dpSimplify(pc.pts, 0.25);
      const p = idx.map((i) => [r1(pc.pts[i][0]), r1(pc.pts[i][1])]);
      if (p.length < 2) continue;
      const o = { p, k: t.railway };
      if (isYes(t.bridge)) o.b = 1;
      if (t.embedded === 'yes' || t.railway === 'tram' || t.railway === 'funicular' || t.embedded === 'partial') o.e = 1;
      out.rail.push(o);
    }
  }
  console.log(`  rail: ${out.rail.length} polylines`);

  if (out.buildings.length < 500) console.warn(`WARNING: only ${out.buildings.length} buildings - data looks incomplete`);
  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  // compact JSON: one feature per line keeps the file diff-able yet small
  const json = JSON.stringify(out);
  fs.writeFileSync(OUT, json);
  console.log(`Wrote ${path.relative(ROOT, OUT)}  (${(json.length / 1048576).toFixed(2)} MB)`);
  return { out: OUT, buildings: out.buildings.length, roads: out.roads.length };
}

runCli(import.meta.url, 'engine/tools/bake-map.mjs', 'bake OpenStreetMap data (Overpass) into a city.json', SPEC, bakeMap);
