// Map data: the city.json contract (load + validate + origin offset), spawn search on the road network and the
// deterministic procedural fallback city (same schema) used when no baked map can be fetched.
//
// city.json (all coordinates metres in the BAKE frame: x = east, z = SOUTH, around `center`; rings are [[x, z], ...]):
//   { v, source, center: {lat, lon}, size, outer (land square edge, m), bounds: {minX, maxX, minZ, maxZ},
//     buildings: [{ p: ring, hl?: holes, h, mh?, lv?, t?, r?, rh?, c?, m?, rc?, n? }],
//     roads: [{ p: polyline, c: class, l?: lanes, o?: oneway, w?: width, n?: name, b?: bridge, ly?: layer,
//               x?: junction vertex indices, g?: [grounded start, grounded end], sv?: service kind }],
//     water / land / piers / parks (k: kind) / parking / plazas: [{ p: ring, hl?: holes }],
//     rail: [{ p, k, b?, e? }], trees / signals / lamps: [[x, z]] }
// prepareMapData() shifts everything by the spawn point (world origin = spawn) and returns the world-frame
// bounds / outer box; the offset (bake = world + offset) is what Terrain.setOffset and plugins use.
import { mulberry32, pointInRing } from './geom.js';
import { roadStyle } from './RoadNetwork.js';

/** Array keys of the city.json contract (missing ones are filled with []). */
export const MAP_KEYS = ['buildings', 'roads', 'water', 'land', 'piers', 'parks', 'parking', 'plazas', 'rail', 'trees', 'signals', 'lamps'];

/** Base URL for default asset paths (Vite BASE_URL in the browser, '/' elsewhere). */
export const assetBase = () => (import.meta.env && import.meta.env.BASE_URL) || '/';

/** Throws unless `data` looks like a usable city.json. */
export function validateMapData(data, { minRoads = 10 } = {}) {
  if (!data || !Array.isArray(data.buildings) || !Array.isArray(data.roads) || data.roads.length < minRoads) throw new Error('invalid city.json');
  return data;
}

/** Fetch + validate a city.json. Throws on HTTP / validation errors (MapWorld falls back to the procedural city). */
export async function loadMapData(url = `${assetBase()}map/city.json`, opts) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return validateMapData(await res.json(), opts);
}

/**
 * Fill missing arrays, find the spawn and shift the whole map so the spawn is the world origin (in place).
 * @param {object} data city.json
 * @param {object} [spawnOpts] see findSpawn
 * @returns {{ offset: {x, z}, bounds, outerBox, spawn: { headingRad, road } }}
 */
export function prepareMapData(data, spawnOpts = {}) {
  for (const k of MAP_KEYS) if (!Array.isArray(data[k])) data[k] = [];
  let b = data.bounds;
  if (!b || ![b.minX, b.maxX, b.minZ, b.maxZ].every(Number.isFinite)) {
    b = { minX: Infinity, maxX: -Infinity, minZ: Infinity, maxZ: -Infinity };
    for (const r of data.roads) for (const p of r.p || []) { b.minX = Math.min(b.minX, p[0]); b.maxX = Math.max(b.maxX, p[0]); b.minZ = Math.min(b.minZ, p[1]); b.maxZ = Math.max(b.maxZ, p[1]); }
    if (!isFinite(b.minX)) b = { minX: -800, maxX: 800, minZ: -800, maxZ: 800 };
  }
  const sp = findSpawn(data, b, spawnOpts);
  const ox = sp.x, oz = sp.z;
  const sh = (p) => { if (p) { p[0] -= ox; p[1] -= oz; } };
  const shRing = (r) => { if (Array.isArray(r)) for (const p of r) sh(p); };
  const shPoly = (o) => { if (!o) return; shRing(o.p); if (Array.isArray(o.hl)) o.hl.forEach(shRing); };
  data.buildings.forEach(shPoly); data.water.forEach(shPoly); data.land.forEach(shPoly); data.piers.forEach(shPoly);
  data.parks.forEach(shPoly); data.parking.forEach(shPoly); data.plazas.forEach(shPoly);
  data.roads.forEach((r) => shRing(r && r.p)); data.rail.forEach((r) => shRing(r && r.p));
  data.trees.forEach(sh); data.signals.forEach(sh); data.lamps.forEach(sh);
  const outer = (data.outer || 9000) / 2;
  return {
    offset: { x: ox, z: oz },
    bounds: { minX: b.minX - ox, maxX: b.maxX - ox, minZ: b.minZ - oz, maxZ: b.maxZ - oz },
    outerBox: { minX: -outer - ox, maxX: outer - ox, minZ: -outer - oz, maxZ: outer - oz },
    spawn: { headingRad: sp.heading, road: sp.name },
  };
}

/**
 * Spawn pose in the bake frame: the right-hand lane of a wide ground-level road segment (>= 35 m) near the map
 * centre, clear of buildings. heading: rotation.y such that local -Z points along the road.
 * @param {object} [o]
 * @param {RegExp|string|((name: string) => boolean)} [o.road] only consider roads with this name (falls back to any road)
 * @param {{x: number, z: number, heading?: number}} [o.at] explicit spawn (bake frame); skips the search
 */
export function findSpawn(data, b, o = {}) {
  if (o.at && Number.isFinite(o.at.x) && Number.isFinite(o.at.z)) return { x: o.at.x, z: o.at.z, heading: o.at.heading || 0, name: o.at.name || null };
  const cx = (b.minX + b.maxX) / 2, cz = (b.minZ + b.maxZ) / 2;
  const roadOk = !o.road ? null : typeof o.road === 'function' ? o.road : o.road instanceof RegExp ? (n) => o.road.test(n) : (n) => n === o.road;
  let cands = [];
  for (const r of data.roads) {
    if (!r || !Array.isArray(r.p) || r.p.length < 2 || r.b || (r.ly || 0) !== 0) continue;
    const st = roadStyle(r);
    if (st.minor || st.link) continue;
    const wide = /^(primary|secondary|trunk|tertiary)$/.test(st.base) && st.hw >= 6;
    for (let i = 0; i < r.p.length - 1; i++) {
      const a = r.p[i], c = r.p[i + 1];
      const dx = c[0] - a[0], dz = c[1] - a[1], L = Math.hypot(dx, dz);
      if (!(L > 35)) continue;
      const mx = (a[0] + c[0]) / 2, mz = (a[1] + c[1]) / 2;
      const d = Math.hypot(mx - cx, mz - cz) + (wide ? 0 : 5000) - Math.min(st.hw, 12) * 8;
      cands.push({ d, mx, mz, dx: dx / L, dz: dz / L, st, name: r.n });
    }
  }
  cands.sort((p, q) => p.d - q.d);
  if (roadOk) { const f = cands.filter((c) => c.name && roadOk(c.name)); if (f.length) cands = f; }
  const blocked = (x, z) => {
    for (const bd of data.buildings) {
      const p = bd && bd.p; if (!p || p.length < 3) continue;
      let inBox = false;
      // cheap bbox test on the first few points before the full test
      let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
      for (const q of p) { if (q[0] < minX) minX = q[0]; if (q[0] > maxX) maxX = q[0]; if (q[1] < minZ) minZ = q[1]; if (q[1] > maxZ) maxZ = q[1]; }
      inBox = x > minX - 3 && x < maxX + 3 && z > minZ - 3 && z < maxZ + 3;
      if (inBox && (pointInRing(x, z, p) || pointInRing(x + 3, z, p) || pointInRing(x - 3, z, p) || pointInRing(x, z + 3, p) || pointInRing(x, z - 3, p))) return true;
    }
    return false;
  };
  for (const c of cands.slice(0, 40)) {
    const st = c.st;
    const off = st.oneway ? 0 : Math.max(0, st.hw - st.parkW - st.laneW * 0.5);
    const x = c.mx - c.dz * off, z = c.mz + c.dx * off; // right-hand lane
    if (blocked(x, z)) continue;
    return { x, z, heading: Math.atan2(-c.dx, -c.dz), name: c.name };
  }
  return { x: 0, z: 0, heading: 0, name: null };
}

// ====================================================================== procedural fallback
// Deterministic procedural fallback city in the same schema as city.json (used when the baked map cannot be
// fetched). Grid downtown with towers, a waterfront to the east with piers, an elevated motorway with a ramp,
// parks and signalised avenues.
/** @param {number} [seed] @param {{ waterfrontName?: string, center?: {lat, lon} }} [o] names / geo reference of the fallback */
export function generateProceduralCity(seed = 1337, { waterfrontName = 'Harbor Boulevard', center = { lat: 0, lon: 0 } } = {}) {
  const rnd = mulberry32(seed);
  const SIZE = 2000, H = SIZE / 2;
  const BX = 112, BZ = 86;            // street spacing (centre lines)
  const COAST = 760;                   // water east of this x
  const r1 = (v) => Math.round(v * 10) / 10;
  const xs = [], zs = [];
  for (let x = -H + 24; x < COAST - 40; x += BX) xs.push(x);
  for (let z = -H + 20; z <= H - 20; z += BZ) zs.push(z);
  const avenueX = (i) => i % 3 === 1, avenueZ = (k) => k % 4 === 2;
  const motorK = Math.floor(zs.length * 0.62); // motorway runs mid-block between zs[motorK] and zs[motorK+1]
  const zMotor = (zs[motorK] + zs[motorK + 1]) / 2;
  const roads = [], signals = [];
  const xEnd = COAST - 18;
  // N-S streets
  xs.forEach((x, i) => {
    const p = zs.map((z) => [r1(x), r1(z)]);
    roads.push({ p, c: avenueX(i) ? 'primary' : i % 2 ? 'tertiary' : 'residential', l: avenueX(i) ? 4 : 2, n: `Street ${i + 1}`, x: p.map((_, k) => k).slice(1, -1) });
  });
  // the waterfront boulevard
  {
    const p = zs.map((z) => [r1(xEnd), r1(z)]);
    roads.push({ p, c: 'primary', l: 4, n: waterfrontName, x: p.map((_, k) => k).slice(1, -1) });
  }
  // E-W streets
  zs.forEach((z, k) => {
    const p = xs.map((x) => [r1(x), r1(z)]).concat([[r1(xEnd), r1(z)]]);
    roads.push({ p, c: avenueZ(k) ? 'secondary' : 'residential', l: avenueZ(k) ? 4 : 2, o: !avenueZ(k) && k % 2 ? 1 : 0, n: `Avenue ${k + 1}`, x: p.map((_, q) => q).slice(1, -1) });
    if (avenueZ(k)) xs.forEach((x, i) => { if (avenueX(i)) signals.push([r1(x), r1(z)]); });
  });
  // elevated motorway (grounded at the west edge, leaves over the bay to the east)
  roads.push({ p: [[-H, r1(zMotor)], [-H + 260, r1(zMotor)], [0, r1(zMotor)], [COAST + 300, r1(zMotor - 60)], [H + 600, r1(zMotor - 160)]], c: 'motorway', l: 3, o: 0, b: 1, ly: 2, g: [1, 0], n: 'Bay Freeway' });

  // blocks
  const buildings = [], parks = [];
  const downtown = [-150, -120];
  for (let i = 0; i < xs.length; i++) {
    for (let k = 0; k + 1 < zs.length; k++) {
      const x0 = xs[i], x1 = i + 1 < xs.length ? xs[i + 1] : xEnd, z0 = zs[k], z1 = zs[k + 1];
      if (k === motorK) continue; // motorway corridor
      const sw = 14; // street + sidewalk clearance
      const bx0 = x0 + sw, bx1 = x1 - sw, bz0 = z0 + sw, bz1 = z1 - sw;
      if (bx1 - bx0 < 20 || bz1 - bz0 < 20) continue;
      if (rnd() < 0.05 || (i === 3 && k === 4)) { parks.push({ p: [[bx0, bz0], [bx1, bz0], [bx1, bz1], [bx0, bz1]].map(([a, b]) => [r1(a), r1(b)]), k: 'park' }); continue; }
      const cx = (bx0 + bx1) / 2, cz = (bz0 + bz1) / 2;
      const d = Math.hypot(cx - downtown[0], cz - downtown[1]);
      const tower = Math.exp(-((d / 520) ** 2));
      const nx = 2 + Math.floor(rnd() * 3), nz = 2;
      for (let a = 0; a < nx; a++) for (let b = 0; b < nz; b++) {
        if (tower > 0.6 && rnd() < 0.5 && (a + b) % 2) continue;
        const lx0 = bx0 + ((bx1 - bx0) * a) / nx + 0.6, lx1 = bx0 + ((bx1 - bx0) * (a + 1)) / nx - 0.6;
        const lz0 = bz0 + ((bz1 - bz0) * b) / nz + 0.6, lz1 = bz0 + ((bz1 - bz0) * (b + 1)) / nz - 0.6;
        let h = 10 + rnd() * 16 + tower * (30 + 200 * rnd() ** 2.2);
        const t = h > 60 ? (rnd() < 0.7 ? 'office' : 'commercial') : rnd() < 0.5 ? 'apartments' : 'yes';
        const lv = Math.max(1, Math.round(h / 3.6));
        buildings.push({ p: [[lx0, lz0], [lx1, lz0], [lx1, lz1], [lx0, lz1]].map(([u, v]) => [r1(u), r1(v)]), h: r1(h), lv, t });
      }
    }
  }
  // piers + land
  const piers = [];
  for (let z = -H + 120; z < H - 100; z += 190) {
    if (Math.abs(z - zMotor) < 120) continue;
    const zz = z + rnd() * 40;
    piers.push({ p: [[COAST - 5, zz], [COAST + 190, zz], [COAST + 190, zz + 42], [COAST - 5, zz + 42]].map(([a, b]) => [r1(a), r1(b)]) });
    buildings.push({ p: [[COAST + 15, zz + 6], [COAST + 175, zz + 6], [COAST + 175, zz + 36], [COAST + 15, zz + 36]].map(([a, b]) => [r1(a), r1(b)]), h: 11, t: 'warehouse' });
  }
  const O = 4500;
  const land = [{ p: [[-O, -O], [COAST, -O], [COAST, O], [-O, O]] }];
  return {
    v: 1, source: 'procedural', generated: 'runtime', center, size: SIZE, outer: 2 * O,
    bounds: { minX: -H, maxX: H, minZ: -H, maxZ: H },
    buildings, roads, water: [], land, piers, parks, parking: [], plazas: [], rail: [], trees: [], signals, lamps: [],
  };
}
