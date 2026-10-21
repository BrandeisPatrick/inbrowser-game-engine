// Vegetation renderer. Props.js decides WHERE trees go; this module decides what they look like.
// Contract (keep it): buildTreeMeshes(placements, { groundAt, colliders, seed, rules, mix })
//   placements: [{ x, z, s (scale 0.7..1.5), hue (0..1 random), kind: 'osm'|'street'|'park',
//                  ctx: { road?: name, cls?: road class, side?, dx?, dz?, park?: park kind } }]
//   rules (optional, the game's species rules per road): [{ road: RegExp | (name) => bool, species: 'palm',
//          share: 0..1 (street trees of that road get `species` with this probability, else `other`), other?,
//          radius: m (OSM tree nodes within this distance of such a street tree get `species` with p = share) }]
//          - first matching rule wins; none by default (every street uses the per-class mixes).
//   mix (optional): overrides of the species mixes (MIX keys: residential, minor, major, osm, forest, park, garden, open)
//   returns { group: THREE.Object3D|null, count, update(dt, camPos, darkness, time) }
//
// Species (all procedural, see VegModels.js): London plane (2 variants), ficus/Brisbane-box-like round tree,
// coast live oak, Canary Island date palm, Monterey cypress, pine; plus park shrubs and grass tufts.
// Autumn / flowering colour variants via a per-instance tint.
// LOD: per-species InstancedMesh holds only trees within ~220 m of the camera (re-partitioned when the camera
// moves > 20 m); every tree also has a baked billboard impostor (one instanced draw for all species). The two
// cross-fade with complementary screen-space dither between 150 and 185 m, so nothing pops. Near trees cast
// alpha-tested shadows (custom depth material with the same wind); impostors don't.
import * as THREE from 'three';
import { mulberry32, hashStr } from './geom.js';
import { makeVegAtlas } from './VegTextures.js';
import { getVegModels, SPECIES } from './VegModels.js';
import { createVegMaterial, createImpostorMaterial } from './VegMaterials.js';
import { bakeImpostors as bakeAtlas, impostorViews, DEFAULT_RINGS } from '../render/Impostors.js';

// Tree impostor bake: the engine's generic baker (render/Impostors.js). Every species is rendered from 20
// directions (8 azimuths at 10 and 45 deg elevation, 4 at 80 deg) into an albedo atlas (sRGB, AO baked, alpha)
// and an object-space normal atlas, in a short-lived offscreen WebGL context.
export const IMP_VIEWS = 20;
const VIEWS = impostorViews(DEFAULT_RINGS);
export function impostorView(v) { return VIEWS[v]; }
/**
 * models: [{ geo, center: Vector3, radius }] in species order. atlasLevels: mip levels of the foliage atlas.
 * Returns { albedo, normal, layout: {cols, cellU, cellV, pad} } or null if WebGL is unavailable.
 */
export function bakeVegImpostors(models, atlasLevels, { cols = 12, cellPx = 170, texSize = 2048, ss = 2 } = {}) {
  return bakeAtlas(models, { rings: DEFAULT_RINGS, map: atlasLevels, cols, cellPx, texSize, ss, label: 'trees', name: 'vegImpostor', failNote: ', near-only vegetation' });
}

const MAX_TREES = 12000, MAX_SHRUBS = 6000, MAX_GRASS = 24000;
const TREE_FADE = { start: 150, end: 185 };
const SHRUB_FADE = { start: 95, end: 125 };
const GRASS_FADE = { start: 48, end: 68 };
const MARGIN = 35;          // near-set radius = fade end + margin (camera may move this far before a re-partition lands)
const REPART_DIST = 20;     // metres of camera travel that trigger a re-partition
const SPI = Object.fromEntries(SPECIES.map((s, i) => [s, i]));

const TINTS = {
  autumn: [[1.0, 0.74, 0.2], [1.0, 0.5, 0.12], [0.88, 0.27, 0.12]],
  flower: [[0.66, 0.5, 1.0], [1.0, 0.56, 0.74], [1.0, 0.94, 0.86]],
};
const MIX = {
  residential: [['planeA', 0.3], ['planeB', 0.2], ['round', 0.5]],
  minor: [['planeA', 0.35], ['planeB', 0.25], ['round', 0.4]],
  major: [['planeA', 0.45], ['planeB', 0.35], ['round', 0.2]],
  osm: [['planeA', 0.3], ['planeB', 0.25], ['round', 0.33], ['oak', 0.12]],
  forest: [['pine', 0.45], ['cypress', 0.4], ['oak', 0.15]],
  park: [['cypress', 0.22], ['pine', 0.18], ['oak', 0.25], ['planeA', 0.12], ['planeB', 0.08], ['round', 0.15]],
  garden: [['round', 0.35], ['oak', 0.2], ['planeB', 0.15], ['cypress', 0.15], ['palm', 0.15]],
  open: [['oak', 0.3], ['round', 0.3], ['planeA', 0.2], ['cypress', 0.1], ['pine', 0.1]],
};
const STREET_CLASS = { residential: 'residential', living_street: 'residential', tertiary: 'minor', unclassified: 'minor', secondary: 'major', primary: 'major', trunk: 'major' };
const PARK_CLASS = { forest: 'forest', wood: 'forest', scrub: 'forest', park: 'park', common: 'park', recreation_ground: 'park', village_green: 'park', dog_park: 'park', garden: 'garden' };

function pick(list, r) {
  let s = 0; for (const e of list) s += e[1];
  let x = r * s;
  for (const e of list) { x -= e[1]; if (x <= 0) return e[0]; }
  return list[list.length - 1][0];
}

class PtGrid {
  constructor(cell) { this.c = cell; this.m = new Map(); }
  key(i, j) { return i * 73856093 ^ j * 19349663; }
  add(x, z, v = 1) { const k = this.key(Math.floor(x / this.c), Math.floor(z / this.c)); let a = this.m.get(k); if (!a) this.m.set(k, (a = [])); a.push(x, z, v); }
  count(x, z, r, max = 99) {
    let n = 0; const c = this.c;
    for (let i = Math.floor((x - r) / c); i <= Math.floor((x + r) / c); i++) for (let j = Math.floor((z - r) / c); j <= Math.floor((z + r) / c); j++) {
      const a = this.m.get(this.key(i, j)); if (!a) continue;
      for (let q = 0; q < a.length; q += 3) if ((a[q] - x) ** 2 + (a[q + 1] - z) ** 2 < r * r && ++n >= max) return n;
    }
    return n;
  }
}

const _m = new THREE.Matrix4(), _q = new THREE.Quaternion(), _s = new THREE.Vector3(), _p = new THREE.Vector3(), _e = new THREE.Euler();

const ruleTest = (r, name) => (typeof r.road === 'function' ? !!r.road(name) : r.road.test(name));

export function buildTreeMeshes(placements, { groundAt = () => 0, colliders = null, seed = 1, rules = [], mix = null } = {}) {
  const n = Math.min(placements.length, MAX_TREES);
  if (!n) return { group: null, count: 0, update() {} };
  const t0 = (globalThis.performance || Date).now();
  const rnd = mulberry32(seed);
  const models = getVegModels();
  const MIXES = mix ? { ...MIX, ...mix } : MIX;
  const RULES = (rules || []).filter((r) => r && r.road && SPI[r.species] !== undefined);

  // ---------------------------------------------------------- species zones (rules: e.g. palm-lined boulevards)
  const zones = RULES.map(() => new PtGrid(40));
  for (let i = 0; i < n; i++) {
    const p = placements[i], road = p.ctx && p.ctx.road;
    if (!road) continue;
    for (let k = 0; k < RULES.length; k++) if (ruleTest(RULES[k], road)) { zones[k].add(p.x, p.z); break; }
  }

  // ---------------------------------------------------------- species, transforms, colliders
  const SP = new Uint8Array(n), X = new Float32Array(n), Y = new Float32Array(n), Z = new Float32Array(n);
  const M = new Float32Array(n * 16), TN = new Float32Array(n * 4);
  const IP = new Float32Array(n * 4), IS = new Float32Array(n * 4);
  const counts = new Array(SPECIES.length).fill(0);
  const parkTrees = [];
  const parkGrid = new PtGrid(16);
  const season = { autumn: 0, flower: 0 };
  for (let i = 0; i < n; i++) {
    const p = placements[i];
    const { x, z } = p, s = p.s || 1, hue = p.hue ?? rnd();
    const ctx = p.ctx || {};
    const r1 = rnd(), r2 = rnd(), r3 = rnd();
    let sp, blockR = r1;
    if (p.kind === 'street') {
      const road = ctx.road || '';
      const rule = road ? RULES.find((r) => ruleTest(r, road)) : null;
      if (rule) sp = r1 < (rule.share ?? 1) ? rule.species : (rule.other || rule.species);
      else {
        const key = `${road || ctx.cls || 'x'}|${Math.floor(x / 110)}|${Math.floor(z / 110)}`;
        blockR = hashStr(key);
        const mixL = MIXES[STREET_CLASS[ctx.cls] || 'minor'];
        sp = pick(mixL, r1 < 0.85 ? blockR : r2); // one species per block, 15% replacements
      }
    } else if (p.kind === 'park') {
      sp = pick(MIXES[PARK_CLASS[ctx.park] || 'open'], r1);
    } else {
      let k = 0;
      for (; k < RULES.length; k++) if (zones[k].count(x, z, RULES[k].radius ?? 30, 1) && r1 < (RULES[k].share ?? 1)) break;
      sp = k < RULES.length ? RULES[k].species : pick(MIXES.osm, r1);
    }
    const si = SPI[sp], mdl = models[sp];
    SP[i] = si; counts[si]++;
    // seasonal tint (planted blocks share it)
    let tint = null, tr = p.kind === 'street' ? hashStr(`t${blockR}`) : r3;
    if ((sp === 'planeA' || sp === 'planeB') && tr < 0.06) tint = ['autumn', TINTS.autumn[(r2 * 3) | 0], 0.55 + r3 * 0.3];
    else if (sp === 'round' && tr < 0.05) tint = ['autumn', TINTS.autumn[(r2 * 2) | 0], 0.6 + r3 * 0.25];
    else if (sp === 'round' && tr < 0.13) tint = ['flower', TINTS.flower[(hashStr(`f${blockR}`) * 3) | 0], 0.62 + r3 * 0.2];
    else if (sp === 'oak' && tr < 0.03) tint = ['autumn', TINTS.autumn[0], 0.45];
    if (tint) { season[tint[0]]++; TN.set([tint[1][0], tint[1][1], tint[1][2], tint[2]], i * 4); }
    else TN.set([1, 1, 1, 0], i * 4);
    // scale
    let base = s;
    if (sp === 'palm') base = p.kind === 'park' ? s * 0.85 : 0.88 + (s - 0.7) * 0.4;
    else if (sp === 'oak') base = s * 0.95;
    const sxz = base * (0.92 + 0.16 * r2), sy = base * (0.92 + 0.18 * hue);
    // ground: lowest of 5 samples around the trunk (trunks are buried 0.6 m, so this never floats)
    const rr = Math.max(0.4, mdl.trunkR * sxz * 1.4);
    let y0 = groundAt(x, z);
    for (const [ox, oz] of [[rr, 0], [-rr, 0], [0, rr], [0, -rr]]) { const g = groundAt(x + ox, z + oz); if (Number.isFinite(g) && g < y0) y0 = g; }
    if (!Number.isFinite(y0)) y0 = 0;
    const yaw = hue * Math.PI * 14 + r3 * 6.28;
    const tilt = sp === 'palm' ? 0.07 : 0.025;
    _q.setFromEuler(_e.set((r2 - 0.5) * tilt, yaw, (r3 - 0.5) * tilt, 'YXZ'));
    _m.compose(_p.set(x, y0, z), _q, _s.set(sxz, sy, sxz));
    _m.toArray(M, i * 16);
    X[i] = x; Y[i] = y0; Z[i] = z;
    IP[i * 4] = x; IP[i * 4 + 1] = y0; IP[i * 4 + 2] = z; IP[i * 4 + 3] = yaw;
    IS[i * 4] = sxz; IS[i * 4 + 1] = sy; IS[i * 4 + 2] = si; IS[i * 4 + 3] = 0;
    if (colliders && mdl.trunkR > 0) {
      const rc = mdl.trunkR * sxz * 1.1;
      colliders.add(x - rc, y0 - 0.25, x + rc, y0 + mdl.colH * sy, z - rc, z + rc, 2);
    }
    if (p.kind === 'park') { parkTrees.push(i); parkGrid.add(x, z); }
  }

  // ---------------------------------------------------------- shrubs + grass tufts around some park trees
  const SH = [], GR = []; // [x, y, z, yaw, scale]
  const colOut = [];
  const freeSpot = (x, z, yRef, r) => {
    const g = groundAt(x, z);
    if (!Number.isFinite(g) || Math.abs(g - yRef) > 0.9) return null;
    if (colliders && colliders.query(x, z, r, colOut).length) return null;
    return g;
  };
  for (const i of parkTrees) {
    const p = placements[i], x = X[i], z = Z[i], yRef = Y[i];
    const interior = parkGrid.count(x, z, 16, 3) >= 2;
    const pk = p.ctx && p.ctx.park;
    const pShrub = pk === 'garden' ? 0.7 : 0.45;
    if (rnd() < pShrub && SH.length < MAX_SHRUBS * 5) {
      const k = 1 + ((rnd() * 3) | 0);
      for (let j = 0; j < k; j++) {
        const a = rnd() * 6.28, d = 2.2 + rnd() * (interior ? 4.5 : 2.0);
        const sx = x + Math.cos(a) * d, sz = z + Math.sin(a) * d;
        const g = freeSpot(sx, sz, yRef, 0.8);
        if (g !== null) SH.push(sx, g, sz, rnd() * 6.28, 0.7 + rnd() * 0.7);
      }
    }
    if (rnd() < 0.42 && GR.length < MAX_GRASS * 5) {
      const a = rnd() * 6.28, d = 1.2 + rnd() * (interior ? 3.5 : 1.5);
      const cx = x + Math.cos(a) * d, cz = z + Math.sin(a) * d;
      const k = 8 + ((rnd() * 12) | 0);
      for (let j = 0; j < k; j++) {
        const b = rnd() * 6.28, e = Math.sqrt(rnd()) * 2.6;
        const gx = cx + Math.cos(b) * e, gz = cz + Math.sin(b) * e;
        if ((gx - x) ** 2 + (gz - z) ** 2 < 0.8) continue;
        const g = freeSpot(gx, gz, yRef, 0.25);
        if (g !== null) GR.push(gx, g, gz, rnd() * 6.28, 0.7 + rnd() * 0.6);
      }
    }
  }
  const nSh = Math.min(MAX_SHRUBS, SH.length / 5), nGr = Math.min(MAX_GRASS, GR.length / 5);

  const stats = {
    species: Object.fromEntries(SPECIES.map((s, i) => [s, counts[i]])),
    season, shrubs: nSh, grass: nGr,
    trisPerModel: Object.fromEntries(Object.entries(models).map(([k, v]) => [k, v.tris])),
    impostors: false, buildMs: 0,
  };
  if (typeof document === 'undefined') { // headless (node sims): colliders only
    stats.buildMs = Math.round((globalThis.performance || Date).now() - t0);
    return { group: null, count: n, stats, update() {} };
  }

  // ---------------------------------------------------------- GPU resources
  const atlas = makeVegAtlas(seed + 3);
  const U = { uTime: { value: 0 }, uViewPos: { value: new THREE.Vector3(0, -1e6, 0) } };
  const treeMat = createVegMaterial(atlas.texture, U, { ...TREE_FADE, wind: 1, drop: 0.1 });
  const shrubMat = createVegMaterial(atlas.texture, U, { ...SHRUB_FADE, wind: 1, drop: 0.12 });
  const grassMat = createVegMaterial(atlas.texture, U, { ...GRASS_FADE, wind: 1, drop: 0 });
  const group = new THREE.Group(); group.name = 'trees';

  const mkNear = (name, mat, cap, shadow) => {
    const src = models[name].geo, g = new THREE.BufferGeometry();
    for (const a of ['position', 'normal', 'uv', 'color', 'veg']) g.setAttribute(a, src.getAttribute(a));
    g.setIndex(src.getIndex());
    const tint = new THREE.InstancedBufferAttribute(new Float32Array(Math.max(1, cap) * 4), 4);
    tint.setUsage(THREE.DynamicDrawUsage);
    g.setAttribute('aTint', tint);
    const mesh = new THREE.InstancedMesh(g, mat, Math.max(1, cap));
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    mesh.count = 0; mesh.visible = false;
    mesh.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1);
    mesh.castShadow = shadow; mesh.receiveShadow = true;
    if (shadow) mesh.customDepthMaterial = mat.userData.depth;
    mesh.name = `veg-${name}`;
    group.add(mesh);
    return { mesh, tint, mat: mesh.instanceMatrix.array, tin: tint.array };
  };
  const near = SPECIES.map((name, i) => mkNear(name, treeMat, counts[i], true));
  const shrubMesh = nSh ? mkNear('shrub', shrubMat, nSh, true) : null;
  const grassMesh = nGr ? mkNear('grass', grassMat, nGr, false) : null;

  // shrub / grass instance data (static per instance; partitioned like trees)
  const mkSmall = (arr, cnt) => {
    const Mx = new Float32Array(cnt * 16), P = new Float32Array(cnt * 3);
    for (let k = 0; k < cnt; k++) {
      const [x, y, z, yaw, sc] = [arr[k * 5], arr[k * 5 + 1], arr[k * 5 + 2], arr[k * 5 + 3], arr[k * 5 + 4]];
      _q.setFromAxisAngle(_p.set(0, 1, 0), yaw);
      _m.compose(_p.set(x, y, z), _q, _s.set(sc, sc * (0.85 + 0.3 * ((k * 0.618) % 1)), sc));
      _m.toArray(Mx, k * 16);
      P[k * 3] = x; P[k * 3 + 1] = y; P[k * 3 + 2] = z;
    }
    return { M: Mx, P, n: cnt };
  };
  const shrubData = shrubMesh ? mkSmall(SH, nSh) : null;
  const grassData = grassMesh ? mkSmall(GR, nGr) : null;
  if (shrubMesh) shrubMesh.tin.fill(0); // no tint
  if (grassMesh) grassMesh.tin.fill(0);

  // ---------------------------------------------------------- impostors
  let impMesh = null;
  const bakeModels = SPECIES.map((name) => { const g = models[name].geo; return { geo: g, center: g.boundingSphere.center.clone(), radius: g.boundingSphere.radius }; });
  const baked = bakeVegImpostors(bakeModels, atlas.levels);
  if (baked) {
    const ig = new THREE.InstancedBufferGeometry();
    ig.setAttribute('position', new THREE.Float32BufferAttribute([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0], 3));
    ig.setAttribute('normal', new THREE.Float32BufferAttribute([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], 3));
    ig.setAttribute('uv', new THREE.Float32BufferAttribute([0, 0, 1, 0, 1, 1, 0, 1], 2));
    ig.setIndex([0, 1, 2, 0, 2, 3]);
    ig.setAttribute('iPos', new THREE.InstancedBufferAttribute(IP, 4));
    ig.setAttribute('iScl', new THREE.InstancedBufferAttribute(IS, 4));
    ig.setAttribute('aTint', new THREE.InstancedBufferAttribute(TN, 4));
    ig.instanceCount = n;
    const spec = bakeModels.map((b) => ({ cx: b.center.x, cy: b.center.y, cz: b.center.z, r: b.radius }));
    const impMat = createImpostorMaterial(baked.albedo, baked.normal, U, spec, baked.layout, TREE_FADE);
    impMesh = new THREE.Mesh(ig, impMat);
    impMesh.frustumCulled = false;
    impMesh.castShadow = false; impMesh.receiveShadow = true;
    impMesh.name = 'veg-impostors';
    group.add(impMesh);
    stats.impostors = true;
  } else {
    // no impostors: keep near models out to a longer range instead
    treeMat.userData.uFade.value.set(380, 420, 1, 0.1);
  }
  const treeFadeEnd = baked ? TREE_FADE.end : 420;

  // ---------------------------------------------------------- near-set partitioning
  const last = new THREE.Vector3(Infinity, 0, 0);
  let sinceRepart = 1e9;
  const cnt = new Int32Array(SPECIES.length);
  const flush = (o, c, cx, cy, cz, rad) => {
    o.mesh.count = c; o.mesh.visible = c > 0;
    o.mesh.instanceMatrix.clearUpdateRanges(); o.mesh.instanceMatrix.addUpdateRange(0, Math.max(1, c) * 16); o.mesh.instanceMatrix.needsUpdate = true;
    o.tint.clearUpdateRanges(); o.tint.addUpdateRange(0, Math.max(1, c) * 4); o.tint.needsUpdate = true;
    o.mesh.boundingSphere.center.set(cx, cy, cz); o.mesh.boundingSphere.radius = rad;
  };
  const partSmall = (o, d, fadeEnd, cx, cy, cz) => {
    if (!o) return;
    const R2 = (fadeEnd + MARGIN) ** 2, P = d.P, Ms = d.M, dst = o.mat;
    let c = 0;
    for (let k = 0; k < d.n; k++) {
      const dx = P[k * 3] - cx, dy = P[k * 3 + 1] - cy, dz = P[k * 3 + 2] - cz;
      if (dx * dx + dy * dy + dz * dz > R2) continue;
      const s = k * 16, t = c * 16;
      for (let j = 0; j < 16; j++) dst[t + j] = Ms[s + j];
      c++;
    }
    flush(o, c, cx, cy, cz, fadeEnd + MARGIN + 5);
  };
  function repartition(cx, cy, cz) {
    const R2 = (treeFadeEnd + MARGIN) ** 2;
    cnt.fill(0);
    for (let i = 0; i < n; i++) {
      const dx = X[i] - cx, dy = Y[i] - cy, dz = Z[i] - cz;
      if (dx * dx + dy * dy + dz * dz > R2) continue;
      const si = SP[i], k = cnt[si]++, o = near[si];
      const s = i * 16, t = k * 16, dst = o.mat;
      for (let j = 0; j < 16; j++) dst[t + j] = M[s + j];
      const s4 = i * 4, t4 = k * 4, tn = o.tin;
      tn[t4] = TN[s4]; tn[t4 + 1] = TN[s4 + 1]; tn[t4 + 2] = TN[s4 + 2]; tn[t4 + 3] = TN[s4 + 3];
    }
    for (let si = 0; si < near.length; si++) flush(near[si], cnt[si], cx, cy, cz, treeFadeEnd + MARGIN + 30);
    partSmall(shrubMesh, shrubData, SHRUB_FADE.end, cx, cy, cz);
    partSmall(grassMesh, grassData, GRASS_FADE.end, cx, cy, cz);
    last.set(cx, cy, cz);
    sinceRepart = 0;
  }

  function update(dt, camPos, darkness, time) {
    U.uTime.value = Number.isFinite(time) ? time : U.uTime.value + (dt || 0);
    if (!camPos) return;
    U.uViewPos.value.copy(camPos);
    sinceRepart += dt || 0;
    const d2 = last.distanceToSquared(camPos);
    if (d2 > REPART_DIST * REPART_DIST && (sinceRepart > 0.1 || d2 > 60 * 60)) repartition(camPos.x, camPos.y, camPos.z);
  }

  stats.buildMs = Math.round(performance.now() - t0);
  console.info('[trees]', n, 'trees', stats);
  return {
    group, count: n, stats, update,
    /** debug: current near-instance counts per species */
    get nearCounts() { return Object.fromEntries(SPECIES.map((s, i) => [s, near[i].mesh.count])); },
  };
}
