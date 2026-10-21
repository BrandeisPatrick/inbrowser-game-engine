// Street furniture placed from the road network (all instanced, everything stands on `groundAt(x,z)`):
// street lights (cobra heads on arterials, per-street styles from the `streets.lampStyle` rule - e.g. ornate
// trident standards or twin globes - extra `streets.featureLamps` standards, acorn lamps at OSM lamp nodes) + 8 real PointLights at night on the lamps nearest the view (engine
// NearestLights: only lamps whose light can reach the frustum compete; >8 forward lights cost ~0.4-0.8 ms each
// on Apple GPUs, and the 16 nearest-by-distance lamps of old were mostly behind / under the camera); traffic signals (mast
// arms over wide approaches, pedestal poles otherwise, visored heads with backplates, pedestrian heads
// with walk / flashing-hand phases, push buttons, illuminated street-name signs) driven by the phase
// Traffic.js reads (`signals[].state`); trees (placement here, rendering in Vegetation.js); parked cars
// (Vehicles.js); signs (stop / all-way, green street-name blades, one-way, speed limit, street
// cleaning, no parking, bus stop); hydrants, bins, benches, bike racks, news boxes, mailboxes, meters,
// planters, bollards, bus shelters with glass roofs, manholes / storm drains (road decals) and wooden
// utility poles with sagging wires.
//
// Draw calls: one BatchedMesh for all furniture (per-instance frustum + distance culling), one for parked
// cars (LOD swap), signal lenses, sign panels (atlas), road decals, shelter glass (+ shadow passes).
//
// City-specific placement is passed in by the game (all optional; road-name predicates are RegExp or (name, road) => bool):
//   streets.lampStyle(name, road)   -> furniture lamp type for a (non-bridge) road, or null for the generic cobra rule
//   streets.featureLamps            [{ road, style, step = 28 }] extra lamp standards along those roads (split carriageways)
//   streets.boulevard(name, road)   car-free kerbs (no parked cars / meters / cleaning signs), dense furniture + cafés
//   streets.promenade(name, road)   dense bins / bollards / benches
//   streets.utilityPoles(name, road) false = wiring underground on that residential street (default true)
//   signStyle                       Furniture.js buildSignAtlas style (posters, manhole text, blade colours)
//   treeRules / treeMix             Vegetation.js species rules / mixes
import * as THREE from 'three';
import { RBIT, samplePolyline } from './RoadNetwork.js';
import { pointInPoly, ringBounds, signedArea, mulberry32, hashStr, clamp } from './geom.js';
import { buildTreeMeshes } from './Vegetation.js';
import { NearestLights } from './NearestLights.js';
import { makeVehicleGeometry, VEHICLE_SPECS, pickVehicleColor, createVehicleMaterial, PartBuilder } from './Vehicles.js';
import {
  FURNITURE_META, makeFurnitureGeometry, makeShelterGlassGeometry, makePanelGeometry, makeDecalGeometry,
  createFurnitureMaterial, createPanelMaterial, createShelterGlassMaterial, buildSignAtlas,
} from './Furniture.js';

const _m = new THREE.Matrix4(), _q = new THREE.Quaternion(), _s = new THREE.Vector3(), _p = new THREE.Vector3(), _e = new THREE.Euler();
const _q2 = new THREE.Quaternion(), _n = new THREE.Vector3();
const UP = new THREE.Vector3(0, 1, 0);

function setTRS(mesh, i, x, y, z, yaw, sx = 1, sy = 1, sz = 1) {
  _q.setFromAxisAngle(UP, yaw);
  _m.compose(_p.set(x, y, z), _q, _s.set(sx, sy, sz));
  mesh.setMatrixAt(i, _m);
}
// yaw that rotates local +X onto direction (ux, uz)
const yawX = (ux, uz) => Math.atan2(-uz, ux);
// yaw that rotates local -Z onto direction (ux, uz)
const yawNegZ = (ux, uz) => Math.atan2(-ux, -uz);
// yaw that rotates local +Z (a panel's front normal) onto direction (ux, uz)
const yawFace = (ux, uz) => Math.atan2(ux, uz);
// local (lx, lz) rotated by yaw
const rotXZ = (yaw, lx, lz) => [lx * Math.cos(yaw) + lz * Math.sin(yaw), -lx * Math.sin(yaw) + lz * Math.cos(yaw)];

class PointGrid { // cheap proximity test for placed props
  constructor(cell = 8) { this.cell = cell; this.m = new Map(); }
  k(i, j) { return i * 100003 + j; }
  add(x, z) { const k = this.k(Math.floor(x / this.cell), Math.floor(z / this.cell)); let a = this.m.get(k); if (!a) { a = []; this.m.set(k, a); } a.push(x, z); }
  near(x, z, r) {
    const c = this.cell, i0 = Math.floor((x - r) / c), i1 = Math.floor((x + r) / c), j0 = Math.floor((z - r) / c), j1 = Math.floor((z + r) / c);
    for (let i = i0; i <= i1; i++) for (let j = j0; j <= j1; j++) {
      const a = this.m.get(this.k(i, j)); if (!a) continue;
      for (let q = 0; q < a.length; q += 2) if ((a[q] - x) ** 2 + (a[q + 1] - z) ** 2 < r * r) return true;
    }
    return false;
  }
}

function distToInt(ints, s) {
  let best = Infinity, r = 0;
  for (const it of ints) { const d = Math.abs(it.s - s); if (d < best) { best = d; r = it.r; } }
  return [best, r];
}

const POINT_LIGHTS = 8; // real lamp PointLights (see header)
// distance (m) beyond which a furniture type is hidden (per-instance culling in the BatchedMesh)
const CULL = {
  cobra: 800, cobraRes: 650, pathOfGold: 700, doubleGlobe: 600, acorn: 400, mastPole: 700, mastArm: 700, pedPole: 550,
  sigHead: 480, pedHead: 260, signPost: 380, hydrant: 220, bollard: 200, trashCan: 230, bench: 230, bikeRack: 170,
  newsBox: 190, mailbox: 220, meter: 160, shelter: 480, utilityPole: 1000, utilityPoleTx: 1000, planter: 260,
  aframe: 150, cafeSet: 180, pot: 200,
};
const CHAIR_COLS = [0x1d4d3a, 0x161616, 0x8a1c1c, 0x2a3f6b, 0x7a7d80, 0x4b3621];
const BOARD_COLS = [0x1b1b1b, 0x5a3a22, 0x24452f, 0x7a1f1f, 0x1f3354];
const RANK = { motorway: 6, trunk: 5, primary: 4, secondary: 3, tertiary: 2, unclassified: 1, residential: 1, road: 1, living_street: 0 };
const PARKED = [['sedan', 0.34], ['hatchback', 0.18], ['suv', 0.27], ['pickup', 0.08], ['minivan', 0.08], ['taxi', 0.01], ['police_sedan', 0.004]];
const HYDRANT_TOPS = [[0xf2f2ee, 0.35], [0x1f4fa8, 0.25], [0xc62026, 0.18], [0x161616, 0.12], [0xe8b81e, 0.1]];
const NEWS_COLS = [0xc62026, 0x1f4fa8, 0xe8b81e, 0xf2f2ee, 0x1d6b3a, 0x161616, 0x2c8fc9];
const POSTERS = ['poster0', 'poster1', 'poster2', 'poster3', 'poster4', 'poster5'];

/** Instances of one furniture type, collected before the BatchedMesh is built. */
class FurnitureSet {
  constructor() { this.types = new Map(); this.n = 0; }
  add(type, x, y, z, yaw = 0, o = null) {
    let L = this.types.get(type); if (!L) { L = []; this.types.set(type, L); }
    L.push({ x, y, z, yaw, o }); this.n++;
  }
  count(type) { const L = this.types.get(type); return L ? L.length : 0; }
}

const roadRule = (v, dflt) => (typeof v === 'function' ? v : v instanceof RegExp ? (name) => v.test(name) : () => dflt);

export class Props {
  constructor({ net, raster, colliders, data, isLand, groundAt, seed = 7, streets = null, signStyle = null, treeRules = null, treeMix = null }) {
    this.net = net; this.raster = raster; this.colliders = colliders; this.data = data; this.isLand = isLand;
    this.gy = groundAt || (() => 0);
    this.seed = seed;
    const S = streets || {};
    this.streets = {
      lampStyle: typeof S.lampStyle === 'function' ? S.lampStyle : () => null,
      featureLamps: (S.featureLamps || []).filter((f) => f && f.road && f.style),
      boulevard: roadRule(S.boulevard, false), promenade: roadRule(S.promenade, false), utilityPoles: roadRule(S.utilityPoles, true),
    };
    this.signStyle = signStyle; this.treeRules = treeRules; this.treeMix = treeMix;
    this.rnd = mulberry32(seed);
    this.group = new THREE.Group();
    this.group.name = 'props';
    this.lampHeads = [];      // [x,y,z]
    this.signals = [];        // {heads:[{idx:[r,y,g], group}], peds:[{idx, group}], offset, state}
    this.pointLights = [];
    this._cullCam = new THREE.Vector3(1e9, 0, 0);
    this._cullTimer = 0;
    this.counts = {};
    this.F = new FurnitureSet();
    this.panels = [];         // sign faces: {x,y,z,yaw,w,h,a,b,glow,pitch}
    this.decals = [];         // road decals: {x,y,z,yaw,w,h,a,nrm}
    this.shelterMats = [];
    this.wireChunks = new Map();
  }

  build() {
    this.grid = new PointGrid(8);
    this.buildStreetLights();
    this.buildSignals();
    const c0 = this.colliders.boxes ? this.colliders.boxes.length : 0;
    this.buildTrees();
    // tree trunks (trees.js registers kind-2 colliders): signs / meters keep clear of them
    this.treeGrid = new PointGrid(8);
    if (this.colliders.boxes) for (let i = c0; i < this.colliders.boxes.length; i++) { const b = this.colliders.boxes[i]; if (b.kind === 2) this.treeGrid.add((b.minX + b.maxX) / 2, (b.minZ + b.maxZ) / 2); }
    this.poleGrid = new PointGrid(8);
    for (const p of this._sigPolePts || []) { this.grid.add(p[0], p[1]); this.poleGrid.add(p[0], p[1]); }
    this.buildParkedCars();
    this.buildFeatureLamps();
    this.buildStreetFurniture();
    this.buildFrontage();
    this.buildUtilityPoles();
    this.buildPlazaFurniture();
    this.finishFurniture();
    this.buildPointLights();
    return this.group;
  }

  rasterFree(x, z, r, bits) { return !this.raster || !this.raster.any(x, z, r, bits); }

  // -------------------------------------------------------------- street lights
  buildStreetLights() {
    const P = [];
    for (const R of this.net.list) {
      const st = R.st;
      if (R.filler || st.minor) continue;
      const name = R.name || '';
      const style = R.bridge ? 'cobra' : this.streets.lampStyle(name, R)
        || (/^(primary|secondary|tertiary|trunk|motorway)$/.test(st.base) ? 'cobra' : 'cobraRes');
      if (R.bridge) {
        if (!/^(motorway|trunk)/.test(st.cls)) continue;
        for (let s = 20, k = 0; s < R.pp.L - 5; s += 42, k++) {
          const [x, z, dx, dz, si, t] = samplePolyline(R.pts, R.pp, s);
          const h = R.hs[si] + (R.hs[si + 1] - R.hs[si]) * t;
          const g = R.hg ? R.hg[si] + (R.hg[si + 1] - R.hg[si]) * t : 0;
          if (h - g < 3) continue;
          const side = k % 2 ? 1 : -1, off = st.hw - 0.18;
          // a deck overhead (the Bay Bridge lower deck): no 9 m pole through it (bridge.js hangs ceiling lamps)
          let roof = false;
          this.net.deckHash.forEachNear(x - dz * off * side, z + dx * off * side, 0.5, (sg, tt) => { if (sg.ha + (sg.hb - sg.ha) * tt > h + 2.5) roof = true; });
          if (roof) continue;
          P.push([x - dz * off * side, h + 0.95, z + dx * off * side, dz * side, -dx * side, true, style]);
        }
        continue;
      }
      if (!st.sidewalk && st.markings !== 2) continue;
      const spacing = st.sidewalk ? 30 : 45;
      for (const side of [-1, 1]) {
        for (let s = side > 0 ? 8 : 23; s < R.pp.L - 3; s += spacing) {
          const [di, ri] = distToInt(R.ints, s);
          let ss = s;
          if (di < ri + 2) ss = s + (ri + 2 - di) + 0.5;
          if (ss > R.pp.L - 2) continue;
          const [x, z, dx, dz] = samplePolyline(R.pts, R.pp, ss);
          const off = st.hw + (st.sidewalk ? 0.55 : 1.2);
          const px = x - dz * off * side, pz = z + dx * off * side;
          if (!this.rasterFree(px, pz, 0.6, RBIT.BUILDING | RBIT.ROAD | RBIT.WATER)) continue;
          if (this.grid.near(px, pz, 12)) continue;
          this.grid.add(px, pz);
          // arm points toward the carriageway
          P.push([px, this.gy(px, pz) + 0.17, pz, dz * side, -dx * side, false, style]);
        }
      }
    }
    const n = P.length;
    this.counts.lamps = n;
    const styles = {};
    for (const p of P) {
      const style = p[6];
      styles[style] = (styles[style] || 0) + 1;
      // cobra arms reach over the road; a twin-globe crossarm (alongKerb) runs along the kerb
      const yaw = FURNITURE_META[style] && FURNITURE_META[style].alongKerb ? yawX(-p[4], p[3]) : yawX(p[3], p[4]);
      this.F.add(style, p[0], p[1], p[2], yaw);
      const L = FURNITURE_META[style].light, [lx, lz] = rotXZ(yaw, L[0], L[2]);
      this.lampHeads.push([p[0] + lx, p[1] + L[1], p[2] + lz]);
      if (!p[5]) this.colliders.add(p[0] - 0.22, p[1] - 0.2, p[0] + 0.22, p[1] + 8, p[2] - 0.22, p[2] + 0.22, 3);
    }
    this.counts.lampStyles = styles;
  }

  /**
   * Extra lamp standards (streets.featureLamps: [{ road, style, step }]) along streets whose split carriageways
   * defeat the generic pass (e.g. a grand boulevard's ornate standards).
   */
  buildFeatureLamps() {
    const rules = this.streets.featureLamps;
    if (!rules.length) return;
    const lampGrid = new PointGrid(16);
    for (const h of this.lampHeads) lampGrid.add(h[0], h[2]);
    for (const rule of rules) this.buildFeatureLampRule(rule, lampGrid);
  }
  buildFeatureLampRule(rule, lampGrid) {
    const match = roadRule(rule.road, false), style = rule.style, step = rule.step || 28;
    const meta = FURNITURE_META[style] || FURNITURE_META.pathOfGold;
    let n = 0;
    for (const R of this.net.list) {
      const st = R.st;
      if (R.filler || R.bridge || st.minor || !match(R.name || '', R)) continue;
      for (const side of [-1, 1]) for (let s = side > 0 ? 6 : 20; s < R.pp.L - 3; s += step) {
        const [di, ri] = distToInt(R.ints, s);
        if (di < ri + 2) continue;
        const [x, z, dx, dz] = samplePolyline(R.pts, R.pp, s);
        for (const off of [st.hw + 0.55, st.hw + 1.1, st.hw + 1.8]) {
          const px = x - dz * off * side, pz = z + dx * off * side;
          if (lampGrid.near(px, pz, 14) || !this.spot(px, pz, 0.5, 1.2, false)) continue;
          if (this.raster && this.raster.any(px, pz, 0.4, RBIT.ROAD | RBIT.SERVICE)) continue;
          const y = this.baseY(px, pz);
          this.F.add(style, px, y, pz, yawX(dz * side, -dx * side));
          this.lampHeads.push([px, y + meta.light[1], pz]);
          lampGrid.add(px, pz); this.grid.add(px, pz);
          this.colliders.add(px - 0.3, y - 0.2, px + 0.3, y + 7.5, pz - 0.3, pz + 0.3, 3);
          n++;
          break;
        }
      }
    }
    this.counts.lamps += n; this.counts.lampStyles[style] = (this.counts.lampStyles[style] || 0) + n;
  }

  // -------------------------------------------------------------- traffic signals
  buildSignals() {
    const pts = (this.data.signals || []).filter((p) => p && isFinite(p[0]) && isFinite(p[1]));
    // cluster nodes of the same junction (traffic.js repeats this exactly: infos[i] <-> signals[i])
    const clusters = [];
    for (const p of pts) {
      let c = clusters.find((c) => (c.x - p[0]) ** 2 + (c.z - p[1]) ** 2 < 22 * 22);
      if (c) { c.n++; c.x += (p[0] - c.x) / c.n; c.z += (p[1] - c.z) / c.n; c.pts.push(p); }
      else clusters.push({ x: p[0], z: p[1], n: 1, pts: [p] });
    }
    this.sigGrid = new PointGrid(32);
    for (const p of pts) this.sigGrid.add(p[0], p[1]);
    this._sigPolePts = [];
    const heads = [], peds = [];
    for (const c of clusters) {
      const found = [];
      for (const p of c.pts) {
        this.net.segHash.forEachNear(p[0], p[1], 3, (seg) => {
          const R = seg.road;
          if (!R || R.st.minor || R.st.base === 'service') return;
          const dx = seg.bx - seg.ax, dz = seg.bz - seg.az, L = Math.hypot(dx, dz) || 1;
          found.push({ R, hw: R.st.hw, dir: [dx / L, dz / L], ow: R.st.oneway, seg, name: (R.r && R.r.n) || '' });
        });
      }
      if (!found.length) continue;
      found.sort((a, b) => b.hw - a.hw);
      const A = found[0];
      const B = found.find((f) => Math.abs(f.dir[0] * A.dir[0] + f.dir[1] * A.dir[1]) < 0.8);
      if (!B) continue;
      // junction centre: intersection of the two centre lines
      const cen = lineIntersect(A.seg, B.seg) || [c.x, c.z];
      if ((cen[0] - c.x) ** 2 + (cen[1] - c.z) ** 2 > 30 * 30) continue;
      const sig = { heads: [], peds: [], offset: this.rnd() * 60, state: -1, ped: -1 };
      this.sigGrid.add(cen[0], cen[1]);
      const addPole = (road, other, sgn, group) => {
        // pole on the far-right corner for traffic travelling sgn*dir along `road`
        const f = [road.dir[0] * sgn, road.dir[1] * sgn], r = [-f[1], f[0]];
        const px = cen[0] + r[0] * (road.hw + 0.8) + f[0] * (other.hw + 1.2);
        const pz = cen[1] + r[1] * (road.hw + 0.8) + f[1] * (other.hw + 1.2);
        // never on a carriageway: a third road through the junction (a diagonal avenue) or a wider piece of the
        // crossing street can put the computed corner inside the asphalt, where the pole would block traffic
        if (!this.rasterFree(px, pz, 0.3, RBIT.BUILDING | RBIT.WATER) || !this.rasterFree(px, pz, 0.45, RBIT.ROAD | RBIT.SERVICE)) return;
        const by = this.gy(px, pz) + 0.17; // pole base (sidewalk top)
        this._sigPolePts.push([px, pz, by, 0]);
        const armLen = Math.max(2.5, Math.min(road.ow ? road.hw * 1.6 : road.hw * 0.95, 11));
        const armYaw = yawX(-r[0], -r[1]), face = [-f[0], -f[1]];
        this._sigPolePts[this._sigPolePts.length - 1][3] = armLen >= 6 ? 0.2 : 0.1; // pole radius at blade height
        if (armLen >= 6) { // mast arm over wide approaches, heads hung above each lane group
          this.F.add('mastPole', px, by, pz, armYaw);
          this.F.add('mastArm', px, by + 6.3, pz, armYaw, { sx: armLen });
          this.colliders.add(px - 0.3, by - 0.2, px + 0.3, by + 7.0, pz - 0.3, pz + 0.3, 3);
          const nh = Math.max(1, Math.min(3, Math.round(armLen / 3.4)));
          for (let k = 0; k < nh; k++) {
            const d = armLen - 0.6 - k * 3.3;
            if (d < 1) break;
            heads.push({ x: px - r[0] * d, z: pz - r[1] * d, y: by + 6.2, face, group, sig });
          }
          heads.push({ x: px - r[0] * 0.5, z: pz - r[1] * 0.5, y: by + 4.6, face, group, sig }); // near-side pole head
          if (other.name && armLen >= 5) { // internally lit street-name sign (cross street) on the arm
            const d = Math.min(2.4, armLen * 0.4);
            this.panels.push({ x: px - r[0] * d, y: by + 6.3 + 0.36, z: pz - r[1] * d, yaw: yawFace(face[0], face[1]), w: 2.1, h: 0.4, a: 'name:' + other.name, b: 'name:' + other.name, glow: 1 });
          }
        } else { // pedestal pole with a side-mounted head
          this.F.add('pedPole', px, by, pz, armYaw);
          this.colliders.add(px - 0.2, by - 0.2, px + 0.2, by + 4.7, pz - 0.2, pz + 0.2, 3);
          heads.push({ x: px - r[0] * 0.46, z: pz - r[1] * 0.46, y: by + 4.45, face, group, sig });
        }
        // pedestrian heads for the two crosswalks at this corner: across `road` (walk with `other`), across `other`
        peds.push({ x: px, y: by, z: pz, face: [-r[0], -r[1]], group: 1 - group, sig });
        peds.push({ x: px, y: by, z: pz, face, group, sig });
      };
      addPole(A, B, 1, 0); addPole(A, B, -1, 0); addPole(B, A, 1, 1); addPole(B, A, -1, 1);
      this.signals.push(sig);
    }
    this.counts.signals = this.signals.length;
    if (!heads.length) return;
    for (const h of heads) this.F.add('sigHead', h.x, h.y, h.z, yawNegZ(h.face[0], h.face[1]));
    for (const p of peds) this.F.add('pedHead', p.x, p.y, p.z, yawNegZ(p.face[0], p.face[1]));
    const bulbGeo = new THREE.CircleGeometry(0.105, 8).rotateY(Math.PI);
    this.bulbMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
    const bulbs = new THREE.InstancedMesh(bulbGeo, this.bulbMat, heads.length * 3 + peds.length);
    bulbs.name = 'signal-lenses';
    const off = new THREE.Color(0.025, 0.025, 0.025);
    let bi = 0;
    for (const h of heads) {
      const yaw = yawNegZ(h.face[0], h.face[1]);
      const idx = [];
      for (let k = 0; k < 3; k++) {
        setTRS(bulbs, bi, h.x + h.face[0] * 0.128, h.y - 0.6 + 0.335 * (1 - k), h.z + h.face[1] * 0.128, yaw);
        bulbs.setColorAt(bi, off);
        idx.push(bi++);
      }
      h.sig.heads.push({ idx, group: h.group });
    }
    for (const p of peds) {
      setTRS(bulbs, bi, p.x + p.face[0] * 0.525, p.y + 2.75, p.z + p.face[1] * 0.525, yawNegZ(p.face[0], p.face[1]), 1.2, 1.2, 1);
      bulbs.setColorAt(bi, off);
      p.sig.peds.push({ idx: bi++, group: p.group });
    }
    this.bulbs = bulbs;
    bulbs.instanceMatrix.needsUpdate = true; bulbs.computeBoundingSphere(); bulbs.castShadow = false;
    if (bulbs.instanceColor) bulbs.instanceColor.needsUpdate = true;
    this.group.add(bulbs);
    this.counts.signalHeads = heads.length; this.counts.pedHeads = peds.length;
    this._colors = {
      off: new THREE.Color(0.025, 0.025, 0.025),
      red: new THREE.Color(5.0, 0.25, 0.12), yellow: new THREE.Color(4.5, 2.6, 0.2), green: new THREE.Color(0.2, 4.0, 1.6),
      walk: new THREE.Color(1.7, 1.8, 1.7), dont: new THREE.Color(3.0, 1.05, 0.08),
    };
  }

  updateSignals(time) {
    if (!this.bulbs) return;
    const C = this._colors;
    let dirty = false;
    for (const s of this.signals) {
      const t = (time + s.offset) % 40;
      // 0..16 A green, 16..19 A yellow, 19..20 all red, 20..36 B green, 36..39 B yellow, 39..40 all red
      const st = t < 16 ? 0 : t < 19 ? 1 : t < 20 ? 2 : t < 36 ? 3 : t < 39 ? 4 : 5;
      if (st !== s.state) {
        s.state = st; dirty = true;
        for (const h of s.heads) {
          const a = h.group === 0;
          let light; // 0 red 1 yellow 2 green
          if (st === 0) light = a ? 2 : 0; else if (st === 1) light = a ? 1 : 0; else if (st === 3) light = a ? 0 : 2;
          else if (st === 4) light = a ? 0 : 1; else light = 0;
          this.bulbs.setColorAt(h.idx[0], light === 0 ? C.red : C.off);
          this.bulbs.setColorAt(h.idx[1], light === 1 ? C.yellow : C.off);
          this.bulbs.setColorAt(h.idx[2], light === 2 ? C.green : C.off);
        }
      }
      // pedestrians walk with their parallel traffic: WALK, then flashing hand until the green ends
      const blink = Math.floor(t * 2) % 2 ? 1 : 2;
      const pa = t < 7 ? 0 : t < 16 ? blink : 1, pb = t >= 20 && t < 27 ? 0 : t >= 27 && t < 36 ? blink : 1;
      const key = pa * 3 + pb;
      if (key !== s.ped && s.peds.length) {
        s.ped = key; dirty = true;
        for (const p of s.peds) { const v = p.group === 0 ? pa : pb; this.bulbs.setColorAt(p.idx, v === 0 ? C.walk : v === 1 ? C.dont : C.off); }
      }
    }
    if (dirty && this.bulbs.instanceColor) this.bulbs.instanceColor.needsUpdate = true;
  }

  // -------------------------------------------------------------- trees
  buildTrees() {
    const T = []; // [x, z, scale, hue]
    const rnd = this.rnd;
    const blocked = RBIT.BUILDING | RBIT.ROAD | RBIT.WATER | RBIT.SERVICE;
    const tryAdd = (x, z, s, minGap = 4, kind = 'osm', ctx = null) => {
      if (!isFinite(x) || !isFinite(z)) return false;
      if (!this.rasterFree(x, z, 0.8, blocked)) return false;
      if (this.raster && this.raster.any(x, z, 1.6, RBIT.BUILDING)) return false;
      if (this.grid.near(x, z, minGap)) return false;
      if (this.isLand && !this.isLand(x, z)) return false;
      this.grid.add(x, z);
      T.push([x, z, s, rnd(), kind, ctx]);
      return true;
    };
    for (const p of this.data.trees || []) if (p) tryAdd(p[0], p[1], 0.8 + rnd() * 0.5, 2.5, 'osm', { road: null });
    // street trees
    for (const R of this.net.list) {
      const st = R.st;
      if (R.filler || R.bridge || st.minor || st.sidewalk < 2.5) continue;
      const prob = { residential: 0.85, living_street: 0.8, tertiary: 0.7, unclassified: 0.7, secondary: 0.55, primary: 0.4, trunk: 0.3 }[st.base] ?? 0;
      if (!prob) continue;
      for (const side of [-1, 1]) {
        if (rnd() > prob + 0.15) continue;
        for (let s = 5 + rnd() * 5; s < R.pp.L - 4; s += 8 + rnd() * 4) {
          if (rnd() > prob) continue;
          const [di, ri] = distToInt(R.ints, s);
          if (di < ri + 5) continue;
          const [x, z, dx, dz] = samplePolyline(R.pts, R.pp, s);
          const off = st.hw + 1.0;
          tryAdd(x - dz * off * side, z + dx * off * side, 0.7 + rnd() * 0.45, 4, 'street', { road: R.name || R.n || null, cls: st.base, side, dx, dz });
        }
      }
    }
    // parks
    const dens = { park: 1 / 110, garden: 1 / 90, wood: 1 / 30, forest: 1 / 30, scrub: 1 / 80, grass: 1 / 300, village_green: 1 / 200, recreation_ground: 1 / 250, common: 1 / 150, playground: 1 / 400, dog_park: 1 / 300 };
    for (const pk of this.data.parks || []) {
      if (!pk || !pk.p || pk.p.length < 3) continue;
      const d = dens[pk.k] ?? 1 / 200;
      const area = Math.abs(signedArea(pk.p));
      let n = Math.min(450, Math.floor(area * d));
      if (n < 1 && area > 60 && rnd() < area * d) n = 1;
      const bb = ringBounds(pk.p);
      for (let i = 0, tries = 0; i < n && tries < n * 6; tries++) {
        const x = bb.minX + rnd() * (bb.maxX - bb.minX), z = bb.minZ + rnd() * (bb.maxZ - bb.minZ);
        if (!pointInPoly(x, z, pk.p, pk.hl)) continue;
        if (this.raster && this.raster.any(x, z, 1.2, RBIT.SIDEWALK)) continue;
        if (tryAdd(x, z, 0.8 + rnd() * 0.7, 3.5, 'park', { park: pk.k || null })) i++;
      }
    }
    // ---- rendering is delegated to trees.js (species, LOD, wind); placement stays here
    const placements = T.map(([x, z, s, hue, kind, ctx]) => ({ x, z, s, hue, kind, ctx }));
    this.trees = buildTreeMeshes(placements, { groundAt: this.gy, colliders: this.colliders, seed: 1234, rules: this.treeRules || [], mix: this.treeMix });
    this.counts.trees = this.trees.count;
    if (this.trees.group) this.group.add(this.trees.group);
  }

  // -------------------------------------------------------------- parked cars
  buildParkedCars() {
    const C = [];
    const rnd = this.rnd;
    for (const R of this.net.list) {
      const st = R.st;
      if (R.filler || R.bridge || st.minor || !(st.parkW > 0)) continue;
      if (this.streets.boulevard(R.name || '', R)) continue; // car-free boulevard
      for (const side of [-1, 1]) {
        if (rnd() < 0.12) continue;
        const occ = 0.45 + rnd() * 0.4;
        for (let s = 6 + rnd() * 4; s < R.pp.L - 5; s += 6.3) {
          if (C.length > 900) break;
          if (rnd() > occ) continue;
          const [di, ri] = distToInt(R.ints, s);
          if (di < ri + 7.5) continue;
          const [x, z, dx, dz] = samplePolyline(R.pts, R.pp, s);
          const off = st.hw - st.parkW * 0.5 - 0.05;
          const px = x - dz * off * side, pz = z + dx * off * side;
          if (this.raster && this.raster.any(px, pz, 1.2, RBIT.SERVICE | RBIT.BUILDING | RBIT.WATER)) continue;
          if (this.grid.near(px, pz, 2.5)) continue;
          const fwd = st.oneway || side > 0 ? 1 : -1;
          C.push([px, pz, dx * fwd, dz * fwd]);
        }
      }
    }
    const n = C.length;
    this.counts.parkedCars = n;
    if (!n) return;
    // one BatchedMesh: every parked type at two LODs (swapped by distance in update())
    const types = PARKED.map(([t]) => t), wsum = PARKED.reduce((a, b) => a + b[1], 0);
    const geos = types.map((t) => [makeVehicleGeometry(t), makeVehicleGeometry(t, { lod: 1 })]);
    for (const pair of geos) for (const g of pair) { g.deleteAttribute('uv'); }
    let maxV = 0; for (const pair of geos) for (const g of pair) maxV += g.attributes.position.count;
    const mat = patchParkedMaterial(createVehicleMaterial({ brake: false, lamps: false }));
    this.parkedMat = mat;
    const cars = new THREE.BatchedMesh(n, maxV, maxV, mat);
    cars.name = 'parked-cars';
    const gids = geos.map(([a, b]) => [cars.addGeometry(a), cars.addGeometry(b)]);
    const col = new THREE.Color();
    const _qp = new THREE.Quaternion(), XA = new THREE.Vector3(1, 0, 0);
    const P = this.parked = { mesh: cars, ids: new Int32Array(n), type: new Uint8Array(n), x: new Float32Array(n), y: new Float32Array(n), z: new Float32Array(n), st: new Int8Array(n).fill(-2), gids };
    C.forEach(([x, z, fx, fz], i) => {
      let r = rnd() * wsum, ti = 0; while (ti < PARKED.length - 1 && r > PARKED[ti][1]) { r -= PARKED[ti][1]; ti++; }
      const spec = VEHICLE_SPECS[types[ti]];
      const yaw = yawNegZ(fx, fz);
      // sit on the street, pitched to its grade (nose up when the street climbs ahead)
      const y = this.gy(x, z) + 0.05;
      const grade = (this.gy(x + fx * 2, z + fz * 2) - this.gy(x - fx * 2, z - fz * 2)) / 4;
      _q.setFromAxisAngle(UP, yaw).multiply(_qp.setFromAxisAngle(XA, Math.atan(grade)));
      _m.compose(_p.set(x, y, z), _q, _s.set(1, 1, 1));
      const id = cars.addInstance(gids[ti][1]);
      cars.setMatrixAt(id, _m);
      const kind = types[ti] === 'taxi' ? 'taxi' : types[ti].startsWith('police') ? 'police' : 'car';
      cars.setColorAt(id, pickVehicleColor(rnd, kind, col));
      P.ids[i] = id; P.type[i] = ti; P.x[i] = x; P.y[i] = y; P.z[i] = z;
      // colliders: a car parked at an angle to the axes is reported as thin AABB slices along its length (one AABB of
      // a 30-degree car bulged ~0.7 m into the driving lane); axis-aligned cars keep one box. Shrunk a little.
      const hl = spec.L / 2 - 0.2, hw = spec.W / 2 - 0.05, afx = Math.abs(fx), afz = Math.abs(fz);
      const dy = Math.abs(grade) * hl;
      const ns = Math.min(afx, afz) < 0.06 ? 1 : 3, sl = (2 * hl) / ns;
      const sx = afx * sl * 0.5 + afz * hw, sz = afz * sl * 0.5 + afx * hw;
      for (let k = 0; k < ns; k++) {
        const o = -hl + sl * (k + 0.5), cx = x + fx * o, cz = z + fz * o;
        this.colliders.add(cx - sx, y - dy, cx + sx, y + spec.H + dy, cz - sz, cz + sz, 4);
      }
    });
    cars.perObjectFrustumCulled = true; cars.sortObjects = false; cars.frustumCulled = false;
    cars.castShadow = true; cars.receiveShadow = true;
    this.group.add(cars);
  }

  // -------------------------------------------------------------- street furniture
  /** Free sidewalk spot: not on a road / driveway / water, clear of buildings and other props. */
  spot(x, z, rb = 0.4, gap = 0.8, walk = true) {
    if (!Number.isFinite(x) || !Number.isFinite(z)) return false;
    const r = this.raster;
    if (r) {
      if (r.get(x, z) & (RBIT.ROAD | RBIT.SERVICE | RBIT.WATER)) return false;
      if (r.any(x, z, rb, RBIT.BUILDING)) return false;
      if (walk && this._swBits && !(r.get(x, z) & RBIT.SIDEWALK)) return false;
    }
    if (gap > 0 && this.grid.near(x, z, gap)) return false;
    if (this.isLand && !this.isLand(x, z)) return false;
    return true;
  }
  baseY(x, z) { return this.gy(x, z) + (this.raster && (this.raster.get(x, z) & RBIT.SIDEWALK) ? 0.17 : 0.06); }
  /** Post + sign face (+ optional plaque under it). yaw: panel front normal. */
  sign(x, y, z, key, yaw, w, h, cy, back = 'back', plaque = null) {
    if ((this.treeGrid && this.treeGrid.near(x, z, 1.3)) || (this.poleGrid && this.poleGrid.near(x, z, 1.0))) return false;
    this.F.add('signPost', x, y, z, 0, { sy: cy + h / 2 + 0.04 });
    const fx = Math.sin(yaw) * 0.045, fz = Math.cos(yaw) * 0.045;
    this.panels.push({ x: x + fx, y: y + cy, z: z + fz, yaw, w, h, a: key, b: back });
    if (plaque) this.panels.push({ x: x + fx, y: y + cy - h / 2 - 0.13, z: z + fz, yaw, w: 0.46, h: 0.155, a: plaque, b: 'back' });
    this.grid.add(x, z);
    return true;
  }
  /** Nearest signal pole [x, z, baseY, radius] within r of (x, z), or null. */
  nearestPole(x, z, r) {
    let best = null, bd = r * r;
    for (const p of this._sigPolePts || []) { const d = (p[0] - x) ** 2 + (p[1] - z) ** 2; if (d < bd) { bd = d; best = p; } }
    return best;
  }
  collide(x, y, z, hx, hz, h, kind = 3) { this.colliders.add(x - hx, y - 0.2, x + hx, y + h, z - hz, z + hz, kind); }
  decal(x, z, yaw, w, h, key) {
    const e = 0.6, gx = (this.gy(x + e, z) - this.gy(x - e, z)) / (2 * e), gz = (this.gy(x, z + e) - this.gy(x, z - e)) / (2 * e);
    this.decals.push({ x, y: this.gy(x, z) + 0.05 + 0.012, z, yaw, w, h, a: key, nrm: [-gx, 1, -gz] });
  }

  buildStreetFurniture() {
    const rnd = mulberry32((this.seed * 7919) ^ 0x51f3), F = this.F, net = this.net;
    const r = this.raster;
    this._swBits = false;
    if (r) for (let i = 0; i < r.data.length; i += 61) if (r.data[i] & RBIT.SIDEWALK) { this._swBits = true; break; }
    const at = (R, s, lat) => {
      const [x, z, dx, dz] = samplePolyline(R.pts, R.pp, clamp(s, 0, R.pp.L));
      return { x: x - dz * lat, z: z + dx * lat, dx, dz, cx: x, cz: z };
    };
    const pick = (list) => { let t = 0; for (const e of list) t += e[1]; let u = rnd() * t; for (const e of list) { u -= e[1]; if (u <= 0) return e[0]; } return list[list.length - 1][0]; };
    const col = (hex) => new THREE.Color(hex);
    const cnt = { stop: 0, blades: 0, oneway: 0, speed: 0, parkSigns: 0, hydrants: 0, trash: 0, mailboxes: 0, newsBoxes: 0, bikeRacks: 0, benches: 0, planters: 0, meters: 0, bollards: 0, shelters: 0, manholes: 0, drains: 0 };
    const nameGrid = new PointGrid(16);
    const shelters = [], speeds = [];
    const toRoadYaw = (p, side) => yawNegZ(side * p.dz, -side * p.dx);   // local -Z toward the carriageway
    const awayYaw = (p, side) => yawNegZ(-side * p.dz, side * p.dx);     // local -Z toward the buildings

    for (const R of net.list) {
      const st = R.st;
      if (R.filler || R.bridge || st.minor) continue;
      const L = R.pp.L, name = R.name || '', hw = st.hw, sw = st.sidewalk || 0;
      const drivable = st.base !== 'service' && st.base !== 'busway';
      const commercial = /^(primary|secondary|tertiary|unclassified)$/.test(st.base);
      const blvd = this.streets.boulevard(name, R), prom = this.streets.promenade(name, R);
      const rk = RANK[st.base] ?? 1;

      // ---------------- per intersection
      for (const it of R.ints) {
        const c = at(R, it.s, 0);
        const nearSig = this.sigGrid && this.sigGrid.near(c.cx, c.cz, 26);
        let maxOther = -1, cross = null;
        net.segHash.forEachNear(c.cx, c.cz, 1.2, (seg) => {
          const r2 = seg.road; if (!r2 || r2.st.minor) return;
          const ddx = seg.bx - seg.ax, ddz = seg.bz - seg.az, l = Math.hypot(ddx, ddz) || 1;
          if (Math.abs((ddx * c.dx + ddz * c.dz) / l) > 0.9) return; // parallel: this road
          if (r2.st.base !== 'service') maxOther = Math.max(maxOther, RANK[r2.st.base] ?? 1);
          const nm = (r2.r && r2.r.n) || '';
          if (!cross || (nm && !cross.name)) cross = { dx: ddx / l, dz: ddz / l, name: nm };
        });
        // stop signs (minor approach / low-class all-way, as traffic.js controls them)
        if (drivable && !nearSig && maxOther >= 0 && rk < 5 && (rk < maxOther || (rk <= 2 && maxOther <= 2))) {
          const allWay = rk <= 2 && maxOther <= 2;
          for (const dir of st.oneway ? [1] : [1, -1]) {
            const s = it.s - dir * (it.r + 4.6); if (s < 1 || s > L - 1) continue;
            const p = at(R, s, dir * (hw + 0.55));
            if (!this.spot(p.x, p.z, 0.3, 0.5, false)) continue;
            this.sign(p.x, this.baseY(p.x, p.z), p.z, 'stop', yawFace(-p.dx * dir, -p.dz * dir), 0.76, 0.76, 2.5, 'back', allWay ? 'allway' : null);
            cnt.stop++;
          }
        }
        // green street-name blades on a corner post (one per junction)
        if (name && cross && cross.name && cross.name !== name && !nameGrid.near(c.cx, c.cz, 20)) {
          // a signal pole on a corner of this junction carries the blades (bracket-mounted, off the pole)
          const pole = this.nearestPole(c.cx, c.cz, it.r + hw + 4);
          if (pole) {
            const [px, pz, by, pr] = pole;
            const ux = c.cx - px, uz = c.cz - pz; // toward the junction centre: blades hang away from it
            for (const [d, k, y] of [[[c.dx, c.dz], name, 3.0], [[cross.dx, cross.dz], cross.name, 3.22]]) {
              const sg = d[0] * ux + d[1] * uz > 0 ? -1 : 1, o = pr + 0.47;
              this.panels.push({ x: px + d[0] * o * sg, y: by + y, z: pz + d[1] * o * sg, yaw: yawFace(-d[1], d[0]), w: 0.9, h: 0.17, a: 'name:' + k, b: 'name:' + k });
            }
            nameGrid.add(c.cx, c.cz); cnt.blades++; cnt.bladesOnSignals = (cnt.bladesOnSignals || 0) + 1;
          } else for (const [ds, side] of [[1, 1], [-1, -1], [1, -1], [-1, 1]]) {
            const s = it.s + ds * (it.r + 1.0); if (s < 0.5 || s > L - 0.5) continue;
            const p = at(R, s, side * (hw + 0.6));
            if (!this.spot(p.x, p.z, 0.3, 0.6, sw > 0) || this.poleGrid.near(p.x, p.z, 2.5) || this.treeGrid.near(p.x, p.z, 1.4)) continue;
            const y = this.baseY(p.x, p.z);
            F.add('signPost', p.x, y, p.z, 0, { sy: 3.36 });
            this.panels.push({ x: p.x, y: y + 3.0, z: p.z, yaw: yawFace(-p.dz, p.dx), w: 0.9, h: 0.17, a: 'name:' + name, b: 'name:' + name });
            this.panels.push({ x: p.x, y: y + 3.22, z: p.z, yaw: yawFace(-cross.dz, cross.dx), w: 0.9, h: 0.17, a: 'name:' + cross.name, b: 'name:' + cross.name });
            this.grid.add(p.x, p.z); nameGrid.add(c.cx, c.cz); cnt.blades++;
            break;
          }
        }
        // ONE WAY (parallel to the street, arrow along the traffic) for the cross street
        if (st.oneway && drivable && st.base !== 'motorway' && !st.link && maxOther >= 0) {
          const s = it.s + it.r + 1.6;
          if (s < L - 0.5) {
            const p = at(R, s, hw + 0.55);
            if (this.spot(p.x, p.z, 0.3, 0.5, false)) { this.sign(p.x, this.baseY(p.x, p.z), p.z, 'oneR', yawFace(-p.dz, p.dx), 0.9, 0.3, 2.45, 'oneL'); cnt.oneway++; }
          }
        }
        // speed limit after the junction, one per ~400 m per direction of a named street
        if (drivable && commercial && sw > 0) {
          for (const dir of st.oneway ? [1] : [1, -1]) {
            const s = it.s + dir * (it.r + 16); if (s < 2 || s > L - 2) continue;
            const p = at(R, s, dir * (hw + 0.55));
            if (speeds.some((q) => q.name === name && (q.x - p.x) ** 2 + (q.z - p.z) ** 2 < 380 * 380 && q.dx * p.dx * dir + q.dz * p.dz * dir > 0.5)) continue;
            if (!this.spot(p.x, p.z, 0.3, 0.6)) continue;
            this.sign(p.x, this.baseY(p.x, p.z), p.z, rk >= 3 ? 'speed30' : 'speed25', yawFace(-p.dx * dir, -p.dz * dir), 0.6, 0.75, 2.2);
            speeds.push({ name, x: p.x, z: p.z, dx: p.dx * dir, dz: p.dz * dir }); cnt.speed++;
          }
        }
        // bus shelter on the far side of the junction (~every 300 m per direction on arterials)
        if (drivable && sw >= 2.7 && /^(primary|secondary|tertiary)$/.test(st.base)) {
          for (const dir of st.oneway ? [1] : [1, -1]) this.tryShelter(R, it.s + dir * (it.r + 10), dir, at, shelters, cnt, rnd);
        }
        // storm drains in the gutter before the corners
        if (drivable && sw > 0) for (const side of [-1, 1]) for (const ds of [-1, 1]) {
          if (rnd() > 0.42) continue;
          const s = it.s + ds * (it.r + 4.8); if (s < 1 || s > L - 1) continue;
          const p = at(R, s, side * (hw - 0.3));
          if (r && !(r.get(p.x, p.z) & RBIT.ROAD)) continue;
          this.decal(p.x, p.z, yawX(p.dx, p.dz), 0.95, 0.42, 'drain'); cnt.drains++;
        }
        if (drivable && rnd() < 0.3) { this.decal(c.cx + (rnd() - 0.5) * 2, c.cz + (rnd() - 0.5) * 2, rnd() * 6.28, 0.76, 0.76, 'manhole'); cnt.manholes++; }
        // corner furniture
        if (sw <= 0) continue;
        for (const side of [-1, 1]) for (const ds of [-1, 1]) {
          const S = (d) => it.s + ds * (it.r + d);
          const inRange = (s) => s > 0.5 && s < L - 0.5;
          const kerb = (d, lat) => { const s = S(d); return inRange(s) ? at(R, s, side * (hw + lat)) : null; };
          let p;
          if (rnd() < 0.32 && (p = kerb(3.2, 0.6)) && this.spot(p.x, p.z, 0.3, 0.9)) {
            const y = this.baseY(p.x, p.z);
            F.add('hydrant', p.x, y, p.z, toRoadYaw(p, side), { color: col(pick(HYDRANT_TOPS)) });
            this.grid.add(p.x, p.z); this.collide(p.x, y, p.z, 0.18, 0.18, 0.85); cnt.hydrants++;
          }
          if (!commercial) continue;
          if (rnd() < (blvd || prom ? 0.8 : 0.35) && (p = kerb(2.0, 0.65)) && this.spot(p.x, p.z, 0.35, 0.9)) {
            const y = this.baseY(p.x, p.z);
            F.add('trashCan', p.x, y, p.z, rnd() * 6.28, { color: col(rnd() < 0.8 ? 0x14402a : 0x1b1d1f) });
            this.grid.add(p.x, p.z); this.collide(p.x, y, p.z, 0.3, 0.3, 1.04); cnt.trash++;
          }
          if (rnd() < 0.07 && (p = kerb(4.4, 0.6)) && this.spot(p.x, p.z, 0.4, 1.0)) {
            const y = this.baseY(p.x, p.z);
            F.add('mailbox', p.x, y, p.z, awayYaw(p, side));
            this.grid.add(p.x, p.z); this.collide(p.x, y, p.z, 0.3, 0.3, 1.25); cnt.mailboxes++;
          }
          if (rnd() < 0.12) {
            const k = 2 + Math.floor(rnd() * 3);
            for (let j = 0; j < k; j++) {
              if (!(p = kerb(5.8 + j * 0.56, 0.55)) || !this.spot(p.x, p.z, 0.3, 0.45)) continue;
              const y = this.baseY(p.x, p.z);
              F.add('newsBox', p.x, y, p.z, awayYaw(p, side), { color: col(NEWS_COLS[Math.floor(rnd() * NEWS_COLS.length)]) });
              this.grid.add(p.x, p.z); this.collide(p.x, y, p.z, 0.25, 0.25, 1.1); cnt.newsBoxes++;
            }
          }
          if (rnd() < (blvd ? 0.4 : 0.14)) {
            const k = 1 + Math.floor(rnd() * 3);
            for (let j = 0; j < k; j++) {
              if (!(p = kerb(8 + j * 1.1, 0.7)) || !this.spot(p.x, p.z, 0.3, 0.6)) continue;
              F.add('bikeRack', p.x, this.baseY(p.x, p.z), p.z, yawX(p.dx, p.dz));
              this.grid.add(p.x, p.z); cnt.bikeRacks++;
            }
          }
          if (blvd || prom) for (let j = 0; j < 3; j++) {
            if (!(p = kerb(1.0 + j * 1.4, 0.35)) || !this.spot(p.x, p.z, 0.15, 0.5)) continue;
            const y = this.baseY(p.x, p.z);
            F.add('bollard', p.x, y, p.z, 0); this.grid.add(p.x, p.z); this.collide(p.x, y, p.z, 0.1, 0.1, 0.95); cnt.bollards++;
          }
        }
      }

      if (sw <= 0) { if (drivable) this.roadDecals(R, at, rnd, cnt); continue; }
      // ---------------- mid-block
      const ints = R.ints.map((it) => it.s).sort((a, b) => a - b);
      const bounds = [0, ...ints, L];
      for (let k = 0; k < bounds.length - 1; k++) { // hydrant mid-block on long blocks
        if (bounds[k + 1] - bounds[k] < 120 || rnd() > 0.6) continue;
        const side = rnd() < 0.5 ? -1 : 1, p = at(R, (bounds[k] + bounds[k + 1]) / 2, side * (hw + 0.6));
        if (!this.spot(p.x, p.z, 0.3, 0.9)) continue;
        const y = this.baseY(p.x, p.z);
        F.add('hydrant', p.x, y, p.z, toRoadYaw(p, side), { color: col(pick(HYDRANT_TOPS)) });
        this.grid.add(p.x, p.z); this.collide(p.x, y, p.z, 0.18, 0.18, 0.85); cnt.hydrants++;
      }
      for (const side of [-1, 1]) {
        // parking meters, one per stall on commercial parking lanes
        if (/^(secondary|tertiary)$/.test(st.base) && st.parkW > 0 && !blvd) {
          for (let s = 11.15; s < L - 5; s += 6.3) {
            const [di, ri] = distToInt(R.ints, s);
            if (di < ri + 7.5 || rnd() > 0.6) continue;
            const p = at(R, s, side * (hw + 0.5));
            if (!this.spot(p.x, p.z, 0.15, 0.7) || this.treeGrid.near(p.x, p.z, 1.1)) continue;
            F.add('meter', p.x, this.baseY(p.x, p.z), p.z, toRoadYaw(p, side));
            this.grid.add(p.x, p.z); cnt.meters++;
          }
        }
        // parking regulation signs, mounted across the sidewalk so the parking lane can read them
        const parkKey = st.parkW > 0 && !blvd ? 'clean' : /^(primary|secondary|tertiary)$/.test(st.base) ? 'noPark' : null;
        if (parkKey && drivable) {
          const step = parkKey === 'clean' ? 70 : 150;
          for (let s = 25 + rnd() * 20; s < L - 10; s += step + rnd() * 20) {
            const [di, ri] = distToInt(R.ints, s);
            if (di < ri + 6) continue;
            const p = at(R, s, side * (hw + 0.45));
            if (!this.spot(p.x, p.z, 0.25, 0.8)) continue;
            this.sign(p.x, this.baseY(p.x, p.z), p.z, parkKey, yawFace(p.dx, p.dz), 0.4, 0.6, 2.15, parkKey);
            cnt.parkSigns++;
          }
        }
        // benches (boulevards / promenades, occasional elsewhere)
        const benchStep = blvd || prom ? 45 : commercial && sw >= 3.4 ? 90 : 0;
        if (benchStep) for (let s = 20 + rnd() * 20; s < L - 10; s += benchStep) {
          if (!(blvd || prom) && rnd() > 0.25) continue;
          const [di, ri] = distToInt(R.ints, s);
          if (di < ri + 6) continue;
          const p = at(R, s, side * (hw + sw * 0.62));
          const q = at(R, s - 0.85, side * (hw + sw * 0.62)), q2 = at(R, s + 0.85, side * (hw + sw * 0.62));
          if (!this.spot(p.x, p.z, 0.45, 1.6) || !this.spot(q.x, q.z, 0.3, 0) || !this.spot(q2.x, q2.z, 0.3, 0)) continue;
          const y = this.baseY(p.x, p.z);
          F.add('bench', p.x, y, p.z, toRoadYaw(p, side));
          this.grid.add(p.x, p.z); this.grid.add(q.x, q.z); this.grid.add(q2.x, q2.z);
          this.collide(p.x, y, p.z, Math.abs(p.dx) * 0.9 + Math.abs(p.dz) * 0.3, Math.abs(p.dz) * 0.9 + Math.abs(p.dx) * 0.3, 0.85); cnt.benches++;
        }
        // planters on wide downtown sidewalks (building side)
        if (commercial && sw >= 3.4) for (let s = 15 + rnd() * 15; s < L - 8; s += 35) {
          if (rnd() > (blvd ? 0.4 : 0.2)) continue;
          const [di, ri] = distToInt(R.ints, s);
          if (di < ri + 5) continue;
          const lat = hw + sw - 0.6, p = at(R, s, side * lat), q = at(R, s - 0.7, side * lat), q2 = at(R, s + 0.7, side * lat);
          if (!this.spot(p.x, p.z, 0.4, 1.6) || !this.spot(q.x, q.z, 0.3, 0) || !this.spot(q2.x, q2.z, 0.3, 0)) continue;
          const y = this.baseY(p.x, p.z);
          F.add('planter', p.x, y, p.z, yawX(p.dx, p.dz), { color: new THREE.Color().setHSL(0.24 + rnd() * 0.08, 0.35 + rnd() * 0.2, 0.42 + rnd() * 0.12) });
          this.grid.add(p.x, p.z); this.grid.add(q.x, q.z); this.grid.add(q2.x, q2.z);
          this.collide(p.x, y, p.z, Math.abs(p.dx) * 0.7 + Math.abs(p.dz) * 0.3, Math.abs(p.dz) * 0.7 + Math.abs(p.dx) * 0.3, 0.62); cnt.planters++;
        }
      }
      if (drivable) this.roadDecals(R, at, rnd, cnt);
    }
    Object.assign(this.counts, cnt);
  }

  /**
   * Shop-front life on commercial streets, on the paving between the kerb furniture and the building line:
   * café tables with chairs, A-frame boards, potted shrubs flanking doors (all instanced furniture).
   */
  buildFrontage() {
    const r = this.raster;
    if (!r || !this._swBits) return;
    const rnd = mulberry32((this.seed * 31337) ^ 0x7a11), F = this.F;
    const cnt = { cafeSets: 0, aframes: 0, pots: 0 };
    const col = (hex) => new THREE.Color(hex);
    for (const R of this.net.list) {
      const st = R.st;
      if (R.filler || R.bridge || st.minor || !st.sidewalk) continue;
      if (!/^(primary|secondary|tertiary|unclassified|residential)$/.test(st.base)) continue;
      const busy = /^(primary|secondary|tertiary)$/.test(st.base) ? 1 : 0.35, blvd = this.streets.boulevard(R.name || '', R);
      const L = R.pp.L, hw = st.hw, sw = st.sidewalk;
      for (const side of [-1, 1]) {
        for (let s = 5 + rnd() * 8; s < L - 5; s += 6 + rnd() * 10) {
          if (rnd() > 0.42 * busy) continue;
          const [di, ri] = distToInt(R.ints, s);
          if (di < ri + 5) continue;
          const [x, z, dx, dz] = samplePolyline(R.pts, R.pp, s);
          const nx = -dz * side, nz = dx * side;
          // building line: first building cell behind the kerb, reached over continuous paving
          let bl = 0;
          for (let d = hw + 1.0; d <= hw + sw + 7.6; d += 0.4) {
            const v = r.get(x + nx * d, z + nz * d);
            if (v & RBIT.BUILDING) { bl = d; break; }
            if (!(v & RBIT.SIDEWALK) || (v & (RBIT.ROAD | RBIT.SERVICE))) break;
          }
          if (!bl) continue;
          const room = bl - (hw + 1.1); // clear walking width between the kerb-side furniture zone and the wall
          const P = (d, t = 0) => [x + nx * d + dx * t, z + nz * d + dz * t];
          const ok = (p, rb, gap) => this.spot(p[0], p[1], rb, gap) && !this.treeGrid.near(p[0], p[1], 1.4) && !this.poleGrid.near(p[0], p[1], 1.2);
          const k = rnd();
          if (k < (blvd ? 0.38 : 0.26) && room >= 3.0) {
            // café: 1-2 bistro sets along the wall, chairs facing each other along the street
            const n = 1 + Math.floor(rnd() * 2), c = col(CHAIR_COLS[Math.floor(rnd() * CHAIR_COLS.length)]);
            for (let j = 0; j < n; j++) {
              const p = P(bl - 1.05, (j - (n - 1) / 2) * 1.9);
              if (!ok(p, 0.5, 1.1)) continue;
              const y = this.baseY(p[0], p[1]);
              F.add('cafeSet', p[0], y, p[1], yawX(dx, dz) + (rnd() - 0.5) * 0.3, { color: c });
              this.grid.add(p[0], p[1]); this.collide(p[0], y, p[1], 0.75, 0.75, 0.8); cnt.cafeSets++;
            }
          } else if (k < 0.7 && room >= 2.0) {
            // A-frame board near the door, boards facing along the sidewalk
            const p = P(bl - 0.75 - rnd() * 0.4, (rnd() - 0.5) * 2);
            if (!ok(p, 0.35, 1.0)) continue;
            const y = this.baseY(p[0], p[1]);
            F.add('aframe', p[0], y, p[1], yawFace(dx, dz) + (rnd() - 0.5) * 0.25, { color: col(BOARD_COLS[Math.floor(rnd() * BOARD_COLS.length)]) });
            this.grid.add(p[0], p[1]); cnt.aframes++;
          } else if (room >= 1.6) {
            // pair of potted shrubs flanking an entrance
            const c = new THREE.Color().setHSL(0.24 + rnd() * 0.07, 0.4 + rnd() * 0.2, 0.3 + rnd() * 0.1);
            for (const t of [-1.05, 1.05]) {
              const p = P(bl - 0.45, t);
              if (!ok(p, 0.32, 0.8)) continue;
              const y = this.baseY(p[0], p[1]);
              F.add('pot', p[0], y, p[1], rnd() * 6.28, { color: c });
              this.grid.add(p[0], p[1]); this.collide(p[0], y, p[1], 0.3, 0.3, 1.3); cnt.pots++;
            }
          }
        }
      }
    }
    Object.assign(this.counts, cnt);
  }

  roadDecals(R, at, rnd, cnt) {
    const st = R.st, lim = Math.max(0, st.hw - st.parkW - 1.0);
    for (let s = 12 + rnd() * 25; s < R.pp.L - 6; s += 38 + rnd() * 30) {
      const p = at(R, s, (rnd() * 2 - 1) * lim);
      if (this.raster && !(this.raster.get(p.x, p.z) & RBIT.ROAD)) continue;
      this.decal(p.x, p.z, rnd() * 6.28, 0.76, 0.76, 'manhole'); cnt.manholes++;
    }
  }

  /** Bus shelter on the right of travel direction `dir` at arc position s (far side of a junction). */
  tryShelter(R, s, dir, at, shelters, cnt, rnd) {
    const st = R.st, name = R.name || '', L = R.pp.L;
    if (s < 3 || s > L - 3) return;
    const lat = st.hw + 1.35, p = at(R, s, dir * lat), tx = p.dx * dir, tz = p.dz * dir;
    for (const q of shelters) {
      const d2 = (q.x - p.x) ** 2 + (q.z - p.z) ** 2;
      if (d2 < 60 * 60 || (q.name === name && d2 < 300 * 300 && q.tx * tx + q.tz * tz > 0.5)) return;
    }
    const r = this.raster;
    for (const u of [-2.1, 0, 2.1]) for (const l of [st.hw + 0.6, st.hw + 2.1]) {
      const q = at(R, s + u, dir * l);
      if (r && ((r.get(q.x, q.z) & (RBIT.ROAD | RBIT.SERVICE | RBIT.WATER)) || r.any(q.x, q.z, 0.3, RBIT.BUILDING))) return;
    }
    if (this._swBits && r && !(r.get(p.x, p.z) & RBIT.SIDEWALK)) return;
    if (this.grid.near(p.x, p.z, 2.4)) return;
    const [di, ri] = distToInt(R.ints, s);
    if (di < ri + 3) return;
    const ends = [at(R, s - 2, dir * lat), at(R, s + 2, dir * lat)];
    const y = Math.min(this.baseY(p.x, p.z), ...ends.map((q) => this.baseY(q.x, q.z))); // legs reach the low end
    const yaw = yawNegZ(dir * p.dz, -dir * p.dx); // open side (local -Z) toward the kerb
    this.F.add('shelter', p.x, y, p.z, yaw);
    _q.setFromAxisAngle(UP, yaw); _m.compose(_p.set(p.x, y, p.z), _q, _s.set(1, 1, 1));
    this.shelterMats.push(_m.clone());
    const W = (lx, lz) => { const [ox, oz] = rotXZ(yaw, lx, lz); return [p.x + ox, p.z + oz]; };
    const ax = Math.cos(yaw), az = -Math.sin(yaw); // local +X in world
    const k1 = POSTERS[Math.floor(rnd() * POSTERS.length)], k2 = POSTERS[Math.floor(rnd() * POSTERS.length)];
    let [px, pz] = W(2.08 + 0.092, 0.05);
    this.panels.push({ x: px, y: y + 1.25, z: pz, yaw: yawFace(ax, az), w: 1.12, h: 1.75, a: k1, b: 'back', glow: 1.2 });
    [px, pz] = W(2.08 - 0.092, 0.05);
    this.panels.push({ x: px, y: y + 1.25, z: pz, yaw: yawFace(-ax, -az), w: 1.12, h: 1.75, a: k2, b: 'back', glow: 1.2 });
    const [bx, bz] = W(-2.6, -0.5); // bus-stop flag on its own post
    if (this.spot(bx, bz, 0.2, 0, false)) this.sign(bx, this.baseY(bx, bz), bz, 'bus', yawFace(ax, az), 0.42, 0.72, 2.35, 'bus');
    for (const u of [-2, -1, 0, 1, 2]) for (const v of [-0.75, 0, 0.7]) { const [gx, gz] = W(u, v); this.grid.add(gx, gz); }
    const ex = Math.abs(ax) * 2.1 + Math.abs(az) * 0.78, ez = Math.abs(az) * 2.1 + Math.abs(ax) * 0.78;
    this.colliders.add(p.x - ex, y - 0.2, p.x + ex, y + 2.7, p.z - ez, p.z + ez, 3);
    shelters.push({ x: p.x, z: p.z, name, tx, tz }); cnt.shelters++;
  }

  // -------------------------------------------------------------- utility poles + wires (residential streets)
  buildUtilityPoles() {
    const rnd = mulberry32((this.seed * 104729) ^ 0x2b1d);
    const att = FURNITURE_META.utilityPole.wires;
    let poles = 0, spans = 0;
    for (const R of this.net.list) {
      const st = R.st, name = R.name || '';
      if (R.filler || R.bridge || st.minor || !st.sidewalk) continue;
      if (!/^(residential|unclassified|living_street)$/.test(st.base) || !this.streets.utilityPoles(name, R)) continue;
      if (hashStr(name || String(R.pts[0])) > 0.7) continue; // most downtown wiring is underground
      const side = hashStr(name + '#') < 0.5 ? -1 : 1;
      let prev = null;
      for (let s = 6 + rnd() * 6; s < R.pp.L - 4; s += 34 + rnd() * 6) {
        const [di, ri] = distToInt(R.ints, s);
        if (di < ri + 2.5) { prev = null; continue; }
        const [x, z, dx, dz] = samplePolyline(R.pts, R.pp, s);
        const px = x - dz * side * (st.hw + 0.42), pz = z + dx * side * (st.hw + 0.42);
        if (!this.spot(px, pz, 0.35, 1.5)) { prev = null; continue; }
        const y = this.baseY(px, pz), yaw = yawX(-dz, dx);
        this.F.add(rnd() < 0.12 ? 'utilityPoleTx' : 'utilityPole', px, y, pz, yaw);
        this.grid.add(px, pz); this.collide(px, y, pz, 0.17, 0.17, 11); poles++;
        const pts = att.map(([lx, ly, lz]) => { const [ox, oz] = rotXZ(yaw, lx, lz); return [px + ox, y + ly, pz + oz]; });
        if (prev && Math.hypot(px - prev.x, pz - prev.z) < 55) { this.addSpan(prev.pts, pts); spans++; }
        prev = { x: px, z: pz, pts };
      }
    }
    this.counts.utilityPoles = poles; this.counts.wireSpans = spans;
  }
  addSpan(a, b) {
    const mx = (a[0][0] + b[0][0]) / 2, mz = (a[0][2] + b[0][2]) / 2, key = `${Math.floor(mx / 300)},${Math.floor(mz / 300)}`;
    let B = this.wireChunks.get(key); if (!B) { B = new PartBuilder(); this.wireChunks.set(key, B); }
    const wire = { c: [0.015, 0.015, 0.016], rough: 0.6, metal: 0.3 };
    for (let w = 0; w < a.length; w++) {
      const p = a[w], q = b[w], span = Math.hypot(q[0] - p[0], q[2] - p[2]), sag = 0.2 + span * 0.014 + (w === 3 ? 0.25 : 0);
      const pts = [];
      for (let i = 0; i <= 8; i++) { const t = i / 8; pts.push([p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t - sag * 4 * t * (1 - t), p[2] + (q[2] - p[2]) * t]); }
      B.tube(pts, w === 3 ? 0.022 : 0.012, 3, wire);
    }
  }

  // -------------------------------------------------------------- plazas: OSM lamps, bollards at pedestrian streets
  buildPlazaFurniture() {
    const F = this.F;
    let acorn = 0, bol = 0;
    for (const p of this.data.lamps || []) {
      if (!p || !isFinite(p[0]) || !isFinite(p[1])) continue;
      if (!this.spot(p[0], p[1], 0.5, 8, false)) continue;
      const y = this.baseY(p[0], p[1]);
      F.add('acorn', p[0], y, p[1], 0);
      this.lampHeads.push([p[0], y + FURNITURE_META.acorn.light[1], p[1]]);
      this.grid.add(p[0], p[1]); this.collide(p[0], y, p[1], 0.15, 0.15, 4.6); acorn++;
    }
    for (const R of this.net.list) {
      if (R.st.cls !== 'pedestrian' || R.pts.length < 2) continue;
      for (const end of [0, 1]) {
        const a = end ? R.pts[R.pts.length - 1] : R.pts[0], b = end ? R.pts[R.pts.length - 2] : R.pts[1];
        let road = false;
        this.net.segHash.forEachNear(a[0], a[1], 4, (seg) => { if (seg.road && !seg.road.st.minor && seg.road.st.base !== 'service') road = true; });
        if (!road) continue;
        const dx = b[0] - a[0], dz = b[1] - a[1], l = Math.hypot(dx, dz) || 1, ux = dx / l, uz = dz / l;
        const n = clamp(Math.floor((R.st.hw * 2) / 1.5), 2, 6);
        for (let k = 0; k < n; k++) {
          const o = (k - (n - 1) / 2) * 1.5, x = a[0] + ux * 1.4 - uz * o, z = a[1] + uz * 1.4 + ux * o;
          if (!this.spot(x, z, 0.15, 0.6, false)) continue;
          const y = this.baseY(x, z);
          F.add('bollard', x, y, z, 0); this.grid.add(x, z); this.collide(x, y, z, 0.1, 0.1, 0.95); bol++;
        }
      }
    }
    this.counts.acornLamps = acorn; this.counts.bollards = (this.counts.bollards || 0) + bol;
  }

  // -------------------------------------------------------------- meshes
  finishFurniture() {
    // ---- one BatchedMesh for every furniture type (+ wire chunks)
    this.furnMat = createFurnitureMaterial();
    this.headMat = this.furnMat; // lamp heads glow through the furniture material's night emission
    const entries = [];
    for (const [type, list] of this.F.types) entries.push({ type, geo: makeFurnitureGeometry(type), list, cull: CULL[type] || 300 });
    for (const [key, B] of this.wireChunks) {
      const geo = B.build('furniture');
      const c = geo.boundingSphere.center;
      entries.push({ type: 'wire:' + key, geo, list: [{ x: c.x, y: c.y, z: c.z, yaw: 0, o: { identity: true } }], cull: 700 + geo.boundingSphere.radius });
    }
    let maxV = 0, maxI = 0;
    for (const e of entries) { maxV += e.geo.attributes.position.count; maxI += e.list.length; }
    const counts = {};
    if (maxI) {
      const bm = new THREE.BatchedMesh(maxI, maxV, maxV, this.furnMat);
      bm.name = 'street-furniture';
      const cx = new Float32Array(maxI), cy = new Float32Array(maxI), cz = new Float32Array(maxI), r2 = new Float32Array(maxI), ids = new Int32Array(maxI);
      let k = 0, tris = 0;
      for (const e of entries) {
        const gid = bm.addGeometry(e.geo);
        counts[e.type.startsWith('wire:') ? 'wires' : e.type] = (counts[e.type.startsWith('wire:') ? 'wires' : e.type] || 0) + e.list.length;
        tris += e.geo.userData.tris * e.list.length;
        for (const it of e.list) {
          const id = bm.addInstance(gid);
          const o = it.o || {};
          if (o.identity) _m.identity();
          else {
            _q.setFromEuler(_e.set(o.pitch || 0, it.yaw, 0, 'YXZ'));
            _m.compose(_p.set(it.x, it.y, it.z), _q, _s.set(o.sx || 1, o.sy || 1, o.sz || 1));
          }
          bm.setMatrixAt(id, _m);
          if (o.color) bm.setColorAt(id, o.color);
          ids[k] = id; cx[k] = it.x; cy[k] = it.y; cz[k] = it.z; r2[k] = e.cull * e.cull; k++;
        }
      }
      bm.perObjectFrustumCulled = true; bm.sortObjects = false; bm.frustumCulled = false;
      bm.castShadow = true; bm.receiveShadow = true;
      this.group.add(bm);
      this.furniture = { mesh: bm, ids, cx, cy, cz, r2, vis: new Uint8Array(maxI).fill(1), n: maxI };
      this.counts.furnitureInstances = maxI; this.counts.furnitureTris = tris;
    }
    this.counts.furniture = counts;

    // ---- sign atlas, panels, road decals
    const names = new Set();
    for (const p of this.panels) for (const k of [p.a, p.b]) if (k && k.startsWith('name:')) names.add(k.slice(5));
    this.atlas = buildSignAtlas([...names], this.signStyle || undefined);
    const uvOf = (k) => (k && k.startsWith('name:') ? this.atlas.nameUV(k.slice(5)) : this.atlas.uv(k || 'back'));
    const makeInst = (list, geo, mat, name, decal) => {
      const n = list.length;
      if (!n) return null;
      const g = geo.clone();
      const A = new Float32Array(n * 4), Bv = new Float32Array(n * 4), G = new Float32Array(n);
      const mesh = new THREE.InstancedMesh(g, mat, n);
      list.forEach((p, i) => {
        A.set(uvOf(p.a), i * 4); Bv.set(uvOf(p.b || p.a), i * 4); G[i] = p.glow || 0;
        if (decal) {
          _n.set(p.nrm[0], p.nrm[1], p.nrm[2]).normalize();
          _q.setFromUnitVectors(UP, _n).multiply(_q2.setFromAxisAngle(UP, p.yaw));
          _m.compose(_p.set(p.x, p.y, p.z), _q, _s.set(p.w, 1, p.h));
        } else {
          _q.setFromEuler(_e.set(p.pitch || 0, p.yaw, 0, 'YXZ'));
          _m.compose(_p.set(p.x, p.y, p.z), _q, _s.set(p.w, p.h, 1));
        }
        mesh.setMatrixAt(i, _m);
      });
      g.setAttribute('aUvA', new THREE.InstancedBufferAttribute(A, 4));
      g.setAttribute('aUvB', new THREE.InstancedBufferAttribute(Bv, 4));
      g.setAttribute('aGlow', new THREE.InstancedBufferAttribute(G, 1));
      mesh.name = name; mesh.castShadow = false; mesh.receiveShadow = true;
      mesh.instanceMatrix.needsUpdate = true; mesh.computeBoundingSphere();
      this.group.add(mesh);
      return mesh;
    };
    this.panelMat = createPanelMaterial(this.atlas.texture, { cull: 420 });
    this.decalMat = createPanelMaterial(this.atlas.texture, { decal: true, cull: 140 });
    this.panelMesh = makeInst(this.panels, makePanelGeometry(), this.panelMat, 'sign-panels', false);
    this.decalMesh = makeInst(this.decals, makeDecalGeometry(), this.decalMat, 'road-decals', true);
    this.counts.signPanels = this.panels.length; this.counts.decals = this.decals.length;
    // ---- bus shelter glazing
    if (this.shelterMats.length) {
      const glass = new THREE.InstancedMesh(makeShelterGlassGeometry(), createShelterGlassMaterial(), this.shelterMats.length);
      this.shelterMats.forEach((m, i) => glass.setMatrixAt(i, m));
      glass.name = 'shelter-glass'; glass.castShadow = false; glass.receiveShadow = false; glass.renderOrder = 2;
      glass.instanceMatrix.needsUpdate = true; glass.computeBoundingSphere();
      this.group.add(glass);
    }
  }

  // -------------------------------------------------------------- lights
  buildPointLights() {
    this.lightPool = new NearestLights({ count: POINT_LIGHTS, color: 0xffc98a, distance: 48, decay: 2, parent: this.group, yOffset: -0.35, name: 'lamp-light' });
    this.pointLights = this.lightPool.lights;
    this._lampN = -1;
  }

  /** Distance culling of furniture instances + parked-car LOD (throttled to camera movement). */
  updateCulling(dt, cam) {
    this._cullTimer -= dt;
    if (this._cullTimer > 0 && this._cullCam.distanceToSquared(cam) < 36) return;
    this._cullTimer = 0.5; this._cullCam.copy(cam);
    const F = this.furniture;
    if (F) {
      const { mesh, ids, cx, cy, cz, r2, vis, n } = F;
      for (let i = 0; i < n; i++) {
        const dx = cx[i] - cam.x, dy = cy[i] - cam.y, dz = cz[i] - cam.z;
        const v = dx * dx + dz * dz + dy * dy < r2[i] ? 1 : 0;
        if (v !== vis[i]) { vis[i] = v; mesh.setVisibleAt(ids[i], v === 1); }
      }
    }
    const P = this.parked;
    if (P) {
      const { mesh, ids, type, x, y, z, st, gids } = P;
      for (let i = 0; i < ids.length; i++) {
        const d2 = (x[i] - cam.x) ** 2 + (z[i] - cam.z) ** 2 + (y[i] - cam.y) ** 2;
        const s = d2 < 42 * 42 ? 0 : d2 < 520 * 520 ? 1 : -1;
        if (s === st[i]) continue;
        if (s < 0) mesh.setVisibleAt(ids[i], false);
        else {
          if (st[i] < 0) mesh.setVisibleAt(ids[i], true);
          mesh.setGeometryIdAt(ids[i], gids[type[i]][s]);
        }
        st[i] = s;
      }
    }
  }

  /**
   * @param {number} dt @param {THREE.Vector3} cam camera position @param {number} darkness 0..1 @param {number} time
   * @param {{position, frustum, direction}} [view] camera frustum + view direction (World.update): the lamp lights
   *        go to lamps the view can see
   */
  update(dt, cam, darkness, time, view) {
    this.updateSignals(time);
    if (this.trees && this.trees.update) this.trees.update(dt, cam, darkness, time);
    const on = darkness > 0.3;
    const night = darkness > 0.05 ? 0.4 + darkness * 4 : 0;
    if (this.furnMat) this.furnMat.userData.u.uNight.value = night;
    if (this.panelMat) this.panelMat.userData.u.uNight.value = darkness > 0.05 ? 0.15 + darkness * 1.6 : 0;
    if (this.parkedMat) { const u = this.parkedMat.userData.pu; u.uDarkP.value = clamp((darkness - 0.12) / 0.45, 0, 1); u.uTimeP.value = time || 0; }
    if (cam) this.updateCulling(dt || 0, cam);
    const P = this.lightPool;
    if (!P) return;
    if (this.lampHeads.length !== this._lampN) { this._lampN = this.lampHeads.length; P.setPositions(this.lampHeads); } // the bridge adds deck lamps later
    const v = view || (cam ? (this._view || (this._view = { position: null })) : null);
    if (v && !view) v.position = cam;
    P.update(dt || 0, v, on ? 260 * Math.min(1, darkness * 1.3) : 0);
  }
}

/**
 * Parked cars: no lit lamps, but the tail-lamp lenses glint red in the night (retro-reflectors catching the
 * street lights) and a police light bar idles with a slow alternating red / blue flash.
 */
function patchParkedMaterial(m) {
  const pu = { uDarkP: { value: 0 }, uTimeP: { value: 0 } };
  const prev = m.onBeforeCompile;
  m.onBeforeCompile = (sh, r) => {
    prev(sh, r);
    Object.assign(sh.uniforms, pu);
    sh.vertexShader = sh.vertexShader
      .replace('uniform float uDark;', 'uniform float uDark; uniform float uDarkP; uniform float uTimeP;')
      .replace('vLampE = vec3( 0.0 );', `vLampE = vec3( 0.0 );
if ( aLamp > 1.5 && aLamp < 2.5 ) vLampE = vec3( 1.0, 0.02, 0.01 ) * ( 0.02 + uDarkP * 0.16 );
else if ( aLamp > 4.5 ) {
  float ph = sin( uTimeP * 3.3 );
  float on = aLamp < 5.5 ? smoothstep( 0.2, 0.7, ph ) : smoothstep( 0.2, 0.7, -ph );
  vLampE = ( aLamp < 5.5 ? vec3( 1.0, 0.04, 0.03 ) : vec3( 0.06, 0.2, 1.0 ) ) * on * ( 0.6 + uDarkP * 3.0 );
}`);
  };
  m.customProgramCacheKey = () => 'roadster-vehicle-parked-1';
  m.userData.pu = pu;
  return m;
}

function lineIntersect(s1, s2) {
  const x1 = s1.ax, z1 = s1.az, x2 = s1.bx, z2 = s1.bz, x3 = s2.ax, z3 = s2.az, x4 = s2.bx, z4 = s2.bz;
  const d = (x1 - x2) * (z3 - z4) - (z1 - z2) * (x3 - x4);
  if (Math.abs(d) < 1e-6) return null;
  const t = ((x1 - x3) * (z3 - z4) - (z1 - z3) * (x3 - x4)) / d;
  return [x1 + t * (x2 - x1), z1 + t * (z2 - z1)];
}
