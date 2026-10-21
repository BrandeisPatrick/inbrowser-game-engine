// Traffic simulation core: lane graph + per-step AI, no scene objects (THREE math only), so it runs on the main
// thread (Traffic.js in-thread mode, headless Node sims: dev/traffic-sim.mjs) and in a Web Worker
// (traffic.worker.js) alike. Traffic.js is the main-thread presenter (instanced meshes, lights, colliders).
//
// Graph: drivable roads (TrafficSim.roadsFrom(net)) are split at shared vertices (snapped within 1.6 m,
// height-aware so decks never connect to the streets they fly over) into edges; each edge gets right-hand lanes offset from
// the centreline (oneway: all lanes one way; two-way: floor(lanes/2) per direction, >= 1), trimmed back from
// crossings to the painted stop line (and from acute merges / lane-count changes so connectors can taper).
// Duplicate parallel carriageways are switched off. Lanes are joined at nodes by cubic-Bezier "connector"
// lanes (straight / right / left by lane position, U-turns only at dead ends) with conflict sets computed
// spatially (also across nearby nodes) plus "swept zones" where a connector crosses another road's lane.
//
// Ground: the sim never queries the terrain. At build time every lane (and connector) gets its ground height baked
// every <= HSTEP_FINE m (`groundAt(x, z, yRef)` = world.getGroundHeight, walked along the lane exactly as a
// vehicle would query it: decks within 1.6 m above the previous sample win on bridge lanes, ground lanes ignore decks
// overhead), kept at 2x / 4x that spacing where the street is evenly graded; vehicles interpolate it at their front /
// rear axle (y = mean, pitch from the difference). A worker gets the baked array (`baked: { heights, index }`) instead.
//
// Behaviour: IDM car-following (T ~1.6 s, s0 3 m) along a 6-element planned path, curve-speed look-ahead,
// traffic signals read from `sigState[info.si]` (the SAME phase the bulbs show: Props.signals[].state, copied in
// every frame; yellow = stop if you can), stop signs (minor approaches + low-class all-way), first-come connector
// reservations with "don't block the box", deadlock breakers, braking for / never moving into the player
// (setPlayer: info object or getter), far vehicles recycled out of the camera frustum (setView).
//
//   const sim = new TrafficSim({ roads, signals, groundAt, spawn, seed, count });  sim.build();
//   per frame: sim.readSignals(props.signals); sim.setView(x, y, z, pvMatrixElements); sim.setPlayer(info); sim.update(dt);
//   sim.writePoses(f32) -> POSE_STRIDE floats per vehicle; sim.queryColliders(x, z, r, out); sim.summary()
import * as THREE from 'three';
import { VEHICLE_SPECS, pickVehicleColor } from './Vehicles.js';
import { mulberry32 } from './geom.js';

const DRIVABLE = new Set(['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'unclassified', 'residential', 'living_street', 'road']);
const SPEED = { motorway: 27, trunk: 22, primary: 15, secondary: 14, tertiary: 13, unclassified: 11.5, residential: 11, living_street: 6, road: 10 };
const RANK = { motorway: 6, trunk: 5, primary: 4, secondary: 3, tertiary: 2, unclassified: 1, residential: 1, road: 1, living_street: 0 };
const TURN_W = { S: 3, R: 1.1, L: 1, U: 0.15 };
const PATH_N = 6;           // planned path elements kept ahead (lanes + connectors)
const CTRL_NONE = 0, CTRL_SIGNAL = 1, CTRL_STOP = 2, CTRL_YIELD = 3;
const B_COMF = 2.6;         // comfortable deceleration (m/s^2)
const HEADWAY = 1.6;        // s
const GAP0 = 3.0;           // m, standstill bumper gap
const RECYCLE_R = 700;      // m from camera
const CELL = 32;            // collider grid cell
const CONF_D = 3.0;         // connectors whose paths come closer than this conflict (car width + margin)
const HSTEP_FINE = 1.5;     // m, spacing of the baked ground-height samples along a lane (kept where needed)...
const HTOL = 0.015;         // ...m: samples dropped (spacing x2 / x4) where the straight line stays this close
const KEY_OFF = 4096;
const ckey = (i, k) => (i + KEY_OFF) * 8192 + (k + KEY_OFF);
const MAX_SLICES = 10;
const SIG_UNKNOWN = 255;
/** Floats per vehicle in writePoses(): x, y, z, yaw, pitch, brake, speed, yaw rate, flags (1 active, 2 respawned). */
export const POSE_STRIDE = 9;

const _sph = new THREE.Sphere(), _p = new THREE.Vector3(), _pm = new THREE.Matrix4();
const P0 = { x: 0, z: 0, dx: 0, dz: -1, y: 0 }, P1 = { x: 0, z: 0, dx: 0, dz: -1, y: 0 };
const P2 = { x: 0, z: 0, dx: 0, dz: -1, y: 0 }, P3 = { x: 0, z: 0, dx: 0, dz: -1, y: 0 };
const Q0 = { x: 0, z: 0, fx: 0, fz: -1 }, Q1 = { x: 0, z: 0, fx: 0, fz: -1 };

// ------------------------------------------------------------------ vehicle types (geometry: Vehicles.js)
// [name, share of the fleet, max acceleration m/s^2, paint palette]; L / W / Hc / WB come from the geometry specs.
const TYPE_DEFS = [
  ['sedan', 0.34, 2.0, 'car'], ['hatchback', 0.12, 2.0, 'car'], ['suv', 0.22, 1.8, 'car'], ['pickup', 0.05, 1.7, 'car'],
  ['minivan', 0.06, 1.7, 'car'], ['taxi', 0.04, 2.0, 'taxi'], ['police', 0.015, 2.1, 'police'],
  ['boxtruck', 0.065, 1.15, 'truck'], ['bus', 0.05, 1.0, 'bus'],
];
export const TRAFFIC_TYPES = TYPE_DEFS.map(([name, share, amax, pal]) => {
  const s = VEHICLE_SPECS[name];
  return { name, L: s.L, W: s.W, H: s.H, Hc: s.Hc, WB: s.WB, amax, share, pal };
});
/** Vehicles per type for a fleet of n (the same split the meshes use). */
export function typeCounts(n) {
  const counts = TRAFFIC_TYPES.map((t) => Math.round(t.share * n));
  counts[0] += n - counts.reduce((a, b) => a + b, 0);
  return counts.map((c) => Math.max(0, c));
}

/**
 * Player footprint for the AI from a getter result / object: AABB {minX..maxZ} | THREE.Vector3 | Object3D | {object3d}.
 * Returns { x, y, z, fx, fz, hl, hw, rx, rz } (half extents along its own forward / right axes) or null.
 */
export function playerInfoFrom(src, out = {}) {
  let r;
  try { r = typeof src === 'function' ? src() : src; } catch { return null; }
  if (!r) return null;
  if (Number.isFinite(r.minX)) {
    out.x = (r.minX + r.maxX) / 2; out.y = r.minY; out.z = (r.minZ + r.maxZ) / 2; out.rx = (r.maxX - r.minX) / 2; out.rz = (r.maxZ - r.minZ) / 2;
    out.fx = 1; out.fz = 0; out.hl = (r.maxX - r.minX) / 2; out.hw = (r.maxZ - r.minZ) / 2;
    return out;
  }
  const o = r.object3d || r;
  const p = o.position || o;
  if (!p || !Number.isFinite(p.x)) return null;
  let fx = 0, fz = -1;
  if (o.quaternion) { _p.set(0, 0, -1).applyQuaternion(o.quaternion); const l = Math.hypot(_p.x, _p.z); if (l > 0.2) { fx = _p.x / l; fz = _p.z / l; } }
  const hl = 2.3, hw = 1.0;
  out.x = p.x; out.y = p.y; out.z = p.z; out.fx = fx; out.fz = fz; out.hl = hl; out.hw = hw;
  out.rx = Math.abs(fx) * hl + Math.abs(fz) * hw; out.rz = Math.abs(fz) * hl + Math.abs(fx) * hw;
  return out;
}

/**
 * Junction signal layout for the sim: one entry per Props signal (`props.signals[i]` <-> `infos[k].si`): junction centre
 * and the two phase directions. Clusters the city's signal nodes exactly like Props.buildSignals and reads the two
 * crossing roads from net.segHash; if that does not line up with the rendered signals, derives the layout from the
 * signal heads (`props.bulbs` instance matrices). Plain data (structured-cloneable for the worker).
 */
export function buildSignalInfos({ data, net, props }) {
  const P = props;
  if (!P || !Array.isArray(P.signals) || !P.signals.length) return [];
  let infos = [];
  try {
    const pts = (data && data.signals || []).filter((p) => p && isFinite(p[0]) && isFinite(p[1]));
    const clusters = [];
    for (const p of pts) { // identical clustering to Props.buildSignals so infos[i] <-> props.signals[i]
      const c = clusters.find((c) => (c.x - p[0]) ** 2 + (c.z - p[1]) ** 2 < 22 * 22);
      if (c) { c.n++; c.x += (p[0] - c.x) / c.n; c.z += (p[1] - c.z) / c.n; c.pts.push(p); }
      else clusters.push({ x: p[0], z: p[1], n: 1, pts: [p] });
    }
    for (const c of clusters) {
      const found = [];
      for (const p of c.pts) {
        net.segHash.forEachNear(p[0], p[1], 3, (seg) => {
          const R = seg.road;
          if (!R || R.st.minor || R.st.base === 'service') return;
          const dx = seg.bx - seg.ax, dz = seg.bz - seg.az, L = Math.hypot(dx, dz) || 1;
          found.push({ hw: R.st.hw, dir: [dx / L, dz / L], seg });
        });
      }
      if (!found.length) continue;
      found.sort((a, b) => b.hw - a.hw);
      const A = found[0];
      const B = found.find((f) => Math.abs(f.dir[0] * A.dir[0] + f.dir[1] * A.dir[1]) < 0.8);
      if (!B) continue;
      const cen = lineIntersect(A.seg, B.seg) || [c.x, c.z];
      if ((cen[0] - c.x) ** 2 + (cen[1] - c.z) ** 2 > 30 * 30) continue;
      infos.push({ x: cen[0], z: cen[1], a: A.dir, b: B.dir });
    }
  } catch (e) { infos = []; }
  if (infos.length === P.signals.length) {
    infos.forEach((f, i) => { f.si = i; f.offset = P.signals[i].offset || 0; });
  } else {
    // fallback: derive junction centre + phase directions from the rendered signal heads
    infos = [];
    const m = new THREE.Matrix4(), pos = new THREE.Vector3(), q = new THREE.Quaternion(), sc = new THREE.Vector3();
    P.signals.forEach((sig, si) => {
      if (!P.bulbs || !sig.heads || !sig.heads.length) return;
      let sx = 0, sz = 0; const dir = [null, null];
      for (const h of sig.heads) {
        P.bulbs.getMatrixAt(h.idx[0], m); m.decompose(pos, q, sc);
        sx += pos.x; sz += pos.z;
        if (!dir[h.group]) { const f = new THREE.Vector3(0, 0, 1).applyQuaternion(q); dir[h.group] = [f.x, f.z]; }
      }
      if (!dir[0] || !dir[1]) return;
      infos.push({ x: sx / sig.heads.length, z: sz / sig.heads.length, a: dir[0], b: dir[1], si, offset: sig.offset || 0 });
    });
    console.warn('[Traffic] signal layout mismatch, using head-derived signals', infos.length);
  }
  return infos;
}

// ------------------------------------------------------------------ polyline helpers

function cumLen(xs, zs) {
  const n = xs.length, c = new Float64Array(n);
  for (let i = 1; i < n; i++) c[i] = c[i - 1] + Math.hypot(xs[i] - xs[i - 1], zs[i] - zs[i - 1]);
  return c;
}
function cumPts(pts) {
  const c = [0];
  for (let i = 1; i < pts.length; i++) c.push(c[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
  return c;
}
function pointAt(pts, cum, s) {
  const n = pts.length;
  if (s <= 0) return [pts[0][0], pts[0][1]];
  if (s >= cum[n - 1]) return [pts[n - 1][0], pts[n - 1][1]];
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (cum[mid] <= s) lo = mid; else hi = mid; }
  const t = (s - cum[lo]) / ((cum[lo + 1] - cum[lo]) || 1);
  return [pts[lo][0] + (pts[lo + 1][0] - pts[lo][0]) * t, pts[lo][1] + (pts[lo + 1][1] - pts[lo][1]) * t];
}
function cutPolyline(pts, cum, a, b) {
  const out = [pointAt(pts, cum, a)];
  for (let i = 1; i < pts.length - 1; i++) if (cum[i] > a + 0.05 && cum[i] < b - 0.05) out.push([pts[i][0], pts[i][1]]);
  out.push(pointAt(pts, cum, b));
  return out;
}
/** Offset a polyline to its right (travel direction) by `off` metres using vertex miters. */
function offsetPolyline(pts, off) {
  const n = pts.length, d = [];
  for (let i = 0; i < n - 1; i++) {
    const dx = pts[i + 1][0] - pts[i][0], dz = pts[i + 1][1] - pts[i][1], L = Math.hypot(dx, dz) || 1e-6;
    d.push([dx / L, dz / L]);
  }
  const xs = [], zs = [];
  for (let i = 0; i < n; i++) {
    const a = d[Math.max(0, i - 1)], b = d[Math.min(n - 2, i)];
    let mx = -a[1] - b[1], mz = a[0] + b[0];
    const l = Math.hypot(mx, mz);
    if (l < 1e-3) { mx = -b[1]; mz = b[0]; } else { mx /= l; mz /= l; }
    const k = 1 / Math.max(0.5, mx * -b[1] + mz * b[0]);
    const x = pts[i][0] + mx * k * off, z = pts[i][1] + mz * k * off;
    if (xs.length && i < n - 1) { // drop points that would make the offset line run backwards (tight inner corners)
      const px = xs[xs.length - 1], pz = zs[zs.length - 1];
      if ((x - px) * b[0] + (z - pz) * b[1] < 0.05 && (x - px) * a[0] + (z - pz) * a[1] < 0.05) continue;
    }
    xs.push(x); zs.push(z);
  }
  if (xs.length < 2) { const e = pts[n - 1]; const bb = d[n - 2]; xs.push(e[0] - bb[1] * off); zs.push(e[1] + bb[0] * off); }
  return [xs, zs];
}
/** Point (x, z), unit direction (dx, dz) and baked ground height y (when the lane has one) at arc length s. */
function sampleLane(L, s, out) {
  const c = L.cum, n = c.length, xs = L.x, zs = L.z;
  if (L.hy) out.y = laneY(L, s);
  if (n < 2) { out.x = xs[0]; out.z = zs[0]; return out; }
  let i;
  if (s <= 0) { i = 0; s = 0; } else if (s >= L.len) { i = n - 2; s = L.len; } else {
    let lo = 0, hi = n - 1;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (c[mid] <= s) lo = mid; else hi = mid; }
    i = lo;
  }
  const sl = c[i + 1] - c[i];
  const t = sl > 1e-6 ? (s - c[i]) / sl : 0;
  const dx = xs[i + 1] - xs[i], dz = zs[i + 1] - zs[i];
  out.x = xs[i] + dx * t; out.z = zs[i] + dz * t;
  if (sl > 1e-6) { out.dx = dx / sl; out.dz = dz / sl; }
  return out;
}
/** Baked ground height of lane L at arc length s (linear between the samples). */
function laneY(L, s) {
  const h = L.hy, n = h.length;
  if (n < 2) return h[0];
  let u = s * L.hk;
  if (!(u > 0)) return h[0];
  if (u >= n - 1) return h[n - 1];
  const i = u | 0; u -= i;
  return h[i] + (h[i + 1] - h[i]) * u;
}

function ptSegD2(px, pz, ax, az, bx, bz) {
  const dx = bx - ax, dz = bz - az, L2 = dx * dx + dz * dz;
  let t = L2 > 1e-9 ? ((px - ax) * dx + (pz - az) * dz) / L2 : 0;
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  const x = ax + dx * t - px, z = az + dz * t - pz;
  return x * x + z * z;
}
function segSegD2(ax, az, bx, bz, cx, cz, dx, dz) {
  const o1 = (bx - ax) * (cz - az) - (bz - az) * (cx - ax), o2 = (bx - ax) * (dz - az) - (bz - az) * (dx - ax);
  const o3 = (dx - cx) * (az - cz) - (dz - cz) * (ax - cx), o4 = (dx - cx) * (bz - cz) - (dz - cz) * (bx - cx);
  if (((o1 > 0) !== (o2 > 0)) && ((o3 > 0) !== (o4 > 0))) return 0;
  return Math.min(ptSegD2(ax, az, cx, cz, dx, dz), ptSegD2(bx, bz, cx, cz, dx, dz), ptSegD2(cx, cz, ax, az, bx, bz), ptSegD2(dx, dz, ax, az, bx, bz));
}
function polyMinD2(A, B, lim2) {
  let best = Infinity;
  for (let i = 0; i < A.x.length - 1; i++) for (let j = 0; j < B.x.length - 1; j++) {
    const d = segSegD2(A.x[i], A.z[i], A.x[i + 1], A.z[i + 1], B.x[j], B.z[j], B.x[j + 1], B.z[j + 1]);
    if (d < best) { best = d; if (best < lim2) return best; }
  }
  return best;
}
function lineIntersect(s1, s2) {
  const x1 = s1.ax, z1 = s1.az, x2 = s1.bx, z2 = s1.bz, x3 = s2.ax, z3 = s2.az, x4 = s2.bx, z4 = s2.bz;
  const d = (x1 - x2) * (z3 - z4) - (z1 - z2) * (x3 - x4);
  if (Math.abs(d) < 1e-6) return null;
  const t = ((x1 - x3) * (z3 - z4) - (z1 - z3) * (x3 - x4)) / d;
  return [x1 + t * (x2 - x1), z1 + t * (z2 - z1)];
}
const lineAngle = (ax, az, bx, bz) => { const d = Math.abs(ax * bx + az * bz); return Math.acos(Math.min(1, d)); };
/** [lanes forward, lanes backward] for a road style (same rule as the lane builder). */
function laneSplit(st) {
  if (st.oneway) return [Math.max(1, Math.min(6, st.lanes)), 0];
  const n = Math.max(1, Math.min(4, Math.floor(st.lanes / 2)));
  return [n, n];
}
/** 2D oriented-box overlap (SAT). Boxes: centre, forward unit (fx,fz), half length / half width. */
function obb2(ax, az, afx, afz, ahl, ahw, bx, bz, bfx, bfz, bhl, bhw) {
  const dx = bx - ax, dz = bz - az;
  const axes = _obbAxes;
  axes[0] = afx; axes[1] = afz; axes[2] = -afz; axes[3] = afx; axes[4] = bfx; axes[5] = bfz; axes[6] = -bfz; axes[7] = bfx;
  for (let i = 0; i < 8; i += 2) {
    const ux = axes[i], uz = axes[i + 1];
    const ra = ahl * Math.abs(afx * ux + afz * uz) + ahw * Math.abs(-afz * ux + afx * uz);
    const rb = bhl * Math.abs(bfx * ux + bfz * uz) + bhw * Math.abs(-bfz * ux + bfx * uz);
    if (Math.abs(dx * ux + dz * uz) > ra + rb) return false;
  }
  return true;
}
const _obbAxes = new Float64Array(8);
const wrapPi = (a) => { while (a > Math.PI) a -= 2 * Math.PI; while (a < -Math.PI) a += 2 * Math.PI; return a; };

// ------------------------------------------------------------------ TrafficSim
export class TrafficSim {
  /**
   * @param {object} o
   * @param {Array} o.roads  TrafficSim.roadsFrom(net) (plain data: pts, st, bridge, hs, filler, name)
   * @param {Array} [o.signals]  buildSignalInfos(...) ([{ x, z, a, b, si, offset }])
   * @param {(x, z, yRef) => number} [o.groundAt]  drivable-surface height (world.getGroundHeight) for the height bake
   * @param {{heights: Float32Array, index: Uint32Array}} [o.baked]  pre-baked heights (worker) instead of groundAt
   * @param {{x, z}} [o.spawn]  initial vehicles spread around this point
   * @param {number} [o.seed=7]
   * @param {number} [o.count=250]
   */
  constructor({ roads = [], signals = [], groundAt = null, baked = null, spawn = null, seed = 7, count = 250 } = {}) {
    this.roads = roads; this.sigInfos = signals; this.groundAt = groundAt; this.baked = baked;
    this.spawn = spawn || { x: 0, z: 0 };
    this.seed = seed;
    this.target = Number.isFinite(count) ? Math.max(0, Math.floor(count)) : 250;
    this.count = 0;
    this.rnd = mulberry32((seed * 2654435761) ^ 0x7a3f19);
    this.sigState = new Uint8Array(Math.max(1, signals.reduce((m, f) => Math.max(m, (f.si | 0) + 1), 0))).fill(SIG_UNKNOWN);
    this.time = 0;
    this.sigEntries = 0; this.redEntries = 0;
    this._player = null; this._plOut = {}; this._pl = null;
    this.lanes = [];
    this.nodes = [];
    this.vehicles = [];
    this._touched = [];
    this._grid = new Map();
    this._gridKeys = [];
    this._frustum = new THREE.Frustum();
    this._hasFrustum = false;
    this._cam = new THREE.Vector3(0, 30, 0);
    this.stats = {};
  }

  /** Drivable roads of a RoadNetwork as plain data (the graph input; also what the worker receives). */
  static roadsFrom(net) {
    const out = [];
    for (const R of (net && net.list) || []) {
      const st = R.st;
      if (!st || st.minor || !DRIVABLE.has(st.base) || !R.pts || R.pts.length < 2) continue;
      out.push({
        pts: R.pts, bridge: !!R.bridge, hs: R.bridge && R.hs ? R.hs : null, filler: !!R.filler, name: R.name,
        st: { minor: st.minor, base: st.base, hw: st.hw, parkW: st.parkW, lanes: st.lanes, oneway: st.oneway, link: st.link, markings: st.markings },
      });
    }
    return out;
  }

  /** Player for the AI: info object { x, y, z, fx, fz, hl, hw, rx, rz } (playerInfoFrom), a getter / Object3D, or null. */
  setPlayer(p) { this._player = p || null; }
  /** Camera position (+ optional view-projection matrix elements for out-of-view respawns). */
  setView(x, y, z, pv) {
    this._cam.set(x, y, z);
    if (pv) { _pm.fromArray(pv); this._frustum.setFromProjectionMatrix(_pm); this._hasFrustum = true; }
  }
  /** Copy the rendered signal phases (Props.signals[].state) into sigState. */
  readSignals(signals) {
    const S = this.sigState, n = Math.min(S.length, signals ? signals.length : 0);
    for (let i = 0; i < n; i++) { const st = signals[i].state; S[i] = st >= 0 && st < 255 ? st : SIG_UNKNOWN; }
  }

  build() {
    if (!this.roads.length || !this.target) return false;
    const t0 = performance.now();
    this._buildGraph();
    const t1 = performance.now();
    this._buildControls();
    this._buildSpawnPoints();
    const t2 = performance.now();
    if (this.baked) this._attachHeights(this.baked); else this._bakeHeights();
    const t3 = performance.now();
    this._buildVehicles();
    this._spawnInitial();
    this.stats.buildMs = Math.round(performance.now() - t0);
    this.stats.graphMs = Math.round(t1 - t0);
    this.stats.heightMs = Math.round(t3 - t2);
    return true;
  }


  // ================================================================ lane graph
  _buildGraph() {
    const roads = this.roads;
    // ---- snap vertices (height-aware so decks don't join the streets below)
    const SN2 = 1.6 * 1.6, cell = 4, P = [], grid = new Map();
    const snap = (x, z, h, ri) => {
      const i0 = Math.floor(x / cell), k0 = Math.floor(z / cell);
      let best = -1, bd = SN2;
      for (let di = -1; di <= 1; di++) for (let dk = -1; dk <= 1; dk++) {
        const a = grid.get(ckey(i0 + di, k0 + dk)); if (!a) continue;
        for (const id of a) {
          const p = P[id]; if (Math.abs(p.h - h) > 1.2) continue;
          const d2 = (p.x - x) ** 2 + (p.z - z) ** 2;
          if (d2 < bd) { bd = d2; best = id; }
        }
      }
      if (best < 0) {
        best = P.length; P.push({ x, z, h, n: 0, last: -1, self: 0, node: -1 });
        const k = ckey(i0, k0); let a = grid.get(k); if (!a) { a = []; grid.set(k, a); } a.push(best);
      }
      const p = P[best];
      if (p.last !== ri) { p.n++; p.last = ri; } else p.self++;
      return best;
    };
    const rv = roads.map((R, ri) => {
      const ids = [];
      for (let i = 0; i < R.pts.length; i++) {
        const h = R.bridge && R.hs ? R.hs[i] || 0 : 0;
        const id = snap(R.pts[i][0], R.pts[i][1], h, ri);
        if (ids.length && ids[ids.length - 1].id === id) continue;
        ids.push({ id, i });
      }
      return ids;
    });
    const nodes = this.nodes;
    const nodeOf = (id) => { const p = P[id]; if (p.node < 0) { p.node = nodes.length; nodes.push({ id: nodes.length, x: p.x, z: p.z, h: p.h, ends: [], inc: [], out: [], conns: [], sig: null }); } return nodes[p.node]; };
    // ---- edges
    const edges = this.edges = [];
    roads.forEach((R, ri) => {
      const ids = rv[ri];
      if (ids.length < 2) return;
      const st = R.st;
      const isNode = (k) => k === 0 || k === ids.length - 1 || P[ids[k].id].n >= 2 || P[ids[k].id].self > 0;
      let start = 0;
      for (let k = 1; k < ids.length; k++) {
        if (!isNode(k)) continue;
        const a = nodeOf(ids[start].id), b = nodeOf(ids[k].id);
        const pts = [[a.x, a.z]];
        const hs = R.bridge && R.hs ? [P[ids[start].id].h] : null; // the deck profile (height reference for the bake)
        for (let q = start + 1; q < k; q++) { const v = R.pts[ids[q].i]; pts.push([v[0], v[1]]); if (hs) hs.push(R.hs[ids[q].i] || 0); }
        pts.push([b.x, b.z]);
        if (hs) hs.push(P[ids[k].id].h);
        start = k;
        const cum = cumPts(pts), len = cum[cum.length - 1];
        if (len < 0.4 || (a === b && len < 8)) continue;
        const e = {
          id: edges.length, pts, cum, len, a, b, st, bridge: !!R.bridge, filler: !!R.filler,
          rank: RANK[st.base] ?? 1, hw: st.hw,
          v0: R.filler ? 12 : (st.link ? Math.min(17, (SPEED[st.base] || 11) * 0.8) : (SPEED[st.base] || 11)) * (R.bridge && st.base === 'motorway' ? 1.05 : 1),
          trimA: 0, trimB: 0, name: R.name, hs,
        };
        edges.push(e);
        const dA = Math.min(6, len * 0.5), pA = pointAt(pts, cum, dA), pB = pointAt(pts, cum, len - dA);
        const ua = [pA[0] - a.x, pA[1] - a.z], ub = [pB[0] - b.x, pB[1] - b.z];
        const la = Math.hypot(ua[0], ua[1]) || 1, lb = Math.hypot(ub[0], ub[1]) || 1;
        a.ends.push({ e, start: true, dx: ua[0] / la, dz: ua[1] / la });
        b.ends.push({ e, start: false, dx: ub[0] / lb, dz: ub[1] / lb });
      }
    });
    // ---- trims: pull lane ends back from crossings (to the painted stop line where there is one)
    for (const n of nodes) {
      if (n.ends.length < 2) continue;
      for (const end of n.ends) {
        let r = 0, acute = 0;
        for (const o of n.ends) {
          if (o === end) continue;
          if (lineAngle(end.dx, end.dz, o.dx, o.dz) > 0.45) r = Math.max(r, o.e.hw);
          else if (end.dx * o.dx + end.dz * o.dz > 0) { // acute merge / diverge: lanes overlap until the roads separate
            const sa = Math.sin(Math.acos(Math.min(1, end.dx * o.dx + end.dz * o.dz)));
            acute = Math.max(acute, Math.min(28, (end.e.hw + o.e.hw + 0.6) / Math.max(0.25, sa)));
          }
        }
        let t = 0;
        if (r > 0) t = n.ends.length >= 3 ? r + (end.e.st.markings === 1 ? 4.4 : 0.8) : r * 0.8;
        t = Math.max(t, acute);
        if (n.ends.length === 2 && r === 0 && !acute) { // lane-count change at a continuation: taper over the connector
          const o = n.ends[0] === end ? n.ends[1] : n.ends[0];
          const dl = Math.abs(laneSplit(end.e.st)[end.start ? 1 : 0] - laneSplit(o.e.st)[o.start ? 0 : 1]) + Math.abs(laneSplit(end.e.st)[end.start ? 0 : 1] - laneSplit(o.e.st)[o.start ? 1 : 0]);
          if (dl > 0) t = Math.min(16, 8 + 4 * dl);
        }
        end.trim = t;
        if (end.start) end.e.trimA = Math.max(end.e.trimA, t); else end.e.trimB = Math.max(end.e.trimB, t);
      }
    }
    // ---- lanes
    const lanes = this.lanes;
    for (const e of edges) {
      let tA = e.trimA, tB = e.trimB;
      const avail = Math.max(0, e.len - 1.2);
      if (tA + tB > avail) { const k = avail / (tA + tB); tA *= k; tB *= k; }
      const centre = cutPolyline(e.pts, e.cum, tA, e.len - tB);
      const st = e.st, ch = Math.max(1.3, st.hw - st.parkW);
      let nf, nb, w;
      [nf, nb] = laneSplit(st);
      w = st.oneway ? (2 * ch) / nf : ch / nf;
      const mk = (pts, from, to, k, n, off, fwd) => {
        const [xs, zs] = offsetPolyline(pts, off);
        const cum = cumLen(xs, zs);
        const L = {
          id: lanes.length, conn: false, x: xs, z: zs, cum, len: cum[cum.length - 1], edge: e, fwd, k, n,
          from, to, v0: e.v0, rank: e.rank, bridge: e.bridge, outs: [], ctrl: CTRL_NONE, sig: null, grp: 0,
          occ: 0, veh: [], endTrim: fwd ? tB : tA,
        };
        lanes.push(L); from.out.push(L); to.inc.push(L);
        return L;
      };
      for (let k = 0; k < nf; k++) mk(centre, e.a, e.b, k, nf, st.oneway ? -ch + (k + 0.5) * w : (k + 0.5) * w, true);
      if (nb) {
        const rev = centre.slice().reverse();
        for (let k = 0; k < nb; k++) mk(rev, e.b, e.a, k, nb, (k + 0.5) * w, false);
      }
    }
    this.laneCount = lanes.length;
    this._dedupeLanes();
    // ---- connectors
    let nConn = 0;
    for (const n of nodes) {
      if (!n.inc.length || !n.out.length) continue;
      // outgoing lanes grouped by (edge, direction), sorted by k
      const groups = new Map();
      for (const L of n.out) { if (L.off) continue; const key = L.edge.id * 2 + (L.fwd ? 1 : 0); let g = groups.get(key); if (!g) { g = []; groups.set(key, g); } g.push(L); }
      for (const g of groups.values()) g.sort((a, b) => a.k - b.k);
      for (const Li of n.inc) {
        if (Li.off) continue;
        const ni = Li.x.length;
        let dix = Li.x[ni - 1] - Li.x[ni - 2], diz = Li.z[ni - 1] - Li.z[ni - 2];
        const dl = Math.hypot(dix, diz) || 1; dix /= dl; diz /= dl;
        const opts = [];
        for (const g of groups.values()) {
          const Lo = g[0];
          let dox = Lo.x[1] - Lo.x[0], doz = Lo.z[1] - Lo.z[0];
          const ol = Math.hypot(dox, doz) || 1; dox /= ol; doz /= ol;
          const a = Math.atan2(dix * doz - diz * dox, dix * dox + diz * doz);
          const sameEdge = Lo.edge === Li.edge && Lo.fwd !== Li.fwd;
          const turn = sameEdge || Math.abs(a) > 2.55 ? 'U' : Math.abs(a) < 0.6 ? 'S' : a > 0 ? 'R' : 'L';
          opts.push({ g, turn, a });
        }
        const lanePermits = (t) => Li.n === 1 || t === 'S' || (t === 'R' && Li.k === Li.n - 1) || (t === 'L' && Li.k === 0);
        let use = opts.filter((o) => o.turn !== 'U' && lanePermits(o.turn)), forced = false;
        if (!use.length) { use = opts.filter((o) => o.turn !== 'U'); forced = true; }
        if (!use.length) { use = opts; forced = false; }
        const nonU = opts.filter((o) => o.turn !== 'U').length;
        for (const o of use) {
          const g = o.g, no = g.length;
          let kk;
          if ((forced || nonU === 1) && o.turn !== 'U') kk = Math.max(0, Math.min(no - 1, Math.round(((Li.k + 0.5) * no) / Li.n - 0.5))); // only way on: keep lanes parallel
          else if (o.turn === 'R') kk = no - 1;
          else if (o.turn === 'L' || o.turn === 'U') kk = 0;
          else kk = Math.max(0, Math.min(no - 1, Math.round(((Li.k + 0.5) * no) / Li.n - 0.5)));
          this._makeConnector(n, Li, g[kk], o.turn, o.a);
          nConn++;
        }
      }
    }
    this._buildConflicts();
    this._buildZones();
    this.stats.nodes = nodes.length; this.stats.edges = edges.length; this.stats.lanes = this.laneCount; this.stats.connectors = nConn;
  }

  /** Disable lanes that run on top of another edge's lane for most of their length (duplicate OSM carriageways). */
  _dedupeLanes() {
    const lanes = this.lanes, cell = 8, H = new Map();
    for (const L of lanes) {
      for (let i = 0; i < L.x.length - 1; i++) {
        const sl = L.cum[i + 1] - L.cum[i];
        const i0 = Math.floor((Math.min(L.x[i], L.x[i + 1]) - 3) / cell), i1 = Math.floor((Math.max(L.x[i], L.x[i + 1]) + 3) / cell);
        const k0 = Math.floor((Math.min(L.z[i], L.z[i + 1]) - 3) / cell), k1 = Math.floor((Math.max(L.z[i], L.z[i + 1]) + 3) / cell);
        if (!(sl > 1e-3)) continue;
        for (let a = i0; a <= i1; a++) for (let b = k0; b <= k1; b++) {
          const k = ckey(a, b); let arr = H.get(k); if (!arr) { arr = []; H.set(k, arr); } arr.push(L.id, i);
        }
      }
    }
    this._laneHash = H; this._laneCell = cell;
    const score = (L) => L.rank * 100 + L.edge.st.lanes * 10 + Math.min(9, L.len / 50);
    let off = 0;
    for (const L of lanes) {
      if (L.len < 12) continue;
      let hits = 0, tot = 0;
      const votes = new Map();
      for (let s = 2; s < L.len - 2; s += 4) {
        sampleLane(L, s, P0); tot++;
        const arr = H.get(ckey(Math.floor(P0.x / cell), Math.floor(P0.z / cell)));
        if (!arr) continue;
        let hit = -1;
        for (let j = 0; j < arr.length && hit < 0; j += 2) {
          const M = lanes[arr[j]]; if (M.edge === L.edge || M.off) continue;
          const i = arr[j + 1];
          const dx = M.x[i + 1] - M.x[i], dz = M.z[i + 1] - M.z[i], l = Math.hypot(dx, dz) || 1;
          if (Math.abs((dx * P0.dx + dz * P0.dz) / l) < 0.8) continue;
          if (ptSegD2(P0.x, P0.z, M.x[i], M.z[i], M.x[i + 1], M.z[i + 1]) < 2.2 * 2.2) hit = M.id;
        }
        if (hit >= 0) { hits++; votes.set(hit, (votes.get(hit) || 0) + 1); }
      }
      if (!tot || hits / tot < 0.5) continue;
      let other = -1, best = 0;
      for (const [id, c] of votes) if (c > best) { best = c; other = id; }
      const M = lanes[other];
      if (score(L) <= score(M)) { L.off = true; off++; }
    }
    this.stats.lanesOff = off;
  }

  /** Conflict sets between connectors whose swept paths come within CONF_D (also across nearby nodes). */
  _buildConflicts() {
    const cs = this.lanes.filter((l) => l.conn), cell = 24, H = new Map();
    for (const c of cs) {
      let a = Infinity, b = -Infinity, d = Infinity, e = -Infinity;
      for (let i = 0; i < c.x.length; i++) { a = Math.min(a, c.x[i]); b = Math.max(b, c.x[i]); d = Math.min(d, c.z[i]); e = Math.max(e, c.z[i]); }
      c.bb = [a - CONF_D, b + CONF_D, d - CONF_D, e + CONF_D];
      for (let i = Math.floor(c.bb[0] / cell); i <= Math.floor(c.bb[1] / cell); i++) for (let k = Math.floor(c.bb[2] / cell); k <= Math.floor(c.bb[3] / cell); k++) {
        const key = ckey(i, k); let arr = H.get(key); if (!arr) { arr = []; H.set(key, arr); } arr.push(c);
      }
    }
    const seen = new Set(), D2 = CONF_D * CONF_D;
    let n = 0;
    for (const arr of H.values()) {
      for (let i = 0; i < arr.length; i++) for (let j = i + 1; j < arr.length; j++) {
        let A = arr[i], B = arr[j];
        if (A.id > B.id) { const t = A; A = B; B = t; }
        const key = A.id * 65536 + B.id;
        if (seen.has(key)) continue;
        seen.add(key);
        if (A.src === B.src || A.dst === B.src || B.dst === A.src) continue; // same source (car-following) / sequential
        if (A.bb[1] < B.bb[0] || B.bb[1] < A.bb[0] || A.bb[3] < B.bb[2] || B.bb[3] < A.bb[2]) continue;
        // straight pairs: centrelines 2.5 m apart are fine; turning bodies sweep wider
        const dd = A.turn === 'S' && B.turn === 'S' ? 2.5 * 2.5 : D2;
        if (A.dst === B.dst || polyMinD2(A, B, dd) < dd) { A.conf.push(B.id); B.conf.push(A.id); n++; }
      }
    }
    this.stats.conflicts = n;
  }

  /** Stretches of plain lanes swept by connectors of other roads (messy multi-way junctions). */
  _buildZones() {
    const lanes = this.lanes, H = this._laneHash, cell = this._laneCell, D2 = CONF_D * CONF_D;
    let n = 0;
    for (const C of lanes) {
      if (!C.conn) continue;
      C.lz = [];
      const hit = new Map();
      for (let j = 0; j < C.x.length - 1; j++) {
        const i0 = Math.floor((Math.min(C.x[j], C.x[j + 1]) - CONF_D) / cell), i1 = Math.floor((Math.max(C.x[j], C.x[j + 1]) + CONF_D) / cell);
        const k0 = Math.floor((Math.min(C.z[j], C.z[j + 1]) - CONF_D) / cell), k1 = Math.floor((Math.max(C.z[j], C.z[j + 1]) + CONF_D) / cell);
        for (let a = i0; a <= i1; a++) for (let b = k0; b <= k1; b++) {
          const arr = H.get(ckey(a, b)); if (!arr) continue;
          for (let q = 0; q < arr.length; q += 2) {
            const M = lanes[arr[q]];
            if (M.off || M.id === C.src || M.id === C.dst) continue;
            const i = arr[q + 1];
            if (segSegD2(C.x[j], C.z[j], C.x[j + 1], C.z[j + 1], M.x[i], M.z[i], M.x[i + 1], M.z[i + 1]) >= D2) continue;
            let r = hit.get(M.id); if (!r) { r = [Infinity, -Infinity]; hit.set(M.id, r); }
            r[0] = Math.min(r[0], M.cum[i]); r[1] = Math.max(r[1], M.cum[i + 1]);
          }
        }
      }
      for (const [id, r] of hit) {
        const M = lanes[id];
        if (r[1] < 2.5) continue; // grazes the very start of a lane (exit of the same junction): ignore
        const z = { lane: M, conn: C, s0: Math.max(0, r[0]), s1: Math.min(M.len, r[1]) };
        C.lz.push(z);
        (M.zones || (M.zones = [])).push(z);
        n++;
      }
    }
    this.stats.zones = n;
  }

  _makeConnector(node, Li, Lo, turn, ang) {
    const ni = Li.x.length;
    const x0 = Li.x[ni - 1], z0 = Li.z[ni - 1], x3 = Lo.x[0], z3 = Lo.z[0];
    let d0x = x0 - Li.x[ni - 2], d0z = z0 - Li.z[ni - 2]; const l0 = Math.hypot(d0x, d0z) || 1; d0x /= l0; d0z /= l0;
    let d3x = Lo.x[1] - x3, d3z = Lo.z[1] - z3; const l3 = Math.hypot(d3x, d3z) || 1; d3x /= l3; d3z /= l3;
    const dist = Math.hypot(x3 - x0, z3 - z0);
    const xs = [], zs = [];
    if (dist < 0.6) { xs.push(x0, x3); zs.push(z0, z3); if (dist < 1e-3) { xs[1] += d3x * 0.05; zs[1] += d3z * 0.05; } }
    else {
      const h = turn === 'U' ? Math.max(3, dist * 0.9) : dist * 0.42;
      const x1 = x0 + d0x * h, z1 = z0 + d0z * h, x2 = x3 - d3x * h, z2 = z3 - d3z * h;
      const segs = Math.max(2, Math.min(10, Math.ceil(dist / 2.2)));
      for (let i = 0; i <= segs; i++) {
        const t = i / segs, u = 1 - t;
        const b0 = u * u * u, b1 = 3 * u * u * t, b2 = 3 * u * t * t, b3 = t * t * t;
        xs.push(b0 * x0 + b1 * x1 + b2 * x2 + b3 * x3); zs.push(b0 * z0 + b1 * z1 + b2 * z2 + b3 * z3);
      }
    }
    const cum = cumLen(xs, zs), len = cum[cum.length - 1];
    let v0 = Math.min(Li.v0, Lo.v0);
    const turnA = Math.abs(ang);
    if (turnA > 0.2 && len > 1) v0 = Math.min(v0, Math.max(3.5, Math.sqrt(2.6 * (len / turnA))));
    if (turn !== 'S') v0 = Math.min(v0, turn === 'U' ? 4 : 8);
    const c = {
      id: this.lanes.length, conn: true, x: xs, z: zs, cum, len, node, src: Li.id, dst: Lo.id, turn, w: TURN_W[turn],
      v0, rank: Lo.rank, bridge: Li.bridge || Lo.bridge, conf: [], occ: 0, holders: [], veh: [], outs: [], ctrl: CTRL_NONE,
    };
    this.lanes.push(c);
    Li.outs.push(c.id);
    node.conns.push(c);
    return c;
  }

  _buildControls() {
    const infos = this.sigInfos;
    for (const n of this.nodes) {
      if (n.ends.length < 3) continue;
      let best = null, bd = 20 * 20;
      for (const f of infos) { const d = (f.x - n.x) ** 2 + (f.z - n.z) ** 2; if (d < bd) { bd = d; best = f; } }
      n.sig = best;
    }
    let ns = 0, nst = 0;
    for (const n of this.nodes) {
      if (n.ends.length < 3) continue;
      for (const L of n.inc) {
        if (!(L.endTrim > 0.5) || L.bridge || L.rank >= 5) continue;
        if (n.sig) {
          if (L.from.sig === n.sig) continue; // already inside this junction (dual carriageway)
          const ni = L.x.length;
          let dx = L.x[ni - 1] - L.x[ni - 2], dz = L.z[ni - 1] - L.z[ni - 2]; const l = Math.hypot(dx, dz) || 1; dx /= l; dz /= l;
          const f = n.sig;
          L.ctrl = CTRL_SIGNAL; L.sig = f;
          L.grp = Math.abs(dx * f.a[0] + dz * f.a[1]) >= Math.abs(dx * f.b[0] + dz * f.b[1]) ? 0 : 1;
          ns++;
          continue;
        }
        // unsignalised crossing: minor approaches stop; low-class equal approaches all-way stop (SF style)
        let maxOther = -1;
        const ni = L.x.length;
        let dx = L.x[ni - 1] - L.x[ni - 2], dz = L.z[ni - 1] - L.z[ni - 2]; const l = Math.hypot(dx, dz) || 1; dx /= l; dz /= l;
        for (const o of n.ends) {
          if (o.e === L.edge) continue;
          if (lineAngle(dx, dz, o.dx, o.dz) > 0.45) maxOther = Math.max(maxOther, o.e.rank);
        }
        if (maxOther < 0) continue;
        if (L.rank < maxOther || (L.rank <= 2 && maxOther <= 2)) { L.ctrl = CTRL_STOP; nst++; }
        else L.ctrl = CTRL_YIELD;
      }
    }
    this.stats.signals = infos.length; this.stats.signalLanes = ns; this.stats.stopLanes = nst;
  }

  // ================================================================ spawn points
  _buildSpawnPoints() {
    const sp = this.spawnPts = [];
    this._spGrid = new Map();
    for (const L of this.lanes) {
      if (L.conn || L.off || L.len < 24 || !L.outs.length) continue;
      for (let s = 8; s < L.len - 14; s += 22) {
        sampleLane(L, s, P0);
        const id = sp.length;
        sp.push({ lane: L.id, s, x: P0.x, z: P0.z });
        const k = ckey(Math.floor(P0.x / 64), Math.floor(P0.z / 64));
        let a = this._spGrid.get(k); if (!a) { a = []; this._spGrid.set(k, a); } a.push(id);
      }
    }
    this.stats.spawnPoints = sp.length;
  }

  _anyNear(x, z, r) {
    const G = this._grid, r2 = r * r;
    for (let i = Math.floor((x - r) / CELL); i <= Math.floor((x + r) / CELL); i++) for (let k = Math.floor((z - r) / CELL); k <= Math.floor((z + r) / CELL); k++) {
      const a = G.get(ckey(i, k)); if (!a) continue;
      for (const u of a) if (u.active && (u.x - x) ** 2 + (u.z - z) ** 2 < r2) return true;
    }
    return false;
  }

  _laneFree(lane, s, gap) {
    for (const u of this.lanes[lane].veh) if (Math.abs(u.s - s) < gap + u.L * 0.5) return false;
    return true;
  }

  _visible(x, y, z) {
    const c = this._cam;
    const d2 = (x - c.x) ** 2 + (z - c.z) ** 2;
    if (!this._hasFrustum) return d2 < 340 * 340;
    if (d2 > 2600 * 2600) return false;
    return this._frustum.intersectsSphere(_sph.set(_p.set(x, y + 1.5, z), 9));
  }

  _pickSpawn(cx, cz, rmin, rmax, hidden) {
    const sp = this.spawnPts;
    if (!sp.length) return null;
    for (let t = 0; t < 36; t++) {
      let p;
      if (t < 30) {
        const r = rmin + this.rnd() * (rmax - rmin), th = this.rnd() * Math.PI * 2;
        const a = this._spGrid.get(ckey(Math.floor((cx + r * Math.cos(th)) / 64), Math.floor((cz + r * Math.sin(th)) / 64)));
        if (!a) continue;
        p = sp[a[Math.floor(this.rnd() * a.length)]];
      } else {
        p = sp[Math.floor(this.rnd() * sp.length)];
        const d2 = (p.x - cx) ** 2 + (p.z - cz) ** 2;
        if (d2 > rmax * rmax || d2 < rmin * rmin) continue;
      }
      if (hidden && this._visible(p.x, laneY(this.lanes[p.lane], p.s), p.z)) continue;
      if (!this._laneFree(p.lane, p.s, 16) || this._anyNear(p.x, p.z, 25)) continue;
      return p;
    }
    return null;
  }

  _choose(id) {
    if (id < 0) return -1;
    const L = this.lanes[id];
    if (L.conn) return L.dst;
    const o = L.outs;
    if (!o.length) return -1;
    if (o.length === 1) return o[0];
    let tot = 0;
    for (const c of o) tot += this.lanes[c].w;
    let r = this.rnd() * tot;
    for (const c of o) { r -= this.lanes[c].w; if (r <= 0) return c; }
    return o[o.length - 1];
  }

  _place(v, p) {
    this._releaseAll(v);
    const L = this.lanes[p.lane];
    v.lane = L.id; v.s = p.s; v.prev = -1;
    v.path[0] = this._choose(L.id);
    for (let k = 1; k < PATH_N; k++) v.path[k] = this._choose(v.path[k - 1]);
    v.v = L.v0 * v.drv * (0.55 + this.rnd() * 0.3);
    v.hardT = 0; v.hardBy = null;
    v.acc = 0; v.stopT = 0; v.stopDone = -1; v.sigGo = -1; v.waitT = 0; v.deadT = 0; v.idleT = 0; v.stun = 0; v.brake = 0;
    v.y = laneY(L, p.s); v.respawned = true;
    L.veh.push(v); if (L.veh.length === 1) this._touched.push(L);
    this._pose(v, 0, true);
  }

  // ================================================================ ground heights (baked; the sim never sees the terrain)
  /**
   * Bake every lane's ground height with groundAt(x, z, yRef) (see header): sampled every <= HSTEP_FINE m, then kept at
   * 2x / 4x that spacing where the dropped samples lie within HTOL of the straight line between the kept ones (flat /
   * evenly graded streets), so hills and deck ramps keep the fine samples.
   */
  _bakeHeights() {
    const lanes = this.lanes, nf = new Uint32Array(lanes.length);
    let totalF = 0;
    for (const L of lanes) totalF += (nf[L.id] = L.off ? 1 : 4 * Math.max(1, Math.ceil(L.len / (4 * HSTEP_FINE))) + 1);
    const F = new Float32Array(totalF), off = new Uint32Array(lanes.length);
    let o = 0;
    for (const L of lanes) { off[L.id] = o; L.hy = F.subarray(o, o + nf[L.id]); L.hk = nf[L.id] > 1 ? (nf[L.id] - 1) / L.len : 0; o += nf[L.id]; }
    // plain lanes first (id order), then connectors: a connector's walk starts where its source lane ends
    for (const L of lanes) if (!L.conn && !L.off) this._bakeLane(L);
    for (const L of lanes) if (L.conn) this._bakeLane(L);
    // decimate where linear interpolation reproduces the fine samples
    const fac = new Uint8Array(lanes.length);
    let total = 0;
    for (const L of lanes) {
      const h = L.hy, n = h.length;
      let k = 1;
      if (n > 1) for (const f of [4, 2]) {
        let ok = true;
        for (let a = 0; a + f < n && ok; a += f) for (let j = 1; j < f; j++) { const v = h[a] + ((h[a + f] - h[a]) * j) / f; if (Math.abs(v - h[a + j]) > HTOL) { ok = false; break; } }
        if (ok) { k = f; break; }
      }
      fac[L.id] = k;
      total += n > 1 ? (n - 1) / k + 1 : 1;
    }
    const H = new Float32Array(total), index = new Uint32Array(lanes.length + 1);
    o = 0;
    for (const L of lanes) {
      const src = L.hy, k = fac[L.id], n = src.length > 1 ? (src.length - 1) / k + 1 : 1;
      index[L.id] = o;
      for (let i = 0; i < n; i++) H[o + i] = src[i * k];
      L.hy = H.subarray(o, o + n); L.hk = n > 1 ? (n - 1) / L.len : 0;
      o += n;
    }
    index[lanes.length] = o;
    this.baked = { heights: H, index };
    this.stats.heightSamples = total;
  }

  _bakeLane(L) {
    const G = this.groundAt, h = L.hy, n = h.length, bridge = L.bridge;
    if (!G) { h.fill(0); return; }
    const e = L.conn ? null : L.edge, ref = bridge && e && e.hs ? e : null;
    let prev = L.conn ? laneY(this.lanes[L.src], this.lanes[L.src].len) : NaN;
    for (let k = 0; k < n; k++) {
      const s = n > 1 ? Math.min(L.len, k / L.hk) : 0;
      sampleLane(L, s, P2);
      // the height a vehicle at the previous sample would query (Traffic.js before the bake: getGroundHeight(x, z, y)
      // on bridges, (x, z, y - 1.5) on ground lanes so decks overhead are ignored)
      let yRef;
      if (ref) yRef = edgeRefHeight(ref, P2.x, P2.z);
      else if (prev === prev) yRef = bridge ? prev : prev - 1.5;
      else yRef = bridge ? undefined : -1e4; // first sample: highest surface on bridges, bare ground otherwise
      let y = G(P2.x, P2.z, yRef);
      if (!Number.isFinite(y)) y = prev === prev ? prev : 0;
      h[k] = y; prev = y;
    }
  }

  /** Use heights baked elsewhere (the main thread bakes, the worker attaches). */
  _attachHeights({ heights, index }) {
    if (!index || index.length !== this.lanes.length + 1) throw new Error(`[TrafficSim] baked heights for ${index ? index.length - 1 : 0} lanes, graph has ${this.lanes.length}`);
    for (const L of this.lanes) {
      const o = index[L.id], n = index[L.id + 1] - o;
      L.hy = heights.subarray(o, o + n); L.hk = n > 1 ? (n - 1) / L.len : 0;
    }
    this.baked = { heights, index };
  }

  /**
   * Dev check of the bake: baked height vs groundAt(x, z, yRef) (as the pre-bake sim queried it at a vehicle axle) every
   * `step` m along every lane. Returns { samples, max, p99, p999, over10cm, worst: { lane, s, x, z, baked, live, bridge } }.
   */
  bakeError(groundAt, step = 0.5) {
    const errs = [], by = {};
    let worst = null;
    for (const L of this.lanes) {
      if (L.off || !L.hy) continue;
      for (let s = 0; s <= L.len; s += step) {
        sampleLane(L, s, P2);
        const yb = laneY(L, s), live = groundAt(P2.x, P2.z, L.bridge ? yb : yb - 1.5);
        if (!Number.isFinite(live)) continue;
        const e = Math.abs(yb - live);
        errs.push(e);
        if (e > 0.1) { const k = (L.bridge ? 'bridge' : 'ground') + (L.conn ? 'Conn' : 'Lane'); by[k] = (by[k] || 0) + 1; }
        if (!worst || e > worst.err) worst = { err: e, lane: L.id, conn: !!L.conn, s: +s.toFixed(1), x: +P2.x.toFixed(1), z: +P2.z.toFixed(1), baked: +yb.toFixed(3), live: +live.toFixed(3), bridge: !!L.bridge };
      }
    }
    errs.sort((a, b) => a - b);
    const q = (f) => +errs[Math.min(errs.length - 1, Math.floor(errs.length * f))].toFixed(4);
    return { samples: errs.length, max: q(1), p99: q(0.99), p999: q(0.999), p9999: q(0.9999), over10cm: errs.filter((e) => e > 0.1).length, over10cmBy: by, worst };
  }

  /** Baked ground height on lane `id` at arc length s. */
  laneHeight(id, s) { const L = this.lanes[id]; return L && L.hy ? laneY(L, s) : 0; }

  // ================================================================ signals (same state as props.signals)
  /** 0 red, 1 yellow, 2 green for phase group g (0 = props road A, 1 = road B). */
  _light(info, g) {
    let st = this.sigState[info.si];
    if (st === SIG_UNKNOWN || st === undefined) { const t = (this.time + (info.offset || 0)) % 40; st = t < 16 ? 0 : t < 19 ? 1 : t < 20 ? 2 : t < 36 ? 3 : t < 39 ? 4 : 5; }
    if (st === 0) return g === 0 ? 2 : 0;
    if (st === 1) return g === 0 ? 1 : 0;
    if (st === 3) return g === 0 ? 0 : 2;
    if (st === 4) return g === 0 ? 0 : 1;
    return 0;
  }

  // ================================================================ vehicles
  _buildVehicles() {
    const counts = typeCounts(this.target), col = new THREE.Color();
    TRAFFIC_TYPES.forEach((T, ti) => {
      for (let i = 0; i < counts[ti]; i++) {
        pickVehicleColor(this.rnd, T.pal, col);
        const rgb = [col.r, col.g, col.b];
        const id = this.vehicles.length;
        this.vehicles.push({
          id, type: ti, ii: i, L: T.L, W: T.W, Hc: T.Hc, WB: T.WB,
          amax: T.amax * (0.9 + this.rnd() * 0.2), drv: 0.88 + this.rnd() * 0.22, T: HEADWAY * (0.85 + this.rnd() * 0.35),
          lane: -1, prev: -1, s: 0, v: 0, acc: 0, path: new Array(PATH_N).fill(-1), res: [], stopT: 0, stopDone: -1, sigGo: -1, waitT: 0, deadT: 0, idleT: 0, stun: 0,
          x: 0, y: 0, z: 0, yaw: 0, pitch: 0, fx: 0, fz: -1, brake: 0, yawRate: 0, active: false, respawned: false,
          box: { minX: 0, minY: -1e4, maxX: 0, maxY: -1e4, minZ: 0, maxZ: 0 }, // overall bounds (quick reject only)
          boxes: Array.from({ length: MAX_SLICES }, () => ({ minX: 0, minY: -1e4, maxX: 0, maxY: -1e4, minZ: 0, maxZ: 0, kind: 'vehicle', id, vx: 0, vz: 0 })),
          nb: 0, dsAcc: 0, hardT: 0, hardBy: null, rgb,
        });
      }
    });
    this.typeCounts = counts;
  }

  _spawnInitial() {
    // spread around the spawn (camera unknown during load): uniform in radius => denser near the player
    const sp = this.spawn;
    this._cam.set(sp.x, 30, sp.z);
    let n = 0;
    for (const v of this.vehicles) {
      let p = null;
      for (let t = 0; t < 6 && !p; t++) {
        p = this._pickSpawn(sp.x, sp.z, 14, 620, false);
        if (p && (p.x - sp.x) ** 2 + (p.z - sp.z) ** 2 < 16 * 16) p = null;
      }
      if (!p) continue;
      this._place(v, p);
      v.active = true; n++;
    }
    this.count = n;
    this._rebuildGrid();
  }

  // ================================================================ simulation
  /** Advance by dt seconds (sub-stepped at <= 50 ms): AI steps, respawns, poses, collider grid. */
  update(dt, cameraPos) {
    if (!this.vehicles.length) return;
    dt = Number.isFinite(dt) ? Math.min(Math.max(dt, 0), 0.1) : 0;
    if (cameraPos) this._cam.copy(cameraPos);
    this.time += dt;
    const p = this._player;
    this._pl = !p ? null : (typeof p !== 'function' && Number.isFinite(p.hl)) ? p : playerInfoFrom(p, this._plOut);
    if (dt > 0) {
      const n = Math.ceil(dt / 0.05), h = dt / n;
      for (let i = 0; i < n; i++) this._step(h);
    }
    this._recycle();
    for (const v of this.vehicles) {
      if (!v.active) continue;
      const y0 = v.yaw;
      this._pose(v, dt, false);
      v.yawRate = dt > 0 && !v.respawned ? wrapPi(v.yaw - y0) / dt : 0;
    }
    this._rebuildGrid();
  }

  /** Pack every vehicle's pose: POSE_STRIDE floats (x, y, z, yaw, pitch, brake, speed, yaw rate, flags). */
  writePoses(out) {
    const V = this.vehicles, S = POSE_STRIDE;
    for (let i = 0; i < V.length; i++) {
      const v = V[i], o = i * S;
      out[o] = v.x; out[o + 1] = v.y; out[o + 2] = v.z; out[o + 3] = v.yaw; out[o + 4] = v.pitch; out[o + 5] = v.brake;
      out[o + 6] = v.v; out[o + 7] = v.yawRate; out[o + 8] = (v.active ? 1 : 0) | (v.respawned ? 2 : 0);
      v.respawned = false;
    }
    return out;
  }

  _pose(v, dt, snap) {
    const L = this.lanes[v.lane];
    // both axles on the path: body centre between them, yaw along rear -> front; heights = the baked ground at the axles
    const hb2 = v.WB * 0.5;
    this._ahead(v, hb2, P1);
    this._behind(v, hb2, P0);
    let dx = P1.x - P0.x, dz = P1.z - P0.z;
    const d = Math.hypot(dx, dz);
    if (d < 0.2) { sampleLane(L, v.s, P0); dx = P0.dx; dz = P0.dz; P1.x = P0.x; P1.z = P0.z; P1.y = P0.y; } else { dx /= d; dz /= d; }
    const ty = Math.atan2(-dx, -dz);
    if (snap) v.yaw = ty; else v.yaw = wrapPi(v.yaw + wrapPi(ty - v.yaw) * Math.min(1, dt * 20));
    const fx = -Math.sin(v.yaw), fz = -Math.cos(v.yaw);
    v.fx = fx; v.fz = fz;
    const x = (P0.x + P1.x) * 0.5, z = (P0.z + P1.z) * 0.5;
    let hf = P1.y, hr = P0.y;
    if (!Number.isFinite(hf)) hf = v.y; if (!Number.isFinite(hr)) hr = v.y;
    v.x = x; v.z = z; v.y = (hf + hr) * 0.5;
    v.pitch = Math.max(-0.4, Math.min(0.4, Math.atan2(hf - hr, v.WB)));
    // brake lamp: decelerating or holding at standstill
    const tb = v.acc < -0.8 ? Math.min(1, 0.45 - v.acc / 4) : v.v < 0.4 ? 0.75 : 0;
    v.brake += (tb - v.brake) * Math.min(1, (dt || 0.016) * 12);
    vehicleSlices(v, x, z, fx, fz, Math.min(hf, hr), v.y + v.Hc, v.v);
    v.dsAcc = 0;
  }


  _step(h) {
    const V = this.vehicles, lanes = this.lanes;
    for (const L of this._touched) L.veh.length = 0;
    this._touched.length = 0;
    for (const v of V) {
      if (!v.active) continue;
      const L = lanes[v.lane];
      if (!L.veh.length) this._touched.push(L);
      L.veh.push(v);
    }
    for (const L of this._touched) {
      const a = L.veh;
      if (a.length > 1) a.sort((p, q) => p.s - q.s);
      for (let i = 0; i < a.length; i++) a[i].bi = i;
    }
    for (const v of V) if (v.active) v.acc = this._accel(v, h);
    for (const v of V) {
      if (!v.active) continue;
      v.v = Math.max(0, v.v + v.acc * h);
      if (v.stun > 0) v.stun -= h;
      let ds = v.v * h;
      if (ds > 0 && this._hardBlocked(v, ds)) { ds = 0; v.v = 0; v.acc = Math.min(v.acc, 0); if (v.hardBy) v.hardT += h; }
      else if (ds > 0) { v.hardT = 0; v.hardBy = null; }
      v.idleT = v.v < 0.1 ? v.idleT + h : 0;
      v.dsAcc += ds;
      v.s += ds;
      let L = lanes[v.lane];
      let guard = 0;
      while (v.s > L.len && guard++ < 8) {
        const nid = v.path[0];
        if (nid < 0) { v.s = L.len; v.v = 0; v.deadT += h; break; }
        v.s -= L.len;
        this._enter(v, nid);
        L = lanes[nid];
      }
    }
  }

  _reserve(v, c) { if (v.res.indexOf(c.id) < 0) { v.res.push(c.id); c.occ++; c.holders.push(v); } }
  _release(v, c) {
    const i = v.res.indexOf(c.id);
    if (i < 0) return;
    v.res.splice(i, 1); c.occ--;
    const j = c.holders.indexOf(v); if (j >= 0) c.holders.splice(j, 1);
  }
  _releaseAll(v) { while (v.res.length) this._release(v, this.lanes[v.res[0]]); }

  _enter(v, nid) {
    const lanes = this.lanes;
    const old = lanes[v.lane], N = lanes[nid];
    if (!old.conn && N.conn && old.ctrl === CTRL_SIGNAL) { // diagnostics: junction entries at signals / on red
      this.sigEntries++;
      if (this._light(old.sig, old.grp) === 0 && v.sigGo !== old.id) this.redEntries++;
    }
    if (old.conn) this._release(v, old);
    if (N.conn) this._reserve(v, N);
    v.prev = v.lane; v.lane = nid;
    const P = v.path;
    for (let k = 0; k < PATH_N - 1; k++) P[k] = P[k + 1];
    P[PATH_N - 1] = this._choose(P[PATH_N - 2]);
    if (P[0] < 0) { P[0] = this._choose(nid); for (let k = 1; k < PATH_N; k++) P[k] = this._choose(P[k - 1]); }
    v.stopT = 0; v.waitT = 0;
  }

  /** Connector Cn is unusable now: a conflicting connector is reserved, or its exit lane is backed up. */
  _connBlocked(v, Cn) {
    const lanes = this.lanes;
    // after a long wait, ignore reservations held by vehicles that are themselves stuck - never vehicles in the junction
    for (const cid of Cn.conf) {
      const K = lanes[cid];
      if (K.occ <= 0 || v.res.indexOf(cid) >= 0) continue;
      if (K.veh.length) return 'conf' + cid; // never into a junction path crossed by a vehicle that is in it
      if (v.waitT < 12) return 'conf' + cid;
      for (const u of K.holders) if (u.v > 0.3) return 'conf' + cid;
    }
    const D = lanes[Cn.dst];
    if (D.veh.length) { const u = D.veh[0]; if (u !== v && u.s - u.L * 0.5 < v.L + 2 && u.v < 3) return 'full'; }
    if (Cn.lz) for (const z of Cn.lz) for (const u of z.lane.veh) {
      if (u === v || u.lane === v.lane) continue; // our own lane: those are followers
      const f = u.s + u.L * 0.5, r = u.s - u.L * 0.5;
      if (f > z.s0 - 1 && r < z.s1 + 1) {                              // inside the swept stretch
        // ...unless it is itself held at its own stop line waiting for a junction (mutual wait): older waiter goes
        const atGate = u.v < 0.3 && f > z.lane.len - 3 && u.waitT > 4;
        if (!(atGate && v.waitT > 4 && (v.waitT > u.waitT + 1 || (Math.abs(v.waitT - u.waitT) <= 1 && v.id < u.id)))) return 'zone#' + u.id;
      } else if (u.v > 0.5 && f <= z.s0 - 1 && z.s0 - f < u.v * 2.5 + 2) return 'arriving#' + u.id; // about to arrive
    }
    if (D.zones) for (const z of D.zones) if (z.conn.occ > 0 && v.res.indexOf(z.conn.id) < 0 && !this._holderBehind(v, z.conn)) return 'exitzone' + z.conn.id; // exit is swept
    return '';
  }

  /** Gate from normal lane G into connector Cn, distEnd = front bumper to the stop line. 0 go, 1 control, 2 conflict. */
  _gate(v, G, Cn, distEnd, h, afterId) {
    const vel = v.v;
    const commitD = Math.max(4, (vel * vel) / (2 * 2.4) + 3);
    if (v.res.indexOf(Cn.id) >= 0) {
      // light changed after commitment but we can still stop comfortably, or we fell back (slow leader): release
      if ((G.ctrl === CTRL_SIGNAL && v.sigGo !== G.id && distEnd > (vel * vel) / 6 + 1.5 && this._light(G.sig, G.grp) !== 2) || distEnd > commitD + 4) this._release(v, Cn);
      else return 0;
    }
    let block = 0;
    if (G.ctrl === CTRL_SIGNAL) {
      const li = this._light(G.sig, G.grp);
      if (li === 0) block = 1;
      else if (li === 1 && v.sigGo !== G.id) { if (distEnd > (vel * vel) / (2 * 3.2) + 0.5) block = 1; else v.sigGo = G.id; }
    } else if (G.ctrl === CTRL_STOP && v.stopDone !== G.id) {
      block = 1;
      if (vel < 0.5 && distEnd < 4) { v.stopT += h; if (v.stopT > 0.7) { v.stopDone = G.id; block = 0; } }
    }
    if (!block && distEnd < 45) {
      let conf = this._connBlocked(v, Cn);
      // don't block the box: a very short exit lane means the next junction must be free too
      if (!conf && afterId >= 0 && this.lanes[Cn.dst].len < v.L + 4 && this.lanes[afterId].conn) conf = this._connBlocked(v, this.lanes[afterId]) && 'next:' + this._connBlocked(v, this.lanes[afterId]);
      if (conf) { block = 2; v.whyJ = conf; if (vel < 0.6) v.waitT += h; }
    }
    if (!block && distEnd < commitD) this._reserve(v, Cn);
    return block;
  }

  /** A holder of connector C is queued behind v (yielding to it would deadlock). */
  _holderBehind(v, C) {
    for (const u of C.holders) if (u.lane === v.lane || u.path.indexOf(v.lane) >= 0) return true;
    return false;
  }

  /** u is ahead of v and overlapping its corridor. */
  _ahead2(v, u) {
    const dx = u.x - v.x, dz = u.z - v.z;
    return dx * v.fx + dz * v.fz > 0.5 && Math.abs(-dx * v.fz + dz * v.fx) < (v.W + u.W) * 0.5 + 0.4;
  }

  _accel(v, h) {
    const lanes = this.lanes, L = lanes[v.lane], vel = v.v, T = v.T, amax = v.amax;
    let v0 = L.v0 * (L.conn ? Math.min(1, v.drv) : v.drv);
    let worst = 0, why = ''; // most negative IDM interaction term (+ its cause, for debugging)
    const sq = 2 * Math.sqrt(amax * B_COMF);
    const obst = (gap, vl, s0, tag) => {
      const ss = s0 + Math.max(0, vel * T + (vel * (vel - vl)) / sq);
      const g = Math.max(gap, 0.08);
      const r = ss / g, a = -amax * r * r;
      if (a < worst) { worst = a; why = tag; }
    };
    const halfL = v.L * 0.5;
    // ---- leader along the lane / planned path
    const bk = L.veh;
    if (v.bi + 1 < bk.length) {
      const u = bk[v.bi + 1];
      obst(u.s - v.s - (u.L * 0.5 + halfL), u.v, GAP0, 'lead');
    } else {
      if (L.conn) { // vehicles ahead on sibling connectors (same source lane) overlap until the paths separate
        for (const cid of lanes[L.src].outs) {
          if (cid === L.id) continue;
          for (const u of lanes[cid].veh) if (u.s > v.s && this._ahead2(v, u)) obst(u.s - v.s - (u.L * 0.5 + halfL), u.v, GAP0, 'sibling');
        }
      }
      let dist = L.len - v.s;
      for (let k = 0; k < PATH_N && dist < 160; k++) {
        const nid = v.path[k]; if (nid < 0) break;
        const N = lanes[nid];
        let first = N.veh.length ? N.veh[0] : null;
        if (N.conn) {
          for (const cid of lanes[N.src].outs) {
            if (cid === nid) continue;
            for (const u of lanes[cid].veh) if (this._ahead2(v, u)) { if (!first || u.s < first.s) first = u; break; }
          }
        }
        if (first) { obst(dist + first.s - (first.L * 0.5 + halfL), first.v, GAP0, 'path'); break; }
        dist += N.len;
      }
    }
    // ---- swept zones on this plain lane: stop short while their connector is reserved (never inside a junction)
    if (!L.conn && L.zones) {
      for (const z of L.zones) {
        if (z.conn.occ <= 0 || v.res.indexOf(z.conn.id) >= 0) continue;
        const d = z.s0 - v.s - halfL;
        if (!(d > -0.5 && d < 70) || this._holderBehind(v, z.conn)) continue;
        obst(d - 0.5, 0, 1.0, 'zone');
      }
    }
    // ---- proximity guard: anything close ahead in our corridor heading the same way (catch-all)
    {
      const G = this._grid, i0 = Math.floor((v.x - 14) / CELL), i1 = Math.floor((v.x + 14) / CELL), k0 = Math.floor((v.z - 14) / CELL), k1 = Math.floor((v.z + 14) / CELL);
      for (let i = i0; i <= i1; i++) for (let k = k0; k <= k1; k++) {
        const a = G.get(ckey(i, k)); if (!a) continue;
        for (const u of a) {
          if (u === v || u.fx * v.fx + u.fz * v.fz < 0.3 || Math.abs(u.y - v.y) > 2.5) continue;
          const dx = u.x - v.x, dz = u.z - v.z, along = dx * v.fx + dz * v.fz;
          if (along < 0.5 || along > 13 || Math.abs(-dx * v.fz + dz * v.fx) > (v.W + u.W) * 0.5 + 0.3) continue;
          obst(along - (u.L + v.L) * 0.5, u.v, 1.5, 'guard');
        }
      }
    }
    // ---- gates (lane end -> junction) along the planned path + curve speeds
    let off = -v.s, elem = L, idx = -1, gates = 0;
    while (elem) {
      if (idx >= 0) {
        const vc = elem.v0 * Math.min(1, v.drv);
        if (vc < v0) v0 = Math.min(v0, Math.sqrt(vc * vc + 2 * 1.8 * Math.max(0, off - halfL)));
      }
      if (off - halfL > 90 || idx >= PATH_N - 1) break;
      const nextId = v.path[idx + 1];
      const endD = off + elem.len - halfL;
      if (!elem.conn) {
        if (nextId < 0) { obst(endD - 0.4, 0, 0.6, 'deadend'); if (idx < 0 && vel < 0.3 && endD < 3) v.deadT += h; break; }
        if (endD < 90 && gates < 2) {
          gates++;
          if (elem.ctrl === CTRL_YIELD) v0 = Math.min(v0, Math.sqrt(36 + 4 * Math.max(0, endD)));
          const gb = this._gate(v, elem, lanes[nextId], endD, h, idx + 3 < PATH_N ? v.path[idx + 3] : -1);
          if (gb) { obst(endD - 0.3, 0, 1.0, gb === 1 ? 'control' : 'junction'); break; }
        }
      }
      if (nextId < 0) break;
      off += elem.len; elem = lanes[nextId]; idx++;
    }
    // ---- the player
    const pl = this._pl;
    if (pl) {
      const dx = pl.x - v.x, dz = pl.z - v.z;
      if (dx * dx + dz * dz < 45 * 45 && Math.abs(pl.y - v.y) < 3.5) {
        const along = dx * v.fx + dz * v.fz, lat = Math.abs(-dx * v.fz + dz * v.fx);
        // player's half extent along our forward / lateral axes
        const ef = pl.hl * Math.abs(pl.fx * v.fx + pl.fz * v.fz) + pl.hw * Math.abs(-pl.fz * v.fx + pl.fx * v.fz);
        const el = pl.hl * Math.abs(-pl.fx * v.fz + pl.fz * v.fx) + pl.hw * Math.abs(pl.fz * v.fz + pl.fx * v.fx);
        if (along > 0 && along < 25 + halfL + ef && lat < v.W * 0.5 + el + 0.35) obst(along - halfL - ef, 0, 5.0, 'player');
        // contact (oriented boxes, SAT): stop for a few seconds
        if (Math.abs(along) < halfL + ef + 0.15 && lat < v.W * 0.5 + el + 0.15) {
          const pa = Math.abs(dx * pl.fx + dz * pl.fz), pb = Math.abs(-dx * pl.fz + dz * pl.fx);
          const va = halfL * Math.abs(v.fx * pl.fx + v.fz * pl.fz) + v.W * 0.5 * Math.abs(-v.fz * pl.fx + v.fx * pl.fz);
          const vb = halfL * Math.abs(-v.fx * pl.fz + v.fz * pl.fx) + v.W * 0.5 * Math.abs(v.fz * pl.fz + v.fx * pl.fx);
          if (pa < pl.hl + va + 0.15 && pb < pl.hw + vb + 0.15) v.stun = 3;
        }
      }
    }
    v.why = why;
    v0 = Math.max(0.5, v0);
    const free = Math.max(-2, 1 - (vel / v0) ** 4);
    let a = amax * free + worst;
    if (v.stun > 0) a = -8;
    return Math.max(-9, Math.min(amax, a));
  }

  _recycle() {
    const c = this._cam, R2 = RECYCLE_R * RECYCLE_R;
    let budget = 8;
    for (const v of this.vehicles) {
      if (!v.active || budget <= 0) continue;
      const far = (v.x - c.x) ** 2 + (v.z - c.z) ** 2 > R2;
      // geometric deadlock: blocked by a vehicle that is (transitively) blocked by us - nobody can move without overlap
      let cyc = false;
      if (v.hardT > 4) { let u = v.hardBy; for (let k = 0; k < 6 && u; k++) { if (u === v) { cyc = true; break; } u = u.hardT > 2 ? u.hardBy : null; } }
      const stuck = v.deadT > 1.5 || v.idleT > 90 || cyc;
      if (!far && !stuck) continue;
      if (!far && this._visible(v.x, v.y, v.z) && v.deadT < 25 && v.idleT < 240 && !(cyc && v.hardT > 15)) continue;
      const p = this._pickSpawn(c.x, c.z, 110, 620, true);
      if (!p) { budget--; continue; }
      this.lanes[v.lane].veh = this.lanes[v.lane].veh.filter((u) => u !== v);
      this._place(v, p);
      budget--;
    }
  }

  _ahead(v, d, out) {
    let L = this.lanes[v.lane], s = v.s + d, k = 0;
    while (s > L.len && k < PATH_N && v.path[k] >= 0) { s -= L.len; L = this.lanes[v.path[k++]]; }
    return sampleLane(L, s, out);
  }

  _behind(v, d, out) {
    const s = v.s - d;
    if (s >= 0 || v.prev < 0) return sampleLane(this.lanes[v.lane], s, out);
    const Pl = this.lanes[v.prev];
    return sampleLane(Pl, Pl.len + s, out);
  }

  /** Footprint pose (axle model, as _pose) of v if it were at arc position s on its current lane / path. */
  _poseAt(v, s, out) {
    const lanes = this.lanes, hb2 = v.WB * 0.5;
    let L = lanes[v.lane], sf = s + hb2, k = 0;
    while (sf > L.len && k < PATH_N && v.path[k] >= 0) { sf -= L.len; L = lanes[v.path[k++]]; }
    sampleLane(L, sf, P3);
    const sr = s - hb2;
    if (sr >= 0 || v.prev < 0) sampleLane(lanes[v.lane], sr, P2); else { const Pl = lanes[v.prev]; sampleLane(Pl, Pl.len + sr, P2); }
    let dx = P3.x - P2.x, dz = P3.z - P2.z; const d = Math.hypot(dx, dz);
    if (d < 0.2) { dx = v.fx; dz = v.fz; } else { dx /= d; dz /= d; }
    out.x = (P2.x + P3.x) * 0.5; out.z = (P2.z + P3.z) * 0.5; out.fx = dx; out.fz = dz;
    return out;
  }

  /**
   * Hard safety, independent of all junction logic: refuse a move that would bring v within 0.5 m of the player's
   * oriented box or >0.1 m into another vehicle - if the move creates the overlap, or approaches something ahead.
   */
  _hardBlocked(v, ds) {
    const hl = v.L * 0.5, hw = v.W * 0.5;
    let n = null, c = null; // predicted / current footprints (computed lazily)
    const pl = this._pl;
    if (pl && Math.abs(pl.y - v.y) < 3 && (pl.x - v.x) ** 2 + (pl.z - v.z) ** 2 < 15 * 15) {
      n = this._poseAt(v, v.s + ds, Q1);
      if (obb2(n.x, n.z, n.fx, n.fz, hl + 0.25, hw + 0.25, pl.x, pl.z, pl.fx, pl.fz, pl.hl + 0.25, pl.hw + 0.25)) {
        c = this._poseAt(v, v.s, Q0);
        const ahead = (pl.x - c.x) * c.fx + (pl.z - c.z) * c.fz > 0;
        if (ahead || !obb2(c.x, c.z, c.fx, c.fz, hl + 0.25, hw + 0.25, pl.x, pl.z, pl.fx, pl.fz, pl.hl + 0.25, pl.hw + 0.25)) { v.why = 'player!'; v.hardBy = null; return true; }
      }
    }
    const G = this._grid, i0 = Math.floor((v.x - 16) / CELL), i1 = Math.floor((v.x + 16) / CELL), k0 = Math.floor((v.z - 16) / CELL), k1 = Math.floor((v.z + 16) / CELL);
    for (let i = i0; i <= i1; i++) for (let k = k0; k <= k1; k++) {
      const a = G.get(ckey(i, k)); if (!a) continue;
      for (const u of a) {
        if (u === v || !u.active || Math.abs(u.y - v.y) > 2.5) continue;
        const ux = u.x + u.fx * u.dsAcc, uz = u.z + u.fz * u.dsAcc;
        const lim = hl + u.L * 0.5 + 1.5;
        if ((ux - v.x) ** 2 + (uz - v.z) ** 2 > lim * lim) continue;
        if (!n) n = this._poseAt(v, v.s + ds, Q1);
        const uhl = u.L * 0.5 - 0.05, uhw = u.W * 0.5 - 0.05;
        if (!obb2(n.x, n.z, n.fx, n.fz, hl - 0.05, hw - 0.05, ux, uz, u.fx, u.fz, uhl, uhw)) continue;
        if (!c) c = this._poseAt(v, v.s, Q0);
        const ahead = (ux - c.x) * c.fx + (uz - c.z) * c.fz > 0;
        if (ahead || !obb2(c.x, c.z, c.fx, c.fz, hl - 0.05, hw - 0.05, ux, uz, u.fx, u.fz, uhl, uhw)) { v.why = 'vehicle!'; v.hardBy = u; return true; }
      }
    }
    return false;
  }

  // ================================================================ colliders
  _rebuildGrid() { rebuildVehicleGrid(this._grid, this._gridKeys, this.vehicles); }

  /** Append vehicle AABBs {minX,minY,maxX,maxY,minZ,maxZ,kind:'vehicle'} within `radius` of (x,z). Shared objects: read only. */
  queryColliders(x, z, radius, out = []) {
    if (!this.vehicles.length) return out;
    return queryVehicleGrid(this._grid, x, z, radius, out);
  }

  /** Debug summary (+ junction entries at signals / on red, vehicles held by the player rules). */
  summary() {
    let moving = 0, stopped = 0, onConn = 0, sum = 0, n = 0, forPlayer = 0;
    for (const v of this.vehicles) {
      if (!v.active) continue;
      n++; sum += v.v;
      if (v.v > 0.5) moving++; else stopped++;
      if (this.lanes[v.lane].conn) onConn++;
      if (v.why === 'player' || v.why === 'player!') forPlayer++;
    }
    return { count: n, moving, stopped, onConn, avgSpeed: n ? sum / n : 0, sigEntries: this.sigEntries, redEntries: this.redEntries, forPlayer, time: this.time };
  }
}

/** Height of a bridge edge's own deck profile (e.hs along e.pts) at the point of the centreline nearest (x, z). */
function edgeRefHeight(e, x, z) {
  const P = e.pts, H = e.hs;
  let best = Infinity, h = H[0];
  for (let i = 0; i < P.length - 1; i++) {
    const ax = P[i][0], az = P[i][1], dx = P[i + 1][0] - ax, dz = P[i + 1][1] - az, L2 = dx * dx + dz * dz;
    let t = L2 > 1e-9 ? ((x - ax) * dx + (z - az) * dz) / L2 : 0;
    t = t < 0 ? 0 : t > 1 ? 1 : t;
    const qx = ax + dx * t - x, qz = az + dz * t - z, d2 = qx * qx + qz * qz;
    if (d2 < best) { best = d2; h = H[i] + (H[i + 1] - H[i]) * t; }
  }
  return h;
}

/**
 * Collider slices of a vehicle (v.box / v.boxes / v.nb) at centre (x, z), forward (fx, fz), y0..y1, speed: a rotated
 * vehicle is reported as thin AABB slices along its length so the union hugs the oriented footprint (one AABB of a
 * 45-degree car would stick out ~1.4 m sideways into the next lane). Shared by the sim and the presenter (snapshots).
 */
export function vehicleSlices(v, x, z, fx, fz, y0, y1, speed) {
  const b = v.box, hl = v.L * 0.5, hw = v.W * 0.5, ax = Math.abs(fx), az = Math.abs(fz);
  const ex = ax * hl + az * hw, ez = az * hl + ax * hw;
  b.minX = x - ex; b.maxX = x + ex; b.minZ = z - ez; b.maxZ = z + ez; b.minY = y0; b.maxY = y1;
  const n = Math.min(ax, az) < 0.06 ? 1 : Math.min(MAX_SLICES, Math.ceil(v.L / (v.W * 0.5)));
  const sl = v.L / n, sx = ax * sl * 0.5 + az * hw, sz = az * sl * 0.5 + ax * hw;
  for (let k = 0; k < n; k++) {
    const o = -hl + sl * (k + 0.5), cx = x + fx * o, cz = z + fz * o, q = v.boxes[k];
    q.minX = cx - sx; q.maxX = cx + sx; q.minZ = cz - sz; q.maxZ = cz + sz; q.minY = y0; q.maxY = y1;
    q.vx = fx * speed; q.vz = fz * speed;
  }
  v.nb = n;
}

/** Append the collider slices of the vehicles in grid G (cell CELL) within `radius` of (x, z) to out (shared objects). */
export function queryVehicleGrid(G, x, z, radius, out) {
  if (!Number.isFinite(x) || !Number.isFinite(z)) return out;
  const r = Math.max(0, radius || 0), rr = r + 6.5;
  const i0 = Math.floor((x - rr) / CELL), i1 = Math.floor((x + rr) / CELL), k0 = Math.floor((z - rr) / CELL), k1 = Math.floor((z + rr) / CELL);
  if ((i1 - i0 + 1) * (k1 - k0 + 1) > 1024) return out;
  const r2 = r * r;
  for (let i = i0; i <= i1; i++) for (let k = k0; k <= k1; k++) {
    const a = G.get(ckey(i, k));
    if (!a) continue;
    for (let j = 0; j < a.length; j++) {
      const u = a[j];
      let b = u.box;
      let dx = x < b.minX ? b.minX - x : x > b.maxX ? x - b.maxX : 0;
      let dz = z < b.minZ ? b.minZ - z : z > b.maxZ ? z - b.maxZ : 0;
      if (dx * dx + dz * dz > r2) continue;
      for (let q = 0; q < u.nb; q++) {
        b = u.boxes[q];
        dx = x < b.minX ? b.minX - x : x > b.maxX ? x - b.maxX : 0;
        dz = z < b.minZ ? b.minZ - z : z > b.maxZ ? z - b.maxZ : 0;
        if (dx * dx + dz * dz <= r2) out.push(b);
      }
    }
  }
  return out;
}

/** Rebuild a vehicle grid (Map ckey -> [vehicle]) from objects with x, z, active; keys = reusable key list. */
export function rebuildVehicleGrid(G, keys, vehicles) {
  if (G.size > 6000) { G.clear(); keys.length = 0; }
  for (const k of keys) { const a = G.get(k); if (a) a.length = 0; }
  keys.length = 0;
  for (const v of vehicles) {
    if (!v.active) continue;
    const k = ckey(Math.floor(v.x / CELL), Math.floor(v.z / CELL));
    let a = G.get(k);
    if (!a) { a = []; G.set(k, a); }
    if (!a.length) keys.push(k);
    a.push(v);
  }
}
