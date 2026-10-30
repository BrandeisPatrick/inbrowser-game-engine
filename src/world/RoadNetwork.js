// Road network: carriageway ribbons with procedural lane markings / crosswalks / stop lines (shader),
// sidewalks with kerbs, minor paths, elevated decks (profile solved over the bridge graph) with
// barriers and piers, and rail tracks. With a GroundSurface (Terrain.js) everything follows the real
// terrain: ground roads get a smoothed centre-line profile (junction nodes pinned so crossing roads
// agree) whose corridors flatten the ground surface; ribbons, sidewalks, paths and rails are sampled
// from that surface; elevated decks start at the real ground at their grounded ends. Long suspended bridges can hold
// their decks level over the water (`deckHolds` option, e.g. a bay crossing's upper / lower decks).
import * as THREE from 'three';
import { SegmentHash, MinHeap } from '../physics/spatial.js';
import { clamp, signedArea } from './geom.js';

export const RBIT = { ROAD: 1, BUILDING: 2, WATER: 4, PARK: 8, SERVICE: 16, SIDEWALK: 32, PROP: 64 };

const LANE_W = { motorway: 3.7, trunk: 3.6, primary: 3.5, secondary: 3.4, tertiary: 3.3, unclassified: 3.2, residential: 3.2, living_street: 3.0, service: 3.0, busway: 3.5, road: 3.2 };
const MINOR = new Set(['pedestrian', 'footway', 'path', 'cycleway', 'track', 'steps', 'bridleway']);
const MAJOR = new Set(['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'unclassified', 'residential', 'living_street', 'road']);

export function roadStyle(r) {
  const cls = r.c || 'residential';
  if (MINOR.has(cls)) {
    const w = cls === 'pedestrian' ? clamp(r.w || 7, 3, 20) : cls === 'steps' ? 2.5 : cls === 'track' ? 3 : clamp(r.w || 2.4, 1.2, 6);
    return { cls, base: cls, minor: true, hw: w / 2, lanes: 0, oneway: false, parkW: 0, markings: 0, sidewalk: 0, major: false, drivable: cls === 'pedestrian' || cls === 'track' };
  }
  const link = cls.endsWith('_link');
  const base = link ? cls.slice(0, -5) : cls;
  const oneway = !!r.o;
  let lanes = r.l || 0;
  if (!lanes) {
    const def = { motorway: oneway ? 3 : 4, trunk: oneway ? 2 : 4, primary: oneway ? 2 : 4, secondary: 2, tertiary: 2, busway: 1, service: 1, living_street: 1 };
    lanes = link ? 1 : def[base] ?? (oneway ? 1 : 2);
  }
  let laneW = LANE_W[base] ?? 3.2;
  if (base === 'service') { lanes = Math.min(lanes, 2); laneW = r.sv === 'driveway' || r.sv === 'alley' ? 2.9 : 3.0; }
  const parkW = !r.b && !link && /^(secondary|tertiary|residential|unclassified|living_street)$/.test(base) ? 2.4 : 0;
  let w = lanes * laneW + 2 * parkW;
  if (r.w > lanes * 2.6 && r.w < 60) w = r.w;
  const markings = base === 'motorway' || base === 'trunk' ? 2 : base === 'service' || base === 'living_street' ? 0 : 1;
  const sidewalk = !r.b && !link && /^(primary|secondary|tertiary|residential|unclassified|living_street|trunk)$/.test(base)
    ? (/^(primary|secondary|trunk)$/.test(base) ? 3.4 : 2.7) : 0;
  return { cls, base, link, minor: false, oneway, lanes, laneW, parkW: Math.min(parkW, w / 4), hw: w / 2, markings, sidewalk, major: MAJOR.has(base), drivable: true };
}

// ------------------------------------------------------------------ materials
/** asphalt: procedural canvas texture (8 m tile); pbr (optional): {map, normalMap, roughnessMap, size} photo set
 *  tiled at its real-world size, with the procedural texture kept as a large-scale patch / crack layer. */
export function createRoadMaterial(asphalt, pbr = null) {
  const m = new THREE.MeshStandardMaterial({ map: pbr ? pbr.map : asphalt, roughness: 0.92, metalness: 0, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -4 });
  const macro = { uMacro: { value: asphalt }, uMacroK: { value: 1 } };
  if (pbr) {
    const r = 8 / (pbr.size || 3); // ribbon uv = metres / 8
    for (const t of [pbr.map, pbr.normalMap, pbr.roughnessMap]) if (t) t.repeat.set(r, r);
    if (pbr.normalMap) { m.normalMap = pbr.normalMap; m.normalScale.set(0.7, 0.7); }
    if (pbr.roughnessMap) { m.roughnessMap = pbr.roughnessMap; m.roughness = 1; }
    macro.uMacroK.value = 1 / r;
  }
  m.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, macro);
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', `#include <common>
attribute vec4 aRoad; attribute vec4 aRoad2; attribute vec4 aInt;
varying vec4 vRoad; varying vec4 vRoad2; varying vec4 vInt;`)
      .replace('#include <begin_vertex>', `#include <begin_vertex>
vRoad = aRoad; vRoad2 = aRoad2; vInt = aInt;`);
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', `#include <common>
varying vec4 vRoad; varying vec4 vRoad2; varying vec4 vInt;
float rLine(float d, float w, float aa) { return 1.0 - smoothstep(w - aa, w + aa, abs(d)); }
float rBand(float x, float a, float b, float aa) { return smoothstep(a - aa, a + aa, x) * (1.0 - smoothstep(b - aa, b + aa, x)); }
float rHash(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
uniform sampler2D uMacro; uniform float uMacroK;`)
      .replace('#include <map_fragment>', `#include <map_fragment>
${pbr ? `{ // procedural patches / cracks / tone over the photo asphalt (8 m tile)
  vec3 mc = texture2D(uMacro, vMapUv * uMacroK * 0.31 + 0.17).rgb;
  vec3 mc2 = texture2D(uMacro, vMapUv * uMacroK * 0.083 + 0.61).rgb;
  diffuseColor.rgb *= clamp(mc.g / 0.048, 0.6, 1.35) * clamp(mc2.g / 0.048, 0.8, 1.2) * 0.92;
}` : ''}
float rPaint = 0.0; vec3 rPaintCol = vec3(0.86); float rTrack = 0.0; float rPatchK = 0.0;
{
  float ac = vRoad.x, s = vRoad.y, hw = vRoad.z, pw = vRoad.w;
  float lanes = vRoad2.x, ow = vRoad2.y, mode = vRoad2.z;
  float ch = hw - pw;
  float aaA = fwidth(ac) * 0.6 + 0.002;
  float aaS = fwidth(s) * 0.6 + 0.002;
  float dP = s - vInt.x, dN = vInt.z - s;
  float dI = dN; float rI = vInt.w; float sideN = 1.0;
  if (dP < dN) { dI = dP; rI = vInt.y; sideN = -1.0; }
  float rA = abs(rI);              // junction half-width (negative: geometric junction, no stop line)
  float hasI = step(0.01, rA);
  float ad = abs(ac);
  // the junction box is shared by several ribbons: keep it plain (identical world-space asphalt)
  float box = mix(1.0, smoothstep(rA - 0.4, rA + 1.0, dI), hasI);
  // lane geometry: lane width + coordinate across the lanes of one direction
  float nL = ow < 0.5 ? max(1.0, floor(lanes * 0.5 + 0.01)) : max(1.0, lanes);
  float lw = ow < 0.5 ? ch / nL : 2.0 * ch / nL;
  float lx = ow < 0.5 ? ad : ac + ch;
  float paintWear = 0.0;
  if (hw > 0.5) {
    // ---- wear: darker polished tyre tracks in every lane, oil drip strip near stop lines
    float f = (fract(lx / lw) - 0.5) * lw;                // metres from the lane centre
    float inLanes = step(ad, ch - 0.05);
    float wp = 1.0 - smoothstep(0.16, 0.46, abs(abs(f) - 0.8));
    float mk = texture2D(uMacro, vec2(s * 0.011, ac * 0.04 + hw * 0.1) + 0.37).g / 0.048;
    float stopZone = hasI * (1.0 - smoothstep(6.0, 30.0, dI)) * smoothstep(rA + 4.0, rA + 7.0, dI);
    float drip = (1.0 - smoothstep(0.1, 0.38, abs(f))) * stopZone;
    rTrack = clamp(wp * clamp(0.45 + 0.4 * mk, 0.3, 1.0) + drip * 0.8, 0.0, 1.0) * inLanes * box;
    paintWear = wp * inLanes;
    // ---- utility-cut patches: lane-sized rectangles of newer (dark) or older (grey) asphalt + sealed seams
    vec2 cell = vec2(floor(s / 11.0), floor((ac + hw) / 3.3));
    float h0 = rHash(cell + vec2(hw * 1.7, lanes * 0.37));
    if (h0 < 0.14) {
      float u = s - cell.x * 11.0, v = ac + hw - cell.y * 3.3;
      float l0 = 0.5 + rHash(cell + 1.7) * 3.0, l1 = l0 + 1.8 + rHash(cell + 3.1) * 6.0;
      float w0 = 0.2 + rHash(cell + 5.3) * 0.6, w1 = 3.3 - 0.2 - rHash(cell + 7.9) * 0.8;
      float inP = rBand(u, l0, l1, aaS) * rBand(v, w0, w1, aaA);
      float seam = inP * (1.0 - rBand(u, l0 + 0.06, l1 - 0.06, aaS) * rBand(v, w0 + 0.06, w1 - 0.06, aaA));
      rPatchK = (h0 < 0.085 ? -1.0 : 1.0) * inP * box * step(ad, hw - 0.2);
      diffuseColor.rgb *= 1.0 - 0.4 * seam * box;
    }
    // long trench patch along a lane (water / gas main), now and then
    float cs = floor(s / 60.0), ht = rHash(vec2(cs, floor(lx / lw) + hw * 0.13));
    if (ht < 0.07) {
      float u = s - cs * 60.0, t0 = 4.0 + rHash(vec2(cs, 2.2)) * 20.0, t1 = t0 + 14.0 + rHash(vec2(cs, 3.3)) * 22.0;
      float c0 = (floor(lx / lw) + 0.25 + rHash(vec2(cs, 4.4)) * 0.4) * lw;
      float inT = rBand(u, t0, t1, aaS) * rBand(lx, c0, c0 + 0.9, aaA);
      rPatchK = min(rPatchK, -inT * box);
    }
    diffuseColor.rgb *= 1.0 - 0.17 * rTrack;
    if (rPatchK < 0.0) diffuseColor.rgb *= 1.0 + 0.17 * rPatchK;                       // fresh, darker asphalt
    else if (rPatchK > 0.0) diffuseColor.rgb = mix(diffuseColor.rgb, vec3(dot(diffuseColor.rgb, vec3(0.33)) * 1.25), rPatchK * 0.8); // old grey patch
  }
  float white = 0.0, yellow = 0.0;
  if (mode > 0.5) {
    float cw0 = rA + 1.0, cw1 = rA + 4.0, st0 = cw1 + 1.2, st1 = st0 + 0.45;
    // lane lines end at the stop line (crosswalk zone on motorways: just short of the junction)
    float keep = hasI > 0.5 ? (mode < 1.5 ? smoothstep(st1 + 0.15, st1 + 0.5, dI) : smoothstep(rA + 0.6, rA + 1.2, dI)) : 1.0;
    float dash = rBand(fract(s / 12.0), 0.0, 0.25, aaS / 12.0);     // 3 m dashes, 9 m gaps (US urban)
    float kk = floor(lx / lw + 0.5);                                // nearest lane divider
    float divider = (kk >= 1.0 && kk < nL - 0.5) ? rLine(lx - kk * lw, 0.05, aaA) * dash : 0.0;
    if (ow < 0.5) {
      // two-way: double yellow centre line (two 10 cm lines, 10 cm apart)
      if (lanes >= 2.0) yellow = rLine(ad - 0.1, 0.05, aaA);
      white = divider;
    } else {
      white = divider;
      if (mode > 1.5) yellow = rLine(ac + ch - 0.3, 0.075, aaA);    // freeway left edge
    }
    if (pw > 0.1) white = max(white, rLine(ad - ch, 0.05, aaA));     // parking-lane lines
    else if (mode > 1.5) white = max(white, rLine(ow < 0.5 ? ad - (ch - 0.3) : ac - (ch - 0.3), 0.075, aaA));
    white *= keep; yellow *= keep;
    if (mode < 1.5 && hasI > 0.5) {
      // continental crosswalk: 0.6 m bars on 1.2 m centres, symmetric about the road axis, crisp ends
      float bars = rBand(fract(ac / 1.2 + 0.25), 0.0, 0.5, aaA / 1.2);
      float cw = rBand(dI, cw0, cw1, aaS) * bars * step(ad, hw - 0.3);
      white = max(white, cw);
      // stop line 1.2 m before the crosswalk, approach lanes only (one-way: only before the next junction)
      if (rI > 0.0) {
        float side = ow > 0.5 ? step(0.0, sideN) : step(0.0, ac * sideN) * smoothstep(0.2, 0.24, ad);
        white = max(white, rBand(dI, st0, st1, aaS) * side * step(ad, ch - 0.15));
      }
      paintWear *= 1.0 - rBand(dI, cw0 - 0.5, st1 + 0.5, aaS) * 0.6;   // crosswalks / stop lines kept crisp
    }
  }
  float wear = (0.72 + 0.4 * clamp(diffuseColor.g * 3.0, 0.0, 1.0)) * (1.0 - 0.35 * paintWear);
  rPaint = clamp(max(white, yellow) * wear, 0.0, 1.0);
  rPaintCol = yellow > white ? vec3(0.86, 0.6, 0.08) : vec3(0.9, 0.9, 0.88);
  diffuseColor.rgb = mix(diffuseColor.rgb, rPaintCol, rPaint);
}`)
      .replace('#include <roughnessmap_fragment>', pbr ? `#include <roughnessmap_fragment>
roughnessFactor *= (1.0 - 0.14 * rTrack) * (rPatchK < 0.0 ? 0.94 : 1.0);
roughnessFactor = mix(roughnessFactor, 0.6, rPaint);` : `float roughnessFactor = mix(roughness * (1.0 - 0.14 * rTrack), 0.6, rPaint);`);
  };
  m.customProgramCacheKey = () => (pbr ? 'roadster-road-pbr-2' : 'roadster-road-2');
  return m;
}

/** Yellow truncated-dome tactile warning pads (uv: 0.5 m tiles of 8 x 8 domes). */
let _tactileMat = null;
export function createTactileMaterial() {
  if (_tactileMat) return _tactileMat;
  let map = null;
  try {
    const S = 128, c = document.createElement('canvas'); c.width = c.height = S;
    const g = c.getContext('2d');
    g.fillStyle = '#c99a1c'; g.fillRect(0, 0, S, S);
    const cell = S / 8;
    for (let i = 0; i < 8; i++) for (let k = 0; k < 8; k++) {
      const x = (i + 0.5) * cell, y = (k + 0.5) * cell, r = cell * 0.32;
      g.fillStyle = 'rgba(70,50,8,0.55)'; g.beginPath(); g.arc(x + 1.2, y + 1.6, r, 0, 7); g.fill();
      const gr = g.createRadialGradient(x - r * 0.3, y - r * 0.35, r * 0.1, x, y, r);
      gr.addColorStop(0, '#f6d35a'); gr.addColorStop(1, '#d8a628');
      g.fillStyle = gr; g.beginPath(); g.arc(x, y, r, 0, 7); g.fill();
    }
    map = new THREE.CanvasTexture(c);
    map.colorSpace = THREE.SRGBColorSpace; map.wrapS = map.wrapT = THREE.RepeatWrapping; map.anisotropy = 4;
  } catch { map = null; }
  _tactileMat = new THREE.MeshStandardMaterial({ map, color: map ? 0xffffff : 0xd8a628, roughness: 0.62, metalness: 0, polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -2 });
  return _tactileMat;
}

// ------------------------------------------------------------------ buffers
class RoadBuf {
  constructor() { this.pos = []; this.uv = []; this.r = []; this.r2 = []; this.int = []; this.nor = []; this.idx = []; this.n = 0; }
  v(x, y, z, ac, s, hw, pw, lanes, ow, mode, ip, rp, inx, rn, n) {
    this.pos.push(x, y, z); if (n) this.nor.push(n[0], n[1], n[2]); else this.nor.push(0, 1, 0); this.uv.push(x / 8, z / 8);
    this.r.push(ac, s, hw, pw); this.r2.push(lanes, ow, mode, 0); this.int.push(ip, rp, inx, rn);
    return this.n++;
  }
  build(mat) {
    if (!this.n) return null;
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(this.pos, 3));
    g.setAttribute('normal', new THREE.Float32BufferAttribute(this.nor, 3));
    g.setAttribute('uv', new THREE.Float32BufferAttribute(this.uv, 2));
    g.setAttribute('aRoad', new THREE.Float32BufferAttribute(this.r, 4));
    g.setAttribute('aRoad2', new THREE.Float32BufferAttribute(this.r2, 4));
    g.setAttribute('aInt', new THREE.Float32BufferAttribute(this.int, 4));
    g.setIndex(this.n > 65535 ? new THREE.Uint32BufferAttribute(this.idx, 1) : new THREE.Uint16BufferAttribute(this.idx, 1));
    g.computeBoundingSphere();
    const m = new THREE.Mesh(g, mat);
    m.receiveShadow = true; m.matrixAutoUpdate = false; m.updateMatrix();
    return m;
  }
}

/** Generic buffer: position, normal, uv, color. */
export class SolidBuf {
  constructor(uvScale = 1) { this.pos = []; this.nor = []; this.uv = []; this.col = []; this.idx = []; this.n = 0; this.uvScale = uvScale; }
  v(x, y, z, nx, ny, nz, c, u, w) {
    this.pos.push(x, y, z); this.nor.push(nx, ny, nz);
    if (u === undefined) {
      if (Math.abs(ny) >= 0.5) { u = x / this.uvScale; w = z / this.uvScale; }
      else { u = (x * Math.abs(nz) + z * Math.abs(nx)) / this.uvScale; w = y / this.uvScale; }
    }
    this.uv.push(u, w); this.col.push(c[0], c[1], c[2]);
    return this.n++;
  }
  /** Triangle with automatic winding so the face normal agrees with (nx,ny,nz) of its first vertex. */
  tri(a, b, c) {
    const P = this.pos, N = this.nor;
    const ax = P[a * 3], ay = P[a * 3 + 1], az = P[a * 3 + 2];
    const ux = P[b * 3] - ax, uy = P[b * 3 + 1] - ay, uz = P[b * 3 + 2] - az;
    const vx = P[c * 3] - ax, vy = P[c * 3 + 1] - ay, vz = P[c * 3 + 2] - az;
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    if (nx * N[a * 3] + ny * N[a * 3 + 1] + nz * N[a * 3 + 2] >= 0) this.idx.push(a, b, c); else this.idx.push(a, c, b);
  }
  quad(a, b, c, d) { this.tri(a, b, c); this.tri(a, c, d); }
  /** Vertical or arbitrary quad from 4 points with explicit normal. */
  quadP(p0, p1, p2, p3, n, c) {
    const a = this.v(p0[0], p0[1], p0[2], n[0], n[1], n[2], c), b = this.v(p1[0], p1[1], p1[2], n[0], n[1], n[2], c);
    const cc = this.v(p2[0], p2[1], p2[2], n[0], n[1], n[2], c), d = this.v(p3[0], p3[1], p3[2], n[0], n[1], n[2], c);
    this.quad(a, b, cc, d);
  }
  box(cx, y0, cz, sx, sy, sz, ang, c) {
    const ca = Math.cos(ang), sa = Math.sin(ang);
    const P = [[-sx / 2, -sz / 2], [sx / 2, -sz / 2], [sx / 2, sz / 2], [-sx / 2, sz / 2]].map(([x, z]) => [cx + x * ca - z * sa, cz + x * sa + z * ca]);
    const y1 = y0 + sy;
    for (let i = 0; i < 4; i++) {
      const a = P[i], b = P[(i + 1) % 4];
      const mx = (a[0] + b[0]) / 2 - cx, mz = (a[1] + b[1]) / 2 - cz, l = Math.hypot(mx, mz) || 1;
      this.quadP([a[0], y0, a[1]], [b[0], y0, b[1]], [b[0], y1, b[1]], [a[0], y1, a[1]], [mx / l, 0, mz / l], c);
    }
    this.quadP([P[0][0], y1, P[0][1]], [P[1][0], y1, P[1][1]], [P[2][0], y1, P[2][1]], [P[3][0], y1, P[3][1]], [0, 1, 0], c);
    this.quadP([P[0][0], y0, P[0][1]], [P[1][0], y0, P[1][1]], [P[2][0], y0, P[2][1]], [P[3][0], y0, P[3][1]], [0, -1, 0], c);
    return P;
  }
  build(mat, { cast = false, receive = true } = {}) {
    if (!this.n) return null;
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(this.pos, 3));
    g.setAttribute('normal', new THREE.Float32BufferAttribute(this.nor, 3));
    g.setAttribute('uv', new THREE.Float32BufferAttribute(this.uv, 2));
    g.setAttribute('color', new THREE.Float32BufferAttribute(this.col, 3));
    g.setIndex(this.n > 65535 ? new THREE.Uint32BufferAttribute(this.idx, 1) : new THREE.Uint16BufferAttribute(this.idx, 1));
    g.computeBoundingSphere();
    const m = new THREE.Mesh(g, mat);
    m.castShadow = cast; m.receiveShadow = receive; m.matrixAutoUpdate = false; m.updateMatrix();
    return m;
  }
}

// ------------------------------------------------------------------ polyline helpers
function prep(pts) {
  const n = pts.length;
  const d = [], s = [0];
  for (let i = 0; i < n - 1; i++) {
    const dx = pts[i + 1][0] - pts[i][0], dz = pts[i + 1][1] - pts[i][1];
    const L = Math.hypot(dx, dz) || 1e-6;
    d.push([dx / L, dz / L, L]);
    s.push(s[i] + L);
  }
  // miter (right-pointing) per vertex
  const m = [];
  for (let i = 0; i < n; i++) {
    const a = d[Math.max(0, i - 1)], b = d[Math.min(n - 2, i)];
    const na = [-a[1], a[0]], nb = [-b[1], b[0]];
    let mx = na[0] + nb[0], mz = na[1] + nb[1];
    const l = Math.hypot(mx, mz);
    if (l < 1e-3) { mx = nb[0]; mz = nb[1]; } else { mx /= l; mz /= l; }
    const k = 1 / Math.max(0.5, mx * nb[0] + mz * nb[1]);
    m.push([mx * k, mz * k]);
  }
  return { d, s, m, L: s[n - 1] };
}
export function samplePolyline(pts, pp, s) {
  // returns [x, z, dirx, dirz, segIndex, t]
  const S = pp.s;
  let i = 0;
  if (s <= 0) i = 0; else if (s >= pp.L) i = pts.length - 2;
  else { let lo = 0, hi = S.length - 1; while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (S[mid] <= s) lo = mid; else hi = mid; } i = lo; }
  i = clamp(i, 0, pts.length - 2);
  const t = clamp((s - S[i]) / (pp.d[i][2] || 1), 0, 1);
  return [pts[i][0] + (pts[i + 1][0] - pts[i][0]) * t, pts[i][1] + (pts[i + 1][1] - pts[i][1]) * t, pp.d[i][0], pp.d[i][1], i, t];
}
function subdivide(pts, maxLen, flags) {
  const out = [], fl = [];
  for (let i = 0; i < pts.length; i++) {
    out.push(pts[i]); fl.push(flags ? flags[i] : 0);
    if (i + 1 < pts.length) {
      const L = Math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]);
      const n = Math.floor(L / maxLen);
      for (let k = 1; k <= n; k++) {
        const t = k / (n + 1);
        out.push([pts[i][0] + (pts[i + 1][0] - pts[i][0]) * t, pts[i][1] + (pts[i + 1][1] - pts[i][1]) * t]); fl.push(0);
      }
    }
  }
  return { pts: out, flags: fl };
}
/**
 * Deck subdivision: `coarse` m spacing, `fine` m inside the ramp-foot zones (zone[0] m from the start, zone[1] m
 * from the end; see RoadNetwork.footZone), with a vertex exactly where each zone ends. Keeps the input point
 * objects for the original vertices (R.dints matches them by identity). Returns { pts, flags, s } (s: arc length).
 */
function subdivideDeck(pts, flags, pp, zone, fine = 4, coarse = 18) {
  const L = pp.L, za = Math.min(zone[0] || 0, L), zb = Math.min(zone[1] || 0, L);
  const cuts = [];
  if (za > 0 && za < L) cuts.push(za);
  if (zb > 0 && L - zb > 0 && Math.abs(L - zb - za) > 0.5) cuts.push(L - zb);
  const inZone = (s) => (za > 0 && s < za + 1e-6) || (zb > 0 && s > L - zb - 1e-6);
  const out = [], fl = [], ss = [];
  for (let i = 0; i < pts.length; i++) {
    out.push(pts[i]); fl.push(flags ? flags[i] : 0); ss.push(pp.s[i]);
    if (i + 1 >= pts.length) continue;
    const s0 = pp.s[i], s1 = pp.s[i + 1], a = pts[i], b = pts[i + 1];
    const at = (s) => { const t = (s - s0) / Math.max(1e-9, s1 - s0); out.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]); fl.push(0); ss.push(s); };
    const bps = [s0, ...cuts.filter((c) => c > s0 + 0.5 && c < s1 - 0.5).sort((p, q) => p - q), s1];
    for (let k = 0; k + 1 < bps.length; k++) {
      const p = bps[k], q = bps[k + 1], maxLen = inZone((p + q) / 2) ? fine : coarse;
      const n = Math.floor((q - p) / maxLen);
      for (let j = 1; j <= n; j++) at(p + ((q - p) * j) / (n + 1));
      if (k + 2 < bps.length) at(q);
    }
  }
  return { pts: out, flags: fl, s: ss };
}
/** Part of a polyline between arc lengths s0 and s1 (interpolated end points, the original vertices between). */
function subPolyline(pts, pp, s0, s1) {
  const out = [samplePolyline(pts, pp, s0).slice(0, 2)];
  for (let i = 1; i < pts.length - 1; i++) if (pp.s[i] > s0 + 0.05 && pp.s[i] < s1 - 0.05) out.push(pts[i]);
  out.push(samplePolyline(pts, pp, s1).slice(0, 2));
  return out;
}
/** Copy of pts with extra vertices at arc positions ss (only strictly inside segments). */
function insertVertices(pts, pp, ss) {
  const add = ss.filter((s) => s > 0.05 && s < pp.L - 0.05).sort((a, b) => a - b);
  if (!add.length) return pts;
  const out = [pts[0]];
  let k = 0;
  for (let i = 0; i < pts.length - 1; i++) {
    while (k < add.length && add[k] < pp.s[i + 1] - 0.05) {
      if (add[k] > pp.s[i] + 0.05) { const t = (add[k] - pp.s[i]) / pp.d[i][2]; out.push([pts[i][0] + (pts[i + 1][0] - pts[i][0]) * t, pts[i][1] + (pts[i + 1][1] - pts[i][1]) * t]); }
      k++;
    }
    out.push(pts[i + 1]);
  }
  return out;
}
const key = (p) => `${Math.round(p[0] * 10)},${Math.round(p[1] * 10)}`;
// ramp-foot zones (RoadNetwork.footZone): sampling step, give-up gap to the next street / service road, reach from a
// grounded end (m), and the lift of a deck pinned to a street it runs over (its ribbon draws on top of the street's
// instead of z-fighting)
const FOOT_STEP = 0.5, FOOT_GAP = 15, FOOT_GAP_SERVICE = 30, FOOT_MAX = 60, ZONE_LIFT = 0.02;

// ------------------------------------------------------------------ network
/**
 * Deck hold rule (RoadNetwork opts.deckHolds): the longest named motorway / trunk bridge matching `name` that runs
 * more than `minWet` m over water (and is longer than `minLen`) is the hold axis; its parallel decks are held at
 * `top` m (lower deck - layer <= 1 - `lowerGap` m below) from `anchorage` m inland of the first shore (s = 0 at the
 * end nearer the map origin) to the next shore reached at least `farShore` m later; decks within `maxOffset` m of
 * the axis are held. Beyond the anchorage they ramp down to the ground at the normal grade.
 */
export const DEFAULT_DECK_HOLD = { top: 58, lowerGap: 9, anchorage: 146, minWet: 800, minLen: 1200, farShore: 1500, maxOffset: 45 };

export class RoadNetwork {
  /**
   * @param {{ roads: object[], rail?: object[], raster: object, isWater: Function, surface: object, deckHolds?: object[] }} o
   *   roads, rail (city.json arrays), raster (Raster with RBIT bits), isWater(x, z), surface (GroundSurface),
   *   deckHolds: [{ name: RegExp, ...DEFAULT_DECK_HOLD }] suspended decks held level over the water (none by default)
   */
  constructor({ roads, rail, raster, isWater, surface, deckHolds = [] }) {
    this.deckHolds = (deckHolds || []).filter((h) => h && h.name).map((h) => ({ ...DEFAULT_DECK_HOLD, ...h }));
    this.heldDecks = [];
    this.input = roads || [];
    this.surface = surface || null; // GroundSurface (heightAt/normalAt/baseAt) or null = flat
    this.corridors = [];            // ground-road profiles for GroundSurface.applyCorridors
    this.railIn = rail || [];
    this.raster = raster;           // Raster with RBIT bits (city area)
    this.isWater = isWater || (() => false);
    this.roadBuf = new RoadBuf();
    this.solid = new SolidBuf(6);   // sidewalks, kerbs, paths, bridge structure, piers (concrete)
    this.railBuf = new SolidBuf(1);
    this.deckHash = new SegmentHash(16);
    this.segHash = new SegmentHash(16); // ground-level drivable carriageways
    this.list = [];                 // processed roads for props: {pts, pp, st, ints:[{s,r}], bridge, hs, filler}
    this.piers = [];                // collider boxes [minX,minY,maxX,maxY,minZ,maxZ]
    this.minimap = [];
    this.minimapClass = [];
  }

  /** Ground surface height / normal (0 / up without a surface). */
  gh(x, z) { return this.surface ? this.surface.heightAt(x, z) : 0; }
  gn(x, z) { return this.surface ? this.surface.normalAt(x, z, [0, 1, 0]) : [0, 1, 0]; }
  /** SolidBuf vertex on the ground surface + yOff, with the surface normal. */
  vg(S, x, z, yOff, c) { const n = this.gn(x, z); return S.v(x, this.gh(x, z) + yOff, z, n[0], n[1], n[2], c); }
  quadG(S, p0, p1, p2, p3, yOff, c) { S.quad(this.vg(S, p0[0], p0[1], yOff, c), this.vg(S, p1[0], p1[1], yOff, c), this.vg(S, p2[0], p2[1], yOff, c), this.vg(S, p3[0], p3[1], yOff, c)); }

  /** Parse roads, intersections, raster and ground-road height profiles (-> this.corridors). */
  prepare() {
    this._prepared = true;
    const roads = [];
    for (const r of this.input) {
      if (!r || !Array.isArray(r.p) || r.p.length < 2) continue;
      const pts = r.p.filter((p) => p && isFinite(p[0]) && isFinite(p[1]));
      if (pts.length < 2) continue;
      // drop zero-length steps
      const P = [pts[0]], F = [r.x && r.x.includes(0) ? 1 : 0];
      const xs = new Set(r.x || []);
      for (let i = 1; i < pts.length; i++) {
        const l = P[P.length - 1];
        if (Math.hypot(pts[i][0] - l[0], pts[i][1] - l[1]) < 0.05) { if (xs.has(i)) F[F.length - 1] = 1; continue; }
        P.push(pts[i]); F.push(xs.has(i) ? 1 : 0);
      }
      if (P.length < 2) continue;
      const st = roadStyle(r);
      roads.push({ r, pts: P, flags: F, st, bridge: !!(r.b || (r.ly > 0 && /^(motorway|trunk)/.test(st.cls))) && !st.minor, filler: !!r.filler });
    }
    this.roads = roads;

    // ---- intersections: crossing half-widths per flagged node
    const nodes = new Map();
    for (let ri = 0; ri < roads.length; ri++) {
      const R = roads[ri];
      if (R.st.minor || R.filler) continue;
      R.flags.forEach((f, i) => {
        if (!f) return;
        const k = key(R.pts[i]);
        let e = nodes.get(k); if (!e) { e = []; nodes.set(k, e); }
        const dirs = [];
        if (i > 0) dirs.push(Math.atan2(R.pts[i][1] - R.pts[i - 1][1], R.pts[i][0] - R.pts[i - 1][0]));
        if (i < R.pts.length - 1) dirs.push(Math.atan2(R.pts[i + 1][1] - R.pts[i][1], R.pts[i + 1][0] - R.pts[i][0]));
        e.push({ ri, i, hw: R.st.hw + (R.st.sidewalk ? 0 : 0), dirs, bridge: R.bridge, service: R.st.base === 'service' });
      });
    }
    const lineAngle = (a, b) => { let d = Math.abs(a - b) % Math.PI; return Math.min(d, Math.PI - d); };
    for (const R of roads) {
      R.ints = []; R.xints = [];
      R.pp0 = prep(R.pts);
      if (R.st.minor || R.filler) continue;
      const pp = R.pp0;
      R.flags.forEach((f, i) => {
        if (!f) return;
        const e = nodes.get(key(R.pts[i]));
        if (!e) return;
        // a bend between two ways of a street (plus only footpaths at the node) is not an intersection
        if (e.reduce((a, o) => a + o.dirs.length, 0) < 3) return;
        const myDir = Math.atan2(R.pts[Math.min(i + 1, R.pts.length - 1)][1] - R.pts[Math.max(0, i - 1)][1], R.pts[Math.min(i + 1, R.pts.length - 1)][0] - R.pts[Math.max(0, i - 1)][0]);
        let r = 0;
        for (const o of e) {
          if (roads[o.ri] === R || o.service) continue;
          if (o.dirs.some((a) => lineAngle(a, myDir) > 0.45)) r = Math.max(r, o.hw);
        }
        if (r > 0) R.ints.push({ s: pp.s[i], r, i });
      });
    }
    // junctions (for the sidewalk corners): flagged nodes where at least one road got an intersection
    this.junctions = [];
    for (const e of nodes.values()) {
      const members = [];
      for (const o of e) {
        const R = roads[o.ri];
        const it = R.ints.find((q) => q.i === o.i);
        members.push({ R, s: R.pp0.s[o.i], r: it ? it.r : 0 });
      }
      if (members.some((m) => m.r > 0) && members.length >= 2) {
        const p = roads[e[0].ri].pts[e[0].i];
        this.junctions.push({ x: p[0], z: p[1], members });
      }
    }
    this.findGeometricJunctions(roads);
    this.footZones(roads);

    // ---- raster of ground-level carriageways (needed before sidewalks / piers); a ramp-foot zone is paving too
    if (this.raster) {
      for (const R of roads) {
        if (R.st.minor || R.filler) continue;
        if (R.bridge) { for (const poly of R.zonePolys) this.raster.fillPolygon([poly], RBIT.ROAD); continue; }
        const pp = prep(R.pts);
        const bit = R.st.base === 'service' ? RBIT.SERVICE : RBIT.ROAD;
        for (let i = 0; i < R.pts.length - 1; i++) {
          const a = R.pts[i], b = R.pts[i + 1], ma = pp.m[i], mb = pp.m[i + 1], hw = R.st.hw;
          this.raster.fillPolygon([[[a[0] - ma[0] * hw, a[1] - ma[1] * hw], [b[0] - mb[0] * hw, b[1] - mb[1] * hw], [b[0] + mb[0] * hw, b[1] + mb[1] * hw], [a[0] + ma[0] * hw, a[1] + ma[1] * hw]]], bit);
        }
      }
    }

    this.groundProfiles(roads);
  }

  /**
   * Junctions OSM gives no shared node for: a road ending on (or inside) another carriageway without a
   * shared vertex, and centre lines crossing mid-segment. Stored as R.xints [{s, r, x, z}] - they fade the
   * lane markings, get crosswalks (no stop line), cut the sidewalks and get corners. The OSM `ints` (used by
   * props and the traffic controls) stay as they were.
   */
  findGeometricJunctions(roads) {
    const G = roads.filter((R) => !R.bridge && !R.st.minor && !R.filler && R.st.base !== 'service');
    const C = 24, grid = new Map(), segs = [];
    const ck = (i, k) => (i + 4096) * 8192 + (k + 4096);
    for (const R of G) {
      const P = R.pts, pp = R.pp0;
      for (let i = 0; i < P.length - 1; i++) {
        const a = P[i], b = P[i + 1], id = segs.length;
        segs.push({ R, a, b, s0: pp.s[i], L: pp.d[i][2], dx: pp.d[i][0], dz: pp.d[i][1] });
        const i0 = Math.floor((Math.min(a[0], b[0]) - 2) / C), i1 = Math.floor((Math.max(a[0], b[0]) + 2) / C);
        const k0 = Math.floor((Math.min(a[1], b[1]) - 2) / C), k1 = Math.floor((Math.max(a[1], b[1]) + 2) / C);
        for (let x = i0; x <= i1; x++) for (let z = k0; z <= k1; z++) { const k = ck(x, z); let l = grid.get(k); if (!l) grid.set(k, (l = [])); l.push(id); }
      }
    }
    const near = (R, s, d) => R.ints.some((q) => Math.abs(q.s - s) < d) || R.xints.some((q) => Math.abs(q.s - s) < d);
    let n = 0;
    const add = (A, sA, B, sB, x, z) => {
      if (near(A, sA, 3) || near(B, sB, 3)) return;
      A.xints.push({ s: sA, r: B.st.hw, x, z }); B.xints.push({ s: sB, r: A.st.hw, x, z });
      this.junctions.push({ x, z, members: [{ R: A, s: sA, r: B.st.hw }, { R: B, s: sB, r: A.st.hw }], geo: true });
      n++;
    };
    // (1) centre lines crossing mid-segment
    const seen = new Set();
    for (const l of grid.values()) {
      for (let p = 0; p < l.length; p++) for (let q = p + 1; q < l.length; q++) {
        const A = segs[l[p]], B = segs[l[q]];
        if (A.R === B.R || (A.R.r.ly || 0) !== (B.R.r.ly || 0)) continue;
        const pk = l[p] < l[q] ? l[p] * 1048576 + l[q] : l[q] * 1048576 + l[p];
        if (seen.has(pk)) continue; seen.add(pk);
        if (Math.abs(A.dx * B.dx + A.dz * B.dz) > 0.9) continue;
        const den = A.dx * B.dz - A.dz * B.dx;
        const ex = B.a[0] - A.a[0], ez = B.a[1] - A.a[1];
        const t = (ex * B.dz - ez * B.dx) / den, u = (ex * A.dz - ez * A.dx) / den;
        if (t < 0 || t > A.L || u < 0 || u > B.L) continue;
        add(A.R, A.s0 + t, B.R, B.s0 + u, A.a[0] + A.dx * t, A.a[1] + A.dz * t);
      }
    }
    // (2) a road ending on / inside another carriageway without a shared vertex
    for (const R of G) {
      const P = R.pts, pp = R.pp0, L = pp.L;
      for (const end of [0, 1]) {
        const s = end ? L : 0;
        if (near(R, s, 3)) continue;
        const p = end ? P[P.length - 1] : P[0], d = end ? pp.d[pp.d.length - 1] : pp.d[0];
        const ox = end ? d[0] : -d[0], oz = end ? d[1] : -d[1]; // outward
        let best = null;
        const ci = Math.floor(p[0] / C), cz = Math.floor(p[1] / C);
        for (let x = ci - 1; x <= ci + 1; x++) for (let z = cz - 1; z <= cz + 1; z++) {
          const l = grid.get(ck(x, z)); if (!l) continue;
          for (const id of l) {
            const S = segs[id];
            if (S.R === R || (S.R.r.ly || 0) !== (R.r.ly || 0) || Math.abs(S.dx * ox + S.dz * oz) > 0.9) continue;
            const den = ox * S.dz - oz * S.dx;
            const ex = S.a[0] - p[0], ez = S.a[1] - p[1];
            const u = (ex * S.dz - ez * S.dx) / den, t = (ex * oz - ez * ox) / den;
            if (t < -0.5 || t > S.L + 0.5) continue;
            const lim = S.R.st.hw + 1.5;
            if (u < -lim || u > lim || (best && Math.abs(u) >= Math.abs(best.u))) continue;
            best = { S, u, t: clamp(t, 0, S.L) };
          }
        }
        if (!best) continue;
        const S = best.S;
        add(R, end ? L + best.u : -best.u, S.R, S.s0 + best.t, p[0] + ox * best.u, p[1] + oz * best.u);
      }
    }
    for (const R of roads) R.xints.sort((a, b) => a.s - b.s);
    this.geoJunctions = n;
  }

  /** Smoothed centre-line height profile of every ground-level carriageway (5 m samples). */
  groundProfiles(roads) {
    this.corridors = [];
    for (const R of roads) {
      if (R.bridge || R.st.minor) continue;
      const n0 = R.pts.length;
      const sd = subdivide(R.pts, 5, R.flags.map((f, i) => (f || i === 0 || i === n0 - 1 ? 1 : 0)));
      R.gpts = sd.pts;
      if (!this.surface) continue;
      const pin = sd.flags, n = R.gpts.length;
      const raw = R.gpts.map((p) => this.surface.baseAt(p[0], p[1]));
      const h = raw.slice(), tmp = new Float64Array(n);
      // [1 2 1]/4 relaxation x12 (~12 m sigma) with junction nodes / ends pinned to the terrain
      for (let it = 0; it < 12 && n > 2; it++) {
        for (let i = 0; i < n; i++) tmp[i] = pin[i] || i === 0 || i === n - 1 ? raw[i] : 0.25 * h[i - 1] + 0.5 * h[i] + 0.25 * h[i + 1];
        for (let i = 0; i < n; i++) h[i] = tmp[i];
      }
      R.ghs = h;
      this.corridors.push({ pts: R.gpts, hs: h, core: R.st.hw + (R.st.sidewalk || 0) + 3, blend: 8 });
    }
  }

  build() {
    if (!this._prepared) this.prepare();
    const roads = this.roads;
    // ---- bridge deck heights
    this.solveBridges(roads);
    // ---- sidewalk corners, kerb ramps, tactile pads (sets the per-side sidewalk cuts)
    this.buildCorners();

    // ---- geometry
    for (const R of roads) {
      const st = R.st;
      if (st.minor) { this.minorPath(R); continue; }
      let pts = R.bridge ? R.dpts : (R.gpts || R.pts);
      const ints = R.bridge ? R.dints : R.ints;
      // markings also stop / get crosswalks at the geometric junctions (negative radius: no stop line);
      // those sit mid-segment, so the polyline gets a vertex there (the ribbon reads junctions per vertex)
      let mints = ints;
      if (!R.bridge && R.xints && R.xints.length) {
        mints = ints.concat(R.xints.map((q) => ({ s: q.s, r: -q.r }))).sort((a, b) => a.s - b.s);
        pts = insertVertices(pts, prep(pts), R.xints.map((q) => q.s));
      }
      const pp = prep(pts);
      this.ribbon(pts, pp, R.bridge ? R.hs : null, st, mints, R.bridge ? R.dzone : null);
      if (R.bridge) this.deckStructure(R, pts, pp);
      else if (!R.filler) this.sidewalks(R, pts, pp);
      // hs: per-point height such that the drivable surface is hs + 0.05 (decks AND ground roads)
      const hs = R.bridge ? R.hs : pts.map((p) => this.gh(p[0], p[1]));
      if (!R.bridge) {
        // ext: hw -> round end caps, so the outside wedge of a bend and the junction interior count as carriageway
        for (let i = 0; i < pts.length - 1; i++) this.segHash.add({ ax: pts[i][0], az: pts[i][1], bx: pts[i + 1][0], bz: pts[i + 1][1], ha: hs[i] + 0.05, hb: hs[i + 1] + 0.05, hw: st.hw, ext: st.hw, road: R });
      } else {
        // a ramp-foot zone is ground carriageway (asphalt lift, sidewalk cut), not deck: see footZones
        for (let i = 0; i < pts.length - 1; i++) if (R.dzone[i] && R.dzone[i + 1]) this.segHash.add({ ax: pts[i][0], az: pts[i][1], bx: pts[i + 1][0], bz: pts[i + 1][1], ha: hs[i] + 0.05, hb: hs[i + 1] + 0.05, hw: st.hw, ext: 0.6, road: R, zone: true });
      }
      this.list.push({ pts, pp, st, ints, bridge: R.bridge, hs, hg: R.bridge ? R.hg : null, filler: R.filler, name: R.r.n });
      if (!R.filler) { this.minimap.push(pts.map((p) => [p[0], p[1]])); this.minimapClass.push(st.cls); }
    }
    this.rails();
  }

  // ---------------------------------------------------------------- carriageway ribbon
  ribbon(pts, pp, hs, st, ints, zone = null) {
    const B = this.roadBuf, n = pts.length;
    const lanes = st.lanes, ow = st.oneway ? 1 : 0, mode = st.markings, hw = st.hw, pw = st.parkW;
    const yBase = 0.05;
    let ip = 0;
    for (let i = 0; i < n - 1; i++) {
      const s0 = pp.s[i], s1 = pp.s[i + 1];
      while (ip < ints.length && ints[ip].s < s0 - 1e-6) ip++;
      // previous intersection: last with s <= s0; next: first with s >= s1
      let prev = null, next = null;
      for (let q = ints.length - 1; q >= 0; q--) if (ints[q].s <= s0 + 1e-6) { prev = ints[q]; break; }
      for (let q = 0; q < ints.length; q++) if (ints[q].s >= s1 - 1e-6) { next = ints[q]; break; }
      const sp = prev ? prev.s : -1e5, rp = prev ? prev.r : 0, sn = next ? next.s : 1e5, rn = next ? next.r : 0;
      const a = pts[i], b = pts[i + 1], ma = pp.m[i], mb = pp.m[i + 1];
      const P0 = [a[0] - ma[0] * hw, a[1] - ma[1] * hw], P1 = [a[0] + ma[0] * hw, a[1] + ma[1] * hw];
      const P2 = [b[0] - mb[0] * hw, b[1] - mb[1] * hw], P3 = [b[0] + mb[0] * hw, b[1] + mb[1] * hw];
      // decks: flat cross-section at the solved deck height; ground roads: sampled from the ground
      // surface (whose road corridors are flattened to the centre-line profile) so it never pokes through;
      // a deck's ramp-foot zone (zone[k]) is ground too, ZONE_LIFT above the street it runs over
      const flat = (k) => hs && !(zone && zone[k]);
      const yv = (P, k) => (flat(k) ? hs[k] : this.gh(P[0], P[1]) + (hs ? ZONE_LIFT : 0)) + yBase;
      const nv = (P, k) => (flat(k) ? null : this.gn(P[0], P[1]));
      const v0 = B.v(P0[0], yv(P0, i), P0[1], -hw, s0, hw, pw, lanes, ow, mode, sp, rp, sn, rn, nv(P0, i));
      const v1 = B.v(P1[0], yv(P1, i), P1[1], hw, s0, hw, pw, lanes, ow, mode, sp, rp, sn, rn, nv(P1, i));
      const v2 = B.v(P2[0], yv(P2, i + 1), P2[1], -hw, s1, hw, pw, lanes, ow, mode, sp, rp, sn, rn, nv(P2, i + 1));
      const v3 = B.v(P3[0], yv(P3, i + 1), P3[1], hw, s1, hw, pw, lanes, ow, mode, sp, rp, sn, rn, nv(P3, i + 1));
      upTri(B, v0, v2, v1); upTri(B, v1, v2, v3);
    }
  }

  // ---------------------------------------------------------------- sidewalks with kerbs
  /** Sidewalk cut intervals on `side` around every junction (corner patches override the default cut). */
  sidewalkCuts(R, side) {
    const out = [];
    for (const it of R.xints && R.xints.length ? R.ints.concat(R.xints) : R.ints) {
      const r = Math.abs(it.r) + 0.2, ov = R.cutOv && R.cutOv.get(cutKey(it.s, side));
      const b = ov && ov.b !== undefined ? ov.b : r, f = ov && ov.f !== undefined ? ov.f : r;
      out.push([it.s - b, it.s + f, !!(ov && ov.b !== undefined), !!(ov && ov.f !== undefined)]);
    }
    return out;
  }
  sidewalks(R, pts, pp) {
    const st = R.st, sw = st.sidewalk;
    if (!sw) return;
    const hw = st.hw, L = pp.L, step = 1;
    for (const side of [-1, 1]) {
      const cuts = this.sidewalkCuts(R, side);
      const ok = [];
      for (let s = 0; s <= L + 1e-6; s += step) {
        const [x, z, dx, dz] = samplePolyline(pts, pp, Math.min(s, L));
        const nx = -dz * side, nz = dx * side;
        let good = !cuts.some(([a, b]) => s > a && s < b);
        if (good && this.raster) {
          for (const off of [hw + 0.9, hw + sw - 0.3]) {
            const v = this.raster.get(x + nx * off, z + nz * off);
            if (v & (RBIT.ROAD | RBIT.WATER)) { good = false; break; }
          }
        }
        ok.push(good);
      }
      // intervals
      let start = -1;
      for (let k = 0; k <= ok.length; k++) {
        if (k < ok.length && ok[k]) { if (start < 0) start = k; continue; }
        if (start >= 0) {
          let a = start === 0 ? 0 : start * step - step * 0.5, b = k === ok.length ? L : Math.min(L, (k - 1) * step + step * 0.5);
          // snap the ends exactly onto the junction cuts so kerbs and corner patches meet without a gap
          let capA = true, capB = true;
          for (const [c0, c1, , pf] of cuts) if (a > c0 && a < c1 + step * 1.01) { a = c1; if (pf) capA = false; }
          for (const [c0, c1, pb] of cuts) if (b < c1 && b > c0 - step * 1.01) { b = c0; if (pb) capB = false; }
          if (b - a > 1.0) this.sidewalkStrip(pts, pp, a, b, side, hw, sw, capA, capB);
          start = -1;
        }
      }
    }
  }
  /**
   * Outer edge of the paving at a sidewalk point: downtown sidewalks run to the building line, so the paving
   * extends (up to 7 m) to a building just behind the standard sidewalk, never onto roads / water / parks.
   * (x,z) centre-line point, (nx,nz) unit normal toward the sidewalk, d0 standard outer offset.
   */
  frontage(x, z, nx, nz, d0) {
    const r = this.raster;
    if (!r) return d0;
    for (let d = d0 + 0.5; d <= d0 + 7.01; d += 0.5) {
      const v = r.get(x + nx * d, z + nz * d);
      if (v & RBIT.BUILDING) return d + 0.35;
      if (v & (RBIT.ROAD | RBIT.SERVICE | RBIT.WATER | RBIT.PARK)) return Math.max(d0, d - 0.9);
    }
    return d0;
  }
  sidewalkStrip(pts, pp, a, b, side, hw, sw, capA = true, capB = true) {
    // sub-polyline between s=a and s=b (<= 2.5 m steps so the frontage edge follows the buildings)
    let sub = [];
    const pa = samplePolyline(pts, pp, a), pb = samplePolyline(pts, pp, b);
    sub.push([pa[0], pa[1]]);
    for (let i = 1; i < pts.length - 1; i++) if (pp.s[i] > a + 0.05 && pp.s[i] < b - 0.05) sub.push(pts[i]);
    sub.push([pb[0], pb[1]]);
    if (sub.length < 2) return;
    // outer (frontage) offset per vertex, measured along the unit normal; miter-scaled like the kerb
    const extOf = (sb, qq) => sb.map((p, i) => {
      const m = qq.m[i], ml = Math.hypot(m[0], m[1]) || 1;
      return this.frontage(p[0], p[1], m[0] / ml * side, m[1] / ml * side, (hw + sw) * ml) / ml;
    });
    let q = prep(sub), ext = extOf(sub, q);
    // refine where the building line jumps between vertices (one midpoint level)
    if (this.raster && ext.some((e, i) => i > 0 && Math.abs(e - ext[i - 1]) > 0.6)) {
      const s2 = [sub[0]];
      for (let i = 1; i < sub.length; i++) {
        if (Math.abs(ext[i] - ext[i - 1]) > 0.6 && q.d[i - 1][2] > 1.2) s2.push([(sub[i][0] + sub[i - 1][0]) / 2, (sub[i][1] + sub[i - 1][1]) / 2]);
        s2.push(sub[i]);
      }
      if (s2.length > sub.length) { sub = s2; q = prep(sub); ext = extOf(sub, q); }
    }
    const S = this.solid;
    const yTop = 0.17, yRoad = 0.04;
    const kerbC = [0.82, 0.81, 0.78], walkC = [0.72, 0.71, 0.68], faceC = [0.75, 0.74, 0.71];
    const kerbW = 0.3;
    const n = sub.length;
    const off = (i, d) => [sub[i][0] + q.m[i][0] * d * side, sub[i][1] + q.m[i][1] * d * side];
    const G = (p) => this.gh(p[0], p[1]);
    for (let i = 0; i < n - 1; i++) {
      const i0 = off(i, hw), i1 = off(i + 1, hw);
      const k0 = off(i, hw + kerbW), k1 = off(i + 1, hw + kerbW);
      const o0 = off(i, ext[i]), o1 = off(i + 1, ext[i + 1]);
      // kerb top
      this.quadG(S, i0, i1, k1, k0, yTop, kerbC);
      // walk
      this.quadG(S, k0, k1, o1, o0, yTop, walkC);
      // kerb face (faces toward the road: -side * right normal)
      const dx = sub[i + 1][0] - sub[i][0], dz = sub[i + 1][1] - sub[i][1], l = Math.hypot(dx, dz) || 1;
      const fn = [dz / l * side, 0, -dx / l * side];
      const gi0 = G(i0), gi1 = G(i1), go0 = G(o0), go1 = G(o1);
      S.quadP([i0[0], gi0 + yRoad, i0[1]], [i1[0], gi1 + yRoad, i1[1]], [i1[0], gi1 + yTop, i1[1]], [i0[0], gi0 + yTop, i0[1]], fn, faceC);
      S.quadP([o0[0], go0 - 0.4, o0[1]], [o1[0], go1 - 0.4, o1[1]], [o1[0], go1 + yTop, o1[1]], [o0[0], go0 + yTop, o0[1]], [-fn[0], 0, -fn[2]], faceC);
    }
    // end caps (not where a corner patch continues the paving)
    for (const i of [0, n - 1]) {
      if (i === 0 ? !capA : !capB) continue;
      const i0 = off(i, hw), o0 = off(i, ext[i]);
      const d = i === 0 ? q.d[0] : q.d[q.d.length - 1];
      const sg = i === 0 ? -1 : 1;
      const gi = G(i0), go = G(o0);
      S.quadP([i0[0], gi + yRoad, i0[1]], [o0[0], go + yRoad, o0[1]], [o0[0], go + yTop, o0[1]], [i0[0], gi + yTop, i0[1]], [d[0] * sg, 0, d[1] * sg], faceC);
    }
    if (this.raster) {
      for (let i = 0; i < n - 1; i++) {
        const i0 = off(i, hw), i1 = off(i + 1, hw), o0 = off(i, ext[i]), o1 = off(i + 1, ext[i + 1]);
        this.raster.fillPolygon([[i0, i1, o1, o0]], RBIT.SIDEWALK);
      }
    }
  }

  // ---------------------------------------------------------------- sidewalk corners, kerb ramps, tactile pads
  /**
   * Fill every junction corner between two sidewalked arms with paving: kerb line + rounded kerb return,
   * a kerb ramp (flared, with a yellow tactile pad) where each crosswalk lands, and the outer corner at the
   * building line. Sets R.cutOv so the straight sidewalk strips start exactly where the patch ends.
   */
  buildCorners() {
    this.tactile = new SolidBuf(1);
    this.cornerCount = 0; this.rampCount = 0;
    if (!this.junctions) return;
    const ppOf = (R) => R._ppg || (R._ppg = prep(R.gpts || R.pts));
    for (const J of this.junctions) {
      if (J.members.some((m) => m.R.bridge || m.R.filler || m.R.st.minor)) continue;
      const arms = [];
      for (const m of J.members) {
        const R = m.R, pts = R.gpts || R.pts, pp = ppOf(R), L = pp.L;
        const all = R.xints && R.xints.length ? R.ints.concat(R.xints) : R.ints;
        for (const dir of [1, -1]) {
          const room = dir > 0 ? L - m.s : m.s;
          if (room < 1.5) continue;
          let gap = Infinity;
          for (const q of all) { const d = (q.s - m.s) * dir; if (d > 0.5 && d < gap) gap = d; }
          const q = samplePolyline(pts, pp, clamp(m.s + dir * Math.min(6, room), 0, L));
          let dx = q[0] - J.x, dz = q[1] - J.z; const l = Math.hypot(dx, dz);
          if (l < 0.5) continue;
          dx /= l; dz /= l;
          arms.push({ R, pts, pp, s: m.s, dir, dx, dz, ang: Math.atan2(dz, dx), hw: R.st.hw, sw: R.st.sidewalk || 0, r: Math.abs(m.r), avail: Math.min(room - 0.5, gap * 0.5 - 0.3) });
        }
      }
      if (arms.length < 2) continue;
      arms.sort((a, b) => a.ang - b.ang);
      for (let k = 0; k < arms.length; k++) {
        const A = arms[k], B = arms[(k + 1) % arms.length];
        let th = B.ang - A.ang; if (th <= 1e-6) th += Math.PI * 2;
        try { this.cornerPatch(J, A, B, th); } catch (e) { /* degenerate corner: keep the plain cut */ }
      }
    }
  }
  /** Point on arm's real polyline at distance t from the junction, offset `o` toward its `side`. */
  armPoint(arm, t, o, side) {
    const q = samplePolyline(arm.pts, arm.pp, clamp(arm.s + arm.dir * t, 0, arm.pp.L));
    return [q[0] - q[3] * o * side, q[1] + q[2] * o * side];
  }
  cornerPatch(J, A, B, th) {
    if (!A.sw || !B.sw || !(A.r > 0) || !(B.r > 0)) return;
    if (A.R === B.R && A.dir === B.dir) return;
    const sideA = A.dir > 0 ? 1 : -1, sideB = B.dir > 0 ? -1 : 1;
    const N = [J.x, J.z];
    const dA = [A.dx, A.dz], mA = [-A.dz, A.dx], dB = [B.dx, B.dz], mB = [-B.dz, B.dx];
    const hA = A.hw, hB = B.hw;
    // arm frames: A offsets toward +mA, B toward -mB (both into the corner)
    const pA = (t, o) => [N[0] + dA[0] * t + mA[0] * o, N[1] + dA[1] * t + mA[1] * o];
    const pB = (t, o) => [N[0] + dB[0] * t - mB[0] * o, N[1] + dB[1] * t - mB[1] * o];
    const open = th >= 2.8;
    // kerb ramps centred on each arm's crosswalk (the road shader paints it |r| + 1 .. |r| + 4 out)
    const depA = Math.min(1.2, A.sw - 0.5), depB = Math.min(1.2, B.sw - 0.5);
    let rampA = depA > 0.6 ? { c0: A.r + 1.4, c1: A.r + 3.6, f0: A.r + 0.9, f1: A.r + 4.1, dep: depA } : null;
    let rampB = depB > 0.6 ? { c0: B.r + 1.4, c1: B.r + 3.6, f0: B.r + 0.9, f1: B.r + 4.1, dep: depB } : null;
    let tKA = 0, tKB = 0, T = 0, rc = 0, K = null, u = null;
    if (!open) {
      const rhs = [-mB[0] * hB - mA[0] * hA, -mB[1] * hB - mA[1] * hA];
      const det = -dA[0] * dB[1] + dB[0] * dA[1];
      if (Math.abs(det) < 1e-4) return;
      tKA = (-rhs[0] * dB[1] + dB[0] * rhs[1]) / det;
      tKB = (dA[0] * rhs[1] - dA[1] * rhs[0]) / det;
      if (tKA < -0.5 || tKB < -0.5 || tKA > 40 || tKB > 40) return;
      const tanH = Math.tan(th / 2);
      // kerb return radius: up to 2 m, but the ramps (when there is room for them) stay on the straight kerb
      let Tlim = 2.0 / tanH;
      if (rampA && rampA.f0 - tKA - 0.1 < 0.6 / tanH) rampA = null;
      if (rampB && rampB.f0 - tKB - 0.1 < 0.6 / tanH) rampB = null;
      if (rampA) Tlim = Math.min(Tlim, rampA.f0 - tKA - 0.1);
      if (rampB) Tlim = Math.min(Tlim, rampB.f0 - tKB - 0.1);
      T = Math.max(0, Tlim); rc = T * tanH;
      if (rc < 0.25) { rc = 0; T = 0; }
      K = pA(tKA, hA);
      const ux = dA[0] + dB[0], uz = dA[1] + dB[1], ul = Math.hypot(ux, uz) || 1;
      u = [ux / ul, uz / ul];
    }
    // strip starts + frontage widths (iterate once: the outer corner may push the start out)
    let tSA = Math.max(open ? A.r + 0.5 : tKA + T + 0.3, rampA ? rampA.f1 + 0.3 : 0);
    let tSB = Math.max(open ? B.r + 0.5 : tKB + T + 0.3, rampB ? rampB.f1 + 0.3 : 0);
    let eA = hA + A.sw, eB = hB + B.sw, O = null, tOA = 0, tOB = 0;
    const outer = () => {
      const rhs = [-mB[0] * eB - mA[0] * eA, -mB[1] * eB - mA[1] * eA];
      const det = -dA[0] * dB[1] + dB[0] * dA[1];
      tOA = (-rhs[0] * dB[1] + dB[0] * rhs[1]) / det; tOB = (dA[0] * rhs[1] - dA[1] * rhs[0]) / det;
      O = pA(tOA, eA);
    };
    const front = () => {
      const fa = this.armPoint(A, tSA, 0, sideA), fb = this.armPoint(B, tSB, 0, sideB);
      const qa = samplePolyline(A.pts, A.pp, clamp(A.s + A.dir * tSA, 0, A.pp.L)), qb = samplePolyline(B.pts, B.pp, clamp(B.s + B.dir * tSB, 0, B.pp.L));
      eA = this.frontage(fa[0], fa[1], -qa[3] * sideA, qa[2] * sideA, hA + A.sw);
      eB = this.frontage(fb[0], fb[1], -qb[3] * sideB, qb[2] * sideB, hB + B.sw);
    };
    front();
    if (!open) {
      outer();
      if (tOA > tSA - 0.2 || tOB > tSB - 0.2) { tSA = Math.max(tSA, tOA + 0.3); tSB = Math.max(tSB, tOB + 0.3); front(); outer(); }
    }
    if (tSA > A.avail || tSB > B.avail) return;
    // ---- outline: kerb (with ramp notches) from A's strip end round to B's, then the outer edge back
    const kerb = [], contour = []; // kerb: {p, h}; contour: {p, k (on the kerb)}
    const yTop = 0.17, yLow = 0.055;
    const K2 = (p, h) => { kerb.push({ p, h }); contour.push({ p, k: true }); };
    const PAk = this.armPoint(A, tSA, hA, sideA), PBk = this.armPoint(B, tSB, hB, sideB);
    K2(PAk, yTop);
    const notchA = rampA ? [[rampA.f1, 0, yTop], [rampA.c1, 0, yLow], [rampA.c0, 0, yLow], [rampA.f0, 0, yTop]] : [];
    for (const [t, , h] of notchA) {
      kerb.push({ p: pA(t, hA), h });
      if (h === yLow) contour.push({ p: pA(t, hA + rampA.dep), k: false });
      else contour.push({ p: pA(t, hA), k: true });
    }
    if (!open) {
      if (rc > 0) {
        const C = [K[0] + u[0] * rc / Math.sin(th / 2), K[1] + u[1] * rc / Math.sin(th / 2)];
        const TA = pA(tKA + T, hA), TB = pB(tKB + T, hB);
        const a0 = Math.atan2(TA[1] - C[1], TA[0] - C[0]);
        let a1 = Math.atan2(TB[1] - C[1], TB[0] - C[0]);
        let sw = a1 - a0; while (sw > Math.PI) sw -= 2 * Math.PI; while (sw < -Math.PI) sw += 2 * Math.PI;
        const ns = Math.max(2, Math.ceil(Math.abs(sw) / 0.22));
        for (let i = 0; i <= ns; i++) { const a = a0 + (sw * i) / ns; K2([C[0] + Math.cos(a) * rc, C[1] + Math.sin(a) * rc], yTop); }
      } else K2(K, yTop);
    } else {
      // open side (T junction far side, a bend's outside): round the junction centre
      const a0 = Math.atan2(mA[1], mA[0]), sw = th - Math.PI;
      K2(pA(0, hA), yTop);
      const ns = sw > 0.05 ? Math.ceil(sw / 0.22) : 0;
      for (let i = 1; i < ns; i++) { const a = a0 + (sw * i) / ns, rr = hA + (hB - hA) * (i / ns); K2([N[0] + Math.cos(a) * rr, N[1] + Math.sin(a) * rr], yTop); }
      K2(pB(0, hB), yTop);
    }
    const notchB = rampB ? [[rampB.f0, 0, yTop], [rampB.c0, 0, yLow], [rampB.c1, 0, yLow], [rampB.f1, 0, yTop]] : [];
    for (const [t, , h] of notchB) {
      kerb.push({ p: pB(t, hB), h });
      if (h === yLow) contour.push({ p: pB(t, hB + rampB.dep), k: false });
      else contour.push({ p: pB(t, hB), k: true });
    }
    K2(PBk, yTop);
    const outerPts = [this.armPoint(B, tSB, eB, sideB)];
    if (!open) { if (tOA < tSA - 0.05 && tOB < tSB - 0.05) outerPts.push(O); }
    else {
      outerPts.push(pB(0, eB));
      const a0 = Math.atan2(-mB[1], -mB[0]), sw = th - Math.PI, ns = sw > 0.05 ? Math.ceil(sw / 0.22) : 0;
      for (let i = 1; i < ns; i++) { const a = a0 - (sw * i) / ns, rr = eB + (eA - eB) * (i / ns); outerPts.push([N[0] + Math.cos(a) * rr, N[1] + Math.sin(a) * rr]); }
      outerPts.push(pA(0, eA));
    }
    outerPts.push(this.armPoint(A, tSA, eA, sideA));
    for (const p of outerPts) contour.push({ p, k: false });
    // ---- validate: a simple polygon, clear of carriageways and water
    const P2 = contour.map((c) => c.p);
    for (let i = 0; i < P2.length; i++) if (!isFinite(P2[i][0]) || !isFinite(P2[i][1])) return;
    if (Math.abs(signedArea(P2)) < 1 || selfIntersects(P2)) return;
    if (this.raster) {
      const probes = [pA(Math.max(0.5, tSA - 0.5), hA + 1.0), pB(Math.max(0.5, tSB - 0.5), hB + 1.0)];
      if (!open) probes.push(rc > 0 ? [K[0] + u[0] * (rc / Math.sin(th / 2) + 0.4), K[1] + u[1] * (rc / Math.sin(th / 2) + 0.4)] : [K[0] + u[0] * 1.2, K[1] + u[1] * 1.2]);
      else { const bx = mA[0] - mB[0], bz = mA[1] - mB[1], bl = Math.hypot(bx, bz); if (bl > 0.1) probes.push([N[0] + bx / bl * ((hA + hB) / 2 + 1.0), N[1] + bz / bl * ((hA + hB) / 2 + 1.0)]); }
      for (const p of probes) if (this.raster.get(p[0], p[1]) & (RBIT.ROAD | RBIT.WATER)) return;
    }
    let faces;
    try { faces = THREE.ShapeUtils.triangulateShape(P2.map((p) => new THREE.Vector2(p[0], p[1])), []); } catch { return; }
    if (!faces || faces.length < P2.length - 2) return;
    // ---- commit: cut overrides for the strips, geometry, raster
    setCut(A.R, A.s, sideA, A.dir, tSA); setCut(B.R, B.s, sideB, B.dir, tSB);
    const S = this.solid;
    const kerbC = [0.82, 0.81, 0.78], walkC = [0.72, 0.71, 0.68], faceC = [0.75, 0.74, 0.71];
    const base = contour.map((c) => this.vg(S, c.p[0], c.p[1], yTop, c.k ? kerbC : walkC));
    for (const f of faces) S.tri(base[f[0]], base[f[1]], base[f[2]]);
    // kerb faces (toward the junction centre), tapering down into the ramps
    for (let i = 0; i + 1 < kerb.length; i++) {
      const a = kerb[i], b = kerb[i + 1];
      if (Math.max(a.h, b.h) < 0.07) continue;
      const ex = b.p[0] - a.p[0], ez = b.p[1] - a.p[1], el = Math.hypot(ex, ez);
      if (el < 0.02) continue;
      let nx = ez / el, nz = -ex / el;
      if (nx * (N[0] - (a.p[0] + b.p[0]) / 2) + nz * (N[1] - (a.p[1] + b.p[1]) / 2) < 0) { nx = -nx; nz = -nz; }
      const ga = this.gh(a.p[0], a.p[1]), gb = this.gh(b.p[0], b.p[1]);
      S.quadP([a.p[0], ga + 0.04, a.p[1]], [b.p[0], gb + 0.04, b.p[1]], [b.p[0], gb + b.h, b.p[1]], [a.p[0], ga + a.h, a.p[1]], [nx, 0, nz], faceC);
    }
    // outer (building side) face
    for (let i = 0; i + 1 < outerPts.length; i++) {
      const a = outerPts[i], b = outerPts[i + 1], ex = b[0] - a[0], ez = b[1] - a[1], el = Math.hypot(ex, ez);
      if (el < 0.02) continue;
      let nx = ez / el, nz = -ex / el;
      if (nx * (N[0] - (a[0] + b[0]) / 2) + nz * (N[1] - (a[1] + b[1]) / 2) > 0) { nx = -nx; nz = -nz; }
      const ga = this.gh(a[0], a[1]), gb = this.gh(b[0], b[1]);
      S.quadP([a[0], ga - 0.4, a[1]], [b[0], gb - 0.4, b[1]], [b[0], gb + yTop, b[1]], [a[0], ga + yTop, a[1]], [nx, 0, nz], faceC);
    }
    // ramps (sloped quad + flared sides) and tactile pads
    for (const [ramp, P, h] of [[rampA, pA, hA], [rampB, pB, hB]]) {
      if (!ramp) continue;
      const g = (p) => this.gh(p[0], p[1]);
      const V = (p, y) => [p[0], g(p) + y, p[1]];
      const k0 = P(ramp.c0, h), k1 = P(ramp.c1, h), i0 = P(ramp.c0, h + ramp.dep), i1 = P(ramp.c1, h + ramp.dep);
      quad3(S, V(k0, yLow), V(k1, yLow), V(i1, yTop), V(i0, yTop), [0.7, 0.69, 0.66]);
      tri3(S, V(P(ramp.f1, h), yTop), V(k1, yLow), V(i1, yTop), walkC);
      tri3(S, V(P(ramp.f0, h), yTop), V(k0, yLow), V(i0, yTop), walkC);
      // tactile warning pad (truncated domes, yellow) on the lower 0.6 m of the ramp
      const pad = (t, o) => { const p = P(t, h + o); return [p[0], g(p) + yLow + (o / ramp.dep) * (yTop - yLow) + 0.006, p[1]]; };
      const t0 = ramp.c0 + 0.12, t1 = ramp.c1 - 0.12, o0 = 0.06, o1 = 0.66;
      const c = [1, 1, 1], U = this.tactile;
      const ux = (t1 - t0) / 0.5, vy = (o1 - o0) / 0.5;
      const v = [pad(t0, o0), pad(t1, o0), pad(t1, o1), pad(t0, o1)];
      const nrm = triNormal(v[0], v[1], v[2]);
      const ids = [[0, 0], [ux, 0], [ux, vy], [0, vy]].map((w, i) => U.v(v[i][0], v[i][1], v[i][2], nrm[0], nrm[1], nrm[2], c, w[0], w[1]));
      U.quad(ids[0], ids[1], ids[2], ids[3]);
      this.rampCount++;
    }
    if (this.raster) this.raster.fillPolygon([P2], RBIT.SIDEWALK);
    this.cornerCount++;
  }
  minorPath(R) {
    const st = R.st, pts = this.surface ? subdivide(R.pts, 5).pts : R.pts, pp = prep(pts), hw = st.hw;
    const S = this.solid;
    const c = st.cls === 'pedestrian' ? [0.78, 0.75, 0.70] : st.cls === 'cycleway' ? [0.55, 0.42, 0.38] : st.cls === 'steps' ? [0.7, 0.7, 0.68] : [0.66, 0.64, 0.6];
    const y = st.cls === 'pedestrian' ? 0.075 : 0.065;
    for (let i = 0; i < pts.length - 1; i++) {
      const a = pts[i], b = pts[i + 1], ma = pp.m[i], mb = pp.m[i + 1];
      this.quadG(S, [a[0] - ma[0] * hw, a[1] - ma[1] * hw], [b[0] - mb[0] * hw, b[1] - mb[1] * hw], [b[0] + mb[0] * hw, b[1] + mb[1] * hw], [a[0] + ma[0] * hw, a[1] + ma[1] * hw], y, c);
    }
    if (st.cls === 'pedestrian') this.list.push({ pts, pp, st, ints: [], bridge: false, hs: pts.map((p) => this.gh(p[0], p[1])), hg: null, filler: false, name: R.r.n });
  }

  // ---------------------------------------------------------------- bridges
  solveBridges(roads) {
    const streets = this.footStreets; // ground carriageways (segHash is filled later, in the ribbon pass)
    const nodeIdx = new Map(), nodes = [];
    const nodeOf = (p) => {
      const k = key(p);
      let i = nodeIdx.get(k);
      if (i === undefined) { i = nodes.length; nodeIdx.set(k, i); nodes.push({ p, adj: [], target: -Infinity, grounded: false, g: 0 }); }
      return i;
    };
    const B = roads.filter((R) => R.bridge);
    for (const R of B) {
      // the deck is subdivided finely inside a ramp-foot zone (footZones), with a vertex exactly where the zone
      // ends, so the climb starts at the carriageway edge
      const sd = subdivideDeck(R.pts, R.flags, R.pp0, R.zone);
      R.dpts = sd.pts; R.dflags = sd.flags; R.ds = sd.s;
      const L = R.pp0.L;
      R.dzone = R.ds.map((s) => (R.zone[0] > 0 && s <= R.zone[0] + 1e-6) || (R.zone[1] > 0 && s >= L - R.zone[1] - 1e-6));
      const motor = /^(motorway|trunk)/.test(R.st.cls);
      const ly = Math.max(1, R.r.ly || 1);
      const base = motor ? Math.max(8, 6.5 * ly) : 6 * ly;
      R.dn = R.dpts.map((p) => nodeOf(p));
      R.dpts.forEach((p, i) => {
        const nd = nodes[R.dn[i]];
        const water = this.isWater(p[0], p[1]);
        // absolute heights: ground under the node (0 over the bay, as before) + the layer clearance. build() runs
        // after the corridors flattened the surface, so this is the ground the mesh ends up at.
        nd.g = water ? 0 : this.gh(p[0], p[1]);
        nd.target = Math.max(nd.target, nd.g + (water ? (motor ? 44 + 8 * ly : 14) : base));
        if (i > 0) {
          const L = Math.hypot(p[0] - R.dpts[i - 1][0], p[1] - R.dpts[i - 1][1]);
          nd.adj.push([R.dn[i - 1], L, 1]); nodes[R.dn[i - 1]].adj.push([R.dn[i], L, 1]); // [node, length, grade multiplier]
        }
      });
      if (R.gend[0]) nodes[R.dn[0]].grounded = true;
      if (R.gend[1]) nodes[R.dn[R.dn.length - 1]].grounded = true;
    }
    // ramp-foot zones: the deck is part of the street it runs over (grounded; ZONE_LIFT above the ground so the
    // two ribbons do not z-fight). The feet themselves stay exactly on the ground, where the ground road continues.
    for (const R of B) {
      const n = R.dn.length;
      for (let i = 0; i < n; i++) {
        if (!R.dzone[i] || (i === 0 && R.gend[0]) || (i === n - 1 && R.gend[1])) continue;
        const nd = nodes[R.dn[i]];
        if (!nd.zone) { nd.zone = true; nd.grounded = true; nd.g += ZONE_LIFT; }
      }
    }
    const grade = 0.06, MIN_CLEAR = 4.6, MIN_CLEAR_SERVICE = 3.6; // m: minimum deck clearance over a crossed street / service road
    const N = nodes.length;
    const relaxDown = (h, fixed) => { // min-plus: h[j] <= h[i] + grade * L
      const heap = new MinHeap();
      for (let i = 0; i < N; i++) if (Number.isFinite(h[i])) heap.push(h[i], i);
      while (heap.size) {
        const i = heap.pop(); const ki = heap.lastKey;
        if (ki > h[i] + 1e-9) continue;
        for (const [j, L, gm] of nodes[i].adj) {
          const cand = h[i] + grade * (gm || 1) * L;
          if (cand < h[j] - 1e-6 && !(fixed && fixed[j])) { h[j] = cand; heap.push(cand, j); }
        }
      }
    };
    const relaxUp = (lo) => { // max-minus: lo[j] >= lo[i] - grade * L
      const heap = new MinHeap();
      for (let i = 0; i < N; i++) if (lo[i] > -Infinity) heap.push(-lo[i], i);
      while (heap.size) {
        const i = heap.pop(); const ki = -heap.lastKey;
        if (ki < lo[i] - 1e-9) continue;
        for (const [j, L, gm] of nodes[i].adj) {
          const cand = lo[i] - grade * (gm || 1) * L;
          if (cand > lo[j] + 1e-6) { lo[j] = cand; heap.push(-cand, j); }
        }
      }
    };
    // suspended bridges (deckHolds): hold the decks level between the anchorage and the far shore
    const hold = new Map();
    for (const rule of this.deckHolds) for (const [i, v] of this.holdDecks(B, nodes, rule)) hold.set(i, v);
    // reachability from the grounded ends alone: a held height never exceeds it (no cliff at a ramp foot)
    const ug = new Float64Array(N).fill(Infinity);
    nodes.forEach((n, i) => { if (n.grounded) ug[i] = n.g; });
    relaxDown(ug);
    const fixed = new Uint8Array(N), pin = new Float64Array(N);
    for (const [i, v] of hold) { if (nodes[i].grounded) continue; fixed[i] = 1; pin[i] = Math.min(v, ug[i]); }
    // upper envelope: targets, limited to `grade` from grounded ends / held decks
    const h = new Float64Array(N);
    nodes.forEach((n, i) => { h[i] = n.grounded ? n.g : fixed[i] ? pin[i] : n.target; });
    relaxDown(h, fixed);
    // lower envelope from grounded ends and held decks (grounded ends sit at different terrain heights,
    // and a held deck must ramp down at most `grade` instead of dropping off a cliff)
    const lo = new Float64Array(N).fill(-Infinity);
    nodes.forEach((n, i) => { if (n.grounded) lo[i] = n.g; else if (fixed[i]) lo[i] = pin[i]; });
    relaxUp(lo);
    for (let i = 0; i < N; i++) { if (lo[i] > h[i]) h[i] = lo[i]; if (fixed[i]) h[i] = pin[i]; }
    // ---- clearance over the streets a deck crosses: a deck hanging 0.5-3 m over a carriageway is neither a
    // bridge nor a ramp foot (cars clipped through freeway ramps over 1st / 2nd / Harrison Street). Deck nodes get
    // a lower bound of street + MIN_CLEAR, propagated at the grade; samples within 30 m of the piece's own
    // grounded end (the ramp merging into that street; measured from the end of its ramp-foot zone, which is part
    // of the street) and over the deck's own ground continuation are exempt. A service road (parking aisle,
    // driveway) only asks for MIN_CLEAR_SERVICE (a car passes under the slab).
    const clear = new Float64Array(N).fill(-Infinity);
    for (const R of B) {
      const nm = R.r.n || '', Lp = R.pp0.L;
      const exA = R.gend[0] ? R.zone[0] + 30 : -1, exB = R.gend[1] ? Lp - R.zone[1] - 30 : Lp + 1;
      for (let i = 0; i + 1 < R.dpts.length; i++) {
        const a = R.dpts[i], b = R.dpts[i + 1], L = Math.hypot(b[0] - a[0], b[1] - a[1]);
        if (L < 1e-6) continue;
        let gm = 1;
        for (const e of nodes[R.dn[i]].adj) if (e[0] === R.dn[i + 1]) gm = e[2] || 1;
        const n = Math.max(1, Math.ceil(L / 3));
        for (let k = 0; k <= n; k++) {
          const t = k / n, x = a[0] + (b[0] - a[0]) * t, z = a[1] + (b[1] - a[1]) * t, s = R.ds[i] + L * t;
          if (s < exA || s > exB) continue;
          let over = 0;
          streets.forEachNear(x, z, 0, (sg, tt, lat) => { if (Math.abs(lat) <= sg.hw && (sg.road.r.n || '') !== nm) over = Math.max(over, sg.road.st.base === 'service' ? MIN_CLEAR_SERVICE : MIN_CLEAR); });
          if (!over) continue;
          const need = this.gh(x, z) + 0.05 + over, da = L * t, db = L - da;
          if (need - grade * gm * da > clear[R.dn[i]]) clear[R.dn[i]] = need - grade * gm * da;
          if (need - grade * gm * db > clear[R.dn[i + 1]]) clear[R.dn[i + 1]] = need - grade * gm * db;
        }
      }
    }
    {
      let any = false;
      for (let i = 0; i < N; i++) if (clear[i] > h[i] + 0.05 && !fixed[i] && !nodes[i].grounded) { lo[i] = Math.max(lo[i], clear[i]); any = true; }
      if (any) {
        relaxUp(lo);
        for (let i = 0; i < N; i++) if (lo[i] > h[i] && !fixed[i]) h[i] = lo[i];
      }
    }
    // ---- grounded ends must meet the ground. The grade limit alone leaves a ramp foot in the air (or buried)
    // when a piece's other end is too far up / down for 6 % over its length (Harrison Street over the freeway cut
    // climbs 10 %: the deck started 2 m above the street, a cliff cars fell off). Such a piece gets the grade it
    // needs over its climb (the part between its grounded runs: foot + ramp-foot zone), its grounded runs are
    // pinned to the ground, and the envelopes are solved again.
    let repin = false;
    const setMul = (a, b, mul) => {
      for (const e of nodes[a].adj) if (e[0] === b) e[2] = Math.max(e[2] || 1, mul);
      for (const e of nodes[b].adj) if (e[0] === a) e[2] = Math.max(e[2] || 1, mul);
    };
    for (const R of B) {
      const n = R.dn.length, s = R.ds;
      let a = 0; while (a < n && nodes[R.dn[a]].grounded) a++;           // nodes 0 .. a-1: grounded run at the start
      let b = n - 1; while (b >= 0 && nodes[R.dn[b]].grounded) b--;     // nodes b+1 .. n-1: grounded run at the end
      const whole = a > b;                                               // no climb: the piece lies on the ground
      const sa = Math.max(0, a - 1), sb = Math.min(n - 1, b + 1);       // shoulders: the climb runs between them
      const Lc = whole ? 0 : Math.max(1, s[sb] - s[sa]);
      for (let e = 0; e < 2; e++) {
        const run = e === 0 ? [0, a - 1] : [b + 1, n - 1];
        if (run[0] > run[1]) continue;
        let off = 0;
        for (let i = run[0]; i <= run[1]; i++) if (!fixed[R.dn[i]]) off = Math.max(off, Math.abs(h[R.dn[i]] - nodes[R.dn[i]].g));
        if (off < 0.25) continue;
        if (!whole) {
          // the grade the climb needs from this shoulder: to the far shoulder's height, and to every clearance
          // bound along the climb (a bound the grade cannot reach from the pinned shoulder would be a cliff)
          const sh = e === 0 ? sa : sb, ni = R.dn[sh], far = R.dn[e === 0 ? sb : sa], gsh = nodes[ni].g;
          let need = Math.abs(h[far] - gsh) / Lc;
          for (let j = sa + 1; j < sb; j++) {
            const c = clear[R.dn[j]];
            if (c > gsh) need = Math.max(need, (c - gsh) / Math.max(1, Math.abs(s[j] - s[sh])));
          }
          const mul = Math.max(1, (need / grade) * 1.02);
          for (let i = sa + 1; i <= sb; i++) setMul(R.dn[i - 1], R.dn[i], mul);
        }
        for (let i = run[0]; i <= run[1]; i++) { const ni = R.dn[i]; if (!fixed[ni]) { fixed[ni] = 1; pin[ni] = nodes[ni].g; } }
        repin = true;
      }
    }
    if (repin) {
      nodes.forEach((n, i) => { h[i] = fixed[i] ? pin[i] : n.grounded ? n.g : n.target; });
      relaxDown(h, fixed);
      lo.fill(-Infinity);
      nodes.forEach((n, i) => { if (n.grounded) lo[i] = n.g; else if (fixed[i]) lo[i] = pin[i]; if (clear[i] > lo[i] && !fixed[i] && !n.grounded) lo[i] = clear[i]; });
      relaxUp(lo);
      for (let i = 0; i < N; i++) { if (lo[i] > h[i]) h[i] = lo[i]; if (fixed[i]) h[i] = pin[i]; }
    }
    for (const R of B) {
      R.hs = R.dn.map((i) => h[i]);
      R.hg = R.dn.map((i) => nodes[i].g);
      const pp = prep(R.dpts);
      R.dints = [];
      R.dflags.forEach((f, i) => { if (f) { const it = R.ints.find((q) => q.i !== undefined && R.pts[q.i] === R.dpts[i]); if (it) R.dints.push({ s: pp.s[i], r: it.r }); } });
      for (let i = 0; i < R.dpts.length - 1; i++) {
        if (R.dzone[i] && R.dzone[i + 1]) continue; // a ramp-foot zone is street, not deck (segHash, in build())
        this.deckHash.add({ ax: R.dpts[i][0], az: R.dpts[i][1], bx: R.dpts[i + 1][0], bz: R.dpts[i + 1][1], ha: R.hs[i] + 0.05, hb: R.hs[i + 1] + 0.05, hw: R.st.hw - 0.1, ext: 0.6, road: R });
      }
    }
    this.bridgeCount = B.length;
  }

  /**
   * Ramp-foot zone of a bridge piece at its grounded end `end` (0 = start, 1 = end): the arc length from that end
   * over which the deck is part of the street it runs over. OSM ends a ramp's bridge way where the span ends, a
   * few metres past (or inside) the street the ramp meets, and the approach beyond is a ground road here; so the
   * deck used to climb at the grade from a foot inside / beside the carriageway, hanging 0.5-1.3 m over it for
   * 20-30 m (cars drove through the slab). Walking from the foot in FOOT_STEP steps (at most FOOT_MAX m), the zone
   * runs to the last sample whose centre line lies inside another road's ground carriageway (not a same-name road,
   * not the deck's own continuation: a segment at the foot node leaving along the deck's direction): a street
   * within FOOT_GAP m of the foot / the previous hit, a service road (driveway, parking aisle: crossed at grade
   * like a ramp's own foot) within FOOT_GAP_SERVICE m. The deck is pinned to the ground inside the zone and starts
   * its climb at the carriageway edge. Returns 0 when the foot is clear of every other carriageway.
   */
  footZones(roads) {
    const groundKeys = new Set();
    for (const R of roads) if (!R.bridge && !R.st.minor) for (const p of R.pts) groundKeys.add(key(p));
    const streets = new SegmentHash(32);
    for (const G of roads) {
      if (G.bridge || G.st.minor) continue;
      for (let i = 0; i + 1 < G.pts.length; i++) streets.add({ ax: G.pts[i][0], az: G.pts[i][1], bx: G.pts[i + 1][0], bz: G.pts[i + 1][1], hw: G.st.hw, ext: G.st.hw, road: G });
    }
    this.footStreets = streets;
    this.footZoneCount = 0;
    for (const R of roads) {
      if (!R.bridge) continue;
      const g = R.r.g || [0, 0], L = R.pp0.L;
      R.gend = [!!(g[0] || groundKeys.has(key(R.pts[0]))), !!(g[1] || groundKeys.has(key(R.pts[R.pts.length - 1])))];
      R.zone = [R.gend[0] ? this.footZone(R, streets, 0) : 0, R.gend[1] ? this.footZone(R, streets, 1) : 0];
      // zone outlines for the raster (one quad per segment of the zone's part of the centre line)
      R.zonePolys = [];
      const spans = [];
      if (R.zone[0] + R.zone[1] >= L - 1e-6) spans.push([0, L]);
      else { if (R.zone[0] > 0) spans.push([0, R.zone[0]]); if (R.zone[1] > 0) spans.push([L - R.zone[1], L]); }
      for (const [s0, s1] of spans) {
        const sub = subPolyline(R.pts, R.pp0, s0, s1), pp = prep(sub), hw = R.st.hw;
        for (let i = 0; i + 1 < sub.length; i++) {
          const a = sub[i], b = sub[i + 1], ma = pp.m[i], mb = pp.m[i + 1];
          R.zonePolys.push([[a[0] - ma[0] * hw, a[1] - ma[1] * hw], [b[0] - mb[0] * hw, b[1] - mb[1] * hw], [b[0] + mb[0] * hw, b[1] + mb[1] * hw], [a[0] + ma[0] * hw, a[1] + ma[1] * hw]]);
        }
        this.footZoneCount++;
      }
    }
  }

  footZone(R, streets, end) {
    const pts = R.pts, pp = R.pp0, L = pp.L, nm = R.r.n || '';
    const fp = end ? pts[pts.length - 1] : pts[0], fk = key(fp);
    const d = end ? pp.d[pp.d.length - 1] : pp.d[0];
    let last = -1;
    for (let u = 0; u <= Math.min(L, FOOT_MAX) + 1e-9; u += FOOT_STEP) {
      const gap = u - Math.max(last, 0);
      if (gap > FOOT_GAP_SERVICE) break;
      const [x, z] = samplePolyline(pts, pp, end ? L - u : u);
      let over = false;
      streets.forEachNear(x, z, 0, (sg, t, lat) => {
        if (over || Math.abs(lat) > sg.hw) return;
        const G = sg.road;
        if (nm && (G.r.n || '') === nm) return;
        if (gap > FOOT_GAP && G.st.base !== 'service') return;
        if (key([sg.ax, sg.az]) === fk || key([sg.bx, sg.bz]) === fk) {
          const l = Math.hypot(sg.bx - sg.ax, sg.bz - sg.az) || 1;
          if (Math.abs(((sg.bx - sg.ax) * d[0] + (sg.bz - sg.az) * d[1]) / l) > 0.7) return;
        }
        over = true;
      });
      if (over) last = u;
    }
    return last < 0 ? 0 : Math.min(L, last + FOOT_STEP * 0.5);
  }

  /** One deckHolds rule (see DEFAULT_DECK_HOLD): returns Map(nodeIndex -> held deck height). */
  holdDecks(B, nodes, rule) {
    const hold = new Map();
    const { top, lowerGap, anchorage, minWet, minLen, farShore, maxOffset } = rule;
    const len = (p) => { let L = 0; for (let i = 1; i < p.length; i++) L += Math.hypot(p[i][0] - p[i - 1][0], p[i][1] - p[i - 1][1]); return L; };
    let axis = null, best = 0;
    for (const R of B) {
      if (!/^(motorway|trunk)$/.test(R.st.base) || !rule.name.test(R.r.n || '')) continue;
      const pts = R.dpts;
      let wet = 0;
      for (let i = 1; i < pts.length; i++) {
        const a = pts[i - 1], b = pts[i];
        if (this.isWater((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)) wet += Math.hypot(b[0] - a[0], b[1] - a[1]);
      }
      if (wet > minWet && len(pts) > minLen && wet > best) { best = wet; axis = pts; }
    }
    if (!axis) return hold;
    let pts = axis.map((p) => [p[0], p[1]]);
    if (Math.hypot(pts[0][0], pts[0][1]) > Math.hypot(pts[pts.length - 1][0], pts[pts.length - 1][1])) pts.reverse(); // s = 0 at the end nearer the origin
    const pp = prep(pts), L = pp.L;
    let shore = null, far = L;
    for (let s = 0; s < L; s += 4) {
      const [x, z] = samplePolyline(pts, pp, s);
      const w = this.isWater(x, z);
      if (shore === null) { if (w) shore = s; } else if (!w && s > shore + farShore) { far = s; break; }
    }
    if (shore === null) return hold;
    const s0 = shore - anchorage, s1 = far;
    const project = (x, z) => {
      let bs = 0, bu = Infinity, bd = null;
      for (let i = 0; i < pts.length - 1; i++) {
        const a = pts[i], d = pp.d[i];
        const t = clamp((x - a[0]) * d[0] + (z - a[1]) * d[1], 0, d[2]);
        const u = Math.hypot(a[0] + d[0] * t - x, a[1] + d[1] * t - z);
        if (u < bu) { bu = u; bs = pp.s[i] + t; bd = d; }
      }
      return [bs, bu, bd];
    };
    for (const R of B) {
      if (!/^(motorway|trunk)$/.test(R.st.base)) continue;
      const lower = (R.r.ly || 1) <= 1;
      R.dpts.forEach((p, i) => {
        const [s, u, d] = project(p[0], p[1]);
        if (u > maxOffset || s < s0 || s > s1) return;
        const q = R.dpts[Math.min(i + 1, R.dpts.length - 1)], r = R.dpts[Math.max(0, i - 1)];
        const dx = q[0] - r[0], dz = q[1] - r[1], l = Math.hypot(dx, dz) || 1;
        if (Math.abs((dx * d[0] + dz * d[1]) / l) < 0.95) return; // parallel decks only (not ramps)
        hold.set(R.dn[i], lower ? top - lowerGap : top);
      });
    }
    this.heldDecks.push({ name: String(rule.name), s0, s1, shore, top, lower: top - lowerGap, nodes: hold.size });
    return hold;
  }

  deckStructure(R, pts, pp) {
    const S = this.solid, hs = R.hs, hw = R.st.hw, hg = R.hg || hs.map(() => 0);
    const concrete = [0.62, 0.61, 0.58], under = [0.42, 0.41, 0.4], barrier = [0.75, 0.74, 0.7];
    const depth = /^(motorway|trunk)/.test(R.st.cls) ? 1.6 : 1.2, bh = 0.95, bw = 0.35;
    const n = pts.length;
    const edge = (i, d) => [pts[i][0] + pp.m[i][0] * d, pts[i][1] + pp.m[i][1] * d];
    for (let i = 0; i < n - 1; i++) {
      const h0 = hs[i], h1 = hs[i + 1];
      if (h0 - hg[i] < 0.25 && h1 - hg[i + 1] < 0.25) continue;
      const dx = pp.d[i][0], dz = pp.d[i][1];
      const right = [-dz, 0, dx], left = [dz, 0, -dx];
      for (const sg of [-1, 1]) {
        const o0 = edge(i, hw * sg), o1 = edge(i + 1, hw * sg);
        const in0 = edge(i, (hw - bw) * sg), in1 = edge(i + 1, (hw - bw) * sg);
        const nOut = sg > 0 ? right : left, nIn = sg > 0 ? left : right;
        // outer face (deck edge + barrier)
        S.quadP([o0[0], h0 - depth, o0[1]], [o1[0], h1 - depth, o1[1]], [o1[0], h1 + bh, o1[1]], [o0[0], h0 + bh, o0[1]], nOut, concrete);
        // barrier inner face
        S.quadP([in0[0], h0 + 0.04, in0[1]], [in1[0], h1 + 0.04, in1[1]], [in1[0], h1 + bh, in1[1]], [in0[0], h0 + bh, in0[1]], nIn, barrier);
        // barrier top
        S.quadP([in0[0], h0 + bh, in0[1]], [in1[0], h1 + bh, in1[1]], [o1[0], h1 + bh, o1[1]], [o0[0], h0 + bh, o0[1]], [0, 1, 0], barrier);
      }
      // underside
      const l0 = edge(i, -hw), l1 = edge(i + 1, -hw), r0 = edge(i, hw), r1 = edge(i + 1, hw);
      S.quadP([l0[0], h0 - depth, l0[1]], [l1[0], h1 - depth, l1[1]], [r1[0], h1 - depth, r1[1]], [r0[0], h0 - depth, r0[1]], [0, -1, 0], under);
    }
    // end caps where the deck stops in the air (clipped at the map edge or unconnected)
    for (const i of [0, n - 1]) {
      if (hs[i] - hg[i] < 1) continue;
      const l = edge(i, -hw), r = edge(i, hw), d = i === 0 ? pp.d[0] : pp.d[n - 2], sg = i === 0 ? -1 : 1;
      S.quadP([l[0], hs[i] - depth, l[1]], [r[0], hs[i] - depth, r[1]], [r[0], hs[i] + 0.05, r[1]], [l[0], hs[i] + 0.05, l[1]], [d[0] * sg, 0, d[1] * sg], concrete);
    }
    // piers + slab colliders
    const L = pp.L;
    const over = (s) => this.isWater(...samplePolyline(pts, pp, s).slice(0, 2));
    const spacing = (s) => (over(s) ? 70 : 32);
    for (let s = 14; s < L - 4; ) {
      const [x, z, dx, dz, si, t] = samplePolyline(pts, pp, s);
      const hh = hs[si] + (hs[si + 1] - hs[si]) * t;
      const gg = hg[si] + (hg[si + 1] - hg[si]) * t;
      if (hh - gg > 4.2) {
        const water = this.isWater(x, z);
        const cols = hw > 7.5 ? [-(hw - 2.6), hw - 2.6] : [0];
        const cs = water ? 3.2 : hw > 7.5 ? 1.5 : 1.7;
        const ang = Math.atan2(dz, dx);
        let placed = 0;
        for (const off of cols) {
          let cx = x - dz * off, cz = z + dx * off, okp = false;
          for (const sh of [0, 4, -4, 8, -8]) {
            const px = cx + dx * sh, pz = cz + dz * sh;
            if (!this.raster || !this.raster.any(px, pz, cs * 0.7 + 0.3, RBIT.ROAD)) { cx = px; cz = pz; okp = true; break; }
          }
          if (!okp) continue;
          const yb = water ? -3 : this.gh(cx, cz) - 0.5;
          const top = hh - depth;
          S.box(cx, yb, cz, cs, top - yb, cs, ang, [0.6, 0.59, 0.56]);
          const ext = cs * 0.75;
          this.piers.push([cx - ext, yb, cx + ext, top, cz - ext, cz + ext]);
          placed++;
        }
        if (placed === 2) { // cross beam
          S.box(x, hh - depth - 1.1, z, 1.4, 1.1, hw * 2 - 2, ang, [0.58, 0.57, 0.54]);
        }
      }
      s += spacing(s);
    }
    for (let s = 0; s < L; s += 6) {
      const s1 = Math.min(L, s + 6);
      const A = samplePolyline(pts, pp, s), Bp = samplePolyline(pts, pp, s1);
      const ha = hs[A[4]] + (hs[A[4] + 1] - hs[A[4]]) * A[5], hb = hs[Bp[4]] + (hs[Bp[4] + 1] - hs[Bp[4]]) * Bp[5];
      const ga = hg[A[4]] + (hg[A[4] + 1] - hg[A[4]]) * A[5], gb = hg[Bp[4]] + (hg[Bp[4] + 1] - hg[Bp[4]]) * Bp[5];
      const hmin = Math.min(ha, hb);
      if (Math.min(ha - ga, hb - gb) < 4.5) continue;
      const xs = [], zs = [];
      for (const P of [A, Bp]) for (const sg of [-1, 1]) { xs.push(P[0] - P[3] * hw * sg); zs.push(P[1] + P[2] * hw * sg); }
      this.piers.push([Math.min(...xs), hmin - depth, Math.max(...xs), hmin - 0.35, Math.min(...zs), Math.max(...zs)]);
    }
  }

  // ---------------------------------------------------------------- rails
  rails() {
    const S = this.railBuf;
    const ballast = [0.45, 0.42, 0.38];
    for (const r of this.railIn) {
      if (!r || !Array.isArray(r.p) || r.p.length < 2 || r.b) continue;
      let pts = r.p.filter((p) => p && isFinite(p[0]) && isFinite(p[1]));
      if (pts.length < 2) continue;
      if (this.surface) pts = subdivide(pts, 5).pts;
      const pp = prep(pts);
      const embedded = !!r.e;
      const G = (p) => this.gh(p[0], p[1]);
      // street-running rails sit flush in the asphalt (worn steel heads), ballasted track stands proud
      const y0 = embedded ? 0.045 : 0.2, y1 = embedded ? 0.066 : y0 + 0.07;
      const steel = embedded ? [0.36, 0.36, 0.37] : [0.55, 0.55, 0.56];
      if (!embedded) {
        for (let i = 0; i < pts.length - 1; i++) {
          const a = pts[i], b = pts[i + 1], ma = pp.m[i], mb = pp.m[i + 1], w = 1.7;
          this.quadG(this.solid, [a[0] - ma[0] * w, a[1] - ma[1] * w], [b[0] - mb[0] * w, b[1] - mb[1] * w], [b[0] + mb[0] * w, b[1] + mb[1] * w], [a[0] + ma[0] * w, a[1] + ma[1] * w], 0.12, ballast);
        }
      }
      for (const g of [-0.7175, 0.7175]) {
        for (let i = 0; i < pts.length - 1; i++) {
          const a = pts[i], b = pts[i + 1], ma = pp.m[i], mb = pp.m[i + 1];
          const p0 = [a[0] + ma[0] * (g - 0.035), a[1] + ma[1] * (g - 0.035)], p1 = [b[0] + mb[0] * (g - 0.035), b[1] + mb[1] * (g - 0.035)];
          const q0 = [a[0] + ma[0] * (g + 0.035), a[1] + ma[1] * (g + 0.035)], q1 = [b[0] + mb[0] * (g + 0.035), b[1] + mb[1] * (g + 0.035)];
          this.quadG(S, p0, p1, q1, q0, y1, steel);
          const dx = pp.d[i][0], dz = pp.d[i][1];
          const gp0 = G(p0), gp1 = G(p1), gq0 = G(q0), gq1 = G(q1);
          S.quadP([p0[0], gp0 + y0, p0[1]], [p1[0], gp1 + y0, p1[1]], [p1[0], gp1 + y1, p1[1]], [p0[0], gp0 + y1, p0[1]], [dz, 0, -dx], steel);
          S.quadP([q0[0], gq0 + y0, q0[1]], [q1[0], gq1 + y0, q1[1]], [q1[0], gq1 + y1, q1[1]], [q0[0], gq0 + y1, q0[1]], [-dz, 0, dx], steel);
        }
      }
    }
  }

  meshes(roadMat, concreteMat, railMat) {
    const out = [];
    const a = this.roadBuf.build(roadMat); if (a) { a.name = 'roads'; out.push(a); }
    const b = this.solid.build(concreteMat, { cast: true, receive: true }); if (b) { b.name = 'sidewalks+structures'; out.push(b); }
    const c = this.railBuf.build(railMat); if (c) { c.name = 'rails'; out.push(c); }
    if (this.tactile && this.tactile.n) {
      const t = this.tactile.build(createTactileMaterial(), { cast: false, receive: true });
      if (t) { t.name = 'tactile-pads'; out.push(t); }
    }
    return out;
  }
}

/** Sidewalk cut override key (junction arc position + side). */
const cutKey = (s, side) => `${Math.round(s * 10)}|${side}`;
function setCut(R, s, side, dir, t) {
  if (!R.cutOv) R.cutOv = new Map();
  const k = cutKey(s, side);
  let e = R.cutOv.get(k); if (!e) { e = {}; R.cutOv.set(k, e); }
  if (dir > 0) e.f = t; else e.b = t;
}
function segX(a, b, c, d) {
  const d1x = b[0] - a[0], d1z = b[1] - a[1], d2x = d[0] - c[0], d2z = d[1] - c[1];
  const den = d1x * d2z - d1z * d2x;
  if (Math.abs(den) < 1e-12) return false;
  const t = ((c[0] - a[0]) * d2z - (c[1] - a[1]) * d2x) / den, u = ((c[0] - a[0]) * d1z - (c[1] - a[1]) * d1x) / den;
  return t > 1e-6 && t < 1 - 1e-6 && u > 1e-6 && u < 1 - 1e-6;
}
/** True if any two non-adjacent edges of the closed ring cross. */
function selfIntersects(P) {
  const n = P.length;
  for (let i = 0; i < n; i++) for (let j = i + 2; j < n; j++) {
    if (i === 0 && j === n - 1) continue;
    if (segX(P[i], P[(i + 1) % n], P[j], P[(j + 1) % n])) return true;
  }
  return false;
}
function triNormal(a, b, c) {
  const ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2], vx = c[0] - a[0], vy = c[1] - a[1], vz = c[2] - a[2];
  let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
  const l = Math.hypot(nx, ny, nz) || 1; if (ny < 0) { nx = -nx; ny = -ny; nz = -nz; }
  return [nx / l, ny / l, nz / l];
}
/** Up-facing planar quad / triangle from 3D points (normal from the geometry). */
function quad3(S, a, b, c, d, col) { const n = triNormal(a, b, c); S.quad(S.v(...a, ...n, col), S.v(...b, ...n, col), S.v(...c, ...n, col), S.v(...d, ...n, col)); }
function tri3(S, a, b, c, col) { const n = triNormal(a, b, c); S.tri(S.v(...a, ...n, col), S.v(...b, ...n, col), S.v(...c, ...n, col)); }

function upTri(B, a, b, c) {
  const P = B.pos;
  const ax = P[a * 3], az = P[a * 3 + 2];
  const cy = (P[b * 3 + 2] - az) * (P[c * 3] - ax) - (P[b * 3] - ax) * (P[c * 3 + 2] - az);
  if (cy > 0) B.idx.push(a, b, c); else B.idx.push(a, c, b);
}
export function quadUp(S, p0, p1, p2, p3, y, c) {
  const a = S.v(p0[0], y, p0[1], 0, 1, 0, c), b = S.v(p1[0], y, p1[1], 0, 1, 0, c);
  const d = S.v(p2[0], y, p2[1], 0, 1, 0, c), e = S.v(p3[0], y, p3[1], 0, 1, 0, c);
  S.quad(a, b, d, e);
}
