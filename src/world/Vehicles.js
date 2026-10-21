// Procedural vehicle library: real-proportioned cars, vans, trucks and a 40 ft low-floor city bus built from
// lofted side profiles (smooth bevelled bodies with tumblehome, wheel arches cut into the sill line,
// glass / pillars / lamps assigned per face), lathe wheels (tyre, sidewall, 5/6-spoke rims), mirrors,
// door seams, grille, plates and lamps. ~700-1,400 triangles each (LOD 0), ~300-450 (LOD 1).
//
// Geometry is non-indexed with attributes the vehicle material reads:
//   color (vec3, linear), aPaint (1 = body paint, multiplied by the instance colour), aLamp (0 none,
//   1 head, 2 tail/brake, 3 amber, 4 lit sign, 5/6 police red/blue), aSurf (roughness, metalness, clearcoat).
// Local frame: forward = -Z, up = +Y, right = +X, origin on the ground between the axles.
//
//   makeVehicleGeometry(type, { lod, livery }) -> THREE.BufferGeometry (a fresh clone; userData: tris, spec)
//                                          livery (bus only): { lower, stripe } vertex materials ({ c, rough, metal, cc })
//                                          for the lower body band and the window-line stripe (DEFAULT_BUS_LIVERY)
//   VEHICLE_SPECS[type]                 -> { L, W, H, Hc, WB, wheelR, ... } (L/W/Hc drive traffic colliders)
//   pickVehicleColor(rnd, kind, out)    -> realistic paint distribution ('car' | 'taxi' | 'truck' | 'bus' | 'police')
//   createVehicleMaterial({ brake, lamps }) -> MeshPhysicalMaterial (clearcoat) with userData.u.uDark;
//                                          brake: reads the per-instance `aBrake` attribute (traffic)
import * as THREE from 'three';

const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const lerp = (a, b, t) => a + (b - a) * t;
const sstep = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };

// ------------------------------------------------------------------ part builder (shared with furniture.js)
/**
 * Accumulates non-indexed triangles with per-vertex material data. Material objects:
 * { c: [r,g,b] linear, paint 0|1, lamp, rough, metal, cc (clearcoat), emit }.
 */
export class PartBuilder {
  constructor() { this.P = []; this.N = []; this.C = []; this.A = []; }
  get triCount() { return this.P.length / 9; }
  _v(p, n, m) {
    this.P.push(p[0], p[1], p[2]); this.N.push(n[0], n[1], n[2]); this.C.push(m.c[0], m.c[1], m.c[2]);
    this.A.push(m.paint || 0, m.lamp || 0, m.rough ?? 0.5, m.metal ?? 0, m.cc ?? 0, m.emit ?? 0);
  }
  /** Triangle; with vertex normals the winding is fixed to agree with them, else flat & as given. */
  tri(a, b, c, m, na, nb, nc) {
    const ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2], vx = c[0] - a[0], vy = c[1] - a[1], vz = c[2] - a[2];
    let fx = uy * vz - uz * vy, fy = uz * vx - ux * vz, fz = ux * vy - uy * vx;
    const fl = Math.hypot(fx, fy, fz);
    if (fl < 1e-10) return;
    fx /= fl; fy /= fl; fz /= fl;
    if (na) {
      if (fx * (na[0] + nb[0] + nc[0]) + fy * (na[1] + nb[1] + nc[1]) + fz * (na[2] + nb[2] + nc[2]) < 0) { [b, c] = [c, b]; [nb, nc] = [nc, nb]; }
      this._v(a, na, m); this._v(b, nb, m); this._v(c, nc, m);
    } else { const n = [fx, fy, fz]; this._v(a, n, m); this._v(b, n, m); this._v(c, n, m); }
  }
  /** Quad a-b-c-d with an outward normal hint n (flat). */
  quad(a, b, c, d, m, n) {
    if (n) { this.tri(a, b, c, m, n, n, n); this.tri(a, c, d, m, n, n, n); } else { this.tri(a, b, c, m); this.tri(a, c, d, m); }
  }
  /** Add a THREE geometry (indexed or not) transformed by matrix; mat may be a function(triIndex, centroid) -> material. */
  geo(g, mat, matrix) {
    const pos = g.attributes.position, nor = g.attributes.normal, idx = g.index;
    const M = matrix || new THREE.Matrix4(), NM = new THREE.Matrix3().getNormalMatrix(M);
    const v = new THREE.Vector3(), n = new THREE.Vector3();
    const cnt = idx ? idx.count : pos.count;
    const P = [], Nn = [];
    for (let i = 0; i < cnt; i++) {
      const k = idx ? idx.getX(i) : i;
      v.fromBufferAttribute(pos, k).applyMatrix4(M); P.push([v.x, v.y, v.z]);
      if (nor) { n.fromBufferAttribute(nor, k).applyMatrix3(NM).normalize(); Nn.push([n.x, n.y, n.z]); }
    }
    for (let i = 0; i + 2 < cnt; i += 3) {
      const m = typeof mat === 'function' ? mat(i / 3, P[i], P[i + 1], P[i + 2]) : mat;
      if (!m) continue;
      if (nor) this.tri(P[i], P[i + 1], P[i + 2], m, Nn[i], Nn[i + 1], Nn[i + 2]);
      else this.tri(P[i], P[i + 1], P[i + 2], m);
    }
    return this;
  }
  /** Axis-aligned box (optionally rotated about Y by yaw / X by pitch) from centre + size. */
  box(cx, cy, cz, sx, sy, sz, m, { yaw = 0, pitch = 0, roll = 0, skipBottom = false } = {}) {
    const g = new THREE.BoxGeometry(sx, sy, sz);
    const M = new THREE.Matrix4().compose(new THREE.Vector3(cx, cy, cz), new THREE.Quaternion().setFromEuler(new THREE.Euler(pitch, yaw, roll, 'YXZ')), new THREE.Vector3(1, 1, 1));
    if (skipBottom) return this.geo(g, (t, a, b, c) => (t === 6 || t === 7 ? null : m), M);
    return this.geo(g, m, M);
  }
  /**
   * Lathe around the local Y axis: profile [[r, y], ...]; seg radial segments; mat(j, i) per band/segment.
   * Vertex normals from the profile (smooth around). matrix places it. A profile point with r = 0 closes a fan.
   */
  lathe(profile, seg, mat, matrix, phase = 0) {
    const M = matrix || new THREE.Matrix4(), NM = new THREE.Matrix3().getNormalMatrix(M);
    const n2 = [];
    for (let j = 0; j < profile.length - 1; j++) {
      const dr = profile[j + 1][0] - profile[j][0], dy = profile[j + 1][1] - profile[j][1], l = Math.hypot(dr, dy) || 1;
      n2.push([dy / l, -dr / l]);
    }
    const pn = profile.map((_, j) => {
      const a = n2[Math.max(0, j - 1)], b = n2[Math.min(n2.length - 1, j)];
      const x = a[0] + b[0], y = a[1] + b[1], l = Math.hypot(x, y) || 1;
      return [x / l, y / l];
    });
    const v = new THREE.Vector3(), nn = new THREE.Vector3();
    const P = (j, i, segNorm) => {
      const ph = phase + (i / seg) * Math.PI * 2, c = Math.cos(ph), s = Math.sin(ph);
      const [r, y] = profile[j];
      v.set(r * c, y, r * s).applyMatrix4(M);
      const nr = segNorm ? segNorm[0] : pn[j][0], ny = segNorm ? segNorm[1] : pn[j][1];
      nn.set(nr * c, ny, nr * s).applyMatrix3(NM).normalize();
      return [[v.x, v.y, v.z], [nn.x, nn.y, nn.z]];
    };
    for (let j = 0; j < profile.length - 1; j++) {
      for (let i = 0; i < seg; i++) {
        const m = typeof mat === 'function' ? mat(j, i) : mat;
        if (!m) continue;
        const flat = m.flat ? n2[j] : null;
        const [a, na] = P(j, i, flat), [b, nb] = P(j, i + 1, flat), [c, nc] = P(j + 1, i + 1, flat), [d, nd] = P(j + 1, i, flat);
        this.tri(a, b, c, m, na, nb, nc); // zero-area triangles (fan at the axis) are skipped
        this.tri(a, c, d, m, na, nc, nd);
      }
    }
    return this;
  }
  /** Tube along a polyline of [x,y,z] points (seg sides), constant or per-point radius. */
  tube(pts, radius, seg, m, { caps = false } = {}) {
    const rad = (i) => (Array.isArray(radius) ? radius[i] : radius);
    const T = new THREE.Vector3(), Nr = new THREE.Vector3(), Bn = new THREE.Vector3(), up = new THREE.Vector3(0, 1, 0);
    const rings = [];
    for (let i = 0; i < pts.length; i++) {
      const a = pts[Math.max(0, i - 1)], b = pts[Math.min(pts.length - 1, i + 1)];
      T.set(b[0] - a[0], b[1] - a[1], b[2] - a[2]).normalize();
      const ref = Math.abs(T.dot(up)) > 0.9 ? new THREE.Vector3(1, 0, 0) : up;
      Nr.crossVectors(ref, T).normalize(); Bn.crossVectors(T, Nr).normalize();
      const ring = [];
      for (let k = 0; k < seg; k++) {
        const ph = (k / seg) * Math.PI * 2, c = Math.cos(ph), s = Math.sin(ph);
        const nx = Nr.x * c + Bn.x * s, ny = Nr.y * c + Bn.y * s, nz = Nr.z * c + Bn.z * s;
        ring.push([[pts[i][0] + nx * rad(i), pts[i][1] + ny * rad(i), pts[i][2] + nz * rad(i)], [nx, ny, nz]]);
      }
      rings.push(ring);
    }
    for (let i = 0; i < pts.length - 1; i++) for (let k = 0; k < seg; k++) {
      const [a, na] = rings[i][k], [b, nb] = rings[i][(k + 1) % seg], [c, nc] = rings[i + 1][(k + 1) % seg], [d, nd] = rings[i + 1][k];
      this.tri(a, b, c, m, na, nb, nc); this.tri(a, c, d, m, na, nc, nd);
    }
    if (caps) for (const i of [0, pts.length - 1]) {
      const a = pts[Math.max(0, i - 1)], b = pts[Math.min(pts.length - 1, i + 1)];
      const t = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], l = Math.hypot(...t) || 1, sg = i === 0 ? -1 : 1;
      const n = [t[0] / l * sg, t[1] / l * sg, t[2] / l * sg];
      for (let k = 0; k < seg; k++) this.tri(pts[i], rings[i][k][0], rings[i][(k + 1) % seg][0], m, n, n, n);
    }
    return this;
  }
  /**
   * Build a BufferGeometry. layout 'vehicle': aPaint, aLamp, aSurf(rough, metal, clearcoat);
   * 'furniture': aSurf(rough, metal, emit, paint).
   */
  build(layout = 'vehicle') {
    const n = this.P.length / 3;
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(this.P, 3));
    g.setAttribute('normal', new THREE.Float32BufferAttribute(this.N, 3));
    g.setAttribute('color', new THREE.Float32BufferAttribute(this.C, 3));
    const A = this.A;
    if (layout === 'vehicle') {
      const pa = new Float32Array(n), la = new Float32Array(n), su = new Float32Array(n * 3);
      for (let i = 0; i < n; i++) { pa[i] = A[i * 6]; la[i] = A[i * 6 + 1]; su[i * 3] = A[i * 6 + 2]; su[i * 3 + 1] = A[i * 6 + 3]; su[i * 3 + 2] = A[i * 6 + 4]; }
      g.setAttribute('aPaint', new THREE.BufferAttribute(pa, 1));
      g.setAttribute('aLamp', new THREE.BufferAttribute(la, 1));
      g.setAttribute('aSurf', new THREE.BufferAttribute(su, 3));
    } else {
      const su = new Float32Array(n * 4);
      for (let i = 0; i < n; i++) { su[i * 4] = A[i * 6 + 2]; su[i * 4 + 1] = A[i * 6 + 3]; su[i * 4 + 2] = A[i * 6 + 5]; su[i * 4 + 3] = A[i * 6]; }
      g.setAttribute('aSurf', new THREE.BufferAttribute(su, 4));
    }
    g.computeBoundingBox(); g.computeBoundingSphere();
    g.userData.tris = n / 3;
    return g;
  }
}

// ------------------------------------------------------------------ vertex materials
export const VM = {
  paint: { c: [1, 1, 1], paint: 1, rough: 0.36, metal: 0.4, cc: 1 },
  glass: { c: [0.022, 0.027, 0.032], rough: 0.03, metal: 0.2, cc: 1 },
  trim: { c: [0.028, 0.028, 0.03], rough: 0.72, metal: 0.0, cc: 0 },
  trimGloss: { c: [0.016, 0.016, 0.018], rough: 0.3, metal: 0.1, cc: 0.7 },
  chrome: { c: [0.8, 0.8, 0.82], rough: 0.12, metal: 1, cc: 0 },
  head: { c: [0.5, 0.52, 0.55], lamp: 1, rough: 0.05, metal: 0.9, cc: 1 },
  tail: { c: [0.3, 0.012, 0.016], lamp: 2, rough: 0.12, metal: 0.1, cc: 1 },
  amber: { c: [0.55, 0.24, 0.02], lamp: 3, rough: 0.12, metal: 0.1, cc: 1 },
  sign: { c: [0.92, 0.8, 0.42], lamp: 4, rough: 0.3, metal: 0, cc: 0.5 },
  dest: { c: [0.05, 0.04, 0.02], lamp: 4, rough: 0.2, metal: 0, cc: 0.8 },
  lbRed: { c: [0.42, 0.02, 0.02], lamp: 5, rough: 0.15, metal: 0, cc: 1 },
  lbBlue: { c: [0.02, 0.05, 0.45], lamp: 6, rough: 0.15, metal: 0, cc: 1 },
  tyre: { c: [0.026, 0.026, 0.028], rough: 0.92, metal: 0 },
  rim: { c: [0.6, 0.61, 0.63], rough: 0.26, metal: 0.92 },
  rimDark: { c: [0.035, 0.035, 0.04], rough: 0.55, metal: 0.4 },
  steelRim: { c: [0.72, 0.72, 0.71], rough: 0.35, metal: 0.7 },
  hubBlack: { c: [0.05, 0.05, 0.055], rough: 0.45, metal: 0.5 },
  plate: { c: [0.78, 0.8, 0.84], rough: 0.4, metal: 0.2 },
  grille: { c: [0.03, 0.031, 0.034], rough: 0.4, metal: 0.5 },
  under: { c: [0.016, 0.016, 0.018], rough: 0.95, metal: 0 },
  liner: { c: [0.03, 0.03, 0.032], rough: 0.88, metal: 0 },
  seam: { c: [0.008, 0.008, 0.009], rough: 0.8, metal: 0, cc: 0 },
  white: { c: [0.84, 0.85, 0.84], rough: 0.3, metal: 0.05, cc: 1 },
  black: { c: [0.012, 0.012, 0.014], rough: 0.28, metal: 0.25, cc: 1 },
  skirt: { c: [0.07, 0.07, 0.075], rough: 0.6, metal: 0.1, cc: 0 },
  frame: { c: [0.09, 0.09, 0.095], rough: 0.4, metal: 0.4, cc: 0.4 },
};

// ------------------------------------------------------------------ specs
// Profiles are [fraction of length from the nose, value]; heights in metres; plan = half-width factor.
const CAR_PLAN = [[0, 0.83], [0.012, 0.9], [0.04, 0.955], [0.12, 0.99], [0.25, 1], [0.8, 1], [0.92, 0.985], [0.975, 0.95], [0.993, 0.9], [1, 0.85]];
const BASE = {
  sedan: {
    L: 4.72, W: 1.83, H: 1.45, WB: 2.8, ohF: 0.95, wheelR: 0.335, wheelW: 0.225, track: 1.58, rim: 'alloy',
    top: [[0, 0.64], [0.012, 0.74], [0.05, 0.795], [0.16, 0.85], [0.31, 0.95], [0.4, 1.24], [0.46, 1.41], [0.53, 1.45], [0.64, 1.44], [0.72, 1.37], [0.8, 1.17], [0.86, 1.03], [0.95, 1.02], [0.985, 0.97], [1, 0.84]],
    belt: [0.94, 1.02], bot: [[0, 0.4], [0.05, 0.32], [0.2, 0.3], [0.8, 0.3], [0.95, 0.33], [1, 0.42]], plan: CAR_PLAN, roofW: 0.76,
    zones: { cowl: 0.31, roofF: 0.46, roofR: 0.72, deck: 0.86 }, side: [[0.315, 0.553], [0.574, 0.79]], seams: [0.3, 0.563, 0.797],
    lights: { head: [0, 0.05], tail: [0.95, 1] },
  },
  hatchback: {
    L: 4.26, W: 1.79, H: 1.47, WB: 2.63, ohF: 0.86, wheelR: 0.32, wheelW: 0.215, track: 1.54, rim: 'alloy',
    top: [[0, 0.64], [0.012, 0.75], [0.05, 0.81], [0.16, 0.87], [0.33, 0.97], [0.42, 1.25], [0.49, 1.43], [0.56, 1.47], [0.8, 1.455], [0.86, 1.42], [0.93, 1.25], [0.975, 1.05], [0.99, 0.99], [1, 0.86]],
    belt: [0.96, 1.04], bot: [[0, 0.4], [0.05, 0.32], [0.2, 0.3], [0.8, 0.3], [0.95, 0.33], [1, 0.42]], plan: CAR_PLAN, roofW: 0.77,
    zones: { cowl: 0.33, roofF: 0.49, roofR: 0.86, deck: 0.975 }, side: [[0.335, 0.6], [0.62, 0.83]], seams: [0.32, 0.61, 0.835],
    lights: { head: [0, 0.05], tail: [0.93, 1] },
  },
  suv: {
    L: 4.6, W: 1.86, H: 1.69, WB: 2.69, ohF: 0.93, wheelR: 0.36, wheelW: 0.235, track: 1.6, rim: 'alloy', cladding: true, roofRails: true,
    top: [[0, 0.78], [0.012, 0.88], [0.05, 0.94], [0.16, 1.0], [0.3, 1.08], [0.38, 1.38], [0.445, 1.62], [0.51, 1.68], [0.88, 1.665], [0.935, 1.63], [0.975, 1.3], [0.99, 1.12], [1, 0.98]],
    belt: [1.08, 1.16], bot: [[0, 0.5], [0.05, 0.42], [0.2, 0.4], [0.8, 0.4], [0.95, 0.44], [1, 0.52]], plan: CAR_PLAN, roofW: 0.8,
    zones: { cowl: 0.3, roofF: 0.445, roofR: 0.935, deck: 0.985 }, side: [[0.305, 0.545], [0.565, 0.755], [0.772, 0.905]], seams: [0.295, 0.555, 0.762],
    lights: { head: [0, 0.05], tail: [0.935, 0.99] },
  },
  pickup: {
    L: 5.4, W: 1.91, H: 1.79, WB: 3.24, ohF: 0.9, wheelR: 0.385, wheelW: 0.25, track: 1.61, rim: 'chrome', cladding: true, bed: 0.6,
    top: [[0, 0.92], [0.012, 1.02], [0.04, 1.08], [0.13, 1.13], [0.255, 1.2], [0.32, 1.48], [0.37, 1.73], [0.42, 1.79], [0.57, 1.78], [0.59, 1.74], [0.598, 1.3], [0.6, 1.26], [1, 1.26]],
    belt: [1.2, 1.26], bot: [[0, 0.56], [0.05, 0.48], [0.2, 0.48], [0.85, 0.48], [0.95, 0.52], [1, 0.58]],
    plan: [[0, 0.86], [0.012, 0.92], [0.04, 0.965], [0.12, 0.995], [0.25, 1], [0.97, 1], [0.995, 0.985], [1, 0.97]], roofW: 0.82,
    zones: { cowl: 0.255, roofF: 0.37, roofR: 0.59, deck: 0.598 }, side: [[0.26, 0.445], [0.462, 0.575]], seams: [0.25, 0.453, 0.585],
    lights: { head: [0, 0.04], tail: [0.985, 1] },
  },
  minivan: {
    L: 5.16, W: 1.99, H: 1.74, WB: 3.0, ohF: 0.98, wheelR: 0.35, wheelW: 0.235, track: 1.7, rim: 'alloy',
    top: [[0, 0.74], [0.012, 0.84], [0.05, 0.9], [0.13, 0.96], [0.22, 1.05], [0.3, 1.38], [0.39, 1.69], [0.45, 1.74], [0.91, 1.73], [0.955, 1.68], [0.985, 1.25], [0.995, 1.08], [1, 0.96]],
    belt: [1.05, 1.13], bot: [[0, 0.46], [0.05, 0.38], [0.2, 0.36], [0.8, 0.36], [0.95, 0.4], [1, 0.48]], plan: CAR_PLAN, roofW: 0.8,
    zones: { cowl: 0.22, roofF: 0.39, roofR: 0.955, deck: 0.988 }, side: [[0.225, 0.39], [0.405, 0.655], [0.672, 0.925]], seams: [0.215, 0.398, 0.665],
    lights: { head: [0, 0.05], tail: [0.955, 0.99] },
  },
  police: {
    L: 5.05, W: 2.0, H: 1.78, WB: 3.03, ohF: 0.98, wheelR: 0.38, wheelW: 0.245, track: 1.7, rim: 'steel', lightbar: true, pushBar: true,
    top: [[0, 0.82], [0.012, 0.93], [0.05, 0.99], [0.16, 1.05], [0.28, 1.12], [0.36, 1.42], [0.43, 1.72], [0.49, 1.78], [0.9, 1.765], [0.945, 1.73], [0.98, 1.32], [0.992, 1.14], [1, 1.0]],
    belt: [1.12, 1.2], bot: [[0, 0.52], [0.05, 0.44], [0.2, 0.42], [0.8, 0.42], [0.95, 0.46], [1, 0.54]], plan: CAR_PLAN, roofW: 0.8,
    zones: { cowl: 0.28, roofF: 0.43, roofR: 0.945, deck: 0.988 }, side: [[0.285, 0.535], [0.553, 0.74], [0.757, 0.915]], seams: [0.275, 0.545, 0.746],
    lights: { head: [0, 0.05], tail: [0.945, 0.99] }, livery: 'bw',
  },
};
BASE.taxi = { ...BASE.sedan, rim: 'steel', taxiSign: true };
BASE.police_sedan = { ...BASE.sedan, rim: 'steel', lightbar: true, pushBar: true, livery: 'bw' };
BASE.boxtruck = {
  L: 7.4, W: 2.44, H: 3.42, WB: 4.3, ohF: 1.1, wheelR: 0.4, wheelW: 0.24, wheelWR: 0.5, track: 1.74, rim: 'truck',
  cabL: 2.15, cabW: 2.04,
  top: [[0, 1.22], [0.02, 1.38], [0.05, 1.45], [0.2, 2.3], [0.26, 2.42], [0.96, 2.42], [1, 2.36]],
  belt: [1.45, 1.52], bot: [[0, 0.62], [0.05, 0.56], [1, 0.56]], plan: [[0, 0.9], [0.02, 0.955], [0.06, 1], [1, 1]], roofW: 0.9,
  zones: { cowl: 0.05, roofF: 0.205, roofR: 0.985, deck: 1.01 }, side: [[0.06, 0.56]], seams: [0.04, 0.58],
};
BASE.bus = { L: 12.2, W: 2.59, H: 3.02, WB: 7.0, ohF: 2.45, wheelR: 0.5, wheelW: 0.3, wheelWR: 0.52, track: 2.1, rim: 'truck' };

/** Public dimensions (metres). Hc = collision height incl. roof equipment. */
export const VEHICLE_SPECS = {};
for (const [k, s] of Object.entries(BASE)) {
  const extra = s.lightbar ? 0.13 : s.taxiSign ? 0.28 : k === 'bus' ? 0.24 : 0;
  VEHICLE_SPECS[k] = { L: s.L, W: s.W, H: s.H, Hc: +(s.H + extra).toFixed(2), WB: s.WB, wheelR: s.wheelR, track: s.track, ohF: s.ohF };
}
export const VEHICLE_TYPES = Object.keys(BASE);

// ------------------------------------------------------------------ helpers
/** Monotone cubic (PCHIP) interpolation through [[x, y], ...] sorted by x; clamps outside. */
function pchip(pts) {
  const n = pts.length, xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  if (n === 1) return () => ys[0];
  const h = [], d = [];
  for (let i = 0; i < n - 1; i++) { h.push(xs[i + 1] - xs[i]); d.push((ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i])); }
  const m = new Array(n).fill(0);
  m[0] = d[0]; m[n - 1] = d[n - 2];
  for (let i = 1; i < n - 1; i++) {
    if (d[i - 1] * d[i] <= 0) m[i] = 0;
    else { const w1 = 2 * h[i] + h[i - 1], w2 = h[i] + 2 * h[i - 1]; m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i]); }
  }
  return (x) => {
    if (x <= xs[0]) return ys[0];
    if (x >= xs[n - 1]) return ys[n - 1];
    let i = 0; while (i < n - 2 && x > xs[i + 1]) i++;
    const t = (x - xs[i]) / h[i], t2 = t * t, t3 = t2 * t;
    return (2 * t3 - 3 * t2 + 1) * ys[i] + (t3 - 2 * t2 + t) * h[i] * m[i] + (-2 * t3 + 3 * t2) * ys[i + 1] + (t3 - t2) * h[i] * m[i + 1];
  };
}
function linear(pts) {
  return (x) => {
    if (x <= pts[0][0]) return pts[0][1];
    for (let i = 0; i < pts.length - 1; i++) if (x <= pts[i + 1][0]) return lerp(pts[i][1], pts[i + 1][1], (x - pts[i][0]) / (pts[i + 1][0] - pts[i][0]));
    return pts[pts.length - 1][1];
  };
}
function stationList(raw, z0, z1, maxGap, minGap = 0.025) {
  const zs = raw.filter((z) => z >= z0 - 1e-6 && z <= z1 + 1e-6).map((z) => clamp(z, z0, z1)).sort((a, b) => a - b);
  zs.unshift(z0); zs.push(z1);
  const out = [];
  for (const z of zs) if (!out.length || z - out[out.length - 1] > minGap) out.push(z); else if (z === z1) out[out.length - 1] = z1;
  const res = [out[0]];
  for (let i = 1; i < out.length; i++) {
    const g = out[i] - out[i - 1], k = Math.ceil(g / maxGap);
    for (let j = 1; j < k; j++) res.push(out[i - 1] + (g * j) / k);
    res.push(out[i]);
  }
  return res;
}

/**
 * Loft cross-sections (half sections, bottom-outer -> top-centre, mirrored) along stations zs.
 * classify(band, zMid, kMin, i) -> material. Smooth vertex normals across the skin. Returns the rings.
 */
function loft(B, zs, sec, classify, { bandIds = null, pick = null } = {}) {
  const S = zs.length;
  const secs = zs.map((z) => sec(z));
  if (pick) for (const s of secs) s.pts = pick.map((j) => s.pts[j]);
  const n = secs[0].pts.length, R = 2 * n - 1;
  const G = secs.map((s, i) => {
    const r = [];
    for (let j = 0; j < n; j++) r.push([s.pts[j][0], s.pts[j][1], zs[i]]);
    for (let j = n - 2; j >= 0; j--) r.push([-s.pts[j][0], s.pts[j][1], zs[i]]);
    return r;
  });
  const NN = G.map((r) => r.map(() => [0, 0, 0]));
  const acc = (i, k, f) => { const a = NN[i][k]; a[0] += f[0]; a[1] += f[1]; a[2] += f[2]; };
  const fnorm = (a, b, c) => {
    const ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2], vx = c[0] - a[0], vy = c[1] - a[1], vz = c[2] - a[2];
    return [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx];
  };
  for (let i = 0; i < S - 1; i++) for (let k = 0; k < R - 1; k++) {
    const a = G[i][k], b = G[i][k + 1], c = G[i + 1][k + 1], d = G[i + 1][k];
    const f1 = fnorm(a, b, c), f2 = fnorm(a, c, d);
    acc(i, k, f1); acc(i, k + 1, f1); acc(i + 1, k + 1, f1);
    acc(i, k, f2); acc(i + 1, k + 1, f2); acc(i + 1, k, f2);
  }
  for (const r of NN) for (const v of r) { const l = Math.hypot(v[0], v[1], v[2]) || 1; v[0] /= l; v[1] /= l; v[2] /= l; }
  for (let i = 0; i < S - 1; i++) {
    const zm = (zs[i] + zs[i + 1]) * 0.5, kmin = Math.min(secs[i].k ?? 1, secs[i + 1].k ?? 1);
    for (let k = 0; k < R - 1; k++) {
      const hb = k < n - 1 ? k : R - 2 - k;
      const m = classify(bandIds ? bandIds[hb] : hb, zm, kmin, i, k < n - 1 ? 1 : -1);
      if (!m) continue;
      const a = G[i][k], b = G[i][k + 1], c = G[i + 1][k + 1], d = G[i + 1][k];
      // winding a-b-c / a-c-d is outward by construction (du x dv); pass normals for smooth shading
      pushTri(B, a, b, c, m, NN[i][k], NN[i][k + 1], NN[i + 1][k + 1]);
      pushTri(B, a, c, d, m, NN[i][k], NN[i + 1][k + 1], NN[i + 1][k]);
    }
  }
  return { G, secs, n };
}
function pushTri(B, a, b, c, m, na, nb, nc) {
  const ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2], vx = c[0] - a[0], vy = c[1] - a[1], vz = c[2] - a[2];
  if (Math.hypot(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx) < 1e-9) return;
  B._v(a, na, m); B._v(b, nb, m); B._v(c, nc, m);
}
/** Planar cap over a ring (earcut), facing +/-Z. mat may be a function(centroidY) -> material. */
function cap(B, ring, z, dir, mat) {
  const pts = [];
  for (const p of ring) { const l = pts[pts.length - 1]; if (!l || Math.hypot(l[0] - p[0], l[1] - p[1]) > 1e-4) pts.push(p); }
  if (pts.length > 2 && Math.hypot(pts[0][0] - pts[pts.length - 1][0], pts[0][1] - pts[pts.length - 1][1]) < 1e-4) pts.pop();
  if (pts.length < 3) return;
  const faces = THREE.ShapeUtils.triangulateShape(pts.map((p) => new THREE.Vector2(p[0], p[1])), []);
  const n = [0, 0, dir];
  for (const f of faces) {
    const a = [pts[f[0]][0], pts[f[0]][1], z], b = [pts[f[1]][0], pts[f[1]][1], z], c = [pts[f[2]][0], pts[f[2]][1], z];
    const m = typeof mat === 'function' ? mat((a[1] + b[1] + c[1]) / 3, (a[0] + b[0] + c[0]) / 3) : mat;
    if (m) B.tri(a, b, c, m, n, n, n);
  }
}
/** Wheel: lathe tyre + rim around the X axis at (x, r, z); side = +1 right / -1 left. */
function wheel(B, x, z, r, w, side, style, lod) {
  const seg = lod ? 6 : 10;
  const prof = lod
    ? [[r, -0.45 * w], [r, 0.4 * w], [0.66 * r, 0.5 * w], [0, 0.44 * w]]
    : [[r, -0.45 * w], [r, 0.38 * w], [0.9 * r, 0.5 * w], [0.64 * r, 0.47 * w], [0.2 * r, 0.4 * w], [0, 0.42 * w]];
  const rimM = style === 'chrome' ? VM.chrome : style === 'steel' ? VM.steelRim : style === 'truck' ? VM.steelRim : VM.rim;
  const spokes = style === 'steel' ? 0 : style === 'truck' ? 2 : 1;
  const mat = (j, i) => {
    if (lod) return j < 2 ? VM.tyre : rimM;
    if (j < 3) return VM.tyre;
    if (j === 3) return spokes === 1 ? (i % 2 ? VM.rimDark : rimM) : spokes === 2 ? (i % 3 === 0 ? VM.rimDark : rimM) : (style === 'steel' ? VM.hubBlack : rimM);
    return style === 'steel' ? VM.steelRim : rimM;
  };
  const M = new THREE.Matrix4().makeRotationZ(side > 0 ? -Math.PI / 2 : Math.PI / 2).setPosition(x, r, z);
  B.lathe(prof, seg, mat, M, 0);
}

/** Wheel-well liner: inner wall + arch roof quads (stop the eye seeing through the arch). */
function archLiner(B, sd, xIn, xOut, za, rA, top) {
  const z0 = za - rA * 0.95, z1 = za + rA * 0.95, y0 = 0.12;
  B.quad([sd * xIn, y0, z0], [sd * xIn, y0, z1], [sd * xIn, top, z1], [sd * xIn, top, z0], VM.under, [sd, 0, 0]);
  B.quad([sd * xIn, top, z0], [sd * xIn, top, z1], [sd * xOut, top, z1], [sd * xOut, top, z0], VM.under, [0, -1, 0]);
}

// ------------------------------------------------------------------ cars (sedan, hatch, SUV, pickup, minivan, taxi, police)
function buildCar(B, s, lod) {
  const L = s.L, z0 = -L / 2, z1 = L / 2, hwMax = s.W / 2;
  const F = (f) => z0 + f * L, fr = (z) => (z - z0) / L;
  const topF = pchip(s.top), botF = linear(s.bot), planF = pchip(s.plan);
  const Z = s.zones;
  const zF = z0 + s.ohF, zR = zF + s.WB, rA = s.wheelR + 0.055;
  const archY = (z) => {
    let y = -1;
    for (const za of [zF, zR]) { const d = z - za; if (Math.abs(d) < rA) y = Math.max(y, s.wheelR + Math.sqrt(rA * rA - d * d)); }
    return y;
  };
  const beltF = (f) => lerp(s.belt[0], s.belt[1], clamp((f - Z.cowl) / (Z.deck - Z.cowl), 0, 1));
  const sec = (z) => {
    const f = fr(z), top = topF(f), belt = beltF(f);
    const yb = Math.max(botF(f), archY(z));
    let ym = Math.min(top, belt), yt = Math.max(top, ym);
    if (f < Z.cowl - 0.005 || f > Z.deck + 0.003) { ym = top; yt = top; }
    const k = sstep(0.015, 0.28, yt - ym);
    const hw = planF(f) * hwMax, hwT = hw * s.roofW, H = Math.max(0.02, ym - yb);
    const p = [[hw * 0.93, yb], [hw * 0.995, yb + 0.34 * H], [hw, yb + 0.74 * H], [hw * 0.955, ym - 0.012 * Math.min(1, H / 0.1)]];
    if (s.bed && f >= s.bed - 1e-4) {
      p.push([hw * 0.905, ym], [hw * 0.875, ym - 0.42], [hw * 0.6, ym - 0.44], [0, ym - 0.44]);
    } else {
      const wbX = lerp(hw * 0.7, lerp(hw * 0.92, hwT, 0.22), k), wbY = lerp(ym + 0.012, ym + 0.02, k);
      const wtX = lerp(hw * 0.45, hwT, k), wtY = lerp(ym + 0.022, yt - 0.06, k);
      const reX = Math.max(0.05, wtX - lerp(hw * 0.2, 0.09, k)), reY = lerp(ym + 0.028, yt - 0.012, k);
      p.push([wbX, wbY], [wtX, wtY], [reX, reY], [0, lerp(ym + 0.032, yt + 0.02, k)]);
    }
    return { pts: p, k, f, hw, ym, yt, yb };
  };
  // stations
  const raw = [];
  for (const k of ['cowl', 'roofF', 'roofR', 'deck']) raw.push(F(Z[k]));
  if (!lod) { for (const [f] of s.top) raw.push(F(f)); for (const [a, b] of s.side) raw.push(F(a), F(b)); raw.push(F(0.004), F(0.996)); }
  else { raw.push(F(s.lights ? s.lights.head[1] : 0.04), F(0.97)); if (s.side.length > 1) raw.push(F(s.side[0][1])); }
  if (s.bed) raw.push(F(s.bed), F(s.bed - 0.002), F(Z.roofR));
  const archT = lod ? [-1, -0.5, 0.5, 1] : [-1, -0.85, -0.4, 0.4, 0.85, 1];
  for (const za of [zF, zR]) for (const t of archT) raw.push(za + t * rA);
  const zs = stationList(raw, z0, z1, lod ? 1.6 : 0.7, lod ? 0.06 : 0.045);

  const inSide = (f) => { // 1 glass, 2 pillar between windows, 0 none
    for (const [a, b] of s.side) if (f >= a && f <= b) return 1;
    if (f > s.side[0][0] && f < s.side[s.side.length - 1][1]) return 2;
    return 0;
  };
  const bumpF = fr(zF - rA - 0.02), bumpR = fr(zR + rA + 0.02);
  const liv = s.livery === 'bw';
  const livWhite = (band, f, region) => (band <= 3 && f > s.seams[0] && f < s.seams[2]) || ((band === 5 || band === 6) && region === 'roof');
  const classify = (band, zm, kmin, i, side) => {
    const f = fr(zm);
    const region = f < Z.cowl ? 'nose' : f < Z.roofF ? 'ws' : f < Z.roofR ? 'roof' : f < Z.deck ? 'rw' : 'tail';
    let m = VM.paint;
    if (band === 0) { if (s.cladding || f < bumpF || f > bumpR) m = VM.trim; }
    else if (band === 3) { if (kmin > 0.5) m = VM.trimGloss; }
    else if (band === 4) { if (kmin > 0.15 && region !== 'nose' && region !== 'tail') { const g = inSide(f); m = g === 1 ? VM.glass : g === 2 ? VM.trimGloss : VM.paint; } }
    else if (band === 6) { if ((region === 'ws' || region === 'rw') && kmin > 0.03) m = VM.glass; }
    if (s.bed && f > s.bed - 0.004 && band >= 4) m = VM.liner;
    // wrap-around lamps on the nose / tail strips
    if (s.lights && !(s.bed && f > s.bed)) {
      const hl = s.lights.head, tl = s.lights.tail;
      if (f >= hl[0] && f <= hl[1] && (band === 2 || band === 3)) m = VM.head;
      if (f >= hl[0] && f <= hl[1] * 0.3 && band === 1) m = VM.amber;
      if (f >= tl[0] && f <= tl[1] && (band === 2 || band === 3)) m = VM.tail;
    }
    if (liv && m === VM.paint) m = livWhite(band, f, region) ? VM.white : VM.black;
    return m;
  };
  const { G, secs } = loft(B, zs, sec, classify, lod ? { pick: [0, 2, 3, 5, 7], bandIds: [0, 2, 4, 6] } : {});
  // caps: front fascia / rear panel
  const fascia = liv ? VM.black : VM.paint;
  cap(B, G[0].map((p) => [p[0], p[1]]), z0, -1, (y) => (y < secs[0].yb + 0.1 ? VM.trim : fascia));
  const S = G.length - 1;
  cap(B, G[S].map((p) => [p[0], p[1]]), z1, 1, (y) => (y < secs[S].yb + 0.1 ? VM.trim : fascia));
  if (s.bed) cap(B, G[S].map((p) => [p[0], p[1]]), z1 - 0.04, -1, (y) => (y > secs[S].ym - 0.46 ? VM.liner : null));
  // underbody + wheels
  const hwm = hwMax * 0.98, sill = s.bot[2][1], uy0 = 0.16, uy1 = sill + 0.06;
  B.box(0, (uy0 + uy1) / 2, (zF + zR) / 2, s.track - s.wheelW - 0.1, uy1 - uy0, s.WB + 2 * rA, VM.under);
  for (const za of [zF, zR]) {
    for (const sd of [-1, 1]) wheel(B, sd * (s.track / 2), za, s.wheelR, s.wheelW, sd, s.rim, lod);
    if (!lod) for (const sd of [-1, 1]) archLiner(B, sd, s.track / 2 - s.wheelW / 2 - 0.05, hwm - 0.02, za, rA, s.wheelR + rA - 0.01);
  }
  // fascia details
  const y0 = secs[0].yb, yTopF = topF(0);
  const fw = planF(0) * hwMax;
  if (!lod) {
    const gy = lerp(y0, yTopF, 0.62);
    B.box(0, gy, z0 - 0.006, Math.min(0.75, fw * 0.85), (yTopF - y0) * 0.36, 0.03, VM.grille);
    B.box(0, lerp(y0, yTopF, 0.17), z0 - 0.004, fw * 1.25, (yTopF - y0) * 0.2, 0.03, VM.trim);
    B.box(0, lerp(y0, yTopF, 0.3), z0 - 0.012, 0.305, 0.152, 0.012, VM.plate);
    const yr = lerp(secs[S].yb, topF(1), 0.62);
    B.box(0, yr, z1 + 0.008, 0.305, 0.152, 0.012, VM.plate);
    // mirrors
    const fm = Z.cowl + 0.035, zm = F(fm), hm = planF(fm) * hwMax, ym = beltF(fm) + 0.07;
    for (const sd of [-1, 1]) {
      B.box(sd * (hm + 0.03), ym - 0.03, zm + 0.04, 0.08, 0.05, 0.11, VM.trim);
      B.box(sd * (hm + 0.11), ym + 0.02, zm + 0.05, 0.17, 0.115, 0.085, liv ? VM.black : VM.paint, { yaw: sd * 0.12 });
    }
    // door seams: thin dark strips over the lower side bands
    for (const fs of s.seams) {
      const zsm = F(fs), sc = sec(zsm), p = sc.pts;
      for (const sd of [-1, 1]) for (let j = 0; j < 3; j++) {
        const a = p[j], b = p[j + 1];
        const dx = b[0] - a[0], dy = b[1] - a[1], l = Math.hypot(dx, dy) || 1, nx = dy / l, ny = -dx / l;
        const o = 0.004, w = 0.006;
        const A = [sd * (a[0] + nx * o), a[1] + ny * o], Bp = [sd * (b[0] + nx * o), b[1] + ny * o];
        B.quad([A[0], A[1], zsm - w], [Bp[0], Bp[1], zsm - w], [Bp[0], Bp[1], zsm + w], [A[0], A[1], zsm + w], VM.seam, [sd * nx, ny, 0]);
      }
    }
  }
  // roof equipment
  const roofY = (f) => topF(f) + 0.02;
  if (s.roofRails && !lod) {
    const a = Z.roofF + 0.04, b = Z.roofR - 0.02, fm = (a + b) / 2, hwT = planF(fm) * hwMax * s.roofW * 0.86;
    for (const sd of [-1, 1]) B.box(sd * hwT, roofY(fm) + 0.035, F(fm), 0.04, 0.05, (b - a) * L, VM.trimGloss);
  }
  if (s.lightbar) {
    const fm = (Z.roofF + Z.roofR) / 2 - 0.04, y = roofY(fm), zc = F(fm);
    B.box(0, y + 0.03, zc, 1.2, 0.05, 0.28, VM.trim);
    B.box(-0.3, y + 0.09, zc, 0.56, 0.07, 0.24, VM.lbRed);
    B.box(0.3, y + 0.09, zc, 0.56, 0.07, 0.24, VM.lbBlue);
  }
  if (s.pushBar) {
    B.box(0, lerp(y0, yTopF, 0.45), z0 - 0.12, fw * 1.3, 0.34, 0.06, VM.trim);
    if (!lod) for (const sd of [-1, 1]) B.box(sd * 0.32, lerp(y0, yTopF, 0.5), z0 - 0.07, 0.06, 0.42, 0.12, VM.trim);
  }
  if (s.bed) for (const sd of [-1, 1]) B.box(sd * (planF(1) * hwMax - 0.05), 1.0, z1 + 0.004, 0.1, 0.36, 0.03, VM.tail);
  if (s.taxiSign) {
    const fm = (Z.roofF + Z.roofR) / 2, y = roofY(fm), zc = F(fm);
    B.box(0, y + 0.02, zc, 0.5, 0.04, 0.2, VM.trim);
    B.box(0, y + 0.14, zc, 0.62, 0.2, 0.2, VM.sign);
  }
}

// ------------------------------------------------------------------ box truck (cab-over cab + rounded box body)
function roundedBox(B, x0, x1, y0, y1, za, zb, r, m, { endA = m, endB = m, seg = 3 } = {}) {
  const prof = [];
  const cx = [x1 - r, x0 + r], cy = [y0 + r, y1 - r];
  const corner = (px, py, a0) => { for (let i = 0; i <= seg; i++) { const a = a0 + (i / seg) * Math.PI / 2; prof.push([px + Math.cos(a) * r, py + Math.sin(a) * r, Math.cos(a), Math.sin(a)]); } };
  corner(cx[0], cy[0], -Math.PI / 2); corner(cx[0], cy[1], 0); corner(cx[1], cy[1], Math.PI / 2); corner(cx[1], cy[0], Math.PI);
  for (let i = 0; i < prof.length; i++) {
    const p = prof[i], q = prof[(i + 1) % prof.length];
    const a = [p[0], p[1], za], b = [q[0], q[1], za], c = [q[0], q[1], zb], d = [p[0], p[1], zb];
    const na = [p[2], p[3], 0], nb = [q[2], q[3], 0];
    B.tri(a, b, c, m, na, nb, nb); B.tri(a, c, d, m, na, nb, na);
  }
  const ctr = [(x0 + x1) / 2, (y0 + y1) / 2];
  for (const [z, dir, mm] of [[za, -1, endA], [zb, 1, endB]]) {
    if (!mm) continue;
    const n = [0, 0, dir];
    for (let i = 0; i < prof.length; i++) {
      const p = prof[i], q = prof[(i + 1) % prof.length];
      B.tri([ctr[0], ctr[1], z], [p[0], p[1], z], [q[0], q[1], z], mm, n, n, n);
    }
  }
}
function buildBoxTruck(B, s, lod) {
  const L = s.L, z0 = -L / 2, z1 = L / 2, Lc = s.cabL, hwc = s.cabW / 2;
  const zF = z0 + s.ohF, zR = zF + s.WB, rA = s.wheelR + 0.06;
  // cab: reuse the car loft with a cab-only spec
  const cab = { ...s, L: Lc, W: s.cabW, ohF: s.ohF, WB: 99, lights: null, seams: s.seams, cladding: false };
  const topF = pchip(s.top), botF = linear(s.bot), planF = pchip(s.plan), Z = s.zones;
  const F = (g) => z0 + g * Lc, gr = (z) => (z - z0) / Lc;
  const archY = (z) => { const d = z - zF; return Math.abs(d) < rA ? s.wheelR + Math.sqrt(rA * rA - d * d) : -1; };
  const sec = (z) => {
    const g = gr(z), top = topF(g), belt = lerp(s.belt[0], s.belt[1], g);
    const yb = Math.max(botF(g), archY(z));
    let ym = Math.min(top, belt), yt = Math.max(top, ym);
    if (g < Z.cowl - 0.005) { ym = top; yt = top; }
    const k = sstep(0.015, 0.3, yt - ym);
    const hw = planF(g) * hwc, hwT = hw * s.roofW, H = Math.max(0.02, ym - yb);
    const p = [[hw * 0.96, yb], [hw * 0.995, yb + 0.34 * H], [hw, yb + 0.74 * H], [hw * 0.985, ym - 0.01]];
    const wbX = lerp(hw * 0.75, lerp(hw * 0.97, hwT, 0.3), k), wbY = lerp(ym + 0.01, ym + 0.02, k);
    const wtX = lerp(hw * 0.5, hwT, k), wtY = lerp(ym + 0.02, yt - 0.07, k);
    const reX = Math.max(0.05, wtX - lerp(hw * 0.2, 0.1, k)), reY = lerp(ym + 0.026, yt - 0.012, k);
    p.push([wbX, wbY], [wtX, wtY], [reX, reY], [0, lerp(ym + 0.03, yt + 0.015, k)]);
    return { pts: p, k, yb, ym, yt };
  };
  const raw = [];
  for (const k of ['cowl', 'roofF', 'roofR']) raw.push(F(Z[k]));
  if (!lod) { for (const [g] of s.top) raw.push(F(g)); for (const [a, b] of s.side) raw.push(F(a), F(b)); }
  for (const t of (lod ? [-1, -0.5, 0.5, 1] : [-1, -0.85, -0.4, 0.4, 0.85, 1])) raw.push(zF + t * rA);
  const zs = stationList(raw, z0, z0 + Lc, lod ? 1.2 : 0.6, lod ? 0.06 : 0.04);
  const classify = (band, zm, kmin) => {
    const g = gr(zm);
    const region = g < Z.cowl ? 'nose' : g < Z.roofF ? 'ws' : 'roof';
    if (band === 0) return VM.trim;
    if (band === 3 && kmin > 0.5) return VM.trimGloss;
    if (band === 4 && kmin > 0.15 && region !== 'nose') { for (const [a, b] of s.side) if (g >= a && g <= b) return VM.glass; }
    if (band === 6 && region === 'ws' && kmin > 0.03) return VM.glass;
    return VM.paint;
  };
  void cab;
  const { G, secs } = loft(B, zs, sec, classify, lod ? { pick: [0, 2, 3, 5, 7], bandIds: [0, 2, 4, 6] } : {});
  cap(B, G[0].map((p) => [p[0], p[1]]), z0, -1, (y) => (y < secs[0].yb + 0.12 ? VM.trim : VM.paint));
  const S = G.length - 1;
  cap(B, G[S].map((p) => [p[0], p[1]]), z0 + Lc, 1, VM.trim);
  // grille + bumper + lamps on the cab front
  B.box(0, 0.95, z0 - 0.01, 1.2, 0.34, 0.03, VM.grille);
  B.box(0, 0.62, z0 - 0.06, s.cabW + 0.04, 0.24, 0.14, VM.trim);
  for (const sd of [-1, 1]) {
    B.box(sd * (hwc - 0.24), 1.02, z0 - 0.012, 0.3, 0.16, 0.03, VM.head);
    B.box(sd * (hwc - 0.04), 1.05, z0 + 0.06, 0.04, 0.12, 0.14, VM.amber);
    if (!lod) { // west-coast mirrors on arms
      B.box(sd * (hwc + 0.12), 1.9, z0 + 0.45, 0.22, 0.03, 0.03, VM.frame);
      B.box(sd * (hwc + 0.24), 1.85, z0 + 0.45, 0.05, 0.42, 0.18, VM.trim);
    }
  }
  if (!lod) B.box(0, 0.62, z0 - 0.141, 0.305, 0.152, 0.01, VM.plate);
  // box body
  const bz0 = z0 + Lc + 0.1, bz1 = z1 - 0.02, hwb = s.W / 2, by0 = 1.04, by1 = s.H;
  roundedBox(B, -hwb, hwb, by0, by1, bz0, bz1, 0.07, VM.paint, { seg: lod ? 1 : 3, endB: VM.paint });
  // rear: roll-up door frame, slats, lamps, bumper, plate
  if (!lod) {
    for (let k = 1; k < 9; k++) { const y = by0 + 0.08 + k * ((by1 - by0 - 0.2) / 9); B.quad([-hwb + 0.12, y, bz1 + 0.004], [hwb - 0.12, y, bz1 + 0.004], [hwb - 0.12, y + 0.012, bz1 + 0.004], [-hwb + 0.12, y + 0.012, bz1 + 0.004], VM.seam, [0, 0, 1]); }
    for (const sd of [-1, 1]) B.box(sd * (hwb - 0.06), (by0 + by1) / 2, bz1 + 0.01, 0.08, by1 - by0 - 0.06, 0.02, VM.frame);
    B.box(0, by1 - 0.06, bz1 + 0.01, s.W - 0.04, 0.08, 0.02, VM.frame);
    B.box(0, 1.25, bz1 + 0.013, 0.305, 0.152, 0.01, VM.plate);
  }
  for (const sd of [-1, 1]) {
    B.box(sd * (hwb - 0.2), 1.12, bz1 + 0.012, 0.22, 0.12, 0.02, VM.tail);
    B.box(sd * (hwb - 0.01), by1 - 0.08, bz1 - 0.05, 0.03, 0.06, 0.06, VM.amber);
  }
  B.box(0, 0.62, bz1 - 0.05, s.W - 0.3, 0.12, 0.1, VM.trim);
  // chassis: rails, fuel tank, steps, mudflaps, under-box shadow block
  B.box(0, 0.8, (z0 + Lc + z1) / 2 - 0.2, 0.95, 0.24, z1 - z0 - Lc - 0.2, VM.under);
  B.box(0, 0.97, (bz0 + bz1) / 2, s.W - 0.1, 0.12, bz1 - bz0 - 0.1, VM.trim);
  if (!lod) {
    const tank = new THREE.CylinderGeometry(0.24, 0.24, 0.9, 10, 1).rotateX(Math.PI / 2);
    B.geo(tank, VM.chrome, new THREE.Matrix4().setPosition(-0.75, 0.62, zF + 1.5));
    for (const sd of [-1, 1]) B.box(sd * (hwb - 0.15), 0.62, zR + s.wheelR + 0.2, 0.36, 0.5, 0.03, VM.trim);
  }
  // wheels: single front, wide (dual-look) rear
  for (const sd of [-1, 1]) {
    wheel(B, sd * (s.track / 2), zF, s.wheelR, s.wheelW, sd, 'truck', lod);
    wheel(B, sd * (s.track / 2 + 0.06), zR, s.wheelR, s.wheelWR, sd, 'truck', lod);
  }
}

/** Bus livery (vertex materials): lower body band + window-line stripe. Games pass their own (e.g. a transit agency's). */
export const DEFAULT_BUS_LIVERY = {
  lower: { c: [0.36, 0.37, 0.39], rough: 0.32, metal: 0.55, cc: 1 },
  stripe: { c: [0.03, 0.13, 0.42], rough: 0.32, metal: 0.05, cc: 1 },
};

// ------------------------------------------------------------------ city bus (40 ft low-floor)
function buildBus(B, s, lod, livery = DEFAULT_BUS_LIVERY) {
  const LOWER = livery.lower || DEFAULT_BUS_LIVERY.lower, STRIPE = livery.stripe || DEFAULT_BUS_LIVERY.stripe;
  const L = s.L, z0 = -L / 2, z1 = L / 2, hwM = s.W / 2;
  const zF = z0 + s.ohF, zR = zF + s.WB, rA = s.wheelR + 0.06;
  const planF = pchip([[0, 0.93], [0.006, 0.975], [0.018, 1], [0.985, 1], [0.995, 0.975], [1, 0.95]]);
  const archY = (z) => { let y = -1; for (const za of [zF, zR]) { const d = z - za; if (Math.abs(d) < rA) y = Math.max(y, s.wheelR + Math.sqrt(rA * rA - d * d)); } return y; };
  const sec = (z) => {
    const f = (z - z0) / L, hw = planF(f) * hwM;
    const yb = Math.max(0.32, archY(z));
    const y1 = Math.max(yb + 0.03, 0.44), y2 = Math.max(y1 + 0.03, 0.98), y3 = Math.max(y2 + 0.03, 1.2);
    const p = [[hw * 0.985, yb], [hw, y1], [hw, y2], [hw, y3], [hw * 0.993, 2.36], [hw * 0.958, 2.8], [hw * 0.83, 2.975], [0, 3.01]];
    return { pts: p, k: 1, yb };
  };
  const raw = lod ? [z0 + 0.1, z1 - 0.1] : [z0 + 0.03, z0 + 0.1, z0 + 0.25, z1 - 0.25, z1 - 0.1, z1 - 0.03];
  for (const za of [zF, zR]) for (const t of (lod ? [-1, -0.5, 0.5, 1] : [-1, -0.8, -0.4, 0.4, 0.8, 1])) raw.push(za + t * rA);
  const zs = stationList(raw, z0, z1, lod ? 4 : 2.5, 0.03);
  const classify = (band, zm) => {
    const f = (zm - z0) / L;
    if (band === 0) return VM.skirt;
    if (band === 1) return LOWER;
    if (band === 2) return STRIPE;
    if (band === 3) return f > 0.004 && f < 0.975 ? VM.glass : VM.white;
    return VM.white;
  };
  const { G } = loft(B, zs, sec, classify, lod ? { pick: [0, 2, 3, 4, 7], bandIds: [1, 2, 3, 4] } : {});
  cap(B, G[0].map((p) => [p[0], p[1]]), z0, -1, (y) => (y < 0.6 ? VM.skirt : VM.white));
  cap(B, G[G.length - 1].map((p) => [p[0], p[1]]), z1, 1, (y) => (y < 0.6 ? VM.skirt : y < 0.98 ? LOWER : VM.white));
  const fw = hwM * 0.93;
  // front: windscreen, destination sign, lamps, bumper, bike rack
  const zf = z0 - 0.006;
  B.quad([-fw + 0.08, 1.0, zf], [fw - 0.08, 1.0, zf], [fw - 0.1, 2.55, zf], [-fw + 0.1, 2.55, zf], VM.glass, [0, 0, -1]);
  B.quad([-fw + 0.22, 2.6, zf], [fw - 0.22, 2.6, zf], [fw - 0.22, 2.86, zf], [-fw + 0.22, 2.86, zf], VM.dest, [0, 0, -1]);
  B.box(0, 0.45, z0 - 0.06, s.W - 0.1, 0.26, 0.14, VM.trim);
  for (const sd of [-1, 1]) {
    B.box(sd * (fw - 0.26), 0.74, z0 - 0.01, 0.34, 0.14, 0.03, VM.head);
    B.box(sd * (fw - 0.06), 0.74, z0 + 0.04, 0.04, 0.12, 0.1, VM.amber);
  }
  if (!lod) {
    B.box(0, 0.62, z0 - 0.32, 1.5, 0.04, 0.04, VM.frame);
    B.box(0, 0.98, z0 - 0.32, 1.5, 0.04, 0.04, VM.frame);
    for (const x of [-0.7, 0, 0.7]) B.box(x, 0.8, z0 - 0.22, 0.04, 0.4, 0.22, VM.frame);
    B.box(0, 0.6, z0 - 0.14, 0.305, 0.152, 0.01, VM.plate);
    for (const sd of [-1, 1]) { // mirrors on bent arms
      B.box(sd * (hwM + 0.12), 2.62, z0 + 0.2, 0.26, 0.04, 0.04, VM.frame);
      B.box(sd * (hwM + 0.24), 2.3, z0 - 0.02, 0.06, 0.55, 0.2, VM.trim);
    }
  }
  // rear: window, lamps, bumper
  const zb = z1 + 0.006;
  B.quad([-fw + 0.35, 1.9, zb], [fw - 0.35, 1.9, zb], [fw - 0.35, 2.45, zb], [-fw + 0.35, 2.45, zb], VM.glass, [0, 0, 1]);
  B.box(0, 0.45, z1 + 0.05, s.W - 0.1, 0.26, 0.12, VM.trim);
  for (const sd of [-1, 1]) {
    B.box(sd * (fw - 0.08), 1.25, z1 + 0.01, 0.16, 0.6, 0.03, VM.tail);
    B.box(sd * (fw - 0.08), 0.85, z1 + 0.01, 0.16, 0.14, 0.03, VM.amber);
  }
  // side: window pillars, doors (kerb side = +X), wheel skirts
  const xs = hwM + 0.004;
  if (!lod) {
    for (let z = zF + 0.7; z < z1 - 0.5; z += 1.42) {
      if (z > 1.45 && z < 2.85) continue; // rear door
      for (const sd of [-1, 1]) B.quad([sd * xs, 1.2, z - 0.04], [sd * xs, 1.2, z + 0.04], [sd * xs * 0.993, 2.36, z + 0.04], [sd * xs * 0.993, 2.36, z - 0.04], VM.frame, [sd, 0, 0]);
    }
  }
  for (const [za, zb2] of [[z0 + 0.3, z0 + 1.48], [1.55, 2.72]]) {
    B.quad([xs + 0.002, 0.36, za], [xs + 0.002, 0.36, zb2], [xs + 0.002, 2.5, zb2], [xs + 0.002, 2.5, za], VM.glass, [1, 0, 0]);
    if (!lod) {
      const zc = (za + zb2) / 2;
      B.quad([xs + 0.004, 0.36, zc - 0.03], [xs + 0.004, 0.36, zc + 0.03], [xs + 0.004, 2.5, zc + 0.03], [xs + 0.004, 2.5, zc - 0.03], VM.frame, [1, 0, 0]);
      B.quad([xs + 0.004, 2.46, za], [xs + 0.004, 2.46, zb2], [xs + 0.004, 2.52, zb2], [xs + 0.004, 2.52, za], VM.frame, [1, 0, 0]);
    }
  }
  // roof pods (batteries / HVAC)
  B.box(0, 3.1, 1.2, 1.9, 0.2, 3.6, VM.white);
  if (!lod) { B.box(0, 3.08, -2.6, 1.5, 0.16, 1.6, VM.white); B.box(0, 3.14, 4.6, 2.0, 0.26, 1.8, LOWER); }
  // underbody + wheels
  B.box(0, 0.3, (zF + zR) / 2, s.W - 0.3, 0.2, s.WB + 2 * rA, VM.under);
  for (const sd of [-1, 1]) {
    wheel(B, sd * (s.track / 2 + 0.04), zF, s.wheelR, s.wheelW, sd, 'truck', lod);
    wheel(B, sd * (s.track / 2 + 0.1), zR, s.wheelR, s.wheelWR, sd, 'truck', lod);
    if (!lod) for (const za of [zF, zR]) B.box(sd * (s.track / 2 - 0.25), 0.75, za, 0.04, 0.85, rA * 1.9, VM.under);
  }
}

// ------------------------------------------------------------------ public API
const _cache = new Map();
/**
 * Vehicle geometry (non-indexed, vehicle attribute layout). Returns a clone, so callers may add attributes.
 * @param {string} type one of VEHICLE_TYPES
 * @param {{lod?: 0|1, livery?: {lower?: object, stripe?: object}}} [opts] livery: bus colours (DEFAULT_BUS_LIVERY)
 */
export function makeVehicleGeometry(type, opts = {}) {
  const lod = opts.lod ? 1 : 0;
  const livery = type === 'bus' && opts.livery ? opts.livery : null;
  const key = livery ? `${type}:${lod}:${JSON.stringify(livery)}` : `${type}:${lod}`;
  let g = _cache.get(key);
  if (!g) {
    const s = BASE[type] || BASE.sedan;
    const B = new PartBuilder();
    if (type === 'bus') buildBus(B, s, lod, livery || DEFAULT_BUS_LIVERY);
    else if (type === 'boxtruck') buildBoxTruck(B, s, lod);
    else buildCar(B, s, lod);
    g = B.build('vehicle');
    g.userData.spec = VEHICLE_SPECS[type] || VEHICLE_SPECS.sedan;
    g.userData.type = type; g.userData.lod = lod;
    _cache.set(key, g);
  }
  const c = g.clone();
  c.userData = { ...g.userData };
  return c;
}

// ------------------------------------------------------------------ paint
const PALETTES = {
  // [weight, [sRGB hex variants]] - US fleet colour shares (white 24, black 20, grey 18, silver 14, blue 9, red 8, other 7)
  car: [
    [0.24, [0xeeeeea, 0xf4f3ee, 0xe1e3e2]], [0.2, [0x0a0a0c, 0x111215, 0x17181b]], [0.18, [0x4b4f54, 0x60646a, 0x383b3f, 0x707379]],
    [0.14, [0xa6a9ae, 0xb8bbc0, 0x9a9ea3]], [0.09, [0x1f3f7c, 0x2c5ca8, 0x14234a, 0x41689e]], [0.08, [0x9e1416, 0xb81e1e, 0x6d0f14]],
    [0.02, [0x2c4a33, 0x1e3b2b, 0x4d6b4a]], [0.015, [0xb9a27e, 0x9e8a6a]], [0.01, [0x5b3a28, 0x4a3326]], [0.008, [0xd1621b, 0xc4501a]],
    [0.005, [0xe2b92b]], [0.012, [0x2e7f86, 0x1f6b70]],
  ],
  taxi: [[0.7, [0xf2b705, 0xf5c518]], [0.15, [0x2e7d4f]], [0.1, [0xc8202a]], [0.05, [0xeeeeea]]],
  truck: [[0.68, [0xf2f2f0, 0xe9eae8]], [0.08, [0xc8102e]], [0.08, [0x1f4e9a]], [0.07, [0xd8d9db]], [0.05, [0xf2b705]], [0.04, [0x0f5a2c]]],
  bus: [[1, [0xffffff]]], police: [[1, [0xffffff]]],
};
/** Realistic paint colour into `out` (linear THREE.Color). rnd: () => [0,1). */
export function pickVehicleColor(rnd, kind = 'car', out = new THREE.Color()) {
  const P = PALETTES[kind] || PALETTES.car;
  let tot = 0; for (const [w] of P) tot += w;
  let r = rnd() * tot, e = P[P.length - 1];
  for (const p of P) { r -= p[0]; if (r <= 0) { e = p; break; } }
  const hex = e[1][Math.floor(rnd() * e[1].length) % e[1].length];
  out.setHex(hex);
  if (kind === 'car') out.multiplyScalar(0.94 + rnd() * 0.1);
  return out;
}

// ------------------------------------------------------------------ material
/**
 * Clearcoated paint, glass and lamps from the vertex attributes. userData.u.uDark (0..1) scales lamps.
 * @param {{brake?: boolean, lamps?: boolean}} [o] brake: per-instance `aBrake` drives brake-light intensity;
 *        lamps: false = parked cars (no lit lamps)
 */
export function createVehicleMaterial({ brake = true, lamps = true } = {}) {
  const m = new THREE.MeshPhysicalMaterial({ vertexColors: true, roughness: 0.4, metalness: 0.3, clearcoat: 1, clearcoatRoughness: 0.07 });
  const u = { uDark: { value: 0 } };
  m.onBeforeCompile = (sh) => {
    sh.uniforms.uDark = u.uDark;
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', `#include <common>
attribute float aPaint; attribute float aLamp; attribute vec3 aSurf;${brake ? '\nattribute float aBrake;' : ''}
uniform float uDark; varying vec3 vLampE; varying vec3 vSurf;`)
      .replace('#include <color_vertex>', `
vColor = vec3( 1.0 );
#ifdef USE_COLOR
vColor *= color;
#endif
vec3 tintC = vec3( 1.0 );
#ifdef USE_INSTANCING_COLOR
tintC = instanceColor.xyz;
#endif
#ifdef USE_BATCHING_COLOR
tintC = getBatchingColor( getIndirectIndex( gl_DrawID ) );
#endif
vColor = mix( vColor, vColor * tintC, aPaint );
vSurf = aSurf;
vLampE = vec3( 0.0 );
${lamps ? `float brk = ${brake ? 'aBrake' : '0.0'};
if ( aLamp > 0.5 && aLamp < 1.5 ) vLampE = vec3( 1.0, 0.93, 0.8 ) * ( 0.12 + uDark * 7.0 );
else if ( aLamp > 1.5 && aLamp < 2.5 ) vLampE = vec3( 1.0, 0.0, 0.0 ) * ( 0.03 + uDark * 0.45 + brk * ( 0.85 + uDark * 0.3 ) );
else if ( aLamp > 2.5 && aLamp < 3.5 ) vLampE = vec3( 1.0, 0.42, 0.04 ) * ( 0.02 + uDark * 0.6 );
else if ( aLamp > 3.5 && aLamp < 4.5 ) vLampE = vec3( 1.0, 0.72, 0.28 ) * ( 0.05 + uDark * 2.2 );
else if ( aLamp > 4.5 && aLamp < 5.5 ) vLampE = vec3( 1.0, 0.04, 0.04 ) * ( uDark * 0.5 );
else if ( aLamp > 5.5 ) vLampE = vec3( 0.08, 0.2, 1.0 ) * ( uDark * 0.5 );` : ''}`);
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vLampE; varying vec3 vSurf;')
      .replace('#include <roughnessmap_fragment>', 'float roughnessFactor = vSurf.x;')
      .replace('#include <metalnessmap_fragment>', 'float metalnessFactor = vSurf.y;')
      .replace('#include <lights_physical_fragment>', '#include <lights_physical_fragment>\n#ifdef USE_CLEARCOAT\nmaterial.clearcoat *= vSurf.z;\n#endif')
      .replace('#include <emissivemap_fragment>', '#include <emissivemap_fragment>\ntotalEmissiveRadiance += vLampE;');
  };
  m.customProgramCacheKey = () => `roadster-vehicle-3-${brake ? 1 : 0}${lamps ? 1 : 0}`;
  m.userData.u = u;
  return m;
}
