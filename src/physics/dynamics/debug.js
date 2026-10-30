// Debug line geometry of a dynamics world (pure JS, Node-safe, allocation-free): writes line segments into a
// caller's Float32Array, 6 floats per segment (x0 y0 z0 x1 y1 z1), and optionally per-vertex colours in a second
// Float32Array with the same layout (r g b per vertex). src/render/PhysicsDebug.js feeds a LineSegments with it.
//
//   const n = writeDebugLines(world, positions, maxLines, { aabbs: true }, colors);   // n segments written
//
// Per body (pose from bodies.pos / quat, i.e. the last step, not an interpolated render pose):
//   sphere   3 great circles (16 segments each)                                   48 lines
//   capsule  2 rings + 4 side lines + 4 cap half-circles (8 segments each)         68 lines
//   box      12 edges                                                              12 lines
//   convex   shape.edges (vertex index pairs); face loops without edges; its points' local box without either
//   trimesh  its AABB (bodies.aabb; inside a compound: its local bounds, oriented) 12 lines
//   compound its children
// contacts: normal from the contact point, 0.2 m long (1 line each); joints: a 3-axis cross at each anchor plus the
// segment between them (7 lines; broken joints are skipped); aabbs (off by default): 12 lines per body.
// Colours (colorByState): static blue, kinematic yellow, sleeping grey, awake dynamic green; contacts red, joints
// magenta, AABBs dim cyan. Writing stops at `max` lines (and at the arrays' capacity).
import { BODY, FLAG, SHAPE } from './state.js';
import { meshBounds } from './queries.js';

const SEG = 16, HALF = 8;
// unit circle table (debug only: Math.cos / sin are fine here, nothing in a step reads it)
const COS = new Float64Array(SEG + 1), SIN = new Float64Array(SEG + 1);
for (let i = 0; i <= SEG; i++) { const a = (i / SEG) * Math.PI * 2; COS[i] = Math.cos(a); SIN[i] = Math.sin(a); }
COS[SEG] = 1; SIN[SEG] = 0;

const COLOR = Object.freeze({
  static: [0.3, 0.55, 1], kinematic: [1, 0.85, 0.2], asleep: [0.5, 0.5, 0.5], awake: [0.25, 0.95, 0.35],
  plain: [0.85, 0.85, 0.85], contact: [1, 0.25, 0.2], joint: [1, 0.35, 1], aabb: [0.2, 0.6, 0.65],
});
const DEFAULTS = Object.freeze({ bodies: true, contacts: true, joints: true, aabbs: false, colorByState: true });

// writer state
let _o = null, _c = null, _n = 0, _max = 0, _r = 1, _g = 1, _b = 1;
// current transform (world = p + q * local)
let _px = 0, _py = 0, _pz = 0, _qx = 0, _qy = 0, _qz = 0, _qw = 1;
let _wx = 0, _wy = 0, _wz = 0;

function color(c) { _r = c[0]; _g = c[1]; _b = c[2]; }

function line(x0, y0, z0, x1, y1, z1) {
  if (_n >= _max) return;
  const i = _n * 6, o = _o;
  o[i] = x0; o[i + 1] = y0; o[i + 2] = z0; o[i + 3] = x1; o[i + 4] = y1; o[i + 5] = z1;
  if (_c) { const c = _c; c[i] = _r; c[i + 1] = _g; c[i + 2] = _b; c[i + 3] = _r; c[i + 4] = _g; c[i + 5] = _b; }
  _n++;
}

function xf(x, y, z) {
  const qx = _qx, qy = _qy, qz = _qz, qw = _qw;
  const tx = 2 * (qy * z - qz * y), ty = 2 * (qz * x - qx * z), tz = 2 * (qx * y - qy * x);
  _wx = _px + x + qw * tx + (qy * tz - qz * ty);
  _wy = _py + y + qw * ty + (qz * tx - qx * tz);
  _wz = _pz + z + qw * tz + (qx * ty - qy * tx);
}

// segment in the current local frame
function lineL(x0, y0, z0, x1, y1, z1) {
  if (_n >= _max) return;
  xf(x0, y0, z0); const ax = _wx, ay = _wy, az = _wz;
  xf(x1, y1, z1);
  line(ax, ay, az, _wx, _wy, _wz);
}

// box edges in the current local frame
function boxL(x0, y0, z0, x1, y1, z1) {
  lineL(x0, y0, z0, x1, y0, z0); lineL(x0, y1, z0, x1, y1, z0); lineL(x0, y0, z1, x1, y0, z1); lineL(x0, y1, z1, x1, y1, z1);
  lineL(x0, y0, z0, x0, y1, z0); lineL(x1, y0, z0, x1, y1, z0); lineL(x0, y0, z1, x0, y1, z1); lineL(x1, y0, z1, x1, y1, z1);
  lineL(x0, y0, z0, x0, y0, z1); lineL(x1, y0, z0, x1, y0, z1); lineL(x0, y1, z0, x0, y1, z1); lineL(x1, y1, z0, x1, y1, z1);
}

// world-space box edges
function boxW(x0, y0, z0, x1, y1, z1) {
  line(x0, y0, z0, x1, y0, z0); line(x0, y1, z0, x1, y1, z0); line(x0, y0, z1, x1, y0, z1); line(x0, y1, z1, x1, y1, z1);
  line(x0, y0, z0, x0, y1, z0); line(x1, y0, z0, x1, y1, z0); line(x0, y0, z1, x0, y1, z1); line(x1, y0, z1, x1, y1, z1);
  line(x0, y0, z0, x0, y0, z1); line(x1, y0, z0, x1, y0, z1); line(x0, y1, z0, x0, y1, z1); line(x1, y1, z0, x1, y1, z1);
}

// circle of radius r centred at (cx, cy, cz) in the plane of local axes u, v (0 = x, 1 = y, 2 = z), full or half
function circleL(cx, cy, cz, r, u, v, segs, sign) {
  for (let i = 0; i < segs; i++) {
    const c0 = COS[i] * r, s0 = SIN[i] * r * sign, c1 = COS[i + 1] * r, s1 = SIN[i + 1] * r * sign;
    lineL(
      cx + (u === 0 ? c0 : v === 0 ? s0 : 0), cy + (u === 1 ? c0 : v === 1 ? s0 : 0), cz + (u === 2 ? c0 : v === 2 ? s0 : 0),
      cx + (u === 0 ? c1 : v === 0 ? s1 : 0), cy + (u === 1 ? c1 : v === 1 ? s1 : 0), cz + (u === 2 ? c1 : v === 2 ? s1 : 0));
  }
}

function drawShape(shape, B, id, top) {
  switch (shape.type) {
    case SHAPE.SPHERE: {
      const r = shape.radius;
      circleL(0, 0, 0, r, 0, 1, SEG, 1); circleL(0, 0, 0, r, 1, 2, SEG, 1); circleL(0, 0, 0, r, 2, 0, SEG, 1);
      return;
    }
    case SHAPE.CAPSULE: {
      const r = shape.radius, h = shape.halfHeight;
      circleL(0, h, 0, r, 2, 0, SEG, 1); circleL(0, -h, 0, r, 2, 0, SEG, 1);
      lineL(r, -h, 0, r, h, 0); lineL(-r, -h, 0, -r, h, 0); lineL(0, -h, r, 0, h, r); lineL(0, -h, -r, 0, h, -r);
      // cap half-circles: angle 0..pi from +u towards +y (top) / -y (bottom)
      circleL(0, h, 0, r, 0, 1, HALF, 1); circleL(0, h, 0, r, 2, 1, HALF, 1);
      circleL(0, -h, 0, r, 0, 1, HALF, -1); circleL(0, -h, 0, r, 2, 1, HALF, -1);
      return;
    }
    case SHAPE.BOX: { const x = shape.half[0], y = shape.half[1], z = shape.half[2]; boxL(-x, -y, -z, x, y, z); return; }
    case SHAPE.CONVEX: {
      const P = shape.points, E = shape.edges;
      if (E && E.length >= 2) {
        for (let i = 0; i + 1 < E.length; i += 2) { const a = E[i] * 3, b = E[i + 1] * 3; lineL(P[a], P[a + 1], P[a + 2], P[b], P[b + 1], P[b + 2]); }
      } else if (shape.faceStart && shape.faceVerts) {
        // face loops (faceStart: offsets into faceVerts, one past the end last)
        const S = shape.faceStart, F = shape.faceVerts;
        for (let f = 0; f + 1 < S.length; f++) {
          for (let k = S[f]; k < S[f + 1]; k++) {
            const a = F[k] * 3, b = F[k + 1 < S[f + 1] ? k + 1 : S[f]] * 3;
            lineL(P[a], P[a + 1], P[a + 2], P[b], P[b + 1], P[b + 2]);
          }
        }
      } else if (P && P.length >= 3) { // no topology: the local bounding box of its points
        let x0 = P[0], y0 = P[1], z0 = P[2], x1 = x0, y1 = y0, z1 = z0;
        for (let i = 3; i + 2 < P.length; i += 3) {
          const x = P[i], y = P[i + 1], z = P[i + 2];
          if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; if (z < z0) z0 = z; if (z > z1) z1 = z;
        }
        boxL(x0, y0, z0, x1, y1, z1);
      }
      return;
    }
    case SHAPE.TRIMESH: {
      if (top) { const A = B.aabb, o = id * 6; boxW(A[o], A[o + 1], A[o + 2], A[o + 3], A[o + 4], A[o + 5]); }
      else { const b = meshBounds(shape); boxL(b[0], b[1], b[2], b[3], b[4], b[5]); }
      return;
    }
    case SHAPE.COMPOUND: {
      const ch = shape.children;
      const px = _px, py = _py, pz = _pz, qx = _qx, qy = _qy, qz = _qz, qw = _qw;
      for (let i = 0; i < ch.length; i++) {
        const c = ch[i], cp = c.pos, cq = c.quat;
        // child pose = parent * (cp, cq)
        _px = px; _py = py; _pz = pz; _qx = qx; _qy = qy; _qz = qz; _qw = qw;
        xf(cp[0], cp[1], cp[2]);
        const bx = cq[0], by = cq[1], bz = cq[2], bw = cq[3];
        _qx = qw * bx + qx * bw + qy * bz - qz * by;
        _qy = qw * by - qx * bz + qy * bw + qz * bx;
        _qz = qw * bz + qx * by - qy * bx + qz * bw;
        _qw = qw * bw - qx * bx - qy * by - qz * bz;
        _px = _wx; _py = _wy; _pz = _wz;
        drawShape(c.shape, B, id, false);
      }
      _px = px; _py = py; _pz = pz; _qx = qx; _qy = qy; _qz = qz; _qw = qw;
      return;
    }
    default: return;
  }
}

function cross(x, y, z, s) {
  line(x - s, y, z, x + s, y, z); line(x, y - s, z, x, y + s, z); line(x, y, z - s, x, y, z + s);
}

/**
 * Write the world's debug lines into `out` (6 floats per segment) and, when given, per-vertex colours into
 * `colors` (same layout). Returns the number of segments written (<= max).
 * @param {object} world   DynamicsWorld (bodies, contacts?, joints?)
 * @param {Float32Array} out
 * @param {number} max     line budget
 * @param {object} [o]     bodies, contacts, joints (default true), aabbs (default false), colorByState (default true)
 * @param {Float32Array} [colors]
 */
export function writeDebugLines(world, out, max, o = DEFAULTS, colors = null) {
  if (!o) o = DEFAULTS;
  _o = out; _c = colors || null; _n = 0;
  _max = Math.min(max | 0, (out.length / 6) | 0);
  if (_c && _c.length < _max * 6) _max = (_c.length / 6) | 0;
  const B = world.bodies, byState = o.colorByState !== false;
  if (B && o.bodies !== false) {
    const P = B.pos, Q = B.quat;
    for (let id = 0; id < B.count && _n < _max; id++) {
      const shape = B.alive[id] ? B.shape[id] : null;
      if (!shape) continue;
      if (!byState) color(COLOR.plain);
      else {
        const t = B.type[id];
        color(t === BODY.STATIC ? COLOR.static : t === BODY.KINEMATIC ? COLOR.kinematic : (B.flags[id] & FLAG.ASLEEP) ? COLOR.asleep : COLOR.awake);
      }
      const p = id * 3, q = id * 4;
      _px = P[p]; _py = P[p + 1]; _pz = P[p + 2]; _qx = Q[q]; _qy = Q[q + 1]; _qz = Q[q + 2]; _qw = Q[q + 3];
      drawShape(shape, B, id, true);
    }
  }
  if (B && o.aabbs) {
    color(COLOR.aabb);
    const A = B.aabb;
    for (let id = 0; id < B.count && _n < _max; id++) {
      if (!B.alive[id]) continue;
      const k = id * 6;
      boxW(A[k], A[k + 1], A[k + 2], A[k + 3], A[k + 4], A[k + 5]);
    }
  }
  const C = world.contacts;
  if (C && o.contacts !== false) {
    color(COLOR.contact);
    for (let i = 0; i < C.count && _n < _max; i++) {
      const x = C.px[i], y = C.py[i], z = C.pz[i];
      line(x, y, z, x + C.nx[i] * 0.2, y + C.ny[i] * 0.2, z + C.nz[i] * 0.2);
    }
  }
  const J = world.joints;
  if (J && B && o.joints !== false) {
    color(COLOR.joint);
    const P = B.pos, Q = B.quat;
    for (let j = 0; j < J.count && _n < _max; j++) {
      if ((J.alive && !J.alive[j]) || (J.broken && J.broken[j])) continue;
      let ax, ay, az;
      const a = J.a[j], b = J.b[j], k = j * 3;
      if (a >= 0) {
        _px = P[a * 3]; _py = P[a * 3 + 1]; _pz = P[a * 3 + 2]; _qx = Q[a * 4]; _qy = Q[a * 4 + 1]; _qz = Q[a * 4 + 2]; _qw = Q[a * 4 + 3];
        xf(J.anchorA[k], J.anchorA[k + 1], J.anchorA[k + 2]); ax = _wx; ay = _wy; az = _wz;
      } else { ax = J.anchorA[k]; ay = J.anchorA[k + 1]; az = J.anchorA[k + 2]; }
      let bx, by, bz;
      if (b >= 0) {
        _px = P[b * 3]; _py = P[b * 3 + 1]; _pz = P[b * 3 + 2]; _qx = Q[b * 4]; _qy = Q[b * 4 + 1]; _qz = Q[b * 4 + 2]; _qw = Q[b * 4 + 3];
        xf(J.anchorB[k], J.anchorB[k + 1], J.anchorB[k + 2]); bx = _wx; by = _wy; bz = _wz;
      } else { bx = J.anchorB[k]; by = J.anchorB[k + 1]; bz = J.anchorB[k + 2]; }
      cross(ax, ay, az, 0.1); cross(bx, by, bz, 0.1);
      line(ax, ay, az, bx, by, bz);
    }
  }
  const n = _n;
  _o = null; _c = null; // do not retain the caller's arrays
  return n;
}

/** Line counts per shape type (sphere, capsule, box, trimesh) and per contact / joint / AABB, for budgeting. */
export const DEBUG_LINES = Object.freeze({ sphere: 3 * SEG, capsule: 2 * SEG + 4 + 4 * HALF, box: 12, trimesh: 12, contact: 1, joint: 7, aabb: 12 });
