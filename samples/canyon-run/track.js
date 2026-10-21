// The course: a closed, winding loop (centripetal Catmull-Rom through a perturbed circle) resampled into a
// 2 m polyline with arc length, tangents and curvature; the canyon-floor elevation profile along it (gentle
// rolls + three kicker crests), the floor half-width, the checkpoint list and a cheap nearest-point tracker.
import * as THREE from 'three';
import { Simplex, Noise } from '@vibe/engine';
const { smoothstep } = Noise;

export const TRACK_HALF_WIDTH = 6.5;     // dirt ribbon half width (m)
export const GATE_HALF_WIDTH = TRACK_HALF_WIDTH + 2.2;
const STEP = 2;                          // polyline spacing (m)

export class Track {
  constructor({ seed = 7, checkpointSpacing = 250 } = {}) {
    const noise = this.noise = new Simplex(seed + 101);
    // control points: perturbed circle, r in ~[280, 780] m
    const K = 30, ctrl = [];
    for (let k = 0; k < K; k++) {
      const th = (k / K) * Math.PI * 2;
      const r = 540 + 120 * Math.sin(2 * th + 0.7) + 80 * Math.sin(3 * th + 1.9) + 48 * Math.sin(5 * th + 0.3) + 26 * Math.sin(7 * th + 2.2);
      const a = th + 0.07 * Math.sin(4 * th + 0.4);
      ctrl.push(new THREE.Vector3(Math.cos(a) * r, 0, Math.sin(a) * r));
    }
    const curve = new THREE.CatmullRomCurve3(ctrl, true, 'centripetal');
    const approxLen = curve.getLength();
    const n = Math.round(approxLen / STEP);
    const pts = curve.getSpacedPoints(n); // n + 1 points, last == first
    pts.pop();
    // start at the point nearest the south-east straight (so the gantry sits on a gentle section)
    this.n = n;
    this.x = new Float32Array(n); this.z = new Float32Array(n); this.y = new Float32Array(n);
    this.s = new Float32Array(n); this.tx = new Float32Array(n); this.tz = new Float32Array(n);
    this.curv = new Float32Array(n); this.halfFloor = new Float32Array(n);
    let best = 0, bestK = Infinity;
    for (let i = 0; i < n; i++) { // pick the straightest spot as the start line
      const a = pts[(i - 6 + n) % n], b = pts[i], c = pts[(i + 6) % n];
      const k = Math.abs((b.x - a.x) * (c.z - b.z) - (b.z - a.z) * (c.x - b.x));
      if (pts[i].z > 200 && k < bestK) { bestK = k; best = i; }
    }
    for (let i = 0; i < n; i++) { const p = pts[(i + best) % n]; this.x[i] = p.x; this.z[i] = p.z; }
    let s = 0;
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      this.s[i] = s;
      s += Math.hypot(this.x[j] - this.x[i], this.z[j] - this.z[i]);
    }
    this.length = s;
    for (let i = 0; i < n; i++) {
      const a = (i - 1 + n) % n, b = (i + 1) % n;
      let dx = this.x[b] - this.x[a], dz = this.z[b] - this.z[a];
      const l = Math.hypot(dx, dz) || 1; dx /= l; dz /= l;
      this.tx[i] = dx; this.tz[i] = dz;
    }
    for (let i = 0; i < n; i++) { // signed curvature over +-8 m
      const a = (i - 4 + n) % n, b = (i + 4) % n;
      const da = Math.atan2(this.tz[b], this.tx[b]) - Math.atan2(this.tz[a], this.tx[a]);
      const w = Math.atan2(Math.sin(da), Math.cos(da));
      this.curv[i] = w / 16;
    }
    // kicker crests (fraction of the lap), chosen on straight-ish sections
    this.crests = [0.205, 0.53, 0.79].map((f) => this._straightest(f * s, 160));
    for (let i = 0; i < n; i++) { this.y[i] = this.elevation(this.s[i]); this.halfFloor[i] = this.floorHalfWidth(this.s[i]); }

    // checkpoints: every ~checkpointSpacing m, index 0 = start / finish
    const N = Math.max(4, Math.round(s / checkpointSpacing));
    this.checkpoints = [];
    for (let k = 0; k < N; k++) {
      const i = Math.round((k / N) * n) % n;
      this.checkpoints.push({ index: k, i, s: this.s[i], x: this.x[i], z: this.z[i], y: this.y[i], tx: this.tx[i], tz: this.tz[i], heading: Math.atan2(-this.tx[i], -this.tz[i]) });
    }
    let minR = Infinity; for (let i = 0; i < n; i++) minR = Math.min(minR, 1 / Math.max(1e-6, Math.abs(this.curv[i])));
    this.minRadius = minR;
  }

  _straightest(s0, win) {
    let bestS = s0, bestC = Infinity;
    for (let d = -win; d <= win; d += 4) {
      const i = this.indexAt(s0 + d);
      let c = 0; for (let k = -12; k <= 12; k += 3) c += Math.abs(this.curv[(i + k + this.n) % this.n]);
      if (c < bestC) { bestC = c; bestS = s0 + d; }
    }
    return bestS;
  }

  indexAt(s) { const L = this.length; s = ((s % L) + L) % L; return Math.min(this.n - 1, Math.round(s / STEP)) % this.n; }

  /** Canyon-floor height (m) at arc length s: periodic rolls + kicker crests (steep up, long run-out). */
  elevation(s) {
    const L = this.length, u = (s / L) * Math.PI * 2;
    let e = 7 * Math.sin(2 * u + 0.5) + 4.5 * Math.sin(5 * u + 1.3) + 1.8 * Math.sin(11 * u + 0.2);
    if (this.crests) for (const c of this.crests) {
      let d = s - c; if (d > L / 2) d -= L; if (d < -L / 2) d += L;
      if (d > -18 && d < 40) e += d < 0 ? 2.4 * smoothstep(-18, 0, d) : 2.4 * (1 - smoothstep(0, 40, d));
    }
    return e;
  }

  /** Half width of the flat canyon floor (m): narrow slots and wide washes. */
  floorHalfWidth(s) {
    const L = this.length, u = (s / L) * Math.PI * 2;
    const w = 0.5 + 0.5 * Math.sin(3 * u + 2.0) * 0.8 + 0.2 * Math.sin(7 * u + 0.6);
    return 15 + 26 * Math.max(0, w) * w;
  }

  /**
   * Nearest polyline point to (x, z), searching +-window points around `hint` (or globally when hint < 0).
   * Writes into out { i, s, lateral (signed, + = right of travel), dist, along } and returns it.
   */
  nearest(x, z, hint, out, window = 40) {
    const n = this.n;
    let bi = 0, bd = Infinity;
    if (hint < 0) { for (let i = 0; i < n; i += 2) { const d = (this.x[i] - x) ** 2 + (this.z[i] - z) ** 2; if (d < bd) { bd = d; bi = i; } } hint = bi; window = 4; bd = Infinity; }
    for (let k = -window; k <= window; k++) {
      const i = (hint + k + n) % n;
      const d = (this.x[i] - x) ** 2 + (this.z[i] - z) ** 2;
      if (d < bd) { bd = d; bi = i; }
    }
    // project onto the segment to the better neighbour
    const dx = x - this.x[bi], dz = z - this.z[bi];
    const along = dx * this.tx[bi] + dz * this.tz[bi];
    const lat = dx * -this.tz[bi] + dz * this.tx[bi]; // right of travel = tangent rotated (-tz, tx)
    out.i = bi; out.s = this.s[bi] + along; out.along = along; out.lateral = lat; out.dist = Math.abs(lat);
    return out;
  }

  /** Minimap polyline ([[x, z], ...] closed). */
  minimapData() {
    const road = [];
    for (let i = 0; i < this.n; i += 3) road.push([this.x[i], this.z[i]]);
    road.push([this.x[0], this.z[0]]);
    return { roads: [road] };
  }
}
