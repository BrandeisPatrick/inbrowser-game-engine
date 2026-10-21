// A small pool of real PointLights driven from a large set of light positions (street lamps, windows, ...).
// Forward-lit materials pay for every real light on every lit fragment, and on Apple / ANGLE the per-light cost
// climbs steeply past ~8 lights, so the pool stays small and each light is spent where it can be SEEN: only
// candidates whose light sphere (radius = light.distance) reaches the view frustum compete, ranked by distance
// to a point a little ahead of the camera. Newly assigned slots fade in instead of popping. Selection is
// throttled (camera moved / turned / timer), so the per-frame cost is a fade step.
//
//   const pool = new NearestLights({ count: 8, color: 0xffc98a, distance: 48, decay: 2, parent: group });
//   pool.setPositions(xyzFloat32Array);                       // or [[x,y,z], ...]
//   pool.update(dt, { position, frustum, direction }, intensity);   // intensity 0 switches the pool off
import * as THREE from 'three';

const _sph = new THREE.Sphere();

export class NearestLights {
  /**
   * @param {object} o
   * @param {number} [o.count=8]
   * @param {number|THREE.Color} [o.color=0xffffff] @param {number} [o.distance=40] @param {number} [o.decay=2]
   * @param {THREE.Object3D} [o.parent]  lights are added here
   * @param {number} [o.fade=0.25]  fade-in time (s) of a newly assigned light
   * @param {number} [o.lead=14]  ranking point ahead of the camera (m, horizontal)
   * @param {number} [o.heightWeight=0.25]  vertical distance weight in the ranking (lamps far below an aerial
   *        camera still rank by their horizontal distance)
   * @param {number} [o.yOffset=0]  added to each position
   */
  constructor(o = {}) {
    this.count = o.count ?? 8;
    this.fade = o.fade ?? 0.25;
    this.lead = o.lead ?? 14;
    this.hw = o.heightWeight ?? 0.25;
    this.yOffset = o.yOffset ?? 0;
    this.lights = [];
    for (let i = 0; i < this.count; i++) {
      const l = new THREE.PointLight(o.color ?? 0xffffff, 0, o.distance ?? 40, o.decay ?? 2);
      l.visible = false; l.castShadow = false; l.name = o.name || 'pool-light';
      this.lights.push(l);
      if (o.parent) o.parent.add(l);
    }
    this.reach = o.distance ?? 40;
    this.P = new Float32Array(0); this.n = 0;
    this.slot = new Int32Array(this.count).fill(-1); // candidate index per light
    this.k = new Float32Array(this.count);           // fade 0..1 per light
    this._timer = 0; this._last = new THREE.Vector3(1e9, 0, 0); this._lastDir = new THREE.Vector3(); this._on = false; this._fresh = true;
    this._best = []; this.selections = 0;
  }

  /** Candidate positions: Float32Array [x,y,z,...] or an array of [x,y,z]. */
  setPositions(p) {
    if (p instanceof Float32Array) this.P = p;
    else { this.P = new Float32Array(p.length * 3); p.forEach((q, i) => { this.P[i * 3] = q[0]; this.P[i * 3 + 1] = q[1]; this.P[i * 3 + 2] = q[2]; }); }
    this.n = this.P.length / 3;
    this.slot.fill(-1); this._timer = 0; this._last.set(1e9, 0, 0);
  }

  /**
   * @param {number} dt
   * @param {{position: THREE.Vector3, frustum?: THREE.Frustum, direction?: THREE.Vector3}} view
   * @param {number} intensity  0 = off
   */
  update(dt, view, intensity) {
    const on = intensity > 0 && !!view && !!view.position;
    if (on !== this._on) { this._on = on; for (const l of this.lights) l.visible = on; this.slot.fill(-1); this.k.fill(0); this._fresh = true; }
    if (!on) return;
    const cam = view.position, dir = view.direction || null;
    this._timer -= dt;
    const turned = dir ? this._lastDir.dot(dir) < 0.985 : false;
    const moved = this._last.distanceToSquared(cam);
    if (this._timer <= 0 || turned || moved > 16) this._select(cam, view.frustum || null, dir, moved > 900 || this._fresh);
    this._fresh = false;
    for (let i = 0; i < this.count; i++) {
      const l = this.lights[i];
      if (this.slot[i] < 0) { l.intensity = 0; continue; }
      this.k[i] = this.fade > 0 ? Math.min(1, this.k[i] + dt / this.fade) : 1;
      l.intensity = intensity * this.k[i];
    }
  }

  /** `snap`: no fade-in (switch-on, teleports / camera cuts). */
  _select(cam, frustum, dir, snap = false) {
    this._timer = 0.3; this._last.copy(cam); if (dir) this._lastDir.copy(dir);
    this.selections++;
    const P = this.P, n = this.n, N = this.count, hw = this.hw;
    let ax = cam.x, az = cam.z;
    if (dir) { const l = Math.hypot(dir.x, dir.z); if (l > 1e-3) { ax += (dir.x / l) * this.lead; az += (dir.z / l) * this.lead; } }
    const best = this._best; best.length = 0; // [d, i] sorted ascending, at most N
    let worst = Infinity;
    for (let i = 0; i < n; i++) {
      const x = P[i * 3], y = P[i * 3 + 1] + this.yOffset, z = P[i * 3 + 2];
      const d = (x - ax) ** 2 + (z - az) ** 2 + (y - cam.y) ** 2 * hw;
      if (best.length === N && d >= worst) continue;
      if (frustum && !frustum.intersectsSphere(_sph.set(_sph.center.set(x, y, z), this.reach))) continue;
      if (best.length < N) { best.push([d, i]); if (best.length === N) { best.sort((a, b) => a[0] - b[0]); worst = best[N - 1][0]; } }
      else { best[N - 1] = [d, i]; best.sort((a, b) => a[0] - b[0]); worst = best[N - 1][0]; }
    }
    // keep lights on candidates that stay selected (no fade restart), give the rest to newcomers
    const want = new Set(best.map((b) => b[1]));
    for (let s = 0; s < N; s++) if (this.slot[s] >= 0 && !want.has(this.slot[s])) this.slot[s] = -1;
    for (let s = 0; s < N; s++) if (this.slot[s] >= 0) want.delete(this.slot[s]);
    for (const i of want) {
      const s = this.slot.indexOf(-1); if (s < 0) break;
      this.slot[s] = i; this.k[s] = snap ? 1 : 0;
      this.lights[s].position.set(P[i * 3], P[i * 3 + 1] + this.yOffset, P[i * 3 + 2]);
      this.lights[s].updateMatrixWorld();
    }
  }
}
