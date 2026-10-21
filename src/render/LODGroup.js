// LODGroup: near / far partition of N static instances by camera distance — the pattern behind the city's
// vegetation (full models near the camera in per-type InstancedMeshes, impostors for everything else).
//   - per-instance hysteresis: an instance enters the near set inside `nearRadius` and leaves it only beyond
//     `nearRadius + hysteresis`, so instances on the boundary do not flip every re-partition
//   - throttled: re-partitions only after the camera moved `moveThreshold` metres since the last one, at most
//     once per `minInterval` seconds unless it jumped more than `jumpDistance` (teleport / fast flight)
//   - results are index lists into your own instance arrays (`near`/`nearCount`, optional `far`/`farCount`),
//     plus copyNear() helpers that pack the near instances' matrices / attributes into an InstancedPool
// Sizing: pick nearRadius = (distance where the near model has fully faded out) + (moveThreshold + margin),
// because the camera can travel moveThreshold before the next partition lands. update() is allocation-free.
//
//   const lod = new LODGroup({ positions, nearRadius: 220, moveThreshold: 20 });
//   per frame: if (lod.update(camera.position, dt)) lod.copyNear(matrices, nearPool);
import * as THREE from 'three';
import { touchRange, newRange } from './bufferRanges.js';

export class LODGroup {
  /**
   * @param {object} o
   * @param {Float32Array|number[]} o.positions  x, y, z per instance (world space)
   * @param {number} [o.count]               instance count (default positions.length / 3)
   * @param {number} o.nearRadius            near-set radius (m)
   * @param {number} [o.hysteresis]          extra radius before a near instance drops out (default 5% of nearRadius)
   * @param {number} [o.moveThreshold=20]    camera travel (m) that triggers a re-partition
   * @param {number} [o.minInterval=0.1]     minimum seconds between re-partitions ...
   * @param {number} [o.jumpDistance=60]     ... unless the camera moved farther than this
   * @param {boolean} [o.trackFar=false]     also fill `far` / `farCount`
   * @param {boolean} [o.horizontal=false]   measure distance in xz only
   * @param {(lod: LODGroup) => void} [o.onPartition]
   */
  constructor(o) {
    this.positions = o.positions;
    this.n = o.count ?? Math.floor(o.positions.length / 3);
    this.nearRadius = o.nearRadius;
    this.hysteresis = o.hysteresis ?? o.nearRadius * 0.05;
    this.moveThreshold = o.moveThreshold ?? 20;
    this.minInterval = o.minInterval ?? 0.1;
    this.jumpDistance = o.jumpDistance ?? 60;
    this.horizontal = !!o.horizontal;
    this.onPartition = o.onPartition || null;
    this.isNear = new Uint8Array(this.n);
    this.near = new Uint32Array(this.n); this.nearCount = 0;
    this.far = o.trackFar ? new Uint32Array(this.n) : null; this.farCount = 0;
    this.center = new THREE.Vector3(Infinity, 0, 0); // camera position of the last partition
    this.sincePartition = 1e9;
    this.partitions = 0;
    this._range = newRange();
  }

  /** Force a re-partition on the next update (e.g. after moving instances). */
  invalidate() { this.center.set(Infinity, 0, 0); }

  /** Returns true when the near set was rebuilt this call. */
  update(cameraPos, dt = 0) {
    this.sincePartition += dt || 0;
    const c = this.center;
    const dx = cameraPos.x - c.x, dy = this.horizontal ? 0 : cameraPos.y - c.y, dz = cameraPos.z - c.z;
    const d2 = dx * dx + dy * dy + dz * dz;
    const T = this.moveThreshold, J = this.jumpDistance;
    if (!(d2 > T * T) || !(this.sincePartition >= this.minInterval || d2 > J * J)) return false;
    this.partition(cameraPos.x, cameraPos.y, cameraPos.z);
    return true;
  }

  /** Rebuild the near (and far) index lists around (cx, cy, cz) now. */
  partition(cx, cy, cz) {
    const P = this.positions, flag = this.isNear, near = this.near, far = this.far;
    const rin = this.nearRadius, rout = rin + this.hysteresis, rin2 = rin * rin, rout2 = rout * rout, hz = this.horizontal;
    let nn = 0, nf = 0;
    for (let i = 0; i < this.n; i++) {
      const k = i * 3, dx = P[k] - cx, dy = hz ? 0 : P[k + 1] - cy, dz = P[k + 2] - cz;
      const d2 = dx * dx + dy * dy + dz * dz;
      const isN = d2 < rin2 || (flag[i] === 1 && d2 < rout2);
      flag[i] = isN ? 1 : 0;
      if (isN) near[nn++] = i;
      else if (far) far[nf++] = i;
    }
    this.nearCount = nn; this.farCount = nf;
    this.center.set(cx, cy, cz);
    this.sincePartition = 0;
    this.partitions++;
    if (this.onPartition) this.onPartition(this);
  }

  /**
   * Pack the near instances' matrices (src: 16 floats per instance, all N) into an InstancedPool (or anything
   * with mesh.instanceMatrix + setCount) and refresh its bounds. Returns the near count.
   */
  copyNear(srcMatrices, pool) {
    const n = this.nearCount, idx = this.near;
    pool.setCount(n);
    const a = pool.mesh.instanceMatrix.array;
    for (let j = 0; j < n; j++) {
      const s = idx[j] * 16, t = j * 16;
      for (let e = 0; e < 16; e++) a[t + e] = srcMatrices[s + e];
    }
    touchRange(pool.mesh.instanceMatrix, pool._mr || this._range, 0, Math.max(1, n) * 16);
    if (pool.updateBounds) pool.updateBounds();
    return n;
  }

  /** Pack a per-instance attribute (itemSize floats per instance, all N) for the near set into dst. */
  copyNearAttribute(src, itemSize, dst) {
    const n = this.nearCount, idx = this.near;
    for (let j = 0; j < n; j++) {
      const s = idx[j] * itemSize, t = j * itemSize;
      for (let e = 0; e < itemSize; e++) dst[t + e] = src[s + e];
    }
    return n;
  }
}
