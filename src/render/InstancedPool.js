// InstancedPool: a growable THREE.InstancedMesh with cheap per-instance setters.
//   - add() / removeAt() (swap-remove) / setCount(n); capacity doubles on demand up to maxCapacity
//   - setTRS(i, x, y, z, yaw, sx, sy, sz) fast path (no Matrix4), setTransform(i, pos, quat, scale),
//     setPosition, setMatrix / getMatrix, getPosition, setColor
//   - partial GPU uploads: setters merge into one reused update range per attribute (no commit call needed)
//   - culling: the mesh bounding sphere is kept from the instance translations + `cullRadius` (the radius of one
//     instance), so three's frustum test covers the whole pool without computeBoundingSphere() every frame;
//     `maxDistance` + update(cameraPos) hides the pool when it is entirely farther than that
// Setters, update() and the bounds bookkeeping are allocation-free (growing reallocates, amortised).
//
//   const pool = new InstancedPool(geo, mat, { capacity: 256, colors: true, cullRadius: 2 });
//   scene.add(pool.mesh);
//   const i = pool.add(); pool.setTRS(i, x, 0, z, yaw, 1.2); pool.setColor(i, 1, 0.5, 0.2);
import * as THREE from 'three';
import { touchRange, newRange } from './bufferRanges.js';

export class InstancedPool {
  /**
   * @param {THREE.BufferGeometry} geometry
   * @param {THREE.Material} material
   * @param {object} [o]
   * @param {number} [o.capacity=64]       initial capacity
   * @param {number} [o.maxCapacity=Infinity]
   * @param {number} [o.growth=2]          capacity multiplier when full
   * @param {boolean} [o.colors=false]     allocate instanceColor
   * @param {number} [o.cullRadius]        radius of one instance (default: geometry bounding radius)
   * @param {number} [o.maxDistance=Infinity] hide the whole pool beyond this camera distance (see update)
   * @param {boolean} [o.dynamic=true]     DynamicDrawUsage for the instance buffers
   */
  constructor(geometry, material, o = {}) {
    this.geometry = geometry; this.material = material;
    this.capacity = Math.max(1, o.capacity ?? 64);
    this.maxCapacity = o.maxCapacity ?? Infinity;
    this.growth = Math.max(1.25, o.growth ?? 2);
    this.dynamic = o.dynamic !== false;
    if (!geometry.boundingSphere) geometry.computeBoundingSphere();
    this.cullRadius = o.cullRadius ?? geometry.boundingSphere.radius;
    this.maxDistance = o.maxDistance ?? Infinity;
    this.mesh = new THREE.InstancedMesh(geometry, material, this.capacity);
    this.mesh.count = 0;
    this.mesh.name = o.name || 'instancedPool';
    if (o.castShadow) this.mesh.castShadow = true;
    if (o.receiveShadow !== false) this.mesh.receiveShadow = true;
    if (this.dynamic) this.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    if (o.colors) {
      this.mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(this.capacity * 3).fill(1), 3);
      if (this.dynamic) this.mesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
    }
    this.mesh.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 0);
    this._mr = newRange(); this._cr = newRange();
    this._min = new THREE.Vector3(Infinity, Infinity, Infinity);
    this._max = new THREE.Vector3(-Infinity, -Infinity, -Infinity);
    this._boundsDirty = false;
    this._boundsStale = false;
    this._wasCulled = false;
  }

  get count() { return this.mesh.count; }
  /** Active instance count (instances >= n are not drawn; their data is kept). Grows capacity if needed. */
  setCount(n) {
    n = Math.max(0, n | 0);
    if (n > this.capacity && !this._grow(n)) n = this.capacity;
    this.mesh.count = n;
    return n;
  }
  /** Append an instance (identity matrix, white); returns its index, or -1 at maxCapacity. */
  add() {
    const i = this.mesh.count;
    if (i >= this.capacity && !this._grow(i + 1)) return -1;
    this.mesh.count = i + 1;
    const a = this.mesh.instanceMatrix.array, k = i * 16;
    for (let j = 0; j < 16; j++) a[k + j] = (j % 5 === 0) ? 1 : 0;
    touchRange(this.mesh.instanceMatrix, this._mr, k, 16);
    if (this.mesh.instanceColor) this.setColor(i, 1, 1, 1);
    return i;
  }
  /** Swap-remove: the last instance moves into slot i. Returns the old index of the moved instance (or -1). */
  removeAt(i) {
    const last = this.mesh.count - 1;
    if (i < 0 || i > last) return -1;
    if (i !== last) {
      const a = this.mesh.instanceMatrix.array;
      a.copyWithin(i * 16, last * 16, last * 16 + 16);
      touchRange(this.mesh.instanceMatrix, this._mr, i * 16, 16);
      const c = this.mesh.instanceColor;
      if (c) { c.array.copyWithin(i * 3, last * 3, last * 3 + 3); touchRange(c, this._cr, i * 3, 3); }
    }
    this.mesh.count = last;
    return i !== last ? last : -1;
  }

  /** Position + rotation about +Y + scale, written straight into the instance matrix. */
  setTRS(i, x, y, z, yaw = 0, sx = 1, sy = sx, sz = sx) {
    const a = this.mesh.instanceMatrix.array, k = i * 16, c = Math.cos(yaw), s = Math.sin(yaw);
    a[k] = c * sx; a[k + 1] = 0; a[k + 2] = -s * sx; a[k + 3] = 0;
    a[k + 4] = 0; a[k + 5] = sy; a[k + 6] = 0; a[k + 7] = 0;
    a[k + 8] = s * sz; a[k + 9] = 0; a[k + 10] = c * sz; a[k + 11] = 0;
    a[k + 12] = x; a[k + 13] = y; a[k + 14] = z; a[k + 15] = 1;
    touchRange(this.mesh.instanceMatrix, this._mr, k, 16);
    this._extend(x, y, z);
  }
  /** Full TRS from a Vector3, Quaternion and Vector3 scale. */
  setTransform(i, position, quaternion, scale) {
    _m.compose(position, quaternion, scale);
    this.setMatrix(i, _m);
  }
  /** Translation only (rotation / scale kept). */
  setPosition(i, x, y, z) {
    const a = this.mesh.instanceMatrix.array, k = i * 16;
    a[k + 12] = x; a[k + 13] = y; a[k + 14] = z;
    touchRange(this.mesh.instanceMatrix, this._mr, k + 12, 3);
    this._extend(x, y, z);
  }
  setMatrix(i, m) {
    const k = i * 16;
    m.toArray(this.mesh.instanceMatrix.array, k);
    touchRange(this.mesh.instanceMatrix, this._mr, k, 16);
    const e = m.elements; this._extend(e[12], e[13], e[14]);
  }
  getMatrix(i, out) { return out.fromArray(this.mesh.instanceMatrix.array, i * 16); }
  getPosition(i, out) { const a = this.mesh.instanceMatrix.array, k = i * 16; return out.set(a[k + 12], a[k + 13], a[k + 14]); }
  setColor(i, r, g, b) {
    const c = this.mesh.instanceColor;
    if (!c) return;
    const a = c.array, k = i * 3; a[k] = r; a[k + 1] = g; a[k + 2] = b;
    touchRange(c, this._cr, k, 3);
  }
  /** Copy n instance matrices (16 floats each) from `src` starting at instance srcStart into slots [dst, dst + n). */
  copyMatrices(src, srcStart, dst, n) {
    const a = this.mesh.instanceMatrix.array;
    if (dst + n > this.capacity && !this._grow(dst + n)) n = this.capacity - dst;
    a.set(src.subarray(srcStart * 16, (srcStart + n) * 16), dst * 16);
    touchRange(this.mesh.instanceMatrix, this._mr, dst * 16, n * 16);
    this._boundsStale = true;
  }

  /** Recompute the bounding sphere from the active instances' translations (O(n), allocation-free). */
  updateBounds() {
    const a = this.mesh.instanceMatrix.array, n = this.mesh.count;
    this._min.set(Infinity, Infinity, Infinity); this._max.set(-Infinity, -Infinity, -Infinity);
    for (let i = 0; i < n; i++) { const k = i * 16; this._extend(a[k + 12], a[k + 13], a[k + 14]); }
    this._boundsStale = false;
    this._applyBounds();
  }

  /**
   * Per frame (optional): refresh the bounding sphere when instances moved, and hide the pool when the camera
   * is farther than maxDistance from every instance. Returns true when the pool is visible.
   */
  update(cameraPos) {
    if (this._boundsStale) this.updateBounds();
    else if (this._boundsDirty) this._applyBounds();
    const bs = this.mesh.boundingSphere;
    if (cameraPos && this.maxDistance < Infinity) {
      const d = cameraPos.distanceTo(bs.center) - bs.radius;
      const culled = d > this.maxDistance * (this._wasCulled ? 0.97 : 1); // 3% hysteresis
      this._wasCulled = culled;
      this.mesh.visible = !culled && this.mesh.count > 0;
    } else this.mesh.visible = this.mesh.count > 0;
    return this.mesh.visible;
  }

  _extend(x, y, z) {
    const mn = this._min, mx = this._max;
    if (x < mn.x) mn.x = x; if (y < mn.y) mn.y = y; if (z < mn.z) mn.z = z;
    if (x > mx.x) mx.x = x; if (y > mx.y) mx.y = y; if (z > mx.z) mx.z = z;
    this._boundsDirty = true;
  }
  _applyBounds() {
    const bs = this.mesh.boundingSphere, mn = this._min, mx = this._max;
    if (mn.x === Infinity) { bs.center.set(0, 0, 0); bs.radius = 0; }
    else {
      bs.center.set((mn.x + mx.x) / 2, (mn.y + mx.y) / 2, (mn.z + mx.z) / 2);
      bs.radius = Math.hypot(mx.x - mn.x, mx.y - mn.y, mx.z - mn.z) / 2 + this.cullRadius;
    }
    this._boundsDirty = false;
  }

  _grow(need) {
    if (need > this.maxCapacity) return false;
    const cap = Math.min(this.maxCapacity, Math.max(need, Math.ceil(this.capacity * this.growth)));
    const mesh = this.mesh;
    // free the old GPU buffers (WebGLObjects listens for 'dispose' on instanced meshes), then swap in new ones
    mesh.dispatchEvent({ type: 'dispose' });
    const m = new THREE.InstancedBufferAttribute(new Float32Array(cap * 16), 16);
    m.array.set(mesh.instanceMatrix.array);
    if (this.dynamic) m.setUsage(THREE.DynamicDrawUsage);
    mesh.instanceMatrix = m;
    if (mesh.instanceColor) {
      const c = new THREE.InstancedBufferAttribute(new Float32Array(cap * 3).fill(1), 3);
      c.array.set(mesh.instanceColor.array);
      if (this.dynamic) c.setUsage(THREE.DynamicDrawUsage);
      mesh.instanceColor = c;
    }
    this.capacity = cap;
    return true;
  }

  dispose() { this.mesh.dispose(); }
}
const _m = new THREE.Matrix4();
