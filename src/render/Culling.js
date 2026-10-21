// ChunkCuller: frustum + distance culling for many static chunk objects (merged meshes, groups), driving
// object.visible so hidden chunks cost nothing in the render traversal (three skips invisible subtrees).
//   - world bounding spheres are computed once at add() (chunks are static; call refresh(obj) after moving one)
//   - distance culling is re-evaluated for at most `budget` chunks per frame (round-robin), with a 5% hysteresis
//     band; a camera jump > `teleportDistance` forces a full pass that frame
//   - frustum culling runs every frame on the distance-visible chunks (6 plane tests each). Shadow casters are
//     only distance-culled by default, because hiding them would also remove shadows they cast into view from
//     outside the frustum (three frustum-tests them per shadow camera anyway); `frustumHidesShadowCasters: true`
//     overrides that
// update() is allocation-free.
//
//   const culler = new ChunkCuller({ maxDistance: 1500, budget: 128 });
//   for (const m of chunkMeshes) culler.add(m);            // or add(m, { maxDistance: 600 })
//   per frame: culler.update(camera);  culler.stats -> { total, visible, tested }
import * as THREE from 'three';

export class ChunkCuller {
  /**
   * @param {object} [o]
   * @param {number} [o.maxDistance=Infinity]  default per-chunk draw distance (camera to sphere surface, m)
   * @param {number} [o.budget=128]            chunks whose distance is re-evaluated per frame
   * @param {boolean} [o.frustum=true]
   * @param {boolean} [o.frustumHidesShadowCasters=false]
   * @param {number} [o.teleportDistance=100]  camera jump that forces a full distance pass
   * @param {number} [o.padding=0]             added to every bounding radius (m)
   */
  constructor(o = {}) {
    this.maxDistance = o.maxDistance ?? Infinity;
    this.budget = Math.max(1, o.budget ?? 128);
    this.frustumOn = o.frustum !== false;
    this.hideCasters = !!o.frustumHidesShadowCasters;
    this.teleportDistance = o.teleportDistance ?? 100;
    this.padding = o.padding ?? 0;
    this.chunks = [];        // { obj, sphere, maxDist, near (distance-visible), selfFrustum }
    this._cursor = 0;
    this._frustum = new THREE.Frustum();
    this._pv = new THREE.Matrix4();
    this._last = new THREE.Vector3(Infinity, 0, 0);
    this.stats = { total: 0, visible: 0, tested: 0 };
  }

  /** Register a chunk. opts: { maxDistance, radius, center (world) } override the computed bounds. */
  add(obj, opts = {}) {
    const c = { obj, sphere: new THREE.Sphere(), maxDist: opts.maxDistance ?? this.maxDistance, near: true, selfFrustum: false, opts };
    this._bounds(c);
    c.selfFrustum = this.frustumOn && (this.hideCasters || !this._castsShadow(obj));
    if (c.selfFrustum) obj.frustumCulled = false; // we test it; skip three's second test
    this.chunks.push(c);
    this.stats.total = this.chunks.length;
    return c;
  }
  remove(obj) {
    const i = this.chunks.findIndex((c) => c.obj === obj);
    if (i < 0) return false;
    const c = this.chunks[i];
    if (c.selfFrustum) obj.frustumCulled = true;
    obj.visible = true;
    this.chunks.splice(i, 1);
    this.stats.total = this.chunks.length;
    return true;
  }
  /** Recompute a chunk's world bounds (after it moved). */
  refresh(obj) { const c = this.chunks.find((q) => q.obj === obj); if (c) this._bounds(c); }

  update(camera) {
    const ch = this.chunks, n = ch.length;
    if (!n) return;
    camera.updateMatrixWorld();
    const cp = _cp.setFromMatrixPosition(camera.matrixWorld);
    this._pv.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    this._frustum.setFromProjectionMatrix(this._pv, camera.coordinateSystem);
    const jumped = this._last.distanceToSquared(cp) > this.teleportDistance * this.teleportDistance;
    this._last.copy(cp);
    // distance: budgeted round-robin (full pass after a jump)
    const count = jumped ? n : Math.min(n, this.budget);
    for (let k = 0; k < count; k++) {
      if (this._cursor >= n) this._cursor = 0;
      const c = ch[this._cursor++];
      const d = cp.distanceTo(c.sphere.center) - c.sphere.radius;
      c.near = d < c.maxDist * (c.near ? 1.05 : 1);
    }
    // frustum: every frame for distance-visible chunks
    let vis = 0;
    for (let i = 0; i < n; i++) {
      const c = ch[i];
      const v = c.near && (!c.selfFrustum || this._frustum.intersectsSphere(c.sphere));
      if (c.obj.visible !== v) c.obj.visible = v;
      if (v) vis++;
    }
    this.stats.visible = vis; this.stats.tested = count;
  }

  _bounds(c) {
    const o = c.opts, obj = c.obj;
    obj.updateWorldMatrix(true, true);
    if (o.center && o.radius) c.sphere.set(c.sphere.center.copy(o.center), o.radius);
    else if (obj.geometry && !obj.isInstancedMesh) {
      if (!obj.geometry.boundingSphere) obj.geometry.computeBoundingSphere();
      c.sphere.copy(obj.geometry.boundingSphere).applyMatrix4(obj.matrixWorld);
    } else {
      _box.setFromObject(obj);
      if (_box.isEmpty()) c.sphere.set(c.sphere.center.setFromMatrixPosition(obj.matrixWorld), 0);
      else _box.getBoundingSphere(c.sphere);
    }
    if (o.radius && !o.center) c.sphere.radius = o.radius;
    c.sphere.radius += this.padding;
  }
  _castsShadow(obj) { let s = false; obj.traverse((q) => { if (q.castShadow) s = true; }); return s; }
}
const _cp = new THREE.Vector3(), _box = new THREE.Box3();
