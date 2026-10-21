// Per-shadow-pass caster control without touching materials or the shadow renderer.
//
// three.js renders every castShadow mesh into every shadow map (each cascade), testing the caster's layers
// against the MAIN camera, so layers cannot select casters per cascade. These helpers use the per-object
// onBeforeShadow / onAfterShadow hooks instead (called around each shadow draw with that pass's shadow camera):
//
//   shadowCount(mesh, (shadowCamera, mainCount) => n)   InstancedMesh / InstancedBufferGeometry: draw only the
//                                                       first n instances in that shadow pass (0 = no draw call)
//   shadowDrawRange(mesh, (shadowCamera) => n)          plain mesh: draw only its first n indices in that pass
//   skipShadowPasses(mesh, (shadowCamera) => bool)      any mesh: skip it in the passes where the test is true
//   shadowFrustum(light)                                the Frustum of a light's shadow camera for this frame
//                                                       (valid as soon as the light's matrices are updated)
//
// Convention (see ENGINE.md, Conventions): objects that should cast only into the near cascade are tagged with
// layer NEAR_SHADOW_LAYER (1) in addition to their normal layer 0, and get skipShadowPasses(mesh, isFarCascade).
import * as THREE from 'three';

export const NEAR_SHADOW_LAYER = 1;

const instanced = (m) => m.isInstancedMesh ? 'mesh' : m.geometry && m.geometry.isInstancedBufferGeometry ? 'geo' : null;

/**
 * Different instance counts per shadow pass. Instances must be ordered so each pass's casters are a prefix; a pass
 * may draw MORE instances than the main pass (casters outside the view): an InstancedMesh is clamped to its
 * capacity, an InstancedBufferGeometry to its instanceCount.
 * @param {THREE.Mesh} mesh  InstancedMesh, or a Mesh with an InstancedBufferGeometry
 * @param {(shadowCamera: THREE.Camera, mainCount: number) => number} countFor
 */
export function shadowCount(mesh, countFor) {
  const kind = instanced(mesh);
  if (!kind) throw new Error('shadowCount: not an instanced mesh');
  let saved = 0;
  const before = mesh.onBeforeShadow, after = mesh.onAfterShadow;
  mesh.onBeforeShadow = function (renderer, object, camera, shadowCamera, geometry, depthMaterial, group) {
    before.call(this, renderer, object, camera, shadowCamera, geometry, depthMaterial, group);
    if (kind === 'mesh') { saved = this.count; this.count = Math.min(this.instanceMatrix.count, Math.max(0, countFor(shadowCamera, saved) | 0)); }
    else { saved = geometry.instanceCount; geometry.instanceCount = Math.min(saved, Math.max(0, countFor(shadowCamera, saved) | 0)); }
  };
  mesh.onAfterShadow = function (renderer, object, camera, shadowCamera, geometry, depthMaterial, group) {
    if (kind === 'mesh') this.count = saved; else geometry.instanceCount = saved;
    after.call(this, renderer, object, camera, shadowCamera, geometry, depthMaterial, group);
  };
  return mesh;
}

/**
 * Draw only the first `count` indices (vertices when non-indexed) of a plain mesh in a shadow pass: geometry
 * ordered [important triangles | detail triangles] casts its detail only where it matters. countFor returns
 * Infinity (or >= the full count) to draw everything, 0 to skip (an empty draw: a few microseconds of state
 * setup, no GPU work).
 * @param {THREE.Mesh} mesh
 * @param {(shadowCamera: THREE.Camera, mesh: THREE.Mesh) => number} countFor
 */
export function shadowDrawRange(mesh, countFor) {
  if (mesh.isBatchedMesh) throw new Error('shadowDrawRange: BatchedMesh draws from its own multi-draw list (not drawRange)');
  let saved = -1;
  const before = mesh.onBeforeShadow, after = mesh.onAfterShadow;
  mesh.onBeforeShadow = function (renderer, object, camera, shadowCamera, geometry, depthMaterial, group) {
    before.call(this, renderer, object, camera, shadowCamera, geometry, depthMaterial, group);
    saved = -1;
    const n = countFor(shadowCamera, this);
    if (n < geometry.drawRange.count) { saved = geometry.drawRange.count; geometry.drawRange.count = Math.max(0, n); }
  };
  mesh.onAfterShadow = function (renderer, object, camera, shadowCamera, geometry, depthMaterial, group) {
    if (saved !== -1) { geometry.drawRange.count = saved; saved = -1; }
    after.call(this, renderer, object, camera, shadowCamera, geometry, depthMaterial, group);
  };
  return mesh;
}

/**
 * Skip a caster in selected shadow passes (instanced: no draw call; plain: an empty draw, see shadowDrawRange;
 * not for BatchedMesh).
 * Tags the mesh with NEAR_SHADOW_LAYER when used for the far cascade (see the convention above).
 * @param {THREE.Mesh} mesh
 * @param {(shadowCamera: THREE.Camera) => boolean} skip
 */
export function skipShadowPasses(mesh, skip) {
  mesh.layers.enable(NEAR_SHADOW_LAYER);
  if (instanced(mesh)) return shadowCount(mesh, (cam, n) => (skip(cam) ? 0 : n));
  return shadowDrawRange(mesh, (cam) => (skip(cam) ? 0 : Infinity));
}

const _lp = new THREE.Vector3(), _lt = new THREE.Vector3(), _pv = new THREE.Matrix4(), _scratch = new WeakMap();
/**
 * Frustum of a directional / spot light's shadow camera from the light's current world matrices (what the next
 * shadow render will use). Computed on a scratch camera: the light's own shadow camera / matrix are not touched,
 * so a cascade that is re-rendered only every other frame keeps the matrix its map was rendered with.
 */
export function shadowFrustum(light, out = new THREE.Frustum()) {
  if (!light || !light.shadow || !light.shadow.camera) return null;
  const sc = light.shadow.camera;
  let c = _scratch.get(sc);
  if (!c) { c = new THREE.Camera(); _scratch.set(sc, c); }
  _lp.setFromMatrixPosition(light.matrixWorld);
  _lt.setFromMatrixPosition(light.target ? light.target.matrixWorld : light.matrixWorld);
  c.position.copy(_lp);
  c.up.copy(sc.up); c.lookAt(_lt); // same rule as LightShadow.updateMatrices
  c.updateMatrixWorld(true);
  _pv.copy(c.matrixWorld).invert().premultiply(sc.projectionMatrix);
  return out.setFromProjectionMatrix(_pv, sc.coordinateSystem);
}
