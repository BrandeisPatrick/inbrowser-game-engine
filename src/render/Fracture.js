// FractureMesh: the visual of a FracturedBody (physics/dynamics/fracture.js): a THREE.Group with one THREE.Mesh per
// piece. Each piece's geometry is built from its collision hull's own topology (shape.points / faceStart / faceVerts /
// planes, i.e. the re-centred body frame, so mesh pose = body pose): polygon faces fan-triangulated, flat normals,
// UVs box-projected from the ORIGINAL shape's frame (1 UV unit per metre, so a texture runs continuously across the
// intact body and the cut faces show the same material inside). Faces on the original surface and cut faces form two
// geometry groups when `interiorMaterial` is given (material index 0 = outside, 1 = inside). (three's ConvexGeometry
// of the same points gives the same solid but re-hulls them and triangulates every face separately.)
//
//   const fm = new FractureMesh(fb, world, material, { interiorMaterial: rockInside, interpolate: true });
//   scene.add(fm.group);
//   after each fixed step:  fm.capture();          (interpolate: previous <- current, current <- the solved pose; the
//                                                   sandbox BodyPools.capture() contract, not InstancedPool.capturePrev(),
//                                                   which runs at frame end for G-buffer velocity)
//   per frame:              fm.update(alpha);      (alpha = the engine's render interpolation factor)
//   after a teleport:       fm.snap();
//
// Without `interpolate`, update() copies world.getPose(id) into each mesh (alpha ignored). Meshes of pieces whose body
// is gone (fb.dispose(), removeBody; a reused id holds another shape) are hidden when the world exposes `bodies`.
// Pieces without surface flags (piece.surface null) draw every face with the outside material. update(), capture() and snap()
// are allocation-free; the constructor allocates. dispose() frees the geometries and detaches the group; the materials
// belong to the caller.
import * as THREE from 'three';

const _p = new Float64Array(3), _q = new Float64Array(4);

/**
 * BufferGeometry of one fracture piece (see the header): positions in the piece's body frame, flat normals, box-projected
 * UVs in the original frame, and (split = true) groups 0 = faces on the original surface, 1 = cut faces.
 * @param {{ shape: object, surface?: Uint8Array|null }} piece  a fracturePieces() entry
 * @param {boolean} [split=false]
 */
export function pieceGeometry(piece, split = false) {
  const s = piece.shape, p = s.points, fs = s.faceStart, fv = s.faceVerts, pl = s.planes, m = s.faceCount;
  const sur = piece.surface || null;
  const so = s.offset; // local = original + so, so original = local - so
  let nv = 0, nt = 0;
  for (let f = 0; f < m; f++) { const k = fs[f + 1] - fs[f]; nv += k; nt += k - 2; }
  const pos = new Float32Array(nv * 3), nor = new Float32Array(nv * 3), uv = new Float32Array(nv * 2);
  const index = nv > 65535 ? new Uint32Array(nt * 3) : new Uint16Array(nt * 3);
  let v = 0, t = 0, outsideTris = 0;
  // two passes when split: original-surface faces first, then the cut faces
  for (let pass = 0; pass < (split ? 2 : 1); pass++) {
    for (let f = 0; f < m; f++) {
      if (split && (sur && sur[f] === 0 ? 1 : 0) !== pass) continue; // no surface flags: everything outside
      const nx = pl[f * 4], ny = pl[f * 4 + 1], nz = pl[f * 4 + 2];
      const ax = Math.abs(nx), ay = Math.abs(ny), az = Math.abs(nz);
      const base = v;
      for (let k = fs[f]; k < fs[f + 1]; k++) {
        const i = fv[k] * 3, x = p[i], y = p[i + 1], z = p[i + 2];
        pos[v * 3] = x; pos[v * 3 + 1] = y; pos[v * 3 + 2] = z;
        nor[v * 3] = nx; nor[v * 3 + 1] = ny; nor[v * 3 + 2] = nz;
        const gx = x - so[0], gy = y - so[1], gz = z - so[2]; // the original frame (= local + piece.offset)
        if (ax >= ay && ax >= az) { uv[v * 2] = nx > 0 ? -gz : gz; uv[v * 2 + 1] = gy; }
        else if (ay >= az) { uv[v * 2] = gx; uv[v * 2 + 1] = ny > 0 ? -gz : gz; }
        else { uv[v * 2] = nz > 0 ? gx : -gx; uv[v * 2 + 1] = gy; }
        v++;
      }
      for (let k = base + 1; k + 1 < v; k++) { index[t++] = base; index[t++] = k; index[t++] = k + 1; }
    }
    if (pass === 0) outsideTris = t / 3;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
  g.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  g.setIndex(new THREE.BufferAttribute(index, 1));
  if (split) { g.addGroup(0, outsideTris * 3, 0); g.addGroup(outsideTris * 3, (nt - outsideTris) * 3, 1); }
  g.computeBoundingSphere();
  return g;
}

export class FractureMesh {
  /**
   * @param {object} fb        FracturedBody (ids, pieces)
   * @param {object} world     anything with getPose(id, outPos, outQuat) (the DynamicsWorld)
   * @param {THREE.Material} material
   * @param {object} [o]
   * @param {boolean} [o.castShadow=true]
   * @param {boolean} [o.receiveShadow=true]
   * @param {boolean} [o.interpolate=false]  render between the last two capture()s
   * @param {THREE.Material|null} [o.interiorMaterial=null]  material of the cut faces (default: `material`)
   */
  constructor(fb, world, material, { castShadow = true, receiveShadow = true, interpolate = false, interiorMaterial = null } = {}) {
    this.fb = fb;
    this.world = world;
    this.interpolate = !!interpolate;
    this.group = new THREE.Group();
    this.group.name = 'fracture';
    this.meshes = [];
    this._p0 = []; this._q0 = []; this._p1 = []; this._q1 = [];
    const split = !!interiorMaterial;
    const mat = split ? [material, interiorMaterial] : material;
    for (let i = 0; i < fb.pieces.length; i++) {
      const mesh = new THREE.Mesh(pieceGeometry(fb.pieces[i], split), mat);
      mesh.name = `fracturePiece${i}`;
      mesh.castShadow = castShadow; mesh.receiveShadow = receiveShadow;
      this.group.add(mesh);
      this.meshes.push(mesh);
      this._p0.push(new THREE.Vector3()); this._q0.push(new THREE.Quaternion());
      this._p1.push(new THREE.Vector3()); this._q1.push(new THREE.Quaternion());
    }
    this.snap();
    this.update(1);
  }

  /** After each fixed step (interpolate): previous <- current, current <- the solved pose. */
  capture() {
    const ids = this.fb.ids, W = this.world;
    for (let i = 0; i < this.meshes.length; i++) {
      if (!this._alive(i)) continue;
      this._p0[i].copy(this._p1[i]); this._q0[i].copy(this._q1[i]);
      W.getPose(ids[i], _p, _q);
      this._p1[i].set(_p[0], _p[1], _p[2]); this._q1[i].set(_q[0], _q[1], _q[2], _q[3]);
    }
  }

  /** previous = current = the world's pose (construction, teleports, scene resets). */
  snap() {
    const ids = this.fb.ids, W = this.world;
    for (let i = 0; i < this.meshes.length; i++) {
      if (!this._alive(i)) continue;
      W.getPose(ids[i], _p, _q);
      this._p1[i].set(_p[0], _p[1], _p[2]); this._q1[i].set(_q[0], _q[1], _q[2], _q[3]);
      this._p0[i].copy(this._p1[i]); this._q0[i].copy(this._q1[i]);
    }
  }

  /** Per frame: mesh poses from the world (interpolate: between the last two captures at alpha). Allocation-free. */
  update(alpha = 1) {
    const ids = this.fb.ids, W = this.world, M = this.meshes;
    for (let i = 0; i < M.length; i++) {
      const mesh = M[i];
      if (!this._alive(i)) { mesh.visible = false; continue; }
      mesh.visible = true;
      if (this.interpolate) {
        mesh.position.lerpVectors(this._p0[i], this._p1[i], alpha);
        mesh.quaternion.slerpQuaternions(this._q0[i], this._q1[i], alpha);
      } else {
        W.getPose(ids[i], _p, _q);
        mesh.position.set(_p[0], _p[1], _p[2]);
        mesh.quaternion.set(_q[0], _q[1], _q[2], _q[3]);
      }
    }
  }

  // piece i's body still exists and is still that piece (ids are reused after removals: compare the shape; worlds
  // without a BodyStore, e.g. a pose source, count as alive)
  _alive(i) {
    if (this.fb.disposed) return false;
    const B = this.world.bodies, id = this.fb.ids[i];
    return !B || !B.alive || (id >= 0 && id < B.count && B.alive[id] === 1 && B.shape[id] === this.fb.pieces[i].shape);
  }

  /** Free the geometries and detach the group (materials are the caller's). */
  dispose() {
    for (let i = 0; i < this.meshes.length; i++) this.meshes[i].geometry.dispose();
    if (this.group.parent) this.group.parent.remove(this.group);
    this.group.clear();
    this.meshes.length = 0;
  }
}
