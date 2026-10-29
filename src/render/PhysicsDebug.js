// PhysicsDebug: wireframe overlay of a dynamics world in one draw call — body shapes coloured by state (static blue,
// kinematic yellow, sleeping grey, awake green), contact normals (red), joint anchors (magenta), optional AABBs.
// One LineSegments over a fixed-capacity BufferGeometry (Float32 position + colour, DynamicDraw); update() rewrites
// the segments with physics/dynamics/debug.js writeDebugLines and uploads only the written range. Drawn on top
// (no depth test). Allocation-free per frame.
//
//   const dbg = new PhysicsDebug(dynamicsWorld, { maxLines: 32768, aabbs: false });
//   scene.add(dbg);
//   per frame, AFTER the physics step: dbg.update();
//   dbg.enabled = false;   // hides it (update() then does nothing)
import * as THREE from 'three';
import { touchRange, newRange } from './bufferRanges.js';
import { writeDebugLines } from '../physics/dynamics/debug.js';

export class PhysicsDebug extends THREE.LineSegments {
  /**
   * @param {object} world   DynamicsWorld (bodies, contacts, joints)
   * @param {object} [o]
   * @param {number} [o.maxLines=65536]  segment capacity (2 vertices each)
   * @param {boolean} [o.bodies=true]    shapes
   * @param {boolean} [o.contacts=true]  contact normals
   * @param {boolean} [o.joints=true]    joint anchors
   * @param {boolean} [o.aabbs=false]    world AABBs
   * @param {boolean} [o.colorByState=true]
   * @param {number} [o.renderOrder=1000]
   */
  constructor(world, { maxLines = 65536, renderOrder = 1000, ...options } = {}) {
    const max = Math.max(1, maxLines | 0);
    const g = new THREE.BufferGeometry();
    const pos = new THREE.BufferAttribute(new Float32Array(max * 6), 3).setUsage(THREE.DynamicDrawUsage);
    const col = new THREE.BufferAttribute(new Float32Array(max * 6), 3).setUsage(THREE.DynamicDrawUsage);
    g.setAttribute('position', pos);
    g.setAttribute('color', col);
    g.setDrawRange(0, 0);
    const m = new THREE.LineBasicMaterial({ vertexColors: true, depthTest: false, depthWrite: false, transparent: true, opacity: 0.9, toneMapped: false });
    super(g, m);
    this.name = 'physicsDebug';
    this.world = world;
    this.maxLines = max;
    this.options = { bodies: true, contacts: true, joints: true, aabbs: false, colorByState: true, ...options };
    this.count = 0; // segments written by the last update()
    this.renderOrder = renderOrder;
    this.frustumCulled = false;
    this._pos = pos; this._col = col;
    this._rp = newRange(); this._rc = newRange();
  }

  /** Visibility toggle; while disabled update() skips the work. */
  get enabled() { return this.visible; }
  set enabled(v) { this.visible = !!v; }

  /** Rewrite the lines from the world's current state; call once per frame after the physics step. Returns the segment count. */
  update() {
    if (!this.visible || !this.world) return this.count;
    const n = writeDebugLines(this.world, this._pos.array, this.maxLines, this.options, this._col.array);
    if (n > 0) {
      touchRange(this._pos, this._rp, 0, n * 6);
      touchRange(this._col, this._rc, 0, n * 6);
    }
    this.geometry.setDrawRange(0, n * 2);
    this.count = n;
    return n;
  }

  /** Debug geometry is not pickable (scene raycasts skip it). */
  raycast() {}

  dispose() {
    this.geometry.dispose();
    this.material.dispose();
  }
}
