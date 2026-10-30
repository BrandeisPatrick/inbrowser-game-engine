// ThirdPersonRig — over-the-shoulder character camera: mouse / stick look around a focus above the target,
// critically-damped follow, zoom, collision pull-in, ground clamp, optional auto-recentre.
//
//   const rig = new ThirdPersonRig(camera, canvas, { physics: engine.physics, distance: 4, shoulder: 0.4 });  // canvas may be null
//   rig.setTarget(player.object3d, () => player.state);   // state (optional): { facing | heading } radians
//   rig.look(mouse.dx, mouse.dy)        // pixels: x sensitivity, invertY applied here
//   rig.turn(-lookX, -lookY)            // radians, already scaled (+dyaw turns left, +dpitch looks up)
//   rig.update(dt);  rig.getForward(v); rig.getRight(v)   // yaw-only basis for camera-relative movement
//   rig.zoom(1.1); rig.snap(); rig.recentre(); rig.state  // { yaw, pitch, distance } (same object every frame)
//
// Angles (same as FirstPersonRig / FreeFlyRig): yaw 0 looks down -Z, positive turns left (counter-clockwise from
// above, the engine's heading convention); pitch > 0 looks up. The camera looks exactly along (yaw, pitch) (screen
// centre = aim) from pivot - forward * distance, pivot = focus + right * shoulder, focus = target position +
// (0, height, 0) followed by a critically-damped spring (`damping` = natural frequency in 1/s; follow lag ~2 v / damping;
// sub-stepped at max(120, 4 * damping) Hz so it stays stable and overshoot-free for any damping).
// setTarget() turns the rig behind the target (its facing: state.facing, else state.heading, else local -Z);
// autoRecentre (seconds, 0 = off) eases yaw back to the facing after that long without look input.
// Collision (collide, with physics / queryColliders / raycast / getGroundHeight): the focus -> camera segment is cast
// against the colliders near the target (CameraCollision gather + segHit, as in ChaseRig). The camera is pulled in to the first hit minus collisionMargin (instantly,
// never closer than minDistance) and eases back out, then is pushed out of boxes (near plane). With a ground
// function the camera never goes below ground + minY (also when collide is false). Input is external by default:
// pointerLock: true makes the rig lock the pointer on click and read mousemove itself; with a domElement the wheel
// zooms (wheelZoom). look() / turn() are ignored while inactive (and for NaN deltas); activate() snaps. No allocations per frame (apart from what
// queryColliders returns).
import * as THREE from 'three';
import { CameraRig, clamp, damp, wrapPi, springStep } from './CameraRig.js';
import { bindLook, targetPosition, targetFacing } from './FirstPersonRig.js';

const EMPTY = Object.freeze({});
const SPRING_HZ = 120;   // minimum focus-spring sub-step rate (also >= 4 * damping: stable, monotonic, frame-rate independent)
const _tpos = new THREE.Vector3();
const _goal = new THREE.Vector3();
const _desired = new THREE.Vector3();
const _tmp = new THREE.Vector3();
const _euler = new THREE.Euler(0, 0, 0, 'YXZ');

export class ThirdPersonRig extends CameraRig {
  /**
   * @param {THREE.PerspectiveCamera} camera
   * @param {HTMLElement|null} [domElement]  wheel zoom / pointer lock surface (no DOM input without one)
   * @param {object} [o]
   * @param {object} [o.physics=null]        PhysicsWorld-like (getGroundHeight / queryColliders / raycast); CameraRig
   *                                         / CameraCollision options (minY, canopyFromCollider, thinKinds...) also accepted
   * @param {number} [o.distance=4] [o.height=1.6] [o.shoulder=0.4]  metres (shoulder > 0 = camera right of the target)
   * @param {number[]} [o.pitch=[-1.2, 1.0]] pitch range (radians, > 0 looks up)
   * @param {number} [o.sensitivity=0.0025]  radians per look() unit (pixel)
   * @param {number} [o.damping=12]          follow spring natural frequency (1/s); also the zoom easing rate
   * @param {boolean} [o.collide=true] [o.active=true] [o.invertY=false] [o.wheelZoom=true] [o.pointerLock=false]
   * @param {number} [o.minDistance=0.6]     closest the collision pull-in brings the camera to the focus
   * @param {number[]} [o.zoom=[1.5, 8]]     distance range for zoom() / the wheel
   * @param {number} [o.autoRecentre=0]      seconds without look input before yaw eases to the target's facing (0 = off)
   * @param {number} [o.recentreRate=2.5]    easing rate of the auto-recentre (1/s)
   * @param {number} [o.collisionMargin=0.25] [o.collisionRecover=4]  stand-off from a hit (m), ease-out rate (1/s)
   */
  constructor(camera, domElement, o = {}) {
    super(camera, o);
    this.domElement = domElement || null;
    this.distance = o.distance ?? 4;
    this.height = o.height ?? 1.6;
    this.shoulder = o.shoulder ?? 0.4;
    const P = o.pitch || [-1.2, 1.0], Z = o.zoom || [1.5, 8];
    this.minPitch = P[0]; this.maxPitch = P[1];
    this.minZoom = Z[0]; this.maxZoom = Z[1];
    this.sensitivity = o.sensitivity ?? 0.0025;
    this.damping = o.damping ?? 12;
    this.collide = o.collide !== false;
    this.invertY = !!o.invertY;
    this.minDistance = o.minDistance ?? 0.6;
    this.wheelZoom = o.wheelZoom !== false;
    this.autoRecentre = o.autoRecentre || 0;
    this.recentreRate = o.recentreRate ?? 2.5;
    this.margin = o.collisionMargin ?? 0.25;
    this.recover = o.collisionRecover ?? 4;
    this.yaw = 0;
    this.pitch = clamp(-0.3, this.minPitch, this.maxPitch);
    this.focus = new THREE.Vector3();          // smoothed focus point (world)
    this.state = { yaw: 0, pitch: this.pitch, distance: this.distance };
    this._focusVel = new THREE.Vector3();
    this._dist = this.distance;                // eased zoom distance
    this._occ = 1;                             // allowed fraction of the focus -> camera segment
    this._idle = 0;                            // seconds since the last look input
    this._unbind = bindLook(this, this.domElement, !!o.pointerLock, (e) => {
      if (!this.wheelZoom) return;
      e.preventDefault();
      this.zoom(Math.exp((e.deltaY || 0) * 0.001));
    });
  }

  /** Feed look deltas in pixels (x sensitivity; mouse right / down positive, invertY flips y); ignored while inactive. */
  look(dx, dy) { const k = this.sensitivity; this.turn(-dx * k, (this.invertY ? dy : -dy) * k); }

  /** Rotate by radians: +dyaw turns left, +dpitch looks up (pitch clamped); ignored while inactive or for NaN. */
  turn(dyaw, dpitch) {
    if (!this.enabled || dyaw !== dyaw || dpitch !== dpitch) return;
    this.yaw = wrapPi(this.yaw + dyaw);
    this.pitch = clamp(this.pitch + dpitch, this.minPitch, this.maxPitch);
    if (dyaw || dpitch) this._idle = 0;
  }

  /** Multiply the zoom distance (> 1 = farther), clamped to the zoom range. */
  zoom(factor) { this.distance = clamp(this.distance * factor, this.minZoom, this.maxZoom); }

  /** Turn behind the target (yaw = its facing) instantly. */
  recentre() { if (this.target) this.yaw = targetFacing(this.target, this.getState && this.getState(), this.yaw); }

  /** Follow an Object3D; also turns the rig behind it. */
  setTarget(object3d, stateGetter) { super.setTarget(object3d, stateGetter); this.recentre(); }

  /** Place the camera now (when active with a target), else on the next update. */
  snap() { super.snap(); if (this.enabled && this.target) this.update(0); }
  activate() { super.activate(); this.snap(); }

  /** Horizontal unit forward of the view yaw (for camera-relative movement). */
  getForward(out) { return out.set(-Math.sin(this.yaw), 0, -Math.cos(this.yaw)); }
  /** Horizontal unit right of the view yaw. */
  getRight(out) { return out.set(Math.cos(this.yaw), 0, -Math.sin(this.yaw)); }

  update(dt) {
    if (!this.enabled || !this.target) return;
    dt = clamp(dt || 0, 0, 0.1);
    const cam = this.camera, C = this.collision, snap = this._snapPending;
    const st = (this.getState && this.getState()) || EMPTY;
    this.pitch = clamp(this.pitch, this.minPitch, this.maxPitch);
    this.distance = clamp(this.distance, this.minZoom, this.maxZoom);
    this._idle += dt;
    if (this.autoRecentre > 0 && this._idle >= this.autoRecentre && !snap) {
      this.yaw += wrapPi(targetFacing(this.target, st, this.yaw) - this.yaw) * damp(this.recentreRate, dt);
    }
    this.yaw = wrapPi(this.yaw);

    // focus: critically-damped spring toward target + height, in fixed sub-steps
    targetPosition(this.target, _tpos);
    _goal.copy(_tpos); _goal.y += this.height;
    if (snap) {
      this.focus.copy(_goal); this._focusVel.set(0, 0, 0);
      this._dist = this.distance; this._occ = 1;
    } else {
      const hz = Math.max(SPRING_HZ, 4 * this.damping), k = this.damping * this.damping;
      const n = Math.ceil(dt * hz - 1e-6), h = n > 0 ? dt / n : 0;
      for (let i = 0; i < n; i++) springStep(this.focus, this._focusVel, _goal, k, 1, h, _tmp);
      this._dist += (this.distance - this._dist) * damp(this.damping, dt);
    }

    // orientation is exactly (yaw, pitch); desired = focus + right * shoulder - forward * distance
    _euler.set(this.pitch, this.yaw, 0, 'YXZ');
    cam.quaternion.setFromEuler(_euler);
    cam.up.set(0, 1, 0);
    const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch), sy = Math.sin(this.yaw), cy = Math.cos(this.yaw);
    const d = this._dist, s = this.shoulder;
    _desired.set(cy * s + sy * cp * d, -sp * d, -sy * s + cy * cp * d).add(this.focus);

    // collision: pull in along focus -> desired (instantly), ease back out
    let f = 1;
    if (this.collide) {
      C.gather(_tpos, d + Math.abs(s) + 2);
      const len = Math.hypot(d, s);
      const t = C.segHit(this.focus, _desired, 0.15);
      const allowed = t < 1 && len > 1e-6 ? clamp((t * len - this.margin) / len, Math.min(1, this.minDistance / len), 1) : 1;
      if (snap || allowed < this._occ) this._occ = allowed;
      else this._occ += (allowed - this._occ) * damp(this.recover, dt);
      f = this._occ;
    } else C.focus.copy(_tpos);
    cam.position.copy(_desired).sub(this.focus).multiplyScalar(f).add(this.focus);
    if (this.collide) C.pushOut(cam.position);
    if (C.getGroundHeight) C.clampAboveGround(cam.position, this.minY);

    const S = this.state; S.yaw = this.yaw; S.pitch = this.pitch; S.distance = d * f;
    this._snapPending = false;
  }

  dispose() { if (this._unbind) this._unbind(); this._unbind = null; super.dispose(); }
}

export default ThirdPersonRig;
