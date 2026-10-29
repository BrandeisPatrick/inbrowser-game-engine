// FirstPersonRig — character eye camera: mouse / stick look from target position + eyeHeight, optional head bob.
//
//   const rig = new FirstPersonRig(camera, canvas, { eyeHeight: 1.65, bob: 0.04 });   // domElement may be null
//   rig.setTarget(player.object3d, () => player.state);   // state (optional): { speed | velocity, grounded, facing | heading }
//   rig.look(mouse.dx, mouse.dy)        // pixels: x sensitivity, invertY applied here
//   rig.turn(-lookX, -lookY)            // radians, already scaled (+dyaw turns left, +dpitch looks up)
//   rig.update(dt);  rig.getForward(v); rig.getRight(v)   // yaw-only basis for camera-relative movement
//   rig.snap(); rig.recentre(); rig.state                 // { yaw, pitch, distance: 0 } (same object every frame)
//
// Angles (same as ThirdPersonRig / FreeFlyRig): yaw 0 looks down -Z, positive turns left (counter-clockwise from
// above, the engine's heading convention); pitch > 0 looks up. setTarget() turns the rig to the target's facing
// (state.facing, else state.heading, else the target's local -Z). Head bob (bob > 0, metres): one dip of up to `bob`
// per step at bobFreq steps/s (scaled with speed / bobSpeed), half-amplitude sideways sway, faded in by the state's
// horizontal speed (state.speed, else |velocity.xz|) while state.grounded !== false. Eye-height changes (crouch)
// ease in. With a ground function (physics / getGroundHeight) the eye stays >= ground + minY. Input is external by
// default; pointerLock: true makes the rig lock the pointer on click and read mousemove itself. look() / turn() are
// ignored while inactive (and for NaN deltas); activate() snaps. The bob phase restarts once the bob has faded out,
// so every walk starts on the same stride. No allocations per frame.
//
// Also exports the small helpers ThirdPersonRig shares: bindLook, targetPosition, targetFacing.
import * as THREE from 'three';
import { CameraRig, clamp, smooth01, damp, wrapPi } from './CameraRig.js';

const EMPTY = Object.freeze({});
const _q = new THREE.Quaternion();
const _tpos = new THREE.Vector3();
const _v = new THREE.Vector3();
const _euler = new THREE.Euler(0, 0, 0, 'YXZ');

/** World position of a rig target (Object3D, Vector3 or {x, y, z}) into out. */
export function targetPosition(t, out) {
  if (t.isVector3) return out.copy(t);
  if (t.getWorldPosition) return t.getWorldPosition(out);
  return out.set(t.x || 0, t.y || 0, t.z || 0);
}

/** Facing yaw of a target: state.facing, else state.heading, else its local -Z (0 = -Z, + = left); fallback if none. */
export function targetFacing(t, st, fallback) {
  if (st && Number.isFinite(st.facing)) return st.facing;
  if (st && Number.isFinite(st.heading)) return st.heading;
  if (!t || !t.getWorldQuaternion) return fallback;
  _v.set(0, 0, -1).applyQuaternion(t.getWorldQuaternion(_q));
  return _v.x * _v.x + _v.z * _v.z < 1e-8 ? fallback : Math.atan2(-_v.x, -_v.z);
}

/**
 * Optional DOM look input for a rig with look(dx, dy) and enabled: with pointerLock, a click locks the pointer and
 * mousemove while locked feeds rig.look(); onWheel(e) receives wheel events (it calls preventDefault itself when it
 * uses one). Returns an unbind function, or null without an element.
 */
export function bindLook(rig, el, pointerLock, onWheel) {
  if (!el || !el.addEventListener) return null;
  const doc = () => el.ownerDocument || globalThis.document || null;
  const L = {};
  if (pointerLock) {
    L.click = () => {
      const d = doc();
      if (!rig.enabled || !el.requestPointerLock || (d && d.pointerLockElement === el)) return;
      const p = el.requestPointerLock();
      if (p && p.catch) p.catch(() => {});   // rejected without a user gesture / right after an exit
    };
    L.mousemove = (e) => {
      const d = doc();
      if (rig.enabled && d && d.pointerLockElement === el) rig.look(e.movementX || 0, e.movementY || 0);
    };
  }
  if (onWheel) L.wheel = (e) => { if (rig.enabled) onWheel(e); };
  for (const k in L) el.addEventListener(k, L[k], k === 'wheel' ? { passive: false } : undefined);
  return () => { for (const k in L) el.removeEventListener(k, L[k]); };
}

export class FirstPersonRig extends CameraRig {
  /**
   * @param {THREE.PerspectiveCamera} camera
   * @param {HTMLElement|null} [domElement]  only used with pointerLock: true
   * @param {object} [o]
   * @param {object} [o.physics=null]       ground clamp (getGroundHeight); CameraRig options also accepted
   * @param {number} [o.eyeHeight=1.65]     metres above the target position
   * @param {number[]} [o.pitch=[-1.5, 1.5]] pitch range (radians, > 0 looks up)
   * @param {number} [o.sensitivity=0.0025] radians per look() unit (pixel)
   * @param {number} [o.bob=0]              head-bob dip in metres (0 = off)
   * @param {number} [o.bobFreq=2]          steps per second at bobSpeed
   * @param {number} [o.bobSpeed=4]         horizontal speed (m/s) of full bob
   * @param {boolean} [o.active=true] [o.invertY=false] [o.pointerLock=false]
   */
  constructor(camera, domElement, o = {}) {
    super(camera, o);
    this.domElement = domElement || null;
    this.eyeHeight = o.eyeHeight ?? 1.65;
    const P = o.pitch || [-1.5, 1.5];
    this.minPitch = P[0]; this.maxPitch = P[1];
    this.sensitivity = o.sensitivity ?? 0.0025;
    this.bob = o.bob || 0;
    this.bobFreq = o.bobFreq ?? 2;
    this.bobSpeed = o.bobSpeed ?? 4;
    this.invertY = !!o.invertY;
    this.yaw = 0;
    this.pitch = clamp(0, this.minPitch, this.maxPitch);
    this.state = { yaw: 0, pitch: this.pitch, distance: 0 };
    this._eye = this.eyeHeight;
    this._bobW = 0;
    this._bobPhase = 0;
    this._unbind = bindLook(this, this.domElement, !!o.pointerLock, null);
  }

  /** Feed look deltas in pixels (x sensitivity; mouse right / down positive, invertY flips y); ignored while inactive. */
  look(dx, dy) { const k = this.sensitivity; this.turn(-dx * k, (this.invertY ? dy : -dy) * k); }

  /** Rotate by radians: +dyaw turns left, +dpitch looks up (pitch clamped); ignored while inactive or for NaN. */
  turn(dyaw, dpitch) {
    if (!this.enabled || dyaw !== dyaw || dpitch !== dpitch) return;
    this.yaw = wrapPi(this.yaw + dyaw);
    this.pitch = clamp(this.pitch + dpitch, this.minPitch, this.maxPitch);
  }

  /** Turn to the target's facing. */
  recentre() { if (this.target) this.yaw = targetFacing(this.target, this.getState && this.getState(), this.yaw); }

  /** Follow an Object3D; also turns the view to its facing. */
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
    const cam = this.camera, snap = this._snapPending;
    const st = (this.getState && this.getState()) || EMPTY;
    this.yaw = wrapPi(this.yaw);
    this.pitch = clamp(this.pitch, this.minPitch, this.maxPitch);

    _euler.set(this.pitch, this.yaw, 0, 'YXZ');
    cam.quaternion.setFromEuler(_euler);
    cam.up.set(0, 1, 0);
    this._eye = snap ? this.eyeHeight : this._eye + (this.eyeHeight - this._eye) * damp(12, dt);
    targetPosition(this.target, _tpos);
    cam.position.copy(_tpos);
    cam.position.y += this._eye;

    // head bob: dip per step + half-rate sway, faded by grounded horizontal speed
    if (this.bob > 0) {
      const v = st.velocity;
      const speed = Number.isFinite(st.speed) ? Math.abs(st.speed) : v ? Math.hypot(v.x, v.z) : 0;
      const w = st.grounded === false ? 0 : smooth01(0.2, this.bobSpeed, speed);
      if (snap) { this._bobW = w; this._bobPhase = 0; } else this._bobW += (w - this._bobW) * damp(8, dt);
      if (this._bobW > 1e-3) this._bobPhase = (this._bobPhase + Math.PI * this.bobFreq * clamp(speed / this.bobSpeed, 0.6, 1.5) * dt) % (2 * Math.PI);
      else this._bobPhase = 0;   // faded out: the next walk starts on the same stride
      const a = this.bob * this._bobW, s = Math.sin(this._bobPhase), side = 0.5 * a * s;
      cam.position.y -= a * s * s;                 // = a (1 - cos 2φ) / 2: one dip per step, sway once per stride
      cam.position.x += Math.cos(this.yaw) * side;
      cam.position.z -= Math.sin(this.yaw) * side;
    } else this._bobW = 0;

    const C = this.collision;
    if (C.getGroundHeight) { C.focus.copy(_tpos); C.clampAboveGround(cam.position, this.minY); }
    const s = this.state; s.yaw = this.yaw; s.pitch = this.pitch; s.distance = 0;
    this._snapPending = false;
  }

  dispose() { if (this._unbind) this._unbind(); this._unbind = null; super.dispose(); }
}

export default FirstPersonRig;
