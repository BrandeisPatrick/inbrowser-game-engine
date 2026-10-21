// FreeFlyRig — WASD fly-through camera for debugging.
//   W/S forward/back · A/D strafe · E/Space up · Q/C down · Shift = fast (×boost) · Alt = slow
//   drag (any button) to look, or click to pointer-lock with { pointerLock: true } · wheel = speed
//   const rig = new FreeFlyRig(camera, canvas, { speed: 30 }); rig.update(dt);
//   const rig = new FreeFlyRig(camera, canvas, { input: vehicleInput.map });   // keys through an InputMap
// Raw-key mode (no `input`): keys are only captured while the rig is enabled. InputMap mode: the rig defines
// always-on actions (`<actionPrefix>x|y|z|fast|slow`, default prefix 'freeFly.') on the given map unless they
// exist already (rebind them like any action), so they keep working while the game sets `map.enabled = false`
// to silence its own controls; the map's owner calls map.update() each frame as usual. (InputMap ignores keys
// pressed while Alt is held, so in that mode press Alt after the move key for slow flight; the wheel also works.)
import * as THREE from 'three';
import { CameraRig, clamp, damp } from './CameraRig.js';

const KEYS = new Set(['KeyW', 'KeyA', 'KeyS', 'KeyD', 'KeyQ', 'KeyE', 'KeyC', 'Space', 'ShiftLeft', 'ShiftRight', 'AltLeft', 'AltRight']);

/** Default InputMap actions of the free-fly rig (always-on). */
export function freeFlyActions(prefix = 'freeFly.') {
  return {
    [`${prefix}x`]: { type: 'axis', alwaysOn: true, bindings: [{ key: 'KeyD', value: 1 }, { key: 'KeyA', value: -1 }] },
    [`${prefix}y`]: { type: 'axis', alwaysOn: true, bindings: [{ key: 'KeyE', value: 1 }, { key: 'Space', value: 1 }, { key: 'KeyQ', value: -1 }, { key: 'KeyC', value: -1 }] },
    [`${prefix}z`]: { type: 'axis', alwaysOn: true, bindings: [{ key: 'KeyS', value: 1 }, { key: 'KeyW', value: -1 }] },
    [`${prefix}fast`]: { type: 'button', alwaysOn: true, bindings: ['ShiftLeft', 'ShiftRight'] },
    [`${prefix}slow`]: { type: 'button', alwaysOn: true, bindings: ['AltLeft', 'AltRight'] },
  };
}

export class FreeFlyRig extends CameraRig {
  /**
   * @param {THREE.PerspectiveCamera} camera
   * @param {HTMLElement} [domElement]
   * @param {object} [o]  speed (m/s, 20), boost (4), lookSpeed (0.0025 rad/px), damping (8), pointerLock (false),
   *                      collide (false: clamp above ground using getGroundHeight / physics), active (true),
   *                      input (InputMap: keys through always-on actions instead of raw listeners),
   *                      actionPrefix ('freeFly.')
   */
  constructor(camera, domElement, o = {}) {
    super(camera, o);
    this.domElement = domElement || null;
    this.input = o.input || null;
    if (this.input) {
      const P = o.actionPrefix ?? 'freeFly.', defs = freeFlyActions(P);
      for (const k of Object.keys(defs)) if (!this.input.actions.has(k)) this.input.define(k, defs[k]);
      this._act = { x: `${P}x`, y: `${P}y`, z: `${P}z`, fast: `${P}fast`, slow: `${P}slow` };
    }
    this.speed = o.speed ?? 20;
    this.boost = o.boost ?? 4;
    this.lookSpeed = o.lookSpeed ?? 0.0025;
    this.damping = o.damping ?? 8;
    this.pointerLock = !!o.pointerLock;
    this.collide = !!o.collide;
    this.yaw = 0; this.pitch = 0;
    this.velocity = new THREE.Vector3();
    this._keys = new Set();
    this._wish = new THREE.Vector3();
    this._euler = new THREE.Euler(0, 0, 0, 'YXZ');
    this._drag = null;
    this.syncFromCamera();
    this._bind();
  }

  /** Adopt the camera's current orientation (call after placing the camera by hand). */
  syncFromCamera() {
    this._euler.setFromQuaternion(this.camera.quaternion, 'YXZ');
    this.yaw = this._euler.y; this.pitch = this._euler.x;
  }

  activate() { super.activate(); this.syncFromCamera(); this.velocity.set(0, 0, 0); }
  deactivate() { super.deactivate(); this._keys.clear(); }

  _bind() {
    if (typeof window === 'undefined') return;
    if (!this.input) this._bindKeys();
    this._bindPointer();
  }

  _bindKeys() {
    this._onKeyDown = (e) => {
      if (!this.enabled || e.metaKey || e.ctrlKey) return;
      const t = e.target && e.target.tagName;
      if (t === 'INPUT' || t === 'TEXTAREA' || t === 'SELECT') return;
      if (KEYS.has(e.code)) { this._keys.add(e.code); if (e.code === 'Space') e.preventDefault(); }
    };
    this._onKeyUp = (e) => this._keys.delete(e.code);
    this._onBlur = () => this._keys.clear();
    window.addEventListener('keydown', this._onKeyDown);
    window.addEventListener('keyup', this._onKeyUp);
    window.addEventListener('blur', this._onBlur);
  }

  _bindPointer() {
    const el = this.domElement;
    if (!el) return;
    this._onDown = (e) => {
      if (!this.enabled) return;
      if (this.pointerLock && el.requestPointerLock) { el.requestPointerLock(); return; }
      this._drag = { x: e.clientX, y: e.clientY, id: e.pointerId };
      if (el.setPointerCapture) try { el.setPointerCapture(e.pointerId); } catch { /* synthetic */ }
    };
    this._onMove = (e) => {
      if (!this.enabled) return;
      if (this.pointerLock && document.pointerLockElement === el) { this.look(e.movementX, e.movementY); return; }
      const d = this._drag;
      if (!d || d.id !== e.pointerId) return;
      this.look(e.clientX - d.x, e.clientY - d.y);
      d.x = e.clientX; d.y = e.clientY;
    };
    this._onUp = () => { this._drag = null; };
    this._onWheel = (e) => { if (!this.enabled) return; e.preventDefault(); this.speed = clamp(this.speed * Math.exp(-e.deltaY * 0.001), 0.5, 2000); };
    el.addEventListener('pointerdown', this._onDown);
    el.addEventListener('pointermove', this._onMove);
    el.addEventListener('pointerup', this._onUp);
    el.addEventListener('pointercancel', this._onUp);
    el.addEventListener('wheel', this._onWheel, { passive: false });
  }

  look(dxPx, dyPx) {
    this.yaw -= dxPx * this.lookSpeed;
    this.pitch = clamp(this.pitch - dyPx * this.lookSpeed, -1.55, 1.55);
  }

  update(dt) {
    if (!this.enabled) return;
    dt = clamp(dt || 0, 0, 0.1);
    let fast, slow;
    const w = this._wish;
    if (this.input) {
      const I = this.input, A = this._act;
      w.set(I.value(A.x), I.value(A.y), I.value(A.z));
      fast = I.down(A.fast); slow = I.down(A.slow);
    } else {
      const k = this._keys, has = (c) => k.has(c);
      w.set(
        (has('KeyD') ? 1 : 0) - (has('KeyA') ? 1 : 0),
        (has('KeyE') || has('Space') ? 1 : 0) - (has('KeyQ') || has('KeyC') ? 1 : 0),
        (has('KeyS') ? 1 : 0) - (has('KeyW') ? 1 : 0),
      );
      fast = has('ShiftLeft') || has('ShiftRight'); slow = has('AltLeft') || has('AltRight');
    }
    if (w.lengthSq() > 1) w.normalize();
    const sp = this.speed * (fast ? this.boost : 1) * (slow ? 0.2 : 1);
    this._euler.set(this.pitch, this.yaw, 0, 'YXZ');
    const cam = this.camera;
    cam.quaternion.setFromEuler(this._euler);
    cam.up.set(0, 1, 0);
    // horizontal moves follow the view; vertical is world-up
    const vy = w.y; w.y = 0; w.applyQuaternion(cam.quaternion); w.y += vy;
    w.multiplyScalar(sp);
    this.velocity.lerp(w, damp(this.damping, dt));
    cam.position.addScaledVector(this.velocity, dt);
    if (this.collide && this.collision.getGroundHeight) { this.collision.focus.copy(cam.position); this.collision.clampAboveGround(cam.position, this.minY); }
    this._snapPending = false;
  }

  dispose() {
    if (typeof window !== 'undefined' && this._onKeyDown) {
      window.removeEventListener('keydown', this._onKeyDown);
      window.removeEventListener('keyup', this._onKeyUp);
      window.removeEventListener('blur', this._onBlur);
    }
    const el = this.domElement;
    if (el && this._onDown) {
      el.removeEventListener('pointerdown', this._onDown);
      el.removeEventListener('pointermove', this._onMove);
      el.removeEventListener('pointerup', this._onUp);
      el.removeEventListener('pointercancel', this._onUp);
      el.removeEventListener('wheel', this._onWheel);
    }
  }
}
