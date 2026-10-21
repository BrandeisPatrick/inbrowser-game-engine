// OrbitRig — dev / free orbit around a point or an Object3D.
//   left-drag rotate · right-drag (or shift+drag) pan · wheel / pinch zoom · optional auto-rotate.
//   const rig = new OrbitRig(camera, canvas, { target: mesh, distance: 12, autoRotate: 0.3 });
//   rig.update(dt) every frame. Ground clamp / collision via the CameraRig options (getGroundHeight, physics).
import * as THREE from 'three';
import { CameraRig, clamp, damp } from './CameraRig.js';

export class OrbitRig extends CameraRig {
  /**
   * @param {THREE.PerspectiveCamera} camera
   * @param {HTMLElement} [domElement]  pointer input surface (no input without one)
   * @param {object} [o]
   * @param {THREE.Object3D|THREE.Vector3} [o.target]
   * @param {number} [o.distance=10] [o.minDistance=1] [o.maxDistance=2000]
   * @param {number} [o.azimuth=0.6] [o.elevation=0.35]  radians (elevation above the horizon)
   * @param {number} [o.minElevation=-1.2] [o.maxElevation=1.45]
   * @param {number} [o.autoRotate=0]   rad/s
   * @param {number} [o.damping=12]     smoothing rate (higher = snappier)
   * @param {number} [o.rotateSpeed=0.005] [o.panSpeed=1] [o.zoomSpeed=1]
   */
  constructor(camera, domElement, o = {}) {
    super(camera, o);
    this.domElement = domElement || null;
    this.center = new THREE.Vector3();
    this.offset = new THREE.Vector3(o.offsetX || 0, o.offsetY ?? 0.5, o.offsetZ || 0); // look-at offset from the target
    this.distance = o.distance ?? 10;
    this.minDistance = o.minDistance ?? 1; this.maxDistance = o.maxDistance ?? 2000;
    this.azimuth = o.azimuth ?? 0.6; this.elevation = o.elevation ?? 0.35;
    this.minElevation = o.minElevation ?? -1.2; this.maxElevation = o.maxElevation ?? 1.45;
    this.autoRotate = o.autoRotate || 0;
    this.damping = o.damping ?? 12;
    this.rotateSpeed = o.rotateSpeed ?? 0.005; this.panSpeed = o.panSpeed ?? 1; this.zoomSpeed = o.zoomSpeed ?? 1;
    this.pan = new THREE.Vector3();              // user pan added to the target position
    // smoothed state
    this._az = this.azimuth; this._el = this.elevation; this._dist = this.distance;
    this._center = new THREE.Vector3();
    this._goal = new THREE.Vector3();
    this._right = new THREE.Vector3(); this._fwd = new THREE.Vector3();
    this._pointers = new Map();
    this._pinch = 0;
    if (o.target) this.setTarget(o.target);
    this._bind();
  }

  setTarget(target, stateGetter) {
    super.setTarget(target, stateGetter);
    this.pan.set(0, 0, 0);
  }

  _targetPos(out) {
    const t = this.target;
    if (!t) return out.set(0, 0, 0);
    if (t.isVector3) return out.copy(t);
    if (t.getWorldPosition) return t.getWorldPosition(out);
    return out.set(t.x || 0, t.y || 0, t.z || 0);
  }

  _bind() {
    const el = this.domElement;
    if (!el || !el.addEventListener) return;
    this._onDown = (e) => {
      if (!this.enabled) return;
      this._pointers.set(e.pointerId, { x: e.clientX, y: e.clientY, button: e.button, shift: e.shiftKey });
      if (el.setPointerCapture) try { el.setPointerCapture(e.pointerId); } catch { /* synthetic */ }
      if (this._pointers.size === 2) this._pinch = this._pinchDist();
    };
    this._onMove = (e) => {
      const p = this._pointers.get(e.pointerId);
      if (!p || !this.enabled) return;
      const dx = e.clientX - p.x, dy = e.clientY - p.y;
      p.x = e.clientX; p.y = e.clientY;
      if (this._pointers.size === 2) {
        const d = this._pinchDist();
        if (this._pinch > 0 && d > 0) this.zoom(this._pinch / d);
        this._pinch = d;
        return;
      }
      if (p.button === 2 || p.button === 1 || p.shift) this.panBy(dx, dy);
      else this.rotate(-dx * this.rotateSpeed, dy * this.rotateSpeed);
    };
    this._onUp = (e) => { this._pointers.delete(e.pointerId); this._pinch = 0; };
    this._onWheel = (e) => { if (!this.enabled) return; e.preventDefault(); this.zoom(Math.exp(e.deltaY * 0.001 * this.zoomSpeed)); };
    this._onContext = (e) => { if (this.enabled) e.preventDefault(); };
    el.addEventListener('pointerdown', this._onDown);
    el.addEventListener('pointermove', this._onMove);
    el.addEventListener('pointerup', this._onUp);
    el.addEventListener('pointercancel', this._onUp);
    el.addEventListener('wheel', this._onWheel, { passive: false });
    el.addEventListener('contextmenu', this._onContext);
  }
  _pinchDist() {
    const it = [...this._pointers.values()];
    return it.length < 2 ? 0 : Math.hypot(it[0].x - it[1].x, it[0].y - it[1].y);
  }

  rotate(dAz, dEl) {
    this.azimuth += dAz;
    this.elevation = clamp(this.elevation + dEl, this.minElevation, this.maxElevation);
  }
  zoom(factor) { this.distance = clamp(this.distance * factor, this.minDistance, this.maxDistance); }
  /** Pan in screen space (pixels). */
  panBy(dxPx, dyPx) {
    const h = this.domElement ? this.domElement.clientHeight || 600 : 600;
    const s = (2 * this._dist * Math.tan((this.camera.fov || 60) * Math.PI / 360)) / h * this.panSpeed;
    this._right.set(Math.cos(this._az), 0, -Math.sin(this._az));
    this._fwd.set(Math.sin(this._az), 0, Math.cos(this._az));
    this.pan.addScaledVector(this._right, -dxPx * s).addScaledVector(this._fwd, -dyPx * s);
  }

  update(dt) {
    if (!this.enabled) return;
    dt = clamp(dt || 0, 0, 0.1);
    if (this.autoRotate && this._pointers.size === 0) this.azimuth += this.autoRotate * dt;
    this._targetPos(this._goal).add(this.offset).add(this.pan);
    const snap = this._snapPending, k = snap ? 1 : damp(this.damping, dt);
    this._az += (this.azimuth - this._az) * k;
    this._el += (this.elevation - this._el) * k;
    this._dist += (this.distance - this._dist) * k;
    this._center.lerp(this._goal, snap ? 1 : damp(this.damping * 1.5, dt));
    this.center.copy(this._center);
    const cam = this.camera, ce = Math.cos(this._el);
    cam.position.set(
      this._center.x + Math.sin(this._az) * ce * this._dist,
      this._center.y + Math.sin(this._el) * this._dist,
      this._center.z + Math.cos(this._az) * ce * this._dist,
    );
    const C = this.collision;
    if (C.getGroundHeight) { C.focus.copy(this._center); C.clampAboveGround(cam.position, this.minY); }
    cam.up.set(0, 1, 0);
    cam.lookAt(this._center);
    this._snapPending = false;
  }

  dispose() {
    const el = this.domElement;
    if (!el || !this._onDown) return;
    el.removeEventListener('pointerdown', this._onDown);
    el.removeEventListener('pointermove', this._onMove);
    el.removeEventListener('pointerup', this._onUp);
    el.removeEventListener('pointercancel', this._onUp);
    el.removeEventListener('wheel', this._onWheel);
    el.removeEventListener('contextmenu', this._onContext);
  }
}
