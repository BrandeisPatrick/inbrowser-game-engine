// ChaseRig — smooth third-person / mounted (hood, cockpit) / cinematic-orbit vehicle camera with
// occlusion avoidance. Modes are pluggable configs:
//
//   const rig = new ChaseRig(camera, domElement, { minY: 0.6, getGroundHeight, queryColliders });  // or { physics }
//   rig.setTarget(vehicleObject3d, () => vehicle.state);   // state: { velocity, speedKmh, mode, thrust, gForce, grounded, wheelSlip }
//   rig.cycleMode(); rig.mode = 'cockpit'; rig.setLookBack(true); rig.snap(); rig.update(dt);
//   rig.addMode('roof', { type: 'mounted', offset: new THREE.Vector3(0, 2.2, 0.5), pitch: -0.1 });
//
// Mode types:
//   follow   spring-damped chase behind the follow frame (velocity-biased yaw, flight orientation blend,
//            speed-dependent distance/height/FOV, canopy ducking, climbs when squeezed); uses the rig tunables
//   mounted  rigidly attached at a local offset (car forward = local -Z), optional pitch and head tilt
//   orbit    slow cinematic orbit that only uses unobstructed positions (sweeps or cuts past blocked angles)
//   custom   { update(rig, dt, ctx) } — ctx: { state, speed, speedN, snap }
// Each config may add fov(speedN) -> degrees and shake (amplitude factor; orbit defaults to none).
//
// Occlusion: each frame the segment pivot (above the target) -> desired camera position is cast against the
// colliders near the target (CameraCollision). The camera is pulled in to the first hit minus a margin
// (instantly) and eases back out (slowly). No allocations per frame (apart from what queryColliders returns).
// Collision defaults are the Roadster world rules: canopyFromCollider = treeCanopy (kind-2 trunks carry a crown,
// the camera ducks under / pulls in front of it) and thinKinds = [2, 3] (trunks / poles only block near the
// camera). Pass canopyFromCollider: null and / or thinKinds: [] to turn them off, or your own rule / kinds.
//
// Flight framing (follow mode): with flightFraming true (default) the chase frame blends to the target's full
// orientation (reduced roll) and banks with the turn while state.mode === 'flight'; false keeps the ground
// framing always (jumping ground vehicles); 'auto' only switches once 'flight' has lasted flightFramingDelay
// seconds (ignores short hops, frames real flights).
import * as THREE from 'three';
import { CameraRig, clamp, smooth01, damp, wrapPi, treeCanopy } from './CameraRig.js';

const _UP = new THREE.Vector3(0, 1, 0);
const CINE_RADII = [1, 0.72, 0.5], CINE_LIFTS = [0, 2.5, 6], CINE_DIRS = [1, -1];
const ROADSTER_THIN_KINDS = Object.freeze([2, 3]);

/** ChaseRig collision defaults (Roadster world rules) unless the caller chose its own (null / [] disable). */
function withChaseDefaults(o) {
  return {
    ...o,
    canopyFromCollider: o.canopyFromCollider === undefined ? treeCanopy : o.canopyFromCollider,
    thinKinds: o.thinKinds === undefined ? ROADSTER_THIN_KINDS : o.thinKinds,
  };
}

/** Default Roadster mode set (order = cycle order). Getters keep the legacy rig fields authoritative. */
export function defaultChaseModes(rig) {
  return {
    chase: { type: 'follow' },
    hood: { type: 'mounted', get offset() { return rig.offsets.hood; } },
    cockpit: { type: 'mounted', get offset() { return rig.offsets.cockpit; }, get pitch() { return rig.cockpitPitch; }, headTilt: true },
    cinematic: { type: 'orbit' },
  };
}

export class ChaseRig extends CameraRig {
  /**
   * @param {THREE.PerspectiveCamera} camera
   * @param {HTMLElement} [domElement]
   * @param {object} [opts]  minY, getGroundHeight, queryColliders, physics, occlusion, collisionMargin,
   *                          offsets: { hood, cockpit }, modes: { name: config }, modeOrder: [names],
   *                          active (false = start inactive), flightFraming (true | false | 'auto'),
   *                          flightFramingDelay (s, 'auto' only, 0.6), canopyFromCollider (treeCanopy | null | fn),
   *                          thinKinds ([2, 3] | [] | kinds)
   */
  constructor(camera, domElement, opts = {}) {
    super(camera, withChaseDefaults(opts));
    this.domElement = domElement || null;
    this.flightFraming = opts.flightFraming ?? true;
    this.flightFramingDelay = opts.flightFramingDelay ?? 0.6;
    this._flightT = 0;

    // tunables
    this.distMin = 7; this.distMax = 11;
    this.heightMin = 2.2; this.heightMax = 2.9;
    this.fovMin = 55; this.fovMax = 72;
    this.speedForMax = 300 / 3.6;      // m/s at which distance/FOV reach max
    this.flightRollFactor = 0.4;       // how much of the target roll the chase cam inherits in flight
    this.flightBank = 0.006;           // extra bank per (rad/s yaw rate * m/s speed) in flight
    this.orbitRadius = 12;
    this.orbitSpeed = 0.18;            // rad/s
    this.shakeScale = opts.shakeScale ?? 0.6;   // 0 disables camera shake entirely
    this.margin = opts.collisionMargin ?? 0.4;   // stand-off from a hit surface along the ray
    this.minDist = 2.4;                           // below this the chase cam climbs instead

    this.offsets = {
      // bonnet cam just ahead of the windscreen; cockpit = driver's eye over the yoke and screen
      hood: (opts.offsets && opts.offsets.hood) || new THREE.Vector3(0, 0.95, -0.95),
      cockpit: (opts.offsets && opts.offsets.cockpit) || new THREE.Vector3(-0.35, 0.965, 0.15),
    };
    this.cockpitPitch = -0.05;         // look slightly down over the yoke
    this.lookBackOffset = new THREE.Vector3(0, 1.02, 0.1);

    this.modes = { ...defaultChaseModes(this), ...(opts.modes || {}) };
    this.modeOrder = opts.modeOrder ? opts.modeOrder.slice() : Object.keys(this.modes);
    this._modeIndex = 0;
    this.lookBack = false;

    // --- reusable scratch (no per-frame allocations) ---
    this._carPos = new THREE.Vector3();
    this._carQuat = new THREE.Quaternion();
    this._fwd = new THREE.Vector3();
    this._vdir = new THREE.Vector3();
    this._dir = new THREE.Vector3();
    this._followDir = new THREE.Vector3(0, 0, -1);
    this._euler = new THREE.Euler(0, 0, 0, 'YXZ');
    this._qGround = new THREE.Quaternion();
    this._qFlight = new THREE.Quaternion();
    this._qDesired = new THREE.Quaternion();
    this._qFollow = new THREE.Quaternion();
    this._qTmp = new THREE.Quaternion();
    this._idealOffset = new THREE.Vector3();
    this._offset = new THREE.Vector3(0, 3, 9);
    this._offsetVel = new THREE.Vector3();
    this._tmp = new THREE.Vector3();
    this._tmp2 = new THREE.Vector3();
    this._look = new THREE.Vector3();
    this._lookSmoothed = new THREE.Vector3();
    this._camUp = new THREE.Vector3(0, 1, 0);
    this._prevVel = new THREE.Vector3();
    this._accel = new THREE.Vector3();
    this._yAxis = new THREE.Vector3(0, 1, 0);
    this._xAxis = new THREE.Vector3(1, 0, 0);
    this._pivot = new THREE.Vector3();
    this._desired = new THREE.Vector3();
    this._cand = new THREE.Vector3();
    this._ctx = { state: null, speed: 0, speedN: 0, snap: false };

    this._flightBlend = 0;
    this._fov = camera.fov || 60;
    this._orbitAngle = 0;
    this._time = 0;
    this._hasPrevVel = false;
    this._lastLookBack = false;

    // occlusion state
    this._occLimit = Infinity;
    this._duck = 0;
    this._elev = 0;
    this._roll = 0;
    this._prevYaw = 0;
    this._hasYaw = false;
    // cinematic state
    this._cineR = this.orbitRadius;
    this._cineH = 0;
    this._cineTarget = null;   // target orbit angle when sweeping past an obstruction
    this._cineCheck = 0;
    this._cineCut = 0;
    this.debug = { hitT: 1, occLimit: Infinity, duck: 0, elev: 0, boxes: 0, canopies: 0 };
  }

  get nearMargin() { return this.collision.nearMargin; }
  set nearMargin(v) { this.collision.nearMargin = v; }

  get mode() { return this.modeOrder[this._modeIndex]; }
  set mode(m) {
    const i = this.modeOrder.indexOf(m);
    if (i >= 0 && i !== this._modeIndex) this._setModeIndex(i);
  }
  get modeConfig() { return this.modes[this.mode]; }

  /**
   * Register a mode; it joins the cycle unless { cycle: false }.
   * @param {string} name
   * @param {object} config
   * @param {{ cycle?: boolean, index?: number }} [o]  index: position in the cycle (default: appended)
   */
  addMode(name, config, { cycle = true, index } = {}) {
    this.modes[name] = config;
    if (cycle && !this.modeOrder.includes(name)) {
      if (index == null) this.modeOrder.push(name); else this.modeOrder.splice(index, 0, name);
    }
    return this;
  }
  removeMode(name) {
    const cur = this.mode;
    delete this.modes[name];
    this.modeOrder = this.modeOrder.filter((n) => n !== name);
    const i = this.modeOrder.indexOf(cur);
    this._modeIndex = i >= 0 ? i : 0;
    return this;
  }

  cycleMode() {
    this._setModeIndex((this._modeIndex + 1) % this.modeOrder.length);
    return this.mode;
  }

  setLookBack(on) { this.lookBack = !!on; }

  _setModeIndex(i) {
    const prev = this.modes[this.modeOrder[this._modeIndex]];
    this._modeIndex = i;
    const m = this.modes[this.modeOrder[i]];
    if (m && m.type === 'orbit' && this.target) {
      this._orbitAngle = Math.atan2(this._offset.x, this._offset.z);
      this._cineR = this.orbitRadius; this._cineTarget = null; this._cineCheck = 0;
    }
    if (prev && prev.type === 'mounted' && m && (m.type === 'follow' || m.type === 'orbit')) {
      this._offsetVel.set(0, 0, 0);
      this._occLimit = 3; // ease out from the target instead of popping through anything behind it
    }
  }

  _readCar() {
    const t = this.target;
    t.updateWorldMatrix(true, false);
    t.matrixWorld.decompose(this._carPos, this._carQuat, this._tmp);
  }

  /** Pull `desired` toward `pivot` to the first obstruction (smoothed), then near-plane push-out + ground clamp. */
  _resolve(pivot, desired, out, dt, snap) {
    const C = this.collision;
    const dist = desired.distanceTo(pivot);
    const tHit = C.segHit(pivot, desired, 0.15);
    let allowed = dist;
    if (tHit < 1) allowed = Math.max(0.6, tHit * dist - this.margin);
    if (snap || allowed < this._occLimit) this._occLimit = allowed;
    else this._occLimit += (allowed - this._occLimit) * damp(1.6, dt);
    if (this._occLimit > dist + 6) this._occLimit = dist + 6;
    const d = Math.min(dist, this._occLimit);
    out.copy(desired).sub(pivot);
    if (dist > 1e-6) out.multiplyScalar(d / dist);
    out.add(pivot);
    C.pushOut(out);
    C.clampAboveGround(out, this.minY);
    this.debug.hitT = tHit; this.debug.occLimit = this._occLimit;
    return allowed;
  }

  // ------------------------------------------------------------------------------------------
  update(dt) {
    if (!this.target || !this.enabled) return;
    dt = clamp(dt || 0, 0, 0.1);
    this._time += dt;
    const cam = this.camera;
    const state = (this.getState && this.getState()) || {};
    this._readCar();

    const carQuat = this._carQuat;
    const vel = state.velocity;
    const speed = vel ? vel.length() : (state.speedKmh || 0) / 3.6;
    const speedN = clamp(speed / this.speedForMax, 0, 1);
    const thrust = state.thrust || 0;
    const gForce = state.gForce || 0;
    const flightNow = state.mode === 'flight';
    this._flightT = flightNow ? this._flightT + dt : 0;
    const ff = this.flightFraming;
    const inFlight = ff === false ? false : ff === 'auto' ? this._flightT > this.flightFramingDelay : flightNow;

    // target acceleration (for camera lag "weight")
    if (vel) {
      if (this._hasPrevVel && dt > 0) {
        this._tmp.copy(vel).sub(this._prevVel).divideScalar(dt);
        if (this._tmp.lengthSq() > 200 * 200) this._tmp.set(0, 0, 0);   // reject teleports / resets
        this._accel.lerp(this._tmp, damp(8, dt));
      }
      this._prevVel.copy(vel);
      this._hasPrevVel = true;
    }

    this._flightBlend += ((inFlight ? 1 : 0) - this._flightBlend) * damp(2.5, dt);

    const snap = this._snapPending;
    const cfg = this.modes[this.mode] || { type: 'follow' };
    const type = cfg.type;

    if (type === 'mounted') {
      this._updateMounted(cfg, dt);
    } else {
      // ---------- desired follow orientation (kept current in every non-mounted mode) ----------
      this._fwd.set(0, 0, -1).applyQuaternion(carQuat);
      this._dir.set(this._fwd.x, 0, this._fwd.z);
      if (this._dir.lengthSq() < 1e-6) this._dir.set(0, 0, -1);
      this._dir.normalize();
      if (vel && speed > 2) {
        this._vdir.set(vel.x, 0, vel.z);
        const hl = this._vdir.length();
        if (hl > 1) {
          this._vdir.divideScalar(hl);
          const fwdDot = this._vdir.dot(this._dir);
          if (fwdDot > 0) {
            const w = 0.55 * smooth01(3, 18, hl) * smooth01(0.0, 0.5, fwdDot);
            this._dir.lerp(this._vdir, w).normalize();
          }
        }
      }
      if (snap) this._followDir.copy(this._dir);
      else this._followDir.lerp(this._dir, damp(5, dt)).normalize();
      if (this._followDir.lengthSq() < 1e-6) this._followDir.copy(this._dir);
      const yaw = Math.atan2(-this._followDir.x, -this._followDir.z);
      this._qGround.setFromAxisAngle(this._yAxis, yaw);

      // speed-dependent bank in flight (from the follow yaw rate)
      let yawRate = 0;
      if (this._hasYaw && dt > 0) yawRate = wrapPi(yaw - this._prevYaw) / dt;
      this._prevYaw = yaw; this._hasYaw = true;
      const bankT = snap ? 0 : clamp(yawRate * speed * this.flightBank, -0.35, 0.35) * this._flightBlend;
      this._roll += (bankT - this._roll) * damp(2.2, dt);

      // flight: full target orientation with reduced roll
      this._euler.setFromQuaternion(carQuat, 'YXZ');
      this._euler.z *= this.flightRollFactor;
      this._euler.x = clamp(this._euler.x, -1.2, 1.2) * 0.85;
      this._qFlight.setFromEuler(this._euler);

      this._qDesired.copy(this._qGround).slerp(this._qFlight, this._flightBlend);
      if (snap) this._qFollow.copy(this._qDesired);
      else this._qFollow.slerp(this._qDesired, damp(inFlight ? 3.5 : 6, dt));

      this.collision.gather(this._carPos, Math.max(this.distMax, this.orbitRadius) + 6);
      if (type === 'orbit') this._updateOrbit(dt, snap);
      else if (type === 'custom' && cfg.update) {
        const c = this._ctx; c.state = state; c.speed = speed; c.speedN = speedN; c.snap = snap;
        cfg.update(this, dt, c);
      } else this._updateFollow(dt, speedN, snap);
    }

    // ---------- FOV ----------
    let fovTarget;
    if (cfg.fov) fovTarget = cfg.fov(speedN);
    else if (type === 'orbit') fovTarget = 48 + 8 * speedN;
    else if (type === 'follow') fovTarget = this.fovMin + (this.fovMax - this.fovMin) * speedN;
    else fovTarget = 68 + 6 * speedN;
    fovTarget = clamp(fovTarget, 40, 80);
    this._fov = snap ? fovTarget : this._fov + (fovTarget - this._fov) * damp(3, dt);
    if (Math.abs(cam.fov - this._fov) > 0.01) {
      cam.fov = this._fov;
      cam.updateProjectionMatrix();
    }

    // ---------- flight bank + shake (applied after lookAt as small rotations) ----------
    if (type === 'follow' && Math.abs(this._roll) > 1e-4) cam.rotateZ(this._roll);
    const shake = cfg.shake ?? (type === 'orbit' ? 0 : type === 'follow' ? 1 : 0.6);
    if (shake > 0) {
      // gForce includes gravity: a car at rest reads 1 g, so only load ABOVE ~1.15 g (hard braking, landings,
      // boost) shakes the camera. Frequencies stay well below half the frame rate to avoid aliasing jitter.
      const amp = this.shakeScale * (
        0.0012 * clamp(gForce - 1.15, 0, 4) +
        0.0025 * thrust * thrust +
        0.0008 * speedN * speedN * speedN +
        0.0025 * (state.grounded ? (state.wheelSlip || 0) : 0)
      ) * shake;
      if (amp > 1e-5) {
        const t = this._time;
        const sx = (Math.sin(t * 11.3) + 0.5 * Math.sin(t * 17.9 + 1.3) + 0.25 * Math.sin(t * 23.1 + 2.1)) * amp;
        const sy = (Math.sin(t * 9.7 + 0.7) + 0.5 * Math.sin(t * 15.1 + 2.4) + 0.25 * Math.sin(t * 21.7)) * amp;
        cam.rotateX(sx);
        cam.rotateY(sy);
      }
    }

    this._snapPending = false;
  }

  _distanceForSpeed(speedN) { return this.distMin + (this.distMax - this.distMin) * speedN; }

  _updateFollow(dt, speedN, snap) {
    const cam = this.camera, C = this.collision;
    const dist = this._distanceForSpeed(speedN);
    const height = this.heightMin + (this.heightMax - this.heightMin) * speedN;

    // ideal offset in the follow frame: behind (+Z) and above; tight spots raise the elevation
    const back = this.lookBack ? -dist : dist;
    const base = Math.atan2(height, dist), el = base + this._elev * 0.75;
    const r = Math.hypot(height, dist);
    this._idealOffset.set(0, Math.sin(el) * r, Math.sign(back) * Math.cos(el) * r).applyQuaternion(this._qFollow);

    if (snap || this.lookBack !== this._lastLookBack) {
      this._offset.copy(this._idealOffset);
      this._offsetVel.set(0, 0, 0);
    } else {
      // spring-damper on the target-relative offset (stays stable at 400 km/h), slightly under-damped
      const k = 45, c = 2 * 0.86 * Math.sqrt(k);
      this._tmp.copy(this._idealOffset).sub(this._offset).multiplyScalar(k);
      this._tmp.addScaledVector(this._offsetVel, -c);
      this._tmp.addScaledVector(this._accel, -1.6);      // inertia: acceleration pushes the camera back
      this._offsetVel.addScaledVector(this._tmp, dt);
      this._offset.addScaledVector(this._offsetVel, dt);
      const maxDev = 3.5;
      this._tmp.copy(this._offset).sub(this._idealOffset);
      const dev = this._tmp.length();
      if (dev > maxDev) this._offset.copy(this._idealOffset).addScaledVector(this._tmp, maxDev / dev);
    }
    this._lastLookBack = this.lookBack;

    this._camUp.copy(_UP).applyQuaternion(this._qFollow);
    this._pivot.copy(this._carPos).addScaledVector(this._camUp, 1.05);
    this._desired.copy(this._carPos).add(this._offset);

    // duck under tree canopies (the crown starts ~3 m up; lower the camera rather than zoom in)
    let duckT = 0;
    if (C.occlusion && C.nCan) {
      const d = this._desired, K = C.canopies;
      const floorY = C.ground(d.x, d.z, this._carPos.y + 1.5) + this.minY + 0.5;
      for (let i = 0; i < C.nCan; i++) {
        const o = i * 5;
        if (d.y > K[o + 1]) continue;                       // above the crown centre: pull-in handles it
        const ex = (d.x - K[o]) / K[o + 3], ez = (d.z - K[o + 2]) / K[o + 3], h2 = ex * ex + ez * ez;
        if (h2 >= 1) continue;
        const yb = K[o + 1] - K[o + 4] * Math.sqrt(1 - h2) - 0.4;   // crown underside minus clearance
        const need = d.y - Math.max(yb, floorY);
        if (need > duckT) duckT = need;
      }
    }
    this._duck += (duckT - this._duck) * damp(duckT > this._duck ? 10 : 1.5, dt);
    if (snap) this._duck = duckT;
    this._desired.y -= this._duck;

    const allowed = this._resolve(this._pivot, this._desired, cam.position, dt, snap);
    // squeezed in hard (alley, wall right behind): climb for a higher, clearer view
    const elevT = allowed < this.minDist ? 1 : allowed < this.minDist * 1.8 && this._elev > 0.05 ? this._elev : 0;
    this._elev += (elevT - this._elev) * damp(elevT > this._elev ? 2.5 : 0.6, dt);
    this.debug.duck = this._duck; this.debug.elev = this._elev;
    this.debug.boxes = C.boxes ? C.boxes.length : 0; this.debug.canopies = C.nCan;

    // look at a point slightly above the target, a touch ahead in the travel direction
    this._look.copy(this._carPos).addScaledVector(this._camUp, 1.1);
    this._tmp.set(0, 0, this.lookBack ? 2 : -2.5).applyQuaternion(this._qFollow);
    this._look.add(this._tmp);
    if (snap) this._lookSmoothed.copy(this._look);
    else this._lookSmoothed.lerp(this._look, damp(14, dt));
    cam.up.copy(this._camUp);
    cam.lookAt(this._lookSmoothed);
  }

  /** Is an orbit offset (relative to the target) usable: segment clear, not in a crown, above ground. */
  _cineClear(offX, offY, offZ) {
    const C = this.collision;
    const c = this._cand.set(this._carPos.x + offX, this._carPos.y + offY, this._carPos.z + offZ);
    const g = C.ground(c.x, c.z, this._carPos.y + 1.5) + this.minY + 0.3;
    if (c.y < g) c.y = g;
    for (let i = 0; i < C.nCan; i++) if (C.canopyD(i, c.x, c.y, c.z) < 1.3) return false;
    return C.segHit(this._pivot, c, 0.3) >= 0.999;
  }

  _updateOrbit(dt, snap) {
    const cam = this.camera, C = this.collision;
    const r0 = this.orbitRadius;
    this._pivot.copy(this._carPos); this._pivot.y += 0.9;
    const hBase = 2.5 + 1.5 * Math.sin(this._time * 0.13);

    // advance the orbit (faster while sweeping past an obstruction)
    if (this._cineTarget !== null) {
      const d = wrapPi(this._cineTarget - this._orbitAngle), step = 1.3 * dt;
      if (Math.abs(d) <= step) { this._orbitAngle = this._cineTarget; this._cineTarget = null; } else this._orbitAngle += Math.sign(d) * step;
    } else this._orbitAngle += this.orbitSpeed * dt;
    this._cineCut = Math.max(0, this._cineCut - dt);

    // periodically verify the orbit position; search for a clear one if blocked
    this._cineCheck -= dt;
    if ((snap || this._cineCheck <= 0) && C.occlusion && (C.boxes || C.nCan || C.getGroundHeight)) {
      this._cineCheck = 0.15;
      const a = this._cineTarget !== null ? this._cineTarget : this._orbitAngle;
      const h = hBase + this._cineH;
      if (!this._cineClear(Math.sin(a) * this._cineR, h, Math.cos(a) * this._cineR)) {
        let found = false;
        search:
        for (let k = 1; k <= 12; k++) {
          for (let di = 0; di < 2; di++) {
            const dir = CINE_DIRS[di];
            const aa = a + dir * k * (Math.PI / 12);
            for (let ri = 0; ri < 3; ri++) for (let li = 0; li < 3; li++) {
              const rr = r0 * CINE_RADII[ri], lh = CINE_LIFTS[li];
              if (this._cineClear(Math.sin(aa) * rr, hBase + lh, Math.cos(aa) * rr)) {
                this._cineR = rr; this._cineH = lh;
                const dA = dir * k * (Math.PI / 12);
                if (Math.abs(dA) <= Math.PI / 4 && !snap) this._cineTarget = aa;   // sweep there
                else if (this._cineCut <= 0 || snap) {                               // cut there
                  this._orbitAngle = aa; this._cineTarget = null; this._cineCut = 2.5; snap = true;
                }
                found = true;
                break search;
              }
            }
          }
        }
        if (!found) { this._cineR = r0 * 0.5; this._cineH = 7; }   // nothing clear: high and close, pull-in does the rest
      } else if (this._cineTarget === null && (this._cineR < r0 || this._cineH > 0)) {
        // drift back to the nominal orbit when that is clear again
        if (this._cineClear(Math.sin(a) * r0, hBase, Math.cos(a) * r0)) { this._cineR += (r0 - this._cineR) * 0.25; this._cineH *= 0.75; }
      }
    }

    const r = this._cineR, h = hBase + this._cineH;
    this._idealOffset.set(Math.sin(this._orbitAngle) * r, h, Math.cos(this._orbitAngle) * r);
    if (snap) {
      this._offset.copy(this._idealOffset);
      this._offsetVel.set(0, 0, 0);
    } else {
      const k = 12, c = 2 * Math.sqrt(k);
      this._tmp.copy(this._idealOffset).sub(this._offset).multiplyScalar(k);
      this._tmp.addScaledVector(this._offsetVel, -c);
      this._offsetVel.addScaledVector(this._tmp, dt);
      this._offset.addScaledVector(this._offsetVel, dt);
    }
    this._desired.copy(this._carPos).add(this._offset);
    this._resolve(this._pivot, this._desired, cam.position, dt, snap);
    this.debug.boxes = C.boxes ? C.boxes.length : 0; this.debug.canopies = C.nCan;

    this._look.copy(this._carPos);
    this._look.y += 0.7;
    if (snap) this._lookSmoothed.copy(this._look);
    else this._lookSmoothed.lerp(this._look, damp(10, dt));
    cam.up.copy(_UP); // free of target roll
    cam.lookAt(this._lookSmoothed);
  }

  _updateMounted(cfg, dt) {
    const cam = this.camera;
    const local = this.lookBack ? this.lookBackOffset : cfg.offset;
    this._tmp.copy(local).applyQuaternion(this._carQuat);
    cam.position.copy(this._carPos).add(this._tmp);
    cam.quaternion.copy(this._carQuat);
    const pitch = cfg.pitch || 0;
    if (this.lookBack) {
      this._qTmp.setFromAxisAngle(this._yAxis, Math.PI);
      cam.quaternion.multiply(this._qTmp);
    } else if (pitch) {
      this._qTmp.setFromAxisAngle(this._xAxis, pitch);
      cam.quaternion.multiply(this._qTmp);
    }
    // slight head tilt with lateral / longitudinal accel for cockpit feel (no allocations)
    if (cfg.headTilt) {
      this._tmp2.copy(this._accel).applyQuaternion(this._qTmp.copy(this._carQuat).invert());
      cam.rotateZ(clamp(this._tmp2.x * -0.0025, -0.04, 0.04));
      cam.rotateX(clamp(this._tmp2.z * 0.0015, -0.03, 0.03));
    }
    cam.up.copy(_UP).applyQuaternion(this._carQuat);
    // keep the spring state coherent so leaving the mounted view is smooth
    this._offset.copy(cam.position).sub(this._carPos);
    this._lookSmoothed.copy(this._carPos);
    this._qFollow.copy(this._carQuat);
    this._roll = 0;
    void dt;
  }
}

export default ChaseRig;
