// 3D audio: a listener that follows a camera and emitters attached to Object3Ds (PannerNode).
//
//   const space = new PositionalAudio(bus, { listener: camera });
//   const horn = space.add(trafficCar, { bus: 'effects', refDistance: 6 });   // returns an Emitter
//   someVoice.connect(horn.input);
//   space.update();                                                           // once per frame
//
// Works before the context exists: emitters are realised on bus start. No allocations per update.
import * as THREE from 'three';

const _p = new THREE.Vector3(), _q = new THREE.Quaternion(), _s = new THREE.Vector3();
const _f = new THREE.Vector3(), _u = new THREE.Vector3();

function setXYZ(node, prefix, x, y, z, ctx, legacy) {
  if (node[`${prefix}X`]) {
    const t = ctx.currentTime;
    node[`${prefix}X`].setTargetAtTime(x, t, 0.02);
    node[`${prefix}Y`].setTargetAtTime(y, t, 0.02);
    node[`${prefix}Z`].setTargetAtTime(z, t, 0.02);
  } else if (legacy) legacy.call(node, x, y, z);
}

export class AudioListener3D {
  constructor(ctx) { this.ctx = ctx; }
  /** Copy the world pose of an Object3D (camera) onto the AudioContext listener. */
  update(object3d) {
    const L = this.ctx && this.ctx.listener;
    if (!L || !object3d) return;
    object3d.updateWorldMatrix(true, false);
    object3d.matrixWorld.decompose(_p, _q, _s);
    _f.set(0, 0, -1).applyQuaternion(_q);
    _u.set(0, 1, 0).applyQuaternion(_q);
    setXYZ(L, 'position', _p.x, _p.y, _p.z, this.ctx, L.setPosition);
    if (L.forwardX) {
      const t = this.ctx.currentTime;
      L.forwardX.setTargetAtTime(_f.x, t, 0.02); L.forwardY.setTargetAtTime(_f.y, t, 0.02); L.forwardZ.setTargetAtTime(_f.z, t, 0.02);
      L.upX.setTargetAtTime(_u.x, t, 0.02); L.upY.setTargetAtTime(_u.y, t, 0.02); L.upZ.setTargetAtTime(_u.z, t, 0.02);
    } else if (L.setOrientation) L.setOrientation(_f.x, _f.y, _f.z, _u.x, _u.y, _u.z);
  }
}

export class Emitter {
  /**
   * @param {BaseAudioContext} ctx
   * @param {AudioNode} out
   * @param {object} o  object3d, refDistance=8, maxDistance=600, rolloff=1, distanceModel='inverse',
   *                    panningModel='HRTF', gain=1, cone: { inner, outer, outerGain }
   */
  constructor(ctx, out, o = {}) {
    this.ctx = ctx;
    this.object3d = o.object3d || null;
    this.input = ctx.createGain();
    this.input.gain.value = o.gain ?? 1;
    const p = this.panner = ctx.createPanner();
    p.panningModel = o.panningModel || 'HRTF';
    p.distanceModel = o.distanceModel || 'inverse';
    p.refDistance = o.refDistance ?? 8;
    p.maxDistance = o.maxDistance ?? 600;
    p.rolloffFactor = o.rolloff ?? 1;
    if (o.cone) { p.coneInnerAngle = o.cone.inner ?? 360; p.coneOuterAngle = o.cone.outer ?? 360; p.coneOuterGain = o.cone.outerGain ?? 0; }
    this.input.connect(p).connect(out);
  }
  update() {
    const o = this.object3d;
    if (!o) return;
    o.updateWorldMatrix(true, false);
    o.matrixWorld.decompose(_p, _q, _s);
    setXYZ(this.panner, 'position', _p.x, _p.y, _p.z, this.ctx, this.panner.setPosition);
    _f.set(0, 0, -1).applyQuaternion(_q);
    setXYZ(this.panner, 'orientation', _f.x, _f.y, _f.z, this.ctx, this.panner.setOrientation);
  }
  dispose() { try { this.input.disconnect(); this.panner.disconnect(); } catch { /* gone */ } }
}

export class PositionalAudio {
  /**
   * @param {import('./AudioBus.js').AudioBus} bus
   * @param {object} [o]  listener: Object3D (usually the camera)
   */
  constructor(bus, o = {}) {
    this.bus = bus;
    this.listenerObject = o.listener || null;
    this.listener = null;
    this.emitters = [];
    this._pending = [];
    bus.onStart((ctx) => {
      this.listener = new AudioListener3D(ctx);
      for (const p of this._pending) p.resolve(this._make(p.object3d, p.opts));
      this._pending.length = 0;
    });
  }
  _make(object3d, opts) {
    const e = new Emitter(this.bus.ctx, this.bus.bus(opts.bus || 'effects'), { ...opts, object3d });
    this.emitters.push(e);
    return e;
  }
  /** Attach an emitter to object3d. Returns the Emitter, or (before start) a Promise of it. */
  add(object3d, opts = {}) {
    if (this.bus.ctx) return this._make(object3d, opts);
    return new Promise((resolve) => this._pending.push({ object3d, opts, resolve }));
  }
  remove(e) { const i = this.emitters.indexOf(e); if (i >= 0) { this.emitters.splice(i, 1); e.dispose(); } }
  update() {
    if (!this.listener) return;
    if (this.listenerObject) this.listener.update(this.listenerObject);
    for (let i = 0; i < this.emitters.length; i++) this.emitters[i].update();
  }
  dispose() { for (const e of this.emitters) e.dispose(); this.emitters.length = 0; }
}
