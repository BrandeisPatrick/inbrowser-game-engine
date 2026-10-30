// AnimationController — animation state machine with 1D blend trees, timed crossfades, root motion and events,
// driving one THREE.AnimationMixer.
//
//   const anim = new AnimationController(character, gltf.animations, {
//     states: {
//       idle: { clip: 'Idle' },
//       walk: { clip: 'Walk', markers: [{ at: 0.1, name: 'footstep' }, { at: 0.6, name: 'footstep' }] },
//       run:  { clip: 'Run', speed: 1.1 },
//       jump: { clip: 'Jump', loop: false, onEnd: 'move' },
//     },
//     trees: { move: { type: '1d', param: 'speed', sync: true,
//                      points: [{ at: 0, state: 'idle' }, { at: 2, state: 'walk' }, { at: 5.5, state: 'run' }] } },
//     transitions: { default: 0.2, 'move->jump': 0.05, '*->idle': 0.3, 'jump->*': 0.15 },
//     rootMotion: { bone: 'Hips', xz: true, y: false, yaw: false },   // or false (default)
//     events,                                                         // an Events (default: a new one, anim.events)
//   });
//   anim.play('move');                                  // a state or a tree; crossfades from whatever is current
//   anim.setParam('speed', v); anim.update(dt);         // per frame (allocation-free)
//   anim.resync();                                      // after bones were written by hand (Ragdoll.deactivate)
//   anim.events.on('marker', (name, state) => {});      // 'loop' (state), 'end' (state); last arg = the controller
//   anim.applyRootMotion();                             // or read anim.rootMotion.deltaPosition / deltaYaw
//
// States: { clip (name in `clips`, or an AnimationClip), loop = true, speed = 1 (time scale, < 0 plays backwards;
// a looping state entered from rest starts at its end), clamp = true, onEnd = null, markers = null }. A one-shot
// (loop: false) holds its last frame once finished, and while it is faded out; when it finishes as the current state
// it emits 'end', then onEnd runs: a state / tree name is played (crossfade from the held frame), a function is
// called as fn(stateName, controller). It stays current (holding) until the next play(); play() of the current node
// is a no-op, except that a finished one-shot restarts. clamp: false (explicit opt-in, three.js' raw unclamped
// behaviour): if 'end' / onEnd neither played another node nor restarted it, its weight drops to 0, every node still
// fading out snaps to 0 with it and `current` becomes null (the skeleton returns to its rest pose).
// Trees: { type: '1d', param, points: [{ at, state }], sync = false }. The param value blends linearly between the
// two neighbouring points (clamped to the end points). sync: true matches the members' normalized phase: every
// member runs at clipDuration / D, D = sum of weight * duration / |speed| (the blended cycle length), and members
// that join are aligned to the heaviest one. Members of the current tree keep running at weight 0.
// Crossfades: play(name, { fade }) or the transition table (`from->to`, then `*->to`, then `from->*`, then
// `default`; seconds). The incoming node goes from its current weight to 1, every other node is scaled down by the
// same factor, so the node weights always sum to 1. Nothing current = instant. A state shared by several nodes
// (a state and a tree containing it) is one action: its weights add, so such switches are seamless. Fades and
// clips both run at dt * timeScale.
// Events (through `events`, positional): 'end' (state, ctl) when a one-shot of the current node finishes; 'loop'
// (state, ctl) and 'marker' (markerName, state, ctl) from the current state, or the dominant (heaviest) member of
// the current tree, so synced members do not double up. Markers are normalized times [0, 1), emitted once per
// crossing (a state entered at phase 0 counts a marker at 0).
// Root motion (opt-in): the controller samples the bone's ORIGINAL position (and yaw) tracks for every playing
// action, from its previous to its current clip time (loop wraps detected and unwrapped), weights the motion by the
// action's blend weight and exposes rootMotion.deltaPosition (root-local space, per update) and deltaYaw (radians,
// about +Y). The stripped components (xz; y with y: true; yaw about +Y with yaw: true) are pinned to their value at
// clip time 0 in per-state clones of the clips, so the skeleton stays in place; the caller's clips are never
// modified. Root space assumes the transforms between `root` and the bone's parent are constant (measured at
// construction). rootMotion.enabled = false stops the deltas (the clips stay stripped).
import * as THREE from 'three';
import { Events } from '../core/Events.js';

const NO_OPTS = Object.freeze({});
const TAU = Math.PI * 2;
const _UP = new THREE.Vector3(0, 1, 0);
const _v = new THREE.Vector3(), _v2 = new THREE.Vector3();
const _q = new THREE.Quaternion(), _q2 = new THREE.Quaternion();
const _m = new THREE.Matrix4();

const wrapPi = (a) => a - TAU * Math.floor((a + Math.PI) / TAU);
/** Number of marker crossings at + k in (p0, p1] (closed: [p0, p1]); backwards: [p1, p0). Phases are unwrapped. */
function crossings(p0, p1, at, closed) {
  if (p1 >= p0) return Math.floor(p1 - at) - (closed ? Math.ceil(p0 - at) - 1 : Math.floor(p0 - at));
  return (closed ? Math.floor(p0 - at) + 1 : Math.ceil(p0 - at)) - Math.ceil(p1 - at);
}
function removeItem(arr, item) {
  const i = arr.indexOf(item);
  if (i < 0) return false;
  for (let j = i + 1; j < arr.length; j++) arr[j - 1] = arr[j];
  arr.length--;
  return true;
}
function trackOf(clip, bone, prop) {
  for (const t of clip.tracks) {
    let p;
    try { p = THREE.PropertyBinding.parseTrackName(t.name); } catch { continue; }
    if (p.nodeName === bone && p.propertyName === prop) return t;
  }
  return null;
}

export class AnimationController {
  /**
   * @param {THREE.Object3D} root  mixer root (the character)
   * @param {THREE.AnimationClip[]} clips
   * @param {object} o
   * @param {object} o.states  { name: { clip, loop, speed, clamp, onEnd, markers } } (see header)
   * @param {object} [o.trees]  { name: { type: '1d', param, points: [{ at, state }], sync } }
   * @param {object} [o.transitions={ default: 0.2 }]  crossfade seconds keyed 'from->to', '*->to', 'from->*'
   * @param {object|boolean} [o.rootMotion=false]  { bone: 'Hips', xz: true, y: false, yaw: false } (true = these)
   * @param {Events} [o.events]  emitter for 'end' / 'loop' / 'marker' (default: a new Events)
   */
  constructor(root, clips, { states = {}, trees = {}, transitions = { default: 0.2 }, rootMotion = false, events = null } = {}) {
    this.root = root;
    this.mixer = new THREE.AnimationMixer(root);
    this.events = events || new Events();
    /** Multiplies dt for clips and fades (the mixer's own timeScale also applies to clips). */
    this.timeScale = 1;
    this.params = {};
    this.states = {};
    this.trees = {};
    this._list = [];        // state records
    this._nodes = [];       // nodes with weight (fading out ..., current last)
    this._current = null;
    this._fade = 0; this._u = 1;
    this._endQ = [];
    this._ownClips = [];
    this._disposed = false;

    const rmo = rootMotion === true ? {} : rootMotion;
    this.rootMotion = {
      enabled: !!rmo, bone: rmo ? rmo.bone || 'Hips' : null,
      xz: rmo ? rmo.xz !== false : false, y: rmo ? !!rmo.y : false, yaw: rmo ? !!rmo.yaw : false,
      deltaPosition: new THREE.Vector3(), deltaYaw: 0,
    };
    this._basis = new THREE.Matrix4(); this._basisInv = new THREE.Matrix4();
    this._basisQ = new THREE.Quaternion(); this._basisQInv = new THREE.Quaternion();
    if (rmo) {
      const bone = root.getObjectByName(this.rootMotion.bone);
      if (!bone) { console.warn(`AnimationController: root motion bone '${this.rootMotion.bone}' not found`); this.rootMotion.enabled = false; }
      else {
        for (let o = bone.parent; o && o !== root; o = o.parent) this._basis.premultiply(_m.compose(o.position, o.quaternion, o.scale));
        this._basisInv.copy(this._basis).invert();
        this._basis.decompose(_v, this._basisQ, _v2);
        this._basisQInv.copy(this._basisQ).invert();
      }
    }

    // transition table: from -> to -> seconds
    this._fadeDefault = 0.2; this._fadeTbl = {};
    for (const key in transitions) {
      const sec = +transitions[key];
      if (key === 'default') { this._fadeDefault = sec; continue; }
      const i = key.indexOf('->');
      if (i < 0) continue;
      const from = key.slice(0, i).trim(), to = key.slice(i + 2).trim();
      (this._fadeTbl[from] || (this._fadeTbl[from] = {}))[to] = sec;
    }

    const used = new Set();
    for (const name in states) this._addState(name, states[name], clips, used);
    for (const name in trees) this._addTree(name, trees[name]);
  }

  _addState(name, spec, clips, used) {
    const src = typeof spec.clip === 'string' ? THREE.AnimationClip.findByName(clips, spec.clip) : spec.clip;
    if (!src) throw new Error(`AnimationController: state '${name}': clip '${spec.clip}' not found`);
    const rmOn = this.rootMotion.enabled;
    const posT = rmOn ? trackOf(src, this.rootMotion.bone, 'position') : null;
    const quatT = rmOn && this.rootMotion.yaw ? trackOf(src, this.rootMotion.bone, 'quaternion') : null;
    let clip = src;
    if (posT || quatT || used.has(src)) { clip = src.clone(); this._ownClips.push(clip); }   // one action per state
    used.add(src);
    const loop = spec.loop !== false;
    const action = this.mixer.clipAction(clip);
    action.setLoop(loop ? THREE.LoopRepeat : THREE.LoopOnce, Infinity);
    action.clampWhenFinished = true;           // one-shots hold their last frame while fading out (see header)
    action.play(); action.stop();              // bind now: activation in update() reuses the cached bindings
    const s = {
      name, isTree: false, clip, action, duration: clip.duration, loop,
      speed: spec.speed ?? 1, clamp: spec.clamp !== false, onEnd: spec.onEnd ?? null,
      markers: spec.markers && spec.markers.length ? spec.markers.map((m) => ({ at: +m.at, name: m.name })) : null,
      rm: null, w: 0, ts: 1, nw: 0, nw0: 0, active: false, ended: false, fresh: true, closed: false, inCur: false,
      ta: 0, tb: 0, wraps: 0,
    };
    if (posT || quatT) s.rm = this._makeRootMotion(clip, posT, quatT);
    if (this.states[name] || this.trees[name]) throw new Error(`AnimationController: duplicate node '${name}'`);
    this.states[name] = s; this._list.push(s);
  }

  /** Samplers over the caller's (original) tracks + strip the clone's copies of them. */
  _makeRootMotion(clip, posT, quatT) {
    const R = this.rootMotion, dur = clip.duration;
    const r = {
      pos: posT ? posT.createInterpolant(new Float64Array(3)) : null,
      quat: quatT ? quatT.createInterpolant(new Float64Array(4)) : null,
      p0: new THREE.Vector3(), cyclePos: new THREE.Vector3(), yaw0: 0, cycleYaw: 0,
      prev: new THREE.Vector3(), prevYaw: 0,
    };
    r.yaw0 = this._sample(r, 0, r.p0);
    this._sample(r, dur, r.cyclePos); r.cyclePos.sub(r.p0);
    if (posT) {
      const vals = trackOf(clip, R.bone, 'position').values;
      for (let i = 0; i < vals.length; i += 3) {
        _v.fromArray(vals, i).applyMatrix4(this._basis);
        if (R.xz) { _v.x = r.p0.x; _v.z = r.p0.z; }
        if (R.y) _v.y = r.p0.y;
        _v.applyMatrix4(this._basisInv).toArray(vals, i);
      }
    }
    if (quatT) {
      let acc = 0, prev = r.yaw0;            // unwrapped yaw over one cycle (turns past 180 degrees survive)
      for (let i = 0, ts = quatT.times; i <= ts.length; i++) {
        const t = i < ts.length ? ts[i] : dur;
        if (i < ts.length && (t <= 0 || t >= dur)) continue;
        const y = this._sample(r, t, _v2);
        acc += wrapPi(y - prev); prev = y;
      }
      r.cycleYaw = acc;
      const vals = trackOf(clip, R.bone, 'quaternion').values;
      for (let i = 0; i < vals.length; i += 4) {
        _q.fromArray(vals, i).premultiply(this._basisQ);
        _q2.setFromAxisAngle(_UP, r.yaw0 - 2 * Math.atan2(_q.y, _q.w));
        _q.premultiply(_q2).premultiply(this._basisQInv).toArray(vals, i);
      }
    }
    return r;
  }

  /** Root-space bone position at clip time t into out; returns the bone's yaw about +Y (0 without a yaw track). */
  _sample(r, t, out) {
    if (r.pos) out.fromArray(r.pos.evaluate(t)).applyMatrix4(this._basis); else out.set(0, 0, 0);
    if (!r.quat) return 0;
    _q.fromArray(r.quat.evaluate(t)).premultiply(this._basisQ);
    return wrapPi(2 * Math.atan2(_q.y, _q.w));   // twist about +Y: additive under a premultiplied yaw
  }

  _addTree(name, spec) {
    if ((spec.type || '1d') !== '1d') throw new Error(`AnimationController: tree '${name}': only type '1d' is supported`);
    if (this.states[name] || this.trees[name]) throw new Error(`AnimationController: duplicate node '${name}'`);
    if (!Array.isArray(spec.points) || !spec.points.length) throw new Error(`AnimationController: tree '${name}' needs points`);
    if (spec.points.some((p) => !Number.isFinite(+p.at))) throw new Error(`AnimationController: tree '${name}': every point needs a finite 'at'`);
    const points = spec.points.map((p) => {
      const s = this.states[p.state];
      if (!s) throw new Error(`AnimationController: tree '${name}': unknown state '${p.state}'`);
      return { at: +p.at, s };
    }).sort((a, b) => a.at - b.at);
    const members = [];
    for (const p of points) if (!members.includes(p.s)) members.push(p.s);
    this.trees[name] = {
      name, isTree: true, param: spec.param, sync: !!spec.sync, points, members,
      mw: new Float64Array(points.length), dom: 0, nw: 0, nw0: 0,
    };
  }

  /** Name of the current state / tree (null before the first play, or after a clamp: false one-shot ended). */
  get current() { return this._current ? this._current.name : null; }

  /** Set a blend-tree parameter. */
  setParam(name, value) { this.params[name] = value; return this; }
  /** Read a parameter (0 when unset). */
  getParam(name) { return this.params[name] ?? 0; }

  /** Blend weight of a state (summed over every node using it) or a tree's fade weight, as of the last update(). */
  getWeight(name) {
    const s = this.states[name];
    if (s) return s.w;
    const t = this.trees[name];
    return t ? t.nw : 0;
  }

  /**
   * Make a state or tree current, crossfading from whatever plays now.
   * @param {string} name
   * @param {object} [o]
   * @param {number} [o.fade]  seconds (default: the transition table; instant when nothing is current)
   */
  play(name, o = NO_OPTS) {
    const node = this.states[name] || this.trees[name];
    if (!node) { console.warn(`AnimationController: no state or tree '${name}'`); return this; }
    const cur = this._current, nodes = this._nodes;
    if (node === cur) {
      if (!node.isTree && node.ended) this._restart(node);
      return this;
    }
    let fade = o.fade;
    if (fade == null) fade = cur ? this._lookupFade(cur.name, name) : nodes.length ? this._fadeDefault : 0;
    const was = removeItem(nodes, node);
    for (let i = 0; i < nodes.length; i++) nodes[i].nw0 = nodes[i].nw;
    node.nw0 = was ? node.nw : 0;
    nodes.push(node);
    this._current = node;
    this._fade = fade; this._u = fade > 0 ? 0 : 1;
    if (!node.isTree) { if (!node.loop) this._restart(node); } else if (node.sync) this._alignPhase(node);
    return this;
  }

  _lookupFade(from, to) {
    const f = this._fadeTbl[from], any = this._fadeTbl['*'];
    let v = f && f[to];
    if (v != null) return v;
    v = any && any[to];
    if (v != null) return v;
    v = f && f['*'];
    return v != null ? v : this._fadeDefault;
  }

  _restart(s) {
    s.action.reset();
    if (s.speed < 0) s.action.time = s.duration;
    s.ended = false;
    if (s.active) this._resync(s);
  }

  /** The action's time was set: restart marker / root-motion tracking from it. */
  _resync(s) {
    s.fresh = true;
    if (s.rm) s.rm.prevYaw = this._sample(s.rm, s.action.time, s.rm.prev);
  }

  /** Synced tree entering: members without weight jump to the heaviest member's normalized phase. */
  _alignPhase(t) {
    const M = t.members;
    let ref = null;
    for (let i = 0; i < M.length; i++) if (M[i].active && M[i].w > 0 && (!ref || M[i].w > ref.w)) ref = M[i];
    const phase = ref && ref.duration > 0 ? ref.action.time / ref.duration : 0;
    for (let i = 0; i < M.length; i++) {
      const s = M[i];
      if (s === ref || (s.active && s.w > 0)) continue;
      s.action.time = phase * s.duration;
      if (s.active) this._resync(s);
    }
  }

  _treeWeights(t) {
    let v = this.params[t.param] ?? 0;
    if (v !== v) v = 0;
    const P = t.points, mw = t.mw, n = P.length;
    for (let i = 0; i < n; i++) mw[i] = 0;
    if (v <= P[0].at) { mw[0] = 1; t.dom = 0; return; }
    if (v >= P[n - 1].at) { mw[n - 1] = 1; t.dom = n - 1; return; }
    for (let i = 0; i < n - 1; i++) {
      if (v >= P[i + 1].at) continue;
      const span = P[i + 1].at - P[i].at, f = span > 0 ? (v - P[i].at) / span : 1;
      mw[i] = 1 - f; mw[i + 1] = f; t.dom = f < 0.5 ? i : i + 1;
      return;
    }
  }

  _syncScales(t) {
    const P = t.points, mw = t.mw;
    let D = 0;
    for (let i = 0; i < P.length; i++) if (mw[i] > 0) D += mw[i] * P[i].s.duration / Math.abs(P[i].s.speed || 1);
    if (!(D > 0)) return;
    const M = t.members;
    for (let i = 0; i < M.length; i++) M[i].ts = (M[i].speed < 0 ? -M[i].duration : M[i].duration) / D;
  }

  /**
   * Advance fades, blend weights and the mixer by dt (scaled by timeScale), extract root motion, emit events.
   * Allocation-free.
   */
  update(dt) {
    if (this._disposed) return this;
    const step = dt * this.timeScale, nodes = this._nodes, list = this._list, cur = this._current;
    if (this._u < 1) this._u = this._fade > 0 ? Math.min(1, this._u + Math.abs(step) / this._fade) : 1;
    const u = this._u;

    // node weights -> per-state weights and time scales
    for (let i = 0; i < list.length; i++) { const s = list[i]; s.w = 0; s.ts = s.speed; s.inCur = false; }
    for (let i = 0; i < nodes.length; i++) {
      const n = nodes[i];
      n.nw = n === cur ? n.nw0 + (1 - n.nw0) * u : n.nw0 * (1 - u);
      if (!n.isTree) { n.w += n.nw; continue; }
      this._treeWeights(n);
      for (let k = 0; k < n.points.length; k++) n.points[k].s.w += n.nw * n.mw[k];
      if (n.sync) this._syncScales(n);     // the current node is last, so its sync wins
    }
    if (cur) {
      if (!cur.isTree) cur.ts = cur.speed;   // a fading-out synced tree containing this state must not set its tempo
      if (cur.isTree) for (let k = 0; k < cur.members.length; k++) cur.members[k].inCur = true;
      else cur.inCur = true;
    }
    for (let i = 0; i < list.length; i++) {
      const s = list[i], a = s.action;
      if (s.w > 0 || s.inCur) {
        if (!s.active) {
          if (s.loop && s.speed < 0 && a.time === 0) a.time = s.duration;   // reversed loop: start at the end
          a.play(); s.active = true; this._resync(s);
        }
        a.setEffectiveWeight(s.w);
        a.setEffectiveTimeScale(s.ts);
        s.ta = a.time;
      } else if (s.active) { a.stop(); s.active = false; s.ended = false; }
    }
    // drop faded-out nodes
    if (u >= 1) {
      let k = 0;
      for (let i = 0; i < nodes.length; i++) if (nodes[i] === cur) nodes[k++] = cur; else nodes[i].nw = 0;
      nodes.length = k;
    }

    this.mixer.update(step);

    // clip-time spans, root motion, one-shot ends
    const R = this.rootMotion, rmOn = R.enabled, D = R.deltaPosition, k = step * this.mixer.timeScale;
    D.set(0, 0, 0); R.deltaYaw = 0;
    const endQ = this._endQ;
    endQ.length = 0;
    for (let i = 0; i < list.length; i++) {
      const s = list[i];
      if (!s.active) continue;
      const a = s.action, tb = a.time, dur = s.duration;
      s.tb = tb;
      s.wraps = s.loop && dur > 0 ? Math.round((s.ta + k * a.getEffectiveTimeScale() - tb) / dur) : 0;
      s.closed = s.fresh; s.fresh = false;
      if (s.rm) this._accumulate(s, rmOn);
      if (s.inCur && !s.loop && !s.ended && (s.ts >= 0 ? tb >= dur : tb <= 0)) { s.ended = true; endQ.push(s); }
    }

    // events (listeners may call play(); everything they could disturb was read above)
    const E = this.events;
    const em = cur ? (cur.isTree ? cur.points[cur.dom].s : cur) : null;
    if (em && em.active) {
      const n = Math.abs(em.wraps);
      for (let j = 0; j < n; j++) E.emit('loop', em.name, this);
      const M = em.markers, dur = em.duration;
      if (M && dur > 0) {
        const p0 = em.ta / dur, p1 = (em.tb + em.wraps * dur) / dur;
        for (let i = 0; i < M.length; i++) {
          const c = crossings(p0, p1, M[i].at, em.closed);
          for (let j = 0; j < c; j++) E.emit('marker', M[i].name, em.name, this);
        }
      }
    }
    for (let i = 0; i < endQ.length; i++) {
      const s = endQ[i];
      E.emit('end', s.name, this);
      if (typeof s.onEnd === 'string') this.play(s.onEnd);
      else if (typeof s.onEnd === 'function') s.onEnd(s.name, this);
      // unclamped drop, unless a listener / onEnd moved on or restarted it (_restart clears `ended`)
      if (!s.clamp && s.ended && this._current === s) { this._current = null; s.nw = 0; removeItem(this._nodes, s); this._u = 1; }
    }
    endQ.length = 0;
    return this;
  }

  /** Weighted root motion of one action over [ta, tb] (+ wraps whole cycles), in root space. */
  _accumulate(s, on) {
    const r = s.rm, R = this.rootMotion;
    const yaw = this._sample(r, s.tb, _v);
    _v2.subVectors(_v, r.prev).addScaledVector(r.cyclePos, s.wraps);
    const dyaw = r.quat ? wrapPi(yaw - r.prevYaw + s.wraps * r.cycleYaw) : 0;
    if (on && s.w > 0) {
      // with yaw extraction the clip's path turns with the extracted heading: express it in the current heading
      if (r.quat) _v2.applyAxisAngle(_UP, r.yaw0 - (r.prevYaw + 0.5 * dyaw));
      if (!R.xz) _v2.x = _v2.z = 0;
      if (!R.y) _v2.y = 0;
      R.deltaPosition.addScaledVector(_v2, s.w);
      if (R.yaw) R.deltaYaw += dyaw * s.w;
    }
    r.prev.copy(_v); r.prevYaw = yaw;
  }

  /**
   * Apply this update's root motion to an object (default: root): position += deltaPosition (scaled and rotated by
   * the object's own transform), then turn it by deltaYaw about its parent's +Y.
   */
  applyRootMotion(target = this.root) {
    const R = this.rootMotion;
    _v.copy(R.deltaPosition).multiply(target.scale).applyQuaternion(target.quaternion);
    target.position.add(_v);
    if (R.deltaYaw !== 0) target.quaternion.premultiply(_q.setFromAxisAngle(_UP, R.deltaYaw));
    return this;
  }

  /**
   * Make the mixer write every animated channel on its next update. A THREE.PropertyMixer only writes a value that
   * differs from the one it wrote last, so after bones were written by hand (a Ragdoll's deactivate(), a pose restore,
   * an IK pass kept across frames) a channel whose blended clip value has not changed since (a constant track, a held
   * pose) would keep the hand-written value. Call it once after writing, before the next update(). Allocation-free; the
   * mixer's original (rest) values, which partial weights blend towards, are kept. Reads the mixer's bindings
   * (three's internal `_bindings`, each a PropertyMixer with the buffer layout [incoming | accu0 | accu1 | orig | ...]).
   * @returns {this}
   */
  resync() {
    const bindings = this.mixer._bindings || [];
    for (let i = 0; i < bindings.length; i++) {
      const pm = bindings[i], n = pm.valueSize, buf = pm.buffer;
      if (!pm.binding || !buf) continue;
      pm.binding.getValue(buf, n);                               // accu0 := the property as it is now
      for (let k = 0; k < n; k++) buf[2 * n + k] = buf[n + k];   // accu1 too: the next blend differs, so it is written
    }
    return this;
  }

  /** Stop everything and release the mixer's caches for this root and the controller's clip clones. */
  dispose() {
    if (this._disposed) return;
    this._disposed = true;
    this.mixer.stopAllAction();
    for (const c of this._ownClips) this.mixer.uncacheClip(c);
    this.mixer.uncacheRoot(this.root);
    this._nodes.length = 0; this._current = null; this._list.length = 0;
  }
}

export default AnimationController;
