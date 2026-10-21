// Action-map input: named actions bound to keys, gamepad buttons/axes, pointer and virtual (touch) sources.
//
//   const input = new InputMap({ pointerTarget: canvas });
//   input.defineAll({
//     steer:    { type: 'axis', range: [-1, 1], smoothing: ramp(0.15, 0.07),
//                 bindings: [{ key: 'KeyD', value: 1 }, { key: 'KeyA', value: -1 }, { padAxis: 0 }, { virtual: 'stick.x' }] },
//     throttle: { type: 'axis', range: [0, 1], bindings: ['KeyW', { key: 'ArrowUp', when: '!flight' }, { padButton: 7 }] },
//     jump:     { type: 'button', bindings: ['Space', { padButton: 0 }] },
//   });
//   input.setContext('flight', true);      // bindings with when: 'flight' / '!flight' switch over
//   input.update(dt);                       // once per frame
//   input.value('steer'); input.down('jump'); input.justPressed('jump'); input.justReleased('jump');
//
// `input.enabled = false` silences every action except "always-on" ones: an action defined with
// alwaysOn: true (or a single binding carrying alwaysOn: true) keeps working, e.g. menu / camera / debug keys
// that must still respond while a free-fly camera or a text overlay has taken over the controls.
//
// Combining: bindings with smooth=true (keys, mouse buttons by default) are OR-ed per direction
// (holding A and ← is not double), then ramped with the action's `smoothing` so digital play feels
// analog. Bindings with smooth=false (gamepad, touch, mouse deltas) are summed and clamped to ±1, unsmoothed.
// The action value is whichever of the two groups has the larger magnitude, clamped to `range`.
// Edges are latched from key events, so a tap shorter than a frame still yields one justPressed.
// Key events with Cmd / Ctrl / Alt held are ignored (browser shortcuts), but the modifier keys themselves are
// tracked, so 'ShiftLeft' / 'AltLeft' / ... can be bound (e.g. a "slow" modifier).
// Safe to import in Node; DOM listeners are only attached when `window` exists.
import { GamepadState } from './Gamepad.js';

/** Move `cur` toward `target`; rate `up` when moving away from zero, `center` otherwise (units/s). */
export function approach(cur, target, dt, up, center) {
  const rising = Math.abs(target) > Math.abs(cur) && Math.sign(target) !== -Math.sign(cur);
  const rate = rising ? up : center;
  const d = target - cur;
  const step = rate * dt;
  return Math.abs(d) <= step ? target : cur + Math.sign(d) * step;
}

/** Smoothing spec from seconds-to-full-deflection and seconds-to-recentre. */
export function ramp(upSeconds, downSeconds = upSeconds) { return { up: 1 / upSeconds, down: 1 / downSeconds }; }

function isTypingTarget(el) {
  if (!el) return false;
  const tag = el.tagName;
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || el.isContentEditable === true;
}

const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const MOUSE_AXES = ['x', 'y', 'wheel'];
const MODIFIER_CODES = new Set(['ShiftLeft', 'ShiftRight', 'AltLeft', 'AltRight', 'ControlLeft', 'ControlRight', 'MetaLeft', 'MetaRight']);

/** Normalise a binding descriptor (string = key code). */
export function normalizeBinding(b) {
  if (typeof b === 'string') b = { key: b };
  const n = { ...b };
  if (n.key != null) n.kind = 'key';
  else if (n.padButton != null) n.kind = 'padButton';
  else if (n.padAxis != null) n.kind = 'padAxis';
  else if (n.mouse != null) n.kind = typeof n.mouse === 'number' ? 'mouseButton' : 'mouseAxis';
  else if (n.virtual != null || n.touch != null) { n.kind = 'virtual'; n.virtual = n.virtual ?? n.touch; }
  else throw new Error(`InputMap: unknown binding ${JSON.stringify(b)}`);
  if (n.kind === 'mouseAxis' && !MOUSE_AXES.includes(n.mouse)) throw new Error(`InputMap: mouse axis must be one of ${MOUSE_AXES}`);
  n.value = n.value ?? 1;
  n.smooth = n.smooth ?? (n.kind === 'key' || n.kind === 'mouseButton');
  const w = n.when == null ? [] : Array.isArray(n.when) ? n.when : [n.when];
  n._req = w.filter((c) => c[0] !== '!');
  n._not = w.filter((c) => c[0] === '!').map((c) => c.slice(1));
  return n;
}

export class InputMap {
  /**
   * @param {object} [o]
   * @param {EventTarget} [o.target=window]   keyboard listener target
   * @param {EventTarget} [o.pointerTarget]    pointer listener target (attached lazily when a mouse binding exists)
   * @param {boolean} [o.preventDefault=true] preventDefault on bound keys (browser shortcuts with Cmd/Ctrl/Alt are never captured)
   * @param {number} [o.deadzone=0.12]        default gamepad axis deadzone
   * @param {boolean} [o.gamepad=true]        poll the gamepad in update()
   * @param {object} [o.actions]              initial action definitions (see defineAll)
   */
  constructor(o = {}) {
    this.enabled = true;
    this.preventDefault = o.preventDefault !== false;
    this.useGamepad = o.gamepad !== false;
    this.gamepad = new GamepadState({ deadzone: o.deadzone ?? 0.12 });
    this.gamepadConnected = false;
    this.actions = new Map();
    this.contexts = new Set();
    this.virtual = new Map();
    this.dt = 0;

    this._keys = new Set();
    this._keyEdges = new Set();
    this._boundKeys = new Set();
    this._mouseButtons = 0; this._mouseEdges = 0;
    this._mouse = { x: 0, y: 0, wheel: 0 };       // deltas accumulated since the last update
    this._mouseFrame = { x: 0, y: 0, wheel: 0 };  // deltas of the current frame
    this._virtualEdges = new Set();
    this._lastTime = 0;
    this._capture = null;

    this._target = o.target || (typeof window !== 'undefined' ? window : null);
    this._pointerTarget = o.pointerTarget || null;
    this._pointerAttached = false;
    this._onKeyDown = this._onKeyDown.bind(this);
    this._onKeyUp = this._onKeyUp.bind(this);
    this._onBlur = () => { this._keys.clear(); this._mouseButtons = 0; };
    this._onPadConnect = () => { this.gamepadConnected = true; };
    this._onPadDisconnect = () => { this.gamepadConnected = false; };
    this._onPointerDown = (e) => { const bit = 1 << e.button; if (!(this._mouseButtons & bit)) this._mouseEdges |= bit; this._mouseButtons |= bit; };
    this._onPointerUp = (e) => { this._mouseButtons &= ~(1 << e.button); };
    this._onPointerMove = (e) => { this._mouse.x += e.movementX || 0; this._mouse.y += e.movementY || 0; };
    this._onWheel = (e) => { this._mouse.wheel += e.deltaY || 0; };
    if (this._target && this._target.addEventListener) {
      this._target.addEventListener('keydown', this._onKeyDown, { passive: false });
      this._target.addEventListener('keyup', this._onKeyUp);
      if (typeof window !== 'undefined') {
        window.addEventListener('blur', this._onBlur);
        window.addEventListener('gamepadconnected', this._onPadConnect);
        window.addEventListener('gamepaddisconnected', this._onPadDisconnect);
      }
    }
    if (o.actions) this.defineAll(o.actions);
  }

  // ------------------------------------------------------------------ definition
  /**
   * Define (or replace) an action.
   * @param {string} name
   * @param {object} spec  { type: 'axis'|'button', range?: [min,max], smoothing?: {up,down}, bindings: Binding[],
   *                         alwaysOn?: bool (keeps working while `enabled` is false) }
   *   Binding: 'KeyW' | { key, value? } | { padButton, value?, binary? } | { padAxis, value?, deadzone?, half?: 1|-1 }
   *          | { mouse: 0|1|2 } | { mouse: 'x'|'y'|'wheel', value? (scale) } | { virtual|touch: name, value? }
   *   Any binding may carry when: 'ctx' | '!ctx' | [...], smooth: bool and alwaysOn: bool.
   */
  define(name, spec = {}) {
    const type = spec.type || 'axis';
    const range = spec.range || (type === 'button' ? [0, 1] : [-1, 1]);
    const a = {
      name, type, min: range[0], max: range[1],
      smoothing: spec.smoothing || null,
      bindings: (spec.bindings || []).map(normalizeBinding),
      alwaysOn: !!spec.alwaysOn, _liveBindings: false,
      value: 0, smoothed: 0, analog: 0, down: false, prevDown: false, justPressed: false, justReleased: false,
      padDown: false,
    };
    this.actions.set(name, a);
    this._reindex();
    return this;
  }
  defineAll(map) { for (const k of Object.keys(map)) this.define(k, map[k]); return this; }
  undefine(name) { this.actions.delete(name); this._reindex(); return this; }

  bind(name, binding) { const a = this._action(name); a.bindings.push(normalizeBinding(binding)); this._reindex(); return this; }
  unbind(name, predicate) {
    const a = this._action(name);
    a.bindings = a.bindings.filter((b) => !predicate(b));
    this._reindex();
    return this;
  }
  /** Replace all bindings of an action. */
  rebind(name, bindings) { const a = this._action(name); a.bindings = bindings.map(normalizeBinding); this._reindex(); return this; }
  /** Serializable bindings (for saving user rebinds). */
  getBindings() {
    const out = {};
    for (const [k, a] of this.actions) out[k] = a.bindings.map(({ kind, _req, _not, ...rest }) => rest);
    return out;
  }
  loadBindings(obj) { for (const k of Object.keys(obj || {})) if (this.actions.has(k)) this.rebind(k, obj[k]); return this; }

  /**
   * Resolve with the next physical input as a binding descriptor (for rebinding UIs):
   * a key ({ key }), gamepad button ({ padButton }) or stick deflection ({ padAxis, value: ±1 }).
   */
  captureBinding({ timeout = 10000 } = {}) {
    return new Promise((resolve) => {
      const done = (b) => { clearTimeout(t); this._capture = null; resolve(b); };
      const t = setTimeout(() => done(null), timeout);
      this._capture = { done, padPrev: null };
    });
  }

  _action(name) {
    const a = this.actions.get(name);
    if (!a) throw new Error(`InputMap: unknown action "${name}"`);
    return a;
  }
  _reindex() {
    this._boundKeys.clear();
    let mouse = false;
    for (const a of this.actions.values()) {
      a._liveBindings = a.bindings.some((b) => b.alwaysOn);
      for (const b of a.bindings) {
        if (b.kind === 'key') this._boundKeys.add(b.key);
        if (b.kind === 'mouseButton' || b.kind === 'mouseAxis') mouse = true;
      }
    }
    if (mouse) this._attachPointer();
  }
  _attachPointer() {
    if (this._pointerAttached) return;
    const t = this._pointerTarget || (typeof window !== 'undefined' ? window : null);
    if (!t || !t.addEventListener) return;
    this._pointerAttached = t;
    t.addEventListener('pointerdown', this._onPointerDown);
    t.addEventListener('pointermove', this._onPointerMove);
    t.addEventListener('wheel', this._onWheel, { passive: true });
    if (typeof window !== 'undefined') window.addEventListener('pointerup', this._onPointerUp);
  }

  // ------------------------------------------------------------------ contexts / virtual sources
  /** Enable or disable a named context (bindings with `when` filter on it). */
  setContext(name, on = true) { if (on) this.contexts.add(name); else this.contexts.delete(name); return this; }
  hasContext(name) { return this.contexts.has(name); }
  /** Write a virtual source (touch controls, scripted input, network). */
  setVirtual(name, v) {
    const prev = this.virtual.get(name) || 0;
    if (Math.abs(prev) <= 0.5 && Math.abs(v) > 0.5) this._virtualEdges.add(name);
    this.virtual.set(name, v);
  }

  // ------------------------------------------------------------------ queries
  value(name) { const a = this.actions.get(name); return a ? a.value : 0; }
  down(name) { const a = this.actions.get(name); return !!(a && a.down); }
  justPressed(name) { const a = this.actions.get(name); return !!(a && a.justPressed); }
  justReleased(name) { const a = this.actions.get(name); return !!(a && a.justReleased); }
  /** Raw key state by KeyboardEvent.code. */
  keyDown(code) { return this._keys.has(code); }
  /** Set of KeyboardEvent.code values currently held (read only). */
  get pressedKeys() { return this._keys; }
  /** Pointer deltas of the current frame. */
  get mouseDelta() { return this._mouseFrame; }

  // ------------------------------------------------------------------ events
  _onKeyDown(e) {
    if (isTypingTarget(e.target) || (typeof document !== 'undefined' && isTypingTarget(document.activeElement))) return;
    // keep browser shortcuts (Cmd+R etc.): keys pressed with a modifier held are ignored, the modifiers themselves count
    if ((e.metaKey || e.ctrlKey || e.altKey) && !MODIFIER_CODES.has(e.code)) return;
    if (this._capture && !e.repeat) { e.preventDefault(); this._capture.done({ key: e.code }); return; }
    const bound = this._boundKeys.has(e.code);
    if (bound && this.preventDefault) e.preventDefault();
    if (!e.repeat) this._keyEdges.add(e.code);
    this._keys.add(e.code);
  }
  _onKeyUp(e) {
    this._keys.delete(e.code);
    if (this._boundKeys.has(e.code) && this.preventDefault && !isTypingTarget(e.target)) e.preventDefault();
  }

  // ------------------------------------------------------------------ per frame
  _active(b) {
    for (let i = 0; i < b._req.length; i++) if (!this.contexts.has(b._req[i])) return false;
    for (let i = 0; i < b._not.length; i++) if (this.contexts.has(b._not[i])) return false;
    return true;
  }

  /** Raw value of one binding this frame. */
  _read(b) {
    const gp = this.gamepad;
    switch (b.kind) {
      case 'key': return this._keys.has(b.key) ? b.value : 0;
      case 'padButton': if (!gp.pad) return 0; return b.binary ? (gp.pressed(b.padButton) ? b.value : 0) : gp.button(b.padButton) * b.value;
      case 'padAxis': {
        if (!gp.pad) return 0;
        let v = gp.axis(b.padAxis, b.deadzone ?? gp.deadzone);
        if (b.half === 1) v = v > 0 ? v : 0; else if (b.half === -1) v = v < 0 ? -v : 0;
        return v * b.value;
      }
      case 'mouseButton': return this._mouseButtons & (1 << b.mouse) ? b.value : 0;
      case 'mouseAxis': return this._mouseFrame[b.mouse] * b.value;
      case 'virtual': return (this.virtual.get(b.virtual) || 0) * b.value;
    }
    return 0;
  }
  /** Binding counts as "held" (for button actions and edges). */
  _held(b) {
    if (b.kind === 'padButton') return !!this.gamepad.pad && this.gamepad.pressed(b.padButton);
    return Math.abs(this._read(b)) > 0.5;
  }
  _edge(b) {
    if (b.kind === 'key') return this._keyEdges.has(b.key);
    if (b.kind === 'mouseButton') return !!(this._mouseEdges & (1 << b.mouse));
    if (b.kind === 'virtual') return this._virtualEdges.has(b.virtual);
    return false;
  }

  /** Call once per rendered frame (dt optional; measured internally otherwise). */
  update(dt) {
    const now = typeof performance !== 'undefined' ? performance.now() : Date.now();
    if (!(dt > 0)) dt = this._lastTime ? (now - this._lastTime) / 1000 : 1 / 60;
    this._lastTime = now;
    dt = Math.min(Math.max(dt, 0), 0.1);
    this.dt = dt;

    const mf = this._mouseFrame, m = this._mouse;
    mf.x = m.x; mf.y = m.y; mf.wheel = m.wheel; m.x = m.y = m.wheel = 0;
    if (this.useGamepad) this.gamepadConnected = !!this.gamepad.poll();
    if (this._capture) this._pollCapture();

    const on = this.enabled;
    for (const a of this.actions.values()) {
      let pos = 0, neg = 0, analog = 0, held = false, edge = false, padHeld = false;
      const all = on || a.alwaysOn;
      if (all || a._liveBindings) {
        for (let i = 0; i < a.bindings.length; i++) {
          const b = a.bindings[i];
          if (!all && !b.alwaysOn) continue;
          if ((b._req.length || b._not.length) && !this._active(b)) continue;
          const v = this._read(b);
          if (b.smooth) { if (v > pos) pos = v; else if (v < neg) neg = v; } else analog += v;
          if (a.type === 'button' && this._held(b)) {
            held = true;
            if (b.kind === 'padButton' || b.kind === 'padAxis') padHeld = true;
          }
          if (!edge && this._edge(b)) edge = true;
        }
      }
      const digital = pos + neg;
      if (a.smoothing) a.smoothed = approach(a.smoothed, digital, dt, a.smoothing.up, a.smoothing.down);
      else a.smoothed = digital;
      analog = clamp(analog, -1, 1);
      a.analog = analog;
      a.prevDown = a.down;
      if (a.type === 'button') {
        a.down = held;
        a.value = held ? 1 : 0;
        // keys / mouse / virtual latch edges from events; the gamepad is polled (its own previous state)
        a.justPressed = edge || (padHeld && !a.padDown);
        a.padDown = padHeld;
      } else {
        const v = Math.abs(analog) > Math.abs(a.smoothed) ? analog : a.smoothed;
        a.value = clamp(v, a.min, a.max);
        a.down = Math.abs(a.value) > 0.5;
        a.justPressed = a.down && !a.prevDown;
      }
      a.justReleased = !a.down && a.prevDown;
    }
    this._keyEdges.clear();
    this._mouseEdges = 0;
    this._virtualEdges.clear();
    return this;
  }

  _pollCapture() {
    const gp = this.gamepad, c = this._capture;
    if (!gp.pad) return;
    const n = gp.pad.buttons.length, prev = c.padPrev;
    const cur = c.padPrev = [];
    for (let i = 0; i < n; i++) cur.push(gp.pressed(i));
    if (prev) for (let i = 0; i < n; i++) if (cur[i] && !prev[i]) { c.done({ padButton: i }); return; }
    for (let i = 0; i < gp.pad.axes.length; i++) {
      const v = gp.pad.axes[i] || 0;
      if (Math.abs(v) > 0.6) { c.done({ padAxis: i, value: Math.sign(v), half: Math.sign(v) }); return; }
    }
  }

  dispose() {
    if (this._target && this._target.removeEventListener) {
      this._target.removeEventListener('keydown', this._onKeyDown);
      this._target.removeEventListener('keyup', this._onKeyUp);
    }
    if (typeof window !== 'undefined') {
      window.removeEventListener('blur', this._onBlur);
      window.removeEventListener('gamepadconnected', this._onPadConnect);
      window.removeEventListener('gamepaddisconnected', this._onPadDisconnect);
      window.removeEventListener('pointerup', this._onPointerUp);
    }
    const t = this._pointerAttached;
    if (t) {
      t.removeEventListener('pointerdown', this._onPointerDown);
      t.removeEventListener('pointermove', this._onPointerMove);
      t.removeEventListener('wheel', this._onWheel);
      this._pointerAttached = false;
    }
  }
}
