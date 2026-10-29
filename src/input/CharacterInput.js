// Character control preset on top of InputMap: move / look / jump / sprint for third- and first-person games.
//
//   const input = new CharacterInput({ element: canvas });
//   const lock = new PointerLock({ element: canvas });              // optional: the game decides when to request
//   canvas.addEventListener('click', () => lock.request());
//   input.update(dt); const c = input.controls;                     // once per frame
//   yaw -= c.lookX; pitch = clamp(pitch - c.lookY, -1.4, 1.4);     // radians this frame
//
// `controls` (one object, reused by every update()):
//   moveX -1..1 (+1 = right), moveY -1..1 (+1 = forward)   keys ramp in / out (ramp), sticks raw; the pair is clamped
//                                                          to the unit circle (clampMove: false keeps the square)
//   lookX, lookY   radians to turn this frame, +x = right, +y = down (mouse convention; invertY flips). Mouse and
//                  touchpad drags are raw deltas (never smoothed) × mouseSpeed / touchSpeed rad/px; sticks are a turn
//                  rate (stickSpeed rad/s at full deflection) shaped by stickCurve and ramped by the action's
//                  `smoothing`; all × lookSensitivity. Fresh each update() (0 when nothing moved / map disabled).
//   jumpHeld, sprint, crouch                held
//   jump, interact, toggleCamera, menu      edges: true for exactly one update()
// Mouse look only flows while the pointer is locked (on `element`; any lock when `element` is the document) or the
// hold button (right mouse; holdButton) is down, so the page stays usable unlocked. `lookActive` reports that, or a
// touchpad drag. The update in which mouse look turns on drops its delta (cursor travel before the lock / press).
//
// Keyboard: WASD / arrows move, Space jump, Shift sprint, Ctrl / C crouch, E / F interact, V camera, Esc menu,
//           mouse look (locked, or right-drag). Ctrl+W closes the tab in most browsers: C is the safe crouch-walk.
// Gamepad (standard mapping): left stick move, right stick look, A jump, LS click / LB sprint, B crouch, X interact,
//           Y camera, Start menu.
// Touch (phones, or touch: true): left stick move, right half of the screen drags to look, JUMP / RUN / USE.
//   TouchControls has no drag-delta control, so 'touchpad' layout entries are built here: a transparent overlay
//   (below a 96 px top inset by default) under the sticks / buttons, adding its finger's pixel deltas to the virtual
//   sources `${id}.x` / `${id}.y`; mouse pointers are ignored on it (they look through right-drag / pointer lock).
//   It sits in TouchControls' root (z-index 30): interactive HUD elements over it need z-index > 30.
// Virtual sources bound to lookX / lookY are pixel deltas, consumed like mouse movement: update() reads them and
//   zeroes them, so scripted look writes map.setVirtual('look.x', px) once per frame (a level would turn forever).
// toggleCamera and menu are alwaysOn (they still fire while `enabled` is false). A custom `actions` map may leave
// actions out: missing ones read as 0 / false. Safe to import in Node (DOM is only touched when constructed in a page).
import { InputMap, approach, ramp } from './InputMap.js';
import { TouchControls, isTouchDevice } from './TouchControls.js';
import { PAD_BUTTONS as B, PAD_AXES as AX } from './Gamepad.js';
import { PointerLock } from './PointerLock.js';

const MOVE = ramp(0.1, 0.06);   // full speed in ~0.1 s, stop a little faster
const LOOK = ramp(0.1, 0.05);   // right-stick turn rate in / out (mouse deltas are never ramped)

/** Default action map. Copy and edit to rebind; pass as `actions` to CharacterInput. */
export const CHARACTER_ACTIONS = {
  moveX: { type: 'axis', range: [-1, 1], smoothing: MOVE, bindings: [
    { key: 'KeyD', value: 1 }, { key: 'ArrowRight', value: 1 }, { key: 'KeyA', value: -1 }, { key: 'ArrowLeft', value: -1 },
    { padAxis: AX.LX }, { virtual: 'stick.x' }] },
  moveY: { type: 'axis', range: [-1, 1], smoothing: MOVE, bindings: [
    { key: 'KeyW', value: 1 }, { key: 'ArrowUp', value: 1 }, { key: 'KeyS', value: -1 }, { key: 'ArrowDown', value: -1 },
    { padAxis: AX.LY, value: -1 }, { virtual: 'stick.y', value: -1 }] },
  lookX: { type: 'axis', smoothing: LOOK, bindings: [{ mouse: 'x' }, { padAxis: AX.RX }, { virtual: 'look.x' }] },
  lookY: { type: 'axis', smoothing: LOOK, bindings: [{ mouse: 'y' }, { padAxis: AX.RY }, { virtual: 'look.y' }] },
  jump: { type: 'button', bindings: ['Space', { padButton: B.A }, { virtual: 'jump' }] },
  sprint: { type: 'button', bindings: ['ShiftLeft', 'ShiftRight', { padButton: B.LS }, { padButton: B.LB }, { virtual: 'sprint' }] },
  crouch: { type: 'button', bindings: ['ControlLeft', 'KeyC', { padButton: B.B }, { virtual: 'crouch' }] },
  interact: { type: 'button', bindings: ['KeyE', 'KeyF', { padButton: B.X }, { virtual: 'interact' }] },
  toggleCamera: { type: 'button', alwaysOn: true, bindings: ['KeyV', { padButton: B.Y }, { virtual: 'cam' }] },
  menu: { type: 'button', alwaysOn: true, bindings: ['Escape', { padButton: B.START }, { virtual: 'menu' }] },
};

/**
 * Default touch layout (landscape phone). 'touchpad' entries are handled by CharacterInput:
 * { type: 'touchpad', id, side: 'left'|'right', width (fraction of the screen), top / bottom (px insets) }.
 * Other ids bound by CHARACTER_ACTIONS for custom layouts: 'crouch', 'cam', 'menu'.
 */
export const CHARACTER_TOUCH_LAYOUT = [
  { type: 'touchpad', id: 'look', side: 'right', width: 0.5, top: 96, bottom: 0 },
  { type: 'stick', id: 'stick', side: 'left', x: 120, y: 200, radius: 66 },
  { type: 'button', id: 'jump', label: 'JUMP', side: 'right', x: 74, y: 190, size: 76, fontSize: 12 },
  { type: 'button', id: 'sprint', label: 'RUN', side: 'right', x: 168, y: 160, size: 60 },
  { type: 'button', id: 'interact', label: 'USE', side: 'right', x: 74, y: 290, size: 58 },
];

// missing actions read as 0 / false (custom maps need not define every character action)
const val = (A, name) => { const a = A.get(name); return a ? a.value : 0; };
const held = (A, name) => { const a = A.get(name); return !!(a && a.down); };
const pressed = (A, name) => { const a = A.get(name); return !!(a && a.justPressed); };
const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const BUTTON_BIT = [1, 4, 2, 8, 16];   // MouseEvent.button index -> MouseEvent.buttons bit
const LOOK_ACTIONS = ['lookX', 'lookY'];

export class CharacterInput {
  /**
   * @param {object} [o]
   * @param {EventTarget} [o.element=document]  pointer target (canvas or document); also the pointer-lock element checked
   * @param {'auto'|boolean} [o.touch='auto']   on-screen controls ('auto' = touch-first devices only)
   * @param {boolean} [o.gamepad=true]          poll the gamepad
   * @param {object} [o.actions=CHARACTER_ACTIONS]
   * @param {Array} [o.touchLayout=CHARACTER_TOUCH_LAYOUT]
   * @param {number} [o.lookSensitivity=1]      multiplies every look source
   * @param {number} [o.mouseSpeed=0.002]       rad per mouse px
   * @param {number} [o.touchSpeed=0.005]       rad per touchpad px
   * @param {number} [o.stickSpeed=3]           rad/s at full stick deflection
   * @param {number} [o.stickCurve=1.6]         stick response exponent (1 = linear)
   * @param {boolean} [o.invertY=false]
   * @param {number|null} [o.holdButton=2]      mouse button that enables look while held (null: pointer lock only)
   * @param {boolean} [o.clampMove=true]        clamp (moveX, moveY) to the unit circle
   */
  constructor({
    element = typeof document !== 'undefined' ? document : null, touch = 'auto', gamepad = true,
    actions = CHARACTER_ACTIONS, touchLayout = CHARACTER_TOUCH_LAYOUT, lookSensitivity = 1,
    mouseSpeed = 0.002, touchSpeed = 0.005, stickSpeed = 3, stickCurve = 1.6, invertY = false,
    holdButton = 2, clampMove = true,
  } = {}) {
    this.element = element;
    this.lookSensitivity = lookSensitivity;
    this.mouseSpeed = mouseSpeed; this.touchSpeed = touchSpeed;
    this.stickSpeed = stickSpeed; this.stickCurve = stickCurve;
    this.invertY = invertY; this.holdButton = holdButton; this.clampMove = clampMove;
    this.controls = {
      moveX: 0, moveY: 0, lookX: 0, lookY: 0,
      jump: false, jumpHeld: false, sprint: false, crouch: false, interact: false, toggleCamera: false, menu: false,
    };
    this.lookActive = false;
    this.pointerLocked = false;
    this.gamepadConnected = false;

    this.map = new InputMap({ pointerTarget: element || undefined, gamepad, deadzone: 0.12, actions });
    this.touch = null;
    this._touchLayout = touchLayout;
    this._pads = [];
    this._src = { mouse: 0, virtual: 0, pad: 0, keyPos: 0, keyNeg: 0 };   // reused InputMap.readSources output
    this._rate = [0, 0];   // ramped stick turn rate per look axis
    this._holding = false;
    this._mouseWasLive = false;

    // hold-to-look: track the button from `buttons` (chorded presses arrive as pointermove, not pointerdown)
    this._onPointer = (e) => {
      if (e.pointerType === 'touch' || this.holdButton == null) return;
      const was = this._holding;
      this._holding = (e.buttons & BUTTON_BIT[this.holdButton]) !== 0;
      const el = this.element;
      if (this._holding && !was && e.type === 'pointerdown' && el && el.setPointerCapture) {
        try { el.setPointerCapture(e.pointerId); } catch { /* synthetic pointer */ }
      }
    };
    this._onBlur = () => { this._holding = false; };
    this._onContextMenu = (e) => {
      if (this.holdButton !== 2 || !this.map.enabled) return;
      const t = e.target;
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
      e.preventDefault();
    };
    if (typeof window !== 'undefined' && element && element.addEventListener) {
      element.addEventListener('pointerdown', this._onPointer);
      element.addEventListener('pointermove', this._onPointer);
      element.addEventListener('contextmenu', this._onContextMenu);
      window.addEventListener('pointerup', this._onPointer);
      window.addEventListener('blur', this._onBlur);
    }
    if (typeof document !== 'undefined' && (touch === true || (touch === 'auto' && isTouchDevice()))) this.enableTouch(true);
  }

  get enabled() { return this.map.enabled; }
  set enabled(v) { this.map.enabled = !!v; }

  /** Show / hide the on-screen touch controls (created on first use). */
  enableTouch(on = true) {
    if (on && !this.touch) {
      const L = this._touchLayout;
      this.touch = new TouchControls({ input: this.map, layout: L.filter((c) => c.type !== 'touchpad'), show: true });
      for (const c of L) if (c.type === 'touchpad') this._pads.push(this._touchpad(c));
    } else if (this.touch) {
      this.touch.setVisible(on);
      if (!on) for (const p of this._pads) { p.pointer = null; p.dx = p.dy = 0; }
    }
    return this.touch;
  }

  /** Transparent drag surface under the touch buttons; accumulates pixel deltas of one finger. */
  _touchpad(c) {
    const root = this.touch.root, el = document.createElement('div'), s = el.style;
    el.className = 'ui-touch-pad';
    s.position = 'absolute'; s.pointerEvents = 'auto'; s.touchAction = 'none'; s.webkitTapHighlightColor = 'transparent';
    s.top = `${c.top ?? 0}px`; s.bottom = `${c.bottom ?? 0}px`; s.width = `${(c.width ?? 0.5) * 100}%`;
    if (c.side === 'left') s.left = '0'; else s.right = '0';
    root.insertBefore(el, root.firstChild);   // earlier sibling = painted under the sticks / buttons
    const pad = { id: c.id, sx: `${c.id}.x`, sy: `${c.id}.y`, el, pointer: null, lx: 0, ly: 0, dx: 0, dy: 0 };
    el.addEventListener('pointerdown', (e) => {
      if (pad.pointer != null || e.pointerType === 'mouse') return;   // the mouse already looks (right-drag / lock)
      pad.pointer = e.pointerId; pad.lx = e.clientX; pad.ly = e.clientY;
      try { el.setPointerCapture(e.pointerId); } catch { /* synthetic pointer */ }
      e.preventDefault();
    });
    el.addEventListener('pointermove', (e) => {
      if (e.pointerId !== pad.pointer) return;
      pad.dx += e.clientX - pad.lx; pad.dy += e.clientY - pad.ly;
      pad.lx = e.clientX; pad.ly = e.clientY;
    });
    const end = (e) => { if (e.pointerId === pad.pointer) pad.pointer = null; };
    el.addEventListener('pointerup', end);
    el.addEventListener('pointercancel', end);
    el.addEventListener('lostpointercapture', end);
    return pad;
  }

  /**
   * One look axis this update, radians before sensitivity: raw deltas (mouse when `mouse`, virtual / touchpad px)
   * + stick and key turn rate (curved, ramped by the action's smoothing) × dt. Zeroes the action's virtual sources.
   */
  _look(i, mouse) {
    const m = this.map, r = this._rate, name = LOOK_ACTIONS[i], a = m.actions.get(name);
    if (!a) { r[i] = 0; return 0; }
    const S = m.readSources(name, this._src);
    const analog = clamp(S.pad, -1, 1), digital = S.keyPos + S.keyNeg;
    let t = Math.abs(analog) > Math.abs(digital) ? analog : digital;
    if (this.stickCurve !== 1 && t !== 0) t = Math.sign(t) * Math.abs(t) ** this.stickCurve;
    const sm = a.smoothing;
    r[i] = sm ? approach(r[i], t, m.dt, sm.up, sm.down) : t;
    for (let k = 0; k < a.bindings.length; k++) { const b = a.bindings[k]; if (b.kind === 'virtual') m.setVirtual(b.virtual, 0); }
    return (mouse ? S.mouse : 0) * this.mouseSpeed + S.virtual * this.touchSpeed + r[i] * this.stickSpeed * m.dt;
  }

  /** Call once per rendered frame (dt optional; measured internally otherwise). Fills `controls`; returns this. */
  update(dt) {
    const m = this.map, c = this.controls, pads = this._pads;
    let dragging = false;
    for (let i = 0; i < pads.length; i++) {
      const p = pads[i];
      m.setVirtual(p.sx, (m.virtual.get(p.sx) || 0) + p.dx); m.setVirtual(p.sy, (m.virtual.get(p.sy) || 0) + p.dy);
      p.dx = p.dy = 0;
      if (p.pointer != null) dragging = true;
    }
    m.update(dt);
    const A = m.actions;

    let x = val(A, 'moveX'), y = val(A, 'moveY');
    if (this.clampMove) { const l = Math.hypot(x, y); if (l > 1) { x /= l; y /= l; } }
    c.moveX = x; c.moveY = y;

    this.pointerLocked = PointerLock.isLocked(this.element);
    const live = this.pointerLocked || this._holding;
    const mouse = live && this._mouseWasLive;   // the update in which mouse look turns on drops its delta
    this._mouseWasLive = live;
    const s = this.lookSensitivity;
    c.lookX = this._look(0, mouse) * s;
    c.lookY = this._look(1, mouse) * s * (this.invertY ? -1 : 1);
    this.lookActive = live || dragging;

    c.jump = pressed(A, 'jump');
    c.jumpHeld = held(A, 'jump');
    c.sprint = held(A, 'sprint');
    c.crouch = held(A, 'crouch');
    c.interact = pressed(A, 'interact');
    c.toggleCamera = pressed(A, 'toggleCamera');
    c.menu = pressed(A, 'menu');
    this.gamepadConnected = m.gamepadConnected;
    return this;
  }

  dispose() {
    this.map.dispose();
    if (this.touch) { this.touch.dispose(); this.touch = null; }
    this._pads.length = 0;
    const el = this.element;
    if (typeof window !== 'undefined' && el && el.removeEventListener) {
      el.removeEventListener('pointerdown', this._onPointer);
      el.removeEventListener('pointermove', this._onPointer);
      el.removeEventListener('contextmenu', this._onContextMenu);
      window.removeEventListener('pointerup', this._onPointer);
      window.removeEventListener('blur', this._onBlur);
    }
  }
}
