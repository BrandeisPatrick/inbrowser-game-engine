// Vehicle control preset on top of InputMap: driving + thruster flight (the Roadster scheme).
//
// Analog outputs:  throttle 0..1, brake 0..1, steer -1..1 (+1 = right), thrust 0..1,
//                  pitch -1..1 (+1 = nose up), roll -1..1 (+1 = bank right, mirrors steer),
//                  yaw -1..1 (+1 = yaw right)
// Bool outputs:    boost, handbrake, lookBack
// Edge outputs:    toggleCamera, reset, toggleHelp — true for exactly one update() call.
//
// Keyboard: W/S ↑/↓ throttle/brake (brake from standstill = reverse), A/D ←/→ steer / roll,
//           Space thrust, Shift boost, Q/E yaw, I/K pitch (I = nose down, K = nose up),
//           X handbrake, C camera, R reset, H help, B look back.
//           While airborne (vehicle calls setFlightMode(true)) ↑/↓ pitch instead of throttle/brake
//           (the 'flight' input context).
// Gamepad (standard mapping): left stick X steer/roll, left stick Y pitch (push = nose down),
//           RT throttle, LT brake, A thrust, B boost, Y camera, X reset, right stick X / LB-RB yaw,
//           right-stick click handbrake, Back/Select help, D-pad down look back.
// Touch (phones, or touch: true): left stick steer/pitch, GAS / BRAKE / LIFT / BOOST buttons, CAM / RESET / HELP.
// A custom `actions` map may leave any of these actions out (e.g. no flight actions for a ground vehicle):
// missing actions read as 0 / false.
import { InputMap, ramp } from './InputMap.js';
import { TouchControls, isTouchDevice } from './TouchControls.js';
import { PAD_BUTTONS as B, PAD_AXES as AX } from './Gamepad.js';

const STEER = ramp(0.15, 0.07);   // full deflection in ~0.15 s, faster re-centring
const PEDAL = ramp(0.12, 0.06);
const AXIS = ramp(0.2, 0.1);      // pitch / yaw keys
const THRUST = ramp(0.1, 0.1);

/** Default action map. Copy and edit to rebind; pass as `actions` to VehicleInput. */
export const VEHICLE_ACTIONS = {
  throttle: { type: 'axis', range: [0, 1], smoothing: PEDAL, bindings: [
    'KeyW', { key: 'ArrowUp', when: '!flight' }, { padButton: B.RT }, { virtual: 'gas' }] },
  brake: { type: 'axis', range: [0, 1], smoothing: PEDAL, bindings: [
    'KeyS', { key: 'ArrowDown', when: '!flight' }, { padButton: B.LT }, { virtual: 'brake' }] },
  steer: { type: 'axis', range: [-1, 1], smoothing: STEER, bindings: [
    { key: 'KeyD', value: 1 }, { key: 'ArrowRight', value: 1 }, { key: 'KeyA', value: -1 }, { key: 'ArrowLeft', value: -1 },
    { padAxis: AX.LX }, { virtual: 'stick.x' }] },
  pitch: { type: 'axis', range: [-1, 1], smoothing: AXIS, bindings: [
    { key: 'KeyK', value: 1 }, { key: 'ArrowDown', value: 1, when: 'flight' },
    { key: 'KeyI', value: -1 }, { key: 'ArrowUp', value: -1, when: 'flight' },
    { padAxis: AX.LY }, { virtual: 'stick.y' }] },
  yaw: { type: 'axis', range: [-1, 1], smoothing: AXIS, bindings: [
    { key: 'KeyE', value: 1 }, { key: 'KeyQ', value: -1 },
    { padAxis: AX.RX }, { padButton: B.RB, value: 1, binary: true }, { padButton: B.LB, value: -1, binary: true }] },
  thrust: { type: 'axis', range: [0, 1], smoothing: THRUST, bindings: ['Space', { padButton: B.A }, { virtual: 'lift' }] },
  boost: { type: 'button', bindings: ['ShiftLeft', 'ShiftRight', { padButton: B.B }, { virtual: 'boost' }] },
  handbrake: { type: 'button', bindings: ['KeyX', { padButton: B.RS }] },
  lookBack: { type: 'button', bindings: ['KeyB', { padButton: B.DOWN }] },
  toggleCamera: { type: 'button', bindings: ['KeyC', { padButton: B.Y }, { virtual: 'cam' }] },
  reset: { type: 'button', bindings: ['KeyR', { padButton: B.X }, { virtual: 'reset' }] },
  toggleHelp: { type: 'button', bindings: ['KeyH', { padButton: B.BACK }, { padButton: B.START }, { virtual: 'help' }] },
};

/** Default touch layout (landscape phone; clear of the HUD corner panels at phone scale). */
export const VEHICLE_TOUCH_LAYOUT = [
  { type: 'stick', id: 'stick', side: 'left', x: 120, y: 220, radius: 66 },
  { type: 'button', id: 'gas', label: 'GAS', side: 'right', x: 70, y: 230, size: 76 },
  { type: 'button', id: 'brake', label: 'BRK', side: 'right', x: 162, y: 200, size: 60 },
  { type: 'button', id: 'lift', label: 'LIFT', side: 'right', x: 70, y: 326, size: 66 },
  { type: 'button', id: 'boost', label: 'BST', side: 'right', x: 160, y: 290, size: 54 },
  { type: 'button', id: 'cam', label: 'CAM', side: 'right', x: 36, y: 130, size: 46, top: true, fontSize: 10 },
  { type: 'button', id: 'reset', label: 'RST', side: 'right', x: 36, y: 184, size: 46, top: true, fontSize: 10 },
  { type: 'button', id: 'help', label: '?', side: 'left', x: 36, y: 150, size: 40, top: true },
];

// missing actions read as 0 / false (custom maps need not define every vehicle action)
const val = (A, name) => { const a = A.get(name); return a ? a.value : 0; };
const held = (A, name) => { const a = A.get(name); return !!(a && a.down); };
const pressed = (A, name) => { const a = A.get(name); return !!(a && a.justPressed); };

export class VehicleInput {
  /**
   * @param {HTMLElement} [domElement]  pointer target (kept for API compatibility)
   * @param {object} [opts]
   * @param {object} [opts.actions=VEHICLE_ACTIONS]
   * @param {'auto'|boolean} [opts.touch='auto']  on-screen controls ('auto' = touch-first devices only)
   * @param {Array} [opts.touchLayout=VEHICLE_TOUCH_LAYOUT]
   */
  constructor(domElement = null, opts = {}) {
    this.domElement = domElement;
    // public state
    this.throttle = 0; this.brake = 0; this.steer = 0; this.thrust = 0;
    this.pitch = 0; this.roll = 0; this.yaw = 0;
    this.boost = false; this.handbrake = false; this.lookBack = false;
    this.toggleCamera = false; this.reset = false; this.toggleHelp = false;
    this.gamepadConnected = false;
    this.flightMode = false;

    this.map = new InputMap({ pointerTarget: domElement || undefined, deadzone: 0.12, actions: opts.actions || VEHICLE_ACTIONS });
    this.touch = null;
    this._touchLayout = opts.touchLayout || VEHICLE_TOUCH_LAYOUT;
    const t = opts.touch ?? 'auto';
    if (typeof document !== 'undefined' && (t === true || (t === 'auto' && isTouchDevice()))) this.enableTouch(true);
  }

  /** Vehicle calls this so ↑/↓ map to pitch while airborne. */
  setFlightMode(on) { this.flightMode = !!on; this.map.setContext('flight', this.flightMode); }

  /** Show / hide the on-screen touch controls (created on first use). */
  enableTouch(on = true) {
    if (on && !this.touch) this.touch = new TouchControls({ input: this.map, layout: this._touchLayout, show: true });
    else if (this.touch) this.touch.setVisible(on);
    return this.touch;
  }

  /** Call once per rendered frame (dt optional; measured internally otherwise). */
  update(dt) {
    const m = this.map.update(dt);
    const A = m.actions;
    this.steer = val(A, 'steer');
    this.roll = this.steer;
    this.throttle = val(A, 'throttle');
    this.brake = val(A, 'brake');
    this.pitch = val(A, 'pitch');
    this.yaw = val(A, 'yaw');
    this.thrust = val(A, 'thrust');
    this.boost = held(A, 'boost');
    this.handbrake = held(A, 'handbrake');
    this.lookBack = held(A, 'lookBack');
    this.toggleCamera = pressed(A, 'toggleCamera');
    this.reset = pressed(A, 'reset');
    this.toggleHelp = pressed(A, 'toggleHelp');
    this.gamepadConnected = m.gamepadConnected;
    return this;
  }

  dispose() {
    this.map.dispose();
    if (this.touch) { this.touch.dispose(); this.touch = null; }
  }
}
