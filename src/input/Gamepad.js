// Gamepad polling with deadzones (W3C "standard" mapping). No allocations per poll.
// Safe to import in Node (navigator is only touched inside poll()).

/** Standard-mapping button indices. */
export const PAD_BUTTONS = Object.freeze({
  A: 0, B: 1, X: 2, Y: 3, LB: 4, RB: 5, LT: 6, RT: 7, BACK: 8, START: 9,
  LS: 10, RS: 11, UP: 12, DOWN: 13, LEFT: 14, RIGHT: 15, HOME: 16,
});
/** Standard-mapping axis indices (Y axes are +1 when pushed down / pulled back). */
export const PAD_AXES = Object.freeze({ LX: 0, LY: 1, RX: 2, RY: 3 });

/** Radial-free scaled deadzone: |v| < dz -> 0, otherwise rescaled so the output still reaches ±1. */
export function applyDeadzone(v, dz = 0.12) {
  const a = Math.abs(v);
  if (a < dz) return 0;
  return Math.sign(v) * Math.min(1, (a - dz) / (1 - dz));
}

export class GamepadState {
  /**
   * @param {object} [o]
   * @param {number} [o.deadzone=0.12]  axis deadzone
   * @param {number|null} [o.index=null]  pad index to read; null = first connected pad
   */
  constructor(o = {}) {
    this.deadzone = o.deadzone ?? 0.12;
    this.index = o.index ?? null;
    this.pad = null;
    this.connected = false;
    this.id = '';
  }

  /** Read the current pad snapshot (call once per frame). Returns the Gamepad or null. */
  poll() {
    let pad = null;
    if (typeof navigator !== 'undefined' && navigator.getGamepads) {
      const pads = navigator.getGamepads();
      if (this.index != null) { const p = pads[this.index]; if (p && p.connected) pad = p; } else {
        for (const p of pads) if (p && p.connected) { pad = p; break; }
      }
    }
    this.pad = pad;
    this.connected = !!pad;
    this.id = pad ? pad.id : '';
    return pad;
  }

  /** Analog button value 0..1 (triggers are analog; face buttons 0/1). */
  button(i) {
    const b = this.pad && this.pad.buttons[i];
    if (!b) return 0;
    return typeof b === 'object' ? b.value || (b.pressed ? 1 : 0) : b;
  }

  /** Button considered down (pressed flag or value past half travel). */
  pressed(i) {
    const b = this.pad && this.pad.buttons[i];
    return !!(b && (b.pressed || b.value > 0.5));
  }

  /** Axis value with deadzone applied (dz overrides the default). */
  axis(i, dz = this.deadzone) {
    const v = this.pad ? this.pad.axes[i] || 0 : 0;
    return applyDeadzone(v, dz);
  }

  /** Best-effort rumble (Chrome `vibrationActuator`). strength 0..1, duration ms. */
  rumble(strength = 0.5, duration = 120) {
    const a = this.pad && this.pad.vibrationActuator;
    if (!a || !a.playEffect) return;
    try { a.playEffect('dual-rumble', { duration, strongMagnitude: strength, weakMagnitude: strength * 0.6 }); } catch { /* unsupported */ }
  }
}
