// On-screen virtual sticks and buttons for touch screens (DOM + pointer events).
// Writes into an InputMap's virtual sources, so actions bind to them with { virtual: 'stick.x' }:
//   stick  id -> `${id}.x`, `${id}.y`   (-1..1, +x right, +y down — same convention as gamepad axes)
//   button id -> `${id}`                 (0 / 1)
//
//   const touch = new TouchControls({ input, layout: [
//     { type: 'stick', id: 'stick', side: 'left', x: 110, y: 110, radius: 64 },
//     { type: 'button', id: 'jump', label: 'A', side: 'right', x: 70, y: 90, size: 68 },
//   ], show: 'auto' });   // 'auto' = only on coarse-pointer (touch-first) devices
//
// x / y are the centre's distance in CSS px from the chosen bottom corner (or top corner with top: true).

const CSS = `
.ui-touch{position:fixed;inset:0;pointer-events:none;z-index:30;user-select:none;-webkit-user-select:none;touch-action:none}
.ui-touch.hidden{display:none}
.ui-touch-stick,.ui-touch-btn{position:absolute;pointer-events:auto;touch-action:none;-webkit-tap-highlight-color:transparent}
.ui-touch-stick{border-radius:50%;background:rgba(255,255,255,.07);border:1.5px solid rgba(255,255,255,.22);
  box-shadow:inset 0 0 24px rgba(0,0,0,.35);-webkit-backdrop-filter:blur(6px);backdrop-filter:blur(6px)}
.ui-touch-knob{position:absolute;left:50%;top:50%;border-radius:50%;background:rgba(255,255,255,.28);
  border:1.5px solid rgba(255,255,255,.55);transform:translate(-50%,-50%);transition:background .15s}
.ui-touch-stick.on .ui-touch-knob{background:rgba(255,255,255,.45)}
.ui-touch-btn{display:flex;align-items:center;justify-content:center;border-radius:50%;
  background:rgba(255,255,255,.08);border:1.5px solid rgba(255,255,255,.25);color:rgba(255,255,255,.85);
  font:600 13px/1 -apple-system,system-ui,sans-serif;letter-spacing:.08em;transition:background .1s,transform .1s;
  -webkit-backdrop-filter:blur(6px);backdrop-filter:blur(6px)}
.ui-touch-btn.on{background:rgba(var(--ui-accent-rgb,232,33,39),.45);border-color:var(--ui-accent,#e82127);transform:scale(.94)}
`;

let styleInjected = false;
function injectCSS() {
  if (styleInjected || typeof document === 'undefined') return;
  const st = document.createElement('style');
  st.id = 'ui-touch-style'; st.textContent = CSS;
  document.head.appendChild(st);
  styleInjected = true;
}

function capture(node, id) { try { node.setPointerCapture(id); } catch { /* synthetic / already released pointer */ } }

/** True on touch-first devices (phones/tablets), false on mouse/trackpad machines. */
export function isTouchDevice() {
  if (typeof window === 'undefined' || !window.matchMedia) return false;
  return window.matchMedia('(pointer: coarse)').matches && !window.matchMedia('(any-pointer: fine)').matches;
}

export class TouchControls {
  /**
   * @param {object} o
   * @param {import('./InputMap.js').InputMap} o.input
   * @param {Array<object>} o.layout
   * @param {HTMLElement} [o.container=document.body]
   * @param {'auto'|boolean} [o.show='auto']
   */
  constructor(o = {}) {
    this.input = o.input;
    this.layout = o.layout || [];
    this.container = o.container || document.body;
    injectCSS();
    this.root = document.createElement('div');
    this.root.className = 'ui-touch';
    this.container.appendChild(this.root);
    this.controls = [];
    for (const c of this.layout) this.controls.push(c.type === 'stick' ? this._stick(c) : this._button(c));
    const show = o.show ?? 'auto';
    this.setVisible(show === 'auto' ? isTouchDevice() : !!show);
  }

  get visible() { return !this.root.classList.contains('hidden'); }
  setVisible(on) {
    this.root.classList.toggle('hidden', !on);
    if (!on) this.releaseAll();
  }

  _place(node, c, w, h) {
    const s = node.style;
    s.width = `${w}px`; s.height = `${h}px`;
    const x = c.x ?? 100, y = c.y ?? 100;
    if (c.side === 'right') s.right = `${x - w / 2}px`; else s.left = `${x - w / 2}px`;
    if (c.top) s.top = `${y - h / 2}px`; else s.bottom = `${y - h / 2}px`;
  }

  _stick(c) {
    const r = c.radius ?? 64, knobR = c.knob ?? Math.round(r * 0.42);
    const base = document.createElement('div');
    base.className = 'ui-touch-stick';
    this._place(base, c, r * 2, r * 2);
    const knob = document.createElement('div');
    knob.className = 'ui-touch-knob';
    knob.style.width = knob.style.height = `${knobR * 2}px`;
    base.appendChild(knob);
    this.root.appendChild(base);
    const ctl = { type: 'stick', id: c.id, el: base, pointer: null, x: 0, y: 0 };
    const set = (x, y) => {
      ctl.x = x; ctl.y = y;
      knob.style.transform = `translate(calc(-50% + ${(x * r).toFixed(1)}px), calc(-50% + ${(y * r).toFixed(1)}px))`;
      this.input && this.input.setVirtual(`${c.id}.x`, x);
      this.input && this.input.setVirtual(`${c.id}.y`, y);
    };
    const move = (e) => {
      const b = base.getBoundingClientRect();
      let dx = (e.clientX - (b.left + b.width / 2)) / r, dy = (e.clientY - (b.top + b.height / 2)) / r;
      const l = Math.hypot(dx, dy);
      if (l > 1) { dx /= l; dy /= l; }
      set(dx, dy);
    };
    base.addEventListener('pointerdown', (e) => {
      if (ctl.pointer != null) return;
      ctl.pointer = e.pointerId; capture(base, e.pointerId); base.classList.add('on');
      move(e); e.preventDefault();
    });
    base.addEventListener('pointermove', (e) => { if (e.pointerId === ctl.pointer) move(e); });
    const end = (e) => { if (e.pointerId !== ctl.pointer) return; ctl.pointer = null; base.classList.remove('on'); set(0, 0); };
    base.addEventListener('pointerup', end);
    base.addEventListener('pointercancel', end);
    ctl.release = () => { ctl.pointer = null; base.classList.remove('on'); set(0, 0); };
    return ctl;
  }

  _button(c) {
    const size = c.size ?? 64;
    const b = document.createElement('div');
    b.className = 'ui-touch-btn';
    b.textContent = c.label ?? c.id;
    this._place(b, c, size, size);
    if (c.fontSize) b.style.fontSize = `${c.fontSize}px`;
    this.root.appendChild(b);
    const ctl = { type: 'button', id: c.id, el: b, pointers: new Set() };
    const sync = () => {
      const on = ctl.pointers.size > 0;
      b.classList.toggle('on', on);
      this.input && this.input.setVirtual(c.id, on ? 1 : 0);
    };
    b.addEventListener('pointerdown', (e) => { ctl.pointers.add(e.pointerId); capture(b, e.pointerId); sync(); e.preventDefault(); });
    const end = (e) => { ctl.pointers.delete(e.pointerId); sync(); };
    b.addEventListener('pointerup', end);
    b.addEventListener('pointercancel', end);
    ctl.release = () => { ctl.pointers.clear(); sync(); };
    return ctl;
  }

  releaseAll() { for (const c of this.controls) c.release(); }

  dispose() { this.releaseAll(); this.root.remove(); }
}
