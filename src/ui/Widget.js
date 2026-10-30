// Widget base: owns a root element, mounts into an element or another widget, caches DOM writes,
// and throttles text updates (DOM text is the expensive part of a per-frame HUD).
//
//   class Speed extends Widget { constructor(o) { super(o); this.el = el('div', 'speed'); this.mountTo(o.parent); }
//     update(dt, kmh) { if (this.tickText(dt)) this.setText('v', this.el, String(Math.round(kmh))); } }
import { injectTheme } from './theme.js';
import { host } from './dom.js';

export class Widget {
  /**
   * @param {object} [o]
   * @param {HTMLElement|Widget} [o.parent]  mount target (subclasses mount after building their DOM)
   * @param {number} [o.textHz=15]           tickText() rate
   */
  constructor(o = {}) {
    injectTheme();
    this.el = null;
    this.textHz = o.textHz ?? 15;
    this._textTimer = 1;               // first tickText() fires immediately
    this._cache = Object.create(null);
  }

  /** Append this widget's element to parent (element or widget). */
  mount(parent) { const h = host(parent); if (h && this.el) h.appendChild(this.el); return this; }
  mountTo(parent) { return parent ? this.mount(parent) : this; }
  unmount() { if (this.el) this.el.remove(); return this; }

  /** Accumulate dt; true at most textHz times per second (true on the first call). */
  tickText(dt) {
    this._textTimer += dt;
    if (this._textTimer >= 1 / this.textHz) { this._textTimer = 0; return true; }
    return false;
  }

  setText(key, node, str) { if (this._cache[key] !== str) { this._cache[key] = str; node.textContent = str; } }
  setHTML(key, node, str) { if (this._cache[key] !== str) { this._cache[key] = str; node.innerHTML = str; } }
  setAttr(key, node, name, value) { if (this._cache[key] !== value) { this._cache[key] = value; node.setAttribute(name, value); } }
  setStyle(key, node, prop, value) { if (this._cache[key] !== value) { this._cache[key] = value; node.style[prop] = value; } }
  toggleClass(key, node, cls, on) { on = !!on; if (this._cache[key] !== on) { this._cache[key] = on; node.classList.toggle(cls, on); } }

  setVisible(on) { if (this.el) this.el.style.display = on ? '' : 'none'; return this; }

  /**
   * Per-frame update; subclasses take further positional args (see each widget).
   * @param {number} dt
   * @param {...*} rest
   */
  // eslint-disable-next-line no-unused-vars
  update(dt, ...rest) {}

  dispose() { this.unmount(); }
}
