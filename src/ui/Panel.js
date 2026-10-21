// UILayer — full-screen, click-through overlay root with resolution-aware scale (--ui-scale).
// Panel   — container, optionally glass and pinned to a corner (scaled with the layer).
//
//   const layer = new UILayer({ parent: document.body });              // .ui-root
//   const panel = new Panel({ parent: layer, corner: 'br', className: 'my-panel' });
//   new Gauge({ parent: panel, ... });
//   layer.onScale((s) => minimap.setScale(s));
import { Widget } from './Widget.js';
import { el, clamp, host } from './dom.js';

export class UILayer extends Widget {
  /**
   * @param {object} [o]
   * @param {HTMLElement} [o.parent=document.body]
   * @param {string} [o.className]  extra classes on the root
   * @param {object|false} [o.autoScale]  { refW: 1440, refH: 900, minW: 820, min: 0.45, max: 1.25 }:
   *        scale = clamp(min(w / refW, h / refH, w / minW), min, max)
   */
  constructor(o = {}) {
    super(o);
    this.el = el('div', `ui-root${o.className ? ` ${o.className}` : ''}`);
    if (o.zIndex != null) this.el.style.zIndex = String(o.zIndex);
    this.mount(o.parent || document.body);
    this.autoScale = o.autoScale === false ? null : { refW: 1440, refH: 900, minW: 820, min: 0.45, max: 1.25, ...(o.autoScale || {}) };
    this.scale = 1;
    this._scaleCbs = [];
    this._onResize = this._onResize.bind(this);
    if (this.autoScale) { window.addEventListener('resize', this._onResize); this._onResize(); }
  }

  /** cb(scale) now and on every resize. */
  onScale(cb) { this._scaleCbs.push(cb); cb(this.scale); return this; }

  _onResize() {
    const a = this.autoScale, w = window.innerWidth, h = window.innerHeight;
    const s = clamp(Math.min(w / a.refW, h / a.refH, w / a.minW), a.min, a.max);
    this.scale = s;
    this.el.style.setProperty('--ui-scale', s.toFixed(3));
    for (const cb of this._scaleCbs) cb(s);
  }

  dispose() {
    window.removeEventListener('resize', this._onResize);
    super.dispose();
  }
}

export class Panel extends Widget {
  /**
   * @param {object} [o]
   * @param {HTMLElement|Widget} [o.parent]
   * @param {'br'|'bl'|'tl'|'tr'|null} [o.corner]
   * @param {boolean} [o.glass=true]
   * @param {string} [o.className]
   * @param {object} [o.style]  inline style overrides
   */
  constructor(o = {}) {
    super(o);
    const cls = ['ui-w', 'ui-panel'];
    if (o.corner) cls.push('ui-corner', `ui-${o.corner}`);
    if (o.glass !== false) cls.push('ui-glass');
    if (o.className) cls.push(o.className);
    this.el = el(o.tag || 'div', cls.join(' '));
    if (o.style) Object.assign(this.el.style, o.style);
    this.mountTo(o.parent);
  }
  /** Append a widget or element. */
  add(child) { const h = host(child); if (child && child.el) child.mount(this); else if (h) this.el.appendChild(h); return child; }
}
