// Gauge — SVG arc gauge with ticks, a centre slot and a footer label.
//   const g = new Gauge({ parent, sweep: 240, radius: 88, footer: 'POWER' });
//   g.center.append(...);  g.setValue(0..1);  g.setColor('#fff', 'drop-shadow(...)');  g.setFooter('POWER 40%')
import { Widget } from './Widget.js';
import { injectStyle } from './theme.js';
import { el, svgEl, polar, arcPath, clamp } from './dom.js';

const CSS = `
.ui-gauge{position:relative}
.ui-gauge svg{position:absolute;inset:0;width:100%;height:100%;overflow:visible}
.ui-gauge-center{position:absolute;left:0;right:0;text-align:center}
.ui-gauge-foot{position:absolute;bottom:4px;left:0;right:0;text-align:center}
`;

export class Gauge extends Widget {
  /**
   * @param {object} [o]  width=220 height=200 cx=110 cy=108 radius=88 sweep=240 thickness=8
   *   ticks=12 majorEvery=3 trackColor tickColor color='#fff' glow centerTop=30 footer=null minVisible=0.008
   */
  constructor(o = {}) {
    super(o);
    injectStyle('ui-gauge', CSS);
    const W = o.width ?? 220, H = o.height ?? 200, cx = o.cx ?? W / 2, cy = o.cy ?? 108, R = o.radius ?? 88;
    const sweep = o.sweep ?? 240, a0 = -sweep / 2, a1 = sweep / 2, thick = o.thickness ?? 8;
    this.length = (sweep / 360) * 2 * Math.PI * R;
    this.minVisible = o.minVisible ?? 0.008;
    const root = this.el = el('div', `ui-w ui-gauge${o.className ? ` ${o.className}` : ''}`);
    root.style.width = `${W}px`; root.style.height = `${H}px`;
    const svg = this.svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}` }, root);
    svgEl('path', { d: arcPath(cx, cy, R, a0, a1), fill: 'none', stroke: o.trackColor || 'rgba(255,255,255,.09)', 'stroke-width': thick, 'stroke-linecap': 'round' }, svg);
    const ticks = o.ticks ?? 12, major = o.majorEvery ?? 3;
    for (let i = 0; i <= ticks; i++) {
      const a = a0 + (sweep * i) / ticks;
      const [x0, y0] = polar(cx, cy, R + 9, a);
      const [x1, y1] = polar(cx, cy, R + (i % major === 0 ? 15 : 12), a);
      svgEl('line', { x1: x0, y1: y0, x2: x1, y2: y1, stroke: o.tickColor || 'rgba(255,255,255,.25)', 'stroke-width': i % major === 0 ? 1.5 : 1 }, svg);
    }
    const L = this.length;
    this.arc = svgEl('path', {
      d: arcPath(cx, cy, R, a0, a1), fill: 'none', stroke: o.color || '#fff', 'stroke-width': thick, 'stroke-linecap': 'round',
      'stroke-dasharray': `${L} ${L + 10}`, 'stroke-dashoffset': L,
    }, svg);
    this.arc.style.transition = 'stroke .25s';
    this.arc.style.filter = o.glow ?? 'drop-shadow(0 0 6px rgba(255,255,255,.35))';
    this.center = el('div', 'ui-gauge-center', root);
    this.center.style.top = `${o.centerTop ?? 30}px`;
    this.footer = o.footer != null ? el('div', 'ui-label ui-gauge-foot', root, o.footer) : null;
    this.mountTo(o.parent);
  }

  /** v in 0..1 (call every frame; the arc hides below minVisible so the round cap doesn't leave a dot). */
  setValue(v) {
    v = clamp(v, 0, 1);
    this.arc.setAttribute('stroke-dashoffset', (this.length * (1 - v)).toFixed(1));
    const on = v > this.minVisible;
    if (on !== this._cache.on) { this._cache.on = on; this.arc.style.opacity = on ? '1' : '0'; }
  }
  setColor(color, glow) {
    if (color === this._cache.color) return;
    this._cache.color = color;
    this.arc.setAttribute('stroke', color);
    if (glow != null) this.arc.style.filter = glow;
  }
  setFooter(text) { if (this.footer) this.setText('foot', this.footer, text); }
}
