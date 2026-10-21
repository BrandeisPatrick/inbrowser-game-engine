// GMeter — 2D acceleration dot (friction-circle style) with a label / value column.
//   const g = new GMeter({ parent, label: 'G-force', range: 2.5 });
//   g.update(dt, lateralG, longitudinalG) every frame (smoothed);  g.setValueText('1.2')
import { Widget } from './Widget.js';
import { injectStyle, theme } from './theme.js';
import { el, svgEl, clamp } from './dom.js';
import { Readout } from './Readout.js';

const CSS = `
.ui-gmeter{display:flex;align-items:center;gap:12px}
.ui-gmeter svg{width:78px;height:78px;flex:none;overflow:visible}
`;

export class GMeter extends Widget {
  /** @param {object} [o]  label='G-force', unit='g', range=2.5 (g at the rim), smoothing=12 (1/s), accent, accentRgb */
  constructor(o = {}) {
    super(o);
    injectStyle('ui-gmeter', CSS);
    const accent = o.accent || theme.accent, rgb = o.accentRgb || theme.accentRgb;
    this.range = o.range ?? 2.5;
    this.smoothing = o.smoothing ?? 12;
    this.gx = 0; this.gy = 0;
    const root = this.el = el('div', `ui-w ui-gmeter${o.className ? ` ${o.className}` : ''}`);
    const gsvg = svgEl('svg', { viewBox: '-40 -40 80 80' }, root);
    svgEl('circle', { cx: 0, cy: 0, r: 36, fill: 'rgba(255,255,255,.03)', stroke: 'rgba(255,255,255,.25)', 'stroke-width': 1 }, gsvg);
    svgEl('circle', { cx: 0, cy: 0, r: 24, fill: 'none', stroke: 'rgba(255,255,255,.12)', 'stroke-width': 1 }, gsvg);
    svgEl('circle', { cx: 0, cy: 0, r: 12, fill: 'none', stroke: 'rgba(255,255,255,.12)', 'stroke-width': 1 }, gsvg);
    svgEl('line', { x1: -36, y1: 0, x2: 36, y2: 0, stroke: 'rgba(255,255,255,.12)', 'stroke-width': 1 }, gsvg);
    svgEl('line', { x1: 0, y1: -36, x2: 0, y2: 36, stroke: 'rgba(255,255,255,.12)', 'stroke-width': 1 }, gsvg);
    this.trail = svgEl('circle', { cx: 0, cy: 0, r: 7, fill: `rgba(${rgb},.18)` }, gsvg);
    this.dot = svgEl('circle', { cx: 0, cy: 0, r: 4, fill: accent }, gsvg);
    this.dot.style.filter = `drop-shadow(0 0 4px rgba(${rgb},.9))`;
    this.readout = new Readout({ parent: root, label: o.label ?? 'G-force', unit: o.unit ?? 'g', value: '0.0', size: 'md' });
    this.mountTo(o.parent);
  }

  /** gx: lateral (+right), gy: longitudinal (+accelerating), in g. */
  update(dt, gx, gy) {
    const R = this.range;
    gx = clamp(gx, -R, R); gy = clamp(gy, -R, R);
    const k = 1 - Math.exp(-this.smoothing * dt);
    this.gx += (gx - this.gx) * k;
    this.gy += (gy - this.gy) * k;
    const s = 36 / R;
    let dx = this.gx * s, dy = -this.gy * s;
    const dl = Math.hypot(dx, dy);
    if (dl > 34) { dx *= 34 / dl; dy *= 34 / dl; }
    this.dot.setAttribute('cx', dx.toFixed(2));
    this.dot.setAttribute('cy', dy.toFixed(2));
    this.trail.setAttribute('cx', (dx * 0.7).toFixed(2));
    this.trail.setAttribute('cy', (dy * 0.7).toFixed(2));
  }
  /** Magnitude of the smoothed vector (g). */
  get magnitude() { return Math.hypot(this.gx, this.gy); }
  setValueText(str) { this.readout.set(str); }
}
