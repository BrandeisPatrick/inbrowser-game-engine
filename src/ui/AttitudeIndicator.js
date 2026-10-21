// AttitudeIndicator — artificial horizon (pitch ladder, bank scale + pointer, fixed aircraft symbol).
//   const adi = new AttitudeIndicator({ parent, size: 112 });  adi.set(pitchDeg, bankDeg) every frame
import { Widget } from './Widget.js';
import { injectStyle, theme } from './theme.js';
import { el, svgEl, polar, clamp } from './dom.js';

const CSS = `
.ui-adi{flex:none;border-radius:50%;overflow:hidden;border:1px solid rgba(255,255,255,.18);box-shadow:inset 0 0 18px rgba(0,0,0,.6)}
.ui-adi svg{width:100%;height:100%;display:block}
`;

export class AttitudeIndicator extends Widget {
  /** @param {object} [o]  size=112, sky, ground, accent, pxPerDeg=1.4, maxPitch=70, className */
  constructor(o = {}) {
    super(o);
    injectStyle('ui-adi', CSS);
    const size = o.size ?? 112, accent = o.accent || theme.accent;
    this.pxPerDeg = o.pxPerDeg ?? 1.4;
    this.maxPitch = o.maxPitch ?? 70;
    const root = this.el = el('div', `ui-w ui-adi${o.className ? ` ${o.className}` : ''}`);
    root.style.width = root.style.height = `${size}px`;
    const asvg = svgEl('svg', { viewBox: '-60 -60 120 120' }, root);
    const horizon = this.horizon = svgEl('g', {}, asvg);
    svgEl('rect', { x: -200, y: -400, width: 400, height: 400, fill: o.sky || '#2f6fa8' }, horizon);
    svgEl('rect', { x: -200, y: 0, width: 400, height: 400, fill: o.ground || '#5a3d22' }, horizon);
    svgEl('line', { x1: -200, y1: 0, x2: 200, y2: 0, stroke: '#fff', 'stroke-width': 1.2 }, horizon);
    for (let p = -60; p <= 60; p += 10) {
      if (p === 0) continue;
      const y = -p * this.pxPerDeg;
      const w = p % 20 === 0 ? 22 : 12;
      svgEl('line', { x1: -w / 2, y1: y, x2: w / 2, y2: y, stroke: 'rgba(255,255,255,.8)', 'stroke-width': 0.8 }, horizon);
      if (p % 20 === 0) {
        const t = svgEl('text', { x: w / 2 + 3, y: y + 2.5, fill: 'rgba(255,255,255,.75)', 'font-size': 6.5, 'font-family': 'inherit' }, horizon);
        t.textContent = String(Math.abs(p));
      }
    }
    // roll scale (fixed) + bank pointer (rotates)
    for (const a of [-60, -45, -30, -20, -10, 0, 10, 20, 30, 45, 60]) {
      const [x0, y0] = polar(0, 0, 52, a);
      const [x1, y1] = polar(0, 0, a % 30 === 0 ? 45 : 48, a);
      svgEl('line', { x1: x0, y1: y0, x2: x1, y2: y1, stroke: '#fff', 'stroke-width': 1 }, asvg);
    }
    this.bankPointer = svgEl('path', { d: 'M0 -44 L-4 -37 L4 -37 Z', fill: accent }, asvg);
    // fixed aircraft symbol
    svgEl('path', { d: 'M-28 0 L-10 0 L-5 5 L0 0 L5 5 L10 0 L28 0', fill: 'none', stroke: accent, 'stroke-width': 2.4, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }, asvg);
    svgEl('circle', { cx: 0, cy: 0, r: 1.6, fill: accent }, asvg);
    this.mountTo(o.parent);
  }

  /** pitch: + nose up (deg); bank: + right wing down (deg). */
  set(pitchDeg, bankDeg) {
    const pitchPx = clamp(pitchDeg, -this.maxPitch, this.maxPitch) * this.pxPerDeg;
    this.horizon.setAttribute('transform', `rotate(${(-bankDeg).toFixed(2)}) translate(0 ${pitchPx.toFixed(2)})`);
    this.bankPointer.setAttribute('transform', `rotate(${(-bankDeg).toFixed(2)})`);
  }
}
