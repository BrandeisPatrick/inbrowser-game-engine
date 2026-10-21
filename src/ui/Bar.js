// Bar — labelled horizontal fill bar (label + value text row, transform-based fill), or a vertical
// (optionally bipolar) track like a vertical-speed indicator.
//   const b = new Bar({ parent, label: 'Boost', text: '100%', fill: 'linear-gradient(90deg,#ff6a3d,var(--ui-accent))' });
//   b.setValue(0..1) every frame;  b.setText('80%') on text ticks;  b.setWarn(true) pulses in the accent colour
//   const v = new Bar({ parent, vertical: true, bipolar: true, length: 42, thickness: 6 });  v.setValue(-1..1)
import { Widget } from './Widget.js';
import { injectStyle } from './theme.js';
import { el, clamp } from './dom.js';

const CSS = `
.ui-bar{position:relative;height:4px;width:100%;border-radius:2px;background:rgba(255,255,255,.1);overflow:hidden}
.ui-bar>i{position:absolute;inset:0;transform-origin:0 50%;transform:scaleX(0);border-radius:2px;background:var(--ui-bar-fill,#fff)}
.ui-barw.ui-warn .ui-bar>i{background:var(--ui-accent);animation:uiPulse .6s ease-in-out infinite alternate}
.ui-barw.ui-warn .ui-label{color:var(--ui-accent)}
@keyframes uiPulse{from{opacity:1;box-shadow:0 0 0 rgba(var(--ui-accent-rgb),0)}to{opacity:.35;box-shadow:0 0 10px rgba(var(--ui-accent-rgb),.8)}}
.ui-vbar{position:relative;width:6px;height:42px;border-radius:3px;background:rgba(255,255,255,.1);overflow:hidden}
.ui-vbar.bipolar::after{content:"";position:absolute;left:-2px;right:-2px;top:50%;height:1px;background:rgba(255,255,255,.5)}
.ui-vbar>div{position:absolute;left:0;right:0;bottom:0;height:100%;transform-origin:50% 100%;transform:scaleY(0);background:var(--ui-bar-fill,#9fe3a8)}
.ui-vbar.bipolar>div{top:0;bottom:auto;height:50%}
.ui-vbar.bipolar>div.dn{top:50%;transform-origin:50% 0;background:var(--ui-bar-fill-neg,#ffb057)}
`;

export class Bar extends Widget {
  /**
   * @param {object} [o]
   *   label, text (value text, right of the label), fill (CSS background of the fill), className,
   *   thickness (px), vertical, bipolar, length (px, vertical), fillNeg (bipolar negative colour)
   */
  constructor(o = {}) {
    super(o);
    injectStyle('ui-bar', CSS);
    this.vertical = !!o.vertical;
    this.bipolar = !!o.bipolar;
    if (this.vertical) {
      const track = this.el = el('div', `ui-w ui-vbar${this.bipolar ? ' bipolar' : ''}${o.className ? ` ${o.className}` : ''}`);
      if (o.length) track.style.height = `${o.length}px`;
      if (o.thickness) { track.style.width = `${o.thickness}px`; track.style.borderRadius = `${o.thickness / 2}px`; }
      if (o.fill) track.style.setProperty('--ui-bar-fill', o.fill);
      if (o.fillNeg) track.style.setProperty('--ui-bar-fill-neg', o.fillNeg);
      this.fill = el('div', '', track);
      this.fillNeg = this.bipolar ? el('div', 'dn', track) : null;
      this.track = track;
    } else {
      const root = this.el = el('div', `ui-w ui-barw${o.className ? ` ${o.className}` : ''}`);
      if (o.fill) root.style.setProperty('--ui-bar-fill', o.fill);
      if (o.label != null || o.text != null) {
        const row = el('div', 'ui-row', root);
        this.labelEl = el('span', 'ui-label', row, o.label ?? '');
        this.textEl = el('span', 'ui-label ui-num', row, o.text ?? '');
      }
      const track = this.track = el('div', 'ui-bar', root);
      if (o.thickness) { track.style.height = `${o.thickness}px`; track.style.borderRadius = `${o.thickness / 2}px`; }
      this.fill = el('i', '', track);
    }
    this.mountTo(o.parent);
  }

  /** 0..1 (bipolar: -1..1). Cheap; call every frame. */
  setValue(v) {
    if (this.vertical) {
      if (this.bipolar) {
        const a = clamp(Math.abs(v), 0, 1);
        this.fill.style.transform = `scaleY(${v > 0 ? a.toFixed(3) : 0})`;
        this.fillNeg.style.transform = `scaleY(${v < 0 ? a.toFixed(3) : 0})`;
      } else this.fill.style.transform = `scaleY(${clamp(v, 0, 1).toFixed(3)})`;
    } else this.fill.style.transform = `scaleX(${clamp(v, 0, 1).toFixed(3)})`;
  }
  setText(keyOrStr, node, str) {
    if (node !== undefined) return super.setText(keyOrStr, node, str);   // Widget.setText(key, node, str)
    if (this.textEl) super.setText('text', this.textEl, keyOrStr);
  }
  setLabel(str) { if (this.labelEl) super.setText('label', this.labelEl, str); }
  /** Solid fill colour (overrides the CSS fill). */
  setColor(c) { this.setStyle('color', this.fill, 'background', c); }
  setWarn(on) { this.toggleClass('warn', this.el, 'ui-warn', on); }
}
