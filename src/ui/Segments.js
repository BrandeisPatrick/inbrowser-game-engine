// Segments — a row of letters with one lit (gear selector P R N D, drive modes, ...).
//   const g = new Segments({ parent, items: ['P', 'R', 'N', 'D'] });  g.select('D')
import { Widget } from './Widget.js';
import { injectStyle } from './theme.js';
import { el } from './dom.js';

const CSS = `
.ui-segments{display:flex;gap:10px;justify-content:center;margin-bottom:6px;font-size:12px;font-weight:500;letter-spacing:.1em}
.ui-segments span{color:rgba(255,255,255,.28);transition:color .2s}
.ui-segments span.on{color:#fff}
`;

export class Segments extends Widget {
  constructor(o = {}) {
    super(o);
    injectStyle('ui-segments', CSS);
    this.el = el('div', `ui-w ui-segments${o.className ? ` ${o.className}` : ''}`);
    this.items = {};
    for (const k of o.items || []) this.items[k] = el('span', '', this.el, k);
    this.selected = null;
    this.mountTo(o.parent);
  }
  select(key) {
    if (key === this.selected) return;
    if (this.selected && this.items[this.selected]) this.items[this.selected].classList.remove('on');
    if (this.items[key]) this.items[key].classList.add('on');
    this.selected = key;
  }
}
