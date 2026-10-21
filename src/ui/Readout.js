// Readout — a number with optional label above, small unit suffix, unit label below and a secondary line.
//   const r = new Readout({ parent, label: 'Altitude AGL', unit: 'm', size: 'lg' });  r.set('12.4')
//   const s = new Readout({ parent, size: 'xl', unitLabel: 'km/h', sub: '0 mph' });   s.set('88'); s.setSub('55 mph')
// size: 'xl' (78px) | 'lg' (36px) | 'md' (28px) | 'sm' (15px) | null (inherit).
// contents: true lays the parts out directly in the parent (display: contents) — handy inside flex columns.
import { Widget } from './Widget.js';
import { injectStyle } from './theme.js';
import { el } from './dom.js';

const CSS = `
.ui-readout.contents{display:contents}
.ui-rv.xl{font-size:78px;line-height:1;font-weight:200;letter-spacing:-.03em}
.ui-rv.lg{font-size:36px;font-weight:200;line-height:1;letter-spacing:-.02em}
.ui-rv.md{font-size:28px;font-weight:200;line-height:1}
.ui-rv.sm{font-size:15px;font-weight:300}
.ui-rv small{color:var(--ui-mute)}
.ui-rv.lg small{font-size:14px;margin-left:3px}
.ui-rv.md small{font-size:12px;margin-left:2px}
.ui-readout-unit{margin-top:2px}
.ui-readout-sub{margin-top:6px;font-size:13px;color:var(--ui-mute)}
`;

export class Readout extends Widget {
  /**
   * @param {object} [o]  label, value='0', unit (small suffix), unitLabel (label line under the value),
   *   sub (secondary line), size, className (on the wrapper), valueClass, contents
   */
  constructor(o = {}) {
    super(o);
    injectStyle('ui-readout', CSS);
    this.unit = o.unit || '';
    const root = this.el = el('div', `ui-w ui-readout${o.contents ? ' contents' : ''}${o.className ? ` ${o.className}` : ''}`);
    this.labelEl = o.label != null ? el('div', 'ui-label', root, o.label) : null;
    this.valueEl = el('div', `ui-rv ui-num${o.size ? ` ${o.size}` : ''}${o.valueClass ? ` ${o.valueClass}` : ''}`, root);
    this.unitLabelEl = o.unitLabel != null ? el('div', 'ui-label ui-readout-unit', root, o.unitLabel) : null;
    this.subEl = o.sub != null ? el('div', 'ui-readout-sub ui-num', root, o.sub) : null;
    this.set(o.value ?? '0');
    this.mountTo(o.parent);
  }
  /** Cached: only touches the DOM when the string changes. */
  set(value, unit = this.unit) {
    if (unit) this.setHTML('v', this.valueEl, `${value}<small>${unit}</small>`);
    else this.setText('v', this.valueEl, String(value));
  }
  setSub(str) { if (this.subEl) this.setText('sub', this.subEl, str); }
  setLabel(str) { if (this.labelEl) this.setText('label', this.labelEl, str); }
}
