// HelpOverlay — centred glass card with a title, subtitle, key/action table, footnote and hint line.
//   const h = new HelpOverlay({ parent: layer, title: 'GAME <b>·</b> CONTROLS', rows: [['W', 'Forward']],
//                               hint: 'PRESS <kbd>H</kbd> TO HIDE' });
//   h.show(true); h.toggle(); h.visible
import { Widget } from './Widget.js';
import { injectStyle } from './theme.js';
import { el } from './dom.js';

const CSS = `
.ui-help{position:absolute;left:50%;top:50%;width:min(460px,calc(100vw - 32px));padding:22px 26px 18px;
  transform:translate(-50%,-50%) scale(var(--ui-scale));transition:opacity .35s;opacity:1}
.ui-help.hidden{opacity:0}
.ui-help h3{margin:0 0 4px;font-size:13px;font-weight:600;letter-spacing:.32em;color:#fff}
.ui-help h3 b{color:var(--ui-accent);font-weight:600}
.ui-help table{width:100%;border-collapse:collapse;margin-top:12px;font-size:13px}
.ui-help td{padding:5px 0;border-bottom:1px solid rgba(255,255,255,.06)}
.ui-help td:first-child{color:#fff;font-weight:500;white-space:nowrap;padding-right:16px}
.ui-help td:last-child{color:rgba(235,240,247,.7);text-align:right}
.ui-help .pad{margin-top:10px;font-size:11px;color:rgba(235,240,247,.5);line-height:1.5}
.ui-help .hint{margin-top:12px;text-align:center;font-size:11px;letter-spacing:.2em;color:rgba(235,240,247,.75)}
.ui-help .hint kbd{display:inline-block;padding:1px 7px;border:1px solid rgba(255,255,255,.35);border-radius:4px;font-family:inherit;color:#fff}
@media (max-width:760px){.ui-help td:last-child{text-align:left}}
`;

export class HelpOverlay extends Widget {
  /** @param {object} [o]  title (HTML), subtitle, rows [[key, action]], footnote, hint (HTML), visible=false */
  constructor(o = {}) {
    super(o);
    injectStyle('ui-help', CSS);
    this.el = el('div', `ui-w ui-help ui-glass hidden${o.className ? ` ${o.className}` : ''}`);
    this.setContent(o);
    this.visible = false;
    this.mountTo(o.parent);
    if (o.visible) this.show(true);
  }
  setContent({ title = '', subtitle = '', rows = [], footnote = '', hint = '' } = {}) {
    let tr = '';
    for (const [k, v] of rows) tr += `<tr><td>${k}</td><td>${v}</td></tr>`;
    this.el.innerHTML = `<h3>${title}</h3>
      ${subtitle ? `<div class="ui-label">${subtitle}</div>` : ''}
      <table>${tr}</table>
      ${footnote ? `<div class="pad">${footnote}</div>` : ''}
      ${hint ? `<div class="hint">${hint}</div>` : ''}`;
  }
  show(on) { this.visible = !!on; this.el.classList.toggle('hidden', !on); return this.visible; }
  toggle() { return this.show(!this.visible); }
}
