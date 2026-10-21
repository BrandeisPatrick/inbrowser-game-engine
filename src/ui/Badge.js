// Badge — pill label with an "on" state (accent glow) and a flip + shine animation on change.
//   const b = new Badge({ parent, text: 'GROUND' });  b.set('FLIGHT', true, { animate: true })
import { Widget } from './Widget.js';
import { injectStyle } from './theme.js';
import { el } from './dom.js';

const CSS = `
.ui-badge{position:relative;padding:5px 12px;border-radius:999px;font-size:11px;font-weight:600;letter-spacing:.28em;
  border:1px solid rgba(255,255,255,.22);color:#fff;overflow:hidden}
.ui-badge.on{border-color:var(--ui-accent);background:rgba(var(--ui-accent-rgb),.18);box-shadow:0 0 18px rgba(var(--ui-accent-rgb),.35)}
.ui-badge.anim{animation:uiBadge .7s cubic-bezier(.2,.8,.2,1)}
.ui-badge.anim::after{content:"";position:absolute;inset:0;background:linear-gradient(100deg,transparent 30%,rgba(255,255,255,.55) 50%,transparent 70%);
  transform:translateX(-120%);animation:uiSweep .7s ease-out}
@keyframes uiBadge{0%{transform:rotateX(90deg) scale(.9);opacity:.2}60%{transform:rotateX(-12deg) scale(1.06);opacity:1}100%{transform:none}}
@keyframes uiSweep{to{transform:translateX(120%)}}
`;

export class Badge extends Widget {
  constructor(o = {}) {
    super(o);
    injectStyle('ui-badge', CSS);
    this.el = el('div', `ui-w ui-badge${o.className ? ` ${o.className}` : ''}`, null, o.text ?? '');
    this.on = false;
    this.mountTo(o.parent);
  }
  set(text, on = this.on, { animate = false } = {}) {
    this.el.textContent = text;
    this.on = !!on;
    this.el.classList.toggle('on', this.on);
    if (animate) this.pulse();
  }
  /** Replay the flip + shine animation. */
  pulse() {
    this.el.classList.remove('anim');
    void this.el.offsetWidth; // restart
    this.el.classList.add('anim');
  }
}
