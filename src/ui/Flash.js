// Flash — centred transient message with an accent underline (restarts on every call).
//   const f = new Flash({ parent: layer });  f.show('FLIGHT MODE')
import { Widget } from './Widget.js';
import { injectStyle } from './theme.js';
import { el } from './dom.js';

const CSS = `
.ui-flash{position:absolute;z-index:3;left:50%;top:24%;transform:translate(-50%,-50%);font-size:28px;font-weight:300;
  letter-spacing:.35em;text-transform:uppercase;white-space:nowrap;opacity:0;text-shadow:0 2px 20px rgba(0,0,0,.6);pointer-events:none}
.ui-flash::after{content:"";display:block;height:2px;margin:10px auto 0;width:60%;background:var(--ui-accent);border-radius:1px}
.ui-flash.show{animation:uiFlash 2.1s cubic-bezier(.2,.8,.2,1) forwards}
@keyframes uiFlash{0%{opacity:0;transform:translate(-50%,-40%) scale(.96);letter-spacing:.6em}
  12%{opacity:1;transform:translate(-50%,-50%) scale(1);letter-spacing:.35em}
  75%{opacity:1}100%{opacity:0;transform:translate(-50%,-56%)}}
`;

export class Flash extends Widget {
  constructor(o = {}) {
    super(o);
    injectStyle('ui-flash', CSS);
    this.el = el('div', `ui-w ui-flash${o.className ? ` ${o.className}` : ''}`);
    this.mountTo(o.parent);
  }
  show(msg) {
    const f = this.el;
    f.textContent = String(msg);
    f.classList.remove('show');
    void f.offsetWidth; // restart animation
    f.classList.add('show');
  }
}
