// LoadingScreen — full-screen loading overlay (title, progress bar, label) with a fade-out.
// Matches the Roadster index.html overlay so games can share it.
//   const ls = new LoadingScreen({ title: 'ROADSTER', label: 'Loading city…' });
//   ls.setProgress(0.4, 'Building car…');  await ls.hide();   // fades out (500 ms) and removes itself
//   LoadingScreen.adopt(document.getElementById('loading'))   // wrap an existing inline overlay
import { injectStyle, theme } from './theme.js';

const CSS = `
.ui-loading{position:fixed;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;color:#ddd;
  background:#06080c;z-index:100;font-family:-apple-system,system-ui,sans-serif}
.ui-loading h2{margin:0;font-weight:500;letter-spacing:.2em}
.ui-loading .bar{width:320px;max-width:calc(100vw - 48px);height:4px;background:#222;margin-top:16px;border-radius:2px;overflow:hidden}
.ui-loading .bar>div{height:100%;width:0%;background:var(--ui-accent,${theme.accent});transition:width .2s}
.ui-loading small{opacity:.6;margin-top:10px}
`;

export class LoadingScreen {
  /** @param {object} [o]  title, label, parent=document.body, accent, background */
  constructor(o = {}) {
    if (o.adopt) {
      this.el = o.adopt;
      this.barEl = o.adopt.querySelector('.bar > div');
      this.labelEl = o.adopt.querySelector('small');
      this.titleEl = o.adopt.querySelector('h2');
    } else {
      injectStyle('ui-loading', CSS);
      const root = this.el = document.createElement('div');
      root.className = 'ui-loading';
      this.titleEl = document.createElement('h2');
      this.titleEl.textContent = o.title ?? 'LOADING';
      const bar = document.createElement('div'); bar.className = 'bar';
      this.barEl = document.createElement('div');
      bar.appendChild(this.barEl);
      this.labelEl = document.createElement('small');
      this.labelEl.textContent = o.label ?? 'Loading…';
      root.append(this.titleEl, bar, this.labelEl);
      if (o.accent) root.style.setProperty('--ui-accent', o.accent);
      if (o.background) root.style.background = o.background;
      (o.parent || document.body).appendChild(root);
    }
    this.progress = 0;
  }

  static adopt(element) { return new LoadingScreen({ adopt: element }); }

  setProgress(p, label) {
    this.progress = Math.min(1, Math.max(0, p || 0));
    if (this.barEl) this.barEl.style.width = `${Math.round(this.progress * 100)}%`;
    if (label && this.labelEl) this.labelEl.textContent = label;
  }
  setLabel(text) { if (this.labelEl) this.labelEl.textContent = text; }
  /** Show an error in the label (bar stops). */
  error(msg) { this.setLabel(msg); if (this.barEl) this.barEl.style.background = '#a33'; }

  /** Fade out over ms and remove. Resolves when gone. */
  hide(ms = 500) {
    const e = this.el;
    if (!e || !e.isConnected) return Promise.resolve();
    e.style.transition = `opacity ${ms / 1000}s`;
    e.style.opacity = '0';
    e.style.pointerEvents = 'none';
    return new Promise((res) => setTimeout(() => { e.remove(); res(); }, ms + 50));
  }
  dispose() { if (this.el) this.el.remove(); }
}
