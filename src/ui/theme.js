// UI theme: CSS variables + base classes, injected once as a <style> (the kit's "theme.css").
//   setTheme({ accent: '#00c2ff' })   // updates the CSS variables and the JS values canvas/SVG widgets read
// Variables: --ui-accent --ui-accent-rgb --ui-regen --ui-ok --ui-warn --ui-text --ui-dim --ui-mute
//            --ui-font --ui-glass --ui-glass-border --ui-radius --ui-gutter --ui-scale
// Base classes: ui-root (overlay layer), ui-glass, ui-corner + ui-br/bl/tl/tr, ui-label, ui-num, ui-row.

/** JS-side theme values (canvas / SVG attributes can't read CSS variables). */
export const theme = {
  accent: '#e82127',
  accentRgb: '232,33,39',
  regen: '#3e9bff',
  ok: '#4cd27a',
  warn: '#ffb057',
  text: '#f2f4f7',
  font: '"Inter","SF Pro Display",-apple-system,system-ui,"Segoe UI",Roboto,sans-serif',
};

const VAR_OF = { accent: '--ui-accent', accentRgb: '--ui-accent-rgb', regen: '--ui-regen', ok: '--ui-ok', warn: '--ui-warn', text: '--ui-text', font: '--ui-font' };

export const THEME_CSS = `
:root{--ui-accent:${theme.accent};--ui-accent-rgb:${theme.accentRgb};--ui-regen:${theme.regen};--ui-ok:${theme.ok};--ui-warn:${theme.warn};
  --ui-text:${theme.text};--ui-dim:rgba(235,240,247,.55);--ui-mute:rgba(235,240,247,.6);--ui-font:${theme.font};
  --ui-glass:linear-gradient(160deg,rgba(22,26,33,.62),rgba(8,10,14,.55));--ui-glass-border:rgba(255,255,255,.08);
  --ui-radius:18px;--ui-gutter:22px;--ui-scale:1}
.ui-root{position:fixed;inset:0;pointer-events:none;z-index:20;color:var(--ui-text);font-family:var(--ui-font);
  font-weight:300;-webkit-font-smoothing:antialiased;user-select:none}
.ui-root *,.ui-w,.ui-w *{box-sizing:border-box}
.ui-glass{background:var(--ui-glass);border:1px solid var(--ui-glass-border);border-radius:var(--ui-radius);
  box-shadow:0 10px 40px rgba(0,0,0,.35),inset 0 1px 0 rgba(255,255,255,.05);
  -webkit-backdrop-filter:blur(14px) saturate(130%);backdrop-filter:blur(14px) saturate(130%)}
.ui-corner{position:absolute;transform:scale(var(--ui-scale))}
.ui-br{right:var(--ui-gutter);bottom:var(--ui-gutter);transform-origin:100% 100%}
.ui-bl{left:var(--ui-gutter);bottom:var(--ui-gutter);transform-origin:0 100%}
.ui-tl{left:var(--ui-gutter);top:var(--ui-gutter);transform-origin:0 0}
.ui-tr{right:var(--ui-gutter);top:var(--ui-gutter);transform-origin:100% 0}
.ui-label{font-size:10px;letter-spacing:.22em;text-transform:uppercase;color:var(--ui-dim);font-weight:500}
.ui-num{font-variant-numeric:tabular-nums;font-feature-settings:"tnum"}
.ui-row{display:flex;justify-content:space-between;align-items:baseline;width:100%;margin-bottom:6px}
`;

const injected = new Set();
/** Inject a stylesheet once per id (no-op outside the browser). */
export function injectStyle(id, css) {
  if (injected.has(id) || typeof document === 'undefined') return;
  injected.add(id);
  if (document.getElementById(id)) return;
  const st = document.createElement('style');
  st.id = id;
  st.textContent = css;
  document.head.appendChild(st);
}

export function injectTheme() { injectStyle('ui-theme', THEME_CSS); }

/** Override theme values (JS + CSS variables on `el`, default :root). Unknown keys are set as --ui-<key>. */
export function setTheme(vars = {}, el = typeof document !== 'undefined' ? document.documentElement : null) {
  for (const k of Object.keys(vars)) {
    if (k in theme) theme[k] = vars[k];
    if (el) el.style.setProperty(VAR_OF[k] || `--ui-${k}`, vars[k]);
  }
  if (vars.accent && !vars.accentRgb && /^#[0-9a-f]{6}$/i.test(vars.accent)) {
    const n = parseInt(vars.accent.slice(1), 16);
    theme.accentRgb = `${n >> 16},${(n >> 8) & 255},${n & 255}`;
    if (el) el.style.setProperty('--ui-accent-rgb', theme.accentRgb);
  }
}
