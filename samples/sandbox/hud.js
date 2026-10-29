// Sandbox HUD, built only from the engine UI kit (amber accent): the scene name + its one-line description and the
// dynamics counters (step ms with its broad / narrow / solve split, bodies, awake, contacts, pairs, islands, joints,
// fps, worker latency, a warning when the contact buffer overflows) top left, the scene list (links that reload the page with ?scene=) top right, a key strip
// bottom left, Flash messages and the HelpOverlay (H). Numbers refresh 4x a second; nothing allocates between
// refreshes.
import { UILayer, Panel, Readout, Flash, HelpOverlay, setTheme, injectStyle, UIDom } from '@vibe/engine/ui';

export const ACCENT = '#f2a33a';

const CSS = `
.sb-stats{min-width:250px;max-width:300px;padding:14px 18px 12px}
.sb-stats .title{font-size:15px;font-weight:600;letter-spacing:.32em;color:#fff;text-transform:uppercase}
.sb-stats .info{margin-top:6px;font-size:12px;line-height:1.45;color:var(--ui-dim)}
.sb-stats .step{margin-top:14px}
.sb-stats .grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px 14px;margin-top:14px}
.sb-stats .meta{margin-top:12px;font-size:10px;letter-spacing:.2em;color:var(--ui-dim);text-transform:uppercase}
.sb-stats .extra{margin-top:6px;font-size:12px;letter-spacing:.12em;color:#fff}
.sb-stats .warn{margin-top:6px;font-size:11px;letter-spacing:.14em;color:var(--ui-warn);text-transform:uppercase}
.sb-scenes{display:flex;flex-direction:column;gap:2px;padding:10px 12px;pointer-events:auto}
.sb-scenes a{padding:4px 10px;border-radius:6px;font-size:11px;letter-spacing:.2em;text-transform:uppercase;color:var(--ui-dim);text-decoration:none}
.sb-scenes a:hover{color:#fff;background:rgba(255,255,255,.06)}
.sb-scenes a.on{color:#fff;background:rgba(var(--ui-accent-rgb),.2)}
.sb-keys{display:flex;flex-wrap:wrap;gap:6px 16px;padding:10px 14px;max-width:min(820px,calc(100vw - 44px));font-size:11px;letter-spacing:.12em;color:var(--ui-dim)}
.sb-keys kbd{display:inline-block;margin-right:6px;padding:1px 6px;border:1px solid rgba(255,255,255,.3);border-radius:4px;font-family:inherit;color:#fff;letter-spacing:.04em}
@media (max-width:760px){.sb-keys{display:none}.sb-scenes{display:none}}
`;

const KEYS = [['Drag', 'Orbit'], ['Wheel', 'Zoom'], ['V', 'Free cam'], ['Space', 'Throw ball'], ['R', 'Reset'], ['F4', 'Debug'], ['1–8', 'Time'], ['H', 'Help'], ['F3', 'Profiler']];
const CAR_KEYS = [['W S', 'Drive'], ['A D', 'Steer'], ['X', 'Handbrake']];
const HELP_ROWS = [
  ['Left-drag', 'Orbit the camera'], ['Right-drag · Shift-drag', 'Pan'], ['Wheel', 'Zoom (free cam: speed)'],
  ['V', 'Camera: orbit ↔ free fly'], ['W A S D · Q E', 'Free fly (Shift fast, Alt slow)'],
  ['Space', 'Throw a ball from the camera'], ['R', 'Reset the scene'], ['F4', 'Physics debug lines'],
  ['W S · A D · X · Shift', 'Car: drive, steer, handbrake, boost'], ['1 – 8', 'Time of day'], ['H', 'This help'], ['F3', 'Profiler'],
];

/** The sample skin: main.js calls it once, before building anything that reads theme.accent (loading bar). */
export function applyTheme() {
  setTheme({ accent: ACCENT, ok: '#7be08c', warn: '#ff7a59' });
  injectStyle('sandbox-hud', CSS);
}

const fmt = (v, d) => (Number.isFinite(v) ? v.toFixed(d) : '—');
const int = (v) => (Number.isFinite(v) ? String(Math.round(v)) : '—');
const ema = (cur, v, k) => (Number.isFinite(v) ? (Number.isFinite(cur) ? cur + (v - cur) * k : v) : cur);

export class HUD {
  /**
   * @param {object} o
   * @param {string[]} o.scenes  scene ids (the links)
   * @param {string} o.current   the scene on this page
   * @param {Function} o.href    (sceneId) => url of the page with that scene
   * @param {boolean} [o.worker] the dynamics run in a worker (shows the latency)
   */
  constructor(o) {
    const L = this.layer = new UILayer({ parent: document.body });
    const { el } = UIDom;
    // --- scene + stats (top left)
    const sp = new Panel({ parent: L, corner: 'tl', className: 'sb-stats' });
    this.title = el('div', 'title', sp.el, '');
    this.info = el('div', 'info', sp.el, '');
    const stepBox = el('div', 'step', sp.el);
    this.step = new Readout({ parent: stepBox, label: 'Step', size: 'lg', unit: 'ms', value: '—', sub: 'broad — · narrow — · solve —' });
    const grid = el('div', 'grid', sp.el);
    const small = (label) => new Readout({ parent: grid, label, size: 'sm', value: '—' });
    this.bodies = small('Bodies'); this.awake = small('Awake'); this.contacts = small('Contacts');
    this.pairs = small('Pairs'); this.islands = small('Islands'); this.joints = small('Joints');
    this.extra = el('div', 'extra ui-num', sp.el, '');
    this.warn = el('div', 'warn ui-num', sp.el, '');
    this.meta = el('div', 'meta ui-num', sp.el, '— fps');
    // --- scene links (top right)
    const sc = new Panel({ parent: L, corner: 'tr', className: 'sb-scenes' });
    for (const id of o.scenes) {
      const a = el('a', id === o.current ? 'on' : '', sc.el, id);
      a.href = o.href(id);
    }
    // --- keys (bottom left)
    this.keys = new Panel({ parent: L, corner: 'bl', className: 'sb-keys' });
    this.setCarKeys(false);
    this.flash = new Flash({ parent: L });
    this.help = new HelpOverlay({
      parent: L, title: 'PHYSICS <b>·</b> SANDBOX', subtitle: 'XPBD rigid bodies · joints · ragdolls · kinematic car',
      rows: HELP_ROWS,
      footnote: '?scene=stack|joints|chain|ragdoll|car|bench1000 · ?bodies=N (bench1000) · ?worker=1 runs the dynamics '
        + 'in a worker · ?debug=1 shows the debug lines · ?nofx=1 · ?t=noon|golden|night · ?bench=1&amp;size=WxH',
      hint: 'PRESS <kbd>H</kbd> TO HIDE',
    });
    this.worker = !!o.worker;
    this.onHelp = null; // (visible) => void after every help toggle (main.js silences the input map)
    this._t = 1; this._ms = NaN; this._broad = NaN; this._narrow = NaN; this._solve = NaN;
  }

  setScene(name, info) { this.title.textContent = name; this.info.textContent = info || ''; }
  setCarKeys(on) {
    const k = on ? CAR_KEYS.concat(KEYS) : KEYS;
    this.keys.el.innerHTML = k.map(([a, b]) => `<span><kbd>${a}</kbd>${b}</span>`).join('');
  }
  showHelp(on) { this.help.show(on); if (this.onHelp) this.onHelp(this.help.visible); }
  toggleHelp() { this.help.toggle(); if (this.onHelp) this.onHelp(this.help.visible); }

  /**
   * Per frame. s: { stats (dyn.stats or null), fps, latency (ms, worker), substeps, hz, extra () => string | null,
   * overflow () => contacts dropped by the last step }
   * (a reused object). Step times are smoothed (EMA) so the readout is legible. force: refresh the text now (after a
   * scene build) instead of at the next 4 Hz tick.
   */
  update(dt, s, force = false) {
    const st = s.stats;
    if (st) {
      const k = 1 - Math.exp(-dt * 4);
      this._ms = ema(this._ms, st.ms, k); this._broad = ema(this._broad, st.msBroad, k);
      this._narrow = ema(this._narrow, st.msNarrow, k); this._solve = ema(this._solve, st.msSolve, k);
    }
    this._t += dt;
    if (!force && this._t < 0.25) return;
    this._t = 0;
    this.step.set(fmt(this._ms, 2));
    this.step.setSub(`broad ${fmt(this._broad, 2)} · narrow ${fmt(this._narrow, 2)} · solve ${fmt(this._solve, 2)}`);
    this.bodies.set(int(st?.bodies)); this.awake.set(int(st?.awake)); this.contacts.set(int(st?.contacts));
    this.pairs.set(int(st?.pairs)); this.islands.set(int(st?.islands)); this.joints.set(int(st?.joints));
    const extra = typeof s.extra === 'function' ? s.extra() : '';
    if (this.extra.textContent !== (extra || '')) this.extra.textContent = extra || '';
    const over = typeof s.overflow === 'function' ? s.overflow() : 0;
    const w = over > 0 ? `contact buffer full · ${over} dropped` : '';
    if (this.warn.textContent !== w) this.warn.textContent = w;
    let m = `${Math.round(s.fps || 0)} fps · ${s.hz} hz × ${s.substeps}`;
    if (this.worker) m += ` · worker ${fmt(s.latency, 1)} ms`;
    if (this.meta.textContent !== m) this.meta.textContent = m;
  }
}
