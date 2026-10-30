// Character HUD, built only from the engine UI kit (teal accent): speed Readout + grounded / animation / camera
// Badges and a PUSH / RIDE badge (shown only then), fps (engine profiler) and the model in use (top left), a key strip
// (bottom left), a click-to-play prompt,
// a first-person crosshair, Flash messages and the HelpOverlay (H, Esc, or the pointer lock released with Esc).
import { UILayer, Panel, Readout, Badge, Flash, HelpOverlay, setTheme, injectStyle, UIDom } from '@vibe/engine/ui';

export const ACCENT = '#2fd4bd';

const CSS = `
.ch-stats{min-width:230px;padding:14px 18px 12px}
.ch-stats .row{display:flex;gap:8px;margin-top:12px;flex-wrap:wrap}
.ch-stats .ui-badge{letter-spacing:.2em;padding:4px 10px}
.ch-stats .meta{margin-top:10px;font-size:10px;letter-spacing:.2em;color:var(--ui-dim);text-transform:uppercase}
.ch-keys{display:flex;flex-wrap:wrap;gap:6px 16px;padding:10px 14px;max-width:min(760px,calc(100vw - 44px));font-size:11px;letter-spacing:.12em;color:var(--ui-dim)}
.ch-keys kbd{display:inline-block;margin-right:6px;padding:1px 6px;border:1px solid rgba(255,255,255,.3);border-radius:4px;font-family:inherit;color:#fff;letter-spacing:.04em}
.ch-prompt{position:absolute;left:50%;top:64%;transform:translate(-50%,-50%) scale(var(--ui-scale));padding:10px 22px;font-size:12px;font-weight:500;
  letter-spacing:.3em;color:#fff;white-space:nowrap;transition:opacity .3s}
.ch-prompt.hidden{opacity:0}
.ch-cross{position:absolute;left:50%;top:50%;width:6px;height:6px;margin:-3px 0 0 -3px;border-radius:50%;background:rgba(255,255,255,.8);
  box-shadow:0 0 0 1px rgba(0,0,0,.35);opacity:0;transition:opacity .2s}
.ch-cross.on{opacity:1}
@media (max-width:760px){.ch-keys{display:none}}
`;

const KEYS = [['WASD', 'Move'], ['Shift', 'Run'], ['Space', 'Jump / Mantle'], ['Mouse', 'Look'], ['V', 'Camera'], ['K', 'Ragdoll'], ['R', 'Respawn'], ['1–8', 'Time'], ['H', 'Help'], ['F3', 'Profiler']];
const HELP_ROWS = [
  ['W A S D / arrows · L-stick', 'Move (camera-relative)'], ['Shift · LB / L3', 'Run'], ['Space · A', 'Jump'],
  ['Space toward a 0.6–1.4 m ledge', 'Mantle (near the jump apex: no press)'], ['Walk into a crate', 'Push (up to 120 kg)'],
  ['Mouse (click to lock) · R-stick', 'Look'], ['Right-drag', 'Look without the lock'], ['Wheel', 'Zoom (third person, orbit)'],
  ['V · Y', 'Camera: third → first → orbit'], ['K', 'Ragdoll (again: get up)'], ['R', 'Respawn + reset the crates'], ['1 – 8', 'Time of day'],
  ['H · Esc', 'This help'], ['F3', 'Profiler'],
];
const CAM_LABEL = { third: 'THIRD', first: 'FIRST', orbit: 'ORBIT' };

/** The sample skin: call before building anything that reads theme.accent (loading bar, character accent). */
export function applyTheme() {
  setTheme({ accent: ACCENT, ok: '#7be08c', warn: '#ffb057' });
  injectStyle('character-hud', CSS);
}

export class HUD {
  constructor() {
    applyTheme();
    const L = this.layer = new UILayer({ parent: document.body });
    const { el } = UIDom;
    // --- stats (top left)
    const sp = new Panel({ parent: L, corner: 'tl', className: 'ch-stats' });
    this.speed = new Readout({ parent: sp, label: 'Speed', size: 'lg', unit: 'm/s', value: '0.0', sub: '0 km/h' });
    const row = el('div', 'row', sp.el);
    this.ground = new Badge({ parent: row, text: 'GROUNDED' });
    this.anim = new Badge({ parent: row, text: 'IDLE' });
    this.cam = new Badge({ parent: row, text: 'THIRD' });
    this.act = new Badge({ parent: row, text: '' }); // PUSH / RIDE
    this.act.setVisible(false);
    this.fps = el('div', 'meta ui-num', sp.el, '— fps');
    this.model = el('div', 'meta', sp.el, '');
    // --- keys (bottom left)
    const kp = new Panel({ parent: L, corner: 'bl', className: 'ch-keys' });
    kp.el.innerHTML = KEYS.map(([k, v]) => `<span><kbd>${k}</kbd>${v}</span>`).join('');
    // --- prompt, crosshair, messages, help
    this.prompt = el('div', 'ch-prompt ui-glass', L.el, 'CLICK TO PLAY');
    this.cross = el('div', 'ch-cross', L.el);
    this.flash = new Flash({ parent: L });
    this.help = new HelpOverlay({
      parent: L, title: 'CHARACTER <b>·</b> PLAYGROUND', subtitle: 'Kinematic controller · animation blend tree · camera rigs',
      rows: HELP_ROWS,
      footnote: 'Walls, 0.3 m stairs, a ramp onto a 2 m platform, 0.5 / 1 / 1.5 m crates and a low-ceiling corridor; '
        + 'pushable crates and a 400 kg one near the start, an elevator platform off the 2 m platform (falls over 9 m/s knock you down). '
        + '?model=&lt;url&gt; loads any rigged glTF, ?model=procedural forces the built-in rig, ?t=noon|golden|night sets the time.',
      hint: 'CLICK TO PLAY · <kbd>H</kbd> TO HIDE',
    });
    this.locked = false; this.camName = 'third';
    this.onHelp = null; // (visible) => void, called whenever the help card may have changed
    this._fpsT = 1;
  }

  setModel(text) { this.model.textContent = text; }
  setCamera(name) {
    this.camName = name;
    this.cam.setText('t', this.cam.el, CAM_LABEL[name] || name.toUpperCase());
    this._sync();
  }
  setLocked(on) { this.locked = !!on; this._sync(); }
  showHelp(on) { this.help.show(on); this._sync(); }
  toggleHelp() { this.help.toggle(); this._sync(); }
  // the prompt shows while look needs a click (not locked, no help card, not the orbit camera); dot in first person
  _sync() {
    this.prompt.classList.toggle('hidden', this.locked || this.help.visible || this.camName === 'orbit');
    this.cross.classList.toggle('on', this.locked && this.camName === 'first');
    if (this.onHelp) this.onHelp(this.help.visible);
  }

  /** Per frame. s: { speed (m/s, horizontal), grounded, anim (label), action ('PUSH' | 'RIDE' | ''), fps } (reused). */
  update(dt, s) {
    this._fpsT += dt;
    if (!this.speed.tickText(dt)) return;
    this.speed.set(s.speed.toFixed(1));
    this.speed.setSub(`${Math.round(s.speed * 3.6)} km/h`);
    const g = this.ground;
    g.setText('t', g.el, s.grounded ? 'GROUNDED' : 'AIRBORNE');
    g.toggleClass('on', g.el, 'on', !s.grounded);
    this.anim.setText('t', this.anim.el, s.anim);
    const a = this.act, act = s.action || '';
    a.setText('t', a.el, act);
    a.toggleClass('on', a.el, 'on', act !== '');
    a.setStyle('d', a.el, 'display', act ? '' : 'none');
    if (this._fpsT >= 0.5) { this._fpsT = 0; this.fps.textContent = `${Math.round(s.fps)} fps`; }
  }
}
