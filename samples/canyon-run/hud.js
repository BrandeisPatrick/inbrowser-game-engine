// HUD built only from the engine UI kit, re-skinned (amber accent, warm glass, condensed type) so it reads as
// a different game: speed Gauge + gear Segments (bottom right), run timer Readout + checkpoint / lap Badges
// (top left), track Minimap with checkpoint markers (top right), split Flash, help overlay, menu + finish cards.
import {
  UILayer, Panel, Gauge, Readout, Segments, Badge, Minimap, Flash, HelpOverlay, Bar, setTheme, injectStyle, UIDom,
} from '@vibe/engine';
import { formatTime, formatDelta } from './race.js';

export const ACCENT = '#ff8a1e';

const CSS = `
:root{--ui-font:"Avenir Next Condensed","Avenir Next","DIN Alternate","Roboto Condensed",system-ui,sans-serif}
.cr-speed{width:250px;padding:10px 14px 12px}
.cr-speed .ui-rv.xl{font-weight:500;letter-spacing:-.02em}
.cr-timer{min-width:230px;padding:14px 18px 14px}
.cr-timer .ui-rv.lg{font-size:44px;font-weight:500}
.cr-timer .row{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap}
.cr-split{margin-top:8px;font-size:15px;font-weight:600;letter-spacing:.06em;min-height:18px}
.cr-split.good{color:var(--ui-ok)} .cr-split.bad{color:var(--ui-warn)}
.cr-map{padding:10px}
.cr-off{border-color:var(--ui-warn)!important;color:var(--ui-warn)}
.cr-card{position:absolute;left:50%;top:68%;transform:translate(-50%,-50%) scale(var(--ui-scale));width:min(520px,calc(100vw - 32px));
  padding:26px 30px 22px;text-align:center;pointer-events:auto;transition:opacity .35s}
.cr-card.hidden{opacity:0;pointer-events:none}
.cr-card.cr-finish{top:50%}
.cr-card h1{margin:0;font-size:46px;font-weight:700;letter-spacing:.22em;color:#fff}
.cr-card h1 b{color:var(--ui-accent);font-weight:700}
.cr-card .tag{margin-top:4px}
.cr-card .big{font-size:58px;font-weight:500;margin:10px 0 2px;font-variant-numeric:tabular-nums}
.cr-card .facts{display:flex;justify-content:center;gap:26px;margin:16px 0 6px}
.cr-card .facts div{display:flex;flex-direction:column;gap:2px}
.cr-card .facts b{font-size:20px;font-weight:500;color:#fff}
.cr-card .go{margin-top:16px;font-size:13px;letter-spacing:.28em;color:#fff}
.cr-card .go kbd,.cr-card .hint kbd{display:inline-block;padding:1px 8px;border:1px solid rgba(255,255,255,.4);border-radius:4px;font-family:inherit}
.cr-card .hint{margin-top:10px;font-size:11px;letter-spacing:.18em;color:var(--ui-dim)}
.cr-card table{width:100%;border-collapse:collapse;margin-top:12px;font-size:13px;font-variant-numeric:tabular-nums}
.cr-card td{padding:3px 0;border-bottom:1px solid rgba(255,255,255,.06)}
.cr-card td:first-child{text-align:left;color:var(--ui-dim)} .cr-card td:last-child{text-align:right}
.cr-card .good{color:var(--ui-ok)} .cr-card .bad{color:var(--ui-warn)}
.cr-card .best{display:inline-block;margin-top:4px;padding:4px 12px;border-radius:999px;background:rgba(var(--ui-accent-rgb),.22);color:#fff;letter-spacing:.3em;font-size:12px;font-weight:600}
.cr-count{position:absolute;left:50%;top:38%;transform:translate(-50%,-50%);font-size:120px;font-weight:700;color:#fff;
  text-shadow:0 4px 30px rgba(0,0,0,.6);opacity:0;pointer-events:none}
.cr-count.show{animation:crCount .6s ease-out forwards}
@keyframes crCount{0%{opacity:0;transform:translate(-50%,-50%) scale(1.5)}20%{opacity:1;transform:translate(-50%,-50%) scale(1)}100%{opacity:0;transform:translate(-50%,-50%) scale(.9)}}
@media (max-width:760px){.cr-map{display:none}}
.cr-touch .cr-speed{right:50%;bottom:auto;top:var(--ui-gutter);transform-origin:50% 0;transform:translateX(50%) scale(calc(var(--ui-scale) * .62))}
`;

/** The Canyon Run skin: call before building anything that reads theme.accent (gates, buggy paint, loading bar). */
export function applyTheme() {
  setTheme({
    accent: ACCENT, ok: '#7be08c', warn: '#ffd166',
    glass: 'linear-gradient(160deg,rgba(46,30,19,.62),rgba(20,12,8,.56))', 'glass-border': 'rgba(255,196,150,.13)',
    radius: '10px',
  });
  injectStyle('canyon-hud', CSS);
}

export class HUD {
  constructor({ track, race }) {
    applyTheme();
    this.track = track; this.race = race;
    const L = this.layer = new UILayer({ parent: document.body });
    const { el } = UIDom;
    // --- speed (bottom right)
    const sp = new Panel({ parent: L, corner: 'br', className: 'cr-speed' });
    this.gauge = new Gauge({ parent: sp, sweep: 250, radius: 92, width: 222, height: 196, ticks: 10, majorEvery: 2, color: ACCENT, glow: `drop-shadow(0 0 8px rgba(255,138,30,.55))`, footer: 'KM/H' });
    this.speed = new Readout({ parent: this.gauge.center, size: 'xl', value: '0' });
    this.gear = new Segments({ parent: sp, items: ['R', 'N', '1', '2', '3', '4', '5'] });
    this.slipBar = new Bar({ parent: sp, label: 'Slip', text: '', thickness: 3, fill: `linear-gradient(90deg,#ffd166,${ACCENT})` });
    // --- timer (top left)
    const tp = new Panel({ parent: L, corner: 'tl', className: 'cr-timer' });
    this.timer = new Readout({ parent: tp, label: 'Run time', size: 'lg', value: '0:00.00', sub: 'BEST --:--.--' });
    this.split = el('div', 'cr-split ui-num', tp.el, '');
    const row = el('div', 'row', tp.el);
    this.cpBadge = new Badge({ parent: row, text: 'CP 0/0' });
    this.lapBadge = new Badge({ parent: row, text: 'LAP 1/1' });
    this.offBadge = new Badge({ parent: row, text: 'OFF TRACK', className: 'cr-off' });
    this.offBadge.setVisible(false);
    // --- minimap (top right)
    const mp = new Panel({ parent: L, corner: 'tr', className: 'cr-map' });
    this.map = new Minimap({ parent: mp, size: 210, background: '#1a110b', strokes: [[22, 'rgba(120,80,50,.45)'], [12, 'rgba(240,196,150,.9)']], footLeft: 'NEXT —', footRight: '' });
    L.onScale((s) => this.map.setScale(s));
    this.map.setData(track.minimapData());
    this.markers = track.checkpoints.map((cp) => this.map.addMarker({ x: cp.x, z: cp.z, color: 'rgba(255,255,255,.35)', radius: 3, clamp: false }));
    this.nextMarker = this.map.addMarker({ x: 0, z: 0, color: ACCENT, radius: 5.5 });
    // --- transient messages
    this.flash = new Flash({ parent: L });
    this.count = el('div', 'cr-count ui-num', L.el, '');
    // --- help
    this.help = new HelpOverlay({
      parent: L, title: 'CANYON <b>·</b> RUN', subtitle: 'Off-road time trial',
      rows: [
        ['W / ↑ · RT', 'Throttle'], ['S / ↓ · LT', 'Brake / reverse'], ['A D / ← → · L-stick', 'Steer'],
        ['Space / X · B', 'Handbrake'], ['R · X', 'Reset to last checkpoint'], ['C · Y', 'Camera mode'], ['B', 'Look back'],
        ['F', 'Free-fly camera (WASD / drag)'], ['1 – 8', 'Time of day'], ['Enter · A / Start', 'Start / restart'],
        ['Esc', 'Menu'], ['M', 'Mute'], ['F3', 'Profiler'],
      ],
      footnote: 'Stay on the dirt track: the sand is slow. Pass every arch in order; splits compare with your best run.',
      hint: 'PRESS <kbd>H</kbd> TO HIDE',
    });
    // --- menu + finish cards
    this.menu = el('div', 'cr-card ui-glass', L.el);
    this.finish = el('div', 'cr-card cr-finish ui-glass hidden', L.el);
    this.renderMenu();
    this._t = 1;
  }

  renderMenu() {
    const r = this.race, tr = this.track;
    this.menu.innerHTML = `<h1>CANYON <b>RUN</b></h1><div class="ui-label tag">Off-road time trial · rally buggy</div>
      <div class="facts"><div><span class="ui-label">Length</span><b>${(tr.length / 1000).toFixed(2)} km</b></div>
      <div><span class="ui-label">Checkpoints</span><b>${tr.checkpoints.length - 1} + finish</b></div>
      <div><span class="ui-label">Laps</span><b>${r.laps}</b></div>
      <div><span class="ui-label">Best</span><b>${r.best ? formatTime(r.best.time) : '—'}</b></div></div>
      <div class="go">PRESS <kbd>ENTER</kbd> TO START</div>
      <div class="hint"><kbd>H</kbd> CONTROLS · <kbd>1</kbd>–<kbd>8</kbd> TIME OF DAY · DRAG TO ORBIT</div>`;
  }
  showMenu(on) { this.menuOn = on; this.menu.classList.toggle('hidden', !on || this.help.visible); if (on) this.renderMenu(); }
  /** Help and the menu card share the centre of the screen: showing one hides the other. */
  toggleHelp() { this.help.toggle(); this.menu.classList.toggle('hidden', !this.menuOn || this.help.visible); }
  setTouch(on) { this.layer.el.classList.toggle('cr-touch', !!on); }

  showFinish(res) {
    if (!res) { this.finish.classList.add('hidden'); return; }
    const rows = [];
    const n = res.splits.length;
    for (let i = 0; i < n; i++) {
      const isFinish = i === n - 1;
      const d = res.bestSplits ? res.splits[i] - res.bestSplits[i] : NaN;
      const cell = Number.isFinite(d) ? `<span class="${d <= 0 ? 'good' : 'bad'}">${formatDelta(d)}</span>` : '';
      if (n > 10 && !isFinish && i % 2 === 1) continue; // keep the table short
      rows.push(`<tr><td>${isFinish ? 'Finish' : `CP ${i + 1}`}</td><td>${formatTime(res.splits[i])}</td><td>${cell}</td></tr>`);
    }
    const vs = res.isBest ? (Number.isFinite(res.prevBest) ? `<span class="best">NEW BEST · ${formatDelta(res.time - res.prevBest)}</span>` : '<span class="best">FIRST RUN · BEST SET</span>')
      : `<span class="ui-label">Best ${formatTime(res.prevBest)} · <span class="bad">${formatDelta(res.time - res.prevBest)}</span></span>`;
    this.finish.innerHTML = `<h1>FINISH</h1><div class="big">${formatTime(res.time)}</div>${vs}
      <table>${rows.join('')}</table>
      <div class="hint">RESETS ${res.resets}</div>
      <div class="go"><kbd>ENTER</kbd> RUN AGAIN · <kbd>ESC</kbd> MENU</div>`;
    this.finish.classList.remove('hidden');
  }

  countdown(text) {
    const c = this.count;
    c.textContent = text; c.classList.remove('show'); void c.offsetWidth; c.classList.add('show');
  }

  setOffTrack(on) { this.offBadge.setVisible(on); }

  showSplit(delta, label) {
    const good = Number.isFinite(delta) && delta <= 0;
    this.split.className = `cr-split ui-num${Number.isFinite(delta) ? (good ? ' good' : ' bad') : ''}`;
    this.split.textContent = Number.isFinite(delta) ? `${label}  ${formatDelta(delta)}` : label;
  }

  markGates() {
    const next = this.race.next, cps = this.track.checkpoints, racing = this.race.state === 'racing';
    for (let k = 0; k < cps.length; k++) {
      const m = this.markers[k];
      m.color = !racing ? 'rgba(255,255,255,.35)' : (k !== 0 && k < next && next !== 0) ? 'rgba(123,224,140,.8)' : 'rgba(255,255,255,.4)';
    }
    const cp = cps[next];
    this.nextMarker.x = cp.x; this.nextMarker.z = cp.z;
  }

  /** Per frame. s: buggy.state, b: buggy (pose / gear), dt */
  update(dt, b) {
    const r = this.race, vc = b.vc;
    const kmh = Math.abs(vc.forwardSpeed) * 3.6;
    this.gauge.setValue(kmh / 200);
    this.slipBar.setValue(vc.wheelSlip);
    const p = b.object3d.position;
    this.map.update(dt, p.x, p.z, b.headingDeg, 380 + kmh * 2.2, '#ffffff'); // positional: no per-frame object
    if (!this.speed.tickText(dt)) return;
    this.speed.set(String(Math.round(kmh)));
    this.gear.select(vc.reversing ? 'R' : kmh < 2 && !b.throttle ? 'N' : String(b.gear));
    this.timer.set(formatTime(r.state === 'racing' || r.state === 'finished' ? r.time : 0));
    this.timer.setSub(`BEST ${r.best ? formatTime(r.best.time) : '--:--.--'}`);
    const done = r.splitCount, total = r.totalCheckpoints;
    this.cpBadge.setText('t', this.cpBadge.el, `CP ${Math.min(done, total)}/${total}`);
    this.lapBadge.setText('t', this.lapBadge.el, `LAP ${Math.min(r.lap + 1, r.laps)}/${r.laps}`);
    const cp = this.track.checkpoints[r.next];
    const dist = Math.hypot(cp.x - p.x, cp.z - p.z);
    this.map.setFoot(r.state === 'racing' ? `NEXT ${dist < 1000 ? `${dist.toFixed(0)} m` : `${(dist / 1000).toFixed(1)} km`}` : 'CANYON RUN', `${String(Math.round(((b.headingDeg % 360) + 360) % 360)).padStart(3, '0')}°`);
  }
}
