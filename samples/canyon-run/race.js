// Time-trial rules: menu -> countdown (gantry lights) -> racing (checkpoint sequence with splits, off-track
// detection, reset to the last checkpoint) -> finished. Best run (time + splits) persists in localStorage.
// Checkpoints are gate planes: crossing one = the signed distance along the gate tangent changes sign while
// inside the pillars. Allocation-free per frame; events go out through small callbacks.
import { TRACK_HALF_WIDTH, GATE_HALF_WIDTH } from './track.js';

const STORE_KEY = 'canyonRun.best.v1';

export function formatTime(t) {
  if (!Number.isFinite(t)) return '--:--.--';
  const m = Math.floor(t / 60), s = t - m * 60;
  return `${m}:${s < 10 ? '0' : ''}${s.toFixed(2)}`;
}
export function formatDelta(d) { return `${d < 0 ? '−' : '+'}${Math.abs(d).toFixed(2)}`; }

function loadBest() {
  try { const v = JSON.parse(localStorage.getItem(STORE_KEY) || 'null'); if (v && Number.isFinite(v.time) && Array.isArray(v.splits)) return v; } catch { /* private mode / bad JSON */ }
  return null;
}
function saveBest(b) { try { localStorage.setItem(STORE_KEY, JSON.stringify(b)); } catch { /* storage blocked */ } }

export class Race {
  /** @param {object} o  track, laps (1), on: { countdown(n), go(), checkpoint(info), lap(n), finish(info), offTrack(bool), missed() } */
  constructor({ track, laps = 1, on = {} }) {
    this.track = track; this.laps = laps; this.on = on;
    this.cps = track.checkpoints;
    this.state = 'menu';
    this.time = 0; this.countdownT = 0; this._lights = 0;
    this.next = 1; this.lap = 0; this.lastCp = 0; this.resets = 0;
    this.splits = new Float64Array(this.cps.length * laps);
    this.splitCount = 0;
    this.best = loadBest();
    this.offTrack = false;
    this.near = { i: 0, s: 0, lateral: 0, dist: 0, along: 0 };
    this._hint = -1;
    this._prevA = NaN;
    this._missedShown = false;
    this.lastDelta = NaN;
    this.result = null;
  }

  get checkpointCount() { return this.cps.length; }  // includes the finish line (index 0)
  get totalCheckpoints() { return this.cps.length * this.laps; }

  toMenu() { this.state = 'menu'; this.time = 0; this._lights = 0; }
  startCountdown() {
    this.state = 'countdown'; this.countdownT = 0; this._lights = 0;
    this.time = 0; this.next = 1; this.lap = 0; this.lastCp = 0; this.resets = 0; this.splitCount = 0;
    this._prevA = NaN; this._missedShown = false; this.lastDelta = NaN; this.result = null; this._hint = -1;
  }

  /** Per frame, after the vehicle moved. pos: interpolated buggy position. */
  update(dt, pos) {
    const tr = this.track;
    tr.nearest(pos.x, pos.z, this._hint, this.near);
    this._hint = this.near.i;
    const off = this.near.dist > TRACK_HALF_WIDTH + 0.7;
    if (off !== this.offTrack) { this.offTrack = off; if (this.on.offTrack) this.on.offTrack(off); }
    if (this.state === 'countdown') {
      this.countdownT += dt;
      const lights = Math.min(5, Math.floor(this.countdownT / 0.62));
      if (lights !== this._lights && lights > 0) { this._lights = lights; if (this.on.countdown) this.on.countdown(lights); }
      if (this.countdownT >= 3.7) { this.state = 'racing'; this.time = 0; if (this.on.go) this.on.go(); }
      return;
    }
    if (this.state !== 'racing') return;
    this.time += dt;
    // gate crossing for the next checkpoint
    const cp = this.cps[this.next];
    const dx = pos.x - cp.x, dz = pos.z - cp.z;
    const a = dx * cp.tx + dz * cp.tz, lat = Math.abs(dx * -cp.tz + dz * cp.tx);
    if (Math.abs(a) < 40 && lat < GATE_HALF_WIDTH + 1.2) {
      if (this._prevA < 0 && a >= 0) this._pass();
      this._prevA = a;
    } else this._prevA = Math.abs(a) < 40 ? a : NaN;
    // missed it? (progress well past the gate along the polyline)
    if (!this._missedShown && this.state === 'racing') {
      let ds = this.near.s - cp.s; const L = tr.length;
      if (ds > L / 2) ds -= L; if (ds < -L / 2) ds += L;
      if (ds > 45 && ds < 400) { this._missedShown = true; if (this.on.missed) this.on.missed(); }
    }
  }

  _pass() {
    const k = this.next, idx = this.splitCount++;
    this.splits[idx] = this.time;
    const best = this.best && this.best.splits.length === this.totalCheckpoints ? this.best.splits[idx] : NaN;
    this.lastDelta = Number.isFinite(best) ? this.time - best : NaN;
    this.lastCp = k; this._prevA = NaN; this._missedShown = false;
    if (k === 0) {
      this.lap++;
      if (this.lap >= this.laps) { this._finish(); return; }
      if (this.on.lap) this.on.lap(this.lap);
    }
    this.next = (k + 1) % this.cps.length;
    if (this.on.checkpoint) this.on.checkpoint({ index: k, number: idx + 1, time: this.time, delta: this.lastDelta, next: this.next });
  }

  _finish() {
    this.state = 'finished';
    const prev = this.best;
    const isBest = !prev || this.time < prev.time;
    const splits = Array.from(this.splits.subarray(0, this.splitCount));
    if (isBest) { this.best = { time: this.time, splits, date: Date.now() }; saveBest(this.best); }
    this.result = { time: this.time, splits, bestSplits: prev && prev.splits.length === splits.length ? prev.splits : null, isBest, prevBest: prev ? prev.time : NaN, resets: this.resets };
    if (this.on.finish) this.on.finish(this.result);
  }

  /** Spawn pose for a reset: the last passed gate (or the grid), a few metres past it. */
  respawnPose(out) {
    const tr = this.track, cp = this.cps[this.lastCp];
    const i = (cp.i + (this.state === 'racing' && this.splitCount > 0 ? 4 : -7) + tr.n) % tr.n;
    out.x = tr.x[i]; out.z = tr.z[i]; out.heading = Math.atan2(-tr.tx[i], -tr.tz[i]);
    return out;
  }
  noteReset() { if (this.state === 'racing') this.resets++; this._prevA = NaN; this._missedShown = false; }
}
