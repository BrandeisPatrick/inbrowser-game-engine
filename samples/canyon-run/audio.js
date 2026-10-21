// Buggy sound, all synthesised with the engine's AudioBus + Synth voices (no samples):
//   engine growl: saw + sub-octave square + fifth through a soft clip and a resonant low-pass, amplitude-modulated
//   at the firing rate (LFO), plus a band-passed noise "rasp" under load; a 5-speed auto gearbox drives the rpm.
//   gravel: brown noise band-passed by speed (louder on sand), tyre scrub on slip; wind by speed^2;
//   desert ambience: slow gusting low-passed noise on the ambience bus; one-shots via playTone / playNoiseBurst.
import { AudioBus, OscStack, NoiseVoice, LFO, noiseBuffer, playTone, playNoiseBurst } from '@vibe/engine';

const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const REDLINE = 7400;

export class BuggyAudio {
  constructor({ volume = 0.75 } = {}) {
    this.bus = new AudioBus({ volume });
    this.started = false;
    this.rpm = 900; this._gustT = 0; this._gust = 0.5;
    this.bus.onStart((ctx) => { this._build(ctx); this.started = true; });
  }
  start() { return this.bus.start(); }
  toggleMute() { return this.bus.toggleMute(); }

  _build(ctx) {
    const n = this.n = {};
    const eng = this.bus.bus('engine'), fx = this.bus.bus('effects'), amb = this.bus.bus('ambience');
    this.white = noiseBuffer(ctx, 'white', 2.1);
    const brown = noiseBuffer(ctx, 'brown', 3.3), pink = noiseBuffer(ctx, 'pink', 2.7);
    n.motor = new OscStack(ctx, {
      voices: [{ type: 'sawtooth', gain: 0.55 }, { type: 'square', gain: 0.32, ratio: 0.5 }, { type: 'sawtooth', gain: 0.18, ratio: 1.5, detune: 7 }],
      mix: 0.45, drive: 3.2, filter: { type: 'lowpass', frequency: 700, Q: 4 }, gain: 0, out: eng, frequency: 40,
    });
    n.lumpy = new LFO(ctx, { frequency: 20, depth: 0, target: n.motor.output.gain, type: 'triangle' });
    n.rasp = new NoiseVoice(ctx, { buffer: this.white, rate: 1, filters: [{ type: 'bandpass', frequency: 900, Q: 2.2 }], gain: 0, out: eng });
    n.gravel = new NoiseVoice(ctx, { buffer: brown, rate: 1.4, filters: [{ type: 'bandpass', frequency: 260, Q: 0.7 }, { type: 'highshelf', frequency: 1800, gain: 4 }], gain: 0, out: fx });
    n.scrub = new NoiseVoice(ctx, { buffer: pink, rate: 1.1, filters: [{ type: 'bandpass', frequency: 1300, Q: 1.4 }], gain: 0, out: fx });
    n.wind = new NoiseVoice(ctx, { buffer: this.white, rate: 0.7, filters: [{ type: 'lowpass', frequency: 500, Q: 0.6 }], gain: 0, out: fx });
    n.desert = new NoiseVoice(ctx, { buffer: brown, rate: 0.6, filters: [{ type: 'lowpass', frequency: 380, Q: 0.4 }, { type: 'peaking', frequency: 160, gain: 5, Q: 0.8 }], gain: 0.0, out: amb });
    n.whistle = new NoiseVoice(ctx, { buffer: pink, rate: 0.9, filters: [{ type: 'bandpass', frequency: 1700, Q: 9 }], gain: 0, out: amb });
  }

  _set(p, v, tc) { p.setTargetAtTime(v, this.bus.ctx.currentTime, tc); }

  /** s: { rpm, throttle, speed (m/s), slip, grounded, sand } (rpm from the buggy's virtual gearbox) */
  update(dt, s) {
    const ctx = this.bus.ctx;
    if (!ctx || ctx.state !== 'running' || !this.n) return;
    const n = this.n;
    this.rpm = s.rpm;
    const f = this.rpm / 60;                                // fundamental (Hz), ~15 .. 123
    const load = clamp(s.throttle, 0, 1);
    this._set(n.motor.oscillators[0].frequency, f * 2, 0.03);
    this._set(n.motor.oscillators[1].frequency, f, 0.03);
    this._set(n.motor.oscillators[2].frequency, f * 3, 0.03);
    this._set(n.motor.filter.frequency, 380 + f * 9 + load * 1500, 0.05);
    this._set(n.motor.output.gain, 0.09 + 0.12 * load + 0.04 * (this.rpm / REDLINE), 0.05);
    this._set(n.lumpy.osc.frequency, f * 0.5, 0.05);
    this._set(n.lumpy.depth.gain, 0.05 + 0.04 * load, 0.1);
    this._set(n.rasp.output.gain, 0.02 + 0.07 * load * (this.rpm / REDLINE), 0.05);
    this._set(n.rasp.filters[0].frequency, 500 + f * 14, 0.05);
    const sN = clamp(s.speed / 50, 0, 1);
    const ground = s.grounded ? 1 : 0;
    this._set(n.gravel.output.gain, ground * (0.05 + (s.sand ? 0.28 : 0.13) * Math.min(1, s.speed / 25)), 0.08);
    this._set(n.gravel.filters[0].frequency, 180 + 500 * sN, 0.1);
    this._set(n.scrub.output.gain, ground * 0.22 * clamp(s.slip, 0, 1) * Math.min(1, s.speed / 6), 0.05);
    this._set(n.wind.output.gain, 0.36 * sN * sN, 0.1);
    this._set(n.wind.filters[0].frequency, 300 + 2600 * sN, 0.1);
    // desert gusts (ambience)
    this._gustT -= dt;
    if (this._gustT <= 0) { this._gustT = 2.5 + Math.random() * 5; this._gust = 0.25 + Math.random() * 0.75; }
    this._set(n.desert.output.gain, 0.06 + 0.12 * this._gust, 1.6);
    this._set(n.desert.filters[0].frequency, 250 + 300 * this._gust, 1.6);
    this._set(n.whistle.output.gain, 0.012 * this._gust * this._gust, 2.0);
  }

  /** Checkpoint chime (rising arpeggio; `good` = faster than best split). */
  chime(good = true) {
    const ctx = this.bus.ctx; if (!ctx || !this.n) return;
    const out = this.bus.bus('ui'), base = good ? 880 : 660, t = ctx.currentTime;
    playTone(ctx, out, { type: 'triangle', pitch: [[0, base]], peak: 0.22, decay: 0.25, when: t });
    playTone(ctx, out, { type: 'triangle', pitch: [[0, base * 1.25]], peak: 0.2, decay: 0.25, when: t + 0.09 });
    playTone(ctx, out, { type: 'sine', pitch: [[0, base * 1.5]], peak: 0.2, decay: 0.45, when: t + 0.18 });
  }
  /** Countdown beep (go = high, long). */
  beep(go = false) {
    const ctx = this.bus.ctx; if (!ctx || !this.n) return;
    playTone(ctx, this.bus.bus('ui'), { type: 'square', pitch: [[0, go ? 1046 : 523]], peak: 0.13, decay: go ? 0.6 : 0.22, filter: { type: 'lowpass', frequency: 2600 } });
  }
  finishFanfare() {
    const ctx = this.bus.ctx; if (!ctx || !this.n) return;
    const out = this.bus.bus('ui'), t = ctx.currentTime;
    [523, 659, 784, 1046].forEach((hz, i) => playTone(ctx, out, { type: 'triangle', pitch: [[0, hz]], peak: 0.2, decay: i === 3 ? 0.9 : 0.25, when: t + i * 0.12 }));
  }
  thump(strength) {
    const ctx = this.bus.ctx; if (!ctx || !this.n) return;
    playNoiseBurst(ctx, this.bus.bus('effects'), this.white, { peak: clamp(strength, 0.1, 0.8), decay: 0.25, filter: { type: 'lowpass', frequency: 320 }, rate: 0.6 });
  }
}
