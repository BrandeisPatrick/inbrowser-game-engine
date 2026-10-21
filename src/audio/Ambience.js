// CityAmbience — procedural background bed on the 'ambience' bus:
//   wind (brown noise with slow gusts, stronger at altitude), distant city hum + traffic swell by day
//   (quieter at night), gulls near the water (sparse one-shot cries, panned, with a little reverb).
//
//   const amb = new CityAmbience(bus, { level: 1 });
//   amb.setEnvironment({ nearWater: 0..1 | bool, darkness: 0..1 });   // optional; defaults to daytime inland
//   amb.update(dt, { speedN, altitude });                              // per frame (cheap: a few setTargetAtTime)
//   amb.gull(1);                                                       // trigger a call by hand
import { noiseBuffer, NoiseVoice, LFO, makeGain, playTone } from './Synth.js';

function clamp(v, a, b) { return v < a ? a : v > b ? b : v; }

export class CityAmbience {
  /**
   * @param {import('./AudioBus.js').AudioBus} bus
   * @param {object} [o]  level (1), busName ('ambience'), gulls (true)
   */
  constructor(bus, o = {}) {
    this.bus = bus;
    this.level = o.level ?? 1;
    this.busName = o.busName || 'ambience';
    this.gulls = o.gulls !== false;
    this.enabled = true;
    this.env = { nearWater: 0, darkness: 0 };
    this._s = { nearWater: 0, darkness: 0 };   // smoothed
    this._gullTimer = 3 + Math.random() * 4;
    this._n = null;
    bus.onStart((ctx) => this._build(ctx));
  }

  setEnvironment(env = {}) {
    if (env.nearWater != null) this.env.nearWater = clamp(+env.nearWater || 0, 0, 1);
    if (env.darkness != null) this.env.darkness = clamp(+env.darkness || 0, 0, 1);
  }
  setEnabled(on) { this.enabled = !!on; }
  setLevel(v) { this.level = clamp(+v || 0, 0, 2); }

  _build(ctx) {
    const out = this.bus.bus(this.busName);
    const n = this._n = {};
    n.out = makeGain(ctx, 0);
    n.out.connect(out);
    const brown = noiseBuffer(ctx, 'brown', 5.3);
    const pink = noiseBuffer(ctx, 'pink', 4.7);
    // wind: low rumble with slow gusts (LFOs on gain and cutoff)
    n.wind = new NoiseVoice(ctx, { buffer: brown, rate: 0.9, filters: [{ type: 'lowpass', frequency: 420, Q: 0.5 }, { type: 'highpass', frequency: 35 }], gain: 0, out: n.out });
    n.gustF = new LFO(ctx, { frequency: 0.071, depth: 170, target: n.wind.filters[0].frequency });
    n.gustG = new LFO(ctx, { frequency: 0.113, depth: 0, target: n.wind.output.gain });
    // distant city: band-limited pink hum + slow traffic swell
    n.hum = new NoiseVoice(ctx, { buffer: pink, rate: 1, filters: [{ type: 'lowpass', frequency: 210, Q: 0.6 }, { type: 'highpass', frequency: 45 }], gain: 0, out: n.out });
    n.swell = new NoiseVoice(ctx, { buffer: pink, rate: 0.63, filters: [{ type: 'bandpass', frequency: 560, Q: 0.7 }], gain: 0, out: n.out });
    n.swellLFO = new LFO(ctx, { frequency: 0.047, depth: 0, target: n.swell.output.gain });
    n.gullBus = makeGain(ctx, 1);
    n.gullBus.connect(n.out);
    n.gullReverb = null;
    n.out.gain.setTargetAtTime(this.level, ctx.currentTime, 1.5);   // fade in
  }

  /** Play one gull call now (strength 0..1). Called automatically near water; public for demos / triggers. */
  gull(strength = 1) {
    const ctx = this.bus.ctx, n = this._n;
    if (!ctx || !n) return;
    if (!n.gullReverb) n.gullReverb = this.bus.reverbSend(n.gullBus, 0.4);
    const calls = 1 + Math.floor(Math.random() * 3);
    const pan = (Math.random() * 2 - 1) * 0.8;
    const base = 1050 + Math.random() * 350;
    let t = ctx.currentTime + 0.02;
    for (let i = 0; i < calls; i++) {
      const long = i === 0 && Math.random() < 0.6;
      const dur = long ? 0.42 : 0.2 + Math.random() * 0.08;
      playTone(ctx, n.gullBus, {
        type: 'sawtooth', when: t, pan, peak: 0.08 * strength, attack: 0.025, decay: dur,
        pitch: [[0, base * 0.85], [0.05, base * 1.65], [dur, base * (long ? 1.05 : 1.2)]],
        filter: { type: 'bandpass', frequency: base * 1.5, Q: 1.4 },
      });
      t += dur + 0.06 + Math.random() * 0.12;
    }
  }

  update(dt, { speedN = 0, altitude = 0 } = {}) {
    const ctx = this.bus.ctx, n = this._n;
    if (!ctx || !n || ctx.state !== 'running') return;
    const k = 1 - Math.exp(-0.6 * dt), s = this._s, e = this.env;
    s.nearWater += (e.nearWater - s.nearWater) * k;
    s.darkness += (e.darkness - s.darkness) * k;
    const day = 1 - s.darkness;
    const alt = clamp(altitude / 200, 0, 1);
    const duck = this.enabled ? 1 - 0.55 * clamp(speedN, 0, 1) : 0;   // the car's own wind takes over at speed
    const t = ctx.currentTime;
    n.out.gain.setTargetAtTime(this.enabled ? this.level : 0, t, 0.5);
    n.wind.output.gain.setTargetAtTime((0.03 + 0.05 * alt) * duck, t, 0.4);
    n.gustG.depth.gain.setTargetAtTime((0.012 + 0.03 * alt) * duck, t, 0.4);
    const city = (0.35 + 0.65 * day) * (1 - 0.5 * alt) * duck;
    n.hum.output.gain.setTargetAtTime(0.055 * city, t, 0.6);
    n.swell.output.gain.setTargetAtTime(0.022 * city * (0.3 + 0.7 * day), t, 0.6);
    n.swellLFO.depth.gain.setTargetAtTime(0.014 * city * day, t, 0.6);
    // gulls: sparse, only near water, mostly by day, not high up
    if (this.gulls && this.enabled) {
      const g = s.nearWater * (0.15 + 0.85 * day) * (1 - 0.8 * alt);
      this._gullTimer -= dt;
      if (this._gullTimer <= 0) {
        this._gullTimer = (2.5 + Math.random() * 6) / Math.max(0.35, g);
        if (g > 0.08) this.gull(g);
      }
    }
  }
}
