// VehicleAudio — Web Audio synthesis for an electric car with thrusters (no sample files), built from
// AudioBus + Synth voices, plus an optional CityAmbience bed.
//
//   const audio = new VehicleAudio({ volume: 0.7 });          // or { bus } to share an AudioBus
//   // inside a user-gesture handler (keydown / pointerdown):
//   audio.start();               // idempotent; creates the graph once, resumes the context
//   audio.update(dt, state)      // every frame: { speedKmh, rpmLike, thrust, grounded, mode, wheelSlip,
//                                //   boostLevel, thrusterLevel, damage, altitude }
//   audio.setMuted(bool); audio.setVolume(0..1); audio.toggleMute(); audio.impact(0..1);
//   audio.setEnvironment({ nearWater, darkness });             // optional ambience inputs
//
// Layers: EV motor whine (engine bus), inverter whistle, thruster roar + hiss (engine), tyre slip, wind,
// impact thump (effects), ambience (ambience bus).
import { AudioBus } from './AudioBus.js';
import { noiseBuffer, loopSource, makeFilter, makeGain, chain, OscStack, LFO, NoiseVoice } from './Synth.js';
import { CityAmbience } from './Ambience.js';

function clamp(v, a, b) { return v < a ? a : v > b ? b : v; }

export class VehicleAudio {
  /**
   * @param {object} [opts]
   * @param {number} [opts.volume=0.7]
   * @param {AudioBus} [opts.bus]          share a bus (else one is created)
   * @param {boolean|object} [opts.ambience=true]  CityAmbience (options object allowed)
   * @param {number} [opts.topSpeed=400/3.6]   m/s at which pitch / wind reach their max
   */
  constructor(opts = {}) {
    this.bus = opts.bus || new AudioBus({ volume: opts.volume ?? 0.7 });
    if (!opts.bus && opts.volume != null) this.bus.volume = opts.volume;
    this.topSpeed = opts.topSpeed ?? 400 / 3.6;
    this.started = false;
    this._prevDamage = null;
    this._lastImpact = -1;
    this._n = null; // node refs
    const amb = opts.ambience ?? true;
    this.ambience = amb ? new CityAmbience(this.bus, typeof amb === 'object' ? amb : {}) : null;
    this.bus.onStart((ctx) => { this._build(ctx); this.started = true; });
  }

  get ctx() { return this.bus.ctx; }
  get muted() { return this.bus.muted; }
  set muted(m) { this.bus.setMuted(m); }
  get volume() { return this.bus.volume; }
  set volume(v) { this.bus.setVolume(v); }

  /** Must be called from a user gesture. Safe to call repeatedly. Returns a Promise. */
  start() { return this.bus.start(); }
  get running() { return this.bus.running; }

  setMuted(m) { this.bus.setMuted(m); }
  toggleMute() { return this.bus.toggleMute(); }
  setVolume(v) { this.bus.setVolume(v); }
  setEnvironment(env) { if (this.ambience) this.ambience.setEnvironment(env); }

  // ------------------------------------------------------------------ graph
  _build(ctx) {
    const n = this._n = {};
    const engine = this.bus.bus('engine'), fx = this.bus.bus('effects');
    const white = noiseBuffer(ctx, 'white', 2.3);
    const brown = noiseBuffer(ctx, 'brown', 3.1);

    // ---- EV motor: detuned saw + triangle (+ sub sine) -> soft clip -> lowpass ----
    n.motor = new OscStack(ctx, {
      voices: [
        { type: 'sawtooth', gain: 0.45, ratio: 1 },
        { type: 'triangle', gain: 0.7, ratio: 1.004, detune: 9 },
        { type: 'sine', gain: 0.5, ratio: 0.5 },
      ],
      mix: 0.5, drive: 2.2, oversample: '2x',
      filter: { type: 'lowpass', Q: 2.5, frequency: 900 },
      gain: 0, out: engine, frequency: 80,
    });
    for (const o of n.motor.oscillators) o.frequency.value = 80;

    // ---- inverter whistle ----
    n.iOsc = ctx.createOscillator(); n.iOsc.type = 'sine'; n.iOsc.frequency.value = 2000;
    n.iVib = new LFO(ctx, { frequency: 5.5, depth: 18, target: n.iOsc.frequency });
    n.iGain = makeGain(ctx, 0);
    chain(n.iOsc, n.iGain, engine);
    n.iOsc.start();

    // ---- tyre slip: white noise -> bandpass ~1 kHz (+ narrow squeal band) ----
    n.tSrc = loopSource(ctx, white, 1);
    n.tBP = makeFilter(ctx, { type: 'bandpass', frequency: 1000, Q: 1.6 });
    n.tSq = makeFilter(ctx, { type: 'bandpass', frequency: 1900, Q: 12 });
    n.tGain = makeGain(ctx, 0);
    chain(n.tSrc, n.tBP, n.tGain);
    chain(n.tSrc, n.tSq, makeGain(ctx, 2.2), n.tGain);
    n.tGain.connect(fx);

    // ---- wind: white noise -> lowpass, gain ~ speed^2 ----
    n.wind = new NoiseVoice(ctx, { buffer: white, rate: 0.77, filters: [{ type: 'highpass', frequency: 60 }, { type: 'lowpass', frequency: 400, Q: 0.7 }], gain: 0, out: fx });

    // ---- thruster roar: brown noise -> bandpass 200..600 Hz, rumble LFO on gain ----
    n.thrust = new NoiseVoice(ctx, { buffer: brown, rate: 1, filters: [{ type: 'bandpass', frequency: 250, Q: 0.9 }, { type: 'lowshelf', frequency: 160, gain: 6 }], gain: 0, out: engine });
    n.thLFO = new LFO(ctx, { frequency: 7, depth: 0, target: n.thrust.output.gain });
    // hiss layer
    n.hiss = new NoiseVoice(ctx, { buffer: white, rate: 1.13, filters: [{ type: 'highpass', frequency: 3200 }, { type: 'peaking', frequency: 6500, gain: 5, Q: 1 }], gain: 0, out: engine });

    // ---- impact thump (persistent nodes, enveloped on demand) ----
    n.impOsc = ctx.createOscillator(); n.impOsc.type = 'sine'; n.impOsc.frequency.value = 55;
    n.impGain = makeGain(ctx, 0);
    chain(n.impOsc, n.impGain, fx);
    n.impOsc.start();
    n.impNoise = new NoiseVoice(ctx, { buffer: brown, rate: 1.6, filters: [{ type: 'lowpass', frequency: 900 }], gain: 0, out: fx });
  }

  _set(param, value, tc = 0.05) {
    param.setTargetAtTime(value, this.bus.ctx.currentTime, tc);
  }

  impact(strength = 1) {
    const ctx = this.bus.ctx;
    if (!ctx || ctx.state !== 'running' || !this._n) return;
    const n = this._n, t = ctx.currentTime;
    if (t - this._lastImpact < 0.08) return;
    this._lastImpact = t;
    const s = clamp(strength, 0.15, 1);
    const f = n.impOsc.frequency, g = n.impGain.gain, ng = n.impNoise.output.gain;
    f.cancelScheduledValues(t); f.setValueAtTime(110 + 40 * s, t); f.exponentialRampToValueAtTime(38, t + 0.25);
    g.cancelScheduledValues(t); g.setValueAtTime(0.0001, t);
    g.exponentialRampToValueAtTime(0.9 * s, t + 0.008); g.exponentialRampToValueAtTime(0.0001, t + 0.35); g.setValueAtTime(0, t + 0.36);
    ng.cancelScheduledValues(t); ng.setValueAtTime(0.0001, t);
    ng.exponentialRampToValueAtTime(0.7 * s, t + 0.005); ng.exponentialRampToValueAtTime(0.0001, t + 0.18); ng.setValueAtTime(0, t + 0.19);
  }

  // ------------------------------------------------------------------ per frame
  update(dt, s) {
    if (!s) return;
    // damage delta -> thump (tracked even before audio starts so no spurious hit on start)
    const dmg = s.damage || 0;
    if (this._prevDamage !== null) {
      const d = dmg - this._prevDamage;
      if (d > 0.004) this.impact(clamp(d * 12, 0.2, 1));
    }
    this._prevDamage = dmg;

    const ctx = this.bus.ctx;
    if (!ctx || ctx.state !== 'running' || !this._n) return;
    const n = this._n;
    const speed = Math.abs(s.speedKmh || 0) / 3.6;
    const sN = clamp(speed / this.topSpeed, 0, 1);
    const rpm = clamp(s.rpmLike || 0, 0, 1);
    const thrust = clamp(s.thrust || 0, 0, 1);
    const grounded = s.grounded !== false && s.mode !== 'flight';
    const slip = clamp(s.wheelSlip || 0, 0, 1);
    const boost = s.boostLevel != null && rpm > 0.95 ? 1 : 0;

    // motor whine: 80..900 Hz gear-less, slight extra pitch under load
    const f = 80 + 820 * Math.pow(sN, 0.85) + 25 * rpm;
    const m = n.motor, osc = m.oscillators;
    this._set(osc[0].frequency, f, 0.03);
    this._set(osc[1].frequency, f * 1.004, 0.03);
    this._set(osc[2].frequency, f * 0.5, 0.03);
    this._set(m.filter.frequency, clamp(500 + f * 2.2 + 1800 * rpm, 300, 9000), 0.05);
    const moving = clamp(speed / 3, 0, 1);
    const motorVol = (0.025 + 0.05 * sN) * Math.max(moving, rpm) + 0.16 * rpm * (0.4 + 0.6 * moving) + 0.03 * boost;
    this._set(m.output.gain, motorVol, 0.06);

    // inverter whistle ~2..6 kHz, proportional to power
    const power = rpm * (0.25 + 0.75 * clamp(speed / 30, 0, 1));
    this._set(n.iOsc.frequency, 2000 + 4000 * sN, 0.05);
    this._set(n.iGain.gain, 0.018 * power, 0.08);

    // tyres
    const tyre = grounded ? slip * clamp(speed / 4, 0, 1) : 0;
    this._set(n.tGain.gain, 0.32 * tyre * tyre + 0.08 * tyre, 0.04);
    this._set(n.tBP.frequency, 850 + 400 * slip, 0.05);
    this._set(n.tSq.frequency, 1700 + 500 * slip + 300 * sN, 0.08);

    // wind ~ speed^2
    const wind = sN * sN;
    this._set(n.wind.output.gain, 0.55 * wind + (grounded ? 0 : 0.03 * clamp(speed / 10, 0, 1)), 0.1);
    this._set(n.wind.filters[1].frequency, 350 + 3200 * sN, 0.1);

    // thrusters
    const tl = s.thrusterLevel == null ? 1 : clamp(s.thrusterLevel, 0, 1);
    const sputter = tl < 0.05 ? 0.35 : 1;
    const th = thrust * sputter;
    this._set(n.thrust.output.gain, 0.6 * th, 0.05);
    this._set(n.thLFO.depth.gain, 0.16 * th, 0.05);
    this._set(n.thLFO.osc.frequency, 5 + 6 * th, 0.2);
    this._set(n.thrust.filters[0].frequency, 200 + 400 * th, 0.08);
    this._set(n.hiss.output.gain, 0.09 * th * (0.6 + 0.4 * tl), 0.05);

    if (this.ambience) this.ambience.update(dt, { speedN: sN, altitude: s.altitude || 0 });
  }

  dispose() { this.bus.dispose(); this._n = null; this.started = false; }
}
