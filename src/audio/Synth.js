// Synth building blocks (no sample files): noise buffers, looping sources, filters, oscillator stacks,
// LFOs, envelopes, soft clipping and a couple of one-shot voices.
//
//   const white = noiseBuffer(ctx, 'white', 2.3);
//   const wind = new NoiseVoice(ctx, { buffer: white, rate: 0.77, filters: [{ type: 'lowpass', frequency: 400 }], out: bus.bus('effects') });
//   wind.set('gain', 0.3); wind.set(0, 1200);          // gain, or filter #0 frequency
//   const motor = new OscStack(ctx, { voices: [{ type: 'sawtooth' }, { type: 'triangle', detune: 9, ratio: 1.004 }], drive: 2.2,
//                                     filter: { type: 'lowpass', frequency: 900, Q: 2.5 }, out });
//   motor.setFrequency(220);
//   Envelope.perc(gainNode.gain, ctx.currentTime, { peak: 0.8, attack: 0.01, decay: 0.3 });

function clamp(v, a, b) { return v < a ? a : v > b ? b : v; }

/**
 * Looping noise buffer. type: 'white' | 'pink' | 'brown'. The loop seam is crossfaded.
 * (white/brown use the same generators as the original Roadster synth.)
 */
export function noiseBuffer(ctx, type = 'white', seconds = 2) {
  const len = Math.floor(ctx.sampleRate * seconds);
  const buf = ctx.createBuffer(1, len, ctx.sampleRate);
  const d = buf.getChannelData(0);
  if (type === 'white') {
    for (let i = 0; i < len; i++) d[i] = Math.random() * 2 - 1;
  } else if (type === 'pink') { // Paul Kellet's refined filter
    let b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0;
    for (let i = 0; i < len; i++) {
      const w = Math.random() * 2 - 1;
      b0 = 0.99886 * b0 + w * 0.0555179; b1 = 0.99332 * b1 + w * 0.0750759; b2 = 0.969 * b2 + w * 0.153852;
      b3 = 0.8665 * b3 + w * 0.3104856; b4 = 0.55 * b4 + w * 0.5329522; b5 = -0.7616 * b5 - w * 0.016898;
      d[i] = (b0 + b1 + b2 + b3 + b4 + b5 + b6 + w * 0.5362) * 0.11;
      b6 = w * 0.115926;
    }
  } else { // brown
    let last = 0;
    for (let i = 0; i < len; i++) {
      last = (last + 0.02 * (Math.random() * 2 - 1)) / 1.02;
      d[i] = last * 3.5;
    }
  }
  // crossfade the loop seam
  const fade = Math.min(2048, len >> 4);
  for (let i = 0; i < fade; i++) {
    const t = i / fade;
    d[len - fade + i] = d[len - fade + i] * (1 - t) + d[i] * t;
  }
  return buf;
}

/** Started, looping AudioBufferSourceNode. */
export function loopSource(ctx, buffer, rate = 1) {
  const s = ctx.createBufferSource();
  s.buffer = buffer;
  s.loop = true;
  s.playbackRate.value = rate;
  s.start();
  return s;
}

/**
 * BiquadFilterNode; only the given parameters are set (others keep WebAudio defaults).
 * @param {AudioContext} ctx
 * @param {{ type?: BiquadFilterType, frequency?: number, Q?: number, gain?: number }} [o]
 */
export function makeFilter(ctx, { type = 'lowpass', frequency, Q, gain } = {}) {
  const f = ctx.createBiquadFilter();
  f.type = type;
  if (frequency != null) f.frequency.value = frequency;
  if (Q != null) f.Q.value = Q;
  if (gain != null) f.gain.value = gain;
  return f;
}

export function makeGain(ctx, value = 1) { const g = ctx.createGain(); g.gain.value = value; return g; }

/** tanh soft-clip WaveShaper curve, normalised to ±1. */
export function softClipCurve(k, n = 1024) {
  const c = new Float32Array(n);
  const norm = Math.tanh(k);
  for (let i = 0; i < n; i++) {
    const x = (i / (n - 1)) * 2 - 1;
    c[i] = Math.tanh(k * x) / norm;
  }
  return c;
}

export function makeShaper(ctx, drive = 2, oversample = '2x') {
  const s = ctx.createWaveShaper();
  s.curve = softClipCurve(drive);
  s.oversample = oversample;
  return s;
}

/** Connect nodes in series; returns the last one. */
export function chain(...nodes) {
  for (let i = 0; i < nodes.length - 1; i++) nodes[i].connect(nodes[i + 1]);
  return nodes[nodes.length - 1];
}

/** Smoothly move an AudioParam toward value (time constant tc seconds). */
export function setParam(param, value, ctx, tc = 0.05) { param.setTargetAtTime(value, ctx.currentTime, tc); }

/** Looping noise -> filter chain -> gain. */
export class NoiseVoice {
  /**
   * @param {BaseAudioContext} ctx
   * @param {object} o  buffer | type+seconds, rate=1, filters=[{type,frequency,Q,gain}], gain=0, out
   */
  constructor(ctx, o = {}) {
    this.ctx = ctx;
    this.source = loopSource(ctx, o.buffer || noiseBuffer(ctx, o.type || 'white', o.seconds || 2), o.rate ?? 1);
    this.filters = (o.filters || []).map((f) => makeFilter(ctx, f));
    this.output = makeGain(ctx, o.gain ?? 0);
    chain(this.source, ...this.filters, this.output);
    if (o.out) this.output.connect(o.out);
  }
  /** set('gain', v) | set('rate', v) | set(filterIndex, frequency) */
  set(what, value, tc = 0.05) {
    const p = what === 'gain' ? this.output.gain : what === 'rate' ? this.source.playbackRate : this.filters[what].frequency;
    p.setTargetAtTime(value, this.ctx.currentTime, tc);
  }
  connect(node) { return this.output.connect(node); }
  stop() { try { this.source.stop(); } catch { /* already stopped */ } this.output.disconnect(); }
}

/**
 * Detuned oscillator stack -> mix -> optional soft clip -> optional filter -> gain.
 * voices: [{ type='sine', detune=0, gain=1, ratio=1 }]; setFrequency(f) drives voice i at f * ratio.
 */
export class OscStack {
  constructor(ctx, o = {}) {
    this.ctx = ctx;
    this.voices = (o.voices || [{ type: 'sine' }]).map((v) => ({ ratio: 1, gain: 1, detune: 0, type: 'sine', ...v }));
    this.mix = makeGain(ctx, o.mix ?? 1);
    this.oscillators = this.voices.map((v) => {
      const osc = ctx.createOscillator();
      osc.type = v.type;
      if (v.detune) osc.detune.value = v.detune;
      chain(osc, makeGain(ctx, v.gain), this.mix);
      return osc;
    });
    let tail = this.mix;
    this.shaper = o.drive ? makeShaper(ctx, o.drive, o.oversample || '2x') : null;
    if (this.shaper) tail = chain(tail, this.shaper);
    this.filter = o.filter ? makeFilter(ctx, o.filter) : null;
    if (this.filter) tail = chain(tail, this.filter);
    this.output = makeGain(ctx, o.gain ?? 0);
    tail.connect(this.output);
    if (o.out) this.output.connect(o.out);
    const f0 = o.frequency ?? 110;
    this.oscillators.forEach((osc, i) => { osc.frequency.value = f0 * this.voices[i].ratio; osc.start(); });
  }
  setFrequency(f, tc = 0.03) {
    const t = this.ctx.currentTime;
    for (let i = 0; i < this.oscillators.length; i++) this.oscillators[i].frequency.setTargetAtTime(f * this.voices[i].ratio, t, tc);
  }
  setGain(v, tc = 0.06) { this.output.gain.setTargetAtTime(v, this.ctx.currentTime, tc); }
  connect(node) { return this.output.connect(node); }
  stop() { for (const o of this.oscillators) try { o.stop(); } catch { /* stopped */ } this.output.disconnect(); }
}

/** Oscillator modulating an AudioParam: target += depth * wave(rate). */
export class LFO {
  constructor(ctx, { frequency = 1, depth = 0, type = 'sine', target = null } = {}) {
    this.ctx = ctx;
    this.osc = ctx.createOscillator();
    this.osc.type = type;
    this.osc.frequency.value = frequency;
    this.depth = makeGain(ctx, depth);
    this.osc.connect(this.depth);
    if (target) this.depth.connect(target);
    this.osc.start();
  }
  connect(param) { this.depth.connect(param); return this; }
  setDepth(v, tc = 0.05) { this.depth.gain.setTargetAtTime(v, this.ctx.currentTime, tc); }
  setRate(v, tc = 0.2) { this.osc.frequency.setTargetAtTime(v, this.ctx.currentTime, tc); }
  stop() { try { this.osc.stop(); } catch { /* stopped */ } }
}

/** Envelope helpers on AudioParams (exponential ramps need strictly positive targets: floor). */
export const Envelope = {
  /** Percussive: floor -> peak in attack, back to floor in decay, then hard 0. */
  perc(param, t, { peak = 1, attack = 0.005, decay = 0.25, floor = 0.0001 } = {}) {
    param.cancelScheduledValues(t);
    param.setValueAtTime(floor, t);
    param.exponentialRampToValueAtTime(Math.max(floor, peak), t + attack);
    param.exponentialRampToValueAtTime(floor, t + attack + decay);
    param.setValueAtTime(0, t + attack + decay + 0.01);
  },
  /** ADSR with a known hold time (seconds at sustain before release). */
  adsr(param, t, { peak = 1, attack = 0.01, decay = 0.1, sustain = 0.6, hold = 0.2, release = 0.3, floor = 0.0001 } = {}) {
    param.cancelScheduledValues(t);
    param.setValueAtTime(floor, t);
    param.linearRampToValueAtTime(peak, t + attack);
    param.setTargetAtTime(peak * sustain, t + attack, decay / 3);
    param.setValueAtTime(peak * sustain, t + attack + decay + hold);
    param.exponentialRampToValueAtTime(floor, t + attack + decay + hold + release);
    param.setValueAtTime(0, t + attack + decay + hold + release + 0.01);
  },
  /** Exponential pitch / value sweep. */
  sweep(param, t, from, to, duration) {
    param.cancelScheduledValues(t);
    param.setValueAtTime(Math.max(1e-4, from), t);
    param.exponentialRampToValueAtTime(Math.max(1e-4, to), t + duration);
  },
};

/**
 * One-shot tonal blip (UI clicks, chirps, birds): oscillator with pitch path + percussive envelope.
 * pitch: [[timeOffset, Hz], ...] (exponential ramps between points). Returns the end time.
 */
export function playTone(ctx, out, { type = 'sine', pitch = [[0, 880]], peak = 0.3, attack = 0.005, decay = 0.2, filter = null, pan = 0, when = 0 } = {}) {
  const t = Math.max(ctx.currentTime, when || ctx.currentTime);
  const osc = ctx.createOscillator();
  osc.type = type;
  osc.frequency.setValueAtTime(pitch[0][1], t);
  for (let i = 1; i < pitch.length; i++) osc.frequency.exponentialRampToValueAtTime(Math.max(1, pitch[i][1]), t + pitch[i][0]);
  const g = makeGain(ctx, 0);
  Envelope.perc(g.gain, t, { peak, attack, decay });
  let tail = chain(osc, ...(filter ? [makeFilter(ctx, filter)] : []), g);
  if (pan && ctx.createStereoPanner) { const p = ctx.createStereoPanner(); p.pan.value = clamp(pan, -1, 1); tail = chain(tail, p); }
  tail.connect(out);
  const end = t + attack + decay + 0.05;
  osc.start(t); osc.stop(end);
  osc.onended = () => { try { tail.disconnect(); } catch { /* gone */ } };
  return end;
}

/** One-shot filtered noise burst (impacts, whooshes, clicks). */
export function playNoiseBurst(ctx, out, buffer, { peak = 0.5, attack = 0.004, decay = 0.2, filter = { type: 'lowpass', frequency: 1200 }, rate = 1, when = 0 } = {}) {
  const t = Math.max(ctx.currentTime, when || ctx.currentTime);
  const src = ctx.createBufferSource();
  src.buffer = buffer; src.playbackRate.value = rate;
  const g = makeGain(ctx, 0);
  Envelope.perc(g.gain, t, { peak, attack, decay });
  chain(src, ...(filter ? [makeFilter(ctx, filter)] : []), g).connect(out);
  src.start(t, Math.random() * Math.max(0, buffer.duration - attack - decay - 0.1));
  src.stop(t + attack + decay + 0.05);
  src.onended = () => { try { g.disconnect(); } catch { /* gone */ } };
}
