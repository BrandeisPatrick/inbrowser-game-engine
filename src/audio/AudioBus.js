// AudioBus — WebAudio context lifecycle + mixer.
//
//   const bus = new AudioBus({ volume: 0.7 });
//   bus.unlockOnGesture();                 // or call bus.start() inside your own keydown / pointerdown handler
//   bus.onStart((ctx) => buildGraph(ctx)); // runs once, synchronously, as soon as the context exists
//   voice.connect(bus.bus('effects'));     // buses: engine, effects, ambience, ui (+ any name, created on demand)
//   bus.setMuted(true); bus.toggleMute(); bus.setVolume(0.5); bus.setBusVolume('ambience', 0.3);
//   const send = bus.reverbSend(node, 0.2); // simple algorithmic reverb (created lazily)
//
// Graph: bus gains -> master gain -> compressor -> destination; reverb return -> master.
// Safe to import in Node (no AudioContext until start()).

function clamp(v, a, b) { return v < a ? a : v > b ? b : v; }

export const DEFAULT_BUSES = ['engine', 'effects', 'ambience', 'ui'];

export class AudioBus {
  /**
   * @param {object} [o]
   * @param {number} [o.volume=0.7]  master volume 0..1
   * @param {string[]} [o.buses]     bus names created at start
   * @param {object|null} [o.compressor]  DynamicsCompressor settings (null = none)
   * @param {number} [o.reverbSeconds=2.4]  impulse length of the reverb send
   * @param {string} [o.latencyHint='interactive']
   */
  constructor(o = {}) {
    this.ctx = null;
    this.volume = o.volume ?? 0.7;
    this.muted = false;
    this.busNames = o.buses || DEFAULT_BUSES;
    this.compressorSettings = o.compressor === undefined ? { threshold: -14, knee: 12, ratio: 4, attack: 0.005, release: 0.2 } : o.compressor;
    this.reverbSeconds = o.reverbSeconds ?? 2.4;
    this.latencyHint = o.latencyHint || 'interactive';
    this.master = null;
    this.compressor = null;
    this.buses = {};
    this._busVolume = {};
    this._reverb = null;
    this._onStart = [];
    this._unlock = null;
  }

  get started() { return !!this.ctx; }
  get running() { return !!this.ctx && this.ctx.state === 'running'; }
  get now() { return this.ctx ? this.ctx.currentTime : 0; }

  /** Must be called from a user gesture. Safe to call repeatedly. Resolves true when running. */
  start() {
    const AC = typeof window !== 'undefined' && (window.AudioContext || window.webkitAudioContext);
    if (!AC) return Promise.resolve(false);
    if (!this.ctx) {
      this.ctx = new AC({ latencyHint: this.latencyHint });
      this._buildMaster();
      const cbs = this._onStart; this._onStart = [];
      for (const cb of cbs) cb(this.ctx, this);
    }
    if (this.ctx.state !== 'running') return this.ctx.resume().then(() => true, () => false);
    return Promise.resolve(true);
  }

  /** Run cb(ctx, bus) once the context exists (immediately if it already does). */
  onStart(cb) { if (this.ctx) cb(this.ctx, this); else this._onStart.push(cb); return this; }

  /** Start on the first keydown / pointerdown / touchend on target. Returns a remover. */
  unlockOnGesture(target = typeof window !== 'undefined' ? window : null) {
    if (!target || this._unlock) return this._unlock || (() => {});
    const evs = ['keydown', 'pointerdown', 'touchend'];
    const go = () => { this.start(); off(); };
    const off = () => { for (const e of evs) target.removeEventListener(e, go); this._unlock = null; };
    for (const e of evs) target.addEventListener(e, go);
    this._unlock = off;
    return off;
  }

  _buildMaster() {
    const ctx = this.ctx;
    this.master = ctx.createGain();
    this.master.gain.value = 0;
    let out = ctx.destination;
    const c = this.compressorSettings;
    if (c) {
      const comp = this.compressor = ctx.createDynamicsCompressor();
      comp.threshold.value = c.threshold; comp.knee.value = c.knee; comp.ratio.value = c.ratio;
      comp.attack.value = c.attack; comp.release.value = c.release;
      comp.connect(ctx.destination);
      out = comp;
    }
    this.master.connect(out);
    for (const name of this.busNames) this.bus(name);
    this._applyMaster();
  }

  /** GainNode of a named bus (created on demand once started; null before start). */
  bus(name) {
    if (!this.ctx) return null;
    let g = this.buses[name];
    if (!g) {
      g = this.buses[name] = this.ctx.createGain();
      g.gain.value = this._busVolume[name] ?? 1;
      g.connect(this.master);
    }
    return g;
  }

  setBusVolume(name, v, tc = 0.05) {
    this._busVolume[name] = clamp(+v || 0, 0, 2);
    const g = this.ctx && this.bus(name);
    if (g) g.gain.setTargetAtTime(this._busVolume[name], this.ctx.currentTime, tc);
  }
  getBusVolume(name) { return this._busVolume[name] ?? 1; }

  setMuted(m) { this.muted = !!m; this._applyMaster(); }
  toggleMute() { this.setMuted(!this.muted); return this.muted; }
  setVolume(v) { this.volume = clamp(+v || 0, 0, 1); this._applyMaster(); }

  _applyMaster() {
    if (!this.ctx || !this.master) return;
    const g = this.muted ? 0 : this.volume;
    this.master.gain.setTargetAtTime(g, this.ctx.currentTime, 0.05);
  }

  /** Reverb input node (ConvolverNode with a generated stereo impulse), created on first use. */
  get reverb() {
    if (!this.ctx) return null;
    if (!this._reverb) {
      const ctx = this.ctx, sr = ctx.sampleRate, len = Math.max(1, Math.floor(sr * this.reverbSeconds));
      const ir = ctx.createBuffer(2, len, sr);
      for (let ch = 0; ch < 2; ch++) {
        const d = ir.getChannelData(ch);
        for (let i = 0; i < len; i++) {
          const t = i / len;
          d[i] = (Math.random() * 2 - 1) * Math.pow(1 - t, 2.6) * (i < sr * 0.004 ? i / (sr * 0.004) : 1);
        }
      }
      const conv = ctx.createConvolver();
      conv.buffer = ir;
      const ret = ctx.createGain(); ret.gain.value = 0.9;
      conv.connect(ret).connect(this.master);
      this._reverb = conv;
    }
    return this._reverb;
  }

  /** Route `node` into the reverb at `amount` (returns the send GainNode for later changes). */
  reverbSend(node, amount = 0.2) {
    const r = this.reverb;
    if (!r) return null;
    const g = this.ctx.createGain();
    g.gain.value = amount;
    node.connect(g).connect(r);
    return g;
  }

  suspend() { return this.ctx ? this.ctx.suspend() : Promise.resolve(); }
  resume() { return this.ctx ? this.ctx.resume() : Promise.resolve(); }

  dispose() {
    if (this._unlock) this._unlock();
    if (this.ctx && this.ctx.close) this.ctx.close();
    this.ctx = null; this.master = null; this.buses = {}; this._reverb = null;
  }
}
