// Recorded audio: an `audio` asset type for core/Assets (-> AudioBuffer) and a SamplePlayer for one-shots, loops and
// positional sounds on the AudioBus mixer.
//
//   registerAudioLoader(assets, bus);                  // before assets.load(); bus.unlockOnGesture() as usual
//   assets.add('door', { type: 'audio', url: 'sfx/door.ogg' });
//   const sfx = new SamplePlayer(bus, { positional: space });          // bus 'effects'; space: the game's PositionalAudio
//   const v = sfx.play(assets.get('door'), { gain: 0.8, rate: 1.1 });   // -> voice
//   const hum = sfx.playAt(assets.get('hum'), machine, { loop: true, fadeIn: 0.5, refDistance: 4 });
//   hum.setGain(0.3, 0.2); hum.setRate(0.9); hum.stop(0.5); hum.playing;
//   sfx.stopAll(0.2); sfx.dispose();
//
// Decoding needs an AudioContext, which AudioBus creates in start() (a user gesture: autoplay policy). The loader
// downloads at once and hands its Assets concurrency slot back, then decodes with the bus context when it exists
// (suspended is fine) and otherwise, by default, in an OfflineAudioContext at `sampleRate` (AudioBuffers play in any
// context, resampled when the rates differ), so loading never waits for a click. `{ offline: false }` (or no
// OfflineAudioContext) waits for the bus to start instead (bus.onStart): until the first gesture assets.load() then
// waits on the audio keys, and so does Engine.load with every system registered after the assets system.
// Voices: { source (AudioBufferSourceNode), gain (GainNode), playing, stop(fade = 0), setRate(r), setGain(g, t = 0) };
// source -> gain -> the bus (`bus` option per play, else the player's, default 'effects' like PositionalAudio) or,
// for playAt, -> a PositionalAudio emitter
// (Positional.js; removed when the voice ends). Before the bus has started (or without a buffer) play / playAt return
// an inert voice (playing false, methods no-ops), so game code never needs a null check. One voice + nodes per play
// (not per frame); SamplePlayer.update() (per frame, no allocation) updates its own PositionalAudio when it made one.
// Safe to import in Node (no AudioContext until a sound loads or plays).
import * as THREE from 'three';
import { PositionalAudio } from './Positional.js';

const _v = new THREE.Vector3();
const noop = function () { return this; };
/** The voice returned while there is nothing to play (no context yet / no buffer). */
const INERT = Object.freeze({ source: null, gain: null, playing: false, stop: noop, setRate: noop, setGain: noop });

const decode = (ctx, buf) => new Promise((res, rej) => ctx.decodeAudioData(buf, res, rej)); // callback form: old Safari

/**
 * Register the `audio` type: fetch with assets.fetch, decode with the bus's AudioContext once it exists, before that
 * in an OfflineAudioContext (see header). Hazard: with `offline: false` (or where OfflineAudioContext is missing) an
 * audio key cannot finish before the user's first gesture starts the bus, so assets.load() and Engine.load (every
 * system after this one, systems load in order) wait for that click / key press.
 * @param {import('../core/Assets.js').Assets} assets
 * @param {import('./AudioBus.js').AudioBus} bus
 * @param {object} [o]
 * @param {boolean} [o.offline=true]  decode before the bus starts, in an OfflineAudioContext (false: wait for start)
 * @param {number} [o.sampleRate=48000]  sample rate of that offline context (the bus's own rate is known only once it
 *   started; 48 kHz is the common hardware rate, other rates are resampled at playback)
 */
export function registerAudioLoader(assets, bus, { offline = true, sampleRate = 48000 } = {}) {
  let off = null;
  const context = () => {
    if (bus.ctx) return bus.ctx;
    const OAC = offline && (globalThis.OfflineAudioContext || globalThis.webkitOfflineAudioContext);
    if (OAC) return off || (off = new OAC(1, 1, sampleRate));
    return new Promise((resolve) => bus.onStart((ctx) => resolve(ctx))); // no offline decoding: wait for the gesture
  };
  assets.register('audio', async (d, a, freeSlot) => {
    const buf = await (await a.fetchOk(d.url, d.init)).arrayBuffer();
    if (freeSlot) freeSlot(); // what is left waits on the context / decoder, not the network
    return decode(await context(), buf);
  });
  return assets;
}

class SampleVoice {
  constructor(player, source, gain, emitter) {
    this.player = player; this.source = source; this.gain = gain; this.emitter = emitter;
    this.playing = true;
    source.onended = () => this._end();
  }
  /** Stop now, or fade out over `fade` seconds and stop. */
  stop(fade = 0) {
    if (!this.playing) return this;
    const t = this.source.context.currentTime;
    if (fade > 0) {
      const g = this.gain.gain;
      g.cancelScheduledValues(t); g.setValueAtTime(g.value, t); g.linearRampToValueAtTime(0, t + fade);
      try { this.source.stop(t + fade); } catch { /* not started */ }
    } else {
      this.playing = false;
      try { this.source.stop(); } catch { /* not started */ }
    }
    return this;
  }
  /** Playback rate (1 = original speed and pitch). */
  setRate(r) { const p = this.source.playbackRate; p.setValueAtTime(r, this.source.context.currentTime); return this; }
  /** Gain, at once or ramped linearly over `t` seconds. */
  setGain(g, t = 0) {
    const p = this.gain.gain, now = this.source.context.currentTime;
    p.cancelScheduledValues(now);
    if (t > 0) { p.setValueAtTime(p.value, now); p.linearRampToValueAtTime(g, now + t); } else p.setValueAtTime(g, now);
    return this;
  }
  _end() {
    if (!this.player) return;
    this.playing = false;
    this.source.onended = null;
    try { this.source.disconnect(); this.gain.disconnect(); } catch { /* gone */ }
    if (this.emitter && this.player.positional) this.player.positional.remove(this.emitter);
    this.player.voices.delete(this);
    this.player = null;
  }
}

export class SamplePlayer {
  /**
   * @param {import('./AudioBus.js').AudioBus} bus
   * @param {object} [o]
   * @param {string} [o.bus='effects']  default mixer bus (AudioBus DEFAULT_BUSES; PositionalAudio's default too)
   * @param {PositionalAudio} [o.positional]  shared emitter space for playAt (else one is made on first use)
   * @param {THREE.Object3D} [o.listener]  listener (camera) of that own space; call update() per frame then
   */
  constructor(bus, { bus: busName = 'effects', positional = null, listener = null } = {}) {
    this.bus = bus;
    this.busName = busName;
    this.positional = positional;
    this.listener = listener;
    this._ownSpace = false;
    /** Voices still playing. */
    this.voices = new Set();
  }

  /**
   * Play `buffer` -> voice (see header).
   * @param {AudioBuffer} buffer
   * @param {{gain?: number, rate?: number, loop?: boolean, offset?: number, fadeIn?: number, bus?: string}} [o]
   */
  play(buffer, o = {}) { return this._play(buffer, o, null); }

  /**
   * Play `buffer` from `object3d` through a PositionalAudio emitter (follows the object while it plays). `o` also
   * takes the emitter options: refDistance, maxDistance, rolloff, distanceModel, panningModel, cone.
   */
  playAt(buffer, object3d, o = {}) {
    if (!this.bus.ctx || !buffer) return INERT;
    if (!this.positional) { this.positional = new PositionalAudio(this.bus, { listener: this.listener }); this._ownSpace = true; }
    const e = this.positional.add(object3d, {
      bus: o.bus || this.busName, refDistance: o.refDistance, maxDistance: o.maxDistance, rolloff: o.rolloff,
      distanceModel: o.distanceModel, panningModel: o.panningModel, cone: o.cone,
    });
    // start at the object, not at the origin: Emitter.update() glides toward its target (setTargetAtTime)
    object3d.updateWorldMatrix(true, false);
    _v.setFromMatrixPosition(object3d.matrixWorld);
    const p = e.panner;
    if (p.positionX) { p.positionX.value = _v.x; p.positionY.value = _v.y; p.positionZ.value = _v.z; } else p.setPosition(_v.x, _v.y, _v.z);
    e.update();
    return this._play(buffer, o, e);
  }

  _play(buffer, { gain = 1, rate = 1, loop = false, offset = 0, fadeIn = 0, bus } = {}, emitter) {
    const ctx = this.bus.ctx;
    if (!ctx || !buffer) return INERT;
    const src = ctx.createBufferSource(), g = ctx.createGain(), t = ctx.currentTime;
    src.buffer = buffer;
    src.loop = !!loop;
    src.playbackRate.value = rate;
    if (fadeIn > 0) { g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(gain, t + fadeIn); } else g.gain.value = gain;
    src.connect(g).connect(emitter ? emitter.input : this.bus.bus(bus || this.busName));
    const v = new SampleVoice(this, src, g, emitter);
    this.voices.add(v);
    src.start(t, Math.max(0, offset));
    return v;
  }

  /** Per frame (no allocation): updates the PositionalAudio this player made itself (a shared one is the caller's). */
  update() { if (this._ownSpace) this.positional.update(); }

  /** Stop every playing voice (fading over `fade` seconds). */
  stopAll(fade = 0) { for (const v of this.voices) v.stop(fade); }

  dispose() {
    for (const v of [...this.voices]) { v.stop(0); v._end(); }
    if (this._ownSpace) { this.positional.dispose(); this.positional = null; this._ownSpace = false; }
  }
}
