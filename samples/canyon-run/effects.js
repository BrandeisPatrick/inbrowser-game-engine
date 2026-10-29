// Dust devils: every so often a whirlwind spins up somewhere ahead of the player on the canyon floor or a mesa,
// wanders with the wind for ~15-25 s and dies down. Drawn with one pooled particle system (normal blending):
// particles are spawned around a widening funnel with a tangential + upward velocity, which reads as rotation.
// The particles are sun-lit (lit + the atmosphere), so they dim in shade, at dusk and at night.
// Systems: the CPU ParticleSystem by default; `gpu: true` (main.js `?gpuparticles=1`) runs the same spawn calls on
// the engine's GpuParticleSystem (GPGPU ping-pong textures, deterministic slots; falls back to the CPU system with one
// warning where WebGL2 float render targets are missing), `collide: true` (`?devilscollide=1`) adds its depth-buffer
// collision (main.js hands it PostFX's scene depth). Not switchable at runtime: the system is chosen at construction.
// Randomness comes from a seeded mulberry32 (seed option, default 7), sprite rotations included (drawn here and
// written into the CPU system's slot / passed to the GPU spawn), so one seed gives one spawn sequence on either
// system; poseDevil() (bench harness) restarts it and steps one devil at a fixed spot with a fixed dt.
import { ParticleSystem, GpuParticleSystem, mulberry32 } from '@vibe/engine';

const MAX_DEVILS = 2;

export class DustDevils {
  /**
   * @param {object} o
   * @param {import('three').Object3D} o.scene
   * @param {import('three').WebGLRenderer} [o.renderer]  needed for gpu: true
   * @param {object} o.terrain  Terrain (heightAt)
   * @param {object} o.track  Track (n, x, z, tx, tz)
   * @param {object} [o.atmosphere]  lights the dust (sun factor)
   * @param {boolean} [o.gpu=false]  GpuParticleSystem (when supported) instead of the CPU ParticleSystem
   * @param {boolean} [o.collide=false]  gpu only: depth-buffer collision (ps.setDepth from main.js)
   * @param {number} [o.seed=7]
   */
  constructor({ scene, renderer = null, terrain, track, atmosphere, gpu = false, collide = false, seed = 7 }) {
    this.terrain = terrain; this.track = track;
    this.seed = seed;
    this.rand = mulberry32(seed);
    let ps = null;
    if (gpu) {
      if (GpuParticleSystem.isSupported(renderer)) {
        try {
          ps = new GpuParticleSystem(renderer, { max: 16384, blending: 'normal', drag: 0.25, fadeIn: 0.5, spin: 0.4, name: 'dustDevils', lit: true, litAmbient: 0.12, sun: atmosphere, depthCollision: !!collide });
        } catch (e) { console.warn('[canyon-run] GPU dust devils failed to start, using the CPU particle system:', e.message); }
      } else console.warn('[canyon-run] GPU particles unsupported here (WebGL2 + EXT_color_buffer_float): the dust devils stay on the CPU particle system');
    }
    /** True when the devils run on the GpuParticleSystem. */
    this.gpu = !!ps;
    this.ps = ps || new ParticleSystem({ max: 1400, blending: 'normal', drag: 0.25, fadeIn: 0.5, spin: 0.4, name: 'dustDevils', lit: true, litAmbient: 0.12, sun: atmosphere });
    this.ps.mesh.renderOrder = 9;
    scene.add(this.ps.mesh);
    this.devils = [];
    for (let i = 0; i < MAX_DEVILS; i++) this.devils.push({ on: false, x: 0, z: 0, vx: 0, vz: 0, age: 0, life: 1, acc: 0, k: 1 });
    this.timer = 4;
    this.wind = { x: 2.2, z: 1.1 };
  }

  /** Devil centre at track point i, `side` m right of travel (negative: left). */
  _at(d, i, side) {
    const tr = this.track;
    d.x = tr.x[i] - tr.tz[i] * side; d.z = tr.z[i] + tr.tx[i] * side;
  }

  /** Life, strength and drift (with the wind) of a devil that starts now. */
  _start(d) {
    const r = this.rand;
    d.on = true; d.age = 0; d.life = 15 + r() * 12; d.acc = 0; d.k = 0.7 + r() * 0.6;
    d.vx = this.wind.x * (0.6 + r()); d.vz = this.wind.z * (0.6 + r());
  }

  _spawn(d, px, pz, hint) {
    const r = this.rand, n = this.track.n;
    // somewhere 120..380 m ahead along the course, off to one side of the track
    const i = (hint + 60 + ((r() * 130) | 0)) % n;
    const side = (r() < 0.5 ? -1 : 1) * (12 + r() * 30);
    this._at(d, i, side);
    if ((d.x - px) ** 2 + (d.z - pz) ** 2 < 60 * 60) return false;
    this._start(d);
    return true;
  }

  /**
   * Per frame. px/pz: player position, hint: player's nearest track index, camera: the view (the GPU system's
   * update needs it; ignored by the CPU system).
   */
  update(dt, px, pz, hint, camera = null) {
    this.timer -= dt;
    if (this.timer <= 0) {
      this.timer = 9 + this.rand() * 14;
      for (let j = 0; j < this.devils.length; j++) if (!this.devils[j].on) { this._spawn(this.devils[j], px, pz, hint); break; }
    }
    this._step(dt, camera);
  }

  /** Move the live devils, spawn their particles, step the particle system. Allocation-free. */
  _step(dt, camera) {
    const te = this.terrain, ps = this.ps, r = this.rand, cpu = !this.gpu, spin = ps.spin;
    for (let j = 0; j < this.devils.length; j++) {
      const d = this.devils[j];
      if (!d.on) continue;
      d.age += dt;
      if (d.age > d.life) { d.on = false; continue; }
      d.x += d.vx * dt; d.z += d.vz * dt;
      d.vx += (r() - 0.5) * 2 * dt; d.vz += (r() - 0.5) * 2 * dt;
      const env = Math.min(1, d.age / 3, (d.life - d.age) / 4) * d.k;
      const g = te.heightAt(d.x, d.z);
      d.acc += 85 * env * dt;
      while (d.acc >= 1) {
        d.acc -= 1;
        const h = Math.pow(r(), 1.6) * 26 * d.k, rad = 0.8 + h * 0.22 + r() * 0.6;
        const a = r() * 6.2832, c = Math.cos(a), s = Math.sin(a);
        const swirl = 7 + h * 0.15, up = 2.2 + r() * 2;
        const l = 0.9 + r() * 0.15, life = 1.6 + r() * 1.6;
        // sprite rotation from the seeded sequence too (ParticleSystem.spawn would draw it from Math.random)
        const rot = r() * 6.283, rotV = (r() * 2 - 1) * spin;
        const slot = ps.spawn(d.x + c * rad, g + h, d.z + s * rad,
          -s * swirl + d.vx + c * 0.6, up, c * swirl + d.vz + s * 0.6,
          life, 1.2 + h * 0.06, 3.5 + h * 0.18, 0.16 + 0.08 * (1 - h / 30),
          0.86 * l, 0.68 * l, 0.5 * l, 0.9 * l, 0.8 * l, 0.68 * l, 0.25, 0, rot, rotV);
        if (cpu && slot >= 0) { ps.rot[slot] = rot; ps.rotV[slot] = rotV; }
      }
    }
    ps.update(dt, camera);
  }

  /**
   * Kill every devil and particle, restart the seeded sequence and stop the live spawning (timer parked), e.g. for
   * the bench harness. The next live devil comes only after `timer` is set again.
   * @param {number} [seed]  default: the constructor's
   */
  reset(seed = this.seed) {
    this.ps.clear();
    this.rand = mulberry32(seed);
    for (let j = 0; j < this.devils.length; j++) this.devils[j].on = false;
    this.timer = 1e9;
  }

  /**
   * Bench pose: reset(), start one devil at track point `index`, `side` m right of travel (with the fixed wind and
   * the seeded life / strength / drift), then run `steps` updates of `dt`, all with `camera` (the GPU system's update
   * takes it). Same spawn sequence on the CPU and the GPU system; nothing moves afterwards until the next update().
   * @param {number} index  track point
   * @param {number} side  m, + = right of travel
   * @param {number} [steps=180]
   * @param {import('three').Camera|null} [camera]
   * @param {number} [dt=1/60]
   * @returns {object} the posed devil ({ x, z, vx, vz, age, life, k, ... })
   */
  poseDevil(index, side, steps = 180, camera = null, dt = 1 / 60) {
    this.reset();
    const n = this.track.n, d = this.devils[0];
    this._at(d, ((Math.round(index) % n) + n) % n, side);
    this._start(d);
    for (let s = 0; s < steps; s++) this._step(dt, camera);
    return d;
  }

  get active() { let n = 0; for (let j = 0; j < this.devils.length; j++) if (this.devils[j].on) n++; return n; }
}
