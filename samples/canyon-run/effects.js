// Dust devils: every so often a whirlwind spins up somewhere ahead of the player on the canyon floor or a mesa,
// wanders with the wind for ~15-25 s and dies down. Drawn with one pooled ParticleSystem (normal blending):
// particles are spawned around a widening funnel with a tangential + upward velocity, which reads as rotation.
// The particles are sun-lit (ParticleSystem lit + the atmosphere), so they dim in shade, at dusk and at night.
import { ParticleSystem } from '@vibe/engine';

const MAX_DEVILS = 2;

export class DustDevils {
  constructor({ scene, terrain, track, atmosphere }) {
    this.terrain = terrain; this.track = track;
    this.ps = new ParticleSystem({ max: 1400, blending: 'normal', drag: 0.25, fadeIn: 0.5, spin: 0.4, name: 'dustDevils', lit: true, litAmbient: 0.12, sun: atmosphere });
    this.ps.mesh.renderOrder = 9;
    scene.add(this.ps.mesh);
    this.devils = [];
    for (let i = 0; i < MAX_DEVILS; i++) this.devils.push({ on: false, x: 0, z: 0, vx: 0, vz: 0, age: 0, life: 1, acc: 0, k: 1 });
    this.timer = 4;
    this.wind = { x: 2.2, z: 1.1 };
  }

  _spawn(d, px, pz, hint) {
    const tr = this.track, n = tr.n;
    // somewhere 120..380 m ahead along the course, off to one side of the track
    const i = (hint + 60 + ((Math.random() * 130) | 0)) % n;
    const side = (Math.random() < 0.5 ? -1 : 1) * (12 + Math.random() * 30);
    d.x = tr.x[i] - tr.tz[i] * side; d.z = tr.z[i] + tr.tx[i] * side;
    if ((d.x - px) ** 2 + (d.z - pz) ** 2 < 60 * 60) return false;
    d.on = true; d.age = 0; d.life = 15 + Math.random() * 12; d.acc = 0; d.k = 0.7 + Math.random() * 0.6;
    d.vx = this.wind.x * (0.6 + Math.random()); d.vz = this.wind.z * (0.6 + Math.random());
    return true;
  }

  /** Per frame. px/pz: player position, hint: player's nearest track index. */
  update(dt, px, pz, hint) {
    this.timer -= dt;
    if (this.timer <= 0) {
      this.timer = 9 + Math.random() * 14;
      for (let j = 0; j < this.devils.length; j++) if (!this.devils[j].on) { this._spawn(this.devils[j], px, pz, hint); break; }
    }
    const te = this.terrain, ps = this.ps;
    for (let j = 0; j < this.devils.length; j++) {
      const d = this.devils[j];
      if (!d.on) continue;
      d.age += dt;
      if (d.age > d.life) { d.on = false; continue; }
      d.x += d.vx * dt; d.z += d.vz * dt;
      d.vx += (Math.random() - 0.5) * 2 * dt; d.vz += (Math.random() - 0.5) * 2 * dt;
      const env = Math.min(1, d.age / 3, (d.life - d.age) / 4) * d.k;
      const g = te.heightAt(d.x, d.z);
      d.acc += 85 * env * dt;
      while (d.acc >= 1) {
        d.acc -= 1;
        const h = Math.pow(Math.random(), 1.6) * 26 * d.k, r = 0.8 + h * 0.22 + Math.random() * 0.6;
        const a = Math.random() * 6.2832, c = Math.cos(a), s = Math.sin(a);
        const swirl = 7 + h * 0.15, up = 2.2 + Math.random() * 2;
        const l = 0.9 + Math.random() * 0.15;
        ps.spawn(d.x + c * r, g + h, d.z + s * r,
          -s * swirl + d.vx + c * 0.6, up, c * swirl + d.vz + s * 0.6,
          1.6 + Math.random() * 1.6, 1.2 + h * 0.06, 3.5 + h * 0.18, 0.16 + 0.08 * (1 - h / 30),
          0.86 * l, 0.68 * l, 0.5 * l, 0.9 * l, 0.8 * l, 0.68 * l, 0.25, 0);
      }
    }
    ps.update(dt);
  }

  get active() { let n = 0; for (let j = 0; j < this.devils.length; j++) if (this.devils[j].on) n++; return n; }
}
