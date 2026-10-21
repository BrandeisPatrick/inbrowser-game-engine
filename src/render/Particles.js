// Pooled sprite particles: one instanced draw per system, simple physics (gravity, exponential drag, a ground
// plane or terrain callback with bounce), lifetimes with size / colour / alpha curves (fade in, quadratic fade
// out), additive or alpha blending, optional velocity-stretched streaks (sparks), optional sun-lit brightness
// (unlit sprites that dim in shade / at night). Fixed capacity, swap-remove on death, partial GPU uploads;
// spawn / emit / update are allocation-free.
//
//   const sparks = new ParticleSystem({ max: 512, blending: 'additive', gravity: 9.8, streak: 0.04 });
//   scene.add(sparks.mesh);
//   sparks.spawn(x, y, z, vx, vy, vz, life, size0, size1, alpha, r, g, b);
//   const smoke = new ParticleEmitter(system, { rate: 40, position, velocity, spread: 1.5, life: [1, 2] });
//   per frame: smoke.update(dt); system.update(dt);
//   const dust = new ParticleSystem({ blending: 'normal', floorAt: (x, z) => terrain.heightAt(x, z), lit: true, sun: atmosphere });
import * as THREE from 'three';
import { touchRange, newRange } from './bufferRanges.js';

/**
 * 0..1 direct-sun factor from an Atmosphere-like object ({ sun: Light, sunIntensity, darkness }): the sun
 * light's share of its full intensity by day, 0 once the moon takes over (darkness > 0.5).
 */
export function atmosphereSunFactor(a) {
  if (!a) return 1;
  const d = a.darkness || 0;
  if (d > 0.5) return 0;
  const s = a.sun && a.sunIntensity > 0 ? a.sun.intensity / a.sunIntensity : 1 - d;
  return (s < 0 ? 0 : s > 1 ? 1 : s) * (1 - d);
}

const VS = /* glsl */`
attribute vec3 iPos;
attribute vec3 iVel;
attribute vec4 iData;   // size, alpha, rotation, -
attribute vec3 iColor;
uniform float uStreak;
uniform float uLight;
varying vec2 vUv;
varying float vAlpha;
varying vec3 vColor;
#include <fog_pars_vertex>
void main() {
	vUv = uv;
	vec4 mvPosition = modelViewMatrix * vec4( iPos, 1.0 );
	vec2 q = position.xy;
	float size = iData.x;
	if ( uStreak > 0.0 ) {
		vec2 d = ( modelViewMatrix * vec4( iVel, 0.0 ) ).xy;
		float l = length( d );
		vec2 dir = l > 1e-4 ? d / l : vec2( 1.0, 0.0 );
		vec2 nrm = vec2( - dir.y, dir.x );
		float len = size + l * uStreak;
		mvPosition.xy += dir * ( q.x - 0.5 ) * len + nrm * q.y * size;
	} else {
		float c = cos( iData.z ), s = sin( iData.z );
		mvPosition.xy += vec2( c * q.x - s * q.y, s * q.x + c * q.y ) * size;
	}
	vAlpha = iData.y;
	vColor = iColor * uLight;
	gl_Position = projectionMatrix * mvPosition;
	#include <fog_vertex>
}`;
const FS = /* glsl */`
uniform sampler2D uMap;
varying vec2 vUv;
varying float vAlpha;
varying vec3 vColor;
#include <fog_pars_fragment>
void main() {
	vec4 t = texture2D( uMap, vUv );
	gl_FragColor = vec4( vColor * t.rgb, t.a * vAlpha );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
}`;

let _dot = null;
/** Shared soft round sprite (64 px canvas). */
export function softDotTexture() {
  if (_dot) return _dot;
  const c = document.createElement('canvas'); c.width = c.height = 64;
  const g = c.getContext('2d'), gr = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  gr.addColorStop(0, 'rgba(255,255,255,1)'); gr.addColorStop(0.4, 'rgba(255,255,255,0.45)'); gr.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = gr; g.fillRect(0, 0, 64, 64);
  _dot = new THREE.CanvasTexture(c); _dot.colorSpace = THREE.SRGBColorSpace;
  return _dot;
}

export class ParticleSystem {
  /**
   * @param {object} [o]
   * @param {number} [o.max=1024]
   * @param {THREE.Texture} [o.map]          sprite (default: soft dot)
   * @param {'additive'|'normal'} [o.blending='additive']
   * @param {number} [o.gravity=0]           m/s^2 downward (per-particle override via spawn args)
   * @param {number} [o.drag=0]              1/s exponential velocity damping
   * @param {number} [o.floor=-Infinity]     ground height: particles bounce (o.bounce) / settle there
   * @param {Function} [o.floorAt]           (x, z) => ground height: terrain collision per particle (overrides floor;
   *                                         called once per live particle per update)
   * @param {number} [o.bounce=0.35]
   * @param {number} [o.fadeIn=0.12]         s
   * @param {number} [o.streak=0]            > 0 stretches sprites along their screen velocity (s)
   * @param {number} [o.light=1]             colour multiplier (HDR glow for additive sparks)
   * @param {boolean} [o.lit=false]          scale the brightness by a sun factor: light * (litAmbient + (1 - litAmbient) * sun)
   * @param {number} [o.litAmbient=0.15]     brightness share that remains without direct sun
   * @param {number|Function|object} [o.sun] sun factor source while lit: a number 0..1 (also settable as
   *                                         `sunFactor`), () => 0..1, or an Atmosphere (atmosphereSunFactor)
   * @param {number} [o.spin=1.2]            max random rotation speed (rad/s)
   */
  constructor(o = {}) {
    const max = this.max = o.max ?? 1024;
    this.count = 0;
    this.gravity = o.gravity ?? 0; this.drag = o.drag ?? 0; this.floor = o.floor ?? -Infinity;
    this.floorAt = typeof o.floorAt === 'function' ? o.floorAt : null;
    this.bounce = o.bounce ?? 0.35; this.fadeInTime = o.fadeIn ?? 0.12; this.spin = o.spin ?? 1.2;
    this.lit = !!o.lit; this.litAmbient = o.litAmbient ?? 0.15;
    this.sunSource = typeof o.sun === 'number' ? null : o.sun || null;
    this.sunFactor = typeof o.sun === 'number' ? o.sun : 1;
    this._light = o.light ?? 1;
    const f = (n) => new Float32Array(max * n);
    this.p = f(3); this.v = f(3); this.age = f(1); this.life = f(1); this.s0 = f(1); this.s1 = f(1);
    this.a0 = f(1); this.rot = f(1); this.rotV = f(1); this.c = f(3); this.c1 = f(3); this.dragA = f(1); this.grav = f(1);
    const quad = new THREE.InstancedBufferGeometry();
    quad.setAttribute('position', new THREE.Float32BufferAttribute([-0.5, -0.5, 0, 0.5, -0.5, 0, 0.5, 0.5, 0, -0.5, 0.5, 0], 3));
    quad.setAttribute('uv', new THREE.Float32BufferAttribute([0, 0, 1, 0, 1, 1, 0, 1], 2));
    quad.setIndex([0, 1, 2, 0, 2, 3]);
    this.aPos = new THREE.InstancedBufferAttribute(f(3), 3).setUsage(THREE.DynamicDrawUsage);
    this.aVel = new THREE.InstancedBufferAttribute(f(3), 3).setUsage(THREE.DynamicDrawUsage);
    this.aData = new THREE.InstancedBufferAttribute(f(4), 4).setUsage(THREE.DynamicDrawUsage);
    this.aColor = new THREE.InstancedBufferAttribute(f(3), 3).setUsage(THREE.DynamicDrawUsage);
    quad.setAttribute('iPos', this.aPos); quad.setAttribute('iVel', this.aVel);
    quad.setAttribute('iData', this.aData); quad.setAttribute('iColor', this.aColor);
    quad.instanceCount = 0;
    this.geometry = quad;
    this.material = new THREE.ShaderMaterial({
      uniforms: THREE.UniformsUtils.merge([THREE.UniformsLib.fog, { uMap: { value: null }, uStreak: { value: o.streak ?? 0 }, uLight: { value: o.light ?? 1 } }]),
      vertexShader: VS, fragmentShader: FS,
      transparent: true, depthWrite: false, fog: true,
      blending: (o.blending ?? 'additive') === 'additive' ? THREE.AdditiveBlending : THREE.NormalBlending,
    });
    this.material.uniforms.uMap.value = o.map || softDotTexture();
    this.mesh = new THREE.Mesh(quad, this.material);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = o.renderOrder ?? 8;
    this.mesh.name = o.name || 'particles';
    this.mesh.visible = false;
    this._r3 = [newRange(), newRange(), newRange()]; this._r4 = newRange();
  }

  /**
   * Spawn one particle; returns its slot or -1 when full. Colour lerps (r,g,b) -> (r1,g1,b1) over the life;
   * size s0 -> s1 (sqrt curve); alpha a0 with fade-in and quadratic fade-out. drag / grav default to the system's.
   */
  spawn(x, y, z, vx, vy, vz, life, s0, s1, a0, r, g, b, r1 = r, g1 = g, b1 = b, drag = this.drag, grav = this.gravity) {
    if (this.count >= this.max) return -1;
    const i = this.count++, i3 = i * 3;
    this.p[i3] = x; this.p[i3 + 1] = y; this.p[i3 + 2] = z;
    this.v[i3] = vx; this.v[i3 + 1] = vy; this.v[i3 + 2] = vz;
    this.age[i] = 0; this.life[i] = Math.max(1e-3, life); this.s0[i] = s0; this.s1[i] = s1; this.a0[i] = a0;
    this.rot[i] = Math.random() * 6.283; this.rotV[i] = (Math.random() * 2 - 1) * this.spin;
    this.c[i3] = r; this.c[i3 + 1] = g; this.c[i3 + 2] = b;
    this.c1[i3] = r1; this.c1[i3 + 1] = g1; this.c1[i3 + 2] = b1;
    this.dragA[i] = drag; this.grav[i] = grav;
    return i;
  }

  /** Spawn from a (reusable) params object: { x, y, z, vx, vy, vz, life, size0, size1, alpha, r, g, b, r1?, g1?, b1? }. */
  emit(p) {
    return this.spawn(p.x, p.y, p.z, p.vx, p.vy, p.vz, p.life, p.size0, p.size1 ?? p.size0, p.alpha ?? 1, p.r ?? 1, p.g ?? 1, p.b ?? 1,
      p.r1 ?? p.r ?? 1, p.g1 ?? p.g ?? 1, p.b1 ?? p.b ?? 1, p.drag ?? this.drag, p.gravity ?? this.gravity);
  }

  _kill(i) {
    const j = --this.count;
    if (i === j) return;
    const i3 = i * 3, j3 = j * 3;
    for (let k = 0; k < 3; k++) { this.p[i3 + k] = this.p[j3 + k]; this.v[i3 + k] = this.v[j3 + k]; this.c[i3 + k] = this.c[j3 + k]; this.c1[i3 + k] = this.c1[j3 + k]; }
    this.age[i] = this.age[j]; this.life[i] = this.life[j]; this.s0[i] = this.s0[j]; this.s1[i] = this.s1[j];
    this.a0[i] = this.a0[j]; this.rot[i] = this.rot[j]; this.rotV[i] = this.rotV[j]; this.dragA[i] = this.dragA[j]; this.grav[i] = this.grav[j];
  }

  update(dt) {
    const P = this.p, V = this.v, fl0 = this.floor, bo = this.bounce, floorAt = this.floorAt;
    for (let i = 0; i < this.count; i++) {
      this.age[i] += dt;
      if (this.age[i] >= this.life[i]) { this._kill(i); i--; continue; }
      const i3 = i * 3, dr = Math.exp(-this.dragA[i] * dt);
      V[i3] *= dr; V[i3 + 1] = V[i3 + 1] * dr - this.grav[i] * dt; V[i3 + 2] *= dr;
      P[i3] += V[i3] * dt; P[i3 + 1] += V[i3 + 1] * dt; P[i3 + 2] += V[i3 + 2] * dt;
      const fl = floorAt ? floorAt(P[i3], P[i3 + 2]) : fl0;
      if (P[i3 + 1] < fl) { P[i3 + 1] = fl; V[i3 + 1] = Math.abs(V[i3 + 1]) * bo; V[i3] *= 0.6; V[i3 + 2] *= 0.6; }
      this.rot[i] += this.rotV[i] * dt;
    }
    const n = this.count, AP = this.aPos.array, AV = this.aVel.array, D = this.aData.array, C = this.aColor.array;
    for (let i = 0; i < n; i++) {
      const i3 = i * 3, i4 = i * 4, t = this.age[i] / this.life[i];
      AP[i3] = P[i3]; AP[i3 + 1] = P[i3 + 1]; AP[i3 + 2] = P[i3 + 2];
      AV[i3] = V[i3]; AV[i3 + 1] = V[i3 + 1]; AV[i3 + 2] = V[i3 + 2];
      const fin = this.fadeInTime > 0 ? Math.min(1, this.age[i] / this.fadeInTime) : 1;
      D[i4] = this.s0[i] + (this.s1[i] - this.s0[i]) * Math.sqrt(t);
      D[i4 + 1] = this.a0[i] * fin * (1 - t) * (1 - t);
      D[i4 + 2] = this.rot[i];
      for (let k = 0; k < 3; k++) C[i3 + k] = this.c[i3 + k] + (this.c1[i3 + k] - this.c[i3 + k]) * t;
    }
    const m = Math.max(1, n);
    touchRange(this.aPos, this._r3[0], 0, m * 3); touchRange(this.aVel, this._r3[1], 0, m * 3); touchRange(this.aColor, this._r3[2], 0, m * 3);
    touchRange(this.aData, this._r4, 0, m * 4);
    this.geometry.instanceCount = n;
    this.mesh.visible = n > 0;
    if (this.lit) this._applyLight();
  }

  /** Current sun factor (0..1) of a lit system: from `sun` (function / Atmosphere) or `sunFactor`. */
  get sunLevel() {
    const src = this.sunSource;
    const s = typeof src === 'function' ? src() : src ? atmosphereSunFactor(src) : this.sunFactor;
    return s < 0 ? 0 : s > 1 ? 1 : s || 0;
  }
  _applyLight() {
    const k = this.lit ? this.litAmbient + (1 - this.litAmbient) * this.sunLevel : 1;
    this.material.uniforms.uLight.value = this._light * k;
  }

  /** Colour multiplier (e.g. dim unlit smoke at night: 1 - 0.85 * atmosphere.darkness); lit systems scale it by the sun. */
  get light() { return this._light; }
  set light(v) { this._light = v; this._applyLight(); }

  clear() { this.count = 0; this.geometry.instanceCount = 0; this.mesh.visible = false; }
  dispose() { this.geometry.dispose(); this.material.dispose(); }
}

/**
 * Continuous emitter feeding a ParticleSystem: `rate` particles/s around `position` with `velocity` plus a
 * random `spread` (m/s, sphere) and `jitter` (m). Mutate position / velocity / rate per frame; allocation-free.
 */
export class ParticleEmitter {
  constructor(system, o = {}) {
    this.system = system;
    this.rate = o.rate ?? 30;
    this.position = (o.position || new THREE.Vector3()).clone();
    this.velocity = (o.velocity || new THREE.Vector3(0, 1, 0)).clone();
    this.spread = o.spread ?? 1;
    this.jitter = o.jitter ?? 0;
    this.life = o.life || [1, 2];
    this.size = o.size || [0.5, 1.5];
    this.alpha = o.alpha ?? 1;
    this.color = o.color || [1, 1, 1];
    this.colorEnd = o.colorEnd || this.color;
    this.drag = o.drag; this.gravity = o.gravity;
    this.enabled = o.enabled !== false;
    this._acc = 0;
  }
  update(dt) {
    if (!this.enabled) return 0;
    this._acc += this.rate * dt;
    let n = 0;
    const s = this.system, p = this.position, v = this.velocity, c = this.color, c1 = this.colorEnd;
    const drag = this.drag ?? s.drag, grav = this.gravity ?? s.gravity;
    while (this._acc >= 1) {
      this._acc -= 1;
      // uniform direction in a sphere scaled by spread
      let ux, uy, uz, l2;
      do { ux = Math.random() * 2 - 1; uy = Math.random() * 2 - 1; uz = Math.random() * 2 - 1; l2 = ux * ux + uy * uy + uz * uz; } while (l2 > 1 || l2 < 1e-6);
      const j = this.jitter, sp = this.spread;
      const life = this.life[0] + Math.random() * (this.life[1] - this.life[0]);
      if (s.spawn(p.x + (Math.random() - 0.5) * 2 * j, p.y + (Math.random() - 0.5) * 2 * j, p.z + (Math.random() - 0.5) * 2 * j,
        v.x + ux * sp, v.y + uy * sp, v.z + uz * sp, life, this.size[0], this.size[1], this.alpha,
        c[0], c[1], c[2], c1[0], c1[1], c1[2], drag, grav) < 0) { this._acc = 0; break; }
      n++;
    }
    return n;
  }
}
