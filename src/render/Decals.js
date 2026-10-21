// DecalTrail: capped ribbon decals laid on surfaces — skid marks, tyre tracks, scorch trails, footprints.
// A ring buffer of quads in one draw call: the newest segment overwrites the oldest, and the oldest
// `fadeTail` fraction fades out before it is reused, so the cap is never visible. Several independent trails
// (e.g. one per wheel) share the buffer. Partial GPU uploads; feed() is allocation-free.
//
//   const skids = new DecalTrail({ maxQuads: 1600, trails: 4, color: 0x030303, opacity: 0.78 });
//   scene.add(skids.mesh);
//   per frame, per wheel: skids.feed(i, contactPoint, wheelAxis /* unit */, 0.12, slipIntensity /* 0 ends */);
//   on terrain: new DecalTrail({ lift: 0.08 }) and feed(i, surfacePoint, axis, w, k, surfaceNormal?) — the ribbon
//   is raised `lift` metres off the surface along the normal (world up when no normal is given).
import * as THREE from 'three';
import { touchRange, newRange } from './bufferRanges.js';

const VS = /* glsl */`
attribute float aAlpha;
attribute float aId;
uniform float uHead;
uniform float uCap;
uniform float uFadeTail;
varying float vAlpha;
varying vec2 vUv;
#include <fog_pars_vertex>
void main() {
	float age = ( uHead - aId ) / uCap;
	vAlpha = aAlpha * ( 1.0 - smoothstep( 1.0 - uFadeTail, 1.0, age ) );
	vUv = uv;
	vec4 mvPosition = modelViewMatrix * vec4( position, 1.0 );
	gl_Position = projectionMatrix * mvPosition;
	#include <fog_vertex>
}`;
const FS = /* glsl */`
uniform vec3 uColor;
uniform float uEdge;
uniform float uTread;
uniform sampler2D uMap;
uniform float uHasMap;
varying float vAlpha;
varying vec2 vUv;
#include <fog_pars_fragment>
void main() {
	float edge = smoothstep( 0.0, uEdge, vUv.y ) * smoothstep( 1.0, 1.0 - uEdge, vUv.y );
	float tread = mix( 1.0, 0.8 + 0.2 * step( 0.5, fract( vUv.y * 4.0 ) ), uTread );
	vec4 t = uHasMap > 0.5 ? texture2D( uMap, vUv ) : vec4( 1.0 );
	gl_FragColor = vec4( uColor * t.rgb, vAlpha * edge * tread * t.a );
	#include <colorspace_fragment>
	#include <fog_fragment>
}`;

export class DecalTrail {
  /**
   * @param {object} [o]
   * @param {number} [o.maxQuads=1600]   ring capacity (segments)
   * @param {number} [o.trails=4]        independent trails feeding the ring
   * @param {number|THREE.Color} [o.color=0x030303]
   * @param {number} [o.opacity=0.78]    alpha at intensity 1
   * @param {number} [o.fadeTail=0.3]    oldest fraction of the ring that fades out
   * @param {number} [o.minSegment=0.2]  metres between segments
   * @param {number} [o.maxSegment=2]    a longer gap restarts the trail (teleport / reset)
   * @param {number} [o.edge=0.18]       soft edge across the ribbon (uv fraction)
   * @param {boolean} [o.tread=true]     subtle tyre-tread banding
   * @param {THREE.Texture} [o.map]      optional texture (u along the segment, v across)
   * @param {number} [o.lift=0]          metres the ribbon is raised off the fed surface point (along the feed normal,
   *                                     world up by default): avoids z-fighting on coarse terrain
   */
  constructor(o = {}) {
    const max = this.max = o.maxQuads ?? 1600;
    this.head = 0;
    this.lift = o.lift ?? 0;
    this.opacity = o.opacity ?? 0.78;
    this.minSegment = o.minSegment ?? 0.2;
    this.maxSegment = o.maxSegment ?? 2;
    const nV = max * 4;
    this.pos = new THREE.BufferAttribute(new Float32Array(nV * 3), 3).setUsage(THREE.DynamicDrawUsage);
    this.alpha = new THREE.BufferAttribute(new Float32Array(nV), 1).setUsage(THREE.DynamicDrawUsage);
    this.id = new THREE.BufferAttribute(new Float32Array(nV).fill(-1e9), 1).setUsage(THREE.DynamicDrawUsage);
    const uv = new Float32Array(nV * 2), idx = new Uint32Array(max * 6);
    for (let q = 0; q < max; q++) {
      const v = q * 4;
      uv.set([0, 0, 0, 1, 1, 0, 1, 1], v * 2);
      idx.set([v, v + 1, v + 2, v + 1, v + 3, v + 2], q * 6);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', this.pos);
    g.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
    g.setAttribute('aAlpha', this.alpha);
    g.setAttribute('aId', this.id);
    g.setIndex(new THREE.BufferAttribute(idx, 1));
    g.setDrawRange(0, 0);
    this.geometry = g;
    this.material = new THREE.ShaderMaterial({
      uniforms: THREE.UniformsUtils.merge([THREE.UniformsLib.fog, {
        uHead: { value: 0 }, uCap: { value: max }, uFadeTail: { value: o.fadeTail ?? 0.3 },
        uColor: { value: new THREE.Color(o.color ?? 0x030303) }, uEdge: { value: o.edge ?? 0.18 },
        uTread: { value: o.tread === false ? 0 : 1 }, uMap: { value: o.map || null }, uHasMap: { value: o.map ? 1 : 0 },
      }]),
      vertexShader: VS, fragmentShader: FS, transparent: true, depthWrite: false, fog: true, side: THREE.DoubleSide,
      polygonOffset: true, polygonOffsetFactor: o.polygonOffset ?? -4, polygonOffsetUnits: o.polygonOffset ?? -4,
    });
    this.mesh = new THREE.Mesh(g, this.material);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = o.renderOrder ?? 2;
    this.mesh.name = o.name || 'decalTrail';
    this.trail = Array.from({ length: o.trails ?? 4 }, () => ({ on: false, l: new THREE.Vector3(), r: new THREE.Vector3(), c: new THREE.Vector3(), a: 0 }));
    this._l = new THREE.Vector3(); this._r = new THREE.Vector3();
    this._rp = newRange(); this._ra = newRange(); this._ri = newRange();
  }

  _quad(l0, r0, l1, r1, a0, a1) {
    const q = this.head % this.max, v = q * 4, P = this.pos.array, o = v * 3;
    P[o] = l0.x; P[o + 1] = l0.y; P[o + 2] = l0.z; P[o + 3] = r0.x; P[o + 4] = r0.y; P[o + 5] = r0.z;
    P[o + 6] = l1.x; P[o + 7] = l1.y; P[o + 8] = l1.z; P[o + 9] = r1.x; P[o + 10] = r1.y; P[o + 11] = r1.z;
    const A = this.alpha.array, I = this.id.array, id = this.head;
    A[v] = a0; A[v + 1] = a0; A[v + 2] = a1; A[v + 3] = a1;
    I[v] = id; I[v + 1] = id; I[v + 2] = id; I[v + 3] = id;
    touchRange(this.pos, this._rp, v * 3, 12); touchRange(this.alpha, this._ra, v, 4); touchRange(this.id, this._ri, v, 4);
    this.head++;
    this.geometry.setDrawRange(0, Math.min(this.head, this.max) * 6);
    this.material.uniforms.uHead.value = this.head;
  }

  /**
   * Extend trail i. point: world point on the surface; side: unit vector across the ribbon (e.g. wheel axis);
   * halfWidth (m); intensity 0..1 (<= 0.02 ends the trail, so the next feed starts a new one);
   * normal (optional unit surface normal): direction of the `lift` offset (world up when omitted).
   */
  feed(i, point, side, halfWidth, intensity, normal) {
    const tr = this.trail[i];
    if (intensity <= 0.02) { tr.on = false; return; }
    const l = this._l.copy(point).addScaledVector(side, -halfWidth);
    const r = this._r.copy(point).addScaledVector(side, halfWidth);
    const lift = this.lift;
    if (lift) {
      if (normal) { l.addScaledVector(normal, lift); r.addScaledVector(normal, lift); }
      else { l.y += lift; r.y += lift; }
    }
    if (!tr.on) { tr.on = true; tr.l.copy(l); tr.r.copy(r); tr.c.copy(point); tr.a = intensity; return; }
    const d2 = tr.c.distanceToSquared(point);
    if (d2 < this.minSegment * this.minSegment) return;
    if (d2 > this.maxSegment * this.maxSegment) { tr.l.copy(l); tr.r.copy(r); tr.c.copy(point); tr.a = intensity; return; }
    this._quad(tr.l, tr.r, l, r, tr.a * this.opacity, intensity * this.opacity);
    tr.l.copy(l); tr.r.copy(r); tr.c.copy(point); tr.a = intensity;
  }

  /** End trail i (next feed starts a fresh segment). */
  end(i) { this.trail[i].on = false; }

  /** Number of live segments. */
  get count() { return Math.min(this.head, this.max); }

  clear() {
    this.head = 0; this.geometry.setDrawRange(0, 0);
    for (const t of this.trail) t.on = false;
    this.id.array.fill(-1e9); this.id.clearUpdateRanges(); this.id.needsUpdate = true;
    this.material.uniforms.uHead.value = 0;
  }

  dispose() { this.geometry.dispose(); this.material.dispose(); }
}
