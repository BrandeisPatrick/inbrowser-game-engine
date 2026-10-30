// GPU-simulated sprite particles: the ParticleSystem's look and spawn API (same arguments, same physics, same
// shader output) with the per-particle work moved to the GPU, for tens of thousands of live sprites. Opt-in: the CPU
// ParticleSystem stays the default; swap the class where a system needs more than a few thousand particles.
//
//   if (GpuParticleSystem.isSupported(renderer)) {
//     const dust = new GpuParticleSystem(renderer, { max: 65536, blending: 'normal', drag: 0.25, lit: true, sun: atmosphere });
//     scene.add(dust.mesh);
//     dust.spawn(x, y, z, vx, vy, vz, life, size0, size1, alpha, r, g, b, r1, g1, b1, drag, grav);  // = ParticleSystem.spawn
//     per frame, after the camera moved: dust.update(dt, camera);
//     terrain contact without a CPU callback: { depthCollision: true } + dust.setDepth(postfx.sceneDepth, camera)
//   }
//
// Spawning stays on the CPU and is deterministic: slots are handed out sequentially from a ring (no randomness in
// GLSL; the sprite rotation comes from an integer hash of the spawn serial, or the optional rot / rotV arguments), so
// one spawn sequence always lands in the same slots with the same data. When the ring is full the oldest slot is
// overwritten; spawn returns -1 only once this frame's staging is full (stagingPerFrame).
//
// Data (W = min(1024, nextPow2(max)) texels wide, H = ceil(max / W) rows, slot s at texel (s % W, s / W)):
//   texturePos  RGBA32F ping-pong  xyz position (m), w age (s)                } GPUComputationRenderer variables, one
//   textureVel  RGBA32F ping-pong  xyz velocity (m/s), w life (s)            } fragment pass each; dead = age >= life
//   attrA       RGBA32F per slot   size0, size1, drag (1/s), gravity (m/s^2 along gravityDir)    CPU-written at spawn,
//   attrB       RGBA32F per slot   alpha0, rotation (12 bit angle : 12 bit rad/s in +-16),        uploaded as the
//                                  colour0, colour1 (7:7:7 bits + 3 bit shared exponent, 0..8)    touched texel ranges
//   staging     RGBA32F (SW x SH)  2 texels per spawn of this frame: (x, y, z, 0), (vx, vy, vz, life)
// The frame's spawns occupy the contiguous ring range [spawnStart, spawnStart + spawnCount) (mod max), so the compute
// pass finds a slot's staging texels from its own index (k = slot - spawnStart mod max -> texels 2k, 2k + 1): the
// per-slot "spawn id" is implicit and needs no texture.
//
// Per frame (update): advance the CPU clock (live-count histogram, draw window), upload the staged texels, then two
// compute passes (pos, vel; each reads the previous pos + vel + attrA and repeats the same step, so both agree)
// over only the texel rows that hold the draw window [tail, head) (all rows once the window wraps around the ring),
// with renderer.autoClear off around them (a clear ignores the viewport). Rows outside the window keep stale texels:
// harmless, because every slot that enters the window is either staged this update (state from staging) or was in
// the previous window (computed last update), and only window slots are drawn or sorted.
// The step mirrors ParticleSystem.update exactly: age += dt, die at age >= life, v = v * exp(-drag dt) + gravityDir *
// grav * dt, p += v dt, floor plane bounce (y = floor, vy = |vy| * bounce, vxz *= 0.6), then the optional depth-buffer
// collision. The mesh draws window-many instances starting at the tail slot; each vertex texelFetches pos, vel, attrA,
// attrB by instance, so the CPU uploads nothing per live particle. Dead slots collapse outside the clip volume.
//
// Cost model (measure, do not trust): CPU per spawn = ~20 typed-array writes + 2 integer hashes; per frame = O(1)
// bookkeeping plus texSubImage2D of the touched texels only (staging 32 B and attrA + attrB 32 B per spawn).
// GPU per frame = 2 passes x (window rows x W) fragments (4 fetches, ~80 B each), the instanced draw (4 vertices x 4
// fetches per window slot) and, as with the CPU system, the sprites' fill rate, which dominates for large soft dust.
// sort: true adds a GPU bitonic sort of the window (back to front, for normal blending): 1 key pass + L (L + 1) / 2
// passes with L = log2(nextPow2(window)) (136 passes at 65536: several ms of CPU in three.js render calls plus one
// render-target round trip each on tiled GPUs). The CPU system does not sort either; leave it off unless needed.
// First numbers (headless Chrome, ANGLE Metal, Apple GPU; 65536 live, 4096 spawns per frame): spawn ~55 ns each,
// update ~0.2 ms CPU, 0 B allocated by this module per frame (three's renderer.render inside each compute / sort
// pass allocates a few hundred bytes); sort: +0.8 ms CPU, ~4.4 ms GPU. Matches ParticleSystem to 1e-5 m after 150
// steps (same spawns); the GPU live set equals the CPU's exactly.
//
// Requirements (isSupported): WebGL2 with EXT_color_buffer_float (RGBA32F / RG32F render targets for the compute and
// sort passes) and >= 5 vertex texture units (WebGL2 guarantees 16). Float textures are only read with texelFetch /
// NEAREST, so OES_texture_float_linear is not needed. The constructor throws a clear Error when unsupported: fall back
// to ParticleSystem. Depth collision samples an UnsignedInt / float DepthTexture (standard 0..1 depth, perspective
// camera, no logarithmic or reversed depth).
//
// Depth collision (depthCollision: true + setDepth): the compute pass projects each particle with the view-projection
// of the camera that rendered the depth (captured in mesh.onBeforeRender when that camera draws the particles, i.e. the
// previous frame's render, TAA jitter included; falls back to the camera's current matrices when the mesh was not
// drawn), linearises the scene depth with near / far, reconstructs the surface point under the particle and its
// normal from the depth gradient (the smaller of the left / right and up / down view-space differences), and collides
// when the particle is below that surface plane (measured along the normal, so grazing views work) by less than
// depthThickness x its size + its travel this step: it is pushed back onto the plane and its velocity is reflected
// like the floor bounce (normal part * bounce, tangential part * 0.6). Screen-space limits: nothing collides
// off-screen or behind closer geometry, and thin foreground objects can catch particles passing just behind them.
//
// Notes: stagingPerFrame limits spawns per update() call, not per rendered frame (fixed-step catch-up calls update
// several times per frame, each with its own staging). The compute / sort passes are renderer.render calls, so with
// Engine's profiler (renderer.info.autoReset = false) they count as draw calls of the frame (2 + sortPasses).
// three's GPUComputationRenderer.dispose() does not dispose its pass-through material (one leftover program per
// system; a three.js issue).
//
// Not supported (CPU-only): floorAt (a per-particle JS callback, warned on construction and assignment; use floor or
// depthCollision), reading positions back, sprite sizes in pixels together with streaks.
import * as THREE from 'three';
import { GPUComputationRenderer } from 'three/addons/misc/GPUComputationRenderer.js';
import { softDotTexture, atmosphereSunFactor } from './Particles.js';

const TAU = Math.PI * 2;
const BUCKET_W = 1 / 64;   // live-count histogram: death times quantised to 1/64 s ...
const BUCKETS = 4096;      // ... over a 64 s horizon (longer lives are counted dead after 64 s; drawing is unaffected)
const WINDOW_SLACK = 1e-3; // s: the draw window keeps a slot this long past its CPU death time (GPU float32 age sums)
const ROT_V_RANGE = 16;    // rad/s: packed rotation speed range (+-)
const nextPow2 = (n) => { let p = 1; while (p < n) p *= 2; return p; };

/** Deterministic 32-bit integer hash (lowbias32). */
export function hash32(x) {
  x = (x ^ (x >>> 16)) >>> 0;
  x = Math.imul(x, 0x7feb352d); x = (x ^ (x >>> 15)) >>> 0;
  x = Math.imul(x, 0x846ca68b); x = (x ^ (x >>> 16)) >>> 0;
  return x;
}

/** RGB (0..8, linear) -> one float holding an exact 24-bit integer: 3-bit shared exponent, 7 bits per channel. */
export function packColor(r, g, b) {
  r = r > 0 ? r : 0; g = g > 0 ? g : 0; b = b > 0 ? b : 0;
  const m = r > g ? (r > b ? r : b) : (g > b ? g : b);
  let e = 0, s = 1 / 16; // scale 2^(e - 4): the smallest that holds the largest channel
  while (e < 7 && m > s) { e++; s *= 2; }
  const k = 127 / s;
  let qr = Math.round(r * k), qg = Math.round(g * k), qb = Math.round(b * k);
  if (qr > 127) qr = 127; if (qg > 127) qg = 127; if (qb > 127) qb = 127;
  return e * 2097152 + qr * 16384 + qg * 128 + qb;
}

/** Inverse of packColor (the vertex shader's unpackColor), into out[o..o+2]. */
export function unpackColor(f, out, o = 0) {
  const u = f | 0, s = Math.pow(2, (u >> 21) - 4) / 127;
  out[o] = ((u >> 14) & 127) * s; out[o + 1] = ((u >> 7) & 127) * s; out[o + 2] = (u & 127) * s;
  return out;
}

/** Rotation (rad) + rotation speed (rad/s, +-16) -> one float holding an exact 24-bit integer (12 : 12 bits). */
export function packRotation(rot, rotV) {
  let a = Math.round((rot / TAU) * 4096) % 4096; if (a < 0) a += 4096; if (!(a >= 0)) a = 0;
  let v = Math.round(rotV * (2048 / ROT_V_RANGE)) + 2048; v = v < 0 ? 0 : v > 4095 ? 4095 : v; if (!(v >= 0)) v = 2048;
  return a * 4096 + v;
}

/**
 * CPU side of the GPU particle system, WebGL-free (unit-testable in Node): the sequential slot ring, this frame's
 * staging texels, the per-slot attribute arrays, the live-count histogram and the draw window [tail, tail + window).
 * GpuParticleSystem uploads its arrays; nothing here allocates after construction.
 */
export class GpuParticleStaging {
  /**
   * @param {object} [o]
   * @param {number} [o.max=65536]            ring slots
   * @param {number} [o.stagingPerFrame=4096] spawns accepted between two flush() calls (clamped to max)
   * @param {number} [o.spin=1.2]             max hashed rotation speed (rad/s) when spawn gets no rotV
   */
  constructor(o = {}) {
    const max = this.max = Math.max(1, (o.max ?? 65536) | 0);
    const W = this.width = Math.min(1024, nextPow2(max)), H = this.height = Math.ceil(max / W);
    const cap = this.capacity = Math.max(1, Math.min(max, (o.stagingPerFrame ?? 4096) | 0));
    const SW = this.stagingWidth = Math.min(1024, nextPow2(cap * 2));
    const SH = this.stagingHeight = Math.ceil((cap * 2) / SW);
    this.spin = o.spin ?? 1.2;
    this.staging = new Float32Array(SW * SH * 4);
    this.attrA = new Float32Array(W * H * 4);
    this.attrB = new Float32Array(W * H * 4);
    this.deathAt = new Float64Array(max);
    this.slotBucket = new Int32Array(max);
    this.buckets = new Int32Array(BUCKETS);
    this.spawned = 0; this.dropped = 0; this.overwritten = 0; // cumulative
    this.reset();
  }

  /** Forget every particle and restart the ring at slot 0 (and the rotation hash at serial 0). */
  reset() {
    this.head = 0;      // next slot
    this.start = 0;     // first slot staged since the last flush
    this.n = 0;         // spawns staged since the last flush
    this.tail = 0;      // oldest slot of the draw window
    this.window = 0;    // slots in the draw window (superset of the live ones)
    this.live = 0;      // live count (histogram, exact to BUCKET_W)
    this.time = 0;
    this.bucketDone = 0;
    this.serial = 0;
    this.deathAt.fill(-Infinity); this.slotBucket.fill(-1); this.buckets.fill(0);
  }

  /** Stage one particle (ParticleSystem.spawn arguments, all given) -> its slot, or -1 when this frame's staging is full. */
  spawn(x, y, z, vx, vy, vz, life, s0, s1, a0, r, g, b, r1, g1, b1, drag, grav, rot, rotV) {
    if (this.n >= this.capacity) { this.dropped++; return -1; }
    const max = this.max, slot = this.head, t = this.time;
    if (!(life > 1e-3)) life = 1e-3;
    // the ring overwrites its oldest slot once every slot is in the draw window
    if (this.deathAt[slot] > t) this.overwritten++;
    const ob = this.slotBucket[slot];
    if (ob > this.bucketDone) { this.buckets[ob % BUCKETS]--; this.live--; }
    if (this.window === max) this.tail = this.tail + 1 === max ? 0 : this.tail + 1; else this.window++;
    const death = t + life;
    this.deathAt[slot] = death;
    let bk = Math.floor(death / BUCKET_W) + 1;
    const bmax = this.bucketDone + BUCKETS - 1;
    if (bk > bmax) bk = bmax;
    this.slotBucket[slot] = bk; this.buckets[bk % BUCKETS]++; this.live++;
    const S = this.staging, k8 = this.n * 8;
    S[k8] = x; S[k8 + 1] = y; S[k8 + 2] = z; S[k8 + 3] = 0;
    S[k8 + 4] = vx; S[k8 + 5] = vy; S[k8 + 6] = vz; S[k8 + 7] = life;
    const A = this.attrA, B = this.attrB, s4 = slot * 4, sr = this.serial++;
    A[s4] = s0; A[s4 + 1] = s1; A[s4 + 2] = drag; A[s4 + 3] = grav;
    if (rot === undefined) rot = hash32(sr * 2 + 1) * (TAU / 4294967296);
    if (rotV === undefined) rotV = (hash32(sr * 2 + 2) * (2 / 4294967296) - 1) * this.spin;
    B[s4] = a0; B[s4 + 1] = packRotation(rot, rotV); B[s4 + 2] = packColor(r, g, b); B[s4 + 3] = packColor(r1, g1, b1);
    this.n++; this.spawned++;
    this.head = slot + 1 === max ? 0 : slot + 1;
    return slot;
  }

  /** Advance the clock by dt: retire the histogram buckets that ended, shrink the draw window from its old end. */
  advance(dt) {
    const t = this.time += dt;
    const nb = Math.floor(t / BUCKET_W);
    if (nb - this.bucketDone >= BUCKETS) { this.buckets.fill(0); this.live = 0; this.bucketDone = nb; }
    else while (this.bucketDone < nb) { const k = ++this.bucketDone % BUCKETS; this.live -= this.buckets[k]; this.buckets[k] = 0; }
    // stops at the first slot still alive (a long-lived particle keeps younger dead ones in the window: conservative)
    const D = this.deathAt, max = this.max;
    while (this.window > 0 && D[this.tail] + WINDOW_SLACK <= t) { this.tail = this.tail + 1 === max ? 0 : this.tail + 1; this.window--; }
  }

  /** The staged spawns were handed to the GPU: the next frame's staging starts at the current head. */
  flush() { this.start = this.head; this.n = 0; }

  /**
   * Slots staged since the last flush as up to two inclusive slot ranges (two after a ring wrap):
   * out = [first0, last0, first1, last1], -1 = unused.
   */
  stagedSlots(out) {
    out[0] = out[1] = out[2] = out[3] = -1;
    const n = this.n; if (!n) return out;
    const s = this.start, e = s + n - 1, max = this.max;
    out[0] = s;
    if (e < max) out[1] = e; else { out[1] = max - 1; out[2] = 0; out[3] = e - max; }
    return out;
  }

  /** Texel rows holding the draw window: out = [firstRow, rowCount] (every row once the window wraps the ring). */
  windowRows(out) {
    if (!this.window) { out[0] = 0; out[1] = 0; return out; }
    const W = this.width, s = this.tail, e = s + this.window - 1;
    if (e < this.max) { out[0] = (s / W) | 0; out[1] = ((e / W) | 0) - out[0] + 1; } else { out[0] = 0; out[1] = this.height; }
    return out;
  }
}

// ---- shaders ------------------------------------------------------------------------------------------------------
// GPUComputationRenderer prepends `uniform sampler2D texturePos; uniform sampler2D textureVel;` and the resolution define.
const COMPUTE = /* glsl */`
uniform highp sampler2D uStaging;
uniform highp sampler2D uAttrA;
uniform int uSpawnStart, uSpawnCount, uMax, uW, uStagingW;
uniform float uDt, uFloor, uBounce;
uniform vec3 uGravDir;
#ifdef DEPTH_COLLIDE
uniform highp sampler2D uDepth;
uniform int uDepthOn;
uniform mat4 uDepthViewProj, uDepthView, uDepthProjInv, uDepthCamWorld;
uniform vec2 uDepthTexel, uDepthNF;
uniform float uDepthThickness;
// distance along the view axis of the depth sample at uv (perspective 0..1 depth)
float depthDist( vec2 uv ) {
	float d = texture( uDepth, uv ).r;
	return uDepthNF.x * uDepthNF.y / ( uDepthNF.y - d * ( uDepthNF.y - uDepthNF.x ) );
}
vec3 depthViewPos( vec2 uv ) {
	vec4 v = uDepthProjInv * vec4( uv * 2.0 - 1.0, 0.0, 1.0 );
	v.xyz /= v.w;
	return v.xyz * ( depthDist( uv ) / - v.z );
}
void collideDepth( inout vec4 P, inout vec4 V, vec4 A ) {
	vec4 clip = uDepthViewProj * vec4( P.xyz, 1.0 );
	if ( clip.w <= uDepthNF.x ) return;
	vec2 uv = clip.xy / clip.w * 0.5 + 0.5;
	if ( uv.x <= 0.0 || uv.y <= 0.0 || uv.x >= 1.0 || uv.y >= 1.0 ) return;
	float scene = depthDist( uv );
	if ( scene >= uDepthNF.y * 0.999 ) return; // sky
	float size = A.x + ( A.y - A.x ) * sqrt( clamp( P.w / max( V.w, 1e-3 ), 0.0, 1.0 ) );
	float tol = uDepthThickness * size + length( V.xyz ) * uDt;
	float behind = clip.w - scene; // along the view axis: up to 1 / cos(view, normal) x the true penetration
	if ( behind <= 0.0 || behind > tol * 16.0 ) return;
	vec3 c0 = depthViewPos( uv );
	vec3 l = depthViewPos( uv - vec2( uDepthTexel.x, 0.0 ) ), r = depthViewPos( uv + vec2( uDepthTexel.x, 0.0 ) );
	vec3 b = depthViewPos( uv - vec2( 0.0, uDepthTexel.y ) ), t = depthViewPos( uv + vec2( 0.0, uDepthTexel.y ) );
	vec3 dx = abs( r.z - c0.z ) < abs( c0.z - l.z ) ? r - c0 : c0 - l;
	vec3 dy = abs( t.z - c0.z ) < abs( c0.z - b.z ) ? t - c0 : c0 - b;
	vec3 nv = cross( dx, dy );
	float nl = length( nv );
	if ( nl < 1e-12 ) return;
	nv /= nl;
	if ( dot( nv, c0 ) > 0.0 ) nv = - nv; // facing the camera (view-space origin)
	// penetration below the surface plane, along its normal
	vec3 pv = ( uDepthView * vec4( P.xyz, 1.0 ) ).xyz;
	float pen = dot( c0 - pv, nv );
	if ( pen <= 0.0 || pen > tol ) return;
	vec3 n = normalize( mat3( uDepthCamWorld ) * nv );
	P.xyz += n * ( pen + 0.01 );
	float vn = dot( V.xyz, n );
	if ( vn < 0.0 ) V.xyz = ( V.xyz - vn * n ) * 0.6 - n * ( vn * uBounce );
}
#endif
void stepParticle( out vec4 P, out vec4 V ) {
	ivec2 c = ivec2( gl_FragCoord.xy );
	int slot = c.y * uW + c.x;
	int k = slot - uSpawnStart;
	if ( k < 0 ) k += uMax;
	if ( slot < uMax && k < uSpawnCount ) { // spawned this frame: initial state from the staging texels 2k, 2k + 1
		int t = k * 2;
		P = texelFetch( uStaging, ivec2( t % uStagingW, t / uStagingW ), 0 );
		V = texelFetch( uStaging, ivec2( ( t + 1 ) % uStagingW, ( t + 1 ) / uStagingW ), 0 );
	} else {
		P = texelFetch( texturePos, c, 0 );
		V = texelFetch( textureVel, c, 0 );
	}
	if ( ! ( P.w < V.w ) ) return; // dead: unchanged
	P.w += uDt;
	if ( ! ( P.w < V.w ) ) return; // died this step (ParticleSystem kills before integrating)
	vec4 A = texelFetch( uAttrA, c, 0 ); // size0, size1, drag, gravity
	float dr = exp( - A.z * uDt );
	V.xyz = V.xyz * dr + uGravDir * ( A.w * uDt );
	P.xyz += V.xyz * uDt;
	if ( P.y < uFloor ) { P.y = uFloor; V.y = abs( V.y ) * uBounce; V.xz *= 0.6; }
#ifdef DEPTH_COLLIDE
	if ( uDepthOn == 1 ) collideDepth( P, V, A );
#endif
}`;
const COMPUTE_POS = COMPUTE + '\nvoid main() { vec4 P, V; stepParticle( P, V ); gl_FragColor = P; }\n';
const COMPUTE_VEL = COMPUTE + '\nvoid main() { vec4 P, V; stepParticle( P, V ); gl_FragColor = V; }\n';

// bitonic sort of the draw window by view-space z (ascending = far to near), dead / padding last; ties by slot
const SORT_KEYS = /* glsl */`
uniform highp sampler2D uPos;
uniform highp sampler2D uVel;
uniform mat4 uView;
uniform int uFirst, uWindow, uMax, uW;
void main() {
	ivec2 c = ivec2( gl_FragCoord.xy );
	int i = c.y * uW + c.x;
	float key = 2e30, slot = 0.0;
	if ( i < uWindow ) {
		int s = uFirst + i;
		if ( s >= uMax ) s -= uMax;
		ivec2 sc = ivec2( s % uW, s / uW );
		vec4 P = texelFetch( uPos, sc, 0 ), V = texelFetch( uVel, sc, 0 );
		key = P.w < V.w ? ( uView * vec4( P.xyz, 1.0 ) ).z : 1e30;
		slot = float( s );
	}
	gl_FragColor = vec4( key, slot, 0.0, 0.0 );
}`;
const SORT_STEP = /* glsl */`
uniform highp sampler2D uSrc;
uniform int uJ, uK, uW;
void main() {
	ivec2 c = ivec2( gl_FragCoord.xy );
	int i = c.y * uW + c.x, p = i ^ uJ;
	vec2 a = texelFetch( uSrc, c, 0 ).xy, b = texelFetch( uSrc, ivec2( p % uW, p / uW ), 0 ).xy;
	bool aFirst = a.x < b.x || ( a.x == b.x && a.y <= b.y );
	bool keepFirst = ( ( i & uK ) == 0 ) == ( i < p );
	gl_FragColor = vec4( keepFirst == aFirst ? a : b, 0.0, 0.0 );
}`;

const VS = /* glsl */`
uniform highp sampler2D uPos;
uniform highp sampler2D uVel;
uniform highp sampler2D uAttrA;
uniform highp sampler2D uAttrB;
#ifdef SORTED
uniform highp sampler2D uOrder;
#endif
uniform int uFirst, uMax, uW;
uniform float uStreak, uLight, uFadeIn;
uniform vec3 uTint;
uniform vec2 uPx;
varying vec2 vUv;
varying float vAlpha;
varying vec3 vColor;
#include <fog_pars_vertex>
vec3 unpackColor( float f ) {
	int u = int( f );
	return vec3( float( ( u >> 14 ) & 127 ), float( ( u >> 7 ) & 127 ), float( u & 127 ) ) * ( exp2( float( u >> 21 ) - 4.0 ) / 127.0 );
}
void main() {
#ifdef SORTED
	int slot = int( texelFetch( uOrder, ivec2( gl_InstanceID % uW, gl_InstanceID / uW ), 0 ).y );
#else
	int slot = uFirst + gl_InstanceID;
	if ( slot >= uMax ) slot -= uMax;
#endif
	ivec2 c = ivec2( slot % uW, slot / uW );
	vec4 P = texelFetch( uPos, c, 0 );   // xyz, age
	vec4 V = texelFetch( uVel, c, 0 );   // xyz, life
	vec4 A = texelFetch( uAttrA, c, 0 ); // size0, size1, drag, gravity
	vec4 B = texelFetch( uAttrB, c, 0 ); // alpha0, rotation, colour0, colour1
	float t = clamp( P.w / max( V.w, 1e-3 ), 0.0, 1.0 );
	float size = A.x + ( A.y - A.x ) * sqrt( t );
	vUv = uv;
	vec4 mvPosition = modelViewMatrix * vec4( P.xyz, 1.0 );
	vec2 q = position.xy, off;
	if ( uStreak > 0.0 ) {
		vec2 d = ( modelViewMatrix * vec4( V.xyz, 0.0 ) ).xy;
		float l = length( d );
		vec2 dir = l > 1e-4 ? d / l : vec2( 1.0, 0.0 );
		vec2 nrm = vec2( - dir.y, dir.x );
		mvPosition.xy += dir * ( q.x - 0.5 ) * l * uStreak;
		off = dir * ( q.x - 0.5 ) * size + nrm * q.y * size;
	} else {
		int rp = int( B.y );
		float rot = float( rp >> 12 ) * ( 6.283185307 / 4096.0 ) + float( ( rp & 4095 ) - 2048 ) * ( ${(ROT_V_RANGE / 2048).toFixed(8)} ) * P.w;
		float cr = cos( rot ), sr = sin( rot );
		off = vec2( cr * q.x - sr * q.y, sr * q.x + cr * q.y ) * size;
	}
#ifdef SCREEN_SIZE
	gl_Position = projectionMatrix * mvPosition;
	gl_Position.xy += off * uPx * gl_Position.w;
#else
	mvPosition.xy += off;
	gl_Position = projectionMatrix * mvPosition;
#endif
	float fin = uFadeIn > 0.0 ? min( 1.0, P.w / uFadeIn ) : 1.0;
	vAlpha = B.x * fin * ( 1.0 - t ) * ( 1.0 - t );
	vColor = mix( unpackColor( B.z ), unpackColor( B.w ), t ) * uTint * uLight;
	#include <fog_vertex>
	if ( ! ( P.w < V.w ) ) gl_Position = vec4( 0.0, 0.0, 2.0, 1.0 ); // dead: outside the clip volume
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

const _v2 = new THREE.Vector2();

export class GpuParticleSystem {
  /** true when this renderer can run the system: WebGL2 + EXT_color_buffer_float + vertex texture fetch. */
  static isSupported(renderer) {
    if (!renderer || !renderer.capabilities || !renderer.extensions || typeof renderer.extensions.has !== 'function') return false;
    const c = renderer.capabilities;
    if (c.isWebGL2 === false || (c.maxVertexTextures ?? 0) < 5) return false;
    return !!renderer.extensions.has('EXT_color_buffer_float');
  }

  /**
   * @param {THREE.WebGLRenderer} renderer
   * @param {object} [o]  ParticleSystem options (same defaults) plus:
   * @param {number} [o.max=65536]
   * @param {THREE.Texture} [o.texture]       sprite (alias o.map; default: soft dot)
   * @param {'additive'|'normal'|number} [o.blending='additive']  or a THREE blending constant
   * @param {number|number[]|THREE.Vector3} [o.gravity=0]  m/s^2 downward (as ParticleSystem), or an acceleration
   *                                          vector: its length is the default per-particle grav, its direction gravityDir
   * @param {number} [o.drag=0]               1/s
   * @param {number|null} [o.floor=null]      ground plane height (y); null = none
   * @param {boolean} [o.depthCollision=false] collide with the scene depth (setDepth)
   * @param {number} [o.depthThickness=1]     collide while behind the depth by < this x size (+ the step's travel)
   * @param {number} [o.bounce=0.35]
   * @param {number} [o.streak=0]           s; needs sizeAttenuation (streak length is in view-space metres)
   * @param {boolean} [o.sizeAttenuation=true] false: sizes are drawing-buffer pixels (no streaks)
   * @param {boolean} [o.fog=true]
   * @param {number} [o.stagingPerFrame=4096] spawns accepted per update() call (spawn returns -1 beyond)
   * @param {number|THREE.Color} [o.color=0xffffff] tint multiplied into every particle colour
   * @param {boolean} [o.sort=false]          GPU bitonic back-to-front sort of the window (normal blending)
   *   fadeIn, light, lit, litAmbient, sun, spin, name, renderOrder: as ParticleSystem
   */
  constructor(renderer, o = {}) {
    if (!GpuParticleSystem.isSupported(renderer)) {
      throw new Error('GpuParticleSystem: needs a WebGL2 renderer with EXT_color_buffer_float (RGBA32F render targets) and vertex texture fetch; use ParticleSystem instead');
    }
    this.floorAt = o.floorAt; // warns when a function
    this.renderer = renderer;
    const st = this.staging = new GpuParticleStaging({ max: o.max ?? 65536, stagingPerFrame: o.stagingPerFrame ?? 4096, spin: o.spin ?? 1.2 });
    const max = this.max = st.max, W = st.width, H = st.height;
    this.gravityDir = new THREE.Vector3(0, -1, 0);
    const gv = o.gravity ?? 0;
    if (typeof gv === 'number') this.gravity = gv;
    else {
      const gx = gv.x ?? gv[0] ?? 0, gy = gv.y ?? gv[1] ?? 0, gz = gv.z ?? gv[2] ?? 0, l = Math.hypot(gx, gy, gz);
      this.gravity = l;
      if (l > 0) this.gravityDir.set(gx / l, gy / l, gz / l);
    }
    this.drag = o.drag ?? 0;
    this.floor = o.floor ?? -Infinity;
    this.bounce = o.bounce ?? 0.35;
    this.fadeInTime = o.fadeIn ?? 0.12;
    this.lit = !!o.lit; this.litAmbient = o.litAmbient ?? 0.15;
    this.sunSource = typeof o.sun === 'number' ? null : o.sun || null;
    this.sunFactor = typeof o.sun === 'number' ? o.sun : 1;
    this._light = o.light ?? 1;
    this.depthCollision = !!o.depthCollision;
    this.depthThickness = o.depthThickness ?? 1;
    this.sort = !!o.sort;
    this.sizeAttenuation = o.sizeAttenuation !== false;
    const caps = renderer.capabilities;
    if (this.depthCollision && caps && caps.reverseDepthBuffer === true) console.warn('GpuParticleSystem: depthCollision assumes standard 0..1 depth; reverseDepthBuffer is not supported');
    if (!this.sizeAttenuation && (o.streak ?? 0) > 0) console.warn('GpuParticleSystem: streak needs sizeAttenuation (streak length is view-space metres, sizes are pixels)');
    this.stats = { spawned: 0, dropped: 0, overwritten: 0, staged: 0, window: 0, live: 0, computeRows: 0, sortPasses: 0, uploadRanges: 0 };

    // ---- CPU-written textures (row-ranged uploads of the touched texels) ----
    const dataTex = (data, w, h, name) => {
      const t = new THREE.DataTexture(data, w, h, THREE.RGBAFormat, THREE.FloatType);
      t.minFilter = t.magFilter = THREE.NearestFilter; t.generateMipmaps = false; t.name = name; t.needsUpdate = true;
      return t;
    };
    this._stagingTex = dataTex(st.staging, st.stagingWidth, st.stagingHeight, 'GpuParticles.staging');
    this._attrATex = dataTex(st.attrA, W, H, 'GpuParticles.attrA');
    this._attrBTex = dataTex(st.attrB, W, H, 'GpuParticles.attrB');
    // pooled update ranges per texture; `used` returns to 0 once three uploaded them (onUpdate), -1 = full upload pending
    const pool = (tex, n) => { const p = []; for (let i = 0; i < n; i++) p.push({ start: 0, count: 0 }); p.used = 0; tex.onUpdate = () => { p.used = 0; }; return p; };
    this._poolS = pool(this._stagingTex, st.stagingHeight + 1); this._poolA = pool(this._attrATex, H + 4); this._poolB = pool(this._attrBTex, H + 4);
    this._slots = new Int32Array(4); this._rows = new Int32Array(2);

    // ---- compute: pos / vel ping-pong ----
    const gpu = this._gpu = new GPUComputationRenderer(W, H, renderer);
    const U = this._cu = {
      uStaging: { value: this._stagingTex }, uAttrA: { value: this._attrATex },
      uSpawnStart: { value: 0 }, uSpawnCount: { value: 0 }, uMax: { value: max }, uW: { value: W }, uStagingW: { value: st.stagingWidth },
      uDt: { value: 0 }, uFloor: { value: -1e30 }, uBounce: { value: this.bounce }, uGravDir: { value: this.gravityDir },
      uDepth: { value: null }, uDepthOn: { value: 0 }, uDepthViewProj: { value: new THREE.Matrix4() }, uDepthView: { value: new THREE.Matrix4() }, uDepthProjInv: { value: new THREE.Matrix4() },
      uDepthCamWorld: { value: new THREE.Matrix4() }, uDepthTexel: { value: new THREE.Vector2(1, 1) }, uDepthNF: { value: new THREE.Vector2(0.1, 1000) },
      uDepthThickness: { value: this.depthThickness },
    };
    this._posVar = gpu.addVariable('texturePos', COMPUTE_POS, gpu.createTexture()); // zeros: age 0 >= life 0 = dead
    this._velVar = gpu.addVariable('textureVel', COMPUTE_VEL, gpu.createTexture());
    for (const v of [this._posVar, this._velVar]) {
      gpu.setVariableDependencies(v, [this._posVar, this._velVar]);
      Object.assign(v.material.uniforms, U);
      if (this.depthCollision) v.material.defines.DEPTH_COLLIDE = 1;
    }
    const err = gpu.init();
    if (err) throw new Error('GpuParticleSystem: ' + err);
    this._depthSrc = null; this._depthCam = null; this._near = undefined; this._far = undefined; this._snapFresh = false;

    // ---- optional sort ----
    if (this.sort) {
      const SH = nextPow2(H), rtOpts = { type: THREE.FloatType, format: THREE.RGFormat, depthBuffer: false, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter, generateMipmaps: false };
      this._sortRT = [new THREE.WebGLRenderTarget(W, SH, rtOpts), new THREE.WebGLRenderTarget(W, SH, rtOpts)];
      this._keyMat = gpu.createShaderMaterial(SORT_KEYS, { uPos: { value: null }, uVel: { value: null }, uView: { value: new THREE.Matrix4() }, uFirst: { value: 0 }, uWindow: { value: 0 }, uMax: { value: max }, uW: { value: W } });
      this._sortMat = gpu.createShaderMaterial(SORT_STEP, { uSrc: { value: null }, uJ: { value: 1 }, uK: { value: 2 }, uW: { value: W } });
      this._sortCam = null;
    }

    // ---- render ----
    const quad = new THREE.InstancedBufferGeometry();
    quad.setAttribute('position', new THREE.Float32BufferAttribute([-0.5, -0.5, 0, 0.5, -0.5, 0, 0.5, 0.5, 0, -0.5, 0.5, 0], 3));
    quad.setAttribute('uv', new THREE.Float32BufferAttribute([0, 0, 1, 0, 1, 1, 0, 1], 2));
    quad.setIndex([0, 1, 2, 0, 2, 3]);
    quad.instanceCount = 0;
    this.geometry = quad;
    const defines = {};
    if (!this.sizeAttenuation) defines.SCREEN_SIZE = 1;
    if (this.sort) defines.SORTED = 1;
    const bl = o.blending ?? 'additive';
    this.material = new THREE.ShaderMaterial({
      uniforms: THREE.UniformsUtils.merge([THREE.UniformsLib.fog, {
        uMap: { value: null }, uStreak: { value: o.streak ?? 0 }, uLight: { value: this._light }, uFadeIn: { value: this.fadeInTime },
        uTint: { value: new THREE.Color(o.color ?? 0xffffff) }, uFirst: { value: 0 }, uMax: { value: max }, uW: { value: W },
        uPx: { value: new THREE.Vector2(1, 1) },
      }]),
      vertexShader: VS, fragmentShader: FS, defines,
      transparent: true, depthWrite: false, fog: o.fog !== false,
      blending: typeof bl === 'number' ? bl : bl === 'additive' ? THREE.AdditiveBlending : THREE.NormalBlending,
    });
    const mu = this.material.uniforms; // textures set after merge (UniformsUtils.merge clones textures)
    mu.uMap.value = o.texture || o.map || softDotTexture();
    mu.uPos = { value: null }; mu.uVel = { value: null }; mu.uAttrA = { value: this._attrATex }; mu.uAttrB = { value: this._attrBTex };
    mu.uOrder = { value: null };
    this.mesh = new THREE.Mesh(quad, this.material);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = o.renderOrder ?? 8;
    this.mesh.name = o.name || 'gpuParticles';
    this.mesh.visible = false;
    // the camera that draws the particles also rendered this frame's scene depth: keep its matrices for next frame's collisions
    this.mesh.onBeforeRender = (r, scene, camera) => {
      if (this.depthCollision && camera === this._depthCam) { this._snapshot(camera); this._snapFresh = true; }
    };
    this._bindState();
  }

  get spin() { return this.staging.spin; }
  set spin(v) { this.staging.spin = v; }
  /** Live particles (CPU bookkeeping, exact to 1/64 s for lives up to 64 s). */
  get count() { return this.staging.live; }
  /** ParticleSystem's terrain callback is CPU-only: always null here; assigning a function warns. */
  get floorAt() { return null; }
  set floorAt(fn) { if (typeof fn === 'function') console.warn('GpuParticleSystem: floorAt is CPU-only and ignored; use floor or depthCollision'); }
  get streak() { return this.material.uniforms.uStreak.value; }
  set streak(v) { this.material.uniforms.uStreak.value = v; }

  /**
   * Spawn one particle (ParticleSystem.spawn arguments and semantics) -> its slot, or -1 when this update's staging is
   * full (stagingPerFrame). Optional rot (rad) / rotV (rad/s) replace the hashed sprite rotation. Packing limits:
   * colours clamp to 0..8 per channel (negatives / NaN to 0; brighter HDR goes through light / tint / the sun factor,
   * which are uniforms), rotV clamps to +-16 rad/s (rotation is stored to 1/4096 turn).
   */
  spawn(x, y, z, vx, vy, vz, life, s0, s1, a0, r, g, b, r1 = r, g1 = g, b1 = b, drag = this.drag, grav = this.gravity, rot, rotV) {
    return this.staging.spawn(x, y, z, vx, vy, vz, life, s0, s1, a0, r, g, b, r1, g1, b1, drag, grav, rot, rotV);
  }

  /** Spawn from a (reusable) params object: { x, y, z, vx, vy, vz, life, size0, size1, alpha, r, g, b, r1?, g1?, b1?, drag?, gravity? }. */
  emit(p) {
    return this.spawn(p.x, p.y, p.z, p.vx, p.vy, p.vz, p.life, p.size0, p.size1 ?? p.size0, p.alpha ?? 1, p.r ?? 1, p.g ?? 1, p.b ?? 1,
      p.r1 ?? p.r ?? 1, p.g1 ?? p.g ?? 1, p.b1 ?? p.b ?? 1, p.drag ?? this.drag, p.gravity ?? this.gravity, p.rot, p.rotV);
  }

  /**
   * Depth collision source: a depth texture (or an object with `sceneDepth` / a function returning one, read every
   * update) and the camera that renders it; near / far default to that camera's.
   */
  setDepth(depthTexture, camera, near, far) {
    this._depthSrc = depthTexture || null;
    if (camera) this._depthCam = camera;
    this._near = near; this._far = far;
    this._snapFresh = false;
  }

  _depthTexture() {
    const s = this._depthSrc;
    if (!s) return null;
    const t = typeof s === 'function' ? s() : s.isTexture ? s : s.sceneDepth;
    return t && t.isTexture ? t : null;
  }

  _snapshot(cam) {
    const u = this._cu;
    u.uDepthViewProj.value.multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse);
    u.uDepthView.value.copy(cam.matrixWorldInverse);
    u.uDepthProjInv.value.copy(cam.projectionMatrixInverse);
    u.uDepthCamWorld.value.copy(cam.matrixWorld);
    u.uDepthNF.value.set(this._near ?? cam.near, this._far ?? cam.far);
  }

  /** Mark texels first..last (inclusive) of a W-wide texture dirty: one pooled update range per touched row. */
  _touch(tex, pool, first, last, W) {
    const ur = tex.updateRanges;
    tex.needsUpdate = true;
    if (pool.used === -1) return 0; // full upload already pending
    let n = 0;
    for (let s = first; s <= last;) {
      const rowEnd = Math.min(last, ((s / W) | 0) * W + W - 1);
      if (pool.used >= pool.length) { ur.length = 0; pool.used = -1; return n; } // too many pending (not drawn): upload all
      const r = pool[pool.used++];
      r.start = s * 4; r.count = (rowEnd - s + 1) * 4;
      ur.push(r); n++;
      s = rowEnd + 1;
    }
    return n;
  }

  _bindState() {
    const mu = this.material.uniforms, gpu = this._gpu;
    mu.uPos.value = gpu.getCurrentRenderTarget(this._posVar).texture;
    mu.uVel.value = gpu.getCurrentRenderTarget(this._velVar).texture;
  }

  /**
   * Per frame: upload this frame's spawns, step the simulation on the GPU, update the draw window. `camera` (optional)
   * is used for sorting and as the depth camera when setDepth named none. Allocation-free.
   */
  update(dt, camera) {
    const st = this.staging, u = this._cu, gpu = this._gpu, stats = this.stats;
    st.advance(dt);
    const n = st.n, W = st.width;
    let ranges = 0;
    if (n > 0) {
      ranges += this._touch(this._stagingTex, this._poolS, 0, n * 2 - 1, st.stagingWidth);
      const sl = st.stagedSlots(this._slots);
      for (let i = 0; i < 4; i += 2) {
        if (sl[i] < 0) continue;
        ranges += this._touch(this._attrATex, this._poolA, sl[i], sl[i + 1], W);
        ranges += this._touch(this._attrBTex, this._poolB, sl[i], sl[i + 1], W);
      }
    }
    let rows = 0;
    if (st.window > 0) {
      u.uDt.value = dt; u.uSpawnStart.value = st.start; u.uSpawnCount.value = n;
      u.uFloor.value = Number.isFinite(this.floor) ? this.floor : -1e30; // null / -Infinity / NaN = no floor
      u.uBounce.value = this.bounce;
      if (this.depthCollision) {
        if (camera && !this._depthCam) this._depthCam = camera;
        const tex = this._depthTexture(), cam = this._depthCam;
        if (cam && !this._snapFresh) this._snapshot(cam);
        u.uDepth.value = tex; u.uDepthOn.value = tex && cam ? 1 : 0;
        if (tex && tex.image) u.uDepthTexel.value.set(1 / Math.max(1, tex.image.width), 1 / Math.max(1, tex.image.height));
        u.uDepthThickness.value = this.depthThickness;
        this._snapFresh = false;
      }
      const wr = st.windowRows(this._rows);
      rows = wr[1];
      const pr = this._posVar.renderTargets, vr = this._velVar.renderTargets;
      pr[0].viewport.set(0, wr[0], W, rows); pr[1].viewport.set(0, wr[0], W, rows);
      vr[0].viewport.set(0, wr[0], W, rows); vr[1].viewport.set(0, wr[0], W, rows);
      const ac = this.renderer.autoClear;
      this.renderer.autoClear = false; // a clear ignores the viewport: it would rewrite every row of the target
      gpu.compute();
      this.renderer.autoClear = ac;
      this._bindState();
    }
    stats.staged = n;
    st.flush();
    const win = st.window, mu = this.material.uniforms;
    mu.uFirst.value = st.tail;
    mu.uFadeIn.value = this.fadeInTime;
    this.geometry.instanceCount = win;
    this.mesh.visible = win > 0;
    stats.sortPasses = this.sort && win > 0 ? this._sortWindow(camera) : 0;
    if (!this.sizeAttenuation) { this.renderer.getDrawingBufferSize(_v2); mu.uPx.value.set(2 / Math.max(1, _v2.x), 2 / Math.max(1, _v2.y)); }
    if (this.lit) this._applyLight();
    stats.spawned = st.spawned; stats.dropped = st.dropped; stats.overwritten = st.overwritten;
    stats.window = win; stats.live = st.live; stats.computeRows = rows; stats.uploadRanges = ranges;
  }

  _sortWindow(camera) {
    const st = this.staging, gpu = this._gpu, W = st.width, rt = this._sortRT, km = this._keyMat.uniforms, sm = this._sortMat.uniforms;
    if (camera) this._sortCam = camera;
    const cam = this._sortCam;
    const N = nextPow2(Math.max(2, st.window)), w = Math.min(N, W), h = Math.max(1, N / W);
    rt[0].viewport.set(0, 0, w, h); rt[1].viewport.set(0, 0, w, h);
    km.uPos.value = this.material.uniforms.uPos.value; km.uVel.value = this.material.uniforms.uVel.value;
    if (cam) km.uView.value.copy(cam.matrixWorldInverse); else km.uView.value.identity();
    km.uFirst.value = st.tail; km.uWindow.value = st.window;
    const ac = this.renderer.autoClear;
    this.renderer.autoClear = false; // each pass writes its whole region: no clear (a load / store per pass on tiled GPUs)
    gpu.doRenderTarget(this._keyMat, rt[0]);
    let src = 0, passes = 1;
    for (let k = 2; k <= N; k *= 2) {
      for (let j = k >> 1; j > 0; j >>= 1) {
        sm.uSrc.value = rt[src].texture; sm.uJ.value = j; sm.uK.value = k;
        gpu.doRenderTarget(this._sortMat, rt[1 - src]);
        src = 1 - src; passes++;
      }
    }
    this.renderer.autoClear = ac;
    this.material.uniforms.uOrder.value = rt[src].texture;
    return passes;
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

  /** Kill everything (CPU bookkeeping only: slots outside the draw window are never drawn or read again). */
  clear() { this.staging.reset(); this.geometry.instanceCount = 0; this.mesh.visible = false; }

  dispose() {
    this._gpu.dispose();
    this._stagingTex.dispose(); this._attrATex.dispose(); this._attrBTex.dispose();
    this._posVar.material.dispose(); this._velVar.material.dispose();
    if (this._sortRT) { this._sortRT[0].dispose(); this._sortRT[1].dispose(); this._keyMat.dispose(); this._sortMat.dispose(); }
    this.geometry.dispose(); this.material.dispose();
  }
}
