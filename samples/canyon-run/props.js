// Desert props: boulders, slabs and hoodoo spires (static box colliders + InstancedPool near sets selected by
// LODGroup, baked impostors for everything far), saguaro cacti (thin colliders, slight sway) and dry shrubs
// (alpha-tested cross quads with wind sway). Near models dither out exactly where the impostors dither in.
import * as THREE from 'three';
import { mergeGeometries, mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js';
import {
  InstancedPool, LODGroup, bakeImpostors, ImpostorBatch, windSway, ditherFade, applyPatches, makeDepthMaterial,
  makeCanvas, finishTexture, Simplex, mulberry32 as rng, Noise,
} from '@vibe/engine';
import { TRACK_HALF_WIDTH } from './track.js';
import { WORLD_HALF } from './terrain.js';
const { smoothstep } = Noise;

export const PROP_FADE = { start: 250, end: 290 };   // near rocks / cacti -> impostors
const SHRUB_FADE = { start: 115, end: 150 };
const KIND_ROCK = 1, KIND_THIN = 3;

// ------------------------------------------------------------------ geometry
const srgb = new THREE.Color();
function paint(g, fn) {
  const p = g.attributes.position, n = p.count, col = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    fn(p.getX(i), p.getY(i), p.getZ(i), srgb, i);
    col[i * 3] = srgb.r; col[i * 3 + 1] = srgb.g; col[i * 3 + 2] = srgb.b;
  }
  g.setAttribute('color', new THREE.BufferAttribute(col, 3));
  return g;
}
function clean(g) { // position-only, welded, smooth normals
  const out = new THREE.BufferGeometry();
  out.setAttribute('position', g.getAttribute('position'));
  if (g.index) out.setIndex(g.index);
  const m = mergeVertices(out, 1e-4);
  m.computeVertexNormals();
  return m;
}

/** Sandstone colour by height band + crevice darkening (sRGB in, linear out). */
function sandstone(ns, x, y, z, disp, out, base = 1) {
  const b = Math.floor(y * 2.2 + ns.noise(x * 0.7, z * 0.7) * 0.6);
  const k = ((b * 7919) % 5 + 5) % 5;
  const pal = [[0.72, 0.38, 0.22], [0.82, 0.5, 0.3], [0.66, 0.33, 0.2], [0.88, 0.64, 0.44], [0.76, 0.44, 0.26]][k];
  const ao = 0.62 + 0.38 * smoothstep(-0.16, 0.12, disp);
  const l = base * ao * (0.9 + 0.12 * ns.noise(x * 3.1, z * 3.1 + y));
  return out.setRGB(pal[0] * l, pal[1] * l, pal[2] * l, THREE.SRGBColorSpace);
}

function rockGeometry(kind, seed) {
  const ns = new Simplex(seed);
  let g;
  if (kind === 'spire') {
    // hoodoo: lathe profile (waisted column with a cap rock), radius ~1, height 1
    const prof = [[0.001, -0.06], [1.0, -0.06], [0.92, 0.12], [0.7, 0.3], [0.56, 0.5], [0.5, 0.66], [0.62, 0.78], [0.86, 0.86], [0.9, 0.93], [0.6, 0.99], [0.001, 1.0]]
      .map(([r, y]) => new THREE.Vector2(r, y));
    g = clean(new THREE.LatheGeometry(prof, 14));
  } else g = clean(new THREE.IcosahedronGeometry(1, 3));
  const p = g.attributes.position, disp = new Float32Array(p.count);
  const v = new THREE.Vector3();
  for (let i = 0; i < p.count; i++) {
    v.fromBufferAttribute(p, i);
    let d;
    if (kind === 'spire') {
      const r = Math.hypot(v.x, v.z);
      d = 0.16 * ns.fbm(v.x * 2.2, v.z * 2.2 + v.y * 5, 3) + 0.05 * Math.sin(v.y * 40);
      if (r > 1e-3) { const k = 1 + d; v.x *= k; v.z *= k; }
    } else {
      d = 0.28 * ns.fbm(v.x * 1.3 + v.y * 0.7, v.z * 1.3 - v.y * 0.9, 3) + 0.08 * ns.noise(v.x * 5, v.z * 5 + v.y * 3);
      v.multiplyScalar(1 + d);
      if (kind === 'boulder') { v.y = v.y * 0.72; if (v.y < -0.25) v.y = -0.25 + (v.y + 0.25) * 0.3; v.y += 0.22; }
      else { v.x *= 1.35; v.y = v.y * 0.42; if (v.y < -0.12) v.y = -0.12 + (v.y + 0.12) * 0.3; v.y += 0.1; } // slab
    }
    disp[i] = d;
    p.setXYZ(i, v.x, v.y, v.z);
  }
  g.computeVertexNormals();
  paint(g, (x, y, z, c, i) => sandstone(ns, x, y * (kind === 'spire' ? 6 : 1), z, disp[i], c, kind === 'slab' ? 0.92 : 1));
  g.computeBoundingSphere();
  return g;
}

function cactusGeometry() {
  const parts = [];
  const ribbed = (rTop, rBot, h, seg = 12, hs = 6) => {
    const c = new THREE.CylinderGeometry(rTop, rBot, h, seg, hs, true);
    const p = c.attributes.position;
    for (let i = 0; i < p.count; i++) {
      const x = p.getX(i), z = p.getZ(i), a = Math.atan2(z, x), k = 1 + 0.09 * Math.cos(a * 10);
      p.setX(i, x * k); p.setZ(i, z * k);
    }
    return c;
  };
  const cap = (r) => new THREE.SphereGeometry(r, 12, 5, 0, Math.PI * 2, 0, Math.PI / 2);
  parts.push(ribbed(0.3, 0.36, 5.2).translate(0, 2.6, 0), cap(0.3).translate(0, 5.2, 0));
  // two arms: elbow out, then up
  for (const [side, y0, len, up] of [[1, 2.1, 0.75, 1.6], [-1, 2.9, 0.6, 1.2]]) {
    const out = ribbed(0.2, 0.2, len, 10, 2).rotateZ(Math.PI / 2).translate(side * (0.3 + len / 2), y0, 0);
    const rise = ribbed(0.19, 0.21, up, 10, 3).translate(side * (0.3 + len), y0 + up / 2, 0);
    parts.push(out, rise, cap(0.19).translate(side * (0.3 + len), y0 + up, 0), new THREE.SphereGeometry(0.21, 10, 6).translate(side * (0.3 + len), y0, 0));
  }
  const g = mergeGeometries(parts.map((q) => { const n = q.toNonIndexed(); n.deleteAttribute('uv'); return n; }));
  g.computeVertexNormals();
  paint(g, (x, y, z, c) => {
    const a = Math.atan2(z, x), rib = 0.5 + 0.5 * Math.cos(a * 10);
    const l = 0.75 + 0.25 * rib, top = smoothstep(4.6, 5.4, y) * 0.25;
    c.setRGB((0.34 + top * 0.5) * l, (0.46 + top * 0.3) * l, (0.24 + top * 0.1) * l, THREE.SRGBColorSpace);
  });
  g.computeBoundingSphere();
  return g;
}

function shrubTexture(renderer) {
  const S = 256, c = makeCanvas(S), g = c.getContext('2d');
  const r = rng(77);
  g.clearRect(0, 0, S, S);
  g.lineCap = 'round';
  const twig = (x, y, a, len, w, depth) => {
    const x2 = x + Math.cos(a) * len, y2 = y + Math.sin(a) * len;
    const t = depth / 4;
    g.strokeStyle = `rgb(${110 + 50 * t + r() * 30 | 0},${88 + 40 * t + r() * 25 | 0},${58 + 20 * t | 0})`;
    g.lineWidth = w; g.beginPath(); g.moveTo(x, y); g.lineTo(x2, y2); g.stroke();
    if (depth > 0) for (let k = 0; k < 2 + (r() < 0.5); k++) twig(x2, y2, a + (r() - 0.5) * 1.3, len * (0.55 + r() * 0.2), w * 0.65, depth - 1);
    else if (r() < 0.7) { g.fillStyle = `rgba(${120 + r() * 40 | 0},${125 + r() * 30 | 0},${70 + r() * 20 | 0},0.95)`; g.beginPath(); g.arc(x2, y2, 2 + r() * 3, 0, 6.3); g.fill(); }
  };
  for (let k = 0; k < 9; k++) twig(S / 2 + (r() - 0.5) * 40, S - 4, -Math.PI / 2 + (r() - 0.5) * 1.8, 52 + r() * 30, 5, 4);
  const t = finishTexture(new THREE.CanvasTexture(c), renderer, true, false);
  t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
  return t;
}
function shrubGeometry() {
  const planes = [];
  for (let k = 0; k < 3; k++) planes.push(new THREE.PlaneGeometry(1.6, 1.25).translate(0, 0.6, 0).rotateY((k / 3) * Math.PI));
  const g = mergeGeometries(planes);
  const n = g.attributes.normal; // soft, dome-like normals (no hard card shading)
  const p = g.attributes.position;
  for (let i = 0; i < n.count; i++) { const v = new THREE.Vector3(p.getX(i), p.getY(i) + 0.3, p.getZ(i)).normalize(); n.setXYZ(i, v.x, v.y, v.z); }
  g.computeBoundingSphere();
  return g;
}

// ------------------------------------------------------------------ rock detail patch (world triplanar from the texture array)
function patchRockDetail(mat, arrays) {
  const u = { tAlb: { value: arrays.albedo }, tNrm: { value: arrays.normal } };
  mat.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, u);
    sh.vertexShader = sh.vertexShader.replace('#include <common>', '#include <common>\nvarying vec3 vRWPos;\nvarying vec3 vRWNrm;')
      .replace('#include <fog_vertex>', `#include <fog_vertex>
	{
		vec4 rwp = vec4( transformed, 1.0 ); vec3 rwn = objectNormal;
	#ifdef USE_INSTANCING
		rwp = instanceMatrix * rwp; rwn = mat3( instanceMatrix ) * rwn;
	#endif
		vRWPos = ( modelMatrix * rwp ).xyz; vRWNrm = normalize( mat3( modelMatrix ) * rwn );
	}`);
    sh.fragmentShader = sh.fragmentShader.replace('#include <common>', `#include <common>
uniform highp sampler2DArray tAlb;
uniform highp sampler2DArray tNrm;
varying vec3 vRWPos;
varying vec3 vRWNrm;
vec3 rkN;`)
      .replace('#include <map_fragment>', `#include <map_fragment>
	{
		vec3 Ng = normalize( vRWNrm ), P = vRWPos / 2.6;
		vec3 bw = pow( abs( Ng ), vec3( 4.0 ) ); bw /= ( bw.x + bw.y + bw.z );
		vec4 a = texture( tAlb, vec3( P.zy, 1.0 ) ) * bw.x + texture( tAlb, vec3( P.xz, 1.0 ) ) * bw.y + texture( tAlb, vec3( P.xy, 1.0 ) ) * bw.z;
		vec4 nX = texture( tNrm, vec3( P.zy, 1.0 ) ), nY = texture( tNrm, vec3( P.xz, 1.0 ) ), nZ = texture( tNrm, vec3( P.xy, 1.0 ) );
		diffuseColor.rgb *= ( 0.5 + 1.25 * dot( a.rgb, vec3( 0.299, 0.587, 0.114 ) ) / 0.36 ) * ( 0.6 + 0.4 * a.a );
		vec2 tX = nX.xy * 2.0 - 1.0, tY = nY.xy * 2.0 - 1.0, tZ = nZ.xy * 2.0 - 1.0;
		rkN = normalize( Ng + 1.2 * ( vec3( 0.0, tX.y, tX.x ) * sign( Ng.x ) * bw.x + vec3( tY.x, 0.0, tY.y ) * bw.y + vec3( tZ.x, tZ.y, 0.0 ) * sign( Ng.z ) * bw.z ) );
	}`)
      .replace('#include <normal_fragment_maps>', 'normal = normalize( ( viewMatrix * vec4( rkN, 0.0 ) ).xyz );');
  };
  mat.customProgramCacheKey = () => 'canyon-rock-v1';
  return mat;
}

// ------------------------------------------------------------------ near/far instance set
class NearSet {
  /** One model type: all instance matrices + an LODGroup choosing the near ones for an InstancedPool. */
  constructor(name, geometry, material, n, { nearRadius, castShadow = true, depthMaterial = null, cullRadius }) {
    this.n = 0; this.cap = n;
    this.pos = new Float32Array(n * 3); this.mats = new Float32Array(n * 16);
    this.pool = new InstancedPool(geometry, material, { capacity: 64, name: `${name}-near`, castShadow, cullRadius });
    if (depthMaterial) this.pool.mesh.customDepthMaterial = depthMaterial;
    this.nearRadius = nearRadius;
    this.lod = null;
  }
  push(x, y, z, yaw, sx, sy) {
    const i = this.n++, c = Math.cos(yaw), s = Math.sin(yaw), m = this.mats, k = i * 16;
    m[k] = c * sx; m[k + 1] = 0; m[k + 2] = -s * sx; m[k + 3] = 0;
    m[k + 4] = 0; m[k + 5] = sy; m[k + 6] = 0; m[k + 7] = 0;
    m[k + 8] = s * sx; m[k + 9] = 0; m[k + 10] = c * sx; m[k + 11] = 0;
    m[k + 12] = x; m[k + 13] = y; m[k + 14] = z; m[k + 15] = 1;
    this.pos[i * 3] = x; this.pos[i * 3 + 1] = y; this.pos[i * 3 + 2] = z;
    return i;
  }
  finish() { this.lod = new LODGroup({ positions: this.pos, count: this.n, nearRadius: this.nearRadius, moveThreshold: 18, horizontal: false }); }
  update(cameraPos, dt) {
    if (this.lod.update(cameraPos, dt)) this.lod.copyNear(this.mats, this.pool);
    this.pool.update(cameraPos);
  }
}

// ------------------------------------------------------------------ props
export class Props {
  /**
   * @param {object} o  scene, renderer, terrain, track, physics, arrays (texture arrays), U (shared time uniforms)
   */
  constructor({ scene, renderer, terrain, track, physics, arrays, U, seed = 5 }) {
    this.scene = scene; this.terrain = terrain; this.track = track; this.physics = physics; this.U = U;
    const r = this.rand = rng(seed);
    // --- models ---
    const geos = this.geos = {
      boulder: rockGeometry('boulder', seed + 1), slab: rockGeometry('slab', seed + 2), spire: rockGeometry('spire', seed + 3), cactus: cactusGeometry(),
    };
    const fadeOut = ditherFade(U, { start: PROP_FADE.start, end: PROP_FADE.end, mode: 'out' });
    const rockMat = applyPatches(patchRockDetail(new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.93, envMapIntensity: 0.7 }), arrays), [fadeOut]);
    const rockDepth = makeDepthMaterial(rockMat);
    const cactusMat = applyPatches(new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.75, envMapIntensity: 0.8 }), [
      windSway(U, { strength: 0.035, weight: 'height', height: 6, frequency: 0.7 }), fadeOut,
    ]);
    const shrubMat = applyPatches(new THREE.MeshStandardMaterial({ map: shrubTexture(renderer), alphaTest: 0.45, alphaToCoverage: true, side: THREE.DoubleSide, roughness: 0.95 }), [
      windSway(U, { strength: 0.12, weight: 'height', height: 1.2, frequency: 1.3, flutter: 0.015 }),
      ditherFade(U, { start: SHRUB_FADE.start, end: SHRUB_FADE.end, mode: 'out' }),
    ]);
    const near = PROP_FADE.end + 26;
    this.sets = {
      boulder: new NearSet('boulder', geos.boulder, rockMat, 1400, { nearRadius: near, depthMaterial: rockDepth }),
      slab: new NearSet('slab', geos.slab, rockMat, 500, { nearRadius: near, depthMaterial: rockDepth }),
      spire: new NearSet('spire', geos.spire, rockMat, 160, { nearRadius: near + 40, depthMaterial: rockDepth, cullRadius: 1.2 }),
      cactus: new NearSet('cactus', geos.cactus, cactusMat, 900, { nearRadius: near, depthMaterial: makeDepthMaterial(cactusMat) }),
      shrub: new NearSet('shrub', shrubGeometry(), shrubMat, 2600, { nearRadius: SHRUB_FADE.end + 24, castShadow: false }),
    };
    this.models = ['boulder', 'slab', 'spire', 'cactus'];
    this.instances = []; // far list: [x, y, z, yaw, sx, sy, model]
    this._scatter();
    this.setList = Object.values(this.sets);
    for (const set of this.setList) { set.finish(); scene.add(set.pool.mesh); }
    // --- impostors for everything beyond PROP_FADE ---
    this.far = null;
    const baked = bakeImpostors(this.models.map((k) => ({ geometry: geos[k] })), { label: 'canyon-run', name: 'desertProps', cellPx: 128, texSize: 2048 });
    if (baked) {
      const n = this.instances.length / 7;
      const far = this.far = new ImpostorBatch(baked, { capacity: n, viewPos: U.uViewPos, fade: PROP_FADE, name: 'props-far', roughness: 0.92 });
      const I = this.instances;
      for (let i = 0; i < n; i++) { const k = i * 7; far.setInstance(i, I[k], I[k + 1], I[k + 2], I[k + 3], I[k + 4], I[k + 5], I[k + 6]); }
      scene.add(far.mesh);
    } else {
      for (const m of [rockMat, cactusMat]) m.userData.patchUniforms.uFade.value.set(900, 1000, 0, 0);
    }
    this.stats = { rocks: this.sets.boulder.n + this.sets.slab.n + this.sets.spire.n, cacti: this.sets.cactus.n, shrubs: this.sets.shrub.n, colliders: physics.staticBoxes.length };
  }

  _ok(x, z, clear) {
    if (Math.abs(x) > WORLD_HALF - 140 || Math.abs(z) > WORLD_HALF - 140) return false;
    if (this.terrain.trackDistance(x, z) < TRACK_HALF_WIDTH + clear) return false;
    // keep the gates / gantry surroundings clear
    for (const cp of this.track.checkpoints) if ((cp.x - x) ** 2 + (cp.z - z) ** 2 < 22 * 22) return false;
    return true;
  }

  _place(model, x, z, yaw, sx, sy, sink, collider) {
    const te = this.terrain;
    // sit on the lowest ground under the footprint so nothing floats on slopes
    const R = sx * 0.8;
    const y = Math.min(te.heightAt(x, z), te.heightAt(x + R, z), te.heightAt(x - R, z), te.heightAt(x, z + R), te.heightAt(x, z - R)) - sink;
    this.sets[model].push(x, y, z, yaw, sx, sy);
    if (model !== 'shrub') this.instances.push(x, y, z, yaw, sx, sy, this.models.indexOf(model));
    if (collider) {
      const [hx, top, kind] = collider;
      this.physics.addStaticBox(x - hx, y, x + hx, y + top, z - hx, z + hx, kind);
    }
  }

  _scatter() {
    const tr = this.track, te = this.terrain, r = this.rand;
    const side = (i, d) => [tr.x[i] - tr.tz[i] * d, tr.z[i] + tr.tx[i] * d];
    // floor boulders + slabs along the canyon
    for (let i = 0; i < tr.n; i += 4) {
      for (const sgn of [-1, 1]) {
        if (r() > 0.42) continue;
        const w0 = tr.halfFloor[i];
        const d = sgn * (TRACK_HALF_WIDTH + 3 + r() * Math.max(2, w0 - TRACK_HALF_WIDTH + 6));
        const [x, z] = side(i, d);
        const s = r() < 0.82 ? 0.35 + Math.pow(r(), 2.2) * 2.2 : 1.5 + r() * 2.5;
        if (!this._ok(x, z, 2 + s)) continue;
        const model = r() < 0.7 ? 'boulder' : 'slab';
        this._place(model, x, z, r() * 6.283, s, s * (0.8 + r() * 0.5), s * 0.12, s > 0.55 ? [s * 0.72, s * 0.85, KIND_ROCK] : null);
      }
    }
    // wall-foot rubble: big boulders where the floor meets the cliffs
    for (let i = 0; i < tr.n; i += 9) {
      for (const sgn of [-1, 1]) {
        if (r() > 0.55) continue;
        const d = sgn * (tr.halfFloor[i] + 4 + r() * 14);
        const [x, z] = side(i, d);
        const s = 2.2 + r() * 3.8;
        if (!this._ok(x, z, 4 + s)) continue;
        this._place(r() < 0.6 ? 'boulder' : 'slab', x, z, r() * 6.283, s, s * (0.7 + r() * 0.5), s * 0.2, [s * 0.75, s * 0.8, KIND_ROCK]);
      }
    }
    // hoodoo spires: in the wide washes and up on the mesas
    let tries = 0, spires = 0;
    while (spires < 150 && tries++ < 6000) {
      let x, z;
      if (r() < 0.3) { const i = (r() * tr.n) | 0; if (tr.halfFloor[i] < 26) continue; [x, z] = side(i, (r() < 0.5 ? -1 : 1) * (TRACK_HALF_WIDTH + 12 + r() * (tr.halfFloor[i] - TRACK_HALF_WIDTH - 14))); }
      else { x = (r() - 0.5) * 1700; z = (r() - 0.5) * 1700; if (te.wallFactor(x, z) < 0.98) continue; }
      if (te.flatness(x, z) < 0.94 || !this._ok(x, z, 12)) continue;
      const h = 7 + r() * 17, w = 1.6 + r() * 2.4;
      this._place('spire', x, z, r() * 6.283, w, h, 0.4, [w * 0.62, h * 0.97, KIND_ROCK]);
      spires++;
    }
    // mesa boulders
    for (let k = 0; k < 340; k++) {
      const x = (r() - 0.5) * 1760, z = (r() - 0.5) * 1760;
      if (te.wallFactor(x, z) < 0.9 || te.flatness(x, z) < 0.88 || !this._ok(x, z, 6)) continue;
      const s = 0.8 + Math.pow(r(), 2) * 4;
      this._place(r() < 0.6 ? 'boulder' : 'slab', x, z, r() * 6.283, s, s * (0.7 + r() * 0.5), s * 0.15, s > 1 ? [s * 0.72, s * 0.8, KIND_ROCK] : null);
    }
    // saguaros: canyon floor (denser) and gentle mesa ground
    for (let k = 0, placed = 0; k < 9000 && placed < 760; k++) {
      let x, z;
      if (r() < 0.6) { const i = (r() * tr.n) | 0; [x, z] = side(i, (r() < 0.5 ? -1 : 1) * (TRACK_HALF_WIDTH + 4 + r() * (tr.halfFloor[i] + 10))); }
      else { x = (r() - 0.5) * 1760; z = (r() - 0.5) * 1760; }
      if (te.flatness(x, z) < 0.86 || !this._ok(x, z, 4)) continue;
      const s = 0.65 + r() * 0.7;
      this._place('cactus', x, z, r() * 6.283, s, s * (0.85 + r() * 0.3), 0.1, [0.32 * s, 5.4 * s, KIND_THIN]);
      placed++;
    }
    // dry shrubs everywhere that is not too steep
    for (let k = 0, placed = 0; k < 14000 && placed < 2500; k++) {
      let x, z;
      if (r() < 0.55) { const i = (r() * tr.n) | 0; [x, z] = side(i, (r() < 0.5 ? -1 : 1) * (TRACK_HALF_WIDTH + 1.5 + r() * (tr.halfFloor[i] + 12))); }
      else { x = (r() - 0.5) * 1760; z = (r() - 0.5) * 1760; }
      if (te.flatness(x, z) < 0.8 || !this._ok(x, z, 1.2)) continue;
      const s = 0.55 + r() * 0.9;
      this._place('shrub', x, z, r() * 6.283, s, s * (0.7 + r() * 0.5), 0.05, null);
      placed++;
    }
  }

  /** Per frame (after the camera moved). */
  update(cameraPos, dt) {
    const L = this.setList;
    for (let i = 0; i < L.length; i++) L[i].update(cameraPos, dt);
  }

  get nearCounts() { const o = {}; for (const k in this.sets) o[k] = this.sets[k].pool.count; return o; }
}
