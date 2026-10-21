// Desert canyon heightfield: a 2 km x 2 km, 4 m grid of mesas (terraced fBm) with a winding canyon carved along
// the track spline (flat floor across the track, ragged terraced walls), a constant-height rim at the map border
// and a low-res far ring of buttes to the horizon. Built as 4 x 4 chunk meshes sharing one splat material
// (engine TextureSets array: dirt / rock / aerial macro; strata banding by height, biplanar rock on steep walls).
// heightAt() interpolates exactly like the mesh triangles, so physics and visuals agree to the millimetre.
import * as THREE from 'three';
import { Simplex, Noise } from '@vibe/engine';
import { TRACK_HALF_WIDTH } from './track.js';
const { clamp, smoothstep } = Noise;

export const WORLD_HALF = 1024;      // play area half size (m)
export const CELL = 4;               // grid spacing (m)
const N = (WORLD_HALF * 2) / CELL;   // 512 cells
const V = N + 1;                     // 513 vertices per side
const CHUNK = 128;                   // cells per chunk side (4 x 4 chunks)
export const RIM = 150;              // constant height of the map border (m)
const REACH = 112;                   // canyon influence radius around the centreline (m)

const yieldFrame = () => new Promise((r) => setTimeout(r, 0));

export class Terrain {
  constructor(track, { seed = 11 } = {}) {
    this.track = track;
    this.noise = new Simplex(seed);
    this.H = new Float32Array(V * V);      // heights
    this.D = new Float32Array(V * V);      // distance to the centreline (m, capped at REACH)
    this.S = new Float32Array(V * V);      // arc length of the nearest centreline point
    this.T = new Float32Array(V * V);      // wall factor 0 (floor) .. 1 (mesa)
    this.chunks = [];
    this.mesh = null;
  }

  // ------------------------------------------------------------------ generation
  async build(onProgress = () => {}) {
    const t0 = performance.now();
    this._distanceField();
    onProgress(0.25);
    await yieldFrame();
    this._heights();
    onProgress(0.6);
    await yieldFrame();
    this._normals();
    this.ms = performance.now() - t0;
    onProgress(1);
  }

  _distanceField() {
    const tr = this.track, D = this.D, S = this.S;
    D.fill(REACH); S.fill(0);
    const step = 2; // every 2nd polyline point (4 m segments)
    const n = tr.n;
    for (let k = 0; k < n; k += step) {
      const a = k, b = (k + step) % n;
      const ax = tr.x[a], az = tr.z[a], bx = tr.x[b], bz = tr.z[b];
      const sa = tr.s[a], sb = b === 0 ? tr.length : tr.s[b];
      const ex = bx - ax, ez = bz - az, el2 = ex * ex + ez * ez || 1;
      const i0 = Math.max(0, Math.floor((Math.min(ax, bx) - REACH + WORLD_HALF) / CELL));
      const i1 = Math.min(N, Math.ceil((Math.max(ax, bx) + REACH + WORLD_HALF) / CELL));
      const j0 = Math.max(0, Math.floor((Math.min(az, bz) - REACH + WORLD_HALF) / CELL));
      const j1 = Math.min(N, Math.ceil((Math.max(az, bz) + REACH + WORLD_HALF) / CELL));
      for (let j = j0; j <= j1; j++) {
        const z = j * CELL - WORLD_HALF;
        for (let i = i0; i <= i1; i++) {
          const x = i * CELL - WORLD_HALF;
          let t = ((x - ax) * ex + (z - az) * ez) / el2;
          t = t < 0 ? 0 : t > 1 ? 1 : t;
          const px = ax + ex * t - x, pz = az + ez * t - z;
          const d = Math.sqrt(px * px + pz * pz);
          const v = j * V + i;
          if (d < D[v]) { D[v] = d; S[v] = sa + (sb - sa) * t; }
        }
      }
    }
  }

  /** Mesa / plateau height away from the canyon (terraced fBm + buttes), before the rim blend. */
  mesa(x, z) {
    const ns = this.noise;
    let m = 58 + 30 * ns.fbm(x / 640, z / 640, 3) + 16 * ns.ridged(x / 260 + 3.1, z / 260 - 1.7, 3);
    const L = 13, q = m / L, k = Math.floor(q), f = q - k;
    m = (k + smoothstep(0.62, 0.96, f)) * L;                  // crisp mesa steps
    return m + 1.6 * ns.noise(x / 23, z / 23) + 0.6 * ns.noise(x / 7, z / 7);
  }

  /** Rim blend weight: 0 inside, 1 at (and beyond) the play-area border. */
  rimWeight(x, z) {
    const q = Math.max(Math.abs(x), Math.abs(z));
    return smoothstep(800 + 50 * this.noise.noise(x / 170, z / 170), 1000, q);
  }

  _heights() {
    const tr = this.track, ns = this.noise, H = this.H, D = this.D, S = this.S, T = this.T;
    for (let j = 0; j < V; j++) {
      const z = j * CELL - WORLD_HALF;
      for (let i = 0; i < V; i++) {
        const x = i * CELL - WORLD_HALF, v = j * V + i;
        const M = this.mesa(x, z);
        let h = M, t = 1;
        const d = D[v];
        if (d < REACH) {
          const s = S[v], e = tr.elevation(s), w0 = tr.floorHalfWidth(s);
          const dn = d + 7 * ns.noise(x / 46, z / 46) + 2.5 * ns.noise(x / 12, z / 12);
          const wallW = 30 + 14 * ns.noise(x / 310, z / 310);
          t = clamp((dn - w0) / wallW, 0, 1);
          const tt = t * 3, kk = Math.floor(tt), ff = tt - kk;
          const terr = Math.min(1, (kk + smoothstep(0.3, 0.92, ff)) / 3);
          const prof = 0.72 * terr + 0.28 * smoothstep(0, 1, t);
          // floor: dunes / ripples outside the driving strip, dead flat across the track
          const flat = smoothstep(TRACK_HALF_WIDTH + 1.5, TRACK_HALF_WIDTH + 9, d);
          const dune = flat * (0.7 * ns.noise(x / 34, z / 34) + 0.25 * ns.noise(x / 9, z / 9)) * (1 - t);
          h = e + dune + (M - e) * prof;
        }
        const rw = this.rimWeight(x, z);
        H[v] = h + (RIM - h) * rw;
        T[v] = Math.max(t, rw);
      }
    }
  }

  _normals() {
    const H = this.H, nrm = this.NRM = new Float32Array(V * V * 3);
    for (let j = 0; j < V; j++) for (let i = 0; i < V; i++) {
      const l = H[j * V + Math.max(0, i - 1)], r = H[j * V + Math.min(N, i + 1)];
      const u = H[Math.max(0, j - 1) * V + i], d = H[Math.min(N, j + 1) * V + i];
      const sx = (i === 0 || i === N) ? CELL : 2 * CELL, sz = (j === 0 || j === N) ? CELL : 2 * CELL;
      let nx = -(r - l) / sx, ny = 1, nz = -(d - u) / sz;
      const k = 1 / Math.hypot(nx, ny, nz); nx *= k; ny *= k; nz *= k;
      const o = (j * V + i) * 3; nrm[o] = nx; nrm[o + 1] = ny; nrm[o + 2] = nz;
    }
  }

  // ------------------------------------------------------------------ queries (physics, placement)
  /** Height at (x, z), interpolated on the same triangles as the mesh (diagonal from (i+1, j) to (i, j+1)). */
  heightAt(x, z) {
    let gx = (x + WORLD_HALF) / CELL, gz = (z + WORLD_HALF) / CELL;
    if (!(gx > 0)) gx = 0; else if (gx > N - 1e-4) gx = N - 1e-4;
    if (!(gz > 0)) gz = 0; else if (gz > N - 1e-4) gz = N - 1e-4;
    const i = gx | 0, j = gz | 0, fx = gx - i, fz = gz - j, H = this.H, v = j * V + i;
    if (fx + fz <= 1) {
      const a = H[v];
      return a + (H[v + 1] - a) * fx + (H[v + V] - a) * fz;
    }
    const d = H[v + V + 1];
    return d + (H[v + V] - d) * (1 - fx) + (H[v + 1] - d) * (1 - fz);
  }
  /** Bilinear lookup of a per-vertex field. */
  _field(F, x, z) {
    const gx = clamp((x + WORLD_HALF) / CELL, 0, N - 1e-4), gz = clamp((z + WORLD_HALF) / CELL, 0, N - 1e-4);
    const i = gx | 0, j = gz | 0, fx = gx - i, fz = gz - j, v = j * V + i;
    return (F[v] * (1 - fx) + F[v + 1] * fx) * (1 - fz) + (F[v + V] * (1 - fx) + F[v + V + 1] * fx) * fz;
  }
  trackDistance(x, z) { return this._field(this.D, x, z); }
  wallFactor(x, z) { return this._field(this.T, x, z); }
  /** Up component of the surface normal (1 flat .. 0 vertical). */
  flatness(x, z) {
    const e = 2;
    const hx = this.heightAt(x + e, z) - this.heightAt(x - e, z), hz = this.heightAt(x, z + e) - this.heightAt(x, z - e);
    return 1 / Math.hypot(hx / (2 * e), 1, hz / (2 * e));
  }
  normalAt(x, z, out) {
    const e = 0.75;
    const hx = this.heightAt(x + e, z) - this.heightAt(x - e, z), hz = this.heightAt(x, z + e) - this.heightAt(x, z - e);
    return out.set(-hx / (2 * e), 1, -hz / (2 * e)).normalize();
  }

  // ------------------------------------------------------------------ meshes
  buildMeshes(material) {
    const group = new THREE.Group();
    group.name = 'terrain';
    const H = this.H, NRM = this.NRM, D = this.D, T = this.T, S = this.S, L = this.track.length;
    const cv = CHUNK + 1;
    const index = new Uint16Array(CHUNK * CHUNK * 6);
    let o = 0;
    for (let j = 0; j < CHUNK; j++) for (let i = 0; i < CHUNK; i++) {
      const a = j * cv + i, b = a + 1, c = a + cv, d = c + 1;
      index[o++] = a; index[o++] = c; index[o++] = b;
      index[o++] = b; index[o++] = c; index[o++] = d;
    }
    const idxAttr = new THREE.BufferAttribute(index, 1);
    for (let cj = 0; cj < N / CHUNK; cj++) for (let ci = 0; ci < N / CHUNK; ci++) {
      const pos = new Float32Array(cv * cv * 3), nor = new Float32Array(cv * cv * 3), dat = new Float32Array(cv * cv * 3);
      for (let j = 0; j < cv; j++) for (let i = 0; i < cv; i++) {
        const gi = ci * CHUNK + i, gj = cj * CHUNK + j, v = gj * V + gi, k = (j * cv + i) * 3;
        pos[k] = gi * CELL - WORLD_HALF; pos[k + 1] = H[v]; pos[k + 2] = gj * CELL - WORLD_HALF;
        nor[k] = NRM[v * 3]; nor[k + 1] = NRM[v * 3 + 1]; nor[k + 2] = NRM[v * 3 + 2];
        dat[k] = D[v]; dat[k + 1] = T[v]; dat[k + 2] = S[v] / L;
      }
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      g.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
      g.setAttribute('aData', new THREE.BufferAttribute(dat, 3));
      g.setIndex(idxAttr);
      g.computeBoundingSphere();
      g.computeBoundingBox();
      const m = new THREE.Mesh(g, material);
      m.name = `terrain-${ci}-${cj}`;
      m.receiveShadow = true; m.castShadow = true;
      m.matrixAutoUpdate = false; m.updateMatrix();
      group.add(m);
      this.chunks.push(m);
    }
    this.mesh = group;
    return group;
  }

  /** Low-res ring of buttes from the rim to the horizon (one draw, no shadows). */
  buildFarRing(material) {
    const ns = this.noise, SP = 128, EXT = 6144, n = (EXT * 2) / SP, vv = n + 1;
    const pos = new Float32Array(vv * vv * 3), dat = new Float32Array(vv * vv * 3), hts = new Float32Array(vv * vv);
    for (let j = 0; j < vv; j++) for (let i = 0; i < vv; i++) {
      const x = i * SP - EXT, z = j * SP - EXT, v = j * vv + i;
      const q = Math.max(Math.abs(x), Math.abs(z)) - WORLD_HALF;
      let h = RIM;
      if (q > 0) {
        const r = ns.ridged(x / 1300 + 7.7, z / 1300 - 2.2, 4);
        let b = 260 * Math.max(0, r - 0.45);
        const L = 40, k = Math.floor(b / L); b = (k + smoothstep(0.55, 0.95, b / L - k)) * L; // buttes
        h = RIM - 95 * smoothstep(0, 1400, q) + b * smoothstep(100, 900, q) + 18 * ns.fbm(x / 700, z / 700, 2);
      }
      hts[v] = h;
      pos[v * 3] = x; pos[v * 3 + 1] = h; pos[v * 3 + 2] = z;
      dat[v * 3] = 999; dat[v * 3 + 1] = 1; dat[v * 3 + 2] = 0;
    }
    const nor = new Float32Array(vv * vv * 3);
    for (let j = 0; j < vv; j++) for (let i = 0; i < vv; i++) {
      const l = hts[j * vv + Math.max(0, i - 1)], r = hts[j * vv + Math.min(n, i + 1)];
      const u = hts[Math.max(0, j - 1) * vv + i], d = hts[Math.min(n, j + 1) * vv + i];
      let nx = -(r - l) / (2 * SP), ny = 1, nz = -(d - u) / (2 * SP);
      const k = 1 / Math.hypot(nx, ny, nz), v = (j * vv + i) * 3;
      nor[v] = nx * k; nor[v + 1] = ny * k; nor[v + 2] = nz * k;
    }
    const idx = [];
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const x0 = i * SP - EXT, z0 = j * SP - EXT;
      if (x0 >= -WORLD_HALF && x0 + SP <= WORLD_HALF && z0 >= -WORLD_HALF && z0 + SP <= WORLD_HALF) continue; // play area
      const a = j * vv + i, b = a + 1, c = a + vv, d = c + 1;
      idx.push(a, c, b, b, c, d);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
    g.setAttribute('aData', new THREE.BufferAttribute(dat, 3));
    g.setIndex(idx);
    g.computeBoundingSphere();
    const m = new THREE.Mesh(g, material);
    m.name = 'farRing'; m.matrixAutoUpdate = false; m.receiveShadow = false; m.castShadow = false;
    return m;
  }
}

// ------------------------------------------------------------------ splat material
const TERRAIN_VERT_PARS = /* glsl */`
attribute vec3 aData;
varying vec3 vData;
varying vec3 vWPos;
varying vec3 vWNrm;
`;
const TERRAIN_VERT = /* glsl */`
#include <begin_vertex>
vData = aData;
vWPos = ( modelMatrix * vec4( transformed, 1.0 ) ).xyz;
vWNrm = normalize( mat3( modelMatrix ) * objectNormal );
`;
const TERRAIN_FRAG_PARS = /* glsl */`
uniform highp sampler2DArray tAlb;
uniform highp sampler2DArray tNrm;
uniform vec4 uTile;      // dirt tile (m), rock tile (m), macro tile (m), track half width
uniform vec3 uSand, uVarnish, uShoulder;
uniform vec3 uStrataA, uStrataB, uStrataC;
varying vec3 vData;
varying vec3 vWPos;
varying vec3 vWNrm;
vec3 trnN; float trnRough; float trnAO;
float trnHash( vec2 p ) { return fract( sin( dot( p, vec2( 127.1, 311.7 ) ) ) * 43758.5453 ); }
float trnLuma( vec3 c ) { return dot( c, vec3( 0.299, 0.587, 0.114 ) ); }
vec3 trnStrata( float y, float n ) {
	float b = y * 0.115 + n * 0.9;
	float f = fract( b ), k = floor( b );
	float pick = fract( sin( k * 12.9898 ) * 43758.5453 );
	vec3 c = pick < 0.4 ? uStrataA : pick < 0.78 ? uStrataB : uStrataC;
	return c * ( 0.86 + 0.22 * smoothstep( 0.0, 0.25, f ) * smoothstep( 1.0, 0.7, f ) );
}
`;
const TERRAIN_MAP = /* glsl */`
{
	vec3 P = vWPos;
	vec3 Ng = normalize( vWNrm );
	float slope = 1.0 - Ng.y;
	float dTrack = vData.x, wall = vData.y;
	float dist = length( P - cameraPosition );
	// macro variation (90 m aerial set) breaks tiling everywhere
	vec4 mac = texture( tAlb, vec3( P.xz / uTile.z, 2.0 ) );
	float macro = 0.72 + 0.56 * trnLuma( mac.rgb ) / 0.42;
	// sand / dirt (top projection)
	vec2 uvD = P.xz / uTile.x;
	vec4 dA = texture( tAlb, vec3( uvD, 0.0 ) );
	vec4 dN = texture( tNrm, vec3( uvD, 0.0 ) );
	vec4 dA2 = texture( tAlb, vec3( uvD * 0.23 + 0.37, 0.0 ) );
	float dl = trnLuma( dA.rgb ) * 0.65 + trnLuma( dA2.rgb ) * 0.35;
	vec3 sand = mix( uSand, uVarnish, smoothstep( 0.25, 0.9, wall ) * 0.55 ) * ( 0.55 + 0.9 * dl / 0.32 );
	// compacted, lighter shoulder along the track (where the ribbon fades out)
	float sh = 1.0 - smoothstep( uTile.w - 1.0, uTile.w + 5.5, dTrack );
	sand = mix( sand, uShoulder * ( 0.7 + 0.6 * dl / 0.32 ), sh * 0.85 );
	// rock: biplanar on steep walls, coloured by height strata
	vec3 bw = pow( abs( Ng ), vec3( 4.0 ) ); bw /= ( bw.x + bw.y + bw.z );
	// (skipped on the flat floor: 6 fetches saved on most of the screen; explicit gradients keep mips valid in the branch)
	vec2 uX = P.zy / uTile.y, uY = P.xz / uTile.y, uZ = P.xy / uTile.y;
	vec2 gXx = dFdx( uX ), gXy = dFdy( uX ), gYx = dFdx( uY ), gYy = dFdy( uY ), gZx = dFdx( uZ ), gZy = dFdy( uZ );
	vec4 rX = vec4( 0.36 ), rY = vec4( 0.36 ), rZ = vec4( 0.36 ), nX = vec4( 0.5, 0.5, 0.8, 1.0 ), nY = nX, nZ = nX;
	if ( slope > 0.08 ) {
		rX = textureGrad( tAlb, vec3( uX, 1.0 ), gXx, gXy ); rY = textureGrad( tAlb, vec3( uY, 1.0 ), gYx, gYy ); rZ = textureGrad( tAlb, vec3( uZ, 1.0 ), gZx, gZy );
		nX = textureGrad( tNrm, vec3( uX, 1.0 ), gXx, gXy ); nY = textureGrad( tNrm, vec3( uY, 1.0 ), gYx, gYy ); nZ = textureGrad( tNrm, vec3( uZ, 1.0 ), gZx, gZy );
	}
	vec4 rA = rX * bw.x + rY * bw.y + rZ * bw.z;
	float sn = sin( P.x * 0.011 + P.z * 0.007 ) * 0.5 + sin( P.x * 0.031 - P.z * 0.023 ) * 0.25;
	vec3 rock = trnStrata( P.y, sn ) * ( 0.45 + 1.1 * trnLuma( rA.rgb ) / 0.36 );
	// noisy threshold hides the 4 m triangle pattern along the wall foot; scree (fallen rock) tints the transition
	float rn = sin( P.x * 0.21 + sin( P.z * 0.17 ) * 2.0 ) * 0.5 + sin( P.z * 0.13 - P.x * 0.05 ) * 0.5;
	float rockW = smoothstep( 0.2, 0.5, slope + rn * 0.07 + ( trnLuma( dA2.rgb ) - 0.3 ) * 0.4 );
	sand = mix( sand, rock * 0.8, smoothstep( 0.12, 0.3, slope ) * ( 1.0 - rockW ) * 0.45 );
	vec3 alb = mix( sand, rock, rockW ) * macro;
	diffuseColor.rgb = alb;
	// normals: UDN blends of the tangent-space maps per projection
	vec2 tY = mix( dN.xy, nY.xy, rockW ) * 2.0 - 1.0;
	vec2 tX = nX.xy * 2.0 - 1.0, tZ = nZ.xy * 2.0 - 1.0;
	float nStr = mix( 0.9, 1.4, rockW ) * ( 1.0 - smoothstep( 120.0, 600.0, dist ) * 0.6 );
	vec3 pX = vec3( 0.0, tX.y, tX.x ) * sign( Ng.x );
	vec3 pY = vec3( tY.x, 0.0, tY.y );
	vec3 pZ = vec3( tZ.x, tZ.y, 0.0 ) * sign( Ng.z );
	trnN = normalize( Ng + nStr * ( pX * bw.x + pY * bw.y + pZ * bw.z ) );
	trnRough = clamp( mix( dN.b, nY.b * bw.y + nX.b * bw.x + nZ.b * bw.z, rockW ) * 0.35 + 0.68, 0.55, 1.0 );
	trnAO = mix( dA.a, rA.a, rockW );
	diffuseColor.rgb *= 0.55 + 0.45 * trnAO;
}
`;

/**
 * MeshStandardMaterial patched into the desert splat. arrays: makeTextureArray() result with layers
 * [dirt, rock, aerial macro]; sizes come from the manifest (metres per tile).
 */
export function createTerrainMaterial(arrays) {
  const m = new THREE.MeshStandardMaterial({ roughness: 0.92, metalness: 0, envMapIntensity: 0.65 });
  const u = {
    tAlb: { value: arrays.albedo }, tNrm: { value: arrays.normal },
    uTile: { value: new THREE.Vector4(3.4, 6.5, Math.max(40, arrays.sizes[2] || 90), TRACK_HALF_WIDTH) },
    // linear albedos (desert sand ~0.3-0.45 reflectance; red sandstone strata)
    uSand: { value: new THREE.Color(0.46, 0.27, 0.14) }, uVarnish: { value: new THREE.Color(0.30, 0.14, 0.07) },
    uShoulder: { value: new THREE.Color(0.50, 0.34, 0.20) },
    uStrataA: { value: new THREE.Color(0.38, 0.13, 0.055) }, uStrataB: { value: new THREE.Color(0.50, 0.23, 0.095) }, uStrataC: { value: new THREE.Color(0.55, 0.36, 0.19) },
  };
  m.userData.uniforms = u;
  m.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, u);
    sh.vertexShader = sh.vertexShader.replace('#include <common>', `#include <common>\n${TERRAIN_VERT_PARS}`).replace('#include <begin_vertex>', TERRAIN_VERT);
    sh.fragmentShader = sh.fragmentShader.replace('#include <common>', `#include <common>\n${TERRAIN_FRAG_PARS}`)
      .replace('#include <map_fragment>', TERRAIN_MAP)
      .replace('#include <roughnessmap_fragment>', 'float roughnessFactor = trnRough;')
      .replace('#include <normal_fragment_maps>', 'normal = normalize( ( viewMatrix * vec4( trnN, 0.0 ) ).xyz );');
  };
  m.customProgramCacheKey = () => 'canyon-terrain-v1';
  return m;
}
