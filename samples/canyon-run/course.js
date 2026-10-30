// The race course on top of the terrain: the dirt ribbon draped along the track (tyre ruts, ragged
// alpha-to-coverage edges, chequered start line), checkpoint arches (one InstancedPool + an emissive strip pool
// whose instance colours show next / upcoming / passed) and the start-finish gantry with countdown lights.
// Pillars and towers are static box colliders.
import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { InstancedPool, makeCanvas, finishTexture, theme } from '@vibe/engine';
import { TRACK_HALF_WIDTH, GATE_HALF_WIDTH } from './track.js';
import { KIND_THIN } from './props.js';

const col = new THREE.Color();
function box(w, h, d, x, y, z, hex) {
  const g = new THREE.BoxGeometry(w, h, d).toNonIndexed();
  g.translate(x, y, z);
  g.deleteAttribute('uv');
  col.set(hex);
  const n = g.attributes.position.count, c = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) { c[i * 3] = col.r; c[i * 3 + 1] = col.g; c[i * 3 + 2] = col.b; }
  g.setAttribute('color', new THREE.BufferAttribute(c, 3));
  return g;
}

// ------------------------------------------------------------------ ribbon
const RIB_VERT_PARS = /* glsl */`
attribute vec3 aSide;
varying vec3 vSide;
varying vec3 vRPos;
varying vec3 vRNrm;
varying vec2 vRUv;
`;
const RIB_VERT = /* glsl */`
#include <begin_vertex>
vSide = aSide; vRUv = uv;
vRPos = ( modelMatrix * vec4( transformed, 1.0 ) ).xyz;
vRNrm = normalize( mat3( modelMatrix ) * objectNormal );
`;
const RIB_FRAG_PARS = /* glsl */`
uniform highp sampler2DArray tAlb;
uniform highp sampler2DArray tNrm;
uniform vec3 uDirt;
uniform float uLen, uHalf;
varying vec3 vSide;
varying vec3 vRPos;
varying vec3 vRNrm;
varying vec2 vRUv;
vec3 ribN; float ribRough;
float ribLuma( vec3 c ) { return dot( c, vec3( 0.299, 0.587, 0.114 ) ); }
`;
const RIB_MAP = /* glsl */`
{
	float lat = vRUv.x * uHalf, s = vRUv.y;
	vec2 uv = vRPos.xz / 2.4;
	vec4 a = texture( tAlb, vec3( uv, 0.0 ) ), nn = texture( tNrm, vec3( uv, 0.0 ) );
	vec4 a2 = texture( tAlb, vec3( uv * 0.21 + 0.5, 0.0 ) );
	float l = ribLuma( a.rgb ) * 0.6 + ribLuma( a2.rgb ) * 0.4;
	// ruts: the racing line's wheel tracks plus two fainter lanes, gently meandering
	float r = 0.0, dr = 0.0;
	for ( int k = 0; k < 6; k ++ ) {
		float c = k == 0 ? -0.86 : k == 1 ? 0.86 : k == 2 ? -3.3 : k == 3 ? -1.6 : k == 4 ? 1.6 : 3.3;
		float w = k < 2 ? 1.0 : 0.45;
		c += 0.45 * sin( s * 0.019 + float( k ) * 1.7 ) + 0.15 * sin( s * 0.07 + float( k ) );
		float q = ( lat - c ) / 0.3, g = exp( - q * q );
		r += w * g; dr += w * ( -2.0 * q / 0.3 ) * g;
	}
	r = min( r, 1.2 );
	vec3 c = uDirt * ( 0.62 + 0.75 * l / 0.32 );
	c *= 1.0 - 0.32 * r;
	c *= 0.93 + 0.07 * sin( s * 2.7 + lat * 0.4 ); // faint washboard
	// chequered start / finish line
	float sl = min( s, uLen - s );
	if ( sl < 1.6 ) {
		float ch = mod( floor( lat / 0.8 + 100.0 ) + floor( s / 0.8 ), 2.0 );
		c = mix( c, mix( vec3( 0.025 ), vec3( 0.85 ), ch ), smoothstep( 1.6, 1.3, sl ) );
	}
	diffuseColor.rgb = c;
	// ragged edge (alpha to coverage under MSAA, alpha test otherwise)
	float edge = abs( vRUv.x ) + ( ribLuma( a2.rgb ) - 0.3 ) * 0.9 + ( l - 0.3 ) * 0.25;
	diffuseColor.a = smoothstep( 1.02, 0.78, edge );
	vec2 t = nn.xy * 2.0 - 1.0;
	vec3 Ng = normalize( vRNrm );
	vec3 nW = normalize( Ng + vec3( t.x * 0.7, 0.0, t.y * 0.7 ) + vSide * ( dr * 0.05 ) );
	ribN = normalize( ( viewMatrix * vec4( nW, 0.0 ) ).xyz );
	ribRough = mix( 0.97, 0.82, clamp( r, 0.0, 1.0 ) );
}
`;

// ------------------------------------------------------------------ gantry texture (sign + chequers)
function gantryTexture(renderer) {
  const c = makeCanvas(1024, 512), g = c.getContext('2d');
  g.fillStyle = '#111317'; g.fillRect(0, 0, 1024, 256);
  g.fillStyle = theme.accent; g.fillRect(0, 0, 1024, 14); g.fillRect(0, 242, 1024, 14);
  g.font = '800 132px "Inter", system-ui, sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle';
  g.fillStyle = '#f4efe6'; g.fillText('CANYON RUN', 512, 132);
  for (let y = 0; y < 4; y++) for (let x = 0; x < 32; x++) { g.fillStyle = (x + y) % 2 ? '#0c0c0c' : '#f2f2f2'; g.fillRect(x * 32, 256 + y * 64, 32, 64); }
  return finishTexture(new THREE.CanvasTexture(c), renderer, true, false);
}

export class Course {
  /** @param {object} o  scene, renderer, terrain, track, physics, arrays (texture arrays) */
  constructor({ scene, renderer, terrain, track, physics, arrays }) {
    this.track = track; this.terrain = terrain;
    this.ribbon = this._ribbon(arrays);
    scene.add(this.ribbon);
    this._gates(scene, physics);
    this._gantry(scene, renderer, physics);
    this.setGateStates(-1);
  }

  _ribbon(arrays) {
    const tr = this.track, te = this.terrain, n = tr.n, X = 9; // vertices across
    const rings = n + 1, pos = new Float32Array(rings * X * 3), nor = new Float32Array(rings * X * 3), uv = new Float32Array(rings * X * 2), side = new Float32Array(rings * X * 3);
    const nv = new THREE.Vector3();
    for (let r = 0; r < rings; r++) {
      const i = r % n, s = r === n ? tr.length : tr.s[i];
      const sx = -tr.tz[i], sz = tr.tx[i];
      for (let k = 0; k < X; k++) {
        const u = (k / (X - 1)) * 2 - 1, lat = u * (TRACK_HALF_WIDTH + 0.4);
        const x = tr.x[i] + sx * lat, z = tr.z[i] + sz * lat, o = (r * X + k);
        pos[o * 3] = x; pos[o * 3 + 1] = te.heightAt(x, z) + 0.05; pos[o * 3 + 2] = z;
        te.normalAt(x, z, nv); nor[o * 3] = nv.x; nor[o * 3 + 1] = nv.y; nor[o * 3 + 2] = nv.z;
        uv[o * 2] = u; uv[o * 2 + 1] = s;
        side[o * 3] = sx; side[o * 3 + 1] = 0; side[o * 3 + 2] = sz;
      }
    }
    const idx = new Uint32Array(n * (X - 1) * 6);
    let o = 0;
    for (let r = 0; r < n; r++) for (let k = 0; k < X - 1; k++) {
      const a = r * X + k, b = a + 1, c = a + X, d = c + 1;
      // track runs along +s; keep CCW seen from above for either turning direction
      idx[o++] = a; idx[o++] = b; idx[o++] = c; idx[o++] = b; idx[o++] = d; idx[o++] = c;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
    g.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
    g.setAttribute('aSide', new THREE.BufferAttribute(side, 3));
    g.setIndex(new THREE.BufferAttribute(idx, 1));
    g.computeBoundingSphere();
    const u = {
      tAlb: { value: arrays.albedo }, tNrm: { value: arrays.normal }, uDirt: { value: new THREE.Color(0.30, 0.16, 0.085) },
      uLen: { value: tr.length }, uHalf: { value: TRACK_HALF_WIDTH + 0.4 },
    };
    const m = new THREE.MeshStandardMaterial({ roughness: 0.95, alphaToCoverage: true, alphaTest: 0.02, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2, envMapIntensity: 0.6 });
    m.onBeforeCompile = (sh) => {
      Object.assign(sh.uniforms, u);
      sh.vertexShader = sh.vertexShader.replace('#include <common>', `#include <common>\n${RIB_VERT_PARS}`).replace('#include <begin_vertex>', RIB_VERT);
      sh.fragmentShader = sh.fragmentShader.replace('#include <common>', `#include <common>\n${RIB_FRAG_PARS}`)
        .replace('#include <map_fragment>', RIB_MAP)
        .replace('#include <roughnessmap_fragment>', 'float roughnessFactor = ribRough;')
        .replace('#include <normal_fragment_maps>', 'normal = ribN;');
    };
    m.customProgramCacheKey = () => 'canyon-ribbon-v1';
    const mesh = new THREE.Mesh(g, m);
    mesh.name = 'trackRibbon'; mesh.receiveShadow = true; mesh.renderOrder = 1;
    mesh.matrixAutoUpdate = false;
    return mesh;
  }

  _gates(scene, physics) {
    const GW = GATE_HALF_WIDTH + 0.5;
    const body = mergeGeometries([
      box(1.0, 8.2, 1.0, -GW, 3.1, 0, 0x2b2d31), box(1.0, 8.2, 1.0, GW, 3.1, 0, 0x2b2d31),
      box(1.7, 0.7, 1.7, -GW, 0.15, 0, 0x8a8178), box(1.7, 0.7, 1.7, GW, 0.15, 0, 0x8a8178),
      box(2 * GW + 1.2, 0.95, 1.0, 0, 7.45, 0, 0x34363b),
      box(2 * GW - 1.0, 1.05, 0.16, 0, 6.35, 0, 0x16171a),
      box(2 * GW - 1.0, 0.18, 0.2, 0, 6.95, 0, theme.accent), box(2 * GW - 1.0, 0.18, 0.2, 0, 5.75, 0, theme.accent),
    ]);
    const glow = mergeGeometries([
      box(0.14, 5.2, 0.3, -GW + 0.56, 3.2, 0, 0xffffff), box(0.14, 5.2, 0.3, GW - 0.56, 3.2, 0, 0xffffff),
      box(2 * GW - 1.6, 0.14, 0.34, 0, 6.95 - 0.02, 0, 0xffffff),
    ]);
    const tr = this.track, cps = tr.checkpoints;
    const mat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.55, metalness: 0.35 });
    this.gatePool = new InstancedPool(body, mat, { capacity: cps.length, castShadow: true, name: 'gates', dynamic: false });
    this.glowPool = new InstancedPool(glow, new THREE.MeshBasicMaterial({ vertexColors: false }), { capacity: cps.length, colors: true, name: 'gateGlow', receiveShadow: false });
    this.gateSlots = [];
    for (let k = 1; k < cps.length; k++) {
      const cp = cps[k], y = this.terrain.heightAt(cp.x, cp.z);
      const i = this.gatePool.add(); this.gatePool.setTRS(i, cp.x, y, cp.z, cp.heading);
      const j = this.glowPool.add(); this.glowPool.setTRS(j, cp.x, y, cp.z, cp.heading);
      this.gateSlots[k] = j;
      this._pillarColliders(physics, cp, y, GW, 0.5, 8.0);
    }
    this.gatePool.updateBounds(); this.glowPool.updateBounds();
    scene.add(this.gatePool.mesh, this.glowPool.mesh);
  }

  _pillarColliders(physics, cp, y, GW, half, h) {
    const sx = -cp.tz, sz = cp.tx;
    const c = Math.abs(Math.cos(cp.heading)), s = Math.abs(Math.sin(cp.heading)), e = half * (c + s);
    for (const sg of [-1, 1]) {
      const x = cp.x + sx * GW * sg, z = cp.z + sz * GW * sg;
      physics.addStaticBox(x - e, y - 1, x + e, y + h, z - e, z + e, KIND_THIN);
    }
  }

  _gantry(scene, renderer, physics) {
    const cp = this.track.checkpoints[0], GW = GATE_HALF_WIDTH + 1.6, y = this.terrain.heightAt(cp.x, cp.z);
    const parts = [
      box(1.6, 11, 1.6, -GW, 4.5, 0, 0x1d1f23), box(1.6, 11, 1.6, GW, 4.5, 0, 0x1d1f23),
      box(2.2, 0.8, 2.2, -GW, 0.3, 0, 0x8a8178), box(2.2, 0.8, 2.2, GW, 0.3, 0, 0x8a8178),
      box(2 * GW + 1.6, 1.7, 1.5, 0, 9.6, 0, 0x26282d),
      box(0.25, 11, 1.62, -GW - 0.8, 4.5, 0, theme.accent), box(0.25, 11, 1.62, GW + 0.8, 4.5, 0, theme.accent),
    ];
    // diagonal truss braces on the towers (vertex-coloured slabs)
    for (let k = 0; k < 5; k++) for (const sg of [-1, 1]) parts.push(box(1.75, 0.16, 1.75, sg * GW, 0.9 + k * 1.9, 0, 0x3a3d43));
    const base = new THREE.Mesh(mergeGeometries(parts), new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.5, metalness: 0.4 }));
    // sign (top half of the atlas) on both faces + chequered banners (bottom half) under the beam
    const W = 2 * GW - 1.6, sign = [];
    const quad = (w, h, x, yy, z, flip, uv0, uv1) => {
      const g = new THREE.PlaneGeometry(w, h);
      const u = g.attributes.uv;
      u.setXY(0, 0, uv1); u.setXY(1, 1, uv1); u.setXY(2, 0, uv0); u.setXY(3, 1, uv0);
      if (flip) g.rotateY(Math.PI);
      g.translate(x, yy, z);
      return g;
    };
    sign.push(quad(W, W / 4, 0, 12.0, 0.05, false, 0.5, 1.0), quad(W, W / 4, 0, 12.0, -0.05, true, 0.5, 1.0));
    sign.push(quad(W, 1.25, 0, 8.05, 0.77, false, 0.0, 0.5), quad(W, 1.25, 0, 8.05, -0.77, true, 0.0, 0.5));
    sign.push(box(W, W / 4 + 0.3, 0.08, 0, 12.0, 0, 0x111317), box(0.3, 1.6, 0.3, -W * 0.3, 10.9, 0, 0x2a2c30), box(0.3, 1.6, 0.3, W * 0.3, 10.9, 0, 0x2a2c30));
    const signGeo = mergeGeometries(sign.slice(0, 4));
    const signMesh = new THREE.Mesh(signGeo, new THREE.MeshStandardMaterial({ map: gantryTexture(renderer), roughness: 0.6, side: THREE.DoubleSide, emissive: 0xffffff, emissiveIntensity: 0.0 }));
    const signFrame = new THREE.Mesh(mergeGeometries(sign.slice(4)), base.material);
    const g = new THREE.Group();
    g.add(base, signMesh, signFrame);
    // countdown lights: 5 pods on each face of the beam
    const lightGeo = new THREE.CylinderGeometry(0.36, 0.36, 0.2, 16).rotateX(Math.PI / 2);
    this.lights = new InstancedPool(lightGeo, new THREE.MeshBasicMaterial(), { capacity: 10, colors: true, name: 'countdownLights', receiveShadow: false });
    for (let f = 0; f < 2; f++) for (let k = 0; k < 5; k++) {
      const i = this.lights.add();
      this.lights.setTRS(i, (k - 2) * 1.5, 9.6, f ? -0.82 : 0.82, 0);
    }
    this.lights.updateBounds();
    g.add(this.lights.mesh);
    g.position.set(cp.x, y, cp.z); g.rotation.y = cp.heading;
    g.traverse((o) => { if (o.isMesh) { o.castShadow = o !== this.lights.mesh; o.receiveShadow = true; } });
    g.updateMatrixWorld(true);
    scene.add(g);
    this.gantry = g;
    this._pillarColliders(physics, cp, y, GW, 0.8, 11);
    this.setCountdown(0);
  }

  /** Countdown lights: n red (1..5), or 'go' (all green), 0 = off. */
  setCountdown(n) {
    const L = this.lights;
    for (let f = 0; f < 2; f++) for (let k = 0; k < 5; k++) {
      const i = f * 5 + k;
      if (n === 'go') L.setColor(i, 0.5, 9, 1.2);
      else if (k < n) L.setColor(i, 10, 0.35, 0.2);
      else L.setColor(i, 0.06, 0.02, 0.02);
    }
  }

  /** Gate colours: `next` checkpoint index highlighted (accent), earlier ones passed (green), -1 = idle. */
  setGateStates(next, passedAll = false) {
    const P = this.glowPool, cps = this.track.checkpoints, a = new THREE.Color(theme.accent);
    for (let k = 1; k < cps.length; k++) {
      const i = this.gateSlots[k];
      if (next < 0) P.setColor(i, 1.2, 1.1, 1.0);
      else if (passedAll || (k < next && next !== 0)) P.setColor(i, 0.25, 2.2, 0.6);
      else if (k === next) P.setColor(i, a.r * 9, a.g * 9, a.b * 9);
      else P.setColor(i, 0.9, 0.85, 0.8);
    }
  }
}
