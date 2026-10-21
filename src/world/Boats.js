// Vessels on the water: ferries looping from a terminal (docking with a dwell at the terminal), a container ship
// on a long loop, and a fleet of sailboats on small loops in open water with heel / pitch / roll from the wind and
// the swell. Every vessel trails an additive foam wake (Kelvin V + turbulent centre wash) that rides the same
// Gerstner swell as the water. The routes are the game's (options: ferryDock + ferryRoute, shipRoute, sailZones).
// Draw calls: ferries, ship, sailboats (InstancedMesh each) + wakes (one InstancedMesh) = 4. ~6k triangles.
// Routes are validated at build time against isLand + the shoreline distance field and never cross the `bridge`
// segments (lines to keep clear of, e.g. a bridge with piers in the water), so hulls never cut through land or piers.
// The boat geometry builders are exported for the moored boats in shore.js.
import * as THREE from 'three';
import { waveHeight, waveSlope, waveUniforms, WAVE_GLSL } from './Water.js';
import { mulberry32, clamp } from './geom.js';

// ====================================================================== geometry helpers
class GeoBuf {
  constructor() { this.pos = []; this.nor = []; this.col = []; this.emi = []; }
  tri(a, b, c, col, emi = 0) {
    const ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2], vx = c[0] - a[0], vy = c[1] - a[1], vz = c[2] - a[2];
    let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const l = Math.hypot(nx, ny, nz) || 1; nx /= l; ny /= l; nz /= l;
    for (const p of [a, b, c]) { this.pos.push(p[0], p[1], p[2]); this.nor.push(nx, ny, nz); this.col.push(col[0], col[1], col[2]); this.emi.push(emi); }
  }
  quad(a, b, c, d, col, emi = 0) { this.tri(a, b, c, col, emi); this.tri(a, c, d, col, emi); }
  /** Axis-aligned box (x: beam, y: up, z: length, bow at -z). */
  box(x0, y0, z0, x1, y1, z1, col, emi = 0, bottom = false) {
    const p = (x, y, z) => [x, y, z];
    this.quad(p(x0, y1, z0), p(x0, y1, z1), p(x1, y1, z1), p(x1, y1, z0), col, emi); // top
    this.quad(p(x0, y0, z1), p(x1, y0, z1), p(x1, y1, z1), p(x0, y1, z1), col, emi); // +z
    this.quad(p(x1, y0, z0), p(x0, y0, z0), p(x0, y1, z0), p(x1, y1, z0), col, emi); // -z
    this.quad(p(x1, y0, z1), p(x1, y0, z0), p(x1, y1, z0), p(x1, y1, z1), col, emi); // +x
    this.quad(p(x0, y0, z0), p(x0, y0, z1), p(x0, y1, z1), p(x0, y1, z0), col, emi); // -x
    if (bottom) this.quad(p(x0, y0, z0), p(x1, y0, z0), p(x1, y0, z1), p(x0, y0, z1), col, emi);
  }
  /** Double-sided triangle (sails). */
  tri2(a, b, c, col) { this.tri(a, b, c, col); this.tri(a, c, b, col); }
  geometry() {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(this.pos, 3));
    g.setAttribute('normal', new THREE.Float32BufferAttribute(this.nor, 3));
    g.setAttribute('color', new THREE.Float32BufferAttribute(this.col, 3));
    g.setAttribute('aEmit', new THREE.Float32BufferAttribute(this.emi, 1));
    g.computeBoundingSphere();
    return g;
  }
}

/**
 * Hull from stations along z (bow at -L/2): each station [z fraction 0..1 from bow, half-beam fraction, deck
 * height fraction]. Sides split at the waterline into two colours (boot stripe optional).
 */
function hull(B, { L, beam, freeboard, draft, stations, top, side, bottomCol, stripe = null, stripeY = null, deckCol }) {
  const S = stations.map(([f, hb, dh]) => ({ z: -L / 2 + f * L, hb: hb * beam / 2, yt: freeboard * dh }));
  for (let i = 0; i + 1 < S.length; i++) {
    const a = S[i], b = S[i + 1];
    for (const sgn of [-1, 1]) {
      const pa = (y, s) => [sgn * s.hb * (y < 0 ? 0.82 : 1), y, s.z];
      // below the waterline (tapering), boot band, topsides
      const yb = -draft;
      const ys = stripeY ?? 0.25;
      const q = (y0, y1, col) => {
        const A0 = pa(y0, a), A1 = pa(y1, a), B0 = pa(y0, b), B1 = pa(y1, b);
        if (sgn > 0) B.quad(A0, B0, B1, A1, col); else B.quad(A0, A1, B1, B0, col);
      };
      q(yb, 0.05, bottomCol);
      if (stripe) { q(0.05, ys, side); q(ys, ys + stripe.h, stripe.col); q(ys + stripe.h, Math.min(a.yt, b.yt), side); }
      else q(0.05, Math.min(a.yt, b.yt), side);
    }
    // deck
    const da = [[-a.hb, a.yt, a.z], [a.hb, a.yt, a.z]], db = [[-b.hb, b.yt, b.z], [b.hb, b.yt, b.z]];
    B.quad(da[0], db[0], db[1], da[1], deckCol || top);
    // bottom
    B.quad([-a.hb * 0.82, -draft, a.z], [a.hb * 0.82, -draft, a.z], [b.hb * 0.82, -draft, b.z], [-b.hb * 0.82, -draft, b.z], bottomCol);
  }
  // transom
  const e = S[S.length - 1];
  B.quad([e.hb * 0.82, -draft, e.z], [-e.hb * 0.82, -draft, e.z], [-e.hb, e.yt, e.z], [e.hb, e.yt, e.z], side);
}

const C = (r, g, b) => [r, g, b];
const WHITE = C(0.82, 0.83, 0.82), OFFW = C(0.74, 0.74, 0.71), GLASS = C(0.05, 0.07, 0.09), DARK = C(0.06, 0.06, 0.065);

/** Bay ferry (~42 m catamaran-style, San Francisco Bay style): white with a navy / red stripe, two decks with window bands. */
export function ferryGeometry() {
  const B = new GeoBuf();
  const L = 42, beam = 11;
  hull(B, {
    L, beam, freeboard: 2.4, draft: 1.6, side: WHITE, bottomCol: C(0.12, 0.13, 0.15), deckCol: C(0.45, 0.46, 0.46),
    stripe: { h: 0.55, col: C(0.06, 0.16, 0.36) }, stripeY: 0.6,
    stations: [[0, 0.18, 1.15], [0.07, 0.62, 1.08], [0.18, 0.9, 1.02], [0.35, 1, 1], [0.8, 1, 1], [1, 0.96, 1]],
  });
  // main cabin
  const red = C(0.55, 0.08, 0.07);
  B.box(-4.6, 2.4, -12, 4.6, 4.9, 15, WHITE);
  B.box(-4.65, 3.1, -11.5, 4.65, 4.3, 14.5, GLASS, 1);
  B.box(-4.62, 2.4, -12.05, 4.62, 2.65, 15.05, red);
  // upper deck cabin
  B.box(-3.8, 4.9, -7, 3.8, 7.0, 9, WHITE);
  B.box(-3.85, 5.5, -6.5, 3.85, 6.5, 8.5, GLASS, 1);
  // wheelhouse
  B.box(-2.6, 7.0, -6.5, 2.6, 8.8, -2.5, WHITE);
  B.box(-2.65, 7.6, -6.6, 2.65, 8.4, -2.4, GLASS, 1);
  B.box(-0.12, 8.8, -4.6, 0.12, 11.5, -4.3, DARK);
  // nav lights (emissive at night)
  B.box(-2.75, 8.0, -4.6, -2.6, 8.2, -4.4, C(0.9, 0.05, 0.05), 3);
  B.box(2.6, 8.0, -4.6, 2.75, 8.2, -4.4, C(0.05, 0.9, 0.1), 3);
  B.box(-0.1, 11.5, -4.55, 0.1, 11.7, -4.35, C(1, 1, 0.9), 3);
  return B.geometry();
}

/** Container ship (~210 m): dark hull with red bottom, white bridge house aft, stacked containers. */
export function shipGeometry() {
  const B = new GeoBuf();
  const L = 210, beam = 32;
  hull(B, {
    L, beam, freeboard: 11, draft: 9, side: C(0.08, 0.1, 0.14), bottomCol: C(0.4, 0.07, 0.06), deckCol: C(0.25, 0.26, 0.27),
    stripe: { h: 0.9, col: C(0.75, 0.75, 0.72) }, stripeY: 9.5,
    stations: [[0, 0.1, 1.2], [0.06, 0.55, 1.12], [0.15, 0.92, 1.04], [0.3, 1, 1], [0.85, 1, 1], [1, 0.9, 1.02]],
  });
  // containers (bow at -z): bays from z=-80 to +55, bridge house at z 62..80
  const cols = [C(0.55, 0.12, 0.08), C(0.1, 0.25, 0.5), C(0.62, 0.6, 0.55), C(0.15, 0.4, 0.3), C(0.7, 0.45, 0.1), C(0.4, 0.4, 0.42), C(0.12, 0.12, 0.14), C(0.5, 0.18, 0.35)];
  const rnd = mulberry32(55);
  for (let z = -80; z < 56; z += 13) {
    const tiers = 2 + Math.floor(rnd() * 4);
    for (let x = -14; x < 14; x += 2.5) {
      const h = 11 + 2.6 * (tiers - (rnd() < 0.25 ? 1 : 0));
      B.box(x, 11, z, x + 2.42, h, z + 12.2, cols[Math.floor(rnd() * cols.length)]);
    }
  }
  B.box(-15, 11, 60, 15, 26, 80, WHITE);
  B.box(-15.1, 21, 59.9, 15.1, 23, 80.1, GLASS, 1);
  B.box(-16, 26, 62, 16, 28, 70, WHITE);
  B.box(-2.5, 28, 72, 2.5, 36, 77, C(0.55, 0.1, 0.08));
  B.box(-0.2, 28, 64, 0.2, 40, 64.4, DARK);
  B.box(-0.15, 40, 64.05, 0.15, 40.3, 64.35, C(1, 1, 0.9), 3);
  return B.geometry();
}

/** ~11 m sloop: white hull, mast, mainsail + jib (both double-sided). */
export function sailboatGeometry(sailCol = C(0.86, 0.85, 0.8), sails = true) {
  const B = new GeoBuf();
  hull(B, {
    L: 11, beam: 3.6, freeboard: 1.0, draft: 0.5, side: WHITE, bottomCol: C(0.1, 0.12, 0.2), deckCol: C(0.6, 0.55, 0.45),
    stations: [[0, 0.05, 1.25], [0.12, 0.6, 1.1], [0.35, 1, 1], [0.75, 0.95, 1], [1, 0.75, 1]],
  });
  B.box(-1.0, 1.0, -0.8, 1.0, 1.5, 2.0, OFFW);          // coachroof
  B.box(-0.07, 1.0, -1.2, 0.07, 15, -1.0, C(0.7, 0.7, 0.72)); // mast
  B.box(-0.06, 1.9, -1.0, 0.06, 2.0, 3.6, C(0.6, 0.6, 0.6));  // boom
  if (sails) {
    B.tri2([0, 2.1, -1.0], [0, 14.6, -1.05], [0, 2.1, 3.5], sailCol);  // main
    B.tri2([0, 1.2, -5.2], [0, 13.5, -1.25], [0, 1.6, -1.6], sailCol); // jib
  } else B.box(-0.18, 1.95, -0.9, 0.18, 2.25, 3.5, C(0.1, 0.2, 0.4)); // furled main under its cover
  B.box(-0.05, 15, -1.15, 0.05, 15.15, -1.05, C(1, 1, 0.95), 3);    // masthead light
  return B.geometry();
}

/** Small fishing boat (moored): white hull, wheelhouse, gantry. */
export function fishingBoatGeometry() {
  const B = new GeoBuf();
  hull(B, {
    L: 14, beam: 4.6, freeboard: 1.6, draft: 1.0, side: C(0.75, 0.74, 0.7), bottomCol: C(0.45, 0.1, 0.08), deckCol: C(0.4, 0.38, 0.35),
    stripe: { h: 0.25, col: C(0.1, 0.25, 0.45) }, stripeY: 1.2,
    stations: [[0, 0.1, 1.3], [0.15, 0.7, 1.1], [0.4, 1, 1], [1, 0.92, 1]],
  });
  B.box(-1.6, 1.6, -3.0, 1.6, 3.8, 0.5, WHITE);
  B.box(-1.65, 2.8, -3.05, 1.65, 3.4, 0.55, GLASS, 1);
  B.box(-0.08, 3.8, -1.5, 0.08, 7.5, -1.3, DARK);
  B.box(-2.0, 1.6, 4.5, -1.85, 4.5, 4.7, DARK); B.box(1.85, 1.6, 4.5, 2.0, 4.5, 4.7, DARK); B.box(-2.0, 4.4, 4.5, 2.0, 4.6, 4.7, DARK);
  return B.geometry();
}

/** Motor launch / small cruiser (moored). */
export function launchGeometry() {
  const B = new GeoBuf();
  hull(B, {
    L: 9, beam: 3.0, freeboard: 1.1, draft: 0.5, side: C(0.8, 0.8, 0.78), bottomCol: C(0.12, 0.15, 0.22), deckCol: C(0.55, 0.5, 0.42),
    stations: [[0, 0.08, 1.2], [0.2, 0.75, 1.05], [0.45, 1, 1], [1, 0.95, 1]],
  });
  B.box(-1.1, 1.1, -1.5, 1.1, 2.2, 1.2, OFFW);
  B.box(-1.12, 1.6, -1.55, 1.12, 2.0, 1.25, GLASS, 1);
  return B.geometry();
}

// ====================================================================== material
/** Vertex-coloured PBR boat material; aEmit windows / nav lights glow at night (uNight). */
export function createBoatMaterial({ bob = false, wetLevel = null } = {}) {
  const m = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.55, metalness: 0.05 });
  const u = { uNight: { value: 0 }, uTime: { value: 0 }, uWet: { value: wetLevel ?? -1e6 } };
  m.userData.u = u;
  m.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, u);
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', `#include <common>\nattribute float aEmit;\nvarying float vEmit;\nvarying float vWetY;\nuniform float uTime;${bob ? '\nattribute float aBob;' : ''}`)
      .replace('#include <begin_vertex>', `#include <begin_vertex>\nvEmit = aEmit;\nvWetY = ${bob ? '(aBob > 0.0 ? 1e6 : (modelMatrix * vec4(transformed, 1.0)).y)' : '1e6'};${bob ? `
if ( aBob > 0.0 ) { // moored hulls ride the swell (phase per boat)
  transformed.y += sin( uTime * 1.1 + aBob * 6.2831 ) * 0.07 + sin( uTime * 0.53 + aBob * 17.0 ) * 0.04;
}` : ''}`);
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', '#include <common>\nvarying float vEmit;\nvarying float vWetY;\nuniform float uNight;\nuniform float uWet;')
      .replace('#include <color_fragment>', `#include <color_fragment>
{ // pilings / fenders: wet, weed-dark tide zone
  float yw = vWetY - uWet;
  diffuseColor.rgb *= 1.0 - 0.45 * (1.0 - smoothstep(0.1, 1.1, yw));
  diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.03, 0.045, 0.03), (1.0 - smoothstep(-0.2, 0.5, yw)) * 0.8);
}`)
      .replace('#include <emissivemap_fragment>', `#include <emissivemap_fragment>
if ( vEmit > 0.5 ) {
  vec3 ec = vEmit > 2.5 ? diffuseColor.rgb * 9.0 : vec3( 1.0, 0.72, 0.42 ) * 0.9;
  totalEmissiveRadiance += ec * uNight;
  diffuseColor.rgb *= 1.0 - 0.6 * uNight * step( vEmit, 2.5 );
}`);
  };
  m.customProgramCacheKey = () => (bob ? 'roadster-boat-bob-2' : 'roadster-boat-2');
  return m;
}

// ====================================================================== wakes
function createWakeMaterial(noiseMap) {
  const u = THREE.UniformsUtils.merge([THREE.UniformsLib.fog, {
    uTime: { value: 0 }, uLight: { value: 1 }, uNoise: { value: null },
  }]);
  Object.assign(u, waveUniforms());
  u.uNoise.value = noiseMap;
  const m = new THREE.ShaderMaterial({
    uniforms: u,
    vertexShader: /* glsl */`
#include <common>
#include <fog_pars_vertex>
uniform float uTime;
attribute vec3 aWake; // beam (m), length (m), strength
varying vec2 vL;      // metres across / behind the stern
varying vec3 vW;
varying vec3 vWPos;
${WAVE_GLSL}
void main() {
  vec4 wp = modelMatrix * instanceMatrix * vec4(position, 1.0);
  vec3 n;
  vec3 d = gerstner(wp.xz, length(wp.xz - cameraPosition.xz), uTime, n);
  wp.y += d.y + 0.07;
  vL = vec2(position.x * (aWake.x + 0.72 * aWake.y), position.z * aWake.y);
  vW = aWake;
  vWPos = wp.xyz;
  vec4 mvPosition = viewMatrix * wp;
  gl_Position = projectionMatrix * mvPosition;
  #include <fog_vertex>
}`,
    fragmentShader: /* glsl */`
#include <common>
#include <fog_pars_fragment>
uniform float uTime;
uniform float uLight;
uniform sampler2D uNoise;
varying vec2 vL;
varying vec3 vW;
varying vec3 vWPos;
void main() {
  float beam = vW.x, len = vW.y;
  float z = max(vL.y, 0.0), ax = abs(vL.x);
  float zf = z / len;
  float n = texture2D(uNoise, vWPos.xz / 6.0 + vec2(0.0, uTime * 0.05)).a;
  float n2 = texture2D(uNoise, vWPos.xz / 19.0).b;
  // Kelvin arms at ~19.5 deg from the stern corners
  float arm = beam * 0.5 + z * 0.354;
  float aw = 0.6 + z * 0.05;
  float arms = exp(-pow((ax - arm) / aw, 2.0)) * (0.5 + 0.7 * n);
  // turbulent wash behind the hull
  float wash = (1.0 - smoothstep(beam * 0.35, beam * 0.65 + z * 0.08, ax)) * (0.55 + 0.6 * n) * exp(-zf * 3.5);
  float bow = exp(-z * 0.25) * smoothstep(beam * 0.75, beam * 0.45, ax) * 0.4;
  float a = (arms * exp(-zf * 2.6) * 0.75 + wash + bow) * vW.z * smoothstep(0.0, 0.04, zf) * (0.75 + 0.5 * n2);
  a *= 1.0 - smoothstep(0.85, 1.0, zf);
  vec3 col = vec3(0.62, 0.66, 0.66) * uLight * a;
  #ifdef USE_FOG
    #ifdef FOG_EXP2
      float fogFactor = 1.0 - exp( - fogDensity * fogDensity * vFogDepth * vFogDepth );
    #else
      float fogFactor = smoothstep( fogNear, fogFar, vFogDepth );
    #endif
    col *= 1.0 - fogFactor;
  #endif
  gl_FragColor = vec4(col, 1.0);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}`,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, fog: true,
    polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -4,
  });
  return m;
}

// ====================================================================== routes
/** Closed Catmull-Rom path sampled by arc length. */
class Path {
  constructor(pts, closed = true, step = 4) {
    const cv = new THREE.CatmullRomCurve3(pts.map(([x, z]) => new THREE.Vector3(x, 0, z)), closed, 'centripetal');
    const L = cv.getLength(), n = Math.max(16, Math.ceil(L / step));
    this.p = cv.getSpacedPoints(n).map((v) => [v.x, v.z]);
    if (closed) this.p.pop();
    this.closed = closed;
    this.n = this.p.length;
    this.len = L;
    this.seg = L / (closed ? this.n : this.n - 1);
  }
  at(s, out) {
    const L = this.len;
    s = this.closed ? ((s % L) + L) % L : clamp(s, 0, L);
    const f = s / this.seg, i = Math.floor(f), t = f - i;
    const a = this.p[i % this.n], b = this.p[(i + 1) % this.n];
    out.x = a[0] + (b[0] - a[0]) * t; out.z = a[1] + (b[1] - a[1]) * t;
    const dx = b[0] - a[0], dz = b[1] - a[1], l = Math.hypot(dx, dz) || 1;
    out.dx = dx / l; out.dz = dz / l;
    return out;
  }
}

function segDist(x, z, s) {
  const dx = s[2] - s[0], dz = s[3] - s[1], L2 = dx * dx + dz * dz;
  const t = L2 > 0 ? clamp(((x - s[0]) * dx + (z - s[1]) * dz) / L2, 0, 1) : 0;
  return Math.hypot(s[0] + dx * t - x, s[1] + dz * t - z);
}
function segCross(a, b, c, d) {
  const o = (p, q, r) => (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]);
  return (o(a, b, c) > 0) !== (o(a, b, d) > 0) && (o(c, d, a) > 0) !== (o(c, d, b) > 0);
}

// ====================================================================== fleet
const WIND_TO = [0.93, -0.36]; // wind blows toward the ENE

export class Boats {
  /**
   * @param {object} o
   * @param {(x:number,z:number)=>boolean} o.isLand
   * @param {(x:number,z:number)=>number} o.shoreDist signed distance to shore (m)
   * @param {number} o.level water level
   * @param {Array<[number,number,number,number]>} o.bridge segments [x0,z0,x1,z1] to keep clear of (bridge lines)
   * @param {THREE.Texture} o.noiseMap
   * @param {{x:number,z:number}|null} o.ferryDock ferry terminal (water side of the terminal building); null = no ferries
   * @param {Array<[number,number]>} [o.ferryRoute] closed ferry loop as offsets from the dock (first point = dock)
   * @param {number} [o.ferries=2] ferries on the loop (evenly phased)
   * @param {Array<[number,number]>} [o.shipRoute] closed container-ship loop (world x, z); none by default
   * @param {Array<[number,number,number]>} [o.sailZones] sailboat zones [cx, cz, size] (world); none by default
   * @param {number} [o.sailCount=9]
   */
  constructor(o) {
    this.o = o;
    this.group = new THREE.Group();
    this.group.name = 'boats';
    this.list = [];
    this._v = { x: 0, z: 0, dx: 0, dz: 1 };
    this._q = new THREE.Quaternion(); this._e = new THREE.Euler(0, 0, 0, 'YXZ'); this._m = new THREE.Matrix4();
    this._s = new THREE.Vector3(1, 1, 1); this._p = new THREE.Vector3(); this._sl = [0, 0];
  }

  /** Route clear of land (margin m from the shore; less near the ferry terminal) and of the `bridge` lines? */
  _clear(path, margin) {
    const { isLand, shoreDist, bridge, ferryDock: D } = this.o;
    for (let i = 0; i < path.n; i += 2) {
      const [x, z] = path.p[i];
      const nearDock = D && Math.hypot(x - D.x, z - D.z) < 420;
      if (isLand(x, z) || shoreDist(x, z) < (nearDock ? Math.min(margin, 14) : margin)) return false;
      if (bridge) for (const s of bridge) if (segDist(x, z, s) < 70) return false;
    }
    if (bridge) for (let i = 0; i < path.n; i++) {
      const a = path.p[i], b = path.p[(i + 1) % path.n];
      for (const s of bridge) if (segCross(a, b, [s[0], s[1]], [s[2], s[3]])) return false;
    }
    return true;
  }

  build() {
    const o = this.o, rnd = mulberry32(31337);
    const mat = this.material = createBoatMaterial();
    const D = o.ferryDock;
    // ---- ferries: closed loop from the terminal (ferryRoute offsets from the dock)
    const ferryPts = D && Array.isArray(o.ferryRoute) && o.ferryRoute.length > 2 ? o.ferryRoute.map(([dx, dz]) => [D.x + dx, D.z + dz]) : null;
    const fpath = ferryPts ? this._fit(ferryPts, 70) : null;
    const routes = [];
    if (fpath) {
      const nf = o.ferries ?? 2;
      for (let k = 0; k < nf; k++) routes.push({ kind: 'ferry', path: fpath, speed: 11, phase: k ? fpath.len * (k / nf) : 0, dock: 0, beam: 11, len: 42, draft: 1.6, wake: 170 });
    }
    // ---- container ship: long loop out in the open water
    const sp = Array.isArray(o.shipRoute) && o.shipRoute.length > 2 ? o.shipRoute : null;
    const spath = sp ? this._fit(sp, 160, true) : null;
    if (spath) routes.push({ kind: 'ship', path: spath, speed: 6.5, phase: spath.len * 0.15, beam: 32, len: 210, draft: 9, wake: 520 });
    // ---- sailboats: small loops in open water
    let tries = 0, sails = 0;
    const zones = o.sailZones || [], sailCount = zones.length ? (o.sailCount ?? 9) : 0;
    while (sails < sailCount && tries < 400) {
      tries++;
      const z = zones[sails % zones.length];
      const cx = z[0] + (rnd() - 0.5) * z[2], cz = z[1] + (rnd() - 0.5) * z[2];
      const r1 = 120 + rnd() * 260, r2 = 60 + rnd() * 140, a0 = rnd() * Math.PI;
      const pts = [];
      for (let k = 0; k < 7; k++) {
        const a = (k / 7) * Math.PI * 2, rr = 1 + (rnd() - 0.5) * 0.3;
        const ex = Math.cos(a) * r1 * rr, ez = Math.sin(a) * r2 * rr;
        pts.push([cx + ex * Math.cos(a0) - ez * Math.sin(a0), cz + ex * Math.sin(a0) + ez * Math.cos(a0)]);
      }
      const p = new Path(rnd() < 0.5 ? pts : pts.reverse(), true, 3);
      if (!this._clear(p, 45)) continue;
      routes.push({ kind: 'sail', path: p, speed: 3 + rnd() * 2.5, phase: rnd() * p.len, beam: 3.6, len: 11, draft: 0.5, wake: 34, rock: rnd() * 10 });
      sails++;
    }
    // ---- instanced meshes
    const make = (geo, n, name) => {
      if (!n) return null;
      const im = new THREE.InstancedMesh(geo, mat, n);
      im.name = name; im.castShadow = false; im.receiveShadow = true; // no shadow pass: keeps the fleet at 4 draw calls
      im.frustumCulled = false; // instances span the whole bay
      im.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      this.group.add(im);
      return im;
    };
    const nF = routes.filter((r) => r.kind === 'ferry').length, nS = routes.filter((r) => r.kind === 'ship').length, nB = routes.filter((r) => r.kind === 'sail').length;
    this.meshes = { ferry: make(ferryGeometry(), nF, 'ferries'), ship: make(shipGeometry(), nS, 'container-ship'), sail: make(sailboatGeometry(), nB, 'sailboats') };
    const idx = { ferry: 0, ship: 0, sail: 0 };
    for (const r of routes) { r.mesh = this.meshes[r.kind]; r.i = idx[r.kind]++; r.s = r.phase; r.dwell = r.kind === 'ferry' && r.phase === 0 ? 20 : 0; r.v = r.speed; this.list.push(r); }
    // ---- wakes
    const wg = new THREE.PlaneGeometry(1, 1, 6, 16).rotateX(-Math.PI / 2).translate(0, 0, 0.5); // z: 0 (stern) .. 1
    const wm = this.wakeMat = createWakeMaterial(o.noiseMap);
    const wakes = new THREE.InstancedMesh(wg, wm, Math.max(1, this.list.length));
    wakes.name = 'wakes'; wakes.frustumCulled = false; wakes.renderOrder = 2;
    wakes.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    const aW = new THREE.InstancedBufferAttribute(new Float32Array(Math.max(1, this.list.length) * 3), 3);
    aW.setUsage(THREE.DynamicDrawUsage);
    wg.setAttribute('aWake', aW);
    this.wakes = wakes; this.aWake = aW;
    this.group.add(wakes);
    this.tris = Object.values(this.meshes).reduce((a, im) => a + (im ? (im.geometry.attributes.position.count / 3) * im.count : 0), 0) + this.list.length * 6 * 16 * 2;
    this.update(0, null, 0, 0);
    return this.group;
  }

  /** Fit a closed route; if it touches land, push the offending points away from the shore a few times. */
  _fit(pts, margin, open = false) {
    void open;
    const { shoreDist } = this.o;
    let P = pts.map((p) => p.slice());
    for (let it = 0; it < 8; it++) {
      const path = new Path(P, true, 6);
      if (this._clear(path, margin)) return new Path(P, true, 3);
      // move each control point down the distance gradient (away from land)
      P = P.map(([x, z], i) => {
        if (i === 0 && this.o.ferryDock && Math.hypot(x - this.o.ferryDock.x, z - this.o.ferryDock.z) < 1) return [x, z];
        const d = shoreDist(x, z);
        if (d > margin * 2.5) return [x, z];
        const e = 20, gx = shoreDist(x + e, z) - shoreDist(x - e, z), gz = shoreDist(x, z + e) - shoreDist(x, z - e), l = Math.hypot(gx, gz) || 1;
        const push = margin * 2.5 - d;
        return [x + (gx / l) * push, z + (gz / l) * push];
      });
    }
    console.warn('[Boats] route rejected (land / bridge)', pts[0]);
    return null;
  }

  /**
   * @param {number} dt
   * @param {THREE.Vector3|null} cam
   * @param {number} time
   * @param {number} night 0..1
   * @param {number} [light] daylight factor for the foam wakes
   */
  update(dt, cam, time, night, light = 1) {
    const o = this.o, v = this._v, e = this._e, m = this._m, q = this._q, s = this._s, p = this._p, sl = this._sl;
    const cx = cam ? cam.x : 0, cz = cam ? cam.z : 0;
    let wi = 0;
    for (const r of this.list) {
      // speed: ferries slow into the terminal and dwell there
      let spd = r.speed;
      if (r.kind === 'ferry') {
        const L = r.path.len;
        const s0 = ((r.s % L) + L) % L;
        const toDock = L - s0, fromDock = s0;
        if (r.dwell > 0) { r.dwell -= dt; spd = 0; }
        else {
          spd = r.speed * clamp(Math.min(toDock, fromDock * 1.4) / 320, 0.08, 1);
          if (toDock < 3 && toDock >= 0) { r.dwell = 45; r.s = Math.ceil(r.s / L) * L; }
        }
      }
      r.v += (spd - r.v) * Math.min(1, dt * 0.5);
      r.s += r.v * dt;
      r.path.at(r.s, v);
      const yaw = Math.atan2(-v.dx, -v.dz); // local -z (bow) along the path
      const hs = waveHeight(v.x, v.z, time, cx, cz);
      waveSlope(v.x, v.z, time, cx, cz, sl);
      const big = r.kind === 'ship' ? 0.05 : r.kind === 'ferry' ? 0.35 : 1;
      // pitch / roll from the swell slope in the boat frame
      const fwdSlope = sl[0] * v.dx + sl[1] * v.dz, sideSlope = sl[0] * -v.dz + sl[1] * v.dx;
      let pitch = Math.atan(fwdSlope) * big, roll = Math.atan(sideSlope) * big;
      if (r.kind === 'sail') {
        const lat = v.dx * WIND_TO[1] - v.dz * WIND_TO[0]; // wind across the boat
        roll -= (0.2 + 0.08 * Math.sin(time * 0.7 + r.rock)) * lat - Math.sin(time * 1.3 + r.rock) * 0.03; // heel to leeward
        pitch += Math.sin(time * 1.1 + r.rock * 2) * 0.02;
      } else if (r.kind === 'ferry') roll += Math.sin(time * 0.6 + r.i) * 0.006;
      e.set(pitch, yaw, roll, 'YXZ');
      q.setFromEuler(e);
      p.set(v.x, o.level + hs * (r.kind === 'ship' ? 0.2 : 0.85), v.z);
      m.compose(p, q, s);
      r.mesh.setMatrixAt(r.i, m);
      // wake behind the stern
      const sternX = v.x + v.dx * -r.len * 0.5, sternZ = v.z + v.dz * -r.len * 0.5;
      e.set(0, yaw, 0, 'YXZ'); q.setFromEuler(e);
      const strength = clamp(r.v / r.speed, 0, 1) * (r.kind === 'sail' ? 0.55 : 1);
      const wl = r.wake * (0.4 + 0.6 * strength);
      p.set(sternX, o.level, sternZ);
      this._s2 = this._s2 || new THREE.Vector3();
      this._s2.set(r.beam + 0.72 * wl, 1, wl);
      m.compose(p, q, this._s2);
      this.wakes.setMatrixAt(wi, m);
      this.aWake.setXYZ(wi, r.beam, wl, strength);
      wi++;
    }
    for (const k in this.meshes) if (this.meshes[k]) this.meshes[k].instanceMatrix.needsUpdate = true;
    this.wakes.instanceMatrix.needsUpdate = true;
    this.aWake.needsUpdate = true;
    this.material.userData.u.uNight.value = night;
    const wu = this.wakeMat.uniforms;
    wu.uTime.value = time;
    wu.uLight.value = light;
  }
}
