// Headless checks for the volumetric sun light pass (no browser, no WebGL): options + validation, output sizing,
// the physics the shader implements (JS ports of its Henyey-Greenstein phase, its per-sample optical depth and its
// march, against fine numeric integration and against the final pass's analytic fog `layer()`, ported verbatim from
// PostFX.js), the shader text (both shadow paths, every cascade count, the documented uniforms), the cascade walk
// (a port of the shader's sunVisibility averaged over xi = a port of shadowCascade.js's blend, 1-4 cascades), the
// depth-aware history tap, define switching only on change, and render() on a stub renderer (zero output, the sun /
// moon gate, uniforms, temporal ping-pong / freshness with and without a CameraHistory, no allocation per frame),
// with a fake and a real (headless) Atmosphere. Pixels and timings are measured in the browser (ENGINE.md, measurement protocol).
// Run: node test/volumetric.mjs
import { bytesPerCall, gate } from './helpers/alloc.mjs';
import { readFileSync } from 'node:fs';
import * as THREE from 'three';
import { VolumetricPass, VOLUMETRIC_DEFAULTS } from '../src/render/passes/VolumetricPass.js';
import { CameraHistory } from '../src/render/CameraHistory.js';
import { Atmosphere } from '../src/render/Atmosphere.js';
import { cascadeShadowChunkText } from '../src/render/shadowCascade.js';

let failures = 0, passed = 0;
const results = [];
function check(name, ok, detail = '') {
  if (ok) passed++; else failures++;
  results.push(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail !== '' ? `   (${detail})` : ''}`);
  return ok;
}
const throws = (fn, re) => { try { fn(); return false; } catch (e) { return re ? re.test(e.message) : true; } };
const rel = (a, b) => Math.abs(a - b) / Math.max(Math.abs(b), 1e-30);
const fmt = (x) => (Math.abs(x) < 1e-3 && x !== 0 ? x.toExponential(2) : x.toFixed(5));
const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const DEG = Math.PI / 180;

// allocation gates (test/helpers/alloc.mjs: exact bytes per call of TurboFan code, with a control) report through these
const note = (s) => results.push(`NOTE  ${s}`);
const inRange = (name, v, lo, hi, unit) => check(name, Number.isFinite(v) && v >= lo && v <= hi, `${Number.isFinite(v) ? +v.toFixed(3) : v} ${unit} in [${lo}, ${hi}]`);

// composite Simpson over [a, b] with n (even) intervals
function simpson(f, a, b, n) {
  const h = (b - a) / n;
  let s = f(a) + f(b);
  for (let i = 1; i < n; i++) s += f(a + i * h) * (i & 1 ? 4 : 2);
  return s * h / 3;
}

// enough of WebGLRenderer for render(): draws and clears are counted no-ops (as test/postfx-options.mjs)
function stubRenderer() {
  let bound = null;
  const stats = { draws: 0, clears: 0, lastTarget: null };
  return {
    stats,
    getClearColor: (c) => c.setRGB(0, 0, 0), getClearAlpha: () => 1, setClearColor() {},
    clear() { stats.clears++; }, render() { stats.draws++; stats.lastTarget = bound; },
    getRenderTarget: () => bound, setRenderTarget(t) { bound = t; },
  };
}

// a fake Atmosphere with controllable cascades (count, packed, maps rendered yet), shadowCascadeInfo as Atmosphere's
function fakeAtmosphere() {
  const lights = [0, 1, 2, 3].map((i) => {
    const l = new THREE.DirectionalLight();
    l.shadow.bias = -0.00003 * (i + 1);
    // an orthographic shadow matrix: bias * ortho(half 40 * 2^i, depth 1..2500) * a rotated view
    const half = 40 * 2 ** i, view = new THREE.Matrix4().makeRotationX(-1.1 + 0.1 * i).setPosition(5 * i, -300, 20);
    const ortho = new THREE.Matrix4().makeOrthographic(-half, half, half, -half, 1, 2500);
    l.shadow.matrix.set(0.5, 0, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0, 1).multiply(ortho).multiply(view);
    return l;
  });
  const maps = lights.map((l, i) => Object.assign(new THREE.Texture(), { name: `packed${i}` }));
  const depths = lights.map((l, i) => Object.assign(new THREE.DepthTexture(4, 4), { name: `depth${i}` }));
  const st = { count: 2, packed: true, ready: true };
  return {
    st, maps, depths, lights,
    sunDir: new THREE.Vector3(0.3, 0.45, -0.8).normalize(),
    sunColor: new THREE.Color(1, 0.9, 0.8),
    fogParams: { haze: 0.0003, hazeK: 1 / 700, hazeBase: 10, mist: 0.0004, mistK: 1 / 40, sunScatter: 0 },
    shadowCascadeInfo(out) {
      const n = st.count;
      for (let i = 0; i < n; i++) {
        const k = i % 4;
        out.lights[i] = lights[k]; out.cameras[i] = lights[k].shadow.camera; out.matrices[i] = lights[k].shadow.matrix;
        out.halfSizes[i] = 40 * 2 ** k;
        out.maps[i] = st.ready ? maps[k] : null;
        out.depthTextures[i] = st.ready ? (st.packed ? maps[k] : depths[k]) : null;
      }
      out.lights.length = out.cameras.length = out.matrices.length = out.halfSizes.length = out.maps.length = out.depthTextures.length = n;
      out.count = n;
      out.packed = st.packed;
      return out;
    },
  };
}

const W = 1600, H = 900;
const makeCamera = (y = 2) => {
  const cam = new THREE.PerspectiveCamera(60, W / H, 0.1, 6000);
  cam.position.set(3, y, 10); cam.lookAt(40, y + 4, -60); cam.updateMatrixWorld();
  return cam;
};
const readBuffer = { depthTexture: new THREE.DepthTexture(W, H) };

// ================================================================================================= options
{
  const keys = 'scale,maxPixels,steps,maxDistance,g,density,temporal,blend,jitter';
  check('VOLUMETRIC_DEFAULTS: the nine options, contract values', Object.keys(VOLUMETRIC_DEFAULTS).join() === keys
    && VOLUMETRIC_DEFAULTS.scale === 0.25 && VOLUMETRIC_DEFAULTS.maxPixels === Infinity && VOLUMETRIC_DEFAULTS.steps === 16
    && VOLUMETRIC_DEFAULTS.maxDistance === 400 && VOLUMETRIC_DEFAULTS.g === 0.55 && VOLUMETRIC_DEFAULTS.density === 1
    && VOLUMETRIC_DEFAULTS.temporal === true && VOLUMETRIC_DEFAULTS.blend === 0.9 && VOLUMETRIC_DEFAULTS.jitter === true);
  check('VOLUMETRIC_DEFAULTS frozen', Object.isFrozen(VOLUMETRIC_DEFAULTS));
  const cam = makeCamera(), atmo = fakeAtmosphere();
  const vol = new VolumetricPass(cam, null, atmo);
  check('constructs without a renderer; properties = defaults', Object.keys(VOLUMETRIC_DEFAULTS).every((k) => vol[k] === VOLUMETRIC_DEFAULTS[k]));
  check('Pass interface: needsSwap false, enabled, texture', vol.needsSwap === false && vol.enabled === true && vol.texture.isTexture && vol.history === null);
  const en = Object.getOwnPropertyDescriptor(vol, 'enabled');
  check("enabled is Pass's plain data property (no accessor: kept in the .d.ts)", !!en && en.value === true && en.writable && !('enabled' in Object.getOwnPropertyDescriptors(VolumetricPass.prototype)));
  const v2 = new VolumetricPass(cam, null, atmo, { steps: 24, g: -0.3, temporal: false, maxPixels: null });
  check('options applied (steps, g, temporal; maxPixels null = Infinity)', v2.steps === 24 && v2.g === -0.3 && v2.temporal === false && v2.maxPixels === Infinity);
  const bad = {
    scale: [0, 1.5, NaN, '0.5'], maxPixels: [0, -5, NaN], steps: [0, 2.5, 129, NaN], maxDistance: [0, -1, Infinity, NaN],
    g: [1, -1, NaN], density: [-0.1, Infinity, NaN], temporal: [1, 'false', null], blend: [1, -0.1, NaN], jitter: [0, 'yes'],
  };
  for (const k in bad) {
    const ctor = bad[k].every((v) => throws(() => new VolumetricPass(cam, null, atmo, { [k]: v }), new RegExp(`\\b${k} must be`)));
    const set = bad[k].every((v) => throws(() => { vol[k] = v; }, new RegExp(`\\b${k} must be`))) && vol[k] === VOLUMETRIC_DEFAULTS[k];
    check(`${k}: bad values throw a clear Error (constructor and setter), value kept`, ctor && set);
  }
  check('unknown option throws, naming it', throws(() => new VolumetricPass(cam, null, atmo, { stepz: 8 }), /unknown option 'stepz'/));
  check('no atmosphere / no camera throws', throws(() => new VolumetricPass(cam, null, null), /atmosphere/) && throws(() => new VolumetricPass(null, null, atmo), /camera/));
  check('setters accept valid values', (() => { const v = new VolumetricPass(cam, null, atmo); v.steps = 32; v.g = 0; v.density = 0; v.blend = 0; v.jitter = false; v.maxDistance = 1000; return v.steps === 32 && v.g === 0 && v.density === 0 && v.blend === 0 && !v.jitter && v.maxDistance === 1000; })());
  vol.dispose(); v2.dispose();
}

// ================================================================================================= sizing
{
  const vol = new VolumetricPass(makeCamera(), null, fakeAtmosphere());
  vol.setSize(W, H);
  check('setSize 1600x900 at scale 0.25: 400x225', vol.width === 400 && vol.height === 225 && vol.texture.image.width === 400 && vol.texture.image.height === 225);
  const t = vol.texture, other = vol._rt[0].texture === t ? vol._rt[1] : vol._rt[0];
  check('both targets sized alike', other.width === 400 && other.height === 225);
  check('output RG16F, linear, no mipmaps', [vol._rt[0], vol._rt[1]].every((rt) => rt.texture.type === THREE.HalfFloatType && rt.texture.format === THREE.RGFormat
    && rt.texture.minFilter === THREE.LinearFilter && rt.texture.magFilter === THREE.LinearFilter && rt.texture.generateMipmaps === false && !rt.depthBuffer));
  vol.maxPixels = 50000; // setter resizes at once (AOPass budget rule: the largest uniform scale that fits, floored)
  const s = Math.sqrt(50000 / (W * H));
  check('maxPixels 50000: floored uniform scale, under budget', vol.width === Math.floor(W * s) && vol.height === Math.floor(H * s) && vol.width * vol.height <= 50000, `${vol.width}x${vol.height}`);
  vol.maxPixels = Infinity; vol.scale = 0.5;
  check('scale setter resizes at once: 800x450', vol.width === 800 && vol.height === 450);
  vol.scale = 0.25;
  vol.setSize(2, 2);
  const tiny = vol.width === 1 && vol.height === 1;
  vol.setSize(0, 0);
  check('min 1x1 (2x2 and 0x0 buffers)', tiny && vol.width === 1 && vol.height === 1);
  vol.setSize(3024, 1964);
  check('3024x1964 at 0.25: 756x491', vol.width === 756 && vol.height === 491);
  vol.dispose();
}

// ================================================================================================= phase function
{
  const vol = new VolumetricPass(makeCamera(), null, fakeAtmosphere());
  vol.setSize(W, H);
  const r = stubRenderer();
  const P = vol.material.uniforms.uPhase.value;
  // JS port of the shader: phase = uPhase.x / pow( uPhase.y - uPhase.z * mu, 1.5 ), with the uniforms the pass sets
  const phase = (mu) => P.x / Math.pow(P.y - P.z * mu, 1.5);
  for (const g of [-0.9, -0.5, 0, 0.55, 0.8, 0.95]) {
    vol.g = g; vol.render(r, null, readBuffer);
    const integral = 2 * Math.PI * simpson(phase, -1, 1, 400000);
    check(`HG g=${g}: integrates to 1 over the sphere (1e-3)`, Math.abs(integral - 1) < 1e-3, fmt(integral));
  }
  vol.g = 0.55; vol.render(r, null, readBuffer);
  check('HG g=0.55: toward the sun (mu 1) brighter than away, = (1-g^2)/(4pi(1-g)^3)', phase(1) > 10 * phase(-1) && rel(phase(1), (1 - 0.55 ** 2) / (4 * Math.PI * (1 - 0.55) ** 3)) < 1e-12);
  check('shader: phase from uPhase and dot(dir, uSunDir)', vol.material.fragmentShader.includes('uPhase.x / pow( uPhase.y - uPhase.z * dot( dir, uSunDir ), 1.5 )'));
  vol.dispose();
}

// ================================================================================================= one medium
// The final pass's analytic optical depth, PostFX.js FinalShader, ported verbatim:
//   float layer( float y0, float y1, float dist, float d, float k ) {
//     float x = k * ( y1 - y0 );
//     float f = abs( x ) > 1e-3 ? ( 1.0 - exp( -x ) ) / x : 1.0 - 0.5 * x;
//     return d * exp( -k * clamp( y0, -100.0, 4000.0 ) ) * dist * f;
//   }
// called as layer( uCamPos.y - uFogA.z, wp.y - uFogA.z, dist, uFogA.x, uFogA.y ) + layer( uCamPos.y - uFogB.z, ... ),
// uFogA = (haze, hazeK, hazeBase), uFogB = (mist, mistK, 0)
function layer(y0, y1, dist, d, k) {
  const x = k * (y1 - y0);
  const f = Math.abs(x) > 1e-3 ? (1.0 - Math.exp(-x)) / x : 1.0 - 0.5 * x;
  return d * Math.exp(-k * clamp(y0, -100.0, 4000.0)) * dist * f;
}
const finalTau = (fp, cy, wy, dist) => layer(cy - fp.hazeBase, wy - fp.hazeBase, dist, fp.haze, fp.hazeK) + layer(cy - 0, wy - 0, dist, fp.mist, fp.mistK);
// JS port of the shader's per-sample medium: uMedium = (haze sigma at the camera, haze k, mist sigma, mist k)
function sampleMedium(M, dirY, t) {
  const xh = clamp(M.y * dirY * t, -60, 60), xm = clamp(M.w * dirY * t, -60, 60);
  const eh = Math.exp(-xh), em = Math.exp(-xm);
  const fh = Math.abs(xh) > 1e-3 ? (1 - eh) / xh : 1 - 0.5 * xh, fm = Math.abs(xm) > 1e-3 ? (1 - em) / xm : 1 - 0.5 * xm;
  return { sigma: M.x * eh + M.z * em, tau: t * (M.x * fh + M.z * fm) };
}
// JS port of the shader's march (vis = 1, no phase): sum sigma * exp(-tau) * dt at t = (i + o) * dt
function march(M, dirY, L, steps, o) {
  const dt = L / steps;
  let sum = 0;
  for (let i = 0; i < steps; i++) { const s = sampleMedium(M, dirY, (i + o) * dt); sum += s.sigma * Math.exp(-s.tau); }
  return sum * dt;
}
{
  const src = readFileSync(new URL('../src/render/PostFX.js', import.meta.url), 'utf8').replace(/\s+/g, ' ');
  check('layer() port matches PostFX.js GLSL text', src.includes('float layer( float y0, float y1, float dist, float d, float k ) { float x = k * ( y1 - y0 ); float f = abs( x ) > 1e-3 ? ( 1.0 - exp( -x ) ) / x : 1.0 - 0.5 * x; return d * exp( -k * clamp( y0, -100.0, 4000.0 ) ) * dist * f; }')
    && src.includes('layer( uCamPos.y - uFogA.z, wp.y - uFogA.z, dist, uFogA.x, uFogA.y ) + layer( uCamPos.y - uFogB.z, wp.y - uFogB.z, dist, uFogB.x, uFogB.y )'));
  const frag = VolumetricPassFrag();
  check('shader medium = the ported per-sample code', frag.includes('vec2 x = clamp( kdy * t, - 60.0, 60.0 );') && frag.includes('vec2 kdy = uMedium.yw * dir.y;')
    && frag.includes('float sigma = dot( uMedium.xz, e );') && frag.includes('float tau = t * dot( uMedium.xz, f );') && frag.includes('sum += sigma * exp( - tau ) * sunVisibility( t, xi );'));

  const atmo = fakeAtmosphere(), fp = atmo.fogParams;
  const sigmaAt = (h) => fp.haze * Math.exp(-fp.hazeK * (h - fp.hazeBase)) + fp.mist * Math.exp(-fp.mistK * h);
  const rays = [[2, 0, 'level'], [2, 0.3, 'slant up'], [120, 0, 'level'], [120, -0.25, 'slant down'], [120, 0.5, 'slant up']];
  const L = 400;
  for (const cy of [2, 120]) {
    const cam = makeCamera(cy), vol = new VolumetricPass(cam, null, atmo);
    vol.setSize(W, H); vol.render(stubRenderer(), null, readBuffer);
    const M = vol.material.uniforms.uMedium.value;
    check(`camera at ${cy} m: uMedium = density x sigma at the camera, per layer`, rel(M.x, fp.haze * Math.exp(-fp.hazeK * (cy - fp.hazeBase))) < 1e-12
      && rel(M.z, fp.mist * Math.exp(-fp.mistK * cy)) < 1e-12 && M.y === fp.hazeK && M.w === fp.mistK);
    vol.density = 2.5; vol.render(stubRenderer(), null, readBuffer);
    check(`camera at ${cy} m: density scales both layers`, rel(M.x, 2.5 * fp.haze * Math.exp(-fp.hazeK * (cy - fp.hazeBase))) < 1e-12 && rel(M.z, 2.5 * fp.mist * Math.exp(-fp.mistK * cy)) < 1e-12);
    vol.density = 1; vol.render(stubRenderer(), null, readBuffer);
    for (const [ry, dirY, kind] of rays) {
      if (ry !== cy) continue;
      const fine = simpson((s) => sigmaAt(cy + dirY * s), 0, L, 20000);
      const port = sampleMedium(M, dirY, L).tau, ana = finalTau(fp, cy, cy + dirY * L, L);
      check(`${cy} m ${kind}: per-sample optical depth = fine integration (1 %)`, rel(port, fine) < 0.01, `${fmt(port)} vs ${fmt(fine)}`);
      check(`${cy} m ${kind}: = the final pass's layer() (2 %)`, rel(port, ana) < 0.02 && rel(Math.exp(-port), Math.exp(-ana)) < 0.02, `${fmt(port)} vs ${fmt(ana)}`);
      // the in-scatter march: unshadowed and without phase, its integral is the fog opacity 1 - T(L) of the final pass
      const opacity = 1 - Math.exp(-ana);
      let jittered = 0;
      for (let k = 0; k < 256; k++) jittered += march(M, dirY, L, 16, (k + 0.5) / 256) / 256; // offsets the frames integrate
      check(`${cy} m ${kind}: jittered march (16 steps, offsets averaged) = 1 - T of the final pass (0.5 %)`, rel(jittered, opacity) < 0.005, `${fmt(jittered)} vs ${fmt(opacity)}`);
      const mid = march(M, dirY, L, 16, 0.5);
      check(`${cy} m ${kind}: midpoint march (jitter off) within 1 %`, rel(mid, opacity) < 0.01, `${fmt(mid)}`);
    }
    vol.dispose();
  }
  // per-sample optical depth at every sample point too (not only the end): T exact whatever the step count
  const vol = new VolumetricPass(makeCamera(2), null, atmo);
  vol.render(stubRenderer(), null, readBuffer);
  const M = vol.material.uniforms.uMedium.value;
  let worst = 0;
  for (const t of [3, 25, 77, 150, 260, 399]) worst = Math.max(worst, rel(sampleMedium(M, 0.3, t).tau, simpson((s) => sigmaAt(2 + 0.3 * s), 0, t, 4000)));
  check('per-sample optical depth exact at every t (2 m slant up)', worst < 1e-6, fmt(worst));
  vol.dispose();
}
function VolumetricPassFrag() { const v = new VolumetricPass(makeCamera(), null, fakeAtmosphere()); const s = v.material.fragmentShader; v.dispose(); return s; }

// ================================================================================================= shader text
// C preprocessor conditionals (+ object-like #define recorded for later #if); other lines kept
function evalIf(expr, defs) {
  const e = expr.replace(/\/\/.*$/, '')
    .replace(/defined\s*\(\s*(\w+)\s*\)/g, (_, n) => (n in defs ? '1' : '0'))
    .replace(/\b[A-Za-z_]\w*\b/g, (n) => (n in defs ? String(defs[n] === '' ? 1 : defs[n]) : '0'));
  return !!Function(`return ( ${e} );`)();
}
function preprocess(src, defs0) {
  const defs = { ...defs0 }, out = [], stack = [];
  let active = true;
  for (const line of src.split('\n')) {
    const m = /^\s*#\s*(if|ifdef|ifndef|elif|else|endif|define)\b\s*(.*)$/.exec(line);
    if (!m) { if (active) out.push(line); continue; }
    const [, d, rest] = m;
    if (d === 'define') { if (active) { const [n, ...v] = rest.split(/\s+/); defs[n] = v.join(' '); out.push(line); } continue; }
    if (d === 'if' || d === 'ifdef' || d === 'ifndef') {
      const c = d === 'if' ? evalIf(rest, defs) : d === 'ifdef' ? rest.trim() in defs : !(rest.trim() in defs);
      stack.push({ parent: active, taken: c }); active = active && c;
    } else if (d === 'elif') {
      const top = stack[stack.length - 1], c = !top.taken && evalIf(rest, defs);
      top.taken ||= c; active = top.parent && c;
    } else if (d === 'else') {
      const top = stack[stack.length - 1]; active = top.parent && !top.taken; top.taken = true;
    } else { active = stack.pop().parent; }
  }
  return { text: out.join('\n'), defs, balanced: stack.length === 0 };
}
const balanced = (s, open, close) => { let d = 0; for (const ch of s) { if (ch === open) d++; else if (ch === close && --d < 0) return false; } return d === 0; };
{
  const vol = new VolumetricPass(makeCamera(), null, fakeAtmosphere());
  const fs = vol.material.fragmentShader, u = vol.material.uniforms;
  check('shader: both shadow paths behind VOL_SHADOW_PACKED', fs.includes('#ifdef VOL_SHADOW_PACKED') && fs.includes('uniform sampler2D tShadow0;') && fs.includes('uniform sampler2DShadow tShadow0;'));
  check("packed path = three's texture2DCompare (unpackRGBAToDepth, reversed-depth aware)", fs.includes('#include <packing>') && fs.includes('float d = unpackRGBAToDepth( texture2D( map, c.xy ) );')
    && fs.includes('return step( c.z + bias, d );') && fs.includes('#ifdef USE_REVERSED_DEPTH_BUFFER'));
  check("compare path = the 'fast' kernel's tap: texture( sampler2DShadow, vec3( uv, z ) )", fs.includes('float volTap( sampler2DShadow map, vec3 c, float bias ) { return texture( map, vec3( c.xy, c.z + bias ) ); }'));
  const chunk = cascadeShadowChunkText(4);
  check('cascade walk text: the patch bands (0.39-0.47, far fade 0.43-0.495), z > 1 goes on, tap when w <= xi else xi /= w',
    ['smoothstep( 0.39, 0.47, max(', 'smoothstep( 0.43, 0.495, max('].every((s) => fs.includes(s) && chunk.includes(s)) && chunk.includes('( rc0.z > 1.0 ) ? 1.0')
    && fs.includes('if ( c.z > 1.0 ) return false;') && fs.includes('float w = smoothstep( 0.39, 0.47, max( e.x, e.y ) );') && fs.includes('if ( w <= xi ) return true;')
    && fs.includes('xi /= w;') && fs.includes('c.z > 1.0 ) return 1.0;') && fs.includes('s = mix( s, 1.0, smoothstep( 0.43, 0.495, max( e.x, e.y ) ) );'));
  const documented = ['tDepth', 'tHistory', 'tShadow0', 'tShadow1', 'tShadow2', 'tShadow3', 'uShadowMatrix', 'uShadowOrigin', 'uShadowBias', 'uProjInv', 'uCamWorld',
    'uReproj', 'uInvVPw', 'uJitter', 'uPrevJitter', 'uSunDir', 'uPhase', 'uMedium', 'uSteps', 'uMarch', 'uClip', 'uTemporal'];
  const declared = (n) => new RegExp(`uniform [^;]*\\b${n}\\b`).test(fs);
  check('every documented uniform declared in the shader and present on the material', documented.every((n) => declared(n) && u[n] && 'value' in u[n]), documented.filter((n) => !declared(n) || !u[n]).join());
  check('material uniforms: none undocumented', Object.keys(u).every((n) => documented.includes(n)));
  check('initial defines: VOL_CASCADES 1, packed', vol.material.defines.VOL_CASCADES === 1 && 'VOL_SHADOW_PACKED' in vol.material.defines);
  let combos = true, why = '';
  for (let n = 1; n <= 4; n++) for (const packed of [true, false]) {
    const p = preprocess(fs, packed ? { VOL_CASCADES: n, VOL_SHADOW_PACKED: 1 } : { VOL_CASCADES: n });
    const t = p.text, want = packed ? 'sampler2D' : 'sampler2DShadow', other = packed ? 'sampler2DShadow' : 'sampler2D';
    const maps = [0, 1, 2, 3].filter((k) => t.includes(`uniform ${want} tShadow${k};`));
    const ok = p.balanced && balanced(t, '{', '}') && balanced(t, '(', ')')
      && maps.join() === [...Array(n).keys()].join() && !/uniform sampler2D(Shadow)? tShadow/.test(t.replace(new RegExp(`uniform ${want} tShadow\\d;`, 'g'), ''))
      && (t.match(/float volTap\(/g) || []).length === 1 && t.includes(`float volTap( ${want} map`) && !t.includes(`float volTap( ${other} map`)
      && p.defs.VOL_LAST === String(n - 1) && p.defs.VOL_LAST_MAP === `tShadow${n - 1}`
      && [0, 1, 2].every((k) => t.includes(`if ( volPick( c, xi ) ) return volTap( tShadow${k}, c, uShadowBias[ ${k} ] );`) === (k < n - 1))
      && [0, 1, 2, 3].every((k) => t.includes(`volDC[ ${k} ] = mat3( uShadowMatrix[ ${k} ] ) * dir;`) === (k < n))
      && t.includes('s = mix( s, 1.0, smoothstep( 0.43, 0.495') === (n > 1);
    if (!ok) { combos = false; why += `${n}/${packed ? 'packed' : 'compare'} `; }
  }
  check('preprocessed for 1-4 cascades x both paths: one map per cascade of the right type, inner walk, far fade with > 1 map', combos, why.trim());
  vol.dispose();
}

// ================================================================================================= cascade walk vs the patch
// One shadow tap per sample with xi, averaged over xi uniform in [0, 1), must equal shadowCascade.js's blended
// lookup. Shadow coordinates per cascade cs[k] = { x, y, z }; tap(k, c) = a plain compare (bias 0) for both ports.
const sstep = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
const mixf = (a, b, t) => a * (1 - t) + b * t; // GLSL mix
const edgeOf = (c) => Math.max(Math.abs(c.x - 0.5), Math.abs(c.y - 0.5));
// port of the shader's volPick + sunVisibility
function volVisibility(cs, n, tap, xi) {
  for (let k = 0; k < n - 1; k++) {
    const c = cs[k];
    if (c.z > 1) continue;
    const w = sstep(0.39, 0.47, edgeOf(c));
    if (w <= xi) return tap(k, c);
    xi /= w;
  }
  const c = cs[n - 1];
  if (c.x < 0 || c.x > 1 || c.y < 0 || c.y > 1 || c.z > 1) return 1;
  let s = tap(n - 1, c);
  if (n > 1) s = mixf(s, 1, sstep(0.43, 0.495, edgeOf(c)));
  return s;
}
// its exact average over xi: piecewise constant, breaks at w0, w0 w1, w0 w1 w2 (each interval evaluated at its middle)
function volAverage(cs, n, tap) {
  const br = [0, 1];
  let p = 1;
  for (let k = 0; k < n - 1; k++) { if (cs[k].z > 1) continue; p *= sstep(0.39, 0.47, edgeOf(cs[k])); br.push(p); }
  br.sort((a, b) => a - b);
  let s = 0;
  for (let i = 0; i + 1 < br.length; i++) if (br[i + 1] > br[i]) s += (br[i + 1] - br[i]) * volVisibility(cs, n, tap, (br[i] + br[i + 1]) / 2);
  return s;
}
// three's getShadow around the filter: frustum test (uv in 0..1, z <= 1) else lit
const getShadow = (tap, k, c) => (c.x >= 0 && c.x <= 1 && c.y >= 0 && c.y <= 1 && c.z <= 1 ? tap(k, c) : 1);
// port of shadowCascade.js twoCascadeText (receiveShadow)
function patch2(cs, tap) {
  const N = cs[0], F = cs[1];
  const W = N.z > 1 ? 1 : sstep(0.39, 0.47, edgeOf(N));
  let SN = 1, SF = 1;
  if (W < 1) SN = getShadow(tap, 0, N);
  if (W > 0) { SF = getShadow(tap, 1, F); SF = mixf(SF, 1, sstep(0.43, 0.495, edgeOf(F))); }
  return mixf(SN, SF, W);
}
// port of shadowCascade.js generalText / cascadeLevel (2-4 cascades)
function patchN(cs, n, tap) {
  const level = (k) => {
    const c = cs[k], far = k === n - 1;
    const W = far ? 0 : c.z > 1 ? 1 : sstep(0.39, 0.47, edgeOf(c));
    let S = 1, R = 1;
    if (W < 1) S = getShadow(tap, k, c);
    if (far) S = mixf(S, 1, sstep(0.43, 0.495, edgeOf(c)));
    else if (W > 0) R = level(k + 1);
    return mixf(S, R, W);
  };
  return level(0);
}
{
  let seed = 12345;
  const rnd = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; };
  const depthAt = (k, c) => 0.5 + 0.45 * Math.sin(12.9898 * c.x + 78.233 * c.y + 3.1 * k);
  const tap = (k, c) => (c.z <= depthAt(k, c) ? 1 : 0);
  // edge distance 0..0.65 on one axis (inside, the 0.39-0.47 band, the 0.43-0.495 fade, outside), z > 1 one in five
  const coord = () => {
    const e = rnd() * 0.65, onX = rnd() < 0.5, o = (rnd() * 2 - 1) * e, sg = rnd() < 0.5 ? -1 : 1;
    return { x: 0.5 + (onX ? sg * e : o), y: 0.5 + (onX ? o : sg * e), z: rnd() < 0.2 ? 1 + rnd() * 0.2 : rnd() };
  };
  const cover = { zOver: 0, band: 0, fade: 0, outside: 0 };
  let worst = 0, worst2 = 0, mixed = 0, cases = 0;
  for (const n of [1, 2, 3, 4]) {
    for (let i = 0; i < 5000; i++) {
      const cs = [coord(), coord(), coord(), coord()];
      const want = n === 1 ? getShadow(tap, 0, cs[0]) : n === 2 ? patch2(cs, tap) : patchN(cs, n, tap);
      const got = volAverage(cs, n, tap);
      worst = Math.max(worst, Math.abs(got - want));
      if (n === 2) worst2 = Math.max(worst2, Math.abs(patch2(cs, tap) - patchN(cs, 2, tap)));
      if (want > 0 && want < 1) mixed++;
      cases++;
      for (let k = 0; k < n - 1; k++) {
        if (cs[k].z > 1) cover.zOver++; else if (edgeOf(cs[k]) > 0.39 && edgeOf(cs[k]) < 0.47) cover.band++;
      }
      const f = cs[n - 1], ef = edgeOf(f);
      if (n > 1 && ef > 0.43 && ef < 0.495 && f.z <= 1) cover.fade++;
      if (ef > 0.5) cover.outside++;
    }
  }
  check('xi-averaged single tap = the patch\'s blended lookup, 1-4 cascades (1e-9)', worst < 1e-9, `${cases} samples, worst ${fmt(worst)}`);
  check('covered: inner z > 1, blend band, far fade, outside the far map, blended results', Object.values(cover).every((v) => v > 200) && mixed > 500,
    `${cover.zOver} / ${cover.band} / ${cover.fade} / ${cover.outside}, ${mixed} blended`);
  check('2-cascade patch text = the general text at 2 cascades (ports agree)', worst2 < 1e-12);
  // a fixed xi is one sample of that average: deterministic (jitter off) picks the near map up to the band middle
  const one = (e) => [{ x: 0.5 + e, y: 0.5, z: 0.5 }, { x: 0.5, y: 0.5, z: 0.5 }];
  const tapK = (k) => k; // returns the cascade index tapped
  check('jitter off (xi 0.5): near map up to 0.43 of the box, then the next', volVisibility(one(0.42), 2, tapK, 0.5) === 0 && volVisibility(one(0.44), 2, tapK, 0.5) === 1);
}

// ================================================================================================= history tap
// port of the shader's depth-aware 2x2 history tap; tex(ix, iy) -> [r, g]; returns the history r or null (rejected)
function historyTap(tex, w, h, huv, z, zPrev) {
  const px = huv[0] * w - 0.5, py = huv[1] * h - 0.5, x0 = Math.floor(px), y0 = Math.floor(py), fx = px - x0, fy = py - y0;
  const at = (x, y) => tex(clamp(x, 0, w - 1), clamp(y, 0, h - 1));
  const T = [at(x0, y0), at(x0 + 1, y0), at(x0, y0 + 1), at(x0 + 1, y0 + 1)];
  const m = T.map(([, g]) => 1 - Math.min(Math.abs(g - z), Math.abs(g - zPrev)) / (0.1 * z + 0.3));
  if (Math.max(...m) < 0) return null;
  const wt = [(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy].map((v, i) => v * Math.max(1e-3, m[i]));
  return wt.reduce((a, v, i) => a + v * T[i][0], 0) / wt.reduce((a, v) => a + v, 0);
}
{
  const fs = VolumetricPassFrag();
  check('shader: 2x2 texelFetch history tap, aoAt weights, TAAPass tolerance, reject only if all four fail',
    ['vec2 ha = texelFetch( tHistory, clamp( h0, ivec2( 0 ), hm ), 0 ).rg;', 'vec2 hd = texelFetch( tHistory, clamp( h0 + 1, ivec2( 0 ), hm ), 0 ).rg;',
      'vec4 m = 1.0 - min( abs( hz - z ), abs( hz - zPrev ) ) / ( 0.1 * z + 0.3 );', 'if ( max( max( m.x, m.y ), max( m.z, m.w ) ) >= 0.0 ) {',
      '* max( vec4( 1e-3 ), m );', 'r = mix( r, dot( w, vec4( ha.x, hb.x, hc.x, hd.x ) ) / dot( w, vec4( 1.0 ) ), uTemporal.x );'].every((x) => fs.includes(x))
    && !fs.includes('texture2D( tHistory'));
  // a silhouette against the sky in the history: left column 10 m (r 0.2), right column sky (r 0.9); tap halfway
  const sil = (x) => (x === 0 ? [0.2, 10] : [0.9, 60000]);
  const hs = historyTap(sil, 2, 2, [0.5, 0.5], 10, 10.2);
  check('silhouette, surface pixel: history kept (its own texels), no bleed from the sky', hs !== null && Math.abs(hs - 0.2) < 1e-3, hs === null ? 'rejected' : fmt(hs));
  const bilinearG = 0.5 * 10 + 0.5 * 60000;
  check('  (the bilinear-g test rejected it: g 30005 vs 10 m)', Math.abs(bilinearG - 10) > 0.1 * 10 + 0.3);
  const hk = historyTap(sil, 2, 2, [0.5, 0.5], 60000, 60000);
  check('silhouette, sky pixel: history kept, the sky texels', hk !== null && Math.abs(hk - 0.9) < 1e-3, hk === null ? 'rejected' : fmt(hk));
  check('all four texels another surface (50 m vs 10 m): rejected', historyTap(() => [0.5, 50], 2, 2, [0.5, 0.5], 10, 10.5) === null);
  const fwd = historyTap(() => [0.4, 11.2], 2, 2, [0.3, 0.7], 10, 11.2);
  check('camera moving forward (history at the previous depth): kept through zPrev', fwd !== null && Math.abs(fwd - 0.4) < 1e-12);
  const flat = historyTap((x, y) => [0.1 * x + 0.3 * y, 20], 2, 2, [0.6, 0.35], 20, 20);
  check('one surface: plain bilinear', Math.abs(flat - (0.1 * (0.6 * 2 - 0.5) + 0.3 * (0.35 * 2 - 0.5))) < 1e-12);
}

// ================================================================================================= define switching
{
  const atmo = fakeAtmosphere(), vol = new VolumetricPass(makeCamera(), null, atmo);
  vol.setSize(W, H);
  const r = stubRenderer(), m = vol.material, d = m.defines;
  const base = Object.getOwnPropertyDescriptor(THREE.Material.prototype, 'needsUpdate');
  let nu = 0;
  Object.defineProperty(m, 'needsUpdate', { configurable: true, set(v) { if (v === true) nu++; base.set.call(this, v); } });
  atmo.st.count = 2; atmo.st.packed = true;
  vol.render(r, null, readBuffer);
  check('2 packed cascades: VOL_CASCADES 2 + VOL_SHADOW_PACKED, needsUpdate once', d.VOL_CASCADES === 2 && 'VOL_SHADOW_PACKED' in d && nu === 1);
  for (let i = 0; i < 5; i++) vol.render(r, null, readBuffer);
  check('same cascades x5: no needsUpdate', nu === 1);
  check('packed: tShadow0/1 = the packed maps, tShadow2/3 unbound', m.uniforms.tShadow0.value === atmo.maps[0] && m.uniforms.tShadow1.value === atmo.maps[1] && m.uniforms.tShadow2.value === null && m.uniforms.tShadow3.value === null);
  atmo.st.count = 3; vol.render(r, null, readBuffer); vol.render(r, null, readBuffer);
  check('count 2 -> 3: one needsUpdate, VOL_CASCADES 3', nu === 2 && d.VOL_CASCADES === 3);
  atmo.st.packed = false; vol.render(r, null, readBuffer); vol.render(r, null, readBuffer);
  check('packed -> compare: one needsUpdate, VOL_SHADOW_PACKED removed', nu === 3 && !('VOL_SHADOW_PACKED' in d) && d.VOL_CASCADES === 3);
  check('compare: tShadow0..2 = the depth textures', [0, 1, 2].every((k) => m.uniforms[`tShadow${k}`].value === atmo.depths[k]));
  atmo.st.count = 4; atmo.st.packed = true; vol.render(r, null, readBuffer);
  check('count + path at once: one needsUpdate', nu === 4 && d.VOL_CASCADES === 4 && 'VOL_SHADOW_PACKED' in d);
  atmo.st.count = 6; vol.render(r, null, readBuffer);
  check('more than 4 cascades: clamped to 4, no needsUpdate', nu === 4 && d.VOL_CASCADES === 4);
  atmo.st.count = 1; vol.render(r, null, readBuffer);
  check('count 1: one needsUpdate', nu === 5 && d.VOL_CASCADES === 1);
  atmo.st.ready = false; atmo.st.count = 3; vol.render(r, null, readBuffer);
  check('maps not rendered: defines untouched (zero output)', nu === 5 && d.VOL_CASCADES === 1);
  vol.dispose();
}

// ================================================================================================= render plumbing
{
  const atmo = fakeAtmosphere(), cam = makeCamera(5), vol = new VolumetricPass(cam, null, atmo);
  vol.setSize(W, H);
  const r = stubRenderer(), u = vol.material.uniforms;
  // zero output
  atmo.st.ready = false;
  vol.render(r, null, readBuffer); vol.render(r, null, readBuffer);
  check('maps null: one clear, no draw, texture valid', r.stats.clears === 1 && r.stats.draws === 0 && vol.texture.isTexture);
  atmo.st.ready = true;
  vol.render(r, null, { depthTexture: null });
  check('no depth texture: zero output (no draw)', r.stats.draws === 0);
  vol.render(r, null, readBuffer);
  check('maps ready: one draw into the output texture', r.stats.draws === 1 && r.stats.lastTarget.texture === vol.texture && u.uTemporal.value.y === 1);
  // gate: Atmosphere's sun / moon switch (day = smoothstep(-2, 3 deg, elevation) > 0.02), no weight (sun.intensity has it)
  const sun = atmo.sunDir.clone(), elev = (deg) => atmo.sunDir.set(Math.cos(deg * DEG), Math.sin(deg * DEG), 0);
  elev(-20); vol.render(r, null, readBuffer);
  check('sun at -20 deg: zero output (cleared, no draw), gate 0 read back', r.stats.draws === 1 && r.stats.clears === 2 && u.uMarch.value.y === 0);
  elev(1); vol.render(r, null, readBuffer);
  check('sun at +1 deg: marched with gate 1 (no smoothstep weight)', r.stats.draws === 2 && u.uMarch.value.y === 1);
  elev(-1.3); vol.render(r, null, readBuffer);
  check("sun at -1.3 deg (day 0.053 > 0.02: still the sun's cascades): marched with gate 1", r.stats.draws === 3 && u.uMarch.value.y === 1);
  elev(-1.7); vol.render(r, null, readBuffer);
  check('sun at -1.7 deg (day 0.010: the moon\'s cascades): zero output, gate 0', r.stats.draws === 3 && r.stats.clears === 3 && u.uMarch.value.y === 0);
  atmo.sunDir.copy(sun); vol.render(r, null, readBuffer);
  check('high sun: gate 1', u.uMarch.value.y === 1);
  vol.density = 0; const c0 = r.stats.clears; vol.render(r, null, readBuffer);
  check('density 0: zero output, the frame scalars still written', r.stats.clears === c0 + 1 && r.stats.draws === 4 && u.uMedium.value.x === 0 && u.uMedium.value.z === 0);
  vol.density = 1; vol.render(r, null, readBuffer);
  check('frame uniforms: steps, maxDistance, sun direction, near / far, camera matrices', u.uSteps.value === 16 && u.uMarch.value.x === 400 && u.uSunDir.value.equals(atmo.sunDir)
    && u.uClip.value.x === 0.1 && u.uClip.value.y === 6000 && u.uProjInv.value.equals(cam.projectionMatrixInverse) && u.uCamWorld.value.equals(cam.matrixWorld) && u.tDepth.value === readBuffer.depthTexture);
  const camPos = new THREE.Vector3().setFromMatrixPosition(cam.matrixWorld);
  let origin = true;
  for (let i = 0; i < 2; i++) origin &&= u.uShadowOrigin.value[i].distanceTo(camPos.clone().applyMatrix4(atmo.lights[i].shadow.matrix)) < 1e-9 && u.uShadowMatrix.value[i].equals(atmo.lights[i].shadow.matrix);
  check('shadow uniforms: matrices, camera origin through each, light bias per cascade', origin && u.uShadowBias.value[0] === Math.fround(atmo.lights[0].shadow.bias) && u.uShadowBias.value[1] === Math.fround(atmo.lights[1].shadow.bias) && u.uShadowBias.value[2] === 0);
  // a sample at distance t along a ray: shadow coordinate = origin + mat3(M) * dir * t (what the shader steps)
  const dir = new THREE.Vector3(0.2, -0.1, -1).normalize(), M = atmo.lights[1].shadow.matrix, tt = 137;
  const direct = camPos.clone().addScaledVector(dir, tt).applyMatrix4(M);
  const stepped = u.uShadowOrigin.value[1].clone().addScaledVector(dir.clone().applyMatrix3(new THREE.Matrix3().setFromMatrix4(M)), tt);
  check('shadow coordinate is affine in t (origin + mat3(M) dir t = M (cam + dir t))', direct.distanceTo(stepped) < 1e-9);
  vol.steps = 40; vol.maxDistance = 900; vol.render(r, null, readBuffer);
  check('live options re-read per render', u.uSteps.value === 40 && u.uMarch.value.x === 900);
  vol.jitter = false; vol.render(r, null, readBuffer);
  const off = u.uMarch.value.z === 0 && u.uMarch.value.w === 0;
  vol.jitter = true; const offs = [];
  for (let i = 0; i < 4; i++) { vol.render(r, null, readBuffer); offs.push(u.uMarch.value.w); }
  check('jitter off: no noise; on: frame offset in [0, 1), changing every frame', off && u.uMarch.value.z === 1 && offs.every((x) => x >= 0 && x < 1) && new Set(offs).size === 4);
  vol.dispose();
}

// ================================================================================================= temporal, with a CameraHistory
{
  const atmo = fakeAtmosphere(), cam = makeCamera(2), hist = new CameraHistory(cam);
  const vol = new VolumetricPass(cam, hist, atmo);
  vol.setSize(W, H);
  const r = stubRenderer(), u = vol.material.uniforms;
  const frame = (render = true) => { hist.begin(W, H); if (render) vol.render(r, null, readBuffer); hist.end(); };
  frame();
  const t1 = vol.texture;
  check('first frame: fresh, tHistory = the depth texture (valid, never read)', u.uTemporal.value.y === 1 && u.tHistory.value === readBuffer.depthTexture);
  frame();
  check('next frame: history = last output, written to the other target, not fresh', u.uTemporal.value.y === 0 && u.tHistory.value === t1 && vol.texture !== t1 && r.stats.lastTarget.texture === vol.texture);
  check('reprojection uniforms = CameraHistory (uReproj, jitter, inverse row 4)', u.uReproj.value.equals(hist.uniforms.uReproj.value) && u.uJitter.value.equals(hist.uniforms.uJitter.value)
    && u.uPrevJitter.value.equals(hist.uniforms.uPrevJitter.value) && u.uInvVPw.value.x === hist.uniforms.uInvViewProj.value.elements[3] && u.uInvVPw.value.w === hist.uniforms.uInvViewProj.value.elements[15]);
  check('blend uniform = blend', u.uTemporal.value.x === 0.9);
  frame(false); frame();
  check('a frame this pass skipped: fresh', u.uTemporal.value.y === 1);
  frame();
  const steady = u.uTemporal.value.y === 0;
  vol.reset(); frame();
  check('reset(): fresh, then accumulates again', steady && u.uTemporal.value.y === 1 && (frame(), u.uTemporal.value.y === 0));
  hist.reset(); frame();
  check('CameraHistory.reset(): fresh', u.uTemporal.value.y === 1);
  frame(); vol.setSize(W, H); frame();
  check('setSize: fresh', u.uTemporal.value.y === 1);
  vol.temporal = false; frame();
  const tOff = vol.texture; frame(); frame();
  check('temporal off: one target, fresh, tHistory = the depth texture', vol.texture === tOff && u.uTemporal.value.y === 1 && u.tHistory.value === readBuffer.depthTexture && r.stats.lastTarget.texture === tOff);
  vol.temporal = true; frame();
  check('temporal back on: fresh first', u.uTemporal.value.y === 1 && (frame(), u.uTemporal.value.y === 0));
  atmo.st.ready = false; frame(); atmo.st.ready = true; frame();
  check('after zero output: fresh', u.uTemporal.value.y === 1);
  // no allocation per frame (begin / render / end), steady state
  gate(inRange, note, 'render() with CameraHistory (begin / render / end)', bytesPerCall((n) => { for (let i = 0; i < n; i++) frame(); }, 50000), { unit: 'frame' });
  vol.dispose();
}

// ================================================================================================= temporal, no CameraHistory
{
  const atmo = fakeAtmosphere(), cam = makeCamera(2), vol = new VolumetricPass(cam, null, atmo);
  vol.setSize(W, H);
  const r = stubRenderer(), u = vol.material.uniforms;
  vol.render(r, null, readBuffer);
  check('no history: first frame fresh', u.uTemporal.value.y === 1);
  vol.render(r, null, readBuffer);
  let ident = 0;
  for (let i = 0; i < 16; i++) ident = Math.max(ident, Math.abs(u.uReproj.value.elements[i] - (i % 5 === 0 ? 1 : 0)));
  check('no history, static camera: accumulates, uReproj = I', u.uTemporal.value.y === 0 && ident < 1e-9, fmt(ident));
  // moved camera: uReproj takes a point's current NDC to its previous clip position
  const X = new THREE.Vector3(20, 3, -50);
  const prevVP = new THREE.Matrix4().multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse);
  cam.position.x += 0.7; cam.rotation.y += 0.02; cam.updateMatrixWorld();
  vol.render(r, null, readBuffer);
  const cur = new THREE.Vector4(X.x, X.y, X.z, 1).applyMatrix4(new THREE.Matrix4().multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse));
  const ndc = new THREE.Vector4(cur.x / cur.w, cur.y / cur.w, cur.z / cur.w, 1);
  const pc = ndc.clone().applyMatrix4(u.uReproj.value), want = new THREE.Vector4(X.x, X.y, X.z, 1).applyMatrix4(prevVP);
  const iw = u.uInvVPw.value, hw = iw.x * ndc.x + iw.y * ndc.y + iw.z * ndc.z + iw.w;
  check('no history, moving camera: uReproj = previous view-projection * inverse(current)', Math.abs(pc.x / pc.w - want.x / want.w) < 1e-6 && Math.abs(pc.y / pc.w - want.y / want.w) < 1e-6);
  check('no history: previous view depth = pc.w / dot(uInvVPw, ndc)', Math.abs(pc.w / hw - want.w) < 1e-4, fmt(pc.w / hw - want.w));
  // frames this pass skipped (the composer saw enabled false): the reprojection is from the last frame IT rendered
  const lastVP = new THREE.Matrix4().multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse);
  cam.position.z -= 1.5; cam.updateMatrixWorld(); // a frame the pass did not render
  cam.position.z -= 1.5; cam.rotation.y -= 0.03; cam.updateMatrixWorld();
  vol.render(r, null, readBuffer);
  const cur2 = new THREE.Vector4(X.x, X.y, X.z, 1).applyMatrix4(new THREE.Matrix4().multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse));
  const ndc2 = new THREE.Vector4(cur2.x / cur2.w, cur2.y / cur2.w, cur2.z / cur2.w, 1);
  const pc2 = ndc2.clone().applyMatrix4(u.uReproj.value), want2 = new THREE.Vector4(X.x, X.y, X.z, 1).applyMatrix4(lastVP);
  check('no history, frames skipped: reprojects from the last frame the pass rendered', u.uTemporal.value.y === 0
    && Math.abs(pc2.x / pc2.w - want2.x / want2.w) < 1e-6 && Math.abs(pc2.y / pc2.w - want2.y / want2.w) < 1e-6);
  const hist = new CameraHistory(cam);
  vol.history = hist; hist.begin(W, H); hist.end(); hist.begin(W, H); vol.render(r, null, readBuffer); hist.end();
  const swapped = u.uTemporal.value.y === 1;
  hist.begin(W, H); vol.render(r, null, readBuffer); hist.end();
  vol.history = null; vol.render(r, null, readBuffer);
  check('history replaced (null -> CameraHistory -> null, as PostFX setGBuffer): fresh each time', swapped && u.uTemporal.value.y === 1 && (vol.render(r, null, readBuffer), u.uTemporal.value.y === 0));
  gate(inRange, note, 'render() without history', bytesPerCall((n) => { for (let i = 0; i < n; i++) vol.render(r, null, readBuffer); }, 50000), { unit: 'frame' });
  vol.dispose();
}

// ================================================================================================= real (headless) Atmosphere
{
  const scene = new THREE.Scene(), cam = makeCamera(2);
  const atmo = new Atmosphere({ scene, renderer: null, camera: cam, timeOfDay: 'afternoon', shadows: { cascades: 3 } });
  const vol = new VolumetricPass(cam, null, atmo);
  vol.setSize(W, H);
  const r = stubRenderer(), u = vol.material.uniforms;
  vol.render(r, null, readBuffer);
  check('Atmosphere before the first shadow render (maps null): zero output', r.stats.draws === 0 && r.stats.clears === 1);
  for (const l of atmo.cascadeLights) l.shadow.map = new THREE.WebGLRenderTarget(4, 4);
  vol.render(r, null, readBuffer);
  check('Atmosphere, 3 cascades, packed maps: marched, VOL_CASCADES 3, maps bound', r.stats.draws === 1 && vol.material.defines.VOL_CASCADES === 3 && 'VOL_SHADOW_PACKED' in vol.material.defines
    && [0, 1, 2].every((k) => u[`tShadow${k}`].value === atmo.cascadeLights[k].shadow.map.texture));
  const fp = atmo.fogParams;
  check('Atmosphere: medium from its fogParams, sun gate from its sunDir', rel(u.uMedium.value.x, fp.haze * Math.exp(-fp.hazeK * (2 - (fp.hazeBase || 0)))) < 1e-12
    && rel(u.uMedium.value.z, fp.mist * Math.exp(-fp.mistK * 2)) < 1e-12 && u.uMarch.value.y > 0.9 && u.uSunDir.value.equals(atmo.sunDir));
  for (const l of atmo.cascadeLights) { const m = l.shadow.map; l.shadow.map = new THREE.WebGLRenderTarget(4, 4, { depthTexture: new THREE.DepthTexture(4, 4) }); m.dispose(); }
  vol.render(r, null, readBuffer);
  check("Atmosphere with depth textures ('fast' / 'pcss'): compare path, depth textures bound", !('VOL_SHADOW_PACKED' in vol.material.defines)
    && [0, 1, 2].every((k) => u[`tShadow${k}`].value === atmo.cascadeLights[k].shadow.map.depthTexture));
  atmo.setTime(22); vol.render(r, null, readBuffer);
  check('Atmosphere at night: zero output (no draw)', r.stats.draws === 2 && u.uMarch.value.y === 0);
  let agree = true, nSun = 0, nMoon = 0;
  for (let hr = 18.3; hr <= 19.0; hr += 0.005) {
    atmo.setTime(hr); vol.render(r, null, readBuffer);
    const sunCasts = atmo.lightDir.equals(atmo.sunDir); // Atmosphere: the cascade lights follow the sun while day > 0.02
    if (sunCasts) nSun++; else nMoon++;
    agree &&= u.uMarch.value.y === (sunCasts ? 1 : 0);
  }
  check("Atmosphere sunset sweep: gate 1 exactly while the cascades are the sun's", agree && nSun > 0 && nMoon > 0, `${nSun} sun / ${nMoon} moon`);
  atmo.setTime(15.5);
  gate(inRange, note, 'render() on a real Atmosphere (its shadowCascadeInfo)', bytesPerCall((n) => { for (let i = 0; i < n; i++) vol.render(r, null, readBuffer); }, 50000), { unit: 'frame' });
  check('dispose: no throw', !throws(() => vol.dispose()));
}

console.log(results.join('\n'));
console.log(failures ? `\n${failures} of ${passed + failures} checks FAILED` : `\nall ${passed} checks passed`);
process.exit(failures ? 1 : 0);
