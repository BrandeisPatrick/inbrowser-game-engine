// Headless checks for the 0.8.0 lens passes (no browser, no WebGL): the thin-lens circle of confusion and the Vogel
// disc of DoFPass, the 3D LUT helpers of render/Lut.js (identity, function LUTs, the CPU lookup against an independent
// GL-style trilinear sampler, the loader's texel convention), and MotionBlurPass / DoFPass option validation, target
// sizing, shader contracts (early-outs, uniforms) and an allocation-free render() on a stub renderer (draws are no-ops,
// so the GLSL itself is checked in the browser).
// Run: node test/blur-dof-lut.mjs
import v8 from 'node:v8';
import * as THREE from 'three';
import { MotionBlurPass, MOTION_BLUR_DEFAULTS } from '../src/render/passes/MotionBlurPass.js';
import { DoFPass, DOF_DEFAULTS, circleOfConfusion, vogelDisc } from '../src/render/passes/DoFPass.js';
import { LUT_GLSL, LUT_DEFAULT_SIZE, identityLut, lutFromFunction, lutLookup, disposeLut } from '../src/render/Lut.js';
import { parseCubeLUT } from '../src/render/loaders.js';
import { CameraHistory } from '../src/render/CameraHistory.js';

let failures = 0, total = 0;
const results = [];
function check(name, value, lo = 1, hi = lo, unit = '') {
  if (typeof value === 'boolean') value = value ? 1 : 0;
  const ok = Number.isFinite(value) && value >= lo && value <= hi;
  total++;
  if (!ok) failures++;
  results.push(`${ok ? 'PASS' : 'FAIL'}  ${name.padEnd(72)} ${fmt(value)} ${unit}   [${fmt(lo).trim()}${hi !== lo ? ' .. ' + fmt(hi).trim() : ''}]`);
  return ok;
}
const fmt = (x) => (!Number.isFinite(x) ? String(x) : Number.isInteger(x) ? String(x) : Math.abs(x) < 1e-3 ? x.toExponential(2) : x.toFixed(4)).padStart(10);
const throws = (fn, re) => { try { fn(); return false; } catch (e) { return re ? re.test(e.message) : true; } };
const near = (a, b, eps) => Math.abs(a - b) <= eps;

// deterministic random numbers (mulberry32)
let seed = 0x9e3779b9;
const rnd = () => { seed |= 0; seed = (seed + 0x6d2b79f5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };

performance.now(); process.memoryUsage();
function measureAlloc(fn) {
  const prof = new v8.GCProfiler();
  prof.start();
  const h0 = process.memoryUsage().heapUsed;
  fn();
  const h1 = process.memoryUsage().heapUsed;
  return { bytes: h1 - h0, gcs: prof.stop().statistics.length };
}

// enough of WebGLRenderer for render(): binds targets, counts draws and clears (no allocation), optionally records
// each draw with the target it went to, whether that target was cleared first and the autoClear state at the time
function stubRenderer() {
  const r = {
    bound: null, draws: 0, clears: 0, autoClear: true, log: null, clearedBound: null, clearArgs: '',
    getRenderTarget() { return r.bound; },
    setRenderTarget(t) { r.bound = t; },
    clear(color, depth, stencil) { r.clears++; r.clearedBound = r.bound; if (r.log) r.clearArgs = `${color},${depth},${stencil}`; },
    render(mesh) {
      r.draws++;
      if (r.log) r.log.push({ target: r.bound, material: mesh.material, cleared: r.clearedBound === r.bound, autoClear: r.autoClear, clearArgs: r.clearArgs });
      r.clearedBound = null;
    },
  };
  return r;
}
// every logged draw went to a target cleared (colour only) right before it, with autoClear off
const clearedDraws = (log) => log.every((e) => e.cleared && e.autoClear === false && e.clearArgs === 'true,false,false');
// a readBuffer stand-in (the compositor's colorSource proxy): colour + scene depth
const readBuffer = { texture: new THREE.Texture(), depthTexture: new THREE.DepthTexture(4, 4) };
// GLSL sanity that needs no compiler: balanced brackets and no leftover template placeholders
function balanced(src) {
  const open = '([{', close = ')]}', st = [];
  for (const ch of src) {
    if (open.includes(ch)) st.push(ch);
    else if (close.includes(ch)) { if (st.pop() !== open[close.indexOf(ch)]) return false; }
  }
  return st.length === 0 && !src.includes('${') && !src.includes('undefined');
}

// ================================================================================================ circle of confusion
{
  // 50 mm f/2 (A = 25 mm), focus 10 m, 1440 px over a 24 mm sensor (60 px / mm). cocMm = A |d - D| f / (d (D - f))
  const H = 1440, S = 24;
  const at5 = circleOfConfusion(5, 10, 50, 2, S, H), at20 = circleOfConfusion(20, 10, 50, 2, S, H);
  // 25 * 5000 * 50 / (5000 * 9950) = 0.12562814 mm; 25 * 10000 * 50 / (20000 * 9950) = 0.06281407 mm
  check('CoC 50 mm f/2, focus 10 m, object 5 m = -0.125628 mm * 60', at5, -7.537689 - 1e-5, -7.537689 + 1e-5, 'px');
  check('CoC 50 mm f/2, focus 10 m, object 20 m = +0.062814 mm * 60', at20, 3.768844 - 1e-5, 3.768844 + 1e-5, 'px');
  check('CoC sign: near negative, far positive', at5 < 0 && at20 > 0);
  check('CoC at the focus distance', Math.abs(circleOfConfusion(10, 10, 50, 2, S, H)), 0, 1e-12, 'px');
  let mono = true, prev = -Infinity;
  for (let d = 0.5; d < 500; d *= 1.1) { const c = circleOfConfusion(d, 10, 50, 2, S, H); mono &&= c > prev; prev = c; }
  check('CoC increases monotonically with distance', mono);
  const inf = circleOfConfusion(Infinity, 10, 50, 2, S, H);
  check('CoC at infinity = A f / (D - f) = 0.125628 mm (= -CoC at D / 2)', inf, 7.537689 - 1e-5, 7.537689 + 1e-5, 'px');
  check('CoC at D / 2 is minus the CoC at infinity', Math.abs(at5 + inf), 0, 1e-9, 'px');
  // hyperfocal distance for c = 0.03 mm: Hf = f^2 / (N c) + f; focused there, everything beyond is within c
  const c = 0.03, Hf = (50 * 50 / (2 * c) + 50) / 1000;
  let worst = 0, sane = true;
  for (let d = Hf; d < 1e6; d *= 1.5) { const mm = circleOfConfusion(d, Hf, 50, 2, S, 1) * S; sane &&= mm >= -1e-12; worst = Math.max(worst, mm); }
  check('hyperfocal focus: 0 <= CoC <= c beyond the focus distance', sane && worst <= c + 1e-12, 1);
  check('hyperfocal focus: CoC at infinity = c (0.03 mm)', circleOfConfusion(Infinity, Hf, 50, 2, S, 1) * S, c - 1e-9, c + 1e-9, 'mm');
  check('hyperfocal focus: CoC at Hf / 2 = -c (near limit)', circleOfConfusion(Hf / 2, Hf, 50, 2, S, 1) * S, -c - 1e-9, -c + 1e-9, 'mm');
  check('px scale linear in bufferHeight (2880 = 2 x 1440)', circleOfConfusion(5, 10, 50, 2, S, 2880) / at5, 2 - 1e-12, 2 + 1e-12);
  check('px scale inverse in sensorHeight (12 mm = 2 x 24 mm)', circleOfConfusion(5, 10, 50, 2, 12, H) / at5, 2 - 1e-12, 2 + 1e-12);
  check('f/4 halves the CoC of f/2', circleOfConfusion(20, 10, 50, 4, S, H) / at20, 0.5 - 1e-12, 0.5 + 1e-12);
  check('unclamped: 0.3 m at 50 mm f/2 is beyond 64 px', -circleOfConfusion(0.3, 10, 50, 2, S, H), 64, Infinity, 'px');
}

// ================================================================================================ Vogel disc
{
  const n = 24, k = vogelDisc(n);
  let inside = true, mx = 0, my = 0, rising = true, prevR = -1;
  for (let i = 0; i < n; i++) {
    const r = Math.hypot(k[2 * i], k[2 * i + 1]);
    inside &&= r <= 1 + 1e-6; rising &&= r > prevR; prevR = r; mx += k[2 * i] / n; my += k[2 * i + 1] / n;
  }
  check('vogelDisc(24): 48 numbers, all inside the unit disc', k.length === 48 && inside);
  check('vogelDisc(24): radius sqrt((i + 0.5) / n), increasing', rising && near(Math.hypot(k[0], k[1]), Math.sqrt(0.5 / 24), 1e-6));
  check('vogelDisc(24): |centroid|', Math.hypot(mx, my), 0, 0.06);
}

// ================================================================================================ LUTs
{
  check('LUT_GLSL declares engineApplyLut exactly', LUT_GLSL.includes('vec3 engineApplyLut( highp sampler3D lut, vec3 c, float size ) { float s = ( size - 1.0 ) / size; return texture( lut, c * s + 0.5 / size ).rgb; }'));
  check('LUT_GLSL balanced', balanced(LUT_GLSL));
  const id = identityLut();
  const t = id.texture;
  check('identityLut(): size LUT_DEFAULT_SIZE (16), title identity', id.size === 16 && LUT_DEFAULT_SIZE === 16 && id.title === 'identity');
  check('identityLut(): Data3DTexture RGBA half float, 16^3 x 4 Uint16', t.isData3DTexture && t.format === THREE.RGBAFormat && t.type === THREE.HalfFloatType
    && t.image.data instanceof Uint16Array && t.image.data.length === 16 ** 3 * 4 && t.image.width === 16 && t.image.height === 16 && t.image.depth === 16);
  check('identityLut(): linear, clamp-to-edge (s, t, r), no mipmaps, uploaded', t.minFilter === THREE.LinearFilter && t.magFilter === THREE.LinearFilter
    && t.wrapS === THREE.ClampToEdgeWrapping && t.wrapT === THREE.ClampToEdgeWrapping && t.wrapR === THREE.ClampToEdgeWrapping && !t.generateMipmaps && t.version > 0);
  check('identityLut(): domain [0, 1]^3 (loader shape)', id.domainMin.join() === '0,0,0' && id.domainMax.join() === '1,1,1');
  // texel convention: x = r fastest, y = g, z = b; alpha 1
  const h = (v) => THREE.DataUtils.fromHalfFloat(v), D = t.image.data, at = (x, y, z, ch) => h(D[((z * 16 + y) * 16 + x) * 4 + ch]);
  check('texel (x 3, y 7, z 11) = (3, 7, 11) / 15, alpha 1', near(at(3, 7, 11, 0), 3 / 15, 5e-4) && near(at(3, 7, 11, 1), 7 / 15, 5e-4) && near(at(3, 7, 11, 2), 11 / 15, 5e-4) && at(3, 7, 11, 3) === 1);
  const out = [0, 0, 0];
  let worst = 0;
  for (let i = 0; i < 20; i++) {
    const r = rnd(), g = rnd(), b = rnd();
    lutLookup(id, r, g, b, out);
    worst = Math.max(worst, Math.abs(out[0] - r), Math.abs(out[1] - g), Math.abs(out[2] - b));
  }
  check('identity lookup returns the input (20 random points)', worst, 0, 1e-3);
  check('lookup writes into and returns out', lutLookup(id, 0.5, 0.5, 0.5, out) === out);
  lutLookup(id, -0.5, 1.5, 0.25, out);
  check('lookup clamps inputs to 0..1 (clamp-to-edge)', near(out[0], 0, 1e-6) && near(out[1], 1, 1e-6) && near(out[2], 0.25, 1e-3));
  check('identityLut(33): odd size, lookup identity', (() => { const l = identityLut(33); const o = lutLookup(l, 0.123, 0.456, 0.789); return near(o[0], 0.123, 1e-3) && near(o[1], 0.456, 1e-3) && near(o[2], 0.789, 1e-3); })());

  // independent GL-style sampler: u = c (size - 1) / size + 0.5 / size, texel coord u * size - 0.5, clamp-to-edge
  const glSample = (lut, c) => {
    const n = lut.size, s = (n - 1) / n, d = lut.texture.image.data, half = lut.texture.type === THREE.HalfFloatType;
    const tc = c.map((v) => (v * s + 0.5 / n) * n - 0.5);
    const i0 = tc.map(Math.floor), f = tc.map((v, k) => v - i0[k]), cl = (v) => Math.min(n - 1, Math.max(0, v));
    const res = [0, 0, 0];
    for (let corner = 0; corner < 8; corner++) {
      const o = [corner & 1, (corner >> 1) & 1, (corner >> 2) & 1];
      const w = o.reduce((p, b, k) => p * (b ? f[k] : 1 - f[k]), 1);
      const x = cl(i0[0] + o[0]), y = cl(i0[1] + o[1]), z = cl(i0[2] + o[2]);
      for (let ch = 0; ch < 3; ch++) { const v = d[((z * n + y) * n + x) * 4 + ch]; res[ch] += w * (half ? h(v) : v); }
    }
    return res;
  };
  const grade = (r, g, b, o) => { o[0] = r * r; o[1] = 0.5 * (g + b * b); o[2] = 0.2 + 0.6 * r * b + 0.1 * g * g * (3 - 2 * g); };
  const fl = lutFromFunction(17, grade, 'test-grade');
  let gl = 0;
  for (let i = 0; i < 200; i++) {
    const c = [rnd() * 1.4 - 0.2, rnd() * 1.4 - 0.2, rnd() * 1.4 - 0.2];
    const a = lutLookup(fl, c[0], c[1], c[2]), b = glSample(fl, c);
    gl = Math.max(gl, Math.abs(a[0] - b[0]), Math.abs(a[1] - b[1]), Math.abs(a[2] - b[2]));
  }
  check('lutLookup = GL half-texel trilinear mapping (200 points, -0.2..1.2)', gl, 0, 1e-9);
  check('lutFromFunction: title, size, half float', fl.title === 'test-grade' && fl.size === 17 && fl.texture.type === THREE.HalfFloatType && fl.texture.image.data.length === 17 ** 3 * 4);
  // grid points come back exactly (the function's value, half-float rounded)
  const ref = [0, 0, 0], q = (v) => h(THREE.DataUtils.toHalfFloat(v));
  let exact = true;
  for (let i = 0; i < 60; i++) {
    const x = Math.floor(rnd() * 17) / 16, y = Math.floor(rnd() * 17) / 16, z = Math.floor(rnd() * 17) / 16;
    grade(x, y, z, ref); lutLookup(fl, x, y, z, out);
    exact &&= out[0] === q(ref[0]) && out[1] === q(ref[1]) && out[2] === q(ref[2]);
  }
  check('lutFromFunction: grid points exact (60 random grid points)', exact);
  let between = 0;
  for (let i = 0; i < 200; i++) {
    const r = rnd(), g = rnd(), b = rnd();
    grade(r, g, b, ref); lutLookup(fl, r, g, b, out);
    between = Math.max(between, Math.abs(out[0] - ref[0]), Math.abs(out[1] - ref[1]), Math.abs(out[2] - ref[2]));
  }
  check('lutFromFunction: smooth function within 0.02 between grid points', between, 0, 0.02);
  let reused = null, same = true;
  lutFromFunction(3, (r, g, b, o) => { if (reused === null) reused = o; same &&= o === reused && Array.isArray(o) && o.length === 3; o[0] = r; o[1] = g; o[2] = b; });
  check('lutFromFunction: out is one reused 3-array', same && reused !== null);
  check('lutFromFunction: bad size / fn / non-finite output throw', throws(() => lutFromFunction(1, grade), /2\.\.256/) && throws(() => lutFromFunction(257, grade), /2\.\.256/)
    && throws(() => lutFromFunction(2.5, grade), /2\.\.256/) && throws(() => lutFromFunction(4, null), /function/) && throws(() => lutFromFunction(4, (r, g, b, o) => { o[0] = NaN; }), /non-finite/));

  // the Assets loader's LUTs are interchangeable: same texel layout, lutLookup reads float data too
  const n = 5, rows = [];
  for (let b = 0; b < n; b++) for (let g = 0; g < n; g++) for (let r = 0; r < n; r++) rows.push(`${r / (n - 1)} ${g / (n - 1)} ${b / (n - 1)}`);
  const cube = `TITLE "identity5"\nLUT_3D_SIZE ${n}\n${rows.join('\n')}\n`;
  const loaded = parseCubeLUT(cube), mine = identityLut(n);
  check('parseCubeLUT identity texels = identityLut texels (bit-equal)', loaded.texture.image.data.length === mine.texture.image.data.length && loaded.texture.image.data.every((v, i) => v === mine.texture.image.data[i]));
  const lf = parseCubeLUT(cube, { float: true });
  lutLookup(lf, 0.3, 0.6, 0.9, out);
  check('lutLookup on a float .cube LUT (identity)', near(out[0], 0.3, 1e-9) && near(out[1], 0.6, 1e-9) && near(out[2], 0.9, 1e-9));
  let disposed = 0;
  mine.texture.addEventListener('dispose', () => disposed++);
  disposeLut(mine);
  check('disposeLut disposes the texture', disposed, 1);
  // constant inputs: a double computed by the caller and passed to a non-inlined call is boxed on the caller's side
  for (let i = 0; i < 50000; i++) lutLookup(fl, 0.31, 0.47, 0.73, out); // warm (optimised code)
  const lutAlloc = measureAlloc(() => { for (let i = 0; i < 50000; i++) lutLookup(fl, 0.31, 0.47, 0.73, out); });
  check('lutLookup into out x50000: heap growth', lutAlloc.bytes, -Infinity, 64 * 1024, 'B');
}

// ================================================================================================ MotionBlurPass
{
  const cam = new THREE.PerspectiveCamera(60, 16 / 9, 0.1, 6000);
  const hist = new CameraHistory(cam);
  check('MOTION_BLUR_DEFAULTS: strength 1, maxBlur 24, tileSize 20, samples 8, shutter 0, frozen',
    MOTION_BLUR_DEFAULTS.strength === 1 && MOTION_BLUR_DEFAULTS.maxBlur === 24 && MOTION_BLUR_DEFAULTS.tileSize === 20 && MOTION_BLUR_DEFAULTS.samples === 8
    && MOTION_BLUR_DEFAULTS.shutter === 0 && Object.keys(MOTION_BLUR_DEFAULTS).join() === 'strength,maxBlur,tileSize,samples,shutter' && Object.isFrozen(MOTION_BLUR_DEFAULTS));
  const gbTex = { velocity: new THREE.Texture(), normal: new THREE.Texture() };
  const mb = new MotionBlurPass(cam, hist, gbTex);
  check('motion blur: defaults on the instance', mb.strength === 1 && mb.maxBlur === 24 && mb.tileSize === 20 && mb.samples === 8 && mb.shutter === 0);
  check('motion blur: Pass with needsSwap false, enabled', mb instanceof MotionBlurPass && mb.needsSwap === false && mb.enabled === true);
  check('motion blur: reset() is a no-op', !throws(() => mb.reset()));
  check('motion blur: needs a CameraHistory', throws(() => new MotionBlurPass(cam, null, gbTex), /CameraHistory/));
  check('motion blur: unknown option throws', throws(() => new MotionBlurPass(cam, hist, gbTex, { strenght: 1 }), /unknown option 'strenght'/));
  check('motion blur: strength 0 / -1 / NaN / Infinity throw', throws(() => new MotionBlurPass(cam, hist, gbTex, { strength: 0 }), /strength/)
    && throws(() => { mb.strength = -1; }, /strength/) && throws(() => { mb.strength = NaN; }) && throws(() => { mb.strength = Infinity; }) && mb.strength === 1);
  check('motion blur: maxBlur 0 / "24" throw', throws(() => { mb.maxBlur = 0; }, /maxBlur/) && throws(() => { mb.maxBlur = '24'; }, /maxBlur/));
  check('motion blur: tileSize 0 / 129 / 2.5 throw', throws(() => { mb.tileSize = 0; }, /tileSize/) && throws(() => { mb.tileSize = 129; }) && throws(() => { mb.tileSize = 2.5; }) && mb.tileSize === 20);
  check('motion blur: samples 1 / 65 / 8.5 throw', throws(() => { mb.samples = 1; }, /samples/) && throws(() => { mb.samples = 65; }) && throws(() => { mb.samples = 8.5; }) && mb.samples === 8);
  check('motion blur: shutter -0.01 / NaN / "1" throw', throws(() => { mb.shutter = -0.01; }, /shutter/) && throws(() => { mb.shutter = NaN; }, /shutter/)
    && throws(() => new MotionBlurPass(cam, hist, gbTex, { shutter: '1' }), /shutter/) && mb.shutter === 0);
  check("motion blur: 'scale' is no longer an option (unknown option throws)", throws(() => new MotionBlurPass(cam, hist, gbTex, { scale: 0.5 }), /unknown option 'scale'/) && !('scale' in mb));
  const mb32 = new MotionBlurPass(cam, hist, gbTex, { tileSize: 32, strength: 3, maxBlur: 10 });
  check('constructor applies the options before setSize (uTile 32, 1 x 1 targets)', mb32._tileMat.uniforms.uTile.value === 32 && mb32._reconstructMat.uniforms.uTile.value === 32
    && mb32._out.width === 1 && mb32._vel.width === 1);
  mb32.dispose();

  mb.setSize(3024, 1964);
  const vel = mb._vel;
  check('setSize 3024x1964: velocity MRT 2 x RG16F at full size', vel.textures.length === 2 && vel.width === 3024 && vel.height === 1964
    && vel.textures.every((x) => x.format === THREE.RGFormat && x.type === THREE.HalfFloatType));
  check('setSize: tile rows ceil(3024 / 20) x 1964 = 152 x 1964', mb._tileX.width === 152 && mb._tileX.height === 1964);
  check('setSize: tile max / neighbour max 152 x ceil(1964 / 20) = 99', mb._tile.width === 152 && mb._tile.height === 99 && mb._neighbor.width === 152 && mb._neighbor.height === 99);
  check('setSize: output RGBA16F 3024 x 1964, linear', mb._out.width === 3024 && mb._out.height === 1964 && mb.texture.type === THREE.HalfFloatType
    && mb.texture.format === THREE.RGBAFormat && mb.texture.minFilter === THREE.LinearFilter);
  mb.tileSize = 32;
  check('tileSize 32: tiles 95 x 62, uTile follows', mb._tile.width === 95 && mb._tile.height === 62 && mb._tileX.width === 95 && mb._tileMat.uniforms.uTile.value === 32
    && mb._reconstructMat.uniforms.uTile.value === 32);
  mb.tileSize = 20;
  check('output stays at the drawing-buffer size whatever the tile size', mb._out.width === 3024 && mb._out.height === 1964 && !('uStep' in mb._reconstructMat.uniforms));
  mb.setSize(1, 1);
  check('setSize 1 x 1: every target 1 x 1', [mb._vel, mb._tileX, mb._tile, mb._neighbor, mb._out].every((rt) => rt.width === 1 && rt.height === 1));
  mb.setSize(0, 0);
  check('setSize 0 x 0: clamped to 1 x 1', [mb._vel, mb._tile, mb._out].every((rt) => rt.width === 1 && rt.height === 1));
  mb.setSize(1001, 7);
  check('setSize 1001 x 7: tiles 51 x 1', mb._tile.width === 51 && mb._tile.height === 1 && mb._tileX.height === 7);
  mb.setSize(1600, 900);

  // shader contracts
  const vf = mb._velocityMat.fragmentShader, rf = mb._reconstructMat.fragmentShader;
  check('velocity shader: flags bit0 without noHistory -> G-buffer velocity, else uReproj', vf.includes('& ( ENGINE_GB_WROTE | ENGINE_GB_NO_HISTORY ) ) == ENGINE_GB_WROTE ) v = texelFetch( tVelocity')
    && vf.includes('uReproj * vec4( uv * 2.0 - 1.0, d * 2.0 - 1.0, 1.0 )') && vf.includes('pc.w > 0.0 ? uv - ( pc.xy / pc.w * 0.5 + 0.5 ) : vec2( 0.0 )'));
  check('velocity shader: MRT location 1 = (|V|, linear depth), half extent clamp', vf.includes('layout( location = 1 ) out highp vec4 mbVZ;') && vf.includes('v *= uRes * uBlur.x;') && vf.includes('if ( l > uBlur.y )'));
  check('reconstruction: early-out copies the centre texel', rf.includes('if ( dot( vn, vn ) < 0.0625 ) { gl_FragColor = cx; return; }'));
  check('reconstruction: paper weights (cone x2, cylinder x2, soft depth)', rf.includes('front( zy, zx ) * cone( r, vy ) + front( zx, zy ) * cone( r, vx ) + cylinder( r, vy ) * cylinder( r, vx ) * 2.0'));
  check('reconstruction: r = texel-snapped |Y - X| (the paper), output at the pixel itself', rf.includes('r = length( vec2( iy - ix ) )') && rf.includes('vec2 X = gl_FragCoord.xy;'));
  check('reconstruction: MB_SAMPLES define = samples', mb._reconstructMat.defines.MB_SAMPLES === 8 && rf.includes('i < MB_SAMPLES'));
  check('all motion blur shaders balanced', [mb._velocityMat, mb._tileMat, mb._neighborMat, mb._reconstructMat].every((m) => balanced(m.fragmentShader) && balanced(m.vertexShader)));
  check('velocity uniforms: CameraHistory uReproj / uJitter objects shared', mb._velocityMat.uniforms.uReproj === hist.uniforms.uReproj && mb._velocityMat.uniforms.uJitter === hist.uniforms.uJitter);
  mb.samples = 12;
  check('samples 12: define updated, material recompiles', mb._reconstructMat.defines.MB_SAMPLES === 12 && mb._reconstructMat.version > 0);
  mb.samples = 8;

  // render on the stub: 5 draws in order, uniforms, defines
  const r = stubRenderer();
  r.log = [];
  mb.strength = 2; mb.maxBlur = 60;
  mb.render(r, null, readBuffer);
  const order = r.log.map((e) => e.target);
  check('render: 5 draws: velocity, tile rows, tile max, neighbour max, output', r.log.length === 5 && order[0] === mb._vel && order[1] === mb._tileX && order[2] === mb._tile
    && order[3] === mb._neighbor && order[4] === mb._out && r.log[4].material === mb._reconstructMat);
  check('render: every target colour-cleared before its draw, autoClear off during the pass', clearedDraws(r.log) && r.clears === 5);
  check('render: autoClear restored (true)', r.autoClear === true);
  r.autoClear = false;
  mb.render(r, null, readBuffer);
  check('render: autoClear restored (false)', r.autoClear === false);
  r.autoClear = true;
  const vu = mb._velocityMat.uniforms;
  check('render: G-buffer velocity / normal bound, scene depth, no MB_CAMERA_ONLY', vu.tVelocity.value === gbTex.velocity && vu.tNormal.value === gbTex.normal
    && vu.tDepth.value === readBuffer.depthTexture && !('MB_CAMERA_ONLY' in mb._velocityMat.defines));
  const blur = vu.uBlur.value;
  check('render: uBlur = (strength / 2, min(maxBlur / 2, tileSize)) = (1, 20)', blur.x === 1 && blur.y === 20);
  mb.maxBlur = 24;
  mb.render(r, null, readBuffer);
  check('render: maxBlur 24 -> uBlur.y 12; near / far from the camera', blur.y === 12 && vu.uNear.value === 0.1 && vu.uFar.value === 6000);
  // shutter: k = strength * clamp(shutter / dt, 0, 4); shutter 0 or dt <= 0 / missing: frame-relative k = strength
  mb.render(r, null, readBuffer, 1 / 120);
  check('shutter 0: frame-relative whatever dt (dt 1/120 -> uBlur.x = strength / 2)', blur.x === 1);
  mb.shutter = 1 / 60;
  mb.render(r, null, readBuffer, 1 / 60);
  const x60 = blur.x;
  mb.render(r, null, readBuffer, 1 / 120);
  const x120 = blur.x;
  check('shutter 1/60: dt 1/60 -> 1 (k = strength), dt 1/120 -> 2 (same streak in px per second)', Math.abs(x60 - 1) < 1e-12 && Math.abs(x120 - 2) < 1e-12);
  mb.render(r, null, readBuffer, 1 / 30);
  check('shutter 1/60, dt 1/30 -> 0.5 (half the inter-frame motion)', blur.x, 0.5 - 1e-12, 0.5 + 1e-12);
  mb.render(r, null, readBuffer, 1 / 1000);
  check('shutter / dt clamped to 4 (dt 1 ms -> 4 x strength / 2)', blur.x, 4 - 1e-12, 4 + 1e-12);
  mb.render(r, null, readBuffer, 0);
  const x0 = blur.x;
  mb.render(r, null, readBuffer);
  check('shutter > 0 with dt 0 / undefined: frame-relative (bench renders)', x0 === 1 && blur.x === 1);
  check('render: reconstruction reads the colour, the (|V|, z) attachment and the neighbour max', mb._reconstructMat.uniforms.tColor.value === readBuffer.texture
    && mb._reconstructMat.uniforms.tVZ.value === mb._vel.textures[1] && mb._reconstructMat.uniforms.tNeighbor.value === mb._neighbor.texture);
  mb.gbuffer = null;
  mb.render(r, null, readBuffer);
  check('gbuffer null: MB_CAMERA_ONLY (depth reprojection everywhere)', 'MB_CAMERA_ONLY' in mb._velocityMat.defines && vu.tVelocity.value === null);
  mb.gbuffer = gbTex;
  mb.render(r, null, readBuffer);
  check('gbuffer back: MB_CAMERA_ONLY removed', !('MB_CAMERA_ONLY' in mb._velocityMat.defines));
  r.log = null;
  // allocation with the shutter path live (a constant dt: a dt computed in the loop would be boxed by this caller)
  for (let i = 0; i < 50000; i++) mb.render(r, null, readBuffer, 0.0125); // warm (optimised code)
  const a = measureAlloc(() => { for (let i = 0; i < 50000; i++) mb.render(r, null, readBuffer, 0.0125); });
  check('motion blur render() x50000: heap growth', a.bytes, -Infinity, 64 * 1024, 'B');
  check('motion blur render() x50000: GCs', a.gcs, 0, 0);
  check('motion blur dispose(): no throw', !throws(() => mb.dispose()));
}

// ================================================================================================ DoFPass
{
  const cam = new THREE.PerspectiveCamera(60, 16 / 9, 0.1, 6000);
  check('DOF_DEFAULTS: 12 m, 50 mm, f/2, 24 mm, maxCoC 12, 0.5, 24, nearBlur, !stableDepth, frozen',
    DOF_DEFAULTS.focusDistance === 12 && DOF_DEFAULTS.focalLength === 50 && DOF_DEFAULTS.fStop === 2 && DOF_DEFAULTS.sensorHeight === 24 && DOF_DEFAULTS.maxCoC === 12
    && DOF_DEFAULTS.scale === 0.5 && DOF_DEFAULTS.samples === 24 && DOF_DEFAULTS.nearBlur === true && DOF_DEFAULTS.stableDepth === false && Object.isFrozen(DOF_DEFAULTS));
  const dof = new DoFPass(cam);
  check('DoF: defaults on the instance', dof.focusDistance === 12 && dof.focalLength === 50 && dof.fStop === 2 && dof.sensorHeight === 24 && dof.maxCoC === 12
    && dof.scale === 0.5 && dof.samples === 24 && dof.nearBlur === true && dof.stableDepth === false);
  check('DoF: Pass with needsSwap false, enabled', dof.needsSwap === false && dof.enabled === true && !throws(() => dof.reset()));
  check('DoF: unknown option throws', throws(() => new DoFPass(cam, { fstop: 2 }), /unknown option 'fstop'/));
  check('DoF: focusDistance 0 / -3 / NaN throw', throws(() => new DoFPass(cam, { focusDistance: 0 }), /focusDistance/) && throws(() => { dof.focusDistance = -3; }) && throws(() => { dof.focusDistance = NaN; }) && dof.focusDistance === 12);
  check('DoF: focus closer than the focal length throws (0.04 m at 50 mm)', throws(() => new DoFPass(cam, { focusDistance: 0.04 }), /exceed focalLength/) && throws(() => { dof.focalLength = 12001; }, /exceed/) && dof.focalLength === 50);
  check('DoF: fStop / sensorHeight / focalLength <= 0 throw', throws(() => { dof.fStop = 0; }, /fStop/) && throws(() => { dof.sensorHeight = -24; }, /sensorHeight/) && throws(() => { dof.focalLength = 0; }, /focalLength/));
  check('DoF: maxCoC 0 / 65 throw', throws(() => { dof.maxCoC = 0; }, /maxCoC/) && throws(() => { dof.maxCoC = 65; }, /maxCoC/) && dof.maxCoC === 12);
  check('DoF: scale 0 / 2 throw, samples 3 / 65 / 4.5 throw', throws(() => { dof.scale = 0; }, /scale/) && throws(() => { dof.scale = 2; }) && throws(() => { dof.samples = 3; }, /samples/)
    && throws(() => { dof.samples = 65; }) && throws(() => { dof.samples = 4.5; }) && dof.scale === 0.5 && dof.samples === 24);
  check("DoF: nearBlur / stableDepth reject non-booleans ('false', 0, 1, 'true')", throws(() => { dof.nearBlur = 'false'; }, /nearBlur: expected a boolean/)
    && throws(() => { dof.nearBlur = 0; }) && throws(() => { dof.stableDepth = 1; }, /stableDepth/) && throws(() => new DoFPass(cam, { nearBlur: 'true' }), /nearBlur/)
    && throws(() => new DoFPass(cam, { stableDepth: 'yes' }), /stableDepth/) && dof.nearBlur === true && dof.stableDepth === false);
  const dof1 = new DoFPass(cam, { scale: 1, maxCoC: 24 });
  check('constructor applies the options before setSize (scale 1: uScale 1, uRatio (1, 1), uTile = tile)', dof1._uScale.value === 1
    && dof1._prefilterMat.uniforms.uRatio.value.x === 1 && dof1._prefilterMat.uniforms.uRatio.value.y === 1 && dof1._tile === 24
    && dof1._tileMat.uniforms.uTile.value === 24 && dof1._nearMat.uniforms.uTile.value === 24);
  dof1.dispose();

  dof.setSize(3024, 1964);
  check('setSize 3024x1964: fields 1512 x 982 RGBA16F', [dof._pre, dof._far, dof._near].every((rt) => rt.width === 1512 && rt.height === 982 && rt.texture.type === THREE.HalfFloatType));
  check('setSize: near tile = ceil(12 * 0.5) = 6 texels, tiles 252 x 164 RG16F', dof._tile === 6 && dof._nearTile.width === 252 && dof._nearTile.height === 164
    && dof._nearMax.width === 252 && dof._nearMax.height === 164 && dof._nearTile.texture.format === THREE.RGFormat);
  check('setSize: output 3024 x 1964 RGBA16F, uRatio (2, 2), uScale 0.5', dof._out.width === 3024 && dof._out.height === 1964 && dof.texture.type === THREE.HalfFloatType
    && dof._prefilterMat.uniforms.uRatio.value.x === 2 && dof._prefilterMat.uniforms.uRatio.value.y === 2 && dof._uScale.value === 0.5);
  dof.maxCoC = 24;
  check('maxCoC 24: tile 12 texels, tiles 126 x 82', dof._tile === 12 && dof._nearTile.width === 126 && dof._nearTile.height === 82 && dof._tileMat.uniforms.uTile.value === 12 && dof._nearMat.uniforms.uTile.value === 12);
  dof.maxCoC = 12;
  dof.scale = 1;
  check('scale 1: fields full size, tile 12', dof._pre.width === 3024 && dof._far.height === 1964 && dof._tile === 12 && dof._uScale.value === 1);
  dof.scale = 0.5;
  dof.setSize(1, 1);
  check('setSize 1 x 1: every target 1 x 1', [dof._pre, dof._nearTile, dof._nearMax, dof._far, dof._near, dof._out].every((rt) => rt.width === 1 && rt.height === 1));
  dof.setSize(0, 0);
  check('setSize 0 x 0: clamped to 1 x 1', dof._out.width === 1 && dof._pre.width === 1);
  dof.setSize(1440 * 16 / 9, 1440);
  check('cocAt(d) = circleOfConfusion with the buffer height', near(dof.cocAt(5), circleOfConfusion(5, 12, 50, 2, 24, 1440), 1e-12));

  // shader contracts
  const cf = dof._compositeMat.fragmentShader, ff = dof._farMat.fragmentShader, nf = dof._nearMat.fragmentShader, pf = dof._prefilterMat.fragmentShader;
  check('composite: early-out copies the input texel (|coc| < 0.5, near alpha 0)', cf.includes('if ( abs( coc ) < 0.5 && nf.a <= 0.0 ) { gl_FragColor = c; return; }'));
  check('prefilter: near CoC kept once a tap is near-blurred (-min >= 0.5), else the largest far CoC', pf.includes('float coc = - lo >= 0.5 ? lo : hi;'));
  check('composite: far mix smoothstep(0.5, 2, coc), near premultiplied over', cf.includes('smoothstep( 0.5, 2.0, coc )') && cf.includes('col * ( 1.0 - nf.a ) + nf.rgb'));
  check('CoC GLSL: K (1 - D / z) clamped to +-maxCoC', pf.includes('clamp( uCoc.x * ( 1.0 - uCoc.y / z ), - uCoc.z, uCoc.z )') && cf.includes('float dofCoc( float d )'));
  check('far gather: scatter-as-gather weight on the tap\'s own CoC', ff.includes('clamp( t.a * uScale - length( o ) + 1.0, 0.0, 1.0 )') && ff.includes('uniform vec2 uKernel[ DOF_SAMPLES ];'));
  check('near gather: dilated radius, energy weight, premultiplied output', nf.includes('texelFetch( tNearMax, ip / uTile, 0 ).r') && nf.includes('k / ( n * n )') && nf.includes('gl_FragColor = vec4( sum / max( a, 1.0 ), min( a, 1.0 ) );'));
  check('all DoF shaders balanced', [dof._prefilterMat, dof._tileMat, dof._dilateMat, dof._farMat, dof._nearMat, dof._compositeMat].every((m) => balanced(m.fragmentShader) && balanced(m.vertexShader)));
  check('gathers share one Vogel kernel uniform of 2 x samples floats', dof._farMat.uniforms.uKernel === dof._nearMat.uniforms.uKernel && dof._uKernel.value.length === 48
    && dof._farMat.defines.DOF_SAMPLES === 24 && dof._nearMat.defines.DOF_SAMPLES === 24);
  dof.samples = 32;
  check('samples 32: both gathers recompile with a 64-float kernel', dof._farMat.defines.DOF_SAMPLES === 32 && dof._nearMat.defines.DOF_SAMPLES === 32 && dof._uKernel.value.length === 64
    && dof._farMat.uniforms.uKernel.value === dof._uKernel.value);
  dof.samples = 24;

  // render on the stub
  const r = stubRenderer();
  r.log = [];
  dof.render(r, null, readBuffer);
  const order = r.log.map((e) => e.target);
  check('render (nearBlur): 6 draws: prefilter, near tile, dilate, near, far, composite', r.log.length === 6 && order[0] === dof._pre && order[1] === dof._nearTile
    && order[2] === dof._nearMax && order[3] === dof._near && order[4] === dof._far && order[5] === dof._out && 'DOF_NEAR' in dof._compositeMat.defines);
  check('render: every target colour-cleared before its draw, autoClear off during the pass', clearedDraws(r.log) && r.clears === 6);
  r.autoClear = false;
  dof.render(r, null, readBuffer);
  const acFalse = r.autoClear;
  r.autoClear = true;
  dof.render(r, null, readBuffer);
  check('render: autoClear restored (false and true)', acFalse === false && r.autoClear === true);
  const K = circleOfConfusion(Infinity, 12, 50, 2, 24, 1440);
  check('render: uCoc = (K px, focus m, maxCoC), camera near / far', near(dof._uCoc.value.x, K, 1e-9) && dof._uCoc.value.y === 12 && dof._uCoc.value.z === 12
    && dof._uNear.value === 0.1 && dof._uFar.value === 6000 && dof._prefilterMat.uniforms.uCoc === dof._compositeMat.uniforms.uCoc);
  check('render: prefilter + composite read the colour and the scene depth', dof._prefilterMat.uniforms.tColor.value === readBuffer.texture && dof._prefilterMat.uniforms.tDepth.value === readBuffer.depthTexture
    && dof._compositeMat.uniforms.tColor.value === readBuffer.texture && dof._compositeMat.uniforms.tDepth.value === readBuffer.depthTexture);
  dof.focusDistance = 30; dof.fStop = 8;
  dof.render(r, null, readBuffer);
  check('live lens: uCoc follows focusDistance / fStop', near(dof._uCoc.value.x, circleOfConfusion(Infinity, 30, 50, 8, 24, 1440), 1e-9) && dof._uCoc.value.y === 30);
  dof.nearBlur = false;
  r.log = [];
  dof.render(r, null, readBuffer);
  check('nearBlur false: 3 draws (prefilter, far, composite), no DOF_NEAR', r.log.length === 3 && r.log[0].target === dof._pre && r.log[1].target === dof._far
    && r.log[2].target === dof._out && !('DOF_NEAR' in dof._compositeMat.defines) && clearedDraws(r.log));
  dof.nearBlur = true;
  dof.stableDepth = true;
  dof.render(r, null, readBuffer);
  check('stableDepth: DOF_STABLE_DEPTH defined, closest of 5 depths in the composite', 'DOF_STABLE_DEPTH' in dof._compositeMat.defines && 'DOF_NEAR' in dof._compositeMat.defines
    && /#ifdef DOF_STABLE_DEPTH[\s\S]*d = min\( d, texelFetch\( tDepth, clamp\( ip \+ ivec2\( 1, 1 \)/.test(dof._compositeMat.fragmentShader));
  dof.stableDepth = false;
  dof.render(r, null, readBuffer);
  check('stableDepth off: define removed', !('DOF_STABLE_DEPTH' in dof._compositeMat.defines));
  r.log = null;
  for (let i = 0; i < 50000; i++) dof.render(r, null, readBuffer); // warm (optimised code)
  const a = measureAlloc(() => { for (let i = 0; i < 50000; i++) dof.render(r, null, readBuffer); });
  check('DoF render() x50000: heap growth', a.bytes, -Infinity, 64 * 1024, 'B');
  check('DoF render() x50000: GCs', a.gcs, 0, 0);
  // a pass-through configuration the lead can gate on: 10 mm f/22 focused at 10 m, 1964 px
  let worst = 0;
  for (let d = 0.8; d < 6000; d *= 1.05) worst = Math.max(worst, Math.abs(circleOfConfusion(d, 10, 10, 22, 24, 1964)));
  check('gate lens 10 mm f/22 @ 10 m: |CoC| < 0.5 px from 0.8 m to 6 km at 1964 px', worst, 0, 0.5, 'px');
  check('DoF dispose(): no throw', !throws(() => dof.dispose()));
}

console.log(results.join('\n'));
console.log(failures ? `\n${failures} of ${total} check(s) FAILED` : `\nall ${total} checks passed`);
process.exit(failures ? 1 : 0);
