// Headless checks for the post pipeline's CPU side (no browser, no WebGL): CameraHistory (jitter sequence, matrices,
// exact restore, reset, no allocation), the TAA reconstruction weights, resolveAA (the anti-aliasing decision: the
// 0.6.0 rule with the G-buffer off, TAA / single-sampled with it on), and PostFX option plumbing on a stub renderer
// (construction only: nothing here draws). 0.8.0 compositor: the final / TAA shader texts pinned to 0.7.0 with the
// new define blocks stripped, default = no new define / pass / target, option validation, set() / get() paths that
// create and dispose passes, the colorSource chain, allocation-free _source / set / render, timings keys.
// Run: node test/postfx-options.mjs
import v8 from 'node:v8';
import crypto from 'node:crypto';
import * as THREE from 'three';
import { CameraHistory, jitterPattern } from '../src/render/CameraHistory.js';
import { GBUFFER_UNIFORMS } from '../src/render/GBuffer.js';
import { PostFX, resolveAA, AA_MODES, FINAL_SHADER } from '../src/render/PostFX.js';
import { BloomPass } from '../src/render/passes/BloomPass.js';
import { taaFilterWeights, TAA_REJECTS, TAA_SHADER } from '../src/render/passes/TAAPass.js';
import { SSRPass, SSR_COMPOSITE_GLSL, SSR_DEFAULTS } from '../src/render/passes/SSRPass.js';
import { VolumetricPass } from '../src/render/passes/VolumetricPass.js';
import { MotionBlurPass } from '../src/render/passes/MotionBlurPass.js';
import { DoFPass, DOF_DEFAULTS } from '../src/render/passes/DoFPass.js';
import { LUT_GLSL, identityLut } from '../src/render/Lut.js';

let failures = 0;
const results = [];
function check(name, value, lo = 1, hi = lo, unit = '') {
  if (typeof value === 'boolean') value = value ? 1 : 0;
  const ok = Number.isFinite(value) && value >= lo && value <= hi;
  if (!ok) failures++;
  results.push(`${ok ? 'PASS' : 'FAIL'}  ${name.padEnd(60)} ${fmt(value)} ${unit}   [${fmt(lo).trim()}${hi !== lo ? ' .. ' + fmt(hi).trim() : ''}]`);
  return ok;
}
const fmt = (x) => (!Number.isFinite(x) ? String(x) : Number.isInteger(x) ? String(x) : Math.abs(x) < 1e-3 ? x.toExponential(2) : x.toFixed(4)).padStart(10);
const throws = (fn, re) => { try { fn(); return false; } catch (e) { return re ? re.test(e.message) : true; } };
const sameBits = (a, b) => a.elements.every((v, i) => Object.is(v, b.elements[i]));
const maxDiff = (a, b) => a.elements.reduce((m, v, i) => Math.max(m, Math.abs(v - b.elements[i])), 0);
const maxDiffIdentity = (m) => maxDiff(m, new THREE.Matrix4());

performance.now(); process.memoryUsage();
function measureAlloc(fn) {
  const prof = new v8.GCProfiler();
  prof.start();
  const h0 = process.memoryUsage().heapUsed;
  fn();
  const h1 = process.memoryUsage().heapUsed;
  return { bytes: h1 - h0, gcs: prof.stop().statistics.length };
}

// pixel position (drawing-buffer px, y up) of a world point through a projection * view
const _v = new THREE.Vector4();
function pixel(proj, cam, p, W, H) {
  _v.set(p.x, p.y, p.z, 1).applyMatrix4(cam.matrixWorldInverse).applyMatrix4(proj);
  return [(_v.x / _v.w * 0.5 + 0.5) * W, (_v.y / _v.w * 0.5 + 0.5) * H];
}

// ================================================================================================= jitter pattern
{
  const h16 = jitterPattern('halton16');
  check('halton16: 16 pairs', h16.length, 32);
  check('halton16[0] = (0, -1/6)', h16[0] === 0 && Math.abs(h16[1] + 1 / 6) < 1e-12);
  check('halton16[1] = (-0.25, 1/6)', h16[2] === -0.25 && Math.abs(h16[3] - 1 / 6) < 1e-12);
  check('halton16[3] = (-0.375, -1/18)', h16[6] === -0.375 && Math.abs(h16[7] + 1 / 18) < 1e-12);
  let inside = true, mx = 0, my = 0;
  for (let i = 0; i < 16; i++) { inside &&= Math.abs(h16[2 * i]) < 0.5 && Math.abs(h16[2 * i + 1]) < 0.5; mx += h16[2 * i] / 16; my += h16[2 * i + 1] / 16; }
  check('halton16: offsets inside (-0.5, 0.5)', inside);
  check('halton16: |mean x|', Math.abs(mx), 0, 0.04, 'px');
  check('halton16: |mean y|', Math.abs(my), 0, 0.04, 'px');
  check('halton8: 8 pairs, prefix of halton16', jitterPattern('halton8').length === 16 && jitterPattern('halton8').every((v, i) => v === h16[i]));
  check('custom pattern [[x, y]]', jitterPattern([[0.25, -0.25], [-0.25, 0.25]]).join() === '0.25,-0.25,-0.25,0.25');
  check('unknown pattern throws', throws(() => jitterPattern('sobol'), /unknown jitter pattern/));
}

// ================================================================================================= CameraHistory
{
  const W = 1600, H = 900;
  const cam = new THREE.PerspectiveCamera(60, W / H, 0.1, 6000);
  cam.position.set(12, 3, 40); cam.lookAt(0, 1, 0); cam.updateMatrixWorld();
  const P0 = cam.projectionMatrix.clone(), Pi0 = cam.projectionMatrixInverse.clone();
  const hist = new CameraHistory(cam);
  const u = hist.uniforms;
  check('history: uniforms are reusable { value } objects', ['uViewProj', 'uPrevViewProj', 'uInvViewProj', 'uUnjitteredProj', 'uReproj', 'uJitter', 'uPrevJitter', 'uFrame'].every((k) => u[k] && 'value' in u[k]));

  hist.begin(W, H);
  const pat = jitterPattern('halton16');
  check('frame 1: fresh, uFrame 0, frame counter 1', hist.fresh && u.uFrame.value === 0 && hist.frame === 1);
  check('frame 1: uJitter = halton16[0] (pixels)', u.uJitter.value.x === pat[0] && u.uJitter.value.y === pat[1]);
  const e = cam.projectionMatrix.elements, e0 = P0.elements;
  check('jitter: elements[8] += 2 jx / width', e[8] === e0[8] + pat[0] * 2 / W);
  check('jitter: elements[9] += 2 jy / height', e[9] === e0[9] + pat[1] * 2 / H);
  check('jitter: every other element untouched', e.every((v, i) => i === 8 || i === 9 || Object.is(v, e0[i])));
  check('jitter: projectionMatrixInverse is the inverse', maxDiffIdentity(new THREE.Matrix4().multiplyMatrices(cam.projectionMatrix, cam.projectionMatrixInverse)), 0, 1e-12);
  check('engineUnjitteredProj = the camera projection, exact', sameBits(GBUFFER_UNIFORMS.engineUnjitteredProj.value, P0) && sameBits(u.uUnjitteredProj.value, P0));
  const vp1 = new THREE.Matrix4().multiplyMatrices(P0, cam.matrixWorldInverse);
  check('uViewProj = unjittered proj * view', maxDiff(u.uViewProj.value, vp1), 0, 1e-15);
  check('fresh: enginePrevViewProj = uPrevViewProj = this frame (no motion)', sameBits(GBUFFER_UNIFORMS.enginePrevViewProj.value, u.uViewProj.value) && sameBits(u.uPrevViewProj.value, u.uViewProj.value));
  check('uInvViewProj * uViewProj = I', maxDiffIdentity(new THREE.Matrix4().multiplyMatrices(u.uInvViewProj.value, u.uViewProj.value)), 0, 1e-12);
  check('fresh: uReproj = I', maxDiffIdentity(u.uReproj.value), 0, 1e-12);
  // the documented sign: content moves by -jitter pixels (TAA samples the current frame at p - uJitter)
  const X = new THREE.Vector3(3, 2, -5);
  const [ux, uy] = pixel(P0, cam, X, W, H), [jx, jy] = pixel(cam.projectionMatrix, cam, X, W, H);
  check('jittered image: point drawn at p - jitter (x)', Math.abs(jx - ux + pat[0]), 0, 1e-6, 'px');
  check('jittered image: point drawn at p - jitter (y)', Math.abs(jy - uy + pat[1]), 0, 1e-6, 'px');
  hist.end();
  check('end(): projection restored bit-exact', sameBits(cam.projectionMatrix, P0));
  check('end(): projection inverse restored bit-exact', sameBits(cam.projectionMatrixInverse, Pi0));
  check('end() twice is harmless', (hist.end(), sameBits(cam.projectionMatrix, P0)));

  // frame 2: the camera moved; prev = frame 1's unjittered view-projection
  const vpPrev = u.uViewProj.value.clone();
  cam.position.set(13, 3.2, 38); cam.lookAt(0, 1.2, 0); cam.updateMatrixWorld();
  hist.begin(W, H);
  check('frame 2: not fresh, uFrame 1', !hist.fresh && u.uFrame.value === 1);
  check('frame 2: uPrevViewProj = frame 1 uViewProj, exact', sameBits(u.uPrevViewProj.value, vpPrev) && sameBits(GBUFFER_UNIFORMS.enginePrevViewProj.value, vpPrev));
  check('frame 2: uPrevJitter = frame 1 jitter', u.uPrevJitter.value.x === pat[0] && u.uPrevJitter.value.y === pat[1]);
  check('frame 2: uJitter = halton16[1]', u.uJitter.value.x === pat[2] && u.uJitter.value.y === pat[3]);
  // uReproj maps a point's current unjittered NDC (+ depth) to its previous clip position
  _v.set(X.x, X.y, X.z, 1).applyMatrix4(u.uViewProj.value);
  const ndc = new THREE.Vector4(_v.x / _v.w, _v.y / _v.w, _v.z / _v.w, 1);
  const viaReproj = ndc.clone().applyMatrix4(u.uReproj.value), direct = new THREE.Vector4(X.x, X.y, X.z, 1).applyMatrix4(vpPrev);
  check('uReproj: current NDC -> previous NDC (x)', Math.abs(viaReproj.x / viaReproj.w - direct.x / direct.w), 0, 1e-9);
  check('uReproj: current NDC -> previous NDC (y)', Math.abs(viaReproj.y / viaReproj.w - direct.y / direct.w), 0, 1e-9);
  const ivw = u.uInvViewProj.value.elements, hw = ivw[3] * ndc.x + ivw[7] * ndc.y + ivw[11] * ndc.z + ivw[15];
  check('uReproj.w / invVP row 3 = previous view depth', Math.abs(viaReproj.w / hw - direct.w), 0, 1e-6, 'm');
  hist.end();

  // the sequence: 16 periodic, every begin()/end() restores
  let seqOk = true, restoreOk = true;
  const Pm = cam.projectionMatrix.clone();
  for (let f = 2; f < 40; f++) {
    hist.begin(W, H);
    const k = (f % 16) * 2;
    seqOk &&= u.uJitter.value.x === pat[k] && u.uJitter.value.y === pat[k + 1];
    hist.end();
    restoreOk &&= sameBits(cam.projectionMatrix, Pm);
  }
  check('jitter sequence: frame f uses halton16[f % 16]', seqOk);
  check('restore exact over 38 frames', restoreOk);
  hist.begin(W, H); hist.begin(W, H); hist.end();
  check('begin() twice: jitter not stacked', sameBits(cam.projectionMatrix, Pm));

  // reset: next frame fresh, no motion
  cam.position.x += 50; cam.updateMatrixWorld();
  hist.reset();
  const f0 = hist.frame;
  hist.begin(W, H);
  check('reset(): fresh, prev = current', hist.fresh && sameBits(u.uPrevViewProj.value, u.uViewProj.value) && maxDiffIdentity(u.uReproj.value) < 1e-12);
  check('reset(): jitter pattern restarts, uFrame keeps counting', u.uJitter.value.x === pat[0] && u.uJitter.value.y === pat[1] && u.uFrame.value === f0);
  hist.end();
  hist.begin(W, H);
  check('after reset: next frame has history', !hist.fresh);
  hist.end();

  // jitter off: matrices still published, camera untouched
  const plain = new CameraHistory(cam, { jitter: false });
  const Pa = cam.projectionMatrix.clone(), Pia = cam.projectionMatrixInverse.clone();
  plain.begin(W, H);
  check('jitter off: projection untouched during the frame', sameBits(cam.projectionMatrix, Pa) && sameBits(cam.projectionMatrixInverse, Pia));
  check('jitter off: uJitter (0, 0)', plain.uniforms.uJitter.value.x === 0 && plain.uniforms.uJitter.value.y === 0);
  check('jitter off: matrices published', sameBits(GBUFFER_UNIFORMS.engineUnjitteredProj.value, Pa) && maxDiff(plain.uniforms.uViewProj.value, new THREE.Matrix4().multiplyMatrices(Pa, cam.matrixWorldInverse)) < 1e-15);
  plain.end();
  check('jitter off: end() leaves it bit-identical', sameBits(cam.projectionMatrix, Pa) && sameBits(cam.projectionMatrixInverse, Pia));
  plain.jitter = true; plain.begin(W, H);
  check('jitter switched on live', !sameBits(cam.projectionMatrix, Pa) && plain.uniforms.uJitter.value.x === -0.25);
  plain.end();

  // orthographic: same content shift (-jitter pixels)
  const ortho = new THREE.OrthographicCamera(-40, 40, 22.5, -22.5, 0.1, 500);
  ortho.position.set(0, 50, 0); ortho.lookAt(0, 0, 0); ortho.updateMatrixWorld();
  const Po = ortho.projectionMatrix.clone(), oh = new CameraHistory(ortho, { pattern: [[0.3, -0.2]] });
  oh.begin(W, H);
  const [ox0, oy0] = pixel(Po, ortho, X, W, H), [ox1, oy1] = pixel(ortho.projectionMatrix, ortho, X, W, H);
  check('orthographic: point drawn at p - jitter', Math.abs(ox1 - ox0 + 0.3) < 1e-6 && Math.abs(oy1 - oy0 - 0.2) < 1e-6);
  oh.end();
  check('orthographic: restored exactly', sameBits(ortho.projectionMatrix, Po));

  // end() keeps a projection recomputed between begin() and end()
  hist.begin(W, H);
  cam.fov = 70; cam.updateProjectionMatrix();
  const P70 = cam.projectionMatrix.clone(), Pi70 = cam.projectionMatrixInverse.clone();
  hist.end();
  check('end(): projection recomputed meanwhile is kept', sameBits(cam.projectionMatrix, P70) && sameBits(cam.projectionMatrixInverse, Pi70) && !hist._applied);
  cam.fov = 60; cam.updateProjectionMatrix();
  check('end(): a (0, 0) offset leaves the camera untouched', (() => {
    const z = new CameraHistory(cam, { pattern: [[0, 0]] }), P = cam.projectionMatrix.clone();
    z.begin(W, H); const same = sameBits(cam.projectionMatrix, P) && !z._applied; z.end(); return same;
  })());
  // TAA's depth path: a point in front of this camera but behind the previous one reprojects with clip w < 0
  {
    const h2 = new CameraHistory(cam, { jitter: false });
    cam.position.set(0, 2, 0); cam.lookAt(0, 2, -10); cam.updateMatrixWorld(); h2.begin(W, H); h2.end();
    cam.lookAt(0, 2, 10); cam.updateMatrixWorld(); h2.begin(W, H); h2.end(); // turned around
    const Y = new THREE.Vector4(0, 2, 8, 1).applyMatrix4(h2.uniforms.uViewProj.value);
    const ndcY = new THREE.Vector4(Y.x / Y.w, Y.y / Y.w, Y.z / Y.w, 1).applyMatrix4(h2.uniforms.uReproj.value);
    check('point behind the previous camera: uReproj w <= 0 (TAA blend 0)', Y.w > 0 && ndcY.w <= 0);
  }

  // nothing allocates per frame
  for (let i = 0; i < 50000; i++) { hist.begin(W, H); hist.end(); } // warm (optimised code, no IC growth)
  const a = measureAlloc(() => { for (let i = 0; i < 200000; i++) { hist.begin(W, H); hist.end(); } });
  check('begin()/end() x200000: heap growth', a.bytes, -Infinity, 64 * 1024, 'B');
  check('begin()/end() x200000: GCs', a.gcs, 0, 0);
}

// ================================================================================================= TAA weights
{
  const w = taaFilterWeights(0, 0);
  check('weights: sum 1 (no jitter)', Math.abs(w.reduce((s, v) => s + v, 0) - 1), 0, 1e-6);
  check('weights: centre heaviest, symmetric (no jitter)', w[4] > w[1] && w[1] === w[3] && w[3] === w[5] && w[5] === w[7] && w[0] === w[8]);
  const half = taaFilterWeights(0.5, 0);
  check('weights: jitter +0.5 px x -> columns -1 and 0 equal', Math.abs(half[3] - half[4]) < 1e-7 && half[5] < half[3]);
  const out = new Float32Array(9);
  for (let i = 0; i < 50000; i++) taaFilterWeights(i % 7 * 0.1 - 0.3, 0.2, out); // warm (and a young generation emptied by it)
  const a = measureAlloc(() => { for (let i = 0; i < 50000; i++) taaFilterWeights(i % 7 * 0.1 - 0.3, 0.2, out); });
  check('taaFilterWeights into out: no GC', a.gcs, 0, 0);
}

// ================================================================================================= resolveAA
{
  const caps = { maxSamples: 4 };
  const r = (o, c = caps) => resolveAA(o, c);
  // the 0.6.0 _applyAA rule, verbatim, for the parity sweep
  const old = (msaa, max, dark, px, maxPx) => (msaa === 'auto' ? (dark > 0.35 || px > maxPx ? 0 : max) : Math.min(max, msaa | 0));
  let parity = true, n = 0;
  for (const msaa of ['auto', 0, 1, 2, 4, 8]) for (const max of [0, 4, 8]) for (const dark of [0, 0.35, 0.36, 1]) for (const px of [1e6, 4e6, 6e6]) for (const maxPx of [Infinity, 4e6]) {
    const o = r({ msaa, darkness: dark, pixels: px, msaaMaxPixels: maxPx }, { maxSamples: max });
    const want = old(msaa, max, dark, px, maxPx);
    parity &&= o.samples === want && o.fxaa === (want === 0) && !o.taa && o.mode === (want > 0 ? 'msaa' : 'fxaa');
    n++;
  }
  check(`'auto' without G-buffer = the 0.6.0 rule (${n} cases)`, parity);
  check('defaults: 4x MSAA by day', r({}).samples === 4 && r({}).mode === 'msaa' && !r({}).fxaa);
  check('defaults: FXAA at night', r({ darkness: 1 }).fxaa && r({ darkness: 1 }).samples === 0);
  check("'msaa': MSAA whatever darkness / size", r({ aa: 'msaa', darkness: 1, pixels: 9e6, msaaMaxPixels: 4e6 }).samples === 4);
  check("'msaa' with msaa 2", r({ aa: 'msaa', msaa: 2 }).samples === 2);
  check("'msaa' without MSAA support: FXAA, forced", r({ aa: 'msaa' }, { maxSamples: 0 }).fxaa && r({ aa: 'msaa' }, { maxSamples: 0 }).forced);
  check("'fxaa': 0 samples", r({ aa: 'fxaa' }).samples === 0 && r({ aa: 'fxaa' }).fxaa && r({ aa: 'fxaa' }).mode === 'fxaa');
  check("'taa' without G-buffer throws (names gbuffer)", throws(() => r({ aa: 'taa' }), /gbuffer/));
  check('unknown aa throws', throws(() => r({ aa: 'smaa' }), /unknown aa/));
  check('AA_MODES', AA_MODES.join() === 'auto,msaa,fxaa,taa');
  let gbZero = true;
  for (const aa of AA_MODES) for (const dark of [0, 1]) gbZero &&= r({ aa, gbuffer: true, darkness: dark }).samples === 0;
  check('G-buffer: always 0 samples', gbZero);
  check("G-buffer 'auto' = TAA, no FXAA", r({ gbuffer: true }).taa && !r({ gbuffer: true }).fxaa && r({ gbuffer: true }).mode === 'taa');
  check("G-buffer 'taa' = TAA", r({ aa: 'taa', gbuffer: true }).taa);
  check("G-buffer 'fxaa' = FXAA", r({ aa: 'fxaa', gbuffer: true }).fxaa && !r({ aa: 'fxaa', gbuffer: true }).taa);
  check("G-buffer 'msaa' = FXAA, forced", r({ aa: 'msaa', gbuffer: true }).fxaa && r({ aa: 'msaa', gbuffer: true }).forced);
  const out = {};
  check('writes into out', resolveAA({}, caps, out) === out && out.samples === 4);
}

// ================================================================================================= PostFX on a stub renderer
// enough of WebGLRenderer for construction and for render() to run every pass's CPU side (draws are no-ops)
const stub = (maxSamples = 4, caps = {}) => {
  let bound = null;
  return {
    getSize: (v) => v.set(800, 450), getPixelRatio: () => 1, capabilities: { maxSamples, ...caps }, extensions: { has: () => true },
    toneMappingExposure: 1, autoClear: true, info: { autoReset: true, reset() {} },
    clear() {}, render() {}, getRenderTarget: () => bound, setRenderTarget(t) { bound = t; },
    getClearColor: (c) => c.setRGB(0, 0, 0), getClearAlpha: () => 1, setClearColor() {}, getActiveCubeFace: () => 0, getActiveMipmapLevel: () => 0,
  };
};
{
  const scene = new THREE.Scene(), cam = new THREE.PerspectiveCamera(60, 16 / 9, 0.1, 1000);
  const fx = new PostFX(stub(), scene, cam, { atmosphere: null });
  const c = fx.composer, rt1 = c.renderTarget1, rt2 = c.renderTarget2;
  check('default: no G-buffer / history / TAA', fx.gbuffer === null && fx.history === null && fx.taaPass === null);
  check('default: aa auto, 4x MSAA targets, no FXAA', fx.aa === 'auto' && rt1.samples === 4 && rt2.samples === 4 && fx.uniforms.uFXAA.value === 0);
  check('default: pass order scene, exposure, ao, bloom, final', c.passes.join() === [fx.renderPass, fx.exposurePass, fx.aoPass, fx.bloomPass, fx.finalPass].join() && c.passes.length === 5);
  check('default: no pass swaps', c.passes.every((p) => !p.needsSwap));
  check('bloomPass is a BloomPass', fx.bloomPass instanceof BloomPass);
  check('colorSource / sceneDepth = the composer readBuffer', fx.colorSource === c.readBuffer.texture && fx.sceneDepth === c.readBuffer.depthTexture && fx.sceneDepth !== null);
  check('bloom source = readBuffer itself without TAA', fx._source(c.readBuffer) === c.readBuffer);
  fx.setTimeOfDay(1);
  check('night: FXAA, targets single-sampled', fx.uniforms.uFXAA.value === 1 && rt2.samples === 0);
  fx.setTimeOfDay(0);
  fx.setMSAA(0);
  check('setMSAA(0): FXAA', fx.uniforms.uFXAA.value === 1 && rt2.samples === 0);
  fx.setMSAA('auto');
  fx.setAA('msaa'); fx.setTimeOfDay(1);
  check("setAA('msaa'): MSAA at night", rt2.samples === 4 && fx.uniforms.uFXAA.value === 0);
  fx.setMSAA('auto');
  check("setMSAA selects aa 'auto'", fx.aa === 'auto' && rt2.samples === 0);
  fx.setTimeOfDay(0);
  check("setAA('taa') without G-buffer throws, state unchanged", throws(() => fx.setAA('taa'), /gbuffer/) && fx.aa === 'auto');
  check("constructor { aa: 'taa' } without G-buffer throws", throws(() => new PostFX(stub(), scene, cam, { aa: 'taa' }), /gbuffer/));
  fx.set('ao.enabled', false); fx.set('bloom.enabled', false);
  check("set('ao.enabled' / 'bloom.enabled')", !fx.aoPass.enabled && !fx.bloomPass.enabled);
  fx.set('taa.blend', 0.8); fx.set('taa.sharpen', 0.25); fx.set('taa.clamp', 'minmax');
  check("set('taa.*') without G-buffer: kept for later", fx.taaOptions.blend === 0.8 && fx.taaOptions.sharpen === 0.25 && fx.taaOptions.clamp === 'minmax');
  check("set('taa.clamp', bad) throws", throws(() => fx.set('taa.clamp', 'box'), /unknown value/));
  check('set(unknown) throws', throws(() => fx.set('dof.enabled', true), /unknown path/));
  check("set('taa.blend', NaN / '0.8') throws, value kept", throws(() => fx.set('taa.blend', NaN), /finite/) && throws(() => fx.set('taa.gamma', '0.8'), /finite/) && fx.taaOptions.blend === 0.8);
  check('o.taa: unknown key throws', throws(() => new PostFX(stub(), scene, cam, { taa: { blnd: 0.8 } }), /unknown taa option 'blnd'/));
  check('o.taa: bad value throws', throws(() => new PostFX(stub(), scene, cam, { taa: { sharpen: Infinity } }), /finite/) && throws(() => new PostFX(stub(), scene, cam, { taa: { clamp: 'box' } }), /unknown value/));
  check('o.taa: defaults (stableDepth off) + overrides', (() => { const t = new PostFX(stub(), scene, cam, { taa: { blend: 0.8 } }).taaOptions; return t.blend === 0.8 && t.stableDepth === false && t.historyFormat === 'rgba16f'; })());
  check('o.taa cost knobs: defaults reproduce the measured output', (() => { const t = new PostFX(stub(), scene, cam).taaOptions; return t.reject === 'quad' && t.velocityFirst === false && t.snap === false && TAA_REJECTS.join() === 'quad,early,single'; })());
  check("o.taa.reject: bad value throws", throws(() => new PostFX(stub(), scene, cam, { taa: { reject: 'none' } }), /unknown value/));
  check('G-buffer with reversedDepthBuffer throws', throws(() => new PostFX(stub(4, { reversedDepthBuffer: true }), scene, cam, { gbuffer: true }), /standard depth buffer/));
  fx.set('jitter', false);
  check("set('jitter')", fx.jitter === false);
  check('timings: zeroed keys', ['scene', 'exposure', 'ao', 'taa', 'bloom', 'final', 'flare', 'total'].every((k) => fx.timings[k] === 0));

  // no atmosphere: fog from fogColor / fogParams
  const nf = new PostFX(stub(), scene, cam, { atmosphere: null, fogColor: 0x8899aa, fogParams: { haze: 0.001 } });
  cam.updateMatrixWorld();
  nf._syncFrame(0);
  check('atmosphere null: fogColor kept (linear Color)', nf.fogColor instanceof THREE.Color && nf.fogColor.getHex() === 0x8899aa);
  check('atmosphere null: fogParams -> uFogA', nf.uniforms.uFogA.value.x === 0.001 && nf.uniforms.uFogB.value.x === 0.0001);
  check('atmosphere null: setTimeOfDay(darkness) grades', (nf.setTimeOfDay(0.8), nf.grade.el !== undefined && nf.uniforms.uFXAA.value === 1));

  // G-buffer mode: GBuffer builds its MRT target and classifies materials without a GL context
  let gfx = null, why = '';
  try { gfx = new PostFX(stub(), scene, cam, { atmosphere: null, gbuffer: true }); } catch (e) { why = e.message; }
  check(`gbuffer: constructs on the stub renderer${why ? ` (${why})` : ''}`, gfx !== null);
  if (gfx) {
    const gc = gfx.composer;
    check('gbuffer: history + TAA + GBuffer', gfx.history instanceof CameraHistory && gfx.taaPass !== null && gfx.gbuffer !== null);
    check('gbuffer: readBuffer is the GBuffer target itself', gc.readBuffer === gfx.gbuffer.target && gc.renderTarget2 === gfx.gbuffer.target);
    check('gbuffer: pass order scene, exposure, ao, taa, bloom, final', gc.passes.join() === [gfx.renderPass, gfx.exposurePass, gfx.aoPass, gfx.taaPass, gfx.bloomPass, gfx.finalPass].join());
    check("gbuffer: aa 'auto' = TAA, jitter on, FXAA off", gfx.taaPass.enabled && gfx.history.jitter && gfx.uniforms.uFXAA.value === 0);
    check('gbuffer: no pass swaps', gc.passes.every((p) => !p.needsSwap));
    check('gbuffer: colorSource = TAA texture, bloom reads it', gfx.colorSource === gfx.taaPass.texture && gfx._source(gc.readBuffer).texture === gfx.taaPass.texture);
    check('gbuffer: sceneDepth = G-buffer depth', gfx.sceneDepth === gc.readBuffer.depthTexture);
    gfx.set('aa', 'fxaa');
    check("gbuffer 'fxaa': TAA off, no jitter, FXAA", !gfx.taaPass.enabled && !gfx.history.jitter && gfx.uniforms.uFXAA.value === 1 && gfx.colorSource === gc.readBuffer.texture);
    gfx.set('aa', 'taa'); gfx.set('jitter', false);
    check("set('jitter', false) under TAA", gfx.taaPass.enabled && !gfx.history.jitter);
    gfx.set('taa.blend', 0.85); gfx.set('taa.historyFormat', 'r11g11b10');
    check("set('taa.*') reaches the live pass", gfx.taaPass.blend === 0.85 && gfx.taaPass.historyFormat === 'r11g11b10');
    gfx.setGBuffer(false);
    check('setGBuffer(false): forward targets back, TAA gone, aa auto', gfx.gbuffer === null && gfx.taaPass === null && gfx.history === null && gfx.aa === 'auto' && gc.passes.length === 5 && gc.readBuffer.samples === 4 && gc.readBuffer.depthTexture !== null);
    gfx.setGBuffer(true);
    check('setGBuffer(true) again: TAA settings kept', gfx.taaPass.blend === 0.85 && gfx.taaPass.historyFormat === 'r11g11b10' && gc.readBuffer === gfx.gbuffer.target);
    gfx.set('taa.historyFormat', 'rgba16f'); gfx.set('jitter', true);

    // shader guards (TAA): depth path with the surface behind the previous camera -> blend 0; NaN-safe fresh path
    const fsrc = gfx.taaPass.material.fragmentShader;
    check('TAA shader: pc.w <= 0 on the depth path -> no history', /bool behind = ! wrote && pc\.w <= 0\.0;/.test(fsrc) && /if \( behind \|\|/.test(fsrc) && /behind \? vec2\( 0\.0 \)/.test(fsrc));
    check('TAA shader: fresh path NaN / Inf-safe', /uFresh > 0\.5 \) \{ gl_FragColor = vec4\( finite\( texelFetch\( tColor/.test(fsrc));

    // cost knobs: defaults add no define (same program as before them); each knob maps to its define, live
    const tdefs = gfx.taaPass.material.defines;
    const knobDefs = ['TAA_VELOCITY_FIRST', 'TAA_REJECT_EARLY', 'TAA_REJECT_SINGLE', 'TAA_SNAP'];
    check('cost knobs off by default: only the 0.7.0 defines', knobDefs.every((k) => !(k in tdefs)) && 'TAA_CATMULL_ROM' in tdefs && 'TAA_DEPTH_REJECT' in tdefs);
    gfx.set('taa.velocityFirst', true); gfx.set('taa.reject', 'early'); gfx.set('taa.snap', true);
    check("set('taa.velocityFirst' / 'reject' / 'snap') -> live pass + defines", gfx.taaPass.velocityFirst && gfx.taaPass.reject === 'early' && gfx.taaPass.snap
      && 'TAA_VELOCITY_FIRST' in tdefs && 'TAA_REJECT_EARLY' in tdefs && 'TAA_SNAP' in tdefs && !('TAA_REJECT_SINGLE' in tdefs));
    gfx.set('taa.reject', 'single');
    check("reject 'single': its define only", 'TAA_REJECT_SINGLE' in tdefs && !('TAA_REJECT_EARLY' in tdefs));
    check("set('taa.reject', bad) throws, value kept", throws(() => gfx.set('taa.reject', 'all'), /unknown value/) && gfx.taaPass.reject === 'single');
    check('velocity-first shader: flags read only for a zero velocity', /int flags = vel\.x != 0\.0 \|\| vel\.y != 0\.0 \? ENGINE_GB_WROTE : engineGbFlags\( texelFetch\( tNormal, bp, 0 \)\.a \);/.test(fsrc));
    check('snap shader: one texelFetch at a texel-centred history position', /if \( all\( lessThan\( abs\( sp - sn \), vec2\( 1\.0 \/ 512\.0 \) \) \) \) return texelFetch\( tHistory/.test(fsrc));
    gfx.set('taa.velocityFirst', false); gfx.set('taa.reject', 'quad'); gfx.set('taa.snap', false);
    check('knobs back off: defines back to the defaults', knobDefs.every((k) => !(k in tdefs)));
    check('TAAPass knob validation', throws(() => { gfx.taaPass.reject = 'x'; }, /unknown reject/));

    // tracked objects survive G-buffer rebuilds
    const box = new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshStandardMaterial());
    const late = new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshStandardMaterial());
    scene.add(box, late);
    const pool = { capturePrev() {} };
    check('track(): tracked now', gfx.track(box) === 1 && gfx.gbuffer.stats.tracked === 1);
    gfx.trackPool(pool);
    gfx.setGBuffer(false);
    check('track() while the G-buffer is off: kept, 0 now', gfx.track(late) === 0 && gfx._tracked.length === 2);
    gfx.setGBuffer(true);
    check('rebuild replays track() / trackPool()', gfx.gbuffer.stats.tracked === 2 && gfx.gbuffer.stats.pools === 1);
    gfx.untrack(late); gfx.untrackPool(pool); gfx.setGBuffer(false); gfx.setGBuffer(true);
    check('untrack() / untrackPool() drop them from the replay', gfx.gbuffer.stats.tracked === 1 && gfx.gbuffer.stats.pools === 0);

    // render() on the stub: jitter applied and removed, DEPTH_CLOSEST only while TAA jitters with stableDepth
    gfx.history.reset();
    const Pr = cam.projectionMatrix.clone();
    gfx.render(0);
    const defs = gfx.finalPass.material.defines;
    check('render(): camera projection restored, frame counted', sameBits(cam.projectionMatrix, Pr) && gfx.history.frame === 1);
    { // nothing copies the TAA result: Final and Bloom sample the history target TAA just wrote; the spare is never bound
      const r = gfx.renderer, bound = new Set(), srt = r.setRenderTarget;
      r.setRenderTarget = (t) => { bound.add(t); srt(t); };
      gfx.render(0);
      r.setRenderTarget = srt;
      const tex = gfx.taaPass.texture;
      check('no copy of the TAA output: Final / Bloom read the history texture', gfx.uniforms.tColor.value === tex && gfx.bloomPass.materialHighPassFilter.uniforms.tDiffuse.value === tex
        && !bound.has(gfx._spare) && bound.has(gfx.taaPass._rt[gfx.taaPass._cur]));
    }
    check('stableDepth off (default): no DEPTH_CLOSEST', !('DEPTH_CLOSEST' in defs));
    gfx.set('taa.stableDepth', true); gfx.render(0);
    check('stableDepth on + TAA jitter: DEPTH_CLOSEST', 'DEPTH_CLOSEST' in defs);
    gfx.set('jitter', false); gfx.render(0);
    check('stableDepth on, jitter off: no DEPTH_CLOSEST', !('DEPTH_CLOSEST' in defs));
    gfx.set('jitter', true); gfx.set('aa', 'fxaa'); gfx.render(0);
    check('stableDepth on, FXAA: no DEPTH_CLOSEST', !('DEPTH_CLOSEST' in defs));
    gfx.set('aa', 'taa'); gfx.set('taa.stableDepth', false);

    // timed render with dt undefined takes the composer clock's delta, like EffectComposer.render
    let deltas = 0;
    const clock = gc.clock, getDelta = clock.getDelta;
    clock.getDelta = function () { deltas++; return getDelta.call(this); };
    gfx.set('timing', true); gfx.render();
    check('timing + render(): composer clock delta used once, timings filled', deltas === 1 && Number.isFinite(gfx.timings.total) && gfx.timings.taa >= 0);
    clock.getDelta = getDelta; gfx.set('timing', false);

    // resize: G-buffer target, TAA history, readBuffer identity, history reset
    gfx.render(0);
    gfx.setSize(1024, 512, 1);
    const t = gfx.gbuffer.target, h0 = gfx.taaPass._rt[0], h1 = gfx.taaPass._rt[1];
    check('setSize: G-buffer target 1024x512', t.width === 1024 && t.height === 512 && gfx.gbuffer.velocity.image.width === 1024);
    check('setSize: TAA history targets 1024x512', h0.width === 1024 && h0.height === 512 && h1.width === 1024 && h1.height === 512);
    check('setSize: readBuffer is still the G-buffer target', gc.readBuffer === t && gfx.sceneDepth === t.depthTexture);
    check('setSize: TAA history reset', gfx.taaPass._reset === true);
    check('dispose (G-buffer): no throw', !throws(() => gfx.dispose()));
  }
  check('dispose (forward): no throw', !throws(() => fx.dispose()) && !throws(() => nf.dispose()));
}

// ================================================================================================= 0.8.0 compositor
// sha1 of the 0.7.0 shader texts, recorded before the 0.8.0 hooks were added. Each new hook lives in an `#ifdef FINAL_*`
// / `#ifdef TAA_SSR` ... `#endif` block of whole lines: with those blocks stripped the text must be 0.7.0's, byte for
// byte (three's program cache key holds the defines, not the text, so absent defines = the 0.7.0 program).
const SHA1_FINAL_FRAG_070 = '816b2e9e1d645383b36ca3e14b37a619771ca8f8';
const SHA1_FINAL_VERT_070 = '1a054fa8fff601268e0ad95145745883b20b84c8';
const SHA1_TAA_FRAG_070 = '4286168f6dd087274889ed1775d1f7b3b0119bca';
const NEW_FINAL_DEFINES = ['FINAL_SSR', 'FINAL_VOLUMETRIC', 'FINAL_LUT'];
const sha1 = (t) => crypto.createHash('sha1').update(t).digest('hex');
// drop every block opened by a line matching `open` (#ifdef NAME) through its matching #endif (nesting-aware)
function stripBlocks(src, open) {
  const out = [];
  let depth = 0;
  for (const line of src.split('\n')) {
    const t = line.trim();
    if (depth > 0) {
      if (/^#\s*if(def|ndef)?\b/.test(t)) depth++;
      else if (/^#\s*endif\b/.test(t)) depth--;
      continue;
    }
    if (open.test(t)) { depth = 1; continue; }
    out.push(line);
  }
  return out.join('\n');
}
const FINAL_OPEN = /^#ifdef FINAL_[A-Z_]+$/, TAA_OPEN = /^#ifdef TAA_SSR$/;
// a stand-in Atmosphere: what PostFX and VolumetricPass read (one packed cascade, so the volume pass marches)
const casMap = new THREE.Texture(), casMat = new THREE.Matrix4(), casLight = { shadow: { bias: -0.0005 } };
const atmoStub = () => ({
  darkness: 0, sunEl: 30 * Math.PI / 180, version: 0, skyCube: null,
  sunDir: new THREE.Vector3(0.3, 0.8, 0.2).normalize(), sunColor: new THREE.Color(1, 0.9, 0.8), sun: { intensity: 3 },
  fogParams: { haze: 0.0002, hazeK: 1 / 700, hazeBase: 0, mist: 0.0001, mistK: 1 / 40, sunScatter: 0.05 },
  setFogMode() {}, setFlareScale() {}, renderFlare() {},
  shadowCascadeInfo(out) { out.count = 1; out.packed = true; out.maps[0] = casMap; out.matrices[0] = casMat; out.lights[0] = casLight; return out; },
});
// spy on a pass instance's dispose (counts calls, still disposes)
const spyDispose = (p) => { const box = { n: 0 }, d = p.dispose; p.dispose = function () { box.n++; return d.call(this); }; return box; };
const keysOf = (fx) => fx.composer.passes.map((p) => fx._passKeys.get(p)).join();
// bytes allocated by fn: heap growth plus what the GCs during it freed
function allocated(fn) {
  const prof = new v8.GCProfiler();
  prof.start();
  const h0 = process.memoryUsage().heapUsed;
  fn();
  const h1 = process.memoryUsage().heapUsed;
  let freed = 0;
  for (const g of prof.stop().statistics) freed += g.beforeGC.heapStatistics.usedHeapSize - g.afterGC.heapStatistics.usedHeapSize;
  return h1 - h0 + freed;
}
{
  // ---- shader texts
  const fFrag = FINAL_SHADER.fragmentShader, tFrag = TAA_SHADER.fragmentShader;
  check('final fragment shader, FINAL_* blocks stripped = 0.7.0 (sha1)', sha1(stripBlocks(fFrag, FINAL_OPEN)) === SHA1_FINAL_FRAG_070);
  check('final vertex shader = 0.7.0 (sha1)', sha1(FINAL_SHADER.vertexShader) === SHA1_FINAL_VERT_070);
  check('TAA fragment shader, TAA_SSR blocks stripped = 0.7.0 (sha1)', sha1(stripBlocks(tFrag, TAA_OPEN)) === SHA1_TAA_FRAG_070);
  check('final shader: one block per hook (SSR 2, volumetric 4, LUT 2)', (fFrag.match(/#ifdef FINAL_SSR\n/g) || []).length === 2
    && (fFrag.match(/#ifdef FINAL_VOLUMETRIC\n/g) || []).length === 4 && (fFrag.match(/#ifdef FINAL_LUT\n/g) || []).length === 2);
  check('TAA shader: 3 TAA_SSR blocks', (tFrag.match(/#ifdef TAA_SSR\n/g) || []).length === 3);
  check('final shader carries SSR_COMPOSITE_GLSL and LUT_GLSL', fFrag.includes(SSR_COMPOSITE_GLSL) && fFrag.includes(LUT_GLSL.trim()) && tFrag.includes(SSR_COMPOSITE_GLSL));
  const at = (t) => fFrag.indexOf(t);
  const iSsr = at('col = engineSsrComposite( col, texture2D( tSSR, vUv ) );'), iAO = at('if ( uHasAO > 0.5 ) {'), iFetch = at('else col = texture2D( tColor, vUv ).rgb;');
  check('FINAL_SSR: composite after the colour fetch, before AO and fog', iSsr > iFetch && iSsr < iAO);
  const iVol = at('col += uVolColor * volumeAt( depth );'), iFogEnd = at('dbgT = T;'), iBloom = at('if ( uHasBloom > 0.5 ) col += texture2D( tBloom, vUv ).rgb;');
  check('FINAL_VOLUMETRIC: col += uVolColor * vol after the fog (every pixel), before bloom', iVol > iFogEnd && iVol < iBloom && fFrag.slice(iFogEnd, iVol).includes('}\n      }'));
  const iLut = at('o = mix( o, engineApplyLut( tLUT, o, uLutSize ), uLutStrength );'), iSRGB = at('vec3 o = toSRGB( clamp( col, 0.0, 1.0 ) );'), iGrain = at('if ( uGrain > 0.0 )');
  check('FINAL_LUT: after toSRGB, before grain / dither', iLut > iSRGB && iLut < iGrain);
  check('volume upsample: aoAt weights, sky = 60000 m', /\( 0\.06 \* z \+ 0\.15 \)/.test(stripBlocks(fFrag, /^#ifdef AO_UPSAMPLE_DEPTH$/)) && fFrag.includes(': 60000.0 );') && fFrag.includes('uniform highp sampler3D tLUT;'));
  check('TAA_SSR: one tSSR fetch, composited into every tap before the YCoCg tonemap', (tFrag.match(/texture2D\( tSSR, vUv \)/g) || []).length === 1
    && tFrag.includes('#undef TAA_TAP') && tFrag.includes('vec3 c = tm( engineSsrComposite( texelFetch( tColor, clamp( ip + ivec2( X, Y ), ivec2( 0 ), mx ), 0 ).rgb, ssrC ) );'));
  check('TAA_SSR: the fresh-frame output composites too', tFrag.includes('gl_FragColor = vec4( finite( engineSsrComposite( texelFetch( tColor, ip, 0 ).rgb, ssrC ) ), z ); return; }'));

  // ---- defaults: no new define, the 0.7.0 pass list, no new target
  const scene = new THREE.Scene(), cam = new THREE.PerspectiveCamera(60, 16 / 9, 0.1, 1000);
  cam.position.set(0, 2, 10); cam.updateMatrixWorld();
  const fx = new PostFX(stub(), scene, cam, { atmosphere: null });
  const gfx = new PostFX(stub(), scene, cam, { atmosphere: null, gbuffer: true });
  check('defaults: final + TAA materials use the exported texts', fx.finalPass.material.fragmentShader === FINAL_SHADER.fragmentShader && gfx.taaPass.material.fragmentShader === TAA_SHADER.fragmentShader);
  const boundIn = (f) => { const r = f.renderer, set = new Set(), srt = r.setRenderTarget; r.setRenderTarget = (t) => { set.add(t); srt(t); }; f.render(0); r.setRenderTarget = srt; return set; };
  const known070 = (f) => {
    const b = f.bloomPass, k = new Set([null, f.composer.readBuffer, f.exposurePass.lumRT, f.exposurePass.rtA, f.exposurePass.rtB, f.aoPass.rtA, f.aoPass.rtB, b.renderTargetBright, ...b.renderTargetsHorizontal, ...b.renderTargetsVertical]);
    if (f.taaPass) for (const rt of f.taaPass._rt) k.add(rt);
    return k;
  };
  for (const [name, f, n] of [['forward', fx, 5], ['G-buffer', gfx, 6]]) {
    const b1 = boundIn(f), b2 = boundIn(f), k = known070(f);
    const d = f.finalPass.material.defines;
    check(`default ${name}: no FINAL_* define after render()`, NEW_FINAL_DEFINES.every((x) => !(x in d)));
    check(`default ${name}: ${n} passes, no 0.8.0 pass`, f.composer.passes.length === n && f.ssrPass === null && f.volumetricPass === null && f.motionBlurPass === null && f.dofPass === null);
    check(`default ${name}: render() binds only 0.7.0 targets`, [...b1, ...b2].every((t) => k.has(t)));
    check(`default ${name}: getters off`, f.ssr === false && f.volumetric === false && f.motionBlur === 0 && f.dof === null && f.lut === null);
  }
  check('default G-buffer: no TAA_SSR define, taaPass.ssr null', !('TAA_SSR' in gfx.taaPass.material.defines) && gfx.taaPass.ssr === null);
  check('default G-buffer: AO normals depth, aoPass.gbuffer set', gfx.aoPass.normals === 'depth' && gfx.aoPass.gbuffer === gfx.gbuffer && fx.aoPass.gbuffer === null);
  check('timings keys (0.8.0 order)', Object.keys(fx.timings).join() === 'scene,exposure,ao,ssr,volumetric,taa,motionBlur,dof,bloom,final,flare,total');
  check('kept options: defaults of the pass modules', fx.ssrOptions.steps === SSR_DEFAULTS.steps && fx.dofOptions.fStop === DOF_DEFAULTS.fStop && fx.volumetricOptions.intensity === 1 && fx.motionBlurOptions.strength === 0 && fx.lutOptions.texture === null);

  // ---- option validation (constructor): every bad value throws, naming the option
  const mk = (o) => () => new PostFX(stub(), scene, cam, { atmosphere: null, ...o });
  const mkA = (o) => () => new PostFX(stub(), scene, cam, { atmosphere: atmoStub(), ...o });
  check("ssr without gbuffer throws (names gbuffer)", throws(mk({ ssr: true }), /ssr needs gbuffer: true/));
  check("motionBlur without gbuffer throws", throws(mk({ motionBlur: 0.5 }), /motionBlur needs gbuffer: true/) && throws(mk({ motionBlur: {} }), /motionBlur needs gbuffer/));
  check("ao.normals 'gbuffer' without gbuffer throws", throws(mk({ ao: { normals: 'gbuffer' } }), /ao\.normals 'gbuffer' needs gbuffer: true/));
  check('volumetric without an atmosphere throws', throws(mk({ volumetric: true }), /volumetric needs an atmosphere/));
  check('ssr: bad shape / unknown key throw', throws(mk({ gbuffer: true, ssr: 'on' }), /\[PostFX\] ssr: expected/) && throws(mk({ gbuffer: true, ssr: { stepz: 8 } }), /unknown ssr option 'stepz'/));
  check('ssr.steps: 0 / 2.5 / 200 throw', throws(mk({ gbuffer: true, ssr: { steps: 0 } }), /ssr\.steps/) && throws(mk({ gbuffer: true, ssr: { steps: 2.5 } }), /ssr\.steps/) && throws(mk({ gbuffer: true, ssr: { steps: 200 } }), /ssr\.steps/));
  check('ssr.scale 0 / 1.5, maxDistance NaN, edgeFade 0.6 throw', throws(mk({ gbuffer: true, ssr: { scale: 0 } }), /ssr\.scale/) && throws(mk({ gbuffer: true, ssr: { scale: 1.5 } }), /ssr\.scale/)
    && throws(mk({ gbuffer: true, ssr: { maxDistance: NaN } }), /ssr\.maxDistance/) && throws(mk({ gbuffer: true, ssr: { edgeFade: 0.6 } }), /ssr\.edgeFade/));
  check("ssr.source 'history' throws", throws(mk({ gbuffer: true, ssr: { source: 'history' } }), /ssr\.source/));
  check('motionBlur: -1 / NaN / shape / unknown key throw', throws(mk({ gbuffer: true, motionBlur: -1 }), /motionBlur\.strength/) && throws(mk({ gbuffer: true, motionBlur: NaN }), /motionBlur\.strength/)
    && throws(mk({ gbuffer: true, motionBlur: 'fast' }), /\[PostFX\] motionBlur: expected/) && throws(mk({ gbuffer: true, motionBlur: { scale: 0.5 } }), /unknown motionBlur option 'scale'/));
  check('motionBlur.samples 1 / tileSize 0 / maxBlur 0 throw', throws(mk({ gbuffer: true, motionBlur: { samples: 1 } }), /motionBlur\.samples/) && throws(mk({ gbuffer: true, motionBlur: { tileSize: 0 } }), /motionBlur\.tileSize/)
    && throws(mk({ gbuffer: true, motionBlur: { maxBlur: 0 } }), /motionBlur\.maxBlur/));
  check('motionBlur 0 / false without gbuffer: no pass, no throw', !throws(mk({ motionBlur: 0 })) && !throws(mk({ motionBlur: false })));
  check('volumetric: g 1 / blend 1 / intensity -1 / steps 0 / unknown key throw', throws(mkA({ volumetric: { g: 1 } }), /volumetric\.g/) && throws(mkA({ volumetric: { blend: 1 } }), /volumetric\.blend/)
    && throws(mkA({ volumetric: { intensity: -1 } }), /volumetric\.intensity/) && throws(mkA({ volumetric: { steps: 0 } }), /volumetric\.steps/) && throws(mkA({ volumetric: { stepz: 8 } }), /unknown volumetric option 'stepz'/));
  check('dof: shape / unknown key / NaN throw', throws(mk({ dof: 5 }), /\[PostFX\] dof: expected/) && throws(mk({ dof: { fstop: 2 } }), /unknown dof option 'fstop'/) && throws(mk({ dof: { fStop: NaN } }), /dof\.fStop/));
  const lut4 = identityLut(4);
  check('lut: bad texture / size / strength / unknown key throw', throws(mk({ lut: { texture: new THREE.Texture(), size: 4 } }), /lut\.texture/) && throws(mk({ lut: { texture: lut4.texture, size: 1 } }), /lut\.size/)
    && throws(mk({ lut: { ...lut4, strength: 'x' } }), /lut\.strength/) && throws(mk({ lut: { ...lut4, gamma: 2 } }), /unknown lut option 'gamma'/) && throws(mk({ lut: 'warm' }), /\[PostFX\] lut: expected/));
  check("ao.normals: unknown value throws", throws(mk({ gbuffer: true, ao: { normals: 'view' } }), /ao\.normals/));
  {
    const v = new PostFX(stub(), scene, cam, { atmosphere: null, lut: lut4 });
    check('lut: the Assets value as it is -> strength 1, size from it', v.lut.texture === lut4.texture && v.lut !== v.lutOptions && v.lutOptions.texture === lut4.texture && v.lutOptions.size === 4 && v.lutOptions.strength === 1);
    v.dispose();
  }

  // ---- everything on: pass order, getters, hooks
  const atmo = atmoStub();
  const all = new PostFX(stub(), scene, cam, { atmosphere: atmo, gbuffer: true, ssr: { steps: 16, jitter: true }, volumetric: { intensity: 2 }, motionBlur: 0.5, dof: { focusDistance: 8 }, lut: lut4, ao: { normals: 'gbuffer' } });
  check('all on: order scene, exposure, ao, ssr, volumetric, taa, motionBlur, dof, bloom, final', keysOf(all) === 'scene,exposure,ao,ssr,volumetric,taa,motionBlur,dof,bloom,final');
  check('all on: pass classes', all.ssrPass instanceof SSRPass && all.volumetricPass instanceof VolumetricPass && all.motionBlurPass instanceof MotionBlurPass && all.dofPass instanceof DoFPass);
  check('all on: no pass swaps', all.composer.passes.every((p) => !p.needsSwap));
  check('all on: options reach the passes', all.ssrPass.steps === 16 && all.motionBlurPass.strength === 0.5 && all.dofPass.focusDistance === 8 && all.volumetricPass.steps === 16);
  check('all on: getters (dof / lut as copies)', all.ssr === true && all.volumetric === true && all.motionBlur === 0.5 && all.dof !== all.dofOptions && all.dof.focusDistance === 8
    && all.lut !== all.lutOptions && all.lut.texture === lut4.texture && all.lut.strength === 1);
  check('all on: SSR jitter in effect under TAA', all.ssrPass.jitter === true && all.taaPass.enabled);
  check('all on: TAA composites SSR (taaPass.ssr), AO reads the G-buffer normals', all.taaPass.ssr === all.ssrPass && all.aoPass.normals === 'gbuffer' && all.aoPass.gbuffer === all.gbuffer);
  check('all on: volume pass shares the CameraHistory', all.volumetricPass.history === all.history && all.ssrPass.history === all.history && all.motionBlurPass.history === all.history);
  all.render(0);
  const fd = all.finalPass.material.defines, fu = all.uniforms;
  check('TAA on + SSR: TAA_SSR, tSSR bound; no FINAL_SSR', 'TAA_SSR' in all.taaPass.material.defines && all.taaPass.material.uniforms.tSSR.value === all.ssrPass.texture && !('FINAL_SSR' in fd));
  check('volumetric: FINAL_VOLUMETRIC, tVolume = pass texture', 'FINAL_VOLUMETRIC' in fd && fu.tVolume.value === all.volumetricPass.texture);
  check('volumetric: uVolColor = sunColor * sun.intensity * intensity', Math.abs(fu.uVolColor.value.r - 6) < 1e-6 && Math.abs(fu.uVolColor.value.g - 5.4) < 1e-6 && Math.abs(fu.uVolColor.value.b - 4.8) < 1e-6);
  check('volumetric on: analytic sun glow off (uSunScatter 0)', fu.uSunScatter.value === 0);
  check('lut: FINAL_LUT, tLUT / uLutSize / uLutStrength', 'FINAL_LUT' in fd && fu.tLUT.value === lut4.texture && fu.uLutSize.value === 4 && fu.uLutStrength.value === 1);
  all.set('aa', 'fxaa'); all.render(0);
  check('TAA off + SSR: FINAL_SSR, tSSR = SSR texture', 'FINAL_SSR' in fd && fu.tSSR.value === all.ssrPass.texture);
  all.ssrPass.enabled = false; all.render(0);
  check('SSR pass disabled: no FINAL_SSR', !('FINAL_SSR' in fd));
  all.ssrPass.enabled = true; all.set('aa', 'taa'); all.render(0);
  check('TAA back on: FINAL_SSR gone', !('FINAL_SSR' in fd) && 'TAA_SSR' in all.taaPass.material.defines);
  all.debug = 'volumetric';
  check("debug 'volumetric' round trip", all.debug === 'volumetric' && fu.uDebug.value === 4);
  all.debug = null;

  // ---- colorSource chain: last enabled of TAA -> MotionBlur -> DoF; MotionBlur reads up to TAA, DoF up to MotionBlur
  const seen = {}, mbR = MotionBlurPass.prototype.render, dofR = DoFPass.prototype.render;
  MotionBlurPass.prototype.render = function (r, w, rb) { seen.mb = rb.texture; seen.mbDepth = rb.depthTexture; };
  DoFPass.prototype.render = function (r, w, rb) { seen.dof = rb.texture; seen.dofDepth = rb.depthTexture; };
  try {
    const rb = all.composer.readBuffer, bloomIn = () => all.bloomPass.materialHighPassFilter.uniforms.tDiffuse.value;
    const reads = () => { seen.mb = seen.dof = undefined; all.render(0); return [seen.mb, seen.dof, bloomIn(), fu.tColor.value]; };
    let [m, d, b, f] = reads();
    check('chain TAA + MB + DoF: MB <- TAA, DoF <- MB, Bloom / Final <- DoF', m === all.taaPass.texture && d === all.motionBlurPass.texture && b === all.dofPass.texture && f === all.dofPass.texture && all.colorSource === all.dofPass.texture);
    check('chain: the proxies carry the scene depth', seen.mbDepth === rb.depthTexture && seen.dofDepth === rb.depthTexture);
    all.dofPass.enabled = false; [m, d, b, f] = reads();
    check('chain TAA + MB: Bloom / Final <- MB', m === all.taaPass.texture && d === undefined && b === all.motionBlurPass.texture && f === all.motionBlurPass.texture && all.colorSource === all.motionBlurPass.texture);
    all.motionBlurPass.enabled = false; [m, d, b, f] = reads();
    check('chain TAA only: Bloom / Final <- TAA', b === all.taaPass.texture && f === all.taaPass.texture && all.colorSource === all.taaPass.texture);
    all.dofPass.enabled = true; [m, d, b, f] = reads();
    check('chain TAA + DoF (MB disabled): DoF <- TAA', d === all.taaPass.texture && f === all.dofPass.texture);
    all.motionBlurPass.enabled = true; all.set('aa', 'fxaa'); [m, d, b, f] = reads();
    check('chain TAA off: MB <- scene colour, DoF <- MB, Final <- DoF', m === rb.texture && d === all.motionBlurPass.texture && f === all.dofPass.texture);
    all.set('aa', 'taa');
    const dfx = new PostFX(stub(), scene, cam, { atmosphere: null, dof: true });
    const drb = dfx.composer.readBuffer;
    seen.dof = undefined; dfx.render(0);
    check('forward + DoF alone: 6 passes, DoF <- scene colour, Bloom / Final <- DoF', dfx.composer.passes.length === 6 && keysOf(dfx) === 'scene,exposure,ao,dof,bloom,final' && seen.dof === drb.texture
      && dfx.uniforms.tColor.value === dfx.dofPass.texture && dfx.bloomPass.materialHighPassFilter.uniforms.tDiffuse.value === dfx.dofPass.texture && dfx.colorSource === dfx.dofPass.texture);
    dfx.dofPass.enabled = false;
    check('forward, DoF disabled: colorSource = scene colour, _source = readBuffer', dfx.colorSource === drb.texture && dfx._source(drb) === drb);
    dfx.dispose();
  } finally {
    MotionBlurPass.prototype.render = mbR; DoFPass.prototype.render = dofR;
  }

  // ---- set() / get(): create, update, remove (dispose), G-buffer rules
  const s = new PostFX(stub(), scene, cam, { atmosphere: atmoStub(), gbuffer: true });
  s.set('ssr.steps', 32);
  check("set('ssr.steps') while SSR is off: kept", s.ssrOptions.steps === 32 && s.ssrPass === null && s.get('ssr.steps') === 32);
  s.set('ssr', true);
  check("set('ssr', true): pass at its place with the kept options, TAA hooked", s.ssrPass instanceof SSRPass && s.ssrPass.steps === 32 && keysOf(s) === 'scene,exposure,ao,ssr,taa,bloom,final' && s.taaPass.ssr === s.ssrPass && s.get('ssr') === true);
  s.set('ssr.thickness', 0.8);
  check("set('ssr.thickness') reaches the live pass", s.ssrPass.thickness === 0.8 && s.ssrOptions.thickness === 0.8);
  check("set('ssr.steps', 300) throws, nothing changes", throws(() => s.set('ssr.steps', 300), /ssr\.steps/) && s.ssrPass.steps === 32 && s.ssrOptions.steps === 32);
  s.set('ssr', { maxDistance: 60 });
  check("set('ssr', options) replaces the options, live", s.ssrPass.maxDistance === 60 && s.ssrPass.steps === SSR_DEFAULTS.steps && s.ssrOptions.thickness === SSR_DEFAULTS.thickness);
  let sp = s.ssrPass, spy = spyDispose(sp);
  s.set('ssr', false);
  check("set('ssr', false): removed, disposed, TAA unhooked", s.ssrPass === null && spy.n === 1 && !s.composer.passes.includes(sp) && s.taaPass.ssr === null && !s._passKeys.has(sp));
  s.render(0);
  check('SSR removed: TAA_SSR define dropped at the next render', !('TAA_SSR' in s.taaPass.material.defines));
  s.set('motionBlur', 0.75);
  check("set('motionBlur', 0.75): pass created between TAA and Bloom", s.motionBlurPass instanceof MotionBlurPass && s.motionBlurPass.strength === 0.75 && s.motionBlur === 0.75 && keysOf(s) === 'scene,exposure,ao,taa,motionBlur,bloom,final');
  s.set('motionBlur.samples', 12); s.set('motionBlur', 1.5);
  check("set('motionBlur.samples' / number): live update", s.motionBlurPass.samples === 12 && s.motionBlurPass.strength === 1.5 && s.get('motionBlur') === 1.5);
  s.set('motionBlur.shutter', 0.01);
  check("set('motionBlur.shutter') round trip (live pass + get), bad values throw", s.motionBlurPass.shutter === 0.01 && s.get('motionBlur.shutter') === 0.01
    && throws(() => s.set('motionBlur.shutter', -1), /motionBlur\.shutter/) && throws(() => s.set('motionBlur.shutter', Infinity), /motionBlur\.shutter/) && s.get('motionBlur.shutter') === 0.01);
  s.set('motionBlur.shutter', 0);
  sp = s.motionBlurPass; spy = spyDispose(sp);
  s.set('motionBlur', 0);
  check("set('motionBlur', 0): removed + disposed, samples kept", s.motionBlurPass === null && spy.n === 1 && s.motionBlur === 0 && s.motionBlurOptions.samples === 12);
  s.set('motionBlur.strength', 1);
  check("set('motionBlur.strength', 1): created again", s.motionBlurPass !== null && s.motionBlurPass.samples === 12);
  sp = s.motionBlurPass; spy = spyDispose(sp);
  s.set('motionBlur.strength', 0);
  check("set('motionBlur.strength', 0): removed + disposed", s.motionBlurPass === null && spy.n === 1);
  s.set('motionBlur', { strength: 2, maxBlur: 16 });
  check("set('motionBlur', options)", s.motionBlurPass.strength === 2 && s.motionBlurPass.maxBlur === 16 && s.motionBlurOptions.samples === 8);
  s.set('dof', { focusDistance: 5 });
  check("set('dof', options): created after motion blur", s.dofPass instanceof DoFPass && s.dofPass.focusDistance === 5 && keysOf(s) === 'scene,exposure,ao,taa,motionBlur,dof,bloom,final');
  s.set('dof.fStop', 4);
  check("set('dof.fStop') live + get", s.dofPass.fStop === 4 && s.get('dof.fStop') === 4 && s.get('dof').fStop === 4 && s.get('dof') !== s.dofOptions);
  // the lens rule (focusDistance * 1000 > focalLength) holds for the old and the new pair: applied in an order that keeps
  // it at every step (a shorter focal length first, a longer one last)
  s.set('dof', { focusDistance: 0.04, focalLength: 30 });
  check("set('dof', { focusDistance 0.04, focalLength 30 }) on a 5 m / 50 mm pass", s.dofPass.focusDistance === 0.04 && s.dofPass.focalLength === 30 && s.get('dof.focusDistance') === 0.04 && s.get('dof.focalLength') === 30);
  s.set('dof', { focusDistance: 12, focalLength: 85 });
  check("set('dof', { 12 m, 85 mm }) back from 0.04 m / 30 mm", s.dofPass.focusDistance === 12 && s.dofPass.focalLength === 85);
  check("set('dof', invalid lens pair) throws, pass and options unchanged", throws(() => s.set('dof', { focusDistance: 0.02, focalLength: 30 }), /dof\.focusDistance/)
    && s.dofPass.focusDistance === 12 && s.dofPass.focalLength === 85 && s.get('dof.focusDistance') === 12 && s.get('dof.focalLength') === 85);
  check("set('dof.focalLength', 20000) throws (lens rule), unchanged", throws(() => s.set('dof.focalLength', 20000), /must exceed/) && s.dofPass.focalLength === 85 && s.dofOptions.focalLength === 85);
  { // a harness plan entry: get() before, set(), restore with set(path, before) -> the pass is back where it was
    const before = s.get('dof'), passBefore = s.dofPass;
    s.set('dof', { focusDistance: 3 });
    const moved = s.dofPass.focusDistance === 3 && before.focusDistance === 12;
    s.set('dof', before);
    check("dof: get -> set -> set(before) round trip restores the pass (the copy is not mutated)", moved && s.dofPass === passBefore && s.dofPass.focusDistance === 12 && s.dofPass.focalLength === 85 && s.get('dof').focalLength === 85);
    s.set('lut', lut4); s.set('lut.strength', 0.7);
    const lb = s.get('lut');
    s.set('lut', { ...lut4, strength: 0.2 }); s.set('lut.strength', 0.3);
    const lm = s.lutOptions.strength === 0.3 && lb.strength === 0.7;
    s.set('lut', lb);
    check("lut: get -> set -> set(before) round trip", lm && s.lutOptions.strength === 0.7 && s.lutOptions.texture === lut4.texture && s.get('lut').strength === 0.7);
    const nb = s.get('lut'); s.set('lut', null); s.set('lut', nb); s.set('lut', null);
    check("lut: restoring null removes it", s.lut === null && s.get('lut') === null);
  }
  sp = s.dofPass; spy = spyDispose(sp);
  s.set('dof', null);
  check("set('dof', null): removed + disposed", s.dofPass === null && spy.n === 1 && s.dof === null);
  s.set('volumetric', true);
  check("set('volumetric', true): before TAA", s.volumetricPass instanceof VolumetricPass && keysOf(s) === 'scene,exposure,ao,volumetric,taa,motionBlur,bloom,final' && s.volumetricPass.history === s.history);
  s.set('volumetric.intensity', 0.5); s.set('volumetric.steps', 24); s._syncFrame(0);
  check("set('volumetric.*'): intensity -> uVolColor, steps -> pass", s.volumetricPass.steps === 24 && Math.abs(s.uniforms.uVolColor.value.r - 1.5) < 1e-6 && s.uniforms.uSunScatter.value === 0);
  sp = s.volumetricPass; spy = spyDispose(sp);
  s.set('volumetric', false); s._syncFrame(0);
  check("set('volumetric', false): removed + disposed, sun glow back", s.volumetricPass === null && spy.n === 1 && s.uniforms.uSunScatter.value === 0.05);
  s.set('lut', lut4); s.set('lut.strength', 0.4); s.render(0);
  check("set('lut') / set('lut.strength')", 'FINAL_LUT' in s.finalPass.material.defines && s.uniforms.uLutStrength.value === 0.4 && s.get('lut.strength') === 0.4 && s.get('lut').strength === 0.4);
  s.set('lut', null); s.render(0);
  check("set('lut', null): FINAL_LUT gone", !('FINAL_LUT' in s.finalPass.material.defines) && s.lut === null);
  s.set('ao.normals', 'gbuffer');
  check("set('ao.normals', 'gbuffer')", s.aoPass.normals === 'gbuffer' && s.get('ao.normals') === 'gbuffer');
  s.set('ssr', true);
  const ssr0 = s.ssrPass, mb0 = s.motionBlurPass, sSpy = spyDispose(ssr0), mSpy = spyDispose(mb0);
  s.setGBuffer(false);
  check('setGBuffer(false): SSR + motion blur torn down and disposed, effective getters off', s.ssrPass === null && s.motionBlurPass === null && sSpy.n === 1 && mSpy.n === 1 && s.ssr === false && s.motionBlur === 0);
  check('setGBuffer(false): AO back to depth normals (request kept), pass list 5', s.aoPass.normals === 'depth' && s.aoPass.gbuffer === null && s.get('ao.normals') === 'gbuffer' && s.composer.passes.length === 5);
  check('set(ssr / motionBlur / ao.normals gbuffer) without the G-buffer throws', throws(() => s.set('ssr', true), /needs gbuffer/) && throws(() => s.set('motionBlur', 1), /needs gbuffer/)
    && throws(() => s.set('motionBlur.strength', 1), /needs gbuffer/) && throws(() => s.set('ao.normals', 'gbuffer'), /needs gbuffer/));
  s.set('volumetric', true);
  check('forward path: volumetric works without a CameraHistory', s.volumetricPass !== null && s.volumetricPass.history === null && !throws(() => s.render(0)));
  s.setGBuffer(true);
  check('setGBuffer(true): SSR + motion blur rebuilt from the kept options', s.ssrPass !== null && s.ssrPass !== ssr0 && s.motionBlurPass !== null && s.motionBlurPass.strength === 2 && s.motionBlurPass.maxBlur === 16 && s.taaPass.ssr === s.ssrPass);
  check('setGBuffer(true): AO normals gbuffer again, volume pass on the new history', s.aoPass.normals === 'gbuffer' && s.aoPass.gbuffer === s.gbuffer && s.volumetricPass.history === s.history);
  check('rebuilt order', keysOf(s) === 'scene,exposure,ao,ssr,volumetric,taa,motionBlur,bloom,final');
  check("set('volumetric', true) without an atmosphere throws", throws(() => gfx.set('volumetric', true), /needs an atmosphere/) && gfx.volumetricPass === null);
  check('unknown paths throw (set / get)', throws(() => s.set('ssr.stepz', 1), /unknown path/) && throws(() => s.set('volumetric.enabled', true), /unknown path/) && throws(() => s.get('dof.enabled'), /unknown path/));
  check('get(): existing paths', s.get('aa') === 'auto' && s.get('gbuffer') === true && s.get('jitter') === true && s.get('taa.blend') === 0.9 && s.get('ao.enabled') === true && s.get('bloom.enabled') === true);
  s.set('taa.blend', 0.8);
  check("set('taa.blend') still reaches the live pass", s.taaPass.blend === 0.8 && s.taaOptions.blend === 0.8);

  // SSR's march jitter (off by default since the TAA hook cannot converge it; requested here) only while TAA runs; ssr.jitter stays the request
  check('SSR jitter off by default (request and effect)', s.ssrPass.jitter === false && s.get('ssr.jitter') === false);
  s.set('ssr.jitter', true);
  check('SSR jitter on under TAA', s.ssrPass.jitter === true && s.get('ssr.jitter') === true);
  s.set('aa', 'fxaa');
  check("aa 'fxaa': SSR jitter off in effect, request kept", s.ssrPass.jitter === false && s.get('ssr.jitter') === true && s.ssrOptions.jitter === true);
  s.set('ssr', { maxDistance: 50 });
  check("set('ssr', options) under FXAA: options replaced (jitter back to its default), still off in effect", s.ssrPass.jitter === false && s.ssrPass.maxDistance === 50 && s.get('ssr.jitter') === false);
  s.set('ssr', false); s.set('ssr', true);
  check('SSR created under FXAA: jitter off', s.ssrPass.jitter === false);
  s.set('ssr.jitter', false); s.set('aa', 'taa');
  check("ssr.jitter false: stays off under TAA", s.ssrPass.jitter === false && s.get('ssr.jitter') === false);
  s.set('ssr.jitter', true);
  check("ssr.jitter true under TAA: on", s.ssrPass.jitter === true);

  // a throw leaves nothing staged; motionBlur options replace (strength back to 1); messages quote strings
  check("set('volumetric', options) without an atmosphere: throws, nothing staged", throws(() => gfx.set('volumetric', { steps: 8 }), /needs an atmosphere/) && gfx._next === null && gfx.volumetricOptions.steps === 16);
  check('ssr shape error quotes the value', throws(() => s.set('ssr', 'on'), /got 'on'/));
  s.set('motionBlur', { maxBlur: 16 });
  check("set('motionBlur', { maxBlur }) replaces the options: strength 1 (documented)", s.motionBlurPass.strength === 1 && s.motionBlurPass.maxBlur === 16 && s.motionBlur === 1);
  s.set('motionBlur', { strength: 2, maxBlur: 16 });
  {
    const warn = console.warn; let warned = 0;
    console.warn = () => { warned++; };
    const ok = !throws(() => s.set('lut', { ...lut4, domainMin: 1 })) && !throws(() => s.set('lut', { ...lut4, domainMax: [1, 1, 2] }));
    console.warn = warn;
    check('lut: non-array / other domain warns, no TypeError', ok && warned === 2);
    s.set('lut', null);
  }

  // ---- timings with every pass, then zeroed for a removed one
  s.set('dof', true); s.set('timing', true); s.render(0);
  check('timed render: every 0.8.0 pass timed', ['ssr', 'volumetric', 'motionBlur', 'dof', 'taa'].every((k) => Number.isFinite(s.timings[k]) && s.timings[k] >= 0) && [...s._passKeys.values()].length === s.composer.passes.length);
  s.timings.ssr = 5; s.set('ssr', false); s.render(0);
  check('timed render: a removed pass reads 0', s.timings.ssr === 0);
  s.set('timing', false); s.set('ssr', true);

  // ---- allocation-free: _source, value set() paths, render() with every pass on
  const rb = s.composer.readBuffer;
  for (let i = 0; i < 20000; i++) { s._source(rb); s._source(rb, 1); s._source(rb, 2); }
  check('_source() x600000: bytes allocated', allocated(() => { for (let i = 0; i < 200000; i++) { s._source(rb); s._source(rb, 1); s._source(rb, 2); } }), -Infinity, 64 * 1024, 'B');
  // value paths one at a time (a harness sweeps one knob), small integers. V8 boxes a double (16 B HeapNumber) that a
  // megamorphic keyed store writes into a double-represented field (ssr.thickness after set() has seen every path):
  // at most that per call, i.e. no object / array / closure / string (>= 32 B each without pointer compression)
  s.set('lut', lut4);
  let worst = 0, worstPath = '';
  for (const [path, base] of [['ssr.thickness', 0], ['ssr.steps', 20], ['volumetric.intensity', 0], ['volumetric.density', 0], ['motionBlur.strength', 0],
    ['motionBlur', 0], ['motionBlur.maxBlur', 20], ['dof.focusDistance', 10], ['dof.fStop', 1], ['lut.strength', -1], ['taa.sharpen', -1]]) {
    const run = (n) => { for (let i = 0; i < n; i++) s.set(path, base + (i & 1) + 1); };
    run(20000);
    const b = allocated(() => run(50000)) / 50000;
    if (b > worst) { worst = b; worstPath = path; }
  }
  check(`value set() paths: worst bytes / call (${worstPath}; one boxed double = 16)`, worst, -Infinity, 20, 'B');
  // render(): a budget, not a measure of real allocation: the figure (~600 B / frame) is dominated by the measurement
  // itself (a sampling profiler shows ~14 B / frame allocated, and a fresh instance matches 0.7.0's)
  for (let i = 0; i < 300; i++) s.render(0.016);
  const perFrame = allocated(() => { for (let i = 0; i < 3000; i++) s.render(0.016); }) / 3000;
  check('render() every pass on (stub renderer): bytes / frame, budget', perFrame, -Infinity, 2048, 'B');

  check('dispose (every pass on): no throw', !throws(() => { all.dispose(); s.dispose(); gfx.dispose(); fx.dispose(); }));
}

console.log(results.join('\n'));
console.log(failures ? `\n${failures} check(s) FAILED` : '\nAll post-pipeline option checks passed.');
process.exit(failures ? 1 : 0);
