// Headless checks for the post pipeline's CPU side (no browser, no WebGL): CameraHistory (jitter sequence, matrices,
// exact restore, reset, no allocation), the TAA reconstruction weights, resolveAA (the anti-aliasing decision: the
// 0.6.0 rule with the G-buffer off, TAA / single-sampled with it on), and PostFX option plumbing on a stub renderer
// (construction only: nothing here draws).
// Run: node test/postfx-options.mjs
import v8 from 'node:v8';
import * as THREE from 'three';
import { CameraHistory, jitterPattern } from '../src/render/CameraHistory.js';
import { GBUFFER_UNIFORMS } from '../src/render/GBuffer.js';
import { PostFX, resolveAA, AA_MODES } from '../src/render/PostFX.js';
import { BloomPass } from '../src/render/passes/BloomPass.js';
import { taaFilterWeights, TAA_REJECTS } from '../src/render/passes/TAAPass.js';

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
{
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

console.log(results.join('\n'));
console.log(failures ? `\n${failures} check(s) FAILED` : '\nAll post-pipeline option checks passed.');
process.exit(failures ? 1 : 0);
