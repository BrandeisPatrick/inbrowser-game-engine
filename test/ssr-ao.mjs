// Headless checks for the screen-space reflection pass and AOPass's G-buffer normals (no browser, no WebGL):
//   SSRPass  defaults, option validation (messages name the option), live properties -> uniforms / defines (a program
//            change only when a define changes), output sizing (scale, maxPixels budget, 1 x 1 minimum), the output
//            target format, the shader contract (uniforms, defines, decode helpers, premultiplied output) as text,
//            SSR_COMPOSITE_GLSL verbatim, render() bindings and the empty-composite fallback on a stub renderer, no
//            allocation per render, and a CPU port of the fragment shader's trace (same steps and names) on an
//            analytic depth buffer: the perspective-correct interpolation, a floor reflecting a wall at the mirrored
//            point, a ray over the wall missing into the sky, a normal-mapped self-hit rejected at the march start, a
//            ray with R.z > 0 passing 1 cm in front of a panel (miss), a ray passing behind a sign on its way to a wall
//            (the hit lands on the wall), and the metalTint stopgap's energy bound.
//   AOPass   normals 'depth' (default) pinned to the GTAO fragment shader text and defines of the file before the
//            option existed (sha256 taken from the committed 0.7.0 AOPass), 'gbuffer' switching the define and text
//            only on a state change, the fallback warning once, no allocation per render.
// Real compilation and the image are checked in the browser by the lead (PostFX wiring is the compositor's).
// Run: node test/ssr-ao.mjs
import { bytesPerCall, gate } from './helpers/alloc.mjs';
import { createHash } from 'node:crypto';
import * as THREE from 'three';
import { Pass } from 'three/addons/postprocessing/Pass.js';
import { SSRPass, SSR_DEFAULTS, SSR_SOURCES, SSR_COMPOSITE_GLSL } from '../src/render/passes/SSRPass.js';
import { AOPass, AO_NORMALS } from '../src/render/passes/AOPass.js';
import { CameraHistory } from '../src/render/CameraHistory.js';
import { GBUFFER_GLSL_DECODE } from '../src/render/GBuffer.js';

let failures = 0, passes = 0;
const results = [];
function check(name, ok, detail = '') {
  if (ok) passes++; else failures++;
  results.push(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail !== '' ? `  (${detail})` : ''}`);
  return ok;
}
const throws = (fn, re) => { try { fn(); return false; } catch (e) { return re ? re.test(e.message) : true; } };
const sha = (s) => createHash('sha256').update(s).digest('hex');
const sameBits = (a, b) => a.elements.every((v, i) => Object.is(v, b.elements[i]));
const near = (a, b, tol) => Math.abs(a - b) <= tol;

// allocation gates (test/helpers/alloc.mjs: exact bytes per call of TurboFan code, with a control) report through these
const note = (s) => results.push(`NOTE  ${s}`);
const inRange = (name, v, lo, hi, unit) => check(name, Number.isFinite(v) && v >= lo && v <= hi, `${Number.isFinite(v) ? +v.toFixed(3) : v} ${unit} in [${lo}, ${hi}]`);

// enough of WebGLRenderer for the passes' render() (draws are no-ops); records bound targets and clears
function stubRenderer() {
  const clearColor = new THREE.Color(0.2, 0.3, 0.4);
  const r = {
    bound: null, clears: 0, clearAlphaSet: 1, clearColorAtClear: [0, 0, 0, 0], autoClear: true,
    getClearColor: (c) => c.copy(clearColor), getClearAlpha: () => r.clearAlphaSet,
    setClearColor(c, a = 1) { clearColor.copy(c); r.clearAlphaSet = a; },
    setRenderTarget(t) { r.bound = t; }, getRenderTarget: () => r.bound,
    clear() { r.clears++; const a = r.clearColorAtClear; a[0] = clearColor.r; a[1] = clearColor.g; a[2] = clearColor.b; a[3] = r.clearAlphaSet; },
    render() {}, getActiveCubeFace: () => 0, getActiveMipmapLevel: () => 0,
    clearColor,
  };
  return r;
}

// a tiny GLSL text lint: balanced delimiters outside comments, every #if* closed
function glslBalanced(src) {
  const code = src.replace(/\/\/[^\n]*/g, '').replace(/\/\*[\s\S]*?\*\//g, '');
  const pairs = { ')': '(', '}': '{', ']': '[' }, stack = [];
  for (const ch of code) {
    if (ch === '(' || ch === '{' || ch === '[') stack.push(ch);
    else if (ch in pairs) { if (stack.pop() !== pairs[ch]) return false; }
  }
  const opens = (code.match(/^\s*#\s*if(def|ndef)?\b/gm) || []).length, closes = (code.match(/^\s*#\s*endif\b/gm) || []).length;
  return stack.length === 0 && opens === closes;
}

// ================================================================================================= SSRPass: options
const cam = new THREE.PerspectiveCamera(60, 16 / 9, 0.1, 1000);
cam.position.set(3, 2, 10); cam.lookAt(0, 1, 0); cam.updateMatrixWorld();
const hist = new CameraHistory(cam);
const gbStub = { normal: new THREE.Texture() };
{
  const want = { scale: 0.5, maxPixels: Infinity, steps: 24, refine: 4, maxDistance: 80, thickness: 0.4, thicknessScale: 0, maxRoughness: 0.5, edgeFade: 0.1, jitter: false, metalTint: true, source: 'current' };
  check('SSR_DEFAULTS: exactly the documented keys', Object.keys(SSR_DEFAULTS).join() === Object.keys(want).join(), Object.keys(SSR_DEFAULTS).join());
  for (const k in want) check(`SSR_DEFAULTS.${k} = ${String(want[k])}`, Object.is(SSR_DEFAULTS[k], want[k]), String(SSR_DEFAULTS[k]));
  check('SSR_DEFAULTS frozen', Object.isFrozen(SSR_DEFAULTS));
  check("SSR_SOURCES = ['current'] ('history' reserved)", SSR_SOURCES.join() === 'current');

  const p = new SSRPass(cam, hist, gbStub);
  check('constructed: every option a live property at its default', Object.keys(want).every((k) => Object.is(p[k], want[k])));
  check('a Pass, needsSwap false, enabled', p instanceof Pass && p.needsSwap === false && p.enabled === true);
  check('camera / history / gbuffer kept as plain properties', p.camera === cam && p.history === hist && p.gbuffer === gbStub);
  check('unknown option keys are ignored (PostFX validates names)', !throws(() => new SSRPass(cam, hist, gbStub, { enabled: false, foo: 1 })));
  const o = new SSRPass(cam, null, null, { steps: 16, refine: 0, maxDistance: 120, thickness: 1, thicknessScale: 0.02, maxRoughness: 0.7, edgeFade: 0, jitter: true, metalTint: false, scale: 0.25, maxPixels: null });
  check('constructor options applied', o.steps === 16 && o.refine === 0 && o.maxDistance === 120 && o.thickness === 1 && o.thicknessScale === 0.02 && o.maxRoughness === 0.7 && o.edgeFade === 0 && o.jitter && !o.metalTint && o.scale === 0.25);
  check('constructor options reach uniforms / defines', o.material.uniforms.uThicknessScale.value === 0.02 && o.material.uniforms.uMaxRoughness.value === 0.7 && 'SSR_JITTER' in o.material.defines && !('SSR_METAL_TINT' in o.material.defines));
  check('maxPixels null -> Infinity', o.maxPixels === Infinity);
  check('undefined options take the default', (() => { const q = new SSRPass(cam, hist, gbStub, { steps: undefined, thickness: undefined }); return q.steps === 24 && q.thickness === 0.4; })());

  const bad = [
    ['scale', 0], ['scale', 1.5], ['scale', NaN], ['scale', '0.5'], ['maxPixels', 0], ['maxPixels', -5],
    ['steps', 0], ['steps', 129], ['steps', 2.5], ['refine', -1], ['refine', 17], ['refine', 1.5],
    ['maxDistance', 0], ['maxDistance', Infinity], ['thickness', 0], ['thickness', -1], ['thicknessScale', -0.1], ['thicknessScale', 2], ['thicknessScale', NaN],
    ['maxRoughness', 0], ['maxRoughness', 1.1], ['edgeFade', -0.1], ['edgeFade', 0.6], ['source', 'foo'],
  ];
  let named = true, listed = [];
  for (const [k, v] of bad) {
    const ok = throws(() => new SSRPass(cam, hist, gbStub, { [k]: v }), new RegExp(`^\\[SSRPass\\] ${k}: `));
    if (!ok) { named = false; listed.push(`${k}=${String(v)}`); }
  }
  check(`constructor: ${bad.length} bad values throw, the message names the option`, named, listed.join(' '));
  check("source 'history': reserved, says so", throws(() => new SSRPass(cam, hist, gbStub, { source: 'history' }), /source: 'history' is reserved/));
  check('range message shows the range and the value', throws(() => new SSRPass(cam, hist, gbStub, { steps: 0 }), /steps: expected an integer in \[1, 128\], got 0/)
    && throws(() => new SSRPass(cam, hist, gbStub, { maxRoughness: 0 }), /maxRoughness: expected a number in \(0, 1\], got 0/));
  let setterOk = true;
  for (const [k, v] of bad) {
    const before = p[k];
    setterOk &&= throws(() => { p[k] = v; }, new RegExp(`^\\[SSRPass\\] ${k}: `)) && Object.is(p[k], before);
  }
  check('setters: the same bad values throw and keep the old value', setterOk);
}

// ================================================================================================= SSRPass: defines, sizing, target
{
  const p = new SSRPass(cam, hist, gbStub);
  const m = p.material, d = m.defines;
  check('defines at defaults: SSR_STEPS 24, SSR_REFINE 4, SSR_METAL_TINT (no SSR_JITTER)', JSON.stringify(d) === '{"SSR_STEPS":24,"SSR_REFINE":4,"SSR_METAL_TINT":1}', JSON.stringify(d));
  let v = m.version;
  p.steps = 24; p.refine = 4; p.jitter = false; p.metalTint = true;
  check('same values: no program change (material.version kept)', m.version === v);
  p.steps = 16;
  check('steps = 16 -> SSR_STEPS 16, one program change', d.SSR_STEPS === 16 && m.version === v + 1);
  v = m.version; p.refine = 0;
  check('refine = 0 -> SSR_REFINE 0', d.SSR_REFINE === 0 && m.version === v + 1);
  p.jitter = true; p.metalTint = false;
  check('jitter true -> SSR_JITTER, metalTint false -> SSR_METAL_TINT removed', 'SSR_JITTER' in d && !('SSR_METAL_TINT' in d));
  v = m.version; p.maxDistance = 50; p.thickness = 0.8; p.thicknessScale = 0.01; p.maxRoughness = 0.4; p.edgeFade = 0.2;
  check('uniform options: no program change, uniforms written', m.version === v && m.uniforms.uThicknessScale.value === 0.01 && m.uniforms.uThickness.value === 0.8);
  p.jitter = false; p.metalTint = true; p.steps = 24; p.refine = 4;
  check('back to defaults: the default defines again', JSON.stringify(d) === '{"SSR_STEPS":24,"SSR_REFINE":4,"SSR_METAL_TINT":1}');

  const tex = p.texture, rt = p._rt;
  check('output: RGBA16F (HalfFloatType, RGBAFormat)', tex.type === THREE.HalfFloatType && tex.format === THREE.RGBAFormat);
  check('output: LinearFilter, no mipmaps', tex.minFilter === THREE.LinearFilter && tex.magFilter === THREE.LinearFilter && tex.generateMipmaps === false);
  check('output: no depth / stencil buffer', rt.depthBuffer === false && rt.stencilBuffer === false);
  const size = () => [tex.image.width, tex.image.height];
  p.setSize(3024, 1964);
  check('setSize 3024x1964, scale 0.5 -> 1512x982', size().join() === '1512,982', size().join());
  check('uRes = the full drawing-buffer size', p.material.uniforms.uRes.value.x === 3024 && p.material.uniforms.uRes.value.y === 1964);
  p.scale = 0.25;
  check('scale = 0.25 resizes at once -> 756x491', size().join() === '756,491', size().join());
  p.scale = 0.5; p.maxPixels = 0.6e6;
  const [bw, bh] = size();
  check('maxPixels 0.6e6: within the budget, aspect kept', bw * bh <= 0.6e6 && Math.abs(bw / bh - 3024 / 1964) < 0.01 && bw * bh > 0.58e6, `${bw}x${bh}`);
  p.maxPixels = Infinity;
  check('maxPixels Infinity -> 1512x982 again', size().join() === '1512,982');
  p.setSize(2560, 1440);
  check('setSize 2560x1440 -> 1280x720', size().join() === '1280,720');
  p.setSize(3, 3); p.scale = 0.1;
  check('minimum 1 x 1', size().join() === '1,1');
  p.setSize(1, 1);
  check('1 x 1 drawing buffer -> 1 x 1', size().join() === '1,1');
  check('texture identity kept across resizes', p.texture === tex);
  check('reset() is a no-op that exists', typeof p.reset === 'function' && !throws(() => p.reset()));
}

// ================================================================================================= SSRPass: shader contract
{
  const p = new SSRPass(cam, hist, gbStub), fs = p.material.fragmentShader, u = p.material.uniforms;
  const uniforms = ['tColor', 'tDepth', 'tNormal', 'uProj', 'uProjInv', 'uRes', 'uNear', 'uFar', 'uMaxDistance', 'uThickness', 'uThicknessScale', 'uMaxRoughness', 'uEdgeFade', 'uFrame'];
  check('material uniforms = the documented list', Object.keys(u).sort().join() === [...uniforms].sort().join(), Object.keys(u).join());
  check('every documented uniform is declared in the shader', uniforms.every((n) => new RegExp(`uniform [^;]*\\b${n}\\b`).test(fs)));
  check('declarations: highp depth sampler, int uFrame, mat4 projections', /uniform highp sampler2D tDepth;/.test(fs) && /uniform int uFrame;/.test(fs) && /uniform mat4 uProj;/.test(fs) && /uniform mat4 uProjInv;/.test(fs));
  check('G-buffer decode pasted (engineOctDecode / engineGbMetal / engineGbFlags)', fs.includes(GBUFFER_GLSL_DECODE) && /engineOctDecode\( gb\.rg \)/.test(fs) && /engineGbMetal\( gb\.a \)/.test(fs) && /engineGbFlags\( gb\.a \)/.test(fs));
  check('rejects pixels without wroteGBuffer / hasNormal, or too rough', /\( flags & ENGINE_GB_WROTE \) == 0 \|\| \( flags & ENGINE_GB_HAS_NORMAL \) == 0 \|\| rough >= uMaxRoughness/.test(fs));
  check('march loop over SSR_STEPS, bisection over SSR_REFINE', /for \( int i = 0; i < SSR_STEPS; i \+\+ \)/.test(fs) && /for \( int i = 0; i < SSR_REFINE; i \+\+ \)/.test(fs));
  check('jitter and metal tint under their defines', /#ifdef SSR_JITTER[\s\S]*uFrame & 63[\s\S]*#else\s+float o = 0\.5;/.test(fs) && /#ifdef SSR_METAL_TINT/.test(fs));
  check('hit test: behind the surface at the sample, span reaching into the band', fs.includes('if ( sd < 1.0 && zRay > zScene && min( zRay, zPrev ) < zScene + band( zScene ) ) { uHit = u; zHit = zScene; break; }') && !/max\( zRay, zPrev \)/.test(fs));
  check('band = thickness + depth * thicknessScale, used by the start check too', fs.includes('float band( float z ) { return uThickness + z * uThicknessScale; }') && fs.includes('zPrev < zStart + band( zStart )'));
  check('bisection: band of its surface, or not nearer than the hit surface', fs.includes('if ( zr > zs && ( zr < zs + band( zs ) || zs > zHit - band( zHit ) ) ) { b = m; zHit = zs; } else a = m;'));
  check('metalTint stopgap: own hue, at most own brightness', fs.includes('c *= mix( vec3( 1.0 ), own / max( maxC, 1e-3 ) * min( 1.0, maxC / max( max( c.r, max( c.g, c.b ) ), 1e-3 ) ), metal );'));
  check('perspective-correct: 1 / w interpolated, both ends projected once', /float k0 = 1\.0 \/ H0\.w, k1 = 1\.0 \/ H1\.w;/.test(fs) && /float zRay = 1\.0 \/ mix\( k0, k1, u \);/.test(fs) && (fs.match(/uProj \*/g) || []).length === 2);
  check('premultiplied output (radiance * w, w), zero by default', /gl_FragColor = vec4\( max\( c, 0\.0 \) \* w, w \);/.test(fs) && /gl_FragColor = vec4\( 0\.0 \);/.test(fs));
  check('NaN / Inf scene colour guarded', /! all\( lessThan\( abs\( c \), vec3\( 65000\.0 \) \) \)/.test(fs));
  check('shader text: delimiters balanced, #if / #endif matched', glslBalanced(fs));
  check('no GLSL3 qualifiers that ShaderMaterial would reject (no #version, no out vec4)', !/#version/.test(fs) && !/\bout\s+(highp\s+)?vec4\b/.test(fs));
  check('SSR_COMPOSITE_GLSL verbatim', SSR_COMPOSITE_GLSL === 'vec3 engineSsrComposite( vec3 col, vec4 ssr ) { return col * ( 1.0 - ssr.a ) + ssr.rgb; }');
  check('SSR_COMPOSITE_GLSL balanced', glslBalanced(SSR_COMPOSITE_GLSL));
  // the composite as maths: a = 0 is the identity, a = 1 is the reflection
  const comp = (col, s) => col.map((c, i) => c * (1 - s[3]) + s[i]);
  check('composite: weight 0 leaves the colour, weight 1 replaces it', comp([1, 2, 3], [0, 0, 0, 0]).join() === '1,2,3' && comp([1, 2, 3], [4, 5, 6, 1]).join() === '4,5,6');
}

// ================================================================================================= SSRPass: render() on a stub
{
  const r = stubRenderer();
  const p = new SSRPass(cam, hist, gbStub);
  p.setSize(64, 32);
  const read = new THREE.WebGLRenderTarget(64, 32, { depthTexture: new THREE.DepthTexture(64, 32) });
  hist.begin(64, 32); // jittered projection, as during PostFX.render
  p.edgeFade = 0;
  p.render(r, null, read);
  const u = p.material.uniforms;
  check('render: tColor / tDepth = readBuffer texture / depthTexture', u.tColor.value === read.texture && u.tDepth.value === read.depthTexture);
  check('render: tNormal = gbuffer.normal', u.tNormal.value === gbStub.normal);
  check('render: uProj / uProjInv = the (jittered) camera projection, exact', sameBits(u.uProj.value, cam.projectionMatrix) && sameBits(u.uProjInv.value, cam.projectionMatrixInverse) && hist.uniforms.uJitter.value.lengthSq() > 0);
  check('render: uNear / uFar from the camera', u.uNear.value === cam.near && u.uFar.value === cam.far);
  check('render: uFrame = history.uniforms.uFrame', u.uFrame.value === hist.uniforms.uFrame.value);
  check('render: edgeFade 0 -> uEdgeFade 1e-4 (no undefined smoothstep)', u.uEdgeFade.value === 1e-4);
  p.maxDistance = 33; p.thickness = 0.7; p.thicknessScale = 0.005; p.maxRoughness = 0.55; p.edgeFade = 0.15;
  p.render(r, null, read);
  check('render: live options -> uniforms', u.uMaxDistance.value === 33 && u.uThickness.value === 0.7 && u.uThicknessScale.value === 0.005 && u.uMaxRoughness.value === 0.55 && u.uEdgeFade.value === 0.15);
  check('render: draws into its own target', r.bound === p._rt);
  hist.end();
  hist.begin(64, 32); p.render(r, null, read); hist.end();
  check('render: uFrame follows the history', u.uFrame.value === hist.uniforms.uFrame.value && u.uFrame.value > 0);
  check('history null: uFrame 0', (() => { const q = new SSRPass(cam, null, gbStub); q.render(r, null, read); return q.material.uniforms.uFrame.value === 0; })());

  // no G-buffer / no depth: the target is cleared to (0, 0, 0, 0), the renderer's clear colour restored
  const q = new SSRPass(cam, hist, null), clears = r.clears;
  r.setClearColor(new THREE.Color(0.2, 0.3, 0.4), 1);
  q.render(r, null, read);
  check('gbuffer null: clears its target to (0, 0, 0, 0)', r.clears === clears + 1 && r.bound === q._rt && r.clearColorAtClear.join() === '0,0,0,0');
  check('gbuffer null: clear colour / alpha restored', near(r.clearColor.r, 0.2, 1e-6) && near(r.clearColor.b, 0.4, 1e-6) && r.clearAlphaSet === 1);
  const noDepth = new THREE.WebGLRenderTarget(64, 32);
  p.render(r, null, noDepth);
  check('no depth texture: cleared, not traced', r.clears === clears + 2 && r.bound === p._rt);

  // nothing allocates per render
  gate(inRange, note, 'SSR render()', bytesPerCall((n) => { for (let i = 0; i < n; i++) p.render(r, null, read); }, 50000), { unit: 'frame' });
  check('dispose: no throw', !throws(() => { p.dispose(); q.dispose(); }));
}

// ================================================================================================= SSR trace: CPU port
// The fragment shader's trace in JS, step for step (same names), so its maths can be checked against geometry. The
// review found two bugs that an earlier port mirrored line for line, so the cases below are built from the geometry
// (an expected result computed independently), and `legacy` re-runs the 0.8.0-draft predicates to show that each
// case does reproduce the bug it guards against.
const smoothstep = (a, b, x) => { const t = Math.min(Math.max((x - a) / (b - a), 0), 1); return t * t * (3 - 2 * t); };
const mix = (a, b, t) => a + (b - a) * t;
function envBRDF(NdotV, roughness) {
  const r = [roughness * -1 + 1, roughness * -0.0275 + 0.0425, roughness * -0.572 + 1.04, roughness * 0.022 - 0.04];
  const a004 = Math.min(r[0] * r[0], 2 ** (-9.28 * NdotV)) * r[0] + r[1];
  return [-1.04 * a004 + r[2], 1.04 * a004 + r[3]];
}
// the metalTint stopgap, as in the shader: hue of `own`, at most its brightness
function metalTint(c, own, metal) {
  const maxC = Math.max(...own), maxR = Math.max(Math.max(...c), 1e-3);
  return c.map((v, i) => v * mix(1, own[i] / Math.max(maxC, 1e-3) * Math.min(1, maxC / maxR), metal));
}
function ssrTrace(sc, ix, iy, p, legacy = {}) {
  const { W, H, near: n, far: f } = sc;
  const linearDepth = (d) => n * f / (f - (f - n) * d);
  const depthAt = (x, y) => sc.depth[Math.trunc(y) * W + Math.trunc(x)];
  const band = (z) => p.thickness + z * p.thicknessScale;
  const i = iy * W + ix;
  if (!sc.wrote[i] || sc.rough[i] >= p.maxRoughness) return { w: 0, why: 'rejected' };
  const d = sc.depth[i];
  if (d >= 1) return { w: 0, why: 'sky' };
  const hp = new THREE.Vector4((ix + 0.5) / W * 2 - 1, (iy + 0.5) / H * 2 - 1, d * 2 - 1, 1).applyMatrix4(sc.projInv);
  const P = new THREE.Vector3(hp.x / hp.w, hp.y / hp.w, hp.z / hp.w);
  const V = P.clone().negate().normalize(), N = sc.normal[i];
  const NdotV = N.dot(V);
  const R = V.clone().negate().reflect(N); // reflect(-V, N)
  const fab = envBRDF(Math.min(Math.max(NdotV, 0), 1), sc.rough[i]);
  const fr = Math.min(Math.max((p.maxRoughness - sc.rough[i]) / 0.2, 0), 1);
  const share = Math.min(Math.max(mix(0.04, 1, sc.metal[i]) * fab[0] + fab[1], 0), 1);
  let w = share * fr * fr * (3 - 2 * fr) * (1 - smoothstep(0, 0.25, R.z));
  if (NdotV <= 0 || w <= 0) return { w: 0, why: 'weight', R };
  const len = R.z > 0 ? Math.min(p.maxDistance, (-n - P.z) / R.z) : p.maxDistance;
  const H0 = new THREE.Vector4(P.x, P.y, P.z, 1).applyMatrix4(sc.proj);
  const E = P.clone().addScaledVector(R, len), H1 = new THREE.Vector4(E.x, E.y, E.z, 1).applyMatrix4(sc.proj);
  const k0 = 1 / H0.w, k1 = 1 / H1.w;
  const S0 = [(H0.x * k0 * 0.5 + 0.5) * W, (H0.y * k0 * 0.5 + 0.5) * H], S1 = [(H1.x * k1 * 0.5 + 0.5) * W, (H1.y * k1 * 0.5 + 0.5) * H];
  const dS = [S1[0] - S0[0], S1[1] - S0[1]];
  const lim = [dS[0] >= 0 ? W - 0.5 : 0.5, dS[1] >= 0 ? H - 0.5 : 0.5];
  const te = [Math.abs(dS[0]) > 1e-4 ? (lim[0] - S0[0]) / dS[0] : 1, Math.abs(dS[1]) > 1e-4 ? (lim[1] - S0[1]) / dS[1] : 1];
  const uEnd = Math.min(1, te[0], te[1]);
  const u0 = 2 / Math.max(Math.hypot(dS[0], dS[1]), 1e-4);
  if (uEnd <= u0) return { w: 0, why: 'span', R };
  const at = (u) => [mix(S0[0], S1[0], u), mix(S0[1], S1[1], u)];
  let zPrev = 1 / mix(k0, k1, u0);
  const zStart = linearDepth(depthAt(...at(u0)));
  if (zPrev > zStart && zPrev < zStart + band(zStart)) return { w: 0, why: 'start-inside', R };
  const o = 0.5, du = (uEnd - u0) / p.steps;
  let uPrev = u0, uHit = -1, zHit = 0;
  for (let s = 0; s < p.steps; s++) {
    const u = u0 + (s + o) * du, zRay = 1 / mix(k0, k1, u), sd = depthAt(...at(u)), zScene = linearDepth(sd);
    const behind = legacy.hit ? Math.max(zRay, zPrev) > zScene : zRay > zScene;
    if (sd < 1 && behind && Math.min(zRay, zPrev) < zScene + band(zScene)) { uHit = u; zHit = zScene; break; }
    uPrev = u; zPrev = zRay;
  }
  if (uHit < 0) return { w: 0, why: 'miss', R };
  let a = uPrev, b = uHit;
  for (let s = 0; s < p.refine; s++) {
    const m = 0.5 * (a + b);
    const zs = linearDepth(depthAt(...at(m))), zr = 1 / mix(k0, k1, m);
    const hitSide = legacy.bisect === 'band' ? zr > zs && zr < zs + band(zs) // the review's band-only predicate
      : legacy.bisect ? zr > zs // the draft
        : zr > zs && (zr < zs + band(zs) || zs > zHit - band(zHit));
    if (hitSide) { b = m; zHit = zs; } else a = m;
  }
  const hitPx = at(b), hitUv = [hitPx[0] / W, hitPx[1] / H];
  const dist = len * b * k1 / mix(k0, k1, b);
  const e = Math.min(hitUv[0], 1 - hitUv[0], hitUv[1], 1 - hitUv[1]);
  w *= smoothstep(0, p.edgeFade, e) * (1 - smoothstep(0.7, 1, dist / p.maxDistance));
  return { w, why: 'hit', hitPx, hitUv, hitZ: linearDepth(depthAt(...hitPx)), dist, P, R, share };
}

// Analytic scene: floor y = 0, a 6 m wall at z = -30 (world), plus optional rectangles { c, n, u, v, hu, hv } (centre,
// normal, in-plane axes, half extents), ray-cast at every texel centre; the camera 1.5 m up looking slightly down.
function buildScene(W, H, camera, rects = []) {
  camera.updateMatrixWorld(); camera.updateProjectionMatrix();
  const n = camera.near, f = camera.far, N = W * H;
  const sc = { W, H, near: n, far: f, proj: camera.projectionMatrix.clone(), projInv: camera.projectionMatrixInverse.clone(),
    depth: new Float64Array(N), normal: new Array(N), wrote: new Uint8Array(N), rough: new Float64Array(N), metal: new Float64Array(N),
    world: new Array(N), id: new Int8Array(N) };
  const o = camera.position, dir = new THREE.Vector3(), pt = new THREE.Vector3(), q = new THREE.Vector3();
  const toView = (v) => v.clone().transformDirection(camera.matrixWorldInverse);
  const floorN = toView(new THREE.Vector3(0, 1, 0)), wallN = toView(new THREE.Vector3(0, 0, 1));
  const rectN = rects.map((r) => toView(r.n));
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = y * W + x;
    dir.set((x + 0.5) / W * 2 - 1, (y + 0.5) / H * 2 - 1, 0.5).unproject(camera).sub(o).normalize();
    let t = Infinity, nrm = null, id = -1;
    if (dir.y < 0) { t = -o.y / dir.y; nrm = floorN; id = 0; }
    if (dir.z < 0) { const tw = (-30 - o.z) / dir.z, yw = o.y + dir.y * tw; if (tw < t && yw >= 0 && yw <= 6) { t = tw; nrm = wallN; id = 1; } }
    rects.forEach((r, k) => {
      const dn = dir.dot(r.n);
      if (Math.abs(dn) < 1e-9) return;
      const tr = q.copy(r.c).sub(o).dot(r.n) / dn;
      if (!(tr > 0 && tr < t)) return;
      q.copy(o).addScaledVector(dir, tr).sub(r.c);
      if (Math.abs(q.dot(r.u)) <= r.hu && Math.abs(q.dot(r.v)) <= r.hv) { t = tr; nrm = dn < 0 ? rectN[k] : rectN[k].clone().negate(); id = 2 + k; }
    });
    if (t === Infinity) { sc.depth[i] = 1; sc.normal[i] = floorN; continue; }
    pt.copy(o).addScaledVector(dir, t);
    sc.world[i] = pt.clone();
    const z = -pt.clone().applyMatrix4(camera.matrixWorldInverse).z; // linear view depth
    sc.depth[i] = f * (1 - n / z) / (f - n); // standard depth, the inverse of linearDepth()
    sc.normal[i] = nrm; sc.wrote[i] = 1; sc.rough[i] = 0.05; sc.metal[i] = 0; sc.id[i] = id;
  }
  return sc;
}

{
  // perspective-correct interpolation: at any u the screen point and 1 / mix(k0, k1, u) are those of the ray point at
  // t = len * u * k1 / mix(k0, k1, u)
  const P = new THREE.Vector3(1.2, -1.4, -9), R = new THREE.Vector3(0.3, 0.5, -0.8).normalize(), len = 80;
  const proj = cam.projectionMatrix, H0 = new THREE.Vector4(P.x, P.y, P.z, 1).applyMatrix4(proj);
  const E = P.clone().addScaledVector(R, len), H1 = new THREE.Vector4(E.x, E.y, E.z, 1).applyMatrix4(proj);
  const k0 = 1 / H0.w, k1 = 1 / H1.w;
  let worst = 0, worstPx = 0;
  for (const u of [0, 0.1, 0.37, 0.5, 0.81, 1]) {
    const t = len * u * k1 / mix(k0, k1, u), X = P.clone().addScaledVector(R, t);
    const hx = new THREE.Vector4(X.x, X.y, X.z, 1).applyMatrix4(proj);
    worst = Math.max(worst, Math.abs(1 / mix(k0, k1, u) - -X.z) / -X.z);
    worstPx = Math.max(worstPx, Math.abs(hx.x / hx.w - mix(H0.x * k0, H1.x * k1, u)), Math.abs(hx.y / hx.w - mix(H0.y * k0, H1.y * k1, u)));
  }
  check(`1 / mix(k0, k1, u) = the ray's linear depth (rel. error ${worst.toExponential(1)})`, worst < 1e-12);
  check(`screen position affine in u, distance t = len u k1 / mix(k0, k1, u) (NDC error ${worstPx.toExponential(1)})`, worstPx < 1e-12);

  const W = 320, H = 180, c = new THREE.PerspectiveCamera(60, W / H, 0.1, 1000);
  c.position.set(0, 1.5, 0); c.lookAt(0, 0.5, -10);
  const sc = buildScene(W, H, c);
  const p = { ...SSR_DEFAULTS };
  const pixelOf = (X) => { const v = X.clone().project(c); return [(v.x * 0.5 + 0.5) * W, (v.y * 0.5 + 0.5) * H]; };
  const camMirror = new THREE.Vector3(c.position.x, -c.position.y, c.position.z);
  const wallHitOf = (F) => { const d = F.clone().sub(camMirror); return camMirror.clone().addScaledVector(d, (-30 - camMirror.z) / d.z); };
  // floor point 10 m ahead: its mirror ray meets the wall at y = 3 (x = 0)
  const [fx, fy] = pixelOf(new THREE.Vector3(0, 0, -10)).map(Math.floor);
  const F = sc.world[fy * W + fx], X = wallHitOf(F), want = pixelOf(X);
  const r1 = ssrTrace(sc, fx, fy, p);
  const err = r1.hitPx ? Math.hypot(r1.hitPx[0] - want[0], r1.hitPx[1] - want[1]) : Infinity;
  check(`floor 10 m ahead reflects the wall at the mirrored point (${err.toFixed(2)} px off at ${W}x${H})`, r1.why === 'hit' && err < 1.5, `${r1.why} ${r1.hitPx?.map((v) => v.toFixed(1))} vs ${want.map((v) => v.toFixed(1))}`);
  check(`hit distance along the ray = |wall point - floor point| (${r1.dist?.toFixed(2)} m vs ${X.distanceTo(F).toFixed(2)})`, r1.why === 'hit' && Math.abs(r1.dist - X.distanceTo(F)) < 0.5);
  check(`hit weight = specular share * fades, inside (0, share] (${r1.w.toFixed(3)}, share ${r1.share?.toFixed(3)})`, r1.w > 0 && r1.w <= r1.share + 1e-12);
  // why the bisection also accepts a midpoint behind the hit's own (thick) surface: with the band alone, a stride that
  // crossed the wall by more than the band never refines and reads the colour a stride past the crossing
  const rb = ssrTrace(sc, fx, fy, p, { bisect: 'band' }), eb = Math.hypot(rb.hitPx[0] - want[0], rb.hitPx[1] - want[1]);
  check(`band-only bisection would stop a stride past the wall crossing (${eb.toFixed(2)} px, ${rb.dist.toFixed(1)} m vs ${X.distanceTo(F).toFixed(1)})`, rb.why === 'hit' && rb.dist - X.distanceTo(F) > 2 && err < eb);
  // more floor points: every hit lands within 1.5 px of the mirrored wall point
  let allNear = true, nHits = 0;
  for (const [x, z] of [[-3, -8], [2, -12], [4, -16], [-6, -20], [1, -24]]) {
    const [px, py] = pixelOf(new THREE.Vector3(x, 0, z)).map(Math.floor);
    const Fp = sc.world[py * W + px], r = ssrTrace(sc, px, py, p);
    if (r.why !== 'hit') { allNear = false; continue; }
    nHits++;
    const wp = pixelOf(wallHitOf(Fp));
    allNear &&= Math.hypot(r.hitPx[0] - wp[0], r.hitPx[1] - wp[1]) < 1.5;
  }
  check(`5 more floor points hit the wall at their mirrored points (${nHits} / 5)`, allNear && nHits === 5);
  // floor point 5 m ahead: the mirror ray passes over the 6 m wall into the sky -> miss (environment stays)
  const [sx, sy] = pixelOf(new THREE.Vector3(0, 0, -5)).map(Math.floor);
  const r2 = ssrTrace(sc, sx, sy, p);
  check(`ray over the wall into the sky: miss, weight 0 (${r2.why})`, r2.w === 0 && (r2.why === 'miss' || r2.why === 'span'));
  // the same floor pixel with maxDistance 10: the ray ends before the wall -> miss
  check('maxDistance shorter than the way to the wall: miss', ssrTrace(sc, fx, fy, { ...p, maxDistance: 10 }).w === 0);
  // a wall pixel reflects back toward the camera (R.z > 0.25): no trace
  const [wx, wy] = pixelOf(new THREE.Vector3(0, 3, -30)).map(Math.floor);
  check('wall facing the camera: toward-camera ray, weight 0', ssrTrace(sc, wx, wy, p).why === 'weight');
  // sky / non-writer / too rough
  check('sky pixel: rejected', ssrTrace(sc, 0, H - 1, p).w === 0);
  const ri = fy * W + fx;
  sc.rough[ri] = 0.5;
  check('roughness = maxRoughness (0.5): rejected before the trace', ssrTrace(sc, fx, fy, p).why === 'rejected');
  sc.rough[ri] = 0.4;
  const rFade = ssrTrace(sc, fx, fy, p);
  check(`roughness 0.4 (inside the 0.2 fade): weaker than at 0.05 (${rFade.w.toFixed(4)} < ${r1.w.toFixed(4)})`, rFade.why === 'hit' && rFade.w > 0 && rFade.w < r1.w);
  sc.rough[ri] = 0.3;
  check('roughness 0.3: full roughness weight (fade starts above it)', (() => { const r = ssrTrace(sc, fx, fy, p); return r.why === 'hit' && near(r.w / r.share, r1.w / r1.share, 1e-9); })());
  sc.rough[ri] = 0.05;
  sc.wrote[ri] = 0;
  check('no wroteGBuffer: rejected', ssrTrace(sc, fx, fy, p).why === 'rejected');
  sc.wrote[ri] = 1;
  // a normal-mapped normal tilted 4.5 deg away from the camera (the view ray meets the floor at 8.5 deg): N.V > 0, but
  // the mirror ray leaves 0.5 deg below the floor it starts on: inside the default 0.4 m band at the march start
  const n0 = sc.normal[ri];
  sc.normal[ri] = new THREE.Vector3(0, Math.cos(4.5 * Math.PI / 180), -Math.sin(4.5 * Math.PI / 180)).transformDirection(c.matrixWorldInverse);
  const r3 = ssrTrace(sc, fx, fy, p);
  check(`normal-mapped self-hit at the default thickness: the march starts inside the floor -> miss (${r3.why})`, r3.why === 'start-inside' && r3.w === 0);
  sc.normal[ri] = n0;
  // thicknessScale widens the band with depth: a ray passing 1.5 m behind the far wall's edge region is no hit at 0.4 m
  check('thicknessScale 0 = the fixed band (same result as before the knob)', (() => { const r = ssrTrace(sc, fx, fy, { ...p, thicknessScale: 0 }); return r.why === 'hit' && r.hitPx[0] === r1.hitPx[0] && r.hitPx[1] === r1.hitPx[1]; })());

  // MUST-FIX 1: a ray whose depth decreases (R.z > 0) passing just IN FRONT of a surface is no hit. Floor pixel 10 m
  // ahead, normal tilted 41 deg toward the camera (R nearly straight up, view-space R.z ~0.1, full weight), and a panel
  // containing the ray's direction set 1 cm behind it (farther from the camera) between 1.5 m and 4.5 m up the ray.
  // The ray starts at the floor point of the texel centre (what the trace reconstructs), not at (0, 0, -10).
  {
    const tilt = 41 * Math.PI / 180, Nw = new THREE.Vector3(0, Math.cos(tilt), Math.sin(tilt));
    const Fw = sc.world[fy * W + fx], Iw = Fw.clone().sub(c.position).normalize(), Rw = Iw.clone().reflect(Nw);
    const across = new THREE.Vector3(1, 0, 0), pn = new THREE.Vector3().crossVectors(Rw, across).normalize();
    if (pn.dot(c.position.clone().sub(Fw)) < 0) pn.negate(); // panel normal toward the camera
    const panel = { c: Fw.clone().addScaledVector(Rw, 3).addScaledVector(pn, -0.01), n: pn, u: Rw.clone(), v: across, hu: 1.5, hv: 1 };
    const s2 = buildScene(W, H, c, [panel]);
    const qx = fx, qy = fy, qi = qy * W + qx;
    s2.normal[qi] = Nw.clone().transformDirection(c.matrixWorldInverse);
    const now = ssrTrace(s2, qx, qy, p), old = ssrTrace(s2, qx, qy, p, { hit: true });
    check(`R.z > 0 case set up: R.z ${now.R.z.toFixed(3)} in (0, 0.25), the panel on the ray's screen path`, now.R.z > 0 && now.R.z < 0.25 && s2.id.some((v) => v === 2));
    check(`R.z > 0, ray 1 cm in front of a panel: the draft predicate hit it (${old.why}), the fix misses (${now.why})`, old.why === 'hit' && s2.id[Math.trunc(old.hitPx[1]) * W + Math.trunc(old.hitPx[0])] === 2 && now.why === 'miss' && now.w === 0);
  }

  // MUST-FIX 2: the ray passes behind a floating sign (by metres) on its way to the wall: the last miss sample is on the
  // sign, and the bisection must not collapse onto it. Sign at z = -12 (the ray passes it 1.8 m lower: no contact),
  // covering the screen row of the wall hit, so the last miss sample (o = 0.5) is on the sign and the hit sample on
  // the wall one row up. The wall point itself is hidden behind the sign: the right answer is the wall next to it.
  {
    const yAtRow = (py) => { // world y where the sight line through (column of the hit, row py) meets z = -12
      const d = new THREE.Vector3(want[0] / W * 2 - 1, py / H * 2 - 1, 0.5).unproject(c).sub(c.position);
      return c.position.y + d.y * (-12 - c.position.z) / d.z;
    };
    const topY = yAtRow(Math.floor(want[1]) + 0.95); // the sign covers the hit's texel row, not the next
    const sign = { c: new THREE.Vector3(0, (1.0 + topY) / 2, -12), n: new THREE.Vector3(0, 0, 1), u: new THREE.Vector3(1, 0, 0), v: new THREE.Vector3(0, 1, 0), hu: 0.6, hv: (topY - 1.0) / 2 };
    const s3 = buildScene(W, H, c, [sign]);
    const now = ssrTrace(s3, fx, fy, p), old = ssrTrace(s3, fx, fy, p, { bisect: true });
    const idAt = (r) => s3.id[Math.trunc(r.hitPx[1]) * W + Math.trunc(r.hitPx[0])];
    const e2 = now.hitPx ? Math.hypot(now.hitPx[0] - want[0], now.hitPx[1] - want[1]) : Infinity;
    check(`occluder then wall: the draft bisection read the sign (id ${old.hitPx ? idAt(old) : '-'}, ${old.hitZ?.toFixed(1)} m)`, old.why === 'hit' && idAt(old) === 2);
    check(`occluder then wall: the fix lands on the wall next to the mirrored point (${e2.toFixed(2)} px, ${now.hitZ?.toFixed(1)} m)`, now.why === 'hit' && idAt(now) === 1 && e2 < 1.5);
  }

  check('metal at roughness 0: specular share ~1 (envBRDF)', (() => { const f = envBRDF(0.8, 0); return f[0] + f[1] > 0.97; })());
  check('dielectric at roughness 0, N.V = 1: share ~0.04 (Schlick F0)', (() => { const f = envBRDF(1, 0); const s = 0.04 * f[0] + f[1]; return s > 0.039 && s < 0.05; })());
  check('dielectric at roughness 0, grazing: share ~1 (Schlick)', (() => { const f = envBRDF(0, 0); return 0.04 * f[0] + f[1] > 0.95; })());
  check('rough dielectric (0.6, N.V 0.5): little share (< 0.1)', (() => { const f = envBRDF(0.5, 0.6); return 0.04 * f[0] + f[1] < 0.1; })());

  // MUST-FIX 3: the metalTint stopgap (the shader's expression): a dark navy metal (the gantry pillar) reflecting bright
  // sand keeps its own hue and at most its own brightness; a black metal reflects nothing; metalness 0 is untouched
  const sand = [2.4, 2.0, 1.4], navy = [0.012, 0.013, 0.016];
  const tinted = metalTint(sand, navy, 1);
  check(`metalTint: navy metal over sand -> no brighter than the pixel (${Math.max(...tinted).toFixed(4)} <= ${Math.max(...navy)})`, Math.max(...tinted) <= Math.max(...navy) + 1e-12);
  check('metalTint: radiance times the pixel\'s hue (component ratios of sand * navy)', near(tinted[0] / tinted[2], sand[0] * navy[0] / (sand[2] * navy[2]), 1e-9) && near(tinted[1] / tinted[2], sand[1] * navy[1] / (sand[2] * navy[2]), 1e-9));
  check('metalTint: a darker reflection keeps its own brightness (scaled by hue only)', (() => { const t = metalTint([0.005, 0.005, 0.005], navy, 1); return near(Math.max(...t), 0.005, 1e-12); })());
  check('metalTint: black metal in deep shadow reflects nothing', metalTint(sand, [0, 0, 0], 1).every((v) => v === 0));
  check('metalTint: metalness 0 leaves the radiance', metalTint(sand, navy, 0).every((v, i) => v === sand[i]));
}

// ================================================================================================= AOPass normals
// sha256 of the GTAO material's fragmentShader and JSON.stringify(defines) (insertion order: three's program cache key
// iterates the defines) of the committed 0.7.0 AOPass with default options, taken before the normals option existed.
const AO_FRAG_SHA = 'f63baa34b925c3ff172b0cc34902ebe21ba208165e6afb77af7b405846be9768';
const AO_DEFINES_SHA = '137fff3a94eb4cc7bfe6c9a08d7db6bb6651272d1952187bb619ac450b8879b2';
{
  const warns = [], warn = console.warn;
  console.warn = (...a) => warns.push(a.join(' '));
  try {
    const ao = new AOPass(cam);
    const m = ao.material, fragSha = () => sha(m.fragmentShader), defSha = () => sha(JSON.stringify(m.defines));
    check("AO default: fragmentShader byte-identical to 0.7.0 (sha256 pinned)", fragSha() === AO_FRAG_SHA, fragSha());
    check('AO default: defines identical to 0.7.0 (sha256 pinned, insertion order)', defSha() === AO_DEFINES_SHA, JSON.stringify(m.defines));
    check("AO default: normals 'depth', gbuffer null, NORMAL_VECTOR_TYPE 0, no AO_GBUFFER_NORMAL", ao.normals === 'depth' && ao.gbuffer === null && m.defines.NORMAL_VECTOR_TYPE === 0 && !('AO_GBUFFER_NORMAL' in m.defines));
    check("AO_NORMALS = ['depth', 'gbuffer']", AO_NORMALS.join() === 'depth,gbuffer');
    check('AO: no warning at construction', warns.length === 0, warns.join(' | '));
    check('AO: unknown normals throws, names the option', throws(() => new AOPass(cam, { normals: 'mesh' }), /\[AOPass\] normals: unknown value 'mesh'/));
    check('AO: setter validates, keeps the value', throws(() => { ao.normals = 'normal'; }, /normals: unknown value/) && ao.normals === 'depth');

    // 'gbuffer' requested, no G-buffer: still the 0.7.0 program, one warning on the first render
    const r = stubRenderer();
    const read = new THREE.WebGLRenderTarget(64, 32, { depthTexture: new THREE.DepthTexture(64, 32) });
    const g = new AOPass(cam, { normals: 'gbuffer' });
    const gm = g.material, v0 = gm.version;
    check("normals 'gbuffer' without a G-buffer: depth program (text + defines pinned)", sha(gm.fragmentShader) === AO_FRAG_SHA && sha(JSON.stringify(gm.defines)) === AO_DEFINES_SHA && g.normals === 'gbuffer');
    g.render(r, null, read); g.render(r, null, read); g.render(r, null, read);
    const fallbackWarns = warns.filter((w) => /normals 'gbuffer' without a G-buffer/.test(w));
    check("fallback: one console.warn over three renders", fallbackWarns.length === 1, warns.join(' | '));
    check('fallback: no program change while waiting', gm.version === v0 && gm.uniforms.tNormal.value === null);

    // G-buffer set: define + patched text, one program change; renders bind the normal texture without more changes
    const gb = { normal: new THREE.Texture() };
    g.gbuffer = gb;
    check('gbuffer set: AO_GBUFFER_NORMAL defined, NORMAL_VECTOR_TYPE still 0', gm.defines.AO_GBUFFER_NORMAL === 1 && gm.defines.NORMAL_VECTOR_TYPE === 0);
    check('gbuffer set: exactly one material.needsUpdate', gm.version === v0 + 1);
    const pf = gm.fragmentShader;
    check('gbuffer text: decode helpers + #ifdef AO_GBUFFER_NORMAL branch in getViewNormal', pf.includes(GBUFFER_GLSL_DECODE) && /vec3 getViewNormal\(const vec2 uv\) \{\s*#ifdef AO_GBUFFER_NORMAL/.test(pf) && /#elif NORMAL_VECTOR_TYPE == 2/.test(pf));
    check('gbuffer text: octahedral decode where hasNormal, depth reconstruction elsewhere', /if \(\(engineGbFlags\(gbN\.a\) & ENGINE_GB_HAS_NORMAL\) != 0\) return engineOctDecode\(gbN\.rg\);\s*return computeNormalFromDepth\(uv\);/.test(pf));
    // the patch only inserts: removing the inserted text gives the 0.7.0 text back
    const unpatched = pf.replace(GBUFFER_GLSL_DECODE + '\n\t\t', '').replace(/#ifdef AO_GBUFFER_NORMAL[\s\S]*?#elif NORMAL_VECTOR_TYPE == 2/, '#if NORMAL_VECTOR_TYPE == 2');
    check('gbuffer text = 0.7.0 text + the inserted blocks only', sha(unpatched) === AO_FRAG_SHA);
    check('gbuffer text: delimiters balanced, #if / #endif matched', glslBalanced(pf));
    check('AO output contract unchanged (FRAGMENT_OUTPUT define: r = ao, g = linear depth)', gm.defines.FRAGMENT_OUTPUT === 'vec4( mix( ao, 1.0, smoothstep( AO_FADE_NEAR, AO_FADE_FAR, -viewPos.z ) ), -viewPos.z, 0.0, 1.0 )');
    for (let i = 0; i < 10; i++) g.render(r, null, read);
    check('render: tNormal = gbuffer.normal', gm.uniforms.tNormal.value === gb.normal);
    check('10 renders: no per-frame needsUpdate', gm.version === v0 + 1);
    g.gbuffer = gb; g.normals = 'gbuffer';
    check('same G-buffer / same normals again: no program change', gm.version === v0 + 1);

    // nothing allocates per render on the G-buffer path
    gate(inRange, note, "AO render() ('gbuffer')", bytesPerCall((n) => { for (let i = 0; i < n; i++) g.render(r, null, read); }, 50000), { unit: 'frame' });

    // back to 'depth': the 0.7.0 program exactly
    g.normals = 'depth';
    check("normals 'depth': define removed, text + defines back to the pinned 0.7.0 hashes", !('AO_GBUFFER_NORMAL' in gm.defines) && sha(gm.fragmentShader) === AO_FRAG_SHA && sha(JSON.stringify(gm.defines)) === AO_DEFINES_SHA);
    check("normals 'depth': one program change, tNormal unbound", gm.version === v0 + 2 && gm.uniforms.tNormal.value === null);
    g.normals = 'gbuffer';
    check("'gbuffer' again (G-buffer still set): patched program", 'AO_GBUFFER_NORMAL' in gm.defines && gm.fragmentShader === pf && gm.version === v0 + 3);
    g.gbuffer = null; // PostFX tears its G-buffer down
    check('gbuffer = null: back to the 0.7.0 program', !('AO_GBUFFER_NORMAL' in gm.defines) && sha(gm.fragmentShader) === AO_FRAG_SHA && sha(JSON.stringify(gm.defines)) === AO_DEFINES_SHA && gm.version === v0 + 4);
    g.render(r, null, read);
    check('torn down after a warning: no second warning', warns.filter((w) => /without a G-buffer/.test(w)).length === 1);
    check('depth mode with a G-buffer set: unchanged program', (() => { const d = new AOPass(cam); d.gbuffer = gb; return sha(d.material.fragmentShader) === AO_FRAG_SHA && !('AO_GBUFFER_NORMAL' in d.material.defines) && d.material.version === 0; })());
    check('AO dispose: no throw', !throws(() => { ao.dispose(); g.dispose(); }));
  } finally {
    console.warn = warn;
  }
}

console.log(results.join('\n'));
console.log(failures ? `\n${failures} of ${passes + failures} check(s) FAILED` : `\nall ${passes} checks passed`);
process.exit(failures ? 1 : 0);
