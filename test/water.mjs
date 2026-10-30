// Water refraction (0.8.0) + the mid-frame scene copy, headless (no WebGL):
//  - createWaterMaterial without refraction is the 0.7.0 material byte for byte (sha1 of the onBeforeCompile output
//    pinned BEFORE the refraction work, same uniform list, same cache key and defines), also after an on / off trip;
//  - refraction adds exactly the documented uniforms, the WATER_REFRACTION define and the '-refr' key suffix, and its
//    GLSL only lives in '#ifdef WATER_REFRACTION' blocks (removing them gives the 0.7.0 text back);
//  - setWaterRefraction / setWaterSceneTextures / updateWaterRefraction and the Water mesh hook;
//  - SceneCopy: sizing, attach / detach chaining, once-per-render frame stamp, the invalid-target rules (canvas,
//    MSAA, no depth texture: no copy, one warning each) and the save / bind / draw / restore sequence on a stub renderer;
//  - MapWorld `water: { refraction: true }` (headless procedural city): sceneCopy, defines, the bay hook chain,
//    setSize forwarding, setWaterRefraction at runtime; the default world is untouched.
// Run: node test/water.mjs (imports the DOM shim itself). The GPU side (the copy, the GLSL) is checked in the browser.
import './helpers/dom-shim.mjs';
import { createHash } from 'node:crypto';
import * as THREE from 'three';
import {
  createWaterMaterial, setWaterRefraction, setWaterSceneTextures, updateWaterRefraction, waterRefractionOn, Water,
  WATER_REFRACTION_DEFAULTS, WATER_REFRACTION_UNIFORMS,
} from '../src/world/Water.js';
import { SceneCopy } from '../src/render/SceneCopy.js';
import { MapWorld } from '../src/world/MapWorld.js';
import { generateProceduralCity } from '../src/world/MapData.js';

let failures = 0;
const results = [];
function check(name, ok, detail = '') {
  if (!ok) failures++;
  results.push(`${ok ? 'PASS' : 'FAIL'}  ${name}${!ok && detail ? `  (${detail})` : ''}`);
  return ok;
}
const warns = [];
const warn0 = console.warn;
console.warn = (...a) => warns.push(a.join(' '));
for (const k of ['info', 'log']) console[k] = () => {}; // the world build logs its stats

// ---- pinned 0.7.0 output (computed from the unmodified Water.js before the refraction work) ----------------------
const HASH_V = '742b4ffe53f8934cacfc925f5be56a604a2bd52c'; // bay and pond: the same text (the pond differs by define)
const HASH_F = '3955ed711b9133ced426cd6fff693f5d053049ae';
const UNIFORMS_070 = ['uTime', 'uNight', 'uFar', 'uChop', 'uNrm', 'uNoise', 'uShoreF', 'uShoreC', 'uFineRect', 'uCoarseRect',
  'uDeep', 'uShallow', 'uFoamCol', 'uGlowCol', 'uPiers', 'uPierN', 'uPierR', 'uWaveA', 'uWaveB'];
const sha1 = (s) => createHash('sha1').update(s).digest('hex');
const compile = (m) => {
  const sh = { vertexShader: THREE.ShaderLib.standard.vertexShader, fragmentShader: THREE.ShaderLib.standard.fragmentShader, uniforms: {} };
  m.onBeforeCompile(sh, null);
  return sh;
};
const strip = (s) => s.replace(/#ifdef WATER_REFRACTION\n[\s\S]*?\n#endif \/\/ WATER_REFRACTION\n/g, '');
const same = (a, b) => a.length === b.length && a.every((k, i) => k === b[i]);

// ==================================================================== material, refraction off
for (const pond of [false, true]) {
  const tag = pond ? 'pond' : 'bay';
  const m = createWaterMaterial({ normalMap: null, noiseMap: null, pond });
  const sh = compile(m);
  check(`${tag}: vertex shader = 0.7.0 (sha1)`, sha1(sh.vertexShader) === HASH_V, sha1(sh.vertexShader));
  check(`${tag}: fragment shader = 0.7.0 (sha1)`, sha1(sh.fragmentShader) === HASH_F, sha1(sh.fragmentShader));
  check(`${tag}: uniform list = 0.7.0`, same(Object.keys(sh.uniforms), UNIFORMS_070), Object.keys(sh.uniforms).join(','));
  check(`${tag}: cache key = 0.7.0`, m.customProgramCacheKey() === `roadster-${tag}-3`, m.customProgramCacheKey());
  check(`${tag}: defines = 0.7.0`, JSON.stringify(m.defines) === (pond ? '{"WATER_POND":""}' : '{"STANDARD":""}'), JSON.stringify(m.defines));
  check(`${tag}: no refraction state without the option`, !waterRefractionOn(m) && m.userData.refraction === undefined && m.userData.u.uRefractOn === undefined);
}

// ==================================================================== material, refraction on
{
  const off = compile(createWaterMaterial({ normalMap: null, noiseMap: null }));
  for (const pond of [false, true]) {
    const tag = pond ? 'pond' : 'bay';
    const m = createWaterMaterial({ normalMap: null, noiseMap: null, pond, refraction: true });
    const sh = compile(m);
    const added = Object.keys(sh.uniforms).filter((k) => !UNIFORMS_070.includes(k));
    check(`${tag} + refraction: adds exactly the documented uniforms`, same(added.slice().sort(), WATER_REFRACTION_UNIFORMS.slice().sort()) && same(Object.keys(sh.uniforms).slice(0, UNIFORMS_070.length), UNIFORMS_070), added.join(','));
    check(`${tag} + refraction: WATER_REFRACTION define`, m.defines.WATER_REFRACTION === '' && (!pond || m.defines.WATER_POND === ''));
    check(`${tag} + refraction: cache key suffix '-refr'`, m.customProgramCacheKey() === `roadster-${tag}-3-refr`, m.customProgramCacheKey());
    check(`${tag} + refraction: vertex shader unchanged`, sha1(sh.vertexShader) === HASH_V);
    check(`${tag} + refraction: GLSL only in WATER_REFRACTION blocks (stripped = 0.7.0)`, sha1(strip(sh.fragmentShader)) === HASH_F && sh.fragmentShader !== off.fragmentShader);
    const blocks = sh.fragmentShader.match(/#ifdef WATER_REFRACTION\n/g) || [];
    check(`${tag} + refraction: three blocks (declarations, main, emissive)`, blocks.length === 3, String(blocks.length));
    check(`${tag} + refraction: declares every refraction uniform`, WATER_REFRACTION_UNIFORMS.every((k) => new RegExp(`uniform \\w+( \\w+)? ${k};`).test(sh.fragmentShader)));
    check(`${tag} + refraction: transmitted light added as emissive`, sh.fragmentShader.includes('totalEmissiveRadiance += wRefr * ( 1.0 - wFoam );'));
    const cnt = (t, c) => t.split(c).length - 1;
    const full = sh.fragmentShader.match(/#ifdef WATER_REFRACTION\n[\s\S]*?\n#endif \/\/ WATER_REFRACTION\n/g) || [];
    check(`${tag} + refraction: every block has balanced braces / parentheses`, full.length === 3 && full.every((b) => cnt(b, '{') === cnt(b, '}') && cnt(b, '(') === cnt(b, ')')));
    check(`${tag} + refraction: starts opaque (uRefractOn 0, no textures)`, m.userData.u.uRefractOn.value === 0 && m.userData.u.uSceneColor.value === null && m.userData.u.uSceneDepth.value === null);
  }
  const m = createWaterMaterial({ normalMap: null, noiseMap: null, refraction: true });
  const u = m.userData.u, D = WATER_REFRACTION_DEFAULTS;
  check('refraction defaults -> uRefract / uAbsorb', u.uRefract.value.equals(new THREE.Vector4(D.strength, D.murk, D.foamDepth, D.maxDepth)) && u.uAbsorb.value.equals(new THREE.Vector3(...D.absorption)));
  const m2 = createWaterMaterial({ normalMap: null, noiseMap: null, refraction: { strength: 0.05, murk: 0.3, foamDepth: 0.8, maxDepth: 12, absorption: [0.5, 0.2, 0.1] } });
  const u2 = m2.userData.u;
  check('refraction options -> uRefract / uAbsorb', u2.uRefract.value.equals(new THREE.Vector4(0.05, 0.3, 0.8, 12)) && u2.uAbsorb.value.equals(new THREE.Vector3(0.5, 0.2, 0.1)));
  const m3 = createWaterMaterial({ normalMap: null, noiseMap: null, refraction: { murk: 1 } });
  check('partial options keep the other defaults', m3.userData.u.uRefract.value.equals(new THREE.Vector4(D.strength, 1, D.foamDepth, D.maxDepth)));
  let threw = false;
  try { createWaterMaterial({ normalMap: null, noiseMap: null, refraction: { strenght: 1 } }); } catch { threw = true; }
  check('unknown refraction option throws', threw);
  const pondSh = compile(createWaterMaterial({ normalMap: null, noiseMap: null, pond: true, refraction: true })).fragmentShader;
  check('pond: nominal bed + no depth foam under WATER_POND', /#ifdef WATER_POND\n\s+dep \+= 2\.0;/.test(pondSh) && /#ifndef WATER_POND\n\s+\/\/ depth foam/.test(pondSh));
}

// ==================================================================== setWaterRefraction (runtime A/B)
{
  const m = createWaterMaterial({ normalMap: null, noiseMap: null });
  let v = m.version;
  check('setWaterRefraction(on) on a plain material: changed', setWaterRefraction(m, true) === true && m.version === v + 1);
  check('  define + key suffix + uniforms added', m.defines.WATER_REFRACTION === '' && m.customProgramCacheKey() === 'roadster-bay-3-refr' && WATER_REFRACTION_UNIFORMS.every((k) => m.userData.u[k] !== undefined));
  check('  compiles with the refraction blocks', compile(m).fragmentShader.includes('uniform sampler2D uSceneColor;'));
  v = m.version;
  check('setWaterRefraction(on) again: no needsUpdate', setWaterRefraction(m, true) === false && m.version === v);
  check('setWaterRefraction(off): changed', setWaterRefraction(m, false) === true && m.version === v + 1 && m.defines.WATER_REFRACTION === undefined && m.customProgramCacheKey() === 'roadster-bay-3');
  v = m.version;
  check('setWaterRefraction(off) again: no needsUpdate', setWaterRefraction(m, false) === false && m.version === v);
  const sh = compile(m);
  check('after on -> off: shaders = 0.7.0 again (sha1)', sha1(sh.vertexShader) === HASH_V && sha1(sh.fragmentShader) === HASH_F);
}

// ==================================================================== scene textures + per-draw uniforms
{
  const m = createWaterMaterial({ normalMap: null, noiseMap: null, refraction: true });
  const u = m.userData.u;
  const tc = new THREE.Texture(), td = new THREE.Texture(), src = { valid: true };
  setWaterSceneTextures(m, tc, td, src);
  check('setWaterSceneTextures sets the samplers + source', u.uSceneColor.value === tc && u.uSceneDepth.value === td && m.userData.refraction.source === src);
  const rt = new THREE.WebGLRenderTarget(320, 200);
  const r = { getRenderTarget: () => rt };
  const cam = new THREE.PerspectiveCamera(60, 1.6, 0.25, 4500);
  updateWaterRefraction(m, r, cam);
  check('updateWaterRefraction: size of the bound target, camera near / far, on', u.uSceneRes.value.x === 320 && u.uSceneRes.value.y === 200 && u.uNearFar.value.x === 0.25 && u.uNearFar.value.y === 4500 && u.uRefractOn.value === 1);
  src.valid = false; updateWaterRefraction(m, r, cam);
  check('  source copy invalid this frame -> off', u.uRefractOn.value === 0);
  src.valid = true; updateWaterRefraction(m, { getRenderTarget: () => null }, cam);
  check('  canvas target -> off', u.uRefractOn.value === 0);
  updateWaterRefraction(m, r, cam);
  setWaterSceneTextures(m, null, null);
  check('setWaterSceneTextures(null) -> off at once', u.uRefractOn.value === 0 && u.uSceneColor.value === null);
  updateWaterRefraction(m, r, cam);
  check('  and stays off per draw without textures', u.uRefractOn.value === 0);
  const plain = createWaterMaterial({ normalMap: null, noiseMap: null });
  updateWaterRefraction(plain, r, cam);
  check('updateWaterRefraction on a plain material: no-op', plain.userData.u.uRefractOn === undefined && same(Object.keys(plain.userData.u), UNIFORMS_070));

  // the bay mesh's own hook: follows the camera, sets uFar, and the refraction uniforms once they exist
  const wm = createWaterMaterial({ normalMap: null, noiseMap: null });
  const water = new Water({ material: wm, level: -0.6 });
  const mesh = water.build();
  cam.position.set(10.2, 5, -3.9); cam.updateMatrixWorld();
  mesh.onBeforeRender(r, null, cam);
  check('Water hook without refraction: uFar only, no new uniforms', wm.userData.u.uFar.value === 4500 && same(Object.keys(wm.userData.u), UNIFORMS_070) && mesh.matrixWorld.elements[12] === 10 && mesh.matrixWorld.elements[14] === -4);
  setWaterRefraction(wm, true); setWaterSceneTextures(wm, tc, td, src);
  mesh.onBeforeRender(r, null, cam);
  check('Water hook with refraction: per-draw uniforms', wm.userData.u.uRefractOn.value === 1 && wm.userData.u.uNearFar.value.x === 0.25 && wm.userData.u.uSceneRes.value.x === 320);
}

// ==================================================================== SceneCopy
function stubRenderer(target = null) {
  const log = [];
  const r = {
    log, target, face: 0, mip: 0, autoClear: true,
    xr: { enabled: true }, shadowMap: { autoUpdate: true },
    info: { render: { frame: 100 } },
    extensions: { has: (n) => n === 'EXT_color_buffer_float' },
    state: { viewport: (v) => log.push(['viewport', v.x, v.y, v.z, v.w]) },
    getRenderTarget: () => r.target,
    setRenderTarget(t, face = 0, mip = 0) { log.push(['target', t, face, mip]); r.target = t; r.face = face; r.mip = mip; },
    getActiveCubeFace: () => r.face,
    getActiveMipmapLevel: () => r.mip,
    getCurrentViewport: (v) => v.set(3, 4, 300, 200),
    getDrawingBufferSize: (v) => v.set(1513, 983),
    clear(c, d, s) { log.push(['clear', c, d, s]); },
    render(scene, camera) {
      r.info.render.frame++; // three increments it on every render() call
      log.push(['render', scene, camera, r.target, r.xr.enabled, r.shadowMap.autoUpdate, r.autoClear]);
    },
  };
  return r;
}
const sceneTarget = (o = {}) => new THREE.WebGLRenderTarget(64, 32, { depthTexture: new THREE.DepthTexture(64, 32), ...o });
{
  // sizing
  const c0 = new SceneCopy(stubRenderer());
  check('size from the renderer drawing buffer, rounded with scale 0.5', c0.width === 757 && c0.height === 492 && c0.target.width === 757 && c0.target.height === 492, `${c0.width}x${c0.height}`);
  const col = c0.color, dep = c0.depth;
  c0.setSize(3024, 1964);
  check('setSize: drawing-buffer px * scale', c0.width === 1512 && c0.height === 982 && c0.target.width === 1512);
  check('textures survive setSize (same objects)', c0.color === col && c0.depth === dep);
  c0.setSize(0, NaN);
  check('setSize: min 1x1 (0 / NaN)', c0.width === 1 && c0.height === 1);
  const mv = c0.material.version;
  c0.setSize(3024, 1964);
  c0.setScale(1);
  check('setScale(1): full size from the last setSize, one-texel depth path (recompile)', c0.width === 3024 && c0.height === 1964 && c0.material.defines.COPY_FULL === '' && c0.material.version === mv + 1);
  c0.setScale(0.25);
  check('setScale(0.25): quarter size, 2x2 depth path again', c0.width === 756 && c0.height === 491 && c0.material.defines.COPY_FULL === undefined && c0.material.version === mv + 2 && c0.color === col);
  c0.setScale(0.5);
  const c1 = new SceneCopy(null, { scale: 0.25, width: 2, height: 3 });
  check('scale 0.25 of 2x3 -> 1x1, explicit size, null renderer', c1.width === 1 && c1.height === 1 && c1.scale === 0.25);
  check('colour + depth = one MRT target (count 2)', c0.target.textures.length === 2 && c0.color === c0.target.textures[0] && c0.depth === c0.target.textures[1]);
  check('colour RGBA16F linear', col.type === THREE.HalfFloatType && col.format === THREE.RGBAFormat && col.minFilter === THREE.LinearFilter && col.magFilter === THREE.LinearFilter);
  check('depth R32F nearest', dep.type === THREE.FloatType && dep.format === THREE.RedFormat && dep.minFilter === THREE.NearestFilter && dep.magFilter === THREE.NearestFilter);
  check('copy target has no depth buffer', c0.target.depthBuffer === false);
  const cd = new SceneCopy(null, { color: false }), cc = new SceneCopy(null, { depth: false });
  check('depth only / colour only: one attachment each', cd.color === null && cd.depth === cd.target.texture && cd.target.textures.length === 1 && cc.depth === null && cc.color === cc.target.texture && cc.target.textures.length === 1);
  check('MRT shader writes depth at location 1, single depth at 0', c0.material.defines.COPY_DEPTH_LOCATION === 1 && cd.material.defines.COPY_DEPTH_LOCATION === 0 && cc.material.defines.COPY_DEPTH === undefined);
  check('copy material: GLSL3, no depth test / write, no blending', c0.material.glslVersion === THREE.GLSL3 && !c0.material.depthTest && !c0.material.depthWrite && c0.material.blending === THREE.NoBlending);
  let threw = 0;
  try { new SceneCopy(null, { color: false, depth: false }); } catch { threw++; }
  try { new SceneCopy(null, { scale: 0 }); } catch { threw++; }
  check('nothing to copy / scale 0 throw', threw === 2);
}
{
  // attach / detach chaining
  const copy = new SceneCopy(null);
  const mesh = new THREE.Mesh();
  const calls = [];
  // the existing hook records the target bound when it runs: the scene target again, i.e. after the copy
  const own = function (renderer, scene, camera, geometry, material, group) { calls.push([this, renderer, scene, camera, geometry, material, group, renderer.target, copy.copies]); };
  mesh.onBeforeRender = own;
  copy.attach(mesh);
  check('attach replaces the hook with a chain', mesh.onBeforeRender !== own);
  const st0 = sceneTarget();
  const r = stubRenderer(st0);
  mesh.onBeforeRender(r, 's', 'c', 'g', 'm', 'grp');
  check('chain: the copy ran, then the existing hook with the same args and this', copy.copies === 1 && calls.length === 1 && calls[0][0] === mesh && calls[0][1] === r && calls[0][3] === 'c' && calls[0][6] === 'grp');
  check('the existing hook runs after the copy (it sees the copy done and the scene target bound again)', calls[0][7] === st0 && calls[0][8] === 1);
  const hook = mesh.onBeforeRender;
  copy.attach(mesh);
  check('attach twice: one chain', mesh.onBeforeRender === hook);
  copy.detach(mesh);
  check('detach restores the original hook', mesh.onBeforeRender === own && copy.valid === false);
  const bare = new THREE.Mesh();
  copy.attach(bare); copy.detach(bare);
  check('detach on a mesh without an own hook removes the own property', !Object.prototype.hasOwnProperty.call(bare, 'onBeforeRender') && bare.onBeforeRender === THREE.Object3D.prototype.onBeforeRender);
  // chained over by someone else: detach leaves a pass-through
  const m2 = new THREE.Mesh();
  let outer = 0, inner = 0;
  m2.onBeforeRender = () => { inner++; };
  copy.attach(m2);
  const mine = m2.onBeforeRender;
  m2.onBeforeRender = function (...a) { outer++; mine.apply(this, a); };
  copy.detach(m2);
  const r2 = stubRenderer(sceneTarget()), n0 = copy.copies;
  m2.onBeforeRender(r2, null, null, null, null, null);
  check('detach under a foreign chain: pass-through (no copy, inner hook still runs)', outer === 1 && inner === 1 && copy.copies === n0);
  copy.attach(m2); r2.info.render.frame++;
  m2.onBeforeRender(r2, null, null, null, null, null);
  check('re-attach under the foreign chain reactivates it', copy.copies === n0 + 1 && inner === 2);
}
{
  // invalid targets: no copy, one warning per reason
  const cases = [['canvas', null, 'canvas'], ['multisampled', sceneTarget({ samples: 4 }), 'multisampled'], ['no depth texture', new THREE.WebGLRenderTarget(64, 32), 'depthTexture']];
  for (const [name, target, word] of cases) {
    const copy = new SceneCopy(null);
    const mesh = new THREE.Mesh(); copy.attach(mesh);
    const r = stubRenderer(target);
    warns.length = 0;
    mesh.onBeforeRender(r, null, null, null, null, null);
    const early = warns.length;
    for (let i = 0; i < 3; i++) { r.info.render.frame++; mesh.onBeforeRender(r, null, null, null, null, null); }
    check(`${name}: no copy (no target switch, no draw), valid false`, r.log.length === 0 && copy.copies === 0 && copy.valid === false && copy.reason !== null);
    check(`${name}: one warning naming the reason, deferred to the next render`, early === 0 && warns.length === 1 && warns[0].includes(word), warns.join(' | '));
  }
  const skipCase = (name, setup, word, reason) => {
    const copy = new SceneCopy(null);
    const r = stubRenderer(sceneTarget());
    setup(r);
    const mesh = new THREE.Mesh(); copy.attach(mesh);
    warns.length = 0;
    for (let i = 0; i < 3; i++) { r.info.render.frame++; mesh.onBeforeRender(r, null, null, null, null, null); }
    check(`${name}: no copy, one warning, reason '${reason}'`, copy.copies === 0 && r.log.length === 0 && copy.reason === reason && warns.length === 1 && warns[0].includes(word), warns.join(' | '));
  };
  skipCase('no float colour targets', (r) => { r.extensions.has = () => false; }, 'EXT_color_buffer_float', 'unsupported');
  skipCase('reversed depth buffer', (r) => { r.capabilities = { reversedDepthBuffer: true }; }, 'reversed depth', 'reversed');
  // three's transmission pass: an MSAA target inside the same render() call, then the main pass copies: no warning
  const copy = new SceneCopy(null);
  const st = sceneTarget(), ms = sceneTarget({ samples: 4 });
  const r = stubRenderer(st);
  const mesh = new THREE.Mesh(); copy.attach(mesh);
  warns.length = 0;
  for (let i = 0; i < 4; i++) {
    r.info.render.frame++;
    r.target = ms; mesh.onBeforeRender(r, null, null, null, null, null); // transmission pass: skipped
    r.target = st; mesh.onBeforeRender(r, null, null, null, null, null); // main pass: copied
  }
  check('transmission-pass skip followed by a copy in the same render: no warning, valid', warns.length === 0 && copy.copies === 4 && copy.valid === true && copy.reason === null, warns.join(' | '));
  r.target = ms; r.info.render.frame++; mesh.onBeforeRender(r, null, null, null, null, null);
  r.info.render.frame++; mesh.onBeforeRender(r, null, null, null, null, null);
  check('  a render with no copy at all still warns once', warns.length === 1 && warns[0].includes('multisampled'));
}
{
  // the valid copy: save / bind / clear / draw / restore
  const copy = new SceneCopy(null);
  const st = sceneTarget();
  const r = stubRenderer(st);
  r.face = 2; r.mip = 1;
  const a = new THREE.Mesh(), b = new THREE.Mesh();
  let aHook = 0, bHook = 0;
  a.onBeforeRender = () => { aHook++; }; b.onBeforeRender = () => { bHook++; };
  const aOwn = a.onBeforeRender, bOwn = b.onBeforeRender;
  copy.attach(a).attach(b);
  warns.length = 0;
  a.onBeforeRender(r, null, null, null, null, null);
  const L = r.log;
  check('sequence: bind copy target, clear colour, draw, restore target (face, mip), restore viewport', L.length === 5
    && L[0][0] === 'target' && L[0][1] === copy.target
    && L[1][0] === 'clear' && L[1][1] === true && L[1][2] === false && L[1][3] === false
    && L[2][0] === 'render' && L[2][1] === copy._quad._mesh && L[2][3] === copy.target
    && L[3][0] === 'target' && L[3][1] === st && L[3][2] === 2 && L[3][3] === 1
    && L[4][0] === 'viewport' && L[4][1] === 3 && L[4][2] === 4 && L[4][3] === 300 && L[4][4] === 200, JSON.stringify(L.map((e) => e[0])));
  check('during the draw: xr off, shadow auto-update off, autoClear off', L[2][4] === false && L[2][5] === false && L[2][6] === false);
  check('after: xr, shadow auto-update, autoClear restored', r.xr.enabled === true && r.shadowMap.autoUpdate === true && r.autoClear === true && r.target === st);
  check('the quad samples attachment 0 and the depth texture', copy.material.uniforms.tColor.value === st.texture && copy.material.uniforms.tDepth.value === st.depthTexture);
  check('valid, stamped with the frame after the copy, no warning', copy.valid === true && copy.frame === r.info.render.frame && copy.copies === 1 && warns.length === 0);
  b.onBeforeRender(r, null, null, null, null, null);
  check('second mesh in the same render: the copy is reused', copy.copies === 1 && L.length === 5 && aHook === 1 && bHook === 1);
  r.info.render.frame++; // the next frame's scene render
  b.onBeforeRender(r, null, null, null, null, null);
  a.onBeforeRender(r, null, null, null, null, null);
  check('next render: one new copy for both meshes', copy.copies === 2 && L.length === 10 && aHook === 2 && bHook === 2);
  const g = new THREE.WebGLRenderTarget(64, 32, { count: 3, depthTexture: new THREE.DepthTexture(64, 32) }); // a G-buffer-like MRT
  r.target = g; r.info.render.frame++;
  a.onBeforeRender(r, null, null, null, null, null);
  check('MRT scene target (G-buffer): copies attachment 0 + depth', copy.copies === 3 && copy.material.uniforms.tColor.value === g.textures[0] && copy.material.uniforms.tDepth.value === g.depthTexture);
  r.target = null; r.info.render.frame++;
  a.onBeforeRender(r, null, null, null, null, null);
  check('then an invalid target: valid drops to false', copy.valid === false && copy.copies === 3);
  copy.clear = false; r.target = st; r.info.render.frame++;
  const n = L.length;
  a.onBeforeRender(r, null, null, null, null, null);
  check('clear: false skips the clear', copy.copies === 4 && !L.slice(n).some((e) => e[0] === 'clear'));
  // a throwing draw still restores everything
  r.render = () => { throw new Error('boom'); };
  r.info.render.frame++;
  let threw = false;
  try { a.onBeforeRender(r, null, null, null, null, null); } catch { threw = true; }
  check('a failing draw restores target and flags (finally)', threw && r.target === st && r.face === 2 && r.mip === 1 && r.xr.enabled === true && r.shadowMap.autoUpdate === true && r.autoClear === true);
  copy.dispose();
  check('dispose detaches every mesh (original hooks back)', a.onBeforeRender === aOwn && b.onBeforeRender === bOwn && copy.valid === false);
}

// ==================================================================== MapWorld plumbing (headless procedural city)
// the procedural city has no ponds (data.water is empty): the same city plus one 30 x 20 m pond west of the spawn
function cityWithPond() {
  const map = generateProceduralCity(1337);
  map.water = [{ p: [[-60, -100], [-30, -100], [-30, -80], [-60, -80]] }];
  return map;
}
{
  const scene = new THREE.Scene();
  const world = new MapWorld({ scene, renderer: null, data: { map: cityWithPond(), farlandUrl: false }, terrain: false, traffic: false, streaming: false, water: { refraction: true } });
  await world.load(() => {});
  const copy = world.sceneCopy;
  const bay = world.waterfront && world.waterfront.water && world.waterfront.water.mesh;
  check('world: water.refraction creates world.sceneCopy (scale 0.5)', copy instanceof SceneCopy && copy.scale === 0.5 && world.waterRefraction === true);
  check('world: bay + pond materials have the define and the copy textures', waterRefractionOn(world.mat.water) && waterRefractionOn(world.mat.pond)
    && world.mat.water.userData.u.uSceneColor.value === copy.color && world.mat.water.userData.u.uSceneDepth.value === copy.depth
    && world.mat.pond.userData.u.uSceneColor.value === copy.color && world.mat.water.userData.refraction.source === copy);
  check('world: the bay mesh exists and is attached', !!bay && copy._recs.has(bay));
  const st = sceneTarget();
  const r = stubRenderer(st);
  const cam = new THREE.PerspectiveCamera(60, 1.5, 0.1, 6000);
  cam.position.set(792, 3.9, 104); cam.updateMatrixWorld();
  bay.onBeforeRender(r, scene, cam, bay.geometry, bay.material, null);
  const u = world.mat.water.userData.u;
  check('world: drawing the bay copies, then the Water hook sets the per-draw uniforms', copy.copies === 1 && copy.valid && u.uRefractOn.value === 1 && u.uSceneRes.value.x === 64 && u.uNearFar.value.y === 6000 && bay.matrixWorld.elements[12] === 792);
  world.setSize(800, 600, 2);
  check('world.setSize forwards floor(css * pr) to the copy', copy.width === 800 && copy.height === 600);
  world.setWaterRefraction(false);
  const v0 = copy.copies;
  r.info.render.frame++;
  bay.onBeforeRender(r, scene, cam, bay.geometry, bay.material, null);
  check('world.setWaterRefraction(false): define off, copy detached', !waterRefractionOn(world.mat.water) && !waterRefractionOn(world.mat.pond) && copy.copies === v0 && !copy._recs.has(bay) && world.waterRefraction === false && u.uRefractOn.value === 0);
  world.setWaterRefraction(true);
  r.info.render.frame++;
  bay.onBeforeRender(r, scene, cam, bay.geometry, bay.material, null);
  check('world.setWaterRefraction(true): back on, same copy re-attached', waterRefractionOn(world.mat.water) && world.sceneCopy === copy && copy.copies === v0 + 1 && u.uRefractOn.value === 1);
  // ponds: attached, drawn after the opaque city while on, per-draw uniforms from their own (chained) hook
  const ponds = world._ponds, pu = world.mat.pond.userData.u;
  check('world: the pond mesh exists, is attached and draws at renderOrder 1', !!ponds && ponds.name === 'ponds' && ponds.renderOrder === 1 && copy._recs.has(ponds));
  r.info.render.frame++;
  ponds.onBeforeRender(r, scene, cam, ponds.geometry, ponds.material, null);
  check('world: drawing the ponds copies and sets the pond uniforms', copy.copies === v0 + 2 && pu.uRefractOn.value === 1 && pu.uSceneRes.value.x === 64 && pu.uNearFar.value.y === 6000);
  bay.onBeforeRender(r, scene, cam, bay.geometry, bay.material, null);
  check('world: the bay in the same render reuses the pond copy', copy.copies === v0 + 2 && u.uRefractOn.value === 1);
  world.setWaterRefraction(false);
  check('world.setWaterRefraction(false): pond renderOrder restored, detached', ponds.renderOrder === 0 && !copy._recs.has(ponds));
  r.info.render.frame++;
  ponds.onBeforeRender(r, scene, cam, ponds.geometry, ponds.material, null);
  check('  pond draws copy nothing while off', copy.copies === v0 + 2 && pu.uRefractOn.value === 0);
}
{
  const scene = new THREE.Scene();
  const world = new MapWorld({ scene, renderer: null, data: { map: cityWithPond(), farlandUrl: false }, terrain: false, traffic: false, streaming: false });
  await world.load(() => {});
  const bay = world.waterfront.water.mesh, ponds = world._ponds;
  check('default world: no sceneCopy, refraction off', world.sceneCopy === null && world.waterRefraction === false);
  const sb = compile(world.mat.water), sp = compile(world.mat.pond);
  check('default world: bay + pond shaders = 0.7.0 (sha1), uniforms unchanged', sha1(sb.fragmentShader) === HASH_F && sha1(sp.fragmentShader) === HASH_F && sha1(sb.vertexShader) === HASH_V
    && same(Object.keys(sb.uniforms), UNIFORMS_070) && world.mat.water.customProgramCacheKey() === 'roadster-bay-3');
  const r = stubRenderer(sceneTarget());
  const cam = new THREE.PerspectiveCamera(60, 1.5, 0.1, 6000); cam.updateMatrixWorld();
  bay.onBeforeRender(r, scene, cam, bay.geometry, bay.material, null);
  check('default world: drawing the bay copies nothing', r.log.length === 0 && same(Object.keys(world.mat.water.userData.u), UNIFORMS_070));
  check('default world: ponds at renderOrder 0 without an own hook', !!ponds && ponds.renderOrder === 0 && !Object.prototype.hasOwnProperty.call(ponds, 'onBeforeRender'));
  // a hook the pond mesh already has (MeshChunker / GBuffer chain hooks) survives the refraction wiring
  let pondHook = 0, pondHookOn = -1;
  const pondOwn = function () { pondHook++; pondHookOn = world.mat.pond.userData.u.uRefractOn.value; };
  ponds.onBeforeRender = pondOwn;
  world.setSize(1000, 500, 1.5);
  world.setWaterRefraction(true);
  check('runtime setWaterRefraction(true): copy created at the last setSize, attached', world.sceneCopy instanceof SceneCopy && world.sceneCopy.width === 750 && world.sceneCopy.height === 375 && world.sceneCopy._recs.has(bay) && world.sceneCopy._recs.has(ponds) && waterRefractionOn(world.mat.water));
  r.info.render.frame++;
  ponds.onBeforeRender(r, scene, cam, ponds.geometry, ponds.material, null);
  check('pond hook chained: copy, pond uniforms, then the existing hook', world.sceneCopy.copies === 1 && pondHook === 1 && pondHookOn === 1);
  world.setWaterRefraction(false);
  r.info.render.frame++;
  ponds.onBeforeRender(r, scene, cam, ponds.geometry, ponds.material, null);
  check('  off: the existing hook still runs, no copy', pondHook === 2 && world.sceneCopy.copies === 1 && ponds.renderOrder === 0);
}

console.warn = warn0;
console.error(results.join('\n'));
console.error(failures ? `\n${failures} check(s) FAILED` : `\nall ${results.length} checks passed`);
process.exit(failures ? 1 : 0);
