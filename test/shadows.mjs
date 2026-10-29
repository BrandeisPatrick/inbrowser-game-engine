// Sun shadow cascades + kernels (headless, no renderer): the global chunk patches and Atmosphere's cascade setup.
// Run: node test/shadows.mjs. Checks
//   - `cascades: 2` patches lights_fragment_begin byte-identically to 0.6.0 (expected text snapshotted below, before
//     the 2-4 cascade work), and keeps doing so after an Atmosphere with 2 cascades is built;
//   - the 3 / 4 cascade texts: balanced, guarded by NUM_DIR_LIGHT_SHADOWS > N-1, and after three's light-number
//     substitution + loop unrolling + preprocessing (emulated here) balanced, free of redeclarations and with one
//     cascade lookup per map, for 1-4 directional shadows; the upgrade keeps other patches of the chunk;
//   - Atmosphere: cascade lights (names, order, black, autoUpdate), boxes by height step, round-robin map updates,
//     shadowCascadeInfo shape + reuse, 'pcss' constants;
//   - the 'pcss' chunk: wrapped in its define (inert without it), template uniforms shared by reference, per-material
//     define switching between the three kernels.
// Pixel identity and timings are measured in the browser (ENGINE.md, measurement protocol).
import { createHash } from 'node:crypto';
import * as THREE from 'three';

let failures = 0;
const results = [];
function check(name, ok, detail = '') {
  if (!ok) failures++;
  results.push(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? `   (${detail})` : ''}`);
  return ok;
}

// stock chunks, before any engine module patches them (engine modules are imported dynamically below)
const STOCK_LIGHTS = THREE.ShaderChunk.lights_fragment_begin;
const STOCK_PARS = THREE.ShaderChunk.shadowmap_pars_fragment;
const ORIG = 'directLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( directionalShadowMap[ i ], directionalLightShadow.shadowMapSize, directionalLightShadow.shadowIntensity, directionalLightShadow.shadowBias, directionalLightShadow.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;';

// 0.6.0 replacement block (snapshot of shadowCascade.js output before the 2-4 cascade change, exact bytes)
const EXPECTED_2 = "\n\t\t#if ( UNROLLED_LOOP_INDEX == 0 ) && ( NUM_DIR_LIGHT_SHADOWS > 1 )\n\t\tif ( dot( directionalLights[ 1 ].color, vec3( 1.0 ) ) <= 0.0 && dot( directionalLights[ 1 ].direction, directionalLight.direction ) > 0.9999 ) {\n\t\t\tvec3 rcN = vDirectionalShadowCoord[ 0 ].xyz / vDirectionalShadowCoord[ 0 ].w;\n\t\t\tvec3 rcF = vDirectionalShadowCoord[ 1 ].xyz / vDirectionalShadowCoord[ 1 ].w;\n\t\t\tvec2 rcE = abs( rcN.xy - 0.5 );\n\t\t\tfloat rcW = ( rcN.z > 1.0 ) ? 1.0 : smoothstep( 0.39, 0.47, max( rcE.x, rcE.y ) );\n\t\t\tfloat rcS = 1.0;\n\t\t\tif ( receiveShadow ) {\n\t\t\t\tDirectionalLightShadow rcFarS = directionalLightShadows[ 1 ];\n\t\t\t\tfloat rcSN = 1.0, rcSF = 1.0;\n\t\t\t\tif ( rcW < 1.0 ) rcSN = getShadow( directionalShadowMap[ 0 ], directionalLightShadow.shadowMapSize, directionalLightShadow.shadowIntensity, directionalLightShadow.shadowBias, directionalLightShadow.shadowRadius, vDirectionalShadowCoord[ 0 ] );\n\t\t\t\tif ( rcW > 0.0 ) {\n\t\t\t\t\trcSF = getShadow( directionalShadowMap[ 1 ], rcFarS.shadowMapSize, rcFarS.shadowIntensity, rcFarS.shadowBias, rcFarS.shadowRadius, vDirectionalShadowCoord[ 1 ] );\n\t\t\t\t\tvec2 rcEF = abs( rcF.xy - 0.5 );\n\t\t\t\t\trcSF = mix( rcSF, 1.0, smoothstep( 0.43, 0.495, max( rcEF.x, rcEF.y ) ) );\n\t\t\t\t}\n\t\t\t\trcS = mix( rcSN, rcSF, rcW );\n\t\t\t}\n\t\t\tdirectLight.color *= rcS;\n\t\t} else {\n\t\t\tdirectLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( directionalShadowMap[ 0 ], directionalLightShadow.shadowMapSize, directionalLightShadow.shadowIntensity, directionalLightShadow.shadowBias, directionalLightShadow.shadowRadius, vDirectionalShadowCoord[ 0 ] ) : 1.0;\n\t\t}\n\t\t#elif ( UNROLLED_LOOP_INDEX == 1 )\n\t\tif ( dot( directLight.color, vec3( 1.0 ) ) > 0.0 ) {\n\t\t\tDirectionalLightShadow rcFarS = directionalLightShadows[ 1 ];\n\t\t\tdirectLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( directionalShadowMap[ 1 ], rcFarS.shadowMapSize, rcFarS.shadowIntensity, rcFarS.shadowBias, rcFarS.shadowRadius, vDirectionalShadowCoord[ 1 ] ) : 1.0;\n\t\t}\n\t\t#else\n\t\tdirectLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( directionalShadowMap[ i ], directionalLightShadow.shadowMapSize, directionalLightShadow.shadowIntensity, directionalLightShadow.shadowBias, directionalLightShadow.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;\n\t\t#endif";
// sha-256 of the whole patched lights_fragment_begin with three r180 (0.6.0)
const EXPECTED_2_SHA_R180 = '815790342674c6216767d4afebc905cf3a016e4cab157b5804a5902477749a58';
const sha = (s) => createHash('sha256').update(s).digest('hex');

// ---------------------------------------------------------------- GLSL helpers: three's program assembly, emulated
const balanced = (s, open, close) => { let d = 0; for (const ch of s) { if (ch === open) d++; else if (ch === close && --d < 0) return false; } return d === 0; };
function directivesBalanced(s) {
  let d = 0;
  for (const l of s.split('\n')) {
    const m = /^\s*#\s*(if|ifdef|ifndef|endif)\b/.exec(l);
    if (m) d += m[1] === 'endif' ? -1 : 1;
    if (d < 0) return false;
  }
  return d === 0;
}
function evalIf(expr, defs) {
  const e = expr.replace(/\/\/.*$/, '')
    .replace(/defined\s*\(\s*(\w+)\s*\)/g, (_, n) => (n in defs ? '1' : '0'))
    .replace(/\b[A-Za-z_]\w*\b/g, (n) => (n in defs ? String(defs[n] === '' ? 1 : defs[n]) : '0'));
  return !!Function(`return ( ${e} );`)();
}
/** C preprocessor conditionals only (#if / #ifdef / #ifndef / #elif / #else / #endif); other lines are kept. */
function preprocess(src, defs) {
  const out = [], stack = [];
  let active = true;
  for (const line of src.split('\n')) {
    const m = /^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)$/.exec(line);
    if (!m) { if (active) out.push(line); continue; }
    const [, d, rest] = m;
    if (d === 'if' || d === 'ifdef' || d === 'ifndef') {
      const v = active && (d === 'if' ? evalIf(rest, defs) : (rest.trim() in defs) === (d === 'ifdef'));
      stack.push({ parent: active, taken: v });
      active = v;
    } else {
      const top = stack[stack.length - 1];
      if (!top) throw new Error('unbalanced #' + d);
      if (d === 'elif') { active = top.parent && !top.taken && evalIf(rest, defs); if (active) top.taken = true; }
      else if (d === 'else') { active = top.parent && !top.taken; top.taken = true; }
      else { stack.pop(); active = top.parent; }
    }
  }
  if (stack.length) throw new Error('unterminated #if');
  return out.join('\n');
}
// three r180 WebGLProgram: replaceLightNums, then unrollLoops
const UNROLL = /#pragma unroll_loop_start\s+for\s*\(\s*int\s+i\s*=\s*(\d+)\s*;\s*i\s*<\s*(\d+)\s*;\s*i\s*\+\+\s*\)\s*{([\s\S]+?)}\s+#pragma unroll_loop_end/g;
function assembleLights(chunk, nDir) {
  const nums = { NUM_DIR_LIGHTS: nDir, NUM_SPOT_LIGHTS: 0, NUM_SPOT_LIGHT_MAPS: 0, NUM_SPOT_LIGHT_COORDS: 0, NUM_RECT_AREA_LIGHTS: 0, NUM_POINT_LIGHTS: 0, NUM_HEMI_LIGHTS: 1, NUM_DIR_LIGHT_SHADOWS: nDir, NUM_SPOT_LIGHT_SHADOWS: 0, NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS: 0, NUM_POINT_LIGHT_SHADOWS: 0 };
  let s = chunk;
  for (const k of Object.keys(nums).sort((a, b) => b.length - a.length)) s = s.replace(new RegExp(`\\b${k}\\b`, 'g'), String(nums[k]));
  s = s.replace(UNROLL, (m, a, b, body) => {
    let r = '';
    for (let i = +a; i < +b; i++) r += body.replace(/\[\s*i\s*\]/g, `[ ${i} ]`).replace(/UNROLLED_LOOP_INDEX/g, i);
    return r;
  });
  return preprocess(s, { USE_SHADOWMAP: '', RE_Direct: '', RE_IndirectDiffuse: '', RE_IndirectSpecular: '' });
}
/** Names declared twice in one scope (the unrolled light loop shares one scope across iterations). */
function redeclarations(src) {
  const code = src.replace(/\/\/.*$/gm, '').replace(/\/\*[\s\S]*?\*\//g, '');
  const scopes = [new Set()], dup = [];
  const TYPE = /^\s*(?:const\s+)?(?:float|int|bool|vec2|vec3|vec4|mat2|mat3|DirectionalLightShadow|DirectionalLight|IncidentLight)\s+([\s\S]+)$/;
  let stmt = '', depth = 0;
  const flush = () => {
    const m = TYPE.exec(stmt);
    if (m && !/^\s*\w+\s*\(/.test(m[1])) {
      let d = 0, part = '';
      const names = [];
      for (const ch of m[1] + ',') {
        if (ch === '(') d++; else if (ch === ')') d--;
        if (ch === ',' && d === 0) { names.push(part.trim().split(/[\s=[]/)[0]); part = ''; } else part += ch;
      }
      for (const n of names) { const top = scopes[scopes.length - 1]; if (top.has(n)) dup.push(n); top.add(n); }
    }
    stmt = '';
  };
  for (const ch of code) {
    if (ch === '(') depth++;
    if (ch === ')') depth--;
    if (depth > 0) { stmt += ch; continue; }
    if (ch === '{') { stmt = ''; scopes.push(new Set()); } else if (ch === '}') { stmt = ''; scopes.pop(); } else if (ch === ';') flush(); else stmt += ch;
  }
  return dup;
}
const lookups = (s, k) => s.split(`getShadow( directionalShadowMap[ ${k} ]`).length - 1;

// ---------------------------------------------------------------- 1. two cascades: byte-identical to 0.6.0
const SC = await import('../src/render/shadowCascade.js');
check('2-cascade text = 0.6.0 snapshot', SC.cascadeShadowChunkText(2) === EXPECTED_2);
check('stock chunk holds the line the patch replaces', STOCK_LIGHTS.includes(ORIG));
const { Atmosphere } = await import('../src/render/Atmosphere.js'); // installs the 2-cascade patch at module load
const expectedChunk = STOCK_LIGHTS.replace(ORIG, EXPECTED_2);
check('patched lights_fragment_begin = stock with the 0.6.0 block', THREE.ShaderChunk.lights_fragment_begin === expectedChunk);
if (THREE.REVISION === '180') check('patched chunk sha-256 (r180)', sha(THREE.ShaderChunk.lights_fragment_begin) === EXPECTED_2_SHA_R180);
check('installedCascades() after module load', SC.installedCascades() === 2);
const headless = (shadows, extra = {}) => new Atmosphere({ scene: new THREE.Scene(), renderer: null, camera: new THREE.PerspectiveCamera(), shadows, ...extra });
const a2 = headless({ cascades: 2 });
check('cascades: 2 Atmosphere leaves the chunk text unchanged', THREE.ShaderChunk.lights_fragment_begin === expectedChunk && SC.installedCascades() === 2);
check('installCascadeShadowChunk(2) again: no change', SC.installCascadeShadowChunk(2) === true && THREE.ShaderChunk.lights_fragment_begin === expectedChunk);
for (const n of [1, 2]) {
  const g = assembleLights(expectedChunk, n);
  check(`0.6.0 text, ${n} dir shadow(s): balanced, no redeclaration`, balanced(g, '{', '}') && balanced(g, '(', ')') && redeclarations(g).length === 0, redeclarations(g).join(','));
}

// ---------------------------------------------------------------- 2. three and four cascades: the general text
for (const max of [3, 4]) {
  const t = SC.cascadeShadowChunkText(max);
  check(`${max}-cascade text: balanced braces / parens / #if`, balanced(t, '{', '}') && balanced(t, '(', ')') && directivesBalanced(t));
  check(`${max}-cascade text: NUM_DIR_LIGHT_SHADOWS > ${max - 1} branch`, t.includes(`NUM_DIR_LIGHT_SHADOWS > ${max - 1}`) && !t.includes(`NUM_DIR_LIGHT_SHADOWS > ${max}`));
  check(`${max}-cascade text: black + parallel guard per cascade light`, [...Array(max - 1)].every((_, j) => t.includes(`dot( directionalLights[ ${j + 1} ].color, vec3( 1.0 ) ) <= 0.0 && dot( directionalLights[ ${j + 1} ].direction, directionalLight.direction ) > 0.9999`)));
  check(`${max}-cascade text: no stray loop index`, !/\[\s*i\s*\]/.test(t.replaceAll(ORIG, '')));
  const chunk = STOCK_LIGHTS.replace(ORIG, t);
  for (let n = 1; n <= 4; n++) {
    const g = assembleLights(chunk, n);
    const dup = redeclarations(g);
    check(`${max}-cascade text, ${n} dir shadow(s): balanced, no redeclaration`, balanced(g, '{', '}') && balanced(g, '(', ')') && dup.length === 0, dup.join(','));
    const counts = [0, 1, 2, 3].slice(0, n).map((k) => lookups(g, k));
    // index 0: cascade lookup of each map it walks + its stock fallback; index k > 0: the stock lookup (non-black)
    const want = counts.map((_, k) => (n === 1 ? 1 : k < max ? 2 : 1));
    check(`${max}-cascade text, ${n} dir shadow(s): one cascade lookup per map`, counts.join() === want.join(), counts.join());
    check(`${max}-cascade text, ${n} dir shadow(s): cascade count up to ${Math.min(n, max)}`, n === 1 ? !g.includes('rcN') : g.includes(`rcN = ${Math.min(n, max)};`) && !g.includes(`rcN = ${Math.min(n, max) + 1};`));
  }
}

// ---------------------------------------------------------------- 3. Atmosphere cascades (headless)
const order = (a) => { const o = []; a.scene.traverse((x) => { if (x.isDirectionalLight && x.castShadow) o.push(x); }); return o; };
check('2 cascades: sun + sunFar only, 0.6.0 boxes / maps', a2.cascadeCount === 2 && !a2.sunMid && a2.sunFar.name === 'sunFarCascade'
  && a2.sun.userData.half === 45 && a2.sunFar.userData.half === 260 && a2.sun.shadow.mapSize.x === 2048 && a2.sunFar.shadow.mapSize.x === 4096
  && a2.sun.shadow.radius === 2 && a2.sunFar.shadow.radius === 1.4 && a2.sunFar.shadow.normalBias === (520 / 4096) * 1.4 && a2.sunFar.shadow.bias === -0.00003);
const atmos = { 1: headless({ cascades: 1 }), 2: a2 };
// the upgrade must keep other patches of the chunk (e.g. clustered lights inserting before the spot loop)
const MARK = '\n// test: another patch of lights_fragment_begin\n';
THREE.ShaderChunk.lights_fragment_begin = MARK + THREE.ShaderChunk.lights_fragment_begin;
atmos[3] = headless({ cascades: 3 });
check('cascades: 3 upgrades the chunk to the 3-cascade text, other patches kept', SC.installedCascades() === 3
  && THREE.ShaderChunk.lights_fragment_begin === MARK + STOCK_LIGHTS.replace(ORIG, SC.cascadeShadowChunkText(3)));
atmos[4] = headless({ cascades: 4 });
check('cascades: 4 upgrades it again', SC.installedCascades() === 4 && THREE.ShaderChunk.lights_fragment_begin === MARK + STOCK_LIGHTS.replace(ORIG, SC.cascadeShadowChunkText(4)));
check('a later 2 / 3 request never downgrades', SC.installCascadeShadowChunk(3) && SC.installCascadeShadowChunk(2) && SC.installedCascades() === 4);
check('cascades: 7 clamps to 4', headless({ cascades: 7 }).cascadeCount === 4);

const NAMES = { 1: 'sun', 2: 'sun,sunFarCascade', 3: 'sun,sunMidCascade,sunFarCascade', 4: 'sun,sunMidCascade,sunMid2Cascade,sunFarCascade' };
const info = { count: 0, lights: [], cameras: [], matrices: [], halfSizes: [], depthTextures: [] };
for (let n = 1; n <= 4; n++) {
  const a = atmos[n], L = a.cascadeLights;
  check(`${n} cascade(s): lights in three's shadow order`, order(a).map((l) => l.name).join() === NAMES[n] && L.map((l) => l.name).join() === NAMES[n]);
  if (n > 1) {
    check(`${n} cascades: sunFar is the last cascade, sunMid / sunMid2 between`, L[n - 1] === a.sunFar && (n < 3 || L[1] === a.sunMid) && (n < 4 || L[2] === a.sunMid2));
    check(`${n} cascades: extra lights black, shadow-only, autoUpdate off`, L.slice(1).every((l) => l.intensity === 0 && l.color.getHex() === 0 && l.castShadow && l.shadow.autoUpdate === false));
    check(`${n} cascades: boxes grow outwards, normalBias 1.4 texel`, L.every((l, i) => (i === 0 || l.userData.half > L[i - 1].userData.half) && Math.abs(l.shadow.normalBias - (2 * l.userData.half / l.shadow.mapSize.x) * 1.4) < 1e-12));
  }
  const r = a.shadowCascadeInfo(info);
  check(`${n} cascade(s): shadowCascadeInfo shape`, r === info && r.count === n && ['lights', 'cameras', 'matrices', 'halfSizes', 'maps', 'depthTextures'].every((k) => Array.isArray(r[k]) && r[k].length === n)
    && r.lights.every((l, i) => l === L[i] && r.cameras[i] === l.shadow.camera && r.matrices[i] === l.shadow.matrix && r.matrices[i].isMatrix4 && r.halfSizes[i] === l.userData.half)
    && r.maps.every((m) => m === null) && r.depthTextures.every((m) => m === null) && r.packed === true);
}
const arrays = ['lights', 'matrices', 'maps'].map((k) => info[k]);
atmos[4].sunMid.castShadow = false;
atmos[4].shadowCascadeInfo(info);
check('shadowCascadeInfo reuses its arrays and skips non-casting cascades', info.count === 3 && info.lights.length === 3 && ['lights', 'matrices', 'maps'].every((k, i) => info[k] === arrays[i]) && info.lights[1] === atmos[4].sunMid2);
atmos[4].sunMid.castShadow = true;

// round-robin map updates (the near map auto-updates every frame): the far map on even frames for 2 cascades as in
// 0.6.0; all on the first (dirty) frame
const SCHED = { 2: '01 01 00 01 00 01 00 01', 3: '011 001 010 001 010 001 010 001', 4: '0111 0001 0100 0010 0100 0001 0100 0010' };
for (let n = 2; n <= 4; n++) {
  const a = headless({ cascades: n }), p = new THREE.Vector3(0, 2, 0), log = [];
  for (let f = 0; f < 8; f++) {
    for (const l of a.cascadeLights) l.shadow.needsUpdate = false;
    a.update(p, 1 / 60);
    log.push(a.cascadeLights.map((l) => (l.shadow.needsUpdate ? 1 : 0)).join(''));
  }
  check(`${n} cascades: map update schedule`, log.join(' ') === SCHED[n], log.join(' '));
  a.setTime(12); // dirty: every cascade re-renders on the next frame
  for (const l of a.cascadeLights) l.shadow.needsUpdate = false;
  a.update(p, 1 / 60);
  check(`${n} cascades: a time change re-renders every map`, a.cascadeLights.slice(1).every((l) => l.shadow.needsUpdate));
}

// boxes by camera height (hysteresis 70 / 200 up, 50 / 160 down); middle boxes step with the far box
{
  const a = headless({ cascades: 4 }), p = new THREE.Vector3(0, 2, 0);
  const halves = () => a.cascadeLights.map((l) => l.userData.half).join(',');
  const seen = [];
  for (const y of [2, 80, 250, 180, 150, 60, 40]) { p.y = y; a.update(p, 1 / 60); seen.push(halves()); }
  check('4 cascades: far + middle boxes step with height (hysteresis)', seen.join(' | ') === '45,65,90,260 | 45,75,125,420 | 45,85,170,650 | 45,85,170,650 | 45,75,125,420 | 45,75,125,420 | 45,65,90,260', seen.join(' | '));
  const b = headless({ cascades: 3, mid: [[100, 130, 160]] });
  p.set(0, 2, 0); b.update(p, 1 / 60);
  check('shadows.mid overrides the middle boxes (one step per frame)', b.sunMid.userData.half === 100 && (p.y = 250, b.update(p, 1 / 60), b.sunMid.userData.half === 130) && (b.update(p, 1 / 60), b.sunMid.userData.half === 160));
  check('3 cascades: default middle boxes [75, 95, 120]', JSON.stringify(atmos[3].midHalfSteps) === '[[75,95,120]]');
}

// ---------------------------------------------------------------- 4. the 'pcss' kernel chunk
const K = await import('../src/render/shadowKernel.js');
check('SHADOW_KERNELS lists pcf, fast, pcss', K.SHADOW_KERNELS.join() === 'pcf,fast,pcss' && K.PCSS_SHADOW_DEFINE === 'ENGINE_SHADOW_PCSS');
const G = K.PCSS_SHADOW_GLSL.trim();
check('pcss GLSL wrapped in its define', G.startsWith('#if defined( USE_SHADOWMAP ) && defined( ENGINE_SHADOW_PCSS ) && defined( SHADOWMAP_TYPE_PCF ) && NUM_DIR_LIGHT_SHADOWS > 0') && G.endsWith('#endif') && directivesBalanced(G) && balanced(G, '{', '}') && balanced(G, '(', ')'));
check('pcss GLSL inert without the define', preprocess(G, { USE_SHADOWMAP: '', SHADOWMAP_TYPE_PCF: '', NUM_DIR_LIGHT_SHADOWS: 2 }).trim() === '');
for (let n = 1; n <= 4; n++) {
  const g = preprocess(G, { USE_SHADOWMAP: '', SHADOWMAP_TYPE_PCF: '', ENGINE_SHADOW_PCSS: '', NUM_DIR_LIGHT_SHADOWS: n });
  check(`pcss GLSL with the define, ${n} dir shadow(s): overload + cascade picks`, g.includes('float getShadow( int cascade,') && g.includes('#define directionalShadowMap enginePcssIndex')
    && [0, 1, 2, 3].every((k) => g.includes(`directionalShadowMap[ ${k} ], engineDirShadowDepth[ ${k} ], engineDirShadowPcss[ ${k} ]`) === k < n) && redeclarations(g).length === 0);
}
check('pcss GLSL off above 4 dir shadows', preprocess(G, { USE_SHADOWMAP: '', SHADOWMAP_TYPE_PCF: '', ENGINE_SHADOW_PCSS: '', NUM_DIR_LIGHT_SHADOWS: 5 }).trim() === '');
check('shadowmap_pars_fragment untouched before a kernel is installed', THREE.ShaderChunk.shadowmap_pars_fragment === STOCK_PARS);
check('installPcssShadowChunk appends the pcss GLSL', K.installPcssShadowChunk() === true && THREE.ShaderChunk.shadowmap_pars_fragment === STOCK_PARS + K.PCSS_SHADOW_GLSL && K.installPcssShadowChunk() === true && THREE.ShaderChunk.shadowmap_pars_fragment === STOCK_PARS + K.PCSS_SHADOW_GLSL);
const SU = THREE.ShaderLib.standard.uniforms;
check('pcss template uniforms on the lit templates', !!(SU.engineDirShadowDepth && SU.engineDirShadowPcss && THREE.ShaderLib.lambert.uniforms.engineDirShadowPcss && THREE.ShaderLib.shadow.uniforms.engineDirShadowPcss && !THREE.ShaderLib.basic.uniforms.engineDirShadowPcss));
const cloned = THREE.UniformsUtils.clone(SU);
check('pcss uniform values shared by reference through clone', cloned.engineDirShadowPcss.value === SU.engineDirShadowPcss.value && cloned.engineDirShadowDepth.value === SU.engineDirShadowDepth.value && [0, 1, 2, 3].every((i) => SU.engineDirShadowPcss.value[i].isVector4));
K.setPcssShadowParams([new THREE.Vector4(1, 2, 3, 4)], 1);
check('setPcssShadowParams fills in order, clears the rest', SU.engineDirShadowPcss.value[0].equals(new THREE.Vector4(1, 2, 3, 4)) && SU.engineDirShadowPcss.value[1].equals(new THREE.Vector4(0, 0, 0, 0)));

// per-material selection (MeshStandardMaterial keeps its own STANDARD define)
const m = new THREE.MeshStandardMaterial();
const v0 = m.version;
const engineDefines = (mat) => Object.keys(mat.defines || {}).filter((k) => k.startsWith('ENGINE_')).join();
check("'pcss' sets ENGINE_SHADOW_PCSS only", K.setMaterialShadowKernel(m, 'pcss') && engineDefines(m) === 'ENGINE_SHADOW_PCSS' && m.version > v0);
check("'fast' without its chunk installed selects 'pcf'", K.setMaterialShadowKernel(m, 'fast') && engineDefines(m) === '' && Object.keys(m.defines).join() === 'STANDARD');
K.installFastShadowChunk();
check('fast + pcss chunks both appended, fast GLSL unchanged in front', THREE.ShaderChunk.shadowmap_pars_fragment.startsWith(STOCK_PARS + K.PCSS_SHADOW_GLSL) && THREE.ShaderChunk.shadowmap_pars_fragment.includes('#if defined( USE_SHADOWMAP ) && defined( ENGINE_SHADOW_FAST )'));
check("'fast' sets ENGINE_SHADOW_FAST only", K.setMaterialShadowKernel(m, 'fast') && engineDefines(m) === 'ENGINE_SHADOW_FAST');
check("'pcss' replaces 'fast'", K.setMaterialShadowKernel(m, 'pcss') && engineDefines(m) === 'ENGINE_SHADOW_PCSS');
check("'pcss' again: no change", K.setMaterialShadowKernel(m, 'pcss') === false);
check("'pcf' clears the defines (compiles like none)", K.setMaterialShadowKernel(m, 'pcf') && engineDefines(m) === '' && Object.keys(m.defines).join() === 'STANDARD');
check('unlit materials never take a kernel', K.setMaterialShadowKernel(new THREE.MeshBasicMaterial(), 'pcss') === false);
const sm = new THREE.ShaderMaterial({ lights: true, uniforms: { engineDirShadowDepth: SU.engineDirShadowDepth } });
check("a lights ShaderMaterial without the pcss uniform takes 'fast', not 'pcss'", K.setMaterialShadowKernel(sm, 'pcss') === false && K.setMaterialShadowKernel(sm, 'fast') === true);
const scene = new THREE.Scene();
scene.add(new THREE.Mesh(new THREE.BoxGeometry(), [new THREE.MeshStandardMaterial(), new THREE.MeshLambertMaterial()]), new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial()));
check('applyShadowKernel counts switched materials', K.applyShadowKernel(scene, 'pcss') === 2 && K.applyShadowKernel(scene, 'pcss') === 0 && K.applyShadowKernel(scene, 'pcf') === 2);

// Atmosphere side: headless falls back to 'pcf' (warns); the pcss constants follow the boxes
const warn = console.warn; const warned = [];
console.warn = (...a) => warned.push(a.join(' '));
const ah = headless({ cascades: 3, kernel: 'pcss' });
check("headless: kernel 'pcss' falls back to 'pcf' with a warning", ah.shadowKernel === 'pcf' && warned.some((w) => w.includes("pcss shadow kernel unavailable (no renderer), using 'pcf'")));
check('unknown kernel: warning, nothing switched', ah.setShadowKernel('vsm') === 0 && warned.some((w) => w.includes("unknown shadow kernel 'vsm'")));
console.warn = warn;
ah.update(new THREE.Vector3(0, 2, 0), 1 / 60);
ah._syncPcss();
{
  const P = SU.engineDirShadowPcss.value, L = ah.cascadeLights, tan = Math.tan(0.5 * Math.PI / 180), tN = 90 / 2048;
  const ok = L.every((l, i) => {
    const texel = (2 * l.userData.half) / l.shadow.mapSize.x, range = l.shadow.camera.far - l.shadow.camera.near;
    return Math.abs(P[i].x - range * tan / texel) < 1e-6 * P[i].x && Math.abs(P[i].y - 0.5 * Math.min(24, 24 * tN / texel)) < 1e-9 && Math.abs(P[i].z - 8 * texel / range) < 1e-12 && P[i].w === 0;
  });
  check('pcss constants per cascade (penumbra scale, radius cap = 24 near texels, slope limit)', ok && P[0].y === 12, [0, 1, 2].map((i) => `${P[i].x.toFixed(1)}/${P[i].y.toFixed(2)}`).join(' '));
}

// ---------------------------------------------------------------- 5. robustness (review fixes)
// each box holds the previous one entirely (near in middle, middle in far), at every height step, sun and heading:
// the corners of box k (mid-depth) projected by cascade k+1's shadow matrix land inside its map
const corner = new THREE.Vector3();
function containment(a) {
  const L = a.cascadeLights;
  for (const l of L) l.shadow.updateMatrices(l); // what three's shadow pass computes
  for (let k = 0; k + 1 < L.length; k++) {
    const cam = L[k].shadow.camera, M = L[k + 1].shadow.matrix;
    for (const [sx, sy] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) {
      corner.set(sx, sy, 0).unproject(cam).applyMatrix4(M);
      if (!(corner.x >= 0 && corner.x <= 1 && corner.y >= 0 && corner.y <= 1 && corner.z >= 0 && corner.z <= 1)) return `box ${k} corner ${sx},${sy} at ${corner.x.toFixed(3)},${corner.y.toFixed(3)} of box ${k + 1}`;
    }
  }
  return '';
}
for (const n of [2, 3, 4]) {
  const fails = [];
  for (const hours of [17.55, 13, 8]) {
    const a = headless({ cascades: n }, { timeOfDay: hours }), p = new THREE.Vector3();
    for (const yaw of [0, 1.1, 2.6, 4.2]) {
      for (const y of [2, 45, 69, 75, 150, 199, 250, 180, 100, 60, 30]) {
        a.camera.rotation.set(-0.15, yaw, 0); a.camera.updateMatrixWorld();
        p.set(37.3 * yaw, y, -11.7 * yaw);
        for (let i = 0; i < 3; i++) { a._farDirty = true; a.update(p, 1 / 60); }
        const r = containment(a);
        if (r) fails.push(`${hours} h, yaw ${yaw}, y ${y}: ${r}`);
      }
    }
  }
  check(`${n} cascades: every box holds the previous one (heights 2-250 m, 3 suns, 4 headings)`, fails.length === 0, fails.slice(0, 2).join(' | '));
}

// a jump refits every middle / far map at once (3+ cascades); 2 cascades keep the 0.6.0 schedule
for (const n of [2, 4]) {
  const a = headless({ cascades: n }), p = new THREE.Vector3(0, 2, 0);
  for (let i = 0; i < 8; i++) a.update(p, 1 / 60);
  for (const l of a.cascadeLights) l.shadow.needsUpdate = false;
  p.set(800, 2, -600); a.update(p, 1 / 60); // frame 9: odd, only the first middle map is due by the schedule
  const upd = a.cascadeLights.slice(1).map((l) => (l.shadow.needsUpdate ? 1 : 0)).join('');
  check(`${n} cascades: a teleport on an odd frame refits ${n === 2 ? 'nothing early (0.6.0)' : 'all middle / far maps'}`, upd === (n === 2 ? '0' : '111'), upd);
}

// shadow-casting directional lights that are not the cascades: before the sun (build warns), added later (the
// kernel falls back to 'pcf'); the texture unit budget of 'pcss'
const FAKE = (maxTextures) => ({ shadowMap: { type: THREE.PCFShadowMap }, capabilities: { maxTextures, maxTextureSize: 4096 } });
const warned2 = [];
console.warn = (...a) => warned2.push(a.join(' '));
{
  const s0 = new THREE.Scene(), pre = new THREE.DirectionalLight();
  pre.name = 'gameSun'; pre.castShadow = true; s0.add(pre);
  new Atmosphere({ scene: s0, renderer: null, shadows: { cascades: 2 } });
  check('build warns about a shadow-casting directional light before the sun', warned2.some((w) => w.includes("'gameSun' precedes the sun")));
  warned2.length = 0;
  const s1 = new THREE.Scene(), off = new THREE.DirectionalLight();
  s1.add(off); // castShadow false: not a shadow
  new Atmosphere({ scene: s1, renderer: null, shadows: { cascades: 2 } });
  check('no warning for a non-casting light before the sun', !warned2.some((w) => w.includes('precedes')));

  const af = headless({ cascades: 3 });
  af.renderer = FAKE(16); // after build: setShadowKernel only reads shadowMap.type and capabilities
  check("fake PCF renderer: 'fast' selected", af.setShadowKernel('fast') >= 0 && af.shadowKernel === 'fast');
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshStandardMaterial());
  af.scene.add(mesh);
  check("objects added while 'fast' follow", engineDefines(mesh.material) === 'ENGINE_SHADOW_FAST');
  const quiet = new THREE.DirectionalLight();
  af.scene.add(quiet);
  check("a non-casting directional light added later keeps 'fast'", af.shadowKernel === 'fast');
  const g = new THREE.Group(), lamp = new THREE.DirectionalLight();
  lamp.name = 'headlamp'; lamp.castShadow = true; g.add(lamp);
  af.scene.add(g);
  check("a shadow-casting directional light added later (nested) falls back to 'pcf'", af.shadowKernel === 'pcf' && engineDefines(mesh.material) === ''
    && warned2.some((w) => w.includes("fast shadow kernel unavailable (another shadow-casting directional light 'headlamp' was added), using 'pcf'")));
  af.scene.remove(g);
  check("without it 'fast' is available again", af.setShadowKernel('fast') > 0 && af.shadowKernel === 'fast');

  warned2.length = 0;
  const a4 = headless({ cascades: 4 }), a3 = headless({ cascades: 3 });
  a4.renderer = FAKE(16); a3.renderer = FAKE(16);
  a4.setShadowKernel('pcss'); a3.setShadowKernel('pcss');
  check("'pcss' with 4 cascades on 16 texture units falls back to 'pcf'", a4.shadowKernel === 'pcf' && warned2.some((w) => w.includes('8 shadow texture units leave 8 or fewer of 16 for materials')));
  check("'pcss' with 3 cascades on 16 units is available", a3.shadowKernel === 'pcss');
  a4.renderer = FAKE(32); a4.setShadowKernel('pcss');
  check("'pcss' with 4 cascades on 32 units is available", a4.shadowKernel === 'pcss');
}
console.warn = warn;

console.error(results.join('\n'));
console.error(failures ? `\n${failures} shadow check(s) FAILED` : '\nAll shadow checks passed.');
process.exit(failures ? 1 : 0);
