// G-buffer chunk patch + material classification + instance velocity, headless (no WebGL): the anchors the patch
// needs exist in this three.js, every inserted block preprocesses away without its define (stock shaders stay
// byte-identical up to the preprocessor), the expanded GLSL is sane as text (balanced braces, declarations before
// use, one main), GBuffer.sync() / track() classify and patch a hand-built scene as documented, the draw-buffer mask
// hooks keep three's cached state, and InstancedPool({ velocity: true }) keeps / captures previous matrices.
// Run: node test/gbuffer-chunks.mjs (npm test runs it). Real compilation and rendering are checked in the browser.
import * as THREE from 'three';

const PATCHED = ['clipping_planes_pars_vertex', 'project_vertex', 'clipping_planes_pars_fragment', 'normal_fragment_maps',
  'roughnessmap_fragment', 'metalnessmap_fragment', 'dithering_fragment'];
const ORIG = Object.fromEntries(PATCHED.map((k) => [k, THREE.ShaderChunk[k]]));
const G = await import('../src/render/GBuffer.js');
const { InstancedPool } = await import('../src/render/InstancedPool.js');

let failures = 0, passes = 0;
function check(name, ok, detail = '') {
  if (ok) passes++; else { failures++; console.log(`FAIL  ${name}${detail ? `  (${detail})` : ''}`); }
  return ok;
}
const warns = [];
console.warn = (...a) => warns.push(a.join(' '));

// ---- a minimal GLSL preprocessor: evaluates only conditionals that name ENGINE_* / gl_FragColor --------------------
function evalCond(kind, rest, defs) {
  if (kind === 'ifdef') return defs.has(rest.trim());
  if (kind === 'ifndef') return !defs.has(rest.trim());
  let e = rest.replace(/defined\s*\(\s*(\w+)\s*\)/g, (_, n) => (defs.has(n) ? '1' : '0')).replace(/defined\s+(\w+)/g, (_, n) => (defs.has(n) ? '1' : '0'));
  if (!/^[\s01!&|()]*$/.test(e)) throw new Error(`unsupported condition: ${rest}`);
  return !!Function(`return (${e});`)();
}
function preprocess(src, defsIn) {
  const defs = new Set(defsIn), out = [], stack = [];
  const active = (n = stack.length) => { for (let i = 0; i < n; i++) if (!stack[i].active) return false; return true; };
  for (const line of src.split('\n')) {
    const m = line.match(/^\s*#\s*(ifdef|ifndef|if|elif|else|endif)\b(.*)$/);
    if (!m) {
      if (!active()) continue;
      const d = line.match(/^\s*#\s*define\s+(ENGINE_\w+)/);
      if (d) defs.add(d[1]);
      out.push(line);
      continue;
    }
    const [, kind, rest] = m;
    if (kind === 'if' || kind === 'ifdef' || kind === 'ifndef') {
      const ours = /ENGINE_|gl_FragColor/.test(rest);
      if (ours) { const v = evalCond(kind, rest, defs); stack.push({ ours, active: v, taken: v }); }
      else { if (active()) out.push(line); stack.push({ ours, active: true }); }
    } else if (kind === 'elif' || kind === 'else') {
      const f = stack[stack.length - 1];
      if (f.ours) { const v = !f.taken && (kind === 'else' || evalCond('if', rest, defs)); f.active = v; f.taken = f.taken || v; }
      else if (active()) out.push(line);
    } else {
      const f = stack.pop();
      if (!f.ours && active()) out.push(line);
    }
  }
  if (stack.length) throw new Error('unbalanced #if');
  return out.join('\n');
}
const resolveIncludes = (s) => s.replace(/^[ \t]*#include +<([\w\d./]+)>/gm, (_, n) => resolveIncludes(THREE.ShaderChunk[n]));
const balanced = (s, o, c) => { let d = 0; for (const ch of s) { if (ch === o) d++; else if (ch === c && --d < 0) return false; } return d === 0; };
// net depth: stock chunks keep alternative #ifdef branches with an opening brace each (lights_physical_pars_fragment),
// so whole expanded shaders are compared with their stock expansion instead of being required to balance
const net = (s, o, c) => { let d = 0; for (const ch of s) { if (ch === o) d++; else if (ch === c) d--; } return d; };
const sameNesting = (a, b) => net(a, '{', '}') === net(b, '{', '}') && net(a, '(', ')') === net(b, '(', ')');
const before = (s, a, b) => { const i = s.indexOf(a), j = s.indexOf(b); return i >= 0 && j >= 0 && i < j; };
const count = (s, sub) => s.split(sub).length - 1;

// ---- anchors in this three.js -------------------------------------------------------------------------------------
const L = THREE.ShaderLib;
const mainAt = (s) => s.search(/void\s+main\s*\(/);
const incAt = (s, c) => s.indexOf(`#include <${c}>`);
for (const k of ['basic', 'lambert', 'phong', 'standard', 'physical', 'toon', 'matcap']) {
  const v = L[k].vertexShader, f = L[k].fragmentShader;
  check(`${k}: clipping_planes_pars_vertex at global scope`, incAt(v, 'clipping_planes_pars_vertex') >= 0 && incAt(v, 'clipping_planes_pars_vertex') < mainAt(v));
  check(`${k}: project_vertex in main`, incAt(v, 'project_vertex') > mainAt(v));
  check(`${k}: clipping_planes_pars_fragment at global scope`, incAt(f, 'clipping_planes_pars_fragment') >= 0 && incAt(f, 'clipping_planes_pars_fragment') < mainAt(f));
  const incs = f.match(/#include <\w+>/g);
  check(`${k}: dithering_fragment is the last fragment chunk`, incs[incs.length - 1] === '#include <dithering_fragment>');
  if (k !== 'basic') check(`${k}: normal_fragment_maps after normal_fragment_begin`, incAt(f, 'normal_fragment_maps') > incAt(f, 'normal_fragment_begin') && incAt(f, 'normal_fragment_begin') > mainAt(f));
}
for (const k of ['standard', 'physical']) {
  const f = L[k].fragmentShader;
  check(`${k}: roughness / metalness chunks before the writes`, incAt(f, 'roughnessmap_fragment') > mainAt(f) && incAt(f, 'metalnessmap_fragment') > mainAt(f) && incAt(f, 'metalnessmap_fragment') < incAt(f, 'dithering_fragment'));
}
for (const k of ['points', 'sprite', 'dashed', 'depth', 'distanceRGBA', 'normal']) {
  const f = L[k].fragmentShader;
  check(`${k}: zero-writer host chunk at global scope`, incAt(f, 'clipping_planes_pars_fragment') >= 0 && incAt(f, 'clipping_planes_pars_fragment') < mainAt(f));
}
check('shadow: no host chunk (ShadowMaterial is masked)', incAt(L.shadow.fragmentShader, 'clipping_planes_pars_fragment') < 0);
check('project_vertex anchors', ORIG.project_vertex.includes('mvPosition = modelViewMatrix * mvPosition;') && ORIG.project_vertex.includes('gl_Position = projectionMatrix * mvPosition;'));
check('roughnessFactor / metalnessFactor declared by their chunks', ORIG.roughnessmap_fragment.includes('float roughnessFactor') && ORIG.metalnessmap_fragment.includes('float metalnessFactor'));
check('normal_fragment_begin declares normal', THREE.ShaderChunk.normal_fragment_begin.includes('vec3 normal ='));

// ---- install: inert without the defines ---------------------------------------------------------------------------
check('installGBufferChunks() true', G.installGBufferChunks() === true);
check('installGBufferChunks() idempotent', G.installGBufferChunks() === true && count(THREE.ShaderChunk.project_vertex, 'vEngineClipCur =') === 1);
for (const k of PATCHED) {
  const p = THREE.ShaderChunk[k];
  check(`${k}: patched = original + block`, p.startsWith(ORIG[k]) && p.length > ORIG[k].length);
  check(`${k}: preprocesses to the original without a define`, preprocess(p, []) === ORIG[k]);
  check(`${k}: ...and without ENGINE_* but with gl_FragColor / USE_INSTANCING`, preprocess(p, ['gl_FragColor', 'USE_INSTANCING', 'USE_BATCHING']) === ORIG[k]);
  const t = G.GBUFFER_CHUNK_TAILS[k];
  check(`${k}: tail inert alone`, preprocess(t, ['gl_FragColor']).trim() === '');
  check(`${k}: tail starts / ends with a newline`, t.startsWith('\n') && t.endsWith('\n'));
  check(`${k}: tail braces / parens / #if balanced`, balanced(t, '{', '}') && balanced(t, '(', ')') && count(t, '#endif') === (t.match(/^\s*#\s*if/gm) || []).length);
  const firsts = t.split('\n').filter((l) => /^#/.test(l)); // top-level directives (blocks are not indented at top level)
  check(`${k}: every top-level block is gated by an ENGINE_GBUFFER* define`, firsts.filter((l) => /^#if/.test(l)).every((l) => /ENGINE_GBUFFER/.test(l)));
}
check('GBUFFER_VERTEX_TAIL is the project_vertex tail', G.GBUFFER_VERTEX_TAIL === G.GBUFFER_CHUNK_TAILS.project_vertex);
check('decode GLSL balanced', balanced(G.GBUFFER_GLSL_DECODE, '{', '}') && balanced(G.GBUFFER_GLSL_DECODE, '(', ')'));
for (const fn of ['vec3 engineOctDecode( vec2 e )', 'float engineGbMetal( float a )', 'int engineGbFlags( float a )']) check(`decode has ${fn}`, G.GBUFFER_GLSL_DECODE.includes(fn));
for (const k of ['standard', 'basic', 'points', 'sprite', 'depth']) check(`ShaderLib.${k} carries the shared view uniforms`, !!(L[k].uniforms.engineUnjitteredProj && L[k].uniforms.engineViewToPrevClip));
check('ShaderLib.shadow untouched (no host chunk)', !L.shadow.uniforms.engineViewToPrevClip);
{
  // UniformsUtils.clone keeps the template value by reference and it reads the live matrix
  const u = THREE.UniformsUtils.clone(L.physical.uniforms);
  G.GBUFFER_UNIFORMS.engineUnjitteredProj.value.makeTranslation(1, 2, 3);
  check('cloned template uniform follows GBUFFER_UNIFORMS', u.engineUnjitteredProj.value.elements[13] === 2 && u.engineUnjitteredProj.value === L.physical.uniforms.engineUnjitteredProj.value);
  G.GBUFFER_UNIFORMS.engineUnjitteredProj.value.identity();
}

// ---- expanded shaders with the defines -----------------------------------------------------------------------------
{
  const fs = preprocess(resolveIncludes(L.physical.fragmentShader), ['ENGINE_GBUFFER', 'gl_FragColor']);
  check('physical fs: one main', count(fs, 'void main') === 1 && !fs.includes('engineGbUserMain'));
  check('physical fs: outputs declared before main', before(fs, 'layout( location = 1 ) out highp vec4 engineGbNormal;', 'void main') && before(fs, 'layout( location = 2 ) out highp vec4 engineGbVelocity;', 'void main'));
  check('physical fs: globals + encoder before main', before(fs, 'vec3 engineGbN = vec3( 0.0 );', 'void main') && before(fs, 'vec2 engineOctEncode( vec3 n )', 'void main'));
  check('physical fs: roughness / metalness captured after their declaration', before(fs, 'float roughnessFactor', 'engineGbRoughness = roughnessFactor;') && before(fs, 'float metalnessFactor', 'engineGbMetalness = metalnessFactor;'));
  check('physical fs: normal captured after normal_fragment_maps', before(fs, 'vec3 normal =', 'engineGbN = normal;'));
  check('physical fs: writes at the end of main', before(fs, 'engineGbN = normal;', 'engineGbNormal = vec4(') && fs.lastIndexOf('engineGbVelocity = vec4(') > fs.lastIndexOf('gl_FragColor'));
  const stockFs = preprocess(resolveIncludes(L.physical.fragmentShader), ['gl_FragColor']);
  check('physical fs: additions keep the brace / paren nesting', sameNesting(fs, stockFs));
  const g3 = preprocess(resolveIncludes(L.physical.fragmentShader), ['ENGINE_GBUFFER']);
  check('GLSL3 (no gl_FragColor): outputs not declared, writes kept', !g3.includes('out highp vec4 engineGbNormal') && g3.includes('engineGbNormal = vec4('));
  const novel = preprocess(resolveIncludes(L.physical.fragmentShader), ['ENGINE_GBUFFER', 'ENGINE_GBUFFER_NOVEL', 'gl_FragColor']);
  check('NOVEL: no velocity math, noHistory flag', !novel.includes('vEngineClipCur.xy / vEngineClipCur.w') && novel.includes('engineGbFlagSum += 8.0;'));
  const basic = preprocess(resolveIncludes(L.basic.fragmentShader), ['ENGINE_GBUFFER', 'gl_FragColor']);
  check('basic fs: writes without a normal (engineGbN stays 0: encoded 0.5)', basic.includes('engineGbNormal = vec4(') && !basic.includes('engineGbN = normal;'));

  const vs = preprocess(resolveIncludes(L.physical.vertexShader), ['ENGINE_GBUFFER', 'ENGINE_GBUFFER_PREV', 'USE_INSTANCING', 'ENGINE_GBUFFER_INSTANCE_PREV']);
  check('physical vs: declarations before main', before(vs, 'attribute mat4 instancePrevMatrix;', 'void main') && before(vs, 'uniform mat4 enginePrevModelView;', 'void main') && before(vs, 'uniform mat4 engineViewToPrevClip;', 'void main'));
  check('physical vs: clip positions after the final mvPosition', before(vs, 'mvPosition = modelViewMatrix * mvPosition;', 'vEngineClipPrev = engineViewToPrevClip *') && before(vs, 'mvPosition = modelViewMatrix * mvPosition;', 'vEngineClipCur = engineUnjitteredProj * mvPosition;'));
  check('physical vs: instance + model motion', vs.includes('( instancePrevMatrix - instanceMatrix ) * engineGbL') && vs.includes('( enginePrevModelView - modelViewMatrix ) * engineGbL'));
  check('physical vs: additions keep the brace / paren nesting', sameNesting(vs, preprocess(resolveIncludes(L.physical.vertexShader), [])));
  const st = preprocess(resolveIncludes(L.physical.vertexShader), ['ENGINE_GBUFFER']);
  check('static vs: camera-only (no displacement terms)', !st.includes('engineGbDL') && st.includes('vEngineClipPrev = engineViewToPrevClip * vec4( mvPosition.xyz + engineGbD, 1.0 );'));

  const pz = preprocess(resolveIncludes(L.points.fragmentShader), ['ENGINE_GBUFFER_ZERO', 'gl_FragColor']);
  check('points ZERO: wrapper main, then the renamed user main', before(pz, 'void engineGbUserMain();', 'void main() {') && before(pz, 'void main() {', '#define main engineGbUserMain') && count(pz, 'void main()') === 2);
  check('points ZERO: zero writes in the wrapper', pz.includes('engineGbNormal = vec4( 0.0 );') && pz.includes('engineGbVelocity = vec4( 0.0 );'));
  check('points ZERO: additions keep the brace / paren nesting', sameNesting(pz, preprocess(resolveIncludes(L.points.fragmentShader), ['gl_FragColor'])));
  const both = preprocess(resolveIncludes(L.basic.fragmentShader), ['ENGINE_GBUFFER', 'ENGINE_GBUFFER_ZERO', 'gl_FragColor']);
  check('writer define wins over ZERO', !both.includes('engineGbUserMain') && both.includes('engineGbNormal = vec4( engineGbE'));
  const g3z = preprocess(resolveIncludes(L.points.fragmentShader), ['ENGINE_GBUFFER_ZERO']);
  check('ZERO inert without gl_FragColor (GLSL3)', !g3z.includes('engineGbUserMain'));
}

// ---- octahedral normal + flag packing (JS mirror of the GLSL) -----------------------------------------------------
{
  const sgn = (v) => (v >= 0 ? 1 : -1);
  const enc = (x, y, z) => {
    const s = Math.abs(x) + Math.abs(y) + Math.abs(z); x /= s; y /= s; z /= s;
    return z >= 0 ? [x, y] : [(1 - Math.abs(y)) * sgn(x), (1 - Math.abs(x)) * sgn(y)];
  };
  const dec = (ex, ey) => {
    const fx = ex * 2 - 1, fy = ey * 2 - 1; let x = fx, y = fy; const z = 1 - Math.abs(fx) - Math.abs(fy);
    const t = Math.max(-z, 0); x += x >= 0 ? -t : t; y += y >= 0 ? -t : t;
    const l = Math.hypot(x, y, z); return [x / l, y / l, z / l];
  };
  const q8 = (v) => Math.round(v * 255) / 255;
  let worst = 0, seed = 7;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647) * 2 - 1;
  const dirs = [[0, 0, 1], [0, 0, -1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0.577, -0.577, -0.577]];
  for (let i = 0; i < 20000; i++) dirs.push([rnd(), rnd(), rnd()]);
  for (const d of dirs) {
    const l = Math.hypot(...d); if (l < 1e-3) continue;
    const n = d.map((v) => v / l), e = enc(...n), r = dec(q8(e[0] * 0.5 + 0.5), q8(e[1] * 0.5 + 0.5));
    worst = Math.max(worst, Math.acos(Math.min(1, n[0] * r[0] + n[1] * r[1] + n[2] * r[2])) * 180 / Math.PI);
  }
  check('octahedral RGBA8 round trip < 1 degree', worst < 1, `${worst.toFixed(3)} deg`);
  let packOk = true;
  for (let mi = 0; mi <= 20; mi++) for (const flags of [0, 2, 8, 10]) {
    const metal = mi / 20, a = (Math.floor(metal * 15 + 0.5) * 16 + flags + 1) / 255;
    const byte = Math.round(q8(a) * 255);
    if ((byte & 15) !== flags + 1 || Math.abs((byte >> 4) / 15 - metal) > 1 / 30 + 1e-9) packOk = false;
  }
  check('metalness / flags packing round trip (bit0 always set by a writer)', packOk);
}

// ---- GBuffer: target -------------------------------------------------------------------------------------------
const fakeRenderer = (o = {}) => {
  const calls = [];
  const gl = {
    COLOR: 0x1800,
    drawBuffers: (l) => calls.push(['drawBuffers', l.slice()]),
    clearBufferfv: (b, i, v) => calls.push(['clearBufferfv', b, i, Array.from(v)]),
  };
  let rt = o.rt ?? null;
  return {
    calls, gl,
    extensions: { has: (n) => (o.float ?? true) && n === 'EXT_color_buffer_float' },
    getDrawingBufferSize: (v) => v.set(64, 32),
    getContext: () => gl,
    getRenderTarget: () => rt,
    setRenderTarget: (t, face = 0, mip = 0) => { rt = t; calls.push(['setRenderTarget', t, face, mip]); },
    getActiveCubeFace: () => 0, getActiveMipmapLevel: () => 0,
    state: { buffers: { color: { setMask: (m) => calls.push(['colorMask', m]) } } },
    get rt() { return rt; }, set rt(v) { rt = v; },
  };
};
{
  const gb = new G.GBuffer(fakeRenderer());
  const t = gb.target, [c, n, v] = t.textures;
  check('target: 3 attachments, samples 0, 64x32', t.textures.length === 3 && t.samples === 0 && t.width === 64 && t.height === 32);
  check('target: colour R11G11B10F with EXT_color_buffer_float', c.type === THREE.UnsignedInt101111Type && c.format === THREE.RGBFormat);
  check('target: normal RGBA8, NoColorSpace, nearest', n.type === THREE.UnsignedByteType && n.format === THREE.RGBAFormat && n.colorSpace === THREE.NoColorSpace && n.minFilter === THREE.NearestFilter && n.magFilter === THREE.NearestFilter);
  check('target: velocity RG16F, nearest', v.type === THREE.HalfFloatType && v.format === THREE.RGFormat && v.minFilter === THREE.NearestFilter);
  check('target: depth texture UnsignedInt', t.depthTexture && t.depthTexture.isDepthTexture && t.depthTexture.type === THREE.UnsignedIntType);
  check('getters', gb.color === c && gb.normal === n && gb.velocity === v && gb.depth === t.depthTexture);
  gb.setSize(100, 50);
  check('setSize', t.width === 100 && t.height === 50 && n.image.width === 100);
  warns.length = 0;
  const gh = new G.GBuffer(fakeRenderer({ float: false }), { width: 8, height: 8 });
  const hm = new THREE.MeshStandardMaterial();
  check('no EXT_color_buffer_float: unsupported (warned), sync / autoSync inert', gh.supported === false && !gh.autoSync && warns.some((w) => w.includes('EXT_color_buffer_float'))
    && gh.sync(new THREE.Mesh(new THREE.BoxGeometry(), hm)) === 0 && !(hm.defines && hm.defines.ENGINE_GBUFFER !== undefined));
  check('explicit colour format', new G.GBuffer(fakeRenderer(), { width: 4, height: 4, color: { type: THREE.HalfFloatType } }).color.format === THREE.RGBAFormat);

  const r = fakeRenderer();
  const prevRT = { textures: [{}] };
  r.rt = prevRT;
  gb.clear(r);
  const cl = r.calls.filter((x) => x[0] === 'clearBufferfv');
  check('clear(): attachments 1 and 2 to zero via clearBufferfv', cl.length === 2 && cl[0][2] === 1 && cl[1][2] === 2 && cl.every((x) => x[1] === r.gl.COLOR && x[3].every((z) => z === 0)));
  check('clear(): colour mask on, target bound, previous target restored', r.calls.some((x) => x[0] === 'colorMask' && x[1] === true) && r.calls.findIndex((x) => x[0] === 'setRenderTarget' && x[1] === gb.target) < r.calls.findIndex((x) => x[0] === 'clearBufferfv') && r.rt === prevRT);
}

// ---- GBuffer.sync(): classification -----------------------------------------------------------------------------
const box = new THREE.BoxGeometry();
const mesh = (m) => new THREE.Mesh(box, m);
const has = (m, d) => !!(m.defines && m.defines[d] !== undefined);
const W = 'ENGINE_GBUFFER', Z = 'ENGINE_GBUFFER_ZERO', NV = 'ENGINE_GBUFFER_NOVEL';
{
  const scene = new THREE.Scene();
  const M = {
    std: new THREE.MeshStandardMaterial(),
    phys: new THREE.MeshPhysicalMaterial(),
    lambert: new THREE.MeshLambertMaterial(),
    novel: new THREE.MeshStandardMaterial({ userData: { gbuffer: 'novel' } }),
    off: new THREE.MeshStandardMaterial({ userData: { gbuffer: false } }),
    glass: new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.5 }),
    additiveOpaque: new THREE.MeshLambertMaterial({ blending: THREE.AdditiveBlending }),
    points: new THREE.PointsMaterial(),
    sprite: new THREE.SpriteMaterial(),
    line: new THREE.LineBasicMaterial(),
    sky: new THREE.ShaderMaterial({ vertexShader: 'void main() { gl_Position = vec4( 0.0 ); }', fragmentShader: 'void main() { gl_FragColor = vec4( 1.0 ); }', side: THREE.BackSide, depthWrite: false }),
    shaderWriter: new THREE.ShaderMaterial({ userData: { gbuffer: true }, vertexShader: 'void main() {}', fragmentShader: 'void main() {}' }),
    author: new THREE.ShaderMaterial({ defines: { ENGINE_GBUFFER: 1 }, vertexShader: 'void main() {}', fragmentShader: 'void main() {}' }),
    raw: new THREE.RawShaderMaterial({ vertexShader: 'void main() {}', fragmentShader: 'void main() {}' }),
    glsl3: new THREE.ShaderMaterial({ glslVersion: THREE.GLSL3, vertexShader: 'void main() {}', fragmentShader: 'out vec4 c; void main() { c = vec4( 1.0 ); }' }),
    shadow: new THREE.ShadowMaterial(),
    custom: new THREE.MeshBasicMaterial({ transparent: true, blending: THREE.CustomBlending }),
    hidden: new THREE.MeshStandardMaterial({ colorWrite: false }),
  };
  const O = {};
  for (const k of ['std', 'phys', 'lambert', 'novel', 'off', 'glass', 'additiveOpaque', 'sky', 'shaderWriter', 'author', 'raw', 'glsl3', 'shadow', 'custom', 'hidden']) scene.add(O[k] = mesh(M[k]));
  scene.add(O.points = new THREE.Points(box, M.points));
  scene.add(O.sprite = new THREE.Sprite(M.sprite));
  scene.add(O.line = new THREE.Line(box, M.line));
  const group = new THREE.Group();
  scene.add(group);
  group.add(O.multi = new THREE.Mesh(box, [M.std, M.custom]));
  const versions = Object.fromEntries(Object.entries(M).map(([k, m]) => [k, m.version]));
  const skyBefore = M.sky.onBeforeCompile;

  const gb = new G.GBuffer(fakeRenderer());
  const changed = gb.sync(scene);
  for (const k of ['std', 'phys', 'lambert', 'novel', 'shaderWriter']) check(`writer: ${k} gets ENGINE_GBUFFER`, has(M[k], W) && !has(M[k], Z));
  check('writer: novel gets ENGINE_GBUFFER_NOVEL (others not)', has(M.novel, NV) && !has(M.std, NV));
  for (const k of ['off', 'glass', 'additiveOpaque', 'points', 'sprite', 'line', 'sky']) check(`zero: ${k} gets ENGINE_GBUFFER_ZERO only`, has(M[k], Z) && !has(M[k], W));
  for (const k of ['raw', 'glsl3', 'shadow', 'custom']) check(`mask: ${k} gets no define`, !has(M[k], Z) && !has(M[k], W));
  check('author define left alone', has(M.author, W) && !has(M.author, Z));
  check('colorWrite false: zero class (outputs declared, nothing written)', !has(M.hidden, W) && has(M.hidden, Z));
  check('needsUpdate on every material whose defines changed', ['std', 'novel', 'glass', 'sky', 'points'].every((k) => M[k].version > versions[k]) && M.raw.version === versions.raw && M.author.version === versions.author);
  check('sync() returns the number of materials changed (5 writers + 8 zero)', changed === 13, String(changed));
  check('ShaderMaterial writer gets the shared uniform objects', M.shaderWriter.uniforms.engineUnjitteredProj === G.GBUFFER_UNIFORMS.engineUnjitteredProj && M.shaderWriter.uniforms.engineViewToPrevClip === G.GBUFFER_UNIFORMS.engineViewToPrevClip);
  const hooked = (o) => Object.prototype.hasOwnProperty.call(o, 'onBeforeRender') && Object.prototype.hasOwnProperty.call(o, 'onAfterRender');
  for (const k of ['raw', 'glsl3', 'shadow', 'custom', 'multi']) check(`mask hooks on ${k}`, hooked(O[k]));
  for (const k of ['std', 'glass', 'sky', 'points', 'sprite', 'line', 'author', 'hidden']) check(`no hooks on ${k}`, !hooked(O[k]));
  warns.length = 0;
  const optT = new THREE.ShaderMaterial({ transparent: true, userData: { gbuffer: true }, vertexShader: 'void main() {}', fragmentShader: 'void main() {}' });
  const optR = new THREE.RawShaderMaterial({ userData: { gbuffer: 'novel' }, vertexShader: 'void main() {}', fragmentShader: 'void main() {}' });
  gb.sync(mesh(optT)); gb.sync(mesh(optR));
  check('opt-in ShaderMaterial that is transparent / raw: warned, not a writer', !has(optT, W) && has(optT, Z) && !has(optR, W)
    && warns.filter((w) => w.includes('userData.gbuffer')).length === 2);
  gb.unsync(mesh(optT)); // (the meshes are throwaway; their materials leave the stats)
  for (const m of [optT, optR]) m.dispatchEvent({ type: 'dispose' });
  check('stats (writers incl. the author one, zero, masked)', gb.stats.materialsDefined === 6 && gb.stats.materialsZero === 8 && gb.stats.materialsMasked === 4, JSON.stringify(gb.stats));
  check('sync() again changes nothing', gb.sync(scene) === 0);

  // ShaderMaterial zero writer: chained onBeforeCompile prepends the guarded wrapper once
  check('sky: onBeforeCompile chained', M.sky.onBeforeCompile !== skyBefore && M.sky.customProgramCacheKey() !== new THREE.ShaderMaterial().customProgramCacheKey());
  const sh = { fragmentShader: M.sky.fragmentShader, vertexShader: M.sky.vertexShader, uniforms: {} };
  M.sky.onBeforeCompile(sh, null);
  const once = sh.fragmentShader;
  M.sky.onBeforeCompile(sh, null);
  check('sky: wrapper prepended once', once.includes('#define main engineGbUserMain') && sh.fragmentShader === once && once.endsWith(M.sky.fragmentShader));
  const sz = preprocess(once + '\n' + resolveIncludes('#include <clipping_planes_pars_fragment>'), ['ENGINE_GBUFFER_ZERO', 'gl_FragColor']);
  check('sky: include guard (prepend + host chunk -> one wrapper)', count(sz, 'void engineGbUserMain();') === 1);
  check('sky: inert without the define', preprocess(once, ['gl_FragColor']).trim() === M.sky.fragmentShader.trim());

  // draw-buffer masks: only while an MRT target is bound, only for materials without a G-buffer define
  const r = fakeRenderer();
  const mrt = gb.target, cam = new THREE.PerspectiveCamera();
  r.rt = mrt;
  O.custom.onBeforeRender(r, scene, cam, box, M.custom, null);
  O.custom.onAfterRender(r, scene, cam, box, M.custom, null);
  const db = r.calls.filter((x) => x[0] === 'drawBuffers').map((x) => x[1].join(','));
  check('mask: [C0, NONE, NONE] then [C0, C1, C2]', db.length === 2 && db[0] === `${0x8ce0},0,0` && db[1] === `${0x8ce0},${0x8ce1},${0x8ce2}`, db.join(' | '));
  r.calls.length = 0;
  O.multi.onBeforeRender(r, scene, cam, box, M.std, null); O.multi.onAfterRender(r, scene, cam, box, M.std, null);
  check('mask: a writer group of a hooked mesh draws unmasked', r.calls.length === 0);
  O.multi.onBeforeRender(r, scene, cam, box, M.custom, null); O.multi.onAfterRender(r, scene, cam, box, M.custom, null);
  check('mask: the masked group of the same mesh', r.calls.length === 2);
  r.calls.length = 0; r.rt = null;
  O.custom.onBeforeRender(r, scene, cam, box, M.custom, null); O.custom.onAfterRender(r, scene, cam, box, M.custom, null);
  r.rt = { textures: [{}] };
  O.custom.onBeforeRender(r, scene, cam, box, M.custom, null); O.custom.onAfterRender(r, scene, cam, box, M.custom, null);
  check('mask: no GL calls for the canvas or single-attachment targets', r.calls.length === 0);
  let prevCalled = 0;
  const obj = mesh(new THREE.MeshBasicMaterial({ transparent: true, blending: THREE.CustomBlending }));
  obj.onBeforeRender = () => { prevCalled++; };
  scene.add(obj); // 'childadded' syncs it
  check('childadded: new object classified + hooked', hooked(obj) && gb.stats.materialsMasked === 5);
  r.rt = mrt; r.calls.length = 0;
  obj.onBeforeRender(r, scene, cam, box, obj.material, null); obj.onAfterRender(r, scene, cam, box, obj.material, null);
  check('chained hook calls the previous one', prevCalled === 1 && r.calls.length === 2);

  // engineViewToPrevClip refreshed by renders of the synced scene into an MRT target
  const camera = new THREE.PerspectiveCamera(60, 2, 0.1, 100);
  camera.position.set(3000.25, 12, -4500.5); camera.rotation.set(0.1, 0.7, 0); camera.updateMatrixWorld();
  G.GBUFFER_UNIFORMS.enginePrevViewProj.value.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
  scene.onBeforeRender(r, scene, camera, { textures: [{}] });
  check('single-target render leaves engineViewToPrevClip', G.GBUFFER_UNIFORMS.engineViewToPrevClip.value.equals(new THREE.Matrix4()));
  scene.onBeforeRender(r, scene, camera, mrt);
  const expect = new THREE.Matrix4().multiplyMatrices(G.GBUFFER_UNIFORMS.enginePrevViewProj.value, camera.matrixWorld);
  check('MRT render: engineViewToPrevClip = prevViewProj * camera.matrixWorld', G.GBUFFER_UNIFORMS.engineViewToPrevClip.value.equals(expect));
  const e = G.GBUFFER_UNIFORMS.engineViewToPrevClip.value.elements, p = camera.projectionMatrix.elements;
  check('same camera: view -> previous clip is the projection (precise far from the origin)', e.every((x, i) => Math.abs(x - p[i]) < 1e-9));

  // re-classification + unsync
  M.std.transparent = true; gb.sync(O.std);
  check('re-sync: a material turned transparent switches writer -> zero', !has(M.std, W) && has(M.std, Z));
  M.std.transparent = false; gb.sync(O.std);
  gb.unsync(scene);
  check('unsync: every define removed', Object.entries(M).every(([k, m]) => k === 'author' ? has(m, W) : !has(m, W) && !has(m, Z) && !has(m, NV)));
  check('unsync: onBeforeCompile / cache key restored', !Object.prototype.hasOwnProperty.call(M.sky, 'onBeforeCompile') && !Object.prototype.hasOwnProperty.call(M.sky, 'customProgramCacheKey'));
  check('unsync: hooks removed (a pre-existing own hook restored)', !hooked(O.custom) && !hooked(O.multi) && obj.onAfterRender === THREE.Object3D.prototype.onAfterRender && Object.prototype.hasOwnProperty.call(obj, 'onBeforeRender'));
  obj.onBeforeRender(r, scene, camera, box, obj.material, null);
  check('unsync: restored hook is the original', prevCalled === 2 && r.calls.length === 2);
  check('unsync: scene listeners detached', scene.onBeforeRender === THREE.Object3D.prototype.onBeforeRender && !Object.prototype.hasOwnProperty.call(scene, 'onBeforeRender'));
  const late = mesh(new THREE.MeshStandardMaterial());
  scene.add(late);
  check('unsync: childadded no longer syncs', !has(late.material, W));
  check('stats after unsync', gb.stats.materialsDefined === 0 && gb.stats.materialsZero === 0 && gb.stats.materialsMasked === 0, JSON.stringify(gb.stats));

  // nonWriters: 'mask' -> every non-writer is masked (A/B)
  const gm = new G.GBuffer(fakeRenderer(), { nonWriters: 'mask' });
  gm.sync(scene);
  check("nonWriters 'mask': transparent / points / sky masked, writers defined", !has(M.glass, Z) && hooked(O.glass) && hooked(O.points) && hooked(O.sky) && has(M.lambert, W) && !hooked(O.lambert));
  gm.dispose();
  check('dispose(): defines removed', !has(M.lambert, W) && !hooked(O.glass));
  check('background texture warning without autoSync only', (() => {
    const s2 = new THREE.Scene(); s2.background = new THREE.Texture();
    new G.GBuffer(fakeRenderer()).sync(s2);
    const quiet = !warns.some((w) => w.includes('scene.background'));
    new G.GBuffer(fakeRenderer(), { autoSync: false }).sync(s2);
    return quiet && warns.some((w) => w.includes('scene.background'));
  })());

  // sync() re-asserts: a define removed elsewhere (e.g. another GBuffer's unsync) comes back
  gb.sync(scene);
  delete M.lambert.defines[W];
  gb.sync(O.lambert);
  check('sync() re-asserts a removed define', has(M.lambert, W));
  gb.unsync(scene);
}

// ---- autoSync: draw-time classification through Material.prototype.onBeforeRender ----------------------------------
{
  const gb = new G.GBuffer(fakeRenderer());
  const r = fakeRenderer(), cam = new THREE.PerspectiveCamera(), scene = new THREE.Scene();
  const deep = mesh(new THREE.MeshStandardMaterial()), odd = mesh(new THREE.MeshBasicMaterial({ transparent: true, blending: THREE.CustomBlending }));
  const v0 = deep.material.version;
  r.rt = { textures: [{}, {}, {}] }; // another MRT target: not this GBuffer's
  deep.material.onBeforeRender(r, scene, cam, box, deep, null);
  check('autoSync: draws into other targets are ignored', !has(deep.material, W));
  r.rt = gb.target;
  deep.material.onBeforeRender(r, scene, cam, box, deep, null);
  check('autoSync: unseen writer classified before its program is chosen', has(deep.material, W) && deep.material.version > v0 && gb.stats.materialsDefined === 1);
  r.calls.length = 0;
  odd.onBeforeRender(r, scene, cam, box, odd.material, null);
  odd.material.onBeforeRender(r, scene, cam, box, odd, null);
  odd.onAfterRender(r, scene, cam, box, odd.material, null);
  const db = r.calls.filter((x) => x[0] === 'drawBuffers').map((x) => x[1].join(','));
  check('autoSync: unseen masked material hooks its object and masks this draw', db.length === 2 && db[0] === `${0x8ce0},0,0` && gb.stats.materialsMasked === 1, db.join(' | '));
  r.calls.length = 0;
  odd.onBeforeRender(r, scene, cam, box, odd.material, null); odd.material.onBeforeRender(r, scene, cam, box, odd, null); odd.onAfterRender(r, scene, cam, box, odd.material, null);
  check('autoSync: next draw masked once by the object hook', r.calls.filter((x) => x[0] === 'drawBuffers').length === 2);
  const off = new G.GBuffer(fakeRenderer(), { autoSync: false }), m3 = new THREE.MeshStandardMaterial();
  r.rt = off.target;
  m3.onBeforeRender(r, scene, cam, box, mesh(m3), null);
  check('autoSync: false -> no draw-time classification', !has(m3, W));
  // an invisible material is skipped by sync() and classified when it draws
  r.rt = gb.target;
  const inv = mesh(new THREE.MeshStandardMaterial({ visible: false }));
  gb.sync(inv);
  check('visible: false skipped by sync()', !has(inv.material, W));
  inv.material.visible = true;
  inv.material.onBeforeRender(r, scene, cam, box, inv, null);
  check('autoSync: a skipped material that became visible is classified when it draws', has(inv.material, W));
  gb.dispose(); off.dispose();
  const own = (o, k) => Object.prototype.hasOwnProperty.call(o, k);
  check('dispose(): objects hooked at draw time (outside any synced root) are unhooked', !own(odd, 'onBeforeRender') && !own(odd, 'onAfterRender'));
  r.calls.length = 0;
  odd.onBeforeRender(r, scene, cam, box, odd.material, null); odd.onAfterRender(r, scene, cam, box, odd.material, null);
  check('dispose(): no draw-buffer calls afterwards', r.calls.length === 0);
  r.rt = gb.target;
  const m4 = new THREE.MeshStandardMaterial();
  m4.onBeforeRender(r, scene, cam, box, mesh(m4), null);
  check('autoSync: a disposed GBuffer no longer classifies', !has(m4, W));
}

// ---- GBuffer.track(): per-object velocity -----------------------------------------------------------------------
{
  const scene = new THREE.Scene();
  const shared = new THREE.MeshStandardMaterial({ name: 'paint' });
  const car = mesh(shared), other = mesh(shared), solo = mesh(new THREE.MeshStandardMaterial());
  const skinlike = new THREE.Group();
  skinlike.add(mesh(new THREE.MeshLambertMaterial()), mesh(new THREE.MeshLambertMaterial()));
  scene.add(car, other, solo, skinlike);
  const gb = new G.GBuffer(fakeRenderer());
  gb.sync(scene);
  warns.length = 0;
  check('track(): returns meshes tracked', gb.track(car) === 1 && gb.track(skinlike) === 2 && gb.track(car) === 0);
  check('track(): shared material cloned (warned once), the other mesh keeps the original', car.material !== shared && other.material === shared && warns.filter((w) => w.includes('shared')).length === 1);
  check('track(): clone keeps G-buffer + own defines, adds ENGINE_GBUFFER_PREV', has(car.material, W) && has(car.material, 'STANDARD') && has(car.material, 'ENGINE_GBUFFER_PREV') && !has(shared, 'ENGINE_GBUFFER_PREV'));
  gb.track(solo);
  const soloMat = solo.material;
  check('track(): unshared material patched in place', soloMat.isMeshStandardMaterial && has(soloMat, 'ENGINE_GBUFFER_PREV') && solo.material === soloMat);
  check('stats.tracked', gb.stats.tracked === 4);
  const sh = { uniforms: {}, vertexShader: '', fragmentShader: '' };
  car.material.onBeforeCompile(sh, null);
  check('track(): onBeforeCompile injects the per-mesh uniform', sh.uniforms.enginePrevModelView && sh.uniforms.enginePrevModelView.value.isMatrix4);
  const sh2 = { uniforms: {}, vertexShader: '', fragmentShader: '' };
  solo.material.onBeforeCompile(sh2, null);
  check('track(): uniforms differ per mesh', sh2.uniforms.enginePrevModelView !== sh.uniforms.enginePrevModelView);
  check('track(): cache key differs from an untracked material', car.material.customProgramCacheKey() !== shared.customProgramCacheKey());

  const camera = new THREE.PerspectiveCamera(); camera.position.set(2500, 3, 1800); camera.updateMatrixWorld();
  const u = sh.uniforms.enginePrevModelView.value;
  car.position.set(2510, 0, 1800); car.updateMatrixWorld();
  car.material.onBeforeRender(fakeRenderer(), scene, camera, box, car, null);
  const mv = new THREE.Matrix4().multiplyMatrices(camera.matrixWorldInverse, car.matrixWorld);
  check('first frame: previous model-view = current (no motion yet)', u.equals(mv));
  gb.endFrame();
  car.position.set(2512, 0, 1800); car.updateMatrixWorld();
  car.material.onBeforeRender(fakeRenderer(), scene, camera, box, car, null);
  const prevMV = mv.clone();
  check('next frame: previous model-view = view * last frame matrixWorld', u.equals(prevMV));
  gb.endFrame();
  gb.endFrame(); // not drawn this frame -> no previous next frame
  car.material.onBeforeRender(fakeRenderer(), scene, camera, box, car, null);
  check('a frame without a draw drops the history', u.equals(new THREE.Matrix4().multiplyMatrices(camera.matrixWorldInverse, car.matrixWorld)));

  const mats = new THREE.ShaderMaterial({ vertexShader: 'void main() {}', fragmentShader: 'void main() {}' });
  const sm = mesh(mats); scene.add(sm);
  gb.track(sm);
  check('track(ShaderMaterial): uniform set directly, no compile patch', mats.uniforms.enginePrevModelView && mats.uniforms.enginePrevModelView.value.isMatrix4 && !mats.customProgramCacheKey().includes('engine-gbprev'));

  gb.untrack(scene);
  check('untrack(): define + patches removed, clone kept', car.material !== shared && !has(car.material, 'ENGINE_GBUFFER_PREV') && !has(solo.material, 'ENGINE_GBUFFER_PREV')
    && !Object.prototype.hasOwnProperty.call(solo.material, 'onBeforeRender') && !Object.prototype.hasOwnProperty.call(solo.material, 'onBeforeCompile') && !mats.uniforms.enginePrevModelView && gb.stats.tracked === 0);
  check('{ clone: true } always clones', (() => { const m = new THREE.MeshStandardMaterial(), o = mesh(m); scene.add(o); gb.track(o, { clone: true }); return o.material !== m && has(o.material, 'ENGINE_GBUFFER_PREV'); })());

  // a clone of a tracked mesh shares its tracked material: tracking it clones the material without the first mesh's
  // velocity hooks (no double chain, the tag once, its own uniform, the user's own patch kept)
  let userCompile = 0;
  const base = new THREE.MeshStandardMaterial();
  base.onBeforeCompile = () => { userCompile++; };
  base.customProgramCacheKey = () => 'user-key';
  const first = mesh(base); scene.add(first);
  gb.track(first);
  const second = first.clone(); scene.add(second); // Object3D.clone shares the (tracked) material
  gb.track(second);
  const k2 = second.material.customProgramCacheKey();
  check('clone of a tracked material: own material, tag once, user key kept', second.material !== base && count(k2, 'engine-gbprev') === 1 && k2.startsWith('user-key'), k2);
  const shA = { uniforms: {}, vertexShader: '', fragmentShader: '' }, shB = { uniforms: {}, vertexShader: '', fragmentShader: '' };
  userCompile = 0;
  base.onBeforeCompile(shA, null); second.material.onBeforeCompile(shB, null);
  check('clone of a tracked material: user onBeforeCompile runs once each, separate uniforms', userCompile === 2 && shA.uniforms.enginePrevModelView && shB.uniforms.enginePrevModelView && shA.uniforms.enginePrevModelView !== shB.uniforms.enginePrevModelView);
  const uA = shA.uniforms.enginePrevModelView.value, before = uA.clone();
  second.position.x += 5; second.updateMatrixWorld();
  second.material.onBeforeRender(fakeRenderer(), scene, camera, box, second, null);
  check('clone of a tracked material: drawing the clone leaves the first mesh\'s uniform alone', uA.equals(before) && shB.uniforms.enginePrevModelView.value.equals(new THREE.Matrix4().multiplyMatrices(camera.matrixWorldInverse, second.matrixWorld)));

  // a zero-writer ShaderMaterial shared by two meshes: the tracked clone gets one zero chain
  const zs = new THREE.ShaderMaterial({ vertexShader: 'void main() {}', fragmentShader: 'void main() { gl_FragColor = vec4( 1.0 ); }' });
  const za = mesh(zs), zb = mesh(zs); scene.add(za, zb); // 'childadded' -> zero class
  gb.track(za);
  const zk = za.material.customProgramCacheKey();
  const zsh = { uniforms: {}, vertexShader: zs.vertexShader, fragmentShader: zs.fragmentShader };
  za.material.onBeforeCompile(zsh, null);
  check('zero ShaderMaterial clone: one zero chain, wrapper once, own uniform', za.material !== zs && count(zk, 'engine-gbzero') === 1 && count(zsh.fragmentShader, '#define main engineGbUserMain') === 1
    && za.material.uniforms.enginePrevModelView && za.material.uniforms.enginePrevModelView !== zs.uniforms.enginePrevModelView && has(za.material, Z), zk);
}

// ---- InstancedPool velocity -------------------------------------------------------------------------------------
{
  const geo = new THREE.BoxGeometry(), mat = new THREE.MeshStandardMaterial();
  const plain = new InstancedPool(geo, new THREE.MeshStandardMaterial(), { capacity: 4 });
  check('pool without velocity: unchanged', plain.mesh.geometry === geo && plain.instancePrevMatrix === null && !plain.velocity && !has(plain.material, 'ENGINE_GBUFFER_INSTANCE_PREV'));
  plain.capturePrev();

  const pool = new InstancedPool(geo, mat, { capacity: 4, velocity: true });
  const pv = pool.instancePrevMatrix;
  check('velocity: instancePrevMatrix 16 * capacity, itemSize 16, dynamic', pv.isInstancedBufferAttribute && pv.array.length === 64 && pv.itemSize === 16 && pv.usage === THREE.DynamicDrawUsage);
  check('velocity: on a view geometry; the caller geometry untouched', pool.mesh.geometry !== geo && pool.mesh.geometry.attributes.instancePrevMatrix === pv && !geo.attributes.instancePrevMatrix
    && pool.mesh.geometry.attributes.position === geo.attributes.position && pool.mesh.geometry.index === geo.index);
  check('velocity: material define', mat.defines.ENGINE_GBUFFER_INSTANCE_PREV === 1);
  check('velocity: initialised from the instance matrices', pv.array.every((x, i) => x === pool.mesh.instanceMatrix.array[i]));
  const a = pool.add(); pool.setTRS(a, 1, 2, 3);
  const b = pool.add(); pool.setTRS(b, 10, 0, 0);
  const at = (arr, i, k) => arr[i * 16 + k];
  check('fresh instances: previous = current', at(pv.array, a, 12) === 1 && at(pv.array, b, 12) === 10);
  pool.capturePrev();
  pool.setTRS(a, 5, 2, 3);
  check('moved instance: previous keeps last frame', at(pv.array, a, 12) === 1 && at(pool.mesh.instanceMatrix.array, a, 12) === 5);
  const v0 = pv.version;
  pool.capturePrev();
  check('capturePrev(): copies + uploads the active range', at(pv.array, a, 12) === 5 && pv.version > v0 && pv.updateRanges.length === 1 && pv.updateRanges[0].start === 0 && pv.updateRanges[0].count === 32);
  const v1 = pv.version;
  pool.capturePrev();
  check('capturePrev(): skipped when nothing moved', pv.version === v1);
  pool.setTRS(b, 11, 0, 0); // b moves this frame, then a is removed: b lands in slot a with its previous matrix
  pool.removeAt(a);
  check('removeAt(): the previous matrix moves with the instance', at(pool.mesh.instanceMatrix.array, a, 12) === 11 && at(pv.array, a, 12) === 10);
  const gb = new G.GBuffer(fakeRenderer());
  gb.trackPool(pool);
  gb.endFrame();
  check('GBuffer.endFrame() captures tracked pools', at(pv.array, a, 12) === 11 && gb.stats.pools === 1);
  const view = pool.mesh.geometry;
  let disposed = 0;
  view.addEventListener('dispose', () => disposed++);
  for (let i = 0; i < 6; i++) { const k = pool.add(); pool.setTRS(k, 100 + i, 0, 0); }
  const pv2 = pool.instancePrevMatrix;
  check('growth: new attribute sized to the capacity, contents kept', pool.capacity >= 7 && pv2 !== pv && pv2.array.length === pool.capacity * 16 && at(pv2.array, a, 12) === 11);
  check('growth: new view, old view disposed without its shared attributes', pool.mesh.geometry !== view && disposed === 1 && !view.attributes.position && view.index === null && pool.mesh.geometry.attributes.instancePrevMatrix === pv2);
  check('growth: fresh instances seeded', at(pv2.array, pool.count - 1, 12) === 105);
  pool.capturePrev();
  check('capturePrev() after growth', pv2.array.every((x, i) => x === pool.mesh.instanceMatrix.array[i]));
  gb.untrackPool(pool);
  pool.dispose();
  check('dispose(): geometry restored, attribute released', pool.mesh.geometry === geo && pool.instancePrevMatrix === null && !geo.attributes.instancePrevMatrix);
  const pool2 = new InstancedPool(geo, mat, { capacity: 2, velocity: true });
  check('two velocity pools share a geometry, each with its own attribute', pool2.mesh.geometry !== pool.mesh.geometry && pool2.instancePrevMatrix !== pv2 && pool2.mesh.geometry.attributes.position === geo.attributes.position);
  // setCount() re-activating kept slots: previous = current for them (no first-frame motion)
  const p3 = new InstancedPool(geo, mat, { capacity: 4, velocity: true });
  p3.setCount(3); p3.setTRS(2, 7, 0, 0); p3.capturePrev();
  p3.setCount(1);
  p3.mesh.instanceMatrix.array[2 * 16 + 12] = 9; // moved while inactive (data kept)
  p3.setCount(3);
  check('setCount() seeds re-activated slots', p3.instancePrevMatrix.array[2 * 16 + 12] === 9 && p3.instancePrevMatrix.array[1 * 16 + 12] === p3.mesh.instanceMatrix.array[1 * 16 + 12]);
}

console.log(`gbuffer-chunks: ${passes} passed, ${failures} failed`);
process.exit(failures ? 1 : 0);
