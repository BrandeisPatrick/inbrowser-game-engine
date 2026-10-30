// Clustered lights (src/render/ClusteredLights.js), headless: the CPU cluster assignment against a brute force
// (points sampled in every cell of the grid: each light whose lit region holds the point must be in the list of the
// cell the shader picks for it), the depth-slice mapping, the data-texture layouts and upload ranges (a GPU-storage
// model: every queued request alone yields what the shader reads, even into fresh storage), the index-list
// caps, and the chunk patch (inert text without ENGINE_CLUSTERED, anchors, uniforms, material selection).
// Run: node test/clustered.mjs
import * as THREE from 'three';
import {
  ClusteredLights, installClusteredChunks, applyClustered, syncClustered, clusteredUniforms, CLUSTERED_DEFINE, MAX_PER_CELL,
} from '../src/render/ClusteredLights.js';

let failures = 0;
const results = [];
const fmt = (x) => (typeof x === 'number' ? (Number.isInteger(x) ? String(x) : x.toFixed(4)) : String(x));
function check(name, value, lo, hi = lo) {
  const ok = typeof value === 'number' ? Number.isFinite(value) && value >= lo && value <= hi : value === lo;
  if (!ok) failures++;
  results.push(`${ok ? 'PASS' : 'FAIL'}  ${name.padEnd(64)} ${fmt(value).padStart(10)}   [${fmt(lo)}${hi !== lo ? ' .. ' + fmt(hi) : ''}]`);
  return ok;
}
// what three does once it has uploaded a texture (WebGLTextures.uploadTexture)
function simulateUpload(t) { t.clearUpdateRanges(); }
function mulberry32(a) {
  return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

// --- chunk patch (originals first: importing the module patches nothing) ----------------------------------------
const C = THREE.ShaderChunk;
const ORIG = { pars: C.lights_pars_begin, frag: C.lights_fragment_begin };
const SPOT_ANCHOR = '#if ( NUM_SPOT_LIGHTS > 0 ) && defined( RE_Direct )';
const RE_DIRECT = 'RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );';
check('anchor: spot block once in lights_fragment_begin', ORIG.frag.split(SPOT_ANCHOR).length - 1, 1);
check('anchor: r180 RE_Direct call in the point loop', ORIG.frag.includes(RE_DIRECT) ? 1 : 0, 1);
check('anchor: getDistanceAttenuation / getSpotAttenuation unconditional',
  /^float getDistanceAttenuation\(/m.test(ORIG.pars) && /^float getSpotAttenuation\(/m.test(ORIG.pars) ? 1 : 0, 1);
check('nothing patched before install', C.lights_pars_begin === ORIG.pars && C.lights_fragment_begin === ORIG.frag ? 1 : 0, 1);
check('installClusteredChunks()', installClusteredChunks() ? 1 : 0, 1);
check('install is idempotent', installClusteredChunks() && C.lights_pars_begin.split('struct EngineClLight').length === 2 ? 1 : 0, 1);

// line-level preprocessor: drop every #if block whose condition is a conjunction with defined( ENGINE_CLUSTERED )
function withoutDefine(src) {
  const out = [], lines = src.split('\n');
  let skip = 0, bad = 0;
  for (const line of lines) {
    const t = line.trim();
    if (skip) {
      if (/^#if/.test(t)) skip++;
      else if (/^#endif/.test(t)) skip--;
      else if (skip === 1 && /^#(else|elif)/.test(t)) bad++;
      continue;
    }
    if (/^#if\b/.test(t) && t.includes(`defined( ${CLUSTERED_DEFINE} )`)) { if (t.includes('||')) bad++; skip = 1; continue; }
    out.push(line);
  }
  return bad || skip ? null : out.join('\n');
}
check('lights_pars_begin without the define == three', withoutDefine(C.lights_pars_begin) === ORIG.pars ? 1 : 0, 1);
check('lights_fragment_begin without the define == three', withoutDefine(C.lights_fragment_begin) === ORIG.frag ? 1 : 0, 1);
const loopAt = C.lights_fragment_begin.indexOf(`#if defined( ${CLUSTERED_DEFINE} ) && defined( RE_Direct )`);
check('loop sits after directLight, right before the spot block', loopAt > C.lights_fragment_begin.indexOf('IncidentLight directLight;')
  && C.lights_fragment_begin.indexOf(SPOT_ANCHOR) > loopAt && C.lights_fragment_begin.indexOf('#if ( NUM_POINT_LIGHTS > 0 )') < loopAt ? 1 : 0, 1);
check('loop calls RE_Direct with the r180 signature', C.lights_fragment_begin.slice(loopAt, C.lights_fragment_begin.indexOf(SPOT_ANCHOR)).includes(RE_DIRECT) ? 1 : 0, 1);
check('directional shadow line untouched (cascade anchor)', C.lights_fragment_begin.includes('directLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( directionalShadowMap[ i ]') ? 1 : 0, 1);
check(`loop bound = MAX_PER_CELL (${MAX_PER_CELL})`, C.lights_fragment_begin.includes(`engineClK < ${MAX_PER_CELL};`) ? 1 : 0, 1);

const U = clusteredUniforms();
const lit = ['standard', 'physical', 'lambert', 'phong', 'toon'];
check('uniforms injected into the lit ShaderLib templates', lit.filter((k) => THREE.ShaderLib[k].uniforms.engineClLights === U.engineClLights
  && THREE.ShaderLib[k].uniforms.engineClParams2 === U.engineClParams2).length, lit.length);
check('basic / depth templates untouched', THREE.ShaderLib.basic.uniforms.engineClLights || THREE.ShaderLib.depth.uniforms.engineClLights ? 0 : 1, 1);
const cloned = THREE.UniformsUtils.clone(THREE.ShaderLib.physical.uniforms); // what a built-in material gets at compile
check('cloned uniforms keep the shared textures and vec4 arrays', cloned.engineClLights.value === U.engineClLights.value
  && cloned.engineClCells.value === U.engineClCells.value && cloned.engineClParams.value === U.engineClParams.value
  && cloned.engineClTiles.value === U.engineClTiles.value ? 1 : 0, 1);

// material selection
{
  const scene = new THREE.Scene();
  const geo = new THREE.BoxGeometry();
  const mats = [new THREE.MeshStandardMaterial(), new THREE.MeshPhysicalMaterial(), new THREE.MeshLambertMaterial(),
    new THREE.MeshPhongMaterial(), new THREE.MeshToonMaterial(), new THREE.MeshBasicMaterial(), new THREE.ShadowMaterial(),
    new THREE.ShaderMaterial({ lights: true, uniforms: THREE.UniformsUtils.merge([THREE.UniformsLib.lights]) }), new THREE.ShaderMaterial()];
  for (const m of mats) scene.add(new THREE.Mesh(geo, m));
  scene.add(new THREE.Mesh(geo, [new THREE.MeshStandardMaterial(), new THREE.MeshBasicMaterial()]));
  check('applyClustered(on): lit materials switched', applyClustered(scene, true), 7);
  check('applyClustered(on) again: no change', applyClustered(scene, true), 0);
  check('define set on Physical, kept STANDARD/PHYSICAL', mats[1].defines[CLUSTERED_DEFINE] === '' && 'PHYSICAL' in mats[1].defines ? 1 : 0, 1);
  check('lights:true ShaderMaterial got the shared uniform objects', mats[7].uniforms.engineClCells === U.engineClCells ? 1 : 0, 1);
  check('Basic / Shadow / unlit ShaderMaterial untouched', [5, 6, 8].filter((i) => mats[i].defines && CLUSTERED_DEFINE in mats[i].defines).length, 0);
  check('applyClustered(off)', applyClustered(scene, false), 7);
  check('define removed', mats.filter((m) => m.defines && CLUSTERED_DEFINE in m.defines).length, 0);
  check('syncClustered(on)', syncClustered(scene, true), 7);
  const g = new THREE.Group(); scene.add(g);
  const late = new THREE.Mesh(geo, new THREE.MeshStandardMaterial()); g.add(late); // added below a later child
  const deep = new THREE.Group(); deep.add(new THREE.Mesh(geo, new THREE.MeshLambertMaterial())); late.add(deep);
  check('syncClustered: meshes added later, at any depth', (CLUSTERED_DEFINE in late.material.defines) && (CLUSTERED_DEFINE in deep.children[0].material.defines) ? 1 : 0, 1);
  syncClustered(scene, false);
  const after = new THREE.Mesh(geo, new THREE.MeshStandardMaterial()); g.add(after);
  check('syncClustered(off): listeners removed', CLUSTERED_DEFINE in after.material.defines ? 0 : 1, 1);
  // a subtree removed while on loses its listeners: re-parenting it elsewhere re-applies nothing, off covers the rest
  syncClustered(scene, true);
  const h = new THREE.Group(); scene.add(h);
  scene.remove(h);
  const orphan = new THREE.Mesh(geo, new THREE.MeshStandardMaterial()); h.add(orphan);
  check('removed while on: adding below it is not synced', CLUSTERED_DEFINE in orphan.material.defines ? 0 : 1, 1);
  scene.add(h);
  const back = new THREE.Mesh(geo, new THREE.MeshStandardMaterial()); h.add(back);
  check('re-added to the synced tree: switched on and listened again', (CLUSTERED_DEFINE in orphan.material.defines) && (CLUSTERED_DEFINE in back.material.defines) ? 1 : 0, 1);
  scene.remove(h);
  syncClustered(scene, false);
  const later = new THREE.Mesh(geo, new THREE.MeshStandardMaterial()); h.add(later);
  check('off after a removal: the removed subtree does not react', CLUSTERED_DEFINE in later.material.defines ? 0 : 1, 1);
  let listening = 0;
  const count = (o) => { const l = o._listeners; if (l && ((l.childadded && l.childadded.length) || (l.childremoved && l.childremoved.length))) listening++; };
  scene.traverse(count); h.traverse(count);
  check('off: no childadded / childremoved listener left in or out of the tree', listening, 0);
}

// --- slice mapping ----------------------------------------------------------------------------------------------
{
  const cl = new ClusteredLights();
  const [, , gz] = cl.grid;
  let rt = 0, below = 0;
  for (let s = 0; s < gz; s++) {
    if (cl.sliceOf(cl.sliceDepth(s) * (1 + 1e-9)) === s) rt++;
    if (s > 0 && cl.sliceOf(cl.sliceDepth(s) * (1 - 1e-9)) === s - 1) below++;
  }
  check('slice round trip: sliceOf(sliceDepth(s)+) == s', rt, gz);
  check('slice round trip: sliceOf(sliceDepth(s)-) == s - 1', below, gz - 1);
  check('sliceDepth(0) = zNear, sliceDepth(gz) = zFar', Math.abs(cl.sliceDepth(0) - 0.5) + Math.abs(cl.sliceDepth(gz) - 400), 0, 1e-9);
  const rnd = mulberry32(3);
  let same = 0, f32 = 0;
  const P = cl.uniforms.engineClParams.value, P2 = cl.uniforms.engineClParams2.value;
  for (let k = 0; k < 10000; k++) {
    const z = 0.5 * Math.pow(800, rnd());
    const ref = Math.floor(Math.log(z / 0.5) / Math.log(400 / 0.5) * gz);
    if (cl.sliceOf(z) === Math.min(gz - 1, ref)) same++;
    // the shader's float32 parameters: within 1e-4 of a slice of the CPU value
    const g = Math.log(z) * P[3] + P2[0], c = Math.log(z) * cl.sliceScale + cl.sliceBias;
    if (Math.abs(g - c) < 1e-4) f32++;
  }
  check('sliceOf == floor(log(z/zNear)/log(zFar/zNear)*gz)', same, 10000);
  check('float32 shader params give the same slice value (1e-4)', f32, 10000);
  check('clamped: z < zNear -> 0, z > zFar -> gz - 1, z <= 0 -> 0', cl.sliceOf(0.01) === 0 && cl.sliceOf(1e5) === gz - 1 && cl.sliceOf(-3) === 0 ? 1 : 0, 1);
  cl.dispose();
}

// --- assignment vs brute force ----------------------------------------------------------------------------------
// lights in world space around a camera; region test = what the shader lights (distance < range, and for spots
// angle cosine > cosOuter); brute force samples a 4 x 4 x 4 jittered grid + the 8 inner corners in every cell
function scenario({ camera, seed, nLights = 200, maxLights = 256, maxIndices = 65536, W = 1600, H = 900, spots = 0.3 }) {
  camera.updateWorldMatrix(true, false);
  const rnd = mulberry32(seed);
  const cl = new ClusteredLights({ maxLights, maxIndices, timing: true });
  cl.setSize(W, H);
  const inv = camera.matrixWorld;
  const lights = [];
  const persp = !!camera.isPerspectiveCamera;
  for (let i = 0; i < nLights; i++) {
    const kind = rnd();
    let v;
    if (kind < 0.6) { // in or near the view volume
      const d = persp ? 0.2 + Math.pow(rnd(), 2) * 450 : rnd() * 300;
      const hw = persp ? d * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * camera.aspect * 1.3 : (camera.right - camera.left) * 0.65;
      const hh = persp ? d * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * 1.3 : (camera.top - camera.bottom) * 0.65;
      v = new THREE.Vector3((rnd() * 2 - 1) * hw, (rnd() * 2 - 1) * hh, -d);
    } else if (kind < 0.8) v = new THREE.Vector3((rnd() * 2 - 1) * 10, (rnd() * 2 - 1) * 10, (rnd() * 2 - 1) * 10); // around the camera
    else v = new THREE.Vector3((rnd() * 2 - 1) * 400, (rnd() * 2 - 1) * 100, (rnd() * 2 - 1) * 400);
    v.applyMatrix4(inv); // view -> world
    const range = rnd() < 0.1 ? 40 + rnd() * 40 : 0.5 + rnd() * 25;
    const col = rnd() < 0.02 ? [0, 0, 0] : [rnd() * 5, rnd() * 5, rnd() * 5];
    const L = { i, p: v, range, col, spot: rnd() < spots };
    if (L.spot) {
      L.dir = new THREE.Vector3(rnd() * 2 - 1, rnd() * 2 - 1, rnd() * 2 - 1).normalize();
      const angle = THREE.MathUtils.degToRad(3 + rnd() * 87), pen = rnd();
      L.cosOuter = Math.cos(angle); L.cosInner = Math.cos(angle * (1 - pen));
      cl.setSpot(i, v.x, v.y, v.z, range, ...col, 2, L.dir.x, L.dir.y, L.dir.z, L.cosInner, L.cosOuter);
    } else cl.setLight(i, v.x, v.y, v.z, range, ...col, 2);
    lights.push(L);
  }
  cl.update(camera);
  const V = camera.matrixWorldInverse;
  for (const L of lights) {
    L.vp = L.p.clone().applyMatrix4(V);
    if (L.spot) L.vd = L.dir.clone().negate().transformDirection(V); // towards the light, as the shader compares
  }
  return { cl, lights, W, H };
}

function cellList(cl, c, out) {
  const w = cl.cellData[c], start = w & 0xffffff, n = w >>> 24;
  out.clear();
  for (let k = 0; k < n; k++) out.add(cl.cellData[start + k]);
  return n;
}

function bruteForce({ cl, lights, W, H }, camera, rnd, perCell = 4) {
  const [gx, gy, gz] = cl.grid;
  const P = camera.projectionMatrix.elements, persp = !!camera.isPerspectiveCamera;
  const list = new Set();
  let samples = 0, missed = 0, capped = 0, pairs = new Set(), assigned = 0;
  const camNear = persp ? camera.near : 0;
  for (let c = 0; c < gx * gy * gz; c++) assigned += cl.cellData[c] >>> 24;
  const pts = [];
  for (let c = 0; c < gx * gy * gz; c++) {
    const tx = c % gx, ty = ((c / gx) | 0) % gy, s = (c / (gx * gy)) | 0;
    const z0 = s === 0 ? Math.max(camNear, 0.01) : cl.sliceDepth(s), z1 = s === gz - 1 ? cl.zFar : cl.sliceDepth(s + 1);
    pts.length = 0;
    for (let a = 0; a < perCell; a++) for (let b = 0; b < perCell; b++) for (let d = 0; d < perCell; d++) pts.push([(a + rnd()) / perCell, (b + rnd()) / perCell, (d + rnd()) / perCell]);
    for (let k = 0; k < 8; k++) pts.push([k & 1 ? 1 - 1e-7 : 1e-7, k & 2 ? 1 - 1e-7 : 1e-7, k & 4 ? 1 - 1e-7 : 1e-7]);
    for (const [fa, fb, fc] of pts) {
      const fx = (tx + fa) * W / gx, fy = (ty + fb) * H / gy, depth = z0 + (z1 - z0) * fc;
      const nx = fx / W * 2 - 1, ny = fy / H * 2 - 1;
      let x, y;
      if (persp) { x = (nx + P[8]) / P[0] * depth; y = (ny + P[9]) / P[5] * depth; } else { x = (nx - P[12]) / P[0]; y = (ny - P[13]) / P[5]; }
      const cell = cl.cellOf(fx, fy, depth);
      if (cellList(cl, cell, list) >= MAX_PER_CELL) { capped++; continue; } // a full list may legitimately miss lights
      samples++;
      for (const L of lights) {
        if (L.col[0] + L.col[1] + L.col[2] === 0) continue;
        const dx = L.vp.x - x, dy = L.vp.y - y, dz = L.vp.z + depth;
        const d2 = dx * dx + dy * dy + dz * dz;
        if (d2 >= L.range * L.range) continue;
        if (L.spot) { const d = Math.sqrt(d2); if (!(d > 0) || (dx * L.vd.x + dy * L.vd.y + dz * L.vd.z) / d <= L.cosOuter) continue; }
        pairs.add(cell * 4096 + L.i);
        if (!list.has(L.i)) missed++;
      }
    }
  }
  return { samples, missed, capped, pairs: pairs.size, assigned };
}

{
  const camera = new THREE.PerspectiveCamera(60, 16 / 9, 0.1, 1000);
  camera.position.set(12, 3, -7); camera.lookAt(40, 1, -120);
  // a harsh mix (a fifth of the lights around the camera, a tenth with 40-80 m ranges): capacity raised so that only
  // the per-cell cap can drop pairs; points in cells whose list is full are not judged
  const sc = scenario({ camera, seed: 7, maxIndices: 1 << 20 });
  const st = sc.cl.stats;
  check('perspective: 200 lights, overflow', st.overflow, 0);
  check('perspective: clustered lights (in view)', st.lights, 40, 200);
  const bf = bruteForce(sc, camera, mulberry32(11));
  check(`perspective: brute force, ${bf.samples} points x 200 lights: missed`, bf.missed, 0);
  check('perspective: sampled true pairs found', bf.pairs, 1000, Infinity);
  check('perspective: assigned / sampled pairs (tightness)', st.indices / bf.pairs, 1, 4);
  check('perspective: update() ms (200 lights)', st.ms, 0, 50);
  // a tilted, rolled camera with an off-centre (view offset / jittered) projection
  const cam2 = new THREE.PerspectiveCamera(75, 4 / 3, 0.2, 800);
  cam2.position.set(-30, 25, 10); cam2.lookAt(0, 0, -60); cam2.rotateZ(0.4);
  cam2.setViewOffset(1600, 1200, 37.3, -21.7, 1600, 1200); // shifts elements 8 / 9
  const sc2 = scenario({ camera: cam2, seed: 99, W: 1600, H: 1200, maxIndices: 1 << 20 });
  const bf2 = bruteForce(sc2, cam2, mulberry32(5));
  check(`perspective (rolled, off-centre): missed of ${bf2.pairs} pairs`, bf2.missed, 0);
  check('perspective (rolled, off-centre): overflow', sc2.cl.stats.overflow, 0);
  // orthographic
  const ortho = new THREE.OrthographicCamera(-80, 80, 45, -45, -50, 600);
  ortho.position.set(0, 120, 60); ortho.lookAt(0, 0, -40);
  const sc3 = scenario({ camera: ortho, seed: 21, nLights: 120, maxIndices: 1 << 20 });
  const bf3 = bruteForce(sc3, ortho, mulberry32(8), 3);
  check(`orthographic: missed of ${bf3.pairs} pairs`, bf3.missed, 0);
  // a camera under a scaled parent: the view matrix carries the scale; three normalises the rotated spot axis and
  // measures `range` in view units, so must the encoding and the cone bounds
  const rig = new THREE.Group(); rig.scale.setScalar(2); rig.position.set(5, 0, 0); rig.rotation.y = 0.3;
  const camS = new THREE.PerspectiveCamera(60, 16 / 9, 0.1, 1000); rig.add(camS);
  camS.position.set(0, 2, 4); rig.updateMatrixWorld(true); camS.lookAt(10, 1, -100); rig.updateMatrixWorld(true);
  const sc4 = scenario({ camera: camS, seed: 33, nLights: 120, maxIndices: 1 << 20, spots: 0.6 });
  let decodeErr = 0, nSpots = 0;
  for (const Lt of sc4.lights) {
    if (!Lt.spot) continue;
    nSpots++;
    const o = (2 * 256 + Lt.i) * 4, D = sc4.cl.lightData, kk = Math.hypot(D[o], D[o + 1], D[o + 2]);
    const cosOuter = -D[o + 3] / kk, cosInner = cosOuter + 1 / kk;
    decodeErr = Math.max(decodeErr, Math.abs(D[o] / kk - Lt.vd.x) + Math.abs(D[o + 1] / kk - Lt.vd.y) + Math.abs(D[o + 2] / kk - Lt.vd.z),
      Math.abs(cosOuter - Lt.cosOuter), Math.abs(cosInner - Math.max(Lt.cosInner, Lt.cosOuter + 1e-4)));
  }
  check(`scaled camera parent: ${nSpots} spots decode to three's direction / cosines`, decodeErr, 0, 1e-4);
  const bf4 = bruteForce(sc4, camS, mulberry32(13), 3);
  check(`scaled camera parent: missed of ${bf4.pairs} pairs`, bf4.missed, 0);
  // determinism: same input, same data
  const a = scenario({ camera, seed: 7, maxIndices: 1 << 20 }), b = scenario({ camera, seed: 7, maxIndices: 1 << 20 });
  check('deterministic (cells + lights data)', Buffer.compare(Buffer.from(a.cl.cellData.buffer), Buffer.from(b.cl.cellData.buffer)) === 0
    && Buffer.compare(Buffer.from(a.cl.lightData.buffer), Buffer.from(b.cl.lightData.buffer)) === 0 ? 1 : 0, 1);
  // lists ascending, starting on texel boundaries, inside the index region
  let order = 0, align = 0, inside = 0;
  const { cl } = sc;
  for (let c = 0; c < cl.cellCount; c++) {
    const w = cl.cellData[c], s0 = w & 0xffffff, n = w >>> 24;
    if (!n) continue;
    if (s0 % 4) align++;
    if (s0 < cl.cellBase || s0 + n > cl.cellBase + cl.maxIndices) inside++;
    for (let k = 1; k < n; k++) if (cl.cellData[s0 + k] <= cl.cellData[s0 + k - 1]) order++;
  }
  check('index lists ascending', order, 0);
  check('index lists start on a texel (4 elements)', align, 0);
  check('index lists inside [base, base + maxIndices)', inside, 0);
  // lights wholly beyond zFar are dropped, ones reaching inside are kept in the last slice
  const far = new ClusteredLights({ maxLights: 4 });
  far.setSize(1600, 900);
  const fwd = new THREE.Vector3(); camera.getWorldDirection(fwd);
  const at = (d) => camera.position.clone().addScaledVector(fwd, d);
  const p1 = at(410), p2 = at(460);
  far.setLight(0, p1.x, p1.y, p1.z, 20, 1, 1, 1);
  far.setLight(1, p2.x, p2.y, p2.z, 20, 1, 1, 1);
  far.update(camera);
  const lst = new Set();
  cellList(far, far.cellOf(800, 450, 415), lst);
  check('beyond zFar: reaching light kept, distant one dropped', far.stats.lights === 1 && lst.has(0) && !lst.has(1) ? 1 : 0, 1);
  far.dispose();
}

// --- caps -------------------------------------------------------------------------------------------------------
{
  const camera = new THREE.PerspectiveCamera(60, 16 / 9, 0.1, 1000);
  camera.position.set(0, 2, 0); camera.lookAt(0, 2, -10); camera.updateMatrixWorld(true);
  const put = (cl, n) => { for (let i = 0; i < n; i++) cl.setLight(i, ((i * 7) % 11) - 5, 2, -20 - (i % 13), 30, 1, 1, 1); };
  const ref = new ClusteredLights({ maxLights: 100 }); ref.setSize(1600, 900); put(ref, 100); ref.update(camera);
  check('per-cell cap: maxPerCell above the cap', ref.stats.maxPerCell, MAX_PER_CELL + 1, 100);
  let capped = 0, total = 0;
  for (let c = 0; c < ref.cellCount; c++) { const n = ref.cellData[c] >>> 24; if (n > MAX_PER_CELL) capped++; total += n; }
  check(`per-cell cap: no list longer than ${MAX_PER_CELL}`, capped, 0);
  check('per-cell cap: overflow counted', ref.stats.overflow, 1, Infinity);
  check('per-cell cap: indices = stored pairs', ref.stats.indices, total);
  const small = new ClusteredLights({ maxLights: 100, maxIndices: 64 }); small.setSize(1600, 900); put(small, 100);
  small.cellData.fill(0xdeadbeef, small.cellBase + small.maxIndices);
  small.update(camera);
  let beyond = 0;
  for (let e = small.cellBase + small.maxIndices; e < small.cellData.length; e++) if (small.cellData[e] !== 0xdeadbeef) beyond++;
  check('index cap: nothing written beyond base + maxIndices', beyond, 0);
  check('index cap: stats.indices <= maxIndices', small.stats.indices, 1, 64);
  check('index cap: indices + overflow = all pairs', small.stats.indices + small.stats.overflow, ref.stats.indices + ref.stats.overflow);
  // a capped list is the lowest-index prefix of the uncapped one (near cells kept whole, far ones cut)
  const a = new Set(), b = new Set();
  let prefix = 0, cut = 0;
  for (let c = 0; c < small.cellCount; c++) {
    const n = cellList(small, c, a); cellList(ref, c, b);
    if (!n) continue;
    const full = [...b].sort((x, y) => x - y).slice(0, n), got = [...a].sort((x, y) => x - y);
    if (full.join() !== got.join()) prefix++;
    if (n < b.size) cut++;
  }
  check('index cap: every stored list is a prefix of the full list', prefix, 0);
  check('index cap: cells truncated', cut, 0, 2);
}

// --- texture layouts, strides, upload ranges --------------------------------------------------------------------
{
  const cl = new ClusteredLights(); // defaults
  const { lights, cells } = cl.textures;
  check('lights texture: 1024 x 3, RGBA32F, nearest', lights.image.width === 1024 && lights.image.height === 3 && lights.format === THREE.RGBAFormat
    && lights.type === THREE.FloatType && lights.minFilter === THREE.NearestFilter && lights.magFilter === THREE.NearestFilter && !lights.generateMipmaps ? 1 : 0, 1);
  check('lights data length (floats)', lights.image.data.length, 1024 * 4 * 3);
  check('cells texture: 1024 x 17, RGBA32UI, nearest', cells.image.width === 1024 && cells.image.height === Math.ceil((3456 + 65536) / 4096)
    && cells.format === THREE.RGBAIntegerFormat && cells.type === THREE.UnsignedIntType && cells.minFilter === THREE.NearestFilter ? 1 : 0, 1);
  check('cell base (index lists start)', cl.cellBase, 3456);
  const wide = new ClusteredLights({ maxLights: 1500 });
  check('1500 lights: 2 bands -> 1024 x 6 (the shared textures follow the bound instance)', wide.textures.lights.image.width === 1024 && wide.textures.lights.image.height === 6 ? 1 : 0, 1);
  const camera = new THREE.PerspectiveCamera(60, 16 / 9, 0.1, 1000);
  camera.position.set(1, 2, 3); camera.lookAt(0, 0, -50); camera.updateMatrixWorld(true);
  wide.setLight(1030, 4, 1, -30, 10, 0.5, 0.25, 0.125, 1.5);
  wide.update(camera);
  const vp = new THREE.Vector3(4, 1, -30).applyMatrix4(camera.matrixWorldInverse);
  const o0 = ((1 * 3 + 0) * 1024 + 6) * 4, o1 = ((1 * 3 + 1) * 1024 + 6) * 4, D = wide.lightData;
  check('band 1 row 0: view position + range', Math.abs(D[o0] - vp.x) + Math.abs(D[o0 + 1] - vp.y) + Math.abs(D[o0 + 2] - vp.z) + Math.abs(D[o0 + 3] - 10), 0, 1e-4);
  check('band 1 row 1: colour + decay', D[o1] === 0.5 && D[o1 + 1] === 0.25 && D[o1 + 2] === 0.125 && D[o1 + 3] === 1.5 ? 1 : 0, 1);
  wide.dispose();

  // spot encoding: row 0 w < 0, row 2 = direction towards the light * k, -cosOuter * k
  cl.setSize(1280, 720);
  const ci = Math.cos(0.3), co = Math.cos(0.5);
  cl.setSpot(5, 2, 6, -20, 25, 3, 2, 1, 2, 0, -1, -0.2, ci, co);
  cl.setLight(2, -3, 1, -15, 12, 1, 1, 1);
  cl.update(camera);
  const L = cl.lightData, r0 = 5 * 4, r2 = (2 * 1024 + 5) * 4;
  const k = Math.hypot(L[r2], L[r2 + 1], L[r2 + 2]);
  const dir = new THREE.Vector3(0, 1, 0.2).normalize().transformDirection(camera.matrixWorldInverse);
  check('spot: row 0 w = -range', L[r0 + 3], -25);
  check('spot: decoded direction (towards the light, view space)', Math.abs(L[r2] / k - dir.x) + Math.abs(L[r2 + 1] / k - dir.y) + Math.abs(L[r2 + 2] / k - dir.z), 0, 1e-5);
  check('spot: decoded coneCos / penumbraCos', Math.abs(-L[r2 + 3] / k - co) + Math.abs(-L[r2 + 3] / k + 1 / k - ci), 0, 1e-5);
  check('tiles uniform: tile px, 1 / size', cl.uniforms.engineClTiles.value.join() === [1280 / 16, 720 / 9, 1 / 1280, 1 / 720].map(Math.fround).join() ? 1 : 0, 1);
  check('params2.w = clustered light count', cl.uniforms.engineClParams2.value[3], 2);

  // upload ranges: every request holds everything the shader can read (rows 0 + 1 of [0, count), row 2 of the spot
  // span, the used prefix of the cells texture), from the first request after binding on
  const rowsOf = (t) => JSON.stringify(t.updateRanges.map((r) => [Math.floor(r.start / 4 / 1024), (r.start / 4) % 1024, r.count / 4]));
  const ALL6 = JSON.stringify([[0, 0, 6], [1, 0, 6], [2, 5, 1]]);
  check('first update after bind: ranges rows 0 + 1 [0, count), row 2 spot span', rowsOf(lights), ALL6);
  const used = cl.cellBase + cl.stats.cells * 4; // single lights: one padded texel each... at most
  const crs = cells.updateRanges;
  check('cells ranges: whole rows from 0, last one partial', crs.every((r, j) => r.start === j * 4096 && r.count <= 4096) && crs.length >= 1 ? 1 : 0, 1);
  check('cells ranges cover the used prefix', crs.reduce((s, r) => s + r.count, 0), cl.cellBase + 4, used);
  simulateUpload(lights); simulateUpload(cells);
  const v0 = lights.version;
  cl.setLight(2, -3, 1, -15, 12, 2, 2, 2);
  cl.update(camera);
  check('next frame: the same complete rows (no change tracking)', rowsOf(lights), ALL6);
  check('lights: needsUpdate (version bumped)', lights.version, v0 + 1);
  cl.update(camera); // the previous request was never uploaded: this one replaces it and is complete on its own
  check('a request three never uploaded is replaced by a complete one', rowsOf(lights), ALL6);
  simulateUpload(lights); simulateUpload(cells);
  cl.enabled = false; cl.update(camera);
  check('disabled: params2.w = 0', cl.uniforms.engineClParams2.value[3], 0);
  cl.enabled = true;
  cl.count = 3; cl.update(camera);
  check('count = 3 truncates (spot 5 cleared)', cl.stats.lights === 1 && cl.count === 3 ? 1 : 0, 1);
  cl.clearLight(2);
  check('clearLight(top) shrinks count', cl.count, 0);
  // no light reaches the view (daytime): loop off, nothing uploaded; the next lit frame queues all it reads
  simulateUpload(lights); simulateUpload(cells);
  const lv = lights.version, cv = cells.version;
  cl.setLight(1, 0, 0, 500, 5, 1, 1, 1); // behind the camera
  cl.update(camera);
  check('no visible light: lightCount 0, no upload queued', cl.uniforms.engineClParams2.value[3] === 0 && lights.version === lv && cells.version === cv
    && cl.stats.bytes === 0 && lights.updateRanges.length + cells.updateRanges.length === 0 ? 1 : 0, 1);
  cl.setLight(2, -3, 1, -15, 12, 1, 1, 1);
  cl.update(camera);
  check('next lit frame: rows 0 + 1 of [0, count)', rowsOf(lights), JSON.stringify([[0, 0, 3], [1, 0, 3]]));
  check('next lit frame: lightCount 1, cells queued', cl.uniforms.engineClParams2.value[3] === 1 && cells.version === cv + 1 && cells.updateRanges.length > 0 ? 1 : 0, 1);
  cl.dispose();
  check('dispose: shared textures back to 1 x 1, loop off', lights.image.width === 1 && cells.image.width === 1 && cl.uniforms.engineClParams2.value[3] === 0 ? 1 : 0, 1);
}

// --- GPU storage model --------------------------------------------------------------------------------------------
// three uploads a DataTexture with texStorage2D + texSubImage2D: on its first upload (and after any re-allocation:
// dispose, context restore, a second renderer) only the queued ranges reach storage whose other contents are not
// ours, and a request it never uploads is simply replaced by the next. Model: storage filled with garbage (NaN /
// 0xdeadbeef) whenever it is (re)allocated, synced with the queued ranges only on "rendered" frames; after each
// sync, walk exactly what the shader reads (cell word -> list -> light rows 0, 1, and 2 for spots) on the GPU copy.
{
  const camera = new THREE.PerspectiveCamera(60, 16 / 9, 0.1, 1000);
  const gpu = (tex) => ({ tex, data: null, version: -1 });
  function sync(g, fresh) {
    const t = g.tex, src = t.image.data;
    if (fresh || !g.data || g.data.length !== src.length) {
      g.data = new src.constructor(src.length).fill(src instanceof Float32Array ? NaN : 0xdeadbeef); g.version = -1;
    }
    if (t.version === g.version) return;
    if (t.updateRanges.length === 0) g.data.set(src);
    else for (const r of t.updateRanges) g.data.set(src.subarray(r.start, r.start + r.count), r.start);
    t.clearUpdateRanges(); g.version = t.version;
  }
  function readable(cl, gc, gl) {
    if (cl.uniforms.engineClParams2.value[3] === 0) return 0; // loop off: the shader reads nothing
    const CD = cl.cellData, LD = cl.lightData, LW = cl.lightsWidth;
    let bad = 0;
    for (let c = 0; c < cl.cellCount; c++) {
      const w = gc[c];
      if (w !== CD[c]) { bad++; continue; }
      const st = w & 0xffffff, n = w >>> 24;
      for (let j = 0; j < n; j++) {
        const li = gc[st + j];
        if (li !== CD[st + j] || !(li < cl.count)) { bad++; continue; }
        const o0 = (((li >> 10) * 3) * LW + (li & 1023)) * 4;
        for (let row = 0; row < 3; row++) {
          if (row === 2 && !(gl[o0 + 3] < 0)) break; // row 2: spots only (row 0 w < 0)
          for (let q = 0; q < 4; q++) if (!Object.is(gl[o0 + row * LW * 4 + q], LD[o0 + row * LW * 4 + q])) bad++;
        }
      }
    }
    return bad;
  }
  const rnd = mulberry32(17);
  const other = new ClusteredLights({ maxLights: 16 }); // binds first; cl takes the textures over below
  const cl = new ClusteredLights({ maxLights: 96 }); cl.setSize(1280, 720);
  const G = { c: gpu(cl.textures.cells), l: gpu(cl.textures.lights) };
  let frames = 0, synced = 0, fresh = 0, dark = 0, bad = 0;
  for (let f = 0; f < 400; f++) {
    // world changes: lights set / respotted / cleared / truncated, camera moves (lights leave and enter the view)
    for (let j = 0, m = 1 + Math.floor(rnd() * 6); j < m; j++) {
      const i = Math.floor(rnd() * 96), x = (rnd() * 2 - 1) * 60, z = -rnd() * 120, r = 2 + rnd() * 20, col = rnd() < 0.05 ? 0 : 1 + rnd();
      const u = rnd();
      if (u < 0.55) cl.setLight(i, x, 3, z, r, col, col * 0.8, col * 0.6, 2);
      else if (u < 0.85) cl.setSpot(i, x, 6, z, r, col, col, col, 2, rnd() - 0.5, -1, rnd() - 0.5, 0.95, 0.8);
      else cl.clearLight(i);
    }
    if (rnd() < 0.03) cl.count = Math.floor(rnd() * cl.count);
    const away = rnd() < 0.1; // looking at the sky: nothing in view
    camera.position.set((rnd() * 2 - 1) * 20, 2, 10 + rnd() * 10);
    camera.lookAt(camera.position.x + (rnd() * 2 - 1) * 20, away ? 1000 : 2, camera.position.z - 60);
    if (f === 200) other.update(camera); // another instance takes the textures, cl takes them back next update
    cl.update(camera);
    frames++;
    if (cl.uniforms.engineClParams2.value[3] === 0) dark++;
    if (rnd() < 0.3) continue; // nothing drawn with the textures this frame: three never uploads this request
    const re = f === 0 || f === 201 || rnd() < 0.05; // first upload, after the rebind, or storage re-allocated
    if (re) fresh++;
    sync(G.c, re); sync(G.l, re); synced++;
    bad += readable(cl, G.c.data, G.l.data);
  }
  check(`GPU model: ${synced} synced of ${frames} frames (${fresh} into fresh storage, ${dark} dark): values read != CPU`, bad, 0);
  check('GPU model: frames with lights in view', frames - dark, 200, 400);
  other.dispose(); cl.dispose();
}

// --- street grid at night, engine defaults: 320 lamps every 25 m on both sides of streets 100 m apart -----------
{
  const camera = new THREE.PerspectiveCamera(60, 16 / 9, 0.1, 1000);
  camera.position.set(3, 1.6, 10); camera.lookAt(3, 1.6, -100); camera.updateMatrixWorld(true);
  const cl = new ClusteredLights({ timing: true }); cl.setSize(3024, 1964);
  let i = 0;
  for (let a = -200; a <= 200; a += 100) for (let b = 0; b < 16; b++) for (const side of [-6, 6]) {
    cl.setLight(i++, a + side, 6, -b * 25, 20, 2, 1.6, 1.1); // streets along z
    cl.setLight(i++, -200 + b * 25, 6, -((a + 200) / 100) * 100 + side, 20, 2, 1.6, 1.1); // cross streets along x
  }
  for (let f = 0; f < 20; f++) { cl.update(camera); simulateUpload(cl.textures.lights); simulateUpload(cl.textures.cells); }
  const st = cl.stats;
  check(`street grid: ${i} lamps, ${st.lights} clustered, ${st.indices} pairs, overflow`, st.overflow, 0);
  check(`street grid: longest cluster list (cap ${MAX_PER_CELL})`, st.maxPerCell, 1, MAX_PER_CELL);
  check('street grid: bytes uploaded per frame (KB)', st.bytes / 1024, 1, 120);
  check('street grid: update() ms', st.ms, 0, 5);
  cl.dispose();
}

// --- 1024 lights: CPU cost --------------------------------------------------------------------------------------
{
  const camera = new THREE.PerspectiveCamera(60, 16 / 9, 0.1, 1000);
  camera.position.set(0, 3, 0); camera.lookAt(0, 3, -10); camera.updateMatrixWorld(true);
  const cl = new ClusteredLights({ maxLights: 1024, timing: true }); cl.setSize(3024, 1964);
  const rnd = mulberry32(4);
  for (let i = 0; i < 1024; i++) cl.setLight(i, (rnd() * 2 - 1) * 150, 4 + rnd() * 4, -rnd() * 400, 12 + rnd() * 12, 1, 0.8, 0.5);
  for (let f = 0; f < 50; f++) cl.update(camera); // warm-up
  let ms = 0;
  for (let f = 0; f < 200; f++) { camera.position.x = Math.sin(f * 0.05); cl.update(camera); ms += cl.stats.ms; }
  check(`1024 street lights: update() ms (${cl.stats.indices} pairs, ${cl.stats.cells} cells)`, ms / 200, 0, 5);
  check('1024 street lights: overflow share (far cells over the per-cell cap)', cl.stats.overflow / (cl.stats.indices + cl.stats.overflow), 0, 0.02);
  cl.dispose();
}

for (const r of results) console.log(r);
console.log(failures ? `\n${failures} check(s) failed` : `\nall ${results.length} checks passed`);
process.exit(failures ? 1 : 0);
