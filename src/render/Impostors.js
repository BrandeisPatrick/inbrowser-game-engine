// Billboard impostors: bake any set of meshes from a ring layout of view directions into an albedo atlas
// (sRGB, alpha, tint / vertex colour baked) and an object-space normal atlas, then draw thousands of instances
// as camera-facing quads that pick and blend the two nearest baked views per elevation ring (lit with the
// baked normals by a patched MeshStandardMaterial). Baking uses a short-lived offscreen WebGL context so it
// never touches the game renderer's state; 2x supersampled, box-filtered on the GPU (alpha-weighted), transparent
// texels dilated, mipmapped with coverage-preserving alpha. The atlas is texSize wide and only as many rows as the
// cells need (a power of two), so the CPU post-processing and the texture memory scale with the model count.
//
//   const baked = bakeImpostors([meshA, { geometry, map, center, radius }], { rings: DEFAULT_RINGS });
//   const field = new ImpostorBatch(baked, { capacity: 5000, fade: { start: 150, end: 185 } });
//   scene.add(field.mesh); field.setInstance(i, x, y, z, yaw, sx, sy, modelIndex); field.count = n;
//   per frame: field.update(camera.position)   (or share a uViewPos uniform via `viewPos`)
//
// API
//   DEFAULT_RINGS = [{ el: 10, n: 8 }, { el: 45, n: 8 }, { el: 80, n: 4 }]   (20 views per model)
//   impostorViews(rings) -> [{ az, el }]   (az radians from +z toward +x, el degrees)
//   bakeImpostors(models, opts) -> { albedo, normal, layout: { cols, cellU, cellV, pad, views, rings }, models } | null
//     models: THREE.Mesh | { geometry|geo, center?, radius?, map?, color? }
//     opts: rings, cellPx (170), texSize (2048), cols (texSize / cellPx), ss (2), map (Texture or mip levels [{data,width,height}]
//           shared by all models), alphaTest (0.5), normals (true), label ('impostors'), name ('impostor'), failNote,
//           timing (an object that receives the phase times in ms: context, render, downsample, post, total;
//           they are also printed with the console line)
//   createImpostorMaterial(baked, { viewPos, fade, color, roughness, metalness, envMapIntensity, alphaToCoverage })
//   ImpostorBatch(baked, { capacity, viewPos, fade, color, material }) — mesh, setInstance, setColor, count, update
import * as THREE from 'three';
import { FullScreenQuad } from 'three/addons/postprocessing/Pass.js';
import { buildMipChain, dataTextureFromMips, fillTransparent, dilateRGB } from './MipChain.js';
import { GLSL_IGN } from './MaterialPatches.js';
import { GBUFFER_VERTEX_TAIL } from './GBuffer.js'; // G-buffer velocity varyings (the bake shaders replace project_vertex)

export const DEFAULT_RINGS = [{ el: 10, n: 8 }, { el: 45, n: 8 }, { el: 80, n: 4 }];

/** View list for a ring layout (ring-major, azimuth k / n * 2pi). */
export function impostorViews(rings = DEFAULT_RINGS) {
  const out = [];
  for (const r of rings) for (let k = 0; k < r.n; k++) out.push({ az: (k / r.n) * Math.PI * 2, el: r.el });
  return out;
}

const BAKE_VS = /* glsl */`
varying vec2 vUv; varying vec3 vN; varying vec3 vC;
void main() {
	vUv = uv; vN = normal;
#ifdef HAS_VCOLOR
	vC = color;
#else
	vC = vec3( 1.0 );
#endif
	gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
}`;
const BAKE_COMMON_FS = /* glsl */`
uniform sampler2D map;
uniform vec3 uTint;
uniform float uAlphaTest;
varying vec2 vUv; varying vec3 vN; varying vec3 vC;
vec4 bakeTexel() {
#ifdef HAS_MAP
	vec4 t = texture2D( map, vUv );
#else
	vec4 t = vec4( 1.0 );
#endif
	if ( t.a < uAlphaTest ) discard;
	return t;
}
`;
const BAKE_ALBEDO_FS = BAKE_COMMON_FS + /* glsl */`
vec3 toSRGB( vec3 c ) { c = max( c, 0.0 ); return mix( c * 12.92, 1.055 * pow( c, vec3( 1.0 / 2.4 ) ) - 0.055, step( 0.0031308, c ) ); }
void main() {
	vec4 t = bakeTexel();
#ifdef HAS_TINT
	t.rgb *= uTint;
#endif
	gl_FragColor = vec4( toSRGB( t.rgb * vC ), 1.0 );
}`;
const BAKE_NORMAL_FS = BAKE_COMMON_FS + /* glsl */`
void main() { vec4 t = bakeTexel(); gl_FragColor = vec4( normalize( vN ) * 0.5 + 0.5, 1.0 ); }`;

// K x K box filter of a supersampled row target: alpha-weighted colour mean, mean alpha, uEmpty where nothing was drawn
const DOWN_VS = /* glsl */`
void main() { gl_Position = vec4( position.xy, 0.0, 1.0 ); }`;
const DOWN_FS = /* glsl */`
uniform sampler2D tSrc;
uniform vec3 uEmpty;
void main() {
	ivec2 o = ivec2( gl_FragCoord.xy ) * K;
	vec4 acc = vec4( 0.0 );
	for ( int j = 0; j < K; j ++ ) for ( int i = 0; i < K; i ++ ) { vec4 c = texelFetch( tSrc, o + ivec2( i, j ), 0 ); acc += vec4( c.rgb * c.a, c.a ); }
	gl_FragColor = acc.a > 0.0 ? vec4( acc.rgb / acc.a, acc.a / float( K * K ) ) : vec4( uEmpty, 0.0 );
}`;
// (three compiles WebGL2 shaders as GLSL ES 3.00 with its gl_FragColor alias, so texelFetch is available as is)
const downsampleMaterial = (k, empty) => new THREE.ShaderMaterial({
  vertexShader: DOWN_VS, fragmentShader: DOWN_FS, defines: { K: k },
  uniforms: { tSrc: { value: null }, uEmpty: { value: new THREE.Vector3(empty[0], empty[1], empty[2]) } },
  depthTest: false, depthWrite: false, blending: THREE.NoBlending,
});

function normModel(m) {
  const mesh = m && m.isMesh ? m : null;
  const geo = mesh ? mesh.geometry : m.geometry || m.geo;
  if (!geo.boundingSphere) geo.computeBoundingSphere();
  const mat = mesh ? (Array.isArray(mesh.material) ? mesh.material[0] : mesh.material) : null;
  return {
    geo,
    center: m.center || geo.boundingSphere.center,
    radius: m.radius || geo.boundingSphere.radius,
    map: m.map !== undefined ? m.map : mat ? mat.map || null : undefined,
    color: m.color || (mat && mat.color) || null,
  };
}

/**
 * Bake impostor atlases. Returns null without a DOM / WebGL (callers fall back to near-only rendering).
 * Throws only when the atlas cannot hold models x views cells.
 */
export function bakeImpostors(models, o = {}) {
  if (typeof document === 'undefined' || typeof WebGL2RenderingContext === 'undefined') return null; // headless: no bake
  const rings = o.rings || DEFAULT_RINGS, views = impostorViews(rings), VIEWS = views.length;
  const cellPx = o.cellPx ?? 170, texSize = o.texSize ?? 2048, ss = o.ss ?? 2;
  const cols = o.cols ?? Math.floor(texSize / cellPx);
  const label = o.label || 'impostors', name = o.name || 'impostor';
  const M = models.map(normModel);
  const total = M.length * VIEWS;
  const rows = Math.ceil(total / cols);
  if (rows * cellPx > texSize || cols * cellPx > texSize) throw new Error('impostor atlas too small');
  let r = null;
  const t0 = performance.now();
  const T = o.timing || {}, tm = (k, t) => { T[k] = (T[k] || 0) + (performance.now() - t); }; // phase times (ms), always collected: a few clock reads
  try {
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = 8;
    r = new THREE.WebGLRenderer({ canvas, antialias: false, alpha: true, powerPreference: 'high-performance' });
    r.setPixelRatio(1);
    tm('context', t0);
    r.autoClear = false; // cells share one row target; clear once per row
    const own = [];
    const shared = Array.isArray(o.map) ? (own.push(dataTextureFromMips(o.map, THREE.SRGBColorSpace)), own[0]) : o.map || null; // own Texture object for this context
    const matCache = new Map();
    const mk = (fs, map, hasColor, tint) => new THREE.ShaderMaterial({
      vertexShader: BAKE_VS, fragmentShader: fs, vertexColors: hasColor, side: THREE.DoubleSide,
      uniforms: { map: { value: map }, uTint: { value: new THREE.Color(tint ? tint.r : 1, tint ? tint.g : 1, tint ? tint.b : 1) }, uAlphaTest: { value: o.alphaTest ?? 0.5 } },
      defines: Object.assign({}, map ? { HAS_MAP: '' } : {}, hasColor ? { HAS_VCOLOR: '' } : {}, tint && (tint.r !== 1 || tint.g !== 1 || tint.b !== 1) ? { HAS_TINT: '' } : {}),
    });
    const matsFor = (m) => {
      const map = m.map === undefined ? shared : m.map || shared;
      const hasColor = !!m.geo.getAttribute('color');
      const key = `${map ? map.uuid : '-'}|${hasColor}|${m.color ? m.color.getHexString() : '-'}`;
      let pair = matCache.get(key);
      if (!pair) { pair = [mk(BAKE_ALBEDO_FS, map, hasColor, m.color), mk(BAKE_NORMAL_FS, map, hasColor, m.color)]; matCache.set(key, pair); }
      return pair;
    };
    const geos = M.map((m) => {
      const g = new THREE.BufferGeometry();
      for (const a of ['position', 'normal', 'uv', 'color']) if (m.geo.getAttribute(a)) g.setAttribute(a, m.geo.getAttribute(a));
      g.setIndex(m.geo.getIndex());
      if (!g.getAttribute('normal')) g.computeVertexNormals();
      return g;
    });
    const mats = M.map(matsFor);
    const scene = new THREE.Scene();
    const mesh = new THREE.Mesh(geos[0], mats[0][0]);
    mesh.frustumCulled = false;
    scene.add(mesh);
    const cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 100);
    const cpx = cellPx * ss, rowW = cols * cpx;
    const rt = new THREE.WebGLRenderTarget(rowW, cpx, { depthBuffer: true, generateMipmaps: false, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter });
    // atlas: texSize wide, as many rows as the cells need (power of two; the full square used to be processed)
    const W = texSize, usedH = rows * cellPx;
    let H = 1; while (H < usedH) H <<= 1; if (H > texSize) H = texSize;
    const NW = W >> 1, NH = H >> 1;
    const withNormals = o.normals !== false;
    const albedo = new Uint8Array(W * H * 4), normal = withNormals ? new Uint8Array(NW * NH * 4) : null;
    const ncell = cellPx >> 1;
    // the supersampled row is box-filtered on the GPU into a 1x row (albedo: ss x ss -> cellPx; normals: 2ss x 2ss
    // -> cellPx / 2 for the half-size normal atlas) and read back at that size: a quarter of the bytes, no CPU filter
    const lowOpts = { depthBuffer: false, generateMipmaps: false, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter };
    const lowA = new THREE.WebGLRenderTarget(cols * cellPx, cellPx, lowOpts), lowN = withNormals ? new THREE.WebGLRenderTarget(cols * ncell, ncell, lowOpts) : null;
    const bufA = new Uint8Array(cols * cellPx * cellPx * 4), bufN = withNormals ? new Uint8Array(cols * ncell * ncell * 4) : null;
    const downA = downsampleMaterial(ss, [0, 0, 0]), downN = withNormals ? downsampleMaterial(ss * 2, [0.5, 1, 0.5]) : null;
    const quad = new FullScreenQuad(downA);
    r.setClearColor(0x000000, 0);
    for (let pass = 0; pass < (withNormals ? 2 : 1); pass++) {
      const low = pass === 0 ? lowA : lowN, lowBuf = pass === 0 ? bufA : bufN, down = pass === 0 ? downA : downN;
      const cell = pass === 0 ? cellPx : ncell, atlas = pass === 0 ? albedo : normal, AW = pass === 0 ? W : NW, lw = cols * cell;
      for (let row = 0; row < rows; row++) {
        const tr = performance.now();
        rt.viewport.set(0, 0, rowW, cpx);
        r.setRenderTarget(rt);
        r.clear(true, true, false);
        for (let c = 0; c < cols; c++) {
          const idx = row * cols + c;
          if (idx >= total) break;
          const sp = Math.floor(idx / VIEWS), { az, el } = views[idx % VIEWS];
          const m = M[sp];
          mesh.geometry = geos[sp];
          mesh.material = mats[sp][pass];
          const R = m.radius, C = m.center, e = (el * Math.PI) / 180;
          cam.left = -R; cam.right = R; cam.top = R; cam.bottom = -R; cam.near = R * 0.5; cam.far = R * 6;
          cam.position.set(C.x + Math.sin(az) * Math.cos(e) * R * 3, C.y + Math.sin(e) * R * 3, C.z + Math.cos(az) * Math.cos(e) * R * 3);
          cam.up.set(0, 1, 0);
          cam.lookAt(C);
          cam.updateProjectionMatrix();
          rt.viewport.set(c * cpx, 0, cpx, cpx);
          r.setRenderTarget(rt);
          r.render(scene, cam);
        }
        down.uniforms.tSrc.value = rt.texture;
        quad.material = down;
        r.setRenderTarget(low);
        quad.render(r);
        r.readRenderTargetPixels(low, 0, 0, lw, cell, lowBuf);
        tm('render', tr);
        const td = performance.now();
        // the filtered row goes straight into the atlas rows (cells past the last model stay blank)
        const span = Math.min(cols, total - row * cols) * cell * 4;
        for (let y = 0; y < cell; y++) atlas.set(lowBuf.subarray(y * lw * 4, y * lw * 4 + span), (row * cell + y) * AW * 4);
        tm('downsample', td);
      }
    }
    // dilate colours / normals into transparent texels (per cell), then mip chains
    const tp = performance.now();
    const regA = [], regN = [];
    for (let idx = 0; idx < total; idx++) {
      const c = idx % cols, row = Math.floor(idx / cols);
      fillTransparent(albedo, W, c * cellPx, row * cellPx, cellPx, cellPx, 8);
      if (withNormals) for (let pass = 0; pass < 2; pass++) dilateRGB(normal, NW, c * ncell, row * ncell, ncell, ncell);
      regA.push({ x: c * cellPx, y: row * cellPx, w: cellPx, h: cellPx, cov: true });
      regN.push({ x: c * ncell, y: row * ncell, w: ncell, h: ncell, cov: false });
    }
    const albedoTex = dataTextureFromMips(buildMipChain(albedo, W, H, regA, true), THREE.SRGBColorSpace);
    const normalTex = withNormals ? dataTextureFromMips(buildMipChain(normal, NW, NH, regN, false), THREE.NoColorSpace) : null;
    albedoTex.name = `${name}Albedo`; if (normalTex) normalTex.name = `${name}Normal`;
    tm('post', tp);
    rt.dispose(); lowA.dispose(); if (lowN) lowN.dispose(); downA.dispose(); if (downN) downN.dispose(); quad.dispose();
    for (const p of matCache.values()) p.forEach((m) => m.dispose()); geos.forEach((g) => g.dispose()); own.forEach((t) => t.dispose());
    tm('total', t0);
    console.info(`[${label}] impostors baked: ${total} views in ${Math.round(performance.now() - t0)} ms (context ${Math.round(T.context)}, render ${Math.round(T.render)}, downsample ${Math.round(T.downsample)}, post ${Math.round(T.post)})`);
    return {
      albedo: albedoTex, normal: normalTex,
      layout: { cols, cellU: cellPx / W, cellV: cellPx / H, pad: 1.5 / cellPx, views: VIEWS, rings: rings.map((q) => ({ el: q.el, n: q.n })) },
      models: M.map((m) => ({ cx: m.center.x, cy: m.center.y, cz: m.center.z, r: m.radius })),
    };
  } catch (e) {
    console.warn(`[${label}] impostor bake failed${o.failNote || ''}`, e);
    return null;
  } finally {
    if (r) { r.dispose(); r.forceContextLoss(); }
  }
}

// ------------------------------------------------------------------ drawing
const MAX_RINGS = 4;

const IMP_VERT_PARS = /* glsl */`
uniform vec3 uViewPos;
uniform vec4 uFade;                    // x fade-in start, y fade-in end (end <= start: always visible)
uniform vec4 uModels[ IMP_MODELS ];    // local bounding sphere centre xyz, radius
uniform vec4 uAtlas;                   // columns, cell width (uv), cell height (uv), inner padding (cell fraction)
uniform vec4 uRings[ ${MAX_RINGS} ];   // elevation (deg), views in ring, first view index, -
uniform vec2 uRingInfo;                // ring count, views per model
attribute vec4 iPos;                   // x, base y, z, yaw
attribute vec4 iScl;                   // horizontal scale, vertical scale, model index, -
#ifdef IMP_COLOR
attribute vec3 iColor;
varying vec3 vIColor;
#endif
varying float vFade;
varying vec2 vQ;
varying vec4 vCA;
varying vec4 vCB;
varying vec3 vW;
varying vec2 vYaw;
vec3 impC; vec3 impR; vec3 impU; float impRw; float impRh;
vec2 impCell( float m, float view ) {
	float idx = m * uRingInfo.y + view;
	float col = mod( idx, uAtlas.x ), r = floor( idx / uAtlas.x );
	return vec2( col * uAtlas.y, r * uAtlas.z );
}
`;

const IMP_BEGINNORMAL = /* glsl */`
{
	float m = floor( iScl.z + 0.5 );
	vec4 S = uModels[ int( m ) ];
	float cy = cos( iPos.w ), sy = sin( iPos.w );
	vYaw = vec2( cy, sy );
	impC = iPos.xyz + vec3( S.x * cy + S.z * sy, S.y, - S.x * sy + S.z * cy ) * vec3( iScl.x, iScl.y, iScl.x );
	vec3 toCam = uViewPos - impC;
	float dist = length( toCam );
	vec3 f = toCam / max( dist, 1e-3 );
	impR = cross( vec3( 0.0, 1.0, 0.0 ), f );
	float rl = length( impR );
	impR = rl > 1e-4 ? impR / rl : vec3( 1.0, 0.0, 0.0 );
	impU = cross( f, impR );
	impRw = S.w * iScl.x; impRh = S.w * iScl.y;
	vFade = uFade.y > uFade.x ? smoothstep( uFade.x, uFade.y, dist ) : 1.0;
#ifdef IMP_COLOR
	vIColor = iColor;
#endif
	vQ = position.xy * 0.5 + 0.5;
	// view direction in the model's local frame -> two baked cells in each of the two nearest elevation rings
	vec3 fl = vec3( f.x * cy - f.z * sy, f.y, f.x * sy + f.z * cy );
	float az = fract( atan( fl.x, fl.z ) / 6.2831853 );
	float el = degrees( asin( clamp( fl.y, -1.0, 1.0 ) ) );
	int nr = int( uRingInfo.x + 0.5 );
	int r0 = 0;
	for ( int i = 1; i < ${MAX_RINGS}; i ++ ) if ( i < nr && el >= uRings[ i ].x ) r0 = i;
	int r1 = min( r0 + 1, nr - 1 );
	vec4 R0 = uRings[ r0 ], R1 = uRings[ r1 ];
	float we = r1 == r0 ? 0.0 : clamp( ( el - R0.x ) / ( R1.x - R0.x ), 0.0, 1.0 );
	float a = az * R0.y, a0 = floor( a ), wa = a - a0;
	float b = az * R1.y, b0 = floor( b ), wb = b - b0;
	vCA = vec4( impCell( m, R0.z + a0 ), impCell( m, R0.z + mod( a0 + 1.0, R0.y ) ) );
	vCB = vec4( impCell( m, R1.z + b0 ), impCell( m, R1.z + mod( b0 + 1.0, R1.y ) ) );
	vW = vec3( wa, wb, we );
}
vec3 objectNormal = normalize( uViewPos - impC );
`;

const IMP_BEGIN = /* glsl */`
vec3 transformed = impC + impR * position.x * impRw + impU * position.y * impRh;
if ( vFade <= 0.0 ) transformed = impC;
`;
const IMP_PROJECT = /* glsl */`
vec4 mvPosition = modelViewMatrix * vec4( transformed, 1.0 );
gl_Position = projectionMatrix * mvPosition;
` + GBUFFER_VERTEX_TAIL;
const IMP_WORLDPOS = /* glsl */`
#if defined( USE_ENVMAP ) || defined( DISTANCE ) || defined ( USE_SHADOWMAP ) || defined ( USE_TRANSMISSION ) || NUM_SPOT_LIGHT_COORDS > 0
	vec4 worldPosition = modelMatrix * vec4( transformed, 1.0 );
#endif
`;
const IMP_FRAG_PARS = /* glsl */`
uniform sampler2D uNormalAtlas;
uniform vec4 uAtlas;
varying float vFade;
varying vec2 vQ;
varying vec4 vCA;
varying vec4 vCB;
varying vec3 vW;
varying vec2 vYaw;
#ifdef IMP_COLOR
varying vec3 vIColor;
#endif
vec4 iNrm;
` + GLSL_IGN;
// complementary to MaterialPatches.ditherFade 'out' with the same start / end
const IMP_FADE_DISCARD = /* glsl */`
if ( vFade < 0.999 && mpIGN( gl_FragCoord.xy ) <= 1.0 - vFade ) discard;
`;
const IMP_MAP = /* glsl */`
vec2 iq = uAtlas.w + vQ * ( 1.0 - 2.0 * uAtlas.w );
vec2 ics = uAtlas.yz;
vec4 iA = mix( texture2D( map, vCA.xy + iq * ics ), texture2D( map, vCA.zw + iq * ics ), vW.x );
vec4 iB = mix( texture2D( map, vCB.xy + iq * ics ), texture2D( map, vCB.zw + iq * ics ), vW.y );
diffuseColor *= mix( iA, iB, vW.z );
#ifdef IMP_COLOR
diffuseColor.rgb *= vIColor;
#endif
#ifdef IMP_NORMALS
vec4 nA = mix( texture2D( uNormalAtlas, vCA.xy + iq * ics ), texture2D( uNormalAtlas, vCA.zw + iq * ics ), vW.x );
vec4 nB = mix( texture2D( uNormalAtlas, vCB.xy + iq * ics ), texture2D( uNormalAtlas, vCB.zw + iq * ics ), vW.y );
iNrm = mix( nA, nB, vW.z );
#endif
`;
const IMP_NORMAL = /* glsl */`
float faceDirection = 1.0;
#ifdef IMP_NORMALS
vec3 nObj = iNrm.xyz * 2.0 - 1.0;
nObj = length( nObj ) > 0.05 ? normalize( nObj ) : vec3( 0.0, 1.0, 0.0 );
vec3 nW = vec3( vYaw.x * nObj.x + vYaw.y * nObj.z, nObj.y, - vYaw.y * nObj.x + vYaw.x * nObj.z );
vec3 normal = normalize( ( viewMatrix * vec4( nW, 0.0 ) ).xyz );
#else
vec3 normal = normalize( vNormal );
#endif
vec3 nonPerturbedNormal = normal;
`;

/**
 * MeshStandardMaterial drawing an InstancedBufferGeometry quad per instance (attributes iPos, iScl, optional
 * iColor) from a bake. opts.viewPos: shared { value: Vector3 } camera-position uniform (else its own, set it
 * via material.userData.uViewPos). opts.fade: { start, end } dithered fade-in by distance (complements a near
 * model using ditherFade 'out' over the same range); null = always visible.
 */
export function createImpostorMaterial(baked, o = {}) {
  const L = baked.layout, rings = L.rings || DEFAULT_RINGS;
  if (rings.length > MAX_RINGS) throw new Error(`impostors: at most ${MAX_RINGS} elevation rings`);
  const nModels = Math.max(1, baked.models.length);
  const uViewPos = o.viewPos || { value: new THREE.Vector3() };
  const uFade = { value: new THREE.Vector4(o.fade ? o.fade.start : 0, o.fade ? o.fade.end : 0, 0, 0) };
  const uModels = { value: Array.from({ length: nModels }, (_, i) => { const s = baked.models[i]; return s ? new THREE.Vector4(s.cx, s.cy, s.cz, s.r) : new THREE.Vector4(0, 0, 0, 1); }) };
  const uAtlas = { value: new THREE.Vector4(L.cols, L.cellU, L.cellV, L.pad) };
  let first = 0;
  const uRings = { value: Array.from({ length: MAX_RINGS }, (_, i) => { const r = rings[i]; const v = r ? new THREE.Vector4(r.el, r.n, first, 0) : new THREE.Vector4(90, 1, 0, 0); if (r) first += r.n; return v; }) };
  const uRingInfo = { value: new THREE.Vector2(rings.length, first) };
  const uNormalAtlas = { value: baked.normal };
  const m = new THREE.MeshStandardMaterial({
    map: baked.albedo, roughness: o.roughness ?? 0.85, metalness: o.metalness ?? 0, alphaTest: o.alphaTest ?? 0.5,
    alphaToCoverage: o.alphaToCoverage ?? true, side: THREE.DoubleSide, envMapIntensity: o.envMapIntensity ?? 0.9,
  });
  m.defines = Object.assign({ IMP_MODELS: nModels }, baked.normal ? { IMP_NORMALS: '' } : {}, o.color ? { IMP_COLOR: '' } : {});
  m.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, { uViewPos, uFade, uModels, uAtlas, uRings, uRingInfo, uNormalAtlas });
    sh.vertexShader = sh.vertexShader.replace('#include <common>', '#include <common>\n' + IMP_VERT_PARS)
      .replace('#include <beginnormal_vertex>', IMP_BEGINNORMAL)
      .replace('#include <begin_vertex>', IMP_BEGIN)
      .replace('#include <project_vertex>', IMP_PROJECT)
      .replace('#include <worldpos_vertex>', IMP_WORLDPOS);
    sh.fragmentShader = sh.fragmentShader.replace('#include <common>', '#include <common>\n' + IMP_FRAG_PARS)
      .replace('#include <clipping_planes_fragment>', '#include <clipping_planes_fragment>\n' + IMP_FADE_DISCARD)
      .replace('#include <map_fragment>', IMP_MAP)
      .replace('#include <normal_fragment_begin>', IMP_NORMAL);
  };
  m.customProgramCacheKey = () => `engine-impostor-v1|${nModels}|${!!baked.normal}|${!!o.color}`;
  m.userData.uViewPos = uViewPos; m.userData.uFade = uFade;
  return m;
}

/**
 * A batch of impostor instances: one draw call. Instance data lives in typed arrays (no per-frame allocation);
 * capacity is fixed (create a new batch to grow).
 */
export class ImpostorBatch {
  constructor(baked, o = {}) {
    const cap = this.capacity = Math.max(1, o.capacity ?? 1024);
    const g = new THREE.InstancedBufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0], 3));
    g.setAttribute('normal', new THREE.Float32BufferAttribute([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], 3));
    g.setAttribute('uv', new THREE.Float32BufferAttribute([0, 0, 1, 0, 1, 1, 0, 1], 2));
    g.setIndex([0, 1, 2, 0, 2, 3]);
    this.iPos = new THREE.InstancedBufferAttribute(new Float32Array(cap * 4), 4);
    this.iScl = new THREE.InstancedBufferAttribute(new Float32Array(cap * 4), 4);
    g.setAttribute('iPos', this.iPos); g.setAttribute('iScl', this.iScl);
    if (o.color) { this.iColor = new THREE.InstancedBufferAttribute(new Float32Array(cap * 3).fill(1), 3); g.setAttribute('iColor', this.iColor); }
    g.instanceCount = 0;
    this.geometry = g;
    this.material = o.material || createImpostorMaterial(baked, o);
    this.mesh = new THREE.Mesh(g, this.material);
    this.mesh.frustumCulled = false; // instances span the world; the vertex shader collapses faded ones
    this.mesh.castShadow = false; this.mesh.receiveShadow = true;
    this.mesh.name = o.name || 'impostors';
    this._count = 0;
  }
  get count() { return this._count; }
  set count(n) { this._count = Math.max(0, Math.min(this.capacity, n | 0)); this.geometry.instanceCount = this._count; }
  /** Place instance i: base position, yaw (rad), horizontal / vertical scale, model index. */
  setInstance(i, x, y, z, yaw = 0, sx = 1, sy = sx, model = 0) {
    const p = this.iPos.array, s = this.iScl.array, k = i * 4;
    p[k] = x; p[k + 1] = y; p[k + 2] = z; p[k + 3] = yaw;
    s[k] = sx; s[k + 1] = sy; s[k + 2] = model; s[k + 3] = 0;
    this.iPos.needsUpdate = true; this.iScl.needsUpdate = true;
    if (i >= this._count) this.count = i + 1;
  }
  setColor(i, r, g, b) {
    if (!this.iColor) return;
    const c = this.iColor.array, k = i * 3; c[k] = r; c[k + 1] = g; c[k + 2] = b; this.iColor.needsUpdate = true;
  }
  /** Copy the camera position into the view uniform (skip when sharing a viewPos uniform updated elsewhere). */
  update(cameraPos) { this.material.userData.uViewPos.value.copy(cameraPos); }
  dispose() { this.geometry.dispose(); this.material.dispose(); }
}
