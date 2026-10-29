// G-buffer for the forward renderer: one multiple-render-target pass writes the lit colour plus a view-space normal /
// material target and a screen-space velocity target, for TAA, SSR, AO and other screen-space passes.
//
//   target (WebGLRenderTarget, count 3, samples 0: MSAA + MRT would resolve every attachment):
//     textures[0] colour    R11G11B10F (UnsignedInt101111Type + RGBFormat), or color: { format, type }. Every
//                           attachment format here needs EXT_color_buffer_float (WebGL2 core cannot render to RG16F
//                           or any float format): without it GBuffer.supported is false (warned) and nothing is patched
//     textures[1] normal    RGBA8, NoColorSpace, nearest: rg = octahedral view-space normal (* 0.5 + 0.5),
//                           b = roughness, a = (round(metalness * 15) << 4 | flags) / 255
//     textures[2] velocity  RG16F, nearest: (uvCur - uvPrev) of the surface, unjittered, UV units (history at uv - v)
//     depthTexture          DepthTexture, UnsignedIntType
//   flags (low nibble of normal.a): bit0 wroteGBuffer, bit1 hasNormal, bit3 noHistory (ENGINE_GBUFFER_NOVEL, or the
//   point was behind the previous camera). normal.a == 0 = nothing wrote this pixel: normal and velocity are
//   meaningless there (sky, transparent-only pixels, the clear colour); reproject from depth instead.
//   Later WebGPU: the attachments map to rg11b10ufloat / rgba8unorm / rg16float.
//
// API
//   installGBufferChunks() -> boolean   idempotent global patch of THREE.ShaderChunk + THREE.ShaderLib uniforms;
//                                       warns and returns false when a three.js anchor is missing
//   createGBufferTarget(w, h, { colorType, colorFormat, depthTexture }) -> WebGLRenderTarget
//   GBUFFER_UNIFORMS   shared uniform objects: enginePrevViewProj (previous UNJITTERED view-projection) and
//                      engineUnjitteredProj (this frame's unjittered projection), written by CameraHistory before
//                      the scene pass (mutate or reassign `.value`, never replace the uniform objects themselves);
//                      engineViewToPrevClip is derived from them per render
//   updateGBufferView(camera)   engineViewToPrevClip = enginePrevViewProj * camera.matrixWorld (doubles, on the CPU).
//                      Runs by itself before every render of a sync()ed Scene into an MRT target (chained
//                      scene.onBeforeRender); call it yourself when the scene root was not sync()ed
//   GBUFFER_VERTEX_TAIL / GBUFFER_CHUNK_TAILS   the appended GLSL, for shaders that replace a patched chunk
//   GBUFFER_GLSL_DECODE   engineOctDecode(rg), engineGbMetal(a), engineGbFlags(a) for consumer passes
//   new GBuffer(renderer, { width, height, color: { format, type }, nonWriters: 'zero' | 'mask', autoSync: true })
//     target, color, normal, velocity, depth; setSize(w, h); dispose(); stats (getter, same object)
//     sync(root) / unsync(root)   classify every material under root (see below); a Scene root also follows
//                                 'childadded' (direct children; deeper additions: autoSync, or sync() them)
//     track(object3d, { clone }) / untrack(object3d)   per-object velocity of dynamic meshes
//     trackPool(pool) / untrackPool(pool)               InstancedPool({ velocity: true }) instances
//     endFrame()      after the frame rendered: previous transforms <- current (allocation-free)
//     clear(renderer) attachments 1 and 2 to 0 (see "Clearing")
//
// sync(root): every material of every Mesh / Points / Line / Sprite under root is one of
//   writer  opaque (not transparent, Normal/No blending) MeshStandard/Physical/Lambert/Phong/Toon/Basic/Matcap
//           materials (userData.gbuffer !== false; 'novel' adds ENGINE_GBUFFER_NOVEL: velocity 0 + noHistory, for
//           surfaces whose image changes without moving, e.g. video screens or scrolling emissive signs; the sea stays a
//           plain writer: camera-only velocity is the better TAA trade for an animated surface that stays in place),
//           and opaque, non-raw ShaderMaterials with userData.gbuffer === true | 'novel' (warned when transparent or
//           raw; they get the GBUFFER_UNIFORMS objects in .uniforms and must include the chunks listed below, or append
//           GBUFFER_CHUNK_TAILS themselves) -> material.defines.ENGINE_GBUFFER = 1.
//           A ShaderMaterial whose author set ENGINE_GBUFFER itself (userData.gbuffer unset) is left as it is.
//   zero    every other material that draws into the target and can declare the outputs: built-in materials whose
//           fragment shader includes clipping_planes_pars_fragment (all but ShadowMaterial) and non-raw, non-GLSL3
//           ShaderMaterials (Sky, Particles, Decals, ...), with any blending but CustomBlending ->
//           ENGINE_GBUFFER_ZERO: main() is wrapped to also write vec4(0) to both extra outputs, which leaves the
//           G-buffer unchanged under every built-in blending mode (each draw buffer blends with its own source alpha,
//           0) and marks opaque unlit surfaces (the sky) as "not written". ShaderMaterials get it through a chained
//           onBeforeCompile that prepends the block; built-ins through the chunk. colorWrite: false materials (depth
//           pre-passes) take this class too (nothing is written; the outputs exist for Chrome's draw validation).
//   mask    the rest (CustomBlending, ShadowMaterial, RawShaderMaterial, GLSL3 ShaderMaterials; every non-writer with
//           nonWriters: 'mask') -> chained object onBeforeRender / onAfterRender hooks narrow the draw buffers to
//           attachment 0 while an MRT target is bound (gl.drawBuffers([COLOR_ATTACHMENT0, NONE, NONE]), then back to
//           [COLOR_ATTACHMENT0..n-1], which is what three's WebGLState cached for that framebuffer).
//           Cost: on ANGLE / Metal (Apple GPUs) a draw-buffer change alters the Metal render pass descriptor, which
//           ends the render encoder (FramebufferMtl::syncState -> onDrawFrameBufferChangedState, renderPassChanged):
//           each masked draw stores and reloads every attachment. Measured (headless Chrome, M4, 3024x1964, a lit
//           scene + N additive quads, ms per scene pass): N = 16: zero 0.7, mask 1.5; N = 64: 0.6-0.8 vs 1.9;
//           N = 256: 1.1-1.4 vs 2.3. Hence 'zero' by default; 'mask' stays for A/B and what cannot write zeros.
//   Every draw into the target needs this: Chrome rejects a draw whose fragment shader has no output for an active
//   draw buffer (GL_INVALID_OPERATION "Active draw buffers with missing fragment shader outputs"), so an
//   unclassified material simply disappears. autoSync (default) is the safety net: Material.prototype.onBeforeRender
//   (a no-op in three, called for every draw before the program is chosen) classifies a material the first time it
//   draws into this target without a G-buffer define - objects added below an existing child (MapWorld tiles),
//   swapped materials, three's own scene.background mesh - at the cost of two property checks per draw elsewhere.
//   Materials with their own onBeforeRender that skip the prototype's are not covered (sync() them).
//   Defines change -> material.needsUpdate (the program is rebuilt on its next draw). sync() re-asserts the defines
//   every time and follows a material whose class changed (transparent toggled). Not covered: scene.overrideMaterial
//   (classified like any material when autoSync sees it).
//
// GLSL contract (installGBufferChunks; every block is inert without its define: with no ENGINE_GBUFFER* define a
// shader preprocesses to the stock code, and the defines are the program-cache-key switches)
//   clipping_planes_pars_vertex  += #ifdef ENGINE_GBUFFER: uniform mat4 engineUnjitteredProj, engineViewToPrevClip;
//                                   varying vec4 vEngineClipCur, vEngineClipPrev; ENGINE_GBUFFER_PREV: uniform mat4
//                                   enginePrevModelView; USE_INSTANCING && ENGINE_GBUFFER_INSTANCE_PREV: attribute
//                                   mat4 instancePrevMatrix
//   project_vertex               += GBUFFER_VERTEX_TAIL: vEngineClipCur = engineUnjitteredProj * mvPosition;
//                                   vEngineClipPrev = engineViewToPrevClip * (mvPosition + d), where d is the
//                                   view-space displacement of the vertex since the previous frame: 0 for static
//                                   geometry (camera-only velocity), mat3(modelViewMatrix) * ((instancePrevMatrix -
//                                   instanceMatrix) * L) for InstancedPool velocity, + (enginePrevModelView -
//                                   modelViewMatrix) * L for tracked meshes (L = `transformed`, batched / instanced).
//                                   Everything stays in camera-relative space (enginePrevModelView and
//                                   engineViewToPrevClip are composed in doubles on the CPU), so far-from-origin
//                                   worlds keep exact velocity (ANGLE / Metal, 3 km from the origin: within 0.001 px
//                                   of a CPU reference for camera, tracked-mesh and instance motion), and offsets
//                                   that patches add to mvPosition between the instance and the model-view transform
//                                   (MaterialPatches windSway, VegMaterials vegWind) are part of both positions
//                                   instead of reading as motion (a still frame measures |velocity| < 1e-5).
//                                   Skinned / morphed meshes: `transformed` is this frame's pose, so velocity is the
//                                   model-matrix motion only. Shaders that replace project_vertex append
//                                   GBUFFER_VERTEX_TAIL after setting mvPosition / gl_Position.
//   clipping_planes_pars_fragment += #ifdef ENGINE_GBUFFER: (only when gl_FragColor is defined, i.e. not GLSL3)
//                                   layout(location = 1) out highp vec4 engineGbNormal; layout(location = 2) out highp
//                                   vec4 engineGbVelocity; varyings; globals vec3 engineGbN (0), float
//                                   engineGbRoughness (1), engineGbMetalness (0), engineGbFlagSum (0); engineOctEncode().
//                                   #if ENGINE_GBUFFER_ZERO && !ENGINE_GBUFFER: the zero-writing main() wrapper
//                                   (`#define main engineGbUserMain`). GLSL3 ShaderMaterials that opt in declare
//                                   both outputs themselves (vec4 velocity: the zero path needs its alpha).
//   normal_fragment_maps         += engineGbN = normal; engineGbFlagSum += 2.0;  (final view-space normal; shaders
//                                   that replace this chunk, e.g. Water / Buildings, append the tail or set engineGbN)
//   roughnessmap_fragment        += engineGbRoughness = roughnessFactor;
//   metalnessmap_fragment        += engineGbMetalness = metalnessFactor;
//   dithering_fragment           += the writes (last chunk of every lit mesh shader).
//
// Clearing: gl.clear (renderer.clear, autoClear, PostFX lateClear) writes the clear colour to all three attachments.
// The cheap way is clear alpha 0 for the scene render (renderer.setClearAlpha(0) around it): normal.a = 0 = "not
// written" and nothing else is read there. clear(renderer) instead runs gl.clearBufferfv(gl.COLOR, 1 | 2, 0) on this
// target; call it after the colour clear and before the scene draws (on ANGLE / Metal a clear issued before the
// shadow passes costs a store + load of the target).
import * as THREE from 'three';

const D_WRITE = 'ENGINE_GBUFFER';
const D_NOVEL = 'ENGINE_GBUFFER_NOVEL';
const D_ZERO = 'ENGINE_GBUFFER_ZERO';
const D_PREV = 'ENGINE_GBUFFER_PREV';
const D_INSTANCE_PREV = 'ENGINE_GBUFFER_INSTANCE_PREV';
/** The defines this module reads (all are per material; values are ignored, presence counts). */
export const GBUFFER_DEFINES = Object.freeze({ WRITE: D_WRITE, NOVEL: D_NOVEL, ZERO: D_ZERO, PREV: D_PREV, INSTANCE_PREV: D_INSTANCE_PREV });

/**
 * Shared uniform objects (mutate `.value` in place). CameraHistory writes enginePrevViewProj and engineUnjitteredProj
 * each frame before the scene pass; engineViewToPrevClip is derived (updateGBufferView).
 */
export const GBUFFER_UNIFORMS = {
  enginePrevViewProj: { value: new THREE.Matrix4() },
  engineUnjitteredProj: { value: new THREE.Matrix4() },
  engineViewToPrevClip: { value: new THREE.Matrix4() },
};

/**
 * engineViewToPrevClip = enginePrevViewProj * camera.matrixWorld: current view space -> previous clip space.
 * @param {THREE.Camera} camera the camera rendering the G-buffer (matrixWorld up to date)
 */
export function updateGBufferView(camera) {
  GBUFFER_UNIFORMS.engineViewToPrevClip.value.multiplyMatrices(GBUFFER_UNIFORMS.enginePrevViewProj.value, camera.matrixWorld);
}

const VERTEX_PARS = /* glsl */`
#ifdef ENGINE_GBUFFER
	uniform mat4 engineUnjitteredProj;
	uniform mat4 engineViewToPrevClip;
	varying vec4 vEngineClipCur;
	varying vec4 vEngineClipPrev;
	#ifdef ENGINE_GBUFFER_PREV
		uniform mat4 enginePrevModelView;
	#endif
	#if defined( USE_INSTANCING ) && defined( ENGINE_GBUFFER_INSTANCE_PREV )
		attribute mat4 instancePrevMatrix;
	#endif
#endif
`;

/** Appended to project_vertex (needs mvPosition, transformed, modelViewMatrix; instanceMatrix / batchingMatrix when used). */
export const GBUFFER_VERTEX_TAIL = /* glsl */`
#ifdef ENGINE_GBUFFER
	{
		vec3 engineGbD = vec3( 0.0 );
		#if defined( ENGINE_GBUFFER_PREV ) || ( defined( USE_INSTANCING ) && defined( ENGINE_GBUFFER_INSTANCE_PREV ) )
			vec4 engineGbL = vec4( transformed, 1.0 );
			#ifdef USE_BATCHING
				engineGbL = batchingMatrix * engineGbL;
			#endif
			vec3 engineGbDL = vec3( 0.0 );
			#if defined( USE_INSTANCING ) && defined( ENGINE_GBUFFER_INSTANCE_PREV )
				engineGbDL = ( ( instancePrevMatrix - instanceMatrix ) * engineGbL ).xyz;
			#endif
			#ifdef ENGINE_GBUFFER_PREV
				#ifdef USE_INSTANCING
					engineGbL = instanceMatrix * engineGbL;
				#endif
				engineGbD = mat3( enginePrevModelView ) * engineGbDL + ( ( enginePrevModelView - modelViewMatrix ) * engineGbL ).xyz;
			#else
				engineGbD = mat3( modelViewMatrix ) * engineGbDL;
			#endif
		#endif
		vEngineClipCur = engineUnjitteredProj * mvPosition;
		vEngineClipPrev = engineViewToPrevClip * vec4( mvPosition.xyz + engineGbD, 1.0 );
	}
#endif
`;

const FRAGMENT_PARS = /* glsl */`
#ifdef ENGINE_GBUFFER
	#ifdef gl_FragColor
		layout( location = 1 ) out highp vec4 engineGbNormal;
		layout( location = 2 ) out highp vec4 engineGbVelocity;
	#endif
	varying highp vec4 vEngineClipCur;
	varying highp vec4 vEngineClipPrev;
	vec3 engineGbN = vec3( 0.0 );
	float engineGbRoughness = 1.0;
	float engineGbMetalness = 0.0;
	float engineGbFlagSum = 0.0;
	vec2 engineOctEncode( vec3 n ) {
		n /= abs( n.x ) + abs( n.y ) + abs( n.z );
		return n.z >= 0.0 ? n.xy : ( 1.0 - abs( n.yx ) ) * vec2( n.x >= 0.0 ? 1.0 : - 1.0, n.y >= 0.0 ? 1.0 : - 1.0 );
	}
#endif
`;

/** Zero-writing main() wrapper (include-guarded: a ShaderMaterial gets it prepended and may include the chunk too). */
const ZERO_MAIN = /* glsl */`
#if defined( ENGINE_GBUFFER_ZERO ) && ! defined( ENGINE_GBUFFER ) && defined( gl_FragColor ) && ! defined( ENGINE_GBUFFER_ZERO_MAIN )
	#define ENGINE_GBUFFER_ZERO_MAIN
	layout( location = 1 ) out highp vec4 engineGbNormal;
	layout( location = 2 ) out highp vec4 engineGbVelocity;
	void engineGbUserMain();
	void main() {
		engineGbUserMain();
		engineGbNormal = vec4( 0.0 );
		engineGbVelocity = vec4( 0.0 );
	}
	#define main engineGbUserMain
#endif
`;

const NORMAL_TAIL = /* glsl */`
#ifdef ENGINE_GBUFFER
	engineGbN = normal;
	engineGbFlagSum += 2.0;
#endif
`;
const ROUGHNESS_TAIL = /* glsl */`
#ifdef ENGINE_GBUFFER
	engineGbRoughness = roughnessFactor;
#endif
`;
const METALNESS_TAIL = /* glsl */`
#ifdef ENGINE_GBUFFER
	engineGbMetalness = metalnessFactor;
#endif
`;
const WRITE_TAIL = /* glsl */`
#ifdef ENGINE_GBUFFER
	{
		highp vec2 engineGbVel = vec2( 0.0 );
		#ifdef ENGINE_GBUFFER_NOVEL
			engineGbFlagSum += 8.0;
		#else
			if ( vEngineClipPrev.w > 0.0 ) engineGbVel = ( vEngineClipCur.xy / vEngineClipCur.w - vEngineClipPrev.xy / vEngineClipPrev.w ) * 0.5;
			else engineGbFlagSum += 8.0;
		#endif
		vec2 engineGbE = dot( engineGbN, engineGbN ) > 0.0 ? engineOctEncode( normalize( engineGbN ) ) * 0.5 + 0.5 : vec2( 0.5 );
		float engineGbA = floor( clamp( engineGbMetalness, 0.0, 1.0 ) * 15.0 + 0.5 ) * 16.0 + engineGbFlagSum + 1.0;
		engineGbNormal = vec4( engineGbE, clamp( engineGbRoughness, 0.0, 1.0 ), engineGbA / 255.0 );
		engineGbVelocity = vec4( engineGbVel, 0.0, 0.0 );
	}
#endif
`;

/**
 * The GLSL appended to each patched chunk. Every block starts and ends with a newline, so it can be concatenated
 * after any code (e.g. `.replace('#include <normal_fragment_maps>', myNormal + GBUFFER_CHUNK_TAILS.normal_fragment_maps)`).
 */
export const GBUFFER_CHUNK_TAILS = Object.freeze({
  clipping_planes_pars_vertex: VERTEX_PARS,
  project_vertex: GBUFFER_VERTEX_TAIL,
  clipping_planes_pars_fragment: FRAGMENT_PARS + ZERO_MAIN.slice(1),
  normal_fragment_maps: NORMAL_TAIL,
  roughnessmap_fragment: ROUGHNESS_TAIL,
  metalnessmap_fragment: METALNESS_TAIL,
  dithering_fragment: WRITE_TAIL,
});

/**
 * Decode helpers for passes that read the G-buffer (paste once into a shader): engineOctDecode(normal.rg) -> unit
 * view-space normal, engineGbMetal(normal.a) -> metalness 0..1, engineGbFlags(normal.a) -> bits (ENGINE_GB_WROTE 1,
 * ENGINE_GB_HAS_NORMAL 2, ENGINE_GB_NO_HISTORY 8). Velocity is texture(velocity, uv).rg (history at uv - rg).
 */
export const GBUFFER_GLSL_DECODE = /* glsl */`
const int ENGINE_GB_WROTE = 1;
const int ENGINE_GB_HAS_NORMAL = 2;
const int ENGINE_GB_NO_HISTORY = 8;
vec3 engineOctDecode( vec2 e ) {
	vec2 f = e * 2.0 - 1.0;
	vec3 n = vec3( f, 1.0 - abs( f.x ) - abs( f.y ) );
	float t = max( - n.z, 0.0 );
	n.x += n.x >= 0.0 ? - t : t;
	n.y += n.y >= 0.0 ? - t : t;
	return normalize( n );
}
int engineGbByte( float a ) { return int( a * 255.0 + 0.5 ); }
float engineGbMetal( float a ) { return float( engineGbByte( a ) >> 4 ) / 15.0; }
int engineGbFlags( float a ) { return engineGbByte( a ) & 15; }
`;

// Shared values for the built-in templates. UniformsUtils.clone() copies Matrix4 values per material at compile time,
// so the templates hold plain objects whose `elements` getter reads the live matrix: cloned by reference, uploaded
// by three's mat4 setter like a Matrix4, and a reassigned GBUFFER_UNIFORMS value is still followed.
const liveMatrix = (u) => ({ get elements() { return u.value.elements; } });
const TEMPLATE_UNIFORMS = {
  engineUnjitteredProj: liveMatrix(GBUFFER_UNIFORMS.engineUnjitteredProj),
  engineViewToPrevClip: liveMatrix(GBUFFER_UNIFORMS.engineViewToPrevClip),
};

let installed = null;

/**
 * Install the chunk + template patch (idempotent). false (with a warning) when three's chunks changed.
 * @returns {boolean}
 */
export function installGBufferChunks() {
  if (installed !== null) return installed;
  const C = THREE.ShaderChunk, L = THREE.ShaderLib;
  if (typeof C.project_vertex === 'string' && C.project_vertex.includes('vEngineClipCur')) return (installed = true); // another copy of this module
  const lit = ['basic', 'lambert', 'phong', 'standard', 'physical', 'toon', 'matcap'];
  const need = [
    [C.project_vertex, 'mvPosition = modelViewMatrix * mvPosition;'],
    [C.project_vertex, 'gl_Position = projectionMatrix * mvPosition;'],
    [C.normal_fragment_begin, 'vec3 normal ='],
    [C.roughnessmap_fragment, 'float roughnessFactor'],
    [C.metalnessmap_fragment, 'float metalnessFactor'],
    [C.dithering_fragment, 'DITHERING'],
    [C.clipping_planes_pars_vertex, 'NUM_CLIPPING_PLANES'],
    [C.clipping_planes_pars_fragment, 'NUM_CLIPPING_PLANES'],
    [C.normal_fragment_maps, 'normal ='],
  ];
  for (const k of lit) {
    const s = L[k] || {};
    need.push([globalBeforeMain(s.vertexShader, 'clipping_planes_pars_vertex') ? s.vertexShader : null, '#include <project_vertex>']);
    need.push([globalBeforeMain(s.fragmentShader, 'clipping_planes_pars_fragment') ? lastInclude(s.fragmentShader) : null, 'dithering_fragment']);
  }
  if (!need.every(([text, s]) => typeof text === 'string' && text.includes(s))) {
    console.warn('[render] G-buffer: three.js shader chunks changed, the G-buffer is unavailable');
    return (installed = false);
  }
  for (const k in GBUFFER_CHUNK_TAILS) C[k] = appendBlock(C[k], GBUFFER_CHUNK_TAILS[k]);
  for (const k in L) {
    const s = L[k];
    if (!s || !s.uniforms || typeof s.vertexShader !== 'string' || !s.vertexShader.includes('#include <clipping_planes_pars_vertex>')) continue;
    for (const u in TEMPLATE_UNIFORMS) if (!s.uniforms[u]) s.uniforms[u] = { value: TEMPLATE_UNIFORMS[u] };
  }
  return (installed = true);
}

// chunk + block, keeping the chunk's own trailing-newline convention (the stripped block leaves the chunk as it was)
function appendBlock(chunk, block) { return chunk.endsWith('\n') ? chunk + block.slice(1) : chunk + block.slice(0, -1); }
function globalBeforeMain(src, chunk) {
  if (typeof src !== 'string') return false;
  const i = src.indexOf(`#include <${chunk}>`), m = src.search(/void\s+main\s*\(/);
  return i >= 0 && m > i;
}
function lastInclude(src) {
  const all = typeof src === 'string' ? src.match(/#include <[a-z_]+>/g) : null;
  return all ? all[all.length - 1] : null;
}

/**
 * The scene target: count 3, samples 0, colour textures[0] (colorType / colorFormat, default HalfFloat RGBA),
 * normal RGBA8 and velocity RG16F (both NoColorSpace, nearest), a DepthTexture (UnsignedIntType unless given).
 * @param {number} width
 * @param {number} height
 * @param {{colorType?: number, colorFormat?: number, depthTexture?: THREE.DepthTexture}} [o]
 * @returns {THREE.WebGLRenderTarget}
 */
export function createGBufferTarget(width, height, { colorType = THREE.HalfFloatType, colorFormat = THREE.RGBAFormat, depthTexture } = {}) {
  const w = Math.max(1, width | 0), h = Math.max(1, height | 0);
  const rt = new THREE.WebGLRenderTarget(w, h, {
    count: 3, samples: 0, type: colorType, format: colorFormat, depthBuffer: true, stencilBuffer: false,
    depthTexture: depthTexture || new THREE.DepthTexture(w, h, THREE.UnsignedIntType),
    generateMipmaps: false, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter,
  });
  const [c, n, v] = rt.textures;
  c.name = 'GBuffer.color';
  n.name = 'GBuffer.normal'; n.type = THREE.UnsignedByteType; n.format = THREE.RGBAFormat;
  v.name = 'GBuffer.velocity'; v.type = THREE.HalfFloatType; v.format = THREE.RGFormat;
  for (const t of [n, v]) { t.colorSpace = THREE.NoColorSpace; t.minFilter = t.magFilter = THREE.NearestFilter; }
  return rt;
}

// ---- material classification ---------------------------------------------------------------------------------
const WRITE = 1, ZERO = 2, MASK = 3, AUTHOR = 4, SKIP = 0;
const hasOwn = (o, k) => Object.prototype.hasOwnProperty.call(o, k);
const isBuiltinWriter = (m) => !!(m.isMeshStandardMaterial || m.isMeshLambertMaterial || m.isMeshPhongMaterial
  || m.isMeshToonMaterial || m.isMeshBasicMaterial || m.isMeshMatcapMaterial);
// built-in materials whose fragment shader includes clipping_planes_pars_fragment at global scope (ShadowMaterial does not)
const hasZeroChunk = (m) => !!(isBuiltinWriter(m) || m.isMeshNormalMaterial || m.isMeshDepthMaterial || m.isMeshDistanceMaterial
  || m.isLineBasicMaterial || m.isPointsMaterial || m.isSpriteMaterial);
const plainShader = (m) => m.isShaderMaterial && !m.isRawShaderMaterial && m.glslVersion !== THREE.GLSL3;
const blended = (m) => m.transparent === true || (m.blending !== THREE.NormalBlending && m.blending !== THREE.NoBlending);
const hasDefine = (m, d) => !!(m.defines && m.defines[d] !== undefined);

/** Set or remove one define; true when it changed. */
function setDefine(m, d, on) {
  if (on === hasDefine(m, d)) return false;
  if (on) (m.defines || (m.defines = {}))[d] = 1;
  else delete m.defines[d]; // an emptied defines object keys and compiles like none
  return true;
}

/**
 * Chain fn after material.onBeforeCompile. The cache key keeps what the previous chain produced (its own
 * customProgramCacheKey, or the previous onBeforeCompile's source, which three's default key is) plus `tag`, so two
 * materials whose earlier onBeforeCompile differ never share a program.
 */
function chainCompile(m, fn, tag) {
  const rec = { ownOBC: hasOwn(m, 'onBeforeCompile'), ownKey: hasOwn(m, 'customProgramCacheKey'), prev: m.onBeforeCompile, prevKey: m.customProgramCacheKey };
  const prev = rec.prev, prevKey = rec.prevKey;
  rec.obc = function (shader, renderer) { prev.call(this, shader, renderer); fn(shader, renderer); };
  rec.key = rec.ownKey ? function () { return `${prevKey.call(this)}|${tag}`; } : function () { return `${prev.toString()}|${tag}`; };
  m.onBeforeCompile = rec.obc;
  m.customProgramCacheKey = rec.key;
  return rec;
}
/** Undo chainCompile when nothing chained on top since (else the chain stays: its fn is inert without its define). */
function unchainCompile(m, rec) {
  if (m.onBeforeCompile !== rec.obc || m.customProgramCacheKey !== rec.key) return false;
  if (rec.ownOBC) m.onBeforeCompile = rec.prev; else delete m.onBeforeCompile;
  if (rec.ownKey) m.customProgramCacheKey = rec.prevKey; else delete m.customProgramCacheKey;
  return true;
}
/** Chain fn on o[name] (before: prev runs first; fnFirst: fn runs first). Returns an unchain record. */
function chainMethod(o, name, fn, fnFirst = false) {
  const rec = { o, name, own: hasOwn(o, name), prev: o[name], active: true };
  const prev = rec.prev;
  rec.f = fnFirst
    ? function (a, b, c, d, e, f) { if (rec.active) fn.call(this, a, b, c, d, e, f); prev.call(this, a, b, c, d, e, f); }
    : function (a, b, c, d, e, f) { prev.call(this, a, b, c, d, e, f); if (rec.active) fn.call(this, a, b, c, d, e, f); };
  o[name] = rec.f;
  return rec;
}
function unchainMethod(rec) {
  rec.active = false; // still called (and a no-op) if something chained on top
  if (rec.o[rec.name] !== rec.f) return;
  if (rec.own) rec.o[rec.name] = rec.prev; else delete rec.o[rec.name];
}

// ShaderMaterials writing zeros: the block prepended once per material, whichever GBuffer classified it
const ZERO_CHAINS = new WeakMap();
const prependZero = (shader) => { if (!shader.fragmentShader.includes('ENGINE_GBUFFER_ZERO_MAIN')) shader.fragmentShader = ZERO_MAIN + shader.fragmentShader; };

// ---- draw-buffer masks (hooks of 'mask' objects) --------------------------------------------------------------
const GL_COLOR_ATTACHMENT0 = 0x8ce0, GL_NONE = 0;
const MASKS = [], FULLS = [];
let maskedCount = 0; // > 0 while a hook narrowed the draw buffers of an MRT target with that many attachments
function drawBufferList(lists, n, full) {
  let l = lists[n];
  if (!l) { l = lists[n] = []; for (let i = 0; i < n; i++) l.push(i === 0 || full ? GL_COLOR_ATTACHMENT0 + i : GL_NONE); }
  return l;
}
function maskBegin(renderer, material) {
  if (hasDefine(material, D_WRITE) || hasDefine(material, D_ZERO)) return;
  const rt = renderer.getRenderTarget();
  if (rt === null || !rt.textures || rt.textures.length < 2) return;
  maskedCount = rt.textures.length;
  renderer.getContext().drawBuffers(drawBufferList(MASKS, maskedCount, false));
}
function maskEnd(renderer) {
  if (maskedCount === 0) return;
  renderer.getContext().drawBuffers(drawBufferList(FULLS, maskedCount, true)); // what three's WebGLState cached
  maskedCount = 0;
}

// renders of a sync()ed Scene into an MRT target refresh the view -> previous-clip matrix
function sceneBeforeRender(renderer, scene, camera, renderTarget) {
  if (renderTarget && renderTarget.textures && renderTarget.textures.length > 1 && camera) updateGBufferView(camera);
}

// autoSync: GBuffers whose target gets the draw-time classification. Material.prototype.onBeforeRender (a no-op in
// three) runs for every draw after the object's hooks and before the program is chosen; while no GBuffer is alive,
// or for draws into other targets, it returns after two checks. Materials with their own onBeforeRender that do not
// call the prototype's are not covered (sync() them).
const LIVE = [];
let lazyInstalled = false;
function lazyMaterialHook(renderer, scene, camera, geometry, object) {
  if (LIVE.length === 0) return;
  const d = this.defines;
  if (d && (d[D_WRITE] !== undefined || d[D_ZERO] !== undefined)) return;
  const rt = renderer.getRenderTarget();
  if (rt === null) return;
  for (let i = 0; i < LIVE.length; i++) if (LIVE[i].target === rt) { LIVE[i]._lazy(this, object, renderer); return; }
}
function installLazyHook() {
  if (lazyInstalled) return;
  lazyInstalled = true;
  const P = THREE.Material.prototype, base = P.onBeforeRender; // three's no-op, or whatever replaced it
  P.onBeforeRender = function (renderer, scene, camera, geometry, object, group) {
    base.call(this, renderer, scene, camera, geometry, object, group);
    lazyMaterialHook.call(this, renderer, scene, camera, geometry, object, group);
  };
}

const ZERO4 = new Float32Array(4);
const _size = new THREE.Vector2();

const OWN_HOOKS = ['onBeforeCompile', 'customProgramCacheKey', 'onBeforeRender'];

/**
 * Shallow material copy for per-object uniforms: shared uniform objects, textures, defines and the material's own
 * patches. `unwound(key)` gives the own function the material had before this module chained onto it (undefined:
 * none), so the clone gets the user's hooks, not another mesh's velocity hook or a second zero-writer chain; the
 * module's defines are dropped too (the caller classifies and patches the clone itself).
 */
function cloneMaterial(m, unwound) {
  const ud = m.userData, u = m.isShaderMaterial ? m.uniforms : null;
  m.userData = {}; // Material.copy JSON-clones userData and ShaderMaterial.copy clones every uniform value
  if (u) m.uniforms = {};
  let c;
  try { c = m.clone(); } finally { m.userData = ud; if (u) m.uniforms = u; }
  c.userData = { ...ud };
  if (u) { c.uniforms = { ...u }; delete c.uniforms.enginePrevModelView; }
  if (m.defines) c.defines = { ...m.defines }; // built-in copy() resets defines to the type's own
  for (const k of OWN_HOOKS) { const f = unwound(k); if (f !== undefined) c[k] = f; }
  if (c.defines) { delete c.defines[D_PREV]; delete c.defines[D_ZERO]; }
  return c;
}

export class GBuffer {
  /**
   * @param {THREE.WebGLRenderer} renderer
   * @param {object} [o]
   * @param {number} [o.width]  drawing-buffer px (default: the renderer's drawing buffer)
   * @param {number} [o.height]
   * @param {{format?: number, type?: number}} [o.color] colour attachment (default R11G11B10F); every attachment
   *   needs EXT_color_buffer_float: without it `supported` is false (warned) and sync / track do nothing
   * @param {'zero'|'mask'} [o.nonWriters='zero'] how materials that do not write the G-buffer keep it intact
   * @param {boolean} [o.autoSync=true] classify unseen materials when they draw into this target (see _lazy)
   */
  constructor(renderer, { width, height, color, nonWriters = 'zero', autoSync = true } = {}) {
    this.renderer = renderer;
    // RG16F velocity (and R11G11B10F / RGBA16F colour) are colour-renderable only with EXT_color_buffer_float
    const floatOk = !(renderer && renderer.extensions && renderer.extensions.has) || renderer.extensions.has('EXT_color_buffer_float');
    if (!floatOk) console.warn('[GBuffer] EXT_color_buffer_float is unavailable: the G-buffer target cannot be rendered (supported = false)');
    /** False when the G-buffer cannot work here (three.js chunks changed, or no EXT_color_buffer_float). */
    this.supported = floatOk && installGBufferChunks();
    let w = width, h = height;
    if (!(w > 0 && h > 0) && renderer && renderer.getDrawingBufferSize) { renderer.getDrawingBufferSize(_size); w = _size.x; h = _size.y; }
    const fmt = color && (color.format !== undefined || color.type !== undefined)
      ? { colorFormat: color.format ?? THREE.RGBAFormat, colorType: color.type ?? THREE.HalfFloatType }
      : { colorFormat: THREE.RGBFormat, colorType: THREE.UnsignedInt101111Type };
    this.target = createGBufferTarget(w || 1, h || 1, fmt);
    this.nonWriters = nonWriters === 'mask' ? 'mask' : 'zero';
    this._stats = { materialsDefined: 0, materialsZero: 0, materialsMasked: 0, tracked: 0, pools: 0 };
    this._statsDirty = false;
    this._mats = new Map();      // material -> { kind, novel, own, chain }
    this._hooks = new Map();     // object -> { before, after } chain records (a Map: dispose() unhooks them all)
    this._scenes = new Map();    // Scene -> { onChildAdded, before }
    this._recs = [];             // tracked meshes (array: endFrame iterates without an iterator)
    this._recOf = new Map();     // mesh -> record
    this._owner = new Map();     // material -> record (a tracked material belongs to one mesh)
    this._pools = [];
    this._warned = new Set();
    this._changed = 0;
    this._syncObject = (o) => this._syncObj(o);
    this._unsyncObject = (o) => this._unsyncObj(o);
    this._onMatDispose = (e) => this._forget(e.target);
    this.autoSync = !!autoSync && this.supported;
    if (this.autoSync) { installLazyHook(); LIVE.push(this); }
  }

  /**
   * { materialsDefined (writers), materialsZero, materialsMasked, tracked (meshes), pools }: materials classified and
   * not unsynced / disposed since. The same object every time, recounted when something changed.
   */
  get stats() { if (this._statsDirty) this._countStats(); return this._stats; }

  get color() { return this.target.textures[0]; }
  get normal() { return this.target.textures[1]; }
  get velocity() { return this.target.textures[2]; }
  get depth() { return this.target.depthTexture; }

  /** Drawing-buffer size (reallocates the attachments when it changes). */
  setSize(width, height) { this.target.setSize(Math.max(1, width | 0), Math.max(1, height | 0)); }

  /**
   * Classify every material under root (header). A Scene root also gets a 'childadded' listener and the chained
   * onBeforeRender that keeps engineViewToPrevClip current. Returns the number of materials whose defines changed.
   */
  sync(root) {
    if (!root || !this.supported) return 0;
    if (root.isScene && !this._scenes.has(root)) {
      const rec = { onChildAdded: (e) => this.sync(e.child), before: chainMethod(root, 'onBeforeRender', sceneBeforeRender) };
      root.addEventListener('childadded', rec.onChildAdded);
      this._scenes.set(root, rec);
      if (!this.autoSync && root.background && root.background.isTexture) this._warnOnce('background', '[GBuffer] scene.background textures are drawn by an internal three.js mesh that sync() cannot see: use autoSync, or draw the sky as a scene object');
    }
    this._changed = 0;
    root.traverse(this._syncObject);
    this._statsDirty = true;
    return this._changed;
  }

  /** Undo sync() under root: defines and zero-writer chains removed, hooks removed, a Scene's listeners detached. */
  unsync(root) {
    if (!root) return;
    root.traverse(this._unsyncObject);
    const s = this._scenes.get(root);
    if (s) { root.removeEventListener('childadded', s.onChildAdded); unchainMethod(s.before); this._scenes.delete(root); }
    this._statsDirty = true;
  }

  /**
   * Per-object velocity for dynamic meshes (every Mesh under object3d; skinned meshes: model-matrix motion only).
   * Each tracked material needs its own uniform: a material shared with another mesh in the same scene graph (or
   * with another tracked mesh) is cloned (shallow: same textures, shared uniform objects, same patches) and a
   * warning is printed once; { clone: true } always clones. Track after adding to the scene, so that sharing is
   * visible. A clone of a tracked mesh (Object3D.clone) shares its material: track it too. Returns the count added.
   */
  track(object3d, { clone = false } = {}) {
    if (!object3d) return 0;
    const meshes = [];
    object3d.traverse((o) => { if (o.isMesh && o.material && !this._recOf.has(o)) meshes.push(o); });
    if (!meshes.length) return 0;
    let top = object3d;
    while (top.parent) top = top.parent;
    const first = new Map(), shared = new Set();
    top.traverse((o) => {
      if (!o.material) return;
      const list = Array.isArray(o.material) ? o.material : [o.material];
      for (const m of list) {
        if (!m) continue;
        const f = first.get(m);
        if (f === undefined) first.set(m, o); else if (f !== o) shared.add(m);
      }
    });
    for (const mesh of meshes) this._trackMesh(mesh, shared, clone);
    this._statsDirty = true;
    return meshes.length;
  }

  /** Stop tracking the meshes under object3d (their materials keep their clones, minus the velocity patch). */
  untrack(object3d) {
    if (!object3d) return;
    object3d.traverse((o) => {
      const rec = this._recOf.get(o);
      if (rec) this._untrackRec(rec);
    });
    this._statsDirty = true;
  }

  /** endFrame() calls pool.capturePrev() (InstancedPool with velocity: true). */
  trackPool(pool) { if (pool && !this._pools.includes(pool)) this._pools.push(pool); this._statsDirty = true; }
  untrackPool(pool) { const i = this._pools.indexOf(pool); if (i >= 0) this._pools.splice(i, 1); this._statsDirty = true; }

  /** After the frame rendered: previous transforms <- this frame's, for tracked meshes and pools. Allocation-free. */
  endFrame() {
    const r = this._recs;
    for (let i = 0; i < r.length; i++) {
      const rec = r[i];
      // a mesh that was not drawn this frame has no usable previous transform next frame
      if (rec.drawn) rec.prevWorld.copy(rec.mesh.matrixWorld);
      rec.hasPrev = rec.drawn;
      rec.drawn = false;
    }
    const p = this._pools;
    for (let i = 0; i < p.length; i++) p[i].capturePrev();
  }

  /**
   * Attachments 1 (normal) and 2 (velocity) of this target to 0 with gl.clearBufferfv; colour and depth untouched.
   * Restores the previous render target. See the header for when (or whether) to call it.
   */
  clear(renderer = this.renderer) {
    const prev = renderer.getRenderTarget(), face = renderer.getActiveCubeFace(), mip = renderer.getActiveMipmapLevel();
    if (prev !== this.target) renderer.setRenderTarget(this.target);
    const gl = renderer.getContext();
    renderer.state.buffers.color.setMask(true); // clearBuffer* honours the colour write mask
    gl.clearBufferfv(gl.COLOR, 1, ZERO4);
    gl.clearBufferfv(gl.COLOR, 2, ZERO4);
    if (prev !== this.target) renderer.setRenderTarget(prev, face, mip);
  }

  dispose() {
    for (const scene of [...this._scenes.keys()]) this.unsync(scene);
    for (const o of [...this._hooks.keys()]) this._unhook(o); // also objects hooked at draw time outside any synced root
    for (const [m, rec] of this._mats) { if (this._revert(m, rec)) m.needsUpdate = true; m.removeEventListener('dispose', this._onMatDispose); }
    this._mats.clear();
    while (this._recs.length) this._untrackRec(this._recs[this._recs.length - 1]);
    this._pools.length = 0;
    this._statsDirty = true;
    const i = LIVE.indexOf(this);
    if (i >= 0) LIVE.splice(i, 1);
    this.target.dispose();
  }

  // ---- internals ----------------------------------------------------------------------------------------------
  _syncObj(o) {
    if (!(o.isMesh || o.isPoints || o.isLine || o.isSprite)) return;
    const mat = o.material;
    let mask = false;
    if (Array.isArray(mat)) { for (let i = 0; i < mat.length; i++) if (mat[i] && this._syncMaterial(mat[i]) === MASK) mask = true; }
    else if (mat && this._syncMaterial(mat) === MASK) mask = true;
    if (mask) this._hook(o); else this._unhook(o);
  }

  _unsyncObj(o) {
    this._unhook(o);
    const mat = o.material;
    if (!mat) return;
    const list = Array.isArray(mat) ? mat : [mat];
    for (const m of list) {
      const rec = m && this._mats.get(m);
      if (!rec) continue;
      if (this._revert(m, rec)) m.needsUpdate = true;
      m.removeEventListener('dispose', this._onMatDispose);
      this._mats.delete(m);
    }
  }

  _kind(m) {
    if (m.visible === false) return SKIP; // not drawn (classified again when it draws: see _lazy)
    const zeroOk = plainShader(m) || (!m.isShaderMaterial && hasZeroChunk(m));
    if (m.colorWrite === false) return zeroOk ? ZERO : MASK; // writes nothing; ZERO declares the outputs at no cost
    const ud = m.userData ? m.userData.gbuffer : undefined;
    const opaque = !blended(m);
    if (m.isShaderMaterial) {
      if (ud === true || ud === 'novel') {
        if (opaque && !m.isRawShaderMaterial) return WRITE;
        this._warnOnce(`optin:${m.uuid}`, `[GBuffer] ShaderMaterial '${m.name || m.uuid}' has userData.gbuffer = ${ud} but is ${m.isRawShaderMaterial ? 'a RawShaderMaterial' : 'blended'}: it does not write the G-buffer`);
      }
      const prev = this._mats.get(m);
      if (ud === undefined && hasDefine(m, D_WRITE) && !(prev && prev.own)) return AUTHOR;
    } else if (opaque && ud !== false && isBuiltinWriter(m)) return WRITE;
    if (this.nonWriters === 'zero' && m.blending !== THREE.CustomBlending && zeroOk) return ZERO;
    return MASK;
  }

  /** Classify + apply one material; returns its kind. */
  _syncMaterial(m) {
    const kind = this._kind(m);
    const novel = kind === WRITE && m.userData.gbuffer === 'novel';
    let rec = this._mats.get(m), changed = false;
    if (rec === undefined) {
      rec = { kind, novel, own: false, chain: null };
      this._mats.set(m, rec);
      m.addEventListener('dispose', this._onMatDispose);
      this._statsDirty = true;
    } else if (rec.kind !== kind || rec.novel !== novel) {
      changed = this._revert(m, rec);
      rec.kind = kind; rec.novel = novel; rec.own = false; rec.chain = null;
      this._statsDirty = true;
    }
    // (re)assert idempotently: a define removed elsewhere (another GBuffer's unsync, user code) comes back
    if (kind === WRITE) {
      changed = setDefine(m, D_WRITE, true) || changed;
      changed = setDefine(m, D_NOVEL, novel) || changed;
      rec.own = true;
      if (m.isShaderMaterial) {
        for (const k of ['engineUnjitteredProj', 'engineViewToPrevClip']) if (!m.uniforms[k]) m.uniforms[k] = GBUFFER_UNIFORMS[k];
      }
    } else if (kind === ZERO) {
      changed = setDefine(m, D_ZERO, true) || changed;
      if (m.isShaderMaterial && !ZERO_CHAINS.has(m)) { rec.chain = chainCompile(m, prependZero, 'engine-gbzero'); ZERO_CHAINS.set(m, rec.chain); }
    }
    if (changed) { m.needsUpdate = true; this._changed++; }
    return kind;
  }

  /**
   * Draw-time safety net (autoSync, via Material.prototype.onBeforeRender while this target is bound): a material
   * this GBuffer has not classified yet (an object added below an existing child, a swapped material, three's own
   * background mesh) is classified before its program is chosen; Chrome would otherwise reject the draw (no output
   * for an active draw buffer). A newly masked material hooks its object and masks this draw.
   */
  _lazy(m, object, renderer) {
    let rec = this._mats.get(m);
    if (rec === undefined || rec.kind === SKIP) { this._syncMaterial(m); rec = this._mats.get(m); } // SKIP: was invisible
    if (rec.kind === MASK && object && !this._hooks.has(object)) { this._hook(object); maskBegin(renderer, m); }
  }

  /** Remove what _syncMaterial added; true when a define changed. */
  _revert(m, rec) {
    let changed = false;
    if (rec.kind === WRITE && rec.own) { changed = setDefine(m, D_WRITE, false) || changed; changed = setDefine(m, D_NOVEL, false) || changed; }
    if (rec.kind === ZERO) {
      changed = setDefine(m, D_ZERO, false) || changed;
      if (rec.chain) { unchainCompile(m, rec.chain); ZERO_CHAINS.delete(m); }
    }
    return changed;
  }

  _forget(m) {
    m.removeEventListener('dispose', this._onMatDispose);
    this._mats.delete(m);
    this._statsDirty = true;
  }

  _hook(o) {
    if (this._hooks.has(o)) return;
    this._hooks.set(o, {
      before: chainMethod(o, 'onBeforeRender', function (renderer, scene, camera, geometry, material) { maskBegin(renderer, material); }),
      after: chainMethod(o, 'onAfterRender', function (renderer) { maskEnd(renderer); }, true), // restore before any nested render
    });
  }

  _unhook(o) {
    const h = this._hooks.get(o);
    if (!h) return;
    unchainMethod(h.before); unchainMethod(h.after);
    this._hooks.delete(o);
  }

  _trackMesh(mesh, shared, forceClone) {
    const rec = { mesh, prevWorld: new THREE.Matrix4().copy(mesh.matrixWorld), u: { value: new THREE.Matrix4() }, hasPrev: false, drawn: false, mats: [] };
    const list = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
    const out = [], done = new Map();
    let replaced = false;
    for (const m of list) {
      if (!m) { out.push(m); continue; }
      let t = done.get(m);
      if (t === undefined) {
        t = m;
        if (forceClone || shared.has(m) || this._owner.has(m)) {
          if (!forceClone) this._warnOnce('shared', `[GBuffer] track(): material '${m.name || m.type}' is shared; cloned for per-object velocity (warned once)`);
          t = cloneMaterial(m, (k) => this._unwound(m, k));
          if (this._mats.has(m)) this._syncMaterial(t);
        }
        this._patchTracked(t, rec);
        done.set(m, t);
      }
      if (t !== m) replaced = true;
      out.push(t);
    }
    if (replaced) mesh.material = Array.isArray(mesh.material) ? out : out[0];
    this._recOf.set(mesh, rec);
    this._recs.push(rec);
  }

  /**
   * The own value m[key] had before this module's chains (tracked-mesh compile / render hooks, zero-writer compile
   * chain), unwinding them innermost-last; undefined when it had none (the prototype's).
   */
  _unwound(m, key) {
    const chains = [];
    const owner = this._owner.get(m);
    if (owner) for (const e of owner.mats) if (e.m === m) { if (e.compile) chains.push(e.compile); chains.push(e.render); }
    const z = ZERO_CHAINS.get(m);
    if (z) chains.push(z);
    let own = hasOwn(m, key), value = m[key];
    for (let n = 0; own && n <= chains.length; n++) {
      let hit = false;
      for (const c of chains) {
        if (key === 'onBeforeCompile' && c.obc === value) { own = c.ownOBC; value = c.prev; hit = true; break; }
        if (key === 'customProgramCacheKey' && c.key === value) { own = c.ownKey; value = c.prevKey; hit = true; break; }
        if (key === 'onBeforeRender' && c.f === value) { own = c.own; value = c.prev; hit = true; break; }
      }
      if (!hit) break;
    }
    return own ? value : undefined;
  }

  _patchTracked(m, rec) {
    const entry = { m, compile: null, render: null };
    if (m.isShaderMaterial) m.uniforms.enginePrevModelView = rec.u;
    else entry.compile = chainCompile(m, (shader) => { shader.uniforms.enginePrevModelView = rec.u; }, 'engine-gbprev');
    // per draw, with the drawing camera: the previous model matrix in this frame's view space (doubles), uploaded
    // with the material's uniforms (the material is unique to this mesh, so three refreshes them for each draw)
    entry.render = chainMethod(m, 'onBeforeRender', function (renderer, scene, camera, geometry, object) {
      if (object === rec.mesh) {
        rec.u.value.multiplyMatrices(camera.matrixWorldInverse, rec.hasPrev ? rec.prevWorld : object.matrixWorld);
        rec.drawn = true;
      } else rec.u.value.multiplyMatrices(camera.matrixWorldInverse, object.matrixWorld); // a stray sharer: no motion
    });
    setDefine(m, D_PREV, true);
    m.needsUpdate = true;
    this._owner.set(m, rec);
    rec.mats.push(entry);
  }

  _untrackRec(rec) {
    for (const e of rec.mats) {
      const m = e.m;
      if (e.compile) unchainCompile(m, e.compile);
      else if (m.uniforms && m.uniforms.enginePrevModelView === rec.u) delete m.uniforms.enginePrevModelView;
      unchainMethod(e.render);
      if (setDefine(m, D_PREV, false)) m.needsUpdate = true;
      if (this._owner.get(m) === rec) this._owner.delete(m);
    }
    rec.mats.length = 0;
    this._recOf.delete(rec.mesh);
    const i = this._recs.indexOf(rec);
    if (i >= 0) { this._recs[i] = this._recs[this._recs.length - 1]; this._recs.pop(); }
  }

  _countStats() {
    let d = 0, z = 0, k = 0;
    for (const rec of this._mats.values()) { if (rec.kind === WRITE || rec.kind === AUTHOR) d++; else if (rec.kind === ZERO) z++; else if (rec.kind === MASK) k++; }
    const s = this._stats;
    s.materialsDefined = d; s.materialsZero = z; s.materialsMasked = k; s.tracked = this._recs.length; s.pools = this._pools.length;
    this._statsDirty = false;
  }

  _warnOnce(key, msg) {
    if (this._warned.has(key)) return;
    this._warned.add(key);
    console.warn(msg);
  }
}
