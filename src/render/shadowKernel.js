// Opt-in directional shadow kernels. 'pcf' (default) is three's PCFShadowMap: r180 packs shadow depth into
// an RGBA8 colour target, so every lookup is 17 nearest fetches + unpack + compare (34 in the cascade blend band),
// 1.5 ms of a 15.8 ms Canyon Run frame on the terrain alone at 3024x1964. 'fast' samples a hardware depth texture
// that replaces the depth renderbuffer of the same shadow render target, through a sampler2DShadow: each tap is a
// 2x2 bilinear PCF done by the texture unit, 5 taps per lookup. Same maps, matrices, bias and normalBias.
//
// Like the cascade patch (shadowCascade.js) it is a global ShaderChunk patch, inert unless selected: appended to
// shadowmap_pars_fragment under `ENGINE_SHADOW_FAST && SHADOWMAP_TYPE_PCF` are a sampler2DShadow array, a getShadow()
// overload taking a sampler2DShadow and `#define directionalShadowMap <that array>`, so every directional lookup
// (three's light loop, the cascade blend, ShadowMaterial's mask) resolves to the overload with no call site edited,
// while spot shadows keep getShadow( sampler2D, ... ). The patch is installed on the first 'fast' selection and,
// without the define, preprocesses to the stock code: a page that never selects 'fast' compiles byte-identical
// shaders. Selection is per material (the define is in three's program cache key; chunk text is not), see
// applyShadowKernel; Atmosphere.setShadowKernel wires all of it for its cascades.
//
// Kernel: centre + 4 taps on a rotated grid at s * (2,1), (-1,2), (-2,-1), (1,-2) texels. Projected on x, y they
// sit at 0, +-s, +-2s (evenly spaced: a straight edge gets 5 overlapping 1-texel ramps, not steps), and the second
// moment matches three's 17 taps (per axis 7.5 r^2 / 17 texel^2; a bilinear tap adds 1/12):
// s^2 = 0.2206 r^2 - 0.0417 -> s = 0.92 texels at the near map's radius 2.0, 0.63 at the far map's 1.4.
//
// 'pcss' (percentage-closer soft shadows, ENGINE_SHADOW_PCSS, same per-material selection): contact-hardening
// penumbrae, narrow where the caster touches the receiver and widening with the distance between them. It reads the
// same render through both of its attachments: the RGBA-packed colour map that three keeps rendering (and binds as
// directionalShadowMap) gives raw blocker depths, the compare-mode depth texture of 'fast' gives bilinear PCF taps.
// (One DepthTexture cannot be sampled both raw and with compare in WebGL2 without sampler objects, and a depth
// texture read raw must be NEAREST; the packed map costs nothing extra: three renders it either way.) Per lookup:
//   1. the 'fast' kernel (5 compare taps) at three's PCF width: the contact penumbra, never sharper than 'pcf';
//   2. blocker search: 8 packed-depth taps on a Vogel disc of the widest filter radius, the same pattern for every
//      pixel (neighbours share texels: a per-pixel rotation here cost ~0.3 ms more at 3024x1964 through texture
//      cache misses). No blocker: the contact kernel's result (so features thinner than the search spacing still
//      shadow); all 8 blocked and the contact kernel dark: umbra, 0;
//   3. directional-light penumbra width = (receiver - mean blocker depth) x depth range x tan(angular size), in
//      texels; the filter radius (half of it, clamped to [contact width, max]) blends from the contact kernel to a
//      12-tap Vogel disc of compare taps, rotated per pixel by interleaved gradient noise (optionally re-seeded per
//      frame, for TAA), as it grows from 1.5x to 2.5x the contact width.
//   Every tap compares against the receiver plane (depth slope from screen-space derivatives, clamped), so wide
//   kernels do not self-shadow lit slopes. The derivatives run inside getShadow, which the cascade blend calls under
//   non-uniform control flow (rcW < 1, rcW > 0): undefined by the spec, fine on hardware that keeps quad helper lanes
//   running (all current GPUs); at worst a 2x2 quad on a blend-band edge, where that lookup's weight is ~0, gets a
//   wrong slope, and the clamp bounds it. The umbra early-out can step inside a wide penumbra (all 8 sparse taps
//   blocked + a dark contact kernel while the full filter would see some light): a slightly harder inner edge. Cost: 13 fetches where unshadowed or deep in shadow, 25 in a wide
//   penumbra (pcf: 17 unpacked, fast: 5), plus N more texture units per material (both maps are bound).
// Per-cascade constants come from setPcssShadowParams (Atmosphere fills them from the boxes and the light size). The
// lookup is selected by cascade index: `directionalShadowMap[ n ]` becomes `enginePcssIndex[ n ]` (an int), which
// resolves to a getShadow( int, ... ) overload that picks both maps and the constants of cascade n.
import * as THREE from 'three';

export const SHADOW_KERNELS = ['pcf', 'fast', 'pcss'];
export const FAST_SHADOW_DEFINE = 'ENGINE_SHADOW_FAST';
export const PCSS_SHADOW_DEFINE = 'ENGINE_SHADOW_PCSS';
const UNIFORM = 'engineDirShadowDepth';
const PCSS_UNIFORM = 'engineDirShadowPcss';

// Value of the sampler array on every material: built-in lit materials clone their uniforms from THREE.ShaderLib
// when a program is compiled, so the patch adds the uniform to the lit templates. One module-level array-like (not
// an Array: UniformsUtils.clone slices Arrays but keeps plain objects by reference) means every compiled material
// reads the depth textures of the latest setFastShadowMaps(), including after an Atmosphere rebuild. Entries must
// cover NUM_DIR_LIGHT_SHADOWS: a sampler2DShadow left on its default unit 0 next to a sampler2D is a GL error.
const MAPS = { length: 0 };
// 'pcss' constants per directional shadow (same sharing rule; flatten() reads exactly the shader's array size, so four
// entries always exist): x penumbra width in texels per unit of normalised depth between receiver and blocker,
// y widest filter radius = blocker search radius (texels), z receiver-plane slope limit (normalised depth per texel),
// w noise seed (frame index for TAA, 0 = a fixed pattern).
const PCSS = { length: 4 };
for (let i = 0; i < 4; i++) PCSS[i] = new THREE.Vector4(0, 0, 0, 0);

const GLSL = /* glsl */`
#if defined( USE_SHADOWMAP ) && defined( ${FAST_SHADOW_DEFINE} ) && defined( SHADOWMAP_TYPE_PCF ) && NUM_DIR_LIGHT_SHADOWS > 0
	uniform sampler2DShadow ${UNIFORM}[ NUM_DIR_LIGHT_SHADOWS ];
	float getShadow( sampler2DShadow shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord ) {
		float shadow = 1.0;
		shadowCoord.xyz /= shadowCoord.w;
		shadowCoord.z += shadowBias;
		bool inFrustum = shadowCoord.x >= 0.0 && shadowCoord.x <= 1.0 && shadowCoord.y >= 0.0 && shadowCoord.y <= 1.0;
		if ( inFrustum && shadowCoord.z <= 1.0 ) {
			vec2 s = sqrt( max( 0.2206 * shadowRadius * shadowRadius - 0.0417, 0.0 ) ) / shadowMapSize;
			vec3 p = shadowCoord.xyz;
			shadow = 0.2 * (
				texture( shadowMap, p ) +
				texture( shadowMap, vec3( p.xy + s * vec2( 2.0, 1.0 ), p.z ) ) +
				texture( shadowMap, vec3( p.xy + s * vec2( - 1.0, 2.0 ), p.z ) ) +
				texture( shadowMap, vec3( p.xy - s * vec2( 2.0, 1.0 ), p.z ) ) +
				texture( shadowMap, vec3( p.xy - s * vec2( - 1.0, 2.0 ), p.z ) ) );
		}
		return mix( 1.0, shadow, shadowIntensity );
	}
	#define directionalShadowMap ${UNIFORM}
#endif
`;

// unit-disc Vogel (golden-angle) points: even coverage for any count, rotated per pixel
const vogel = (n) => Array.from({ length: n }, (_, i) => {
  const r = Math.sqrt((i + 0.5) / n), t = i * 2.39996323;
  return `vec2( ${(r * Math.cos(t)).toFixed(4)}, ${(r * Math.sin(t)).toFixed(4)} )`;
}).join(', ');

const pcssPick = (n) => `enginePcssShadow( directionalShadowMap[ ${n} ], ${UNIFORM}[ ${n} ], ${PCSS_UNIFORM}[ ${n} ], shadowMapSize, shadowIntensity, shadowBias, shadowRadius, shadowCoord )`;

const PCSS_GLSL = /* glsl */`
#if defined( USE_SHADOWMAP ) && defined( ${PCSS_SHADOW_DEFINE} ) && defined( SHADOWMAP_TYPE_PCF ) && NUM_DIR_LIGHT_SHADOWS > 0 && NUM_DIR_LIGHT_SHADOWS <= 4
	uniform sampler2DShadow ${UNIFORM}[ NUM_DIR_LIGHT_SHADOWS ];
	uniform vec4 ${PCSS_UNIFORM}[ NUM_DIR_LIGHT_SHADOWS ];
	const vec2 enginePcssSearch[ 8 ] = vec2[ 8 ]( ${vogel(8)} );
	const vec2 enginePcssDisc[ 12 ] = vec2[ 12 ]( ${vogel(12)} );
	const ivec4 enginePcssIndex = ivec4( 0, 1, 2, 3 );
	float enginePcssShadow( sampler2D packedMap, sampler2DShadow depthMap, vec4 pcss, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord ) {
		shadowCoord.xyz /= shadowCoord.w;
		shadowCoord.z += shadowBias;
		// receiver plane: depth change per texel along the map's x / y (derivatives before any branch of this function;
		// callers may be in non-uniform flow, see the header)
		vec3 ddx = dFdx( shadowCoord.xyz ), ddy = dFdy( shadowCoord.xyz );
		float det = ddx.x * ddy.y - ddx.y * ddy.x;
		vec2 slope = abs( det ) > 1e-20 ? vec2( ddy.y * ddx.z - ddx.y * ddy.z, ddx.x * ddy.z - ddy.x * ddx.z ) / ( det * shadowMapSize ) : vec2( 0.0 );
		slope = clamp( slope, - pcss.z, pcss.z );
		if ( shadowCoord.x < 0.0 || shadowCoord.x > 1.0 || shadowCoord.y < 0.0 || shadowCoord.y > 1.0 || shadowCoord.z > 1.0 ) return 1.0;
		vec2 texel = 1.0 / shadowMapSize;
		vec3 p = shadowCoord.xyz;
		// 1. contact kernel: the 'fast' kernel (three's PCF width at shadowRadius)
		float s = sqrt( max( 0.2206 * shadowRadius * shadowRadius - 0.0417, 0.0 ) );
		vec2 g1 = s * vec2( 2.0, 1.0 ), g2 = s * vec2( - 1.0, 2.0 );
		float lit = 0.2 * (
			texture( depthMap, p ) +
			texture( depthMap, vec3( p.xy + g1 * texel, p.z + dot( slope, g1 ) ) ) +
			texture( depthMap, vec3( p.xy + g2 * texel, p.z + dot( slope, g2 ) ) ) +
			texture( depthMap, vec3( p.xy - g1 * texel, p.z - dot( slope, g1 ) ) ) +
			texture( depthMap, vec3( p.xy - g2 * texel, p.z - dot( slope, g2 ) ) ) );
		// 2. blocker search: raw depths from the packed map over the widest filter radius (one pattern for every pixel:
		// neighbours read the same texels, which keeps the texture cache warm; only the filter below is rotated)
		float nb = 0.0, zb = 0.0;
		for ( int k = 0; k < 8; k ++ ) {
			vec2 o = enginePcssSearch[ k ] * pcss.y;
			float d = unpackRGBAToDepth( texture( packedMap, p.xy + o * texel ) );
			#ifdef USE_REVERSED_DEPTH_BUFFER
			if ( d > p.z + dot( slope, o ) ) { nb += 1.0; zb += d; }
			#else
			if ( d < p.z + dot( slope, o ) ) { nb += 1.0; zb += d; }
			#endif
		}
		// no blocker: contact kernel; umbra (may harden the inner edge of a wide penumbra a little, see the header)
		if ( nb == 0.0 || ( nb == 8.0 && lit == 0.0 ) ) return mix( 1.0, lit, shadowIntensity );
		// 3. penumbra: filter radius from the mean blocker distance, at least the contact kernel's, at most pcss.y
		float rMin = 2.0 * sqrt( max( 0.441 * shadowRadius * shadowRadius - 0.0833, 0.0 ) );
		float rad = max( min( 0.5 * abs( p.z - zb / nb ) * pcss.x, pcss.y ), rMin );
		float wide = smoothstep( 1.5 * rMin, 2.5 * rMin, rad );
		if ( wide > 0.0 ) {
			float a = 6.2831853 * fract( 52.9829189 * fract( dot( gl_FragCoord.xy + 5.588238 * pcss.w, vec2( 0.06711056, 0.00583715 ) ) ) );
			mat2 rot = mat2( cos( a ), sin( a ), - sin( a ), cos( a ) );
			float sw = 0.0;
			for ( int k = 0; k < 12; k ++ ) {
				vec2 o = rot * enginePcssDisc[ k ] * rad;
				sw += texture( depthMap, vec3( p.xy + o * texel, p.z + dot( slope, o ) ) );
			}
			lit = mix( lit, sw * ( 1.0 / 12.0 ), wide );
		}
		return mix( 1.0, lit, shadowIntensity );
	}
	float getShadow( int cascade, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord ) {
		#if NUM_DIR_LIGHT_SHADOWS > 3
		if ( cascade == 3 ) return ${pcssPick(3)};
		#endif
		#if NUM_DIR_LIGHT_SHADOWS > 2
		if ( cascade == 2 ) return ${pcssPick(2)};
		#endif
		#if NUM_DIR_LIGHT_SHADOWS > 1
		if ( cascade == 1 ) return ${pcssPick(1)};
		#endif
		return ${pcssPick(0)};
	}
	#define directionalShadowMap enginePcssIndex
#endif
`;

/** The GLSL appended to shadowmap_pars_fragment by installPcssShadowChunk (inert without ENGINE_SHADOW_PCSS). */
export const PCSS_SHADOW_GLSL = PCSS_GLSL;

let installed = null;
let pcssInstalled = null;

// the r180 shadow code both kernels rely on: RGBA-packed directional maps looked up through getShadow( map[ n ], ... )
function chunksOk(what, fallback) {
  const C = THREE.ShaderChunk;
  const need = [
    [C.shadowmap_pars_fragment, 'uniform sampler2D directionalShadowMap[ NUM_DIR_LIGHT_SHADOWS ];'],
    [C.shadowmap_pars_fragment, 'float getShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord )'],
    [C.shadowmap_pars_fragment, 'unpackRGBAToDepth'], // still RGBA-packed maps: the depth attachment is free to replace
    [C.lights_fragment_begin, 'getShadow( directionalShadowMap['],
  ];
  if (need.every(([chunk, s]) => typeof chunk === 'string' && chunk.includes(s))) return true;
  console.warn(`[render] ${what} shadow kernel: three.js shadow chunk changed, keeping '${fallback}'`);
  return false;
}

// lit templates get the shared uniform values (by reference: see MAPS)
function addTemplateUniform(name, value) {
  for (const k in THREE.ShaderLib) {
    const u = THREE.ShaderLib[k].uniforms;
    if (u && u.directionalShadowMap && !u[name]) u[name] = { value };
  }
}

/** Install the chunk + template patch (idempotent). false (with a warning) when three's shadow code changed. */
export function installFastShadowChunk() {
  if (installed !== null) return installed;
  if (!chunksOk('fast', 'pcf')) { installed = false; return false; }
  THREE.ShaderChunk.shadowmap_pars_fragment += GLSL;
  addTemplateUniform(UNIFORM, MAPS);
  installed = true;
  return true;
}

/**
 * Install the 'pcss' chunk + template patch (idempotent; independent of the 'fast' one, which it does not need).
 * false (with a warning) when three's shadow code changed.
 */
export function installPcssShadowChunk() {
  if (pcssInstalled !== null) return pcssInstalled;
  if (!chunksOk('pcss', 'pcf')) { pcssInstalled = false; return false; }
  THREE.ShaderChunk.shadowmap_pars_fragment += PCSS_GLSL;
  addTemplateUniform(UNIFORM, MAPS);
  addTemplateUniform(PCSS_UNIFORM, PCSS);
  pcssInstalled = true;
  return true;
}

/**
 * Depth attachment for a shadow map that the fast kernel can sample: compare mode (LEQUAL, GEQUAL with a reversed
 * depth buffer, matching texture2DCompare) + linear filtering = 2x2 bilinear PCF per tap. 24-bit (UnsignedIntType):
 * 6e-8 of the depth range is far below the 3e-5 bias, and a float type trips three's float-linear filtering check.
 */
export function createShadowDepthTexture(reversedDepth = false) {
  const d = new THREE.DepthTexture(1, 1); // sized to its render target when attached
  d.compareFunction = reversedDepth ? THREE.GreaterEqualCompare : THREE.LessEqualCompare;
  d.minFilter = d.magFilter = THREE.LinearFilter;
  return d;
}

/**
 * Make `light.shadow.map` a target like the one three creates for PCF (RGBA8 packed depth, nearest) whose depth
 * buffer is `depth`, so the stock and the fast kernel both read the same render. Disposes a previous map (its
 * renderbuffer would sit unused) and flags a re-render. mapSize must already fit maxTextureSize (Atmosphere clamps
 * it). Returns false when the map already has `depth`. Allocates only when it (re)creates.
 */
export function attachShadowDepth(light, depth) {
  const s = light.shadow;
  if (s.map && s.map.depthTexture === depth) return false;
  // a retired target still listing `depth` would dispose it along with itself later: unlink it first
  if (depth.renderTarget && depth.renderTarget !== s.map) depth.renderTarget.depthTexture = null;
  if (s.map) s.map.dispose(); // with its depth texture, if any: a DepthTexture object is re-attachable after that
  const rt = new THREE.WebGLRenderTarget(s.mapSize.x, s.mapSize.y, { minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter, depthTexture: depth });
  rt.texture.name = `${light.name}.shadowMap`;
  s.map = rt;
  s.needsUpdate = true; // lights with autoUpdate off (the far cascade) render their new map on the next frame
  return true;
}

/**
 * Point the fast kernel's sampler array at depth textures, in three's directional shadow order (shadow-casting
 * directional lights in scene order). `maps` is read, not kept (pass a reused array; count <= maps.length).
 */
export function setFastShadowMaps(maps, count = maps.length) {
  for (let i = 0; i < count; i++) MAPS[i] = maps[i];
  for (let i = count; i < MAPS.length; i++) MAPS[i] = undefined;
  MAPS.length = count;
}

/**
 * 'pcss' constants per directional shadow, in three's order (as setFastShadowMaps). `params` is read, not kept:
 * Vector4-likes (x, y, z, w as documented at PCSS; count <= 4). Allocation-free.
 */
export function setPcssShadowParams(params, count = params.length) {
  const n = Math.min(count, 4);
  for (let i = 0; i < n; i++) PCSS[i].copy(params[i]);
  for (let i = n; i < 4; i++) PCSS[i].set(0, 0, 0, 0);
}

/**
 * Lit materials whose uniforms carry the kernel's uniforms: built-ins (patched templates), or a lights: true
 * ShaderMaterial given them (uniforms.engineDirShadowDepth = THREE.ShaderLib.standard.uniforms.engineDirShadowDepth,
 * and for 'pcss' also engineDirShadowPcss, after install). Anything else never gets a define: its sampler2DShadow
 * would be left unbound.
 */
function takesKernel(m, pcss) {
  return !!(m.isMeshStandardMaterial || m.isMeshLambertMaterial || m.isMeshPhongMaterial || m.isMeshToonMaterial || m.isShadowMaterial
    || (m.isShaderMaterial && m.lights === true && m.uniforms && m.uniforms[UNIFORM] && (!pcss || m.uniforms[PCSS_UNIFORM])));
}

/**
 * Select a material's kernel through its ENGINE_SHADOW_FAST / ENGINE_SHADOW_PCSS define (at most one; needsUpdate
 * when it changes; three keeps every program, so later switches do not recompile). 'fast' needs
 * installFastShadowChunk() and setFastShadowMaps() first, 'pcss' installPcssShadowChunk(), setFastShadowMaps() and
 * setPcssShadowParams(). A kernel whose chunk is not installed (or that the material cannot take) selects 'pcf'.
 * Returns true when the material changed.
 */
export function setMaterialShadowKernel(m, kernel) {
  if (!m || !takesKernel(m, false)) return false;
  const pcss = kernel === 'pcss' && pcssInstalled === true && takesKernel(m, true);
  const fast = kernel === 'fast' && installed === true;
  const d = m.defines;
  const hasFast = !!(d && FAST_SHADOW_DEFINE in d), hasPcss = !!(d && PCSS_SHADOW_DEFINE in d);
  if (fast === hasFast && pcss === hasPcss) return false;
  if (hasFast && !fast) delete d[FAST_SHADOW_DEFINE]; // an emptied defines object keys and compiles like none
  if (hasPcss && !pcss) delete d[PCSS_SHADOW_DEFINE];
  if (fast && !hasFast) (m.defines || (m.defines = {}))[FAST_SHADOW_DEFINE] = '';
  if (pcss && !hasPcss) (m.defines || (m.defines = {}))[PCSS_SHADOW_DEFINE] = '';
  m.needsUpdate = true;
  return true;
}

const _apply = { kernel: 'pcf', n: 0 };
function applyTo(o) {
  const m = o.material;
  if (!m) return;
  if (Array.isArray(m)) { for (let i = 0; i < m.length; i++) if (setMaterialShadowKernel(m[i], _apply.kernel)) _apply.n++; }
  else if (setMaterialShadowKernel(m, _apply.kernel)) _apply.n++;
}

/** setMaterialShadowKernel on every material under root (custom depth materials untouched). Returns the count changed. */
export function applyShadowKernel(root, kernel) {
  _apply.kernel = kernel; _apply.n = 0;
  if (root) root.traverse(applyTo);
  return _apply.n;
}
