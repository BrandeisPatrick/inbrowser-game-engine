// Opt-in cheaper directional shadow kernel. 'pcf' (default) is three's PCFShadowMap: r180 packs shadow depth into
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
// applyShadowKernel; Atmosphere.setShadowKernel wires all of it for its sun + far cascade.
//
// Kernel: centre + 4 taps on a rotated grid at s * (2,1), (-1,2), (-2,-1), (1,-2) texels. Projected on x, y they
// sit at 0, +-s, +-2s (evenly spaced: a straight edge gets 5 overlapping 1-texel ramps, not steps), and the second
// moment matches three's 17 taps (per axis 7.5 r^2 / 17 texel^2; a bilinear tap adds 1/12):
// s^2 = 0.2206 r^2 - 0.0417 -> s = 0.92 texels at the near map's radius 2.0, 0.63 at the far map's 1.4.
import * as THREE from 'three';

export const SHADOW_KERNELS = ['pcf', 'fast'];
export const FAST_SHADOW_DEFINE = 'ENGINE_SHADOW_FAST';
const UNIFORM = 'engineDirShadowDepth';

// Value of the sampler array on every material: built-in lit materials clone their uniforms from THREE.ShaderLib
// when a program is compiled, so the patch adds the uniform to the lit templates. One module-level array-like (not
// an Array: UniformsUtils.clone slices Arrays but keeps plain objects by reference) means every compiled material
// reads the depth textures of the latest setFastShadowMaps(), including after an Atmosphere rebuild. Entries must
// cover NUM_DIR_LIGHT_SHADOWS: a sampler2DShadow left on its default unit 0 next to a sampler2D is a GL error.
const MAPS = { length: 0 };

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

let installed = null;

/** Install the chunk + template patch (idempotent). false (with a warning) when three's shadow code changed. */
export function installFastShadowChunk() {
  if (installed !== null) return installed;
  const C = THREE.ShaderChunk;
  const need = [
    [C.shadowmap_pars_fragment, 'uniform sampler2D directionalShadowMap[ NUM_DIR_LIGHT_SHADOWS ];'],
    [C.shadowmap_pars_fragment, 'float getShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord )'],
    [C.shadowmap_pars_fragment, 'unpackRGBAToDepth'], // still RGBA-packed maps: the depth attachment is free to replace
    [C.lights_fragment_begin, 'getShadow( directionalShadowMap['],
  ];
  if (!need.every(([chunk, s]) => typeof chunk === 'string' && chunk.includes(s))) {
    console.warn("[render] fast shadow kernel: three.js shadow chunk changed, keeping 'pcf'");
    installed = false;
    return false;
  }
  C.shadowmap_pars_fragment += GLSL;
  for (const k in THREE.ShaderLib) {
    const u = THREE.ShaderLib[k].uniforms;
    if (u && u.directionalShadowMap && !u[UNIFORM]) u[UNIFORM] = { value: MAPS };
  }
  installed = true;
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
 * Lit materials whose uniforms carry the sampler array: built-ins (patched templates), or a lights: true ShaderMaterial
 * given it (uniforms.engineDirShadowDepth = THREE.ShaderLib.standard.uniforms.engineDirShadowDepth, after install).
 * Anything else never gets the define: its sampler2DShadow would be left unbound.
 */
function takesKernel(m) {
  return !!(m.isMeshStandardMaterial || m.isMeshLambertMaterial || m.isMeshPhongMaterial || m.isMeshToonMaterial || m.isShadowMaterial
    || (m.isShaderMaterial && m.lights === true && m.uniforms && m.uniforms[UNIFORM]));
}

/**
 * Select a material's kernel through its ENGINE_SHADOW_FAST define (needsUpdate when it changes; three keeps both
 * programs, so later switches do not recompile). 'fast' needs installFastShadowChunk() and setFastShadowMaps() first.
 * Returns true when the material changed.
 */
export function setMaterialShadowKernel(m, kernel) {
  if (!m || !takesKernel(m)) return false;
  const fast = kernel === 'fast' && installed === true;
  if (fast === !!(m.defines && FAST_SHADOW_DEFINE in m.defines)) return false;
  if (fast) (m.defines || (m.defines = {}))[FAST_SHADOW_DEFINE] = '';
  else delete m.defines[FAST_SHADOW_DEFINE]; // an emptied defines object keys and compiles like none
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
