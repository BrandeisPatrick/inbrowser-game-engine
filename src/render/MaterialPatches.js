// Reusable onBeforeCompile patches for built-in three.js materials (MeshStandardMaterial, MeshLambertMaterial,
// MeshDepthMaterial, ...), composable on one material:
//   windSway    world-space sway of foliage / flags / cables: weight from object height, an attribute or GLSL
//   ditherFade  screen-space dithered distance fade-out ('out') or fade-in ('in'); an 'out' and an 'in'
//               material with the same start/end cross-fade with complementary dither (near model <-> impostor),
//               so LOD transitions never pop and never double-blend
//   time / view uniforms shared by any number of materials (createTimeUniforms / updateTimeUniforms)
//
//   const U = createTimeUniforms();                       // { uTime, uViewPos } shared objects
//   const mat = applyPatches(new THREE.MeshStandardMaterial({ color }), [
//     windSway(U, { strength: 0.3, weight: 'height', height: 6 }),
//     ditherFade(U, { start: 150, end: 185, mode: 'out' }),
//   ]);
//   mesh.customDepthMaterial = makeDepthMaterial(mat);   // shadows sway / fade the same way
//   per frame: updateTimeUniforms(U, dt, camera.position)  (allocation-free)
//
// Displacements are applied after the instance matrix and before the model matrix (InstancedMesh-friendly;
// world space for meshes at the scene root). `mpInstancePos` (world position of the instance / object origin)
// is available to every patch for per-instance phase and distance.
import * as THREE from 'three';

/** Interleaved gradient noise (stable per pixel), 0..1. Used for dithered fades. */
export const GLSL_IGN = /* glsl */'float mpIGN( vec2 p ) { return fract( 52.9829189 * fract( dot( p, vec2( 0.06711056, 0.00583715 ) ) ) ); }\n';
/** Cheap 2D hash, 0..1 (per-instance phase / jitter). */
export const GLSL_HASH = /* glsl */'float mpHash( vec2 p ) { return fract( sin( dot( p, vec2( 12.9898, 78.233 ) ) ) * 43758.5453 ); }\n';

/** Shared uniform objects: uTime (s) and uViewPos (camera world position). */
export function createTimeUniforms() {
  return { uTime: { value: 0 }, uViewPos: { value: new THREE.Vector3(0, -1e6, 0) } };
}
/** Advance uTime by dt (or set it with `time`), copy the camera position. Allocation-free. */
export function updateTimeUniforms(U, dt, cameraPos, time) {
  U.uTime.value = Number.isFinite(time) ? time : U.uTime.value + (dt || 0);
  if (cameraPos && U.uViewPos) U.uViewPos.value.copy(cameraPos);
}

/**
 * Wind sway.
 * @param {object} U  shared uniforms with uTime
 * @param {object} [o]
 * @param {number} [o.strength=1]  displacement scale (m at weight 1)
 * @param {number} [o.frequency=1] time scale
 * @param {number[]} [o.direction=[0.8, 0.6]] prevailing wind direction (xz)
 * @param {'height'|string} [o.weight='height']  'height': (position.y / height)^2 (object space; 0 at the base),
 *   'attr:name.c': an attribute component (e.g. 'attr:veg.x'), or any GLSL float expression
 * @param {number} [o.height=1]  object-space height that reaches weight 1 (weight 'height')
 * @param {number} [o.flutter=0] small high-frequency jitter amplitude (m)
 * Returns a patch; patch.uniforms.uWind.value = Vector4(dirX, dirZ, strength, frequency) can be animated.
 */
export function windSway(U, o = {}) {
  const dir = o.direction || [0.8, 0.6], l = Math.hypot(dir[0], dir[1]) || 1;
  const uWind = { value: new THREE.Vector4(dir[0] / l, dir[1] / l, o.strength ?? 1, o.frequency ?? 1) };
  const uWindB = { value: new THREE.Vector4(1 / (o.height ?? 1), o.flutter ?? 0, 0, 0) };
  const w = o.weight ?? 'height';
  let attrDecl = '', weightExpr;
  if (w === 'height') weightExpr = 'pow( clamp( position.y * uWindB.x, 0.0, 1.0 ), 2.0 )';
  else if (w.startsWith('attr:')) {
    const [name, comp] = w.slice(5).split('.');
    attrDecl = `attribute vec4 ${name};\n`;
    weightExpr = `${name}.${comp || 'x'}`;
  } else weightExpr = w;
  return {
    key: `wind(${w})`,
    uniforms: { uTime: U.uTime, uWind, uWindB },
    depth: true,
    vertexPars: `uniform float uTime;\nuniform vec4 uWind;\nuniform vec4 uWindB;\n${attrDecl}`,
    vertexBegin: /* glsl */`
	{
		float mpW = ${weightExpr};
		if ( mpW > 0.0 ) {
			float t = uTime * uWind.w, ph = mpHash( mpInstancePos.xz ) * 6.2831853;
			vec2 dirW = uWind.xy;
			float g = 0.55 + 0.45 * sin( t * 0.33 + mpInstancePos.x * 0.004 + mpInstancePos.z * 0.003 );
			float s1 = sin( t * 1.2 + ph ) * 0.6 + sin( t * 2.3 + ph * 1.7 ) * 0.25;
			float s2 = cos( t * 1.05 + ph * 2.1 ) * 0.35;
			mpOffset.xz += ( dirW * ( 0.35 + s1 ) + vec2( - dirW.y, dirW.x ) * s2 ) * g * mpW * uWind.z;
			mpOffset.y -= abs( s1 ) * mpW * g * 0.12 * uWind.z;
			if ( uWindB.y > 0.0 ) mpOffset += vec3( 0.6, 1.0, - 0.45 ) * sin( t * 6.3 + dot( position, vec3( 17.0, 31.0, 23.0 ) ) + ph ) * uWindB.y * mpW;
		}
	}
`,
  };
}

/**
 * Dithered distance fade (per instance, by the distance from uViewPos to the instance origin).
 * mode 'out': visible near, gone beyond `end`. mode 'in': invisible before `start`, solid beyond `end`.
 * Fully faded instances collapse to a point in the vertex shader (no fragment cost).
 * Returns a patch; patch.uniforms.uFade.value = Vector4(start, end, mode 0|1, -) can be changed live.
 */
export function ditherFade(U, o = {}) {
  const uFade = { value: new THREE.Vector4(o.start ?? 100, o.end ?? 120, o.mode === 'in' ? 1 : 0, 0) };
  return {
    key: 'dither',
    uniforms: { uViewPos: U.uViewPos, uFade },
    depth: true,
    vertexPars: 'uniform vec3 uViewPos;\nuniform vec4 uFade;\nvarying float vMpFade;\n',
    vertexBegin: /* glsl */`
	{
		float s = smoothstep( uFade.x, uFade.y, distance( uViewPos, mpInstancePos ) );
		vMpFade = uFade.z > 0.5 ? s : 1.0 - s;
		if ( vMpFade <= 0.0 ) { transformed = vec3( 0.0 ); mpCollapsed = true; }
	}
`,
    fragmentPars: 'uniform vec4 uFade;\nvarying float vMpFade;\n',
    fragmentDiscard: /* glsl */`
	if ( vMpFade < 0.999 ) {
		float n = mpIGN( gl_FragCoord.xy );
		if ( uFade.z > 0.5 ? n <= 1.0 - vMpFade : n > vMpFade ) discard;
	}
`,
  };
}

const PROJECT_ANCHOR = 'mvPosition = modelViewMatrix * mvPosition;';
const WORLD_ANCHOR = 'worldPosition = modelMatrix * worldPosition;';

function patchVertex(vs, patches) {
  const C = THREE.ShaderChunk;
  const pars = patches.map((p) => p.vertexPars || '').join('');
  const begin = patches.map((p) => p.vertexBegin || '').join('');
  vs = vs.replace('#include <common>', `#include <common>\n${GLSL_HASH}vec3 mpOffset;\nvec3 mpInstancePos;\nbool mpCollapsed;\n${pars}`);
  vs = vs.replace('#include <begin_vertex>', `#include <begin_vertex>
mpOffset = vec3( 0.0 );
mpCollapsed = false;
#ifdef USE_INSTANCING
	mpInstancePos = ( modelMatrix * vec4( instanceMatrix[ 3 ].xyz, 1.0 ) ).xyz;
#else
	mpInstancePos = modelMatrix[ 3 ].xyz;
#endif
${begin}`);
  if (!C.project_vertex.includes(PROJECT_ANCHOR)) console.warn('[MaterialPatches] project_vertex anchor missing');
  vs = vs.replace('#include <project_vertex>', C.project_vertex.replace(PROJECT_ANCHOR, `if ( ! mpCollapsed ) mvPosition.xyz += mpOffset;\n${PROJECT_ANCHOR}`));
  if (vs.includes('#include <worldpos_vertex>')) {
    vs = vs.replace('#include <worldpos_vertex>', C.worldpos_vertex.replace(WORLD_ANCHOR, `if ( ! mpCollapsed ) worldPosition.xyz += mpOffset;\n\t${WORLD_ANCHOR}`));
  }
  return vs;
}
function patchFragment(fs, patches) {
  const pars = patches.map((p) => p.fragmentPars || '').join('');
  const disc = patches.map((p) => p.fragmentDiscard || '').join('');
  if (!pars && !disc) return fs;
  fs = fs.replace('#include <common>', `#include <common>\n${GLSL_IGN}${pars}`);
  return fs.replace('#include <clipping_planes_fragment>', `#include <clipping_planes_fragment>\n${disc}`);
}

/**
 * Install patches on a material (chains an existing onBeforeCompile). Returns the material; the merged uniforms
 * are on material.userData.patchUniforms and the patch list on material.userData.patches.
 * @param {import('three').Material} material
 * @param {object[]} patches
 * @param {{ cacheKey?: string }} [o]  cacheKey: appended to customProgramCacheKey (defaults to the patch names)
 */
export function applyPatches(material, patches, { cacheKey } = {}) {
  const prev = material.onBeforeCompile;
  const uniforms = Object.assign({}, ...patches.map((p) => p.uniforms || {}));
  material.onBeforeCompile = (sh, r) => {
    if (prev && prev !== THREE.Material.prototype.onBeforeCompile) prev.call(material, sh, r);
    Object.assign(sh.uniforms, uniforms);
    sh.vertexShader = patchVertex(sh.vertexShader, patches);
    sh.fragmentShader = patchFragment(sh.fragmentShader, patches);
  };
  const key = cacheKey || `mp:${patches.map((p) => p.key).join('|')}`;
  const prevKey = material.customProgramCacheKey;
  material.customProgramCacheKey = () => `${prevKey.call(material)}|${key}`;
  material.userData.patchUniforms = uniforms;
  material.userData.patches = patches;
  material.needsUpdate = true;
  return material;
}

/**
 * Shadow-caster material carrying the same vertex patches (and dithered discards) as `material`:
 * assign to mesh.customDepthMaterial. Copies map / alphaTest / side so alpha-tested cut-outs cast correctly.
 */
export function makeDepthMaterial(material, extra = {}) {
  const patches = (material.userData.patches || []).filter((p) => p.depth !== false);
  const d = new THREE.MeshDepthMaterial({
    depthPacking: THREE.RGBADepthPacking,
    map: material.map || null, alphaMap: material.alphaMap || null,
    alphaTest: material.alphaTest || 0, side: material.side, ...extra,
  });
  return applyPatches(d, patches);
}
