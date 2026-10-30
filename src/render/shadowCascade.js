// Cascaded sun shadows (2-4 cascades) without touching any material: a global ShaderChunk patch for the directional
// light loop. When the scene's shadow-casting directional lights 1 .. N-1 are black (intensity 0) and point the same
// way as light 0, light 0's shadow lookup walks the cascades: the near map (light 0) inside the near box, then each
// middle map (lights 1 .. N-2) outside the previous box and inside its own, then the far map (light N-1), with a blend
// band at every box edge and a fade at the far box edge. The black lights add no light. Any other light setup
// compiles to the stock three.js code path.
// Lookups keep the stock form getShadow( directionalShadowMap[ n ], ... ): the opt-in kernels (shadowKernel.js)
// swap the filter by overloading getShadow and renaming directionalShadowMap, so every kernel blends the same way.
//
// installCascadeShadowChunk(maxCascades = 2) installs the text for up to maxCascades cascades. The 2-cascade text is
// the original (0.6.0) one, byte for byte; asking for 3 or 4 later swaps that block (only that block: other patches
// of the chunk stay) for the general text, which handles 2, 3 and 4 cascades by NUM_DIR_LIGHT_SHADOWS and the
// black-light test. Chunk text is not in three's program cache key: programs compiled before an upgrade keep the old
// text, which is equivalent for the setups it handles (an upgrade happens before any light of the new setup exists).
import * as THREE from 'three';

/** Most cascades the patch handles (lights 0 .. 3). */
export const MAX_CASCADES = 4;

const ORIG = 'directLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( directionalShadowMap[ i ], directionalLightShadow.shadowMapSize, directionalLightShadow.shadowIntensity, directionalLightShadow.shadowBias, directionalLightShadow.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;';

let installed = null;  // null: not tried, false: three's chunk changed, true: installed
let installedMax = 0;  // cascades the installed text handles
let installedText = ''; // the replacement block now in lights_fragment_begin

// 0.6.0 two-cascade replacement (kept verbatim: `cascades: 2` compiles byte-identical shaders)
function twoCascadeText() {
  const near = ORIG.replace(/\[ i \]/g, '[ 0 ]');
  const far = ORIG.replace(/\[ i \]/g, '[ 1 ]').replace(/directionalLightShadow\./g, 'rcFarS.');
  return `
		#if ( UNROLLED_LOOP_INDEX == 0 ) && ( NUM_DIR_LIGHT_SHADOWS > 1 )
		if ( dot( directionalLights[ 1 ].color, vec3( 1.0 ) ) <= 0.0 && dot( directionalLights[ 1 ].direction, directionalLight.direction ) > 0.9999 ) {
			vec3 rcN = vDirectionalShadowCoord[ 0 ].xyz / vDirectionalShadowCoord[ 0 ].w;
			vec3 rcF = vDirectionalShadowCoord[ 1 ].xyz / vDirectionalShadowCoord[ 1 ].w;
			vec2 rcE = abs( rcN.xy - 0.5 );
			float rcW = ( rcN.z > 1.0 ) ? 1.0 : smoothstep( 0.39, 0.47, max( rcE.x, rcE.y ) );
			float rcS = 1.0;
			if ( receiveShadow ) {
				DirectionalLightShadow rcFarS = directionalLightShadows[ 1 ];
				float rcSN = 1.0, rcSF = 1.0;
				if ( rcW < 1.0 ) rcSN = getShadow( directionalShadowMap[ 0 ], directionalLightShadow.shadowMapSize, directionalLightShadow.shadowIntensity, directionalLightShadow.shadowBias, directionalLightShadow.shadowRadius, vDirectionalShadowCoord[ 0 ] );
				if ( rcW > 0.0 ) {
					rcSF = getShadow( directionalShadowMap[ 1 ], rcFarS.shadowMapSize, rcFarS.shadowIntensity, rcFarS.shadowBias, rcFarS.shadowRadius, vDirectionalShadowCoord[ 1 ] );
					vec2 rcEF = abs( rcF.xy - 0.5 );
					rcSF = mix( rcSF, 1.0, smoothstep( 0.43, 0.495, max( rcEF.x, rcEF.y ) ) );
				}
				rcS = mix( rcSN, rcSF, rcW );
			}
			directLight.color *= rcS;
		} else {
			${near}
		}
		#elif ( UNROLLED_LOOP_INDEX == 1 )
		if ( dot( directLight.color, vec3( 1.0 ) ) > 0.0 ) {
			DirectionalLightShadow rcFarS = directionalLightShadows[ 1 ];
			${far}
		}
		#else
		${ORIG}
		#endif`;
}

// light k is a cascade of light 0: black and parallel to it (the sanity check keeps unrelated black lights out)
const isCascade = (k) => `dot( directionalLights[ ${k} ].color, vec3( 1.0 ) ) <= 0.0 && dot( directionalLights[ ${k} ].direction, directionalLight.direction ) > 0.9999`;

function lookup(k) {
  const s = k === 0 ? 'directionalLightShadow' : `rcL${k}`;
  return `getShadow( directionalShadowMap[ ${k} ], ${s}.shadowMapSize, ${s}.shadowIntensity, ${s}.shadowBias, ${s}.shadowRadius, vDirectionalShadowCoord[ ${k} ] )`;
}

// Level k >= 1 of the cascade walk, inside the previous level's `rcW > 0` branch: looks up map k where this level is
// not fully blended out, then either fades it at the box edge (map k is the far cascade: rcN == k + 1) or blends
// on to level k + 1 over this box's band. Stores the result in rcR{k-1}. Each map has a single lookup site.
function cascadeLevel(k, max, d, L) {
  L(`DirectionalLightShadow rcL${k} = directionalLightShadows[ ${k} ];`, d);
  L(`vec3 rc${k} = vDirectionalShadowCoord[ ${k} ].xyz / vDirectionalShadowCoord[ ${k} ].w;`, d);
  L(`vec2 rcE${k} = abs( rc${k}.xy - 0.5 );`, d);
  L(`bool rcF${k} = rcN == ${k + 1};`, d);
  L(`float rcW${k} = rcF${k} ? 0.0 : ( rc${k}.z > 1.0 ) ? 1.0 : smoothstep( 0.39, 0.47, max( rcE${k}.x, rcE${k}.y ) );`, d);
  L(`float rcS${k} = 1.0, rcR${k} = 1.0;`, d);
  L(`if ( rcW${k} < 1.0 ) rcS${k} = ${lookup(k)};`, d);
  L(`if ( rcF${k} ) rcS${k} = mix( rcS${k}, 1.0, smoothstep( 0.43, 0.495, max( rcE${k}.x, rcE${k}.y ) ) );`, d);
  if (k + 1 < max) {
    L(`#if NUM_DIR_LIGHT_SHADOWS > ${k + 1}`, d);
    L(`else if ( rcW${k} > 0.0 ) {`, d);
    cascadeLevel(k + 1, max, d + 1, L);
    L('}', d);
    L('#endif', d);
  }
  L(`rcR${k - 1} = mix( rcS${k}, rcR${k}, rcW${k} );`, d);
}

// General text for up to max cascades. Index 0 counts its cascades (rcN: light 0 plus the black, parallel shadow
// lights right after it, as far as NUM_DIR_LIGHT_SHADOWS allows), walks them (near, middle ..., far: the same bands
// as the 2-cascade text, 0.39-0.47 of the box half size between cascades and a fade to unshadowed over 0.43-0.495 at
// the far box edge) or does the stock lookup when there is none. Other indices skip the lookup of a black light (it
// adds nothing; a cascade's map was read by index 0) and keep the stock one otherwise.
function generalText(max) {
  const near = ORIG.replace(/\[ i \]/g, '[ 0 ]');
  const out = [''];
  const L = (s, d) => out.push('\t\t' + '\t'.repeat(d) + s);
  L('#if ( UNROLLED_LOOP_INDEX == 0 ) && ( NUM_DIR_LIGHT_SHADOWS > 1 )', 0);
  L('{', 0);
  L('int rcN = 1;', 1);
  for (let k = 1; k < max; k++) {
    if (k > 1) L(`#if NUM_DIR_LIGHT_SHADOWS > ${k}`, k);
    L(`if ( ${isCascade(k)} ) {`, k);
    L(`rcN = ${k + 1};`, k + 1);
  }
  for (let k = max - 1; k >= 1; k--) {
    L('}', k);
    if (k > 1) L('#endif', k);
  }
  L('if ( rcN > 1 ) {', 1);
  L('float rcS = 1.0;', 2);
  L('if ( receiveShadow ) {', 2);
  L('vec3 rc0 = vDirectionalShadowCoord[ 0 ].xyz / vDirectionalShadowCoord[ 0 ].w;', 3);
  L('vec2 rcE0 = abs( rc0.xy - 0.5 );', 3);
  L('float rcW0 = ( rc0.z > 1.0 ) ? 1.0 : smoothstep( 0.39, 0.47, max( rcE0.x, rcE0.y ) );', 3);
  L('float rcS0 = 1.0, rcR0 = 1.0;', 3);
  L(`if ( rcW0 < 1.0 ) rcS0 = ${lookup(0)};`, 3);
  L('if ( rcW0 > 0.0 ) {', 3);
  cascadeLevel(1, max, 4, L);
  L('}', 3);
  L('rcS = mix( rcS0, rcR0, rcW0 );', 3);
  L('}', 2);
  L('directLight.color *= rcS;', 2);
  L('} else {', 1);
  L(near, 2);
  L('}', 1);
  L('}', 0);
  L('#elif ( UNROLLED_LOOP_INDEX > 0 )', 0);
  L('if ( dot( directLight.color, vec3( 1.0 ) ) > 0.0 ) {', 0);
  L(ORIG, 1);
  L('}', 0);
  L('#else', 0);
  L(ORIG, 0);
  L('#endif', 0);
  return out.join('\n');
}

/**
 * The block that replaces three's directional shadow line in lights_fragment_begin for up to `maxCascades` (2..4)
 * cascades. 2 returns the 0.6.0 text verbatim.
 */
export function cascadeShadowChunkText(maxCascades = 2) {
  const max = Math.max(2, Math.min(MAX_CASCADES, Math.round(maxCascades) || 2));
  return max === 2 ? twoCascadeText() : generalText(max);
}

/** Cascades the installed patch handles (0 before install or when three's chunk changed). */
export function installedCascades() { return installed === true ? installedMax : 0; }

/**
 * Install (or upgrade) the global chunk patch for up to `maxCascades` (2..4) cascades. Idempotent; never downgrades.
 * Returns true when that many cascades are available, false (with a warning) when three's chunk changed.
 */
export function installCascadeShadowChunk(maxCascades = 2) {
  const want = Math.max(2, Math.min(MAX_CASCADES, Math.round(maxCascades) || 2));
  if (installed === false) return false;
  if (installed === true && want <= installedMax) return true;
  const C = THREE.ShaderChunk;
  const text = cascadeShadowChunkText(want);
  if (installed === null) {
    if (!C.lights_fragment_begin.includes(ORIG)) {
      console.warn('[render] shadow cascade patch: three.js chunk changed, using a single cascade');
      installed = false;
      return false;
    }
    C.lights_fragment_begin = C.lights_fragment_begin.replace(ORIG, text);
  } else {
    if (!C.lights_fragment_begin.includes(installedText)) {
      console.warn(`[render] shadow cascade patch: lights_fragment_begin was replaced, keeping ${installedMax} cascades`);
      return false;
    }
    C.lights_fragment_begin = C.lights_fragment_begin.replace(installedText, () => text);
  }
  installed = true;
  installedMax = want;
  installedText = text;
  return true;
}
