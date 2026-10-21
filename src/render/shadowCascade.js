// Two-cascade sun shadows without touching any material: a global ShaderChunk patch for the directional
// light loop. When the scene has two shadow-casting directional lights where light 1 is black (intensity 0)
// and points the same way as light 0, light 0's shadow lookup picks the near map inside the near box and
// the far map (light 1) outside it, with a blend band and a fade at the far box edge. Light 1 adds no light.
// Any other light setup compiles to the stock three.js code path.
import * as THREE from 'three';

let installed = null;

export function installCascadeShadowChunk() {
  if (installed !== null) return installed;
  const C = THREE.ShaderChunk;
  const orig = 'directLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( directionalShadowMap[ i ], directionalLightShadow.shadowMapSize, directionalLightShadow.shadowIntensity, directionalLightShadow.shadowBias, directionalLightShadow.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;';
  if (!C.lights_fragment_begin.includes(orig)) {
    console.warn('[render] shadow cascade patch: three.js chunk changed, using a single cascade');
    installed = false;
    return false;
  }
  const near = orig.replace(/\[ i \]/g, '[ 0 ]');
  const far = orig.replace(/\[ i \]/g, '[ 1 ]').replace(/directionalLightShadow\./g, 'rcFarS.');
  const repl = `
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
		${orig}
		#endif`;
  C.lights_fragment_begin = C.lights_fragment_begin.replace(orig, repl);
  installed = true;
  return true;
}
