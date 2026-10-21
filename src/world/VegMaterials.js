// Vegetation materials: MeshStandardMaterial patched via onBeforeCompile.
//  - world-space wind sway (crown-weighted via the `veg` attribute, zero on trunks) + leaf flutter
//  - dithered distance fade (near model fades out exactly where the impostor fades in: complementary dither)
//  - per-instance season tint (aTint.rgb, mix in aTint.a) + per-tree colour jitter from a position hash
//  - leaf cards keep their crown normals on back faces; wrap diffuse + back-lit translucency for leaves
//  - impostors: camera-facing billboards picking/blending baked views (8 azimuths x 3 elevations)
import * as THREE from 'three';
import { ShaderChunk } from 'three';

const COMMON_PARS = /* glsl */`
float vegIGN( vec2 p ) { return fract( 52.9829189 * fract( dot( p, vec2( 0.06711056, 0.00583715 ) ) ) ); }
float vegHash( vec2 p ) { return fract( sin( dot( p, vec2( 12.9898, 78.233 ) ) ) * 43758.5453 ); }
`;

// leaf colour: per-tree jitter + season tint (shared by near models and impostors)
const TINT_FN = /* glsl */`
vec3 vegTint( vec3 c, float jit, vec4 tint ) {
  c *= ( 0.86 + 0.28 * jit ) * mix( vec3( 1.05, 1.03, 0.86 ), vec3( 0.93, 1.0, 1.08 ), fract( jit * 7.7 ) );
  float l = dot( c, vec3( 0.299, 0.587, 0.114 ) );
  return mix( c, l * tint.rgb * 2.4, tint.a );
}
`;

const LIGHT_PATCH_FROM = 'reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );\n}';
const LIGHT_PATCH_TO = /* glsl */`reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
	#ifdef VEG_LIGHTING
	{
		float vnl = dot( geometryNormal, directLight.direction );
		float vWrap = saturate( ( vnl + 0.8 ) / 1.8 ) - saturate( vnl );
		float vBack = pow( saturate( dot( - geometryViewDir, directLight.direction ) ), 3.0 );
		reflectedLight.directDiffuse += directLight.color * BRDF_Lambert( material.diffuseColor ) * vLeaf * ( vWrap + vBack * 1.1 * vec3( 1.0, 1.15, 0.5 ) );
	}
	#endif
}`;

function patchLights(fs) {
  const chunk = ShaderChunk.lights_physical_pars_fragment;
  if (!chunk.includes(LIGHT_PATCH_FROM)) { console.warn('[trees] lighting patch anchor not found'); return fs; }
  return fs.replace('#include <lights_physical_pars_fragment>', chunk.replace(LIGHT_PATCH_FROM, LIGHT_PATCH_TO));
}

// ------------------------------------------------------------------ near models (InstancedMesh)
const VEG_VERT_PARS = /* glsl */`
uniform float uTime;
uniform vec3 uViewPos;
uniform vec4 uFade; // x fade start, y fade end, z wind strength, w per-instance card drop fraction
attribute vec4 veg;
#ifdef VEG_TINT
attribute vec4 aTint;
varying vec4 vTint;
#endif
varying float vLeaf;
varying float vFade;
varying float vJit;
vec3 vegWind;
` + COMMON_PARS;

const VEG_BEGIN = /* glsl */`
vec3 transformed = vec3( position );
vegWind = vec3( 0.0 );
{
#ifdef USE_INSTANCING
	vec3 iP = instanceMatrix[ 3 ].xyz;
#else
	vec3 iP = vec3( 0.0 );
#endif
	float ih = vegHash( iP.xz );
	vJit = ih;
	vLeaf = veg.y;
#ifdef VEG_TINT
	vTint = aTint;
#endif
	float dist = distance( uViewPos, iP );
	vFade = 1.0 - smoothstep( uFade.x, uFade.y, dist );
#ifdef VEG_DEPTH
	vFade *= 1.0 - smoothstep( 95.0, 125.0, dist ); // shadow casters only near the camera (shadow box is +-130 m)
#endif
	if ( vFade <= 0.0 || ( veg.y > 0.5 && fract( veg.z * 13.17 + ih * 7.31 ) < uFade.w ) ) transformed = vec3( 0.0, -20.0, 0.0 );
	float w = veg.x * uFade.z;
	if ( w > 0.0 ) {
		float t = uTime, ph = ih * 6.2831853;
		vec2 dirW = vec2( 0.8, 0.6 );
		float g = 0.55 + 0.45 * sin( t * 0.33 + iP.x * 0.004 + iP.z * 0.003 );
		float s1 = sin( t * 1.2 + ph ) * 0.6 + sin( t * 2.3 + ph * 1.7 ) * 0.25;
		float s2 = cos( t * 1.05 + ph * 2.1 ) * 0.35;
		vegWind.xz = ( dirW * ( 0.35 + s1 ) + vec2( - dirW.y, dirW.x ) * s2 ) * g * w;
		vegWind.y = - abs( s1 ) * w * g * 0.12;
		float fl = sin( t * 6.3 + veg.z * 61.0 + ph ) * veg.w * 0.05 * ( 0.4 + g );
		vegWind += vec3( 0.6, 1.0, - 0.45 ) * fl;
	}
}
`;

const VEG_PROJECT = /* glsl */`
vec4 mvPosition = vec4( transformed, 1.0 );
#ifdef USE_INSTANCING
	mvPosition = instanceMatrix * mvPosition;
#endif
mvPosition.xyz += vegWind;
mvPosition = modelViewMatrix * mvPosition;
gl_Position = projectionMatrix * mvPosition;
`;

const VEG_WORLDPOS = /* glsl */`
#if defined( USE_ENVMAP ) || defined( DISTANCE ) || defined ( USE_SHADOWMAP ) || defined ( USE_TRANSMISSION ) || NUM_SPOT_LIGHT_COORDS > 0
	vec4 worldPosition = vec4( transformed, 1.0 );
	#ifdef USE_INSTANCING
		worldPosition = instanceMatrix * worldPosition;
	#endif
	worldPosition.xyz += vegWind;
	worldPosition = modelMatrix * worldPosition;
#endif
`;

const VEG_FRAG_PARS = /* glsl */`
varying float vLeaf;
varying float vFade;
varying float vJit;
#ifdef VEG_TINT
varying vec4 vTint;
#endif
` + COMMON_PARS + TINT_FN;

const VEG_FADE_DISCARD = /* glsl */`
if ( vFade < 0.999 && vegIGN( gl_FragCoord.xy ) > vFade ) discard;
`;

const VEG_MAP = /* glsl */`
#ifdef USE_MAP
	diffuseColor *= texture2D( map, vMapUv );
#endif
#ifdef VEG_TINT
	if ( vLeaf > 0.5 ) diffuseColor.rgb = vegTint( diffuseColor.rgb, vJit, vTint );
#endif
`;

const VEG_NORMAL = /* glsl */`
float faceDirection = gl_FrontFacing ? 1.0 : - 1.0;
vec3 normal = normalize( vNormal );
#ifdef DOUBLE_SIDED
	normal *= mix( faceDirection, 1.0, step( 0.5, vLeaf ) );
#endif
vec3 nonPerturbedNormal = normal;
`;

function patchVertex(vs, withWorldPos) {
  vs = vs.replace('#include <common>', '#include <common>\n' + VEG_VERT_PARS)
    .replace('#include <begin_vertex>', VEG_BEGIN)
    .replace('#include <project_vertex>', VEG_PROJECT);
  if (withWorldPos) vs = vs.replace('#include <worldpos_vertex>', VEG_WORLDPOS);
  return vs;
}

/**
 * Near-model material. U: shared { uTime, uViewPos } uniform objects. fade: { start, end, wind, drop }.
 * Returns the material; material.userData.depth is the matching customDepthMaterial (alpha-tested, same wind).
 */
export function createVegMaterial(atlas, U, fade) {
  const uFade = { value: new THREE.Vector4(fade.start, fade.end, fade.wind ?? 1, fade.drop ?? 0) };
  const m = new THREE.MeshStandardMaterial({
    map: atlas, vertexColors: true, roughness: 0.8, metalness: 0, side: THREE.DoubleSide,
    alphaTest: 0.5, alphaToCoverage: true, envMapIntensity: 0.9,
  });
  m.defines = { VEG_LIGHTING: '', VEG_TINT: '' };
  m.onBeforeCompile = (sh) => {
    sh.uniforms.uTime = U.uTime; sh.uniforms.uViewPos = U.uViewPos; sh.uniforms.uFade = uFade;
    sh.vertexShader = patchVertex(sh.vertexShader, true);
    let fs = sh.fragmentShader.replace('#include <common>', '#include <common>\n' + VEG_FRAG_PARS)
      .replace('#include <clipping_planes_fragment>', '#include <clipping_planes_fragment>\n' + VEG_FADE_DISCARD)
      .replace('#include <map_fragment>', VEG_MAP)
      .replace('#include <normal_fragment_begin>', VEG_NORMAL);
    sh.fragmentShader = patchLights(fs);
  };
  m.customProgramCacheKey = () => 'veg-near-v1';
  m.userData.uFade = uFade;

  const d = new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking, map: atlas, alphaTest: 0.5, side: THREE.DoubleSide });
  d.defines = { VEG_DEPTH: '' };
  d.onBeforeCompile = (sh) => {
    sh.uniforms.uTime = U.uTime; sh.uniforms.uViewPos = U.uViewPos; sh.uniforms.uFade = uFade;
    sh.vertexShader = patchVertex(sh.vertexShader, false);
    sh.fragmentShader = sh.fragmentShader.replace('#include <common>', '#include <common>\nvarying float vFade;\n' + COMMON_PARS)
      .replace('#include <clipping_planes_fragment>', '#include <clipping_planes_fragment>\n' + VEG_FADE_DISCARD);
  };
  d.customProgramCacheKey = () => 'veg-depth-v1';
  m.userData.depth = d;
  return m;
}

// ------------------------------------------------------------------ impostors (InstancedBufferGeometry quad)
const IMP_VERT_PARS = /* glsl */`
uniform vec3 uViewPos;
uniform vec4 uFade;          // x fade-in start, y fade-in end
uniform vec4 uSpecies[ 8 ];  // local bounding sphere centre xyz, radius
uniform vec4 uAtlas;         // columns, cell width (uv), cell height (uv), inner padding (cell fraction)
attribute vec4 iPos;         // x, ground y, z, yaw
attribute vec4 iScl;         // horizontal scale, vertical scale, species, -
attribute vec4 aTint;
varying vec4 vTint;
varying float vFade;
varying float vJit;
varying vec2 vQ;
varying vec4 vCA;
varying vec4 vCB;
varying vec3 vW;
varying vec2 vYaw;
vec3 impC; vec3 impR; vec3 impU; float impRw; float impRh;
vec2 impCell( float sp, float row, float k ) {
  float idx = sp * 20.0 + ( row < 1.5 ? row * 8.0 + k : 16.0 + k );
  float col = mod( idx, uAtlas.x ), r = floor( idx / uAtlas.x );
  return vec2( col * uAtlas.y, r * uAtlas.z );
}
` + COMMON_PARS;

const IMP_BEGINNORMAL = /* glsl */`
{
	float sp = floor( iScl.z + 0.5 );
	vec4 S = uSpecies[ int( sp ) ];
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
	vFade = smoothstep( uFade.x, uFade.y, dist );
	vJit = vegHash( iPos.xz );
	vTint = aTint;
	vQ = position.xy * 0.5 + 0.5;
	// view direction in the tree's local frame -> baked cell pair per elevation row
	vec3 fl = vec3( f.x * cy - f.z * sy, f.y, f.x * sy + f.z * cy );
	float az = atan( fl.x, fl.z );
	float el = degrees( asin( clamp( fl.y, -1.0, 1.0 ) ) );
	float e = clamp( ( el - 10.0 ) / 35.0, 0.0, 2.0 );
	float e0 = min( floor( e ), 1.0 ), we = e - e0, e1 = e0 + 1.0;
	float a8 = fract( az / 6.2831853 ) * 8.0;
	float a0 = floor( a8 ), wa = a8 - a0;
	float n1 = e1 > 1.5 ? 4.0 : 8.0;
	float b4 = fract( az / 6.2831853 ) * n1;
	float b0 = floor( b4 ), wb = b4 - b0;
	vCA = vec4( impCell( sp, e0, a0 ), impCell( sp, e0, mod( a0 + 1.0, 8.0 ) ) );
	vCB = vec4( impCell( sp, e1, b0 ), impCell( sp, e1, mod( b0 + 1.0, n1 ) ) );
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
`;

const IMP_WORLDPOS = /* glsl */`
#if defined( USE_ENVMAP ) || defined( DISTANCE ) || defined ( USE_SHADOWMAP ) || defined ( USE_TRANSMISSION ) || NUM_SPOT_LIGHT_COORDS > 0
	vec4 worldPosition = modelMatrix * vec4( transformed, 1.0 );
#endif
`;

const IMP_FRAG_PARS = /* glsl */`
uniform sampler2D uNormalAtlas;
uniform vec4 uAtlas;
varying vec4 vTint;
varying float vFade;
varying float vJit;
varying vec2 vQ;
varying vec4 vCA;
varying vec4 vCB;
varying vec3 vW;
varying vec2 vYaw;
float vLeaf;
` + COMMON_PARS + TINT_FN;

const IMP_FADE_DISCARD = /* glsl */`
if ( vFade < 0.999 && vegIGN( gl_FragCoord.xy ) <= 1.0 - vFade ) discard;
`;

const IMP_MAP = /* glsl */`
vLeaf = 1.0;
vec2 iq = uAtlas.w + vQ * ( 1.0 - 2.0 * uAtlas.w );
vec2 ics = uAtlas.yz;
vec4 iA = mix( texture2D( map, vCA.xy + iq * ics ), texture2D( map, vCA.zw + iq * ics ), vW.x );
vec4 iB = mix( texture2D( map, vCB.xy + iq * ics ), texture2D( map, vCB.zw + iq * ics ), vW.y );
vec4 iAlb = mix( iA, iB, vW.z );
vec4 nA = mix( texture2D( uNormalAtlas, vCA.xy + iq * ics ), texture2D( uNormalAtlas, vCA.zw + iq * ics ), vW.x );
vec4 nB = mix( texture2D( uNormalAtlas, vCB.xy + iq * ics ), texture2D( uNormalAtlas, vCB.zw + iq * ics ), vW.y );
vec4 iNrm = mix( nA, nB, vW.z );
diffuseColor *= iAlb;
diffuseColor.rgb = vegTint( diffuseColor.rgb * 0.93, vJit, vTint );
`;

const IMP_NORMAL = /* glsl */`
float faceDirection = 1.0;
vec3 nObj = iNrm.xyz * 2.0 - 1.0;
nObj = length( nObj ) > 0.05 ? normalize( nObj ) : vec3( 0.0, 1.0, 0.0 );
vec3 nW = vec3( vYaw.x * nObj.x + vYaw.y * nObj.z, nObj.y, - vYaw.y * nObj.x + vYaw.x * nObj.z );
vec3 normal = normalize( ( viewMatrix * vec4( nW, 0.0 ) ).xyz );
vec3 nonPerturbedNormal = normal;
`;

/**
 * Impostor material. species: [{cx,cy,cz,r}] (local bounding spheres, index = species id), atlas layout
 * { cols, cellU, cellV, pad } (uv units; pad = cell fraction).
 */
export function createImpostorMaterial(albedo, normalTex, U, species, layout, fade) {
  const uFade = { value: new THREE.Vector4(fade.start, fade.end, 0, 0) };
  const uSpecies = { value: Array.from({ length: 8 }, (_, i) => { const s = species[i]; return s ? new THREE.Vector4(s.cx, s.cy, s.cz, s.r) : new THREE.Vector4(0, 0, 0, 1); }) };
  const uAtlas = { value: new THREE.Vector4(layout.cols, layout.cellU, layout.cellV, layout.pad) };
  const uNormalAtlas = { value: normalTex };
  const m = new THREE.MeshStandardMaterial({ map: albedo, roughness: 0.85, metalness: 0, alphaTest: 0.5, alphaToCoverage: true, side: THREE.DoubleSide, envMapIntensity: 0.9 });
  m.defines = { VEG_LIGHTING: '' };
  m.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, { uViewPos: U.uViewPos, uFade, uSpecies, uAtlas, uNormalAtlas });
    sh.vertexShader = sh.vertexShader.replace('#include <common>', '#include <common>\n' + IMP_VERT_PARS)
      .replace('#include <beginnormal_vertex>', IMP_BEGINNORMAL)
      .replace('#include <begin_vertex>', IMP_BEGIN)
      .replace('#include <project_vertex>', IMP_PROJECT)
      .replace('#include <worldpos_vertex>', IMP_WORLDPOS);
    const fs = sh.fragmentShader.replace('#include <common>', '#include <common>\n' + IMP_FRAG_PARS)
      .replace('#include <clipping_planes_fragment>', '#include <clipping_planes_fragment>\n' + IMP_FADE_DISCARD)
      .replace('#include <map_fragment>', IMP_MAP)
      .replace('#include <normal_fragment_begin>', IMP_NORMAL);
    sh.fragmentShader = patchLights(fs);
  };
  m.customProgramCacheKey = () => 'veg-impostor-v1';
  m.userData.uFade = uFade;
  return m;
}
