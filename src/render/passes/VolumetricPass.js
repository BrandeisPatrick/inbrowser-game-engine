// Volumetric sun light ("god rays"): the sun's single scattering in the fog medium, shadowed by the sun's cascade
// maps, ray-marched per pixel from the scene depth at a fraction of the drawing buffer, accumulated over frames.
// PostFX order: Scene -> Exposure -> AO -> ... -> Volumetric -> TAA -> ... -> Final; the final pass composites it.
//
// Physics. The medium is THE fog of the final pass (PostFX FinalShader `layer()`): two exponential height layers,
//   sigma(h) = density * ( haze * exp(-hazeK * (h - hazeBase)) + mist * exp(-mistK * h) )        (1/m, h = world y)
// read from atmosphere.fogParams every frame. (Atmosphere fogMode 'scene' fogs the materials with FogExp2 and draws
// no post fog: this medium then matches nothing on screen.) The final pass already applies the medium's extinction
// and its sky-coloured ambient in-scatter analytically; this pass adds only the sun's single-scattering share along
// the view ray from the camera to the surface (or `maxDistance` m, whichever is nearer; the sky marches all of it):
//   r = gate * HG(mu, g) * integral_0^L sigma(t) * vis(t) * T(0 -> t) dt,   T = exp(-tau),  tau = integral of sigma
//   HG(mu, g) = (1 - g^2) / (4 pi (1 + g^2 - 2 g mu)^1.5)   (integrates to 1 over the sphere; mu = dot(view ray,
//   atmosphere.sunDir), sunDir points TOWARD the sun, so g > 0 brightens the view toward the sun)
// Tuning note: the sky cube the fog's ambient in-scatter samples still holds the Sky shader's Mie aureole, so near the
// sun that ambient term and this HG lobe both count some forward scattering; g and the final pass's intensity are
// tuned with that overlap in mind (it is not a physical split).
// The integral is estimated with `steps` samples at t_i = (i + o) * L / steps. tau(0 -> t_i) is the closed form of
// the exponential layers along the ray, the expression of `layer()` (density at the camera, height clamped to
// -100..4000 m like layer(), times t * (1 - exp(-x)) / x with x = k * dy): T at every sample is exact whatever the
// step count, and the unshadowed, phase-free integral equals the final pass's fog opacity 1 - T(L)
// (test/volumetric.mjs). Per sample: one exp per layer (sigma and tau share it) and the T exp. o = 0.5 (midpoint
// rule) with jitter off; with jitter on o = fract(interleaved gradient noise(pixel) + 0.618034 * frame), so the
// temporal blend integrates the offsets and the march is unbiased.
// vis(t): the sun's cascade shadow maps (Atmosphere.shadowCascadeInfo), ONE tap per sample, the cascade picked by
// shadowCascade.js's walk: at an inner cascade with z <= 1 the patch blends on to the next cascade with weight
// w = smoothstep(0.39, 0.47, max |uv - 0.5|); here that cascade is tapped when w <= xi, else xi /= w and the walk goes
// on (z > 1: on, xi kept), so over xi uniform in [0, 1) every cascade gets exactly the patch's weight (the test
// averages a port of both). The last (far) cascade: three's frustum test (uv in 0..1, z <= 1, else lit), faded to
// lit over 0.43..0.495 like the patch (no fade with a single map). xi: a per-sample golden-ratio value with jitter
// on, 0.5 with jitter off. The tap: define VOL_SHADOW_PACKED (shadowCascadeInfo().packed) -> the RGBA-packed map,
// three's texture2DCompare (one plain compare, unpackRGBAToDepth, reversed-depth aware); else the 'fast' kernel's
// compare-mode DepthTexture through a sampler2DShadow, texture( map, vec3( uv, z ) ) (2x2 hardware PCF). Each
// cascade's light.shadow.bias is added to z. VOL_CASCADES (1..4) = shadowCascadeInfo().count; both defines change
// (material.needsUpdate) only when they differ.
// NOTE shadow coverage: a sample outside every cascade box is LIT. The far box (Atmosphere shadows.far, 260 m half
// size at low camera heights, led 0.55 of it along the view) fades to lit from 0.43 of its width, about 370 m ahead
// of the camera and much less behind / to the sides, so with maxDistance 400 the air in front of a shadowed hill out
// there carries unshadowed in-scatter. Keep maxDistance inside the far box (or enlarge shadows.far) where it shows.
// gate: a sun / moon switch, 1 while Atmosphere's day factor smoothstep(-2, 3 deg, sun elevation) is > 0.02 (the
// cascades are the sun's), else 0 (they follow the moon: Atmosphere.lightDir), so the pass is dark at night. It
// carries neither the sun's colour nor its intensity (atmosphere.sun.intensity already includes the day factor).
// Zero output (one clear to (0, 60000), then nothing until it changes; the history restarts): no depth texture, no
// shadow-casting cascade, a cascade map not rendered yet (null), gate 0, or an empty medium (density * fog 0). The
// frame's scalar uniforms (uMarch with the gate, uPhase, uMedium, uClip, uSunDir, uSteps) are written first; the
// shadow and camera matrices keep the last marched frame's.
//
// Temporal (temporal true): ping-pong targets. The previous result is reprojected with the camera (CameraHistory
// uReproj: current unjittered NDC -> previous clip, uv = xy / w * 0.5 + 0.5, the TAA jitter of both frames removed)
// and read with one depth-aware 2x2 tap: the four history texels around that position, each weighted bilinearly x
// max(1e-3, 1 - min(|g - z|, |g - zPrev|) / (0.1 z + 0.3)) (aoAt's upsample weight with TAAPass's tolerance against
// the depth now, z, and the depth the surface had in the previous frame, zPrev); then mix(current, history, blend)
// on r. The history is rejected (this frame only) when none of the four texels is within the tolerance (TAAPass's
// quad rule: a silhouette keeps the texels of its own surface), on the first frame / after reset() / setSize() / a
// toggle of `temporal` / `history` replaced / after a zero output, when the previous clip w <= 0, when the history uv
// is off screen, and (with a CameraHistory) after any frame this pass did not render (uFrame continuity, TAAPass).
// Without a CameraHistory (`history` null) the pass reprojects from the view-projection of the last frame IT
// rendered, so frames it skipped still reproject correctly. tHistory holds the depth texture (bound, never read) on
// fresh frames and with temporal false (one target, output = this frame).
//
// Output `texture` (RG16F, HalfFloatType + RGFormat, LinearFilter, no mipmaps), `scale` of the drawing buffer
// (`maxPixels` budget as AOPass.setSize):
//   r = sun in-scatter, scalar >= 0 (1/sr per unit of sun irradiance: radiance = sun irradiance * r), after the
//       blend; NaN / Inf written as 0
//   g = linear view depth (m) of the depth texel the pixel marched (this frame's, not blended: blended depth would lag
//       camera motion), 60000 for the sky (depth == 1): AOPass's convention
// The final pass must, for EVERY pixel (sky included: shafts against the sky), after its fog:
//   z = depth < 1 ? near * far / (far - (far - near) * depth) : 60000.0
//   col += sunColor * intensity * depthAwareUpsample(texture, z)    (aoAt's AO_UPSAMPLE_DEPTH weights on r / g)
// with sunColor = atmosphere.sunColor (normalised colour) and intensity = knob * atmosphere.sun.intensity (sunColor *
// sun.intensity is the sun light's irradiance, so the shafts stay consistent with how the sun lights surfaces). The
// fog's analytic sun glow (uSunScatter) double-counts forward scattering while this pass is on.
//
// Uniforms: tDepth, tHistory, tShadow0..3 (sampler2D packed / sampler2DShadow), uShadowMatrix[4], uShadowOrigin[4]
// (camera position in each map's space: a sample's coordinate is origin + mat3(matrix) * dir * t), uShadowBias[4],
// uProjInv, uCamWorld, uReproj, uInvVPw, uJitter, uPrevJitter, uSunDir, uPhase ((1 - g^2) / 4 pi, 1 + g^2, 2 g),
// uMedium (haze sigma at the camera, hazeK, mist sigma at the camera, mistK), uSteps, uMarch (maxDistance, gate,
// noise on, frame offset), uClip (near, far), uTemporal (blend, fresh). Per-frame scalars are packed into vectors:
// a double stored into a shared `{ value }` uniform object is boxed by V8, one allocation per store.
// Cost per output pixel: 1 depth read + steps * (1 shadow tap + 3 exp + ~25 ALU) + 4 history texel reads
// (temporal), plus per pixel 1 mat3 * vec3 per cascade and the HG pow. Defaults at 3024x1964: 756x491 = 0.37 MP x 16
// = 5.9 M shadow taps.
// Options (VOLUMETRIC_DEFAULTS; live properties, validated on construction and assignment, re-read every render):
//   scale 0.25 (0..1], maxPixels Infinity, steps 16 (1..128), maxDistance 400 (m; see the shadow coverage NOTE),
//   g 0.55 (-1..1), density 1 (>= 0), temporal true, blend 0.9 [0..1), jitter true. scale / maxPixels resize at once.
// API: new VolumetricPass(camera, history (CameraHistory | null), atmosphere, options); render(renderer, writeBuffer,
// readBuffer) (reads readBuffer.depthTexture only), setSize(width, height) (drawing-buffer px), texture, width,
// height, reset(), dispose(); `enabled` (Pass), `history` and `atmosphere` are live. needsSwap false.
// Allocation-free per frame (the shadowCascadeInfo object is the pass's own). Perspective camera and a standard
// depth buffer (as TAAPass).
import * as THREE from 'three';
import { Pass, FullScreenQuad } from 'three/addons/postprocessing/Pass.js';

/** g-channel value for the sky (AOPass's convention; the final pass's depth-aware upsample uses the same). */
const FAR_DEPTH = 60000;
const MAX_CASCADES = 4;
const DEG = Math.PI / 180;

/**
 * @typedef {object} VolumetricOptions
 * @property {number} [scale] output size as a fraction of the drawing buffer, (0, 1] (0.25)
 * @property {number} [maxPixels] pixel budget of the output (Infinity: none), as AOPass
 * @property {number} [steps] march samples per pixel, integer 1..128 (16)
 * @property {number} [maxDistance] march length cap (m; the sky marches all of it) (400). Samples beyond the far
 *   cascade box (Atmosphere shadows.far) are LIT: keep it inside that box where distant shadowed terrain shows
 * @property {number} [g] Henyey-Greenstein asymmetry toward the sun, (-1, 1) (0.55)
 * @property {number} [density] multiplier on the atmosphere's fog medium, >= 0 (1)
 * @property {boolean} [temporal] reproject and blend the previous result (true)
 * @property {number} [blend] history share per frame, [0, 1) (0.9)
 * @property {boolean} [jitter] per-pixel + per-frame march offset for the temporal blend; false: deterministic (true)
 */

/**
 * @typedef {object} VolumetricSettings
 * @property {number} scale
 * @property {number} maxPixels
 * @property {number} steps
 * @property {number} maxDistance
 * @property {number} g
 * @property {number} density
 * @property {boolean} temporal
 * @property {number} blend
 * @property {boolean} jitter
 */

/**
 * Default options (see the header).
 * @type {Readonly<VolumetricSettings>}
 */
export const VOLUMETRIC_DEFAULTS = Object.freeze({
  scale: 0.25, maxPixels: Infinity, steps: 16, maxDistance: 400, g: 0.55, density: 1, temporal: true, blend: 0.9,
  jitter: true,
});

const RULES = {
  scale: [(v) => typeof v === 'number' && v > 0 && v <= 1, 'a number in (0, 1]'],
  maxPixels: [(v) => typeof v === 'number' && v >= 1, 'a number >= 1 (Infinity or null: no budget)'],
  steps: [(v) => Number.isInteger(v) && v >= 1 && v <= 128, 'an integer in 1..128'],
  maxDistance: [(v) => Number.isFinite(v) && v > 0, 'a finite number > 0 (m)'],
  g: [(v) => typeof v === 'number' && v > -1 && v < 1, 'a number in (-1, 1)'],
  density: [(v) => Number.isFinite(v) && v >= 0, 'a finite number >= 0'],
  temporal: [(v) => typeof v === 'boolean', 'a boolean'],
  blend: [(v) => typeof v === 'number' && v >= 0 && v < 1, 'a number in [0, 1)'],
  jitter: [(v) => typeof v === 'boolean', 'a boolean'],
};

const show = (v) => (typeof v === 'string' ? `'${v}'` : String(v));

function checkOption(key, v) {
  const r = RULES[key];
  if (!r) throw new Error(`[VolumetricPass] unknown option '${key}' (${Object.keys(RULES).join(', ')})`);
  if (key === 'maxPixels' && v === null) v = Infinity;
  if (!r[0](v)) throw new Error(`[VolumetricPass] ${key} must be ${r[1]} (got ${show(v)})`);
  return v;
}

const VolumetricShader = {
  vertexShader: /* glsl */`
    varying vec2 vUv;
    void main() { vUv = uv; gl_Position = vec4( position.xy, 0.0, 1.0 ); }`,
  fragmentShader: /* glsl */`
    #include <packing>
    #define VOL_FAR_DEPTH ${FAR_DEPTH.toFixed(1)}
    uniform highp sampler2D tDepth;
    uniform sampler2D tHistory;
    #ifdef VOL_SHADOW_PACKED
    uniform sampler2D tShadow0;
    #if VOL_CASCADES > 1
    uniform sampler2D tShadow1;
    #endif
    #if VOL_CASCADES > 2
    uniform sampler2D tShadow2;
    #endif
    #if VOL_CASCADES > 3
    uniform sampler2D tShadow3;
    #endif
    // three's texture2DCompare on the RGBA-packed map: 1 (lit) where z is not behind the stored depth
    float volTap( sampler2D map, vec3 c, float bias ) {
      float d = unpackRGBAToDepth( texture2D( map, c.xy ) );
    #ifdef USE_REVERSED_DEPTH_BUFFER
      return step( d, c.z + bias );
    #else
      return step( c.z + bias, d );
    #endif
    }
    #else
    uniform sampler2DShadow tShadow0;
    #if VOL_CASCADES > 1
    uniform sampler2DShadow tShadow1;
    #endif
    #if VOL_CASCADES > 2
    uniform sampler2DShadow tShadow2;
    #endif
    #if VOL_CASCADES > 3
    uniform sampler2DShadow tShadow3;
    #endif
    // the 'fast' kernel's compare-mode depth texture (one tap: 2x2 hardware PCF)
    float volTap( sampler2DShadow map, vec3 c, float bias ) { return texture( map, vec3( c.xy, c.z + bias ) ); }
    #endif
    #if VOL_CASCADES == 1
    #define VOL_LAST 0
    #define VOL_LAST_MAP tShadow0
    #elif VOL_CASCADES == 2
    #define VOL_LAST 1
    #define VOL_LAST_MAP tShadow1
    #elif VOL_CASCADES == 3
    #define VOL_LAST 2
    #define VOL_LAST_MAP tShadow2
    #else
    #define VOL_LAST 3
    #define VOL_LAST_MAP tShadow3
    #endif

    uniform mat4 uShadowMatrix[ 4 ]; // light.shadow.matrix: world -> shadow uv + depth (orthographic: w = 1)
    uniform vec3 uShadowOrigin[ 4 ]; // the camera position through each matrix
    uniform float uShadowBias[ 4 ];  // light.shadow.bias per cascade
    uniform mat4 uProjInv, uCamWorld; // this frame's (jittered) camera
    uniform mat4 uReproj;            // current unjittered NDC -> previous clip
    uniform vec4 uInvVPw;            // 4th row of inverse(viewProj): homogeneous w of a reconstructed point
    uniform vec2 uJitter, uPrevJitter; // TAA jitter of this / the previous frame (drawing-buffer px)
    uniform vec3 uSunDir;            // toward the sun
    uniform vec3 uPhase;             // (1 - g^2) / (4 pi), 1 + g^2, 2 g
    uniform vec4 uMedium;            // haze sigma at the camera (1/m), haze k, mist sigma at the camera, mist k
    uniform int uSteps;
    uniform vec4 uMarch;             // max distance (m), sun gate, noise on (0 / 1), frame offset (0..1)
    uniform vec2 uClip;              // camera near, far
    uniform vec2 uTemporal;          // blend, fresh (1: no history)
    varying vec2 vUv;

    vec3 volDC[ 4 ]; // per pixel: shadow-space change per metre along the ray, by cascade

    // shadowCascade.js's walk at an inner cascade: z > 1 goes on with xi kept; else the patch blends on to the next
    // cascade with weight w, so this cascade is tapped when w <= xi and the walk goes on with xi / w (uniform again)
    bool volPick( vec3 c, inout float xi ) {
      if ( c.z > 1.0 ) return false;
      vec2 e = abs( c.xy - 0.5 );
      float w = smoothstep( 0.39, 0.47, max( e.x, e.y ) );
      if ( w <= xi ) return true;
      xi /= w;
      return false;
    }

    float sunVisibility( float t, float xi ) {
      vec3 c;
    #if VOL_CASCADES > 1
      c = uShadowOrigin[ 0 ] + volDC[ 0 ] * t;
      if ( volPick( c, xi ) ) return volTap( tShadow0, c, uShadowBias[ 0 ] );
    #endif
    #if VOL_CASCADES > 2
      c = uShadowOrigin[ 1 ] + volDC[ 1 ] * t;
      if ( volPick( c, xi ) ) return volTap( tShadow1, c, uShadowBias[ 1 ] );
    #endif
    #if VOL_CASCADES > 3
      c = uShadowOrigin[ 2 ] + volDC[ 2 ] * t;
      if ( volPick( c, xi ) ) return volTap( tShadow2, c, uShadowBias[ 2 ] );
    #endif
      // the last (far) cascade: three's frustum test, then (with cascades) the patch's fade to lit at the box edge
      c = uShadowOrigin[ VOL_LAST ] + volDC[ VOL_LAST ] * t;
      if ( c.x < 0.0 || c.x > 1.0 || c.y < 0.0 || c.y > 1.0 || c.z > 1.0 ) return 1.0;
      float s = volTap( VOL_LAST_MAP, c, uShadowBias[ VOL_LAST ] );
    #if VOL_CASCADES > 1
      vec2 e = abs( c.xy - 0.5 );
      s = mix( s, 1.0, smoothstep( 0.43, 0.495, max( e.x, e.y ) ) );
    #endif
      return s;
    }

    float ign( vec2 p ) { return fract( 52.9829189 * fract( dot( p, vec2( 0.06711056, 0.00583715 ) ) ) ); }

    void main() {
      // the depth texel under this pixel's centre; its own centre is the ray (depth and ray agree)
      ivec2 dsz = textureSize( tDepth, 0 );
      ivec2 ip = min( ivec2( vUv * vec2( dsz ) ), dsz - 1 );
      float d = texelFetch( tDepth, ip, 0 ).x;
      vec2 uv = ( vec2( ip ) + 0.5 ) / vec2( dsz );
      bool sky = d >= 1.0;
      // the final pass's linearisation
      float z = sky ? VOL_FAR_DEPTH : uClip.x * uClip.y / ( uClip.y - ( uClip.y - uClip.x ) * d );
      vec4 vp = uProjInv * vec4( uv * 2.0 - 1.0, d * 2.0 - 1.0, 1.0 );
      vec3 v = vp.xyz / vp.w;
      float dist = length( v );
      vec3 dir = normalize( mat3( uCamWorld ) * ( v / dist ) );
      float len = sky ? uMarch.x : min( dist, uMarch.x );
      float dt = len / float( uSteps );

      float phase = uPhase.x / pow( uPhase.y - uPhase.z * dot( dir, uSunDir ), 1.5 );
      volDC[ 0 ] = mat3( uShadowMatrix[ 0 ] ) * dir;
    #if VOL_CASCADES > 1
      volDC[ 1 ] = mat3( uShadowMatrix[ 1 ] ) * dir;
    #endif
    #if VOL_CASCADES > 2
      volDC[ 2 ] = mat3( uShadowMatrix[ 2 ] ) * dir;
    #endif
    #if VOL_CASCADES > 3
      volDC[ 3 ] = mat3( uShadowMatrix[ 3 ] ) * dir;
    #endif

      float o = uMarch.z > 0.5 ? fract( ign( gl_FragCoord.xy ) + uMarch.w ) : 0.5;
      vec2 kdy = uMedium.yw * dir.y; // k * dy per metre of ray, haze / mist
      float sum = 0.0;
      for ( int i = 0; i < uSteps; i ++ ) {
        float t = ( float( i ) + o ) * dt;
        // the two layers from the camera to t: density e = exp(-x) relative to the camera's, optical depth
        // sigma_cam * t * (1 - e) / x (layer() of the final pass); x clamped where no surface is (60 scale heights)
        vec2 x = clamp( kdy * t, - 60.0, 60.0 );
        vec2 e = exp( - x );
        vec2 f = vec2( abs( x.x ) > 1e-3 ? ( 1.0 - e.x ) / x.x : 1.0 - 0.5 * x.x,
                       abs( x.y ) > 1e-3 ? ( 1.0 - e.y ) / x.y : 1.0 - 0.5 * x.y );
        float sigma = dot( uMedium.xz, e );
        float tau = t * dot( uMedium.xz, f );
        float xi = uMarch.z > 0.5 ? fract( o + 0.618034 * float( i ) ) : 0.5;
        sum += sigma * exp( - tau ) * sunVisibility( t, xi );
      }
      float r = uMarch.y * phase * sum * dt;

      if ( uTemporal.y < 0.5 ) {
        vec2 uvJ = uv + uJitter / vec2( dsz ); // the unjittered position of what this texel shows
        vec4 ndc = vec4( uvJ * 2.0 - 1.0, d * 2.0 - 1.0, 1.0 );
        vec4 pc = uReproj * ndc;
        if ( pc.w > 0.0 ) {
          // where the previous (jittered) frame drew it
          vec2 huv = pc.xy / pc.w * 0.5 + 0.5 - uPrevJitter / vec2( dsz );
          if ( huv.x >= 0.0 && huv.y >= 0.0 && huv.x <= 1.0 && huv.y <= 1.0 ) {
            float zPrev = sky ? VOL_FAR_DEPTH : pc.w / dot( uInvVPw, ndc ); // view depth the surface had, static
            // depth-aware 2x2 history tap: aoAt's weights with TAAPass's tolerance and quad rule
            ivec2 hm = textureSize( tHistory, 0 ) - 1;
            vec2 hp = huv * vec2( hm + 1 ) - 0.5, hf = fract( hp );
            ivec2 h0 = ivec2( floor( hp ) );
            vec2 ha = texelFetch( tHistory, clamp( h0, ivec2( 0 ), hm ), 0 ).rg;
            vec2 hb = texelFetch( tHistory, clamp( h0 + ivec2( 1, 0 ), ivec2( 0 ), hm ), 0 ).rg;
            vec2 hc = texelFetch( tHistory, clamp( h0 + ivec2( 0, 1 ), ivec2( 0 ), hm ), 0 ).rg;
            vec2 hd = texelFetch( tHistory, clamp( h0 + 1, ivec2( 0 ), hm ), 0 ).rg;
            vec4 hz = vec4( ha.y, hb.y, hc.y, hd.y );
            vec4 m = 1.0 - min( abs( hz - z ), abs( hz - zPrev ) ) / ( 0.1 * z + 0.3 ); // >= 0: within tolerance
            if ( max( max( m.x, m.y ), max( m.z, m.w ) ) >= 0.0 ) {
              vec4 w = vec4( ( 1.0 - hf.x ) * ( 1.0 - hf.y ), hf.x * ( 1.0 - hf.y ),
                             ( 1.0 - hf.x ) * hf.y, hf.x * hf.y ) * max( vec4( 1e-3 ), m );
              r = mix( r, dot( w, vec4( ha.x, hb.x, hc.x, hd.x ) ) / dot( w, vec4( 1.0 ) ), uTemporal.x );
            }
          }
        }
      }
      gl_FragColor = vec4( r >= 0.0 && r < 65000.0 ? r : 0.0, z, 0.0, 1.0 ); // NaN / Inf: 0
    }`,
};

export class VolumetricPass extends Pass {
  /**
   * @param {THREE.PerspectiveCamera} camera
   * @param {import('../CameraHistory.js').CameraHistory|null} history camera matrices + TAA jitter (null: the pass
   *   reprojects with the camera itself)
   * @param {import('../Atmosphere.js').Atmosphere} atmosphere sunDir, fogParams, shadowCascadeInfo(out)
   * @param {VolumetricOptions} [o]
   */
  constructor(camera, history, atmosphere, o = {}) {
    super();
    if (!camera || !camera.isCamera) throw new Error('[VolumetricPass] needs a camera');
    const a = atmosphere;
    if (!a || typeof a.shadowCascadeInfo !== 'function' || !a.sunDir || !a.fogParams) {
      throw new Error('[VolumetricPass] needs an atmosphere (sunDir, fogParams, shadowCascadeInfo(out))');
    }
    if (o === null || typeof o !== 'object') throw new Error('[VolumetricPass] options must be an object');
    this.needsSwap = false;
    this.camera = camera;
    /**
     * CameraHistory of the frame (null: none). Live.
     * @type {import('../CameraHistory.js').CameraHistory|null}
     */
    this.history = history || null;
    /** Live. */
    this.atmosphere = atmosphere;
    /** @type {VolumetricSettings} */
    const opts = { ...VOLUMETRIC_DEFAULTS };
    for (const k in o) opts[k] = checkOption(k, o[k]);
    this._o = opts;
    this._W = 1; this._H = 1; this._sized = false;
    // caller-owned shadowCascadeInfo object: the arrays grow once, then are reused
    this._info = {
      count: 0, lights: [], cameras: [], matrices: [], halfSizes: [], maps: [], depthTextures: [], packed: true,
    };
    this._count = 1; this._packed = true; // the defines the material holds
    const rtOpt = {
      type: THREE.HalfFloatType, format: THREE.RGFormat, depthBuffer: false, stencilBuffer: false,
      minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter, generateMipmaps: false,
    };
    this._rt = [new THREE.WebGLRenderTarget(1, 1, rtOpt), new THREE.WebGLRenderTarget(1, 1, rtOpt)];
    this._rt[0].texture.name = 'Volumetric.a'; this._rt[1].texture.name = 'Volumetric.b';
    this._cur = 0;
    this._reset = true;
    this._zeroed = false; // the current target holds the zero output
    this._lastFrame = -2;
    this._lastHistory = this.history; // a different history (PostFX setGBuffer) restarts ours
    this._frame = 0; // own frame count (noise without a history)
    // reprojection without a CameraHistory: from the last frame this pass rendered
    this._vp = new THREE.Matrix4(); this._prevVP = new THREE.Matrix4(); this._inv = new THREE.Matrix4();
    this._hasPrev = false;
    this._clear = new THREE.Color();
    const m4 = () => new THREE.Matrix4(), v3 = () => new THREE.Vector3();
    this.material = new THREE.ShaderMaterial({
      name: 'PostFXVolumetric',
      defines: { VOL_CASCADES: 1, VOL_SHADOW_PACKED: 1 },
      uniforms: {
        tDepth: { value: null }, tHistory: { value: null },
        tShadow0: { value: null }, tShadow1: { value: null }, tShadow2: { value: null }, tShadow3: { value: null },
        uShadowMatrix: { value: [m4(), m4(), m4(), m4()] },
        uShadowOrigin: { value: [v3(), v3(), v3(), v3()] },
        uShadowBias: { value: new Float32Array(4) },
        uProjInv: { value: new THREE.Matrix4() }, uCamWorld: { value: new THREE.Matrix4() },
        uReproj: { value: new THREE.Matrix4() }, uInvVPw: { value: new THREE.Vector4(0, 0, 0, 1) },
        uJitter: { value: new THREE.Vector2() }, uPrevJitter: { value: new THREE.Vector2() },
        uSunDir: { value: new THREE.Vector3(0, 1, 0) },
        uPhase: { value: new THREE.Vector3() },
        uMedium: { value: new THREE.Vector4() },
        uSteps: { value: 16 },
        // per-frame scalars packed in vectors (see the header: no boxed doubles per frame)
        uMarch: { value: new THREE.Vector4(400, 0, 1, 0) }, uClip: { value: new THREE.Vector2(0.1, 1000) },
        uTemporal: { value: new THREE.Vector2(0.9, 1) },
      },
      vertexShader: VolumetricShader.vertexShader,
      fragmentShader: VolumetricShader.fragmentShader,
      blending: THREE.NoBlending, depthTest: false, depthWrite: false,
    });
    const u = this.material.uniforms;
    this._shadowU = [u.tShadow0, u.tShadow1, u.tShadow2, u.tShadow3];
    this._quad = new FullScreenQuad(this.material);
  }

  /**
   * The result of the last render: RG16F, r = sun in-scatter (>= 0), g = linear view depth (m; sky 60000).
   * @returns {THREE.Texture}
   */
  get texture() { return this._rt[this._cur].texture; }
  /** Output width (px). */
  get width() { return this._rt[0].width; }
  /** Output height (px). */
  get height() { return this._rt[0].height; }

  /** Output size as a fraction of the drawing buffer, (0, 1]; resizes at once. @returns {number} */
  get scale() { return this._o.scale; }
  /** @param {number} v */
  set scale(v) { this._o.scale = checkOption('scale', v); if (this._sized) this.setSize(this._W, this._H); }
  /** Pixel budget of the output (Infinity: none; null sets Infinity); resizes at once. @returns {number} */
  get maxPixels() { return this._o.maxPixels; }
  /** @param {number|null} v */
  set maxPixels(v) { this._o.maxPixels = checkOption('maxPixels', v); if (this._sized) this.setSize(this._W, this._H); }
  /** March samples per pixel (integer 1..128). @returns {number} */
  get steps() { return this._o.steps; }
  /** @param {number} v */
  set steps(v) { this._o.steps = checkOption('steps', v); }
  /**
   * March length cap (m); the sky marches all of it. Samples beyond the far cascade box (Atmosphere shadows.far) are
   * LIT: keep it inside that box where distant shadowed terrain shows (header, shadow coverage NOTE).
   * @returns {number}
   */
  get maxDistance() { return this._o.maxDistance; }
  /** @param {number} v */
  set maxDistance(v) { this._o.maxDistance = checkOption('maxDistance', v); }
  /** Henyey-Greenstein asymmetry toward the sun, (-1, 1). @returns {number} */
  get g() { return this._o.g; }
  /** @param {number} v */
  set g(v) { this._o.g = checkOption('g', v); }
  /** Multiplier on the atmosphere's fog medium (>= 0; 0 outputs 0). @returns {number} */
  get density() { return this._o.density; }
  /** @param {number} v */
  set density(v) { this._o.density = checkOption('density', v); }
  /** Reproject and blend the previous result (a toggle restarts the history). @returns {boolean} */
  get temporal() { return this._o.temporal; }
  /** @param {boolean} v */
  set temporal(v) { if (checkOption('temporal', v) !== this._o.temporal) { this._o.temporal = v; this._reset = true; } }
  /** History share per frame, [0, 1). @returns {number} */
  get blend() { return this._o.blend; }
  /** @param {number} v */
  set blend(v) { this._o.blend = checkOption('blend', v); }
  /** Per-pixel + per-frame march offset (false: midpoint samples, deterministic). @returns {boolean} */
  get jitter() { return this._o.jitter; }
  /** @param {boolean} v */
  set jitter(v) { this._o.jitter = checkOption('jitter', v); }

  /** The next frame starts a new history (teleport, camera cut). */
  reset() { this._reset = true; }

  /**
   * @param {number} width drawing-buffer width (px)
   * @param {number} height drawing-buffer height (px)
   */
  setSize(width, height) {
    this._W = width; this._H = height; this._sized = true;
    const W = Math.max(1, width), H = Math.max(1, height), s0 = this._o.scale, max = this._o.maxPixels;
    let w = Math.max(1, Math.round(W * s0)), h = Math.max(1, Math.round(H * s0));
    if (w * h > max) { // over the budget: the largest uniform scale that fits, floored (AOPass.setSize)
      const s = Math.sqrt(max / (W * H));
      w = Math.max(1, Math.floor(W * s)); h = Math.max(1, Math.floor(H * s));
    }
    this._rt[0].setSize(w, h); this._rt[1].setSize(w, h);
    this._reset = true;
    this._zeroed = false;
  }

  /**
   * March, blend and write `texture`. Reads readBuffer.depthTexture only; writeBuffer is not touched.
   * @param {THREE.WebGLRenderer} renderer
   * @param {THREE.WebGLRenderTarget|null} writeBuffer
   * @param {THREE.WebGLRenderTarget} readBuffer
   */
  render(renderer, writeBuffer, readBuffer) {
    const atmo = this.atmosphere, o = this._o, cam = this.camera, h = this.history, u = this.material.uniforms;
    const depth = readBuffer ? readBuffer.depthTexture : null;
    const info = atmo.shadowCascadeInfo(this._info);
    const n = Math.min(info.count, MAX_CASCADES), packed = !!info.packed;
    const f = h !== null ? h.uniforms.uFrame.value : this._frame;
    this._frame++;
    // Doubles stay in this function and go to uniforms by field stores (no calls with number arguments: a call V8
    // does not inline boxes them, which allocates every frame).
    // gate: the sun / moon switch of Atmosphere.setTime (day = smoothstep(-2, 3 deg, elevation) > 0.02: the sun's
    // cascades). No smoothstep weight here: atmosphere.sun.intensity, which the final pass applies, carries it.
    const sy = atmo.sunDir.y, el = Math.asin(sy < -1 ? -1 : sy > 1 ? 1 : sy);
    let day = (el + 2 * DEG) / (5 * DEG);
    day = day < 0 ? 0 : day > 1 ? 1 : day;
    day = day * day * (3 - 2 * day);
    const gate = day > 0.02 ? 1 : 0;
    // the medium at the camera (layer(): height relative to each layer's base, clamped to -100..4000 m)
    const ce = cam.matrixWorld.elements, cx = ce[12], cy = ce[13], cz = ce[14];
    const fp = atmo.fogParams, dens = o.density, yh = cy - (fp.hazeBase || 0);
    const sH = dens * fp.haze * Math.exp(-fp.hazeK * (yh < -100 ? -100 : yh > 4000 ? 4000 : yh));
    const sM = dens * fp.mist * Math.exp(-fp.mistK * (cy < -100 ? -100 : cy > 4000 ? 4000 : cy));
    // the frame's scalars first (read back consistently on zero frames too)
    const clip = u.uClip.value, mr = u.uMarch.value, tp = u.uTemporal.value;
    clip.x = cam.near; clip.y = cam.far;
    u.uSunDir.value.copy(atmo.sunDir);
    const g = o.g, ph = u.uPhase.value, md = u.uMedium.value;
    ph.x = (1 - g * g) / (4 * Math.PI); ph.y = 1 + g * g; ph.z = 2 * g;
    md.x = sH; md.y = fp.hazeK; md.z = sM; md.w = fp.mistK;
    u.uSteps.value = o.steps;
    mr.x = o.maxDistance;
    mr.y = gate;
    mr.z = o.jitter ? 1 : 0;
    mr.w = o.jitter ? ((f % 4096) * 0.6180339887498949) % 1 : 0;
    tp.x = o.blend;

    let ready = depth !== null && depth !== undefined && n > 0 && gate > 0 && sH + sM > 0;
    for (let i = 0; ready && i < n; i++) if (!(packed ? info.maps[i] : info.depthTextures[i])) ready = false;
    if (!ready) { this._zero(renderer); this._lastFrame = f; return; }

    if (n !== this._count || packed !== this._packed) { // one program per (count, packed); recompiled only on change
      const d = this.material.defines;
      d.VOL_CASCADES = n;
      if (packed) d.VOL_SHADOW_PACKED = 1; else delete d.VOL_SHADOW_PACKED;
      this._count = n; this._packed = packed;
      this.material.needsUpdate = true;
    }
    const mats = u.uShadowMatrix.value, orig = u.uShadowOrigin.value, bias = u.uShadowBias.value;
    for (let i = 0; i < MAX_CASCADES; i++) {
      if (i < n) {
        const m = info.matrices[i], e = m.elements, ov = orig[i], l = info.lights[i];
        mats[i].copy(m);
        ov.x = e[0] * cx + e[4] * cy + e[8] * cz + e[12]; // affine (orthographic shadow camera): w = 1
        ov.y = e[1] * cx + e[5] * cy + e[9] * cz + e[13];
        ov.z = e[2] * cx + e[6] * cy + e[10] * cz + e[14];
        bias[i] = l ? l.shadow.bias : 0;
        this._shadowU[i].value = packed ? info.maps[i] : info.depthTextures[i];
      } else {
        this._shadowU[i].value = null;
        bias[i] = 0;
      }
    }
    u.tDepth.value = depth;
    u.uProjInv.value.copy(cam.projectionMatrixInverse);
    u.uCamWorld.value.copy(cam.matrixWorld);

    // reprojection: CameraHistory's matrices (previous frame), or the camera's own (last frame this pass rendered)
    let fresh = this._reset || f !== this._lastFrame + 1 || h !== this._lastHistory;
    this._lastHistory = h;
    if (h !== null) {
      const hu = h.uniforms;
      fresh = fresh || h.fresh;
      u.uReproj.value.copy(hu.uReproj.value);
      const iv = hu.uInvViewProj.value.elements, w = u.uInvVPw.value;
      w.x = iv[3]; w.y = iv[7]; w.z = iv[11]; w.w = iv[15];
      u.uJitter.value.copy(hu.uJitter.value);
      u.uPrevJitter.value.copy(hu.uPrevJitter.value);
    } else {
      this._vp.multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse);
      this._inv.multiplyMatrices(cam.matrixWorld, cam.projectionMatrixInverse);
      fresh = fresh || !this._hasPrev;
      u.uReproj.value.multiplyMatrices(this._hasPrev ? this._prevVP : this._vp, this._inv);
      this._prevVP.copy(this._vp); this._hasPrev = true;
      const iv = this._inv.elements, w = u.uInvVPw.value;
      w.x = iv[3]; w.y = iv[7]; w.z = iv[11]; w.w = iv[15];
      u.uJitter.value.set(0, 0); u.uPrevJitter.value.set(0, 0);
    }
    this._lastFrame = f;
    let target;
    if (o.temporal) {
      const prev = this._rt[this._cur];
      this._cur ^= 1;
      target = this._rt[this._cur];
      u.tHistory.value = fresh ? depth : prev.texture; // fresh: a valid texture, never read
      tp.y = fresh ? 1 : 0;
      this._reset = false;
    } else {
      target = this._rt[this._cur];
      u.tHistory.value = depth; // never read
      tp.y = 1;
      this._reset = true; // switching temporal on starts a new history
    }
    renderer.setRenderTarget(target);
    this._quad.render(renderer);
    this._zeroed = false;
  }

  dispose() {
    this._rt[0].dispose(); this._rt[1].dispose();
    this.material.dispose();
    this._quad.dispose();
  }

  // zero output (r 0, g sky depth) in the current target, once; the next marched frame starts a new history
  _zero(renderer) {
    this._reset = true;
    this._hasPrev = false;
    if (this._zeroed) return;
    renderer.getClearColor(this._clear);
    const ca = renderer.getClearAlpha();
    // setRenderTarget before setClearColor: the clear colour is converted for the bound target
    renderer.setRenderTarget(this._rt[this._cur]);
    renderer.setClearColor(_zeroClear, 1);
    renderer.clear(true, false, false);
    renderer.setClearColor(this._clear, ca);
    this._zeroed = true;
  }
}
const _zeroClear = new THREE.Color(0, FAR_DEPTH, 0);
