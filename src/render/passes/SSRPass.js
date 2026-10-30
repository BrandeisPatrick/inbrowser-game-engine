// Screen-space reflections on the G-buffer path (PostFX order: Scene (MRT) -> Exposure -> AO -> SSR -> ... -> TAA ->
// ... -> Final). One full-screen pass at `scale` of the drawing buffer traces, per output pixel, the mirror ray of the
// G-buffer normal through this frame's depth buffer and returns the scene colour it hits, premultiplied by the share of
// the forward-shaded colour that reflection replaces. The forward colour already holds the environment-map reflection,
// so the composite swaps that share for the screen-space one; where the trace misses, the weight is 0 and the
// environment reflection stays. The consumer (TAA on every 3x3 tap, or the final pass) pastes SSR_COMPOSITE_GLSL:
//   engineSsrComposite(col, ssr) = col * (1 - ssr.a) + ssr.rgb        (ssr = texture(ssr.texture, uv), bilinear)
//
// Per output pixel:
//   1. The full-resolution texel under the pixel centre. G-buffer normal.a: flags bit0 wroteGBuffer and bit1 hasNormal
//      both set, else (0, 0, 0, 0) after this one read (sky, unlit, transparent-only, the clear colour); roughness (b)
//      >= maxRoughness: (0, 0, 0, 0) too.
//   2. View-space position from the depth through camera.projectionMatrixInverse (jittered under TAA, like the depth),
//      V toward the camera, N = engineOctDecode(normal.rg) (the shaded normal, normal maps included), R = reflect(-V, N).
//   3. Weight before tracing: specular share * roughness fade * toward-camera fade (below). N.V <= 0 or weight 0:
//      (0, 0, 0, 0), no trace.
//   4. The ray, origin + R * maxDistance (shortened to the near plane when it heads there), is projected once at both
//      ends. The march interpolates the screen position and 1 / w (w = linear depth) linearly in screen space, which is
//      perspective-correct (no view-space steps projected one by one). It covers the on-screen part of the segment,
//      from two full-resolution pixels past the origin to where the segment leaves the screen (at one pixel the
//      nearest depth texel can still be the reflector's own, off by half a texel of its depth slope, which reads as a
//      hit on a grazing floor): `steps` samples at (i + o) / steps of that span, o = 0.5, or with `jitter` the
//      interleaved-gradient noise of the output pixel offset by 5.588238 * (uFrame & 63) per frame (off by default,
//      see Limitations). A ray whose start lies inside a surface (behind the depth there by less than the band: a
//      normal-mapped normal turned the reflection under the surface it leaves, e.g. water waves) is a miss: it would
//      only find that surface again.
//   5. Band of a surface at linear depth z: thickness + z * thicknessScale (m). Sample i hits when the ray is behind the
//      surface at that sample's depth texel (nearest; depth 1 = sky never hits) and its depth span since the previous
//      sample reaches into [scene, scene + band]: a stride that crosses a surface between two samples counts, a ray
//      that comes out from behind a surface or passes in front of one (a ray whose depth decreases, R.z > 0) does not.
//      `refine` bisections between the last miss and the hit then home in on the crossing: a midpoint behind its surface
//      becomes the hit end when it lies inside that surface's band, or when that surface is not nearer than the hit
//      end's by more than a band (the thick surface the stride crossed); a midpoint in front of the depth buffer, or
//      behind a nearer surface by more than the band (a post the ray passed behind on its way to a wall), becomes the
//      miss end, so the colour is never read from the occluder.
//   6. No hit inside the span (the ray left the screen, reached maxDistance or the near plane, or saw only sky): miss,
//      (0, 0, 0, 0). Hit: reflected radiance = the scene colour at the hit uv, one bilinear tap; weight *= edge fade *
//      distance fade; output (radiance * weight, weight). A NaN / Inf scene colour reflects as nothing.
//      metalTint (a stopgap heuristic until the G-buffer carries F0, for the lead to measure): the share's F0 =
//      mix(0.04, 1, metalness) assumes a white metal (three uses mix(0.04, albedo, metalness)), so a dark metal would
//      paste bright surroundings over itself (a navy pillar showed 35 % of the sand). With own = this pixel's scene
//      colour and maxC its largest component, the radiance is multiplied by mix(1, own / maxC * min(1, maxC /
//      maxComponent(radiance)), metalness): the reflection takes the pixel's hue and at most its own brightness, so a
//      metal cannot reflect brighter than it renders (and a black metal in deep shadow reflects nothing).
// Weights (hit confidence is 1 for a hit, 0 for a miss, times these):
//   specular share  F0 * A + B with three's DFGApprox (A, B) (Karis' analytic env-BRDF fit, the term three's physical
//                   material applies to its environment reflection) and F0 = mix(0.04, 1, metalness), f90 = 1: Schlick's
//                   Fresnel on N.V at roughness 0, flattened as roughness grows, so rough dielectrics get ~0.05.
//   roughness       1 up to maxRoughness - 0.2, smoothstep to 0 at maxRoughness (0 at and above it; default 0.5: full
//                   weight up to roughness 0.3, the first knob to measure).
//   toward camera   a ray with view-space R.z > 0 approaches the camera: what it would hit faces away or lies off
//                   screen. 1 at R.z <= 0, smoothstep to 0 (miss) at R.z = 0.25, so the cut line does not pop.
//   screen edge     smoothstep(0, edgeFade, distance of the hit uv to the nearest border) (fraction of the screen).
//   distance        1 - smoothstep(0.7, 1.0, metres along the ray to the hit / maxDistance).
// Output `texture`: RGBA16F (HalfFloatType, RGBAFormat), LinearFilter, no mipmaps, round(w * scale) x round(h * scale)
// (capped at maxPixels, at least 1 x 1), PREMULTIPLIED: rgb = reflected scene-linear HDR radiance * weight, a = weight
// in 0..1. Sample it bilinearly at the full-resolution uv. Every texel is written each render; a disabled pass keeps
// its last content (the compositor composites only while `enabled`). Without a G-buffer or a depth texture render()
// clears it to (0, 0, 0, 0) (a no-op composite).
// Options = SSR_DEFAULTS, each a live property (validated; steps / refine / jitter / metalTint are defines, so a change
// recompiles once; scale / maxPixels resize at once; the rest are uniforms written by their setters): scale 0.5,
// maxPixels Infinity (pixel budget, as AOPass), steps 24, refine 4, maxDistance 80 (m), thickness 0.4 (m),
// thicknessScale 0 (m of band per m of depth: a fixed 0.4 m band flickers with the sample phase on far grazing ground
// while the camera moves), maxRoughness 0.5, edgeFade 0.1, jitter false (see Limitations), metalTint true, source
// 'current' (this frame's scene colour; 'history' is reserved and throws for now).
// Uniforms: tColor, tDepth, tNormal, uProj, uProjInv, uRes (full-resolution size), uNear, uFar, uMaxDistance,
// uThickness, uThicknessScale, uMaxRoughness, uEdgeFade, uFrame (copied from history.uniforms.uFrame). Defines:
// SSR_STEPS, SSR_REFINE (values), SSR_JITTER, SSR_METAL_TINT (present when on).
// API: new SSRPass(camera, history, gbuffer, opts); render(renderer, writeBuffer, readBuffer) (readBuffer.texture = the
// HDR scene colour, readBuffer.depthTexture = its depth); setSize(w, h) (drawing-buffer pixels); reset() (no state:
// no-op); texture; dispose(); camera / history / gbuffer are plain properties (reassign after a rebuild).
// Cost: reads per output pixel that passes step 1: normal 1, depth 1, 1 depth tap at the march start, up to `steps`
// depth taps (the march stops at the first hit), `refine` depth taps and 1 colour tap on a hit, +1 colour tap (own
// colour) with metalTint where metalness > 0: at the defaults at most 2 + 1 + 24 + 4 + 2 = 33 taps; a pixel rejected
// in step 1 or 3 costs 1-2 reads. The depth taps are full-resolution texelFetches along a line (no hierarchical depth).
// Limitations:
//   - jitter does not integrate under this TAA: the TAA_SSR hook composites the SAME SSR texel into all 9 taps, so the
//     clip box gets no variance from the reflection and the history is clipped to the current frame's hit / miss
//     every frame; wherever hit / miss depends on the sample phase the speckle is permanent (seen on Canyon Run's
//     gantry pillar; solid with jitter off). Jitter needs a temporal filter of this pass's own (later); off by default.
//   - no roughness blur: the hit colour is one bilinear tap, so a glossy surface reflects as sharply as a mirror up
//     to the roughness fade (the "mirror look" at roughness >= 0.3); a roughness-driven blur is a later item.
//   - metalTint is a heuristic (step 6): the right fix is F0 (albedo luminance) in the G-buffer (0.9.0).
//   - 8-bit octahedral normals are quantised by up to ~0.7 deg, which turns the mirror ray by up to twice that (about
//     0.5 m of wander at 40 m, up to ~1 m): a faint waviness in reflections on flat floors.
//   - the reflected colour is this frame's pre-TAA HDR scene colour, so the reflected point carries no AO / fog /
//     bloom of its own.
//   - one depth layer: a stride can step over an object thinner than it, a ray passing within the band behind an
//     object hits it, there is no back-face test at the hit.
//   - the half-resolution texture upsampled bilinearly bleeds about one full-resolution pixel of reflection across
//     silhouettes (TAA hides most of it).
//   - perspective cameras with a standard depth buffer only (linear depth from near / far, as TAAPass).
import * as THREE from 'three';
import { Pass, FullScreenQuad } from 'three/addons/postprocessing/Pass.js';
import { GBUFFER_GLSL_DECODE } from '../GBuffer.js';

/**
 * @typedef {object} SSROptions
 * @property {number} [scale] output size as a fraction of the drawing buffer, (0, 1]
 * @property {number} [maxPixels] pixel budget of the output (Infinity / null: none)
 * @property {number} [steps] march samples, integer 1..128
 * @property {number} [refine] bisection steps after a hit, integer 0..16
 * @property {number} [maxDistance] ray length in view space (m)
 * @property {number} [thickness] depth behind a surface that still counts as a hit (m)
 * @property {number} [thicknessScale] band growth per metre of the surface's depth (m / m), [0, 1]
 * @property {number} [maxRoughness] roughness at and above which the weight is 0, (0, 1]
 * @property {number} [edgeFade] fraction of the screen over which hits fade at the borders, [0, 0.5]
 * @property {boolean} [jitter] per-pixel, per-frame march start (needs a temporal filter TAA does not provide)
 * @property {boolean} [metalTint] tint reflections on metals by the pixel's own colour
 * @property {string} [source] 'current' (this frame's scene colour; 'history' reserved)
 */

/**
 * Default options (every one is also a live property of SSRPass).
 * @type {Readonly<Required<SSROptions>>}
 */
export const SSR_DEFAULTS = Object.freeze({
  scale: 0.5,
  maxPixels: Infinity,
  steps: 24,
  refine: 4,
  maxDistance: 80,
  thickness: 0.4,
  thicknessScale: 0,
  maxRoughness: 0.5,
  edgeFade: 0.1,
  jitter: false,
  metalTint: true,
  source: 'current',
});

/** Accepted `source` values ('history', the reprojected previous TAA output, is reserved). */
export const SSR_SOURCES = Object.freeze(['current']);

/** Composite of the premultiplied SSR texel over the forward colour (paste into the consuming shader). */
export const SSR_COMPOSITE_GLSL = 'vec3 engineSsrComposite( vec3 col, vec4 ssr ) { return col * ( 1.0 - ssr.a ) + ssr.rgb; }';

const SSRShader = {
  vertexShader: /* glsl */`
    varying vec2 vUv;
    void main() { vUv = uv; gl_Position = vec4( position.xy, 0.0, 1.0 ); }`,
  fragmentShader: /* glsl */`
    uniform sampler2D tColor; // this frame's HDR scene colour (linear filtering)
    uniform highp sampler2D tDepth; // its depth, standard 0..1
    uniform sampler2D tNormal; // G-buffer normal + roughness + metalness / flags (nearest)
    uniform mat4 uProj; // the camera's projection (jittered under TAA, as the depth) and its inverse
    uniform mat4 uProjInv;
    uniform vec2 uRes; // full-resolution size (px) of the three textures
    uniform float uNear, uFar;
    uniform float uMaxDistance, uThickness, uThicknessScale, uMaxRoughness, uEdgeFade;
    uniform int uFrame;
    varying vec2 vUv;

    ${GBUFFER_GLSL_DECODE}

    float linearDepth( float d ) { return uNear * uFar / ( uFar - ( uFar - uNear ) * d ); } // standard (not reversed) depth
    // depth at a march position in full-resolution pixels (nearest texel; positions stay inside [0.5, uRes - 0.5])
    float depthAt( vec2 px ) { return texelFetch( tDepth, ivec2( px ), 0 ).x; }
    // depth behind a surface at linear depth z that still counts as that surface (m)
    float band( float z ) { return uThickness + z * uThicknessScale; }

    // three's DFGApprox (Karis' analytic fit of the split-sum env BRDF): specular = F0 * A + f90 * B
    vec2 envBRDF( float NdotV, float roughness ) {
      const vec4 c0 = vec4( - 1.0, - 0.0275, - 0.572, 0.022 );
      const vec4 c1 = vec4( 1.0, 0.0425, 1.04, - 0.04 );
      vec4 r = roughness * c0 + c1;
      float a004 = min( r.x * r.x, exp2( - 9.28 * NdotV ) ) * r.x + r.y;
      return vec2( - 1.04, 1.04 ) * a004 + r.zw;
    }

    void main() {
      gl_FragColor = vec4( 0.0 ); // every early return is "no reflection": the forward colour stays
      ivec2 ip = min( ivec2( vUv * uRes ), ivec2( uRes ) - 1 ); // the full-resolution texel under this pixel
      vec4 gb = texelFetch( tNormal, ip, 0 );
      int flags = engineGbFlags( gb.a );
      float rough = gb.b;
      if ( ( flags & ENGINE_GB_WROTE ) == 0 || ( flags & ENGINE_GB_HAS_NORMAL ) == 0 || rough >= uMaxRoughness ) return;
      float d = texelFetch( tDepth, ip, 0 ).x;
      if ( d >= 1.0 ) return;

      // view-space surface point, view vector, normal, mirror direction
      vec2 uv = ( vec2( ip ) + 0.5 ) / uRes;
      vec4 hp = uProjInv * vec4( uv * 2.0 - 1.0, d * 2.0 - 1.0, 1.0 );
      vec3 P = hp.xyz / hp.w;
      vec3 V = - normalize( P );
      vec3 N = engineOctDecode( gb.rg );
      float NdotV = dot( N, V );
      vec3 R = reflect( - V, N );

      // weight before tracing: specular share * roughness fade * toward-camera fade
      float metal = engineGbMetal( gb.a );
      vec2 fab = envBRDF( clamp( NdotV, 0.0, 1.0 ), rough );
      float fr = clamp( ( uMaxRoughness - rough ) / 0.2, 0.0, 1.0 );
      float w = clamp( mix( 0.04, 1.0, metal ) * fab.x + fab.y, 0.0, 1.0 ) * fr * fr * ( 3.0 - 2.0 * fr )
              * ( 1.0 - smoothstep( 0.0, 0.25, R.z ) );
      if ( NdotV <= 0.0 || w <= 0.0 ) return;

      // the ray in screen space: both ends projected once; screen position and 1 / w are affine along the segment
      float len = R.z > 0.0 ? min( uMaxDistance, ( - uNear - P.z ) / R.z ) : uMaxDistance; // stop at the near plane
      vec4 H0 = uProj * vec4( P, 1.0 ), H1 = uProj * vec4( P + R * len, 1.0 );
      float k0 = 1.0 / H0.w, k1 = 1.0 / H1.w; // 1 / linear depth at both ends
      vec2 S0 = ( H0.xy * k0 * 0.5 + 0.5 ) * uRes, S1 = ( H1.xy * k1 * 0.5 + 0.5 ) * uRes; // full-resolution pixels
      vec2 dS = S1 - S0;
      // march span in the segment parameter u (u = 1 at the ray end): from two pixels past the origin to where the
      // segment leaves the screen (the centres of the border texels)
      vec2 lim = mix( vec2( 0.5 ), uRes - 0.5, step( 0.0, dS ) );
      vec2 te = vec2( abs( dS.x ) > 1e-4 ? ( lim.x - S0.x ) / dS.x : 1.0, abs( dS.y ) > 1e-4 ? ( lim.y - S0.y ) / dS.y : 1.0 );
      float uEnd = min( 1.0, min( te.x, te.y ) );
      float u0 = 2.0 / max( length( dS ), 1e-4 ); // two pixels past the origin (see the header)
      if ( uEnd <= u0 ) return; // the ray stays near the origin's pixel or leaves the screen at once
      float zPrev = 1.0 / mix( k0, k1, u0 ); // linear depth of the ray at the march start
      float zStart = linearDepth( depthAt( mix( S0, S1, u0 ) ) );
      if ( zPrev > zStart && zPrev < zStart + band( zStart ) ) return; // starts inside a surface (normal-mapped self-hit)

    #ifdef SSR_JITTER
      // interleaved-gradient noise (Jimenez), a new pattern each frame (needs a temporal filter: see Limitations)
      float o = fract( 52.9829189 * fract( dot( gl_FragCoord.xy + 5.588238 * float( uFrame & 63 ), vec2( 0.06711056, 0.00583715 ) ) ) );
    #else
      float o = 0.5;
    #endif
      float du = ( uEnd - u0 ) / float( SSR_STEPS );
      float uPrev = u0, uHit = - 1.0, zHit = 0.0;
      for ( int i = 0; i < SSR_STEPS; i ++ ) {
        float u = u0 + ( float( i ) + o ) * du;
        float zRay = 1.0 / mix( k0, k1, u );
        float sd = depthAt( mix( S0, S1, u ) );
        float zScene = linearDepth( sd );
        // hit: the ray is behind the surface here and its depth span since the previous sample reaches into
        // [scene, scene + band] (crossing in one stride counts; coming out from behind or passing in front does not)
        if ( sd < 1.0 && zRay > zScene && min( zRay, zPrev ) < zScene + band( zScene ) ) { uHit = u; zHit = zScene; break; }
        uPrev = u; zPrev = zRay;
      }
      if ( uHit < 0.0 ) return; // miss: left the screen, reached maxDistance / the near plane, or only sky

      // bisection between the last miss and the hit. A midpoint behind its surface is the new hit end when it lies
      // inside that surface's band, or when that surface is not nearer than the hit end's by more than a band (the
      // same thick surface the stride crossed, or one the march stepped over before it); behind a nearer surface by
      // more than the band (an occluder the ray passed behind on its way to the hit) or in front: the new miss end
      float a = uPrev, b = uHit;
      for ( int i = 0; i < SSR_REFINE; i ++ ) {
        float m = 0.5 * ( a + b );
        float zs = linearDepth( depthAt( mix( S0, S1, m ) ) ), zr = 1.0 / mix( k0, k1, m );
        if ( zr > zs && ( zr < zs + band( zs ) || zs > zHit - band( zHit ) ) ) { b = m; zHit = zs; } else a = m;
      }

      vec2 hitUv = mix( S0, S1, b ) / uRes;
      float dist = len * b * k1 / mix( k0, k1, b ); // metres along the ray (t / w is affine in screen space too)
      vec2 e = min( hitUv, 1.0 - hitUv );
      w *= smoothstep( 0.0, uEdgeFade, min( e.x, e.y ) ) * ( 1.0 - smoothstep( 0.7, 1.0, dist / uMaxDistance ) );

      vec3 c = textureLod( tColor, hitUv, 0.0 ).rgb; // bilinear; explicit LOD (non-uniform control flow)
    #ifdef SSR_METAL_TINT
      if ( metal > 0.0 ) {
        // stopgap until the G-buffer carries F0 (see the header): the share assumed a white metal, so the reflected
        // radiance takes this pixel's hue and at most its brightness (a metal cannot reflect brighter than it renders)
        vec3 own = texelFetch( tColor, ip, 0 ).rgb;
        float maxC = max( own.r, max( own.g, own.b ) );
        c *= mix( vec3( 1.0 ), own / max( maxC, 1e-3 ) * min( 1.0, maxC / max( max( c.r, max( c.g, c.b ) ), 1e-3 ) ), metal );
      }
    #endif
      if ( ! all( lessThan( abs( c ), vec3( 65000.0 ) ) ) ) c = vec3( 0.0 ); // NaN / Inf from the scene: no reflection
      gl_FragColor = vec4( max( c, 0.0 ) * w, w );
    }`,
};

export class SSRPass extends Pass {
  /**
   * @param {THREE.PerspectiveCamera} camera the scene camera (its projection is jittered under TAA, like the depth)
   * @param {import('../CameraHistory.js').CameraHistory|null} history uFrame for the jittered march start (null: frame 0)
   * @param {import('../GBuffer.js').GBuffer|null} gbuffer normal + roughness + metalness / flags attachment
   * @param {SSROptions} [o] options, defaults SSR_DEFAULTS (unknown keys are ignored: PostFX validates names)
   */
  constructor(camera, history, gbuffer, o = {}) {
    super();
    this.needsSwap = false;
    this.camera = camera;
    /** @type {import('../CameraHistory.js').CameraHistory|null} */
    this.history = history ?? null;
    /** @type {import('../GBuffer.js').GBuffer|null} */
    this.gbuffer = gbuffer ?? null;
    const opt = { ...SSR_DEFAULTS };
    for (const k in o) if (o[k] !== undefined) opt[k] = o[k]; // undefined = the default
    this._scale = checkRange(opt.scale, 'scale', 0, 1, true);
    this._maxPixels = checkRange(opt.maxPixels ?? Infinity, 'maxPixels', 1, Infinity);
    this._steps = checkInt(opt.steps, 'steps', 1, 128);
    this._refine = checkInt(opt.refine, 'refine', 0, 16);
    this._maxDistance = checkRange(opt.maxDistance, 'maxDistance', 0, 1e5, true);
    this._thickness = checkRange(opt.thickness, 'thickness', 0, 1e5, true);
    this._thicknessScale = checkRange(opt.thicknessScale, 'thicknessScale', 0, 1);
    this._maxRoughness = checkRange(opt.maxRoughness, 'maxRoughness', 0, 1, true);
    this._edgeFade = checkRange(opt.edgeFade, 'edgeFade', 0, 0.5);
    this._jitter = !!opt.jitter;
    this._metalTint = !!opt.metalTint;
    this._source = checkSource(opt.source);
    this._w = 1; this._h = 1; // drawing-buffer size of the last setSize
    this._rt = new THREE.WebGLRenderTarget(1, 1, {
      type: THREE.HalfFloatType, format: THREE.RGBAFormat, depthBuffer: false, stencilBuffer: false,
      minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter, generateMipmaps: false,
    });
    this._rt.texture.name = 'SSR';
    this.material = new THREE.ShaderMaterial({
      name: 'PostFXSSR',
      defines: {},
      uniforms: {
        tColor: { value: null }, tDepth: { value: null }, tNormal: { value: null },
        uProj: { value: new THREE.Matrix4() }, uProjInv: { value: new THREE.Matrix4() },
        uRes: { value: new THREE.Vector2(1, 1) },
        uNear: { value: 0.1 }, uFar: { value: 1000 },
        uMaxDistance: { value: this._maxDistance }, uThickness: { value: this._thickness }, uThicknessScale: { value: this._thicknessScale },
        uMaxRoughness: { value: this._maxRoughness }, uEdgeFade: { value: edgeFadeUniform(this._edgeFade) },
        uFrame: { value: 0 },
      },
      vertexShader: SSRShader.vertexShader,
      fragmentShader: SSRShader.fragmentShader,
      blending: THREE.NoBlending, depthTest: false, depthWrite: false,
    });
    this._syncDefines();
    this._quad = new FullScreenQuad(this.material);
    this._clear = new THREE.Color();
  }

  /** The reflections of the last render: RGBA16F, premultiplied (rgb = radiance * weight, a = weight). */
  get texture() { return this._rt.texture; }

  /** Output size as a fraction of the drawing buffer, (0, 1] (resizes at once). */
  get scale() { return this._scale; }
  set scale(v) { this._scale = checkRange(v, 'scale', 0, 1, true); this.setSize(this._w, this._h); }
  /** Pixel budget of the output (Infinity / null: none; resizes at once). */
  get maxPixels() { return this._maxPixels; }
  set maxPixels(v) { this._maxPixels = checkRange(v ?? Infinity, 'maxPixels', 1, Infinity); this.setSize(this._w, this._h); }
  /** March samples per ray, integer 1..128 (define SSR_STEPS). */
  get steps() { return this._steps; }
  set steps(v) { this._steps = checkInt(v, 'steps', 1, 128); this._syncDefines(); }
  /** Bisection steps after a hit, integer 0..16 (define SSR_REFINE). */
  get refine() { return this._refine; }
  set refine(v) { this._refine = checkInt(v, 'refine', 0, 16); this._syncDefines(); }
  // The uniform options go to their uniforms here, not in render(): writing a double into a `{ value }` uniform boxes
  // a new heap number in V8 (the shared { value } shape holds every type), so render() writes only what changed.
  /** Ray length in view space (m). */
  get maxDistance() { return this._maxDistance; }
  set maxDistance(v) { this.material.uniforms.uMaxDistance.value = this._maxDistance = checkRange(v, 'maxDistance', 0, 1e5, true); }
  /** A sample hits when the ray is behind the scene depth by less than this (m). */
  get thickness() { return this._thickness; }
  set thickness(v) { this.material.uniforms.uThickness.value = this._thickness = checkRange(v, 'thickness', 0, 1e5, true); }
  /** Band growth per metre of the surface's depth (m / m): band = thickness + depth * thicknessScale. */
  get thicknessScale() { return this._thicknessScale; }
  set thicknessScale(v) { this.material.uniforms.uThicknessScale.value = this._thicknessScale = checkRange(v, 'thicknessScale', 0, 1); }
  /** Weight 0 at and above this roughness, fading in over 0.2 below it. */
  get maxRoughness() { return this._maxRoughness; }
  set maxRoughness(v) { this.material.uniforms.uMaxRoughness.value = this._maxRoughness = checkRange(v, 'maxRoughness', 0, 1, true); }
  /** Fraction of the screen over which hits fade out toward the borders (0: no fade). */
  get edgeFade() { return this._edgeFade; }
  set edgeFade(v) { this._edgeFade = checkRange(v, 'edgeFade', 0, 0.5); this.material.uniforms.uEdgeFade.value = edgeFadeUniform(this._edgeFade); }
  /** Per-pixel, per-frame march start (define SSR_JITTER). Off by default: TAA does not integrate it (see the header). */
  get jitter() { return this._jitter; }
  set jitter(v) { this._jitter = !!v; this._syncDefines(); }
  /** Tint reflections on metals by the pixel's own normalised colour (define SSR_METAL_TINT). */
  get metalTint() { return this._metalTint; }
  set metalTint(v) { this._metalTint = !!v; this._syncDefines(); }
  /** Colour the reflections sample: 'current' (this frame's scene colour); 'history' is reserved. */
  get source() { return this._source; }
  set source(v) { this._source = checkSource(v); }

  /** No temporal state of its own (the jitter follows history.uniforms.uFrame): kept for symmetry with TAAPass. */
  reset() {}

  /**
   * @param {number} width drawing-buffer width (px)
   * @param {number} height drawing-buffer height (px)
   */
  setSize(width, height) {
    this._w = width; this._h = height;
    let w = Math.max(1, Math.round(width * this._scale)), h = Math.max(1, Math.round(height * this._scale));
    if (w * h > this._maxPixels) { // over the budget: the largest uniform scale that fits, floored (as AOPass)
      const s = Math.sqrt(this._maxPixels / (width * height));
      w = Math.max(1, Math.floor(width * s)); h = Math.max(1, Math.floor(height * s));
    }
    this._rt.setSize(w, h);
    this.material.uniforms.uRes.value.set(Math.max(1, width), Math.max(1, height));
  }

  render(renderer, writeBuffer, readBuffer) {
    const gb = this.gbuffer, depth = readBuffer.depthTexture;
    if (gb === null || !depth) { // nothing to trace: an empty (no-op) composite
      renderer.getClearColor(this._clear);
      const ca = renderer.getClearAlpha();
      renderer.setRenderTarget(this._rt); // before setClearColor: the clear colour is converted for the bound target
      renderer.setClearColor(_black, 0);
      renderer.clear(true, false, false);
      renderer.setClearColor(this._clear, ca);
      return;
    }
    const u = this.material.uniforms, cam = this.camera, h = this.history;
    u.tColor.value = readBuffer.texture; // source 'current'
    u.tDepth.value = depth;
    u.tNormal.value = gb.normal;
    u.uProj.value.copy(cam.projectionMatrix);
    u.uProjInv.value.copy(cam.projectionMatrixInverse);
    if (u.uNear.value !== cam.near) u.uNear.value = cam.near; // only on change: no boxed double per frame
    if (u.uFar.value !== cam.far) u.uFar.value = cam.far;
    u.uFrame.value = h !== null ? h.uniforms.uFrame.value : 0;
    renderer.setRenderTarget(this._rt);
    this._quad.render(renderer);
  }

  dispose() {
    this._rt.dispose();
    this.material.dispose();
    this._quad.dispose();
  }

  _syncDefines() {
    const d = this.material.defines;
    let changed = false;
    if (d.SSR_STEPS !== this._steps) { d.SSR_STEPS = this._steps; changed = true; }
    if (d.SSR_REFINE !== this._refine) { d.SSR_REFINE = this._refine; changed = true; }
    if (this._jitter !== ('SSR_JITTER' in d)) { if (this._jitter) d.SSR_JITTER = 1; else delete d.SSR_JITTER; changed = true; }
    if (this._metalTint !== ('SSR_METAL_TINT' in d)) { if (this._metalTint) d.SSR_METAL_TINT = 1; else delete d.SSR_METAL_TINT; changed = true; }
    if (changed) this.material.needsUpdate = true;
  }
}

const _black = new THREE.Color(0, 0, 0);

// smoothstep(0, 0, x) is undefined in GLSL: edgeFade 0 uploads 1e-4 (a fade narrower than any pixel)
const edgeFadeUniform = (f) => Math.max(f, 1e-4);

const show = (v) => (typeof v === 'string' ? `'${v}'` : String(v));

/** @param {*} v @param {string} what @param {number} lo @param {number} hi @param {boolean} [loOpen] @returns {number} */
function checkRange(v, what, lo, hi, loOpen = false) {
  if (typeof v !== 'number' || Number.isNaN(v) || (loOpen ? v <= lo : v < lo) || v > hi) {
    throw new Error(`[SSRPass] ${what}: expected a number in ${loOpen ? '(' : '['}${lo}, ${hi}], got ${show(v)}`);
  }
  return v;
}

/** @param {*} v @param {string} what @param {number} lo @param {number} hi @returns {number} */
function checkInt(v, what, lo, hi) {
  if (!Number.isInteger(v) || v < lo || v > hi) throw new Error(`[SSRPass] ${what}: expected an integer in [${lo}, ${hi}], got ${show(v)}`);
  return v;
}

/** @param {*} v @returns {string} */
function checkSource(v) {
  if (v === 'history') throw new Error("[SSRPass] source: 'history' is reserved (not implemented yet); use 'current'");
  if (!SSR_SOURCES.includes(v)) throw new Error(`[SSRPass] source: unknown value ${show(v)} ('current')`);
  return v;
}
