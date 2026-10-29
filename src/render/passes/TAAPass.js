// Temporal anti-aliasing resolve, on the G-buffer path (PostFX order: Scene (MRT) -> Exposure -> AO -> TAA -> Bloom
// -> Final; Bloom and Final read `texture`). The scene pass renders with the JITTERED camera (CameraHistory.begin);
// the unjittered projection is not used to draw anything: this pass removes the jitter. Per pixel:
//   1. Current colour, reconstructed at the unjittered pixel centre: the 3x3 neighbourhood of the jittered frame, each
//      texel weighted by a Blackman-Harris window, exp(-2.29 d^2), of its distance d = offset + uJitter to that centre
//      (the unjittered point at pixel p was drawn at p - uJitter: the sample offset by -uJitter). The 9 weights depend
//      only on the jitter, so they are computed on the CPU once per frame. Neighbourhood maths run in YCoCg with Karis'
//      luma tonemap c / (1 + Y), so a bright texel does not dominate the filter, the clip or the blend.
//   2. Motion of the front-most surface among the pixel and its 4 diagonals (closest depth: a moving silhouette keeps
//      its own motion). G-buffer flags bit0 (wroteGBuffer): the velocity attachment (uvCur - uvPrev, unjittered
//      matrices). Otherwise (sky, materials without the G-buffer output): reprojected from the depth with the camera
//      alone, uReproj = prevViewProj * inverse(viewProj) (CameraHistory, composed in doubles on the CPU).
//   3. History at uv - velocity: Catmull-Rom (5 bilinear taps, 'catmull-rom') or one bilinear tap ('bilinear'), then
//      constrained to the current neighbourhood: 'variance' (box mean +- gamma * sigma in YCoCg, gamma 1, the history
//      clipped toward the mean along its offset) or 'minmax' (clamped to the 3x3 min / max).
//   4. Blend 0 (current frame only) on the first frame after reset() / setSize() / CameraHistory.reset() / any frame
//      this pass skipped, when the history position is off screen, the surface has noHistory (flags bit3) or, on the
//      depth path, was behind the previous camera (previous clip w <= 0), or it is disoccluded. Disocclusion
//      ('rgba16f' history only: its alpha holds each pixel's linear depth): no texel of the 2x2 history footprint lies
//      within 10 % + 0.3 m of where the surface was (static world: uReproj) or of where it is now (moving with the
//      camera). Otherwise mix(current, history, blend): blend 0.9 = 10 % new per frame.
//   5. Optional sharpen of the current sample before the blend: current + sharpen * (current - 3x3 mean), against the
//      softening of the reconstruction filter.
// Output: history[cur] of a ping-pong pair (rgb = resolved scene-linear HDR, a = linear depth), `texture`. The first
// frame after a reset outputs the input texel unchanged (NaN / Inf written as black). needsSwap is false: the
// composer's buffers are untouched.
// Options (defaults): blend 0.9, clamp 'variance' | 'minmax', gamma 1, sharpen 0, filter 'catmull-rom' | 'bilinear',
// historyFormat 'rgba16f' | 'r11g11b10' (half the history traffic; no alpha, so no disocclusion test, and the 6 / 5-bit
// mantissas freeze the 10 % update: 0.4-0.7 mean / 255 off at Canyon Run's poses; stochastic rounding traded that for
// per-frame noise, 0.5-1.0 off, so it was dropped), and the cost knobs below. All are live properties.
// Cost (Canyon Run, 3024x1964, Apple M4, whole-frame deltas when each part is removed): about 2.3 ms above a pass that
// only copies the colour into the RGBA16F target; G-buffer flags + velocity reads 0.75, the Catmull-Rom taps beyond a
// bilinear one 0.5, the colour 3x3 beyond its centre 0.5, the disocclusion taps 0.34, the depth dilation 0.1: the
// pass is bound by full-screen reads (each extra full-resolution texture ~0.3 ms), not by its arithmetic (a mediump /
// restructured shader measured the same). Knobs, defaults = the output above (all measured meanAbs against it at the
// four Canyon Run poses, after the harness's 32-frame convergence):
//   velocityFirst (false): read the velocity first and the flags only where it is exactly 0 (a writer with a non-zero
//     velocity has bit0 and not noHistory). Bit-identical (meanAbs 0). Skips the normal-texture read wherever
//     something moves on screen; a static camera stores velocity 0, so a static view still reads it.
//   reject 'quad' | 'early' | 'single': 'early' tests the history texel nearest the reprojected position first and the
//     2x2 only when it fails (the same rule; meanAbs <= 0.0022, borderline compares under different float
//     contraction), 'single' tests that texel only (meanAbs 0.0014-0.0050; more rejection on moving silhouettes).
//   snap (false): where the history position lies on a texel centre (to 1/512 px: static camera and surface, e.g. the
//     car in the chase view) one fetch replaces the 5 Catmull-Rom taps, which there are that texel (meanAbs <= 0.0027).
//     A static benchmark pose is all such pixels, so it shows the most this saves; a moving view saves little.
//   All three together: meanAbs <= 0.0027 (with 'single' <= 0.0053).
// SSR (0.8.0, PostFX sets `ssr` to its SSRPass, or null): with an enabled SSR pass the define TAA_SSR composites the
// reflection buffer (premultiplied: engineSsrComposite(c, ssr) = c * (1 - ssr.a) + ssr.rgb) into the current frame
// before the resolve: one tSSR fetch at the pixel, applied to the centre and to every neighbourhood tap before the
// YCoCg conversion, so the variance clip box contains the reflection (clipped toward a neighbourhood without it, a
// history with reflections would flicker), and to the fresh-frame output. Without it the shader text is 0.7.0's.
// API: render(renderer, writeBuffer, readBuffer), setSize(w, h), reset(), texture, ssr, dispose(); taaFilterWeights(jx, jy).
// Perspective cameras and a standard depth buffer (not reversedDepthBuffer / logarithmicDepthBuffer): the depth
// reprojection and disocclusion test linearise depth with near / far. Output is NaN / Inf-safe (black).
import * as THREE from 'three';
import { Pass, FullScreenQuad } from 'three/addons/postprocessing/Pass.js';
import { GBUFFER_GLSL_DECODE } from '../GBuffer.js';
import { SSR_COMPOSITE_GLSL } from './SSRPass.js';

export const TAA_CLAMPS = ['variance', 'minmax'];
export const TAA_FILTERS = ['catmull-rom', 'bilinear'];
export const TAA_HISTORY_FORMATS = ['rgba16f', 'r11g11b10'];
export const TAA_REJECTS = ['quad', 'early', 'single'];

/**
 * Reconstruction weights of the 3x3 neighbourhood (row-major from (-1, -1)) for a jitter of (jx, jy) pixels: a
 * Blackman-Harris window, exp(-2.29 d^2), of each texel's distance d = offset + jitter to the unjittered pixel centre,
 * normalised to sum 1. Writes into `out` (9 numbers) and returns it.
 */
export function taaFilterWeights(jx, jy, out = new Float32Array(9)) {
  let s = 0;
  for (let i = 0; i < 9; i++) {
    const dx = (i % 3) - 1 + jx, dy = ((i / 3) | 0) - 1 + jy;
    s += (out[i] = Math.exp(-2.29 * (dx * dx + dy * dy)));
  }
  for (let i = 0; i < 9; i++) out[i] /= s;
  return out;
}

/** @internal The TAA shader sources (test/postfx-options.mjs pins the fragment text without its TAA_SSR blocks). */
export const TAA_SHADER = {
  vertexShader: /* glsl */`
    varying vec2 vUv;
    void main() { vUv = uv; gl_Position = vec4( position.xy, 0.0, 1.0 ); }`,
  fragmentShader: /* glsl */`
    uniform sampler2D tColor;
    uniform highp sampler2D tDepth;
    uniform sampler2D tVelocity;
    uniform sampler2D tNormal;
    uniform sampler2D tHistory;
    uniform mat4 uReproj;
    uniform vec4 uInvVPw; // 4th row of inverse(viewProj): homogeneous w of a reconstructed world point
    uniform vec2 uJitter; // pixels
    uniform float uWeights[ 9 ];
    uniform float uBlend, uGamma, uSharpen, uFresh, uNear, uFar;
    uniform vec2 uRes;
    varying vec2 vUv;

    ${GBUFFER_GLSL_DECODE}

    vec3 toYCoCg( vec3 c ) { return vec3( dot( c, vec3( 0.25, 0.5, 0.25 ) ), dot( c, vec3( 0.5, 0.0, - 0.5 ) ), dot( c, vec3( - 0.25, 0.5, - 0.25 ) ) ); }
    vec3 fromYCoCg( vec3 c ) { return vec3( c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z ); }
    // Karis: scale Y, Co, Cg by 1 / (1 + Y) (and back), so HDR highlights do not flicker through the filter and clip
    vec3 tm( vec3 c ) { vec3 y = toYCoCg( c ); return y / ( 1.0 + y.x ); }
    vec3 itm( vec3 y ) { return fromYCoCg( y / max( 1.0 - y.x, 1e-3 ) ); }
    float linearDepth( float d ) { return uNear * uFar / ( uFar - ( uFar - uNear ) * d ); } // standard (not reversed) depth
    // a NaN / Inf from the scene would stay in the history for good: black instead (comparisons with NaN fail)
    vec3 finite( vec3 o ) { return all( lessThan( abs( o ), vec3( 65000.0 ) ) ) ? max( o, 0.0 ) : vec3( 0.0 ); }
    #ifdef TAA_SSR
    uniform sampler2D tSSR;
    ${SSR_COMPOSITE_GLSL}
    #endif

    vec3 historyAt( vec2 uv ) {
    #ifdef TAA_SNAP
      // history position on a texel centre (to 1/512 px: static camera and surface): Catmull-Rom there is that texel
      // (weights 0, 1, 0, 0), so one fetch replaces the 5 taps (pixels in motion take the full filter)
      vec2 sp = uv * uRes - 0.5, sn = floor( sp + 0.5 );
      if ( all( lessThan( abs( sp - sn ), vec2( 1.0 / 512.0 ) ) ) ) return texelFetch( tHistory, clamp( ivec2( sn ), ivec2( 0 ), ivec2( uRes ) - 1 ), 0 ).rgb;
    #endif
    #ifdef TAA_CATMULL_ROM
      // 5-tap Catmull-Rom from bilinear taps (the 4 corner taps of the 9-tap version dropped, weights renormalised)
      vec2 rcp = 1.0 / uRes, p = uv * uRes, t1 = floor( p - 0.5 ) + 0.5, f = p - t1, f2 = f * f, f3 = f2 * f;
      vec2 w0 = f2 - 0.5 * ( f3 + f ), w1 = 1.5 * f3 - 2.5 * f2 + 1.0, w3 = 0.5 * ( f3 - f2 ), w2 = 1.0 - w0 - w1 - w3;
      vec2 w12 = w1 + w2, t0 = ( t1 - 1.0 ) * rcp, t3 = ( t1 + 2.0 ) * rcp, t12 = ( t1 + w2 / w12 ) * rcp;
      float a = w12.x * w0.y, b = w0.x * w12.y, c = w12.x * w12.y, d = w3.x * w12.y, e = w12.x * w3.y;
      vec3 s = texture2D( tHistory, vec2( t12.x, t0.y ) ).rgb * a + texture2D( tHistory, vec2( t0.x, t12.y ) ).rgb * b
             + texture2D( tHistory, t12 ).rgb * c + texture2D( tHistory, vec2( t3.x, t12.y ) ).rgb * d
             + texture2D( tHistory, vec2( t12.x, t3.y ) ).rgb * e;
      return max( s / ( a + b + c + d + e ), 0.0 );
    #else
      return texture2D( tHistory, uv ).rgb;
    #endif
    }

    #define TAA_CLOSEST( X, Y ) { ivec2 q = clamp( ip + ivec2( X, Y ), ivec2( 0 ), mx ); float e = texelFetch( tDepth, q, 0 ).x; if ( e < d ) { d = e; bp = q; } }
    #define TAA_TAP( X, Y, I ) { vec3 c = tm( texelFetch( tColor, clamp( ip + ivec2( X, Y ), ivec2( 0 ), mx ), 0 ).rgb ); cur += c * uWeights[ I ]; m1 += c; m2 += c * c; lo = min( lo, c ); hi = max( hi, c ); }
    #ifdef TAA_SSR
    // every tap with this pixel's reflection composited (ssrC, fetched once in main)
    #undef TAA_TAP
    #define TAA_TAP( X, Y, I ) { vec3 c = tm( engineSsrComposite( texelFetch( tColor, clamp( ip + ivec2( X, Y ), ivec2( 0 ), mx ), 0 ).rgb, ssrC ) ); cur += c * uWeights[ I ]; m1 += c; m2 += c * c; lo = min( lo, c ); hi = max( hi, c ); }
    #endif

    void main() {
      ivec2 ip = ivec2( gl_FragCoord.xy ), mx = ivec2( uRes ) - 1;
      float d = texelFetch( tDepth, ip, 0 ).x;
      ivec2 bp = ip;
      TAA_CLOSEST( - 1, - 1 ) TAA_CLOSEST( 1, - 1 ) TAA_CLOSEST( - 1, 1 ) TAA_CLOSEST( 1, 1 )
      float z = linearDepth( d );
    #ifdef TAA_SSR
      vec4 ssrC = texture2D( tSSR, vUv );
      if ( uFresh > 0.5 ) { gl_FragColor = vec4( finite( engineSsrComposite( texelFetch( tColor, ip, 0 ).rgb, ssrC ) ), z ); return; }
    #endif
      if ( uFresh > 0.5 ) { gl_FragColor = vec4( finite( texelFetch( tColor, ip, 0 ).rgb ), z ); return; }

      // current frame at the unjittered pixel centre + neighbourhood moments (tonemapped YCoCg)
      vec3 cur = vec3( 0.0 ), m1 = vec3( 0.0 ), m2 = vec3( 0.0 ), lo = vec3( 1e4 ), hi = vec3( - 1e4 );
      TAA_TAP( - 1, - 1, 0 ) TAA_TAP( 0, - 1, 1 ) TAA_TAP( 1, - 1, 2 )
      TAA_TAP( - 1, 0, 3 ) TAA_TAP( 0, 0, 4 ) TAA_TAP( 1, 0, 5 )
      TAA_TAP( - 1, 1, 6 ) TAA_TAP( 0, 1, 7 ) TAA_TAP( 1, 1, 8 )
      vec3 mean = m1 / 9.0;
      if ( uSharpen > 0.0 ) { cur += ( cur - mean ) * uSharpen; cur.x = clamp( cur.x, 0.0, 0.999 ); }

      // motion of the front-most surface: G-buffer velocity, or the camera reprojection of its depth
      vec2 uvJ = ( vec2( bp ) + 0.5 + uJitter ) / uRes; // where that texel's surface is, unjittered
      vec4 ndc = vec4( uvJ * 2.0 - 1.0, d * 2.0 - 1.0, 1.0 );
      vec4 pc = uReproj * ndc; // previous clip position if the surface is static
    #ifdef TAA_VELOCITY_FIRST
      // a G-buffer writer that stores a non-zero velocity has bit0 set and not noHistory (noHistory writes 0; other
      // materials write 0 or leave both attachments as they were): the flags are read only where the velocity is 0
      // (static camera + surface, noHistory, nothing written). Same result, one full-screen read less in motion.
      vec2 vel = texelFetch( tVelocity, bp, 0 ).rg;
      int flags = vel.x != 0.0 || vel.y != 0.0 ? ENGINE_GB_WROTE : engineGbFlags( texelFetch( tNormal, bp, 0 ).a );
      bool wrote = ( flags & ENGINE_GB_WROTE ) != 0;
      bool behind = ! wrote && pc.w <= 0.0;
      if ( ! wrote ) vel = behind ? vec2( 0.0 ) : uvJ - ( pc.xy / pc.w * 0.5 + 0.5 );
    #else
      int flags = engineGbFlags( texelFetch( tNormal, bp, 0 ).a );
      bool wrote = ( flags & ENGINE_GB_WROTE ) != 0;
      // depth path: a point behind the previous camera (pc.w <= 0) projects to garbage, often inside [0, 1]: no
      // history, as the G-buffer's noHistory flag does for its writers
      bool behind = ! wrote && pc.w <= 0.0;
      vec2 vel = wrote ? texelFetch( tVelocity, bp, 0 ).rg : behind ? vec2( 0.0 ) : uvJ - ( pc.xy / pc.w * 0.5 + 0.5 );
    #endif
      vec2 huv = vUv - vel;
      float a = uBlend;
      if ( behind || ( flags & ENGINE_GB_NO_HISTORY ) != 0 || huv.x < 0.0 || huv.y < 0.0 || huv.x > 1.0 || huv.y > 1.0 ) a = 0.0;
    #ifdef TAA_DEPTH_REJECT
      float zPrev = pc.w / dot( uInvVPw, ndc ); // view depth the surface had, static
     #if defined( TAA_REJECT_EARLY ) || defined( TAA_REJECT_SINGLE )
      // the texel nearest the history position first (one of the 2x2): it decides almost every pixel
      float tol = 0.1 * z + 0.3, hn = texelFetch( tHistory, clamp( ivec2( huv * uRes ), ivec2( 0 ), mx ), 0 ).a;
      if ( min( abs( hn - zPrev ), abs( hn - z ) ) > tol ) {
      #ifdef TAA_REJECT_EARLY
        ivec2 h0 = clamp( ivec2( floor( huv * uRes - 0.5 ) ), ivec2( 0 ), mx ), h1 = min( h0 + 1, mx );
        vec4 hz = vec4( texelFetch( tHistory, h0, 0 ).a, texelFetch( tHistory, ivec2( h1.x, h0.y ), 0 ).a,
                        texelFetch( tHistory, ivec2( h0.x, h1.y ), 0 ).a, texelFetch( tHistory, h1, 0 ).a );
        vec4 e = min( abs( hz - zPrev ), abs( hz - z ) );
        if ( min( min( e.x, e.y ), min( e.z, e.w ) ) > tol ) a = 0.0;
      #else
        a = 0.0;
      #endif
      }
     #else
      ivec2 h0 = clamp( ivec2( floor( huv * uRes - 0.5 ) ), ivec2( 0 ), mx ), h1 = min( h0 + 1, mx );
      vec4 hz = vec4( texelFetch( tHistory, h0, 0 ).a, texelFetch( tHistory, ivec2( h1.x, h0.y ), 0 ).a,
                      texelFetch( tHistory, ivec2( h0.x, h1.y ), 0 ).a, texelFetch( tHistory, h1, 0 ).a );
      vec4 e = min( abs( hz - zPrev ), abs( hz - z ) );
      if ( min( min( e.x, e.y ), min( e.z, e.w ) ) > 0.1 * z + 0.3 ) a = 0.0;
     #endif
    #endif

      vec3 h = tm( historyAt( huv ) );
    #ifdef TAA_MINMAX
      h = clamp( h, lo, hi );
    #else
      vec3 ext = max( sqrt( max( m2 / 9.0 - mean * mean, 0.0 ) ) * uGamma, vec3( 1e-5 ) ), v = h - mean, k = abs( v / ext );
      float m = max( k.x, max( k.y, k.z ) );
      if ( m > 1.0 ) h = mean + v / m; // clip toward the mean
    #endif
      gl_FragColor = vec4( finite( itm( mix( cur, h, a ) ) ), z );
    }`,
};

export class TAAPass extends Pass {
  /**
   * @param {THREE.PerspectiveCamera} camera
   * @param {import('../CameraHistory.js').CameraHistory} history
   * @param {import('../GBuffer.js').GBuffer} gbuffer velocity + normal (flags) attachments
   * @param {object} [o] blend (0.9), clamp ('variance'), gamma (1), sharpen (0), filter ('catmull-rom'),
   *   historyFormat ('rgba16f'), reject ('quad'), velocityFirst (false), snap (false)
   */
  constructor(camera, history, gbuffer, o = {}) {
    super();
    this.needsSwap = false;
    this.camera = camera;
    this.history = history;
    this.gbuffer = gbuffer;
    /** History weight per frame (0 = no accumulation). */
    this.blend = o.blend ?? 0.9;
    /** Variance clip box half-size in standard deviations. */
    this.gamma = o.gamma ?? 1;
    /** Sharpening of the current sample (0 = off). */
    this.sharpen = o.sharpen ?? 0;
    this._clamp = check(o.clamp ?? 'variance', TAA_CLAMPS, 'clamp');
    this._filter = check(o.filter ?? 'catmull-rom', TAA_FILTERS, 'filter');
    this._format = check(o.historyFormat ?? 'rgba16f', TAA_HISTORY_FORMATS, 'historyFormat');
    this._reject = check(o.reject ?? 'quad', TAA_REJECTS, 'reject');
    this._velocityFirst = !!(o.velocityFirst ?? false);
    this._snap = !!(o.snap ?? false);
    /** @type {{ enabled: boolean, texture: THREE.Texture } | null} SSR pass to composite before the resolve (PostFX sets it). */
    this.ssr = null;
    this._ssrOn = false; // TAA_SSR defined
    this._w = 1; this._h = 1;
    this._rt = [null, null];
    this._makeTargets();
    this._cur = 0;
    this._reset = true;
    this._lastFrame = -2;
    const hu = history.uniforms;
    this.material = new THREE.ShaderMaterial({
      name: 'PostFXTAA',
      defines: {},
      uniforms: {
        tColor: { value: null }, tDepth: { value: null }, tVelocity: { value: null }, tNormal: { value: null }, tHistory: { value: null },
        tSSR: { value: null }, // TAA_SSR only
        uReproj: hu.uReproj, uJitter: hu.uJitter, // CameraHistory's own uniform objects
        uInvVPw: { value: new THREE.Vector4() },
        uWeights: { value: new Float32Array(9) },
        uBlend: { value: 0.9 }, uGamma: { value: 1 }, uSharpen: { value: 0 }, uFresh: { value: 1 },
        uNear: { value: 0.1 }, uFar: { value: 1000 },
        uRes: { value: new THREE.Vector2(1, 1) },
      },
      vertexShader: TAA_SHADER.vertexShader,
      fragmentShader: TAA_SHADER.fragmentShader,
      blending: THREE.NoBlending, depthTest: false, depthWrite: false,
    });
    this._syncDefines();
    this._quad = new FullScreenQuad(this.material);
  }

  /** The resolved colour of the last render (scene-linear HDR; alpha = linear depth with the 'rgba16f' history). */
  get texture() { return this._rt[this._cur].texture; }

  /** 'variance' | 'minmax' */
  get clamp() { return this._clamp; }
  set clamp(v) { this._clamp = check(v, TAA_CLAMPS, 'clamp'); this._syncDefines(); }
  /** History sampling: 'catmull-rom' | 'bilinear' */
  get filter() { return this._filter; }
  set filter(v) { this._filter = check(v, TAA_FILTERS, 'filter'); this._syncDefines(); }
  /** 'rgba16f' | 'r11g11b10' (re-creates the history targets and resets the history) */
  get historyFormat() { return this._format; }
  set historyFormat(v) {
    v = check(v, TAA_HISTORY_FORMATS, 'historyFormat');
    if (v === this._format) return;
    this._format = v;
    for (const rt of this._rt) rt.dispose();
    this._makeTargets();
    this._syncDefines();
    this._reset = true;
  }

  /** Disocclusion test taps: 'quad' (the 2x2 history footprint) | 'early' (same result) | 'single' (nearest texel). */
  get reject() { return this._reject; }
  set reject(v) { this._reject = check(v, TAA_REJECTS, 'reject'); this._syncDefines(); }
  /** Read the G-buffer flags only where the velocity is 0 (same result). */
  get velocityFirst() { return this._velocityFirst; }
  set velocityFirst(v) { this._velocityFirst = !!v; this._syncDefines(); }
  /** One history fetch instead of the Catmull-Rom taps where the history position is on a texel centre. */
  get snap() { return this._snap; }
  set snap(v) { this._snap = !!v; this._syncDefines(); }

  /** The next frame outputs its input and starts a new history (teleport, camera cut). */
  reset() { this._reset = true; }

  setSize(width, height) {
    this._w = width; this._h = height;
    for (const rt of this._rt) rt.setSize(width, height);
    this.material.uniforms.uRes.value.set(width, height);
    this._reset = true;
  }

  render(renderer, writeBuffer, readBuffer) {
    const u = this.material.uniforms, h = this.history, gb = this.gbuffer, cam = this.camera;
    const f = h.uniforms.uFrame.value, s = this.ssr, ssrOn = s !== null && s.enabled;
    if (ssrOn !== this._ssrOn) { this._ssrOn = ssrOn; this._syncDefines(); }
    const fresh = this._reset || h.fresh || f !== this._lastFrame + 1; // a skipped frame leaves a stale history
    this._reset = false;
    this._lastFrame = f;
    u.tColor.value = readBuffer.texture;
    u.tDepth.value = readBuffer.depthTexture;
    u.tVelocity.value = gb.velocity;
    u.tNormal.value = gb.normal;
    u.tHistory.value = this._rt[this._cur].texture;
    u.tSSR.value = ssrOn ? s.texture : null;
    this._cur ^= 1;
    u.uFresh.value = fresh ? 1 : 0;
    u.uBlend.value = this.blend;
    u.uGamma.value = this.gamma;
    u.uSharpen.value = this.sharpen;
    u.uNear.value = cam.near; u.uFar.value = cam.far;
    const iv = h.uniforms.uInvViewProj.value.elements;
    u.uInvVPw.value.set(iv[3], iv[7], iv[11], iv[15]);
    const j = h.uniforms.uJitter.value;
    taaFilterWeights(j.x, j.y, u.uWeights.value);
    renderer.setRenderTarget(this._rt[this._cur]);
    this._quad.render(renderer);
  }

  dispose() {
    for (const rt of this._rt) rt.dispose();
    this.material.dispose();
    this._quad.dispose();
  }

  _makeTargets() {
    const fmt = this._format === 'r11g11b10' ? { format: THREE.RGBFormat, type: THREE.UnsignedInt101111Type } : { type: THREE.HalfFloatType };
    const opt = { ...fmt, depthBuffer: false, stencilBuffer: false, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter, generateMipmaps: false };
    for (let i = 0; i < 2; i++) {
      const rt = this._rt[i] = new THREE.WebGLRenderTarget(this._w, this._h, opt);
      rt.texture.name = `TAA.history${i}`;
    }
  }

  _syncDefines() {
    const d = this.material.defines, want = {
      TAA_CATMULL_ROM: this._filter === 'catmull-rom',
      TAA_MINMAX: this._clamp === 'minmax',
      TAA_DEPTH_REJECT: this._format === 'rgba16f',
      TAA_REJECT_EARLY: this._reject === 'early',
      TAA_REJECT_SINGLE: this._reject === 'single',
      TAA_VELOCITY_FIRST: this._velocityFirst,
      TAA_SNAP: this._snap,
      TAA_SSR: this._ssrOn,
    };
    let changed = false;
    for (const k in want) {
      if (want[k] === (k in d)) continue;
      if (want[k]) d[k] = 1; else delete d[k];
      changed = true;
    }
    if (changed) this.material.needsUpdate = true;
  }
}

function check(v, allowed, what) {
  if (!allowed.includes(v)) throw new Error(`[TAAPass] unknown ${what} '${v}' (${allowed.map((a) => `'${a}'`).join(' | ')})`);
  return v;
}
