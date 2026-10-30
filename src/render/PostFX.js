// Post-processing pipeline (EffectComposer from three/addons). Two scene-target modes:
//
// Forward (default, `gbuffer: false`; the 0.6.0 pipeline, bit-identical with every new option at its default):
//   ScenePass   -> scene into a 4x MSAA HDR target (R11G11B10F, RGBA16F fallback) with a resolved depth texture
//   ExposurePass-> 64x64 log-luminance pyramid -> adapted 1x1 luminance (eye adaptation, GPU only)
//   AOPass      -> GTAO from that depth at half resolution + bilateral blur (no second scene render)
//   BloomPass   -> UnrealBloom at quarter resolution with an HDR soft-knee bright pass (not blended here)
//   FinalPass   -> one full-resolution shader: AO (bilinear or depth-aware upsample), height/distance fog with
//                  sky-coloured inscatter, bloom add, exposure + ACES filmic (same curve as OutputPass), colour
//                  grading (contrast, saturation, split toning), vignette, edge chromatic aberration, night grain,
//                  dither, sRGB. Writes the scene depth to the default framebuffer so the lens flare can occlude.
// G-buffer (`gbuffer: true`): Scene (MRT: colour, normal + flags, velocity, depth; single-sampled: MSAA on an MRT
//   target would resolve every attachment) -> Exposure -> AO -> TAA -> Bloom -> Final. Bloom and Final read
//   `colorSource` (TAAPass.texture while TAA is on, else the scene colour); Exposure and AO read the scene target.
//   CameraHistory (`history`) publishes the camera matrices for the velocity output and jitters the projection for
//   the frame while TAA is on (restored after the composer, before the flare). The G-buffer target is cleared late
//   too (see ScenePass): its attachments 1-2 are reset by GBuffer.clear right after the target's own clear.
//   Under TAA, AO and fog are still applied after the resolve from this frame's jittered depth; AO computed before
//   TAA is a later step. taa.stableDepth (off by default: +2.3 ms at 3024x1964 for a 0.005 mean change) makes the
//   final pass take the closest depth of the pixel and its diagonals while TAA jitters, so silhouette AO / fog stop
//   alternating with the jitter, at the price of a 1-px rim of foreground fog / AO on the background around objects.
//   TAA and the final pass's depth maths assume a standard depth buffer (no reversedDepthBuffer).
//   The camera's projection is jittered between CameraHistory.begin and end inside render(): call
//   camera.updateProjectionMatrix() outside render().
// Then the Atmosphere's lens-flare overlay. Anti-aliasing (`aa`, see resolveAA): 'auto' = forward: 4x MSAA on the
// scene target by day (alpha-to-coverage foliage, thin bridge cables), FXAA inside the final pass once it is dark or
// the drawing buffer is larger than msaaMaxPixels; G-buffer: TAA. 'msaa' | 'fxaa' fixed ('msaa' falls back to FXAA
// on the G-buffer); 'taa' needs the G-buffer (throws without it). The final pass's FXAA is off under TAA.
// No passes swap (needsSwap false everywhere): EffectComposer's readBuffer is the scene target for the whole frame.
// Without an atmosphere (`atmosphere: null`): no sky cube, so fog inscatter uses the `fogColor` option (linear HDR)
// with `fogParams` densities, or there is no fog; setTimeOfDay(darkness) grades from darkness alone; no flare.
// API: render(dt), setSize(w, h, pixelRatio), setTimeOfDay(darkness), setEnabled(bool), setMSAA(mode, maxPixels),
// setAA(mode), setGBuffer(bool), syncGBuffer(), track(obj) / untrack(obj) / trackPool(pool) / untrackPool(pool)
// (per-object velocity, kept across setGBuffer rebuilds), set(path, value), setAOResolution({ scale, maxPixels,
// upsample }), debug, dispose(); gbuffer (GBuffer | null), history (CameraHistory | null), taaPass (TAAPass | null),
// sceneDepth, colorSource, timings.
// Options (defaults are the Roadster tunings): atmosphere, width/height/pixelRatio, samples, msaa ('auto' | n),
// msaaMaxPixels, lateClear, compactHDR, grade (keyframes by sun elevation, see DEFAULT_GRADE), ao { radius,
// thickness, samples, resolutionScale, maxPixels, upsample, compact, fadeNear, fadeFar } (see AOPass), bloom
// { strength, radius, threshold, resolutionScale }, fog (bool), fogFarFade (fraction of camera.far where the far
// fade starts), fogMinSkyElevation, autoExposure; fogColor (null: no fog without a sky cube), fogParams ({ haze,
// hazeK, hazeBase, mist, mistK }, used without an atmosphere), gbuffer (false), aa ('auto'), taa { blend 0.9, clamp
// 'variance', gamma 1, sharpen 0, filter 'catmull-rom', historyFormat 'rgba16f' (see TAAPass; 'r11g11b10' measured 0.5
// mean worse), stableDepth false (see above), reject 'quad', velocityFirst false, snap false (cost knobs, see
// TAAPass); unknown keys / values throw }, jitter (true; with TAA only),
// jitterPattern ('halton16', see CameraHistory), timing (false: true fills `timings` with per-pass CPU ms of the last
// frame: scene, exposure, ao, taa, bloom, final, flare, total; CPU submission only, and performance.now() is
// coarsened to 0.1 ms without cross-origin isolation).
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { Pass, FullScreenQuad } from 'three/addons/postprocessing/Pass.js';
import { AOPass } from './passes/AOPass.js';
import { BloomPass } from './passes/BloomPass.js';
import { ExposurePass } from './passes/ExposurePass.js';
import { TAAPass, TAA_CLAMPS, TAA_FILTERS, TAA_HISTORY_FORMATS, TAA_REJECTS } from './passes/TAAPass.js';
import { CameraHistory } from './CameraHistory.js';
import { GBuffer, installGBufferChunks } from './GBuffer.js';

const DEG = Math.PI / 180;
const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const sstep = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
const mix = (a, b, t) => a + (b - a) * t;

const FinalShader = {
  uniforms: {
    tColor: { value: null }, tDepth: { value: null }, tAO: { value: null }, tBloom: { value: null }, tSky: { value: null }, tAdapt: { value: null },
    uAE: { value: new THREE.Vector4(0.3, 0.5, 0.7, 2.0) }, // target luminance, strength, min, max multiplier
    uHasAO: { value: 1 }, uHasBloom: { value: 1 }, uHasFog: { value: 1 },
    uProjInv: { value: new THREE.Matrix4() }, uCamWorld: { value: new THREE.Matrix4() }, uCamPos: { value: new THREE.Vector3() },
    uNear: { value: 0.1 }, uFar: { value: 6000 },
    uAO: { value: 0.75 },
    uFogA: { value: new THREE.Vector4(0.0002, 1 / 700, 0, 0.72) }, // haze density, haze falloff, haze base height, far-fade start (x far)
    uFogB: { value: new THREE.Vector4(0.0001, 1 / 40, 0.0, 0.02) }, // mist density, mist falloff, mist base height, min sky-lookup elevation
    uSunDir: { value: new THREE.Vector3(0, 1, 0) }, uSunCol: { value: new THREE.Color(1, 1, 1) }, uSunScatter: { value: 0 },
    uExposure: { value: 0.9 },
    uShadowTint: { value: new THREE.Vector3() }, uHighTint: { value: new THREE.Vector3() }, uWB: { value: new THREE.Vector3(1, 1, 1) },
    uContrast: { value: 0.15 }, uSaturation: { value: 1.05 }, uVignette: { value: 0.3 }, uCA: { value: 0.006 },
    uGrain: { value: 0 }, uTime: { value: 0 }, uRes: { value: new THREE.Vector2(1, 1) },
    uDebug: { value: 0 }, // 1 AO, 2 bloom, 3 fog transmittance
    uFXAA: { value: 0 },
    uFogColor: { value: new THREE.Color() }, // fog inscatter without a sky cube (FOG_SKY undefined), linear HDR
  },
  // The eye-adaptation multiplier is one value for the whole screen: evaluated per vertex of the full-screen triangle
  // and passed flat (same value), instead of a texture read + exp2 + pow in each of ~6 M pixels at Retina size.
  vertexShader: /* glsl */`
    uniform sampler2D tAdapt;
    uniform vec4 uAE;
    varying vec2 vUv;
    flat varying float vAE;
    void main() {
      vUv = uv;
      vAE = clamp( pow( uAE.x / exp2( textureLod( tAdapt, vec2( 0.5 ), 0.0 ).r ), uAE.y ), uAE.z, uAE.w );
      gl_Position = vec4( position.xy, 0.0, 1.0 );
    }`,
  fragmentShader: /* glsl */`
    uniform sampler2D tColor;
    uniform highp sampler2D tDepth;
    uniform sampler2D tAO;
    uniform sampler2D tBloom;
    #ifdef FOG_SKY
    uniform samplerCube tSky;
    #endif
    uniform vec3 uFogColor;
    uniform vec4 uAE;
    uniform float uHasAO, uHasBloom, uHasFog;
    uniform mat4 uProjInv, uCamWorld;
    uniform vec3 uCamPos;
    uniform float uNear, uFar, uAO;
    uniform vec4 uFogA, uFogB;
    uniform vec3 uSunDir, uSunCol;
    uniform float uSunScatter, uExposure;
    uniform vec3 uShadowTint, uHighTint, uWB;
    uniform float uContrast, uSaturation, uVignette, uCA, uGrain, uTime;
    uniform vec2 uRes;
    uniform int uDebug;
    uniform float uFXAA;
    varying vec2 vUv;
    flat varying float vAE;

    // ACES filmic, identical to three's ACESFilmicToneMapping / OutputPass
    vec3 RRTAndODTFit( vec3 v ) {
      vec3 a = v * ( v + 0.0245786 ) - 0.000090537;
      vec3 b = v * ( 0.983729 * v + 0.4329510 ) + 0.238081;
      return a / b;
    }
    vec3 acesFilmic( vec3 color ) {
      const mat3 ACESInputMat = mat3( vec3( 0.59719, 0.07600, 0.02840 ), vec3( 0.35458, 0.90834, 0.13383 ), vec3( 0.04823, 0.01566, 0.83777 ) );
      const mat3 ACESOutputMat = mat3( vec3( 1.60475, -0.10208, -0.00327 ), vec3( -0.53108, 1.10813, -0.07276 ), vec3( -0.07367, -0.00605, 1.07602 ) );
      color *= uExposure / 0.6;
      color = ACESInputMat * color;
      color = RRTAndODTFit( color );
      color = ACESOutputMat * color;
      return clamp( color, 0.0, 1.0 );
    }
    vec3 toSRGB( vec3 c ) { return mix( c * 12.92, 1.055 * pow( c, vec3( 0.41666 ) ) - 0.055, step( 0.0031308, c ) ); }
    // FXAA (console-style, 9 taps) on the HDR input, used when the scene target is not multisampled.
    // Taps are blended in a compressed space (c / (1 + max(c))) so bright lights do not smear.
    vec3 fxC( vec3 c ) { return c / ( 1.0 + max( c.r, max( c.g, c.b ) ) ); }
    vec3 fxU( vec3 c ) { return c / max( 1e-4, 1.0 - max( c.r, max( c.g, c.b ) ) ); }
    float fxL( vec3 c ) { return dot( c, vec3( 0.299, 0.587, 0.114 ) ); }
    vec3 fxaa( vec2 uv ) {
      vec2 rcp = 1.0 / uRes;
      vec3 nw = fxC( texture2D( tColor, uv + vec2( -0.5, -0.5 ) * rcp ).rgb ), ne = fxC( texture2D( tColor, uv + vec2( 0.5, -0.5 ) * rcp ).rgb );
      vec3 sw = fxC( texture2D( tColor, uv + vec2( -0.5, 0.5 ) * rcp ).rgb ), se = fxC( texture2D( tColor, uv + vec2( 0.5, 0.5 ) * rcp ).rgb );
      vec3 m = fxC( texture2D( tColor, uv ).rgb );
      float lNW = fxL( nw ), lNE = fxL( ne ), lSW = fxL( sw ), lSE = fxL( se ), lM = fxL( m );
      float lMin = min( lM, min( min( lNW, lNE ), min( lSW, lSE ) ) ), lMax = max( lM, max( max( lNW, lNE ), max( lSW, lSE ) ) );
      if ( lMax - lMin < max( 0.03, lMax * 0.1 ) ) return fxU( m ); // no edge
      vec2 dir = vec2( -( ( lNW + lNE ) - ( lSW + lSE ) ), ( lNW + lSW ) - ( lNE + lSE ) );
      float red = max( ( lNW + lNE + lSW + lSE ) * 0.03125, 1.0 / 128.0 );
      dir = clamp( dir / ( min( abs( dir.x ), abs( dir.y ) ) + red ), vec2( -8.0 ), vec2( 8.0 ) ) * rcp;
      vec3 a = 0.5 * ( fxC( texture2D( tColor, uv + dir * ( 1.0 / 3.0 - 0.5 ) ).rgb ) + fxC( texture2D( tColor, uv + dir * ( 2.0 / 3.0 - 0.5 ) ).rgb ) );
      vec3 b = a * 0.5 + 0.25 * ( fxC( texture2D( tColor, uv - dir * 0.5 ).rgb ) + fxC( texture2D( tColor, uv + dir * 0.5 ).rgb ) );
      float lB = fxL( b );
      return fxU( ( lB < lMin || lB > lMax ) ? a : b );
    }
    float hash12( vec2 p ) { vec3 p3 = fract( vec3( p.xyx ) * 0.1031 ); p3 += dot( p3, p3.yzx + 33.33 ); return fract( ( p3.x + p3.y ) * p3.z ); }

    // AO at this pixel. AO_UPSAMPLE_DEPTH (ao.upsample 'depth'): joint bilateral upsample instead of the bilinear tap,
    // the four AO texels around the pixel weighted bilinearly and by how close their linear depth (g) is to the
    // pixel's own (the AO blur's tolerance), so AO computed at a lower resolution keeps crisp silhouettes: no dark
    // halo on the background behind an object, no light rim along its edge. Texels of another surface keep a 1e-3
    // weight, so a pixel that matches none of them (a pole thinner than an AO texel) gets the plain bilinear value.
    float aoAt( float depth ) {
    #ifdef AO_UPSAMPLE_DEPTH
      float z = uNear * uFar / ( uFar - ( uFar - uNear ) * depth ); // linear view depth (m), the g channel's units
      ivec2 mx = textureSize( tAO, 0 ) - 1;
      vec2 p = vUv * vec2( mx + 1 ) - 0.5, f = fract( p );
      ivec2 i = ivec2( floor( p ) );
      vec2 a = texelFetch( tAO, clamp( i, ivec2( 0 ), mx ), 0 ).rg;
      vec2 b = texelFetch( tAO, clamp( i + ivec2( 1, 0 ), ivec2( 0 ), mx ), 0 ).rg;
      vec2 c = texelFetch( tAO, clamp( i + ivec2( 0, 1 ), ivec2( 0 ), mx ), 0 ).rg;
      vec2 d = texelFetch( tAO, clamp( i + 1, ivec2( 0 ), mx ), 0 ).rg;
      vec4 w = vec4( ( 1.0 - f.x ) * ( 1.0 - f.y ), f.x * ( 1.0 - f.y ), ( 1.0 - f.x ) * f.y, f.x * f.y )
             * max( vec4( 1e-3 ), 1.0 - abs( vec4( a.y, b.y, c.y, d.y ) - z ) / ( 0.06 * z + 0.15 ) );
      return dot( w, vec4( a.x, b.x, c.x, d.x ) ) / dot( w, vec4( 1.0 ) );
    #else
      return texture2D( tAO, vUv ).r;
    #endif
    }

    #ifdef DEPTH_CLOSEST
    // Under TAA the colour is resolved over the jitter but the depth is this frame's jittered sample, so a silhouette
    // pixel would get AO / fog of the object one frame and of the background the next: take the closest depth of the
    // pixel and its 4 diagonals instead (the TAA pass's velocity choice), stable over the jitter sequence.
    float closestDepth() {
      ivec2 p = ivec2( gl_FragCoord.xy ), mx = textureSize( tDepth, 0 ) - 1;
      float d = texelFetch( tDepth, p, 0 ).x;
      d = min( d, texelFetch( tDepth, clamp( p + ivec2( - 1, - 1 ), ivec2( 0 ), mx ), 0 ).x );
      d = min( d, texelFetch( tDepth, clamp( p + ivec2( 1, - 1 ), ivec2( 0 ), mx ), 0 ).x );
      d = min( d, texelFetch( tDepth, clamp( p + ivec2( - 1, 1 ), ivec2( 0 ), mx ), 0 ).x );
      return min( d, texelFetch( tDepth, clamp( p + ivec2( 1, 1 ), ivec2( 0 ), mx ), 0 ).x );
    }
    #endif

    // optical depth through an exponential layer (density d at height 0, falloff k) from y0 to y1 over dist
    float layer( float y0, float y1, float dist, float d, float k ) {
      float x = k * ( y1 - y0 );
      float f = abs( x ) > 1e-3 ? ( 1.0 - exp( -x ) ) / x : 1.0 - 0.5 * x;
      return d * exp( -k * clamp( y0, -100.0, 4000.0 ) ) * dist * f;
    }

    void main() {
    #ifdef DEPTH_CLOSEST
      float depth = closestDepth();
    #else
      float depth = texture2D( tDepth, vUv ).x;
    #endif
      vec2 dc = vUv - 0.5;
      vec2 ca = dc * dot( dc, dc ) * uCA;
      float dbgT = 1.0;
      vec3 col;
      if ( uFXAA > 0.5 ) col = fxaa( vUv );
      else if ( uCA != 0.0 ) col = vec3( texture2D( tColor, vUv - ca ).r, texture2D( tColor, vUv ).g, texture2D( tColor, vUv + ca ).b );
      else col = texture2D( tColor, vUv ).rgb; // uCA 0: the one texel all three taps would read

      if ( depth < 1.0 ) {
        if ( uHasAO > 0.5 ) {
          float ao = aoAt( depth );
          float lum = dot( col, vec3( 0.2126, 0.7152, 0.0722 ) );
          ao = mix( ao, 1.0, smoothstep( 2.0, 10.0, lum ) ); // emissives / hot highlights keep their energy
          col *= mix( 1.0, ao, uAO );
        }
        if ( uHasFog > 0.5 ) {
          vec4 vp = uProjInv * vec4( vUv * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0 );
          vp.xyz /= vp.w;
          vec3 wp = ( uCamWorld * vec4( vp.xyz, 1.0 ) ).xyz;
          vec3 rd = wp - uCamPos;
          float dist = length( rd );
          vec3 dir = rd / max( dist, 1e-3 );
          float tau = layer( uCamPos.y - uFogA.z, wp.y - uFogA.z, dist, uFogA.x, uFogA.y )
                    + layer( uCamPos.y - uFogB.z, wp.y - uFogB.z, dist, uFogB.x, uFogB.y );
          float T = exp( -tau ) * ( 1.0 - smoothstep( uFogA.w * uFar, 0.985 * uFar, dist ) );
          vec3 fd = normalize( vec3( dir.x, max( dir.y, uFogB.w ), dir.z ) );
        #ifdef FOG_SKY
          vec3 fogCol = textureCube( tSky, fd ).rgb;
        #else
          vec3 fogCol = uFogColor; // no sky cube (no atmosphere): the fogColor option
        #endif
          if ( uSunScatter != 0.0 ) fogCol += uSunCol * uSunScatter * pow( max( dot( dir, uSunDir ), 0.0 ), 10.0 ); // 0 at night, high sun
          col = col * T + fogCol * ( 1.0 - T );
          dbgT = T;
        }
      }
      if ( uHasBloom > 0.5 ) col += texture2D( tBloom, vUv ).rgb;
      if ( uAE.y > 0.0 ) col *= vAE;

      col = acesFilmic( col * uWB );
      // grading in display-linear space
      float l = dot( col, vec3( 0.2126, 0.7152, 0.0722 ) );
      col = max( mix( vec3( l ), col, uSaturation ), 0.0 );
      col += uShadowTint * ( 1.0 - smoothstep( 0.0, 0.45, l ) ) + uHighTint * smoothstep( 0.3, 1.0, l );
      vec3 p = sqrt( max( col, 0.0 ) );          // contrast S-curve in a perceptual space
      p = mix( p, p * p * ( 3.0 - 2.0 * p ), uContrast );
      col = p * p;
      vec2 vc = dc * vec2( uRes.x / uRes.y, 1.0 );
      col *= mix( 1.0, smoothstep( 1.05, 0.25, length( vc ) ), uVignette );
      vec3 o = toSRGB( clamp( col, 0.0, 1.0 ) );
      if ( uGrain > 0.0 ) o += ( hash12( vUv * uRes + fract( uTime * 7.31 ) * 913.0 ) - 0.5 ) * uGrain * ( 1.0 - 0.6 * l ); // film grain in display space
      o += ( hash12( vUv * uRes + 17.0 ) - 0.5 ) / 255.0; // dither: no banding in the sky gradients
      gl_FragColor = vec4( o, 1.0 );
      if ( uDebug == 1 ) gl_FragColor = vec4( vec3( aoAt( depth ) ), 1.0 );
      else if ( uDebug == 2 ) gl_FragColor = vec4( toSRGB( clamp( texture2D( tBloom, vUv ).rgb, 0.0, 1.0 ) ), 1.0 );
      else if ( uDebug == 3 ) gl_FragColor = vec4( vec3( dbgT ), 1.0 );
      gl_FragDepth = depth; // the lens-flare overlay depth-tests against the scene
    }`,
};

class FinalPass extends Pass {
  constructor(fx) {
    super();
    this.fx = fx;
    this.needsSwap = false;
    this.material = new THREE.ShaderMaterial({
      name: 'PostFXFinal',
      uniforms: THREE.UniformsUtils.clone(FinalShader.uniforms),
      vertexShader: FinalShader.vertexShader,
      fragmentShader: FinalShader.fragmentShader,
      depthTest: true, depthWrite: true, depthFunc: THREE.AlwaysDepth,
      toneMapped: false,
    });
    this._quad = new FullScreenQuad(this.material);
    this._aoUp = false; // AO_UPSAMPLE_DEPTH defined (aoPass.upsample === 'depth')
    this._sky = false; // FOG_SKY defined (the atmosphere has a sky cube)
    this._closest = false; // DEPTH_CLOSEST defined (TAA on and jittering, taa.stableDepth)
  }
  setSize(w, h) { this.material.uniforms.uRes.value.set(w, h); }
  render(renderer, writeBuffer, readBuffer) {
    const u = this.material.uniforms, fx = this.fx, taa = fx.taaPass !== null && fx.taaPass.enabled;
    const up = fx.aoPass.upsample === 'depth';
    const sky = fx.atmo && fx.atmo.skyCube, hasSky = !!sky;
    const closest = taa && fx.history.jitter && fx.taaOptions.stableDepth !== false; // no jitter: depth is stable
    // defines, not branches: nothing extra in the modes that do not need them; one cached program per combination
    if (up !== this._aoUp || hasSky !== this._sky || closest !== this._closest) {
      this._aoUp = up; this._sky = hasSky; this._closest = closest;
      const d = this.material.defines;
      if (up) d.AO_UPSAMPLE_DEPTH = 1; else delete d.AO_UPSAMPLE_DEPTH;
      if (hasSky) d.FOG_SKY = 1; else delete d.FOG_SKY; // no sampler bound to an unset cube
      if (closest) d.DEPTH_CLOSEST = 1; else delete d.DEPTH_CLOSEST;
      this.material.needsUpdate = true;
    }
    u.tColor.value = taa ? fx.taaPass.texture : readBuffer.texture;
    u.tDepth.value = readBuffer.depthTexture;
    u.uHasAO.value = fx.aoPass.enabled ? 1 : 0;
    u.tAO.value = fx.aoPass.aoTexture;
    u.uHasBloom.value = fx.bloomPass.enabled ? 1 : 0;
    u.tBloom.value = fx.bloomPass.bloomTexture;
    u.uHasFog.value = (hasSky || fx.fogColor) && fx.fog ? 1 : 0;
    u.tSky.value = sky;
    if (!hasSky && fx.fogColor) u.uFogColor.value.copy(fx.fogColor);
    u.tAdapt.value = fx.exposurePass.texture;
    u.uAE.value.y = fx.exposurePass.enabled && fx.autoExposure && fx.grade ? fx.grade.aeS : 0;
    renderer.setRenderTarget(this.renderToScreen ? null : writeBuffer);
    this._quad.render(renderer);
  }
  dispose() { this.material.dispose(); this._quad.dispose(); }
}

// The scene pass. With fx.lateClear, renderer.render() clears the target itself (autoClear, in WebGLBackground.render,
// which runs after shadowMap.render) instead of RenderPass clearing it before render(). On ANGLE / Metal a clear opens
// the bound target's render pass at once (FrameBufferMtl clearWithLoadOpRenderPassNotStarted: load action clear,
// store), so the shadow-map passes drawn next close that pass with nothing but the clear in it: the cleared
// (multisampled) colour + depth are written to memory, then loaded back for the scene draws, a full-screen store +
// load every frame (MSAA and FXAA targets alike) that the late clear does not do. Same clear values (renderer clear
// colour, autoClearColor / Depth / Stencil); with any RenderPass option set it falls back to RenderPass.
// G-buffer: renderer.clear clears every draw buffer with the clear colour, so attachments 1-2 (normal + flags,
// velocity) need GBuffer.clear after it. For the frame's scene pass renderer.clear is wrapped (fx._gbClear, restored
// in finally): whichever clear hits the G-buffer target, late inside render() or RenderPass's early one, is followed
// at once by GBuffer.clear, so the late clear keeps its saving with three colour attachments. A pass that does not
// clear at all (RenderPass.clear false) gets GBuffer.clear before render().
class ScenePass extends RenderPass {
  constructor(scene, camera, fx) { super(scene, camera); this.fx = fx; }
  render(renderer, writeBuffer, readBuffer, deltaTime, maskActive) {
    const fx = this.fx, gb = fx.gbuffer;
    if (gb === null) { this._render(renderer, writeBuffer, readBuffer, deltaTime, maskActive); return; }
    // per frame, for this pass only: swap in the pre-bound wrapper (no allocation) and put three's method back in
    // finally. Not setClearAlpha(0) instead: a scene.background Colour clears with alpha 1, which would mark the
    // background pixels as G-buffer writers (flags bit0) with the clear colour as their velocity.
    fx._rendererClear = renderer.clear;
    renderer.clear = fx._gbClear;
    try {
      if (!this.clear) { renderer.setRenderTarget(readBuffer); fx._clearGBuffer(); }
      this._render(renderer, writeBuffer, readBuffer, deltaTime, maskActive);
    } finally {
      renderer.clear = fx._rendererClear;
    }
  }
  _render(renderer, writeBuffer, readBuffer, deltaTime, maskActive) {
    const plain = this.clear && !this.clearDepth && this.overrideMaterial === null && this.clearColor === null && this.clearAlpha === null;
    if (!this.fx.lateClear || !plain) { super.render(renderer, writeBuffer, readBuffer, deltaTime, maskActive); return; }
    const ac = renderer.autoClear;
    renderer.autoClear = true;
    renderer.setRenderTarget(this.renderToScreen ? null : readBuffer);
    renderer.render(this.scene, this.camera);
    renderer.autoClear = ac;
  }
}

// BloomPass reading fx.colorSource (the TAA resolve while TAA is on) instead of the scene target.
class SourceBloomPass extends BloomPass {
  constructor(fx, o) { super(o); this.fx = fx; }
  render(renderer, writeBuffer, readBuffer, deltaTime) { super.render(renderer, writeBuffer, this.fx._source(readBuffer), deltaTime); }
}

export const AA_MODES = ['auto', 'msaa', 'fxaa', 'taa'];

/**
 * The anti-aliasing decision, pure (test/postfx-options.mjs). Writes into and returns `out`:
 * { mode: 'msaa' | 'fxaa' | 'taa', samples (scene target MSAA count), fxaa (final-pass FXAA), taa, forced (the
 * requested mode is unavailable here and fell back) }.
 *   G-buffer off: 'auto' = the 0.6.0 rule: msaa 'auto' -> caps.maxSamples by day, 0 (FXAA) once darkness > 0.35 or
 *     pixels > msaaMaxPixels; msaa n -> min(maxSamples, n) (0: FXAA). 'msaa' = msaa n or maxSamples whatever the
 *     darkness / size; 'fxaa' = 0 samples; 'taa' throws (it reprojects with the G-buffer velocity).
 *   G-buffer on: always 0 samples (MSAA on an MRT target resolves every attachment and a depth per attachment);
 *     'auto' and 'taa' = TAA, 'fxaa' = FXAA, 'msaa' = FXAA with forced: true.
 * @param {object} [opts] aa ('auto'), gbuffer (false), msaa ('auto'), msaaMaxPixels (Infinity), darkness (0), pixels (0)
 * @param {object} [caps] maxSamples (0)
 * @param {object} [out]
 */
export function resolveAA(opts = {}, caps = {}, out = {}) {
  const aa = opts.aa ?? 'auto', gb = !!opts.gbuffer, max = caps.maxSamples ?? 0, msaa = opts.msaa ?? 'auto';
  if (!AA_MODES.includes(aa)) throw new Error(`[PostFX] unknown aa '${aa}' ('auto' | 'msaa' | 'fxaa' | 'taa')`);
  if (aa === 'taa' && !gb) throw new Error("[PostFX] aa 'taa' needs gbuffer: true (TAA reprojects with the G-buffer velocity and flags)");
  let samples = 0, taa = false, forced = false;
  if (gb) {
    taa = aa === 'auto' || aa === 'taa';
    forced = aa === 'msaa';
  } else if (aa === 'auto') {
    samples = msaa === 'auto'
      ? ((opts.darkness ?? 0) > 0.35 || (opts.pixels ?? 0) > (opts.msaaMaxPixels ?? Infinity) ? 0 : max)
      : Math.min(max, msaa | 0);
  } else if (aa === 'msaa') {
    samples = msaa === 'auto' ? max : Math.min(max, msaa | 0);
    forced = max === 0;
  }
  out.mode = taa ? 'taa' : samples > 0 ? 'msaa' : 'fxaa';
  out.samples = samples;
  out.fxaa = !taa && samples === 0;
  out.taa = taa;
  out.forced = forced;
  return out;
}

let gbufferChunks = false; // installGBufferChunks() done (global ShaderChunk patch, once per page)

// grading / exposure / bloom keyframes by sun elevation (degrees)
export const DEFAULT_GRADE = [
  { el: -40, wb: [0.93, 0.98, 1.08], aeK: 0.04, aeS: 0.45, aeMax: 1.6, exp: 1.35, bT: 0.85, bS: 0.85, bR: 0.62, sat: 1.0, con: 0.14, split: 0.9, sh: [-0.004, 0.0, 0.012], hi: [0.016, 0.006, -0.008], vig: 0.38, grain: 0.022, ao: 0.6 },
  { el: -8, wb: [0.95, 0.98, 1.07], aeK: 0.05, aeS: 0.45, aeMax: 1.6, exp: 1.3, bT: 0.95, bS: 0.75, bR: 0.6, sat: 1.02, con: 0.14, split: 0.9, sh: [-0.006, 0.0, 0.016], hi: [0.02, 0.008, -0.01], vig: 0.36, grain: 0.016, ao: 0.65 },
  { el: -3, wb: [1.0, 0.98, 1.02], aeK: 0.09, aeS: 0.5, aeMax: 1.9, exp: 1.15, bT: 1.3, bS: 0.55, bR: 0.55, sat: 1.06, con: 0.15, split: 1.0, sh: [-0.008, 0.002, 0.018], hi: [0.026, 0.01, -0.012], vig: 0.34, grain: 0.008, ao: 0.7 },
  { el: 2, wb: [1.06, 1.0, 0.92], aeK: 0.16, aeS: 0.5, aeMax: 2.0, exp: 0.95, bT: 2.2, bS: 0.32, bR: 0.5, sat: 1.1, con: 0.17, split: 1.0, sh: [-0.010, 0.004, 0.018], hi: [0.03, 0.012, -0.016], vig: 0.32, grain: 0, ao: 0.78 },
  { el: 10, wb: [1.045, 1.0, 0.95], aeK: 0.19, aeS: 0.5, aeMax: 2.0, exp: 0.92, bT: 2.6, bS: 0.24, bR: 0.45, sat: 1.08, con: 0.17, split: 1.0, sh: [-0.009, 0.004, 0.016], hi: [0.026, 0.01, -0.014], vig: 0.3, grain: 0, ao: 0.8 },
  { el: 25, wb: [1.015, 1.0, 0.985], aeK: 0.21, aeS: 0.5, aeMax: 2.0, exp: 0.9, bT: 2.9, bS: 0.2, bR: 0.4, sat: 1.05, con: 0.16, split: 0.5, sh: [-0.006, 0.002, 0.01], hi: [0.012, 0.006, -0.006], vig: 0.28, grain: 0, ao: 0.8 },
  { el: 50, wb: [1.0, 1.0, 1.0], aeK: 0.22, aeS: 0.5, aeMax: 2.0, exp: 0.88, bT: 3.0, bS: 0.18, bR: 0.4, sat: 1.04, con: 0.15, split: 0.35, sh: [-0.004, 0.002, 0.008], hi: [0.008, 0.005, -0.004], vig: 0.26, grain: 0, ao: 0.8 },
  // high sun up to the zenith: the 50 deg grade held with a gentle exposure / bloom extrapolation (below 50 deg unchanged)
  { el: 90, wb: [1.0, 1.0, 1.0], aeK: 0.22, aeS: 0.5, aeMax: 2.0, exp: 0.86, bT: 3.1, bS: 0.17, bR: 0.4, sat: 1.04, con: 0.15, split: 0.3, sh: [-0.004, 0.002, 0.008], hi: [0.008, 0.005, -0.004], vig: 0.26, grain: 0, ao: 0.8 },
];
/** Interpolated grade for a sun elevation (degrees) from keyframes sorted by `el`. */
export function gradeAt(keys, elDeg) {
  let i = 0;
  while (i < keys.length - 2 && elDeg > keys[i + 1].el) i++;
  const a = keys[i], b = keys[i + 1], t = sstep(a.el, b.el, elDeg), o = {};
  for (const k in a) o[k] = Array.isArray(a[k]) ? a[k].map((v, j) => mix(v, b[k][j], t)) : mix(a[k], b[k], t);
  return o;
}

export class PostFX {
  /**
   * @param {THREE.WebGLRenderer} renderer
   * @param {THREE.Scene} scene
   * @param {THREE.PerspectiveCamera} camera
   * @param {object} [o] atmosphere (an Atmosphere, or null), width/height (CSS px), pixelRatio, samples (MSAA,
   *   default 4), msaa, msaaMaxPixels, lateClear, compactHDR, grade, ao, bloom, fog, fogFarFade, fogMinSkyElevation,
   *   autoExposure, fogColor, fogParams, gbuffer, aa, taa, jitter, jitterPattern, timing (see header)
   */
  constructor(renderer, scene, camera, o = {}) {
    this.renderer = renderer; this.scene = scene; this.camera = camera;
    this.atmo = o.atmosphere || null;
    const size = renderer.getSize(new THREE.Vector2());
    this.width = o.width ?? size.x; this.height = o.height ?? size.y;
    this.pixelRatio = o.pixelRatio ?? renderer.getPixelRatio();
    this.enabled = true;
    this.fog = o.fog ?? true;
    this.exposureScale = 1;
    this.autoExposure = o.autoExposure ?? true;
    this.gradeKeys = o.grade || DEFAULT_GRADE;
    this.fogFarFade = o.fogFarFade ?? 0.72;
    this.fogMinSkyEl = o.fogMinSkyElevation ?? 0.02;
    // without a sky cube (no atmosphere): fog inscatter colour (linear HDR; null = no fog) and densities
    this.fogColor = o.fogColor != null ? new THREE.Color(o.fogColor) : null;
    this.fogParams = { haze: 0.0002, hazeK: 1 / 700, hazeBase: 0, mist: 0.0001, mistK: 1 / 40, ...(o.fogParams || {}) };
    this.maxSamples = Math.min(o.samples ?? 4, renderer.capabilities.maxSamples || 0);
    // 'auto': 4x MSAA while the scene is cheap to shade (day), FXAA in the final pass once it is dark (the night
    // scene's 16 street-lamp point lights make every MSAA edge fragment expensive: ~11 ms vs ~0.3 ms for FXAA)
    this.msaa = o.msaa ?? 'auto';
    // ... and FXAA whenever the drawing buffer has more than msaaMaxPixels pixels (default Infinity: no limit). At
    // Retina density aliasing is far less visible, and the 4x path's cost is mostly the resolve (colour + depth blit,
    // see ScenePass), not sample shading: Canyon Run at 3024x1964, 4x -> FXAA saves 2.7 ms, 4x -> 2x only 0.4 ms
    this.msaaMaxPixels = o.msaaMaxPixels ?? Infinity;
    // clear the scene target after the shadow maps (see ScenePass): identical output, ~1 ms at 3024x1964 on Apple GPUs
    this.lateClear = o.lateClear ?? true;
    this._compactHDR = o.compactHDR;
    /** Anti-aliasing mode: 'auto' | 'msaa' | 'fxaa' | 'taa' (setAA; resolved by resolveAA). */
    this.aa = o.aa ?? 'auto';
    /** TAA settings, kept across setGBuffer rebuilds (set('taa.*') updates them and the live pass). */
    this.taaOptions = {
      blend: 0.9, clamp: 'variance', gamma: 1, sharpen: 0, filter: 'catmull-rom', historyFormat: 'rgba16f', stableDepth: false,
      reject: 'quad', velocityFirst: false, snap: false, // cost knobs (see TAAPass)
    };
    for (const k in o.taa || {}) this._setTaaOption(k, o.taa[k]); // validated: unknown keys / values throw
    /** Sub-pixel camera jitter while TAA is on. */
    this.jitter = o.jitter ?? true;
    this.jitterPattern = o.jitterPattern ?? 'halton16';
    /** Per-pass CPU ms in `timings` (render loop run pass by pass; off: composer.render, nothing measured). */
    this.timing = !!o.timing;
    this.timings = { scene: 0, exposure: 0, ao: 0, taa: 0, bloom: 0, final: 0, flare: 0, total: 0 };
    this._aaIn = { aa: 'auto', gbuffer: false, msaa: 'auto', msaaMaxPixels: Infinity, darkness: 0, pixels: 0 };
    this._aaCaps = { maxSamples: this.maxSamples };
    this._aaOut = resolveAA({ aa: this.aa, gbuffer: !!o.gbuffer }, this._aaCaps); // throws before anything is built
    this._warnedForced = false;
    this.gbuffer = null; this.history = null; this.taaPass = null;
    this._tracked = []; this._trackOpts = []; this._trackedPools = []; // replayed into every new GBuffer
    this._spare = null; // composer writeBuffer in G-buffer mode (no pass writes it: never allocated)
    this._srcProxy = { texture: null, depthTexture: null }; // readBuffer stand-in for the bloom pass under TAA
    this._rendererClear = null;
    this._gbClearing = false;
    this._gbClear = (color = true, depth = true, stencil = true) => { // see ScenePass
      const r = this.renderer;
      this._rendererClear.call(r, color, depth, stencil);
      if (color && this.gbuffer !== null && r.getRenderTarget() === this.gbuffer.target) this._clearGBuffer();
    };

    const W = Math.max(1, Math.floor(this.width * this.pixelRatio)), H = Math.max(1, Math.floor(this.height * this.pixelRatio));
    this._W = W; this._H = H; // drawing-buffer size, for msaaMaxPixels
    this.composer = new EffectComposer(renderer, this._makeSceneTarget());

    this.renderPass = new ScenePass(scene, camera, this);
    this.exposurePass = new ExposurePass();
    // AO defaults: half resolution capped at 0.6 Mpx (a 14-inch Retina MacBook's half-res buffer is 1.5 Mpx: -1.3 to
    // -1.7 ms measured at 3024x1964, full-res mean pixel change <= 0.04 / 255 and < 0.5 % of pixels over 2 / 255 with
    // the depth-aware upsample; 1080p-class buffers are under the cap and unchanged), compact RG16F buffers
    this.aoPass = new AOPass(camera, { radius: 1.8, thickness: 1.6, samples: 12, resolutionScale: 0.5, maxPixels: 0.6e6, upsample: 'depth', compact: true, ...(o.ao || {}) });
    this.bloomPass = new SourceBloomPass(this, { strength: 0.3, radius: 0.45, threshold: 2.8, resolutionScale: 0.5, ...(o.bloom || {}) });
    this.finalPass = new FinalPass(this);
    for (const p of [this.renderPass, this.exposurePass, this.aoPass, this.bloomPass, this.finalPass]) this.composer.addPass(p);
    this.uniforms = this.finalPass.material.uniforms;
    this._passKeys = new Map([[this.renderPass, 'scene'], [this.exposurePass, 'exposure'], [this.aoPass, 'ao'], [this.bloomPass, 'bloom'], [this.finalPass, 'final']]);

    // the composer works in drawing-buffer pixels (floored exactly like the canvas), so no fractional targets
    this.composer.setPixelRatio(1);
    this.composer.setSize(W, H);
    if (o.gbuffer) this._buildGBuffer(true);
    if (this.atmo) {
      this.atmo.setFogMode('post');
      if (this.atmo.setFlareScale) this.atmo.setFlareScale(H);
    }
    this._ver = -1;
    this._time = 0;
    this.setTimeOfDay(this.atmo ? this.atmo.darkness : 0);
  }

  // convenience handles (e.g. window.__game.postfx.ao.enabled = false)
  get ao() { return this.aoPass; }
  get bloom() { return this.bloomPass; }
  /** Debug view: 'ao' | 'bloom' | 'fog' | null. */
  set debug(v) { this.uniforms.uDebug.value = { ao: 1, bloom: 2, fog: 3 }[v] || 0; }
  get debug() { return [null, 'ao', 'bloom', 'fog'][this.uniforms.uDebug.value]; }
  /** Scene depth texture of either mode (the resolved MSAA depth, or the G-buffer depth). */
  get sceneDepth() { return this.composer.readBuffer.depthTexture; }
  /** HDR colour Bloom and Final read: the TAA resolve while TAA is on, else the scene colour. */
  get colorSource() { return this.taaPass !== null && this.taaPass.enabled ? this.taaPass.texture : this.composer.readBuffer.texture; }

  /** CSS size + pixel ratio, exactly as given to renderer.setPixelRatio / setSize (call after those). */
  setSize(w, h, pixelRatio = this.pixelRatio) {
    this.width = w; this.height = h; this.pixelRatio = pixelRatio;
    const W = Math.max(1, Math.floor(w * pixelRatio)), H = Math.max(1, Math.floor(h * pixelRatio));
    this._W = W; this._H = H;
    if (this.gbuffer) this.gbuffer.setSize(W, H); // before the composer resizes its target (see _bindGBuffer)
    this.composer.setSize(W, H);
    if (this.gbuffer) this._bindGBuffer();
    this._applyAA(this._aaDark); // msaaMaxPixels: the new size may change 'auto' (a no-op while it is Infinity)
    if (this.atmo && this.atmo.setFlareScale) this.atmo.setFlareScale(H);
  }

  setEnabled(on) {
    const was = this.enabled;
    this.enabled = !!on;
    if (this.enabled && !was && this.history) this.history.reset(); // frames drawn without it: no valid history
    if (this.atmo) this.atmo.setFogMode(this.enabled && this.fog ? 'post' : 'scene');
  }

  /** Per-time look: exposure, bloom threshold/strength, grading. `darkness` 0 (day) .. 1 (night). */
  setTimeOfDay(darkness = 0) {
    const elDeg = this.atmo && Number.isFinite(this.atmo.sunEl) ? this.atmo.sunEl / DEG : mix(20, -20, clamp(darkness, 0, 1));
    const g = this.grade = gradeAt(this.gradeKeys, elDeg);
    const u = this.uniforms;
    this.baseExposure = (this.renderer.toneMappingExposure || 0.9) / 0.9 * g.exp;
    u.uSaturation.value = g.sat;
    u.uContrast.value = g.con;
    u.uShadowTint.value.set(g.sh[0] * g.split, g.sh[1] * g.split, g.sh[2] * g.split);
    u.uHighTint.value.set(g.hi[0] * g.split, g.hi[1] * g.split, g.hi[2] * g.split);
    u.uVignette.value = g.vig;
    u.uWB.value.set(g.wb[0], g.wb[1], g.wb[2]);
    u.uGrain.value = g.grain;
    u.uAO.value = g.ao;
    this._applyAA(darkness);
    u.uAE.value.set(g.aeK, this.autoExposure ? g.aeS : 0, 0.6, g.aeMax);
    this.exposurePass.reset();
    this.bloomPass.threshold = g.bT;
    this.bloomPass.strength = g.bS;
    this.bloomPass.radius = g.bR;
    if (this.atmo) this._ver = this.atmo.version;
  }

  _applyAA(darkness = this.atmo ? this.atmo.darkness : this._aaDark ?? 0) {
    this._aaDark = darkness; // setSize re-applies the size rule with the same darkness
    const i = this._aaIn;
    i.aa = this.aa; i.gbuffer = this.gbuffer !== null; i.msaa = this.msaa; i.msaaMaxPixels = this.msaaMaxPixels;
    i.darkness = darkness; i.pixels = this._W * this._H;
    const r = resolveAA(i, this._aaCaps, this._aaOut);
    const want = r.samples;
    if (this.gbuffer === null) { // the G-buffer target is single-sampled (resolveAA gives 0 there)
      for (const rt of [this.composer.renderTarget1, this.composer.renderTarget2]) {
        if (rt.samples !== want) { rt.samples = want; rt.dispose(); }
      }
    }
    this.uniforms.uFXAA.value = r.fxaa ? 1 : 0;
    if (this.taaPass) this.taaPass.enabled = r.taa; // a re-enabled TAA notices the frames it skipped (fresh history)
    if (this.history) this.history.jitter = r.taa && this.jitter;
    if (r.forced && !this._warnedForced) {
      this._warnedForced = true;
      console.warn(`[PostFX] aa '${this.aa}' is unavailable ${this.gbuffer ? 'on the single-sampled G-buffer' : 'without MSAA support'}: using FXAA`);
    }
  }

  /**
   * Anti-aliasing: 'auto' (default), or a fixed MSAA sample count (0 = FXAA only). maxPixels (optional) replaces
   * msaaMaxPixels, the drawing-buffer pixel count above which 'auto' uses FXAA (Infinity / null: no limit). Selects
   * aa 'auto' (the rule these parameters drive) while the G-buffer is off; on the G-buffer they are kept for later.
   */
  setMSAA(mode, maxPixels) {
    this.msaa = mode;
    if (maxPixels !== undefined) this.msaaMaxPixels = maxPixels ?? Infinity;
    if (this.gbuffer === null) this.aa = 'auto';
    this._applyAA();
  }

  /** Anti-aliasing mode: 'auto' | 'msaa' | 'fxaa' | 'taa' (see resolveAA; 'taa' without the G-buffer throws). */
  setAA(mode) {
    resolveAA({ aa: mode, gbuffer: this.gbuffer !== null }, this._aaCaps); // validate before changing anything
    this.aa = mode;
    this._applyAA();
    return this;
  }

  /**
   * Switch the G-buffer scene target on / off at runtime: rebuilds the scene targets (and the TAA pass + camera
   * history), so it costs a shader recompile of the patched materials the first time. Off with aa 'taa' -> 'auto'.
   */
  setGBuffer(on) {
    on = !!on;
    if (on === (this.gbuffer !== null)) return this;
    this._buildGBuffer(on);
    this._applyAA();
    return this;
  }

  /** Let the G-buffer pick up objects added to the scene since the last sync (GBuffer.sync); no-op when it is off. */
  syncGBuffer() {
    if (this.gbuffer) this.gbuffer.sync(this.scene);
    return this;
  }

  /**
   * Per-object velocity for dynamic meshes (GBuffer.track). PostFX keeps the list and replays it whenever the
   * G-buffer is (re)built, so call it once, G-buffer on or off. Returns the meshes tracked now (0 while it is off).
   */
  track(object3d, opts) {
    if (!object3d) return 0;
    const i = this._tracked.indexOf(object3d);
    if (i < 0) { this._tracked.push(object3d); this._trackOpts.push(opts); } else this._trackOpts[i] = opts;
    return this.gbuffer ? this.gbuffer.track(object3d, opts) : 0;
  }

  untrack(object3d) {
    const i = this._tracked.indexOf(object3d);
    if (i >= 0) { this._tracked.splice(i, 1); this._trackOpts.splice(i, 1); }
    if (this.gbuffer) this.gbuffer.untrack(object3d);
  }

  /** InstancedPool({ velocity: true }) instances (GBuffer.trackPool), replayed like track(). */
  trackPool(pool) {
    if (pool && !this._trackedPools.includes(pool)) this._trackedPools.push(pool);
    if (this.gbuffer) this.gbuffer.trackPool(pool);
  }

  untrackPool(pool) {
    const i = this._trackedPools.indexOf(pool);
    if (i >= 0) this._trackedPools.splice(i, 1);
    if (this.gbuffer) this.gbuffer.untrackPool(pool);
  }

  _setTaaOption(key, value) {
    switch (key) {
      case 'blend': case 'sharpen': case 'gamma':
        if (typeof value !== 'number' || !Number.isFinite(value)) throw new Error(`[PostFX] taa.${key}: expected a finite number, got ${value}`);
        break;
      case 'clamp': checkOption(value, TAA_CLAMPS, 'taa.clamp'); break;
      case 'filter': checkOption(value, TAA_FILTERS, 'taa.filter'); break;
      case 'historyFormat': checkOption(value, TAA_HISTORY_FORMATS, 'taa.historyFormat'); break;
      case 'reject': checkOption(value, TAA_REJECTS, 'taa.reject'); break;
      case 'stableDepth': case 'velocityFirst': case 'snap': value = !!value; break;
      default: throw new Error(`[PostFX] unknown taa option '${key}' (${Object.keys(this.taaOptions).join(', ')})`);
    }
    this.taaOptions[key] = value;
  }

  /**
   * Runtime knob by path, for harness settings: 'aa' (setAA), 'gbuffer' (setGBuffer), 'jitter', 'taa.blend',
   * 'taa.sharpen', 'taa.gamma', 'taa.clamp', 'taa.filter', 'taa.historyFormat', 'taa.stableDepth', 'taa.reject',
   * 'taa.velocityFirst', 'taa.snap', 'ao.enabled', 'bloom.enabled', 'timing'. Unknown paths throw.
   */
  set(path, value) {
    switch (path) {
      case 'aa': return this.setAA(value);
      case 'gbuffer': return this.setGBuffer(value);
      case 'jitter': this.jitter = !!value; this._applyAA(); break;
      case 'taa.blend': case 'taa.sharpen': case 'taa.gamma': case 'taa.clamp': case 'taa.filter': case 'taa.historyFormat':
      case 'taa.stableDepth': case 'taa.reject': case 'taa.velocityFirst': case 'taa.snap':
        this._setTaaOption(path.slice(4), value); break;
      case 'ao.enabled': this.aoPass.enabled = !!value; break;
      case 'bloom.enabled': this.bloomPass.enabled = !!value; break;
      case 'timing': this.timing = !!value; break;
      default: throw new Error(`[PostFX] set: unknown path '${path}'`);
    }
    if (this.taaPass && path.startsWith('taa.') && path !== 'taa.stableDepth') this.taaPass[path.slice(4)] = this.taaOptions[path.slice(4)];
    return this;
  }

  /**
   * AO buffer size and upsampling at runtime (console A/B); the AO targets resize at once and omitted fields keep
   * their value. scale: of the drawing buffer; maxPixels: pixel budget of the AO buffers (Infinity / null: none);
   * upsample: 'linear' (bilinear tap) | 'depth' (depth-aware, aoAt in the final pass). Returns the AO buffer size.
   */
  setAOResolution({ scale, maxPixels, upsample } = {}) {
    const ao = this.aoPass;
    if (upsample !== undefined) ao.upsample = upsample;
    ao.setResolution(scale ?? ao.scale, maxPixels === undefined ? ao.maxPixels : maxPixels);
    return { width: ao.rtA.width, height: ao.rtA.height };
  }

  _syncFrame(dt) {
    const a = this.atmo, u = this.uniforms, cam = this.camera;
    if (a && a.version !== this._ver) this.setTimeOfDay(a.darkness);
    this._time += dt || 0;
    u.uTime.value = this._time % 1000;
    u.uExposure.value = this.baseExposure * this.exposureScale;
    u.uProjInv.value.copy(cam.projectionMatrixInverse);
    u.uCamWorld.value.copy(cam.matrixWorld);
    u.uCamPos.value.setFromMatrixPosition(cam.matrixWorld);
    u.uNear.value = cam.near; u.uFar.value = cam.far;
    if (a) {
      const fp = a.fogParams;
      u.uFogA.value.set(fp.haze, fp.hazeK, fp.hazeBase || 0, this.fogFarFade);
      u.uFogB.value.set(fp.mist, fp.mistK, 0, this.fogMinSkyEl);
      u.uSunDir.value.copy(a.sunDir);
      u.uSunCol.value.copy(a.sunColor);
      u.uSunScatter.value = fp.sunScatter;
    } else {
      const fp = this.fogParams;
      u.uFogA.value.set(fp.haze, fp.hazeK, fp.hazeBase || 0, this.fogFarFade);
      u.uFogB.value.set(fp.mist, fp.mistK, 0, this.fogMinSkyEl);
      u.uSunScatter.value = 0;
    }
  }

  render(dt) {
    const r = this.renderer;
    if (!this.enabled) {
      r.render(this.scene, this.camera);
      if (this.atmo) this.atmo.renderFlare(r, this.camera);
      return;
    }
    const timed = this.timing, t0 = timed ? performance.now() : 0, h = this.history, gb = this.gbuffer;
    this.camera.updateMatrixWorld();
    if (gb !== null && this.composer.readBuffer !== gb.target) this._bindGBuffer();
    if (h !== null) h.begin(this._W, this._H); // jitter (TAA): every pass up to Final sees the jittered camera
    this._syncFrame(dt);
    const info = r.info, ar = info.autoReset;
    info.autoReset = false; info.reset();
    try {
      if (timed) this._renderTimed(dt); else this.composer.render(dt);
    } finally {
      if (h !== null) h.end();
    }
    if (gb !== null) gb.endFrame();
    if (this.atmo) {
      const t1 = timed ? performance.now() : 0;
      this.atmo.renderFlare(r, this.camera);
      if (timed) this.timings.flare = performance.now() - t1;
    }
    info.autoReset = ar;
    if (timed) this.timings.total = performance.now() - t0;
  }

  dispose() {
    this.composer.dispose();
    this.aoPass.dispose(); this.bloomPass.dispose(); this.finalPass.dispose(); this.exposurePass.dispose();
    if (this.taaPass) this.taaPass.dispose();
    if (this.gbuffer) this.gbuffer.dispose();
    if (this.atmo) this.atmo.setFogMode('scene');
  }

  // EffectComposer.render, pass by pass with performance.now() around each (CPU submission time; the GPU runs later)
  _renderTimed(dt) {
    const c = this.composer, r = this.renderer, t = this.timings, passes = c.passes, keys = this._passKeys;
    if (dt === undefined) dt = c.clock.getDelta(); // as EffectComposer.render
    const cur = r.getRenderTarget();
    for (let i = 0; i < passes.length; i++) {
      const p = passes[i], k = keys.get(p);
      if (p.enabled === false) { if (k !== undefined) t[k] = 0; continue; }
      p.renderToScreen = c.renderToScreen && c.isLastEnabledPass(i);
      const t0 = performance.now();
      p.render(r, c.writeBuffer, c.readBuffer, dt, false);
      if (p.needsSwap) c.swapBuffers();
      if (k !== undefined) t[k] = performance.now() - t0;
    }
    if (this.taaPass === null) t.taa = 0;
    r.setRenderTarget(cur);
  }

  // the forward scene target: HDR, MSAA (samples set by _applyAA), resolved depth texture
  _makeSceneTarget() {
    const W = this._W, H = this._H, renderer = this.renderer, samples = this.maxSamples;
    const depthTexture = new THREE.DepthTexture(W, H, THREE.UnsignedIntType);
    // R11G11B10 float halves the MSAA bandwidth of RGBA16F (~1 ms at 1500x1300 on an M4); needs EXT_color_buffer_float
    const small = this._compactHDR !== false && renderer.extensions.has('EXT_color_buffer_float');
    const fmt = small ? { format: THREE.RGBFormat, type: THREE.UnsignedInt101111Type } : { type: THREE.HalfFloatType };
    const rt = new THREE.WebGLRenderTarget(W, H, { ...fmt, samples, depthBuffer: true, stencilBuffer: false, depthTexture });
    rt.texture.name = 'PostFX.scene';
    rt.texture.generateMipmaps = false;
    return rt;
  }

  // Swap the composer's scene targets. EffectComposer renders into and reads `readBuffer` (= renderTarget2, the clone
  // it makes of the target it is given), so in G-buffer mode readBuffer / renderTarget2 is the GBuffer's own target
  // (never a clone: the velocity / normal getters must see what the scene pass drew) and writeBuffer a spare that no
  // pass writes (no pass swaps; the final pass draws to the screen), so it is never allocated.
  _buildGBuffer(on) {
    const c = this.composer, W = this._W, H = this._H;
    if (on) {
      // TAA's depth reprojection / disocclusion and the final pass linearise a standard depth buffer
      if (this.renderer.capabilities.reversedDepthBuffer) throw new Error('[PostFX] G-buffer / TAA need a standard depth buffer (reversedDepthBuffer is on)');
      if (!gbufferChunks) {
        if (installGBufferChunks() === false) throw new Error('[PostFX] G-buffer: installGBufferChunks() failed (three.js shader chunks changed)');
        gbufferChunks = true;
      }
      const gb = new GBuffer(this.renderer, { width: W, height: H });
      if (!gb.supported) { // no EXT_color_buffer_float (GBuffer warned): stay on the forward path, aa 'taa' resolves to 'auto'
        gb.dispose();
        console.warn('[PostFX] G-buffer unavailable on this renderer: forward scene pass kept');
        return false;
      }
      this.gbuffer = gb;
      gb.sync(this.scene);
      for (let i = 0; i < this._tracked.length; i++) gb.track(this._tracked[i], this._trackOpts[i]);
      for (const pool of this._trackedPools) gb.trackPool(pool);
      c.renderTarget1.dispose(); c.renderTarget2.dispose();
      const spare = this._spare = new THREE.WebGLRenderTarget(W, H, { type: THREE.HalfFloatType, depthBuffer: false, stencilBuffer: false });
      spare.texture.name = 'PostFX.spare';
      c.renderTarget1 = c.writeBuffer = spare;
      this._bindGBuffer();
      this.history = new CameraHistory(this.camera, { pattern: this.jitterPattern, jitter: false }); // _applyAA sets jitter
      this.taaPass = new TAAPass(this.camera, this.history, gb, this.taaOptions);
      this.taaPass.enabled = false;
      c.insertPass(this.taaPass, c.passes.indexOf(this.bloomPass));
      this._passKeys.set(this.taaPass, 'taa');
    } else {
      c.removePass(this.taaPass);
      this._passKeys.delete(this.taaPass);
      this.taaPass.dispose();
      this.taaPass = null; this.history = null;
      this.gbuffer.dispose(); this.gbuffer = null;
      this._spare.dispose(); this._spare = null;
      const rt = this._makeSceneTarget();
      c.renderTarget1 = c.writeBuffer = rt;
      c.renderTarget2 = c.readBuffer = rt.clone();
      c.renderTarget2.texture.name = 'EffectComposer.rt2';
      if (this.aa === 'taa') this.aa = 'auto';
    }
  }

  _bindGBuffer() {
    const c = this.composer;
    c.renderTarget2 = c.readBuffer = this.gbuffer.target;
  }

  _clearGBuffer() {
    if (this._gbClearing) return;
    this._gbClearing = true;
    try { this.gbuffer.clear(this.renderer); } finally { this._gbClearing = false; }
  }

  // readBuffer for the bloom pass: itself, or (TAA on) a stand-in whose texture is the TAA resolve
  _source(readBuffer) {
    const t = this.taaPass;
    if (t === null || !t.enabled) return readBuffer;
    const s = this._srcProxy;
    s.texture = t.texture; s.depthTexture = readBuffer.depthTexture;
    return s;
  }
}

function checkOption(v, allowed, what) {
  if (!allowed.includes(v)) throw new Error(`[PostFX] ${what}: unknown value '${v}' (${allowed.map((a) => `'${a}'`).join(' | ')})`);
  return v;
}
