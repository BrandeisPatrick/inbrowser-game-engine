// Post-processing pipeline (EffectComposer from three/addons):
//   RenderPass  -> scene into a 4x MSAA HDR target (R11G11B10F, RGBA16F fallback) with a resolved depth texture
//   ExposurePass-> 64x64 log-luminance pyramid -> adapted 1x1 luminance (eye adaptation, GPU only)
//   AOPass      -> GTAO from that depth at half resolution + bilateral blur (no second scene render)
//   BloomPass   -> UnrealBloom at quarter resolution with an HDR soft-knee bright pass (not blended here)
//   FinalPass   -> one full-resolution shader: AO (bilinear or depth-aware upsample), height/distance fog with
//                  sky-coloured inscatter, bloom add, exposure + ACES filmic (same curve as OutputPass), colour
//                  grading (contrast, saturation, split toning), vignette, edge chromatic aberration, night grain,
//                  dither, sRGB. Writes the scene depth to the default framebuffer so the lens flare can occlude.
// Then the Atmosphere's lens-flare overlay. Anti-aliasing: 4x MSAA on the scene target by day (alpha-to-coverage
// foliage, thin bridge cables), FXAA inside the final pass once it is dark or the drawing buffer is larger than
// msaaMaxPixels (see _applyAA).
// API: render(dt), setSize(w, h, pixelRatio), setTimeOfDay(darkness), setEnabled(bool), setMSAA(mode, maxPixels),
// setAOResolution({ scale, maxPixels, upsample }), debug, dispose().
// Options (defaults are the Roadster tunings): atmosphere, width/height/pixelRatio, samples, msaa ('auto' | n),
// msaaMaxPixels, lateClear, compactHDR, grade (keyframes by sun elevation, see DEFAULT_GRADE), ao { radius,
// thickness, samples, resolutionScale, maxPixels, upsample, compact, fadeNear, fadeFar } (see AOPass), bloom
// { strength, radius, threshold, resolutionScale }, fog (bool), fogFarFade (fraction of camera.far where the far
// fade starts), fogMinSkyElevation, autoExposure.
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { Pass, FullScreenQuad } from 'three/addons/postprocessing/Pass.js';
import { AOPass } from './passes/AOPass.js';
import { BloomPass } from './passes/BloomPass.js';
import { ExposurePass } from './passes/ExposurePass.js';

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
    uniform samplerCube tSky;
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

    // optical depth through an exponential layer (density d at height 0, falloff k) from y0 to y1 over dist
    float layer( float y0, float y1, float dist, float d, float k ) {
      float x = k * ( y1 - y0 );
      float f = abs( x ) > 1e-3 ? ( 1.0 - exp( -x ) ) / x : 1.0 - 0.5 * x;
      return d * exp( -k * clamp( y0, -100.0, 4000.0 ) ) * dist * f;
    }

    void main() {
      float depth = texture2D( tDepth, vUv ).x;
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
          vec3 fogCol = textureCube( tSky, fd ).rgb;
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
  }
  setSize(w, h) { this.material.uniforms.uRes.value.set(w, h); }
  render(renderer, writeBuffer, readBuffer) {
    const u = this.material.uniforms, fx = this.fx;
    const up = fx.aoPass.upsample === 'depth';
    if (up !== this._aoUp) { // a define, not a branch: nothing extra in 'linear' mode; one cached program per mode
      this._aoUp = up;
      if (up) this.material.defines.AO_UPSAMPLE_DEPTH = 1; else delete this.material.defines.AO_UPSAMPLE_DEPTH;
      this.material.needsUpdate = true;
    }
    u.tColor.value = readBuffer.texture;
    u.tDepth.value = readBuffer.depthTexture;
    u.uHasAO.value = fx.aoPass.enabled ? 1 : 0;
    u.tAO.value = fx.aoPass.aoTexture;
    u.uHasBloom.value = fx.bloomPass.enabled ? 1 : 0;
    u.tBloom.value = fx.bloomPass.bloomTexture;
    const sky = fx.atmo && fx.atmo.skyCube;
    u.uHasFog.value = sky && fx.fog ? 1 : 0;
    u.tSky.value = sky;
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
class ScenePass extends RenderPass {
  constructor(scene, camera, fx) { super(scene, camera); this.fx = fx; }
  render(renderer, writeBuffer, readBuffer, deltaTime, maskActive) {
    const plain = this.clear && !this.clearDepth && this.overrideMaterial === null && this.clearColor === null && this.clearAlpha === null;
    if (!this.fx.lateClear || !plain) { super.render(renderer, writeBuffer, readBuffer, deltaTime, maskActive); return; }
    const ac = renderer.autoClear;
    renderer.autoClear = true;
    renderer.setRenderTarget(this.renderToScreen ? null : readBuffer);
    renderer.render(this.scene, this.camera);
    renderer.autoClear = ac;
  }
}

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
   * @param {object} [o] atmosphere (an Atmosphere), width/height (CSS px), pixelRatio, samples (MSAA, default 4),
   *   msaa, msaaMaxPixels, lateClear, compactHDR, grade, ao, bloom, fog, fogFarFade, fogMinSkyElevation,
   *   autoExposure (see header)
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
    const samples = this.maxSamples;

    const W = Math.max(1, Math.floor(this.width * this.pixelRatio)), H = Math.max(1, Math.floor(this.height * this.pixelRatio));
    this._W = W; this._H = H; // drawing-buffer size, for msaaMaxPixels
    const depthTexture = new THREE.DepthTexture(W, H, THREE.UnsignedIntType);
    // R11G11B10 float halves the MSAA bandwidth of RGBA16F (~1 ms at 1500x1300 on an M4); needs EXT_color_buffer_float
    const small = o.compactHDR !== false && renderer.extensions.has('EXT_color_buffer_float');
    const fmt = small ? { format: THREE.RGBFormat, type: THREE.UnsignedInt101111Type } : { type: THREE.HalfFloatType };
    const rt = new THREE.WebGLRenderTarget(W, H, { ...fmt, samples, depthBuffer: true, stencilBuffer: false, depthTexture });
    rt.texture.name = 'PostFX.scene';
    rt.texture.generateMipmaps = false;
    this.composer = new EffectComposer(renderer, rt);

    this.renderPass = new ScenePass(scene, camera, this);
    this.exposurePass = new ExposurePass();
    // AO defaults: half resolution capped at 0.6 Mpx (a 14-inch Retina MacBook's half-res buffer is 1.5 Mpx: -1.3 to
    // -1.7 ms measured at 3024x1964, full-res mean pixel change <= 0.04 / 255 and < 0.5 % of pixels over 2 / 255 with
    // the depth-aware upsample; 1080p-class buffers are under the cap and unchanged), compact RG16F buffers
    this.aoPass = new AOPass(camera, { radius: 1.8, thickness: 1.6, samples: 12, resolutionScale: 0.5, maxPixels: 0.6e6, upsample: 'depth', compact: true, ...(o.ao || {}) });
    this.bloomPass = new BloomPass({ strength: 0.3, radius: 0.45, threshold: 2.8, resolutionScale: 0.5, ...(o.bloom || {}) });
    this.finalPass = new FinalPass(this);
    for (const p of [this.renderPass, this.exposurePass, this.aoPass, this.bloomPass, this.finalPass]) this.composer.addPass(p);
    this.uniforms = this.finalPass.material.uniforms;

    // the composer works in drawing-buffer pixels (floored exactly like the canvas), so no fractional targets
    this.composer.setPixelRatio(1);
    this.composer.setSize(W, H);
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

  /** CSS size + pixel ratio, exactly as given to renderer.setPixelRatio / setSize (call after those). */
  setSize(w, h, pixelRatio = this.pixelRatio) {
    this.width = w; this.height = h; this.pixelRatio = pixelRatio;
    const W = Math.max(1, Math.floor(w * pixelRatio)), H = Math.max(1, Math.floor(h * pixelRatio));
    this._W = W; this._H = H;
    this.composer.setSize(W, H);
    this._applyAA(this._aaDark); // msaaMaxPixels: the new size may change 'auto' (a no-op while it is Infinity)
    if (this.atmo && this.atmo.setFlareScale) this.atmo.setFlareScale(H);
  }

  setEnabled(on) {
    this.enabled = !!on;
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

  _applyAA(darkness = this.atmo ? this.atmo.darkness : 0) {
    this._aaDark = darkness; // setSize re-applies the size rule with the same darkness
    const want = this.msaa === 'auto'
      ? (darkness > 0.35 || this._W * this._H > this.msaaMaxPixels ? 0 : this.maxSamples)
      : Math.min(this.maxSamples, this.msaa | 0);
    for (const rt of [this.composer.renderTarget1, this.composer.renderTarget2]) {
      if (rt.samples !== want) { rt.samples = want; rt.dispose(); }
    }
    this.uniforms.uFXAA.value = want > 0 ? 0 : 1;
  }

  /**
   * Anti-aliasing: 'auto' (default), or a fixed MSAA sample count (0 = FXAA only). maxPixels (optional) replaces
   * msaaMaxPixels, the drawing-buffer pixel count above which 'auto' uses FXAA (Infinity / null: no limit).
   */
  setMSAA(mode, maxPixels) {
    this.msaa = mode;
    if (maxPixels !== undefined) this.msaaMaxPixels = maxPixels ?? Infinity;
    this._applyAA();
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
    }
  }

  render(dt) {
    const r = this.renderer;
    if (!this.enabled) {
      r.render(this.scene, this.camera);
      if (this.atmo) this.atmo.renderFlare(r, this.camera);
      return;
    }
    this.camera.updateMatrixWorld();
    this._syncFrame(dt);
    const info = r.info, ar = info.autoReset;
    info.autoReset = false; info.reset();
    this.composer.render(dt);
    if (this.atmo) this.atmo.renderFlare(r, this.camera);
    info.autoReset = ar;
  }

  dispose() {
    this.composer.dispose();
    this.aoPass.dispose(); this.bloomPass.dispose(); this.finalPass.dispose(); this.exposurePass.dispose();
    if (this.atmo) this.atmo.setFogMode('scene');
  }
}
