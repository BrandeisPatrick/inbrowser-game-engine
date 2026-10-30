// Depth of field from a thin-lens model: gather-based near / far fields at reduced resolution, composited over the
// sharp full-resolution colour. PostFX order: ... -> TAA -> MotionBlur -> DoF -> Bloom -> Final; render() reads
// readBuffer.texture (the compositor's colorSource proxy) and readBuffer.depthTexture (the scene depth, standard 0..1
// perspective depth, linearised with camera.near / far). Opt-in: the compositor adds the pass only when asked for.
//
// Circle of confusion (thin lens): an object at d (m) with focus distance D (m), focal length f (mm), aperture
// A = f / fStop (mm), all in mm: cocMm = A (d - D) f / (d (D - f)), signed (negative = in front of the focus plane =
// near field), = K (1 - D / d) with K = A f / (D - f) the CoC of a point at infinity. The image's vertical extent is the
// sensor's: px = cocMm * bufferHeight / sensorHeight (drawing-buffer px; the camera's own fov is not consulted).
// circleOfConfusion() is that function in JS (tests, autofocus read-outs); the shader evaluates K (1 - D / z) with K in
// px from the CPU, clamped to +-maxCoC.
//
//   1. Prefilter (RGBA16F at `scale`, 0.5 by default): per texel the 2x2 full-resolution footprint (4 depth + 4 colour
//      fetches at every scale, bilinear weights, so at scale 1 one texel has all the weight): CoC of each; the near
//      CoC is kept whenever one of the footprint's taps is near-blurred (min <= -0.5 px), else the largest far CoC
//      (-min >= 0.5 ? min : max), so a near silhouette keeps its coverage in the near gather even against a more
//      defocused background (the far gather gives near taps no weight, so the far side loses nothing); the colours
//      averaged with weights bilinear * (max(coc_i * sign(coc), 0) + 1e-3), i.e. favouring texels on the kept CoC's
//      side, so an in-focus edge does not tint the blurred layer.
//      rgb = colour, a = signed CoC (full-resolution px). One RGBA16F fetch per gather tap carries both.
//   2. Near-field CoC dilation (nearBlur): tile max of the near CoC (max(-coc, 0)) over T x T prefilter texels,
//      T = ceil(maxCoC * scale) (RG16F, ceil(w' / T) x ceil(h' / T)), then the max over the 3x3 tiles: the radius R of
//      the largest near disc that can reach each texel.
//   3. Far field (RGBA16F at `scale`): far CoC c0 = max(coc, 0) of the texel; below 0.5 px it outputs the prefilter
//      colour (the composite keeps the sharp texel there); else a `samples`-tap Vogel (golden-angle) disc of radius c0,
//      each tap texel-snapped and weighted clamp(coc_tap - r + 1) (scatter-as-gather: a tap counts only where its own
//      disc reaches this texel, so nearer, sharper samples do not bleed into the background), centre weight 1;
//      rgb = weighted mean. The +1 softness lets an in-focus tap count within one texel (a ~2 px ring of the sharp
//      foreground in the blurred background at scale 0.5): a knob to measure (+0.5 halves it), not changed blind.
//   4. Near field (nearBlur, RGBA16F at `scale`, premultiplied): 0 where R < 0.5 px; else the same disc at radius R,
//      tap weight clamp(n - r + 1) * smoothstep(0.5, 2, n) * R^2 / (samples * max(n, 0.5)^2) (n = the tap's near CoC:
//      each near point spreads its colour over its own disc area, so a near object fully covering the disc sums to
//      alpha 1 and its blurred silhouette fades over sharp pixels beyond it); alpha = min(sum, 1), rgb = colour * alpha.
//   5. Composite (full resolution, RGBA16F): coc of the pixel from its own depth (stableDepth: the closest depth of
//      the pixel and its 4 diagonals, +4 reads, so a silhouette does not flip between sharp and blurred with the TAA
//      jitter; the price is a 1-px sharp rim of background around in-focus objects); where |coc| < 0.5 px and the near
//      field's alpha is 0 the input texel itself, bit-exact (a far focus / high f-number gives byte-identical output);
//      else mix(sharp, far field, smoothstep(0.5, 2, coc)), then the near field over it: col * (1 - a) + near.rgb.
//      The fields are upsampled bilinearly; alpha = the input's alpha.
// Reads per full-resolution pixel (scale 0.5, samples 24): prefilter 8 per texel = 2; near tile max 0.25; far gather
// 25 per texel = 6.25 (in focus / near: 0.25); near gather 6.25 (nothing near: 0.25); composite 3 (colour, depth,
// near) + 1 far where coc > 0.5 (+4 with stableDepth). Worst case ~18.75, a sharp frame ~6; nearBlur false drops the
// near tile, near gather and near composite reads (~11.25).
// Every target is cleared before its draw (autoClear off around the pass, renderer.clear(colour) per target, autoClear
// restored): the draws overwrite every texel, and a load-action clear is free on tile-based GPUs where loading the
// previous contents is not (passes before this one leave autoClear false).
// Options (DOF_DEFAULTS; live properties, validated: unknown keys, bad numbers and non-boolean flags throw):
// focusDistance 12 (m, > focalLength / 1000), focalLength 50 (mm), fStop 2.0, sensorHeight 24 (mm), maxCoC 12 (px at
// full resolution, 0 < maxCoC <= 64), scale 0.5 (resolution of the fields, 0 < scale <= 1), samples 24 (disc taps,
// integer 4..64; a define: changing it recompiles), nearBlur true (boolean; false: far field only, cheaper),
// stableDepth false (boolean; see 5; off: each pixel's own depth).
// The lens options are plain numbers read in render(): animate focusDistance freely (autofocus).
// API: render(renderer, writeBuffer, readBuffer), setSize(w, h) (drawing-buffer px), texture (output), farTexture /
// nearTexture / prefilterTexture (debug views), cocAt(d) (px, current lens and buffer), reset() (no-op), dispose();
// circleOfConfusion(), vogelDisc().
// Perspective cameras and a standard depth buffer (not reversedDepthBuffer / logarithmicDepthBuffer).
import * as THREE from 'three';
import { Pass, FullScreenQuad } from 'three/addons/postprocessing/Pass.js';

/** Default options (see the header). */
export const DOF_DEFAULTS = Object.freeze({ focusDistance: 12, focalLength: 50, fStop: 2.0, sensorHeight: 24, maxCoC: 12, scale: 0.5, samples: 24, nearBlur: true, stableDepth: false });

const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));

/**
 * Thin-lens circle of confusion in pixels, signed: negative in front of the focus plane (near field), positive behind
 * it. cocMm = A (d - D) f / (d (D - f)) with A = f / fStop, then * bufferHeight / sensorHeight. d = Infinity gives the
 * CoC of a point at infinity. No clamping (the pass clamps to maxCoC).
 * @param {number} d object distance (m)
 * @param {number} focusDistance D (m)
 * @param {number} focalLength f (mm)
 * @param {number} fStop N (aperture diameter f / N)
 * @param {number} sensorHeight mm spanned by the frame's height
 * @param {number} bufferHeight px spanned by the frame's height
 * @returns {number} px
 */
export function circleOfConfusion(d, focusDistance, focalLength, fStop, sensorHeight, bufferHeight) {
  const f = focalLength, D = focusDistance * 1000, A = f / fStop;
  return (A * f / (D - f)) * (1 - D / (d * 1000)) * bufferHeight / sensorHeight;
}

/**
 * Vogel (golden-angle) disc of n points inside the unit disc, flat [x0, y0, x1, y1, ...]: point i at radius
 * sqrt((i + 0.5) / n), angle i * the golden angle (even area coverage, no rings).
 * @param {number} n
 * @param {Float32Array} [out]
 * @returns {Float32Array}
 */
export function vogelDisc(n, out = new Float32Array(2 * n)) {
  for (let i = 0; i < n; i++) {
    const r = Math.sqrt((i + 0.5) / n), a = i * GOLDEN_ANGLE;
    out[2 * i] = r * Math.cos(a); out[2 * i + 1] = r * Math.sin(a);
  }
  return out;
}

const VERTEX = /* glsl */`
  varying vec2 vUv;
  void main() { vUv = uv; gl_Position = vec4( position.xy, 0.0, 1.0 ); }`;

// signed CoC (full-resolution px) of a standard 0..1 depth: uCoc = (K px, focus distance m, maxCoC px)
const COC_GLSL = /* glsl */`
  uniform vec3 uCoc;
  uniform float uNear, uFar;
  float dofCoc( float d ) {
    float z = uNear * uFar / ( uFar - ( uFar - uNear ) * d );
    return clamp( uCoc.x * ( 1.0 - uCoc.y / z ), - uCoc.z, uCoc.z );
  }`;

// 1. prefilter: rgb = colour of the footprint's kept layer, a = signed CoC
const PREFILTER_FRAG = /* glsl */`
  uniform sampler2D tColor;
  uniform highp sampler2D tDepth;
  uniform vec2 uRatio; // full-resolution px per prefilter texel
  ${COC_GLSL}

  void main() {
    vec2 P = gl_FragCoord.xy * uRatio - 0.5, f = fract( P ); // bilinear footprint of this texel's centre
    ivec2 mx = textureSize( tColor, 0 ) - 1, i0 = clamp( ivec2( floor( P ) ), ivec2( 0 ), mx ), i1 = min( i0 + 1, mx );
    ivec2 p1 = ivec2( i1.x, i0.y ), p2 = ivec2( i0.x, i1.y );
    vec4 bw = vec4( ( 1.0 - f.x ) * ( 1.0 - f.y ), f.x * ( 1.0 - f.y ), ( 1.0 - f.x ) * f.y, f.x * f.y );
    vec4 c = vec4( dofCoc( texelFetch( tDepth, i0, 0 ).x ), dofCoc( texelFetch( tDepth, p1, 0 ).x ),
                   dofCoc( texelFetch( tDepth, p2, 0 ).x ), dofCoc( texelFetch( tDepth, i1, 0 ).x ) );
    vec4 cs = c * step( 1e-6, bw ); // taps outside the footprint (weight 0) take no part in the choice
    float lo = min( min( cs.x, cs.y ), min( cs.z, cs.w ) ), hi = max( max( cs.x, cs.y ), max( cs.z, cs.w ) );
    float coc = - lo >= 0.5 ? lo : hi; // near wins once any tap is near-blurred
    vec4 w = bw * ( max( c * sign( coc ), 0.0 ) + 1e-3 );
    vec3 col = texelFetch( tColor, i0, 0 ).rgb * w.x + texelFetch( tColor, p1, 0 ).rgb * w.y
             + texelFetch( tColor, p2, 0 ).rgb * w.z + texelFetch( tColor, i1, 0 ).rgb * w.w;
    gl_FragColor = vec4( col / ( w.x + w.y + w.z + w.w ), coc );
  }`;

// 2a. near CoC tile max over uTile x uTile prefilter texels
const NEAR_TILE_FRAG = /* glsl */`
  uniform sampler2D tPre;
  uniform int uTile;

  void main() {
    ivec2 o = ivec2( gl_FragCoord.xy ) * uTile, e = min( o + uTile, textureSize( tPre, 0 ) );
    float m = 0.0;
    for ( int y = o.y; y < e.y; y ++ ) {
      for ( int x = o.x; x < e.x; x ++ ) m = max( m, - texelFetch( tPre, ivec2( x, y ), 0 ).a );
    }
    gl_FragColor = vec4( m, 0.0, 0.0, 1.0 );
  }`;

// 2b. max over the 3x3 tiles
const NEAR_DILATE_FRAG = /* glsl */`
  uniform sampler2D tTile;

  void main() {
    ivec2 o = ivec2( gl_FragCoord.xy ), mx = textureSize( tTile, 0 ) - 1;
    float m = 0.0;
    for ( int y = - 1; y <= 1; y ++ ) {
      for ( int x = - 1; x <= 1; x ++ ) m = max( m, texelFetch( tTile, clamp( o + ivec2( x, y ), ivec2( 0 ), mx ), 0 ).r );
    }
    gl_FragColor = vec4( m, 0.0, 0.0, 1.0 );
  }`;

// 3. far field gather
const FAR_FRAG = /* glsl */`
  uniform sampler2D tPre;
  uniform float uScale; // prefilter texels per full-resolution px
  uniform vec2 uKernel[ DOF_SAMPLES ];

  void main() {
    ivec2 ip = ivec2( gl_FragCoord.xy ), mx = textureSize( tPre, 0 ) - 1;
    vec4 c0 = texelFetch( tPre, ip, 0 );
    if ( c0.a < 0.5 ) { gl_FragColor = vec4( c0.rgb, 1.0 ); return; } // not far-blurred: the composite keeps the sharp texel
    float R = c0.a * uScale;
    vec3 sum = c0.rgb;
    float ws = 1.0;
    for ( int i = 0; i < DOF_SAMPLES; i ++ ) {
      vec2 o = uKernel[ i ] * R;
      vec4 t = texelFetch( tPre, clamp( ivec2( floor( gl_FragCoord.xy + o ) ), ivec2( 0 ), mx ), 0 );
      float w = clamp( t.a * uScale - length( o ) + 1.0, 0.0, 1.0 ); // the tap's own disc reaches this texel
      sum += t.rgb * w;
      ws += w;
    }
    gl_FragColor = vec4( sum / ws, 1.0 );
  }`;

// 4. near field gather (premultiplied)
const NEAR_FRAG = /* glsl */`
  uniform sampler2D tPre;
  uniform sampler2D tNearMax;
  uniform int uTile;
  uniform float uScale;
  uniform vec2 uKernel[ DOF_SAMPLES ];

  void main() {
    ivec2 ip = ivec2( gl_FragCoord.xy ), mx = textureSize( tPre, 0 ) - 1;
    float R = texelFetch( tNearMax, ip / uTile, 0 ).r; // largest near CoC around (full-resolution px)
    if ( R < 0.5 ) { gl_FragColor = vec4( 0.0 ); return; }
    float Rs = R * uScale, k = Rs * Rs / float( DOF_SAMPLES ), hs = 0.5 * uScale;
    vec3 sum = vec3( 0.0 );
    float a = 0.0;
    for ( int i = 0; i < DOF_SAMPLES; i ++ ) {
      vec2 o = uKernel[ i ] * Rs;
      vec4 t = texelFetch( tPre, clamp( ivec2( floor( gl_FragCoord.xy + o ) ), ivec2( 0 ), mx ), 0 );
      float n = max( - t.a * uScale, hs ); // the tap's near CoC radius in texels (>= 0.5 px)
      float w = clamp( n - length( o ) + 1.0, 0.0, 1.0 ) * smoothstep( 0.5, 2.0, - t.a ) * k / ( n * n );
      sum += t.rgb * w;
      a += w;
    }
    gl_FragColor = vec4( sum / max( a, 1.0 ), min( a, 1.0 ) );
  }`;

// 5. composite at full resolution
const COMPOSITE_FRAG = /* glsl */`
  uniform sampler2D tColor;
  uniform highp sampler2D tDepth;
  uniform sampler2D tFar;
  #ifdef DOF_NEAR
  uniform sampler2D tNear;
  #endif
  varying vec2 vUv;
  ${COC_GLSL}

  void main() {
    ivec2 ip = ivec2( gl_FragCoord.xy );
    vec4 c = texelFetch( tColor, ip, 0 );
    float d = texelFetch( tDepth, ip, 0 ).x;
  #ifdef DOF_STABLE_DEPTH
    // under TAA jitter: the closest depth of the pixel and its diagonals, so a silhouette keeps one CoC every frame
    ivec2 mx = textureSize( tDepth, 0 ) - 1;
    d = min( d, texelFetch( tDepth, clamp( ip + ivec2( - 1, - 1 ), ivec2( 0 ), mx ), 0 ).x );
    d = min( d, texelFetch( tDepth, clamp( ip + ivec2( 1, - 1 ), ivec2( 0 ), mx ), 0 ).x );
    d = min( d, texelFetch( tDepth, clamp( ip + ivec2( - 1, 1 ), ivec2( 0 ), mx ), 0 ).x );
    d = min( d, texelFetch( tDepth, clamp( ip + ivec2( 1, 1 ), ivec2( 0 ), mx ), 0 ).x );
  #endif
    float coc = dofCoc( d );
  #ifdef DOF_NEAR
    vec4 nf = texture2D( tNear, vUv );
  #else
    vec4 nf = vec4( 0.0 );
  #endif
    if ( abs( coc ) < 0.5 && nf.a <= 0.0 ) { gl_FragColor = c; return; } // in focus, nothing blurred in front: the input texel
    vec3 col = c.rgb;
    if ( coc > 0.5 ) col = mix( col, texture2D( tFar, vUv ).rgb, smoothstep( 0.5, 2.0, coc ) );
    gl_FragColor = vec4( col * ( 1.0 - nf.a ) + nf.rgb, c.a );
  }`;

const KEYS = Object.keys(DOF_DEFAULTS);

/**
 * @typedef {object} DoFOptions
 * @property {number} [focusDistance=12] m
 * @property {number} [focalLength=50] mm
 * @property {number} [fStop=2]
 * @property {number} [sensorHeight=24] mm
 * @property {number} [maxCoC=12] px at full resolution
 * @property {number} [scale=0.5] resolution of the blur fields
 * @property {number} [samples=24] disc taps
 * @property {boolean} [nearBlur=true]
 * @property {boolean} [stableDepth=false] composite CoC from the closest of 5 depths (under TAA jitter)
 */

export class DoFPass extends Pass {
  /**
   * @param {THREE.PerspectiveCamera} camera near / far linearise the depth
   * @param {DoFOptions} [o] see the header
   */
  constructor(camera, o = {}) {
    super();
    for (const k in o) if (!KEYS.includes(k)) throw new Error(`[DoFPass] unknown option '${k}' (${KEYS.join(', ')})`);
    this.needsSwap = false;
    this.camera = camera;
    const d = DOF_DEFAULTS;
    this._focus = positive(o.focusDistance ?? d.focusDistance, 'focusDistance');
    this._focal = positive(o.focalLength ?? d.focalLength, 'focalLength');
    lensCheck(this._focus, this._focal);
    this._fStop = positive(o.fStop ?? d.fStop, 'fStop');
    this._sensor = positive(o.sensorHeight ?? d.sensorHeight, 'sensorHeight');
    this._maxCoC = maxCoc(o.maxCoC ?? d.maxCoC);
    this._scale = unitScale(o.scale ?? d.scale);
    this._samples = integer(o.samples ?? d.samples, 4, 64, 'samples');
    this._nearBlur = flag(o.nearBlur ?? d.nearBlur, 'nearBlur');
    this._stableDepth = flag(o.stableDepth ?? d.stableDepth, 'stableDepth');
    this._nearDefined = null; // DOF_NEAR defined in the composite, synced in render()
    this._stableDefined = null; // DOF_STABLE_DEPTH defined in the composite, synced in render()
    this._w = 1; this._h = 1;
    this._tile = 1;

    const rgba = { type: THREE.HalfFloatType, depthBuffer: false, stencilBuffer: false, generateMipmaps: false };
    const target = (name, opt) => { const rt = new THREE.WebGLRenderTarget(1, 1, opt); rt.texture.name = name; return rt; };
    this._pre = target('DoF.prefilter', { ...rgba, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter });
    const rg = { ...rgba, format: THREE.RGFormat, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter };
    this._nearTile = target('DoF.nearTileMax', rg);
    this._nearMax = target('DoF.nearDilated', rg);
    this._far = target('DoF.far', { ...rgba, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter });
    this._near = target('DoF.near', { ...rgba, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter });
    this._out = target('DoF.out', { ...rgba, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter });

    // shared uniform objects (prefilter + composite: CoC; far + near: kernel, scale)
    this._uCoc = { value: new THREE.Vector3() };
    this._uNear = { value: 0.1 };
    this._uFar = { value: 1000 };
    this._uKernel = { value: vogelDisc(this._samples) };
    this._uScale = { value: 0.5 };
    const mat = (name, fragmentShader, uniforms, defines = {}) => new THREE.ShaderMaterial({
      name, defines, uniforms, vertexShader: VERTEX, fragmentShader,
      blending: THREE.NoBlending, depthTest: false, depthWrite: false,
    });
    const coc = { uCoc: this._uCoc, uNear: this._uNear, uFar: this._uFar };
    this._prefilterMat = mat('PostFXDoFPrefilter', PREFILTER_FRAG, { tColor: { value: null }, tDepth: { value: null }, uRatio: { value: new THREE.Vector2(2, 2) }, ...coc });
    this._tileMat = mat('PostFXDoFNearTile', NEAR_TILE_FRAG, { tPre: { value: this._pre.texture }, uTile: { value: 1 } });
    this._dilateMat = mat('PostFXDoFNearDilate', NEAR_DILATE_FRAG, { tTile: { value: this._nearTile.texture } });
    const gather = { DOF_SAMPLES: this._samples };
    this._farMat = mat('PostFXDoFFar', FAR_FRAG, { tPre: { value: this._pre.texture }, uScale: this._uScale, uKernel: this._uKernel }, { ...gather });
    this._nearMat = mat('PostFXDoFNear', NEAR_FRAG, {
      tPre: { value: this._pre.texture }, tNearMax: { value: this._nearMax.texture }, uTile: { value: 1 }, uScale: this._uScale, uKernel: this._uKernel,
    }, { ...gather });
    this._compositeMat = mat('PostFXDoF', COMPOSITE_FRAG, {
      tColor: { value: null }, tDepth: { value: null }, tFar: { value: this._far.texture }, tNear: { value: this._near.texture }, ...coc,
    });
    this._quad = new FullScreenQuad(null);
    this._resize(); // option-dependent sizes / uniforms right away (setSize() repeats it at the real size)
  }

  /** The composited colour of the last render (RGBA16F, full resolution; alpha = the input's alpha). */
  get texture() { return this._out.texture; }
  /** Debug: far field (rgb) at `scale`. */
  get farTexture() { return this._far.texture; }
  /** Debug: near field (premultiplied rgb, alpha = coverage) at `scale`. */
  get nearTexture() { return this._near.texture; }
  /** Debug: prefiltered colour (rgb) + signed CoC in full-resolution px (a) at `scale`. */
  get prefilterTexture() { return this._pre.texture; }

  /** Focus distance (m, > focalLength / 1000). */
  get focusDistance() { return this._focus; }
  set focusDistance(v) { lensCheck(positive(v, 'focusDistance'), this._focal); this._focus = v; }
  /** Focal length (mm). */
  get focalLength() { return this._focal; }
  set focalLength(v) { lensCheck(this._focus, positive(v, 'focalLength')); this._focal = v; }
  /** f-number: aperture diameter = focalLength / fStop. */
  get fStop() { return this._fStop; }
  set fStop(v) { this._fStop = positive(v, 'fStop'); }
  /** Sensor height (mm) mapped onto the drawing buffer's height. */
  get sensorHeight() { return this._sensor; }
  set sensorHeight(v) { this._sensor = positive(v, 'sensorHeight'); }
  /** CoC clamp (px at full resolution, 0 < maxCoC <= 64); sets the near tile size, resizes the tile targets. */
  get maxCoC() { return this._maxCoC; }
  set maxCoC(v) { this._maxCoC = maxCoc(v); this._resize(); }
  /** Resolution of the blur fields (0 < scale <= 1). */
  get scale() { return this._scale; }
  set scale(v) { this._scale = unitScale(v); this._resize(); }
  /** Near field (blur in front of the focus plane spreading over sharp pixels); false = far field only. */
  get nearBlur() { return this._nearBlur; }
  set nearBlur(v) { this._nearBlur = flag(v, 'nearBlur'); }
  /** Composite CoC from the closest depth of the pixel and its 4 diagonals (set it while TAA jitters the depth). */
  get stableDepth() { return this._stableDepth; }
  set stableDepth(v) { this._stableDepth = flag(v, 'stableDepth'); }
  /** Disc taps of the gathers (integer 4..64; recompiles both gather shaders). */
  get samples() { return this._samples; }
  set samples(v) {
    v = integer(v, 4, 64, 'samples');
    if (v === this._samples) return;
    this._samples = v;
    this._uKernel.value = vogelDisc(v);
    for (const m of [this._farMat, this._nearMat]) { m.defines.DOF_SAMPLES = v; m.needsUpdate = true; }
  }

  /** Signed CoC in px of an object at d metres with the current lens and drawing-buffer height (unclamped). */
  cocAt(d) { return circleOfConfusion(d, this._focus, this._focal, this._fStop, this._sensor, this._h); }

  /** No history: nothing to reset. */
  reset() {}

  /**
   * @param {number} width drawing-buffer px
   * @param {number} height drawing-buffer px
   */
  setSize(width, height) {
    this._w = Math.max(1, Math.floor(width) || 1);
    this._h = Math.max(1, Math.floor(height) || 1);
    this._resize();
  }

  /**
   * @param {THREE.WebGLRenderer} renderer
   * @param {THREE.WebGLRenderTarget|null} writeBuffer unused
   * @param {{texture: THREE.Texture, depthTexture: THREE.Texture}} readBuffer colour + scene depth
   */
  render(renderer, writeBuffer, readBuffer) {
    const cam = this.camera, near = this._nearBlur, stable = this._stableDepth;
    if (near !== this._nearDefined || stable !== this._stableDefined) { // defines: one cached program per combination
      this._nearDefined = near; this._stableDefined = stable;
      const d = this._compositeMat.defines;
      if (near) d.DOF_NEAR = 1; else delete d.DOF_NEAR;
      if (stable) d.DOF_STABLE_DEPTH = 1; else delete d.DOF_STABLE_DEPTH;
      this._compositeMat.needsUpdate = true;
    }
    // K = circleOfConfusion(Infinity, ...) = A f / (D - f) in px, inlined (a call returning a double boxes it)
    const f = this._focal, D = this._focus * 1000;
    this._uCoc.value.set((f / this._fStop) * f / (D - f) * this._h / this._sensor, this._focus, this._maxCoC);
    if (this._uNear.value !== cam.near) this._uNear.value = cam.near; // on change only: a stored double is boxed
    if (this._uFar.value !== cam.far) this._uFar.value = cam.far;

    const pu = this._prefilterMat.uniforms, cu = this._compositeMat.uniforms;
    pu.tColor.value = readBuffer.texture; pu.tDepth.value = readBuffer.depthTexture;
    cu.tColor.value = readBuffer.texture; cu.tDepth.value = readBuffer.depthTexture;

    const ac = renderer.autoClear;
    renderer.autoClear = false;
    this._draw(renderer, this._pre, this._prefilterMat);
    if (near) {
      this._draw(renderer, this._nearTile, this._tileMat);
      this._draw(renderer, this._nearMax, this._dilateMat);
      this._draw(renderer, this._near, this._nearMat);
    }
    this._draw(renderer, this._far, this._farMat);
    this._draw(renderer, this._out, this._compositeMat);
    renderer.autoClear = ac;
  }

  // one draw into a cleared target (autoClear is off around the pass: see the header)
  _draw(renderer, target, material) {
    renderer.setRenderTarget(target);
    renderer.clear(true, false, false);
    this._quad.material = material;
    this._quad.render(renderer);
  }

  dispose() {
    for (const rt of [this._pre, this._nearTile, this._nearMax, this._far, this._near, this._out]) rt.dispose();
    for (const m of [this._prefilterMat, this._tileMat, this._dilateMat, this._farMat, this._nearMat, this._compositeMat]) m.dispose();
    this._quad.dispose();
  }

  // every target from the drawing-buffer size, scale and maxCoC
  _resize() {
    const W = this._w, H = this._h;
    const pw = Math.max(1, Math.round(W * this._scale)), ph = Math.max(1, Math.round(H * this._scale));
    const s = ph / H; // prefilter texels per full-resolution px (the CoC is measured against the height)
    const T = this._tile = Math.max(1, Math.ceil(this._maxCoC * s));
    const tw = Math.max(1, Math.ceil(pw / T)), th = Math.max(1, Math.ceil(ph / T));
    this._pre.setSize(pw, ph);
    this._far.setSize(pw, ph);
    this._near.setSize(pw, ph);
    this._nearTile.setSize(tw, th);
    this._nearMax.setSize(tw, th);
    this._out.setSize(W, H);
    this._prefilterMat.uniforms.uRatio.value.set(W / pw, H / ph);
    this._uScale.value = s;
    this._tileMat.uniforms.uTile.value = T;
    this._nearMat.uniforms.uTile.value = T;
  }
}

function positive(v, what) {
  if (typeof v !== 'number' || !Number.isFinite(v) || v <= 0) throw new Error(`[DoFPass] ${what}: expected a finite number > 0, got ${v}`);
  return v;
}
function integer(v, lo, hi, what) {
  if (!Number.isInteger(v) || v < lo || v > hi) throw new Error(`[DoFPass] ${what}: expected an integer in ${lo}..${hi}, got ${v}`);
  return v;
}
function unitScale(v) {
  if (typeof v !== 'number' || !(v > 0 && v <= 1)) throw new Error(`[DoFPass] scale: expected a number in (0, 1], got ${v}`);
  return v;
}
function maxCoc(v) {
  if (typeof v !== 'number' || !(v > 0 && v <= 64)) throw new Error(`[DoFPass] maxCoC: expected a number in (0, 64] px, got ${v}`);
  return v;
}
function flag(v, what) {
  if (typeof v !== 'boolean') throw new Error(`[DoFPass] ${what}: expected a boolean, got ${typeof v === 'string' ? `'${v}'` : v}`);
  return v;
}
// a thin lens focuses only beyond its focal length
function lensCheck(focusDistance, focalLength) {
  if (!(focusDistance * 1000 > focalLength)) throw new Error(`[DoFPass] focusDistance (${focusDistance} m) must exceed focalLength (${focalLength} mm) / 1000`);
}
