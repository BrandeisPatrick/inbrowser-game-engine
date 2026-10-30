// Per-pixel motion blur, McGuire, Hennessy, Bukowski, Osman 2012, "A reconstruction filter for plausible motion blur"
// (I3D '12). PostFX order: ... -> TAA -> MotionBlur -> DoF -> Bloom -> Final; render() reads readBuffer.texture (the
// compositor's colorSource proxy: the TAA resolve while TAA is on, else the scene colour) and readBuffer.depthTexture
// (the scene depth). Opt-in: the compositor adds the pass only when motion blur is asked for.
//
//   1. Velocity (full resolution, one MRT draw into two RG16F attachments). Motion of the surface at each pixel, in
//      pixels: G-buffer flags bit0 (wroteGBuffer) without bit3 (noHistory) -> the velocity attachment (uvCur - uvPrev,
//      unjittered) * resolution; elsewhere (sky, materials without the G-buffer output, noHistory surfaces, whose
//      stored velocity is 0 by design, or no G-buffer at all: `gbuffer` null) the camera-only
//      motion of its depth, uv - uvPrev with uvPrev = (c.xy / c.w) * 0.5 + 0.5, c = uReproj * ndc (CameraHistory:
//      current NDC -> previous clip; 0 when c.w <= 0, behind the previous camera). The pixel position carries this
//      frame's jitter (uJitter), as in TAAPass, because the depth was drawn jittered. Stored as the paper's HALF
//      extent V = motion * k / 2, its length clamped to min(maxBlur / 2, tileSize) (the blur of a pixel spans X +- V,
//      so `maxBlur` is the longest streak and a streak never reaches past the 3x3 tile neighbourhood), with
//        k = strength                                    shutter 0 (default): frame-relative, streak = the inter-frame
//                                                        motion * strength (halves at 120 Hz against 60 Hz)
//        k = strength * clamp(shutter / dt, 0, 4)        shutter > 0 (seconds) and render()'s deltaTime dt > 0: the
//                                                        motion over `shutter` seconds, the same streak at any frame
//                                                        rate (dt <= 0 or missing, e.g. a bench render with dt 0:
//                                                        frame-relative k = strength)
//      (uBlur = (k / 2, max half extent)).
//        attachment 0  rg = V (px, signed)          -> tile max
//        attachment 1  r = |V| (px), g = linear view depth (m, standard depth linearised with near / far)
//   2. Tile max (RG16F, ceil(w / tileSize) x ceil(h / tileSize)): the V of greatest length per tileSize^2 tile, as two
//      separable passes (rows into a ceil(w / tileSize) x h target, then columns).
//   3. Neighbour max (same size): the longest V among the tile and its 8 neighbours = V_N, the dominant blur through
//      every pixel of the tile.
//   4. Reconstruction (output, RGBA16F): |V_N| < 0.25 px (a streak shorter than 0.5 px anywhere around) -> the input
//      texel itself, bit-exact (a still camera and scene give byte-identical output). Otherwise the paper's filter:
//      centre weight 1 / max(|V_X|, 0.5), then `samples` taps at Y = X + V_N t, t = mix(-1, 1, (i + j + 1) /
//      (samples + 1)) with j = interleaved-gradient noise - 0.5 (a fixed per-pixel jitter: no temporal flicker, since
//      TAA ran before), each texel-snapped and weighted
//        f * cone(r, |V_Y|) + b * cone(r, |V_X|) + 2 cylinder(r, |V_Y|) cylinder(r, |V_X|),
//      r = |Y - X| (texel-snapped, as in the paper), cone = clamp(1 - r / v), cylinder = 1 - smoothstep(0.95 v, 1.05 v,
//      r), f = soft "Y in front of X", b = soft "X in front of Y" (1 - (z_a - z_b) / extent, extent = 0.03 min(z) +
//      0.05 m: depth-relative, since the paper's fixed SOFT_Z_EXTENT does not fit a 0.1 m .. 6 km depth range). Output
//      rgb = weighted mean, alpha = the centre texel's alpha.
// Reads per output pixel: velocity pass 3 (depth, flags, velocity; 2 where the flags say camera-only), tile rows 1
// (each velocity texel once), tile columns 1 / tileSize, neighbour max 9 per tile; reconstruction 2 where it early-outs
// (neighbour max, colour) and 3 + 2 * samples otherwise (19 at the default 8). Writes: 8 B velocity + 8 B output.
// Every target is cleared before its draw (autoClear off around the pass, renderer.clear(colour) per target, autoClear
// restored): the draws overwrite every texel, and a load-action clear is free on tile-based GPUs where loading the
// previous contents is not (passes before this one leave autoClear false).
// Options (MOTION_BLUR_DEFAULTS; live properties, validated, unknown keys throw): strength 1 (> 0; 1 = a 360-degree
// shutter while shutter is 0), maxBlur 24 (px, > 0), tileSize 20 (px, integer 1..128), samples 8 (integer 2..64, a
// define: changing it recompiles), shutter 0 (seconds, >= 0; see 1). The output is full resolution (the pass is read
// with texelFetch downstream).
// API: render(renderer, writeBuffer, readBuffer, deltaTime), setSize(w, h) (drawing-buffer px), reset() (no-op: no
// history), texture (output), velocityTexture / neighborMaxTexture (debug views), dispose(). needsSwap false.
// Perspective cameras and a standard depth buffer (not reversedDepthBuffer / logarithmicDepthBuffer).
import * as THREE from 'three';
import { Pass, FullScreenQuad } from 'three/addons/postprocessing/Pass.js';
import { GBUFFER_GLSL_DECODE } from '../GBuffer.js';

/** Default options (see the header). */
export const MOTION_BLUR_DEFAULTS = Object.freeze({ strength: 1, maxBlur: 24, tileSize: 20, samples: 8, shutter: 0 });

const VERTEX = /* glsl */`
  varying vec2 vUv;
  void main() { vUv = uv; gl_Position = vec4( position.xy, 0.0, 1.0 ); }`;

// 1. velocity: location 0 = V (half extent, px), location 1 = (|V|, linear depth)
const VELOCITY_FRAG = /* glsl */`
  uniform highp sampler2D tDepth;
  #ifndef MB_CAMERA_ONLY
  uniform sampler2D tVelocity;
  uniform sampler2D tNormal;
  #endif
  uniform mat4 uReproj;
  uniform vec2 uJitter, uRes;
  uniform vec2 uBlur; // x = k / 2 (UV motion -> half extent), y = largest half extent (px)
  uniform float uNear, uFar;
  layout( location = 1 ) out highp vec4 mbVZ;

  ${GBUFFER_GLSL_DECODE}

  void main() {
    ivec2 ip = ivec2( gl_FragCoord.xy );
    float d = texelFetch( tDepth, ip, 0 ).x;
    vec2 v;
  #ifndef MB_CAMERA_ONLY
    // a G-buffer writer with history: its own velocity. noHistory surfaces (ENGINE_GBUFFER_NOVEL writes velocity 0)
    // fall through to the camera motion, so a video screen still streaks while the camera pans
    if ( ( engineGbFlags( texelFetch( tNormal, ip, 0 ).a ) & ( ENGINE_GB_WROTE | ENGINE_GB_NO_HISTORY ) ) == ENGINE_GB_WROTE ) v = texelFetch( tVelocity, ip, 0 ).rg;
    else
  #endif
    {
      // camera-only: where this depth sample's surface was in the previous frame (uReproj: NDC -> previous clip)
      vec2 uv = ( gl_FragCoord.xy + uJitter ) / uRes;
      vec4 pc = uReproj * vec4( uv * 2.0 - 1.0, d * 2.0 - 1.0, 1.0 );
      v = pc.w > 0.0 ? uv - ( pc.xy / pc.w * 0.5 + 0.5 ) : vec2( 0.0 );
    }
    v *= uRes * uBlur.x; // UV -> px, half extent of the streak
    float l = length( v );
    if ( l > uBlur.y ) { v *= uBlur.y / l; l = uBlur.y; }
    gl_FragColor = vec4( v, 0.0, 1.0 );
    mbVZ = vec4( l, min( uNear * uFar / ( uFar - ( uFar - uNear ) * d ), 65000.0 ), 0.0, 1.0 );
  }`;

// 2. tile max, one direction: uDir (1, 0) reduces tileSize texels of a row, (0, 1) of a column
const TILE_FRAG = /* glsl */`
  uniform sampler2D tSrc;
  uniform ivec2 uDir;
  uniform int uTile;

  void main() {
    ivec2 o = ivec2( gl_FragCoord.xy ), sz = textureSize( tSrc, 0 );
    ivec2 p = o + uDir * o * ( uTile - 1 ); // first texel of this output's run
    int n = min( uTile, uDir.x == 1 ? sz.x - p.x : sz.y - p.y );
    vec2 best = vec2( 0.0 );
    float bl = - 1.0;
    for ( int i = 0; i < n; i ++ ) {
      vec2 v = texelFetch( tSrc, p + uDir * i, 0 ).rg;
      float l = dot( v, v );
      if ( l > bl ) { bl = l; best = v; }
    }
    gl_FragColor = vec4( best, 0.0, 1.0 );
  }`;

// 3. neighbour max: the longest V of the 3x3 tiles
const NEIGHBOR_FRAG = /* glsl */`
  uniform sampler2D tTile;

  void main() {
    ivec2 o = ivec2( gl_FragCoord.xy ), mx = textureSize( tTile, 0 ) - 1;
    vec2 best = vec2( 0.0 );
    float bl = - 1.0;
    for ( int y = - 1; y <= 1; y ++ ) {
      for ( int x = - 1; x <= 1; x ++ ) {
        vec2 v = texelFetch( tTile, clamp( o + ivec2( x, y ), ivec2( 0 ), mx ), 0 ).rg;
        float l = dot( v, v );
        if ( l > bl ) { bl = l; best = v; }
      }
    }
    gl_FragColor = vec4( best, 0.0, 1.0 );
  }`;

// 4. reconstruction (MB_SAMPLES taps along the neighbourhood's dominant velocity)
const RECONSTRUCT_FRAG = /* glsl */`
  uniform sampler2D tColor;
  uniform sampler2D tVZ;
  uniform sampler2D tNeighbor;
  uniform int uTile;

  float cone( float r, float v ) { return clamp( 1.0 - r / v, 0.0, 1.0 ); }
  float cylinder( float r, float v ) { return 1.0 - smoothstep( 0.95 * v, 1.05 * v, r ); }
  // 1 where depth a is in front of (or level with) depth b, 0 once it is behind by the soft extent
  float front( float a, float b ) { return clamp( 1.0 - ( a - b ) / ( 0.03 * min( a, b ) + 0.05 ), 0.0, 1.0 ); }

  void main() {
    vec2 X = gl_FragCoord.xy;
    ivec2 mx = textureSize( tColor, 0 ) - 1, ix = min( ivec2( X ), mx );
    vec2 vn = texelFetch( tNeighbor, ix / uTile, 0 ).rg;
    vec4 cx = texelFetch( tColor, ix, 0 );
    if ( dot( vn, vn ) < 0.0625 ) { gl_FragColor = cx; return; } // |V_N| < 0.25 px: no streak reaches here, input texel
    vec2 vzx = texelFetch( tVZ, ix, 0 ).rg;
    float vx = max( vzx.x, 0.5 ), zx = vzx.y;
    float w = 1.0 / vx;
    vec3 sum = cx.rgb * w;
    float j = fract( 52.9829189 * fract( dot( gl_FragCoord.xy, vec2( 0.06711056, 0.00583715 ) ) ) ) - 0.5; // IGN
    for ( int i = 0; i < MB_SAMPLES; i ++ ) {
      float t = mix( - 1.0, 1.0, ( float( i ) + j + 1.0 ) / ( float( MB_SAMPLES ) + 1.0 ) );
      ivec2 iy = clamp( ivec2( floor( X + vn * t ) ), ivec2( 0 ), mx );
      vec2 vzy = texelFetch( tVZ, iy, 0 ).rg;
      float vy = max( vzy.x, 0.5 ), zy = vzy.y, r = length( vec2( iy - ix ) );
      float a = front( zy, zx ) * cone( r, vy ) + front( zx, zy ) * cone( r, vx ) + cylinder( r, vy ) * cylinder( r, vx ) * 2.0;
      w += a;
      sum += texelFetch( tColor, iy, 0 ).rgb * a;
    }
    gl_FragColor = vec4( sum / w, cx.a );
  }`;

const KEYS = Object.keys(MOTION_BLUR_DEFAULTS);

/**
 * @typedef {object} MotionBlurOptions
 * @property {number} [strength=1] streak = inter-frame motion * strength
 * @property {number} [maxBlur=24] longest streak (px)
 * @property {number} [tileSize=20] px
 * @property {number} [samples=8] reconstruction taps
 * @property {number} [shutter=0] exposure (s); 0 = frame-relative
 */

export class MotionBlurPass extends Pass {
  /**
   * @param {THREE.PerspectiveCamera} camera near / far linearise the depth
   * @param {import('../CameraHistory.js').CameraHistory} history uReproj (camera-only velocity) and uJitter
   * @param {import('../GBuffer.js').GBuffer|null} gbuffer velocity + normal (flags) attachments; null: camera-only
   * @param {MotionBlurOptions} [o] see the header
   */
  constructor(camera, history, gbuffer, o = {}) {
    super();
    if (!history || !history.uniforms || !history.uniforms.uReproj) throw new Error('[MotionBlurPass] needs a CameraHistory (camera-only velocity, jitter)');
    for (const k in o) if (!KEYS.includes(k)) throw new Error(`[MotionBlurPass] unknown option '${k}' (${KEYS.join(', ')})`);
    this.needsSwap = false;
    this.camera = camera;
    this.history = history;
    /** G-buffer whose velocity / flags attachments are read; null = camera-only velocity everywhere. */
    this.gbuffer = gbuffer ?? null;
    const d = MOTION_BLUR_DEFAULTS;
    this._strength = positive(o.strength ?? d.strength, 'strength');
    this._maxBlur = positive(o.maxBlur ?? d.maxBlur, 'maxBlur');
    this._tileSize = integer(o.tileSize ?? d.tileSize, 1, 128, 'tileSize');
    this._samples = integer(o.samples ?? d.samples, 2, 64, 'samples');
    this._shutter = nonNegative(o.shutter ?? d.shutter, 'shutter');
    this._w = 1; this._h = 1;
    this._camOnly = null; // MB_CAMERA_ONLY defined (gbuffer null), synced in render()

    const rg = { type: THREE.HalfFloatType, format: THREE.RGFormat, depthBuffer: false, stencilBuffer: false, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter, generateMipmaps: false };
    this._vel = new THREE.WebGLRenderTarget(1, 1, { ...rg, count: 2 });
    this._vel.textures[0].name = 'MotionBlur.velocity';
    this._vel.textures[1].name = 'MotionBlur.speedDepth';
    this._tileX = new THREE.WebGLRenderTarget(1, 1, rg);
    this._tileX.texture.name = 'MotionBlur.tileRows';
    this._tile = new THREE.WebGLRenderTarget(1, 1, rg);
    this._tile.texture.name = 'MotionBlur.tileMax';
    this._neighbor = new THREE.WebGLRenderTarget(1, 1, rg);
    this._neighbor.texture.name = 'MotionBlur.neighborMax';
    this._out = new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, depthBuffer: false, stencilBuffer: false, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter, generateMipmaps: false });
    this._out.texture.name = 'MotionBlur.out';

    const hu = history.uniforms;
    const mat = (name, fragmentShader, uniforms, defines = {}) => new THREE.ShaderMaterial({
      name, defines, uniforms, vertexShader: VERTEX, fragmentShader,
      blending: THREE.NoBlending, depthTest: false, depthWrite: false,
    });
    this._velocityMat = mat('PostFXMotionBlurVelocity', VELOCITY_FRAG, {
      tDepth: { value: null }, tVelocity: { value: null }, tNormal: { value: null },
      uReproj: hu.uReproj, uJitter: hu.uJitter, // CameraHistory's own uniform objects
      uRes: { value: new THREE.Vector2(1, 1) }, uBlur: { value: new THREE.Vector2(0.5, 12) },
      uNear: { value: 0.1 }, uFar: { value: 1000 },
    });
    this._tileMat = mat('PostFXMotionBlurTileMax', TILE_FRAG, { tSrc: { value: null }, uDir: { value: new THREE.Vector2(1, 0) }, uTile: { value: 20 } });
    this._neighborMat = mat('PostFXMotionBlurNeighborMax', NEIGHBOR_FRAG, { tTile: { value: null } });
    this._reconstructMat = mat('PostFXMotionBlur', RECONSTRUCT_FRAG, {
      tColor: { value: null }, tVZ: { value: this._vel.textures[1] }, tNeighbor: { value: this._neighbor.texture },
      uTile: { value: 20 },
    }, { MB_SAMPLES: this._samples });
    this._quad = new FullScreenQuad(null);
    this._resize(); // option-dependent sizes / uniforms right away (setSize() repeats it at the real size)
  }

  /** The blurred colour of the last render (RGBA16F; alpha = the input's alpha). */
  get texture() { return this._out.texture; }
  /** Debug: per-pixel half-extent velocity V (px, rg) of the last render. */
  get velocityTexture() { return this._vel.textures[0]; }
  /** Debug: per-tile neighbourhood max velocity V_N (px, rg) of the last render. */
  get neighborMaxTexture() { return this._neighbor.texture; }

  /** Streak length = inter-frame motion * strength (> 0). */
  get strength() { return this._strength; }
  set strength(v) { this._strength = positive(v, 'strength'); }
  /** Longest streak (px, > 0); the half extent is also capped at tileSize. */
  get maxBlur() { return this._maxBlur; }
  set maxBlur(v) { this._maxBlur = positive(v, 'maxBlur'); }
  /** Tile edge (px, integer 1..128); resizes the tile targets. */
  get tileSize() { return this._tileSize; }
  set tileSize(v) { this._tileSize = integer(v, 1, 128, 'tileSize'); this._resize(); }
  /** Reconstruction taps (integer 2..64; recompiles the shader). */
  get samples() { return this._samples; }
  set samples(v) {
    v = integer(v, 2, 64, 'samples');
    if (v === this._samples) return;
    this._samples = v;
    this._reconstructMat.defines.MB_SAMPLES = v;
    this._reconstructMat.needsUpdate = true;
  }
  /** Exposure time (s, >= 0): > 0 makes the streak frame-rate independent (see the header); 0 = frame-relative. */
  get shutter() { return this._shutter; }
  set shutter(v) { this._shutter = nonNegative(v, 'shutter'); }

  /** No history: nothing to reset (the camera history lives in CameraHistory). */
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
   * @param {THREE.WebGLRenderTarget} readBuffer colour to blur + scene depth (texture, depthTexture)
   * @param {number} [deltaTime] frame time (s), for `shutter`
   */
  render(renderer, writeBuffer, readBuffer, deltaTime) {
    const gb = this.gbuffer, cam = this.camera;
    const camOnly = gb === null;
    if (camOnly !== this._camOnly) {
      this._camOnly = camOnly;
      const d = this._velocityMat.defines;
      if (camOnly) d.MB_CAMERA_ONLY = 1; else delete d.MB_CAMERA_ONLY;
      this._velocityMat.needsUpdate = true;
    }
    const vu = this._velocityMat.uniforms;
    vu.tDepth.value = readBuffer.depthTexture;
    vu.tVelocity.value = camOnly ? null : gb.velocity;
    vu.tNormal.value = camOnly ? null : gb.normal;
    // near / far stored only on change: a double written into a plain `value` is boxed (a heap number per frame); the
    // blur scale goes into a Vector2's fields, which hold doubles in place
    if (vu.uNear.value !== cam.near) vu.uNear.value = cam.near;
    if (vu.uFar.value !== cam.far) vu.uFar.value = cam.far;
    const sh = this._shutter, k = sh > 0 && deltaTime > 0 ? Math.min(sh / deltaTime, 4) : 1;
    const blur = vu.uBlur.value;
    blur.x = 0.5 * this._strength * k;
    blur.y = Math.min(0.5 * this._maxBlur, this._tileSize);

    const ac = renderer.autoClear;
    renderer.autoClear = false;
    this._draw(renderer, this._vel, this._velocityMat);
    const tu = this._tileMat.uniforms;
    tu.tSrc.value = this._vel.textures[0]; tu.uDir.value.set(1, 0);
    this._draw(renderer, this._tileX, this._tileMat);
    tu.tSrc.value = this._tileX.texture; tu.uDir.value.set(0, 1);
    this._draw(renderer, this._tile, this._tileMat);
    this._neighborMat.uniforms.tTile.value = this._tile.texture;
    this._draw(renderer, this._neighbor, this._neighborMat);
    this._reconstructMat.uniforms.tColor.value = readBuffer.texture;
    this._draw(renderer, this._out, this._reconstructMat);
    renderer.autoClear = ac;
  }

  dispose() {
    for (const rt of [this._vel, this._tileX, this._tile, this._neighbor, this._out]) rt.dispose();
    for (const m of [this._velocityMat, this._tileMat, this._neighborMat, this._reconstructMat]) m.dispose();
    this._quad.dispose();
  }

  // one draw into a cleared target (autoClear is off around the pass: see the header)
  _draw(renderer, target, material) {
    renderer.setRenderTarget(target);
    renderer.clear(true, false, false);
    this._quad.material = material;
    this._quad.render(renderer);
  }

  // every target from the drawing-buffer size and tileSize
  _resize() {
    const W = this._w, H = this._h, T = this._tileSize;
    const tw = Math.max(1, Math.ceil(W / T)), th = Math.max(1, Math.ceil(H / T));
    this._vel.setSize(W, H);
    this._tileX.setSize(tw, H);
    this._tile.setSize(tw, th);
    this._neighbor.setSize(tw, th);
    this._out.setSize(W, H);
    this._velocityMat.uniforms.uRes.value.set(W, H);
    this._tileMat.uniforms.uTile.value = T;
    this._reconstructMat.uniforms.uTile.value = T;
  }
}

function positive(v, what) {
  if (typeof v !== 'number' || !Number.isFinite(v) || v <= 0) throw new Error(`[MotionBlurPass] ${what}: expected a finite number > 0, got ${v}`);
  return v;
}
function integer(v, lo, hi, what) {
  if (!Number.isInteger(v) || v < lo || v > hi) throw new Error(`[MotionBlurPass] ${what}: expected an integer in ${lo}..${hi}, got ${v}`);
  return v;
}
function nonNegative(v, what) {
  if (typeof v !== 'number' || !Number.isFinite(v) || v < 0) throw new Error(`[MotionBlurPass] ${what}: expected a finite number >= 0, got ${v}`);
  return v;
}
