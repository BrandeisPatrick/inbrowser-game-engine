// Ground-truth ambient occlusion from the MAIN pass depth buffer (no second scene render): the GTAO shader
// from three/addons with normals reconstructed from depth, at half resolution, followed by a separable
// depth-aware blur. Because it reads the resolved scene depth, instanced meshes, alpha-tested foliage,
// vertex-animated materials and depthWrite:false effects are all handled exactly as they were drawn.
// Output: `aoTexture` (r = ao 0..1, g = linear view depth in metres), sampled by the final pass.
// Size: resolutionScale of the drawing buffer, capped at maxPixels (default Infinity here; PostFX passes 0.8e6). GTAO costs per AO pixel, and a
// 2x Retina buffer makes half resolution 1.5 MP (Canyon Run at 3024x1964: 2.7 ms; 0.92 MP at 2560x1440: 1.55 ms), so a
// budget keeps the pass at one cost whatever the device pixel ratio. `upsample` ('linear' | 'depth') is how the final
// pass samples aoTexture: below half resolution a bilinear tap leaves wider halos at silhouettes, the depth-aware
// upsample does not. setResolution(scale, maxPixels) resizes at once; `compact` (see the setter) halves the traffic.
import * as THREE from 'three';
import { Pass, FullScreenQuad } from 'three/addons/postprocessing/Pass.js';
import { GTAOShader, generateMagicSquareNoise } from 'three/addons/shaders/GTAOShader.js';

const FAR_DEPTH = 60000; // g-channel value for sky / cleared texels

const BlurShader = {
  uniforms: {
    tAO: { value: null },
    uTexel: { value: new THREE.Vector2() },
    uDir: { value: new THREE.Vector2(1, 0) },
  },
  vertexShader: /* glsl */`
    varying vec2 vUv;
    void main() { vUv = uv; gl_Position = vec4( position.xy, 0.0, 1.0 ); }`,
  fragmentShader: /* glsl */`
    uniform sampler2D tAO;
    uniform vec2 uTexel;
    uniform vec2 uDir;
    varying vec2 vUv;
    void main() {
      vec2 c = texture2D( tAO, vUv ).rg;
      float z = c.g;
      float tol = 0.06 * z + 0.15; // depth tolerance grows with distance
      float sum = c.r * 0.38, wsum = 0.38;
      for ( int i = 1; i <= 2; i ++ ) {
        float w0 = i == 1 ? 0.24 : 0.07;
        vec2 o = uDir * uTexel * float( i );
        vec2 a = texture2D( tAO, vUv + o ).rg;
        vec2 b = texture2D( tAO, vUv - o ).rg;
        float wa = w0 * clamp( 1.0 - abs( a.g - z ) / tol, 0.0, 1.0 );
        float wb = w0 * clamp( 1.0 - abs( b.g - z ) / tol, 0.0, 1.0 );
        sum += a.r * wa + b.r * wb; wsum += wa + wb;
      }
      gl_FragColor = vec4( sum / wsum, z, 0.0, 1.0 );
    }`,
};

export class AOPass extends Pass {
  /**
   * @param {THREE.PerspectiveCamera} camera
   * @param {object} [o] radius (m), thickness (m), samples, resolutionScale, maxPixels, upsample ('linear' | 'depth'),
   *   compact, fadeNear/fadeFar (m)
   */
  constructor(camera, o = {}) {
    super();
    this.needsSwap = false;
    this.camera = camera;
    this.scale = o.resolutionScale ?? 0.5;
    this.maxPixels = o.maxPixels ?? Infinity;
    this.upsample = o.upsample ?? 'linear'; // read by the final pass
    this.fadeNear = o.fadeNear ?? 140;
    this.fadeFar = o.fadeFar ?? 320;
    this._compact = !!o.compact;
    this._size = new THREE.Vector2(1, 1); // drawing-buffer size of the last setSize, for setResolution
    const opt = { type: THREE.HalfFloatType, format: this._compact ? THREE.RGFormat : THREE.RGBAFormat, depthBuffer: false, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter };
    this.rtA = new THREE.WebGLRenderTarget(1, 1, opt);
    this.rtB = new THREE.WebGLRenderTarget(1, 1, opt);
    this.rtA.texture.name = 'AO.a'; this.rtB.texture.name = 'AO.b';
    this.rtA.texture.generateMipmaps = this.rtB.texture.generateMipmaps = false;

    let frag = GTAOShader.fragmentShader;
    // clamp the sampling radius on screen (cheap and cache-friendly when the car fills the view)
    frag = frag.replace('float radiusToUse = radius;', 'float radiusToUse = min( radius, -viewPos.z * 0.35 );');
    // early out beyond the fade distance (most of an aerial view): no normal reconstruction, no horizon search
    frag = frag.replace('vec3 viewNormal = getViewNormal(vUv);', 'if ( -viewPos.z > AO_FADE_FAR ) { gl_FragColor = vec4( 1.0, -viewPos.z, 0.0, 1.0 ); return; }\n\t\t\tvec3 viewNormal = getViewNormal(vUv);');
    if (!frag.includes('AO_FADE_FAR ) {')) console.warn('[render] GTAO early-out patch point missing');
    this.material = new THREE.ShaderMaterial({
      name: 'PostFXGTAO',
      defines: {
        ...GTAOShader.defines,
        PERSPECTIVE_CAMERA: 1,
        SAMPLES: o.samples ?? 12,
        NORMAL_VECTOR_TYPE: 0, // reconstruct normals from depth
        DEPTH_SWIZZLING: 'x',
        SCREEN_SPACE_RADIUS: 0,
        SCENE_CLIP_BOX: 0,
        AO_FADE_NEAR: this.fadeNear.toFixed(1),
        AO_FADE_FAR: this.fadeFar.toFixed(1),
        FRAGMENT_OUTPUT: 'vec4( mix( ao, 1.0, smoothstep( AO_FADE_NEAR, AO_FADE_FAR, -viewPos.z ) ), -viewPos.z, 0.0, 1.0 )',
      },
      uniforms: THREE.UniformsUtils.clone(GTAOShader.uniforms),
      vertexShader: GTAOShader.vertexShader,
      fragmentShader: frag,
      blending: THREE.NoBlending,
      depthTest: false,
      depthWrite: false,
    });
    const u = this.material.uniforms;
    u.tNoise.value = generateMagicSquareNoise();
    u.radius.value = o.radius ?? 1.8;
    u.thickness.value = o.thickness ?? 1.6;
    u.distanceExponent.value = o.distanceExponent ?? 1.6;
    u.distanceFallOff.value = 1.0;
    u.scale.value = o.power ?? 1.0;

    this.blur = new THREE.ShaderMaterial({
      name: 'PostFXAOBlur',
      uniforms: THREE.UniformsUtils.clone(BlurShader.uniforms),
      vertexShader: BlurShader.vertexShader,
      fragmentShader: BlurShader.fragmentShader,
      blending: THREE.NoBlending, depthTest: false, depthWrite: false,
    });
    this._quad = new FullScreenQuad(null);
    this._clear = new THREE.Color();
  }

  get aoTexture() { return this.rtA.texture; }

  setSize(width, height) {
    this._size.set(width, height);
    let w = Math.max(1, Math.round(width * this.scale)), h = Math.max(1, Math.round(height * this.scale));
    if (w * h > this.maxPixels) { // over the budget: the largest uniform scale that fits, floored (never above it)
      const s = Math.sqrt(this.maxPixels / (width * height));
      w = Math.max(1, Math.floor(width * s)); h = Math.max(1, Math.floor(height * s));
    }
    this.rtA.setSize(w, h); this.rtB.setSize(w, h);
    this.material.uniforms.resolution.value.set(w, h);
    this.blur.uniforms.uTexel.value.set(1 / w, 1 / h);
  }

  /** Runtime resize (console A/B): scale and / or maxPixels (null: no cap), applied to the current buffer size. */
  setResolution(scale = this.scale, maxPixels = this.maxPixels) {
    this.scale = scale;
    this.maxPixels = maxPixels ?? Infinity;
    this.setSize(this._size.x, this._size.y);
  }

  /**
   * compact: RG16F targets (all the AO buffers hold: r = ao, g = depth) instead of RGBA16F, and a clear before each
   * blur draw, so a tile-based GPU starts those passes from the clear instead of loading the target's previous
   * contents (the blur overwrites every texel anyway). Same output, about half the AO buffer traffic; needs the
   * float colour-buffer support the RGBA16F targets already need.
   */
  get compact() { return this._compact; }
  set compact(v) {
    if (!!v === this._compact) return;
    this._compact = !!v;
    for (const rt of [this.rtA, this.rtB]) { rt.texture.format = this._compact ? THREE.RGFormat : THREE.RGBAFormat; rt.dispose(); }
  }

  render(renderer, writeBuffer, readBuffer) {
    const depth = readBuffer.depthTexture;
    if (!depth) return;
    const cam = this.camera, u = this.material.uniforms;
    u.tDepth.value = depth;
    u.cameraNear.value = cam.near; u.cameraFar.value = cam.far;
    u.cameraProjectionMatrix.value.copy(cam.projectionMatrix);
    u.cameraProjectionMatrixInverse.value.copy(cam.projectionMatrixInverse);
    u.cameraWorldMatrix.value.copy(cam.matrixWorld);

    renderer.getClearColor(this._clear);
    const ca = renderer.getClearAlpha(), ac = renderer.autoClear;
    renderer.autoClear = false;
    renderer.setRenderTarget(this.rtA); // before setClearColor: the clear colour is converted for the bound target
    renderer.setClearColor(_skyClear, 1);
    renderer.clear(true, false, false);
    this._quad.material = this.material;
    this._quad.render(renderer);
    // separable bilateral blur (5x5 footprint = the GTAO noise period)
    const b = this.blur.uniforms, clr = this._compact;
    this._quad.material = this.blur;
    b.tAO.value = this.rtA.texture; b.uDir.value.set(1, 0);
    renderer.setRenderTarget(this.rtB); if (clr) renderer.clear(true, false, false);
    this._quad.render(renderer);
    b.tAO.value = this.rtB.texture; b.uDir.value.set(0, 1);
    renderer.setRenderTarget(this.rtA); if (clr) renderer.clear(true, false, false);
    this._quad.render(renderer);

    renderer.setClearColor(this._clear, ca);
    renderer.autoClear = ac;
  }

  dispose() {
    this.rtA.dispose(); this.rtB.dispose(); this.material.dispose(); this.blur.dispose(); this._quad.dispose();
    this.material.uniforms.tNoise.value.dispose();
  }
}
const _skyClear = new THREE.Color(1, FAR_DEPTH, 0);
