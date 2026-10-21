// UnrealBloomPass (three/addons) driven at quarter resolution with an HDR bright-pass (soft-knee threshold,
// firefly clamp, 4-tap box downsample so small emitters - tail lights, lamps - stay stable). The final
// additive blend is NOT done here: the result (`bloomTexture`) is added in the final grading pass, which
// saves a full-resolution read-modify-write.
import * as THREE from 'three';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';

const BrightShader = {
  uniforms: {
    tDiffuse: { value: null },
    uTexel: { value: new THREE.Vector2() },
    uThreshold: { value: 2.0 },
    uKnee: { value: 0.8 },
    uMax: { value: 40.0 },
  },
  vertexShader: /* glsl */`
    varying vec2 vUv;
    void main() { vUv = uv; gl_Position = vec4( position.xy, 0.0, 1.0 ); }`,
  fragmentShader: /* glsl */`
    uniform sampler2D tDiffuse;
    uniform vec2 uTexel;
    uniform float uThreshold, uKnee, uMax;
    varying vec2 vUv;
    vec3 tap( vec2 o ) { return min( texture2D( tDiffuse, vUv + o * uTexel ).rgb, vec3( uMax ) ); }
    void main() {
      vec3 c = 0.25 * ( tap( vec2( -1.0, -1.0 ) ) + tap( vec2( 1.0, -1.0 ) ) + tap( vec2( -1.0, 1.0 ) ) + tap( vec2( 1.0, 1.0 ) ) );
      float l = max( c.r, max( c.g, c.b ) );
      float s = clamp( l - uThreshold + uKnee, 0.0, 2.0 * uKnee );
      s = s * s / ( 4.0 * uKnee + 1e-4 );
      float k = max( s, l - uThreshold ) / max( l, 1e-4 );
      gl_FragColor = vec4( c * k, 1.0 );
    }`,
};

export class BloomPass extends UnrealBloomPass {
  constructor({ strength = 0.4, radius = 0.5, threshold = 2.0, resolutionScale = 0.5 } = {}) {
    super(new THREE.Vector2(256, 256), strength, radius, threshold);
    this.resScale = resolutionScale; // bright pass at (full * resScale / 2)
    this.knee = 0.8;
    this.maxBrightness = 14; // firefly / sun-disc clamp (scene-linear)
    this.materialHighPassFilter.dispose();
    this.materialHighPassFilter = new THREE.ShaderMaterial({
      name: 'PostFXBloomBright',
      uniforms: THREE.UniformsUtils.clone(BrightShader.uniforms),
      vertexShader: BrightShader.vertexShader,
      fragmentShader: BrightShader.fragmentShader,
      depthTest: false, depthWrite: false,
    });
    this.bloomFactors = [1.0, 0.85, 0.65, 0.45, 0.3];
    this.compositeMaterial.uniforms.bloomFactors.value = this.bloomFactors;
  }

  get bloomTexture() { return this.renderTargetsHorizontal[0].texture; }

  setSize(width, height) {
    super.setSize(Math.max(2, width * this.resScale), Math.max(2, height * this.resScale));
    const f = 2 / this.resScale; // downsample factor of the bright pass
    this.materialHighPassFilter.uniforms.uTexel.value.set((f / 4) / width, (f / 4) / height);
  }

  render(renderer, writeBuffer, readBuffer) {
    renderer.getClearColor(this._oldClearColor);
    this._oldClearAlpha = renderer.getClearAlpha();
    const oldAutoClear = renderer.autoClear;
    renderer.autoClear = false;
    renderer.setClearColor(this.clearColor, 0);

    const hp = this.materialHighPassFilter.uniforms;
    hp.tDiffuse.value = readBuffer.texture;
    hp.uThreshold.value = this.threshold;
    hp.uKnee.value = Math.max(0.05, this.knee);
    hp.uMax.value = this.maxBrightness;
    this._fsQuad.material = this.materialHighPassFilter;
    renderer.setRenderTarget(this.renderTargetBright);
    this._fsQuad.render(renderer);

    let input = this.renderTargetBright;
    for (let i = 0; i < this.nMips; i++) {
      const m = this.separableBlurMaterials[i];
      this._fsQuad.material = m;
      m.uniforms.colorTexture.value = input.texture;
      m.uniforms.direction.value = UnrealBloomPass.BlurDirectionX;
      renderer.setRenderTarget(this.renderTargetsHorizontal[i]);
      this._fsQuad.render(renderer);
      m.uniforms.colorTexture.value = this.renderTargetsHorizontal[i].texture;
      m.uniforms.direction.value = UnrealBloomPass.BlurDirectionY;
      renderer.setRenderTarget(this.renderTargetsVertical[i]);
      this._fsQuad.render(renderer);
      input = this.renderTargetsVertical[i];
    }
    const cu = this.compositeMaterial.uniforms;
    this._fsQuad.material = this.compositeMaterial;
    cu.bloomStrength.value = this.strength;
    cu.bloomRadius.value = this.radius;
    cu.bloomTintColors.value = this.bloomTintColors;
    renderer.setRenderTarget(this.renderTargetsHorizontal[0]);
    this._fsQuad.render(renderer);

    renderer.setClearColor(this._oldClearColor, this._oldClearAlpha);
    renderer.autoClear = oldAutoClear;
  }
}
