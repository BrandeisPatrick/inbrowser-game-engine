// Eye adaptation, GPU only (no readback): the HDR scene is reduced to a 64x64 centre-weighted log-luminance
// texture, mipmapped to 1x1, and blended over time into a 1x1 "adapted luminance" target. The final pass
// reads it and applies a partial, clamped exposure correction (a dark street opens up ~1 stop, the bright
// sky in aerial views does not blow out). Costs well under 0.1 ms.
import * as THREE from 'three';
import { Pass, FullScreenQuad } from 'three/addons/postprocessing/Pass.js';

const VS = /* glsl */`varying vec2 vUv; void main() { vUv = uv; gl_Position = vec4( position.xy, 0.0, 1.0 ); }`;

export class ExposurePass extends Pass {
  constructor() {
    super();
    this.needsSwap = false;
    this.speed = 1.6; // 1/s
    this.lumRT = new THREE.WebGLRenderTarget(64, 64, { type: THREE.HalfFloatType, depthBuffer: false, minFilter: THREE.LinearMipmapLinearFilter, magFilter: THREE.LinearFilter, generateMipmaps: true });
    this.lumRT.texture.name = 'Exposure.lum';
    const o = { type: THREE.HalfFloatType, depthBuffer: false, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter };
    this.rtA = new THREE.WebGLRenderTarget(1, 1, o);
    this.rtB = new THREE.WebGLRenderTarget(1, 1, o);
    this.lumMat = new THREE.ShaderMaterial({
      name: 'PostFXLum',
      uniforms: { tColor: { value: null } },
      vertexShader: VS,
      fragmentShader: /* glsl */`
        uniform sampler2D tColor; varying vec2 vUv;
        void main() {
          vec2 o = vec2( 0.25 / 64.0 );
          vec3 c = 0.25 * ( texture2D( tColor, vUv + o ).rgb + texture2D( tColor, vUv - o ).rgb + texture2D( tColor, vUv + vec2( o.x, -o.y ) ).rgb + texture2D( tColor, vUv - vec2( o.x, -o.y ) ).rgb );
          float l = dot( c, vec3( 0.2126, 0.7152, 0.0722 ) );
          vec2 d = ( vUv - 0.5 ) * vec2( 1.0, 1.25 );
          float w = 1.0 - 0.7 * smoothstep( 0.1, 0.7, length( d ) ); // centre-weighted metering
          gl_FragColor = vec4( log2( clamp( l, 1e-4, 32.0 ) ) * w, w, 0.0, 1.0 );
        }`,
      depthTest: false, depthWrite: false, blending: THREE.NoBlending,
    });
    this.adaptMat = new THREE.ShaderMaterial({
      name: 'PostFXAdapt',
      uniforms: { tLum: { value: null }, tPrev: { value: null }, uRate: { value: 1 } },
      vertexShader: VS,
      fragmentShader: /* glsl */`
        uniform sampler2D tLum, tPrev; uniform float uRate; varying vec2 vUv;
        void main() {
          vec2 s = textureLod( tLum, vec2( 0.5 ), 6.0 ).rg;
          float cur = s.r / max( s.g, 1e-4 );
          float prev = texture2D( tPrev, vec2( 0.5 ) ).r;
          gl_FragColor = vec4( mix( prev, cur, uRate ), 0.0, 0.0, 1.0 ); // adapted log2 luminance
        }`,
      depthTest: false, depthWrite: false, blending: THREE.NoBlending,
    });
    this._quad = new FullScreenQuad(null);
    this._snap = true;
  }

  /** log2 adapted luminance (1x1 texture, red channel). */
  get texture() { return this.rtA.texture; }

  /** Jump straight to the current scene luminance on the next frame (teleport, time-of-day change). */
  reset() { this._snap = true; }

  render(renderer, writeBuffer, readBuffer, deltaTime) {
    this.lumMat.uniforms.tColor.value = readBuffer.texture;
    this._quad.material = this.lumMat;
    renderer.setRenderTarget(this.lumRT);
    this._quad.render(renderer);
    const t = this.rtA; this.rtA = this.rtB; this.rtB = t; // rtB = previous
    const u = this.adaptMat.uniforms;
    u.tLum.value = this.lumRT.texture;
    u.tPrev.value = this.rtB.texture;
    const dt = Math.min(0.25, Math.max(0, deltaTime || 0));
    u.uRate.value = this._snap ? 1 : 1 - Math.exp(-dt * this.speed);
    this._snap = false;
    this._quad.material = this.adaptMat;
    renderer.setRenderTarget(this.rtA);
    this._quad.render(renderer);
  }

  dispose() { this.lumRT.dispose(); this.rtA.dispose(); this.rtB.dispose(); this.lumMat.dispose(); this.adaptMat.dispose(); this._quad.dispose(); }
}
