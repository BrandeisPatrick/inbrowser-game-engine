// Mid-frame copy of the opaque scene, for materials that look through or behind their surface (water refraction in
// 0.8.0, glass later): when an attached mesh is about to draw, the colour and depth the scene target holds at that
// moment (everything drawn before the mesh: the whole opaque scene for a mesh with renderOrder 1 like the bay) are
// copied at `scale` into a small target its material samples in the same draw.
//
//   const copy = new SceneCopy(renderer, { scale: 0.5, color: true, depth: true });
//   copy.attach(mesh);              // chains mesh.onBeforeRender: the copy first, then the mesh's own hook
//   material: tScene <- copy.color, tDepth <- copy.depth (the same Texture objects for the copy's whole life)
//   resize:   copy.setSize(drawingBufferWidth, drawingBufferHeight)   (the scene target's size; rounded with scale)
//   copy.valid, copy.frame, copy.copies, copy.detach(mesh), copy.dispose()
//   measurement knobs: copy.setScale(s) (0 < s <= 1, re-sizes from the last setSize), copy.clear (boolean)
//
// Source: renderer.getRenderTarget() while the attached mesh draws. Its attachment 0 (target.texture, also on the
// G-buffer's 3-attachment MRT target) and its depthTexture are read; the copy target is separate, so there is no
// feedback loop (the scene target is not bound during the copy, and sampling textures of an unbound target is
// legal). Skipped, with `valid` = false and `reason` set, when the source is:
//   - the canvas (null target: no PostFX, or PostFX disabled),
//   - multisampled (samples > 0): three resolves MSAA only at the END of renderer.render, so mid-frame the target's
//     texture still holds last frame's image (and depth). Refraction therefore needs PostFX on the G-buffer path
//     (gbuffer: true) or a single-sampled forward target (aa: 'fxaa'; 'auto' turns MSAA off by night or above
//     msaaMaxPixels),
//   - without a depthTexture (only when depth is copied),
//   - on a renderer without float colour targets (EXT_color_buffer_float),
//   - on a renderer with a reversed depth buffer (capabilities.reversedDepthBuffer): the farthest-of-2x2 depth
//     filter and the consumers' linearisation (Water.js) assume a standard depth buffer.
// The console.warn (one per reason) is deferred to the next render() call and only fires when no valid copy was taken
// in between: three's own transmission pass re-renders opaque objects into a 4x MSAA target inside the same render()
// call, so a scene with transmissive materials skips there and copies in the main pass, which is not worth a warning.
//
// Once per frame: the stamp is renderer.info.render.frame, which three increments on every render() call whatever
// info.autoReset says. The copy's own quad render increments it too, so the stamp kept is the value AFTER the copy:
// further attached meshes of the same scene render (bay + ponds) see that value and reuse the copy; the next
// render() call (next frame, or another pass that draws the scene) takes a new one.
//
// The copy (the Reflector.js pattern for rendering from inside onBeforeRender, which three supports through its
// render-state / render-list stacks): save the bound target with its cube face and mip level, the current viewport,
// xr.enabled, shadowMap.autoUpdate and autoClear; bind the copy target, clear its colour (`clear`, default true: on
// tiled GPUs (ANGLE / Metal) the clear becomes the copy pass's load action instead of a load of the old contents;
// every texel is overwritten either way), draw one full-screen triangle (FullScreenQuad) with xr, shadow updates
// and autoClear off; restore everything (in finally).
//   target   colour + depth: ONE MRT target (count 2): textures[0] RGBA16F linear-filtered colour, textures[1] R32F
//            nearest depth. One draw and one render pass instead of two, and the fragment reads each source texel
//            once for both outputs. Colour or depth alone: a single attachment.
//   colour   bilinear at the copy texel centre: a 2x2 box at scale 0.5 (PostFX's scene targets are linear-filtered).
//   depth    raw 0..1 (linearise with the scene camera's near / far), the FARTHEST of the 2x2 source texels around
//            the copy texel centre (texelFetch; one texel at scale 1): a foreground silhouette never spreads into the
//            copy, so a water pixel next to a hull or a pile reads a depth at least as far as its own background (no
//            false contact-foam halo); up to one copy texel of foreground colour can still bleed in at silhouettes.
// Cost: one quad at scale (a bilinear colour read + 4 depth fetches and 2 writes per copy texel: a quarter of the
// pixels at 0.5) plus the render-target switch in the middle of the scene pass. On tiled GPUs (Apple, ANGLE / Metal)
// that switch ends the scene target's render pass at the copy and resumes it afterwards: every attachment (G-buffer:
// three colours + depth) is stored and loaded back once per frame, which likely dominates at Retina sizes; on
// immediate-mode GPUs it is a framebuffer bind. Measure it (the lead's knob: the material switch + this copy).
// Per frame nothing allocates (hooks are bound at attach(), vectors preallocated).
// Renderer statistics: the copy is a nested renderer.render, so it counts one draw call (info.render.calls) and
// advances info.render.frame. With three's default info.autoReset = true that nested render also runs info.reset()
// in the middle of the frame, so the frame's calls / triangles only count what was drawn after the copy; the Engine
// (autoReset false, one reset per tick) and PostFX are unaffected.
import * as THREE from 'three';
import { FullScreenQuad } from 'three/addons/postprocessing/Pass.js';

const COPY_VERT = /* glsl */`
out vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4( position.xy, 0.0, 1.0 );
}`;

const COPY_FRAG = /* glsl */`
uniform sampler2D tColor;
uniform highp sampler2D tDepth;
in vec2 vUv;
#ifdef COPY_COLOR
layout( location = 0 ) out vec4 oColor;
#endif
#ifdef COPY_DEPTH
layout( location = COPY_DEPTH_LOCATION ) out vec4 oDepth;
#endif
void main() {
#ifdef COPY_COLOR
  oColor = vec4( texture( tColor, vUv ).rgb, 1.0 );
#endif
#ifdef COPY_DEPTH
  ivec2 size = textureSize( tDepth, 0 ), hi = size - 1;
#ifdef COPY_FULL
  float d = texelFetch( tDepth, clamp( ivec2( vUv * vec2( size ) ), ivec2( 0 ), hi ), 0 ).r;
#else
  // the 2x2 source texels around the copy texel centre (exactly its block at scale 0.5), farthest wins
  ivec2 a = clamp( ivec2( floor( vUv * vec2( size ) - 0.5 ) ), ivec2( 0 ), hi ), b = min( a + 1, hi );
  float d = max( max( texelFetch( tDepth, a, 0 ).r, texelFetch( tDepth, ivec2( b.x, a.y ), 0 ).r ),
                 max( texelFetch( tDepth, ivec2( a.x, b.y ), 0 ).r, texelFetch( tDepth, b, 0 ).r ) );
#endif
  oDepth = vec4( d, 0.0, 0.0, 1.0 );
#endif
}`;

const WARN = {
  canvas: 'the scene renders to the canvas (no render target)',
  msaa: 'the scene target is multisampled: three resolves MSAA only at the end of renderer.render, so mid-frame its texture is last frame\'s',
  nodepth: 'the scene target has no depthTexture',
  unsupported: 'float colour targets (EXT_color_buffer_float) are unavailable',
  reversed: 'the renderer uses a reversed depth buffer (the depth filter and its consumers assume a standard one)',
};

export class SceneCopy {
  /**
   * @param {THREE.WebGLRenderer|null} renderer  for the initial size (its drawing buffer) and the float-target check;
   *   null is allowed (headless): copies use the renderer each attached mesh draws with
   * @param {object} [o]
   * @param {number} [o.scale=0.5]  copy resolution relative to the scene target (0 < scale <= 1)
   * @param {boolean} [o.color=true]  copy the HDR colour (RGBA16F, linear filtering)
   * @param {boolean} [o.depth=true]  copy the raw depth (R32F, nearest; needs a source depthTexture)
   * @param {boolean} [o.clear=true]  clear the copy target's colour before the quad (a load action on tiled GPUs)
   * @param {number} [o.width]  initial scene-target width in drawing-buffer px (default: the renderer's drawing buffer, or 1)
   * @param {number} [o.height]  initial scene-target height in drawing-buffer px
   */
  constructor(renderer, { scale = 0.5, color = true, depth = true, clear = true, width, height } = {}) {
    if (!color && !depth) throw new Error('[SceneCopy] nothing to copy: color and depth are both off');
    if (!(scale > 0 && scale <= 1)) throw new Error(`[SceneCopy] scale must be in (0, 1], got ${scale}`);
    this.renderer = renderer || null;
    this.scale = scale;
    this.clear = !!clear;
    /** true when the last attempt copied (this frame, from a single-sampled target with the needed depth texture) */
    this.valid = false;
    /** renderer.info.render.frame right after the last copy (-1: none yet); attached meshes of the same render reuse it */
    this.frame = -1;
    /** number of copies taken */
    this.copies = 0;
    /** why the last attempt was skipped ('canvas' | 'msaa' | 'nodepth' | 'unsupported' | 'reversed'), null after a copy */
    this.reason = null;
    this.width = 1; this.height = 1;
    const opts = { type: THREE.HalfFloatType, format: THREE.RGBAFormat, depthBuffer: false, stencilBuffer: false, generateMipmaps: false, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter };
    const rt = this.target = new THREE.WebGLRenderTarget(1, 1, color && depth ? { ...opts, count: 2 } : opts);
    /** @type {THREE.Texture|null} HDR colour of the scene before the attached mesh (RGBA16F, linear) */
    this.color = color ? rt.textures[0] : null;
    /** @type {THREE.Texture|null} raw scene depth 0..1 (R32F, nearest), farthest of each 2x2 */
    this.depth = depth ? rt.textures[color ? 1 : 0] : null;
    if (this.color) this.color.name = 'SceneCopy.color';
    if (this.depth) {
      const d = this.depth;
      d.name = 'SceneCopy.depth'; d.type = THREE.FloatType; d.format = THREE.RedFormat;
      d.minFilter = d.magFilter = THREE.NearestFilter; d.colorSpace = THREE.NoColorSpace;
    }
    const defines = {};
    if (color) defines.COPY_COLOR = '';
    if (depth) { defines.COPY_DEPTH = ''; defines.COPY_DEPTH_LOCATION = color ? 1 : 0; }
    if (scale >= 0.99) defines.COPY_FULL = '';
    this.material = new THREE.ShaderMaterial({
      name: 'SceneCopy', glslVersion: THREE.GLSL3, defines,
      uniforms: { tColor: { value: null }, tDepth: { value: null } },
      vertexShader: COPY_VERT, fragmentShader: COPY_FRAG,
      depthTest: false, depthWrite: false, blending: THREE.NoBlending,
    });
    this._quad = new FullScreenQuad(this.material);
    this._recs = new Map(); // mesh -> { hook, prev, own, active }
    this._warned = new Set();
    this._supported = null; this._reversed = false; this._extOf = null;
    this._pending = null; this._pendingFrame = -1; this._pendingSamples = 0; // a skip not warned yet (see header)
    this._vp = new THREE.Vector4();
    this._srcW = 1; this._srcH = 1;
    let W = width, H = height;
    if ((!(W > 0) || !(H > 0)) && renderer && typeof renderer.getDrawingBufferSize === 'function') {
      const v = renderer.getDrawingBufferSize(new THREE.Vector2());
      W = v.x; H = v.y;
    }
    this.setSize(W || 1, H || 1);
  }

  /**
   * Size of the scene target in drawing-buffer px (floor(css * pixelRatio), as PostFX computes it); the copy is
   * max(1, round(size * scale)). Sampling is in normalised uv, so a stale size only costs resolution, never alignment.
   * @param {number} width
   * @param {number} height
   */
  setSize(width, height) {
    this._srcW = width; this._srcH = height; // for setScale
    const px = (v) => { const n = Math.round((Number.isFinite(+v) ? +v : 0) * this.scale); return n > 1 ? n : 1; };
    const W = px(width), H = px(height);
    this.width = W; this.height = H;
    this.target.setSize(W, H);
  }

  /**
   * Copy resolution at runtime (a measurement knob; the constructor's `scale` otherwise): re-sizes the copy from the
   * last setSize and switches the one-texel depth path (scale >= 0.99) on or off (one recompile of the copy shader).
   * @param {number} scale  0 < scale <= 1
   */
  setScale(scale) {
    if (!(scale > 0 && scale <= 1)) throw new Error(`[SceneCopy] scale must be in (0, 1], got ${scale}`);
    this.scale = scale;
    const full = scale >= 0.99, d = this.material.defines;
    if (full !== (d.COPY_FULL !== undefined)) {
      if (full) d.COPY_FULL = ''; else delete d.COPY_FULL;
      this.material.needsUpdate = true;
    }
    this.setSize(this._srcW, this._srcH);
  }

  /**
   * Copy the scene before `mesh` draws: chains mesh.onBeforeRender (the copy first, then the hook it had). Idempotent.
   * @param {THREE.Object3D} mesh
   * @returns {this}
   */
  attach(mesh) {
    const had = this._recs.get(mesh);
    if (had) { had.active = true; return this; }
    const self = this;
    const rec = { hook: null, prev: mesh.onBeforeRender, own: Object.prototype.hasOwnProperty.call(mesh, 'onBeforeRender'), active: true };
    rec.hook = function (renderer, scene, camera, geometry, material, group) {
      if (rec.active) self._before(renderer);
      rec.prev.call(this, renderer, scene, camera, geometry, material, group);
    };
    mesh.onBeforeRender = rec.hook;
    this._recs.set(mesh, rec);
    return this;
  }

  /**
   * Stop copying for `mesh`: restores the hook it had (if another hook was chained over this one since, the copy's
   * hook stays in that chain as a pass-through until the next attach()).
   * @param {THREE.Object3D} mesh
   * @returns {this}
   */
  detach(mesh) {
    const rec = this._recs.get(mesh);
    if (!rec) return this;
    if (mesh.onBeforeRender === rec.hook) {
      if (rec.own) mesh.onBeforeRender = rec.prev; else delete mesh.onBeforeRender;
      this._recs.delete(mesh);
    } else rec.active = false;
    let any = false;
    for (const r of this._recs.values()) if (r.active) { any = true; break; }
    if (!any) this.valid = false; // nothing copies any more: materials reading `valid` stop sampling
    return this;
  }

  /** Detach every mesh, free the target and the quad. */
  dispose() {
    for (const mesh of [...this._recs.keys()]) this.detach(mesh);
    this._recs.clear();
    this.target.dispose();
    this.material.dispose();
    this._quad.dispose();
    this.valid = false;
  }

  // the copy, from an attached mesh's onBeforeRender (inside the scene's renderer.render)
  _before(renderer) {
    const rt = renderer.getRenderTarget(), info = renderer.info.render;
    // a skip from an earlier render() call with no valid copy since: now it is worth a warning
    if (this._pending !== null && info.frame !== this._pendingFrame) this._warnPending();
    let reason = null;
    if (rt === null) reason = 'canvas';
    else if (rt.samples > 0) reason = 'msaa';
    else if (this.depth !== null && !rt.depthTexture) reason = 'nodepth';
    else if (!this._caps(renderer)) reason = 'unsupported';
    else if (this._reversed) reason = 'reversed';
    if (reason !== null) {
      this.valid = false; this.reason = reason;
      if (this._pending === null && !this._warned.has(reason)) {
        this._pending = reason; this._pendingFrame = info.frame; this._pendingSamples = rt !== null ? rt.samples : 0;
      }
      return;
    }
    this._pending = null; // e.g. the transmission pass skipped, the main pass copies: no warning
    if (this.valid && info.frame === this.frame) return; // already copied during this render() call
    const u = this.material.uniforms;
    u.tColor.value = rt.texture;
    u.tDepth.value = this.depth !== null ? rt.depthTexture : null;
    const face = renderer.getActiveCubeFace(), mip = renderer.getActiveMipmapLevel();
    renderer.getCurrentViewport(this._vp);
    const xr = renderer.xr, sm = renderer.shadowMap;
    const xrOn = xr ? xr.enabled : false, smAuto = sm ? sm.autoUpdate : false, autoClear = renderer.autoClear;
    if (xr) xr.enabled = false;          // no XR camera substitution for the quad
    if (sm) sm.autoUpdate = false;       // no shadow-map work in the nested render
    renderer.autoClear = false;          // the nested render must not clear (see `clear`)
    try {
      renderer.setRenderTarget(this.target);
      if (this.clear) renderer.clear(true, false, false);
      this._quad.render(renderer);
    } finally {
      if (xr) xr.enabled = xrOn;
      if (sm) sm.autoUpdate = smAuto;
      renderer.autoClear = autoClear;
      renderer.setRenderTarget(rt, face, mip);
      renderer.state.viewport(this._vp); // an ArrayCamera sub-view (or any viewport) the scene pass had set
    }
    this.valid = true; this.reason = null;
    this.frame = info.frame;
    this.copies++;
  }

  _warnPending() {
    const reason = this._pending;
    this._pending = null;
    if (this._warned.has(reason)) return;
    this._warned.add(reason);
    console.warn(`[SceneCopy] copy skipped: ${WARN[reason]}${reason === 'msaa' ? ` (samples ${this._pendingSamples})` : ''}. Refraction needs a single-sampled scene target with a depth texture: PostFX gbuffer: true, or aa: 'fxaa'.`);
  }

  // renderer capabilities, checked once per renderer: float colour attachments (RGBA16F / R32F) need
  // EXT_color_buffer_float (returns that); a reversed depth buffer is recorded in _reversed
  _caps(renderer) {
    if (renderer !== this._extOf) {
      const ext = renderer.extensions;
      this._supported = !ext || typeof ext.has !== 'function' || ext.has('EXT_color_buffer_float');
      this._reversed = !!renderer.capabilities && renderer.capabilities.reversedDepthBuffer === true;
      this._extOf = renderer;
    }
    return this._supported;
  }
}
