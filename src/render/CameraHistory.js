// Camera matrices across frames + the sub-pixel jitter of temporal anti-aliasing. One per camera; PostFX creates it
// when the G-buffer is on (`postfx.history`), and the G-buffer velocity output, TAAPass and later screen-space passes
// (SSR, volumetrics) read what it publishes.
//
// Per frame (PostFX.render does both):
//   history.begin(width, height)  after camera.updateMatrixWorld(), before the scene pass (drawing-buffer pixels):
//     1. the previous frame's UNJITTERED view-projection -> GBUFFER_UNIFORMS.enginePrevViewProj and uPrevViewProj
//        (on the first frame and after reset(): this frame's own, so nothing moves and `fresh` is true);
//     2. the projection as the camera set it -> GBUFFER_UNIFORMS.engineUnjitteredProj and uUnjitteredProj; uViewProj,
//        uInvViewProj (camera.matrixWorld * projectionMatrixInverse, not a numeric inverse) and uReproj from it;
//     3. jitter on: this frame's offset of the pattern (pixels, -0.5..0.5) goes into the camera's projection in place,
//        perspective `elements[8] += 2 jx / width`, `elements[9] += 2 jy / height` (orthographic: elements[12] / [13]
//        -= the same), and projectionMatrixInverse is re-inverted, so every pass of the frame (scene, AO, fog)
//        renders and reconstructs positions with the jittered camera. Image content moves by -jitter pixels: the
//        unjittered point at pixel p is drawn at p - jitter, which is the offset TAAPass undoes.
//   history.end()  after the composer: restores the camera's projection and its inverse exactly (copies of the values
//     begin() found), but only while the camera still holds the jittered elements: if something recomputed the
//     projection in between (updateProjectionMatrix), that newer matrix is kept. With jitter off (or a (0, 0) offset)
//     the camera is never touched: bit-identical output.
// The matrices are published with jitter off too (SSR / volumetrics use them). Call camera.updateProjectionMatrix()
// outside begin() .. end(): inside, that frame's remaining passes lose the jitter (end() then keeps the new matrix).
//
// Options: pattern ('halton16' (default: Halton 2,3, 16 samples) | 'halton8' | [[x, y], ...] pixel offsets in
// -0.5..0.5), jitter (true).
// API: begin(width, height), end(), reset() (teleport / cut: the next frame has no history and the jitter pattern
// restarts, so a capture N frames after reset() is reproducible), setPattern(pattern), `jitter` (bool, live), `frame`
// (begin() count, never reset: uFrame), `fresh` (this frame has no previous one), `uniforms`: reused uniform
// objects that pass materials reference directly: uViewProj, uPrevViewProj, uInvViewProj, uUnjitteredProj, uReproj
// (uPrevViewProj * uInvViewProj: current NDC -> previous clip, composed in doubles on the CPU), uJitter / uPrevJitter
// (Vector2, pixels), uFrame (int).
import * as THREE from 'three';
import { GBUFFER_UNIFORMS } from './GBuffer.js';

/** Radical inverse of i in base b (Halton sequence element). */
function halton(i, b) {
  let f = 1, r = 0;
  while (i > 0) { f /= b; r += f * (i % b); i = Math.floor(i / b); }
  return r;
}

/**
 * Jitter offsets of a pattern in pixels, flat [x0, y0, x1, y1, ...] (Float64Array).
 * 'halton16' / 'halton8': Halton (2, 3) elements 1..n minus 0.5; an array of [x, y] pairs is taken as given.
 * @param {'halton16'|'halton8'|number[][]} [pattern]
 */
export function jitterPattern(pattern = 'halton16') {
  if (Array.isArray(pattern)) {
    if (!pattern.length) throw new Error('[CameraHistory] empty jitter pattern');
    const a = new Float64Array(pattern.length * 2);
    pattern.forEach((p, i) => { a[2 * i] = +p[0]; a[2 * i + 1] = +p[1]; });
    return a;
  }
  const n = pattern === 'halton16' ? 16 : pattern === 'halton8' ? 8 : 0;
  if (!n) throw new Error(`[CameraHistory] unknown jitter pattern '${pattern}' ('halton16' | 'halton8' | [[x, y], ...])`);
  const a = new Float64Array(n * 2);
  for (let i = 0; i < n; i++) { a[2 * i] = halton(i + 1, 2) - 0.5; a[2 * i + 1] = halton(i + 1, 3) - 0.5; }
  return a;
}

export class CameraHistory {
  /**
   * @param {THREE.PerspectiveCamera|THREE.OrthographicCamera} camera
   * @param {{ pattern?: string, jitter?: boolean }} [o] pattern ('halton16'), jitter (true)
   */
  constructor(camera, { pattern = 'halton16', jitter = true } = {}) {
    this.camera = camera;
    /** Apply the sub-pixel jitter in begin() (live; the matrices are published either way). */
    this.jitter = !!jitter;
    this.frame = 0;
    this._k = 0; // position in the jitter pattern (restarts on reset())
    /** True during a frame that has no previous one (first frame, after reset()). */
    this.fresh = true;
    this._reset = true;
    this._applied = false; // begin() jittered the camera; end() restores
    this._ji = 8; // the two projection elements the jitter changed (8 / 9, orthographic 12 / 13) and their values
    this._je0 = 0; this._je1 = 0;
    this._proj = new THREE.Matrix4(); // the camera's own projection / inverse, as begin() found them
    this._projInv = new THREE.Matrix4();
    this.setPattern(pattern);
    this.uniforms = {
      uViewProj: { value: new THREE.Matrix4() },
      uPrevViewProj: { value: new THREE.Matrix4() },
      uInvViewProj: { value: new THREE.Matrix4() },
      uUnjitteredProj: { value: new THREE.Matrix4() },
      uReproj: { value: new THREE.Matrix4() },
      uJitter: { value: new THREE.Vector2() },
      uPrevJitter: { value: new THREE.Vector2() },
      uFrame: { value: 0 },
    };
  }

  /** Jitter pattern: 'halton16' | 'halton8' | [[x, y], ...] (pixels, -0.5..0.5). */
  setPattern(pattern) {
    this._seq = jitterPattern(pattern);
    this.pattern = pattern;
  }

  /** Number of jitter positions in the pattern. */
  get patternLength() { return this._seq.length >> 1; }

  /**
   * The next begin() starts a new history: no previous matrices (velocity 0), `fresh` (TAA outputs its input), and the
   * jitter pattern from its first position.
   */
  reset() { this._reset = true; }

  /**
   * Before the scene pass: publish this frame's matrices, then jitter the camera (see the header).
   * @param {number} width drawing-buffer width (px)
   * @param {number} height drawing-buffer height (px)
   */
  begin(width, height) {
    const cam = this.camera, u = this.uniforms;
    if (this._applied) this.end(); // begin() twice without end(): never stack two jitters
    const vp = u.uViewProj.value, prev = u.uPrevViewProj.value, inv = u.uInvViewProj.value;
    const fresh = this.fresh = this._reset;
    this._reset = false;
    if (fresh) this._k = 0;
    if (!fresh) prev.copy(vp); // last frame's unjittered view-projection
    u.uUnjitteredProj.value.copy(cam.projectionMatrix);
    vp.multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse);
    inv.multiplyMatrices(cam.matrixWorld, cam.projectionMatrixInverse);
    if (fresh) prev.copy(vp);
    u.uReproj.value.multiplyMatrices(prev, inv);
    GBUFFER_UNIFORMS.enginePrevViewProj.value.copy(prev);
    GBUFFER_UNIFORMS.engineUnjitteredProj.value.copy(cam.projectionMatrix);

    const j = u.uJitter.value, s = this._seq, i = (this._k++ % (s.length >> 1)) * 2; // frames since reset()
    u.uPrevJitter.value.copy(j);
    if (this.jitter && width > 0 && height > 0 && (s[i] !== 0 || s[i + 1] !== 0)) {
      j.set(s[i], s[i + 1]);
      this._proj.copy(cam.projectionMatrix);
      this._projInv.copy(cam.projectionMatrixInverse);
      const e = cam.projectionMatrix.elements, dx = j.x * 2 / width, dy = j.y * 2 / height;
      // perspective: clip.x += dx * z_view, w = -z_view -> ndc.x -= dx; orthographic (w = 1): translate by -dx
      const k = this._ji = cam.isOrthographicCamera ? 12 : 8;
      if (k === 12) { e[12] -= dx; e[13] -= dy; } else { e[8] += dx; e[9] += dy; }
      this._je0 = e[k]; this._je1 = e[k + 1];
      cam.projectionMatrixInverse.copy(cam.projectionMatrix).invert();
      this._applied = true;
    } else j.set(0, 0);
    u.uFrame.value = this.frame++;
  }

  /**
   * After the composer: the camera's projection and inverse exactly as begin() found them, if the camera still holds
   * the jittered matrix (else it was recomputed meanwhile and is kept).
   */
  end() {
    if (!this._applied) return;
    this._applied = false;
    const cam = this.camera, e = cam.projectionMatrix.elements, k = this._ji;
    if (e[k] !== this._je0 || e[k + 1] !== this._je1) return;
    cam.projectionMatrix.copy(this._proj);
    cam.projectionMatrixInverse.copy(this._projInv);
  }
}
