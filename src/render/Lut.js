// 3D colour-grading LUTs for the final pass: the GLSL lookup, LUTs built in code (identity, any function of r, g, b)
// and a CPU sampler that reproduces the GPU lookup exactly (tests, previews, colour pickers).
//
// Data layout (the same as the Assets 'lut' loader, parseCubeLUT in render/loaders.js, so a LUT from either is
// interchangeable): { size, texture, title, domainMin, domainMax }, texture = THREE.Data3DTexture size^3, RGBAFormat,
// HalfFloatType (Uint16Array of half floats; the loader's `float: true` gives FloatType / Float32Array, which
// lutLookup reads too), LinearFilter, ClampToEdgeWrapping, no mipmaps. Texel (r, g, b) of the grid lives at x = r
// (fastest), y = g, z = b: data index ((b * size + g) * size + r) * 4; grid value i stands for the input i / (size - 1);
// alpha is 1. domainMin / domainMax are carried for the loader's shape ([0, 0, 0] / [1, 1, 1] here); the lookup does
// not apply them (inputs are display-space 0..1).
//
// Lookup (LUT_GLSL, pasted by the compositor into the final pass under FINAL_LUT, applied to the sRGB output before
// grain / dither as `o = mix( o, engineApplyLut( tLUT, o, uLutSize ), uLutStrength )`):
//   uvw = c * (size - 1) / size + 0.5 / size, a hardware trilinear tap. In texel units uvw * size - 0.5 = c * (size - 1):
//   input 0 lands on the first texel centre, input 1 on the last, so grid points come back exactly and the
//   clamp-to-edge wrap is the same as clamping c to 0..1. lutLookup does that in doubles: clamp c to 0..1, texel
//   coordinate c * (size - 1), trilinear blend of the 8 texels (half floats decoded with DataUtils.fromHalfFloat).
//   GPUs blend with fixed-point weights (8 fractional bits on most), so the GPU result differs from lutLookup by up
//   to ~1 / 512 of a grid step (identity, size 16: < 1.3e-4) on top of the half-float storage (< 4.9e-4).
//
// API
//   LUT_GLSL                         vec3 engineApplyLut( highp sampler3D lut, vec3 c, float size )
//   LUT_DEFAULT_SIZE                 16 (grid points per axis of identityLut())
//   identityLut(size = 16)           -> LUT whose lookup returns its input (title 'identity')
//   lutFromFunction(size, fn, title) -> LUT with texel (r, g, b) = fn(r, g, b, out) at r, g, b = i / (size - 1);
//                                       fn writes the result into `out` (a reused 3-array) and returns nothing
//   lutLookup(lut, r, g, b, out = [0, 0, 0]) -> out: CPU trilinear sample, the GPU's coordinate mapping
//   disposeLut(lut)                  frees the texture
// size is an integer 2..256 (the loader's range). No DOM / WebGL: safe in Node.
import * as THREE from 'three';

/** Grid points per axis of identityLut() when no size is given. */
export const LUT_DEFAULT_SIZE = 16;

/**
 * GLSL (paste once): engineApplyLut(lut, c, size) -> the graded colour of display-space c (0..1) through a size^3
 * LUT texture, with the half-texel mapping that puts inputs 0 and 1 on the first and last texel centres.
 */
export const LUT_GLSL = /* glsl */`
vec3 engineApplyLut( highp sampler3D lut, vec3 c, float size ) { float s = ( size - 1.0 ) / size; return texture( lut, c * s + 0.5 / size ).rgb; }
`;

/**
 * @typedef {object} EngineLut
 * @property {number} size grid points per axis
 * @property {THREE.Data3DTexture} texture RGBA half float (or float from the loader), linear, clamp-to-edge
 * @property {string} title
 * @property {number[]} domainMin
 * @property {number[]} domainMax
 */

const identityFn = (r, g, b, out) => { out[0] = r; out[1] = g; out[2] = b; };

function checkSize(size) {
  if (!Number.isInteger(size) || size < 2 || size > 256) throw new Error(`[Lut] size must be an integer in 2..256, got ${size}`);
  return size;
}

/**
 * A LUT whose lookup returns its input (grid value i / (size - 1) at every texel).
 * @param {number} [size=16] grid points per axis (2..256)
 * @returns {EngineLut}
 */
export function identityLut(size = LUT_DEFAULT_SIZE) {
  return lutFromFunction(size, identityFn, 'identity');
}

/**
 * A LUT sampled from a function: texel (r, g, b) holds fn(r, g, b, out) for r, g, b = i / (size - 1).
 * @param {number} size grid points per axis (2..256)
 * @param {(r: number, g: number, b: number, out: number[]) => void} fn writes the graded colour into out[0..2]
 * @param {string} [title='']
 * @returns {EngineLut}
 */
export function lutFromFunction(size, fn, title = '') {
  checkSize(size);
  if (typeof fn !== 'function') throw new Error('[Lut] lutFromFunction: fn must be a function (r, g, b, out) => void');
  const n = size, data = new Uint16Array(n * n * n * 4), out = [0, 0, 0], k = 1 / (n - 1);
  const one = THREE.DataUtils.toHalfFloat(1);
  let i = 0;
  for (let z = 0; z < n; z++) {
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) {
        out[0] = out[1] = out[2] = 0;
        fn(x * k, y * k, z * k, out);
        for (let c = 0; c < 3; c++) {
          const v = +out[c];
          if (!Number.isFinite(v)) throw new Error(`[Lut] lutFromFunction: fn(${x * k}, ${y * k}, ${z * k}) gave a non-finite channel ${c} (${out[c]})`);
          data[i + c] = THREE.DataUtils.toHalfFloat(v);
        }
        data[i + 3] = one;
        i += 4;
      }
    }
  }
  const texture = new THREE.Data3DTexture(data, n, n, n);
  texture.name = `LUT.${title || 'function'}`;
  texture.format = THREE.RGBAFormat;
  texture.type = THREE.HalfFloatType;
  texture.minFilter = texture.magFilter = THREE.LinearFilter;
  texture.wrapS = texture.wrapT = texture.wrapR = THREE.ClampToEdgeWrapping;
  texture.generateMipmaps = false;
  texture.unpackAlignment = 1;
  texture.needsUpdate = true;
  return { size: n, texture, title: String(title), domainMin: [0, 0, 0], domainMax: [1, 1, 1] };
}

/**
 * CPU trilinear sample of a LUT with the GPU mapping of LUT_GLSL (see the header): inputs clamped to 0..1, texel
 * coordinate c * (size - 1). Reads half-float (Uint16Array) or float texture data. Writes into and returns `out`.
 * @param {{size: number, texture: THREE.Data3DTexture}} lut
 * @param {number} r
 * @param {number} g
 * @param {number} b
 * @param {number[]} [out]
 * @returns {number[]}
 */
export function lutLookup(lut, r, g, b, out = [0, 0, 0]) {
  const n = lut.size, data = /** @type {ArrayLike<number>} */ (lut.texture.image.data), half = lut.texture.type === THREE.HalfFloatType;
  const m = n - 1;
  const fx = clamp01(r) * m, fy = clamp01(g) * m, fz = clamp01(b) * m;
  const x0 = Math.min(Math.floor(fx), m), y0 = Math.min(Math.floor(fy), m), z0 = Math.min(Math.floor(fz), m);
  const x1 = Math.min(x0 + 1, m), y1 = Math.min(y0 + 1, m), z1 = Math.min(z0 + 1, m);
  const tx = fx - x0, ty = fy - y0, tz = fz - z0;
  for (let c = 0; c < 3; c++) {
    const c00 = lerp(texel(data, half, n, x0, y0, z0, c), texel(data, half, n, x1, y0, z0, c), tx);
    const c10 = lerp(texel(data, half, n, x0, y1, z0, c), texel(data, half, n, x1, y1, z0, c), tx);
    const c01 = lerp(texel(data, half, n, x0, y0, z1, c), texel(data, half, n, x1, y0, z1, c), tx);
    const c11 = lerp(texel(data, half, n, x0, y1, z1, c), texel(data, half, n, x1, y1, z1, c), tx);
    out[c] = lerp(lerp(c00, c10, ty), lerp(c01, c11, ty), tz);
  }
  return out;
}

// channel c of texel (x, y, z): x fastest, 4 channels per texel
function texel(data, half, n, x, y, z, c) {
  const v = data[((z * n + y) * n + x) * 4 + c];
  return half ? THREE.DataUtils.fromHalfFloat(v) : v;
}

/**
 * Frees the LUT's texture (identityLut / lutFromFunction results; Assets-loaded LUTs are freed by assets.release).
 * @param {{texture: THREE.Texture}} lut
 */
export function disposeLut(lut) {
  if (lut && lut.texture) lut.texture.dispose();
}

const clamp01 = (v) => (v > 0 ? (v < 1 ? v : 1) : 0); // NaN -> 0
const lerp = (a, b, t) => a + (b - a) * t;
