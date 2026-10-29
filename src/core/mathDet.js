// Deterministic math for the simulation step path. Results must be bit-identical on every JS engine, so the
// functions here use only IEEE-754 correctly rounded operations (+ - * / Math.sqrt), negation and comparisons, plus
// 32-bit integer ops (Math.imul, shifts, xor) for hashing. Math.sin / cos / atan / atan2 / acos / exp / pow are
// implementation-defined (V8, SpiderMonkey and JavaScriptCore round them differently): never use them in a step.
// Pure JS, no dependencies, allocation-free (mulberry32 creates its generator once).
// Math.sqrt: ECMA-262 lists it among the implementation-approximated functions, but every engine compiles it to the
// hardware square root (SSE2 sqrtsd, ARM fsqrt), which IEEE-754 requires to be correctly rounded, so in practice it
// is exact and identical everywhere (the usual assumption of deterministic JS simulations).
//
//   mulberry32(seed) -> () => [0, 1)      seeded PRNG, same algorithm as render/noise.js (core cannot import render)
//   hashU32(h, v) -> uint32               mix a uint32 v into running hash h (murmur3_32 block step)
//   hashF64(h, x) -> uint32               mix a float64 by its bit pattern (low then high uint32 word)
//   hashF64Array(arr, h = 2166136261, start = 0, end = arr.length) -> uint32   (any array-like of numbers)
//   detAtan(x), detAtan2(y, x), detAsin(x), detAcos(x)
//   detSqrt (= Math.sqrt, correctly rounded by IEEE-754), clamp(v, a, b)
//
// Hashing. A state hash (positions, velocities, ...) compared across runs or machines is how determinism is
// checked: h = hashF64(h, v) per value, or hashF64Array over a Float64Array / Float32Array. Bit patterns are hashed,
// so -0 and +0 differ (as they should for bit-identity); every NaN hashes as the canonical quiet NaN. The halves are
// read through a module-level Float64Array / Uint32Array view: little-endian, which is every platform browsers and
// Node run on. The value is not finalized (no fmix); compare hashes, do not use their low bits as a uniform random.
//
// Inverse trig. detAtan reduces |x| to [0, 0.66] (atan(x) = pi/2 - atan(1/x) above tan(3pi/8), pi/4 + atan((x-1) /
// (x+1)) above 0.66) and evaluates the degree 4/5 rational minimax approximation of Cephes' atan.c (S. L. Moshier,
// freely distributable; the same constants Go's math.Atan uses), with the pi/2 and pi/4 tails (MOREBITS) added back.
// detAtan2 = atan(|y / x|) placed in its quadrant (pi - q keeps the pi tail), with Math.atan2's special cases
// reproduced exactly (signed zeros, zero x, infinities, NaN). detAcos(x) = 2 atan(sqrt((1 - x) / (1 + x))) (no
// cancellation near +-1: 1 - x and 1 + x are exact there), detAsin(x) = atan(x / sqrt((1 - x)(1 + x))); |x| > 1 or
// NaN -> NaN.
// Measured against V8's fdlibm (test/core.mjs: 1e6 random inputs each, all quadrants, the axes, magnitudes from
// 1e-300 to 1e300): max absolute error detAtan2 4.4e-16, detAcos 4.4e-16 (1 ulp near pi), detAsin 2.2e-16, detAtan
// 2.2e-16. The required bound is 1e-9.

/** Seeded PRNG (mulberry32): returns () => [0, 1). Identical sequence to render/noise.js mulberry32. */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---- hashing -------------------------------------------------------------------------------------------------
const _f64 = new Float64Array(1), _u32 = new Uint32Array(_f64.buffer);

// murmur3_32 block step on int32s
function mix(h, v) {
  let k = Math.imul(v, 0xcc9e2d51);
  k = (k << 15) | (k >>> 17);
  h ^= Math.imul(k, 0x1b873593);
  h = (h << 13) | (h >>> 19);
  return (Math.imul(h, 5) + 0xe6546b64) | 0;
}
function mixF64(h, x) {
  if (x !== x) return mix(mix(h, 0), 0x7ff80000); // canonical NaN
  _f64[0] = x;
  return mix(mix(h, _u32[0] | 0), _u32[1] | 0);
}

/** Mix uint32 `v` into running hash `h`; returns a uint32. */
export function hashU32(h, v) { return mix(h | 0, v | 0) >>> 0; }

/** Mix float64 `x` (its bit pattern) into running hash `h`; returns a uint32. */
export function hashF64(h, x) { return mixF64(h | 0, x) >>> 0; }

/** Hash arr[start .. end) as float64 bit patterns, continuing from `h` (default: the FNV offset basis). */
export function hashF64Array(arr, h = 2166136261, start = 0, end = arr.length) {
  h |= 0;
  for (let i = start; i < end; i++) h = mixF64(h, arr[i]);
  return h >>> 0;
}

// ---- inverse trig (only + - * / sqrt and comparisons) ---------------------------------------------------------
const PIO2 = 1.57079632679489661923, PIO4 = 0.785398163397448309616, PI = 3.14159265358979323846;
const PI_LO = 1.2246467991473531772e-16;       // pi - PI
const MOREBITS = 6.123233995736765886130e-17; // pi/2 - PIO2
const T3P8 = 2.41421356237309504880;          // tan(3pi/8)
const P0 = -8.750608600031904122785e-1, P1 = -1.615753718733365076637e1, P2 = -7.500855792314704667340e1;
const P3 = -1.228866684490136173410e2, P4 = -6.485021904942025371773e1;
const Q0 = 2.485846490142306297962e1, Q1 = 1.650270098316988542046e2, Q2 = 4.328810604912902668951e2;
const Q3 = 4.853903996359136964868e2, Q4 = 1.945506571482613964425e2;

// atan on [0, 0.66]
function xatan(x) {
  const z = x * x;
  return x * (z * ((((P0 * z + P1) * z + P2) * z + P3) * z + P4) / (((((z + Q0) * z + Q1) * z + Q2) * z + Q3) * z + Q4)) + x;
}
// atan for x >= 0 (+Infinity -> pi/2)
function satan(x) {
  if (x <= 0.66) return xatan(x);
  if (x > T3P8) return PIO2 - xatan(1 / x) + MOREBITS;
  return PIO4 + xatan((x - 1) / (x + 1)) + 0.5 * MOREBITS;
}

/** Deterministic Math.atan (odd; atan(+-0) = +-0, atan(+-Infinity) = +-pi/2, NaN -> NaN). */
export function detAtan(x) {
  if (x === 0 || x !== x) return x;
  return x < 0 ? -satan(-x) : satan(x);
}

/** Deterministic Math.atan2(y, x), including its signed-zero / infinity / NaN cases. Max error 4.4e-16. */
export function detAtan2(y, x) {
  if (x !== x || y !== y) return NaN;
  if (y === 0) {
    if (x > 0 || (x === 0 && 1 / x > 0)) return y;  // +-0 for x >= +0
    return 1 / y > 0 ? PI : -PI;                      // x <= -0
  }
  if (x === 0) return y > 0 ? PIO2 : -PIO2;
  const ax = x < 0 ? -x : x, ay = y < 0 ? -y : y;
  if (ax === Infinity) {
    if (ay === Infinity) return x > 0 ? (y > 0 ? PIO4 : -PIO4) : (y > 0 ? 3 * PIO4 : -3 * PIO4);
    return x > 0 ? (y > 0 ? 0 : -0) : (y > 0 ? PI : -PI);
  }
  if (ay === Infinity) return y > 0 ? PIO2 : -PIO2;
  const q = satan(ay / ax); // in [0, pi/2]
  const r = x > 0 ? q : PI - (q - PI_LO);
  return y > 0 ? r : -r;
}

/** Deterministic Math.acos: [-1, 1] -> [0, pi]; |x| > 1 or NaN -> NaN. Max error 4.4e-16. */
export function detAcos(x) {
  if (!(x >= -1 && x <= 1)) return NaN;
  return 2 * satan(Math.sqrt((1 - x) / (1 + x)));
}

/** Deterministic Math.asin: [-1, 1] -> [-pi/2, pi/2] (asin(+-0) = +-0); |x| > 1 or NaN -> NaN. */
export function detAsin(x) {
  if (!(x >= -1 && x <= 1)) return NaN;
  if (x === 0) return x;
  const t = x / Math.sqrt((1 - x) * (1 + x)); // +-Infinity at +-1 -> +-pi/2
  return t < 0 ? -satan(-t) : satan(t);
}

/** Math.sqrt: correctly rounded in practice on every engine (hardware sqrt, see header), so deterministic as is. */
export const detSqrt = Math.sqrt;

/** v clamped to [a, b]. */
export const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
