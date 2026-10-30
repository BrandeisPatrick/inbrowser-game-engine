// GPU particles, CPU side: the module must import without WebGL, and the spawn bookkeeping that feeds the GPU
// (GpuParticleStaging: sequential slot ring, staging texels, per-slot attributes, per-frame staging limit, draw
// window, live count) must be deterministic. Run: node --import ./test/helpers/dom-shim.mjs test/gpu-particles.mjs
// (the shim is optional: nothing here touches the DOM).
import { GpuParticleSystem, GpuParticleStaging, packColor, unpackColor, packRotation, hash32 } from '../src/render/GpuParticles.js';

let failures = 0;
const results = [];
const fmt = (x) => (typeof x === 'number' ? (Number.isInteger(x) ? String(x) : x.toFixed(5)) : String(x));
function check(name, value, lo, hi = lo) {
  const ok = typeof value === 'number' ? Number.isFinite(value) && value >= lo && value <= hi : value === lo;
  if (!ok) failures++;
  results.push(`${ok ? 'PASS' : 'FAIL'}  ${name.padEnd(62)} ${fmt(value).padStart(10)}   [${fmt(lo)}${hi !== lo ? ' .. ' + fmt(hi) : ''}]`);
  return ok;
}
// spawn k with recognisable values: position = (k, 2k, 3k), velocity = (-k, 0.5, 0), life 1 + k / 100
const spawnK = (st, k, life = 1 + k / 100) => st.spawn(k, 2 * k, 3 * k, -k, 0.5, 0, life, 0.4, 2.5, 0.3, 0.62, 0.45, 0.31, 0.72, 0.6, 0.47, 1.25, -0.15);

// --- module / support --------------------------------------------------------------------------------------------
check('module imports in Node (no WebGL)', typeof GpuParticleSystem, 'function');
check('isSupported(undefined)', GpuParticleSystem.isSupported(undefined), false);
const fake = (ext) => ({ capabilities: { isWebGL2: true, maxVertexTextures: 16 }, extensions: { has: (n) => ext && n === 'EXT_color_buffer_float' } });
check('isSupported: no EXT_color_buffer_float', GpuParticleSystem.isSupported(fake(false)), false);
check('isSupported: WebGL2 + EXT_color_buffer_float', GpuParticleSystem.isSupported(fake(true)), true);
check('isSupported: no vertex textures', GpuParticleSystem.isSupported({ ...fake(true), capabilities: { maxVertexTextures: 0 } }), false);
let msg = '';
try { new GpuParticleSystem(fake(false)); } catch (e) { msg = e.message; }
check('constructor without support throws a clear error', /EXT_color_buffer_float/.test(msg) && /ParticleSystem/.test(msg), true);

// --- layout ------------------------------------------------------------------------------------------------------
const big = new GpuParticleStaging({ max: 65536, stagingPerFrame: 4096 });
check('65536 slots: texture width', big.width, 1024);
check('65536 slots: texture rows', big.height, 64);
check('staging 4096: width x rows (2 texels per spawn)', big.stagingWidth * 100 + big.stagingHeight, 1024 * 100 + 8);
const small = new GpuParticleStaging({ max: 100, stagingPerFrame: 4096 });
check('100 slots: 128 x 1 texture', small.width * 10 + small.height, 1281);
check('staging clamped to the ring size', small.capacity, 100);

// --- deterministic slots + staging writes --------------------------------------------------------------------------
{
  const st = new GpuParticleStaging({ max: 16, stagingPerFrame: 8 });
  let seq = '';
  for (let k = 0; k < 5; k++) seq += spawnK(st, k) + ',';
  check('sequential slots 0..4', seq, '0,1,2,3,4,');
  check('head after 5 spawns', st.head, 5);
  check('staged this frame', st.n, 5);
  const S = st.staging;
  check('staging texel 2k = (x, y, z, age 0)', [S[3 * 8], S[3 * 8 + 1], S[3 * 8 + 2], S[3 * 8 + 3]].join(), '3,6,9,0');
  check('staging texel 2k+1 = (vx, vy, vz, life)', S[3 * 8 + 4] === -3 && S[3 * 8 + 5] === 0.5 && Math.abs(S[3 * 8 + 7] - 1.03) < 1e-6, true);
  const A = st.attrA, B = st.attrB;
  check('attrA slot 3 = size0, size1, drag, grav', [A[12], A[13], A[14], A[15]].map((v) => +v.toFixed(3)).join(), '0.4,2.5,1.25,-0.15');
  check('attrB slot 3 alpha0', +B[12].toFixed(4), 0.3);
  const c = unpackColor(B[14], [0, 0, 0]), c1 = unpackColor(B[15], [0, 0, 0]);
  check('colour0 round trip (max error)', Math.max(Math.abs(c[0] - 0.62), Math.abs(c[1] - 0.45), Math.abs(c[2] - 0.31)), 0, 0.004);
  check('colour1 round trip (max error)', Math.max(Math.abs(c1[0] - 0.72), Math.abs(c1[1] - 0.6), Math.abs(c1[2] - 0.47)), 0, 0.004);
  check('packed attributes are exact float32 integers < 2^24', [B[13], B[14], B[15]].every((v) => Number.isInteger(v) && v < 16777216 && Math.fround(v) === v), true);
  const out = st.stagedSlots(new Int32Array(4));
  check('staged slot range', Array.from(out).join(), '0,4,-1,-1');
  check('draw window = 5 slots from 0', st.tail * 100 + st.window, 5);
}

// --- per-frame staging limit + wraparound + overwrite of the oldest -------------------------------------------------
{
  const st = new GpuParticleStaging({ max: 10, stagingPerFrame: 7 });
  let r = '';
  for (let k = 0; k < 8; k++) r += spawnK(st, k, 100) + ',';
  check('8th spawn of a 7-per-frame staging returns -1', r, '0,1,2,3,4,5,6,-1,');
  check('dropped counter', st.dropped, 1);
  st.advance(1 / 60); st.flush();
  check('flush: next frame starts at the head', st.start * 10 + st.n, 70);
  r = '';
  for (let k = 0; k < 7; k++) r += spawnK(st, 10 + k, 100) + ',';
  check('frame 2 wraps the ring: slots 7,8,9,0,1,2,3', r, '7,8,9,0,1,2,3,');
  const out = st.stagedSlots(new Int32Array(4));
  check('frame 2 staged slots split at the wrap', Array.from(out).join(), '7,9,0,3');
  check('4 live slots overwritten', st.overwritten, 4);
  check('ring full: window = max, tail = head', st.window * 100 + st.tail * 10 + st.head, 1044);
  check('live count = max (overwrites replace)', st.live, 10);
  check('slot 0 now holds frame-2 spawn #3 (x = 13)', st.staging[3 * 8], 13);
  const rows = st.windowRows(new Int32Array(2));
  check('wrapped window covers every row', rows[0] * 10 + rows[1], 1);
}

// --- texel rows of the draw window (compute pass rows) ---------------------------------------------------------------
{
  const st = new GpuParticleStaging({ max: 4096, stagingPerFrame: 4096 }); // 1024 x 4
  for (let k = 0; k < 1500; k++) spawnK(st, k, k < 1100 ? 0.1 : 5);
  st.advance(0.2); st.flush();
  check('short-lived slots leave the window from the tail', st.tail * 10000 + st.window, 1100 * 10000 + 400);
  const rows = st.windowRows(new Int32Array(2));
  check('window rows: first 1, count 1', rows[0] * 10 + rows[1], 11);
  check('live count after 0.2 s', st.live, 400);
}

// --- window spanning several rows without a wrap; staging a whole ring in one frame; a > 64 s step ------------------
{
  const st = new GpuParticleStaging({ max: 4096, stagingPerFrame: 4096 }); // 1024 x 4
  for (let k = 0; k < 3500; k++) spawnK(st, k, k < 1100 ? 0.1 : 5);
  st.advance(0.2); st.flush();
  const rows = st.windowRows(new Int32Array(2));
  check('window slots 1100..3499 (tail row 1, head row 3): rows [1, 3]', rows[0] * 10 + rows[1], 13);
}
{
  const st = new GpuParticleStaging({ max: 10, stagingPerFrame: 10 });
  for (let k = 0; k < 10; k++) spawnK(st, k, 100);
  const out = st.stagedSlots(new Int32Array(4));
  check('n === max from slot 0: one range 0..9', Array.from(out).join(), '0,9,-1,-1');
  check('n === max: 11th spawn hits the staging limit', spawnK(st, 10, 100), -1);
  st.advance(1 / 60); st.flush();
  for (let k = 0; k < 3; k++) spawnK(st, k, 100);
  st.advance(1 / 60); st.flush();
  for (let k = 0; k < 10; k++) spawnK(st, k, 100);
  st.stagedSlots(out);
  check('n === max from slot 3: ranges 3..9 and 0..2', Array.from(out).join(), '3,9,0,2');
  check('overwrites: 3 (second update) + 10 (third: the whole ring)', st.overwritten, 13);
}
{
  const st = new GpuParticleStaging({ max: 64, stagingPerFrame: 64 });
  spawnK(st, 0, 1000); for (let k = 1; k < 6; k++) spawnK(st, k, 0.5);
  st.flush();
  st.advance(100); // >= 64 s: the histogram resets in one go
  check('step >= 64 s resets the histogram (lives > 64 s count as dead)', st.live, 0);
  check('the 1000 s particle still holds the draw window (tail 0, 6 slots)', st.tail * 100 + st.window, 6);
  spawnK(st, 6, 1);
  check('counting resumes after the reset', st.live, 1);
  st.advance(2);
  check('and ends after the new particle\'s life', st.live, 0);
  check('overwriting a reset slot is not counted twice', (spawnK(st, 7, 1), st.live), 1);
}

// --- live count + window over time --------------------------------------------------------------------------------
{
  const st = new GpuParticleStaging({ max: 64, stagingPerFrame: 64 });
  for (let k = 0; k < 10; k++) spawnK(st, k, 1.0);
  for (let k = 0; k < 5; k++) spawnK(st, k, 3.0);
  st.flush();
  check('count after spawning 15', st.live, 15);
  st.advance(0.5); st.flush();
  check('count at 0.5 s', st.live, 15);
  for (let i = 0; i < 40; i++) st.advance(1 / 60); // 1.17 s
  check('count after the 1 s lives ended', st.live, 5);
  check('window = the 5 long-lived slots from slot 10', st.tail * 100 + st.window, 10 * 100 + 5);
  st.advance(2);
  check('all dead after 3 s', st.live * 100 + st.window, 0);
  st.reset();
  check('reset restarts the ring at slot 0', spawnK(st, 0), 0);
}

// --- a long-lived particle holds the window (conservative), the count stays exact ---------------------------------
{
  const st = new GpuParticleStaging({ max: 64, stagingPerFrame: 64 });
  spawnK(st, 0, 10); for (let k = 1; k < 20; k++) spawnK(st, k, 0.5);
  st.flush(); st.advance(1);
  check('live = 1 (long-lived)', st.live, 1);
  check('window keeps 0..19 behind the long-lived slot 0', st.tail * 100 + st.window, 20);
}

// --- determinism: two systems fed the same sequence hold identical bytes ------------------------------------------
{
  const run = () => {
    const st = new GpuParticleStaging({ max: 300, stagingPerFrame: 50, spin: 0.7 });
    let x = 12345; const rnd = () => ((x = hash32(x)) / 4294967296);
    for (let f = 0; f < 20; f++) {
      for (let i = 0; i < 40; i++) st.spawn(rnd() * 10, rnd(), rnd(), rnd() - 0.5, rnd(), 0, 0.5 + rnd() * 3, 0.4, 2, 0.2, rnd(), rnd(), rnd(), rnd(), rnd(), rnd(), 1.25, -0.15);
      st.advance(1 / 60); st.flush();
    }
    return st;
  };
  const a = run(), b = run();
  const same = (u, v) => Buffer.compare(Buffer.from(u.buffer), Buffer.from(v.buffer)) === 0;
  check('same spawn sequence -> identical attrA / attrB / staging bytes', same(a.attrA, b.attrA) && same(a.attrB, b.attrB) && same(a.staging, b.staging), true);
  check('same head / tail / window / live', [a.head, a.tail, a.window, a.live].join() === [b.head, b.tail, b.window, b.live].join(), true);
  check('ring wrapped (800 spawns into 300 slots)', a.spawned, 800);
  // rotation: hashed from the spawn serial, independent of Math.random
  const st = new GpuParticleStaging({ max: 8, spin: 1 });
  const saved = Math.random; Math.random = () => { throw new Error('Math.random used'); };
  let threw = false; try { spawnK(st, 0); } catch { threw = true; } finally { Math.random = saved; }
  check('spawn never calls Math.random', threw, false);
  spawnK(st, 1, 1); st.spawn(0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, Math.PI, -2);
  const rp = st.attrB[2 * 4 + 1], ang = (rp >> 12) * (2 * Math.PI / 4096), vel = ((rp & 4095) - 2048) * (16 / 2048);
  check('explicit rot / rotV arguments round trip', Math.abs(ang - Math.PI) < 1e-3 && Math.abs(vel + 2) < 1e-2, true);
}

// --- packing -------------------------------------------------------------------------------------------------------
{
  const o = [0, 0, 0];
  unpackColor(packColor(2.0, 1.0, 0.05), o);
  // shared exponent: every channel is quantised in steps of (max channel scale) / 127, so the error is bounded by the
  // brightest channel, not by the channel's own value (0.05 next to 2.0 comes back within ~6 % of itself)
  const errHdr = Math.max(Math.abs(o[0] - 2), Math.abs(o[1] - 1), Math.abs(o[2] - 0.05));
  check('HDR colour (2, 1, 0.05): abs error <= max channel / 127', errHdr, 0, 2 / 127);
  unpackColor(packColor(0.02, 0.01, 0.005), o);
  check('dark colour (0.02, 0.01, 0.005) round trip, max error', Math.max(Math.abs(o[0] - 0.02), Math.abs(o[1] - 0.01), Math.abs(o[2] - 0.005)), 0, 2.5e-4);
  check('colour clamps at 8 and below 0', packColor(100, -1, NaN), 7 * 2097152 + 127 * 16384);
  check('rotation pack range', packRotation(-0.001, 1e9) >= 0 && packRotation(-0.001, 1e9) < 16777216, true);
}

console.log(results.join('\n'));
console.log(failures ? `\n${failures} check(s) FAILED` : `\nall ${results.length} checks passed`);
process.exit(failures ? 1 : 0);
