// Headless checks for the dynamics broadphase (src/physics/dynamics/broadphase.js): the pair list against an O(n^2)
// brute force (same keys, same order) on fresh and incrementally moved scenes, history independence, query / raycast
// against brute force, remove, invalid boxes, the radix sort, allocation after warm-up, and update timing.
// Run: node test/dyn-broadphase.mjs   (no browser, no WebGL)
import v8 from 'node:v8';
import { atLeast, bytesPerCall, gate } from './helpers/alloc.mjs';
import vm from 'node:vm';
import * as MathDet from '../src/core/mathDet.js';
import { BodyStore, BODY, FLAG, pairKey, pairA, pairB } from '../src/physics/dynamics/state.js';
import { Broadphase, radixSortU32 } from '../src/physics/dynamics/broadphase.js';

let failures = 0;
const results = [];
function check(name, value, lo = 1, hi = lo, unit = '') {
  if (typeof value === 'boolean') value = value ? 1 : 0;
  const ok = Number.isFinite(value) && value >= lo && value <= hi;
  if (!ok) failures++;
  results.push(`${ok ? 'PASS' : 'FAIL'}  ${name.padEnd(60)} ${fmt(value)} ${unit}   [${fmt(lo).trim()}${hi !== lo ? ' .. ' + fmt(hi).trim() : ''}]`);
  return ok;
}
const fmt = (x) => (!Number.isFinite(x) ? String(x) : Number.isInteger(x) ? String(x) : Math.abs(x) < 1e-3 ? x.toExponential(2) : x.toFixed(3)).padStart(10);
const note = (s) => results.push(`NOTE  ${s}`);

// ---- scene helpers ---------------------------------------------------------------------------------------------
function placeBox(bodies, id, rng, L) {
  const o = id * 6;
  for (let k = 0; k < 3; k++) {
    const s = 0.5 + 2.5 * rng(), c = rng() * L;
    bodies.aabb[o + k] = c - s / 2; bodies.aabb[o + 3 + k] = c + s / 2;
  }
}
function randomizeBody(bodies, id, rng) {
  const r = rng();
  bodies.type[id] = r < 0.1 ? BODY.STATIC : r < 0.15 ? BODY.KINEMATIC : BODY.DYNAMIC;
  bodies.flags[id] = rng() < 0.1 ? FLAG.ASLEEP : 0;
  bodies.layer[id] = 1 << ((rng() * 4) | 0);
  bodies.mask[id] = rng() < 0.6 ? -1 : (rng() * 16) | 0;
}
/** n boxes (0.5-3 m per axis) in an L-metre cube: 10 % static, 5 % kinematic, 10 % asleep, random layer / mask. */
function makeScene(n, L, seed, cap = 4096) {
  const rng = MathDet.mulberry32(seed);
  const bodies = new BodyStore(cap);
  for (let i = 0; i < n; i++) { const id = bodies.alloc(); placeBox(bodies, id, rng, L); randomizeBody(bodies, id, rng); }
  return { bodies, rng };
}
const overlap = (ab, a, b) => {
  const oa = a * 6, ob = b * 6;
  return ab[oa] <= ab[ob + 3] && ab[ob] <= ab[oa + 3] && ab[oa + 1] <= ab[ob + 4] && ab[ob + 1] <= ab[oa + 4] &&
    ab[oa + 2] <= ab[ob + 5] && ab[ob + 2] <= ab[oa + 5];
};
const finiteBox = (ab, o) => { for (let k = 0; k < 6; k++) if (!Number.isFinite(ab[o + k])) return false; return ab[o] <= ab[o + 3] && ab[o + 1] <= ab[o + 4] && ab[o + 2] <= ab[o + 5]; };
/**
 * Reference pair list, independent of the tree: sweep-and-prune on x over the alive ids with valid boxes (sorted by
 * minX), every x-overlapping pair tested on y / z and the pair rule, keys sorted ascending.
 */
function brutePairs(bodies) {
  const { alive, type, flags, layer, mask, aabb: ab, count } = bodies;
  const ids = [];
  for (let id = 0; id < count; id++) if (alive[id] && finiteBox(ab, id * 6)) ids.push(id);
  ids.sort((p, q) => ab[p * 6] - ab[q * 6]);
  const keys = [];
  for (let i = 0; i < ids.length; i++) {
    const a = ids[i], maxX = ab[a * 6 + 3];
    for (let j = i + 1; j < ids.length && ab[ids[j] * 6] <= maxX; j++) {
      const b = ids[j];
      if (type[a] !== BODY.DYNAMIC && type[b] !== BODY.DYNAMIC) continue;
      if ((flags[a] & FLAG.ASLEEP) && (flags[b] & FLAG.ASLEEP)) continue;
      if ((layer[a] & mask[b]) === 0 || (layer[b] & mask[a]) === 0) continue;
      if (!overlap(ab, a, b)) continue;
      keys.push(pairKey(a, b));
    }
  }
  return Uint32Array.from(keys).sort();   // typed-array sort is numeric
}
/** Number of differences between bp.pairs[0 .. pairCount) and the reference keys (0 = same keys, same order). */
function pairDiff(bp, keys) {
  let d = Math.abs(bp.pairCount - keys.length);
  const n = Math.min(bp.pairCount, keys.length);
  for (let i = 0; i < n; i++) if (bp.pairs[i] !== keys[i]) d++;
  return d;
}
function moveFraction(bodies, rng, frac, L, step) {
  for (let id = 0; id < bodies.count; id++) {
    if (!bodies.alive[id] || rng() >= frac) continue;
    const o = id * 6;
    if (rng() < 0.1) { placeBox(bodies, id, rng, L); continue; }          // teleport
    for (let k = 0; k < 3; k++) { const d = (rng() * 2 - 1) * step; bodies.aabb[o + k] += d; bodies.aabb[o + 3 + k] += d; }
  }
}

// ================================================================================================= pairs vs brute force
const sparse = makeScene(2000, 100, 1234);
const dense = makeScene(2000, 25, 5678);
for (const [label, sc] of [['2000 boxes / 100 m cube', sparse], ['2000 boxes / 25 m cube', dense]]) {
  const bp = new Broadphase(16, { margin: 0.1 });   // tiny capacity: exercises every growth path
  bp.update(sc.bodies);
  const ref = brutePairs(sc.bodies);
  check(`pairs == brute force, ${label} (diffs)`, pairDiff(bp, ref), 0, 0);
  check(`  pair count, ${label}`, bp.pairCount, label.includes('100') ? 30 : 2000, label.includes('100') ? 400 : 20000);
  check(`  tree valid after build, ${label}`, bp.validate() === '');
  check(`  tree height, ${label} (AVL bound ~1.44 log2 n)`, bp.stats.height, 10, 22);
  sc.bp = bp;

  // 50 rounds: move 20 % of the bodies (10 % of those teleport), flip a few sleep / layer / type states
  let bad = 0, invalid = 0, maxPairs = 0;
  for (let round = 0; round < 50; round++) {
    moveFraction(sc.bodies, sc.rng, 0.2, label.includes('100') ? 100 : 25, 3);
    for (let id = 0; id < sc.bodies.count; id++) {
      const r = sc.rng();
      if (r < 0.02) sc.bodies.flags[id] ^= FLAG.ASLEEP;
      else if (r < 0.03) { sc.bodies.layer[id] = 1 << ((sc.rng() * 4) | 0); sc.bodies.mask[id] = sc.rng() < 0.5 ? -1 : (sc.rng() * 16) | 0; }
      else if (r < 0.035) sc.bodies.type[id] = (sc.rng() * 3) | 0;
    }
    bp.update(sc.bodies);
    if (pairDiff(bp, brutePairs(sc.bodies)) !== 0) bad++;
    if (round % 10 === 9 && bp.validate() !== '') invalid++;
    if (bp.pairCount > maxPairs) maxPairs = bp.pairCount;
  }
  check(`50 x (20 % moved + state flips): rounds != brute, ${label}`, bad, 0, 0);
  check(`  tree valid every 10 rounds, ${label} (failures)`, invalid, 0, 0);
  note(`${label}: ${maxPairs} pairs max over the rounds, ${bp.stats.candidates} candidates, ${bp.stats.nodes} nodes, height ${bp.stats.height}`);

  // history independence: a fresh broadphase (different margins) on the final state emits the identical array
  for (const margin of [0, 1]) {
    const fresh = new Broadphase(4096, { margin });
    fresh.update(sc.bodies);
    let d = Math.abs(fresh.pairCount - bp.pairCount);
    for (let i = 0; i < Math.min(fresh.pairCount, bp.pairCount); i++) if (fresh.pairs[i] !== bp.pairs[i]) d++;
    check(`  fresh build (margin ${margin}) == incremental pairs, ${label}`, d, 0, 0);
  }
}

// ================================================================================================= query vs brute force
{
  const { bodies, bp } = dense, rng = MathDet.mulberry32(99);
  const out = new Int32Array(4096);
  let bad = 0, total = 0;
  for (let i = 0; i < 200; i++) {
    const cx = rng() * 35 - 5, cy = rng() * 35 - 5, cz = rng() * 35 - 5, hx = rng() * 6, hy = rng() * 6, hz = rng() * 6;
    const filter = rng() < 0.5 ? -1 : (rng() * 16) | 0;
    const n = bp.query(cx - hx, cy - hy, cz - hz, cx + hx, cy + hy, cz + hz, out, filter);
    const ref = [];
    const ab = bodies.aabb;
    for (let id = 0; id < bodies.count; id++) {
      if (!bodies.alive[id] || (bodies.layer[id] & filter) === 0) continue;
      const o = id * 6;
      if (ab[o] <= cx + hx && ab[o + 3] >= cx - hx && ab[o + 1] <= cy + hy && ab[o + 4] >= cy - hy && ab[o + 2] <= cz + hz && ab[o + 5] >= cz - hz) ref.push(id);
    }
    let d = Math.abs(n - ref.length);
    for (let k = 0; k < Math.min(n, ref.length); k++) if (out[k] !== ref[k]) d++;
    if (d) bad++;
    total += n;
  }
  check('query: 200 random boxes == brute force (ascending ids)', bad, 0, 0);
  check('  query: hits over the 200 boxes', total, 200, Infinity);
  // truncation: the lowest ids, full count in queryTotal
  const n = bp.query(-1e3, -1e3, -1e3, 1e3, 1e3, 1e3, out, -1);
  const small = new Int32Array(3);
  const w = bp.query(-1e3, -1e3, -1e3, 1e3, 1e3, 1e3, small, -1);
  check('query: truncation writes out.length lowest ids', w === 3 && small[0] === out[0] && small[1] === out[1] && small[2] === out[2]);
  check('query: queryTotal is the untruncated count', bp.queryTotal, n, n);
  check('query: whole-world box returns every alive body', n, bodies.size, bodies.size);
}

// ================================================================================================= raycast vs brute force
{
  const { bodies, bp } = dense, rng = MathDet.mulberry32(7);
  const out = new Int32Array(4096), ab = bodies.aabb, M = bp.margin;
  // exact segment-vs-box slab test with division (independent of the broadphase's inverse-multiply form)
  const rayHits = (o, pad, ox, oy, oz, d, maxT) => {
    let t0 = 0, t1 = maxT;
    const org = [ox, oy, oz];
    for (let k = 0; k < 3; k++) {
      const lo = ab[o + k] - pad, hi = ab[o + 3 + k] + pad;
      if (d[k] === 0) { if (org[k] < lo || org[k] > hi) return false; continue; }
      let ta = (lo - org[k]) / d[k], tb = (hi - org[k]) / d[k];
      if (ta > tb) { const s = ta; ta = tb; tb = s; }
      if (ta > t0) t0 = ta;
      if (tb < t1) t1 = tb;
      if (t0 > t1) return false;
    }
    return true;
  };
  let missing = 0, extra = 0, unsorted = 0, exactTotal = 0, candTotal = 0;
  for (let i = 0; i < 200; i++) {
    const ox = rng() * 45 - 10, oy = rng() * 45 - 10, oz = rng() * 45 - 10;
    let d = [rng() * 2 - 1, rng() * 2 - 1, rng() * 2 - 1];
    if (i % 5 === 0) d[(i / 5) % 3] = 0;                 // some rays in a coordinate plane
    if (i % 25 === 0) { d = [0, 0, 0]; d[(i / 25) % 3] = rng() < 0.5 ? -1 : 1; }   // some axis-aligned rays
    const len = Math.hypot(d[0], d[1], d[2]); d = d.map((x) => x / len);
    const maxT = 5 + rng() * 60;
    const n = bp.raycast(ox, oy, oz, d[0], d[1], d[2], maxT, out);
    const got = new Set();
    for (let k = 0; k < n; k++) { got.add(out[k]); if (k && out[k] <= out[k - 1]) unsorted++; }
    for (let id = 0; id < bodies.count; id++) {
      if (!bodies.alive[id]) continue;
      const exact = rayHits(id * 6, 0, ox, oy, oz, d, maxT);
      if (exact) { exactTotal++; if (!got.has(id)) missing++; }
      // fat box lies within tight +- 2 margin (sizes never change in this test)
      if (got.has(id) && !rayHits(id * 6, 2 * M + 1e-9, ox, oy, oz, d, maxT)) extra++;
    }
    candTotal += n;
  }
  check('raycast: exact tight-box hits missing from candidates (200 rays)', missing, 0, 0);
  check('raycast: candidates outside tight + 2 margin', extra, 0, 0);
  check('raycast: candidates in ascending id order (violations)', unsorted, 0, 0);
  check('raycast: exact hits over the 200 rays', exactTotal, 100, Infinity);
  note(`raycast: ${candTotal} candidates for ${exactTotal} exact hits over 200 rays`);
}

// ================================================================================================= remove / re-alloc / invalid boxes
{
  const { bodies, bp, rng } = dense;
  const removed = [];
  while (removed.length < 100) {
    const id = (rng() * bodies.count) | 0;
    if (!bodies.alive[id]) continue;
    bodies.free(id);
    if (removed.length % 2 === 0) bp.remove(id);   // half via remove(), half only freed (update drops dead ids)
    removed.push(id);
  }
  bp.update(bodies);
  let refs = 0;
  for (let i = 0; i < bp.pairCount; i++) { const a = pairA(bp.pairs[i]), b = pairB(bp.pairs[i]); if (!bodies.alive[a] || !bodies.alive[b]) refs++; }
  check('remove: pairs referencing a removed id', refs, 0, 0);
  check('remove: pairs == brute force after removal', pairDiff(bp, brutePairs(bodies)), 0, 0);
  check('remove: proxies == live bodies', bp.stats.proxies, bodies.size, bodies.size);
  check('remove: tree valid', bp.validate() === '');
  const out = new Int32Array(4096);
  const n = bp.query(-1e3, -1e3, -1e3, 1e3, 1e3, 1e3, out);
  let qref = 0;
  for (let k = 0; k < n; k++) if (!bodies.alive[out[k]]) qref++;
  check('remove: query never returns a removed id', qref, 0, 0);
  // ids come back through alloc with new boxes / states
  for (let i = 0; i < 100; i++) { const id = bodies.alloc(); placeBox(bodies, id, rng, 25); randomizeBody(bodies, id, rng); }
  bp.update(bodies);
  check('re-alloc: pairs == brute force after ids are reused', pairDiff(bp, brutePairs(bodies)), 0, 0);
  // a ground slab spanning the whole scene (SAH stops at the root: the tree's worst insert) and a degenerate flat box
  const ground = bodies.alloc();
  bodies.aabb.set([-50, -1, -50, 75, 2, 75], ground * 6); bodies.type[ground] = BODY.STATIC; bodies.layer[ground] = 1; bodies.mask[ground] = -1;
  const flat = bodies.alloc();
  bodies.aabb.set([5, 10, 5, 15, 10, 15], flat * 6); bodies.type[flat] = BODY.DYNAMIC; bodies.layer[flat] = 1; bodies.mask[flat] = -1;
  bp.update(bodies);
  check('ground slab + zero-thickness box: pairs == brute force', pairDiff(bp, brutePairs(bodies)), 0, 0);
  check('ground slab + zero-thickness box: tree valid', bp.validate() === '');
  note(`after the ground slab: tree height ${bp.stats.height}, ${bp.pairCount} pairs`);
  // a NaN / inverted box gets no proxy; restoring it brings the body back
  const victim = pairA(bp.pairs[0]), saved = bodies.aabb.slice(victim * 6, victim * 6 + 6);
  bodies.aabb[victim * 6] = NaN;
  bp.update(bodies);
  let nanRefs = 0;
  for (let i = 0; i < bp.pairCount; i++) if (pairA(bp.pairs[i]) === victim || pairB(bp.pairs[i]) === victim) nanRefs++;
  check('invalid box: NaN aabb body is never paired', nanRefs, 0, 0);
  check('invalid box: pairs == brute force', pairDiff(bp, brutePairs(bodies)), 0, 0);
  bodies.aabb.set(saved, victim * 6);
  bp.update(bodies);
  check('invalid box: restored body paired again (== brute force)', pairDiff(bp, brutePairs(bodies)), 0, 0);
  check('invalid box: tree valid', bp.validate() === '');
}

// ================================================================================================= ids above 32767
// pair keys of ids >= 32768 exceed 2^31: 1500 live bodies spread over ids 0 .. 65535 in a 20 m cube
{
  const rng = MathDet.mulberry32(65536);
  const bodies = new BodyStore(65536);
  for (let i = 0; i < 65536; i++) bodies.alloc();
  const keepId = new Uint8Array(65536);
  for (let id = 0; id < 65536; id++) {
    if (rng() < 1500 / 65536 || id === 65535) { keepId[id] = 1; placeBox(bodies, id, rng, 20); randomizeBody(bodies, id, rng); }
  }
  for (let id = 65535; id >= 0; id--) if (!keepId[id]) bodies.free(id);   // descending: O(1) per free in the sorted free list
  const bp = new Broadphase(1024);
  bp.update(bodies);
  let high = 0;
  for (let i = 0; i < bp.pairCount; i++) if (pairB(bp.pairs[i]) >= 32768 && pairA(bp.pairs[i]) >= 32768) high++;
  check('ids up to 65535: pairs == brute force', pairDiff(bp, brutePairs(bodies)), 0, 0);
  check('  pairs with both ids >= 32768 (keys >= 2^31)', high, 50, Infinity);
}

// ================================================================================================= radix sort
{
  const rng = MathDet.mulberry32(42);
  let bad = 0;
  for (const n of [0, 1, 7, 47, 48, 1000, 100000]) {
    const a = new Uint32Array(n), tmp = new Uint32Array(n);
    for (let i = 0; i < n; i++) a[i] = rng() < 0.2 ? (rng() * 64) >>> 0 : (rng() * 4294967296) >>> 0;
    const ref = Array.from(a).sort((x, y) => x - y);
    radixSortU32(a, n, tmp);
    for (let i = 0; i < n; i++) if (a[i] !== ref[i]) { bad++; break; }
  }
  check('radixSortU32 == Array sort (n = 0 .. 1e5, keys up to 2^32 - 1)', bad, 0, 0);
}

// ================================================================================================= allocation after warm-up
// Exact bytes per update of a loop forced into TurboFan, over a GC-free window (test/helpers/alloc.mjs; one 16-byte
// HeapNumber per update would read 16 B), with its control (an array per update) and a second one: { timing: true }
// (performance.now() boxes its result). The sampling heap profiler is no help here: starting it deoptimizes the
// running code, and unoptimized code boxes doubles.
{
  let gc = globalThis.gc;
  if (!gc) { try { v8.setFlagsFromString('--expose-gc'); gc = vm.runInNewContext('gc'); } catch { gc = null; } }
  const { bodies, rng } = makeScene(2000, 100, 31337);
  const bp = new Broadphase(4096);
  // pre-generated moves (1 % of the bodies per update) so the measured loop runs only update() and typed-array writes
  const U = 4000, K = 20, ids = new Int32Array(U * K), dv = new Float64Array(U * K * 3);
  for (let i = 0; i < U * K; i++) { ids[i] = (rng() * 2000) | 0; for (let k = 0; k < 3; k++) dv[i * 3 + k] = (rng() * 2 - 1) * 0.5; }
  let next = 0;
  const runOn = (target) => (count) => {
    for (let c = 0; c < count; c++, next = (next + 1) % U) {
      const ab = bodies.aabb;
      for (let j = next * K, e = j + K; j < e; j++) {
        const o = ids[j] * 6, d = j * 3;
        ab[o] += dv[d]; ab[o + 3] += dv[d]; ab[o + 1] += dv[d + 1]; ab[o + 4] += dv[d + 1]; ab[o + 2] += dv[d + 2]; ab[o + 5] += dv[d + 2];
      }
      target.update(bodies);
    }
  };
  const run = runOn(bp);
  for (let w = 0; w < 100; w++) run(10);   // warm-up: buffer growth
  gate(check, note, 'update() of 2000 bodies (1 % moved)', bytesPerCall(run, 2500, { warm: 1000 }), { unit: 'update' });
  const timed = runOn(new Broadphase(4096, { timing: true }));
  atLeast(check, note, 'control: { timing: true } is detected (B / update)', bytesPerCall(timed, 2500, { warm: 1000 }), 16);
  if (gc) gc();
  const h0 = process.memoryUsage().heapUsed;
  run(200);
  if (gc) gc();
  check('alloc: retained heap growth over 200 updates (gc before/after)', (process.memoryUsage().heapUsed - h0) / 1024, -Infinity, 64, 'KB');
  check('alloc: pairs still == brute force at the end', pairDiff(bp, brutePairs(bodies)), 0, 0);
  if (!gc) note('no gc available: growth measured without a gc first (GC-free rounds still bound the allocation)');
}

// ================================================================================================= timing
for (const [n, limit] of [[2000, 5], [10000, 15]]) {
  const { bodies, rng } = makeScene(n, 100, 2024 + n, 16384);
  const bp = new Broadphase(4096);
  let t = performance.now();
  bp.update(bodies);
  const build = performance.now() - t;
  const U = 250, K = Math.max(1, Math.round(n * 0.01));
  const ids = new Int32Array(U * K), dv = new Float64Array(U * K * 3);
  for (let i = 0; i < U * K; i++) { ids[i] = (rng() * n) | 0; for (let k = 0; k < 3; k++) dv[i * 3 + k] = (rng() * 2 - 1) * 0.5; }
  let ms = 0;
  for (let u = 0; u < U; u++) {
    const ab = bodies.aabb;
    for (let j = u * K, e = j + K; j < e; j++) {
      const o = ids[j] * 6;
      ab[o] += dv[j * 3]; ab[o + 3] += dv[j * 3]; ab[o + 1] += dv[j * 3 + 1]; ab[o + 4] += dv[j * 3 + 1]; ab[o + 2] += dv[j * 3 + 2]; ab[o + 5] += dv[j * 3 + 2];
    }
    t = performance.now();
    bp.update(bodies);
    if (u >= 50) ms += performance.now() - t;
  }
  ms /= U - 50;
  check(`perf: ms per update, ${n} bodies, 1 % moving`, ms, 0, limit, 'ms');
  check(`  pairs == brute force after ${U} updates, ${n} bodies`, pairDiff(bp, brutePairs(bodies)), 0, 0);
  // heavier churn for the record: 20 % of the bodies moving every update
  const K2 = Math.round(n * 0.2);
  let ms20 = 0;
  for (let u = 0; u < 30; u++) {
    const ab = bodies.aabb;
    for (let j = 0; j < K2; j++) {
      const o = ((rng() * n) | 0) * 6;
      for (let k = 0; k < 3; k++) { const d = (rng() * 2 - 1) * 0.5; ab[o + k] += d; ab[o + 3 + k] += d; }
    }
    t = performance.now();
    bp.update(bodies);
    if (u >= 10) ms20 += performance.now() - t;
  }
  ms20 /= 20;
  note(`${n} bodies / 100 m cube: build ${build.toFixed(2)} ms; ${bp.pairCount} pairs, ${bp.stats.candidates} candidates, ` +
    `${bp.stats.nodes} nodes, height ${bp.stats.height}; ${ms.toFixed(3)} ms/update at 1 % moving, ${ms20.toFixed(3)} ms at 20 %`);
}

console.log(results.join('\n'));
console.log(failures ? `\n${failures} check(s) FAILED` : '\nAll broadphase checks passed.');
process.exit(failures ? 1 : 0);
