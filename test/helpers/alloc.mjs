// Deterministic allocation gates for the headless suites. "This code allocates nothing" is a statement about the
// optimised code: V8 running a function in Ignition / Sparkplug boxes a 16 B HeapNumber for every double it reads
// from a Float64Array or passes to a call it does not inline, and when it tiers up depends on how fast the machine
// is (a shared 2-core CI runner tiers later than an M4; dyn-fracture's impactBreak ran unoptimised in ~1 run in 4).
// So every gate here forces the loop under test into TurboFan (natives syntax, enabled at runtime:
// %PrepareFunctionForOptimization / %OptimizeFunctionOnNextCall), optionally the callees it does not inline too, and
// counts the bytes of one GC-free window:
//   - heapUsed (process.memoryUsage) across a window with no GC (GCProfiler) is exact to the byte: 1e3 / 1e4 / 1e5
//     two-field objects read 33 568 / 321 680 / 3 202 912 B on every repeat, 32 B each over a constant floor (the
//     measurement's own result objects, calibrated with run(0)). v8 total_allocated_bytes is no help here: it moves
//     in whole allocation-buffer steps (~256 KB, also when read twice with nothing in between).
//   - a window is valid only when no GC ran in it and the forced functions are still TurboFan code afterwards; an
//     invalid window is reported and retried after a fresh warm-up, and after `tries` invalid windows the reading is
//     NaN, which every suite's check() fails. A deopt never passes silently.
//   - every reading comes with a control: the same loop plus one array per unit, measured the same way, which must
//     read above the gate's budget, so the gate is known to see an allocation of that size.
// The bit layout of %GetOptimizationStatus changes between V8 versions, so the TurboFan bits are calibrated on probe
// functions at load. Needs Node >= 26 (the V8 the budgets and CI are on, see .github/workflows/ci.yml): elsewhere
// ALLOC.ok is false and the suites skip these gates with ALLOC.reason as a note.
import { readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import v8 from 'node:v8';
import vm from 'node:vm';

let gc = globalThis.gc ?? null;
if (!gc) { try { v8.setFlagsFromString('--expose-gc'); gc = vm.runInNewContext('gc'); } catch { gc = null; } }

let prep = null, opt = null, status = null, never = null, finalize = () => {}, TURBO = 0, MARKED = 0;
const ALLOC = { ok: false, reason: '' };
try {
  const major = Number(process.versions.node.split('.')[0]);
  if (major < 26) throw new Error(`Node ${process.versions.node}: the allocation gates need Node >= 26`);
  if (!gc) throw new Error('no gc() (--expose-gc could not be set)');
  if (typeof v8.GCProfiler !== 'function') throw new Error('no v8.GCProfiler');
  v8.setFlagsFromString('--allow-natives-syntax');
  const natives = (body) => new Function('f', body);
  prep = natives('%PrepareFunctionForOptimization(f);');
  opt = natives('%OptimizeFunctionOnNextCall(f);');
  status = natives('return %GetOptimizationStatus(f);');
  never = natives('%NeverOptimizeFunction(f);');
  try { finalize = natives('%FinalizeOptimization();'); finalize(); } catch { finalize = () => {}; }
  // calibration: a TurboFan'd probe against an interpreted one (distinct sources: closures of one source share their
  // SharedFunctionInfo and so their tiering state). %OptimizeFunctionOnNextCall without the prepare is a fatal CHECK.
  const probe = (k) => new Function('x', `return x + ${k}; // test/helpers/alloc.mjs probe ${k}`);
  const pt = probe(1), pi = probe(2);
  prep(pt); pt(1); pt(2); opt(pt);
  const sMarked = status(pt);
  pt(3); pi(1);
  TURBO = status(pt) & ~status(pi); MARKED = sMarked & ~status(pi);
  if (!TURBO || !MARKED) throw new Error(`could not calibrate %GetOptimizationStatus (V8 ${process.versions.v8})`);
  ALLOC.ok = true;
} catch (err) {
  ALLOC.reason = `allocation gates skipped: ${err.message}`;
}
export { ALLOC };

const turbo = (f) => (status(f) & TURBO) === TURBO;
const isNative = (f) => /\{\s*\[native code\]\s*\}\s*$/.test(Function.prototype.toString.call(f));

/**
 * The JS functions of modules / classes / objects, for bytesPerCall's `force`: a function itself, a class's static
 * methods and its prototype's methods and accessors, an object's (module namespace's) function-valued properties and
 * the classes among them. Native functions are skipped.
 */
export function functionsOf(...sources) {
  const out = new Set();
  const addOwn = (obj) => {
    for (const k of Object.getOwnPropertyNames(obj)) {
      if (k === 'constructor') continue;
      const d = Object.getOwnPropertyDescriptor(obj, k);
      for (const f of [d.value, d.get, d.set]) if (typeof f === 'function') addFn(f);
    }
  };
  const addFn = (f) => {
    if (out.has(f) || isNative(f)) return;
    out.add(f);
    if (/^class\b/.test(Function.prototype.toString.call(f))) { addOwn(f); if (f.prototype) addOwn(f.prototype); }
  };
  for (const s of sources) {
    if (typeof s === 'function') addFn(s);
    else if (s && typeof s === 'object') for (const k of Object.keys(s)) { const v = s[k]; if (typeof v === 'function') addFn(v); }
  }
  return [...out];
}

/** functionsOf() every module of a source directory (a file: URL ending in '/'): `force` for a whole subsystem. */
export async function functionsOfDir(dir) {
  const files = readdirSync(fileURLToPath(dir)).filter((f) => f.endsWith('.js')).sort();
  return functionsOf(...(await Promise.all(files.map((f) => import(new URL(f, dir).href)))));
}

const window = (run, n) => {
  gc();
  const prof = new v8.GCProfiler();
  prof.start();
  const h0 = process.memoryUsage().heapUsed;
  run(n);
  const h1 = process.memoryUsage().heapUsed;
  return { bytes: h1 - h0, gcs: prof.stop().statistics.length };
};
// The measuring code itself keeps one tier: never optimised, warmed here (process.memoryUsage builds its result
// object after reading the heap, inside the window: tiering up mid-gate moved the floor from ~3.5 KB to 256 B)
if (ALLOC.ok) {
  never(window); never(process.memoryUsage);
  const idle = (n) => n;
  for (let k = 0; k < 200; k++) window(idle, 0);
}

// One forced, validated window of run(n): { per, bytes, n, floor, tries, why }. The loop is prepared before its
// warm-up and optimised on its next call. The `force` callees are prepared only after the loop has compiled: a
// function prepared earlier no longer tiers up by itself and changes what the loop inlines (FractureMesh.update
// prepared before its loop compiled read 62 B / frame; prepared after it, 0.3, as with V8's own tiering). Forced then,
// they compile on the loop's next call outside the window if it did not inline them, and stay marked (never entered)
// if it did; each must be TurboFan or still marked after the window. Without them a whole-subsystem loop reads what
// its callees' tiers happen to be (PostFX.render 160-195 B / frame with its passes in Maglev or TurboFan, a dynamics
// step 59-185 B). The floor is run(0) through the same path.
// A window with a GC, or after which a forced function is no longer TurboFan, is invalid: reported, re-warmed, retried
// (after a GC with half the units: under load the young generation is smaller; ~1 MB of garbage triggered a scavenge
// on the M4's efficiency cores). So is a reading over `budget` (a callee V8 has not tiered up yet boxes too; a real
// allocation in TurboFan code reads the same on every try, so it still fails): every such reading is listed in `why`.
function measure(run, n, { warm, tries, budget, force = [] }) {
  const why = [];
  prep(run);
  run(warm);
  for (let t = 1; t <= tries; t++) {
    opt(run);
    run(1);
    if (force.length) { for (const f of force) { prep(f); opt(f); } run(1); }
    finalize();
    let floor = Infinity;
    for (let k = 0; k < 3; k++) floor = Math.min(floor, window(run, 0).bytes);
    const w = window(run, n), per = (w.bytes - floor) / n;
    const cold = force.filter((f) => !turbo(f) && !(status(f) & MARKED)), valid = w.gcs === 0 && turbo(run) && cold.length === 0;
    if (valid && (per <= budget || t === tries)) return { per, bytes: w.bytes - floor, n, floor, tries: t, why };
    why.push(`try ${t}: ${w.gcs ? `${w.gcs} GC(s) in the window of ${n}` : !turbo(run) ? 'the loop deoptimised'
      : cold.length ? `deoptimised: ${cold.slice(0, 3).map((f) => f.name || '(anonymous)').join(', ')}${cold.length > 3 ? ` and ${cold.length - 3} more` : ''}` : `${per.toFixed(2)} B / unit`}`);
    if (w.gcs) n = Math.max(1, Math.floor(n / 2));
    prep(run);
    run(warm);
  }
  return { per: NaN, bytes: NaN, n, floor: NaN, tries, why };
}

let controls = 0;
/**
 * Bytes allocated per run(1) by TurboFan code, counted exactly over one GC-free window of run(n), plus its control.
 * run(k) does k units of work: a loop over the calls under test. Accumulate results into a sink (nothing may be dead
 * code), and pass no freshly computed doubles to calls (TurboFan boxes a double argument of a call it does not
 * inline: 16 B charged to the test's own loop; take them from an array of boxed values instead, see BOXED).
 * `force`: the callees to force too, after the loop has compiled (see measure(); functionsOf / functionsOfDir).
 * run(warm) (default max(2000, n)) first collects the type feedback TurboFan compiles from. Keep n x (bytes per
 * unit) well under 1 MB, or the window holds a GC (it is then retried with fewer units). The control,
 * { for (i < k) { run(1); keep = new Array(len); } } (its own source, so its own tiering state), allocates more than
 * `budget` bytes per unit (len = ceil(budget / 8), at least 1: >= 56 B) over ~256 KB at most, so a control reading
 * above the budget shows the gate would catch that allocation.
 * @returns {{ per: number, bytes: number, n: number, floor: number, tries: number, why: string[], control: number }}
 *   per / control in B per unit, NaN (why says why) when no valid window was found in `tries`
 */
export function bytesPerCall(run, n, { warm = Math.max(2000, n), tries = 5, budget = 1, force = [] } = {}) {
  if (!ALLOC.ok) return { per: NaN, bytes: NaN, n, floor: NaN, tries: 0, why: [ALLOC.reason], control: NaN };
  const m = measure(run, n, { warm, tries, budget, force });
  const len = Number.isFinite(budget) ? Math.max(1, Math.ceil(budget / 8)) : 1;
  const ctl = new Function('run', 'box', `// test/helpers/alloc.mjs control ${controls++}
    return (k) => { for (let i = 0; i < k; i++) { run(1); box.keep = new Array(${len}); } };`)(run, { keep: null });
  const cn = Math.max(1, Math.min(n, Math.floor(262144 / (48 + 8 * len))));
  const c = measure(ctl, cn, { warm: Math.min(warm, 200), tries, budget: Infinity, force });
  if (c.why.length) m.why.push(`control: ${c.why.join('; ')}`);
  return { ...m, control: c.per };
}

/**
 * Numbers as boxed values for a measured loop's call arguments: an array with generic (tagged) elements, so xs[i] is
 * the stored HeapNumber, passed on as it is (a double array or arithmetic would box a fresh one per call).
 */
export function BOXED(...xs) { const a = [null, ...xs]; a.shift(); return a; }

/** One line for a note(): the reading, its window, the control and any invalid windows retried. */
export function describe(m, unit = 'call') {
  if (!ALLOC.ok) return ALLOC.reason;
  const head = Number.isFinite(m.per)
    ? `${m.per.toFixed(3)} B / ${unit} (${m.bytes} B over ${m.n} in TurboFan, GC-free; floor ${m.floor} B; try ${m.tries}); control ${Number.isFinite(m.control) ? m.control.toFixed(1) : 'NaN'} B / ${unit}`
    : `no valid window in ${m.tries} tries`;
  return m.why.length ? `${head}; ${m.why.join('; ')}` : head;
}

/**
 * Report one gate through a suite's check(name, value, lo, hi, unit) and note(line): bytes per unit within
 * [0, budget] (one boxed double = 16 B), the control above the budget, and describe() as a note. Without the V8
 * features (ALLOC.ok false) only the note, saying why the gate is skipped.
 */
export function gate(check, note, name, m, { budget = 1, unit = 'call' } = {}) {
  if (!ALLOC.ok) { note(`alloc: ${name}: ${ALLOC.reason}`); return; }
  check(`alloc: ${name}: B / ${unit} (TurboFan)`, m.per, 0, budget, 'B');
  check(`alloc: ${name}: control (+1 array / ${unit}) counted`, m.control, budget, Infinity, 'B');
  note(`alloc: ${name}: ${describe(m, unit)}`);
}

/** A reading that must be at least `lo` B per unit (a suite's own control), through check() / note() as gate() does. */
export function atLeast(check, note, name, m, lo) {
  if (!ALLOC.ok) { note(`alloc: ${name}: ${ALLOC.reason}`); return; }
  check(`alloc: ${name}`, m.per, lo, Infinity, 'B');
}
