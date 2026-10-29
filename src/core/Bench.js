// Bench harness: the in-page half of the browser smoke test and benchmark (test/browser/smoke.mjs, bench.mjs). A
// sample creates one once it has loaded, registers its camera poses and settings knobs, and calls setReady(); the
// harness is also window.__sample, so Playwright (or the console) can drive it: fixed poses, a GPU-synchronised
// throughput bench (Profiler.benchThroughput), framebuffer captures with a content hash, pixel diffs, and run(plan),
// a poses x settings matrix that records ms per frame and the pixel difference against the first setting.
//
//   const harness = createBenchHarness(engine, {
//     name: 'canyon-run',
//     poses: { chase: (camera, h) => { camera.position.set(0, 4, 14); camera.lookAt(0, 1, 0); } },
//     after: (camera, h) => culler.update(camera),                      // camera-driven systems, after each pose
//     quiet: (h) => { dust.clear(); postfx.autoExposure = false; },     // drift sources (particles, eye adaptation)
//     settings: { 'postfx.aa': (v) => postfx.setMSAA(v === 'fxaa' ? 0 : 'auto'), 'shadows.kernel': { get, set } },
//     settle: (h) => world.whenIdle(),                                  // optional: async work a pose starts
//   });
//   const bp = readBenchParams();                 // ?bench=1&size=WxH
//   if (bp.size) harness.resize(bp.size.w, bp.size.h);
//   harness.setReady();
//
// API: name, engine, ready, poses (ids), settings (ids), frozen, setReady(), pose(id) -> id, settle() -> Promise,
// render(), bench(frames = 24, { warmup = 3 }) -> ms per frame,
// capture(scale = 4, into) -> { w, h, scale, data, hash }, diff(a, b) -> { meanAbs, p99, max, pctOver2, pctOver8 },
// set(path, value), get(path), run(plan) -> Promise<{ name, renderer, size, frames, warmup, ms, rows }>,
// info() -> { renderer, size, drawCalls, triangles, memory }, resize(w, h), size, freeze(), resume().
// readBenchParams(search) parses the bench query.
//
// Frames. Every harness frame is engine.renderFn(scene, camera, dt) with dt = options.dt (default 0): time-driven
// render state (eye adaptation, film grain) then holds still, so a pose renders the same image on every call, while
// the GPU does the same work as in a live frame. pose(id) freezes the engine loop (engine.stop()) so rAF ticks cannot
// move the camera or the simulation between calls (Playwright evaluates each call in its own task); resume()
// restarts it. Order of pose(id): poses[id](camera, harness), camera.updateMatrixWorld(), after(camera, harness),
// quiet(harness), 2 warm frames. settle(): when the sample passed a `settle` hook (async work a pose starts, e.g.
// tile builds), awaits it, then runs after() and quiet() again and renders 2 warm frames; without one it resolves at
// once. The scripts (and run()) await it after every pose; it never waits on a timer.
// Exposure: with dt = 0 the adapted luminance holds whatever the last live frame left, which differs between page
// loads. For hashes that reproduce across loads `quiet` must set postfx.autoExposure = false and call
// postfx.setTimeOfDay(darkness) (which resets the adaptation to the current frame's luminance).
// Captures read the default framebuffer with gl.readPixels in the same task as their render (no
// preserveDrawingBuffer needed) into one reused full-size buffer, then box-downsample by `scale` (1..16): `data` holds
// the raw per-channel SUMS over each scale x scale box (Uint16Array, top-down RGBA rows, at most 16 * 16 * 255, so
// exact: divide by scale^2 for the 0..255 mean) and the hash is 32-bit FNV-1a over those values. diff() compares the
// RGB of two captures of the same size and scale at that sub-byte precision: per pixel d = (|dr| + |dg| + |db|) / 3 in
// 0..255 (fractional); meanAbs, p99, max of d, pctOver2 / pctOver8 = % of pixels with d > 2 / d > 8.
// Settings: `settings[path]` is a setter function (get(path) then returns the last value set through the harness)
// or { set(v), get() }. In run() every setting starts from the state run() found: a knob that an earlier setting
// changed and this one does not name is set back first, and at the end every changed knob is restored (both only
// when its previous value is known, i.e. the knob has get() or was set through the harness before).
// Size: resize(w, h) pins the drawing buffer to exactly w x h device px: the container becomes w x h CSS px, adaptive
// resolution is switched off and engine.setPixelRatio(1) + engine.resize() run, so engine.onResize (PostFX) follows.
// Call it after PostFX is attached to engine.onResize. It leaves engine.adaptive off and the pixel ratio at 1 on
// purpose (also after resume()): a measured page must not change its resolution.
// This is tooling: bench() allocates nothing per frame, capture() reuses its readback buffer (and `into`'s array),
// run() allocates rows, one capture per pose and a diff histogram per row.

const nowMs = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
const r2 = (v) => Math.round(v * 100) / 100;
const r3 = (v) => Math.round(v * 1000) / 1000;
const r4 = (v) => Math.round(v * 10000) / 10000;

/**
 * Parse the bench query of a sample URL: `?bench=1&size=WxH`.
 * @param {string} [search]  location.search (default: the current page's, '' without a window)
 * @returns {{ enabled: boolean, size: { w: number, h: number } | null }}
 */
export function readBenchParams(search) {
  const s = search ?? (typeof location !== 'undefined' ? location.search : '');
  const p = new URLSearchParams(s);
  const enabled = p.has('bench') && p.get('bench') !== '0';
  const m = /^(\d+)\s*[x*,]\s*(\d+)$/i.exec(p.get('size') || '');
  const size = m && +m[1] > 0 && +m[2] > 0 ? { w: +m[1], h: +m[2] } : null;
  return { enabled, size };
}

/**
 * @typedef {object} BenchCapture
 * @property {number} w  width in px (drawing-buffer width / scale, rounded down)
 * @property {number} h  height in px
 * @property {number} scale  box size (1..16); data / scale^2 is the 0..255 mean
 * @property {Uint16Array} data  w * h * 4 per-channel sums over each scale x scale box, RGBA, top-down rows
 * @property {string} hash  32-bit FNV-1a of data (each value as 2 bytes, low first), 8 hex digits
 */

/**
 * @typedef {object} BenchDiff
 * @property {number} meanAbs  mean per-pixel difference (0..255, sub-byte precision)
 * @property {number} p99  99th percentile of the per-pixel difference
 * @property {number} max  largest per-pixel difference
 * @property {number} pctOver2  % of pixels differing by more than 2
 * @property {number} pctOver8  % of pixels differing by more than 8
 */

/**
 * @typedef {object} BenchPlan
 * @property {string[]} [poses]  pose ids (default: every registered pose)
 * @property {number} [frames=24]  timed frames per pose and setting
 * @property {number} [warmup=3]  untimed frames before each bench
 * @property {number} [scale=4]  capture downsampling
 * @property {{ label: string, set?: Object<string, *> }[]} [settings]  settings to compare, each applied to the state
 *   run() found (not cumulative); the first one is the reference for the diffs (default: one setting, 'current',
 *   that changes nothing)
 * @property {boolean} [restore=true]  set every changed knob back afterwards (when its previous value is known)
 * @property {Function} [onRow]  called with each row as it is measured (console use)
 */

/**
 * @typedef {object} BenchRow
 * @property {string} setting
 * @property {string} pose
 * @property {number} ms  mean ms per frame (Profiler.benchThroughput)
 * @property {number} fps  1000 / ms
 * @property {string} hash  capture hash of the pose under this setting
 * @property {number|null} meanAbs  diff against the first setting at the same pose (null on the first setting)
 * @property {number|null} p99
 * @property {number|null} max
 * @property {number|null} pctOver2
 * @property {number|null} pctOver8
 * @property {number} drawCalls
 * @property {number} triangles
 */

/**
 * Create the bench harness of a sample and expose it as window.__sample (see header).
 * @param {import('./Engine.js').Engine} engine  the sample's Engine (renderer, scene, camera, renderFn, profiler)
 * @param {object} [o]
 * @param {string} [o.name='sample']
 * @param {Object<string, Function>} [o.poses]  id -> (camera, harness) => void: places the camera
 * @param {Function} [o.quiet]  (harness) => void: removes drift sources (particles, eye adaptation) after a pose
 * @param {Function} [o.after]  (camera, harness) => void: updates camera-driven systems (culling, LOD) after a pose
 * @param {Object<string, Function|{ set: Function, get?: Function }>} [o.settings]  knobs by path, e.g. 'postfx.aa'
 * @param {Function} [o.settle]  (harness) => Promise: resolves once async work started by a pose has landed
 * @param {number} [o.dt=0]  seconds passed to renderFn for every harness frame (see header)
 * @param {boolean} [o.global=true]  assign window.__sample
 */
export function createBenchHarness(engine, o = {}) {
  if (!engine || !engine.renderer) throw new Error('createBenchHarness: engine required');
  const E = engine, R = E.renderer, P = E.profiler;
  const gl = R.getContext();
  const poseFns = o.poses || {};
  const knobs = o.settings || {};
  const quiet = typeof o.quiet === 'function' ? o.quiet : null;
  const after = typeof o.after === 'function' ? o.after : null;
  const settleHook = typeof o.settle === 'function' ? o.settle : null;
  const dt = Number.isFinite(o.dt) ? o.dt : 0;
  const lastSet = new Map();
  let readback = new Uint8Array(0); // full drawing buffer, reused while the size holds
  const renderFrame = () => { R.info.reset(); E.renderFn(E.scene, E.camera, dt); };

  const h = {
    name: o.name || 'sample',
    engine: E,
    ready: false,
    /** Registered pose ids. */
    poses: Object.keys(poseFns),
    /** Registered settings paths. */
    settings: Object.keys(knobs),
    /** True while pose() holds the engine loop (see header). */
    frozen: false,

    /** Mark the sample loaded: the smoke / bench scripts wait for this. */
    setReady() { h.ready = true; return h; },

    /** Stop the engine's rAF loop so nothing moves between harness calls. */
    freeze() { if (!h.frozen) { E.stop(); h.frozen = true; } return h; },
    /** Restart the engine's rAF loop after freeze() / pose(). */
    resume() { if (h.frozen) { h.frozen = false; E.start(); } return h; },

    /** Render one frame through engine.renderFn. */
    render() { renderFrame(); return h; },

    /**
     * Apply a registered pose: freeze, place the camera, after(), quiet(), 2 warm frames.
     * @param {string} id
     * @returns {string}
     */
    pose(id) {
      const fn = poseFns[id];
      if (typeof fn !== 'function') throw new Error(`${h.name}: unknown pose '${id}' (poses: ${h.poses.join(', ') || 'none'})`);
      h.freeze();
      fn(E.camera, h);
      E.camera.updateMatrixWorld();
      if (after) after(E.camera, h);
      if (quiet) quiet(h);
      renderFrame(); renderFrame();
      return id;
    },

    /**
     * Await the sample's settle hook (if any), then re-run after() and quiet() and render 2 warm frames (see header).
     * @returns {Promise<void>}
     */
    async settle() {
      if (!settleHook) return;
      await settleHook(h);
      if (after) after(E.camera, h);
      if (quiet) quiet(h);
      renderFrame(); renderFrame();
    },

    /**
     * GPU-synchronised throughput of the current view: Profiler.benchThroughput over engine.renderFn.
     * @param {number} [frames=24]
     * @param {{ warmup?: number }} [opts]
     * @returns {number} mean ms per frame
     */
    bench(frames = 24, { warmup = 3 } = {}) {
      return P.benchThroughput(renderFrame, frames, { warmup });
    },

    /**
     * Render one frame and read the default framebuffer, box-downsampled by `scale` (see header).
     * @param {number} [scale=4]  1..16
     * @param {BenchCapture|null} [into]  a previous capture of the same size whose data array is reused
     * @returns {BenchCapture}
     */
    capture(scale = 4, into = null) {
      renderFrame();
      const W = gl.drawingBufferWidth, H = gl.drawingBufferHeight;
      const S = Math.max(1, Math.min(scale | 0, 16, W, H)); // 16 * 16 * 255 fits a Uint16
      if (readback.length !== W * H * 4) readback = new Uint8Array(W * H * 4);
      R.setRenderTarget(null);
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null); // three's state cache does not track READ_FRAMEBUFFER binds
      gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, readback);
      const w = Math.max(1, Math.floor(W / S)), hh = Math.max(1, Math.floor(H / S));
      const n = w * hh * 4;
      const data = into && into.data instanceof Uint16Array && into.data.length === n ? into.data : new Uint16Array(n);
      const px = readback, row = W * 4;
      for (let y = 0; y < hh; y++) {
        const src0 = (H - 1 - y * S) * row; // readPixels rows are bottom-up; output is top-down
        let o2 = y * w * 4;
        for (let x = 0; x < w; x++, o2 += 4) {
          let r = 0, g = 0, b = 0, a = 0;
          for (let j = 0; j < S; j++) {
            let k = src0 - j * row + x * S * 4;
            for (let i = 0; i < S; i++, k += 4) { r += px[k]; g += px[k + 1]; b += px[k + 2]; a += px[k + 3]; }
          }
          data[o2] = r; data[o2 + 1] = g; data[o2 + 2] = b; data[o2 + 3] = a;
        }
      }
      let hash = 0x811c9dc5;
      for (let i = 0; i < n; i++) { const v = data[i]; hash = Math.imul(hash ^ (v & 255), 0x01000193); hash = Math.imul(hash ^ (v >>> 8), 0x01000193); }
      const out = into || {};
      out.w = w; out.h = hh; out.data = data; out.scale = S;
      out.hash = (hash >>> 0).toString(16).padStart(8, '0');
      return /** @type {BenchCapture} */ (out);
    },

    /**
     * Pixel difference of two captures of the same size and scale (RGB, sub-byte precision; see header).
     * @param {BenchCapture} a
     * @param {BenchCapture} b
     * @returns {BenchDiff}
     */
    diff(a, b) {
      if (!a || !b || a.w !== b.w || a.h !== b.h || a.scale !== b.scale) throw new Error(`${h.name}: diff needs captures of the same size and scale (${a && a.w}x${a && a.h}/${a && a.scale} vs ${b && b.w}x${b && b.h}/${b && b.scale})`);
      const A = a.data, B = b.data, n = a.w * a.h;
      const q = 3 * a.scale * a.scale; // channel-sum difference units per 1/255 of d
      const top = 255 * q, hist = new Uint32Array(top + 1); // per-pixel |dr| + |dg| + |db| of the box sums
      const t2 = 2 * q, t8 = 8 * q;
      let sum = 0, max = 0, over2 = 0, over8 = 0;
      for (let p = 0, k = 0; p < n; p++, k += 4) {
        const s = Math.abs(A[k] - B[k]) + Math.abs(A[k + 1] - B[k + 1]) + Math.abs(A[k + 2] - B[k + 2]);
        sum += s; hist[s]++;
        if (s > max) max = s;
        if (s > t2) { over2++; if (s > t8) over8++; }
      }
      let acc = 0, p99 = 0;
      const lim = 0.99 * n;
      for (let s = 0; s <= top; s++) { acc += hist[s]; if (acc >= lim) { p99 = s; break; } }
      return { meanAbs: r4(sum / q / n), p99: r3(p99 / q), max: r3(max / q), pctOver2: r3(100 * over2 / n), pctOver8: r3(100 * over8 / n) };
    },

    /**
     * Set a registered knob.
     * @param {string} path
     * @param {*} value
     */
    set(path, value) {
      const k = knobs[path];
      const fn = typeof k === 'function' ? k : k && typeof k.set === 'function' ? k.set : null;
      if (!fn) throw new Error(`${h.name}: unknown setting '${path}' (settings: ${h.settings.join(', ') || 'none'})`);
      fn(value);
      lastSet.set(path, value);
      return h;
    },

    /**
     * Current value of a knob: its get() when it has one, else the last value set through the harness.
     * @param {string} path
     * @returns {*}
     */
    get(path) {
      const k = knobs[path];
      if (!k) throw new Error(`${h.name}: unknown setting '${path}' (settings: ${h.settings.join(', ') || 'none'})`);
      return typeof k === 'object' && typeof k.get === 'function' ? k.get() : lastSet.get(path);
    },

    /**
     * Pin the drawing buffer to exactly w x h device px at pixel ratio 1 (see header).
     * @param {number} width  CSS px
     * @param {number} height  CSS px
     */
    resize(width, height) {
      const w = Math.max(1, Math.round(width)), hh = Math.max(1, Math.round(height));
      const st = E.container && E.container.style;
      if (st) { st.right = 'auto'; st.bottom = 'auto'; st.width = `${w}px`; st.height = `${hh}px`; }
      E.adaptive = false;
      E.setPixelRatio(1);
      E.resize(); // camera aspect
      return h.size;
    },

    /** Drawing-buffer size and pixel ratio. */
    get size() { return { w: gl.drawingBufferWidth, h: gl.drawingBufferHeight, pixelRatio: R.getPixelRatio() }; },

    /**
     * Render one frame and report the renderer, size and that frame's statistics (plain object).
     * @returns {{ renderer: string, size: { w: number, h: number, pixelRatio: number }, drawCalls: number,
     *   triangles: number, memory: { geometries: number, textures: number, programs: number, jsHeapMB: number|null } }}
     */
    info() {
      renderFrame();
      const ri = R.info, pm = typeof performance !== 'undefined' ? /** @type {any} */ (performance).memory : null;
      return {
        renderer: P.gpuRenderer,
        size: h.size,
        drawCalls: ri.render.calls,
        triangles: ri.render.triangles,
        memory: { geometries: ri.memory.geometries, textures: ri.memory.textures, programs: ri.programs ? ri.programs.length : 0, jsHeapMB: pm ? r2(pm.usedJSHeapSize / 1048576) : null },
      };
    },

    /**
     * Measure every setting on every pose (see BenchPlan / BenchRow). Yields to the event loop between rows.
     * @param {BenchPlan} [plan]
     * @returns {Promise<{ name: string, renderer: string, size: { w: number, h: number, pixelRatio: number },
     *   frames: number, warmup: number, ms: number, rows: BenchRow[] }>}
     */
    async run(plan = {}) {
      const ids = plan.poses && plan.poses.length ? plan.poses : h.poses;
      for (const id of ids) if (typeof poseFns[id] !== 'function') throw new Error(`${h.name}: unknown pose '${id}'`);
      const frames = Math.max(1, plan.frames | 0 || 24), warmup = plan.warmup ?? 3, scale = plan.scale || 4;
      const sets = plan.settings && plan.settings.length ? plan.settings : [{ label: 'current', set: {} }];
      const prev = new Map();
      for (const s of sets) for (const path in s.set || {}) if (!prev.has(path)) prev.set(path, h.get(path));
      const t0 = nowMs(), rows = [], base = new Map();
      let scratch = null;
      try {
        for (let si = 0; si < sets.length; si++) {
          const s = sets[si], label = s.label || `setting${si}`, want = s.set || {};
          for (const [path, v] of prev) if (!(path in want) && v !== undefined) h.set(path, v); // not cumulative
          for (const path in want) h.set(path, want[path]);
          for (const id of ids) {
            h.pose(id);
            await h.settle();
            let cap, d = null;
            if (si === 0) { cap = h.capture(scale); base.set(id, cap); } else { cap = scratch = h.capture(scale, scratch); d = h.diff(base.get(id), cap); }
            const hash = cap.hash, ri = R.info.render, drawCalls = ri.calls, triangles = ri.triangles;
            const ms = h.bench(frames, { warmup });
            const row = {
              setting: label, pose: id, ms: r3(ms), fps: r2(1000 / ms), hash,
              meanAbs: d ? d.meanAbs : null, p99: d ? d.p99 : null, max: d ? d.max : null,
              pctOver2: d ? d.pctOver2 : null, pctOver8: d ? d.pctOver8 : null, drawCalls, triangles,
            };
            rows.push(row);
            if (typeof plan.onRow === 'function') plan.onRow(row);
            await new Promise((res) => setTimeout(res, 0));
          }
        }
      } finally {
        if (plan.restore !== false) for (const [path, v] of prev) if (v !== undefined) h.set(path, v);
      }
      return { name: h.name, renderer: P.gpuRenderer, size: h.size, frames, warmup, ms: Math.round(nowMs() - t0), rows };
    },
  };

  if (o.global !== false && typeof window !== 'undefined') /** @type {any} */ (window).__sample = h;
  return h;
}
