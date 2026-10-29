// Engine core: renderer, scene, camera, fixed-step simulation loop with per-frame render,
// systems registry, resize, adaptive resolution, profiler. No game knowledge.
//
// A system is any object with some of:
//   load?(onProgress(p 0..1, label))   awaited in registration order by engine.load() (weight: loadWeight, default 1)
//   earlyUpdate?(dt)                    once per frame BEFORE the fixed steps (input polling, edge-triggered actions)
//   fixedUpdate?(fixedDt)               0..maxSubsteps times per frame at fixedHz (simulation)
//   update?(dt, alpha)                  once per frame after the fixed steps (alpha = interpolation factor)
//   lateUpdate?(dt, alpha)              once per frame after every update (things that follow the final camera)
//   dispose?()
// Each phase runs systems in registration order, except that a system with a numeric `priority` field runs before
// systems with a higher value (default 0; equal priorities keep registration order; read once by addSystem).
// `engine.setSystemEnabled(systemOrName, false)` skips a system in every phase until it is enabled again (a
// dedicated flag, not the system's own `enabled` field: InputMap, Ambience and the rigs give that field a meaning
// of their own and keep running while it is false).
// `engine.clock` (core/Clock.js) scales the frame time: `timeScale` slows or speeds the simulation and the per-frame
// dt, `paused` stops both (the loop keeps rendering), `step(n)` runs n fixed steps while paused. The profiler, the
// overlay and adaptive resolution always see the unscaled wall dt. Per-system CPU time (all phases, summed per frame) is
// recorded in engine.profiler under the system's name, plus 'sim' (whole fixed loop) and 'render'.
//
// Frame pacing. requestAnimationFrame drives tick(). `targetFps` (option and live property; 0 = tick on every
// callback, the default) caps the tick rate: a callback ticks once the time since the last tick is at least
// 1000 / targetFps - 1 ms. The 1 ms absorbs timestamp jitter and timer clamping, so 60 on a 120 Hz screen ticks on
// every 2nd callback and never stretches to a 3rd; a cap that does not divide the refresh rate settles on the next
// rate below it that does (50 on 120 Hz runs at 40), so intervals stay even. Skipped callbacks only sample the
// refresh rate; dt is the real time since the previous tick (the fixed-step accumulator sees skipped time once,
// still bounded by maxSubsteps). `refreshRate` (Hz, 0 until known) is the display rate from the raw callback
// interval, ticked or not (estimator in _frame); until it is known (~5 callbacks after start(), 10 at most) the
// callbacks only measure, because a heavy first frame would show the frame rate instead of the screen's.
// Adaptive resolution (every 2 s, steps of 0.25): budget = 1000 / 60 ms while uncapped (the original 18.5 / 11.5 ms
// thresholds: the pixel ratio drops when the tick-interval EMA exceeds 1.11 x budget and rises under 0.69 x budget),
// so a 120 Hz screen keeps full resolution at 60 fps as before; with targetFps the budget is 1000 / min(targetFps,
// max(refreshRate, 60)) ms, e.g. targetFps 120 on a 120 Hz display = 9.25 / 5.75 ms (chases 120 fps at lower
// resolution), targetFps 60 anywhere = 18.5 / 11.5 ms. A reading under 60 Hz is not trusted for the budget (a scene
// missing every other vsync from its first frame looks exactly like a 30 Hz screen). Under vsync the interval never
// drops below the budget, so in practice the ratio only comes down (as it always did on 60 Hz screens).
import * as THREE from 'three';
import { Profiler, ProfilerOverlay } from './Profiler.js';
import { Clock } from './Clock.js';

const noop = () => {};
const now = () => performance.now();

export class Engine {
  /**
   * @param {object} o
   * @param {HTMLElement} o.container
   * @param {number} [o.fixedHz=120]
   * @param {number} [o.maxSubsteps=4]
   * @param {number} [o.targetFps=0]  cap on ticks per second, 0 = every rAF callback (live: engine.targetFps = 60;
   *   see header). At or below fixedHz / maxSubsteps (30 by default) a tick needs every substep and jitter makes the
   *   simulation drop time (slow motion): raise maxSubsteps for caps that low.
   * @param {boolean} [o.adaptiveResolution=true]  scale the pixel ratio to hold the frame budget (see header)
   * @param {number} [o.maxPixelRatio=2]
   * @param {object} [o.renderer]  extra WebGLRenderer options (e.g. { antialias: false } when post-processing)
   * @param {boolean} [o.shadows=true]
   * @param {number} [o.shadowType=THREE.PCFShadowMap]
   * @param {number} [o.exposure=0.9]  tone-mapping exposure (ACES filmic)
   * @param {number} [o.fov=60] [o.near=0.1] [o.far=6000]
   * @param {boolean} [o.overlay=true]  F3 profiler overlay
   * @param {HTMLElement} [o.overlayParent=document.body]
   * @param {number|boolean} [o.profile=false]  dump profiler.snapshot() with console.table after N s (true = 10 s)
   * @param {object} [o.physics]  optional physics world, exposed as engine.physics (e.g. a PhysicsWorld)
   * @param {Clock} [o.clock]  time control (default: a new Clock at timeScale 1, not paused)
   */
  constructor(o = {}) {
    this.container = o.container || document.body;
    this.fixedDt = 1 / (o.fixedHz || 120);
    this.maxSubsteps = o.maxSubsteps || 4;
    this.clock = o.clock || new Clock();
    this.systems = [];
    this.physics = o.physics || null;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance', ...(o.renderer || {}) });
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = o.exposure ?? 0.9;
    this.renderer.shadowMap.enabled = o.shadows !== false;
    this.renderer.shadowMap.type = o.shadowType ?? THREE.PCFShadowMap;
    this.renderer.info.autoReset = false; // reset once per frame in tick(): stats cover every pass of the frame
    this.container.appendChild(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(o.fov || 60, 1, o.near || 0.1, o.far || 6000);
    this.profiler = new Profiler(this.renderer);
    this.overlay = o.overlay === false ? null : new ProfilerOverlay(this.profiler, o.overlayParent || document.body);
    this.maxPR = Math.min(window.devicePixelRatio || 1, o.maxPixelRatio ?? 2);
    this.adaptive = o.adaptiveResolution !== false;
    this.pixelRatio = this.maxPR; this._frameEMA = 1 / 60; this._prTimer = 0;
    /** Cap on ticks per second, 0 = every rAF callback (default). Live: read on every callback. */
    this.targetFps = o.targetFps > 0 ? +o.targetFps : 0;
    /** Display refresh rate (Hz) estimated from rAF timestamps; 0 until known. */
    this.refreshRate = 0;
    this._rafT = 0; this._rafMs = 0; this._rafN = 0; this._rafLong = 0; this._calib = 0; // refresh estimator (_frame)
    /** Override to render through a composer: (scene, camera, dt) => void */
    this.renderFn = (scene, camera) => this.renderer.render(scene, camera);
    /** Called with (cssWidth, cssHeight, pixelRatio) on resize / pixel-ratio changes (PostFX.setSize hooks here). */
    this.onResize = noop;
    this.time = 0; this.frameCount = 0; this.dt = 0; this.alpha = 0; this.substeps = 0;
    this._running = false; this._acc = 0; this._last = 0; this._raf = 0;
    this._profileAt = o.profile ? (o.profile === true ? 10 : +o.profile) : 0;
    this._onWindowResize = () => this.resize();
    window.addEventListener('resize', this._onWindowResize);
    this.setPixelRatio(this.maxPR);
    this.resize();
  }

  /** Register a system (see header). `name` labels it in the profiler (default: constructor name). */
  addSystem(s, name) {
    s.__name = name || s.__name || (s.constructor && s.constructor !== Object ? s.constructor.name : '') || `system${this.systems.length}`;
    s.__ms = 0;
    s.__pr = Number.isFinite(s.priority) ? s.priority : 0;
    if (s.__on === undefined) s.__on = true;
    const S = this.systems;
    let i = S.length; // insert after the last system whose priority is <= ours (stable for equal priorities)
    while (i > 0 && S[i - 1].__pr > s.__pr) i--;
    S.splice(i, 0, s);
    return s;
  }
  /** Skip (false) or run (true) a system in every phase; `s` is the system or its name. */
  setSystemEnabled(s, on = true) { const sys = typeof s === 'string' ? this.getSystem(s) : s; if (sys) sys.__on = !!on; return sys; }
  isSystemEnabled(s) { const sys = typeof s === 'string' ? this.getSystem(s) : s; return !!sys && sys.__on !== false; }
  removeSystem(s) { const i = this.systems.indexOf(s); if (i >= 0) this.systems.splice(i, 1); }
  getSystem(name) { return this.systems.find((s) => s.__name === name) || null; }

  /** Await every system's load() in registration order; onProgress(0..1, label) across all of them. */
  async load(onProgress = noop) {
    const loaders = this.systems.filter((s) => typeof s.load === 'function');
    const weights = loaders.map((s) => (Number.isFinite(s.loadWeight) && s.loadWeight > 0 ? s.loadWeight : 1));
    const total = weights.reduce((a, b) => a + b, 0) || 1;
    let done = 0;
    for (let i = 0; i < loaders.length; i++) {
      const s = loaders[i], w = weights[i], base = done;
      await s.load((p, label) => onProgress((base + w * Math.min(1, Math.max(0, p || 0))) / total, label || s.__name));
      done += w;
    }
    onProgress(1, 'Ready');
  }

  get size() {
    return { width: this.container.clientWidth || window.innerWidth, height: this.container.clientHeight || window.innerHeight };
  }

  setPixelRatio(pr) {
    this.pixelRatio = pr;
    const { width, height } = this.size;
    this.renderer.setPixelRatio(pr);
    this.renderer.setSize(width, height);
    this.onResize(width, height, pr);
  }
  resize() {
    const { width, height } = this.size;
    this.camera.aspect = width / height; this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
    this.onResize(width, height, this.pixelRatio);
  }

  start() { if (this._running) return; this._running = true; this._last = now(); this._rafT = 0; this._calib = 10; this._raf = requestAnimationFrame(this._frame); }
  stop() { this._running = false; cancelAnimationFrame(this._raf); }

  _frame = (t) => {
    if (!this._running) return;
    cancelAnimationFrame(this._raf); // exactly one pending frame, even if _frame was invoked by hand
    this._raf = requestAnimationFrame(this._frame);
    // Refresh estimate: mean, then 20-sample EMA, of the raw callback interval, whether or not the callback ticks.
    // Only an interval within 0.7-1.5x the estimate refines it. A longer one is a dropped frame (busy main thread or
    // GPU), not the screen: following it would let the frame budget slide down with the frame rate and stall
    // adaptive resolution. A shorter one restarts the estimate (a faster screen, or a slow frame seeded it), and so
    // do 10 s of nothing but longer ones (the rate itself dropped: another screen, Low Power Mode, Energy Saver's
    // 30 fps; also a scene that never fits one refresh period, which then reads as its own rate). Published once
    // 4 intervals agree with it.
    const raw = this._rafT ? t - this._rafT : 0; this._rafT = t;
    if (raw > 1 && raw < 250) { // 0 / NaN: first callback, or a call by hand; >= 250 ms: hidden tab, breakpoint
      const m = this._rafMs;
      if (raw > m * 0.7 && raw < m * 1.5) {
        this._rafMs = m + (raw - m) / Math.min(20, ++this._rafN + 1); this._rafLong = 0;
        if (this._rafN >= 4) this.refreshRate = this.profiler.frame.refresh = 1000 / this._rafMs;
      } else if (!m || raw < m * 0.7 || (this._rafLong += raw) > 10000) { this._rafMs = raw; this._rafN = 0; this._rafLong = 0; }
    }
    if (!this.refreshRate && this._calib > 0) { this._calib--; this._last = t; return; } // measure first (header)
    // pacing (see header): a callback less than 1000 / targetFps - 1 ms after the last tick is skipped
    const fps = this.targetFps;
    if (fps > 0 && t - this._last < 1000 / fps - 1) return;
    let dt = (t - this._last) / 1000; this._last = t;
    if (dt > 0.25) dt = 0.25; // tab was hidden
    else if (!(dt >= 0)) dt = 0; // first rAF timestamp can precede start()
    this.tick(dt);
  };

  /**
   * Advance one frame by `dt` seconds: early -> fixed steps -> update -> late -> render. Driven by the rAF loop
   * after start(); call it directly for manual stepping (tests, capture, paused-tab debugging).
   */
  tick(dt) {
    const P = this.profiler; P.frameBegin();
    const raw = dt > 0.25 ? 0.25 : dt >= 0 ? dt : 0; // wall dt: profiler, overlay, adaptive resolution
    const clock = this.clock; dt = clock.advance(raw); // scaled dt for the systems (0 while paused)
    this.time += dt; this.frameCount++; this.dt = dt;
    const S = this.systems;
    for (let i = 0; i < S.length; i++) S[i].__ms = 0;
    // early (input)
    for (let i = 0; i < S.length; i++) { const s = S[i]; if (s.earlyUpdate && s.__on !== false) { const t0 = now(); s.earlyUpdate(dt); s.__ms += now() - t0; } }
    // fixed-step simulation (forced steps: clock.step(n) while paused; the surplus beyond maxSubsteps is re-queued)
    this._acc += dt; let steps = 0, forced = clock.paused ? clock.takeSteps() : 0;
    const tSim = now();
    while (steps < this.maxSubsteps && (forced > 0 || this._acc >= this.fixedDt)) {
      for (let i = 0; i < S.length; i++) { const s = S[i]; if (s.fixedUpdate && s.__on !== false) { const t0 = now(); s.fixedUpdate(this.fixedDt); s.__ms += now() - t0; } }
      if (forced > 0) forced--; else this._acc -= this.fixedDt;
      steps++;
    }
    if (forced > 0) clock.step(forced);
    if (steps === this.maxSubsteps && !clock.paused) this._acc = 0;
    P.record('sim', now() - tSim);
    this.substeps = steps; P.frame.steps = steps;
    const alpha = this.alpha = this._acc / this.fixedDt;
    // per-frame systems
    for (let i = 0; i < S.length; i++) { const s = S[i]; if (s.update && s.__on !== false) { const t0 = now(); s.update(dt, alpha); s.__ms += now() - t0; } }
    for (let i = 0; i < S.length; i++) { const s = S[i]; if (s.lateUpdate && s.__on !== false) { const t0 = now(); s.lateUpdate(dt, alpha); s.__ms += now() - t0; } }
    for (let i = 0; i < S.length; i++) { const s = S[i]; if (s.earlyUpdate || s.fixedUpdate || s.update || s.lateUpdate) P.record(s.__name, s.__ms); }
    // render
    P.begin('render'); P.gpuBegin('frame');
    this.renderer.info.reset();
    this.renderFn(this.scene, this.camera, dt);
    P.gpuEnd(); P.end('render');
    if (this.overlay) this.overlay.update(raw);
    P.frameEnd(raw);
    if (this._profileAt && clock.unscaledTime >= this._profileAt) { this._profileAt = 0; console.table(P.snapshot()); }
    if (this.adaptive && raw < 0.2) {
      this._frameEMA += (raw - this._frameEMA) * 0.05; this._prTimer += raw;
      if (this._prTimer > 2) {
        this._prTimer = 0;
        // uncapped: the 60 fps budget as before (a 120 Hz screen keeps full resolution at 60 fps); capped: the cap, but
        // never a budget tighter than the screen allows. targetFps: 120 is how a game asks for a ProMotion display's rate
        const fps = this.targetFps, budget = 1 / (fps > 0 ? Math.min(fps, Math.max(this.refreshRate, 60)) : 60); // s; header
        if (this._frameEMA > budget * 1.11 && this.pixelRatio > 1) this.setPixelRatio(Math.max(1, this.pixelRatio - 0.25));
        else if (this._frameEMA < budget * 0.69 && this.pixelRatio < this.maxPR) this.setPixelRatio(Math.min(this.maxPR, this.pixelRatio + 0.25));
      }
    }
  }

  dispose() {
    this.stop(); window.removeEventListener('resize', this._onWindowResize);
    for (const s of this.systems) if (s.dispose) s.dispose();
    if (this.overlay) this.overlay.dispose();
    this.renderer.dispose(); this.renderer.domElement.remove();
  }
}
