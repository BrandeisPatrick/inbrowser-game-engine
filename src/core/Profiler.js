// CPU + GPU frame profiler. CPU: per-system timers. GPU: EXT_disjoint_timer_query_webgl2 (when available)
// around named scopes; results arrive a few frames later and are folded into rolling averages.
// `gpuApprox` is true when the timer runs on ANGLE over Metal (Chrome / Edge on macOS, detected through
// WEBGL_debug_renderer_info): there the query brackets command submission rather than GPU execution, so the
// numbers are indicative only (they are shown as "gpu~"). For a hard throughput number use
// benchThroughput(renderFn, frames): renders N frames back to back, waits for the GPU, returns ms / frame.
export class Profiler {
  constructor(renderer, { window: win = 60 } = {}) {
    this.renderer = renderer;
    this.gl = renderer ? renderer.getContext() : null;
    this.ext = this.gl ? this.gl.getExtension('EXT_disjoint_timer_query_webgl2') : null;
    this.gpuRenderer = '';
    if (this.gl) {
      try {
        // RENDERER is unmasked in Firefox / Safari; Chrome masks it ("WebKit WebGL") -> the debug extension
        let name = String(this.gl.getParameter(this.gl.RENDERER) || '');
        if (!name || /^WebKit WebGL$/i.test(name)) {
          const dbg = this.gl.getExtension('WEBGL_debug_renderer_info');
          if (dbg) name = String(this.gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) || name);
        }
        this.gpuRenderer = name;
      } catch { /* privacy-restricted contexts */ }
    }
    /** GPU timings are approximate (timer query on ANGLE / Metal). */
    this.gpuApprox = !!this.ext && /ANGLE/i.test(this.gpuRenderer) && /Metal/i.test(this.gpuRenderer);
    this._px = new Uint8Array(4);
    this.win = win;
    this.cpu = new Map();   // name -> { last, avg, max }
    this.gpu = new Map();
    this._open = new Map(); // cpu scope starts
    this._queries = [];     // pending gpu queries [{name, q}]
    this._gpuActive = null;
    // ms/avg: CPU time of the frame; interval/fps: wall time between frames (what the player sees: ticks, so a
    // targetFps cap shows here); refresh: display refresh rate in Hz, 0 until known (the Engine sets it from rAF times)
    this.frame = { ms: 0, avg: 0, interval: 0, fps: 0, refresh: 0, calls: 0, triangles: 0, memory: null, steps: 0 };
    this._frameStart = 0;
    this.enabled = true;
  }
  begin(name) { if (this.enabled) this._open.set(name, performance.now()); }
  end(name) {
    if (!this.enabled) return;
    const t0 = this._open.get(name); if (t0 === undefined) return;
    this._fold(this.cpu, name, performance.now() - t0);
  }
  /** Fold an externally measured CPU time (ms) into scope `name` (e.g. a system's per-frame total). */
  record(name, ms) { if (this.enabled) this._fold(this.cpu, name, ms); }
  /** Measure a sync function. */
  scope(name, fn) { this.begin(name); try { return fn(); } finally { this.end(name); } }
  gpuBegin(name) {
    if (!this.ext || this._gpuActive) return;
    const q = this.gl.createQuery();
    this.gl.beginQuery(this.ext.TIME_ELAPSED_EXT, q);
    this._gpuActive = { name, q };
  }
  gpuEnd() {
    if (!this.ext || !this._gpuActive) return;
    this.gl.endQuery(this.ext.TIME_ELAPSED_EXT);
    this._queries.push(this._gpuActive); this._gpuActive = null;
  }
  frameBegin() { this._frameStart = performance.now(); }
  /** @param {number} [dt] seconds since the previous frame (for fps); without it fps falls back to CPU time. */
  frameEnd(dt) {
    const ms = performance.now() - this._frameStart;
    const f = this.frame; f.ms = ms; f.avg += (ms - f.avg) / Math.min(this.win, 120);
    if (dt > 0) f.interval += (dt * 1000 - f.interval) / (f.interval ? Math.min(this.win, 120) : 1);
    const per = f.interval || f.avg; f.fps = per > 0 ? 1000 / per : 0;
    if (this.renderer) { f.calls = this.renderer.info.render.calls; f.triangles = this.renderer.info.render.triangles; f.memory = this.renderer.info.memory; }
    this._collectGpu();
  }
  _collectGpu() {
    if (!this.ext) return;
    const gl = this.gl;
    const disjoint = gl.getParameter(this.ext.GPU_DISJOINT_EXT);
    for (let i = this._queries.length - 1; i >= 0; i--) {
      const { name, q } = this._queries[i];
      if (!gl.getQueryParameter(q, gl.QUERY_RESULT_AVAILABLE)) continue;
      if (!disjoint) this._fold(this.gpu, name, gl.getQueryParameter(q, gl.QUERY_RESULT) / 1e6);
      gl.deleteQuery(q); this._queries.splice(i, 1);
    }
    if (this._queries.length > 32) { for (const { q } of this._queries.splice(0, 16)) gl.deleteQuery(q); }
  }
  /**
   * Throughput benchmark: calls renderFn(i) `frames` times back to back (after `warmup` untimed calls), then
   * blocks until the GPU has finished (1-pixel readback) and returns the mean wall time per frame (ms).
   * Includes CPU submission and GPU execution, excludes vsync / compositing: the number to compare optimisations
   * by. Stalls the page while it runs; call it from the console or a debug key, not every frame.
   */
  benchThroughput(renderFn, frames = 60, { warmup = 3 } = {}) {
    if (typeof renderFn !== 'function') throw new Error('Profiler.benchThroughput: renderFn required');
    frames = Math.max(1, frames | 0);
    for (let i = 0; i < warmup; i++) renderFn(-1 - i);
    this._sync();
    const t0 = performance.now();
    for (let i = 0; i < frames; i++) renderFn(i);
    this._sync();
    const ms = (performance.now() - t0) / frames;
    this.lastBench = { frames, ms, fps: 1000 / ms };
    return ms;
  }
  // wait for the GPU: a 1-pixel readback of the default framebuffer forces the queue to drain
  _sync() {
    const gl = this.gl, r = this.renderer;
    if (!gl) return;
    const prev = r && r.getRenderTarget ? r.getRenderTarget() : null;
    if (prev && r.setRenderTarget) r.setRenderTarget(null);
    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, this._px);
    if (prev && r.setRenderTarget) r.setRenderTarget(prev);
  }

  _fold(map, name, v) {
    let e = map.get(name);
    if (!e) { e = { last: 0, avg: v, max: 0, n: 0 }; map.set(name, e); }
    e.last = v; e.avg += (v - e.avg) / Math.min(this.win, e.n + 1); e.max = Math.max(e.max * 0.98, v); e.n++;
  }
  /** Plain-object snapshot for console.table / overlays. */
  snapshot() {
    const rows = [];
    for (const [k, v] of this.cpu) rows.push({ scope: `cpu:${k}`, avg: +v.avg.toFixed(2), max: +v.max.toFixed(2) });
    for (const [k, v] of this.gpu) rows.push({ scope: `gpu:${k}`, avg: +v.avg.toFixed(2), max: +v.max.toFixed(2), ...(this.gpuApprox ? { approx: true } : {}) });
    rows.push({ scope: 'frame(cpu)', avg: +this.frame.avg.toFixed(2), max: +this.frame.ms.toFixed(2), fps: +this.frame.fps.toFixed(0), hz: +this.frame.refresh.toFixed(0), calls: this.frame.calls, tris: this.frame.triangles });
    return rows;
  }
}

/** Minimal DOM overlay (toggle with F3). `style` overrides the placement (default: top centre, clear of HUD corners). */
export class ProfilerOverlay {
  constructor(profiler, container = document.body, style = {}) {
    this.p = profiler; this.visible = false; this._t = 0;
    this.el = document.createElement('pre');
    Object.assign(this.el.style, { position: 'fixed', left: '50%', top: '8px', transform: 'translateX(-50%)', margin: 0, padding: '6px 8px', font: '11px/1.35 ui-monospace, Menlo, monospace', color: '#cfe', background: 'rgba(0,0,0,.55)', borderRadius: '6px', zIndex: 50, display: 'none', pointerEvents: 'none', whiteSpace: 'pre', ...style });
    container.appendChild(this.el);
    this._onKey = (e) => { if (e.code === 'F3') { e.preventDefault(); this.toggle(); } };
    window.addEventListener('keydown', this._onKey);
  }
  dispose() { window.removeEventListener('keydown', this._onKey); this.el.remove(); }
  toggle(v = !this.visible) { this.visible = v; this.el.style.display = v ? 'block' : 'none'; }
  update(dt) {
    if (!this.visible) return;
    this._t += dt; if (this._t < 0.25) return; this._t = 0;
    const f = this.p.frame;
    let s = `${f.fps.toFixed(0)} fps${f.refresh ? ` / ${f.refresh.toFixed(0)} Hz` : ''}  cpu ${f.avg.toFixed(1)} ms  calls ${f.calls}  tris ${(f.triangles / 1e6).toFixed(2)}M  steps ${f.steps}\n`;
    for (const [k, v] of this.p.cpu) s += `cpu ${k.padEnd(12)} ${v.avg.toFixed(2).padStart(6)} ms\n`;
    const g = this.p.gpuApprox ? 'gpu~' : 'gpu ';
    for (const [k, v] of this.p.gpu) s += `${g}${k.padEnd(12)} ${v.avg.toFixed(2).padStart(6)} ms\n`;
    if (f.memory) s += `geom ${f.memory.geometries}  tex ${f.memory.textures}`;
    this.el.textContent = s;
  }
}
