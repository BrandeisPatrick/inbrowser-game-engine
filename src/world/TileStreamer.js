// TileStreamer: generic scheduler for content built per tile around the camera (MapWorld streaming).
//
// Tiles live on per-layer grids, keyed by (i, j); each tile knows its footprint (minX..maxZ), an optional extra reach
// (e.g. tall buildings are seen from farther away) and how to build / free itself:
//   build(level, from) -> generator (yield between units of work: time slicing) | Promise | result. Builds compute
//                         only; nothing touches the scene until the result is committed. Result: { bytes, attach() }
//   dispose()           -> level 0: remove + free everything (GPU buffers, instance buffers)
//   downgrade(level)    -> optional: drop detail down to `level` (default: dispose)
// A layer's want(tile, view) gives the level a tile should have (0 none, 1 low, 2 full, ...); the default is distance
// based: level 1 within radius (+ altitude scaling, + reach), freed beyond radius + keep (hysteresis), with an optional
// detail level from layer.detail = { metric(tile, x, y, z) -> distance, radius, keep }. A tile may have a `parent`
// tile (e.g. detail that attaches to its parent's objects): it is only built while the parent is, and freed with it.
// Per frame (update): re-plan when the camera moved (priority = distance to the threshold, using the position the camera
// will reach `lookAhead` s from now, so the direction of travel comes first), free tiles that left their keep band,
// then resume / start builds (at most maxConcurrent in flight) until budgetMs is spent, and commit finished results
// (attach) within uploadBytesPerFrame (the GPU upload happens on the next render). Tiles within urgentRadius that
// have nothing yet (teleports) are built regardless of the budget. Pinned tiles never drop below level 1.
//
//   const s = new TileStreamer({ budgetMs: 3 });
//   const L = s.addLayer({ name: 'buildings', radius: 900, keep: 250 });
//   s.addTile(L, i, j, { minX, maxX, minZ, maxZ, reach, build(level, from) {...}, dispose() {...} });
//   await s.prewarm(view, { radius: 500, onProgress });  per frame: s.update(view, dt);  s.stats
const now = () => (globalThis.performance || Date).now();
const nextTick = () => new Promise((r) => setTimeout(r, 0));

export class TileStreamer {
  /**
   * @param {object} [o]
   * @param {number} [o.budgetMs=3]               build time per frame
   * @param {number} [o.maxConcurrent=2]          builds in flight (generators interleave by priority; promises run alongside)
   * @param {number} [o.uploadBytesPerFrame=6e6]  geometry committed per frame (at least one result per frame)
   * @param {number} [o.lookAhead=1.5]            s of camera travel used for priorities / wants
   * @param {number} [o.urgentRadius=0]           m: missing tiles this close are built at once (teleports)
   * @param {number} [o.replanDistance=8]         m of camera travel between re-plans (also every 0.25 s)
   * @param {number} [o.altitudeScale=4]          extra radius per metre of camera height above the ground
   */
  constructor(o = {}) {
    this.budgetMs = o.budgetMs ?? 3;
    this.maxConcurrent = Math.max(1, o.maxConcurrent ?? 2);
    this.uploadBytesPerFrame = o.uploadBytesPerFrame ?? 6e6;
    this.lookAhead = o.lookAhead ?? 1.5;
    this.urgentRadius = o.urgentRadius ?? 0;
    this.replanDistance = o.replanDistance ?? 8;
    this.altitudeScale = o.altitudeScale ?? 4;
    this.layers = [];
    this.tiles = [];
    this._active = [];     // tiles with a build in flight
    this._ready = [];      // finished results waiting for their commit
    this._queue = [];      // tiles needing work, sorted by priority
    this._last = { x: Infinity, y: 0, z: 0, t: 0 };
    this._vel = { x: 0, z: 0 };
    this._sincePlan = 1e9;
    this.enabled = true;
    this.onCommit = null;  // (tile, level, view, fromLevel) => void  (instrumentation: pop-in checks)
    this.stats = { tiles: 0, built: 0, pending: 0, building: 0, builds: 0, disposed: 0, upgrades: 0, cancelled: 0, msThisFrame: 0, maxMsFrame: 0, bytesThisFrame: 0, residentBytes: 0 };
  }

  /** layer: { name, cell?, radius, keep, detail?: { metric, radius, keep }, want?(tile, view) } */
  addLayer(o) {
    const L = { name: o.name || `layer${this.layers.length}`, cell: o.cell || 0, radius: o.radius ?? 900, keep: o.keep ?? 250, detail: o.detail || null, want: o.want || null, maxLevel: o.maxLevel ?? (o.detail ? 2 : 1), tiles: new Map() };
    this.layers.push(L);
    return L;
  }

  /** Register a tile (state: level 0). spec: { minX, maxX, minZ, maxZ, reach?, pinned?, build, dispose, downgrade?, ...data } */
  addTile(layer, i, j, spec) {
    const t = Object.assign(spec, { layer, i, j, key: `${layer.name}:${i},${j}`, level: 0, want: 0, task: null, prio: Infinity, bytes: 0, d: Infinity, children: [] });
    t.reach = t.reach || 0;
    if (t.parent) t.parent.children.push(t);
    layer.tiles.set(`${i},${j}`, t);
    this.tiles.push(t);
    this.stats.tiles = this.tiles.length;
    return t;
  }
  getTile(layer, i, j) { return layer.tiles.get(`${i},${j}`) || null; }

  // ---------------------------------------------------------------- policy
  /** Horizontal distance from (x, z) to the tile footprint (0 inside). */
  static rectDist(t, x, z) {
    const dx = x < t.minX ? t.minX - x : x > t.maxX ? x - t.maxX : 0;
    const dz = z < t.minZ ? t.minZ - z : z > t.maxZ ? z - t.maxZ : 0;
    return Math.sqrt(dx * dx + dz * dz);
  }
  /** Effective radius of a layer for a view (scaled with the camera's height above the ground). */
  radiusFor(layer, view) { return layer.radius + this.altitudeScale * Math.max(0, view.h || 0) + (view.extra || 0); }

  _want(t, view, px, pz) {
    const L = t.layer;
    if (t.parent && t.parent.level < 1) { t.prio = Infinity; return 0; }
    if (L.want) return L.want(t, view, this, px, pz);
    const R = this.radiusFor(L, view) + t.reach;
    const d = Math.min(TileStreamer.rectDist(t, view.x, view.z), TileStreamer.rectDist(t, px, pz));
    t.d = d;
    let want = d <= R ? 1 : (t.level >= 1 && d <= R + L.keep ? 1 : 0);
    if (t.pinned && want < 1) want = 1;
    if (want >= 1 && L.detail && L.maxLevel >= 2) {
      const D = L.detail, m = Math.min(D.metric(t, view.x, view.y, view.z), D.metric(t, px, view.y, pz));
      t.dd = m;
      if (m <= D.radius || (t.level >= 2 && m <= D.radius + (D.keep || 0))) want = 2;
    }
    // priority: how far inside (negative) / outside the threshold that matters for the next level
    t.prio = t.level < 1 ? d - R : L.detail ? (t.dd ?? Infinity) - L.detail.radius : d - R;
    return want;
  }

  _plan(view) {
    const la = this.lookAhead, px = view.x + this._vel.x * la, pz = view.z + this._vel.z * la;
    const Q = this._queue; Q.length = 0;
    for (const t of this.tiles) {
      t.want = this._want(t, view, px, pz);
      if (t.want !== t.level || (t.task && t.task.level !== t.want)) Q.push(t);
    }
    Q.sort((a, b) => a.prio - b.prio);
    this._sincePlan = 0;
    this._last.x = view.x; this._last.y = view.y; this._last.z = view.z;
  }

  // ---------------------------------------------------------------- per frame
  /**
   * @param {{x, y, z, h, extra?}} view  camera position, its height above the ground, extra radius (m; e.g. open views)
   * @param {number} dt
   */
  update(view, dt = 0) {
    const S = this.stats;
    if (!this.enabled || !this.tiles.length) { S.msThisFrame = 0; return; }
    const t0 = now();
    // camera velocity (direction of travel) from its movement; a jump (teleport) resets it
    const P = this._prev || (this._prev = { x: Infinity, z: 0 }), L = this._last;
    if (P.x !== Infinity && dt > 0) {
      const vx = (view.x - P.x) / dt, vz = (view.z - P.z) / dt;
      if (vx * vx + vz * vz < 400 * 400) { const k = Math.min(1, dt * 3); this._vel.x += (vx - this._vel.x) * k; this._vel.z += (vz - this._vel.z) * k; } else { this._vel.x = 0; this._vel.z = 0; }
    }
    P.x = view.x; P.z = view.z;
    this._sincePlan += dt;
    const moved2 = (view.x - L.x) ** 2 + (view.z - L.z) ** 2 + (view.y - L.y) ** 2;
    if (L.x === Infinity || moved2 > this.replanDistance ** 2 || this._sincePlan > 0.25) this._plan(view);
    this._lastFrame = view;
    // frees + downgrades (cheap) and cancellations
    for (const t of this._queue) {
      if (t.task && t.task.level > t.want) this._cancel(t);
      if (t.level > t.want && !t.task) this._lower(t, t.want);
    }
    // start / resume builds by priority until the budget is spent
    const budget = this.budgetMs;
    let bytes = 0;
    for (;;) {
      this._fill();
      const act = this._active;
      if (!act.length) break;
      // highest priority generator in flight
      let best = null;
      for (const t of act) if (t.task.iter && !t.task.done && (!best || t.prio < best.prio)) best = t;
      if (!best) break;
      const urgent = this.urgentRadius > 0 && best.level < 1 && !best.parent && best.d < this.urgentRadius;
      if (!urgent && now() - t0 >= budget) break;
      this._step(best);
      if (!urgent && now() - t0 >= budget) break;
    }
    // commit finished results (attach) within the upload budget, by priority
    if (this._ready.length) {
      this._ready.sort((a, b) => a.prio - b.prio);
      while (this._ready.length) {
        const t = this._ready[0], r = t.task.result, b = (r && r.bytes) || 0;
        if (bytes > 0 && bytes + b > this.uploadBytesPerFrame) break;
        this._ready.shift();
        this._commit(t);
        bytes += b;
      }
    }
    S.bytesThisFrame = bytes;
    let pending = 0, built = 0;
    for (const t of this._queue) if (t.want > t.level) pending++;
    for (const t of this.tiles) if (t.level > 0) built++;
    S.pending = pending; S.building = this._active.length; S.built = built;
    S.msThisFrame = now() - t0;
    if (S.msThisFrame > S.maxMsFrame) S.maxMsFrame = S.msThisFrame;
  }

  /** Start tasks for the most urgent queued tiles while fewer than maxConcurrent are in flight. */
  _fill() {
    const act = this._active;
    if (act.length >= this.maxConcurrent) return;
    for (const t of this._queue) {
      if (act.length >= this.maxConcurrent) break;
      if (t.task || t.want <= t.level) continue;
      this._start(t, t.want);
    }
  }

  _start(t, level) {
    const from = t.level;
    let r;
    try { r = t.build(level, from); } catch (e) { console.warn(`[TileStreamer] ${t.key} build failed`, e); t.failed = true; t.want = t.level; return; }
    const task = t.task = { level, from, iter: null, promise: null, result: null, done: false };
    if (r && typeof r.next === 'function') task.iter = r;
    else if (r && typeof r.then === 'function') {
      task.promise = r;
      r.then((res) => { if (t.task === task) { task.result = res; task.done = true; this._finish(t); } }, (e) => { if (t.task === task) { console.warn(`[TileStreamer] ${t.key} build failed`, e); this._drop(t); } });
    } else { task.result = r; task.done = true; }
    this._active.push(t);
    if (task.done) this._finish(t);
  }

  _step(t) {
    const task = t.task;
    let r;
    try { r = task.iter.next(); } catch (e) { console.warn(`[TileStreamer] ${t.key} build failed`, e); t.failed = true; this._drop(t); return; }
    if (r.done) { task.result = r.value; task.done = true; this._finish(t); }
  }

  _finish(t) {
    const i = this._active.indexOf(t); if (i >= 0) this._active.splice(i, 1);
    this._ready.push(t);
  }

  _drop(t) {
    const i = this._active.indexOf(t); if (i >= 0) this._active.splice(i, 1);
    const j = this._ready.indexOf(t); if (j >= 0) this._ready.splice(j, 1);
    if (t.task && t.task.result && t.task.result.discard) t.task.result.discard();
    t.task = null;
  }

  _cancel(t) {
    if (t.task && t.task.iter && t.task.iter.return) { try { t.task.iter.return(); } catch { /* ignore */ } }
    this._drop(t);
    this.stats.cancelled++;
  }

  _commit(t) {
    const task = t.task, r = task.result;
    t.task = null;
    try {
      if (r && r.attach) r.attach();
      t.level = task.level;
      t.bytes = (task.from > 0 ? t.bytes : 0) + ((r && r.bytes) || 0);
      this.stats.residentBytes += (r && r.bytes) || 0;
      if (task.from > 0) this.stats.upgrades++; else this.stats.builds++;
      if (this.onCommit) this.onCommit(t, task.level, this._lastFrame, task.from);
    } catch (e) { console.warn(`[TileStreamer] ${t.key} attach failed`, e); }
  }

  /** Drop a tile to `level` (0 = dispose); children go first. */
  _lower(t, level) {
    if (level <= 0) for (const c of t.children) { if (c.task) this._cancel(c); if (c.level > 0) this._lower(c, 0); }
    try {
      if (level <= 0 || !t.downgrade) { this.stats.residentBytes -= t.bytes; t.dispose(); t.level = 0; t.bytes = 0; this.stats.disposed++; }
      else { const freed = t.downgrade(level) || 0; this.stats.residentBytes -= freed; t.bytes -= freed; t.level = level; }
    } catch (e) { console.warn(`[TileStreamer] ${t.key} dispose failed`, e); t.level = level; }
  }

  // ---------------------------------------------------------------- load time
  /**
   * Build (and commit) every tile the policy wants within `radius` of the view, plus pinned tiles, before the game
   * starts (the loading screen waits only for these). Yields to the event loop every ~sliceMs.
   */
  async prewarm(view, { radius = 500, onProgress = null, sliceMs = 40 } = {}) {
    this._lastFrame = null;
    let k = 0, ts = now(), total = 0;
    for (let pass = 0; pass < 4; pass++) { // parents first, then the children they enable
      this._plan(view);
      const list = this.tiles.filter((t) => t.want > t.level && !t.task && (t.pinned || (t.parent && t.parent.pinned) || t.d <= radius));
      if (!list.length) break;
      total += list.length;
      for (const t of list) {
        this._start(t, t.want);
        while (t.task && !t.task.done) {
          if (t.task.iter) this._step(t);
          else await t.task.promise.catch(() => {});
          if (now() - ts > sliceMs) { await nextTick(); ts = now(); }
        }
        const ri = this._ready.indexOf(t);
        if (ri >= 0) { this._ready.splice(ri, 1); this._commit(t); }
        k++;
        if (onProgress) onProgress(Math.min(1, k / Math.max(total, 1)), t);
      }
    }
    this._last.x = Infinity; // re-plan on the first update
    return k;
  }

  /** Free every tile (world disposal). */
  disposeAll() {
    for (const t of this.tiles) { if (t.task) this._cancel(t); if (t.level > 0) this._lower(t, 0); }
  }
}
