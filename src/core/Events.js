// Events: a small typed emitter for game events (score, damage, checkpoint, 'entity:destroyed', ...). Pure JS,
// Node-safe, no dependencies. Types are any Map key (normally a string).
//
//   const events = new Events();
//   events.on('hit', onHit, ctx);      // fn.call(ctx, a, b, c, d); returns fn (keep it to remove the listener)
//   events.once('ready', onReady);     // removed after its first call
//   events.emit('hit', target, damage, point);   // positional, up to 4 args; returns true when a listener ran
//   events.off('hit', onHit, ctx);     // registrations of onHit with that ctx (no ctx: every ctx); off('hit') = all
//   events.has('hit'); events.clear();
//
// Semantics. Listeners run in registration order. An emit dispatches to the listeners that were registered when
// that emit started: a listener added during an emit (any listener, of any type) first runs on the next emit of
// its type; a listener removed during an emit of its type stays registered until that emit returns, so it still
// receives the rest of the current dispatch (snapshot semantics, as in Node's EventEmitter or three.js'
// EventDispatcher) but not nested emits that start after the removal. A once listener is disarmed right before
// its call, so a nested emit of the same type from inside it never calls it twice. Registering the same fn twice
// calls it twice. A listener that throws propagates out of emit; the rest of that dispatch is skipped and the
// bookkeeping stays consistent (try/finally).
//
// Storage / allocation scheme. One record per type holds four dense parallel arrays (fns, ctxs, once, dead) that grow
// only in on()/once(). emit() walks them by index up to the length read when it started (that is the snapshot:
// appended listeners lie beyond it), so it never copies an array or creates a closure: no allocation per emit.
// dead[i] is 0 for a live listener, -1 for a once listener that already fired, otherwise the emit serial number
// current at its removal; emit #s skips slot i when dead[i] !== 0 && s > dead[i]. Removals outside any emit of the
// type compact the arrays at once; removals during an emit mark the slot and set `dirty`, and the outermost emit
// of the type compacts in place (stable, so order is kept) when it returns. Records are kept after their last
// listener goes, so a type that is subscribed and unsubscribed every frame does not churn.

export class Events {
  constructor() {
    /** @type {Map<any, {fns: Function[], ctxs: any[], once: number[], dead: number[], live: number, depth: number, dirty: boolean}>} */
    this._types = new Map();
    this._serial = 0; // emit counter (for removals during an emit)
  }

  /**
   * Subscribe fn to `type`; called as fn.call(ctx, a, b, c, d). Returns fn.
   * @param {*} type
   * @param {Function} fn
   * @param {*} [ctx]
   */
  on(type, fn, ctx) { return this._add(type, fn, ctx, 0); }

  /** Like on(), but the listener is removed after its first call. Returns fn (off(type, fn) works before it fires). */
  once(type, fn, ctx) { return this._add(type, fn, ctx, 1); }

  /**
   * Remove the registrations of fn for `type` whose ctx is `ctx` (without ctx: every registration of fn, any ctx);
   * without fn, every listener of `type`. Pass the ctx when instances share a prototype method
   * (`on('hit', this.onHit, this)` -> `off('hit', this.onHit, this)`), or one instance unsubscribes them all.
   * During an emit of `type` the removal takes effect when that emit returns (see header).
   */
  off(type, fn, ctx) {
    const r = this._types.get(type);
    if (r === undefined || r.live === 0) return;
    const fns = r.fns, ctxs = r.ctxs, dead = r.dead;
    for (let i = 0; i < fns.length; i++) {
      if (dead[i] !== 0 || (fn !== undefined && (fns[i] !== fn || (ctx !== undefined && ctxs[i] !== ctx)))) continue;
      dead[i] = r.depth > 0 ? this._serial : -1; r.live--; r.dirty = true;
    }
    if (r.depth === 0 && r.dirty) compact(r);
  }

  /** True when `type` has at least one listener (not counting ones removed during the current emit). */
  has(type) { const r = this._types.get(type); return r !== undefined && r.live > 0; }

  /**
   * Call the listeners of `type` with up to four positional args. Allocation-free. Returns true when at least one
   * listener ran.
   */
  emit(type, a, b, c, d) {
    const r = this._types.get(type);
    if (r === undefined || r.live === 0) return false;
    const fns = r.fns, ctxs = r.ctxs, once = r.once, dead = r.dead, n = fns.length, s = ++this._serial;
    let ran = false;
    r.depth++;
    try {
      for (let i = 0; i < n; i++) {
        const k = dead[i];
        if (k !== 0 && s > k) continue; // fired once listener (-1), or removed before this emit started
        if (once[i] === 1) { if (k === 0) r.live--; dead[i] = -1; r.dirty = true; } // disarm before the call
        ran = true;
        fns[i].call(ctxs[i], a, b, c, d);
      }
    } finally {
      if (--r.depth === 0 && r.dirty) compact(r);
    }
    return ran;
  }

  /** Remove every listener of every type (deferred for types that are emitting, like off). */
  clear() { for (const type of this._types.keys()) this.off(type); }

  _add(type, fn, ctx, once) {
    if (typeof fn !== 'function') throw new TypeError('Events: listener must be a function');
    let r = this._types.get(type);
    if (r === undefined) { r = { fns: [], ctxs: [], once: [], dead: [], live: 0, depth: 0, dirty: false }; this._types.set(type, r); }
    r.fns.push(fn); r.ctxs.push(ctx); r.once.push(once); r.dead.push(0); r.live++;
    return fn;
  }
}

// stable in-place compaction of a record's parallel arrays (drops dead slots)
function compact(r) {
  const fns = r.fns, ctxs = r.ctxs, once = r.once, dead = r.dead;
  let k = 0;
  for (let i = 0; i < fns.length; i++) {
    if (dead[i] !== 0) continue;
    if (k !== i) { fns[k] = fns[i]; ctxs[k] = ctxs[i]; once[k] = once[i]; dead[k] = 0; }
    k++;
  }
  fns.length = ctxs.length = once.length = dead.length = k;
  r.dirty = false;
}
