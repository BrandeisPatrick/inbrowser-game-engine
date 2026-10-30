// Assets: a loader registry with URL dedupe, reference counts and weighted progress that plugs into Engine.load().
//
//   const assets = new Assets({ base: '/', concurrency: 6 });           // fetch: injectable (tests, auth headers)
//   assets.register('csv', (desc, assets) => assets.fetchOk(desc.url).then((r) => r.text()).then(parseCsv),
//                   { dispose(value) {} });
//   assets.add('level', { type: 'json', url: 'levels/1.json', weight: 2 }).add('font', { type: 'bin', url: 'f.bin' });
//   engine.addSystem(assets.system({ loadWeight: 3 }));                  // engine.load() awaits it with the others
//   await assets.load((p, key) => bar.set(p, key));                      // or standalone; resolves to `assets`
//   assets.get('level'); assets.peek('font'); assets.has('level'); assets.status('font'); await assets.loadOne('late');
//   assets.release('level'); assets.dispose();
//
// Types. Built in: json, text, bin (ArrayBuffer), blob, all through `assets.fetch` (`init` in the descriptor is passed
// to fetch). render/loaders.js adds gltf / texture / hdr / lut, audio/Samples.js adds audio. A loader is
// `(desc, assets, freeSlot) -> Promise<value>`: desc = the add() descriptor with `url` resolved against `base` (unless
// absolute: scheme or leading '/') and `key` = the first key that asked for it; `freeSlot()` hands the concurrency slot
// back early (call it after the download when the rest waits on something else, e.g. audio waiting for a gesture).
// `dispose(value)` frees a value. A loader must never call assets.load() (it would wait for every loading key, its own
// included: deadlock); nested loads use add() + loadOne(). A loader that waits on anything but the network (a user
// gesture, another system) holds up load() and with it Engine.load, where systems load in order, so every later
// system waits too: audio/Samples.js therefore decodes offline by default.
// Dedupe. Keys with the same type + resolved URL + other options share one load and one value; every key holds one
// reference. release(key) drops it; the last reference disposes the value and evicts it (a later add loads afresh).
// Options compare by value for primitives, plain objects and arrays (`init: { cache: 'no-store' }` literals match),
// by reference for functions and every other object (Headers, AbortSignal, class instances: an `init` holding one
// dedupes only with the same object). Keys without a url never share.
// Scheduling. load() queues every pending key in add order with at most `concurrency` loader calls in flight (shared
// by overlapping load() calls); loadOne(key) starts at once (on-demand loads, and loads nested inside a loader, never
// wait for a slot). Keys added while load() runs wait for the next load() / loadOne().
// Progress. load(onProgress(p, label)): weighted fraction of the batch's keys finished (weights as they were when
// load() was called), once per key as it finishes, label = that key; (1, '') at once when nothing is pending.
// Engine.load's per-system callback fits as is. A throwing onProgress is logged (console.error), never fatal.
// Errors. A failure rejects load() once the rest of the batch settled, with an Error naming the first failed key and
// URL (`.key`, `.url`, `.cause`, `.errors` = one keyed Error per failed key); keys released meanwhile do not count. A
// failed key stays failed (get throws, loadOne rejects) until add(key, ...) again retries it.
// No DOM / WebGL: safe in Node; nothing is fetched before load() / loadOne().

const noop = () => {};
const SCHEME = /^[a-z][a-z\d+.-]*:/i;

/** `url` against `base`: kept when it has a scheme (http:, data:, blob:) or starts with '/', else base + url. */
function resolveUrl(url, base) {
  if (SCHEME.test(url)) return url;
  if (SCHEME.test(base)) return new URL(url, base).href;
  if (url[0] === '/') return url;
  return (base ? base.replace(/\/?$/, '/') : '') + url.replace(/^\.\//, '');
}

// option fingerprint for dedupe (see header): primitives, plain objects and arrays by value; functions and other
// objects by identity; a cyclic plain object makes the descriptor unique
const _ids = new WeakMap();
let _uid = 0;
const idOf = (v) => { let i = _ids.get(v); if (!i) _ids.set(v, (i = `#${++_uid}`)); return i; };
const byRef = (v) => typeof v === 'function' || (v !== null && typeof v === 'object' && !Array.isArray(v) && Object.getPrototypeOf(v) !== Object.prototype && Object.getPrototypeOf(v) !== null);
function optsKey(o) {
  const ks = Object.keys(o).sort();
  if (!ks.length) return '';
  try { return JSON.stringify(ks.map((k) => [k, o[k]]), (k, v) => (byRef(v) ? idOf(v) : v)); } catch { return idOf(o); }
}

export class Assets {
  /**
   * @param {object} [o]
   * @param {string} [o.base='/']  prefix for relative URLs (a path or an absolute URL)
   * @param {number} [o.concurrency=6]  loader calls in flight during load()
   * @param {Function} [o.fetch=globalThis.fetch]  (url, init) -> Promise<Response>
   */
  constructor({ base = '/', concurrency = 6, fetch = globalThis.fetch } = {}) {
    this.base = base;
    this.concurrency = Math.max(1, Math.floor(concurrency) || 1);
    // called unbound: window.fetch throws "Illegal invocation" with `this` = assets
    this.fetch = fetch ? (url, init) => fetch(url, init) : () => Promise.reject(new Error('Assets: no fetch available'));
    this._types = new Map();
    this._entries = new Map(); // key -> { key, res, weight, state: 'pending' | 'loading' | 'loaded' | 'failed', promise, error }
    this._res = new Map();     // dedupe id -> { id, type, desc, refs, promise, value, loaded, gone, wait, job }
    this._queue = [];
    this._inFlight = 0;
    const via = (m) => (d, a) => a.fetchOk(d.url, d.init).then((r) => r[m]());
    this.register('json', via('json')).register('text', via('text')).register('bin', via('arrayBuffer')).register('blob', via('blob'));
  }

  /** fetch(url, init) that rejects with `HTTP <status>` unless the response is ok. */
  async fetchOk(url, init) {
    const r = await this.fetch(url, init);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r;
  }

  /**
   * Register (or replace) a type. `load(desc, assets, freeSlot) -> Promise<value>`; `dispose(value)` frees one.
   * @returns {this}
   */
  register(type, load, { dispose = null } = {}) {
    if (typeof load !== 'function') throw new TypeError(`Assets: loader for "${type}" is not a function`);
    this._types.set(type, { load, dispose });
    return this;
  }

  /**
   * Queue `key` (see header for dedupe). Re-adding a key with the same descriptor is a no-op, re-adding a failed key
   * retries it, re-adding a live key with another descriptor throws (release it first).
   * @param {string} key
   * @param {{type: string, url?: string, weight?: number}} desc  plus type-specific options
   * @returns {this}
   */
  add(key, desc = {}) {
    const { type, url, weight = 1, ...opts } = desc;
    if (!this._types.has(type)) throw new Error(`Assets: unknown type "${type}" for "${key}" (register it first)`);
    const abs = url == null ? null : resolveUrl(String(url), this.base);
    const id = `${type}\n${abs ?? `\0${key}`}\n${optsKey(opts)}`; // url-less: the key itself (re-add = no-op)
    const w = Number.isFinite(weight) && weight >= 0 ? weight : 1;
    const old = this._entries.get(key);
    if (old) {
      const dead = old.state === 'failed' || old.res.gone;
      if (!dead && old.res.id === id) { old.weight = w; return this; }
      if (!dead) throw new Error(`Assets: "${key}" is already added with another descriptor (release it first)`);
      this._entries.delete(key);
    }
    let r = this._res.get(id);
    if (!r) {
      r = { id, type, desc: { ...opts, key, type, url: abs, weight: w }, refs: 0, promise: null, value: undefined, loaded: false, gone: false, wait: null, job: null };
      this._res.set(id, r);
    }
    r.refs++;
    this._entries.set(key, { key, res: r, weight: w, state: r.loaded ? 'loaded' : 'pending', promise: r.loaded ? Promise.resolve(r.value) : null, error: null });
    return this;
  }

  /**
   * Load everything pending (and wait for keys already loading), at most `concurrency` loader calls in flight.
   * @param {(p: number, label: string) => void} [onProgress]
   * @returns {Promise<this>}
   */
  load(onProgress = noop) {
    const report = (p, label) => { try { onProgress(p, label); } catch (err) { console.error('Assets: onProgress threw', err); } };
    const batch = [], ws = []; // weights snapshot: a re-add during the batch cannot push progress to 1 early
    for (const e of this._entries.values()) if (e.state === 'pending' || e.state === 'loading') { batch.push(e); ws.push(e.weight); }
    if (!batch.length) { report(1, ''); return Promise.resolve(this); }
    let total = 0, done = 0, n = 0;
    for (const w of ws) total += w;
    const errors = [];
    return new Promise((resolve, reject) => {
      const fin = (i, err) => {
        const e = batch[i];
        done += ws[i]; n++;
        if (err && this._entries.get(e.key) === e) errors.push(err); // released (or re-added) meanwhile: not a failure
        report(total > 0 ? Math.min(1, done / total) : n / batch.length, e.key);
        if (n < batch.length) return;
        if (!errors.length) { resolve(this); return; }
        const f = errors[0], more = errors.length > 1 ? ` (+${errors.length - 1} more: ${errors.slice(1).map((x) => x.key).join(', ')})` : '';
        const x = new Error(f.message + more, { cause: f.cause });
        x.key = f.key; x.url = f.url; x.errors = errors;
        reject(x);
      };
      batch.forEach((e, i) => this._start(e, true).then(() => fin(i), (err) => fin(i, err)));
    });
  }

  /** Load one added key now (no slot wait) -> Promise<value>. Rejects for a key never added or failed. */
  loadOne(key) {
    const e = this._entries.get(key);
    if (!e) return Promise.reject(new Error(`Assets: "${key}" was never added`));
    if (e.state === 'loading') this._kick(e.res); // queued behind the concurrency cap: start it now
    return this._start(e, false);
  }

  /** Loaded value of `key`; throws when it is not loaded (yet), failed or unknown. */
  get(key) {
    const e = this._entries.get(key);
    if (e && e.state !== 'failed' && e.res.loaded) return e.res.value;
    throw new Error(`Assets: "${key}" is ${!e ? 'not added' : e.state === 'failed' ? `failed (${e.error.message})` : 'not loaded yet'}`);
  }
  /** True when get(key) would return a value. */
  has(key) { const e = this._entries.get(key); return !!e && e.state !== 'failed' && e.res.loaded; }
  /** Loaded value of `key`, or undefined. */
  peek(key) { const e = this._entries.get(key); return e && e.state !== 'failed' && e.res.loaded ? e.res.value : undefined; }
  /** 'pending' | 'loading' | 'loaded' | 'failed', or null for a key never added (or released). */
  status(key) {
    const e = this._entries.get(key);
    return !e ? null : e.state === 'failed' ? 'failed' : e.res.loaded ? 'loaded' : e.state;
  }

  /** Drop the reference `key` holds; the last reference disposes the shared value (also once an in-flight load lands). */
  release(key) {
    const e = this._entries.get(key);
    if (!e) return this;
    this._entries.delete(key);
    const r = e.res;
    if (e.state === 'failed' || r.gone || --r.refs > 0) return this;
    r.gone = true;
    if (this._res.get(r.id) === r) this._res.delete(r.id);
    if (r.job) { r.job.res(undefined); r.job = null; } // still queued: never loaded
    if (r.loaded) {
      const v = r.value, t = this._types.get(r.type);
      r.value = undefined; r.loaded = false;
      if (t && t.dispose) t.dispose(v);
    }
    return this;
  }

  /** Release every key. */
  dispose() { for (const k of [...this._entries.keys()]) this.release(k); }

  /**
   * Engine system: `engine.addSystem(assets.system())`, so engine.load() runs load() with its progress.
   * @returns {{ load: (onProgress?: (p: number, label: string) => void) => Promise<Assets>, loadWeight: number, __name: string }}
   */
  system({ loadWeight = 1, name = 'assets' } = {}) {
    return { load: (onProgress) => this.load(onProgress), loadWeight, __name: name };
  }

  // --- internals
  _start(e, queued) {
    if (e.promise) return e.promise;
    e.state = 'loading';
    const r = e.res;
    e.promise = (queued ? this._enqueue(r) : this._kick(r)).then((v) => {
      if (e.state === 'loading') e.state = 'loaded';
      return v;
    }, (err) => {
      const x = new Error(`Assets: failed to load "${e.key}" (${r.desc.url ?? r.type}): ${err && err.message ? err.message : err}`, { cause: err });
      x.key = e.key; x.url = r.desc.url;
      e.state = 'failed'; e.error = x;
      throw x;
    });
    return e.promise;
  }
  _enqueue(r) {
    if (r.promise) return r.promise;
    if (!r.wait) r.wait = new Promise((res, rej) => { r.job = { res, rej }; this._queue.push(r); });
    this._pump();
    return r.wait;
  }
  _pump() {
    while (this._queue.length && this._inFlight < this.concurrency) {
      const r = this._queue.shift();
      if (r.job) this._kick(r); // else started by loadOne, or released, meanwhile
    }
  }
  _kick(r) {
    if (r.promise) return r.promise;
    const t = this._types.get(r.type);
    let held = true;
    const freeSlot = () => { if (held) { held = false; this._inFlight--; this._pump(); } };
    this._inFlight++;
    r.promise = (async () => t.load(r.desc, this, freeSlot))().then((v) => {
      freeSlot();
      if (r.gone) { if (t.dispose) t.dispose(v); } // released while loading
      else { r.value = v; r.loaded = true; }
      return v;
    }, (err) => {
      freeSlot();
      r.gone = true; // failed: evicted, so add() of a key for it loads afresh
      if (this._res.get(r.id) === r) this._res.delete(r.id);
      throw err;
    });
    if (r.job) { r.promise.then(r.job.res, r.job.rej); r.job = null; }
    return r.promise;
  }
}
