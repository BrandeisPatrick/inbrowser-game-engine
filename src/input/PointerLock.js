// Pointer lock helper: request / exit / track the lock, with raw (un-accelerated) mouse movement where supported.
//
//   const lock = new PointerLock({ element: canvas, onChange: (locked) => pauseMenu.toggle(!locked && lock.releasedByUser) });
//   canvas.addEventListener('click', () => lock.request());   // must run inside a user gesture (click / key)
//   lock.locked; lock.supported; lock.error; lock.exit(); lock.dispose();
//   PointerLock.isLocked(canvas);   // static: is this element locked (a document / window / null: is anything locked)
//
// request() asks for { unadjustedMovement: true } first and retries without the option when the platform rejects it
// (NotSupportedError); it resolves true / false and never rejects. Failures (a rejected request, or `pointerlockerror`
// on browsers whose requestPointerLock returns nothing) land in `error` and onError(err) — e.g. a re-request within
// ~1 s of the user pressing Esc, or a request outside a user gesture. Nothing is requested automatically: the game
// decides when (click to play, a resume button...).
// `locked` follows `pointerlockchange` for this element; onChange(locked) fires on its transitions only.
// `releasedByUser` is true when the last unlock did not come from exit() (Esc, tab switch, focus loss): browsers
// typically swallow the Esc keydown that releases a lock, so a game opens its pause menu from onChange(false) instead.
// Safe to import and construct in Node (supported = false, no listeners).

export class PointerLock {
  /**
   * @param {object} [o]
   * @param {Element} [o.element=document.body]  element to lock (a Document resolves to its body)
   * @param {(locked: boolean) => void} [o.onChange]  called when this element gains / loses the lock
   * @param {(err: Error) => void} [o.onError]        called when a request fails
   * @param {boolean} [o.unadjustedMovement=true]     ask for raw mouse deltas (no OS acceleration); falls back when unsupported
   */
  constructor({ element = null, onChange = null, onError = null, unadjustedMovement = true } = {}) {
    const g = typeof document !== 'undefined' ? document : null;
    let el = element || (g && g.body) || null;
    if (el && el.nodeType === 9) el = el.body || el.documentElement;
    this.element = el;
    this.onChange = onChange;
    this.onError = onError;
    this.unadjustedMovement = unadjustedMovement;
    this.doc = (el && el.ownerDocument) || g;
    this.supported = !!(this.doc && el && typeof el.requestPointerLock === 'function' && 'pointerLockElement' in this.doc);
    this.locked = false;
    this.releasedByUser = false;
    this.error = null;
    this._exiting = false;
    this._legacy = null;   // resolver of a request made through the pre-promise API (settled by the events)

    this._onLockChange = () => {
      const was = this.locked;
      this.locked = PointerLock.isLocked(this.element);
      if (this.locked === was) return;   // another element of the page changed lock
      if (this.locked) this.releasedByUser = false;
      else { this.releasedByUser = !this._exiting; this._exiting = false; }
      this._settle(this.locked);
      if (this.onChange) this.onChange(this.locked);
    };
    this._onLockError = () => {
      if (!this._legacy) return;   // promise-based requests report through their promise
      this._fail(new Error('PointerLock: request failed (pointerlockerror)'));
      this._settle(false);
    };
    if (this.supported) {
      this.doc.addEventListener('pointerlockchange', this._onLockChange);
      this.doc.addEventListener('pointerlockerror', this._onLockError);
      this.locked = PointerLock.isLocked(el);
    }
  }

  /**
   * True when `element` holds the pointer lock; for a Document, window or null: when anything in it does.
   * @param {Element|Document|Window|null} [element]
   */
  static isLocked(element = null) {
    const g = typeof document !== 'undefined' ? document : null;
    const own = element && element.ownerDocument;
    if (own) {
      const root = element.getRootNode ? element.getRootNode() : own;   // shadow DOM: the shadow root reports the element
      return (root.pointerLockElement ?? own.pointerLockElement) === element;
    }
    const doc = element && element.nodeType === 9 ? element : g;
    return !!(doc && doc.pointerLockElement);
  }

  /** Request the lock (call from a user gesture). Resolves true when locked, false on failure. */
  request() {
    if (!this.supported) return Promise.resolve(false);
    if (this.locked) return Promise.resolve(true);
    this.error = null;
    if (!this.unadjustedMovement) return this._request();
    let p;
    try { p = this.element.requestPointerLock({ unadjustedMovement: true }); } catch { return this._request(); }
    if (!p || typeof p.then !== 'function') return this._wait();
    return p.then(() => true, (err) => (err && err.name === 'NotSupportedError' ? this._request() : this._fail(err)));
  }

  /** Release the lock if this element holds it. */
  exit() {
    if (!this.supported || !PointerLock.isLocked(this.element)) return;
    this._exiting = true;
    this.doc.exitPointerLock();
  }

  _request() {
    let p;
    try { p = this.element.requestPointerLock(); } catch (err) { return Promise.resolve(this._fail(err)); }
    if (!p || typeof p.then !== 'function') return this._wait();
    return p.then(() => true, (err) => this._fail(err));
  }
  _wait() {
    return new Promise((resolve) => {
      const done = (v) => { if (this._legacy === done) this._legacy = null; clearTimeout(t); resolve(v); };
      const t = setTimeout(() => done(this.locked), 3000);
      if (this._legacy) this._legacy(this.locked);
      this._legacy = done;
    });
  }
  _settle(v) { if (this._legacy) this._legacy(v); }
  _fail(err) {
    this.error = err || new Error('PointerLock: request failed');
    if (this.onError) this.onError(this.error);
    return false;
  }

  dispose() {
    this.exit();
    this._settle(false);
    if (this.supported) {
      this.doc.removeEventListener('pointerlockchange', this._onLockChange);
      this.doc.removeEventListener('pointerlockerror', this._onLockError);
    }
    this.onChange = this.onError = null;
  }
}
