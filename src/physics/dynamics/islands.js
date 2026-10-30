// Simulation islands and sleeping for the dynamics world (pure JS, deterministic, Node-safe, allocation-free).
// An island is a set of awake dynamic bodies connected by contacts or joints; static, kinematic and WORLD_BODY
// partners never join islands (they are shared supports, not links). Islands are rebuilt every step with an
// Int32Array union-find (path compression, iterative, union by smallest id so each root is the smallest member id
// and the island numbering is ascending in it: deterministic and independent of contact order).
//
//   const islands = new Islands(4096);
//   const n = islands.build(bodies, contacts, joints);      // islandOf[id] = 0..n-1 (awake dynamic), -1 otherwise
//   islands.updateSleep(bodies, contacts, joints, dt, { linear: 0.05, angular: 0.05, time: 0.5 });
//   islands.wake(bodies, id);                               // wakes the whole sleeping island of `id`
//
// Sleeping. Each awake dynamic body accumulates `sleepTime` while |v| < linear and |w| < angular (reset to 0
// otherwise). An island falls asleep when every member's sleepTime >= time and no member touches (contact) or is jointed
// to a moving kinematic body; its members get FLAG.ASLEEP, zero velocities, and are linked into a circular list (`next`) so that waking
// any one of them wakes them all in O(island size) without a rebuild (sleeping bodies produce no contacts, so the
// union-find could not find their island again). Invariant: an awake body is a list of one (next[id] === id).

import { BODY, FLAG } from './state.js';

const DYNAMIC = BODY.DYNAMIC, KINEMATIC = BODY.KINEMATIC, ASLEEP = FLAG.ASLEEP;

export class Islands {
  /** @param {number} [capacity=4096] body capacity (same as the BodyStore). */
  constructor(capacity = 4096) {
    this.capacity = capacity;
    this.count = 0;                               // islands found by the last build
    this.parent = new Int32Array(capacity);       // union-find forest; -1 = not part of any island this step
    this.islandOf = new Int32Array(capacity).fill(-1); // island index per body (-1: static/kinematic/asleep/dead)
    this.next = new Int32Array(capacity);         // circular list of each sleeping island (next[id] === id when awake)
    for (let i = 0; i < capacity; i++) this.next[i] = i;
    this._minSleep = new Float64Array(capacity);  // per island: smallest member sleepTime
    this._noSleep = new Uint8Array(capacity);     // per island: touches a moving kinematic body
    this._head = new Int32Array(capacity);        // per island: first / last member while linking a sleeping island
    this._tail = new Int32Array(capacity);
  }

  /** Reset the per-body state of a freshly allocated (or freed) id. */
  reset(id) { this.next[id] = id; this.islandOf[id] = -1; this.parent[id] = -1; }

  // root of x with path compression (iterative)
  _find(x) {
    const p = this.parent;
    let r = x;
    while (p[r] !== r) r = p[r];
    while (p[x] !== r) { const n = p[x]; p[x] = r; x = n; }
    return r;
  }

  // union by smallest id (the root of every set is its smallest member)
  _union(a, b) {
    const p = this.parent;
    if (p[a] < 0 || p[b] < 0) return;
    const ra = this._find(a), rb = this._find(b);
    if (ra === rb) return;
    if (ra < rb) p[rb] = ra; else p[ra] = rb;
  }

  /**
   * Group the awake dynamic bodies into islands over this step's contacts and the live joints.
   * @param {import('./state.js').BodyStore} B
   * @param {import('./state.js').ContactBuffer} contacts
   * @param {object|null} J  JointStore (count, alive, broken, a, b) or null
   * @returns {number} island count (also `this.count`); `islandOf[id]` and `B.island[id]` hold the index
   */
  build(B, contacts, J) {
    const n = B.count, p = this.parent, alive = B.alive, type = B.type, flags = B.flags;
    for (let i = 0; i < n; i++) p[i] = alive[i] === 1 && type[i] === DYNAMIC && (flags[i] & ASLEEP) === 0 ? i : -1;
    const ca = contacts.a, cb = contacts.b, cn = contacts.count;
    for (let i = 0; i < cn; i++) {
      const a = ca[i], b = cb[i];
      if (a >= 0 && b >= 0 && a !== b) this._union(a, b);
    }
    if (J) {
      const ja = J.a, jb = J.b, jalive = J.alive, jbroken = J.broken;
      for (let j = 0; j < J.count; j++) {
        if (!jalive[j] || (jbroken && jbroken[j])) continue;
        const a = ja[j], b = jb[j];
        if (a >= 0 && b >= 0 && a < n && b < n && a !== b) this._union(a, b);
      }
    }
    // number islands in ascending root order (the root is the smallest member, so it is met first)
    const io = this.islandOf, island = B.island;
    let k = 0;
    for (let i = 0; i < n; i++) {
      if (p[i] < 0) { io[i] = -1; continue; }
      const r = this._find(i);
      if (r === i) { io[i] = k; this._noSleep[k] = 0; k++; } else io[i] = io[r];
      island[i] = io[i];
    }
    // islands touching (contact) or jointed to a moving kinematic body may not sleep
    for (let i = 0; i < cn; i++) this._kinematicLink(B, ca[i], cb[i]);
    if (J) {
      const ja = J.a, jb = J.b, jalive = J.alive, jbroken = J.broken;
      for (let j = 0; j < J.count; j++) if (jalive[j] && !(jbroken && jbroken[j])) this._kinematicLink(B, ja[j], jb[j]);
    }
    this.count = k;
    return k;
  }

  // mark the island of the dynamic side of (a, b) as unable to sleep when the other side is a moving kinematic body
  _kinematicLink(B, a, b) {
    if (a < 0 || b < 0 || a >= B.count || b >= B.count) return;
    const type = B.type, io = this.islandOf;
    let dyn = -1, kin = -1;
    if (type[a] === KINEMATIC) { kin = a; dyn = b; } else if (type[b] === KINEMATIC) { kin = b; dyn = a; }
    if (kin < 0 || io[dyn] < 0) return;
    const o = kin * 3, vel = B.vel, w = B.angVel;
    if (vel[o] !== 0 || vel[o + 1] !== 0 || vel[o + 2] !== 0 || w[o] !== 0 || w[o + 1] !== 0 || w[o + 2] !== 0) this._noSleep[io[dyn]] = 1;
  }

  /**
   * Accumulate sleep timers and put whole islands to sleep (call after `build`).
   * @param {number} dt step length (s)
   * @param {{linear:number, angular:number, time:number}|false|null} params thresholds (m/s, rad/s, s); falsy: no sleeping
   * @returns {number} bodies put to sleep this call
   */
  updateSleep(B, contacts, J, dt, params) {
    if (!params) return 0;
    const n = B.count, io = this.islandOf, k = this.count, minT = this._minSleep;
    const lin2 = params.linear * params.linear, ang2 = params.angular * params.angular, tSleep = params.time;
    const vel = B.vel, w = B.angVel, st = B.sleepTime;
    for (let i = 0; i < k; i++) minT[i] = Infinity;
    for (let i = 0; i < n; i++) {
      const isl = io[i];
      if (isl < 0) continue;
      const o = i * 3;
      const v2 = vel[o] * vel[o] + vel[o + 1] * vel[o + 1] + vel[o + 2] * vel[o + 2];
      const w2 = w[o] * w[o] + w[o + 1] * w[o + 1] + w[o + 2] * w[o + 2];
      if (v2 < lin2 && w2 < ang2) st[i] += dt; else st[i] = 0;
      if (st[i] < minT[isl]) minT[isl] = st[i];
    }
    let any = false;
    const head = this._head, tail = this._tail;
    for (let i = 0; i < k; i++) {
      head[i] = -1; tail[i] = -1;
      if (minT[i] >= tSleep && this._noSleep[i] === 0) any = true; else minT[i] = -1; // -1 marks "stays awake"
    }
    if (!any) return 0;
    const next = this.next, flags = B.flags;
    let slept = 0;
    for (let i = 0; i < n; i++) {
      const isl = io[i];
      if (isl < 0 || minT[isl] < 0) continue;
      flags[i] |= ASLEEP;
      const o = i * 3;
      vel[o] = 0; vel[o + 1] = 0; vel[o + 2] = 0; w[o] = 0; w[o + 1] = 0; w[o + 2] = 0;
      if (head[isl] < 0) head[isl] = i; else next[tail[isl]] = i;
      tail[isl] = i;
      io[i] = -1;
      slept++;
    }
    for (let i = 0; i < k; i++) if (head[i] >= 0) next[tail[i]] = head[i];
    return slept;
  }

  /**
   * Wake the sleeping island that contains `id` (every member: ASLEEP cleared, sleepTime 0, list unlinked).
   * @returns {number} bodies woken (0 when `id` was awake)
   */
  wake(B, id) {
    if (id < 0 || id >= B.count || (B.flags[id] & ASLEEP) === 0) return 0;
    const next = this.next, flags = B.flags, st = B.sleepTime;
    let j = id, woken = 0;
    for (let guard = 0; guard < B.count; guard++) {
      flags[j] &= ~ASLEEP; st[j] = 0;
      const nx = next[j];
      next[j] = j;
      woken++;
      j = nx;
      if (j === id) break;
    }
    return woken;
  }

  /** Put one body to sleep on its own (velocities zeroed); touching awake bodies wake it again. */
  sleepBody(B, id) {
    if (id < 0 || id >= B.count || !B.alive[id] || B.type[id] !== DYNAMIC) return;
    if ((B.flags[id] & ASLEEP) !== 0) return;
    B.flags[id] |= ASLEEP;
    const o = id * 3;
    B.vel[o] = 0; B.vel[o + 1] = 0; B.vel[o + 2] = 0; B.angVel[o] = 0; B.angVel[o + 1] = 0; B.angVel[o + 2] = 0;
    this.next[id] = id; this.islandOf[id] = -1;
  }
}
