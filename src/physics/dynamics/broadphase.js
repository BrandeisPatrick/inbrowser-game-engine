// Broadphase of the dynamics world (pure JS, deterministic, Node-safe, no THREE): a dynamic AABB tree of FAT boxes
// (the body's tight AABB grown by `margin` on every side) plus a persistent, sorted list of candidate pairs.
//
//   const bp = new Broadphase(4096, { margin: 0.1 });
//   bp.update(bodies);              // after the solver refreshed bodies.aabb; fills bp.pairs[0 .. bp.pairCount)
//   for (let i = 0; i < bp.pairCount; i++) { const k = bp.pairs[i], a = pairA(k), b = pairB(k); ... }   // a < b
//   const n = bp.query(minX, minY, minZ, maxX, maxY, maxZ, ids, layerMask);   // ids: Int32Array, ascending
//   const m = bp.raycast(ox, oy, oz, dx, dy, dz, maxT, ids);                   // fat-box candidates, ascending
//   bp.remove(id);                  // when a body is freed (update() also drops proxies of dead ids by itself)
//
// Tree: binary, stored in typed arrays indexed by node (box: Float64Array, min x y z then max x y z at node * 6, the
// same layout as BodyStore.aabb; child1 / child2 / parent / height / nodeBody: Int32Array; child1 === -1 marks a
// leaf; free nodes are chained through `parent` with height -1). Insertion picks the sibling by the surface-area
// heuristic and every insert / remove walks back to the root refitting boxes and applying AVL-style rotations
// (Box2D b2DynamicTree), so the height stays O(log n). A proxy is re-inserted only when its tight box leaves its
// fat box. Traversals use an explicit Int32Array stack.
//
// Pairs. The broadphase keeps every pair whose FAT boxes overlap and that passes the permanent filters (at least one
// body BODY.DYNAMIC, and (layer[a] & mask[b]) !== 0 && (layer[b] & mask[a]) !== 0) as a sorted candidate list. An
// update re-queries the tree only for the proxies that were (re)inserted or whose type / layer / mask changed (they
// are "dirty"): candidates touching a dirty id are dropped, each dirty proxy's fat box is queried against the tree
// (a pair of two dirty proxies is emitted once, by the smaller id, so no duplicates arise), the new keys are radix-
// sorted and merged into the kept ones. `pairs` is then the candidates whose TIGHT boxes overlap (inclusive, touching
// counts, as collision.js aabbOverlap) and that are not BOTH FLAG.ASLEEP. So a pair is skipped only when both bodies
// are non-dynamic, both asleep, or the layer test fails (a sleeping dynamic body on a static one without FLAG.ASLEEP
// IS emitted). Since tight boxes always lie inside the fat ones, `pairs` is exactly { pairKey(a, b) : both alive,
// tight boxes overlap, rule passes } in ascending key order: a pure function of the BodyStore contents, independent
// of insertion history, tree shape and the margin. Steady-state cost: an O(n) containment scan of the bodies, O(log n)
// per dirty proxy, and O(candidates) for the merge and the tight test; not a full tree pass per update.
//
// Determinism: the step path uses comparisons, + - * / and integer bit ops only; no Set / Map, no comparator
// closures (in-place LSD radix sort on Uint32Array, insertion sort below 48 keys). Bodies whose AABB is not finite or
// is inverted (min > max) get no proxy: never paired, never returned by query / raycast.
//
// Allocation: update() / query() / raycast() / remove() allocate nothing once warm. Internal typed arrays grow by
// doubling (copying) when a capacity is exceeded: node arrays when the tree needs more than 2 * capacity nodes, the
// per-body arrays when the BodyStore's capacity is larger than this broadphase's, the candidate / new-pair / sort
// buffers, the traversal stack and the query buffer. `pairs` grows by doubling too, so re-read `bp.pairs` after every
// update(); never keep the array. `stats.ms` is filled only with { timing: true }: performance.now() allocates its
// result (~70-100 B per update in Node 26), so timing is off by default.
//
// Filtering: query() and raycast() keep a body when (layer[id] & filterMask) !== 0 only; the pair rule (body type,
// sleep, the other body's mask) does not apply to them.
//
// Staleness: query() and raycast() walk the proxies as of the last update(); query() then tests the tight AABB read
// from the BodyStore at call time. A body moved since the last update() may be missed (outside its fat box) or stale.
// remove(id) takes effect in the tree at once and in `pairs` at the next update().

import { BODY, FLAG } from './state.js';

const NULL = -1;
const DYNAMIC = BODY.DYNAMIC;
const ASLEEP = FLAG.ASLEEP;
const INSERTION_SORT_MAX = 48;

// module-level scratch: radix digit counts (shared, the sort is not re-entrant)
const _counts = new Uint32Array(256);

/**
 * Sort a[0 .. n) ascending in place (uint32 keys): LSD radix sort, 8-bit digits, passes whose digit is the same for
 * every key are skipped; insertion sort below 48 keys. `tmp` is scratch of length >= n. No allocation.
 * @param {Uint32Array} a @param {number} n @param {Uint32Array} tmp @returns {Uint32Array} a
 */
export function radixSortU32(a, n, tmp) {
  if (n < INSERTION_SORT_MAX) {
    for (let i = 1; i < n; i++) {
      const k = a[i];
      let j = i - 1;
      while (j >= 0 && a[j] > k) { a[j + 1] = a[j]; j--; }
      a[j + 1] = k;
    }
    return a;
  }
  const c = _counts;
  let src = a, dst = tmp;
  for (let shift = 0; shift < 32; shift += 8) {
    c.fill(0);
    for (let i = 0; i < n; i++) c[(src[i] >>> shift) & 255]++;
    if (c[(src[0] >>> shift) & 255] === n) continue;   // constant digit: nothing to reorder
    let sum = 0;
    for (let d = 0; d < 256; d++) { const t = c[d]; c[d] = sum; sum += t; }
    for (let i = 0; i < n; i++) { const k = src[i]; dst[c[(k >>> shift) & 255]++] = k; }
    const t = src; src = dst; dst = t;
  }
  if (src !== a) for (let i = 0; i < n; i++) a[i] = src[i];
  return a;
}

/** True when the six AABB values at a[o] are finite and min <= max on every axis. */
function validAabb(a, o) {
  const x0 = a[o], y0 = a[o + 1], z0 = a[o + 2], x1 = a[o + 3], y1 = a[o + 4], z1 = a[o + 5];
  return x0 <= x1 && y0 <= y1 && z0 <= z1 && x0 - x0 === 0 && y0 - y0 === 0 && z0 - z0 === 0 &&
    x1 - x1 === 0 && y1 - y1 === 0 && z1 - z1 === 0;
}

/** Half the surface area of a box (the SAH cost measure; stays meaningful for flat boxes). */
function halfArea(x0, y0, z0, x1, y1, z1) {
  const dx = x1 - x0, dy = y1 - y0, dz = z1 - z0;
  return dx * dy + dy * dz + dz * dx;
}

/** Cost of descending into node `o / 6` when inserting the box l*: area of the union, minus its own area if internal. */
function descendCost(box, o, isLeaf, lx0, ly0, lz0, lx1, ly1, lz1) {
  const x0 = box[o], y0 = box[o + 1], z0 = box[o + 2], x1 = box[o + 3], y1 = box[o + 4], z1 = box[o + 5];
  const u = halfArea(x0 < lx0 ? x0 : lx0, y0 < ly0 ? y0 : ly0, z0 < lz0 ? z0 : lz0,
    x1 > lx1 ? x1 : lx1, y1 > ly1 ? y1 : ly1, z1 > lz1 ? z1 : lz1);
  return isLeaf ? u : u - halfArea(x0, y0, z0, x1, y1, z1);
}

/** Dynamic AABB tree broadphase with a persistent sorted pair list (see the file header). */
export class Broadphase {
  /**
   * @param {number} [capacity=4096] initial body-id capacity (node arrays hold 2 * capacity); grows on demand
   * @param {{ margin?: number, timing?: boolean }} [opts] margin: fat-AABB growth on every side (m), >= 0;
   *   timing: fill stats.ms (performance.now() allocates ~100 B per call in Node 26, so it is off by default)
   */
  constructor(capacity = 4096, { margin = 0.1, timing = false } = {}) {
    capacity = Math.max(16, capacity | 0);
    this.margin = margin > 0 ? margin : 0;
    /** Measure update() into stats.ms (allocates a little per update: the timer's return value). */
    this.timing = timing;
    /** Pair keys pairKey(a, b) (a < b), ascending, valid in [0, pairCount). Grows by doubling: re-read after update(). */
    this.pairs = new Uint32Array(capacity);
    this.pairCount = 0;
    /** Matches found by the last query() / raycast(), before truncation to out.length. */
    this.queryTotal = 0;
    /** Diagnostics of the last update(); the same object every time. */
    this.stats = { nodes: 0, proxies: 0, pairs: 0, candidates: 0, dirty: 0, height: 0, ms: 0 };
    this.root = NULL;
    this._bodies = null;
    this._hi = 0;           // one past the highest id that may hold a proxy
    this._proxies = 0;
    this._nodeCount = 0;
    this._allocBodyArrays(capacity);
    this._nodeCap = 0;
    this._freeNode = NULL;
    this.box = new Float64Array(0);
    this.child1 = this.child2 = this.parent = this.height = this.nodeBody = new Int32Array(0);
    this._growNodes(2 * capacity);
    this._cand = new Uint32Array(capacity); this._candCount = 0;   // sorted candidate keys
    this._tmp = new Uint32Array(capacity);                         // merge target / radix scratch (same length as _cand)
    this._new = new Uint32Array(256); this._newCount = 0;          // keys found by this update's dirty queries
    this._stack = new Int32Array(64);
    this._qbuf = new Uint32Array(256); this._qtmp = new Uint32Array(256);
  }

  /**
   * Refresh every alive body's proxy from bodies.aabb (insert new ones, re-insert those whose tight box left the fat
   * box, drop dead ones), then recompute `pairs` / `pairCount`.
   * @param {import('./state.js').BodyStore} bodies
   */
  update(bodies) {
    const t0 = this.timing ? performance.now() : 0;
    this._bodies = bodies;
    if (bodies.capacity > this._bodyCap) this._growBodyArrays(bodies.capacity);
    const n = bodies.count, hi = n > this._hi ? n : this._hi;
    const alive = bodies.alive, ab = bodies.aabb, type = bodies.type, layer = bodies.layer, mask = bodies.mask;
    const leafOf = this._leaf, sType = this._sType, sLayer = this._sLayer, sMask = this._sMask, moved = this._moved;
    let nMoved = 0;
    for (let id = 0; id < hi; id++) {
      const leaf = leafOf[id], o = id * 6;
      if (id >= n || alive[id] === 0 || !validAabb(ab, o)) {
        if (leaf !== NULL) this._destroyProxy(id);
        continue;
      }
      if (leaf === NULL) {
        this._createProxy(id, ab, o);
        moved[nMoved++] = id;
      } else {
        const box = this.box, f = leaf * 6;
        if (ab[o] < box[f] || ab[o + 1] < box[f + 1] || ab[o + 2] < box[f + 2] ||
            ab[o + 3] > box[f + 3] || ab[o + 4] > box[f + 4] || ab[o + 5] > box[f + 5]) {
          this._removeLeaf(leaf);
          this._setFat(leaf, ab, o);
          this._insertLeaf(leaf);
          this._markDirty(id);
          moved[nMoved++] = id;
        } else if (type[id] !== sType[id] || layer[id] !== sLayer[id] || mask[id] !== sMask[id]) {
          this._markDirty(id);
          moved[nMoved++] = id;
        }
      }
      sType[id] = type[id]; sLayer[id] = layer[id]; sMask[id] = mask[id];
    }
    this._hi = n;

    if (this._dirtyCount > 0) {
      // 1. drop the candidates that touch a dirty (re-inserted, re-filtered or removed) id
      const dirty = this._dirty;
      let cand = this._cand, k = 0;
      for (let i = 0, c = this._candCount; i < c; i++) {
        const key = cand[i];
        if (dirty[key >>> 16] === 0 && dirty[key & 0xffff] === 0) cand[k++] = key;
      }
      // 2. query the tree with every dirty proxy's fat box
      this._newCount = 0;
      for (let i = 0; i < nMoved; i++) this._queryProxy(moved[i], type, layer, mask);
      // 3. sort the new keys and merge them into the kept ones (disjoint sets: every new key has a dirty id)
      const m = this._newCount;
      if (m > 0) {
        this._ensureCand(k + m);
        cand = this._cand;
        const nw = this._new, out = this._tmp;
        radixSortU32(nw, m, out);
        let i = 0, j = 0, w = 0;
        while (i < k && j < m) { const a = cand[i], b = nw[j]; if (a < b) { out[w++] = a; i++; } else { out[w++] = b; j++; } }
        while (i < k) out[w++] = cand[i++];
        while (j < m) out[w++] = nw[j++];
        this._tmp = cand; this._cand = out;
        k = w;
      }
      this._candCount = k;
      const ids = this._dirtyIds;
      for (let i = 0, c = this._dirtyCount; i < c; i++) dirty[ids[i]] = 0;
      this._dirtyCount = 0;
    }

    // 4. pairs = candidates whose tight boxes overlap and that are not both asleep
    const cand = this._cand, cc = this._candCount, flags = bodies.flags;
    if (this.pairs.length < cc) {
      let len = this.pairs.length;
      while (len < cc) len *= 2;
      this.pairs = new Uint32Array(len);
    }
    const pairs = this.pairs;
    let pc = 0;
    for (let i = 0; i < cc; i++) {
      const key = cand[i], a = key >>> 16, b = key & 0xffff;
      if ((flags[a] & ASLEEP) !== 0 && (flags[b] & ASLEEP) !== 0) continue;
      const oa = a * 6, ob = b * 6;
      if (ab[oa] > ab[ob + 3] || ab[ob] > ab[oa + 3] || ab[oa + 1] > ab[ob + 4] || ab[ob + 1] > ab[oa + 4] ||
          ab[oa + 2] > ab[ob + 5] || ab[ob + 2] > ab[oa + 5]) continue;
      pairs[pc++] = key;
    }
    this.pairCount = pc;

    const s = this.stats;
    s.nodes = this._nodeCount; s.proxies = this._proxies; s.pairs = pc; s.candidates = cc; s.dirty = nMoved;
    s.height = this.root === NULL ? 0 : this.height[this.root];
    if (this.timing) s.ms = performance.now() - t0;
  }

  /**
   * Alive bodies whose TIGHT aabb overlaps the box (inclusive) and whose layer has a bit of `filterMask`, in ascending
   * id order. Writes at most out.length ids: when more match, the out.length LOWEST ids are written (deterministic)
   * and `queryTotal` holds the full count. Returns the number written.
   * @param {Int32Array} out @param {number} [filterMask=-1] @returns {number}
   */
  query(minX, minY, minZ, maxX, maxY, maxZ, out, filterMask = -1) {
    this.queryTotal = 0;
    const bodies = this._bodies;
    if (bodies === null || this.root === NULL || !(minX <= maxX && minY <= maxY && minZ <= maxZ)) return 0;
    const ab = bodies.aabb, layer = bodies.layer, alive = bodies.alive;
    const box = this.box, c1 = this.child1, c2 = this.child2, nb = this.nodeBody;
    let stack = this._stack, sp = 0, q = this._qbuf, nq = 0;
    stack[sp++] = this.root;
    while (sp > 0) {
      const nd = stack[--sp], o = nd * 6;
      if (box[o] > maxX || box[o + 1] > maxY || box[o + 2] > maxZ || box[o + 3] < minX || box[o + 4] < minY || box[o + 5] < minZ) continue;
      const a = c1[nd];
      if (a === NULL) {
        const id = nb[nd], t = id * 6;
        if (alive[id] === 0 || (layer[id] & filterMask) === 0) continue;
        if (ab[t] > maxX || ab[t + 1] > maxY || ab[t + 2] > maxZ || ab[t + 3] < minX || ab[t + 4] < minY || ab[t + 5] < minZ) continue;
        if (nq === q.length) q = this._growQuery();
        q[nq++] = id;
      } else {
        if (sp + 2 > stack.length) stack = this._growStack();
        stack[sp++] = a; stack[sp++] = c2[nd];
      }
    }
    return this._emitSorted(nq, out);
  }

  /**
   * Candidate bodies for a ray o + t * d, t in [0, maxT] (d need not be unit; t is in units of d): alive bodies whose
   * FAT aabb the ray segment touches, whose layer has a bit of `filterMask`, ascending id. A superset of the exact
   * ray-vs-tight-aabb hits; the caller runs the exact test. Truncation as query(). Returns the number written.
   * @param {Int32Array} out @param {number} [filterMask=-1] @returns {number}
   */
  raycast(ox, oy, oz, dx, dy, dz, maxT, out, filterMask = -1) {
    this.queryTotal = 0;
    const bodies = this._bodies;
    if (bodies === null || this.root === NULL || !(maxT >= 0)) return 0;
    if (!(ox - ox === 0 && oy - oy === 0 && oz - oz === 0 && dx - dx === 0 && dy - dy === 0 && dz - dz === 0)) return 0;
    const ix = dx !== 0 ? 1 / dx : 0, iy = dy !== 0 ? 1 / dy : 0, iz = dz !== 0 ? 1 / dz : 0;
    const layer = bodies.layer, alive = bodies.alive;
    const box = this.box, c1 = this.child1, c2 = this.child2, nb = this.nodeBody;
    let stack = this._stack, sp = 0, q = this._qbuf, nq = 0;
    stack[sp++] = this.root;
    while (sp > 0) {
      const nd = stack[--sp], o = nd * 6;
      let tmin = 0, tmax = maxT;
      if (dx === 0) { if (ox < box[o] || ox > box[o + 3]) continue; }
      else {
        let ta = (box[o] - ox) * ix, tb = (box[o + 3] - ox) * ix;
        if (ta > tb) { const s = ta; ta = tb; tb = s; }
        if (ta > tmin) tmin = ta;
        if (tb < tmax) tmax = tb;
        if (tmin > tmax) continue;
      }
      if (dy === 0) { if (oy < box[o + 1] || oy > box[o + 4]) continue; }
      else {
        let ta = (box[o + 1] - oy) * iy, tb = (box[o + 4] - oy) * iy;
        if (ta > tb) { const s = ta; ta = tb; tb = s; }
        if (ta > tmin) tmin = ta;
        if (tb < tmax) tmax = tb;
        if (tmin > tmax) continue;
      }
      if (dz === 0) { if (oz < box[o + 2] || oz > box[o + 5]) continue; }
      else {
        let ta = (box[o + 2] - oz) * iz, tb = (box[o + 5] - oz) * iz;
        if (ta > tb) { const s = ta; ta = tb; tb = s; }
        if (ta > tmin) tmin = ta;
        if (tb < tmax) tmax = tb;
        if (tmin > tmax) continue;
      }
      const a = c1[nd];
      if (a === NULL) {
        const id = nb[nd];
        if (alive[id] === 0 || (layer[id] & filterMask) === 0) continue;
        if (nq === q.length) q = this._growQuery();
        q[nq++] = id;
      } else {
        if (sp + 2 > stack.length) stack = this._growStack();
        stack[sp++] = a; stack[sp++] = c2[nd];
      }
    }
    return this._emitSorted(nq, out);
  }

  /** Drop body `id`'s proxy (call when the body is freed). Its pairs disappear at the next update(). */
  remove(id) {
    if (id >= 0 && id < this._bodyCap && this._leaf[id] !== NULL) this._destroyProxy(id);
  }

  // ---- pair search ----------------------------------------------------------------------------------------------

  /** Append to _new every candidate pair of dirty proxy `m` (fat-box overlap + permanent filters). */
  _queryProxy(m, type, layer, mask) {
    const leaf = this._leaf[m], box = this.box, c1 = this.child1, c2 = this.child2, nb = this.nodeBody, dirty = this._dirty;
    const f = leaf * 6;
    const minX = box[f], minY = box[f + 1], minZ = box[f + 2], maxX = box[f + 3], maxY = box[f + 4], maxZ = box[f + 5];
    const dynM = type[m] === DYNAMIC, layM = layer[m], maskM = mask[m];
    let stack = this._stack, sp = 0, nw = this._new, nc = this._newCount;
    stack[sp++] = this.root;
    while (sp > 0) {
      const nd = stack[--sp], o = nd * 6;
      if (box[o] > maxX || box[o + 1] > maxY || box[o + 2] > maxZ || box[o + 3] < minX || box[o + 4] < minY || box[o + 5] < minZ) continue;
      const a = c1[nd];
      if (a === NULL) {
        if (nd === leaf) continue;
        const other = nb[nd];
        if (dirty[other] !== 0 && other < m) continue;                   // found (or to be found) by other's query
        if (!dynM && type[other] !== DYNAMIC) continue;                    // both static / kinematic
        if ((layM & mask[other]) === 0 || (layer[other] & maskM) === 0) continue;
        if (nc === nw.length) { this._newCount = nc; nw = this._growNew(); }
        nw[nc++] = other < m ? other * 65536 + m : m * 65536 + other;
      } else {
        if (sp + 2 > stack.length) stack = this._growStack();
        stack[sp++] = a; stack[sp++] = c2[nd];
      }
    }
    this._newCount = nc;
  }

  /** Sort _qbuf[0 .. nq) and copy the lowest min(nq, out.length) ids into out. */
  _emitSorted(nq, out) {
    const q = this._qbuf;
    if (this._qtmp.length < q.length) this._qtmp = new Uint32Array(q.length);
    radixSortU32(q, nq, this._qtmp);
    const w = nq < out.length ? nq : out.length;
    for (let i = 0; i < w; i++) out[i] = q[i];
    this.queryTotal = nq;
    return w;
  }

  // ---- proxies --------------------------------------------------------------------------------------------------

  _createProxy(id, ab, o) {
    const leaf = this._allocNode();
    this.nodeBody[leaf] = id;
    this._setFat(leaf, ab, o);
    this._insertLeaf(leaf);
    this._leaf[id] = leaf;
    this._proxies++;
    this._markDirty(id);
  }

  _destroyProxy(id) {
    const leaf = this._leaf[id];
    this._removeLeaf(leaf);
    this._releaseNode(leaf);
    this._leaf[id] = NULL;
    this._proxies--;
    this._markDirty(id);
  }

  _setFat(leaf, ab, o) {
    const box = this.box, f = leaf * 6, m = this.margin;
    box[f] = ab[o] - m; box[f + 1] = ab[o + 1] - m; box[f + 2] = ab[o + 2] - m;
    box[f + 3] = ab[o + 3] + m; box[f + 4] = ab[o + 4] + m; box[f + 5] = ab[o + 5] + m;
  }

  // Invariant the pair dedupe in _queryProxy (`dirty[other] !== 0 && other < m`) relies on: every dirty id that has
  // a leaf is also in this update's `moved` list (dirty ids come from _createProxy / re-insertion / a filter change,
  // all of which push to `moved`, or from _destroyProxy, which leaves no leaf), so the skipped pair is found by the
  // smaller id's own query.
  _markDirty(id) {
    if (this._dirty[id] === 0) { this._dirty[id] = 1; this._dirtyIds[this._dirtyCount++] = id; }
  }

  // ---- tree -----------------------------------------------------------------------------------------------------

  _allocNode() {
    if (this._freeNode === NULL) this._growNodes(this._nodeCap * 2);
    const nd = this._freeNode;
    this._freeNode = this.parent[nd];
    this.parent[nd] = NULL; this.child1[nd] = NULL; this.child2[nd] = NULL; this.height[nd] = 0; this.nodeBody[nd] = NULL;
    this._nodeCount++;
    return nd;
  }

  _releaseNode(nd) {
    this.parent[nd] = this._freeNode; this.height[nd] = -1; this.child1[nd] = NULL; this.child2[nd] = NULL; this.nodeBody[nd] = NULL;
    this._freeNode = nd;
    this._nodeCount--;
  }

  /** box[dst] = box[a] U box[b] */
  _union(dst, a, b) {
    const box = this.box, d = dst * 6, p = a * 6, q = b * 6;
    box[d] = box[p] < box[q] ? box[p] : box[q];
    box[d + 1] = box[p + 1] < box[q + 1] ? box[p + 1] : box[q + 1];
    box[d + 2] = box[p + 2] < box[q + 2] ? box[p + 2] : box[q + 2];
    box[d + 3] = box[p + 3] > box[q + 3] ? box[p + 3] : box[q + 3];
    box[d + 4] = box[p + 4] > box[q + 4] ? box[p + 4] : box[q + 4];
    box[d + 5] = box[p + 5] > box[q + 5] ? box[p + 5] : box[q + 5];
  }

  _insertLeaf(leaf) {
    if (this.root === NULL) { this.root = leaf; this.parent[leaf] = NULL; return; }
    const np = this._allocNode();   // before caching the arrays: allocation may grow them
    const box = this.box, c1 = this.child1, c2 = this.child2, par = this.parent, h = this.height;
    const l = leaf * 6;
    const lx0 = box[l], ly0 = box[l + 1], lz0 = box[l + 2], lx1 = box[l + 3], ly1 = box[l + 4], lz1 = box[l + 5];
    // descend to the best sibling (surface-area heuristic, Box2D b2DynamicTree::InsertLeaf)
    let index = this.root;
    while (c1[index] !== NULL) {
      const o = index * 6, a = c1[index], b = c2[index];
      const x0 = box[o], y0 = box[o + 1], z0 = box[o + 2], x1 = box[o + 3], y1 = box[o + 4], z1 = box[o + 5];
      const area = halfArea(x0, y0, z0, x1, y1, z1);
      const combined = halfArea(x0 < lx0 ? x0 : lx0, y0 < ly0 ? y0 : ly0, z0 < lz0 ? z0 : lz0,
        x1 > lx1 ? x1 : lx1, y1 > ly1 ? y1 : ly1, z1 > lz1 ? z1 : lz1);
      const cost = 2 * combined;                 // new parent for this node and the leaf
      const inherit = 2 * (combined - area);     // growth pushed onto every ancestor when descending
      const cost1 = descendCost(box, a * 6, c1[a] === NULL, lx0, ly0, lz0, lx1, ly1, lz1) + inherit;
      const cost2 = descendCost(box, b * 6, c1[b] === NULL, lx0, ly0, lz0, lx1, ly1, lz1) + inherit;
      if (cost < cost1 && cost < cost2) break;
      index = cost1 < cost2 ? a : b;
    }
    const sibling = index, oldParent = par[sibling];
    par[np] = oldParent;
    h[np] = h[sibling] + 1;
    this._union(np, sibling, leaf);
    if (oldParent !== NULL) { if (c1[oldParent] === sibling) c1[oldParent] = np; else c2[oldParent] = np; }
    else this.root = np;
    c1[np] = sibling; c2[np] = leaf; par[sibling] = np; par[leaf] = np;
    this._refitUp(par[leaf]);
  }

  _removeLeaf(leaf) {
    if (leaf === this.root) { this.root = NULL; return; }
    const c1 = this.child1, c2 = this.child2, par = this.parent;
    const p = par[leaf], gp = par[p], sibling = c1[p] === leaf ? c2[p] : c1[p];
    if (gp !== NULL) {
      if (c1[gp] === p) c1[gp] = sibling; else c2[gp] = sibling;
      par[sibling] = gp;
      this._releaseNode(p);
      this._refitUp(gp);
    } else {
      this.root = sibling; par[sibling] = NULL;
      this._releaseNode(p);
    }
    par[leaf] = NULL;
  }

  /** Walk from `index` to the root: rotate if unbalanced, then refresh height and box. */
  _refitUp(index) {
    const c1 = this.child1, c2 = this.child2, par = this.parent, h = this.height;
    while (index !== NULL) {
      index = this._balance(index);
      const a = c1[index], b = c2[index];
      h[index] = 1 + (h[a] > h[b] ? h[a] : h[b]);
      this._union(index, a, b);
      index = par[index];
    }
  }

  /**
   * AVL-style rotation of internal node iA when its children's heights differ by more than 1 (Box2D
   * b2DynamicTree::Balance). Returns the subtree's new root.
   */
  _balance(iA) {
    const c1 = this.child1, c2 = this.child2, par = this.parent, h = this.height;
    if (c1[iA] === NULL || h[iA] < 2) return iA;
    const iB = c1[iA], iC = c2[iA];
    const balance = h[iC] - h[iB];
    if (balance > 1) {                 // rotate C up
      const iF = c1[iC], iG = c2[iC];
      c1[iC] = iA; par[iC] = par[iA]; par[iA] = iC;
      const pC = par[iC];
      if (pC !== NULL) { if (c1[pC] === iA) c1[pC] = iC; else c2[pC] = iC; } else this.root = iC;
      if (h[iF] > h[iG]) {
        c2[iC] = iF; c2[iA] = iG; par[iG] = iA;
        this._union(iA, iB, iG); this._union(iC, iA, iF);
        h[iA] = 1 + (h[iB] > h[iG] ? h[iB] : h[iG]); h[iC] = 1 + (h[iA] > h[iF] ? h[iA] : h[iF]);
      } else {
        c2[iC] = iG; c2[iA] = iF; par[iF] = iA;
        this._union(iA, iB, iF); this._union(iC, iA, iG);
        h[iA] = 1 + (h[iB] > h[iF] ? h[iB] : h[iF]); h[iC] = 1 + (h[iA] > h[iG] ? h[iA] : h[iG]);
      }
      return iC;
    }
    if (balance < -1) {                // rotate B up
      const iD = c1[iB], iE = c2[iB];
      c1[iB] = iA; par[iB] = par[iA]; par[iA] = iB;
      const pB = par[iB];
      if (pB !== NULL) { if (c1[pB] === iA) c1[pB] = iB; else c2[pB] = iB; } else this.root = iB;
      if (h[iD] > h[iE]) {
        c2[iB] = iD; c1[iA] = iE; par[iE] = iA;
        this._union(iA, iC, iE); this._union(iB, iA, iD);
        h[iA] = 1 + (h[iC] > h[iE] ? h[iC] : h[iE]); h[iB] = 1 + (h[iA] > h[iD] ? h[iA] : h[iD]);
      } else {
        c2[iB] = iE; c1[iA] = iD; par[iD] = iA;
        this._union(iA, iC, iD); this._union(iB, iA, iE);
        h[iA] = 1 + (h[iC] > h[iD] ? h[iC] : h[iD]); h[iB] = 1 + (h[iA] > h[iE] ? h[iA] : h[iE]);
      }
      return iB;
    }
    return iA;
  }

  /**
   * Check the tree invariants (parent / child links, heights, boxes enclose children, leaf fat boxes enclose their
   * body's aabb as of the last update, leaf <-> proxy maps, free list; balance is a heuristic, not checked). Debug /
   * test helper, allocates; returns '' when consistent, else the first problem found.
   * @returns {string}
   */
  validate() {
    const c1 = this.child1, c2 = this.child2, par = this.parent, h = this.height, box = this.box, nb = this.nodeBody;
    if (this.root === NULL) return this._proxies === 0 && this._nodeCount === 0 ? '' : 'empty root but proxies / nodes left';
    if (par[this.root] !== NULL) return 'root has a parent';
    let leaves = 0, nodes = 0;
    const stack = [this.root];
    while (stack.length) {
      const nd = stack.pop();
      nodes++;
      if (h[nd] < 0) return `node ${nd} is on the free list`;
      if (c1[nd] === NULL) {
        if (c2[nd] !== NULL || h[nd] !== 0) return `leaf ${nd} malformed`;
        const id = nb[nd];
        if (id < 0 || this._leaf[id] !== nd) return `leaf ${nd} / body ${id} map mismatch`;
        const ab = this._bodies !== null ? this._bodies.aabb : null;
        for (let k = 0; ab !== null && k < 3; k++) {
          if (ab[id * 6 + k] < box[nd * 6 + k] || ab[id * 6 + 3 + k] > box[nd * 6 + 3 + k]) return `leaf ${nd}: fat box misses body ${id}`;
        }
        leaves++;
        continue;
      }
      const a = c1[nd], b = c2[nd];
      if (par[a] !== nd || par[b] !== nd) return `node ${nd}: child parent link broken`;
      if (h[nd] !== 1 + Math.max(h[a], h[b])) return `node ${nd}: height`;
      for (let k = 0; k < 3; k++) {
        const lo = Math.min(box[a * 6 + k], box[b * 6 + k]), hi = Math.max(box[a * 6 + 3 + k], box[b * 6 + 3 + k]);
        if (box[nd * 6 + k] > lo || box[nd * 6 + 3 + k] < hi) return `node ${nd}: box does not enclose children`;
      }
      stack.push(a, b);
    }
    if (leaves !== this._proxies) return `leaves ${leaves} !== proxies ${this._proxies}`;
    if (nodes !== this._nodeCount) return `reachable nodes ${nodes} !== allocated ${this._nodeCount}`;
    let free = 0;
    for (let nd = this._freeNode; nd !== NULL; nd = par[nd]) { free++; if (free > this._nodeCap) return 'free list cycle'; }
    if (free + nodes !== this._nodeCap) return `free ${free} + used ${nodes} !== capacity ${this._nodeCap}`;
    return '';
  }

  // ---- storage (growth paths: allocate, documented in the header) ------------------------------------------------

  _allocBodyArrays(cap) {
    this._bodyCap = cap;
    this._leaf = new Int32Array(cap).fill(NULL);   // body id -> leaf node
    this._sType = new Uint8Array(cap);              // type / layer / mask snapshot at the last query of the proxy
    this._sLayer = new Int32Array(cap);
    this._sMask = new Int32Array(cap);
    this._dirty = new Uint8Array(cap);
    this._dirtyIds = new Int32Array(cap); this._dirtyCount = 0;
    this._moved = new Int32Array(cap);
  }

  _growBodyArrays(cap) {
    const old = { leaf: this._leaf, t: this._sType, l: this._sLayer, m: this._sMask, d: this._dirty, di: this._dirtyIds };
    const dc = this._dirtyCount;
    this._allocBodyArrays(cap);
    this._leaf.set(old.leaf); this._sType.set(old.t); this._sLayer.set(old.l); this._sMask.set(old.m);
    this._dirty.set(old.d); this._dirtyIds.set(old.di); this._dirtyCount = dc;
  }

  _growNodes(cap) {
    const old = this._nodeCap;
    const grow = (a, Type, k) => { const b = new Type(cap * k); b.set(a); return b; };
    this.box = grow(this.box, Float64Array, 6);
    this.child1 = grow(this.child1, Int32Array, 1); this.child2 = grow(this.child2, Int32Array, 1);
    this.parent = grow(this.parent, Int32Array, 1); this.height = grow(this.height, Int32Array, 1);
    this.nodeBody = grow(this.nodeBody, Int32Array, 1);
    for (let i = cap - 1; i >= old; i--) {   // chain the new nodes, lowest index first
      this.parent[i] = this._freeNode; this.height[i] = -1; this.child1[i] = NULL; this.child2[i] = NULL; this.nodeBody[i] = NULL;
      this._freeNode = i;
    }
    this._nodeCap = cap;
  }

  _ensureCand(n) {
    if (this._cand.length >= n) return;
    let len = this._cand.length;
    while (len < n) len *= 2;
    const c = new Uint32Array(len); c.set(this._cand);
    this._cand = c; this._tmp = new Uint32Array(len);
  }

  _growNew() {
    const b = new Uint32Array(this._new.length * 2); b.set(this._new);
    this._new = b;
    if (this._tmp.length < b.length) this._ensureCand(b.length);   // radix scratch must hold the new keys
    return b;
  }

  _growStack() {
    const b = new Int32Array(this._stack.length * 2); b.set(this._stack);
    return (this._stack = b);
  }

  _growQuery() {
    const b = new Uint32Array(this._qbuf.length * 2); b.set(this._qbuf);
    return (this._qbuf = b);
  }
}
