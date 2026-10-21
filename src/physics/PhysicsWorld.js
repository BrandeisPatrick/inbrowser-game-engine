// Engine-facing physics world (pure JS, deterministic, Node-safe):
//   static AABB colliders in a uniform grid, pluggable dynamic collider providers (traffic, NPCs),
//   an optional heightfield, an optional surface provider (per-point tyre grip / rolling resistance multipliers),
//   and queries: colliders near a point, ground/support height, surface, raycast, sphere cast.
// Bodies (e.g. VehicleController) query it each step; it also implements the legacy world query API
// (getGroundHeight / queryColliders), so it can be passed anywhere a game `world` was expected.
import { Vector3 } from 'three';
import { ColliderGrid } from './spatial.js';
import { rayAabb, sphereCastAabb, closestPointOnAabb } from './collision.js';

// ---- collision layers ------------------------------------------------------------------------------------
// A box's layer bit comes from its `kind`: numeric kinds 0..23 map to bit `kind`; string kinds ('vehicle', ...)
// are given bits 24..29 on first use; the heightfield is LAYER.GROUND (bit 30). Query masks OR these together.
export const LAYER = Object.freeze({ GROUND: 1 << 30, OTHER: 1 << 31, ALL: -1 });
const _named = new Map();
let _nextNamed = 24;
/** Layer bit of a collider `kind` (number 0..23 or string). */
export function layerOf(kind) {
  if (kind === undefined || kind === null) return 1;
  if (typeof kind === 'number') return kind >= 0 && kind < 24 ? 1 << kind : LAYER.OTHER;
  let bit = _named.get(kind);
  if (bit === undefined) { bit = _nextNamed < 30 ? 1 << _nextNamed++ : LAYER.OTHER; _named.set(kind, bit); }
  return bit;
}
layerOf('vehicle'); // reserve bit 24 for traffic so masks are stable across runs

const EMPTY_OPTS = Object.freeze({});
const makeHit = () => ({ distance: 0, point: new Vector3(), normal: new Vector3(), box: null, kind: null, ground: false });
const _o = new Vector3(), _d = new Vector3(), _c = new Vector3(), _cp = new Vector3(), _n = new Vector3(), _bn = new Vector3();

export class PhysicsWorld {
  /**
   * @param {object} [o]
   * @param {number} [o.cellSize=32]   static grid cell size (m)
   * @param {ColliderGrid} [o.grid]    adopt an existing grid (e.g. one a world builder already filled)
   * @param {object|Function} [o.heightfield]  { heightAt(x, z, y) } or (x, z, y) => height
   * @param {number} [o.groundLevel=0] ground height when there is no heightfield
   * @param {number} [o.stepHeight=0.35] box tops up to this far above refY count as support (kerbs, decks)
   * @param {Function} [o.surface]     surface provider fn(x, z, out) (see setSurfaceProvider)
   */
  constructor(o = {}) {
    this.grid = o.grid || new ColliderGrid(o.cellSize || 32);
    this.providers = [];
    this.groundLevel = Number.isFinite(o.groundLevel) ? o.groundLevel : 0;
    this.stepHeight = o.stepHeight ?? 0.35;
    this.heightfield = null;
    this._hf = null;
    this._surface = null;
    this._tmp = [];
    this._dyn = [];
    this._dynSeen = new Set();
    // raycast state (bound visitor: no per-call closures)
    this._rc = { ox: 0, oy: 0, oz: 0, dx: 0, dy: 0, dz: 0, pad: 0, mask: -1, filter: null, box: null, n: new Vector3() };
    this._visit = (b, best) => this._testBox(b, best);
    if (o.heightfield) this.setHeightfield(o.heightfield);
    if (o.surface) this.setSurfaceProvider(o.surface);
  }

  // ---- static colliders ------------------------------------------------------------------------------
  /** Add a static axis-aligned box; returns its index (-1 if degenerate). */
  addStaticBox(minX, minY, maxX, maxY, minZ, maxZ, kind = 0) { return this.grid.add(minX, minY, maxX, maxY, minZ, maxZ, kind); }
  /** All static boxes (shared objects: read only). */
  get staticBoxes() { return this.grid.boxes; }
  /** Static boxes whose XZ footprint is within r of (x,z). Clears and fills `out`. */
  queryStatic(x, z, r, out = []) { return this.grid.query(x, z, r, out); }

  // ---- dynamic colliders -----------------------------------------------------------------------------
  /** Register fn(x, z, r, out) that APPENDS moving boxes near (x,z) to `out` (e.g. traffic.queryColliders). */
  addDynamicProvider(fn) { if (typeof fn === 'function' && !this.providers.includes(fn)) this.providers.push(fn); return fn; }
  removeDynamicProvider(fn) { const i = this.providers.indexOf(fn); if (i >= 0) this.providers.splice(i, 1); }
  /** Dynamic boxes near (x,z). Clears and fills `out`. */
  queryDynamic(x, z, r, out = []) {
    out.length = 0;
    for (let i = 0; i < this.providers.length; i++) this.providers[i](x, z, r, out);
    return out;
  }
  /** Static + dynamic boxes near (x,z) (the legacy `world.queryColliders` contract). Clears and fills `out`. */
  queryColliders(x, z, r, out = []) {
    this.grid.query(x, z, r, out);
    for (let i = 0; i < this.providers.length; i++) this.providers[i](x, z, r, out);
    return out;
  }

  // ---- heightfield -----------------------------------------------------------------------------------
  /** hf: { heightAt(x, z, y) } | (x, z, y) => number | null. `y` is the querying object's height (lets a
   *  heightfield ignore bridge decks far above it). */
  setHeightfield(hf) {
    this.heightfield = hf || null;
    this._hf = !hf ? null : typeof hf === 'function' ? hf : typeof hf.heightAt === 'function' ? (x, z, y) => hf.heightAt(x, z, y) : null;
    return this;
  }
  /** Ground height at (x,z) (groundLevel without a heightfield; non-finite heights fall back to it). */
  groundHeight(x, z, y) {
    if (!this._hf) return this.groundLevel;
    const h = this._hf(x, z, y);
    return typeof h === 'number' && Number.isFinite(h) ? h : this.groundLevel;
  }
  /** Legacy alias of groundHeight (so a PhysicsWorld can stand in for a game `world`). */
  getGroundHeight(x, z, y) { return this.groundHeight(x, z, y); }

  // ---- surfaces --------------------------------------------------------------------------------------
  /**
   * fn(x, z, out) describes the ground material at (x,z): either write out.mu / out.crr (grip and rolling-
   * resistance MULTIPLIERS, default 1) and optionally out.tag, or return an object { mu?, crr?, tag? } (e.g. a
   * shared constant). null removes it. VehicleController asks once per wheel per step.
   */
  setSurfaceProvider(fn) { this._surface = typeof fn === 'function' ? fn : null; return this; }
  /** Surface multipliers at (x,z): fills and returns `out` ({ mu, crr, tag }; 1, 1, null without a provider). */
  surfaceAt(x, z, out = { mu: 1, crr: 1, tag: null }) {
    out.mu = 1; out.crr = 1; out.tag = null;
    if (!this._surface) return out;
    const r = this._surface(x, z, out);
    if (r && r !== out) { out.mu = r.mu ?? 1; out.crr = r.crr ?? 1; out.tag = r.tag ?? null; }
    return out;
  }

  /** Ground normal at (x,z) from central differences of the heightfield (e = sample spacing, m). */
  groundNormal(x, z, y, out = new Vector3(), e = 0.5) {
    if (!this._hf) return out.set(0, 1, 0);
    const hx = this.groundHeight(x + e, z, y) - this.groundHeight(x - e, z, y);
    const hz = this.groundHeight(x, z + e, y) - this.groundHeight(x, z - e, y);
    return out.set(-hx / (2 * e), 1, -hz / (2 * e)).normalize();
  }

  /**
   * Highest walkable surface under (x,z) that is not above refY + step: the ground or the top of a collider
   * box (rooftops, bridge decks, kerbs).
   */
  supportAt(x, z, refY, step = this.stepHeight) {
    let s = this.groundHeight(x, z, refY);
    const cs = this.queryColliders(x, z, 0, this._tmp);
    const lim = refY + step;
    for (let i = 0; i < cs.length; i++) {
      const b = cs[i];
      if (x > b.minX && x < b.maxX && z > b.minZ && z < b.maxZ && b.maxY <= lim && b.maxY > s) s = b.maxY;
    }
    return s;
  }

  // ---- casts -----------------------------------------------------------------------------------------
  /**
   * Cast a ray against boxes (static + dynamic) and the heightfield.
   * @param {Vector3} origin
   * @param {Vector3} dir      need not be normalised
   * @param {number} [maxDist=1e4]
   * @param {object} [opts]  mask (LAYER bits, see layerOf; default all), filter(box) => bool,
   *   statics/dynamics/ground (default true), out (hit object to reuse), hfStep {min, max} march step
   * @returns {{distance, point, normal, box, kind, ground}|null}
   */
  raycast(origin, dir, maxDist = 1e4, opts = EMPTY_OPTS) { return this._cast(origin, dir, 0, maxDist, opts); }

  /**
   * Sweep a sphere of `radius` along a ray; same options/result as raycast (`distance` = centre travel,
   * `point` = contact point). Boxes are exact (faces, edges, corners); the heightfield is tested under the
   * sphere centre (exact on flat ground, slightly late on steep slopes).
   */
  sphereCast(origin, dir, radius, maxDist = 1e4, opts = EMPTY_OPTS) { return this._cast(origin, dir, Math.max(0, radius || 0), maxDist, opts); }

  _cast(origin, dir, radius, maxDist, opts) {
    const len = Math.hypot(dir.x, dir.y, dir.z);
    if (!(len > 1e-12) || !Number.isFinite(origin.x + origin.y + origin.z)) return null;
    maxDist = Math.min(Number.isFinite(maxDist) ? maxDist : 1e4, 1e5);
    if (!(maxDist > 0)) return null;
    const rc = this._rc;
    rc.ox = origin.x; rc.oy = origin.y; rc.oz = origin.z;
    rc.dx = dir.x / len; rc.dy = dir.y / len; rc.dz = dir.z / len;
    rc.pad = radius; rc.mask = opts.mask ?? LAYER.ALL; rc.filter = opts.filter || null; rc.box = null;
    let best = maxDist, box = null, ground = false;
    // static boxes: DDA over the grid (cells padded by the sphere radius via a slightly longer reach)
    if (opts.statics !== false) {
      if (radius > 0) best = this._castBoxesAlongRay(best, true);
      else if (typeof this.grid.raycast === 'function') best = this.grid.raycast(rc.ox, rc.oz, rc.dx, rc.dz, best, this._visit);
      else best = this._castBoxesAlongRay(best, true);
      if (rc.box) { box = rc.box; _bn.copy(rc.n); }
    }
    // dynamic boxes
    if (opts.dynamics !== false && this.providers.length) {
      rc.box = null;
      best = this._castBoxesAlongRay(best, false);
      if (rc.box) { box = rc.box; _bn.copy(rc.n); }
    }
    // heightfield
    if (opts.ground !== false && (rc.mask & LAYER.GROUND) !== 0) {
      const t = this._marchHeightfield(best, radius, opts.hfStep);
      if (t < best) { best = t; ground = true; box = null; }
    }
    if (!box && !ground) return null;
    const hit = opts.out || makeHit();
    hit.distance = best; hit.box = box; hit.kind = box ? box.kind : null; hit.ground = ground;
    _c.set(rc.ox + rc.dx * best, rc.oy + rc.dy * best, rc.oz + rc.dz * best); // ray point / sphere centre at hit
    if (ground) {
      this.groundNormal(_c.x, _c.z, _c.y, hit.normal);
      if (radius > 0) hit.point.set(_c.x, this.groundHeight(_c.x, _c.z, _c.y), _c.z);
      else hit.point.copy(_c);
    } else if (radius > 0) {
      closestPointOnAabb(box, _c, _cp);
      _n.subVectors(_c, _cp);
      const dl = _n.length();
      if (dl > 1e-6) hit.normal.copy(_n).divideScalar(dl); else hit.normal.copy(_bn);
      hit.point.copy(_cp);
    } else {
      hit.normal.copy(_bn);
      hit.point.copy(_c);
    }
    return hit;
  }

  // box visitor shared by grid DDA and the radius-query path
  _testBox(b, best) {
    const rc = this._rc;
    if (rc.mask !== LAYER.ALL && (rc.mask & layerOf(b.kind)) === 0) return Infinity;
    if (rc.filter && !rc.filter(b)) return Infinity;
    const t = rc.pad > 0
      ? sphereCastAabb(rc.ox, rc.oy, rc.oz, rc.dx, rc.dy, rc.dz, b, best, rc.pad, _n)
      : rayAabb(rc.ox, rc.oy, rc.oz, rc.dx, rc.dy, rc.dz, b, best, 0, _n);
    if (t < best) { rc.box = b; rc.n.copy(_n); }
    return t;
  }

  // Gather boxes with overlapping radius queries along the ray's XZ path (dynamic providers, sphere casts,
  // grids without raycast()). Each box is tested once per call.
  _castBoxesAlongRay(best, statics) {
    const rc = this._rc, seen = this._dynSeen, out = this._dyn;
    seen.clear();
    const hl = Math.hypot(rc.dx, rc.dz) * best;      // horizontal reach
    const S = 16, n = Math.max(1, Math.ceil(hl / S));
    const r = (hl / n) * 0.5 * Math.SQRT2 + rc.pad + 0.01;
    for (let i = 0; i < n; i++) {
      const t = ((i + 0.5) / n) * best;
      const x = rc.ox + rc.dx * t, z = rc.oz + rc.dz * t;
      if (statics) this.grid.query(x, z, r, out); else this.queryDynamic(x, z, r, out);
      for (let j = 0; j < out.length; j++) {
        const b = out[j];
        if (seen.has(b)) continue;
        seen.add(b);
        const th = this._testBox(b, best);
        if (th < best) best = th;
      }
    }
    return best;
  }

  // March the ray (or sphere centre) over the heightfield with an adaptive step, then bisect the crossing.
  _marchHeightfield(maxT, radius, step) {
    const rc = this._rc;
    const minStep = (step && step.min) || 0.25, maxStep = (step && step.max) || 8;
    const gapAt = (t) => {
      const y = rc.oy + rc.dy * t;
      return y - radius - this.groundHeight(rc.ox + rc.dx * t, rc.oz + rc.dz * t, y);
    };
    let t = 0, gap = gapAt(0);
    if (gap <= 0) return 0;
    for (let guard = 0; guard < 100000 && t < maxT; guard++) {
      const tn = Math.min(maxT, t + Math.min(maxStep, Math.max(minStep, gap * 0.9)));
      const gn = gapAt(tn);
      if (gn <= 0) {
        let a = t, b = tn;
        for (let k = 0; k < 24 && b - a > 1e-4; k++) { const m = (a + b) * 0.5; if (gapAt(m) > 0) a = m; else b = m; }
        return b;
      }
      t = tn; gap = gn;
    }
    return Infinity;
  }

  // ---- adapters --------------------------------------------------------------------------------------
  /**
   * Wrap a legacy game world ({ getGroundHeight, queryColliders, colliders?: ColliderGrid }) without copying:
   * its ColliderGrid is shared (when present) and its ground query becomes the heightfield.
   * `o.dynamic`: extra providers fn(x, z, r, out) for moving colliders (e.g. traffic). Without a shareable grid
   * the world's own queryColliders (assumed to include everything) becomes the only provider instead.
   * `o.deckStep`: passed as the 4th argument of world.getGroundHeight(x, z, y, step) when given (how far above the
   * querying height an elevated deck may be and still count as the ground; MapWorld defaults to 1.6 m).
   */
  static fromWorld(world, o = {}) {
    const { dynamic = [], deckStep, ...rest } = o;
    const grid = world && world.colliders instanceof ColliderGrid ? world.colliders : undefined;
    const pw = new PhysicsWorld({ ...rest, grid });
    if (world && typeof world.getGroundHeight === 'function') {
      if (Number.isFinite(deckStep)) pw.setHeightfield((x, z, y) => world.getGroundHeight(x, z, y, deckStep));
      else pw.setHeightfield((x, z, y) => world.getGroundHeight(x, z, y));
    }
    if (world && typeof world.surfaceAt === 'function') pw.setSurfaceProvider((x, z, out) => world.surfaceAt(x, z, out));
    if (grid) for (const fn of dynamic) pw.addDynamicProvider(fn);
    else if (world && typeof world.queryColliders === 'function') pw.addDynamicProvider((x, z, r, out) => { const a = world.queryColliders(x, z, r); if (a) for (let i = 0; i < a.length; i++) out.push(a[i]); });
    return pw;
  }
}
