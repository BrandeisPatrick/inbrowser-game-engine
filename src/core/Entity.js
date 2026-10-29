// Entity + EntitySystem: a thin, opt-in component layer over THREE.Object3D. Nothing else in the engine needs it.
//
//   const entities = engine.addSystem(new EntitySystem(), 'entities');   // register once
//   const car = new Entity('car'); scene.add(car); entities.add(car);
//   const hp = car.addComponent(new Health(100));   // sets hp.entity = car, then hp.onAttach?.(car)
//   car.getComponent(Health); car.getComponent('health');   // constructor (instanceof), or c.name / constructor.name
//   (constructor names do not survive minification: look up by constructor, or give components a `name`)
//   car.removeComponent(hp); car.destroy();
//
// A component is any object with some of:
//   onAttach?(entity)          after it joins entity.components (component.entity is already set)
//   onDetach?()                before it leaves (component.entity is still set; cleared to null afterwards)
//   fixedUpdate?(dt)           Engine fixed steps        } run only while the entity is in an EntitySystem, in that
//   update?(dt, alpha)         once per frame            } system's Engine phase. The hooks present are sampled when
//   lateUpdate?(dt, alpha)     once per frame, after all } the component is registered (adding a method later does
//                                                          nothing until it is re-attached)
// Entity: addComponent(c) -> c (moves c from another entity if needed), getComponent(TypeOrName) -> c | null,
// removeComponent(c) -> c | null, components (array, attach order), system (its EntitySystem or null), isEntity,
// destroy(): destroys descendant entities first (also below plain Object3D nodes), detaches every component (last
// attached first), leaves its EntitySystem and its parent. A destroyed entity is plain again and may be reused.
// EntitySystem: add(entity) -> entity (moves it from another system), remove(entity) -> entity, entities (array;
// swap-removed, so order is not stable: iterate backwards when removing in a loop), phases { fixed, update, late }
// (the dense per-phase arrays, read only), and the Engine hooks fixedUpdate / update / lateUpdate / dispose
// (dispose destroys every entity still registered). Joining or leaving a system does not call onAttach/onDetach.
//
// Per-frame cost / safety. Each phase is a dense array holding only components that implement that hook, filled
// when a component is registered (attach to an entity in a system, or system.add of its entity), so the loops call
// a[i].update(dt, alpha) with no `if (c.update)` probe. A component keeps its slot index per phase in __fi / __ui /
// __li (-1 = not in that phase; like Engine's __name / __ms on systems), so removal is O(1) swap-remove.
// Changes during a phase loop (a component destroying its own entity, spawning another, ...) are safe without a
// per-frame snapshot: each loop runs to the length read when it started, so components registered meanwhile are
// appended past it and first run in the next pass of that phase (a component added during fixedUpdate still gets
// this frame's update); a removal during any loop overwrites the slot with a shared no-op placeholder instead of
// moving anything, so a component removed mid-loop is not called again (not even later in the same loop), and the
// outermost loop compacts the arrays in place (stable) when it returns. Nothing allocates.
import * as THREE from 'three';

const noop = () => {};
const NOOP = Object.freeze({ fixedUpdate: noop, update: noop, lateUpdate: noop }); // placeholder for removed slots

export class Entity extends THREE.Object3D {
  /** @param {string} [name] */
  constructor(name) {
    super();
    if (name !== undefined) this.name = name;
    this.isEntity = true;
    /** Attached components, in attach order. Mutate only through addComponent / removeComponent. */
    this.components = [];
    /** @type {EntitySystem|null} */
    this.system = null;
    this.__si = -1; // index in system.entities
  }

  /** Attach a component (detaching it from another entity first). Returns it. */
  addComponent(c) {
    if (c.entity === this) return c;
    if (c.entity && c.entity.isEntity) c.entity.removeComponent(c);
    this.components.push(c);
    c.entity = this;
    if (this.system) this.system._register(c);
    if (c.onAttach) c.onAttach(this);
    return c;
  }

  /**
   * First component that is an instance of `type` (a constructor) or whose `name` / constructor name equals it
   * (a string). Null when none. Constructor names change under minification: prefer the constructor or `c.name`.
   */
  getComponent(type) {
    const cs = this.components;
    if (typeof type === 'function') { for (let i = 0; i < cs.length; i++) if (cs[i] instanceof type) return cs[i]; }
    else for (let i = 0; i < cs.length; i++) { const c = cs[i]; if (c.name === type || (c.constructor && c.constructor.name === type)) return c; }
    return null;
  }

  /** Detach a component: leaves the system phases, then onDetach(), then component.entity = null. Returns it (null if not attached here). */
  removeComponent(c) {
    const cs = this.components, i = cs.indexOf(c);
    if (i < 0) return null;
    for (let j = i + 1; j < cs.length; j++) cs[j - 1] = cs[j];
    cs.length--;
    if (this.system) this.system._unregister(c);
    if (c.onDetach) c.onDetach();
    c.entity = null;
    return c;
  }

  /** Destroy descendant entities, detach every component, leave the EntitySystem and the parent (see header). */
  destroy() {
    destroyChildren(this);
    const cs = this.components;
    while (cs.length) this.removeComponent(cs[cs.length - 1]);
    if (this.system) this.system.remove(this);
    if (this.parent) this.parent.remove(this);
  }
}

// destroy entities below obj, looking through plain Object3D nodes (which stay where they are)
function destroyChildren(obj) {
  const ch = obj.children;
  for (let i = ch.length - 1; i >= 0; i--) {
    const c = ch[i];
    if (c === undefined) continue; // a destroy() removed more than one child
    if (c.isEntity) c.destroy(); else destroyChildren(c);
  }
}

export class EntitySystem {
  constructor() {
    /** Registered entities (swap-removed: order is not stable). */
    this.entities = [];
    /** Dense per-phase component arrays (read only; may hold a no-op placeholder while a phase runs). */
    this.phases = { fixed: [], update: [], late: [] };
    this._depth = 0; this._dirty = false;
  }

  /** Register an entity (and its components' hooks). Returns it. */
  add(entity) {
    if (!entity || !entity.isEntity) throw new TypeError('EntitySystem.add: expected an Entity');
    if (entity.system === this) return entity;
    if (entity.system) entity.system.remove(entity);
    entity.system = this;
    entity.__si = this.entities.push(entity) - 1;
    const cs = entity.components;
    for (let i = 0; i < cs.length; i++) this._register(cs[i]);
    return entity;
  }

  /** Unregister an entity (its components stay attached to it). Returns it. */
  remove(entity) {
    if (entity.system !== this) return entity;
    const cs = entity.components;
    for (let i = 0; i < cs.length; i++) this._unregister(cs[i]);
    const es = this.entities, i = entity.__si, last = es.pop();
    if (last !== entity) { es[i] = last; last.__si = i; }
    entity.system = null; entity.__si = -1;
    return entity;
  }

  /** Engine hook: fixedUpdate(dt) of every registered component that has one. */
  fixedUpdate(dt) {
    const a = this.phases.fixed, n = a.length;
    this._depth++;
    try { for (let i = 0; i < n; i++) a[i].fixedUpdate(dt); } finally { if (--this._depth === 0 && this._dirty) this._compact(); }
  }

  /** Engine hook: update(dt, alpha) of every registered component that has one. */
  update(dt, alpha) {
    const a = this.phases.update, n = a.length;
    this._depth++;
    try { for (let i = 0; i < n; i++) a[i].update(dt, alpha); } finally { if (--this._depth === 0 && this._dirty) this._compact(); }
  }

  /** Engine hook: lateUpdate(dt, alpha) of every registered component that has one. */
  lateUpdate(dt, alpha) {
    const a = this.phases.late, n = a.length;
    this._depth++;
    try { for (let i = 0; i < n; i++) a[i].lateUpdate(dt, alpha); } finally { if (--this._depth === 0 && this._dirty) this._compact(); }
  }

  /** Engine hook: destroy every registered entity. */
  dispose() {
    const es = this.entities;
    while (es.length) es[es.length - 1].destroy();
  }

  _register(c) {
    const p = this.phases;
    c.__fi = typeof c.fixedUpdate === 'function' ? p.fixed.push(c) - 1 : -1;
    c.__ui = typeof c.update === 'function' ? p.update.push(c) - 1 : -1;
    c.__li = typeof c.lateUpdate === 'function' ? p.late.push(c) - 1 : -1;
  }

  _unregister(c) {
    const p = this.phases;
    if (c.__fi >= 0) this._drop(p.fixed, c.__fi, '__fi');
    if (c.__ui >= 0) this._drop(p.update, c.__ui, '__ui');
    if (c.__li >= 0) this._drop(p.late, c.__li, '__li');
    c.__fi = c.__ui = c.__li = -1;
  }

  // remove slot i of a phase array: swap-remove when idle, placeholder + deferred compaction inside a phase loop
  _drop(a, i, key) {
    if (this._depth > 0) { a[i] = NOOP; this._dirty = true; return; }
    const last = a.pop();
    if (i < a.length) { a[i] = last; last[key] = i; }
  }

  _compact() {
    const p = this.phases;
    compact(p.fixed, '__fi'); compact(p.update, '__ui'); compact(p.late, '__li');
    this._dirty = false;
  }
}

// stable in-place removal of placeholders, re-indexing the components that move
function compact(a, key) {
  let k = 0;
  for (let i = 0; i < a.length; i++) {
    const c = a[i];
    if (c === NOOP) continue;
    if (k !== i) { a[k] = c; c[key] = k; }
    k++;
  }
  a.length = k;
}
