// Sandbox scenes on the XPBD dynamics world, plus the instanced visuals every scene shares.
//
//   SCENES[name](ctx) -> { info, focus (Vector3) | target (Object3D), view, poses: { overview(cam), close(cam) },
//                          benchSteps (fixed steps simulated before a bench pose), earlyUpdate?(dt),
//                          fixedUpdate?(dt) (runs BEFORE dyn.step), afterStep?(dt) (after dyn.step and the pools'
//                          capture: the scene's own interpolated meshes), update?(dt, alpha), setInputEnabled?(on),
//                          status?(), dispose() }
//   ctx = { dyn, physics, scene, pools, params, THREE, rand, statics, renderer }   (rand: seeded mulberry32, fresh for
//   every build; statics: what buildStatics returned; renderer: texture anisotropy)
//   buildStatics(name, { physics, scene, renderer }) adds a scene's PhysicsWorld boxes ONCE per page, before the
//   DynamicsClient is created (PhysicsWorld boxes cannot be removed, and a worker client copies them at start-up):
//   bench1000's pit, buoyancy's pool (water.js).
//   SCENE_HZ[name]: the fixed-step rate a scene wants (default 60; the car's controller is tuned at 120).
//   SCENE_PARTICLES[name]: the 0.9.0 scenes (cloth, rope, soft, buoyancy, destructible) and their world's particle
//   capacity (maxParticles; 0 = none). They use the in-thread DynamicsWorld (ctx.dyn.world: particle store and solvers,
//   buoyancy, fracture glue joints), so main.js never starts the worker for them. Their particle objects (Cloth.grid,
//   Rope.chain, SoftBody.lattice), particle solvers (ParticleCollider, Buoyancy) and render/SoftMeshes meshes go
//   through softKit, which captures the meshes after every step, draws them per frame and undoes everything on
//   dispose(), so a rebuild (R, every bench pose) starts from the same store / solver state as the first build.
//
// BodyPools: one InstancedPool per shape type (box, sphere, and one per capsule size: a capsule cannot be scaled
// from a unit one without squashing its caps), per-body palette colours, sleeping bodies dimmed. track(id, ...)
// registers a body (compound parts: one record per part with its local offset); capture() after every fixed step
// keeps the previous and current pose for render interpolation (in-thread); snap(id) after a teleport (setPose) so
// the body does not streak from its old place; sync(alpha) writes the instance matrices once per frame (worker client:
// getPose with ahead = min(maxExtrapolation, time - snapshotTime + alpha * fixedDt), the latest snapshot carried to
// the render time).
//
// Scenes build through a small builder that records every body / joint it adds, so dispose() removes exactly those.
// Determinism: everything random comes from ctx.rand; random rotations are built from normalised random vectors
// (no Math.sin / cos), so a scene is laid out identically on every load. Allocation: the per-frame visuals and the
// per-step drivers allocate nothing; scene builds and deliberate spawn events do (the ragdoll scene adds a doll's
// 11 bodies and 10 joints mid-run, main.js's thrown balls), as does growing a pool's capacity.
import * as THREE from 'three';
import { Shapes, JOINT, Cloth, Rope, SoftBody, ParticleCollider, Buoyancy, FracturedBody, FRACTURE_BREAK_FORCE } from '@vibe/engine/physics';
import { InstancedPool, ClothMesh, RopeMesh, SoftBodyMesh, FractureMesh, makeCanvas, finishTexture } from '@vibe/engine/render';
import { createCar } from './car.js';
import { createPool } from './water.js';

export const PALETTE = [0xe8b04a, 0xd9643a, 0x4f9fd6, 0x6cbf84, 0xc76bb8, 0xe6dccb, 0x8a96a8, 0xf07f6a];
const GREY = 0x9a958c, SKIN = 0xe9c9a8;
const DIM = 0.42;                       // colour factor of sleeping bodies
const Q0 = [0, 0, 0, 1];
const S2 = Math.SQRT1_2;
const BODY_KEYS = ['mass', 'density', 'friction', 'restitution', 'linearDamping', 'angularDamping', 'layer', 'mask', 'userData'];
const SHAPE_OF = { box: (d) => Shapes.box(d[0], d[1], d[2]), sphere: (d) => Shapes.sphere(d[0]), capsule: (d) => Shapes.capsule(d[0], d[1]) };
// collision layer bits used by the scenes (a body collides with another when each one's mask has the other's layer)
const DECOR = 1 << 1;                   // static props a jointed body passes through (rails, axle posts, gantries)
const NO_DECOR = ~DECOR;
// Hinge angle convention (physics/dynamics/joints.js): phi is the right-handed angle of body A relative to body B
// about the axis, 0 in the creation pose. The knee / elbow joints have A = the upper segment, B = the lower one, so a
// flexion of the lower segment by +t about the axis is phi = -t.
const KNEE_LIMITS = [-2.4, 0];          // axis +X (the ragdoll faces +Z): flexion swings the foot back
const ELBOW_LIMITS = [-2.4, 0];         // axis (0, -side, 0): flexion swings the hand forward

const _v = new THREE.Vector3(), _a = new THREE.Vector3(), _b = new THREE.Vector3(), _s = new THREE.Vector3();
const _q = new THREE.Quaternion(), _col = new THREE.Color();
const _wv = new THREE.Vector3(), _wo = new THREE.Vector3(), _ws = new THREE.Vector3(), _wq = new THREE.Quaternion(), _wm = new THREE.Matrix4(); // BodyPools._write

// ------------------------------------------------------------------ instanced visuals
export class BodyPools {
  /**
   * @param {THREE.Scene} scene
   * @param {object} dyn  DynamicsClient (getPose, isAwake)
   * @param {object} [o]  interpolate (default true; false for a worker client), fixedDt (s, worker extrapolation),
   *   textures { box, round }, velocity (InstancedPool({ velocity: true }): previous instance matrices for G-buffer
   *   velocity, default false), onBucket(pool) called for every pool created (now and later: PostFX.trackPool)
   */
  constructor(scene, dyn, o = {}) {
    this.scene = scene; this.dyn = dyn;
    this.interpolate = o.interpolate !== false;
    this.fixedDt = o.fixedDt || 1 / 60;
    this.velocity = !!o.velocity;
    this.onBucket = o.onBucket || null;
    const tex = o.textures || {};
    this.boxMaterial = new THREE.MeshStandardMaterial({ map: tex.box || null, roughness: 0.72, metalness: 0.02 });
    this.roundMaterial = new THREE.MeshStandardMaterial({ map: tex.round || null, roughness: 0.5, metalness: 0.05 });
    this.buckets = [];
    this.box = this._bucket(new THREE.BoxGeometry(1, 1, 1), this.boxMaterial, 'sandbox.boxes');
    this.sphere = this._bucket(new THREE.SphereGeometry(1, 24, 16), this.roundMaterial, 'sandbox.spheres');
    this.capsules = new Map();          // 'r:hh' -> bucket
    this.records = [];
    this.byId = new Map();              // body id -> its records
  }

  _bucket(geometry, material, name) {
    const pool = new InstancedPool(geometry, material, { capacity: 64, colors: true, castShadow: true, cullRadius: 2.5, name, velocity: this.velocity });
    pool.mesh.visible = false;
    this.scene.add(pool.mesh);
    const b = { pool, owners: [] };     // owners[slot] = record
    this.buckets.push(b);
    if (this.onBucket) this.onBucket(pool);
    return b;
  }
  _capsule(r, hh) {
    const key = `${r.toFixed(3)}:${hh.toFixed(3)}`;
    let b = this.capsules.get(key);
    if (!b) { b = this._bucket(new THREE.CapsuleGeometry(r, 2 * hh, 6, 14), this.roundMaterial, `sandbox.capsules ${key}`); this.capsules.set(key, b); }
    return b;
  }

  /**
   * Draw body `id` as a box (dims = half extents), sphere ([r]) or capsule ([r, halfHeight], local Y).
   * @param {number} id
   * @param {'box'|'sphere'|'capsule'} kind
   * @param {number[]} dims
   * @param {number|string} [color]
   * @param {object} [o]  isStatic (never moves, never dimmed), offset [x, y, z] / rotation [x, y, z, w] (part of a compound)
   */
  track(id, kind, dims, color = 0xffffff, o = {}) {
    const bk = kind === 'box' ? this.box : kind === 'sphere' ? this.sphere : this._capsule(dims[0], dims[1]);
    const slot = bk.pool.add();
    if (slot < 0) return null;
    _col.set(color);
    const rec = {
      id, bucket: bk, slot, i: this.records.length, isStatic: !!o.isStatic, awake: -1,
      sx: kind === 'box' ? 2 * dims[0] : kind === 'sphere' ? dims[0] : 1,
      sy: kind === 'box' ? 2 * dims[1] : kind === 'sphere' ? dims[0] : 1,
      sz: kind === 'box' ? 2 * dims[2] : kind === 'sphere' ? dims[0] : 1,
      off: o.offset ? new THREE.Vector3().fromArray(o.offset) : null,
      rot: o.rotation ? new THREE.Quaternion().fromArray(o.rotation) : null,
      r: _col.r, g: _col.g, b: _col.b,
      p0: new THREE.Vector3(), q0: new THREE.Quaternion(), p1: new THREE.Vector3(), q1: new THREE.Quaternion(),
      pos: new THREE.Vector3(), quat: new THREE.Quaternion(), // last rendered body pose
    };
    bk.owners[slot] = rec;
    this.records.push(rec);
    let list = this.byId.get(id);
    if (!list) this.byId.set(id, list = []);
    list.push(rec);
    this.dyn.getPose(id, rec.p1, rec.q1);
    rec.p0.copy(rec.p1); rec.q0.copy(rec.q1); rec.pos.copy(rec.p1); rec.quat.copy(rec.q1);
    this._write(rec);
    bk.pool.setColor(slot, rec.r, rec.g, rec.b);
    return rec;
  }
  /** Stop drawing body `id` (all its parts). */
  untrack(id) {
    const list = this.byId.get(id);
    if (!list) return;
    this.byId.delete(id);
    for (let k = 0; k < list.length; k++) this._remove(list[k]);
  }
  _remove(rec) {
    const bk = rec.bucket, P = bk.pool, O = bk.owners;
    const moved = P.removeAt(rec.slot); // swap-remove: the last instance moved into rec.slot
    if (moved >= 0) { const m = O[moved]; m.slot = rec.slot; O[rec.slot] = m; }
    O.length = P.count;
    const R = this.records, last = R.pop();
    if (last !== rec) { R[rec.i] = last; last.i = rec.i; }
  }
  /** Drop every record (scene reset). */
  clear() {
    for (const bk of this.buckets) { bk.pool.setCount(0); bk.owners.length = 0; bk.pool.mesh.visible = false; }
    this.records.length = 0;
    this.byId.clear();
  }
  get count() { return this.records.length; }

  /** After each fixed step (in-thread client): previous <- current, current <- the solved pose. */
  capture() {
    if (!this.interpolate) return;
    const R = this.records, dyn = this.dyn;
    for (let i = 0; i < R.length; i++) {
      const r = R[i];
      if (r.isStatic) continue;
      r.p0.copy(r.p1); r.q0.copy(r.q1);
      dyn.getPose(r.id, r.p1, r.q1);
    }
  }
  /** After a teleport (setPose) of body `id`: previous = current = its new pose (no interpolation streak). */
  snap(id) {
    const list = this.byId.get(id);
    if (!list) return;
    for (let k = 0; k < list.length; k++) {
      const r = list[k];
      this.dyn.getPose(id, r.p1, r.q1);
      r.p0.copy(r.p1); r.q0.copy(r.q1); r.pos.copy(r.p1); r.quat.copy(r.q1);
    }
  }
  /** Per frame: interpolated (or worker-extrapolated) poses into the instance matrices; sleep dimming; bounds. */
  sync(alpha) {
    const R = this.records, dyn = this.dyn, lerp = this.interpolate;
    // worker client: carry the latest snapshot to the render time (the steps not answered yet + the render alpha)
    const ahead = lerp ? 0 : Math.max(0, Math.min(dyn.maxExtrapolation ?? 0.1, (dyn.time - dyn.snapshotTime) + alpha * this.fixedDt));
    for (let i = 0; i < R.length; i++) {
      const r = R[i];
      if (!r.isStatic) {
        if (lerp) { r.pos.lerpVectors(r.p0, r.p1, alpha); r.quat.slerpQuaternions(r.q0, r.q1, alpha); }
        else dyn.getPose(r.id, r.pos, r.quat, ahead);
        this._write(r);
      }
      const awake = r.isStatic || dyn.isAwake(r.id) ? 1 : 0;
      if (awake !== r.awake) {
        r.awake = awake;
        const k = awake ? 1 : DIM;
        r.bucket.pool.setColor(r.slot, r.r * k, r.g * k, r.b * k);
      }
    }
    for (let i = 0; i < this.buckets.length; i++) {
      const P = this.buckets[i].pool;
      if (P.count > 0) P.updateBounds();
      P.mesh.visible = P.count > 0;
    }
  }
  /** The last rendered pose of body `id` (false when it is not tracked). */
  pose(id, outV, outQ) {
    const l = this.byId.get(id);
    if (!l) return false;
    outV.copy(l[0].pos);
    if (outQ) outQ.copy(l[0].quat);
    return true;
  }
  _write(r) {
    _wv.copy(r.pos); _wq.copy(r.quat);
    if (r.off) _wv.add(_wo.copy(r.off).applyQuaternion(r.quat));
    if (r.rot) _wq.multiply(r.rot);
    _wm.compose(_wv, _wq, _ws.set(r.sx, r.sy, r.sz));
    r.bucket.pool.setMatrix(r.slot, _wm);
  }
}

// ------------------------------------------------------------------ helpers
/** Records the bodies / joints a scene adds (dispose() removes them) and tracks their visuals. */
function builder(ctx) {
  const { dyn, pools } = ctx;
  const bodies = [], joints = [];
  const add = (shape, pos, o) => {
    const d = { type: o.type || 'dynamic', shape, position: pos, quaternion: o.q || Q0 };
    for (const k of BODY_KEYS) if (o[k] !== undefined) d[k] = o[k];
    const id = dyn.addBody(d);
    if (!(id >= 0)) return -1;
    bodies.push(id);
    return id;
  };
  const simple = (kind, dims, pos, o) => {
    const id = add(SHAPE_OF[kind](dims), pos, o);
    if (id >= 0 && o.visible !== false) pools.track(id, kind, dims, o.color ?? 0xffffff, { isStatic: o.type === 'static' });
    return id;
  };
  return {
    box: (hx, hy, hz, pos, o = {}) => simple('box', [hx, hy, hz], pos, o),
    sphere: (r, pos, o = {}) => simple('sphere', [r], pos, o),
    capsule: (r, hh, pos, o = {}) => simple('capsule', [r, hh], pos, o),
    /** parts: [{ kind, dims, position, quaternion, color }] (keep them symmetric about the origin: the centre of mass). */
    compound(parts, pos, o = {}) {
      const shape = Shapes.compound(parts.map((p) => ({ shape: SHAPE_OF[p.kind](p.dims), position: p.position || [0, 0, 0], quaternion: p.quaternion || Q0 })));
      const id = add(shape, pos, o);
      if (id >= 0 && o.visible !== false) {
        for (const p of parts) pools.track(id, p.kind, p.dims, p.color ?? o.color ?? 0xffffff, { offset: p.position, rotation: p.quaternion, isStatic: o.type === 'static' });
      }
      return id;
    },
    joint(desc) {
      const j = dyn.addJoint(desc);
      if (j !== -1 && j != null) joints.push(j);
      return j;
    },
    dispose() {
      for (const j of joints) dyn.removeJoint(j);
      for (const id of bodies) { pools.untrack(id); dyn.removeBody(id); }
      joints.length = 0; bodies.length = 0;
    },
  };
}

/** Line segments updated per frame (springs, ropes). */
function lines(scene, n, color) {
  const pos = new Float32Array(n * 6);
  const geo = new THREE.BufferGeometry();
  const attr = new THREE.BufferAttribute(pos, 3).setUsage(THREE.DynamicDrawUsage);
  geo.setAttribute('position', attr);
  const mat = new THREE.LineBasicMaterial({ color });
  const seg = new THREE.LineSegments(geo, mat);
  seg.frustumCulled = false;
  scene.add(seg);
  return {
    set(i, ax, ay, az, bx, by, bz) { const k = i * 6; pos[k] = ax; pos[k + 1] = ay; pos[k + 2] = az; pos[k + 3] = bx; pos[k + 4] = by; pos[k + 5] = bz; },
    commit() { attr.needsUpdate = true; },
    dispose() { scene.remove(seg); geo.dispose(); mat.dispose(); },
  };
}

function look(cam, px, py, pz, tx, ty, tz) { cam.position.set(px, py, pz); cam.up.set(0, 1, 0); cam.lookAt(tx, ty, tz); }

/**
 * The 0.9.0 parts a scene adds to the in-thread world: particle objects (anything with dispose(): Cloth / Rope /
 * SoftBody), particle solvers (world.addParticleSolver, removed again on dispose) and SoftMeshes (added to the scene,
 * captured after every step, drawn per frame). dispose() undoes it all, meshes first, then the objects and solvers in
 * reverse order of creation.
 */
function softKit(ctx) {
  const world = ctx.dyn.world, scene = ctx.scene;
  if (!world || !world.particles) throw new Error('[sandbox] this scene needs the in-thread dynamics world with particles (SCENE_PARTICLES)');
  const P = world.particles, C = world.constraints, objects = [], solvers = [], meshes = [];
  return {
    world, store: P,
    add(o) { objects.push(o); return o; },
    solver(sv) { world.addParticleSolver(sv); solvers.push(sv); return sv; },
    mesh(m) { scene.add(m.mesh); meshes.push(m); return m; },
    capture() { for (let i = 0; i < meshes.length; i++) meshes[i].capture(); },
    draw(alpha) { for (let i = 0; i < meshes.length; i++) meshes[i].update(alpha); },
    /** HUD line (4 Hz): alive particles and constraints of the whole world. */
    status() {
      let n = 0;
      for (let i = 0, e = P.count; i < e; i++) n += P.alive[i] ? 1 : 0;
      return `${n} particles · ${C.size} constraints`;
    },
    dispose() {
      for (let i = meshes.length - 1; i >= 0; i--) meshes[i].dispose();
      for (let i = objects.length - 1; i >= 0; i--) objects[i].dispose();
      for (let i = solvers.length - 1; i >= 0; i--) world.removeParticleSolver(solvers[i]);
      meshes.length = 0; objects.length = 0; solvers.length = 0;
    },
  };
}
/**
 * A particle rope of rest length `length` hung between the world points a and b (level, |ab| < length): Rope.chain
 * builds it straight along a -> b (`length` long), its first particle pinned at a and its last pinned at the world
 * point b (a WORLD_BODY pin); the particles are then moved onto the parabola through a and b whose arc length is `length`, at equal arc
 * steps (the chords are a hair shorter than the rest length: no stretch, no jolt), close to the hanging rope's rest
 * shape so it starts nearly still. The tethers (stretch 0) keep rest = the distance along the rope, an upper bound.
 */
function slackLine(world, a, b, length, segments, o) {
  const dx = b[0] - a[0], dy = b[1] - a[1], dz = b[2] - a[2], span = Math.sqrt(dx * dx + dy * dy + dz * dz), f = length / span;
  const rope = Rope.chain(world, { ...o, from: a, to: [a[0] + dx * f, a[1] + dy * f, a[2] + dz * f], segments, pins: ['from', { end: 'to', body: -1, anchor: b }] });
  // y(t) = a.y + t dy - 4 sag t (1 - t) along t in [0, 1]; sag by bisection on the sampled arc length
  const M = 2048, arc = new Float64Array(M + 1);
  const curve = (sag) => {
    let px = a[0], py = a[1], pz = a[2];
    for (let i = 1; i <= M; i++) {
      const t = i / M, x = a[0] + dx * t, y = a[1] + dy * t - 4 * sag * t * (1 - t), z = a[2] + dz * t;
      arc[i] = arc[i - 1] + Math.sqrt((x - px) * (x - px) + (y - py) * (y - py) + (z - pz) * (z - pz));
      px = x; py = y; pz = z;
    }
    return arc[M];
  };
  let lo = 0, hi = length;
  for (let it = 0; it < 60; it++) { const mid = (lo + hi) / 2; if (curve(mid) < length) lo = mid; else hi = mid; }
  const sag = (lo + hi) / 2, total = curve(sag), P = world.particles;
  for (let k = 1, i = 1; k <= segments; k++) {
    const want = (total * k) / segments;
    while (i < M && arc[i] < want) i++;
    const t = (i - 1 + (want - arc[i - 1]) / (arc[i] - arc[i - 1])) / M;
    const p = rope.ids[k] * 3;
    P.pos[p] = P.prevPos[p] = a[0] + dx * t;
    P.pos[p + 1] = P.prevPos[p + 1] = a[1] + dy * t - 4 * sag * t * (1 - t);
    P.pos[p + 2] = P.prevPos[p + 2] = a[2] + dz * t;
  }
  return rope;
}
/** n x n checker of two CSS colours with a light seam line (cloth), a fresh CanvasTexture (the scene disposes it). */
function checkerTexture(renderer, a, b, n = 8) {
  const S = 256, cell = S / n, c = makeCanvas(S), g = c.getContext('2d');
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) { g.fillStyle = (x + y) & 1 ? a : b; g.fillRect(x * cell, y * cell, cell, cell); }
  g.fillStyle = 'rgba(255,255,255,0.55)';
  g.fillRect(0, 0, S, 3); g.fillRect(0, S - 3, S, 3); g.fillRect(0, 0, 3, S); g.fillRect(S - 3, 0, 3, S);
  return finishTexture(new THREE.CanvasTexture(c), renderer, true, false);
}
/** Brick courses (1 m texture: 4 courses of 0.5 m bricks, half-brick offset) in greys the piece colours tint. */
function brickTexture(renderer) {
  const S = 256, c = makeCanvas(S), g = c.getContext('2d'), H = S / 4, W = S / 2;
  g.fillStyle = '#8c8680'; g.fillRect(0, 0, S, S); // mortar
  for (let r = 0; r < 4; r++) {
    for (let k = -1; k < 3; k++) {
      const x = k * W + (r & 1 ? W / 2 : 0);
      g.fillStyle = (r + k) & 1 ? '#f2f0ec' : '#e2ded8';
      g.fillRect(x + 3, r * H + 3, W - 6, H - 6);
    }
  }
  return finishTexture(new THREE.CanvasTexture(c), renderer);
}
/** Uniform random unit quaternion (rejection sampling in the unit 4-ball: + - * / sqrt only). */
function randomQuat(rand) {
  for (;;) {
    const x = rand() * 2 - 1, y = rand() * 2 - 1, z = rand() * 2 - 1, w = rand() * 2 - 1, l = x * x + y * y + z * z + w * w;
    if (l > 1e-4 && l <= 1) { const s = 1 / Math.sqrt(l); return [x * s, y * s, z * s, w * s]; }
  }
}
/** Random rotation about +Y (normalised (0, y, 0, w)). */
function randomYaw(rand) {
  const y = rand() * 2 - 1, w = rand() * 2 - 1, l = Math.sqrt(y * y + w * w);
  return l > 1e-6 ? [0, y / l, 0, w / l] : [0, 0, 0, 1];
}
const qArr = (q) => [q.x, q.y, q.z, q.w];

// ------------------------------------------------------------------ static world (once per page)
/** PhysicsWorld boxes + their meshes for scene `name` (bench1000: the 20 m pit; buoyancy: the pool). Call before
 *  creating the client. */
export function buildStatics(name, { physics, scene, renderer }) {
  if (name === 'buoyancy') return createPool({ physics, scene, renderer, center: [0, 0], size: 12, level: 0.8 });
  if (name !== 'bench1000') return null;
  const IN = 10, T = 0.5, H = 2.5;
  const walls = [
    [-IN - T, IN + T, -IN - T, -IN], [-IN - T, IN + T, IN, IN + T], // north, south (minX, maxX, minZ, maxZ)
    [-IN - T, -IN, -IN, IN], [IN, IN + T, -IN, IN],                 // west, east
  ];
  const group = new THREE.Group(); group.name = 'pit';
  const mat = new THREE.MeshStandardMaterial({ color: GREY, roughness: 0.9, metalness: 0 });
  for (const [x0, x1, z0, z1] of walls) {
    physics.addStaticBox(x0, 0, x1, H, z0, z1);
    const m = new THREE.Mesh(new THREE.BoxGeometry(x1 - x0, H, z1 - z0), mat);
    m.position.set((x0 + x1) / 2, H / 2, (z0 + z1) / 2);
    m.castShadow = true; m.receiveShadow = true;
    group.add(m);
  }
  scene.add(group);
  return group;
}

/** Fixed-step rate per scene (Hz); 60 when not listed. */
export const SCENE_HZ = { car: 120, destructible: 120 }; // the fracture glue is tuned at h = 1/480 (120 Hz x 4)
/** The 0.9.0 scenes: their world's particle capacity (DynamicsWorld maxParticles, 0 = none); in-thread only (header). */
export const SCENE_PARTICLES = { cloth: 1720, rope: 128, soft: 1024, buoyancy: 64, destructible: 0 }; // constraints: 8 x this (cloth 13 688)

// ------------------------------------------------------------------ scenes
export const SCENES = {
  /** 10-high pyramid of 0.5 m boxes, a 1 m ball rolling in. */
  stack(ctx) {
    const B = builder(ctx), H = 0.25, GAP = 0.02, ROWS = 10;
    for (let k = 0; k < ROWS; k++) {
      const n = ROWS - k;
      for (let i = 0; i < n; i++) B.box(H, H, H, [(i - (n - 1) / 2) * (2 * H + GAP), H + k * 2 * H, 0], { color: PALETTE[k % PALETTE.length], friction: 0.7 });
    }
    const ball = B.sphere(0.5, [-14, 0.5, 0], { color: 0xf2f2f2, friction: 0.5 });
    if (ball >= 0) ctx.dyn.setVelocity(ball, 7, 0, 0, 0, 0, -14); // rolling (w = v / r about -Z)
    return {
      info: '55 boxes of 0.5 m in a 10-high pyramid; a 1 m ball (524 kg) rolls in from the left.',
      benchSteps: 97, // the ball reaches the pyramid (measured)
      focus: new THREE.Vector3(0, 1.5, 0), view: { distance: 14, elevation: 0.3, azimuth: 0.5 },
      poses: { overview: (cam) => look(cam, 7, 8, 13, 0, 1.6, 0), close: (cam) => look(cam, 2.6, 2.1, 4.2, 0, 1.5, 0) },
      dispose: () => B.dispose(),
    };
  },

  /** Hinge door, motorised paddle wheel, slider piston, distance spring, fixed pair. */
  joints(ctx) {
    const B = builder(ctx), dyn = ctx.dyn, rand = ctx.rand;
    // hinge door (x = -6): hinged to the world at its -X edge, pushed every 2.5 s
    const DX = -6, DW = 0.6, DH = 0.9, DT = 0.04;
    B.box(0.06, 1.05, 0.06, [DX - DW - 0.12, 1.05, 0], { type: 'static', color: GREY });
    const door = B.box(DW, DH, DT, [DX, 1.0, 0], { mass: 40, angularDamping: 0.3, color: PALETTE[0] });
    B.joint({ type: JOINT.HINGE, bodyA: door, bodyB: -1, anchorA: [-DW, 0, 0], anchorB: [DX - DW, 1.0, 0], axisA: [0, 1, 0], axisB: [0, 1, 0], limits: [-1.7, 1.7] });
    // motorised paddle wheel (x = -3), fed with balls dropped from above (8 balls, recycled in turn)
    const WX = -3, WY = 1.5;
    const wheel = B.compound([
      { kind: 'box', dims: [0.85, 0.07, 0.14] },
      { kind: 'box', dims: [0.07, 0.85, 0.14] },
    ], [WX, WY, 0], { density: 400, color: PALETTE[2] });
    B.joint({ type: JOINT.HINGE, bodyA: wheel, bodyB: -1, anchorA: [0, 0, 0], anchorB: [WX, WY, 0], axisA: [0, 0, 1], axisB: [0, 0, 1], motor: { velocity: 2, maxForce: 4000 } });
    B.box(0.05, WY / 2, 0.05, [WX, WY / 2, -0.26], { type: 'static', layer: DECOR, color: GREY }); // axle post
    const balls = [];
    for (let k = 0; k < 8; k++) balls.push(B.sphere(0.12, [WX + 0.25 + (rand() - 0.5) * 0.3, 3.6 + k * 0.45, 0], { density: 600, restitution: 0.3, color: PALETTE[(k + 3) % PALETTE.length] }));
    // slider piston (x = 0): along X between limits, driven by an alternating force
    const piston = B.box(0.3, 0.3, 0.3, [0, 0.75, 0], { mass: 80, mask: NO_DECOR, color: PALETTE[1] });
    B.joint({ type: JOINT.SLIDER, bodyA: piston, bodyB: -1, anchorA: [0, 0, 0], anchorB: [0, 0.75, 0], axisA: [1, 0, 0], axisB: [1, 0, 0], limits: [-1, 1] });
    B.box(1.45, 0.04, 0.1, [0, 0.39, 0], { type: 'static', layer: DECOR, color: GREY }); // rail
    // spring (x = 3): a distance joint with compliance (k = 1300 N/m, 40 kg: ~0.9 Hz), started stretched
    const SA = [3, 3.4, 0], SH = 0.22;
    const spring = B.box(SH, SH, SH, [3.25, 1.25, 0], { mass: 40, linearDamping: 0.05, color: PALETTE[3] });
    B.joint({ type: JOINT.DISTANCE, bodyA: spring, bodyB: -1, anchorA: [0, SH, 0], anchorB: SA, distance: 1.2, compliance: 1 / 1300 });
    B.box(0.05, 1.75, 0.05, [3.85, 1.75, 0], { type: 'static', layer: DECOR, color: GREY });
    B.box(0.45, 0.04, 0.04, [3.42, 3.46, 0], { type: 'static', layer: DECOR, color: GREY });
    const springLine = lines(ctx.scene, 1, 0x2a2d33);
    // fixed pair (x = 6): two welded boxes dropped tilted
    const qp = new THREE.Quaternion(0.3, 0, 0.5, 1).normalize(), FC = new THREE.Vector3(6, 3, 0);
    const fa = _a.set(-0.255, 0, 0).applyQuaternion(qp).add(FC).toArray(), fb = _b.set(0.255, 0, 0).applyQuaternion(qp).add(FC).toArray();
    const pa = B.box(0.25, 0.25, 0.25, fa, { q: qArr(qp), color: PALETTE[4] });
    const pb = B.box(0.25, 0.25, 0.25, fb, { q: qArr(qp), color: PALETTE[7] });
    B.joint({ type: JOINT.FIXED, bodyA: pa, bodyB: pb, anchorA: [0.255, 0, 0], anchorB: [-0.255, 0, 0] });

    let t = 0, nextDoor = 0.6, doorSign = 1, nextPiston = 0, pistonSign = -1, nextBall = 1.5, ball = 0;
    return {
      info: 'Hinge door (pushed every 2.5 s) · motorised paddle wheel fed with balls · slider piston · distance spring (compliance) · fixed pair.',
      benchSteps: 120, // 2 s: door swung, wheel spinning with balls on it, spring stretched, pair landed
      focus: new THREE.Vector3(0, 1.3, 0), view: { distance: 13, elevation: 0.32, azimuth: 0 },
      poses: { overview: (cam) => look(cam, 0, 6.5, 12.5, 0, 1.3, 0), close: (cam) => look(cam, -4.4, 2.4, 4.6, -4.3, 1.2, 0) },
      fixedUpdate(dt) {
        t += dt;
        if (t >= nextDoor && door >= 0) { // impulse at the free edge along the door's normal
          nextDoor += 2.5;
          dyn.getPose(door, _v, _q);
          _a.set(DW, 0, 0).applyQuaternion(_q).add(_v); _b.set(0, 0, doorSign * 40).applyQuaternion(_q);
          dyn.applyImpulse(door, _a.x, _a.y, _a.z, _b.x, _b.y, _b.z);
          doorSign = -doorSign;
        }
        if (t >= nextPiston) { nextPiston += 1.25; pistonSign = -pistonSign; }
        if (piston >= 0) dyn.applyForce(piston, pistonSign * 1800, 0, 0);
        if (t >= nextBall) { // recycle the next ball: back above the wheel, at rest
          nextBall += 0.9;
          const id = balls[ball]; ball = (ball + 1) % balls.length;
          if (id >= 0) { dyn.setPose(id, WX + 0.25 + (rand() - 0.5) * 0.3, 3.6, (rand() - 0.5) * 0.1, 0, 0, 0, 1); dyn.setVelocity(id, 0, 0, 0, 0, 0, 0); ctx.pools.snap(id); }
        }
      },
      update() {
        if (spring < 0 || !ctx.pools.pose(spring, _v, _q)) return;
        _a.set(0, SH, 0).applyQuaternion(_q).add(_v);
        springLine.set(0, SA[0], SA[1], SA[2], _a.x, _a.y, _a.z);
        springLine.commit();
      },
      dispose() { springLine.dispose(); B.dispose(); },
    };
  },

  /** 20 ball-jointed capsule links and a 20-segment rope, released horizontally. */
  chain(ctx) {
    const B = builder(ctx), LINK = 1 << 2, NO_LINK = ~LINK; // links / rope beads / gantry: no collisions among them
    const Y = 8, R = 0.05, HH = 0.1, L = 2 * (HH + R);
    // chain: hangs from CA, laid out along -X (link local +Y -> world -X: 90 deg about +Z)
    const CA = [-0.5, Y, -0.6], QL = [0, 0, S2, S2];
    let prev = -1;
    for (let i = 0; i < 20; i++) {
      const id = B.capsule(R, HH, [CA[0] - L * (i + 0.5), Y, CA[2]], { q: QL, density: 4000, layer: LINK, mask: NO_LINK, color: i & 1 ? PALETTE[6] : PALETTE[0] });
      B.joint({ type: JOINT.BALL, bodyA: id, bodyB: prev, anchorA: [0, -L / 2, 0], anchorB: prev < 0 ? CA : [0, L / 2, 0] });
      prev = id;
    }
    // rope: 20 beads on distance joints (rope: they only pull), a heavy ball at the end, laid out along +X
    const RA = [0.5, Y, 0.6], SEG = 0.3, beads = [];
    prev = -1;
    for (let k = 0; k < 20; k++) {
      const last = k === 19;
      const id = B.sphere(last ? 0.2 : 0.06, [RA[0] + SEG * (k + 1), Y, RA[2]], { density: last ? 3000 : 1500, layer: LINK, mask: NO_LINK, color: last ? PALETTE[1] : PALETTE[5] });
      B.joint({ type: JOINT.DISTANCE, rope: true, distance: SEG, bodyA: id, bodyB: prev, anchorA: [0, 0, 0], anchorB: prev < 0 ? RA : [0, 0, 0] });
      beads.push(id);
      prev = id;
    }
    // gantry (static, on the link layer: the chain and rope pass it, thrown balls hit it)
    B.box(3.4, 0.08, 0.9, [0, Y + 0.1, 0], { type: 'static', layer: LINK, color: GREY });
    for (const s of [-1, 1]) B.box(0.1, (Y + 0.18) / 2, 0.1, [s * 3.5, (Y + 0.18) / 2, 0], { type: 'static', layer: LINK, color: GREY });
    const rope = lines(ctx.scene, 20, 0x2a2d33);
    return {
      info: '20 ball-jointed capsule links and a rope of 20 distance joints (rope mode) with a heavy end, released horizontally.',
      benchSteps: 75, // 1.25 s: both mid-swing
      focus: new THREE.Vector3(0, 5, 0), view: { distance: 15, elevation: 0.12, azimuth: 0 },
      poses: { overview: (cam) => look(cam, 0, 7.5, 15, 0, 5, 0), close: (cam) => look(cam, -3.5, 6.5, 5, -1.5, 6, -0.6) },
      update() {
        let ax = RA[0], ay = RA[1], az = RA[2];
        for (let k = 0; k < beads.length; k++) {
          if (beads[k] < 0 || !ctx.pools.pose(beads[k], _v)) continue;
          rope.set(k, ax, ay, az, _v.x, _v.y, _v.z);
          ax = _v.x; ay = _v.y; az = _v.z;
        }
        rope.commit();
      },
      dispose() { rope.dispose(); B.dispose(); },
    };
  },

  /** 5 capsule ragdolls (hinge elbows / knees, d6 shoulders / hips / spine / neck) dropped from 3 m onto a bench. */
  ragdoll(ctx) {
    const B = builder(ctx), rand = ctx.rand, COUNT = 5, EVERY = 0.9;
    B.box(1.3, 0.3, 0.45, [0, 0.3, 0], { type: 'static', color: GREY }); // bench
    // one doll: random yaw, a tilt of up to ~60 deg about X (normalised vectors, no trig), lowest point at 3 m
    const drop = (n) => {
      const T = new THREE.Quaternion().fromArray(randomYaw(rand)).multiply(new THREE.Quaternion((rand() - 0.5) * 1.2, 0, 0, 1).normalize());
      let minY = Infinity;
      for (const p of RAGDOLL.parts) minY = Math.min(minY, _v.fromArray(p.pos).sub(RAGDOLL.pivot).applyQuaternion(T).y - p.extent);
      const base = new THREE.Vector3((rand() - 0.5) * 1.0, 3 - minY, (rand() - 0.5) * 0.5);
      const ids = {};
      for (const p of RAGDOLL.parts) {
        const pos = _v.fromArray(p.pos).sub(RAGDOLL.pivot).applyQuaternion(T).add(base).toArray();
        const q = qArr(T.clone().multiply(_q.fromArray(p.quat)));
        const o = { q, friction: 0.8, color: p.skin ? SKIN : PALETTE[n % PALETTE.length] };
        ids[p.name] = p.kind === 'sphere' ? B.sphere(p.dims[0], pos, o) : B.capsule(p.dims[0], p.dims[1], pos, o);
      }
      for (const j of RAGDOLL.joints) if (ids[j.a] >= 0 && ids[j.b] >= 0) B.joint({ ...j.desc, bodyA: ids[j.a], bodyB: ids[j.b] });
    };
    drop(0);
    let t = 0, dropped = 1;
    return {
      info: '5 ragdolls of 11 capsules (hinge elbows and knees; d6 shoulders, hips, spine and neck with swing / twist limits) dropped from 3 m onto a bench, one every 0.9 s.',
      benchSteps: 150, // 2.5 s: two dolls down, the third falling
      focus: new THREE.Vector3(0, 0.8, 0), view: { distance: 9, elevation: 0.38, azimuth: 0.6 },
      poses: { overview: (cam) => look(cam, 5.5, 5.5, 7.5, 0, 0.8, 0), close: (cam) => look(cam, 2.2, 1.6, 2.8, 0, 0.6, 0) },
      fixedUpdate(dt) { // the previous doll has landed (3 m takes 0.78 s) before the next one appears above it
        t += dt;
        if (dropped < COUNT && t >= dropped * EVERY) drop(dropped++);
      },
      dispose: () => B.dispose(),
    };
  },

  /** VehicleController car (kinematic box in the dynamics world) among 30 crates. */
  car(ctx) {
    const B = builder(ctx), rand = ctx.rand;
    const crate = (h, x, y, z, q, color) => B.box(h, h, h, [x, y, z], { q, density: 120, friction: 0.6, color });
    const C = 0.45, P = 0.92;
    for (let k = 0; k < 3; k++) for (let i = 0; i < 3 - k; i++) crate(C, (i - (2 - k) / 2) * P, C + k * 2 * C, -14, Q0, PALETTE[k]);   // 3-2-1 pyramid
    for (let k = 0; k < 3; k++) for (let i = 0; i < 4; i++) crate(C, (i - 1.5) * P, C + k * 2 * C, -28, Q0, PALETTE[3 + (k & 1)]); // 4 x 3 wall
    const placed = [];
    for (let tries = 0; placed.length < 12 * 2 && tries < 400; tries++) {                                                          // 12 scattered
      const x = (rand() - 0.5) * 32, z = 2 - rand() * 36, h = 0.3 + rand() * 0.2, q = randomYaw(rand), c = PALETTE[Math.floor(rand() * PALETTE.length)];
      if ((Math.abs(x) < 3 && z > -10) || (Math.abs(x) < 2.5 && Math.abs(z + 14) < 2.2) || (Math.abs(x) < 3 && Math.abs(z + 28) < 2)) continue;
      let ok = true;
      for (let i = 0; i < placed.length; i += 2) { const dx = placed[i] - x, dz = placed[i + 1] - z; if (dx * dx + dz * dz < 2.2) { ok = false; break; } }
      if (!ok) continue;
      placed.push(x, z);
      crate(h, x, h, z, q, c);
    }
    const solid = ctx.params.get('solid') !== '0'; // ?solid=0: bulldozer
    const car = createCar(ctx, { position: [0, 0, 6], heading: 0, solidCrates: solid });
    return {
      info: solid
        ? 'Roadster VehicleController (W S A D, X handbrake) as a kinematic box among 30 crates; the crates are solid for the car, so it stops at them.'
        : 'Roadster VehicleController (W S A D, X handbrake) as a kinematic box among 30 crates (?solid=0: the car ignores them and its box bulldozes them).',
      benchSteps: 60, // 0.5 s at 120 Hz: crates settled, the car at rest (no input)
      target: car.object3d, view: { distance: 10, elevation: 0.28, azimuth: 0, offsetY: 1 },
      car,
      poses: {
        overview: (cam) => look(cam, 0, 24, 18, 0, 0, -12),
        close: (cam) => { const p = car.object3d.position; look(cam, p.x + 3.5, p.y + 2.2, p.z + 7, p.x, p.y + 0.8, p.z - 2); },
      },
      earlyUpdate: car.earlyUpdate, fixedUpdate: car.fixedUpdate, update: car.update, setInputEnabled: car.setInputEnabled,
      status: () => `${Math.round(car.vc.speed * 3.6)} km/h`,
      dispose() { car.dispose(); B.dispose(); },
    };
  },

  /** ?bodies=N (default 1000) mixed boxes / spheres / capsules dropped into the pit (buildStatics): the measurement scene. */
  bench1000(ctx) {
    const B = builder(ctx), rand = ctx.rand;
    const N = Math.max(1, Math.min(60000, (+ctx.params.get('bodies') || 1000) | 0));
    const G = 12, PITCH = 1.5, Y0 = 3, CAPS = [[0.18, 0.25], [0.22, 0.3], [0.15, 0.35]];
    for (let i = 0; i < N; i++) {
      const lay = Math.floor(i / (G * G)), j = i % (G * G), gx = j % G, gz = Math.floor(j / G);
      const pos = [(gx - (G - 1) / 2) * PITCH + (rand() - 0.5) * 0.16, Y0 + lay * PITCH + rand() * 0.2, (gz - (G - 1) / 2) * PITCH + (rand() - 0.5) * 0.16];
      const q = randomQuat(rand), color = PALETTE[Math.floor(rand() * PALETTE.length)], t = rand();
      if (t < 0.4) B.box(0.18 + rand() * 0.17, 0.18 + rand() * 0.17, 0.18 + rand() * 0.17, pos, { q, color });
      else if (t < 0.75) B.sphere(0.22 + rand() * 0.18, pos, { q, color });
      else { const c = CAPS[Math.floor(rand() * CAPS.length)]; B.capsule(c[0], c[1], pos, { q, color }); }
    }
    return {
      info: `${N} mixed boxes, spheres and capsules dropped into a 20 m pit (static walls): the measurement scene.`,
      benchSteps: 90, // 1.5 s: the layers raining into the pit (most contacts)
      focus: new THREE.Vector3(0, 1, 0), view: { distance: 30, elevation: 0.62, azimuth: 0.35 },
      poses: { overview: (cam) => look(cam, 0, 24, 25, 0, 0, 0), close: (cam) => look(cam, 7.5, 3.6, 9.5, 2, 0.6, 1.5) },
      dispose: () => B.dispose(),
    };
  },

  // ---------------------------------------------------------------- 0.9.0: particles, soft bodies, buoyancy, fracture
  // (in-thread world: SCENE_PARTICLES; the particle objects and their meshes go through softKit)

  /** 40 x 40 cloth pinned at two corners, draped over a box and a ball. */
  cloth(ctx) {
    const B = builder(ctx), K = softKit(ctx), world = K.world;
    B.box(0.5, 0.5, 0.5, [-0.25, 0.5, -0.1], { type: 'static', color: PALETTE[2] }); // 1 m box
    B.sphere(0.3, [0.72, 0.3, 0.05], { type: 'static', color: PALETTE[1] });          // 0.6 m ball beside it
    K.solver(new ParticleCollider(world, { friction: 0.5, margin: 0.005 }));
    // laid flat at 1.7 m (grid +Y -> world -Z: the grid's top row is the far edge, behind the box), its far corners
    // pinned; it falls onto the box and the ball and hangs down their sides
    const cloth = K.add(Cloth.grid(world, {
      nx: 40, ny: 40, spacing: 0.05, position: [0, 1.7, 0], quaternion: [-S2, 0, 0, S2], mass: 1.2,
      stretch: 0, shear: 1e-6, bend: 2e-3, radius: 0.022, pins: ['top-left', 'top-right'],
    }));
    const tex = checkerTexture(ctx.renderer, '#d9643a', '#e6dccb');
    const mat = new THREE.MeshStandardMaterial({ map: tex, side: THREE.DoubleSide, roughness: 0.82, metalness: 0 });
    K.mesh(new ClothMesh(K.store, cloth, mat));
    return {
      info: 'A 40 x 40 particle cloth (1.95 m, 1.2 kg: XPBD stretch / shear / bend constraints + tethers) pinned at two corners, falling onto a 1 m box and a 0.6 m ball (two-way particle collider).',
      benchSteps: 240, // 4 s: draped over the box and the ball (still swaying by a few cm/s)
      focus: new THREE.Vector3(0, 0.8, 0), view: { distance: 6, elevation: 0.32, azimuth: 0.55 },
      poses: { overview: (cam) => look(cam, 3.4, 2.9, 4.2, 0, 0.8, 0), close: (cam) => look(cam, 1.9, 1.35, 1.9, 0.3, 0.65, 0.05) },
      afterStep: () => K.capture(),
      update: (dt, alpha) => K.draw(alpha),
      status: () => K.status(),
      dispose() { K.dispose(); mat.dispose(); tex.dispose(); B.dispose(); },
    };
  },

  /** A pendulum rope with a 5 kg ball, a rope bridge between two posts catching two crates, a rope on a turning wheel. */
  rope(ctx) {
    const B = builder(ctx), K = softKit(ctx), world = K.world;
    K.solver(new ParticleCollider(world, { friction: 0.6, margin: 0.005 }));
    const mat = new THREE.MeshStandardMaterial({ color: 0xcdb487, roughness: 0.9, metalness: 0 });
    const tube = (r) => K.mesh(new RopeMesh(K.store, r, { radius: 0.035, radialSegments: 7, material: mat }));
    // 1. pendulum (x = -4.5): pinned under a gantry at 4 m (beam and post behind the swing plane z = 0), laid out
    //    horizontally, a 5 kg ball tied to its end
    const PA = [-4.5, 4, 0], BR = 0.15;
    B.box(0.06, 0.05, 0.33, [PA[0], PA[1] + 0.1, -0.27], { type: 'static', color: GREY });
    B.box(0.06, (PA[1] + 0.15) / 2, 0.06, [PA[0], (PA[1] + 0.15) / 2, -0.6], { type: 'static', color: GREY });
    const ball = B.sphere(BR, [PA[0] + 2 + BR + 0.04, PA[1], 0], { mass: 5, color: PALETTE[1] });
    // the ball is a body pin of the 'to' end: the builder tethers every particle to the pinned 'from' end and adds a
    // body tether from the pin to the ball, so the 5 kg ball cannot drag the 0.8 kg rope longer (Gauss-Seidel would
    // ping-pong the light last particle between the ball and its tether)
    const r1 = K.add(Rope.chain(world, {
      from: PA, to: [PA[0] + 2, PA[1], 0], segments: 20, mass: 0.8, radius: 0.04, stretch: 0, bend: 0.5,
      pins: ball >= 0 ? ['from', { end: 'to', body: ball, anchor: [-(BR + 0.04), 0, 0] }] : ['from'],
    }));
    tube(r1);
    // 2. a rope bridge (x = -1.5 .. 1.5): two slack lines of 4 kg (3.2 m over the 3 m between two posts) tied together
    //    by rungs (distance constraints every 3rd particle: on one line, or two loose ones, a box rolls off or slips
    //    through), two 2 kg crates dropped onto it from 0.3 / 0.6 m (a 4 m/s landing bounces them off: the lines
    //    are only 0.13 kg per particle)
    const LY = 1.6, LX = 1.5, LZ = 0.13, LN = 30;
    for (const s of [-1, 1]) B.box(0.08, 0.85, LZ + 0.08, [s * (LX + 0.08), 0.85, 0], { type: 'static', color: GREY });
    const lines2 = [-LZ, LZ].map((z) => K.add(slackLine(world, [-LX, LY, z], [LX, LY, z], 3.2, LN, { mass: 4, radius: 0.04, stretch: 0, bend: 0.2 })));
    for (const l of lines2) tube(l);
    const C = world.constraints, X = K.store.pos, rungIds = [], rungs = [];
    for (let k = 3; k < LN; k += 3) {
      const a = lines2[0].ids[k], b = lines2[1].ids[k];
      rungs.push(C.addDistance(a, b, Math.abs(X[b * 3 + 2] - X[a * 3 + 2]), 0));
      rungIds.push(a, b);
    }
    K.add({ dispose: () => { for (const c of rungs) C.remove(c); } });
    K.mesh(new RopeMesh(K.store, { ids: Int32Array.from(rungIds), count: rungIds.length, length: rungs.length * 2 * LZ }, { radius: 0.025, radialSegments: 6, material: mat, pairs: true, joints: false }));
    for (let k = 0; k < 2; k++) B.box(0.2, 0.15, 0.24, [(k - 0.5) * 0.7, 1.6 + k * 0.3, (ctx.rand() - 0.5) * 0.04], { q: randomYaw(ctx.rand), mass: 2, friction: 0.7, color: PALETTE[(k + 2) % PALETTE.length] });
    // 3. a rope on a motorised wheel (x = 4.5): tied to the end of one arm, hanging free
    const WX = 4.5, WY = 3.2, ARM = 0.85;
    const wheel = B.compound([{ kind: 'box', dims: [ARM, 0.07, 0.14] }, { kind: 'box', dims: [0.07, ARM, 0.14] }], [WX, WY, 0], { density: 400, color: PALETTE[2] });
    B.joint({ type: JOINT.HINGE, bodyA: wheel, bodyB: -1, anchorA: [0, 0, 0], anchorB: [WX, WY, 0], axisA: [0, 0, 1], axisB: [0, 0, 1], motor: { velocity: 0.8, maxForce: 4000 } });
    B.box(0.05, WY / 2, 0.05, [WX, WY / 2, -0.26], { type: 'static', layer: DECOR, color: GREY }); // axle post
    const r3 = K.add(Rope.chain(world, {
      from: [WX + ARM, WY, 0.2], to: [WX + ARM, WY - 1.6, 0.2], segments: 16, mass: 0.3, radius: 0.04, stretch: 0, bend: 0.5,
      pins: [{ end: 'from', body: wheel, anchor: [ARM, 0, 0.2] }],
    }));
    tube(r3);
    return {
      info: 'Particle ropes (XPBD distance + bending + tethers, two-way particle collider): a pendulum with a 5 kg ball tied to its end, a rope bridge (two slack lines and rungs) catching two crates, a rope tied to a motorised wheel.',
      benchSteps: 180, // 3 s: the pendulum swinging back, the crates resting on the bridge, the wheel turned 2.4 rad
      focus: new THREE.Vector3(0, 2, 0), view: { distance: 13, elevation: 0.18, azimuth: 0 },
      poses: { overview: (cam) => look(cam, 0, 4.2, 12.5, 0, 2, 0), close: (cam) => look(cam, 1.4, 2.4, 3.6, 0, 1.4, 0) },
      afterStep: () => K.capture(),
      update: (dt, alpha) => K.draw(alpha),
      status: () => K.status(),
      dispose() { K.dispose(); mat.dispose(); B.dispose(); },
    };
  },

  /** Two soft cubes and a soft ball dropped onto the ground and a wedge; a rigid box lands on one cube. */
  soft(ctx) {
    const B = builder(ctx), K = softKit(ctx), world = K.world;
    // wedge: two static slabs tilted +-30 deg about Z leaning on each other, ridge at ~0.8 m along Z at x = 1.6
    const WX = 1.6, S15 = 0.25881904510252074, C15 = 0.9659258262890683; // sin / cos 15 deg (half of 30)
    for (const s of [-1, 1]) B.box(0.75, 0.08, 0.7, [WX - s * 0.6, 0.36, 0], { type: 'static', q: [0, 0, s * S15, C15], color: GREY });
    const jelly = (o) => K.add(SoftBody.lattice(world, { shape: 'box', size: [0.8, 0.8, 0.8], resolution: 6, mass: 30, edge: 2e-4, volume: 0, ...o }));
    const c1 = jelly({ position: [-1.7, 0.75, 0] });                              // onto the ground
    const c2 = jelly({ position: [0.85, 1.55, 0.05], quaternion: [0, 0, S15, C15] }); // flat onto the wedge's left slope
    const sp = K.add(SoftBody.lattice(world, { shape: 'sphere', radius: 0.5, resolution: 6, position: [WX + 0.12, 2.2, 0.1], mass: 15, edge: 1e-4, volume: 0, pressure: 1e-7 }));
    // the collider after the lattices (their tet-volume solver registers with the first one): the contacts run last in
    // every sweep, so the volume pass cannot push resting particles back into the floor or the slope
    K.solver(new ParticleCollider(world, { friction: 0.6, margin: 0.005 }));
    const m1 = new THREE.MeshStandardMaterial({ color: PALETTE[3], roughness: 0.35, metalness: 0 });
    const m2 = new THREE.MeshStandardMaterial({ color: PALETTE[4], roughness: 0.35, metalness: 0 });
    const m3 = new THREE.MeshStandardMaterial({ color: PALETTE[0], roughness: 0.3, metalness: 0 });
    K.mesh(new SoftBodyMesh(K.store, c1, m1));
    K.mesh(new SoftBodyMesh(K.store, c2, m2));
    const pr = K.store.radius[sp.ids[0]];
    const ballGeo = new THREE.SphereGeometry(0.5 + pr, 32, 18);
    K.mesh(new SoftBodyMesh(K.store, sp, m3, { geometry: ballGeo }));
    // a 0.6 x 0.24 x 0.6 m rigid box (35 kg) dropped onto the first cube (a smaller, heavier box sinks into the
    // particles under it, tilts and slides off); a curb stops the ball rolling away off the ridge
    B.box(0.3, 0.12, 0.3, [-1.7, 1.5, 0.02], { q: randomYaw(ctx.rand), density: 400, friction: 0.6, color: PALETTE[6] });
    B.box(0.12, 0.2, 1.4, [3.9, 0.2, 0], { type: 'static', color: GREY });
    return {
      info: 'Tetrahedral soft bodies (XPBD edge + tet-volume constraints, two-way particle collider): two 30 kg 0.8 m cubes (6 x 6 x 6 cells), one sliding down a wedge, and a 1 m ball (voxel lattice + pressure, a skinned round mesh) rolling off its ridge; a 35 kg rigid box lands on the left cube.',
      benchSteps: 200, // 3.3 s: the box resting on the dented cube, the second cube off the slope, the ball at the curb
      focus: new THREE.Vector3(0, 0.6, 0), view: { distance: 7.5, elevation: 0.3, azimuth: 0.4 },
      poses: { overview: (cam) => look(cam, 3.2, 3.4, 6.4, 0, 0.5, -0.3), close: (cam) => look(cam, -0.4, 1.3, 2.4, -1.6, 0.5, 0) },
      afterStep: () => K.capture(),
      update: (dt, alpha) => K.draw(alpha),
      status: () => K.status(),
      dispose() { K.dispose(); m1.dispose(); m2.dispose(); m3.dispose(); ballGeo.dispose(); B.dispose(); },
    };
  },

  /** Floating, neutral and sinking bodies and a rope in the pool (buildStatics: water.js) on the world's Gerstner swell. */
  buoyancy(ctx) {
    const B = builder(ctx), K = softKit(ctx), world = K.world, rand = ctx.rand, pool = ctx.statics;
    if (!pool || !pool.fluid) throw new Error('[sandbox] buoyancy: no pool (buildStatics)');
    const [cx, cz] = pool.center;
    B.box(0.3, 0.3, 0.3, [cx - 3, 1.7, cz - 2], { q: randomYaw(rand), density: 500, color: PALETTE[0] });      // floats half out
    B.box(0.25, 0.25, 0.25, [cx - 0.8, 1.8, cz + 2.2], { q: randomYaw(rand), density: 950, color: PALETTE[3] }); // barely floats
    B.sphere(0.25, [cx + 1.6, 1.9, cz - 2.6], { density: 2000, color: PALETTE[6] });                            // sinks
    B.box(1, 0.1, 0.2, [cx + 2.4, 1.3, cz + 1.4], { q: randomYaw(rand), density: 400, color: PALETTE[1] });     // plank
    K.solver(new ParticleCollider(world, { friction: 0.4, margin: 0.005 }));
    const rope = K.add(Rope.chain(world, { from: [cx - 4.2, 1.2, cz + 3.6], to: [cx - 1.4, 1.2, cz + 4.4], segments: 20, mass: 1.5, radius: 0.05, stretch: 0, bend: 0.3, pins: [] }));
    const mat = new THREE.MeshStandardMaterial({ color: 0xe8b04a, roughness: 0.8, metalness: 0 });
    K.mesh(new RopeMesh(K.store, rope, { radius: 0.05, radialSegments: 7, material: mat }));
    // every dynamic body (thrown balls too) and every free particle floats; heightAt is the pool's (NO_FLUID outside)
    // the pool clock has one writer (fixedUpdate: setTime before every step), so no advance of Buoyancy's own
    const buoy = K.solver(new Buoyancy(world, { fluid: { ...pool.fluid, drag: 1.2, angularDrag: 0.6, time: 0 }, bodies: true, particles: true, samples: 4, advanceTime: false }));
    let n = 0, h = 0; // pool clock: t = n * h (fixed steps since the build, the fixed dt)
    return {
      info: 'Buoyancy on the world Water swell (the pool surface and the physics share waveHeight): a 500 kg/m³ crate floats, a 950 kg/m³ one barely, a 2000 kg/m³ ball sinks, a plank rides the waves, a 20-segment rope floats. (The swell uses Math.sin / hypot: bit-identical within one JS engine, not across browsers.)',
      benchSteps: 300, // 5 s: bodies floating at their depths, the ball on the floor
      focus: new THREE.Vector3(cx, 0.8, cz), view: { distance: 15, elevation: 0.42, azimuth: 0.6 },
      poses: { overview: (cam) => look(cam, cx + 8.5, 6.5, cz + 10, cx, 0.6, cz), close: (cam) => look(cam, cx + 4.6, 1.9, cz + 5.2, cx + 1.4, 0.8, cz + 1.2) },
      fixedUpdate(dt) { // step k (1-based) runs from (k - 1) h to k h and floats the bodies on the waves at (k - 1) h
        h = dt; pool.time = n * dt; n++;
        buoy.setTime(pool.time);
      },
      afterStep: () => K.capture(),
      // the surface is drawn at the render time of the interpolated bodies, (k - 1 + alpha) h after k steps: one step
      // (h) ahead of the heights that step k sampled (beginStep at (k - 1) h); invisible at 60 Hz, noted, not hidden
      update(dt, alpha) {
        pool.setTime(n > 0 ? (n - 1 + alpha) * h : 0);
        K.draw(alpha);
      },
      status: () => `${K.status()} · t ${pool.time.toFixed(1)} s`,
      dispose() { K.dispose(); mat.dispose(); B.dispose(); pool.time = 0; pool.setTime(0); },
    };
  },

  /** A 40-piece pre-fractured wall glued by breakable fixed joints; a 30 kg ball hits it 1.8 s in (Space throws more). */
  destructible(ctx) {
    const B = builder(ctx), world = ctx.dyn.world, rand = ctx.rand;
    if (!world) throw new Error('[sandbox] destructible: needs the in-thread world (SCENE_PARTICLES)');
    // 120 Hz (SCENE_HZ): the glue (FracturedBody defaults: FRACTURE_BREAK_FORCE, FRACTURE_COMPLIANCE) is tuned at
    // h = 1/480; at 60 Hz the same impulse reads as half the force. fb.impactPlugin (a world plugin) frees the struck
    // piece and the glue within IMPACT.radius of any contact closing faster than IMPACT.speed BEFORE the contact solve,
    // and re-collides the freed pieces in that same step, so a hit (the scene's ball, main.js's thrown ones) opens a
    // hole instead of being spread through the glued wall. The scene's shot carries a thrown ball's momentum (30 kg at
    // BALL_SPEED ~ 600 N s) as 60 kg at 10 m/s: no CCD, so a ball first touches up to one step's travel deep, and at
    // 10 m/s that is 8 cm per 120 Hz step instead of 17-18 cm; it is launched at step FIRE (1.6 s) and aimed so that its
    // front is 5 mm from the wall face after FLY steps (no damping: the travel is exact). The pieces are cut finer around
    // the point it hits (fracturePieces `impact`, wall frame, 8 cm inside the front face).
    // The cut pattern decides whether this hit throws pieces (headless at benchSteps: 0-2 pieces moved > 0.3 m on the
    // patterns of page seeds 1-8 and of fracture seeds 41-104, 3 on fracture seed 35): the wall uses fracture seed
    // FSEED = 35, chosen for that (?fseed=N for another pattern), not the page seed.
    const FIRE = 192, FLY = 36, BALL = 0.25, MASS = 60, SPEED = 10, HIT = [0.15, 1.05], IMPACT = { speed: 3, radius: 0.6 };
    const FSEED = (+ctx.params.get('fseed') | 0) || 35;
    const fb = new FracturedBody(world, {
      shape: Shapes.box(2, 1, 0.25), position: [0, 1, 0], pieces: 40, seed: FSEED, density: 600, friction: 0.6,
      impact: [HIT[0], HIT[1] - 1, 0.17],
    });
    const plugin = world.addParticleSolver(fb.impactPlugin(IMPACT));
    // one brick texture tinted per piece (a few close stone colours), cut faces in a plain interior material
    const bricks = brickTexture(ctx.renderer);
    const mats = [0xc9b9a6, 0xb9a48d, 0xd8cab6, 0xa99582].map((c) => new THREE.MeshStandardMaterial({ map: bricks, color: c, roughness: 0.85, metalness: 0 }));
    const inner = new THREE.MeshStandardMaterial({ color: 0x8f8a83, roughness: 0.95, metalness: 0 });
    const fm = new FractureMesh(fb, world, mats[0], { interpolate: true, interiorMaterial: inner });
    const pairs = mats.map((m) => [m, inner]);
    for (let i = 0; i < fm.meshes.length; i++) fm.meshes[i].material = pairs[Math.floor(rand() * pairs.length)];
    ctx.scene.add(fm.group);
    let n = 0;
    return {
      info: `A 4 x 2 x 0.5 m wall (600 kg/m³) cut into ${fb.ids.length} convex pieces (fracture seed ${FSEED}${FSEED === 35 ? ', picked because this hit throws pieces' : ''}; ?fseed=N) glued by ${fb.joints.length} breakable fixed joints (${FRACTURE_BREAK_FORCE / 1000} kN, compliant) at 120 Hz; a hit closing faster than ${IMPACT.speed} m/s frees the struck piece and the glue within ${IMPACT.radius} m before the contact solve; a 60 kg ball hits at 10 m/s 1.9 s in. Space throws more.`,
      benchSteps: 320, // 2.67 s at 120 Hz: 0.77 s after the hit (step 228), three pieces out and still moving
      fb, // the FracturedBody (window.__sandbox.scene.fb)
      focus: new THREE.Vector3(0, 1, 0), view: { distance: 10, elevation: 0.22, azimuth: 0.5 },
      poses: { overview: (cam) => look(cam, 5.5, 3.2, 8, 0, 1, 0), close: (cam) => look(cam, 1.6, 1.5, 3.4, 0, 0.9, -0.3) },
      fixedUpdate(dt) {
        if (n++ !== FIRE) return;
        const t = FLY * dt; // aimed so that gravity brings it down onto HIT
        const id = B.sphere(BALL, [HIT[0], HIT[1] + 4.905 * t * t, 0.25 + BALL + 0.005 + SPEED * t], { mass: MASS, friction: 0.5, restitution: 0.2, linearDamping: 0, color: 0xf2f2f2 });
        if (id >= 0) ctx.dyn.setVelocity(id, 0, 0, -SPEED, 0, 0, 0);
      },
      afterStep() { fb.update(); fm.capture(); },
      update: (dt, alpha) => fm.update(alpha),
      status: () => `${fb.ids.length} pieces · ${fb.brokenCount} / ${fb.joints.length} glue joints broken`,
      dispose() {
        world.removeParticleSolver(plugin); // first: a rebuild registers a fresh one (reset determinism)
        fm.dispose(); for (const m of mats) m.dispose(); inner.dispose(); bricks.dispose(); fb.dispose(); B.dispose();
      },
    };
  },
};

// ------------------------------------------------------------------ ragdoll layout
// Standing pose, feet at y = 0, facing +Z, arms out along +-X (T-pose). Parts leave 2-4 cm gaps at every joint so
// neighbours do not touch at rest. Joint anchors / axes are given in world space here and converted to each body's
// local frame once (a rigid transform of the whole doll leaves local anchors unchanged).
const RAGDOLL = (() => {
  const QA = [0, 0, -S2, S2]; // -90 deg about Z: capsule local +Y -> world +X (arms)
  const parts = [
    { name: 'pelvis', kind: 'capsule', dims: [0.12, 0.02], pos: [0, 1.10, 0], quat: Q0 },
    { name: 'chest', kind: 'capsule', dims: [0.15, 0.12], pos: [0, 1.53, 0], quat: Q0 },
    { name: 'head', kind: 'sphere', dims: [0.11], pos: [0, 1.94, 0], quat: Q0, skin: true },
  ];
  for (const s of [-1, 1]) {
    const side = s < 0 ? 'R' : 'L';
    parts.push(
      { name: `thigh${side}`, kind: 'capsule', dims: [0.075, 0.15], pos: [s * 0.1, 0.725, 0], quat: Q0 },
      { name: `shin${side}`, kind: 'capsule', dims: [0.06, 0.15], pos: [s * 0.1, 0.25, 0], quat: Q0, skin: true },
      { name: `upperArm${side}`, kind: 'capsule', dims: [0.05, 0.1], pos: [s * 0.34, 1.68, 0], quat: QA },
      { name: `forearm${side}`, kind: 'capsule', dims: [0.045, 0.1], pos: [s * 0.655, 1.68, 0], quat: QA, skin: true },
    );
  }
  for (const p of parts) p.extent = p.dims[0] + (p.dims[1] || 0); // bounding radius (lift-off height)
  const byName = Object.fromEntries(parts.map((p) => [p.name, p]));
  const pa = new THREE.Vector3(), qa = new THREE.Quaternion();
  const toLocal = (p, w) => pa.fromArray(w).sub(_s.fromArray(p.pos)).applyQuaternion(qa.fromArray(p.quat).invert()).toArray();
  const axisLocal = (p, w) => pa.fromArray(w).applyQuaternion(qa.fromArray(p.quat).invert()).toArray();
  const joints = [];
  const joint = (type, a, b, at, axis, extra) => {
    const A = byName[a], Bp = byName[b];
    joints.push({ a, b, desc: { type, anchorA: toLocal(A, at), anchorB: toLocal(Bp, at), axisA: axisLocal(A, axis), axisB: axisLocal(Bp, axis), ...extra } });
  };
  joint(JOINT.D6, 'pelvis', 'chest', [0, 1.25, 0], [0, 1, 0], { swing: 0.5, twist: [-0.4, 0.4] });
  joint(JOINT.D6, 'chest', 'head', [0, 1.815, 0], [0, 1, 0], { swing: 0.6, twist: [-0.8, 0.8] });
  for (const s of [-1, 1]) {
    const side = s < 0 ? 'R' : 'L';
    joint(JOINT.D6, 'chest', `upperArm${side}`, [s * 0.18, 1.68, 0], [s, 0, 0], { swing: 1.4, twist: [-0.8, 0.8] });
    joint(JOINT.HINGE, `upperArm${side}`, `forearm${side}`, [s * 0.5, 1.68, 0], [0, -s, 0], { limits: ELBOW_LIMITS });
    joint(JOINT.D6, 'pelvis', `thigh${side}`, [s * 0.1, 0.955, 0], [0, -1, 0], { swing: 1.2, twist: [-0.4, 0.4] });
    joint(JOINT.HINGE, `thigh${side}`, `shin${side}`, [s * 0.1, 0.48, 0], [1, 0, 0], { limits: KNEE_LIMITS });
  }
  return { parts, joints, pivot: new THREE.Vector3(0, 1, 0) };
})();
