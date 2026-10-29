// Sandbox scenes on the XPBD dynamics world, plus the instanced visuals every scene shares.
//
//   SCENES[name](ctx) -> { info, focus (Vector3) | target (Object3D), view, poses: { overview(cam), close(cam) },
//                          benchSteps (fixed steps simulated before a bench pose), earlyUpdate?(dt),
//                          fixedUpdate?(dt) (runs BEFORE dyn.step), update?(dt, alpha), setInputEnabled?(on), status?(),
//                          dispose() }
//   ctx = { dyn, physics, scene, pools, params, THREE, rand }   (rand: seeded mulberry32, fresh for every build)
//   buildStatics(name, { physics, scene, renderer }) adds a scene's PhysicsWorld boxes ONCE per page, before the
//   DynamicsClient is created (PhysicsWorld boxes cannot be removed, and a worker client copies them at start-up).
//   SCENE_HZ[name]: the fixed-step rate a scene wants (default 60; the car's controller is tuned at 120).
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
import { Shapes, JOINT } from '@vibe/engine/physics';
import { InstancedPool } from '@vibe/engine/render';
import { createCar } from './car.js';

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
/** PhysicsWorld boxes + their meshes for scene `name` (bench1000: the 20 m pit). Call before creating the client. */
export function buildStatics(name, { physics, scene }) {
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
export const SCENE_HZ = { car: 120 };

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
