// Dynamics of the character playground: an in-thread XPBD world (DynamicsClient({ worker: false })) holding six
// pushable crates (0.5 / 0.7 m, 15-60 kg, one stacked on another), one 400 kg crate (heavier than the controller's
// pushMass, so a wall: the near-spawn mantle ledge) and a kinematic 3 x 0.3 x 2 m platform that shuttles between the
// west face of the 2 m platform (its top at the same 2 m) and a point 5 m up between the pillars (an elevator ride: a
// jump off its high end, or a step off its north / south edge, lands at > 9 m/s: the hard-fall ragdoll).
// attachToPhysicsWorld publishes every body to the PhysicsWorld as a 'rigid' record (the controller collides with it,
// pushes it or rides it: cc.attachDynamics). Records are the bodies' AABBs WITH the world's 1 cm margin, so the feet
// stand 1 cm above every crate and the elevator, and the capsule stops 1 cm short of their faces. One InstancedPool of
// unit boxes draws them (per-body colour, interpolated between fixed steps like the sandbox's BodyPools).
//
//   const props = new PlaygroundDynamics({ scene, physics, velocity });   // world + visuals, start layout
//   props.attach(cc, opts)        cc.attachDynamics(props.dyn, opts), redone on every rebuild
//   props.onRebuild = (dyn) => {} after reset() replaced the world (main.js points its ragdoll at the new client)
//   fixed step, AFTER cc.step:    props.step(dt)   platform target from the step count, dyn.step(dt), poses captured
//   per frame:                    props.sync(alpha)
//   props.reset()                 dispose the client and build a fresh one in the start layout (bench poses, respawn)
//   props.setVisible(on)          (the idle / run / aerial bench poses hide the props: their captures predate them)
//   props.platformT(k)            the platform's progress A -> B (0..1) after k fixed steps
// Why the world steps after the controller: cc.step reads the bridge records of the previous dyn.step either way, and
// stepping after it lets the impulses the controller applies while pushing (dyn.applyImpulse inside cc.step) act in
// the same fixed step, with the platform's kinematic target set right before the step that moves it.
// Determinism: the layout is constant, the platform follows a smoothed triangle wave of the fixed-step count (+ - * /
// only, no clock), the in-thread world is deterministic, and reset() starts from a NEW client, so no solver history
// (warm starts, friction anchors, sleep timers) reaches a bench pose: a reset world is bit-identical to a page load's.
// Allocation: step() and sync() allocate nothing; construction and reset() do.
import * as THREE from 'three';
import { DynamicsClient, attachToPhysicsWorld, boxShape } from '@vibe/engine/physics';
import { InstancedPool } from '@vibe/engine/render';

/** Pushable crates resting on the ground (or on crate `on`): centre x / z, edge (m), mass (kg). [0]: the push pose's. */
export const CRATES = [
  { x: -2.0, z: 2.6, size: 0.7, mass: 30, color: '#c99a52' },
  { x: -3.2, z: 5.0, size: 0.5, mass: 15, color: '#d9b26a' },
  { x: -4.2, z: 6.6, size: 0.7, mass: 45, color: '#b5874a' },
  { x: -2.6, z: 7.4, size: 0.5, mass: 20, color: '#d1a560' },
  { x: -5.4, z: 4.6, size: 0.7, mass: 60, color: '#a97c45' },
  { x: -5.4, z: 4.6, size: 0.5, mass: 25, color: '#dcb978', on: 4 },
];
/** The crate the controller cannot move (mass > pushMass): a 1.1 m ledge east of the spawn. */
export const HEAVY = { x: 3.0, z: 3.4, size: 1.1, mass: 400, color: '#5d6672' };
/**
 * Kinematic platform (half extents hx / hy / hz): centre z fixed, (x, y) shuttles on a straight line between A (east
 * edge 5 cm off the 2 m platform's west face, top at 2 m like it: its record 1 cm higher) and B (x 1, top at 5 m:
 * between the pillars, above them at that end). One period (there and back) lasts `period` fixed steps (12 s at
 * 120 Hz); `dwell` of each half of the triangle wave is spent at each end (1.8 s).
 */
export const PLATFORM = {
  hx: 1.5, hy: 0.15, hz: 1.0, z: -7.2, xA: 5.45, yA: 1.85, xB: 1.0, yB: 4.85, period: 1440, dwell: 0.15, color: '#3fae9f',
};
const GAP = 0.002;           // spawn clearance above the support (no initial overlap)
const BODY = { friction: 0.6, restitution: 0.05, linearDamping: 0.05, angularDamping: 0.1 };

export class PlaygroundDynamics {
  /**
   * @param {object} o
   * @param {THREE.Scene} o.scene
   * @param {import('@vibe/engine/physics').PhysicsWorld} o.physics  static world (ground + boxes) and query hub
   * @param {boolean} [o.velocity=false]  InstancedPool({ velocity }) for G-buffer motion vectors (PostFX.trackPool)
   */
  constructor({ scene, physics, velocity = false }) {
    this.physics = physics;
    this.pool = new InstancedPool(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshStandardMaterial({ roughness: 0.72, metalness: 0.03 }),
      { capacity: 16, colors: true, castShadow: true, cullRadius: 1.9, name: 'playground.dynamics', velocity });
    scene.add(this.pool.mesh);
    /** One per drawn body: { id, slot, sx, sy, sz, p0, q0, p1, q1, pos, quat } (previous / current / rendered pose). */
    this.records = [];
    this.dyn = null;
    this.crates = [];          // body ids of CRATES (same order)
    this.heavy = -1;
    this.platform = -1;
    /** Fixed steps since the last (re)build: the platform's clock. */
    this.steps = 0;
    this.visible = true;
    /** (dyn) => void, called after reset() built a new world. */
    this.onRebuild = null;
    this._cc = null; this._ccOpts = null;
    this._m = new THREE.Matrix4(); this._s = new THREE.Vector3();
    this._build();
  }

  /** Bind the controller: cc.attachDynamics(dyn, opts) now and after every rebuild (no-op without that API). */
  attach(cc, opts = {}) {
    this._cc = cc; this._ccOpts = opts;
    if (typeof cc.attachDynamics === 'function') cc.attachDynamics(this.dyn, opts);
    return this;
  }

  /** Platform progress A -> B (0..1) after k fixed steps: a triangle wave of k, clamped for the dwell, smoothstepped
   *  (+ - * / only). */
  platformT(k) {
    const P = PLATFORM, u = (k % P.period) / P.period, tri = u < 0.5 ? 2 * u : 2 - 2 * u;
    let s = (tri - P.dwell) / (1 - 2 * P.dwell);
    s = s < 0 ? 0 : s > 1 ? 1 : s;
    return s * s * (3 - 2 * s);
  }

  _build() {
    const dyn = this.dyn = new DynamicsClient({ worker: false, physics: this.physics, substeps: 4, maxBodies: 64, maxContacts: 2048, maxJoints: 64 });
    attachToPhysicsWorld(dyn, this.physics);
    this.steps = 0;
    const shapes = new Map();  // one shape per crate size
    const cube = (s) => { let sh = shapes.get(s); if (!sh) shapes.set(s, sh = boxShape(s / 2, s / 2, s / 2)); return sh; };
    const tops = [];
    this.crates.length = 0;
    for (let i = 0; i < CRATES.length; i++) {
      const c = CRATES[i], h = c.size / 2, base = c.on === undefined ? 0 : tops[c.on] + GAP;
      this.crates.push(dyn.addBody({ shape: cube(c.size), position: [c.x, base + h + GAP, c.z], mass: c.mass, ...BODY }));
      tops.push(base + c.size + GAP);
    }
    this.heavy = dyn.addBody({ shape: cube(HEAVY.size), position: [HEAVY.x, HEAVY.size / 2 + GAP, HEAVY.z], mass: HEAVY.mass, ...BODY });
    const P = PLATFORM;
    const t0 = this.platformT(0);
    this.platform = dyn.addBody({ type: 'kinematic', shape: boxShape(P.hx, P.hy, P.hz), position: [P.xA + (P.xB - P.xA) * t0, P.yA + (P.yB - P.yA) * t0, P.z], friction: 0.8, restitution: 0 });

    if (!this.records.length) {      // first build: one pool slot per body (ids are the same on every rebuild)
      const add = (id, sx, sy, sz, color) => {
        const slot = this.pool.add(), c = new THREE.Color(color);
        this.pool.setColor(slot, c.r, c.g, c.b);
        this.records.push({ id, slot, sx, sy, sz, p0: new THREE.Vector3(), q0: new THREE.Quaternion(), p1: new THREE.Vector3(), q1: new THREE.Quaternion(), pos: new THREE.Vector3(), quat: new THREE.Quaternion() });
      };
      for (let i = 0; i < CRATES.length; i++) { const s = CRATES[i].size; add(this.crates[i], s, s, s, CRATES[i].color); }
      add(this.heavy, HEAVY.size, HEAVY.size, HEAVY.size, HEAVY.color);
      add(this.platform, 2 * P.hx, 2 * P.hy, 2 * P.hz, P.color);
    } else {
      const R = this.records;
      for (let i = 0; i < R.length; i++) R[i].id = i < CRATES.length ? this.crates[i] : i === CRATES.length ? this.heavy : this.platform;
    }
    this.snap();
    if (this._cc && typeof this._cc.attachDynamics === 'function') this._cc.attachDynamics(dyn, this._ccOpts);
  }

  /**
   * A fresh world in the start layout: the controller is detached, the old client disposed (its bodies leave the
   * PhysicsWorld), a new one built and the controller re-attached; then onRebuild(dyn). Deactivate anything holding
   * bodies of the old world first (the ragdoll). Allocates (bench poses, respawn).
   */
  reset() {
    if (this._cc && typeof this._cc.detachDynamics === 'function') this._cc.detachDynamics();
    if (this.dyn) this.dyn.dispose();
    this._build();
    if (this.onRebuild) this.onRebuild(this.dyn);
    return this;
  }

  /** One fixed step (call after cc.step): the platform's target for the end of this step, dyn.step, poses captured. */
  step(dt) {
    const P = PLATFORM, t = this.platformT(++this.steps);
    this.dyn.setKinematicTarget(this.platform, P.xA + (P.xB - P.xA) * t, P.yA + (P.yB - P.yA) * t, P.z, 0, 0, 0, 1);
    this.dyn.step(dt);
    const R = this.records, dyn = this.dyn;
    for (let i = 0; i < R.length; i++) {
      const r = R[i];
      r.p0.copy(r.p1); r.q0.copy(r.q1);
      dyn.getPose(r.id, r.p1, r.q1);
    }
    return this;
  }

  /** previous = current = the bodies' poses now (after a rebuild or a teleport: no interpolation streak). */
  snap() {
    const R = this.records, dyn = this.dyn;
    for (let i = 0; i < R.length; i++) {
      const r = R[i];
      dyn.getPose(r.id, r.p1, r.q1);
      r.p0.copy(r.p1); r.q0.copy(r.q1);
    }
    this.sync(1);
    return this;
  }

  /** Per frame: poses interpolated by alpha (0 = previous fixed step, 1 = current) into the instance matrices. */
  sync(alpha) {
    const R = this.records, P = this.pool, m = this._m, s = this._s;
    for (let i = 0; i < R.length; i++) {
      const r = R[i];
      r.pos.lerpVectors(r.p0, r.p1, alpha);
      r.quat.slerpQuaternions(r.q0, r.q1, alpha);
      P.setMatrix(r.slot, m.compose(r.pos, r.quat, s.set(r.sx, r.sy, r.sz)));
    }
    P.updateBounds();
    P.mesh.visible = this.visible && P.count > 0;
    return this;
  }

  /** Show / hide every prop (crates + platform: pool mesh, shadows included). */
  setVisible(on) {
    this.visible = !!on;
    this.pool.mesh.visible = this.visible && this.pool.count > 0;
    return this;
  }

  dispose() {
    if (this._cc && typeof this._cc.detachDynamics === 'function') this._cc.detachDynamics();
    if (this.dyn) { this.dyn.dispose(); this.dyn = null; }
    this.pool.mesh.removeFromParent();
    this.pool.dispose();
    this.pool.geometry.dispose(); this.pool.material.dispose();
  }
}
