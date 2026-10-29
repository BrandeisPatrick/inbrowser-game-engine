// Sandbox car: a VehicleController (ROADSTER_SPEC, no thrust input) driven by VehicleInput on the flat ground and
// mirrored into the dynamics world as a KINEMATIC box (the spec's body OBB) that follows the controller every fixed
// step through setKinematicTarget, so it shoves the crates. attachToPhysicsWorld (main.js) makes the crates 'rigid'
// colliders the controller drives against; the controller gets a filtered view of the PhysicsWorld that hides the
// car's own kinematic box (it would otherwise collide with itself). Visual: a chassis + cabin box and 4 wheel
// cylinders (no assets), interpolated between fixed steps as samples/canyon-run/buggy.js does (the wheels follow
// their contact points, steer and spin). Allocation-free per frame and per step.
//
// main.js attaches the bridge with mask ~PROXY_LAYER, so the proxy (on PROXY_LAYER) is not published as a collider;
// the controller's filtered world also drops a record whose id is the proxy's (a bridge attached without the mask).
// The controller treats 'rigid' boxes as immovable walls (it bounces off a crate rather than pushing it);
// solidCrates: false hides every 'rigid' box from it instead, and the kinematic box bulldozes the crates.
//
//   const car = createCar(ctx, { position: [0, 0, 6], heading: 0 });
//   earlyUpdate: car.earlyUpdate(dt) · fixedUpdate (after dyn.step): car.fixedUpdate(dt) · update: car.update(dt, alpha)
import * as THREE from 'three';
import { VehicleController, ROADSTER_SPEC, Shapes } from '@vibe/engine/physics';
import { VehicleInput, inputRamp, PAD_BUTTONS as B, PAD_AXES as AX } from '@vibe/engine/input';

// ground driving only: Space throws balls in the sandbox (no thrust), R resets the scene, V / C / H are the page's
const PEDAL = inputRamp(0.12, 0.06), STEER = inputRamp(0.15, 0.07);
export const CAR_ACTIONS = {
  throttle: { type: 'axis', range: [0, 1], smoothing: PEDAL, bindings: ['KeyW', 'ArrowUp', { padButton: B.RT }, { virtual: 'gas' }] },
  brake: { type: 'axis', range: [0, 1], smoothing: PEDAL, bindings: ['KeyS', 'ArrowDown', { padButton: B.LT }, { virtual: 'brake' }] },
  steer: { type: 'axis', range: [-1, 1], smoothing: STEER, bindings: [
    { key: 'KeyD', value: 1 }, { key: 'ArrowRight', value: 1 }, { key: 'KeyA', value: -1 }, { key: 'ArrowLeft', value: -1 },
    { padAxis: AX.LX }, { virtual: 'stick.x' }] },
  handbrake: { type: 'button', bindings: ['KeyX', { padButton: B.B }] },
  boost: { type: 'button', bindings: ['ShiftLeft', 'ShiftRight', { padButton: B.X }] },
};
const CAR_TOUCH = [
  { type: 'stick', id: 'stick', side: 'left', x: 120, y: 160, radius: 66 },
  { type: 'button', id: 'gas', label: 'GAS', side: 'right', x: 76, y: 160, size: 82 },
  { type: 'button', id: 'brake', label: 'BRK', side: 'right', x: 172, y: 128, size: 62 },
];

/** Collision layer bit of the car's kinematic proxy (main.js leaves it out of attachToPhysicsWorld). */
export const PROXY_LAYER = 1 << 3;

const _p = new THREE.Vector3(), _c = new THREE.Vector3();

/**
 * @param {object} ctx  the sandbox scene context ({ dyn, physics, scene })
 * @param {object} [o]  position [x, y, z] (y ignored: the car rests on the ground), heading (rad, 0 = -Z), color
 * @returns {object} { vc, input, object3d, bodyId, controls, earlyUpdate(dt), fixedUpdate(dt), update(dt, alpha), setInputEnabled(on), dispose() }
 */
export function createCar(ctx, o = {}) {
  const { dyn, physics, scene } = ctx;
  const vc = new VehicleController(ROADSTER_SPEC);
  const S = vc.spec, H = S.obb.half, CY = S.obb.centerY, R = S.wheelRadius;
  const controls = { throttle: 0, brake: 0, steer: 0, handbrake: false, boost: false, thrust: 0, pitch: 0, yaw: 0 };
  const car = { vc, controls, bodyId: -1, inputEnabled: true };

  // ---- the controller's world: the PhysicsWorld minus the car's own kinematic box (bridge records are
  // { ..., kind: 'rigid', id, vx, vz }), or minus every dynamic body with solidCrates: false
  const solid = o.solidCrates !== false;
  const isSelf = (b) => b.kind === 'rigid' && (!solid || b.id === car.bodyId);
  const world = {
    getGroundHeight: (x, z, y) => physics.getGroundHeight(x, z, y),
    surfaceAt: (x, z, out) => physics.surfaceAt(x, z, out),
    queryColliders(x, z, r, out = []) {
      physics.queryColliders(x, z, r, out);
      let n = 0;
      for (let i = 0; i < out.length; i++) { const b = out[i]; if (!isSelf(b)) out[n++] = b; }
      out.length = n;
      return out;
    },
  };
  const centre = (out) => out.set(0, CY, 0).applyQuaternion(vc.quaternion).add(vc.position); // body OBB centre

  const pos = o.position || [0, 0, 0];
  vc.reset(_p.set(pos[0], 0, pos[2]), o.heading || 0, world);
  centre(_c);
  const q0 = vc.quaternion;
  car.bodyId = dyn.addBody({
    type: 'kinematic', shape: Shapes.box(H[0], H[1], H[2]), position: [_c.x, _c.y, _c.z], quaternion: [q0.x, q0.y, q0.z, q0.w],
    friction: 0.5, restitution: 0.1, layer: PROXY_LAYER, userData: { car: true },
  });
  const prevPos = vc.position.clone(), prevQuat = vc.quaternion.clone();

  // ---- input (touch controls on touch-first devices only)
  const input = car.input = new VehicleInput(null, { actions: CAR_ACTIONS, touchLayout: CAR_TOUCH });

  // ---- visual: chassis + cabin + lights, 4 wheels (tyre, rim, a spoke bar so the spin reads)
  const root = car.object3d = new THREE.Group(); root.name = 'car';
  const geos = [], mats = [];
  const mesh = (geo, mat, x, y, z, parent = root) => {
    const m = new THREE.Mesh(geo, mat);
    m.position.set(x, y, z); m.castShadow = true; m.receiveShadow = true;
    parent.add(m);
    return m;
  };
  const mat = (o2) => { const m = new THREE.MeshStandardMaterial(o2); mats.push(m); return m; };
  const geo = (g) => { geos.push(g); return g; };
  const paint = mat({ color: o.color ?? 0xd6402f, roughness: 0.35, metalness: 0.3 });
  const glass = mat({ color: 0x1a1f26, roughness: 0.15, metalness: 0.6 });
  const tyre = mat({ color: 0x18181a, roughness: 0.9, metalness: 0 });
  const rim = mat({ color: 0x9a9da3, roughness: 0.4, metalness: 0.6 });
  const head = mat({ color: 0xffffff, emissive: 0xfff1da, emissiveIntensity: 1.5 });
  const tail = mat({ color: 0x440000, emissive: 0xff2010, emissiveIntensity: 1.2 });
  const len = S.length - 0.1, wid = 2 * (S.halfTrack - 0.13);
  mesh(geo(new THREE.BoxGeometry(wid, 0.38, len)), paint, 0, 0.47, 0);
  mesh(geo(new THREE.BoxGeometry(wid - 0.2, 0.42, len * 0.46)), glass, 0, 0.87, 0.25);
  mesh(geo(new THREE.BoxGeometry(wid - 0.24, 0.05, len * 0.42)), paint, 0, 1.105, 0.25);
  const lamp = geo(new THREE.BoxGeometry(0.34, 0.08, 0.04));
  for (const s of [-1, 1]) { mesh(lamp, head, s * 0.52, 0.56, -len / 2 - 0.01); mesh(lamp, tail, s * 0.52, 0.56, len / 2 + 0.01); }
  const tyreGeo = geo(new THREE.CylinderGeometry(R, R, 0.26, 24).rotateZ(Math.PI / 2));
  const rimGeo = geo(new THREE.CylinderGeometry(R * 0.62, R * 0.62, 0.27, 16).rotateZ(Math.PI / 2));
  const spokeGeo = geo(new THREE.BoxGeometry(0.275, R * 1.15, 0.07));
  const wheels = [], wheelLocal = S.wheels.map((w) => new THREE.Vector3(w[0], 0, w[2]));
  for (let i = 0; i < 4; i++) {
    const g = new THREE.Group();
    mesh(tyreGeo, tyre, 0, 0, 0, g); mesh(rimGeo, rim, 0, 0, 0, g); mesh(spokeGeo, tyre, 0, 0, 0, g);
    g.position.set(wheelLocal[i].x, R, wheelLocal[i].z);
    root.add(g); wheels.push(g);
  }
  root.position.copy(vc.position); root.quaternion.copy(vc.quaternion);
  scene.add(root);

  /** Per frame, before the fixed steps: poll the controls (zero + handbrake while disabled, e.g. free-fly camera). */
  car.earlyUpdate = (dt) => {
    input.update(dt);
    if (!car.inputEnabled) { controls.throttle = 0; controls.brake = 0; controls.steer = 0; controls.handbrake = true; controls.boost = false; return; }
    controls.throttle = input.throttle; controls.brake = input.brake; controls.steer = input.steer;
    controls.handbrake = input.handbrake; controls.boost = input.boost;
  };
  /** Fixed step (after dyn.step): advance the controller, then aim the kinematic box at its new pose. */
  car.fixedUpdate = (dt) => {
    prevPos.copy(vc.position); prevQuat.copy(vc.quaternion);
    vc.step(dt, controls, world);
    centre(_c);
    const q = vc.quaternion;
    if (car.bodyId >= 0) dyn.setKinematicTarget(car.bodyId, _c.x, _c.y, _c.z, q.x, q.y, q.z, q.w);
  };
  /** Per frame: interpolated pose, wheels on their contact points, front steer, spin. */
  car.update = (dt, alpha) => {
    root.position.lerpVectors(prevPos, vc.position, alpha);
    root.quaternion.slerpQuaternions(prevQuat, vc.quaternion, alpha);
    const W = vc.wheels;
    for (let i = 0; i < 4; i++) {
      const g = wheels[i];
      g.position.y = R + W[i].y;
      g.rotation.set(-vc.wheelSpin, i < 2 ? vc.steerAngle : 0, 0, 'YXZ');
    }
  };
  car.setInputEnabled = (on) => { car.inputEnabled = !!on; input.map.enabled = !!on; };
  car.dispose = () => {
    scene.remove(root);
    for (const g of geos) g.dispose();
    for (const m of mats) m.dispose();
    input.dispose();
    if (car.bodyId >= 0) { dyn.removeBody(car.bodyId); car.bodyId = -1; }
  };
  return car;
}
