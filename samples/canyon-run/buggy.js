// Rally buggy: a new VehicleController spec (light, high ride, soft knobbly tyres, ~180 km/h, flight: false),
// a procedural tube-frame mesh (2 draws + 1 instanced draw for the four wheels), render interpolation between
// fixed steps, the controller's sprung body (the wheels follow their own contact points), per-wheel surfaces (the
// PhysicsWorld surface provider set up in main.js), sun-lit dust from the rear wheels (ParticleSystem with
// terrain collision) and tyre tracks (DecalTrail lifted along the terrain normal). Allocation-free per frame.
import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { VehicleController, InstancedPool, ParticleSystem, DecalTrail, theme } from '@vibe/engine';

const DEG = Math.PI / 180;
const GEARS = [0, 500, 300, 207, 158, 122];   // virtual ratios (rpm per m/s of wheel speed)
const IDLE_RPM = 900, REDLINE = 7400;

/** Partial spec merged over ROADSTER_SPEC by resolveVehicleSpec (no flight package). */
export const BUGGY_SPEC = {
  // chassis
  mass: 760, length: 3.7, width: 1.95, height: 1.85,
  axleFront: -1.35, axleRear: 1.25, halfTrack: 0.82, wheelRadius: 0.42, cgHeight: 0.58, yawInertia: 1050,
  bodyPoints: [
    [-0.9, 0.36, -1.8], [0.9, 0.36, -1.8], [-0.9, 0.36, 1.75], [0.9, 0.36, 1.75],
    [-0.9, 1.0, -1.7], [0.9, 1.0, -1.7], [-0.9, 1.0, 1.7], [0.9, 1.0, 1.7],
    [-0.55, 1.86, -0.3], [0.55, 1.86, -0.3], [-0.55, 1.86, 0.5], [0.55, 1.86, 0.5],
  ],
  obb: { centerY: 1.06, half: [0.95, 0.74, 1.84] },
  stepHeight: 0.42, colliderRadius: 6, sweepStep: 0.35,
  normalClamp: 1.0,          // follow the canyon's slopes (the Roadster's 0.15 m caps the ground plane at ~3 deg)
  // soft, slidey dirt tyres (lower B = progressive; per-wheel surface multipliers from the surface provider)
  mu: 1.0, Bf: 7.5, Br: 8.5, C: 1.32, tyreLowSpeed: 2.2, gripCap: 0.92, handbrakeGrip: 0.32, handbrakeForce: 0.85,
  // steering
  maxSteer: 33 * DEG, steerBase: 0.075, steerGain: 1.1, steerSlipGain: 0.9, airSteer: 0.1,
  // powertrain: 4WD, ~150 kW, limiter at 50 m/s (180 km/h)
  motorPeakForce: 7600, motorPower: 150e3, vMax: 50, limiterRamp: 3.5, reverseMax: 9, reverseForce: 4200, frontDrive: 0.38,
  brakeDecel: 9.5,
  // aero (open cockpit, big frontal area, almost no downforce)
  CdA: 0.95, CdA_side: 2.6, CdA_top: 3.6, ClA: 0.12, downforceFront: 0.45, Crr: 0.018,
  // no thrusters / flight package; airborne: a little pitch / yaw, gentle self-levelling
  flight: false, boostPower: 1,
  airControl: 'arcade', airPitchRate: 0.5, airYawRate: 0.5,
  autoLevelTau: 1.6, levelPitchRateMax: 0.6, levelRollRateMax: 0.9, attitudeGain: 3, maxAngAccel: 2.5, authorityBase: 1, authorityTank: 0,
  // long-travel suspension (visual compression of the wheels)
  suspensionTravel: 0.2, suspensionSag: 0.4,
  // landings / impacts (long-travel suspension soaks up a lot)
  hardLandingSpeed: 11, hardLandingBounce: 0.22, landingDamageBase: 0.02, landingDamagePerMs: 0.01,
  wallRestitution: 0.25, wallFriction: 0.3, wallDamageSpeed: 7, wallDamagePerMs: 0.006, wallYawKick: 0.6,
  roofFriction: 0.7, autoRightDelay: 1.2, autoRightTau: 0.35,
};

/**
 * Ground materials returned by the PhysicsWorld surface provider (main.js), per wheel: mu / crr multiply the
 * spec's grip (mu 1.0) and rolling resistance (Crr 0.018); dust scales the rear-wheel dust.
 */
export const SURFACES = {
  track: { tag: 'track', mu: 1.0, crr: 1.0, dust: 0.75 },
  sand: { tag: 'sand', mu: 0.78, crr: 0.2 / 0.018, dust: 1.0 },
};

// ------------------------------------------------------------------ geometry helpers (vertex coloured)
const _c = new THREE.Color();
function colored(g, hex, k = 1) {
  g = g.index ? g.toNonIndexed() : g;
  if (g.attributes.uv) g.deleteAttribute('uv');
  if (typeof hex === 'number' || typeof hex === 'string') _c.set(hex); else _c.setRGB(hex[0], hex[1], hex[2]);
  const n = g.attributes.position.count, c = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) { c[i * 3] = _c.r * k; c[i * 3 + 1] = _c.g * k; c[i * 3 + 2] = _c.b * k; }
  g.setAttribute('color', new THREE.BufferAttribute(c, 3));
  return g;
}
const box = (w, h, d, x, y, z, hex) => colored(new THREE.BoxGeometry(w, h, d).translate(x, y, z), hex);
const _a = new THREE.Vector3(), _b = new THREE.Vector3(), _up = new THREE.Vector3(0, 1, 0), _q = new THREE.Quaternion();
function tube(p0, p1, r, hex) {
  _a.fromArray(p0); _b.fromArray(p1);
  const len = _a.distanceTo(_b);
  const g = new THREE.CylinderGeometry(r, r, len, 7, 1, true);
  _q.setFromUnitVectors(_up, _b.clone().sub(_a).normalize());
  g.applyQuaternion(_q).translate((_a.x + _b.x) / 2, (_a.y + _b.y) / 2, (_a.z + _b.z) / 2);
  return colored(g, hex);
}
const mirror = (p) => [-p[0], p[1], p[2]];

function buggyBodyGeometry() {
  const ACC = theme.accent, DARK = 0x1f2125, MID = 0x3b3e44, SEAT = 0x101113, CAGE = 0x2a2b2e;
  const P = [];
  P.push(box(1.1, 0.08, 2.75, 0, 0.5, -0.05, DARK));                                        // floor pan
  P.push(box(0.16, 0.34, 1.55, -0.63, 0.66, 0.05, ACC), box(0.16, 0.34, 1.55, 0.63, 0.66, 0.05, ACC)); // side pods
  P.push(box(1.0, 0.3, 0.72, 0, 0.66, -1.5, ACC), box(0.86, 0.22, 0.3, 0, 0.84, -1.2, DARK)); // nose + dash
  P.push(box(0.98, 0.5, 0.95, 0, 0.82, 1.2, MID), box(0.7, 0.16, 0.7, 0, 1.12, 1.2, DARK));  // engine cover + intake
  P.push(box(1.2, 0.12, 0.25, 0, 0.62, 1.72, DARK));                                        // rear bumper plate
  for (const s of [-1, 1]) {
    P.push(box(0.48, 0.14, 0.5, s * 0.28, 0.62, 0.22, SEAT));                              // seat base
    P.push(colored(new THREE.BoxGeometry(0.48, 0.78, 0.12).rotateX(-0.2).translate(s * 0.28, 1.02, 0.52), SEAT)); // seat back
    P.push(box(0.4, 0.5, 0.28, s * 0.28, 1.0, 0.3, s < 0 ? 0xf2f2f2 : 0xd8d8d8));         // torso (suit)
    P.push(colored(new THREE.SphereGeometry(0.165, 12, 9).translate(s * 0.28, 1.43, 0.27), s < 0 ? ACC : 0xf4f4f4)); // helmet
    P.push(box(0.26, 0.09, 0.08, s * 0.28, 1.45, 0.12, 0x0b0d10));                         // visor
    P.push(colored(new THREE.CylinderGeometry(0.06, 0.06, 0.62, 6).rotateZ(Math.PI / 2 - s * 0.5).translate(s * 0.28 - s * 0.12, 0.98, 0.05), 0xe8e8e8)); // arms
  }
  P.push(colored(new THREE.TorusGeometry(0.15, 0.022, 6, 16).rotateX(-1.1).translate(-0.28, 1.08, -0.12), 0x111111)); // wheel
  P.push(colored(new THREE.TorusGeometry(0.3, 0.11, 8, 18).translate(0, 1.05, 1.62), 0x141414)); // spare tyre
  // roll cage + frame (both sides)
  const T = [];
  const side = [
    [[0.62, 0.55, 0.62], [0.56, 1.78, 0.52]],       // main hoop leg
    [[0.62, 0.55, -0.78], [0.5, 1.72, -0.28]],      // A-pillar
    [[0.5, 1.72, -0.28], [0.56, 1.78, 0.52]],       // roof rail
    [[0.56, 1.78, 0.52], [0.5, 0.82, 1.62]],        // rear stay
    [[0.62, 0.92, -0.78], [0.62, 0.92, 0.62]],      // side bar
    [[0.62, 0.55, -0.78], [0.42, 0.56, -1.92]],     // front frame rail
    [[0.42, 0.56, -1.92], [0.42, 0.9, -1.75]],      // bumper upright
    [[0.62, 0.55, 0.62], [0.5, 0.6, 1.7]],          // rear frame rail
    [[0.5, 0.55, -1.3], [0.82, 0.42, -1.35]],       // front lower arm
    [[0.48, 1.08, -1.22], [0.8, 0.5, -1.35]],       // front shock
    [[0.5, 0.58, 1.22], [0.82, 0.42, 1.25]],        // rear lower arm
    [[0.48, 1.12, 1.0], [0.8, 0.5, 1.25]],          // rear shock
  ];
  for (const [a, b] of side) { T.push(tube(a, b, 0.036, CAGE), tube(mirror(a), mirror(b), 0.036, CAGE)); }
  T.push(tube([-0.56, 1.78, 0.52], [0.56, 1.78, 0.52], 0.036, CAGE), tube([-0.5, 1.72, -0.28], [0.5, 1.72, -0.28], 0.036, CAGE));
  T.push(tube([-0.42, 0.9, -1.75], [0.42, 0.9, -1.75], 0.04, ACC), tube([-0.56, 1.78, 0.52], [0.5, 1.72, -0.28], 0.03, CAGE));
  T.push(tube([-0.5, 0.82, 1.62], [0.5, 0.82, 1.62], 0.036, CAGE));
  for (const sg of [-1, 1]) T.push(colored(new THREE.CylinderGeometry(0.05, 0.05, 0.5, 8).rotateX(Math.PI / 2).translate(sg * 0.22, 0.72, 1.8), 0x777777)); // exhausts
  // light bar housing on the roof
  P.push(box(1.12, 0.13, 0.14, 0, 1.87, -0.24, DARK));
  return mergeGeometries([...P, ...T]);
}

function buggyGlowGeometry() {
  const G = [];
  for (let k = 0; k < 4; k++) G.push(colored(new THREE.BoxGeometry(0.22, 0.08, 0.03).translate(-0.375 + k * 0.25, 1.87, -0.315), [6, 6, 5.4]));
  for (const s of [-1, 1]) {
    G.push(colored(new THREE.CylinderGeometry(0.09, 0.09, 0.04, 14).rotateX(Math.PI / 2).translate(s * 0.32, 0.68, -1.87), [6, 6, 5.2]));
    G.push(colored(new THREE.BoxGeometry(0.16, 0.08, 0.03).translate(s * 0.4, 0.86, 1.68), [5, 0.18, 0.1]));
  }
  return mergeGeometries(G);
}

function wheelGeometry(R = 0.42, W = 0.34) {
  const P = [];
  const tire = new THREE.CylinderGeometry(R - 0.035, R - 0.035, W, 28, 1, false).rotateZ(Math.PI / 2);
  P.push(colored(tire, 0x18181a));
  const knobs = 22;
  for (let k = 0; k < knobs; k++) for (const row of [-1, 1]) {
    const a = ((k + (row > 0 ? 0.5 : 0)) / knobs) * Math.PI * 2;
    const g = new THREE.BoxGeometry(W * 0.42, 0.06, 0.1).translate(row * W * 0.24, R - 0.03, 0).rotateX(a);
    P.push(colored(g, 0x1c1c1e));
  }
  const rim = new THREE.CylinderGeometry(0.235, 0.235, W + 0.012, 18, 1, false).rotateZ(Math.PI / 2);
  P.push(colored(rim, 0x9a9da3));
  for (const s of [-1, 1]) {
    P.push(colored(new THREE.CylinderGeometry(0.09, 0.09, 0.02, 12).rotateZ(Math.PI / 2).translate(s * (W / 2 + 0.012), 0, 0), theme.accent));
    for (let k = 0; k < 5; k++) P.push(colored(new THREE.BoxGeometry(0.02, 0.2, 0.05).translate(s * (W / 2 + 0.008), 0.11, 0).rotateX((k / 5) * Math.PI * 2), 0x55585e));
  }
  return mergeGeometries(P);
}

// ------------------------------------------------------------------ buggy
const _p = new THREE.Vector3(), _v = new THREE.Vector3(), _fwd = new THREE.Vector3(), _right = new THREE.Vector3();
const _qa = new THREE.Quaternion(), _qb = new THREE.Quaternion(), _qc = new THREE.Quaternion(), _one = new THREE.Vector3(1, 1, 1);
const _n = new THREE.Vector3(), _X = new THREE.Vector3(1, 0, 0), _Y = new THREE.Vector3(0, 1, 0), _e = new THREE.Euler();

export class Buggy {
  /** @param {object} o  scene, physics (PhysicsWorld), terrain, atmosphere (lights the dust) */
  constructor({ scene, physics, terrain, atmosphere }) {
    this.physics = physics; this.terrain = terrain;
    const vc = this.vc = new VehicleController(BUGGY_SPEC);
    vc.hold = true;                  // countdown / menus: main.js toggles vc.hold
    this.spec = vc.spec;
    this.controls = { throttle: 0, brake: 0, steer: 0, handbrake: false, thrust: 0, pitch: 0, yaw: 0, boost: false };
    this.surface = SURFACES.track;   // majority surface under the rear wheels (dust, audio)
    // render interpolation
    this.prevPos = new THREE.Vector3(); this.prevQuat = new THREE.Quaternion();
    // scene graph: root follows the (interpolated) controller pose, body carries the suspension motion
    const root = this.object3d = new THREE.Group(); root.name = 'buggy';
    const body = this.body = new THREE.Group();
    this.bodyMesh = new THREE.Mesh(buggyBodyGeometry(), new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.48, metalness: 0.25 }));
    this.bodyMesh.castShadow = true; this.bodyMesh.receiveShadow = true;
    this.glowMat = new THREE.MeshBasicMaterial({ vertexColors: true });
    this.glowMesh = new THREE.Mesh(buggyGlowGeometry(), this.glowMat);
    body.add(this.bodyMesh, this.glowMesh);
    root.add(body);
    // headlights (one spot light, always present so the light count never changes -> no shader recompiles)
    const spot = this.spot = new THREE.SpotLight(0xfff1da, 0, 140, 0.62, 0.55, 1.3);
    spot.position.set(0, 1.5, -1.6); spot.target.position.set(0, -2, -30); spot.castShadow = false;
    body.add(spot, spot.target);
    scene.add(root);
    // wheels: one instanced draw (4 instances, matrices written each frame)
    this.wheels = new InstancedPool(wheelGeometry(this.spec.wheelRadius), new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.82, metalness: 0.1 }), { capacity: 4, castShadow: true, name: 'buggyWheels', cullRadius: 0.5 });
    for (let i = 0; i < 4; i++) this.wheels.add();
    scene.add(this.wheels.mesh);
    this.wheelLocal = this.spec.wheels.map((w) => new THREE.Vector3(w[0], 0, w[2]));
    this.droop = new Float32Array(4);
    // effects: dust collides with the terrain and dims in shade / at night (lit by the atmosphere's sun)
    this.dust = new ParticleSystem({ max: 1100, blending: 'normal', drag: 1.25, fadeIn: 0.12, spin: 0.7, name: 'buggyDust',
      floorAt: (x, z) => terrain.heightAt(x, z), bounce: 0.1, lit: true, litAmbient: 0.12, sun: atmosphere });
    this.dust.mesh.renderOrder = 9;
    scene.add(this.dust.mesh);
    this.tracks = new DecalTrail({ maxQuads: 2600, trails: 4, color: 0x2e2015, opacity: 0.62, minSegment: 0.3, fadeTail: 0.35, edge: 0.3, lift: 0.09, name: 'tyreTracks' });
    scene.add(this.tracks.mesh);
    this._dustAcc = new Float32Array(2);
    this.state = { velocity: vc.velocity, speedKmh: 0, mode: 'ground', thrust: 0, gForce: 1, grounded: true, wheelSlip: 0 };
    this.landing = 0;   // last landing impact (m/s), consumed by update() (suspension) and the audio thump
    this.lastLanding = 0;
    // virtual 5-speed auto gearbox (HUD gear + engine sound)
    this.gear = 1; this.rpm = IDLE_RPM; this.headingDeg = 0; this.throttle = 0; this._shiftT = 0;
  }

  reset(pos, heading) {
    const vc = this.vc;
    vc.reset(_p.set(pos.x, pos.y, pos.z), heading, this.physics);
    this.prevPos.copy(vc.position); this.prevQuat.copy(vc.quaternion);
    this.object3d.position.copy(vc.position); this.object3d.quaternion.copy(vc.quaternion);
    for (let i = 0; i < 4; i++) this.tracks.end(i);
    this.object3d.updateMatrixWorld(true);
  }

  /** Copy input into the reusable controls object (call once per frame). */
  setInput(inp, enabled) {
    const c = this.controls;
    if (!enabled) { c.throttle = 0; c.brake = 0; c.steer = 0; c.handbrake = true; c.yaw = 0; return; }
    c.throttle = inp.throttle; c.brake = inp.brake; c.steer = inp.steer; c.handbrake = inp.handbrake;
    c.yaw = inp.steer * 0.6; // airborne (arcade air control): a little yaw from the steering
  }

  fixedUpdate(dt) {
    const vc = this.vc;
    this.prevPos.copy(vc.position); this.prevQuat.copy(vc.quaternion);
    vc.step(dt, this.controls, this.physics);
    if (vc.landedThisStep) this.landing = Math.max(this.landing, vc.landingImpact);
  }

  /** Per frame: interpolated pose, suspension, wheels, dust, tyre tracks. */
  update(dt, alpha, darkness = 0) {
    const vc = this.vc, root = this.object3d;
    root.position.lerpVectors(this.prevPos, vc.position, alpha);
    root.quaternion.slerpQuaternions(this.prevQuat, vc.quaternion, alpha);
    // suspension: the controller's body is the sprung mass (its per-wheel springs give the heave / pitch / roll),
    // so the body mesh stays on the root; landings are kept only for the audio thump
    if (this.landing > 0) { this.lastLanding = this.landing; this.landing = 0; }
    this.body.position.y = 0;
    this.body.rotation.set(0, 0, 0);
    root.updateMatrixWorld(true);
    // wheels: each follows its own contact point (W[i].y: local height vs the static position, negative when
    // extended / hanging in the air); steer + spin
    const R = this.spec.wheelRadius, P = this.wheels, W = vc.wheels;
    _qb.setFromAxisAngle(_X, -vc.wheelSpin);
    for (let i = 0; i < 4; i++) {
      this.droop[i] += (-W[i].y - this.droop[i]) * (1 - Math.exp(-dt * 30));
      _v.copy(this.wheelLocal[i]); _v.y = R - this.droop[i];
      _p.copy(_v).applyQuaternion(root.quaternion).add(root.position);
      _qa.copy(root.quaternion);
      if (i < 2) { _qc.setFromAxisAngle(_Y, vc.steerAngle); _qa.multiply(_qc); }
      _qa.multiply(_qb);
      P.setTransform(i, _p, _qa, _one);
    }
    P.update(null);
    // headlights / glow by darkness
    this.spot.intensity = 260 * darkness;
    this.glowMat.color.setScalar(0.35 + 1.4 * darkness);
    this._effects(dt, darkness);
    // state for the camera / HUD / audio
    const st = this.state;
    st.speedKmh = vc.speed * 3.6; st.gForce = vc.gForce; st.grounded = vc.grounded; st.wheelSlip = vc.wheelSlip;
    st.mode = vc.mode; // always 'ground' (flight: false); the chase rig uses flightFraming: false anyway
    _fwd.set(0, 0, -1).applyQuaternion(root.quaternion);
    this.headingDeg = Math.atan2(_fwd.x, -_fwd.z) * 57.29578;
    this.throttle = this.controls.throttle;
    this._gearbox(dt);
  }

  _gearbox(dt) {
    const v = Math.abs(this.vc.wheelSpeed), thr = this.controls.throttle;
    this._shiftT = Math.max(0, this._shiftT - dt);
    let rpm = IDLE_RPM + v * GEARS[this.gear];
    if (rpm > 6900 && this.gear < 5) { this.gear++; this._shiftT = 0.18; } else if (rpm < 3300 && this.gear > 1) { this.gear--; this._shiftT = 0.12; }
    rpm = IDLE_RPM + v * GEARS[this.gear] + thr * 500;
    if (!this.vc.grounded) rpm += thr * 2200;           // free-revving in the air
    rpm = Math.min(REDLINE, Math.max(IDLE_RPM, rpm));
    if (this._shiftT > 0) rpm *= 0.86;
    this.rpm += (rpm - this.rpm) * Math.min(1, dt * 12);
  }

  _effects(dt, darkness) {
    const vc = this.vc, root = this.object3d, te = this.terrain, W = vc.wheels;
    _fwd.set(0, 0, -1).applyQuaternion(root.quaternion);
    _right.set(1, 0, 0).applyQuaternion(root.quaternion);
    const speed = vc.speed, slip = vc.wheelSlip;
    let sand = 0;
    for (let i = 0; i < 4; i++) {
      const w = this.wheelLocal[i], sf = W[i].tag === 'sand' ? SURFACES.sand : SURFACES.track;
      if (i >= 2 && sf === SURFACES.sand) sand++;
      _p.copy(w).applyQuaternion(root.quaternion).add(root.position);
      const onGround = vc.grounded && vc.wheelGaps[i] < 0.08 && speed > 1.5;
      const inten = !onGround ? 0 : sf === SURFACES.sand ? Math.min(1, 0.32 + slip * 0.7) : Math.min(1, Math.max(0, slip - 0.12) * 1.1);
      _p.y = te.heightAt(_p.x, _p.z);
      te.normalAt(_p.x, _p.z, _n);
      this.tracks.feed(i, _p, _right, 0.15, inten, _n);
      if (i < 2 || !onGround) continue;
      // rear wheels throw dust: proportional to slip, speed and the surface
      const rate = (slip * 90 + Math.min(1, speed / 35) * 55) * sf.dust * Math.min(1, 0.25 + speed / 10);
      let acc = this._dustAcc[i - 2] + rate * dt;
      while (acc >= 1) {
        acc -= 1;
        const sx = (Math.random() - 0.5) * 2.2, sy = Math.random() * 1.4 + 0.4, sz = (Math.random() - 0.5) * 2.2;
        const back = -0.18 * speed;
        const life = 1.2 + Math.random() * 1.4, s1 = 2.4 + slip * 2.4 + Math.random();
        const l = 0.92 + Math.random() * 0.12;
        this.dust.spawn(_p.x + (Math.random() - 0.5) * 0.4, _p.y + 0.25, _p.z + (Math.random() - 0.5) * 0.4,
          _fwd.x * back + sx + _right.x * (i === 2 ? -0.8 : 0.8), sy, _fwd.z * back + sz + _right.z * (i === 2 ? -0.8 : 0.8),
          life, 0.45, s1, 0.15 + slip * 0.09, 0.62 * l, 0.45 * l, 0.31 * l, 0.72 * l, 0.6 * l, 0.47 * l, 1.25, -0.15);
      }
      this._dustAcc[i - 2] = acc;
    }
    this.surface = sand === 2 ? SURFACES.sand : sand === 0 ? SURFACES.track : this.surface;
    this.dust.update(dt);
  }
}

/** Heading (rad, three's rotation.y convention) of the buggy. */
export function headingOf(q) { _e.setFromQuaternion(q, 'YXZ'); return _e.y; }
