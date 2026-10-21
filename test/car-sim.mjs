// Headless physics checks for the Roadster car and the engine's PhysicsWorld queries (no browser, no WebGL).
// Run: npm test  (node test/car-sim.mjs; `@vibe/engine` resolves to this package itself)
import { Vector3 } from 'three';
import { VehicleController, ROADSTER_SPEC, PhysicsWorld, LAYER, layerOf, G } from '@vibe/engine/physics';

const CarPhysics = VehicleController; // spec defaults to ROADSTER_SPEC; ({ hoverAssist }) is accepted as a partial spec

const DT = 1 / 120;
let failures = 0;
const results = [];
function check(name, value, lo, hi, unit = '') {
  const ok = Number.isFinite(value) && value >= lo && value <= hi;
  if (!ok) failures++;
  results.push(`${ok ? 'PASS' : 'FAIL'}  ${name.padEnd(42)} ${fmt(value)} ${unit}   [${lo} .. ${hi}]`);
  return ok;
}
const fmt = (x) => (Number.isFinite(x) ? x.toFixed(2).padStart(8) : String(x).padStart(8));

// flat world with optional boxes
function makeWorld(boxes = []) {
  return {
    getGroundHeight: () => 0,
    queryColliders(x, z, r) {
      return boxes.filter((b) => x + r > b.minX && x - r < b.maxX && z + r > b.minZ && z - r < b.maxZ);
    },
  };
}
const ctl = (o = {}) => ({ throttle: 0, brake: 0, steer: 0, thrust: 0, pitch: 0, roll: 0, yaw: 0, boost: false, handbrake: false, ...o });
const kmh = (ms) => ms * 3.6;

function run(car, world, seconds, controls, onStep) {
  const n = Math.round(seconds / DT);
  for (let i = 0; i < n; i++) {
    const c = typeof controls === 'function' ? controls(i * DT, car) : controls;
    car.step(DT, c, world);
    if (onStep && onStep(i * DT, car) === false) return i * DT;
  }
  return seconds;
}
function assertSane(car, label) {
  const p = car.position, v = car.velocity;
  if (![p.x, p.y, p.z, v.x, v.y, v.z].every(Number.isFinite)) { failures++; results.push(`FAIL  ${label}: NaN state`); }
}

// --- 1. 0-100 km/h -----------------------------------------------------------------
{
  const world = makeWorld();
  for (const boost of [false, true]) {
    const car = new CarPhysics();
    car.reset(new Vector3(0, 0, 0), 0, world);
    let t100 = NaN;
    run(car, world, 6, ctl({ throttle: 1, boost }), (t, c) => {
      if (c.forwardSpeed >= 100 / 3.6) { t100 = t + DT; return false; }
    });
    assertSane(car, '0-100');
    if (boost) check('0-100 km/h, boost (thruster launch)', t100, 0.95, 1.35, 's');
    else check('0-100 km/h, motors only', t100, 1.6, 2.6, 's');
  }
}

// --- 2. top speed after 30 s -----------------------------------------------------
{
  const world = makeWorld();
  const car = new CarPhysics();
  car.reset(new Vector3(0, 0, 0), 0, world);
  let t300 = NaN;
  run(car, world, 30, ctl({ throttle: 1 }), (t, c) => { if (Number.isNaN(t300) && kmh(c.forwardSpeed) >= 300) t300 = t; });
  check('0-300 km/h', t300, 6, 16, 's');
  check('top speed after 30 s', kmh(car.forwardSpeed), 385, 415, 'km/h');
  check('stays on the road (|x| drift)', Math.abs(car.position.x), 0, 0.5, 'm');
  check('rides within the suspension at top speed (y)', car.position.y, -0.06, 0.01, 'm'); // downforce compresses the springs ~1 cm
}

// --- 3. braking 100 -> 0 -----------------------------------------------------------
{
  const world = makeWorld();
  const car = new CarPhysics();
  car.reset(new Vector3(0, 0, 0), 0, world);
  car.velocity.set(0, 0, -100 / 3.6);
  const z0 = car.position.z;
  let stopT = NaN;
  run(car, world, 6, ctl({ brake: 1 }), (t, c) => { if (c.forwardSpeed <= 0.05) { stopT = t; return false; } });
  const dist = Math.abs(car.position.z - z0);
  check('braking distance 100-0', dist, 26, 38, 'm');
  check('braking time 100-0', stopT, 1.8, 3.0, 's');
}

// --- 4. hover lift-off and landing ------------------------------------------------
{
  const world = makeWorld();
  const car = new CarPhysics();
  car.reset(new Vector3(0, 0, 0), 0, world);
  let liftT = NaN, flightT = NaN, maxAlt = 0, landT = NaN, landV = 0, flightSeen = false;
  // 3 s full thrust, then release (hover-assist controls the descent)
  run(car, world, 30, (t) => ctl({ thrust: t < 3 ? 1 : 0 }), (t, c) => {
    if (Number.isNaN(liftT) && c.position.y > 0.05) liftT = t;
    if (c.mode === 'flight' && !flightSeen) { flightSeen = true; flightT = t; }
    maxAlt = Math.max(maxAlt, c.altitude);
    if (flightSeen && c.grounded && Number.isNaN(landT)) { landT = t; landV = c.landingImpact; }
    if (!Number.isNaN(landT) && t > landT + 1) return false;
  });
  check('lift-off time (thrust on -> wheels up)', liftT, 0, 0.25, 's');
  check('flight mode after lift-off', flightT - liftT, 0.05, 0.3, 's');
  check('max altitude (3 s at 2.2 g, then coast)', maxAlt, 60, 120, 'm');
  check('landed (time)', landT, 5, 30, 's');
  check('touchdown sink speed (hover assist)', landV, 0, 3, 'm/s');
  check('no damage from soft landing', car.damage, 0, 0.001);
  check('tank used then recharging', car.tank, 0.5, 0.99);
  check('ground mode after landing', car.mode === 'ground' ? 1 : 0, 1, 1);
  // free hover: 45% thrust keeps altitude roughly constant
  const car2 = new CarPhysics({ hoverAssist: false });
  car2.reset(new Vector3(0, 0, 0), 0, world);
  run(car2, world, 2, ctl({ thrust: 1 }));
  const y0 = car2.position.y; const vy0 = car2.velocity.y;
  run(car2, world, 1.0, ctl({ thrust: 0 }));
  check('climb speed after 2 s full thrust', vy0, 15, 26, 'm/s');
  check('altitude gained 2 s full thrust', y0, 15, 30, 'm');
}

// --- 5. drop without thrust assist -> hard landing ---------------------------------
{
  const world = makeWorld();
  const car = new CarPhysics({ hoverAssist: false });
  car.reset(new Vector3(0, 0, 0), 0, world);
  car.position.y = 25; car.grounded = false; car.airTime = 1;
  let impactV = 0, bounced = false;
  run(car, world, 5, ctl(), (t, c) => {
    if (c.landedThisStep && !impactV) impactV = c.landingImpact;
    if (impactV && c.velocity.y > 1) bounced = true;
  });
  check('hard landing impact speed (25 m drop)', impactV, 18, 23, 'm/s');
  check('hard landing damage', car.damage, 0.3, 0.8);
  check('hard landing bounced', bounced ? 1 : 0, 1, 1);
  check('never below ground', car.position.y, -0.001, 0.05, 'm');
}

// --- 6. wall collision -------------------------------------------------------------------
{
  // wall 2 m thick across the road 40 m ahead
  const wall = { minX: -20, maxX: 20, minY: 0, maxY: 10, minZ: -42, maxZ: -40 };
  const world = makeWorld([wall]);
  const car = new CarPhysics();
  car.reset(new Vector3(0, 0, 0), 0, world);
  car.velocity.set(0, 0, -20); // 72 km/h
  let minFront = Infinity, hitV = 0;
  run(car, world, 4, ctl({ throttle: 0.2 }), (t, c) => {
    minFront = Math.min(minFront, c.position.z - 2.3); // front bumper z
    if (c.impact > hitV) hitV = c.impact;
  });
  check('wall: impact speed registered', hitV, 15, 27, 'm/s');
  check('wall: no penetration (front >= wall)', minFront - wall.maxZ, -0.05, 1, 'm');
  check('wall: rebound speed after impact', Math.abs(car.velocity.z) , 0, 5, 'm/s');
  check('wall: damage', car.damage, 0.1, 0.4);
  // high-speed (tunnelling) test: 300 km/h into a 0.6 m pillar
  const pillar = { minX: -3, maxX: 3, minY: 0, maxY: 10, minZ: -60.6, maxZ: -60 };
  const w2 = makeWorld([pillar]);
  const c2 = new CarPhysics();
  c2.reset(new Vector3(0, 0, 0), 0, w2);
  c2.velocity.set(0, 0, -300 / 3.6);
  run(c2, w2, 2, ctl());
  check('300 km/h into 0.6 m pillar: no tunnelling', c2.position.z, -60 + 2.29, 10, 'm');
  // glancing hit: scrape along a wall
  const side = { minX: 1.5, maxX: 3, minY: 0, maxY: 10, minZ: -200, maxZ: 0 };
  const w3 = makeWorld([side]);
  const c3 = new CarPhysics();
  c3.reset(new Vector3(0, 0, 0), 0.08 /* slightly left */ * -1, w3);
  c3.velocity.set(0.08 * 30, 0, -30);
  run(c3, w3, 3, ctl({ throttle: 0.5 }));
  check('scrape: car stays outside side wall', c3.position.x, -50, 1.5 - 0.95, 'm');
  check('scrape: still moving forward', -c3.velocity.z, 10, 60, 'm/s');
  assertSane(c3, 'scrape');
}

// --- 7. handling: high-speed steering stability, low-speed turning, handbrake ------------
{
  const world = makeWorld();
  const car = new CarPhysics();
  car.reset(new Vector3(0, 0, 0), 0, world);
  car.velocity.set(0, 0, -300 / 3.6);
  let maxBeta = 0, maxLat = 0;
  run(car, world, 3, ctl({ throttle: 0.6, steer: 1 }), (t, c) => {
    const beta = Math.abs(Math.atan2(c.velocity.dot(new Vector3(1, 0, 0).applyQuaternion(c.quaternion)), Math.max(1, c.forwardSpeed)));
    maxBeta = Math.max(maxBeta, beta);
    maxLat = Math.max(maxLat, Math.abs(c.localAccel.x));
  });
  check('300 km/h full-right: sideslip stays small', maxBeta * 180 / Math.PI, 0, 8, 'deg');
  check('300 km/h full-right: lateral accel', maxLat / G, 0.8, 1.6, 'g');
  check('300 km/h full-right: turns right', car.heading < -0.3 ? 1 : 0, 1, 1);

  // low-speed full lock circle radius
  const c2 = new CarPhysics();
  c2.reset(new Vector3(0, 0, 0), 0, world);
  run(c2, world, 2, ctl({ throttle: 0.15 }));
  let wy = 0, sp = 0;
  run(c2, world, 3, (t, c) => ctl({ throttle: c.forwardSpeed < 6 ? 0.1 : 0, steer: -1 }), (t, c) => { wy = c.angularVelocity.y; sp = c.forwardSpeed; });
  const R = sp / Math.abs(wy);
  check('turning radius at ~6 m/s full lock', R, 3.5, 7, 'm');
  check('full lock left turns left (yaw rate +)', wy > 0 ? 1 : 0, 1, 1);

  // handbrake turn at 60 km/h -> big slide
  const c3 = new CarPhysics();
  c3.reset(new Vector3(0, 0, 0), 0, world);
  c3.velocity.set(0, 0, -60 / 3.6);
  let maxSlip = 0, maxB = 0;
  run(c3, world, 1.5, (t) => ctl({ steer: t < 1.0 ? 0.6 : 0, handbrake: t > 0.1 && t < 0.8 }), (t, c) => {
    maxSlip = Math.max(maxSlip, c.wheelSlip);
    const b = Math.abs(Math.atan2(c.velocity.dot(new Vector3(1, 0, 0).applyQuaternion(c.quaternion)), Math.max(1, Math.abs(c.forwardSpeed))));
    maxB = Math.max(maxB, b);
  });
  check('handbrake turn: sideslip angle', maxB * 180 / Math.PI, 15, 120, 'deg');
  check('handbrake turn: wheelSlip reported', maxSlip, 0.5, 1);
  assertSane(c3, 'handbrake');

  // reverse
  const c4 = new CarPhysics();
  c4.reset(new Vector3(0, 0, 0), 0, world);
  run(c4, world, 4, ctl({ brake: 1 }));
  check('reverse with brake held from stop', kmh(c4.forwardSpeed), -55, -30, 'km/h');
}

// --- 8. flight attitude, auto-level, empty tank glide, roof landing, auto-right -------
{
  const world = makeWorld();
  const car = new CarPhysics();
  car.reset(new Vector3(0, 0, 0), 0, world);
  run(car, world, 2, ctl({ thrust: 1 }));
  run(car, world, 1, ctl({ thrust: 1, roll: 1, steer: 1 }));
  const bankDeg = (c) => Math.atan2(-new Vector3(1, 0, 0).applyQuaternion(c.quaternion).y, new Vector3(0, 1, 0).applyQuaternion(c.quaternion).y) * 180 / Math.PI;
  const b1 = bankDeg(car);
  run(car, world, 3, ctl({ thrust: 0.5 }));
  const b2 = bankDeg(car);
  check('flight: roll input banks right', b1, 30, 180, 'deg');
  check('flight: auto-level after 3 s', Math.abs(b2), 0, 6, 'deg');
  // forward flight
  run(car, world, 4, ctl({ thrust: 0.47, throttle: 1 }));
  check('flight: forward thrust speed after 4 s', kmh(car.speed), 80, 240, 'km/h');

  // empty tank: thrust caps at 0.6 g, car comes down, never stuck
  const c2 = new CarPhysics();
  c2.reset(new Vector3(0, 0, 0), 0, world);
  run(c2, world, 3, ctl({ thrust: 1 }));
  c2.tank = 0;
  let landed = false, impactV = 0;
  run(c2, world, 25, ctl({ thrust: 1 }), (t, c) => { if (c.grounded) { landed = true; impactV = c.landingImpact; return false; } });
  check('empty tank: comes back down', landed ? 1 : 0, 1, 1);
  check('empty tank: touchdown speed (flare reserve)', impactV, 0, 8, 'm/s');

  // roof landing
  const roof = { minX: -10, maxX: 10, minY: 0, maxY: 15, minZ: -40, maxZ: -20 };
  const w3 = makeWorld([roof]);
  const c3 = new CarPhysics();
  c3.reset(new Vector3(0, 0, 0), 0, w3);
  // climb, translate over the roof centre (z = -30) with a simple P-controller, then let go
  const pilot = (t, c) => {
    const uDes = Math.max(-8, Math.min(8, (c.position.z + 30) * 0.6));
    const u = c.forwardSpeed;
    const over = Math.abs(c.position.z + 30) < 2 && Math.abs(u) < 1.5;
    return ctl({ thrust: !over && c.position.y < 20 ? 1 : 0, throttle: u < uDes ? 0.5 : 0, brake: u > uDes + 0.5 ? 0.5 : 0 });
  };
  run(c3, w3, 25, pilot, (t, c) => (c.grounded && c.mode === 'ground' && t > 2 ? false : undefined));
  run(c3, w3, 1, ctl());
  check('rooftop landing: resting height', c3.position.y, 14.95, 15.06, 'm');
  check('rooftop landing: grounded', c3.grounded ? 1 : 0, 1, 1);

  // upside-down auto-right
  const c4 = new CarPhysics();
  c4.reset(new Vector3(0, 0, 0), 0, world);
  c4.quaternion.setFromAxisAngle(new Vector3(0, 0, 1), Math.PI);
  c4.position.y = 2; c4.grounded = false; c4.airTime = 1;
  let uprightT = NaN;
  run(c4, world, 6, ctl(), (t, c) => {
    if (c.grounded && new Vector3(0, 1, 0).applyQuaternion(c.quaternion).y > 0.95 && Number.isNaN(uprightT)) uprightT = t;
  });
  check('upside down: auto-righted (time)', uprightT, 1.0, 4.0, 's');
  check('upside down: ends on wheels', c4.position.y, -0.01, 0.06, 'm');
}

// --- 9. g-force & no-world robustness ---------------------------------------------
{
  const car = new CarPhysics();
  car.reset(new Vector3(5, 0, 5), 1.0, null);
  run(car, null, 2, ctl({ throttle: 1, steer: 0.3 }));
  run(car, null, 2, ctl({ thrust: 1 }));
  run(car, null, 25, ctl({}));
  assertSane(car, 'null world');
  check('null world: back on the ground', car.position.y, -0.001, 0.06, 'm');
  const c2 = new CarPhysics(); c2.reset(new Vector3(), 0, null);
  run(c2, null, 1, ctl());
  check('g-force at rest', c2.gForce, 0.98, 1.02, 'g');
  // heading convention: heading +90deg faces -X
  const c3 = new CarPhysics(); c3.reset(new Vector3(), Math.PI / 2, null);
  run(c3, null, 1, ctl({ throttle: 1 }));
  check('heading +90deg drives toward -X', c3.position.x, -100, -2, 'm');
}

// --- 10. terrain: drive over a hill, crest jump, never below the surface ------------
{
  const hill = (x, z) => 4 * Math.exp(-((z + 150) ** 2) / 600);
  const world = { getGroundHeight: (x, z) => hill(x, z), queryColliders: () => [] };
  const car = new CarPhysics();
  car.reset(new Vector3(0, 0, 0), 0, world);
  let minClear = Infinity, minWheel = Infinity, flew = false, flight = false, maxPitch = 0, airSteps = 0;
  run(car, world, 12, (t, c) => ctl({ throttle: c.forwardSpeed < 45 ? 1 : 0.3 }), (t, c) => {
    minClear = Math.min(minClear, c.position.y - hill(c.position.x, c.position.z));
    for (let i = 0; i < 4; i++) { const w = c._wheelW[i]; minWheel = Math.min(minWheel, w.y - hill(w.x, w.z)); }
    if (!c.grounded) { flew = true; airSteps++; }
    if (c.mode === 'flight') flight = true;
    if (c.grounded) maxPitch = Math.max(maxPitch, Math.abs(Math.asin(new Vector3(0, 0, -1).applyQuaternion(c.quaternion).y)));
  });
  check('hill: body never below terrain (bump stops)', minClear, -0.09, 100, 'm');
  check('hill: wheels never below terrain', minWheel, -0.03, 100, 'm');
  check('hill: crest at ~160 km/h gives air time', flew ? 1 : 0, 1, 1);
  check('hill: a jump without thrust is not flight', flight ? 1 : 0, 0, 0);
  check('hill: airborne long enough to be a jump (steps)', airSteps, 6, 400);
  check('hill: grounded pitch follows slope', maxPitch * 180 / Math.PI, 3, 20, 'deg');
  check('hill: still on the road', Math.abs(car.position.x), 0, 1, 'm');
}

// --- 10b. suspension: settle, kerb, slope break, brake dive, roll, jump landing, moving box ------------
{
  const pitchOf = (c) => Math.asin(Math.max(-1, Math.min(1, new Vector3(0, 0, -1).applyQuaternion(c.quaternion).y)));
  const rollOf = (c) => { const r = new Vector3(1, 0, 0).applyQuaternion(c.quaternion), u = new Vector3(0, 1, 0).applyQuaternion(c.quaternion); return Math.atan2(-r.y, u.y); };
  const world = makeWorld();
  // rest: the body sits at the reference height with the wheels at their static compression, no jitter
  const c0 = new CarPhysics(); c0.reset(new Vector3(0, 0, 0), 0, world);
  let maxDy = 0, prevY = c0.position.y;
  run(c0, world, 3, ctl(), (t, c) => { maxDy = Math.max(maxDy, Math.abs(c.position.y - prevY)); prevY = c.position.y; });
  check('suspension: at rest the body stays put (|dy|/step)', maxDy * 1000, 0, 0.5, 'mm');
  check('suspension: static compression == sag', c0.wheels[0].compression / (c0.spec.suspensionTravel * c0.spec.suspensionSag), 0.97, 1.03);
  // heave ring-down: push the body down 4 cm and release; it must come back damped (no runaway, no snap)
  c0.position.y -= 0.04; c0.velocity.set(0, 0, 0);
  let tUp = NaN, overs = 0, minY = 0;
  run(c0, world, 2, ctl(), (t, c) => { if (Number.isNaN(tUp) && c.position.y > -0.004) tUp = t; if (c.position.y > 0.003) overs++; minY = Math.min(minY, c.position.y); });
  check('suspension: heave recovers within 0.4 s', tUp, 0.03, 0.4, 's');
  check('suspension: heave settles (overshoot steps)', overs, 0, 60);
  check('suspension: ends at rest height', Math.abs(c0.position.y), 0, 0.003, 'm');
  // kerb: a 12 cm box top under the right wheels at 30 km/h -> the body rises over several steps, no snap
  const kerb = { minX: 0.3, maxX: 30, minY: -1, maxY: 0.12, minZ: -400, maxZ: 0 };
  const wk = makeWorld([kerb]);
  const c1 = new CarPhysics(); c1.reset(new Vector3(-0.6, 0, 0), 0, wk);
  c1.velocity.set(0, 0, -30 / 3.6);
  let maxStepPitch = 0, maxStepRoll = 0, maxStepDy = 0, pRoll = rollOf(c1), pPitch = pitchOf(c1), pY = c1.position.y, hit = false, maxRoll = 0, impacts = 0;
  run(c1, wk, 4, (t) => ctl({ throttle: 0.1, steer: t < 1.2 ? 0.08 : 0 }), (t, c) => {
    const r = rollOf(c), pi = pitchOf(c);
    maxStepRoll = Math.max(maxStepRoll, Math.abs(r - pRoll)); maxStepPitch = Math.max(maxStepPitch, Math.abs(pi - pPitch));
    maxStepDy = Math.max(maxStepDy, Math.abs(c.position.y - pY)); maxRoll = Math.max(maxRoll, Math.abs(r));
    pRoll = r; pPitch = pi; pY = c.position.y;
    if (c.wheels[1].contact && c._supports[1] > 0.1) hit = true;
    if (c.impact > 0) impacts++;
  });
  check('kerb: right wheels climbed onto the 12 cm kerb', hit ? 1 : 0, 1, 1);
  check('kerb: body roll per step stays smooth', maxStepRoll * 180 / Math.PI, 0, 0.6, 'deg');
  check('kerb: body rise per step stays smooth', maxStepDy * 1000, 0, 15, 'mm');
  check('kerb: body leans with the kerb', maxRoll * 180 / Math.PI, 1.5, 8, 'deg');
  check('kerb: no impact registered (it is a step, not a wall)', impacts, 0, 0);
  // slope break: climbing 10 % onto a flat top at 50 km/h must not flicker grounded / airborne (the crest is a
  // 10 m vertical curve like a real street; the sharp kink variant below is a genuine small jump)
  // profile: flat, 10 m sag curve into a 10 % climb, 10 m crest curve onto a flat top (both vertical curves smooth)
  const crest = (z) => { const s = -(z + 60); if (s <= 0) return 0; if (s < 10) { const u = s / 10; return 0.5 * u * u; } if (s < 75) return 0.5 + 0.1 * (s - 10); if (s < 85) { const u = (s - 75) / 10; return 7 + (u - u * u * 0.5); } return 7.5; };
  const brk = { getGroundHeight: (x, z) => crest(z), queryColliders: () => [] };
  const c2 = new CarPhysics(); c2.reset(new Vector3(0, 0, 0), 0, brk);
  c2.velocity.set(0, 0, -50 / 3.6);
  let airSteps = 0, episodes = 0, wasG = true, minLoad = Infinity;
  run(c2, brk, 13, (t, c) => ctl({ throttle: c.forwardSpeed < 50 / 3.6 ? 0.6 : 0 }), (t, c) => { if (!c.grounded) airSteps++; if (c.grounded !== wasG && !c.grounded) episodes++; wasG = c.grounded; if (c.grounded && c.position.z < -70 && c.position.z > -150) minLoad = Math.min(minLoad, c.wheels[0].load + c.wheels[1].load + c.wheels[2].load + c.wheels[3].load); });
  check('slope break: drove over the crest', -c2.position.z, 150, 400, 'm');
  check('slope break: no grounded flicker (airborne episodes)', episodes, 0, 0);
  check('slope break: never airborne at 50 km/h', airSteps, 0, 0);
  check('slope break: wheels stay loaded (min total load / weight)', minLoad / (2000 * G), 0.3, 1.5);
  // brake dive and acceleration squat
  const c3 = new CarPhysics(); c3.reset(new Vector3(0, 0, 0), 0, world);
  c3.velocity.set(0, 0, -100 / 3.6);
  let dive = 0;
  run(c3, world, 1.0, ctl({ brake: 1 }), (t, c) => { dive = Math.min(dive, pitchOf(c)); });
  check('brake dive: nose down under full braking', -dive * 180 / Math.PI, 0.25, 3, 'deg');
  const c4 = new CarPhysics(); c4.reset(new Vector3(0, 0, 0), 0, world);
  let squat = 0;
  run(c4, world, 1.0, ctl({ throttle: 1 }), (t, c) => { squat = Math.max(squat, pitchOf(c)); });
  check('squat: nose up under full acceleration', squat * 180 / Math.PI, 0.1, 3, 'deg');
  // body roll in a steady corner leans OUT of the turn
  const c5 = new CarPhysics(); c5.reset(new Vector3(0, 0, 0), 0, world);
  c5.velocity.set(0, 0, -80 / 3.6);
  let rollR = 0;
  run(c5, world, 2.5, ctl({ throttle: 0.4, steer: 1 }), (t, c) => { if (t > 1.5) rollR += rollOf(c); });
  rollR /= 120; // mean over the last second (rollOf: right bank +; outward lean in a right turn = right side UP = negative)
  check('cornering: body rolls outward (right turn, deg)', -rollR * 180 / Math.PI, 0.3, 4, 'deg');
  // landing from a 0.6 m drop at speed: absorbed by the suspension, no damage, no bounce back into the air
  const ledge = { getGroundHeight: (x, z) => (z > -40 ? 0.6 : 0), queryColliders: () => [] };
  const c6 = new CarPhysics(); c6.reset(new Vector3(0, 0.6, 0), 0, ledge);
  c6.velocity.set(0, 0, -60 / 3.6);
  let landT = NaN, reAir = 0, landed = false, modes = new Set();
  run(c6, ledge, 4, ctl({ throttle: 0.2 }), (t, c) => { modes.add(c.mode); if (c.landedThisStep && c.position.z < -40 && Number.isNaN(landT)) landT = t; if (!Number.isNaN(landT) && t > landT + 0.05 && !c.grounded) reAir++; if (!Number.isNaN(landT) && c.grounded) landed = true; });
  check('ledge drop: lands', landed ? 1 : 0, 1, 1);
  check('ledge drop: suspension absorbs it (no re-launch steps)', reAir, 0, 3);
  check('ledge drop: no damage', c6.damage, 0, 0.001);
  check('ledge drop: a ballistic drop is not flight', modes.has('flight') ? 1 : 0, 0, 0);
  // arming: a lift tap in the air arms flight (hover assist slows the fall)
  const c7 = new CarPhysics(); c7.reset(new Vector3(0, 0, 0), 0, world);
  c7.position.y = 12; c7.grounded = false; c7.airTime = 0.5;
  run(c7, world, 0.3, (t) => ctl({ thrust: t < 0.1 ? 0.5 : 0 }));
  check('arming: lift tap mid-air -> flight mode', c7.mode === 'flight' ? 1 : 0, 1, 1);
  let land7 = 0; run(c7, world, 10, ctl(), (t, c) => { if (c.landedThisStep) { land7 = c.landingImpact; return false; } });
  check('arming: hover assist then lands softly', land7, 0, 3, 'm/s');
  // a moving traffic box shoves a parked car with its own velocity
  const box = { minX: -1, maxX: 1, minY: 0, maxY: 1.5, minZ: 3, maxZ: 7, kind: 'vehicle', vx: 0, vz: -6 };
  const wm = { getGroundHeight: () => 0, queryColliders: (x, z, r) => (Math.abs(box.minZ - z) < r + 10 ? [box] : []) };
  const c8 = new CarPhysics(); c8.reset(new Vector3(0, 0, 0), 0, wm);
  run(c8, wm, 2, ctl(), (t, c) => { box.minZ -= 6 * DT; box.maxZ -= 6 * DT; });
  check('traffic shove: pushed car moves with the box', -c8.velocity.z, 1.5, 7, 'm/s');
  assertSane(c8, 'traffic shove');
  // reset on a slope: no first-step snap
  const c9 = new CarPhysics(); c9.reset(new Vector3(0, 0, -100), 0, brk);
  const p0 = pitchOf(c9); run(c9, brk, DT, ctl());
  check('reset on a 10 % slope: body fitted to the slope (nose up)', p0 * 180 / Math.PI, 4.5, 6.5, 'deg');
  check('reset on a slope: first step does not snap', Math.abs(pitchOf(c9) - p0) * 180 / Math.PI, 0, 0.3, 'deg');
}

// --- 11. PhysicsWorld: same car on a PhysicsWorld, raycast / sphereCast ---------------
{
  // the same wall run on a PhysicsWorld must match the legacy world object exactly
  const wall = { minX: -20, maxX: 20, minY: 0, maxY: 10, minZ: -42, maxZ: -40 };
  const pw = new PhysicsWorld();
  pw.addStaticBox(wall.minX, wall.minY, wall.maxX, wall.maxY, wall.minZ, wall.maxZ, 1);
  const a = new VehicleController(ROADSTER_SPEC), b = new VehicleController(ROADSTER_SPEC);
  const legacy = makeWorld([wall]);
  a.reset(new Vector3(0, 0, 0), 0, legacy); b.reset(new Vector3(0, 0, 0), 0, pw);
  a.velocity.set(0, 0, -20); b.velocity.set(0, 0, -20);
  run(a, legacy, 4, ctl({ throttle: 0.2, steer: 0.1 })); run(b, pw, 4, ctl({ throttle: 0.2, steer: 0.1 }));
  check('PhysicsWorld == legacy world (wall run, |dpos|)', a.position.distanceTo(b.position) + Math.abs(a.damage - b.damage), 0, 1e-9, 'm');

  // raycast: wall face, heightfield, masks
  const hill = (x, z) => 4 * Math.exp(-((z + 150) ** 2) / 600);
  pw.setHeightfield({ heightAt: (x, z) => hill(x, z) });
  pw.addStaticBox(-3, 0, 3, 10, -80.6, -80, 3); // pillar further down the road
  const o = new Vector3(0, 1, 0), fwd = new Vector3(0, 0, -1);
  const h1 = pw.raycast(o, fwd, 200);
  check('raycast hits wall face at 40 m (+Z normal)', h1 && h1.box && h1.normal.z > 0.999 ? h1.distance : NaN, 39.999, 40.001, 'm');
  const h2 = pw.raycast(new Vector3(0, 50, -150), new Vector3(0.3, -1, 0), 200);
  check('raycast down onto heightfield (hill top 4 m)', h2 && h2.ground ? h2.point.y : NaN, 3.97, 4.01, 'm');
  const h3 = pw.raycast(o, fwd, 200, { mask: LAYER.ALL & ~layerOf(1) });
  check('raycast mask skips kind 1, hits pillar at 80 m', h3 && h3.box && h3.kind === 3 ? h3.distance : NaN, 79.999, 80.001, 'm');
  // sphereCast: a ray 0.4 m beside the pillar misses it; a 0.6 m sphere on the same line clips its edge
  // at 20 - sqrt(0.6^2 - 0.4^2) = 19.553 m (exact rounded-edge contact, not the inflated-box 19.4 m)
  const beside = new Vector3(3.4, 1, -60);
  const r4 = pw.raycast(beside, fwd, 40, { mask: layerOf(3) });
  const s4 = pw.sphereCast(beside, fwd, 0.6, 40, { mask: layerOf(3) });
  check('sphereCast r=0.6 clips pillar edge the ray misses', !r4 && s4 ? s4.distance : NaN, 19.548, 19.558, 'm');
  const s5 = pw.sphereCast(new Vector3(0, 8, 10), new Vector3(0, -1, 0), 0.5, 50);
  check('sphereCast r=0.5 down onto flat ground', s5 && s5.ground ? s5.distance : NaN, 7.49, 7.51, 'm');
  check('supportAt: wall top within step height', pw.supportAt(0, -41, 9.8), 9.99, 10.01, 'm');
}

// --- perf -------------------------------------------------------------------------
{
  const boxes = [];
  for (let i = 0; i < 40; i++) boxes.push({ minX: i * 7, maxX: i * 7 + 5, minY: 0, maxY: 30, minZ: -8, maxZ: -4 });
  const world = makeWorld(boxes);
  const car = new CarPhysics();
  car.reset(new Vector3(0, 0, 0), -Math.PI / 2, world);
  const t0 = performance.now();
  run(car, world, 60, ctl({ throttle: 1, steer: 0.1 }));
  const ms = performance.now() - t0;
  check('perf: 7200 steps (ms)', ms, 0, 1500, 'ms');
}

console.log(results.join('\n'));
console.log(failures ? `\n${failures} check(s) FAILED` : '\nAll car physics checks passed.');
process.exit(failures ? 1 : 0);
