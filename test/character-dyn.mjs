// Headless checks for the CharacterController's dynamics coupling (attachDynamics): pushing light bodies, heavy and
// kinematic bodies staying walls, pushForce / pushMass / mass, riding kinematic and dynamic bodies (translating,
// sinusoidal, vertical, rotating), frame changes on jumping, landing and walking off, determinism and allocation.
// No browser, no WebGL. Run: node test/character-dyn.mjs
import { Vector3, Quaternion } from 'three';
import { CharacterController } from '../src/physics/CharacterController.js';
import { PhysicsWorld } from '../src/physics/PhysicsWorld.js';
import { DynamicsWorld } from '../src/physics/dynamics/DynamicsWorld.js';
import { DynamicsClient } from '../src/physics/DynamicsClient.js';
import { box } from '../src/physics/dynamics/shapes.js';
import { attachToPhysicsWorld } from '../src/physics/dynamics/bridge.js';

const DT = 1 / 60;
let failures = 0, count = 0;
const results = [];
function check(name, value, lo, hi, unit = '') {
  const ok = Number.isFinite(value) && value >= lo && value <= hi;
  count++;
  if (!ok) failures++;
  results.push(`${ok ? 'PASS' : 'FAIL'}  ${name.padEnd(66)} ${fmt(value)} ${unit}   [${lo} .. ${hi}]`);
  return ok;
}
const fmt = (x) => (Number.isFinite(x) ? (Math.abs(x) < 1e-3 && x !== 0 ? x.toExponential(2) : x.toFixed(4)).padStart(10) : String(x).padStart(10));
const yes = (name, b) => check(name, b ? 1 : 0, 1, 1);
const note = (s) => results.push(`NOTE  ${s}`);

const pos = (dyn, id) => { const p = new Vector3(); dyn.getPose(id, p, null); return p; };
const vel = (dyn, id) => { const v = new Vector3(); dyn.getVelocity(id, v, null); return v; };
// a world (PhysicsWorld, flat ground at 0) with a dynamics world published into it
function scene({ client = false, dynOpts = {} } = {}) {
  const physics = new PhysicsWorld({ groundLevel: 0 });
  const dyn = client ? new DynamicsClient({ physics, worker: false, ...dynOpts }) : new DynamicsWorld({ physics, ...dynOpts });
  attachToPhysicsWorld(dyn, physics);
  return { physics, dyn };
}
// one frame: the character first (it pre-moves with its ride body), then the dynamics step
function frame(cc, physics, dyn, mx = 0, mz = 0, jump = false, sprint = false) {
  cc.step(DT, mx, mz, jump, sprint, physics);
  dyn.step(DT);
}
// a 0.6 m crate of `mass` 1.2 m ahead (+X) of a character at the origin, settled on the ground
function crateScene(mass = 20, opts = {}, { attach = true, client = false } = {}) {
  const s = scene({ client });
  s.crate = s.dyn.addBody({ shape: box(0.3, 0.3, 0.3), position: [1.2, 0.3, 0], mass });
  for (let i = 0; i < 30; i++) s.dyn.step(DT);
  s.cc = new CharacterController();
  if (attach) s.cc.attachDynamics(s.dyn, opts);
  s.cc.teleport(0, 0, 0);
  s.p0 = pos(s.dyn, s.crate);
  return s;
}

// ---- 1. pushing a light crate ----------------------------------------------------------------------------------------
{
  const { physics, dyn, crate, cc, p0 } = crateScene(20);
  const cap = cc.pushForce * DT;
  let firstJ = NaN, firstDrop = NaN, pushed = 0, contactSteps = 0, maxJ = 0, maxOverlap = -Infinity, airborne = 0, rode = 0, vBefore = 0;
  const B = dyn.bodies;
  for (let i = 0; i < 120; i++) {
    vBefore = cc.velocity.x;
    cc.step(DT, 1, 0, false, false, physics);
    if (cc.pushedBody === crate && Number.isNaN(firstJ)) { firstJ = cc.pushImpulse; firstDrop = vBefore - cc.velocity.x; }
    if (!Number.isNaN(firstJ)) { contactSteps++; if (cc.pushedBody === crate) pushed++; }
    maxJ = Math.max(maxJ, cc.pushImpulse);
    if (!cc.grounded) airborne++;
    if (cc.rideBody >= 0 || cc.platformVelocity.lengthSq() > 0) rode++;
    dyn.step(DT);
    maxOverlap = Math.max(maxOverlap, cc.position.x + cc.radius - B.aabb[crate * 6]);
  }
  const p1 = pos(dyn, crate), v1 = vel(dyn, crate), q = new Quaternion();
  dyn.getPose(crate, null, q);
  check('push 20 kg: the crate moved along the push (2 s)', p1.x - p0.x, 0.5, 10, 'm');
  check('push 20 kg: the character advanced', cc.position.x, 1, 10, 'm');
  check('push 20 kg: crate velocity along the push direction', v1.x, 0.5, 3, 'm/s');
  check('push 20 kg: no sideways drift of the crate', Math.abs(p1.z - p0.z), 0, 0.01, 'm');
  check('push 20 kg: the crate is not tipped (rotation angle)', 2 * Math.acos(Math.min(1, Math.abs(q.w))), 0, 0.05, 'rad');
  check('push 20 kg: the crate stays on the ground (y)', p1.y, 0.29, 0.31, 'm');
  check('push 20 kg: pushedBody = the crate in steps after contact (share)', pushed / contactSteps, 0.9, 1);
  check('push 20 kg: first contact impulse = pushForce dt (capped)', firstJ, cap - 1e-12, cap + 1e-12, 'N s');
  check('push 20 kg: impulse never above pushForce dt', maxJ, 0, cap + 1e-12, 'N s');
  check('push 20 kg: first contact, velocity drop = J / mass (80 kg)', firstDrop, firstJ / 80 - 1e-9, firstJ / 80 + 1e-9, 'm/s');
  check('push 20 kg: overlap with the crate AABB after each dynamics step', maxOverlap, -1, 0.05, 'm');
  check('push 20 kg: the character moves at the crate\'s pace (|speed - vx|)', Math.abs(cc.speed - v1.x), 0, 0.1, 'm/s');
  check('push 20 kg: grounded throughout (airborne steps)', airborne, 0, 0);
  check('push 20 kg: no ride on the ground (rideBody / platformVelocity)', rode, 0, 0);
}

// ---- 2. heavy and kinematic bodies are walls; pushForce / pushMass / mass -------------------------------------------------
const FACE = 1.2 - 0.3 - 0.01; // crate's -X AABB face (margin 0.01)
{
  const { physics, dyn, crate, cc, p0 } = crateScene(500);
  let pushedEver = 0;
  for (let i = 0; i < 120; i++) { frame(cc, physics, dyn, 1, 0); if (cc.pushedBody >= 0) pushedEver++; }
  check('500 kg crate: not moved (a wall above pushMass)', pos(dyn, crate).distanceTo(p0), 0, 1e-3, 'm');
  check('500 kg crate: the character stops at its face', cc.position.x, FACE - cc.radius - 0.01, FACE - cc.radius + 1e-6, 'm');
  check('500 kg crate: speed at the end', cc.speed, 0, 1e-9, 'm/s');
  check('500 kg crate: never reported as pushed', pushedEver, 0, 0);
  yes('500 kg crate: blocked, wallNormal -X', cc.blocked && cc.wallNormal.x === -1);
}
{
  const s = scene();
  const k = s.dyn.addBody({ type: 'kinematic', shape: box(0.3, 0.3, 0.3), position: [1.2, 0.3, 0] });
  const cc = new CharacterController().attachDynamics(s.dyn); cc.teleport(0, 0, 0);
  for (let i = 0; i < 90; i++) frame(cc, s.physics, s.dyn, 1, 0);
  check('kinematic box: a wall (character stops at its face)', cc.position.x, FACE - cc.radius - 0.01, FACE - cc.radius + 1e-6, 'm');
  check('kinematic box: not moved, never pushed', pos(s.dyn, k).distanceTo(new Vector3(1.2, 0.3, 0)) + (cc.pushedBody >= 0 ? 1 : 0), 0, 1e-12);
}
const run = (mass, opts, steps = 120) => {
  const s = crateScene(mass, opts);
  let maxJ = 0, held = 0;
  for (let i = 0; i < steps; i++) { frame(s.cc, s.physics, s.dyn, 1, 0); maxJ = Math.max(maxJ, s.cc.pushImpulse); }
  return { ...s, moved: pos(s.dyn, s.crate).x - s.p0.x, maxJ, held };
};
{
  const full = run(20, {}), weak = run(20, { pushForce: 150 });
  check('pushForce 150 N: the crate moves less than at 600 N (ratio)', weak.moved / full.moved, 0.05, 0.95);
  check('pushForce 150 N: impulse <= 150 N x dt', weak.maxJ, 0, 150 * DT + 1e-12, 'N s');
  const stuck = run(20, { pushForce: 50 });   // below the crate's static friction (0.6 x 20 kg x 9.81 = 118 N)
  check('pushForce 50 N (< static friction): the crate stays', Math.abs(stuck.moved), 0, 1e-3, 'm');
  check('pushForce 50 N: the character is held (speed)', stuck.cc.speed, 0, 0.05, 'm/s');
  const zero = run(20, { pushForce: 0 });
  check('pushForce 0: a wall (crate stays, nothing pushed, speed 0)', Math.abs(zero.moved) + zero.maxJ + zero.cc.speed + (zero.cc.pushedBody >= 0 ? 1 : 0), 0, 1e-3);
  const light = run(20, { pushMass: 10 });
  check('pushMass 10 kg: a 20 kg crate is a wall', Math.abs(light.moved) + light.maxJ, 0, 1e-3);
  // a very heavy character loses nothing to the crate on contact
  const s = crateScene(20, { mass: 1e6 });
  let drop = NaN;
  for (let i = 0; i < 60 && Number.isNaN(drop); i++) { const v0 = s.cc.velocity.x; frame(s.cc, s.physics, s.dyn, 1, 0); if (s.cc.pushedBody >= 0) drop = v0 - s.cc.velocity.x; }
  check('mass 1e6 kg: first-contact velocity drop', drop, 0, 1e-4, 'm/s');
}

// ---- 3. without attachDynamics: today's behaviour --------------------------------------------------------------------------
{
  const { physics, dyn, crate, cc, p0 } = crateScene(20, {}, { attach: false });
  for (let i = 0; i < 120; i++) frame(cc, physics, dyn, 1, 0);
  check('not attached: the 20 kg crate is a wall (not moved)', pos(dyn, crate).distanceTo(p0), 0, 1e-3, 'm');
  check('not attached: the character stops at its face', cc.position.x, FACE - cc.radius - 0.01, FACE - cc.radius + 1e-6, 'm');
  yes('not attached: dynamics outputs inert (null, -1, 0)', cc.dynamics === null && cc.pushedBody === -1 && cc.rideBody === -1 && cc.platformVelocity.lengthSq() === 0 && cc.pushImpulse === 0);
}
{
  // attached to a world with no bodies = plain: bit-identical over a scripted run among static boxes and kerbs
  const physics = new PhysicsWorld({ groundLevel: 0 });
  for (let i = 0; i < 5; i++) for (let k = 0; k < 5; k++) if (i !== 2 || k !== 2) physics.addStaticBox(-9 + i * 4, 0, -7.5 + i * 4, [0.2, 0.3, 1.0, 2.0, 0.15][(i + 2 * k) % 5], -9 + k * 4, -7.5 + k * 4);
  const dyn = new DynamicsWorld({ physics });
  attachToPhysicsWorld(dyn, physics);
  const a = new CharacterController(), b = new CharacterController().attachDynamics(dyn), c = new CharacterController().attachDynamics(dyn).detachDynamics();
  const f64 = new Float64Array(6), u32 = new Uint32Array(f64.buffer);
  const hash = (h, k) => { f64[0] = k.position.x; f64[1] = k.position.y; f64[2] = k.position.z; f64[3] = k.velocity.x; f64[4] = k.velocity.y; f64[5] = k.velocity.z; for (let i = 0; i < 12; i++) h = Math.imul(h ^ u32[i], 16777619) >>> 0; return h; };
  let ha = 2166136261, hb = ha, hc = ha, seed = 99, mx = 0, mz = 0;
  const rnd = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; };
  for (let i = 0; i < 1200; i++) {
    if (i % 60 === 0) { mx = rnd() * 2 - 1; mz = rnd() * 2 - 1; }
    const jump = rnd() < 0.01, sprint = i % 300 < 150;
    for (const k of [a, b, c]) k.step(DT, mx, mz, jump, sprint, physics);
    dyn.step(DT);
    ha = hash(ha, a); hb = hash(hb, b); hc = hash(hc, c);
  }
  yes('attached to a world without bodies: bit-identical to a plain controller', ha === hb);
  yes('detachDynamics: bit-identical to a plain controller', ha === hc);
}

// ---- 4. riding a kinematic platform -----------------------------------------------------------------------------------------
function platformScene({ vel = [1, 0, 0], ang = [0, 0, 0], kind = 'rigid', opts = {}, at = [0, 0], client = false, half = [1.5, 0.1, 1.5], y = 0.5 } = {}) {
  const s = scene({ client });
  s.plat = s.dyn.addBody({ type: 'kinematic', shape: box(...half), position: [0, y, 0], kind });
  s.dyn.setVelocity(s.plat, vel[0], vel[1], vel[2], ang[0], ang[1], ang[2]);
  s.dyn.step(DT);
  s.top = () => s.dyn.bodies.aabb[s.plat * 6 + 4];
  s.cc = new CharacterController().attachDynamics(s.dyn, opts);
  s.cc.teleport(pos(s.dyn, s.plat).x + at[0], s.top(), at[1]);
  return s;
}
{
  const { physics, dyn, plat, cc, top } = platformScene();
  const x0 = cc.position.x;
  let notGrounded = 0, maxFeetErr = 0, wrongRide = 0;
  for (let i = 0; i < 120; i++) {
    frame(cc, physics, dyn);
    if (!cc.grounded) notGrounded++;
    if (cc.rideBody !== plat) wrongRide++;
    maxFeetErr = Math.max(maxFeetErr, Math.abs(cc.position.y - top()));
  }
  check('platform 1 m/s, standing 2 s: carried distance / 2 m', (cc.position.x - x0) / 2, 0.95, 1.05);
  check('platform: platformVelocity = (1, 0, 0)', Math.abs(cc.platformVelocity.x - 1) + Math.abs(cc.platformVelocity.y) + Math.abs(cc.platformVelocity.z), 0, 1e-9, 'm/s');
  check('platform: rideBody = the platform every step (misses)', wrongRide, 0, 0);
  check('platform: grounded every step (misses)', notGrounded, 0, 0);
  check('platform: feet on its top after every dynamics step', maxFeetErr, 0, 1e-9, 'm');
  check('platform: relative speed while standing', cc.speed, 0, 1e-9, 'm/s');
  yes('platform: groundKind is the body\'s kind (rigid)', cc.groundKind === 'rigid');

  // ---- 5. jump off it: the platform's horizontal velocity is kept, it lands back on the platform ------------------------
  const off0 = cc.position.x - pos(dyn, plat).x;
  let jumped = false, airVx = [], airRide = 0, landed = false, landSpeed = NaN, offLand = NaN;
  for (let i = 0; i < 90 && !landed; i++) {
    cc.step(DT, 0, 0, i === 0, false, physics);
    if (i === 0) jumped = cc.jumpedThisStep;
    if (!cc.grounded) { airVx.push(cc.velocity.x); if (cc.rideBody >= 0 || cc.platformVelocity.lengthSq() > 0) airRide++; }
    dyn.step(DT);
    if (i > 0 && cc.landedThisStep) { landed = true; landSpeed = cc.landingSpeed; offLand = cc.position.x - pos(dyn, plat).x; }
  }
  yes('jump from the platform: jumpedThisStep', jumped);
  check('jump: horizontal velocity in the air = the platform\'s (max |vx - 1|)', Math.max(...airVx.map((v) => Math.abs(v - 1))), 0, 1e-9, 'm/s');
  check('jump: steps in the air', airVx.length, 30, 80);
  check('jump: no ride / platformVelocity while airborne (steps)', airRide, 0, 0);
  yes('jump: lands back on the platform (rideBody)', landed && cc.rideBody === plat);
  check('jump: lands where it took off on the platform (offset change)', Math.abs(offLand - off0), 0, 0.01, 'm');
  check('jump: landing speed / jumpSpeed', landSpeed / cc.jumpSpeed, 0.98, 1.02);
}
{
  // sinusoidal kinematic target and a vertical elevator: the feet follow exactly (re-placed every step)
  const s = platformScene({ vel: [0, 0, 0] });
  const off0 = s.cc.position.x - pos(s.dyn, s.plat).x;
  let maxErr = 0;
  for (let i = 0; i < 180; i++) {
    const t = (i + 1) * DT;
    s.dyn.setKinematicTarget(s.plat, Math.sin(2 * t), 0.5, 0, 0, 0, 0, 1);
    frame(s.cc, s.physics, s.dyn);
    if (i > 0) maxErr = Math.max(maxErr, Math.abs(s.cc.position.x - pos(s.dyn, s.plat).x - off0));
  }
  check('sinusoidal platform (setKinematicTarget): offset error after each step', maxErr, 0, 2e-3, 'm');
  s.cc.step(DT, 0, 0, false, false, s.physics);   // the start-of-step re-placement removes the prediction error
  check('sinusoidal platform: no drift on the platform after 3 s', Math.abs(s.cc.position.x - s.cc.platformVelocity.x * DT - pos(s.dyn, s.plat).x - off0), 0, 1e-9, 'm');
  const el = platformScene({ vel: [0, 0, 0], y: 0.9 });
  let air = 0, feet = 0, vyErr = 0, y1 = 0.9, y2 = 0.9;   // the targets of the last two steps
  for (let i = 0; i < 240; i++) {
    const t = (i + 1) * DT, y = 0.9 + 0.8 * Math.sin(2 * t);
    el.dyn.setKinematicTarget(el.plat, 0, y, 0, 0, 0, 0, 1);
    el.cc.step(DT, 0, 0, false, false, el.physics);
    if (i > 1) vyErr = Math.max(vyErr, Math.abs(el.cc.platformVelocity.y - (y1 - y2) / DT)); // the last step's motion
    y2 = y1; y1 = y;
    el.dyn.step(DT);
    if (i > 0) { if (!el.cc.grounded) air++; feet = Math.max(feet, Math.abs(el.cc.position.y - el.top())); }
  }
  check('elevator (up to 1.6 m/s up and down): airborne steps', air, 0, 0);
  check('elevator: feet vs platform top after each step (prediction error)', feet, 0, 2e-3, 'm');
  check('elevator: platformVelocity.y = the last step\'s target motion / dt', vyErr, 0, 1e-9, 'm/s');
}
{
  // a platform turning at 1 rad/s about +Y: carried around its axis (rotation included)
  const s = platformScene({ vel: [0, 0, 0], ang: [0, 1, 0], half: [2, 0.1, 2], at: [1, 0] });
  for (let i = 0; i < 90; i++) frame(s.cc, s.physics, s.dyn);
  const r = Math.hypot(s.cc.position.x, s.cc.position.z), a = Math.atan2(-s.cc.position.z, s.cc.position.x);
  check('rotating platform: radius kept (1 m)', r, 1 - 1e-3, 1 + 1e-3, 'm');
  check('rotating platform: turned with it (angle after 1.5 s)', a, 1.5 - 0.02, 1.5 + 0.02, 'rad');
  check('rotating platform: |platformVelocity| = w r', s.cc.platformVelocity.length(), 1 - 1e-3, 1 + 1e-3, 'm/s');
}
{
  // walking on a moving platform: speed is relative, the world velocity adds up
  const s = platformScene({ half: [4, 0.1, 1.5], at: [-2, 0] });
  for (let i = 0; i < 30; i++) frame(s.cc, s.physics, s.dyn, 1, 0);
  const x0 = s.cc.position.x;
  for (let i = 0; i < 60; i++) frame(s.cc, s.physics, s.dyn, 1, 0);
  check('walking on the platform: relative speed = walk', s.cc.speed / s.cc.walk, 1 - 1e-9, 1 + 1e-9);
  check('walking on the platform: world displacement in 1 s (walk + 1 m/s)', s.cc.position.x - x0, 3 - 1e-6, 3 + 1e-6, 'm');
}
{
  // landing on a moving platform (velocity becomes relative), then walking off it (velocity becomes world again)
  const s = platformScene({ y: 0.2 });
  s.cc.teleport(pos(s.dyn, s.plat).x, 1.5, 0);
  let landVx = NaN, leftVx = NaN, groundAfter = false;
  for (let i = 0; i < 300; i++) {
    const mz = i > 90 ? 1 : 0;
    const wasRide = s.cc.rideBody;
    frame(s.cc, s.physics, s.dyn, 0, mz);
    if (s.cc.landedThisStep && Number.isNaN(landVx)) landVx = s.cc.velocity.x;
    if (wasRide >= 0 && s.cc.rideBody < 0 && Number.isNaN(leftVx)) leftVx = s.cc.velocity.x;
    if (!Number.isNaN(leftVx) && s.cc.grounded && s.cc.groundKind === 'ground') groundAfter = true;
    if (i === 80) check('landed on the platform: at rest on it after 1 s (relative speed)', s.cc.speed, 0, 1e-9, 'm/s');
  }
  check('landing on a 1 m/s platform: velocity becomes relative (vx)', landVx, -1 - 1e-9, -1 + 1e-9, 'm/s');
  check('walking off it: the platform\'s velocity carries (vx)', leftVx, 1 - 1e-9, 1 + 1e-9, 'm/s');
  yes('walking off it: lands on the ground', groundAfter);
}
{
  // ride kinds / ride off / teleport
  for (const [label, o] of [['kind not in rideKinds', { kind: 'conveyor' }], ['ride: false', { opts: { ride: false } }]]) {
    const s = platformScene(o);
    const x0 = s.cc.position.x;
    let ride = 0;
    for (let i = 0; i < 20; i++) { frame(s.cc, s.physics, s.dyn); if (s.cc.rideBody >= 0) ride++; }
    check(`${label}: not carried (world displacement, 0.33 s)`, Math.abs(s.cc.position.x - x0) + ride, 0, 1e-9, 'm');
  }
  const s = platformScene();
  for (let i = 0; i < 10; i++) frame(s.cc, s.physics, s.dyn);
  s.cc.teleport(5, 0, 5);
  yes('teleport: leaves the ride (rideBody -1, platformVelocity 0)', s.cc.rideBody === -1 && s.cc.platformVelocity.lengthSq() === 0);
  // standing on a dynamic crate that is being pushed by a force: carried with it
  const c = scene();
  const crate = c.dyn.addBody({ shape: box(0.6, 0.25, 0.6), position: [0, 0.25, 0], mass: 50, friction: 0.05 });
  for (let i = 0; i < 30; i++) c.dyn.step(DT);
  const cc = new CharacterController().attachDynamics(c.dyn);
  cc.teleport(0, c.dyn.bodies.aabb[crate * 6 + 4], 0);
  for (let i = 0; i < 120; i++) { c.dyn.applyForce(crate, 100, 0, 0); frame(cc, c.physics, c.dyn); }
  check('riding a dynamic crate pushed by a force: stays on its spot', Math.abs(cc.position.x - pos(c.dyn, crate).x), 0, 2e-3, 'm');
  check('riding a dynamic crate: carried (crate moved)', pos(c.dyn, crate).x, 1, 10, 'm');
}

// ---- 5b. probes: a removed / reused ride body, a pinned pushable, two pushables, friction vs pushForce ------------------------
{
  // the ride body removed, its id reused by another body far away (a new shape object, and the same shared one)
  for (const shared of [false, true]) {
    const s = platformScene();
    for (let i = 0; i < 30; i++) frame(s.cc, s.physics, s.dyn);
    const shape = s.dyn.bodies.shape[s.plat];
    s.dyn.removeBody(s.plat);
    const other = s.dyn.addBody({ type: 'kinematic', shape: shared ? shape : box(1.5, 0.1, 1.5), position: [50, 5, 30] });
    const p0 = s.cc.position.clone();
    frame(s.cc, s.physics, s.dyn);
    const label = shared ? 'same shape object' : 'another shape';
    check(`reused ride id (${label}): not re-placed onto the new body`, s.cc.position.distanceTo(p0) + (other === s.plat ? 0 : 100), 0, 0.1, 'm');
    yes(`reused ride id (${label}): the ride ends`, s.cc.rideBody === -1 && s.cc.platformVelocity.lengthSq() === 0);
    check(`reused ride id (${label}): the platform's velocity is kept (vx)`, s.cc.velocity.x, 1 - 1e-9, 1 + 1e-9, 'm/s');
  }
  const g = platformScene();
  for (let i = 0; i < 30; i++) frame(g.cc, g.physics, g.dyn);
  g.dyn.removeBody(g.plat);
  frame(g.cc, g.physics, g.dyn);
  check('removed ride body: the platform\'s velocity is kept (vx), falling', g.cc.velocity.x - (g.cc.grounded ? 10 : 0), 1 - 1e-9, 1 + 1e-9, 'm/s');
  const d = platformScene();
  for (let i = 0; i < 30; i++) frame(d.cc, d.physics, d.dyn);
  d.cc.detachDynamics();
  check('detachDynamics while riding: the platform\'s velocity is kept (vx)', d.cc.velocity.x, 1 - 1e-9, 1 + 1e-9, 'm/s');

  // a 20 kg crate pinned against a static wall: it stays, the character is held, its overlap stays within one step's
  // predicted give (J / m dt)
  const w = crateScene(20);
  w.physics.addStaticBox(1.5, 0, 3, 2, -2, 2);       // touching the crate's +X face
  let excess = -Infinity, maxOver = 0;
  for (let i = 0; i < 120; i++) {
    const vb = w.dyn.bodies.vel[w.crate * 3];       // the crate's velocity the step sees (its record's vx)
    w.cc.step(DT, 1, 0, false, false, w.physics);
    const over = w.cc.position.x + w.cc.radius - w.dyn.bodies.aabb[w.crate * 6];
    const away = vb + (w.cc.pushedBody === w.crate ? w.cc.pushImpulse / 20 : 0);   // its velocity after the push
    excess = Math.max(excess, over - (away > 0 ? away : 0) * DT);
    maxOver = Math.max(maxOver, over);
    w.dyn.step(DT);
  }
  check('pinned crate: not moved through the wall', pos(w.dyn, w.crate).x - w.p0.x, -1e-3, 0.03, 'm');
  check('pinned crate: the character is held (speed)', w.cc.speed, 0, 0.05, 'm/s');
  check('pinned crate: overlap with its AABB <= away x dt (excess)', excess, -1, 1e-9, 'm');
  check('pinned crate: overlap with its AABB (max)', maxOver, -1, 0.01, 'm');

  // two crates side by side, both touched: both move, neither tips
  const t = scene();
  const c1 = t.dyn.addBody({ shape: box(0.3, 0.3, 0.3), position: [1.2, 0.3, 0.35], mass: 20 });
  const c2 = t.dyn.addBody({ shape: box(0.3, 0.3, 0.3), position: [1.2, 0.3, -0.35], mass: 20 });
  for (let i = 0; i < 30; i++) t.dyn.step(DT);
  const k = new CharacterController().attachDynamics(t.dyn); k.teleport(0, 0, 0);
  const seen = new Set();
  for (let i = 0; i < 120; i++) { frame(k, t.physics, t.dyn, 1, 0); if (k.pushedBody >= 0) seen.add(k.pushedBody); }
  const q1 = new Quaternion(), q2 = new Quaternion();
  t.dyn.getPose(c1, null, q1); t.dyn.getPose(c2, null, q2);
  check('two crates: both pushed forward (min dx)', Math.min(pos(t.dyn, c1).x, pos(t.dyn, c2).x) - 1.2, 0.3, 10, 'm');
  const up1 = new Vector3(0, 1, 0).applyQuaternion(q1).y, up2 = new Vector3(0, 1, 0).applyQuaternion(q2).y;
  check('two crates: neither tipped over (min up.y; turning about Y is free)', Math.min(up1, up2), 0.99, 1);
  check('two crates: the character advanced', k.position.x, 1, 10, 'm');
  yes('two crates: pushedBody named one of them', seen.size >= 1 && [...seen].every((id) => id === c1 || id === c2));

  // static friction against pushForce: 120 kg (mu 0.6: 706 N > 600 N) holds, 20 kg slides; pushForce Infinity is kept
  const heavy = run(120, {});
  check('120 kg crate (friction 706 N > pushForce 600 N): holds', Math.abs(heavy.moved), 0, 5e-3, 'm');
  check('120 kg crate: the character is held (speed)', heavy.cc.speed, 0, 0.05, 'm/s');
  const inf = crateScene(20, { pushForce: Infinity });
  let firstJ = NaN, v0 = 0;
  for (let i = 0; i < 90 && Number.isNaN(firstJ); i++) { v0 = inf.cc.velocity.x; frame(inf.cc, inf.physics, inf.dyn, 1, 0); if (inf.cc.pushedBody >= 0) firstJ = inf.cc.pushImpulse; }
  check('pushForce Infinity: kept, first contact J = m v (uncapped)', firstJ - (inf.cc.pushForce === Infinity ? 20 * v0 : -1e9), -1e-9, 1e-9, 'N s');
}

// ---- 6. other worlds and kinds ----------------------------------------------------------------------------------------------
{
  const s = crateScene(20, {}, { client: true });
  for (let i = 0; i < 120; i++) frame(s.cc, s.physics, s.dyn, 1, 0);
  check('in-thread DynamicsClient: pushes the crate', pos(s.dyn, s.crate).x - s.p0.x, 0.5, 10, 'm');
  const p = platformScene({ client: true });
  const x0 = p.cc.position.x;
  for (let i = 0; i < 120; i++) frame(p.cc, p.physics, p.dyn);
  check('in-thread DynamicsClient: carried by the platform (/ 2 m)', (p.cc.position.x - x0) / 2, 0.95, 1.05);
  for (const flag of [false, true]) {
    const v = scene();
    const car = v.dyn.addBody({ shape: box(0.3, 0.3, 0.3), position: [1.2, 0.3, 0], mass: 60, kind: 'vehicle' });
    for (let i = 0; i < 30; i++) v.dyn.step(DT);
    const k = new CharacterController({ pushVehicles: flag }).attachDynamics(v.dyn);
    k.teleport(0, 0, 0);
    let pushed = 0;
    for (let i = 0; i < 90; i++) { frame(k, v.physics, v.dyn, 1, 0); if (k.pushedBody === car) pushed++; }
    const moved = pos(v.dyn, car).x - 1.2;
    if (flag) check('pushVehicles: a 60 kg vehicle-kind body is pushed', moved, 0.1, 10, 'm');
    else check('vehicle-kind body without pushVehicles: a wall (moved / pushed steps)', Math.abs(moved) + pushed, 0, 1e-3);
  }
}

// ---- 7. determinism ---------------------------------------------------------------------------------------------------------
{
  const f64 = new Float64Array(6), u32 = new Uint32Array(f64.buffer);
  const once = () => {
    const s = scene();
    const crate = s.dyn.addBody({ shape: box(0.3, 0.3, 0.3), position: [1.2, 0.3, 0], mass: 20 });
    const plat = s.dyn.addBody({ type: 'kinematic', shape: box(1.5, 0.1, 1.5), position: [4, 0.2, 3] });
    s.dyn.setVelocity(plat, 0.5, 0, 0, 0, 0.4, 0);
    const cc = new CharacterController().attachDynamics(s.dyn);
    const p = new Vector3();
    let h = 2166136261;
    for (let i = 0; i < 600; i++) {
      const t = i * DT, mx = t < 4 ? 1 : 0.3, mz = t < 4 ? 0.1 : 1;
      cc.step(DT, mx, mz, i % 150 === 149, i % 200 < 50, s.physics);
      s.dyn.step(DT);
      s.dyn.getPose(crate, p, null);
      f64[0] = cc.position.x; f64[1] = cc.position.y; f64[2] = cc.position.z; f64[3] = p.x; f64[4] = p.y; f64[5] = p.z;
      for (let k = 0; k < 12; k++) h = Math.imul(h ^ u32[k], 16777619) >>> 0;
    }
    return { h, moved: pos(s.dyn, crate).x - 1.2, rides: cc.rideBody };
  };
  const a = once(), b = once();
  yes(`determinism: pose hash equal across two runs (${a.h.toString(16)})`, a.h === b.h);
  // golden: the character and a DynamicsWorld crate (pushed, a platform ridden, jumps); an intended change to either
  // (or to the solver's contacts) updates it. Only + - * / sqrt in both, so it is the same on every platform.
  const GOLDEN = 0x56d4a1c2;
  check(`determinism: golden hash (${GOLDEN.toString(16)}; got ${a.h.toString(16)})`, a.h === GOLDEN ? 1 : 0, 1, 1);
  check('determinism run: the crate was pushed', a.moved, 0.3, 20, 'm');
}

// ---- 8. allocation ----------------------------------------------------------------------------------------------------------
// Every allocation over 1000+ steps is sampled (V8's sampling heap profiler, garbage included) and attributed to the
// function that made it. The plain step already allocates a few hundred bytes per step: V8 boxes numbers passed to
// and returned from functions it did not inline (world queries, the bridge provider), and which ones it inlines shifts
// a little when the step grows. The coupling's own functions must allocate nothing.
{
  const COUPLING = new Set(['_carry', '_rideEnd', '_pushSlot', '_applyPushes', 'readBody', 'fromLocal', 'toLocal', 'pointVelocity', 'ownRecord', 'aliveBody', '_resetDynamics']);
  let inspector = null;
  try { inspector = await import('node:inspector/promises'); } catch { inspector = null; }
  // pushing: a stand-in world whose applyImpulse only counts (so the crate is pushed every step and DynamicsWorld's
  // own impulse code is not measured), and the real one; riding: a platform turning and translating
  const own = crateScene(20, {}, { attach: false });
  let impulses = 0;
  own.cc.attachDynamics({ bodies: own.dyn.bodies, applyForce(id, force) { if (force.x > 0) impulses++; } });
  const real = crateScene(20);
  const plain = crateScene(20, {}, { attach: false });
  const ride = platformScene({ vel: [0.7, 0, 0.2], ang: [0, 0.3, 0], at: [0.3137, 0.2719] });
  for (let i = 0; i < 40; i++) { for (const s of [own, real, plain]) frame(s.cc, s.physics, s.dyn, 1, 0); frame(ride.cc, ride.physics, ride.dyn); }
  const V = real.dyn.bodies.vel, c3 = real.crate * 3;
  const runs = {
    plain: () => plain.cc.step(DT, 1, 0, false, false, plain.physics),
    pushing: () => own.cc.step(DT, 1, 0, false, false, own.physics),
    'pushing (DynamicsWorld)': () => { V[c3] = 0; V[c3 + 1] = 0; V[c3 + 2] = 0; real.cc.step(DT, 1, 0, false, false, real.physics); },
    riding: () => ride.cc.step(DT, 0, 0, false, false, ride.physics),
  };
  const N = 3000;
  for (const k of Object.keys(runs)) for (let i = 0; i < 20000; i++) runs[k]();   // warm up (JIT) every path
  const v8 = await import('node:v8'), vm = await import('node:vm');
  let gc = globalThis.gc;
  if (!gc) { try { v8.setFlagsFromString('--expose-gc'); gc = vm.runInNewContext('gc'); } catch { gc = null; } }
  if (gc) gc();
  const h0 = process.memoryUsage().heapUsed;
  for (const k of Object.keys(runs)) for (let i = 0; i < 20000; i++) runs[k]();
  if (gc) gc();
  const h1 = process.memoryUsage().heapUsed;
  check('alloc: retained heap growth over 80 000 steps (gc before/after)', (h1 - h0) / 1024, -Infinity, 64, 'KB');
  impulses = 0;
  const res = {};
  let sampled = !!inspector;
  for (const k of Object.keys(runs)) {
    if (!sampled) { for (let i = 0; i < N; i++) runs[k](); continue; }
    try {
      const s = new inspector.Session(); s.connect();
      await s.post('HeapProfiler.enable');
      await s.post('HeapProfiler.startSampling', { samplingInterval: 16, includeObjectsCollectedByMajorGC: true, includeObjectsCollectedByMinorGC: true });
      for (let i = 0; i < N; i++) runs[k]();
      const { profile } = await s.post('HeapProfiler.stopSampling');
      await s.post('HeapProfiler.disable'); s.disconnect();
      let coupling = 0, cc = 0, all = 0;
      const walk = (n) => {
        if (n.selfSize && !/^node:/.test(n.callFrame.url)) {
          all += n.selfSize;
          if (/CharacterController\.js$/.test(n.callFrame.url)) { cc += n.selfSize; if (COUPLING.has(n.callFrame.functionName)) coupling += n.selfSize; }
        }
        n.children.forEach(walk);
      };
      walk(profile.head);
      res[k] = { coupling: coupling / N, cc: cc / N, all: all / N };
    } catch (err) { sampled = false; note(`allocation sampling unavailable: ${err.message}`); }
  }
  yes('alloc runs: pushing every step, riding every step', impulses === N && own.cc.pushedBody === own.crate && real.cc.pushedBody === real.crate && ride.cc.rideBody === ride.plat);
  if (sampled) {
    check('alloc: the coupling\'s functions while pushing, 3000 steps (B / step)', res.pushing.coupling, 0, 0.5);
    check('alloc: the coupling\'s functions while riding, 3000 steps (B / step)', res.riding.coupling, 0, 0.5);
    check('alloc: the coupling\'s functions, real DynamicsWorld push (B / step)', res['pushing (DynamicsWorld)'].coupling, 0, 0.5);
    note(`alloc per step (sampled, garbage included; all of it number boxing): plain walking ${res.plain.all.toFixed(1)} B ` +
      `(CharacterController.js ${res.plain.cc.toFixed(1)}), pushing ${res.pushing.all.toFixed(1)} B (${res.pushing.cc.toFixed(1)}), ` +
      `pushing a DynamicsWorld body ${res['pushing (DynamicsWorld)'].all.toFixed(1)} B (DynamicsWorld.applyForce boxes internally), ` +
      `riding ${res.riding.all.toFixed(1)} B (${res.riding.cc.toFixed(1)})`);
  }
}

console.log(results.join('\n'));
console.log(failures ? `\n${failures} of ${count} check(s) FAILED` : `\nall ${count} checks passed`);
process.exit(failures ? 1 : 0);
