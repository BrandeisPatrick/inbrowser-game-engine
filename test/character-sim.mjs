// Headless checks for the kinematic CharacterController (no browser, no WebGL). Run: node test/character-sim.mjs
import { CharacterController } from '../src/physics/CharacterController.js';
import { PhysicsWorld } from '../src/physics/PhysicsWorld.js';

const DT = 1 / 120;
const DEG = Math.PI / 180;
let failures = 0;
const results = [];
function check(name, value, lo, hi, unit = '') {
  const ok = Number.isFinite(value) && value >= lo && value <= hi;
  if (!ok) failures++;
  results.push(`${ok ? 'PASS' : 'FAIL'}  ${name.padEnd(50)} ${fmt(value)} ${unit}   [${lo} .. ${hi}]`);
  return ok;
}
const fmt = (x) => (Number.isFinite(x) ? x.toFixed(4).padStart(10) : String(x).padStart(10));

// flat world with optional boxes (legacy plain-object world, like car-sim's)
function makeWorld(boxes = [], ground = () => 0) {
  return {
    getGroundHeight: (x, z) => ground(x, z),
    queryColliders(x, z, r) {
      return boxes.filter((b) => x + r > b.minX && x - r < b.maxX && z + r > b.minZ && z - r < b.maxZ);
    },
  };
}
const box = (minX, minY, maxX, maxY, minZ, maxZ, kind = 0) => ({ minX, minY, maxX, maxY, minZ, maxZ, kind });

// input: { mx, mz, jump, sprint } or fn(t, cc) -> that; dt defaults to DT
function run(cc, world, seconds, input, onStep, dt = DT) {
  const n = Math.round(seconds / dt);
  for (let i = 0; i < n; i++) {
    const c = typeof input === 'function' ? input(i * dt, cc) : input;
    cc.step(dt, c.mx || 0, c.mz || 0, !!c.jump, !!c.sprint, world);
    if (onStep && onStep(i * dt, cc) === false) return i * dt;
  }
  return seconds;
}
function assertSane(cc, label) {
  const p = cc.position, v = cc.velocity;
  if (![p.x, p.y, p.z, v.x, v.y, v.z].every(Number.isFinite)) { failures++; results.push(`FAIL  ${label}: NaN state`); }
}

// --- 1. flat ground: walk, sprint, stop ---------------------------------------------------------
{
  const world = makeWorld();
  const cc = new CharacterController();
  let tWalk = NaN, air = 0;
  run(cc, world, 1, { mx: 1 }, (t, c) => { if (Number.isNaN(tWalk) && c.speed >= 0.99 * c.walk) tWalk = t + DT; if (!c.grounded && t > 0) air++; });
  check('walk: reaches walk speed (99 %)', tWalk, 0, 0.5, 's');
  let lo = Infinity, hi = 0;
  run(cc, world, 2, { mx: 1 }, (t, c) => { lo = Math.min(lo, c.speed); hi = Math.max(hi, c.speed); if (!c.grounded) air++; });
  check('walk: speed stays within 1 % (min / walk)', lo / cc.walk, 0.99, 1.01);
  check('walk: speed stays within 1 % (max / walk)', hi / cc.walk, 0.99, 1.01);
  check('walk: grounded every step (airborne steps)', air, 0, 0);
  check('walk: feet stay on the ground (|y|)', Math.abs(cc.position.y), 0, 1e-9, 'm');
  check('walk: distance after 3 s', cc.position.x, 5.8, 6, 'm');
  let tRun = NaN;
  run(cc, world, 1, { mx: 1, sprint: true }, (t, c) => { if (Number.isNaN(tRun) && c.speed >= 0.99 * c.run) tRun = t + DT; });
  check('sprint: reaches run speed (99 %)', tRun, 0, 0.5, 's');
  check('sprint: run speed held', cc.speed / cc.run, 0.999, 1.001);
  let tStop = NaN;
  run(cc, world, 1, {}, (t, c) => { if (Number.isNaN(tStop) && c.speed === 0) tStop = t + DT; });
  check('stop: at rest after releasing from a sprint', tStop, 0, 0.3, 's');
  // diagonal input longer than 1 is normalised
  const c2 = new CharacterController();
  run(c2, world, 1, { mx: 3, mz: 3 });
  check('input: |move| > 1 normalised (speed / walk)', c2.speed / c2.walk, 0.999, 1.001);
}

// --- 2. walls: slide, blocked -----------------------------------------------------------------------
{
  const wall = box(-50, 0, 50, 3, 1, 2);
  const world = makeWorld([wall]);
  const cc = new CharacterController();
  const d = Math.SQRT1_2;
  let maxZ = -Infinity, blockedSteps = 0;
  run(cc, world, 3, { mx: d, mz: d }, (t, c) => { maxZ = Math.max(maxZ, c.position.z); if (c.blocked) blockedSteps++; });
  check('wall slide: tangential speed kept (/ walk cos 45)', cc.speed / (cc.walk * d), 0.99, 1.01);
  check('wall slide: blocked set while pushing in', cc.blocked ? 1 : 0, 1, 1);
  check('wall slide: wallNormal points out of the wall', -cc.wallNormal.z, 0.999, 1);
  check('wall slide: no penetration (max z - (wall - r))', maxZ - (wall.minZ - cc.radius), -1e-3, 1e-5, 'm');
  // walking parallel along the wall face does not report blocked
  const c2 = new CharacterController(); c2.teleport(0, 0, wall.minZ - c2.radius - 0.01);
  let b2 = 0;
  run(c2, world, 1, { mx: 1 }, (t, c) => { if (c.blocked) b2++; });
  check('wall: walking parallel is not blocked (steps)', b2, 0, 0);
  // the same run on a PhysicsWorld matches the legacy object exactly
  const pw = new PhysicsWorld();
  pw.addStaticBox(wall.minX, wall.minY, wall.maxX, wall.maxY, wall.minZ, wall.maxZ, 0);
  const c3 = new CharacterController();
  run(c3, pw, 3, { mx: d, mz: d });
  check('PhysicsWorld == legacy world (|dpos|)', c3.position.distanceTo(cc.position), 0, 1e-12, 'm');
}

// --- 3. steps: 0.3 m kerb climbed, 0.5 m not -------------------------------------------------------------
{
  const kerb = box(2, 0, 10, 0.3, -5, 5);
  const world = makeWorld([kerb]);
  const cc = new CharacterController();
  let ups = 0, blk = 0;
  run(cc, world, 3, { mx: 1 }, (t, c) => { if (c.steppedUp) ups++; if (c.blocked) blk++; });
  check('kerb 0.3 m: steppedUp (steps)', ups, 1, 1);
  check('kerb 0.3 m: final y on top', cc.position.y, 0.3 - 1e-9, 0.3 + 1e-9, 'm');
  check('kerb 0.3 m: kept walking (x)', cc.position.x, 5.5, 6, 'm');
  check('kerb 0.3 m: grounded on the box', cc.grounded ? 1 : 0, 1, 1);
  check('kerb 0.3 m: groundKind is the box kind (0)', cc.groundKind === 0 ? 1 : 0, 1, 1);
  check('kerb 0.3 m: no wall hit (blocked steps)', blk, 0, 0);
  const tall = box(2, 0, 10, 0.5, -5, 5);
  const w2 = makeWorld([tall]);
  const c2 = new CharacterController();
  let ups2 = 0;
  run(c2, w2, 3, { mx: 1 }, (t, c) => { if (c.steppedUp) ups2++; });
  check('kerb 0.5 m: not climbed (y)', c2.position.y, 0, 0, 'm');
  check('kerb 0.5 m: held at the face (x - (2 - r))', c2.position.x - (2 - c2.radius), -1e-3, 1e-5, 'm');
  check('kerb 0.5 m: blocked', c2.blocked ? 1 : 0, 1, 1);
  check('kerb 0.5 m: no step-up (steps)', ups2, 0, 0);
  // stepping down off a 0.25 m box: snapping keeps contact
  const low = box(-5, 0, 2, 0.25, -5, 5);
  const w3 = makeWorld([low]);
  const c3 = new CharacterController(); c3.teleport(0, 0.25, 0);
  let air3 = 0;
  run(c3, w3, 2, { mx: 1 }, (t, c) => { if (!c.grounded && t > 0) air3++; });
  check('kerb drop 0.25 m: snapped down (airborne steps)', air3, 0, 0);
  check('kerb drop 0.25 m: on the ground after', c3.position.y, 0, 0, 'm');
}

// --- 4. slopes: 60 deg not walkable, 30 deg walkable, snapping down 10 deg ---------------------------------
{
  const t60 = Math.tan(60 * DEG), t30 = Math.tan(30 * DEG), t10 = Math.tan(10 * DEG);
  const w60 = makeWorld([], (x) => t60 * x);
  const cc = new CharacterController();
  let slopeSteps = 0, jumps = 0, groundedSteps = 0;
  run(cc, w60, 2, (t) => ({ mx: 1, jump: t > 0.5 && t < 0.52 }), (t, c) => {
    if (c.onSlope) slopeSteps++; if (c.jumpedThisStep) jumps++; if (c.grounded) groundedSteps++;
  });
  check('60 deg: slides down while pushing uphill (x)', cc.position.x, -20, -1, 'm');
  check('60 deg: onSlope every step', slopeSteps, 240, 240);
  check('60 deg: never grounded (not walkable)', groundedSteps, 0, 0);
  check('60 deg: cannot jump from it', jumps, 0, 0);
  check('60 deg: stays on the surface (y - h)', Math.abs(cc.position.y - t60 * cc.position.x), 0, 1e-6, 'm');
  const w30 = makeWorld([], (x) => t30 * x);
  const c2 = new CharacterController();
  let g30 = 0;
  run(c2, w30, 2, { mx: 1 }, (t, c) => { if (c.grounded) g30++; });
  check('30 deg: walkable uphill (x after 2 s)', c2.position.x, 3.8, 4, 'm');
  check('30 deg: grounded every step', g30, 240, 240);
  check('30 deg: groundNormal (n.y = cos 30)', c2.groundNormal.y, Math.cos(30 * DEG) - 1e-6, Math.cos(30 * DEG) + 1e-6);
  // flat, then a 10 deg downslope at a kink: snapping keeps contact at run speed; with snap = 0 it launches
  const down = (x) => (x < 0 ? 0 : -t10 * x);
  const w10 = makeWorld([], down);
  const c3 = new CharacterController(); c3.teleport(-1, 0, 0);
  let air3 = 0;
  run(c3, w10, 2, { mx: 1, sprint: true }, (t, c) => { if (!c.grounded && t > 0) air3++; });
  check('snap: 10 deg downslope at run speed (airborne steps)', air3, 0, 0);
  check('snap: still on the slope surface', Math.abs(c3.position.y - down(c3.position.x)), 0, 1e-6, 'm');
  const c4 = new CharacterController({ snap: 0 }); c4.teleport(-1, 0, 0);
  let air4 = 0;
  run(c4, w10, 2, { mx: 1, sprint: true }, (t, c) => { if (!c.grounded && t > 0) air4++; });
  check('snap = 0: the kink launches (airborne steps)', air4, 1, 1000);
}

// --- 4b. heightfield steps and cliffs, steep-slope base, stairs ---------------------------------------------------
{
  // a 1 m terrain cliff is a wall (blocked, never climbed, no slope jitter); a 0.3 m terrain kerb is walked onto
  const cliff = makeWorld([], (x) => (x > 2 ? 1 : 0));
  const cc = new CharacterController();
  let maxY = 0, slope = 0, blk = 0;
  run(cc, cliff, 3, { mx: 1 }, (t, c) => { maxY = Math.max(maxY, c.position.y); if (c.onSlope) slope++; if (c.blocked) blk++; });
  check('terrain cliff 1 m: not climbed (max y)', maxY, 0, 0, 'm');
  check('terrain cliff 1 m: blocked at the face (x)', cc.position.x, 2 - cc.radius - 0.01, 2 - cc.radius + 0.01, 'm');
  check('terrain cliff 1 m: blocked', cc.blocked ? 1 : 0, 1, 1);
  check('terrain cliff 1 m: never onSlope (steps)', slope, 0, 0);
  const kerb = makeWorld([], (x) => (x > 2 ? 0.3 : 0));
  const c2 = new CharacterController();
  let g2 = 0;
  run(c2, kerb, 3, { mx: 1 }, (t, c) => { if (c.grounded) g2++; });
  check('terrain kerb 0.3 m: walked onto (y)', c2.position.y, 0.3, 0.3, 'm');
  check('terrain kerb 0.3 m: grounded every step', g2, 360, 360);
  // flat ground into a 60 deg slope: a wall at its foot (no climbing, no slide oscillation)
  const t60 = Math.tan(60 * DEG);
  const foot = makeWorld([], (x) => (x > 2 ? t60 * (x - 2) : 0));
  const c3 = new CharacterController();
  let maxY3 = 0, slope3 = 0;
  run(c3, foot, 3, { mx: 1, sprint: true }, (t, c) => { maxY3 = Math.max(maxY3, c.position.y); if (c.onSlope) slope3++; });
  check('60 deg slope foot: not climbed (max y)', maxY3, 0, 0, 'm');
  check('60 deg slope foot: blocked', c3.blocked ? 1 : 0, 1, 1);
  check('60 deg slope foot: grounded', c3.grounded ? 1 : 0, 1, 1);
  check('60 deg slope foot: never onSlope (steps)', slope3, 0, 0);
  // stairs: 10 x 0.25 m rise, 0.3 m tread, up then down without leaving the ground
  const steps = [];
  for (let i = 0; i < 10; i++) steps.push(box(2 + i * 0.3, 0, 20, 0.25 * (i + 1), -2, 2));
  const ws = makeWorld(steps);
  const c4 = new CharacterController();
  let ups = 0, air = 0;
  run(c4, ws, 4, { mx: 1 }, (t, c) => { if (c.steppedUp) ups++; if (!c.grounded) air++; });
  check('stairs up: one steppedUp per stair', ups, 10, 10);
  check('stairs up: top reached (y)', c4.position.y, 2.5, 2.5, 'm');
  run(c4, ws, 6, { mx: -1 }, (t, c) => { if (!c.grounded) air++; });
  check('stairs up and down: never airborne (steps)', air, 0, 0);
  check('stairs down: back on the ground (y)', c4.position.y, 0, 0, 'm');
  // running jump onto a 1 m crate lands on its top
  const crate = box(1.5, 0, 4, 1, -2, 2);
  const wc = makeWorld([crate]);
  const c5 = new CharacterController(); c5.teleport(-3, 0, 0);
  let jumped = false;
  run(c5, wc, 2, (t, c) => { const j = !jumped && c.position.x > 0.2; if (j) jumped = true; return { mx: c.position.x < 3 ? 1 : 0, sprint: true, jump: j }; });
  check('running jump onto a 1 m crate: on top (y)', c5.position.y, 1, 1, 'm');
  check('running jump onto a 1 m crate: grounded on it', c5.grounded && c5.groundKind === 0 ? 1 : 0, 1, 1);
}

// --- 5. jump: apex, landing, coyote time, jump buffer, ceiling --------------------------------------------------
{
  const world = makeWorld();
  const cc = new CharacterController();
  run(cc, world, 0.2, {});
  let maxY = 0, lands = 0, landV = NaN, jumps = 0;
  run(cc, world, 2, (t) => ({ jump: t < DT / 2 }), (t, c) => {
    maxY = Math.max(maxY, c.position.y);
    if (c.landedThisStep) { lands++; landV = c.landingSpeed; }
    if (c.jumpedThisStep) jumps++;
  });
  const apex = cc.jumpSpeed * cc.jumpSpeed / (2 * cc.gravity);
  check('jump: apex / (v^2 / 2g)', maxY / apex, 0.999, 1.0000001);
  check('jump: one take-off', jumps, 1, 1);
  check('jump: landedThisStep once', lands, 1, 1);
  check('jump: landingSpeed / jumpSpeed', landV / cc.jumpSpeed, 0.9999, 1.0001);
  check('jump: back on the ground (y)', cc.position.y, 0, 0, 'm');
  check('jump: grounded, vertical speed 0', cc.grounded && cc.velocity.y === 0 ? 1 : 0, 1, 1);

  // coyote time: walking off a 3 m ledge, jump pressed 0.05 s later still jumps; 0.3 s later it does not
  const ledge = box(-10, 0, 0, 3, -10, 10);
  const wl = makeWorld([ledge]);
  for (const [delay, want, label] of [[0.05, 1, 'coyote: jump 0.05 s after leaving a ledge'], [0.3, 0, 'coyote: jump 0.3 s after leaving: none']]) {
    const c = new CharacterController(); c.teleport(-1, 3, 0);
    let offT = NaN, pressed = false, jumped = 0;
    run(c, wl, 2.5, (t, k) => {
      const press = !pressed && !Number.isNaN(offT) && t >= offT + delay;
      if (press) pressed = true;
      return { mx: 1, jump: press };
    }, (t, k) => {
      if (Number.isNaN(offT) && t > 0.05 && !k.grounded) offT = t;
      if (k.jumpedThisStep) jumped++;
    });
    check(label, jumped, want, want);
  }

  // jump buffer: pressed 0.05 s before touchdown jumps on landing; 0.2 s before does not
  for (const [lead, want, label] of [[0.05, 1, 'buffer: press 0.05 s before landing jumps'], [0.2, 0, 'buffer: press 0.2 s before landing: none']]) {
    const c = new CharacterController(); c.teleport(0, 2, 0);
    const tLand = Math.sqrt(2 * 2 / c.gravity);
    let jumped = 0, landT = NaN, jumpT = NaN;
    run(c, world, 1.5, (t) => ({ jump: Math.abs(t - (tLand - lead)) < DT / 2 }), (t, k) => {
      if (k.landedThisStep && Number.isNaN(landT)) landT = t;
      if (k.jumpedThisStep) { jumped++; if (Number.isNaN(jumpT)) jumpT = t; }
    });
    check(label, jumped, want, want);
    if (want) check('buffer: jumps on the step after touchdown (steps)', Math.round((jumpT - landT) / DT), 1, 1);
  }

  // ceiling: a box bottom at 2.5 m stops the jump (head 1.8 m -> feet at most 0.7 m)
  const roof = box(-5, 2.5, 5, 3, -5, 5);
  const wc = makeWorld([roof]);
  const c5 = new CharacterController();
  run(c5, wc, 0.1, {});
  let maxY5 = 0, ceil = 0, land5 = 0;
  run(c5, wc, 1.5, (t) => ({ jump: t < DT / 2 }), (t, c) => { maxY5 = Math.max(maxY5, c.position.y); if (c.hitCeiling) ceil++; if (c.landedThisStep) land5++; });
  check('ceiling: stops the jump (max feet y)', maxY5, 0.7 - 1e-6, 0.7 + 1e-6, 'm');
  check('ceiling: hitCeiling reported once', ceil, 1, 1);
  check('ceiling: falls back and lands', land5, 1, 1);
}

// --- 6. vehicles, falling, hold, null world ------------------------------------------------------------------
{
  // a moving traffic box shoves a standing character with its own velocity
  const car = box(-3, 0, -1, 1.5, -1, 1, 'vehicle'); car.vx = 4; car.vz = 0;
  const wv = makeWorld([car]);
  const cc = new CharacterController();
  let minGap = Infinity, shoved = false;
  run(cc, wv, 1.5, {}, (t, c) => {
    minGap = Math.min(minGap, c.position.x - c.radius - car.maxX);
    if (c.velocity.x > 3.9) shoved = true;
    car.minX += car.vx * DT; car.maxX += car.vx * DT;
  });
  check('vehicle: shoves the character (velocity.x = vx)', cc.velocity.x, 3.99, 4.01, 'm/s');
  check('vehicle: pushed along with the box (x - (maxX + r))', cc.position.x - (car.maxX + cc.radius), -0.05, 0.001, 'm');
  check('vehicle: no deep overlap (min gap)', minGap, -0.05, 1, 'm');
  check('vehicle: shove registered', shoved ? 1 : 0, 1, 1);
  assertSane(cc, 'vehicle shove');

  // fall speed caps at maxFall
  const c2 = new CharacterController(); c2.teleport(0, 1000, 0);
  run(c2, null, 8, {});
  check('fall: speed capped at maxFall', c2.fallSpeed, c2.maxFall - 1e-9, c2.maxFall + 1e-9, 'm/s');
  check('fall: airTime counts', c2.airTime, 7.99, 8.01, 's');
  check('fall: not grounded', c2.grounded || c2.onSlope ? 1 : 0, 0, 0);

  // hold freezes the character on the ground; teleport zeroes the velocity
  const c3 = new CharacterController();
  run(c3, null, 1, { mx: 1, sprint: true });
  c3.hold = true;
  const x3 = c3.position.x;
  run(c3, null, 1, { mx: 1, sprint: true, jump: true });
  check('hold: frozen (dx)', Math.abs(c3.position.x - x3), 0, 0, 'm');
  check('hold: frozen (speed)', c3.speed, 0, 0, 'm/s');
  check('hold: on the ground (y)', c3.position.y, 0, 0, 'm');
  check('hold: grounded', c3.grounded ? 1 : 0, 1, 1);
  check('hold: no jump while held', c3.jumpedThisStep ? 1 : 0, 0, 0);
  c3.hold = false;
  c3.velocity.set(3, 0, 3); c3.teleport(5, 0, 5);
  let land3 = 0;
  run(c3, null, 0.1, {}, (t, c) => { if (c.landedThisStep) land3++; });
  check('teleport: zero velocity', c3.speed, 0, 0, 'm/s');
  check('teleport: no landing event at ground height', land3, 0, 0);
  // null world: flat ground at y = 0, walking in circles stays on it
  const c4 = new CharacterController(); c4.teleport(0, 0.5, 0);
  let minY = Infinity;
  run(c4, null, 4, (t) => ({ mx: Math.cos(t), mz: Math.sin(t) }), (t, c) => { if (t > 0.5) minY = Math.min(minY, c.position.y); });
  check('null world: on the flat ground (final y)', c4.position.y, 0, 0, 'm');
  check('null world: never below it (min y)', minY, 0, 0, 'm');
}

// --- 6b. adapters, crouch / ceilings, push cap, hold in the air, vehicles, jump edge, dt = 1/30, edge cases ---------
{
  // a world that fills `out` and returns nothing (PhysicsWorld provider convention) works like one returning an array
  const wall = box(-50, 0, 50, 3, 1, 2);
  const fills = {
    getGroundHeight: () => 0,
    queryColliders(x, z, r, out) { if (x + r > wall.minX && x - r < wall.maxX && z + r > wall.minZ && z - r < wall.maxZ) out.push(wall); },
  };
  const a = new CharacterController(), b = new CharacterController();
  run(a, fills, 2, { mz: 1 }); run(b, makeWorld([wall]), 2, { mz: 1 });
  check('adapter: fill-out world blocks at the wall (z)', a.position.z, wall.minZ - a.radius - 1e-3, wall.minZ - a.radius, 'm');
  check('adapter: fill-out world == returning world (|dpos|)', a.position.distanceTo(b.position), 0, 0, 'm');

  // crouch: standing up under a 1.5 m ceiling is refused, allowed once clear
  const roof = box(-3, 1.5, 3, 2, -3, 3);
  const wr = makeWorld([roof]);
  const c = new CharacterController(); c.setHeight(1.0);
  run(c, wr, 0.2, {});
  check('crouch: canStand(1.8) under a 1.5 m ceiling', c.canStand(1.8, wr) ? 1 : 0, 0, 0);
  check('crouch: canStand(1.4) under it', c.canStand(1.4, wr) ? 1 : 0, 1, 1);
  check('crouch: setHeight(1.8) refused', c.setHeight(1.8, wr) ? 1 : 0, 0, 0);
  check('crouch: height unchanged after the refusal', c.height, 1.0, 1.0, 'm');
  run(c, wr, 3, { mx: 1 });
  check('crouch: setHeight(1.8) once clear', c.setHeight(1.8, wr) ? 1 : 0, 1, 1);
  check('crouch: stood up (height)', c.height, 1.8, 1.8, 'm');
  // a box arriving over the head (or height assigned directly) clamps the height: no sideways eject
  const live = [];
  const wl = makeWorld(live);
  const c2 = new CharacterController(); c2.teleport(0.3, 0, -0.2);
  run(c2, wl, 0.2, {});
  live.push(box(-3, 1.5, 3, 2, -3, 3));
  let ceil2 = 0;
  run(c2, wl, 0.5, { jump: true }, (t, k) => { if (k.hitCeiling) ceil2++; });
  check('box over the head: no sideways eject (|dxz|)', Math.hypot(c2.position.x - 0.3, c2.position.z + 0.2), 0, 0, 'm');
  check('box over the head: height clamped to fit', c2.height, 1.5, 1.5, 'm');
  check('box over the head: hitCeiling reported', ceil2, 1, 1000);
  check('box over the head: stays on the ground (y)', c2.position.y, 0, 0, 'm');
  const c3 = new CharacterController(); c3.setHeight(1.0); run(c3, wr, 0.1, {});
  c3.height = 1.8; run(c3, wr, DT, {});
  check('height assigned under a ceiling: no eject (|dxz|)', Math.hypot(c3.position.x, c3.position.z), 0, 0, 'm');
  check('height assigned under a ceiling: clamped', c3.height, 1.5, 1.5, 'm');
  // inside a building (teleported): pushed out at most r per step, then clear of it
  const bld = box(-5, 0, 5, 5, -5, 5);
  const wb = makeWorld([bld]);
  const c4 = new CharacterController(); c4.teleport(0.5, 0, 0.2);
  run(c4, wb, DT, {});
  check('inside a box: push capped at r per step (dx)', c4.position.x - 0.5, 0, c4.radius + 1e-5, 'm');
  run(c4, wb, 1, {});
  check('inside a box: out after 1 s (x - (maxX + r))', c4.position.x - (bld.maxX + c4.radius), -1e-3, 1e-5, 'm');

  // hold: in the air it stays put; within snap it rests on the ground
  const c5 = new CharacterController(); c5.teleport(0, 5, 0);
  run(c5, null, 0.2, {});
  const y5 = c5.position.y;
  c5.hold = true; run(c5, null, 1, {});
  check('hold in the air: stays put (dy)', c5.position.y - y5, 0, 0, 'm');
  check('hold in the air: not grounded', c5.grounded ? 1 : 0, 0, 0);
  c5.hold = false;
  let land5 = 0; run(c5, null, 2, {}, (t, k) => { if (k.landedThisStep) land5++; });
  check('hold released in the air: falls and lands once', land5, 1, 1);
  const c6 = new CharacterController(); c6.teleport(0, 0.2, 0); c6.hold = true; run(c6, null, DT, {});
  check('hold within snap: rests on the ground (y)', c6.position.y, 0, 0, 'm');

  // vehicles are never support: landing on a parked one slides off sideways onto the ground
  const parked = box(-1, 0, 1, 1.5, -2, 2, 'vehicle');
  const wv = makeWorld([parked]);
  const c7 = new CharacterController(); c7.teleport(0.2, 3, 0);
  let onVeh = 0;
  run(c7, wv, 2, {}, (t, k) => { if (k.groundKind === 'vehicle') onVeh++; });
  check('vehicle: never support (steps on it)', onVeh, 0, 0);
  check('vehicle: slid off to its side (x - (maxX + r))', c7.position.x - (parked.maxX + c7.radius), -1e-3, 1e-5, 'm');
  check('vehicle: ends on the ground (y)', c7.position.y, 0, 0, 'm');

  // jump is an edge: holding the button jumps once (no bunny hop); release + press jumps again; a jump on the very
  // first step after a teleport works
  const c8 = new CharacterController();
  let j8 = 0;
  run(c8, null, 3, { jump: true }, (t, k) => { if (k.jumpedThisStep) j8++; });
  check('jump held 3 s: one jump', j8, 1, 1);
  run(c8, null, 0.1, {}); run(c8, null, 0.1, { jump: true }, (t, k) => { if (k.jumpedThisStep) j8++; });
  check('jump released and pressed again: second jump', j8, 2, 2);
  const c9 = new CharacterController(); run(c9, null, 0.5, {}); c9.teleport(3, 0, 3);
  run(c9, null, DT, { jump: true });
  check('teleport: jump on the first step after it', c9.jumpedThisStep ? 1 : 0, 1, 1);

  // slopeLimit accessor updates the cached cos / tan: at 65 deg a 60 deg slope is walkable
  const t60 = Math.tan(60 * DEG);
  const c10 = new CharacterController(); c10.slopeLimit = 65;
  let g10 = 0;
  run(c10, makeWorld([], (x) => t60 * x), 1, { mx: 1 }, (t, k) => { if (k.grounded) g10++; });
  check('slopeLimit = 65: reads back', c10.slopeLimit, 65, 65, 'deg');
  check('slopeLimit = 65: 60 deg slope walkable (grounded steps)', g10, 120, 120);
  check('slopeLimit = 65: walked uphill (x)', c10.position.x, 1.8, 2, 'm');

  // steps: a box top exactly at feet + stepHeight is climbed; a 2 cm lip is climbed without a steppedUp event
  const c11 = new CharacterController();
  let ups11 = 0;
  run(c11, makeWorld([box(2, 0, 10, c11.stepHeight, -5, 5)]), 2, { mx: 1 }, (t, k) => { if (k.steppedUp) ups11++; });
  check('step exactly stepHeight: climbed (y)', c11.position.y, c11.stepHeight, c11.stepHeight, 'm');
  check('step exactly stepHeight: steppedUp once', ups11, 1, 1);
  const c12 = new CharacterController();
  let ups12 = 0;
  run(c12, makeWorld([box(2, 0, 10, 0.02, -5, 5)]), 2, { mx: 1 }, (t, k) => { if (k.steppedUp) ups12++; });
  check('2 cm lip: climbed (y)', c12.position.y, 0.02, 0.02, 'm');
  check('2 cm lip: no steppedUp event (< 5 cm)', ups12, 0, 0);

  // two-box wedge: a 0.6 m gap (narrower than the capsule) stops it where it touches both corners, centred
  const gap = 0.3;
  const wedge = makeWorld([box(2, 0, 4, 2, gap, 3), box(2, 0, 4, 2, -3, -gap)]);
  const c13 = new CharacterController(); c13.teleport(0, 0, 0.05);
  let maxX13 = -Infinity;
  run(c13, wedge, 2, { mx: 1, sprint: true }, (t, k) => { maxX13 = Math.max(maxX13, k.position.x); });
  const xw = 2 - Math.sqrt(c13.radius * c13.radius - gap * gap);
  check('wedge: stops touching both corners (x)', c13.position.x, xw - 1e-3, xw + 1e-5, 'm');
  check('wedge: never more than 0.1 mm past that (max x - x)', maxX13 - xw, -1, 1e-4, 'm');
  check('wedge: centred in the gap (|z|)', Math.abs(c13.position.z), 0, 1e-3, 'm');
  assertSane(c13, 'wedge');

  // NaN / Infinity robustness: non-finite heights read as 0; non-finite inputs are ignored
  const c14 = new CharacterController();
  run(c14, makeWorld([], (x) => (x > 1 ? NaN : Infinity)), 1, { mx: 1 });
  check('NaN / Infinity heights read as 0 (y)', c14.position.y, 0, 0, 'm');
  check('NaN / Infinity heights: walks normally (x)', c14.position.x, 1.9, 2, 'm');
  const c15 = new CharacterController();
  c15.step(NaN, NaN, Infinity, true, false, null); c15.teleport(NaN, Infinity, -Infinity); c15.step(Infinity, 1, NaN, false, true, null);
  assertSane(c15, 'non-finite inputs');
  check('non-finite inputs: sane position (|p|)', c15.position.length(), 0, 0.2, 'm');

  // dt = 1/30 (the clamp limit): speed, jump apex / landing, kerb, wall
  const D30 = 1 / 30;
  const c16 = new CharacterController();
  run(c16, makeWorld(), 1, { mx: 1 }, null, D30);
  check('dt 1/30: walk speed', c16.speed / c16.walk, 0.999, 1.001);
  const c17 = new CharacterController();
  let maxY17 = 0, land17 = NaN;
  run(c17, makeWorld(), 1.5, (t) => ({ jump: t < D30 / 2 }), (t, k) => { maxY17 = Math.max(maxY17, k.position.y); if (k.landedThisStep) land17 = k.landingSpeed; }, D30);
  const apex = c17.jumpSpeed * c17.jumpSpeed / (2 * c17.gravity);
  check('dt 1/30: jump apex / (v^2 / 2g)', maxY17 / apex, 0.999, 1.0000001);
  check('dt 1/30: landingSpeed / jumpSpeed', land17 / c17.jumpSpeed, 0.9999, 1.0001);
  const c18 = new CharacterController();
  let ups18 = 0;
  run(c18, makeWorld([box(2, 0, 10, 0.3, -5, 5)]), 2, { mx: 1, sprint: true }, (t, k) => { if (k.steppedUp) ups18++; }, D30);
  check('dt 1/30: 0.3 m kerb climbed at run speed (y)', c18.position.y, 0.3, 0.3, 'm');
  check('dt 1/30: kerb steppedUp once', ups18, 1, 1);
  const c19 = new CharacterController();
  let maxZ19 = -Infinity;
  run(c19, makeWorld([wall]), 2, { mx: 0.6, mz: 0.8, sprint: true }, (t, k) => { maxZ19 = Math.max(maxZ19, k.position.z); }, D30);
  check('dt 1/30: sprint into a wall, no penetration', maxZ19 - (wall.minZ - c19.radius), -1e-3, 1e-5, 'm');
}

// --- 7. determinism: two controllers, identical scripted inputs, 10 s ------------------------------------------
{
  const boxes = [box(-2, 2.6, 2, 3, -2, 2)]; // a ceiling over the origin + a 5 x 5 field of kerbs, steps and walls
  const tops = [0.2, 0.3, 1.0, 2.0, 0.15];
  for (let i = 0; i < 5; i++) for (let k = 0; k < 5; k++) if (i !== 2 || k !== 2) boxes.push(box(-9 + i * 4, 0, -7.5 + i * 4, tops[(i + 2 * k) % 5], -9 + k * 4, -7.5 + k * 4));
  const hill = (x, z) => 0.6 * Math.sin(x * 0.3) * Math.cos(z * 0.25);
  const world = makeWorld(boxes, hill);
  const a = new CharacterController(), b = new CharacterController();
  a.teleport(0, 1, -4); b.teleport(0, 1, -4);
  let seed = 12345;
  const rnd = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; };
  const f64 = new Float64Array(3), u32 = new Uint32Array(f64.buffer);
  const fnv = (hsh, c) => { f64[0] = c.position.x; f64[1] = c.position.y; f64[2] = c.position.z; for (let i = 0; i < 6; i++) hsh = Math.imul(hsh ^ u32[i], 16777619) >>> 0; return hsh; };
  let ha = 2166136261, hb = 2166136261, mx = 0, mz = 0, sprint = false, path = 0, onGround = 0, onBox = 0, events = 0;
  const px = a.position.clone();
  for (let i = 0; i < 1200; i++) {
    if (i % 60 === 0) { const ang = rnd() * 2 * Math.PI; mx = Math.cos(ang); mz = Math.sin(ang) * rnd(); sprint = rnd() < 0.4; }
    if (a.position.x * a.position.x + a.position.z * a.position.z > 64) { mx = -a.position.x * 0.2; mz = -a.position.z * 0.2; } // come back
    const jump = rnd() < 0.005;
    a.step(DT, mx, mz, jump, sprint, world);
    b.step(DT, mx, mz, jump, sprint, world);
    ha = fnv(ha, a); hb = fnv(hb, b);
    path += a.position.distanceTo(px); px.copy(a.position);
    if (a.groundKind === 'ground') onGround++; else if (a.groundKind !== null) onBox++;
    if (a.blocked || a.steppedUp || a.landedThisStep) events++;
  }
  check('determinism: position hashes equal (10 s)', ha === hb ? 1 : 0, 1, 1);
  // golden value: an intended behaviour change updates it (the scene's hill uses Math.sin: V8's is the same everywhere)
  const GOLDEN = 0xb611647c;
  check('determinism: golden hash (b611647c)', ha === GOLDEN ? 1 : 0, 1, 1);
  check('determinism: the run moved (path length)', path, 5, 100, 'm');
  console.log(`determinism run: ${onGround} steps on the heightfield, ${onBox} on boxes, ${events} wall / step / landing events`);
  assertSane(a, 'determinism');
  console.log(`determinism hash ${ha.toString(16)}`);
}

// --- perf ---------------------------------------------------------------------------------------------------------
{
  const boxes = [];
  for (let i = 0; i < 40; i++) boxes.push(box(i * 7, 0, i * 7 + 5, i % 3 === 0 ? 0.3 : 30, -8, -4));
  const world = makeWorld(boxes, (x, z) => 0.2 * Math.sin(x * 0.1));
  const cc = new CharacterController(); cc.teleport(0, 0, -6);
  const t0 = performance.now();
  run(cc, world, 100, (t) => ({ mx: 1, mz: Math.sin(t) * 0.5, sprint: true, jump: (t * 120 | 0) % 240 === 0 }));
  const ms = performance.now() - t0;
  check('perf: 12000 steps (ms)', ms, 0, 300, 'ms');
  assertSane(cc, 'perf');
}

console.log(results.join('\n'));
console.log(failures ? `\n${failures} check(s) FAILED` : '\nAll character controller checks passed.');
process.exit(failures ? 1 : 0);
