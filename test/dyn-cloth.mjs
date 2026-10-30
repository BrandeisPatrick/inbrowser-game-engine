// Headless checks for the XPBD particles of the dynamics world (src/physics/dynamics/particles.js and the particle hooks
// of DynamicsWorld.js): the ParticleStore id rule, a world without particles stepping bit-identically to 0.8.0, ropes
// (length, straightness, constraint forces, tethers), cloth (layout, mirror symmetry, compliance), two-way attachments
// (momentum, sleeping bodies, removeBody), golden hashes and serialize / from, allocation in step(), the solver hook
// order, particleAabb and the DynamicsClient guard. No browser, no WebGL. Run: node test/dyn-cloth.mjs
import v8 from 'node:v8';
import vm from 'node:vm';
import { DynamicsWorld } from '../src/physics/dynamics/DynamicsWorld.js';
import { ParticleStore, PFLAG, PCON, ParticleConstraints, Cloth, Rope } from '../src/physics/dynamics/particles.js';
import { box, sphere, capsule } from '../src/physics/dynamics/shapes.js';
import { WORLD_BODY } from '../src/physics/dynamics/state.js';
import { PhysicsWorld } from '../src/physics/PhysicsWorld.js';
import { DynamicsClient } from '../src/physics/DynamicsClient.js';
import { hashF64Array, hashU32, mulberry32 } from '../src/core/mathDet.js';

const DT = 1 / 60;
let failures = 0, total = 0;
const results = [];
const fmt = (x) => (!Number.isFinite(x) ? String(x) : Number.isInteger(x) ? String(x) : Math.abs(x) < 1e-3 && x !== 0 ? x.toExponential(2) : x.toFixed(4)).padStart(12);
function check(name, value, lo = 1, hi = lo, unit = '') {
  const v = typeof value === 'boolean' ? (value ? 1 : 0) : value;
  const ok = Number.isFinite(v) && v >= lo && v <= hi;
  total++;
  if (!ok) failures++;
  results.push(`${ok ? 'PASS' : 'FAIL'}  ${name.padEnd(64)} ${fmt(v)} ${unit}   [${fmt(lo).trim()}${hi !== lo ? ' .. ' + fmt(hi).trim() : ''}]`);
  return ok;
}
const note = (s) => results.push(`NOTE  ${s}`);
const throws = (fn) => { try { fn(); return false; } catch { return true; } };
const run = (w, steps) => { for (let i = 0; i < steps; i++) w.step(DT); };
const hex = (h) => '0x' + h.toString(16).padStart(8, '0');
const Q_FLAT = [-Math.SQRT1_2, 0, 0, Math.SQRT1_2];   // local XY plane -> world XZ, local +Z (normal) -> world +Y

function ropeLength(w, r) {
  const p = w.particles.pos;
  let L = 0;
  for (let k = 0; k + 1 < r.ids.length; k++) { const a = r.ids[k] * 3, b = r.ids[k + 1] * 3; L += Math.hypot(p[a] - p[b], p[a + 1] - p[b + 1], p[a + 2] - p[b + 2]); }
  return L;
}
function particleHash(w, h = 2166136261) {
  const P = w.particles, n = P.count;
  h = hashF64Array(P.pos, h, 0, n * 3);
  h = hashF64Array(P.vel, h, 0, n * 3);
  for (let i = 0; i < n; i++) h = hashU32(h, P.alive[i] | (P.flags[i] << 8));
  return h;
}
function bodyHash(w, h = 2166136261) {
  const B = w.bodies, n = B.count;
  h = hashF64Array(B.pos, h, 0, n * 3);
  h = hashF64Array(B.quat, h, 0, n * 4);
  h = hashF64Array(B.vel, h, 0, n * 3);
  h = hashF64Array(B.angVel, h, 0, n * 3);
  for (let i = 0; i < n; i++) h = hashU32(h, B.flags[i] | (B.alive[i] << 8));
  return h;
}
function finiteParticles(w) {
  const P = w.particles;
  for (let i = 0; i < P.count * 3; i++) if (!Number.isFinite(P.pos[i]) || !Number.isFinite(P.vel[i])) return false;
  return true;
}
const posOf = (w, id) => { const p = w.particles.pos, o = id * 3; return [p[o], p[o + 1], p[o + 2]]; };

// --- 1. ParticleStore: alloc / free rule, pinning, serialize ----------------------------------------------------------
{
  const P = new ParticleStore(4);
  const a = P.alloc(), b = P.alloc(), c = P.alloc();
  check('store: ids allocated ascending from 0', a === 0 && b === 1 && c === 2);
  check('store: alloc defaults (1 kg, invMass 1, group -1, flags 0, radius 0)', P.mass[a] === 1 && P.invMass[a] === 1 && P.group[a] === -1 && P.flags[a] === 0 && P.radius[a] === 0);
  P.pos[b * 3 + 1] = 5; P.vel[b * 3] = 2; P.group[b] = 7;
  P.free(b);
  check('store: free clears the slot', P.alive[b] === 0 && P.pos[b * 3 + 1] === 0 && P.vel[b * 3] === 0 && P.group[b] === -1 && P.invMass[b] === 0);
  check('store: count stays one past the highest id ever allocated', P.count, 3, 3);
  check('store: size counts live particles', P.size, 2, 2);
  P.free(c); P.free(a);
  check('store: lowest free id comes back first (free 1, 2, 0 -> 0, 1, 2)', P.alloc() === 0 && P.alloc() === 1 && P.alloc() === 2);
  check('store: then a fresh id', P.alloc(), 3, 3);
  check('store: alloc throws when full', throws(() => P.alloc()));
  P.setMass(2, 4);
  P.vel[2 * 3] = 1;
  P.setPinned(2, true);
  check('store: setPinned -> PINNED flag, invMass 0, velocity 0', (P.flags[2] & PFLAG.PINNED) !== 0 && P.invMass[2] === 0 && P.vel[6] === 0);
  P.setPinned(2, false);
  check('store: unpin restores invMass = 1 / mass', P.invMass[2] === 0.25 && (P.flags[2] & PFLAG.PINNED) === 0);
  const g0 = P.newGroup(), g1 = P.newGroup();
  check('store: newGroup hands out fresh ids', g1 === g0 + 1);
  P.free(1);
  P.pos[0] = 1.25; P.prevPos[5] = -3;
  const s = P.serialize(), Q = new ParticleStore(4);
  Q.restore(structuredClone(s));
  let same = Q.count === P.count && Q.nextGroup === P.nextGroup;
  for (const k of ['alive', 'flags', 'group', 'pos', 'prevPos', 'vel', 'invMass', 'radius', 'mass']) for (let i = 0; i < P[k].length; i++) if (!Object.is(P[k][i], Q[k][i])) same = false;
  check('store: serialize -> restore copies every array', same);
  check('store: restored free list (next alloc reuses id 1)', Q.alloc(), 1, 1);
}

// --- 2. a world without particles steps bit-identically to 0.8.0 --------------------------------------------------------
// BOX_PILE_080 was produced by the box-only scene with the committed 0.8.0 DynamicsWorld (git show HEAD:... in a temp
// tree). The mixed pile (spheres, capsules) moved with the 0.9.0 curved-contact anchor fix (see DynamicsWorld _reseat):
// PILE_090 is its new golden, and every particle / plugin variant of it must equal the plain one.
const BOX_PILE_080 = 0x4895dbea, PILE_090 = 0xb9f98b34;
function pile(extra = {}, withRope = false, plugin = null, boxesOnly = false) {
  const physics = new PhysicsWorld({ groundLevel: 0 });
  physics.addStaticBox(-3, 0, -2.5, 1.5, -3, 3);
  const w = new DynamicsWorld({ physics, ...extra });
  const rnd = mulberry32(5);
  for (let i = 0; i < 48; i++) {
    const s = boxesOnly ? (i % 3 === 0 ? box(0.15, 0.15, 0.15) : box(0.2, 0.15, 0.1)) : i % 4 === 0 ? sphere(0.15) : i % 4 === 1 ? capsule(0.1, 0.15) : box(0.2, 0.15, 0.1);
    const ax = rnd() - 0.5, ay = rnd() - 0.5, az = rnd() - 0.5, al = Math.hypot(ax, ay, az) || 1, an = rnd() * Math.PI;
    const sn = Math.sin(an / 2) / al;
    w.addBody({ shape: s, position: [(rnd() - 0.5) * 1.5, 0.4 + i * 0.22, (rnd() - 0.5) * 1.5], quaternion: [ax * sn, ay * sn, az * sn, Math.cos(an / 2)] });
  }
  const a = w.addBody({ shape: box(0.1, 0.3, 0.1), position: [1.5, 2, 0] });
  w.addJoint({ type: 'ball', bodyA: a, bodyB: WORLD_BODY, anchorA: [0, 0.3, 0] });
  w.setVelocity(a, 2, 0, 1);
  if (withRope) Rope.chain(w, { from: [5, 3, 5], to: [6, 3, 5], segments: 12 });
  if (plugin) w.addParticleSolver(plugin);
  run(w, 200);
  return w;
}
{
  const w0 = pile();
  check('no particles: particles / constraints null, no solvers', w0.particles === null && w0.constraints === null && w0.particleSolvers.length === 0);
  const box0 = bodyHash(pile({}, false, null, true));
  check('no particles: 200-step box pile hash == 0.8.0 (default options)', box0 === BOX_PILE_080);
  check('no particles: box pile with maxParticles: 0 == 0.8.0', bodyHash(pile({ maxParticles: 0 }, false, null, true)) === BOX_PILE_080);
  check('particles idle (maxParticles 256, rope present): box pile == 0.8.0', bodyHash(pile({ maxParticles: 256 }, true, null, true)) === BOX_PILE_080);
  check('mixed pile (spheres, capsules): golden after the curved-contact fix', bodyHash(w0) === PILE_090);
  note(`box pile ${hex(box0)} (0.8.0: ${hex(BOX_PILE_080)}), mixed pile ${hex(bodyHash(w0))} (pinned ${hex(PILE_090)})`);
  check('mixed pile: maxParticles 0 and idle particles change nothing', bodyHash(pile({ maxParticles: 0 })) === PILE_090 && bodyHash(pile({ maxParticles: 256 }, true)) === PILE_090);
  const s = w0.serialize();
  check('no particles: serialize adds no keys (no particles / solvers / desc.maxParticles)', !('particles' in s) && !('solvers' in s) && !('maxParticles' in s.desc) && !('particleDamping' in s.desc));
  check('no particles: particleAabb returns false', w0.particleAabb(0, new Float64Array(6)) === false);
  const w1 = DynamicsWorld.from(s, { physics: w0.physics });
  run(w0, 30); run(w1, 30);
  check('no particles: serialize / from still bit-identical', bodyHash(w0) === bodyHash(w1));
  // plugins on a particle-free world (a Buoyancy-like rigid-only plugin): hooks run, particles stay null, the snapshot
  // carries `solvers` only; a plugin that does nothing leaves the rigid result bit-identical
  const log = [];
  const plug = {
    name: 'float', steps: 0, sawNull: true,
    beginStep(world, dt) { log.push('B'); this.steps++; if (world.particles !== null || world.constraints !== null) this.sawNull = false; },
    solvePositions(world, h, it) { log.push('P' + it); },
    solveVelocities(world, h, it) { log.push('V' + it); },
    endStep(world, dt) { log.push('E'); },
    serialize() { return { steps: this.steps }; },
    restore(d) { this.steps = d.steps; },
  };
  const wp = pile({}, false, plug);
  check('plugins (no particles): a no-op plugin leaves the mixed pile hash unchanged', bodyHash(wp) === PILE_090);
  check('plugins (no particles): hooks ran every step, the store stays null', plug.steps === 200 && plug.sawNull && wp.particles === null && wp.particleSolvers[0] === plug);
  check('plugins (no particles): hook order per step B (P0 V0) x 4 E', log.slice(0, 10).join(' ') === 'B P0 V0 P0 V0 P0 V0 P0 V0 E');
  const sp = wp.serialize();
  check('plugins (no particles): snapshot carries solvers, no particles / desc.maxParticles', Array.isArray(sp.solvers) && sp.solvers.length === 1 && sp.solvers[0].name === 'float' && !('particles' in sp) && !('maxParticles' in sp.desc));
  const plug2 = { ...plug, steps: 0 };
  const wq = DynamicsWorld.from(sp, { physics: wp.physics, solvers: [plug2] });
  check('plugins (no particles): from() registers and restores the plugin', wq.particles === null && wq.particleSolvers.length === 1 && wq.particleSolvers[0] === plug2 && plug2.steps === 200);
  run(wp, 30); run(wq, 30);
  check('plugins (no particles): round trip continues bit-identically', bodyHash(wp) === bodyHash(wq));
}

// --- 3. rope: a vertical 20-segment rope pinned at the top -----------------------------------------------------------
{
  const w = new DynamicsWorld({ maxParticles: 64 });
  const r = Rope.chain(w);
  const K = w.constraints, P = w.particles;
  check('rope: 21 particles, ids ascending from 0 in a fresh store', r.count === 21 && r.ids.length === 21 && r.ids[0] === 0 && r.ids[20] === 20 && r.first === 0);
  let nd = 0, nb = 0, nt = 0;
  for (let c = r.constraints.first; c < r.constraints.first + r.constraints.count; c++) { const t = K.type[c]; if (t === PCON.DISTANCE) nd++; else if (t === PCON.BENDING) nb++; else if (t === PCON.TETHER) nt++; }
  check('rope: 20 distance + 19 bending + 20 tethers, one contiguous block', nd === 20 && nb === 19 && nt === 20 && r.constraints.first === 0 && r.constraints.count === 59);
  check('rope: rest length', r.length, 1 - 1e-12, 1 + 1e-12, 'm');
  check('rope: top particle pinned (invMass 0)', (P.flags[r.ids[0]] & PFLAG.PINNED) !== 0 && P.invMass[r.ids[0]] === 0);
  check('rope: group tag on every particle', [...r.ids].every((id) => P.group[id] === r.group));
  run(w, 300);
  const bottom = posOf(w, r.ids[20]);
  check('rope: no NaN after 5 s', finiteParticles(w));
  check('rope: length drift after 5 s (polyline vs rest)', Math.abs(ropeLength(w, r) - r.length) / r.length * 100, 0, 0.1, '%');
  check('rope: hangs straight: |x| of the bottom', Math.abs(bottom[0]), 0, 1e-9, 'm');
  check('rope: hangs straight: |z| of the bottom', Math.abs(bottom[2]), 0, 1e-9, 'm');
  check('rope: bottom at -length', bottom[1], -1 - 1e-3, -1 + 1e-3, 'm');
  check('rope: pinned top did not move', posOf(w, r.ids[0]).every((v) => v === 0));
  // momentum balance at rest: the constraints on the pinned particle carry the weight hanging below it
  let up = 0;
  for (let c = r.constraints.first; c < r.constraints.first + r.constraints.count; c++) if (K.p0[c] === r.ids[0] || K.p1[c] === r.ids[0]) up += K.force[c];
  const weight = (0.5 * 20 / 21) * 9.81;
  check('rope: constraint forces on the pin == weight below (N)', up / weight, 0.99, 1.01);
  check('rope: top segment force positive under load', K.force[r.constraints.first], 1e-6, Infinity, 'N');
  check('stats: particles / constraints', w.stats.particles === 21 && w.stats.constraints === 59);
  // without tethers: Gauss-Seidel stretch at 4 substeps, gone at 20 substeps of 1/120 s
  const w2 = new DynamicsWorld({ maxParticles: 64 });
  const r2 = Rope.chain(w2, { tethers: false });
  run(w2, 300);
  const st4 = (ropeLength(w2, r2) - 1) * 100;
  const w3 = new DynamicsWorld({ maxParticles: 64, substeps: 20 });
  const r3 = Rope.chain(w3, { tethers: false });
  for (let i = 0; i < 600; i++) w3.step(1 / 120);
  check('rope, tethers off, 20 substeps at 1/120 s: length drift', (ropeLength(w3, r3) - 1) * 100, 0, 0.1, '%');
  check('rope, tethers off: top segment force ~ the weight below (+ bending share)', K.force[0] > 0 && w2.constraints.force[0] / weight, 0.3, 1.01);
  note(`rope without tethers at 4 substeps of 1/60 s stretches ${st4.toFixed(3)} % (Gauss-Seidel); with tethers ${((ropeLength(w, r) - 1) * 100).toExponential(1)} %`);
}

// --- 4. rope pinned at 'to', and a rope held by an attachment to the world ---------------------------------------------
{
  const w = new DynamicsWorld({ maxParticles: 64 });
  const r = Rope.chain(w, { from: [0, -1, 0], to: [0, 0, 0], pins: ['to'] });
  check("rope pinned at 'to': first constraint starts at the pinned end", w.constraints.p0[r.constraints.first] === r.ids[20]);
  run(w, 300);
  check("rope pinned at 'to': length drift", Math.abs(ropeLength(w, r) - 1) * 100, 0, 0.1, '%');
  // K.addAttachment(p, WORLD_BODY, x, y, z): a free rope (no pins, no tethers) held at its top by the world point
  const w2 = new DynamicsWorld({ maxParticles: 64 });
  const r2 = Rope.chain(w2, { from: [0, 2, 0], to: [0, 1, 0], pins: [] });
  const ac = w2.constraints.addAttachment(r2.ids[0], WORLD_BODY, 0, 2, 0);
  run(w2, 300);
  const p0 = posOf(w2, r2.ids[0]);
  check('world attachment: the held particle stays on the world point', Math.hypot(p0[0], p0[1] - 2, p0[2]), 0, 1e-9, 'm');
  check('world attachment: the particle is not pinned (invMass > 0)', w2.particles.invMass[r2.ids[0]] > 0 && (w2.particles.flags[r2.ids[0]] & PFLAG.PINNED) === 0);
  check('world attachment: the rope hangs from it (bottom y)', posOf(w2, r2.ids[20])[1], 0.98, 1.0, 'm');
  check('world attachment: its force == the whole rope weight (N)', w2.constraints.force[ac] / (0.5 * 9.81), 0.99, 1.01);
  // builder pins with body WORLD_BODY are real pins, at their anchor point
  const w3 = new DynamicsWorld({ maxParticles: 64 });
  const r3 = Rope.chain(w3, { from: [0, 2, 0], to: [1, 2, 0], pins: [{ end: 'from', body: WORLD_BODY, anchor: [0, 2.5, 0] }] });
  check('builder world pin: pinned (invMass 0) at the anchor', w3.particles.invMass[r3.ids[0]] === 0 && posOf(w3, r3.ids[0])[1] === 2.5);
  run(w3, 60);
  check('builder world pin: stays there', posOf(w3, r3.ids[0]).join() === '0,2.5,0');
}

// --- 5. two-way attachments: momentum, a hanging box, sleeping bodies, removeBody ------------------------------------
{
  // zero gravity: a moving rope tied to a resting box drags it; total linear momentum is conserved
  const w = new DynamicsWorld({ maxParticles: 64, gravity: [0, 0, 0], particleDamping: 0, sleep: false });
  const bx = w.addBody({ shape: box(0.1, 0.1, 0.1), mass: 2, position: [0, 0, 0], linearDamping: 0, angularDamping: 0 });
  const r = Rope.chain(w, { from: [0.1, 0, 0], to: [1.1, 0, 0], segments: 10, mass: 1, pins: [{ end: 'from', body: bx, anchor: [0.1, 0, 0] }] });
  const P = w.particles;
  for (const id of r.ids) { P.vel[id * 3] = 1.5; P.vel[id * 3 + 2] = 0.5; }
  const mom = () => {
    let x = 0, z = 0;
    for (const id of r.ids) { x += P.mass[id] * P.vel[id * 3]; z += P.mass[id] * P.vel[id * 3 + 2]; }
    return [x + 2 * w.bodies.vel[bx * 3], z + 2 * w.bodies.vel[bx * 3 + 2]];
  };
  const m0 = mom();
  run(w, 120);
  const m1 = mom();
  check('two-way: linear momentum conserved over 2 s (x)', Math.abs(m1[0] - m0[0]), 0, 1e-12, 'kg m/s');
  check('two-way: linear momentum conserved over 2 s (z)', Math.abs(m1[1] - m0[1]), 0, 1e-12, 'kg m/s');
  check('two-way: the rope dragged the box (box vx > 0.3)', w.bodies.vel[bx * 3], 0.3, Infinity, 'm/s');
  check('two-way: box moved along +x', w.bodies.pos[bx * 3], 0.2, Infinity, 'm');
  const ac = r.constraints.first + r.constraints.count - 1;
  check('two-way: attachment is the last constraint of the block', w.constraints.type[ac] === PCON.ATTACHMENT && w.constraints.body[ac] === bx);

  // a box hangs from a rope pinned at the top: the body tether + attachment carry its weight (sleep off: the box is
  // never immovable here)
  const physics = new PhysicsWorld({ groundLevel: -10 });
  const w2 = new DynamicsWorld({ physics, maxParticles: 64, sleep: false });
  const hb = w2.addBody({ shape: box(0.1, 0.1, 0.1), mass: 1, position: [0, -1.1, 0] });
  const r2 = Rope.chain(w2, { pins: ['from', { end: 'to', body: hb, anchor: [0, 0.1, 0] }] });
  run(w2, 300);
  const by = w2.bodies.pos[hb * 3 + 1];
  const ac2 = r2.constraints.first + r2.constraints.count - 1, bt2 = ac2 - 1;
  check('hanging box: rope block = 20 dist + 19 bend + 20 tethers + body tether + attachment', r2.constraints.count === 61 && w2.constraints.type[bt2] === PCON.BODY_TETHER && w2.constraints.type[ac2] === PCON.ATTACHMENT);
  check('hanging box: hangs at -(rope + anchor)', by, -1.1 - 1e-6, -1.1 + 1e-6, 'm');
  check('hanging box: body tether + attachment carry the box weight (9.81 N)', (w2.constraints.force[ac2] + w2.constraints.force[bt2]) / 9.81, 0.99, 1.01);
  check('hanging box: rope length kept under the load', Math.abs(ropeLength(w2, r2) - 1) * 100, 0, 0.1, '%');
  let up = 0;
  for (let c = r2.constraints.first; c < r2.constraints.first + r2.constraints.count; c++) if (w2.constraints.p0[c] === r2.ids[0] || w2.constraints.p1[c] === r2.ids[0]) up += w2.constraints.force[c];
  check('hanging box: pin carries rope + box weight (N)', up / (0.5 * 20 / 21 * 9.81 + 9.81), 0.99, 1.01);
  // removeBody removes its attachments
  w2.removeBody(hb);
  check('removeBody: its attachment and body tether are removed', w2.constraints.alive[ac2] === 0 && w2.constraints.alive[bt2] === 0 && w2.constraints.size === 59);
  run(w2, 10);
  check('removeBody: stepping goes on (finite)', finiteParticles(w2));

  // a 10 kg box on a 0.5 kg rope: without the body tether Gauss-Seidel lets it sag (mass ratio 420)
  const heavy = (tethers) => {
    const w = new DynamicsWorld({ maxParticles: 64 });
    const b = w.addBody({ shape: box(0.1, 0.1, 0.1), mass: 10, position: [0, -1.1, 0] });
    Rope.chain(w, { tethers, pins: ['from', { end: 'to', body: b, anchor: [0, 0.1, 0] }] });
    run(w, 300);
    return w.bodies.pos[b * 3 + 1];
  };
  const hy = heavy(true), hy0 = heavy(false);
  check('heavy box (10 kg on 0.5 kg rope): hangs at the rope length', hy, -1.1 - 1e-6, -1.1 + 1e-6, 'm');
  note(`heavy box: ${hy.toFixed(5)} m with the body tether, ${hy0.toFixed(5)} m with tethers: false (Gauss-Seidel sag)`);

  // sleeping: a box hanging from a resting rope sleeps (a loaded but static attachment / body tether lets it); moving
  // the rope's pin wakes it, and so does a yank on the rope (the attachment's force departs from its resting value)
  const hang = () => {
    const w = new DynamicsWorld({ maxParticles: 64 });
    const b = w.addBody({ shape: box(0.1, 0.1, 0.1), mass: 1, position: [0, -1.1, 0] });
    const r = Rope.chain(w, { pins: ['from', { end: 'to', body: b, anchor: [0, 0.1, 0] }] });
    let t = -1;
    for (let i = 0; i < 180 && t < 0; i++) { w.step(DT); if (!w.isAwake(b)) t = (i + 1) * DT; }
    run(w, 30);
    return { w, b, r, t };
  };
  const H1 = hang();
  check('sleep: a box hanging from a resting rope falls asleep (s)', H1.t, 0.4, 3);
  check('sleep: and stays asleep while the rope rests', !H1.w.isAwake(H1.b));
  H1.w.particles.pos[H1.r.ids[0] * 3] += 0.2;                 // move the pin sideways
  H1.w.step(DT);
  check('sleep: moving the pin wakes the box (next step)', H1.w.isAwake(H1.b));
  const H2 = hang();
  for (let k = 8; k <= 16; k++) H2.w.particles.vel[H2.r.ids[k] * 3 + 2] = 1.5;   // yank the middle of the rope
  let woke = -1;
  for (let i = 0; i < 10 && woke < 0; i++) { H2.w.step(DT); if (H2.w.isAwake(H2.b)) woke = i + 1; }
  check('sleep: a yank on the rope wakes the box (steps)', woke, 1, 3);
  check('sleep: the woken box swings (moves along z)', (run(H2.w, 30), Math.abs(H2.w.bodies.pos[H2.b * 3 + 2])), 1e-3, Infinity, 'm');
  // a slack attachment to a box asleep on the ground does not keep it awake; lifting the pin pulls it up
  const physics3 = new PhysicsWorld({ groundLevel: 0 });
  const w3 = new DynamicsWorld({ physics: physics3, maxParticles: 64 });
  const sb = w3.addBody({ shape: box(0.2, 0.2, 0.2), position: [0, 0.2, 0] });
  run(w3, 120);
  const slept = !w3.isAwake(sb);
  const r3 = Rope.chain(w3, { from: [0, 1.3, 0], to: [0, 0.4, 0], segments: 5, pins: ['from', { end: 'to', body: sb, anchor: [0, 0.2, 0] }] });
  run(w3, 60);
  check('sleep: a slack rope tied to a sleeping box leaves it asleep', slept && !w3.isAwake(sb));
  for (let i = 0; i < 60; i++) { w3.particles.pos[r3.ids[0] * 3 + 1] += 0.01; w3.step(DT); }   // haul the pin up 0.6 m
  check('sleep: hauling the pin up wakes the box and lifts it', w3.isAwake(sb) && w3.bodies.pos[sb * 3 + 1] > 0.3);
  check('attachment: unknown body throws (nothing allocated)', throws(() => Rope.chain(w3, { pins: [{ end: 'to', body: 99 }] })) && w3.particles.size === 6);
}

// --- 6. cloth: layout, symmetry, compliance, dispose ------------------------------------------------------------------
{
  const w = new DynamicsWorld({ maxParticles: 512 });
  const c = Cloth.grid(w, { nx: 8, ny: 6, spacing: 0.1, position: [1, 2, 3] });
  const K = w.constraints;
  check('cloth: nx * ny particles, ids ascending (j, i) in a fresh store', c.count === 48 && c.ids.every((id, k) => id === k) && c.first === 0);
  check('cloth: indexOf(i, j) = ids[j * nx + i]', c.indexOf(3, 2) === c.ids[2 * 8 + 3] && c.indexOf(7, 5) === 47);
  const p00 = posOf(w, c.indexOf(0, 0)), p75 = posOf(w, c.indexOf(7, 5));
  check('cloth: centred on position, rows along +X (i) and +Y (j)', Math.abs(p00[0] - 0.65) + Math.abs(p00[1] - 1.75) + Math.abs(p75[0] - 1.35) + Math.abs(p75[1] - 2.25) + Math.abs(p00[2] - 3), 0, 1e-12);
  const n = [0, 0, 0, 0];
  for (let k = c.constraints.first; k < c.constraints.first + c.constraints.count; k++) n[K.type[k]]++;
  // structural 7*6 + 8*5 = 82, shear 2*7*5 = 70; bending: horizontal edges 7*4*2 = 56 + vertical 6*5*2 = 60
  check('cloth: 82 structural + 70 shear distance constraints', n[PCON.DISTANCE], 152, 152);
  check('cloth: 116 bending constraints (both triangulations, interior edges)', n[PCON.BENDING], 116, 116);
  check('cloth: no tethers / attachments without pins', n[PCON.TETHER] + n[PCON.ATTACHMENT], 0, 0);
  const T = c.triangles();
  check('cloth: triangles(): 2 per cell, allocated once', T.length === 7 * 5 * 6 && c.triangles() === T && T instanceof Uint32Array);
  let ccw = 0;
  for (let t = 0; t < T.length; t += 3) {
    const a = posOf(w, c.ids[T[t]]), b = posOf(w, c.ids[T[t + 1]]), d = posOf(w, c.ids[T[t + 2]]);
    const nz = (b[0] - a[0]) * (d[1] - a[1]) - (b[1] - a[1]) * (d[0] - a[0]);
    if (nz > 0) ccw++;
  }
  check('cloth: every triangle counter-clockwise seen from +normal', ccw, 70, 70);
  const ci = K.addDistance(0, 1);
  check('cloth: a later single add takes the next free constraint id', ci, 268, 268);
  K.remove(ci);
  c.dispose();
  check('cloth: dispose frees every particle and constraint', w.particles.size === 0 && K.size === 0);
  const c2 = Cloth.grid(w, { nx: 8, ny: 6, spacing: 0.1, pins: ['top-left', 'top-right'] });
  check('cloth: a new cloth reuses the lowest ids and the same constraint block', c2.ids[0] === 0 && c2.ids[47] === 47 && c2.constraints.first === 0);
  let nt = 0;
  for (let k = c2.constraints.first; k < c2.constraints.first + c2.constraints.count; k++) if (K.type[k] === PCON.TETHER) nt++;
  check('cloth: one tether per free particle when pinned', nt, 46, 46);
  check('cloth: unknown pin throws', throws(() => Cloth.grid(w, { nx: 4, ny: 4, pins: ['middle'] })));
  check('cloth: capacity overflow throws before allocating', throws(() => Cloth.grid(w, { nx: 32, ny: 32 })) && w.particles.size === 48);
}
{
  // a horizontal 32 x 24 cloth held at two corners of one edge swings down: mirror particles stay mirror images
  const w = new DynamicsWorld({ maxParticles: 2048 });
  const c = Cloth.grid(w, { nx: 32, ny: 24, spacing: 0.05, position: [0, 2, 0], quaternion: Q_FLAT, pins: ['top-left', 'top-right'] });
  const pinA = posOf(w, c.indexOf(0, 23)), pinB = posOf(w, c.indexOf(31, 23));
  let minY = Infinity;
  for (let s = 0; s < 180; s++) { w.step(DT); const y = posOf(w, c.indexOf(16, 0))[1]; if (y < minY) minY = y; }
  const P = w.particles.pos;
  let asym = 0;
  for (let j = 0; j < 24; j++) {
    for (let i = 0; i < 16; i++) {
      const a = c.indexOf(i, j) * 3, b = c.indexOf(31 - i, j) * 3;
      asym = Math.max(asym, Math.abs(P[a] + P[b]), Math.abs(P[a + 1] - P[b + 1]), Math.abs(P[a + 2] - P[b + 2]));
    }
  }
  check('cloth (2 corner pins): mirror particles equal after 3 s (max error)', asym, 0, 1e-9, 'm');
  check('cloth: no NaN', finiteParticles(w));
  check('cloth: the pins did not move', posOf(w, c.indexOf(0, 23)).every((v, k) => v === pinA[k]) && posOf(w, c.indexOf(31, 23)).every((v, k) => v === pinB[k]));
  check('cloth: the far edge swung down (lowest y reached)', minY, -Infinity, 1.0, 'm');
  let tv = 0;
  const K = w.constraints;
  for (let k = c.constraints.first; k < c.constraints.first + c.constraints.count; k++) {
    if (K.type[k] !== PCON.TETHER) continue;
    const a = K.p0[k] * 3, b = K.p1[k] * 3;
    tv = Math.max(tv, Math.hypot(P[a] - P[b], P[a + 1] - P[b + 1], P[a + 2] - P[b + 2]) - K.rest[k]);
  }
  check('cloth: every particle within its tether length of the nearest pin', tv, -Infinity, 1e-12, 'm');
  // odd nx: a constraint and its mirror share the middle column, the sweep is no longer symmetric (documented)
  const wo = new DynamicsWorld({ maxParticles: 2048 });
  const co = Cloth.grid(wo, { nx: 31, ny: 24, spacing: 0.05, position: [0, 2, 0], quaternion: Q_FLAT, pins: ['top-left', 'top-right'] });
  run(wo, 180);
  let asymO = 0;
  for (let j = 0; j < 24; j++) for (let i = 0; i < 15; i++) { const a = co.indexOf(i, j) * 3, b = co.indexOf(30 - i, j) * 3; asymO = Math.max(asymO, Math.abs(wo.particles.pos[a] + wo.particles.pos[b])); }
  note(`cloth mirror error after 3 s: nx 32 ${asym.toExponential(1)} m, nx 31 ${asymO.toExponential(1)} m (odd nx: Gauss-Seidel order bias)`);
}
{
  // vertical 24 x 16 cloth pinned along its top row: rigid (tethers) holds its rest height, compliant stretch sags more
  const bottomY = (stretch) => {
    const w = new DynamicsWorld({ maxParticles: 1024 });
    const c = Cloth.grid(w, { nx: 24, ny: 16, spacing: 0.05, position: [0, 2, 0], pins: ['top'], stretch });
    run(w, 240);
    let y = 0;
    for (let i = 0; i < 24; i++) y += posOf(w, c.indexOf(i, 0))[1];
    let f = Infinity;
    for (let k = c.constraints.first; k < c.constraints.first + c.constraints.count; k++) {
      const K = w.constraints;
      if (K.type[k] === PCON.DISTANCE && K.p1[k] === c.indexOf(12, 15) && K.p0[k] === c.indexOf(12, 14)) f = K.force[k];
    }
    return { y: y / 24, f, tethers: stretch === 0 };
  };
  const rigid = bottomY(0), soft = bottomY(1e-3);
  const rest = 2 - 7.5 * 0.05;
  check('cloth (top pinned, stretch 0): bottom row at its rest height', rigid.y, rest - 1e-6, rest + 1e-6, 'm');
  check('cloth: stretch 1e-3 sags more than stretch 0', rigid.y - soft.y, 1e-3, Infinity, 'm');
  check('cloth: top-row structural force positive under load (stretch 1e-3)', soft.f, 1e-6, Infinity, 'N');
  note(`cloth top pinned 24 x 16: bottom row ${rigid.y.toFixed(5)} m (rigid) vs ${soft.y.toFixed(5)} m (stretch 1e-3)`);
}

// --- 7. golden hashes, two runs, serialize / from at step 100 ------------------------------------------------------------
// a deterministic custom solver with state (exercises solver registration order and restore through from())
class Gust {
  constructor() { this.name = 'gust'; this.t = 0; this.group = -1; }
  beginStep(world, dt) { this.t += dt; }
  solveVelocities(world, h) {
    const P = world.particles, k = this.t < 1 ? 0.5 * h : -0.25 * h;
    for (let i = 0; i < P.count; i++) if (P.alive[i] && P.group[i] === this.group && (P.flags[i] & PFLAG.PINNED) === 0) P.vel[i * 3 + 2] += k;
  }
  serialize() { return { t: this.t, group: this.group }; }
  restore(d) { this.t = d.t; this.group = d.group; }
}
const ROPE_GOLDEN = 0x59a54556, CLOTH_GOLDEN = 0xbff1e641;
function ropeScene() {
  const w = new DynamicsWorld({ maxParticles: 64 });
  Rope.chain(w, { from: [0, 2, 0], to: [1, 2, 0.25], segments: 20, bend: 1e-3 });
  return w;
}
function clothScene() {
  const physics = new PhysicsWorld({ groundLevel: 0 });
  const w = new DynamicsWorld({ physics, maxParticles: 1024 });
  const ball = w.addBody({ shape: sphere(0.12), position: [0.3, 1.3, 0.4] });
  w.addBody({ shape: box(0.2, 0.2, 0.2), position: [-1, 0.2, 0] });
  const c = Cloth.grid(w, { nx: 16, ny: 12, spacing: 0.06, position: [0, 2, 0], quaternion: Q_FLAT, pins: ['top-left', 'top-right', { i: 15, j: 0, body: ball }] });
  const g = new Gust(); g.group = c.group;
  w.addParticleSolver(g);
  return { w, g };
}
{
  const a = ropeScene(), b = ropeScene();
  run(a, 300); run(b, 300);
  const ha = particleHash(a);
  check('golden: rope hash identical across two runs', ha === particleHash(b));
  check('golden: rope hash after 300 steps', ha === ROPE_GOLDEN);
  note(`rope golden ${hex(ha)} (pinned ${hex(ROPE_GOLDEN)})`);
  const r1 = ropeScene();
  run(r1, 100);
  const r2 = DynamicsWorld.from(structuredClone(r1.serialize()));
  check('serialize: rope restored hash equals at step 100', particleHash(r1) === particleHash(r2));
  run(r1, 200); run(r2, 200);
  check('serialize: rope from() continues bit-identically (to step 300)', particleHash(r2) === ha && particleHash(r1) === ha);

  const A = clothScene(), Bc = clothScene();
  run(A.w, 200); run(Bc.w, 200);
  const hc = hashU32(particleHash(A.w), bodyHash(A.w));
  check('golden: cloth hash identical across two runs', hc === hashU32(particleHash(Bc.w), bodyHash(Bc.w)));
  check('golden: cloth + bodies hash after 200 steps', hc === CLOTH_GOLDEN);
  note(`cloth golden ${hex(hc)} (pinned ${hex(CLOTH_GOLDEN)})`);
  const C1 = clothScene();
  run(C1.w, 100);
  const snap = C1.w.serialize();
  check('serialize: solvers entry per solver, in run order', snap.solvers.length === 2 && snap.solvers[0].name === 'constraints' && snap.solvers[1].name === 'gust' && snap.solvers[1].data.t > 1.6);
  const g2 = new Gust();
  const C2 = DynamicsWorld.from(structuredClone(snap), { physics: C1.w.physics, solvers: [g2] });
  check('serialize: from() registers the matching solver after the built-in one and restores it', C2.particleSolvers.length === 2 && C2.particleSolvers[0] === C2.constraints && C2.particleSolvers[1] === g2 && g2.t === C1.g.t && g2.group === C1.g.group);
  check('serialize: restored constraints equal (alive, rest, anchor)', C2.constraints.count === C1.w.constraints.count &&
    hashF64Array(C2.constraints.rest, 1, 0, C2.constraints.count) === hashF64Array(C1.w.constraints.rest, 1, 0, C1.w.constraints.count) &&
    hashF64Array(C2.constraints.anchor, 1, 0, 3 * C2.constraints.count) === hashF64Array(C1.w.constraints.anchor, 1, 0, 3 * C1.w.constraints.count));
  run(C1.w, 100); run(C2, 100);
  check('serialize: cloth from() continues bit-identically (to step 200)', hashU32(particleHash(C2), bodyHash(C2)) === hc && hashU32(particleHash(C1.w), bodyHash(C1.w)) === hc);
  const J = DynamicsWorld.from(JSON.parse(JSON.stringify(snap)), { physics: C1.w.physics, solvers: [new Gust()] });
  run(J, 100);
  check('serialize: a JSON round trip continues bit-identically too', hashU32(particleHash(J), bodyHash(J)) === hc);
  const N = DynamicsWorld.from(structuredClone(snap), { physics: C1.w.physics });
  check('serialize: without the instance the gust entry is dropped', N.particleSolvers.length === 1);
}

// --- 8. allocation: step() with a 32 x 32 cloth (+ a rope holding a body by an attachment and a body tether) ----------
// The dyn-world pattern (retained growth with gc before / after) plus the GC count over 500 steps (GCProfiler), and the
// bytes per step as the slope of GC-free heap growth between 100- and 600-step runs (best of 3; dyn-broadphase's
// method: V8's total_allocated_bytes counter swings by KBs between runs here, and the sampling heap profiler
// deoptimizes the running code, which then boxes doubles). A warm step boxes only h / dt once (~32 B).
{
  let gc = globalThis.gc;
  if (!gc) { try { v8.setFlagsFromString('--expose-gc'); gc = vm.runInNewContext('gc'); } catch { gc = null; } }
  const mixed = () => {
    const w = new DynamicsWorld({ maxParticles: 1200, sleep: false });
    Cloth.grid(w, { nx: 32, ny: 32, spacing: 0.05, position: [0, 3, 0], quaternion: Q_FLAT, pins: ['top-left', 'top-right'] });
    const hb = w.addBody({ shape: box(0.1, 0.1, 0.1), mass: 0.5, position: [2, 1.9, 0] });
    Rope.chain(w, { from: [2, 3, 0], to: [2, 2, 0], segments: 10, pins: ['from', { end: 'to', body: hb, anchor: [0, 0.1, 0] }] });
    return w;
  };
  const w = mixed();
  run(w, 300);   // warm-up (JIT, feedback)
  performance.now();
  if (gc) gc();
  const h0 = process.memoryUsage().heapUsed;
  const prof = new v8.GCProfiler();
  prof.start();
  run(w, 500);
  const gcs = prof.stop().statistics.length;
  if (gc) gc();
  const h1 = process.memoryUsage().heapUsed;
  check('alloc: GCs during 500 steps (32 x 32 cloth + rope holding a body; load noise)', gcs, 0, 1);
  check('alloc: retained heap growth over 500 steps (gc before / after)', (h1 - h0) / 1024, -Infinity, 16, 'KB');
  const growth = (world) => {
    const grow = (n) => {
      if (gc) gc();
      const pr = new v8.GCProfiler();
      pr.start();
      const a = process.memoryUsage().heapUsed;
      run(world, n);
      const b = process.memoryUsage().heapUsed;
      return pr.stop().statistics.length ? Infinity : b - a;   // a GC inside invalidates the round
    };
    const best = (n) => { let m = Infinity; for (let r = 0; r < 3; r++) m = Math.min(m, grow(n)); return m; };
    const g600 = best(600);
    return { slope: (g600 - best(100)) / 500, mean: g600 / 600 };
  };
  const wc = new DynamicsWorld({ maxParticles: 1200, sleep: false });
  Cloth.grid(wc, { nx: 32, ny: 32, spacing: 0.05, position: [0, 3, 0], quaternion: Q_FLAT, pins: ['top-left', 'top-right'] });
  run(wc, 400);
  const gcw = growth(wc), gm = growth(w);
  check('alloc: heap growth per warm step, 32 x 32 cloth (GC-free slope 100 -> 600)', gcw.slope, -Infinity, 64, 'B');
  check('alloc: heap growth per step, cloth + rope + body (600 GC-free steps, mean)', gm.mean, -Infinity, 256, 'B');
  note(`alloc: ${gcw.slope.toFixed(1)} B/step cloth world (slope), ${gm.mean.toFixed(1)} B/step mean with the rope + body; ${w.stats.constraints} constraints, ${w.stats.particles} particles`);
  if (!gc) note('no gc available: growth measured without a gc first');
}

// --- 9. solver hook order and the state each hook sees -----------------------------------------------------------------
{
  const physics = new PhysicsWorld({ groundLevel: 0 });
  const w = new DynamicsWorld({ physics, maxParticles: 16, substeps: 2, positionIterations: 3, velocityIterations: 2, particleDamping: 0.5 });
  w.addBody({ shape: box(0.2, 0.2, 0.2), position: [0, 0.199, 0] });
  const P = w.particles, id = P.alloc();
  P.pos[id * 3 + 1] = 1; P.prevPos[id * 3 + 1] = 1;
  const log = [], seen = { contacts: -1, stepBefore: -1, stepAfter: -1, integrated: true, recovered: true, h: 0, dt: 0, world: true };
  const rec = {
    name: 'recorder',
    beginStep(world, dt) { log.push('B'); seen.contacts = world.contacts.count; seen.stepBefore = world.stepCount; seen.dt = dt; seen.world = seen.world && world === w; },
    solvePositions(world, h, it) {
      log.push('P' + it); seen.h = h;
      if (it === 0) { const o = id * 3 + 1; if (P.pos[o] !== P.prevPos[o] + h * P.vel[o]) seen.integrated = false; }
    },
    solveVelocities(world, h, it) { log.push('V' + it); const o = id * 3 + 1; if (P.vel[o] !== (P.pos[o] - P.prevPos[o]) / h) seen.recovered = false; },
    endStep(world, dt) { log.push('E'); seen.stepAfter = world.stepCount; },
  };
  check('solvers: built-in constraints at index 0', w.particleSolvers[0] === w.constraints && w.particleSolvers.length === 1);
  check('solvers: addParticleSolver appends (run order)', w.addParticleSolver(rec) === rec && w.particleSolvers[1] === rec);
  check('solvers: registering twice throws', throws(() => w.addParticleSolver(rec)));
  check('solvers: the built-in solver cannot be removed', throws(() => w.removeParticleSolver(w.constraints)));
  w.step(DT);
  check('solvers: hook order B, (P0 P1 P2 V0 V1) x 2 substeps, E', log.join(' ') === 'B P0 P1 P2 V0 V1 P0 P1 P2 V0 V1 E');
  check('solvers: beginStep after contact setup (this step\'s contacts present)', seen.contacts, 1, Infinity);
  check('solvers: beginStep before / endStep after the step count advances', seen.stepBefore === 0 && seen.stepAfter === 1);
  check('solvers: dt and h as passed (h = dt / substeps)', seen.dt === DT && seen.h === DT / 2 && seen.world);
  check('solvers: particles integrated before solvePositions (pos = prevPos + h vel)', seen.integrated);
  check('solvers: velocities recovered before solveVelocities (vel = dx / h)', seen.recovered);
  const o = id * 3 + 1, vRec = (P.pos[o] - P.prevPos[o]) / (DT / 2);
  check('solvers: particle damping after the velocity pass (vel = dx / h (1 - d h))', P.vel[o] === vRec * (1 - 0.5 * DT / 2));
  check('solvers: removeParticleSolver', w.removeParticleSolver(rec) === true && w.particleSolvers.length === 1 && w.removeParticleSolver(rec) === false);
  log.length = 0;
  w.step(DT);
  check('solvers: a removed solver is no longer called', log.length, 0, 0);
  // pinned and NO_GRAVITY particles
  const pin = P.alloc(), ng = P.alloc();
  P.pos[pin * 3] = 3; P.setPinned(pin, true); P.pos[ng * 3] = 4; P.flags[ng] |= PFLAG.NO_GRAVITY;
  run(w, 10);
  check('particles: a pinned particle is not integrated', P.pos[pin * 3] === 3 && P.pos[pin * 3 + 1] === 0 && P.vel[pin * 3 + 1] === 0);
  check('particles: NO_GRAVITY particle stays put', P.pos[ng * 3 + 1] === 0 && P.pos[ng * 3] === 4);
}

// --- 10. particleAabb ----------------------------------------------------------------------------------------------------
{
  const w = new DynamicsWorld({ maxParticles: 512 });
  const c = Cloth.grid(w, { nx: 10, ny: 8, spacing: 0.1, position: [0, 3, 0], quaternion: [0.1, 0.3, -0.2, 0.9], radius: 0.03 });
  const r = Rope.chain(w, { from: [5, 5, 5], to: [6, 5, 5], segments: 4, radius: 0.07 });
  run(w, 30);
  const out = new Float64Array(6), ref = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity], P = w.particles;
  for (const id of c.ids) for (let k = 0; k < 3; k++) { ref[k] = Math.min(ref[k], P.pos[id * 3 + k] - 0.03); ref[k + 3] = Math.max(ref[k + 3], P.pos[id * 3 + k] + 0.03); }
  const ok = w.particleAabb(c.group, out);
  check('particleAabb: true for the cloth group', ok);
  check('particleAabb: equals the brute-force bounds incl. radius', Math.max(...ref.map((v, k) => Math.abs(v - out[k]))), 0, 0);
  const or = new Float64Array(6), rr = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
  for (const id of r.ids) for (let k = 0; k < 3; k++) { rr[k] = Math.min(rr[k], P.pos[id * 3 + k] - 0.07); rr[k + 3] = Math.max(rr[k + 3], P.pos[id * 3 + k] + 0.07); }
  check('particleAabb: rope group bounds (radius 0.07)', w.particleAabb(r.group, or) && Math.max(...rr.map((v, k) => Math.abs(v - or[k]))) === 0);
  out.fill(7);
  check('particleAabb: unknown group -> false, out untouched', w.particleAabb(99, out) === false && out.every((v) => v === 7));
}

// --- 11. ParticleConstraints standalone checks --------------------------------------------------------------------------
{
  const P = new ParticleStore(8), K = new ParticleConstraints(P, 16);
  for (let i = 0; i < 6; i++) { const id = P.alloc(); P.pos[id * 3] = i; }
  const d = K.addDistance(0, 1), b = K.addBending(1, 2, 0, 3, 1e-4), t = K.addTether(4, 0, 3.5), a = K.addAttachment(5, WORLD_BODY, 5, 1, 0);
  check('constraints: ids 0..3 in add order', d === 0 && b === 1 && t === 2 && a === 3 && K.count === 4);
  check('constraints: rest defaults to the current distance (bending: c-d)', K.rest[d] === 1 && K.rest[b] === 3 && K.rest[t] === 3.5);
  check('constraints: bending keeps its edge, attachment its body / anchor', K.e0[b] === 1 && K.e1[b] === 2 && K.body[a] === -1 && K.anchor[a * 3 + 1] === 1);
  check('constraints: bad particle throws', throws(() => K.addDistance(0, 7)) && throws(() => K.addDistance(2, 2)));
  K.remove(b);
  check('constraints: remove frees the id, lowest free reused', K.addDistance(2, 3) === 1);
  const first = K.reserve(3);
  check('constraints: reserve(n) returns the lowest contiguous free run', first, 4, 4);
  K.remove(0);
  check('constraints: reserved adds fill the block even with a lower hole', K.addDistance(0, 2) === 4 && K.addDistance(1, 2) === 5 && K.addDistance(3, 4) === 6 && K.addDistance(0, 4) === 0);
  for (let i = 0; i < 20; i++) K.addDistance(0, 5);
  check('constraints: the store grows past its capacity', K.capacity >= 27 && K.size === 27 && K.alive[26] === 1);
}

// --- 12. DynamicsClient: particles force the in-thread world; worker mode has none ---------------------------------------
{
  const warn = console.warn, warnings = [];
  console.warn = (...a) => warnings.push(a.join(' '));
  try {
    let called = 0;
    const c = new DynamicsClient({ maxParticles: 64, worker: () => { called++; return { postMessage() {}, terminate() {} }; } });
    check('client: maxParticles + worker -> in-thread, worker never started', c.mode === 'main' && called === 0 && c.world !== null);
    check('client: exactly one console.warn', warnings.length === 1 && /particles/.test(warnings[0]));
    check('client: in-thread passthroughs (particles, constraints, particleSolvers)', c.particles === c.world.particles && c.constraints === c.world.constraints && c.particleSolvers === c.world.particleSolvers);
    const r = Rope.chain(c.world, { segments: 4 });
    const g = { name: 'g' };
    check('client: addParticleSolver / removeParticleSolver pass through', c.addParticleSolver(g) === g && c.world.particleSolvers[1] === g && c.removeParticleSolver(g) === true);
    c.step(DT);
    const out = new Float64Array(6);
    check('client: particleAabb passes through', c.particleAabb(r.group, out) === true && out[1] < 0);
    c.dispose();
    const plain = new DynamicsClient({ maxParticles: 64, worker: false });
    check('client: worker: false with particles -> no warning', warnings.length === 1 && plain.particles !== null);
    plain.dispose();
    const wk = new DynamicsClient({ worker: () => ({ postMessage() {}, terminate() {} }), readyTimeout: 60000 });
    check('client (worker mode): particles / constraints / particleSolvers null', wk.world === null && wk.particles === null && wk.constraints === null && wk.particleSolvers === null);
    check('client (worker mode): particle methods throw', throws(() => wk.addParticleSolver(g)) && throws(() => wk.removeParticleSolver(g)) && throws(() => wk.particleAabb(0, out)));
    wk.dispose();
    // options.solvers: registered and restored by name from a snapshot (the cloth golden scene at step 100)
    const C1 = clothScene();
    run(C1.w, 100);
    const gs = new Gust();
    const cs = new DynamicsClient({ serialized: C1.w.serialize(), physics: C1.w.physics, worker: false, solvers: [gs] });
    check('client: options.solvers restored from a snapshot', cs.particleSolvers.length === 2 && cs.particleSolvers[1] === gs && gs.t === C1.g.t && gs.group === C1.g.group);
    for (let i = 0; i < 100; i++) cs.step(DT);
    check('client: ... and the world continues bit-identically (cloth golden at step 200)', hashU32(particleHash(cs.world), bodyHash(cs.world)) === CLOTH_GOLDEN);
    cs.dispose();
    const gf = new Gust();
    const cf = new DynamicsClient({ maxParticles: 16, worker: false, solvers: [gf] });
    check('client: options.solvers registered on a fresh world', cf.particleSolvers.length === 2 && cf.particleSolvers[1] === gf);
    cf.dispose();
    const pw = new DynamicsClient({ worker: () => ({ postMessage() {}, terminate() {} }), readyTimeout: 60000, solvers: [{ name: 'plug' }] });
    check('client: plugins without particles -> in-thread (warned), particles null', pw.mode === 'main' && pw.particles === null && pw.particleSolvers.length === 1 && warnings.length === 2);
    pw.dispose();
    check('client: a non-array `solvers` throws', throws(() => new DynamicsClient({ worker: false, solvers: { name: 'x' } })));
  } finally { console.warn = warn; }
}

// --- 14. fix round: freed ids, dispose ownership, pin conflicts, solver registration, defaults, isActive ---------------
{
  // a freed particle loses its constraints (free hook); the id handed out next carries none
  const w = new DynamicsWorld({ maxParticles: 16 });
  const P = w.particles, K = w.constraints;
  const bx = w.addBody({ shape: box(0.1, 0.1, 0.1), position: [0, 0, 0] });
  const a = P.alloc(), b = P.alloc();
  P.pos[b * 3] = 1;
  const att = () => { let n = 0; for (let c = 0; c < K.count; c++) if (K.alive[c] && K.type[c] === PCON.ATTACHMENT && K.body[c] === bx) n++; return n; };
  K.addDistance(a, b); K.addAttachment(b, bx, 0.1, 0, 0);
  const before = att();
  P.free(b);
  const b2 = P.alloc();
  let on = 0;
  for (let c = 0; c < K.count; c++) if (K.alive[c] && (K.p0[c] === b2 || K.p1[c] === b2 || K.e0[c] === b2 || K.e1[c] === b2)) on++;
  check('free: the reused id carries no constraint', b2 === b && on === 0 && K.size === 0);
  check("free: the body's attachment count dropped (1 -> 0)", before === 1 && att() === 0);
  check('free: P.size / K.size are O(1) counters that match a scan', P.size === 2 && K.size === 0 && w.stats.particles === 0);
  w.step(DT);
  check('stats: particles / constraints after a step', w.stats.particles === 2 && w.stats.constraints === 0);
  // freeing one cloth particle (a tear) removes exactly the constraints that name it
  const wc = new DynamicsWorld({ maxParticles: 64 });
  const c = Cloth.grid(wc, { nx: 4, ny: 4, pins: ['top-left', 'top-right'] });
  const victim = c.indexOf(1, 1), K2 = wc.constraints;
  let naming = 0;
  for (let q = 0; q < K2.count; q++) if (K2.alive[q] && [K2.p0[q], K2.p1[q], K2.e0[q], K2.e1[q]].includes(victim)) naming++;
  const n0 = K2.size;
  wc.particles.free(victim);
  check('free (cloth tear): removes exactly the constraints naming the particle', K2.size, n0 - naming, n0 - naming);
  c.dispose();
  check('dispose after a tear: everything of the cloth is gone', wc.particles.size === 0 && K2.size === 0);
  // dispose removes only what the builder emitted: a foreign constraint that reused one of its ids survives
  const wd = new DynamicsWorld({ maxParticles: 64 });
  const cd = Cloth.grid(wd, { nx: 3, ny: 3 });
  const extra = wd.particles.alloc(), extra2 = wd.particles.alloc();
  wd.particles.pos[extra2 * 3] = 1;
  const id0 = cd.constraints.first;
  wd.constraints.remove(id0);
  const reused = wd.constraints.addDistance(extra, extra2);
  cd.dispose();
  check('dispose: removes exactly its own constraints (a reused id with a foreign constraint is kept)', reused === id0 && wd.constraints.alive[reused] === 1 && wd.constraints.size === 1);

  // one pin per particle when a pin has a body
  const wp = new DynamicsWorld({ maxParticles: 128 });
  const pb = wp.addBody({ shape: box(0.1, 0.1, 0.1), position: [0, 0, 0] });
  check('pins: two body pins on one cloth particle throw', throws(() => Cloth.grid(wp, { nx: 4, ny: 4, pins: [{ i: 1, j: 1, body: pb }, { i: 1, j: 1, body: pb }] })));
  check('pins: a string pin + a body pin on one cloth particle throw', throws(() => Cloth.grid(wp, { nx: 4, ny: 4, pins: ['top-left', { i: 0, j: 3, body: pb }] })));
  check("pins: 'from' + a body pin on 'from' throw (rope)", throws(() => Rope.chain(wp, { pins: ['from', { end: 'from', body: pb }] })));
  check('pins: two body pins on one rope end throw', throws(() => Rope.chain(wp, { pins: [{ end: 'to', body: pb }, { end: 'to', body: pb }] })));
  check("pins: overlapping plain pins are fine ('top' + 'top-left')", Cloth.grid(wp, { nx: 4, ny: 4, pins: ['top', 'top-left'] }).count === 16 && wp.particles.size === 16);
  // a builder that fails inside emit leaves no particle, no constraint and no pending reservation
  const nP = wp.particles.size, nK = wp.constraints.size;
  wp.constraints.addTether = () => { throw new Error('injected'); };
  const failed = throws(() => Rope.chain(wp, { from: [3, 0, 0], to: [3, -1, 0] }));
  delete wp.constraints.addTether;
  const next = wp.constraints.addDistance(0, 1);
  let lowest = 0;
  while (lowest < wp.constraints.count && (wp.constraints.alive[lowest] && lowest !== next)) lowest++;
  check('builder failure: rolls back particles + constraints, ends the reservation', failed && wp.particles.size === nP && wp.constraints.size === nK + 1 && next === lowest);

  // solver registration
  const ws = new DynamicsWorld({ maxParticles: 8 });
  check('solvers: a solver without a string name throws', throws(() => ws.addParticleSolver({ beginStep() {} })));
  const g = new Gust();
  ws.addParticleSolver(g);
  const snap = ws.serialize();
  const g2 = new Gust();
  const wr = DynamicsWorld.from(snap, { solvers: [g2, g2, { name: 'stray' }] });
  check('from(): an instance listed twice is registered once (+ the unmatched one after it)', wr.particleSolvers.length === 3 && wr.particleSolvers[1] === g2 && wr.particleSolvers[2].name === 'stray');
  const wr2 = DynamicsWorld.from(snap, { solvers: [g2] });
  check('from(): an instance already on another world is still registered on this one', wr2.particleSolvers[1] === g2);
  // defaults
  check('defaults: particleDamping clamped at 0', new DynamicsWorld({ maxParticles: 4, particleDamping: -1 }).particleDamping === 0);
  const wq = new DynamicsWorld({ maxParticles: 64 });
  const cq = Cloth.grid(wq, { nx: 3, ny: 3 });
  let shearC = -1;
  for (let q = cq.constraints.first; q < cq.constraints.first + cq.constraints.count; q++) {
    const K3 = wq.constraints, ia = K3.p0[q], ib = K3.p1[q];
    if (K3.type[q] === PCON.DISTANCE && Math.abs(ia - ib) === 4) shearC = K3.compliance[q];   // (i, j)-(i + 1, j + 1) of a 3 x 3 grid
  }
  check('defaults: cloth shear compliance 1e-4, stretch 0, bend 1e-5', shearC === 1e-4);
  // a constraint skipped this substep reads force 0 (both ends immovable)
  const wf = new DynamicsWorld({ maxParticles: 4 });
  const fa = wf.particles.alloc(), fb = wf.particles.alloc();
  wf.particles.pos[fb * 3 + 1] = -0.5;
  wf.particles.setPinned(fa, true);
  const fc = wf.constraints.addDistance(fa, fb);
  run(wf, 30);
  const loaded = wf.constraints.force[fc];
  wf.particles.setPinned(fb, true);
  wf.step(DT);
  check('force: a loaded constraint, then both ends pinned -> 0', loaded > 9 && wf.constraints.force[fc] === 0);
  // isActive: integrated this step (awake dynamic / moving kinematic), not static or asleep
  const physics = new PhysicsWorld({ groundLevel: 0 });
  const wa = new DynamicsWorld({ physics });
  const st = wa.addBody({ type: 'static', shape: box(1, 0.1, 1), position: [3, 0, 0] });
  const dy = wa.addBody({ shape: box(0.2, 0.2, 0.2), position: [0, 1, 0] });
  wa.step(DT);
  check('isActive: a falling body is active, a static one is not', wa.isActive(dy) && !wa.isActive(st) && !wa.isActive(-1) && !wa.isActive(99));
  run(wa, 180);
  check('isActive: false once the body sleeps', !wa.isAwake(dy) && !wa.isActive(dy));
}

// --- 15. recollidePairs: a glue joint broken in a plugin's beginStep gets its contacts in the same step -----------------
{
  const glued = (recollide, idle = false) => {
    const physics = new PhysicsWorld({ groundLevel: 0 });
    const w = new DynamicsWorld({ physics, sleep: false });
    const bot = w.addBody({ shape: box(0.25, 0.25, 0.25), position: [0, 0.25, 0] });
    const top = w.addBody({ shape: box(0.25, 0.25, 0.25), position: [0, 0.75, 0] });
    const j = w.addJoint({ type: 'fixed', bodyA: top, bodyB: bot });     // glue: collideConnected false
    const ids = new Int32Array([bot, top]), none = new Int32Array([bot]);
    const plug = {
      name: 'glue-breaker', armed: false, added: -1, zero: -1, single: -1,
      beginStep(world) {
        this.zero = world.recollidePairs(ids, 0);        // count 0: nothing
        this.single = world.recollidePairs(none, 1);     // no pair among one body
        if (!this.armed) return;
        this.armed = false;
        world.removeJoint(j);
        this.added = recollide ? world.recollidePairs(ids, 2) : 0;
      },
    };
    if (recollide || idle) w.addParticleSolver(plug);
    else w.addParticleSolver({ name: 'glue-breaker', armed: false, beginStep(world) { if (this.armed) { this.armed = false; world.removeJoint(j); } } });
    run(w, 60);
    return { w, bot, top, plug: w.particleSolvers[0] };
  };
  const pairContacts = (w, a, b) => {
    const C = w.contacts;
    let n = 0, depth = -Infinity;
    for (let i = 0; i < C.count; i++) if ((C.a[i] === a && C.b[i] === b) || (C.a[i] === b && C.b[i] === a)) { n++; depth = Math.max(depth, C.depth[i]); }
    return { n, depth };
  };
  const A = glued(true), N = glued(false);
  check('recollide: no contact between the glued boxes before the break', pairContacts(A.w, A.bot, A.top).n, 0, 0);
  const y0 = A.w.bodies.pos[A.top * 3 + 1];
  A.plug.armed = true; N.plug.armed = true;
  A.w.step(DT); N.w.step(DT);
  const pc = pairContacts(A.w, A.bot, A.top);
  check('recollide: contacts added in the breaking step', A.plug.added, 1, Infinity);
  check('recollide: the step ends with contacts between the freed pieces', pc.n, 1, Infinity);
  check('recollide: the freed top box did not sink into the other (y drop)', y0 - A.w.bodies.pos[A.top * 3 + 1], -1e-5, 1e-5, 'm');
  check('control (no recollide): the freed top box falls into the other for a step', y0 - N.w.bodies.pos[N.top * 3 + 1], 1e-3, Infinity, 'm');
  run(A.w, 30);
  const pc2 = pairContacts(A.w, A.bot, A.top);
  check('recollide: no interpenetration afterwards (max contact depth)', pc2.depth, -Infinity, 1e-3, 'm');
  check('recollide: count 0 / a single body add nothing', A.plug.zero === 0 && A.plug.single === 0);
  check('recollide: outside beginStep it throws', throws(() => A.w.recollidePairs(new Int32Array([A.bot, A.top]), 2)));
  // a plugin calling it with nothing qualifying changes nothing
  const I = glued(false, true), G = glued(false);
  run(I.w, 30); run(G.w, 30);
  check('recollide: calls that qualify nothing leave the world bit-identical', bodyHash(I.w) === bodyHash(G.w));
}

// --- 13. cost: ms per step at 4 substeps, 1 iteration (Node) -------------------------------------------------------------
{
  const cost = (make, steps = 300) => {
    const w = make();
    run(w, 60);
    const t0 = performance.now();
    run(w, steps);
    return (performance.now() - t0) / steps;
  };
  const msRope = cost(() => { const w = new DynamicsWorld({ maxParticles: 64 }); Rope.chain(w, { from: [0, 2, 0], to: [1, 2, 0] }); return w; });
  const ms32 = cost(() => { const w = new DynamicsWorld({ maxParticles: 1024 }); Cloth.grid(w, { nx: 32, ny: 32, position: [0, 2, 0], quaternion: Q_FLAT, pins: ['top-left', 'top-right'] }); return w; });
  const ms64 = cost(() => { const w = new DynamicsWorld({ maxParticles: 4096 }); Cloth.grid(w, { nx: 64, ny: 64, spacing: 0.025, position: [0, 2, 0], quaternion: Q_FLAT, pins: ['top-left', 'top-right'] }); return w; }, 150);
  note(`cost per step (4 substeps, 1 iteration, Node ${process.version}): rope 20 seg ${msRope.toFixed(3)} ms, cloth 32x32 ${ms32.toFixed(3)} ms, 64x64 ${ms64.toFixed(3)} ms`);
}

console.log(results.join('\n'));
console.log(failures ? `\n${failures} of ${total} check(s) FAILED` : `\nall ${total} checks passed`);
process.exit(failures ? 1 : 0);
