// Headless checks for particle coupling and buoyancy (src/physics/dynamics/coupling.js): closestPointOnShape against
// hand-computed cases for every shape type, the volume samples, particles resting on the flat ground, cloth draped
// over a static box, two-way particle-body contacts (a rope swung into a crate, momentum of a head-on hit), friction on
// a slope (a static body and a heightfield), buoyancy equilibria, drag, a wave surface, floating particles,
// determinism (two runs, serialize / from), allocation in the hooks and per-step timings. No browser, no WebGL.
// Run: npm test  (node test/dyn-buoyancy.mjs). Skips (exit 0) while the modules it needs are not all present.
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import v8 from 'node:v8';
import vm from 'node:vm';

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const NEED = ['state.js', 'shapes.js', 'xpbd.js', 'DynamicsWorld.js', 'particles.js', 'coupling.js'];
const missing = NEED.filter((f) => !existsSync(here(`../src/physics/dynamics/${f}`)));
if (missing.length) {
  console.log(`dyn-buoyancy: SKIPPED, dynamics modules missing: ${missing.join(', ')}`);
  process.exit(0);
}
const { DynamicsWorld } = await import('../src/physics/dynamics/DynamicsWorld.js');
const { box, sphere, capsule, convex, trimesh, compound } = await import('../src/physics/dynamics/shapes.js');
const { Cloth, Rope, PFLAG } = await import('../src/physics/dynamics/particles.js');
const { ParticleCollider, Buoyancy, closestPointOnShape, sampleShape } = await import('../src/physics/dynamics/coupling.js');
const { PhysicsWorld } = await import('../src/physics/PhysicsWorld.js');
const { hashF64Array, hashU32, mulberry32 } = await import('../src/core/mathDet.js');

const DT = 1 / 120;
let failures = 0, passed = 0;
const results = [];
const fmt = (x) => (!Number.isFinite(x) ? String(x) : Number.isInteger(x) ? String(x) : Math.abs(x) < 1e-3 && x !== 0 ? x.toExponential(2) : x.toFixed(4)).padStart(12);
function check(name, value, lo, hi, unit = '') {
  const v = typeof value === 'boolean' ? (value ? 1 : 0) : value;
  const ok = Number.isFinite(v) && v >= lo && v <= hi;
  if (!ok) failures++; else passed++;
  results.push(`${ok ? 'PASS' : 'FAIL'}  ${name.padEnd(66)} ${fmt(v)} ${unit}   [${fmt(lo).trim()}${hi !== lo ? ' .. ' + fmt(hi).trim() : ''}]`);
  return ok;
}
const note = (s) => results.push(`NOTE  ${s}`);
const run = (w, seconds, each) => { const n = Math.round(seconds / DT); for (let i = 0; i < n; i++) { if (each) each(i); w.step(DT); } };
const I7 = [0, 0, 0, 0, 0, 0, 1];
const qAxis = (ax, ay, az, deg) => { const h = (deg * Math.PI) / 360, s = Math.sin(h); return [ax * s, ay * s, az * s, Math.cos(h)]; };
const cp = (shape, pose, x, y, z, thickness) => closestPointOnShape(shape, ...pose, x, y, z, new Float64Array(7), thickness);
// max abs difference of a result against [cx, cy, cz, nx, ny, nz, dist]
const err = (r, e) => { let m = 0; for (let k = 0; k < 7; k++) m = Math.max(m, Math.abs(r[k] - e[k])); return m; };

function addParticle(w, x, y, z, { mass = 0.1, radius = 0.05, vx = 0, vy = 0, vz = 0, group = 0 } = {}) {
  const P = w.particles, id = P.alloc(), o = id * 3;
  P.pos[o] = x; P.pos[o + 1] = y; P.pos[o + 2] = z;
  P.prevPos[o] = x; P.prevPos[o + 1] = y; P.prevPos[o + 2] = z;
  P.vel[o] = vx; P.vel[o + 1] = vy; P.vel[o + 2] = vz;
  P.setMass(id, mass); P.radius[id] = radius; P.group[id] = group;
  return id;
}
function particlesFinite(w) {
  const P = w.particles;
  for (let i = 0; i < P.count; i++) {
    if (!P.alive[i]) continue;
    for (let k = 0; k < 3; k++) if (!Number.isFinite(P.pos[i * 3 + k]) || !Number.isFinite(P.vel[i * 3 + k])) return false;
  }
  return true;
}
function hashWorld(w) {
  const B = w.bodies, n = B.count, P = w.particles;
  let h = hashF64Array(B.pos, 2166136261, 0, n * 3);
  h = hashF64Array(B.quat, h, 0, n * 4);
  h = hashF64Array(B.vel, h, 0, n * 3);
  h = hashF64Array(B.angVel, h, 0, n * 3);
  for (let i = 0; i < n; i++) h = hashU32(h, B.flags[i] | (B.alive[i] << 8));
  if (P) { h = hashF64Array(P.pos, h, 0, P.count * 3); h = hashF64Array(P.vel, h, 0, P.count * 3); }
  return h;
}

// ================================================================================ 1. closestPointOnShape
{
  // sphere (r 1) at (1, 2, 3)
  const S = sphere(1), pose = [1, 2, 3, 0, 0, 0, 1];
  check('closest: sphere outside (point, normal, distance)', err(cp(S, pose, 4, 2, 3), [2, 2, 3, 1, 0, 0, 2]), 0, 1e-12);
  check('closest: sphere inside (signed distance < 0, outward normal)', err(cp(S, pose, 1, 2.5, 3), [1, 3, 3, 0, 1, 0, -0.5]), 0, 1e-12);
  check('closest: sphere diagonal outside', err(cp(S, pose, 1 + 2, 2 + 2, 3 + 1), [1 + 2 / 3, 2 + 2 / 3, 3 + 1 / 3, 2 / 3, 2 / 3, 1 / 3, 2]), 0, 1e-12);
  // capsule (r 0.5, half height 1), rotated 90 deg about z: axis along world x
  const C = capsule(0.5, 1), cpose = [0, 0, 0, ...qAxis(0, 0, 1, 90)];
  check('closest: capsule side (segment interior)', err(cp(C, I7, 2, 0.5, 0), [0.5, 0.5, 0, 1, 0, 0, 1.5]), 0, 1e-12);
  check('closest: capsule end cap', err(cp(C, I7, 0, 3, 0), [0, 1.5, 0, 0, 1, 0, 1.5]), 0, 1e-12);
  check('closest: capsule inside', err(cp(C, I7, 0.2, -0.4, 0), [0.5, -0.4, 0, 1, 0, 0, -0.3]), 0, 1e-12);
  check('closest: capsule rotated 90 deg about z (axis along x), end cap', err(cp(C, cpose, -3, 0, 0), [-1.5, 0, 0, -1, 0, 0, 1.5]), 0, 1e-12);
  // box (1, 2, 3): faces, edges, vertices, inside
  const Bx = box(1, 2, 3);
  check('closest: box face region', err(cp(Bx, I7, 2, 0.5, -1), [1, 0.5, -1, 1, 0, 0, 1]), 0, 1e-12);
  check('closest: box edge region', err(cp(Bx, I7, 2, 3, 0.5), [1, 2, 0.5, Math.SQRT1_2, Math.SQRT1_2, 0, Math.SQRT2]), 0, 1e-12);
  check('closest: box vertex region', err(cp(Bx, I7, 2, 3, 4), [1, 2, 3, 1 / Math.sqrt(3), 1 / Math.sqrt(3), 1 / Math.sqrt(3), Math.sqrt(3)]), 0, 1e-12);
  check('closest: box inside -> nearest face (y)', err(cp(Bx, I7, 0.1, -1.8, 0.5), [0.1, -2, 0.5, 0, -1, 0, -0.2]), 0, 1e-12);
  const q30 = qAxis(0, 1, 0, 30), c30 = Math.cos(Math.PI / 6), s30 = Math.sin(Math.PI / 6);
  // the box's local +x face normal rotated 30 deg about y is (cos 30, 0, -sin 30)
  check('closest: rotated box face (normal follows the rotation)', err(cp(Bx, [5, 0, 0, ...q30], 5 + 3 * c30, 0.5, -3 * s30), [5 + c30, 0.5, -s30, c30, 0, -s30, 2]), 0, 1e-12);
  // convex hull of the same box: every region matches the box closed form to 1e-9 (random points, all regions)
  const Cv = convex([-1, -2, -3, 1, -2, -3, -1, 2, -3, 1, 2, -3, -1, -2, 3, 1, -2, 3, -1, 2, 3, 1, 2, 3]);
  const rnd = mulberry32(5);
  let maxDiff = 0, inside = 0, faces = 0, edges = 0, verts = 0;
  const rpose = [0.3, -0.2, 1.1, ...qAxis(0.36, 0.48, 0.8, 37)], [qx, qy, qz, qw] = rpose.slice(3);
  for (let i = 0; i < 2000; i++) {
    // a local point (its region counted), placed in world space by the pose
    const lx = (rnd() * 2 - 1) * 2.5, ly = (rnd() * 2 - 1) * 3.5, lz = (rnd() * 2 - 1) * 4.5;
    const k = (Math.abs(lx) > 1) + (Math.abs(ly) > 2) + (Math.abs(lz) > 3);
    if (k === 0) inside++; else if (k === 1) faces++; else if (k === 2) edges++; else verts++;
    const tx = 2 * (qy * lz - qz * ly), ty = 2 * (qz * lx - qx * lz), tz = 2 * (qx * ly - qy * lx);
    const wx = rpose[0] + lx + qw * tx + (qy * tz - qz * ty), wy = rpose[1] + ly + qw * ty + (qz * tx - qx * tz), wz = rpose[2] + lz + qw * tz + (qx * ty - qy * tx);
    const a = cp(Bx, rpose, wx, wy, wz), b = cp(Cv, rpose, wx, wy, wz);
    maxDiff = Math.max(maxDiff, err(a, b));
  }
  check('closest: convex() box == box closed form, 2000 points, max abs diff', maxDiff, 0, 1e-9);
  note(`closest: convex-vs-box sample regions: ${inside} inside, ${faces} face, ${edges} edge, ${verts} vertex`);
  check('closest: the convex comparison covers every region (min count)', Math.min(inside, faces, edges, verts), 50, 2000);
  // a tetrahedron: vertex and edge regions by hand
  const T = convex([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]); // re-centred: local = input + offset
  const off = T.offset, tp = [-off[0], -off[1], -off[2], 0, 0, 0, 1];  // pose that puts the input frame back at the origin
  check('closest: tetrahedron vertex region (origin corner)', err(cp(T, tp, -1, -1, -1), [0, 0, 0, -1 / Math.sqrt(3), -1 / Math.sqrt(3), -1 / Math.sqrt(3), Math.sqrt(3)]), 0, 1e-12);
  check('closest: tetrahedron edge region (x axis edge)', err(cp(T, tp, 0.5, -1, -1), [0.5, 0, 0, 0, -Math.SQRT1_2, -Math.SQRT1_2, Math.SQRT2]), 0, 1e-12);
  check('closest: tetrahedron slanted face region', err(cp(T, tp, 1, 1, 1), [1 / 3, 1 / 3, 1 / 3, 1 / Math.sqrt(3), 1 / Math.sqrt(3), 1 / Math.sqrt(3), 2 / Math.sqrt(3)]), 0, 1e-12);
  // (0.1, 0.2, 0.3): 0.1 from x = 0, 0.2 from y = 0, 0.3 from z = 0, 0.23 from x + y + z = 1 -> out through x = 0
  check('closest: tetrahedron inside -> least-penetrating face', err(cp(T, tp, 0.1, 0.2, 0.3), [0, 0.2, 0.3, -1, 0, 0, -0.1]), 0, 1e-12);
  // trimesh: one triangle (normal +y: (b - a) x (c - a) with a (0,0,0), b (0,0,1), c (1,0,0)), then a BVH floor
  const tri = trimesh([0, 0, 0, 0, 0, 1, 1, 0, 0], [0, 1, 2]);
  check('closest: trimesh front (distance, face normal)', err(cp(tri, I7, 0.2, 1, 0.3), [0.2, 0, 0.3, 0, 1, 0, 1]), 0, 1e-12);
  check('closest: trimesh front, edge region', err(cp(tri, I7, -1, 1, 0.5), [0, 0, 0.5, -Math.SQRT1_2, Math.SQRT1_2, 0, Math.SQRT2]), 0, 1e-12);
  check('closest: trimesh back within thickness -> inside along +N', err(cp(tri, I7, 0.2, -0.05, 0.3), [0.2, 0, 0.3, 0, 1, 0, -0.05]), 0, 1e-12);
  check('closest: trimesh back beyond thickness -> no contact (Infinity)', cp(tri, I7, 0.2, -0.5, 0.3)[6] === Infinity, 1, 1);
  check('closest: trimesh back, larger thickness -> inside', err(cp(tri, I7, 0.2, -0.5, 0.3, 1), [0.2, 0, 0.3, 0, 1, 0, -0.5]), 0, 1e-12);
  // 20 x 20 grid floor with bumps vs brute force over every triangle
  const G = 20, pos = [], idx = [];
  for (let j = 0; j <= G; j++) for (let i = 0; i <= G; i++) pos.push(i - G / 2, 0.3 * Math.sin(i * 0.7) * Math.cos(j * 0.5), j - G / 2);
  for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) { const a = j * (G + 1) + i, b = a + 1, c = a + G + 1, d = c + 1; idx.push(a, c, b, b, c, d); }
  const floor = trimesh(pos, idx);
  let bvhDiff = 0;
  for (let k = 0; k < 300; k++) {
    const x = (rnd() * 2 - 1) * 9, y = rnd() * 2 - 0.05, z = (rnd() * 2 - 1) * 9;
    const r = cp(floor, I7, x, y, z, 0.1);
    let best = Infinity;
    for (let t = 0; t < idx.length; t += 3) {
      const one = trimesh(pos, [idx[t], idx[t + 1], idx[t + 2]]);
      const q = cp(one, I7, x, y, z, 0.1);
      if (q[6] < Infinity) { const d = Math.hypot(x - q[0], y - q[1], z - q[2]); if (d < best) best = d; }
    }
    bvhDiff = Math.max(bvhDiff, Math.abs(Math.hypot(x - r[0], y - r[1], z - r[2]) - best));
  }
  check('closest: trimesh BVH nearest == brute force (300 points), |d| diff', bvhDiff, 0, 1e-12, 'm');
  // compound: two spheres at x = +-2 and a box above; child selection
  const K = compound([{ shape: sphere(0.5), position: [-2, 0, 0] }, { shape: sphere(0.5), position: [2, 0, 0] }, { shape: box(0.5, 0.5, 0.5), position: [0, 3, 0] }]);
  check('closest: compound picks the nearer child (right sphere)', err(cp(K, I7, 3, 0, 0), [2.5, 0, 0, 1, 0, 0, 0.5]), 0, 1e-12);
  check('closest: compound picks the box child', err(cp(K, I7, 0, 4, 0.2), [0, 3.5, 0.2, 0, 1, 0, 0.5]), 0, 1e-12);
  check('closest: compound inside a child (signed distance < 0)', err(cp(K, I7, -2, 0.2, 0), [-2, 0.5, 0, 0, 1, 0, -0.3]), 0, 1e-12);
  check('closest: compound under a rotated pose', err(cp(K, [1, 0, 0, ...qAxis(0, 0, 1, 90)], 1, 3, 0), [1, 2.5, 0, 0, 1, 0, 0.5]), 0, 1e-12);
}

// ================================================================================ 2. volume samples
{
  const s = sampleShape(sphere(1), 8);
  const V = (4 / 3) * Math.PI;
  check('samples: sphere estimate vs 4/3 pi r^3 at samples 8 (relative)', Math.abs(s.estimate / V - 1), 0, 0.03);
  let sum = 0;
  for (let i = 0; i < s.count; i++) sum += s.weights[i];
  check('samples: sphere weights sum to the shape volume', Math.abs(sum / V - 1), 0, 1e-12);
  check('samples: sphere keeps 100-512 of the 512 grid points', s.count, 100, 512);
  const b = sampleShape(box(0.5, 0.25, 1), 8);
  check('samples: box estimate exact (all 512 cells, full coverage)', Math.abs(b.estimate / 1 - 1) + Math.abs(b.count - 512), 0, 1e-12);
  const c = sampleShape(capsule(0.3, 0.5), 8);
  check('samples: capsule estimate (relative)', Math.abs(c.estimate / c.volume - 1), 0, 0.03);
  const S1 = sphere(1), first = sampleShape(S1, 8);
  check('samples: cached per (shape, n): same set again, a new one for another n', sampleShape(S1, 8) === first && sampleShape(S1, 6) !== first && sampleShape(S1, 6).n === 6, 1, 1);
}

// ================================================================================ 3. rope on the flat ground
{
  const physics = new PhysicsWorld({ groundLevel: 0 });
  const w = new DynamicsWorld({ physics, maxParticles: 256 });
  const pc = w.addParticleSolver(new ParticleCollider(w));
  const rope = Rope.chain(w, { from: [-1.5, 0.6, 0.2], to: [1.5, 0.9, -0.2], segments: 31, mass: 0.5, radius: 0.03, pins: [], tethers: false });
  run(w, 3);
  const P = w.particles;
  let dev = 0;
  for (const id of rope.ids) dev = Math.max(dev, Math.abs(P.pos[id * 3 + 1] - 0.03));
  check('rope: 32 particles, no NaN after 3 s', particlesFinite(w), 1, 1);
  check('rope: every particle rests at y = radius (max |y - r|)', dev, 0, 1e-4, 'm');
  check('rope: one ground contact per particle', pc.stats.ground, 32, 32);
}

// ================================================================================ 4. cloth draped over a static box
{
  // a soft 3 x 3 m cloth (16 x 16, bend compliance 1) over a 1.6 x 1.6 x 0.4 m static box: the top 8 x 8 rest on the box,
  // the skirt hangs from its edges to the ground (a stiff cloth, bend 1e-5 on 4 g particles, spans the box like a plate)
  const physics = new PhysicsWorld({ groundLevel: 0 });
  physics.addStaticBox(-0.8, 0, 0.8, 0.4, -0.8, 0.8);
  const w = new DynamicsWorld({ physics, maxParticles: 512 });
  const pc = w.addParticleSolver(new ParticleCollider(w, { friction: 0.5 }));
  const cloth = Cloth.grid(w, { nx: 16, ny: 16, spacing: 0.2, position: [0, 0.8, 0], quaternion: qAxis(1, 0, 0, -90), mass: 1, bend: 1, tethers: false });
  run(w, 5);
  const P = w.particles, r = P.radius[cloth.ids[0]];
  let topDev = 0, nTop = 0, midMax = 0, rimDown = 0, rimN = 0, inBox = 0;
  for (let j = 0; j < 16; j++) {
    for (let i = 0; i < 16; i++) {
      const id = cloth.indexOf(i, j), x = P.pos[id * 3], y = P.pos[id * 3 + 1], z = P.pos[id * 3 + 2];
      if (i >= 5 && i <= 10 && j >= 5 && j <= 10) { topDev = Math.max(topDev, Math.abs(y - (0.4 + r))); nTop++; }
      const rim = i === 0 || i === 15 || j === 0 || j === 15;
      if (rim) { rimN++; if (y - r < 1e-3) rimDown++; }
      if (rim && ((i >= 6 && i <= 9) || (j >= 6 && j <= 9))) midMax = Math.max(midMax, y - r);
      if (Math.abs(x) < 0.8 && Math.abs(z) < 0.8 && y < 0.4 + r - 1e-3 && y > 0) inBox++;
    }
  }
  check('cloth: 16 x 16, no NaN after 5 s', particlesFinite(w), 1, 1);
  // (the ring 0.1 m from the box edges is lifted ~1.6 mm by the cloth's bending over the edge)
  check('cloth: the 36 particles over the box (>= 0.3 m from its edges) rest at top + radius', topDev + (nTop === 36 ? 0 : 1), 0, 1e-3, 'm');
  check('cloth: the middles of the four edges lie on the ground (max height above r)', midMax, 0, 1e-3, 'm');
  check('cloth: most of the rim lies on the ground (corners fold up)', rimDown / rimN, 0.6, 1);
  check('cloth: no particle inside the box', inBox, 0, 0);
  check('cloth: contacts with the static box this step', pc.stats.statics, 64, 512);
  check('cloth: contacts with the ground this step', pc.stats.ground, 16, 512);
}

// ================================================================================ 5. two-way: a rope swung into a crate; head-on momentum
{
  const physics = new PhysicsWorld({ groundLevel: 0 });
  const w = new DynamicsWorld({ physics, maxParticles: 128 });
  w.addParticleSolver(new ParticleCollider(w));
  const crate = w.addBody({ shape: box(0.25, 0.25, 0.25), position: [-0.25, 0.25, 0], mass: 2, friction: 0.3 });
  run(w, 1); // the crate settles (and falls asleep)
  const asleep = !w.isAwake(crate);
  const x0 = w.bodies.pos[crate * 3];
  // pinned at (-0.7, 1.25), released horizontal to the left: the swing's lower half sweeps through the crate
  Rope.chain(w, { from: [-0.7, 1.25, 0], to: [-1.9, 1.25, 0], segments: 15, mass: 2, radius: 0.04, pins: ['from'] });
  let xMax = x0;
  run(w, 2, () => { xMax = Math.max(xMax, w.bodies.pos[crate * 3]); });
  check('two-way: the crate was asleep before the swing', asleep, 1, 1);
  check('two-way: the swinging rope pushes the 2 kg crate (+x travel)', xMax - x0, 0.05, 5, 'm');
  check('two-way: no NaN', particlesFinite(w), 1, 1);
}
{
  // zero gravity, no damping: a 1 kg sphere at 3 m/s hits a 0.5 kg particle head-on
  const w = new DynamicsWorld({ maxParticles: 16, gravity: [0, 0, 0], particleDamping: 0, sleep: false });
  const pc = w.addParticleSolver(new ParticleCollider(w));
  const s = w.addBody({ shape: sphere(0.3), position: [-1, 0, 0], mass: 1, linearDamping: 0, angularDamping: 0 });
  w.setVelocity(s, 3, 0, 0);
  const p = addParticle(w, 0, 0, 0, { mass: 0.5, radius: 0.05 });
  const mom = () => w.bodies.mass[s] * w.bodies.vel[s * 3] + w.particles.mass[p] * w.particles.vel[p * 3];
  let worst = 0, first = -1, steps = 0;
  for (let k = 0; k < 120; k++) {
    const before = mom(), vp = w.particles.vel[p * 3];
    w.step(DT); steps++;
    const after = mom();
    if (first < 0 && w.particles.vel[p * 3] !== vp) { first = k; check('two-way: momentum over the first contact step (relative change)', Math.abs(after - before) / Math.abs(before), 0, 0.05); }
    worst = Math.max(worst, Math.abs(after - 3) / 3);
  }
  check('two-way: the hit happened (first contact step index)', first, 10, 60);
  check('two-way: total momentum drift over 1 s (relative)', worst, 0, 1e-9);
  check('two-way: zero restitution: sphere and particle share 2 m/s after the hit', Math.abs(w.bodies.vel[s * 3] - 2) + Math.abs(w.particles.vel[p * 3] - 2), 0, 1e-6, 'm/s');
  check('two-way: one body contact', pc.stats.bodies, 1, 1);
}

// ================================================================================ 6. friction on a 30 deg slope
const SIN30 = 0.5, COS30 = Math.sqrt(3) / 2;
function slopeRun(mu, heightfield, wo = {}) {
  const physics = heightfield ? new PhysicsWorld({ heightfield: (x) => 3 - x * (SIN30 / COS30) }) : null;
  const w = new DynamicsWorld({ physics, maxParticles: 64, ...wo });
  const pc = w.addParticleSolver(new ParticleCollider(w, { friction: mu }));
  // surface normal n = (sin 30, cos 30, 0), down-slope t = (cos 30, -sin 30, 0)
  let base = [0, 0, 0];
  if (!heightfield) {
    w.addBody({ type: 'static', shape: box(6, 0.2, 2), position: [0, 0, 0], quaternion: qAxis(0, 0, 1, -30) });
    base = [0.2 * SIN30, 0.2 * COS30, 0];
  } else base = [0, 3, 0];
  const r = 0.05, ids = [];
  for (let k = 0; k < 8; k++) {
    const s = -3 + (k % 4) * 0.2, z = -0.6 + (k >> 2) * 1.2;
    ids.push(addParticle(w, base[0] + r * SIN30 + s * COS30, base[1] + r * COS30 - s * SIN30, z, { mass: 0.1, radius: r, vx: COS30, vy: -SIN30 }));
  }
  const P = w.particles;
  const surfDev = () => { // distance of the centres from the slope plane minus r
    let m = 0;
    for (const id of ids) { const o = id * 3; m = Math.max(m, Math.abs((P.pos[o] - base[0]) * SIN30 + (P.pos[o + 1] - base[1]) * COS30 - r)); }
    return m;
  };
  const speed = () => { let m = 0; for (const id of ids) { const o = id * 3; m = Math.max(m, Math.hypot(P.vel[o], P.vel[o + 1], P.vel[o + 2])); } return m; };
  const x0 = P.pos[ids[0] * 3];
  run(w, 1);
  const v1 = speed(), dev = surfDev();  // measured at 1 s: at mu 0.1 they leave the slope later on
  const x1 = P.pos[ids[0] * 3];
  run(w, 1);
  return { v1, v2: speed(), dx1: x1 - x0, dx2: P.pos[ids[0] * 3] - x1, dev, finite: particlesFinite(w), pc };
}
{
  const stop = slopeRun(0.8, false), slide = slopeRun(0.1, false);
  check('slope (static body): mu 0.8 stops the particles (max speed at 2 s)', stop.v2, 0, 1e-3, 'm/s');
  check('slope (static body): mu 0.8, no creep in the second second', Math.abs(stop.dx2), 0, 1e-4, 'm');
  check('slope (static body): mu 0.8, stopping distance ~ v^2 / 2 g (mu cos - sin) = 0.264 m', stop.dx1 / COS30, 0.2, 0.33, 'm');
  check('slope (static body): mu 0.1 slides, speed at 1 s ~ 1 + 4.05 m/s', slide.v1, 4.5, 5.5, 'm/s');
  check('slope (static body): particles stay on the surface at 1 s (max |dist - r|)', Math.max(stop.dev, slide.dev), 0, 1e-3, 'm');
  check('slope (static body): no NaN', stop.finite && slide.finite, 1, 1);
  const hf = slopeRun(0.8, true), hfs = slopeRun(0.1, true);
  check('slope (heightfield): mu 0.8 stops the particles', hf.v2, 0, 1e-3, 'm/s');
  check('slope (heightfield): mu 0.1 slides', hfs.v1, 4.5, 5.5, 'm/s');
  check('slope (heightfield): particles on the ground plane (max |dist - r|)', Math.max(hf.dev, hfs.dev), 0, 1e-3, 'm');
  // the velocity pass runs once per substep: 2 velocity iterations must not apply the friction budget twice
  const slide2 = slopeRun(0.1, false, { velocityIterations: 2 }), stop2 = slopeRun(0.8, false, { velocityIterations: 2 });
  check('slope: velocityIterations 2 slides like 1 (speed at 1 s, difference)', Math.abs(slide2.v1 - slide.v1), 0, 1e-9, 'm/s');
  check('slope: velocityIterations 2, mu 0.8 still stops', stop2.v2, 0, 1e-3, 'm/s');
}

// ================================================================================ 6b. trimesh floor, kinematic platform, mask
{
  // a static trimesh floor (two triangles, normal +y) at y = 0.5: particles rest on it at 0.5 + r
  const w = new DynamicsWorld({ maxParticles: 64 });
  w.addParticleSolver(new ParticleCollider(w));
  w.addBody({ type: 'static', shape: trimesh([-2, 0, -2, -2, 0, 2, 2, 0, 2, 2, 0, -2], [0, 1, 2, 0, 2, 3]), position: [0, 0.5, 0] });
  const ids = [];
  for (let k = 0; k < 9; k++) ids.push(addParticle(w, (k % 3) - 1 + 0.13, 1 + 0.1 * k, ((k / 3) | 0) - 1 + 0.07, { radius: 0.04 }));
  run(w, 2);
  let dev = 0;
  for (const id of ids) dev = Math.max(dev, Math.abs(w.particles.pos[id * 3 + 1] - 0.54));
  check('trimesh: particles rest on a static trimesh floor at y + r (max dev)', dev, 0, 1e-4, 'm');
}
{
  // a kinematic platform sliding at 0.5 m/s carries particles resting on it (static friction on a moving body)
  const physics = new PhysicsWorld({ groundLevel: -5 });
  const w = new DynamicsWorld({ physics, maxParticles: 64 });
  w.addParticleSolver(new ParticleCollider(w, { friction: 0.8 }));
  const plat = w.addBody({ type: 'kinematic', shape: box(1.5, 0.1, 1.5), position: [0, 0, 0] });
  const ids = [];
  for (let k = 0; k < 4; k++) ids.push(addParticle(w, -0.5 + k * 0.3, 0.15, 0, { radius: 0.05 }));
  run(w, 0.5);
  let px = 0, x0 = 0, p0 = 0;
  // 0.5 s to pick up speed (the platform jumps to 0.5 m/s: the particles slip ~v^2 / 2 mu g = 1.6 cm), then 1.5 s
  run(w, 2, (i) => {
    if (i === 60) { x0 = w.particles.pos[ids[0] * 3]; p0 = px; }
    px = 0.5 * (i + 1) * DT; w.setKinematicTarget(plat, px, 0, 0, 0, 0, 0, 1);
  });
  const carried = w.particles.pos[ids[0] * 3] - x0;
  check('kinematic: particles ride a moving platform (carried / platform travel, last 1.5 s)', carried / (px - p0), 0.999, 1.001);
  check('kinematic: ... and stay on it (max |y - top - r|)', Math.max(...ids.map((id) => Math.abs(w.particles.pos[id * 3 + 1] - 0.15))), 0, 1e-3, 'm');
}
{
  // mask: a collider whose mask excludes layer 2 lets particles fall through bodies on layer 2
  const w = new DynamicsWorld({ maxParticles: 16 });
  w.addParticleSolver(new ParticleCollider(w, { mask: ~2 }));
  w.addBody({ type: 'static', shape: box(1, 0.1, 1), position: [0, 0, 0], layer: 2 });
  w.addBody({ type: 'static', shape: box(1, 0.1, 1), position: [3, 0, 0], layer: 1 });
  const a = addParticle(w, 0, 0.5, 0, { radius: 0.05 }), b = addParticle(w, 3, 0.5, 0, { radius: 0.05 });
  run(w, 1);
  check('mask: the layer-2 body is ignored, the layer-1 body is not', w.particles.pos[a * 3 + 1] < -1 && Math.abs(w.particles.pos[b * 3 + 1] - 0.15) < 1e-4, 1, 1);
}

{
  // trimesh back side: particles 0.03 / 0.06 / 0.09 m behind a floor (thickness 0.1) are pushed out to the front
  const w = new DynamicsWorld({ maxParticles: 16 });
  w.addParticleSolver(new ParticleCollider(w));
  w.addBody({ type: 'static', shape: trimesh([-2, 0, -2, -2, 0, 2, 2, 0, 2, 2, 0, -2], [0, 1, 2, 0, 2, 3]), position: [0, 0.5, 0] });
  const depths = [0.03, 0.065, 0.09], ids = depths.map((d, k) => addParticle(w, k * 0.5 - 0.5, 0.5 - d, 0.2, { radius: 0.04 }));
  run(w, 0.5);
  depths.forEach((d, k) => check(`trimesh: a particle ${d} m behind the floor is pushed out on top`, Math.abs(w.particles.pos[ids[k] * 3 + 1] - 0.54), 0, 1e-4, 'm'));
}
{
  // a static compound (two boxes of different heights): particles rest on each child's top
  const w = new DynamicsWorld({ maxParticles: 16 });
  w.addParticleSolver(new ParticleCollider(w));
  w.addBody({ type: 'static', shape: compound([{ shape: box(0.5, 0.1, 0.5), position: [-1, 0, 0] }, { shape: box(0.5, 0.3, 0.5), position: [1, 0, 0] }]), position: [0, 0, 0] });
  const a = addParticle(w, -1.1, 0.6, 0.1, { radius: 0.05 }), b = addParticle(w, 0.9, 0.8, -0.1, { radius: 0.05 });
  run(w, 1);
  check('compound: particles rest on each child (max dev from top + r)', Math.max(Math.abs(w.particles.pos[a * 3 + 1] - 0.15), Math.abs(w.particles.pos[b * 3 + 1] - 0.35)), 0, 1e-4, 'm');
}
{
  // maxContacts caps the contacts per step; the rest is counted in stats.capped
  const w = new DynamicsWorld({ physics: new PhysicsWorld({ groundLevel: 0 }), maxParticles: 64 });
  const pc = w.addParticleSolver(new ParticleCollider(w, { maxContacts: 10 }));
  for (let k = 0; k < 32; k++) addParticle(w, (k % 8) * 0.2, 0.05, (k >> 3) * 0.2, { radius: 0.05 });
  w.step(DT);
  check('capping: maxContacts 10 with 32 particles on the ground: contacts', pc.stats.contacts, 10, 10);
  check('capping: ... stats.capped', pc.stats.capped, 22, 22);
}
{
  // a crate resting on a pinned cloth never sleeps (particles hold it up), so it falls when the cloth is unpinned
  const w = new DynamicsWorld({ physics: new PhysicsWorld({ groundLevel: 0 }), maxParticles: 256, particleDamping: 3 });
  w.addParticleSolver(new ParticleCollider(w, { friction: 0.6 }));
  const cloth = Cloth.grid(w, { nx: 12, ny: 12, spacing: 0.12, position: [0, 1.5, 0], quaternion: qAxis(1, 0, 0, -90), mass: 1, radius: 0.04, pins: ['left', 'right'] });
  const crate = w.addBody({ shape: box(0.15, 0.15, 0.15), position: [0, 1.75, 0], mass: 2 });
  run(w, 10);
  const B = w.bodies, o = crate * 3;
  const still = Math.hypot(B.vel[o], B.vel[o + 1], B.vel[o + 2]) < 0.05 && Math.hypot(B.angVel[o], B.angVel[o + 1], B.angVel[o + 2]) < 0.05;
  check('carried: the crate rests on the cloth (slow enough to sleep) but stays awake', still && w.isAwake(crate) && B.sleepTime[crate] === 0, 1, 1);
  const y0 = B.pos[o + 1];
  for (const id of cloth.ids) if (w.particles.flags[id] & PFLAG.PINNED) w.particles.setPinned(id, false);
  run(w, 1);
  check('carried: unpinned, the cloth and the crate fall (drop in 1 s)', y0 - B.pos[o + 1], 0.5, 2, 'm');
}
{
  // static-box contact order is a pure function of the box set: a particle on two boxes after serialize / from.
  // Grid cells of 2 m; box B (added first) spans cells 0..2, A spans -2..0, so cell 0 lists [B, A]: a query circle
  // reaching x < 0 visits A first, one that does not visits B first. The original world keeps the answer it cached
  // while a second particle (freed later) stretched the group over x < 0; the restored world asks afresh.
  const physics = new PhysicsWorld({ groundLevel: -10, cellSize: 2 });
  physics.addStaticBox(1.5, 0, 5, 1, -1, 1);
  physics.addStaticBox(-3, 0, 1.5, 1, -1, 1);
  const mk = () => new ParticleCollider(null, { friction: 0.3 });
  const w = new DynamicsWorld({ physics, maxParticles: 8 });
  const pc = w.addParticleSolver(mk());
  const p1 = addParticle(w, -0.5, 1.1, 0, { radius: 0.1 }), p2 = addParticle(w, 1.42, 1.1, 0, { radius: 0.1, vx: 0.3 });
  w.step(DT);
  w.particles.free(p1);
  for (let k = 0; k < 10; k++) w.step(DT);
  const raw1 = physics.queryStatic(pc._sqx[0], pc._sqz[0], pc._sqr[0], []), raw2 = physics.queryStatic(w.particles.pos[p2 * 3], 0, 0.2 + 1, []);
  check('order: precondition: the cached and a fresh grid answer list the boxes in different orders', raw1.length === 2 && raw2.length === 2 && raw1[0] !== raw2[0], 1, 1);
  const C = DynamicsWorld.from(w.serialize(), { physics, solvers: [mk()] });
  const pc2 = C.particleSolvers[1];
  w.step(DT); C.step(DT);
  let same = pc.count === pc2.count && pc.stats.statics === 2;
  for (let k = 0; same && k < pc.count; k++) {
    same = pc.particle[k] === pc2.particle[k] && pc.kind[k] === pc2.kind[k];
    for (let j = 0; j < 3; j++) same = same && pc._cl[k * 3 + j] === pc2._cl[k * 3 + j] && pc._cn[k * 3 + j] === pc2._cn[k * 3 + j];
  }
  check('order: the restored world sees the same two static contacts in the same order', same, 1, 1);
  for (let k = 0; k < 60; k++) { w.step(DT); C.step(DT); }
  check('order: particle on two boxes: serialize / from continues bit-identically', hashWorld(w) === hashWorld(C) ? 1 : 0, 1, 1);
}

// ================================================================================ 7. buoyancy
function fluidWorld(level = 0, o = {}) {
  const w = new DynamicsWorld({ maxParticles: 256, ...o.world });
  const buoy = w.addParticleSolver(new Buoyancy(w, { fluid: { heightAt: level, density: 1000, drag: 1, angularDrag: 0.5, ...o.fluid }, ...o.buoy }));
  return { w, buoy };
}
{
  // density 500 cube (1 m): floats half submerged (sleep off: the exact equilibrium)
  const { w, buoy } = fluidWorld(0, { world: { sleep: false } });
  const id = w.addBody({ shape: box(0.5, 0.5, 0.5), position: [0, 0.3, 0], density: 500 });
  run(w, 15);
  check('buoyancy: density 500 cube floats half submerged (centre vs level, sleep off)', Math.abs(w.bodies.pos[id * 3 + 1]), 0, 1e-3, 'm');
  check('buoyancy: submerged volume = 0.5 m^3 (relative)', Math.abs(buoy.submergedVolume(id) / 0.5 - 1), 0, 2e-3);
  check('buoyancy: it stays upright (|quat.xyz|)', Math.hypot(...w.bodies.quat.slice(id * 4, id * 4 + 3)), 0, 1e-3);
}
{
  // with sleeping on, it falls asleep within the wake band: |offset| <= WAKE_FRACTION m / (density A) = 1 cm
  const { w } = fluidWorld(0);
  const id = w.addBody({ shape: box(0.5, 0.5, 0.5), position: [0, 0.3, 0], density: 500 });
  run(w, 15);
  check('buoyancy: (sleep on) asleep after 15 s', !w.isAwake(id), 1, 1);
  check('buoyancy: (sleep on) its offset stays inside the wake band', Math.abs(w.bodies.pos[id * 3 + 1]), 0, 0.01, 'm');
}
{
  // neutral: density 1000, fully submerged, at rest (sleep off)
  const { w } = fluidWorld(0, { world: { sleep: false } });
  const id = w.addBody({ shape: box(0.4, 0.3, 0.5), position: [0, -3, 0], density: 1000 });
  const sp = w.addBody({ shape: sphere(0.4), position: [3, -2, 0], density: 1000 });
  run(w, 5);
  check('buoyancy: density 1000 box is neutral (drift over 5 s, sleep off)', Math.abs(w.bodies.pos[id * 3 + 1] + 3), 0, 1e-6, 'm');
  check('buoyancy: density 1000 sphere is neutral (drift over 5 s, sleep off)', Math.abs(w.bodies.pos[sp * 3 + 1] + 2), 0, 1e-6, 'm');
  const ng = w.addBody({ shape: box(0.3, 0.3, 0.3), position: [-3, -2, 0], density: 500, noGravity: true });
  run(w, 1);
  check('buoyancy: a NO_GRAVITY body gets no buoyancy (it does not rise)', Math.abs(w.bodies.pos[ng * 3 + 1] + 2), 0, 1e-12, 'm');
}
{
  // a flat floater tilted 10 deg about z: the buoyancy torque turns it back (metacentre above the centre of mass)
  const { w } = fluidWorld(0, { world: { sleep: false } });
  const id = w.addBody({ shape: box(1, 0.2, 1), position: [0, 0, 0], quaternion: qAxis(0, 0, 1, 10), density: 500 });
  run(w, 0.1);
  check('buoyancy: a tilted flat floater gets a restoring spin (-z)', -w.bodies.angVel[id * 3 + 2], 0.05, 10, 'rad/s');
}
{
  // a sleeping floater is re-sampled only when the surface at its centre moves: 1 heightAt call per step while asleep,
  // then the level jumps by 0.3 m at t = 4 s and it wakes and rises
  let calls = 0;
  const heightAt = (x, z, t) => { calls++; return t < 4 ? 0 : 0.3; };
  const { w } = fluidWorld(0, { fluid: { heightAt } });
  const id = w.addBody({ shape: box(0.5, 0.5, 0.5), position: [0, 0.01, 0], density: 500 });
  run(w, 3);
  const asleep = !w.isAwake(id);
  calls = 0;
  run(w, 0.5);
  check('buoyancy: a sleeping floater costs one heightAt call per step (calls / step)', calls / 60, 0, 1);
  check('buoyancy: ... it was asleep', asleep, 1, 1);
  run(w, 6.5);
  check('buoyancy: the level rises 0.3 m at t = 4 s: it wakes and rises with it (t = 10 s)', w.bodies.pos[id * 3 + 1], 0.27, 0.33, 'm');
}
{
  const { w } = fluidWorld(0);
  const id = w.addBody({ shape: box(0.3, 0.3, 0.3), position: [0, -1, 0], density: 2000 });
  run(w, 2);
  check('buoyancy: density 2000 sinks (depth after 2 s)', -1 - w.bodies.pos[id * 3 + 1], 1, 30, 'm');
  check('buoyancy: ... at under the drag-limited terminal speed (9.81 m/s)', -w.bodies.vel[id * 3 + 1], 1, 9.81, 'm/s');
}
{
  // drag damps a bobbing cube: successive peaks decrease
  const { w } = fluidWorld(0);
  const id = w.addBody({ shape: box(0.5, 0.5, 0.5), position: [0, 0.3, 0], density: 500 });
  const ys = [];
  run(w, 6, () => ys.push(w.bodies.pos[id * 3 + 1]));
  const peaks = [];
  for (let i = 1; i + 1 < ys.length; i++) if (ys[i] > ys[i - 1] && ys[i] >= ys[i + 1]) peaks.push(ys[i]);
  let mono = 1;
  for (let i = 1; i < Math.min(peaks.length, 4); i++) if (!(peaks[i] < peaks[i - 1])) mono = 0;
  check('buoyancy: bobbing cube shows >= 3 cycles (peaks in 6 s)', peaks.length, 3, 12);
  check('buoyancy: drag: peak heights decrease monotonically over 3 cycles', mono, 1, 1);
  check('buoyancy: drag: the 3rd peak is below half the first', peaks[2] / peaks[0], 0, 0.5);
  note(`buoyancy: bobbing peaks ${peaks.slice(0, 5).map((p) => p.toFixed(4)).join(', ')} m`);
}
{
  // a wave surface tilts a long box toward its slope; the time argument moves the wave
  const wave = (x, z, t) => 0.3 * Math.sin(0.5 * x - t);
  const { w, buoy } = fluidWorld(0, { fluid: { heightAt: wave } });
  const id = w.addBody({ shape: box(2, 0.25, 0.5), position: [0, 0, 0], density: 500 });
  run(w, 0.1);
  check('buoyancy: wave (surface rising toward +x) spins the long box +z', w.bodies.angVel[id * 3 + 2], 0.01, 10, 'rad/s');
  check('buoyancy: the fluid clock advances with the steps', Math.abs(buoy.fluid.time - 12 * DT), 0, 1e-12, 's');
  const { w: w2 } = fluidWorld(0, { fluid: { heightAt: (x, z, t) => -wave(x, z, t) } });
  const id2 = w2.addBody({ shape: box(2, 0.25, 0.5), position: [0, 0, 0], density: 500 });
  run(w2, 0.1);
  check('buoyancy: the mirrored wave spins it -z', -w2.bodies.angVel[id2 * 3 + 2], 0.01, 10, 'rad/s');
}
{
  // particles of density 500 float half submerged; pinned ones are ignored
  const { w, buoy } = fluidWorld(0.5);
  const r = 0.05, m = 500 * (4 / 3) * Math.PI * r * r * r, ids = [];
  for (let k = 0; k < 64; k++) ids.push(addParticle(w, (k % 8) * 0.3, 0.8 + (k >> 3) * 0.05, ((k >> 3) % 8) * 0.3, { mass: m, radius: r }));
  const pin = addParticle(w, 5, 0.2, 5, { mass: m, radius: r });
  w.particles.setPinned(pin, true);
  run(w, 10);
  let fmean = 0, ydev = 0;
  for (const id of ids) { fmean += buoy.particleFraction(id) / ids.length; ydev = Math.max(ydev, Math.abs(w.particles.pos[id * 3 + 1] - 0.5)); }
  check('buoyancy: particles of density 500 float at fraction 0.5 (mean)', fmean, 0.48, 0.52);
  check('buoyancy: ... centres at the surface (max |y - level|)', ydev, 0, 0.005, 'm');
  check('buoyancy: a pinned particle stays put', Math.abs(w.particles.pos[pin * 3 + 1] - 0.2), 0, 0, 'm');
  check('buoyancy: stats.particles counts the floating ones', buoy.stats.particles, 64, 64);
}
{
  // velocityIterations 2: particle buoyancy still applied once per substep (twice would float them at fraction ~0.24)
  const { w, buoy } = fluidWorld(0.5, { world: { velocityIterations: 2 } });
  const r = 0.05, m = 500 * (4 / 3) * Math.PI * r * r * r, ids = [];
  for (let k = 0; k < 16; k++) ids.push(addParticle(w, (k % 4) * 0.3, 0.8, (k >> 2) * 0.3, { mass: m, radius: r }));
  run(w, 10);
  let fmean = 0;
  for (const id of ids) fmean += buoy.particleFraction(id) / ids.length;
  check('buoyancy: velocityIterations 2, particles still float at fraction 0.5', fmean, 0.48, 0.52);
  // a world that does not pass the iteration index: the second call in a substep is a no-op
  const P = w.particles, id = ids[0], H = DT / 4;
  P.pos[id * 3 + 1] = 0.45; P.vel.fill(0, id * 3, id * 3 + 3);
  for (const s of [buoy]) s.solvePositions(w, H, 0);
  buoy.solveVelocities(w, H);
  const v1 = P.vel[id * 3 + 1];
  buoy.solveVelocities(w, H);
  check('buoyancy: without the iteration index, one velocity pass per substep', v1 > 0 && P.vel[id * 3 + 1] === v1, 1, 1);
}
{
  // registration: bodies: false floats only registered bodies; unregister excludes
  const { w, buoy } = fluidWorld(0, { buoy: { bodies: false } });
  const a = w.addBody({ shape: box(0.5, 0.5, 0.5), position: [0, 0, 0], density: 500 });
  const b = w.addBody({ shape: box(0.5, 0.5, 0.5), position: [3, 0, 0], density: 500 });
  buoy.register(a);
  run(w, 1);
  check('buoyancy: { bodies: false } floats only registered bodies', (Math.abs(w.bodies.pos[a * 3 + 1]) < 0.2) && w.bodies.pos[b * 3 + 1] < -2, 1, 1);
  const sleptBefore = !w.isAwake(a);
  buoy.unregister(a);
  w.step(DT);
  check('buoyancy: unregister(id) wakes the body (it was asleep)', sleptBefore && w.isAwake(a), 1, 1);
  run(w, 1);
  check('buoyancy: unregister(id) stops it', w.bodies.pos[a * 3 + 1] < -1, 1, 1);
}

{
  // Buoyancy serialize -> JSON -> restore: registration, wake state, volumes and time come back; a changed `samples`
  // rebuilds the sample sets
  const { w, buoy } = fluidWorld(0, { buoy: { bodies: false } });
  const a = w.addBody({ shape: box(0.5, 0.5, 0.5), position: [0, 0.2, 0], density: 500 });
  const b = w.addBody({ shape: sphere(0.3), position: [3, 0, 0], density: 600 });
  buoy.register(a); buoy.register(b); buoy.unregister(b);
  run(w, 0.5);
  const d = JSON.parse(JSON.stringify(buoy.serialize()));
  const r = new Buoyancy(null, { fluid: { heightAt: 0 } });
  r.restore(d);
  const same = (x, y) => x.length >= 2 && Array.from(x.slice(0, 2)).every((v, i) => v === y[i]);
  check('restore (JSON): registration, wake reference volumes and centre heights', same(r._reg, buoy._reg) && same(r._vref, buoy._vref) && same(r._hc, buoy._hc), 1, 1);
  check('restore (JSON): submergedVolume() is carried (not 0 until the next step)', r.submergedVolume(a) === buoy.submergedVolume(a) && r.submergedVolume(a) > 0.3, 1, 1);
  check('restore (JSON): fluid time and options', r.fluid.time === buoy.fluid.time && r.bodies === false && r.samples === 8, 1, 1);
  buoy.samples = 6;
  w.step(DT);
  check('samples: a changed `samples` rebuilds the body\'s sample set', buoy._set[a].n, 6, 6);
}

// ================================================================================ 8. determinism: two runs, serialize / from
const WAVE = (x, z, t) => 0.4 + 0.15 * Math.sin(0.7 * x + 0.3 * z - 1.3 * t);
const PC_OPTS = { friction: 0.5, margin: 0.02 };
const BUOY_OPTS = { fluid: { heightAt: WAVE, density: 1000, drag: 1, angularDrag: 0.5 } };
function buildMixed() {
  const physics = new PhysicsWorld({ groundLevel: 0 });
  physics.addStaticBox(2, 0, 3, 0.8, -1, 1);
  const w = new DynamicsWorld({ physics, maxParticles: 1024 });
  w.addParticleSolver(new ParticleCollider(w, PC_OPTS));
  w.addParticleSolver(new Buoyancy(w, BUOY_OPTS));
  const rnd = mulberry32(3);
  for (let i = 0; i < 12; i++) {
    const shape = i % 3 === 0 ? sphere(0.2) : i % 3 === 1 ? box(0.25, 0.15, 0.2) : capsule(0.12, 0.2);
    w.addBody({ shape, position: [(rnd() - 0.5) * 3, 0.6 + i * 0.4, (rnd() - 0.5) * 3], density: 300 + rnd() * 1200 });
  }
  Cloth.grid(w, { nx: 12, ny: 12, spacing: 0.15, position: [0.5, 2.5, 0], quaternion: qAxis(1, 0, 0, -90), mass: 6, bend: 0.1, radius: 0.03, tethers: false });
  Rope.chain(w, { from: [2.5, 2.5, 0], to: [1, 2.5, 0], segments: 20, mass: 0.4, radius: 0.04, pins: ['from'] });
  return { w, physics };
}
{
  const A = buildMixed(), Bm = buildMixed();
  for (let i = 0; i < 240; i++) { A.w.step(DT); Bm.w.step(DT); }
  const hA = hashWorld(A.w), hB = hashWorld(Bm.w);
  check('determinism: mixed scene hash identical across two runs (2 s)', hA === hB ? 1 : 0, 1, 1);
  check('determinism: no NaN', particlesFinite(A.w), 1, 1);
  const snap = A.w.serialize();
  const C = DynamicsWorld.from(snap, { physics: Bm.physics, solvers: [new ParticleCollider(null, PC_OPTS), new Buoyancy(null, BUOY_OPTS)] });
  check('determinism: from() restores both solvers by name (3 particle solvers)', C.particleSolvers.length, 3, 3);
  for (let i = 0; i < 120; i++) { A.w.step(DT); C.step(DT); }
  const hA2 = hashWorld(A.w), hC = hashWorld(C);
  check('determinism: serialize / from continues bit-identically (1 s more)', hA2 === hC ? 1 : 0, 1, 1);
  check('determinism: the fluid clock was carried', C.particleSolvers[2].fluid.time === A.w.particleSolvers[2].fluid.time ? 1 : 0, 1, 1);
  note(`determinism: hash after 2 s 0x${hA.toString(16)}, after 3 s 0x${hA2.toString(16)} / restored 0x${hC.toString(16)}`);
}

// ================================================================================ 9. allocation + timing
// 1024 cloth particles + 50 bodies in a flat fluid (both solvers), after a warm-up; the hooks alone on the live state,
// then the whole world.step. Per step the hooks box the six bounds handed to broadphase.query per particle group
// (~100 B, short-lived); nothing is retained and no GC runs.
function allocScene(fluid) {
  const physics = new PhysicsWorld({ groundLevel: 0 });
  physics.addStaticBox(-6, 0, -5, 1, -6, 6); physics.addStaticBox(5, 0, 6, 1, -6, 6);
  const w = new DynamicsWorld({ physics, maxParticles: 1100, sleep: false });
  const pc = w.addParticleSolver(new ParticleCollider(w));
  const buoy = fluid ? w.addParticleSolver(new Buoyancy(w, { fluid: { heightAt: 0.35 } })) : null;
  const rnd = mulberry32(17);
  for (let i = 0; i < 50; i++) { // one layer on the ground, 5 x 10, boxes and spheres
    const shape = i % 2 ? box(0.3, 0.3, 0.3) : sphere(0.3);
    w.addBody({ shape, position: [((i % 10) - 4.5) * 0.8 + (rnd() - 0.5) * 0.1, 0.31, (((i / 10) | 0) - 2) * 0.8], density: 400 });
  }
  // 30 kg of cloth: 29 g particles of radius 0.05 m (density 56 kg/m^3: in the fluid it floats high)
  const cloth = Cloth.grid(w, { nx: 32, ny: 32, spacing: 0.25, position: [0, 2, 0], quaternion: qAxis(1, 0, 0, -90), mass: 30, radius: 0.05, bend: 0.1, tethers: false });
  return { w, pc, buoy, cloth };
}
{
  let gc = globalThis.gc;
  if (!gc) { try { v8.setFlagsFromString('--expose-gc'); gc = vm.runInNewContext('gc'); } catch { gc = null; } }
  const { w, pc, buoy, cloth } = allocScene(true);
  check('alloc: scene has 1024 particles and 50 bodies', cloth.count + w.bodies.size * 1000, 51024, 51024);
  for (let k = 0; k < 300; k++) w.step(DT); // warm-up: the cloth lands on the bodies and the water
  const grow = (fn) => {
    if (gc) gc();
    const prof = new v8.GCProfiler();
    prof.start();
    const h0 = process.memoryUsage().heapUsed;
    fn();
    const h1 = process.memoryUsage().heapUsed;
    const gcs = prof.stop().statistics.length;
    if (gc) gc();
    return { bytes: h1 - h0, gcs, retained: process.memoryUsage().heapUsed - h0 };
  };
  const H = DT / 4;
  const hooks = () => {
    for (let k = 0; k < 300; k++) {
      buoy.beginStep(w, DT); pc.beginStep(w, DT);
      for (let s = 0; s < 4; s++) { pc.solvePositions(w, H, 0); buoy.solvePositions(w, H, 0); buoy.solveVelocities(w, H, 0); pc.solveVelocities(w, H, 0); }
      pc.endStep(w, DT); buoy.endStep(w, DT);
    }
  };
  hooks(); hooks();
  let best = null;
  for (let r = 0; r < 3; r++) { const g = grow(hooks); if (best === null || g.gcs < best.gcs || (g.gcs === best.gcs && g.bytes < best.bytes)) best = g; }
  check('alloc: coupling hooks, 300 steps x 1024 particles: GCs (best of 3)', best.gcs, 0, 0);
  check('alloc: coupling hooks, 300 steps: retained heap (gc before / after)', best.retained, -Infinity, 4096, 'B');
  check('alloc: coupling hooks, 300 steps: bytes allocated per step (short-lived)', best.bytes / 300, -Infinity, 256, 'B');
  note(`alloc: hooks ran with ${pc.stats.contacts} contacts (${pc.stats.bodies} body, ${pc.stats.statics} static, ${pc.stats.ground} ground), ${buoy.stats.bodies} floating bodies, ${buoy.stats.particles} floating particles; ${(best.bytes / 300).toFixed(0)} B/step allocated (broadphase.query bounds)`);
  // the whole step: retained heap only; what it allocates per step comes from the rigid solver's V8 boxing once the
  // cloth pushes bodies around (narrowphase gjk / support / sphereBox, DynamicsWorld._blockPosition: 0 B in coupling.js)
  const steps = () => { for (let k = 0; k < 300; k++) w.step(DT); };
  const first = grow(steps), whole = grow(steps); // the first window still re-optimises code (heapUsed counts code space)
  note(`alloc: world.step x 300, first window after the hooks loop: retained ${(first.retained / 1024).toFixed(1)} KB`);
  check('alloc: world.step x 300 with both solvers: retained heap (gc before / after)', whole.retained / 1024, -Infinity, 64, 'KB');
  note(`alloc: world.step x 300: ${whole.gcs} GC(s), ${(whole.bytes / 300).toFixed(0)} B/step allocated in total (rigid solver boxing, see dyn-world)`);
  check('alloc: no NaN after 1800 steps', particlesFinite(w), 1, 1);
}
{
  // timings (Node, after warm-up): the collider with the cloth resting on 50 bodies, 2 static boxes and the ground
  const { w, pc } = allocScene(false);
  for (let k = 0; k < 360; k++) w.step(DT);
  pc.timing = true;
  let ms = 0, contacts = 0;
  for (let k = 0; k < 120; k++) { w.step(DT); ms += pc.stats.ms; contacts += pc.stats.contacts; }
  pc.timing = false;
  const msPc = ms / 120;
  check('timing: collider per step, 1024 particles vs 50 bodies (Node)', msPc, 0, 20, 'ms');
  check('timing: ... the cloth is in contact (mean contacts per step)', contacts / 120, 200, 65536);
  note(`timing: collider ${msPc.toFixed(3)} ms/step (beginStep + 4 substeps), ${(contacts / 120).toFixed(0)} contacts/step ` +
    `(${pc.stats.bodies} body, ${pc.stats.statics} static, ${pc.stats.ground} ground)`);
}
{
  // buoyancy for 200 bodies (flat fluid and a wave function)
  for (const [label, heightAt] of [['flat', 0.5], ['wave', (x, z, t) => 0.5 + 0.2 * Math.sin(0.6 * x - t) * Math.cos(0.4 * z)]]) {
    const w = new DynamicsWorld({ maxParticles: 8, sleep: false, maxBodies: 512 });
    const buoy = w.addParticleSolver(new Buoyancy(w, { fluid: { heightAt }, timing: true }));
    const rnd = mulberry32(23);
    for (let i = 0; i < 200; i++) {
      const shape = i % 3 === 0 ? sphere(0.3) : i % 3 === 1 ? box(0.3, 0.2, 0.4) : capsule(0.2, 0.3);
      w.addBody({ shape, position: [(i % 20) * 1.2 - 12, 0.5 + (rnd() - 0.5) * 0.4, ((i / 20) | 0) * 1.2 - 6], density: 300 + rnd() * 500 });
    }
    for (let k = 0; k < 60; k++) w.step(DT);
    let ms = 0;
    for (let k = 0; k < 120; k++) { w.step(DT); ms += buoy.stats.ms; }
    const per = ms / 120;
    check(`timing: buoyancy for 200 bodies, ${label} fluid (Node)`, per, 0, 30, 'ms');
    note(`timing: buoyancy ${label} ${per.toFixed(3)} ms/step for 200 bodies (${buoy.stats.bodies} in the fluid, samples ${buoy.samples})`);
  }
}

console.log(results.join('\n'));
console.log(failures ? `\n${failures} check(s) FAILED (${passed} passed)` : `\nall ${passed} checks passed`);
process.exit(failures ? 1 : 0);
