// Particle coupling of the dynamics world (pure JS, deterministic, Node-safe, no THREE): two-way XPBD contacts between
// the world's particles (particles.js) and its rigid bodies, the PhysicsWorld's static boxes and its ground, plus
// buoyancy / drag for bodies and particles in a fluid whose surface is a height function (src/world/Water.js
// `waveHeight` plugs in). Both are particle solvers: register them with DynamicsWorld.addParticleSolver, which calls
// beginStep once per step after its contact setup (stage 4: broadphase current, active set known), solvePositions in
// every position iteration of every substep after the built-in particle constraints, solveVelocities in every velocity
// iteration (both solvers here act ONCE per substep: in iteration 0, or, from a world that passes no iteration index,
// on the first call after solvePositions' iteration 0), and endStep after stage 7.
//
//   const world = new DynamicsWorld({ physics, maxParticles: 4096 });
//   const pc = world.addParticleSolver(new ParticleCollider(world, { friction: 0.4 }));
//   const buoy = world.addParticleSolver(new Buoyancy(world, { fluid: { heightAt: (x, z, t) => 0.8 + waveHeight(x, z, t, 0, 0) } }));
//   fixed step: buoy.setTime(t); world.step(dt);
//   closestPointOnShape(shape, px, py, pz, qx, qy, qz, qw, x, y, z, out7) -> out7   (cx cy cz nx ny nz dist)
//
// API
//   new ParticleCollider(world = null, { friction = 0.4, margin = 0.02, bodies = true, statics = true, ground = true,
//     mask = -1, maxContacts = 65536, thickness = 0.1, name = 'particle-collider', timing = false })
//       friction: Coulomb coefficient of every particle contact (static at position level, dynamic at velocity level;
//       live: pc.friction). margin: speculative band (m) added to the distance a particle can close within the step.
//       bodies / statics / ground: which partners are collided (the world's bodies; the PhysicsWorld's static boxes;
//       its heightfield or flat groundLevel). mask: layer bits tested against bodies.layer and, for the static world,
//       layerOf(box.kind) / LAYER.GROUND (a body's own bodies.mask is ignored: particles have no layer). maxContacts: hard cap per step (storage starts at min(cap, 4096) and doubles
//       on demand; contacts beyond the cap are dropped and counted in stats.capped). thickness: how far behind a
//       one-sided trimesh triangle a particle still counts as inside (pushed out to the front; the trimesh search and
//       the candidate culls reach at least that far behind the surface). `world` is used only
//       for sizing hints and is not retained: every hook works on the world it is given.
//     stats (reused): contacts, bodies, statics, ground (contacts of each partner kind this step), capped, ms
//       (beginStep + every substep hook, only with timing: true). count / particle / body / kind: this step's contacts
//       (SoA: particle id, body id or WORLD_BODY, 0 body / 1 static box / 2 ground), for inspection.
//     serialize() -> plain options; restore(data). The contacts are rebuilt every step, so nothing else is carried.
//   new Buoyancy(world = null, { fluid: { heightAt = 0, density = 1000, drag = 1, angularDrag = 0.5, time = 0 },
//     bodies = true, particles = true, samples = 8, mask = -1, advanceTime = true, name = 'buoyancy', timing = false })
//       heightAt(x, z, t) -> surface y at (x, z) and fluid time t; a number instead of a function is a flat level (no
//       calls, and the AABB fast paths below). density: kg/m^3. drag: linear drag, 1/s per unit submerged fraction:
//       F = -drag density V_sub v. angularDrag: torque = -angularDrag density V_sub r^2 w (r = boundingRadius). The
//       fields are live (buoy.fluid.*). advanceTime: fluid.time += dt at every endStep (setTime(t) sets it; a scene
//       that drives its own clock calls setTime before every step, which simply overrides the advance).
//       bodies: every dynamic body under gravity (layer & mask; FLAG.NO_GRAVITY bodies are skipped: without the weight
//       the buoyancy would lift them forever) floats unless unregister(id); bodies: false -> only register(id);
//       register / unregister wake the body at the next step.
//       particles: every alive, unpinned particle floats, as a sphere of its collision radius (cloth whose radius is
//       ~0.4 x its spacing is mostly air at that size and floats like foam: pass particles: false, or masses to match).
//     setTime(t); register(id); unregister(id); submergedVolume(id) -> m^3 (last evaluation); particleFraction(i);
//     sampleSet(shape) -> the shape's volume samples (below); stats (reused): bodies (bodies with V_sub > 0),
//     particles (particles with fraction > 0), submerged (total submerged body volume, m^3), ms.
//     serialize() -> { options, fluid, registration, per-body submerged / wake-reference volumes and centre heights };
//     restore(data) accepts typed arrays, plain arrays or their JSON form, and drops the cached sample sets (rebuilt
//     for the restored shapes and `samples`; a live change of `samples` rebuilds them too). heightAt is code: a
//     function is not serialized (the restored solver keeps the one it was constructed with; a number is).
//   sampleShape(shape, n = 8) -> { n, count, points: Float64Array(3c) (shape frame), weights: Float64Array(c) (m^3, sum =
//     shape.volume), cell: Float64Array(3) (cell size per local axis), centroid: Float64Array(3) (weighted), volume,
//     estimate (raw sampled volume, m^3) }   cached per (shape, n); trimesh -> count 0
//
// ParticleCollider: one step.
//   beginStep(world, dt): particle groups are found with a counting pass over particles.group (min .. max range,
//   ascending group id; a range wider than max(1024, 4 x particles) falls back to one group), each group's AABB (pos +-
//   radius) and its maximum speed in the same pass. Per group, ascending: the box grown by margin + (v_max + |g| dt +
//   v_body_max) dt queries world.broadphase (ascending ids, mask vs layer; dynamic, kinematic and static bodies with a
//   shape) and, with a PhysicsWorld, its static boxes (queryStatic on the box's XZ circle, or queryColliders without
//   one; records marked DYN_RECORD and kind 'vehicle' are skipped: they are dynamics bodies or traffic; the kept boxes
//   are insertion-sorted by (minX, minZ, minY, maxX, maxZ, maxY), so their order is a function of the set and not of
//   the grid cell a query happened to visit first; a group's static answer, asked with 1 m of slack, is reused while
//   its circle stays inside the slack and the grid gains no box: the cull below makes the result identical, and grid
//   queries cost an allocation each). Then per
//   particle (ascending id) x its group's candidates: an AABB cull, closestPointOnShape (bodies) or the closed-form
//   AABB closest point (static boxes), and a contact when dist - radius < band, band = margin + (|v_p| + |g| dt + v_b)
//   dt (v_b = |v| + |w| boundingRadius of the body): the speculative band covers what the particle and the body can
//   close within the step, so contacts exist before the surfaces meet and nothing is detected only after it went deep
//   (which would pop the particle out at depth / h). Ground: groundLevel (normal +Y, no calls) without a heightfield,
//   else physics.getGroundHeight(x, z, y) with the normal from one-sided differences over 0.2 m (3 calls: the normal
//   of the plane through (x, z), (x + 0.2, z), (x, z + 0.2), i.e. biased by half a step toward +x / +z on curved
//   ground; PhysicsWorld.groundNormal's central differences would cost 4 more calls per particle); contact
//   point (x, h, z). A contact stores particle, body (or WORLD_BODY), its normal (unit, surface -> particle) and point:
//   in the body frame for a body (both follow its pose in every substep), in world space for the static world. A body
//   is movable when the world integrates it this step (world.isActive(id), or the world's _active array / the ASLEEP
//   flag on a world without it: an awake dynamic body; one woken after stage 1, e.g. by Buoyancy's beginStep, is not).
//   solvePositions(world, h, iteration) (iteration 0 resets the multipliers): per contact, ascending:
//     C = (p - c) . n - r, c / n the stored point / normal at the body's current pose; if C < 0 the XPBD unilateral
//     correction dl = -C / (w_p + w_b) (compliance 0; w_p = particle invMass, 0 when pinned; w_b = m^-1 + (r x n)^T
//     I_w^-1 (r x n) at the lever r = c - x_b for an awake dynamic body, 0 for static / kinematic / sleeping bodies
//     and the static world): p += w_p dl n, the body is displaced by the impulse -dl n at r (xpbd.applyPositional
//     semantics, inlined), lambda_n += dl (>= 0). Static friction (Mueller 2020 sec. 3.5): the relative tangential
//     displacement over the substep, d_t = [(p - p_prev) - (c - c_prev)]_t (c_prev from the body's prevPos / prevQuat
//     when it is integrated, else c - h (v + w x r)), is undone when the multiplier it needs, |d_t| / (w_p + w_b,t),
//     is <= friction lambda_n.
//   solveVelocities(world, h, it) (once per substep, see above): per contact with lambda_n > 0: dynamic friction, an impulse against the relative
//     tangential velocity (pointVelocity of the body) of at most |v_t| / w_t and friction lambda_n / h, then zero
//     restitution: the relative normal velocity is set to 0 (paper eq. 34 with e = 0, which also removes the separation
//     speed a projection created). The body receives the opposite impulses (xpbd.applyVelocityImpulse semantics).
//   endStep(world, dt): per dynamic body the particles touched, the step's impulses on it are summed. Held up (their
//     component against gravity > 10 % of its weight impulse m |g| dt): it never sleeps (sleepTime reset; woken if
//     asleep), so a crate resting on a pinned cloth falls when the cloth is released, like a body held by an
//     attachment. Else a sleeping body (held immovable during the step: it is not integrated) is woken when the
//     non-downward part of the impulses would change its velocity by more than the world's sleep threshold
//     (world.sleepParams.linear: an undocumented DynamicsWorld field, read as is; default 0.05 m/s), e.g. a rope
//     swung into a sleeping crate.
// Buoyancy: one step.
//   beginStep(world, dt): per body (ascending id): its shape's volume samples (sampleShape: an n x n x n regular grid
//   of cell centres over the shape's local AABB, each weighted by the cell's coverage estimated from the signed
//   distance, f = clamp(1/2 - d / s_n, 0, 1) with s_n the cell's extent along the surface normal, kept when f > 0,
//   weights normalised so they sum to shape.volume exactly: a fully submerged body of the fluid's density is exactly
//   neutral) are moved to world space, and each cell counts with its submerged fraction clamp((s - (y - e)) / 2e, 0, 1)
//   (s = heightAt at its x, z; e = half the cell's world-Y extent) at the centre of its submerged part, so the
//   submerged volume is continuous in depth (exact for a level box). F = -density V_sub g at the submerged centroid
//   (force + torque about the body origin), linear drag -min(drag density V_sub, m / dt) v_cm and angular drag
//   -min(angularDrag density V_sub r^2, 1 / (dt tr(I_b^-1))) w (the clamps keep the explicit per-step forces from
//   overshooting), added to bodies.force / torque (the world integrates them in every substep and clears them).
//   Flat fluid: a body whose AABB is clear of the surface (by a cell) skips the per-sample loop (empty or full). A
//   sleeping body gets no force and costs one heightAt call (at its centre): only when the surface there moved by more
//   than 2 % of the body's height since it was last awake is it re-sampled, and woken when its submerged volume
//   changed by more than 2 % of its mass / density (a wave arriving, the level moved). Particles: the surface height under each
//   alive, unpinned particle is sampled once per step (one heightAt call, none for a flat fluid).
//   solveVelocities(world, h, it) (once per substep): per particle, fraction = clamp((s - (y - r)) / 2r, 0, 1), V_sub = 4/3 pi r^3 fraction:
//   v += -h density V_sub g / m (buoyancy as a velocity change: particles have no force accumulator), then drag
//   implicitly, v /= 1 + h drag density V_sub / m.
//   endStep(world, dt): fluid.time += dt (advanceTime).
// closestPointOnShape: shape frame, then back to world. Sphere, capsule (segment + radius), box (clamp; a point inside
// leaves through the nearest face, first axis on ties) in closed form; convex: inside -> the least-penetrating face
// plane; outside -> the projection onto a face plane (positive side) that lands inside the face polygon (its edge
// planes), else the nearest point of the edges (vertices included): exact for any convex polytope; trimesh: the
// nearest triangle through the BVH (iterative, pruned by the node AABB distance), one-sided: a point in front gets the
// unsigned distance and the direction from the closest point; behind the triangle's plane by at most `thickness` it is
// inside (distance = the plane distance, normal = the face normal); farther behind, that triangle is ignored; no
// triangle -> dist = Infinity; compound: the child with the smallest signed distance (first on ties), nested to depth
// 8. dist > 0 outside, < 0 inside; the normal points from the surface toward the point (outward when inside).
//
// Determinism: + - * / Math.sqrt and comparisons only (Math.PI as a constant); every loop runs in ascending particle /
// group / body / contact order; no Set or Map is iterated (a WeakMap caches sample sets and is only looked up, at
// registration). Allocation: the step hooks allocate nothing once warm except where they call user code with doubles
// (heightAt for function fluids and heightfields: V8 boxes the arguments), the six bounds per group handed to
// broadphase.query (~100 B per group and step, short-lived) and a static grid query when a group leaves its cached
// circle; storage grows by doubling on demand (contacts up to maxContacts, groups, candidates). Internally only
// integers, objects and typed arrays cross function boundaries (module scratch, as narrowphase.js), so no double is
// boxed per contact, and a value that is either a number or a call result is forced to a number (`+hf()`): a mixed
// one would be a tagged value boxed per sample. Not re-entrant (module scratch). test/dyn-buoyancy.mjs: 0 GCs and
// ~0 B retained over 300 steps of the hooks with 1024 particles and 50 bodies.
// Measured (Node 26, Apple M4, test/dyn-buoyancy.mjs; machine- and load-dependent, other runs read up to ~0.9 ms): the
// collider with 1024 cloth particles on 50 bodies, 2 static boxes and the ground, ~670 contacts: ~0.4-0.5 ms per step
// (4 substeps); buoyancy for 200 bodies (8^3 samples): ~0.2 ms per step with a flat fluid, ~2 ms with a sine wave
// heightAt (one call per sample: ~80 000 calls).
// Not modelled: particle-particle contacts and self-collision (cloth through cloth); continuous collision beyond the
// speculative band (contacts are found once per step against the start-of-step closest feature: sliding fast off an
// edge meets that feature's plane for the rest of the step, a ghost bump; a very thin body turning fast can be
// passed); one contact per particle and body (the nearest point: a particle in a concave trimesh corner sees one
// triangle at a time); trimesh one-sidedness (a particle arriving from behind passes through); fluid velocity (still
// water) and the body's own displacement of the surface.
import { BODY, FLAG, SHAPE, WORLD_BODY, DYN_RECORD } from './state.js';
import { worldInvInertiaMatrix, displaceBodyFrom } from './xpbd.js';
import { LAYER, layerOf } from '../PhysicsWorld.js';

const STATIC = BODY.STATIC, DYNAMIC = BODY.DYNAMIC, ASLEEP = FLAG.ASLEEP, NO_GRAVITY = FLAG.NO_GRAVITY;
const PINNED = 1;               // PFLAG.PINNED (particles.js)
const MAX_DEPTH = 8;            // compound nesting
const DEFAULT_THICKNESS = 0.1;  // m: trimesh back-side band
const GROUND_STEP = 0.2;        // m: heightfield normal finite difference
const STATIC_SLACK = 1;         // m: extra radius of a group's static-box query (its answer is reused within it)
const TRI_STACK = 256;          // BVH traversal stack (median-split trees are ~log2(n / 2) deep)
const WAKE_FRACTION = 0.02;     // buoyancy: a sleeping body wakes when its buoyancy changed by this fraction of its weight
const SUPPORT_FRACTION = 0.1;   // collider: particles carrying > 10 % of a body's weight keep it awake
const KIND_BODY = 0, KIND_STATIC = 1, KIND_GROUND = 2;
const hasPerf = typeof performance !== 'undefined' && typeof performance.now === 'function';

// ---- closest point on a shape: module scratch (never escapes; nothing below takes or returns a double) ---------------
const _PS = new Float64Array(7 * (MAX_DEPTH + 1));   // pose stack: p (0..2), q (3..6) per nesting level
const _X = new Float64Array(3);                      // query point, world
const _L = new Float64Array(3);                      // query point in the current shape frame
const _LR = new Float64Array(7);                     // local result: point, normal, signed distance
const _RES = new Float64Array(7);                    // world result
const _BEST = new Float64Array(7 * (MAX_DEPTH + 1)); // compound: best child per level
const _CFG = new Float64Array(2);                    // [0] trimesh thickness, [1] trimesh search radius
const _TV = new Float64Array(12);                    // triangle a, b, c, then its closest point
const _stack = new Int32Array(TRI_STACK);

// _L = conj(q) (_X - p), pose at level d
function toLocal(d) {
  const o = d * 7, qx = -_PS[o + 3], qy = -_PS[o + 4], qz = -_PS[o + 5], qw = _PS[o + 6];
  const x = _X[0] - _PS[o], y = _X[1] - _PS[o + 1], z = _X[2] - _PS[o + 2];
  const tx = 2 * (qy * z - qz * y), ty = 2 * (qz * x - qx * z), tz = 2 * (qx * y - qy * x);
  _L[0] = x + qw * tx + (qy * tz - qz * ty);
  _L[1] = y + qw * ty + (qz * tx - qx * tz);
  _L[2] = z + qw * tz + (qx * ty - qy * tx);
}
// _RES = (p + q c, q n, dist) from _LR, pose at level d
function toWorld(d) {
  const o = d * 7, qx = _PS[o + 3], qy = _PS[o + 4], qz = _PS[o + 5], qw = _PS[o + 6];
  let x = _LR[0], y = _LR[1], z = _LR[2];
  let tx = 2 * (qy * z - qz * y), ty = 2 * (qz * x - qx * z), tz = 2 * (qx * y - qy * x);
  _RES[0] = _PS[o] + x + qw * tx + (qy * tz - qz * ty);
  _RES[1] = _PS[o + 1] + y + qw * ty + (qz * tx - qx * tz);
  _RES[2] = _PS[o + 2] + z + qw * tz + (qx * ty - qy * tx);
  x = _LR[3]; y = _LR[4]; z = _LR[5];
  tx = 2 * (qy * z - qz * y); ty = 2 * (qz * x - qx * z); tz = 2 * (qx * y - qy * x);
  _RES[3] = x + qw * tx + (qy * tz - qz * ty);
  _RES[4] = y + qw * ty + (qz * tx - qx * tz);
  _RES[5] = z + qw * tz + (qx * ty - qy * tx);
  _RES[6] = _LR[6];
}
function setLR(cx, cy, cz, nx, ny, nz, d) { _LR[0] = cx; _LR[1] = cy; _LR[2] = cz; _LR[3] = nx; _LR[4] = ny; _LR[5] = nz; _LR[6] = d; }

function localSphere(s) {
  const x = _L[0], y = _L[1], z = _L[2], r = s.radius, d2 = x * x + y * y + z * z;
  if (d2 > 1e-24) {
    const d = Math.sqrt(d2), nx = x / d, ny = y / d, nz = z / d;
    _LR[0] = nx * r; _LR[1] = ny * r; _LR[2] = nz * r; _LR[3] = nx; _LR[4] = ny; _LR[5] = nz; _LR[6] = d - r;
  } else { _LR[0] = 0; _LR[1] = r; _LR[2] = 0; _LR[3] = 0; _LR[4] = 1; _LR[5] = 0; _LR[6] = -r; }
}

function localCapsule(s) {
  const x = _L[0], y = _L[1], z = _L[2], r = s.radius, hh = s.halfHeight;
  const t = y > hh ? hh : y < -hh ? -hh : y;
  const vy = y - t, d2 = x * x + vy * vy + z * z;
  let nx, ny, nz, d;
  if (d2 > 1e-24) { d = Math.sqrt(d2); nx = x / d; ny = vy / d; nz = z / d; } else if (hh > 0) { d = 0; nx = 1; ny = 0; nz = 0; } else { d = 0; nx = 0; ny = 1; nz = 0; }
  _LR[0] = nx * r; _LR[1] = t + ny * r; _LR[2] = nz * r; _LR[3] = nx; _LR[4] = ny; _LR[5] = nz; _LR[6] = d - r;
}

function localBox(s) {
  const h = s.half, hx = h[0], hy = h[1], hz = h[2], x = _L[0], y = _L[1], z = _L[2];
  const bx = x > hx ? hx : x < -hx ? -hx : x, by = y > hy ? hy : y < -hy ? -hy : y, bz = z > hz ? hz : z < -hz ? -hz : z;
  const dx = x - bx, dy = y - by, dz = z - bz, d2 = dx * dx + dy * dy + dz * dz;
  if (d2 > 1e-24) {
    const d = Math.sqrt(d2);
    _LR[0] = bx; _LR[1] = by; _LR[2] = bz; _LR[3] = dx / d; _LR[4] = dy / d; _LR[5] = dz / d; _LR[6] = d;
    return;
  }
  // inside (or on the surface): out through the nearest face, first axis on ties
  const px = hx - (x < 0 ? -x : x), py = hy - (y < 0 ? -y : y), pz = hz - (z < 0 ? -z : z);
  let ax = 0, pen = px;
  if (py < pen) { ax = 1; pen = py; }
  if (pz < pen) { ax = 2; pen = pz; }
  _LR[0] = x; _LR[1] = y; _LR[2] = z; _LR[3] = 0; _LR[4] = 0; _LR[5] = 0; _LR[6] = -pen;
  const c = ax === 0 ? x : ax === 1 ? y : z, sg = c >= 0 ? 1 : -1;
  _LR[ax] = sg * h[ax]; _LR[3 + ax] = sg;
}

function localConvex(s) {
  const pl = s.planes, m = s.faceCount, x = _L[0], y = _L[1], z = _L[2];
  let best = -Infinity, bf = 0;
  for (let f = 0; f < m; f++) {
    const o = f * 4, sd = pl[o] * x + pl[o + 1] * y + pl[o + 2] * z + pl[o + 3];
    if (sd > best) { best = sd; bf = f; }
  }
  if (best <= 0) { // inside: out through the least-penetrating face
    const o = bf * 4, nx = pl[o], ny = pl[o + 1], nz = pl[o + 2];
    setLR(x - best * nx, y - best * ny, z - best * nz, nx, ny, nz, best);
    return;
  }
  // outside: a face whose plane projection lands inside its polygon is the closest feature (x - proj is then in the
  // normal cone at proj), else the nearest point lies on an edge (or a vertex, an edge end)
  const pts = s.points, fs = s.faceStart, fv = s.faceVerts;
  for (let f = 0; f < m; f++) {
    const o = f * 4, nx = pl[o], ny = pl[o + 1], nz = pl[o + 2];
    const sd = nx * x + ny * y + nz * z + pl[o + 3];
    if (!(sd > 0)) continue;
    const qx = x - sd * nx, qy = y - sd * ny, qz = z - sd * nz;
    let inside = true;
    for (let k = fs[f], e = fs[f + 1]; k < e; k++) {
      const a = fv[k] * 3, b = fv[k + 1 < e ? k + 1 : fs[f]] * 3;
      const ex = pts[b] - pts[a], ey = pts[b + 1] - pts[a + 1], ez = pts[b + 2] - pts[a + 2];
      const ix = ny * ez - nz * ey, iy = nz * ex - nx * ez, iz = nx * ey - ny * ex; // n x e: into the face (CCW loop)
      if ((qx - pts[a]) * ix + (qy - pts[a + 1]) * iy + (qz - pts[a + 2]) * iz < 0) { inside = false; break; }
    }
    if (inside) { setLR(qx, qy, qz, nx, ny, nz, sd); return; }
  }
  const E = s.edges;
  let bd2 = Infinity, cx = 0, cy = 0, cz = 0;
  for (let k = 0, ne = E.length; k < ne; k += 2) {
    const a = E[k] * 3, b = E[k + 1] * 3, ax = pts[a], ay = pts[a + 1], az = pts[a + 2];
    const ex = pts[b] - ax, ey = pts[b + 1] - ay, ez = pts[b + 2] - az, ee = ex * ex + ey * ey + ez * ez;
    let t = ee > 0 ? ((x - ax) * ex + (y - ay) * ey + (z - az) * ez) / ee : 0;
    t = t < 0 ? 0 : t > 1 ? 1 : t;
    const px = ax + ex * t, py = ay + ey * t, pz = az + ez * t;
    const dx = x - px, dy = y - py, dz = z - pz, d2 = dx * dx + dy * dy + dz * dz;
    if (d2 < bd2) { bd2 = d2; cx = px; cy = py; cz = pz; }
  }
  const d = Math.sqrt(bd2);
  if (d > 1e-12) setLR(cx, cy, cz, (x - cx) / d, (y - cy) / d, (z - cz) / d, d);
  else { const o = bf * 4; setLR(cx, cy, cz, pl[o], pl[o + 1], pl[o + 2], d); }
}

// closest point of triangle _TV[0..8] (a, b, c) to _L -> _TV[9..11] (Ericson 5.1.5)
function closestTri() {
  const ax = _TV[0], ay = _TV[1], az = _TV[2], bx = _TV[3], by = _TV[4], bz = _TV[5], cx = _TV[6], cy = _TV[7], cz = _TV[8];
  const px = _L[0], py = _L[1], pz = _L[2];
  const abx = bx - ax, aby = by - ay, abz = bz - az, acx = cx - ax, acy = cy - ay, acz = cz - az;
  const apx = px - ax, apy = py - ay, apz = pz - az;
  const d1 = abx * apx + aby * apy + abz * apz, d2 = acx * apx + acy * apy + acz * apz;
  if (d1 <= 0 && d2 <= 0) { _TV[9] = ax; _TV[10] = ay; _TV[11] = az; return; }
  const bpx = px - bx, bpy = py - by, bpz = pz - bz;
  const d3 = abx * bpx + aby * bpy + abz * bpz, d4 = acx * bpx + acy * bpy + acz * bpz;
  if (d3 >= 0 && d4 <= d3) { _TV[9] = bx; _TV[10] = by; _TV[11] = bz; return; }
  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) { const v = d1 / (d1 - d3); _TV[9] = ax + abx * v; _TV[10] = ay + aby * v; _TV[11] = az + abz * v; return; }
  const cpx = px - cx, cpy = py - cy, cpz = pz - cz;
  const d5 = abx * cpx + aby * cpy + abz * cpz, d6 = acx * cpx + acy * cpy + acz * cpz;
  if (d6 >= 0 && d5 <= d6) { _TV[9] = cx; _TV[10] = cy; _TV[11] = cz; return; }
  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) { const w = d2 / (d2 - d6); _TV[9] = ax + acx * w; _TV[10] = ay + acy * w; _TV[11] = az + acz * w; return; }
  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
    const w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    _TV[9] = bx + (cx - bx) * w; _TV[10] = by + (cy - by) * w; _TV[11] = bz + (cz - bz) * w;
    return;
  }
  const den = 1 / (va + vb + vc), v = vb * den, w = vc * den;
  _TV[9] = ax + abx * v + acx * w; _TV[10] = ay + aby * v + acy * w; _TV[11] = az + abz * v + acz * w;
}

function localTrimesh(s) {
  const b = s.bvh, nMin = b.nodeMin, nMax = b.nodeMax, left = b.left, right = b.right, order = b.triOrder;
  const tStart = b.triStart, tCount = b.triCount, pos = s.positions, idx = s.indices;
  const x = _L[0], y = _L[1], z = _L[2], thick = _CFG[0], lim = _CFG[1];
  let best2 = lim * lim, found = false, bsd = 0, bcx = 0, bcy = 0, bcz = 0, bnx = 0, bny = 1, bnz = 0;
  let sp = 0;
  _stack[sp++] = 0;
  while (sp > 0) {
    const node = _stack[--sp], o = node * 3;
    const dx = x < nMin[o] ? nMin[o] - x : x > nMax[o] ? x - nMax[o] : 0;
    const dy = y < nMin[o + 1] ? nMin[o + 1] - y : y > nMax[o + 1] ? y - nMax[o + 1] : 0;
    const dz = z < nMin[o + 2] ? nMin[o + 2] - z : z > nMax[o + 2] ? z - nMax[o + 2] : 0;
    if (dx * dx + dy * dy + dz * dz > best2) continue;
    if (left[node] >= 0) {
      if (sp + 2 <= TRI_STACK) { _stack[sp++] = right[node]; _stack[sp++] = left[node]; }
      continue;
    }
    for (let k = tStart[node], e = k + tCount[node]; k < e; k++) {
      const t = order[k], i0 = idx[t * 3] * 3, i1 = idx[t * 3 + 1] * 3, i2 = idx[t * 3 + 2] * 3;
      _TV[0] = pos[i0]; _TV[1] = pos[i0 + 1]; _TV[2] = pos[i0 + 2];
      _TV[3] = pos[i1]; _TV[4] = pos[i1 + 1]; _TV[5] = pos[i1 + 2];
      _TV[6] = pos[i2]; _TV[7] = pos[i2 + 1]; _TV[8] = pos[i2 + 2];
      closestTri();
      const cx = _TV[9], cy = _TV[10], cz = _TV[11], ex = x - cx, ey = y - cy, ez = z - cz, d2 = ex * ex + ey * ey + ez * ez;
      if (!(d2 < best2)) continue;
      const e1x = _TV[3] - _TV[0], e1y = _TV[4] - _TV[1], e1z = _TV[5] - _TV[2], e2x = _TV[6] - _TV[0], e2y = _TV[7] - _TV[1], e2z = _TV[8] - _TV[2];
      let nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
      const l = Math.sqrt(nx * nx + ny * ny + nz * nz);
      if (!(l > 1e-20)) continue;
      nx /= l; ny /= l; nz /= l;
      const sd = (x - _TV[0]) * nx + (y - _TV[1]) * ny + (z - _TV[2]) * nz;
      if (sd < 0 && -sd > thick) continue; // behind the one-sided triangle beyond its thickness: not a contact side
      best2 = d2; found = true; bsd = sd; bcx = cx; bcy = cy; bcz = cz; bnx = nx; bny = ny; bnz = nz;
    }
  }
  if (!found) { setLR(x, y, z, 0, 1, 0, Infinity); return; }
  if (bsd >= 0) {
    const d = Math.sqrt(best2);
    if (d > 1e-12) setLR(bcx, bcy, bcz, (x - bcx) / d, (y - bcy) / d, (z - bcz) / d, d);
    else setLR(bcx, bcy, bcz, bnx, bny, bnz, 0);
  } else setLR(bcx, bcy, bcz, bnx, bny, bnz, bsd); // inside within the thickness: out along the face normal
}

function closestCompound(s, d) {
  const ch = s.children, o = d * 7, nd = d + 1, bo = d * 7;
  if (nd > MAX_DEPTH) { _RES[6] = Infinity; return; }
  const px = _PS[o], py = _PS[o + 1], pz = _PS[o + 2], qx = _PS[o + 3], qy = _PS[o + 4], qz = _PS[o + 5], qw = _PS[o + 6];
  const c0 = nd * 7;
  let found = false;
  for (let i = 0; i < ch.length; i++) {
    const c = ch[i], cp = c.pos, cq = c.quat, x = cp[0], y = cp[1], z = cp[2], bx = cq[0], by = cq[1], bz = cq[2], bw = cq[3];
    const tx = 2 * (qy * z - qz * y), ty = 2 * (qz * x - qx * z), tz = 2 * (qx * y - qy * x);
    _PS[c0] = px + x + qw * tx + (qy * tz - qz * ty);
    _PS[c0 + 1] = py + y + qw * ty + (qz * tx - qx * tz);
    _PS[c0 + 2] = pz + z + qw * tz + (qx * ty - qy * tx);
    _PS[c0 + 3] = qw * bx + qx * bw + qy * bz - qz * by;
    _PS[c0 + 4] = qw * by - qx * bz + qy * bw + qz * bx;
    _PS[c0 + 5] = qw * bz + qx * by - qy * bx + qz * bw;
    _PS[c0 + 6] = qw * bw - qx * bx - qy * by - qz * bz;
    closestCore(c.shape, nd);
    if (!found || _RES[6] < _BEST[bo + 6]) { for (let k = 0; k < 7; k++) _BEST[bo + k] = _RES[k]; found = true; }
  }
  if (found) for (let k = 0; k < 7; k++) _RES[k] = _BEST[bo + k];
  else _RES[6] = Infinity;
}

// closest point of shape s at the pose of level d to _X -> _RES
function closestCore(s, d) {
  const t = s.type;
  if (t === SHAPE.COMPOUND) { closestCompound(s, d); return; }
  toLocal(d);
  if (t === SHAPE.SPHERE) localSphere(s);
  else if (t === SHAPE.CAPSULE) localCapsule(s);
  else if (t === SHAPE.BOX) localBox(s);
  else if (t === SHAPE.CONVEX) localConvex(s);
  else if (t === SHAPE.TRIMESH) localTrimesh(s);
  else setLR(_L[0], _L[1], _L[2], 0, 1, 0, Infinity);
  toWorld(d);
}

// closest point of the world AABB record bx ({ minX, minY, minZ, maxX, maxY, maxZ }) to _X -> _RES
function closestAabb(bx) {
  const x = _X[0], y = _X[1], z = _X[2];
  const minX = bx.minX, minY = bx.minY, minZ = bx.minZ, maxX = bx.maxX, maxY = bx.maxY, maxZ = bx.maxZ;
  const cx = x < minX ? minX : x > maxX ? maxX : x, cy = y < minY ? minY : y > maxY ? maxY : y, cz = z < minZ ? minZ : z > maxZ ? maxZ : z;
  const dx = x - cx, dy = y - cy, dz = z - cz, d2 = dx * dx + dy * dy + dz * dz;
  _RES[0] = cx; _RES[1] = cy; _RES[2] = cz;
  if (d2 > 1e-24) {
    const d = Math.sqrt(d2);
    _RES[3] = dx / d; _RES[4] = dy / d; _RES[5] = dz / d; _RES[6] = d;
    return;
  }
  // inside: out through the nearest face (order -x +x -y +y -z +z on ties)
  let f = 0, pen = x - minX, p;
  if ((p = maxX - x) < pen) { pen = p; f = 1; }
  if ((p = y - minY) < pen) { pen = p; f = 2; }
  if ((p = maxY - y) < pen) { pen = p; f = 3; }
  if ((p = z - minZ) < pen) { pen = p; f = 4; }
  if ((p = maxZ - z) < pen) { pen = p; f = 5; }
  _RES[0] = x; _RES[1] = y; _RES[2] = z; _RES[3] = 0; _RES[4] = 0; _RES[5] = 0; _RES[6] = -pen;
  const ax = f >> 1, sg = (f & 1) === 1 ? 1 : -1;
  _RES[ax] = ax === 0 ? (sg > 0 ? maxX : minX) : ax === 1 ? (sg > 0 ? maxY : minY) : (sg > 0 ? maxZ : minZ);
  _RES[3 + ax] = sg;
}

/**
 * Closest point of a shape at a pose to the world point (x, y, z): out[0..2] = point on the surface, out[3..5] = unit
 * normal from the surface toward the point (outward when the point is inside), out[6] = signed distance (< 0 inside;
 * Infinity when a trimesh offers no triangle on the point's side). Allocation-free (module scratch: not re-entrant).
 * @param {object} shape  shapes.js shape
 * @param {number} px @param {number} py @param {number} pz   shape position
 * @param {number} qx @param {number} qy @param {number} qz @param {number} qw   shape rotation (unit quaternion)
 * @param {number} x @param {number} y @param {number} z   query point
 * @param {Float64Array|number[]} out   7 numbers
 * @param {number} [thickness=0.1]   trimesh: how far behind a triangle's plane a point still counts as inside
 * @returns {Float64Array|number[]} out
 */
export function closestPointOnShape(shape, px, py, pz, qx, qy, qz, qw, x, y, z, out, thickness = DEFAULT_THICKNESS) {
  _PS[0] = px; _PS[1] = py; _PS[2] = pz; _PS[3] = qx; _PS[4] = qy; _PS[5] = qz; _PS[6] = qw;
  _X[0] = x; _X[1] = y; _X[2] = z;
  _CFG[0] = thickness; _CFG[1] = Infinity;
  closestCore(shape, 0);
  for (let k = 0; k < 7; k++) out[k] = _RES[k];
  return out;
}

// ---- volume samples (construction time: allocates) ------------------------------------------------------------------
const _sampleCache = new WeakMap(); // shape -> [set, ...] (one per grid size)

/**
 * Volume samples of a shape for buoyancy (see the header): an n x n x n grid of cell centres over the shape's local
 * AABB, weighted by coverage and normalised to shape.volume. Cached per (shape, n); a trimesh gets an empty set.
 * @param {object} shape  @param {number} [n=8]
 * @returns {{ n: number, count: number, points: Float64Array, weights: Float64Array, cell: Float64Array, centroid: Float64Array, volume: number, estimate: number }}
 */
export function sampleShape(shape, n = 8) {
  n = Math.max(1, n | 0);
  let list = _sampleCache.get(shape);
  if (list) {
    for (let i = 0; i < list.length; i++) if (list[i].n === n) return list[i];
  } else { list = []; _sampleCache.set(shape, list); }
  const set = buildSamples(shape, n);
  list.push(set);
  return set;
}

function buildSamples(shape, n) {
  const empty = { n, count: 0, points: new Float64Array(0), weights: new Float64Array(0), cell: new Float64Array(3), centroid: new Float64Array(3), volume: shape.volume, estimate: 0 };
  if (shape.type === SHAPE.TRIMESH || !(shape.volume > 0)) return empty;
  const ab = shape.aabb(0, 0, 0, 0, 0, 0, 1, new Float64Array(6), 0);
  const sx = (ab[3] - ab[0]) / n, sy = (ab[4] - ab[1]) / n, sz = (ab[5] - ab[2]) / n;
  const pts = [], cov = [], r = new Float64Array(7);
  let sum = 0;
  for (let k = 0; k < n; k++) {
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < n; i++) {
        const x = ab[0] + (i + 0.5) * sx, y = ab[1] + (j + 0.5) * sy, z = ab[2] + (k + 0.5) * sz;
        closestPointOnShape(shape, 0, 0, 0, 0, 0, 0, 1, x, y, z, r);
        const d = r[6];
        if (!(d < Infinity)) continue;
        // the cell's extent along the surface normal: a linear coverage ramp across it (exact for a face-aligned cell)
        const se = Math.sqrt(r[3] * r[3] * sx * sx + r[4] * r[4] * sy * sy + r[5] * r[5] * sz * sz);
        let f = se > 0 ? 0.5 - d / se : d <= 0 ? 1 : 0;
        f = f < 0 ? 0 : f > 1 ? 1 : f;
        if (!(f > 0)) continue;
        pts.push(x, y, z); cov.push(f); sum += f;
      }
    }
  }
  const count = cov.length;
  if (count === 0) return empty;
  const points = Float64Array.from(pts), weights = new Float64Array(count), centroid = new Float64Array(3);
  const V = shape.volume;
  for (let q = 0; q < count; q++) {
    const w = weights[q] = V * cov[q] / sum;
    centroid[0] += w * points[q * 3]; centroid[1] += w * points[q * 3 + 1]; centroid[2] += w * points[q * 3 + 2];
  }
  centroid[0] /= V; centroid[1] /= V; centroid[2] /= V;
  return { n, count, points, weights, cell: Float64Array.of(sx, sy, sz), centroid, volume: V, estimate: sum * sx * sy * sz };
}

// ---- solver helpers (integers / typed arrays only) -----------------------------------------------------------------
const _M9 = new Float64Array(9);   // world inverse inertia of the current contact's body
const _D6 = new Float64Array(6);   // body displacement (dx dy dz, rotation vector)
const _C = new Float64Array(16);   // current contact: c (0..2), n (3..5), lever r (6..8), c_prev (9..11)
const _BS = new Float64Array(4);   // buoyancy: submerged volume, centroid
const _FL = new Float64Array(4);   // buoyancy: level (flat fluid), time, e (cell half extent), unused

// _C[0..8] = world point, normal and lever of contact k (body frame -> world for bodies; stored world values otherwise)
function contactWorld(pc, B, k) {
  const b = pc.body[k], o = k * 3, L = pc._cl, N = pc._cn;
  if (b < 0) {
    _C[0] = L[o]; _C[1] = L[o + 1]; _C[2] = L[o + 2]; _C[3] = N[o]; _C[4] = N[o + 1]; _C[5] = N[o + 2];
    _C[6] = 0; _C[7] = 0; _C[8] = 0;
    return;
  }
  const P = B.pos, Q = B.quat, p3 = b * 3, q4 = b * 4, qx = Q[q4], qy = Q[q4 + 1], qz = Q[q4 + 2], qw = Q[q4 + 3];
  let x = L[o], y = L[o + 1], z = L[o + 2];
  let tx = 2 * (qy * z - qz * y), ty = 2 * (qz * x - qx * z), tz = 2 * (qx * y - qy * x);
  const rx = x + qw * tx + (qy * tz - qz * ty), ry = y + qw * ty + (qz * tx - qx * tz), rz = z + qw * tz + (qx * ty - qy * tx);
  _C[6] = rx; _C[7] = ry; _C[8] = rz;
  _C[0] = P[p3] + rx; _C[1] = P[p3 + 1] + ry; _C[2] = P[p3 + 2] + rz;
  x = N[o]; y = N[o + 1]; z = N[o + 2];
  tx = 2 * (qy * z - qz * y); ty = 2 * (qz * x - qx * z); tz = 2 * (qx * y - qy * x);
  _C[3] = x + qw * tx + (qy * tz - qz * ty); _C[4] = y + qw * ty + (qz * tx - qx * tz); _C[5] = z + qw * tz + (qx * ty - qy * tx);
}

// _C[9..11] = where contact k's body point was at the start of the substep (h = pc._h[0])
function contactPrev(pc, B, k) {
  const b = pc.body[k];
  if (b < 0) { _C[9] = _C[0]; _C[10] = _C[1]; _C[11] = _C[2]; return; }
  if (pc._cm[k] === 1) {
    const o = k * 3, L = pc._cl, PP = B.prevPos, PQ = B.prevQuat, p3 = b * 3, q4 = b * 4;
    const qx = PQ[q4], qy = PQ[q4 + 1], qz = PQ[q4 + 2], qw = PQ[q4 + 3], x = L[o], y = L[o + 1], z = L[o + 2];
    const tx = 2 * (qy * z - qz * y), ty = 2 * (qz * x - qx * z), tz = 2 * (qx * y - qy * x);
    _C[9] = PP[p3] + x + qw * tx + (qy * tz - qz * ty);
    _C[10] = PP[p3 + 1] + y + qw * ty + (qz * tx - qx * tz);
    _C[11] = PP[p3 + 2] + z + qw * tz + (qx * ty - qy * tx);
    return;
  }
  // not integrated by the solver (static, kinematic, sleeping): back along its velocity (0 when at rest)
  const h = pc._h[0], V = B.vel, W = B.angVel, p3 = b * 3, wx = W[p3], wy = W[p3 + 1], wz = W[p3 + 2];
  const rx = _C[6], ry = _C[7], rz = _C[8];
  _C[9] = _C[0] - h * (V[p3] + wy * rz - wz * ry);
  _C[10] = _C[1] - h * (V[p3 + 1] + wz * rx - wx * rz);
  _C[11] = _C[2] - h * (V[p3 + 2] + wx * ry - wy * rx);
}

// displace body b by the position-level impulse _D6[0..2] at the lever _C[6..8] (the impulse is overwritten)
function pushBody(B, b) {
  const im = B.invMass[b], px = _D6[0], py = _D6[1], pz = _D6[2], rx = _C[6], ry = _C[7], rz = _C[8];
  const cx = ry * pz - rz * py, cy = rz * px - rx * pz, cz = rx * py - ry * px;
  _D6[0] = px * im; _D6[1] = py * im; _D6[2] = pz * im;
  _D6[3] = _M9[0] * cx + _M9[1] * cy + _M9[2] * cz;
  _D6[4] = _M9[3] * cx + _M9[4] * cy + _M9[5] * cz;
  _D6[5] = _M9[6] * cx + _M9[7] * cy + _M9[8] * cz;
  displaceBodyFrom(B, b, _D6, 0);
}

// velocity impulse _D6[0..2] on body b at the lever _C[6..8]
function kickBody(B, b) {
  const im = B.invMass[b], px = _D6[0], py = _D6[1], pz = _D6[2], rx = _C[6], ry = _C[7], rz = _C[8], o = b * 3;
  const cx = ry * pz - rz * py, cy = rz * px - rx * pz, cz = rx * py - ry * px;
  const V = B.vel, W = B.angVel;
  V[o] += px * im; V[o + 1] += py * im; V[o + 2] += pz * im;
  W[o] += _M9[0] * cx + _M9[1] * cy + _M9[2] * cz;
  W[o + 1] += _M9[3] * cx + _M9[4] * cy + _M9[5] * cz;
  W[o + 2] += _M9[6] * cx + _M9[7] * cy + _M9[8] * cz;
}

// generalized inverse mass of the current contact's body along _D6[0..2] (unit), lever _C[6..8], from _M9
function bodyW(B, b) {
  const rx = _C[6], ry = _C[7], rz = _C[8], nx = _D6[0], ny = _D6[1], nz = _D6[2];
  const cx = ry * nz - rz * ny, cy = rz * nx - rx * nz, cz = rx * ny - ry * nx;
  _D6[3] = B.invMass[b] + cx * (_M9[0] * cx + _M9[1] * cy + _M9[2] * cz) + cy * (_M9[3] * cx + _M9[4] * cy + _M9[5] * cz) +
    cz * (_M9[6] * cx + _M9[7] * cy + _M9[8] * cz);
}

// static box a sorts before b: (minX, minZ, minY, maxX, maxZ, maxY) lexicographically (identical boxes keep query order:
// they produce identical contacts, so their order cannot change a result)
function boxBefore(a, b) {
  if (a.minX !== b.minX) return a.minX < b.minX;
  if (a.minZ !== b.minZ) return a.minZ < b.minZ;
  if (a.minY !== b.minY) return a.minY < b.minY;
  if (a.maxX !== b.maxX) return a.maxX < b.maxX;
  if (a.maxZ !== b.maxZ) return a.maxZ < b.maxZ;
  return a.maxY < b.maxY;
}

const growI32 = (a, n) => { const g = new Int32Array(n); g.set(a); return g; };
// typed-array view of serialized data (typed array, plain array, or a JSON'd typed array { 0: .., 1: .. })
const asArray = (v) => (v == null ? [] : ArrayBuffer.isView(v) || Array.isArray(v) ? v : Object.values(v));
const growF64 = (a, n) => { const g = new Float64Array(n); g.set(a); return g; };
const growU8 = (a, n) => { const g = new Uint8Array(n); g.set(a); return g; };

// ---- ParticleCollider -----------------------------------------------------------------------------------------------
export class ParticleCollider {
  /**
   * @param {object|null} [world]  the DynamicsWorld (sizing hint only; not retained)
   * @param {object} [o]  see the file header
   */
  constructor(world = null, o = {}) {
    const {
      friction = 0.4, margin = 0.02, bodies = true, statics = true, ground = true, mask = -1, maxContacts = 65536,
      thickness = DEFAULT_THICKNESS, name = 'particle-collider', timing = false,
    } = o;
    this.name = name;
    this.friction = friction;
    this.margin = margin;
    this.bodies = bodies !== false;
    this.statics = statics !== false;
    this.ground = ground !== false;
    this.mask = mask | 0;
    this.maxContacts = Math.max(1, maxContacts | 0);
    this.thickness = thickness;
    this.timing = timing === true && hasPerf;
    this.stats = { contacts: 0, bodies: 0, statics: 0, ground: 0, capped: 0, ms: 0 };
    this.count = 0;
    this._h = new Float64Array(1);  // [0] the current substep
    this._velDone = false;          // the velocity pass ran in this substep (it runs once, not per velocity iteration)
    this._allocContacts(Math.min(this.maxContacts, 4096));
    // groups (slot = group id - base)
    this._gCap = 0;
    this._allocGroups(64);
    this._cand = new Int32Array(256);    // candidate body ids, per slot ranges
    this._qbuf = new Int32Array(256);    // broadphase query output
    this._sbox = [];                     // candidate static boxes (shared PhysicsWorld records), per slot ranges
    this._sOut = [];                     // static query output
    this._sList = [];                    // per slot: cached static boxes (see _allocGroups)
    this._sGrid = null;                  // the grid the static caches were built on
    const nb = world && world.bodies ? world.bodies.capacity : 64;
    this._bspd = new Float64Array(nb);   // per body: |v| + |w| boundingRadius at beginStep
    this._bmv = new Uint8Array(nb);      // per body: 0 immovable, 1 integrated this step, 2 dynamic but not integrated
    this._wk = new Float64Array(3 * nb); // per body: the step's particle impulse on it (N s; sleep / wake rules)
    this._wf = new Uint8Array(nb);
  }

  _allocContacts(cap) {
    const n = this.count;
    const old = this.particle;
    this.particle = old ? growI32(old, cap) : new Int32Array(cap);
    this.body = old ? growI32(this.body, cap) : new Int32Array(cap);
    this.kind = old ? growU8(this.kind, cap) : new Uint8Array(cap);  // KIND_*
    this._cm = old ? growU8(this._cm, cap) : new Uint8Array(cap);   // the body's _bmv code (1 movable, 2 not integrated)
    this._cn = old ? growF64(this._cn, 3 * cap) : new Float64Array(3 * cap); // normal (body frame for bodies)
    this._cl = old ? growF64(this._cl, 3 * cap) : new Float64Array(3 * cap); // point (body frame for bodies)
    this._lam = old ? growF64(this._lam, cap) : new Float64Array(cap);       // substep normal multiplier (kg m)
    this._imp = old ? growF64(this._imp, 3 * cap) : new Float64Array(3 * cap); // step impulse on the dynamic body (N s)
    this._cap = cap;
    this.count = n;
  }

  _allocGroups(cap) {
    const old = this._gCap > 0;
    this._gCnt = old ? growI32(this._gCnt, cap) : new Int32Array(cap);
    this._gBox = old ? growF64(this._gBox, 6 * cap) : new Float64Array(6 * cap);
    this._gV2 = old ? growF64(this._gV2, cap) : new Float64Array(cap);
    this._gC = old ? growI32(this._gC, 2 * cap) : new Int32Array(2 * cap);   // candidate body range
    this._gS = old ? growI32(this._gS, 2 * cap) : new Int32Array(2 * cap);   // static box range
    // static query cache per slot: where (x, z, radius) it was asked, the grid's box count then (-1: none), the group,
    // and the boxes it kept (this._sList[slot], counted by _sCnt)
    this._sqx = old ? growF64(this._sqx, cap) : new Float64Array(cap);
    this._sqz = old ? growF64(this._sqz, cap) : new Float64Array(cap);
    this._sqr = old ? growF64(this._sqr, cap) : new Float64Array(cap);
    this._sqn = old ? growI32(this._sqn, cap) : new Int32Array(cap).fill(-1);
    this._sqg = old ? growI32(this._sqg, cap) : new Int32Array(cap);
    this._sCnt = old ? growI32(this._sCnt, cap) : new Int32Array(cap);
    if (old) this._sqn.fill(-1, this._gCap);
    this._gCap = cap;
  }

  _ensureBodies(n) {
    if (this._bspd.length >= n) return;
    this._bspd = growF64(this._bspd, n); this._bmv = growU8(this._bmv, n); this._wk = growF64(this._wk, 3 * n); this._wf = growU8(this._wf, n);
  }

  // append a contact from _RES (world point / normal) for particle i and body b (WORLD_BODY: static world)
  _add(B, i, b, kind, movable) {
    const k = this.count;
    if (k >= this._cap) {
      if (this._cap >= this.maxContacts) { this.stats.capped++; return; }
      this._allocContacts(Math.min(this.maxContacts, this._cap * 2));
    }
    const o = k * 3, N = this._cn, L = this._cl;
    this.particle[k] = i; this.body[k] = b; this.kind[k] = kind; this._cm[k] = movable;
    this._lam[k] = 0; this._imp[o] = 0; this._imp[o + 1] = 0; this._imp[o + 2] = 0;
    if (b < 0) {
      L[o] = _RES[0]; L[o + 1] = _RES[1]; L[o + 2] = _RES[2]; N[o] = _RES[3]; N[o + 1] = _RES[4]; N[o + 2] = _RES[5];
    } else {
      // body frame: conj(q) (c - x_b), conj(q) n
      const P = B.pos, Q = B.quat, p3 = b * 3, q4 = b * 4, qx = -Q[q4], qy = -Q[q4 + 1], qz = -Q[q4 + 2], qw = Q[q4 + 3];
      let x = _RES[0] - P[p3], y = _RES[1] - P[p3 + 1], z = _RES[2] - P[p3 + 2];
      let tx = 2 * (qy * z - qz * y), ty = 2 * (qz * x - qx * z), tz = 2 * (qx * y - qy * x);
      L[o] = x + qw * tx + (qy * tz - qz * ty); L[o + 1] = y + qw * ty + (qz * tx - qx * tz); L[o + 2] = z + qw * tz + (qx * ty - qy * tx);
      x = _RES[3]; y = _RES[4]; z = _RES[5];
      tx = 2 * (qy * z - qz * y); ty = 2 * (qz * x - qx * z); tz = 2 * (qx * y - qy * x);
      N[o] = x + qw * tx + (qy * tz - qz * ty); N[o + 1] = y + qw * ty + (qz * tx - qx * tz); N[o + 2] = z + qw * tz + (qx * ty - qy * tx);
    }
    this.count = k + 1;
  }

  /** Per step: collect the contacts of this step (see the header). */
  beginStep(world, dt) {
    const t0 = this.timing ? performance.now() : 0;
    const st = this.stats, Pt = world.particles;
    this.count = 0;
    st.contacts = 0; st.bodies = 0; st.statics = 0; st.ground = 0; st.capped = 0; st.ms = 0;
    if (!Pt) return;
    const n = Pt.count !== undefined ? Pt.count : Pt.capacity;
    const alive = Pt.alive, grp = Pt.group, X = Pt.pos, V = Pt.vel, R = Pt.radius;
    // 1. groups: range of ids, then per slot count, AABB (pos +- radius), max speed^2
    let gmin = 0, gmax = -1, nAlive = 0;
    for (let i = 0; i < n; i++) {
      if (!alive[i]) continue;
      const g = grp[i];
      if (nAlive === 0) { gmin = g; gmax = g; } else { if (g < gmin) gmin = g; if (g > gmax) gmax = g; }
      nAlive++;
    }
    if (nAlive === 0) { this._timed(t0); return; }
    let slots = gmax - gmin + 1;
    const single = slots > Math.max(1024, 4 * nAlive);
    if (single) slots = 1;
    if (slots > this._gCap) { let c = this._gCap; while (c < slots) c *= 2; this._allocGroups(c); }
    const gc = this._gCnt, gb = this._gBox, gv = this._gV2;
    gc.fill(0, 0, slots); gv.fill(0, 0, slots);
    for (let i = 0; i < n; i++) {
      if (!alive[i]) continue;
      const s = single ? 0 : grp[i] - gmin, o = i * 3, b6 = s * 6, r = R[i];
      const x = X[o], y = X[o + 1], z = X[o + 2], vx = V[o], vy = V[o + 1], vz = V[o + 2];
      const v2 = vx * vx + vy * vy + vz * vz;
      if (gc[s] === 0) {
        gb[b6] = x - r; gb[b6 + 1] = y - r; gb[b6 + 2] = z - r; gb[b6 + 3] = x + r; gb[b6 + 4] = y + r; gb[b6 + 5] = z + r;
      } else {
        if (x - r < gb[b6]) gb[b6] = x - r; if (y - r < gb[b6 + 1]) gb[b6 + 1] = y - r; if (z - r < gb[b6 + 2]) gb[b6 + 2] = z - r;
        if (x + r > gb[b6 + 3]) gb[b6 + 3] = x + r; if (y + r > gb[b6 + 4]) gb[b6 + 4] = y + r; if (z + r > gb[b6 + 5]) gb[b6 + 5] = z + r;
      }
      if (v2 > gv[s]) gv[s] = v2;
      gc[s]++;
    }
    // 2. body speeds (band of each pair; the query grows by the fastest)
    const B = world.bodies, nb = B.count, G = world.gravity;
    this._ensureBodies(B.capacity);
    const bs = this._bspd, bmv = this._bmv, BV = B.vel, BW = B.angVel;
    const hasIsActive = typeof world.isActive === 'function', act = world._active instanceof Uint8Array ? world._active : null;
    let vbMax = 0;
    for (let id = 0; id < nb; id++) {
      // movable = integrated this step (a body woken after the world's stage 1, e.g. by Buoyancy's beginStep, is not)
      let m = 0;
      if (B.alive[id] && B.type[id] === DYNAMIC && B.invMass[id] > 0) {
        const on = hasIsActive ? world.isActive(id) : act !== null ? act[id] === 1 : (B.flags[id] & ASLEEP) === 0;
        m = on ? 1 : 2;
      }
      bmv[id] = m;
      let v = 0;
      if (B.alive[id] && B.type[id] !== STATIC && (B.flags[id] & ASLEEP) === 0) {
        const o = id * 3, lv = BV[o] * BV[o] + BV[o + 1] * BV[o + 1] + BV[o + 2] * BV[o + 2];
        const aw = BW[o] * BW[o] + BW[o + 1] * BW[o + 1] + BW[o + 2] * BW[o + 2];
        v = Math.sqrt(lv) + Math.sqrt(aw) * B.boundingRadius[id];
      }
      bs[id] = v;
      if (v > vbMax) vbMax = v;
    }
    const gdt = Math.sqrt(G[0] * G[0] + G[1] * G[1] + G[2] * G[2]) * dt, margin = this.margin;
    // 3. candidates per slot (ascending group id)
    const bp = world.broadphase, ph = world.physics, mask = this.mask;
    const useBodies = this.bodies && bp && typeof bp.query === 'function';
    const useStatics = this.statics && ph && (typeof ph.queryStatic === 'function' || typeof ph.queryColliders === 'function');
    const gC = this._gC, gS = this._gS, sbox = this._sbox, sOut = this._sOut, th = this.thickness;
    const grid = useStatics ? ph.grid : null, boxes = grid ? grid.boxes : null;
    const cacheable = boxes !== null && boxes !== undefined && typeof ph.queryStatic === 'function';
    if (grid !== this._sGrid) { this._sGrid = grid; this._sqn.fill(-1); }
    let nc = 0, ns = 0;
    for (let s = 0; s < slots; s++) {
      gC[s * 2] = nc; gS[s * 2] = ns;
      if (gc[s] === 0) { gC[s * 2 + 1] = nc; gS[s * 2 + 1] = ns; continue; }
      const b6 = s * 6, grow = margin + (Math.sqrt(gv[s]) + gdt + vbMax) * dt;
      const minX = gb[b6] - grow, minY = gb[b6 + 1] - grow, minZ = gb[b6 + 2] - grow;
      const maxX = gb[b6 + 3] + grow, maxY = gb[b6 + 4] + grow, maxZ = gb[b6 + 5] + grow;
      if (useBodies) {
        // the body query reaches `thickness` farther: a particle that far behind a trimesh's triangles is still inside it
        const e = th > grow ? th - grow : 0;
        let q = bp.query(minX - e, minY - e, minZ - e, maxX + e, maxY + e, maxZ + e, this._qbuf, mask);
        if (bp.queryTotal > q) {
          let c = this._qbuf.length;
          while (c < bp.queryTotal) c *= 2;
          this._qbuf = new Int32Array(c);
          q = bp.query(minX - e, minY - e, minZ - e, maxX + e, maxY + e, maxZ + e, this._qbuf, mask);
        }
        const qb = this._qbuf;
        if (nc + q > this._cand.length) { let c = this._cand.length; while (c < nc + q) c *= 2; this._cand = growI32(this._cand, c); }
        const cand = this._cand;
        for (let j = 0; j < q; j++) { const id = qb[j]; if (B.shape[id] !== null && (B.layer[id] & mask) !== 0) cand[nc++] = id; }
      }
      if (useStatics) {
        const cx = (minX + maxX) * 0.5, cz = (minZ + maxZ) * 0.5, hx = (maxX - minX) * 0.5, hz = (maxZ - minZ) * 0.5;
        const r = Math.sqrt(hx * hx + hz * hz), gid = single ? -2147483648 : gmin + s;
        if (this._sList.length <= s) for (let k = this._sList.length; k <= s; k++) this._sList.push([]);
        const list = this._sList[s];
        // the slot's last answer is reused while the circle (cx, cz, r) lies inside the circle it was asked for (r +
        // STATIC_SLACK) and the grid gained no box: a superset of a fresh query, culled below to the same result
        const dqx = cx - this._sqx[s], dqz = cz - this._sqz[s], room = this._sqr[s] - r;
        let cnt;
        if (cacheable && this._sqn[s] === boxes.length && this._sqg[s] === gid && room >= 0 && dqx * dqx + dqz * dqz <= room * room) cnt = this._sCnt[s];
        else {
          sOut.length = 0;
          if (cacheable) ph.queryStatic(cx, cz, r + STATIC_SLACK, sOut); else ph.queryColliders(cx, cz, r, sOut);
          cnt = 0;
          for (let j = 0; j < sOut.length; j++) {
            const bx = sOut[j];
            if (bx[DYN_RECORD] === true || bx.kind === 'vehicle') continue;
            if (mask !== -1 && (mask & layerOf(bx.kind)) === 0) continue;
            // insertion sort by (minX, minZ, minY, maxX, maxZ, maxY): the order is a pure function of the box set, not
            // of the cell the grid query happened to visit first (a cached and a fresh answer agree: serialize / from)
            let q = cnt++;
            while (q > 0 && boxBefore(bx, list[q - 1])) { list[q] = list[q - 1]; q--; }
            list[q] = bx;
          }
          sOut.length = 0;
          this._sCnt[s] = cnt;
          if (cacheable) { this._sqx[s] = cx; this._sqz[s] = cz; this._sqr[s] = r + STATIC_SLACK; this._sqn[s] = boxes.length; this._sqg[s] = gid; }
          else this._sqn[s] = -1;
        }
        for (let j = 0; j < cnt; j++) {
          const bx = list[j];
          if (bx.maxY < minY || bx.minY > maxY || bx.maxX < minX || bx.minX > maxX || bx.maxZ < minZ || bx.minZ > maxZ) continue;
          sbox[ns++] = bx;
        }
      }
      gC[s * 2 + 1] = nc; gS[s * 2 + 1] = ns;
    }
    // 4. contacts: per particle (ascending) x its group's candidates, then the ground
    const cand = this._cand, Ab = B.aabb, BP = B.pos, BQ = B.quat, shp = B.shape, TRI = SHAPE.TRIMESH;
    let useGround = this.ground && ph !== null && ph !== undefined && (mask & LAYER.GROUND) !== 0;
    const flat = useGround && ph.heightfield == null && typeof ph.groundLevel === 'number';
    if (useGround && !flat && typeof ph.getGroundHeight !== 'function') useGround = false;
    const level = flat ? ph.groundLevel : 0;
    for (let i = 0; i < n; i++) {
      if (!alive[i]) continue;
      const s = single ? 0 : grp[i] - gmin, o = i * 3, r = R[i];
      const px = X[o], py = X[o + 1], pz = X[o + 2], vx = V[o], vy = V[o + 1], vz = V[o + 2];
      const bandP = margin + (Math.sqrt(vx * vx + vy * vy + vz * vz) + gdt) * dt;
      _X[0] = px; _X[1] = py; _X[2] = pz;
      for (let j = gC[s * 2], e = gC[s * 2 + 1]; j < e; j++) {
        const b = cand[j], band = bandP + bs[b] * dt, reach = r + band, a6 = b * 6, sh = shp[b];
        const cull = sh.type === TRI && th > reach ? th : reach; // a trimesh's back-side band lies outside its AABB
        if (Ab[a6] > px + cull || Ab[a6 + 3] < px - cull || Ab[a6 + 1] > py + cull || Ab[a6 + 4] < py - cull ||
          Ab[a6 + 2] > pz + cull || Ab[a6 + 5] < pz - cull) continue;
        const p3 = b * 3, q4 = b * 4;
        _PS[0] = BP[p3]; _PS[1] = BP[p3 + 1]; _PS[2] = BP[p3 + 2]; _PS[3] = BQ[q4]; _PS[4] = BQ[q4 + 1]; _PS[5] = BQ[q4 + 2]; _PS[6] = BQ[q4 + 3];
        _CFG[0] = th; _CFG[1] = reach > th ? reach : th; // search a trimesh at least as deep as its back-side band
        closestCore(sh, 0);
        if (!(_RES[6] - r < band)) continue;
        this._add(B, i, b, KIND_BODY, bmv[b]);
        st.bodies++;
      }
      for (let j = gS[s * 2], e = gS[s * 2 + 1]; j < e; j++) {
        const bx = sbox[j], reach = r + bandP;
        if (bx.minX > px + reach || bx.maxX < px - reach || bx.minY > py + reach || bx.maxY < py - reach || bx.minZ > pz + reach || bx.maxZ < pz - reach) continue;
        closestAabb(bx);
        if (!(_RES[6] - r < bandP)) continue;
        this._add(B, i, WORLD_BODY, KIND_STATIC, 0);
        st.statics++;
      }
      if (useGround) {
        let h, nx = 0, ny = 1, nz = 0;
        if (flat) h = level;
        else {
          h = +ph.getGroundHeight(px, pz, py); // a number, so h stays an unboxed double on the flat path too
          const gx = (h - ph.getGroundHeight(px + GROUND_STEP, pz, py)) / GROUND_STEP;
          const gz = (h - ph.getGroundHeight(px, pz + GROUND_STEP, py)) / GROUND_STEP;
          const inv = 1 / Math.sqrt(gx * gx + 1 + gz * gz);
          nx = gx * inv; ny = inv; nz = gz * inv;
        }
        if ((py - h) * ny - r < bandP) {
          _RES[0] = px; _RES[1] = h; _RES[2] = pz; _RES[3] = nx; _RES[4] = ny; _RES[5] = nz;
          this._add(B, i, WORLD_BODY, KIND_GROUND, 0);
          st.ground++;
        }
      }
    }
    st.contacts = this.count;
    this._timed(t0);
  }

  _timed(t0) { if (this.timing) this.stats.ms += performance.now() - t0; }

  /** Per substep and position iteration: non-penetration + static friction (see the header). */
  solvePositions(world, h, iteration) {
    if (iteration === 0) this._velDone = false;
    const nc = this.count;
    if (nc === 0) return;
    const t0 = this.timing ? performance.now() : 0;
    this._h[0] = h;
    if (iteration === 0) this._lam.fill(0, 0, nc);
    const Pt = world.particles, B = world.bodies, X = Pt.pos, XP = Pt.prevPos, IM = Pt.invMass, FL = Pt.flags, R = Pt.radius;
    const lam = this._lam, cm = this._cm, imp = this._imp, mu = this.friction, ih = 1 / h;
    for (let k = 0; k < nc; k++) {
      const i = this.particle[k], b = this.body[k], o = i * 3, ko = k * 3;
      const wp = (FL[i] & PINNED) !== 0 ? 0 : IM[i];
      const mv = cm[k] === 1;
      contactWorld(this, B, k);
      if (mv) worldInvInertiaMatrix(B, b, _M9, 0);
      let nx = _C[3], ny = _C[4], nz = _C[5];
      const C = (X[o] - _C[0]) * nx + (X[o + 1] - _C[1]) * ny + (X[o + 2] - _C[2]) * nz - R[i];
      if (C < 0) {
        let wb = 0;
        if (mv) { _D6[0] = nx; _D6[1] = ny; _D6[2] = nz; bodyW(B, b); wb = _D6[3]; }
        const w = wp + wb;
        if (w > 0) {
          const dl = -C / w;
          lam[k] += dl;
          X[o] += wp * dl * nx; X[o + 1] += wp * dl * ny; X[o + 2] += wp * dl * nz;
          if (cm[k] !== 0) { imp[ko] -= dl * nx * ih; imp[ko + 1] -= dl * ny * ih; imp[ko + 2] -= dl * nz * ih; }
          if (mv) { _D6[0] = -dl * nx; _D6[1] = -dl * ny; _D6[2] = -dl * nz; pushBody(B, b); contactWorld(this, B, k); worldInvInertiaMatrix(B, b, _M9, 0); nx = _C[3]; ny = _C[4]; nz = _C[5]; }
        }
      }
      // static friction: undo the relative tangential slip of this substep while |lambda_t| <= mu lambda_n
      const ln = lam[k];
      if (!(ln > 0) || !(mu > 0)) continue;
      contactPrev(this, B, k);
      let dx = (X[o] - XP[o]) - (_C[0] - _C[9]), dy = (X[o + 1] - XP[o + 1]) - (_C[1] - _C[10]), dz = (X[o + 2] - XP[o + 2]) - (_C[2] - _C[11]);
      const dn = dx * nx + dy * ny + dz * nz;
      dx -= dn * nx; dy -= dn * ny; dz -= dn * nz;
      const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (!(d > 1e-12)) continue;
      const tx = dx / d, ty = dy / d, tz = dz / d;
      let wt = wp;
      if (mv) { _D6[0] = tx; _D6[1] = ty; _D6[2] = tz; bodyW(B, b); wt += _D6[3]; }
      if (!(wt > 0)) continue;
      const lt = d / wt;
      if (lt > mu * ln) continue;
      X[o] -= wp * lt * tx; X[o + 1] -= wp * lt * ty; X[o + 2] -= wp * lt * tz;
      if (cm[k] !== 0) { imp[ko] += lt * tx * ih; imp[ko + 1] += lt * ty * ih; imp[ko + 2] += lt * tz * ih; }
      if (mv) { _D6[0] = lt * tx; _D6[1] = lt * ty; _D6[2] = lt * tz; pushBody(B, b); }
    }
    this._timed(t0);
  }

  /**
   * Per substep, after the world recovered velocities: dynamic friction + zero normal restitution (see the header).
   * Runs once per substep: in velocity iteration 0 (`it`; without it, the first call after solvePositions' iteration 0).
   */
  solveVelocities(world, h, it) {
    if ((it !== undefined && it !== 0) || this._velDone) return;
    this._velDone = true;
    const nc = this.count;
    if (nc === 0) return;
    const t0 = this.timing ? performance.now() : 0;
    const Pt = world.particles, B = world.bodies, V = Pt.vel, IM = Pt.invMass, FL = Pt.flags;
    const lam = this._lam, cm = this._cm, imp = this._imp, mu = this.friction, ih = 1 / h, BV = B.vel, BW = B.angVel;
    for (let k = 0; k < nc; k++) {
      const ln = lam[k];
      if (!(ln > 0)) continue;
      const i = this.particle[k], b = this.body[k], o = i * 3, ko = k * 3;
      const wp = (FL[i] & PINNED) !== 0 ? 0 : IM[i];
      const mv = cm[k] === 1;
      contactWorld(this, B, k);
      if (mv) worldInvInertiaMatrix(B, b, _M9, 0);
      const nx = _C[3], ny = _C[4], nz = _C[5], rx = _C[6], ry = _C[7], rz = _C[8];
      // relative velocity of the particle against the body point
      let ux = 0, uy = 0, uz = 0;
      if (b >= 0) {
        const p3 = b * 3, wx = BW[p3], wy = BW[p3 + 1], wz = BW[p3 + 2];
        ux = BV[p3] + wy * rz - wz * ry; uy = BV[p3 + 1] + wz * rx - wx * rz; uz = BV[p3 + 2] + wx * ry - wy * rx;
      }
      let vx = V[o] - ux, vy = V[o + 1] - uy, vz = V[o + 2] - uz;
      let vn = vx * nx + vy * ny + vz * nz;
      // dynamic friction
      const tx0 = vx - vn * nx, ty0 = vy - vn * ny, tz0 = vz - vn * nz, vt = Math.sqrt(tx0 * tx0 + ty0 * ty0 + tz0 * tz0);
      if (vt > 1e-12 && mu > 0) {
        const tx = tx0 / vt, ty = ty0 / vt, tz = tz0 / vt;
        let wt = wp;
        if (mv) { _D6[0] = tx; _D6[1] = ty; _D6[2] = tz; bodyW(B, b); wt += _D6[3]; }
        if (wt > 0) {
          let j = vt / wt;
          const jmax = mu * ln * ih;
          if (j > jmax) j = jmax;
          V[o] -= wp * j * tx; V[o + 1] -= wp * j * ty; V[o + 2] -= wp * j * tz;
          if (cm[k] !== 0) { imp[ko] += j * tx; imp[ko + 1] += j * ty; imp[ko + 2] += j * tz; }
          if (mv) { _D6[0] = j * tx; _D6[1] = j * ty; _D6[2] = j * tz; kickBody(B, b); }
          // relative normal velocity after the friction impulse (the body may have turned)
          if (b >= 0) {
            const p3 = b * 3, wx = BW[p3], wy = BW[p3 + 1], wz = BW[p3 + 2];
            ux = BV[p3] + wy * rz - wz * ry; uy = BV[p3 + 1] + wz * rx - wx * rz; uz = BV[p3 + 2] + wx * ry - wy * rx;
          }
          vx = V[o] - ux; vy = V[o + 1] - uy; vz = V[o + 2] - uz;
          vn = vx * nx + vy * ny + vz * nz;
        }
      }
      // zero restitution: relative normal velocity -> 0
      if (vn === 0) continue;
      let wn = wp;
      if (mv) { _D6[0] = nx; _D6[1] = ny; _D6[2] = nz; bodyW(B, b); wn += _D6[3]; }
      if (!(wn > 0)) continue;
      const jn = -vn / wn;
      V[o] += wp * jn * nx; V[o + 1] += wp * jn * ny; V[o + 2] += wp * jn * nz;
      if (cm[k] !== 0) { imp[ko] -= jn * nx; imp[ko + 1] -= jn * ny; imp[ko + 2] -= jn * nz; }
      if (mv) { _D6[0] = -jn * nx; _D6[1] = -jn * ny; _D6[2] = -jn * nz; kickBody(B, b); }
    }
    this._timed(t0);
  }

  /**
   * Per step: sleep / wake rules for the dynamic bodies the particles touched (see the header): a body the particles
   * hold up never sleeps (sleepTime reset; woken if asleep); a sleeping body hit hard enough is woken.
   */
  endStep(world, dt) {
    const nc = this.count;
    if (nc === 0) return;
    const B = world.bodies, cm = this._cm, imp = this._imp, wk = this._wk, wf = this._wf, body = this.body;
    let any = false;
    for (let k = 0; k < nc; k++) if (cm[k] !== 0) { const b = body[k]; wf[b] = 0; wk[b * 3] = 0; wk[b * 3 + 1] = 0; wk[b * 3 + 2] = 0; any = true; }
    if (!any) return;
    for (let k = 0; k < nc; k++) {
      if (cm[k] === 0) continue;
      const b = body[k], o = b * 3, ko = k * 3;
      wk[o] += imp[ko]; wk[o + 1] += imp[ko + 1]; wk[o + 2] += imp[ko + 2];
    }
    const G = world.gravity, gl = Math.sqrt(G[0] * G[0] + G[1] * G[1] + G[2] * G[2]);
    const gx = gl > 0 ? G[0] / gl : 0, gy = gl > 0 ? G[1] / gl : 0, gz = gl > 0 ? G[2] / gl : 0;
    const sp = world.sleepParams, thr = sp && sp.linear > 0 ? sp.linear : 0.05;
    const canWake = typeof world.wake === 'function';
    for (let k = 0; k < nc; k++) {
      if (cm[k] === 0) continue;
      const b = body[k], o = b * 3;
      if (wf[b]) continue;
      wf[b] = 1;
      let jx = wk[o], jy = wk[o + 1], jz = wk[o + 2];
      const down = jx * gx + jy * gy + jz * gz; // < 0: the particles push the body against gravity
      const asleep = (B.flags[b] & ASLEEP) !== 0;
      if (-down > SUPPORT_FRACTION * B.mass[b] * gl * dt) { // held up by particles (cloth, a rope): never sleeps
        B.sleepTime[b] = 0;
        if (asleep && canWake) world.wake(b);
        continue;
      }
      if (!asleep) continue;
      if (down > 0) { jx -= down * gx; jy -= down * gy; jz -= down * gz; } // pressing along gravity cannot move it
      const dv = Math.sqrt(jx * jx + jy * jy + jz * jz) * B.invMass[b];
      if (dv > thr && canWake) world.wake(b);
    }
  }

  /** Plain options (the contacts are rebuilt every step). */
  serialize() {
    return {
      name: this.name, friction: this.friction, margin: this.margin, bodies: this.bodies, statics: this.statics,
      ground: this.ground, mask: this.mask, maxContacts: this.maxContacts, thickness: this.thickness,
    };
  }

  restore(d) {
    if (!d) return;
    for (const k of ['friction', 'margin', 'bodies', 'statics', 'ground', 'mask', 'maxContacts', 'thickness']) if (d[k] !== undefined) this[k] = d[k];
    this.count = 0;
  }
}

// Submerged volume and centroid of body id from its samples -> _BS (flat fluid: hf === null, level _FL[0]; else
// hf(x, z, _FL[1])). Every typed-array access sits on a path that runs on every call (a branch that is cold while V8
// optimises the step loop would bail out of the cached optimised code each time it is first taken).
function submergedOf(B, id, set, hf) {
  const p3 = id * 3, q4 = id * 4, P = B.pos, Q = B.quat, A = B.aabb, a6 = id * 6;
  const qx = Q[q4], qy = Q[q4 + 1], qz = Q[q4 + 2], qw = Q[q4 + 3];
  const x2 = qx + qx, y2 = qy + qy, z2 = qz + qz, xx = qx * x2, xy = qx * y2, xz = qx * z2, yy = qy * y2, yz = qy * z2, zz = qz * z2;
  const wx = qw * x2, wy = qw * y2, wz = qw * z2;
  const m0 = 1 - (yy + zz), m1 = xy - wz, m2 = xz + wy, m3 = xy + wz, m4 = 1 - (xx + zz), m5 = yz - wx, m6 = xz - wy, m7 = yz + wx, m8 = 1 - (xx + yy);
  const px = P[p3], py = P[p3 + 1], pz = P[p3 + 2], cell = set.cell, c = set.centroid, lo = A[a6 + 1], hi = A[a6 + 4];
  const ccx = c[0], ccy = c[1], ccz = c[2], vol = set.volume, level = _FL[0], t = _FL[1];
  const e = 0.5 * ((m3 < 0 ? -m3 : m3) * cell[0] + (m4 < 0 ? -m4 : m4) * cell[1] + (m5 < 0 ? -m5 : m5) * cell[2]);
  const flat = hf === null;
  // flat fluid clear of the surface by a cell: empty (mode 0) or full (mode 1) without the per-sample loop
  const mode = !flat ? 2 : lo - e >= level ? 0 : hi + e <= level ? 1 : 2;
  let V = 0, sx = 0, sy = 0, sz = 0;
  if (mode === 1) {
    V = vol;
    sx = (px + m0 * ccx + m1 * ccy + m2 * ccz) * V; sy = (py + m3 * ccx + m4 * ccy + m5 * ccz) * V; sz = (pz + m6 * ccx + m7 * ccy + m8 * ccz) * V;
  } else if (mode === 2) {
    // two loops: in one, `flat ? level : hf(...)` would make the surface height a tagged value (a boxed level per sample)
    const pts = set.points, w = set.weights, nS = set.count, i2e = e > 0 ? 1 / (2 * e) : 0;
    if (flat) {
      for (let q = 0; q < nS; q++) {
        const lx = pts[q * 3], ly = pts[q * 3 + 1], lz = pts[q * 3 + 2], wq = w[q];
        const x = px + m0 * lx + m1 * ly + m2 * lz, y = py + m3 * lx + m4 * ly + m5 * lz, z = pz + m6 * lx + m7 * ly + m8 * lz;
        let f = e > 0 ? (level - (y - e)) * i2e : level >= y ? 1 : 0;
        f = f < 0 ? 0 : f > 1 ? 1 : f;
        const v = wq * f;
        V += v; sx += v * x; sy += v * (y - e + e * f); sz += v * z;
      }
    } else {
      for (let q = 0; q < nS; q++) {
        const lx = pts[q * 3], ly = pts[q * 3 + 1], lz = pts[q * 3 + 2], wq = w[q];
        const x = px + m0 * lx + m1 * ly + m2 * lz, y = py + m3 * lx + m4 * ly + m5 * lz, z = pz + m6 * lx + m7 * ly + m8 * lz;
        const s = +hf(x, z, t);
        let f = e > 0 ? (s - (y - e)) * i2e : s >= y ? 1 : 0;
        f = f < 0 ? 0 : f > 1 ? 1 : f;
        const v = wq * f;
        V += v; sx += v * x; sy += v * (y - e + e * f); sz += v * z;
      }
    }
  }
  let cx = px, cy = py, cz = pz;
  if (V > 0) { cx = sx / V; cy = sy / V; cz = sz / V; }
  _BS[0] = V; _BS[1] = cx; _BS[2] = cy; _BS[3] = cz;
}

// ---- Buoyancy -------------------------------------------------------------------------------------------------------
export class Buoyancy {
  /**
   * @param {object|null} [world]  the DynamicsWorld (sizing hint only; not retained)
   * @param {object} [o]  see the file header
   */
  constructor(world = null, o = {}) {
    const { fluid = {}, bodies = true, particles = true, samples = 8, mask = -1, advanceTime = true, name = 'buoyancy', timing = false } = o;
    this.name = name;
    this.fluid = {
      heightAt: fluid.heightAt ?? 0, density: fluid.density ?? 1000, drag: fluid.drag ?? 1,
      angularDrag: fluid.angularDrag ?? 0.5, time: fluid.time ?? 0,
    };
    this.bodies = bodies !== false;
    this.particles = particles !== false;
    this.samples = Math.max(1, samples | 0);
    this.mask = mask | 0;
    this.advanceTime = advanceTime !== false;
    this.timing = timing === true && hasPerf;
    this.stats = { bodies: 0, particles: 0, submerged: 0, ms: 0 };
    const nb = world && world.bodies ? world.bodies.capacity : 64;
    this._reg = new Uint8Array(nb);         // 0 default, 1 registered, 2 unregistered
    this._set = new Array(nb).fill(null);   // per body: sample set (sampleShape)
    this._setShape = new Array(nb).fill(null);
    this._vsub = new Float64Array(nb);      // last submerged volume
    this._vref = new Float64Array(nb);      // submerged volume when last awake (wake test)
    this._hc = new Float64Array(nb);        // surface height at the body centre when last awake (sleeping re-sample gate)
    this._wakeReq = new Uint8Array(nb);     // register / unregister: wake at the next beginStep
    this._velDone = false;                  // particle buoyancy ran in this substep (once, not per velocity iteration)
    const np = world && world.particles ? world.particles.capacity || 0 : 0;
    this._ps = new Float64Array(Math.max(1, np));  // per particle: surface height (this step)
    this._pf = new Float64Array(Math.max(1, np));  // per particle: last submerged fraction
  }

  /** Set the fluid time (seconds; passed to heightAt). */
  setTime(t) { this.fluid.time = t; }
  /** Make body `id` float (needed with { bodies: false }; undoes unregister). */
  register(id) { this._ensureBodies(id + 1); this._reg[id] = 1; this._wakeReq[id] = 1; }
  /** Exclude body `id` (woken at the next step: its forces change). */
  unregister(id) { this._ensureBodies(id + 1); this._reg[id] = 2; this._vsub[id] = 0; this._vref[id] = 0; this._wakeReq[id] = 1; }
  /** Submerged volume of body `id` at its last evaluation (m^3). */
  submergedVolume(id) { return id >= 0 && id < this._vsub.length ? this._vsub[id] : 0; }
  /** Submerged fraction of particle `i` at the last substep (0..1). */
  particleFraction(i) { return i >= 0 && i < this._pf.length ? this._pf[i] : 0; }
  /** The volume samples this solver uses for a shape (sampleShape(shape, samples)). */
  sampleSet(shape) { return sampleShape(shape, this.samples); }

  _ensureBodies(n) {
    if (this._reg.length >= n) return;
    let c = this._reg.length;
    while (c < n) c *= 2;
    this._reg = growU8(this._reg, c); this._vsub = growF64(this._vsub, c); this._vref = growF64(this._vref, c);
    this._hc = growF64(this._hc, c); this._wakeReq = growU8(this._wakeReq, c);
    const s = new Array(c).fill(null), ss = new Array(c).fill(null);
    for (let i = 0; i < this._set.length; i++) { s[i] = this._set[i]; ss[i] = this._setShape[i]; }
    this._set = s; this._setShape = ss;
  }
  _ensureParticles(n) {
    if (this._ps.length >= n) return;
    this._ps = growF64(this._ps, n); this._pf = growF64(this._pf, n);
  }

  /** Per step: body forces, particle surface heights (see the header). */
  beginStep(world, dt) {
    const t0 = this.timing ? performance.now() : 0;
    const st = this.stats, fl = this.fluid, B = world.bodies, G = world.gravity;
    st.bodies = 0; st.particles = 0; st.submerged = 0; st.ms = 0;
    const flat = typeof fl.heightAt !== 'function';
    const hf = flat ? null : fl.heightAt;
    _FL[0] = flat ? (Number.isFinite(fl.heightAt) ? fl.heightAt : 0) : 0;
    _FL[1] = fl.time;
    const rho = fl.density, gx = G[0], gy = G[1], gz = G[2];
    // bodies
    const nb = B.count;
    this._ensureBodies(B.capacity);
    const reg = this._reg, mask = this.mask, all = this.bodies, N = this.samples, P = B.pos, A = B.aabb;
    const hasIsActive = typeof world.isActive === 'function', act = world._active instanceof Uint8Array ? world._active : null;
    const canWake = typeof world.wake === 'function', level = _FL[0], t = _FL[1];
    let submerged = 0, nBodies = 0;
    const wq = this._wakeReq;
    for (let id = 0; id < nb; id++) {
      if (wq[id] !== 0) { wq[id] = 0; if (B.alive[id] && (B.flags[id] & ASLEEP) !== 0 && canWake) world.wake(id); }
      // dynamic bodies under gravity only (a NO_GRAVITY body would float up forever)
      if (!B.alive[id] || B.type[id] !== DYNAMIC || !(B.invMass[id] > 0) || (B.flags[id] & NO_GRAVITY) !== 0) continue;
      const rg = reg[id];
      if (rg === 2 || (!all && rg !== 1) || (B.layer[id] & mask) === 0) continue;
      const s = B.shape[id];
      if (s === null || s.type === SHAPE.TRIMESH) continue;
      let set = this._set[id];
      if (set === null || this._setShape[id] !== s || set.n !== N) { set = sampleShape(s, N); this._set[id] = set; this._setShape[id] = s; }
      if (set.count === 0) continue;
      const p3 = id * 3, a6 = id * 6;
      const hc = flat ? level : +hf(P[p3], P[p3 + 2], t); // surface at the centre
      const on = hasIsActive ? world.isActive(id) : act !== null ? act[id] === 1 : (B.flags[id] & ASLEEP) === 0;
      if (!on) { // not integrated this step: no force
        if ((B.flags[id] & ASLEEP) === 0) continue;
        // asleep: re-sampled only once the surface at its centre moved by WAKE_FRACTION x its height since it was last
        // awake, then woken when its submerged volume changed by WAKE_FRACTION x mass / density
        const dh = hc - this._hc[id], hgt = A[a6 + 4] - A[a6 + 1];
        if (!((dh < 0 ? -dh : dh) > WAKE_FRACTION * hgt)) continue;
        submergedOf(B, id, set, hf);
        const V = _BS[0], dV = V - this._vref[id];
        this._vsub[id] = V;
        if ((dV < 0 ? -dV : dV) * rho > WAKE_FRACTION * B.mass[id] && canWake) world.wake(id);
        continue;
      }
      submergedOf(B, id, set, hf);
      const V = _BS[0];
      this._vsub[id] = V; this._vref[id] = V; this._hc[id] = hc;
      if (!(V > 0)) continue;
      nBodies++; submerged += V;
      const F = B.force, T = B.torque, m = B.mass[id];
      // buoyancy at the submerged centroid
      const k = -rho * V, fx = k * gx, fy = k * gy, fz = k * gz;
      const rx = _BS[1] - P[p3], ry = _BS[2] - P[p3 + 1], rz = _BS[3] - P[p3 + 2];
      // linear drag (clamped to what stops the body within the step)
      let cl = fl.drag * rho * V;
      if (cl * dt > m) cl = m / dt;
      const v = B.vel;
      F[p3] += fx - cl * v[p3]; F[p3 + 1] += fy - cl * v[p3 + 1]; F[p3 + 2] += fz - cl * v[p3 + 2];
      // angular drag (clamped by the inertia bound tr(I_b^-1) >= the largest eigenvalue of I^-1)
      const br = B.boundingRadius[id], I = B.invInertia, i9 = id * 9, tr = I[i9] + I[i9 + 4] + I[i9 + 8];
      let ca = fl.angularDrag * rho * V * br * br;
      if (ca * dt * tr > 1) ca = 1 / (dt * tr);
      const w = B.angVel;
      T[p3] += ry * fz - rz * fy - ca * w[p3];
      T[p3 + 1] += rz * fx - rx * fz - ca * w[p3 + 1];
      T[p3 + 2] += rx * fy - ry * fx - ca * w[p3 + 2];
    }
    st.bodies = nBodies; st.submerged = submerged;
    // particles: surface height under each (once per step)
    const Pt = world.particles;
    if (this.particles && Pt) {
      const n = Pt.count !== undefined ? Pt.count : Pt.capacity;
      this._ensureParticles(Pt.capacity || n);
      const alive = Pt.alive, X = Pt.pos, ps = this._ps;
      if (flat) { for (let i = 0; i < n; i++) if (alive[i]) ps[i] = level; } // (one loop with `flat ? level : hf()` boxes level)
      else for (let i = 0; i < n; i++) if (alive[i]) ps[i] = +hf(X[i * 3], X[i * 3 + 2], t);
    }
    if (this.timing) st.ms += performance.now() - t0;
  }

  /** Marks the start of a substep (the velocity pass runs once per substep). */
  solvePositions(world, h, iteration) { if (iteration === 0) this._velDone = false; }

  /**
   * Per substep: particle buoyancy and drag as velocity changes (see the header). Once per substep: in velocity
   * iteration 0 (`it`; without it, the first call after solvePositions' iteration 0).
   */
  solveVelocities(world, h, it) {
    if ((it !== undefined && it !== 0) || this._velDone) return;
    this._velDone = true;
    const Pt = world.particles;
    if (!this.particles || !Pt) return;
    const t0 = this.timing ? performance.now() : 0;
    const n = Pt.count !== undefined ? Pt.count : Pt.capacity;
    const alive = Pt.alive, X = Pt.pos, V = Pt.vel, IM = Pt.invMass, M = Pt.mass, R = Pt.radius, FLG = Pt.flags;
    const ps = this._ps, pf = this._pf, fl = this.fluid, rho = fl.density, drag = fl.drag, G = world.gravity;
    const gx = G[0], gy = G[1], gz = G[2], k43 = 4 / 3 * Math.PI;
    let count = 0;
    for (let i = 0; i < n; i++) {
      if (!alive[i]) continue;
      pf[i] = 0;
      if ((FLG[i] & PINNED) !== 0 || !(IM[i] > 0)) continue;
      const o = i * 3, r = R[i];
      if (!(r > 0)) continue;
      let f = (ps[i] - (X[o + 1] - r)) / (2 * r);
      if (!(f > 0)) continue;
      if (f > 1) f = 1;
      pf[i] = f;
      count++;
      const m = M[i] > 0 ? M[i] : 1 / IM[i];
      const Vs = k43 * r * r * r * f, a = -h * rho * Vs / m;
      let vx = V[o] + a * gx, vy = V[o + 1] + a * gy, vz = V[o + 2] + a * gz;
      const s = 1 / (1 + h * drag * rho * Vs / m);
      vx *= s; vy *= s; vz *= s;
      V[o] = vx; V[o + 1] = vy; V[o + 2] = vz;
    }
    this.stats.particles = count;
    if (this.timing) this.stats.ms += performance.now() - t0;
  }

  /** Per step: advance the fluid clock. */
  endStep(world, dt) { if (this.advanceTime) this.fluid.time += dt; }

  /** Options, fluid parameters (heightAt only when it is a number), time, registration, per-body volumes and wake state. */
  serialize() {
    const fl = this.fluid, n = this._reg.length;
    let last = 0;
    for (let i = 0; i < n; i++) if (this._reg[i] !== 0 || this._vref[i] !== 0 || this._vsub[i] !== 0 || this._hc[i] !== 0) last = i + 1;
    return {
      name: this.name, bodies: this.bodies, particles: this.particles, samples: this.samples, mask: this.mask,
      advanceTime: this.advanceTime,
      fluid: { heightAt: typeof fl.heightAt === 'number' ? fl.heightAt : null, density: fl.density, drag: fl.drag, angularDrag: fl.angularDrag, time: fl.time },
      reg: this._reg.slice(0, last), vref: this._vref.slice(0, last), vsub: this._vsub.slice(0, last), hc: this._hc.slice(0, last),
    };
  }

  /** Restore serialize()'s data (typed arrays, plain arrays or their JSON form). */
  restore(d) {
    if (!d) return;
    for (const k of ['bodies', 'particles', 'samples', 'mask', 'advanceTime']) if (d[k] !== undefined) this[k] = d[k];
    const f = d.fluid;
    if (f) {
      if (typeof f.heightAt === 'number') this.fluid.heightAt = f.heightAt;
      for (const k of ['density', 'drag', 'angularDrag', 'time']) if (f[k] !== undefined) this.fluid[k] = f[k];
    }
    const reg = asArray(d.reg), vref = asArray(d.vref), vsub = asArray(d.vsub), hc = asArray(d.hc);
    this._ensureBodies(reg.length);
    this._reg.fill(0); this._vref.fill(0); this._vsub.fill(0); this._hc.fill(0); this._wakeReq.fill(0);
    for (let i = 0; i < reg.length; i++) { this._reg[i] = reg[i]; this._vref[i] = vref[i] || 0; this._vsub[i] = vsub[i] || 0; this._hc[i] = hc[i] || 0; }
    this._set.fill(null); this._setShape.fill(null); // rebuilt for the restored world's shapes and `samples`
  }
}
