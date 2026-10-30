// Pre-fractured rigid bodies (pure JS, deterministic, Node-safe, no THREE objects at runtime). A box or convex shape is
// cut ahead of time into convex pieces by seeded random planes; the pieces become ordinary dynamic bodies glued by
// breakable FIXED joints (joints.js). Nothing here runs in the step: the joint solver breaks a joint itself when its
// force |lambda| / h^2 exceeds breakForce (joints.js, end of solveJointsPositions) and rebuildConnected() then lets the
// two freed pieces collide. This module only builds the pieces / joints and reports what broke.
//
//   const pieces = fracturePieces(box(2, 1, 0.25), { pieces: 40, seed: 7 });      // construction time, allocates
//   const fb = new FracturedBody(world, { pieces, position: [0, 1, 0], density: 600 });
//   per fixed step, after world.step(dt):  if (fb.update() > 0) for (let i = 0; i < fb.justBrokenCount; i++) {
//     const k = fb.justBroken[i]; // glue joint index: pieces fb.pairs[2k], fb.pairs[2k + 1], anchor fb.anchors[3k..]
//   }
//   fb.breakNear(x, y, z, 0.5);  // an explosion: unglue everything within 0.5 m (world point)
//
// API
//   splitConvex(points, nx, ny, nz, d) -> { below, above, section }   (pure, deterministic, allocates)
//       points: Float64Array / number[] (x y z triples) or {x, y, z}[]: a convex hull's vertices, any order (>= 4, not
//       coplanar). The hull's topology comes from shapes.js convex(points, { hull: false }), so its EDGES are the
//       polygon edges of the merged faces (no triangle diagonals). Plane n . x + d = 0 (n need not be unit). below /
//       above: Float64Array(3k) of the vertices on that side (n . x + d <= 0 / >= 0; a vertex within 1e-10 x (1 + max
//       |coordinate|) of the plane goes to both) followed by the edge-plane intersection points, in edge order (edges sorted),
//       deduplicated within 1e-9 m. A side with no vertex strictly beyond the plane is empty (length 0). section: the
//       points on the plane (the cut polygon's corners, unordered).
//       Example: the unit cube (+-0.5) cut by x = 0 gives 8 points per side (4 corners + 4 edge crossings) and two
//       0.5 m^3 halves.
//   fracturePieces(shape, { pieces = 8, seed = 1, impact = null, minVolume = 1e-4, minFraction = 0.1, minArea = 1e-4 })
//       -> pieces[]
//       shape: a box() or convex() (anything with `points` and convex topology). pieces[i] = { shape (convex(),
//       re-centred), offset: Float64Array(3) (the piece's centre of mass in the ORIGINAL shape's frame = where its body
//       sits relative to the original body; note: shape.offset is the convex() re-centring, i.e. -offset), points:
//       Float64Array (the piece's hull vertices in the original frame, exact: shape.points[k] = points[k] +
//       shape.offset), volume, surface: Uint8Array(faceCount) (1: the face lies on the original shape's boundary,
//       0: a cut face; renderers give cut faces the interior material) }. The array also carries the glue graph:
//         pairs    Int32Array(2k)    pieces glued by joint k, ascending (a < b; sorted by a, then b)
//         anchors  Float64Array(3k)  centroid of their shared face (original frame): the joint anchor
//         areas    Float64Array(k)   area of the shared face, m^2 (> minArea)
//         normals  Float64Array(3k)  unit normal of the shared face, pointing from piece a into piece b
//         cuts     Float64Array(6m)  the m successful cut planes: point x y z, unit normal x y z (debugging)
//       Algorithm: start with the shape's hull points (a box: its 8 corners); repeatedly pick a piece and split it by a
//       plane through a seeded random point with a seeded random normal (unit vector by rejection from the cube, no
//       trig), until `pieces` pieces exist or nothing can be split. Without `impact` the piece is the largest by volume
//       (ties: the oldest) and the point is uniform in its AABB, rejected until inside its hull (planes test; after 64
//       rejections a random point between its centroid and a random vertex). With impact ([x, y, z], shape frame) the
//       point is uniform in the ball of radius 0.35 x the shape's bounding radius around it, rejected until inside some
//       piece, and THAT piece is split, so small pieces cluster around the impact; when 8 such tries fail (the pieces
//       there are already at minVolume) the round falls back to the largest piece. A split is rejected (and retried
//       with a new point and plane, up to 8 tries per piece, then the piece is given up) when a fragment has fewer than
//       4 non-coplanar points (convex() throws), a volume under max(minVolume, minFraction x the parent's volume,
//       minFraction x the shape's volume / pieces), or the cut face an area under minArea. The two minFraction terms
//       bound the mass ratio across the glue (the joint solver is exact up to ~300): the parent term when the largest
//       piece is split (<= ~1 / minFraction; 1487 for one 40-piece wall seed without it), the global floor when
//       `impact` splits the hit region again and again (1402 over 40 seeds without it; <= 300 on the test's 12 walls).
//       Pass minFraction 0 for the plain minVolume rule. Contacts of minArea (1 cm^2) or less get no glue: such
//       slivers just touch and collide (the rigid solve would route real load through a sliver joint).
//       The parent is retired and its two fragments are appended (below the plane first), so pieces are ordered by
//       creation and the result is a pure function of (shape, options).
//   new FracturedBody(world, { shape, position = [0, 0, 0], quaternion = [0, 0, 0, 1], pieces = 8 | Array (a
//       fracturePieces() result: it must carry pairs / anchors), seed, impact, minVolume, minFraction (passed on to
//       fracturePieces), mass | density = 1000, friction, restitution, linearDamping, angularDamping, layer, mask,
//       kind = 'rigid', breakForce = FRACTURE_BREAK_FORCE (N), breakStress = 0 (Pa), breakTorque = Infinity,
//       compliance = FRACTURE_COMPLIANCE (5e-7 m/N; 0 = rigid glue), collideConnected = false, userData })
//       One dynamic body per piece at position + quaternion * offset (orientation = quaternion), each with `density`
//       (or mass x its volume share), so the glued assembly has the original mass and centre of mass; one FIXED joint
//       per glue pair, anchored at the shared face's centroid (so the joint force is the force through that face), with
//       breakForce + breakStress x the face area (an additive term: area alone would make slivers paper-thin while the
//       rigid solve still routes load through them; a pieces list without areas gets the plain breakForce).
//       userData is the SAME object on every piece body (fb.pieceOf(id) gives the piece index). Each FracturedBody owns
//       its piece shapes: a pieces list already used by another one is copied (bit-identical shapes), because the shape
//       is how a body id is recognised as still being our piece after ids are reused.
//       Fields: world, ids: Int32Array (piece body ids), joints: Int32Array (glue joint ids), pieces, pairs, anchors
//       (original frame), justBroken: Int32Array (glue joint indices k), justBrokenCount, brokenForce / brokenTorque:
//       Float64Array (J.force / J.torque of joint k when update() reported it), disposed.
//       JointStore capacity: a 40-piece wall makes ~140-170 glue joints (maxJoints defaults to 1024): size maxJoints for
//       the walls alive at once; update() returns the slots of broken glue.
//   fb.intact (getter: no glue joint broken), fb.brokenCount (getter: glue joints broken or gone)
//   fb.update() -> the number of glue joints that broke since the last update() (scans ascending; their indices k in
//       fb.justBroken[0 .. justBrokenCount)). Each break is reported exactly once, whoever broke it (the solver,
//       breakNear / breakAll, or a removeBody of a piece). A reported joint is then removed from the world (world.
//       removeJoint: its JointStore slot is freed, J.force / J.broken of that id are no longer ours; read
//       fb.brokenForce[k] instead). A removed glue joint still counts as broken.
//   fb.breakNear(x, y, z, radius) -> count   breaks every glued joint whose anchor (world, carried by body a's pose)
//       lies within radius of the world point; fb.breakAll() -> count. Both go through the JointStore's public fields
//       (joints.broken[id] = 1, then joints.rebuildConnected() once, as joints.js documents for hand edits) and wake the
//       two pieces, so a piece that only hung on its glue falls. Allocation-free.
//   fb.impactBreak(contacts = world.contacts, { impulse = 100, radius = 0.5, statics = false }) -> count   after a step:
//       every contact of a piece with another body (not a piece of this body; the static world only with statics: a
//       hit rocks the body on its support and those contacts spike) whose normal impulse over the step (lambdaN, N s)
//       exceeds `impulse` breakNear()s its contact point with `radius`: a deterministic game-side rule (e.g. with
//       unbreakable or very strong glue) instead of, or on top of, the solver's force threshold. Allocation-free.
//   fb.impactPlugin({ speed = 4, radius = 0.5, statics = false, name = 'fracture-impact' }) -> plugin   for
//       world.addParticleSolver(plugin): breaks the glue BEFORE the contact solve, at contacts of a piece with another
//       body whose relative normal approach speed exceeds `speed` (m/s), so the struck pieces are free when the impulse
//       lands; the pieces whose glue changed go to world.recollidePairs so they push each other in that same step.
//       Measured (the 30 kg / 20 m/s sphere into the test wall, seeds 1-8, 1 s settle, default glue): a piece thrown
//       > 0.3 m in 6 of 8 walls (1-3 pieces), 90-93 % of the mass still glued in 8 of 8, 0-7 joints broken beyond 1 m;
//       the sphere still rebounds at 0.5-3.2 m/s off its ~0.12 m first-contact penetration (no CCD). Without
//       recollidePairs: 7 of 8 throw but 39-88 % stays glued (freed pieces overlap for a step, then scatter).
//   fb.serialize() -> plain snapshot (typed arrays); FracturedBody.from(world, snapshot) rebinds it to a world restored
//       by DynamicsWorld.from (body and joint ids survive a world round trip; shapes are taken from the world and checked:
//       RangeError when a live body is not a convex of its piece's volume). areas / normals / cuts are not restored.
//   fb.dispose()   removes every piece body (their joints go with them).
//
// Glue graph (how adjacency is decided): it is tracked exactly while cutting, as contact POLYGONS. Splitting piece P by
// plane Q makes the two fragments adjacent across Q with the cut polygon P n Q as their shared face (the on-plane points
// sorted by a trig-free pseudo-angle about their mean). Every earlier contact polygon C of P with a neighbour N (both
// on their separating plane) is clipped by Q (Sutherland-Hodgman): the part below Q becomes the contact of the lower
// fragment with N, the part above that of the upper fragment, and a part with area <= minArea is dropped. Pieces of a
// recursive split are separated by the plane of their last common cut, so P n N lies in one plane and this is exact
// (no tolerance on coplanarity or on "touching"); the anchor is the polygon's area centroid.
//
// Glue defaults: FRACTURE_BREAK_FORCE = 15 kN and FRACTURE_COMPLIANCE = 5e-7 m/N, tuned on test/dyn-fracture.mjs's wall
// (40 pieces of a 4 x 2 x 0.5 m slab, 2400 kg at density 600, 1/120 s steps x 4 substeps, positionIterations 1, the
// joints.js lock-row bound). Glued pieces do not touch, so at rest the joints carry the pieces above them. compliance
// 0 is rigid glue: the rigid contact delivers a hit (a 30 kg sphere at 20 m/s: ~600 N s) within one substep and the
// rigid glue spreads it through the whole assembly in that substep, so the glue either absorbs it everywhere (strong)
// or breaks everywhere (weak). Compliant glue lets the struck piece deflect first and its own joints reach breakForce
// over a few substeps. Sweep, seeds 1-8 (rest: max sag of any hull vertex after 3 s, wall holds, two stacked walls
// hold; the hit: joints broken, broken beyond 1 m of the impact, largest glued component, seeds that throw a piece
// > 0.3 m in 1 s):
//   breakForce compliance  sag mm     rest  2-stack | broken  far    component              thrown
//   50 kN      0           0.10-0.36  8/8   7/8     | 19-57   5-26   49-100 % (>=85: 7/8)   1/8
//   50 kN      5e-8..5e-7  0.11-1.60  8/8   8/8     | 0-12    0-1    96-100 % (8/8)         0/8
//   30 kN      0           0.10-0.36  8/8   5/8     | 40-82   12-46  44-97 % (6/8)          3/8
//   30 kN      5e-8 / 1e-7 0.11-0.67  8/8   8/8     | 8-18    0-5    96-100 % (8/8)         3/8
//   30 kN      5e-7        0.26-1.60  8/8   8/8     | 0-11    0      98-100 % (8/8)         1/8
//   20 kN      1e-7        0.12-0.67  8/8   8/8     | 14-34   0-14   49-97 % (7/8)          4/8
//   20 kN      2e-7        0.16-0.91  8/8   8/8     | 12-27   0-8    52-98 % (7/8)          5/8
//   20 kN      5e-7        0.26-1.60  8/8   8/8     | 9-18    0-2    92-100 % (8/8)         2/8
//   17.5 kN    5e-7        0.26-1.60  8/8   8/8     | 11-24   0-6    91-98 % (8/8)          3/8
//   15 kN      1e-7        0.12-0.67  8/8   8/8     | 22-63   0-29   41-94 % (3/8)          6/8
//   15 kN      5e-7        0.26-1.60  8/8   8/8     | 13-34   0-10   49-97 % (7/8)          6/8   <- default
//   12.5 kN    5e-7        0.26-1.60  8/8   8/8     | 21-50   0-14   46-94 % (7/8)          5/8
//   10 kN      2e-7        0.16-0.91  8/8   8/8     | 32-85   5-29   36-94 % (2/8)          7/8
// No setting throws in >= 7 of 8 walls while keeping >= 85 % glued and sagging < 2 mm: the default throws in 6 of 8 and
// seed 6 (a wall with one weak central cut) keeps 49 %. Compliance also smooths the redundant loop stresses: rigid,
// the lower of two stacked walls peaks at 19.7-50.3 kN, compliant ones hold 8 of 8 even at 15 kN. The responses are
// chaotic in the seed (a piece is thrown or not), so treat the table as tendencies. Forces are per substep (|lambda| /
// h^2): impacts read higher at smaller h, and rest loads grow with the mass the glue holds up (a rigid 1 m^3 crate of
// 1000 kg rests at 0.8-6 kN over 8 / 40 pieces): scale breakForce with the structure. impactBreak() is the
// deterministic game-side alternative (unbreakable or very strong glue, broken around hard contacts).

// Determinism: mulberry32(seed) only, +-*/ and Math.sqrt (three's ConvexHull inside convex() likewise), ascending
// loops, no Set / Map iteration, stable sorts on numbers. Allocation: fracturePieces, the constructor, serialize / from
// and dispose allocate; update(), breakNear(), breakAll() and the getters do not.
// Limits: convex pieces only (concave shapes, compounds, spheres and capsules are not cut); no sub-fracture at runtime
// (a piece never splits again); pieces live until dispose() (no debris lifetime / sleeping cleanup); glue strength is
// breakForce + breakStress x area per joint, not a fracture criterion; needs an in-thread DynamicsWorld (it reads
// world.joints / world.bodies).
import { convex } from './shapes.js';
import { SHAPE } from './state.js';
import { JOINT } from './joints.js';
import { describeShape, shapeFromDescriptor } from './DynamicsWorld.js';
import { mulberry32 } from '../../core/mathDet.js';

/** Default glue strength (N) of FracturedBody (see the header for how it was tuned). */
export const FRACTURE_BREAK_FORCE = 15000;
/** Default glue compliance (m/N) of FracturedBody: 0 = rigid glue, which spreads an impact through the whole assembly. */
export const FRACTURE_COMPLIANCE = 5e-7;

const NO_OPTS = Object.freeze({});
const OWNED = Symbol('fracture.owned'); // a pieces list already handed to a FracturedBody (its shapes are body identities)

const DEDUP = 1e-9;          // points closer than this (m) are one point
const SPLIT_TRIES = 8;       // planes tried on one piece before it is given up
const SAMPLE_TRIES = 64;     // rejection samples for a point inside a piece / the impact ball
const IMPACT_RADIUS = 0.35;  // impact ball radius, x the shape's bounding radius
const SIDE_EPS = 1e-10;      // on-plane tolerance, x the point set's size
const INSIDE_EPS = 1e-9;     // a sample point must lie this far (x size) inside every face plane
const SURFACE_COS = 1 - 1e-9, SURFACE_GAP = 1e-7; // a piece face on an original face plane (gap x (1 + bounding radius))
const ZERO3 = [0, 0, 0], ID4 = [0, 0, 0, 1];

// ---- small helpers (construction time) ------------------------------------------------------------------------------
function toF64(points) {
  if (points instanceof Float64Array) return points;
  if (points.length && typeof points[0] === 'object') {
    const out = new Float64Array(points.length * 3);
    for (let i = 0; i < points.length; i++) { out[i * 3] = points[i].x; out[i * 3 + 1] = points[i].y; out[i * 3 + 2] = points[i].z; }
    return out;
  }
  return Float64Array.from(points);
}
const vx = (v) => (v == null ? 0 : typeof v.x === 'number' ? v.x : +v[0]);
const vy = (v) => (v == null ? 0 : typeof v.y === 'number' ? v.y : +v[1]);
const vz = (v) => (v == null ? 0 : typeof v.z === 'number' ? v.z : +v[2]);
const vw = (v) => (v == null ? 1 : typeof v.w === 'number' ? v.w : +v[3]);

function sizeOf(p) {
  let m = 0;
  for (let i = 0; i < p.length; i++) { const a = p[i] < 0 ? -p[i] : p[i]; if (a > m) m = a; }
  return m;
}
// push x y z onto the number array unless a point within DEDUP is already there
function pushUnique(arr, x, y, z) {
  for (let i = 0; i < arr.length; i += 3) {
    const dx = arr[i] - x, dy = arr[i + 1] - y, dz = arr[i + 2] - z;
    if (dx * dx + dy * dy + dz * dz <= DEDUP * DEDUP) return;
  }
  arr.push(x, y, z);
}

/**
 * Classify hull vertices p (edges e: vertex index pairs) against the unit-normal plane n . x + d = 0 into the number
 * arrays below / above (vertices, then edge crossings) and section (the on-plane points). A side with no vertex
 * strictly beyond eps stays empty.
 */
function clipHull(p, e, nx, ny, nz, d, eps, below, above, section) {
  const n = p.length / 3, s = new Float64Array(n);
  let nb = 0, na = 0;
  for (let i = 0; i < n; i++) {
    const v = nx * p[i * 3] + ny * p[i * 3 + 1] + nz * p[i * 3 + 2] + d;
    s[i] = v;
    if (v < -eps) nb++; else if (v > eps) na++;
  }
  for (let i = 0; i < n; i++) {
    const v = s[i], x = p[i * 3], y = p[i * 3 + 1], z = p[i * 3 + 2];
    if (v <= eps && nb > 0) pushUnique(below, x, y, z);
    if (v >= -eps && na > 0) pushUnique(above, x, y, z);
    if (v >= -eps && v <= eps) pushUnique(section, x, y, z);
  }
  if (nb === 0 || na === 0) return;
  for (let k = 0; k < e.length; k += 2) {
    const i = e[k], j = e[k + 1], si = s[i], sj = s[j];
    if ((si < -eps && sj > eps) || (si > eps && sj < -eps)) {
      const t = si / (si - sj), i3 = i * 3, j3 = j * 3;
      const x = p[i3] + t * (p[j3] - p[i3]), y = p[i3 + 1] + t * (p[j3 + 1] - p[i3 + 1]), z = p[i3 + 2] + t * (p[j3 + 2] - p[i3 + 2]);
      pushUnique(below, x, y, z); pushUnique(above, x, y, z); pushUnique(section, x, y, z);
    }
  }
}

/**
 * Split the hull of `points` by the plane n . x + d = 0 (see the header). Pure and deterministic; allocates.
 * @param {Float64Array|number[]|{x:number,y:number,z:number}[]} points
 * @returns {{ below: Float64Array, above: Float64Array, section: Float64Array }}
 */
export function splitConvex(points, nx, ny, nz, d) {
  const p = toF64(points);
  const l = Math.sqrt(nx * nx + ny * ny + nz * nz);
  if (!(l > 0) || !Number.isFinite(l) || !Number.isFinite(d)) throw new RangeError('splitConvex: the plane needs a finite non-zero normal');
  const t = convex(p, { hull: false }); // vertex i = point i; edges of the merged polygon faces
  const below = [], above = [], section = [];
  clipHull(p, t.edges, nx / l, ny / l, nz / l, d / l, SIDE_EPS * (1 + sizeOf(p)), below, above, section);
  return { below: Float64Array.from(below), above: Float64Array.from(above), section: Float64Array.from(section) };
}

/**
 * convex() of a point set plus its hull vertices in the input frame, exactly (convex() keeps the input order of the hull
 * vertices and stores input + offset, and a - b === a + (-b) in IEEE arithmetic). null when the set is degenerate.
 */
function hullPiece(pts) {
  let s;
  try { s = convex(pts); } catch { return null; } // < 4 points, coplanar, zero volume, non-manifold sliver
  if (!(s.volume > 0) || !Number.isFinite(s.volume)) return null;
  const sp = s.points, o = s.offset, out = new Float64Array(sp.length);
  let j = 0;
  for (let k = 0; k < sp.length; k += 3) {
    while (j < pts.length && !(pts[j] + o[0] === sp[k] && pts[j + 1] + o[1] === sp[k + 1] && pts[j + 2] + o[2] === sp[k + 2])) j += 3;
    if (j < pts.length) { out[k] = pts[j]; out[k + 1] = pts[j + 1]; out[k + 2] = pts[j + 2]; j += 3; }
    else { out[k] = sp[k] - o[0]; out[k + 1] = sp[k + 1] - o[1]; out[k + 2] = sp[k + 2] - o[2]; } // (not reached)
  }
  return { shape: s, pts: out };
}

// strictly inside the piece's hull (planes are in the re-centred frame: local = x + offset)
function insideHull(s, x, y, z, eps) {
  const pl = s.planes, o = s.offset, lx = x + o[0], ly = y + o[1], lz = z + o[2];
  for (let f = 0; f < pl.length; f += 4) if (pl[f] * lx + pl[f + 1] * ly + pl[f + 2] * lz + pl[f + 3] > -eps) return false;
  return true;
}
// uniform unit vector (rejection from the cube; no trig)
function randomUnit(rand, out) {
  for (;;) {
    const x = 2 * rand() - 1, y = 2 * rand() - 1, z = 2 * rand() - 1, l2 = x * x + y * y + z * z;
    if (l2 > 1e-4 && l2 <= 1) { const l = Math.sqrt(l2); out[0] = x / l; out[1] = y / l; out[2] = z / l; return out; }
  }
}
// uniform point in the ball (c, r)
function ballPoint(rand, cx, cy, cz, r, out) {
  for (;;) {
    const x = 2 * rand() - 1, y = 2 * rand() - 1, z = 2 * rand() - 1;
    if (x * x + y * y + z * z <= 1) { out[0] = cx + r * x; out[1] = cy + r * y; out[2] = cz + r * z; return out; }
  }
}
// uniform point of the piece's AABB, rejected until inside its hull; fallback: between its centroid and a random vertex
function pointInPiece(rand, P, eps, out) {
  const p = P.pts;
  let x0 = Infinity, y0 = Infinity, z0 = Infinity, x1 = -Infinity, y1 = -Infinity, z1 = -Infinity;
  for (let i = 0; i < p.length; i += 3) {
    if (p[i] < x0) x0 = p[i]; if (p[i] > x1) x1 = p[i];
    if (p[i + 1] < y0) y0 = p[i + 1]; if (p[i + 1] > y1) y1 = p[i + 1];
    if (p[i + 2] < z0) z0 = p[i + 2]; if (p[i + 2] > z1) z1 = p[i + 2];
  }
  for (let k = 0; k < SAMPLE_TRIES; k++) {
    const x = x0 + (x1 - x0) * rand(), y = y0 + (y1 - y0) * rand(), z = z0 + (z1 - z0) * rand();
    if (insideHull(P.shape, x, y, z, eps)) { out[0] = x; out[1] = y; out[2] = z; return out; }
  }
  const o = P.shape.offset, v = Math.min(p.length / 3 - 1, Math.floor(rand() * (p.length / 3))) * 3, t = 0.9 * rand();
  out[0] = -o[0] + t * (p[v] + o[0]); out[1] = -o[1] + t * (p[v + 1] + o[1]); out[2] = -o[2] + t * (p[v + 2] + o[2]);
  return out;
}

// ---- planar polygons (contact faces) ----------------------------------------------------------------------------------
// Order coplanar points (number array, convex position) counter-clockwise about the unit normal n: pseudo-angle about
// their mean in a plane basis (u, v = n x u), no trig. Returns a Float64Array.
function orderPolygon(pts, nx, ny, nz) {
  const k = pts.length / 3;
  let mx = 0, my = 0, mz = 0;
  for (let i = 0; i < pts.length; i += 3) { mx += pts[i]; my += pts[i + 1]; mz += pts[i + 2]; }
  mx /= k; my /= k; mz /= k;
  // u: n x (the axis least aligned with n), normalised
  const ax = nx < 0 ? -nx : nx, ay = ny < 0 ? -ny : ny, az = nz < 0 ? -nz : nz;
  let ux, uy, uz;
  if (ax <= ay && ax <= az) { ux = 0; uy = nz; uz = -ny; } else if (ay <= az) { ux = -nz; uy = 0; uz = nx; } else { ux = ny; uy = -nx; uz = 0; }
  const ul = Math.sqrt(ux * ux + uy * uy + uz * uz);
  ux /= ul; uy /= ul; uz /= ul;
  const wx = ny * uz - nz * uy, wy = nz * ux - nx * uz, wz = nx * uy - ny * ux;
  const ang = new Float64Array(k), order = [];
  for (let i = 0; i < k; i++) {
    const dx = pts[i * 3] - mx, dy = pts[i * 3 + 1] - my, dz = pts[i * 3 + 2] - mz;
    const x = dx * ux + dy * uy + dz * uz, y = dx * wx + dy * wy + dz * wz, s = (x < 0 ? -x : x) + (y < 0 ? -y : y);
    const q = s > 0 ? y / s : 0; // diamond angle in [0, 4): monotonic in the true angle
    ang[i] = y >= 0 ? (x >= 0 ? q : 2 - q) : (x < 0 ? 2 - q : 4 + q);
    order.push(i);
  }
  order.sort((a, b) => ang[a] - ang[b] || a - b);
  const out = new Float64Array(pts.length);
  for (let i = 0; i < k; i++) { const s = order[i] * 3; out[i * 3] = pts[s]; out[i * 3 + 1] = pts[s + 1]; out[i * 3 + 2] = pts[s + 2]; }
  return out;
}
// Area (about unit normal n) and area centroid of a planar polygon; writes the centroid into out, returns the area.
function polygonArea(poly, nx, ny, nz, out) {
  const k = poly.length / 3;
  if (out) { out[0] = 0; out[1] = 0; out[2] = 0; }
  if (k < 3) return 0;
  const x0 = poly[0], y0 = poly[1], z0 = poly[2];
  let area = 0, cx = 0, cy = 0, cz = 0;
  for (let i = 1; i + 1 < k; i++) {
    const ax = poly[i * 3] - x0, ay = poly[i * 3 + 1] - y0, az = poly[i * 3 + 2] - z0;
    const bx = poly[i * 3 + 3] - x0, by = poly[i * 3 + 4] - y0, bz = poly[i * 3 + 5] - z0;
    const a = ((ay * bz - az * by) * nx + (az * bx - ax * bz) * ny + (ax * by - ay * bx) * nz) / 2;
    area += a; cx += a * (ax + bx) / 3; cy += a * (ay + by) / 3; cz += a * (az + bz) / 3;
  }
  if (out && area > 0) { out[0] = x0 + cx / area; out[1] = y0 + cy / area; out[2] = z0 + cz / area; }
  return area;
}
// The part of a convex polygon on one side of the plane m . x + e = 0 (below: <= 0), Sutherland-Hodgman.
function clipPolygon(poly, mx, my, mz, e, eps, below) {
  const k = poly.length / 3, out = [];
  for (let i = 0; i < k; i++) {
    const j = i + 1 < k ? i + 1 : 0, i3 = i * 3, j3 = j * 3;
    const si = mx * poly[i3] + my * poly[i3 + 1] + mz * poly[i3 + 2] + e;
    const sj = mx * poly[j3] + my * poly[j3 + 1] + mz * poly[j3 + 2] + e;
    if (below ? si <= eps : si >= -eps) out.push(poly[i3], poly[i3 + 1], poly[i3 + 2]);
    if ((si < -eps && sj > eps) || (si > eps && sj < -eps)) {
      const t = si / (si - sj);
      out.push(poly[i3] + t * (poly[j3] - poly[i3]), poly[i3 + 1] + t * (poly[j3 + 1] - poly[i3 + 1]), poly[i3 + 2] + t * (poly[j3 + 2] - poly[i3 + 2]));
    }
  }
  return Float64Array.from(out);
}

// ---- fracturing -------------------------------------------------------------------------------------------------------
// Split node `idx` of the state by the plane through (px, py, pz) with unit normal n; true when committed.
function trySplit(st, idx, px, py, pz, nx, ny, nz) {
  const P = st.nodes[idx];
  const d = -(nx * px + ny * py + nz * pz);
  const below = [], above = [], section = [];
  clipHull(P.pts, P.shape.edges, nx, ny, nz, d, st.eps, below, above, section);
  if (below.length < 12 || above.length < 12 || section.length < 9) return false;
  const L = hullPiece(Float64Array.from(below));
  const vmin = Math.max(st.minVolume, st.minFraction * P.shape.volume, st.floor);
  if (!L || !(L.shape.volume >= vmin)) return false;
  const U = hullPiece(Float64Array.from(above));
  if (!U || !(U.shape.volume >= vmin)) return false;
  const cut = orderPolygon(section, nx, ny, nz);
  if (!(polygonArea(cut, nx, ny, nz, null) > st.minArea)) return false;
  // commit: retire P, append its fragments, hand P's contacts to the fragment(s) on their side of the plane
  P.alive = false;
  const li = st.nodes.length, ui = li + 1;
  st.nodes.push({ pts: L.pts, shape: L.shape, alive: true, stuck: false }, { pts: U.pts, shape: U.shape, alive: true, stuck: false });
  const next = [];
  for (let i = 0; i < st.edges.length; i++) {
    const E = st.edges[i];
    if (E.a !== idx && E.b !== idx) { next.push(E); continue; }
    const lo = clipPolygon(E.poly, nx, ny, nz, d, st.eps, true), hi = clipPolygon(E.poly, nx, ny, nz, d, st.eps, false);
    if (polygonArea(lo, E.nx, E.ny, E.nz, null) > st.minArea) next.push({ a: E.a === idx ? li : E.a, b: E.b === idx ? li : E.b, poly: lo, nx: E.nx, ny: E.ny, nz: E.nz });
    if (polygonArea(hi, E.nx, E.ny, E.nz, null) > st.minArea) next.push({ a: E.a === idx ? ui : E.a, b: E.b === idx ? ui : E.b, poly: hi, nx: E.nx, ny: E.ny, nz: E.nz });
  }
  next.push({ a: li, b: ui, poly: cut, nx, ny, nz }); // normal from the lower fragment into the upper one
  st.edges = next;
  st.alive++;
  st.cuts.push(px, py, pz, nx, ny, nz);
  return true;
}

// one round around the impact: split the piece containing a random point of the impact ball
function impactRound(st, rand, pt, nrm) {
  const ip = st.impact, r = st.impactRadius;
  for (let t = 0; t < SPLIT_TRIES; t++) {
    let idx = -1;
    for (let k = 0; k < SAMPLE_TRIES && idx < 0; k++) {
      ballPoint(rand, ip[0], ip[1], ip[2], r, pt);
      for (let i = 0; i < st.nodes.length; i++) {
        const N = st.nodes[i];
        if (N.alive && insideHull(N.shape, pt[0], pt[1], pt[2], st.insideEps)) { idx = i; break; }
      }
    }
    if (idx < 0) return false; // the ball misses the shape
    randomUnit(rand, nrm);
    if (st.nodes[idx].shape.volume >= 2 * st.minVolume && trySplit(st, idx, pt[0], pt[1], pt[2], nrm[0], nrm[1], nrm[2])) return true;
  }
  return false;
}

/**
 * Cut a box / convex shape into convex pieces (see the header for the algorithm, the fields and the glue graph).
 * @param {object} shape  shapes.js box() or convex()
 * @param {{ pieces?: number, seed?: number, impact?: number[]|null, minVolume?: number, minFraction?: number, minArea?: number }} [o]
 * @returns {Array<{ shape: object, offset: Float64Array, points: Float64Array, volume: number, surface: Uint8Array }>
 *   & { pairs: Int32Array, anchors: Float64Array, areas: Float64Array, normals: Float64Array, cuts: Float64Array }}
 */
export function fracturePieces(shape, o = {}) {
  const { pieces = 8, seed = 1, impact = null, minVolume = 1e-4, minFraction = 0.1, minArea = 1e-4 } = o;
  if (!shape || !shape.points || !shape.edges || (shape.type !== SHAPE.BOX && shape.type !== SHAPE.CONVEX)) {
    throw new TypeError('fracturePieces: a box() or convex() shape is required');
  }
  const target = Math.max(1, Math.floor(pieces));
  const rand = mulberry32(seed);
  const size = sizeOf(shape.points);
  const root = hullPiece(Float64Array.from(shape.points));
  if (!root) throw new RangeError('fracturePieces: degenerate shape');
  const st = {
    nodes: [{ pts: root.pts, shape: root.shape, alive: true, stuck: false }], edges: [], alive: 1, cuts: [],
    eps: SIDE_EPS * (1 + size), insideEps: INSIDE_EPS * (1 + size), minVolume, minFraction, minArea,
    floor: minFraction * root.shape.volume / target, // global volume floor: bounds the mass ratio with `impact` too
    impact: impact ? [vx(impact), vy(impact), vz(impact)] : null, impactRadius: IMPACT_RADIUS * shape.boundingRadius,
  };
  const pt = new Float64Array(3), nrm = new Float64Array(3);
  while (st.alive < target) {
    if (st.impact && impactRound(st, rand, pt, nrm)) continue;
    // the largest piece that can still be split (ties: the oldest)
    let idx = -1, best = -Infinity;
    for (let i = 0; i < st.nodes.length; i++) {
      const N = st.nodes[i];
      if (!N.alive || N.stuck) continue;
      if (!(N.shape.volume >= 2 * minVolume)) { N.stuck = true; continue; }
      if (N.shape.volume > best) { best = N.shape.volume; idx = i; }
    }
    if (idx < 0) break;
    let done = false;
    for (let t = 0; t < SPLIT_TRIES && !done; t++) {
      pointInPiece(rand, st.nodes[idx], st.insideEps, pt);
      randomUnit(rand, nrm);
      done = trySplit(st, idx, pt[0], pt[1], pt[2], nrm[0], nrm[1], nrm[2]);
    }
    if (!done) st.nodes[idx].stuck = true;
  }

  // pieces in creation order
  const R = shape.boundingRadius, sp = shape.planes, gap = SURFACE_GAP * (1 + R);
  const index = new Int32Array(st.nodes.length).fill(-1);
  const out = [];
  for (let i = 0; i < st.nodes.length; i++) {
    const N = st.nodes[i];
    if (!N.alive) continue;
    index[i] = out.length;
    const s = N.shape, so = s.offset, pl = s.planes;
    const surface = new Uint8Array(s.faceCount);
    for (let f = 0; f < s.faceCount; f++) {
      const nx = pl[f * 4], ny = pl[f * 4 + 1], nz = pl[f * 4 + 2];
      const d = pl[f * 4 + 3] + nx * so[0] + ny * so[1] + nz * so[2]; // plane in the original frame
      for (let g = 0; g < sp.length; g += 4) {
        const dd = d - sp[g + 3];
        if (nx * sp[g] + ny * sp[g + 1] + nz * sp[g + 2] > SURFACE_COS && dd <= gap && dd >= -gap) { surface[f] = 1; break; }
      }
    }
    out.push({ shape: s, offset: Float64Array.of(0 - so[0], 0 - so[1], 0 - so[2]), points: N.pts, volume: s.volume, surface });
  }
  // glue pairs, ascending
  const recs = [], c = new Float64Array(3);
  for (let i = 0; i < st.edges.length; i++) {
    const E = st.edges[i];
    let a = index[E.a], b = index[E.b], sg = 1;
    if (a > b) { const t = a; a = b; b = t; sg = -1; }
    const area = polygonArea(E.poly, E.nx, E.ny, E.nz, c);
    recs.push({ a, b, area, cx: c[0], cy: c[1], cz: c[2], nx: sg * E.nx, ny: sg * E.ny, nz: sg * E.nz });
  }
  recs.sort((p, q) => p.a - q.a || p.b - q.b);
  const k = recs.length, pairs = new Int32Array(2 * k), anchors = new Float64Array(3 * k), areas = new Float64Array(k), normals = new Float64Array(3 * k);
  for (let i = 0; i < k; i++) {
    const r = recs[i];
    pairs[i * 2] = r.a; pairs[i * 2 + 1] = r.b; areas[i] = r.area;
    anchors[i * 3] = r.cx; anchors[i * 3 + 1] = r.cy; anchors[i * 3 + 2] = r.cz;
    normals[i * 3] = r.nx; normals[i * 3 + 1] = r.ny; normals[i * 3 + 2] = r.nz;
  }
  return Object.assign(out, { pairs, anchors, areas, normals, cuts: Float64Array.from(st.cuts) });
}

// A copy of a pieces list with fresh, bit-identical shapes (describeShape / shapeFromDescriptor): body identity checks
// compare shapes, so two FracturedBodies never share piece shapes.
function ownCopy(list) {
  const out = list.map((p) => ({ shape: shapeFromDescriptor(describeShape(p.shape)), offset: p.offset, points: p.points, volume: p.volume, surface: p.surface || null }));
  return Object.assign(out, { pairs: list.pairs, anchors: list.anchors, areas: list.areas, normals: list.normals, cuts: list.cuts });
}

// ---- FracturedBody ----------------------------------------------------------------------------------------------------
export class FracturedBody {
  /**
   * Glued pieces of a pre-fractured shape (see the header for the options).
   * @param {object} world  an in-thread DynamicsWorld
   * @param {object} o
   */
  constructor(world, o = {}) {
    const {
      shape = null, position = ZERO3, quaternion = ID4, pieces = 8, seed = 1, impact = null, minVolume = 1e-4, minFraction = 0.1,
      mass, density = 1000, friction, restitution, linearDamping, angularDamping, layer, mask, kind = 'rigid',
      breakForce = FRACTURE_BREAK_FORCE, breakStress = 0, breakTorque = Infinity, compliance = FRACTURE_COMPLIANCE, collideConnected = false,
      userData = null,
    } = o;
    if (!world || !world.joints || typeof world.addBody !== 'function') throw new TypeError('FracturedBody: an in-thread DynamicsWorld is required');
    let list = Array.isArray(pieces) ? pieces : fracturePieces(shape, { pieces, seed, impact, minVolume, minFraction });
    if (!list.pairs || !list.anchors) throw new TypeError('FracturedBody: pieces must be a fracturePieces() result (pairs, anchors)');
    if (list[OWNED]) list = ownCopy(list); // a second body from the same list gets its own shapes (identity checks)
    list[OWNED] = true;
    const n = list.length, k = list.pairs.length / 2;
    let qx = vx(quaternion), qy = vy(quaternion), qz = vz(quaternion), qw = vw(quaternion);
    const ql = Math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
    if (ql > 0) { qx /= ql; qy /= ql; qz /= ql; qw /= ql; } else { qx = 0; qy = 0; qz = 0; qw = 1; }
    const px = vx(position), py = vy(position), pz = vz(position);
    let total = 0;
    for (let i = 0; i < n; i++) total += list[i].volume;
    const useMass = Number.isFinite(mass) && mass > 0;
    const ids = new Int32Array(n).fill(-1), joints = new Int32Array(k).fill(-1);
    const fail = (msg) => { for (let i = 0; i < n; i++) if (ids[i] >= 0) world.removeBody(ids[i]); throw new RangeError(msg); };
    const quat = [qx, qy, qz, qw];
    for (let i = 0; i < n; i++) {
      const P = list[i], ox = P.offset[0], oy = P.offset[1], oz = P.offset[2];
      // position + q offset
      const tx = 2 * (qy * oz - qz * oy), ty = 2 * (qz * ox - qx * oz), tz = 2 * (qx * oy - qy * ox);
      const wx = px + ox + qw * tx + (qy * tz - qz * ty), wy = py + oy + qw * ty + (qz * tx - qx * tz), wz = pz + oz + qw * tz + (qx * ty - qy * tx);
      try {
        ids[i] = world.addBody({
          shape: P.shape, position: [wx, wy, wz], quaternion: quat,
          mass: useMass ? mass * P.volume / total : undefined, density, friction, restitution, linearDamping, angularDamping,
          layer, mask, kind, userData,
        });
      } catch (e) { fail(`FracturedBody: ${e.message}`); }
    }
    const pr = list.pairs, an = list.anchors, ar = list.areas;
    const removeJoints = () => { for (let j = 0; j < k; j++) if (joints[j] >= 0) world.removeJoint(joints[j]); };
    for (let j = 0; j < k; j++) {
      const a = pr[j * 2], b = pr[j * 2 + 1], oa = list[a].offset;
      let id = -1;
      try {
        id = world.addJoint({
          type: JOINT.FIXED, bodyA: ids[a], bodyB: ids[b],
          anchorA: [an[j * 3] - oa[0], an[j * 3 + 1] - oa[1], an[j * 3 + 2] - oa[2]], // the face centroid in a's frame
          breakForce: breakStress > 0 && ar ? breakForce + breakStress * ar[j] : breakForce, breakTorque, compliance, collideConnected,
        });
      } catch (e) { removeJoints(); fail(`FracturedBody: ${e.message}`); }
      if (id < 0) { removeJoints(); fail(`FracturedBody: the JointStore is full (${k} glue joints; raise maxJoints)`); }
      joints[j] = id;
    }
    this._bind(world, list, ids, joints, Int32Array.from(pr), Float64Array.from(an), new Uint8Array(k));
  }

  // every instance gets the same fields in the same order (the constructor and from() both come through here)
  _bind(world, pieces, ids, joints, pairs, anchors, seen) {
    this.world = world;
    this.pieces = pieces;
    this.ids = ids;
    this.joints = joints;
    this.pairs = pairs;
    this.anchors = anchors;
    this.justBroken = new Int32Array(joints.length);
    this.justBrokenCount = 0;
    this.brokenForce = new Float64Array(joints.length); // J.force / J.torque when update() reported the joint
    this.brokenTorque = new Float64Array(joints.length);
    this.disposed = false;
    this._seen = seen; // 1: reported by update()
    // ids sorted ascending with their piece indices (pieceOf / impactBreak: binary search)
    const order = Array.from(ids, (_, i) => i).sort((p, q) => ids[p] - ids[q] || p - q);
    this._sortedIds = Int32Array.from(order, (i) => ids[i]);
    this._sortedPiece = Int32Array.from(order);
    // impact plugin: piece bodies whose glue changed during one scan (world.recollidePairs input)
    this._collect = false;
    this._touchedIds = new Int32Array(ids.length);
    this._touchedFlag = new Uint8Array(ids.length);
    this._touchedCount = 0;
    return this;
  }

  // joint k is still the glue joint we made (alive, between our two pieces), broken or not
  _ours(k) {
    const W = this.world, J = W.joints, S = W.bodies.shape, j = this.joints[k], pa = this.pairs[k * 2], pb = this.pairs[k * 2 + 1];
    const a = this.ids[pa], b = this.ids[pb];
    return J.alive[j] === 1 && J.a[j] === a && J.b[j] === b && S[a] === this.pieces[pa].shape && S[b] === this.pieces[pb].shape;
  }

  // glue joint k still holds: alive, unbroken, still linking its two pieces, and those bodies are still OUR pieces
  // (joint and body ids are reused after removals: another body at the same id has another shape; BodyStore.free
  // nulls the shape, from() keeps the identity)
  _glued(k) {
    const W = this.world, J = W.joints, S = W.bodies.shape, j = this.joints[k], pa = this.pairs[k * 2], pb = this.pairs[k * 2 + 1];
    const a = this.ids[pa], b = this.ids[pb];
    return J.alive[j] === 1 && J.broken[j] === 0 && J.a[j] === a && J.b[j] === b &&
      S[a] === this.pieces[pa].shape && S[b] === this.pieces[pb].shape;
  }

  /** True while no glue joint is broken (or gone). */
  get intact() {
    if (this.disposed) return false;
    for (let k = 0; k < this.joints.length; k++) if (!this._glued(k)) return false;
    return true;
  }

  /** Glue joints broken (or gone) so far. */
  get brokenCount() {
    if (this.disposed) return this.joints.length;
    let n = 0;
    for (let k = 0; k < this.joints.length; k++) if (!this._glued(k)) n++;
    return n;
  }

  /**
   * Glue joints that broke since the last call (indices in justBroken[0 .. justBrokenCount)); their force / torque at
   * that moment go to brokenForce[k] / brokenTorque[k], then the broken joint is removed from the world (frees its
   * JointStore slot). Allocation-free.
   */
  update() {
    let n = 0;
    if (!this.disposed) {
      const seen = this._seen, out = this.justBroken, W = this.world, J = W.joints;
      for (let k = 0; k < this.joints.length; k++) {
        if (seen[k] === 1 || this._glued(k)) continue;
        seen[k] = 1; out[n++] = k;
        if (this._ours(k)) {
          const j = this.joints[k];
          this.brokenForce[k] = J.force[j]; this.brokenTorque[k] = J.torque[j];
          W.removeJoint(j);
        }
      }
    }
    this.justBrokenCount = n;
    return n;
  }

  /** Piece index of body `id` (-1 when it is not one of this body's live pieces). Allocation-free. */
  pieceOf(id) {
    if (this.disposed || !(id >= 0)) return -1;
    const s = this._sortedIds;
    let lo = 0, hi = s.length - 1;
    while (lo <= hi) {
      const m = (lo + hi) >> 1, v = s[m];
      if (v === id) { const i = this._sortedPiece[m]; return this.world.bodies.shape[id] === this.pieces[i].shape ? i : -1; }
      if (v < id) lo = m + 1; else hi = m - 1;
    }
    return -1;
  }

  /**
   * Game-side impact breaking (deterministic, allocation-free): after a step, every contact of `contacts` (a
   * ContactBuffer: world.contacts, whose lambdaN is the normal impulse over the step, N s) between one of this body's
   * live pieces and another body (contacts between two of its own pieces are skipped, and contacts with the static
   * world too unless `statics`: a hit makes the wall rock on its support, whose contacts then spike) whose lambdaN
   * exceeds `impulse` breaks the glue within `radius` of the contact point (breakNear). The breaks show up in the next
   * update(). Returns the number of glue joints broken. Pass a reused options object.
   * @param {object} [contacts]  default world.contacts
   * @param {{ impulse?: number, radius?: number, statics?: boolean }} [o]  impulse threshold (N s, default 100), radius
   *   (m, default 0.5), statics (default false: also break at hard contacts with the static world, e.g. a dropped body)
   */
  impactBreak(contacts, o = NO_OPTS) {
    if (this.disposed) return 0;
    const C = contacts || this.world.contacts, imp = o.impulse ?? 100, r = o.radius ?? 0.5, statics = o.statics === true;
    let n = 0;
    for (let i = 0; i < C.count; i++) {
      if (!(C.lambdaN[i] > imp)) continue;
      const a = C.a[i], b = C.b[i];
      if (!statics && (a < 0 || b < 0)) continue;
      const ours = (this.pieceOf(a) >= 0 ? 1 : 0) + (this.pieceOf(b) >= 0 ? 1 : 0);
      if (ours === 1) n += this.breakNear(C.px[i], C.py[i], C.pz[i], r);
    }
    return n;
  }

  /** Break every glued joint whose world anchor lies within `radius` of (x, y, z); returns the count. Allocation-free. */
  breakNear(x, y, z, radius) {
    if (this.disposed) return 0;
    const W = this.world, J = W.joints, B = W.bodies, pos = B.pos, qu = B.quat, an = J.anchorA, r2 = radius * radius;
    let n = 0;
    for (let k = 0; k < this.joints.length; k++) {
      if (!this._glued(k)) continue;
      const j = this.joints[k], a = J.a[j], o3 = a * 3, o4 = a * 4, j3 = j * 3;
      const qx = qu[o4], qy = qu[o4 + 1], qz = qu[o4 + 2], qw = qu[o4 + 3], lx = an[j3], ly = an[j3 + 1], lz = an[j3 + 2];
      const tx = 2 * (qy * lz - qz * ly), ty = 2 * (qz * lx - qx * lz), tz = 2 * (qx * ly - qy * lx);
      const dx = pos[o3] + lx + qw * tx + (qy * tz - qz * ty) - x;
      const dy = pos[o3 + 1] + ly + qw * ty + (qz * tx - qx * tz) - y;
      const dz = pos[o3 + 2] + lz + qw * tz + (qx * ty - qy * tx) - z;
      if (dx * dx + dy * dy + dz * dz <= r2) { this._break(k); n++; }
    }
    if (n > 0) J.rebuildConnected();
    return n;
  }

  /** Break every glued joint; returns the count. Allocation-free. */
  breakAll() {
    if (this.disposed) return 0;
    let n = 0;
    for (let k = 0; k < this.joints.length; k++) if (this._glued(k)) { this._break(k); n++; }
    if (n > 0) this.world.joints.rebuildConnected();
    return n;
  }

  // glue joint k (glued): the JointStore's public hand-edit path (joints.js: set broken, then rebuildConnected once);
  // wakes both pieces; while the impact plugin scans, records the two pieces whose glue changed
  _break(k) {
    const W = this.world, J = W.joints, j = this.joints[k];
    J.broken[j] = 1;
    W.wake(J.a[j]); W.wake(J.b[j]);
    if (this._collect) { this._touch(this.pairs[k * 2]); this._touch(this.pairs[k * 2 + 1]); }
  }
  _touch(i) {
    if (this._touchedFlag[i] === 1) return;
    this._touchedFlag[i] = 1;
    this._touchedIds[this._touchedCount++] = this.ids[i];
  }

  /**
   * A DynamicsWorld plugin that breaks the glue BEFORE the contact solve: register it with
   * world.addParticleSolver(plugin) (works with maxParticles 0). Its beginStep(world, dt) runs after the step's contact
   * setup and before the substeps, when world.contacts holds this step's contacts but no impulses yet: every contact
   * (ascending) between one of this body's live pieces and another body (the static world only with `statics`) whose
   * relative normal approach speed (v_a - v_b at the contact point, along the normal from a to b; start-of-step
   * velocities) exceeds `speed` breakNear()s the contact point with `radius`, so the struck pieces are already free when
   * the contact impulse lands (impactBreak, after the step, is too late: the hit has already been pushed into the glued
   * body). plugin.broken counts the joints it broke. Deterministic and allocation-free per step. Give each plugin its
   * own `name` when worlds with several of them are serialized (from() matches plugins by name); it has no state to
   * restore. The pieces whose glue changed are handed to world.recollidePairs(ids, count) when the world has it, so
   * their pairs (skipped by this step's joint filter) collide in this very step; without it they collide from the next
   * step on and a struck piece driven into its freed neighbours overlaps them for one step.
   * @param {{ speed?: number, radius?: number, statics?: boolean, name?: string }} [o]
   * @returns {{ name: string, speed: number, radius: number, statics: boolean, broken: number, beginStep: Function }}
   */
  impactPlugin({ speed = 4, radius = 0.5, statics = false, name = 'fracture-impact' } = {}) {
    const fb = this;
    const plugin = {
      name, speed, radius, statics, broken: 0,
      beginStep(world) { plugin.broken += fb._impactScan(world, plugin); },
    };
    return plugin;
  }

  // the plugin's scan of world.contacts (see impactPlugin)
  _impactScan(world, o) {
    if (this.disposed || world !== this.world) return 0;
    const C = world.contacts, B = world.bodies, x = B.pos, v = B.vel, w = B.angVel, speed = o.speed, r = o.radius, statics = o.statics === true;
    let n = 0;
    this._collect = true; this._touchedCount = 0;
    for (let i = 0; i < C.count; i++) {
      const a = C.a[i], b = C.b[i];
      if (!statics && (a < 0 || b < 0)) continue;
      const ia = this.pieceOf(a), ib = this.pieceOf(b);
      if ((ia >= 0 ? 1 : 0) + (ib >= 0 ? 1 : 0) !== 1) continue;
      const px = C.px[i], py = C.py[i], pz = C.pz[i];
      let rvx = 0, rvy = 0, rvz = 0;
      if (a >= 0) {
        const o3 = a * 3, rx = px - x[o3], ry = py - x[o3 + 1], rz = pz - x[o3 + 2];
        rvx += v[o3] + (w[o3 + 1] * rz - w[o3 + 2] * ry); rvy += v[o3 + 1] + (w[o3 + 2] * rx - w[o3] * rz); rvz += v[o3 + 2] + (w[o3] * ry - w[o3 + 1] * rx);
      }
      if (b >= 0) {
        const o3 = b * 3, rx = px - x[o3], ry = py - x[o3 + 1], rz = pz - x[o3 + 2];
        rvx -= v[o3] + (w[o3 + 1] * rz - w[o3 + 2] * ry); rvy -= v[o3 + 1] + (w[o3 + 2] * rx - w[o3] * rz); rvz -= v[o3 + 2] + (w[o3] * ry - w[o3 + 1] * rx);
      }
      if (rvx * C.nx[i] + rvy * C.ny[i] + rvz * C.nz[i] > speed) n += this.breakPiece(ia >= 0 ? ia : ib) + this.breakNear(px, py, pz, r);
    }
    this._collect = false;
    const t = this._touchedCount;
    if (t > 0) this._touchedFlag.fill(0);
    // the pairs among them were skipped by this step's joint filter: collide them now, so the freed pieces push each
    // other in this step's solve instead of overlapping until the next step (DynamicsWorld.recollidePairs, when present)
    if (t > 1 && typeof world.recollidePairs === 'function') world.recollidePairs(this._touchedIds, t);
    return n;
  }

  /** Break every glued joint of piece i (it falls off the body); returns the count. Allocation-free. */
  breakPiece(i) {
    if (this.disposed) return 0;
    let n = 0;
    for (let k = 0; k < this.joints.length; k++) {
      if ((this.pairs[k * 2] === i || this.pairs[k * 2 + 1] === i) && this._glued(k)) { this._break(k); n++; }
    }
    if (n > 0) this.world.joints.rebuildConnected();
    return n;
  }

  /** Plain snapshot (typed arrays) for FracturedBody.from after a DynamicsWorld serialize / from round trip. */
  serialize() {
    return {
      ids: this.ids.slice(), joints: this.joints.slice(), pairs: this.pairs.slice(), anchors: this.anchors.slice(),
      seen: this._seen.slice(), disposed: this.disposed, brokenForce: this.brokenForce.slice(), brokenTorque: this.brokenTorque.slice(),
      pieces: this.pieces.map((p) => ({ offset: p.offset.slice(), points: p.points.slice(), volume: p.volume, surface: p.surface ? p.surface.slice() : null })),
    };
  }

  /**
   * Rebind a serialize() snapshot to `world` (restored by DynamicsWorld.from from the same moment): the piece shapes are
   * the world's own (restored) shapes, checked against the snapshot (a live piece body must be a convex of the piece's
   * volume, else RangeError); a piece whose body is gone gets convex(points) (identical, owned by no body). pieces
   * carries pairs / anchors again; areas, normals and cuts are not restored.
   * @param {object} world
   * @param {object} s
   * @returns {FracturedBody}
   */
  static from(world, s) {
    const ids = Int32Array.from(s.ids), pairs = Int32Array.from(s.pairs), anchors = Float64Array.from(s.anchors), B = world.bodies;
    const pieces = s.pieces.map((p, i) => {
      const id = ids[i], live = id >= 0 && id < B.count && B.alive[id] === 1;
      const shape = live ? B.shape[id] : convex(Float64Array.from(p.points));
      if (!shape || shape.type !== SHAPE.CONVEX || !(Math.abs(shape.volume - p.volume) < 1e-9)) {
        throw new RangeError(`FracturedBody.from: body ${id} is not piece ${i} (restore the world from the same snapshot)`);
      }
      return { shape, offset: Float64Array.from(p.offset), points: Float64Array.from(p.points), volume: p.volume, surface: p.surface ? Uint8Array.from(p.surface) : null };
    });
    Object.assign(pieces, { pairs, anchors });
    pieces[OWNED] = true;
    const fb = Object.create(FracturedBody.prototype)._bind(world, pieces, ids, Int32Array.from(s.joints), pairs, anchors, Uint8Array.from(s.seen));
    if (s.brokenForce) fb.brokenForce.set(s.brokenForce);
    if (s.brokenTorque) fb.brokenTorque.set(s.brokenTorque);
    fb.disposed = !!s.disposed;
    return fb;
  }

  /** Remove every piece body from the world (their joints go with them). */
  dispose() {
    if (this.disposed) return;
    const B = this.world.bodies;
    for (let i = 0; i < this.ids.length; i++) {
      const id = this.ids[i];
      if (id >= 0 && id < B.count && B.alive[id] === 1 && B.shape[id] === this.pieces[i].shape) this.world.removeBody(id);
    }
    this.disposed = true;
    this.justBrokenCount = 0;
  }
}
