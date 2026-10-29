// Narrowphase of the dynamics world (pure JS, deterministic, Node-safe, allocation-free per call): contact manifolds
// for every pair of shapes.js shapes, appended to the shared ContactBuffer (state.js), plus a persistent contact
// cache that carries the solver's accumulated impulses (warm start) from one step to the next.
//
//   const np = new Narrowphase({ maxContacts: 16384, margin: 0.01, tableSize: 65536, timing: false, hfStep: 0.5 });
//   np.beginStep(); contacts.clear();
//   np.collide(bodies, a, b, contacts) -> n                                          body vs body
//   np.collideStaticBox(bodies, a, minX, minY, minZ, maxX, maxY, maxZ, i, contacts) -> n   body vs world AABB i
//   np.collideHeightfield(bodies, a, heightAt, contacts) -> n                        body vs heightAt(x, z, y)
//   ...the solver writes contacts.lambdaN / lambdaT...
//   np.endStep(contacts);                       remember them, keyed (pairKey, key), for the next step's collide()
//   np.stats                                    reused { pairs, contacts, gjk, epa, sat, ms, capped } (reset by
//                                               beginStep; ms only while np.timing, default false; capped = silent
//                                               capacity limits hit)
//   np.evictBody(id) -> n                       drop the cache entries of a removed body (its id may be reused)
//   np.serialize() -> { pairKey, key, lambdaN, lambdaT } (plain arrays); np.restore(s); np.dispose()
//
// Contacts (state.js conventions). The normal is unit length and ALWAYS points from body a to body b (b = WORLD_BODY
// for the static world, so a body resting on the ground gets (0, -1, 0)); the point is the midpoint of the two surface
// points; depth > 0 is penetration, depth < 0 a speculative contact (surfaces separated by less than `margin`; pairs
// farther apart emit nothing; whether to act on it is the solver's job). Up to 4 points per convex pair (a manifold),
// up to 8 per body-trimesh pair; a compound adds its children's.
// `key` is a feature id within the pair (stable while the touching features stay the same, so warm starting
// matches across steps); pairKey = pairKey(a, b), or pairKey(a, WORLD_BODY) >>> 0 = 0xFFFF0000 + a for the static
// world (no body pair can produce it); staticIndex = the box index for collideStaticBox (also hashed into the key,
// since every static box and the heightfield share the body's world pairKey), -1 otherwise. collide(b, a) mirrors
// collide(a, b) with the normals negated, but for two shapes of the same type the keys and the reference-face
// tie-break follow the argument order: keys are stable across steps only because the caller keeps the order fixed
// (DynamicsWorld always passes a < b). Contact order is a pure function of the call order and the poses. A body with
// no shape gets no contacts; two trimeshes never collide (both are static).
//
// Pair algorithms (shapes are reduced to "cores" plus a radius: sphere = point + r, capsule = segment + r, box,
// convex hull and mesh triangle = polytopes with r = 0; a pair is ordered point < segment < box < hull < triangle
// and the normal is negated when the caller's order was the other way):
//   sphere-sphere      centre distance.
//   sphere-capsule     closest point of the capsule segment to the centre.
//   capsule-capsule    segment-segment closest points (Ericson 5.1.9); near-parallel axes (sin^2 < 1e-4) with an
//                      overlapping projection give two points, the ends of the overlap.
//   sphere-box         centre in the box frame clamped to the box; a centre inside pushes out through the nearest
//                      face. Key = Voronoi region (27) or face.
//   capsule-box        exact closest points of segment and box: the squared distance from the segment point at t
//                      to the box is convex and piecewise quadratic, with breakpoints where a coordinate crosses
//                      +-half; each piece is minimised in closed form. A segment inside the box: SAT over the 3 box
//                      axes (both directions) and the 3 cross(axis, segment) axes, faces preferred. Face contacts
//                      (closest box point inside a face, or a face axis) clip the segment to the face's side planes:
//                      up to 2 points (a capsule lying on a box rests on both ends).
//   box-box            SAT over 15 axes (3 + 3 face normals, 9 edge cross products, near-parallel crosses skipped),
//                      any axis separated by more than the margin rejects. The deepest face axis (the first box's
//                      unless the second's is better by 1e-5) wins unless an edge axis beats it by 1e-3 + 5 %; a face
//                      clip that comes out empty (boxes separated corner to corner) goes to GJK instead. Face:
//                      Sutherland-Hodgman clip of the incident face (the other box's face most anti-parallel to
//                      the reference normal) against the reference face's side planes, points above the reference
//                      plane by more than the margin dropped, at most 4 kept (deepest first, then farthest-point
//                      selection: a well spread quad); point = clipped point moved halfway to the reference plane.
//                      Edge: one point, the midpoint of the closest points of the two supporting edges.
//   hull vs sphere / capsule / box / hull
//                      GJK on the cores (Minkowski difference P - Q in a frame centred at P, support mapping of each
//                      core, Voronoi-region closest point on the simplex (Ericson 5.1.5 / 5.1.6), stop when the support
//                      no longer improves |v|^2 by 1e-12 relative, or repeats a vertex, or |v| < 1e-10): separated
//                      cores give the distance and closest points (speculative when depth = rP + rQ - dist >
//                      -margin); overlapping cores go to EPA (polytope expansion from the GJK simplex, blown up to a
//                      tetrahedron by axis / perpendicular / normal supports when degenerate, faces oriented by the
//                      centroid, horizon by edge cancellation, until the support gains < 1e-10) for the normal and
//                      depth. Manifold: two polytopes whose normal is within ~1.1 deg of a face normal of either
//                      (FACE_COS) clip faces exactly as box-box (the reference face is the better aligned one, the
//                      first shape's on a tie; a box is a 6-plane hull); a capsule against a polytope face clips its
//                      segment as in capsule-box; a clip whose deepest point misses the GJK / EPA depth (the deepest
//                      feature hangs over the face's edge) and every other case give one point between the witness
//                      points. Separated-within-margin face contacts get the full clipped manifold too (negative
//                      depths), so the keys do not change when a resting pair touches and separates. Keys come from
//                      face / vertex ids (reference face, incident face, clipped vertex or edge x side-plane ids),
//                      so the same touching features give the same keys every step.
//   trimesh (static) vs any
//                      the other shape's AABB in the mesh frame (+ margin) queries the BVH (shape.queryAabb); each
//                      triangle is a one-sided 3-vertex polytope with "thickness": shapes whose centre is behind its
//                      plane are skipped, GJK gives the closest points; a contact whose normal is not against the
//                      front face (n . N > -1e-6) is dropped. When the cores overlap the triangle, EPA's minimal
//                      translation decides: out through the face (n . N <= -FACE_COS) or through the back (n . N > 0)
//                      -> a face contact pushed along N (clip, else the EPA witness point); out through the triangle's
//                      boundary (a step's riser, a ridge) -> one EPA point tagged with the edge / vertex of the witness.
//                      Face contacts clip as above (triangle = reference face); edge / vertex contacts are 1 point,
//                      keyed by their sorted mesh vertex ids (not the triangle, so warm starting survives the contact
//                      moving to the neighbour). Then the triangles' points are merged: internal-edge "ghost" contacts
//                      are dropped (an edge / vertex contact whose feature belongs to a triangle with a face contact,
//                      or whose triangle lies in the plane of one, within 2.6 deg and 1 mm: no adjacency data is
//                      needed), points within 1 mm are merged (deeper kept), and at most 8 are kept (farthest-point
//                      selection).
//   compound           recurse over the children with their world poses (bounding spheres cull child pairs); the
//                      child index is hashed into the keys.
//   static box         a box with identity rotation through the same paths (collideStaticBox).
//   heightfield        samples under the shape's lowest points: sphere / capsule end: the ground normal N0 at the
//                      centre gives the bottom point c - r N0, where height and normal N are sampled (the centre's
//                      sample is reused when that point lies straight below it); box / hull: only the vertices below
//                      the centre's tangent plane (N0 . (v - c) < 0; at most 8, deepest first), each sampled;
//                      compound: recurse. Each sample is 3 heightAt calls: one-sided differences over hfStep (0.5 m,
//                      or the hfCell option), N = normalize((h - h(x + e)) / e, 1, (h - h(z + e)) / e). One contact
//                      per sample within the margin of the local ground plane; normal -N (body into the ground).
//
// Persistent table: open addressing with linear probing on h = mix(imul(pairKey, 2654435761) ^ key) & mask (power of
// two size), entries in typed arrays (pair and feature keys Int32Array, lambdaN / lambdaT Float64Array, stamp
// Uint32Array). endStep bumps a generation counter and inserts this step's contacts with it; a slot whose stamp is not
// the current generation is empty, so everything not touched this step is evicted in O(1) (no tombstones, no
// clearing). collide() prefills lambdaN / lambdaT of a new contact from the entry with the same (pairKey, key).
//
// Determinism: only + - * / Math.sqrt, comparisons, Math.imul and integer ops; no Set / Map; loops in index order.
// Allocation: module-level scratch only (Float64Array / Int32Array and a few reused proxy records); no double is
// passed to or returned from a non-trivial function or stored in a module `let` (V8 would box it), so once optimised
// collide / collideStaticBox allocate nothing (measured: 0 B over 1e4 calls of every pair type); collideHeightfield
// passes doubles to the user's heightAt (boxed unless V8 inlines it), shape.queryAabb gets six per mesh pair, and
// performance.now() (stats.ms; `timing: false` skips it) can allocate in Node until the caller is optimised.
// Not re-entrant (module scratch): one narrowphase call at a time per thread.
import { SHAPE, WORLD_BODY, pairKey, pairA, pairB } from './state.js';

// ---- constants -----------------------------------------------------------------------------------------------------
const K_POINT = 0, K_SEG = 1, K_BOX = 2, K_HULL = 3, K_TRI = 4; // core kinds, in pair-dispatch order
const FEAT_NONE = 0, FEAT_FACE = 1, FEAT_EDGE = 2, FEAT_VERT = 3; // triangle feature of a mesh candidate

const GJK_ITERS = 64;
const GJK_REL = 1e-12;       // converged when |v|^2 - v.w <= GJK_REL |v|^2
const GJK_TINY2 = 1e-20;     // |v|^2 below this: the cores touch or overlap (EPA)
const EPA_ITERS = 64;
const EPA_TOL = 1e-10;       // converged when the support gains less than EPA_TOL (1 + depth)
const EPA_MAXV = 128, EPA_MAXF = 512, EPA_MAXE = 256;
const FACE_COS = 0.9998;     // |n . face normal| above this (about 1.1 deg): face contact, manifold by clipping
const REF_TOL = 1e-5;        // the second shape's face must be better aligned by this to become the reference face
const SAT_FACE_TOL = 1e-5;   // box-box: B's face axis wins over A's only when its separation is larger by this
const SAT_EDGE_ABS = 1e-3, SAT_EDGE_REL = 0.05; // an edge axis must beat the best face by abs + rel |separation|
const PARALLEL2 = 1e-4;      // capsule-capsule: sin^2 of the axis angle below which the axes are parallel
const CLIP_MERGE2 = 1e-12;   // clipped points closer than 1e-6 m are one point
const MESH_MERGE2 = 1e-6;    // trimesh: points closer than 1 mm are merged (the deeper one is kept)
const MESH_COPLANAR_COS = 0.999, MESH_COPLANAR_DIST = 1e-3; // trimesh: triangle planes within 2.6 deg and 1 mm are one plane
const MANIFOLD_MAX = 4, MESH_MAX = 8;
const HF_MAX = 8;            // heightfield samples per box / hull
const MAXFV = 64;            // vertices read per face
const MAXCLIP = 192;         // clipped polygon capacity
const CAND_CAP = 1024;       // candidate contacts per call
const TRI_CAP = 4096;        // triangles per mesh query
const POSE_SLOTS = 64;       // pose stack (compound recursion)
const VOID_CAP = 1024;       // voided mesh vertices per mesh pair

// key tags (hashed with the feature ids)
const TAG_SS = 0x5101, TAG_SC = 0x5102, TAG_CC = 0x5103, TAG_SB = 0x5104, TAG_CB = 0x5105, TAG_CBE = 0x5106;
const TAG_FACE = 0x5107, TAG_EE = 0x5108, TAG_PT = 0x5109, TAG_CF = 0x510a, TAG_HF = 0x510b, TAG_STATIC = 0x510c;
const TAG_MESH = 0x510d, TAG_CHILD = 0x510e, TAG_MFACE = 0x510f, TAG_MEDGE = 0x5110, TAG_MVERT = 0x5111, TAG_SIDE = 0x40000000;

// Box faces: f = 2 axis + (outward normal negative ? 1 : 0); corners c = (c & 1 ? +hx : -hx, c & 2 ? +hy : -hy,
// c & 4 ? +hz : -hz) (the shapes.js corner numbering), counter-clockwise seen from outside.
const BOX_FACE = new Int32Array([5, 1, 3, 7, 0, 4, 6, 2, 2, 6, 7, 3, 0, 1, 5, 4, 4, 5, 7, 6, 0, 2, 3, 1]);
// GJK tetrahedron faces (three vertices, then the opposite one)
const TET_FACE = new Int32Array([0, 1, 2, 3, 0, 2, 3, 1, 0, 3, 1, 2, 1, 3, 2, 0]);

// ---- module scratch --------------------------------------------------------------------------------------------------
const POSE = new Float64Array(16 * POSE_SLOTS); // slot: p (0..2), q (3..6), R row-major (7..15); 0 = a, 1 = b, 2 = static box
const FR = new Float64Array(15);  // work frame: R (0..8), T (9..11), origin O (12..14); world = R (x + O) + T
const CFG = new Float64Array(2);  // [0] margin of the current call, [1] heightfield finite-difference step
const CW = new Float64Array(8);   // candidate being written: point (work frame), normal (P to Q), depth
const D3 = new Float64Array(3);   // direction argument of support / bestFace / perp
const TMP = new Float64Array(12);
const SC = new Float64Array(16);  // [0] GJK early-out distance, [8] bestFace alignment
const NRM = new Float64Array(10); // pair result: normal P->Q (0..2), depth (3), core witness on P (4..6), on Q (7..9)
const FNRM = new Float64Array(4); // loadFace: world normal, offset (n . x + off = signed distance)
const CLP = new Float64Array(4);  // clipping plane n, d (keep n . x - d <= 0)
const SSI = new Float64Array(12), SSO = new Float64Array(8); // segment-segment closest points in / out
const LS = new Float64Array(6);   // capsule-box: segment start and direction in the box frame
const BT = new Float64Array(8);   // capsule-box breakpoints
const BC = new Float64Array(9), BA = new Float64Array(9), BTV = new Float64Array(3); // box-box C, |C|, t
const AB = new Float64Array(6);   // proxy AABB
const HS = new Float64Array(8);   // heightfield sample: in x z y, out h, N
const HC = new Float64Array(6);   // heightfield centre sample: h, N, x, z (reused for a sphere's bottom point)
const HQ = new Float64Array(8);   // heightfield round sample: centre, r, N0
const HV = new Float64Array(3 * HF_MAX), HD = new Float64Array(HF_MAX), HI = new Int32Array(HF_MAX);
// candidates (world space, final normal sign)
const CPX = new Float64Array(CAND_CAP), CPY = new Float64Array(CAND_CAP), CPZ = new Float64Array(CAND_CAP);
const CNX = new Float64Array(CAND_CAP), CNY = new Float64Array(CAND_CAP), CNZ = new Float64Array(CAND_CAP);
const CD = new Float64Array(CAND_CAP);
const CK = new Int32Array(CAND_CAP), CFT = new Int32Array(CAND_CAP);
const CFA = new Int32Array(CAND_CAP), CFB = new Int32Array(CAND_CAP), CFC = new Int32Array(CAND_CAP);
const CPL = new Float64Array(4 * CAND_CAP); // trimesh candidates: world plane (unit normal, offset) of their triangle
const TPL = new Float64Array(4);            // world plane of the current mesh triangle
const KEEP = new Uint8Array(CAND_CAP), MIND = new Float64Array(CAND_CAP), SELF = new Uint8Array(CAND_CAP);
const SEL = new Int32Array(16);
const VOIDV = new Int32Array(VOID_CAP);
// face clipping
const RFV = new Float64Array(3 * MAXFV), RFI = new Int32Array(MAXFV);
const CA = new Float64Array(3 * MAXCLIP), CAI = new Int32Array(MAXCLIP), CAL = new Int32Array(MAXCLIP);
const CB = new Float64Array(3 * MAXCLIP), CBI = new Int32Array(MAXCLIP), CBL = new Int32Array(MAXCLIP);
const MX = new Float64Array(MAXCLIP), MY = new Float64Array(MAXCLIP), MZ = new Float64Array(MAXCLIP);
const MS = new Float64Array(MAXCLIP), MI = new Int32Array(MAXCLIP);
// GJK
const GW = new Float64Array(12), GP = new Float64Array(12), GQ = new Float64Array(12);
const GIP = new Int32Array(4), GIQ = new Int32Array(4), GL = new Float64Array(4), GV = new Float64Array(3);
const TW = new Float64Array(8), WT = new Float64Array(4);
const GOUT = new Float64Array(8); // witness on P (0..2), on Q (3..5), distance (6)
// EPA
const EW = new Float64Array(3 * EPA_MAXV), EP = new Float64Array(3 * EPA_MAXV), EQ = new Float64Array(3 * EPA_MAXV);
const EIP = new Int32Array(EPA_MAXV), EIQ = new Int32Array(EPA_MAXV);
const FI = new Int32Array(3 * EPA_MAXF), FN = new Float64Array(4 * EPA_MAXF), FAL = new Uint8Array(EPA_MAXF);
const EDG = new Int32Array(2 * EPA_MAXE);
const EOUT = new Float64Array(10); // normal (0..2), depth (3), witness on P (4..6), on Q (7..9)
const TRI_IDX = new Int32Array(TRI_CAP);

// integer state of the current call (Smis only: no doubles in module lets)
let curNP = null, curA = 0, curB = 0, curPair = 0, curStatic = -1;
let candCount = 0, candOverflow = 0, curFlip = false, curSalt = 0, frameWorld = true, poseTop = 3;
let curFeat = FEAT_NONE, curFA = -1, curFB = -1, curFC = -1;
let gjkN = 0, ev = 0, ef = 0, ne = 0, capHits = 0, curMeshSalt = 0;

// Proxy: a shape core at a pose in the work frame. kind K_*, r radius, p position (point / centre), R rotation
// (row-major), h box half extents, s segment ends (6), t triangle vertices (9) + unit normal (9..11), ids triangle
// vertex ids (mesh indices), shape (hull: the convex shape).
function makeProxy() {
  return {
    kind: 0, r: 0.5, p: new Float64Array(3), R: new Float64Array(9), h: new Float64Array(3), s: new Float64Array(6),
    t: new Float64Array(12), ids: new Int32Array(3), shape: null,
  };
}
const PA = makeProxy(), PB = makeProxy(), PT = makeProxy();
// static world box (collideStaticBox): a pseudo shape with the BOX fields the narrowphase reads
const STATIC_BOX = { type: SHAPE.BOX, radius: 0, halfHeight: 0, half: new Float64Array(3), boundingRadius: 0.5, children: null };

// ---- small helpers ---------------------------------------------------------------------------------------------------
/** 32-bit key mixing (int32 result). */
function mixKey(h, v) { h = Math.imul(h ^ v, -1640531535); return h ^ (h >>> 16); }
/** Table slot hash of (pairKey, key) (int32; mask it). */
function hashSlot(pk, key) {
  let h = Math.imul(pk, -1640531535) ^ key;
  h ^= h >>> 15; h = Math.imul(h, 0x2c1b3c6d); return h ^ (h >>> 12);
}

// rotation matrix of the quaternion in pose slot o (offset)
function quatToMat(o) {
  const x = POSE[o + 3], y = POSE[o + 4], z = POSE[o + 5], w = POSE[o + 6];
  const x2 = x + x, y2 = y + y, z2 = z + z;
  const xx = x * x2, xy = x * y2, xz = x * z2, yy = y * y2, yz = y * z2, zz = z * z2, wx = w * x2, wy = w * y2, wz = w * z2;
  POSE[o + 7] = 1 - (yy + zz); POSE[o + 8] = xy - wz; POSE[o + 9] = xz + wy;
  POSE[o + 10] = xy + wz; POSE[o + 11] = 1 - (xx + zz); POSE[o + 12] = yz - wx;
  POSE[o + 13] = xz - wy; POSE[o + 14] = yz + wx; POSE[o + 15] = 1 - (xx + yy);
}
function poseFromBody(bodies, id, slot) {
  const o = slot * 16, p = bodies.pos, q = bodies.quat, i3 = id * 3, i4 = id * 4;
  POSE[o] = p[i3]; POSE[o + 1] = p[i3 + 1]; POSE[o + 2] = p[i3 + 2];
  POSE[o + 3] = q[i4]; POSE[o + 4] = q[i4 + 1]; POSE[o + 5] = q[i4 + 2]; POSE[o + 6] = q[i4 + 3];
  quatToMat(o);
}
// child pose = parent pose * (child.pos, child.quat)
function childPose(ps, child, cs) {
  const b = ps * 16, c = cs * 16, cp = child.pos, cq = child.quat;
  const x = cp[0], y = cp[1], z = cp[2];
  POSE[c] = POSE[b] + POSE[b + 7] * x + POSE[b + 8] * y + POSE[b + 9] * z;
  POSE[c + 1] = POSE[b + 1] + POSE[b + 10] * x + POSE[b + 11] * y + POSE[b + 12] * z;
  POSE[c + 2] = POSE[b + 2] + POSE[b + 13] * x + POSE[b + 14] * y + POSE[b + 15] * z;
  const ax = POSE[b + 3], ay = POSE[b + 4], az = POSE[b + 5], aw = POSE[b + 6];
  const bx = cq[0], by = cq[1], bz = cq[2], bw = cq[3];
  POSE[c + 3] = aw * bx + ax * bw + ay * bz - az * by;
  POSE[c + 4] = aw * by - ax * bz + ay * bw + az * bx;
  POSE[c + 5] = aw * bz + ax * by - ay * bx + az * bw;
  POSE[c + 6] = aw * bw - ax * bx - ay * by - az * bz;
  quatToMat(c);
}
// true when the bounding spheres of two posed shapes are farther apart than the margin
function apart(sA, oA, sB, oB) {
  const ra = sA.boundingRadius, rb = sB.boundingRadius;
  if (!(ra > 0) || !(rb > 0)) return false;
  const a = oA * 16, b = oB * 16;
  const dx = POSE[b] - POSE[a], dy = POSE[b + 1] - POSE[a + 1], dz = POSE[b + 2] - POSE[a + 2];
  const lim = ra + rb + CFG[0];
  return dx * dx + dy * dy + dz * dz > lim * lim;
}

// Unit vector perpendicular to D3 (cross with the least aligned axis) -> TMP[0..2]; (0, 1, 0) for D3 = 0.
function perp() {
  const x = D3[0], y = D3[1], z = D3[2];
  const ax = x < 0 ? -x : x, ay = y < 0 ? -y : y, az = z < 0 ? -z : z;
  let px, py, pz;
  if (ax <= ay && ax <= az) { px = 0; py = z; pz = -y; } else if (ay <= az) { px = -z; py = 0; pz = x; } else { px = y; py = -x; pz = 0; }
  const l = Math.sqrt(px * px + py * py + pz * pz);
  if (!(l > 1e-30)) { TMP[0] = 0; TMP[1] = 1; TMP[2] = 0; return; }
  TMP[0] = px / l; TMP[1] = py / l; TMP[2] = pz / l;
}

// Append the candidate in CW with feature key `key` (work frame -> world, flip, salt).
function pushCand(key) {
  const i = candCount;
  if (i >= CAND_CAP) { candOverflow++; return; }
  let px = CW[0] + FR[12], py = CW[1] + FR[13], pz = CW[2] + FR[14];
  let nx = CW[3], ny = CW[4], nz = CW[5];
  if (curFlip) { nx = -nx; ny = -ny; nz = -nz; }
  if (!frameWorld) {
    const wx = FR[0] * px + FR[1] * py + FR[2] * pz + FR[9];
    const wy = FR[3] * px + FR[4] * py + FR[5] * pz + FR[10];
    const wz = FR[6] * px + FR[7] * py + FR[8] * pz + FR[11];
    const mx = FR[0] * nx + FR[1] * ny + FR[2] * nz;
    const my = FR[3] * nx + FR[4] * ny + FR[5] * nz;
    const mz = FR[6] * nx + FR[7] * ny + FR[8] * nz;
    px = wx; py = wy; pz = wz; nx = mx; ny = my; nz = mz;
  }
  CPX[i] = px; CPY[i] = py; CPZ[i] = pz; CNX[i] = nx; CNY[i] = ny; CNZ[i] = nz; CD[i] = CW[6];
  CK[i] = mixKey(curSalt, key);
  CFT[i] = curFeat; CFA[i] = curFA; CFB[i] = curFB; CFC[i] = curFC;
  if (curFeat !== FEAT_NONE) { CPL[i * 4] = TPL[0]; CPL[i * 4 + 1] = TPL[1]; CPL[i * 4 + 2] = TPL[2]; CPL[i * 4 + 3] = TPL[3]; }
  candCount = i + 1;
}
function copyCand(i, j) {
  CPX[j] = CPX[i]; CPY[j] = CPY[i]; CPZ[j] = CPZ[i]; CNX[j] = CNX[i]; CNY[j] = CNY[i]; CNZ[j] = CNZ[i]; CD[j] = CD[i];
  CK[j] = CK[i]; CFT[j] = CFT[i]; CFA[j] = CFA[i]; CFB[j] = CFB[i]; CFC[j] = CFC[i];
  CPL[j * 4] = CPL[i * 4]; CPL[j * 4 + 1] = CPL[i * 4 + 1]; CPL[j * 4 + 2] = CPL[i * 4 + 2]; CPL[j * 4 + 3] = CPL[i * 4 + 3];
}

// Farthest-point reduction of points base .. base + n (X Y Z, deeper = larger D) to K: the deepest first, then
// repeatedly the point farthest from the chosen ones. Chosen offsets (ascending) in SEL; returns their count.
function reduceFPS(X, Y, Z, D, base, n, K) {
  let best = 0;
  for (let i = 1; i < n; i++) if (D[base + i] > D[base + best]) best = i;
  const bx = X[base + best], by = Y[base + best], bz = Z[base + best];
  for (let i = 0; i < n; i++) {
    const dx = X[base + i] - bx, dy = Y[base + i] - by, dz = Z[base + i] - bz;
    MIND[i] = dx * dx + dy * dy + dz * dz; SELF[i] = 0;
  }
  SELF[best] = 1; SEL[0] = best;
  let cnt = 1;
  while (cnt < K) {
    let far = -1, fd = -1;
    for (let i = 0; i < n; i++) if (SELF[i] === 0 && MIND[i] > fd) { fd = MIND[i]; far = i; }
    if (far < 0) break;
    SELF[far] = 1; SEL[cnt++] = far;
    const fx = X[base + far], fy = Y[base + far], fz = Z[base + far];
    for (let i = 0; i < n; i++) {
      const dx = X[base + i] - fx, dy = Y[base + i] - fy, dz = Z[base + i] - fz;
      const d = dx * dx + dy * dy + dz * dz;
      if (d < MIND[i]) MIND[i] = d;
    }
  }
  for (let i = 1; i < cnt; i++) { const v = SEL[i]; let j = i - 1; while (j >= 0 && SEL[j] > v) { SEL[j + 1] = SEL[j]; j--; } SEL[j + 1] = v; }
  return cnt;
}

// ---- proxies -----------------------------------------------------------------------------------------------------------
// Proxy P for `shape` at pose slot o, positions relative to the frame origin O.
function setProxy(P, shape, o) {
  const b = o * 16, p = P.p, R = P.R;
  p[0] = POSE[b] - FR[12]; p[1] = POSE[b + 1] - FR[13]; p[2] = POSE[b + 2] - FR[14];
  for (let k = 0; k < 9; k++) R[k] = POSE[b + 7 + k];
  const t = shape.type;
  P.shape = null;
  if (t === SHAPE.SPHERE) { P.kind = K_POINT; P.r = shape.radius; return; }
  if (t === SHAPE.CAPSULE) {
    P.kind = K_SEG; P.r = shape.radius;
    const hh = shape.halfHeight, ax = R[1] * hh, ay = R[4] * hh, az = R[7] * hh, s = P.s;
    s[0] = p[0] - ax; s[1] = p[1] - ay; s[2] = p[2] - az; s[3] = p[0] + ax; s[4] = p[1] + ay; s[5] = p[2] + az;
    return;
  }
  if (t === SHAPE.BOX) { P.kind = K_BOX; P.r = 0; const h = shape.half; P.h[0] = h[0]; P.h[1] = h[1]; P.h[2] = h[2]; return; }
  P.kind = K_HULL; P.r = 0; P.shape = shape;
}

// Support point of P's core along D3 -> out[o..o + 2]; returns the vertex id (box corner, hull vertex, segment end,
// triangle corner 0..2; 0 for a point).
function support(P, out, o) {
  const dx = D3[0], dy = D3[1], dz = D3[2], k = P.kind, p = P.p;
  if (k === K_POINT) { out[o] = p[0]; out[o + 1] = p[1]; out[o + 2] = p[2]; return 0; }
  if (k === K_SEG) {
    const s = P.s;
    if (dx * (s[3] - s[0]) + dy * (s[4] - s[1]) + dz * (s[5] - s[2]) > 0) { out[o] = s[3]; out[o + 1] = s[4]; out[o + 2] = s[5]; return 1; }
    out[o] = s[0]; out[o + 1] = s[1]; out[o + 2] = s[2]; return 0;
  }
  if (k === K_TRI) {
    const t = P.t;
    const d0 = dx * t[0] + dy * t[1] + dz * t[2], d1 = dx * t[3] + dy * t[4] + dz * t[5], d2 = dx * t[6] + dy * t[7] + dz * t[8];
    let i = 0;
    if (d1 > d0) i = d2 > d1 ? 2 : 1; else if (d2 > d0) i = 2;
    out[o] = t[i * 3]; out[o + 1] = t[i * 3 + 1]; out[o + 2] = t[i * 3 + 2];
    return i;
  }
  const R = P.R;
  const lx = R[0] * dx + R[3] * dy + R[6] * dz, ly = R[1] * dx + R[4] * dy + R[7] * dz, lz = R[2] * dx + R[5] * dy + R[8] * dz;
  let x, y, z, id;
  if (k === K_BOX) {
    const h = P.h;
    x = lx >= 0 ? h[0] : -h[0]; y = ly >= 0 ? h[1] : -h[1]; z = lz >= 0 ? h[2] : -h[2];
    id = (lx >= 0 ? 1 : 0) | (ly >= 0 ? 2 : 0) | (lz >= 0 ? 4 : 0);
  } else {
    const pts = P.shape.points, n = pts.length;
    let best = 0, bd = pts[0] * lx + pts[1] * ly + pts[2] * lz;
    for (let i = 3, v = 1; i < n; i += 3, v++) {
      const d = pts[i] * lx + pts[i + 1] * ly + pts[i + 2] * lz;
      if (d > bd) { bd = d; best = v; }
    }
    x = pts[best * 3]; y = pts[best * 3 + 1]; z = pts[best * 3 + 2];
    id = best;
  }
  out[o] = p[0] + R[0] * x + R[1] * y + R[2] * z;
  out[o + 1] = p[1] + R[3] * x + R[4] * y + R[5] * z;
  out[o + 2] = p[2] + R[6] * x + R[7] * y + R[8] * z;
  return id;
}

// Minkowski support of P - Q along D3 into slot k of (W, SP, SQ, IP, IQ).
function supportM(P, Q, W, SP, SQ, IP, IQ, k) {
  const o = k * 3;
  IP[k] = support(P, SP, o);
  D3[0] = -D3[0]; D3[1] = -D3[1]; D3[2] = -D3[2];
  IQ[k] = support(Q, SQ, o);
  D3[0] = -D3[0]; D3[1] = -D3[1]; D3[2] = -D3[2];
  W[o] = SP[o] - SQ[o]; W[o + 1] = SP[o + 1] - SQ[o + 1]; W[o + 2] = SP[o + 2] - SQ[o + 2];
}

// AABB of P's core plus its radius, in the work frame -> AB.
function proxyAabb(P) {
  const k = P.kind, p = P.p, r = P.r;
  if (k === K_POINT) { AB[0] = p[0] - r; AB[1] = p[1] - r; AB[2] = p[2] - r; AB[3] = p[0] + r; AB[4] = p[1] + r; AB[5] = p[2] + r; return; }
  if (k === K_SEG) {
    const s = P.s;
    for (let i = 0; i < 3; i++) {
      const a = s[i], b = s[3 + i];
      AB[i] = (a < b ? a : b) - r; AB[3 + i] = (a > b ? a : b) + r;
    }
    return;
  }
  const R = P.R;
  if (k === K_BOX) {
    const h = P.h;
    for (let i = 0; i < 3; i++) {
      const a = R[i * 3], b = R[i * 3 + 1], c = R[i * 3 + 2];
      const e = (a < 0 ? -a : a) * h[0] + (b < 0 ? -b : b) * h[1] + (c < 0 ? -c : c) * h[2];
      AB[i] = p[i] - e; AB[3 + i] = p[i] + e;
    }
    return;
  }
  const pts = P.shape.points, n = pts.length;
  for (let i = 0; i < 3; i++) {
    const a = R[i * 3], b = R[i * 3 + 1], c = R[i * 3 + 2];
    let lo = a * pts[0] + b * pts[1] + c * pts[2], hi = lo;
    for (let j = 3; j < n; j += 3) {
      const d = a * pts[j] + b * pts[j + 1] + c * pts[j + 2];
      if (d < lo) lo = d; else if (d > hi) hi = d;
    }
    AB[i] = p[i] + lo; AB[3 + i] = p[i] + hi;
  }
}

// Face of polytope P whose outward normal is most aligned with the unit direction D3; alignment -> SC[8].
function bestFace(P) {
  const dx = D3[0], dy = D3[1], dz = D3[2], k = P.kind;
  if (k === K_TRI) { const t = P.t; SC[8] = t[9] * dx + t[10] * dy + t[11] * dz; return 0; }
  const R = P.R;
  const lx = R[0] * dx + R[3] * dy + R[6] * dz, ly = R[1] * dx + R[4] * dy + R[7] * dz, lz = R[2] * dx + R[5] * dy + R[8] * dz;
  if (k === K_BOX) {
    const ax = lx < 0 ? -lx : lx, ay = ly < 0 ? -ly : ly, az = lz < 0 ? -lz : lz;
    if (ax >= ay && ax >= az) { SC[8] = ax; return lx >= 0 ? 0 : 1; }
    if (ay >= az) { SC[8] = ay; return ly >= 0 ? 2 : 3; }
    SC[8] = az; return lz >= 0 ? 4 : 5;
  }
  const pl = P.shape.planes, m = pl.length >> 2;
  let best = 0, ba = pl[0] * lx + pl[1] * ly + pl[2] * lz;
  for (let f = 1; f < m; f++) {
    const a = pl[f * 4] * lx + pl[f * 4 + 1] * ly + pl[f * 4 + 2] * lz;
    if (a > ba) { ba = a; best = f; }
  }
  SC[8] = ba;
  return best;
}

// Face f of polytope P in the work frame: vertices -> V (3 per vertex), vertex ids -> I, outward unit normal and
// offset -> FNRM (n . x + FNRM[3] = signed distance). Returns the vertex count.
function loadFace(P, f, V, I) {
  const k = P.kind;
  if (k === K_TRI) {
    const t = P.t;
    for (let i = 0; i < 9; i++) V[i] = t[i];
    I[0] = P.ids[0]; I[1] = P.ids[1]; I[2] = P.ids[2];
    FNRM[0] = t[9]; FNRM[1] = t[10]; FNRM[2] = t[11];
    FNRM[3] = -(t[9] * t[0] + t[10] * t[1] + t[11] * t[2]);
    return 3;
  }
  const p = P.p, R = P.R;
  let n = 0;
  if (k === K_BOX) {
    const h = P.h, ax = f >> 1, sg = (f & 1) !== 0 ? -1 : 1;
    FNRM[0] = sg * R[ax]; FNRM[1] = sg * R[3 + ax]; FNRM[2] = sg * R[6 + ax];
    for (let c = 0; c < 4; c++) {
      const id = BOX_FACE[f * 4 + c];
      const x = (id & 1) !== 0 ? h[0] : -h[0], y = (id & 2) !== 0 ? h[1] : -h[1], z = (id & 4) !== 0 ? h[2] : -h[2];
      V[c * 3] = p[0] + R[0] * x + R[1] * y + R[2] * z;
      V[c * 3 + 1] = p[1] + R[3] * x + R[4] * y + R[5] * z;
      V[c * 3 + 2] = p[2] + R[6] * x + R[7] * y + R[8] * z;
      I[c] = id;
    }
    n = 4;
  } else {
    const s = P.shape, pts = s.points, pl = s.planes, fs = s.faceStart, fv = s.faceVerts;
    const nx = pl[f * 4], ny = pl[f * 4 + 1], nz = pl[f * 4 + 2];
    FNRM[0] = R[0] * nx + R[1] * ny + R[2] * nz; FNRM[1] = R[3] * nx + R[4] * ny + R[5] * nz; FNRM[2] = R[6] * nx + R[7] * ny + R[8] * nz;
    if (fs[f + 1] - fs[f] > MAXFV) capHits++;
    for (let q = fs[f], e = fs[f + 1]; q < e && n < MAXFV; q++) {
      const vi = fv[q], x = pts[vi * 3], y = pts[vi * 3 + 1], z = pts[vi * 3 + 2];
      V[n * 3] = p[0] + R[0] * x + R[1] * y + R[2] * z;
      V[n * 3 + 1] = p[1] + R[3] * x + R[4] * y + R[5] * z;
      V[n * 3 + 2] = p[2] + R[6] * x + R[7] * y + R[8] * z;
      I[n] = vi;
      n++;
    }
  }
  FNRM[3] = -(FNRM[0] * V[0] + FNRM[1] * V[1] + FNRM[2] * V[2]);
  return n;
}

// ---- clipping manifolds ------------------------------------------------------------------------------------------------
// Sutherland-Hodgman step: polygon (sp, si ids, sl edge labels; n points) against the plane CLP (keep n . x <= d),
// into (dp, di, dl); returns the new count. An intersection's id hashes the cut edge's label with `label` (the
// plane's), so the same pair of features gives the same id every step.
function clipPoly(sp, si, sl, n, dp, di, dl, label) {
  const px = CLP[0], py = CLP[1], pz = CLP[2], pd = CLP[3];
  let m = 0;
  for (let k = 0; k < n; k++) {
    const k1 = k + 1 === n ? 0 : k + 1;
    const ax = sp[k * 3], ay = sp[k * 3 + 1], az = sp[k * 3 + 2], bx = sp[k1 * 3], by = sp[k1 * 3 + 1], bz = sp[k1 * 3 + 2];
    const da = px * ax + py * ay + pz * az - pd, db = px * bx + py * by + pz * bz - pd;
    if (da <= 0) {
      if (m < MAXCLIP) { dp[m * 3] = ax; dp[m * 3 + 1] = ay; dp[m * 3 + 2] = az; di[m] = si[k]; dl[m] = sl[k]; m++; } else capHits++;
      if (db > 0 && da < 0 && m < MAXCLIP) { // leaves: the next edge runs along the plane
        const t = da / (da - db);
        dp[m * 3] = ax + (bx - ax) * t; dp[m * 3 + 1] = ay + (by - ay) * t; dp[m * 3 + 2] = az + (bz - az) * t;
        di[m] = mixKey(sl[k], label); dl[m] = label; m++;
      }
    } else if (db < 0 && m < MAXCLIP) { // enters: continues along the cut edge
      const t = da / (da - db);
      dp[m * 3] = ax + (bx - ax) * t; dp[m * 3 + 1] = ay + (by - ay) * t; dp[m * 3 + 2] = az + (bz - az) * t;
      di[m] = mixKey(sl[k], label); dl[m] = sl[k]; m++;
    }
  }
  return m;
}

// Face manifold: reference face rf of R (outward normal rn) against the incident face of I (I's face most
// anti-parallel to rn). refIsP: R is the pair's first shape (normal P -> Q = rn), else the normal is -rn. Emits up
// to 4 points; returns the count.
function clipFaces(R, rf, I, refIsP) {
  const nr = loadFace(R, rf, RFV, RFI);
  const rnx = FNRM[0], rny = FNRM[1], rnz = FNRM[2], roff = FNRM[3];
  D3[0] = -rnx; D3[1] = -rny; D3[2] = -rnz;
  const inf = bestFace(I);
  let n = loadFace(I, inf, CA, CAI);
  for (let k = 0; k < n; k++) CAL[k] = CAI[k];
  let sp = CA, si = CAI, sl = CAL, dp = CB, di = CBI, dl = CBL;
  for (let j = 0; j < nr && n > 0; j++) {
    const j1 = j + 1 === nr ? 0 : j + 1;
    const ex = RFV[j1 * 3] - RFV[j * 3], ey = RFV[j1 * 3 + 1] - RFV[j * 3 + 1], ez = RFV[j1 * 3 + 2] - RFV[j * 3 + 2];
    const sx = ey * rnz - ez * rny, sy = ez * rnx - ex * rnz, sz = ex * rny - ey * rnx; // edge x n: outward side normal
    CLP[0] = sx; CLP[1] = sy; CLP[2] = sz; CLP[3] = sx * RFV[j * 3] + sy * RFV[j * 3 + 1] + sz * RFV[j * 3 + 2];
    n = clipPoly(sp, si, sl, n, dp, di, dl, TAG_SIDE | RFI[j]);
    const tp = sp, ti = si, tl = sl; sp = dp; si = di; sl = dl; dp = tp; di = ti; dl = tl;
  }
  if (n === 0) return 0;
  const margin = CFG[0];
  let m = 0;
  for (let k = 0; k < n; k++) {
    const x = sp[k * 3], y = sp[k * 3 + 1], z = sp[k * 3 + 2];
    const sep = rnx * x + rny * y + rnz * z + roff;
    if (!(sep < margin)) continue;
    let dup = false;
    for (let q = 0; q < m; q++) {
      const dx = MX[q] - x, dy = MY[q] - y, dz = MZ[q] - z;
      if (dx * dx + dy * dy + dz * dz < CLIP_MERGE2) { dup = true; if (-sep > MS[q]) MS[q] = -sep; break; }
    }
    if (dup) continue;
    MX[m] = x; MY[m] = y; MZ[m] = z; MS[m] = -sep; MI[m] = si[k]; m++;
  }
  if (m === 0) return 0;
  let cnt = m;
  if (m > MANIFOLD_MAX) cnt = reduceFPS(MX, MY, MZ, MS, 0, m, MANIFOLD_MAX);
  else for (let q = 0; q < m; q++) SEL[q] = q;
  const base = mixKey(mixKey(mixKey(TAG_FACE, refIsP ? rf : 0x10000 | rf), inf), I.kind);
  const nx = refIsP ? rnx : -rnx, ny = refIsP ? rny : -rny, nz = refIsP ? rnz : -rnz;
  for (let q = 0; q < cnt; q++) {
    const k = SEL[q], h = MS[k] * 0.5; // point moved halfway to the reference plane (sep = -depth)
    CW[0] = MX[k] + rnx * h; CW[1] = MY[k] + rny * h; CW[2] = MZ[k] + rnz * h;
    CW[3] = nx; CW[4] = ny; CW[5] = nz; CW[6] = MS[k];
    pushCand(mixKey(base, MI[k]));
  }
  return cnt;
}

// Capsule (segment proxy S) on face f of polytope Pl: the segment clipped to the face's side planes, each end within
// the margin of the face plane is a contact. segIsP: S is the pair's first shape. Returns the count (0..2).
function capsuleFace(S, Pl, f, segIsP) {
  const nr = loadFace(Pl, f, RFV, RFI);
  const fx = FNRM[0], fy = FNRM[1], fz = FNRM[2], off = FNRM[3];
  const s = S.s, r = S.r, margin = CFG[0];
  const ax = s[0], ay = s[1], az = s[2], dx = s[3] - ax, dy = s[4] - ay, dz = s[5] - az;
  let t0 = 0, t1 = 1, id0 = 0, id1 = 1;
  for (let j = 0; j < nr; j++) {
    const j1 = j + 1 === nr ? 0 : j + 1;
    const ex = RFV[j1 * 3] - RFV[j * 3], ey = RFV[j1 * 3 + 1] - RFV[j * 3 + 1], ez = RFV[j1 * 3 + 2] - RFV[j * 3 + 2];
    const sx = ey * fz - ez * fy, sy = ez * fx - ex * fz, sz = ex * fy - ey * fx;
    const sd = sx * RFV[j * 3] + sy * RFV[j * 3 + 1] + sz * RFV[j * 3 + 2];
    const d0 = sx * ax + sy * ay + sz * az - sd, d1 = sx * (ax + dx) + sy * (ay + dy) + sz * (az + dz) - sd;
    if (d0 > 0 && d1 > 0) return 0;
    if (d0 > 0) {
      const t = d0 / (d0 - d1);
      if (t > t0) { t0 = t; id0 = TAG_SIDE | RFI[j]; }
    } else if (d1 > 0) {
      const t = d0 / (d0 - d1);
      if (t < t1) { t1 = t; id1 = TAG_SIDE | RFI[j]; }
    }
  }
  if (t0 > t1) return 0;
  const sg = segIsP ? -1 : 1, base = mixKey(mixKey(TAG_CF, f), Pl.kind);
  let cnt = 0;
  for (let pass = 0; pass < 2; pass++) {
    if (pass === 1 && t1 - t0 <= 1e-9) break;
    const t = pass === 0 ? t0 : t1;
    const qx = ax + dx * t, qy = ay + dy * t, qz = az + dz * t;
    const sep = fx * qx + fy * qy + fz * qz + off - r;
    if (!(sep < margin)) continue;
    const k = r + sep * 0.5;
    CW[0] = qx - fx * k; CW[1] = qy - fy * k; CW[2] = qz - fz * k;
    CW[3] = sg * fx; CW[4] = sg * fy; CW[5] = sg * fz; CW[6] = -sep;
    pushCand(mixKey(base, pass === 0 ? id0 : id1));
    cnt++;
  }
  return cnt;
}

// ---- GJK / EPA ---------------------------------------------------------------------------------------------------------
// Closest point to the origin on segment GW[i0] GW[i1]: weights -> TW[0..1]; returns the vertex mask (1 = i0, 2 = i1).
function closestSeg(i0, i1) {
  const a = i0 * 3, b = i1 * 3;
  const ax = GW[a], ay = GW[a + 1], az = GW[a + 2];
  const ex = GW[b] - ax, ey = GW[b + 1] - ay, ez = GW[b + 2] - az;
  const ee = ex * ex + ey * ey + ez * ez;
  const t = ee > 0 ? -(ax * ex + ay * ey + az * ez) / ee : 0;
  if (!(t > 0)) { TW[0] = 1; TW[1] = 0; return 1; }
  if (t >= 1) { TW[0] = 0; TW[1] = 1; return 2; }
  TW[0] = 1 - t; TW[1] = t; return 3;
}

// Closest point to the origin on triangle GW[i0] GW[i1] GW[i2] (Ericson 5.1.5): weights -> TW[0..2], point -> TW[3..5],
// squared distance -> TW[6]; returns the vertex mask (bit k = vertex k of the triple).
function closestTri(i0, i1, i2) {
  const a = i0 * 3, b = i1 * 3, c = i2 * 3;
  const ax = GW[a], ay = GW[a + 1], az = GW[a + 2], bx = GW[b], by = GW[b + 1], bz = GW[b + 2], cx = GW[c], cy = GW[c + 1], cz = GW[c + 2];
  const abx = bx - ax, aby = by - ay, abz = bz - az, acx = cx - ax, acy = cy - ay, acz = cz - az;
  let wa = 1, wb = 0, wc = 0, mask = 1;
  const d1 = -(abx * ax + aby * ay + abz * az), d2 = -(acx * ax + acy * ay + acz * az);
  if (d1 <= 0 && d2 <= 0) { wa = 1; mask = 1; } else {
    const d3 = -(abx * bx + aby * by + abz * bz), d4 = -(acx * bx + acy * by + acz * bz);
    if (d3 >= 0 && d4 <= d3) { wa = 0; wb = 1; mask = 2; } else {
      const vc = d1 * d4 - d3 * d2;
      if (vc <= 0 && d1 >= 0 && d3 <= 0) { const v = d1 / (d1 - d3); wa = 1 - v; wb = v; mask = 3; } else {
        const d5 = -(abx * cx + aby * cy + abz * cz), d6 = -(acx * cx + acy * cy + acz * cz);
        if (d6 >= 0 && d5 <= d6) { wa = 0; wc = 1; mask = 4; } else {
          const vb = d5 * d2 - d1 * d6;
          if (vb <= 0 && d2 >= 0 && d6 <= 0) { const w = d2 / (d2 - d6); wa = 1 - w; wc = w; mask = 5; } else {
            const va = d3 * d6 - d5 * d4;
            if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
              const w = (d4 - d3) / ((d4 - d3) + (d5 - d6)); wa = 0; wb = 1 - w; wc = w; mask = 6;
            } else {
              const den = va + vb + vc;
              if (den > 0) { const v = vb / den, w = vc / den; wa = 1 - v - w; wb = v; wc = w; mask = 7; } else {
                // degenerate (collinear) triangle: best of the three edges
                let best = Infinity;
                for (let e = 0; e < 3; e++) { // edges (i0 i1), (i1 i2), (i0 i2)
                  const p = e === 1 ? i1 : i0, q = e === 0 ? i1 : i2;
                  const m2 = closestSeg(p, q);
                  const px = TW[0] * GW[p * 3] + TW[1] * GW[q * 3], py = TW[0] * GW[p * 3 + 1] + TW[1] * GW[q * 3 + 1];
                  const pz = TW[0] * GW[p * 3 + 2] + TW[1] * GW[q * 3 + 2];
                  const d = px * px + py * py + pz * pz;
                  if (d < best) {
                    best = d;
                    if (e === 0) { wa = TW[0]; wb = TW[1]; wc = 0; mask = m2; }
                    else if (e === 1) { wa = 0; wb = TW[0]; wc = TW[1]; mask = m2 << 1; }
                    else { wa = TW[0]; wb = 0; wc = TW[1]; mask = (m2 & 1) | ((m2 & 2) << 1); }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  TW[0] = wa; TW[1] = wb; TW[2] = wc;
  const px = wa * ax + wb * bx + wc * cx, py = wa * ay + wb * by + wc * cy, pz = wa * az + wb * bz + wc * cz;
  TW[3] = px; TW[4] = py; TW[5] = pz; TW[6] = px * px + py * py + pz * pz;
  return mask;
}
// Keep the simplex vertices in `mask` (bit k = slot k) with weights WT[slot]; compacts GW / GP / GQ / ids / GL.
function reduceSimplex(mask) {
  let m = 0;
  for (let k = 0; k < gjkN; k++) {
    if ((mask & (1 << k)) === 0) continue;
    if (m !== k) {
      GW[m * 3] = GW[k * 3]; GW[m * 3 + 1] = GW[k * 3 + 1]; GW[m * 3 + 2] = GW[k * 3 + 2];
      GP[m * 3] = GP[k * 3]; GP[m * 3 + 1] = GP[k * 3 + 1]; GP[m * 3 + 2] = GP[k * 3 + 2];
      GQ[m * 3] = GQ[k * 3]; GQ[m * 3 + 1] = GQ[k * 3 + 1]; GQ[m * 3 + 2] = GQ[k * 3 + 2];
      GIP[m] = GIP[k]; GIQ[m] = GIQ[k];
    }
    GL[m] = WT[k];
    m++;
  }
  gjkN = m;
}

// Closest point of the simplex (gjkN = 2..4 points) to the origin: reduces the simplex to the supporting feature,
// weights in GL, point in GV. Returns 1 when the origin is inside the tetrahedron, -1 for a degenerate tetrahedron
// (nothing changed), 0 otherwise.
function simplexSolve() {
  WT[0] = 0; WT[1] = 0; WT[2] = 0; WT[3] = 0;
  if (gjkN === 2) {
    const m = closestSeg(0, 1);
    WT[0] = TW[0]; WT[1] = TW[1];
    reduceSimplex(m);
  } else if (gjkN === 3) {
    const m = closestTri(0, 1, 2);
    WT[0] = TW[0]; WT[1] = TW[1]; WT[2] = TW[2];
    reduceSimplex(m);
  } else {
    let best = Infinity, bestMask = 0, outside = false;
    let w0 = 0, w1 = 0, w2 = 0, w3 = 0;
    for (let f = 0; f < 4; f++) {
      const i0 = TET_FACE[f * 4], i1 = TET_FACE[f * 4 + 1], i2 = TET_FACE[f * 4 + 2], i3 = TET_FACE[f * 4 + 3];
      const ax = GW[i0 * 3], ay = GW[i0 * 3 + 1], az = GW[i0 * 3 + 2];
      const bx = GW[i1 * 3] - ax, by = GW[i1 * 3 + 1] - ay, bz = GW[i1 * 3 + 2] - az;
      const cx = GW[i2 * 3] - ax, cy = GW[i2 * 3 + 1] - ay, cz = GW[i2 * 3 + 2] - az;
      const dx = GW[i3 * 3] - ax, dy = GW[i3 * 3 + 1] - ay, dz = GW[i3 * 3 + 2] - az;
      const nx = by * cz - bz * cy, ny = bz * cx - bx * cz, nz = bx * cy - by * cx;
      const sp = -(ax * nx + ay * ny + az * nz), sd = dx * nx + dy * ny + dz * nz;
      const nn = nx * nx + ny * ny + nz * nz, dd = dx * dx + dy * dy + dz * dz;
      if (sd * sd <= 1e-20 * nn * dd) return -1;
      if (sp * sd < 0) {
        outside = true;
        const lm = closestTri(i0, i1, i2);
        if (TW[6] < best) {
          best = TW[6];
          let a0 = 0, a1 = 0, a2 = 0, a3 = 0, mk = 0;
          if ((lm & 1) !== 0) { mk |= 1 << i0; if (i0 === 0) a0 = TW[0]; else if (i0 === 1) a1 = TW[0]; else if (i0 === 2) a2 = TW[0]; else a3 = TW[0]; }
          if ((lm & 2) !== 0) { mk |= 1 << i1; if (i1 === 0) a0 = TW[1]; else if (i1 === 1) a1 = TW[1]; else if (i1 === 2) a2 = TW[1]; else a3 = TW[1]; }
          if ((lm & 4) !== 0) { mk |= 1 << i2; if (i2 === 0) a0 = TW[2]; else if (i2 === 1) a1 = TW[2]; else if (i2 === 2) a2 = TW[2]; else a3 = TW[2]; }
          w0 = a0; w1 = a1; w2 = a2; w3 = a3; bestMask = mk;
        }
      }
    }
    if (!outside) return 1;
    WT[0] = w0; WT[1] = w1; WT[2] = w2; WT[3] = w3;
    reduceSimplex(bestMask);
  }
  let vx = 0, vy = 0, vz = 0;
  for (let k = 0; k < gjkN; k++) { const w = GL[k]; vx += w * GW[k * 3]; vy += w * GW[k * 3 + 1]; vz += w * GW[k * 3 + 2]; }
  GV[0] = vx; GV[1] = vy; GV[2] = vz;
  return 0;
}

// Witness points of the current simplex -> GOUT (P 0..2, Q 3..5, |pP - pQ| 6).
function gjkWitness() {
  let px = 0, py = 0, pz = 0, qx = 0, qy = 0, qz = 0;
  for (let k = 0; k < gjkN; k++) {
    const w = GL[k];
    px += w * GP[k * 3]; py += w * GP[k * 3 + 1]; pz += w * GP[k * 3 + 2];
    qx += w * GQ[k * 3]; qy += w * GQ[k * 3 + 1]; qz += w * GQ[k * 3 + 2];
  }
  GOUT[0] = px; GOUT[1] = py; GOUT[2] = pz; GOUT[3] = qx; GOUT[4] = qy; GOUT[5] = qz;
  const dx = px - qx, dy = py - qy, dz = pz - qz;
  GOUT[6] = Math.sqrt(dx * dx + dy * dy + dz * dz);
}

// GJK distance of the cores of P and Q. Returns 1 when separated (GOUT), 0 when the cores touch or overlap (simplex
// left for EPA, GOUT = last witnesses), 2 when provably farther apart than SC[0].
function gjk(P, Q) {
  curNP.stats.gjk++;
  let dx = P.p[0] - Q.p[0], dy = P.p[1] - Q.p[1], dz = P.p[2] - Q.p[2];
  if (dx * dx + dy * dy + dz * dz < 1e-24) { dx = 1; dy = 0; dz = 0; }
  D3[0] = -dx; D3[1] = -dy; D3[2] = -dz;
  supportM(P, Q, GW, GP, GQ, GIP, GIQ, 0);
  gjkN = 1; GL[0] = 1;
  let vx = GW[0], vy = GW[1], vz = GW[2];
  const thr = SC[0], thr2 = thr * thr;
  for (let iter = 0; iter < GJK_ITERS; iter++) {
    const vv = vx * vx + vy * vy + vz * vz;
    if (vv <= GJK_TINY2) { gjkWitness(); return 0; }
    D3[0] = -vx; D3[1] = -vy; D3[2] = -vz;
    const k = gjkN;
    supportM(P, Q, GW, GP, GQ, GIP, GIQ, k);
    const wv = GW[k * 3] * vx + GW[k * 3 + 1] * vy + GW[k * 3 + 2] * vz;
    if (wv > 0 && wv * wv > vv * thr2) return 2;
    if (vv - wv <= GJK_REL * vv) break;
    let dup = false;
    for (let j = 0; j < k; j++) if (GIP[j] === GIP[k] && GIQ[j] === GIQ[k]) { dup = true; break; }
    if (dup) break;
    gjkN = k + 1;
    const r = simplexSolve();
    if (r === 1) { gjkWitness(); return 0; }
    if (r < 0) { gjkN = k; break; }
    const nvv = GV[0] * GV[0] + GV[1] * GV[1] + GV[2] * GV[2];
    vx = GV[0]; vy = GV[1]; vz = GV[2];
    if (nvv >= vv) break;
  }
  gjkWitness();
  if (GOUT[6] * GOUT[6] <= GJK_TINY2) return 0;
  return 1;
}

function epaFace(f, i, j, k) {
  FI[f * 3] = i; FI[f * 3 + 1] = j; FI[f * 3 + 2] = k;
  const ax = EW[i * 3], ay = EW[i * 3 + 1], az = EW[i * 3 + 2];
  const bx = EW[j * 3] - ax, by = EW[j * 3 + 1] - ay, bz = EW[j * 3 + 2] - az;
  const cx = EW[k * 3] - ax, cy = EW[k * 3 + 1] - ay, cz = EW[k * 3 + 2] - az;
  const nx = by * cz - bz * cy, ny = bz * cx - bx * cz, nz = bx * cy - by * cx;
  const l = Math.sqrt(nx * nx + ny * ny + nz * nz);
  FAL[f] = 1;
  if (!(l > 1e-30)) { FN[f * 4] = 0; FN[f * 4 + 1] = 0; FN[f * 4 + 2] = 0; FN[f * 4 + 3] = Infinity; return; }
  const inv = 1 / l;
  FN[f * 4] = nx * inv; FN[f * 4 + 1] = ny * inv; FN[f * 4 + 2] = nz * inv;
  FN[f * 4 + 3] = (nx * ax + ny * ay + nz * az) * inv;
}
// initial tetrahedron face, oriented away from the centroid (TMP[0..2])
function epaFaceOriented(i, j, k) {
  const ax = EW[i * 3], ay = EW[i * 3 + 1], az = EW[i * 3 + 2];
  const bx = EW[j * 3] - ax, by = EW[j * 3 + 1] - ay, bz = EW[j * 3 + 2] - az;
  const cx = EW[k * 3] - ax, cy = EW[k * 3 + 1] - ay, cz = EW[k * 3 + 2] - az;
  const nx = by * cz - bz * cy, ny = bz * cx - bx * cz, nz = bx * cy - by * cx;
  if (nx * (ax - TMP[0]) + ny * (ay - TMP[1]) + nz * (az - TMP[2]) < 0) epaFace(ef++, i, k, j); else epaFace(ef++, i, j, k);
}
function epaEdge(i, j) {
  for (let e = 0; e < ne; e++) {
    if (EDG[e * 2] === j && EDG[e * 2 + 1] === i) { ne--; EDG[e * 2] = EDG[ne * 2]; EDG[e * 2 + 1] = EDG[ne * 2 + 1]; return; }
  }
  if (ne < EPA_MAXE) { EDG[ne * 2] = i; EDG[ne * 2 + 1] = j; ne++; } else capHits++;
}
function epaCopy(from, to) {
  for (let c = 0; c < 3; c++) { EW[to * 3 + c] = EW[from * 3 + c]; EP[to * 3 + c] = EP[from * 3 + c]; EQ[to * 3 + c] = EQ[from * 3 + c]; }
  EIP[to] = EIP[from]; EIQ[to] = EIQ[from];
}
function epaDist2(i, j) {
  const dx = EW[i * 3] - EW[j * 3], dy = EW[i * 3 + 1] - EW[j * 3 + 1], dz = EW[i * 3 + 2] - EW[j * 3 + 2];
  return dx * dx + dy * dy + dz * dz;
}

// EPA from the GJK simplex: penetration normal (P to Q), depth and witnesses -> EOUT. Returns 1, or 0 on failure.
function epa(P, Q) {
  curNP.stats.epa++;
  ev = gjkN;
  for (let k = 0; k < gjkN; k++) {
    for (let c = 0; c < 3; c++) { EW[k * 3 + c] = GW[k * 3 + c]; EP[k * 3 + c] = GP[k * 3 + c]; EQ[k * 3 + c] = GQ[k * 3 + c]; }
    EIP[k] = GIP[k]; EIQ[k] = GIQ[k];
  }
  // blow a degenerate simplex up to a tetrahedron
  if (ev === 1) {
    for (let a = 0; a < 6 && ev < 2; a++) {
      D3[0] = a === 0 ? 1 : a === 1 ? -1 : 0; D3[1] = a === 2 ? 1 : a === 3 ? -1 : 0; D3[2] = a === 4 ? 1 : a === 5 ? -1 : 0;
      supportM(P, Q, EW, EP, EQ, EIP, EIQ, 1);
      if (epaDist2(0, 1) > 1e-20) ev = 2;
    }
  }
  if (ev === 2) {
    const dx = EW[3] - EW[0], dy = EW[4] - EW[1], dz = EW[5] - EW[2], dd = dx * dx + dy * dy + dz * dz;
    D3[0] = dx; D3[1] = dy; D3[2] = dz;
    perp();
    const ux = TMP[0], uy = TMP[1], uz = TMP[2];
    let vx = dy * uz - dz * uy, vy = dz * ux - dx * uz, vz = dx * uy - dy * ux;
    const vl = Math.sqrt(vx * vx + vy * vy + vz * vz);
    if (vl > 0) { vx /= vl; vy /= vl; vz /= vl; }
    for (let a = 0; a < 4 && ev < 3; a++) {
      const sg = (a & 1) !== 0 ? -1 : 1;
      if (a < 2) { D3[0] = sg * ux; D3[1] = sg * uy; D3[2] = sg * uz; } else { D3[0] = sg * vx; D3[1] = sg * vy; D3[2] = sg * vz; }
      supportM(P, Q, EW, EP, EQ, EIP, EIQ, 2);
      const wx = EW[6] - EW[0], wy = EW[7] - EW[1], wz = EW[8] - EW[2];
      const cx = wy * dz - wz * dy, cy = wz * dx - wx * dz, cz = wx * dy - wy * dx;
      if (cx * cx + cy * cy + cz * cz > 1e-20 * dd) ev = 3;
    }
  }
  if (ev === 3) {
    const ax = EW[0], ay = EW[1], az = EW[2];
    const bx = EW[3] - ax, by = EW[4] - ay, bz = EW[5] - az, cx = EW[6] - ax, cy = EW[7] - ay, cz = EW[8] - az;
    let nx = by * cz - bz * cy, ny = bz * cx - bx * cz, nz = bx * cy - by * cx;
    const l = Math.sqrt(nx * nx + ny * ny + nz * nz);
    if (l > 1e-30) {
      nx /= l; ny /= l; nz /= l;
      D3[0] = nx; D3[1] = ny; D3[2] = nz;
      supportM(P, Q, EW, EP, EQ, EIP, EIQ, 3);
      const h1 = nx * (EW[9] - ax) + ny * (EW[10] - ay) + nz * (EW[11] - az);
      D3[0] = -nx; D3[1] = -ny; D3[2] = -nz;
      supportM(P, Q, EW, EP, EQ, EIP, EIQ, 4);
      const h2 = -(nx * (EW[12] - ax) + ny * (EW[13] - ay) + nz * (EW[14] - az));
      if (h2 > h1) { if (h2 > 1e-10) { epaCopy(4, 3); ev = 4; } } else if (h1 > 1e-10) ev = 4;
    }
  }
  if (ev < 4) return 0;
  TMP[0] = (EW[0] + EW[3] + EW[6] + EW[9]) * 0.25; TMP[1] = (EW[1] + EW[4] + EW[7] + EW[10]) * 0.25; TMP[2] = (EW[2] + EW[5] + EW[8] + EW[11]) * 0.25;
  ef = 0;
  epaFaceOriented(0, 1, 2); epaFaceOriented(0, 3, 1); epaFaceOriented(0, 2, 3); epaFaceOriented(1, 3, 2);
  let best = -1;
  for (let iter = 0; iter <= EPA_ITERS; iter++) {
    best = -1;
    let bd = Infinity;
    for (let f = 0; f < ef; f++) if (FAL[f] === 1 && FN[f * 4 + 3] < bd) { bd = FN[f * 4 + 3]; best = f; }
    if (best < 0) return 0;
    if (iter === EPA_ITERS || ev >= EPA_MAXV) break;
    const nx = FN[best * 4], ny = FN[best * 4 + 1], nz = FN[best * 4 + 2];
    D3[0] = nx; D3[1] = ny; D3[2] = nz;
    supportM(P, Q, EW, EP, EQ, EIP, EIQ, ev);
    const wx = EW[ev * 3], wy = EW[ev * 3 + 1], wz = EW[ev * 3 + 2];
    if (nx * wx + ny * wy + nz * wz - bd <= EPA_TOL * (1 + (bd < 0 ? -bd : bd))) break;
    const fb = best * 3;
    if ((EIP[ev] === EIP[FI[fb]] && EIQ[ev] === EIQ[FI[fb]]) || (EIP[ev] === EIP[FI[fb + 1]] && EIQ[ev] === EIQ[FI[fb + 1]]) ||
      (EIP[ev] === EIP[FI[fb + 2]] && EIQ[ev] === EIQ[FI[fb + 2]])) break;
    const nv = ev++;
    ne = 0;
    for (let f = 0; f < ef; f++) {
      if (FAL[f] === 0) continue;
      const i = FI[f * 3];
      if (f !== best && FN[f * 4] * (wx - EW[i * 3]) + FN[f * 4 + 1] * (wy - EW[i * 3 + 1]) + FN[f * 4 + 2] * (wz - EW[i * 3 + 2]) <= 1e-12) continue;
      FAL[f] = 0;
      const j = FI[f * 3 + 1], k = FI[f * 3 + 2];
      epaEdge(i, j); epaEdge(j, k); epaEdge(k, i);
    }
    if (ef + ne > EPA_MAXF) { // compact the dead faces away; give up when still full
      let m = 0;
      for (let f = 0; f < ef; f++) {
        if (FAL[f] === 0) continue;
        if (m !== f) {
          FI[m * 3] = FI[f * 3]; FI[m * 3 + 1] = FI[f * 3 + 1]; FI[m * 3 + 2] = FI[f * 3 + 2];
          for (let c = 0; c < 4; c++) FN[m * 4 + c] = FN[f * 4 + c];
          FAL[m] = 1;
        }
        m++;
      }
      ef = m;
      if (ef + ne > EPA_MAXF) { ev = EPA_MAXV; continue; }
    }
    for (let e = 0; e < ne; e++) epaFace(ef++, EDG[e * 2], EDG[e * 2 + 1], nv);
  }
  // projection of the origin on the closest face -> barycentric weights -> witnesses
  const nx = FN[best * 4], ny = FN[best * 4 + 1], nz = FN[best * 4 + 2], d = FN[best * 4 + 3];
  const i = FI[best * 3], j = FI[best * 3 + 1], k = FI[best * 3 + 2];
  const ax = EW[i * 3], ay = EW[i * 3 + 1], az = EW[i * 3 + 2];
  const v0x = EW[j * 3] - ax, v0y = EW[j * 3 + 1] - ay, v0z = EW[j * 3 + 2] - az;
  const v1x = EW[k * 3] - ax, v1y = EW[k * 3 + 1] - ay, v1z = EW[k * 3 + 2] - az;
  const v2x = nx * d - ax, v2y = ny * d - ay, v2z = nz * d - az;
  const d00 = v0x * v0x + v0y * v0y + v0z * v0z, d01 = v0x * v1x + v0y * v1y + v0z * v1z, d11 = v1x * v1x + v1y * v1y + v1z * v1z;
  const d20 = v2x * v0x + v2y * v0y + v2z * v0z, d21 = v2x * v1x + v2y * v1y + v2z * v1z;
  const den = d00 * d11 - d01 * d01;
  let bv = 0, bw = 0;
  if (den > 0) { bv = (d11 * d20 - d01 * d21) / den; bw = (d00 * d21 - d01 * d20) / den; }
  const bu = 1 - bv - bw;
  EOUT[0] = nx; EOUT[1] = ny; EOUT[2] = nz; EOUT[3] = d;
  for (let c = 0; c < 3; c++) {
    EOUT[4 + c] = bu * EP[i * 3 + c] + bv * EP[j * 3 + c] + bw * EP[k * 3 + c];
    EOUT[7 + c] = bu * EQ[i * 3 + c] + bv * EQ[j * 3 + c] + bw * EQ[k * 3 + c];
  }
  return 1;
}

// ---- pair routines (P.kind <= Q.kind; normal from P to Q; results through pushCand) ------------------------------------
function sphereSphere(P, Q) {
  const a = P.p, b = Q.p, rP = P.r, rQ = Q.r, margin = CFG[0];
  const dx = b[0] - a[0], dy = b[1] - a[1], dz = b[2] - a[2];
  const lim = rP + rQ + margin, d2 = dx * dx + dy * dy + dz * dz;
  if (d2 > lim * lim) return;
  const d = Math.sqrt(d2);
  let nx = 0, ny = 1, nz = 0;
  if (d > 1e-12) { nx = dx / d; ny = dy / d; nz = dz / d; }
  CW[0] = (a[0] + nx * rP + b[0] - nx * rQ) * 0.5; CW[1] = (a[1] + ny * rP + b[1] - ny * rQ) * 0.5; CW[2] = (a[2] + nz * rP + b[2] - nz * rQ) * 0.5;
  CW[3] = nx; CW[4] = ny; CW[5] = nz; CW[6] = rP + rQ - d;
  pushCand(TAG_SS);
}

function sphereCapsule(P, Q) {
  const c = P.p, s = Q.s, rP = P.r, rQ = Q.r, margin = CFG[0];
  const dx = s[3] - s[0], dy = s[4] - s[1], dz = s[5] - s[2], dd = dx * dx + dy * dy + dz * dz;
  let t = 0;
  if (dd > 1e-24) { t = ((c[0] - s[0]) * dx + (c[1] - s[1]) * dy + (c[2] - s[2]) * dz) / dd; t = t < 0 ? 0 : t > 1 ? 1 : t; }
  const qx = s[0] + dx * t, qy = s[1] + dy * t, qz = s[2] + dz * t;
  const vx = qx - c[0], vy = qy - c[1], vz = qz - c[2], v2 = vx * vx + vy * vy + vz * vz;
  const lim = rP + rQ + margin;
  if (v2 > lim * lim) return;
  const d = Math.sqrt(v2);
  let nx, ny, nz;
  if (d > 1e-12) { nx = vx / d; ny = vy / d; nz = vz / d; } else { D3[0] = dx; D3[1] = dy; D3[2] = dz; perp(); nx = TMP[0]; ny = TMP[1]; nz = TMP[2]; }
  CW[0] = (c[0] + nx * rP + qx - nx * rQ) * 0.5; CW[1] = (c[1] + ny * rP + qy - ny * rQ) * 0.5; CW[2] = (c[2] + nz * rP + qz - nz * rQ) * 0.5;
  CW[3] = nx; CW[4] = ny; CW[5] = nz; CW[6] = rP + rQ - d;
  pushCand(TAG_SC);
}

// Closest points of segments SSI[0..5] and SSI[6..11] (Ericson 5.1.9): s, t -> SSO[0..1], points -> SSO[2..4], SSO[5..7].
function closestSegSeg() {
  const p1x = SSI[0], p1y = SSI[1], p1z = SSI[2], p2x = SSI[6], p2y = SSI[7], p2z = SSI[8];
  const d1x = SSI[3] - p1x, d1y = SSI[4] - p1y, d1z = SSI[5] - p1z, d2x = SSI[9] - p2x, d2y = SSI[10] - p2y, d2z = SSI[11] - p2z;
  const rx = p1x - p2x, ry = p1y - p2y, rz = p1z - p2z;
  const a = d1x * d1x + d1y * d1y + d1z * d1z, e = d2x * d2x + d2y * d2y + d2z * d2z, f = d2x * rx + d2y * ry + d2z * rz;
  let s = 0, t = 0;
  if (a <= 1e-24 && e <= 1e-24) { s = 0; t = 0; } else if (a <= 1e-24) { s = 0; t = f / e; t = t < 0 ? 0 : t > 1 ? 1 : t; } else {
    const c = d1x * rx + d1y * ry + d1z * rz;
    if (e <= 1e-24) { t = 0; s = -c / a; s = s < 0 ? 0 : s > 1 ? 1 : s; } else {
      const b = d1x * d2x + d1y * d2y + d1z * d2z, den = a * e - b * b;
      if (den > 0) { s = (b * f - c * e) / den; s = s < 0 ? 0 : s > 1 ? 1 : s; } else s = 0;
      t = (b * s + f) / e;
      if (t < 0) { t = 0; s = -c / a; s = s < 0 ? 0 : s > 1 ? 1 : s; } else if (t > 1) { t = 1; s = (b - c) / a; s = s < 0 ? 0 : s > 1 ? 1 : s; }
    }
  }
  SSO[0] = s; SSO[1] = t;
  SSO[2] = p1x + d1x * s; SSO[3] = p1y + d1y * s; SSO[4] = p1z + d1z * s;
  SSO[5] = p2x + d2x * t; SSO[6] = p2y + d2y * t; SSO[7] = p2z + d2z * t;
}

function capsuleCapsule(P, Q) {
  const s1 = P.s, s2 = Q.s, rP = P.r, rQ = Q.r, margin = CFG[0], rs = rP + rQ;
  for (let i = 0; i < 6; i++) { SSI[i] = s1[i]; SSI[6 + i] = s2[i]; }
  closestSegSeg();
  const c1x = SSO[2], c1y = SSO[3], c1z = SSO[4], c2x = SSO[5], c2y = SSO[6], c2z = SSO[7];
  const vx = c2x - c1x, vy = c2y - c1y, vz = c2z - c1z, v2 = vx * vx + vy * vy + vz * vz;
  const lim = rs + margin;
  if (v2 > lim * lim) return;
  const d = Math.sqrt(v2);
  const d1x = s1[3] - s1[0], d1y = s1[4] - s1[1], d1z = s1[5] - s1[2], d2x = s2[3] - s2[0], d2y = s2[4] - s2[1], d2z = s2[5] - s2[2];
  const a = d1x * d1x + d1y * d1y + d1z * d1z, e = d2x * d2x + d2y * d2y + d2z * d2z, b = d1x * d2x + d1y * d2y + d1z * d2z;
  let nx, ny, nz;
  if (d > 1e-12) { nx = vx / d; ny = vy / d; nz = vz / d; } else {
    let cx = d1y * d2z - d1z * d2y, cy = d1z * d2x - d1x * d2z, cz = d1x * d2y - d1y * d2x;
    const cl = Math.sqrt(cx * cx + cy * cy + cz * cz);
    if (cl > 1e-12) { cx /= cl; cy /= cl; cz /= cl; } else { D3[0] = d1x; D3[1] = d1y; D3[2] = d1z; perp(); cx = TMP[0]; cy = TMP[1]; cz = TMP[2]; }
    const ox = Q.p[0] - P.p[0], oy = Q.p[1] - P.p[1], oz = Q.p[2] - P.p[2];
    if (cx * ox + cy * oy + cz * oz < 0) { cx = -cx; cy = -cy; cz = -cz; }
    nx = cx; ny = cy; nz = cz;
  }
  // near-parallel axes with overlapping projections: the two ends of the overlap
  if (a > 1e-24 && e > 1e-24 && a * e - b * b <= PARALLEL2 * a * e) {
    const u0 = ((s2[0] - s1[0]) * d1x + (s2[1] - s1[1]) * d1y + (s2[2] - s1[2]) * d1z) / a;
    const u1 = ((s2[3] - s1[0]) * d1x + (s2[4] - s1[1]) * d1y + (s2[5] - s1[2]) * d1z) / a;
    let lo = u0 < u1 ? u0 : u1, hi = u0 < u1 ? u1 : u0;
    if (lo < 0) lo = 0;
    if (hi > 1) hi = 1;
    if (hi - lo > 1e-9) {
      let cnt = 0;
      for (let pass = 0; pass < 2; pass++) {
        const s = pass === 0 ? lo : hi;
        const x1 = s1[0] + d1x * s, y1 = s1[1] + d1y * s, z1 = s1[2] + d1z * s;
        let t = ((x1 - s2[0]) * d2x + (y1 - s2[1]) * d2y + (z1 - s2[2]) * d2z) / e;
        t = t < 0 ? 0 : t > 1 ? 1 : t;
        const x2 = s2[0] + d2x * t, y2 = s2[1] + d2y * t, z2 = s2[2] + d2z * t;
        const sep = (x2 - x1) * nx + (y2 - y1) * ny + (z2 - z1) * nz;
        const depth = rs - sep;
        if (depth < -margin) continue;
        CW[0] = (x1 + nx * rP + x2 - nx * rQ) * 0.5; CW[1] = (y1 + ny * rP + y2 - ny * rQ) * 0.5; CW[2] = (z1 + nz * rP + z2 - nz * rQ) * 0.5;
        CW[3] = nx; CW[4] = ny; CW[5] = nz; CW[6] = depth;
        pushCand(TAG_CC + 1 + pass);
        cnt++;
      }
      if (cnt > 0) return;
    }
  }
  CW[0] = (c1x + nx * rP + c2x - nx * rQ) * 0.5; CW[1] = (c1y + ny * rP + c2y - ny * rQ) * 0.5; CW[2] = (c1z + nz * rP + c2z - nz * rQ) * 0.5;
  CW[3] = nx; CW[4] = ny; CW[5] = nz; CW[6] = rs - d;
  pushCand(TAG_CC);
}

function sphereBox(P, Q) {
  const R = Q.R, h = Q.h, q = Q.p, r = P.r, margin = CFG[0];
  const wx = P.p[0] - q[0], wy = P.p[1] - q[1], wz = P.p[2] - q[2];
  const cx = R[0] * wx + R[3] * wy + R[6] * wz, cy = R[1] * wx + R[4] * wy + R[7] * wz, cz = R[2] * wx + R[5] * wy + R[8] * wz;
  const hx = h[0], hy = h[1], hz = h[2];
  const bx = cx > hx ? hx : cx < -hx ? -hx : cx, by = cy > hy ? hy : cy < -hy ? -hy : cy, bz = cz > hz ? hz : cz < -hz ? -hz : cz;
  const dx = bx - cx, dy = by - cy, dz = bz - cz, d2 = dx * dx + dy * dy + dz * dz;
  let lx, ly, lz, mx, my, mz, depth, key;
  if (d2 > 1e-24) {
    const lim = r + margin;
    if (d2 > lim * lim) return;
    const d = Math.sqrt(d2);
    lx = dx / d; ly = dy / d; lz = dz / d;
    depth = r - d;
    mx = (cx + lx * r + bx) * 0.5; my = (cy + ly * r + by) * 0.5; mz = (cz + lz * r + bz) * 0.5;
    key = (cx > hx ? 1 : cx < -hx ? 2 : 0) + 3 * (cy > hy ? 1 : cy < -hy ? 2 : 0) + 9 * (cz > hz ? 1 : cz < -hz ? 2 : 0);
  } else {
    // centre inside: out through the nearest face (first axis on ties)
    const px = hx - (cx < 0 ? -cx : cx), py = hy - (cy < 0 ? -cy : cy), pz = hz - (cz < 0 ? -cz : cz);
    let ax = 0, pen = px;
    if (py < pen) { ax = 1; pen = py; }
    if (pz < pen) { ax = 2; pen = pz; }
    const c = ax === 0 ? cx : ax === 1 ? cy : cz, sg = c >= 0 ? 1 : -1;
    const fx = ax === 0 ? sg : 0, fy = ax === 1 ? sg : 0, fz = ax === 2 ? sg : 0; // face normal (out of the box)
    lx = -fx; ly = -fy; lz = -fz;
    depth = r + pen;
    const k = (pen - r) * 0.5;
    mx = cx + fx * k; my = cy + fy * k; mz = cz + fz * k;
    key = 27 + ax * 2 + (sg > 0 ? 0 : 1);
  }
  CW[0] = q[0] + R[0] * mx + R[1] * my + R[2] * mz; CW[1] = q[1] + R[3] * mx + R[4] * my + R[5] * mz; CW[2] = q[2] + R[6] * mx + R[7] * my + R[8] * mz;
  CW[3] = R[0] * lx + R[1] * ly + R[2] * lz; CW[4] = R[3] * lx + R[4] * ly + R[5] * lz; CW[5] = R[6] * lx + R[7] * ly + R[8] * lz;
  CW[6] = depth;
  pushCand(mixKey(TAG_SB, key));
}

function capsuleBox(P, Q) {
  const R = Q.R, h = Q.h, q = Q.p, s = P.s, r = P.r, margin = CFG[0];
  let ex = s[0] - q[0], ey = s[1] - q[1], ez = s[2] - q[2];
  const a0 = R[0] * ex + R[3] * ey + R[6] * ez, a1 = R[1] * ex + R[4] * ey + R[7] * ez, a2 = R[2] * ex + R[5] * ey + R[8] * ez;
  ex = s[3] - q[0]; ey = s[4] - q[1]; ez = s[5] - q[2];
  const b0 = R[0] * ex + R[3] * ey + R[6] * ez, b1 = R[1] * ex + R[4] * ey + R[7] * ez, b2 = R[2] * ex + R[5] * ey + R[8] * ez;
  LS[0] = a0; LS[1] = a1; LS[2] = a2; LS[3] = b0 - a0; LS[4] = b1 - a1; LS[5] = b2 - a2;
  // breakpoints of the piecewise quadratic, sorted
  BT[0] = 0;
  let nb = 1;
  for (let i = 0; i < 3; i++) {
    const d = LS[3 + i];
    if (d > 1e-15 || d < -1e-15) {
      let t = (h[i] - LS[i]) / d;
      if (t > 0 && t < 1) BT[nb++] = t;
      t = (-h[i] - LS[i]) / d;
      if (t > 0 && t < 1) BT[nb++] = t;
    }
  }
  BT[nb++] = 1;
  for (let i = 1; i < nb; i++) { const v = BT[i]; let j = i - 1; while (j >= 0 && BT[j] > v) { BT[j + 1] = BT[j]; j--; } BT[j + 1] = v; }
  let best = Infinity, bt = 0;
  for (let k = 0; k + 1 < nb; k++) {
    const ta = BT[k], tb = BT[k + 1], tm = (ta + tb) * 0.5;
    let num = 0, den = 0;
    for (let i = 0; i < 3; i++) {
      const d = LS[3 + i], x = LS[i] + tm * d;
      if (x > h[i]) { num += d * (h[i] - LS[i]); den += d * d; } else if (x < -h[i]) { num += d * (-h[i] - LS[i]); den += d * d; }
    }
    let t = den > 1e-30 ? num / den : ta;
    t = t < ta ? ta : t > tb ? tb : t;
    let f = 0; // squared distance from the box to the segment point at t (inline: no double-returning call)
    for (let i = 0; i < 3; i++) {
      const x = LS[i] + t * LS[3 + i], hi = h[i];
      if (x > hi) f += (x - hi) * (x - hi); else if (x < -hi) f += (x + hi) * (x + hi);
    }
    if (f < best) { best = f; bt = t; }
  }
  const qx = LS[0] + bt * LS[3], qy = LS[1] + bt * LS[4], qz = LS[2] + bt * LS[5];
  if (best > 1e-24) {
    const lim = r + margin;
    if (best > lim * lim) return;
    const bx = qx > h[0] ? h[0] : qx < -h[0] ? -h[0] : qx, by = qy > h[1] ? h[1] : qy < -h[1] ? -h[1] : qy;
    const bz = qz > h[2] ? h[2] : qz < -h[2] ? -h[2] : qz;
    const ox = qx > h[0] || qx < -h[0] ? 1 : 0, oy = qy > h[1] || qy < -h[1] ? 1 : 0, oz = qz > h[2] || qz < -h[2] ? 1 : 0;
    if (ox + oy + oz === 1) { // inside a face's region: clip the segment to that face
      const ax = ox === 1 ? 0 : oy === 1 ? 1 : 2, c = ax === 0 ? qx : ax === 1 ? qy : qz;
      if (capsuleFace(P, Q, 2 * ax + (c > 0 ? 0 : 1), true) > 0) return;
    }
    const d = Math.sqrt(best), vx = (bx - qx) / d, vy = (by - qy) / d, vz = (bz - qz) / d;
    const mx = (qx + vx * r + bx) * 0.5, my = (qy + vy * r + by) * 0.5, mz = (qz + vz * r + bz) * 0.5;
    CW[0] = q[0] + R[0] * mx + R[1] * my + R[2] * mz; CW[1] = q[1] + R[3] * mx + R[4] * my + R[5] * mz; CW[2] = q[2] + R[6] * mx + R[7] * my + R[8] * mz;
    CW[3] = R[0] * vx + R[1] * vy + R[2] * vz; CW[4] = R[3] * vx + R[4] * vy + R[5] * vz; CW[5] = R[6] * vx + R[7] * vy + R[8] * vz;
    CW[6] = r - d;
    const region = (qx > h[0] ? 1 : qx < -h[0] ? 2 : 0) + 3 * (qy > h[1] ? 1 : qy < -h[1] ? 2 : 0) + 9 * (qz > h[2] ? 1 : qz < -h[2] ? 2 : 0);
    pushCand(mixKey(TAG_CB, region));
    return;
  }
  // the segment intersects the box: SAT over the face axes (both ways) and the cross(segment, axis) axes
  let fPen = Infinity, fAx = 0, fSg = 1;
  for (let i = 0; i < 3; i++) {
    const lo = LS[i] < LS[i] + LS[3 + i] ? LS[i] : LS[i] + LS[3 + i], hi = LS[i] < LS[i] + LS[3 + i] ? LS[i] + LS[3 + i] : LS[i];
    const pp = h[i] - lo, pm = hi + h[i];
    if (pp < fPen) { fPen = pp; fAx = i; fSg = 1; }
    if (pm < fPen) { fPen = pm; fAx = i; fSg = -1; }
  }
  const dx = LS[3], dy = LS[4], dz = LS[5], dl = Math.sqrt(dx * dx + dy * dy + dz * dz);
  let ePen = Infinity, eAx = 0, eLx = 0, eLy = 0, eLz = 0;
  for (let i = 0; i < 3 && dl > 0; i++) {
    let lx = i === 0 ? 0 : i === 1 ? -dz : dy, ly = i === 0 ? dz : i === 1 ? 0 : -dx, lz = i === 0 ? -dy : i === 1 ? dx : 0;
    const ll = Math.sqrt(lx * lx + ly * ly + lz * lz);
    if (!(ll > 1e-9 * dl)) continue;
    lx /= ll; ly /= ll; lz /= ll;
    const rb = h[0] * (lx < 0 ? -lx : lx) + h[1] * (ly < 0 ? -ly : ly) + h[2] * (lz < 0 ? -lz : lz);
    const sp = LS[0] * lx + LS[1] * ly + LS[2] * lz;
    if (rb - sp < ePen) { ePen = rb - sp; eAx = i; eLx = lx; eLy = ly; eLz = lz; }
    if (sp + rb < ePen) { ePen = sp + rb; eAx = i; eLx = -lx; eLy = -ly; eLz = -lz; }
  }
  if (!(ePen + SAT_EDGE_ABS + SAT_EDGE_REL * fPen < fPen)) {
    if (capsuleFace(P, Q, 2 * fAx + (fSg > 0 ? 0 : 1), true) > 0) return;
    // (cannot normally happen: the deepest end is over the face) fall back to the deepest end point
    const c0 = fSg * LS[fAx], c1 = fSg * (LS[fAx] + LS[3 + fAx]), t = c1 < c0 ? 1 : 0;
    const px = LS[0] + t * dx, py = LS[1] + t * dy, pz = LS[2] + t * dz;
    const ux = fAx === 0 ? fSg : 0, uy = fAx === 1 ? fSg : 0, uz = fAx === 2 ? fSg : 0, k = (fPen - r) * 0.5;
    const mx = px + ux * k, my = py + uy * k, mz = pz + uz * k;
    CW[0] = q[0] + R[0] * mx + R[1] * my + R[2] * mz; CW[1] = q[1] + R[3] * mx + R[4] * my + R[5] * mz; CW[2] = q[2] + R[6] * mx + R[7] * my + R[8] * mz;
    CW[3] = -(R[0] * ux + R[1] * uy + R[2] * uz); CW[4] = -(R[3] * ux + R[4] * uy + R[5] * uz); CW[5] = -(R[6] * ux + R[7] * uy + R[8] * uz);
    CW[6] = fPen + r;
    pushCand(mixKey(TAG_CBE, 64 + fAx * 2 + (fSg > 0 ? 0 : 1)));
    return;
  }
  // edge axis: the capsule moves along u = eL; the box edge parallel to axis eAx on the capsule's side
  const ux = eLx, uy = eLy, uz = eLz;
  const cxs = ux >= 0 ? h[0] : -h[0], cys = uy >= 0 ? h[1] : -h[1], czs = uz >= 0 ? h[2] : -h[2];
  SSI[0] = LS[0]; SSI[1] = LS[1]; SSI[2] = LS[2]; SSI[3] = LS[0] + dx; SSI[4] = LS[1] + dy; SSI[5] = LS[2] + dz;
  SSI[6] = eAx === 0 ? -h[0] : cxs; SSI[7] = eAx === 1 ? -h[1] : cys; SSI[8] = eAx === 2 ? -h[2] : czs;
  SSI[9] = eAx === 0 ? h[0] : cxs; SSI[10] = eAx === 1 ? h[1] : cys; SSI[11] = eAx === 2 ? h[2] : czs;
  closestSegSeg();
  const mx = (SSO[2] - ux * r + SSO[5]) * 0.5, my = (SSO[3] - uy * r + SSO[6]) * 0.5, mz = (SSO[4] - uz * r + SSO[7]) * 0.5;
  CW[0] = q[0] + R[0] * mx + R[1] * my + R[2] * mz; CW[1] = q[1] + R[3] * mx + R[4] * my + R[5] * mz; CW[2] = q[2] + R[6] * mx + R[7] * my + R[8] * mz;
  CW[3] = -(R[0] * ux + R[1] * uy + R[2] * uz); CW[4] = -(R[3] * ux + R[4] * uy + R[5] * uz); CW[5] = -(R[6] * ux + R[7] * uy + R[8] * uz);
  CW[6] = ePen + r;
  pushCand(mixKey(TAG_CBE, eAx * 8 + (ux >= 0 ? 1 : 0) + (uy >= 0 ? 2 : 0) + (uz >= 0 ? 4 : 0)));
}

function boxBox(P, Q) {
  curNP.stats.sat++;
  const RA = P.R, RB = Q.R, ha = P.h, hb = Q.h, margin = CFG[0];
  const Tx = Q.p[0] - P.p[0], Ty = Q.p[1] - P.p[1], Tz = Q.p[2] - P.p[2];
  for (let i = 0; i < 3; i++) {
    BTV[i] = RA[i] * Tx + RA[3 + i] * Ty + RA[6 + i] * Tz;
    for (let j = 0; j < 3; j++) {
      const c = RA[i] * RB[j] + RA[3 + i] * RB[3 + j] + RA[6 + i] * RB[6 + j];
      BC[i * 3 + j] = c; BA[i * 3 + j] = c < 0 ? -c : c;
    }
  }
  let sepA = -Infinity, iA = 0;
  for (let i = 0; i < 3; i++) {
    const t = BTV[i] < 0 ? -BTV[i] : BTV[i];
    const s = t - (ha[i] + hb[0] * BA[i * 3] + hb[1] * BA[i * 3 + 1] + hb[2] * BA[i * 3 + 2]);
    if (s > margin) return;
    if (s > sepA) { sepA = s; iA = i; }
  }
  let sepB = -Infinity, jB = 0;
  for (let j = 0; j < 3; j++) {
    const tb = BTV[0] * BC[j] + BTV[1] * BC[3 + j] + BTV[2] * BC[6 + j];
    const s = (tb < 0 ? -tb : tb) - (ha[0] * BA[j] + ha[1] * BA[3 + j] + ha[2] * BA[6 + j] + hb[j]);
    if (s > margin) return;
    if (s > sepB) { sepB = s; jB = j; }
  }
  let sepE = -Infinity, eI = -1, eJ = -1;
  for (let i = 0; i < 3; i++) {
    const i1 = i === 2 ? 0 : i + 1, i2 = i1 === 2 ? 0 : i1 + 1;
    for (let j = 0; j < 3; j++) {
      const j1 = j === 2 ? 0 : j + 1, j2 = j1 === 2 ? 0 : j1 + 1;
      const c1 = BC[i1 * 3 + j], c2 = BC[i2 * 3 + j];
      const len = Math.sqrt(c1 * c1 + c2 * c2);
      if (len < 1e-6) continue;
      const tl = BTV[i2] * c1 - BTV[i1] * c2;
      const ra = ha[i1] * BA[i2 * 3 + j] + ha[i2] * BA[i1 * 3 + j];
      const rb = hb[j1] * BA[i * 3 + j2] + hb[j2] * BA[i * 3 + j1];
      const s = ((tl < 0 ? -tl : tl) - ra - rb) / len;
      if (s > margin) return;
      if (s > sepE) { sepE = s; eI = i; eJ = j; }
    }
  }
  const useB = sepB > sepA + SAT_FACE_TOL;
  const faceSep = useB ? sepB : sepA;
  SC[1] = sepE;
  if (eI >= 0 && sepE > faceSep + SAT_EDGE_ABS + SAT_EDGE_REL * (faceSep < 0 ? -faceSep : faceSep)) { boxEdge(P, Q, eI, eJ); return; }
  let n;
  if (!useB) n = clipFaces(P, 2 * iA + (BTV[iA] >= 0 ? 0 : 1), Q, true);
  else {
    const tb = BTV[0] * BC[jB] + BTV[1] * BC[3 + jB] + BTV[2] * BC[6 + jB];
    n = clipFaces(Q, 2 * jB + (tb >= 0 ? 1 : 0), P, false);
  }
  if (n === 0) gjkPair(P, Q); // the faces do not overlap after all (separated corner-to-corner): exact GJK
}

// Box-box edge contact on axis a_i x b_j (BC from boxBox, separation in SC[1]): midpoint of the closest points of the
// two supporting edges, depth -separation.
function boxEdge(P, Q, i, j) {
  const RA = P.R, RB = Q.R, ha = P.h, hb = Q.h;
  const Tx = Q.p[0] - P.p[0], Ty = Q.p[1] - P.p[1], Tz = Q.p[2] - P.p[2];
  const i1 = i === 2 ? 0 : i + 1, i2 = i1 === 2 ? 0 : i1 + 1;
  const L = TMP;
  L[i] = 0; L[i1] = -BC[i2 * 3 + j]; L[i2] = BC[i1 * 3 + j]; // a_i x b_j in A's frame
  let nx = RA[0] * L[0] + RA[1] * L[1] + RA[2] * L[2], ny = RA[3] * L[0] + RA[4] * L[1] + RA[5] * L[2], nz = RA[6] * L[0] + RA[7] * L[1] + RA[8] * L[2];
  const nl = Math.sqrt(nx * nx + ny * ny + nz * nz);
  nx /= nl; ny /= nl; nz /= nl;
  if (nx * Tx + ny * Ty + nz * Tz < 0) { nx = -nx; ny = -ny; nz = -nz; }
  let cax = P.p[0], cay = P.p[1], caz = P.p[2], cbx = Q.p[0], cby = Q.p[1], cbz = Q.p[2], bitsA = 0, bitsB = 0;
  for (let k = 0; k < 3; k++) {
    const ax = RA[k], ay = RA[3 + k], az = RA[6 + k], bx = RB[k], by = RB[3 + k], bz = RB[6 + k];
    if (k !== i) {
      const sg = nx * ax + ny * ay + nz * az >= 0 ? 1 : -1;
      if (sg > 0) bitsA |= 1 << k;
      cax += sg * ha[k] * ax; cay += sg * ha[k] * ay; caz += sg * ha[k] * az;
    }
    if (k !== j) {
      const sg = nx * bx + ny * by + nz * bz >= 0 ? -1 : 1;
      if (sg > 0) bitsB |= 1 << k;
      cbx += sg * hb[k] * bx; cby += sg * hb[k] * by; cbz += sg * hb[k] * bz;
    }
  }
  const eax = RA[i] * ha[i], eay = RA[3 + i] * ha[i], eaz = RA[6 + i] * ha[i];
  const ebx = RB[j] * hb[j], eby = RB[3 + j] * hb[j], ebz = RB[6 + j] * hb[j];
  SSI[0] = cax - eax; SSI[1] = cay - eay; SSI[2] = caz - eaz; SSI[3] = cax + eax; SSI[4] = cay + eay; SSI[5] = caz + eaz;
  SSI[6] = cbx - ebx; SSI[7] = cby - eby; SSI[8] = cbz - ebz; SSI[9] = cbx + ebx; SSI[10] = cby + eby; SSI[11] = cbz + ebz;
  closestSegSeg();
  CW[0] = (SSO[2] + SSO[5]) * 0.5; CW[1] = (SSO[3] + SSO[6]) * 0.5; CW[2] = (SSO[4] + SSO[7]) * 0.5;
  CW[3] = nx; CW[4] = ny; CW[5] = nz; CW[6] = -SC[1];
  pushCand(mixKey(mixKey(TAG_EE, i * 8 + bitsA), j * 8 + bitsB));
}

// One contact between the witness points NRM[4..6] (P core) and NRM[7..9] (Q core).
function singlePoint(P, Q) {
  const nx = NRM[0], ny = NRM[1], nz = NRM[2], rP = P.r, rQ = Q.r;
  CW[0] = (NRM[4] + nx * rP + NRM[7] - nx * rQ) * 0.5;
  CW[1] = (NRM[5] + ny * rP + NRM[8] - ny * rQ) * 0.5;
  CW[2] = (NRM[6] + nz * rP + NRM[9] - nz * rQ) * 0.5;
  CW[3] = nx; CW[4] = ny; CW[5] = nz; CW[6] = NRM[3];
  D3[0] = nx; D3[1] = ny; D3[2] = nz;
  const ip = support(P, TMP, 0);
  D3[0] = -nx; D3[1] = -ny; D3[2] = -nz;
  const iq = support(Q, TMP, 3);
  pushCand(mixKey(mixKey(TAG_PT, ip), iq));
}

// True when the candidates c0 .. candCount (a clipped manifold) reach the GJK / EPA depth NRM[3]: a clip that lost
// the deepest feature (it hangs over the reference face's edge: really an edge contact whose normal is within
// FACE_COS of the face) is shallower and is replaced by the exact single point.
function clipReachesDepth(c0) {
  let m = -Infinity;
  for (let i = c0; i < candCount; i++) if (CD[i] > m) m = CD[i];
  const d = NRM[3];
  return m >= d - 1e-6 * (1 + (d < 0 ? -d : d));
}

// Manifold for a GJK / EPA result in NRM: face clipping for face contacts, else one point.
function manifold(P, Q) {
  const nx = NRM[0], ny = NRM[1], nz = NRM[2];
  const qPoly = Q.kind >= K_BOX, c0 = candCount;
  if (P.kind >= K_BOX && qPoly) {
    D3[0] = nx; D3[1] = ny; D3[2] = nz;
    const fP = bestFace(P), aP = SC[8];
    D3[0] = -nx; D3[1] = -ny; D3[2] = -nz;
    const fQ = bestFace(Q), aQ = SC[8];
    if (aP >= FACE_COS || aQ >= FACE_COS) {
      const n = aP >= aQ - REF_TOL ? clipFaces(P, fP, Q, true) : clipFaces(Q, fQ, P, false);
      if (n > 0 && clipReachesDepth(c0)) return;
      candCount = c0;
    }
  } else if (P.kind === K_SEG && qPoly) {
    D3[0] = -nx; D3[1] = -ny; D3[2] = -nz;
    const f = bestFace(Q);
    if (SC[8] >= FACE_COS) {
      if (capsuleFace(P, Q, f, true) > 0 && clipReachesDepth(c0)) return;
      candCount = c0;
    }
  }
  singlePoint(P, Q);
}

// GJK (+ EPA) pair: anything against a hull (and box against hull).
function gjkPair(P, Q) {
  const margin = CFG[0], rP = P.r, rQ = Q.r;
  SC[0] = rP + rQ + margin;
  const r = gjk(P, Q);
  if (r === 2) return;
  if (r === 1) {
    const dist = GOUT[6], depth = rP + rQ - dist;
    if (depth < -margin) return;
    const inv = 1 / dist;
    NRM[0] = (GOUT[3] - GOUT[0]) * inv; NRM[1] = (GOUT[4] - GOUT[1]) * inv; NRM[2] = (GOUT[5] - GOUT[2]) * inv;
    NRM[3] = depth;
    for (let c = 0; c < 6; c++) NRM[4 + c] = GOUT[c];
  } else if (epa(P, Q) === 1) {
    NRM[0] = EOUT[0]; NRM[1] = EOUT[1]; NRM[2] = EOUT[2]; NRM[3] = EOUT[3] + rP + rQ;
    for (let c = 0; c < 6; c++) NRM[4 + c] = EOUT[4 + c];
  } else {
    // EPA failed (degenerate touching cores): normal along the centres, the cores just touching
    let dx = Q.p[0] - P.p[0], dy = Q.p[1] - P.p[1], dz = Q.p[2] - P.p[2];
    const l = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (l > 1e-12) { dx /= l; dy /= l; dz /= l; } else { dx = 0; dy = 1; dz = 0; }
    NRM[0] = dx; NRM[1] = dy; NRM[2] = dz; NRM[3] = rP + rQ;
    for (let c = 0; c < 6; c++) NRM[4 + c] = GOUT[c];
  }
  manifold(P, Q);
}

// Ordered leaf dispatch (P.kind <= Q.kind).
function leafOrdered(P, Q) {
  const kp = P.kind, kq = Q.kind;
  if (kq === K_POINT) sphereSphere(P, Q);
  else if (kq === K_SEG) { if (kp === K_POINT) sphereCapsule(P, Q); else capsuleCapsule(P, Q); }
  else if (kq === K_BOX) { if (kp === K_POINT) sphereBox(P, Q); else if (kp === K_SEG) capsuleBox(P, Q); else boxBox(P, Q); }
  else gjkPair(P, Q);
}
function leafPair(P, Q) {
  if (P.kind > Q.kind) { curFlip = !curFlip; leafOrdered(Q, P); curFlip = !curFlip; } else leafOrdered(P, Q);
}

// ---- trimesh -----------------------------------------------------------------------------------------------------------
// Corner mask (bit k = corner k) of the barycentric support of the triangle witness EOUT[7..9] (EPA).
function triWitnessMask(T) {
  const t = T.t;
  const v0x = t[3] - t[0], v0y = t[4] - t[1], v0z = t[5] - t[2], v1x = t[6] - t[0], v1y = t[7] - t[1], v1z = t[8] - t[2];
  const v2x = EOUT[7] - t[0], v2y = EOUT[8] - t[1], v2z = EOUT[9] - t[2];
  const d00 = v0x * v0x + v0y * v0y + v0z * v0z, d01 = v0x * v1x + v0y * v1y + v0z * v1z, d11 = v1x * v1x + v1y * v1y + v1z * v1z;
  const d20 = v2x * v0x + v2y * v0y + v2z * v0z, d21 = v2x * v1x + v2y * v1y + v2z * v1z, den = d00 * d11 - d01 * d01;
  if (!(den > 0)) return 0;
  const v = (d11 * d20 - d01 * d21) / den, w = (d00 * d21 - d01 * d20) / den, u = 1 - v - w;
  return (u > 1e-7 ? 1 : 0) | (v > 1e-7 ? 2 : 0) | (w > 1e-7 ? 4 : 0);
}

// Emit the mesh edge / vertex contact in CW for the triangle corners in `mask`: tagged for meshPost, keyed by the
// sorted mesh vertex ids (not the triangle) so the key survives the contact moving to the neighbouring triangle.
// Returns false (nothing emitted) unless the mask is an edge or a vertex.
function meshFeatureCand(T, mask) {
  const bits = (mask & 1) + ((mask >> 1) & 1) + ((mask >> 2) & 1);
  if (bits < 1 || bits > 2) return false;
  let a = -1, b = -1;
  for (let k = 0; k < 3; k++) if ((mask & (1 << k)) !== 0) { if (a < 0) a = T.ids[k]; else b = T.ids[k]; }
  if (b >= 0 && b < a) { const x = a; a = b; b = x; }
  curFeat = bits === 2 ? FEAT_EDGE : FEAT_VERT; curFA = a; curFB = b; curFC = -1;
  const salt = curSalt;
  curSalt = curMeshSalt;
  pushCand(bits === 2 ? mixKey(mixKey(TAG_MEDGE, a), b) : mixKey(TAG_MVERT, a));
  curSalt = salt;
  return true;
}

// One triangle T (one-sided, thick) against the core X.
function meshTri(X, T) {
  const t = T.t, Nx = t[9], Ny = t[10], Nz = t[11], margin = CFG[0], rX = X.r, xp = X.p;
  if (Nx * (xp[0] - t[0]) + Ny * (xp[1] - t[1]) + Nz * (xp[2] - t[2]) < 0) return; // centre behind: back side
  D3[0] = -Nx; D3[1] = -Ny; D3[2] = -Nz;
  support(X, TMP, 6);
  const sx = TMP[6], sy = TMP[7], sz = TMP[8];
  const hCore = Nx * (sx - t[0]) + Ny * (sy - t[1]) + Nz * (sz - t[2]), sepFace = hCore - rX;
  if (sepFace > margin) return;
  SC[0] = rX + margin;
  const r = gjk(X, T);
  if (r === 2) return;
  let epaOk = false, endN = 0;
  if (r === 1) {
    const dist = GOUT[6], depth = rX - dist;
    if (depth < -margin) return;
    const inv = 1 / dist;
    const nx = (GOUT[3] - GOUT[0]) * inv, ny = (GOUT[4] - GOUT[1]) * inv, nz = (GOUT[5] - GOUT[2]) * inv;
    const ndN = nx * Nx + ny * Ny + nz * Nz;
    if (ndN > -1e-6) return; // not against the front face
    if (ndN > -FACE_COS) {
      let mask = 0;
      for (let j = 0; j < gjkN; j++) if (GL[j] > 1e-12) mask |= 1 << GIQ[j];
      CW[0] = (GOUT[0] + nx * rX + GOUT[3]) * 0.5; CW[1] = (GOUT[1] + ny * rX + GOUT[4]) * 0.5; CW[2] = (GOUT[2] + nz * rX + GOUT[5]) * 0.5;
      CW[3] = nx; CW[4] = ny; CW[5] = nz; CW[6] = depth;
      if (meshFeatureCand(T, mask)) return;
    }
  } else if (epa(X, T) === 1) {
    // the core overlaps the triangle: EPA's minimal translation decides. Out through the face (or through the back,
    // n . N > 0: the "thickness" pushes it out along N anyway) -> face contact below; out through the triangle's
    // boundary (a step's riser, a ridge) -> one point on that edge / vertex, which meshPost drops when a
    // neighbouring triangle carries the face contact and keeps on a real ridge.
    epaOk = true;
    const nx = EOUT[0], ny = EOUT[1], nz = EOUT[2];
    endN = nx * Nx + ny * Ny + nz * Nz;
    if (endN > -FACE_COS && endN <= 0) {
      CW[0] = (EOUT[4] + nx * rX + EOUT[7]) * 0.5; CW[1] = (EOUT[5] + ny * rX + EOUT[8]) * 0.5; CW[2] = (EOUT[6] + nz * rX + EOUT[9]) * 0.5;
      CW[3] = nx; CW[4] = ny; CW[5] = nz; CW[6] = EOUT[3] + rX;
      if (meshFeatureCand(T, triWitnessMask(T))) return;
    }
  }
  // face contact (separated over the face, or the core overlaps the triangle: pushed out along N)
  curFeat = FEAT_FACE; curFA = T.ids[0]; curFB = T.ids[1]; curFC = T.ids[2];
  if (X.kind >= K_BOX) { if (clipFaces(T, 0, X, false) > 0) return; } else if (X.kind === K_SEG) { if (capsuleFace(X, T, 0, true) > 0) return; }
  if (epaOk) { // empty clip: the EPA witness, not the far support
    CW[0] = (EOUT[4] + EOUT[0] * rX + EOUT[7]) * 0.5; CW[1] = (EOUT[5] + EOUT[1] * rX + EOUT[8]) * 0.5; CW[2] = (EOUT[6] + EOUT[2] * rX + EOUT[9]) * 0.5;
    CW[3] = -Nx; CW[4] = -Ny; CW[5] = -Nz; CW[6] = endN <= -FACE_COS ? EOUT[3] + rX : -sepFace;
    pushCand(TAG_MFACE);
    return;
  }
  const k = (rX + hCore) * 0.5;
  CW[0] = sx - Nx * k; CW[1] = sy - Ny * k; CW[2] = sz - Nz * k;
  CW[3] = -Nx; CW[4] = -Ny; CW[5] = -Nz; CW[6] = -sepFace;
  pushCand(TAG_MFACE);
}

function voidHas(n, v) { for (let i = 0; i < n; i++) if (VOIDV[i] === v) return true; return false; }

// Merge the mesh candidates start .. candCount: internal-edge removal, 1 mm dedupe, cap MESH_MAX.
function meshPost(start) {
  const end = candCount;
  if (end - start < 2) return;
  let nv = 0;
  for (let i = start; i < end; i++) {
    if (CFT[i] !== FEAT_FACE) continue;
    for (let c = 0; c < 3; c++) {
      const v = c === 0 ? CFA[i] : c === 1 ? CFB[i] : CFC[i];
      if (!voidHas(nv, v)) { if (nv < VOID_CAP) VOIDV[nv++] = v; else capHits++; }
    }
  }
  for (let i = start; i < end; i++) {
    const f = CFT[i];
    let keep = 1;
    if (f === FEAT_EDGE) { if (voidHas(nv, CFA[i]) && voidHas(nv, CFB[i])) keep = 0; } else if (f === FEAT_VERT) { if (voidHas(nv, CFA[i])) keep = 0; }
    if (keep === 1 && f !== FEAT_FACE) { // on the plane of a triangle that has a face contact: that contact covers it
      const ax = CPL[i * 4], ay = CPL[i * 4 + 1], az = CPL[i * 4 + 2], ad = CPL[i * 4 + 3], px = CPX[i], py = CPY[i], pz = CPZ[i];
      const hi = ax * px + ay * py + az * pz + ad;
      for (let j = start; j < end; j++) {
        if (CFT[j] !== FEAT_FACE) continue;
        const bx = CPL[j * 4], by = CPL[j * 4 + 1], bz = CPL[j * 4 + 2];
        if (ax * bx + ay * by + az * bz < MESH_COPLANAR_COS) continue;
        const dh = hi - (bx * px + by * py + bz * pz + CPL[j * 4 + 3]);
        if (dh <= MESH_COPLANAR_DIST && dh >= -MESH_COPLANAR_DIST) { keep = 0; break; }
      }
    }
    KEEP[i - start] = keep;
  }
  for (let i = start; i < end; i++) {
    if (KEEP[i - start] === 0) continue;
    for (let j = start; j < i; j++) {
      if (KEEP[j - start] === 0) continue;
      const dx = CPX[i] - CPX[j], dy = CPY[i] - CPY[j], dz = CPZ[i] - CPZ[j];
      if (dx * dx + dy * dy + dz * dz < MESH_MERGE2) {
        if (CD[i] > CD[j]) KEEP[j - start] = 0; else { KEEP[i - start] = 0; break; }
      }
    }
  }
  let m = start;
  for (let i = start; i < end; i++) if (KEEP[i - start] === 1) { if (m !== i) copyCand(i, m); m++; }
  candCount = m;
  if (m - start > MESH_MAX) {
    const k = reduceFPS(CPX, CPY, CPZ, CD, start, m - start, MESH_MAX);
    for (let q = 0; q < k; q++) { const i = start + SEL[q]; if (i !== start + q) copyCand(i, start + q); }
    candCount = start + k;
  }
}

// Shape sX at pose oX against the static triangle mesh sM at pose oM (in the mesh frame).
function meshPair(sX, oX, sM, oM) {
  const slot = poseTop++;
  if (poseTop > POSE_SLOTS) { poseTop--; capHits++; return; }
  const bm = oM * 16, bx = oX * 16, b = slot * 16;
  const dx = POSE[bx] - POSE[bm], dy = POSE[bx + 1] - POSE[bm + 1], dz = POSE[bx + 2] - POSE[bm + 2];
  POSE[b] = POSE[bm + 7] * dx + POSE[bm + 10] * dy + POSE[bm + 13] * dz;
  POSE[b + 1] = POSE[bm + 8] * dx + POSE[bm + 11] * dy + POSE[bm + 14] * dz;
  POSE[b + 2] = POSE[bm + 9] * dx + POSE[bm + 12] * dy + POSE[bm + 15] * dz;
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      POSE[b + 7 + i * 3 + j] = POSE[bm + 7 + i] * POSE[bx + 7 + j] + POSE[bm + 10 + i] * POSE[bx + 10 + j] + POSE[bm + 13 + i] * POSE[bx + 13 + j];
    }
  }
  for (let k = 0; k < 9; k++) FR[k] = POSE[bm + 7 + k];
  FR[9] = POSE[bm]; FR[10] = POSE[bm + 1]; FR[11] = POSE[bm + 2];
  FR[12] = POSE[b]; FR[13] = POSE[b + 1]; FR[14] = POSE[b + 2];
  frameWorld = false;
  setProxy(PA, sX, slot);
  proxyAabb(PA);
  const margin = CFG[0], ox = FR[12], oy = FR[13], oz = FR[14];
  const pos = sM.positions, idx = sM.indices;
  let nt = 0;
  if (typeof sM.queryAabb === 'function') {
    nt = sM.queryAabb(AB[0] + ox - margin, AB[1] + oy - margin, AB[2] + oz - margin,
      AB[3] + ox + margin, AB[4] + oy + margin, AB[5] + oz + margin, TRI_IDX);
    if (nt >= TRI_CAP) { nt = TRI_CAP; capHits++; } // a full array may have been truncated
  } else {
    const minX = AB[0] + ox - margin, minY = AB[1] + oy - margin, minZ = AB[2] + oz - margin;
    const maxX = AB[3] + ox + margin, maxY = AB[4] + oy + margin, maxZ = AB[5] + oz + margin;
    const ntri = (idx.length / 3) | 0;
    for (let t = 0; t < ntri; t++) {
      if (nt === TRI_CAP) { capHits++; break; }
      const i0 = idx[t * 3] * 3, i1 = idx[t * 3 + 1] * 3, i2 = idx[t * 3 + 2] * 3;
      let a = pos[i0], c = pos[i1], d = pos[i2];
      if ((a < c ? (a < d ? a : d) : (c < d ? c : d)) > maxX || (a > c ? (a > d ? a : d) : (c > d ? c : d)) < minX) continue;
      a = pos[i0 + 1]; c = pos[i1 + 1]; d = pos[i2 + 1];
      if ((a < c ? (a < d ? a : d) : (c < d ? c : d)) > maxY || (a > c ? (a > d ? a : d) : (c > d ? c : d)) < minY) continue;
      a = pos[i0 + 2]; c = pos[i1 + 2]; d = pos[i2 + 2];
      if ((a < c ? (a < d ? a : d) : (c < d ? c : d)) > maxZ || (a > c ? (a > d ? a : d) : (c > d ? c : d)) < minZ) continue;
      TRI_IDX[nt++] = t;
    }
  }
  const start = candCount, salt0 = curSalt, meshSalt = mixKey(salt0, TAG_MESH), T = PT.t, ids = PT.ids;
  curMeshSalt = meshSalt;
  PT.kind = K_TRI; PT.r = 0;
  for (let q = 0; q < nt; q++) {
    const t = TRI_IDX[q], i0 = idx[t * 3], i1 = idx[t * 3 + 1], i2 = idx[t * 3 + 2];
    T[0] = pos[i0 * 3] - ox; T[1] = pos[i0 * 3 + 1] - oy; T[2] = pos[i0 * 3 + 2] - oz;
    T[3] = pos[i1 * 3] - ox; T[4] = pos[i1 * 3 + 1] - oy; T[5] = pos[i1 * 3 + 2] - oz;
    T[6] = pos[i2 * 3] - ox; T[7] = pos[i2 * 3 + 1] - oy; T[8] = pos[i2 * 3 + 2] - oz;
    const e1x = T[3] - T[0], e1y = T[4] - T[1], e1z = T[5] - T[2], e2x = T[6] - T[0], e2y = T[7] - T[1], e2z = T[8] - T[2];
    const nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
    const l = Math.sqrt(nx * nx + ny * ny + nz * nz);
    if (!(l > 1e-20)) continue;
    T[9] = nx / l; T[10] = ny / l; T[11] = nz / l;
    // world plane of the triangle (for the coplanar ghost-contact rule)
    const wnx = FR[0] * T[9] + FR[1] * T[10] + FR[2] * T[11], wny = FR[3] * T[9] + FR[4] * T[10] + FR[5] * T[11];
    const wnz = FR[6] * T[9] + FR[7] * T[10] + FR[8] * T[11];
    const vx = T[0] + ox, vy = T[1] + oy, vz = T[2] + oz;
    TPL[0] = wnx; TPL[1] = wny; TPL[2] = wnz;
    TPL[3] = -(wnx * (FR[0] * vx + FR[1] * vy + FR[2] * vz + FR[9]) + wny * (FR[3] * vx + FR[4] * vy + FR[5] * vz + FR[10]) +
      wnz * (FR[6] * vx + FR[7] * vy + FR[8] * vz + FR[11]));
    const pp = PT.p;
    pp[0] = (T[0] + T[3] + T[6]) / 3; pp[1] = (T[1] + T[4] + T[7]) / 3; pp[2] = (T[2] + T[5] + T[8]) / 3;
    ids[0] = i0; ids[1] = i1; ids[2] = i2;
    curSalt = mixKey(meshSalt, t);
    meshTri(PA, PT);
  }
  curSalt = salt0; curFeat = FEAT_NONE; curFA = -1; curFB = -1; curFC = -1;
  meshPost(start);
  frameWorld = true; FR[12] = 0; FR[13] = 0; FR[14] = 0;
  poseTop--;
}

// ---- shape dispatch ----------------------------------------------------------------------------------------------------
function collideShapes(sA, oA, sB, oB) {
  const tA = sA.type, tB = sB.type;
  if (tA === SHAPE.COMPOUND || tB === SHAPE.COMPOUND) {
    const aComp = tA === SHAPE.COMPOUND, comp = aComp ? sA : sB, oc = aComp ? oA : oB;
    const other = aComp ? sB : sA, oo = aComp ? oB : oA, ch = comp.children;
    const salt0 = curSalt, slot = poseTop++;
    if (poseTop > POSE_SLOTS) { poseTop--; capHits++; return; }
    for (let i = 0; i < ch.length; i++) {
      const c = ch[i];
      childPose(oc, c, slot);
      if (apart(c.shape, slot, other, oo)) continue;
      curSalt = mixKey(salt0, (aComp ? TAG_CHILD : TAG_CHILD + 0x100) + i * 0x10000);
      if (aComp) collideShapes(c.shape, slot, other, oo); else collideShapes(other, oo, c.shape, slot);
    }
    curSalt = salt0;
    poseTop--;
    return;
  }
  if (tA === SHAPE.TRIMESH) {
    if (tB === SHAPE.TRIMESH) return;
    curFlip = !curFlip; meshPair(sB, oB, sA, oA); curFlip = !curFlip;
    return;
  }
  if (tB === SHAPE.TRIMESH) { meshPair(sA, oA, sB, oB); return; }
  const a = oA * 16;
  FR[12] = POSE[a]; FR[13] = POSE[a + 1]; FR[14] = POSE[a + 2]; // work frame centred at A (precision)
  setProxy(PA, sA, oA);
  setProxy(PB, sB, oB);
  leafPair(PA, PB);
  FR[12] = 0; FR[13] = 0; FR[14] = 0;
}

// ---- heightfield -------------------------------------------------------------------------------------------------------
// Ground height and normal at HS[0] = x, HS[1] = z (HS[2] = y hint) -> HS[3] = h, HS[4..6] = N. Three heightAt calls:
// one-sided differences over the step e = CFG[1], N = normalize((h - h(x + e)) / e, 1, (h - h(z + e)) / e).
function groundSample(heightAt) {
  const x = HS[0], z = HS[1], y = HS[2], e = CFG[1];
  const h = heightAt(x, z, y);
  const gx = (h - heightAt(x + e, z, y)) / e, gz = (h - heightAt(x, z + e, y)) / e;
  const inv = 1 / Math.sqrt(gx * gx + 1 + gz * gz);
  HS[3] = h; HS[4] = gx * inv; HS[5] = inv; HS[6] = gz * inv;
}
// sphere (or capsule end) centre HQ[0..2], radius HQ[3], ground normal at the centre HQ[4..6]; the centre's sample
// (HC) is reused when the bottom point lies straight below it
function hfRound(heightAt, key) {
  const cx = HQ[0], cy = HQ[1], cz = HQ[2], r = HQ[3];
  const bx = cx - HQ[4] * r, by = cy - HQ[5] * r, bz = cz - HQ[6] * r;
  const ex = bx - HC[4], ez = bz - HC[5];
  if (ex < 1e-12 && ex > -1e-12 && ez < 1e-12 && ez > -1e-12) { HS[3] = HC[0]; HS[4] = HC[1]; HS[5] = HC[2]; HS[6] = HC[3]; } else {
    HS[0] = bx; HS[1] = bz; HS[2] = by;
    groundSample(heightAt);
  }
  const g = HS[3], Nx = HS[4], Ny = HS[5], Nz = HS[6];
  const sc = Nx * (cx - bx) + Ny * (cy - g) + Nz * (cz - bz); // centre above the local ground plane
  const depth = r - sc;
  if (depth < -CFG[0]) return;
  const k = (r + sc) * 0.5;
  CW[0] = cx - Nx * k; CW[1] = cy - Ny * k; CW[2] = cz - Nz * k;
  CW[3] = -Nx; CW[4] = -Ny; CW[5] = -Nz; CW[6] = depth;
  pushCand(key);
}
// vertex HQ[0..2]
function hfVertex(heightAt, key) {
  const vx = HQ[0], vy = HQ[1], vz = HQ[2];
  HS[0] = vx; HS[1] = vz; HS[2] = vy;
  groundSample(heightAt);
  const Nx = HS[4], Ny = HS[5], Nz = HS[6];
  const depth = Ny * (HS[3] - vy);
  if (depth < -CFG[0]) return;
  const k = depth * 0.5;
  CW[0] = vx + Nx * k; CW[1] = vy + Ny * k; CW[2] = vz + Nz * k;
  CW[3] = -Nx; CW[4] = -Ny; CW[5] = -Nz; CW[6] = depth;
  pushCand(key);
}
// keep the HF_MAX lowest vertices (along -N0) in HV / HD / HI, ascending; returns the new count
function hfInsert(n, x, y, z, d, id) {
  if (n === HF_MAX && !(d < HD[n - 1])) return n;
  let i = n === HF_MAX ? n - 1 : n;
  while (i > 0 && HD[i - 1] > d) {
    HD[i] = HD[i - 1]; HI[i] = HI[i - 1];
    HV[i * 3] = HV[i * 3 - 3]; HV[i * 3 + 1] = HV[i * 3 - 2]; HV[i * 3 + 2] = HV[i * 3 - 1];
    i--;
  }
  HD[i] = d; HI[i] = id; HV[i * 3] = x; HV[i * 3 + 1] = y; HV[i * 3 + 2] = z;
  return n === HF_MAX ? n : n + 1;
}
function hfShape(s, o, heightAt) {
  const t = s.type;
  if (t === SHAPE.TRIMESH) return;
  if (t === SHAPE.COMPOUND) {
    const ch = s.children, salt0 = curSalt, slot = poseTop++;
    if (poseTop > POSE_SLOTS) { poseTop--; capHits++; return; }
    for (let i = 0; i < ch.length; i++) {
      childPose(o, ch[i], slot);
      curSalt = mixKey(salt0, TAG_CHILD + i * 0x10000);
      hfShape(ch[i].shape, slot, heightAt);
    }
    curSalt = salt0; poseTop--;
    return;
  }
  const b = o * 16, cx = POSE[b], cy = POSE[b + 1], cz = POSE[b + 2];
  HS[0] = cx; HS[1] = cz; HS[2] = cy;
  groundSample(heightAt);
  const n0x = HS[4], n0y = HS[5], n0z = HS[6];
  HC[0] = HS[3]; HC[1] = n0x; HC[2] = n0y; HC[3] = n0z; HC[4] = cx; HC[5] = cz;
  if (t === SHAPE.SPHERE || t === SHAPE.CAPSULE) {
    HQ[3] = s.radius; HQ[4] = n0x; HQ[5] = n0y; HQ[6] = n0z;
    if (t === SHAPE.SPHERE) { HQ[0] = cx; HQ[1] = cy; HQ[2] = cz; hfRound(heightAt, 0); return; }
    const hh = s.halfHeight, ax = POSE[b + 8] * hh, ay = POSE[b + 11] * hh, az = POSE[b + 14] * hh;
    HQ[0] = cx - ax; HQ[1] = cy - ay; HQ[2] = cz - az; hfRound(heightAt, 0);
    HQ[0] = cx + ax; HQ[1] = cy + ay; HQ[2] = cz + az; HQ[3] = s.radius; HQ[4] = n0x; HQ[5] = n0y; HQ[6] = n0z; hfRound(heightAt, 1);
    return;
  }
  // box corners / hull vertices below the centre along N0, deepest first
  let n = 0;
  const pts = t === SHAPE.BOX ? null : s.points, nv = t === SHAPE.BOX ? 8 : (s.points.length / 3) | 0;
  for (let i = 0; i < nv; i++) {
    let x, y, z;
    if (pts === null) {
      const h = s.half;
      x = (i & 1) !== 0 ? h[0] : -h[0]; y = (i & 2) !== 0 ? h[1] : -h[1]; z = (i & 4) !== 0 ? h[2] : -h[2];
    } else { x = pts[i * 3]; y = pts[i * 3 + 1]; z = pts[i * 3 + 2]; }
    const wx = POSE[b + 7] * x + POSE[b + 8] * y + POSE[b + 9] * z, wy = POSE[b + 10] * x + POSE[b + 11] * y + POSE[b + 12] * z;
    const wz = POSE[b + 13] * x + POSE[b + 14] * y + POSE[b + 15] * z;
    const d = n0x * wx + n0y * wy + n0z * wz;
    if (d < 0) n = hfInsert(n, cx + wx, cy + wy, cz + wz, d, i);
  }
  for (let k = 0; k < n; k++) { HQ[0] = HV[k * 3]; HQ[1] = HV[k * 3 + 1]; HQ[2] = HV[k * 3 + 2]; hfVertex(heightAt, HI[k]); }
}

// ---- per-call setup / output ---------------------------------------------------------------------------------------------
function begin(np, a, b, pk, staticIndex) {
  curNP = np; curA = a; curB = b; curPair = pk; curStatic = staticIndex;
  CFG[0] = np.margin;
  candCount = 0; candOverflow = 0; capHits = 0; curFlip = false; curSalt = 0; frameWorld = true; poseTop = 3;
  curFeat = FEAT_NONE; curFA = -1; curFB = -1; curFC = -1;
  FR[12] = 0; FR[13] = 0; FR[14] = 0;
}

// does the (uint32) pair key involve body id? (world pairs: 0xFFFF0000 + id)
function pairHasBody(pk, id) { return pk >= 0xffff0000 ? pk - 0xffff0000 === id : pairA(pk) === id || pairB(pk) === id; }

function tableFind(np, pk, key) {
  const mask = np._mask, gen = np._gen, st = np._stamp, kp = np._kPair, kk = np._kKey;
  let i = hashSlot(pk, key) & mask;
  for (let probe = 0; probe <= mask; probe++) {
    if (st[i] !== gen) return -1;
    if (kp[i] === pk && kk[i] === key) return i;
    i = (i + 1) & mask;
  }
  return -1;
}

// Candidates -> contacts, written straight into the ContactBuffer arrays (no add() call: a non-inlined call would box
// the doubles), lambdas prefilled from the persistent table; returns the number appended.
function flush(np, contacts) {
  const mc = np.maxContacts, cc = contacts.capacity, cap = mc < cc ? mc : cc;
  const A = contacts.a, Bs = contacts.b, X = contacts.px, Y = contacts.py, Z = contacts.pz;
  const NX = contacts.nx, NY = contacts.ny, NZ = contacts.nz, D = contacts.depth, K = contacts.key, PK = contacts.pairKey;
  const LN = contacts.lambdaN, LT = contacts.lambdaT, SI = contacts.staticIndex, tln = np._ln, tlt = np._lt;
  let i = contacts.count;
  const i0 = i;
  if (candOverflow > 0) contacts.overflow += candOverflow;
  for (let c = 0; c < candCount; c++) {
    if (i >= cap) { contacts.overflow += candCount - c; break; }
    const key = CK[c];
    A[i] = curA; Bs[i] = curB; X[i] = CPX[c]; Y[i] = CPY[c]; Z[i] = CPZ[c]; NX[i] = CNX[c]; NY[i] = CNY[c]; NZ[i] = CNZ[c];
    D[i] = CD[c]; K[i] = key >>> 0; PK[i] = curPair >>> 0; SI[i] = curStatic;
    const slot = tableFind(np, curPair, key);
    if (slot >= 0) { LN[i] = tln[slot]; LT[i] = tlt[slot]; } else { LN[i] = 0; LT[i] = 0; }
    i++;
  }
  contacts.count = i;
  candCount = 0;
  np.stats.contacts += i - i0;
  np.stats.capped += capHits;
  return i - i0;
}

/**
 * Contact generation between bodies (see the file header for the algorithms and conventions) with a persistent
 * warm-start cache.
 */
export class Narrowphase {
  /**
   * @param {{ maxContacts?: number, margin?: number, tableSize?: number }} [o]
   *   maxContacts: contacts this narrowphase appends to a buffer (beyond it they are dropped and counted in
   *   contacts.overflow); margin: speculative contact distance (m); tableSize: persistent table slots (rounded up to a
   *   power of two; keep it >= 2 x the contacts per step); timing: accumulate stats.ms (performance.now() twice
   *   per call, which can allocate in Node; DynamicsWorld times the whole pass itself); hfStep: heightfield
   *   finite-difference step (m), or hfCell: the heightfield's cell size, used as the step when given.
   */
  constructor({ maxContacts = 16384, margin = 0.01, tableSize = 65536, timing = false, hfStep = 0.5, hfCell = 0 } = {}) {
    this.maxContacts = maxContacts;
    this.margin = margin;
    this.timing = timing;
    /** Heightfield finite-difference step (m). */
    this.hfStep = hfCell > 0 ? hfCell : hfStep;
    let size = 16;
    while (size < tableSize) size *= 2;
    this.tableSize = size;
    this._mask = size - 1;
    this._kPair = new Int32Array(size);
    this._kKey = new Int32Array(size);
    this._ln = new Float64Array(size);
    this._lt = new Float64Array(size);
    this._stamp = new Uint32Array(size);
    this._gen = 1;
    /** Reused counters of the current step: pairs tested, contacts emitted, GJK / EPA / box-box SAT runs, ms spent
     * (while timing), capped: silent capacity limits hit (face vertices, clip polygon, EPA horizon, mesh triangles,
     * voided vertices, compound depth, full cache table). */
    this.stats = { pairs: 0, contacts: 0, gjk: 0, epa: 0, sat: 0, ms: 0, capped: 0 };
  }

  /** Start of a step: resets the stats (the cache keeps last step's entries until endStep replaces them). */
  beginStep() {
    const s = this.stats;
    s.pairs = 0; s.contacts = 0; s.gjk = 0; s.epa = 0; s.sat = 0; s.ms = 0; s.capped = 0;
  }

  /**
   * Contacts of body a against body b (shapes from bodies.shape, poses from bodies.pos / quat) appended to
   * `contacts`, normals from a to b. Returns the number appended.
   */
  collide(bodies, a, b, contacts) {
    const t0 = this.timing ? performance.now() : 0;
    this.stats.pairs++;
    const sA = bodies.shape[a], sB = bodies.shape[b];
    if (!sA || !sB) return 0;
    begin(this, a, b, pairKey(a, b) | 0, -1);
    poseFromBody(bodies, a, 0);
    poseFromBody(bodies, b, 1);
    if (!apart(sA, 0, sB, 1)) collideShapes(sA, 0, sB, 1);
    const n = flush(this, contacts);
    if (this.timing) this.stats.ms += performance.now() - t0;
    return n;
  }

  /**
   * Contacts of body a against the static world box [min, max] (identity rotation), b = WORLD_BODY, normals from a
   * into the box; staticIndex is stored on the contacts and hashed into their keys. Returns the number appended.
   */
  collideStaticBox(bodies, a, minX, minY, minZ, maxX, maxY, maxZ, staticIndex, contacts) {
    const t0 = this.timing ? performance.now() : 0;
    this.stats.pairs++;
    const s = bodies.shape[a];
    if (!s || s.type === SHAPE.TRIMESH) return 0;
    begin(this, a, WORLD_BODY, pairKey(a, WORLD_BODY) | 0, staticIndex | 0);
    poseFromBody(bodies, a, 0);
    const o = 32, h = STATIC_BOX.half;
    h[0] = (maxX - minX) * 0.5; h[1] = (maxY - minY) * 0.5; h[2] = (maxZ - minZ) * 0.5;
    STATIC_BOX.boundingRadius = Math.sqrt(h[0] * h[0] + h[1] * h[1] + h[2] * h[2]);
    POSE[o] = (minX + maxX) * 0.5; POSE[o + 1] = (minY + maxY) * 0.5; POSE[o + 2] = (minZ + maxZ) * 0.5;
    POSE[o + 3] = 0; POSE[o + 4] = 0; POSE[o + 5] = 0; POSE[o + 6] = 1;
    quatToMat(o);
    curSalt = mixKey(TAG_STATIC, staticIndex | 0);
    if (!apart(s, 0, STATIC_BOX, 2)) collideShapes(s, 0, STATIC_BOX, 2);
    const n = flush(this, contacts);
    if (this.timing) this.stats.ms += performance.now() - t0;
    return n;
  }

  /**
   * Contacts of body a against the ground heightAt(x, z, y) -> height (b = WORLD_BODY, staticIndex -1), one per
   * sample within the margin (see the header). Returns the number appended.
   */
  collideHeightfield(bodies, a, heightAt, contacts) {
    const t0 = this.timing ? performance.now() : 0;
    this.stats.pairs++;
    const s = bodies.shape[a];
    if (!s || typeof heightAt !== 'function') return 0;
    begin(this, a, WORLD_BODY, pairKey(a, WORLD_BODY) | 0, -1);
    CFG[1] = this.hfStep;
    poseFromBody(bodies, a, 0);
    curSalt = TAG_HF;
    hfShape(s, 0, heightAt);
    const n = flush(this, contacts);
    if (this.timing) this.stats.ms += performance.now() - t0;
    return n;
  }

  /** Store this step's contact impulses (contacts.lambdaN / lambdaT, written by the solver) keyed (pairKey, key);
   * entries not stored now are evicted. */
  endStep(contacts) {
    let gen = this._gen + 1;
    if (gen > 0xffffffff) { this._stamp.fill(0); gen = 1; }
    this._gen = gen;
    const mask = this._mask, st = this._stamp, kp = this._kPair, kk = this._kKey, ln = this._ln, lt = this._lt;
    const n = contacts.count, cpk = contacts.pairKey, ck = contacts.key, cln = contacts.lambdaN, clt = contacts.lambdaT;
    for (let i = 0; i < n; i++) {
      const pk = cpk[i] | 0, key = ck[i] | 0;
      let s = hashSlot(pk, key) & mask, probe = 0;
      while (st[s] === gen && !(kp[s] === pk && kk[s] === key)) {
        if (++probe > mask) { this.stats.capped++; return; } // table full
        s = (s + 1) & mask;
      }
      st[s] = gen; kp[s] = pk; kk[s] = key; ln[s] = cln[i]; lt[s] = clt[i];
    }
  }

  /**
   * Drop every cache entry of body `id` (its pairs with other bodies and with the static world), so a reused id
   * starts without warm-start impulses. O(tableSize): call it from removeBody, not per step. Backward-shift deletion
   * keeps the other entries' probe chains intact. Returns the number of entries dropped.
   */
  evictBody(id) {
    const mask = this._mask, st = this._stamp, kp = this._kPair, kk = this._kKey, ln = this._ln, lt = this._lt, g = this._gen;
    let dropped = 0;
    for (let s = 0; s <= mask; s++) {
      while (st[s] === g && pairHasBody(kp[s] >>> 0, id)) {
        dropped++;
        let hole = s, j = s;
        for (;;) {
          j = (j + 1) & mask;
          if (st[j] !== g) break;
          const home = hashSlot(kp[j], kk[j]) & mask; // entry j stays when its home lies cyclically in (hole, j]
          if (hole <= j ? home > hole && home <= j : home > hole || home <= j) continue;
          kp[hole] = kp[j]; kk[hole] = kk[j]; ln[hole] = ln[j]; lt[hole] = lt[j];
          hole = j;
        }
        st[hole] = 0;
      }
    }
    return dropped;
  }

  /** Number of live cache entries (diagnostic, O(tableSize)). */
  get cached() { let c = 0; const st = this._stamp, g = this._gen; for (let i = 0; i < st.length; i++) if (st[i] === g) c++; return c; }

  /** Live cache entries as plain arrays (slot order), for DynamicsWorld.serialize. */
  serialize() {
    const pk = [], key = [], lambdaN = [], lambdaT = [];
    const st = this._stamp, g = this._gen;
    for (let i = 0; i < st.length; i++) {
      if (st[i] !== g) continue;
      pk.push(this._kPair[i] >>> 0); key.push(this._kKey[i] >>> 0); lambdaN.push(this._ln[i]); lambdaT.push(this._lt[i]);
    }
    return { pairKey: pk, key, lambdaN, lambdaT };
  }

  /** Replace the cache with serialize()'s entries. */
  restore(s) {
    this._stamp.fill(0);
    this._gen = 1;
    if (!s || !s.pairKey) return;
    const n = s.pairKey.length, mask = this._mask;
    for (let i = 0; i < n; i++) {
      const pk = s.pairKey[i] | 0, key = s.key[i] | 0;
      let slot = hashSlot(pk, key) & mask, probe = 0;
      while (this._stamp[slot] === 1 && !(this._kPair[slot] === pk && this._kKey[slot] === key)) {
        if (++probe > mask) return;
        slot = (slot + 1) & mask;
      }
      this._stamp[slot] = 1; this._kPair[slot] = pk; this._kKey[slot] = key; this._ln[slot] = +s.lambdaN[i]; this._lt[slot] = +s.lambdaT[i];
    }
  }

  /** Forget the cache. */
  dispose() { this._stamp.fill(0); this._gen = 1; }
}
