// Ragdoll: capsule bodies and limited joints built from a humanoid skeleton, simulated by a DynamicsWorld (or an
// in-thread DynamicsClient) and written back onto the bones (pure JS over three's math, Node-safe, no DOM).
//
//   import { Ragdoll } from '@vibe/engine/anim';                                  // (src/anim/Ragdoll.js)
//   const rd = Ragdoll.fromSkeleton(skinnedMeshOrRoot, dyn, { bones: 'auto' });   // at load: maps bones, sizes capsules
//   on death:            rd.activate({ velocity: hitVelocity });   // bodies + joints from the bones' current pose
//   per fixed step:      dyn.step(dt); rd.update();                 // bones follow the bodies (after the step)
//   on getting up:       rd.deactivate({ controller: anim });       // bodies gone, the mixer drives the bones again
//   rd.position(out) -> out (hips bone origin);  rd.isResting({ speed = 0.1 }) -> bool;  rd.dispose()
//
// fromSkeleton(skinnedMeshOrRoot, dyn, opts): the candidates are the SkinnedMesh's skeleton bones, else the THREE.Bones
// under the root (else every descendant). opts:
//   bones = 'auto' | { hips, spine, chest, neck, head, upperArmL, lowerArmL, handL, upperArmR, lowerArmR, handR,
//     upperLegL, lowerLegL, footL, upperLegR, lowerLegR, footR } (bone names, exact then case-insensitive, or Object3Ds;
//     an explicit map is used as given, with no auto fill-in). Only hips is required.
//   'auto': names are split into words (camelCase, digits, separators: 'mixamorig:LeftForeArm' -> mixamorig left fore
//     arm; 'UpperArm.L' / 'upperArmL' -> upper arm l), prefixes (mixamorig, bip, def, org, ...) and side words (l / r /
//     left / right) dropped, and names with end / top / tip / nub / twist / roll / ik / pole / target words skipped
//     (leaf markers, helpers). Keys (first listed wins, then the bone nearest the root): hips | pelvis | hip (no side);
//     neck; head; upperArm: upperarm | arm | uparm | humerus; lowerArm: lowerarm | forearm | loarm | elbow; hand: hand |
//     wrist | palm2 | palm (RobotExpressive has three palms and no hand: the middle one); upperLeg: upperleg | upleg |
//     thigh | femur | hip (with a side); lowerLeg: lowerleg | leg | calf | shin | knee; foot: foot | ankle. Clavicles
//     (shoulder / clavicle / collar) stay unmapped (they follow the chest). Spine words (spine, abdomen, torso, chest,
//     upperchest, waist, ...): chest = the deepest of them above the neck (or head / upper arms), spine = the one
//     nearest the hips. This matches the procedural rig (hips, spine, chest, neck, head, upperArmL, lowerArmL, handL,
//     upperLegL, lowerLegL, footL, ...R), RobotExpressive (Hips, Abdomen, Torso, Neck, Head, UpperArm.L, LowerArm.L,
//     Palm2.L, UpperLeg.L, LowerLeg.L, Foot.L) and mixamo / UE / Rigify-style names.
//   density = 985 (kg/m^3; mass = density x capsule volume), radius = 'auto' | number | { [segment or role]: m },
//   length = { [segment or role]: m } (leaf segments), limits = DEFAULT_LIMITS (merged per joint and field:
//   { shoulder: { swing: 0.5 } } keeps the default twist; { swing, twist } makes a BALL joint, [lo, hi] a HINGE, null a
//   free ball), friction = 0.6, restitution = 0, linearDamping = 0.05, angularDamping = 0.1, jointDamping = 4 (1/s,
//   joints.js damping: relative velocity at every joint decays at this rate; it keeps settled limbs from creeping,
//   since spheres / capsules rolling on the ground gain spin in the current solver (its contact anchors ride round with
//   the body within a step); 1-2 is enough once DynamicsWorld re-seats them per substep; 0 = a limp doll), layer =
//   layerOf('ragdoll'), mask = ~layer (no collisions between ragdoll bodies, its own included), kind = 'ragdoll',
//   userData = null (on every body).
//
// Segments (rd.segments, parents first): one per mapped bone, named by its role (hips, spine, ..., footR), a capsule
// from the bone's origin to its chain child's origin (hips -> spine | chest | neck | head; spine -> chest | neck |
// head; chest -> neck | head; neck -> head; upperArm -> lowerArm | hand; lowerArm -> hand; upperLeg -> lowerLeg | foot;
// lowerLeg -> foot; the chain is the role map, not the bone tree, so a foot parented to the rig root works). Leaves
// (head, hands, feet, and a lower limb with nothing mapped below) get `length` or the default (head 0.18, hand 0.16,
// foot 0.22, lowerArm 0.26, lowerLeg 0.42 m, x the leg length / 0.85 m when the legs are mapped) along the direction to
// the centroid of the bone's own children (toes, finger roots, a head-top end) or, without children, the bone's local
// axis nearest to the character's forward direction (feet) or to the parent -> bone direction (the rest).
// Radius 'auto': 0.22 x length clamped to 0.04 .. 0.12, except the head (0.4 x length: a short capsule), the neck (0.5
// x length: no tiny link between chest and head), the hips (at least half the hip-joint spacing), the chest (at least
// a quarter of the shoulder spacing) and the spine (at least the mean of the two), so the torso has a body's volume. halfHeight = max(0, length / 2 - radius): the capsule spans the
// segment (a short thick one is a ball around its middle). Each segment: { name, role, side, bone, parent (index or
// -1), body (id, -1 until active), joint (id or -1), jointType ('ball' | 'hinge' | null for the root), length,
// radius, halfHeight, mass, offset (Vector3: bone -> capsule centre in the bone's local frame, scale included),
// rotation (Quaternion: bone frame -> capsule frame; the capsule's local +Y points along the segment) }.
//
// Rest pose and joints. The pose the bones have when fromSkeleton runs is the REST pose: build the ragdoll before an
// animation moves the skeleton (or after skeleton.pose()). activate() creates the bodies at that rest pose, adds the
// joints there (so every joint angle is measured from the rest pose: joints.js freezes its reference frames at
// creation), then teleports each body to its bone's CURRENT pose (fromPose = true) or to the rest pose carried by the
// hips' current transform (fromPose = false), first bringing that pose inside the joints (parents first: a child's
// subtree is moved onto its parent's anchor and turned back to the swing cone / twist range / hinge axis, so a pose
// outside the limits, e.g. a jump pose, does not whip the doll in the first step). Joints connect each segment (bodyA
// = the child) to its parent segment
// (bodyB) at the child bone's origin. BALL: axisA = the child capsule's +Y, swing = the cone angle between that axis
// and its rest direction in the parent, twist = rotation about it. HINGE (elbows, knees): the axis is chosen so a
// positive joint angle (joints.js phi: the child's right-handed rotation about the axis relative to the parent) is
// FLEXION, so the limits read anatomically, [0, 2.4] for both elbows and both knees, mirrored per side by the axis:
//   knee:  axis = forward x shin direction (the character's left, +X for a +Z-facing character): flexion swings the
//          foot backwards (the procedural rig's "+X bends a knee");
//   elbow: axis = forearm direction x forward (left arm hanging: -X, as the rig's "-X bends an elbow forward"; T-pose
//          left arm: -Y, right arm: +Y, like samples/sandbox's ELBOW axis (0, -side, 0)): flexion swings the hand
//          forwards;
//   a limb already bent at rest uses its bend-plane normal (upper x lower direction) with the same sign.
// forward = (left - right hip (or shoulder) joint) x up (hips -> neck), i.e. +Z for the usual Y-up +Z-facing rigs. Any
// other joint given as [lo, hi] is a hinge whose positive angle swings the segment's far end forwards (a hip hinge
// lifts the thigh, a spine hinge bends forwards).
//
// update() (after dyn.step, before rendering): each segment's bone world matrix = body pose x inverse(rotation) with the
// offset undone (scale as the bone had at activation), written as the bone's LOCAL matrix (parent world inverse x that)
// and its matrixWorld set directly, parents first, hips included. While active the matrices are authoritative: the
// bones' position / quaternion / scale keep their old values until deactivate() sets them from the last pose
// (decomposing every call also updates each bone's Euler `rotation`: measured ~210 B allocated per update() for 15
// bones, against ~2 B without). The skeleton's root Object3D never moves, so set
// skinnedMesh.frustumCulled = false (or give the geometry a large boundingSphere), or the mesh is culled once the doll
// leaves the bind-pose bounds. Unmapped bones (clavicles, fingers, toes, spine links between mapped ones) keep their
// local transforms and follow their parents. Allocation-free, no trig. Round trip: activate() with the bones at rest,
// then update(), leaves every bone's world matrix where it was (within float rounding).
// deactivate({ controller }): removes the joints and bodies, writes the last pose into the bones' position /
// quaternion / scale, restores matrixAutoUpdate and calls controller.resync() (AnimationController): a THREE
// PropertyMixer writes a channel only when its blended value changed since its last write, so without the resync a
// channel whose clip value is constant (or unchanged since the ragdoll began) keeps the ragdoll's pose. Without a
// controller, resync it yourself. Channels no clip animates keep the ragdoll's pose either way.
//
// Limits: no collisions between ragdoll bodies unless `mask` includes the ragdoll layer (jointed pairs never collide);
// no powered ragdoll / motors / animation blending; the mapped bones must form a hierarchy in which each mapped bone's
// offset to its segment parent's bone does not change with animation (bones animated by translation, other than the
// hips, pop to their rest offset on activate); the root (hips) is required. Worker-mode DynamicsClient works for
// activate / update (poses one or two steps behind); isResting reads its snapshot.
import * as THREE from 'three';
import { Vector3, Quaternion, Matrix4 } from 'three';
import { capsule } from '../physics/dynamics/shapes.js';
import { jointAtan2 } from '../physics/dynamics/joints.js';
import { layerOf } from '../physics/PhysicsWorld.js';

/** Joint limits by joint name (the child segment's joint): { swing, twist } -> BALL, [lo, hi] -> HINGE (flexion). */
export const DEFAULT_LIMITS = Object.freeze({
  spine: Object.freeze({ swing: 0.4, twist: 0.3 }),
  chest: Object.freeze({ swing: 0.4, twist: 0.3 }),
  neck: Object.freeze({ swing: 0.7, twist: 0.8 }),
  head: Object.freeze({ swing: 0.4, twist: 0.5 }),
  shoulder: Object.freeze({ swing: 1.5, twist: 0.8 }),
  elbow: Object.freeze([0, 2.4]),
  wrist: Object.freeze({ swing: 1.0, twist: 0.3 }),
  hip: Object.freeze({ swing: 1.0, twist: 0.5 }),
  knee: Object.freeze([0, 2.4]),
  ankle: Object.freeze({ swing: 0.6, twist: 0.2 }),
});

/** Segment names in creation order (parents first). */
export const RAGDOLL_SEGMENTS = Object.freeze(['hips', 'spine', 'chest', 'neck', 'head', 'upperArmL', 'lowerArmL', 'handL',
  'upperArmR', 'lowerArmR', 'handR', 'upperLegL', 'lowerLegL', 'footL', 'upperLegR', 'lowerLegR', 'footR']);

// role -> joint name (DEFAULT_LIMITS key), parent candidates, chain-child candidates ('*' = same side), leaf length (m)
const ROLE = {
  hips: { joint: null, parents: [], children: ['spine', 'chest', 'neck', 'head'], leaf: 0.15 },
  spine: { joint: 'spine', parents: ['hips'], children: ['chest', 'neck', 'head'], leaf: 0.2 },
  chest: { joint: 'chest', parents: ['spine', 'hips'], children: ['neck', 'head'], leaf: 0.25 },
  neck: { joint: 'neck', parents: ['chest', 'spine', 'hips'], children: ['head'], leaf: 0.1 },
  head: { joint: 'head', parents: ['neck', 'chest', 'spine', 'hips'], children: [], leaf: 0.18 },
  upperArm: { joint: 'shoulder', parents: ['chest', 'spine', 'hips'], children: ['lowerArm*', 'hand*'], leaf: 0.28 },
  lowerArm: { joint: 'elbow', parents: ['upperArm*', 'chest', 'spine', 'hips'], children: ['hand*'], leaf: 0.26 },
  hand: { joint: 'wrist', parents: ['lowerArm*', 'upperArm*', 'chest', 'spine', 'hips'], children: [], leaf: 0.16 },
  upperLeg: { joint: 'hip', parents: ['hips'], children: ['lowerLeg*', 'foot*'], leaf: 0.44 },
  lowerLeg: { joint: 'knee', parents: ['upperLeg*', 'hips'], children: ['foot*'], leaf: 0.42 },
  foot: { joint: 'ankle', parents: ['lowerLeg*', 'upperLeg*', 'hips'], children: [], leaf: 0.22 },
};
// hinges flex the far end towards the character's back (knees) or front (everything else); a hinge's upper segment
const BACKWARD = { lowerLeg: 1 };
const HINGE_UPPER = { lowerArm: 'upperArm', lowerLeg: 'upperLeg' };

// ---- name heuristics ----------------------------------------------------------------------------------------------
const SIDED = {
  upperArm: ['upperarm', 'arm', 'uparm', 'humerus'],
  lowerArm: ['lowerarm', 'forearm', 'loarm', 'elbow'],
  hand: ['hand', 'wrist', 'palm2', 'palm'],
  upperLeg: ['upperleg', 'upleg', 'thigh', 'femur', 'hip'],
  lowerLeg: ['lowerleg', 'leg', 'calf', 'shin', 'knee', 'loleg', 'crus'],
  foot: ['foot', 'ankle'],
};
const CENTRE = { hips: ['hips', 'pelvis', 'hip'], neck: ['neck'], head: ['head'] };
const SPINE = ['spine', 'abdomen', 'torso', 'chest', 'upperchest', 'stomach', 'belly', 'waist', 'lowerback', 'upperback', 'back'];
const PREFIX = new Set(['mixamorig', 'mixamo', 'bip', 'def', 'org', 'mch', 'jnt', 'joint', 'bone', 'armature', 'rig', 'skeleton', 'character', 'cc', 'base']);
const SKIP = new Set(['end', 'top', 'tip', 'nub', 'twist', 'roll', 'ik', 'pole', 'target', 'ctrl', 'control', 'helper', 'socket', 'effector', 'null']);
const KNOWN = new Set([...Object.values(SIDED).flat(), ...Object.values(CENTRE).flat(), ...SPINE]);

/** Words of a bone name: camelCase, letter / digit boundaries and separators split, lower case. */
function words(name) {
  return String(name || '')
    .replace(/([a-z])([A-Z])/g, '$1 $2').replace(/([A-Z])([A-Z][a-z])/g, '$1 $2')
    .replace(/([A-Za-z])([0-9])/g, '$1 $2').replace(/([0-9])([A-Za-z])/g, '$1 $2')
    .toLowerCase().split(/[^a-z0-9]+/).filter(Boolean);
}
/** { side: 'L' | 'R' | '', key (no digits), key2 (with digits) } or null (helper / leaf marker / nothing left). */
function parseBoneName(name) {
  let side = '';
  const keep = [], all = [];
  for (const w of words(name)) {
    if (SKIP.has(w)) return null;
    if (w === 'l' || w === 'left') { side = 'L'; continue; }
    if (w === 'r' || w === 'right') { side = 'R'; continue; }
    if (PREFIX.has(w)) continue;
    all.push(w);
    if (!/^[0-9]+$/.test(w)) keep.push(w);
  }
  let key = keep.join(''), key2 = all.join('');
  if (!KNOWN.has(key)) { // fused side: 'leftarm', 'armleft', 'upperarml'
    for (const [s, pre] of [['L', 'left'], ['R', 'right']]) {
      if (key.startsWith(pre) && KNOWN.has(key.slice(pre.length))) { side = s; key = key.slice(pre.length); key2 = key2.slice(pre.length); }
      else if (key.endsWith(pre) && KNOWN.has(key.slice(0, -pre.length))) { side = s; key = key.slice(0, -pre.length); }
    }
    const last = key[key.length - 1];
    if (!KNOWN.has(key) && (last === 'l' || last === 'r') && KNOWN.has(key.slice(0, -1))) { side = last === 'l' ? 'L' : 'R'; key = key.slice(0, -1); }
  }
  return key ? { side, key, key2 } : null;
}
function depthOf(o) { let d = 0; for (let p = o.parent; p; p = p.parent) d++; return d; }
function isAncestor(a, o) { for (let p = o && o.parent; p; p = p.parent) if (p === a) return true; return false; }

/** Map role names to bones by name (see the header). Returns { [segmentName]: Object3D }. */
export function mapBonesAuto(candidates) {
  const parsed = [];
  candidates.forEach((b, order) => { const n = parseBoneName(b.name); if (n) parsed.push({ b, order, depth: depthOf(b), ...n }); });
  const rank = (p, keys) => { let i = keys.indexOf(p.key2); if (i < 0) i = keys.indexOf(p.key); return i; };
  const pick = (keys, side, ok = null) => {
    let best = null, br = Infinity;
    for (const p of parsed) {
      if (p.side !== side || (ok && !ok(p.b))) continue;
      const r = rank(p, keys);
      if (r < 0) continue;
      if (r < br || (r === br && (p.depth < best.depth || (p.depth === best.depth && p.order < best.order)))) { best = p; br = r; }
    }
    return best ? best.b : null;
  };
  const map = {};
  map.hips = pick(CENTRE.hips, '');
  const under = (o) => !map.hips || isAncestor(map.hips, o);
  map.neck = pick(CENTRE.neck, '', under);
  map.head = pick(CENTRE.head, '', under);
  for (const side of ['L', 'R']) for (const role of Object.keys(SIDED)) map[role + side] = pick(SIDED[role], side);
  // spine chain: chest = the deepest spine-like ancestor of the neck (head, upper arms), spine = the one nearest the hips
  const spines = parsed.filter((p) => p.side === '' && SPINE.includes(p.key) && p.b !== map.hips && under(p.b))
    .sort((a, b) => a.depth - b.depth || a.order - b.order);
  const top = map.neck || map.head || map.upperArmL || map.upperArmR;
  let chest = null;
  if (top) for (const p of spines) if (isAncestor(p.b, top)) chest = p.b; // sorted by depth: the last is the deepest
  if (!chest && spines.length > 1) chest = spines[spines.length - 1].b;
  const spine = spines.find((p) => p.b !== chest);
  map.spine = spine ? spine.b : null;
  map.chest = chest;
  for (const k of Object.keys(map)) if (!map[k]) delete map[k];
  return map;
}

function resolveBones(spec, candidates, root) {
  if (spec === 'auto' || spec == null) return mapBonesAuto(candidates);
  const map = {};
  for (const name of RAGDOLL_SEGMENTS) {
    const v = spec[name];
    if (v == null) continue;
    let b = null;
    if (typeof v === 'object' && v.isObject3D) b = v;
    else {
      const s = String(v), lc = s.toLowerCase();
      b = candidates.find((o) => o.name === s) || candidates.find((o) => o.name.toLowerCase() === lc) || (root.getObjectByName && root.getObjectByName(s)) || null;
    }
    if (!b) throw new Error(`Ragdoll.fromSkeleton: bone '${v}' (${name}) not found`);
    map[name] = b;
  }
  return map;
}

// ---- scratch (update path) -----------------------------------------------------------------------------------------
const _c = new Vector3(), _p = new Vector3(), _v = new Vector3(), _s = new Vector3();
const _a = new Vector3(), _b = new Vector3(), _d = new Vector3(), _e = new Vector3(), _f = new Vector3(), _g = new Vector3();
const _qc = new Quaternion(), _qb = new Quaternion(), _q = new Quaternion(), _qt = new Quaternion();
const _sc = new Float64Array(2);
const _W = new Matrix4(), _inv = new Matrix4();
const Y = new Vector3(0, 1, 0);
const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);

export class Ragdoll {
  /**
   * Build a ragdoll description for a skeleton (no bodies yet: see activate()).
   * @param {THREE.Object3D} skinnedMeshOrRoot  a SkinnedMesh (its skeleton's bones) or the root of a bone hierarchy
   * @param {object} dyn  DynamicsWorld or DynamicsClient (addBody, addJoint, setPose, getPose, removeBody, ...)
   * @param {object} [opts]  see the header
   * @returns {Ragdoll}
   */
  static fromSkeleton(skinnedMeshOrRoot, dyn, opts = {}) { return new Ragdoll(skinnedMeshOrRoot, dyn, opts); }

  constructor(root, dyn, opts = {}) {
    if (!root || !root.isObject3D) throw new TypeError('Ragdoll: a SkinnedMesh or an Object3D root is required');
    if (!dyn || typeof dyn.addBody !== 'function') throw new TypeError('Ragdoll: a DynamicsWorld or DynamicsClient is required');
    opts = opts || {};
    this.root = root;
    this.dyn = dyn;
    this.active = false;
    this.disposed = false;
    const layer = opts.layer ?? layerOf('ragdoll');
    this.options = {
      density: opts.density ?? 985, friction: opts.friction ?? 0.6, restitution: opts.restitution ?? 0,
      linearDamping: opts.linearDamping ?? 0.05, angularDamping: opts.angularDamping ?? 0.1, jointDamping: opts.jointDamping ?? 4,
      layer, mask: opts.mask ?? ~layer, kind: opts.kind ?? 'ragdoll', userData: opts.userData ?? null,
    };
    // candidates
    let candidates = [];
    if (root.isSkinnedMesh && root.skeleton) candidates = root.skeleton.bones.slice();
    else {
      root.traverse((o) => { if (o.isBone) candidates.push(o); });
      if (!candidates.length) root.traverse((o) => { if (o !== root) candidates.push(o); });
    }
    const map = this.bones = resolveBones(opts.bones ?? 'auto', candidates, root);
    if (!map.hips) throw new Error('Ragdoll.fromSkeleton: no hips bone found (pass opts.bones)');
    for (const n of Object.keys(map)) map[n].updateWorldMatrix(true, false);
    this.segments = this._build(map, opts);
    this.hips = -1;
    // update order: bone depth (a bone's ancestors are written before it), segment order within a depth
    this._order = this.segments.map((s, i) => [depthOf(s.bone), i]).sort((a, b) => a[0] - b[0] || a[1] - b[1]).map((e) => this.segments[e[1]]);
  }

  // Segments from the mapped bones at their current (rest) pose.
  _build(map, opts) {
    const wp = (b) => new Vector3().setFromMatrixPosition(b.matrixWorld);
    const P = {};
    for (const n of Object.keys(map)) P[n] = wp(map[n]);
    // character frame: up (hips -> neck / head / chest / spine), left (left - right hip or shoulder), forward = left x up
    const up = new Vector3(0, 1, 0), left = new Vector3(1, 0, 0), fwd = new Vector3(0, 0, 1);
    const topN = ['neck', 'head', 'chest', 'spine'].find((n) => P[n] && P[n].distanceToSquared(P.hips) > 1e-12);
    if (topN) up.subVectors(P[topN], P.hips).normalize();
    const pair = P.upperLegL && P.upperLegR ? ['upperLegL', 'upperLegR'] : P.upperArmL && P.upperArmR ? ['upperArmL', 'upperArmR'] : null;
    if (pair) left.subVectors(P[pair[0]], P[pair[1]]);
    left.addScaledVector(up, -left.dot(up));
    if (left.lengthSq() < 1e-12) left.set(1, 0, 0).addScaledVector(up, -up.x);
    left.normalize();
    fwd.crossVectors(left, up).normalize();
    this.forward = fwd; this.up = up; this.left = left;
    const legLen = P.upperLegL && P.lowerLegL && P.footL ? P.upperLegL.distanceTo(P.lowerLegL) + P.lowerLegL.distanceTo(P.footL) : 0;
    const k = legLen > 1e-6 ? legLen / 0.85 : 1;
    const numOr = (spec, name, base, d) => {
      if (typeof spec === 'number') return spec;
      if (spec && typeof spec === 'object') { const v = spec[name] ?? spec[base]; if (typeof v === 'number') return v; }
      return d;
    };
    // limits merged per field: { shoulder: { swing: 0.5 } } keeps the default twist; an array (hinge) or null (free
    // ball) replaces the joint's entry
    const limits = Object.assign({}, DEFAULT_LIMITS);
    for (const [k, u] of Object.entries(opts.limits || {})) {
      const d = DEFAULT_LIMITS[k];
      limits[k] = u && typeof u === 'object' && !Array.isArray(u) && d && !Array.isArray(d) ? Object.assign({}, d, u) : u;
    }

    const segs = [], byName = {};
    for (const name of RAGDOLL_SEGMENTS) {
      const bone = map[name];
      if (!bone) continue;
      const side = /[LR]$/.test(name) && name !== 'hips' ? name[name.length - 1] : '';
      const role = side ? name.slice(0, -1) : name;
      const R = ROLE[role];
      const sub = (n) => (n.endsWith('*') ? n.slice(0, -1) + side : n);
      const parentName = R.parents.map(sub).find((n) => byName[n] !== undefined);
      const childName = R.children.map(sub).find((n) => map[n]);
      const origin = P[name];
      // direction and length
      const dir = new Vector3();
      let length;
      if (childName && P[childName].distanceToSquared(origin) > 1e-12) {
        dir.subVectors(P[childName], origin);
        length = dir.length();
        dir.divideScalar(length);
      } else {
        length = numOr(opts.length, name, role, R.leaf * k);
        // the bone's own children (toes, finger roots, a head-top end), else its local axis nearest to a reference
        const sum = new Vector3();
        let n = 0;
        for (const c of bone.children) {
          if (!c.isObject3D || Object.values(map).includes(c)) continue;
          c.updateWorldMatrix(false, false);
          const d = wp(c).sub(origin);
          if (d.lengthSq() > 1e-12) { sum.add(d.normalize()); n++; }
        }
        if (n > 0 && sum.lengthSq() > 1e-12) dir.copy(sum).normalize();
        else {
          const ref = new Vector3();
          if (role === 'foot') ref.copy(fwd);
          else if (parentName !== undefined && origin.distanceToSquared(P[parentName]) > 1e-12) ref.subVectors(origin, P[parentName]).normalize();
          else ref.copy(up);
          dir.copy(nearestAxis(bone, ref));
        }
      }
      // radius
      let radius = clamp(0.22 * length, 0.04, 0.12);
      if (role === 'head') radius = clamp(0.4 * length, 0.04, 0.12);  // a short capsule, not a ball (balls roll off)
      if (role === 'neck') radius = clamp(0.5 * length, 0.04, 0.12);  // no tiny link between the chest and the head
      if (role === 'hips' && P.upperLegL && P.upperLegR) radius = clamp(Math.max(0.22 * length, 0.5 * P.upperLegL.distanceTo(P.upperLegR)), 0.04, 0.12);
      if (role === 'chest' && P.upperArmL && P.upperArmR) radius = clamp(Math.max(0.22 * length, 0.25 * P.upperArmL.distanceTo(P.upperArmR)), 0.04, 0.12);
      radius = numOr(opts.radius === 'auto' ? null : opts.radius, name, role, radius);
      if (!(radius > 0)) throw new RangeError(`Ragdoll.fromSkeleton: radius of ${name} must be positive`);
      const seg = {
        name, role, side, bone, parent: parentName !== undefined ? byName[parentName] : -1, body: -1, joint: -1,
        jointType: null, length, radius, halfHeight: 0, mass: 0,
        offset: new Vector3(), rotation: new Quaternion(), dir,
        restPos: new Vector3(), restQuat: new Quaternion(),
        shape: null, anchor: [0, 0, 0], axis: [0, 1, 0], ref: [1, 0, 0], limits: null,
        swingMax: Infinity, twistLo: -Infinity, twistHi: Infinity,
        _rotInv: new Quaternion(), _scale: new Vector3(1, 1, 1), _autoUpdate: true,
        // joint frames for the activation clamp: anchor / axis / reference of the child (A, capsule frame) and of the
        // parent (B, its capsule frame, frozen at rest as joints.js freezes them), the pose being set, the subtree
        _anchorA: new Vector3(), _axisA: new Vector3(), _refA: new Vector3(), _anchorB: new Vector3(), _axisB: new Vector3(), _refB: new Vector3(),
        _pos: new Vector3(), _quat: new Quaternion(), _sub: null,
      };
      byName[name] = segs.length;
      segs.push(seg);
    }
    // the spine's 'auto' radius sits between the hips' and the chest's
    if (byName.spine !== undefined && (opts.radius == null || opts.radius === 'auto' || (typeof opts.radius === 'object' && opts.radius.spine == null))) {
      const s = segs[byName.spine], h = byName.hips !== undefined ? segs[byName.hips].radius : 0, c = byName.chest !== undefined ? segs[byName.chest].radius : 0;
      if (h > 0 && c > 0) s.radius = clamp(Math.max(s.radius, 0.5 * (h + c)), 0.04, 0.12);
    }
    // capsule poses at rest, offsets / rotations in the bone frames, shapes, masses, joints
    const density = this.options.density, inv = new Matrix4(), pb = new Vector3(), qb = new Quaternion(), sb = new Vector3();
    for (const s of segs) {
      s.halfHeight = Math.max(0, 0.5 * s.length - s.radius);
      s.shape = capsule(s.radius, s.halfHeight);
      s.mass = density * s.shape.volume;
      s.restPos.copy(P[s.name]).addScaledVector(s.dir, 0.5 * s.length);
      s.bone.matrixWorld.decompose(pb, qb, sb);
      _v.copy(s.dir).applyQuaternion(_q.copy(qb).invert());                  // segment direction in the bone frame
      s.rotation.setFromUnitVectors(Y, _v.normalize());
      s.restQuat.multiplyQuaternions(qb, s.rotation);
      s._rotInv.copy(s.rotation).invert();
      s.offset.copy(s.restPos).applyMatrix4(inv.copy(s.bone.matrixWorld).invert());
      s._scale.copy(sb);
    }
    for (const s of segs) {
      if (s.parent < 0) continue;
      const par = segs[s.parent];
      const jname = ROLE[s.role].joint;
      const lim = limits[jname];
      s.limits = lim;
      // anchor: the child bone's origin, in the child capsule's frame
      _v.copy(P[s.name]).sub(s.restPos).applyQuaternion(_q.copy(s.restQuat).invert());
      s.anchor = [_v.x, _v.y, _v.z];
      if (Array.isArray(lim)) {
        s.jointType = 'hinge';
        const axis = hingeAxis(s, HINGE_UPPER[s.role] === par.role ? par : null, fwd, up, left, BACKWARD[s.role] === 1);
        _v.copy(axis).applyQuaternion(_q.copy(s.restQuat).invert());
        s.axis = [_v.x, _v.y, _v.z];
        s.swingMax = 0; s.twistLo = +lim[0]; s.twistHi = +lim[1];
      } else {
        s.jointType = 'ball';
        s.axis = [0, 1, 0];
        const sw = lim ? lim.swing : null, tw = lim ? lim.twist : null;
        s.swingMax = sw == null ? Infinity : Array.isArray(sw) ? +sw[1] : +sw;
        if (tw != null) { s.twistLo = Array.isArray(tw) ? +tw[0] : -tw; s.twistHi = Array.isArray(tw) ? +tw[1] : +tw; }
      }
      // reference direction (angle zero): local X projected off the axis (Z for an axis near X), as joints.js picks it
      const ax = s._axisA.set(s.axis[0], s.axis[1], s.axis[2]);
      const rf = Math.abs(ax.x) < 0.9 ? s._refA.set(1, 0, 0).addScaledVector(ax, -ax.x) : s._refA.set(0, 0, 1).addScaledVector(ax, -ax.z);
      rf.normalize();
      s.ref = [rf.x, rf.y, rf.z];
      s._anchorA.set(s.anchor[0], s.anchor[1], s.anchor[2]);
      _q.copy(par.restQuat).invert();
      s._anchorB.copy(P[s.name]).sub(par.restPos).applyQuaternion(_q);
      s._axisB.copy(ax).applyQuaternion(s.restQuat).applyQuaternion(_q);
      s._refB.copy(rf).applyQuaternion(s.restQuat).applyQuaternion(_q);
    }
    // each segment's subtree (itself and its descendants; segments come parents first)
    for (let i = segs.length - 1; i >= 0; i--) {
      segs[i]._sub = [i];
      for (let j = i + 1; j < segs.length; j++) if (segs[j].parent === i) segs[i]._sub.push(...segs[j]._sub);
    }
    return segs;
  }

  /**
   * Create the bodies and joints (see the header) and hand the bones to update().
   * @param {object} [o]
   * @param {THREE.Vector3|number[]} [o.velocity=null]  linear velocity given to every body
   * @param {boolean} [o.fromPose=true]  bodies at the bones' current pose; false: the rest pose, carried by the hips
   * @returns {this}
   */
  activate({ velocity = null, fromPose = true } = {}) {
    if (this.active || this.disposed) return this;
    const dyn = this.dyn, S = this.segments, O = this.options;
    for (const s of S) {
      s.body = dyn.addBody({
        shape: s.shape, position: s.restPos, quaternion: s.restQuat, mass: s.mass, friction: O.friction, restitution: O.restitution,
        linearDamping: O.linearDamping, angularDamping: O.angularDamping, layer: O.layer, mask: O.mask, kind: O.kind, userData: O.userData,
      });
    }
    for (const s of S) {
      if (s.parent < 0) continue;
      const desc = { type: s.jointType, bodyA: s.body, bodyB: S[s.parent].body, anchorA: s.anchor, axisA: s.axis, refA: s.ref, damping: O.jointDamping };
      const lim = s.limits;
      if (s.jointType === 'hinge') desc.limits = [lim[0], lim[1]];
      else if (lim) { if (lim.swing != null) desc.swing = lim.swing; if (lim.twist != null) desc.twist = lim.twist; }
      s.joint = dyn.addJoint(desc);
    }
    // teleport to the current pose (or the rest pose carried by the hips' current transform)
    const h = S[0];
    capsulePose(h, _c, _qc);                                  // the hips' current capsule pose (also sets its _scale)
    _q.copy(h.restQuat).invert().premultiply(_qc);            // rest -> current rotation of the hips
    for (const s of S) {
      if (fromPose) capsulePose(s, s._pos, s._quat);
      else {
        s.bone.updateWorldMatrix(true, false); s.bone.matrixWorld.decompose(_v, _qb, s._scale);
        s._pos.copy(s.restPos).sub(h.restPos).applyQuaternion(_q).add(_c);
        s._quat.multiplyQuaternions(_q, s.restQuat);
      }
    }
    this._clampPose();
    for (const s of S) {
      dyn.setPose(s.body, s._pos.x, s._pos.y, s._pos.z, s._quat.x, s._quat.y, s._quat.z, s._quat.w);
      if (velocity) dyn.setVelocity(s.body, velocity.x ?? velocity[0] ?? 0, velocity.y ?? velocity[1] ?? 0, velocity.z ?? velocity[2] ?? 0, 0, 0, 0);
    }
    for (const s of S) { s._autoUpdate = s.bone.matrixAutoUpdate; s.bone.matrixAutoUpdate = false; }
    this.hips = h.body;
    this.active = true;
    return this;
  }

  // Bring the pose being set (segments' _pos / _quat) inside the joints: a limit or lock violated at activation is
  // corrected in full by the rigid limit row in the first substep, and the solver turns that correction into velocity
  // (an arm at 2.5 rad against the 1.5 rad shoulder cone left at ~25 m/s, 93 rad/s). Parents first, per joint: the
  // child's subtree is moved so its anchor sits on the parent's, turned about f1 = aB x aA by the swing beyond the cone
  // (all of it for a hinge), then about its axis by the twist outside [lo, hi] (joints.js's measures; no trig: angles
  // through jointAtan2, rotations from a series sine / cosine), so the doll starts satisfied.
  _clampPose() {
    const S = this.segments;
    for (let i = 1; i < S.length; i++) {
      const s = S[i];
      if (s.parent < 0) continue;
      const p = S[s.parent];
      _a.copy(s._anchorA).applyQuaternion(s._quat).add(s._pos);
      _b.copy(s._anchorB).applyQuaternion(p._quat).add(p._pos);           // the pivot: the parent's anchor
      if (_a.distanceToSquared(_b) > 1e-12) { _d.subVectors(_b, _a); for (const j of s._sub) S[j]._pos.add(_d); } // > 1 um
      // swing: angle between the child's axis aA and its rest direction in the parent aB
      _e.copy(s._axisA).applyQuaternion(s._quat); _f.copy(s._axisB).applyQuaternion(p._quat);
      _d.crossVectors(_f, _e);
      const cl = _d.length(), swing = jointAtan2(cl, _f.dot(_e));
      if (swing > s.swingMax + 1e-9 && cl > 1e-12) this._turn(s, _d.divideScalar(cl), s.swingMax - swing);
      // twist about the half axis, references projected off it
      for (let it = 0; it < 3; it++) {
        _e.copy(s._axisA).applyQuaternion(s._quat); _f.copy(s._axisB).applyQuaternion(p._quat);
        _d.addVectors(_e, _f);
        const hl = _d.length();
        if (!(hl > 1e-9)) break;
        _d.divideScalar(hl);
        _a.copy(s._refA).applyQuaternion(s._quat); _a.addScaledVector(_d, -_a.dot(_d));
        _c.copy(s._refB).applyQuaternion(p._quat); _c.addScaledVector(_d, -_c.dot(_d));
        const phi = jointAtan2(_d.dot(_f.crossVectors(_c, _a)), _c.dot(_a));
        const t = phi < s.twistLo ? s.twistLo : phi > s.twistHi ? s.twistHi : phi;
        if (Math.abs(t - phi) <= 1e-9) break;
        this._turn(s, _d.copy(s._axisA).applyQuaternion(s._quat), t - phi);
      }
    }
  }

  // Turn segment s's subtree by `angle` about the unit world `axis` through s's anchor (+angle: right-handed).
  _turn(s, axis, angle) {
    const S = this.segments;
    sinCos(0.5 * angle, _sc);
    _qt.set(axis.x * _sc[0], axis.y * _sc[0], axis.z * _sc[0], _sc[1]);
    _g.copy(s._anchorA).applyQuaternion(s._quat).add(s._pos);
    for (const j of s._sub) {
      const t = S[j];
      t._pos.sub(_g).applyQuaternion(_qt).add(_g);
      t._quat.premultiply(_qt);
    }
  }

  /** Bones <- bodies (after dyn.step, before rendering). Allocation-free. A body without a pose yet (a worker-mode
   * DynamicsClient before its first snapshot: getPose returns false) leaves its bone as it is. */
  update() {
    if (!this.active) return this;
    const dyn = this.dyn, S = this._order;
    for (let i = 0; i < S.length; i++) {
      const s = S[i], bone = s.bone, parent = bone.parent;
      if (dyn.getPose(s.body, _c, _qc) === false) continue;
      _qb.multiplyQuaternions(_qc, s._rotInv);
      _p.copy(s.offset).multiply(s._scale).applyQuaternion(_qb);
      _W.compose(_c.sub(_p), _qb, s._scale);
      if (parent) {
        parent.updateWorldMatrix(true, false);
        bone.matrix.multiplyMatrices(_inv.copy(parent.matrixWorld).invert(), _W);
      } else bone.matrix.copy(_W);
      bone.matrixWorld.copy(_W);
      bone.matrixWorldNeedsUpdate = true; // the next scene update refreshes its unmapped children from it
    }
    return this;
  }

  /**
   * Remove the bodies and joints; the bones keep their last pose (in position / quaternion / scale) and get
   * matrixAutoUpdate back.
   * @param {object} [o]
   * @param {object} [o.controller]  the character's AnimationController: its resync() is called, so its mixer writes
   *   every channel again (a PropertyMixer only writes a value that changed since its last write, so a channel whose
   *   clip value is constant would keep the ragdoll's pose). Without it, call controller.resync() yourself.
   * @returns {this}
   */
  deactivate({ controller = null } = {}) {
    if (!this.active) return this;
    const dyn = this.dyn, S = this.segments;
    for (let i = S.length - 1; i >= 0; i--) { const s = S[i]; if (s.joint >= 0 && typeof dyn.removeJoint === 'function') dyn.removeJoint(s.joint); s.joint = -1; }
    for (let i = S.length - 1; i >= 0; i--) { const s = S[i]; if (s.body >= 0) dyn.removeBody(s.body); s.body = -1; }
    // the last pose into position / quaternion / scale (update() writes the matrices only)
    for (const s of S) { s.bone.matrix.decompose(s.bone.position, s.bone.quaternion, s.bone.scale); s.bone.matrixAutoUpdate = s._autoUpdate; }
    this.hips = -1;
    this.active = false;
    if (controller && typeof controller.resync === 'function') controller.resync();
    return this;
  }

  /** The hips bone's world origin: where its body puts it (active; as update() would write it) or where it is. */
  position(out = new Vector3()) {
    const h = this.segments[0];
    if (this.active && this.dyn.getPose(h.body, _c, _qc) !== false) {
      _qb.multiplyQuaternions(_qc, h._rotInv);
      _p.copy(h.offset).multiply(h._scale).applyQuaternion(_qb);
      return out.subVectors(_c, _p);
    }
    h.bone.updateWorldMatrix(true, false);
    return out.setFromMatrixPosition(h.bone.matrixWorld);
  }

  /** True when every body sleeps or moves slower than `speed` (m/s); false when inactive. */
  isResting({ speed = 0.1 } = {}) {
    if (!this.active) return false;
    const dyn = this.dyn, s2 = speed * speed;
    for (const s of this.segments) {
      if (typeof dyn.isAwake === 'function' && !dyn.isAwake(s.body)) continue;
      if (dyn.getVelocity(s.body, _v, null) === false) return false; // not in a worker snapshot yet
      if (!(_v.lengthSq() < s2)) return false;
    }
    return true;
  }

  /** Deactivate and drop the references. */
  dispose() {
    this.deactivate();
    this.disposed = true;
    this.dyn = null;
  }
}

// out = [sin x, cos x] from the Taylor series (+ - * / only, so a clamped pose is bit-identical on every engine;
// |x| <= pi: the last term is below 1e-17)
function sinCos(x, out) {
  const x2 = x * x;
  let s = x, c = 1, ts = x, tc = 1;
  for (let n = 1; n <= 16; n++) { tc *= -x2 / ((2 * n - 1) * (2 * n)); ts *= -x2 / ((2 * n) * (2 * n + 1)); c += tc; s += ts; }
  out[0] = s; out[1] = c;
}

// Current capsule pose of segment s from its bone's world matrix (bone scale stored for update()).
function capsulePose(s, outPos, outQuat) {
  s.bone.updateWorldMatrix(true, false);
  s.bone.matrixWorld.decompose(_s, outQuat, s._scale);
  outPos.copy(s.offset).multiply(s._scale).applyQuaternion(outQuat).add(_s);
  outQuat.multiply(s.rotation);
}

// The bone's local axis (+-X, +-Y, +-Z in world space) closest to the unit direction ref.
function nearestAxis(bone, ref) {
  const e = bone.matrixWorld.elements, best = new Vector3(ref.x, ref.y, ref.z);
  let bd = -Infinity;
  for (let c = 0; c < 3; c++) {
    const a = new Vector3(e[c * 4], e[c * 4 + 1], e[c * 4 + 2]);
    if (a.lengthSq() < 1e-24) continue;
    a.normalize();
    const d = a.dot(ref);
    if (Math.abs(d) > bd) { bd = Math.abs(d); best.copy(a).multiplyScalar(d < 0 ? -1 : 1); }
  }
  return best;
}

// Hinge axis (world, rest pose) of child segment s: flexion (a positive joint angle) swings its far end towards the
// character's front (back for a knee); a limb bent at rest uses the bend-plane normal with that sign.
function hingeAxis(s, upper, fwd, up, left, backward) {
  const d = s.dir, a = new Vector3();
  if (backward) a.crossVectors(fwd, d); else a.crossVectors(d, fwd);
  if (a.lengthSq() < 0.01) { if (backward) a.copy(left); else a.crossVectors(d, up); }
  if (a.lengthSq() < 0.01) a.copy(left);
  a.addScaledVector(d, -a.dot(d));
  if (a.lengthSq() < 1e-12) a.set(d.y, d.z, d.x).cross(d);
  a.normalize();
  if (upper) {
    const n = new Vector3().crossVectors(upper.dir, d);
    if (n.lengthSq() > 0.01) { n.normalize(); a.copy(n.dot(a) < 0 ? n.negate() : n); }
  }
  return a;
}
