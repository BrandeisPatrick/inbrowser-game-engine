// Headless checks for the skeleton ragdoll (src/anim/Ragdoll.js): bone auto-mapping (procedural rig, mixamo,
// RobotExpressive names), segment sizing, activation at the bones' pose (exact round trip), hinge sign conventions,
// a dropped doll on the ground (stable, comes to rest, deterministic), deactivation, options, allocation of update().
// No browser, no WebGL. Run: node test/ragdoll.mjs
import { Group, Object3D, Bone, Vector3, Quaternion, Matrix4, Skeleton, SkinnedMesh, BufferGeometry, MeshBasicMaterial, AnimationClip, QuaternionKeyframeTrack } from 'three';
import { Ragdoll, mapBonesAuto, DEFAULT_LIMITS } from '../src/anim/Ragdoll.js';
import { AnimationController } from '../src/anim/AnimationController.js';
import { PhysicsWorld } from '../src/physics/PhysicsWorld.js';
import { DynamicsWorld } from '../src/physics/dynamics/DynamicsWorld.js';
import { DynamicsClient } from '../src/physics/DynamicsClient.js';
import { jointAngle, jointSwing } from '../src/physics/dynamics/joints.js';

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

// ---- rigs ------------------------------------------------------------------------------------------------------------
// The procedural rig of samples/character/rig.js (JOINTS table copied, not imported): [name, parent, joint position]
const PROC = [
  ['hips', null, [0, 0.98, 0]], ['spine', 'hips', [0, 1.1, 0]], ['chest', 'spine', [0, 1.3, 0]], ['neck', 'chest', [0, 1.5, 0]],
  ['head', 'neck', [0, 1.6, 0]], ['upperArmL', 'chest', [0.25, 1.45, 0]], ['lowerArmL', 'upperArmL', [0.25, 1.17, 0]],
  ['upperArmR', 'chest', [-0.25, 1.45, 0]], ['lowerArmR', 'upperArmR', [-0.25, 1.17, 0]], ['upperLegL', 'hips', [0.1, 0.93, 0]],
  ['lowerLegL', 'upperLegL', [0.1, 0.49, 0]], ['footL', 'lowerLegL', [0.1, 0.08, 0]], ['upperLegR', 'hips', [-0.1, 0.93, 0]],
  ['lowerLegR', 'upperLegR', [-0.1, 0.49, 0]], ['footR', 'lowerLegR', [-0.1, 0.08, 0]],
];
const PROC_HANDS = PROC.concat([['handL', 'lowerArmL', [0.25, 0.85, 0]], ['handR', 'lowerArmR', [-0.25, 0.85, 0]]]);
// mixamo-style T-pose (three's loader drops the ':' of 'mixamorig:Hips'), cm armature (scale 0.01), rotated bones
const MIX = (() => {
  const n = (s) => `mixamorig${s}`;
  const t = [
    [n('Hips'), null, [0, 1.0, 0]], [n('Spine'), n('Hips'), [0, 1.1, 0.01]], [n('Spine1'), n('Spine'), [0, 1.22, 0.01]],
    [n('Spine2'), n('Spine1'), [0, 1.36, 0]], [n('Neck'), n('Spine2'), [0, 1.52, -0.01]], [n('Head'), n('Neck'), [0, 1.62, 0.02]],
    [n('HeadTop_End'), n('Head'), [0, 1.82, 0.03]],
  ];
  for (const [S, s] of [['Left', 1], ['Right', -1]]) {
    t.push(
      [n(`${S}Shoulder`), n('Spine2'), [s * 0.06, 1.46, 0]], [n(`${S}Arm`), n(`${S}Shoulder`), [s * 0.18, 1.45, -0.01]],
      [n(`${S}ForeArm`), n(`${S}Arm`), [s * 0.45, 1.45, -0.02]], [n(`${S}Hand`), n(`${S}ForeArm`), [s * 0.7, 1.45, 0]],
      [n(`${S}HandThumb1`), n(`${S}Hand`), [s * 0.73, 1.44, 0.04]], [n(`${S}HandIndex1`), n(`${S}Hand`), [s * 0.79, 1.45, 0.03]],
      [n(`${S}HandMiddle1`), n(`${S}Hand`), [s * 0.8, 1.45, 0]], [n(`${S}HandIndex2`), n(`${S}HandIndex1`), [s * 0.83, 1.45, 0.03]],
      [n(`${S}UpLeg`), n('Hips'), [s * 0.09, 0.95, 0]], [n(`${S}Leg`), n(`${S}UpLeg`), [s * 0.09, 0.52, 0.01]],
      [n(`${S}Foot`), n(`${S}Leg`), [s * 0.09, 0.09, -0.02]], [n(`${S}ToeBase`), n(`${S}Foot`), [s * 0.09, 0.02, 0.12]],
      [n(`${S}Toe_End`), n(`${S}ToeBase`), [s * 0.09, 0.02, 0.2]],
    );
  }
  return t;
})();
// RobotExpressive-style names (three's loader turns 'UpperArm.L' into 'UpperArmL'): legs under Body (beside Hips), feet
// under the root bone (an IK rig), three palms and no hand, pole targets
const ROBOT = (() => {
  const t = [
    ['Bone', null, [0, 0, 0]], ['Body', 'Bone', [0, 0.96, 0]], ['Hips', 'Body', [0, 1.0, 0]], ['Abdomen', 'Hips', [0, 1.1, 0]],
    ['Torso', 'Abdomen', [0, 1.25, 0]], ['Neck', 'Torso', [0, 1.5, 0]], ['Head', 'Neck', [0, 1.58, 0]], ['Head_end', 'Head', [0, 1.85, 0]],
  ];
  for (const [S, s] of [['L', 1], ['R', -1]]) {
    t.push(
      [`Shoulder${S}`, 'Torso', [s * 0.12, 1.45, 0]], [`UpperArm${S}`, `Shoulder${S}`, [s * 0.22, 1.45, 0]],
      [`LowerArm${S}`, `UpperArm${S}`, [s * 0.24, 1.18, 0.02]], [`Palm1${S}`, `LowerArm${S}`, [s * 0.25, 0.95, 0.07]],
      [`Palm2${S}`, `LowerArm${S}`, [s * 0.24, 0.95, 0.06]], [`Palm3${S}`, `LowerArm${S}`, [s * 0.23, 0.95, 0.05]],
      [`Middle1${S}`, `Palm2${S}`, [s * 0.24, 0.88, 0.08]], [`Thumb${S}`, `LowerArm${S}`, [s * 0.22, 0.97, 0.08]],
      [`UpperLeg${S}`, 'Body', [s * 0.12, 0.95, 0]], [`LowerLeg${S}`, `UpperLeg${S}`, [s * 0.12, 0.52, 0.03]],
      [`Foot${S}`, 'Bone', [s * 0.12, 0.1, -0.02]], [`PoleTarget${S}`, 'Bone', [s * 0.12, 0.6, 0.5]],
    );
  }
  return t;
})();

// Build a bone hierarchy whose joints sit at the table's world positions. scale: the armature's scale (every bone's world
// scale); spin: random bone orientations (world), so offsets / rotations are exercised in arbitrary bone frames.
function buildRig(table, { scale = 1, spin = false, seed = 3, skinned = false } = {}) {
  let s = seed >>> 0;
  const rnd = () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296; };
  const root = new Group(); root.name = 'character';
  const arm = new Object3D(); arm.name = 'Armature'; arm.scale.setScalar(scale); root.add(arm);
  root.updateMatrixWorld(true);
  const by = {}, bones = [], W = new Matrix4(), L = new Matrix4(), q = new Quaternion(), sc = new Vector3(scale, scale, scale);
  for (const [name, parent, p] of table) {
    const b = new Bone(); b.name = name;
    const par = parent ? by[parent] : arm;
    par.add(b);
    if (spin) q.set(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5, rnd() - 0.5).normalize(); else q.identity();
    W.compose(new Vector3(p[0], p[1], p[2]), q, sc);
    L.copy(par.matrixWorld).invert().multiply(W);
    L.decompose(b.position, b.quaternion, b.scale);
    b.updateMatrixWorld(true);
    by[name] = b; bones.push(b);
  }
  root.updateMatrixWorld(true);
  let mesh = null;
  if (skinned) {
    mesh = new SkinnedMesh(new BufferGeometry(), new MeshBasicMaterial());
    root.add(mesh);
    mesh.bind(new Skeleton(bones));
    mesh.frustumCulled = false;
  }
  return { root, by, bones, mesh };
}
function newWorld(o = {}) {
  const physics = new PhysicsWorld({ groundLevel: 0 });
  return { physics, dyn: new DynamicsWorld({ physics, ...o }) };
}
const worldPos = (o) => new Vector3().setFromMatrixPosition(o.matrixWorld);
const matErr = (a, b) => { let m = 0; for (let i = 0; i < 16; i++) m = Math.max(m, Math.abs(a.elements[i] - b.elements[i])); return m; };
const snapshot = (rd) => rd.segments.map((s) => s.bone.matrixWorld.clone());
const maxMatErr = (rd, before) => rd.segments.reduce((m, s, i) => Math.max(m, matErr(s.bone.matrixWorld, before[i])), 0);
function anchorGap(dyn, rd) {
  const J = dyn.joints, pa = new Vector3(), pb = new Vector3(), qa = new Quaternion(), qb = new Quaternion();
  let g = 0;
  for (const s of rd.segments) {
    if (s.joint < 0) continue;
    const j = s.joint;
    dyn.getPose(J.a[j], pa, qa); dyn.getPose(J.b[j], pb, qb);
    const A = new Vector3(J.anchorA[j * 3], J.anchorA[j * 3 + 1], J.anchorA[j * 3 + 2]).applyQuaternion(qa).add(pa);
    const B = new Vector3(J.anchorB[j * 3], J.anchorB[j * 3 + 1], J.anchorB[j * 3 + 2]).applyQuaternion(qb).add(pb);
    g = Math.max(g, A.distanceTo(B));
  }
  return g;
}

// ---- 1. auto mapping -------------------------------------------------------------------------------------------------
const SEG = ['hips', 'spine', 'chest', 'neck', 'head', 'upperArmL', 'lowerArmL', 'handL', 'upperArmR', 'lowerArmR', 'handR',
  'upperLegL', 'lowerLegL', 'footL', 'upperLegR', 'lowerLegR', 'footR'];
{
  const proc = buildRig(PROC);
  const m = mapBonesAuto(proc.bones);
  check('auto: procedural rig maps all 15 bones', Object.keys(m).length, 15, 15);
  check('auto: procedural rig, every role on the bone of its name', PROC.filter(([n]) => m[n] && m[n].name === n).length, 15, 15);
  const procH = buildRig(PROC_HANDS);
  const mh = mapBonesAuto(procH.bones);
  check('auto: procedural rig with hand leaves maps 17 (handL / handR)', Object.keys(mh).length + (mh.handL.name === 'handL' && mh.handR.name === 'handR' ? 0 : 100), 17, 17);
  const want = {
    hips: 'Hips', spine: 'Spine', chest: 'Spine2', neck: 'Neck', head: 'Head', upperArmL: 'LeftArm', lowerArmL: 'LeftForeArm', handL: 'LeftHand',
    upperArmR: 'RightArm', lowerArmR: 'RightForeArm', handR: 'RightHand', upperLegL: 'LeftUpLeg', lowerLegL: 'LeftLeg', footL: 'LeftFoot',
    upperLegR: 'RightUpLeg', lowerLegR: 'RightLeg', footR: 'RightFoot',
  };
  const mix = buildRig(MIX, { scale: 0.01, spin: true });
  const mm = mapBonesAuto(mix.bones);
  check('auto: mixamo names map 17 roles', Object.keys(mm).length, 17, 17);
  check('auto: mixamo roles on the expected bones (Spine2 = chest, LeftArm = upper arm)', SEG.filter((k) => mm[k] && mm[k].name === `mixamorig${want[k]}`).length, 17, 17);
  const colon = buildRig(MIX.map(([a, b, p]) => [a.replace('mixamorig', 'mixamorig:'), b && b.replace('mixamorig', 'mixamorig:'), p]));
  check('auto: unsanitised mixamo names (mixamorig:LeftForeArm) map too', Object.keys(mapBonesAuto(colon.bones)).length, 17, 17);
  const wantR = {
    hips: 'Hips', spine: 'Abdomen', chest: 'Torso', neck: 'Neck', head: 'Head', upperArmL: 'UpperArmL', lowerArmL: 'LowerArmL', handL: 'Palm2L',
    upperArmR: 'UpperArmR', lowerArmR: 'LowerArmR', handR: 'Palm2R', upperLegL: 'UpperLegL', lowerLegL: 'LowerLegL', footL: 'FootL',
    upperLegR: 'UpperLegR', lowerLegR: 'LowerLegR', footR: 'FootR',
  };
  const robot = buildRig(ROBOT, { spin: true, seed: 9 });
  const mr = mapBonesAuto(robot.bones);
  check('auto: RobotExpressive names map 17 roles (Palm2 = hand, feet under the root)', SEG.filter((k) => mr[k] && mr[k].name === wantR[k]).length, 17, 17);
  yes('auto: helpers / leaf markers stay unmapped (PoleTarget, Head_end, Shoulder)', !Object.values(mr).some((b) => /Pole|_end|Shoulder/.test(b.name)));
  // explicit maps
  const ex = buildRig(PROC);
  const { dyn } = newWorld();
  const rdx = Ragdoll.fromSkeleton(ex.root, dyn, { bones: { hips: 'HIPS', spine: ex.by.spine, chest: 'chest', upperLegL: 'upperLegL', lowerLegL: 'lowerLegL' } });
  check('explicit map: names (case-insensitive) and Bone objects, nothing auto-filled', rdx.segments.length, 5, 5);
  let threw = 0;
  try { Ragdoll.fromSkeleton(ex.root, dyn, { bones: { hips: 'pelvisX' } }); } catch { threw++; }
  try { Ragdoll.fromSkeleton(buildRig([['root', null, [0, 0, 0]], ['spine', 'root', [0, 1, 0]]]).root, dyn); } catch { threw++; }
  check('errors: an unknown bone name and a rig without hips throw', threw, 2, 2);
}

// ---- 2. segments of the procedural rig -----------------------------------------------------------------------------------
const P = Object.fromEntries(PROC.map(([n, , p]) => [n, new Vector3(...p)]));
{
  const rig = buildRig(PROC, { skinned: true });
  const { dyn } = newWorld();
  const rd = Ragdoll.fromSkeleton(rig.mesh, dyn);   // a SkinnedMesh: its skeleton's bones
  const S = rd.segments, by = Object.fromEntries(S.map((s) => [s.name, s]));
  check('segments: 15 for the procedural rig (from its SkinnedMesh)', S.length, 15, 15);
  yes('segments: parents come first, the hips is the only root', S.every((s, i) => (i === 0 ? s.parent === -1 : s.parent >= 0 && s.parent < i)));
  check('segments: min length (m)', Math.min(...S.map((s) => s.length)), 0.05, 1);
  check('segment lengths: upper leg 0.44, shin 0.41 (bone to chain child)', Math.abs(by.upperLegL.length - 0.44) + Math.abs(by.lowerLegR.length - 0.41), 0, 1e-12);
  check('segment lengths: leaves at their defaults (head 0.18, foot 0.22)', Math.abs(by.head.length - 0.18) + Math.abs(by.footL.length - 0.22), 0, 1e-12);
  check('segments: radii within 0.04 .. 0.12 (min)', Math.min(...S.map((s) => s.radius)), 0.04, 0.12);
  check('segments: radii within 0.04 .. 0.12 (max)', Math.max(...S.map((s) => s.radius)), 0.04, 0.12);
  check('segments: limb radius = 0.22 x length (upper arm)', by.upperArmL.radius / by.upperArmL.length, 0.22 - 1e-12, 0.22 + 1e-12);
  check('segments: total mass (density 985)', S.reduce((a, s) => a + s.mass, 0), 50, 100, 'kg');
  yes('joint types: hinge elbows / knees, ball elsewhere, none on the hips',
    S.every((s) => (s.name === 'hips' ? s.jointType === null : /^lower/.test(s.name) ? s.jointType === 'hinge' : s.jointType === 'ball')));
  check('character frame: forward = +Z (left hip at +X, up +Y)', rd.forward.z, 1 - 1e-12, 1);
  const axisWorld = (s) => new Vector3(...s.axis).applyQuaternion(s.restQuat);
  check('hinge axis: knees +X (flexion swings the foot back), L and R', axisWorld(by.lowerLegL).x + axisWorld(by.lowerLegR).x, 2 - 1e-9, 2 + 1e-9);
  check('hinge axis: hanging elbows -X (flexion swings the hand forward)', axisWorld(by.lowerArmL).x + axisWorld(by.lowerArmR).x, -2 - 1e-9, -2 + 1e-9);
  const mix = buildRig(MIX, { scale: 0.01, spin: true });
  const rm = Ragdoll.fromSkeleton(mix.root, dyn);
  const bm = Object.fromEntries(rm.segments.map((s) => [s.name, s]));
  check('mixamo: 17 segments (hands included)', rm.segments.length, 17, 17);
  check('hinge axis: T-pose elbows -Y (left) / +Y (right), mirrored per side', -axisWorld(bm.lowerArmL).y + axisWorld(bm.lowerArmR).y, 2 - 1e-6, 2 + 1e-6);
  check('mixamo: leaf foot points along the toes (forward and down)', bm.footL.dir.z, 0.7, 1);
  check('mixamo: segment lengths in metres despite the cm armature (forearm)', bm.lowerArmL.length, 0.24, 0.26, 'm');
}

// ---- 3. activation at rest, round trip, posed activation ------------------------------------------------------------------
{
  const rig = buildRig(PROC);
  const { dyn } = newWorld();
  const rd = Ragdoll.fromSkeleton(rig.root, dyn);
  const before = snapshot(rd);
  rd.activate();
  const S = rd.segments;
  check('activate: one body per segment', dyn.bodies.size, S.length, S.length);
  check('activate: joints = segments - 1', dyn.joints.size, S.length - 1, S.length - 1);
  yes('activate: active, hips id = the first body, matrixAutoUpdate off', rd.active && rd.hips === S[0].body && S.every((s) => s.bone.matrixAutoUpdate === false));
  // capsule centres from the table: chain segments halfway to the child, leaves along their default axis
  const childOf = { hips: 'spine', spine: 'chest', chest: 'neck', neck: 'head', upperArmL: 'lowerArmL', upperArmR: 'lowerArmR', upperLegL: 'lowerLegL', lowerLegL: 'footL', upperLegR: 'lowerLegR', lowerLegR: 'footR' };
  const leaf = { head: [0, 0.09, 0], lowerArmL: [0, -0.13, 0], lowerArmR: [0, -0.13, 0], footL: [0, 0, 0.11], footR: [0, 0, 0.11] };
  const p = new Vector3(), q = new Quaternion();
  let cErr = 0, aErr = 0;
  for (const s of S) {
    const want = childOf[s.name] ? P[s.name].clone().lerp(P[childOf[s.name]], 0.5) : P[s.name].clone().add(new Vector3(...leaf[s.name]));
    dyn.getPose(s.body, p, q);
    cErr = Math.max(cErr, p.distanceTo(want));
    aErr = Math.max(aErr, new Vector3(0, 1, 0).applyQuaternion(q).distanceTo(s.dir));
  }
  check('activate at rest: body positions at the capsule centres', cErr, 0, 1e-6, 'm');
  check('activate at rest: capsule +Y along each segment', aErr, 0, 1e-9);
  check('activate at rest: joint anchors coincide', anchorGap(dyn, rd), 0, 1e-9, 'm');
  rd.update();
  check('round trip (procedural): bone world matrices after update()', maxMatErr(rd, before), 0, 1e-6);
  check('round trip: hips bone keeps its local position', rig.by.hips.position.distanceTo(new Vector3(0, 0.98, 0)), 0, 1e-9, 'm');
  check('position() while active: the hips bone origin its body gives', rd.position(new Vector3()).distanceTo(P.hips), 0, 1e-9, 'm');
  // scaled, rotated bone frames
  for (const [label, table, o] of [['mixamo (cm armature, rotated bones)', MIX, { scale: 0.01, spin: true }], ['RobotExpressive-style (feet under the root)', ROBOT, { spin: true, seed: 5 }]]) {
    const r2 = buildRig(table, o);
    const w2 = newWorld();
    const d2 = Ragdoll.fromSkeleton(r2.root, w2.dyn);
    const b2 = snapshot(d2);
    d2.activate(); d2.update();
    check(`round trip (${label})`, maxMatErr(d2, b2), 0, 1e-6);
  }
  // posed activation: knee bent 0.8 rad (the rig's +X), elbow 0.5 rad forward (-X): hinge angles read the flexion
  const r3 = buildRig(PROC);
  const w3 = newWorld();
  const d3 = Ragdoll.fromSkeleton(r3.root, w3.dyn);   // rest = the pose now
  r3.by.lowerLegL.rotation.x = 0.8; r3.by.lowerArmR.rotation.x = -0.5; r3.by.upperArmL.rotation.z = 0.3; r3.by.spine.rotation.x = 0.2;
  r3.root.updateMatrixWorld(true);
  const b3 = snapshot(d3);
  d3.activate();
  const by3 = Object.fromEntries(d3.segments.map((s) => [s.name, s]));
  check('posed: knee hinge angle = +0.8 (flexion positive, from rest)', jointAngle(w3.dyn.joints, w3.dyn.bodies, by3.lowerLegL.joint), 0.8 - 1e-9, 0.8 + 1e-9, 'rad');
  check('posed: elbow hinge angle = +0.5 (flexion positive, from rest)', jointAngle(w3.dyn.joints, w3.dyn.bodies, by3.lowerArmR.joint), 0.5 - 1e-9, 0.5 + 1e-9, 'rad');
  check('posed: straight knee R reads 0', Math.abs(jointAngle(w3.dyn.joints, w3.dyn.bodies, by3.lowerLegR.joint)), 0, 1e-9, 'rad');
  check('posed: shoulder swing = the arm\'s 0.3 rad from rest', jointSwing(w3.dyn.joints, w3.dyn.bodies, by3.upperArmL.joint), 0.3 - 1e-9, 0.3 + 1e-9, 'rad');
  check('posed: joint anchors coincide', anchorGap(w3.dyn, d3), 0, 1e-9, 'm');
  d3.update();
  check('posed: round trip of the posed bones', maxMatErr(d3, b3), 0, 1e-6);
  // fromPose false: the rest pose, carried by the hips' current transform
  const r4 = buildRig(PROC);
  const w4 = newWorld();
  const d4 = Ragdoll.fromSkeleton(r4.root, w4.dyn);
  r4.root.position.set(2, 0, -1); r4.root.rotation.y = 0.7; r4.by.lowerLegL.rotation.x = 1.0;
  r4.root.updateMatrixWorld(true);
  d4.activate({ fromPose: false, velocity: new Vector3(0, 0, 1) });
  const by4 = Object.fromEntries(d4.segments.map((s) => [s.name, s]));
  check('fromPose false: knee at its rest angle (0)', Math.abs(jointAngle(w4.dyn.joints, w4.dyn.bodies, by4.lowerLegL.joint)), 0, 1e-9, 'rad');
  const v = new Vector3();
  w4.dyn.getVelocity(by4.footR.body, v, null);
  check('activate velocity: every body gets it (foot vz)', v.z, 1 - 1e-12, 1 + 1e-12, 'm/s');
}

// ---- 3b. activation from a pose outside the limits; bodies without a pose yet -------------------------------------------
{
  const rig = buildRig(PROC);
  const { dyn } = newWorld();
  const rd = Ragdoll.fromSkeleton(rig.root, dyn);
  const by = Object.fromEntries(rd.segments.map((s) => [s.name, s]));
  rig.by.upperArmL.rotation.z = 2.5;    // arm swung out and up 2.5 rad (shoulder cone 1.5)
  rig.by.upperLegL.rotation.x = -1.25;  // thigh forward 1.25 rad (hip cone 1.0)
  rig.by.lowerLegR.rotation.x = -0.5;   // knee hyperextended 0.5 rad (hinge [0, 2.4])
  rig.by.upperArmR.rotation.y = 1.5;    // arm twisted 1.5 rad about itself (shoulder twist 0.8)
  rig.by.lowerArmL.rotation.y = 0.6;    // forearm turned off its hinge axis (a hinge has no swing)
  rig.root.updateMatrixWorld(true);
  rd.activate();
  const J = dyn.joints, B = dyn.bodies;
  check('limits at activation: shoulder swing clamped to the cone (1.5)', jointSwing(J, B, by.upperArmL.joint), 0, 1.5 + 1e-6, 'rad');
  check('limits at activation: hip swing clamped to the cone (1.0)', jointSwing(J, B, by.upperLegL.joint), 0, 1 + 1e-6, 'rad');
  check('limits at activation: hyperextended knee brought to 0', jointAngle(J, B, by.lowerLegR.joint), -1e-6, 2.4, 'rad');
  check('limits at activation: shoulder twist clamped (|twist| <= 0.8)', Math.abs(jointAngle(J, B, by.upperArmR.joint)), 0, 0.8 + 1e-6, 'rad');
  check('limits at activation: elbow back on its hinge axis (swing)', jointSwing(J, B, by.lowerArmL.joint), 0, 1e-6, 'rad');
  check('limits at activation: joint anchors still coincide', anchorGap(dyn, rd), 0, 1e-9, 'm');
  check('limits at activation: a joint within its limits is left as posed (knee L)', Math.abs(jointAngle(J, B, by.lowerLegL.joint)), 0, 1e-9, 'rad');
  dyn.step(DT);
  const v = new Vector3(), w = new Vector3();
  let vmax = 0, wmax = 0;
  for (const s of rd.segments) { dyn.getVelocity(s.body, v, w); vmax = Math.max(vmax, v.length()); wmax = Math.max(wmax, w.length()); }
  check('limits at activation: first step max |v| (was ~25 m/s unclamped)', vmax, 0, 3, 'm/s');
  check('limits at activation: first step max |w|', wmax, 0, 20, 'rad/s');
  // a worker-mode client returns false from getPose until its snapshot has the body: update() leaves those bones
  const r2 = buildRig(PROC);
  const w2 = newWorld();
  let pending = true;
  const stub = new Proxy(w2.dyn, {
    get(t, k) {
      if (k === 'getPose') return (id, p, q) => (pending ? false : t.getPose(id, p, q));
      const val = t[k];
      return typeof val === 'function' ? val.bind(t) : val;
    },
  });
  const d2 = Ragdoll.fromSkeleton(r2.root, stub);
  const before = snapshot(d2);
  d2.activate({ velocity: [3, 0, 0] });
  w2.dyn.step(DT);
  d2.update();
  check('getPose false (no snapshot yet): bones stay as they were', maxMatErr(d2, before), 0, 0);
  check('getPose false: position() falls back to the hips bone', d2.position(new Vector3()).distanceTo(P.hips), 0, 1e-12, 'm');
  pending = false; d2.update();
  check('getPose answering again: the bones follow (hips moved with 3 m/s)', worldPos(r2.by.hips).x, 0.03, 0.08, 'm');
}

// ---- 4. a doll on the ground ---------------------------------------------------------------------------------------------
function drop(table, rigOpts, seconds, onStep) {
  const rig = buildRig(table, rigOpts);
  const { dyn } = newWorld();
  const rd = Ragdoll.fromSkeleton(rig.root, dyn);
  rd.activate({ velocity: [0.8, 0.4, -1.2] });
  const n = Math.round(seconds / DT);
  for (let i = 0; i < n; i++) { dyn.step(DT); rd.update(); if (onStep) onStep(i, dyn, rd); }
  return { rig, dyn, rd };
}
{
  const v = new Vector3(), p = new Vector3();
  let nan = false, maxV = 0, maxGap = 0, minBottom = Infinity, kneeLo = Infinity, kneeHi = -Infinity, restAt = NaN, hipsAt2 = NaN;
  const qq = new Quaternion(), ax = new Vector3(), hips6 = new Vector3(), hips8 = new Vector3();
  let endBottom = NaN;
  const minBottomNow = (w, r) => {
    let m = Infinity;
    for (const s of r.segments) { w.getPose(s.body, p, qq); ax.set(0, 1, 0).applyQuaternion(qq); m = Math.min(m, p.y - s.radius - s.halfHeight * Math.abs(ax.y)); }
    return m;
  };
  const { dyn, rd } = drop(PROC, {}, 8, (i, w, r) => {
    for (const s of r.segments) {
      w.getVelocity(s.body, v, null); w.getPose(s.body, p, qq);
      if (![v.x, v.y, v.z, p.x, p.y, p.z].every(Number.isFinite)) nan = true;
      maxV = Math.max(maxV, v.length());
      ax.set(0, 1, 0).applyQuaternion(qq);
      minBottom = Math.min(minBottom, p.y - s.radius - s.halfHeight * Math.abs(ax.y)); // lowest point of the capsule
      if (/^lowerLeg/.test(s.name)) { const a = jointAngle(w.joints, w.bodies, s.joint); kneeLo = Math.min(kneeLo, a); kneeHi = Math.max(kneeHi, a); }
    }
    maxGap = Math.max(maxGap, anchorGap(w, r));
    if (i === Math.round(2 / DT) - 1) hipsAt2 = r.position(p).y;
    if (i === Math.round(6 / DT) - 1) hips6.copy(r.position(p));
    if (i === Math.round(8 / DT) - 1) { hips8.copy(r.position(p)); endBottom = minBottomNow(w, r); }
    if (Number.isNaN(restAt) && r.isResting()) restAt = (i + 1) * DT;
  });
  yes('ground: no NaN in 8 s', !nan);
  check('ground: max |v| of any body', maxV, 0, 30, 'm/s');
  check('ground: hips below 1.2 m after 2 s (fell)', hipsAt2, -1, 1.2, 'm');
  check('ground: isResting reached (s)', restAt, 0, 6, 's');
  check('ground: joints hold (max anchor gap over the run)', maxGap, 0, 0.01, 'm');
  check('ground: knees stay within their limits (min, rad)', kneeLo, -0.05, 2.45);
  check('ground: knees stay within their limits (max, rad)', kneeHi, -0.05, 2.45);
  check('ground: lowest point of any capsule during the fall (impacts)', minBottom, -0.06, 1, 'm');
  check('ground: lowest point of any capsule at rest (8 s)', endBottom, -0.005, 0.2, 'm');
  check('ground: settled, hips move < 5 cm from 6 s to 8 s', hips6.distanceTo(hips8), 0, 0.05, 'm');
  // the bones follow: hips bone origin = hips body pose applied to the bone-space offset
  const hs = rd.segments[0], q = new Quaternion();
  dyn.getPose(hs.body, p, q);
  const want = p.clone().sub(hs.offset.clone().multiply(hs._scale).applyQuaternion(q.clone().multiply(hs.rotation.clone().invert())));
  check('ground: hips bone follows its body', worldPos(hs.bone).distanceTo(want), 0, 1e-9, 'm');
  // determinism: two runs, every body pose every step
  const f64 = new Float64Array(7), u32 = new Uint32Array(f64.buffer);
  const hashRun = () => {
    let h = 2166136261;
    const pp = new Vector3(), qq = new Quaternion();
    drop(PROC, {}, 3, (i, w, r) => {
      for (const s of r.segments) {
        w.getPose(s.body, pp, qq);
        f64[0] = pp.x; f64[1] = pp.y; f64[2] = pp.z; f64[3] = qq.x; f64[4] = qq.y; f64[5] = qq.z; f64[6] = qq.w;
        for (let k = 0; k < 14; k++) h = Math.imul(h ^ u32[k], 16777619) >>> 0;
      }
    });
    return h;
  };
  const h1 = hashRun(), h2 = hashRun();
  yes(`determinism: dropped doll pose hash equal across runs (${h1.toString(16)})`, h1 === h2);
  // golden: the procedural doll dropped from rest for 3 s (Ragdoll sizing, joints and activation + the DynamicsWorld
  // solver); an intended change to either updates it. Only + - * / sqrt on this path: the same on every platform.
  const GOLDEN = 0x9e623b70;
  check(`determinism: golden hash (${GOLDEN.toString(16)}; got ${h1.toString(16)})`, h1 === GOLDEN ? 1 : 0, 1, 1);
  const pr = rd.position(new Vector3()), hb = worldPos(rd.segments[0].bone);
  check('position(): = the hips bone origin update() wrote', pr.distanceTo(hb), 0, 1e-9, 'm');
  for (const [label, table, o] of [['mixamo', MIX, { scale: 0.01, spin: true }], ['RobotExpressive-style', ROBOT, { spin: true, seed: 5 }]]) {
    let bad = false, rest = NaN, vmax = 0;
    drop(table, o, 6, (i, w, r) => {
      for (const s of r.segments) { w.getVelocity(s.body, v, null); if (!Number.isFinite(v.x)) bad = true; vmax = Math.max(vmax, v.length()); }
      if (Number.isNaN(rest) && r.isResting()) rest = (i + 1) * DT;
    });
    yes(`ground (${label}): no NaN`, !bad);
    check(`ground (${label}): max |v|`, vmax, 0, 30, 'm/s');
    check(`ground (${label}): isResting reached (s)`, rest, 0, 6, 's');
  }

  // ---- 5. deactivate / reactivate / dispose ---------------------------------------------------------------------------
  const last = snapshot(rd);
  rd.deactivate();
  check('deactivate: bodies removed', dyn.bodies.size, 0, 0);
  check('deactivate: joints removed', dyn.joints.size, 0, 0);
  dyn.step(DT);
  check('deactivate: world.stats.bodies back to 0', dyn.stats.bodies, 0, 0);
  yes('deactivate: matrixAutoUpdate restored, active false, hips -1', rd.segments.every((s) => s.bone.matrixAutoUpdate === true) && !rd.active && rd.hips === -1);
  rd.root.updateMatrixWorld(true);
  check('deactivate: bones keep the last ragdoll pose (scene update)', maxMatErr(rd, last), 0, 1e-6);
  rd.activate();
  check('reactivate: bodies again', dyn.bodies.size, rd.segments.length, rd.segments.length);
  rd.dispose();
  yes('dispose: inactive, bodies removed, world released', !rd.active && rd.disposed && dyn.bodies.size === 0 && rd.dyn === null);
}

// ---- 6. options ------------------------------------------------------------------------------------------------------------
{
  const rig = buildRig(PROC);
  const { dyn } = newWorld();
  const a = Ragdoll.fromSkeleton(rig.root, dyn);
  const b = Ragdoll.fromSkeleton(rig.root, dyn, { density: 1970, radius: { head: 0.1, upperArm: 0.05 }, length: { head: 0.25 }, limits: { knee: [0, 1], shoulder: { swing: 0.5, twist: 0.2 } } });
  const bb = Object.fromEntries(b.segments.map((s) => [s.name, s]));
  check('options: density scales the mass (x2)', b.segments[1].mass / a.segments[1].mass, 2 - 1e-12, 2 + 1e-12);
  check('options: radius map by segment / role (head 0.1, upperArmR 0.05)', Math.abs(bb.head.radius - 0.1) + Math.abs(bb.upperArmR.radius - 0.05), 0, 1e-12);
  check('options: leaf length (head 0.25)', bb.head.length, 0.25, 0.25);
  const c = Ragdoll.fromSkeleton(rig.root, dyn, { radius: 0.06 });
  check('options: one radius for every segment', Math.max(...c.segments.map((s) => Math.abs(s.radius - 0.06))), 0, 0);
  b.activate();
  const J = dyn.joints, B = dyn.bodies;
  check('options: knee hinge limits [0, 1] reach the joint', Math.abs(J.lo[bb.lowerLegL.joint]) + Math.abs(J.hi[bb.lowerLegL.joint] - 1), 0, 0);
  check('options: shoulder swing 0.5 / twist 0.2 reach the joint', Math.abs(J.swingHi[bb.upperArmL.joint] - 0.5) + Math.abs(J.twistHi[bb.upperArmL.joint] - 0.2) + Math.abs(J.twistLo[bb.upperArmL.joint] + 0.2), 0, 1e-15);
  const m = Ragdoll.fromSkeleton(rig.root, dyn, { limits: { shoulder: { swing: 0.5 }, hip: null } });
  const mm = Object.fromEntries(m.segments.map((s2) => [s2.name, s2]));
  yes('options: limits merge per field (swing set, default twist 0.8 kept; null frees a joint)',
    mm.upperArmL.swingMax === 0.5 && mm.upperArmL.twistHi === DEFAULT_LIMITS.shoulder.twist && mm.upperLegL.swingMax === Infinity && mm.upperLegL.twistHi === Infinity);
  check('defaults: elbow limits [0, 2.4] (flexion)', Math.abs(J.lo[bb.lowerArmL.joint] - DEFAULT_LIMITS.elbow[0]) + Math.abs(J.hi[bb.lowerArmL.joint] - DEFAULT_LIMITS.elbow[1]), 0, 0);
  const i0 = bb.hips.body, i1 = bb.chest.body;
  yes('defaults: kind ragdoll, friction 0.6, damping 0.05 / 0.1', dyn.kindOf(i0) === 'ragdoll' && B.friction[i0] === 0.6 && B.linearDamping[i0] === 0.05 && B.angularDamping[i0] === 0.1);
  check('defaults: ragdoll bodies do not collide with each other (layer & mask)', (B.layer[i0] & B.mask[i1]) | (B.layer[i1] & B.mask[i0]), 0, 0);
  check('defaults: ragdoll bodies collide with default-layer bodies', B.mask[i0] & 1, 1, 1);
  b.deactivate();
  const d = Ragdoll.fromSkeleton(rig.root, dyn, { layer: 4, mask: -1, kind: 'doll', userData: { who: 1 } });
  d.activate();
  yes('options: layer / mask / kind / userData reach the bodies', B.layer[d.hips] === 4 && B.mask[d.hips] === -1 && dyn.kindOf(d.hips) === 'doll' && B.userData[d.hips].who === 1);
  d.deactivate();
  // an in-thread DynamicsClient
  const physics = new PhysicsWorld({ groundLevel: 0 });
  const client = new DynamicsClient({ physics, worker: false });
  const r5 = buildRig(PROC);
  const e = Ragdoll.fromSkeleton(r5.root, client);
  const b5 = snapshot(e);
  e.activate(); e.update();
  check('DynamicsClient (in-thread): round trip', maxMatErr(e, b5), 0, 1e-6);
  for (let i = 0; i < 60; i++) { client.step(DT); e.update(); }
  check('DynamicsClient (in-thread): the doll moves (hips height after 1 s)', e.position(new Vector3()).y, -1, 1.0, 'm');
  e.deactivate();
  check('DynamicsClient (in-thread): deactivate removes the bodies', client.bodies.size, 0, 0);
}

// ---- 6b. handing the bones back to the mixer: AnimationController.resync() ------------------------------------------------
{
  const rig = buildRig(PROC);
  const qa = new Quaternion(0.1, 0, 0, 1).normalize();   // a held hips tilt (a constant track)
  const clip = new AnimationClip('hold', 1, [new QuaternionKeyframeTrack('hips.quaternion', [0, 1], [...qa.toArray(), ...qa.toArray()])]);
  const anim = new AnimationController(rig.root, [clip], { states: { hold: { clip: 'hold' } } });
  anim.play('hold'); anim.update(DT); anim.update(DT);
  const hips = rig.by.hips;
  const qd = (q) => Math.max(Math.abs(q.x - qa.x), Math.abs(q.y - qa.y), Math.abs(q.z - qa.z), Math.abs(q.w - qa.w)); // track values are float32
  check('mixer: the constant track holds the hips', qd(hips.quaternion), 0, 1e-6);
  hips.quaternion.set(0, 0, 0, 1);                      // written by hand (what a ragdoll's deactivate does)
  anim.update(DT);
  check('mixer: a constant channel is not rewritten after a hand write (three)', qd(hips.quaternion), 0.01, 1);
  anim.resync(); anim.update(DT);
  check('resync(): the next update writes it again', qd(hips.quaternion), 0, 1e-6);
  const { dyn } = newWorld();
  const rd = Ragdoll.fromSkeleton(rig.root, dyn);
  rd.activate({ velocity: [0, 0, -1.5] });
  for (let i = 0; i < 90; i++) { dyn.step(DT); rd.update(); }
  rd.deactivate({ controller: anim });
  const moved = qd(hips.quaternion);
  anim.update(DT);
  check('deactivate({ controller }): the ragdoll had turned the hips', moved, 0.05, 2);
  check('deactivate({ controller }): the mixer drives them again', qd(hips.quaternion), 0, 1e-6);
}

// ---- 7. allocation of update() -------------------------------------------------------------------------------------------
{
  const v8 = await import('node:v8'), vm = await import('node:vm');
  let gc = globalThis.gc;
  if (!gc) { try { v8.setFlagsFromString('--expose-gc'); gc = vm.runInNewContext('gc'); } catch { gc = null; } }
  const exact = v8.getHeapStatistics().total_allocated_bytes !== undefined;   // Node >= 26
  const allocated = exact ? () => v8.getHeapStatistics().total_allocated_bytes : () => process.memoryUsage().heapUsed;
  const { rd } = drop(PROC, {}, 1);
  const N = 100000;
  const loop = (n) => { for (let i = 0; i < n; i++) rd.update(); };
  loop(N);                                          // warm: the loop and update() optimised before measuring
  let probe = Infinity;                             // the counter read itself (a stats object)
  for (let k = 0; k < 5; k++) { const a = allocated(); probe = Math.min(probe, allocated() - a); }
  if (gc) gc();
  const h0 = process.memoryUsage().heapUsed;
  const t0 = allocated();
  loop(N);
  const per = (allocated() - t0 - probe) / N;
  if (gc) gc();
  const h1 = process.memoryUsage().heapUsed;
  // a real allocation per bone (an Euler update, a boxed number per call) reads >= 200 B per update()
  check('alloc: bytes allocated per update() (100 000 calls, heap counter)', per, -Infinity, 16, 'B');
  check('alloc: retained heap growth over 100 000 update() calls (gc before/after)', (h1 - h0) / 1024, -Infinity, 64, 'KB');
  note(`alloc: ${per.toFixed(2)} B per update() ${exact ? '(total_allocated_bytes)' : '(heapUsed delta, approximate: Node < 26)'}, probe ${probe} B${gc ? '' : '; no gc available'}`);
  // every allocation sampled (garbage included) and attributed: nothing in Ragdoll.js or three's math it calls
  try {
    const inspector = await import('node:inspector/promises');
    const s = new inspector.Session(); s.connect();
    await s.post('HeapProfiler.enable');
    await s.post('HeapProfiler.startSampling', { samplingInterval: 16, includeObjectsCollectedByMajorGC: true, includeObjectsCollectedByMinorGC: true });
    for (let i = 0; i < 3000; i++) rd.update();
    const { profile } = await s.post('HeapProfiler.stopSampling');
    await s.post('HeapProfiler.disable'); s.disconnect();
    let mine = 0;
    const walk = (n) => { if (n.selfSize && /Ragdoll\.js$|three\.(core|module)\.(min\.)?js$|DynamicsWorld\.js$/.test(n.callFrame.url)) mine += n.selfSize; n.children.forEach(walk); };
    walk(profile.head);
    check('alloc: sampled bytes in Ragdoll.js / three / getPose over 3000 update() calls', mine, 0, 0, 'B');
  } catch (err) { note(`allocation sampling unavailable: ${err.message}`); }
  rd.deactivate();
}

console.log(results.join('\n'));
console.log(failures ? `\n${failures} of ${count} check(s) FAILED` : `\nall ${count} checks passed`);
process.exit(failures ? 1 : 0);
