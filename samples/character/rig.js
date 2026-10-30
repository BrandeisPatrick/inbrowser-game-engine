// Character visuals for the character sample: a procedural skinned humanoid with four generated clips (no assets
// needed), and the mapping of a loaded glTF (RobotExpressive or any rigged model) onto the clip names the sample's
// AnimationController states use. Both face +Z (the glTF convention) with their feet at y = 0.
//   buildProceduralCharacter({ height = 1.8, color, accent }) -> { root, skinnedMesh, skeleton, clips, headBone }
//   mapGltfCharacter(gltf, { height = 1.8 })                 -> { root, clips, headBone, scale }
// The procedural rig: 15 THREE.Bones (hips, spine, chest, neck, head, upper / lower arms, upper / lower legs, feet),
// one merged SkinnedMesh of capsule / sphere / box segments (each 100 % weighted to its bone, vertex coloured, one
// draw), and QuaternionKeyframeTracks (plus a hips position track for the bob) built by sampling pose functions.
// Conventions (character space, faces +Z, its left side is +X): a bone's local -X rotation swings a hanging limb
// forward, +X bends a knee, -X bends an elbow forward, +X leans the spine forward, -X tips the head up.
import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

/** Clip names the sample's AnimationController states expect (RobotExpressive's own names). */
export const CLIP_NAMES = ['Idle', 'Walking', 'Running', 'Jump'];

// ------------------------------------------------------------------ skeleton (1.8 m reference, model space)
// [name, parent, joint position at rest]
const JOINTS = [
  ['hips', null, [0, 0.98, 0]],
  ['spine', 'hips', [0, 1.1, 0]],
  ['chest', 'spine', [0, 1.3, 0]],
  ['neck', 'chest', [0, 1.5, 0]],
  ['head', 'neck', [0, 1.6, 0]],
  ['upperArmL', 'chest', [0.25, 1.45, 0]],
  ['lowerArmL', 'upperArmL', [0.25, 1.17, 0]],
  ['upperArmR', 'chest', [-0.25, 1.45, 0]],
  ['lowerArmR', 'upperArmR', [-0.25, 1.17, 0]],
  ['upperLegL', 'hips', [0.1, 0.93, 0]],
  ['lowerLegL', 'upperLegL', [0.1, 0.49, 0]],
  ['footL', 'lowerLegL', [0.1, 0.08, 0]],
  ['upperLegR', 'hips', [-0.1, 0.93, 0]],
  ['lowerLegR', 'upperLegR', [-0.1, 0.49, 0]],
  ['footR', 'lowerLegR', [-0.1, 0.08, 0]],
];
const BONES = JOINTS.map((j) => j[0]);
const BONE_INDEX = Object.fromEntries(BONES.map((n, i) => [n, i]));
const JOINT_POS = Object.fromEntries(JOINTS.map((j) => [j[0], j[2]]));
const HIPS_Y = JOINT_POS.hips[1];

// ------------------------------------------------------------------ geometry
const _col = new THREE.Color();
/** Tag a segment: delete uv, vertex colour, 100 % weight to `bone`. */
function seg(g, bone, color) {
  g.deleteAttribute('uv');
  _col.set(color);
  const n = g.attributes.position.count, c = new Float32Array(n * 3), si = new Uint16Array(n * 4), sw = new Float32Array(n * 4);
  const b = BONE_INDEX[bone];
  for (let i = 0; i < n; i++) { c[i * 3] = _col.r; c[i * 3 + 1] = _col.g; c[i * 3 + 2] = _col.b; si[i * 4] = b; sw[i * 4] = 1; }
  g.setAttribute('color', new THREE.BufferAttribute(c, 3));
  g.setAttribute('skinIndex', new THREE.Uint16BufferAttribute(si, 4));
  g.setAttribute('skinWeight', new THREE.Float32BufferAttribute(sw, 4));
  return g;
}
/** Capsule of radius r, straight length len along axis 'x' | 'y' | 'z', scaled then centred at (x, y, z). */
function cap(r, len, axis, x, y, z, sx = 1, sy = 1, sz = 1) {
  const g = new THREE.CapsuleGeometry(r, len, 5, 14);
  if (axis === 'x') g.rotateZ(Math.PI / 2); else if (axis === 'z') g.rotateX(Math.PI / 2);
  return g.scale(sx, sy, sz).translate(x, y, z);
}
const ball = (r, x, y, z, sx = 1, sy = 1, sz = 1) => new THREE.SphereGeometry(r, 16, 12).scale(sx, sy, sz).translate(x, y, z);
const block = (w, h, d, x, y, z) => new THREE.BoxGeometry(w, h, d).translate(x, y, z);

function bodyGeometry(main, accent) {
  const DARK = '#2a2f39', JOINT = '#474e5c', VISOR = '#0f1217';
  const P = [
    seg(cap(0.115, 0.13, 'x', 0, 0.95, 0, 1, 0.9, 0.85), 'hips', DARK),
    seg(cap(0.12, 0.05, 'y', 0, 1.13, 0, 1, 1, 0.78), 'spine', JOINT),
    seg(cap(0.14, 0.17, 'x', 0, 1.36, 0, 1, 1.18, 0.74), 'chest', main),
    seg(block(0.13, 0.05, 0.03, 0, 1.4, 0.104), 'chest', accent),             // chest stripe (front)
    seg(block(0.2, 0.22, 0.07, 0, 1.35, -0.115), 'chest', JOINT),            // back pack (reads the facing)
    seg(new THREE.CylinderGeometry(0.05, 0.055, 0.12, 12).translate(0, 1.53, 0), 'neck', JOINT),
    seg(ball(0.118, 0, 1.68, 0, 0.95, 1.08, 1), 'head', main),
    seg(cap(0.034, 0.1, 'x', 0, 1.7, 0.092, 1, 1, 0.8), 'head', VISOR),
  ];
  for (const s of [1, -1]) {
    const L = s > 0 ? 'L' : 'R';
    P.push(
      seg(ball(0.068, s * 0.25, 1.44, 0), `upperArm${L}`, main),
      seg(cap(0.052, 0.17, 'y', s * 0.25, 1.3, 0), `upperArm${L}`, main),
      seg(cap(0.045, 0.15, 'y', s * 0.25, 1.03, 0), `lowerArm${L}`, main),
      seg(ball(0.055, s * 0.25, 0.85, 0.005, 0.75, 1.15, 0.95), `lowerArm${L}`, DARK),
      seg(cap(0.075, 0.27, 'y', s * 0.1, 0.71, 0), `upperLeg${L}`, DARK),
      seg(ball(0.06, s * 0.1, 0.49, 0.045, 1, 1, 0.6), `lowerLeg${L}`, JOINT),
      seg(cap(0.062, 0.28, 'y', s * 0.1, 0.29, 0), `lowerLeg${L}`, DARK),
      seg(cap(0.048, 0.15, 'z', s * 0.1, 0.048, 0.045, 1.1, 0.95, 1), `foot${L}`, accent),
    );
  }
  const g = mergeGeometries(P);
  for (const p of P) p.dispose();
  return g;
}

// ------------------------------------------------------------------ clips (sampled pose functions)
const _e = new THREE.Euler(), _q = new THREE.Quaternion();
const newPose = () => { const P = { hipsPos: [0, 0, 0] }; for (const b of BONES) P[b] = [0, 0, 0]; return P; };
const resetPose = (P) => { P.hipsPos.fill(0); for (const b of BONES) P[b].fill(0); };

/** One clip from key times and fill(i, pose) (Euler XYZ per bone + hips position offset), scaled by s. */
function buildClip(name, times, fill, s) {
  const n = times.length, P = newPose();
  const rot = Object.fromEntries(BONES.map((b) => [b, new Float32Array(n * 4)]));
  const pos = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    resetPose(P); fill(i, P);
    for (const b of BONES) { const r = P[b]; _q.setFromEuler(_e.set(r[0], r[1], r[2], 'XYZ')).toArray(rot[b], i * 4); }
    pos[i * 3] = P.hipsPos[0] * s; pos[i * 3 + 1] = (HIPS_Y + P.hipsPos[1]) * s; pos[i * 3 + 2] = P.hipsPos[2] * s;
  }
  const t = Float32Array.from(times);
  const tracks = BONES.map((b) => new THREE.QuaternionKeyframeTrack(`${b}.quaternion`, t, rot[b]));
  tracks.push(new THREE.VectorKeyframeTrack('hips.position', t, pos));
  return new THREE.AnimationClip(name, times[n - 1], tracks);
}
/** Looping clip: n keys over `duration`, pose(w 0..2pi, P); the last key repeats the first (seamless loop). */
function cycleClip(name, duration, n, pose, s) {
  const times = [];
  for (let i = 0; i <= n; i++) times.push((i / n) * duration);
  return buildClip(name, times, (i, P) => pose(((i % n) / n) * Math.PI * 2, P), s);
}

const set = (r, x, y, z) => { r[0] = x; r[1] = y; r[2] = z; };
const pos = (v) => Math.max(0, v);

function idlePose(w, P) {
  const sw = Math.sin(w);
  set(P.hipsPos, 0.008 * sw, -0.008 + 0.004 * Math.sin(2 * w), 0);
  set(P.hips, 0, 0, 0.012 * sw);
  set(P.spine, 0.02, 0, -0.008 * sw);
  set(P.chest, -0.025 * sw, 0, 0);                                    // breathing
  set(P.head, -0.02 + 0.03 * Math.sin(2 * w), 0.12 * Math.sin(w + 0.6), 0); // slow look around
  for (const s of [1, -1]) {
    const L = s > 0 ? 'L' : 'R';
    set(P[`upperArm${L}`], -0.03 + 0.02 * sw, 0, s * (0.1 + 0.02 * sw));
    set(P[`lowerArm${L}`], -0.15, 0, 0);
    set(P[`upperLeg${L}`], -0.04, 0, -s * 0.012 * sw);
    set(P[`lowerLeg${L}`], 0.08, 0, 0);
    set(P[`foot${L}`], -0.04, 0, 0);
  }
}

// gait: legs opposed, arms opposed to the same-side leg; knee flexes in early swing, foot kept near level, toe-off push
const WALK = { thigh: 0.42, thighBias: 0.03, kneeBase: 0.06, knee: 0.62, kneeShift: 0.4, footComp: 0.85, toe: 0.35, arm: 0.38, armBias: 0.02,
  abduct: 0.07, elbow: 0.22, elbowSwing: 0.22, lean: 0.03, lean2: 0.02, pelvisYaw: 0.08, chestYaw: 0.11, bob: 0.018, bobBase: -0.026, bobSign: 1, sway: 0.015 };
const RUN = { thigh: 0.62, thighBias: 0.22, kneeBase: 0.35, knee: 1.3, kneeShift: 0.55, footComp: 0.6, toe: 0.5, arm: 0.72, armBias: 0.15,
  abduct: 0.12, elbow: 1.25, elbowSwing: 0.25, lean: 0.14, lean2: 0.06, pelvisYaw: 0.12, chestYaw: 0.2, bob: 0.03, bobBase: 0.015, bobSign: -1, sway: 0.008 };

function gaitPose(w, P, g) {
  const sw = Math.sin(w);
  // walk: hips highest with the stance leg vertical; run: lowest at mid-stance, highest in the flight phase
  set(P.hipsPos, -g.sway * Math.cos(w), g.bobBase + g.bobSign * g.bob * Math.cos(2 * w), 0);
  set(P.hips, 0, -g.pelvisYaw * sw, 0);
  set(P.spine, g.lean, 0, 0);
  set(P.chest, g.lean2, (g.chestYaw - g.pelvisYaw * 0.4) * sw, 0);
  set(P.head, -(g.lean + g.lean2) * 0.8, -(g.chestYaw - g.pelvisYaw) * sw, 0);
  for (const s of [1, -1]) {
    const L = s > 0 ? 'L' : 'R', ph = s > 0 ? w : w + Math.PI, sp = Math.sin(ph);
    const th = g.thighBias + g.thigh * sp;                                   // thigh forward angle
    const kn = g.kneeBase + g.knee * pos(Math.cos(ph + g.kneeShift)) ** 2;   // knee flexion
    const toe = g.toe * pos(-Math.sin(ph - 0.2)) ** 3;                       // push-off just after the leg is fully back
    set(P[`upperLeg${L}`], -th, 0, 0);
    set(P[`lowerLeg${L}`], kn, 0, 0);
    set(P[`foot${L}`], g.footComp * (th - kn) + toe, 0, 0);
    const arm = g.armBias - g.arm * sp;
    set(P[`upperArm${L}`], -arm, 0, s * g.abduct);
    set(P[`lowerArm${L}`], -(g.elbow + g.elbowSwing * pos(-sp)), 0, 0);
  }
}

// jump keys: crouch (push-off) -> extend -> tuck -> reach for the ground -> landing absorb -> recover
const JUMP_TIMES = [0, 0.12, 0.4, 0.72, 0.9, 1.15];
const JUMP_KEYS = [
  { thigh: 0.75, knee: 1.25, foot: -0.5, spine: 0.32, chest: 0.05, arm: -0.7, elbow: 0.25, abduct: 0.12, y: -0.165, head: -0.25 },
  { thigh: -0.05, knee: 0.05, foot: 0.55, spine: -0.04, chest: -0.04, arm: 2.5, elbow: 0.15, abduct: 0.2, y: 0, head: -0.12 },
  { thigh: 1.25, knee: 1.85, foot: -0.25, spine: 0.22, chest: 0.08, arm: 1.0, elbow: 0.9, abduct: 0.25, y: 0.02, head: -0.15 },
  { thigh: 0.35, knee: 0.35, foot: -0.05, spine: 0.08, chest: 0, arm: 0.55, elbow: 0.4, abduct: 0.45, y: 0, head: -0.08 },
  { thigh: 0.8, knee: 1.3, foot: -0.5, spine: 0.35, chest: 0.06, arm: 0.45, elbow: 0.5, abduct: 0.3, y: -0.16, head: -0.25 },
  { thigh: 0.04, knee: 0.08, foot: -0.04, spine: 0.03, chest: 0, arm: 0.03, elbow: 0.15, abduct: 0.1, y: -0.008, head: -0.02 },
];
function jumpPose(i, P) {
  const k = JUMP_KEYS[i];
  set(P.hipsPos, 0, k.y, 0);
  set(P.spine, k.spine, 0, 0);
  set(P.chest, k.chest, 0, 0);
  set(P.head, k.head, 0, 0);
  for (const s of [1, -1]) {
    const L = s > 0 ? 'L' : 'R', a = s > 0 ? 1 : 0.92; // a touch of asymmetry
    set(P[`upperLeg${L}`], -k.thigh * a, 0, 0);
    set(P[`lowerLeg${L}`], k.knee * a, 0, 0);
    set(P[`foot${L}`], k.foot, 0, 0);
    set(P[`upperArm${L}`], -k.arm * a, 0, s * k.abduct);
    set(P[`lowerArm${L}`], -k.elbow, 0, 0);
  }
}

/** The four procedural clips (Idle 3.2 s, Walking 0.9 s, Running 0.66 s loops; Jump 1.15 s one-shot) for height h. */
export function proceduralClips(height = 1.8) {
  const s = height / 1.8;
  return [
    cycleClip('Idle', 3.2, 16, idlePose, s),
    cycleClip('Walking', 0.9, 24, (w, P) => gaitPose(w, P, WALK), s),
    cycleClip('Running', 0.66, 20, (w, P) => gaitPose(w, P, RUN), s),
    buildClip('Jump', JUMP_TIMES, jumpPose, s),
  ];
}

/**
 * Stylised procedural humanoid. color: suit colour, accent: chest stripe + shoes.
 * @returns {{ root: THREE.Group, skinnedMesh: THREE.SkinnedMesh, skeleton: THREE.Skeleton, clips: THREE.AnimationClip[], headBone: THREE.Bone }}
 */
export function buildProceduralCharacter({ height = 1.8, color = '#e4e7ec', accent = '#2fd4bd' } = {}) {
  const s = height / 1.8;
  const geometry = bodyGeometry(color, accent);
  if (s !== 1) geometry.scale(s, s, s);
  const bones = [], byName = {};
  for (const [name, parent, p] of JOINTS) {
    const b = new THREE.Bone(); b.name = name;
    const pp = parent ? JOINT_POS[parent] : [0, 0, 0];
    b.position.set((p[0] - pp[0]) * s, (p[1] - pp[1]) * s, (p[2] - pp[2]) * s);
    if (parent) byName[parent].add(b);
    bones.push(b); byName[name] = b;
  }
  const material = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.5, metalness: 0.08 });
  const skinnedMesh = new THREE.SkinnedMesh(geometry, material);
  skinnedMesh.name = 'proceduralCharacter';
  skinnedMesh.add(bones[0]);
  const skeleton = new THREE.Skeleton(bones);
  skinnedMesh.bind(skeleton); // updates world matrices and computes the inverse bind matrices
  skinnedMesh.castShadow = true; skinnedMesh.receiveShadow = true;
  skinnedMesh.frustumCulled = false; // the bind-pose bounds do not cover the animated limbs
  const root = new THREE.Group(); root.name = 'character';
  root.add(skinnedMesh);
  return { root, skinnedMesh, skeleton, clips: proceduralClips(height), headBone: byName.head };
}

// ------------------------------------------------------------------ glTF characters
const ROLES = [['Idle', 'idle'], ['Walking', 'walk'], ['Running', 'run'], ['Jump', 'jump']];

/** Best clip for a role: exact name, then a name starting with the key, then one containing it. */
function pickClip(src, name, key) {
  const lc = (c) => c.name.toLowerCase();
  return src.find((c) => c.name === name) || src.find((c) => lc(c) === key) || src.find((c) => lc(c).startsWith(key))
    || src.find((c) => lc(c).includes(key)) || null;
}

/** First bone named like a head (not an `_end` leaf), or null. */
export function findHeadBone(root) {
  let head = null;
  root.traverse((o) => { if (!head && o.isBone && /head/i.test(o.name) && !/_end$/i.test(o.name)) head = o; });
  return head;
}

/**
 * Map a loaded glTF ({ scene, animations, instantiate() }) to { root, clips } with the clip names the controller
 * states expect (CLIP_NAMES). RobotExpressive matches by name (Idle, Walking, Running, Jump); other models map by
 * case-insensitive 'idle' / 'walk' / 'run' / 'jump' and fall back to the first clip. The model is instanced (its own
 * skeleton), normalised to `height` with its feet at y = 0 inside the returned root group, and set to cast shadows.
 */
export function mapGltfCharacter(gltf, { height = 1.8 } = {}) {
  let model = typeof gltf.instantiate === 'function' ? gltf.instantiate() : null;
  if (model && !model.isObject3D) model = model.scene || null;
  if (!model) model = gltf.scene || (gltf.scenes && gltf.scenes[0]);
  if (!model) throw new Error('mapGltfCharacter: glTF has no scene');
  const src = gltf.animations || [];
  const clips = [];
  for (const [name, key] of ROLES) {
    const c = pickClip(src, name, key) || src[0];
    if (!c) continue;
    const m = c.name === name ? c : c.clone();
    m.name = name;
    clips.push(m);
  }
  model.traverse((o) => {
    if (!o.isMesh) return;
    o.castShadow = true; o.receiveShadow = true;
    if (o.isSkinnedMesh) o.frustumCulled = false;
  });
  // normalise: bind-pose bounds (skinned vertices) -> `height` tall, feet on y = 0, centred on the origin in XZ
  model.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(model);
  const h = box.max.y - box.min.y;
  const scale = h > 1e-6 && Number.isFinite(h) ? height / h : 1;
  const root = new THREE.Group(); root.name = 'character';
  const fit = new THREE.Group(); fit.name = 'characterFit';
  fit.scale.setScalar(scale);
  fit.position.set(-(box.min.x + box.max.x) / 2 * scale, -box.min.y * scale, -(box.min.z + box.max.z) / 2 * scale);
  fit.add(model); root.add(fit);
  return { root, clips, headBone: findHeadBone(model), scale };
}
