// Headless checks for the character camera rigs (ThirdPersonRig, FirstPersonRig): no browser, no DOM element.
// Run: node test/rigs.mjs
import { PerspectiveCamera, Object3D, Vector3 } from 'three';
import { ThirdPersonRig } from '../src/camera/ThirdPersonRig.js';
import { FirstPersonRig } from '../src/camera/FirstPersonRig.js';
import { CameraCollision } from '../src/camera/CameraRig.js';

const DT = 1 / 60;
let failures = 0;
const results = [];
function check(name, value, lo, hi, unit = '') {
  const ok = Number.isFinite(value) && value >= lo && value <= hi;
  if (!ok) failures++;
  results.push(`${ok ? 'PASS' : 'FAIL'}  ${name.padEnd(52)} ${fmt(value)} ${unit}   [${lo} .. ${hi}]`);
  return ok;
}
const fmt = (x) => (Number.isFinite(x) ? (Math.abs(x) < 1e-3 && x !== 0 ? x.toExponential(2) : x.toFixed(4)).padStart(10) : String(x).padStart(10));

const camera = () => new PerspectiveCamera(60, 16 / 9, 0.1, 1000);
function target(x = 0, y = 0, z = 0, heading = 0) {
  const t = new Object3D();
  t.position.set(x, y, z); t.rotation.y = heading;
  t.updateMatrixWorld(true);
  return t;
}
function run(rig, seconds, each) {
  const n = Math.round(seconds / DT);
  for (let i = 0; i < n; i++) { rig.update(DT); if (each) each(i); }   // each(i) sees the pose after update i
}
const camFwd = (cam, out = new Vector3()) => out.set(0, 0, -1).applyQuaternion(cam.quaternion);
// world with one box, using the out-filling query contract (returns nothing)
const boxWorld = (box) => ({ getGroundHeight: () => 0, queryColliders: (x, z, r, out) => { out.length = 0; out.push(box); } });

// --- ThirdPersonRig -----------------------------------------------------------------
{
  const rig = new ThirdPersonRig(camera(), null);
  rig.setTarget(target());
  rig.look(0, -1e6);
  check('TP pitch clamps at max (look up)', rig.pitch, 1.0 - 1e-12, 1.0 + 1e-12, 'rad');
  rig.look(0, 1e6);
  check('TP pitch clamps at min (look down)', rig.pitch, -1.2 - 1e-12, -1.2 + 1e-12, 'rad');
  const inv = new ThirdPersonRig(camera(), null, { invertY: true, pitch: [-0.5, 0.7] });
  inv.look(0, -1e6);
  check('TP invertY + custom range: mouse up -> min', inv.pitch, -0.5 - 1e-12, -0.5 + 1e-12, 'rad');
  let maxAbs = 0;
  for (let i = 0; i < 500; i++) { rig.look(997, 0); maxAbs = Math.max(maxAbs, Math.abs(rig.yaw)); }
  check('TP yaw stays wrapped in [-pi, pi]', maxAbs, 0, Math.PI, 'rad');
  rig.yaw = 0.3;
  rig.look(-2 * Math.PI / rig.sensitivity, 0);
  check('TP yaw after one full turn (wrapped)', rig.yaw, 0.3 - 1e-9, 0.3 + 1e-9, 'rad');
}

// settle: distance from focus / pivot, looks at the focus, behind the target (also for a rotated target)
for (const [label, shoulder, heading] of [['shoulder 0', 0, 0], ['shoulder 0.4, heading 1', 0.4, 1]]) {
  const cam = camera(), rig = new ThirdPersonRig(cam, null, { shoulder });
  const t = target(3, 0, -2, heading);
  rig.setTarget(t);
  run(rig, 2);
  const focus = t.position.clone().add(new Vector3(0, rig.height, 0));
  const toFocus = focus.clone().sub(cam.position);
  check(`TP settled |cam - focus| - hypot(d, s) (${label})`, toFocus.length() - Math.hypot(rig.distance, shoulder), -1e-3, 1e-3, 'm');
  check(`TP camera looks at the focus, dot (${label})`, camFwd(cam).dot(toFocus.normalize()), 0.99, 1 + 1e-9, '');
  const tFwd = new Vector3(0, 0, -1).applyQuaternion(t.quaternion);
  check(`TP camera behind the target (${label})`, cam.position.clone().sub(t.position).dot(tFwd), -Infinity, -3, 'm');
  check(`TP state.distance (${label})`, rig.state.distance, rig.distance - 1e-6, rig.distance + 1e-6, 'm');
}

// damping: monotonic approach after a target jump
{
  const cam = camera(), rig = new ThirdPersonRig(cam, null);
  const t = target();
  rig.setTarget(t);
  run(rig, 1);
  const offset = cam.position.clone().sub(rig.focus);
  t.position.set(5, 1, -3); t.updateMatrixWorld(true);
  const ideal = t.position.clone().add(new Vector3(0, rig.height, 0)).add(offset);
  const d0 = cam.position.distanceTo(ideal);
  let prev = d0, worst = -Infinity, first = NaN;
  run(rig, 1.5, (i) => {
    const dd = cam.position.distanceTo(ideal);
    worst = Math.max(worst, dd - prev); prev = dd;
    if (i === 0) first = dd;
  });
  check('TP jump: largest per-frame distance increase', worst, -Infinity, 1e-9, 'm');
  check('TP jump: first frame keeps most of the gap', first / d0, 0.5, 1, '');
  check('TP jump: settled after 1.5 s', cam.position.distanceTo(ideal), 0, 1e-3, 'm');
}

// collision: a wall between camera and target pulls the camera in front of it
{
  const wall = { minX: -5, maxX: 5, minY: 0, maxY: 6, minZ: 2, maxZ: 2.4, kind: 0 };
  const cam = camera(), rig = new ThirdPersonRig(cam, null, { physics: boxWorld(wall) });
  const t = target();
  rig.setTarget(t);
  run(rig, 1);
  const focus = new Vector3(0, rig.height, 0);
  check('TP collision: |cam - focus| < wall distance 2', cam.position.distanceTo(focus), rig.minDistance, 2, 'm');
  check('TP collision: camera z in front of the wall', cam.position.z, 0, 2 - 0.1, 'm');
  check('TP collision: state.distance pulled in', rig.state.distance, 0, 2, 'm');
  const free = new ThirdPersonRig(camera(), null, { physics: boxWorld(wall), collide: false });
  free.setTarget(target());
  run(free, 1);
  check('TP collide: false passes through the wall', free.camera.position.z, 2.4, 10, 'm');
  // PhysicsWorld-style query that returns `out`, passed as plain queryColliders option (legacy contract)
  const legacy = new ThirdPersonRig(camera(), null, { getGroundHeight: () => 0, queryColliders: () => [wall] });
  legacy.setTarget(target());
  run(legacy, 1);
  check('TP collision via legacy queryColliders option', legacy.camera.position.z, 0, 2 - 0.1, 'm');
  // ease back out once the wall is gone
  wall.minZ = wall.maxZ = 100;
  run(rig, 0.1);
  const mid = rig.state.distance;
  run(rig, 3);
  check('TP collision: eases out (not instant)', mid, 0, rig.distance - 0.2, 'm');
  check('TP collision: back at full distance', rig.state.distance, rig.distance - 1e-3, rig.distance + 1e-6, 'm');
}

// ground clamp: looking up from a low focus keeps the camera above ground + minY
{
  for (const [label, opts] of [['collide', {}], ['collide: false', { collide: false }]]) {
    const cam = camera(), rig = new ThirdPersonRig(cam, null, { physics: { getGroundHeight: () => 0.5 }, ...opts });
    const t = target(0, 0.5, 0);
    rig.setTarget(t);
    rig.pitch = 1.0;
    let minY = Infinity;
    run(rig, 1, () => { minY = Math.min(minY, cam.position.y); });
    check(`TP ground clamp: min camera y (${label})`, minY, 0.5 + rig.minY - 1e-9, Infinity, 'm');
  }
}

// zoom clamps
{
  const rig = new ThirdPersonRig(camera(), null);
  rig.setTarget(target());
  rig.zoom(100);
  check('TP zoom(100) clamps to max', rig.distance, 8, 8, 'm');
  rig.zoom(1e-4);
  check('TP zoom(1e-4) clamps to min', rig.distance, 1.5, 1.5, 'm');
  rig.distance = 50; rig.update(DT);
  check('TP distance set directly is clamped on update', rig.distance, 8, 8, 'm');
  // wheel zoom through a stand-in element (EventTarget), then dispose removes the listener
  const el = new EventTarget();
  const w = new ThirdPersonRig(camera(), el);
  const wheel = (dy) => el.dispatchEvent(Object.assign(new Event('wheel', { cancelable: true }), { deltaY: dy }));
  wheel(300);
  check('TP wheel zoom (deltaY 300)', w.distance, 4 * Math.exp(0.3) - 1e-9, 4 * Math.exp(0.3) + 1e-9, 'm');
  w.dispose();
  wheel(300);
  check('TP wheel ignored after dispose', w.distance, 4 * Math.exp(0.3) - 1e-9, 4 * Math.exp(0.3) + 1e-9, 'm');
  // pointer lock: mousemove only counts while this element holds the lock
  const pl = new EventTarget();
  pl.ownerDocument = { pointerLockElement: null };
  const p = new ThirdPersonRig(camera(), pl, { pointerLock: true });
  const move = (x) => pl.dispatchEvent(Object.assign(new Event('mousemove'), { movementX: x, movementY: 0 }));
  move(100);
  check('TP pointer lock: move ignored while unlocked', p.yaw, 0, 0, 'rad');
  pl.ownerDocument.pointerLockElement = pl;
  move(100);
  check('TP pointer lock: move turns while locked', p.yaw, -0.25 - 1e-12, -0.25 + 1e-12, 'rad');
  p.dispose();
}

// auto-recentre, inactive rigs, horizontal basis
{
  const mk = (autoRecentre) => { const r = new ThirdPersonRig(camera(), null, { autoRecentre }); r.setTarget(target()); r.yaw = 2; return r; };
  const a = mk(0.5), b = mk(0);
  run(a, 4); run(b, 4);
  check('TP autoRecentre 0.5 s: yaw back to facing', Math.abs(a.yaw), 0, 0.01, 'rad');
  check('TP autoRecentre 0: yaw kept', b.yaw, 2, 2, 'rad');

  const cam = camera(), rig = new ThirdPersonRig(cam, null, { active: false });
  rig.setTarget(target(10, 0, 0));
  run(rig, 0.5);
  rig.look(500, 0);
  check('TP inactive: camera untouched', cam.position.length(), 0, 0, 'm');
  check('TP inactive: look() ignored', rig.yaw, 0, 0, 'rad');
  rig.activate();
  check('TP activate() snaps to the target', cam.position.x, 10 + 0.4 - 1e-9, 10 + 0.4 + 1e-9, 'm');
  rig.yaw = 0.8;
  const f = rig.getForward(new Vector3()), r = rig.getRight(new Vector3());
  check('TP getForward matches yaw', f.dot(new Vector3(-Math.sin(0.8), 0, -Math.cos(0.8))), 1 - 1e-12, 1 + 1e-12, '');
  check('TP getRight orthogonal, unit', Math.abs(f.dot(r)) + Math.abs(r.length() - 1) + Math.abs(r.y), 0, 1e-12, '');
  rig.update(DT);
  const cf = camFwd(cam); cf.y = 0; cf.normalize();
  check('TP camera horizontal forward = getForward', cf.dot(f), 1 - 1e-9, 1 + 1e-9, '');
  const s1 = rig.state; rig.update(DT);
  check('TP state object reused', s1 === rig.state ? 1 : 0, 1, 1, '');
}

// --- FirstPersonRig -----------------------------------------------------------------
{
  const cam = camera(), rig = new FirstPersonRig(cam, null);
  const t = target(1, 2, 3, 0.4);
  rig.setTarget(t);
  rig.update(DT);
  check('FP eye height above target', cam.position.y - 2, 1.65 - 1e-12, 1.65 + 1e-12, 'm');
  check('FP eye over the target (xz)', Math.hypot(cam.position.x - 1, cam.position.z - 3), 0, 1e-12, 'm');
  check('FP setTarget faces the target heading', rig.yaw, 0.4 - 1e-9, 0.4 + 1e-9, 'rad');
  rig.look(0, -1e6);
  check('FP pitch clamps at max', rig.pitch, 1.5 - 1e-12, 1.5 + 1e-12, 'rad');
  rig.look(0, 1e6);
  check('FP pitch clamps at min', rig.pitch, -1.5 - 1e-12, -1.5 + 1e-12, 'rad');
  rig.pitch = 0; rig.yaw = -2.2; rig.update(DT);
  const f = rig.getForward(new Vector3());
  check('FP camera forward = getForward (pitch 0)', camFwd(cam).dot(f), 1 - 1e-9, 1 + 1e-9, '');
  check('FP getForward matches yaw', f.dot(new Vector3(-Math.sin(-2.2), 0, -Math.cos(-2.2))), 1 - 1e-12, 1 + 1e-12, '');
  rig.pitch = 0.6; rig.update(DT);
  const cf = camFwd(cam);
  check('FP camera pitch angle', Math.asin(cf.y), 0.6 - 1e-9, 0.6 + 1e-9, 'rad');
  cf.y = 0; cf.normalize();
  check('FP horizontal forward at pitch 0.6 = getForward', cf.dot(f), 1 - 1e-9, 1 + 1e-9, '');
}

// head bob bounded, fades when airborne, off when bob = 0
{
  const state = { speed: 4, grounded: true };
  const cam = camera(), rig = new FirstPersonRig(cam, null, { bob: 0.05 });
  const t = target(0, 0, 0);
  rig.setTarget(t, () => state);
  let maxDip = 0, maxUp = -Infinity, maxSide = 0;
  run(rig, 3, () => {
    maxDip = Math.max(maxDip, 1.65 - cam.position.y); maxUp = Math.max(maxUp, cam.position.y - 1.65);
    maxSide = Math.max(maxSide, Math.hypot(cam.position.x, cam.position.z));
  });
  check('FP bob: max dip <= bob', maxDip, 0.03, 0.05 + 1e-9, 'm');
  check('FP bob: never above eye height', maxUp, -Infinity, 1e-9, 'm');
  check('FP bob: sway <= bob / 2', maxSide, 0.01, 0.025 + 1e-9, 'm');
  state.grounded = false;
  run(rig, 1);
  check('FP bob: fades when airborne', Math.abs(cam.position.y - 1.65), 0, 1e-3, 'm');
  const still = new FirstPersonRig(camera(), null, { bob: 0 });
  still.setTarget(target(), () => ({ speed: 6, grounded: true }));
  let dev = 0;
  run(still, 1, () => { dev = Math.max(dev, Math.abs(still.camera.position.y - 1.65)); });
  check('FP bob 0: no motion', dev, 0, 0, 'm');
  const fp = new FirstPersonRig(camera(), null, { active: false });
  fp.setTarget(target(5, 0, 0)); fp.update(DT); fp.look(400, 0);
  check('FP inactive: camera untouched / look ignored', fp.camera.position.length() + Math.abs(fp.yaw), 0, 0, '');
  fp.activate();
  check('FP activate() snaps', fp.camera.position.x, 5, 5, 'm');
  fp.dispose();
}

// --- review fixes: turn() conventions, NaN guard, direct pitch writes, stiff springs, out-filling queries, bob stride
for (const [name, Rig] of [['TP', ThirdPersonRig], ['FP', FirstPersonRig]]) {
  const cam = camera(), rig = new Rig(cam, null);
  rig.setTarget(target());
  rig.pitch = 0; rig.turn(0.1, 0); rig.update(DT);
  check(`${name} turn(+0.1, 0): yaw +0.1`, rig.yaw, 0.1 - 1e-12, 0.1 + 1e-12, 'rad');
  check(`${name} turn(+dyaw) turns left (forward.x < 0)`, camFwd(cam).x, -1, -0.05, '');
  rig.turn(0, 0.2); rig.update(DT);
  check(`${name} turn(0, +0.2): pitch +0.2, looks up`, camFwd(cam).y, Math.sin(0.2) - 1e-9, Math.sin(0.2) + 1e-9, '');
  rig.look(NaN, 5); rig.turn(0.3, NaN); rig.look(1, Infinity - Infinity);
  check(`${name} NaN look/turn ignored`, Math.abs(rig.yaw - 0.1) + Math.abs(rig.pitch - 0.2), 0, 0, 'rad');
  rig.pitch = 5; rig.update(DT);
  check(`${name} direct pitch write clamped by update`, rig.pitch, rig.maxPitch, rig.maxPitch, 'rad');
  rig.pitch = -5; rig.update(DT);
  check(`${name} direct pitch write clamped (low)`, rig.pitch, rig.minPitch, rig.minPitch, 'rad');
}
{
  const inv = new FirstPersonRig(camera(), null, { invertY: true });
  inv.look(0, -1e6);
  check('FP invertY: mouse up -> pitch min', inv.pitch, -1.5, -1.5, 'rad');
  inv.look(0, 100);
  check('FP invertY: mouse down looks up', inv.pitch, -1.5 + 100 * 0.0025 - 1e-12, -1.5 + 100 * 0.0025 + 1e-12, 'rad');
  // turn() resets the auto-recentre idle timer
  const r = new ThirdPersonRig(camera(), null, { autoRecentre: 0.5 });
  r.setTarget(target()); r.yaw = 2;
  run(r, 2, () => r.turn(1e-4, 0));
  check('TP turn() keeps auto-recentre idle', r.yaw, 2, 2.3, 'rad');
}
for (const damping of [60, 200, 1000]) {
  const cam = camera(), rig = new ThirdPersonRig(cam, null, { damping });
  const t = target();
  rig.setTarget(t);
  run(rig, 0.5);
  const offset = cam.position.clone().sub(rig.focus);
  t.position.set(5, 1, -3); t.updateMatrixWorld(true);
  const ideal = t.position.clone().add(new Vector3(0, rig.height, 0)).add(offset);
  let prev = cam.position.distanceTo(ideal), worst = -Infinity;
  for (const dt of [DT, 0.1, 1 / 30, 1 / 144, 0.1, DT, DT, DT, DT, DT]) {
    rig.update(dt);
    const dd = cam.position.distanceTo(ideal); worst = Math.max(worst, dd - prev); prev = dd;
  }
  check(`TP damping ${damping}: no overshoot (largest increase)`, worst, -Infinity, 1e-9, 'm');
  check(`TP damping ${damping}: converged, finite`, prev, 0, 1e-3, 'm');
}
{
  const wall = { minX: -5, maxX: 5, minY: 0, maxY: 6, minZ: 2, maxZ: 2.4, kind: 0 };
  const C = new CameraCollision({ physics: boxWorld(wall) });
  C.gather(new Vector3(0, 0, 0), 10);
  check('CameraCollision: out-filling physics.queryColliders', C.boxes ? C.boxes.length : 0, 1, 1, 'boxes');
  // bob restart: a walk after a stop starts on the same stride as a first walk
  const mk = () => { const st = { speed: 0, grounded: true }, r = new FirstPersonRig(camera(), null, { bob: 0.05 }); r.setTarget(target(), () => st); return { r, st }; };
  const A = mk(), B = mk();
  A.st.speed = 4; run(A.r, 1.37); A.st.speed = 0; run(A.r, 2);
  run(B.r, 2);
  A.st.speed = B.st.speed = 4;
  let diff = 0;
  run(A.r, 0.5, () => { B.r.update(DT); diff = Math.max(diff, A.r.camera.position.distanceTo(B.r.camera.position)); });
  check('FP bob: walk after a stop starts on the same stride', diff, 0, 1e-6, 'm');
}

// --- determinism: two rigs fed the same inputs ---------------------------------------
{
  const wall = { minX: -2, maxX: 2, minY: 0, maxY: 4, minZ: 3, maxZ: 3.5, kind: 0 };
  const make = () => {
    const t = target();
    const tp = new ThirdPersonRig(camera(), null, { physics: boxWorld(wall), autoRecentre: 0.7 });
    const fp = new FirstPersonRig(camera(), null, { bob: 0.04 });
    const st = { speed: 0, grounded: true };
    tp.setTarget(t, () => st); fp.setTarget(t, () => st);
    return { t, tp, fp, st };
  };
  const A = make(), B = make();
  let seed = 12345;
  const rnd = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 4294967296);
  let diff = 0;
  for (let i = 0; i < 1200; i++) {
    const dx = (rnd() - 0.5) * 40, dy = (rnd() - 0.5) * 30, mv = rnd() * 0.1, dt = DT * (0.5 + rnd());
    for (const W of [A, B]) {
      if (i % 200 < 120) W.tp.look(dx, dy), W.fp.look(dx, dy);
      W.t.position.x += mv; W.t.position.z -= mv * 0.5; W.t.rotation.y += 0.01; W.t.updateMatrixWorld(true);
      W.st.speed = mv / dt; W.st.grounded = i % 97 > 10;
      W.tp.update(dt); W.fp.update(dt);
    }
    for (const k of ['tp', 'fp']) {
      const a = A[k].camera, b = B[k].camera;
      diff = Math.max(diff, a.position.distanceTo(b.position), Math.abs(a.quaternion.x - b.quaternion.x), Math.abs(a.quaternion.y - b.quaternion.y),
        Math.abs(a.quaternion.z - b.quaternion.z), Math.abs(a.quaternion.w - b.quaternion.w));
    }
  }
  check('determinism: identical poses (max diff)', diff, 0, 0, '');
  check('determinism: finite pose', A.tp.camera.position.length() + A.fp.camera.position.length(), 0, 1e6, 'm');
}

// --- perf -----------------------------------------------------------------------------
{
  const boxes = [];
  for (let i = 0; i < 40; i++) boxes.push({ minX: i * 3 - 60, maxX: i * 3 - 58, minY: 0, maxY: 8, minZ: -6, maxZ: -4, kind: 0 });
  const rig = new ThirdPersonRig(camera(), null, { physics: { getGroundHeight: () => 0, queryColliders: (x, z, r, out) => { out.length = 0; for (const b of boxes) out.push(b); return out; } } });
  const t = target();
  rig.setTarget(t);
  const t0 = performance.now();
  for (let i = 0; i < 20000; i++) { rig.look(3, 0); t.position.x = Math.sin(i * 0.01) * 20; rig.update(DT); }
  check('perf: 20000 TP updates, 40 boxes (ms)', performance.now() - t0, 0, 1000, 'ms');
}

console.log(results.join('\n'));
console.log(failures ? `\n${failures} check(s) FAILED` : '\nAll camera rig checks passed.');
process.exit(failures ? 1 : 0);
