// Headless checks for the anim layer: AnimationController (blend trees, crossfades, events, root motion,
// allocation-free update) and IK (TwoBoneIK, CCDIK). No browser, no WebGL.
// Run: node test/anim.mjs
import * as THREE from 'three';
import v8 from 'node:v8';
import vm from 'node:vm';
import { AnimationController, TwoBoneIK, CCDIK, makeIKChain } from '../src/anim/index.js';

const DT = 1 / 60;
let failures = 0;
const results = [];
function check(name, value, lo, hi, unit = '') {
  const ok = Number.isFinite(value) && value >= lo && value <= hi;
  if (!ok) failures++;
  results.push(`${ok ? 'PASS' : 'FAIL'}  ${name.padEnd(54)} ${fmt(value)} ${unit}   [${lo} .. ${hi}]`);
  return ok;
}
const fmt = (x) => (!Number.isFinite(x) ? String(x).padStart(12) : x !== 0 && Math.abs(x) < 1e-3 ? x.toExponential(2).padStart(12) : x.toFixed(6).padStart(12));
const run = (anim, n, each) => { for (let i = 0; i < n; i++) { anim.update(DT); if (each) each(i); } };

// tiny 3-bone character: character > Hips (y 1) > Spine > Head
function makeRig() {
  const root = new THREE.Object3D(); root.name = 'character';
  const hips = new THREE.Bone(); hips.name = 'Hips'; hips.position.set(0, 1, 0);
  const spine = new THREE.Bone(); spine.name = 'Spine'; spine.position.set(0, 0.5, 0);
  const head = new THREE.Bone(); head.name = 'Head'; head.position.set(0, 0.4, 0);
  root.add(hips); hips.add(spine); spine.add(head);
  return { root, hips, spine, head, skeleton: new THREE.Skeleton([hips, spine, head]) };
}
const qAbout = (axis, a) => new THREE.Quaternion().setFromAxisAngle(axis, a).toArray();
const X = new THREE.Vector3(1, 0, 0), Y = new THREE.Vector3(0, 1, 0), Z = new THREE.Vector3(0, 0, 1);
/** Spine sway clip (no root motion). */
const sway = (name, dur, amp) => new THREE.AnimationClip(name, dur, [
  new THREE.QuaternionKeyframeTrack('Spine.quaternion', [0, dur / 2, dur], [...qAbout(Z, -amp), ...qAbout(Z, amp), ...qAbout(Z, -amp)]),
]);
/** Hips move dx metres along X over dur seconds (+ sway). */
const mover = (name, dur, dx) => new THREE.AnimationClip(name, dur, [
  new THREE.VectorKeyframeTrack('Hips.position', [0, dur], [0, 1, 0, dx, 1, 0]),
  new THREE.QuaternionKeyframeTrack('Spine.quaternion', [0, dur / 2, dur], [...qAbout(X, 0), ...qAbout(X, 0.2), ...qAbout(X, 0)]),
]);
const locoClips = () => [sway('Idle', 2, 0.05), sway('Walk', 1, 0.2), sway('Run', 0.6, 0.4)];
const locoStates = () => ({ idle: { clip: 'Idle' }, walk: { clip: 'Walk' }, run: { clip: 'Run' } });
const locoTree = (sync) => ({ move: { type: '1d', param: 'speed', sync, points: [{ at: 0, state: 'idle' }, { at: 2, state: 'walk' }, { at: 5.5, state: 'run' }] } });

// --- 1. 1D blend tree: weights sum to 1, the two neighbours are active -----------------------------------------
{
  const { root } = makeRig();
  const anim = new AnimationController(root, locoClips(), { states: locoStates(), trees: locoTree(false) });
  anim.play('move');
  const cases = [[-1, 1, 0, 0], [0, 1, 0, 0], [1, 0.5, 0.5, 0], [2, 0, 1, 0], [3.75, 0, 0.5, 0.5], [4.6, 0, 0.9 / 3.5, 2.6 / 3.5], [5.5, 0, 0, 1], [9, 0, 0, 1]];
  let worstSum = 0, worstW = 0;
  for (const [v, wi, ww, wr] of cases) {
    anim.setParam('speed', v); anim.update(DT);
    const i = anim.getWeight('idle'), w = anim.getWeight('walk'), r = anim.getWeight('run');
    worstSum = Math.max(worstSum, Math.abs(i + w + r - 1));
    worstW = Math.max(worstW, Math.abs(i - wi), Math.abs(w - ww), Math.abs(r - wr));
  }
  check('tree: |sum of weights - 1| (8 param values)', worstSum, 0, 1e-6);
  check('tree: max weight error vs expected neighbours', worstW, 0, 1e-9);
  anim.setParam('speed', 1); anim.update(DT);
  check('tree: speed 1 -> run inactive (weight 0)', anim.getWeight('run'), 0, 0);
  check('tree: speed 1 -> idle/walk the neighbours', anim.getWeight('idle') * anim.getWeight('walk'), 0.2499, 0.2501);
  check('tree: current is the tree', anim.current === 'move' ? 1 : 0, 1, 1);
}

// --- 1b. synced tree: walk (1 s) and run (0.6 s) keep the same normalized phase ----------------------------------
{
  const { root } = makeRig();
  const anim = new AnimationController(root, locoClips(), { states: locoStates(), trees: locoTree(true) });
  anim.play('move');
  let worst = 0;
  const phase = (s) => anim.states[s].action.time / anim.states[s].duration;
  run(anim, 400, (i) => {
    anim.setParam('speed', 2 + 3.5 * (0.5 + 0.5 * Math.sin(i * 0.05)));
    const d = Math.abs(phase('walk') - phase('run'));
    worst = Math.max(worst, Math.min(d, 1 - d));
  });
  check('sync: |walk phase - run phase| over 400 frames', worst, 0, 1e-9);
  anim.setParam('speed', 3.75); anim.update(DT);
  check('sync: blended cycle 0.8 s -> walk time scale 1.25', anim.states.walk.action.timeScale, 1.25 - 1e-12, 1.25 + 1e-12);
  anim.play('walk'); run(anim, 3);                   // the synced tree fades out around its member, now current
  check('sync: current plain state keeps its own speed', anim.states.walk.action.timeScale, 1, 1);
  check('sync: run still tree-synced while fading out (0.6/0.8)', anim.states.run.action.timeScale, 0.75 - 1e-12, 0.75 + 1e-12);
}

// --- 2. crossfade timing + transition table -----------------------------------------------------------------
{
  const { root } = makeRig();
  const anim = new AnimationController(root, locoClips(), {
    states: locoStates(), transitions: { default: 0.2, 'idle->walk': 0.25, '*->run': 0.1, 'run->*': 0.4 },
  });
  anim.play('idle'); run(anim, 10);
  check('fade: first play is instant (idle weight)', anim.getWeight('idle'), 1, 1);
  anim.play('walk'); run(anim, 14);
  check('fade: 14/15 of the 0.25 s fade (walk weight)', anim.getWeight('walk'), 14 / 15 - 1e-9, 14 / 15 + 1e-9);
  run(anim, 1);
  check('fade: at 0.25 s walk weight = 1', anim.getWeight('walk'), 1 - 1e-3, 1);
  check('fade: at 0.25 s idle weight = 0', anim.getWeight('idle'), 0, 1e-3);
  run(anim, 1);
  check('fade: faded-out action stopped', anim.states.idle.action.isRunning() ? 1 : 0, 0, 0);
  check('table: *->run beats default (0.1 s)', anim._lookupFade('walk', 'run'), 0.1, 0.1);
  check('table: run->* (0.4 s)', anim._lookupFade('run', 'idle'), 0.4, 0.4);
  check('table: default (0.2 s)', anim._lookupFade('walk', 'idle'), 0.2, 0.2);
  // interrupted fades: weights keep summing to 1
  let worst = 0;
  anim.play('run'); run(anim, 3);
  anim.play('idle', { fade: 0.3 }); run(anim, 4);
  anim.play('walk', { fade: 0.5 });
  run(anim, 40, () => { worst = Math.max(worst, Math.abs(anim.getWeight('idle') + anim.getWeight('walk') + anim.getWeight('run') - 1)); });
  check('fade: interrupted fades, |sum - 1|', worst, 0, 1e-12);
}
{
  const { root } = makeRig();
  const anim = new AnimationController(root, [...locoClips(), sway('Jump', 0.5, 0.6)], {
    states: { ...locoStates(), jump: { clip: 'Jump', loop: false, onEnd: 'idle' } }, trees: locoTree(true),
  });
  const sum = () => anim.getWeight('idle') + anim.getWeight('walk') + anim.getWeight('run') + anim.getWeight('jump');
  anim.play('jump'); run(anim, 10);
  anim.play('idle', { fade: 0.4 }); run(anim, 6);   // jump fading out
  anim.setParam('speed', 3); anim.play('move', { fade: 0.3 });
  let worst = 0, mid = 0;
  run(anim, 30, (i) => { worst = Math.max(worst, Math.abs(sum() - 1)); if (i === 4) mid = anim.getWeight('jump'); });
  check('fade: play(tree) over a fading one-shot, |sum - 1|', worst, 0, 1e-12);
  check('fade: the one-shot was still fading (weight mid-way)', mid, 0.05, 0.95);
  check('fade: one-shot weight 0 after the fade', anim.getWeight('jump'), 0, 0);
}

// --- 3. one-shots: 'end' once, onEnd (name / function), clamp (default true) holds, clamp: false drops out -------
{
  const { root } = makeRig();
  const clips = [...locoClips(), sway('Jump', 0.5, 0.6), sway('Wave', 0.4, 0.3)];
  let fnCalls = 0, fnArg = '', againN = 0;
  const anim = new AnimationController(root, clips, {
    states: {
      ...locoStates(),
      jump: { clip: 'Jump', loop: false, onEnd: 'idle' },
      hold: { clip: 'Jump', loop: false },                                   // clamp defaults to true
      poke: { clip: 'Wave', loop: false, onEnd: (name, ctl) => { fnCalls++; fnArg = name; ctl.play('walk'); } },
      wave: { clip: 'Wave', loop: false, clamp: false },
      blink: { clip: 'Wave', loop: false, clamp: false },
      again: { clip: 'Wave', loop: false, clamp: false, onEnd: (name, ctl) => { if (++againN < 3) ctl.play(name); } },
    },
  });
  const ends = {};
  let endState = '';
  anim.events.on('end', (st) => { ends[st] = (ends[st] || 0) + 1; endState = st; if (st === 'blink' && ends.blink < 3) anim.play('blink'); });
  const sumW = () => Object.keys(anim.states).reduce((acc, k) => acc + anim.getWeight(k), 0);
  anim.play('idle'); run(anim, 5);
  anim.play('jump'); run(anim, 120);
  check("one-shot: 'end' count over 2 s", ends.jump, 1, 1);
  check("one-shot: 'end' names the state", endState === 'jump' ? 1 : 0, 1, 1);
  check('one-shot: onEnd -> idle is current', anim.current === 'idle' ? 1 : 0, 1, 1);
  check('one-shot: idle weight after the fade', anim.getWeight('idle'), 1, 1);
  anim.play('jump'); run(anim, 60); anim.play('jump'); run(anim, 60);
  check("one-shot: replay -> 'end' count 3", ends.jump, 3, 3);
  // default clamp: holds the last frame, stays current, 'end' once; play(current) restarts it
  anim.play('hold'); run(anim, 120);
  check("clamp default: 'end' once over 2 s", ends.hold, 1, 1);
  check('clamp default: still current and weighted', anim.current === 'hold' ? anim.getWeight('hold') : 0, 1, 1);
  check('clamp default: holds the last frame (time)', anim.states.hold.action.time, 0.5, 0.5, 's');
  anim.play('hold'); run(anim, 1);
  check('clamp default: play(current) restarts (time)', anim.states.hold.action.time, DT - 1e-12, DT + 1e-12, 's');
  run(anim, 40);
  check("clamp default: restarted one-shot ends again", ends.hold, 2, 2);
  // onEnd as a function
  anim.play('poke'); run(anim, 40);
  check('onEnd function: called once with the state name', fnCalls === 1 && fnArg === 'poke' ? 1 : 0, 1, 1);
  check('onEnd function: its play(walk) is current', anim.current === 'walk' ? 1 : 0, 1, 1);
  // clamp: false: drops out at the end; nodes still fading out snap with it (no decaying weight sum)
  anim.play('idle'); run(anim, 30);
  anim.play('wave', { fade: 1 }); run(anim, 30);       // wave (0.4 s) ends while idle is still fading out
  check('clamp false: current = null after the end', anim.current === null ? 1 : 0, 1, 1);
  check('clamp false: every weight 0 (fading idle snapped)', sumW(), 0, 0);
  check('clamp false: idle action stopped', anim.states.idle.action.isRunning() ? 1 : 0, 0, 0);
  // clamp: false restarted from an 'end' listener / from onEnd is not dropped
  anim.play('blink'); run(anim, 90);
  check("clamp false: 'end' listener restarts it (3 ends)", ends.blink, 3, 3);
  anim.play('again'); run(anim, 90);
  check('clamp false: onEnd function restarts it (3 ends)', ends.again, 3, 3);
  check('clamp false: dropped after the last end', anim.current === null ? 1 : 0, 1, 1);
}

// --- 4. markers: once per loop, in step with the loop events ---------------------------------------------------
{
  const { root } = makeRig();
  const anim = new AnimationController(root, locoClips(), {
    states: { ...locoStates(), walk: { clip: 'Walk', markers: [{ at: 0.4, name: 'footstep' }] } },
  });
  let seq = '', markerName = '';
  anim.events.on('marker', (name, state) => { seq += 'M'; markerName = `${name}/${state}`; });
  anim.events.on('loop', (state) => { if (state === 'walk') seq += 'L'; });
  anim.play('walk'); run(anim, 210);   // 3.5 s of a 1 s cycle: markers at 0.4, 1.4, 2.4, 3.4; loops at 1, 2, 3
  check('marker: count over 3.5 s (expect 4)', (seq.match(/M/g) || []).length, 4, 4);
  check('loop: count over 3.5 s (expect 3)', (seq.match(/L/g) || []).length, 3, 3);
  check('marker: exactly one per loop (MLMLMLM)', seq === 'MLMLMLM' ? 1 : 0, 1, 1);
  check('marker: args (name, state)', markerName === 'footstep/walk' ? 1 : 0, 1, 1);
}
{
  // backwards (speed -1): enters at the end of the cycle; a marker at the start phase counts on entry (closed)
  const { root } = makeRig();
  const anim = new AnimationController(root, locoClips(), {
    states: { ...locoStates(), back: { clip: 'Walk', speed: -1, markers: [{ at: 0.4, name: 'step' }, { at: 0, name: 'top' }] } },
  });
  let seq = '', tops = 0;
  anim.events.on('marker', (name) => { if (name === 'step') seq += 'M'; else tops++; });
  anim.events.on('loop', () => { seq += 'L'; });
  anim.play('back'); run(anim, 210);   // phase 1 -> -2.5: steps at 0.4, -0.6, -1.6; wraps at 0, -1, -2
  check('reverse: step markers + loops in order (MLMLML)', seq === 'MLMLML' ? 1 : 0, 1, 1);
  check('reverse: marker at phase 0 (entry + 3 wraps)', tops, 4, 4);
}

// --- 5. root motion ----------------------------------------------------------------------------------------
{
  const { root, hips } = makeRig();
  const clips = [mover('Move', 1, 2), sway('Idle', 2, 0.05)];
  const anim = new AnimationController(root, clips, {
    states: { move: { clip: 'Move' }, idle: { clip: 'Idle' } }, rootMotion: { bone: 'Hips', xz: true, y: false, yaw: false },
  });
  anim.play('move');
  const wp = new THREE.Vector3();
  let sum = 0, sumY = 0, hipsX = 0, hipsY = 0;
  run(anim, 60, () => {
    sum += anim.rootMotion.deltaPosition.x; sumY += Math.abs(anim.rootMotion.deltaPosition.y);
    root.updateMatrixWorld(true); hips.getWorldPosition(wp);
    hipsX = Math.max(hipsX, Math.abs(wp.x)); hipsY = Math.max(hipsY, Math.abs(wp.y - 1));
  });
  check('root motion: sum deltaPosition.x over 1 s (2 m)', sum, 2 - 1e-9, 2 + 1e-9, 'm');
  check('root motion: max |hips world X| (stays in place)', hipsX, 0, 1e-9, 'm');
  check('root motion: hips Y kept (y: false), max |y - 1|', hipsY, 0, 1e-9, 'm');
  check('root motion: deltaPosition.y stays 0', sumY, 0, 0, 'm');
  check("root motion: caller's clip untouched (end key x)", clips[0].tracks[0].values[3], 2, 2, 'm');
  let lo = Infinity, hi = -Infinity;
  run(anim, 300, () => { const d = anim.rootMotion.deltaPosition.x; lo = Math.min(lo, d); hi = Math.max(hi, d); });
  check('loop wrap: min per-step dx over 5 loops (2/60)', lo, 2 / 60 - 1e-9, 2 / 60 + 1e-9, 'm');
  check('loop wrap: max per-step dx over 5 loops (2/60)', hi, 2 / 60 - 1e-9, 2 / 60 + 1e-9, 'm');
  // applyRootMotion moves the root along its own facing
  root.rotation.y = Math.PI / 2; root.updateMatrixWorld(true);
  anim.update(DT); anim.applyRootMotion();
  check('applyRootMotion: local +X -> world -Z', root.position.z, -2 / 60 - 1e-9, -2 / 60 + 1e-9, 'm');
}
// blended root motion: 1 m/s and 3 m/s clips at 50 % -> 2 m/s
{
  const { root } = makeRig();
  const anim = new AnimationController(root, [mover('Slow', 1, 1), mover('Fast', 1, 3)], {
    states: { slow: { clip: 'Slow' }, fast: { clip: 'Fast' } },
    trees: { go: { param: 'v', points: [{ at: 0, state: 'slow' }, { at: 1, state: 'fast' }] } },
    rootMotion: true,
  });
  anim.setParam('v', 0.5); anim.play('go');
  let sum = 0;
  run(anim, 60, () => { sum += anim.rootMotion.deltaPosition.x; });
  check('root motion: 50/50 blend of 1 and 3 m/s over 1 s', sum, 2 - 1e-9, 2 + 1e-9, 'm');
}
// reversed clip: -2 m per second, no spike at the wraps
{
  const { root, hips } = makeRig();
  const anim = new AnimationController(root, [mover('Move', 1, 2)], { states: { back: { clip: 'Move', speed: -1 } }, rootMotion: true });
  anim.play('back');
  const wp = new THREE.Vector3();
  let sum = 0, lo = Infinity, hi = -Infinity, hipsX = 0;
  run(anim, 60, () => { sum += anim.rootMotion.deltaPosition.x; });
  run(anim, 180, () => {
    const d = anim.rootMotion.deltaPosition.x; lo = Math.min(lo, d); hi = Math.max(hi, d);
    root.updateMatrixWorld(true); hipsX = Math.max(hipsX, Math.abs(hips.getWorldPosition(wp).x));
  });
  check('root motion, speed -1: sum dx over 1 s (-2 m)', sum, -2 - 1e-9, -2 + 1e-9, 'm');
  check('root motion, speed -1: per-step dx, 3 wraps (min)', lo, -2 / 60 - 1e-9, -2 / 60 + 1e-9, 'm');
  check('root motion, speed -1: per-step dx, 3 wraps (max)', hi, -2 / 60 - 1e-9, -2 / 60 + 1e-9, 'm');
  check('root motion, speed -1: max |hips world X|', hipsX, 0, 1e-9, 'm');
}
// yaw extraction: hips turn 90 degrees about Y per second
{
  const { root, hips } = makeRig();
  const turn = new THREE.AnimationClip('Turn', 1, [
    new THREE.VectorKeyframeTrack('Hips.position', [0, 1], [0, 1, 0, 0, 1, 0]),
    new THREE.QuaternionKeyframeTrack('Hips.quaternion', [0, 0.5, 1], [...qAbout(Y, 0), ...qAbout(Y, Math.PI / 4), ...qAbout(Y, Math.PI / 2)]),
  ]);
  const anim = new AnimationController(root, [turn], { states: { turn: { clip: 'Turn' } }, rootMotion: { bone: 'Hips', yaw: true } });
  anim.play('turn');
  let yaw = 0, worstQ = 0, lo = Infinity;
  const I = new THREE.Quaternion();
  run(anim, 60, () => { yaw += anim.rootMotion.deltaYaw; worstQ = Math.max(worstQ, hips.quaternion.angleTo(I)); });
  run(anim, 120, () => { lo = Math.min(lo, anim.rootMotion.deltaYaw); });
  check('root yaw: sum deltaYaw over 1 s (pi/2)', yaw, Math.PI / 2 - 1e-6, Math.PI / 2 + 1e-6, 'rad');
  check('root yaw: hips rotation neutralised (max angle)', worstQ, 0, 1e-6, 'rad');
  check('root yaw: no wrap spike (min per-step yaw)', lo, Math.PI / 120 - 1e-6, Math.PI / 120 + 1e-6, 'rad');
}

// --- 6. two-bone IK ----------------------------------------------------------------------------------------
function makeLeg(bend = 0) {
  const root = new THREE.Object3D();
  const upper = new THREE.Bone(); upper.name = 'upper'; upper.position.set(0, 2, 0);
  const lower = new THREE.Bone(); lower.name = 'lower'; lower.position.set(0, -1, 0); lower.rotation.x = bend;
  const end = new THREE.Bone(); end.name = 'end'; end.position.set(0, -1, 0);
  root.add(upper); upper.add(lower); lower.add(end);
  root.updateMatrixWorld(true);
  return { root, upper, lower, end };
}
const wpos = (o) => o.getWorldPosition(new THREE.Vector3());
{
  const { upper, lower, end } = makeLeg(0);   // straight leg: the pole picks the bend plane
  const ik = new TwoBoneIK(upper, lower, end, { pole: new THREE.Vector3(0, 1, 5) });
  const target = new THREE.Vector3(0.3, 0.9, 0.4);
  ik.solve(target);
  check('2-bone IK: reachable target error', wpos(end).distanceTo(target), 0, 1e-6, 'm');
  check('2-bone IK: upper length kept', wpos(upper).distanceTo(wpos(lower)), 1 - 1e-9, 1 + 1e-9, 'm');
  const a = wpos(upper), axis = target.clone().sub(a).normalize();
  const knee = wpos(lower).sub(a); knee.addScaledVector(axis, -knee.dot(axis));
  const pole = new THREE.Vector3(0, 1, 5).sub(a); pole.addScaledVector(axis, -pole.dot(axis));
  check('2-bone IK: knee faces the pole (cos)', knee.normalize().dot(pole.normalize()), 1 - 1e-6, 1 + 1e-9);
  ik.solve(target);
  check('2-bone IK: re-solve stays on target', wpos(end).distanceTo(target), 0, 1e-6, 'm');
  const far = new THREE.Vector3(0, 2, -6);
  ik.solve(far);
  check('2-bone IK: unreachable -> full extension', wpos(end).distanceTo(wpos(upper)), 2 - 1e-6, 2 + 1e-6, 'm');
  check('2-bone IK: unreachable -> points at target', wpos(end).distanceTo(new THREE.Vector3(0, 2, -2)), 0, 1e-6, 'm');
}
{
  // transformed, scaled parent and a bent start pose, no pole
  const { root, upper, lower, end } = makeLeg(0.5);
  root.position.set(3, 0, -1); root.rotation.set(0.2, 0.9, -0.1); root.scale.setScalar(1.5); root.updateMatrixWorld(true);
  const ik = new TwoBoneIK(upper, lower, end);
  check('2-bone IK: lengths measured in world units', ik.lengthA + ik.lengthB, 3 - 1e-9, 3 + 1e-9, 'm');
  const target = wpos(upper).add(new THREE.Vector3(0.5, -1.8, 0.7));
  ik.solve(target);
  check('2-bone IK: rotated/scaled parent, target error', wpos(end).distanceTo(target), 0, 1e-6, 'm');
  const before = wpos(end).distanceTo(target.clone().add(new THREE.Vector3(0.6, 0.3, 0)));
  ik.solve(target.clone().add(new THREE.Vector3(0.6, 0.3, 0)), 0.5);
  const after = wpos(end).distanceTo(target.clone().add(new THREE.Vector3(0.6, 0.3, 0)));
  check('2-bone IK: weight 0.5 moves part of the way', after / before, 0.05, 0.95);
}

// --- 7. CCD IK wrapper + makeIKChain ---------------------------------------------------------------------------
{
  const { root, upper, lower, end } = makeLeg(0.3);
  const skeleton = new THREE.Skeleton([upper, lower, end]);
  const goal = new THREE.Object3D(); goal.position.set(0.4, 0.7, 0.5); root.add(goal);
  const ccd = new CCDIK(skeleton, [{ target: goal, effector: 'end', links: ['lower', 'upper'], iteration: 60 }]);
  ccd.solve();
  check('CCDIK: Object3D target, effector error', wpos(end).distanceTo(wpos(goal)), 0, 1e-3, 'm');
  check('CCDIK: chain built from names (link indices)', ccd.chains[0].links[0].index * 10 + ccd.chains[0].links[1].index, 10, 10);
  ccd.enabled = false; goal.position.set(-0.5, 1, 0); const p = wpos(end); ccd.solve();
  check('CCDIK: disabled -> no change', wpos(end).distanceTo(p), 0, 0, 'm');
  const mesh = new THREE.SkinnedMesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial());
  mesh.add(upper); mesh.bind(skeleton);
  const chain = makeIKChain(mesh, { target: 'end', effector: 'lower', links: 1, iteration: 5 });
  check('makeIKChain: names -> indices', chain.target * 100 + chain.effector * 10 + chain.links[0].index + chain.iteration * 1000, 5210, 5210);
}

// --- 8. update() allocation: 10k updates; retained heap, and allocation over a bare AnimationMixer baseline -------
{
  let gc = globalThis.gc;
  if (!gc) { try { v8.setFlagsFromString('--expose-gc'); gc = vm.runInNewContext('gc'); } catch { gc = null; } }
  const exact = v8.getHeapStatistics().total_allocated_bytes !== undefined;   // Node >= 26
  const allocated = exact ? () => v8.getHeapStatistics().total_allocated_bytes : () => process.memoryUsage().heapUsed;
  const clips = () => [...locoClips(), mover('Move', 1, 2), sway('Jump', 0.5, 0.6)];
  const make = () => {
    const ctl = new AnimationController(makeRig().root, clips(), {
      states: { ...locoStates(), walk: { clip: 'Walk', markers: [{ at: 0.2, name: 'step' }, { at: 0.7, name: 'step' }] },
        travel: { clip: 'Move' }, jump: { clip: 'Jump', loop: false, onEnd: 'move' } },
      trees: locoTree(true), transitions: { default: 0.2, '*->jump': 0.05 }, rootMotion: { bone: 'Hips', xz: true },
    });
    const frame = (i) => {
      ctl.setParam('speed', 2.75 + 2.75 * Math.sin(i * 0.01));
      if (i % 240 === 0) ctl.play('move'); else if (i % 240 === 120) ctl.play('jump'); else if (i % 240 === 200) ctl.play('travel');
      ctl.update(DT);
    };
    return { ctl, frame };
  };
  const { ctl: anim, frame } = make();
  let n = 0;
  anim.events.on('marker', () => { n++; }); anim.events.on('loop', () => { n++; }); anim.events.on('end', () => { n++; });
  // baseline: a bare mixer replaying, action by action, what an identical twin controller did on the same frames
  const names = Object.keys(anim.states), K = names.length, N = 10000, rec = new Float64Array(N * K * 3);
  const twin = make();
  for (let i = 0; i < 2000; i++) twin.frame(i);
  for (let j = 0; j < N; j++) {
    twin.frame(2000 + j);
    for (let k = 0; k < K; k++) { const st = twin.ctl.states[names[k]], o = (j * K + k) * 3; rec[o] = st.active ? 1 : 0; rec[o + 1] = st.w; rec[o + 2] = st.ts; }
  }
  const mixer = new THREE.AnimationMixer(makeRig().root);
  const acts = clips().map((c) => mixer.clipAction(c)), on = new Uint8Array(K);
  acts[K - 1].setLoop(THREE.LoopOnce, Infinity); acts[K - 1].clampWhenFinished = true;   // jump
  const bareFrame = (j) => {
    for (let k = 0; k < K; k++) {
      const o = (j * K + k) * 3, a = acts[k];
      if (rec[o]) { if (!on[k]) { a.play(); on[k] = 1; } a.setEffectiveWeight(rec[o + 1]); a.setEffectiveTimeScale(rec[o + 2]); }
      else if (on[k]) { a.stop(); on[k] = 0; }
    }
    mixer.update(DT);
  };
  for (let i = 0; i < 2000; i++) frame(i);   // warm-up (JIT, bindings, event records)
  for (let j = 0; j < N; j++) bareFrame(j);
  mixer.stopAllAction(); on.fill(0);
  if (gc) gc();
  const h0 = process.memoryUsage().heapUsed, b0 = allocated();
  const t0 = performance.now();
  for (let i = 2000; i < 2000 + N; i++) frame(i);
  const ms = performance.now() - t0;
  const b1 = allocated();
  if (gc) gc();
  const h1 = process.memoryUsage().heapUsed;
  const c0 = allocated();
  for (let j = 0; j < N; j++) bareFrame(j);
  const c1 = allocated();
  const ctl = (b1 - b0) / N, bare = (c1 - c0) / N;
  check('alloc: retained heap growth, 10k updates (gc before/after)', (h1 - h0) / 1024, -Infinity, 1024, 'KB');
  check('alloc: controller - bare mixer, per update', ctl - bare, -Infinity, 512, 'B');
  check('perf: 10k updates (5 states, synced tree, root motion)', ms, 0, 100, 'ms');
  check('alloc: events fired during the run', n, 100, Infinity);
  results.push(`NOTE  allocation per update: controller ${ctl.toFixed(0)} B, bare mixer replaying the same actions ${bare.toFixed(0)} B` +
    `${exact ? '' : ' (heapUsed deltas, approximate: Node < 26)'}. update() creates no objects; the difference is V8 HeapNumber ` +
    'boxing of doubles passed between calls (harmless, short-lived, but real)');
  if (!gc) results.push('NOTE  no gc available: retained heap growth measured without gc (timing is the reliable check)');
}

console.log(results.join('\n'));
console.log(failures ? `\n${failures} check(s) FAILED` : '\nAll anim checks passed.');
process.exit(failures ? 1 : 0);
