// Canyon Run — an off-road time trial built only on the engine (engine/src/**) and three.js; nothing from src/.
// Wiring: Engine (120 Hz fixed step, render interpolation, adaptive resolution, F3 profiler) + PhysicsWorld
// (terrain heightfield + rock / gate box colliders) + VehicleController (BUGGY_SPEC) + Atmosphere (desert sun
// model, keys 1-8) + PostFX (?nofx=1 to disable) + ChaseRig / OrbitRig / FreeFlyRig + VehicleInput (custom
// action map with always-on debug keys, gamepad, touch) + AudioBus synth voices + the UI kit HUD.
// Opt-in (0.8.0; the default page renders and works exactly as without them):
//   ?lamps=1          trackside lamp field (lamps.js): ~120 poles with one ClusteredLights point light each (the
//                     engine's clustered forward lights), lit by the darkness (try ?lamps=1&t=night)
//   ?gpuparticles=1   dust devils on the GpuParticleSystem (effects.js; CPU fallback with a warning). Chosen at load:
//                     not switchable at runtime (no harness setting)
//   ?devilscollide=1  with ?gpuparticles=1: the devils collide with PostFX's scene depth
//   ?ssr=1, ?volumetric=1, ?motionblur=<s>, ?dof=1[&focus=&fstop=], ?lut=warm|bleach|identity, ?aonormals=gbuffer   post-stack
//                     passes of 0.8.0 (samples/shared/fx-flags.js; ssr / motionblur / aonormals need ?gbuffer=1)
// Other knobs: ?nofx=1, ?gbuffer=1, ?aa=, ?cascades=, ?shadows=, ?msaamax=, ?t=, ?laps=, ?fps=, ?profile=, ?bench=1&size=WxH.
import * as THREE from 'three';
import {
  Engine, PhysicsWorld, Atmosphere, DEFAULT_LOOKS, PostFX, DEFAULT_GRADE, ChaseRig, OrbitRig, FreeFlyRig,
  VehicleInput, PAD_BUTTONS as B, PAD_AXES as AX, inputRamp, LoadingScreen, makeTextureArray, createTimeUniforms,
  updateTimeUniforms, ChunkCuller, createBenchHarness, readBenchParams, ClusteredLights, installClusteredChunks, syncClustered,
  setMaterialClustered,
} from '@vibe/engine';
import { Track, TRACK_HALF_WIDTH } from './track.js';
import { Terrain, createTerrainMaterial, WORLD_HALF } from './terrain.js';
import { Course } from './course.js';
import { Props } from './props.js';
import { Buggy, SURFACES } from './buggy.js';
import { DustDevils } from './effects.js';
import { LampField } from './lamps.js';
import { Race, formatTime } from './race.js';
import { HUD, ACCENT, applyTheme } from './hud.js';
import { BuggyAudio } from './audio.js';
import { readFxFlags, aoFlags, loadLut, fxSettings } from '../shared/fx-flags.js';

const params = new URLSearchParams(location.search);
const flag = (k) => params.has(k) && params.get(k) !== '0';
const NOFX = flag('nofx');
const LAMPS = flag('lamps'), GPU_PARTICLES = flag('gpuparticles'), DEVILS_COLLIDE = GPU_PARTICLES && flag('devilscollide');
const LAPS = Math.max(1, Math.min(5, +(params.get('laps') || 1) | 0));
// drawing-buffer pixels above which the post stack uses FXAA instead of 4x MSAA (Retina density hides the difference;
// the 4x resolve costs 2.7 ms at 3024x1964): ?msaamax=N, ?msaamax= (empty) = always MSAA by day, ?msaamax=0 = always FXAA
const MSAA_MAX = params.has('msaamax') ? (params.get('msaamax') === '' ? Infinity : +params.get('msaamax') || 0) : 4e6;

// ------------------------------------------------------------------ desert sky: high sun, long days, clear dry air
const DESERT_SUN = { rise: 6.05, set: 19.45, maxElevation: 74, nightDepth: 35, nightRate: 13, azimuthStart: 70, azimuthSpan: 220 };
export const DESERT_PRESETS = { dawn: 6.35, morning: 8.8, noon: 12.75, afternoon: 15.6, golden: 18.4, sunset: 19.2, dusk: 19.62, night: 22.5 };
const PRESET_KEYS = Object.keys(DESERT_PRESETS);
const DESERT_LOOKS = DEFAULT_LOOKS.map((k) => ({ ...k, cover: k.cover * 0.42, cirrus: k.cirrus * 1.15, haze: k.haze * 0.9, mist: k.mist * 0.25, cop: k.cop * 0.9 }));
const DESERT_GRADE = DEFAULT_GRADE.map((g) => ({ ...g, sat: g.sat + 0.06, con: g.con + 0.02, wb: [g.wb[0] * 1.02, g.wb[1], g.wb[2] * 0.95] }));

// ------------------------------------------------------------------ input: buggy action map on VehicleInput
const PEDAL = inputRamp(0.12, 0.06), STEER = inputRamp(0.17, 0.08);
const BUGGY_ACTIONS = {
  throttle: { type: 'axis', range: [0, 1], smoothing: PEDAL, bindings: ['KeyW', 'ArrowUp', { padButton: B.RT }, { virtual: 'gas' }] },
  brake: { type: 'axis', range: [0, 1], smoothing: PEDAL, bindings: ['KeyS', 'ArrowDown', { padButton: B.LT }, { virtual: 'brake' }] },
  steer: { type: 'axis', range: [-1, 1], smoothing: STEER, bindings: [
    { key: 'KeyD', value: 1 }, { key: 'ArrowRight', value: 1 }, { key: 'KeyA', value: -1 }, { key: 'ArrowLeft', value: -1 },
    { padAxis: AX.LX }, { virtual: 'stick.x' }] },
  handbrake: { type: 'button', bindings: ['Space', 'KeyX', { padButton: B.B }, { virtual: 'hb' }] },
  lookBack: { type: 'button', bindings: ['KeyB', { padButton: B.DOWN }] },
  toggleCamera: { type: 'button', bindings: ['KeyC', { padButton: B.Y }, { virtual: 'cam' }] },
  reset: { type: 'button', bindings: ['KeyR', { padButton: B.X }, { virtual: 'reset' }] },
  toggleHelp: { type: 'button', bindings: ['KeyH', { padButton: B.BACK }, { virtual: 'help' }] },
  start: { type: 'button', bindings: ['Enter', 'NumpadEnter', { padButton: B.START }, { padButton: B.A }, { virtual: 'start' }] },
  menu: { type: 'button', bindings: ['Escape'] },
  // always-on: keep working while the free-fly camera disables the map (input.map.enabled = false)
  freeCam: { type: 'button', alwaysOn: true, bindings: ['KeyF'] },
  mute: { type: 'button', alwaysOn: true, bindings: ['KeyM'] },
};
const TIME_ACTIONS = PRESET_KEYS.map((name, i) => { const a = `time.${name}`; BUGGY_ACTIONS[a] = { type: 'button', alwaysOn: true, bindings: [`Digit${i + 1}`] }; return a; });
const BUGGY_TOUCH = [
  { type: 'stick', id: 'stick', side: 'left', x: 120, y: 150, radius: 66 },
  { type: 'button', id: 'gas', label: 'GAS', side: 'right', x: 76, y: 150, size: 82 },
  { type: 'button', id: 'brake', label: 'BRK', side: 'right', x: 172, y: 118, size: 62 },
  { type: 'button', id: 'hb', label: 'HB', side: 'right', x: 172, y: 210, size: 52 },
  { type: 'button', id: 'start', label: 'GO', side: 'left', x: 36, y: 210, size: 46, top: true, fontSize: 11 },
  { type: 'button', id: 'cam', label: 'CAM', side: 'left', x: 36, y: 262, size: 46, top: true, fontSize: 10 },
  { type: 'button', id: 'reset', label: 'RST', side: 'left', x: 36, y: 314, size: 46, top: true, fontSize: 10 },
  { type: 'button', id: 'help', label: '?', side: 'left', x: 36, y: 366, size: 40, top: true },
];

const nextFrame = () => new Promise((r) => setTimeout(r, 0)); // yield to the loading screen (not rAF: it stalls in hidden tabs)

async function boot() {
  const t0 = performance.now();
  // ?lamps=1: the clustered-light chunk patch goes in before any material compiles (inert without the define)
  let clustered = null;
  if (LAMPS) {
    if (!installClusteredChunks()) console.warn('[canyon-run] clustered lights unavailable (three light chunks changed): the lamps glow but light nothing');
    clustered = new ClusteredLights({ maxLights: 1024, zFar: 420 });
  }
  applyTheme();
  const loading = new LoadingScreen({ title: 'CANYON RUN', label: 'Surveying the canyon…', accent: ACCENT, background: '#140c07' });
  const physics = new PhysicsWorld({ cellSize: 32 });
  const engine = new Engine({
    container: document.getElementById('app'), fixedHz: 120, maxSubsteps: 4, renderer: { antialias: NOFX }, far: 7000, fov: 60,
    physics, profile: params.get('profile') ? +params.get('profile') : false,
    targetFps: +params.get('fps') || 0, // ?fps=60: cap the tick rate (every 2nd frame on a 120 Hz screen: battery, heat)
  });
  const { scene, camera, renderer } = engine;
  const U = createTimeUniforms();
  const t = params.get('t');
  const atmo = new Atmosphere({
    scene, renderer, camera, sun: DESERT_SUN, looks: DESERT_LOOKS, timePresets: DESERT_PRESETS,
    timeOfDay: t == null ? 'noon' : Number.isFinite(+t) ? +t : t,
    // ?shadows=pcf: three's 17-tap kernel; 'fast' (5 hardware-compare taps) measured -2 ms at 3024x1964, pixel-equivalent
    shadows: { cascades: +params.get('cascades') || 2, near: 42, far: [250, 420, 650], kernel: params.get('shadows') || 'fast' }, maxCasterHeight: 170, cloudWind: [9, 3],
  });

  // ---------------------------------------------------------------- world build (one loading system)
  const W = {};
  engine.addSystem({
    async load(progress) {
      const T = W.timings = {}; let tp = performance.now();
      const mark = (k) => { const n = performance.now(); T[k] = Math.round(n - tp); tp = n; };
      progress(0.02, 'Surveying the canyon…');
      const track = W.track = new Track();
      // photo texture sets (Poly Haven CC0, public/textures) decode in parallel with the terrain build
      const texJob = makeTextureArray(renderer, [
        { role: 'dirt', id: 'brown_mud_dry', size: 1.3, proc: 'plain' },
        { role: 'rock', id: 'rock_boulder_dry', size: 1.8, proc: 'stucco' },
        { role: 'terrain_aerial', id: 'rocky_terrain_02', size: 90, proc: 'plain' },
      ], { size: 1024 });
      const terrain = W.terrain = new Terrain(track);
      await terrain.build((p) => progress(0.05 + 0.3 * p, 'Carving the canyon…'));
      mark('terrain');
      physics.setHeightfield((x, z) => terrain.heightAt(x, z));
      const H = WORLD_HALF - 22; // invisible walls just inside the rim
      physics.addStaticBox(-WORLD_HALF - 60, -200, -H, 800, -WORLD_HALF - 60, WORLD_HALF + 60);
      physics.addStaticBox(H, -200, WORLD_HALF + 60, 800, -WORLD_HALF - 60, WORLD_HALF + 60);
      physics.addStaticBox(-WORLD_HALF - 60, -200, WORLD_HALF + 60, 800, -WORLD_HALF - 60, -H);
      physics.addStaticBox(-WORLD_HALF - 60, -200, WORLD_HALF + 60, 800, H, WORLD_HALF + 60);
      progress(0.4, 'Loading rock and dirt…');
      const arrays = W.arrays = await texJob;
      mark('textures');
      const tmat = createTerrainMaterial(arrays);
      scene.add(terrain.buildMeshes(tmat), terrain.buildFarRing(tmat));
      mark('meshes');
      progress(0.55, 'Marking the course…');
      await nextFrame();
      W.course = new Course({ scene, renderer, terrain, track, physics, arrays });
      mark('course');
      progress(0.62, 'Scattering rocks and cacti…');
      await nextFrame();
      W.props = new Props({ scene, renderer, terrain, track, physics, arrays, U });
      mark('props');
      if (LAMPS) { // after the props: the poles keep clear of their colliders
        W.lamps = new LampField({ scene, terrain, track, physics, clustered });
        W.lamps.update(atmo.darkness);
        mark('lamps');
      }
      progress(0.88, 'Fuelling the buggy…');
      await nextFrame();
      W.buggy = new Buggy({ scene, physics, terrain, atmosphere: atmo });
      W.devils = new DustDevils({ scene, renderer, terrain, track, atmosphere: atmo, gpu: GPU_PARTICLES, collide: DEVILS_COLLIDE });
      W.culler = new ChunkCuller({ maxDistance: 2400, budget: 32 });
      for (const m of terrain.chunks) W.culler.add(m);
      W.culler.add(W.course.gantry, { maxDistance: 1600 });
      mark('buggy');
      progress(1, 'Ready');
    },
  }, 'world-load');
  await engine.load((p, label) => loading.setProgress(p, label));
  const { track, terrain, course, props, buggy, devils, culler } = W;
  const lamps = W.lamps || null;

  // ---------------------------------------------------------------- post-processing
  let postfx = null;
  if (!NOFX) {
    try {
      const { width, height } = engine.size;
      // 0.7.0 knobs: ?gbuffer=1 (MRT scene pass: normals + velocity, TAA), ?aa=auto|msaa|fxaa|taa, ?cascades=2|3|4
      // 0.8.0 knobs (samples/shared/fx-flags.js): ?ssr=1, ?volumetric=1, ?motionblur=<s>, ?dof=1, ?lut=<name>, ?aonormals=gbuffer
      const fx = readFxFlags(params), lut = await loadLut(params);
      postfx = new PostFX(renderer, scene, camera, { atmosphere: atmo, width, height, pixelRatio: engine.pixelRatio, grade: DESERT_GRADE, ao: { radius: 1.6, thickness: 1.4, samples: 8, ...aoFlags(params) }, msaaMaxPixels: MSAA_MAX, gbuffer: params.get('gbuffer') === '1', aa: params.get('aa') || 'auto', ...fx, lut });
      buggy.object3d.traverse((o) => { if (o.isMesh) postfx.track(o); }); // the car moves: per-object velocity (replayed whenever the G-buffer is built)
    } catch (e) { console.warn('[canyon-run] post-processing unavailable', e); atmo.setFogMode('scene'); }
  }
  engine.renderFn = (sc, cam, dt) => {
    if (postfx) postfx.render(dt);
    else { renderer.render(sc, cam); atmo.renderFlare(renderer, cam); }
  };
  // clustered tiles map gl_FragCoord of the scene target: the drawing-buffer size, floor(css * pixelRatio) like PostFX
  const clusterSize = (w, h, pr) => { if (clustered) clustered.setSize(Math.floor(w * pr), Math.floor(h * pr)); };
  engine.onResize = (w, h, pr) => { if (postfx) postfx.setSize(w, h, pr); clusterSize(w, h, pr); };
  if (clustered) {
    const sz = engine.size;
    clusterSize(sz.width, sz.height, engine.pixelRatio);
    syncClustered(scene, true); // ENGINE_CLUSTERED on every lit material now in the scene, and on anything added later
    // the lenses glow on their own: unclustered, or their lamp 0.3 m below them blows out every lens underside (~1300
    // lx) into a bloom halo. Material-level, after the sync (the 'childadded' listener never re-applies it)
    if (lamps) setMaterialClustered(lamps.bulbMaterial, false);
  }
  // ?devilscollide=1: PostFX itself as the depth source, set once (its sceneDepth is read on every update, so a G-buffer
  // rebuild is followed). Calling setDepth per frame would clear the system's fresh-snapshot flag, so every update()
  // would re-snapshot the camera's current matrices instead of those of the render that produced the depth
  if (devils.gpu && devils.ps.depthCollision) {
    if (postfx) devils.ps.setDepth(postfx, camera);
    else console.warn('[canyon-run] ?devilscollide needs the post stack (scene depth): no collision with ?nofx');
  }

  // ---------------------------------------------------------------- game objects
  const race = new Race({ track, laps: LAPS });
  // per-wheel ground material: the dirt ribbon vs the sand around it (the race's nearest-point tracker is the hint)
  const near = { i: 0, s: 0, lateral: 0, dist: 0, along: 0 };
  physics.setSurfaceProvider((x, z) => {
    track.nearest(x, z, race.near.i, near, 12);
    return near.dist > TRACK_HALF_WIDTH + 0.7 ? SURFACES.sand : SURFACES.track;
  });
  const hud = new HUD({ track, race });
  const audio = new BuggyAudio();
  audio.bus.unlockOnGesture();
  const input = new VehicleInput(renderer.domElement, { actions: BUGGY_ACTIONS, touchLayout: BUGGY_TOUCH });
  hud.setTouch(!!(input.touch && input.touch.visible)); // phones: the speed panel moves out of the thumb area

  // camera rigs (all start inactive): orbit on the menu / finish screens, chase while racing, free-fly toggle (F).
  // Chase: no Roadster tree-canopy rule, only kind-3 posts are "thin", and never the flight framing over jumps.
  const chase = new ChaseRig(camera, renderer.domElement, { active: false, minY: 0.7, physics, canopyFromCollider: null, thinKinds: [3], flightFraming: false });
  Object.assign(chase, { distMin: 6.3, distMax: 8.8, heightMin: 2.15, heightMax: 2.75, fovMin: 58, fovMax: 75, speedForMax: 50, orbitRadius: 10 });
  chase.offsets.hood.set(0, 1.28, -1.1);
  chase.offsets.cockpit.set(-0.28, 1.45, 0.22);
  chase.setTarget(buggy.object3d, () => buggy.state);
  const orbit = new OrbitRig(camera, renderer.domElement, { active: false, target: buggy.object3d, distance: 12.5, elevation: 0.27, azimuth: 2.5, autoRotate: 0.16, minDistance: 4, maxDistance: 80, minElevation: 0.03, physics });
  // free-fly keys go through the action map (always-on actions), so they work while the map is disabled
  const free = new FreeFlyRig(camera, renderer.domElement, { active: false, input: input.map, speed: 30, collide: true, minY: 0.6, physics });
  const rigs = { chase, orbit, free };
  let rig = null, rigBeforeFree = 'orbit';
  const setRig = (name) => {
    for (const k in rigs) if (k !== name) rigs[k].deactivate();
    rig = rigs[name]; rig.activate(); rig.snap();
    input.map.enabled = name !== 'free'; // the buggy's controls go quiet; always-on actions keep working
  };

  const pose = { x: 0, z: 0, heading: 0 }, spawn = new THREE.Vector3();
  const placeBuggy = (p) => { spawn.set(p.x, terrain.heightAt(p.x, p.z), p.z); buggy.reset(spawn, p.heading); };
  const toGrid = () => {
    const i = (track.n - 7) % track.n;
    placeBuggy({ x: track.x[i], z: track.z[i], heading: Math.atan2(-track.tx[i], -track.tz[i]) });
  };

  const startRun = () => {
    toGrid();
    race.startCountdown();
    course.setCountdown(0); course.setGateStates(1); hud.markGates();
    if (hud.help.visible) hud.toggleHelp();
    hud.showMenu(false); hud.showFinish(null); hud.showSplit(NaN, '');
    if (rig !== free) setRig('chase'); else rigBeforeFree = 'chase';
  };
  const toMenu = () => {
    race.toMenu(); toGrid();
    course.setCountdown(0); course.setGateStates(-1); hud.markGates(); hud.setOffTrack(false);
    hud.showFinish(null); hud.showMenu(true); hud.showSplit(NaN, '');
    if (rig !== free) setRig('orbit'); else rigBeforeFree = 'orbit';
  };
  const resetToCheckpoint = () => {
    placeBuggy(race.respawnPose(pose));
    race.noteReset(); if (rig === chase) chase.snap();
    hud.flash.show('RESET');
  };

  race.on = {
    countdown: (n) => { course.setCountdown(n); audio.beep(false); hud.countdown(String(6 - n)); },
    go: () => { course.setCountdown('go'); audio.beep(true); hud.countdown('GO'); },
    checkpoint: (c) => {
      course.setGateStates(c.next); hud.markGates();
      audio.chime(!(c.delta > 0));
      hud.flash.show(`CP ${c.number}   ${formatTime(c.time)}`);
      hud.showSplit(c.delta, `CP ${c.number}`);
    },
    lap: (n) => hud.flash.show(`LAP ${n + 1} / ${race.laps}`),
    finish: (res) => {
      course.setGateStates(0, true); course.setCountdown('go'); hud.markGates(); hud.setOffTrack(false);
      audio.finishFanfare(); hud.showSplit(res.splits.length && res.bestSplits ? res.time - res.prevBest : NaN, 'FINISH');
      hud.flash.show(res.isBest ? 'NEW BEST' : 'FINISH');
      setTimeout(() => { if (race.state === 'finished') { hud.showFinish(res); if (rig === chase) setRig('orbit'); } }, 1400);
    },
    offTrack: (off) => hud.setOffTrack(off && race.state === 'racing'),
    missed: () => hud.flash.show('MISSED CHECKPOINT · R TO RESET'),
  };

  // ---------------------------------------------------------------- systems (phase order = registration order)
  engine.addSystem({
    earlyUpdate(dt) {
      input.update(dt);
      const m = input.map, st = race.state;
      // always-on keys (also during free-fly): time of day, free camera, mute
      for (let k = 0; k < TIME_ACTIONS.length; k++) {
        if (m.justPressed(TIME_ACTIONS[k])) { atmo.setTimeOfDay(PRESET_KEYS[k]); hud.flash.show(PRESET_KEYS[k]); }
      }
      if (m.justPressed('freeCam')) {
        if (rig === free) setRig(rigBeforeFree);
        else { rigBeforeFree = rig === chase ? 'chase' : 'orbit'; setRig('free'); hud.flash.show('FREE CAMERA'); }
      }
      if (m.justPressed('mute')) hud.flash.show(audio.toggleMute() ? 'MUTED' : 'SOUND ON');
      if (input.toggleHelp) hud.toggleHelp();
      if (st === 'menu' && m.justPressed('start')) startRun();
      else if (st === 'finished' && m.justPressed('start')) startRun();
      else if ((st === 'finished' || st === 'racing' || st === 'countdown') && m.justPressed('menu')) toMenu();
      else if (st === 'racing' && input.reset) resetToCheckpoint();
      if (input.toggleCamera && rig === chase) hud.flash.show(`CAMERA · ${chase.cycleMode().toUpperCase()}`);
      chase.setLookBack(input.lookBack);
      buggy.setInput(input, race.state === 'racing' && rig !== free);
      buggy.vc.hold = st === 'menu' || st === 'countdown'; // frozen on the grid (still rests on the ground)
    },
  }, 'input');
  engine.addSystem({
    fixedUpdate: (dt) => buggy.fixedUpdate(dt),
    update: (dt, alpha) => buggy.update(dt, alpha, atmo.darkness),
  }, 'vehicle');
  engine.addSystem({ update: (dt) => race.update(dt, buggy.object3d.position) }, 'race');
  engine.addSystem({ update: (dt) => rig.update(dt) }, 'camera');
  engine.addSystem({
    lateUpdate(dt) {
      camera.updateMatrixWorld();
      const cp = camera.position;
      updateTimeUniforms(U, dt, cp);
      atmo.update(cp, dt);
      props.update(cp, dt);
      culler.update(camera);
      devils.update(dt, buggy.object3d.position.x, buggy.object3d.position.z, race.near.i, camera);
      if (lamps) { lamps.update(atmo.darkness); clustered.update(camera); } // rows only when the darkness moved; clusters every frame
    },
  }, 'world');
  engine.addSystem({ lateUpdate: (dt) => hud.update(dt, buggy) }, 'hud');
  const aState = { rpm: 900, throttle: 0, speed: 0, slip: 0, grounded: true, sand: false };
  engine.addSystem({
    lateUpdate(dt) {
      const vc = buggy.vc;
      aState.rpm = buggy.rpm; aState.throttle = buggy.controls.throttle; aState.speed = vc.speed; aState.slip = vc.wheelSlip;
      aState.grounded = vc.grounded; aState.sand = buggy.surface === SURFACES.sand;
      audio.update(dt, aState);
      if (buggy.lastLanding > 0) { if (buggy.lastLanding > 3.5) audio.thump(buggy.lastLanding / 14); buggy.lastLanding = 0; }
    },
  }, 'audio');

  toMenu();
  // compile every program now (off the first frames), then render one frame so the first rAF is cheap
  let tc = performance.now();
  renderer.compile(scene, camera);
  W.timings.compile = Math.round(performance.now() - tc);
  const loadMs = Math.round(performance.now() - t0);
  window.__canyon = { timings: W.timings, THREE, engine, physics, atmo, postfx, U, track, terrain, course, props, buggy, devils, race, hud, audio, input, rigs, culler, startRun, toMenu, resetToCheckpoint, loadMs, lamps, clustered };
  console.info(`[canyon-run] ready in ${loadMs} ms · track ${(track.length / 1000).toFixed(2)} km, ${track.checkpoints.length} gates, terrain ${terrain.ms.toFixed(0)} ms, props`, props.stats, `textures ${W.arrays.loaded}/${W.arrays.wanted}`, W.timings);
  if (lamps) console.info(`[canyon-run] lamps: ${lamps.stats.count} of ${lamps.stats.slots} slots, ${lamps.stats.colliders} colliders, clustered (max ${clustered.maxLights}, zFar ${clustered.zFar} m)`, lamps.stats.skipped);
  if (GPU_PARTICLES) console.info(`[canyon-run] dust devils on the ${devils.gpu ? 'GPU' : 'CPU (fallback)'} particle system${devils.gpu && devils.ps.depthCollision ? ', depth collision' : ''}`);
  // measurement harness (window.__sample; `?bench=1&size=WxH` pins the drawing buffer): fixed poses relative to the
  // buggy, with every time-dependent input pinned so captures reproduce across page loads (ENGINE.md "Measurement
  // protocol"): clouds are wall-clock driven, wind sway and dither fades read the time uniforms.
  // 0.8.0 poses: 'night-lamps' (meaningful with ?lamps=1&t=night; without the flag just the track) and 'devils' (one dust
  // devil stepped 180 x 1/60 s from a fixed spot and seed in after(), on whichever particle system the page runs).
  // Harness order (core/Bench.js pose()): pose fn, camera.updateMatrixWorld(), after(), quiet(), 2 frames: quiet runs
  // AFTER after(), so quiet keeps the devil a 'devils' pose just stepped (`posed`) and clears the devils for the others.
  const _f = new THREE.Vector3(), _bp = () => buggy.object3d.position, _bf = () => _f.set(0, 0, -1).applyQuaternion(buggy.object3d.quaternion);
  const _near = { i: 0, s: 0, lateral: 0, dist: 0, along: 0 };
  const buggyIndex = () => { const p = _bp(); return track.nearest(p.x, p.z, -1, _near).i; }; // global search: no live hint
  const DEVIL_POSE = { ahead: 70, side: 18, steps: 180, cam: 48 }; // track points (2 m) ahead of the buggy, m right of travel
  let posed = ''; // id of the pose being applied (after / quiet read it)
  const POSES = {
    straight: (cam) => { const p = _bp(), f = _bf(); cam.position.set(p.x - f.x * 30, p.y + 1.6, p.z - f.z * 30); cam.up.set(0, 1, 0); cam.lookAt(p.x + f.x * 120, p.y + 1, p.z + f.z * 120); },
    chase: (cam) => { const p = _bp(); cam.position.set(p.x, p.y + 4, p.z + 14); cam.up.set(0, 1, 0); cam.lookAt(p.x, p.y + 1, p.z); },
    aerial: (cam) => { const p = _bp(); cam.position.set(p.x, p.y + 120, p.z + 60); cam.up.set(0, 1, 0); cam.lookAt(p.x, p.y, p.z - 200); },
    ahead: (cam) => { const p = _bp(), f = _bf(); cam.position.set(p.x + f.x * 60, p.y + 1.6, p.z + f.z * 60); cam.up.set(0, 1, 0); cam.lookAt(p.x + f.x * 300, p.y + 1, p.z + f.z * 300); },
    // behind the buggy (heading-relative), higher and further back than chase, down the track: ~7 lamp heads in view
    // before the canyon bends behind a wall ~200 m out (more lamps' light spheres reach the view's clusters)
    'night-lamps': (cam) => {
      const p = _bp(), f = _bf(), l = Math.hypot(f.x, f.z) || 1, fx = f.x / l, fz = f.z / l;
      cam.position.set(p.x - fx * 24, p.y + 6, p.z - fz * 24); cam.up.set(0, 1, 0); cam.lookAt(p.x + fx * 110, p.y + 1, p.z + fz * 110);
    },
    // from the track centre 96 m ahead of the buggy towards the devil's spot 140 m ahead, 18 m right of travel, aimed
    // where the wind carries it in the 3 s the pose steps it (~46 m away, the funnel ~1/3 of the frame height)
    devils: (cam) => {
      const n = track.n, i0 = buggyIndex(), j = (i0 + DEVIL_POSE.ahead) % n, k = (i0 + DEVIL_POSE.cam) % n, sd = DEVIL_POSE.side;
      const dx = track.x[j] - track.tz[j] * sd, dz = track.z[j] + track.tx[j] * sd, cx = track.x[k], cz = track.z[k];
      cam.position.set(cx, terrain.heightAt(cx, cz) + 3.5, cz); cam.up.set(0, 1, 0);
      cam.lookAt(dx + devils.wind.x * 3, terrain.heightAt(dx, dz) + 8, dz + devils.wind.z * 3);
    },
  };
  const poses = {};
  for (const id in POSES) poses[id] = (cam, h) => { posed = id; POSES[id](cam, h); };
  const sample = createBenchHarness(engine, {
    name: 'canyon-run',
    poses,
    after: (cam) => {
      culler.update(cam); props.update(cam.position, 0.016); atmo.update(cam.position, 0);
      const su = atmo.sky.material.uniforms; su.uTime.value = 0; su.uCloudOffset.value.set(0.13, 0.41);
      U.uTime.value = 0; U.uViewPos.value.copy(cam.position);
      if (lamps) { lamps.update(atmo.darkness, true); clustered.update(cam); } // rows from the pinned darkness, clusters for this view
      if (posed === 'devils') {
        // depth collision reads the last rendered scene depth: render this view first (TAA jitter restarted), so the
        // steps collide with this pose's depth, not the previous pose's
        // (live particles cleared first: none drawn in that frame, so no jittered draw-time camera snapshot either)
        if (devils.gpu && devils.ps.depthCollision && postfx) { devils.reset(); if (postfx.history) postfx.history.reset(); engine.renderFn(scene, cam, 0); }
        devils.poseDevil((buggyIndex() + DEVIL_POSE.ahead) % track.n, DEVIL_POSE.side, DEVIL_POSE.steps, cam);
      }
    },
    quiet: () => {
      for (const k in rigs) rigs[k].deactivate();
      if (posed === 'devils') devils.timer = 1e9; // keep the devil after() just stepped, no live spawns
      else { devils.ps.clear(); devils.timer = 1e9; for (const d of devils.devils) d.on = false; }
      buggy.dust.clear();
      if (postfx) { postfx.autoExposure = false; postfx.setTimeOfDay(atmo.darkness); }
      if (postfx && postfx.history) { postfx.history.reset(); for (let i = 0; i < 32; i++) postfx.render(0); } // TAA: converge from a clean history
    },
    settings: { // get + set so __sample.run() restores every knob between settings (not cumulative)
      'postfx.enabled': { get: () => !!(postfx && postfx.enabled), set: (v) => { if (postfx) postfx.setEnabled(!!v); } },
      'postfx.aa': { get: () => (postfx ? postfx.aa : null), set: (v) => { if (postfx) postfx.set('aa', v); } },
      'postfx.gbuffer': { get: () => !!(postfx && postfx.gbuffer), set: (v) => { if (postfx) postfx.set('gbuffer', !!v); } },
      'postfx.ao': { get: () => !!(postfx && postfx.aoPass.enabled), set: (v) => { if (postfx) postfx.aoPass.enabled = !!v; } },
      'postfx.bloom': { get: () => !!(postfx && postfx.bloomPass.enabled), set: (v) => { if (postfx) postfx.bloomPass.enabled = !!v; } },
      ...fxSettings(() => postfx), // 0.8.0: postfx.ssr / volumetric / motionBlur / dof / lut.strength / ao.normals (+ their options)
      'shadows.kernel': { get: () => atmo.shadowKernel, set: (v) => atmo.setShadowKernel(v) },
      'time': {
        get: () => atmo.timeOfDay,
        set: (v) => {
          atmo.setTimeOfDay(v); if (postfx) postfx.setTimeOfDay(atmo.darkness);
          if (lamps) { lamps.update(atmo.darkness, true); clustered.update(camera); } // lamp rows + lens glow follow at once
        },
      },
      // ?lamps=1: light rows + lens glow on (darkness-driven) / off (intensity 0, out of every cluster): no recompile.
      // The devils' particle system (?gpuparticles=1) is chosen at load and has no setting.
      'lamps.enabled': {
        get: () => !!(lamps && lamps.enabled),
        set: (v) => { if (lamps) { lamps.setEnabled(!!v); lamps.update(atmo.darkness, true); clustered.update(camera); } },
      },
    },
  });
  const bp = readBenchParams(location.search); if (bp.size) sample.resize(bp.size.w, bp.size.h);
  sample.setReady();
  engine.start();
  if (bp.enabled) sample.freeze(); // ?bench=1: no live ticks before the first pose, so captures reproduce across loads
  loading.hide(450);
}

boot().catch((err) => {
  console.error(err);
  const l = document.querySelector('.ui-loading small');
  if (l) l.textContent = `Failed to start: ${err.message}`;
});
