// Physics sandbox — the XPBD dynamics world (src/physics/dynamics) on the engine (src/**) and three.js.
// Wiring: Engine (fixed step at the scene's rate: 60 Hz, the car and destructible 120 Hz; render interpolation, adaptive resolution,
// F3 profiler) + PhysicsWorld (flat ground at y = 0, static boxes) + DynamicsClient (in-thread; ?worker=1) attached
// to the PhysicsWorld (rigid bodies become 'rigid' colliders for controllers and rigs) + BodyPools (one InstancedPool
// per shape type, sleeping bodies dimmed) + PhysicsDebug (?debug=1, F4) + OrbitRig / FreeFlyRig (V) + InputMap +
// Atmosphere (keys 1-8) + PostFX (?nofx=1 to disable) + the UI kit HUD. The scenes live in scenes.js.
// Systems: input (early) -> physics (fixed: the scene's drivers, then dyn.step, then pose capture, so kinematic targets
// and forces apply in the same step) -> visuals (instance matrices, the scene's per-frame work, debug lines) -> camera
// -> world (atmosphere) -> hud (late).
// The worker is opt-in here (?worker=1), unlike Traffic, which runs off-thread unless told not to: this page is the
// measurement and debugging scene, so by default the world steps synchronously in this thread. Bench poses hand-step
// it, the scene drivers (kinematic car, recycled balls, timed spawns) act on the state of the same step, the render
// interpolates exact poses and the HUD's step ms is the cost being measured. ?worker=1 shows the off-thread mode (one or
// two steps of latency, poses extrapolated from the latest snapshot); the client falls back in-thread when the worker
// cannot start, and dyn.mode (not the URL) decides the interpolation, the HUD and the bench behaviour.
// Query: ?scene=stack|joints|chain|ragdoll|car|bench1000|cloth|rope|soft|buoyancy|destructible (default stack),
// ?bodies=N (bench1000), ?solid=0 (car: the crates are not colliders for the vehicle controller, only the kinematic box
// pushes them), ?fseed=N (destructible: the wall's fracture seed, default 35), ?worker=1 (not for the 0.9.0 scenes listed in SCENE_PARTICLES: particles, soft bodies, buoyancy and
// fracture live on the in-thread world, so they run in-thread and say so in the console),
// ?debug=1, ?substeps=N (4), ?hz=N, ?seed=N, ?t=<preset|hours>, ?nofx=1, ?fps=60, ?profile=N, ?gbuffer=1 (MRT scene
// pass + per-instance velocity for the body pools), ?aa=auto|msaa|fxaa|taa, ?cascades=2|3|4, ?shadows=fast|pcf|pcss,
// 0.8.0: ?ssr=1, ?volumetric=1, ?motionblur=<s>, ?dof=1[&focus=&fstop=], ?lut=warm|bleach, ?aonormals=gbuffer (samples/shared/fx-flags.js),
// ?bench=1&size=WxH (window.__sample: poses overview / close of the current scene; settings physics.debug,
// postfx.aa, postfx.gbuffer and, in-thread, physics.substeps). window.__sandbox = { engine, dyn, physics, scene (the current one), pools, dbg, spawn, reset, ... }.
// Invariant: every scene build (first load, R, a bench pose) ends with present(): the instance pools, the scene's
// per-frame visuals, the debug lines and the HUD are filled from the current physics state, so the next rendered
// frame shows the bodies even when no engine tick runs in between (the bench harness freezes the loop).
// Bench poses are reproducible across page loads: a pose = reset() (same seed, no thrown balls, and one empty step
// between dispose and rebuild so no warm-start / touching-pair state of the removed bodies survives into the reused
// ids: verified bit-identical to a fresh world) + exactly scene.benchSteps fixed steps at the scene's rate through the
// live path (fixedStep, no input) + present(), then the camera. The image does not depend on how long the page ran.
// In-thread only: in worker mode dyn.step is asynchronous, so a pose only presents the latest snapshot (console + HUD
// say so).
import * as THREE from 'three';
import { Engine, MathDet, createBenchHarness, readBenchParams } from '@vibe/engine/core';
import { PhysicsWorld, DynamicsClient, Shapes, attachToPhysicsWorld } from '@vibe/engine/physics';
import { Atmosphere, PostFX, PhysicsDebug, TIME_PRESETS, makeCanvas, finishTexture } from '@vibe/engine/render';
import { InputMap } from '@vibe/engine/input';
import { OrbitRig, FreeFlyRig } from '@vibe/engine/camera';
import { LoadingScreen } from '@vibe/engine/ui';
import { SCENES, SCENE_HZ, SCENE_PARTICLES, BodyPools, buildStatics } from './scenes.js';
import { PROXY_LAYER } from './car.js';
import { HUD, ACCENT, applyTheme } from './hud.js';
import { readFxFlags, aoFlags, loadLut, fxSettings } from '../shared/fx-flags.js';

const params = new URLSearchParams(location.search);
const flag = (k) => params.has(k) && params.get(k) !== '0';
const NOFX = flag('nofx');
// 0.7.0 knobs: ?gbuffer=1 (MRT scene pass, instanced-pool velocity, TAA), ?aa=auto|msaa|fxaa|taa, ?cascades=2|3|4, ?shadows=fast|pcf|pcss
const GBUFFER = flag('gbuffer');
// drawing-buffer pixels above which the post stack uses FXAA instead of 4x MSAA (as in Canyon Run): ?msaamax=N
const MSAA_MAX = params.has('msaamax') ? (params.get('msaamax') === '' ? Infinity : +params.get('msaamax') || 0) : 4e6;
const SCENE_IDS = Object.keys(SCENES);
const SCENE = SCENE_IDS.includes(params.get('scene')) ? params.get('scene') : 'stack';
if (params.has('scene') && params.get('scene') !== SCENE) console.warn(`[sandbox] unknown scene '${params.get('scene')}' (${SCENE_IDS.join(', ')}): stack`);
const WORKER = flag('worker');
// 0.9.0 scenes: the world's particle capacity (0: none); any listed scene needs the in-thread world (dyn.world)
const PARTICLES = SCENE_PARTICLES[SCENE] ?? -1;
const IN_THREAD = PARTICLES >= 0;
const DEBUG = flag('debug');
const SEED = (+params.get('seed') || 1) >>> 0;
const HZ = +params.get('hz') || SCENE_HZ[SCENE] || 60;
const SUBSTEPS = Math.max(1, +params.get('substeps') || 4);
const BODIES = +params.get('bodies') || 1000;
const PRESET_KEYS = Object.keys(TIME_PRESETS);
const BALL_R = 0.25, BALL_MASS = 30, BALL_SPEED = 22, MAX_BALLS = 64;

// keys (the car's own controls are in car.js). The map is silenced while the help card is up (as in the character
// sample): reset / camera / debug / help / time keep working (alwaysOn), Space and the free-fly moves do not. The
// free-fly actions are defined here (FreeFlyRig adopts existing ones) so that Space stays "throw a ball" (FreeFlyRig's
// default binds it to "up"), and without alwaysOn so the help card silences them too.
const ACTIONS = {
  spawn: { type: 'button', bindings: ['Space'] },
  reset: { type: 'button', alwaysOn: true, bindings: ['KeyR'] },
  camera: { type: 'button', alwaysOn: true, bindings: ['KeyV'] },
  debug: { type: 'button', alwaysOn: true, bindings: ['F4'] },
  help: { type: 'button', alwaysOn: true, bindings: ['KeyH'] },
  'freeFly.x': { type: 'axis', bindings: [{ key: 'KeyD', value: 1 }, { key: 'KeyA', value: -1 }] },
  'freeFly.y': { type: 'axis', bindings: [{ key: 'KeyE', value: 1 }, { key: 'KeyQ', value: -1 }] },
  'freeFly.z': { type: 'axis', bindings: [{ key: 'KeyS', value: 1 }, { key: 'KeyW', value: -1 }] },
  'freeFly.fast': { type: 'button', bindings: ['ShiftLeft', 'ShiftRight'] },
  'freeFly.slow': { type: 'button', bindings: ['AltLeft', 'AltRight'] },
};
const TIME_ACTIONS = PRESET_KEYS.map((name, i) => { const a = `time.${name}`; ACTIONS[a] = { type: 'button', alwaysOn: true, bindings: [`Digit${i + 1}`] }; return a; });

// ------------------------------------------------------------------ textures (no assets)
/** 4 m prototype tile: 1 m checker of two close greys, 1 m lines, a brighter 4 m line (as in the character sample). */
function groundTexture(renderer) {
  const S = 512, cell = S / 4, c = makeCanvas(S), g = c.getContext('2d');
  for (let y = 0; y < 4; y++) for (let x = 0; x < 4; x++) { g.fillStyle = (x + y) & 1 ? '#7d838b' : '#858b93'; g.fillRect(x * cell, y * cell, cell, cell); }
  g.fillStyle = 'rgba(255,255,255,0.14)';
  for (let i = 1; i < 4; i++) { g.fillRect(i * cell - 1, 0, 2, S); g.fillRect(0, i * cell - 1, S, 2); }
  g.fillStyle = 'rgba(255,255,255,0.34)';
  g.fillRect(0, 0, 2, S); g.fillRect(S - 1, 0, 1, S); g.fillRect(0, 0, S, 2); g.fillRect(0, S - 1, S, 1);
  return finishTexture(new THREE.CanvasTexture(c), renderer);
}
/** Box faces: white with a darker frame (the instance colour tints it), so edges and rotation read. */
function boxTexture(renderer) {
  const S = 128, c = makeCanvas(S), g = c.getContext('2d');
  g.fillStyle = '#ffffff'; g.fillRect(0, 0, S, S);
  g.fillStyle = '#b8b8b8'; g.fillRect(0, 0, S, 7); g.fillRect(0, S - 7, S, 7); g.fillRect(0, 0, 7, S); g.fillRect(S - 7, 0, 7, S);
  g.fillStyle = 'rgba(0,0,0,0.07)'; g.fillRect(S / 2 - 1, 7, 2, S - 14);
  return finishTexture(new THREE.CanvasTexture(c), renderer);
}
/** Spheres / capsules: 8 meridian stripes and an equator band, so rolling reads. */
function roundTexture(renderer) {
  const W = 256, H = 128, c = makeCanvas(W, H), g = c.getContext('2d');
  for (let i = 0; i < 8; i++) { g.fillStyle = i & 1 ? '#cfcfcf' : '#ffffff'; g.fillRect((i * W) / 8, 0, W / 8, H); }
  g.fillStyle = 'rgba(0,0,0,0.14)'; g.fillRect(0, H / 2 - 3, W, 6);
  return finishTexture(new THREE.CanvasTexture(c), renderer);
}

/** Ground plane (1000 m, 4 m tiles). The collider is PhysicsWorld.groundLevel = 0. */
function buildGround(scene, renderer) {
  const map = groundTexture(renderer);
  map.repeat.set(250, 250);
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(1000, 1000).rotateX(-Math.PI / 2), new THREE.MeshStandardMaterial({ map, roughness: 0.92, metalness: 0 }));
  ground.name = 'ground'; ground.receiveShadow = true;
  scene.add(ground);
  return ground;
}

// ------------------------------------------------------------------ boot
const nextFrame = () => new Promise((r) => setTimeout(r, 0)); // yield to the loading screen

async function boot() {
  const t0 = performance.now();
  applyTheme();
  const loading = new LoadingScreen({ title: 'SANDBOX', label: 'Pouring the ground…', accent: ACCENT, background: '#0e1116' });
  const physics = new PhysicsWorld({ groundLevel: 0, cellSize: 8 });
  const engine = new Engine({
    container: document.getElementById('app'), fixedHz: HZ, maxSubsteps: HZ >= 120 ? 4 : 3, renderer: { antialias: NOFX }, fov: 60,
    physics, profile: params.get('profile') ? +params.get('profile') : false, targetFps: +params.get('fps') || 0,
  });
  const { scene, camera, renderer } = engine;
  const t = params.get('t');
  const atmo = new Atmosphere({
    scene, renderer, camera, timeOfDay: t == null ? 'afternoon' : Number.isFinite(+t) ? +t : t,
    shadows: { cascades: +params.get('cascades') || 2, near: 24, far: [120, 240, 420], kernel: params.get('shadows') || 'fast' }, maxCasterHeight: 40,
  });

  // ---------------------------------------------------------------- static world first (a worker client copies it)
  const W = {};
  engine.addSystem({
    async load(progress) {
      progress(0.3, 'Pouring the ground…');
      await nextFrame();
      W.ground = buildGround(scene, renderer);
      W.statics = buildStatics(SCENE, { physics, scene, renderer });
      progress(1, 'Ready');
    },
  }, 'world-load');
  await engine.load((p, label) => loading.setProgress(p * 0.5, label));

  // ---------------------------------------------------------------- dynamics
  if (WORKER && IN_THREAD) console.info(`[sandbox] ?worker=1 ignored: the ${SCENE} scene needs the in-thread world (particles, buoyancy, fracture)`);
  loading.setProgress(0.6, WORKER && !IN_THREAD ? 'Starting the physics worker…' : 'Starting the physics…');
  const dyn = new DynamicsClient({
    timing: true, // the HUD shows the step time (the world's default is off: timers box doubles)
    worker: WORKER && !IN_THREAD, physics, gravity: [0, -9.81, 0], substeps: SUBSTEPS, maxBodies: Math.min(65536, Math.max(4096, BODIES + 512)),
    maxContacts: Math.max(16384, 12 * BODIES), sleep: { linear: 0.05, angular: 0.05, time: 0.5 }, seed: SEED,
    ...(PARTICLES > 0 ? { maxParticles: PARTICLES } : {}), // only the particle scenes: every other world is created as before
  });
  await dyn.ready;
  const WORKER_ON = dyn.mode === 'worker'; // the client falls back in-thread when the worker fails to start
  if (WORKER && !IN_THREAD && !WORKER_ON) console.warn('[sandbox] ?worker=1 but the dynamics worker did not start: running in-thread');
  // the car / rigs / controllers see the bodies as 'rigid' boxes, except the car's own kinematic proxy
  attachToPhysicsWorld(dyn, physics, { mask: ~PROXY_LAYER });
  const pools = new BodyPools(scene, dyn, { interpolate: !WORKER_ON, fixedDt: engine.fixedDt, velocity: GBUFFER, textures: { box: boxTexture(renderer), round: roundTexture(renderer) } });
  const dbg = new PhysicsDebug(dyn, { maxLines: Math.max(65536, 48 * BODIES) });
  dbg.enabled = DEBUG;
  scene.add(dbg);

  // ---------------------------------------------------------------- HUD, input, camera rigs
  const href = (id) => { const p = new URLSearchParams(location.search); p.set('scene', id); return `${location.pathname}?${p}`; };
  const hud = new HUD({ scenes: SCENE_IDS, current: SCENE, href, worker: WORKER_ON });
  if (WORKER && !WORKER_ON) hud.flash.show(IN_THREAD ? 'IN-THREAD SCENE' : 'NO WORKER · IN-THREAD');
  const canvas = renderer.domElement;
  const map = new InputMap({ actions: ACTIONS });
  const focus = new THREE.Vector3();
  const orbit = new OrbitRig(camera, canvas, { active: false, target: focus, distance: 14, elevation: 0.3, azimuth: 0.5, offsetY: 0, minDistance: 1.5, maxDistance: 150, minElevation: 0.02, physics });
  const free = new FreeFlyRig(camera, canvas, { active: false, input: map, speed: 12, collide: true, minY: 0.3, physics });
  const rigs = { orbit, free };
  let rig = null, current = null, built = 0;
  /** Help card up: the map is silent (bar the alwaysOn keys); free fly: WASD fly the camera, not the car. */
  const syncInput = () => {
    const help = hud.help.visible;
    map.enabled = !help;
    if (current && current.setInputEnabled) current.setInputEnabled(rig !== free && !help);
  };
  hud.onHelp = syncInput;
  const setRig = (name) => {
    for (const k in rigs) if (k !== name) rigs[k].deactivate();
    rig = rigs[name]; rig.activate();
    syncInput();
  };

  // ---------------------------------------------------------------- scene lifecycle
  const ctx = { dyn, physics, scene, pools, params, THREE, rand: null, statics: W.statics, renderer }; // statics: buildStatics' result (the pool)
  const getSubsteps = () => (dyn.world ? dyn.world.substeps : SUBSTEPS); // worker mode: fixed at construction
  const hs = { // HUD state (reused)
    stats: null, fps: 0, latency: NaN, substeps: SUBSTEPS, hz: HZ, extra: null,
    overflow: () => { const c = dyn.contacts; return c ? c.overflow : 0; }, // read at the HUD's 4 Hz refresh only
  };
  const readHud = () => {
    hs.stats = dyn.stats || null; hs.fps = engine.profiler.frame.fps; hs.latency = dyn.latency;
    hs.substeps = getSubsteps(); hs.extra = current.status || null;
    return hs;
  };
  /** One fixed step, the same for the live loop and the bench settle: the scene's drivers (kinematic targets, forces,
   *  spawns, teleports) first so they apply in this step, then the solve, then the pose capture for interpolation
   *  (the scene's own meshes: afterStep, e.g. particle / fracture meshes). */
  const fixedStep = (dt) => {
    if (current.fixedUpdate) current.fixedUpdate(dt);
    dyn.step(dt);
    pools.capture();
    if (current.afterStep) current.afterStep(dt);
  };
  /** Fill everything drawn from the physics state now (see the header invariant). */
  const present = () => {
    pools.capture(); pools.sync(1);
    if (current.update) current.update(0, 1);
    if (dbg.enabled) dbg.update();
    hud.update(0, readHud(), true);
  };
  const build = () => {
    ctx.rand = MathDet.mulberry32(SEED); // fresh per build: R rebuilds the same layout
    current = SCENES[SCENE](ctx);
    hud.setScene(SCENE, current.info);
    hud.setCarKeys(!!current.car);
    if (current.target) orbit.setTarget(current.target);
    else {
      if (current.focus) focus.copy(current.focus); else focus.set(0, 0, 0);
      if (orbit.target !== focus) orbit.setTarget(focus);
    }
    if (!built++) { // the scene's framing on the first build only; a reset keeps the user's orbit
      const v = current.view || {};
      if (v.distance) orbit.distance = v.distance;
      if (v.elevation != null) orbit.elevation = v.elevation;
      if (v.azimuth != null) orbit.azimuth = v.azimuth;
      orbit.offset.y = v.offsetY ?? 0;
      orbit.snap();
    }
    syncInput();
    present();
  };
  const balls = [];
  const clearBalls = () => { for (const id of balls) { pools.untrack(id); dyn.removeBody(id); } balls.length = 0; };
  // One empty step between dispose and rebuild: the reused body ids would otherwise inherit the narrowphase's
  // warm-start entries (and touching pairs) of the bodies just removed, so a rebuilt resting stack would depend on the
  // state it replaced. The step evicts them (nothing is alive to collide).
  const reset = () => { clearBalls(); current.dispose(); pools.clear(); dyn.step(engine.fixedDt); build(); };
  /** Bench pose `name`: the scene exactly benchSteps fixed steps after a fresh build, whatever ran before (header). */
  let benchWarned = false;
  const benchPose = (name) => (cam) => {
    if (WORKER_ON) {
      if (!benchWarned) { benchWarned = true; console.warn('[sandbox] worker mode: bench poses are not settled (dyn.step is asynchronous), they show the latest snapshot; measure without ?worker=1 for reproducible poses'); }
      hud.flash.show('WORKER · POSE NOT SETTLED');
    } else {
      reset();
      const K = current.benchSteps ?? 120;
      for (let i = 0; i < K; i++) fixedStep(engine.fixedDt);
    }
    present();
    current.poses[name](cam);
  };

  // Space: a ball from the camera, thrown along the view direction (applyImpulse); the oldest goes past MAX_BALLS
  const BALL = Shapes.sphere(BALL_R), _dir = new THREE.Vector3();
  const spawn = () => {
    camera.getWorldDirection(_dir);
    const p = camera.position, x = p.x + _dir.x * 1.2, y = Math.max(BALL_R + 0.05, p.y + _dir.y * 1.2), z = p.z + _dir.z * 1.2;
    const id = dyn.addBody({ type: 'dynamic', shape: BALL, position: [x, y, z], quaternion: [0, 0, 0, 1], mass: BALL_MASS, friction: 0.5, restitution: 0.3 });
    if (!(id >= 0)) return -1;
    pools.track(id, 'sphere', [BALL_R], ACCENT);
    const J = BALL_MASS * BALL_SPEED;
    dyn.applyImpulse(id, x, y, z, _dir.x * J, _dir.y * J, _dir.z * J);
    balls.push(id);
    if (balls.length > MAX_BALLS) { const old = balls.shift(); pools.untrack(old); dyn.removeBody(old); }
    return id;
  };

  build();
  setRig('orbit');

  const setDebug = (on) => { dbg.enabled = !!on; if (dbg.enabled) dbg.update(); };

  // ---------------------------------------------------------------- post-processing
  let postfx = null;
  if (!NOFX) {
    try {
      const { width, height } = engine.size;
      // 0.8.0 knobs (samples/shared/fx-flags.js): ?ssr=1, ?volumetric=1, ?motionblur=<s>, ?dof=1, ?lut=<name>, ?aonormals=gbuffer
      const fx = readFxFlags(params), lut = await loadLut(params);
      postfx = new PostFX(renderer, scene, camera, { atmosphere: atmo, width, height, pixelRatio: engine.pixelRatio, ao: { radius: 0.9, thickness: 0.9, samples: 8, ...aoFlags(params) }, msaaMaxPixels: MSAA_MAX, gbuffer: GBUFFER, aa: params.get('aa') || 'auto', ...fx, lut });
      // bodies move: per-instance velocity from the pools (InstancedPool({ velocity }) when ?gbuffer=1), replayed whenever the G-buffer is built
      for (const b of pools.buckets) postfx.trackPool(b.pool);
      pools.onBucket = (pool) => postfx.trackPool(pool);
    } catch (e) { console.warn('[sandbox] post-processing unavailable', e); atmo.setFogMode('scene'); }
  }
  engine.renderFn = (sc, cam, dt) => {
    if (postfx) postfx.render(dt);
    else { renderer.render(sc, cam); atmo.renderFlare(renderer, cam); }
  };
  engine.onResize = (w, h, pr) => { if (postfx) postfx.setSize(w, h, pr); };

  // ---------------------------------------------------------------- systems (phase order = registration order)
  engine.addSystem({
    earlyUpdate(dt) {
      map.update(dt);
      if (map.justPressed('help')) hud.toggleHelp();
      if (map.justPressed('reset')) { reset(); hud.flash.show('RESET'); }
      if (map.justPressed('camera')) { setRig(rig === orbit ? 'free' : 'orbit'); hud.flash.show(rig === free ? 'FREE CAMERA' : 'ORBIT CAMERA'); }
      if (map.justPressed('debug')) { setDebug(!dbg.enabled); hud.flash.show(dbg.enabled ? 'DEBUG ON' : 'DEBUG OFF'); }
      if (map.justPressed('spawn')) spawn();
      for (let k = 0; k < TIME_ACTIONS.length; k++) {
        if (map.justPressed(TIME_ACTIONS[k])) { atmo.setTimeOfDay(PRESET_KEYS[k]); hud.flash.show(PRESET_KEYS[k]); }
      }
      if (current.earlyUpdate) current.earlyUpdate(dt);
    },
  }, 'input');
  engine.addSystem({ fixedUpdate: fixedStep }, 'physics');
  engine.addSystem({
    update(dt, alpha) {
      pools.sync(alpha);
      if (current.update) current.update(dt, alpha);
      if (dbg.enabled) dbg.update();
    },
  }, 'visuals');
  engine.addSystem({ update: (dt) => rig.update(dt) }, 'camera');
  engine.addSystem({
    lateUpdate(dt) {
      camera.updateMatrixWorld();
      atmo.update(camera.position, dt);
    },
  }, 'world');
  engine.addSystem({ lateUpdate: (dt) => hud.update(dt, readHud()) }, 'hud');

  // ---------------------------------------------------------------- bench harness (window.__sample)
  const settings = {
    'physics.debug': { get: () => !!dbg.enabled, set: (v) => setDebug(v) },
    'postfx.aa': { get: () => (postfx ? postfx.aa : null), set: (v) => { if (postfx) postfx.set('aa', v); } },
    'postfx.gbuffer': { get: () => !!(postfx && postfx.gbuffer), set: (v) => { if (postfx) postfx.set('gbuffer', !!v); } },
    ...fxSettings(() => postfx), // 0.8.0: postfx.ssr / volumetric / motionBlur / dof / lut.strength / ao.normals (+ their options)
  };
  // substeps are a field of the in-thread world; a worker's world got them at construction (no knob then)
  if (dyn.world) settings['physics.substeps'] = { get: () => dyn.world.substeps, set: (v) => { dyn.world.substeps = Math.max(1, Math.min(64, v | 0)); } };
  const harness = createBenchHarness(engine, {
    name: 'sandbox',
    poses: { overview: benchPose('overview'), close: benchPose('close') }, // overview: camera fixed above the scene
    after: (cam) => { atmo.update(cam.position, 0); if (dbg.enabled) dbg.update(); }, // then quiet() pins the sky
    quiet() {
      if (postfx) { postfx.autoExposure = false; postfx.setTimeOfDay(atmo.darkness); }
      const su = atmo.sky.material.uniforms; su.uTime.value = 0; su.uCloudOffset.value.set(0.13, 0.41); // pin the sky (stars, clouds)
      if (postfx && postfx.history) { postfx.history.reset(); for (let i = 0; i < 32; i++) postfx.render(0); } // TAA: converge from a clean history
    },
    settings,
  });
  const bp = readBenchParams(location.search);
  if (bp.enabled && bp.size) harness.resize(bp.size.w, bp.size.h);

  renderer.compile(scene, camera);
  const loadMs = Math.round(performance.now() - t0);
  window.__sandbox = {
    THREE, engine, dyn, physics, get scene() { return current; }, pools, dbg, spawn, reset, setDebug, fixedStep, present,
    get particles() { return dyn.world ? dyn.world.particles || null : null; }, statics: W.statics, // 0.9.0 scenes: the ParticleStore, the pool
    get rig() { return rig; }, rigs, setRig, map, hud, atmo, postfx, harness, params, loadMs,
  };
  console.info(`[sandbox] ready in ${loadMs} ms · ${SCENE} · ${pools.count} drawn bodies${PARTICLES > 0 ? ` · ${PARTICLES} particle slots` : ''} · ${WORKER_ON ? 'worker' : 'in-thread'} · ${HZ} Hz x ${getSubsteps()} substeps`);
  engine.start();
  loading.hide(400);
  harness.setReady();
  if (bp.enabled) harness.freeze(); // ?bench=1: no live ticks before the first pose, so captures reproduce across loads
}

boot().catch((err) => {
  console.error(err);
  const l = document.querySelector('.ui-loading small');
  if (l) l.textContent = `Failed to start: ${err.message}`;
});
