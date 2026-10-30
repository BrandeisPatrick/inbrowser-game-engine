// Character — a third-person playground built only on the engine (src/**) and three.js.
// Wiring: Engine (120 Hz fixed step, render interpolation, adaptive resolution, F3 profiler) + PhysicsWorld (flat
// ground + static boxes: walls, a low-ceiling corridor, a staircase of 0.3 m steps, a ramp onto a 2 m platform, crates)
// + CharacterController (kinematic capsule, fixed step) + AnimationController (idle / walk / run 1D blend tree on the
// horizontal speed + one-shot jump, per frame) on RobotExpressive.glb (fetched by tools/fetch-models.mjs, loaded through
// Assets) or the procedural rig of rig.js + ThirdPersonRig / FirstPersonRig / OrbitRig (V cycles) + CharacterInput +
// PointerLock + Atmosphere (keys 1-8) + PostFX (?nofx=1 to disable) + the UI kit HUD. The player is an Entity with
// three components: Motor (controller, fixed step + interpolation), Animator (state machine), Facing (turn + rig state).
// Query: ?model=<url> (any rigged glTF; relative to this page) | ?model=procedural, ?t=<preset|hours>, ?nofx=1,
// ?fps=60, ?profile=N, ?shadows=pcf, ?bench=1&size=WxH (window.__sample: poses idle / run / aerial).
// window.__character = { engine, cc, ctl, rig (the active one), ... } for the console.
import * as THREE from 'three';
import { Engine, Assets, Entity, EntitySystem, Events, createBenchHarness, readBenchParams } from '@vibe/engine/core';
import { PhysicsWorld, CharacterController } from '@vibe/engine/physics';
import { Atmosphere, PostFX, TIME_PRESETS, registerRenderLoaders, InstancedPool, makeCanvas, finishTexture } from '@vibe/engine/render';
import { AnimationController } from '@vibe/engine/anim';
import { CharacterInput, PointerLock } from '@vibe/engine/input';
import { ThirdPersonRig, FirstPersonRig, OrbitRig } from '@vibe/engine/camera';
import { LoadingScreen } from '@vibe/engine/ui';
import { buildProceduralCharacter, mapGltfCharacter } from './rig.js';
import { HUD, ACCENT, applyTheme } from './hud.js';

const params = new URLSearchParams(location.search);
const NOFX = params.has('nofx') && params.get('nofx') !== '0';
// drawing-buffer pixels above which the post stack uses FXAA instead of 4x MSAA (as in Canyon Run): ?msaamax=N
const MSAA_MAX = params.has('msaamax') ? (params.get('msaamax') === '' ? Infinity : +params.get('msaamax') || 0) : 4e6;
// this page's folder (trailing slash) whether it was opened as /character/ or /character/index.html: model URLs resolve here
const PAGE_DIR = new URL(location.pathname.replace(/[^/]*\.html?$/, '').replace(/\/?$/, '/'), location.origin).href;
const MODEL = params.get('model');
const MODEL_URL = MODEL && /^(0|off|none|procedural)$/i.test(MODEL) ? null : MODEL || 'models/RobotExpressive.glb';

const HEIGHT = 1.8;
const PROBE_MS = 2500;
const SPAWN = [0, 0, 5];
const PRESET_KEYS = Object.keys(TIME_PRESETS);
const LOCO = { idle: 0.3, walk: 3.75 };   // HUD animation label thresholds (m/s)
const wrapPi = (a) => { while (a > Math.PI) a -= 2 * Math.PI; while (a < -Math.PI) a += 2 * Math.PI; return a; };

/** HEAD probe: true when url answers with something that is not an HTML page (dev / preview servers answer unknown
 *  paths with an index.html when the request accepts HTML, hence the explicit Accept and the content-type check: a
 *  build without models/ gets a plain 404). Gives up after PROBE_MS (slow or hanging host) -> procedural rig. */
async function probe(url) {
  const ac = typeof AbortController !== 'undefined' ? new AbortController() : null;
  const timer = ac ? setTimeout(() => ac.abort(), PROBE_MS) : 0;
  try {
    const r = await fetch(url, { method: 'HEAD', signal: ac ? ac.signal : undefined, headers: { Accept: 'model/gltf-binary, model/gltf+json, application/octet-stream' } });
    return r.ok && !/text\/html/i.test(r.headers.get('content-type') || '');
  } catch { return false; } finally { clearTimeout(timer); }
}

// ------------------------------------------------------------------ playground
/** 4 m prototype tile: 1 m checker of two close greys, 1 m lines, a brighter 4 m line. */
function groundTexture(renderer) {
  const S = 512, cell = S / 4, c = makeCanvas(S), g = c.getContext('2d');
  for (let y = 0; y < 4; y++) for (let x = 0; x < 4; x++) { g.fillStyle = (x + y) & 1 ? '#7d838b' : '#858b93'; g.fillRect(x * cell, y * cell, cell, cell); }
  g.fillStyle = 'rgba(255,255,255,0.14)';
  for (let i = 1; i < 4; i++) { g.fillRect(i * cell - 1, 0, 2, S); g.fillRect(0, i * cell - 1, S, 2); }
  g.fillStyle = 'rgba(255,255,255,0.34)';
  g.fillRect(0, 0, 2, S); g.fillRect(S - 1, 0, 1, S); g.fillRect(0, 0, S, 2); g.fillRect(0, S - 1, S, 1);
  return finishTexture(new THREE.CanvasTexture(c), renderer);
}

/** Wedge (ramp) from (x0..x1, y 0, zLow) up to height h at zHigh; flat normals, no bottom face. */
function wedgeGeometry(x0, x1, zLow, zHigh, h) {
  const A = [x0, 0, zLow], B = [x1, 0, zLow], C = [x1, h, zHigh], D = [x0, h, zHigh], E = [x0, 0, zHigh], F = [x1, 0, zHigh];
  const tris = [A, B, C, A, C, D, A, D, E, B, F, C, E, D, C, E, C, F]; // CCW from outside (ramp rising toward -Z)
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(tris.flat(), 3));
  g.computeVertexNormals();
  return g;
}

const COLORS = { wall: '#cbc7bf', stair: '#6b8db3', platform: '#e0874a', ramp: '#5fae8b', crate: '#d8b356', pillar: '#98a1ab', low: '#b5ab9d' };

/** Ground plane + every block (one InstancedPool draw) + the ramp; each block is also a static box collider. */
function buildPlayground(scene, renderer, physics) {
  const map = groundTexture(renderer);
  map.repeat.set(250, 250); // 1000 m plane, 4 m tiles
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(1000, 1000).rotateX(-Math.PI / 2), new THREE.MeshStandardMaterial({ map, roughness: 0.92, metalness: 0 }));
  ground.name = 'ground'; ground.receiveShadow = true;
  scene.add(ground);

  const pool = new InstancedPool(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshStandardMaterial({ roughness: 0.78, metalness: 0 }),
    { capacity: 32, colors: true, castShadow: true, cullRadius: 19, name: 'blocks', dynamic: false });
  const col = new THREE.Color();
  const block = (x0, x1, y0, y1, z0, z1, color) => {
    physics.addStaticBox(x0, y0, x1, y1, z0, z1); // (minX, minY, maxX, maxY, minZ, maxZ)
    const i = pool.add();
    pool.setTRS(i, (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2, 0, x1 - x0, y1 - y0, z1 - z0);
    col.set(color); pool.setColor(i, col.r, col.g, col.b);
  };
  // walls: back, right, and a 1.5 m corridor (west) with a 2.6 m ceiling over its middle
  block(-18, 18, 0, 3, -15, -14.5, COLORS.wall);
  block(17.5, 18, 0, 3, -14.5, 8, COLORS.wall);
  block(-16.6, -16.1, 0, 3, -12, 4, COLORS.wall);
  block(-14.6, -14.1, 0, 3, -12, 4, COLORS.wall);
  block(-16.6, -14.1, 2.6, 2.9, -10, -4, COLORS.low);
  // staircase (rising toward -Z): five 0.3 m steps (0.45 m treads) onto a 1.8 m landing
  for (let k = 0; k < 5; k++) block(-11, -8, 0, (k + 1) * 0.3, 2 - (k + 1) * 0.45, 2 - k * 0.45, COLORS.stair);
  block(-11, -8, 0, 1.8, -4.2, 2 - 5 * 0.45, COLORS.stair);
  // 2 m platform, reached by the ramp below
  block(7, 13, 0, 2, -9, -3, COLORS.platform);
  // crates of 0.5 / 1 / 1.5 m (jump apex ~1.27 m: each is one jump up from the previous)
  block(-5.2, -4, 0, 0.5, -8.1, -6.9, COLORS.crate);
  block(-3.8, -2.6, 0, 1, -8.1, -6.9, COLORS.crate);
  block(-2.4, -1.2, 0, 1.5, -8.1, -6.9, COLORS.crate);
  // pillars (camera occlusion) and a 0.6 m wall (jump over)
  for (const z of [-1, -5, -9]) block(3.2, 3.8, 0, 4, z - 0.3, z + 0.3, COLORS.pillar);
  block(-6, 0, 0, 0.6, 9.8, 10.1, COLORS.low);
  pool.update(); // bounds for frustum culling (static from now on)
  scene.add(pool.mesh);

  // ramp: rendered as a wedge, collided as 40 thin slices whose tops sit on the slope at their centres (+-2.5 cm)
  const RX0 = 8.5, RX1 = 11.5, ZLOW = 5, ZHIGH = -3, RH = 2, N = 40, len = (ZLOW - ZHIGH) / N;
  const ramp = new THREE.Mesh(wedgeGeometry(RX0, RX1, ZLOW, ZHIGH, RH), new THREE.MeshStandardMaterial({ color: COLORS.ramp, roughness: 0.8 }));
  ramp.name = 'ramp'; ramp.castShadow = true; ramp.receiveShadow = true;
  scene.add(ramp);
  for (let k = 0; k < N; k++) physics.addStaticBox(RX0, 0, RX1, (RH * (k + 0.5)) / N, ZLOW - (k + 1) * len, ZLOW - k * len);
  return { ground, pool, ramp };
}

// ------------------------------------------------------------------ player components (Entity)
/** Fixed step: the camera-relative intent -> CharacterController; per frame: interpolated pose on the entity. */
class Motor {
  constructor(cc, physics, intent) {
    this.name = 'Motor';
    this.cc = cc; this.physics = physics; this.intent = intent;
    this.prev = new THREE.Vector3().copy(cc.position);
    this.jumped = false; this.landed = false; this.landingSpeed = 0;   // latched for the per-frame components
    this.yOff = 0; // visual offset that eases out step-up pops (stairs, ramp slices)
  }
  reset(x, y, z) {
    this.cc.teleport(x, y, z); this.prev.copy(this.cc.position);
    this.intent.jump = false; this.jumped = false; this.landed = false; this.yOff = 0;
    if (this.entity) { this.entity.position.copy(this.cc.position); this.entity.updateMatrixWorld(); }
  }
  fixedUpdate(dt) {
    const cc = this.cc, I = this.intent;
    this.prev.copy(cc.position);
    cc.step(dt, I.moveX, I.moveZ, I.jump, I.sprint, this.physics);
    I.jump = false; // the press latched by the input system is consumed by the first step after it
    if (cc.landedThisStep) { this.landed = true; this.landingSpeed = cc.landingSpeed; }
    if (cc.jumpedThisStep) this.jumped = true;
    if (cc.steppedUp) this.yOff -= cc.position.y - this.prev.y;
  }
  update(dt, alpha) {
    this.yOff *= Math.exp(-20 * dt);
    this.entity.position.lerpVectors(this.prev, this.cc.position, alpha);
    this.entity.position.y += this.yOff;
  }
}

/** Animation state: locomotion tree driven by the horizontal speed; jump on take-off, back on landing / clip end. */
class Animator {
  constructor(ctl, cc, motor, events) {
    this.name = 'Animator';
    this.ctl = ctl; this.cc = cc; this.motor = motor; this.events = events;
    this.mode = 'locomotion'; this.label = 'IDLE';
    this.frozen = () => false; // the bench harness holds the pose (main.js points this at harness.frozen)
    ctl.events.on('end', (s) => {
      const st = typeof s === 'string' ? s : s && (s.name || s.state);
      if (st === 'jump' && this.mode === 'jump' && this.cc.grounded) this.locomotion(); // airborne: the clamped last frame holds; landing returns
    });
  }
  locomotion() { this.mode = 'locomotion'; this.ctl.play('locomotion'); }
  update(dt) {
    if (this.frozen()) return; // manual engine.tick() while a bench pose is held: the mixer does not advance
    const m = this.motor, cc = this.cc;
    if (m.landed) { m.landed = false; if (this.mode === 'jump') this.locomotion(); this.events.emit('land', m.landingSpeed); }
    if (m.jumped) { m.jumped = false; this.mode = 'jump'; this.ctl.play('jump'); this.events.emit('jump'); }
    this.ctl.setParam('speed', cc.speed);
    this.ctl.update(dt);
    this.label = this.mode === 'jump' ? 'JUMP' : !cc.grounded ? 'FALL' : cc.speed < LOCO.idle ? 'IDLE' : cc.speed < LOCO.walk ? 'WALK' : 'RUN';
  }
}

/**
 * Facing (engine heading: 0 = -Z, + = left) eases toward the move intent (third person / orbit) or the view yaw (first
 * person); the entity carries it (its local -Z is forward; the model inside is turned by PI, glTF faces +Z). Also fills
 * the rig state (one reused object) and hides the head in first person (its bone scaled after the animation pose).
 */
class Facing {
  constructor(cc, intent, state, view) {
    this.name = 'Facing';
    this.cc = cc; this.intent = intent; this.state = state; this.view = view;
    this.firstPerson = false; this.fpRig = null; this.head = null;
    this.headRest = new THREE.Vector3(1, 1, 1); this._fp = false;
  }
  /** Head bone hidden in first person (null: the whole view is hidden instead); its rest scale is restored after. */
  setHead(bone) { this.head = bone || null; if (bone) this.headRest.copy(bone.scale); this._fp = false; }
  update(dt) {
    const st = this.state, I = this.intent, cc = this.cc;
    let goal = st.facing, rate = 10;
    if (this.firstPerson && this.fpRig) { goal = this.fpRig.yaw; rate = 30; }
    else if (I.moveX * I.moveX + I.moveZ * I.moveZ > 0.01) goal = Math.atan2(-I.moveX, -I.moveZ);
    st.facing = wrapPi(st.facing + wrapPi(goal - st.facing) * (1 - Math.exp(-rate * dt)));
    st.speed = cc.speed; st.grounded = cc.grounded; // st.velocity is cc.velocity (same object)
    this.entity.rotation.y = st.facing;
    if (this.firstPerson !== this._fp) { // only on a camera change: no scale tracks touch it in between
      this._fp = this.firstPerson;
      if (this.head) this.head.scale.copy(this.headRest).multiplyScalar(this._fp ? 1e-3 : 1);
      else this.view.visible = !this._fp;
    }
  }
}

// ------------------------------------------------------------------ boot
const nextFrame = () => new Promise((r) => setTimeout(r, 0)); // yield to the loading screen

async function boot() {
  const t0 = performance.now();
  applyTheme();
  const loading = new LoadingScreen({ title: 'CHARACTER', label: 'Laying out the playground…', accent: ACCENT, background: '#0e1116' });
  const physics = new PhysicsWorld({ cellSize: 8 });
  const engine = new Engine({
    container: document.getElementById('app'), fixedHz: 120, maxSubsteps: 4, renderer: { antialias: NOFX }, fov: 62,
    physics, profile: params.get('profile') ? +params.get('profile') : false, targetFps: +params.get('fps') || 0,
  });
  const { scene, camera, renderer } = engine;
  const t = params.get('t');
  const atmo = new Atmosphere({
    scene, renderer, camera, timeOfDay: t == null ? 'afternoon' : Number.isFinite(+t) ? +t : t,
    shadows: { cascades: 2, near: 24, far: [120, 240, 420], kernel: params.get('shadows') || 'fast' }, maxCasterHeight: 40,
  });

  // ---------------------------------------------------------------- load: playground, then the model (optional)
  const W = {};
  engine.addSystem({
    async load(progress) {
      progress(0.2, 'Laying out the playground…');
      await nextFrame();
      W.playground = buildPlayground(scene, renderer, physics);
      progress(1, 'Ready');
    },
  }, 'world-load');
  const assets = new Assets({ base: PAGE_DIR });
  registerRenderLoaders(assets, renderer, { ktx2: true, meshopt: true });
  loading.setProgress(0.02, 'Looking for a character model…');
  const modelFound = MODEL_URL ? await probe(new URL(MODEL_URL, PAGE_DIR).href) : false;
  if (modelFound) {
    assets.add('character', { type: 'gltf', url: MODEL_URL, weight: 3 });
    engine.addSystem(assets.system({ loadWeight: 3 }), 'assets');
  } else if (MODEL_URL) console.info(`[character] no model at ${MODEL_URL} (node tools/fetch-models.mjs downloads RobotExpressive): procedural rig`);
  try {
    await engine.load((p, label) => loading.setProgress(p, label === 'character' ? 'Loading the character…' : label));
  } catch (e) {
    if (!e.key) throw e; // only asset (model) failures fall back; a world-load error must surface
    console.warn('[character] model failed to load, using the procedural rig', e);
  }

  // ---------------------------------------------------------------- character: glTF or procedural rig
  let char = null, source = 'procedural rig';
  const gltf = modelFound ? assets.peek('character') : undefined;
  if (gltf) {
    try { char = mapGltfCharacter(gltf, { height: HEIGHT }); source = MODEL_URL.split('/').pop(); }
    catch (e) { console.warn('[character] glTF not usable as a character, using the procedural rig', e); }
  }
  if (!char) char = buildProceduralCharacter({ height: HEIGHT, accent: ACCENT });
  const ctl = new AnimationController(char.root, char.clips, {
    states: { idle: { clip: 'Idle' }, walk: { clip: 'Walking' }, run: { clip: 'Running' }, jump: { clip: 'Jump', loop: false, clamp: true } },
    trees: { locomotion: { type: '1d', param: 'speed', points: [{ at: 0, state: 'idle' }, { at: 2, state: 'walk' }, { at: 5.5, state: 'run' }] } },
    transitions: { default: 0.15 },
    rootMotion: false,
  });
  ctl.play('locomotion');

  const cc = new CharacterController({ radius: 0.35, height: HEIGHT, walk: 2, run: 5.5, jumpSpeed: 5 });
  cc.teleport(SPAWN[0], SPAWN[1], SPAWN[2]);
  const events = new Events();
  const intent = { moveX: 0, moveZ: 0, jump: false, sprint: false };      // written by the input system
  const state = { facing: 0, speed: 0, grounded: true, velocity: cc.velocity }; // read by the camera rigs
  const player = new Entity('player');
  const view = new THREE.Group(); view.name = 'playerView';
  view.rotation.y = Math.PI; // models face +Z, the entity's forward is -Z
  view.add(char.root);
  player.add(view);
  player.position.copy(cc.position);
  scene.add(player);
  const motor = player.addComponent(new Motor(cc, physics, intent));
  const animator = player.addComponent(new Animator(ctl, cc, motor, events));
  const facing = player.addComponent(new Facing(cc, intent, state, view));
  facing.setHead(char.headBone);

  // ---------------------------------------------------------------- HUD, input, camera rigs
  const hud = new HUD();
  hud.setModel(`model · ${source}`);
  const canvas = renderer.domElement;
  const input = new CharacterInput({ element: document });
  hud.onHelp = (on) => { input.map.enabled = !on; }; // help card up: no movement (toggleCamera / menu are alwaysOn)
  const third = new ThirdPersonRig(camera, canvas, { active: false, physics, distance: 4, height: 1.6, shoulder: 0.4 });
  const first = new FirstPersonRig(camera, canvas, { active: false, physics, eyeHeight: 1.65, bob: 0.03 });
  const orbit = new OrbitRig(camera, canvas, { active: false, target: player, distance: 7, elevation: 0.32, azimuth: 0, offsetY: 1.0, minDistance: 1.5, maxDistance: 60, minElevation: 0.02, physics });
  third.setTarget(player, () => state);
  first.setTarget(player, () => state);
  facing.fpRig = first;
  const rigs = { third, first, orbit }, ORDER = ['third', 'first', 'orbit'];
  let rig = null, rigName = '';
  const yawOf = (r) => (r === orbit ? r.azimuth : r.yaw); // same convention: the camera sits at -forward(yaw)
  const setRig = (name) => {
    const prev = rig;
    for (const k in rigs) if (k !== name) rigs[k].deactivate();
    rig = rigs[name]; rigName = name;
    if (prev && prev !== rig) { const y = yawOf(prev); if (rig === orbit) orbit.azimuth = y; else rig.yaw = y; } // keep the view direction
    rig.activate(); rig.snap();
    facing.firstPerson = name === 'first';
    if (name === 'orbit' && lock.locked) lock.exit(); // orbit drags with the pointer
    hud.setCamera(name);
  };
  let unlockedAt = -1e9;
  const lock = new PointerLock({
    element: canvas,
    onChange(locked) {
      hud.setLocked(locked);
      if (!locked) unlockedAt = performance.now();
      if (locked) hud.showHelp(false);
      else if (lock.releasedByUser) hud.showHelp(true); // Esc: the browser swallows the key, so pause from here
    },
  });
  canvas.addEventListener('click', () => { if (rig !== orbit && !lock.locked) lock.request(); });
  setRig('third');

  const respawn = () => {
    motor.reset(SPAWN[0], SPAWN[1], SPAWN[2]);
    state.facing = 0; player.rotation.y = 0;
    if (rig.recentre) rig.recentre();
    rig.snap();
    hud.flash.show('RESPAWN');
  };
  window.addEventListener('keydown', (e) => {
    if (e.repeat || e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.code === 'KeyH') hud.toggleHelp();
    else if (e.code === 'KeyR') respawn();
    else if (e.code.startsWith('Digit')) {
      const k = +e.code.slice(5) - 1;
      if (k >= 0 && k < PRESET_KEYS.length) { atmo.setTimeOfDay(PRESET_KEYS[k]); hud.flash.show(PRESET_KEYS[k]); }
    }
  });
  events.on('land', (v) => { if (v > 7.5) hud.flash.show('HARD LANDING'); });

  // ---------------------------------------------------------------- post-processing
  let postfx = null;
  if (!NOFX) {
    try {
      const { width, height } = engine.size;
      postfx = new PostFX(renderer, scene, camera, { atmosphere: atmo, width, height, pixelRatio: engine.pixelRatio, ao: { radius: 0.9, thickness: 0.9, samples: 8 }, msaaMaxPixels: MSAA_MAX });
    } catch (e) { console.warn('[character] post-processing unavailable', e); atmo.setFogMode('scene'); }
  }
  engine.renderFn = (sc, cam, dt) => {
    if (postfx) postfx.render(dt);
    else { renderer.render(sc, cam); atmo.renderFlare(renderer, cam); }
  };
  engine.onResize = (w, h, pr) => { if (postfx) postfx.setSize(w, h, pr); };

  // ---------------------------------------------------------------- systems (phase order = registration order)
  const _f = new THREE.Vector3(), _r = new THREE.Vector3();
  engine.addSystem({
    earlyUpdate(dt) {
      input.update(dt);
      const c = input.controls;
      if (c.toggleCamera) setRig(ORDER[(ORDER.indexOf(rigName) + 1) % ORDER.length]);
      if (c.menu && performance.now() - unlockedAt > 300) hud.toggleHelp(); // not the Esc that just released the lock
      // look: the controls are radians (+x right, +y down; sensitivity / invertY already applied), turn() takes
      // radians with + = left / up
      if (rig !== orbit && (c.lookX || c.lookY)) rig.turn(-c.lookX, -c.lookY);
      // camera-relative move (yaw-only basis; the orbit camera's from its view direction)
      if (rig === orbit) { camera.getWorldDirection(_f); _f.y = 0; if (_f.lengthSq() < 1e-8) _f.set(0, 0, -1); _f.normalize(); _r.set(-_f.z, 0, _f.x); }
      else { rig.getForward(_f); rig.getRight(_r); }
      let mx = _r.x * c.moveX + _f.x * c.moveY, mz = _r.z * c.moveX + _f.z * c.moveY;
      const l2 = mx * mx + mz * mz;
      if (l2 > 1) { const k = 1 / Math.sqrt(l2); mx *= k; mz *= k; }
      intent.moveX = mx; intent.moveZ = mz; intent.sprint = c.sprint;
      if (c.jump) intent.jump = true; // latched until a fixed step consumes it (frames with no step keep it)
    },
  }, 'input');
  const entities = engine.addSystem(new EntitySystem(), 'entities');
  entities.add(player);
  engine.addSystem({ update: (dt) => rig.update(dt) }, 'camera');
  engine.addSystem({
    lateUpdate(dt) {
      camera.updateMatrixWorld();
      atmo.update(camera.position, dt);
    },
  }, 'world');
  const info = { speed: 0, grounded: true, anim: 'IDLE', fps: 0 };
  engine.addSystem({
    lateUpdate(dt) {
      info.speed = cc.speed; info.grounded = cc.grounded; info.anim = animator.label; info.fps = engine.profiler.frame.fps;
      hud.update(dt, info);
    },
  }, 'hud');

  // ---------------------------------------------------------------- bench harness (window.__sample)
  const placePlayer = (x, z, heading, speed) => {
    motor.reset(x, 0, z);
    state.facing = heading; player.rotation.y = heading;
    animator.locomotion(); ctl.setParam('speed', speed);
    for (let i = 0; i < 60; i++) ctl.update(1 / 60); // settle the cross-fades
    ctl.mixer.setTime(0);                             // then a fixed clip phase
    player.updateMatrixWorld(true);
  };
  const harness = createBenchHarness(engine, {
    name: 'character',
    poses: {
      idle(cam) { placePlayer(0, 5, 0, 0); cam.position.set(2, 1.6, 1.6); cam.lookAt(0, 1, 5); },          // front three-quarter
      run(cam) { placePlayer(0, 0, 0.6, 5.5); cam.position.set(3.3, 1.5, -2.2); cam.lookAt(0, 0.95, 0); }, // side on
      aerial(cam) { placePlayer(0, 5, 0, 0); cam.position.set(-24, 22, 26); cam.lookAt(0, 0, -4); },
    },
    after: (cam) => atmo.update(cam.position, 0), // then quiet() pins the sky (it runs last)
    quiet() {
      if (postfx) postfx.autoExposure = false;
      const su = atmo.sky.material.uniforms; su.uTime.value = 0; su.uCloudOffset.value.set(0.13, 0.41); // pin the sky (stars, clouds)
    },
    settings: {
      'shadows.kernel': { get: () => atmo.shadowKernel, set: (v) => atmo.setShadowKernel(v) },
      'postfx.ao': { get: () => !!(postfx && postfx.ao.enabled), set: (v) => { if (postfx) postfx.ao.enabled = !!v; } },
    },
  });
  animator.frozen = () => harness.frozen;
  const bp = readBenchParams(location.search);
  if (bp.enabled && bp.size) harness.resize(bp.size.w, bp.size.h);

  renderer.compile(scene, camera);
  const loadMs = Math.round(performance.now() - t0);
  window.__character = {
    THREE, engine, cc, ctl, get rig() { return rig; }, rigs, setRig, player, motor, animator, facing, state, intent, events,
    physics, atmo, postfx, input, lock, hud, assets, harness, char, source, respawn, playground: W.playground, loadMs,
  };
  console.info(`[character] ready in ${loadMs} ms · ${source} · ${char.clips.length} clips · ${physics.staticBoxes.length} colliders`);
  engine.start();
  loading.hide(400);
  harness.setReady();
}

boot().catch((err) => {
  console.error(err);
  const l = document.querySelector('.ui-loading small');
  if (l) l.textContent = `Failed to start: ${err.message}`;
});
