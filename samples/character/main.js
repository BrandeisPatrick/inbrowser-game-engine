// Character — a third-person playground built only on the engine (src/**) and three.js.
// Wiring: Engine (120 Hz fixed step, render interpolation, adaptive resolution, F3 profiler) + PhysicsWorld (flat
// ground + static boxes: walls, a low-ceiling corridor, a staircase of 0.3 m steps, a ramp onto a 2 m platform, crates)
// + CharacterController (kinematic capsule, fixed step) + AnimationController (idle / walk / run 1D blend tree on the
// horizontal speed + one-shot jump, per frame) on RobotExpressive.glb (fetched by tools/fetch-models.mjs, loaded through
// Assets) or the procedural rig of rig.js + ThirdPersonRig / FirstPersonRig / OrbitRig (V cycles) + CharacterInput +
// PointerLock + Atmosphere (keys 1-8) + PostFX (?nofx=1 to disable) + the UI kit HUD. The player is an Entity with
// three components: Motor (controller, fixed step + interpolation), Animator (state machine), Facing (turn + rig state).
// 0.9.0 dynamics (crates.js): an in-thread XPBD world with pushable crates, a 400 kg crate and a kinematic elevator
// platform, stepped by the Motor right after the controller (cc.attachDynamics: push, ride); a Ragdoll of the
// character (K, or a landing faster than 9 m/s; it gets up once resting, after 1.5-4 s); a mantle onto ledges 0.6-1.4 m
// above the feet (jump while moving into one; grabbed without a press when airborne, blocked, moving into a ledge
// 0.05-0.6 m above the feet at or past the jump apex) with both hands IK'd onto the edge.
// Query: ?model=<url> (any rigged glTF; relative to this page) | ?model=procedural, ?t=<preset|hours>, ?nofx=1,
// ?fps=60, ?profile=N, ?shadows=pcf, ?gbuffer=1, ?aa=, ?cascades=, 0.8.0: ?ssr=1, ?volumetric=1, ?motionblur=<s>, ?dof=1,
// ?lut=warm|bleach, ?aonormals=gbuffer (samples/shared/fx-flags.js); ?bench=1&size=WxH (window.__sample: poses idle / run /
// aerial (the dynamics props hidden: those captures predate them), push, ragdoll).
// window.__character = { engine, cc, ctl, rig (the active one), props, ... } for the console.
import * as THREE from 'three';
import { Engine, Assets, Entity, EntitySystem, Events, createBenchHarness, readBenchParams } from '@vibe/engine/core';
import { PhysicsWorld, CharacterController, DYN_RECORD, BODY } from '@vibe/engine/physics';
import { Atmosphere, PostFX, TIME_PRESETS, registerRenderLoaders, InstancedPool, makeCanvas, finishTexture } from '@vibe/engine/render';
import { AnimationController, TwoBoneIK, Ragdoll } from '@vibe/engine/anim';
import { CharacterInput, PointerLock, CHARACTER_ACTIONS } from '@vibe/engine/input';
import { ThirdPersonRig, FirstPersonRig, OrbitRig } from '@vibe/engine/camera';
import { LoadingScreen } from '@vibe/engine/ui';
import { buildProceduralCharacter, mapGltfCharacter, captureRestPose } from './rig.js';
import { PlaygroundDynamics, CRATES } from './crates.js';
import { HUD, ACCENT, applyTheme } from './hud.js';
import { readFxFlags, aoFlags, loadLut, fxSettings } from '../shared/fx-flags.js';

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
const GBUFFER = params.get('gbuffer') === '1';
// mantle: `lo`..`hi` = ledge top above the feet for a jump press (grounded, moving into it); airLo..airHi when it is
// grabbed without one (airborne, blocked, moving into it, vertical speed <= airVy: at or past the jump apex). Two
// phases: the pull (hands planted, the Jump clip's first frame (a crouch), the feet rise to `grip` below the top), then
// the vault (the clip scrubbed to `phase` (the procedural tuck: knees up), up to the top + lift by 80 % of the time,
// over the edge from 62 %; the hands let go as the shoulders leave their reach)
const MANTLE = {
  time: 0.6, lo: 0.6, hi: 1.4, airLo: 0.05, airHi: 0.6, airVy: 1,
  reach: 0.06,      // m between the capsule and the ledge's face at most
  lift: 0.02,       // m above the ledge top the feet end at
  grip: 0.75,       // m below the top the pull lifts the feet to (the procedural rig's 0.6 m arms reach the edge from its crouch)
  pull: 0.4,        // share of `time` spent pulling
  hands: 0.22,      // m each hand beside the character's line, on the top 6 cm in from the edge ...
  palm: 0.05,       // ... and this far above it (the end bone sits at the hand's centre)
  phase: 0.35,      // Jump clip phase the vault ends on
};
const smooth01 = (x) => (x <= 0 ? 0 : x >= 1 ? 1 : x * x * (3 - 2 * x));
const RAGDOLL = { speed: 9, min: 1.5, max: 4, rest: 0.25 }; // landing m/s; get up after min s once resting (< rest m/s), or after max s
const RESTING = { speed: RAGDOLL.rest };                     // rd.isResting options (no object per step)
const RAGDOLL_OPTS = { bones: 'auto', friction: 0.7 };
const PUSH_HOLD = 0.25; // s the PUSH badge stays up after the last push
const wrapPi = (a) => { while (a > Math.PI) a -= 2 * Math.PI; while (a < -Math.PI) a += 2 * Math.PI; return a; };
const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);

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
const _lv = new THREE.Vector3();
/**
 * The ledge for a mantle: the box (static, or a dynamics record of a dynamic body of `dyn` at rest: the targets are
 * fixed when the mantle starts, so kinematic or moving bodies (the elevator, a crate being pushed) are skipped) whose
 * face is within MANTLE.reach of the capsule at feet (x, F, z), radius r, in direction (dx, dz) (unit; the closest point
 * within 45 deg of it), whose top is lo..hi above the feet and which reaches below F + 0.6 (a wall, not an overhang);
 * the nearest such face wins. Fills L (top, face normal nx / nz toward the character, heading facing the wall, target
 * feet x1 / y1 / z1 on the top, inset min(r + 0.05, half the box's depth), hand targets hL / hR) and returns it, or
 * null. cols: reused query array.
 */
function findLedge(physics, dyn, x, F, z, r, dx, dz, lo, hi, cols, L) {
  const list = physics.queryColliders(x, z, r + MANTLE.reach + 0.05, cols);
  const lim = r + MANTLE.reach;
  let best = null, bestD = lim;
  for (let i = 0; i < list.length; i++) {
    const b = list[i], rise = b.maxY - F;
    if (rise < lo || rise > hi || b.minY > F + 0.6) continue;
    if (b[DYN_RECORD] && (!dyn || dyn.bodies.type[b.id] !== BODY.DYNAMIC || !dyn.getVelocity(b.id, _lv, null) || _lv.lengthSq() > 0.0025)) continue;
    const ex = clamp(x, b.minX, b.maxX) - x, ez = clamp(z, b.minZ, b.maxZ) - z, d2 = ex * ex + ez * ez;
    if (d2 < 1e-12 || d2 > bestD * bestD) continue;   // over it (no face), or not the nearest
    const d = Math.sqrt(d2);
    if (ex * dx + ez * dz < 0.7071 * d) continue;      // not ahead
    best = b; bestD = d;
  }
  if (!best) return null;
  const b = best, cx = clamp(x, b.minX, b.maxX), cz = clamp(z, b.minZ, b.maxZ);
  const nx = (x - cx) / bestD, nz = (z - cz) / bestD;
  const depth = (nx < 0 ? -nx : nx) >= (nz < 0 ? -nz : nz) ? b.maxX - b.minX : b.maxZ - b.minZ;
  const inset = Math.min(r + 0.05, depth / 2), m = 0.03;
  L.top = b.maxY; L.nx = nx; L.nz = nz; L.heading = Math.atan2(nx, nz);
  L.x1 = clamp(cx - nx * inset, b.minX, b.maxX); L.y1 = b.maxY + MANTLE.lift; L.z1 = clamp(cz - nz * inset, b.minZ, b.maxZ);
  // hands: on the top 6 cm in from the edge, +-MANTLE.hands along it (the character's left is (-nz, nx): it faces -n)
  const hx = cx - nx * 0.06, hz = cz - nz * 0.06, hy = b.maxY + MANTLE.palm, lx = -nz * MANTLE.hands, lz = nx * MANTLE.hands;
  L.hL.set(clamp(hx + lx, b.minX + m, b.maxX - m), hy, clamp(hz + lz, b.minZ + m, b.maxZ - m));
  L.hR.set(clamp(hx - lx, b.minX + m, b.maxX - m), hy, clamp(hz - lz, b.minZ + m, b.maxZ - m));
  return L;
}

const _rp = new THREE.Vector3();

// The AnimationMixer writes a channel only when its value differs from what the mixer wrote last (PropertyMixer.apply
// compares its two accumulation buffers), so a bone written behind its back (the mantle IK, the rest pose restored
// after a ragdoll) keeps that value for as long as the clip's value holds still (procedural Idle: elbows, knees, feet).
/** The mixer's bindings (PropertyMixer) whose target is one of `bones`. */
function bindingsOf(mixer, bones) { return mixer._bindings.filter((b) => b.binding && bones.includes(b.binding.node)); }
/** Reset the change detection of these bindings to the bones' current values (accumulation buffers 1 and 2 of the
 *  PropertyMixer layout [incoming | accu0 | accu1 | orig | ...]; orig stays the rest pose): the next update rewrites
 *  them. */
function refreshBindings(list) {
  for (let i = 0; i < list.length; i++) { const b = list[i], n = b.valueSize; b.binding.getValue(b.buffer, n); b.binding.getValue(b.buffer, 2 * n); }
}

/**
 * Fixed step: the camera-relative intent -> CharacterController, then the playground's dynamics (props.step, after the
 * controller: its pushes act in the same step, see crates.js); per frame: the interpolated pose on the entity. Modes:
 * 'move' (the controller), 'mantle' (the controller is not stepped; the entity follows a curve onto the ledge, the
 * controller is teleported there at the end), 'ragdoll' (not stepped either; each step the entity moves by the hips'
 * displacement since activation (so bones the ragdoll does not map, e.g. a rig root above the hips, never jump), the
 * controller is teleported with it (the camera follows), and only then rd.update() poses the bones against the
 * entity's transform; the entity does not move again until the next step).
 */
class Motor {
  constructor(cc, physics, intent) {
    this.name = 'Motor';
    this.cc = cc; this.physics = physics; this.intent = intent;
    this.prev = new THREE.Vector3().copy(cc.position);  // entity pose at the previous fixed step ...
    this.cur = new THREE.Vector3().copy(cc.position);   // ... and at this one (the feet, the mantle curve or the hips)
    this.jumped = false; this.landed = false; this.landingSpeed = 0;   // latched for the per-frame components
    this.yOff = 0; // visual offset that eases out step-up pops (stairs, ramp slices)
    this.props = null;     // PlaygroundDynamics (crates.js), stepped after the controller
    this.rd = null;        // Ragdoll on props.dyn (rebuilt with it)
    this.rest = null;      // () => void: bones back to the rest pose + mixer resync (when a ragdoll ends)
    this.mode = 'move';    // 'move' | 'mantle' | 'ragdoll'
    this.modeT = 0;        // s into the current mantle / ragdoll
    this.pushT = Infinity; // s since the controller last pushed a body (HUD)
    this.rdBase = new THREE.Vector3(); // ragdoll: entity position minus hips position at activation
    this._rv = new THREE.Vector3();    // ragdoll: initial velocity
    this.stayDown = false;             // no get-up (the bench's hand-stepped ragdoll pose)
    this.ledge = { top: 0, nx: 0, nz: 0, heading: 0, x0: 0, y0: 0, z0: 0, ya: 0, x1: 0, y1: 0, z1: 0, hL: new THREE.Vector3(), hR: new THREE.Vector3() };
    this._cols = [];
  }
  reset(x, y, z) {
    this.stop();
    this.cc.teleport(x, y, z); this.prev.copy(this.cc.position); this.cur.copy(this.cc.position);
    this.intent.jump = false; this.jumped = false; this.landed = false; this.yOff = 0; this.pushT = Infinity;
    if (this.entity) { this.entity.position.copy(this.cc.position); this.entity.updateMatrixWorld(); }
  }
  /** Leave a mantle / ragdoll at once (respawn, bench poses): bodies removed, bones back to rest, controller free. */
  stop() {
    if (this.rd && this.rd.active) { this.rd.deactivate(); if (this.rest) this.rest(); }
    this.mode = 'move'; this.modeT = 0;
  }
  fixedUpdate(dt) {
    const cc = this.cc, I = this.intent;
    this.prev.copy(this.cur);
    if (this.mode === 'move') {
      if (I.jump && cc.grounded && this._mantle(false)) I.jump = false; // jump at a ledge: climb it instead
      else {
        cc.step(dt, I.moveX, I.moveZ, I.jump, I.sprint, this.physics);
        I.jump = false; // the press latched by the input system is consumed by the first step after it
        if (cc.landedThisStep) { this.landed = true; this.landingSpeed = cc.landingSpeed; }
        if (cc.jumpedThisStep) this.jumped = true;
        if (cc.steppedUp) this.yOff -= cc.position.y - this.prev.y;
        this.pushT = cc.pushedBody >= 0 ? 0 : this.pushT + dt;
        if (!cc.grounded && cc.blocked && cc.velocity.y <= MANTLE.airVy) this._mantle(true); // grab a ledge near the apex
      }
    } else I.jump = false; // no jump buffered through a mantle / ragdoll
    if (this.props) this.props.step(dt);
    if (this.mode === 'mantle') this._mantleStep(dt);
    else if (this.mode === 'ragdoll') this._ragdollStep(dt);
    else this.cur.copy(cc.position);
  }
  update(dt, alpha) {
    if (this.mode === 'ragdoll') return; // placed in the fixed step: the bones were posed against that transform
    this.yOff *= Math.exp(-20 * dt);
    this.entity.position.lerpVectors(this.prev, this.cur, alpha);
    this.entity.position.y += this.yOff;
  }

  // ---- mantle
  _mantle(air) {
    const cc = this.cc, I = this.intent, p = cc.position, L = this.ledge;
    const m2 = I.moveX * I.moveX + I.moveZ * I.moveZ;
    if (m2 < 0.09) return false;
    const k = 1 / Math.sqrt(m2);
    if (!findLedge(this.physics, this.props && this.props.dyn, p.x, p.y, p.z, cc.radius, I.moveX * k, I.moveZ * k, air ? MANTLE.airLo : MANTLE.lo, air ? MANTLE.airHi : MANTLE.hi, this._cols, L)) return false;
    if (!this._fits(L.x1, L.y1, L.z1)) return false;
    L.x0 = p.x; L.y0 = p.y; L.z0 = p.z;
    L.ya = Math.max(p.y, L.top - MANTLE.grip); // feet height at the end of the pull
    this.mode = 'mantle'; this.modeT = 0; this.yOff = 0; // (the controller is not stepped until the mantle ends)
    if (this.entity) this.entity.rotation.y = L.heading; // face the wall now (Facing keeps it; the hand IK reads it)
    return true;
  }
  // pull: up to `ya` in place; vault: up to y1 (done at 80 %) and over the edge from 62 % (the tucked feet clear it by
  // then); smoothstepped. At the end the controller is teleported onto the ledge if the capsule still fits there (a
  // body may have moved onto it meanwhile), else back to where it started.
  _mantleStep(dt) {
    const L = this.ledge, P = MANTLE.pull;
    this.modeT += dt;
    const u = this.modeT < MANTLE.time ? this.modeT / MANTLE.time : 1;
    const y = u <= P ? L.y0 + (L.ya - L.y0) * smooth01(u / P) : L.ya + (L.y1 - L.ya) * smooth01((u - P) / (0.8 - P));
    const f = smooth01((u - 0.62) / 0.38);
    this.cur.set(L.x0 + (L.x1 - L.x0) * f, y, L.z0 + (L.z1 - L.z0) * f);
    if (u >= 1) {
      if (this._fits(L.x1, L.y1, L.z1)) this.cc.teleport(L.x1, L.y1, L.z1); else this.cc.teleport(L.x0, L.y0, L.z0);
      this.cur.copy(this.cc.position); this.mode = 'move';
      this.intent.jump = false; // a press during the mantle does not jump off the ledge
    }
  }
  /** Whether the capsule fits with its feet at (x, y, z) (cc.canStand there; the controller is put back). */
  _fits(x, y, z) {
    const cc = this.cc, p = cc.position, ox = p.x, oy = p.y, oz = p.z;
    p.set(x, y, z);
    const ok = cc.canStand(cc.height, this.physics);
    p.set(ox, oy, oz);
    return ok;
  }

  // ---- ragdoll
  /**
   * Knock the character down: bodies from the current bone pose (rd.activate) with the controller's velocity, or, given
   * `fall` (m/s, the landing speed: the controller already zeroed its vertical velocity on touchdown), with -fall as
   * the vertical part. The controller is not stepped until it gets up. False without a ragdoll.
   */
  startRagdoll(fall = 0) {
    if (this.mode === 'ragdoll' || !this.rd) return false;
    const v = this.cc.velocity;
    this.rd.activate({ velocity: this._rv.set(v.x, fall > 0 ? -fall : v.y, v.z) }); // (it refreshes the bones' world matrices itself)
    this.rdBase.copy(this.entity ? this.entity.position : this.cc.position).sub(this.rd.position(_rp));
    this.mode = 'ragdoll'; this.modeT = 0; this.yOff = 0;
    return true;
  }
  _ragdollStep(dt) {
    const rd = this.rd, cc = this.cc, e = this.entity;
    this.modeT += dt;
    this.cur.copy(this.rdBase).add(rd.position(_rp));
    cc.teleport(this.cur.x, this.cur.y, this.cur.z); // the controller (and so the camera target) follows the hips
    this.prev.copy(this.cur);
    if (e) e.position.copy(this.cur); // rd.update() reads the bones' parents fresh (updateWorldMatrix up the chain)
    rd.update();
    if (!this.stayDown && (this.modeT >= RAGDOLL.max || (this.modeT >= RAGDOLL.min && rd.isResting(RESTING)))) this.getUp();
  }
  /** End a ragdoll: bodies removed, bones back to rest + mixer resync (it re-poses every channel), feet on the support
   *  under the hips. */
  getUp() {
    if (this.mode !== 'ragdoll') return;
    const cc = this.cc;
    this.rd.position(_rp);
    this.rd.deactivate();
    if (this.rest) this.rest();
    const y = this.physics.supportAt(_rp.x, _rp.z, _rp.y, 0.05);
    cc.teleport(_rp.x, y, _rp.z);
    this.cur.copy(cc.position); this.prev.copy(this.cur);
    if (this.entity) this.entity.position.copy(this.cur);
    this.mode = 'move'; this.modeT = 0;
  }
}

/**
 * Animation state: locomotion tree driven by the horizontal speed; jump on take-off, back on landing / clip end; while
 * the Motor mantles, the 'mantle' state (the Jump clip at speed 0, scrubbed by the mantle's progress) with both hands
 * IK'd onto the ledge; while it ragdolls, nothing (the mixer is not updated: the bodies pose the bones), then locomotion.
 */
class Animator {
  constructor(ctl, cc, motor, events) {
    this.name = 'Animator';
    this.ctl = ctl; this.cc = cc; this.motor = motor; this.events = events;
    this.mode = 'locomotion'; this.label = 'IDLE';
    this.arms = [];            // [{ ik: TwoBoneIK, side: 1 left | -1 right, pole: Vector3 }] (mantle hand IK)
    this.armBindings = [];     // the mixer's bindings of the IK'd bones (refreshed after each solve: see refreshBindings)
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
    if (m.mode === 'ragdoll') { this.mode = 'ragdoll'; this.label = 'RAGDOLL'; return; } // (a 'land' listener may have started it)
    if (m.mode === 'mantle') { if (this.mode !== 'mantle') this._mantle(); this._scrub(); }
    else if (this.mode === 'mantle' || this.mode === 'ragdoll') this.locomotion();
    this.ctl.setParam('speed', cc.speed);
    this.ctl.update(dt);
    if (this.mode === 'mantle') this._hands();
    this.label = this.mode === 'mantle' ? 'MANTLE' : this.mode === 'jump' ? 'JUMP' : !cc.grounded ? 'FALL' : cc.speed < LOCO.idle ? 'IDLE' : cc.speed < LOCO.walk ? 'WALK' : 'RUN';
  }
  _mantle() {
    this.mode = 'mantle';
    this.ctl.play('mantle', { fade: 0.08 });
  }
  // the 'mantle' state's speed is 0: its time is set here, the pull's first frame, then on to MANTLE.phase in the vault
  _scrub() {
    const s = this.ctl.states.mantle;
    if (s) s.action.time = s.duration * MANTLE.phase * smooth01((this.motor.modeT / MANTLE.time - MANTLE.pull) / 0.45);
  }
  // both hands onto the ledge, elbows toward a pole behind, below and outside each shoulder. Weight: in over 0.1 s,
  // out over the last 0.15 s, and per arm down from 1 to 0 as the target goes from 1 to 1.25 arm lengths away from the
  // shoulder (the vault lifts the shoulders out of reach: the hands let go instead of the arms pointing at the edge)
  _hands() {
    const A = this.arms, m = this.motor, L = m.ledge, t = m.modeT;
    let w = t / 0.1; if (w > 1) w = 1;
    const tail = (MANTLE.time - t) / 0.15; if (tail < w) w = tail;
    if (!(w > 0)) return;
    const lx = -L.nz, lz = L.nx; // the character's left
    for (let i = 0; i < A.length; i++) {
      const a = A[i], ik = a.ik, P = a.pole, T = a.side > 0 ? L.hL : L.hR, len = ik.lengthA + ik.lengthB;
      ik.upper.updateWorldMatrix(true, false);
      P.setFromMatrixPosition(ik.upper.matrixWorld);
      const wr = clamp((1.25 * len - P.distanceTo(T)) / (0.25 * len), 0, 1);
      if (!(wr > 0)) continue;
      P.x += L.nx * 0.5 + lx * a.side * 0.25; P.y -= 0.3; P.z += L.nz * 0.5 + lz * a.side * 0.25;
      ik.solve(T, w * wr);
    }
    refreshBindings(this.armBindings); // next frame the mixer rewrites the arms from the clip (the IK's pre-solve pose)
  }
}

/**
 * Facing (engine heading: 0 = -Z, + = left) eases toward the move intent (third person / orbit) or the view yaw (first
 * person); the entity carries it (its local -Z is forward; the model inside is turned by PI, glTF faces +Z). Also fills
 * the rig state (one reused object) and hides the head in first person (its bone scaled after the animation pose).
 * Mantling it faces the wall; in a ragdoll it does not turn the entity (the bodies pose the bones against it).
 */
class Facing {
  constructor(cc, intent, state, view) {
    this.name = 'Facing';
    this.cc = cc; this.intent = intent; this.state = state; this.view = view;
    this.firstPerson = false; this.fpRig = null; this.head = null; this.motor = null;
    this.headRest = new THREE.Vector3(1, 1, 1); this._fp = false;
  }
  /** Head bone hidden in first person (null: the whole view is hidden instead); its rest scale is restored after. */
  setHead(bone) { this.head = bone || null; if (bone) this.headRest.copy(bone.scale); this._fp = false; }
  update(dt) {
    const st = this.state, I = this.intent, cc = this.cc, m = this.motor;
    st.speed = cc.speed; st.grounded = cc.grounded; // st.velocity is cc.velocity (same object)
    if (m && m.mode === 'ragdoll') return;
    if (m && m.mode === 'mantle') st.facing = m.ledge.heading;
    else {
      let goal = st.facing, rate = 10;
      if (this.firstPerson && this.fpRig) { goal = this.fpRig.yaw; rate = 30; }
      else if (I.moveX * I.moveX + I.moveZ * I.moveZ > 0.01) goal = Math.atan2(-I.moveX, -I.moveZ);
      st.facing = wrapPi(st.facing + wrapPi(goal - st.facing) * (1 - Math.exp(-rate * dt)));
    }
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
    shadows: { cascades: +params.get('cascades') || 2, near: 24, far: [120, 240, 420], kernel: params.get('shadows') || 'fast' }, maxCasterHeight: 40,
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
  const restPose = captureRestPose(char.root); // the bones as built: restored when a ragdoll ends (before the mixer re-poses them)
  const ctl = new AnimationController(char.root, char.clips, {
    // mantle: the Jump clip at speed 0, its time set by the mantle's progress (Animator._scrub)
    states: { idle: { clip: 'Idle' }, walk: { clip: 'Walking' }, run: { clip: 'Running' }, jump: { clip: 'Jump', loop: false, clamp: true }, mantle: { clip: 'Jump', loop: false, speed: 0 } },
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
  facing.motor = motor;
  // after a ragdoll: bones to the rest pose (channels no clip animates must not keep the ragdoll's values), then every
  // mixer binding re-reads them, so the next update rewrites every animated channel (see refreshBindings)
  motor.rest = () => {
    restPose.restore();
    if (typeof ctl.resync === 'function') ctl.resync();
    else { const B = ctl.mixer._bindings; for (let i = 0; i < B.length; i++) B[i].saveOriginalState(); }
  };
  player.updateMatrixWorld(true);
  // mantle hand IK: upper arm / forearm / hand (procedural handL / handR, RobotExpressive Palm2.L / .R), poles set per frame
  for (const [side, chain] of [[1, char.arms && char.arms.L], [-1, char.arms && char.arms.R]]) {
    if (!chain) continue;
    const pole = new THREE.Vector3();
    animator.arms.push({ side, pole, ik: new TwoBoneIK(chain.upper, chain.lower, chain.end, { pole }) });
  }
  animator.armBindings = bindingsOf(ctl.mixer, animator.arms.flatMap((a) => [a.ik.upper, a.ik.lower]));

  // ---------------------------------------------------------------- dynamics: crates, the elevator platform, the ragdoll
  const props = new PlaygroundDynamics({ scene, physics, velocity: GBUFFER });
  props.attach(cc); // pushMass 120 kg (the 400 kg crate is a wall), pushForce 600 N, rides 'rigid' bodies
  motor.props = props;
  // the ragdoll, built ONCE here: fromSkeleton reads the REST pose in world space (its joint frames come from it), and
  // the bones are still as built (no mixer update yet); the entity is put at the origin facing -Z meanwhile, so the
  // description does not depend on the spawn. props.reset() replaces the dynamics world: the ragdoll (inactive by then,
  // motor.stop()) is pointed at the new client (rd.dyn, read on activate), nothing is rebuilt or restored
  const buildRagdoll = (dyn) => {
    const px = player.position.x, py = player.position.y, pz = player.position.z, ry = player.rotation.y;
    player.position.set(0, 0, 0); player.rotation.y = 0;
    player.updateMatrixWorld(true);
    try {
      const rd = Ragdoll.fromSkeleton(char.root, dyn, RAGDOLL_OPTS), B = rd.bones, p = B.hips.parent;
      // pelvis = the bone the legs hang from when that is the auto hips' parent (RobotExpressive: Body > Hips, UpperLeg.L,
      // UpperLeg.R, and the torso mesh rides Body): otherwise that parent, unmapped, would stand still with the entity.
      // To check in the browser: the animated Hips (its clips bob it by translation) is then an unmapped link between
      // Body and Abdomen, so the robot's upper body keeps the Hips offset of the pose it went down in
      if (p && p.isBone && B.upperLegL && B.upperLegL.parent === p && B.upperLegR && B.upperLegR.parent === p) {
        return Ragdoll.fromSkeleton(char.root, dyn, { ...RAGDOLL_OPTS, bones: { ...B, hips: p } });
      }
      return rd;
    } catch (e) { console.warn('[character] no ragdoll for this model', e); return null; }
    finally { player.position.set(px, py, pz); player.rotation.y = ry; player.updateMatrixWorld(true); }
  };
  motor.rd = buildRagdoll(props.dyn);
  props.onRebuild = (dyn) => { if (motor.rd) motor.rd.dyn = dyn; };
  /** The start layout of the props (the ragdoll moves to the new world); leaves any mantle / ragdoll first. */
  const resetDynamics = () => { motor.stop(); props.reset(); };

  // ---------------------------------------------------------------- HUD, input, camera rigs
  const hud = new HUD();
  hud.setModel(`model · ${source}`);
  const canvas = renderer.domElement;
  const input = new CharacterInput({ element: document, actions: { ...CHARACTER_ACTIONS, ragdoll: { type: 'button', bindings: ['KeyK', { virtual: 'ragdoll' }] } } });
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
    resetDynamics(); // the crates go back too
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
  /** K: knock the character down, or get up early. */
  const toggleRagdoll = () => {
    if (motor.mode === 'ragdoll') motor.getUp();
    else if (motor.startRagdoll()) hud.flash.show('RAGDOLL');
  };
  events.on('land', (v) => {
    if (v > RAGDOLL.speed && motor.startRagdoll(v)) hud.flash.show('KNOCKED DOWN'); // the bodies keep the fall speed
    else if (v > 7.5) hud.flash.show('HARD LANDING');
  });

  // ---------------------------------------------------------------- post-processing
  let postfx = null;
  if (!NOFX) {
    try {
      const { width, height } = engine.size;
      // 0.8.0 knobs (samples/shared/fx-flags.js): ?ssr=1, ?volumetric=1, ?motionblur=<s>, ?dof=1, ?lut=<name>, ?aonormals=gbuffer
      const fx = readFxFlags(params), lut = await loadLut(params);
      postfx = new PostFX(renderer, scene, camera, { atmosphere: atmo, width, height, pixelRatio: engine.pixelRatio, ao: { radius: 0.9, thickness: 0.9, samples: 8, ...aoFlags(params) }, msaaMaxPixels: MSAA_MAX, gbuffer: params.get('gbuffer') === '1', aa: params.get('aa') || 'auto', ...fx, lut });
      player.traverse((o) => { if (o.isMesh) postfx.track(o); }); // the character moves: per-object velocity (replayed whenever the G-buffer is built)
      postfx.trackPool(props.pool); // the props: per-instance velocity (InstancedPool({ velocity }) when ?gbuffer=1)
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
      if (input.map.justPressed('ragdoll')) toggleRagdoll();
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
  engine.addSystem({
    update(dt, alpha) {
      if (animator.frozen()) return; // a bench pose drew them already (and may have hidden them)
      if (!props.visible) props.setVisible(true);
      props.sync(alpha);
    },
  }, 'props');
  engine.addSystem({ update: (dt) => rig.update(dt) }, 'camera');
  engine.addSystem({
    lateUpdate(dt) {
      camera.updateMatrixWorld();
      atmo.update(camera.position, dt);
    },
  }, 'world');
  const info = { speed: 0, grounded: true, anim: 'IDLE', action: '', fps: 0 };
  engine.addSystem({
    lateUpdate(dt) {
      info.speed = cc.speed; info.grounded = cc.grounded; info.anim = animator.label; info.fps = engine.profiler.frame.fps;
      info.action = motor.mode !== 'move' ? '' : cc.rideBody >= 0 ? 'RIDE' : motor.pushT < PUSH_HOLD ? 'PUSH' : '';
      hud.update(dt, info);
    },
  }, 'hud');

  // ---------------------------------------------------------------- bench harness (window.__sample)
  // idle / run / aerial hide the props (their captures predate them: same hashes as 0.8.0); push / ragdoll start from a
  // fresh dynamics world (resetDynamics) and hand-step a fixed number of fixed steps through motor.fixedUpdate (the
  // controller, the mantle / ragdoll logic and props.step), so every load and every repeat reaches the same state
  const FIXED = engine.fixedDt;
  const PUSH_STEPS = 120, DROP_HEIGHT = 3, DROP_STEPS = 330; // 1 s of pushing; 330 steps (2.75 s) after the drop
  const placePlayer = (x, z, heading, speed) => {
    motor.reset(x, 0, z);
    state.facing = heading; player.rotation.y = heading;
    animator.locomotion(); ctl.setParam('speed', speed);
    for (let i = 0; i < 60; i++) ctl.update(1 / 60); // settle the cross-fades
    ctl.mixer.setTime(0);                             // then a fixed clip phase
    player.updateMatrixWorld(true);
  };
  /** After hand-stepping: the entity on its fixed-step pose (no interpolation, no step-up offset), the props drawn. */
  const settleView = () => {
    motor.prev.copy(motor.cur); motor.yOff = 0;
    player.position.copy(motor.cur); player.updateMatrixWorld(true);
    props.sync(1);
  };
  const harness = createBenchHarness(engine, {
    name: 'character',
    poses: {
      idle(cam) { props.setVisible(false); placePlayer(0, 5, 0, 0); cam.position.set(2, 1.6, 1.6); cam.lookAt(0, 1, 5); },          // front three-quarter
      run(cam) { props.setVisible(false); placePlayer(0, 0, 0.6, 5.5); cam.position.set(3.3, 1.5, -2.2); cam.lookAt(0, 0.95, 0); }, // side on
      aerial(cam) { props.setVisible(false); placePlayer(0, 5, 0, 0); cam.position.set(-24, 22, 26); cam.lookAt(0, 0, -4); },
      // walking (-X, walk pose at clip time 0) into the 30 kg crate from 0.3 m off its face, PUSH_STEPS steps: it has slid
      push(cam) {
        resetDynamics(); props.setVisible(true);
        const c = CRATES[0];
        placePlayer(c.x + c.size / 2 + 0.3 + cc.radius, c.z, Math.PI / 2, cc.walk);
        intent.moveX = -1; intent.moveZ = 0; intent.sprint = false; intent.jump = false;
        for (let i = 0; i < PUSH_STEPS; i++) motor.fixedUpdate(FIXED);
        intent.moveX = 0;
        settleView();
        const p = cc.position;
        cam.position.set(p.x - 0.8, 1.4, p.z - 3.4); cam.lookAt(p.x - 0.75, 0.6, p.z); // from -Z: the crate ahead of it
      },
      // the ragdoll from the idle pose (clip time 0) with its feet 3 m over the 15 kg crate, DROP_STEPS steps after the
      // drop (on the ground, not yet resting: isResting first holds at 2.82 s, after a 0.72 m rebound off the crate; no
      // get-up while stepping, a resumed loop gets it up)
      ragdoll(cam) {
        resetDynamics(); props.setVisible(true);
        const c = CRATES[1];
        placePlayer(c.x, c.z, 0.5, 0);
        motor.reset(c.x, DROP_HEIGHT, c.z);
        player.updateMatrixWorld(true);
        motor.startRagdoll();
        motor.stayDown = true;
        for (let i = 0; i < DROP_STEPS; i++) motor.fixedUpdate(FIXED);
        motor.stayDown = false;
        settleView();
        cam.position.set(c.x + 3.2, 2.3, c.z + 3.6); cam.lookAt(c.x - 0.2, 0.45, c.z);
      },
    },
    after: (cam) => atmo.update(cam.position, 0), // then quiet() pins the sky (it runs last)
    quiet() {
      if (postfx) postfx.autoExposure = false;
      const su = atmo.sky.material.uniforms; su.uTime.value = 0; su.uCloudOffset.value.set(0.13, 0.41); // pin the sky (stars, clouds)
      if (postfx && postfx.history) { postfx.history.reset(); for (let i = 0; i < 32; i++) postfx.render(0); } // TAA: converge from a clean history
    },
    settings: {
      'shadows.kernel': { get: () => atmo.shadowKernel, set: (v) => atmo.setShadowKernel(v) },
      'postfx.aa': { get: () => (postfx ? postfx.aa : null), set: (v) => { if (postfx) postfx.set('aa', v); } },
      'postfx.gbuffer': { get: () => !!(postfx && postfx.gbuffer), set: (v) => { if (postfx) postfx.set('gbuffer', !!v); } },
      'postfx.ao': { get: () => !!(postfx && postfx.ao.enabled), set: (v) => { if (postfx) postfx.ao.enabled = !!v; } },
      ...fxSettings(() => postfx), // 0.8.0: postfx.ssr / volumetric / motionBlur / dof / lut.strength / ao.normals (+ their options)
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
    props, get rd() { return motor.rd; }, resetDynamics, toggleRagdoll, findLedge, MANTLE, RAGDOLL,
  };
  console.info(`[character] ready in ${loadMs} ms · ${source} · ${char.clips.length} clips · ${physics.staticBoxes.length} colliders · ${props.records.length} dynamics bodies${motor.rd ? ' · ragdoll' : ''}`);
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
