// Headless checks for the character input preset (CharacterInput + InputMap.readSources + PointerLock): no browser,
// no DOM element (element: null); pointer / key events are fed through the modules' own handlers with plain objects.
// Run: node test/character-input.mjs
import { CharacterInput } from '../src/input/CharacterInput.js';
import { InputMap } from '../src/input/InputMap.js';
import { PointerLock } from '../src/input/PointerLock.js';

const DT = 1 / 60;
let failures = 0;
const results = [];
function check(name, value, lo, hi = lo, unit = '') {
  const v = typeof value === 'boolean' ? +value : value;
  const ok = Number.isFinite(v) && v >= lo - 1e-12 && v <= hi + 1e-12;
  if (!ok) failures++;
  results.push(`${ok ? 'PASS' : 'FAIL'}  ${name.padEnd(56)} ${fmt(v)} ${unit}   [${lo} .. ${hi}]`);
  return ok;
}
const fmt = (x) => (Number.isFinite(x) ? (Math.abs(x) < 1e-3 && x !== 0 ? x.toExponential(2) : x.toFixed(4)).padStart(10) : String(x).padStart(10));
const keyDown = (m, code) => m._onKeyDown({ code, target: null, repeat: false, preventDefault() {} });
const mouseMove = (m, dx, dy) => m._onPointerMove({ movementX: dx, movementY: dy });
// a standard-mapping pad snapshot the map's GamepadState returns instead of navigator.getGamepads()
function fakePad(map) {
  const pad = { connected: true, id: 'test pad', axes: [0, 0, 0, 0], buttons: Array.from({ length: 17 }, () => ({ pressed: false, value: 0 })) };
  map.gamepad.poll = function () { this.pad = pad; this.connected = true; return pad; };
  return pad;
}

// --- edges: a virtual button press is one justPressed --------------------------------
{
  const input = new CharacterInput({ element: null });
  input.map.setVirtual('jump', 1);
  let edges = 0, heldAll = true;
  for (let i = 0; i < 10; i++) { input.update(DT); if (input.controls.jump) edges++; heldAll = heldAll && input.controls.jumpHeld; }
  check('virtual jump: controls.jump true for exactly 1 update', edges, 1);
  check('virtual jump: jumpHeld while the source stays 1', heldAll, 1);
  input.map.setVirtual('jump', 0); input.update(DT);
  check('virtual jump released: jumpHeld false', input.controls.jumpHeld, 0);
  check('update() returns this, controls object reused', input.update(DT) === input && input.controls === input.controls, 1);
  input.dispose();
}

// --- mouse look gating: deltas only while pointer-locked or right button held ---------
{
  const input = new CharacterInput({ element: null }), m = input.map, c = input.controls;
  let sum = 0;
  for (let i = 0; i < 5; i++) { mouseMove(m, 30, -20); input.update(DT); sum += Math.abs(c.lookX) + Math.abs(c.lookY); }
  check('mouse moving, no lock / button: look stays 0', sum, 0);
  check('lookActive false while unlocked', input.lookActive, 0);

  input._onPointer({ type: 'pointerdown', pointerType: 'mouse', buttons: 2, pointerId: 1 });   // right button down
  mouseMove(m, 30, -20); input.update(DT);
  check('right button: activation update drops its delta', Math.abs(c.lookX) + Math.abs(c.lookY), 0);
  check('right button: lookActive', input.lookActive, 1);
  mouseMove(m, 100, -50); input.update(DT);
  check('right button: lookX = px * mouseSpeed (raw)', c.lookX, 100 * input.mouseSpeed);
  check('right button: lookY = px * mouseSpeed (+y down)', c.lookY, -50 * input.mouseSpeed);
  input.update(DT);
  check('deltas consumed: next update without motion is 0', Math.abs(c.lookX) + Math.abs(c.lookY), 0);
  input._onPointer({ type: 'pointerup', pointerType: 'mouse', buttons: 0, pointerId: 1 });
  mouseMove(m, 100, 0); input.update(DT);
  check('right button released: look 0 again', c.lookX, 0);

  globalThis.document = { pointerLockElement: {} };   // something holds the pointer lock (element null = any lock)
  check('PointerLock.isLocked(null) with a lock on the page', PointerLock.isLocked(null), 1);
  mouseMove(m, 40, 0); input.update(DT);
  check('pointer lock: activation update drops its delta', c.lookX, 0);
  mouseMove(m, 40, 10); input.update(DT);
  check('pointer lock: lookX = px * mouseSpeed', c.lookX, 40 * input.mouseSpeed);
  check('pointer lock: pointerLocked + lookActive', input.pointerLocked && input.lookActive, 1);
  input.lookSensitivity = 2; input.invertY = true; mouseMove(m, 10, 10); input.update(DT);
  check('lookSensitivity scales, invertY flips y', c.lookX === 20 * input.mouseSpeed && c.lookY === -20 * input.mouseSpeed, 1);
  delete globalThis.document;
  input.dispose();
}

// --- virtual look sources are per-update pixel deltas (consumed) ---------------------
{
  const input = new CharacterInput({ element: null }), c = input.controls;
  input.map.setVirtual('look.x', 40);
  input.update(DT);
  check('virtual look.x 40 px: lookX = 40 * touchSpeed', c.lookX, 40 * input.touchSpeed);
  input.update(DT);
  check('virtual look.x consumed: 0 on the next update', c.lookX, 0);
  check('virtual look.x zeroed in the map', input.map.virtual.get('look.x'), 0);
  input.dispose();
}

// --- stick look: ramps in, settles at stickSpeed * dt ----------------------------------
{
  const input = new CharacterInput({ element: null }), c = input.controls;
  const pad = fakePad(input.map);
  pad.axes[2] = 1; pad.axes[1] = -1;   // right stick full right, left stick full up
  input.update(DT);
  check('stick look ramps in (first update below full rate)', c.lookX, 1e-9, input.stickSpeed * DT * 0.99, 'rad');
  for (let i = 0; i < 30; i++) input.update(DT);
  check('stick full deflection settles at stickSpeed * dt', c.lookX, input.stickSpeed * DT, input.stickSpeed * DT, 'rad');
  check('left stick up = moveY +1 (forward)', c.moveY, 1);
  check('gamepadConnected', input.gamepadConnected, 1);
  pad.axes[2] = 0;
  for (let i = 0; i < 30; i++) input.update(DT);
  check('stick released: look recentres to 0', c.lookX, 0);
  input.dispose();
}

// --- move: keyboard ramp + unit-circle clamp --------------------------------------------
{
  const input = new CharacterInput({ element: null }), c = input.controls;
  keyDown(input.map, 'KeyW'); keyDown(input.map, 'KeyD');
  input.update(DT);
  check('W+D first update: ramping (below full)', c.moveY, 1e-9, 0.99);
  for (let i = 0; i < 30; i++) input.update(DT);
  check('W+D settled: |move| clamped to 1', Math.hypot(c.moveX, c.moveY), 1);
  check('W+D settled: moveX = moveY = 1/sqrt2', c.moveX, Math.SQRT1_2);
  input.dispose();
}

// --- enabled = false: alwaysOn menu / camera still fire, the rest is silenced ----------
{
  const input = new CharacterInput({ element: null }), c = input.controls;
  input.enabled = false;
  check('enabled passthrough to the map', input.map.enabled, 0);
  keyDown(input.map, 'KeyW'); input.map.setVirtual('look.x', 50); input.map.setVirtual('jump', 1);
  input.map.setVirtual('menu', 1); input.map.setVirtual('cam', 1);
  input.update(DT);
  check('disabled: menu (alwaysOn) still fires', c.menu, 1);
  check('disabled: toggleCamera (alwaysOn) still fires', c.toggleCamera, 1);
  check('disabled: move / look / jump silenced', Math.abs(c.moveY) + Math.abs(c.lookX) + (c.jump ? 1 : 0), 0);
  input.dispose();
}

// --- InputMap.readSources: per-kind raw sums match the hand-built bindings ------------
{
  const m = new InputMap();
  const pad = fakePad(m);
  m.define('probe', { type: 'axis', bindings: [
    { key: 'KeyD', value: 1 }, { key: 'ArrowRight', value: 0.5 }, { key: 'KeyA', value: -1 },
    { mouse: 'x', value: 2 }, { mouse: 'y' },
    { virtual: 'v1' }, { virtual: 'v2', value: -3 },
    { padAxis: 0 }, { padButton: 0, value: 0.5 },
    { key: 'KeyQ', value: -0.25, when: 'ctx' }, { virtual: 'v3', alwaysOn: true },
  ] });
  keyDown(m, 'KeyD'); keyDown(m, 'ArrowRight'); keyDown(m, 'KeyA'); keyDown(m, 'KeyQ');
  mouseMove(m, 250, 7);
  m.setVirtual('v1', 0.75); m.setVirtual('v2', 2); m.setVirtual('v3', 4);
  pad.axes[0] = 1; pad.buttons[0] = { pressed: true, value: 1 };
  m.update(DT);
  const out = { mouse: 0, virtual: 0, pad: 0, keyPos: 0, keyNeg: 0 };
  check('readSources returns the passed object', m.readSources('probe', out) === out, 1);
  check('readSources mouse = 250*2 + 7 (unclamped)', out.mouse, 507);
  check('readSources virtual = 0.75 - 3*2 + 4', out.virtual, 0.75 - 6 + 4);
  check('readSources pad = axis 1 + button 0.5', out.pad, 1.5);
  check('readSources keyPos = strongest + key (OR, not 1.5)', out.keyPos, 1);
  check('readSources keyNeg = -1 (KeyQ inactive w/o ctx)', out.keyNeg, -1);
  m.setVirtual('v1', 0); m.setVirtual('v2', 0);
  pad.axes[0] = 0; pad.buttons[0] = { pressed: false, value: 0 };
  m._keys.delete('KeyA'); m._keys.delete('KeyD'); m._keys.delete('ArrowRight');
  m.setContext('ctx', true); m.update(DT); m.readSources('probe', out);
  check('readSources honours contexts (KeyQ when ctx)', out.keyNeg, -0.25);
  check('readSources mouse reset next frame', out.mouse, 0);
  m.enabled = false; m.update(DT); m.readSources('probe', out);
  check('readSources disabled: only the alwaysOn binding (v3 = 4)', out.virtual, 4);
  check('readSources disabled: keys silenced', out.keyNeg, 0);
  m.readSources('missing', out);
  check('readSources unknown action: all zero', out.mouse + out.virtual + out.pad + out.keyPos + out.keyNeg, 0);
  m.dispose();
}

// --- PointerLock without a DOM -----------------------------------------------------------
{
  const lock = new PointerLock();
  check('PointerLock in Node: supported false, not locked', !lock.supported && !lock.locked, 1);
  check('PointerLock.request() in Node resolves false', (await lock.request()) === false, 1);
  lock.dispose();
}

console.log(results.join('\n'));
if (failures) { console.log(`\n${failures} character-input check(s) FAILED`); process.exit(1); }
console.log('\nAll character-input checks passed.');
