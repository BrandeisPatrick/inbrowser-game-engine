// Headless checks for the core kernel: Events, Clock, Entity / EntitySystem, mathDet (no browser, no WebGL).
// Run: node test/core.mjs (imports the modules by path until the core index exports them).
import v8 from 'node:v8';
import * as THREE from 'three';
import { Events } from '../src/core/Events.js';
import { Clock } from '../src/core/Clock.js';
import { Entity, EntitySystem } from '../src/core/Entity.js';
import { mulberry32, hashU32, hashF64, hashF64Array, detAtan, detAtan2, detAsin, detAcos, detSqrt, clamp } from '../src/core/mathDet.js';
import { mulberry32 as noiseMulberry32 } from '../src/render/noise.js';

let failures = 0;
const results = [];
function check(name, value, lo = 1, hi = lo, unit = '') {
  if (typeof value === 'boolean') value = value ? 1 : 0;
  const ok = Number.isFinite(value) && value >= lo && value <= hi;
  if (!ok) failures++;
  results.push(`${ok ? 'PASS' : 'FAIL'}  ${name.padEnd(52)} ${fmt(value)} ${unit}   [${fmt(lo).trim()}${hi !== lo ? ' .. ' + fmt(hi).trim() : ''}]`);
  return ok;
}
const fmt = (x) => (!Number.isFinite(x) ? String(x) : Number.isInteger(x) ? String(x) : Math.abs(x) < 1e-3 ? x.toExponential(2) : x.toFixed(2)).padStart(10);

// Bytes allocated and GCs run while fn() executes (young-generation growth with a GC count: if no GC ran, the
// heap delta is what fn allocated).
performance.now(); process.memoryUsage(); // first use initializes lazily; keep that out of the measurements
function measureAlloc(fn) {
  const prof = new v8.GCProfiler();
  prof.start();
  const h0 = process.memoryUsage().heapUsed;
  const t0 = performance.now();
  fn();
  const ms = performance.now() - t0;
  const h1 = process.memoryUsage().heapUsed;
  const gcs = prof.stop().statistics.length;
  return { bytes: h1 - h0, gcs, ms };
}

// ================================================================================================= Events
{
  const ev = new Events();
  let log = '';
  const ctx = { tag: 'C' };
  ev.on('t', () => { log += 'a'; });
  ev.on('t', function () { log += this.tag; }, ctx);
  ev.on('t', () => { log += 'b'; });
  ev.emit('t');
  check('events: registration order + ctx (aCb)', log === 'aCb');
  let got = null;
  ev.on('args', (a, b, c, d) => { got = [a, b, c, d]; });
  ev.emit('args', 1, 'x', null, 4);
  check('events: 4 positional args', got[0] === 1 && got[1] === 'x' && got[2] === null && got[3] === 4);
  check('events: emit with no listener returns false', ev.emit('nobody', 1) === false);
  check('events: has()', ev.has('t') && !ev.has('nobody'));
  // once
  let n1 = 0;
  const f1 = ev.once('o', () => n1++);
  check('events: once returns fn', typeof f1 === 'function');
  ev.emit('o'); ev.emit('o'); ev.emit('o');
  check('events: once fires exactly once', n1, 1);
  check('events: once removed afterwards', !ev.has('o'));
  const f2 = ev.once('o2', () => n1++); ev.off('o2', f2); ev.emit('o2');
  check('events: off(type, fn) before a once fires', n1, 1);
  // nested emit of the same type from inside a once listener: still once
  let n2 = 0;
  ev.once('n', () => { n2++; ev.emit('n'); });
  ev.emit('n');
  check('events: once + nested emit of same type fires once', n2, 1);
  // off during emit: deferred (B still receives this emit, not the next)
  log = '';
  const B = () => { log += 'B'; };
  ev.on('d', () => { log += 'A'; ev.off('d', B); });
  ev.on('d', B);
  ev.on('d', () => { log += 'C'; });
  ev.emit('d');
  check('events: off during emit is deferred (ABC)', log === 'ABC');
  log = ''; ev.emit('d');
  check('events: removed listener gone next emit (AC)', log === 'AC');
  // self-removal during emit
  log = '';
  const S = () => { log += 'S'; ev.off('s', S); };
  ev.on('s', S); ev.on('s', () => { log += 'T'; });
  ev.emit('s'); ev.emit('s');
  check('events: self-removal during emit (STT)', log === 'STT');
  // add during emit: not called in the current emit
  log = '';
  let added = false;
  ev.on('a', () => { log += '1'; if (!added) { added = true; ev.on('a', () => { log += '2'; }); } });
  ev.emit('a');
  check('events: listener added during emit not called (1)', log === '1');
  log = ''; ev.emit('a');
  check('events: ...called on the next emit (12)', log === '12');
  // removal during an emit is visible to nested emits that start after it
  log = '';
  const X = () => { log += 'X'; };
  let depth = 0;
  ev.on('r', () => { log += 'R'; if (depth++ === 0) { ev.off('r', X); ev.emit('r'); } });
  ev.on('r', X);
  ev.emit('r');
  check('events: nested emit after removal skips it (RRX)', log === 'RRX');
  // off(type) during emit, clear()
  log = '';
  ev.on('all', () => { log += '1'; ev.off('all'); }); ev.on('all', () => { log += '2'; });
  ev.emit('all'); ev.emit('all');
  check('events: off(type) during emit, deferred (12)', log === '12');
  check('events: off(type) removed everything', !ev.has('all'));
  // a throwing listener leaves the emitter consistent
  const ev2 = new Events();
  let after = 0;
  ev2.on('e', () => { throw new Error('boom'); }); ev2.on('e', () => after++);
  try { ev2.emit('e'); } catch { /* expected */ }
  ev2.off('e'); ev2.on('e', () => after++); ev2.emit('e');
  check('events: consistent after a listener throws', after, 1);
  // shared prototype method, two contexts: off(type, fn, ctx) removes only that ctx
  class Hitter { constructor() { this.n = 0; } onHit() { this.n++; } }
  const h1 = new Hitter(), h2 = new Hitter();
  ev.on('hit', h1.onHit, h1); ev.on('hit', h2.onHit, h2);
  ev.emit('hit'); ev.off('hit', h1.onHit, h1); ev.emit('hit');
  check('events: off(type, fn, ctx) removes only that ctx', h1.n === 1 && h2.n === 2 && ev.has('hit'));
  ev.on('hit', h1.onHit, h1); ev.off('hit', Hitter.prototype.onHit); ev.emit('hit');
  check('events: off(type, fn) without ctx removes every ctx', h1.n === 1 && h2.n === 2 && !ev.has('hit'));
  // off of an unfired once listener mid-dispatch: still gets this dispatch (deferred), never again, live count intact
  log = '';
  const OB = () => { log += 'B'; };
  const OA = () => { log += 'A'; ev.off('ox', OB); };
  ev.on('ox', OA); ev.once('ox', OB);
  ev.emit('ox'); ev.emit('ox');
  check('events: off of unfired once mid-dispatch (ABA)', log === 'ABA');
  ev.off('ox', OA);
  const hasAfterOff = ev.has('ox');
  ev.on('ox', () => { log += 'C'; });
  check('events: ...live count intact afterwards', !hasAfterOff && ev.has('ox') && ev.emit('ox') && log === 'ABAC');
  ev.clear();
  check('events: clear()', !ev.has('t') && !ev.has('args') && !ev.has('d') && !ev.has('a'));
  check('events: emit after clear returns false', ev.emit('t') === false);
}
// allocation + speed: 1e6 emits of a 3-arg event to 3 listeners
{
  const ev = new Events();
  let sum = 0;
  const payload = { v: 1 };
  ev.on('hit', (a, b, c) => { sum = (sum + a + b + c.v) | 0; });
  ev.on('hit', (a, b) => { sum = (sum ^ b) | 0; });
  ev.on('hit', (a) => { sum = (sum - a) | 0; });
  const run = (n) => { for (let i = 0; i < n; i++) ev.emit('hit', i, 7, payload); };
  run(2e5); run(2e5); // warm up (JIT)
  const m = measureAlloc(() => run(1e6));
  check('events: 1e6 emits x 3 listeners (ms)', m.ms, 0, 150, 'ms');
  check('events: GCs during 1e6 emits', m.gcs, 0, 0);
  check('events: heap growth over 1e6 emits (bytes)', m.bytes, -Infinity, 65536, 'B'); // 1 object per emit = 16+ MB
  // control: the same measurement sees a per-emit array copy
  const fns = [(a) => { sum = (sum + a) | 0; }];
  let last = null;
  const leaky = (n) => { for (let i = 0; i < n; i++) { last = fns.slice(); last[0](i); } };
  leaky(1e4);
  const ml = measureAlloc(() => leaky(1e6));
  check('events: control (slice per emit) is detected', ml.gcs > 0 || ml.bytes > 1e6);
}

// ================================================================================================= Clock
{
  const c = new Clock();
  check('clock: defaults', c.timeScale === 1 && c.paused === false && c.time === 0 && c.unscaledTime === 0 && c.frame === 0);
  check('clock: scale at timeScale 1', c.scale(0.016), 0.016, 0.016);
  c.timeScale = 0.25;
  check('clock: scale at timeScale 0.25', c.scale(0.02), 0.005, 0.005);
  c.pause();
  check('clock: scale while paused', c.scale(0.02), 0, 0);
  c.resume();
  for (let i = 0; i < 10; i++) c.advance(0.1);
  c.pause();
  for (let i = 0; i < 5; i++) c.advance(0.1);
  check('clock: time (scaled, not while paused)', c.time, 0.25 - 1e-12, 0.25 + 1e-12, 's');
  check('clock: unscaledTime', c.unscaledTime, 1.5 - 1e-12, 1.5 + 1e-12, 's');
  check('clock: frame', c.frame, 15);
  check('clock: advance returns scaled dt', c.advance(0.1), 0, 0);
  c.step(); c.step(3); c.step(0); c.step(-2); c.step(NaN);
  check('clock: pendingSteps', c.pendingSteps, 4);
  check('clock: takeSteps returns the queue', c.takeSteps(), 4);
  check('clock: takeSteps clears it', c.takeSteps(), 0);
  c.step(2); c.resume();
  check('clock: resume drops queued steps', c.takeSteps(), 0);
  check('clock: toggle -> paused', c.toggle() === true && c.paused);
  check('clock: toggle -> running', c.toggle() === false && !c.paused);
  // timeScale / dt validation: nothing invalid reaches the Engine accumulator
  const cv = new Clock({ timeScale: -1 });
  const ts0 = cv.timeScale;
  cv.timeScale = NaN; const ts1 = cv.timeScale;
  cv.timeScale = -2; const ts2 = cv.timeScale;
  cv.timeScale = Infinity; const ts3 = cv.timeScale;
  cv.timeScale = 0; const ts4 = cv.timeScale;
  check('clock: invalid timeScale (ctor -1, NaN, -2, Inf) -> 1; 0 kept', ts0 === 1 && ts1 === 1 && ts2 === 1 && ts3 === 1 && ts4 === 0);
  cv.timeScale = 2;
  check('clock: scale / advance of NaN or negative dt -> 0', cv.scale(NaN) === 0 && cv.scale(-1) === 0 && cv.advance(NaN) === 0 && cv.advance(-0.5) === 0 && cv.time === 0 && cv.unscaledTime === 0 && cv.frame === 2);
  check('clock: advance(raw) returns the scaled dt', cv.advance(0.25), 0.5, 0.5);
  // the Engine wiring (lead): dt = advance(raw); paused -> takeSteps() forced steps only
  const fixedDt = 1 / 120;
  let acc = 0, steps = 0;
  const tick = (raw) => {
    const dt = c.advance(raw);
    if (c.paused) { const k = c.takeSteps(); for (let i = 0; i < k; i++) steps++; return; }
    acc += dt; while (acc >= fixedDt) { steps++; acc -= fixedDt; }
  };
  c.timeScale = 1;
  for (let i = 0; i < 60; i++) tick(1 / 60);
  check('clock wiring: 1 s at 1x = 120 fixed steps', steps, 119, 120);
  steps = 0; c.timeScale = 0.5;
  for (let i = 0; i < 60; i++) tick(1 / 60);
  check('clock wiring: 1 s at 0.5x = 60 fixed steps', steps, 59, 61);
  steps = 0; c.pause();
  for (let i = 0; i < 60; i++) tick(1 / 60);
  check('clock wiring: paused = 0 steps', steps, 0);
  c.step(3); tick(1 / 60); tick(1 / 60);
  check('clock wiring: paused + step(3) = 3 steps', steps, 3);
}

// ================================================================================================= Entity
class Health { constructor(hp) { this.hp = hp; this.attached = 0; this.detached = 0; } onAttach(e) { this.attached++; this.seen = e; } onDetach() { this.detached++; this.had = this.entity; } }
class Mover { constructor() { this.f = 0; this.u = 0; this.l = 0; } fixedUpdate() { this.f++; } update() { this.u++; } lateUpdate() { this.l++; } }
class Ticker { constructor() { this.u = 0; } update() { this.u++; } }
const consistent = (sys) => {
  const p = sys.phases;
  for (let i = 0; i < p.fixed.length; i++) if (p.fixed[i].__fi !== i) return false;
  for (let i = 0; i < p.update.length; i++) if (p.update[i].__ui !== i) return false;
  for (let i = 0; i < p.late.length; i++) if (p.late[i].__li !== i) return false;
  for (let i = 0; i < sys.entities.length; i++) if (sys.entities[i].__si !== i || sys.entities[i].system !== sys) return false;
  return true;
};
{
  const sys = new EntitySystem();
  const e = new Entity('car');
  check('entity: is an Object3D with name', e instanceof THREE.Object3D && e.name === 'car' && e.isEntity);
  const h = e.addComponent(new Health(100));
  check('entity: addComponent returns c, sets entity, onAttach', h.entity === e && h.attached === 1 && h.seen === e);
  check('entity: addComponent twice is a no-op', e.addComponent(h) === h && e.components.length === 1 && h.attached === 1);
  const named = e.addComponent({ name: 'tag', update() {} });
  const mv = e.addComponent(new Mover());
  check('entity: getComponent(constructor)', e.getComponent(Mover) === mv && e.getComponent(Health) === h);
  check('entity: getComponent(constructor.name)', e.getComponent('Health') === h);
  check('entity: getComponent(c.name)', e.getComponent('tag') === named);
  check('entity: getComponent miss -> null', e.getComponent(Ticker) === null && e.getComponent('nope') === null);
  // phases: Health none, named update, Mover all three
  check('system: no phase entries before add', sys.phases.update.length, 0);
  sys.add(e);
  const p = sys.phases;
  check('system: add registers hooks (fixed/update/late = 1/2/1)', p.fixed.length === 1 && p.update.length === 2 && p.late.length === 1);
  const t = e.addComponent(new Ticker());
  check('system: attach after add registers (update = 3)', p.update.length, 3);
  sys.fixedUpdate(1 / 120); sys.update(1 / 60, 0.5); sys.lateUpdate(1 / 60, 0.5);
  check('system: phases ran once each', mv.f === 1 && mv.u === 1 && mv.l === 1 && t.u === 1);
  check('entity: removeComponent returns c', e.removeComponent(mv) === mv && mv.entity === null);
  check('system: detach swap-removes (fixed/update/late = 0/2/0)', p.fixed.length === 0 && p.update.length === 2 && p.late.length === 0 && consistent(sys));
  check('entity: removeComponent of a stranger -> null', e.removeComponent(new Ticker()) === null);
  const e2 = new Entity('other');
  e2.addComponent(t);
  check('entity: addComponent moves c between entities', t.entity === e2 && e.components.indexOf(t) < 0 && p.update.length === 1);
  sys.add(e2);
  check('system: add(e2)', sys.entities.length === 2 && p.update.length === 2 && consistent(sys));
  sys.remove(e);
  check('system: remove keeps components attached', e.system === null && h.entity === e && h.detached === 0 && p.update.length === 1 && consistent(sys));
  sys.add(e);
  const sys2 = new EntitySystem();
  sys2.add(e);
  let msg = '';
  try { sys.add({}); } catch (err) { msg = err instanceof TypeError ? err.message : ''; }
  let threw3 = false;
  try { sys.add(new THREE.Object3D()); } catch { threw3 = true; }
  check('system: add(non-Entity) throws a clear TypeError', msg === 'EntitySystem.add: expected an Entity' && threw3);
  check('system: add moves an entity between systems', e.system === sys2 && sys.entities.length === 1 && sys2.phases.update.length === 1 && consistent(sys) && consistent(sys2));
  // destroy: descendants (also below a plain Group), components, system, parent
  const scene = new THREE.Scene();
  const root = new Entity('root'), group = new THREE.Group(), kid = new Entity('kid'), kid2 = new Entity('kid2');
  scene.add(root); root.add(group); group.add(kid); root.add(kid2);
  const kh = kid.addComponent(new Health(1)), rh = root.addComponent(new Health(2)), rt = root.addComponent(new Ticker());
  sys.add(root); sys.add(kid); sys.add(kid2);
  const before = sys.entities.length;
  root.destroy();
  check('entity: destroy detaches components (onDetach, entity=null)', rh.detached === 1 && rh.had === root && rh.entity === null && rt.entity === null && root.components.length === 0);
  check('entity: destroy leaves system and parent', root.system === null && root.parent === null && scene.children.length === 0);
  check('entity: destroy destroys descendant entities', kid.system === null && kh.detached === 1 && kid.parent === null && kid2.parent === null && group.parent === root);
  check('system: entities after destroy', sys.entities.length, before - 3);
  check('system: indices consistent after destroy', consistent(sys));
}
// changes during a phase loop
{
  const sys = new EntitySystem();
  const order = [];
  const mk = (id, fn) => { const e = new Entity(id); e.addComponent({ id, update(dt) { order.push(id); if (fn) fn(this, dt); } }); sys.add(e); return e; };
  const eA = mk('A', () => { if (eC.system) eC.destroy(); });           // destroys a later entity
  const eB = mk('B', (c) => { if (!spawned) { spawned = mk('S'); } });   // spawns
  let spawned = null;
  const eC = mk('C');
  const eD = mk('D', (c) => c.entity.destroy());                       // destroys itself
  const eE = mk('E');
  sys.update(1 / 60, 0);
  check('loop: removed-later not called, spawned deferred (ABDE)', order.join('') === 'ABDE');
  check('loop: compacted after the pass (A B E S)', sys.phases.update.length === 4 && sys.phases.update.every((c) => c.update) && consistent(sys));
  order.length = 0; sys.update(1 / 60, 0);
  check('loop: next pass order (ABES)', order.join('') === 'ABES');
  check('loop: destroyed entities left', eC.system === null && eD.system === null && sys.entities.length === 4 && eA.system === sys && eB.system === sys && eE.system === sys);
  // spawn in fixedUpdate runs in update the same frame
  let ran = 0;
  const host = new Entity('host');
  host.addComponent({ fixedUpdate() { if (!ran) { const e = new Entity(); e.addComponent({ update() { ran++; } }); sys.add(e); ran = -1; } } });
  sys.add(host);
  sys.fixedUpdate(1 / 120); sys.update(1 / 60, 0);
  check('loop: spawned in fixedUpdate updates same frame', ran, 0);
  // randomized stress: destroy / spawn / re-attach from inside update, 200 frames
  const rnd = mulberry32(99);
  const stress = new EntitySystem();
  let calls = 0, expected = 0;
  const live = new Set();
  class Chaos {
    update() {
      calls++;
      if (!live.has(this)) throw new Error('called after removal');
      const r = rnd();
      if (r < 0.05) this.entity.destroy();
      else if (r < 0.10 && stress.entities.length) { const o = stress.entities[(rnd() * stress.entities.length) | 0]; o.destroy(); }
      else if (r < 0.20) spawn();
      else if (r < 0.22) { const e = this.entity, c = e.removeComponent(this); e.addComponent(c); }
    }
    onAttach() { live.add(this); }
    onDetach() { live.delete(this); }
  }
  const spawn = () => { const e = new Entity(); e.addComponent(new Chaos()); stress.add(e); };
  for (let i = 0; i < 300; i++) spawn();
  let ok = true, threw = null;
  try {
    for (let f = 0; f < 200; f++) {
      expected = stress.phases.update.length;
      calls = 0;
      stress.update(1 / 60, 0);
      if (calls > expected || !consistent(stress) || stress.phases.update.length !== live.size || stress.phases.update.includes(undefined)) ok = false;
    }
  } catch (err) { threw = err; ok = false; }
  check('loop: 200 random frames, removed never called', threw === null);
  check('loop: 200 random frames, dense arrays consistent', ok);
  check('loop: stress population (entities)', stress.entities.length, 1, 100000);
  stress.dispose();
  check('system: dispose destroys every entity', stress.entities.length === 0 && stress.phases.update.length === 0 && live.size === 0);
}
// per-frame allocation of the phase loops: 1000 components x 1000 frames
{
  const sys = new EntitySystem();
  for (let i = 0; i < 1000; i++) { const e = new Entity(); e.addComponent(new Mover()); sys.add(e); }
  const frames = (n) => { for (let f = 0; f < n; f++) { sys.fixedUpdate(1 / 120); sys.update(1 / 60, 0.5); sys.lateUpdate(1 / 60, 0.5); } };
  frames(1000);
  const m = measureAlloc(() => frames(1000));
  check('system: 3e6 hook calls (ms)', m.ms, 0, 200, 'ms');
  check('system: GCs during 1000 frames', m.gcs, 0, 0);
  check('system: heap growth over 3e6 hook calls (bytes)', m.bytes, -Infinity, 65536, 'B');
}

// ================================================================================================= mathDet
{
  const r = mulberry32(12345);
  const pinned = [0.97972826776094735, 0.30675226449966431, 0.48420542152598500, 0.81793441250920296, 0.50942836934700608];
  let same = true;
  for (let i = 0; i < 5; i++) if (r() !== pinned[i]) same = false;
  check('mulberry32(12345): first 5 values pinned', same);
  const a = mulberry32(777), b = noiseMulberry32(777);
  let eq = true;
  for (let i = 0; i < 10000; i++) if (a() !== b()) eq = false;
  check('mulberry32 == render/noise.js mulberry32 (1e4)', eq);
  // hashing
  const arr = new Float64Array(1000), g = mulberry32(7);
  for (let i = 0; i < arr.length; i++) arr[i] = (g() - 0.5) * 1e3;
  const h1 = hashF64Array(arr), h2 = hashF64Array(arr);
  let hs = 2166136261;
  for (let i = 0; i < arr.length; i++) hs = hashF64(hs, arr[i]);
  check('hashF64Array pinned (2602498353)', h1, 2602498353);
  check('hash deterministic + equals hashF64 chain', h1 === h2 && h1 === hs);
  check('hashU32 pinned (1280657671, 3864292196)', hashU32(2166136261, 0xdeadbeef) === 1280657671 && hashU32(0, 0) === 3864292196);
  check('hashF64 pinned (pi, +0, -0, NaN)', hashF64(2166136261, Math.PI) === 3346376265 && hashF64(2166136261, 0) === 3487615221 && hashF64(2166136261, -0) === 3621801689 && hashF64(2166136261, NaN) === 3833937433);
  check('hash: -0 != +0, NaN canonical', hashF64(1, -0) !== hashF64(1, 0) && hashF64(1, NaN) === hashF64(1, -NaN) && hashF64(1, NaN) === hashF64(1, 0 / 0));
  arr[500] = Math.nextUp ? Math.nextUp(arr[500]) : arr[500] * (1 + Number.EPSILON);
  check('hash: one-ulp change changes the hash', hashF64Array(arr) !== h1);
  check('hashF64Array range', hashF64Array(arr, 123, 10, 20) === hashF64Array(arr.subarray(10, 20), 123));
  check('hash results are uint32', h1 >>> 0 === h1 && hashU32(-1, -1) >>> 0 === hashU32(-1, -1));
  // inverse trig accuracy, 1e6 samples each
  const N = 1e6, R = mulberry32(2024);
  let eA2 = 0, eAc = 0, eAs = 0, eAt = 0;
  for (let i = 0; i < N; i++) {
    let y = R() * 2 - 1, x = R() * 2 - 1;
    const k = i & 7;
    if (k === 1) { y *= 10 ** (R() * 40 - 20); x *= 10 ** (R() * 40 - 20); }      // ratios far from 1
    else if (k === 2) { y *= 10 ** (R() * 600 - 300); x *= 10 ** (R() * 600 - 300); } // extreme magnitudes
    else if (k === 3) { if (R() < 0.5) y = R() < 0.5 ? 0 : -0; else x = R() < 0.5 ? 0 : -0; } // axes
    else if (k === 4) { y = x * (1 + (R() - 0.5) * 1e-6); }                           // near the diagonals
    let e = Math.abs(detAtan2(y, x) - Math.atan2(y, x)); if (e > eA2) eA2 = e;
    const u = k === 5 ? (R() < 0.5 ? -1 : 1) * (1 - R() * 1e-6) : R() * 2 - 1;    // also near +-1
    e = Math.abs(detAcos(u) - Math.acos(u)); if (e > eAc) eAc = e;
    e = Math.abs(detAsin(u) - Math.asin(u)); if (e > eAs) eAs = e;
    const tt = (R() * 2 - 1) * 10 ** (R() * 12 - 6);
    e = Math.abs(detAtan(tt) - Math.atan(tt)); if (e > eAt) eAt = e;
  }
  check('detAtan2 max abs error, 1e6 samples', eA2, 0, 1e-9);
  check('detAcos max abs error, 1e6 samples', eAc, 0, 1e-9);
  check('detAsin max abs error, 1e6 samples', eAs, 0, 1e-9);
  check('detAtan max abs error, 1e6 samples', eAt, 0, 1e-9);
  // exact special cases (Object.is: signed zeros, NaN)
  const vals = [0, -0, 1, -1, 2.5, -2.5, Infinity, -Infinity, NaN, 1e-310, -1e-310];
  let bad = 0;
  for (const y of vals) for (const x of vals) if (!Object.is(detAtan2(y, x), Math.atan2(y, x)) && Math.abs(detAtan2(y, x) - Math.atan2(y, x)) > 1e-15) bad++;
  check('detAtan2 special cases (11 x 11) match Math.atan2', bad, 0);
  let badZ = 0;
  for (const y of [0, -0]) for (const x of [0, -0, 3, -3, Infinity, -Infinity]) if (!Object.is(detAtan2(y, x), Math.atan2(y, x))) badZ++;
  for (const y of [Infinity, -Infinity]) for (const x of [Infinity, -Infinity, 0, -0, 1, -1]) if (!Object.is(detAtan2(y, x), Math.atan2(y, x))) badZ++;
  for (const y of [1, -1]) for (const x of [Infinity, -Infinity, 0, -0]) if (!Object.is(detAtan2(y, x), Math.atan2(y, x))) badZ++;
  check('detAtan2 zero / infinity cases bit-identical', badZ, 0);
  check('detAcos edges (1 -> 0, -1 -> pi, 0 -> pi/2)', Object.is(detAcos(1), 0) && detAcos(-1) === Math.PI && detAcos(0) === Math.PI / 2 && detAcos(-0) === Math.PI / 2);
  check('detAcos |x| > 1 and NaN -> NaN', Number.isNaN(detAcos(1.0000001)) && Number.isNaN(detAcos(-1.5)) && Number.isNaN(detAcos(NaN)) && Number.isNaN(detAcos(Infinity)));
  check('detAsin edges (+-0, +-1, NaN)', Object.is(detAsin(-0), -0) && Object.is(detAsin(0), 0) && detAsin(1) === Math.PI / 2 && detAsin(-1) === -Math.PI / 2 && Number.isNaN(detAsin(2)));
  check('detAtan edges (+-0, +-Inf, NaN)', Object.is(detAtan(-0), -0) && detAtan(Infinity) === Math.PI / 2 && detAtan(-Infinity) === -Math.PI / 2 && Number.isNaN(detAtan(NaN)));
  check('detSqrt / clamp', detSqrt === Math.sqrt && clamp(5, 0, 1) === 1 && clamp(-5, 0, 1) === 0 && clamp(0.5, 0, 1) === 0.5);
  // speed (informational bound)
  let acc = 0;
  const t0 = performance.now();
  for (let i = 0; i < 1e6; i++) acc += detAtan2(i - 5e5, 3e5 - i * 0.7);
  check('detAtan2 1e6 calls (ms)', performance.now() - t0, 0, 200, 'ms');
  if (!Number.isFinite(acc)) failures++;
}

console.log(results.join('\n'));
console.log(failures ? `\n${failures} check(s) FAILED` : '\nAll core checks passed.');
process.exit(failures ? 1 : 0);
