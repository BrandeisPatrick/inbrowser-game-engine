// Headless checks for core/Assets (registry, dedupe, concurrency, refcounts, progress, errors, Engine.load plumbing),
// the .cube LUT parser of render/loaders.js and the audio loader / SamplePlayer of audio/Samples.js against fakes.
// No browser, no WebGL, no WebAudio: fetch, the AudioContext and its nodes are injected fakes. The gltf / texture /
// hdr paths and real decoding need a browser check.
// Run: node test/assets.mjs
import { Object3D } from 'three';
import { Assets } from '../src/core/Assets.js';
import { Engine } from '../src/core/Engine.js';
import { registerRenderLoaders, parseCubeLUT, disposeInstance } from '../src/render/loaders.js';
import { clone as cloneSkinned } from 'three/addons/utils/SkeletonUtils.js';
import { registerAudioLoader, SamplePlayer } from '../src/audio/Samples.js';
import { AudioBus } from '../src/audio/AudioBus.js';
import { PositionalAudio } from '../src/audio/Positional.js';
import * as THREE from 'three';

let failures = 0;
const results = [];
function check(name, value, lo, hi, unit = '') {
  const v = typeof value === 'boolean' ? +value : value;
  const ok = Number.isFinite(v) && v >= lo && v <= hi;
  if (!ok) failures++;
  results.push(`${ok ? 'PASS' : 'FAIL'}  ${name.padEnd(56)} ${fmt(v)} ${unit}   [${lo} .. ${hi}]`);
  return ok;
}
const fmt = (x) => (Number.isFinite(x) ? x.toFixed(3).padStart(9) : String(x).padStart(9));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const settle = async (p) => { try { return { v: await p }; } catch (e) { return { e }; } };

/** Fake fetch over `routes` (resolved url -> { body, delay? }); counts calls and the max number in flight. */
function makeFetch(routes, delay = 1) {
  const f = async (url) => {
    f.calls.push(url); f.inFlight++; f.max = Math.max(f.max, f.inFlight);
    const r = routes[url];
    await sleep(r && r.delay != null ? r.delay : delay);
    f.inFlight--;
    if (!r) return { ok: false, status: 404 };
    const b = r.body;
    return {
      ok: true, status: 200,
      json: async () => JSON.parse(typeof b === 'string' ? b : JSON.stringify(b)),
      text: async () => String(b),
      arrayBuffer: async () => (b instanceof ArrayBuffer ? b : new TextEncoder().encode(String(b)).buffer),
      blob: async () => b,
    };
  };
  f.calls = []; f.inFlight = 0; f.max = 0;
  return f;
}

// --- 1. progress: weights, ordering, labels ------------------------------------------
{
  const fetch = makeFetch({ '/d/a.json': { body: { a: 1 }, delay: 30 }, '/d/b.txt': { body: 'bee', delay: 5 }, '/d/c.bin': { body: 'ccc', delay: 15 } });
  const assets = new Assets({ fetch });
  assets.add('a', { type: 'json', url: 'd/a.json', weight: 1 }).add('b', { type: 'text', url: 'd/b.txt', weight: 2 }).add('c', { type: 'bin', url: 'd/c.bin' });
  const seen = [];
  const r = await assets.load((p, label) => seen.push([p, label]));
  check('load() resolves to the Assets instance', r === assets, 1, 1);
  check('progress callbacks: one per key', seen.length, 3, 3);
  check('progress order = finish order (b, c, a)', seen.map((s) => s[1]).join() === 'b,c,a', 1, 1);
  check('progress after b (weight 2 of 4)', seen[0][0], 0.5, 0.5);
  check('progress after c (weight 1)', seen[1][0], 0.75, 0.75);
  check('progress after a (last)', seen[2][0], 1, 1);
  check('json value', assets.get('a').a, 1, 1);
  check('text value', assets.get('b') === 'bee', 1, 1);
  check('bin value is an ArrayBuffer of 3 bytes', assets.get('c') instanceof ArrayBuffer ? assets.get('c').byteLength : NaN, 3, 3, 'B');
  const empty = [];
  await assets.load((p, l) => empty.push(p));
  check('load() with nothing pending reports 1 once', empty.length === 1 && empty[0] === 1, 1, 1);
  check('has / peek / status of a loaded key', assets.has('a') && assets.peek('b') === 'bee' && assets.status('c') === 'loaded', 1, 1);
  let threw = false; try { assets.get('nope'); } catch { threw = true; }
  check('get() of an unknown key throws', threw, 1, 1);
  check('peek() of an unknown key is undefined', assets.peek('nope') === undefined, 1, 1);
}

// --- 2. URL resolution + dedupe -------------------------------------------------------
{
  const fetch = makeFetch({ '/game/s.json': { body: { n: 7 } }, 'https://cdn.example.com/x/y.json': { body: { n: 8 } }, '/root.json': { body: { n: 9 } } });
  const assets = new Assets({ base: '/game', fetch });
  assets.add('one', { type: 'json', url: 's.json' }).add('two', { type: 'json', url: './s.json' }).add('three', { type: 'json', url: '/game/s.json' })
    .add('abs', { type: 'json', url: 'https://cdn.example.com/x/y.json' }).add('root', { type: 'json', url: '/root.json' })
    .add('asText', { type: 'text', url: 's.json' });
  await assets.load();
  const shared = fetch.calls.filter((u) => u === '/game/s.json').length;
  check('dedupe: 3 json keys + 1 text key on one URL: fetches', shared, 2, 2);
  check('dedupe: keys share one value object', assets.get('one') === assets.get('two') && assets.get('two') === assets.get('three'), 1, 1);
  check('another type on the same URL loads separately', typeof assets.get('asText') === 'string', 1, 1);
  check('absolute URL kept / root-relative kept', assets.get('abs').n === 8 && assets.get('root').n === 9, 1, 1);
  const f2 = makeFetch({ 'https://cdn.example.com/g/a.json': { body: 1 } });
  const a2 = new Assets({ base: 'https://cdn.example.com/g/', fetch: f2 });
  await a2.add('k', { type: 'json', url: 'a.json' }).load();
  check('relative URL against an absolute base', f2.calls[0] === 'https://cdn.example.com/g/a.json', 1, 1);
  const f3 = makeFetch({ '/t.json': { body: 2 } });
  const a3 = new Assets({ fetch: f3 });
  a3.add('x', { type: 'json', url: 't.json', init: { cache: 'no-store' } }).add('y', { type: 'json', url: 't.json' });
  await a3.load();
  check('different options on one URL: two fetches', a3.get('x') === 2 && a3.get('y') === 2 ? f3.calls.length : NaN, 2, 2);
}

// --- 3. concurrency cap ---------------------------------------------------------------
{
  const routes = {};
  for (let i = 0; i < 12; i++) routes[`/c/${i}.json`] = { body: i, delay: 3 + (i % 4) };
  const fetch = makeFetch(routes);
  const assets = new Assets({ concurrency: 3, fetch });
  for (let i = 0; i < 12; i++) assets.add(`k${i}`, { type: 'json', url: `c/${i}.json` });
  const order = [];
  await assets.load((p, k) => order.push(k));
  check('concurrency 3: max fetches in flight', fetch.max, 3, 3);
  check('concurrency 3: all 12 loaded', [...Array(12).keys()].filter((i) => assets.get(`k${i}`) === i).length, 12, 12);
  check('requests start in add order', fetch.calls.join() === Object.keys(routes).join(), 1, 1);
  // two overlapping load() calls share the cap
  const f2 = makeFetch(routes);
  const b = new Assets({ concurrency: 2, fetch: f2 });
  for (let i = 0; i < 6; i++) b.add(`k${i}`, { type: 'json', url: `c/${i}.json` });
  const p1 = b.load();
  for (let i = 6; i < 12; i++) b.add(`k${i}`, { type: 'json', url: `c/${i}.json` });
  await Promise.all([p1, b.load()]);
  check('overlapping load() calls: max in flight', f2.max, 2, 2);
  // loadOne skips the queue (on-demand loads never wait for a slot)
  const f3 = makeFetch({ '/s/slow.json': { body: 1, delay: 40 }, '/s/q.json': { body: 2, delay: 1 }, '/s/now.json': { body: 3, delay: 1 } });
  const c = new Assets({ concurrency: 1, fetch: f3 });
  c.add('slow', { type: 'json', url: 's/slow.json' }).add('q', { type: 'json', url: 's/q.json' }).add('now', { type: 'json', url: 's/now.json' });
  const pAll = c.load();
  const t0 = performance.now();
  const v = await c.loadOne('now');
  const ms = performance.now() - t0;
  check('loadOne of a queued key does not wait for the slot', ms, 0, 30, 'ms');
  check('loadOne value', v, 3, 3);
  await pAll;
  check('queued key still loads after loadOne bumped another', c.get('q'), 2, 2);
  const un = await settle(c.loadOne('never'));
  check('loadOne of a key never added rejects', !!un.e && /never/.test(un.e.message), 1, 1);
  // freeSlot: a loader that hands its slot back early lets the queue move on
  let open; const gate = new Promise((r) => { open = r; });
  const d = new Assets({ concurrency: 1, fetch: makeFetch({ '/f/j.json': { body: 5 } }) });
  d.register('gated', async (desc, a, freeSlot) => { freeSlot(); await gate; return 'g'; });
  d.add('g', { type: 'gated', url: 'g' }).add('j', { type: 'json', url: 'f/j.json' });
  const pd = d.load();
  await sleep(20);
  check('freeSlot(): next key loads while the first waits', d.status('j') === 'loaded' && d.status('g') === 'loading', 1, 1);
  open(); await pd;
  check('freeSlot(): gated key lands afterwards', d.get('g') === 'g', 1, 1);
}

// --- 4. refcounts + dispose -----------------------------------------------------------
{
  const fetch = makeFetch({ '/r/t.json': { body: { t: 1 } }, '/r/slow.json': { body: 2, delay: 20 } });
  const assets = new Assets({ fetch });
  let disposed = 0, last = null;
  assets.register('thing', (d, a) => a.fetchOk(d.url).then((r) => r.json()), { dispose: (v) => { disposed++; last = v; } });
  assets.add('a', { type: 'thing', url: 'r/t.json' }).add('b', { type: 'thing', url: 'r/t.json' });
  await assets.load();
  const val = assets.get('a');
  assets.release('a');
  check('release of 1 of 2 refs: no dispose', disposed, 0, 0);
  check('other key still loaded', assets.get('b') === val, 1, 1);
  assets.release('b');
  check('release of the last ref: dispose called once', disposed, 1, 1);
  check('dispose got the value', last === val, 1, 1);
  check('released key is gone (has / status)', !assets.has('b') && assets.status('b') === null, 1, 1);
  assets.release('b');
  check('double release: no second dispose', disposed, 1, 1);
  assets.add('c', { type: 'thing', url: 'r/t.json' });
  await assets.load();
  check('re-add after eviction fetches again', fetch.calls.length, 2, 2);
  check('re-added value is fresh', assets.get('c') !== val && assets.get('c').t === 1, 1, 1);
  // released while loading: disposed once when it lands, value never exposed
  assets.add('s', { type: 'thing', url: 'r/slow.json' });
  const p = assets.load();
  await sleep(2);
  assets.release('s');
  await p;
  check('released mid-load: disposed once on arrival', disposed, 2, 2);
  // dispose() releases everything
  assets.add('d', { type: 'thing', url: 'r/t.json' });
  await assets.load();
  assets.dispose();
  check('assets.dispose() releases every key', disposed, 3, 3);
  // a key added while its shared resource is already loaded is ready at once
  assets.add('e1', { type: 'json', url: 'r/t.json' });
  await assets.load();
  assets.add('e2', { type: 'json', url: 'r/t.json' });
  check('dedupe onto a loaded value: ready without load()', assets.status('e2') === 'loaded' && assets.get('e2') === assets.get('e1'), 1, 1);
}

// --- 5. errors + retry -----------------------------------------------------------------
{
  const routes = { '/e/ok.json': { body: 1 } };
  const fetch = makeFetch(routes);
  const assets = new Assets({ fetch });
  assets.add('ok', { type: 'json', url: 'e/ok.json' }).add('missing', { type: 'json', url: 'e/missing.json' }).add('alsoMissing', { type: 'text', url: 'e/gone.txt' });
  const r = await settle(assets.load());
  const m = r.e ? r.e.message : '';
  check('failing URL rejects load()', !!r.e, 1, 1);
  check('error names the key', m.includes('"missing"'), 1, 1);
  check('error names the URL', m.includes('/e/missing.json'), 1, 1);
  check('error carries HTTP status + other failures', m.includes('HTTP 404') && m.includes('alsoMissing') && r.e.errors.length === 2, 1, 1);
  check('error .key / .url fields', r.e && r.e.key === 'missing' && r.e.url === '/e/missing.json', 1, 1);
  check('rest of the batch still loaded', assets.get('ok'), 1, 1);
  check('failed key: status failed, get throws', assets.status('missing') === 'failed' && (() => { try { assets.get('missing'); return false; } catch (e) { return /failed/.test(e.message); } })(), 1, 1);
  const lo = await settle(assets.loadOne('missing'));
  check('loadOne of a failed key rejects', !!lo.e, 1, 1);
  routes['/e/missing.json'] = { body: { back: true } };
  assets.add('missing', { type: 'json', url: 'e/missing.json' });
  check('add() of a failed key re-queues it', assets.status('missing') === 'pending', 1, 1);
  await assets.loadOne('missing');
  check('retry after add() loads it', assets.get('missing').back === true, 1, 1);
  let threw = false; try { assets.add('ok', { type: 'json', url: 'e/other.json' }); } catch { threw = true; }
  check('re-adding a live key with another URL throws', threw, 1, 1);
  threw = false; try { assets.add('x', { type: 'nope', url: 'a' }); } catch (e) { threw = /unknown type/.test(e.message); }
  check('unknown type throws at add()', threw, 1, 1);
  const bad = new Assets({ fetch });
  bad.register('boom', () => { throw new Error('sync boom'); });
  const rb = await settle(bad.add('b', { type: 'boom', url: 'x' }).load());
  check('a loader that throws synchronously rejects with its key', !!rb.e && /"b"/.test(rb.e.message) && /sync boom/.test(rb.e.message), 1, 1);
}

// --- 6. Engine.load plumbing: system() with the engine's onProgress(p, label) ------------
{
  const fetch = makeFetch({ '/g/1.json': { body: 1, delay: 5 }, '/g/2.json': { body: 2, delay: 10 } });
  const assets = new Assets({ fetch });
  assets.add('one', { type: 'json', url: 'g/1.json' }).add('two', { type: 'json', url: 'g/2.json' });
  const sys = assets.system({ loadWeight: 3 });
  check('system(): shape { load, loadWeight, __name }', typeof sys.load === 'function' && sys.loadWeight === 3 && sys.__name === 'assets', 1, 1);
  const other = { __name: 'world', loadWeight: 1, async load(cb) { cb(1, ''); } };
  const seen = [];
  await Engine.prototype.load.call({ systems: [other, sys] }, (p, label) => seen.push([+p.toFixed(4), label]));
  check('Engine.load: 4 callbacks (world, one, two, Ready)', seen.length, 4, 4);
  check('Engine.load: world system 1 of 4', seen[0][0], 0.25, 0.25);
  check('Engine.load: after "one" (1 + 3 x 0.5) / 4', seen[1][0], 0.625, 0.625);
  check('Engine.load: labels are the asset keys', seen[1][1] === 'one' && seen[2][1] === 'two' && seen[3][1] === 'Ready', 1, 1);
  check('Engine.load: assets loaded', assets.get('two'), 2, 2);
  const e2 = []; await Engine.prototype.load.call({ systems: [new Assets({ fetch }).system()] }, (p, l) => e2.push(l));
  check('empty assets system: label falls back to its name', e2[0] === 'assets', 1, 1);
}

// --- 7. .cube LUT parser (2 x 2 x 2) ------------------------------------------------------
{
  const rows = [];
  for (let i = 0; i < 8; i++) { const r = i & 1, g = (i >> 1) & 1, b = (i >> 2) & 1; rows.push([0.1 + 0.8 * r, 0.2 + 0.6 * g + 0.05 * i, 0.3 * b + 0.01 * i]); }
  const cube = ['# tiny test LUT', 'TITLE "Tiny Warm"', 'LUT_3D_SIZE 2', 'DOMAIN_MIN 0 0 0', 'DOMAIN_MAX 1 1 1', '',
    ...rows.map((v) => v.map((x) => x.toFixed(6)).join(' '))].join('\r\n');
  const lut = parseCubeLUT(cube, { float: true });
  const d = lut.texture.image.data;
  let maxErr = 0;
  for (let z = 0; z < 2; z++) for (let y = 0; y < 2; y++) for (let x = 0; x < 2; x++) {
    const i = x + y * 2 + z * 4, row = rows[x + 2 * y + 4 * z]; // red fastest in the file = x fastest in the texture
    for (let k = 0; k < 3; k++) maxErr = Math.max(maxErr, Math.abs(d[i * 4 + k] - row[k]));
    maxErr = Math.max(maxErr, Math.abs(d[i * 4 + 3] - 1));
  }
  check('LUT size', lut.size, 2, 2);
  check('LUT title', lut.title === 'Tiny Warm', 1, 1);
  check('LUT float texels = rows (red fastest), alpha 1: max error', maxErr, 0, 1e-6);
  check('LUT texel (1,1,1) green = 0.2 + 0.6 + 0.35', d[7 * 4 + 1], 1.1499, 1.1501);
  const t = lut.texture;
  check('LUT texture: Data3DTexture 2^3 RGBA float', t.isData3DTexture && t.image.width === 2 && t.image.depth === 2 && t.format === THREE.RGBAFormat && t.type === THREE.FloatType, 1, 1);
  check('LUT texture: linear filtering, clamp wrapping', t.minFilter === THREE.LinearFilter && t.magFilter === THREE.LinearFilter && t.wrapS === THREE.ClampToEdgeWrapping && t.wrapT === THREE.ClampToEdgeWrapping && t.wrapR === THREE.ClampToEdgeWrapping, 1, 1);
  const half = parseCubeLUT(cube);
  let hErr = 0;
  for (let i = 0; i < 32; i++) hErr = Math.max(hErr, Math.abs(THREE.DataUtils.fromHalfFloat(half.texture.image.data[i]) - d[i]));
  check('LUT half float (default): type + max error vs float', half.texture.type === THREE.HalfFloatType ? hErr : NaN, 0, 1e-3);
  check('LUT domain', lut.domainMin.join() === '0,0,0' && lut.domainMax.join() === '1,1,1', 1, 1);
  const dom = parseCubeLUT(cube.replace('DOMAIN_MAX 1 1 1', 'DOMAIN_MAX 2 2 4'));
  check('LUT DOMAIN_MAX parsed', dom.domainMax[2], 4, 4);
  const err = (txt) => { try { parseCubeLUT(txt); return ''; } catch (e) { return e.message; } };
  check('LUT: 7 rows for size 2 rejected', /7 rows, expected 8/.test(err(cube.split('\r\n').slice(0, -1).join('\n'))), 1, 1);
  check('LUT: LUT_3D_SIZE 1 rejected', /2\.\.256/.test(err(cube.replace('LUT_3D_SIZE 2', 'LUT_3D_SIZE 1'))), 1, 1);
  check('LUT: missing LUT_3D_SIZE rejected', /data before LUT_3D_SIZE/.test(err(cube.replace('LUT_3D_SIZE 2', ''))), 1, 1);
  check('LUT: 1D LUT rejected', /1D/.test(err('LUT_1D_SIZE 4\n0 0 0')), 1, 1);
  check('LUT: bad number rejected with line', /line 8: bad number/.test(err(cube.replace(rows[1].map((x) => x.toFixed(6)).join(' '), '0.9 x 0.01'))), 1, 1);
  // through Assets with the registered type (no renderer: only the lut path runs in Node)
  const assets = new Assets({ fetch: makeFetch({ '/luts/t.cube': { body: cube } }) });
  const handle = registerRenderLoaders(assets, null);
  assets.add('grade', { type: 'lut', url: 'luts/t.cube', float: true });
  await assets.load();
  check('lut asset type loads via Assets', assets.get('grade').texture.image.data[7 * 4 + 1], 1.1499, 1.1501);
  let types = 0; for (const ty of ['gltf', 'texture', 'hdr', 'lut']) { try { assets.add(`t-${ty}`, { type: ty, url: `x.${ty}` }); types++; } catch { /* unknown */ } }
  check('registerRenderLoaders: 4 types registered', types, 4, 4);
  check('registerRenderLoaders handle: no DRACO without dracoPath', handle.dracoLoader === null && !!handle.gltfLoader, 1, 1);
  for (const ty of ['gltf', 'texture', 'hdr', 'lut']) assets.release(`t-${ty}`);
  assets.release('grade');
}

// --- 8. audio: loader waits for the bus, SamplePlayer voices (fake WebAudio) -------------------------------
{
  const param = (v = 0) => ({ value: v, events: [], setValueAtTime(x, t) { this.value = x; this.events.push(['set', x, t]); }, linearRampToValueAtTime(x, t) { this.events.push(['ramp', x, t]); },
    setTargetAtTime(x, t) { this.value = x; this.events.push(['target', x, t]); }, cancelScheduledValues(t) { this.events.push(['cancel', t]); } });
  class FakeCtx {
    constructor() { this.state = 'suspended'; this.currentTime = 1; this.sampleRate = 48000; this.destination = this.node(); this.decoded = 0; this.listener = { positionX: param(), positionY: param(), positionZ: param(), forwardX: param(), forwardY: param(), forwardZ: param(), upX: param(), upY: param(), upZ: param() }; }
    node(extra = {}) { const ctx = this; return { context: ctx, out: [], connect(n) { this.out.push(n); return n; }, disconnect() { this.out.length = 0; this.cut = true; }, ...extra }; }
    createGain() { return this.node({ gain: param(1) }); }
    createDynamicsCompressor() { return this.node({ threshold: param(), knee: param(), ratio: param(), attack: param(), release: param() }); }
    createPanner() { return this.node({ positionX: param(), positionY: param(), positionZ: param(), orientationX: param(1), orientationY: param(), orientationZ: param() }); }
    createBufferSource() { return this.node({ playbackRate: param(1), loop: false, buffer: null, onended: null, start(t, o) { this.started = [t, o]; }, stop(t) { this.stopAt = t ?? -1; } }); }
    resume() { this.state = 'running'; return Promise.resolve(); }
    decodeAudioData(buf, ok) { this.decoded++; setTimeout(() => ok({ fakeBuffer: true, bytes: buf.byteLength, ctx: this }), 1); }
    close() {}
  }
  globalThis.window = { AudioContext: FakeCtx };
  class FakeOffline extends FakeCtx { constructor(ch, len, rate) { super(); this.args = [ch, len, rate]; } }
  globalThis.OfflineAudioContext = FakeOffline;
  const bus = new AudioBus();
  const fetch = makeFetch({ '/a/hit.ogg': { body: 'OggS....', delay: 2 }, '/a/x.json': { body: 3, delay: 2 } });
  const off = new Assets({ fetch });
  registerAudioLoader(off, bus); // default: offline decoding before the gesture
  off.add('hit', { type: 'audio', url: 'a/hit.ogg' });
  await off.load();
  const ob = off.get('hit');
  check('audio default: decoded offline, no gesture needed', !bus.ctx && ob.ctx instanceof FakeOffline && ob.bytes === 8, 1, 1);
  check('audio default: offline context at 48 kHz', ob.ctx.args[2], 48000, 48000, 'Hz');
  const assets = new Assets({ concurrency: 1, fetch });
  registerAudioLoader(assets, bus, { offline: false });
  assets.add('hit', { type: 'audio', url: 'a/hit.ogg' }).add('x', { type: 'json', url: 'a/x.json' });
  const p = assets.load();
  delete globalThis.OfflineAudioContext;
  const a0 = new Assets({ fetch });
  registerAudioLoader(a0, bus); // default, but no OfflineAudioContext: falls back to waiting
  a0.add('hit', { type: 'audio', url: 'a/hit.ogg' });
  const p0 = a0.load();
  await sleep(25);
  check('audio offline:false: waits for the bus before decoding', assets.status('hit') === 'loading' && !bus.ctx, 1, 1);
  check('audio without OfflineAudioContext: waits for the bus', a0.status('hit') === 'loading', 1, 1);
  check('audio: other keys load meanwhile (slot handed back)', assets.status('x') === 'loaded', 1, 1);
  await bus.start(); // the user gesture
  await Promise.all([p, p0]);
  const buf = assets.get('hit');
  check('audio: decoded with the bus context after start', !!buf.fakeBuffer && buf.ctx === bus.ctx && buf.bytes === 8 && a0.get('hit').ctx === bus.ctx, 1, 1);
  const a2 = new Assets({ fetch });
  registerAudioLoader(a2, bus);
  a2.add('hit', { type: 'audio', url: 'a/hit.ogg' });
  await a2.load();
  check('audio: started bus decodes at once', a2.get('hit').ctx === bus.ctx, 1, 1);

  const before = new SamplePlayer(new AudioBus());
  const inert = before.play(buf);
  check('SamplePlayer before start: inert voice', inert.playing === false && inert.stop(0.1) === inert && inert.setGain(1).setRate(2) === inert, 1, 1);

  const sfx = new SamplePlayer(bus);
  const v = sfx.play(buf, { gain: 0.5, rate: 1.25, loop: true, offset: 0.3 });
  check('play: source -> gain -> default bus "effects"', v.source.out[0] === v.gain && v.gain.out[0] === bus.buses.effects && !bus.buses.sfx, 1, 1);
  check('play: gain / rate / loop / offset applied', v.gain.gain.value === 0.5 && v.source.playbackRate.value === 1.25 && v.source.loop && v.source.started[1] === 0.3, 1, 1);
  check('play: voice tracked while playing', sfx.voices.size === 1 && v.playing, 1, 1);
  const f = sfx.play(buf, { fadeIn: 0.4, gain: 0.8, bus: 'ui' });
  const ev = f.gain.gain.events;
  check('play fadeIn: 0 then ramp to gain over fadeIn', ev[0][1] === 0 && ev[1][0] === 'ramp' && ev[1][1] === 0.8 && Math.abs(ev[1][2] - 1.4) < 1e-9, 1, 1);
  check('play bus option routes to that bus', f.gain.out[0] === bus.buses.ui, 1, 1);
  f.setGain(0.2, 0.5).setRate(0.5);
  const ge = f.gain.gain.events.at(-1);
  check('setGain(g, t) ramps / setRate sets', ge[0] === 'ramp' && ge[1] === 0.2 && f.source.playbackRate.value === 0.5, 1, 1);
  f.stop(0.25);
  check('stop(fade): fades, stops at t + fade, still playing', f.source.stopAt === 1.25 && f.playing && f.gain.gain.events.at(-1)[1] === 0, 1, 1);
  f.source.onended();
  check('ended: voice dropped and disconnected', !f.playing && sfx.voices.size === 1 && f.source.cut && f.gain.cut, 1, 1);

  const space = new PositionalAudio(bus);
  const ps = new SamplePlayer(bus, { positional: space });
  const obj = new Object3D(); obj.position.set(10, 2, -5); obj.updateMatrixWorld();
  const pv = ps.playAt(buf, obj, { refDistance: 3, gain: 0.7 });
  const em = space.emitters[0];
  check('playAt: voice -> emitter input (shared PositionalAudio)', space.emitters.length === 1 && pv.gain.out[0] === em.input, 1, 1);
  check('playAt: panner starts at the object', em.panner.positionX.value === 10 && em.panner.positionY.value === 2 && em.panner.positionZ.value === -5, 1, 1);
  check('playAt: emitter options + voice gain kept apart', em.panner.refDistance === 3 && em.input.gain.value === 1 && pv.gain.gain.value === 0.7, 1, 1);
  pv.source.onended();
  check('playAt: emitter removed when the voice ends', space.emitters.length, 0, 0);
  const own = new SamplePlayer(bus);
  own.playAt(buf, obj); own.update();
  check('playAt without a space: makes its own', own.positional instanceof PositionalAudio && own.positional.emitters.length === 1, 1, 1);
  sfx.stopAll();
  check('stopAll(): every voice stopped', v.playing === false && v.source.stopAt === -1, 1, 1);
  own.dispose(); sfx.dispose();
  check('dispose(): voices cleared', own.voices.size + sfx.voices.size, 0, 0);
  delete globalThis.window;
}

// --- 9. review fixes: throwing onProgress, weight snapshot, url-less re-add, release-then-fail, option identity --
{
  const fetch = makeFetch({ '/p/a.json': { body: 1, delay: 15 }, '/p/b.json': { body: 2, delay: 2 } });
  const assets = new Assets({ fetch });
  assets.add('a', { type: 'json', url: 'p/a.json' }).add('b', { type: 'json', url: 'p/b.json' });
  const logged = []; const ce = console.error; console.error = (...x) => logged.push(x);
  const r = await Promise.race([assets.load(() => { throw new Error('bar broke'); }).then(() => 'done'), sleep(500).then(() => 'hung')]);
  console.error = ce;
  check('throwing onProgress: load() still resolves', r === 'done' && assets.get('a') === 1, 1, 1);
  check('throwing onProgress: logged once per callback', logged.length, 2, 2);
  // weights are snapshotted when load() starts
  const b = new Assets({ fetch: makeFetch({ '/p/a.json': { body: 1, delay: 20 }, '/p/b.json': { body: 2, delay: 2 } }) });
  b.add('a', { type: 'json', url: 'p/a.json' }).add('b', { type: 'json', url: 'p/b.json' });
  const seen = [];
  const pb = b.load((p, k) => seen.push(p));
  b.add('b', { type: 'json', url: 'p/b.json', weight: 5 }); // re-add mid-batch with another weight
  await pb;
  check('re-add mid-batch: progress after first key', seen[0], 0.5, 0.5);
  check('re-add mid-batch: progress reaches 1 only at the end', seen.filter((x) => x >= 1).length === 1 && seen.at(-1) === 1, 1, 1);
  // url-less keys: re-adding the same descriptor is a no-op
  const c = new Assets({ fetch });
  let made = 0;
  c.register('gen', async (d) => ({ n: ++made, seed: d.seed }));
  let threw = false;
  try { c.add('g', { type: 'gen', seed: 3 }).add('g', { type: 'gen', seed: 3 }).add('h', { type: 'gen', seed: 3 }); } catch { threw = true; }
  await c.load();
  check('url-less re-add with the same descriptor: no throw', !threw, 1, 1);
  check('url-less keys never share a value', made === 2 && c.get('g') !== c.get('h'), 1, 1);
  // a key released mid-flight whose load then fails does not reject load()
  const d = new Assets({ fetch: makeFetch({ '/p/ok.json': { body: 1, delay: 2 } }, 10) });
  d.add('gone', { type: 'json', url: 'p/missing.json' }).add('ok', { type: 'json', url: 'p/ok.json' });
  const pd = settle(d.load());
  d.release('gone');
  const rd = await pd;
  check('released key that then fails: load() resolves', !rd.e && d.get('ok') === 1, 1, 1);
  // option identity: plain objects by value, class instances (Headers, AbortSignal) by reference
  const f = makeFetch({ '/p/b.json': { body: 2 } });
  const e = new Assets({ fetch: f });
  class Hdr { constructor(v) { this.v = v; } }
  const h1 = new Hdr(1);
  e.add('p1', { type: 'json', url: 'p/b.json', init: { cache: 'no-store' } }).add('p2', { type: 'json', url: 'p/b.json', init: { cache: 'no-store' } })
    .add('r1', { type: 'json', url: 'p/b.json', init: { headers: h1 } }).add('r2', { type: 'json', url: 'p/b.json', init: { headers: h1 } })
    .add('r3', { type: 'json', url: 'p/b.json', init: { headers: new Hdr(1) } });
  await e.load();
  check('options: equal plain init + same instance share, other instance not', f.calls.length, 3, 3);
}

// --- 10. glTF in Node (GLB with an embedded texture; fake createImageBitmap) + instances ----------------------------
{
  const bin = new ArrayBuffer(60);
  new Float32Array(bin, 0, 9).set([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  new Float32Array(bin, 36, 6).set([0, 0, 1, 0, 0, 1]);
  const json = {
    asset: { version: '2.0' }, scene: 0, scenes: [{ nodes: [0] }], nodes: [{ mesh: 0, name: 'tri' }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0, TEXCOORD_0: 1 }, material: 0 }] }],
    materials: [{ pbrMetallicRoughness: { baseColorTexture: { index: 0 } } }],
    textures: [{ source: 0 }], images: [{ uri: 'data:image/png;base64,iVBORw0KGgo=' }],
    buffers: [{ byteLength: 60 }], bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: 36 }, { buffer: 0, byteOffset: 36, byteLength: 24 }],
    accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: 'VEC3', min: [0, 0, 0], max: [1, 1, 0] }, { bufferView: 1, componentType: 5126, count: 3, type: 'VEC2' }],
  };
  const glb = (j, b) => {
    const je = new TextEncoder().encode(JSON.stringify(j)), jl = Math.ceil(je.length / 4) * 4, bl = Math.ceil(b.byteLength / 4) * 4;
    const out = new ArrayBuffer(28 + jl + bl), dv = new DataView(out), u = new Uint8Array(out);
    dv.setUint32(0, 0x46546c67, true); dv.setUint32(4, 2, true); dv.setUint32(8, out.byteLength, true);
    dv.setUint32(12, jl, true); dv.setUint32(16, 0x4e4f534a, true); u.fill(0x20, 20, 20 + jl); u.set(je, 20);
    dv.setUint32(20 + jl, bl, true); dv.setUint32(24 + jl, 0x004e4942, true); u.set(new Uint8Array(b), 28 + jl);
    return out;
  };
  const bitmaps = [];
  const hadSelf = 'self' in globalThis;
  if (!hadSelf) globalThis.self = globalThis; // GLTFLoader reads self.URL
  globalThis.createImageBitmap = async () => { const bm = { width: 1, height: 1, closed: false, close() { this.closed = true; } }; bitmaps.push(bm); return bm; };
  const draco = { ...json, extensionsUsed: ['KHR_draco_mesh_compression'] };
  const assets = new Assets({ fetch: makeFetch({ '/m/tri.glb': { body: glb(json, bin) }, '/m/draco.glb': { body: glb(draco, bin) } }) });
  registerRenderLoaders(assets, null);
  assets.add('tri', { type: 'gltf', url: 'm/tri.glb' });
  const r = await settle(assets.load());
  delete globalThis.createImageBitmap;
  if (!hadSelf) delete globalThis.self;
  check('gltf: GLB loads in Node', !r.e, 1, 1);
  if (r.e) results.push(`      (${r.e.message})`);
  const v = assets.peek('tri') || {};
  const keys = Object.keys(v).sort().join();
  check('gltf value: scene, animations, cameras, asset, userData, gltf, instantiate', keys === 'animations,asset,cameras,gltf,instantiate,scene,userData', 1, 1);
  check('gltf value: parser dropped (no GLB body / caches kept)', v.gltf && v.gltf.parser === null && v.asset.version === '2.0', 1, 1);
  const mesh = v.scene && v.scene.getObjectByName('tri');
  const mat = mesh && mesh.material, own = mat && mat.map;
  const inst = v.instantiate && v.instantiate();
  check('instantiate(): new root sharing geometry + material', !!inst && inst !== v.scene && inst.getObjectByName('tri').geometry === mesh.geometry, 1, 1);
  check('gltf: parser texture on the material', !!own && own.isTexture && own.image === bitmaps[0], 1, 1);
  if (!own) throw new Error('gltf test: no texture; see the failures above');
  const disposed = new Set(), spy = (o, n) => o.addEventListener('dispose', () => disposed.add(n));
  const env = new THREE.Texture(), foreign = new THREE.Texture(), extra = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial());
  spy(mesh.geometry, 'geometry'); spy(mat, 'material'); spy(own, 'own texture'); spy(env, 'envMap'); spy(foreign, 'foreign'); spy(extra.geometry, 'user mesh');
  mat.envMap = env; mat.lightMap = foreign; v.scene.add(extra); // attached by the game after load
  assets.release('tri');
  check('gltf dispose: geometry + material + parser texture freed', disposed.has('geometry') && disposed.has('material') && disposed.has('own texture'), 1, 1);
  check('gltf dispose: its ImageBitmap closed', bitmaps.length === 1 && bitmaps[0].closed, 1, 1);
  check('gltf dispose: envMap, foreign texture, user mesh untouched', !disposed.has('envMap') && !disposed.has('foreign') && !disposed.has('user mesh'), 1, 1);
  assets.add('draco', { type: 'gltf', url: 'm/draco.glb' });
  const rd = await settle(assets.load());
  check('gltf: Draco file without dracoPath rejects with a hint', !!rd.e && /dracoPath/.test(rd.e.message) && /"draco"/.test(rd.e.message), 1, 1);
  // disposeInstance: a skinned clone frees its own skeleton, nothing shared
  const bone = new THREE.Bone(), geo = new THREE.BufferGeometry(), m2 = new THREE.MeshBasicMaterial();
  geo.setAttribute('position', new THREE.Float32BufferAttribute([0, 0, 0, 1, 0, 0, 0, 1, 0], 3));
  geo.setAttribute('skinIndex', new THREE.Uint16BufferAttribute([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 4));
  geo.setAttribute('skinWeight', new THREE.Float32BufferAttribute([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], 4));
  const sk = new THREE.SkinnedMesh(geo, m2); sk.add(bone); sk.bind(new THREE.Skeleton([bone]));
  const src = new THREE.Group(); src.add(sk);
  const clone = cloneSkinned(src), csk = clone.children[0];
  csk.skeleton.computeBoneTexture(); sk.skeleton.computeBoneTexture();
  const gone = new Set();
  csk.skeleton.boneTexture.addEventListener('dispose', () => gone.add('clone bones'));
  sk.skeleton.boneTexture.addEventListener('dispose', () => gone.add('source bones'));
  geo.addEventListener('dispose', () => gone.add('geometry')); m2.addEventListener('dispose', () => gone.add('material'));
  disposeInstance(clone);
  check('disposeInstance: clone bone texture freed', gone.has('clone bones') && csk.skeleton !== sk.skeleton, 1, 1);
  check('disposeInstance: source skeleton, geometry, material kept', !gone.has('source bones') && !gone.has('geometry') && !gone.has('material'), 1, 1);
}

console.log(results.join('\n'));
console.log(failures ? `\n${failures} check(s) FAILED` : '\nAll asset checks passed.');
process.exit(failures ? 1 : 0);
