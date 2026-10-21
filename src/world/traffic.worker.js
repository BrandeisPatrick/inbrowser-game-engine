// Traffic simulation Web Worker (module worker; created by Traffic.js with
//   new Worker(new URL('./traffic.worker.js', import.meta.url), { type: 'module' })  -> bundled by Vite).
// Runs the same TrafficSim core as the main thread; never sees the terrain (ground heights arrive baked).
//   in : { type: 'init', roads, signals, heights, index, spawn, seed, count }            -> { type: 'ready', buf, t, buildMs, lanes }
//        { type: 'step', dt, seq, view: f32[3 + 16], hasPv, player: f32[9], hasPlayer, sig: u8[], bufs: [ArrayBuffer] }
//   out: { type: 'poses', buf (transferred), t, seq, stepMs, summary }  - one answer per step
// Pose buffers (POSE_STRIDE floats per vehicle) are double-buffered: the main thread returns the snapshot it no longer
// draws with its next step message.
import { TrafficSim, POSE_STRIDE } from './TrafficSim.js';

let sim = null, n = 0;
const free = [];
const player = { x: 0, y: 0, z: 0, fx: 0, fz: -1, hl: 2.3, hw: 1, rx: 0, rz: 0 };

self.onmessage = (ev) => {
  const m = ev.data;
  if (m.type === 'step') {
    if (!sim) return;
    if (m.bufs) for (const b of m.bufs) if (b.byteLength === n * 4) free.push(new Float32Array(b));
    sim.sigState.set(m.sig.length === sim.sigState.length ? m.sig : m.sig.subarray(0, sim.sigState.length));
    const v = m.view;
    sim.setView(v[0], v[1], v[2], m.hasPv ? v.subarray(3, 19) : null);
    if (m.hasPlayer) {
      const p = m.player;
      player.x = p[0]; player.y = p[1]; player.z = p[2]; player.fx = p[3]; player.fz = p[4]; player.hl = p[5]; player.hw = p[6]; player.rx = p[7]; player.rz = p[8];
      sim.setPlayer(player);
    } else sim.setPlayer(null);
    const t0 = performance.now();
    sim.update(m.dt);
    const out = free.pop() || new Float32Array(n);
    sim.writePoses(out);
    const stepMs = performance.now() - t0;
    self.postMessage({ type: 'poses', buf: out.buffer, t: sim.time, seq: m.seq, stepMs, summary: sim.summary() }, [out.buffer]);
    return;
  }
  if (m.type === 'init') {
    try {
      const t0 = performance.now();
      sim = new TrafficSim({ roads: m.roads, signals: m.signals, baked: { heights: m.heights, index: m.index }, spawn: m.spawn, seed: m.seed, count: m.count });
      sim.build();
      n = sim.vehicles.length * POSE_STRIDE;
      free.push(new Float32Array(n));
      const first = sim.writePoses(new Float32Array(n));
      self.postMessage({ type: 'ready', buf: first.buffer, t: sim.time, buildMs: Math.round(performance.now() - t0), lanes: sim.lanes.length }, [first.buffer]);
    } catch (e) {
      self.postMessage({ type: 'error', message: String((e && e.stack) || e) });
    }
  }
};
