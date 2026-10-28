// Procedural texture-array layer worker (module worker; a small pool created by TextureSets.js with
//   new Worker(new URL('./proceduralTextures.worker.js', import.meta.url), { type: 'module' })  -> bundled by Vite).
// Runs the same proceduralTexels() as the main-thread path (proceduralPbr.js: pure JS on noise.js, no three.js), so the
// texels are byte-identical; the main thread only tiles the returned 256 px texels into its RGBA8 arrays.
//   in : { id, kind, P, seed, tint }
//   out: { id, pa, pn (P x P RGBA8, buffers transferred), ms (generation time in this worker) } | { id, error }
import { proceduralTexels } from './proceduralPbr.js';

self.onmessage = (e) => {
  const m = e.data;
  try {
    const t = performance.now();
    const { pa, pn } = proceduralTexels(m.kind, m.P, m.seed, m.tint);
    self.postMessage({ id: m.id, pa, pn, ms: performance.now() - t }, [pa.buffer, pn.buffer]);
  } catch (err) {
    self.postMessage({ id: m.id, error: String(err && err.message ? err.message : err) });
  }
};
