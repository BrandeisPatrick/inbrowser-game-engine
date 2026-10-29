// Runs every headless suite in sequence (`npm test`): each file is a standalone Node script that exits non-zero on
// failure; suites that build world geometry need the no-op canvas shim. Add new suites here, keep car-sim first
// (it is the determinism gate for the roadster vehicle model).
import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const SHIM = here('./helpers/dom-shim.mjs');
const SUITES = [
  { file: 'car-sim.mjs' },
  { file: 'world-build.mjs', shim: true },
  { file: 'core.mjs' },
  { file: 'assets.mjs' },
  { file: 'rigs.mjs' },
  { file: 'character-input.mjs' },
  { file: 'anim.mjs' },
  { file: 'character-sim.mjs' },
  // dynamics (0.6.0): shapes -> broadphase -> narrowphase -> xpbd/joints -> world -> queries/bridge -> worker protocol
  { file: 'dyn-shapes.mjs' },
  { file: 'dyn-broadphase.mjs' },
  { file: 'dyn-narrowphase.mjs' },
  { file: 'dyn-joints.mjs' },
  { file: 'dyn-world.mjs' },
  { file: 'dyn-bridge.mjs', shim: true },
  { file: 'dyn-serialize.mjs' },
];
let failed = 0, ran = 0;
const t0 = performance.now();
for (const s of SUITES) {
  const file = here(`./${s.file}`);
  if (!existsSync(file)) { console.log(`skip  ${s.file} (missing)`); continue; }
  const args = s.shim ? ['--import', SHIM, file] : [file];
  const t = performance.now();
  const r = spawnSync(process.execPath, args, { stdio: 'inherit' });
  ran++;
  const ms = Math.round(performance.now() - t);
  if (r.status !== 0) { failed++; console.log(`FAIL  ${s.file} (exit ${r.status}, ${ms} ms)`); } else console.log(`ok    ${s.file} (${ms} ms)`);
}
console.log(`\n${ran - failed}/${ran} suites passed in ${Math.round(performance.now() - t0)} ms`);
process.exit(failed ? 1 : 0);
