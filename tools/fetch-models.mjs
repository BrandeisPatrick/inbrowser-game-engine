#!/usr/bin/env node
// Download the CC0 sample models used by samples/character (RobotExpressive.glb from the three.js r180 examples)
// into <out>/ plus <out>/CREDITS.md. The models are never committed (samples/character/models/.gitignore ignores
// everything but itself) and the sample runs without them (procedural rig fallback), so this is optional.
//   node tools/fetch-models.mjs                    (skips files already present and valid)
//   node tools/fetch-models.mjs --refresh  (re-download)   --dry-run (list only)   --out <dir>   --help
//   export async function fetchModels(opts)        (same options, camelCase)
// Every download is checked: HTTP 200 and a glTF binary header whose length field matches the byte count.
import fs from 'node:fs/promises';
import path from 'node:path';
import { runCli } from './cli.mjs';

/** Models fetched by default: id, local file, source URL, licence line (printed and written to CREDITS.md). */
export const MODELS = [
  {
    id: 'RobotExpressive', file: 'RobotExpressive.glb',
    url: 'https://raw.githubusercontent.com/mrdoob/three.js/r180/examples/models/gltf/RobotExpressive/RobotExpressive.glb',
    credit: 'RobotExpressive by Tomás Laulhé (modifications by Don McCurdy), CC0 — https://quaternius.com',
    clips: 'Idle, Walking, Running, Jump, WalkJump, Dance, Death, No, Punch, Sitting, Standing, ThumbsUp, Wave, Yes',
  },
];

/** CLI flags / fetchModels options. */
export const SPEC = {
  out: { type: 'string', def: 'samples/character/models', help: 'output directory' },
  ua: { type: 'string', def: 'vibe-engine/1.0 (sample model fetch; CC0 assets)', help: 'HTTP User-Agent' },
  refresh: { type: 'bool', def: false, help: 're-download everything' },
  'dry-run': { type: 'bool', def: false, help: 'print the model list and paths only (no network, no files)' },
};

/** Byte length declared by a GLB header ('glTF' magic, version 2), or -1 when buf is not a GLB. */
function glbLength(buf) {
  if (!buf || buf.length < 12 || buf.readUInt32LE(0) !== 0x46546c67 || buf.readUInt32LE(4) !== 2) return -1;
  return buf.readUInt32LE(8);
}
const validGlb = (buf) => glbLength(buf) === buf.length;

async function download(url, file, { refresh, headers }, tries = 3) {
  if (!refresh) {
    try { const b = await fs.readFile(file); if (validGlb(b)) return { bytes: b.length, cached: true }; } catch { /* missing */ }
  }
  for (let i = 0; ; i++) {
    try {
      const r = await fetch(url, { headers });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const buf = Buffer.from(await r.arrayBuffer());
      if (!validGlb(buf)) throw new Error(`not a complete glTF binary (${buf.length} bytes, header says ${glbLength(buf)})`);
      await fs.writeFile(file, buf);
      return { bytes: buf.length, cached: false };
    } catch (e) {
      if (i + 1 >= tries) throw new Error(`${url}: ${e.message}`);
      await new Promise((res) => setTimeout(res, 800 * (i + 1)));
    }
  }
}

/** Fetch the models. opts: SPEC keys in camelCase (out, ua, refresh, dryRun) + models (default MODELS). */
export async function fetchModels(opts = {}) {
  const d = (k) => (opts[k] !== undefined ? opts[k] : SPEC[k].def);
  const out = path.resolve(d('out')), refresh = !!d('refresh'), models = opts.models || MODELS;
  const rel = path.relative(process.cwd(), out) || '.';
  if (opts.dryRun) {
    console.log(`models plan: ${models.length} model(s) -> ${rel}${refresh ? ' (refresh)' : ''}`);
    for (const m of models) console.log(`  ${m.file.padEnd(22)} ${m.url}\n  ${''.padEnd(22)} ${m.credit}`);
    console.log('  dry run: no requests made, nothing written');
    return { dryRun: true, out, models: models.length };
  }
  await fs.mkdir(out, { recursive: true });
  // never commit downloaded models: a folder without its own .gitignore gets one (ignore everything but itself)
  const gi = path.join(out, '.gitignore');
  try { await fs.access(gi); } catch { await fs.writeFile(gi, '*\n!.gitignore\n'); }
  const headers = { 'User-Agent': d('ua') };
  let total = 0, fetched = 0;
  const done = [];
  for (const m of models) {
    try {
      const r = await download(m.url, path.join(out, m.file), { refresh, headers });
      total += r.bytes; if (!r.cached) fetched += r.bytes;
      done.push(m);
      console.log(`${m.file.padEnd(22)} ${(r.bytes / 1048576).toFixed(2)} MB${r.cached ? ' (cached)' : ''}\n  ${m.credit}`);
    } catch (e) { console.warn(`! ${m.id}: ${e.message}`); }
  }
  const lines = [
    '# Model credits',
    '',
    'Downloaded by `tools/fetch-models.mjs` for the character sample (samples/character). Not committed to the',
    'repository: this folder\'s `.gitignore` ignores everything but itself. The sample falls back to its procedural',
    'rig when a model is missing.',
    '',
    '| File | Licence / author | Source | Clips |',
    '|---|---|---|---|',
    ...done.map((m) => `| ${m.file} | ${m.credit} | <${m.url}> | ${m.clips || ''} |`),
    '',
  ];
  if (done.length) await fs.writeFile(path.join(out, 'CREDITS.md'), lines.join('\n'));
  console.log(`\n${done.length}/${models.length} model(s), ${(total / 1048576).toFixed(2)} MB on disk (${(fetched / 1048576).toFixed(2)} MB downloaded now) -> ${rel}`);
  if (done.length < models.length) throw new Error(`${models.length - done.length} model(s) failed`);
  return { out, models: done.length, bytes: total };
}

runCli(import.meta.url, 'tools/fetch-models.mjs', 'download the CC0 sample character model (RobotExpressive.glb)', SPEC, fetchModels);
