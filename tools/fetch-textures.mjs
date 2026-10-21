#!/usr/bin/env node
// Download CC0 PBR texture sets (Poly Haven, 1K JPG by default) used by the engine world layer into
// <out>/<id>/{diff,nor_gl,rough,ao}.jpg plus <out>/manifest.json (real-world size of each set, so the world can
// tile at true scale) and <out>/CREDITS.md.
//   node engine/tools/fetch-textures.mjs --sets textures.json --out public/textures   (skips files already present)
//   node engine/tools/fetch-textures.mjs --refresh  (re-download everything)   --dry-run (list only)   --help
//   node engine/tools/fetch-textures.mjs --ktx2     (then compress to .ktx2 with compress-textures.mjs; needs its npm deps)
//   export async function fetchTextures(opts)       (same options, camelCase; used by the game's tools/fetch-textures.mjs)
// manifest.json is rewritten on every run; the `ktx2` / `ktx2Maps` fields of compress-textures.mjs are carried over for
// sets whose JPGs did not change (same byte size), so a plain re-fetch keeps the compressed twins in use.
// The engine runs without these files (procedural canvas fallback), so this is optional but recommended.
import fsSync from 'node:fs';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { runCli } from './cli.mjs';

/** CLI flags / fetchTextures options. */
export const SPEC = {
  sets: { type: 'string', def: fileURLToPath(new URL('./textures.default.json', import.meta.url)), help: 'JSON list of { role, id } texture sets' },
  out: { type: 'string', def: 'public/textures', help: 'output directory' },
  res: { type: 'string', def: '1k', help: 'Poly Haven resolution' },
  ua: { type: 'string', def: 'roadster-flying-car/1.0 (texture fetch; CC0 assets)', help: 'HTTP User-Agent' },
  refresh: { type: 'bool', def: false, help: 're-download everything' },
  ktx2: { type: 'bool', def: false, help: 'compress to .ktx2 afterwards (engine/tools/compress-textures.mjs defaults)' },
  'dry-run': { type: 'bool', def: false, help: 'print the set list and paths only (no network, no files)' },
};
let ROOT, OUT, REFRESH, RES, UA, SETS;
const API = 'https://api.polyhaven.com';
function configure(o = {}) {
  const d = (k) => (o[k] !== undefined ? o[k] : SPEC[k].def);
  ROOT = process.cwd(); OUT = path.resolve(d('out')); REFRESH = !!d('refresh'); RES = d('res');
  UA = { 'User-Agent': d('ua') };
  SETS = Array.isArray(o.sets) ? o.sets : JSON.parse(fsSync.readFileSync(path.resolve(d('sets')), 'utf8'));
}

// role -> Poly Haven id list: --sets <file.json> ([{ role, id }]; default textures.default.json next to this tool).
// Roles are what the engine looks up (engine/src/world/FacadeTextures.js PBR_SETS / GROUND_SETS, Shore.js textures).
const MAPS = [ // API key -> local file name
  ['Diffuse', 'diff.jpg'],
  ['nor_gl', 'nor_gl.jpg'],
  ['Rough', 'rough.jpg'],
  ['AO', 'ao.jpg'],
];

async function getJSON(url, tries = 3) {
  for (let i = 0; ; i++) {
    try {
      const r = await fetch(url, { headers: UA });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return await r.json();
    } catch (e) {
      if (i + 1 >= tries) throw new Error(`${url}: ${e.message}`);
      await new Promise((res) => setTimeout(res, 800 * (i + 1)));
    }
  }
}
async function download(url, file, expectSize, tries = 3) {
  if (!REFRESH) {
    try { const st = await fs.stat(file); if (st.size > 0 && (!expectSize || st.size === expectSize)) return { bytes: st.size, cached: true }; } catch { /* missing */ }
  }
  for (let i = 0; ; i++) {
    try {
      const r = await fetch(url, { headers: UA });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const buf = Buffer.from(await r.arrayBuffer());
      if (expectSize && buf.length !== expectSize) throw new Error(`size ${buf.length} != ${expectSize}`);
      await fs.writeFile(file, buf);
      return { bytes: buf.length, cached: false };
    } catch (e) {
      if (i + 1 >= tries) throw new Error(`${url}: ${e.message}`);
      await new Promise((res) => setTimeout(res, 800 * (i + 1)));
    }
  }
}

/** Fetch the sets. opts: SPEC keys in camelCase (sets: file path or array, out, res, ua, refresh, dryRun). */
export async function fetchTextures(opts = {}) {
  configure(opts);
  if (opts.dryRun) {
    console.log(`textures plan: ${SETS.length} sets at ${RES} from ${API} -> ${OUT}${REFRESH ? ' (refresh)' : ''}`);
    for (const s of SETS) console.log(`  ${s.role.padEnd(17)} ${s.id}`);
    console.log('  dry run: no requests made, nothing written');
    return { dryRun: true, out: OUT, sets: SETS.length };
  }
  await fs.mkdir(OUT, { recursive: true });
  let prev = null;
  try { prev = JSON.parse(await fs.readFile(path.join(OUT, 'manifest.json'), 'utf8')); } catch { /* first run */ }
  const manifest = { source: 'Poly Haven (polyhaven.com), CC0 1.0', resolution: RES, sets: {} };
  let total = 0, fetched = 0;
  for (const s of SETS) {
    const dir = path.join(OUT, s.id);
    await fs.mkdir(dir, { recursive: true });
    let files, info;
    try {
      [files, info] = await Promise.all([getJSON(`${API}/files/${s.id}`), getJSON(`${API}/info/${s.id}`)]);
    } catch (e) { console.warn(`! ${s.id}: ${e.message}`); continue; }
    const entry = { role: s.role, id: s.id, name: info.name || s.id, size: (info.dimensions || [2000, 2000]).map((v) => +(v / 1000).toFixed(3)), authors: Object.keys(info.authors || {}), maps: {} };
    const jobs = MAPS.map(async ([key, name]) => {
      const f = files[key] && files[key][RES] && files[key][RES].jpg;
      if (!f || !f.url) return;
      const r = await download(f.url, path.join(dir, name), f.size);
      entry.maps[key] = name; total += r.bytes; if (!r.cached) fetched += r.bytes;
    });
    try { await Promise.all(jobs); } catch (e) { console.warn(`! ${s.id}: ${e.message}`); }
    // keep the compressed twins when their source JPGs are unchanged (compress-textures records srcBytes per map)
    const old = prev && prev.sets && prev.sets[s.role];
    if (old && old.id === s.id && old.ktx2Maps) {
      const keep = {};
      for (const [k, m] of Object.entries(old.ktx2Maps)) {
        try { if (entry.maps[k] && m.srcBytes === fsSync.statSync(path.join(dir, entry.maps[k])).size && fsSync.existsSync(path.join(dir, m.file))) keep[k] = m; } catch { /* dropped */ }
      }
      if (Object.keys(keep).length) { entry.ktx2Maps = keep; entry.ktx2 = Object.keys(entry.maps).every((k) => keep[k]); }
    }
    manifest.sets[s.role] = entry;
    console.log(`${s.role.padEnd(17)} ${s.id.padEnd(22)} ${entry.size.join('x')} m  [${Object.keys(entry.maps).join(', ')}]`);
  }
  if (prev && prev.ktx2) manifest.ktx2 = prev.ktx2;
  await fs.writeFile(path.join(OUT, 'manifest.json'), JSON.stringify(manifest, null, 1));
  const lines = [
    '# Texture credits',
    '',
    'All PBR texture sets in this folder are from **[Poly Haven](https://polyhaven.com)** and are released under',
    '**CC0 1.0 Universal (public domain)**: <https://creativecommons.org/publicdomain/zero/1.0/>. No attribution is',
    'required; it is given here anyway. Files are the 1K JPG maps (diffuse, OpenGL normal, roughness, AO),',
    'downloaded by `tools/fetch-textures.mjs`. The `.ktx2` files next to them are GPU-compressed (Basis Universal)',
    'copies of the same maps made by `engine/tools/compress-textures.mjs`.',
    '',
    '| Role in game | Poly Haven id | Name | Real-world size | Author(s) |',
    '|---|---|---|---|---|',
    ...Object.values(manifest.sets).map((e) => `| ${e.role} | [${e.id}](https://polyhaven.com/a/${e.id}) | ${e.name} | ${e.size.join(' x ')} m | ${e.authors.join(', ')} |`),
    '',
  ];
  await fs.writeFile(path.join(OUT, 'CREDITS.md'), lines.join('\n'));
  console.log(`\n${Object.keys(manifest.sets).length} sets, ${(total / 1048576).toFixed(1)} MB on disk (${(fetched / 1048576).toFixed(1)} MB downloaded now) -> ${path.relative(ROOT, OUT)}`);
  if (opts.ktx2) {
    const { compressTextures } = await import('./compress-textures.mjs');
    await compressTextures({ ...(opts.ktx2Options || {}), dir: OUT });
  }
  return { out: OUT, sets: Object.keys(manifest.sets).length };
}

runCli(import.meta.url, 'engine/tools/fetch-textures.mjs', 'download CC0 PBR texture sets from Poly Haven', SPEC, fetchTextures);
