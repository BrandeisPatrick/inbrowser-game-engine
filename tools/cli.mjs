// Tiny shared CLI helpers for the engine bakers (no npm deps): flag parsing with a spec, --config JSON files,
// --help text, and the "run only when executed directly" guard.
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

/** True when the module at `metaUrl` is the script node was started with. */
export function isMain(metaUrl) {
  return !!process.argv[1] && metaUrl === pathToFileURL(path.resolve(process.argv[1])).href;
}

const camel = (s) => s.replace(/-([a-z])/g, (_, c) => c.toUpperCase());
const conv = (type, v) => {
  if (type === 'number') { const n = Number(v); if (!Number.isFinite(n)) throw new Error(`not a number: ${v}`); return n; }
  if (type === 'numbers') return String(v).split(',').map((x) => { const n = Number(x); if (!Number.isFinite(n)) throw new Error(`not a number list: ${v}`); return n; });
  if (type === 'bool') return v === true || v === 'true' || v === '1';
  return String(v);
};

/**
 * Parse argv against spec { 'flag-name': { type: 'number'|'numbers'|'string'|'bool', def, help } }.
 * Supports --flag value, --flag=value, boolean --flag, --config <file.json> (same keys, camelCase or kebab-case;
 * flags override it) and -h / --help. Returns { opts (camelCase keys), help: boolean }.
 */
export function parseArgs(argv, spec) {
  const out = {}, cli = {};
  let help = false, config = null;
  for (let i = 0; i < argv.length; i++) {
    let a = argv[i];
    if (a === '-h' || a === '--help') { help = true; continue; }
    if (!a.startsWith('--')) throw new Error(`unexpected argument: ${a}`);
    a = a.slice(2);
    let v; const eq = a.indexOf('=');
    if (eq >= 0) { v = a.slice(eq + 1); a = a.slice(0, eq); }
    if (a === 'config') { config = v ?? argv[++i]; continue; }
    const s = spec[a];
    if (!s) throw new Error(`unknown flag --${a} (see --help)`);
    if (s.type === 'bool') cli[camel(a)] = v === undefined ? true : conv('bool', v);
    else cli[camel(a)] = conv(s.type, v ?? argv[++i]);
  }
  for (const [k, s] of Object.entries(spec)) if (s.def !== undefined) out[camel(k)] = s.def;
  if (config) {
    const j = JSON.parse(fs.readFileSync(config, 'utf8'));
    for (const [k, v] of Object.entries(j)) { const key = camel(k), s = spec[k] || spec[k.replace(/[A-Z]/g, (c) => '-' + c.toLowerCase())]; out[key] = s && typeof v === 'string' && s.type !== 'string' ? conv(s.type, v) : v; }
  }
  Object.assign(out, cli);
  return { opts: out, help };
}

/** Usage text from a spec. */
export function usage(name, summary, spec) {
  const rows = Object.entries(spec).map(([k, s]) => {
    const flag = `--${k}${s.type === 'bool' ? '' : s.type === 'numbers' ? ' a,b,..' : ` <${s.type}>`}`;
    const def = s.def === undefined ? '' : ` (default: ${Array.isArray(s.def) ? s.def.join(',') : s.def})`;
    return `  ${flag.padEnd(30)} ${s.help || ''}${def}`;
  });
  return `${name}: ${summary}\n\nUsage: node ${name} [flags]\n\n${rows.join('\n')}\n  ${'--config <file.json>'.padEnd(30)} JSON object with the same keys (flags override it)\n  ${'-h, --help'.padEnd(30)} this text\n`;
}

/** Run an async CLI main with --help handling and a non-zero exit code on errors. */
export function runCli(metaUrl, name, summary, spec, fn) {
  if (!isMain(metaUrl)) return;
  let parsed;
  try { parsed = parseArgs(process.argv.slice(2), spec); } catch (e) { console.error(`${name}: ${e.message}`); process.exit(2); }
  if (parsed.help) { process.stdout.write(usage(name, summary, spec)); return; }
  fn(parsed.opts).catch((e) => { console.error(e); process.exit(1); });
}
