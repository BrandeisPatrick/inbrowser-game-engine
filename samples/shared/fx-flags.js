// 0.8.0 post-stack query flags shared by the samples (every one off by default; the page renders as 0.7.0 without them):
//   ?ssr=1               screen-space reflections (needs ?gbuffer=1)
//   ?volumetric=1        volumetric sun light (shadowed in-scatter through the atmosphere's haze / mist)
//   ?motionblur=<s>      motion blur with strength s (needs ?gbuffer=1), e.g. ?motionblur=1
//   ?dof=1[&focus=<m>&fstop=<f>]   depth of field (thin lens: focus distance m, f-number)
//   ?lut=<name>[&lutstrength=<0..1>]   3D LUT grading from samples/public/luts/<name>.cube (warm, bleach)
//   ?aonormals=gbuffer   GTAO on the G-buffer normal instead of depth-reconstructed normals (needs ?gbuffer=1)
// readFxFlags(params) -> the PostFX option fragment (ssr, volumetric, motionBlur, dof); aoFlags(params) -> the `ao`
// fragment ({ normals }) to spread into the sample's own ao options; loadLut(params) -> the `lut` option (or null) from
// the fetched .cube; fxSettings(() => postfx) -> bench-harness settings with get / set for every 0.8.0 knob through
// postfx.get / postfx.set, so __sample.run(plan) restores them between entries.
import { parseCubeLUT } from '@vibe/engine/render';

/** @param {URLSearchParams} params */
export function readFxFlags(params) {
  const num = (k) => { const v = params.get(k); return v == null || v === '' || !Number.isFinite(+v) ? null : +v; };
  const o = {};
  if (params.get('ssr') === '1') o.ssr = true;
  if (params.get('volumetric') === '1') o.volumetric = true;
  const mb = num('motionblur');
  if (mb) o.motionBlur = mb;
  if (params.get('dof') === '1') {
    o.dof = {};
    const f = num('focus'), s = num('fstop');
    if (f) o.dof.focusDistance = f;
    if (s) o.dof.fStop = s;
  }
  return o;
}

/** @param {URLSearchParams} params */
export function aoFlags(params) {
  const an = params.get('aonormals');
  return an ? { normals: an } : {};
}

/** Fetch + parse the `?lut=<name>` .cube (served from samples/public/luts); null without the flag. */
export async function loadLut(params, base = `${import.meta.env.BASE_URL}luts/`) {
  const name = params.get('lut');
  if (!name) return null;
  const res = await fetch(`${base}${name}.cube`);
  if (!res.ok) { console.warn(`[samples] LUT '${name}' not found (${res.status})`); return null; }
  const lut = parseCubeLUT(await res.text());
  const s = params.get('lutstrength');
  return { texture: lut.texture, size: lut.size, strength: s == null || s === '' ? 1 : +s };
}

export const FX_PATHS = [
  'ssr', 'ssr.scale', 'ssr.maxPixels', 'ssr.steps', 'ssr.refine', 'ssr.maxDistance', 'ssr.thickness', 'ssr.maxRoughness', 'ssr.jitter', 'ssr.metalTint',
  'volumetric', 'volumetric.scale', 'volumetric.maxPixels', 'volumetric.steps', 'volumetric.maxDistance', 'volumetric.temporal', 'volumetric.jitter', 'volumetric.blend', 'volumetric.intensity',
  'motionBlur', 'motionBlur.samples', 'motionBlur.tileSize', 'motionBlur.maxBlur', 'motionBlur.shutter',
  'dof', 'dof.scale', 'dof.samples', 'dof.nearBlur', 'dof.stableDepth', 'dof.maxCoC', 'dof.fStop', 'dof.focusDistance',
  'lut', 'lut.strength', 'ao.normals',
];

/** Harness settings `postfx.<path>` for every 0.8.0 knob. @param {() => (import('@vibe/engine/render').PostFX | null)} fx */
export function fxSettings(fx) {
  const s = {};
  for (const p of FX_PATHS) {
    s[`postfx.${p}`] = {
      get: () => { const f = fx(); return f ? f.get(p) : null; },
      set: (v) => { const f = fx(); if (f) f.set(p, v); },
    };
  }
  return s;
}
