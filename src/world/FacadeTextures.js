// Game texture content. The generic machinery (canvas painters, manifest-driven PBR set loader with procedural
// fallbacks, DataArrayTexture packing) lives in the engine: engine/src/render/TextureSets.js. What stays here is
// building-specific: the facade window-style layers (FACADE / WIN_RECTS, drawn procedurally) and the table of
// PBR material roles the building shader indexes (PBR / PBR_SETS / GROUND_SETS).
import { mulberry32 } from './geom.js';
import {
  makeCanvas, ctx2d, makeDataArrayTexture, makeTextureArray, loadTextureSet, loadTextureManifest, textureBase,
  makeAsphaltTexture, makePavingTexture, makeGroundTexture, makeGrassTexture, makeWaterNormalTexture,
} from '../render/TextureSets.js';

// generic tileable painters (re-exported for existing importers)
export { makeAsphaltTexture, makePavingTexture, makeGroundTexture, makeGrassTexture, makeWaterNormalTexture };

// ------------------------------------------------------------------ window-style layers (DataArrayTexture)
// Each layer = 4 bays x 4 floors, 512 px (128 px per bay / floor). Texture row 0 is the BOTTOM.
// Walls themselves come from the PBR array below; these layers only describe the openings:
// albedo: sRGB colour of trims / frames. data: R = 0 wall (PBR) | 128 trim (layer albedo) | 255 window glass,
//                                           G = roughness, B = metalness.
// WIN_RECTS gives the window openings of each layer in cell fractions [x0, x1, y0, y1] (y from the bottom of
// the floor) so buildings.js can cut real recessed openings that line up with the drawn frames.
export const FACADE = {
  PUNCHED: 0, PAIR: 1, CLASSIC: 2, RIBBON: 3, VERTICAL: 4, INDUSTRIAL: 5,
  GARAGE: 6, MODERN_RES: 7, STOREFRONT: 8, VICTORIAN: 9, OFFICE_MODERN: 10, SMALL: 11,
};
export const WIN_RECTS = [
  [[0.16, 0.84, 0.22, 0.84]],                          // PUNCHED
  [[0.1, 0.45, 0.18, 0.86], [0.55, 0.9, 0.18, 0.86]],  // PAIR
  [[0.18, 0.82, 0.06, 0.92]],                          // CLASSIC
  [[0, 1, 0.26, 0.76]],                                // RIBBON (continuous along the wall)
  [[0.3, 0.7, 0, 1]],                                  // VERTICAL (continuous up the wall)
  [[0.06, 0.94, 0.45, 0.86]],                          // INDUSTRIAL
  [[0.04, 0.96, 0.22, 0.9]],                           // GARAGE (open, no glass)
  [[0.04, 0.96, 0.06, 0.94]],                          // MODERN_RES
  [[0.03, 0.97, 0.08, 0.8]],                           // STOREFRONT
  [[0.28, 0.72, 0.12, 0.86]],                          // VICTORIAN
  [[0.03, 0.97, 0.28, 0.95]],                          // OFFICE_MODERN
  [[0.25, 0.75, 0.22, 0.84]],                          // SMALL
];
const LAYERS = 12, CS = 512, B = 128;

function rgb(r, g, b) { return `rgb(${r | 0},${g | 0},${b | 0})`; }
function D(kind, rough, metal) { return rgb(kind === 'w' ? 255 : kind === 't' ? 128 : 0, rough * 255, metal * 255); }
function eachCell(fn) { for (let f = 0; f < 4; f++) for (let b = 0; b < 4; b++) fn(b * B, f * B, b, f); }
// rect in cell fractions (bottom origin) -> canvas px of cell (cx, cy = top-left of the cell in canvas coords)
function cr(cx, cy, x0, x1, y0, y1) { return [cx + x0 * B, cy + (1 - y1) * B, (x1 - x0) * B, (y1 - y0) * B]; }
function rect(ga, gd, r, col, kind, rough, metal) {
  ga.fillStyle = rgb(col[0], col[1], col[2]); ga.fillRect(r[0], r[1], r[2], r[3]);
  gd.fillStyle = D(kind, rough, metal); gd.fillRect(r[0], r[1], r[2], r[3]);
}
function glass(ga, gd, r) { rect(ga, gd, r, [30, 36, 42], 'w', 0.05, 0); }
/** frame: border of width w (px) in trim colour around a glass rect, plus optional mullions. */
function framed(ga, gd, r, col, w, rough, metal, { vbars = 0, hbars = [] } = {}) {
  rect(ga, gd, r, col, 't', rough, metal);
  glass(ga, gd, [r[0] + w, r[1] + w, r[2] - 2 * w, r[3] - 2 * w]);
  for (let i = 1; i <= vbars; i++) rect(ga, gd, [r[0] + (r[2] * i) / (vbars + 1) - w / 2, r[1], w, r[3]], col, 't', rough, metal);
  for (const f of hbars) rect(ga, gd, [r[0], r[1] + r[3] * (1 - f) - w / 2, r[2], w], col, 't', rough, metal);
}

const LAYER_DRAW = [
  // 0 punched office window: dark bronze frame, transom, stone sill
  (ga, gd) => eachCell((x, y) => {
    framed(ga, gd, cr(x, y, 0.16, 0.84, 0.22, 0.84), [58, 52, 46], 4, 0.4, 0.6, { vbars: 1, hbars: [0.78] });
    rect(ga, gd, cr(x, y, 0.13, 0.87, 0.17, 0.22), [196, 190, 178], 't', 0.8, 0);
  }),
  // 1 pairs of double-hung windows (residential / hotel), white frames
  (ga, gd) => eachCell((x, y) => {
    for (const [a, b] of [[0.1, 0.45], [0.55, 0.9]]) {
      framed(ga, gd, cr(x, y, a, b, 0.18, 0.86), [236, 234, 228], 4, 0.6, 0, { hbars: [0.5] });
      rect(ga, gd, cr(x, y, a - 0.03, b + 0.03, 0.14, 0.18), [214, 208, 196], 't', 0.8, 0);
    }
  }),
  // 2 classic: tall windows between piers, bronze spandrel panels
  (ga, gd) => eachCell((x, y) => {
    framed(ga, gd, cr(x, y, 0.18, 0.82, 0.06, 0.92), [72, 60, 48], 3, 0.4, 0.6, { vbars: 1, hbars: [0.3] });
    rect(ga, gd, cr(x, y, 0.18, 0.82, 0.92, 1), [84, 70, 56], 't', 0.5, 0.6);
    rect(ga, gd, cr(x, y, 0.18, 0.82, 0, 0.06), [84, 70, 56], 't', 0.5, 0.6);
  }),
  // 3 ribbon windows (continuous)
  (ga, gd) => {
    for (let f = 0; f < 4; f++) {
      const y = f * B;
      glass(ga, gd, [0, y + (1 - 0.76) * B, CS, 0.5 * B]);
      for (let b = 0; b < 8; b++) rect(ga, gd, [b * 64 - 2, y + (1 - 0.76) * B, 4, 0.5 * B], [70, 72, 74], 't', 0.35, 0.8);
      rect(ga, gd, [0, y + (1 - 0.76) * B, CS, 3], [70, 72, 74], 't', 0.35, 0.8);
      rect(ga, gd, [0, y + (1 - 0.26) * B - 3, CS, 3], [70, 72, 74], 't', 0.35, 0.8);
    }
  },
  // 4 vertical slots (continuous), dark recessed spandrel panel at each floor line
  (ga, gd) => eachCell((x, y) => {
    glass(ga, gd, cr(x, y, 0.3, 0.7, 0, 1));
    rect(ga, gd, cr(x, y, 0.3, 0.7, 0, 0.2), [52, 50, 48], 't', 0.5, 0.3);
    rect(ga, gd, cr(x, y, 0.3, 0.32, 0, 1), [60, 60, 62], 't', 0.4, 0.7);
    rect(ga, gd, cr(x, y, 0.68, 0.7, 0, 1), [60, 60, 62], 't', 0.4, 0.7);
  }),
  // 5 industrial steel multi-pane windows
  (ga, gd) => eachCell((x, y) => {
    const r = cr(x, y, 0.06, 0.94, 0.45, 0.86);
    framed(ga, gd, r, [48, 52, 50], 3, 0.5, 0.6);
    for (let i = 1; i < 7; i++) rect(ga, gd, [r[0] + (r[2] * i) / 7 - 1, r[1], 2, r[3]], [48, 52, 50], 't', 0.5, 0.6);
    for (let i = 1; i < 3; i++) rect(ga, gd, [r[0], r[1] + (r[3] * i) / 3 - 1, r[2], 2], [48, 52, 50], 't', 0.5, 0.6);
  }),
  // 6 parking garage: open dark slots
  (ga, gd) => eachCell((x, y) => {
    rect(ga, gd, cr(x, y, 0.04, 0.96, 0.22, 0.9), [16, 17, 19], 't', 0.95, 0);
    rect(ga, gd, cr(x, y, 0.04, 0.96, 0.84, 0.9), [40, 41, 44], 't', 0.95, 0);
  }),
  // 7 modern residential: floor-to-ceiling glass, rail bars in the lower part
  (ga, gd) => eachCell((x, y) => {
    framed(ga, gd, cr(x, y, 0.04, 0.96, 0.06, 0.94), [82, 86, 90], 3, 0.3, 0.8, { vbars: 1 });
    rect(ga, gd, cr(x, y, 0.04, 0.96, 0.33, 0.35), [190, 194, 198], 't', 0.3, 0.9);
  }),
  // 8 storefront: big glass, bulkhead, door in one bay of four
  (ga, gd) => eachCell((x, y, b) => {
    framed(ga, gd, cr(x, y, 0.03, 0.97, 0.08, 0.8), [34, 34, 36], 3, 0.4, 0.7, b === 1 ? {} : { vbars: 1 });
    rect(ga, gd, cr(x, y, 0, 1, 0, 0.08), [48, 46, 44], 't', 0.6, 0.2);
    if (b === 1) framed(ga, gd, cr(x, y, 0.36, 0.64, 0.08, 0.72), [26, 26, 28], 4, 0.4, 0.7, { hbars: [0.45] });
  }),
  // 9 Victorian double-hung with white surround and crown
  (ga, gd) => eachCell((x, y) => {
    rect(ga, gd, cr(x, y, 0.22, 0.78, 0.08, 0.92), [238, 236, 230], 't', 0.6, 0);
    rect(ga, gd, cr(x, y, 0.19, 0.81, 0.92, 0.97), [238, 236, 230], 't', 0.6, 0);
    framed(ga, gd, cr(x, y, 0.28, 0.72, 0.12, 0.86), [238, 236, 230], 3, 0.6, 0, { hbars: [0.5] });
  }),
  // 10 modern office: wide windows with thin aluminium frames
  (ga, gd) => eachCell((x, y) => {
    framed(ga, gd, cr(x, y, 0.03, 0.97, 0.28, 0.95), [150, 154, 158], 3, 0.3, 0.9, { vbars: 1 });
  }),
  // 11 small single windows
  (ga, gd) => eachCell((x, y) => {
    framed(ga, gd, cr(x, y, 0.25, 0.75, 0.22, 0.84), [220, 218, 212], 3, 0.6, 0, { hbars: [0.5] });
    rect(ga, gd, cr(x, y, 0.22, 0.78, 0.18, 0.22), [200, 196, 186], 't', 0.8, 0);
  }),
];

const arrayTex = makeDataArrayTexture;

export function makeFacadeTextures(renderer) {
  const ca = makeCanvas(CS), cd = makeCanvas(CS);
  const ga = ctx2d(ca), gd = ctx2d(cd);
  const alb = new Uint8Array(CS * CS * 4 * LAYERS);
  const dat = new Uint8Array(CS * CS * 4 * LAYERS);
  for (let l = 0; l < LAYERS; l++) {
    ga.globalAlpha = 1; gd.globalAlpha = 1;
    ga.fillStyle = '#808080'; ga.fillRect(0, 0, CS, CS);
    gd.fillStyle = D(0, 0.8, 0); gd.fillRect(0, 0, CS, CS);
    LAYER_DRAW[l](ga, gd, mulberry32(1000 + l * 7919));
    const ia = ga.getImageData(0, 0, CS, CS).data, id = gd.getImageData(0, 0, CS, CS).data;
    const base = l * CS * CS * 4, rowB = CS * 4;
    for (let y = 0; y < CS; y++) { // flip: canvas top row -> texture top row (v = 1)
      const src = y * rowB, dst = base + (CS - 1 - y) * rowB;
      alb.set(ia.subarray(src, src + rowB), dst);
      dat.set(id.subarray(src, src + rowB), dst);
    }
  }
  return { albedo: arrayTex(alb, CS, LAYERS, true, renderer), data: arrayTex(dat, CS, LAYERS, false, renderer), layers: LAYERS };
}

// ------------------------------------------------------------------ PBR material sets (Poly Haven CC0, see public/textures)
// Facade / roof materials are packed into two DataArrayTextures sampled by the building shader:
//   A (sRGB):   rgb = albedo (tintable sets normalised to neutral grey so per-building colour can tint them), a = AO
//   N (linear): rg  = tangent-space normal xy (OpenGL convention, z reconstructed), b = roughness
// Every layer has a procedural fallback, so a missing file (offline, fetch-textures not run) never breaks the game.
export const PBR = {
  BRICK_RED: 0, BRICK_PAINTED: 1, BRICK_DARK: 2, CONCRETE_PANEL: 3, CONCRETE_ROUGH: 4, STUCCO_A: 5, STUCCO_B: 6,
  GRANITE: 7, WOOD_SIDING: 8, METAL: 9, ROOF_SLATE: 10, ROOF_TAR: 11, GRASS: 12, PLAIN: 13, FABRIC: 14,
};
export const PBR_SETS = [ // index-aligned with PBR; size = real-world tile size in metres (manifest.json overrides)
  { role: 'brick_red', id: 'red_bricks_04', size: 2.5, tint: false, proc: 'brick' },
  { role: 'brick_painted', id: 'whitewashed_brick', size: 2, tint: true, proc: 'brick' },
  { role: 'brick_dark', id: 'brick_wall_08', size: 1.9, tint: false, proc: 'brick' },
  { role: 'concrete_panel', id: 'concrete_wall_004', size: 2, tint: true, proc: 'panel' },
  { role: 'concrete_rough', id: 'concrete_wall_008', size: 2.71, tint: true, proc: 'concrete' },
  { role: 'stucco_a', id: 'beige_wall_002', size: 3, tint: true, proc: 'stucco' },
  { role: 'stucco_b', id: 'white_stucco', size: 2, tint: true, proc: 'stucco' },
  { role: 'granite', id: 'granite_tile_03', size: 1.8, tint: true, proc: 'panel' },
  { role: 'wood_siding', id: 'brown_planks_05', size: 1, tint: true, proc: 'siding' },
  { role: 'metal_corrugated', id: 'corrugated_iron', size: 1.12, tint: true, proc: 'corrugated' },
  { role: 'roof_slate', id: 'roof_slates_02', size: 3, tint: false, proc: 'slate' },
  { role: 'roof_tar', id: 'bitumen', size: 20, tint: true, proc: 'concrete' },
  { role: 'grass', id: 'leafy_grass', size: 2, tint: false, proc: 'grass' },
  { role: 'plain', id: null, size: 2, tint: true, proc: 'plain' },
  { role: 'fabric', id: null, size: 0.6, tint: true, proc: 'fabric' },
];
export const GROUND_SETS = { asphalt: 'asphalt_02', asphalt2: 'clean_asphalt', sidewalk: 'concrete_pavement', grass: 'leafy_grass', dirt: 'brown_mud_dry' };

/**
 * Load the facade/roof PBR array and the ground texture sets. Never throws.
 * @returns {Promise<{albedo:THREE.DataArrayTexture, normal:THREE.DataArrayTexture, sizes:number[], tintable:number[], loaded:number, ground:object}>}
 */
export async function loadPbrTextures(renderer, { size = 1024, onProgress } = {}) {
  const base = textureBase();
  const manifest = await loadTextureManifest(base);
  // ground sets: plain THREE textures (KTX2 CompressedTextures when compressed, else browser-decoded JPGs), null where
  // missing; fetched in parallel with the array so their transcode / decode overlaps the array packing
  const ground = {};
  const groundP = Promise.all(Object.entries(GROUND_SETS).map(async ([k, id]) => { ground[k] = await loadTextureSet(id, { renderer, base, manifest }); }));
  const arr = await makeTextureArray(renderer, PBR_SETS, { size, base, manifest, onProgress });
  await groundP;
  return { ...arr, ground };
}
