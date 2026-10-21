// Shoreline, piers and distant land.
//
// Waterfront (built from MapWorld._buildGroundAndWater, updated from MapWorld.update):
//  - water: the bay surface (water.js) with the baked shoreline distance field; ponds get a still variant.
//  - walls: seawalls along the city waterfront and piers as a deck edge (lit concrete fascia + coping), a
//    shadowed underside and a recessed dark inner wall (1.2 m promenade / 1.8 m piers) so you look under the
//    deck between the pilings; natural shores get a rock bank. PBR concrete (concrete_wall_008) / rock
//    (rock_boulder_dry) with a wet, algae-dark band at the waterline. One draw call.
//  - dressing (one merged mesh): pilings under the decks (timber on piers, concrete under the promenade),
//    fender piles, rubber fenders, bollards and cleats, a promenade railing and a few moored boats that bob.
//  - rip-rap: instanced boulders (triplanar Poly Haven rock) along natural shores (islands, the mainland shore
//    outside the detailed district). One draw call.
//  - distant land: a land-class texture (urban / scrub / rock; land polygons classed 'main' = the largest,
//    'island' = > 1.5 km2, 'small') drives the ground material blend (createGroundMaterial); instanced scrub / tree
//    blobs on the islands and natural hills; low instanced city blocks over the mainland outside the filler ring
//    and on the `islandFill` island's flat ground (lit windows at night); FarLand: the real relief beyond the map
//    square (farland.json, engine/tools/bake-farland.mjs) with city blocks on the flats, optional container cranes
//    on a waterfront (`farland` options), depth-compressed into the camera far plane and hazed with a
//    height-dependent extension of the scene fog. One draw call each.
//  - boats (Boats.js): ferries, a container ship and sailboats with wakes (routes from the `boats` options).
//
// City-specific options (Waterfront o.*, passed by the game through MapWorld `shore`):
//   ferryDock { x, z (bake frame), dir: [dx, dz], search = 600, clear = 22 }  terminal: walk from (x,z) along dir until
//              the water is `clear` m from the shore; null = no ferries
//   islandFill (landPoly) => bool    island whose flat ground gets built-up context blocks
//   boats      { ferryRoute, ferries, shipRoute, sailZones, sailCount, avoidRoads: RegExp }  (see Boats.js)
//   farland    { url, natural(bx, bz), urbanMax(bx, bz), clusters: [[bx, bz, r, hMin, hMax, density]],
//                cranes: { region: [x0, x1, z0, z1], max, spacing } | null, blockAngle(bx, bz) }  (bake frame)
import * as THREE from 'three';
import { Raster, polygonToBoxes } from '../physics/spatial.js';
import { mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js';
import { mulberry32, hash2, clamp, smoothstep, signedArea } from './geom.js';
import { buildShoreField, createWaterMaterial, makeNoiseTexture, Water } from './Water.js';
import { Boats, createBoatMaterial, fishingBoatGeometry, sailboatGeometry, launchGeometry } from './Boats.js';

// ====================================================================== textures (Poly Haven CC0, fetch-textures.mjs)
const texBase = () => `${(import.meta.env && import.meta.env.BASE_URL) || '/'}textures/`;
const _loader = new THREE.TextureLoader();
let _tex = null;
function loadMap(renderer, id, file, srgb, fb) {
  const t = _loader.load(`${texBase()}${id}/${file}.jpg`, undefined, undefined, () => {
    t.image = fallbackImage(file, fb); t.needsUpdate = true; // offline / not fetched: procedural stand-in
  });
  t.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.anisotropy = renderer ? Math.min(8, renderer.capabilities.getMaxAnisotropy()) : 4;
  return t;
}
function fallbackImage(file, rgb) {
  const S = 64, c = document.createElement('canvas');
  c.width = c.height = S;
  const g = c.getContext('2d'), img = g.createImageData(S, S);
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    const o = (y * S + x) * 4, n = 0.8 + 0.4 * hash2(x * 7 + 3, y * 13 + 1);
    if (file === 'nor_gl') { img.data[o] = 128; img.data[o + 1] = 128; img.data[o + 2] = 255; }
    else if (file === 'rough') { img.data[o] = img.data[o + 1] = img.data[o + 2] = 215; }
    else { img.data[o] = rgb[0] * n; img.data[o + 1] = rgb[1] * n; img.data[o + 2] = rgb[2] * n; }
    img.data[o + 3] = 255;
  }
  g.putImageData(img, 0, 0);
  return c;
}
/** Shared texture set (loaded once, async; materials pick them up when they arrive). */
export function shoreTextures(renderer) {
  if (_tex) return _tex;
  _tex = {
    rock: loadMap(renderer, 'rock_boulder_dry', 'diff', true, [150, 145, 135]),
    rockN: loadMap(renderer, 'rock_boulder_dry', 'nor_gl', false),
    dry: loadMap(renderer, 'withered_grass', 'diff', true, [150, 125, 80]),
    dryN: loadMap(renderer, 'withered_grass', 'nor_gl', false),
    aer: loadMap(renderer, 'aerial_grass_rock', 'diff', true, [120, 110, 60]),
    ter: loadMap(renderer, 'rocky_terrain_02', 'diff', true, [80, 95, 50]),
    conc: loadMap(renderer, 'concrete_wall_008', 'diff', true, [160, 158, 150]),
    concN: loadMap(renderer, 'concrete_wall_008', 'nor_gl', false),
  };
  return _tex;
}

const HASH_GLSL = /* glsl */`
float shHash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
`;

// ====================================================================== ground (land) material
/**
 * Ground for every land polygon: the old urban ground inside the district, darker built-up ground in the
 * filler ring, a street-grid / yard pattern over the outer mainland, and on natural land (islands,
 * islets, the high hills) golden October grass / dark scrub / rock by slope, land class and large-scale
 * noise (15 m and 90 m aerial photo textures + 2 m detail with normal map near the camera).
 * Land class + grid uniforms are filled in later by Waterfront.build (bakeLandClass).
 */
export function createGroundMaterial(urbanMap, renderer) {
  const tx = shoreTextures(renderer);
  const m = new THREE.MeshStandardMaterial({ map: urbanMap, roughness: 0.95, metalness: 0 });
  const u = {
    uClass: { value: null }, uClassRect: { value: new THREE.Vector4(0, 0, 1, 1) },
    uGrid: { value: new THREE.Vector4(0, 0, 1e6, 0) }, uAng: { value: new THREE.Vector4() },
    uBlock: { value: new THREE.Vector3(128, 92, 9.5) }, uCity: { value: new THREE.Vector4(-1e6, -1e6, 1e6, 1e6) },
    uDry: { value: tx.dry }, uAer: { value: tx.aer }, uTer: { value: tx.ter }, uRock: { value: tx.rock },
    uNight: { value: 0 },
  };
  m.userData.u = u;
  m.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, u);
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vGWP;\nvarying vec3 vGWN;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\nvGWP = (modelMatrix * vec4(transformed, 1.0)).xyz;\nvGWN = normalize(mat3(modelMatrix) * objectNormal);');
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', `#include <common>
uniform sampler2D uClass; uniform vec4 uClassRect; uniform vec4 uGrid; uniform vec4 uAng; uniform vec3 uBlock; uniform vec4 uCity;
uniform sampler2D uDry; uniform sampler2D uAer; uniform sampler2D uTer; uniform sampler2D uRock; uniform float uNight;
varying vec3 vGWP; varying vec3 vGWN;
${HASH_GLSL}`)
      .replace('#include <map_fragment>', /* glsl */`#include <map_fragment>
float gNat = 0.0, gDet = 0.0, gStreet = 0.0;
{
  vec2 p = vGWP.xz;
  float dist = distance(cameraPosition, vGWP);
  vec3 urb = diffuseColor.rgb * (0.78 + 0.5 * texture2D(map, vMapUv * 0.037 + 0.31).g);
  vec4 cls = texture2D(uClass, (p - uClassRect.xy) * uClassRect.zw);
  float urban = cls.r;
  bool inCity = p.x > uCity.x && p.x < uCity.z && p.y > uCity.y && p.y < uCity.w;
  vec3 built = urb;
  if (!inCity) {
    // context blocks: same grid as the filler / outer city boxes (per-quadrant street angle)
    vec2 r = p - uGrid.xy;
    int qi = r.x >= 0.0 ? (r.y >= 0.0 ? 0 : 1) : (r.y >= 0.0 ? 3 : 2);
    float a = uAng[qi];
    vec2 ua = vec2(cos(a), sin(a));
    vec2 gc = vec2(dot(r, ua) / uBlock.x, dot(r, vec2(-ua.y, ua.x)) / uBlock.y);
    vec2 id = floor(gc + 0.5);
    vec2 f = (gc - id) * uBlock.xy;
    vec2 e = uBlock.xy * 0.5 - abs(f);
    float ed = min(e.x, e.y);
    float aaw = max(1.0, fwidth(ed) * 1.5);
    float street = 1.0 - smoothstep(uBlock.z - 2.5 - aaw, uBlock.z - 2.5 + aaw, ed);
    float kerb = (1.0 - smoothstep(uBlock.z - aaw, uBlock.z + 0.5 + aaw, ed)) - street;
    float yard = shHash(id * 1.37 + floor((f.x / uBlock.x + 0.5) * 4.0) * 3.1);
    float tn = texture2D(uTer, p / 41.0).g;
    vec3 lot = mix(vec3(0.10, 0.10, 0.095), vec3(0.06, 0.075, 0.04), smoothstep(0.35, 0.75, tn * 0.7 + yard * 0.5));
    vec3 pat = mix(lot, vec3(0.2, 0.195, 0.185), clamp(kerb, 0.0, 1.0));
    pat = mix(pat, vec3(0.035, 0.035, 0.038) * (0.85 + 0.3 * tn), street);
    built = mix(urb * mix(1.0, 0.6, cls.a), pat, 0.85 * cls.a);
    gStreet = street * cls.a * (1.0 - smoothstep(0.0, 4.0, aaw));
  }
  vec3 nat = built;
  if (urban < 0.995) {
    vec2 dx = dFdx(p), dy = dFdy(p);
    vec3 ter = textureGrad(uTer, p / 90.0, dx / 90.0, dy / 90.0).rgb;
    float big = textureGrad(uTer, p / 733.0 + 0.37, dx / 733.0, dy / 733.0).g;
    vec3 aer = textureGrad(uAer, p / 15.0, dx / 15.0, dy / 15.0).rgb;
    float det = 1.0 - smoothstep(45.0, 200.0, dist);
    vec3 grass = aer * vec3(1.2, 1.02, 0.66);
    if (det > 0.0) {
      vec3 dry = textureGrad(uDry, p / 2.0, dx / 2.0, dy / 2.0).rgb;
      grass = mix(grass, dry * vec3(0.95, 0.9, 0.78), det * 0.7);
      grass *= 0.86 + 0.28 * dry.g / max(0.05, dot(dry, vec3(0.333))) * det; // fake micro relief
    }
    grass *= 0.72 + 0.62 * big;
    vec3 scrub = ter * vec3(0.42, 0.55, 0.36) * (0.8 + 0.4 * big);
    float forest = clamp(cls.g + (big - 0.45) * 0.8 + (ter.g - ter.r) * 1.2, 0.0, 1.0);
    nat = mix(grass, scrub, smoothstep(0.3, 0.7, forest));
    float slope = 1.0 - clamp(vGWN.y, 0.0, 1.0);
    float rk = smoothstep(0.28, 0.55, slope + (big - 0.5) * 0.25 + cls.b * 0.25);
    if (rk > 0.0) {
      vec3 n = abs(vGWN) + 1e-3;
      vec3 r3 = (textureGrad(uRock, vGWP.zy / 5.0, dFdx(vGWP.zy) / 5.0, dFdy(vGWP.zy) / 5.0).rgb * n.x
        + textureGrad(uRock, vGWP.xy / 5.0, dFdx(vGWP.xy) / 5.0, dFdy(vGWP.xy) / 5.0).rgb * n.z
        + textureGrad(uRock, p / 5.0, dx / 5.0, dy / 5.0).rgb * n.y) / (n.x + n.y + n.z);
      nat = mix(nat, r3 * vec3(0.78, 0.72, 0.64), rk);
    }
    nat *= mix(0.62, 1.0, smoothstep(0.2, 2.2, vGWP.y)); // damp ground at the waterline
    gNat = 1.0 - urban; gDet = det;
  }
  diffuseColor.rgb = mix(nat, built, urban);
}`)
      .replace('#include <emissivemap_fragment>', '#include <emissivemap_fragment>\ntotalEmissiveRadiance += vec3(1.0, 0.62, 0.32) * (0.05 * uNight * gStreet);');
  };
  m.customProgramCacheKey = () => 'roadster-ground-3';
  return m;
}

/**
 * Land class texture over the map square: R = urban weight, G = scrub / forest, B = rock bias.
 * The mainland (largest polygon) is urban below ~130 m; 'island' polygons (> 1.5 km2) are wooded above 7 m and
 * flat built-up fill below; 'small' islets are rocky with buildings. Also returns CPU arrays for vegetation placement.
 */
export function bakeLandClass(world, N = 256) {
  const OB = world.outerBox;
  const size = Math.max(OB.maxX - OB.minX, OB.maxZ - OB.minZ), cell = size / N;
  const ras = new Raster(OB.minX, OB.minZ, OB.minX + size - 1e-3, OB.minZ + size - 1e-3, cell);
  const polys = world.landPolys.map((L) => ({ L, area: Math.abs(signedArea(L.ring)) })).sort((a, b) => b.area - a.area);
  polys.forEach((p, j) => ras.fillPolygon([p.L.ring], 1 << Math.min(j, 6), 'or'));
  for (const L of world.pierPolys || []) ras.fillPolygon([L.ring], 128, 'or'); // piers: paved, no street grid
  const W = ras.w, H = ras.h;
  // channels: urban, forest/scrub, rock bias, street-grid pattern; id = polygon index + 1 (0 = water)
  const R = new Float32Array(W * H * 4), id = new Int16Array(W * H);
  const role = polys.map((p, j) => (j === 0 ? 'main' : p.area > 1.5e6 ? 'island' : 'small'));
  for (let k = 0; k < H; k++) for (let i = 0; i < W; i++) {
    const j = k * W + i, bits = ras.data[j];
    if (!bits) continue;
    const x = OB.minX + (i + 0.5) * cell, z = OB.minZ + (k + 0.5) * cell;
    let ur = 1, fo = 0.2, rk = 0, pat = 0;
    if (bits & 128 && !(bits & 127)) { id[j] = 1; ur = 1; fo = 0; }
    else {
      const pi = 31 - Math.clz32(bits & 127 & -(bits & 127));
      id[j] = pi + 1;
      const h = world.terrain ? world.terrain.rawHeightAt(x, z) : 0;
      const n = hash2(i * 3 + 7, k * 5 + 1);
      const r = role[pi] || 'small';
      if (r === 'main') { ur = 1 - smoothstep(115, 165, h); fo = 0.2 + 0.4 * (1 - ur); pat = bits & 128 ? 0 : 1; }
      else if (r === 'island') {
        if (h > 7) { ur = 0.05 + 0.1 * smoothstep(20, 7, h); fo = 0.62 + 0.3 * n; rk = 0.3; }
        else { ur = 0.6; fo = 0.1; }
      } else { ur = 0.3; fo = 0.15; rk = 0.8; }
    }
    R[j * 4] = ur; R[j * 4 + 1] = fo; R[j * 4 + 2] = rk; R[j * 4 + 3] = pat;
  }
  // blur (3x3, twice) over land texels for soft transitions
  for (let pass = 0; pass < 2; pass++) {
    const src = R.slice();
    for (let k = 1; k < H - 1; k++) for (let i = 1; i < W - 1; i++) {
      const j = k * W + i;
      if (!id[j]) continue;
      for (let c = 0; c < 4; c++) {
        let s = 0, w = 0;
        for (let dk = -1; dk <= 1; dk++) for (let di = -1; di <= 1; di++) { const q = (k + dk) * W + i + di; if (!id[q]) continue; s += src[q * 4 + c]; w++; }
        R[j * 4 + c] = s / w;
      }
    }
  }
  // dilate land values into the water texels so bilinear filtering never pulls shores toward "natural"
  const filled = Uint8Array.from(id, (v) => (v ? 1 : 0));
  for (let pass = 0; pass < 3; pass++) {
    const src = R.slice(), f0 = filled.slice();
    for (let k = 0; k < H; k++) for (let i = 0; i < W; i++) {
      const j = k * W + i;
      if (f0[j]) continue;
      const acc = [0, 0, 0, 0]; let n = 0;
      for (let dk = -1; dk <= 1; dk++) for (let di = -1; di <= 1; di++) {
        const ii = i + di, kk = k + dk;
        if (ii < 0 || kk < 0 || ii >= W || kk >= H) continue;
        const q = kk * W + ii;
        if (!f0[q]) continue;
        for (let c = 0; c < 4; c++) acc[c] += src[q * 4 + c];
        n++;
      }
      if (n) { for (let c = 0; c < 4; c++) R[j * 4 + c] = acc[c] / n; filled[j] = 1; }
    }
  }
  const data = new Uint8Array(W * H * 4);
  for (let j = 0; j < W * H * 4; j++) data[j] = clamp(R[j], 0, 1) * 255;
  const tex = new THREE.DataTexture(data, W, H, THREE.RGBAFormat);
  tex.magFilter = THREE.LinearFilter; tex.minFilter = THREE.LinearFilter; tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.colorSpace = THREE.NoColorSpace; tex.needsUpdate = true;
  return {
    tex, rect: new THREE.Vector4(OB.minX, OB.minZ, 1 / (W * cell), 1 / (H * cell)), W, H, cell, minX: OB.minX, minZ: OB.minZ, R, role,
    at(x, z) {
      const i = Math.floor((x - OB.minX) / cell), k = Math.floor((z - OB.minZ) / cell);
      if (i < 0 || k < 0 || i >= W || k >= H) return null;
      const j = k * W + i;
      return id[j] ? { urban: R[j * 4], forest: R[j * 4 + 1], rock: R[j * 4 + 2], poly: id[j] - 1, role: role[id[j] - 1] } : null;
    },
  };
}

// ====================================================================== mesh buffer
class MBuf {
  constructor(extra = []) { this.pos = []; this.nor = []; this.uv = []; this.col = []; this.ex = {}; for (const e of extra) this.ex[e] = []; this.idx = []; this.n = 0; }
  v(x, y, z, nx, ny, nz, c, u = 0, w = 0, ex = null) {
    this.pos.push(x, y, z); this.nor.push(nx, ny, nz); this.uv.push(u, w); this.col.push(c[0], c[1], c[2]);
    for (const k in this.ex) this.ex[k].push(ex && ex[k] !== undefined ? ex[k] : 0);
    return this.n++;
  }
  /** quad a b c d (counter-clockwise seen from the normal side) */
  q(a, b, c, d) { this.idx.push(a, b, c, a, c, d); }
  /** flat quad from 4 points with a normal; winding fixed to face the normal. */
  quad(p0, p1, p2, p3, n, c, uvs = null, ex = null) {
    const ux = p1[0] - p0[0], uy = p1[1] - p0[1], uz = p1[2] - p0[2], vx = p2[0] - p0[0], vy = p2[1] - p0[1], vz = p2[2] - p0[2];
    const fx = uy * vz - uz * vy, fy = uz * vx - ux * vz, fz = ux * vy - uy * vx;
    const P = fx * n[0] + fy * n[1] + fz * n[2] >= 0 ? [p0, p1, p2, p3] : [p0, p3, p2, p1];
    const U = uvs ? (P[1] === p1 ? uvs : [uvs[0], uvs[3], uvs[2], uvs[1]]) : null;
    const i = P.map((p, k) => this.v(p[0], p[1], p[2], n[0], n[1], n[2], c, U ? U[k][0] : 0, U ? U[k][1] : 0, ex));
    this.q(i[0], i[1], i[2], i[3]);
  }
  /** vertical cylinder (no caps unless capTop), colour split at splitY. */
  cyl(x, z, y0, y1, r, sides, cLow, cHigh, splitY, capTop = false, ex = null) {
    const ys = splitY > y0 && splitY < y1 ? [y0, splitY, y1] : [y0, y1];
    for (let s = 0; s < sides; s++) {
      const a0 = (s / sides) * Math.PI * 2, a1 = ((s + 1) / sides) * Math.PI * 2;
      const c0 = Math.cos(a0), s0 = Math.sin(a0), c1 = Math.cos(a1), s1 = Math.sin(a1);
      for (let k = 0; k + 1 < ys.length; k++) {
        const col = ys[k + 1] <= splitY + 1e-4 ? cLow : cHigh;
        const ya = ys[k], yb = ys[k + 1];
        const a = this.v(x + c0 * r, ya, z + s0 * r, c0, 0, s0, col, 0, 0, ex), b = this.v(x + c1 * r, ya, z + s1 * r, c1, 0, s1, col, 0, 0, ex);
        const c = this.v(x + c1 * r, yb, z + s1 * r, c1, 0, s1, col, 0, 0, ex), d = this.v(x + c0 * r, yb, z + s0 * r, c0, 0, s0, col, 0, 0, ex);
        this.idx.push(a, d, c, a, c, b);
      }
    }
    if (capTop) {
      const ctr = this.v(x, y1, z, 0, 1, 0, cHigh, 0, 0, ex);
      for (let s = 0; s < sides; s++) {
        const a0 = (s / sides) * Math.PI * 2, a1 = ((s + 1) / sides) * Math.PI * 2;
        const a = this.v(x + Math.cos(a0) * r, y1, z + Math.sin(a0) * r, 0, 1, 0, cHigh, 0, 0, ex), b = this.v(x + Math.cos(a1) * r, y1, z + Math.sin(a1) * r, 0, 1, 0, cHigh, 0, 0, ex);
        this.idx.push(ctr, b, a);
      }
    }
  }
  /** oriented box: centre (x,z), base y0, size (sx along dir, sy, sz across), dir angle (radians, x axis). */
  obox(x, y0, z, sx, sy, sz, ang, c, ex = null, bottom = false) {
    const ca = Math.cos(ang), sa = Math.sin(ang);
    const P = [[-sx / 2, -sz / 2], [sx / 2, -sz / 2], [sx / 2, sz / 2], [-sx / 2, sz / 2]].map(([a, b]) => [x + a * ca - b * sa, z + a * sa + b * ca]);
    const y1 = y0 + sy;
    for (let i = 0; i < 4; i++) {
      const a = P[i], b = P[(i + 1) % 4];
      let nx = b[1] - a[1], nz = -(b[0] - a[0]);
      const l = Math.hypot(nx, nz) || 1; nx /= l; nz /= l;
      const mx = (a[0] + b[0]) / 2 - x, mz = (a[1] + b[1]) / 2 - z;
      if (nx * mx + nz * mz < 0) { nx = -nx; nz = -nz; }
      this.quad([a[0], y0, a[1]], [b[0], y0, b[1]], [b[0], y1, b[1]], [a[0], y1, a[1]], [nx, 0, nz], c, null, ex);
    }
    this.quad([P[0][0], y1, P[0][1]], [P[1][0], y1, P[1][1]], [P[2][0], y1, P[2][1]], [P[3][0], y1, P[3][1]], [0, 1, 0], c, null, ex);
    if (bottom) this.quad([P[0][0], y0, P[0][1]], [P[1][0], y0, P[1][1]], [P[2][0], y0, P[2][1]], [P[3][0], y0, P[3][1]], [0, -1, 0], c, null, ex);
  }
  /** append a non-indexed geometry (position/normal/color[/aEmit]) transformed by matrix m. */
  addGeometry(g, m, ex = null) {
    const P = g.attributes.position, N = g.attributes.normal, Cc = g.attributes.color, E = g.attributes.aEmit;
    const v = new THREE.Vector3(), n = new THREE.Vector3(), nm = new THREE.Matrix3().getNormalMatrix(m);
    const base = this.n;
    for (let i = 0; i < P.count; i++) {
      v.fromBufferAttribute(P, i).applyMatrix4(m); n.fromBufferAttribute(N, i).applyMatrix3(nm).normalize();
      const e = { ...(ex || {}) }; if (E && 'aEmit' in this.ex) e.aEmit = E.getX(i);
      this.v(v.x, v.y, v.z, n.x, n.y, n.z, [Cc.getX(i), Cc.getY(i), Cc.getZ(i)], 0, 0, e);
    }
    for (let i = 0; i < P.count; i++) this.idx.push(base + i);
  }
  geometry() {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(this.pos, 3));
    g.setAttribute('normal', new THREE.Float32BufferAttribute(this.nor, 3));
    g.setAttribute('uv', new THREE.Float32BufferAttribute(this.uv, 2));
    g.setAttribute('color', new THREE.Float32BufferAttribute(this.col, 3));
    for (const k in this.ex) g.setAttribute(k, new THREE.Float32BufferAttribute(this.ex[k], 1));
    g.setIndex(this.n > 65535 ? new THREE.Uint32BufferAttribute(this.idx, 1) : new THREE.Uint16BufferAttribute(this.idx, 1));
    g.computeBoundingSphere();
    return g;
  }
}

// ====================================================================== seawall / pier edge material
function createWallMaterial(renderer, level) {
  const tx = shoreTextures(renderer);
  const m = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.9, metalness: 0, map: tx.conc, normalMap: tx.concN });
  m.normalScale.set(0.9, 0.9);
  const u = { uRock: { value: tx.rock }, uRockN: { value: tx.rockN }, uLevel: { value: level }, uTime: { value: 0 } };
  m.userData.u = u;
  m.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, u);
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vSWP;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\nvSWP = (modelMatrix * vec4(transformed, 1.0)).xyz;');
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', `#include <common>\nuniform sampler2D uRock; uniform sampler2D uRockN; uniform float uLevel; uniform float uTime;\nvarying vec3 vSWP;\n${HASH_GLSL}`)
      // vColor: r = kind (0 seawall, .5 pier, 1 natural bank), g = part (0 face, .33 underside, .67 inner wall, 1 coping), b = tint
      .replace('#include <map_fragment>', /* glsl */`
float wKind = vColor.r, wPart = vColor.g, wTint = vColor.b;
vec2 wuv = vMapUv / 2.71; // uv = metres along the wall / height
vec3 conc = texture2D(map, wuv).rgb * (0.86 + 0.28 * wTint);
vec3 rockC = texture2D(uRock, vMapUv / 3.2).rgb * vec3(0.5, 0.47, 0.43);
vec3 wc = mix(conc * vec3(0.98, 0.97, 0.94), rockC, step(0.75, wKind));
float yw = vSWP.y - uLevel;
float lapse = 0.18 * sin(uTime * 0.9 + vSWP.x * 0.13 + vSWP.z * 0.11);
float n = texture2D(map, wuv * vec2(0.37, 1.3) + 0.21).g;
// tide zone: algae-dark green-black at the waterline, damp darker above, streaks
float wet = 1.0 - smoothstep(0.0, 0.9 + 0.5 * n, yw - lapse);
float algae = 1.0 - smoothstep(-0.1, 0.45 + 0.3 * n, yw - lapse * 0.5);
float streak = smoothstep(0.55, 0.8, texture2D(map, vec2(vMapUv.x * 0.9, 0.0) / 2.71 + 0.5).r) * (1.0 - smoothstep(0.5, 2.6, yw));
wc *= 1.0 - 0.35 * wet - 0.25 * streak;
wc = mix(wc, vec3(0.03, 0.045, 0.03) * (0.7 + 0.6 * n), algae * 0.85);
// pier / promenade decks: shadowed underside and recessed inner wall
if (wPart > 0.2 && wPart < 0.8) wc *= wPart < 0.5 ? 0.22 : 0.16;
if (wPart > 0.9) wc = conc * vec3(1.06, 1.05, 1.02);
diffuseColor.rgb *= wc;
`)
      .replace('#include <color_fragment>', '')
      .replace('#include <roughnessmap_fragment>', '#include <roughnessmap_fragment>\nroughnessFactor = mix(0.92, 0.35, algae * 0.8 + wet * 0.2);')
      .replace('vec3 mapN = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;', 'vec3 mapN = texture2D( normalMap, vNormalMapUv / 2.71 ).xyz * 2.0 - 1.0;')
      .replace('mapN.xy *= normalScale;', 'mapN.xy *= normalScale * (1.0 - 0.6 * algae);');
  };
  m.customProgramCacheKey = () => 'roadster-seawall-1';
  return m;
}

// rock boulders: triplanar rock, wet band (InstancedMesh)
function createRockMaterial(renderer, level) {
  const tx = shoreTextures(renderer);
  const m = new THREE.MeshStandardMaterial({ roughness: 0.85, metalness: 0, flatShading: true });
  const u = { uRock: { value: tx.rock }, uLevel: { value: level } };
  m.userData.u = u;
  m.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, u);
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vRWP;\nvarying vec3 vRWN;')
      .replace('#include <begin_vertex>', `#include <begin_vertex>
#ifdef USE_INSTANCING
  vRWP = (modelMatrix * instanceMatrix * vec4(transformed, 1.0)).xyz;
  vRWN = normalize(mat3(modelMatrix) * mat3(instanceMatrix) * objectNormal);
#else
  vRWP = (modelMatrix * vec4(transformed, 1.0)).xyz;
  vRWN = normalize(mat3(modelMatrix) * objectNormal);
#endif`);
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', '#include <common>\nuniform sampler2D uRock; uniform float uLevel;\nvarying vec3 vRWP;\nvarying vec3 vRWN;')
      .replace('#include <map_fragment>', /* glsl */`
{
  vec3 n = abs(normalize(vRWN)) + 1e-3;
  vec3 c = (texture2D(uRock, vRWP.zy / 2.2).rgb * n.x + texture2D(uRock, vRWP.xz / 2.2).rgb * n.y + texture2D(uRock, vRWP.xy / 2.2).rgb * n.z) / (n.x + n.y + n.z);
  float yw = vRWP.y - uLevel;
  float wet = 1.0 - smoothstep(-0.05, 0.6, yw);
  c *= vec3(0.58, 0.55, 0.5) * (1.0 - 0.45 * wet);
  c = mix(c, vec3(0.035, 0.05, 0.03), (1.0 - smoothstep(-0.4, 0.15, yw)) * 0.75);
  diffuseColor.rgb *= c;
}`)
      .replace('#include <roughnessmap_fragment>', '#include <roughnessmap_fragment>\nroughnessFactor = mix(roughnessFactor, 0.4, 1.0 - smoothstep(-0.05, 0.5, vRWP.y - uLevel));');
  };
  m.customProgramCacheKey = () => 'roadster-riprap-1';
  return m;
}

// low city blocks (instanced boxes): facade window grid, lit at night
function createBlockMaterial({ compress = false } = {}) {
  const m = new THREE.MeshStandardMaterial({ roughness: 0.85, metalness: 0, vertexColors: compress });
  const u = { uNight: { value: 0 }, uFar: { value: 6000 } };
  m.userData.u = u;
  m.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, u);
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', `#include <common>\nvarying vec3 vBWP;\nvarying vec3 vBWN;\nvarying float vBId;\nuniform float uFar;`)
      .replace('#include <begin_vertex>', `#include <begin_vertex>
#ifdef USE_INSTANCING
  vBWP = (modelMatrix * instanceMatrix * vec4(transformed, 1.0)).xyz;
  vBWN = normalize(mat3(modelMatrix) * mat3(instanceMatrix) * objectNormal);
  vBId = float(gl_InstanceID);
#else
  vBWP = (modelMatrix * vec4(transformed, 1.0)).xyz;
  vBWN = normalize(mat3(modelMatrix) * objectNormal);
  vBId = 0.0;
#endif`);
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', `#include <common>\nuniform float uNight;\nvarying vec3 vBWP;\nvarying vec3 vBWN;\nvarying float vBId;\n${HASH_GLSL}`)
      .replace('#include <color_fragment>', /* glsl */`#include <color_fragment>
float bWin = 0.0, bLit = 0.0;
{
  float wall = 1.0 - step(0.5, abs(vBWN.y));
  vec2 t = normalize(vec2(-vBWN.z, vBWN.x) + 1e-5);
  float hc = dot(vBWP.xz, t) / 3.1;
  float fc = (vBWP.y + 0.7) / 3.3;
  float aa = clamp(max(fwidth(hc), fwidth(fc)) * 1.6, 0.0, 1.0);
  vec2 cf = fract(vec2(hc, fc));
  float w = step(0.22, cf.x) * step(cf.x, 0.82) * step(0.32, cf.y) * step(cf.y, 0.86);
  w = mix(w, 0.36, aa) * wall * step(1.2, fc);
  bWin = w;
  float h = shHash(vec2(floor(hc) + vBId * 13.1, floor(fc) * 7.3 + vBId));
  bLit = w * mix(step(0.7, h), 0.3, aa);
  diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.035, 0.04, 0.05), w * 0.75);
  diffuseColor.rgb *= mix(1.0, 0.82, step(0.5, vBWN.y)); // roofs
}`)
      .replace('#include <emissivemap_fragment>', '#include <emissivemap_fragment>\ntotalEmissiveRadiance += vec3(1.0, 0.7, 0.42) * (bLit * uNight * 1.4);');
  };
  m.customProgramCacheKey = () => 'roadster-blocks-1';
  return m;
}

// ====================================================================== helpers
function ringInset(ring, pos, d) {
  // inset each vertex by d along the miter (toward the polygon interior)
  const n = ring.length, out = [];
  const nrm = (a, b) => { const dx = b[0] - a[0], dz = b[1] - a[1], L = Math.hypot(dx, dz) || 1; let ox = dz / L, oz = -dx / L; if (!pos) { ox = -ox; oz = -oz; } return [ox, oz]; };
  for (let i = 0; i < n; i++) {
    const p = ring[i], a = ring[(i - 1 + n) % n], b = ring[(i + 1) % n];
    const n0 = nrm(a, p), n1 = nrm(p, b);
    let mx = n0[0] + n1[0], mz = n0[1] + n1[1];
    const ml = Math.hypot(mx, mz);
    if (ml < 1e-6) { mx = n1[0]; mz = n1[1]; } else { mx /= ml; mz /= ml; }
    const c = mx * n1[0] + mz * n1[1];
    const len = d / Math.max(0.4, c);
    out.push([p[0] - mx * len, p[1] - mz * len]);
  }
  return out;
}
/** split a->b where it crosses the lattice lines of a surface level (x=n*c, z=n*c, x-z=n*c). */
function splitEdge(c, a, b) {
  const ts = [0, 1], dx = b[0] - a[0], dz = b[1] - a[1];
  const fam = (v0, dv) => {
    if (Math.abs(dv) < 1e-9) return;
    const lo = Math.min(v0, v0 + dv), hi = Math.max(v0, v0 + dv);
    for (let k = Math.ceil(lo / c); k * c <= hi; k++) { const t = (k * c - v0) / dv; if (t > 1e-6 && t < 1 - 1e-6) ts.push(t); }
  };
  fam(a[0], dx); fam(a[1], dz); fam(a[0] - a[1], dx - dz);
  ts.sort((p, q) => p - q);
  const out = [];
  for (const t of ts) { if (out.length && t - out[out.length - 1] < 1e-5) continue; out.push(t); }
  return out;
}

// ====================================================================== waterfront
export class Waterfront {
  /**
   * @param {object} o
   * @param {import('./MapWorld.js').MapWorld} o.world
   * @param {THREE.WebGLRenderer} o.renderer
   * @param {number} o.level water level
   * @param {THREE.Material} o.waterMat bay material (water.js), shore field uniforms filled here
   * @param {THREE.Material} o.groundMat ground material (createGroundMaterial)
   */
  constructor(o) {
    this.o = o;
    this.group = new THREE.Group();
    this.group.name = 'waterfront';
    this.stats = {};
    this.mats = [];
  }

  build() {
    const t0 = performance.now();
    const { world, renderer, level } = this.o;
    this.noise = makeNoiseTexture(renderer);
    // ---- shoreline distance field + bay surface
    const field = this.field = buildShoreField(world, renderer);
    const wu = this.o.waterMat.userData.u;
    wu.uShoreF.value = field.fine.tex; wu.uShoreC.value = field.coarse.tex;
    wu.uFineRect.value.copy(field.fine.rect); wu.uCoarseRect.value.copy(field.coarse.rect);
    wu.uNoise.value = this.noise;
    this.water = new Water({ material: this.o.waterMat, level });
    this.group.add(this.water.build());
    if (this.o.pondMat) this.o.pondMat.userData.u.uNoise.value = this.noise;
    // ---- land class -> ground material
    const cls = this.cls = bakeLandClass(world);
    const gu = this.o.groundMat.userData.u;
    gu.uClass.value = cls.tex; gu.uClassRect.value.copy(cls.rect);
    const b = world.bounds;
    gu.uCity.value.set(b.minX - 8, b.minZ - 8, b.maxX + 8, b.maxZ + 8);
    const fg = world._fillerGrid;
    if (fg) { gu.uGrid.value.set(fg.cx, fg.cz, fg.Rout, 0); gu.uAng.value.set(...fg.angles); gu.uBlock.value.set(fg.BW, fg.BH, fg.street); }
    // ---- shore structures
    const edges = this._walls();
    this._dressing(edges);
    this._riprap(edges);
    // ---- distant land
    this._vegetation();
    this._blocks();
    // ---- boats
    this._boats();
    // ---- relief beyond the 9 km map (async: farland.json)
    this._farland();
    this.stats.ms = Math.round(performance.now() - t0);
    this.stats.fieldMs = field.ms;
    console.info('[Waterfront]', this.stats);
    return this.group;
  }

  // ------------------------------------------------------------------ seawalls / pier edges
  _walls() {
    const { world, level } = this.o;
    const S = world.surface, OB = world.outerBox, b = world.bounds;
    const isLand = (x, z) => world.isLand(x, z);
    const e = 2;
    const onBox = (a, c) => (Math.abs(a[0] - OB.minX) < e && Math.abs(c[0] - OB.minX) < e) || (Math.abs(a[0] - OB.maxX) < e && Math.abs(c[0] - OB.maxX) < e)
      || (Math.abs(a[1] - OB.minZ) < e && Math.abs(c[1] - OB.minZ) < e) || (Math.abs(a[1] - OB.maxZ) < e && Math.abs(c[1] - OB.maxZ) < e);
    const inCity = (x, z) => x > b.minX - 40 && x < b.maxX + 40 && z > b.minZ - 40 && z < b.maxZ + 40;
    const buf = new MBuf();
    const y0 = level - 2.5;
    const edges = [];
    const rnd = mulberry32(4401);
    const DEPTH = { city: 1.2, pier: 1.8 };
    const polys = [...world.landPolys.map((L) => ({ ring: L.ring, pier: false })), ...(world.pierPolys || []).map((L) => ({ ring: L.ring, pier: true }))];
    let tris0 = 0;
    for (const P of polys) {
      const ring = P.ring, pos = signedArea(ring) > 0;
      const ins = { city: ringInset(ring, pos, DEPTH.city), pier: ringInset(ring, pos, DEPTH.pier) };
      let uAcc = 0;
      for (let i = 0; i < ring.length; i++) {
        const a = ring[i], c = ring[(i + 1) % ring.length];
        if (onBox(a, c)) continue;
        const dx = c[0] - a[0], dz = c[1] - a[1], L = Math.hypot(dx, dz);
        if (L < 0.05) continue;
        let ox = dz / L, oz = -dx / L; if (!pos) { ox = -ox; oz = -oz; }
        // interior edge (land / another pier on the outside)? -> no wall
        let outLand = 0;
        for (const t of [0.15, 0.5, 0.85]) if (isLand(a[0] + dx * t + ox * 1.6, a[1] + dz * t + oz * 1.6)) outLand++;
        if (outLand === 3) continue;
        const mx = (a[0] + c[0]) / 2, mz = (a[1] + c[1]) / 2;
        const kind = P.pier ? 'pier' : inCity(mx, mz) ? 'city' : 'natural';
        const kc = kind === 'city' ? 0 : kind === 'pier' ? 0.5 : 1;
        const cell = S.levelAt(mx, mz).cell;
        const ts = splitEdge(cell, a, c);
        const pts = ts.map((t) => [a[0] + dx * t, a[1] + dz * t, t]);
        const hs = pts.map((p) => S.heightAt(p[0], p[1]));
        const minTop = Math.min(...hs);
        const over = kind !== 'natural' && minTop - level > 1.45 ? DEPTH[kind] : 0;
        const ia = over ? ins[kind][i] : null, ic = over ? ins[kind][(i + 1) % ring.length] : null;
        const tint = rnd();
        const fas = 0.7;
        for (let j = 0; j + 1 < pts.length; j++) {
          const p = pts[j], q = pts[j + 1], hp = hs[j], hq = hs[j + 1];
          const up = uAcc + p[2] * L, uq = uAcc + q[2] * L;
          const N = [ox, 0, oz];
          if (!over) {
            buf.quad([p[0], y0, p[1]], [q[0], y0, q[1]], [q[0], hq, q[1]], [p[0], hp, p[1]], N, [kc, 0, tint], [[up, y0], [uq, y0], [uq, hq], [up, hp]]);
          } else {
            const yU = minTop - fas;
            buf.quad([p[0], yU, p[1]], [q[0], yU, q[1]], [q[0], hq, q[1]], [p[0], hp, p[1]], N, [kc, 0, tint], [[up, yU], [uq, yU], [uq, hq], [up, hp]]);
          }
          if (kind !== 'natural') { // coping: lip + top strip
            const lo = 0.07, li = 0.38, yc = 0.09;
            buf.quad([p[0] + ox * lo, hp - 0.2, p[1] + oz * lo], [q[0] + ox * lo, hq - 0.2, q[1] + oz * lo], [q[0] + ox * lo, hq + yc, q[1] + oz * lo], [p[0] + ox * lo, hp + yc, p[1] + oz * lo], N, [kc, 1, tint], [[up, hp - 0.2], [uq, hq - 0.2], [uq, hq + yc], [up, hp + yc]]);
            buf.quad([p[0] + ox * lo, hp + yc, p[1] + oz * lo], [q[0] + ox * lo, hq + yc, q[1] + oz * lo], [q[0] - ox * li, hq + yc, q[1] - oz * li], [p[0] - ox * li, hp + yc, p[1] - oz * li], [0, 1, 0], [kc, 1, tint], [[up, 0], [uq, 0], [uq, 0.45], [up, 0.45]]);
          }
        }
        if (over) { // shadowed deck underside + recessed inner wall: one quad each per edge
          const yU = minTop - fas, N = [ox, 0, oz], u0 = uAcc, u1 = uAcc + L;
          buf.quad([a[0], yU, a[1]], [c[0], yU, c[1]], [ic[0], yU, ic[1]], [ia[0], yU, ia[1]], [0, -1, 0], [kc, 0.33, tint], [[u0, 0], [u1, 0], [u1, over], [u0, over]]);
          buf.quad([ia[0], y0, ia[1]], [ic[0], y0, ic[1]], [ic[0], yU, ic[1]], [ia[0], yU, ia[1]], N, [kc, 0.67, tint], [[u0, y0], [u1, y0], [u1, yU], [u0, yU]]);
        }
        uAcc += L;
        edges.push({ a, c, dx: dx / L, dz: dz / L, L, ox, oz, kind, over, minTop, pier: P.pier, pts, hs });
        tris0++;
      }
    }
    const mat = this.wallMat = createWallMaterial(this.o.renderer, level);
    const mesh = new THREE.Mesh(buf.geometry(), mat);
    mesh.name = 'seawalls'; mesh.receiveShadow = true; mesh.castShadow = false;
    mesh.matrixAutoUpdate = false; mesh.updateMatrix();
    this.group.add(mesh);
    this.stats.wallEdges = edges.length;
    this.stats.wallTris = buf.idx.length / 3;
    return edges;
  }

  // ------------------------------------------------------------------ pilings, fenders, bollards, railings, moored boats
  _dressing(edges) {
    const { world, level } = this.o;
    const S = world.surface, isLand = (x, z) => world.isLand(x, z);
    const buf = new MBuf(['aEmit', 'aBob']);
    const rnd = mulberry32(9917);
    const timber = [0.075, 0.058, 0.045], conc = [0.15, 0.148, 0.142]; // under the deck: mostly in shadow
    const black = [0.025, 0.025, 0.027], iron = [0.04, 0.04, 0.042], rail = [0.07, 0.09, 0.09];
    const outsideWater = (x, z, ox, oz) => !isLand(x + ox * 1.4, z + oz * 1.4);
    let piles = 0, bollards = 0, fenders = 0, posts = 0;
    const dock = this.ferryDock = this._findFerryDock();
    for (const E of edges) {
      if (E.kind === 'natural') continue;
      const pier = E.pier, along = (t) => [E.a[0] + E.dx * t, E.a[1] + E.dz * t];
      // pilings: under the deck edge (overhang) or a fender-pile row in front of a low wall
      const step = pier ? 3.6 : 4.2;
      for (let s = step * 0.5; s < E.L; s += step) {
        const [x, z] = along(s);
        if (!outsideWater(x, z, E.ox, E.oz)) continue;
        const top = S.heightAt(x, z);
        if (E.over) {
          const yU = Math.min(top, E.minTop) - 0.7;
          const r = pier ? 0.2 + rnd() * 0.06 : 0.28;
          buf.cyl(x - E.ox * 0.42, z - E.oz * 0.42, level - 1.6, yU, r, 4, pier ? timber : conc, pier ? timber : conc, -99);
          piles++;
          if (E.over > 1.5 && rnd() < 0.35) { buf.cyl(x - E.ox * 1.3 + E.dx * 1.2, z - E.oz * 1.3 + E.dz * 1.2, level - 1.6, yU, r, 4, timber, timber, -99); piles++; }
        } else if (pier || rnd() < 0.5) {
          buf.cyl(x + E.ox * 0.24, z + E.oz * 0.24, level - 1.6, top - 0.05, 0.17, 4, timber, timber, -99, true);
          piles++;
        }
      }
      // rubber fenders on the pier fascia
      if (pier) for (let s = 5; s < E.L - 2; s += 11) {
        const [x, z] = along(s);
        if (!outsideWater(x, z, E.ox, E.oz)) continue;
        const top = S.heightAt(x, z);
        buf.cyl(x + E.ox * 0.2, z + E.oz * 0.2, Math.max(level + 0.1, top - 1.9), top - 0.22, 0.19, 6, black, black, -99);
        fenders++;
      }
      // bollards / cleats on the deck edge
      const bstep = pier ? 22 : 40;
      for (let s = bstep * 0.6; s < E.L - 1; s += bstep) {
        const [x, z] = along(s);
        if (!outsideWater(x, z, E.ox, E.oz)) continue;
        const bx = x - E.ox * 0.6, bz = z - E.oz * 0.6, y = S.heightAt(bx, bz) + 0.08;
        if (rnd() < 0.65) {
          buf.cyl(bx, bz, y, y + 0.42, 0.16, 5, iron, iron, -99);
          buf.cyl(bx, bz, y + 0.42, y + 0.5, 0.23, 5, iron, iron, -99, true);
        } else {
          const ang = Math.atan2(E.dz, E.dx);
          buf.obox(bx, y, bz, 0.18, 0.14, 0.14, ang, iron);
          buf.obox(bx, y + 0.12, bz, 0.52, 0.07, 0.1, ang, iron);
        }
        bollards++;
      }
      // promenade railing (city seawall only)
      if (!pier) {
        const ang = Math.atan2(E.dz, E.dx);
        let prev = null;
        for (let s = 0.6; s < E.L - 0.3; s += 4.0) {
          const [x, z] = along(s);
          if (!outsideWater(x, z, E.ox, E.oz)) { prev = null; continue; }
          const px = x - E.ox * 0.2, pz = z - E.oz * 0.2, y = S.heightAt(px, pz) + 0.1;
          buf.cyl(px, pz, y, y + 1.05, 0.035, 3, rail, rail, -99);
          posts++;
          if (prev) {
            const n = [E.ox, 0, E.oz], tn = [0, 1, 0];
            const P0 = [prev[0], prev[2] + 1.05, prev[1]], P1 = [px, y + 1.05, pz];
            buf.quad([P0[0] + E.ox * 0.035, P0[1] - 0.06, P0[2] + E.oz * 0.035], [P1[0] + E.ox * 0.035, P1[1] - 0.06, P1[2] + E.oz * 0.035], [P1[0] + E.ox * 0.035, P1[1], P1[2] + E.oz * 0.035], [P0[0] + E.ox * 0.035, P0[1], P0[2] + E.oz * 0.035], n, rail);
            buf.quad([P0[0] - E.ox * 0.035, P0[1], P0[2] - E.oz * 0.035], [P1[0] - E.ox * 0.035, P1[1], P1[2] - E.oz * 0.035], [P1[0] + E.ox * 0.035, P1[1], P1[2] + E.oz * 0.035], [P0[0] + E.ox * 0.035, P0[1], P0[2] + E.oz * 0.035], tn, rail);
            buf.quad([P0[0] - E.ox * 0.035, P0[1] - 0.06, P0[2] - E.oz * 0.035], [P1[0] - E.ox * 0.035, P1[1] - 0.06, P1[2] - E.oz * 0.035], [P1[0] - E.ox * 0.035, P1[1], P1[2] - E.oz * 0.035], [P0[0] - E.ox * 0.035, P0[1], P0[2] - E.oz * 0.035], [-E.ox, 0, -E.oz], rail);
          }
          prev = [px, pz, y];
        }
      }
    }
    // moored boats along pier / seawall faces with water outside (keep the ferry slips clear)
    const gens = [() => sailboatGeometry(undefined, false), fishingBoatGeometry, launchGeometry];
    const geos = gens.map((g) => g());
    const dims = [[11, 3.6], [14, 4.6], [9, 3.0]];
    let moored = 0;
    const cands = edges.filter((E) => E.kind !== 'natural' && E.L > 18);
    const order = cands.map((E, i) => [hash2(i * 17 + 5, 33), E]).sort((p, q) => p[0] - q[0]).map((p) => p[1]);
    const m4 = new THREE.Matrix4(), qn = new THREE.Quaternion(), up = new THREE.Vector3(0, 1, 0), sc = new THREE.Vector3(1, 1, 1), ps = new THREE.Vector3();
    const field = this.field, placed = [];
    for (const E of order) {
      if (moored >= 22) break;
      for (let s = 9; s < E.L - 8 && moored < 22; s += 17 + rnd() * 10) {
        if (rnd() < 0.45) continue;
        const t = Math.floor(rnd() * 3), [len, beam] = dims[t];
        const off = beam / 2 + (E.over ? 1.7 : 1.4) + rnd() * 0.6;
        const x = E.a[0] + E.dx * s + E.ox * off, z = E.a[1] + E.dz * s + E.oz * off;
        if (dock && Math.hypot(x - dock.x, z - dock.z) < 140) continue;
        const ok = [-0.5, 0, 0.5].every((f) => {
          const xx = x + E.dx * f * len, zz = z + E.dz * f * len;
          const d = field.distanceAt(xx, zz);
          return !isLand(xx, zz) && d > beam / 2 - 0.2 && d < 16;
        }) && field.distanceAt(x + E.ox * (beam / 2 + 1.5), z + E.oz * (beam / 2 + 1.5)) > beam / 2 + 1;
        if (!ok || placed.some((p) => Math.hypot(p[0] - x, p[1] - z) < 16)) continue;
        const yaw = Math.atan2(-E.dx, -E.dz) + (rnd() < 0.5 ? Math.PI : 0);
        qn.setFromAxisAngle(up, yaw);
        m4.compose(ps.set(x, level, z), qn, sc);
        buf.addGeometry(geos[t], m4, { aBob: 0.05 + rnd() * 0.95 });
        placed.push([x, z]);
        moored++;
      }
    }
    const mat = this.dressMat = createBoatMaterial({ bob: true, wetLevel: level });
    const mesh = new THREE.Mesh(buf.geometry(), mat);
    mesh.name = 'pier-dressing'; mesh.receiveShadow = true; mesh.castShadow = false;
    mesh.matrixAutoUpdate = false; mesh.updateMatrix();
    mesh.geometry.computeBoundingBox();
    this.dressMesh = mesh;
    this.group.add(mesh);
    Object.assign(this.stats, { piles, fenders, bollards, railPosts: posts, moored, dressTris: buf.idx.length / 3 });
  }

  /** Water point for the ferry terminal (o.ferryDock: bake-frame start point + search direction), or null. */
  _findFerryDock() {
    const { world } = this.o, f = this.field, D = this.o.ferryDock;
    if (!D) return null;
    const off = world.offset || { x: 0, z: 0 }; // world = bake - offset
    let x = D.x - off.x, z = D.z - off.z;
    const dir = D.dir || [1, 0], dl = Math.hypot(dir[0], dir[1]), search = D.search ?? 600, clear = D.clear ?? 22;
    for (let s = 0; s < search; s += 4) {
      const xx = x + (dir[0] / dl) * s, zz = z + (dir[1] / dl) * s;
      if (!world.isLand(xx, zz) && f.distanceAt(xx, zz) > clear) return { x: xx, z: zz };
    }
    return { x: x + 300, z: z - 100 };
  }

  // ------------------------------------------------------------------ rip-rap boulders
  _riprap(edges) {
    const { world, level } = this.o;
    const S = world.surface;
    const b = world.bounds, cx = (b.minX + b.maxX) / 2, cz = (b.minZ + b.maxZ) / 2;
    const nat = edges.filter((E) => E.kind === 'natural');
    // spacing grows with distance from the district so the instance count stays bounded
    const spacing = (x, z) => 1.8 + 3.2 * smoothstep(1300, 4200, Math.hypot(x - cx, z - cz));
    let est = 0;
    for (const E of nat) est += (E.L / spacing((E.a[0] + E.c[0]) / 2, (E.a[1] + E.c[1]) / 2)) * 2.4;
    const MAX = 2600, k = est > MAX ? est / MAX : 1;
    const list = [];
    const rnd = mulberry32(7123);
    for (const E of nat) {
      const sp = spacing((E.a[0] + E.c[0]) / 2, (E.a[1] + E.c[1]) / 2) * k;
      for (let s = rnd() * sp; s < E.L; s += sp * (0.75 + rnd() * 0.5)) {
        const x = E.a[0] + E.dx * s, z = E.a[1] + E.dz * s;
        if (world.isLand(x + E.ox * 2.2, z + E.oz * 2.2)) continue;
        const top = S.heightAt(x - E.ox * 0.5, z - E.oz * 0.5);
        const big = sp / 1.8;
        const rows = [[1.0 + rnd() * 1.1, level - 0.35 + rnd() * 0.3, (0.85 + rnd() * 0.8) * Math.min(1.8, big)]];
        if (rnd() < 0.8) rows.push([0.15 + rnd() * 0.5, level + 0.3 + rnd() * 0.3, (0.6 + rnd() * 0.6) * Math.min(1.6, big)]);
        if (top - level > 2.2 && rnd() < 0.6) rows.push([-0.1 + rnd() * 0.3, level + 0.9 + rnd() * 0.5, (0.5 + rnd() * 0.5) * Math.min(1.5, big)]);
        for (const [o, y, s0] of rows) {
          const j = (rnd() - 0.5) * sp * 0.6;
          list.push([x + E.ox * o + E.dx * j, y, z + E.oz * o + E.dz * j, s0, rnd(), rnd(), rnd()]);
        }
      }
    }
    if (!list.length) return;
    // faceted boulder: perturbed icosahedron (shared vertices move together)
    const g = new THREE.IcosahedronGeometry(1, 0);
    const P = g.attributes.position;
    for (let i = 0; i < P.count; i++) {
      const x = P.getX(i), y = P.getY(i), z = P.getZ(i);
      const h = hash2(Math.round(x * 1000) + 3 * Math.round(y * 1000), Math.round(z * 1000) + 7);
      const f = 0.72 + 0.45 * h;
      P.setXYZ(i, x * f, y * f * 0.8, z * f);
    }
    g.computeVertexNormals();
    const mat = this.rockMat = createRockMaterial(this.o.renderer, level);
    const im = new THREE.InstancedMesh(g, mat, list.length);
    im.name = 'riprap';
    const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), e = new THREE.Euler(), p = new THREE.Vector3(), s = new THREE.Vector3(), c = new THREE.Color();
    list.forEach((r, i) => {
      e.set((r[4] - 0.5) * 0.9, r[5] * Math.PI * 2, (r[6] - 0.5) * 0.9);
      q.setFromEuler(e);
      s.set(r[3] * (0.9 + r[5] * 0.4), r[3] * (0.6 + r[6] * 0.35), r[3] * (0.85 + r[4] * 0.4));
      m4.compose(p.set(r[0], r[1], r[2]), q, s);
      im.setMatrixAt(i, m4);
      const t = 0.8 + 0.35 * r[6];
      im.setColorAt(i, c.setRGB(t, t * (0.97 + 0.04 * r[4]), t * (0.92 + 0.06 * r[5])));
    });
    im.instanceMatrix.needsUpdate = true; if (im.instanceColor) im.instanceColor.needsUpdate = true;
    im.computeBoundingSphere();
    im.receiveShadow = true; im.castShadow = false;
    this.rockMesh = im;
    this.group.add(im);
    this.stats.rocks = list.length;
  }

  // ------------------------------------------------------------------ scrub / tree blobs on the natural hills
  _vegetation() {
    const { world } = this.o;
    const cls = this.cls, S = world.surface;
    const rnd = mulberry32(2718);
    const list = [];
    const step = 10;
    const OB = world.outerBox;
    const b = world.bounds;
    for (let z = OB.minZ + step / 2; z < OB.maxZ; z += step) {
      for (let x = OB.minX + step / 2; x < OB.maxX; x += step) {
        const c = cls.at(x, z);
        if (!c) continue;
        if (x > b.minX - 30 && x < b.maxX + 30 && z > b.minZ - 30 && z < b.maxZ + 30) continue; // the district has its own trees
        let pr = smoothstep(0.32, 0.85, c.forest) * (1 - c.urban) * 0.6;
        if (c.role === 'island' && c.urban > 0.4) pr = 0.035; // street trees on built-up island ground
        if (pr <= 0 || rnd() > pr) continue;
        const px = x + (rnd() - 0.5) * step * 0.9, pz = z + (rnd() - 0.5) * step * 0.9;
        if (!world.isLand(px, pz) || this.field.distanceAt(px, pz) > -4) continue;
        const h = S.heightAt(px, pz);
        if (h < 1.5) continue;
        list.push([px, h, pz, rnd(), rnd(), rnd()]);
      }
    }
    if (!list.length) return;
    let g = new THREE.OctahedronGeometry(1, 0); // 8 triangles, smooth-shaded into a soft blob
    g.deleteAttribute('uv'); g.deleteAttribute('normal');
    g = mergeVertices(g);
    const Pp = g.attributes.position;
    for (let i = 0; i < Pp.count; i++) {
      const x = Pp.getX(i), y = Pp.getY(i), z = Pp.getZ(i);
      const hh = hash2(Math.round(x * 997) + 5 * Math.round(y * 997), Math.round(z * 997) + 11);
      const f = 0.85 + 0.25 * hh;
      Pp.setXYZ(i, x * f, y * f, z * f);
    }
    g.computeVertexNormals();
    const mat = new THREE.MeshStandardMaterial({ roughness: 0.92, metalness: 0 });
    mat.onBeforeCompile = (sh) => {
      sh.vertexShader = sh.vertexShader.replace('#include <common>', '#include <common>\nvarying float vVY;').replace('#include <begin_vertex>', '#include <begin_vertex>\nvVY = position.y;');
      sh.fragmentShader = sh.fragmentShader.replace('#include <common>', '#include <common>\nvarying float vVY;')
        .replace('#include <color_fragment>', '#include <color_fragment>\ndiffuseColor.rgb *= 0.45 + 0.55 * smoothstep(-0.9, 0.7, vVY);');
    };
    mat.customProgramCacheKey = () => 'roadster-scrub-1';
    const im = new THREE.InstancedMesh(g, mat, list.length);
    im.name = 'hill-scrub';
    const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), p = new THREE.Vector3(), s = new THREE.Vector3(), c = new THREE.Color(), up = new THREE.Vector3(0, 1, 0);
    const pal = [[0.045, 0.062, 0.04], [0.025, 0.042, 0.028], [0.055, 0.06, 0.03], [0.06, 0.07, 0.05]];
    list.forEach((r, i) => {
      const w = 2.2 + r[3] * 3.6, hgt = w * (0.75 + r[4] * 0.6);
      q.setFromAxisAngle(up, r[5] * 6.283);
      m4.compose(p.set(r[0], r[1] + hgt * 0.55, r[2]), q, s.set(w, hgt, w * (0.85 + r[5] * 0.3)));
      im.setMatrixAt(i, m4);
      const k = pal[Math.floor(r[4] * pal.length) % pal.length], t = 0.8 + r[3] * 0.45;
      im.setColorAt(i, c.setRGB(k[0] * t, k[1] * t, k[2] * t));
    });
    im.instanceMatrix.needsUpdate = true; if (im.instanceColor) im.instanceColor.needsUpdate = true;
    im.computeBoundingSphere();
    im.receiveShadow = true;
    this.group.add(im);
    this.stats.scrub = list.length;
  }

  // ------------------------------------------------------------------ low context blocks (outer mainland, built-up island)
  _blocks() {
    const { world } = this.o;
    const S = world.surface;
    const lots = (world._outerLots || []).slice();
    // built-up island fill (o.islandFill picks the land polygon): blocks on its flat urban ground
    const cls = this.cls, rnd = mulberry32(8080);
    const isl = this.o.islandFill ? world.landPolys.find(this.o.islandFill) : null;
    if (isl) {
      const bb = isl.bb;
      for (let z = bb.minZ + 40; z < bb.maxZ; z += 75) for (let x = bb.minX + 40; x < bb.maxX; x += 95) {
        const c = cls.at(x, z);
        if (!c || c.role !== 'island' || c.urban < 0.45 || rnd() < 0.5) continue;
        const px = x + (rnd() - 0.5) * 30, pz = z + (rnd() - 0.5) * 20;
        if (this.field.distanceAt(px, pz) > -50) continue;
        lots.push([px, pz, 16 + rnd() * 26, 12 + rnd() * 16, 0.62, 4.5 + rnd() * 6, 2 + Math.floor(rnd() * 4)]);
      }
    }
    if (!lots.length) return;
    const pal = [[0.27, 0.26, 0.24], [0.32, 0.31, 0.29], [0.24, 0.22, 0.2], [0.3, 0.27, 0.22], [0.22, 0.23, 0.25], [0.34, 0.31, 0.27], [0.27, 0.21, 0.18]];
    const g = new THREE.BoxGeometry(1, 1, 1).translate(0, 0.5, 0);
    { // drop the bottom face (never seen)
      const idx = Array.from(g.index.array), keep = [];
      for (let i = 0; i < idx.length; i += 3) { const y = [0, 1, 2].map((k) => g.attributes.position.getY(idx[i + k])); if (!(y[0] === 0 && y[1] === 0 && y[2] === 0)) keep.push(idx[i], idx[i + 1], idx[i + 2]); }
      g.setIndex(keep);
    }
    const mat = this.blockMat = createBlockMaterial();
    const im = new THREE.InstancedMesh(g, mat, lots.length);
    im.name = 'context-blocks';
    const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), p = new THREE.Vector3(), s = new THREE.Vector3(), c = new THREE.Color(), up = new THREE.Vector3(0, 1, 0);
    let n = 0;
    for (const L of lots) {
      const [x, z, w, d, ang, h, ci] = L;
      const ca = Math.cos(ang), sa = Math.sin(ang);
      const corners = [[-w / 2, -d / 2], [w / 2, -d / 2], [w / 2, d / 2], [-w / 2, d / 2]].map(([a, bb]) => [x + a * ca - bb * sa, z + a * sa + bb * ca]);
      const hs = [...corners, [x, z]].map((pt) => S.heightAt(pt[0], pt[1]));
      const base = Math.min(...hs) - 0.8, top = Math.max(...hs) + h;
      q.setFromAxisAngle(up, -ang);
      m4.compose(p.set(x, base, z), q, s.set(w, top - base, d));
      im.setMatrixAt(n, m4);
      const k = pal[ci % pal.length], t = 0.85 + hash2(n, 3) * 0.3;
      im.setColorAt(n, c.setRGB(k[0] * t, k[1] * t, k[2] * t));
      n++;
      for (const bx of polygonToBoxes(corners, [])) world.colliders.add(bx[0], base, bx[2], top, bx[1], bx[3], 1);
    }
    im.count = n;
    im.instanceMatrix.needsUpdate = true; if (im.instanceColor) im.instanceColor.needsUpdate = true;
    im.computeBoundingSphere();
    im.castShadow = false; im.receiveShadow = true;
    this.group.add(im);
    this.stats.blocks = n;
  }

  // ------------------------------------------------------------------ boats
  _boats() {
    const { world, level } = this.o;
    const BO = this.o.boats || {};
    const bridge = [];
    if (BO.avoidRoads) for (const r of world.data.roads) {
      if (!r || !r.p || !BO.avoidRoads.test(r.n || '')) continue;
      for (let i = 0; i + 1 < r.p.length; i++) bridge.push([r.p[i][0], r.p[i][1], r.p[i + 1][0], r.p[i + 1][1]]);
    }
    try {
      this.boats = new Boats({
        isLand: (x, z) => world.isLand(x, z), shoreDist: (x, z) => this.field.distanceAt(x, z), level, bridge,
        noiseMap: this.noise, ferryDock: this.ferryDock || this._findFerryDock(),
        ferryRoute: BO.ferryRoute, ferries: BO.ferries, shipRoute: BO.shipRoute, sailZones: BO.sailZones, sailCount: BO.sailCount,
      });
      this.group.add(this.boats.build());
      this.stats.boats = this.boats.list.length;
      this.stats.boatTris = Math.round(this.boats.tris);
    } catch (e) { console.warn('[Waterfront] boats failed', e); this.boats = null; }
  }

  // ------------------------------------------------------------------ relief beyond the map (farland.json)
  _farland() {
    const F = this.o.farland || {};
    if (F.url === false || F.url === null) return;
    const base = (import.meta.env && import.meta.env.BASE_URL) || '/';
    fetch(F.url || `${base}map/farland.json`).then((r) => (r.ok ? r.json() : null)).then((j) => {
      if (!j || !Array.isArray(j.h)) return;
      try {
        const mesh = buildFarLand(j, this.o.world, this.o.level, F);
        if (mesh) { this.far = mesh; this.group.add(mesh); }
      } catch (e) { console.warn('[Waterfront] far land failed', e); }
    }).catch(() => { /* optional */ });
  }

  /**
   * Columns / bridge piers standing in the water: colliders crossing the waterline (collected once the world is
   * complete), merged into clusters (the bridge registers each caisson as many 1.5 m strips) and fitted with a
   * minimum-area oriented rectangle each.
   */
  _collectPiers() {
    const { world, level } = this.o;
    const boxes = [];
    for (const b of world.colliders.boxes) {
      if (!(b.minY < level + 1.5 && b.maxY > level + 0.5)) continue;
      const hx = (b.maxX - b.minX) / 2, hz = (b.maxZ - b.minZ) / 2;
      if (hx > 80 || hz > 80) continue;
      const cx = (b.minX + b.maxX) / 2, cz = (b.minZ + b.maxZ) / 2;
      if (world.isLand(cx, cz) || this.field.distanceAt(cx, cz) < 1.5) continue;
      boxes.push(b);
    }
    // union-find on touching boxes
    const par = boxes.map((_, i) => i);
    const find = (i) => { while (par[i] !== i) { par[i] = par[par[i]]; i = par[i]; } return i; };
    const tol = 0.8;
    const order = boxes.map((b, i) => i).sort((p, q) => boxes[p].minX - boxes[q].minX);
    for (let a = 0; a < order.length; a++) {
      const A = boxes[order[a]];
      for (let c = a + 1; c < order.length; c++) {
        const B = boxes[order[c]];
        if (B.minX > A.maxX + tol) break;
        if (B.minZ <= A.maxZ + tol && B.maxZ >= A.minZ - tol) par[find(order[a])] = find(order[c]);
      }
    }
    const groups = new Map();
    boxes.forEach((b, i) => { const r = find(i); if (!groups.has(r)) groups.set(r, []); groups.get(r).push(b); });
    const out = [];
    for (const g of groups.values()) {
      const pts = [];
      for (const b of g) pts.push([b.minX, b.minZ], [b.maxX, b.minZ], [b.maxX, b.maxZ], [b.minX, b.maxZ]);
      let best = null;
      for (let deg = 0; deg < 90; deg += 3) {
        const a = (deg * Math.PI) / 180, c = Math.cos(a), sn = Math.sin(a);
        let u0 = Infinity, u1 = -Infinity, v0 = Infinity, v1 = -Infinity;
        for (const [x, z] of pts) { const u = x * c + z * sn, v = -x * sn + z * c; if (u < u0) u0 = u; if (u > u1) u1 = u; if (v < v0) v0 = v; if (v > v1) v1 = v; }
        const area = (u1 - u0) * (v1 - v0);
        if (!best || area < best.area) best = { area, c, sn, u: (u0 + u1) / 2, v: (v0 + v1) / 2, hu: (u1 - u0) / 2, hv: (v1 - v0) / 2 };
      }
      // strips of a rotated rectangle overshoot its corners by up to a strip width: shrink a little
      const sh = g.length > 2 ? 0.6 : 0;
      out.push([best.u * best.c - best.v * best.sn, best.u * best.sn + best.v * best.c, Math.max(0.3, best.hu - sh), Math.max(0.3, best.hv - sh), best.c, best.sn]);
    }
    this.piers = out;
    this.stats.waterColumns = out.length;
  }

  _updatePiers(cam) {
    if (!this.piers) this._collectPiers();
    if (!cam || (this._pierCam && Math.abs(cam.x - this._pierCam[0]) + Math.abs(cam.z - this._pierCam[1]) < 15)) return;
    this._pierCam = [cam.x, cam.z];
    const near = [];
    for (const p of this.piers) { const d = Math.hypot(p[0] - cam.x, p[1] - cam.z); if (d < 900) near.push([d, p]); }
    near.sort((a, b) => a[0] - b[0]);
    const u = this.o.waterMat.userData.u, n = Math.min(16, near.length);
    for (let i = 0; i < n; i++) { const p = near[i][1]; u.uPiers.value[i].set(p[0], p[1], p[2], p[3]); u.uPierR.value[i].set(p[4], p[5]); }
    u.uPierN.value = n;
  }

  /** Per frame: uniform updates + boat transforms. */
  update(dt, cam, time, atmo) {
    const night = atmo ? atmo.windowGlow : 0, dark = atmo ? atmo.darkness : 0;
    if (this.water) this.water.update(time, night);
    if (this.o.world.loaded) this._updatePiers(cam);
    if (cam) { // detail LOD: sub-pixel from high up / far away, and they are always-drawn merged meshes
      if (this.dressMesh) {
        const bb = this.dressMesh.geometry.boundingBox;
        const dx = Math.max(bb.min.x - cam.x, 0, cam.x - bb.max.x), dz = Math.max(bb.min.z - cam.z, 0, cam.z - bb.max.z);
        this.dressMesh.visible = cam.y < 320 && Math.hypot(dx, dz) < 1600;
      }
      if (this.rockMesh) this.rockMesh.visible = cam.y < 250;
    }
    if (this.o.pondMat) this.o.pondMat.userData.u.uTime.value = time;
    this.o.groundMat.userData.u.uNight.value = night;
    if (this.wallMat) this.wallMat.userData.u.uTime.value = time;
    if (this.dressMat) { this.dressMat.userData.u.uTime.value = time; this.dressMat.userData.u.uNight.value = night; }
    if (this.blockMat) this.blockMat.userData.u.uNight.value = night;
    if (this.far) {
      const fu = this.far.material.userData.u;
      fu.uNight.value = night;
      const fp = atmo && atmo.fogParams;
      if (fp) { fu.uFogP.value.set(fp.haze ?? 0.0002, fp.hazeK ?? 1 / 700, fp.hazeBase || 0, fp.mist ?? 0.0001); fu.uMistK.value = fp.mistK ?? 1 / 40; }
      const sc = atmo && atmo.skyCubeRT && atmo.skyCubeRT.texture;
      fu.uSkyCube.value = sc || null; fu.uHasSky.value = sc ? 1 : 0;
    }
    if (this.boats) this.boats.update(dt, cam, time, night, 1 - 0.94 * dark);
  }
}

// ====================================================================== far land
/**
 * Mesh of farland.json outside the map square: terrain (sea cells dropped, coast cells kept with the sea
 * vertices pushed under the water), a crack-filling curtain along the square's edge, city blocks on the flats
 * (denser / taller in `clusters`) and container cranes along the `cranes.region` waterfront. Triangles are
 * emitted far -> near from the map centre (painter order for the depth-compressed part).
 * F (bake frame): natural(bx, bz) -> no city (e.g. a national park headland), urbanMax(bx, bz) -> height (m) above
 * which the land stops being urban, clusters [[bx, bz, r, hMin, hMax, density]], cranes { region: [x0, x1, z0, z1],
 * max = 16, spacing = 260 } | null, blockAngle(bx, bz) -> street-grid angle of the far blocks.
 */
export function buildFarLand(j, world, level, F = {}) {
  const off = world.offset || { x: 0, z: 0 };
  const HALF = (world.data && world.data.outer ? world.data.outer : 9000) / 2;
  const n = j.nx, cell = j.cell, x0 = j.origin[0], z0 = j.origin[1];
  const H = j.h;
  const S = world.surface;
  const OB = world.outerBox;
  const hAt = (i, k) => H[clamp(k, 0, n - 1) * n + clamp(i, 0, n - 1)];
  const inBox = (bx, bz) => bx > -HALF + 1 && bx < HALF - 1 && bz > -HALF + 1 && bz < HALF - 1;
  const onBoxEdge = (bx, bz) => Math.abs(Math.abs(bx) - HALF) < 1 && Math.abs(bz) <= HALF + 1 || Math.abs(Math.abs(bz) - HALF) < 1 && Math.abs(bx) <= HALF + 1;
  const SEA_Y = level - 14;
  // ---- vertex heights (world y)
  const Y = new Float32Array(n * n), U = new Float32Array(n * n);
  const natural = F.natural || (() => false), urbanMaxAt = F.urbanMax || (() => 95);
  for (let k = 0; k < n; k++) for (let i = 0; i < n; i++) {
    const bx = x0 + i * cell, bz = z0 + k * cell, h = hAt(i, k);
    let y = h > 0.5 ? h : SEA_Y;
    if (onBoxEdge(bx, bz) || inBox(bx, bz)) { // match / tuck under the detailed ground at the seam
      const wx = bx - off.x, wz = bz - off.z;
      const cx = clamp(wx, OB.minX + 0.5, OB.maxX - 0.5), cz = clamp(wz, OB.minZ + 0.5, OB.maxZ - 0.5);
      y = world.isLand(cx, cz) ? S.heightAt(cx, cz) - 1.0 : SEA_Y;
    }
    Y[k * n + i] = y;
    const urbanMax = natural(bx, bz) ? 0 : urbanMaxAt(bx, bz);
    U[k * n + i] = h > 0.5 ? (1 - smoothstep(urbanMax * 0.6, urbanMax, h)) * (natural(bx, bz) ? 0.15 : 1) : 0;
  }
  const buf = new MBuf(['aUrb']);
  const prims = []; // {d, fn}
  const cxm = (world.bounds.minX + world.bounds.maxX) / 2, czm = (world.bounds.minZ + world.bounds.maxZ) / 2;
  const nrm = (i, k) => {
    const dx = (Y[k * n + Math.min(n - 1, i + 1)] - Y[k * n + Math.max(0, i - 1)]) / (2 * cell);
    const dz = (Y[Math.min(n - 1, k + 1) * n + i] - Y[Math.max(0, k - 1) * n + i]) / (2 * cell);
    const l = Math.hypot(dx, 1, dz); return [-dx / l, 1 / l, -dz / l];
  };
  const colAt = (i, k) => {
    const bx = x0 + i * cell, bz = z0 + k * cell, h = Math.max(0, hAt(i, k)), N = nrm(i, k);
    const nz = hash2(i * 3 + 1, k * 7 + 2), nz2 = hash2(Math.floor(i / 4) + 9, Math.floor(k / 4) + 3);
    const slope = 1 - N[1];
    const north = clamp(-N[2] * 3, 0, 1); // north-facing slopes: oak / bay woodland
    let grass = [0.2, 0.155, 0.075], wood = [0.045, 0.06, 0.032];
    if (natural(bx, bz)) { grass = [0.17, 0.145, 0.075]; }
    const woodW = clamp(0.25 + north * 0.55 + (nz2 - 0.5) * 0.6 + slope * 0.3, 0, 1);
    let c = grass.map((g, q) => g + (wood[q] - g) * woodW);
    const u = U[k * n + i];
    const city = [0.13 + 0.03 * nz, 0.125 + 0.025 * nz, 0.115 + 0.02 * nz];
    c = c.map((v, q) => v + (city[q] - v) * u * 0.85);
    if (h < 4) c = c.map((v, q) => v * 0.75 + [0.06, 0.065, 0.045][q] * 0.25);
    return c;
  };
  const vid = new Int32Array(n * n).fill(-1);
  const vert = (i, k) => {
    const id = k * n + i;
    if (vid[id] >= 0) return vid[id];
    const bx = x0 + i * cell, bz = z0 + k * cell, N = nrm(i, k);
    return (vid[id] = buf.v(bx - off.x, Y[id], bz - off.z, N[0], N[1], N[2], colAt(i, k), 0, 0, { aUrb: U[id] }));
  };
  // terrain cells outside the square
  for (let k = 0; k < n - 1; k++) for (let i = 0; i < n - 1; i++) {
    const bx0 = x0 + i * cell, bz0 = z0 + k * cell, bx1 = bx0 + cell, bz1 = bz0 + cell;
    if (bx0 >= -HALF - 1 && bx1 <= HALF + 1 && bz0 >= -HALF - 1 && bz1 <= HALF + 1) continue;
    const hs = [hAt(i, k), hAt(i + 1, k), hAt(i, k + 1), hAt(i + 1, k + 1)];
    if (hs.every((h) => h <= 0.5)) continue;
    const d = Math.hypot((bx0 + cell / 2) - off.x - cxm, (bz0 + cell / 2) - off.z - czm);
    if (d > 21500 && Math.max(...hs) < 250) continue; // only the high ridges that far out
    prims.push({ d, fn: () => { const a = vert(i, k), b = vert(i + 1, k), c = vert(i + 1, k + 1), e = vert(i, k + 1); buf.idx.push(a, e, c, a, c, b); } });
  }
  // curtain along the square's edge (hides seams between the 20 m and 250 m lattices)
  const curtain = (ax, az, bx, bz, nx, nz) => {
    const L = Math.hypot(bx - ax, bz - az), steps = Math.ceil(L / 20);
    for (let s = 0; s < steps; s++) {
      const t0 = s / steps, t1 = (s + 1) / steps;
      const p = [ax + (bx - ax) * t0, az + (bz - az) * t0], q = [ax + (bx - ax) * t1, az + (bz - az) * t1];
      const ip = [p[0] - nx * 0.5, p[1] - nz * 0.5], iq = [q[0] - nx * 0.5, q[1] - nz * 0.5];
      if (!world.isLand(ip[0], ip[1]) && !world.isLand(iq[0], iq[1])) continue;
      const hp = world.isLand(ip[0], ip[1]) ? S.heightAt(ip[0], ip[1]) : level, hq = world.isLand(iq[0], iq[1]) ? S.heightAt(iq[0], iq[1]) : level;
      const d = Math.hypot((p[0] + q[0]) / 2 - cxm, (p[1] + q[1]) / 2 - czm);
      const c = [0.11, 0.105, 0.095];
      prims.push({ d: d + 1, fn: () => buf.quad([p[0], hp + 0.05, p[1]], [q[0], hq + 0.05, q[1]], [q[0], Math.min(hq, hp) - 25, q[1]], [p[0], Math.min(hq, hp) - 25, p[1]], [-nx, 0, -nz], c, null, { aUrb: 0.6 }) });
    }
  };
  curtain(OB.minX, OB.minZ, OB.maxX, OB.minZ, 0, -1);
  curtain(OB.maxX, OB.minZ, OB.maxX, OB.maxZ, 1, 0);
  curtain(OB.maxX, OB.maxZ, OB.minX, OB.maxZ, 0, 1);
  curtain(OB.minX, OB.maxZ, OB.minX, OB.minZ, -1, 0);
  // ---- city blocks on the flats (+ taller / denser clusters: far downtowns)
  const rnd = mulberry32(5150);
  const sampleH = (bx, bz) => {
    const fx = (bx - x0) / cell, fz = (bz - z0) / cell, i = Math.floor(fx), k = Math.floor(fz), tx = fx - i, tz = fz - k;
    return (hAt(i, k) * (1 - tx) + hAt(i + 1, k) * tx) * (1 - tz) + (hAt(i, k + 1) * (1 - tx) + hAt(i + 1, k + 1) * tx) * tz;
  };
  const clusters = F.clusters || [], blockAngle = F.blockAngle || (() => 0);
  const G = 230;
  let blocks = 0;
  outer: for (let bz = z0 + G; bz < -z0 - G; bz += G) for (let bx = x0 + G; bx < -x0 - G; bx += G) {
    if (Math.abs(bx) < HALF + 150 && Math.abs(bz) < HALF + 150) continue;
    const px = bx + (rnd() - 0.5) * G * 0.7, pz = bz + (rnd() - 0.5) * G * 0.7;
    const h = sampleH(px, pz);
    if (h < 1.5 || natural(px, pz)) continue;
    const ui = clamp(Math.round((px - x0) / cell), 0, n - 1), uk = clamp(Math.round((pz - z0) / cell), 0, n - 1);
    const u = U[uk * n + ui];
    let dens = 0.26 * u, hmin = 7, hmax = 18;
    for (const [cx, cz, r, h0, h1, de] of clusters) {
      const dd = Math.hypot(px - cx, pz - cz);
      if (dd < r) { const f = 1 - dd / r; dens = Math.max(dens, de * (0.5 + 0.5 * f)); hmin = h0; hmax = h0 + (h1 - h0) * f * f; }
    }
    // stay off steep ground
    const slope = Math.abs(sampleH(px + 60, pz) - sampleH(px - 60, pz)) + Math.abs(sampleH(px, pz + 60) - sampleH(px, pz - 60));
    if (slope > 30 || rnd() > dens) continue;
    const w = 30 + rnd() * 45, dpt = 22 + rnd() * 35, hb = hmin + rnd() * (hmax - hmin), ang = blockAngle(bx, bz);
    const wx = px - off.x, wz = pz - off.z;
    const base = Math.min(sampleH(px - w / 2, pz - dpt / 2), sampleH(px + w / 2, pz + dpt / 2), h) - 2;
    const cc = 0.26 + rnd() * 0.22, col = [cc, cc * 0.97, cc * 0.92];
    const d = Math.hypot(wx - cxm, wz - czm);
    prims.push({ d, fn: () => buf.obox(wx, base, wz, w, hb + (h - base), dpt, ang, col, { aUrb: 1.6 }) });
    blocks++;
    if (blocks > 1500) break outer;
  }
  // ---- container cranes along a port waterfront (F.cranes.region)
  const shore = [], CR = F.cranes || null, crReg = CR && CR.region;
  if (crReg) for (let k = 1; k < n - 1; k++) for (let i = 1; i < n - 1; i++) {
    const bx = x0 + i * cell, bz = z0 + k * cell;
    if (bx < crReg[0] || bx > crReg[1] || bz < crReg[2] || bz > crReg[3]) continue;
    if (hAt(i, k) <= 0.5) continue;
    let sx = 0, sz = 0, cnt = 0;
    for (const [di, dk] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) if (hAt(i + di, k + dk) <= 0.5) { sx += di; sz += dk; cnt++; }
    if (cnt) shore.push([bx, bz, sx / cnt, sz / cnt]);
  }
  shore.sort((p, q) => p[1] - q[1] || p[0] - q[0]);
  const cranes = [];
  const crMax = (CR && CR.max) ?? 16, crGap = (CR && CR.spacing) ?? 260;
  for (const s of shore) { if (cranes.length >= crMax) break; if (cranes.every((c) => Math.hypot(c[0] - s[0], c[1] - s[1]) > crGap)) cranes.push(s); }
  for (const [bx, bz, sx, sz] of cranes) {
    const wx = bx - off.x - sx * 60, wz = bz - off.z - sz * 60, ang = Math.atan2(sz, sx); // boom points to the water
    const y = Math.max(2, sampleH(bx - sx * 60, bz - sz * 60));
    const ca = Math.cos(ang), sa = Math.sin(ang);
    const at = (a, bq) => [wx + a * ca - bq * sa, wz + a * sa + bq * ca];
    const white = [0.62, 0.62, 0.6], red = [0.5, 0.1, 0.08];
    const d = Math.hypot(wx - cxm, wz - czm);
    prims.push({ d, fn: () => {
      for (const [a, bq] of [[-10, -9], [-10, 9], [10, -9], [10, 9]]) { const p = at(a, bq); buf.obox(p[0], y, p[1], 2, 40, 2, ang, white, { aUrb: 0 }); }
      let p = at(0, -9); buf.obox(p[0], y + 38, p[1], 22, 3, 2.2, ang, white, { aUrb: 0 });
      p = at(0, 9); buf.obox(p[0], y + 38, p[1], 22, 3, 2.2, ang, white, { aUrb: 0 });
      p = at(18, 0); buf.obox(p[0], y + 41, p[1], 100, 3.2, 3, ang, white, { aUrb: 0 }); // boom over the water
      p = at(-6, 0); buf.obox(p[0], y + 44, p[1], 12, 6, 10, ang, red, { aUrb: 0.5 });   // machinery house
      p = at(-4, 0); buf.obox(p[0], y + 50, p[1], 2.5, 22, 2.5, ang, white, { aUrb: 0 }); // A-frame
    } });
  }
  // painter order: far -> near
  prims.sort((p, q) => q.d - p.d);
  for (const p of prims) p.fn();
  if (!buf.idx.length) return null;
  const geo = buf.geometry();
  geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e6);
  const mat = createFarMaterial(level);
  const mesh = new THREE.Mesh(geo, mat);
  mesh.name = 'far-land';
  mesh.frustumCulled = false;
  mesh.matrixAutoUpdate = false; mesh.updateMatrix();
  mesh.onBeforeRender = (r, s, cam) => { if (cam.far) mat.userData.u.uFar.value = cam.far; };
  mesh.userData.stats = { tris: buf.idx.length / 3, blocks, cranes: cranes.length };
  console.info('[Waterfront] far land', mesh.userData.stats);
  return mesh;
}

function createFarMaterial(level) {
  const m = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.95, metalness: 0 });
  const u = {
    uFar: { value: 6000 }, uNight: { value: 0 }, uLevel: { value: level },
    // exponential haze + mist layers (same model / parameters as the post fog: Atmosphere.fogParams)
    uFogP: { value: new THREE.Vector4(0.0002, 1 / 700, 0, 0.0001) }, uMistK: { value: 1 / 40 },
    uSkyCube: { value: null }, uHasSky: { value: 0 },
  };
  m.userData.u = u;
  m.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, u);
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', '#include <common>\nuniform float uFar;\nattribute float aUrb;\nvarying float vUrb;\nvarying vec3 vFWP;\nvarying vec3 vFWN;\nvarying float vFD;')
      .replace('#include <project_vertex>', '#include <project_vertex>\nif ( vFD > uFar * 0.985 ) gl_Position.z = gl_Position.w; // depth 1: the post fog treats it as sky')
      .replace('#include <begin_vertex>', /* glsl */`
vec3 transformed = vec3( position );
{
  // keep the true position for shading / haze; pull vertices beyond the far plane inside it along the view ray
  vFWP = (modelMatrix * vec4(transformed, 1.0)).xyz;
  vFWN = objectNormal;
  vUrb = aUrb;
  vec3 rel = vFWP - cameraPosition;
  float D = length(rel);
  vFD = D;
  float R0 = uFar * 0.985, R1 = uFar * 0.998;
  if (D > R0) {
    float Dc = R0 + (R1 - R0) * (1.0 - exp(-(D - R0) / 8000.0));
    transformed = cameraPosition + rel * (Dc / D); // modelMatrix is identity for this mesh
  }
}`);
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', `#include <common>
uniform float uNight; uniform float uLevel; uniform float uFar; uniform vec4 uFogP; uniform float uMistK;
uniform samplerCube uSkyCube; uniform float uHasSky;
varying float vUrb; varying vec3 vFWP; varying vec3 vFWN; varying float vFD;
${HASH_GLSL}
float fhLayer(float y0, float y1, float dist, float d, float k) {
  float x = k * (y1 - y0);
  float f = abs(x) > 1e-3 ? (1.0 - exp(-x)) / x : 1.0 - 0.5 * x;
  return d * exp(-k * clamp(y0, -100.0, 4000.0)) * dist * f;
}`)
      .replace('#include <emissivemap_fragment>', /* glsl */`#include <emissivemap_fragment>
{
  // city lights: sparkle on the flats, window bands on the blocks
  float u = clamp(vUrb, 0.0, 1.0);
  float wall = step(1.2, vUrb) * (1.0 - step(0.5, abs(vFWN.y)));
  float sp = shHash(floor(vFWP.xz / 45.0)) * shHash(floor(vFWP.xz / 13.0) + 3.1);
  float lights = u * (0.35 + 0.65 * smoothstep(0.35, 0.9, sp)) + wall * 0.9;
  totalEmissiveRadiance += vec3(1.0, 0.68, 0.38) * (lights * uNight * 0.55);
}`)
      .replace('#include <opaque_fragment>', /* glsl */`#include <opaque_fragment>
// beyond the far plane the post fog / scene fog cannot see this geometry (it is written at depth 1 = sky):
// haze it here with the same layered model, toward the clean sky radiance in the view direction
bool fBeyond = vFD > uFar * 0.985;
{
  vec3 dir = normalize(vFWP - cameraPosition);
  vec3 skyC = vec3(0.6, 0.66, 0.72);
  #ifdef USE_FOG
  skyC = fogColor;
  #endif
  if (uHasSky > 0.5) skyC = textureCube(uSkyCube, normalize(vec3(dir.x, max(dir.y, 0.02), dir.z))).rgb;
  if (vFWP.y < uLevel - 0.05) {
    gl_FragColor.rgb = skyC; // far water: the bay meets the sky
  } else if (fBeyond) {
    float tau = fhLayer(cameraPosition.y - uFogP.z, vFWP.y - uFogP.z, vFD, uFogP.x, uFogP.y) + fhLayer(cameraPosition.y, vFWP.y, vFD, uFogP.w, uMistK);
    float T = exp(-tau * 0.85) * smoothstep(uFar * 0.985, uFar * 1.55, vFD);
    gl_FragColor.rgb = mix(skyC, gl_FragColor.rgb, T);
  }
}`)
      .replace('#include <fog_fragment>', 'if ( !fBeyond ) {\n#include <fog_fragment>\n}');
  };
  m.customProgramCacheKey = () => 'roadster-farland-3';
  return m;
}
