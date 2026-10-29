// Buildings: OSM footprints -> realistic facades.
//
// Materials: ONE MeshPhysicalMaterial (patched) shades every building surface. Per vertex it gets a mode
// (wall with procedural windows, recessed glass pane, plain PBR, glass curtain wall, fire-escape grate,
// sign / awning, clock face, LED lights) and indices into two texture arrays:
//   - window-style layers (FacadeTextures.js FACADE: frames / glass mask, drawn procedurally),
//   - PBR wall / roof sets (Poly Haven CC0 photo-scanned albedo + normal + roughness + AO, real-world scale).
// Glass shows an interior-mapped room (parallax rooms with furniture, blinds, ceiling lights at night) behind
// a reflective pane; texture windows get a parallax reveal so even flat walls read as recessed.
//
// Geometry (all merged, few draw calls):
//   base chunks  (2x2 inner + 4 outer sectors): shells of most buildings, roofs, parapets with coping, cornices,
//                string courses, fascia, rooftop plant (penthouses, HVAC, water tanks, antennas, skylights,
//                roof gardens), storefront signs + awnings, Victorian bay windows, curtain walls.
//   detail cells (4x4 THREE.LOD over the detailed district): near level = walls with REAL openings (piers and
//                spandrels as geometry) + instanced window units (0.12-0.3 m deep reveals, glass pane, sill,
//                lintel) + fire escapes and balconies; far level = the flat shell with texture windows.
// Terrain: every footprint stands at the ground height of its LOWEST outline point; walls continue 1 m below.
import * as THREE from 'three';
import { FACADE, WIN_RECTS, PBR, PBR_SETS } from './FacadeTextures.js';
import { signedArea, sanitizeRing, pointInPoly, ringBounds, centroid, clamp, hashStr, mulberry32 } from './geom.js';
import { GBUFFER_CHUNK_TAILS } from '../render/GBuffer.js'; // G-buffer captures for the chunks this material replaces

export const MODE = { WALL: 0, WALLONLY: 1, PANE: 2, PLAIN: 3, GRATE: 4, SIGN: 5, CLOCK: 6, CURTAIN: 7, LED: 8 };
/** 4th component of aInfo: PBR layer + 64 * mode */
export const enc = (pbr, mode) => pbr + 64 * mode;

// ================================================================== material
const GLSL_COMMON = /* glsl */`
uniform highp sampler2DArray uAlb;
uniform highp sampler2DArray uDat;
uniform highp sampler2DArray uPA;
uniform highp sampler2DArray uPN;
uniform float uPbrSize[16];
uniform float uNight;
uniform float uLitFrac;
uniform float uWinGain;
uniform float uTime;
uniform float uClock;
varying vec2 vFuv;
varying vec2 vWuv;
varying vec4 vInfo;
varying vec3 vTint;
varying vec3 vWPos;
varying vec3 vWall; // street height, overhang (AO) height, 1 = wall vertex
float fHash(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
// interior mapping: room box behind the glass. id = room cell, f = entry point (0..1), rt = tangent-space ray
// (x along the wall, y up, z out of the wall; rt.z < 0 looking in). Returns lit colour (pre-exposure).
vec3 fRoom(vec2 id, vec2 f, vec3 rt, float seed, float lit, vec3 lcol, float day, float store, out float glow) {
  vec3 r = vec3(rt.x * 1.15, rt.y, min(rt.z, -0.06) * 0.8);
  vec3 rr = vec3(r.x >= 0.0 ? max(r.x, 1e-4) : min(r.x, -1e-4), r.y >= 0.0 ? max(r.y, 1e-4) : min(r.y, -1e-4), r.z);
  vec3 p = vec3(f, 0.0);
  vec3 tB = (vec3(rr.x > 0.0 ? 1.0 : 0.0, rr.y > 0.0 ? 1.0 : 0.0, -1.0) - p) / rr;
  glow = 0.0;
  float t = min(tB.x, min(tB.y, tB.z));
  vec3 h = p + rr * t;
  float h1 = fHash(id + seed * 31.7), h2 = fHash(id.yx + seed * 7.3 + 3.1);
  vec3 wallC = mix(mix(vec3(0.7, 0.64, 0.55), vec3(0.5, 0.54, 0.58), h1), vec3(0.78, 0.74, 0.68), h2 * 0.5);
  vec3 floorC = mix(vec3(0.16, 0.14, 0.13), vec3(0.36, 0.26, 0.17), h2);
  vec3 c; float lightTerm;
  if (t == tB.z) { // back wall (+ desk / shelf silhouettes)
    c = wallC; lightTerm = 0.75;
    float dk = step(h.y, 0.34) * step(0.12, h.x) * step(h.x, 0.88);
    c = mix(c, vec3(0.18, 0.15, 0.12), dk * step(0.35, h1));
    c = mix(c, vec3(0.3, 0.32, 0.36), step(0.45, h.y) * step(h.y, 0.8) * step(abs(h.x - 0.5), 0.12) * step(0.7, h2));
  } else if (t == tB.y) {
    if (rr.y > 0.0) { // ceiling with light panels
      c = vec3(0.86, 0.86, 0.84); lightTerm = 0.55;
      vec2 pn = fract(vec2(h.x * 2.0, -h.z * 2.5));
      float panel = step(0.25, pn.x) * step(pn.x, 0.75) * step(0.3, pn.y) * step(pn.y, 0.7);
      glow = panel;
    } else { c = floorC; lightTerm = 0.6; }
  } else { c = wallC * 0.85; lightTerm = 0.6; }
  float depth = clamp(-h.z, 0.0, 1.0);
  // a desk in front of the back wall for half the rooms
  vec3 dmin = vec3(0.15, 0.0, -0.62), dmax = vec3(0.85, 0.3, -0.42);
  vec3 t0 = (dmin - p) / rr, t1 = (dmax - p) / rr;
  vec3 tn = min(t0, t1), tf = max(t0, t1);
  float te = max(max(tn.x, tn.y), tn.z), tx = min(min(tf.x, tf.y), tf.z);
  if (h1 > 0.5 && te < tx && te > 0.0 && te < t) { c = vec3(0.16, 0.13, 0.1) + 0.1 * h2; depth = 0.5; lightTerm = 0.5; }
  float dayK = mix(1.0, 0.25, depth) * lightTerm * (0.07 + 0.1 * store);
  float nightK = lit * (0.55 + 0.45 * lightTerm) * mix(1.0, 0.6, depth);
  glow *= lit;
  return c * (dayK * day) * vec3(1.0, 0.98, 0.95) + c * lcol * nightK;
}
`;

const FRAG_MAIN = /* glsl */`
float fModeF = floor(vInfo.w / 64.0 + 0.001);
float fPbrF = vInfo.w - fModeF * 64.0;
int fMode = int(fModeF + 0.5);
vec3 fNg = normalize(vNormal);
vec3 fUpV = normalize((viewMatrix * vec4(0.0, 1.0, 0.0, 0.0)).xyz);
vec3 fTv = cross(fNg, fUpV);
fTv = dot(fTv, fTv) < 1e-4 ? normalize((viewMatrix * vec4(1.0, 0.0, 0.0, 0.0)).xyz) : normalize(fTv);
vec3 fBv = cross(fTv, fNg);
vec3 fRd = normalize(-vViewPosition);
vec3 fRt = vec3(dot(fRd, fTv), dot(fRd, fBv), dot(fRd, fNg));
int fPi = int(fPbrF + 0.5);
vec2 fWu = vWuv / uPbrSize[fPi];
vec4 fPA = texture(uPA, vec3(fWu, fPbrF));
vec4 fPN = texture(uPN, vec3(fWu, fPbrF));
vec3 fCol = fPA.rgb * vTint;
float fRough = fPN.b;
float fMetal = 0.0;
float fAO = fPA.a;
vec2 fNxy = fPN.rg * 2.0 - 1.0;
float fNStr = 1.0;
vec3 fEmis = vec3(0.0);
float fGlass = 0.0;
vec3 fSpec = vec3(0.04);
float fEnv = 1.0;
float fDay = 1.0 - uNight;
if (fMode == 0 || fMode == 2) {
  float layer = vInfo.x;
  float store = 0.0;
  if (fMode == 0 && vInfo.y >= 0.0 && vFuv.y < 1.0) layer = vInfo.y;
  if (layer > 7.5 && layer < 8.5) store = 1.0;
  if (layer >= 0.0) {
    vec2 uv = vFuv;
    vec4 d0 = texture(uDat, vec3(uv * 0.25, layer));
    bool reveal = false;
    if (fMode == 0 && d0.r > 0.75) {
      vec2 sh = fRt.xy / max(0.25, -fRt.z) * vec2(0.06, 0.05) * (1.0 + store);
      vec4 d1 = texture(uDat, vec3((uv + sh) * 0.25, layer));
      if (d1.r < 0.75) reveal = true; else { uv += sh; d0 = d1; }
    }
    if (reveal) {
      fCol *= 0.42; fNStr = 0.4;
    } else if (d0.r > 0.75) {
      // ---- glass + interior
      vec2 id = floor(uv) + vec2(vInfo.z * 113.0, vInfo.z * 71.0);
      vec2 f = fract(uv);
      float fl = fHash(vec2(vInfo.z * 17.0, floor(uv.y)));
      float hp = mix(fHash(id + 3.7), fl, 0.55);
      float litP = store > 0.5 ? 0.92 : uLitFrac * (0.5 + 1.0 * fract(vInfo.z * 7.13));
      float lit = step(1.0 - litP, hp);
      vec3 lc = mix(vec3(1.0, 0.66, 0.36), vec3(0.78, 0.87, 1.0), step(0.72, fHash(id + 9.1)));
      if (store > 0.5) lc = vec3(1.0, 0.84, 0.64);
      lc *= uWinGain * (0.55 + 0.8 * fHash(id + 5.3)) * (1.0 + 0.6 * store) * uNight;
      float glow = 0.0;
      vec3 room = fRoom(id, f, fRt, vInfo.z, lit, lc, fDay, store, glow);
      room += lc * glow * 1.4;
      // blinds (at the glass plane)
      float hb = fHash(id + 21.7);
      float blind = hb < 0.42 && store < 0.5 ? (0.25 + 0.6 * fHash(id + 2.2)) : 0.0;
      if (f.y > 1.0 - blind) {
        float slat = 0.8 + 0.2 * step(0.5, fract(f.y * 26.0));
        vec3 bc = mix(vec3(0.85, 0.82, 0.74), vec3(0.6, 0.62, 0.62), step(0.5, fHash(id + 4.4))) * slat;
        room = bc * (0.11 * fDay) + bc * lc * lit * 0.65;
      }
      // distant windows: fade the parallax room to its average (no moire)
      float px = max(fwidth(uv.x), fwidth(uv.y));
      float far = smoothstep(0.12, 0.45, px);
      vec3 avg = vec3(0.6, 0.57, 0.52) * (0.04 * fDay * (1.0 + store) + lc * lit * 0.62);
      fEmis = mix(room, avg, far);
      fGlass = 1.0;
      fCol = vec3(0.025);
      fRough = 0.05;
      fSpec = vec3(0.05);
      fEnv = 1.5;
      fNxy = (vec2(fHash(id + 0.3), fHash(id + 0.7)) - 0.5) * 0.035;
      fNStr = 1.0;
    } else if (d0.r > 0.25) {
      // ---- trim (frames, sills, spandrels)
      vec4 ta = texture(uAlb, vec3(uv * 0.25, layer));
      fCol = ta.rgb * mix(1.0, fPA.a, 0.5) * (0.85 + 0.3 * dot(fPA.rgb, vec3(0.33)));
      fRough = d0.g; fMetal = d0.b; fNStr = 0.25;
    }
  }
} else if (fMode == 7) {
  // ---- glass curtain wall: analytic mullion grid, spandrels, interior, coated reflective glass
  float variant = vInfo.x;
  vec2 g = vFuv;
  vec2 fw = fwidth(g) + 1e-4;
  float fu = fract(g.x), fv = fract(g.y);
  float mw = variant > 2.5 ? 0.075 : 0.03;
  float hw = variant > 2.5 ? 0.012 : 0.022;
  float md = min(fu, 1.0 - fu);
  float mull = 1.0 - smoothstep(mw - fw.x, mw + fw.x, md);
  float sp0 = (variant > 0.5 && variant < 1.5) ? 0.0 : ((variant > 1.5 && variant < 2.5) ? 0.33 : 0.23);
  float spand = sp0 > 0.0 ? 1.0 - smoothstep(sp0 - fw.y, sp0 + fw.y, fv) : 0.0;
  float tr = 1.0 - smoothstep(hw - fw.y, hw + fw.y, min(fv, 1.0 - fv));
  float frame = max(mull, tr);
  vec2 cell = vec2(floor(g.x * 0.5), floor(g.y));
  vec2 id = cell + vec2(vInfo.z * 113.0, vInfo.z * 71.0);
  float fl = fHash(vec2(vInfo.z * 17.0, cell.y));
  float hp = mix(fHash(id + 3.7), fl, 0.75);
  float lit = step(1.0 - uLitFrac * (0.6 + 0.9 * fract(vInfo.z * 5.7)), hp);
  vec3 lc = mix(vec3(1.0, 0.86, 0.66), vec3(0.85, 0.92, 1.0), step(0.5, fHash(vec2(vInfo.z, cell.y)))) * uWinGain * (0.6 + 0.6 * fHash(id + 5.3)) * uNight;
  float glow = 0.0;
  vec2 f = vec2(fract(g.x * 0.5), fv);
  vec3 room = fRoom(id, f, fRt, vInfo.z, lit, lc, fDay * 1.2, 0.0, glow) + lc * glow * 1.2;
  float far = smoothstep(0.12, 0.45, max(fwidth(g.x * 0.5), fw.y));
  room = mix(room, vec3(0.55) * (0.04 * fDay + lc * lit * 0.55), far);
  vec3 tint = vTint;
  fGlass = 1.0;
  fCol = tint * 0.05;
  fRough = 0.035;
  fSpec = mix(vec3(0.07), tint * 0.22, 0.75);
  fEmis = room * tint * 0.9;
  fNxy = (vec2(fHash(id + 0.31 + floor(g.x)), fHash(id + 0.77 + floor(g.x))) - 0.5) * 0.02;
  fNStr = 1.0;
  float envK = 1.0;
  // shadow band under each slab edge: top of the vision glass reads darker (deeper, no interior light)
  float slabSh = smoothstep(0.8, 0.985, fv) * (1.0 - spand);
  fEmis *= 1.0 - 0.55 * slabSh; envK *= 1.0 - 0.3 * slabSh;
  if (variant > 0.5 && variant < 1.5) { float sl = 1.0 - smoothstep(0.02, 0.07, fv); fEmis *= 1.0 - 0.8 * sl; envK *= 1.0 - 0.45 * sl; } // floor slab through full-height glass
  if (spand > 0.5) {
    // opaque spandrel panels: dark back-painted glass / painted metal, much less sky than the vision glass
    if (variant > 1.5 && variant < 2.5) { fGlass = 0.0; fCol = vec3(0.24, 0.25, 0.26) * (0.8 + 0.4 * dot(tint, vec3(0.33))); fMetal = 0.55; fRough = 0.42; fEmis = vec3(0.0); }
    else { fCol = tint * 0.018; fEmis = vec3(0.0); fSpec = mix(vec3(0.04), tint * 0.11, 0.6); envK *= 0.55; }
    float band = smoothstep(sp0 - 0.07, sp0 - 0.005, fv); // shade under the vision-glass sill
    fCol *= 1.0 - 0.35 * band; envK *= 1.0 - 0.25 * band;
  }
  if (frame > 0.01) {
    vec3 mc = variant > 2.5 ? vec3(0.75, 0.76, 0.77) : (fract(vInfo.z * 3.3) > 0.5 ? vec3(0.62, 0.64, 0.66) : vec3(0.16, 0.16, 0.17));
    fCol = mix(fCol, mc, frame);
    fEmis *= 1.0 - frame;
    fMetal = mix(fMetal, 0.85, frame);
    fRough = mix(fRough, 0.32, frame);
    fGlass *= 1.0 - step(0.5, frame);
    fNxy = mix(fNxy, vec2((fu < 0.5 ? -1.0 : 1.0) * mull * 0.7, (fv < 0.5 ? -1.0 : 1.0) * tr * 0.5), frame);
  }
  fAO = 1.0;
  fEnv = mix(0.6, 1.8, smoothstep(2.0, 25.0, g.y)) * envK; // scene.environmentIntensity is low (~0.5): glass reflects the full sky
} else if (fMode == 4) {
  // ---- grate (fire escapes, railings): vInfo.x 0 bars, 1 stair treads, 2 ladder
  float pat = vInfo.x;
  vec2 g = vFuv;
  float bar = 0.0, dens;
  if (pat < 0.5) {
    float bx = abs(fract(g.x / 0.12) - 0.5) * 0.12;
    bar = max(step(bx, 0.011), max(step(0.93, g.y), step(g.y, 0.05)));
    dens = fwidth(g.x / 0.12);
  } else if (pat < 1.5) {
    float tv = abs(fract(g.y / 0.22) - 0.5) * 0.22;
    bar = max(step(tv, 0.035), max(step(g.x, 0.09), step(0.91, g.x)));
    dens = fwidth(g.y / 0.22);
  } else {
    float tv = abs(fract(g.y / 0.3) - 0.5) * 0.3;
    bar = max(step(tv, 0.016), max(step(g.x, 0.11), step(0.89, g.x)));
    dens = fwidth(g.y / 0.3);
  }
  if (dens > 0.35) bar = step(fHash(gl_FragCoord.xy), 0.4);
  if (bar < 0.5) discard;
  fCol = vTint; fRough = 0.6; fMetal = 0.5; fNStr = 0.0;
} else if (fMode == 5) {
  // ---- signs (vInfo.x > 0.5: lit, lettering) and awnings (fabric, vInfo.y: stripes)
  if (vInfo.x > 0.5) {
    vec2 g = vFuv;
    float cellw = 0.3;
    float ci = floor(g.x / cellw);
    vec2 lp = vec2(fract(g.x / cellw), (g.y - 0.22) / 0.56);
    float word = step(0.16, fHash(vec2(floor(ci / 5.0), vInfo.z * 37.0)));           // word gaps
    float glyph = step(0.42, fHash(vec2(ci * 3.0 + floor(lp.x * 3.0), floor(lp.y * 5.0) + vInfo.z * 91.0)));
    float lk = glyph * word * step(0.12, lp.x) * step(lp.x, 0.88) * step(0.0, lp.y) * step(lp.y, 1.0);
    float edge = step(0.3, g.x) * step(g.x, max(0.6, vInfo.y - 0.3));
    lk *= edge;
    vec3 lcol = fHash(vec2(vInfo.z, 4.0)) > 0.5 ? vec3(1.0, 0.96, 0.86) : vec3(1.0, 0.85, 0.35);
    fCol = mix(vTint, lcol, lk * 0.9);
    fEmis = mix(vTint * 0.35, lcol * 1.6, lk) * uNight * 2.2;
    fRough = 0.4; fNStr = 0.0;
  } else {
    float stripe = vInfo.y > 0.5 ? step(0.5, fract(vFuv.x / 0.42)) : 0.0;
    fCol = mix(vTint, vec3(0.92, 0.9, 0.86), stripe) * (fPA.rgb / 0.58);
    fRough = 0.9;
  }
} else if (fMode == 6) {
  // ---- clock face (Ferry Building)
  vec2 q = vFuv - 0.5;
  float rr = length(q);
  if (rr < 0.5) {
    float ang = atan(q.x, q.y);
    float hrs = mod(uClock, 12.0);
    float ha = hrs / 12.0 * 6.2831853, ma = fract(uClock) * 6.2831853;
    vec2 hd = vec2(sin(ha), cos(ha)), mdv = vec2(sin(ma), cos(ma));
    float hand = 0.0;
    float th = dot(q, hd), tm = dot(q, mdv);
    hand = max(hand, step(abs(q.x * hd.y - q.y * hd.x), 0.018) * step(-0.03, th) * step(th, 0.24));
    hand = max(hand, step(abs(q.x * mdv.y - q.y * mdv.x), 0.012) * step(-0.03, tm) * step(tm, 0.38));
    float tick = step(0.38, rr) * step(rr, 0.45) * step(abs(fract(ang / 6.2831853 * 12.0 + 0.5) - 0.5), 0.04);
    float rim = step(0.46, rr);
    float ink = max(max(hand, tick), rim);
    fCol = mix(vec3(0.92, 0.9, 0.84), vec3(0.06, 0.06, 0.07), ink);
    fEmis = vec3(1.0, 0.92, 0.75) * (1.0 - ink) * uNight * 1.6;
    fRough = 0.35; fNStr = 0.0;
  }
} else if (fMode == 8) {
  // ---- LEDs: 0 outline bulbs (Embarcadero Center), 1 animated crown (Salesforce Tower), 2 red beacon
  if (vInfo.x < 0.5) {
    float b = 1.0 - smoothstep(0.08, 0.32, length(vec2(fract(vFuv.x / 0.7) - 0.5, (vFuv.y - 0.5) * 0.5)));
    fCol = vec3(0.7); fEmis = vec3(1.0, 0.93, 0.8) * (0.15 + 1.8 * b) * uNight * 2.0;
  } else if (vInfo.x < 1.5) {
    float w = 0.5 + 0.5 * sin(vFuv.y * 0.18 - uTime * 0.7 + 2.0 * sin(vFuv.x * 0.9 + uTime * 0.23));
    w = pow(w, 3.0);
    fCol = vTint * 0.7; fEmis = mix(vec3(0.5, 0.7, 1.0), vec3(1.0, 0.98, 0.95), w) * (0.25 + 2.2 * w) * uNight * 1.6;
  } else {
    float blink = step(0.5, fract(uTime * 0.5 + vInfo.z));
    fCol = vec3(0.6, 0.05, 0.03); fEmis = vec3(1.0, 0.08, 0.04) * (0.3 + 2.5 * blink) * (0.3 + uNight);
  }
  fNStr = 0.0; fRough = 0.3;
}
if (vWall.z > 0.5) {
  float hs = vWPos.y - vWall.x;                                   // height above the street
  float streak = fHash(vec2(floor(vWuv.x * 2.3), 7.0));
  float gr = 1.0 - smoothstep(0.03, 0.55 + 1.1 * streak, hs);     // splash / dirt band, ragged top
  gr *= 0.8 + 0.4 * fHash(floor(vWuv * vec2(5.0, 9.0)));
  float ao = vWall.y > -1e3 ? smoothstep(vWall.y - 1.6, vWall.y, vWPos.y) * (1.0 - smoothstep(vWall.y, vWall.y + 0.08, vWPos.y)) : 0.0;
  if (fGlass < 0.5) {
    fCol *= (1.0 - 0.34 * gr) * mix(vec3(1.0), vec3(0.94, 0.9, 0.84), gr) * (1.0 - 0.36 * ao);
    fRough = mix(fRough, 0.95, 0.6 * gr);
  } else { fEmis *= 1.0 - 0.45 * ao; fEnv *= 1.0 - 0.3 * ao - 0.25 * gr; }
  fAO *= 1.0 - 0.45 * ao;
}
diffuseColor.rgb *= fCol;
`;

function patchVertex(src, inst) {
  const head = inst ? `
attribute vec4 iP; attribute vec4 iD; attribute vec4 iF; attribute vec4 iInfo; attribute vec3 iTint;
attribute vec2 aOff; attribute float aPart;` : `
attribute vec2 aFuv; attribute vec4 aInfo; attribute vec3 aTint; attribute float aWu; attribute vec2 aGy;`;
  return src
    .replace('#include <common>', `#include <common>${head}
varying vec2 vFuv; varying vec2 vWuv; varying vec4 vInfo; varying vec3 vTint; varying vec3 vWPos; varying vec3 vWall;`)
    .replace('#include <beginnormal_vertex>', inst ? `
vec3 wD = vec3(iD.x, 0.0, iD.y); vec3 wO = vec3(iD.y, 0.0, -iD.x);
vec3 objectNormal = wD * normal.x + vec3(0.0, normal.y, 0.0) + wO * normal.z;` : 'vec3 objectNormal = vec3( normal );')
    .replace('#include <begin_vertex>', inst ? `
int wFl = int(iInfo.y + 0.5);
bool wOn = aPart < 1.5 || (aPart < 2.5 ? (wFl & 2) != 0 : (wFl & 1) != 0);
float wZ = position.z * (aPart < 1.5 ? iP.w : 1.0);
vec3 transformed = wOn ? iP.xyz + wD * (position.x * iD.z + aOff.x) + vec3(0.0, position.y * iD.w + aOff.y, 0.0) + wO * wZ : iP.xyz;
vFuv = iF.xy + position.xy * iF.zw;
vInfo = aPart < 0.5 ? vec4(iInfo.x, -1.0, iInfo.z, ${enc(PBR.PLAIN, MODE.PANE)}.0) : vec4(-1.0, -1.0, iInfo.z, iInfo.w);
vTint = iTint;` : `
vec3 transformed = vec3( position );
vFuv = aFuv; vInfo = vec4(aInfo.xy, aInfo.z * 0.001, aInfo.w); vTint = aTint;`)
    .replace('#include <project_vertex>', `#include <project_vertex>
vWPos = (modelMatrix * vec4(transformed, 1.0)).xyz;
{ vec3 wn = normalize(mat3(modelMatrix) * objectNormal);
  vec2 tw = vec2(-wn.z, wn.x); float tl = length(tw);
  vWuv = (abs(wn.y) > 0.7 || tl < 1e-3) ? vWPos.xz : vec2(dot(vWPos.xz, tw / tl), vWPos.y);
  ${inst ? 'vWall = vec3(0.0);' : `// walls: PBR u runs continuously round the footprint (bricks wrap the corners)
  vWall = vec3((aGy.x - 16384.0) * 0.05, aGy.y > 0.5 ? (aGy.y - 16384.0) * 0.05 : -1e4, step(0.5, aGy.x));
  if (vWall.z > 0.5 && abs(wn.y) < 0.7) vWuv.x = aWu;`} }`);
}

function patchDepthVertex(src, inst) {
  const head = inst ? `
attribute vec4 iP; attribute vec4 iD; attribute vec4 iF; attribute vec4 iInfo;
attribute vec2 aOff; attribute float aPart;` : `
attribute vec2 aFuv; attribute vec4 aInfo;`;
  return src
    .replace('#include <common>', `#include <common>${head}
varying vec2 vFuv; varying vec4 vInfo;`)
    .replace('#include <begin_vertex>', inst ? `
vec3 wD = vec3(iD.x, 0.0, iD.y); vec3 wO = vec3(iD.y, 0.0, -iD.x);
int wFl = int(iInfo.y + 0.5);
bool wOn = aPart < 1.5 || (aPart < 2.5 ? (wFl & 2) != 0 : (wFl & 1) != 0);
float wZ = position.z * (aPart < 1.5 ? iP.w : 1.0);
vec3 transformed = wOn ? iP.xyz + wD * (position.x * iD.z + aOff.x) + vec3(0.0, position.y * iD.w + aOff.y, 0.0) + wO * wZ : iP.xyz;
vFuv = vec2(0.0); vInfo = vec4(0.0);` : `
vec3 transformed = vec3( position );
vFuv = aFuv; vInfo = vec4(aInfo.xy, aInfo.z * 0.001, aInfo.w);`);
}
const DEPTH_DISCARD = /* glsl */`
{
  float fModeF = floor(vInfo.w / 64.0 + 0.001);
  if (fModeF > 3.5 && fModeF < 4.5) {
    float pat = vInfo.x; vec2 g = vFuv; float bar;
    if (pat < 0.5) { float bx = abs(fract(g.x / 0.12) - 0.5) * 0.12; bar = max(step(bx, 0.011), max(step(0.93, g.y), step(g.y, 0.05))); }
    else if (pat < 1.5) { float tv = abs(fract(g.y / 0.22) - 0.5) * 0.22; bar = max(step(tv, 0.035), max(step(g.x, 0.09), step(0.91, g.x))); }
    else { float tv = abs(fract(g.y / 0.3) - 0.5) * 0.3; bar = max(step(tv, 0.016), max(step(g.x, 0.11), step(0.89, g.x))); }
    if (bar < 0.5) discard;
  }
}`;

/**
 * Materials for all building surfaces. tex = window layers {albedo, data}; pbr = loadPbrTextures() result.
 * Returns { facade, windows, depth, winDepth } sharing one uniform set (facade.userData.uniforms).
 */
export function createFacadeMaterials(tex, pbr) {
  const sizes = new Array(16).fill(2);
  (pbr.sizes || []).forEach((s, i) => { if (i < 16) sizes[i] = s || 2; });
  const uniforms = {
    uAlb: { value: tex.albedo }, uDat: { value: tex.data },
    uPA: { value: pbr.albedo }, uPN: { value: pbr.normal },
    uPbrSize: { value: sizes },
    uNight: { value: 0 }, uLitFrac: { value: 0.35 }, uWinGain: { value: 1.3 },
    uTime: { value: 0 }, uClock: { value: 17.85 },
  };
  const make = (inst) => {
    const m = new THREE.MeshPhysicalMaterial({ roughness: 1, metalness: 0, ior: 1.5, specularIntensity: 1, envMapIntensity: 1 });
    m.onBeforeCompile = (sh) => {
      Object.assign(sh.uniforms, uniforms);
      sh.vertexShader = patchVertex(sh.vertexShader, inst);
      sh.fragmentShader = sh.fragmentShader
        .replace('#include <common>', `#include <common>\n${GLSL_COMMON}`)
        .replace('#include <map_fragment>', FRAG_MAIN)
        .replace('#include <roughnessmap_fragment>', 'float roughnessFactor = fRough;' + GBUFFER_CHUNK_TAILS.roughnessmap_fragment)
        .replace('#include <metalnessmap_fragment>', 'float metalnessFactor = fMetal;' + GBUFFER_CHUNK_TAILS.metalnessmap_fragment)
        .replace('#include <normal_fragment_maps>', `{
  vec3 q0 = dFdx(-vViewPosition), q1 = dFdy(-vViewPosition);
  vec2 st0 = dFdx(vWuv), st1 = dFdy(vWuv);
  vec3 q1p = cross(q1, normal), q0p = cross(normal, q0);
  vec3 T = q1p * st0.x + q0p * st1.x, Bt = q1p * st0.y + q0p * st1.y;
  float det = max(dot(T, T), dot(Bt, Bt));
  float sc = det == 0.0 ? 0.0 : inversesqrt(det);
  vec2 nxy = fNxy * fNStr;
  vec3 mapN = vec3(nxy, sqrt(max(0.0, 1.0 - dot(nxy, nxy))));
  normal = normalize(T * (mapN.x * sc) + Bt * (mapN.y * sc) + normal * mapN.z);
}` + GBUFFER_CHUNK_TAILS.normal_fragment_maps)
        .replace('#include <emissivemap_fragment>', '#include <emissivemap_fragment>\ntotalEmissiveRadiance += fEmis;')
        .replace('#include <lights_physical_fragment>', `#include <lights_physical_fragment>
if (fGlass > 0.5) { material.specularColor = fSpec; material.specularF90 = 1.0; }`)
        .replace('#include <aomap_fragment>', `{
  float aoI = mix(1.0, fAO, 0.85);
  reflectedLight.indirectDiffuse *= aoI;
  reflectedLight.indirectSpecular *= aoI * fEnv;
  reflectedLight.directDiffuse *= mix(1.0, fAO, 0.35);
}`);
    };
    m.customProgramCacheKey = () => (inst ? 'roadster-facade-win-3' : 'roadster-facade-3');
    // shells are open boxes of single-sided quads: with the default (back-face) shadow pass the sun leaked
    // through roof / wall gaps as long bright streaks across neighbouring facades and streets
    m.shadowSide = THREE.DoubleSide;
    m.userData.uniforms = uniforms;
    return m;
  };
  const makeDepth = (inst) => {
    const d = new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking });
    d.onBeforeCompile = (sh) => {
      sh.vertexShader = patchDepthVertex(sh.vertexShader, inst);
      sh.fragmentShader = sh.fragmentShader
        .replace('#include <common>', '#include <common>\nvarying vec2 vFuv; varying vec4 vInfo;')
        .replace('#include <clipping_planes_fragment>', `#include <clipping_planes_fragment>\n${inst ? '' : DEPTH_DISCARD}`);
    };
    d.customProgramCacheKey = () => (inst ? 'roadster-facade-depth-win-3' : 'roadster-facade-depth-3');
    return d;
  };
  return { facade: make(false), windows: make(true), depth: makeDepth(false), winDepth: makeDepth(true), uniforms };
}

// ================================================================== geometry buffers
/**
 * Growable merged-geometry buffer with the facade vertex layout (pos f32, normal i8, aFuv f32x2, aInfo i16x4,
 * aTint u8x3, aWu f32, aGy i16x2). While `wall` is set ({u0, sg, k, ax, az, dx, dz, gy(x,z), ao}) every vertex
 * gets the continuous wall u (aWu), the street height under it and the overhang height for the AO band (aGy,
 * 0.05 m units + 16384; 0 = not a wall vertex).
 */
export class GeoBuf {
  constructor(cap = 1024) {
    this.n = 0; this.ni = 0; this.wall = null;
    this._alloc(cap, cap * 2);
  }
  _alloc(vc, ic) {
    const grow = (Old, T, size) => { const a = new T(size); if (Old) a.set(Old.subarray(0, Math.min(Old.length, size))); return a; };
    this.pos = grow(this.pos, Float32Array, vc * 3); this.nor = grow(this.nor, Int8Array, vc * 3);
    this.fuv = grow(this.fuv, Float32Array, vc * 2); this.info = grow(this.info, Int16Array, vc * 4);
    this.tint = grow(this.tint, Uint8Array, vc * 3);
    this.wu = grow(this.wu, Float32Array, vc); this.gy = grow(this.gy, Int16Array, vc * 2);
    this.idx = grow(this.idx, Uint32Array, ic);
    this.vcap = vc; this.icap = ic;
  }
  v(x, y, z, nx, ny, nz, fu, fv, info, tint) {
    if (this.n >= this.vcap) this._alloc(this.vcap * 2, this.icap);
    const i = this.n, p = this.pos, N = this.nor, F = this.fuv, I = this.info, T = this.tint;
    p[i * 3] = x; p[i * 3 + 1] = y; p[i * 3 + 2] = z;
    N[i * 3] = Math.round(nx * 127); N[i * 3 + 1] = Math.round(ny * 127); N[i * 3 + 2] = Math.round(nz * 127);
    F[i * 2] = fu; F[i * 2 + 1] = fv;
    I[i * 4] = info[0]; I[i * 4 + 1] = info[1]; I[i * 4 + 2] = Math.round(info[2] * 1000); I[i * 4 + 3] = info[3];
    T[i * 3] = tint[0]; T[i * 3 + 1] = tint[1]; T[i * 3 + 2] = tint[2];
    const W = this.wall;
    if (W) {
      this.wu[i] = (W.u0 + W.sg * ((x - W.ax) * W.dx + (z - W.az) * W.dz)) * W.k;
      const g = W.gy ? W.gy(x, z) : 0;
      this.gy[i * 2] = clamp(Math.round(g * 20) + 16384, 1, 32767);
      this.gy[i * 2 + 1] = W.ao > -1e3 ? clamp(Math.round(W.ao * 20) + 16384, 1, 32767) : 0;
    } else { this.wu[i] = 0; this.gy[i * 2] = 0; this.gy[i * 2 + 1] = 0; }
    return this.n++;
  }
  /** Triangle, wound so that its face normal agrees with vertex a's normal. */
  tri(a, b, c) {
    if (this.ni + 3 > this.icap) this._alloc(this.vcap, this.icap * 2);
    const P = this.pos, N = this.nor;
    const ax = P[a * 3], ay = P[a * 3 + 1], az = P[a * 3 + 2];
    const ux = P[b * 3] - ax, uy = P[b * 3 + 1] - ay, uz = P[b * 3 + 2] - az;
    const vx = P[c * 3] - ax, vy = P[c * 3 + 1] - ay, vz = P[c * 3 + 2] - az;
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const I = this.idx;
    if (nx * N[a * 3] + ny * N[a * 3 + 1] + nz * N[a * 3 + 2] >= 0) { I[this.ni++] = a; I[this.ni++] = b; I[this.ni++] = c; }
    else { I[this.ni++] = a; I[this.ni++] = c; I[this.ni++] = b; }
  }
  quad(a, b, c, d) { this.tri(a, b, c); this.tri(a, c, d); }
  /** Quad from 4 points [x,y,z] with normal n; fuv per corner (array of 4 [u,v]) or a function (p) => [u,v]. */
  q(P, n, info, tint, uv) {
    const k = P.map((p, i) => { const t = typeof uv === 'function' ? uv(p, i) : uv ? uv[i] : [0, 0]; return this.v(p[0], p[1], p[2], n[0], n[1], n[2], t[0], t[1], info, tint); });
    this.quad(k[0], k[1], k[2], k[3]);
  }
  /** Convex polygon (3D points, planar) with normal n. */
  poly(P, n, info, tint, uv) {
    const k = P.map((p, i) => { const t = typeof uv === 'function' ? uv(p, i) : uv ? uv[i] : [0, 0]; return this.v(p[0], p[1], p[2], n[0], n[1], n[2], t[0], t[1], info, tint); });
    for (let i = 1; i + 1 < k.length; i++) this.tri(k[0], k[i], k[i + 1]);
  }
  lift(start, dy) { if (!dy) return; const P = this.pos; for (let i = start; i < this.n; i++) P[i * 3 + 1] += dy; }
  get tris() { return this.ni / 3; }
  geometry() {
    if (!this.n) return null;
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(this.pos.slice(0, this.n * 3), 3));
    g.setAttribute('normal', new THREE.BufferAttribute(this.nor.slice(0, this.n * 3), 3, true));
    g.setAttribute('aFuv', new THREE.BufferAttribute(this.fuv.slice(0, this.n * 2), 2));
    g.setAttribute('aInfo', new THREE.BufferAttribute(this.info.slice(0, this.n * 4), 4, false));
    g.setAttribute('aTint', new THREE.BufferAttribute(this.tint.slice(0, this.n * 3), 3, true));
    g.setAttribute('aWu', new THREE.BufferAttribute(this.wu.slice(0, this.n), 1));
    g.setAttribute('aGy', new THREE.BufferAttribute(this.gy.slice(0, this.n * 2), 2, false));
    g.setIndex(new THREE.BufferAttribute(this.idx.slice(0, this.ni), 1));
    g.computeBoundingSphere(); g.computeBoundingBox();
    return g;
  }
  build(mats, name = 'buildings') {
    const g = this.geometry();
    if (!g) return null;
    const mesh = new THREE.Mesh(g, mats.facade);
    mesh.customDepthMaterial = mats.depth;
    mesh.castShadow = true; mesh.receiveShadow = true;
    mesh.matrixAutoUpdate = false; mesh.updateMatrix();
    mesh.name = name;
    return mesh;
  }
}

/** Instanced window units (reveals + pane + sill + lintel). */
class WinBuf {
  constructor() { this.n = 0; this.cap = 256; this.P = new Float32Array(1024); this.D = new Float32Array(1024); this.F = new Float32Array(1024); this.I = new Float32Array(1024); this.T = new Uint8Array(768); }
  add(x, y, z, depth, dx, dz, w, h, fu, fv, fw, fh, layer, flags, seed, trimW, tint) {
    if (this.n >= this.cap) {
      const g = (a, k) => { const b = new a.constructor(this.cap * 2 * k); b.set(a); return b; };
      this.P = g(this.P, 4); this.D = g(this.D, 4); this.F = g(this.F, 4); this.I = g(this.I, 4); this.T = g(this.T, 3); this.cap *= 2;
    }
    const i = this.n++;
    this.P.set([x, y, z, depth], i * 4); this.D.set([dx, dz, w, h], i * 4); this.F.set([fu, fv, fw, fh], i * 4);
    this.I.set([layer, flags, seed, trimW], i * 4); this.T.set(tint, i * 3);
  }
  lift(start, dy) { if (!dy) return; for (let i = start; i < this.n; i++) this.P[i * 4 + 1] += dy; }
  build(mats, unit) {
    if (!this.n) return null;
    const g = new THREE.InstancedBufferGeometry();
    for (const k of ['position', 'normal', 'aOff', 'aPart']) g.setAttribute(k, unit.getAttribute(k));
    g.setIndex(unit.getIndex());
    const n = this.n;
    g.setAttribute('iP', new THREE.InstancedBufferAttribute(this.P.slice(0, n * 4), 4));
    g.setAttribute('iD', new THREE.InstancedBufferAttribute(this.D.slice(0, n * 4), 4));
    g.setAttribute('iF', new THREE.InstancedBufferAttribute(this.F.slice(0, n * 4), 4));
    g.setAttribute('iInfo', new THREE.InstancedBufferAttribute(this.I.slice(0, n * 4), 4));
    g.setAttribute('iTint', new THREE.InstancedBufferAttribute(this.T.slice(0, n * 3), 3, true));
    g.instanceCount = n;
    // bounds from the instance origins (+ max window size)
    let minX = Infinity, minY = Infinity, minZ = Infinity, maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (let i = 0; i < n; i++) {
      const x = this.P[i * 4], y = this.P[i * 4 + 1], z = this.P[i * 4 + 2], s = Math.max(this.D[i * 4 + 2], this.D[i * 4 + 3]) + 1;
      minX = Math.min(minX, x - s); maxX = Math.max(maxX, x + s); minY = Math.min(minY, y - 1); maxY = Math.max(maxY, y + s); minZ = Math.min(minZ, z - s); maxZ = Math.max(maxZ, z + s);
    }
    g.boundingBox = new THREE.Box3(new THREE.Vector3(minX, minY, minZ), new THREE.Vector3(maxX, maxY, maxZ));
    g.boundingSphere = g.boundingBox.getBoundingSphere(new THREE.Sphere());
    const m = new THREE.Mesh(g, mats.windows);
    m.customDepthMaterial = mats.winDepth;
    m.castShadow = true; m.receiveShadow = true;
    m.matrixAutoUpdate = false; m.updateMatrix();
    m.name = 'building-windows';
    return m;
  }
}

/** Unit window: x,y in [0,1] scaled by the opening size (+ aOff metres), z in depths (parts 0,1) or metres (2,3). */
let _unitWin = null;
function unitWindow() {
  if (_unitWin) return _unitWin;
  const pos = [], nor = [], off = [], part = [], idx = [];
  const V = (x, y, z, n, ox, oy, pt) => { pos.push(x, y, z); nor.push(n[0], n[1], n[2]); off.push(ox, oy); part.push(pt); return pos.length / 3 - 1; };
  const Q = (a, b, c, d, n) => {
    // orient so (b-a)x(c-a) agrees with n in the local frame (x along wall, y up, z out); positions approximate
    const p = (i) => [pos[i * 3] + off[i * 2] * 0.3, pos[i * 3 + 1] + off[i * 2 + 1] * 0.3, pos[i * 3 + 2]];
    const A = p(a), B = p(b), C = p(c);
    const ux = B[0] - A[0], uy = B[1] - A[1], uz = B[2] - A[2], vx = C[0] - A[0], vy = C[1] - A[1], vz = C[2] - A[2];
    const cx = uy * vz - uz * vy, cy = uz * vx - ux * vz, cz = ux * vy - uy * vx;
    // the instance frame (wall dir, up, outward) is LEFT-handed (outward = up x dir), so wind opposite to the local test
    if (cx * n[0] + cy * n[1] + cz * n[2] >= 0) idx.push(a, c, b, a, d, c); else idx.push(a, b, c, a, c, d);
  };
  // pane (part 0) at z = -1 depth
  { const n = [0, 0, 1]; Q(V(0, 0, -1, n, 0, 0, 0), V(1, 0, -1, n, 0, 0, 0), V(1, 1, -1, n, 0, 0, 0), V(0, 1, -1, n, 0, 0, 0), n); }
  // reveals (part 1)
  { const n = [1, 0, 0]; Q(V(0, 0, 0, n, 0, 0, 1), V(0, 0, -1, n, 0, 0, 1), V(0, 1, -1, n, 0, 0, 1), V(0, 1, 0, n, 0, 0, 1), n); }
  { const n = [-1, 0, 0]; Q(V(1, 0, 0, n, 0, 0, 1), V(1, 0, -1, n, 0, 0, 1), V(1, 1, -1, n, 0, 0, 1), V(1, 1, 0, n, 0, 0, 1), n); }
  { const n = [0, 1, 0]; Q(V(0, 0, 0, n, 0, 0, 1), V(1, 0, 0, n, 0, 0, 1), V(1, 0, -1, n, 0, 0, 1), V(0, 0, -1, n, 0, 0, 1), n); }
  { const n = [0, -1, 0]; Q(V(0, 1, 0, n, 0, 0, 1), V(1, 1, 0, n, 0, 0, 1), V(1, 1, -1, n, 0, 0, 1), V(0, 1, -1, n, 0, 0, 1), n); }
  // sill (part 2): x -0.07..W+0.07, y -0.1..0.02, z -0.06..+0.07
  const box = (y, oy0, oy1, z0, z1, ox, pt, faces) => {
    const b = (x, oxs, oy, z, n) => V(x, y, z, n, oxs, oy, pt);
    let n;
    if (faces.includes('t')) { n = [0, 1, 0]; Q(b(0, -ox, oy1, z1, n), b(1, ox, oy1, z1, n), b(1, ox, oy1, z0, n), b(0, -ox, oy1, z0, n), n); }
    n = [0, 0, 1]; Q(b(0, -ox, oy0, z1, n), b(1, ox, oy0, z1, n), b(1, ox, oy1, z1, n), b(0, -ox, oy1, z1, n), n);
    if (faces.includes('b')) { n = [0, -1, 0]; Q(b(0, -ox, oy0, z1, n), b(1, ox, oy0, z1, n), b(1, ox, oy0, z0, n), b(0, -ox, oy0, z0, n), n); }
  };
  box(0, -0.1, 0.025, -0.05, 0.075, 0.07, 2, 'tb');  // sill: top, front, underside (6 tris)
  box(1, 0, 0.2, 0, 0.055, 0.06, 3, 'b');            // lintel: front, soffit (4 tris)
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3));
  g.setAttribute('aOff', new THREE.Float32BufferAttribute(off, 2));
  g.setAttribute('aPart', new THREE.Float32BufferAttribute(part, 1));
  g.setIndex(idx);
  _unitWin = g;
  return g;
}
export const UNIT_WINDOW_TRIS = 20;

// ================================================================== styles
const sRGB = (r, g, b) => new THREE.Color().setRGB(r, g, b, THREE.SRGBColorSpace);
export function tintBytes(c) { return [clamp(Math.round(c.r * 255), 0, 255), clamp(Math.round(c.g * 255), 0, 255), clamp(Math.round(c.b * 255), 0, 255)]; }
export const tintOf = (r, g, b) => tintBytes(sRGB(r, g, b));
const pick = (arr, r) => arr[Math.min(arr.length - 1, Math.floor(r * arr.length))];
function parseColour(s) {
  if (!s || typeof s !== 'string') return null;
  const t = s.trim().toLowerCase().replace(/\s+/g, '');
  if (/^#([0-9a-f]{3}|[0-9a-f]{6})$/.test(t) || (THREE.Color.NAMES && t in THREE.Color.NAMES)) {
    try { const c = new THREE.Color(); c.setStyle(t, THREE.SRGBColorSpace); return c; } catch { return null; }
  }
  return null;
}
// tints are multiplied onto neutral-grey (tintable) PBR albedo ~0.58, i.e. 1.0 => mid grey-white render
const PAL = {
  brickRed: [[1, 0.96, 0.94], [0.92, 0.86, 0.82], [1, 0.9, 0.86], [0.86, 0.8, 0.76], [1, 1, 1]],
  brickPaint: [[0.97, 0.96, 0.92], [0.92, 0.88, 0.8], [0.78, 0.8, 0.8], [0.66, 0.7, 0.62], [0.93, 0.85, 0.72], [0.74, 0.62, 0.55], [0.58, 0.6, 0.63]],
  stone: [[0.98, 0.93, 0.84], [0.93, 0.88, 0.78], [0.86, 0.82, 0.74], [1, 0.97, 0.9], [0.8, 0.77, 0.72], [0.94, 0.88, 0.82]],
  concrete: [[0.86, 0.86, 0.84], [0.76, 0.76, 0.75], [0.66, 0.66, 0.66], [0.92, 0.91, 0.88], [0.8, 0.78, 0.74]],
  stucco: [[1, 0.96, 0.88], [0.97, 0.9, 0.78], [0.9, 0.93, 0.95], [1, 0.92, 0.84], [0.92, 0.96, 0.9], [1, 1, 0.98], [0.96, 0.85, 0.72], [0.98, 0.86, 0.8]],
  victorian: [[0.62, 0.74, 0.86], [0.96, 0.88, 0.62], [0.72, 0.84, 0.72], [0.96, 0.78, 0.76], [0.98, 0.97, 0.94], [0.78, 0.78, 0.8], [0.6, 0.66, 0.58], [0.86, 0.7, 0.86], [0.95, 0.83, 0.62]],
  white: [[0.98, 0.98, 0.97], [0.93, 0.93, 0.91], [0.86, 0.87, 0.88], [0.95, 0.92, 0.86]],
  roof: [[0.5, 0.5, 0.5], [0.38, 0.38, 0.39], [0.62, 0.61, 0.58], [0.3, 0.3, 0.32], [0.88, 0.88, 0.86], [0.72, 0.72, 0.7]],
  pitched: [[0.86, 0.86, 0.88], [1, 0.62, 0.48], [0.62, 0.62, 0.66], [0.9, 0.78, 0.66]],
  glass: [[0.56, 0.7, 0.84], [0.55, 0.76, 0.72], [0.46, 0.48, 0.52], [0.66, 0.58, 0.48], [0.74, 0.8, 0.86], [0.5, 0.62, 0.74]],
  sign: [[0.62, 0.1, 0.08], [0.1, 0.22, 0.48], [0.82, 0.62, 0.12], [0.08, 0.36, 0.2], [0.9, 0.9, 0.86], [0.06, 0.06, 0.07], [0.42, 0.12, 0.34], [0.86, 0.32, 0.08]],
  awning: [[0.55, 0.1, 0.1], [0.08, 0.26, 0.18], [0.12, 0.18, 0.36], [0.08, 0.08, 0.09], [0.66, 0.5, 0.2], [0.8, 0.78, 0.72], [0.42, 0.08, 0.2]],
};
const tb = (rgb, v = 1) => tintBytes(sRGB(rgb[0] * v, rgb[1] * v, rgb[2] * v));

/**
 * Style rules (BuildingBuilder opts.style; the game passes its city's district map):
 *   oldProb(xb, zb)       probability that a building at bake-frame (xb, zb) is pre-war masonry
 *   victorianProb(xb, zb) probability that a small pre-war residential building is a painted wooden Victorian
 *   namedRe               building names that hint at a historic (pre-war) building
 */
export const DEFAULT_BUILDING_STYLE = {
  oldProb: () => 0.4,
  victorianProb: () => 0.3,
  namedRe: /building|hotel|bank|exchange|hall|church|theat|temple|club|lodge|house|cathedral|mission/i,
};

function pickStyle(b, h, area, rnd, ctx, S = DEFAULT_BUILDING_STYLE) {
  const t = b.t || 'yes', mat = (b.m || '').toLowerCase(), name = String(b.n || '');
  const r1 = rnd(), r2 = rnd(), r3 = rnd();
  const isRes = /^(apartments|residential|house|detached|terrace|dormitory|hostel|semidetached_house)$/.test(t);
  const isOffice = /^(office|commercial|hotel|government|bank|mixed_use|university|public|data_center)$/.test(t);
  const isInd = /^(industrial|warehouse|service|garage|garages|shed|hangar|storage_tank|manufacture|transportation|train_station)$/.test(t);
  const shop = /^(commercial|retail|mixed_use|yes|hotel|apartments|office)$/.test(t);
  const st = {
    cat: 'modern', win: FACADE.PUNCHED, store: false, pbr: PBR.CONCRETE_PANEL, tint: tb([0.85, 0.85, 0.83]),
    trimPbr: PBR.STUCCO_B, trimTint: tb([0.9, 0.88, 0.84]), basePbr: -1, baseTint: null,
    roofPbr: PBR.ROOF_TAR, roofTint: tb(pick(PAL.roof, rnd())),
    bayW: 3.2, floorH: 3.6, prewar: false, curtain: false, glassVariant: 0, glassTint: null,
    sill: true, lintel: false, depth: 0.14, cornice: false, courses: 0, fireEscape: false, balconies: false, bays: false,
    water: false,
  };
  const pOld = ctx ? S.oldProb(ctx.xb, ctx.zb) : 0.4;
  const named = S.namedRe.test(name);
  const prewar = h < 95 && !/glass|mirror/.test(mat) && (r3 < pOld || (named && h < 70 && r3 < 0.85));
  if (b.water) { Object.assign(st, { cat: 'industrial', win: -1, pbr: PBR.METAL, tint: tb([0.8, 0.8, 0.8]), floorH: 6, sill: false }); }
  else if (/glass|mirror/.test(mat) || (!prewar && ((h >= 60 && r1 < 0.6) || (h >= 28 && isOffice && r1 < 0.35) || (h >= 28 && r1 < 0.12)))) {
    Object.assign(st, { cat: 'glass', curtain: true, win: -1, bayW: 1.5, floorH: 3.9, glassVariant: Math.floor(r2 * 4), glassTint: tb(pick(PAL.glass, rnd())) });
  } else if (t === 'parking') {
    Object.assign(st, { cat: 'garage', win: FACADE.GARAGE, pbr: PBR.CONCRETE_ROUGH, tint: tb(pick(PAL.concrete, r2)), floorH: 3.1, sill: false, depth: 0.5 });
  } else if (t === 'roof' || t === 'carport' || t === 'bridge' || t === 'pavilion' || t === 'tent') {
    Object.assign(st, { cat: 'shed', win: -1, pbr: PBR.METAL, tint: tb([0.82, 0.82, 0.8]), sill: false });
  } else if (isInd && !/brick/.test(mat)) {
    const metal = r2 < 0.4;
    Object.assign(st, { cat: 'industrial', win: FACADE.INDUSTRIAL, pbr: metal ? PBR.METAL : r2 < 0.75 ? PBR.CONCRETE_ROUGH : PBR.BRICK_RED, tint: metal ? tb(pick(PAL.concrete, r3)) : tb(pick(PAL.concrete, r3)), floorH: 4.5, bayW: 4 });
  } else if (prewar || /brick|stone|sandstone|granite|marble|limestone/.test(mat)) {
    st.prewar = true; st.cornice = h > 7; st.courses = h > 14 ? 1 : 0; st.lintel = true; st.depth = 0.26; st.floorH = 3.7;
    const victorian = h <= 16.5 && (isRes || t === 'house' || (t === 'yes' && area < 320)) && r1 < (ctx ? S.victorianProb(ctx.xb, ctx.zb) : 0.3);
    if (victorian) {
      Object.assign(st, { cat: 'victorian', win: FACADE.VICTORIAN, pbr: PBR.WOOD_SIDING, tint: tb(pick(PAL.victorian, r2)), trimPbr: PBR.PLAIN, trimTint: tb([0.96, 0.95, 0.92]), bayW: 2.3, floorH: 3.3, depth: 0.16, bays: true, courses: 0 });
    } else if (/brick/.test(mat) || r1 < 0.42) {
      const k = rnd();
      Object.assign(st, {
        cat: 'masonry', pbr: k < 0.6 ? PBR.BRICK_RED : k < 0.78 ? PBR.BRICK_DARK : PBR.BRICK_PAINTED,
        win: isInd ? FACADE.INDUSTRIAL : pick([FACADE.PUNCHED, FACADE.PAIR, FACADE.PUNCHED, FACADE.SMALL, FACADE.CLASSIC], r2),
        trimPbr: PBR.CONCRETE_PANEL, trimTint: tb(pick(PAL.stone, r3)), floorH: 3.5, bayW: isInd ? 4 : 3,
      });
      st.tint = st.pbr === PBR.BRICK_PAINTED ? tb(pick(PAL.brickPaint, rnd())) : tb(pick(PAL.brickRed, rnd()));
      st.fireEscape = h >= 11 && h <= 46 && rnd() < 0.75;
    } else if (r1 < 0.78) {
      Object.assign(st, {
        cat: 'masonry', pbr: rnd() < 0.55 ? PBR.CONCRETE_PANEL : PBR.GRANITE, tint: tb(pick(PAL.stone, r2)),
        win: pick([FACADE.CLASSIC, FACADE.PUNCHED, FACADE.CLASSIC, FACADE.PAIR], rnd()), trimPbr: PBR.CONCRETE_PANEL, trimTint: tb(pick(PAL.stone, r3), 1.05),
        basePbr: PBR.GRANITE, baseTint: tb([0.55, 0.52, 0.5]), courses: h > 20 ? 2 : 1,
      });
    } else {
      Object.assign(st, { cat: 'masonry', pbr: rnd() < 0.5 ? PBR.STUCCO_A : PBR.STUCCO_B, tint: tb(pick(PAL.stucco, r2)), win: pick([FACADE.PAIR, FACADE.SMALL, FACADE.PUNCHED], rnd()), trimTint: tb([0.96, 0.95, 0.92]), floorH: 3.3 });
    }
  } else if (isRes || (h < 60 && r1 < 0.3)) {
    const modern = rnd() < 0.7;
    Object.assign(st, modern
      ? { cat: 'res', win: FACADE.MODERN_RES, pbr: rnd() < 0.5 ? PBR.STUCCO_B : PBR.CONCRETE_PANEL, tint: tb(pick(PAL.white, r2)), floorH: 3.1, bayW: 3.4, balconies: h > 9 && rnd() < 0.55, sill: false, depth: 0.12 }
      : { cat: 'res', win: pick([FACADE.PAIR, FACADE.SMALL], r2), pbr: rnd() < 0.5 ? PBR.STUCCO_A : PBR.BRICK_PAINTED, tint: tb(pick(PAL.stucco, r3)), floorH: 3.0, bayW: 3.2, depth: 0.16 });
  } else {
    const k = rnd();
    Object.assign(st, {
      cat: 'office', pbr: k < 0.5 ? PBR.CONCRETE_PANEL : k < 0.8 ? PBR.CONCRETE_ROUGH : PBR.GRANITE, tint: tb(pick(PAL.concrete, r2)),
      win: pick([FACADE.RIBBON, FACADE.OFFICE_MODERN, FACADE.VERTICAL, FACADE.PUNCHED, FACADE.OFFICE_MODERN], r3), floorH: 3.8, depth: 0.14, sill: false,
    });
    if (st.win === FACADE.VERTICAL) st.depth = 0.32;
  }
  st.store = shop && h > 3.6 && !b.mh && !b.water && !st.curtain && st.cat !== 'garage' && st.cat !== 'shed' && st.cat !== 'industrial'
    && (st.prewar || st.cat === 'office' || st.cat === 'res' ? rnd() < (st.prewar ? 0.85 : 0.55) : false);
  if (st.curtain) st.store = h > 20 && rnd() < 0.4;
  if (b.lv > 0 && h / b.lv > 2.6 && h / b.lv < 6) st.floorH = h / b.lv;
  // OSM colour tags
  const tag = parseColour(b.c);
  if (tag && !st.curtain) { const c = new THREE.Color().setRGB(st.tint[0] / 255, st.tint[1] / 255, st.tint[2] / 255).lerp(tag, 0.6); st.tint = tintBytes(c); }
  const rtag = parseColour(b.rc);
  if (rtag) st.roofTint = tintBytes(rtag);
  else if (st.cat === 'res' || st.curtain) st.roofTint = tb(pick([[0.88, 0.88, 0.86], [0.5, 0.5, 0.5], [0.66, 0.66, 0.64]], rnd()));
  // per-building variation
  const v = 0.93 + rnd() * 0.12;
  st.tint = st.tint.map((c) => clamp(Math.round(c * v), 0, 255));
  return st;
}

// ================================================================== builder
export class BuildingBuilder {
  /**
   * @param {object} region {minX,maxX,minZ,maxZ} of the detailed city
   * @param {object} opts groundAt(x,z), streetAt(x,z) -> bool, offset {x,z} (world -> bake frame),
   *   baseGrid [nx,nz]=[2,2], detailGrid [nx,nz]=[4,4], lodDist=700, maxWindows (instances) = 70000,
   *   style: overrides of DEFAULT_BUILDING_STYLE (era / Victorian district rules, historic-name hints),
   *   nearDetail (true): false = skip the near-LOD outputs (real openings, window units, fire escapes, balconies) - the
   *   base and far-LOD geometry is identical either way; grid: false = no global base / detail buffers (tile builders)
   */
  constructor(region, opts = {}) {
    this.region = region;
    this.style = { ...DEFAULT_BUILDING_STYLE, ...(opts.style || {}) };
    this.groundAt = opts.groundAt || null;
    this.streetAt = opts.streetAt || null;
    this.offset = opts.offset || { x: 0, z: 0 };
    this.foundation = this.groundAt ? 1 : 0;
    [this.bnx, this.bnz] = opts.baseGrid || [2, 2];
    [this.dnx, this.dnz] = opts.detailGrid || [4, 4];
    this.lodDist = opts.lodDist || 700;
    this.maxWindows = opts.maxWindows ?? 70000;
    this.detailFloors = opts.detailFloors ?? 6; // real openings on the lowest N floors (street level); texture windows above
    this.nearDetail = opts.nearDetail !== false;
    this.base = [];
    this.cells = [];
    if (opts.grid !== false) for (let i = 0; i < this.bnx * this.bnz + 4; i++) this.base.push(new GeoBuf(4096));
    if (opts.grid !== false) for (let k = 0; k < this.dnz; k++) for (let i = 0; i < this.dnx; i++) {
      this.cells.push({
        near: new GeoBuf(2048), far: new GeoBuf(1024), wins: new WinBuf(),
        cx: region.minX + ((i + 0.5) * (region.maxX - region.minX)) / this.dnx, cz: region.minZ + ((k + 0.5) * (region.maxZ - region.minZ)) / this.dnz,
      });
    }
    this.colliders = []; // [outer, holes, mh, h, base]
    this.count = 0; this.skipped = 0; this.detailed = 0; this.windowCount = 0;
    this.samples = []; // debug: per-building style summary (dev pages)
    this._v2 = (p) => new THREE.Vector2(p[0], p[1]);
  }
  inRegion(x, z) { const r = this.region; return x >= r.minX && x <= r.maxX && z >= r.minZ && z <= r.maxZ; }
  /** Index of the base chunk (bnx x bnz in the region, then the 4 outer sectors) holding a building centred at (x, z). */
  baseIndex(x, z) {
    const r = this.region;
    if (this.inRegion(x, z)) {
      const i = clamp(Math.floor(((x - r.minX) / (r.maxX - r.minX)) * this.bnx), 0, this.bnx - 1);
      const k = clamp(Math.floor(((z - r.minZ) / (r.maxZ - r.minZ)) * this.bnz), 0, this.bnz - 1);
      return k * this.bnx + i;
    }
    const cx = (r.minX + r.maxX) / 2, cz = (r.minZ + r.maxZ) / 2, dx = x - cx, dz = z - cz;
    return this.bnx * this.bnz + (Math.abs(dx) > Math.abs(dz) ? (dx > 0 ? 0 : 1) : dz > 0 ? 2 : 3);
  }
  /** Index + centre of the detail cell (dnx x dnz over the region) of a point. */
  cellIndex(x, z) {
    const r = this.region;
    const i = clamp(Math.floor(((x - r.minX) / (r.maxX - r.minX)) * this.dnx), 0, this.dnx - 1);
    const k = clamp(Math.floor(((z - r.minZ) / (r.maxZ - r.minZ)) * this.dnz), 0, this.dnz - 1);
    return k * this.dnx + i;
  }
  cellCentre(key) {
    const r = this.region, i = key % this.dnx, k = Math.floor(key / this.dnx);
    return [r.minX + ((i + 0.5) * (r.maxX - r.minX)) / this.dnx, r.minZ + ((k + 0.5) * (r.maxZ - r.minZ)) / this.dnz];
  }
  baseFor(x, z) {
    const r = this.region;
    if (this.inRegion(x, z)) {
      const i = clamp(Math.floor(((x - r.minX) / (r.maxX - r.minX)) * this.bnx), 0, this.bnx - 1);
      const k = clamp(Math.floor(((z - r.minZ) / (r.maxZ - r.minZ)) * this.bnz), 0, this.bnz - 1);
      return this.base[k * this.bnx + i];
    }
    const cx = (r.minX + r.maxX) / 2, cz = (r.minZ + r.maxZ) / 2, dx = x - cx, dz = z - cz;
    const s = Math.abs(dx) > Math.abs(dz) ? (dx > 0 ? 0 : 1) : dz > 0 ? 2 : 3;
    return this.base[this.bnx * this.bnz + s];
  }
  cellFor(x, z) {
    const r = this.region;
    const i = clamp(Math.floor(((x - r.minX) / (r.maxX - r.minX)) * this.dnx), 0, this.dnx - 1);
    const k = clamp(Math.floor(((z - r.minZ) / (r.maxZ - r.minZ)) * this.dnz), 0, this.dnz - 1);
    return this.cells[k * this.dnx + i];
  }
  isStreet(x, z) { return this.streetAt ? !!this.streetAt(x, z) : true; }

  /**
   * Footprint of a building as add() would build it, without any geometry: { outer (CCW), holes (CW), h, mh, area,
   * c (centroid), base (ground height of the lowest outline point) }, or null when add() would skip it (claimed by a
   * landmark, broken ring, < 3 m2). The colliders are [outer, holes, mh, h, base] (MapWorld registers them globally
   * when buildings are streamed per tile).
   */
  footprint(b) {
    if (b._lm) return null;
    const outer0 = sanitizeRing(b.p);
    if (!outer0) return null;
    let h = Number(b.h); if (!isFinite(h) || h <= 0) h = 10;
    h = clamp(h, 2.5, 600);
    let mh = Number(b.mh) || 0; mh = clamp(mh, 0, h - 1);
    const outer = signedArea(outer0) > 0 ? outer0 : outer0.slice().reverse();
    const holes = (b.hl || []).map((hr) => sanitizeRing(hr)).filter((hr) => hr && Math.abs(signedArea(hr)) > 2)
      .map((hr) => (signedArea(hr) < 0 ? hr : hr.slice().reverse()));
    const area = Math.abs(signedArea(outer)) - holes.reduce((a, hr) => a + Math.abs(signedArea(hr)), 0);
    if (area < 3) return null;
    const c = centroid(outer);
    let base = 0;
    if (this.groundAt) {
      base = Infinity;
      for (let i = 0; i < outer.length; i++) {
        const a = outer[i], e = outer[(i + 1) % outer.length];
        const n = Math.max(1, Math.ceil(Math.hypot(e[0] - a[0], e[1] - a[1]) / 2.5));
        for (let s = 0; s < n; s++) base = Math.min(base, this.groundAt(a[0] + ((e[0] - a[0]) * s) / n, a[1] + ((e[1] - a[1]) * s) / n));
      }
      if (!Number.isFinite(base)) base = 0;
    }
    return { outer, holes, h, mh, area, c, base };
  }

  /** b: {p, hl?, h, mh?, lv?, t?, r?, rh?, c?, m?, rc?, n?, lod?}; rnd: seeded RNG; fp: footprint(b) if already known */
  add(b, rnd, fp = null) {
    this.lastDetailed = false;
    if (b._lm) return false; // claimed by a hand-modelled landmark (a MapWorld plugin's beforeBuildings hook)
    const F = fp || this.footprint(b);
    if (!F) { this.skipped++; return false; }
    const { outer, holes, h, mh, area, c, base } = F;
    const st = pickStyle(b, h, area, rnd, { xb: c[0] + this.offset.x, zb: c[1] + this.offset.z }, this.style);
    const seed = rnd();
    const B = this.baseFor(c[0], c[1]);
    const floors = Math.max(1, Math.round(h / st.floorH));
    const fh = h / floors;
    const u0 = Math.floor(seed * 4);

    // ---- roof shape
    let rs = (b.r || '').toLowerCase();
    let roofH = 0;
    const convex = isConvex(outer);
    const minW = minWidth(outer);
    if (rs && rs !== 'flat') {
      if (/^(pyramidal|cone|dome|onion|round)$/.test(rs)) {
        const tall = h - mh > 2.2 * Math.sqrt(area);
        roofH = Number(b.rh) > 0 ? Number(b.rh) : tall ? (h - mh) * 0.85 : clamp(0.45 * Math.sqrt(area), 1.5, (h - mh) * 0.6);
        if (!starShaped(outer, c) || holes.length) { rs = ''; roofH = 0; }
      } else if (/^(gabled|hipped|half-hipped|gambrel|mansard|saltbox|quadruple_saltbox|side_hipped)$/.test(rs) && convex && !holes.length) {
        roofH = Number(b.rh) > 0 ? Number(b.rh) : clamp(0.32 * minW, 1.2, 7);
      } else if (rs === 'skillion' && !holes.length) {
        roofH = Number(b.rh) > 0 ? Number(b.rh) : clamp(0.15 * minW, 0.8, 3);
      } else { rs = ''; }
      roofH = Math.min(roofH, (h - mh) * 0.9);
    } else rs = '';
    const wallTop = h - roofH;
    let ring = outer, topH = null;
    if ((rs && /^(gabled|half-hipped|gambrel|saltbox|side_hipped)$/.test(rs)) || (rs && /^(hipped|mansard|quadruple_saltbox)$/.test(rs) && outer.length !== 4)) {
      const ax = longAxis(outer), nx = -ax[1], nz = ax[0];
      let dmax = 0;
      for (const p of outer) dmax = Math.max(dmax, Math.abs((p[0] - c[0]) * nx + (p[1] - c[1]) * nz));
      if (dmax < 0.5) rs = ''; else {
        ring = insertRidgePoints(outer, c, nx, nz);
        topH = (x, z) => wallTop + roofH * (1 - Math.abs((x - c[0]) * nx + (z - c[1]) * nz) / dmax);
        rs = 'gabled';
      }
    } else if (rs === 'skillion') {
      const ax = longAxis(outer), nx = -ax[1], nz = ax[0];
      let d0 = Infinity, d1 = -Infinity;
      for (const p of outer) { const d = (p[0] - c[0]) * nx + (p[1] - c[1]) * nz; d0 = Math.min(d0, d); d1 = Math.max(d1, d); }
      if (d1 - d0 < 0.5) rs = ''; else topH = (x, z) => wallTop + roofH * (((x - c[0]) * nx + (z - c[1]) * nz - d0) / (d1 - d0));
    }
    if (!rs) roofH = 0;
    const flatTop = rs ? wallTop : h;
    const parapet = !rs && h - mh > 7 && area > 60 && b.t !== 'roof' && st.cat !== 'shed' ? clamp(0.6 + h * 0.004, 0.7, 1.6) * (b.lod ? 0.8 : 1) : 0;

    // ---- detail decision (real openings, instanced windows)
    const cell = !b.lod && this.inRegion(c[0], c[1]) ? this.cellFor(c[0], c[1]) : null;
    let detailed = false;
    const ctx0 = { est: 0 };
    if (cell && !st.curtain && st.win >= 0 && !rs) {
      let est = 0;
      const rects = WIN_RECTS[st.win].length;
      for (let i = 0; i < outer.length; i++) {
        const a = outer[i], e = outer[(i + 1) % outer.length], L = Math.hypot(e[0] - a[0], e[1] - a[1]);
        const fr = L >= 2.5 ? frame(a, e, true) : null;
        if (fr && this.isStreet(fr.mx + fr.ox * 4.5, fr.mz + fr.oz * 4.5)) est += Math.max(1, Math.round(L / st.bayW)) * rects * Math.max(1, Math.min(this.detailFloors, Math.round(flatTop / fh)) - Math.ceil(mh / fh));
      }
      detailed = est <= 2600 && this.windowCount + est <= this.maxWindows;
      if (detailed) { this.windowCount += est; ctx0.est = est; } // reserved; corrected below with the real count
    }
    const starts = { B: B.n, near: cell ? cell.near.n : 0, far: cell ? cell.far.n : 0, wins: cell ? cell.wins.n : 0 };
    const ctx = { b, st, h, mh, fh, floors, u0, seed, base, rnd, flatTop, wallTop, parapet, B, cell, detailed, outer, holes, area, c };
    const segInfo = [];

    // ---- walls
    const yBot = mh > 0 ? mh : -this.foundation;
    // overhang heights (absolute) for the facade AO band: under the cornice, under the shop signs / awnings
    const corniceP = clamp(0.35 + h * 0.008, 0.4, 1.1);
    const aoTop = !b.lod && !st.curtain && st.cornice && !rs ? base + flatTop - 0.25 - corniceP * 0.9 : -1e4;
    const aoStore = base + WIN_RECTS[FACADE.STOREFRONT][0][3] * fh + 0.14;
    const doRing = (r, isOuter) => {
      // continuous PBR u round the ring, scaled so the texture also closes at the last corner
      let per = 0; for (let i = 0; i < r.length; i++) per += Math.hypot(r[(i + 1) % r.length][0] - r[i][0], r[(i + 1) % r.length][1] - r[i][1]);
      const tile = (PBR_SETS[st.curtain ? PBR.PLAIN : st.pbr] || {}).size || 2;
      const kU = per > tile ? (Math.max(1, Math.round(per / tile)) * tile) / per : 1;
      let cum = 0;
      for (let i = 0; i < r.length; i++) {
        const a = r[i], e = r[(i + 1) % r.length];
        const f = frame(a, e, isOuter);
        if (!f) continue;
        f.wall = { u0: isOuter ? cum : -cum, sg: isOuter ? 1 : -1, k: kU, ax: f.ax, az: f.az, dx: f.dx, dz: f.dz, gy: this.groundAt, ao: aoTop };
        cum += f.L;
        const ya = topH && isOuter ? topH(a[0], a[1]) : flatTop, yb = topH && isOuter ? topH(e[0], e[1]) : flatTop;
        const street = isOuter && !b.lod && f.L > 3 && this.isStreet(f.mx + f.ox * 4.5, f.mz + f.oz * 4.5);
        const seg = { f, street, ya, yb, isOuter };
        if (isOuter) segInfo.push(seg);
        const storeLayer = st.store && street && mh < 0.5 ? FACADE.STOREFRONT : -1;
        f.wallLo = storeLayer >= 0 ? { ...f.wall, ao: aoStore } : f.wall;
        const nb = f.L >= 0.6 * st.bayW ? Math.max(1, Math.round(f.L / st.bayW)) : f.L / st.bayW;
        const bw = f.L / nb;
        // a plain plinth where the street climbs along the wall (no windows / reveals below street level)
        const gl = this.groundAt && isOuter && mh < 0.5 && !b.lod ? (s) => this.groundAt(f.ax + f.dx * s, f.az + f.dz * s) - base : null;
        const o = { gl, split: storeLayer >= 0 ? fh : null, pInfo: [-1, -1, seed, enc(st.pbr, MODE.WALLONLY)], pTint: st.tint };
        if (st.curtain) {
          o.pInfo = [-1, -1, seed, enc(PBR.GRANITE, MODE.PLAIN)]; o.pTint = tb([0.5, 0.49, 0.48]);
          this.wallPieces(B, f, yBot, ya + (isOuter ? parapet : 0), yb + (isOuter ? parapet : 0), u0, bw, fh, [st.glassVariant, -1, seed, enc(PBR.PLAIN, MODE.CURTAIN)], st.glassTint, o);
          continue;
        }
        const info = [st.win, storeLayer, seed, enc(st.pbr, st.win >= 0 ? MODE.WALL : MODE.PLAIN)];
        if (detailed && f.L >= 2.5 && street) {
          this.shellWall(cell.far, f, yBot, ya, yb, u0, bw, fh, info, st.tint); // far LOD (> lodDist): one quad
          if (this.nearDetail) this.detailWall(ctx, f, yBot, flatTop, nb, bw, storeLayer);
          if (isOuter && parapet > 0) this.shellWall(B, f, flatTop, flatTop + parapet, flatTop + parapet, u0, bw, fh, [-1, -1, seed, enc(st.pbr, MODE.WALLONLY)], st.tint);
        } else {
          this.wallPieces(B, f, yBot, ya + (isOuter ? parapet : 0), yb + (isOuter ? parapet : 0), u0, bw, fh, info, st.tint, o);
        }
      }
    };
    doRing(ring, true);
    for (const hr of holes) doRing(hr, false);

    // ---- parapet inner faces + coping, cornice, string courses, fascia
    const trimInfo = [-1, -1, seed, enc(st.trimPbr, MODE.PLAIN)];
    const roofInfo = [-1, -1, seed, enc(st.roofPbr, MODE.PLAIN)];
    if (parapet > 0) this.parapet(B, outer, flatTop, parapet, [-1, -1, seed, enc(st.pbr, MODE.WALLONLY)], st.tint, trimInfo, st.trimTint, !b.lod);
    else if (!rs && !b.lod && st.cat !== 'shed' && h > 4) this.ledge(B, outer, flatTop - 0.35, 0.35, 0.18, trimInfo, st.trimTint);
    if (!b.lod && !st.curtain) {
      if (st.cornice && !rs) {
        const P = clamp(0.35 + h * 0.008, 0.4, 1.1);
        this.cornice(B, outer, flatTop - 0.25 - P * 0.9, P, P * 0.9, trimInfo, st.trimTint);
      }
      if (st.courses > 0 && floors >= 3) {
        const every = st.courses >= 2 ? Math.max(3, Math.round(floors / 4)) : floors > 6 ? Math.round(floors / 2) : 99;
        for (let f = 1; f < floors - 1; f++) {
          if (f !== 1 && f % every !== 0) continue;
          const y = f * fh - 0.08;
          if (y < mh + 1) continue;
          this.ledge(B, outer, y, 0.26, 0.09, trimInfo, st.trimTint);
        }
      }
      if (st.curtain && !rs && h > 40) this.ledge(B, outer, flatTop - 1.2, 1.2, 0.04, [-1, -1, seed, enc(PBR.PLAIN, MODE.PLAIN)], tb([0.6, 0.62, 0.64]));
    }

    // ---- roof
    if (!rs) {
      this.flatPolygon(B, outer, holes, flatTop, roofInfo, st.roofTint, true);
    } else if (rs === 'gabled' || rs === 'skillion') {
      // eaves: the roof overhangs the walls (0.35 m) and gets a fascia + soffit, so its edge has thickness
      const ov = b.lod ? 0 : 0.35, ringO = ov ? offsetRing(ring, ov) : ring;
      const pieces = rs === 'gabled' ? splitByRidge(ringO, c, longAxis(outer)) : [ringO];
      for (const pc of pieces) this.heightPolygon(B, pc, topH, [-1, -1, seed, enc(PBR.ROOF_SLATE, MODE.PLAIN)], tb(pick(PAL.pitched, rnd())));
      if (ov) this.eaves(B, ring, ringO, (p) => topH(p[0], p[1]), seed, st);
    } else if (/^(hipped|mansard|quadruple_saltbox)$/.test(rs) && outer.length === 4) {
      this.hipRoof(B, outer, wallTop, roofH, [-1, -1, seed, enc(PBR.ROOF_SLATE, MODE.PLAIN)], tb(pick(PAL.pitched, rnd())), b.lod ? 0 : 0.35, seed, st);
    } else {
      const prof = rs === 'dome' || rs === 'round' || rs === 'onion' ? [[1, 0], [0.92, 0.38], [0.71, 0.71], [0.38, 0.92], [0, 1]] : [[1, 0], [0, 1]];
      const tall = rs === 'pyramidal' && h > 60;
      this.taperRoof(B, outer, c, wallTop, roofH, prof, tall ? [-1, -1, seed, enc(st.pbr, MODE.PLAIN)] : [-1, -1, seed, enc(PBR.ROOF_SLATE, MODE.PLAIN)], tall ? st.tint : tb(pick(PAL.pitched, rnd())));
    }
    if (mh > 2) this.flatPolygon(B, outer, holes, mh, roofInfo, st.roofTint, false);

    // ---- extras
    if (!b.lod) {
      if (!rs && h > 6 && area > 60 && b.t !== 'roof' && st.cat !== 'shed') this.rooftop(ctx, B, outer, holes, flatTop, area, h, rnd, seed, st);
      for (const sg of segInfo) {
        // street height above the base along this wall (signs / awnings / bays never end up under the sidewalk)
        const gl = this.groundAt ? (s) => this.groundAt(sg.f.ax + sg.f.dx * s, sg.f.az + sg.f.dz * s) - base : () => 0;
        if (sg.street && st.store && mh < 0.5 && floors >= 1 && sg.f.L >= 4) this.storefront(B, sg.f, st, fh, rnd, seed, u0, gl);
        if (st.bays && sg.street && sg.f.L >= 4.5 && sg.f.L <= 16 && floors >= 2 && rnd() < 0.85) this.bayWindow(B, sg.f, st, fh, floors, flatTop, seed, gl);
      }
      if (this.nearDetail && detailed && st.fireEscape && floors >= 3) {
        const cands = segInfo.filter((sg) => sg.street && sg.f.L >= 3 * st.bayW);
        if (cands.length) this.fireEscape(cell.near, pick(cands, rnd()).f, st, fh, floors, flatTop, rnd, base);
      }
      if (this.nearDetail && detailed && st.balconies && floors >= 3) for (const sg of segInfo) if (sg.street && sg.f.L >= 6) this.balconies(cell.near, sg.f, st, fh, floors, flatTop, rnd, seed);
    }

    // ---- lift everything this building emitted onto its ground height
    if (base) {
      B.lift(starts.B, base);
      if (cell) { cell.near.lift(starts.near, base); cell.far.lift(starts.far, base); cell.wins.lift(starts.wins, base); }
    }
    if (detailed) { this.detailed++; this.windowCount += cell.wins.n - starts.wins - ctx0.est; }
    this.colliders.push([outer, holes, mh, h, base]);
    if (!b.lod && this.samples.length < 6000) this.samples.push({ x: +c[0].toFixed(1), z: +c[1].toFixed(1), y: +base.toFixed(1), h, cat: st.cat, pbr: st.pbr, win: st.win, detailed, fire: !!(detailed && st.fireEscape), store: st.store, n: b.n || '' });
    this.count++;
    this.lastDetailed = detailed;
    return true;
  }

  // ---------------------------------------------------------------- walls
  /** Flat wall quad (per-vertex top heights for gables). info[0] window layer, [1] storefront layer. */
  shellWall(buf, f, y0, ya, yb, u0, bw, fh, info, tint, wall = f.wall) {
    const nb = f.L / bw;
    const p = (s, y) => [f.ax + f.dx * s, y, f.az + f.dz * s];
    const n = [f.ox, 0, f.oz];
    buf.wall = wall || null;
    const k0 = buf.v(...p(0, y0), ...n, u0 + f.uDir0 * nb, y0 / fh, info, tint);
    const k1 = buf.v(...p(f.L, y0), ...n, u0 + f.uDir1 * nb, y0 / fh, info, tint);
    const k2 = buf.v(...p(f.L, yb), ...n, u0 + f.uDir1 * nb, yb / fh, info, tint);
    const k3 = buf.v(...p(0, ya), ...n, u0 + f.uDir0 * nb, ya / fh, info, tint);
    buf.wall = null;
    buf.quad(k0, k1, k2, k3);
  }
  /**
   * Shell wall in pieces: where the street rises along the wall (o.gl(s) = street height above the building
   * base) the part below street level + 0.4 m becomes a plain plinth in ~4 m slices, so texture windows never
   * show under the sidewalk; o.split (shop-front line) splits the wall for the AO band under the shop signs.
   */
  wallPieces(buf, f, y0, ya, yb, u0, bw, fh, info, tint, o) {
    const L = f.L;
    let slices = null;
    if (o.gl && y0 < 0) {
      const n = Math.max(1, Math.ceil(L / 5));
      let any = false;
      slices = [];
      for (let k = 0; k < n; k++) {
        const sa = (L * k) / n, sb = (L * (k + 1)) / n;
        const g = Math.max(o.gl(sa), o.gl(sb), o.gl((sa + sb) / 2));
        slices.push([sa, sb, g > 0.12 ? g + 0.4 : 0]);
        if (g > 0.12) any = true;
      }
      if (!any) slices = null;
    }
    if (!slices && o.split == null) { this.shellWall(buf, f, y0, ya, yb, u0, bw, fh, info, tint); return; }
    if (!slices) slices = [[0, L, 0]];
    const n = [f.ox, 0, f.oz];
    const U = (s) => u0 + (f.uDir0 * (L - s) + f.uDir1 * s) / bw;
    const rect = (sa, sb, yA, tA, tB, inf, tn, wall) => {
      if (Math.min(tA, tB) - yA < 0.02) return;
      buf.wall = wall || null;
      const P = (s, y) => [f.ax + f.dx * s, y, f.az + f.dz * s];
      const k0 = buf.v(...P(sa, yA), ...n, U(sa), yA / fh, inf, tn), k1 = buf.v(...P(sb, yA), ...n, U(sb), yA / fh, inf, tn);
      const k2 = buf.v(...P(sb, tB), ...n, U(sb), tB / fh, inf, tn), k3 = buf.v(...P(sa, tA), ...n, U(sa), tA / fh, inf, tn);
      buf.wall = null;
      buf.quad(k0, k1, k2, k3);
    };
    for (const [sa, sb, pt] of slices) {
      const tA = ya + ((yb - ya) * sa) / L, tB = ya + ((yb - ya) * sb) / L;
      let yS = y0;
      if (pt > y0 + 0.05) { const top = Math.min(pt, tA, tB); rect(sa, sb, y0, top, top, o.pInfo, o.pTint, f.wallLo || f.wall); yS = top; }
      const sp = o.split;
      if (sp != null && sp > yS + 0.05 && sp < Math.min(tA, tB) - 0.05) {
        rect(sa, sb, yS, sp, sp, info, tint, f.wallLo || f.wall);
        rect(sa, sb, sp, tA, tB, info, tint, f.wall);
      } else rect(sa, sb, yS, tA, tB, info, tint, sp != null && yS < sp ? f.wallLo || f.wall : f.wall);
    }
  }

  /** Wall with real openings: solid parts as merged quads, openings as instanced window units. */
  detailWall(ctx, f, yBot, yTop, nb, bw, storeLayer) {
    const { st, fh, floors, mh, u0, seed, base, cell } = ctx;
    const buf = cell.near, wins = cell.wins;
    const L = f.L;
    const layer = st.win;
    const rects = WIN_RECTS[layer];
    const ribbon = rects[0][0] <= 0.001 && rects[0][1] >= 0.999;
    const slot = rects[0][2] <= 0.001 && rects[0][3] >= 0.999;
    const gl = this.groundAt ? (s) => this.groundAt(f.ax + f.dx * s, f.az + f.dz * s) - base : () => 0;
    const openings = [];
    const minRow = Math.max(0, Math.ceil((mh - 0.01) / fh));
    const maxRow = Math.min(floors, Math.floor((yTop + 0.01) / fh), this.detailFloors) - 1;
    const yCut = Math.min(yTop, (maxRow + 1) * fh);
    const trimW = enc(st.trimPbr, MODE.WALLONLY);
    const flagsFor = (store) => (store ? 0 : (st.sill ? 2 : 0) | (st.lintel ? 1 : 0));
    const add = (s0, s1, y0, y1, lay, depth, flags) => {
      if (s1 - s0 < 0.3 || y1 - y0 < 0.3) return;
      const g = Math.max(gl(s0), gl(s1), gl((s0 + s1) / 2));
      if (g + 0.25 > y0) return; // below the (sloping) street / sidewalk: solid wall
      openings.push([s0, s1, y0, y1]);
      // u runs a->b for outer rings (uDir1 = 1) and b->a for holes; window fuv origin is the opening's low-u corner
      const fuA = u0 + (f.uDir0 * (L - s0) + f.uDir1 * s0) / bw, fuB = u0 + (f.uDir0 * (L - s1) + f.uDir1 * s1) / bw;
      const sx = f.uDir1 ? s0 : s1, fu = Math.min(fuA, fuB);
      const dx = f.uDir1 ? f.dx : -f.dx, dz = f.uDir1 ? f.dz : -f.dz;
      wins.add(f.ax + f.dx * sx, y0, f.az + f.dz * sx, depth, dx, dz, s1 - s0, y1 - y0, fu, y0 / fh, (s1 - s0) / bw, (y1 - y0) / fh, lay, flags, seed, trimW, st.trimTint);
    };
    for (let row = minRow; row <= maxRow; row++) {
      const fy = row * fh;
      const store = row === 0 && storeLayer >= 0;
      if (store) {
        const R = WIN_RECTS[FACADE.STOREFRONT][0];
        for (let i = 0; i < Math.round(nb); i++) add(i * bw + R[0] * bw + 0.12, i * bw + R[1] * bw - 0.12, fy + R[2] * fh, fy + R[3] * fh, FACADE.STOREFRONT, 0.32, 0);
        continue;
      }
      if (slot) {
        if (row !== Math.max(minRow, storeLayer >= 0 ? 1 : 0)) continue;
        const y0 = fy + 0.4, y1 = (maxRow + 1) * fh - 0.15;
        for (let i = 0; i < Math.round(nb); i++) add(i * bw + rects[0][0] * bw, i * bw + rects[0][1] * bw, y0, y1, layer, st.depth, 2);
        continue;
      }
      for (const R of rects) {
        if (ribbon) { add(0.25, L - 0.25, fy + R[2] * fh, fy + R[3] * fh, layer, st.depth, flagsFor(false) & 2); continue; }
        for (let i = 0; i < Math.round(nb); i++) add(i * bw + R[0] * bw, i * bw + R[1] * bw, fy + R[2] * fh, fy + R[3] * fh, layer, st.depth, flagsFor(false));
      }
    }
    // solid wall pieces around the openings
    const info = [-1, -1, seed, enc(st.pbr, MODE.WALLONLY)];
    const baseInfo = st.store && storeLayer >= 0 && st.basePbr >= 0 ? [-1, -1, seed, enc(st.basePbr, MODE.WALLONLY)] : null;
    const n = [f.ox, 0, f.oz];
    const P = (s, y) => [f.ax + f.dx * s, y, f.az + f.dz * s];
    if (yTop > yCut + 0.05) this.shellWall(buf, f, yCut, yTop, yTop, u0, bw, fh, [layer, -1, seed, enc(st.pbr, MODE.WALL)], st.tint);
    const lo = f.wallLo || f.wall;
    solidRects(L, yBot, yCut, openings, (s0, s1, y0, y1) => {
      buf.wall = y0 < fh - 0.01 ? lo : f.wall;
      if (baseInfo && y0 < fh - 0.01) {
        const ym = Math.min(y1, fh);
        buf.q([P(s0, y0), P(s1, y0), P(s1, ym), P(s0, ym)], n, baseInfo, st.baseTint);
        buf.wall = f.wall;
        if (y1 > ym + 0.01) buf.q([P(s0, ym), P(s1, ym), P(s1, y1), P(s0, y1)], n, info, st.tint);
      } else buf.q([P(s0, y0), P(s1, y0), P(s1, y1), P(s0, y1)], n, info, st.tint);
      buf.wall = null;
    });
  }

  // ---------------------------------------------------------------- trims
  parapet(buf, outer, y0, ph, wallInfo, tint, capInfo, capTint, coping) {
    const inset = insetRing(outer, 0.35);
    if (!inset) return;
    const out = coping ? offsetRing(outer, 0.06) : outer;
    const n = outer.length, y1 = y0 + ph;
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      const a = inset[i], b = inset[j], oa = out[i], ob = out[j];
      const dx = b[0] - a[0], dz = b[1] - a[1], L = Math.hypot(dx, dz);
      if (L < 0.05) continue;
      const ix = -dz / L, iz = dx / L;
      buf.q([[a[0], y0, a[1]], [b[0], y0, b[1]], [b[0], y1, b[1]], [a[0], y1, a[1]]], [ix, 0, iz], wallInfo, tint);
      const yc = y1 + (coping ? 0.08 : 0);
      buf.q([[oa[0], yc, oa[1]], [ob[0], yc, ob[1]], [b[0], yc, b[1]], [a[0], yc, a[1]]], [0, 1, 0], capInfo, capTint);
      if (coping) {
        const ex = ob[0] - oa[0], ez = ob[1] - oa[1], El = Math.hypot(ex, ez) || 1;
        const nx = ez / El, nz = -ex / El;
        buf.q([[oa[0], y1 - 0.06, oa[1]], [ob[0], y1 - 0.06, ob[1]], [ob[0], yc, ob[1]], [oa[0], yc, oa[1]]], [nx, 0, nz], capInfo, capTint);
      }
    }
  }
  /** Thin projecting band (string course / fascia): front + top + bottom. */
  ledge(buf, outer, y, hgt, proj, info, tint) {
    const off = offsetRing(outer, proj);
    const n = outer.length;
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n, a = outer[i], b = outer[j], oa = off[i], ob = off[j];
      const ex = ob[0] - oa[0], ez = ob[1] - oa[1], El = Math.hypot(ex, ez);
      if (El < 0.05) continue;
      const nx = ez / El, nz = -ex / El;
      buf.q([[oa[0], y, oa[1]], [ob[0], y, ob[1]], [ob[0], y + hgt, ob[1]], [oa[0], y + hgt, oa[1]]], [nx, 0, nz], info, tint);
      buf.q([[a[0], y + hgt, a[1]], [b[0], y + hgt, b[1]], [ob[0], y + hgt, ob[1]], [oa[0], y + hgt, oa[1]]], [0, 1, 0], info, tint);
      buf.q([[a[0], y, a[1]], [b[0], y, b[1]], [ob[0], y, ob[1]], [oa[0], y, oa[1]]], [0, -1, 0], info, tint);
    }
  }
  /** Classical cornice: soffit, bed moulding (sloped), fascia, top. */
  cornice(buf, outer, y, P, H, info, tint) {
    const o1 = offsetRing(outer, P * 0.35), o2 = offsetRing(outer, P);
    const n = outer.length, yb = y + H * 0.35, yt = y + H;
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n, a = outer[i], b = outer[j];
      const ex = o2[j][0] - o2[i][0], ez = o2[j][1] - o2[i][1], El = Math.hypot(ex, ez);
      if (El < 0.05) continue;
      const nx = ez / El, nz = -ex / El;
      // bed moulding: from the wall at y up/out to o1 at yb (sloped, facing out-down)
      const sn = [nx * 0.7, -0.7, nz * 0.7];
      buf.q([[a[0], y, a[1]], [b[0], y, b[1]], [o1[j][0], yb, o1[j][1]], [o1[i][0], yb, o1[i][1]]], sn, info, tint);
      // soffit o1 -> o2 at yb
      buf.q([[o1[i][0], yb, o1[i][1]], [o1[j][0], yb, o1[j][1]], [o2[j][0], yb, o2[j][1]], [o2[i][0], yb, o2[i][1]]], [0, -1, 0], info, tint);
      // fascia
      buf.q([[o2[i][0], yb, o2[i][1]], [o2[j][0], yb, o2[j][1]], [o2[j][0], yt, o2[j][1]], [o2[i][0], yt, o2[i][1]]], [nx, 0, nz], info, tint);
      // top back to the wall line
      buf.q([[a[0], yt, a[1]], [b[0], yt, b[1]], [o2[j][0], yt, o2[j][1]], [o2[i][0], yt, o2[i][1]]], [0, 1, 0], info, tint);
    }
  }

  // ---------------------------------------------------------------- storefront: signs, awnings, blade signs
  storefront(buf, f, st, fh, rnd, seed, u0, gl = () => 0) {
    const nb = Math.max(1, Math.round(f.L / st.bayW)), bw = f.L / nb;
    const P = (s, y, e) => [f.ax + f.dx * s + f.ox * e, y, f.az + f.dz * s + f.oz * e];
    const n = [f.ox, 0, f.oz], d = [f.dx, 0, f.dz];
    const yTopGlass = WIN_RECTS[FACADE.STOREFRONT][0][3] * fh;
    const side = [-1, -1, seed, enc(PBR.PLAIN, MODE.PLAIN)];
    let i = 0;
    while (i < nb) {
      const w = Math.min(nb - i, 1 + Math.floor(rnd() * 3));
      const s0 = i * bw + 0.15, s1 = (i + w) * bw - 0.15;
      i += w;
      if (s1 - s0 < 1.5) continue;
      const len = s1 - s0;
      if (Math.max(gl(s0), gl(s1), gl((s0 + s1) / 2)) > 0.35) continue; // shop front buried by the rising street
      // sign board (front + top)
      if (rnd() < 0.72) {
        const y0 = yTopGlass + 0.12, y1 = Math.min(fh - 0.08, y0 + 0.55 + rnd() * 0.3), e = 0.12 + rnd() * 0.06;
        const col = tb(pick(PAL.sign, rnd()), 0.9 + rnd() * 0.2);
        const info = [rnd() < 0.85 ? 1 : 0, len, rnd(), enc(PBR.PLAIN, MODE.SIGN)];
        buf.q([P(s0, y0, e), P(s1, y0, e), P(s1, y1, e), P(s0, y1, e)], n, info, col, (p, k) => [k === 1 || k === 2 ? len : 0, k >= 2 ? 1 : 0]);
        buf.q([P(s0, y1, 0), P(s1, y1, 0), P(s1, y1, e), P(s0, y1, e)], [0, 1, 0], side, col);
      }
      // awning: double-sided fabric slope + valance, cheeks
      if (rnd() < 0.38) {
        const ya = yTopGlass + 0.05, out = 1.1 + rnd() * 0.5, drop = 0.55 + rnd() * 0.25;
        const col = tb(pick(PAL.awning, rnd()));
        const info = [0, rnd() < 0.35 ? 1 : 0, seed, enc(PBR.FABRIC, MODE.SIGN)];
        const a0 = P(s0, ya, 0.02), a1 = P(s1, ya, 0.02), b1 = P(s1, ya - drop, out), b0 = P(s0, ya - drop, out);
        const sl = Math.hypot(out, drop), un = [f.ox * drop / sl, out / sl, f.oz * drop / sl];
        const uv = (p, k) => [(k === 1 || k === 2 ? s1 : s0), k >= 2 ? 1 : 0];
        buf.q([a0, a1, b1, b0], un, info, col, uv);
        buf.q([a0, a1, b1, b0], un.map((v) => -v), info, col.map((v) => Math.round(v * 0.75)), uv);
        const v0 = P(s0, ya - drop - 0.25, out), v1 = P(s1, ya - drop - 0.25, out);
        buf.q([b0, b1, v1, v0], n, info, col, uv);
        buf.q([b0, b1, v1, v0], n.map((v) => -v), info, col, uv);
        buf.poly([P(s0, ya, 0.02), P(s0, ya - drop, out), P(s0, ya - drop, 0.02)], d.map((v) => -v), info, col);
        buf.poly([P(s1, ya, 0.02), P(s1, ya - drop, out), P(s1, ya - drop, 0.02)], d, info, col);
      }
      // projecting blade sign (older commercial streets)
      if (st.prewar && rnd() < 0.14) {
        const s = rnd() < 0.5 ? s0 + 0.2 : s1 - 0.2, y0 = fh + 0.5, y1 = y0 + 1.8 + rnd() * 1.5, e0 = 0.25, e1 = 1.15, t = 0.09;
        const col = tb(pick(PAL.sign, rnd()));
        const info = [1, e1 - e0, rnd(), enc(PBR.PLAIN, MODE.SIGN)];
        for (const sg of [-1, 1]) {
          const ss = s + sg * t;
          buf.q([P(ss, y0, e0), P(ss, y0, e1), P(ss, y1, e1), P(ss, y1, e0)], d.map((v) => v * sg), info, col, (p, k) => [k === 1 || k === 2 ? e1 - e0 : 0, k >= 2 ? 1 : 0]);
        }
        buf.q([P(s - t, y0, e1), P(s + t, y0, e1), P(s + t, y1, e1), P(s - t, y1, e1)], n, side, col);
      }
    }
    void u0;
  }

  // ---------------------------------------------------------------- Victorian / Edwardian bay window
  bayWindow(buf, f, st, fh, floors, top, seed, gl = () => 0) {
    const W = Math.min(3.8, f.L * 0.5), Pj = 0.75;
    const c = f.L * 0.5 + (f.L > 9 ? (seed - 0.5) * (f.L - W - 2) : 0);
    const s0 = c - W / 2, s1 = c + W / 2, f0 = s0 + Pj, f1 = s1 - Pj;
    const y0 = Math.max(floors >= 3 ? fh + 0.15 : 0.7, Math.max(gl(s0), gl(s1), gl(c)) + 2.6), y1 = top - 0.45;
    if (y1 - y0 < 2) return;
    const P = (s, e, y) => [f.ax + f.dx * s + f.ox * e, y, f.az + f.dz * s + f.oz * e];
    const pts = [[s0, 0], [f0, Pj], [f1, Pj], [s1, 0]];
    const info = [FACADE.VICTORIAN, -1, seed, enc(st.pbr, MODE.WALL)];
    let ucur = 0;
    for (let k = 0; k < 3; k++) {
      const [sa, ea] = pts[k], [sb, eb] = pts[k + 1];
      const A0 = P(sa, ea, y0), B0 = P(sb, eb, y0);
      const dx = B0[0] - A0[0], dz = B0[2] - A0[2], l = Math.hypot(dx, dz);
      const nrm = [dz / l, 0, -dx / l];
      const fl = (y) => y / fh;
      const uw = Math.max(0.6, l / 2.0);
      buf.q([A0, B0, P(sb, eb, y1), P(sa, ea, y1)], nrm, info, st.tint, (p, i) => [ucur + (i === 1 || i === 2 ? uw : 0), fl(p[1])]);
      ucur += uw;
    }
    // caps: bottom soffit and top with a small cornice
    const ring = pts.map(([s, e]) => P(s, e, 0));
    const trim = [-1, -1, seed, enc(st.trimPbr, MODE.PLAIN)];
    buf.poly(ring.map((p) => [p[0], y0, p[2]]), [0, -1, 0], trim, st.trimTint);
    const top2 = pts.map(([s, e]) => P(s + (s < c ? -0.15 : 0.15) * (e === 0 ? 1 : 0.4), e + 0.15, 0));
    buf.poly(top2.map((p) => [p[0], y1 + 0.35, p[2]]), [0, 1, 0], trim, st.trimTint);
    for (let k = 0; k < 3; k++) {
      const a = top2[k], b = top2[k + 1], dx = b[0] - a[0], dz = b[2] - a[2], l = Math.hypot(dx, dz) || 1;
      buf.q([[a[0], y1, a[2]], [b[0], y1, b[2]], [b[0], y1 + 0.35, b[2]], [a[0], y1 + 0.35, a[2]]], [dz / l, 0, -dx / l], trim, st.trimTint);
      const r0 = ring[k], r1 = ring[k + 1];
      buf.q([[r0[0], y1, r0[2]], [r1[0], y1, r1[2]], [b[0], y1, b[2]], [a[0], y1, a[2]]], [0, -1, 0], trim, st.trimTint);
    }
  }

  // ---------------------------------------------------------------- fire escape (near detail level)
  fireEscape(buf, f, st, fh, floors, top, rnd, base) {
    const nb = Math.max(1, Math.round(f.L / st.bayW)), bw = f.L / nb;
    const k = Math.floor(rnd() * Math.max(1, nb - 1));
    const s0 = k * bw + 0.15, s1 = Math.min(f.L - 0.15, (k + 2) * bw - 0.15), Wd = s1 - s0, D = 1.1;
    if (Wd < 2.4) return;
    const P = (s, e, y) => [f.ax + f.dx * s + f.ox * e, y, f.az + f.dz * s + f.oz * e];
    const n = [f.ox, 0, f.oz], d = [f.dx, 0, f.dz];
    const col = tb([0.1, 0.1, 0.11]);
    const solid = [-1, -1, 0.5, enc(PBR.PLAIN, MODE.PLAIN)];
    const bars = [0, 0, 0.5, enc(PBR.PLAIN, MODE.GRATE)];
    const both = (pts, nn, info, uv) => { buf.q(pts, nn, info, col, uv); buf.q(pts, nn.map((v) => -v), info, col, uv); };
    const topRow = Math.min(floors - 1, Math.floor((top - 1) / fh));
    for (let fl = 1; fl <= topRow; fl++) {
      const y = fl * fh + 0.02;
      // slab (grated platform drawn solid, thin)
      buf.q([P(s0, 0.02, y), P(s1, 0.02, y), P(s1, D, y), P(s0, D, y)], [0, 1, 0], solid, col);
      buf.q([P(s0, 0.02, y - 0.06), P(s1, 0.02, y - 0.06), P(s1, D, y - 0.06), P(s0, D, y - 0.06)], [0, -1, 0], solid, col);
      buf.q([P(s0, D, y - 0.06), P(s1, D, y - 0.06), P(s1, D, y), P(s0, D, y)], n, solid, col);
      // railings
      both([P(s0, D, y), P(s1, D, y), P(s1, D, y + 0.95), P(s0, D, y + 0.95)], n, bars, (p, i) => [i === 1 || i === 2 ? Wd : 0, i >= 2 ? 1 : 0]);
      both([P(s0, 0.02, y), P(s0, D, y), P(s0, D, y + 0.95), P(s0, 0.02, y + 0.95)], d, bars, (p, i) => [i === 1 || i === 2 ? D : 0, i >= 2 ? 1 : 0]);
      both([P(s1, 0.02, y), P(s1, D, y), P(s1, D, y + 0.95), P(s1, 0.02, y + 0.95)], d, bars, (p, i) => [i === 1 || i === 2 ? D : 0, i >= 2 ? 1 : 0]);
      // stair to the next landing (alternating direction), treads pattern
      if (fl < topRow) {
        const sw = 0.62, e0 = D - sw - 0.05, e1 = D - 0.05;
        const left = fl % 2 === 0;
        const sa = left ? s0 + 0.3 : s1 - 0.3, sb = left ? s1 - 0.6 : s0 + 0.6;
        const y2 = y + fh;
        const run = Math.hypot(sb - sa, fh);
        const info = [1, 0, 0.5, enc(PBR.PLAIN, MODE.GRATE)];
        const nrm = [0, 1, 0];
        both([P(sa, e0, y), P(sa, e1, y), P(sb, e1, y2), P(sb, e0, y2)], nrm, info, (p, i) => [i === 1 || i === 2 ? 1 : 0, i >= 2 ? run : 0]);
      }
    }
    // drop ladder under the first landing
    const yl = fh + 0.02, sL = s0 + 0.5;
    const ladder = [2, 0, 0.5, enc(PBR.PLAIN, MODE.GRATE)];
    const yBottom = Math.max(yl - 3.2, (this.groundAt ? this.groundAt(...[f.ax + f.dx * sL, f.az + f.dz * sL]) - base : 0) + 2.4);
    if (yl - yBottom > 0.6) both([P(sL, D - 0.08, yBottom), P(sL + 0.45, D - 0.08, yBottom), P(sL + 0.45, D - 0.08, yl), P(sL, D - 0.08, yl)], n, ladder, (p, i) => [i === 1 || i === 2 ? 1 : 0, p[1] - yBottom]);
  }

  // ---------------------------------------------------------------- balconies (near detail level)
  balconies(buf, f, st, fh, floors, top, rnd, seed) {
    const nb = Math.max(1, Math.round(f.L / st.bayW)), bw = f.L / nb;
    const pattern = Math.floor(seed * 3); // 0 every bay, 1 alternate, 2 stacked pairs
    const col = tb([0.86, 0.86, 0.84]), rail = tb([0.2, 0.21, 0.22]);
    const slab = [-1, -1, seed, enc(PBR.CONCRETE_PANEL, MODE.PLAIN)];
    const bars = [0, 0, seed, enc(PBR.PLAIN, MODE.GRATE)];
    const P = (s, e, y) => [f.ax + f.dx * s + f.ox * e, y, f.az + f.dz * s + f.oz * e];
    const n = [f.ox, 0, f.oz], d = [f.dx, 0, f.dz], D = 1.25;
    for (let fl = 1; fl < floors; fl++) {
      const y = fl * fh;
      if (y > top - 1) break;
      for (let i = 0; i < nb; i++) {
        if (pattern === 1 && (i + fl) % 2) continue;
        if (pattern === 2 && i % 3 === 2) continue;
        const s0 = i * bw + 0.2, s1 = (i + 1) * bw - 0.2;
        buf.q([P(s0, 0, y), P(s1, 0, y), P(s1, D, y), P(s0, D, y)], [0, 1, 0], slab, col);
        buf.q([P(s0, 0, y - 0.18), P(s1, 0, y - 0.18), P(s1, D, y - 0.18), P(s0, D, y - 0.18)], [0, -1, 0], slab, col);
        buf.q([P(s0, D, y - 0.18), P(s1, D, y - 0.18), P(s1, D, y), P(s0, D, y)], n, slab, col);
        buf.q([P(s0, 0, y - 0.18), P(s0, D, y - 0.18), P(s0, D, y), P(s0, 0, y)], d.map((v) => -v), slab, col);
        buf.q([P(s1, 0, y - 0.18), P(s1, D, y - 0.18), P(s1, D, y), P(s1, 0, y)], d, slab, col);
        const W = s1 - s0;
        for (const sg of [1, -1]) {
          buf.q([P(s0, D, y), P(s1, D, y), P(s1, D, y + 1.05), P(s0, D, y + 1.05)], n.map((v) => v * sg), bars, rail, (p, k) => [k === 1 || k === 2 ? W : 0, k >= 2 ? 1 : 0]);
          buf.q([P(s0, 0, y), P(s0, D, y), P(s0, D, y + 1.05), P(s0, 0, y + 1.05)], d.map((v) => v * sg), bars, rail, (p, k) => [k === 1 || k === 2 ? D : 0, k >= 2 ? 1 : 0]);
          buf.q([P(s1, 0, y), P(s1, D, y), P(s1, D, y + 1.05), P(s1, 0, y + 1.05)], d.map((v) => v * sg), bars, rail, (p, k) => [k === 1 || k === 2 ? D : 0, k >= 2 ? 1 : 0]);
        }
      }
    }
  }

  // ---------------------------------------------------------------- roofs
  flatPolygon(buf, outer, holes, y, info, tint, up) {
    let faces;
    try { faces = THREE.ShapeUtils.triangulateShape(outer.map(this._v2), holes.map((hr) => hr.map(this._v2))); } catch { return; }
    if (!faces || !faces.length) return;
    const all = outer.concat(...holes);
    const k = all.map((p) => buf.v(p[0], y, p[1], 0, up ? 1 : -1, 0, p[0] / 3, p[1] / 3, info, tint));
    for (const f of faces) if (k[f[0]] !== undefined && k[f[1]] !== undefined && k[f[2]] !== undefined) buf.tri(k[f[0]], k[f[1]], k[f[2]]);
  }
  heightPolygon(buf, ring, topH, info, tint) {
    if (!ring || ring.length < 3) return;
    let faces;
    try { faces = THREE.ShapeUtils.triangulateShape(ring.map(this._v2), []); } catch { return; }
    if (!faces || !faces.length) return;
    const P = ring.map((p) => [p[0], topH(p[0], p[1]), p[1]]);
    let nx = 0, ny = 0, nz = 0;
    for (let i = 0; i < P.length; i++) {
      const a = P[i], b = P[(i + 1) % P.length];
      nx += (a[1] - b[1]) * (a[2] + b[2]); ny += (a[2] - b[2]) * (a[0] + b[0]); nz += (a[0] - b[0]) * (a[1] + b[1]);
    }
    const l = Math.hypot(nx, ny, nz) || 1; nx /= l; ny /= l; nz /= l;
    if (ny < 0) { nx = -nx; ny = -ny; nz = -nz; }
    const k = P.map((p) => buf.v(p[0], p[1], p[2], nx, ny, nz, p[0] / 3, p[2] / 3, info, tint));
    for (const f of faces) if (k[f[0]] !== undefined && k[f[1]] !== undefined && k[f[2]] !== undefined) buf.tri(k[f[0]], k[f[1]], k[f[2]]);
  }
  hipRoof(buf, q, y0, rh, info, tint, ov = 0, seed = 0.5, st = null) {
    const e = (i) => Math.hypot(q[(i + 1) % 4][0] - q[i][0], q[(i + 1) % 4][1] - q[i][1]);
    const s = e(0) + e(2) > e(1) + e(3) ? 0 : 1;
    const mid = (i) => [(q[i][0] + q[(i + 1) % 4][0]) / 2, (q[i][1] + q[(i + 1) % 4][1]) / 2];
    const mA = mid((s + 1) % 4), mB = mid((s + 3) % 4);
    const shortHalf = (e((s + 1) % 4) + e((s + 3) % 4)) / 4;
    const dx = mB[0] - mA[0], dz = mB[1] - mA[1], L = Math.hypot(dx, dz);
    if (L < 2 * shortHalf + 0.2) { this.taperRoof(buf, q, centroid(q), y0, rh, [[1, 0], [0, 1]], info, tint); return; }
    const ux = dx / L, uz = dz / L;
    const rA = [mA[0] + ux * shortHalf, y0 + rh, mA[1] + uz * shortHalf], rB = [mB[0] - ux * shortHalf, y0 + rh, mB[1] - uz * shortHalf];
    // eaves: the same planes continued ov beyond the walls (dropping ov * slope), plus fascia / soffit
    const slope = rh / Math.max(0.5, shortHalf), qO = ov ? offsetRing(q, ov) : q, yE = y0 - ov * slope;
    const C = qO.map((p) => [p[0], yE, p[1]]);
    if (ov) this.eaves(buf, q, qO, () => yE, seed, st);
    const i1 = (s + 1) % 4, i2 = (s + 2) % 4, i3 = (s + 3) % 4, i0 = s;
    this.face(buf, [C[i0], C[i1], rA, rB], info, tint);
    this.face(buf, [C[i1], C[i2], rA], info, tint);
    this.face(buf, [C[i2], C[i3], rB, rA], info, tint);
    this.face(buf, [C[i3], C[i0], rB], info, tint);
  }
  taperRoof(buf, ring, c, y0, rh, prof, info, tint) {
    const rings = prof.map(([s, t]) => ring.map((p) => [c[0] + (p[0] - c[0]) * s, y0 + rh * t, c[1] + (p[1] - c[1]) * s]));
    for (let k = 0; k + 1 < rings.length; k++) {
      const A = rings[k], Bq = rings[k + 1];
      for (let i = 0; i < ring.length; i++) {
        const j = (i + 1) % ring.length;
        if (prof[k + 1][0] === 0) this.face(buf, [A[i], A[j], Bq[i]], info, tint);
        else this.face(buf, [A[i], A[j], Bq[j], Bq[i]], info, tint);
      }
    }
  }
  /** Convex planar face (3D points) oriented upward-ish. */
  face(buf, P, info, tint) {
    const a = P[0], b = P[1], c = P[2];
    let nx = (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]);
    let ny = (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]);
    let nz = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    const l = Math.hypot(nx, ny, nz);
    if (!(l > 1e-8)) return;
    nx /= l; ny /= l; nz /= l;
    if (ny < 0) { nx = -nx; ny = -ny; nz = -nz; }
    buf.poly(P, [nx, ny, nz], info, tint, (p) => [p[0] / 3, p[2] / 3]);
  }
  /** Fascia (0.18 m) round the overhanging roof edge ringO and the soffit back to the wall ring. hAt(p): roof height. */
  eaves(buf, ring, ringO, hAt, seed, st) {
    const n = ring.length, T = 0.18;
    const info = [-1, -1, seed, enc(PBR.PLAIN, MODE.PLAIN)];
    const fas = st && st.cat === 'victorian' ? tb([0.95, 0.94, 0.9]) : tb([0.62, 0.6, 0.57]), sof = fas.map((v) => Math.round(v * 0.72));
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n, a = ringO[i], b2 = ringO[j];
      const ex = b2[0] - a[0], ez = b2[1] - a[1], El = Math.hypot(ex, ez);
      if (El < 0.05) continue;
      const ha = hAt(a), hb = hAt(b2), wa = hAt(ring[i]), wb = hAt(ring[j]);
      buf.q([[a[0], ha - T, a[1]], [b2[0], hb - T, b2[1]], [b2[0], hb, b2[1]], [a[0], ha, a[1]]], [ez / El, 0, -ex / El], info, fas);
      buf.q([[a[0], ha - T, a[1]], [b2[0], hb - T, b2[1]], [ring[j][0], Math.min(wb, hb) - T, ring[j][1]], [ring[i][0], Math.min(wa, ha) - T, ring[i][1]]], [0, -1, 0], info, sof);
    }
  }
  /** Oriented box (bottom optional). info for sides and top (topInfo optional). */
  box(buf, cx, cz, y0, sx, sz, sy, ang, info, tint, topInfo = null, topTint = null) {
    const ca = Math.cos(ang), sa = Math.sin(ang);
    const P = [[-sx / 2, -sz / 2], [sx / 2, -sz / 2], [sx / 2, sz / 2], [-sx / 2, sz / 2]].map(([x, z]) => [cx + x * ca - z * sa, cz + x * sa + z * ca]);
    const y1 = y0 + sy;
    for (let i = 0; i < 4; i++) {
      const a = P[i], b = P[(i + 1) % 4];
      const mx = (a[0] + b[0]) / 2 - cx, mz = (a[1] + b[1]) / 2 - cz, l = Math.hypot(mx, mz) || 1;
      const L = Math.hypot(b[0] - a[0], b[1] - a[1]);
      buf.q([[a[0], y0, a[1]], [b[0], y0, b[1]], [b[0], y1, b[1]], [a[0], y1, a[1]]], [mx / l, 0, mz / l], info, tint, (p, k) => [k === 1 || k === 2 ? L / 3 : 0, p[1] / 3]);
    }
    buf.q(P.map((p) => [p[0], y1, p[1]]), [0, 1, 0], topInfo || info, topTint || tint);
    return P;
  }
  cylinder(buf, cx, cz, y0, r, hgt, seg, info, tint, cap = 'flat', capH = 0, capInfo = null, capTint = null) {
    const y1 = y0 + hgt;
    const pts = [];
    for (let i = 0; i < seg; i++) { const a = (i / seg) * Math.PI * 2; pts.push([Math.cos(a), Math.sin(a)]); }
    for (let i = 0; i < seg; i++) {
      const p = pts[i], q = pts[(i + 1) % seg], m = [(p[0] + q[0]) / 2, (p[1] + q[1]) / 2], ml = Math.hypot(m[0], m[1]);
      buf.q([[cx + p[0] * r, y0, cz + p[1] * r], [cx + q[0] * r, y0, cz + q[1] * r], [cx + q[0] * r, y1, cz + q[1] * r], [cx + p[0] * r, y1, cz + p[1] * r]], [m[0] / ml, 0, m[1] / ml], info, tint, (pp, k) => [((i + (k === 1 || k === 2 ? 1 : 0)) / seg) * 2 * Math.PI * r / 3, pp[1] / 3]);
    }
    if (cap === 'flat') buf.poly(pts.map((p) => [cx + p[0] * r, y1, cz + p[1] * r]), [0, 1, 0], capInfo || info, capTint || tint);
    else if (cap === 'cone') {
      const apex = [cx, y1 + capH, cz];
      for (let i = 0; i < seg; i++) {
        const p = pts[i], q = pts[(i + 1) % seg];
        this.face(buf, [[cx + p[0] * r * 1.06, y1, cz + p[1] * r * 1.06], [cx + q[0] * r * 1.06, y1, cz + q[1] * r * 1.06], apex], capInfo || info, capTint || tint);
      }
    }
  }

  rooftop(ctx, buf, outer, holes, y, area, h, rnd, seed, st) {
    const ang = Math.atan2(...(() => { const a = longAxis(outer); return [a[1], a[0]]; })());
    const bb = ringBounds(outer);
    const ca = Math.cos(ang), sa = Math.sin(ang);
    const fits = (cx, cz, hx, hz) => {
      for (const [x, z] of [[-hx, -hz], [hx, -hz], [hx, hz], [-hx, hz], [0, 0]]) if (!pointInPoly(cx + x * ca - z * sa, cz + x * sa + z * ca, outer, holes)) return false;
      return true;
    };
    const place = (hx, hz, tries = 10) => {
      for (let t = 0; t < tries; t++) {
        const cx = bb.minX + rnd() * (bb.maxX - bb.minX), cz = bb.minZ + rnd() * (bb.maxZ - bb.minZ);
        if (fits(cx, cz, hx + 0.8, hz + 0.8)) return [cx, cz];
      }
      return null;
    };
    const metal = tb([0.78, 0.79, 0.8]);
    const metalInfo = [-1, -1, seed, enc(PBR.METAL, MODE.PLAIN)];
    const plainInfo = [-1, -1, seed, enc(PBR.PLAIN, MODE.PLAIN)];
    const wallInfo = [-1, -1, seed, enc(st.curtain ? PBR.CONCRETE_PANEL : st.pbr, MODE.PLAIN)];
    const wallTint = st.curtain ? tb([0.72, 0.73, 0.74]) : st.tint;
    let bigPent = false;
    // mechanical penthouse / screened plant on tall buildings
    if (h > 40 && area > 400) {
      const c = centroid(outer);
      for (let k = 0; k < 6; k++) {
        const w = Math.sqrt(area) * (0.42 - k * 0.05);
        if (w < 4) break;
        if (fits(c[0], c[1], w / 2 + 1, w * 0.35 + 1)) {
          const ph = 3.5 + rnd() * 4.5;
          if (st.curtain) this.box(buf, c[0], c[1], y, w, w * 0.7, ph, ang, [st.glassVariant, -1, seed, enc(PBR.PLAIN, MODE.CURTAIN)], st.glassTint, plainInfo, tb([0.5, 0.5, 0.5]));
          else this.box(buf, c[0], c[1], y, w, w * 0.7, ph, ang, wallInfo, wallTint, [-1, -1, seed, enc(PBR.ROOF_TAR, MODE.PLAIN)], st.roofTint);
          bigPent = true;
          // louvred plant screen on top
          if (rnd() < 0.6) this.box(buf, c[0], c[1], y + ph, w * 0.6, w * 0.4, 2.2, ang, metalInfo, metal, plainInfo, tb([0.4, 0.4, 0.42]));
          break;
        }
      }
    }
    // elevator / stair penthouse
    if (!bigPent && ctx.floors >= 5 && area > 150) {
      const pos = place(2.2, 3);
      if (pos) this.box(buf, pos[0], pos[1], y, 4.4, 6, 3.6, ang, wallInfo, wallTint, [-1, -1, seed, enc(PBR.ROOF_TAR, MODE.PLAIN)], st.roofTint);
    }
    // HVAC units
    const nH = Math.min(8, Math.floor(area / 500) + (area > 200 ? 1 : 0));
    for (let i = 0; i < nH; i++) {
      const sx = 1.6 + rnd() * 3.4, sz = 1.2 + rnd() * 2.4, sy = 1 + rnd() * 1.6;
      const pos = place(sx / 2, sz / 2, 8);
      if (pos) this.box(buf, pos[0], pos[1], y, sx, sz, sy, ang, metalInfo, metal, plainInfo, tb([0.34, 0.35, 0.36]));
    }
    // timber water tank on a steel stand (older mid-rises)
    if (st.prewar && ctx.floors >= 4 && ctx.floors <= 16 && area > 180 && rnd() < 0.12) {
      const r = 1.7 + rnd() * 0.9;
      const pos = place(r + 0.5, r + 0.5);
      if (pos) {
        const steel = tb([0.16, 0.16, 0.17]);
        for (const [lx, lz] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) this.box(buf, pos[0] + lx * r * 0.6, pos[1] + lz * r * 0.6, y, 0.22, 0.22, 2.0, 0, plainInfo, steel);
        this.box(buf, pos[0], pos[1], y + 1.9, r * 1.9, r * 1.9, 0.12, 0, plainInfo, steel);
        this.cylinder(buf, pos[0], pos[1], y + 2.02, r, 3.2 + rnd() * 1.6, 12, [-1, -1, seed, enc(PBR.WOOD_SIDING, MODE.PLAIN)], tb([0.62, 0.48, 0.36]), 'cone', 0.9, [-1, -1, seed, enc(PBR.ROOF_TAR, MODE.PLAIN)], tb([0.3, 0.3, 0.3]));
      }
    }
    // antennas with red beacons on tall towers
    if (h > 85 && rnd() < 0.4) {
      const nA = 1 + Math.floor(rnd() * 3);
      for (let i = 0; i < nA; i++) {
        const pos = place(0.4, 0.4);
        if (!pos) continue;
        const ah = 8 + rnd() * 18;
        this.box(buf, pos[0], pos[1], y, 0.35, 0.35, ah, ang, metalInfo, metal);
        this.box(buf, pos[0], pos[1], y + ah, 0.5, 0.5, 0.4, 0, [2, -1, rnd(), enc(PBR.PLAIN, MODE.LED)], tb([1, 0.1, 0.05]));
      }
    }
    // skylights on low commercial / industrial roofs
    if (h < 16 && area > 260 && (st.cat === 'industrial' || st.cat === 'masonry' || st.cat === 'office') && rnd() < 0.5) {
      const nS = 1 + Math.floor(rnd() * Math.min(4, area / 300));
      for (let i = 0; i < nS; i++) {
        const lx = 3 + rnd() * 5, wz = 1.4;
        const pos = place(lx / 2, wz / 2);
        if (!pos) continue;
        const P = (x, z, yy) => [pos[0] + x * ca - z * sa, yy, pos[1] + x * sa + z * ca];
        const gi = [1, -1, seed, enc(PBR.PLAIN, MODE.CURTAIN)], gt = tb([0.7, 0.78, 0.82]);
        this.box(buf, pos[0], pos[1], y, lx, wz, 0.35, ang, plainInfo, tb([0.7, 0.7, 0.7]));
        const yr = y + 0.35, yt = yr + 0.6;
        this.face(buf, [P(-lx / 2, -wz / 2, yr), P(lx / 2, -wz / 2, yr), P(lx / 2, 0, yt), P(-lx / 2, 0, yt)], gi, gt);
        this.face(buf, [P(-lx / 2, wz / 2, yr), P(lx / 2, wz / 2, yr), P(lx / 2, 0, yt), P(-lx / 2, 0, yt)], gi, gt);
      }
    }
    // roof garden on some modern low/mid-rises
    if ((st.cat === 'res' || st.curtain || st.cat === 'office') && h < 70 && area > 300 && rnd() < 0.22) {
      const g = insetRing(outer, 2.2);
      if (g && Math.abs(signedArea(g)) > 60 && isConvex(g)) {
        const gy = y + 0.45;
        this.flatPolygon(buf, g, [], gy, [-1, -1, seed, enc(PBR.GRASS, MODE.PLAIN)], tb([0.95, 1, 0.92]), true);
        for (let i = 0; i < g.length; i++) {
          const a = g[i], b = g[(i + 1) % g.length], dx = b[0] - a[0], dz = b[1] - a[1], L = Math.hypot(dx, dz);
          if (L < 0.1) continue;
          buf.q([[a[0], y, a[1]], [b[0], y, b[1]], [b[0], gy + 0.1, b[1]], [a[0], gy + 0.1, a[1]]], [dz / L, 0, -dx / L], [-1, -1, seed, enc(PBR.CONCRETE_PANEL, MODE.PLAIN)], tb([0.8, 0.8, 0.78]));
        }
      }
    }
  }

  // ---------------------------------------------------------------- output
  /**
   * @param {object} mats createFacadeMaterials() result
   * @returns {THREE.Object3D[]} base chunk meshes + one THREE.LOD per detail cell
   */
  build(mats) {
    const out = [];
    this.stats = { baseTris: 0, nearTris: 0, farTris: 0, windows: 0, windowTris: 0, cells: 0, chunks: 0, detailed: this.detailed };
    for (const b of this.base) {
      const m = b.build(mats, 'buildings');
      if (m) { out.push(m); this.stats.baseTris += b.tris; this.stats.chunks++; }
    }
    const unit = unitWindow();
    for (const c of this.cells) {
      if (!c.near.n && !c.wins.n && !c.far.n) continue;
      const lod = new THREE.LOD();
      lod.name = 'buildings-detail';
      lod.position.set(c.cx, 40, c.cz);
      const near = new THREE.Group(); near.name = 'buildings-near';
      const mn = c.near.build(mats, 'buildings-near-walls'); if (mn) near.add(mn);
      const mw = c.wins.build(mats, unit); if (mw) near.add(mw);
      for (const m of near.children) { m.position.set(-c.cx, -40, -c.cz); m.updateMatrix(); }
      const far = c.far.build(mats, 'buildings-far') || new THREE.Object3D();
      far.position.set(-c.cx, -40, -c.cz); far.updateMatrix();
      lod.addLevel(near, 0);
      lod.addLevel(far, this.lodDist, 0.05);
      lod.updateMatrix();
      out.push(lod);
      this.stats.nearTris += c.near.tris; this.stats.farTris += c.far.tris;
      this.stats.windows += c.wins.n; this.stats.windowTris += c.wins.n * UNIT_WINDOW_TRIS; this.stats.cells++;
    }
    this.stats.totalTris = this.stats.baseTris + this.stats.nearTris + this.stats.farTris + this.stats.windowTris;
    return out;
  }
}

// ================================================================== streaming tiles
// Tile builds reuse their scratch GeoBufs (a streaming session builds hundreds of tiles: without the pool every build
// would grow fresh buffers by doubling and drop them, which shows up as GC churn / heap peaks while flying).
const _bufPool = [];
const POOL_MAX = 4, POOL_MAX_VERTS = 1 << 17;
function takeGeoBuf(cap) {
  const b = _bufPool.pop();
  if (!b) return new GeoBuf(cap);
  b.n = 0; b.ni = 0; b.wall = null;
  return b;
}
function releaseGeoBuf(b) { if (b instanceof GeoBuf && _bufPool.length < POOL_MAX && b.vcap <= POOL_MAX_VERTS) _bufPool.push(b); }
/** Drop the pooled tile-build scratch buffers (MapWorld calls this once streaming has been idle for a while). */
export function releaseTileBuffers() { const n = _bufPool.length; _bufPool.length = 0; return n; }

/** Geometry sink with the GeoBuf interface that stores nothing (tile builds that need only some of the outputs). */
export class NullBuf {
  constructor() { this.n = 0; this.ni = 0; this.wall = null; }
  v() { return this.n++; }
  tri() { this.ni += 3; }
  quad() { this.ni += 6; }
  q() { this.n += 4; this.ni += 6; }
  poly(P) { this.n += P.length; this.ni += 3 * Math.max(0, P.length - 2); }
  lift() {}
  add() { this.n++; }
  get tris() { return 0; }
  geometry() { return null; }
  build() { return null; }
}

/**
 * BuildingBuilder for one streaming tile (MapWorld streaming): every building goes to the tile's single base buffer,
 * detailed walls to a per-detail-cell entry that keeps the global cell's centre (the LOD switch point is where the
 * global grid has it, so near / far LOD decisions match a full build). Per building the output is byte-identical to a
 * full build (same per-building seed, same code). mode: 'low' = base + far shells (no real openings), 'full' = base +
 * far + near walls + window units, 'near' = near walls + window units only (upgrades a 'low' tile).
 */
export class TileBuildingBuilder extends BuildingBuilder {
  constructor(region, opts = {}, mode = 'full') {
    super(region, { ...opts, grid: false, nearDetail: mode !== 'low' });
    this.mode = mode;
    this.tileBase = mode === 'near' ? new NullBuf() : takeGeoBuf(4096);
    this.tileCells = new Map();
    this.detailedIds = [];
  }
  baseFor() { return this.tileBase; }
  cellFor(x, z) {
    const key = this.cellIndex(x, z);
    let c = this.tileCells.get(key);
    if (!c) {
      const [cx, cz] = this.cellCentre(key), m = this.mode;
      c = { key, cx, cz, near: m === 'low' ? new NullBuf() : takeGeoBuf(2048), far: m === 'near' ? new NullBuf() : takeGeoBuf(1024), wins: m === 'low' ? new NullBuf() : new WinBuf() };
      this.tileCells.set(key, c);
    }
    return c;
  }
  /**
   * Generator: turn the buffers into meshes, yielding after each geometry (time slicing). Returns
   * { base: Mesh|null, cells: [{ key, cx, cz, far: Mesh|null, near: Group|null }], bytes, tris }. Near groups / far
   * meshes are positioned relative to the cell centre at y = 40 (as BuildingBuilder.build's THREE.LOD levels).
   */
  * meshes(mats) {
    const out = { base: null, cells: [], bytes: 0, tris: 0 };
    const bytes = (m) => { if (m && m.geometry) for (const k in m.geometry.attributes) out.bytes += m.geometry.attributes[k].array.byteLength; if (m && m.geometry && m.geometry.index) out.bytes += m.geometry.index.array.byteLength; };
    if (this.tileBase.n && !(this.tileBase instanceof NullBuf)) { out.base = this.tileBase.build(mats, 'buildings'); bytes(out.base); out.tris += this.tileBase.tris; yield; }
    releaseGeoBuf(this.tileBase); this.tileBase = null;
    const unit = unitWindow();
    for (const c of this.tileCells.values()) {
      const e = { key: c.key, cx: c.cx, cz: c.cz, far: null, near: null };
      if (c.far.n && !(c.far instanceof NullBuf)) { e.far = c.far.build(mats, 'buildings-far'); e.far.position.set(-c.cx, -40, -c.cz); e.far.updateMatrix(); bytes(e.far); out.tris += c.far.tris; yield; }
      if ((c.near.n || c.wins.n) && !(c.near instanceof NullBuf)) {
        const near = new THREE.Group(); near.name = 'buildings-near';
        const mn = c.near.build(mats, 'buildings-near-walls'); if (mn) { near.add(mn); bytes(mn); out.tris += c.near.tris; }
        yield;
        const mw = c.wins.build(mats, unit); if (mw) { near.add(mw); bytes(mw); out.tris += c.wins.n * UNIT_WINDOW_TRIS; out.windows = (out.windows || 0) + c.wins.n; }
        for (const m of near.children) { m.position.set(-c.cx, -40, -c.cz); m.updateMatrix(); }
        e.near = near;
        yield;
      }
      if (e.far || e.near) out.cells.push(e);
      releaseGeoBuf(c.far); releaseGeoBuf(c.near);
    }
    this.tileCells.clear();
    return out;
  }
}

// ================================================================== helpers
/** Wall frame for edge a->b of a ring. Outer rings (positive area) face (dz,-dx); holes face the other way. */
function frame(a, b, isOuter) {
  const dx = b[0] - a[0], dz = b[1] - a[1], L = Math.hypot(dx, dz);
  if (L < 0.05) return null;
  let ox = dz / L, oz = -dx / L;
  if (!isOuter) { ox = -ox; oz = -oz; }
  // u must increase along T = cross(N, up) = (-nz, nx): equals a->b for outer rings, b->a for holes
  const uDir1 = isOuter ? 1 : 0, uDir0 = isOuter ? 0 : 1;
  return { ax: a[0], az: a[1], dx: dx / L, dz: dz / L, ox, oz, L, mx: (a[0] + b[0]) / 2, mz: (a[1] + b[1]) / 2, uDir0, uDir1 };
}
/** Decompose [0,L]x[y0,y1] minus axis-aligned holes into rectangles (bands merged vertically). */
function solidRects(L, y0, y1, holes, emit) {
  const ys = [y0, y1];
  for (const h of holes) { if (h[2] > y0 && h[2] < y1) ys.push(h[2]); if (h[3] > y0 && h[3] < y1) ys.push(h[3]); }
  ys.sort((a, b) => a - b);
  const Y = ys.filter((v, i) => i === 0 || v - ys[i - 1] > 0.004);
  let prevKey = null, prevY0 = 0, prevIv = null;
  const flush = (yEnd) => { if (prevIv) for (const [a, b] of prevIv) emit(a, b, prevY0, yEnd); };
  for (let k = 0; k + 1 < Y.length; k++) {
    const ya = Y[k], yb = Y[k + 1], ym = (ya + yb) / 2;
    const cut = holes.filter((h) => h[2] <= ym && h[3] >= ym).map((h) => [h[0], h[1]]).sort((p, q) => p[0] - q[0]);
    const iv = [];
    let s = 0;
    for (const [a, b] of cut) { if (a > s + 0.004) iv.push([s, a]); s = Math.max(s, b); }
    if (L > s + 0.004) iv.push([s, L]);
    const key = iv.map((p) => `${p[0].toFixed(3)}:${p[1].toFixed(3)}`).join('|');
    if (key !== prevKey) { flush(ya); prevKey = key; prevY0 = ya; prevIv = iv; }
  }
  flush(Y[Y.length - 1]);
}
function isConvex(r) {
  let sign = 0;
  for (let i = 0; i < r.length; i++) {
    const a = r[i], b = r[(i + 1) % r.length], c = r[(i + 2) % r.length];
    const cr = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
    if (Math.abs(cr) < 1e-6) continue;
    const s = Math.sign(cr);
    if (sign && s !== sign) return false;
    sign = s;
  }
  return true;
}
function starShaped(r, c) {
  for (let i = 0; i < r.length; i++) {
    const a = r[i], b = r[(i + 1) % r.length];
    if ((a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0]) <= 0) return false;
  }
  return true;
}
export function longAxis(r) {
  let best = 0, ax = [1, 0];
  for (let i = 0; i < r.length; i++) {
    const a = r[i], b = r[(i + 1) % r.length];
    const dx = b[0] - a[0], dz = b[1] - a[1], L = Math.hypot(dx, dz);
    if (L > best) { best = L; ax = [dx / L, dz / L]; }
  }
  return ax;
}
function minWidth(r) {
  const ax = longAxis(r);
  let d0 = Infinity, d1 = -Infinity;
  for (const p of r) { const d = -p[0] * ax[1] + p[1] * ax[0]; d0 = Math.min(d0, d); d1 = Math.max(d1, d); }
  return d1 - d0;
}
export function insetRing(r, d) {
  const n = r.length, out = [];
  for (let i = 0; i < n; i++) {
    const p0 = r[(i + n - 1) % n], p1 = r[i], p2 = r[(i + 1) % n];
    const ax = p1[0] - p0[0], az = p1[1] - p0[1], la = Math.hypot(ax, az);
    const bx = p2[0] - p1[0], bz = p2[1] - p1[1], lb = Math.hypot(bx, bz);
    if (la < 1e-6 || lb < 1e-6) return null;
    const n0x = -az / la, n0z = ax / la, n1x = -bz / lb, n1z = bx / lb;
    let mx = n0x + n1x, mz = n0z + n1z; const ml = Math.hypot(mx, mz);
    if (ml < 1e-3) { mx = n0x; mz = n0z; } else { mx /= ml; mz /= ml; }
    const cs = Math.max(0.35, mx * n0x + mz * n0z);
    const k = Math.min(d / cs, d * 2.5, la * 0.45, lb * 0.45);
    out.push([p1[0] + mx * k, p1[1] + mz * k]);
  }
  return out;
}
/** Outward offset of a positive-area ring by d (mitred, clamped at sharp corners). */
export function offsetRing(r, d) {
  const n = r.length, out = [];
  for (let i = 0; i < n; i++) {
    const p0 = r[(i + n - 1) % n], p1 = r[i], p2 = r[(i + 1) % n];
    const ax = p1[0] - p0[0], az = p1[1] - p0[1], la = Math.hypot(ax, az) || 1e-6;
    const bx = p2[0] - p1[0], bz = p2[1] - p1[1], lb = Math.hypot(bx, bz) || 1e-6;
    const n0x = az / la, n0z = -ax / la, n1x = bz / lb, n1z = -bx / lb; // outward normals
    let mx = n0x + n1x, mz = n0z + n1z; const ml = Math.hypot(mx, mz);
    if (ml < 1e-3) { mx = n0x; mz = n0z; } else { mx /= ml; mz /= ml; }
    const cs = Math.max(0.4, mx * n0x + mz * n0z);
    const k = Math.min(d / cs, d * 2.5);
    out.push([p1[0] + mx * k, p1[1] + mz * k]);
  }
  return out;
}
function insertRidgePoints(r, c, nx, nz) {
  const out = [];
  for (let i = 0; i < r.length; i++) {
    const a = r[i], b = r[(i + 1) % r.length];
    out.push(a);
    const da = (a[0] - c[0]) * nx + (a[1] - c[1]) * nz, db = (b[0] - c[0]) * nx + (b[1] - c[1]) * nz;
    if ((da > 1e-6 && db < -1e-6) || (da < -1e-6 && db > 1e-6)) {
      const t = da / (da - db);
      out.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]);
    }
  }
  return out;
}
function splitByRidge(ring, c, ax) {
  const nx = -ax[1], nz = ax[0];
  const side = (p) => (p[0] - c[0]) * nx + (p[1] - c[1]) * nz;
  const A = ring.filter((p) => side(p) >= -1e-6), B = ring.filter((p) => side(p) <= 1e-6);
  return [A, B].filter((p) => p.length >= 3 && Math.abs(signedArea(p)) > 0.1);
}
