// Procedural tree / shrub / grass geometry for the near (full-detail) LOD. Each model is one BufferGeometry
// (bark tubes + alpha-tested leaf cards share the vegetation atlas, so a species is ONE instanced draw call).
// Attributes: position, normal, uv, color (baked AO), veg = (windWeight m, leafFlag, cardRandom, flutter).
// Leaf-card normals are "crown normals": blended toward the crown-ellipsoid gradient so a crown of cards
// shades like one soft volume (the shader does not flip them on back faces).
import * as THREE from 'three';
import { mulberry32 } from './geom.js';
import { vegUV } from './VegTextures.js';

const V = (x = 0, y = 0, z = 0) => new THREE.Vector3(x, y, z);
const UP = V(0, 1, 0);
const DEG = Math.PI / 180;
const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
const smooth = (a, b, x) => { const t = clamp01((x - a) / (b - a)); return t * t * (3 - 2 * t); };

class GB {
  constructor() { this.P = []; this.N = []; this.U = []; this.C = []; this.W = []; this.I = []; this.n = 0; }
  v(px, py, pz, nx, ny, nz, u, v, ao, w, leaf, rnd, fl) {
    this.P.push(px, py, pz); this.N.push(nx, ny, nz); this.U.push(u, v); this.C.push(ao, ao, ao); this.W.push(w, leaf, rnd, fl);
    return this.n++;
  }
  tri(a, b, c) { this.I.push(a, b, c); }
  quad(a, b, c, d) { this.I.push(a, b, c, a, c, d); }
  geometry() {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(this.P, 3));
    g.setAttribute('normal', new THREE.Float32BufferAttribute(this.N, 3));
    g.setAttribute('uv', new THREE.Float32BufferAttribute(this.U, 2));
    g.setAttribute('color', new THREE.Float32BufferAttribute(this.C, 3));
    g.setAttribute('veg', new THREE.Float32BufferAttribute(this.W, 4));
    g.setIndex(this.I);
    g.computeBoundingBox(); g.computeBoundingSphere();
    return g;
  }
}

/** Generalised cylinder along pts (parallel-transport frames), bark UVs (u around, v along length). */
function tube(gb, pts, radii, sides, uv, o = {}) {
  const n = pts.length, T = [], N = [], B = [];
  for (let k = 0; k < n; k++) T.push(pts[Math.min(n - 1, k + 1)].clone().sub(pts[Math.max(0, k - 1)]).normalize());
  let nr = Math.abs(T[0].y) < 0.9 ? V(0, 1, 0).cross(T[0]).normalize() : V(1, 0, 0).cross(T[0]).normalize();
  for (let k = 0; k < n; k++) {
    if (k > 0) nr = nr.clone().sub(T[k].clone().multiplyScalar(nr.dot(T[k]))).normalize();
    N.push(nr.clone()); B.push(T[k].clone().cross(nr).normalize());
  }
  const base = gb.n, vOff = o.vOff ?? 0, vScale = o.vScale ?? 9, uSpan = o.uSpan ?? 1;
  let len = 0;
  for (let k = 0; k < n; k++) {
    if (k > 0) len += pts[k].distanceTo(pts[k - 1]);
    const v = uv.v0 + clamp01(vOff + len / vScale) * (uv.v1 - uv.v0);
    const p = pts[k], r = radii[k], ao = o.ao ? o.ao(p.y, k / (n - 1)) : 1;
    for (let j = 0; j <= sides; j++) {
      const a = (j / sides) * Math.PI * 2, c = Math.cos(a), s = Math.sin(a);
      const dx = N[k].x * c + B[k].x * s, dy = N[k].y * c + B[k].y * s, dz = N[k].z * c + B[k].z * s;
      gb.v(p.x + dx * r, p.y + dy * r, p.z + dz * r, dx, dy, dz, uv.u0 + (j / sides) * uSpan * (uv.u1 - uv.u0), v, ao, 0, 0, 0, 0);
    }
  }
  for (let k = 0; k < n - 1; k++) for (let j = 0; j < sides; j++) {
    const a = base + k * (sides + 1) + j, b = a + 1, c = a + sides + 1, d = c + 1;
    gb.tri(a, b, c); gb.tri(b, d, c);
  }
}

/**
 * Bent (V-folded) card: 6 verts / 4 tris. c centre, up/side unit vectors, hw/hh half sizes. Texture "stem"
 * end (region bottom, v1) is at -up. p: { C, Rad, sph, upMix, ao(pos), wind(pos), rnd, flutter, normal? }
 */
function card(gb, c, up, side, hw, hh, fold, uv, p) {
  const nrm = side.clone().cross(up).normalize();
  const base = gb.n, pos = V(), g = V(), cn = V(), n = V();
  for (let row = 0; row < 2; row++) for (let col = 0; col < 3; col++) {
    const sx = col - 1, sy = row * 2 - 1;
    pos.copy(c).addScaledVector(up, hh * sy).addScaledVector(side, hw * sx);
    if (col === 1) pos.addScaledVector(nrm, fold * hw);
    if (p.normal) p.normal(pos, nrm, n);
    else {
      g.copy(pos).sub(p.C); g.x /= p.Rad.x * p.Rad.x; g.y /= p.Rad.y * p.Rad.y; g.z /= p.Rad.z * p.Rad.z;
      if (g.lengthSq() < 1e-8) g.set(0, 1, 0);
      g.normalize();
      cn.copy(nrm); if (cn.dot(g) < 0) cn.negate();
      n.copy(g).multiplyScalar(p.sph).addScaledVector(cn, 1 - p.sph);
      if (p.upMix) n.addScaledVector(UP, p.upMix);
      n.normalize();
    }
    const u = sx < 0 ? uv.u0 : sx > 0 ? uv.u1 : (uv.u0 + uv.u1) * 0.5;
    const v = sy < 0 ? uv.v1 : uv.v0;
    gb.v(pos.x, pos.y, pos.z, n.x, n.y, n.z, u, v, p.ao(pos), p.wind(pos), 1, p.rnd, p.flutter ?? 1);
  }
  gb.quad(base, base + 1, base + 4, base + 3);
  gb.quad(base + 1, base + 2, base + 5, base + 4);
}

function randUnitPerp(up, rnd) {
  const ref = Math.abs(up.y) < 0.95 ? UP : V(1, 0, 0);
  const s0 = ref.clone().cross(up).normalize(), s1 = up.clone().cross(s0);
  const a = rnd() * Math.PI * 2;
  return s0.multiplyScalar(Math.cos(a)).add(s1.multiplyScalar(Math.sin(a)));
}

function kmeans(points, k, keyFn) {
  k = Math.max(1, Math.min(k, points.length));
  const s = points.slice().sort((a, b) => keyFn(a) - keyFn(b));
  const cents = Array.from({ length: k }, (_, i) => s[Math.floor(((i + 0.5) * s.length) / k)].clone());
  const asg = new Array(points.length).fill(0);
  for (let it = 0; it < 8; it++) {
    points.forEach((p, j) => { let b = 0, bd = Infinity; cents.forEach((c, i) => { const d = c.distanceToSquared(p); if (d < bd) { bd = d; b = i; } }); asg[j] = b; });
    cents.forEach((c, i) => { const m = points.filter((_, j) => asg[j] === i); if (m.length) { c.set(0, 0, 0); m.forEach((p) => c.add(p)); c.multiplyScalar(1 / m.length); } });
  }
  return cents.map((c, i) => ({ c, members: points.filter((_, j) => asg[j] === i) })).filter((g) => g.members.length);
}

// ------------------------------------------------------------------ broadleaf crowns (London plane, ficus, oak, Monterey cypress)
function broadleaf(o) {
  const rnd = mulberry32(o.seed);
  const gb = new GB();
  const bark = vegUV(o.bark), leaf = vegUV(o.leaf);
  const C = V((rnd() - 0.5) * o.rx * 0.1, o.cy, (rnd() - 0.5) * o.rz * 0.1);
  const Rad = V(o.rx, o.ry, o.rz);
  const top = o.cy + o.ry, crownBottom = o.cy - o.ry;
  // 1. leaf-cluster anchors on a lumpy ellipsoid shell
  const anchors = [], ph = [rnd() * 6.28, rnd() * 6.28, rnd() * 6.28];
  let minD = o.card * 0.52, tries = 0;
  while (anchors.length < o.cards && tries < 40000) {
    tries++; if (tries % 3000 === 0) minD *= 0.92;
    const u = rnd() * 2 - 1, th = rnd() * Math.PI * 2, s = Math.sqrt(1 - u * u);
    const d = V(s * Math.cos(th), u, s * Math.sin(th));
    if (d.y < -0.45 && rnd() < 0.65) continue;
    const lump = 1 + 0.1 * Math.sin(3 * th + ph[0]) + 0.07 * Math.sin(2 * th + 3 * u + ph[1]) + 0.05 * Math.sin(5 * u + ph[2]);
    const sh = o.shell ?? 0.6;
    const r = (sh + (1 - sh) * Math.sqrt(rnd())) * lump;
    const q = V(C.x + d.x * Rad.x * r, C.y + d.y * Rad.y * r, C.z + d.z * Rad.z * r);
    if (o.flatTop && q.y > top - o.card * 0.5) q.y = top - o.card * 0.5 - rnd() * 0.5;
    if (q.y < o.trunkH + o.card * 0.4) continue;
    let ok = true;
    for (const a of anchors) if (a.distanceToSquared(q) < minD * minD) { ok = false; break; }
    if (ok) anchors.push(q);
  }
  // 2. trunk with root flare (buried 0.6 m so slopes never show a gap)
  const tp = [], tr = [];
  const lean = V((rnd() - 0.5) * 0.3, 0, (rnd() - 0.5) * 0.3);
  const tN = 5, tTop = o.trunkH + 0.35;
  for (let k = 0; k <= tN; k++) {
    const t = k / tN, y = -0.6 + (tTop + 0.6) * t;
    tp.push(V(lean.x * t * t + Math.sin(t * 3 + ph[0]) * 0.03, y, lean.z * t * t + Math.cos(t * 3 + ph[1]) * 0.03));
    tr.push(o.trunkR * (1 - 0.2 * t) * (y < 0.35 ? 1 + 0.5 * Math.pow((0.35 - y) / 0.95, 1.5) : 1));
  }
  tube(gb, tp, tr, o.trunkSides ?? 7, bark, { vScale: 9, vOff: rnd() * 0.15, ao: (y) => 0.7 + 0.3 * smooth(-0.3, 1.6, y) });
  const fork = tp[tN].clone();
  // 3. scaffold limbs toward anchor clusters, twigs toward sub-clusters, leaf cards on the anchors
  const groups = kmeans(anchors, o.limbs, (p) => Math.atan2(p.z - fork.z, p.x - fork.x));
  const r0 = o.trunkR * (o.limbR ?? 0.6);
  const aoFn = (p) => {
    const dx = (p.x - C.x) / Rad.x, dy = (p.y - C.y) / Rad.y, dz = (p.z - C.z) / Rad.z;
    const rn = Math.sqrt(dx * dx + dy * dy + dz * dz);
    return (0.6 + 0.4 * smooth(0.15, 1.0, rn)) * (0.8 + 0.2 * clamp01((p.y - crownBottom) / (top - crownBottom)));
  };
  const windFn = (p) => o.wind * Math.pow(clamp01((p.y - o.trunkH * 0.8) / (top - o.trunkH * 0.8)), 1.3);
  const cp = { C, Rad, sph: o.sph ?? 0.78, upMix: o.upMix ?? 0.12, ao: aoFn, wind: windFn };
  for (const g of groups) {
    const S = fork.clone().add(V(0, -0.3 - rnd() * 0.25, 0));
    const dir = g.c.clone().sub(S);
    const L = dir.length() * (o.reach ?? 0.72);
    dir.normalize();
    const E = S.clone().addScaledVector(dir, L);
    const j1 = randUnitPerp(dir, rnd).multiplyScalar(L * 0.06), j2 = randUnitPerp(dir, rnd).multiplyScalar(L * 0.05);
    const pts = [S, S.clone().lerp(E, 0.36).add(V(0, L * 0.07, 0)).add(j1), S.clone().lerp(E, 0.7).add(V(0, L * 0.05, 0)).add(j2), E];
    tube(gb, pts, [r0, r0 * 0.74, r0 * 0.5, r0 * 0.28], o.limbSides ?? 5, bark, { vScale: 9, vOff: rnd() * 0.3, uSpan: 0.6, ao: () => 0.68 });
    const subs = kmeans(g.members, Math.min(o.twigs, g.members.length), (p) => p.y + p.x * 0.3);
    for (const sg of subs) {
      // attach on the limb at the point nearest the sub-cluster (t >= 0.45)
      let best = pts[3], bd = Infinity;
      for (let t = 0.45; t <= 1.001; t += 0.11) {
        const seg = t * 3, i = Math.min(2, Math.floor(seg)), f = seg - i;
        const q = pts[i].clone().lerp(pts[i + 1], f);
        const d = q.distanceToSquared(sg.c);
        if (d < bd) { bd = d; best = q; }
      }
      const tE = best.clone().lerp(sg.c, 0.75);
      tube(gb, [best, tE], [r0 * 0.28, r0 * 0.08], 3, bark, { vScale: 9, vOff: rnd() * 0.5, uSpan: 0.3, ao: () => 0.6 });
      for (const a of sg.members) {
        const stem = a.clone().sub(tE); if (stem.lengthSq() < 1e-4) stem.set(0, 1, 0); stem.normalize();
        const out = a.clone().sub(C).divide(Rad).normalize();
        let up, side;
        if (o.padTilt && rnd() < (o.padFrac ?? 0.7)) {
          // layered, near-horizontal foliage pad (Monterey cypress): long axis horizontal & outward
          const h = V(out.x, 0, out.z); if (h.lengthSq() < 1e-4) h.set(1, 0, 0); h.normalize();
          const ya = (rnd() - 0.5) * 1.2, c = Math.cos(ya), sn = Math.sin(ya);
          const hx = h.x * c - h.z * sn, hz = h.x * sn + h.z * c;
          const tilt = (o.padTilt[0] + rnd() * (o.padTilt[1] - o.padTilt[0])) * DEG;
          up = V(hx * Math.cos(tilt), Math.sin(tilt), hz * Math.cos(tilt));
          side = V(-hz, 0, hx);
        } else {
          up = stem.multiplyScalar(0.5).add(out.multiplyScalar(0.5)).add(V(0, 0.3, 0)).normalize();
          side = randUnitPerp(up, rnd);
        }
        const hh = o.card * 0.5 * (0.85 + rnd() * 0.3), hw = hh * (0.9 + rnd() * 0.2);
        const c = a.clone().addScaledVector(up, hh * 0.25);
        card(gb, c, up, side, hw, hh, o.fold ?? 0.32, leaf, { ...cp, rnd: rnd(), flutter: 1 });
      }
    }
  }
  const geo = gb.geometry();
  return { geo, tris: gb.I.length / 3, height: top, trunkR: o.trunkR, colH: o.trunkH + 1.2 };
}

// ------------------------------------------------------------------ Canary Island date palm
function palm(o) {
  const rnd = mulberry32(o.seed);
  const gb = new GB();
  const bark = vegUV('barkPalm'), fr = vegUV('frond');
  const H = o.height;
  const tp = [], tr = [];
  const bend = V((rnd() - 0.5) * 0.5, 0, (rnd() - 0.5) * 0.5);
  const N = 8;
  for (let k = 0; k <= N; k++) {
    const t = k / N, y = -0.6 + (H + 0.6) * t;
    tp.push(V(bend.x * Math.sin(t * Math.PI * 0.85), y, bend.z * Math.sin(t * Math.PI * 0.85)));
    const flare = y < 0.6 ? 0.2 * Math.pow((0.6 - y) / 1.2, 1.4) : 0;
    tr.push(o.trunkR * (1.08 - 0.12 * t + flare));
  }
  const tt = tp[N];
  // "pineapple" crown knob of trimmed leaf bases
  for (const [dy, r] of [[0.35, 1.22], [0.85, 1.32], [1.35, 1.12], [1.7, 0.65]]) {
    tp.push(V(tt.x, H + dy, tt.z)); tr.push(o.trunkR * r);
  }
  tube(gb, tp, tr, 8, bark, { vScale: H + 2.5, ao: (y) => 0.68 + 0.32 * smooth(-0.3, 2.2, y) * (y > H + 0.2 ? 0.82 : 1) });
  const crownC = V(tt.x, H + 1.4, tt.z);
  const nF = o.fronds;
  const Rad = V(4.2, 3.2, 4.2);
  const cy0 = crownC.y;
  for (let i = 0; i < nF; i++) {
    const f = i / (nF - 1);
    const az = i * 2.39996 + rnd() * 0.25;
    const elev = (66 - 96 * Math.pow(f, 0.8)) * DEG + (rnd() - 0.5) * 0.14;
    const L = o.frondL * (0.86 + 0.2 * Math.sin(Math.PI * Math.min(1, f * 1.1 + 0.15))) * (0.92 + rnd() * 0.16);
    const dh = V(Math.sin(az), 0, Math.cos(az));
    const start = crownC.clone().addScaledVector(dh, 0.3 + 0.2 * f).add(V(0, 0.45 - 0.75 * f, 0));
    const droop = 0.1 + 0.42 * f;
    const K = 4, rows = [];
    for (let k = 0; k <= K; k++) {
      const t = k / K;
      rows.push(start.clone().addScaledVector(dh, L * t * Math.cos(elev)).add(V(0, L * t * Math.sin(elev) - droop * L * t * t * 0.6, 0)));
    }
    const hw = o.frondW * 0.5, beta = (24 + rnd() * 10) * DEG, fr0 = rnd();
    const base = gb.n;
    for (let k = 0; k <= K; k++) {
      const t = k / K, p = rows[k];
      const tan = rows[Math.min(K, k + 1)].clone().sub(rows[Math.max(0, k - 1)]).normalize();
      const s = tan.clone().cross(UP); if (s.lengthSq() < 1e-6) s.set(1, 0, 0); s.normalize();
      const lu = s.clone().cross(tan).normalize(); if (lu.y < 0) lu.negate();
      const w = hw * (k === 0 ? 0.35 : 1);
      const pts = [
        p.clone().addScaledVector(s, w * Math.cos(beta)).addScaledVector(lu, w * Math.sin(beta)),
        p,
        p.clone().addScaledVector(s, -w * Math.cos(beta)).addScaledVector(lu, w * Math.sin(beta)),
      ];
      const u = fr.u0 + t * (fr.u1 - fr.u0);
      const vs = [fr.v0, (fr.v0 + fr.v1) * 0.5, fr.v1];
      for (let j = 0; j < 3; j++) {
        const q = pts[j];
        const g = q.clone().sub(crownC).divide(Rad).normalize();
        const n = g.multiplyScalar(0.55).addScaledVector(lu, 0.45).normalize();
        const ao = (0.62 + 0.38 * smooth(0, 0.6, t)) * (1 - 0.18 * f) * (q.y < cy0 - 1.5 ? 0.85 : 1);
        gb.v(q.x, q.y, q.z, n.x, n.y, n.z, u, vs[j], ao, o.wind * Math.pow(t, 1.5), 1, fr0, 0.4 + t);
      }
    }
    for (let k = 0; k < K; k++) {
      const a = base + k * 3;
      gb.quad(a, a + 3, a + 4, a + 1);
      gb.quad(a + 1, a + 4, a + 5, a + 2);
    }
  }
  const geo = gb.geometry();
  return { geo, tris: gb.I.length / 3, height: H + 4, trunkR: o.trunkR, colH: H + 1.2 };
}

// ------------------------------------------------------------------ pine (whorled, layered needle pads)
function pine(o) {
  const rnd = mulberry32(o.seed);
  const gb = new GB();
  const bark = vegUV('barkDark'), lf = vegUV('pine');
  const H = o.height;
  const tp = [], tr = [];
  for (let k = 0; k <= 6; k++) {
    const t = k / 6, y = -0.6 + (H + 0.6) * t;
    tp.push(V(Math.sin(t * 2.5) * 0.15, y, Math.cos(t * 2.1) * 0.1 - 0.1));
    tr.push(o.trunkR * Math.max(0.12, 1 - 0.85 * t) * (y < 0.4 ? 1 + 0.45 * Math.pow((0.4 - y) / 1.0, 1.5) : 1));
  }
  tube(gb, tp, tr, 8, bark, { vScale: H + 1, ao: (y) => 0.65 + 0.3 * smooth(-0.3, 1.5, y) });
  const trunkAt = (y) => { const t = clamp01((y + 0.6) / (H + 0.6)) * 6, i = Math.min(5, Math.floor(t)); return tp[i].clone().lerp(tp[i + 1], t - i); };
  const y0 = o.crownBase, C = V(0, (y0 + H) / 2, 0), Rad = V(o.spread + 0.8, (H - y0) / 2 + 1, o.spread + 0.8);
  const aoFn = (p) => (0.55 + 0.45 * smooth(y0, H, p.y)) * (0.8 + 0.2 * smooth(0.2, 1, Math.hypot(p.x / Rad.x, p.z / Rad.z)));
  const windFn = (p) => o.wind * Math.pow(clamp01((p.y - y0 * 0.7) / (H - y0 * 0.7)), 1.3);
  const cp = { C, Rad, sph: 0.65, upMix: 0.2, ao: aoFn, wind: windFn };
  let az = rnd() * 6.28;
  for (let y = y0; y < H - 0.9; y += o.whorl * (0.85 + rnd() * 0.3)) {
    const f = (y - y0) / (H - y0);
    const nB = f > 0.8 ? 3 : 4;
    for (let b = 0; b < nB; b++) {
      az += 2.39996 + (rnd() - 0.5) * 0.4;
      const Lb = (1.0 + o.spread * Math.max(0.15, 1 - ((f - 0.3) / 0.75) ** 2)) * (0.85 + rnd() * 0.3);
      const upA = (4 + 20 * f + (rnd() - 0.5) * 10) * DEG;
      const dh = V(Math.sin(az), 0, Math.cos(az));
      const dir = dh.clone().multiplyScalar(Math.cos(upA)).add(V(0, Math.sin(upA), 0));
      const S = trunkAt(y);
      const E = S.clone().addScaledVector(dir, Lb);
      tube(gb, [S, E], [0.09 * (1 - f * 0.5), 0.03], 3, bark, { vScale: 12, vOff: rnd() * 0.5, uSpan: 0.3, ao: () => 0.6 });
      const side = UP.clone().cross(dh).normalize();
      // main pad lying along the branch (covers it), slightly tilted up
      const tilt = (8 + rnd() * 16) * DEG;
      const up = dh.clone().multiplyScalar(Math.cos(tilt)).add(V(0, Math.sin(tilt), 0)).normalize();
      const hh = Math.max(0.8, Lb * 0.55);
      card(gb, S.clone().lerp(E, 0.55).add(V(0, 0.12, 0)), up, side, Math.min(2.0, 1.0 + Lb * 0.28), hh, 0.32, lf, { ...cp, rnd: rnd(), flutter: 0.6 });
      // drooping under-pad, yawed off the branch, for thickness
      const ya = (rnd() < 0.5 ? -1 : 1) * (0.3 + rnd() * 0.3);
      const dh2 = V(Math.sin(az + ya), 0, Math.cos(az + ya));
      const up3 = dh2.clone().multiplyScalar(Math.cos(-0.12)).add(V(0, Math.sin(-0.12), 0)).normalize();
      card(gb, S.clone().lerp(E, 0.68).add(V(0, -0.3, 0)), up3, UP.clone().cross(dh2).normalize(), 0.9 + rnd() * 0.3, Math.max(0.6, Lb * 0.4), 0.3, lf, { ...cp, rnd: rnd(), flutter: 0.6 });
      // tip tuft, more upright
      const t2 = (35 + rnd() * 25) * DEG;
      const up2 = dh.clone().multiplyScalar(Math.cos(t2)).add(V(0, Math.sin(t2), 0)).normalize();
      card(gb, E.clone().addScaledVector(up2, 0.3), up2, randUnitPerp(up2, rnd), 0.65 + rnd() * 0.25, 0.6 + rnd() * 0.2, 0.3, lf, { ...cp, rnd: rnd(), flutter: 0.8 });
    }
  }
  // leader tuft
  for (let m = 0; m < 2; m++) {
    const a = m * Math.PI * 0.5 + rnd() * 0.3;
    card(gb, V(tp[6].x, H - 0.2, tp[6].z), UP.clone(), V(Math.cos(a), 0, Math.sin(a)), 0.7, 1.0, 0.25, lf, { ...cp, rnd: rnd(), flutter: 0.8 });
  }
  const geo = gb.geometry();
  return { geo, tris: gb.I.length / 3, height: H + 0.8, trunkR: o.trunkR, colH: H * 0.85 };
}

// ------------------------------------------------------------------ shrub & grass tuft
function shrub(o) {
  const rnd = mulberry32(o.seed);
  const gb = new GB();
  const lf = vegUV('shrub');
  const C = V(0, 0.42, 0), Rad = V(0.9, 0.62, 0.9);
  const cp = { C, Rad, sph: 0.85, upMix: 0.15, ao: (p) => 0.5 + 0.5 * smooth(-0.1, 1.0, p.y), wind: (p) => 0.05 * clamp01(p.y / 1.1) };
  for (let i = 0; i < o.cards; i++) {
    const az = (i / o.cards) * Math.PI * 2 + rnd() * 0.5;
    const el = (i % 3 === 0 ? 55 : 15) * DEG + rnd() * 0.3;
    const d = V(Math.sin(az) * Math.cos(el), Math.sin(el), Math.cos(az) * Math.cos(el));
    const c = C.clone().add(V(d.x * Rad.x * 0.45, d.y * Rad.y * 0.4, d.z * Rad.z * 0.45));
    const up = d.clone().add(V(0, 0.6, 0)).normalize();
    card(gb, c, up, randUnitPerp(up, rnd), 0.55 + rnd() * 0.15, 0.5 + rnd() * 0.12, 0.3, lf, { ...cp, rnd: rnd(), flutter: 0.6 });
  }
  return { geo: gb.geometry(), tris: gb.I.length / 3, height: 1.1, trunkR: 0, colH: 0 };
}

function grassTuft() {
  const gb = new GB();
  const lf = vegUV('grass');
  const nup = (pos, nrm, out) => out.copy(nrm).multiplyScalar(0.25).add(V(0, 1, 0)).normalize();
  for (let i = 0; i < 3; i++) {
    const a = (i / 3) * Math.PI;
    const side = V(Math.cos(a), 0, Math.sin(a));
    card(gb, V(0, 0.24, 0), UP.clone(), side, 0.45, 0.3, 0, lf, {
      normal: nup, ao: (p) => 0.55 + 0.45 * clamp01(p.y / 0.5), wind: (p) => 0.07 * clamp01(p.y / 0.5), rnd: i / 3, flutter: 1,
    });
  }
  return { geo: gb.geometry(), tris: gb.I.length / 3, height: 0.55, trunkR: 0, colH: 0 };
}

// ------------------------------------------------------------------ registry
export const SPECIES = ['planeA', 'planeB', 'round', 'oak', 'palm', 'cypress', 'pine'];

let _cache = null;
/** Build (once) every model: { planeA, planeB, round, oak, palm, cypress, pine, shrub, grass } -> {geo, tris, height, trunkR, colH} */
export function getVegModels() {
  if (_cache) return _cache;
  _cache = {
    planeA: broadleaf({ seed: 11, trunkH: 3.1, trunkR: 0.27, cy: 7.7, rx: 4.6, ry: 4.4, rz: 4.4, cards: 70, card: 2.8, limbs: 4, twigs: 4, reach: 0.72, leaf: 'plane', bark: 'barkPlane', wind: 0.3, shell: 0.5 }),
    planeB: broadleaf({ seed: 23, trunkH: 3.5, trunkR: 0.25, cy: 8.7, rx: 3.9, ry: 5.0, rz: 4.1, cards: 72, card: 2.7, limbs: 3, twigs: 5, reach: 0.7, leaf: 'plane', bark: 'barkPlane', wind: 0.32, shell: 0.5 }),
    round: broadleaf({ seed: 37, trunkH: 2.2, trunkR: 0.2, cy: 5.3, rx: 3.3, ry: 3.1, rz: 3.3, cards: 64, card: 2.3, limbs: 4, twigs: 3, reach: 0.7, leaf: 'small', bark: 'barkDark', wind: 0.22, shell: 0.4 }),
    oak: broadleaf({ seed: 41, trunkH: 1.9, trunkR: 0.36, cy: 5.6, rx: 6.0, ry: 3.6, rz: 5.6, cards: 66, card: 3.0, limbs: 5, twigs: 3, reach: 0.8, leaf: 'small', bark: 'barkDark', wind: 0.24, flatTop: true, shell: 0.45 }),
    palm: palm({ seed: 53, height: 10.5, trunkR: 0.46, fronds: 42, frondL: 4.9, frondW: 1.7, wind: 0.45 }),
    cypress: broadleaf({ seed: 61, trunkH: 2.3, trunkR: 0.5, cy: 8.4, rx: 6.8, ry: 4.6, rz: 6.2, cards: 72, card: 3.2, limbs: 5, twigs: 3, reach: 0.78, limbR: 0.66, leaf: 'cypress', bark: 'barkDark', wind: 0.22, flatTop: true, shell: 0.45, padTilt: [8, 34], padFrac: 0.5, upMix: 0.3, sph: 0.65, fold: 0.3 }),
    pine: pine({ seed: 71, height: 16.5, trunkR: 0.38, crownBase: 5.0, spread: 3.4, whorl: 1.5, wind: 0.3 }),
    shrub: shrub({ seed: 83, cards: 10 }),
    grass: grassTuft(),
  };
  return _cache;
}
