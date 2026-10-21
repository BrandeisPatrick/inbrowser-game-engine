// Procedural vegetation texture atlas (one 2048² RGBA DataTexture shared by every near tree, shrub and grass
// tuft): leaf-cluster cards (London plane palmate leaves, small elliptic leaves), Monterey-cypress sprays,
// pine needle tufts, shrub leaves, grass blades, a Canary-palm frond, and three bark strips.
// Painted with Canvas2D, then post-processed on the CPU: transparent texels get leaf-coloured RGB (no dark
// fringes), mipmaps are built in linear space with alpha-weighted colour and *coverage-preserving* alpha
// (per region) so alpha-tested foliage does not thin out / vanish with distance.
import * as THREE from 'three';
import { mulberry32 } from './geom.js';
import { fillTransparent, buildMipChain, dataTextureFromMips } from '../render/MipChain.js';

export const VEG_ATLAS_SIZE = 2048;
// Pixel rects (canvas space: y down). DataTexture is uploaded unflipped, so v = y / size (v = 0 is the top row).
export const VEG_REGIONS = {
  plane: { x: 16, y: 16, w: 736, h: 736, leaf: true },
  small: { x: 784, y: 16, w: 736, h: 736, leaf: true },
  cypress: { x: 16, y: 784, w: 736, h: 736, leaf: true },
  pine: { x: 784, y: 784, w: 736, h: 736, leaf: true },
  shrub: { x: 1552, y: 16, w: 480, h: 480, leaf: true },
  grass: { x: 1552, y: 528, w: 480, h: 480, leaf: true },
  frond: { x: 16, y: 1552, w: 1504, h: 480, leaf: true },
  barkPlane: { x: 1544, y: 1040, w: 152, h: 1000, bark: true },
  barkDark: { x: 1712, y: 1040, w: 136, h: 1000, bark: true },
  barkPalm: { x: 1864, y: 1040, w: 176, h: 1000, bark: true },
};

/** UV rect of a region: {u0,v0,u1,v1}; v0 is the region's top edge (canvas y), v1 its bottom. */
export function vegUV(name) {
  const r = VEG_REGIONS[name], S = VEG_ATLAS_SIZE;
  return { u0: r.x / S, v0: r.y / S, u1: (r.x + r.w) / S, v1: (r.y + r.h) / S };
}

const hsl = (h, s, l, a = 1) => `hsla(${h.toFixed(1)},${Math.max(0, s).toFixed(1)}%,${Math.max(0, Math.min(100, l)).toFixed(1)}%,${a})`;
const lerp = (a, b, t) => a + (b - a) * t;

// ------------------------------------------------------------------ leaf shapes (local: petiole at 0,0, tip at 0,-L)
function palmatePath(ctx, L, rnd) {
  const lobes = [
    { a: 0, r: 0.56 }, { a: 0.92 + (rnd() - 0.5) * 0.15, r: 0.5 }, { a: -0.92 + (rnd() - 0.5) * 0.15, r: 0.5 },
    { a: 1.8, r: 0.33 }, { a: -1.8, r: 0.33 },
  ];
  const k = 0.9 + rnd() * 0.2, rs = 0.25, cy = -0.47 * L, N = 44;
  for (let i = 0; i <= N; i++) {
    const th = -Math.PI + (2 * Math.PI * i) / N;
    let r = rs * 0.92;
    for (const lb of lobes) {
      let d = Math.abs(th - lb.a); if (d > Math.PI) d = 2 * Math.PI - d;
      const w = 0.44;
      if (d < w) r = Math.max(r, rs + (lb.r * k - rs) * Math.pow(1 - d / w, 1.35));
    }
    let db = Math.abs(Math.abs(th) - Math.PI);
    if (db < 0.4) r *= 0.55 + 0.45 * (db / 0.4); // heart-shaped base
    r *= 1 + (rnd() - 0.5) * 0.05; // fine teeth
    const x = Math.sin(th) * r * L, y = cy - Math.cos(th) * r * L;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.closePath();
}
function ovalPath(ctx, L, aspect) {
  const w = aspect * L * 0.75;
  ctx.moveTo(0, 0);
  ctx.bezierCurveTo(w, -0.14 * L, w * 1.05, -0.68 * L, 0, -L);
  ctx.bezierCurveTo(-w * 1.05, -0.68 * L, -w, -0.14 * L, 0, 0);
  ctx.closePath();
}

function drawLeaf(ctx, shape, x, y, ang, sz, h, s, l, rnd, aspect) {
  ctx.save();
  ctx.translate(x, y); ctx.rotate(ang); ctx.translate(0, sz * 0.5);
  ctx.strokeStyle = hsl(h - 12, s * 0.6, l * 0.75);
  ctx.lineWidth = Math.max(1, sz * 0.035);
  ctx.beginPath(); ctx.moveTo(0, sz * 0.14); ctx.lineTo(0, 0); ctx.stroke();
  ctx.beginPath();
  if (shape === 'palmate') palmatePath(ctx, sz, rnd); else ovalPath(ctx, sz, aspect);
  ctx.fillStyle = hsl(h, s, l);
  ctx.fill();
  ctx.save();
  ctx.clip();
  const side = rnd() < 0.5 ? -1 : 1;
  ctx.fillStyle = `rgba(10,20,0,${(0.06 + rnd() * 0.08).toFixed(3)})`;
  ctx.fillRect(side < 0 ? -sz : 0, -sz * 1.2, sz, sz * 1.3);
  const g = ctx.createLinearGradient(0, 0, 0, -sz);
  g.addColorStop(0, 'rgba(0,0,0,0.12)');
  g.addColorStop(1, `rgba(255,255,210,${(0.04 + rnd() * 0.08).toFixed(3)})`);
  ctx.fillStyle = g; ctx.fillRect(-sz, -sz * 1.2, 2 * sz, sz * 1.3);
  ctx.strokeStyle = hsl(h - 6, s * 0.55, Math.min(78, l + 13), 0.5);
  ctx.lineWidth = Math.max(0.7, sz * 0.022);
  ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(0, -sz * 0.9);
  if (shape === 'palmate') {
    for (const a of [0.92, -0.92, 1.8, -1.8]) {
      const r = Math.abs(a) > 1.5 ? 0.3 : 0.46;
      ctx.moveTo(0, -0.12 * sz); ctx.lineTo(Math.sin(a) * r * sz, -0.47 * sz - Math.cos(a) * r * sz);
    }
  }
  ctx.stroke();
  ctx.restore();
  ctx.restore();
}

// Irregular blob envelope
function envelope(ecx, ecy, erx, ery, rnd) {
  const ph = [rnd() * 6.28, rnd() * 6.28, rnd() * 6.28];
  const br = (a) => 1 + 0.1 * Math.sin(3 * a + ph[0]) + 0.07 * Math.sin(5 * a + ph[1]) + 0.05 * Math.sin(8 * a + ph[2]);
  return (x, y, m = 1) => {
    const dx = (x - ecx) / erx, dy = (y - ecy) / ery;
    const r = br(Math.atan2(dy, dx)) * m;
    return dx * dx + dy * dy <= r * r;
  };
}

function paintLeafCluster(ctx, R, rnd, o) {
  const cx = R.x + R.w * 0.5, baseY = R.y + R.h * 0.985;
  const ecx = cx, ecy = R.y + R.h * (o.ecy ?? 0.48), erx = R.w * (o.erx ?? 0.45), ery = R.h * (o.ery ?? 0.44);
  const inside = envelope(ecx, ecy, erx, ery, rnd);
  ctx.save();
  ctx.beginPath(); ctx.rect(R.x, R.y, R.w, R.h); ctx.clip();
  ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  // twigs: a stem from the bottom centre with side shoots into the envelope
  const top = [ecx + (rnd() - 0.5) * erx * 0.3, ecy - ery * 0.55];
  ctx.strokeStyle = hsl(28, 22, 20);
  ctx.lineWidth = o.twigW ?? 7;
  ctx.beginPath(); ctx.moveTo(cx, baseY); ctx.quadraticCurveTo(cx + (rnd() - 0.5) * 40, lerp(baseY, top[1], 0.5), top[0], top[1]); ctx.stroke();
  const nT = 7 + ((rnd() * 4) | 0);
  for (let k = 0; k < nT; k++) {
    const t = 0.12 + (0.8 * k) / nT;
    const sx = lerp(cx, top[0], t), sy = lerp(baseY, top[1], t);
    let ex = 0, ey = 0;
    for (let tries = 0; tries < 30; tries++) {
      ex = ecx + (rnd() * 2 - 1) * erx; ey = ecy + (rnd() * 2 - 1) * ery;
      if (inside(ex, ey, 0.85) && ey < sy) break;
    }
    ctx.lineWidth = (o.twigW ?? 7) * (0.35 + 0.3 * (1 - t));
    ctx.beginPath(); ctx.moveTo(sx, sy); ctx.quadraticCurveTo(lerp(sx, ex, 0.5) + (rnd() - 0.5) * 30, lerp(sy, ey, 0.5) - 20, ex, ey); ctx.stroke();
  }
  // leaves, back to front
  const L = [];
  for (let i = 0, tries = 0; i < o.count && tries < o.count * 30; tries++) {
    const x = ecx + (rnd() * 2 - 1) * erx * 1.1, y = ecy + (rnd() * 2 - 1) * ery * 1.1;
    if (!inside(x, y, 0.96)) continue;
    L.push({ x, y, d: rnd() });
    i++;
  }
  L.sort((a, b) => a.d - b.d);
  for (const f of L) {
    const ax = f.x - cx, ay = f.y - baseY, ox = f.x - ecx, oy = f.y - ecy;
    let ang = Math.atan2(ax * 0.6 + ox, -(ay * 0.6 + oy));
    ang += (rnd() - 0.5) * 1.2;
    const sz = o.size * (0.72 + rnd() * 0.56) * (0.85 + 0.3 * f.d);
    const yN = (f.y - (ecy - ery)) / (2 * ery);
    const h = lerp(o.hue[0], o.hue[1], rnd()), s = lerp(o.sat[0], o.sat[1], rnd());
    const l = lerp(o.light[0], o.light[1], f.d) * (1.1 - 0.2 * yN) + (rnd() - 0.5) * 5;
    drawLeaf(ctx, o.shape, f.x, f.y, ang, sz, h, s, l, rnd, o.aspect ?? 0.42);
  }
  ctx.restore();
}

function paintCypress(ctx, R, rnd) {
  const cx = R.x + R.w / 2, cy = R.y + R.h * 0.52;
  ctx.save(); ctx.beginPath(); ctx.rect(R.x, R.y, R.w, R.h); ctx.clip();
  ctx.lineCap = 'round';
  const clumps = [];
  for (let i = 0; i < 46; i++) {
    const a = rnd() * Math.PI * 2, r = Math.sqrt(rnd()) * 0.36;
    clumps.push({ x: cx + Math.cos(a) * r * R.w, y: cy + Math.sin(a) * r * R.h * 0.82, rad: R.w * (0.075 + rnd() * 0.07), d: rnd() });
  }
  clumps.sort((a, b) => a.d - b.d);
  // a few dark woody twigs underneath
  ctx.strokeStyle = hsl(25, 20, 16); ctx.lineWidth = 6;
  for (let i = 0; i < 6; i++) {
    const c = clumps[(rnd() * clumps.length) | 0];
    ctx.beginPath(); ctx.moveTo(cx, R.y + R.h * 0.97); ctx.quadraticCurveTo(lerp(cx, c.x, 0.5), lerp(R.y + R.h, c.y, 0.6), c.x, c.y); ctx.stroke();
  }
  for (const c of clumps) {
    for (let k = 0; k < 230; k++) {
      const a = rnd() * Math.PI * 2, rr = Math.pow(rnd(), 0.7) * c.rad;
      const px = c.x + Math.cos(a) * rr, py = c.y + Math.sin(a) * rr;
      const len = 8 + rnd() * 12, dir = a + (rnd() - 0.5) * 1.3;
      const edge = rr / c.rad;
      const topLit = py < c.y ? 1 : 0.55;
      const l = 13 + 15 * edge * topLit + c.d * 7 + (rnd() - 0.5) * 5;
      ctx.strokeStyle = hsl(98 + rnd() * 30, 30 + rnd() * 14, l);
      ctx.lineWidth = 3 + rnd() * 2.6;
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.quadraticCurveTo(px + Math.cos(dir) * len * 0.5 + (rnd() - 0.5) * 6, py + Math.sin(dir) * len * 0.5 + (rnd() - 0.5) * 6, px + Math.cos(dir) * len, py + Math.sin(dir) * len);
      ctx.stroke();
    }
  }
  ctx.restore();
}

function paintPine(ctx, R, rnd) {
  const bx = R.x + R.w / 2, by = R.y + R.h * 0.98;
  ctx.save(); ctx.beginPath(); ctx.rect(R.x, R.y, R.w, R.h); ctx.clip();
  ctx.lineCap = 'round';
  const tufts = [];
  const nT = 6;
  for (let t = 0; t < nT; t++) {
    const ang = -Math.PI / 2 + (t / (nT - 1) - 0.5) * 1.7 + (rnd() - 0.5) * 0.15;
    const len = R.h * (0.5 + rnd() * 0.3);
    const ex = bx + Math.cos(ang) * len, ey = by + Math.sin(ang) * len;
    ctx.strokeStyle = hsl(25, 25, 22); ctx.lineWidth = 7;
    ctx.beginPath(); ctx.moveTo(bx, by); ctx.lineTo(ex, ey); ctx.stroke();
    for (let k = 0; k < 6; k++) {
      const f = 0.28 + (0.72 * k) / 5;
      tufts.push({ x: lerp(bx, ex, f), y: lerp(by, ey, f), a: ang, d: rnd(), s: 0.75 + 0.35 * f });
    }
  }
  tufts.sort((a, b) => a.d - b.d);
  for (const tf of tufts) {
    for (let n = 0; n < 95; n++) {
      const a = tf.a + (rnd() - 0.5) * 2.4;
      const L = R.w * (0.1 + rnd() * 0.09) * tf.s;
      const ox = tf.x + (rnd() - 0.5) * 8, oy = tf.y + (rnd() - 0.5) * 8;
      const l = 19 + rnd() * 18 + tf.d * 8;
      ctx.strokeStyle = hsl(82 + rnd() * 26, 30 + rnd() * 20, l);
      ctx.lineWidth = 1.6 + rnd() * 1.1;
      ctx.beginPath(); ctx.moveTo(ox, oy);
      ctx.quadraticCurveTo(ox + Math.cos(a) * L * 0.5, oy + Math.sin(a) * L * 0.5 + 4, ox + Math.cos(a) * L, oy + Math.sin(a) * L);
      ctx.stroke();
    }
  }
  ctx.restore();
}

function paintGrass(ctx, R, rnd) {
  ctx.save(); ctx.beginPath(); ctx.rect(R.x, R.y, R.w, R.h); ctx.clip();
  const by = R.y + R.h;
  const blades = [];
  for (let i = 0; i < 190; i++) blades.push({ x: R.x + 14 + rnd() * (R.w - 28), d: rnd() });
  blades.sort((a, b) => a.d - b.d);
  for (const b of blades) {
    const hgt = R.h * (0.4 + rnd() * 0.56), lean = (rnd() - 0.5) * 0.55 * hgt, w = 4 + rnd() * 7;
    const dry = rnd() < 0.28;
    const h = dry ? 48 + rnd() * 12 : 75 + rnd() * 30, s = dry ? 35 + rnd() * 15 : 32 + rnd() * 22;
    const g = ctx.createLinearGradient(0, by, 0, by - hgt);
    g.addColorStop(0, hsl(h, s * 0.8, 13 + b.d * 5));
    g.addColorStop(1, hsl(h, s, (dry ? 46 : 34) + b.d * 9));
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.moveTo(b.x - w / 2, by + 2);
    ctx.quadraticCurveTo(b.x - w / 4 + lean * 0.35, by - hgt * 0.6, b.x + lean, by - hgt);
    ctx.quadraticCurveTo(b.x + w / 4 + lean * 0.35, by - hgt * 0.6, b.x + w / 2, by + 2);
    ctx.closePath(); ctx.fill();
  }
  ctx.restore();
}

function paintFrond(ctx, R, rnd) {
  const x0 = R.x + 6, x1 = R.x + R.w - 6, cy = R.y + R.h / 2, half = R.h / 2 - 5;
  ctx.save(); ctx.beginPath(); ctx.rect(R.x, R.y, R.w, R.h); ctx.clip();
  ctx.lineCap = 'round';
  const rach = (t) => [lerp(x0, x1, t), cy];
  const N = 125;
  const leaflets = [];
  for (const side of [-1, 1]) {
    for (let i = 0; i < N; i++) {
      const t = 0.03 + (0.96 * i) / N + (rnd() - 0.5) * 0.004;
      leaflets.push({ t, side, d: rnd() });
    }
  }
  leaflets.sort((a, b) => a.d - b.d);
  for (const lf of leaflets) {
    const { t, side } = lf;
    const [px, py] = rach(t);
    const spine = t < 0.13;
    const th = (spine ? 0.62 : 0.78) + (rnd() - 0.5) * 0.12;
    const lmax = half / Math.sin(th);
    const prof = spine ? 0.2 + t * 2.2 : 0.3 + 0.7 * Math.pow(Math.sin(Math.PI * Math.min(1, (t - 0.08) / 1.02)), 0.75);
    const len = lmax * prof * (0.9 + rnd() * 0.1);
    const dx = Math.cos(th), dy = side * Math.sin(th);
    const ex = px + dx * len, ey = py + dy * len;
    const wl = spine ? 2.2 : 4 + 4.5 * prof;
    const mx = px + dx * len * 0.45, my = py + dy * len * 0.45 + 3;
    const nx = -dy, ny = dx;
    const h = spine ? 52 + rnd() * 10 : 78 + rnd() * 18, s = spine ? 40 : 30 + rnd() * 14;
    const l = spine ? 42 + rnd() * 10 : 19 + rnd() * 12 + lf.d * 6;
    ctx.fillStyle = hsl(h, s, l);
    ctx.beginPath(); ctx.moveTo(px, py);
    ctx.quadraticCurveTo(mx + nx * wl, my + ny * wl, ex, ey);
    ctx.quadraticCurveTo(mx - nx * wl, my - ny * wl, px, py);
    ctx.fill();
    if (!spine) {
      ctx.strokeStyle = hsl(h - 8, s * 0.6, l + 12, 0.45); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(lerp(px, ex, 0.9), lerp(py, ey, 0.9)); ctx.stroke();
    }
  }
  for (let k = 0; k < 24; k++) { // rachis, tapering
    const t0 = k / 24, t1 = (k + 1) / 24;
    ctx.strokeStyle = hsl(58, 32, 44 - t0 * 8); ctx.lineWidth = 10 - 7 * t0;
    ctx.beginPath(); ctx.moveTo(...rach(t0)); ctx.lineTo(...rach(t1)); ctx.stroke();
  }
  ctx.restore();
}

// ------------------------------------------------------------------ barks (tile horizontally inside their strip)
function wrapDraw(R, x, fn) { fn(x); if (x - R.x < 40) fn(x + R.w); if (R.x + R.w - x < 40) fn(x - R.w); }

function paintBarkPlane(ctx, R, rnd) {
  ctx.save(); ctx.beginPath(); ctx.rect(R.x, R.y, R.w, R.h); ctx.clip();
  ctx.fillStyle = '#857d66'; ctx.fillRect(R.x, R.y, R.w, R.h);
  const pal = ['#b9b092', '#98936e', '#73695d', '#808366', '#a69574', '#5f5648', '#aea888', '#8a8a70'];
  for (let i = 0; i < 380; i++) {
    const x = R.x + rnd() * R.w, y = R.y + rnd() * R.h;
    const rx = 5 + rnd() * 20, ry = rx * (0.7 + rnd() * 1.1), col = pal[(rnd() * pal.length) | 0];
    const pts = [];
    for (let k = 0; k < 9; k++) { const a = (k / 9) * Math.PI * 2; const f = 0.65 + rnd() * 0.5; pts.push([Math.cos(a) * rx * f, Math.sin(a) * ry * f]); }
    wrapDraw(R, x, (xx) => {
      ctx.fillStyle = col; ctx.beginPath();
      pts.forEach(([px, py], k) => (k ? ctx.lineTo(xx + px, y + py) : ctx.moveTo(xx + px, y + py)));
      ctx.closePath(); ctx.fill();
    });
  }
  for (let i = 0; i < 4000; i++) {
    ctx.fillStyle = rnd() < 0.5 ? 'rgba(40,35,25,0.18)' : 'rgba(255,250,230,0.12)';
    ctx.fillRect(R.x + rnd() * R.w, R.y + rnd() * R.h, 1 + rnd() * 2, 1 + rnd() * 2);
  }
  ctx.restore();
}

function paintBarkDark(ctx, R, rnd) {
  ctx.save(); ctx.beginPath(); ctx.rect(R.x, R.y, R.w, R.h); ctx.clip();
  ctx.fillStyle = '#4a3f35'; ctx.fillRect(R.x, R.y, R.w, R.h);
  ctx.lineCap = 'round';
  for (let i = 0; i < 70; i++) {
    const x = R.x + rnd() * R.w, w = 3 + rnd() * 6, col = rnd() < 0.5 ? '#6e6052' : '#5b4e42';
    const pts = []; let xx = 0;
    for (let y = R.y - 10; y < R.y + R.h + 20; y += 18) { xx += (rnd() - 0.5) * 5; pts.push([xx, y]); }
    wrapDraw(R, x, (bx) => {
      ctx.strokeStyle = col; ctx.lineWidth = w; ctx.beginPath();
      pts.forEach(([px, py], k) => (k ? ctx.lineTo(bx + px, py) : ctx.moveTo(bx + px, py))); ctx.stroke();
    });
  }
  for (let i = 0; i < 40; i++) {
    const x = R.x + rnd() * R.w;
    const pts = []; let xx = 0;
    for (let y = R.y - 10; y < R.y + R.h + 20; y += 14) { xx += (rnd() - 0.5) * 4; pts.push([xx, y]); }
    wrapDraw(R, x, (bx) => {
      ctx.strokeStyle = '#261f19'; ctx.lineWidth = 1.5 + rnd() * 1.5; ctx.beginPath();
      pts.forEach(([px, py], k) => (k ? ctx.lineTo(bx + px, py) : ctx.moveTo(bx + px, py))); ctx.stroke();
    });
  }
  for (let i = 0; i < 260; i++) { // horizontal cracks
    const x = R.x + rnd() * R.w, y = R.y + rnd() * R.h, l = 4 + rnd() * 10;
    ctx.strokeStyle = 'rgba(25,20,15,0.6)'; ctx.lineWidth = 1.2;
    wrapDraw(R, x, (bx) => { ctx.beginPath(); ctx.moveTo(bx, y); ctx.lineTo(bx + l, y + (rnd() - 0.5) * 3); ctx.stroke(); });
  }
  ctx.restore();
}

function paintBarkPalm(ctx, R, rnd) {
  ctx.save(); ctx.beginPath(); ctx.rect(R.x, R.y, R.w, R.h); ctx.clip();
  ctx.fillStyle = '#5d5142'; ctx.fillRect(R.x, R.y, R.w, R.h);
  const cols = 9, dw = R.w / cols, dh = 15;
  for (let row = 0, y = R.y - dh; y < R.y + R.h + dh; row++, y += dh * 0.62) {
    for (let c = 0; c < cols; c++) {
      const x = R.x + (c + (row % 2) * 0.5) * dw;
      const k = 0.85 + rnd() * 0.3;
      wrapDraw(R, x, (xx) => {
        const g = ctx.createLinearGradient(0, y - dh * 0.55, 0, y + dh * 0.55);
        g.addColorStop(0, `rgb(${(160 * k) | 0},${(142 * k) | 0},${(114 * k) | 0})`);
        g.addColorStop(0.6, `rgb(${(118 * k) | 0},${(102 * k) | 0},${(80 * k) | 0})`);
        g.addColorStop(1, `rgb(${(70 * k) | 0},${(58 * k) | 0},${(45 * k) | 0})`);
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.moveTo(xx, y - dh * 0.55); ctx.lineTo(xx + dw * 0.52, y); ctx.lineTo(xx, y + dh * 0.55); ctx.lineTo(xx - dw * 0.52, y);
        ctx.closePath(); ctx.fill();
        ctx.strokeStyle = 'rgba(40,32,24,0.85)'; ctx.lineWidth = 1.4; ctx.stroke();
      });
    }
  }
  for (let i = 0; i < 1500; i++) { // fibres
    ctx.fillStyle = rnd() < 0.5 ? 'rgba(30,24,18,0.25)' : 'rgba(210,190,150,0.12)';
    ctx.fillRect(R.x + rnd() * R.w, R.y + rnd() * R.h, 1, 2 + rnd() * 5);
  }
  ctx.restore();
}

// CPU post-processing + mip chain: engine render/MipChain.js (fillTransparent, buildMipChain, dataTextureFromMips)

/** Paint + post-process the atlas. Returns { texture, levels } or null without a DOM (headless builds). */
export function makeVegAtlas(seed = 7) {
  if (typeof document === 'undefined') return null;
  const S = VEG_ATLAS_SIZE, R = VEG_REGIONS;
  const cv = document.createElement('canvas');
  cv.width = cv.height = S;
  const ctx = cv.getContext('2d', { willReadFrequently: true });
  const rnd = mulberry32(seed);
  // London plane: big palmate leaves, mid green
  paintLeafCluster(ctx, R.plane, rnd, { shape: 'palmate', count: 720, size: 66, hue: [78, 98], sat: [34, 48], light: [30, 43], twigW: 7, erx: 0.48, ery: 0.47 });
  // small elliptic leaves (ficus / Brisbane box / coast live oak): darker, glossier
  paintLeafCluster(ctx, R.small, rnd, { shape: 'oval', count: 2500, size: 34, aspect: 0.5, hue: [86, 112], sat: [32, 46], light: [25, 38], twigW: 5, erx: 0.48, ery: 0.47 });
  paintCypress(ctx, R.cypress, rnd);
  paintPine(ctx, R.pine, rnd);
  paintLeafCluster(ctx, R.shrub, rnd, { shape: 'oval', count: 700, size: 30, aspect: 0.55, hue: [84, 125], sat: [28, 46], light: [15, 33], twigW: 4, ecy: 0.5, erx: 0.47, ery: 0.46 });
  paintGrass(ctx, R.grass, rnd);
  paintFrond(ctx, R.frond, rnd);
  paintBarkPlane(ctx, R.barkPlane, rnd);
  paintBarkDark(ctx, R.barkDark, rnd);
  paintBarkPalm(ctx, R.barkPalm, rnd);

  const img = ctx.getImageData(0, 0, S, S);
  const d = new Uint8Array(img.data.buffer);
  // global: unpainted texels -> neutral dark green, alpha 0
  for (let i = 0; i < d.length; i += 4) if (d[i + 3] < 12) { d[i] = 52; d[i + 1] = 64; d[i + 2] = 34; d[i + 3] = 0; }
  for (const r of Object.values(R)) {
    if (r.leaf) fillTransparent(d, S, Math.max(0, r.x - 12), Math.max(0, r.y - 12), r.w + 24, r.h + 24);
  }
  // bark strips: opaque; horizontal gutters wrap (seam-free u), vertical gutters clamp
  for (const r of Object.values(R)) {
    if (!r.bark) continue;
    for (let y = r.y; y < r.y + r.h; y++) for (let x = r.x; x < r.x + r.w; x++) d[(y * S + x) * 4 + 3] = 255;
    for (let y = r.y; y < r.y + r.h; y++) {
      for (let g = 1; g <= 8; g++) {
        const L = ((y * S) + r.x - g) * 4, Rr = ((y * S) + r.x + r.w - 1 + g) * 4;
        const srcL = ((y * S) + r.x + r.w - g) * 4, srcR = ((y * S) + r.x + g - 1) * 4;
        if (r.x - g >= 0) for (let c = 0; c < 4; c++) d[L + c] = d[srcL + c];
        if (r.x + r.w - 1 + g < S) for (let c = 0; c < 4; c++) d[Rr + c] = d[srcR + c];
      }
    }
    for (let g = 1; g <= 8; g++) {
      const yt = r.y - g, yb = r.y + r.h - 1 + g;
      for (let x = r.x - 8; x < r.x + r.w + 8; x++) {
        if (x < 0 || x >= S) continue;
        if (yt >= 0) for (let c = 0; c < 4; c++) d[(yt * S + x) * 4 + c] = d[(r.y * S + x) * 4 + c];
        if (yb < S) for (let c = 0; c < 4; c++) d[(yb * S + x) * 4 + c] = d[((r.y + r.h - 1) * S + x) * 4 + c];
      }
    }
  }
  const regions = Object.values(R).map((r) => ({ x: r.x, y: r.y, w: r.w, h: r.h, cov: !!r.leaf }));
  const levels = buildMipChain(d, S, S, regions, true);
  const texture = dataTextureFromMips(levels, THREE.SRGBColorSpace);
  texture.anisotropy = 4;
  texture.name = 'vegAtlas';
  return { texture, levels };
}
