// Minimap — heading-up canvas map: polylines pre-rendered once into an offscreen canvas, then a rotated,
// zoomed crop is drawn per frame; north indicator, optional markers, player arrow, footer labels.
//
//   const map = new Minimap({ parent: panel, size: 220 });
//   map.setData({ roads: [[[x, z], ...], ...], bounds: { minX, maxX, minZ, maxZ } });
//   layer.onScale((s) => map.setScale(s));
//   map.update(dt, x, z, headingDeg, viewM, playerColor);            // every frame (positional: no allocation)
//   map.update(dt, { x, z, headingDeg, viewM: 360, playerColor });   // or an options object (reuse yours) ...
//   map.view.x = x; ...; map.update(dt);                              // ... or mutate the built-in map.view
//   const m = map.addMarker({ x, z, color: '#3e9bff', radius: 4 });   // world-space markers (m.x / m.z mutable)
//   map.setFoot(`View ${map.viewMeters} m`, 'N 000°');
// World axes: +X east (right), -Z north (up) — heading 0 = facing -Z.
// Colours: without an explicit `accent` / `accentRgb` the map reads theme.accent / theme.accentRgb every frame,
// so setTheme() restyles an existing minimap too.
import { Widget } from './Widget.js';
import { injectStyle, theme } from './theme.js';
import { el } from './dom.js';

const CSS = `
.ui-minimap canvas{display:block;border-radius:16px}
.ui-minimap-foot{display:flex;justify-content:space-between;padding:6px 4px 0}
`;

export class Minimap extends Widget {
  /**
   * @param {object} [o]  size=220 (css px), background='#0d1117', accent, foot=true,
   *   strokes=[[14, casingColor], [8, roadColor]] (metres, colour) drawn in order, pad=60 (m), maxDim=2560, maxPpm=1.5,
   *   zoomRate=2.5 (1/s smoothing of the view width)
   */
  constructor(o = {}) {
    super(o);
    injectStyle('ui-minimap', CSS);
    this.size = o.size ?? 220;
    this.background = o.background || '#0d1117';
    this._accent = o.accent || null;         // null: follow theme.accent (see the accent getter)
    this._accentRgb = o.accentRgb || null;
    /** Reusable view state for update(dt) without arguments. */
    this.view = { x: 0, z: 0, headingDeg: 0, viewM: 360, playerColor: null };
    this.strokes = o.strokes || [[14, 'rgba(120,132,150,.35)'], [8, 'rgba(196,204,216,.85)']];
    this.pad = o.pad ?? 60; this.maxDim = o.maxDim ?? 2560; this.maxPpm = o.maxPpm ?? 1.5;
    this.zoomRate = o.zoomRate ?? 2.5;
    this.zoom = 1;      // css px per metre (smoothed)
    this.map = null;    // { canvas, ppm, minX, minZ, maxX, maxZ }
    this.markers = [];
    const root = this.el = el('div', `ui-w ui-minimap${o.className ? ` ${o.className}` : ''}`);
    this.canvas = el('canvas', '', root);
    this.canvas.style.width = this.canvas.style.height = `${this.size}px`;
    this.ctx = this.canvas.getContext('2d');
    if (o.foot !== false) {
      const f = el('div', 'ui-minimap-foot', root);
      this.footLeft = el('span', 'ui-label ui-num', f, o.footLeft ?? '');
      this.footRight = el('span', 'ui-label ui-num', f, o.footRight ?? '');
    }
    this.setScale(1);
    this.mountTo(o.parent);
  }

  /** Backing-store resolution for the UI scale (css px stay `size`; device pixels follow dpr * scale). */
  setScale(s) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const px = Math.round(this.size * dpr * s);
    if (this.canvas.width !== px) { this.canvas.width = px; this.canvas.height = px; }
  }

  /** data: { roads: [[ [x,z], ... ], ...], bounds?: {minX,maxX,minZ,maxZ} } (null clears -> grid fallback). */
  setData(data) {
    if (!data || !data.roads) { this.map = null; return; }
    let b = data.bounds;
    if (!b) {
      b = { minX: Infinity, maxX: -Infinity, minZ: Infinity, maxZ: -Infinity };
      for (const r of data.roads) for (const p of r) {
        if (p[0] < b.minX) b.minX = p[0]; if (p[0] > b.maxX) b.maxX = p[0];
        if (p[1] < b.minZ) b.minZ = p[1]; if (p[1] > b.maxZ) b.maxZ = p[1];
      }
    }
    const pad = this.pad;
    const minX = b.minX - pad, minZ = b.minZ - pad;
    const wM = Math.max(1, b.maxX - b.minX + 2 * pad), hM = Math.max(1, b.maxZ - b.minZ + 2 * pad);
    const ppm = Math.min(this.maxPpm, this.maxDim / wM, this.maxDim / hM);
    const cw = Math.ceil(wM * ppm), ch = Math.ceil(hM * ppm);
    const c = document.createElement('canvas');
    c.width = cw; c.height = ch;
    const g = c.getContext('2d');
    g.lineCap = 'round';
    g.lineJoin = 'round';
    const roadW = (m) => Math.max(1.2, m * ppm);
    for (const [metres, color] of this.strokes) {
      g.strokeStyle = color;
      g.lineWidth = roadW(metres);
      g.beginPath();
      for (const r of data.roads) {
        if (!r || r.length < 2) continue;
        g.moveTo((r[0][0] - minX) * ppm, (r[0][1] - minZ) * ppm);
        for (let i = 1; i < r.length; i++) g.lineTo((r[i][0] - minX) * ppm, (r[i][1] - minZ) * ppm);
      }
      g.stroke();
    }
    this.map = { canvas: c, ppm, minX, minZ, maxX: minX + wM, maxZ: minZ + hM };
  }

  addMarker(m) { const mk = { x: 0, z: 0, color: '#fff', radius: 3.5, ...m }; this.markers.push(mk); return mk; }
  removeMarker(m) { const i = this.markers.indexOf(m); if (i >= 0) this.markers.splice(i, 1); }
  clearMarkers() { this.markers.length = 0; }

  /** Ring / player accent: the explicit colour, else the live theme value. Assign to pin a colour. */
  get accent() { return this._accent || theme.accent; }
  set accent(v) { this._accent = v || null; }
  get accentRgb() { return this._accentRgb || theme.accentRgb; }
  set accentRgb(v) { this._accentRgb = v || null; }

  /** Width of the current view in metres. */
  get viewMeters() { return this.size / Math.max(1e-3, this.zoom); }

  setFoot(left, right) {
    if (this.footLeft && left != null) this.setText('fl', this.footLeft, left);
    if (this.footRight && right != null) this.setText('fr', this.footRight, right);
  }

  /**
   * Draw one frame. Either positional update(dt, x, z, headingDeg, viewM, playerColor), an options object
   * update(dt, { x, z, headingDeg, viewM, playerColor }), or update(dt) with `this.view`. viewM: target view
   * width (m, smoothed); playerColor defaults to the accent.
   */
  update(dt, a, z, headingDeg, viewM, playerColor) {
    let x;
    if (typeof a === 'number') x = a;
    else {
      const o = a && typeof a === 'object' ? a : this.view;
      x = o.x; z = o.z; headingDeg = o.headingDeg; viewM = o.viewM; playerColor = o.playerColor;
    }
    x = x ?? 0; z = z ?? 0; headingDeg = headingDeg ?? 0; viewM = viewM ?? 360; playerColor = playerColor || this.accent;
    const cvs = this.canvas, ctx = this.ctx, SIZE = this.size;
    const W = cvs.width, H = cvs.height;
    const k = W / SIZE; // device px per css px
    const zTarget = SIZE / viewM; // css px per metre
    this.zoom += (zTarget - this.zoom) * (1 - Math.exp(-this.zoomRate * dt));
    const zoom = this.zoom * k; // device px per metre

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = this.background;
    ctx.fillRect(0, 0, W, H);

    const cx = W / 2, cy = H / 2;
    const theta = -headingDeg * Math.PI / 180; // rotate world so heading points up
    const px = x, pz = z;

    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(theta);
    if (this.map) {
      const m = this.map;
      const sc = zoom / m.ppm;
      ctx.scale(sc, sc);
      ctx.translate(-(px - m.minX) * m.ppm, -(pz - m.minZ) * m.ppm);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(m.canvas, 0, 0);
    } else {
      // no data: faint 100 m grid so motion is still visible
      ctx.strokeStyle = 'rgba(255,255,255,.07)';
      ctx.lineWidth = 1;
      const step = 100 * zoom;
      const half = Math.hypot(W, H) / 2 + step;
      const ox = -((px * zoom) % step), oz = -((pz * zoom) % step);
      ctx.beginPath();
      for (let gx = -half; gx <= half; gx += step) { ctx.moveTo(gx + ox, -half); ctx.lineTo(gx + ox, half); }
      for (let gz = -half; gz <= half; gz += step) { ctx.moveTo(-half, gz + oz); ctx.lineTo(half, gz + oz); }
      ctx.stroke();
    }
    ctx.restore();

    // vignette ring
    const R = Math.min(W, H) / 2 - 10 * k;
    ctx.strokeStyle = 'rgba(255,255,255,.08)';
    ctx.lineWidth = 1 * k;
    ctx.beginPath(); ctx.arc(cx, cy, R * 0.5, 0, Math.PI * 2); ctx.stroke();

    // north indicator on the ring
    const na = -Math.PI / 2 + theta;
    const nx = cx + Math.cos(na) * R, ny = cy + Math.sin(na) * R;
    ctx.fillStyle = 'rgba(13,17,23,.85)';
    ctx.beginPath(); ctx.arc(nx, ny, 9 * k, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = this.accent; ctx.lineWidth = 1.2 * k; ctx.stroke();
    ctx.fillStyle = '#fff';
    ctx.font = `600 ${10 * k}px -apple-system, system-ui, sans-serif`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText('N', nx, ny + 0.5 * k);

    // markers (world positions, rotated with the map; clamped to the ring)
    if (this.markers.length) {
      const c = Math.cos(theta), s = Math.sin(theta);
      for (const mk of this.markers) {
        const wx = (mk.x - px) * zoom, wz = (mk.z - pz) * zoom;
        let sx = wx * c - wz * s, sy = wx * s + wz * c;
        const d = Math.hypot(sx, sy);
        if (d > R && mk.clamp !== false) { sx *= R / d; sy *= R / d; }
        ctx.fillStyle = mk.color;
        ctx.beginPath(); ctx.arc(cx + sx, cy + sy, mk.radius * k, 0, Math.PI * 2); ctx.fill();
      }
    }

    // player marker (heading-up)
    ctx.save();
    ctx.translate(cx, cy);
    ctx.scale(k, k);
    ctx.fillStyle = `rgba(${this.accentRgb},.18)`;
    ctx.beginPath(); ctx.arc(0, 0, 14, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath();
    ctx.moveTo(0, -9); ctx.lineTo(6.5, 7); ctx.lineTo(0, 3.5); ctx.lineTo(-6.5, 7); ctx.closePath();
    ctx.fillStyle = playerColor;
    ctx.shadowColor = 'rgba(0,0,0,.6)'; ctx.shadowBlur = 4;
    ctx.fill();
    ctx.restore();
  }
}
