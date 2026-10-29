// Minimal DOM / canvas shim for headless (Node) builds of the world layer: procedural canvas textures run against a
// no-op 2D context (blank texels), so MapWorld can build its data and geometry without a browser. Import it first.
const noop = () => {};
function ctx2d(c) {
  const g = { canvas: c, fillStyle: '#000', strokeStyle: '#000', lineWidth: 1, font: '', textAlign: 'left', textBaseline: 'top', globalAlpha: 1, globalCompositeOperation: 'source-over', lineCap: 'butt', lineJoin: 'miter', shadowBlur: 0, shadowColor: '#000', filter: 'none', imageSmoothingEnabled: true };
  for (const k of ['fillRect', 'clearRect', 'strokeRect', 'beginPath', 'closePath', 'moveTo', 'lineTo', 'arc', 'arcTo', 'ellipse', 'rect', 'fill', 'stroke', 'fillText', 'strokeText', 'save', 'restore', 'translate', 'rotate', 'scale', 'transform', 'setTransform', 'resetTransform', 'drawImage', 'quadraticCurveTo', 'bezierCurveTo', 'clip', 'setLineDash']) g[k] = noop;
  g.putImageData = (img, x, y) => { if (x === 0 && y === 0 && img.width === c.width && img.height === c.height) c.__img = img; };
  g.measureText = (t) => ({ width: (t ? t.length : 0) * 8 });
  g.getImageData = (x, y, w, h) => (c.__img && x === 0 && y === 0 && w === c.width && h === c.height ? c.__img : { width: w, height: h, data: new Uint8ClampedArray(w * h * 4) });
  g.createImageData = (w, h) => ({ width: w, height: h, data: new Uint8ClampedArray(w * h * 4) });
  const grad = { addColorStop: noop };
  g.createLinearGradient = () => grad; g.createRadialGradient = () => grad; g.createPattern = () => ({});
  return g;
}
function canvas() {
  const c = { width: 300, height: 150, style: {}, nodeName: 'CANVAS' };
  c.getContext = (kind) => (kind === '2d' ? ctx2d(c) : null);
  c.toDataURL = () => 'data:,'; c.toBlob = (cb) => cb && cb(null);
  c.addEventListener = noop; c.removeEventListener = noop;
  return c;
}
globalThis.document = globalThis.document || {
  createElement: (tag) => (tag === 'canvas' ? canvas() : { style: {}, appendChild: noop, removeChild: noop, addEventListener: noop, removeEventListener: noop, setAttribute: noop, classList: { add: noop, remove: noop } }),
  createElementNS: () => canvas(),
  body: { appendChild: noop, removeChild: noop }, head: { appendChild: noop },
  addEventListener: noop, removeEventListener: noop, querySelector: () => null,
};
globalThis.ImageData = globalThis.ImageData || class ImageData { constructor(d, w, h) { this.data = d; this.width = w; this.height = h; } };
globalThis.Image = globalThis.Image || class Image { constructor() { this.src = ''; this.width = 0; this.height = 0; } addEventListener() {} removeEventListener() {} };
