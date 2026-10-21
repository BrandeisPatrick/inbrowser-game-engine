// UI layer: DOM widget kit with a CSS-variable theme (HUD gauges, bars, readouts, minimap, overlays).
// Theme timing: setTheme() updates the CSS variables (every DOM widget restyles live) and the JS `theme`
// values. Widgets that draw with JS colours read those values: Minimap reads theme.accent each frame (unless
// given an explicit accent), but AttitudeIndicator and GMeter bake theme.accent into their SVG when built and
// LoadingScreen uses it as the CSS fallback -> call setTheme() BEFORE building those (and anything of your
// own that reads `theme`, e.g. vehicle paint or gate textures).
export { theme, THEME_CSS, injectTheme, injectStyle, setTheme } from './theme.js';
export { Widget } from './Widget.js';
export { UILayer, Panel } from './Panel.js';
export { Gauge } from './Gauge.js';
export { Bar } from './Bar.js';
export { Readout } from './Readout.js';
export { AttitudeIndicator } from './AttitudeIndicator.js';
export { GMeter } from './GMeter.js';
export { Badge } from './Badge.js';
export { Segments } from './Segments.js';
export { Minimap } from './Minimap.js';
export { Flash } from './Flash.js';
export { HelpOverlay } from './HelpOverlay.js';
export { LoadingScreen } from './LoadingScreen.js';
import { el, svgEl, polar, arcPath, clamp as domClamp, host, SVGNS } from './dom.js';
/** DOM / SVG helpers used by the widgets (namespaced to avoid clashing with other layers). */
export const UIDom = Object.freeze({ el, svgEl, polar, arcPath, clamp: domClamp, host, SVGNS });
