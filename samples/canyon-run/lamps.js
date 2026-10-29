// Trackside lamp field (opt-in: `?lamps=1`, wired in main.js): a steel pole with an arm over the track edge in a slot
// every ~28 m along the whole course, slots alternating sides, TRACK_HALF_WIDTH + 2.5 m from the centre line, each lamp
// with one point light in the engine's clustered light list (render/ClusteredLights.js: a froxel grid, so a fragment
// pays for the lamps that reach it, not for all of them; real three lights stay the sun / moon and the headlight).
// Two instanced draws (poles: lit, shadow casting; lenses: emissive, glow by darkness, kept out of the clusters by
// main.js so their own lamp does not blow them out), kind-3 thin box colliders (KIND_THIN: the buggy stops at a pole,
// the chase camera treats it as a thin post), and the light rows rewritten only when the darkness moved by more than
// 0.01 since the last write (positions are static; main.js calls ClusteredLights.update(camera) once per frame, after
// the camera moved, to re-cluster them for the view).
// Each pool's bounding sphere spans the whole loop, so every pole draws in the main pass and in every shadow cascade
// (118 x 100 triangles, ~12k per pass: negligible, by design; no LOD).
// Placement (placeLamps, pure, runs in Node): an even number of slots around the loop (so the slots alternate sides all
// the way round; a slot skipped at a gate leaves two neighbouring lamps on one side), each on the canyon floor like
// Props._ok (inside the map, clear of the other stretches of track, not within 22 m of a checkpoint gate, slope
// flatness >= 0.85), slid up to 4 m along the track to clear a rock or cactus collider, the base sunk to the lowest
// ground under the footing. Deterministic: no randomness.
// Light units as THREE.PointLight (linear colour x intensity, distance cutoff, decay), like world/NearestLights.js.
import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { InstancedPool } from '@vibe/engine';
import { TRACK_HALF_WIDTH } from './track.js';
import { WORLD_HALF } from './terrain.js';
import { KIND_THIN } from './props.js';

const POLE_H = 6.4;                     // top of the pole above its base (m)
const ARM = 1.6;                        // arm reach from the pole axis towards the track (m)
const HEAD = [0.72, 0.14, 0.34];        // lamp head box: along the arm, height, across
const HEAD_Y = POLE_H + 0.02;           // head centre height
const LENS_Y = HEAD_Y - HEAD[1] / 2 - 0.012; // lens centre height (a thin plate under the head, 1 cm overlap)
const LIGHT_Y = LENS_Y - 0.3;           // the point light hangs 0.3 m under the lens (the head rim / arm not lit from 0 cm)
const SHIFTS = [0, 1, -1, 2, -2];       // track points (2 m) a slot may slide to clear a collider

/** Lamp-field tuning (placement rules, PointLight-style light, lens glow). */
export const LAMPS = Object.freeze({
  spacing: 28,                          // m between slots (sides alternate)
  offset: TRACK_HALF_WIDTH + 2.5,       // m from the centre line
  clearGates: 22,                       // m around every checkpoint kept free (gates / gantry)
  minFlatness: 0.85,                    // terrain.flatness() at the footing
  max: 1000,                            // lamps (ClusteredLights maxLights 1024: a few slots stay free)
  color: Object.freeze([1.0, 0.82, 0.58]), // linear, warm sodium-ish white
  intensity: 120,                       // x darkness (PointLight units)
  range: 24, decay: 2,                  // PointLight distance / decay
  glow: 9,                              // lens emissiveIntensity x darkness
  epsilon: 0.01,                        // darkness change that rewrites the light rows
});

// ------------------------------------------------------------------ placement
/**
 * Lamp spots along the course (see the header). Pure: reads the track, the terrain and (optionally) the existing
 * static colliders, writes nothing.
 * @param {object} track  Track (n, length, x, z, tx, tz, checkpoints)
 * @param {object} terrain  Terrain (heightAt, flatness, trackDistance)
 * @param {object|null} [physics]  PhysicsWorld: rock / cactus colliders to keep clear of (queryStatic)
 * @param {object} [o]  overrides of LAMPS (spacing, offset, clearGates, minFlatness, max)
 * @returns {{ n: number, slots: number, base: Float32Array, light: Float32Array, skipped: Object<string, number> }}
 *   base: x, y (sunk footing), z, yaw per lamp (setTRS: local +x, the arm, points at the centre line); light: the
 *   point light's x, y, z per lamp; skipped: slots dropped per reason (edge, track, gate, slope, blocked)
 */
export function placeLamps(track, terrain, physics = null, o = {}) {
  const cfg = { ...LAMPS, ...o };
  const tr = track, n = tr.n, cps = tr.checkpoints;
  const slots = Math.max(2, 2 * Math.round(tr.length / (2 * cfg.spacing)));
  const cap = Math.max(0, Math.min(slots, cfg.max | 0));
  const base = new Float32Array(cap * 4), light = new Float32Array(cap * 3);
  const skipped = { edge: 0, track: 0, gate: 0, slope: 0, blocked: 0 };
  const near = [], g2 = cfg.clearGates * cfg.clearGates, lim = WORLD_HALF - 140;
  const reject = (x, z) => {
    if (Math.abs(x) > lim || Math.abs(z) > lim) return 'edge';
    if (terrain.trackDistance(x, z) < TRACK_HALF_WIDTH + 1.5) return 'track'; // another stretch of the course
    for (let c = 0; c < cps.length; c++) if ((cps[c].x - x) ** 2 + (cps[c].z - z) ** 2 < g2) return 'gate';
    if (terrain.flatness(x, z) < cfg.minFlatness) return 'slope';
    if (physics && physics.queryStatic(x, z, 0.9, near).length) return 'blocked';
    return '';
  };
  let m = 0;
  for (let k = 0; k < slots && m < cap; k++) {
    const sgn = k % 2 === 0 ? 1 : -1, d = sgn * cfg.offset, i0 = Math.round((k * n) / slots) % n;
    let why = '';
    for (let t = 0; t < SHIFTS.length; t++) {
      const i = (i0 + SHIFTS[t] + n) % n;
      const x = tr.x[i] - tr.tz[i] * d, z = tr.z[i] + tr.tx[i] * d; // right of travel = (-tz, tx)
      const r = reject(x, z);
      if (r) { if (t === 0) why = r; continue; }
      // arm towards the centre line: local +x -> world (cos yaw, 0, -sin yaw) = sgn * (tz, -tx)
      const yaw = Math.atan2(sgn * tr.tx[i], sgn * tr.tz[i]), c = Math.cos(yaw), s = Math.sin(yaw);
      const e = 0.4, te = terrain; // footing on the lowest ground under it (nothing floats on a slope)
      const y = Math.min(te.heightAt(x, z), te.heightAt(x + e, z), te.heightAt(x - e, z), te.heightAt(x, z + e), te.heightAt(x, z - e)) - 0.05;
      base[m * 4] = x; base[m * 4 + 1] = y; base[m * 4 + 2] = z; base[m * 4 + 3] = yaw;
      light[m * 3] = x + c * ARM; light[m * 3 + 1] = y + LIGHT_Y; light[m * 3 + 2] = z - s * ARM;
      m++; why = '';
      break;
    }
    if (why) skipped[why]++;
  }
  return { n: m, slots, base: base.subarray(0, m * 4), light: light.subarray(0, m * 3), skipped };
}

// ------------------------------------------------------------------ geometry (vertex coloured, local +x = the arm)
function paint(g, hex) {
  g.deleteAttribute('uv');
  const c = new THREE.Color(hex); // sRGB hex -> linear working space
  const n = g.attributes.position.count, col = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) { col[i * 3] = c.r; col[i * 3 + 1] = c.g; col[i * 3 + 2] = c.b; }
  g.setAttribute('color', new THREE.BufferAttribute(col, 3));
  return g;
}
function tube(p0, p1, r, hex) {
  const a = new THREE.Vector3().fromArray(p0), b = new THREE.Vector3().fromArray(p1);
  const g = new THREE.CylinderGeometry(r, r, a.distanceTo(b), 6, 1, true);
  g.applyQuaternion(new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), b.clone().sub(a).normalize()));
  return paint(g.translate((a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2), hex);
}

/** Footing (sunk), tapered pole, arm + brace and the head: one merged, indexed geometry (100 triangles: 32 + 32 + 12 + 12 + 12). */
function poleGeometry() {
  const STEEL = 0x55585d, HEADC = 0x34363a, CONCRETE = 0x8c867c;
  const g = mergeGeometries([
    paint(new THREE.CylinderGeometry(0.22, 0.28, 0.8, 8).translate(0, -0.22, 0), CONCRETE),            // y -0.62 .. 0.18
    paint(new THREE.CylinderGeometry(0.06, 0.1, POLE_H + 0.2, 8).translate(0, (POLE_H - 0.2) / 2, 0), STEEL),
    tube([0, POLE_H - 0.1, 0], [ARM - HEAD[0] * 0.3, HEAD_Y, 0], 0.045, STEEL),                          // arm
    tube([0, POLE_H - 0.95, 0], [ARM * 0.5, POLE_H - 0.06, 0], 0.03, STEEL),                             // brace
    paint(new THREE.BoxGeometry(HEAD[0], HEAD[1], HEAD[2]).translate(ARM, HEAD_Y, 0), HEADC),
  ]);
  g.computeBoundingSphere();
  return g;
}

/** The lens: a thin plate under the head, almost its full footprint (no lit head rim around it). */
function lensGeometry() {
  const g = new THREE.BoxGeometry(HEAD[0] - 0.02, 0.04, HEAD[2] - 0.02).translate(ARM, LENS_Y, 0);
  g.deleteAttribute('uv');
  g.computeBoundingSphere();
  return g;
}

// ------------------------------------------------------------------ lamp field
export class LampField {
  /**
   * @param {object} o
   * @param {THREE.Object3D} o.scene
   * @param {object} o.terrain  Terrain
   * @param {object} o.track  Track
   * @param {object} [o.physics]  PhysicsWorld: kind-3 pole colliders (and the placement keeps clear of existing ones)
   * @param {object} o.clustered  ClusteredLights: light i = lamp i (rows written from setLight)
   * @param {number} [o.max=LAMPS.max]  lamp cap (also capped at clustered.maxLights)
   */
  constructor({ scene, terrain, track, physics = null, clustered, max = LAMPS.max }) {
    this.clustered = clustered;
    const cap = Math.min(max, clustered ? clustered.maxLights : max);
    const P = placeLamps(track, terrain, physics, { max: cap });
    const n = this.n = P.n;
    this.base = P.base; this.light = P.light; this.slots = P.slots; this.skipped = P.skipped;
    this.poleMaterial = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.5, metalness: 0.45, envMapIntensity: 0.8 });
    // lenses: emissive only (main.js turns ENGINE_CLUSTERED off on this material after syncClustered)
    this.bulbMaterial = new THREE.MeshStandardMaterial({ color: 0x1c1a17, roughness: 0.3, metalness: 0, emissiveIntensity: 0 });
    this.bulbMaterial.emissive.setRGB(LAMPS.color[0], LAMPS.color[1], LAMPS.color[2]); // linear
    const cap1 = Math.max(1, n), reach = Math.hypot(ARM + HEAD[0] / 2, HEAD_Y + 0.1); // one lamp's radius about its base
    this.poles = new InstancedPool(poleGeometry(), this.poleMaterial, { capacity: cap1, castShadow: true, name: 'lampPoles', dynamic: false, cullRadius: reach });
    this.bulbs = new InstancedPool(lensGeometry(), this.bulbMaterial, { capacity: cap1, receiveShadow: false, name: 'lampBulbs', dynamic: false, cullRadius: reach });
    let colliders = 0;
    for (let i = 0; i < n; i++) {
      const x = P.base[i * 4], y = P.base[i * 4 + 1], z = P.base[i * 4 + 2], yaw = P.base[i * 4 + 3];
      this.poles.setTRS(this.poles.add(), x, y, z, yaw);
      this.bulbs.setTRS(this.bulbs.add(), x, y, z, yaw);
      if (physics && physics.addStaticBox(x - 0.16, y, x + 0.16, y + POLE_H, z - 0.16, z + 0.16, KIND_THIN) >= 0) colliders++;
    }
    this.poles.updateBounds(); this.bulbs.updateBounds();
    this.poles.update(null); this.bulbs.update(null); // static: visible when not empty, never updated again
    scene.add(this.poles.mesh, this.bulbs.mesh);
    this.enabled = true;
    this._d = -1; // darkness of the last light-row write (-1: never written)
    this.stats = { count: n, lit: 0, colliders, slots: P.slots, skipped: P.skipped };
    if (clustered) clustered.count = n;
  }

  /** Lights + lens glow on (darkness-driven) or off (intensity 0: out of every cluster, no recompile). */
  setEnabled(on) {
    this.enabled = !!on;
    this._d = -1; // rewrite on the next update
  }

  /**
   * Per frame (cheap when nothing changed): rewrite the light rows and the lens glow when the darkness moved by more
   * than LAMPS.epsilon since the last write (or crossed 0, or `force`). Returns true when it wrote. Allocation-free.
   * @param {number} darkness  atmosphere darkness 0 (day) .. 1 (night)
   * @param {boolean} [force=false]
   */
  update(darkness, force = false) {
    const d = this.enabled ? darkness : 0, last = this._d;
    if (!force && last >= 0 && Math.abs(d - last) <= LAMPS.epsilon && (d > 0) === (last > 0)) return false;
    this._d = d;
    const I = LAMPS.intensity * d, C = this.clustered, L = this.light, n = this.n;
    const r = LAMPS.color[0] * I, g = LAMPS.color[1] * I, b = LAMPS.color[2] * I;
    if (C) for (let i = 0; i < n; i++) C.setLight(i, L[i * 3], L[i * 3 + 1], L[i * 3 + 2], LAMPS.range, r, g, b, LAMPS.decay);
    this.bulbMaterial.emissiveIntensity = LAMPS.glow * d;
    this.stats.lit = I > 0 && C ? n : 0;
    return true;
  }
}
