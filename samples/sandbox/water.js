// Sandbox pool (?scene=buoyancy): a square basin of PhysicsWorld static boxes standing on the flat ground (the ground is
// its floor) holding water at `level`, with a wave surface that is the world Water module's Gerstner swell
// (src/world/Water.js: WAVES / waveHeight / WAVE_GLSL / waveUniforms) so the fluid the physics floats bodies on and
// the surface drawn are one height function:
//   heightAt(x, z, t) = level + waveHeight(x, z, t, cx, cz)   inside the basin, NO_FLUID outside (no buoyancy there)
// Determinism: waveHeight uses Math.hypot / Math.sin, which the ECMAScript spec leaves implementation-approximated, and
// Buoyancy.beginStep calls it every step: the buoyancy scene is bit-identical within one JS engine (V8: Node and the
// app's Chromium agree), not across every browser like the rigid step (+ - * / sqrt only).
// - the "camera" of waveHeight's per-wave distance fade is pinned at the pool centre (cx, cz) in both the physics and
//   the shader, so the height never depends on where the camera is (the pool's half diagonal, 8.5 m, is inside the
//   shortest wave's fade start, 9.3 m: nothing fades at all);
// - t is the pool's own clock, which the scene advances in its fixed step (never wall-clock) and sets on the surface
//   per frame (setTime), so a bench pose (reset + a fixed number of steps) always shows the same waves;
// - the surface displaces its vertices vertically only, by poolWave() below: waveHeight's sum in GLSL (the uniforms
//   and NWAVES of WAVE_GLSL), without the Gerstner xz pinch that the world water's gerstner() adds and waveHeight
//   leaves out, so the drawn height at (x, z) equals heightAt(x, z, t) up to float32 and the 12 cm grid's linear
//   interpolation between vertices; the normal is that height field's (-dh/dx, 1, -dh/dz), normalised.
// buildStatics (scenes.js) creates the pool once per page, before the DynamicsClient (PhysicsWorld boxes cannot be
// removed and a worker client copies them at start-up); the scene reads it from ctx.statics.
//
//   const pool = createPool({ physics, scene, renderer });   // 12 x 12 m at (0, 0), water at 0.8 m
//   fixed step: pool.time = n * dt;  buoyancy.setTime(pool.time)      per frame: pool.setTime(renderTime)
import * as THREE from 'three';
import { waveHeight, waveUniforms, WAVE_GLSL } from '@vibe/engine/world';
import { makeCanvas, finishTexture } from '@vibe/engine/render';

/** waveHeight() in GLSL (WAVE_GLSL's uniforms): height and the height field's normal, no xz displacement. */
const POOL_WAVE_GLSL = /* glsl */`
float poolWave(vec2 p, float dist, float t, out vec3 n) {
  float h = 0.0; vec2 g = vec2(0.0);
  for (int i = 0; i < NWAVES; i++) {
    vec4 A = uWaveA[i]; vec4 B = uWaveB[i];
    float a = A.w * (1.0 - smoothstep(B.w, B.w * 1.5, dist)); // waveHeight's fade (fade1 = 1.5 fade0)
    if (a <= 0.0) continue;
    float th = A.z * dot(A.xy, p) - B.x * t + B.z;
    h += a * sin(th);
    g += A.z * a * cos(th) * A.xy;
  }
  n = normalize(vec3(-g.x, 1.0, -g.y));
  return h;
}
`;

/** heightAt outside the basin: far below anything, so no sample point is ever under it. */
export const NO_FLUID = -1e4;

/** Pool floor tiles: 4 x 4 pale blue tiles with grout lines per texture (1 m). */
function tileTexture(renderer) {
  const S = 256, cell = S / 4, c = makeCanvas(S), g = c.getContext('2d');
  for (let y = 0; y < 4; y++) for (let x = 0; x < 4; x++) { g.fillStyle = (x + y) & 1 ? '#9fd0dc' : '#a9d6e0'; g.fillRect(x * cell, y * cell, cell, cell); }
  g.fillStyle = 'rgba(40,70,80,0.35)';
  for (let i = 0; i < 4; i++) { g.fillRect(i * cell, 0, 2, S); g.fillRect(0, i * cell, S, 2); }
  return finishTexture(new THREE.CanvasTexture(c), renderer);
}

/**
 * @param {object} o
 * @param {object} o.physics  PhysicsWorld (the walls become its static boxes)
 * @param {THREE.Object3D} o.scene
 * @param {THREE.WebGLRenderer} [o.renderer]  texture anisotropy
 * @param {number[]} [o.center=[0, 0]]  basin centre (x, z)
 * @param {number} [o.size=12]  inner width (m, square)
 * @param {number} [o.level=0.8]  still-water height above the ground (m)
 * @param {number} [o.wall=0.3]  wall thickness (m)
 * @param {number} [o.height=1.3]  wall height (m)
 * @param {number} [o.segments=96]  surface grid cells per side
 * @returns {object} { center, size, level, bounds [minX, maxX, minZ, maxZ], time, heightAt(x, z, t), fluid, setTime(t),
 *   surface, group, dispose() }
 */
export function createPool(o) {
  const { physics, scene, renderer } = o;
  const C = o.center || [0, 0], cx = C[0], cz = C[1];
  const size = o.size ?? 12, level = o.level ?? 0.8, T = o.wall ?? 0.3, H = o.height ?? 1.3, half = size / 2;
  const minX = cx - half, maxX = cx + half, minZ = cz - half, maxZ = cz + half;
  const group = new THREE.Group(); group.name = 'pool';
  const geos = [], mats = [], texs = [];

  // ---- walls: PhysicsWorld static boxes (minX, minY, maxX, maxY, minZ, maxZ) + their meshes
  const concrete = new THREE.MeshStandardMaterial({ color: 0xd8d2c4, roughness: 0.85, metalness: 0 }); mats.push(concrete);
  const walls = [
    [minX - T, maxX + T, minZ - T, minZ], [minX - T, maxX + T, maxZ, maxZ + T], // north, south
    [minX - T, minX, minZ, maxZ], [maxX, maxX + T, minZ, maxZ],                 // west, east
  ];
  for (const [x0, x1, z0, z1] of walls) {
    physics.addStaticBox(x0, 0, x1, H, z0, z1);
    const g = new THREE.BoxGeometry(x1 - x0, H, z1 - z0); geos.push(g);
    const m = new THREE.Mesh(g, concrete);
    m.position.set((x0 + x1) / 2, H / 2, (z0 + z1) / 2);
    m.castShadow = true; m.receiveShadow = true;
    group.add(m);
  }
  // ---- floor tiles (visual: the collider is the ground at y = 0)
  const tiles = tileTexture(renderer); texs.push(tiles);
  tiles.repeat.set(size, size);
  const floorMat = new THREE.MeshStandardMaterial({ map: tiles, roughness: 0.4, metalness: 0, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2 });
  mats.push(floorMat);
  const floorGeo = new THREE.PlaneGeometry(size, size).rotateX(-Math.PI / 2); geos.push(floorGeo);
  const floor = new THREE.Mesh(floorGeo, floorMat);
  floor.position.set(cx, 0.002, cz); floor.receiveShadow = true;
  group.add(floor);

  // ---- surface: a lit, see-through MeshStandardMaterial displaced by the swell in the vertex shader
  const uniforms = { uTime: { value: 0 }, uPoolCenter: { value: new THREE.Vector2(cx, cz) }, ...waveUniforms() };
  const water = new THREE.MeshStandardMaterial({ color: 0x1f6f80, roughness: 0.06, metalness: 0, transparent: true, opacity: 0.58 });
  water.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, uniforms);
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', `#include <common>\nuniform float uTime;\nuniform vec2 uPoolCenter;\n${WAVE_GLSL}\n${POOL_WAVE_GLSL}`)
      .replace('#include <beginnormal_vertex>', /* glsl */`#include <beginnormal_vertex>
  vec3 poolN;
  vec2 poolP = (modelMatrix * vec4(position, 1.0)).xz;
  float poolH = poolWave(poolP, length(poolP - uPoolCenter), uTime, poolN);
  objectNormal = poolN; // the mesh is unrotated: world normal = object normal`)
      .replace('#include <begin_vertex>', '#include <begin_vertex>\n  transformed.y += poolH; // = waveHeight()');
  };
  water.customProgramCacheKey = () => 'sandbox-pool-water';
  mats.push(water);
  const seg = o.segments ?? 96;
  const surfGeo = new THREE.PlaneGeometry(size, size, seg, seg).rotateX(-Math.PI / 2); geos.push(surfGeo);
  surfGeo.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), half * Math.SQRT2 + 0.5); // waves stay within 0.3 m
  const surface = new THREE.Mesh(surfGeo, water);
  surface.name = 'pool.surface';
  surface.position.set(cx, level, cz);
  surface.receiveShadow = true;
  group.add(surface);
  scene.add(group);

  const heightAt = (x, z, t) => (x < minX || x > maxX || z < minZ || z > maxZ ? NO_FLUID : level + waveHeight(x, z, t, cx, cz));
  const pool = {
    center: [cx, cz], size, level, bounds: [minX, maxX, minZ, maxZ], group, surface,
    /** The pool clock (s): the scene advances it per fixed step. */
    time: 0,
    heightAt,
    /** Defaults for Buoyancy's `fluid` (it copies them: new Buoyancy(world, { fluid: { ...pool.fluid, drag } })). */
    fluid: { heightAt, density: 1000, drag: 1, angularDrag: 0.5, time: 0 },
    /** Wave time drawn by the surface (per frame: the render time of the fixed-step clock). */
    setTime(t) { uniforms.uTime.value = t; },
    dispose() {
      scene.remove(group);
      for (const g of geos) g.dispose();
      for (const m of mats) m.dispose();
      for (const t of texs) t.dispose();
    },
  };
  return pool;
}
