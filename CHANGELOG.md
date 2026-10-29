# Changelog

## 0.4.0 — 2026-10-28
MacBook pass (Apple M-series, Retina 2x, ProMotion): measured in the desktop app's Chromium at a 14-inch MacBook Pro's
pixel count (3024x1964) with `Profiler.benchThroughput`, pixel diffs read back from the framebuffer. Canyon Run's
driving view went from 15.9 to 8.2 ms, the chase view from 17.1 to 8.7 and the aerial view from 20.6 to 8.8 ms, with every
quality-affecting knob measured by full-resolution pixel diff (ENGINE.md, "MacBook pass").
- Render: `PostFX` clears the scene target after the shadow maps (`lateClear`, default on: identical output, ~1 ms
  saved on Apple GPUs where the early clear stored and reloaded the whole multisampled target), caps the AO buffers at
  `ao.maxPixels` (default 0.6 Mpx) with a depth-aware upsample in the final pass (`ao.upsample: 'depth'`, default)
  and compact RG16F AO buffers (`ao.compact`, default on): -1.3 to -2.5 ms at Retina, full-resolution pixel change
  <= 0.04 / 255 mean. `setAOResolution({ scale, maxPixels, upsample })`. The final pass skips chromatic-aberration
  taps and the sun-scatter pow when they are off, and reads the exposure once per vertex (identical output).
- Render: `msaaMaxPixels` (default Infinity): `msaa: 'auto'` uses FXAA above that drawing-buffer pixel count; the 4x
  resolve (colour + depth blit, unavoidable in WebGL2) costs 2.7 ms at 3024x1964 and 4x -> 2x saves only 0.4. Visible
  on high-frequency facades (17 % of pixels change by > 2 / 255 in the city), so it is opt-in; Canyon Run sets 4e6
  (`?msaamax=`).
- Render: opt-in `'fast'` directional shadow kernel (`Atmosphere({ shadows: { kernel: 'fast' } })`,
  `atmo.setShadowKernel('fast' | 'pcf')` at runtime, `render/shadowKernel.js`): 5 hardware-compare taps per lookup
  (a DepthTexture with LEQUAL compare on the same shadow render) instead of three's 17 unpacked taps, the same
  penumbra width: -2 ms at 3024x1964, full-resolution pixel change <= 0.015 / 255 mean. Falls back to 'pcf' with a
  warning when the shadow type is not PCF or another shadow-casting directional light exists. Canyon Run uses it
  (`?shadows=pcf` to compare).
- Core: `Engine({ targetFps })` (live `engine.targetFps`, default 0 = every rAF callback): paces ticks from the
  rAF stream (60 on a 120 Hz ProMotion display ticks every second callback), `engine.refreshRate` (display Hz,
  estimated from the raw callbacks; F3 overlay and `?profile=` table), adaptive resolution with a frame budget of
  1000 / min(targetFps, max(refreshRate, 60)) ms when capped and the original 60 fps budget when not (a 120 Hz
  display keeps full resolution at 60 fps as before; targetFps 120 chases 120 fps at lower resolution). Canyon Run:
  `?fps=60`.
- Textures: the procedural fallback layers of `makeTextureArray` are generated in a module-worker pool
  (`render/proceduralTextures.worker.js`, `render/proceduralPbr.js`; `TextureSets.configure({ proceduralWorkers })`,
  `?procworkers=0`), byte-identical to the in-thread path (checked by hash): main-thread cost per layer 24-68 ms ->
  3-5 ms; in-thread in Node and as the fallback. Stats gain `procIn`, `workerMs`, `procMs`; `makeTextureArray`
  takes `free: false` to keep the CPU copies.
- Dev: `samples/city/` renders the procedural city in the browser at http://localhost:5190/city (served by a
  vite.config.js plugin, dev only; `?t=`, `?traffic=`, `?notraffic`, `?nostream=1`, `?nofx=1`, `?profile=`), so the
  world layer can be seen and measured without the game repository.
- Canyon Run: the terrain splat shader skips fetches whose weight is provably zero (no dirt fetches on steep walls where
  the rock weight is exactly 1, no rock fetches where it is 0; output-neutral, 1 LSB in < 100 of 5.5 M pixels), moves the
  biplanar gradients out of the per-pixel branch (the old code computed them inside it from neighbours that skipped
  it: wrong mips along the wall foot, up to 17 / 255) and gains live detail uniforms (`uDetail` fades, projection
  cut): with the sample's defaults the terrain costs 4.2 ms instead of 7.7 at 3024x1964, pixel change mean <= 0.03 / 255.
- Textures: with little else to overlap, Canyon Run's end-to-end load is unchanged by the worker path (a fresh worker's
  cold JIT makes a layer ~2x slower off-thread); the city's 15-layer facade array keeps its main thread free for the
  rest of its build (15 ms instead of ~130 ms on the main thread).

First pass, 2026-10-27 (same branch; measured on the same M4, see ENGINE.md "Optimisation pass (2026-10-27)"):
- Perf: `bakeImpostors` box-filters the 2x supersampled cells on the GPU (alpha-weighted, one small readback per row)
  instead of on the CPU, and sizes the atlas to the rows its cells need (power of two) instead of the full square, so
  the dilate / mip post-processing and the texture memory scale with the model count. Canyon Run's 80-view bake:
  273-302 ms -> 137-148 ms on a fresh page load, 141 -> 64 ms warm (Apple M4); atlas texels within +-1 of the CPU
  filter. Phase times (context, render, downsample, post) are printed with the bake line and returned via `timing`.
- Perf: merged static meshes get a local origin. `chunkMesh` tiles (`chunkGeometry({ localOrigin })`, on by default
  in `chunkMesh`) and the city's building shells (`MapWorld`, after the plugin hooks; streamed tiles on attach) sit at
  their XZ centre with local vertices, so three.js's front-to-back sort of opaque objects orders them by distance
  instead of creation order (every merged mesh sat at the world origin: identical sort keys). World positions are
  unchanged (<= 0.06 mm, float32). Measured no difference on Apple GPUs (tile-based: opaque overdraw is already
  free); it is for immediate-mode GPUs. New helper `localizeMeshOrigin(mesh)`. Plugins that edit merged building
  geometry by world coordinates still run before the change (tiling step).
- Headless: `Atmosphere` accepts `renderer: null` (lights, sun direction, darkness; no PMREM / sky cube / horizon
  sampling / flare), so `MapWorld` builds in Node as documented; `bakeImpostors` returns null without WebGL instead
  of throwing into a warning. New `test/world-build.mjs` (run by `npm test` through `test/helpers/dom-shim.mjs`, a
  no-op canvas shim) builds the procedural city headless and pins its stats, colliders, physics queries, building
  origins and 10 s of deterministic traffic.
- Perf (bit-identical, content-hash checked): `paintCanvas` hands painters a reusable `[r, g, b]` (no per-texel
  array), `pnoise` without its per-call closure, the mip chain's colour-space branch hoisted out of the texel loop,
  `fillTransparent` / `dilateRGB` without a per-texel offsets array, procedural texture-array layers tiled as 32-bit
  texels. About 10 % on the procedural painters; a per-cell corner cache for `pnoise` and an octave-major cloud fBm
  measured no faster and were not kept.
- Canyon Run: terrain chunks placed at their chunk centre (same local-origin reasoning; physics and visuals unchanged).
- ENGINE.md: "Optimisation pass (2026-10-27)" - the method and the numbers behind the above, including where a
  2560x1440 frame of the sample goes and what did not help (depth pre-pass, draw order on Apple GPUs).

## 0.3.1 — 2026-10-21
- Standalone repository (`inbrowser-game-engine`): the Canyon Run sample, `test/car-sim.mjs` and ENGINE.md moved in
  from the game; `npm run dev` / `npm run build` serve and build the sample, `npm test` runs the physics checks.
  The game ([roadster](https://github.com/BrandeisPatrick/roadster)) embeds this repository as a submodule.

## 0.3.0 — 2026-10-19
- Physics: `VehicleController` is a sprung body — per-wheel spring / damper / anti-roll suspension with bump stops, body
  pitch and roll about the CG (`cgHeight`, `pitchInertia`, `rollInertia`, `suspensionDamping`, `antiRoll`), tyre-radius
  support sampling (kerbs are climbed over several steps), axle loads from the springs. New outputs `wheels[i].y`,
  `.contact`, `bottomed`, `flightArmed`.
- Physics: flight must be armed by a lift input (`flightArmThrust`); a crest jump is ballistic (`mode` stays `'ground'`,
  no hover assist). Step-height boxes are supports, not walls, for the body box. Moving boxes (`vx`, `vz`) shove with
  their velocity. `reset()` / `hold` fit the body to the ground plane (no first-step snap on a hill).
- Physics: `PhysicsWorld.fromWorld({ deckStep })` forwards a deck acceptance height to `world.getGroundHeight(x, z, y, step)`.
- World: `MapWorld.pavingLift` / `carriagewayAt` (exact asphalt ribbon +0.05 with end caps and an edge ramp, sidewalk
  slabs +0.17), `MapWorld.surfaceAt` + `SURFACES` (asphalt / paving / grass / ground / water), `ROAD_LIFT`,
  `SIDEWALK_LIFT`. `RoadNetwork` ground segments carry `ext: hw`.
- World: deck solver pins grounded ends to the ground with the grade a piece needs, and keeps decks 4.6 m above the
  streets they cross. Parked cars at an angle become three collider slices; signal poles never stand on a carriageway.
- Game: the Roadster's body mesh rides on the physical suspension; wheels follow their contact points. The Canyon Run
  buggy does the same.
- Dev: `dev/drive-sim.mjs` drives the car along 23 real streets headless and reports snaps, hops, flickers, impacts.

## 0.2.0 — 2026-10-18
- Fix: chase-camera shake used g-force with a 0.3 g baseline, so a car at rest (1 g) trembled at 6–15 Hz; now only load above 1.15 g shakes, at lower frequencies, default `shakeScale` 0.6 (0 disables).
- Fix: fixed-step judder — vehicles interpolate between physics steps (`alpha`); traffic is 'thin' for camera occlusion.
- Packaged as an npm workspace (`@vibe/engine`) with subpath exports per layer.
- World layer: `MapWorld` with plugin hooks, data contracts (city/terrain/farland JSON), parameterised bakers in `tools/`.
- KTX2/Basis texture pipeline (`tools/compress-textures.mjs`, `TextureSets` KTX2 path with JPG/procedural fallback).
- Traffic simulation core split for Worker execution; tile streaming (`TileStreamer`) around the camera.
- Physics: `PhysicsWorld` (static grid, heightfield, dynamic providers, raycast/sphereCast, surfaces), `VehicleController`
  spec object (`flight`, `airControl`, `normalClamp`, `hold`, per-wheel load), fixes found by the Canyon Run sample.
- Render: Atmosphere (sun/clouds/stars, 2-cascade shadows), PostFX (HDR, eye adaptation, GTAO, bloom, fog, grading),
  InstancedPool / LODGroup / ChunkCuller / Impostors / Decals / Particles, noise.
- Input (action maps, gamepad, touch), camera rigs (chase/orbit/free-fly with occlusion), audio bus + synth, UI widgets.

## 0.1.0 — 2026-10-17
- Core loop + profiler extracted from the Roadster flying-car game.
