# Changelog

## 0.9.0 — 2026-10-29
Unified XPBD (phase 5 of the 1.0 roadmap, ENGINE.md "Unified XPBD (0.9.0)"): particles, cloth, ropes, soft bodies,
particle-rigid coupling, buoyancy, pre-fractured bodies, character dynamics and ragdolls, all on the 0.6.0 solver; a
world without `maxParticles` steps bit-identically to 0.8.0 (pinned), every sample pose of 0.8.0 hashes the same.
- Physics: `DynamicsWorld({ maxParticles, particleDamping })` with `world.particles` (ParticleStore), `world.constraints`
  (distance / bending / attachment / tether), `addParticleSolver` plugins (`{ beginStep, solvePositions, solveVelocities,
  endStep, serialize, restore }`), `isActive`, `particleAabb`, `recollidePairs(ids, n)` (narrowphase + contact setup
  for pairs a plugin unglued in its beginStep; beginStep only); plugins also work on worlds without particles;
  `Cloth.grid`, `Rope.chain`; serialize / from carry particles and plugins; the worker client runs particles in-thread.
- Physics: `SoftBody.lattice` / `tetrahedralize` / `SoftBody.skin` / `VolumeConstraints` (tet volume + global pressure).
- Physics: `ParticleCollider` (particles vs bodies / static boxes / heightfield / trimeshes, two-way, friction) and
  `Buoyancy` (sample-point volumes, drag, `fluid.heightAt`), `closestPointOnShape`.
- Physics: `fracturePieces`, `FracturedBody` (glue joints with breakForce 15 kN, compliance 5e-7, breakStress,
  `impactBreak`), `FractureMesh`; the joint solver bounds a breaking joint's rows at breakForce per substep.
- Physics: `CharacterController.attachDynamics` (push light bodies, ride moving platforms; bit-identical without it).
- Anim: `Ragdoll.fromSkeleton` / activate / update / deactivate, `AnimationController.resync()`.
- Samples: sandbox `?scene=cloth|rope|soft|buoyancy|destructible` (+ `render/SoftMeshes.js`), character crates /
  elevator / ragdoll (K) / mantle with hand IK, poses `push` / `ragdoll`; smoke pages for the five scenes.
- Tests: `dyn-cloth` (181), `dyn-soft` (112), `dyn-buoyancy` (126), `dyn-fracture` (139), `character-dyn` (97),
  `ragdoll` (108); `npm test` runs 31 suites.
- Physics fix: a sphere or capsule rolling on the ground gained spin exponentially (0.6.0 stored contact points as
  material anchors that rotated with the body); contact anchors of bodies whose top-level shape is a sphere or a
  capsule are now re-seated on the surface every substep (a compound of spheres keeps the old anchors). Rolling
  scenes change (the sandbox stack ball), the box-only pins do not.
- Measured (Node 26, M4, ms per 1/120 s step with 4 substeps): cloth 32x32 0.40, 64x64 1.63; 1000 bodies + 200 rope
  particles 3.23; three soft cubes 0.89; 200 floating boxes on a wave 1.93; a 40-piece glued wall 3.29 (the joint
  tree), 3.09 after the hit. Destruction is local (a hole and a few thrown pieces; 90-98 % of a wall stays glued):
  a full collapse needs CCD, a 1.0+ item.

## 0.8.0 — 2026-10-29
Screen-space and volumetric rendering (phase 4 of the 1.0 roadmap, ENGINE.md "Render pipeline (0.8.0)"): reflections,
volumetric sun light, water refraction, motion blur, depth of field and 3D-LUT grading, composed by `PostFX` in the
order Scene -> Exposure -> AO -> SSR -> Volumetric -> TAA -> MotionBlur -> DoF -> Bloom -> Final. Every option is off
by default and the default output is byte-identical to 0.7.0 (shader texts pinned by sha1 with the new blocks under
absent defines; `npm run smoke` hashes every default pose equal to the 0.7.0 gate).
- Render: `PostFX({ ssr, volumetric, motionBlur, dof, lut, ao: { normals } })`, `postfx.set(path, value)` /
  `postfx.get(path)` for every knob (`'ssr'`, `'ssr.<opt>'`, `'volumetric'`, `'volumetric.<opt>'`, `'motionBlur'`,
  `'dof'`, `'lut'`, `'lut.strength'`, `'ao.normals'`, ...), passes created / removed / disposed at runtime,
  `colorSource` = the last enabled of TAA -> MotionBlur -> DoF; FinalPass hooks `FINAL_SSR` / `FINAL_VOLUMETRIC` /
  `FINAL_LUT`, TAA hook `TAA_SSR` (the reflection composited into every 3x3 tap so the variance clip keeps it).
- Render: `SSRPass` — half-resolution screen-space reflections on the G-buffer normal / roughness (perspective-correct
  march, bisection, premultiplied output `col * (1 - a) + rgb`); the specular share of metals is a capped heuristic
  until an F0 nibble lands in the G-buffer (0.9.0), no roughness blur yet, `jitter` off by default (the TAA hook
  cannot converge a jittered march). Canyon Run at 3024x1964: +0.5 ms (0.33-0.37 at 2560x1440), the city's shore
  +1.6 ms (the bay mirrors the warehouse: 1.44 mean change); `steps` / `scale` move it by < 0.1 ms.
- Render: `AOPass({ normals: 'gbuffer' })` — GTAO on the G-buffer normal instead of depth-reconstructed normals.
- Render: `VolumetricPass` — quarter-resolution shadowed sun in-scatter through the final pass's own haze + mist
  layers (Henyey-Greenstein, closed-form optical depth, cascade shadow taps on compare-mode or packed maps, the
  cascade walk of `shadowCascade.js`), temporally accumulated with a 2x2 depth-aware disocclusion; output per unit sun
  irradiance, composited as `sunColor * sun.intensity * intensity * r` after the fog (the analytic sun glow is dropped
  while it is on). Physical units: faint in thin desert air and inside a shadowed canyon; `density` / `intensity` are
  the look knobs. +0.4-0.8 ms at 3024x1964 (scale 0.5: +1.1-1.9), +0.3-0.5 at 2560x1440.
- Render: `MotionBlurPass` (McGuire tile-max reconstruction on the G-buffer velocity, camera reprojection elsewhere,
  exact pass-through when static, `shutter` for frame-rate-independent streaks) and `DoFPass` (signed thin-lens CoC,
  half-resolution far / near gathers, exact pass-through in focus, `circleOfConfusion()`); `Lut.js` (`LUT_GLSL`,
  `identityLut`, `lutFromFunction`, `lutLookup`) with the Assets `lut` value applied in the final pass. Motion blur
  costs 2.0-2.1 ms at 3024x1964 (1.3 at 2560x1440) even on a static frame (the velocity / tile passes are
  bandwidth), DoF 2.6 ms (1.7; `scale` 1 +6.2), the LUT 0 ms; `ao.normals: 'gbuffer'` < 0.1 ms.
- Render: `SceneCopy` — the opaque scene's colour + depth copied mid-frame from a mesh's `onBeforeRender` (one copy
  per frame, target / viewport restored), for refractive materials; needs a single-sampled scene target.
- World: `createWaterMaterial({ refraction })`, `MapWorld({ water: { refraction } })`, `world.setWaterRefraction`: the
  bay sees the scene behind its surface (slope-driven distortion, Beer-Lambert absorption, murk, depth foam);
  refraction off is byte-identical (shader hashes pinned); world-build pins untouched. +0.6-0.7 ms at 3024x1964
  (+0.4-0.5 at 2560x1440), the mid-frame target switch, paid whenever the bay draws.
- Samples: `?ssr=1 ?volumetric=1 ?motionblur=<s> ?dof=1[&focus=&fstop=] ?lut=warm|bleach|identity ?aonormals=gbuffer`
  on every sample (`samples/shared/fx-flags.js`, harness settings `postfx.<path>`); Canyon Run `?lamps=1` (118
  trackside lamps on `ClusteredLights`, +0.3-0.5 ms at night at 3024x1964, pose `night-lamps`, setting
  `lamps.enabled`), `?gpuparticles=1[&devilscollide=1]`
  (dust devils on `GpuParticleSystem`, seeded, pose `devils`); city `?water=refraction` + pose `shore`; procedural
  `.cube` LUTs under `samples/public/luts`; `npm run smoke` adds `canyon-night` and `city-water`.
- Tests: `ssr-ao`, `volumetric` (130), `blur-dof-lut` (128), `water` (117), `postfx-options` 141 -> 264; `npm test`
  runs 25 suites.

## 0.7.0 — 2026-10-29
Render foundations (phase 3 of the 1.0 roadmap, ENGINE.md "Render pipeline (0.7.0)"): a G-buffer path with TAA,
clustered lighting for hundreds of lights, 2-4 shadow cascades with a PCSS kernel, and GPU particles. Every knob is
opt-in and the forward path renders bit-identically to 0.6.0 (same-page A/B diffs 0 / 0.0005 mean).
- Render: `PostFX({ gbuffer: false, aa: 'auto' | 'msaa' | 'fxaa' | 'taa', taa: { blend, clamp, sharpen, filter,
  historyFormat, stableDepth }, jitter })` — the MRT scene pass (`GBuffer`: colour + octahedral normal / roughness /
  metalness / flags + velocity), `CameraHistory` jitter and matrices, `TAAPass` (Catmull-Rom history, variance
  clip, velocity or depth reprojection), `postfx.set(path, value)` knobs, `atmosphere: null` support. Canyon Run at
  3024x1964 (FXAA default): G-buffer + FXAA +0.3-0.5 ms with identical colour, G-buffer + TAA +2.2-2.5 ms (10.65 /
  11.09 / 11.38 / 7.83 ms vs 8.25 / 8.62 / 8.80 / 5.60); at 2560x1440 (4x MSAA default) the G-buffer path is
  cheaper than MSAA: + FXAA -1.2-1.7 ms, + TAA -0.2-0.4 ms. `taa.{stableDepth, reject, snap, velocityFirst}` are
  measured knobs, all off (≤ 0.15 ms either way on static poses).
  `InstancedPool({ velocity: true })` + `postfx.trackPool(pool)` and `postfx.track(mesh)` give moving objects their
  own velocity (PostFX keeps the lists and replays them whenever the G-buffer is rebuilt).
- Render: `ClusteredLights` — up to 1024 point / spot lights in a froxel grid, +2 texture units, a global chunk
  patch under `ENGINE_CLUSTERED` calling three's own `RE_Direct`: 8 clustered lights match 8 real PointLights
  pixel-exactly. World: `MapWorld({ lights: { clustered: true, clusteredRange } })` and `NearestLights({ mode:
  'clustered' })` light every lamp in range instead of the nearest 8: the night city with every lamp within 220 m
  costs +0.6 / +1.0 / 0.0 ms over the pool at 3024x1964 (11.69 / 12.43 / 11.28 vs 11.12 / 11.46 / 11.29).
  `world.setLightMode` switches at runtime; `npm run smoke` runs the clustered night page.
- Render: `shadows.cascades: 1-4` (middle cascades, round-robin updates; 3: +0.2-0.5 ms, 4: +0.3-0.9 ms, pixel
  change 0.003-0.017 mean) and `kernel: 'pcss'` (blocker search + contact-hardening penumbra: 9.1-10.5 ms vs
  'fast' 8.2-8.8 and 'pcf' 9.9-10.9 in Canyon Run at 3024x1964); `atmo.shadowCascadeInfo(out)`; the 2-cascade
  'pcf' chunk text is byte-identical to 0.6.0 (pinned by `test/shadows.mjs`).
- Render: `GpuParticleSystem` — the `ParticleSystem` API over GPUComputationRenderer (65k particles, deterministic
  slots, depth collision), matches the CPU system to 1e-5 m; 0.2 ms CPU for 65k live + 4096 spawns per frame.
- Samples: `?gbuffer=1`, `?aa=`, `?cascades=`, `?shadows=pcss` on every sample (the car, the character and the
  sandbox's instanced bodies are velocity sources); city `?clustered[&range=]`;
  harness `quiet` hooks reset the TAA history and converge 32 frames so captures reproduce.
- Tooling: samples freeze the loop at `setReady()` under `?bench=1` and every harness setting has a getter, so
  captures reproduce across page loads (Canyon Run's car poses differed by up to 0.28 mean before) and
  `__sample.run(plan)` restores knobs between settings (see ENGINE.md "Measurement protocol"); `npm run smoke`
  gains a `city-clustered` night page with an in-page clustered-lights check.
- Tests: `gbuffer-chunks` (251 checks), `clustered` (93), `shadows`, `postfx-options` (141), `gpu-particles`
  (63), `world-lights` (63); `npm test` runs 21 suites.

## 0.6.0 — 2026-10-29
Unified XPBD rigid-body dynamics (phase 2 of the 1.0 roadmap, ENGINE.md "Physics" / "DynamicsWorld"): a pure-JS,
deterministic, Node-testable solver that plugs into the existing `PhysicsWorld` so the vehicle and character
controllers collide with rigid bodies without changes. Opt-in: nothing moves unless a game creates a
`DynamicsWorld`; `npm test` keeps the roadster model bit-identical and the 0.5.0 samples render identically.
- Physics: `DynamicsWorld` — bodies (dynamic / static / kinematic) with sphere, capsule, box, convex hull, static
  triangle mesh and compound shapes; substepped XPBD with manifold contacts, Coulomb friction with persistent
  anchors, restitution, damping, islands and sleeping, kinematic targets that push, forces / impulses, `began` /
  `ended` contact events, ray / shape casts and overlaps, `serialize() / from()` (bit-identical). Joints: fixed,
  ball, hinge (limits, motor, servo), slider, distance (spring / rope), d6 (swing cone, twist), `breakForce`,
  solved exactly per joint tree. `attachToPhysicsWorld(dyn, physics)` bridges bodies into the query hub as
  `kind: 'rigid'` records (layer bit 25). `DynamicsClient` runs the same API in a module Worker (`worker: true`)
  or in-thread. `PhysicsDebug` (render layer) draws shapes, contacts and joints. 7 new headless suites
  (`test/dyn-*.mjs`, 600+ checks: analytic contacts to 1e-15, pendulum period within 0.2 %, chain gap 1e-6 m,
  10-stack drift 3e-15 m, 1000-body pit determinism hash, worker round trip bit-identical).
- Samples: `samples/sandbox/` at `/sandbox/` — scenes `stack`, `joints`, `chain`, `ragdoll`, `car` (the roadster
  controller among crates, `?solid=0` bulldozes them through a kinematic proxy) and `bench1000` (`?bodies=`);
  `?worker=1`, `?debug=1` (F4), `?substeps=`, `?seed=`, bench poses `overview` / `close`. Measured on the M4 in
  Chrome: 1000 mixed bodies 6.5 ms per step falling, 9.5 ms with 960 awake at 60 Hz, worker 10.7 ms off-thread
  with 0.5 ms on the main thread; 5 ragdolls sleep within 17 s; bench poses reset + hand-step so captures
  reproduce across loads.
- Physics: `PhysicsWorld` reserves `layerOf('rigid')` = bit 25; `CharacterController` and the rigs are unchanged
  (bridged bodies are walls to them; two-way coupling is phase 5).

## 0.5.0 — 2026-10-29
General-engine foundations (phase 1 of the 1.0 roadmap, ENGINE.md "Core services", "Character stack",
"Measurement protocol"): the services a non-driving game needs, a third-person character stack, and the tooling
that turns the manual measurement protocol into scripts. Every addition is opt-in; `Engine`, Canyon Run and the
city page render pixel-identically to 0.4.0 (captures hash-equal across page loads with the pinned protocol) and
`npm test` keeps the roadster vehicle model bit-identical.
- Core: `Events` (typed emitter, positional `emit`, allocation-free, safe add/remove during emit), `Clock`
  (`engine.clock`: `timeScale`, `paused`, `step(n)`; the Engine scales the system dt and the fixed accumulator,
  the profiler and adaptive resolution keep the wall dt), `Entity` / `EntitySystem` (thin component layer over
  `Object3D`, dense per-phase arrays), `Engine` systems gain `priority` (ordering) and `engine.setSystemEnabled(s, false)` (skip);
  defaults reproduce 0.4.0 exactly. `test/core.mjs` (107 checks).
- Core: `Assets` loader registry (cache, URL dedupe, refcounts, weighted progress, `assets.system()` for
  `engine.load`) with `registerRenderLoaders` (glTF + KTX2 / Draco / meshopt, texture, HDR with PMREM, `.cube`
  LUT -> `Data3DTexture`) and `registerAudioLoader` / `SamplePlayer` (decoded samples through the existing bus and
  positional emitters; decoding runs in an OfflineAudioContext before the bus starts, so loading never waits for
  a gesture; `disposeInstance(root)` frees a clone's skeletons). `test/assets.mjs` (119 checks).
- Core: `MathDet` deterministic math (`mulberry32`, `hashU32 / hashF64 / hashF64Array`, `detAtan2 / detAcos /
  detAsin / detAtan` from + - * / sqrt only, max error 4.4e-16) for simulation code that must be bit-identical on
  every JS engine.
- Physics: `CharacterController`, a kinematic capsule over the `PhysicsWorld` query API: step climbing, slope
  limit, wall slide, ground snapping, coyote time, jump buffering (the jump edge is detected internally), vehicle
  shove, crouch via `setHeight(h, world)` / `canStand` with auto-crouch under a ceiling; no trig in the step path;
  `test/character-sim.mjs` (134 range checks incl. a golden determinism hash, 12000 steps in ~6 ms).
- Camera: `ThirdPersonRig` and `FirstPersonRig` on `CameraRig` (damped follow, ChaseRig-style collision pull-in,
  ground clamp, `turn(dyaw, dpitch)` in radians, yaw basis for camera-relative movement, head bob).
  `test/rigs.mjs` (83 checks).
- Input: `CharacterInput` (`CHARACTER_ACTIONS` + touch layout; look deltas in radians, live only under pointer
  lock or a right-button hold), `PointerLock` (`request() -> Promise<boolean>`, `releasedByUser`) and
  `InputMap.readSources(name, out)` (per-kind raw sums of an action's bindings). `test/character-input.mjs`
  (45 checks).
- Anim (new layer, `@vibe/engine/anim`): `AnimationController` (states, 1D blend trees with cycle sync,
  transition table, markers, `end` / `loop` / `marker` events, root motion extracted from the hips with the
  horizontal track neutralised on cloned clips; a finished one-shot holds its last frame by default), `TwoBoneIK`,
  `CCDIK` / `makeIKChain` over three's CCDIKSolver. `test/anim.mjs` (78 checks).
- Samples: `samples/` is now a multi-page vite root (`/`, `/canyon-run/`, `/city/`, `/character/`; the dev-only
  `/city` plugin is gone, `vite build` emits every page). New `samples/character/`: third-person playground with
  the CC0 RobotExpressive glTF (`node tools/fetch-models.mjs`, never committed) or a procedural rigged fallback
  built in code, Entity components for motor / animator / facing, pointer-lock mouse look, three rigs.
- Tooling: `createBenchHarness` (`window.__sample` on every sample: poses, GPU-synced bench, captures, diffs,
  `run(plan)`), `npm run smoke` (headless Chrome: boot, no console errors, two equal captures per pose) and
  `npm run bench` (headed Chrome, writes `bench/results/*.json`) through Playwright; `npm test` = `test/all.mjs`;
  `npm run types` emits `types/` from the JSDoc; GitHub Actions CI runs test + build + types.
- Measurement protocol: captures reproduce across page loads only with the sky time, cloud offset, wind time and
  dither-fade view position pinned, exposure frozen and poses run in a fixed order from a fresh load (a stale
  `uViewPos` alone moved 2-5 % of Canyon Run's pixels). The harness hooks in both samples do this.

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
