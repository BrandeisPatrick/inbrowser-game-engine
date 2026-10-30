# inbrowser-game-engine

`@vibe/engine` 1.0: a browser game engine on Three.js / WebGL2, extracted from the Roadster flying-car game
([roadster](https://github.com/BrandeisPatrick/roadster), which embeds this repository as its `engine/` submodule) and
grown into a general engine: a fixed-step core with services, deterministic pure-JS physics (query hub, vehicle and
character controllers, a unified XPBD solver for rigid bodies, joints, cloth, ropes, soft bodies, buoyancy and
fracture), a measured render pipeline (forward MSAA or G-buffer + TAA, clustered lights, cascaded shadows, SSR,
volumetrics, water refraction, motion blur, depth of field, LUTs, GPU particles), animation (state machine, IK,
ragdolls), input, camera rigs, synth audio, UI widgets and a data-driven city world. Design notes, measurements and the
record of decisions: [ENGINE.md](ENGINE.md); releases: [CHANGELOG.md](CHANGELOG.md). MIT.

```js
import { Engine, PhysicsWorld, DynamicsWorld, CharacterController, Atmosphere, PostFX } from '@vibe/engine';
import { Cloth, FracturedBody } from '@vibe/engine/physics';   // one subpath per layer: core, physics, render, anim, input, camera, audio, ui, world
```
Pure ES modules; `three >= 0.170` is the only (peer) dependency; TypeScript declarations emitted from the JSDoc ship
with the package and are exported for the root and every subpath.

## Install
- From git: `npm install github:BrandeisPatrick/inbrowser-game-engine` (no build step; `prepack` emits `types/`), or
  as a git submodule listed in the game's `workspaces` (what roadster does: `node_modules/@vibe/engine -> engine`).
- Standalone: `npm install`, then `npm run dev` serves the samples at http://localhost:5190 (an index page lists the
  four). `npm test` runs the 31 headless suites (`test/all.mjs`; Node >= 20 runs them, the allocation budgets are
  calibrated on Node 26, which CI uses), `npm run build` builds the samples into `dist/`, `npm run types` emits
  `types/` and `npm run types:check` compiles a strict TypeScript consumer against them, `npm run smoke` /
  `npm run bench` drive the samples in Chrome (Playwright; see ENGINE.md "Measurement protocol"), `npm run pack:check`
  lists the tarball (~285 files, ~1.1 MB).

## Samples
Every sample exposes `window.__sample` (poses, settings, `run(plan)`) under `?bench=1&size=WxH`, and they share the
render flags of `samples/shared/fx-flags.js`: `?gbuffer=1`, `?aa=taa|fxaa|msaa`, `?shadows=pcf|fast|pcss`,
`?cascades=2|3|4`, `?ssr=1`, `?volumetric=1`, `?motionblur=1`, `?dof=1`, `?lut=warm|bleach`, `?aonormals=gbuffer`,
`?fps=60`.
- `samples/canyon-run/` — off-road buggy time trial built only on the engine (it drove 15 engine fixes, ENGINE.md
  "Changelog / sample findings"); `?lamps=1&t=night` lights the canyon with clustered lamps, `?gpuparticles=1` runs
  the dust devils on the GPU. Without Poly Haven texture sets under its public dir it falls back to procedural textures.
- `samples/city/` — the procedural city without map data (free-fly camera, traffic, streaming, post stack, F3
  profiler) at http://localhost:5190/city/; `?clustered`, `?water=refraction`, `?notraffic`, `?nostream=1`.
- `samples/sandbox/` — the physics sandbox:
  `?scene=stack|joints|chain|ragdoll|car|bench1000|cloth|rope|soft|buoyancy|destructible`, `?worker=1` runs the rigid
  scenes in a Worker, `?debug=1` draws the solver.
- `samples/character/` — third-person character playground: a glTF character (`node tools/fetch-models.mjs`
  downloads the CC0 RobotExpressive model; `?model=` loads any other) or a procedural rigged fallback, driven by
  `CharacterController`, `AnimationController`, `ThirdPersonRig` / `FirstPersonRig` and `CharacterInput`; it pushes
  crates, rides an elevator, mantles ledges with hand IK and ragdolls on hard falls (K).

## Tests (`npm test`, 31 suites)
- `test/car-sim.mjs` — 90 range checks of the Roadster vehicle model and the PhysicsWorld queries: the determinism
  gate, bit-identical since 0.4.0. The street-drive and traffic checks that need the San Francisco map live in the
  game repository.
- `test/world-build.mjs` — builds the procedural city headless (`MapWorld` with `renderer: null`, the canvas painters on
  `test/helpers/dom-shim.mjs`) and pins its stats, physics queries and 10 s of deterministic traffic; `world-lights`.
- `test/core.mjs`, `assets`, `rigs`, `character-input`, `anim`, `character-sim` — the core services, loaders, camera
  rigs, character input, animation layer and the kinematic character controller.
- `test/dyn-*.mjs` — the dynamics: shapes, broadphase, narrowphase, joints, world, PhysicsWorld bridge, serialize /
  worker protocol (golden hashes), cloth, soft bodies, buoyancy, fracture; `character-dyn`, `ragdoll`.
- `test/gpu-particles.mjs`, `clustered`, `shadows`, `postfx-options`, `gbuffer-chunks`, `ssr-ao`, `volumetric`,
  `blur-dof-lut`, `water` — the render pipeline's shader anchors, options and defaults (shader texts pinned by hash).
- `test/types/check.ts` — the emitted declarations compile for a strict TypeScript consumer (`npm run types:check`).
- `test/browser/smoke.mjs` (12 pages, two identical captures per pose) and `test/browser/bench.mjs` need Chrome and
  run locally; CI (`.github/workflows/ci.yml`, Node 26) runs `npm test`, the build and `types:check`.

## Layers
- `src/core/` — `Engine` (fixed-step loop, systems, adaptive resolution, `clock`), `Profiler`, `Assets`, `Events`,
  `Clock`, `Entity` / `EntitySystem`, deterministic math (`MathDet`), the bench harness (`createBenchHarness`).
- `src/physics/` — `PhysicsWorld` (query hub), `VehicleController`, `CharacterController` (pushes and rides dynamics),
  and `dynamics/`: the unified XPBD solver (`DynamicsWorld`, shapes, joints, queries, `attachToPhysicsWorld`,
  `DynamicsClient` worker; particles, `Cloth`, `Rope`, `SoftBody`, `ParticleCollider`, `Buoyancy`, `FracturedBody`).
- `src/render/` — `Atmosphere`, `PostFX` (G-buffer + TAA, GTAO, SSR, volumetrics, motion blur, DoF, bloom, LUT, FXAA /
  MSAA), `ClusteredLights`, shadow cascades and kernels, `GpuParticleSystem`, instancing / LOD / impostor utilities,
  `TextureSets` (KTX2), `SoftMeshes`, `FractureMesh`, `PhysicsDebug`.
- `src/anim/` — `AnimationController` (states, blend trees, transitions, root motion over `AnimationMixer`),
  `TwoBoneIK`, `CCDIK`, `Ragdoll`.
- `src/input/`, `src/camera/`, `src/audio/`, `src/ui/` — `InputMap` with the vehicle and character presets and
  `PointerLock`; `CameraRig` with chase, orbit, free-fly, third-person and first-person rigs; `AudioBus`, synth voices,
  `SamplePlayer`, positional emitters; DOM HUD widgets.
- `src/world/` — data-driven city worlds: `MapWorld` (map + terrain + options + plugins) and its systems: `Terrain`,
  `MapData`, `RoadNetwork`, `Buildings` + `FacadeTextures`, `Props` + `Furniture` + `Vehicles`, `Traffic`,
  `Vegetation` (+ `Veg*`), `Water` (Gerstner waves, refraction), `Shore`, `Boats`, plus `MeshChunker`, `ShadowPasses`,
  `NearestLights`. API, plugin hooks and the city.json / terrain.json / farland.json contracts: ENGINE.md, "World layer".
- `tools/` — parameterised data bakers (`bake-map`, `bake-terrain`, `bake-farland`, `fetch-textures`, `fetch-models`;
  `--help`, `--dry-run`), see [tools/README.md](tools/README.md).
