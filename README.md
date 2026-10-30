# inbrowser-game-engine

Reusable browser game engine on Three.js / WebGL2, extracted from the Roadster flying-car game
([roadster](https://github.com/BrandeisPatrick/roadster), which embeds this repository as its `engine/`
submodule). Design notes, measurements and the record of decisions: [ENGINE.md](ENGINE.md). Layers: core, physics,
render, anim, input, camera, audio, ui, world. Published as the npm package `@vibe/engine` (MIT); games import it as
a normal package:

```js
import { Engine, PhysicsWorld, VehicleController, Atmosphere, PostFX } from '@vibe/engine';
import { MapWorld } from '@vibe/engine/world';      // subpath per layer: core, physics, render, anim, input, camera, audio, ui, world
```

## Install
- As a git submodule of a game (what roadster does): `git submodule add https://github.com/BrandeisPatrick/inbrowser-game-engine.git engine`,
  list `engine` in the game's `workspaces`, and `npm install` links `node_modules/@vibe/engine -> engine`.
- From git: `npm install github:BrandeisPatrick/inbrowser-game-engine` (pure ES module source, no build step).
- Standalone: `npm install`, then `npm run dev` serves the samples at http://localhost:5190 (Canyon Run at
  `/canyon-run/`; knobs: `?fps=60` tick cap, `?shadows=pcf|fast|pcss`, `?cascades=2|3|4`, `?gbuffer=1`,
  `?aa=taa|fxaa|msaa`, `?msaamax=`, `?procworkers=0`, city `?clustered`, 0.8.0: `?ssr=1`, `?volumetric=1`,
  `?motionblur=1`, `?dof=1`, `?lut=warm`, canyon `?lamps=1&t=night`, `?gpuparticles=1`, city `?water=refraction`, 0.9.0:
  sandbox `?scene=cloth|rope|soft|buoyancy|destructible`; see ENGINE.md "MacBook pass", "Render pipeline (0.7.0)",
  "(0.8.0)" and "Unified XPBD (0.9.0)"), `npm test` runs every headless suite (`test/all.mjs`, Node >= 20), `npm run build` builds the
  samples into `dist/`, `npm run types` emits `types/` from the JSDoc, `npm run smoke` / `npm run bench` drive the
  samples in Chrome (Playwright; see ENGINE.md "Measurement protocol").
- Peer dependency: `three >= 0.170`. `npm run pack:check` lists the tarball (~149 files, ~967 kB) for
  `npm publish --access public` once a registry scope exists.

## Samples and tests
- `samples/canyon-run/` — off-road buggy time trial built only on the engine; it drove 15 engine fixes
  (ENGINE.md, "Changelog / sample findings"). Without Poly Haven texture sets under its public dir it falls back to
  procedural textures.
- `test/car-sim.mjs` — 90 headless checks of the Roadster vehicle model and the PhysicsWorld queries (`npm test`).
  The street-drive and traffic checks that need the San Francisco map live in the game repository (`dev/`).
- `samples/city/` — dev-only harness: the procedural city in the browser at http://localhost:5190/city/ (free-fly
  camera, post stack, F3 profiler), for seeing and measuring the world layer without map data.
- `samples/sandbox/` — physics sandbox for the XPBD `DynamicsWorld` (0.6.0): stacks, joints, chains, ragdolls, the
  roadster among crates, a 1000-body benchmark; `?worker=1` runs the solver in a Worker, `?debug=1` draws it.
- `samples/character/` — third-person character playground: a glTF character (`node tools/fetch-models.mjs`
  downloads the CC0 RobotExpressive model; `?model=` loads any other) or a procedural rigged fallback, driven by
  `CharacterController`, `AnimationController`, `ThirdPersonRig` / `FirstPersonRig` and `CharacterInput`.
- `test/core.mjs`, `test/assets.mjs`, `test/anim.mjs`, `test/rigs.mjs`, `test/character-sim.mjs` — headless checks
  of the 0.5.0 core services, loaders, animation layer, character rigs and the kinematic character controller.
- `test/dyn-*.mjs` — headless checks of the 0.6.0 dynamics: shapes, broadphase, narrowphase, joints, the world,
  the PhysicsWorld bridge and the worker protocol (golden determinism hashes).
- `test/world-build.mjs` — builds the procedural city headless (`MapWorld` with `renderer: null`, no DOM: the canvas
  painters run against `test/helpers/dom-shim.mjs`) and pins its stats, physics queries and 10 s of deterministic
  traffic (`npm test` runs it after car-sim; ~2 s).

## Layers
- `src/core/` — `Engine` (fixed-step loop, systems, adaptive resolution, `clock`), `Profiler`, `Assets`, `Events`,
  `Clock`, `Entity` / `EntitySystem`, deterministic math (`MathDet`), the bench harness (`createBenchHarness`).
- `src/anim/` — `AnimationController` (states, blend trees, transitions, root motion over `AnimationMixer`),
  `TwoBoneIK`, CCD IK helpers.
- `src/physics/` — `PhysicsWorld` (query hub), `VehicleController`, `CharacterController`, and `dynamics/`: the
  unified XPBD rigid-body solver (`DynamicsWorld`, shapes, joints, queries, `attachToPhysicsWorld`,
  `DynamicsClient` worker).
- `src/world/` — data-driven city worlds: `MapWorld` (map + terrain + options + plugins) and its systems: `Terrain`,
  `MapData`, `RoadNetwork`, `Buildings` + `FacadeTextures`, `Props` + `Furniture` + `Vehicles`, `Traffic`,
  `Vegetation` (+ `Veg*`), `Water`, `Shore`, `Boats`, plus `MeshChunker`, `ShadowPasses`, `NearestLights`. API, plugin
  hooks and the city.json / terrain.json / farland.json contracts: ENGINE.md, "World layer".
- `tools/` — parameterised data bakers (`bake-map`, `bake-terrain`, `bake-farland`, `fetch-textures`; `--help`,
  `--dry-run`), see [tools/README.md](tools/README.md).
