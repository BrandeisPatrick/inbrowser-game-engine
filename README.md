# inbrowser-game-engine

Reusable browser game engine on Three.js / WebGL2, extracted from the Roadster flying-car game
([flying-tesla](https://github.com/BrandeisPatrick/flying-tesla), which embeds this repository as its `engine/`
submodule). Design notes, measurements and the record of decisions: [ENGINE.md](ENGINE.md). Layers: core, render,
physics, world, input, camera, audio, ui. Published as the npm package `@vibe/engine` (MIT); games import it as a
normal package:

```js
import { Engine, PhysicsWorld, VehicleController, Atmosphere, PostFX } from '@vibe/engine';
import { MapWorld } from '@vibe/engine/world';      // subpath per layer: core, physics, render, input, camera, audio, ui, world
```

## Install
- As a git submodule of a game (what flying-tesla does): `git submodule add https://github.com/BrandeisPatrick/inbrowser-game-engine.git engine`,
  list `engine` in the game's `workspaces`, and `npm install` links `node_modules/@vibe/engine -> engine`.
- From git: `npm install github:BrandeisPatrick/inbrowser-game-engine` (pure ES module source, no build step).
- Standalone: `npm install`, then `npm run dev` serves the Canyon Run sample at http://localhost:5190, `npm test`
  runs the headless vehicle physics checks (Node >= 20), `npm run build` builds the sample into `dist/`.
- Peer dependency: `three >= 0.170`. `npm run pack:check` lists the tarball (~100 files, ~440 kB) for
  `npm publish --access public` once a registry scope exists.

## Samples and tests
- `samples/canyon-run/` — off-road buggy time trial built only on the engine; it drove 15 engine fixes
  (ENGINE.md, "Changelog / sample findings"). Without Poly Haven texture sets under its public dir it falls back to
  procedural textures.
- `test/car-sim.mjs` — 73 headless checks of the Roadster vehicle model and the PhysicsWorld queries (`npm test`).
  The street-drive and traffic checks that need the San Francisco map live in the game repository (`dev/`).

## Layers
- `src/world/` — data-driven city worlds: `MapWorld` (map + terrain + options + plugins) and its systems: `Terrain`,
  `MapData`, `RoadNetwork`, `Buildings` + `FacadeTextures`, `Props` + `Furniture` + `Vehicles`, `Traffic`,
  `Vegetation` (+ `Veg*`), `Water`, `Shore`, `Boats`, plus `MeshChunker`, `ShadowPasses`, `NearestLights`. API, plugin
  hooks and the city.json / terrain.json / farland.json contracts: ENGINE.md, "World layer".
- `tools/` — parameterised data bakers (`bake-map`, `bake-terrain`, `bake-farland`, `fetch-textures`; `--help`,
  `--dry-run`), see [tools/README.md](tools/README.md).
