# Browser game engine — extracted from Roadster

> This document moved with the engine into its own repository (`inbrowser-game-engine`, npm package `@vibe/engine`).
> Paths written as `engine/src/...` are relative to a game that embeds the engine as `engine/` (roadster does);
> here they are `src/...`. Plain `src/`, `dev/` and `tools/` paths refer to the roadster game, `samples/canyon-run`
> and `test/car-sim.mjs` live here.

Goal: turn the systems built for the flying-car game into a reusable, optimised engine for
**browser games** (WebGL2 via Three.js now, WebGPU later), with first-class lighting and physics,
usable for this driving game and for other genres. Zero build-time dependencies beyond `three`;
optional heavier backends (Rapier WASM physics, KTX2 textures) are plug-ins.

Code lives in `engine/` (imported by the game as `../engine/src/index.js`; later published as
`@vibe/engine`). The game in `src/` becomes a thin consumer: map data, car tuning, HUD skin.

## Layers (each folder is independently importable; lower layers never import higher ones)

| Layer | Folder | What moves in from the game | New |
|---|---|---|---|
| core | `engine/src/core` | fixed-step loop (120 Hz, max substeps), resize, adaptive pixel ratio (`main.js`) | `Engine` (renderer + scene + systems registry + render interpolation), `Profiler` (CPU timers per system, GPU timer queries), `Events`, `Clock`, debug overlay |
| render | `engine/src/render` | `sky.js` (sun/sky/clouds/fog/time of day), PostFX (GTAO, bloom, SMAA, grading), shadow cascades, texture-set loader + canvas fallbacks (`textures.js`), vegetation impostor baker, facade texture arrays | `InstancedPool` + `LODGroup` + `Impostor` utilities, `MaterialLib`, `DecalTrail`, pooled `Particles`, `Culling` (frustum + distance per chunk) |
| physics | `engine/src/physics` | `Physics.js` vehicle model (arcade-realistic, deterministic, Node-testable), AABB `ColliderGrid`, heightfield queries, OBB-vs-AABB SAT, sub-stepped sweeps | `PhysicsWorld` interface (static colliders, heightfield, dynamic AABBs, raycast/spherecast), `VehicleController` (wheels + thrusters as a controller on any body), optional `RapierBackend` for general rigid bodies |
| world | `engine/src/world` | **extracted**: terrain heightfield + draping, map data + procedural fallback, road network, building builder + facade layers, props / furniture / vehicle library, vegetation, traffic AI, water / shore / boats / far relief; bakers in `engine/tools` (the Bay Bridge and the landmarks stay in the game as MapWorld plugins) | `MapWorld` (options + plugin hooks), data contracts for city / terrain / farland JSON, parameterised bakers; `TileStreamer` (building tiles built / freed around the camera, see "Streaming"); traffic AI in a Web Worker (`TrafficSim` core, see "Traffic simulation") |
| input | `engine/src/input` | `Input.js` (keyboard/gamepad, analog smoothing, edge flags) | touch/virtual sticks, rebinding |
| camera | `engine/src/camera` | `ChaseCamera` (modes, occlusion avoidance) | `CameraRig` base, orbit/free/cinematic rigs |
| audio | `engine/src/audio` | WebAudio synth bus (`Audio.js`) | `AudioBus` (mixer, positional emitters, synth voices) |
| ui | `engine/src/ui` | HUD widgets (gauges, minimap, flash, attitude indicator) | DOM widget kit, theme |

## Engine API (target)
```js
import { Engine, Atmosphere, PostFX, PhysicsWorld, VehicleController, Terrain, Input, ChaseCamera } from '../engine/src/index.js';
const engine = new Engine({ container, fixedHz: 120, maxSubsteps: 4, targetFps: 60, adaptiveResolution: true, shadows: 'cascaded', postfx: true });
engine.addSystem(world);          // { load?(progress), fixedUpdate?(dt), update?(dt, alpha), dispose?() }
engine.addSystem(vehicle);
engine.camera.rig = new ChaseCamera(engine.camera, { target: vehicle.object3d, physics: engine.physics });
await engine.load(onProgress);
engine.start();
```
- `fixedUpdate(dt)` runs at the fixed rate; `update(dt, alpha)` runs per frame with interpolation factor.
- `engine.physics` is a `PhysicsWorld` (static AABB grid + heightfield + dynamic bodies + raycasts).
- `engine.profiler.frame` gives per-system CPU ms and per-pass GPU ms (when `EXT_disjoint_timer_query_webgl2` exists).

## Lighting
- Atmosphere: physically based sky, sun disc + lensflare, procedural cloud layer, height fog, time of day (hours or presets), moon/stars; PMREM environment refreshed on time change; emissive scaling by darkness.
- Shadows: camera-following cascaded shadow maps (near sharp / far soft), texel snapping to stop shimmer, per-material bias presets.
- Post: GTAO (half-res + denoise), bloom (threshold per time of day), SMAA, filmic grade, vignette, optional height fog pass. Everything resolution-aware.
- Local lights: nearest-N real point lights + emissive/fake light pools for the rest; headlight cones.

## Physics
- Default backend: pure JS, deterministic, runs in Node (so sims/tests run headless): heightfield ground, static AABB grid (buildings, props), dynamic AABB set (traffic), OBB sweeps for fast bodies, friction/restitution, damage callbacks.
- `VehicleController`: sprung body on four spring / damper units with anti-roll and bump stops (axle loads, load transfer, squat / dive / body roll, kerbs and landings all come out of them), tyre slip model, speed-sensitive steering, boost, thruster flight (armed by a lift input: a jump is just a jump; fly-by-wire auto-level, hover assist, fuel), landing/impact events. Parameterised by a spec object so other cars/bikes/boats/drones can reuse it. See "Suspension & contact model" below.
- Optional `RapierBackend` (`@dimforge/rapier3d-compat`, Apache-2.0, ~2 MB WASM) for general rigid bodies, joints and ragdolls when a game needs them; same `PhysicsWorld` interface.

## World layer (`engine/src/world`, extracted)
`MapWorld` composes a whole city from map data + options + plugins; every system is also usable on its own.

| Module | Public API (main exports) |
|---|---|
| `MapWorld.js` | `MapWorld` (below), `DEFAULT_WATER_LEVEL`, `DEFAULT_STREAMING` |
| `TileStreamer.js` | `TileStreamer({ budgetMs, maxConcurrent, uploadBytesPerFrame, lookAhead, urgentRadius, altitudeScale })` (`addLayer`, `addTile`, `update(view, dt)`, `prewarm(view, { radius })`, `stats`), see "Streaming" |
| `MapData.js` | `loadMapData(url)`, `validateMapData(data)`, `prepareMapData(data, spawnOpts)` -> `{ offset, bounds, outerBox, spawn }`, `findSpawn(data, bounds, { road, at })`, `generateProceduralCity(seed, { waterfrontName, center })`, `MAP_KEYS`, `assetBase()` |
| `Terrain.js` | `Terrain` (`load(url)`, `setData(json)`, `setOffset`, `heightAt`, `rawHeightAt`, `normalAt`, `attachSurface`), `GroundSurface` (nested lattice; `applyCorridors`, `finalize`, `heightAt`, `normalAt`), `sampleLevel`, `drapePolygon`, `clipConvex`, `latticeSplit` |
| `RoadNetwork.js` | `RoadNetwork({ roads, rail, raster, isWater, surface, deckHolds })` (`prepare`, `build`, `meshes`, `list`, `deckHash`, `segHash`, `corridors`, `heldDecks`), `roadStyle`, `RBIT`, `createRoadMaterial`, `createTactileMaterial`, `SolidBuf`, `samplePolyline`, `quadUp`, `DEFAULT_DECK_HOLD` |
| `Buildings.js` + `FacadeTextures.js` | `BuildingBuilder(region, { groundAt, streetAt, offset, baseGrid, detailGrid, lodDist, detailFloors, maxWindows, style, nearDetail, grid })` (`add(b, rnd, footprint?)`, `footprint(b)`, `build`), `TileBuildingBuilder(region, opts, 'low' \| 'full' \| 'near')` (`meshes(mats)` generator), `NullBuf`, `DEFAULT_BUILDING_STYLE` (`oldProb(xb, zb)`, `victorianProb(xb, zb)`, `namedRe`), `createFacadeMaterials`, `GeoBuf`, `MODE`, `enc`, `tintOf`, `insetRing`, `offsetRing`; `FACADE`, `WIN_RECTS`, `makeFacadeTextures`, `PBR`, `PBR_SETS`, `GROUND_SETS`, `loadPbrTextures` |
| `Props.js` + `Furniture.js` + `Vehicles.js` | `Props({ net, raster, colliders, data, isLand, groundAt, seed, streets, signStyle, treeRules, treeMix })`; furniture library (`makeFurnitureGeometry`, `FURNITURE_META`, `buildSignAtlas(names, style)`, `DEFAULT_SIGN_STYLE`, materials); vehicle library (`makeVehicleGeometry(type, { lod, livery })`, `VEHICLE_SPECS`, `pickVehicleColor`, `createVehicleMaterial`, `DEFAULT_BUS_LIVERY`, `PartBuilder`) |
| `Traffic.js` | presenter: `Traffic({ world, data, net, props, seed, count, camera, busLivery, worker })` (`build`, `ready`, `update`, `queryColliders`, `setPlayer`, `setCamera`, `summary`, `pose(i)`, `mode`, `perf`, `sim`), `trafficWorkerDefault()` |
| `TrafficSim.js` + `traffic.worker.js` | simulation core (no scene objects): `TrafficSim({ roads, signals, groundAt \| baked, spawn, seed, count })` (`build`, `update(dt)`, `setView`, `setPlayer`, `readSignals`, `writePoses`, `queryColliders`, `summary`, `bakeError`, `lanes`, `vehicles`), `TrafficSim.roadsFrom(net)`, `buildSignalInfos({ data, net, props })`, `playerInfoFrom(src)`, `TRAFFIC_TYPES`, `POSE_STRIDE`, `typeCounts`; the module worker that runs it |
| `Vegetation.js` (+ `VegModels.js`, `VegTextures.js`, `VegMaterials.js`) | `buildTreeMeshes(placements, { groundAt, colliders, seed, rules, mix })`, `bakeVegImpostors`, `SPECIES`, `makeVegAtlas`, `createVegMaterial`, `createVegImpostorMaterial` |
| `Water.js` / `Shore.js` / `Boats.js` | `Water`, `createWaterMaterial`, `buildShoreField`, Gerstner `waveHeight` / `waveSlope`; `Waterfront({ world, renderer, level, waterMat, pondMat, groundMat, ferryDock, islandFill, boats, farland })`, `createGroundMaterial`, `bakeLandClass`, `buildFarLand(json, world, level, opts)`; `Boats({ ..., ferryDock, ferryRoute, ferries, shipRoute, sailZones, sailCount, bridge })` |
| `geom.js` | `hashStr`, `signedArea`, `pointInRing`, `RingIndex` (exact row-bucketed pointInRing for big rings), `pointInPoly`, `ringBounds`, `centroid`, `sanitizeRing`, `polylineLength` (+ re-exports of `render/noise.js`) |
| `MeshChunker.js`, `ShadowPasses.js`, `NearestLights.js` | tiled static meshes, per-cascade shadow casters, nearest-N real lights (see Conventions) |

```js
const world = new MapWorld({
  scene, renderer,
  data: { mapUrl, terrainUrl, farlandUrl },            // or data.map = a preloaded city.json; farlandUrl: false = none
  timeOfDay: 'golden', atmosphere: { /* render/Atmosphere.js options */ }, waterLevel: -0.6,
  spawn: { road: /^Main Street$/ },                      // or { at: { x, z, heading } } (bake frame)
  roads: { deckHolds: [{ name: /bay bridge/i, top: 58, lowerGap: 9 }] },
  buildings: { style: { oldProb(xb, zb) {}, victorianProb(xb, zb) {}, namedRe } },
  props: { seed, streets: { lampStyle(name) {}, featureLamps: [{ road, style, step }], boulevard, promenade, utilityPoles },
           signStyle: { posters, manholeText, blade } },
  vegetation: { rules: [{ road: /harbor/i, species: 'palm', share: 1, radius: 30 }], mix },
  traffic: { count: 250, seed: 7, busLivery, worker: true } /* or a count, or false; worker: default = module Workers
                                                           available and no ?noworker=1 (Node: in-thread) */,
  streaming: { radius: 900, keep: 250, budgetMs: 3, spawnRadius: 500, pin(building, data) {} } /* DEFAULT_STREAMING;
                                                           false or ?nostream=1 = build everything at load */,
  shore: { ferryDock: { x, z, dir }, islandFill(landPoly) {}, boats: { ferryRoute, shipRoute, sailZones, avoidRoads },
           farland: { natural(bx, bz) {}, urbanMax(bx, bz) {}, clusters, cranes: { region, max, spacing }, blockAngle(bx, bz) {} } },
  chunk: { buildings: 320, structures: 320, ground: 640 }, filler: true, clearZones: [{ x, z, r }],
  plugins: [myBridgePlugin(), myLandmarksPlugin()], physics: false,
});
await world.load((p, label) => {});                      // or engine.addSystem(world.system({ camera }))
world.update(dt, camera.position);                       // per frame, after the camera moved
world.getGroundHeight(x, z, y?); world.queryColliders(x, z, r); world.getSpawn(); world.isLand(x, z);
world.getMinimapData(); world.sunDirection; world.setTimeOfDay('night'); world.stats; world.createPhysics();
world.streamer.stats; world.buildingTiles; world.streamLog;  // streaming: scheduler stats, resident tiles, pop-in log
world.traffic.mode; world.traffic.perf;                   // 'worker' | 'main'; main-thread ms, worker step ms, bytes
```
All game rules default to neutral behaviour (generic cobra lamps, per-class tree mixes, no ferries / ship / cranes,
a blue-striped bus, 0.4 pre-war share); Roadster passes its San Francisco values from `src/world/World.js`.

**Plugin hooks** (how game-specific structures stay in the game): a plugin is
`{ name, beforeBuildings?(ctx), afterBuildings?(ctx), afterProps?(ctx), clearZones?(), chunk?(world), update?(dt, cameraPos, darkness, world), instance? }`
with `ctx = { world, data, net, colliders, mat, tex, groundAt, renderer, offset }`. A phase hook may return an
Object3D (added to `world.group`); afterwards `world[name] = plugin.instance ?? plugin` (null and skipped from then
on if the hook threw). Phases: `beforeBuildings` (roads exist; claim OSM footprints with `b._lm = key` so the
generic builder skips them), `afterBuildings` (merged buildings exist, their colliders are registered right after;
edit merged geometry here - with streaming only the tiles built at load exist: pin the tiles you edit with
`streaming.pin`, or they come back unedited when they are rebuilt), `afterProps` (street furniture exists, e.g. add deck lamps to `world.props.lampHeads`;
traffic is built next). `clearZones()` -> bake-frame circles kept free of filler lots; `chunk(world)` ->
`[{ mesh, cell }]` tiled in the final MeshChunker pass; `update` runs after the props and before traffic. Roadster:
`bayBridgePlugin()` (src/world/bridge.js: afterProps + chunk + update) and `landmarksPlugin()` (src/world/landmarks.js:
beforeBuildings + afterBuildings + clearZones + update).

**Data contracts** (bake frame = metres around `center`, x east, z SOUTH; MapWorld shifts everything by
`world.offset` so the spawn is the origin):

| File | Shape | Baker |
|---|---|---|
| `city.json` | `{ v, source, center: {lat, lon}, size, outer, bounds: {minX, maxX, minZ, maxZ}, buildings: [{ p: ring, hl?, h, mh?, lv?, t?, r?, rh?, c?, m?, rc?, n? }], roads: [{ p: polyline, c: class, l?, o?, w?, n?, b?, ly?, x?: junction indices, g?: [grounded ends], sv? }], water / land / piers / parks (k) / parking / plazas: [{ p, hl? }], rail: [{ p, k, b?, e? }], trees / signals / lamps: [[x, z]] }` (rings `[[x, z], ...]`, 0.1 m) | `engine/tools/bake-map.mjs` (Overpass) |
| `terrain.json` | `{ origin: [x0, z0], cell, nx, nz, h: Int16 decimetres row-major (z rows, then x), outer?: { origin, cell, nx, nz, h }, center, source, datum, frame }` | `engine/tools/bake-terrain.mjs` (Terrarium) |
| `farland.json` | `{ origin: [x0, z0], cell, nx, nz, h: integer metres (water clamped to sea), sea, center, units, frame }` | `engine/tools/bake-farland.mjs` |
| `textures/manifest.json` | `{ source, resolution, sets: { role: { id, name, size: [w, h] m, maps, ktx2?, ktx2Maps?: { Diffuse\|nor_gl\|Rough\|AO: { file, bytes, codec, srgb, width, height, levels, src, srcBytes, psnr, params } } } }, ktx2? }` + `<id>/{diff,nor_gl,rough,ao}.jpg` (+ `.ktx2`) | `engine/tools/fetch-textures.mjs` (+ `compress-textures.mjs`) |

Bakers (`engine/tools/`, see its README): parameterised Node CLIs (`--lat --lon --size --outer ...`, `--out`,
`--cache`, `--refresh`, `--dry-run`, `--help`, `--config file.json`) with San Francisco defaults; each exports an
async function the game's thin wrappers in `tools/` call with the SF arguments (`npm run bake:map` etc.).

## Streaming (building tiles, `engine/src/world/TileStreamer.js`)
`MapWorld` builds the city's building geometry per tile around the camera instead of all at load (default with a
renderer; `streaming: false` or `?nostream=1` = the full build, kept for A/B and used headless).
- **Global at load** (unchanged): map data, terrain + ground, road network + lane graph, water / far land, props +
  vegetation, landmark / bridge plugins, traffic, and every building's footprint + colliders (`BuildingBuilder.footprint`
  gives exactly what `add()` builds on; the 306 k collider boxes are identical to the full build).
- **Tiles.** District tiles = the full build's 7 x 7 base chunks (same buildings, same order, same per-building seeds:
  a built tile is byte-identical to its chunk), filler-ring tiles = 640 m squares. A tile holds the building shells
  plus, per detail cell, a `THREE.LOD` with the far level (flat shells). Each detail cell of a district tile is a child
  tile that adds the near level (real window openings + instanced window units) to its parent's LOD, so near detail
  exists only around the camera. LOD positions / distances are those of the full build, so switching is unchanged.
- **Scheduler** (`TileStreamer`, generic): tiles keyed by (i, j) per layer, each with `build(level)` (a generator that
  yields after every building / geometry, so one tile is spread over several frames), `dispose()`; priority = distance
  to the layer threshold from where the camera will be in `lookAhead` s (direction of travel first); budget `budgetMs`
  per frame, `maxConcurrent` builds, results committed (meshes attached, GPU upload on the next render) within
  `uploadBytesPerFrame`; hysteresis `keep`; tiles closer than `urgentRadius` with nothing built (teleport / reset) are
  built regardless of the budget. Children only build while their parent is built and are freed with it. `stats`:
  `{ tiles, built, pending, building, builds, disposed, cancelled, msThisFrame, maxMsFrame, residentBytes }`.
- **Policy** (`DEFAULT_STREAMING`): district tiles within `radius` 900 m + `heightReach` 2.5 x the tile's tallest
  building (<= 900 m: skylines are seen from far) + `altitudeScale` 6 m per metre of camera height above the ground
  (300 m up = +1.8 km: the whole district) + `openRadius` 1.2 km over open water; filler ring within 3 km (cheap, seen
  across the bay); detail cells within `lodDist` x 1.05 + `detailMargin` 150 m of their LOD point (3D, as THREE.LOD
  measures). Freed beyond those + `keep` / `fillerKeep` / `detailKeep`. Far tiles never build near detail.
- **Load.** Only tiles within `spawnRadius` 500 m of the spawn (plus pinned ones) are built before `load()` resolves;
  the rest streams in during the first seconds of play. Plugin hooks see only those tiles: Roadster pins the tiles of
  the tall OSM blocks beside the Bay Bridge deck (`streaming.pin` in src/world/World.js), which `bridge.js` collapses
  in the merged geometry (verified: 188 vertices / 18 colliders hidden, as before).
- **Scratch buffers** of tile builds are pooled (`TileBuildingBuilder`, up to 4) while tiles stream and handed back to the
  GC after 5 s without streaming work: the flight's heap peak fell from 885 to 627 MB with the pool.
- Not streamed, by measurement: street furniture / parked cars (placement registers colliders and is order dependent;
  the batched meshes hold 4.9 MB and are already culled per instance), vegetation (near models are already a
  camera-partitioned instanced subset of 4k trees; 0.6 s of the 0.7 s "Street furniture" phase is the atlas + impostor
  bake, which is global), sidewalks / structures (the bridge plugin edits that merged mesh, and it carries the decks seen
  from km away), the outer context lots (one 0.3 MB instanced mesh).

**Measured** (main game, M4 / headless Chrome 155 at 1280x800, static build, golden hour; "before" = the tree before
this change, 3 loads each; settled = 8 s after the first frame, after a forced GC):

| | before | streaming (default) | `?nostream=1` |
|---|---|---|---|
| `world.stats.loadMs` | 3031 | 2635 | 2861 |
| time to first frame (navigation -> first engine frame) | 3902 ms | 3486 ms | 3738 ms |
| "Buildings" load phase | 543 ms | 175 ms (19 tiles + footprints / colliders of all 5263) | 379 ms |
| peak JS heap during load | 1131 MB | 889 MB | 1079 MB |
| JS heap at the spawn, settled | 559 MB | 526 MB | 554 MB |
| scene geometry (unique attribute + index bytes) at the spawn | 199 MB | 168 MB (district 40/42 tiles, detail 45/147 cells) | 199 MB |
| renderer geometries at the first frame / settled | 311 / 311 | 318 / 364 | 311 / 311 |

Scripted 60 s runs (camera only; car parked at the spawn; 4 s warm-up at the start pose, then measured; CPU ms per
`engine.tick`):

| | drive (2.4 km of streets at 40 m/s, 2.2 m eye height) | flight (3.6 km at 300 m, 60 m/s) |
|---|---|---|
| frame CPU mean / p99 / max, before | 2.83 / 3.9 / 201.4 ms (one 200 ms hitch at the same spot every run: 10-200 ms) | 2.95 / 3.9 / 4.7 ms |
| frame CPU mean / p99 / max, streaming | 2.57 / 3.9 / 9.8 ms (max - mean 7.2 ms: the same spot, not streaming) | 2.58 / 5.2 / 6.1 ms (max - mean 3.5) |
| streaming work per frame (mean / max) | 0.03 / 3.4 ms; 148 builds, 79 frees | 0.07 / 3.4 ms; 204 builds, 135 frees |
| frames > 8 ms above the mean / frame gaps > 18.9 ms | 0 / 0 | 0 / 0 |
| nearest distance a new tile appeared / late near detail | 992 m / 0 of 34 | none appeared (all resident from 300 m) / 0 of 88 |
| scene geometry at the end, before -> streaming | 199 -> 136 MB | 199 -> 169 MB |
| JS heap after GC at the end, before -> streaming (peak during the run) | 556 -> 495 MB (593 -> 596) | 555 -> 528 MB (862 -> 627) |

Fixed-pose A/B frames (no post-FX, traffic / boats / car hidden, fixed clock; base == base is bit-identical):
spawn street, a street 600 m west, an oblique district view, the 300 m flight view, the skyline from the bay and a
far district corner are pixel-identical to the full build except 0.01 % of the pixels of the flight view (coplanar
roof / shell faces now in different tile meshes, so their draw order differs); `?nostream=1` is bit-identical.

## Traffic simulation (`TrafficSim.js` core, `traffic.worker.js`, `Traffic.js` presenter)
- **Split.** `TrafficSim` = lane graph build + per-step AI, THREE math only (runs in Node: `dev/traffic-sim.mjs`
  imports it directly). `Traffic` = main-thread presenter: instanced meshes (tiered per shadow cascade), headlight
  beams, material uniforms, `queryColliders`, `setPlayer` / `setCamera`, `count`, `typeNames`, `pose(i)`.
- **Ground heights are baked** into the lane graph at build time: every lane and junction connector is sampled every
  1.5 m with `world.getGroundHeight(x, z, yRef)` walked along the lane as a vehicle queried it (bridge lanes follow
  their deck profile, ground lanes ignore decks overhead), then kept at 3 / 6 m spacing where the street is evenly
  graded (161 k floats = 0.65 MB with the real terrain). Vehicles interpolate it at their axles (y = mean, pitch from
  the difference). Measured against the live query every 0.5 m (809 k points, real terrain): p99 1.2 cm, p99.9 3.2 cm, 99.98 % within 10 cm; the
  rest sit on discontinuities of the live surface (deck seams / gaps, a water patch under a filler street), which the
  bake bridges and the old per-vehicle query jumped across. Bake: 48 ms in the browser (dev/traffic.html: H).
- **Worker** (`traffic: { worker }`, default when module Workers exist and no `?noworker=1`; Node: in-thread). The
  main thread builds the same sim (graph + bake + initial spawn: the fallback, and what dev tools read), posts the
  road data + baked heights; the worker rebuilds the identical graph (checked: max initial pose difference 0). Per
  frame ONE message each way: main -> worker `{ dt, camera position + view-projection, player footprint, the 169
  signal phases the bulbs show (copied from Props.signals[].state, so lights and AI always agree), returned pose
  buffers }` (297 B + the buffer, transferred); worker -> main a transferable `Float32Array` of poses (9 floats x 250
  = 9 kB: x, y, z, yaw, pitch, brake, speed, yaw rate, flags), double-buffered. Respawns / recycling stay in the
  worker. The presenter draws the newest snapshot extrapolated to the current frame (speed + yaw rate; one frame when
  on time, more if a snapshot is late), and `queryColliders` answers from that snapshot with the same oriented AABB
  slices (computed lazily per touched vehicle). The player's hard rule (never into its box, stop 5 m behind) uses
  the pose of the frame the step belongs to, exactly as in-thread. If the worker fails to start / answer within 5 s
  the in-thread sim takes over (logged once). Vite bundles the worker (`new Worker(new URL('./traffic.worker.js',
  import.meta.url), { type: 'module' })`, 88 kB).

**Measured** (same setup, car parked at the spawn, 10 s sampled at 50 us + a direct timer around `Traffic.update`;
per frame):

| | before | worker (default) | `?noworker=1` |
|---|---|---|---|
| main thread: `Traffic.update` | 0.55 ms | 0.077 ms | 0.49 ms |
| main thread: `queryColliders` (3 queries / frame from the car physics) | 0.007 ms | 0.010 ms | 0.007 ms |
| worker step (in the worker) | - | 0.37-0.62 ms | - |
| messages / frame | - | 1 out (297 B + returned buffer) + 1 back (9 kB, transferred) | - |
| late snapshots / 4253 | - | 0 | - |
| presented position second difference p50 / p99 (jitter) | - | 0.78 / 3.99 mm | 0.69 / 3.78 mm |
| traffic build on the main thread at load (graph + bake + spawn) | 115 ms | 158 ms (worker rebuilds in parallel: 107 ms) | 158 ms |

Behaviour checks: `node dev/traffic-sim.mjs 60 250` reproduces the old run exactly (93 respawns, 19.8 km/h, 456
junction entries at signals, 0 on red, 0 vehicle overlaps; `PLAYER=1`: 0 overlaps with the player, 1.50 m minimum
clearance); in the game (worker mode, 40 s) no vehicle ever overlapped the parked car, one queued 4.95 m behind it and
0 of 318 signal entries were on red. `?noworker=1` / a worker that throws or fails its build fall back with one warning.

## Optimisation plan (measured, not guessed)
1. Profiler first: CPU per system, GPU per pass; stats overlay (`F3`), `?profile=1` dumps a table after 10 s.
2. Render: merged static chunks with frustum + distance culling, `BatchedMesh`/`InstancedMesh` everywhere, impostors beyond LOD distance, texture arrays to cut draw calls, KTX2/Basis compression of the PBR sets (done: see "Textures" below), shadow cascade budget, half-res post passes, adaptive resolution with a frame-time controller.
3. Simulation: traffic AI in a Worker with transferable pose buffers (done: "Traffic simulation"); vegetation LOD partitioning stays on the main thread (a ~4k-tree distance pass every 20 m of camera travel); spatial hashes reused per frame; zero per-frame allocation policy enforced with an allocation counter in dev.
4. Loading: building tiles streamed around the camera, only the spawn neighbourhood at load (done: "Streaming"), texture packing in parallel with map parsing, cached bakes.
5. Budgets (M-series Mac, 1500×1300): ≤ 12 ms frame by day, ≤ 15 ms at night, ≤ 200 draw calls, ≤ 3 M triangles, load ≤ 4 s.

## Migration steps
1. Core + profiler (new, no game changes) → game loop switched to `Engine` once the post-FX agent lands.
2. Physics: move `Physics.js` + `spatial.js` into `engine/src/physics`, expose `PhysicsWorld`; game keeps `Car.js` as the visual wrapper.
3. Render: move `sky.js`, PostFX, `textures.js` loader, impostor baker; add `InstancedPool`/`LODGroup`.
4. World: move terrain/roads/buildings/props/traffic/water behind a `MapWorld` system; bakers into `engine/tools`. **Done**
   (src/world/World.js is a ~120-line SF composition; screenshots and sims identical).
5. Input/camera/audio/ui moved as-is with small API cleanups.
6. Second game sample (e.g. a drone or a boat on the same map, or a small procedural track) to prove generality.

## Textures (PBR sets, KTX2)
`engine/src/render/TextureSets.js` loads the CC0 photo sets listed in `textures/manifest.json` (fetched by
`engine/tools/fetch-textures.mjs`, 21 Poly Haven sets at 1K: `diff` / `nor_gl` / `rough` / `ao` JPGs) through two paths,
both with procedural fallbacks so nothing ever fails:
- `loadTextureSet(id, { renderer, manifest })` -> plain textures for `MeshStandardMaterial` slots (MapWorld's road,
  sidewalk, grass ground sets);
- `makeTextureArray(renderer, sets)` -> two RGBA8 `DataArrayTexture`s packed per texel (A: sRGB albedo + AO in alpha,
  tintable layers normalised to a neutral mean; N: normal xy + roughness in b) for shaders that index a layer
  (building facades, Canyon Run terrain / course / props).

**KTX2 / Basis Universal.** `engine/tools/compress-textures.mjs` (`npm run textures:compress`) writes a `.ktx2` twin
next to every JPG and records it in the manifest (`ktx2Maps`; JPGs stay as the fallback):

| Map | Codec | Notes |
|---|---|---|
| `diff` | ETC1S (BasisLZ), quality 160, sRGB | 0.27x the JPG bytes; mean PSNR 33 dB vs the JPG (noisy grass / mud / asphalt 25-27 dB, fine in game) |
| `nor_gl` | UASTC LDR 4x4 + Zstandard, linear, normal-map preset, renormalised mips | 1.36x the JPG bytes (ETC1S normals are visibly blocky); 38.6 dB |
| `rough`, `ao` | ETC1S, linear | ~0.3x the JPG bytes; 41.7 / 35.2 dB |

Every file has a full mip chain generated with wrap addressing (the sets tile) and is Y-flipped at encode time
(`KTXorientation: ru`), so a `CompressedTexture` (which cannot be flipped on upload) matches a JPG uploaded with
`flipY`. All 21 sets: 50.96 MB of JPG -> 33.76 MB of KTX2, 84 maps in 28 s wall (209 s CPU) on 8 worker threads (Apple M4).
Encoder: the official Basis Universal 2.50 encoder compiled to WASM, from the `ktx2-encoder` npm package (MIT wrapper,
Apache-2.0 Basis Universal; no `toktx` / `basisu` binary needed), driven through its low-level `BasisEncoder` API so
mip filtering, sRGB, Y flip, normal-map preset and Zstandard can be set per map; `jpeg-js` decodes the JPGs and
`ktx-parse` writes the orientation key. These three are devDependencies of the game; the runtime needs none of them.
`--uastc id/map` promotes one map to UASTC (~6x the ETC1S bytes, 42-52 dB) if ETC1S artefacts ever show; `--etc1s nor_gl`
does the opposite. The transcoder (`basis_transcoder.js` / `.wasm`, the version three's `KTX2Loader` expects) is copied
from `three/examples/jsm/libs/basis` to `public/basis/` by the same tool (`npm run textures:basis`, also `prebuild`).

**Loader.** `loadTextureSet` uses one shared `KTX2Loader` (`setTranscoderPath(basisPath())`, `detectSupport(renderer)`)
when the GPU has a compressed format: UASTC -> ASTC 4x4 (Apple / mobile) or BC7 (desktop), ETC1S -> BC7 / ETC2 / BC1
(three's priority order). Albedo sRGB, data linear, repeat wrap, max anisotropy. Downloads start before the transcoder
is ready; a missing / non-WASM transcoder, a missing / corrupt `.ktx2` or a 20 s transcode timeout falls back to the JPG
for that map. Without a compressed format (the transcoder would produce RGBA8) it uses the JPG directly.
`makeTextureArray` keeps its RGBA8 arrays: layers pack channels from several maps and tintable albedos are rescaled on
the CPU, neither of which can be done to compressed blocks, and a `CompressedArrayTexture` would also need every layer
(including the procedural plain / fabric facade layers) in one transcoded format. So it transcodes `.ktx2` to RGBA8 in
a small worker pool of its own (layer order first, workers retire after 10 s idle) and packs exactly as before; per map
it takes whichever of `.ktx2` / `.jpg` is smaller (`TextureSets.options.arrayKtx2: 'auto'`; normals mostly stay JPG,
since UASTC is usually larger and both end up RGBA8). Output differs from the JPG path only by the codec error. A future
`CompressedArrayTexture` path would need pre-packed per-layer KTX2 (albedo+AO, normal+roughness, baked tint
normalisation) from the compressor and compressed procedural layers.

**Measured** (main game, M4 / Chrome 152, 1K sets, localhost so network time ~0):

| | JPG (`?ktx2=0`) | KTX2 |
|---|---|---|
| bytes downloaded by the loader (13 facade + 5 ground sets) | 39.6 MB | 22.5 MB |
| ground textures in GPU memory (15) | 80 MB RGBA8 | 20 MB (BC7 albedo / rough, ASTC normals) |
| all loader textures in GPU memory | ~240 MB | ~180 MB (the facade arrays are 160 MB RGBA8 either way) |
| load + GPU upload, all 18 sets (isolated, warm HTTP cache, 3 runs) | 317-348 ms + 199-230 ms (original loader: 330 + 200 ms) | 326-347 ms + 86-87 ms |
| main-thread packing of the facade array | 208-239 ms | 175 ms (no canvas readback for KTX2 maps) |
| Canyon Run (3 array sets): bytes / textures ready (in-game, noisy) | 8.6 MB / ~760 ms | 4.6 MB / ~760-900 ms |

A/B screenshots (no post-FX, fixed camera, 960x640): brick close-up mean |diff| 0.73/255 (p99 4, max 12), brick facade
0.34, stucco 0.23, road asphalt 1.63 (p99 9), street 0.68; Canyon Run dirt 1.16 (p99 6), rock wall 0.88, vista 0.65;
JPG-vs-JPG reloads are bit-identical except moving traffic.

**Stats.** `TextureSets.stats` -> `{ ktx2: { formats } | false, sets: { 'set:asphalt_02': { path: 'ktx2' | 'jpg' |
'mixed' | 'procedural' | 'missing', maps, bytes, mapBytes, ms, decodeMs, packMs, gpuBytes, formats }, ... }, totals }`, plus one
console line per load wave: `[TextureSets] 20 sets: ktx2 6 / jpg 1 / mixed 11 / procedural 2 · 22.5 MB downloaded
(ktx2 14.4, jpg 8.0) · ~180 MB GPU · ready in ... · RGBA8 array, BC7, ASTC 4x4`. `?ktx2=0` (or `?nokx`) forces JPG for
A/B comparisons; `TextureSets.configure({ ktx2, arrayKtx2, basisPath, workers, log })` changes the defaults.

## Conventions
- **Shadow casters per cascade.** three.js draws every `castShadow` mesh into every shadow map and tests caster
  layers against the *main* camera, so layers alone cannot pick casters per cascade. Per-pass control goes through
  the object's own `onBeforeShadow` / `onAfterShadow` hooks (`engine/src/world/ShadowPasses.js`), which needs no
  change to the shadow renderer:
  - `shadowCount(instancedMesh, (shadowCamera, n) => count)`: draw only an instance prefix in a pass. Traffic packs
    its instances `[near-cascade casters | rest of the view | off-screen casters whose shadow can reach the view]`
    every frame and draws a different prefix in the main pass and each cascade (membership = sphere vs. that pass's
    frustum via `shadowFrustum`, which reads the light without touching its shadow camera).
  - `shadowDrawRange(mesh, (shadowCamera) => indexCount)` / `skipShadowPasses(mesh, (shadowCamera) => bool)`.
  - **Layer 1 (`NEAR_SHADOW_LAYER`) = "near-shadow-only casters"**: `skipShadowPasses(mesh, isFarCascade)` tags the
    mesh with layer 1 (in addition to layer 0). A render layer that filters casters per cascade natively should skip
    layer-1 objects in every cascade except the near one. (No world mesh uses it at the moment.)
- **Static geometry is merged, then tiled.** World meshes are merged per material while building and split into
  XZ tiles at the end of `MapWorld.load` (`engine/src/world/MeshChunker.js`, `MapWorld._chunkStatic`): sidewalks /
  structures 320 m tiles in the district (x2 in the filler ring, x8 beyond), ground 640 m, plugin meshes via their
  `chunk()` hook (Bay Bridge steel 400 m), the four filler-ring building sectors 640 m (district buildings are
  already built in a 7 x 7 grid of ~315 m chunks). Code that edits merged geometry in place (plugins: bridge,
  landmarks) must run before the tiling step. With streaming (default) buildings are not chunked: they are built per
  tile (district 7 x 7 chunks, 640 m squares in the filler ring) and marked `userData.streamed`.
- **Real point lights: at most 8** (`engine/src/world/NearestLights.js`): forward-lit materials loop over every light
  on every fragment and on Apple GPUs the cost per light climbs steeply past ~8 (16 lamp lights: +4.6 ms at the
  Embarcadero at night). Lights go to lamps whose light sphere reaches the view frustum, nearest to a point ahead of
  the camera; the rest of the city's lamps are emissive only.
- **View-dependent world updates** read the camera from `world.atmo.camera` (the game sets it after load);
  `MapWorld.update(dt, cameraPos)` builds one view frustum per frame and hands it to the systems that cull.
- **Window units** (real openings + instanced window frames) exist only in detail cells within 300 m (8 x 8 cells,
  THREE.LOD with 5 % hysteresis); beyond that the parallax texture windows of the shell take over.
- **API evolution.** New behaviour arrives as an option or spec field whose default reproduces the previous
  behaviour (the Roadster must drive identically: `node tools/car-sim.mjs`). Options that encode one game's world
  rules (tree canopies on kind-2 trunks, thin kinds 2/3) are opt-in in the base class and switched on by the
  game-facing preset (`ChaseRig`), never baked into lower layers.
- **Per-frame calls take positional args or reuse caller-owned objects** (`Minimap.update(dt, x, z, ...)`,
  `PhysicsWorld.surfaceAt(x, z, out)`, `DecalTrail.feed(..., normal)`); nothing in an update path allocates.
- **Rigs and controllers are inert until told otherwise where a switcher owns them:** camera rigs accept
  `{ active: false }`; `VehicleController.hold = true` freezes a vehicle on the ground (countdowns) and is
  never cleared by `reset()`.
- **Input that must survive a disabled map is declared, not hand-wired:** `alwaysOn: true` on an action (or a
  binding) keeps it live while `InputMap.enabled === false` (debug cameras, time-of-day / mute keys).

## Changelog / sample findings (Canyon Run, `samples/canyon-run/`)
The second sample was built only from engine imports; every workaround it needed became an engine fix, and the
sample now uses the engine behaviour directly.
- **Physics** (`VehicleController`, `PhysicsWorld`):
  - `_clampToSurface` removes only the velocity component *into* the ground normal (was: `velocity.y = 0` on any
    contact, which killed the downhill speed and made cars skip off downslopes); the hard-landing bounce stays.
    Touchdown impact / bounce in `step()` use the same normal component (a landing on a downslope keeps its
    downhill speed). Roadster car-sim numbers are unchanged.
  - Ground heading is measured under the current tilt (`_headingUnder`), so steep ground no longer adds a
    spurious yaw every step (bit-identical on flat ground).
  - `spec.normalClamp` (default 0.15 m = before): the ground-plane fit's kerb clamp, which also caps the
    followed slope at ~atan(0.15 / wheelbase) (~3 deg for the Roadster); off-road vehicles use ~1.
  - `liftMaxG: 0` no longer produces NaN (guarded divisions); `spec.flight: false` drops the thruster / flight
    package (thrusterLevel 1, thrust 0, mode always 'ground'); new getters `thrusterLevel`, `thrust`.
  - `spec.airControl: 'flight' | 'arcade' | 'none'` (+ `airPitchRate`, `airYawRate`); flight: false defaults to
    'arcade'.
  - Surface hook: `PhysicsWorld.setSurfaceProvider(fn)` / `surfaceAt(x, z, out)` -> per-wheel `{ mu, crr, tag }`
    multipliers (grip per axle, rolling resistance); `wheels[i]` also carry `load`, `compression`, `mu`, `crr`,
    `tag` (`spec.suspensionTravel`, `spec.suspensionSag` scale the visual compression).
  - `vc.hold = true` freezes the vehicle (no velocity / integration, still rests on the ground).
- **Render**: Atmosphere's twilight term clamps `towards` before `pow()` (NaN on ANGLE/Metal smeared over the
  PMREM environment -> black PBR surfaces around sunset); `DEFAULT_LOOKS` and `DEFAULT_GRADE` gained an el 90
  key (high sun held with slightly clearer air / lower exposure; <= 50 deg unchanged). Noise: seeded
  non-repeating `Simplex` (2D / 3D, fBm, ridged), `simplex2` / `simplex3`, exported with the other noise helpers
  (also as the `Noise` namespace). `ParticleSystem`: `floorAt(x, z)` terrain collision, `lit` / `litAmbient` /
  `sun` (number, function or Atmosphere; `atmosphereSunFactor`). `DecalTrail`: `lift` option + optional feed
  normal.
- **Input**: `VehicleInput` tolerates missing actions; `InputMap` always-on actions / bindings; modifier keys
  (Shift / Alt / Ctrl / Meta) are tracked as bindable keys.
- **Camera**: rigs take `{ active: false }`; `ChaseRig` `flightFraming: true | false | 'auto'`
  (+ `flightFramingDelay`); `CameraCollision` canopy / thin-kind rules are opt-in (ChaseRig keeps the Roadster
  rules by default; `canopyFromCollider: null`, `thinKinds: []` disable them); `FreeFlyRig` takes an `InputMap`
  (`input`, always-on `freeFly.*` actions via `freeFlyActions()`), raw keys remain the fallback.
- **Core / UI**: `Profiler.gpuApprox` (timer query on ANGLE/Metal is indicative only; shown as `gpu~`) and
  `Profiler.benchThroughput(renderFn, frames)` (ms / frame, GPU-synchronised); `Minimap.update` takes positional
  args or its own `map.view`, and follows `theme.accent` live unless given an accent (other widgets bake the
  theme at build time: call `setTheme()` first, see `engine/src/ui/index.js`).

## Changelog: world streaming + traffic worker
- **World**: `TileStreamer` (new, generic) + MapWorld `streaming` option (default on with a renderer; `?nostream=1`
  / `streaming: false` = full build). `BuildingBuilder.footprint(b)`, `add(b, rnd, footprint)`, options `nearDetail`
  and `grid`; `TileBuildingBuilder` (per-tile builder, modes low / full / near) and `NullBuf`. `_chunkStatic` skips
  `userData.streamed` meshes. Plugin hooks see only the tiles built at load; `streaming.pin(building, data)` keeps
  plugin-edited tiles resident (Roadster: the OSM tower block under the Bay Bridge).
- **Traffic**: `Traffic.js` split into `TrafficSim.js` (core, Node / worker safe) + `Traffic.js` (presenter) +
  `traffic.worker.js`; `traffic: { worker }` (default on in browsers with Workers, `?noworker=1` off); ground heights
  baked into the lane graph (the sim no longer queries the world); `summary()` also reports junction entries at signals /
  on red and vehicles held by the player rules; `TrafficSim.bakeError()` dev check. `dev/traffic-sim.mjs` runs the core
  directly and checks the bake (fails on > 0.1 % of samples off by > 10 cm). Behaviour is bit-identical in x / z: the
  60 s headless runs reproduce the old respawn / speed / signal counts exactly.
- **Speed-ups on the way**: `geom.RingIndex` (exact, row-bucketed point-in-ring) for the coastline in
  `MapWorld.isLand` outside the district raster (10x; the full build's "Buildings" phase 514 -> 376 ms, a 152 k-sample
  lane-height bake 110 -> 24 ms in Node); presenter matrices written directly (no Euler / quaternion), frustum planes flattened, shadow-box
  sphere pre-reject.


## Packaging
`engine/` is the npm package `@vibe/engine` (MIT, `engine/package.json`, subpath exports per layer, `files` =
`src` + `tools`). The root `package.json` declares it as a workspace, so the game and `samples/canyon-run`
import `@vibe/engine` / `@vibe/engine/<layer>`; no file in `src/`, `samples/` or `dev/` imports `engine/src`
by relative path any more. Layer `index.js` files are the public surface (deep imports are unsupported).
`CHANGELOG.md` tracks releases; `npm run pack:check` inside `engine/` verifies the tarball.

## Render interpolation (fix for judder, 2026-10-18)
Physics runs at a fixed 120 Hz while frames arrive at a variable rate, so a frame may contain 1, 2 or 3 steps.
Showing the latest step directly made the car's per-frame displacement swing by ~30 %, which reads as the whole
screen shaking at speed (worse on high-DPI screens where frames are slower). The engine already exposes
`alpha = accumulator / fixedDt` to `update(dt, alpha)`; game objects driven by fixed-step physics must keep the
previous pose and blend: `Car.interpolate(alpha)` lerps position and slerps rotation between the last two steps
(measured frame-to-frame speed variation 0.4 % vs 30 % raw). The camera then follows the blended pose. Also:
moving vehicles are 'thin' for camera occlusion (`thinKinds: [2, 3, 'vehicle']`), so traffic passing behind the
player no longer yanks the chase camera in.

## Suspension & contact model (2026-10-19)
**Why.** The first vehicle model clamped a rigid body to the plane through its four wheel supports every step.
Measured on the real map with the new street-drive harness (`node dev/drive-sim.mjs`: a pure-pursuit pilot drives the
Roadster along 23 downtown streets, 21 km at 50 km/h, 80 on the Embarcadero, with terrain, decks, building / pier /
furniture colliders built exactly as the game builds them), that meant: 926 one-step grounded/airborne flickers (each
one zeroed the wheel loads, ran the flight attitude loop for a step and fired a landing event), pitch snaps of up to
7 deg in a single step, 5 cm single-wheel hops along lanes (the +0.05 asphalt lift came from 1 m raster pixels),
wheels hanging 0.2 m in the air on hills, 1-1.8 m teleports onto overpass decks whose ends sat above the street, and
a crest jump entered "flight mode" so the hover assist floated the car down.

**Vehicle (`engine/src/physics/VehicleController.js`).**
- Each wheel hangs from an attach point at the top of its travel along body-down and sits on its support within
  `suspensionTravel` (`wheels[i].y`, `compression`, `contact`); beyond the travel it is in the air (`wheelGaps[i]`).
  The support under a rolling tyre is sampled with the tyre radius (centre + 0.6 R ahead / behind along the velocity,
  box corners within R lift the wheel along the tyre's circle), so a kerb is climbed over several steps.
- Spring rate per corner = static load / sag (`suspensionSag` x travel), damping ratio `suspensionDamping`, anti-roll
  bar `antiRoll` x the axle rate. Forces act along body up at the contact points and give the body pitch / roll
  torques about the CG (`cgHeight`, `pitchInertia`, `rollInertia`); tyre forces act at the contact patches, so braking
  dives, acceleration squats, cornering leans outward; downforce is applied at the axles (`downforceFront`). Axle
  loads for the tyre model are the spring loads (the old smoothed weight-transfer terms are gone).
- `position` is still the reference point that rests at ground height at the static load; the body moves within
  +-travel of it (top speed downforce: -1 cm). Travel exhausted -> the body clamp acts as a bump stop (velocity into
  the surface removed, hard touchdowns above `hardLandingSpeed` still bounce and damage). Body points use the
  wheel-level step reference: a surface more than `stepHeight` above the wheels is an obstacle, never a floor.
- Boxes low enough to be driven onto (top within `stepHeight` of the reference) are supports, not walls, for the body
  OBB. Traffic boxes carry `vx / vz`; the contact response runs in the box's frame so a moving car shoves with its
  momentum.
- Flight is armed by a lift input (`flightArmThrust`) and disarmed by touchdown: airborne without it is ballistic
  with auto-level only (`mode` stays `'ground'`, no hover assist); a lift tap in the air arms at once.
- `reset()` and `hold` fit the body to the plane through the supports with the wheels at static compression (no
  first-step snap when spawning on a hill).

**World (`MapWorld`, `RoadNetwork`, `Props`).**
- `getGroundHeight` adds `pavingLift`: asphalt +0.05 inside the exact ribbon geometry (`net.segHash`, round end caps,
  0.35 m ramp outside the edge), sidewalk slabs / kerb tops +0.17 (raster), else the ground mesh. `surfaceAt`
  (`SURFACES`): asphalt, paving, grass, bare ground and water (mu 0.12, crr x30: the bay is not a road, the car bogs
  down and needs its thrusters). `getGroundHeight(x, z, y, step)`: the physics passes `deckStep` 0.35 (a deck a car
  cannot climb must not become its ground; the traffic lane bake keeps 1.6).
- Deck solver: grounded deck ends are pinned to the ground and a piece whose other end is too high for 6 % gets the
  grade it needs (Harrison Street over the freeway cut: a 2 m cliff at the ramp foot is now a 10 % ramp); deck nodes
  over a crossed street get a lower bound of street + 4.6 m.
- Parked cars at an angle are registered as three AABB slices (one box of a 30-degree car bulged 0.7 m into the
  lane); traffic-signal poles are never placed on a carriageway.

**Measured (drive-sim, same 23 streets).**
| | before | after |
|---|---|---|
| grounded/airborne flickers | 926 | 1 |
| pitch snaps > 0.5 deg/step | 153 | 0 |
| roll snaps > 0.5 deg/step | 140 | 11 (kerbs the pilot clips) |
| vertical hops > 2 cm/step | 167 | 10 |
| flight-mode entries without thrust | 6 | 0 |
| impacts | 21513 (pilot in the parking lane / median) | 26 |

`tools/car-sim.mjs` grew 24 suspension checks (rest, ring-down, 12 cm kerb, rounded slope break, dive / squat / roll
signs, ledge drop, arming, traffic shove, reset on a slope); 73 checks pass.

**Known map artifact (fixed in the deck solver, 2026-10-19).** The feet of the freeway and Transbay bus ramps at 1st,
2nd and Harrison Street overlapped the street they merge into for the first ~30 m of ramp (OSM ends the bridge way
where the span ends, a few metres past the street; the approach beyond is a ground road here); the deck climbed at 6 %
from that foot and hung 0.5-1.3 m over the asphalt, exempt from the clearance bound, and the car drove through the slab.
`RoadNetwork.footZones` now walks from every grounded deck end and pins the deck to the ground while its centre line
runs over another road's carriageway (a street within 15 m of the foot / the previous hit, a service road within 30 m):
that ramp-foot zone is street, not deck (rasterised as road, in `segHash`, ribbon sampled from the ground 2 cm above the
street's, no `deckHash` entry, no slab), and the climb starts at the carriageway edge with the grade it needs.
`node dev/ramp-foot-probe.mjs` lists deck samples 0.4-3.2 m over a ground carriageway: 51 before (nine zones now,
four pieces lie on the ground entirely), 0 after; drive-sim sunk-wheel steps 438 -> 0, everything else unchanged.
