# Changelog

## 0.3.1 — 2026-10-21
- Standalone repository (`inbrowser-game-engine`): the Canyon Run sample, `test/car-sim.mjs` and ENGINE.md moved in
  from the game; `npm run dev` / `npm run build` serve and build the sample, `npm test` runs the physics checks.
  The game ([flying-tesla](https://github.com/BrandeisPatrick/flying-tesla)) embeds this repository as a submodule.

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
