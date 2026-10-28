// MapWorld: a composable city world built from map data (city.json contract, MapData.js) on a real terrain
// (terrain.json, Terrain.js): ground + water + shore, road network, buildings, street furniture, vegetation, parked
// and moving traffic, distant relief (farland.json), physically based sky / sun / fog (render/Atmosphere.js).
// Everything game-specific - which data to load, district style rules, species / lamp / boat rules, hand-made
// structures - comes in through options and plugins (see the constructor and ENGINE.md "MapWorld").
//
//   const world = new MapWorld({ scene, renderer, data: { mapUrl, terrainUrl, farlandUrl }, plugins: [...], ... });
//   await world.load((p, label) => ...);
//   per frame: world.update(dt, camera.position)   (or engine.addSystem(world.system({ camera })))
//   queries:   getGroundHeight(x, z, y?), queryColliders(x, z, r), getSpawn(), isLand(x, z), getMinimapData(),
//              sunDirection, stats
//
// Plugins: { name, beforeBuildings?(ctx), afterBuildings?(ctx), afterProps?(ctx), clearZones?(), chunk?(world),
//            update?(dt, cameraPos, darkness, world), instance? }. Phase hooks get
//   ctx = { world, data, net, colliders, mat, tex, groundAt, renderer, offset }
// and may return an Object3D (added to world.group). After a hook, world[name] = plugin.instance ?? plugin (null if
// the hook threw; a failed plugin is skipped from then on). Order: beforeBuildings (after the road network, before
// the building builder: claim footprints with `_lm`), afterBuildings (merged buildings exist, their colliders are
// registered right after: edit merged geometry here), afterProps (street furniture exists, traffic comes next).
// clearZones() -> [{ x, z, r }] (bake frame) keeps the low-detail filler lots off those sites; chunk(world) ->
// [{ mesh, cell }] extra meshes tiled by the final MeshChunker pass. update runs after the props, before traffic.
// Streaming (default with a renderer; `streaming: false` / ?nostream=1 = everything at load): building geometry is
// built per tile around the camera by a TileStreamer (world.streamer, world.buildingTiles); colliders stay global.
// Phase hooks then see only the tiles built at load (within streaming.spawnRadius of the spawn, plus pinned ones):
// a plugin that edits merged building geometry needs its tiles pinned (streaming.pin), or they are rebuilt unedited.
// Traffic simulates in a Web Worker by default (Traffic.js / TrafficSim.js / traffic.worker.js).
// Coordinates: world = bake frame - world.offset (the spawn point is the origin).
import * as THREE from 'three';
import { Atmosphere, TIME_PRESETS } from '../render/Atmosphere.js';
import { makeAsphaltTexture, makePavingTexture, makeGroundTexture, makeGrassTexture } from '../render/TextureSets.js';
import { ColliderGrid, polygonToBoxes, Raster } from '../physics/spatial.js';
import { PhysicsWorld } from '../physics/PhysicsWorld.js';
import { makeFacadeTextures, loadPbrTextures, PBR } from './FacadeTextures.js';
import { BuildingBuilder, TileBuildingBuilder, createFacadeMaterials, releaseTileBuffers } from './Buildings.js';
import { TileStreamer } from './TileStreamer.js';
import { RoadNetwork, createRoadMaterial, RBIT, SolidBuf } from './RoadNetwork.js';
import { Props } from './Props.js';
import { Terrain, GroundSurface, drapePolygon } from './Terrain.js';
import { loadMapData, prepareMapData, generateProceduralCity, assetBase } from './MapData.js';
import { mulberry32, signedArea, sanitizeRing, pointInRing, ringBounds, hashStr, clamp, lerp, RingIndex } from './geom.js';
import { Traffic } from './Traffic.js';
import { createWaterMaterial, makeWaterNormalMap } from './Water.js';
import { Waterfront, createGroundMaterial } from './Shore.js';
import { replaceWithChunks, localizeMeshOrigin } from './MeshChunker.js';

export const DEFAULT_WATER_LEVEL = -0.6;
/** Asphalt ribbons are drawn this far above the ground mesh (RoadNetwork yBase); the physics ground follows. */
export const ROAD_LIFT = 0.05;
/** Sidewalk slabs / kerb tops above the ground mesh (RoadNetwork sidewalkStrip yTop). */
export const SIDEWALK_LIFT = 0.17;
const ROAD_EDGE_RAMP = 0.35; // m outside a ribbon edge over which the asphalt lift fades to the ground
/** Tyre surfaces by ground class: grip multiplier, rolling-resistance multiplier, tag (see surfaceAt). */
export const SURFACES = Object.freeze({
  asphalt: { mu: 1, crr: 1, tag: 'asphalt' },
  paving: { mu: 0.95, crr: 1.3, tag: 'paving' },
  grass: { mu: 0.7, crr: 6, tag: 'grass' },
  ground: { mu: 0.85, crr: 3, tag: 'ground' },
  water: { mu: 0.12, crr: 30, tag: 'water' },
});
const tick = () => new Promise((r) => setTimeout(r, 0));

/**
 * Streaming defaults (MapWorld `streaming` option). Building tiles = the district's base chunks (baseGrid cells) +
 * fillerCell squares of the context ring; each holds the shells (+ far-LOD shells) of the buildings centred in it. A
 * child 'detail' tile per detail cell of a district tile adds the near-LOD detail (real openings, window units) to its
 * parent's THREE.LOD inside the detail radius. See TileStreamer.js for the scheduler.
 */
export const DEFAULT_STREAMING = {
  radius: 900,          // m: tiles within this (+ reach, + altitude) are built at level 1
  keep: 250,            // m: hysteresis before a built tile is freed
  heightReach: 2.5,     // extra radius per metre of the tile's tallest building (skylines are seen from far away)...
  maxReach: 900,        // ...capped
  altitudeScale: 6,     // extra radius per metre of camera height above the ground (flight)
  openRadius: 1200,     // extra radius while the camera is over open water (nothing hides the skyline)
  detailMargin: 150,    // m: a cell's near detail is built this far before its LOD switch distance (lodDist * 1.05)
  detailKeep: 100,      // m: hysteresis before the near detail is freed again
  spawnRadius: 500,     // m: the loading screen waits for the tiles within this radius of the spawn
  budgetMs: 3,          // build time per frame
  maxConcurrent: 2,
  uploadBytesPerFrame: 6e6,
  urgentRadius: 200,    // m: tiles this close with nothing built (teleport, reset) are built at once
  lookAhead: 1.5,       // s of camera travel that steer priorities
  fillerCell: 640,      // m: tile size in the filler ring
  fillerRadius: 3000,   // m: the low context ring is cheap (~5 MB) and seen across the bay / from the hills: keep it wide
  fillerKeep: 500,
  pin: null,            // (building, data) => true: its tile is built at load and never freed (plugin-edited geometry)
};

/** Streaming options: o.streaming (false | true | {...}); off headless (no renderer) and with ?nostream=1. */
function streamingOptions(o) {
  if (o.streaming === false || !o.renderer) return null;
  try { const q = typeof location !== 'undefined' ? new URLSearchParams(location.search) : null; if (q && q.has('nostream') && q.get('nostream') !== '0') return null; } catch { /* no location */ }
  return { ...DEFAULT_STREAMING, ...(o.streaming && typeof o.streaming === 'object' ? o.streaming : {}) };
}
/** Building shell meshes that get a local origin at the tiling step (Buildings.js GeoBuf.build names). */
const BUILDING_SHELL_NAMES = new Set(['buildings', 'buildings-far', 'buildings-near-walls']);

const geoBytes = (g) => { let b = 0; if (!g) return 0; for (const k in g.attributes) b += g.attributes[k].array.byteLength; if (g.index) b += g.index.array.byteLength; return b; };
/** Free the GPU buffers of a streamed object tree (materials are shared; window units share their unit attributes). */
function disposeStreamed(obj) {
  let freed = 0;
  obj.traverse((m) => {
    const g = m.geometry; if (!g) return;
    if (g.isInstancedBufferGeometry) { for (const k of ['position', 'normal', 'aOff', 'aPart']) g.deleteAttribute(k); g.setIndex(null); }
    freed += geoBytes(g);
    g.dispose();
  });
  return freed;
}

export class MapWorld {
  /**
   * @param {object} o
   * @param {THREE.Scene} o.scene
   * @param {THREE.WebGLRenderer|null} o.renderer  (null: headless data-only use, e.g. Node sims)
   * @param {object} [o.data]  { mapUrl, terrainUrl, farlandUrl, map (preloaded city.json object) }
   *        defaults `${BASE_URL}map/city.json`, `.../terrain.json`, `.../farland.json`; farlandUrl false = none
   * @param {boolean} [o.forceProcedural=false]  skip the map + terrain, build the procedural fallback city
   * @param {boolean} [o.terrain=true]  false = flat ground
   * @param {object} [o.procedural]  generateProceduralCity names ({ waterfrontName, center })
   * @param {object} [o.atmosphere]  render/Atmosphere.js options (sun model, looks, shadows, ...)
   * @param {number|string} [o.timeOfDay='golden']  hours 0..24 or a TIME_PRESETS name
   * @param {number} [o.waterLevel=-0.6]  flat water plane height (m); land never dips below 0
   * @param {number} [o.pbrSize=1024]  PBR texture-set resolution
   * @param {object} [o.spawn]  MapData.findSpawn options ({ road, at })
   * @param {boolean} [o.filler=true]  low-detail context ring (filler buildings + outer lots) around the district
   * @param {Array<{x,z,r}>} [o.clearZones]  extra bake-frame circles kept free of filler lots
   * @param {object} [o.roads]  RoadNetwork options ({ deckHolds })
   * @param {object} [o.buildings]  BuildingBuilder options ({ style, baseGrid, detailGrid, lodDist, detailFloors, maxWindows })
   * @param {object} [o.props]  Props options ({ seed, streets, signStyle })
   * @param {object} [o.vegetation]  Vegetation options ({ rules, mix })
   * @param {object|number|false} [o.traffic]  Traffic options ({ count, seed, busLivery, worker }) or a vehicle count; false = none.
   *        worker: simulate in a Web Worker (default: browsers with Worker support unless ?noworker=1; Node: in-thread)
   * @param {object} [o.shore]  Waterfront options ({ ferryDock, islandFill, boats, farland })
   * @param {object} [o.chunk]  tile sizes (m) of the final static-mesh tiling ({ buildings: 320, structures: 320, ground: 640 })
   * @param {Array<object>} [o.plugins]  see header
   * @param {boolean} [o.physics=false]  true: build `world.physics` (PhysicsWorld.fromWorld) at the end of load()
   * @param {object|boolean} [o.streaming]  building tiles built around the camera instead of all at load (default on
   *        with a renderer; false or ?nostream=1 = build everything at load): DEFAULT_STREAMING fields
   */
  constructor(o = {}) {
    this.scene = o.scene;
    this.renderer = o.renderer || null;
    this.opts = o;
    this.urls = {
      map: (o.data && o.data.mapUrl) || o.mapUrl || `${assetBase()}map/city.json`,
      terrain: (o.data && o.data.terrainUrl) || o.terrainUrl || undefined,
      farland: o.data && o.data.farlandUrl !== undefined ? o.data.farlandUrl : undefined,
    };
    this.waterLevel = o.waterLevel ?? DEFAULT_WATER_LEVEL;
    this.plugins = (o.plugins || []).filter(Boolean);
    this.group = new THREE.Group();
    this.group.name = 'world';
    this.colliders = new ColliderGrid(32);
    this._colOut = [];
    this.time = 0;
    this.source = 'none';
    this.stats = {};
    this._spawn = { position: new THREE.Vector3(0, 0, 0), headingRad: 0 };
    this._sunOut = new THREE.Vector3(0, 1, 0);
    this.bounds = { minX: -800, maxX: 800, minZ: -800, maxZ: 800 };
    this.landPolys = [];
    this.terrain = null;   // heightfield: { heightAt(x,z) } (Terrain.js)
    this.traffic = null;
    this.physics = null;
    for (const p of this.plugins) if (p.name && !(p.name in this)) this[p.name] = null;
    this._failed = new Set();
    this.loaded = false;
    this.streamOpts = streamingOptions(o);
    this.streamer = null;
  }

  // ================================================================== load
  async load(onProgress) {
    const o = this.opts;
    const t0 = performance.now();
    const phases = {}; let phase = 'start', tPhase = t0; // load profile: ms per progress label (stats.loadPhases)
    const P = (f, label) => {
      const t = performance.now(); phases[phase] = Math.round((phases[phase] || 0) + t - tPhase); phase = label; tPhase = t;
      try { onProgress && onProgress(clamp(f, 0, 1), label); } catch { /* ignore */ }
    };
    P(0.01, 'Loading map data');
    // terrain loads in parallel with the map (never throws; flat if missing)
    const terrainP = !o.forceProcedural && o.terrain !== false ? new Terrain().load(this.urls.terrain) : Promise.resolve(new Terrain());
    // CC0 PBR texture sets (public/textures, engine/tools/fetch-textures.mjs) load in parallel; never throws (procedural fallback)
    const pbrP = loadPbrTextures(this.renderer, { size: o.pbrSize || 1024 }).catch((e) => { console.warn('[MapWorld] PBR textures failed', e); return null; });
    let data = null;
    if (!o.forceProcedural) {
      try {
        data = o.data && o.data.map ? o.data.map : await loadMapData(this.urls.map);
        this.source = data.source || 'osm';
      } catch (e) {
        console.warn('[MapWorld] city.json unavailable, using procedural fallback:', e.message || e);
        data = null;
      }
    }
    if (!data) { data = generateProceduralCity(1337, o.procedural || {}); this.source = 'procedural'; }
    this.data = data;
    this._prepare(data);
    P(0.04, 'Terrain');
    this.terrain = await terrainP;
    if (this.source === 'procedural') this.terrain = new Terrain(); // the DEM belongs to the real map: keep the fallback flat
    this.terrain.setOffset(this.offset); // terrain.json is in the bake frame; query in world coordinates
    await tick();

    P(0.06, 'Sky and lighting');
    this.atmo = new Atmosphere({ ...(o.atmosphere || {}), scene: this.scene, renderer: this.renderer, timeOfDay: o.timeOfDay ?? (o.atmosphere && o.atmosphere.timeOfDay) ?? 'golden', build: false });
    this.atmo.build();
    await tick();

    P(0.12, 'Textures');
    const R = this.renderer;
    this.tex = {
      asphalt: makeAsphaltTexture(R), paving: makePavingTexture(R), ground: makeGroundTexture(R),
      grass: makeGrassTexture(R), water: makeWaterNormalMap(R), facade: makeFacadeTextures(R),
    };
    this.tex.pbr = (await pbrP) || await loadPbrTextures(R, { size: 256 }).catch(() => null);
    if (this.tex.pbr) console.info(`[MapWorld] PBR texture sets: ${this.tex.pbr.loaded}/${this.tex.pbr.wanted} loaded (rest procedural)`);
    this._makeMaterials();
    await tick();

    P(0.2, 'Land and water');
    this._buildRaster();
    if (o.filler !== false && this.source !== 'procedural' && this.landPolys.length) this._makeFiller();
    await tick();

    P(0.24, 'Terrain surface');
    this._buildSurface();
    this.net = new RoadNetwork({ roads: data.roads, rail: data.rail, raster: this.raster, isWater: (x, z) => !this.isLand(x, z), surface: this.surface, deckHolds: (o.roads && o.roads.deckHolds) || [] });
    this.net.prepare();                              // road profiles from the terrain
    this.surface.applyCorridors(this.net.corridors); // flatten ground under ground-level roads
    this.surface.finalize();
    this.terrain.attachSurface(this.surface);
    await tick();
    this._buildGroundAndWater();
    await tick();

    P(0.34, 'Road network');
    this.net.build();
    this._buildFlatAreas();
    for (const m of this.net.meshes(this.mat.road, this.mat.concrete, this.mat.rail)) this.group.add(m);
    for (const b of this.net.piers) this.colliders.add(b[0], b[1], b[2], b[3], b[4], b[5], 5);
    this.deckHash = this.net.deckHash;
    await tick();

    P(0.55, 'Buildings');
    await this._buildBuildings(P);
    await tick();

    P(0.86, 'Street furniture');
    const PO = o.props || {}, VO = o.vegetation || {};
    this.props = new Props({
      net: this.net, raster: this.raster, colliders: this.colliders, data, isLand: (x, z) => this.isLand(x, z), groundAt: (x, z) => this._groundAt(x, z),
      seed: PO.seed ?? 4242, streets: PO.streets || null, signStyle: PO.signStyle || null, treeRules: VO.rules || null, treeMix: VO.mix || null,
    });
    this.group.add(this.props.build());
    await tick();

    // ---- plugins: after the street furniture (e.g. a bridge superstructure that hangs deck lamps)
    for (const pl of this.plugins) {
      if (typeof pl.afterProps !== 'function') continue;
      P(0.90, pl.label || pl.name || 'Plugins');
      this._runHook(pl, 'afterProps');
      await tick();
    }

    // ---- moving traffic (Traffic.js)
    P(0.93, 'Traffic');
    const TO = this._trafficOpts();
    if (TO) {
      try {
        this.traffic = new Traffic({ world: this, data, net: this.net, props: this.props, seed: TO.seed ?? 7, count: TO.count, busLivery: TO.busLivery || null, worker: TO.worker });
        const tg = this.traffic.build();
        if (tg) this.group.add(tg);
      } catch (e) { console.warn('[MapWorld] traffic failed', e); this.traffic = null; }
    }
    await tick();

    P(0.96, 'Finalising');
    this._chunkStatic();
    if (this.traffic) await this.traffic.ready; // the traffic worker builds its sim while the static meshes are tiled
    { // spawn on the road surface (ignoring any deck overhead)
      const sp = this._spawn.position;
      sp.y = this.getGroundHeight(sp.x, sp.z, this._groundAt(sp.x, sp.z));
    }
    this.scene.add(this.group);
    this._applyTime();
    this.loaded = true;
    if (o.physics) this.createPhysics();
    this.stats = {
      source: this.source,
      buildings: this.buildingCount, fillerBuildings: this.fillerCount || 0,
      roads: this.net.list.length, bridges: this.net.bridgeCount,
      colliders: this.colliders.boxes.length,
      ...this.props.counts,
      vehicles: this.traffic ? this.traffic.count : 0, trafficSim: this.traffic ? this.traffic.mode : 'none',
      meshes: this.group.children.length,
      terrain: this.terrain.ready ? (this.terrain.info && this.terrain.info.source) || 'terrain.json' : 'flat',
      groundTris: this._groundTris || 0,
      chunkTiles: this._chunkTiles || 0, chunkMs: this._chunkMs || 0,
      streaming: this.streamer ? { ...this.streamLoad, ...this.buildingTiles } : false,
      loadMs: Math.round(performance.now() - t0),
      loadPhases: phases,
    };
    console.info('[MapWorld] ready', this.stats);
    P(1, 'Ready');
  }

  /** Traffic options normalised from o.traffic (number = count, false = none, object = options). */
  _trafficOpts() {
    const t = this.opts.traffic;
    if (t === false) return null;
    if (t && typeof t === 'object') return t;
    return { count: Number.isFinite(t) ? t : undefined };
  }

  // ================================================================== plugins
  _pluginCtx() {
    return { world: this, data: this.data, net: this.net, colliders: this.colliders, mat: this.mat, tex: this.tex, groundAt: (x, z) => this._groundAt(x, z), renderer: this.renderer, offset: this.offset };
  }
  /** Run one phase hook of one plugin: add a returned Object3D, expose world[name], disable the plugin on errors. */
  _runHook(pl, hook) {
    if (this._failed.has(pl)) return;
    try {
      const r = pl[hook](this._pluginCtx());
      if (r && r.isObject3D) this.group.add(r);
      if (pl.name) this[pl.name] = pl.instance ?? pl;
    } catch (e) {
      console.warn(`[MapWorld] plugin ${pl.name || '?'} ${hook} failed`, e);
      this._failed.add(pl);
      if (pl.name) this[pl.name] = null;
    }
  }

  // ================================================================== data prep
  /** Fill / shift the map data (MapData.prepareMapData) and set offset, bounds, outerBox and the spawn. */
  _prepare(data) {
    const r = prepareMapData(data, this.opts.spawn || {});
    this.offset = r.offset;
    this.bounds = r.bounds;
    this.outerBox = r.outerBox;
    this._spawn = { position: new THREE.Vector3(0, 0, 0), headingRad: r.spawn.headingRad, road: r.spawn.road };
  }

  // ================================================================== materials
  _makeMaterials() {
    const T = this.tex;
    // land: urban ground in the district, built-up / street-grid context outside, natural blend (grass, scrub,
    // rock by slope + land class) on the islands and hills - see Shore.js createGroundMaterial
    const ground = createGroundMaterial(T.ground, this.renderer);
    // open water + ponds: Water.js (shore field / noise uniforms are filled in by the Waterfront)
    const water = createWaterMaterial({ normalMap: T.water, noiseMap: null });
    const pond = createWaterMaterial({ normalMap: T.water, noiseMap: null, pond: true });
    const G = (T.pbr && T.pbr.ground) || {};
    const fac = createFacadeMaterials(T.facade, T.pbr);
    this.mat = {
      ground, water, pond,
      road: createRoadMaterial(T.asphalt, G.asphalt || null),
      concrete: this._makeConcreteMaterial(T, G.sidewalk),
      grass: this._makeGrassMaterial(T, G.grass),
      rail: new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.32, metalness: 0.9 }),
      facade: fac.facade, facadeWin: fac.windows, facadeSet: fac,
    };
  }

  /** Sidewalks / kerbs / bridge concrete: PBR paving on up-facing faces, rough cast concrete on vertical faces. */
  _makeConcreteMaterial(T, sw) {
    if (!sw || !T.pbr) return new THREE.MeshStandardMaterial({ map: T.paving, vertexColors: true, roughness: 0.88, metalness: 0 });
    const rep = 6 / (sw.size || 1.8); // SolidBuf(6) uv = metres / 6
    for (const t of [sw.map, sw.normalMap, sw.roughnessMap]) if (t) t.repeat.set(rep, rep);
    const m = new THREE.MeshStandardMaterial({ map: sw.map, normalMap: sw.normalMap, roughnessMap: sw.roughnessMap, vertexColors: true, roughness: 1, metalness: 0 });
    m.normalScale.set(0.8, 0.8);
    const u = { uPA: { value: T.pbr.albedo }, uPN: { value: T.pbr.normal }, uLayer: { value: PBR.CONCRETE_ROUGH }, uSize: { value: T.pbr.sizes[PBR.CONCRETE_ROUGH] || 2.7 } };
    m.onBeforeCompile = (sh) => {
      Object.assign(sh.uniforms, u);
      sh.vertexShader = sh.vertexShader.replace('#include <common>', '#include <common>\nvarying vec2 vRawUv;')
        .replace('#include <uv_vertex>', '#include <uv_vertex>\nvRawUv = uv;');
      sh.fragmentShader = sh.fragmentShader
        .replace('#include <common>', `#include <common>
uniform highp sampler2DArray uPA; uniform highp sampler2DArray uPN; uniform float uLayer; uniform float uSize;
varying vec2 vRawUv;`)
        .replace('#include <map_fragment>', `
float cUp = abs(dot(normalize(vNormal), normalize((viewMatrix * vec4(0.0, 1.0, 0.0, 0.0)).xyz)));
float cFlat = smoothstep(0.55, 0.8, cUp);
vec4 cWall = texture(uPA, vec3(vRawUv * 6.0 / uSize, uLayer));
#include <map_fragment>
diffuseColor.rgb = mix(cWall.rgb * vec3(1.0, 0.99, 0.97) * cWall.a, diffuseColor.rgb, cFlat);
diffuseColor.rgb *= (0.85 + 0.3 * texture2D(map, vMapUv * 0.071 + 0.13).g) * mix(1.0, 1.55, cFlat); // light concrete sidewalks`)
        .replace('#include <roughnessmap_fragment>', '#include <roughnessmap_fragment>\nroughnessFactor = mix(0.92, roughnessFactor, cFlat);')
        .replace('mapN.xy *= normalScale;', 'mapN.xy *= normalScale * cFlat;');
    };
    m.customProgramCacheKey = () => 'roadster-concrete-pbr-1';
    return m;
  }

  /** Parks: photo grass with a large-scale tone variation that hides the tiling. */
  _makeGrassMaterial(T, gs) {
    const opts = { roughness: 0.96, metalness: 0, polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -2 };
    if (!gs) return new THREE.MeshStandardMaterial({ map: T.grass, ...opts });
    const rep = 4 / (gs.size || 2); // SolidBuf(4)
    for (const t of [gs.map, gs.normalMap, gs.roughnessMap]) if (t) t.repeat.set(rep, rep);
    const m = new THREE.MeshStandardMaterial({ map: gs.map, normalMap: gs.normalMap, roughnessMap: gs.roughnessMap, ...opts, roughness: 1 });
    m.onBeforeCompile = (sh) => {
      sh.fragmentShader = sh.fragmentShader.replace('#include <map_fragment>', `#include <map_fragment>
{
  float mac = texture2D(map, vMapUv * 0.061 + 0.27).g;
  float mac2 = texture2D(map, vMapUv * 0.013 + 0.61).r;
  diffuseColor.rgb *= (0.75 + 0.6 * mac) * mix(vec3(1.0), vec3(1.08, 1.0, 0.82), smoothstep(0.1, 0.3, mac2));
}`);
    };
    m.customProgramCacheKey = () => 'roadster-grass-pbr-1';
    return m;
  }

  // ================================================================== land / water
  _buildRaster() {
    const b = this.bounds, m = 60;
    this.raster = new Raster(b.minX - m, b.minZ - m, b.maxX + m, b.maxZ + m, 1);
    const d = this.data;
    this.landPolys = [];
    // big rings (coastlines) get an exact row index: isLand() outside the raster (filler ring, outer land) is hot at load
    const poly = (r) => ({ ring: r, bb: ringBounds(r), idx: r.length > 48 ? new RingIndex(r) : null });
    for (const L of d.land) { const r = sanitizeRing(L && L.p); if (r) this.landPolys.push(poly(r)); }
    this.pierPolys = [];
    for (const L of d.piers) { const r = sanitizeRing(L && L.p); if (r) this.pierPolys.push(poly(r)); }
    if (this.landPolys.length) {
      this.raster.data.fill(RBIT.WATER);
      for (const L of this.landPolys) this.raster.fillPolygon([L.ring], RBIT.WATER, 'clear');
      for (const L of this.pierPolys) this.raster.fillPolygon([L.ring], RBIT.WATER, 'clear');
    }
    let maxH = 30;
    for (const bd of d.buildings) {
      const r = sanitizeRing(bd && bd.p); if (!r) continue;
      if ((bd.mh || 0) > 2.2) continue; // raised parts (skybridges, canopies) do not block the street
      this.raster.fillPolygon([r, ...((bd.hl || []).map((h) => sanitizeRing(h)).filter(Boolean))], RBIT.BUILDING);
      maxH = Math.max(maxH, Math.min(400, bd.h || 0));
    }
    this.maxBuildingHeight = maxH;
    for (const pk of d.parks) { const r = sanitizeRing(pk && pk.p); if (r) this.raster.fillPolygon([r], RBIT.PARK); }
  }

  /** True when (x,z) is land (incl. piers); false over the water. */
  isLand(x, z) {
    if (!this.landPolys.length) return true;
    const r = this.raster;
    if (r) { const i = r.idx(x, z); if (i >= 0) return !(r.data[i] & RBIT.WATER); }
    for (const L of this.landPolys) { const b = L.bb; if (x >= b.minX && x <= b.maxX && z >= b.minZ && z <= b.maxZ && (L.idx ? L.idx.contains(x, z) : pointInRing(x, z, L.ring))) return true; }
    for (const L of this.pierPolys) { const b = L.bb; if (x >= b.minX && x <= b.maxX && z >= b.minZ && z <= b.maxZ && (L.idx ? L.idx.contains(x, z) : pointInRing(x, z, L.ring))) return true; }
    return false;
  }

  /** Nested ground lattice: 5 m over the detailed district, 10 m over the filler ring, 20 m to the horizon. */
  _buildSurface() {
    const b = this.bounds, OB = this.outerBox;
    const cx = (b.minX + b.maxX) / 2, cz = (b.minZ + b.maxZ) / 2;
    const R1 = Math.max(b.maxX - b.minX, b.maxZ - b.minZ) / 2 + 1000 + 250;
    const levels = this.terrain.ready
      ? [
        { minX: b.minX - 60, minZ: b.minZ - 60, maxX: b.maxX + 60, maxZ: b.maxZ + 60, cell: 5 },
        { minX: cx - R1, minZ: cz - R1, maxX: cx + R1, maxZ: cz + R1, cell: 10 },
        { minX: OB.minX - 20, minZ: OB.minZ - 20, maxX: OB.maxX + 20, maxZ: OB.maxZ + 20, cell: 20 },
      ]
      : [{ minX: Math.min(OB.minX, -7000), minZ: Math.min(OB.minZ, -7000), maxX: Math.max(OB.maxX, 7000), maxZ: Math.max(OB.maxZ, 7000), cell: 500 }];
    // land never dips to the water plane: >= 0 m above sea level (0.6 m above the default water level)
    this.surface = new GroundSurface(this.terrain, levels, { minHeight: 0 });
    this.surface.init();
  }

  /** Ground surface height (no road offset / decks); the water level over the water. */
  _groundAt(x, z) { return this.isLand(x, z) ? this.terrain.heightAt(x, z) : this.waterLevel; }

  _buildGroundAndWater() {
    const G = new SolidBuf(6);
    const white = [1, 1, 1];
    const S = this.surface;
    const sink = (buf, c) => ({ vert: (x, y, z, n) => buf.v(x, y, z, n[0], n[1], n[2], c), tri: (a, b, d) => buf.idx.push(a, b, d) });
    let tris = 0;
    if (this.landPolys.length) {
      for (const L of this.landPolys) tris += drapePolygon(S, L.ring, [], 0, sink(G, white));
      for (const L of this.pierPolys) tris += drapePolygon(S, L.ring, [], 0, sink(G, white));
    } else {
      const l = S.levels[S.levels.length - 1];
      tris += drapePolygon(S, [[l.x0, l.z0], [l.x1, l.z0], [l.x1, l.z1], [l.x0, l.z1]], [], 0, sink(G, white));
    }
    this._groundTris = tris;
    this._seawall = null; // seawalls / pier edges are built by the Waterfront (own material)
    const gm = G.build(this.mat.ground); if (gm) { gm.name = 'ground'; this.group.add(gm); }
    // open water, seawalls, piers, rip-rap, boats, distant land (Shore.js / Water.js / Boats.js)
    if (this.landPolys.length) {
      try {
        const SO = this.opts.shore || {};
        this.waterfront = new Waterfront({
          world: this, renderer: this.renderer, level: this.waterLevel, waterMat: this.mat.water, pondMat: this.mat.pond, groundMat: this.mat.ground,
          ferryDock: SO.ferryDock || null, islandFill: SO.islandFill || null, boats: SO.boats || null,
          farland: { ...(SO.farland || {}), ...(this.urls.farland !== undefined ? { url: this.urls.farland } : {}) },
        });
        this.group.add(this.waterfront.build());
      } catch (e) { console.warn('[MapWorld] waterfront failed', e); this.waterfront = null; }
    }
    // ponds / fountains (natural=water inside the city), draped
    const W = new SolidBuf(36);
    for (const w of this.data.water) {
      const r = sanitizeRing(w && w.p); if (!r) continue;
      drapePolygon(S, r, (w.hl || []).map((h) => sanitizeRing(h)).filter(Boolean), 0.07, sink(W, white));
    }
    const wm = W.build(this.mat.pond); if (wm) { wm.name = 'ponds'; this.group.add(wm); }
  }

  _buildFlatAreas() {
    const d = this.data;
    // parks (grass)
    const GS = this.surface;
    const sink = (buf, c) => ({ vert: (x, y, z, n) => buf.v(x, y, z, n[0], n[1], n[2], c), tri: (a, b, e) => buf.idx.push(a, b, e) });
    const Pk = new SolidBuf(4), pkSink = sink(Pk, [1, 1, 1]);
    for (const pk of d.parks) {
      const r = sanitizeRing(pk && pk.p); if (!r) continue;
      drapePolygon(GS, r, (pk.hl || []).map((h) => sanitizeRing(h)).filter(Boolean), 0.03, pkSink);
    }
    const pm = Pk.build(this.mat.grass); if (pm) { pm.name = 'parks'; this.group.add(pm); }
    // plazas -> concrete
    for (const pz of d.plazas) {
      const r = sanitizeRing(pz && pz.p); if (!r) continue;
      drapePolygon(GS, r, (pz.hl || []).map((h) => sanitizeRing(h)).filter(Boolean), 0.08, sink(this.net.solid, [0.8, 0.77, 0.72]));
    }
    // parking lots -> asphalt without markings (into the road buffer)
    const RB = this.net.roadBuf;
    const rbSink = { vert: (x, y, z, n) => RB.v(x, y, z, 0, 0, 1, 0, 0, 0, 0, -1e5, 0, 1e5, 0, [n[0], n[1], n[2]]), tri: (a, b, e) => RB.idx.push(a, b, e) };
    for (const pl of d.parking) {
      const r = sanitizeRing(pl && pl.p); if (!r) continue;
      drapePolygon(GS, r, (pl.hl || []).map((h) => sanitizeRing(h)).filter(Boolean), 0.045, rbSink);
    }
    // seawalls go with the concrete structures
    const S = this.net.solid, W = this._seawall;
    if (W && W.n) {
      const base = S.n;
      for (let i = 0; i < W.n; i++) {
        S.pos.push(W.pos[i * 3], W.pos[i * 3 + 1], W.pos[i * 3 + 2]); S.nor.push(W.nor[i * 3], W.nor[i * 3 + 1], W.nor[i * 3 + 2]);
        S.uv.push(W.uv[i * 2], W.uv[i * 2 + 1]); S.col.push(W.col[i * 3], W.col[i * 3 + 1], W.col[i * 3 + 2]);
      }
      S.n += W.n;
      for (const k of W.idx) S.idx.push(base + k);
    }
  }

  // ================================================================== buildings
  async _buildBuildings(P) {
    if (this.streamOpts) return this._buildBuildingsStreamed(P);
    const b = this.bounds;
    // plugins (e.g. hand-modelled landmarks) claim OSM footprints (`_lm`) so the generic builder skips them
    for (const pl of this.plugins) if (typeof pl.beforeBuildings === 'function') this._runHook(pl, 'beforeBuildings');
    // detail cells ~275 m with the window-unit LOD at 300 m: real openings + instanced window units only near the
    // camera (texture windows are indistinguishable beyond ~250 m); base chunks ~315 m in the district (the four big
    // filler-ring sectors are tiled in _chunkStatic)
    const builder = new BuildingBuilder(b, this._buildingOpts());
    const list = this.data.buildings;
    let n = 0;
    for (let i = 0; i < list.length; i++) {
      const bd = list[i];
      if (!bd) continue;
      if (!bd.lod && this.landPolys.length && Array.isArray(bd.p) && bd.p.length > 2) {
        let sx = 0, sz = 0; for (const q of bd.p) { sx += q[0]; sz += q[1]; }
        if (!this.isLand(sx / bd.p.length, sz / bd.p.length)) bd.water = 1; // bridge towers etc.
      }
      const seedR = mulberry32(((hashStr(String(bd.n || '')) * 1e9) ^ (i * 2654435761)) >>> 0);
      try { if (builder.add(bd, seedR)) n++; } catch (e) { builder.skipped++; }
      if (i % 600 === 599) { P(0.55 + 0.25 * (i / list.length), 'Buildings'); await tick(); }
    }
    this.buildingCount = n - (this.fillerCount || 0);
    for (const m of builder.build(this.mat.facadeSet)) this.group.add(m);
    this.buildingStats = builder.stats;
    this.buildingSamples = builder.samples;
    console.info('[MapWorld] buildings', builder.stats);
    // plugins that build on the merged buildings (their colliders are registered before the buildings')
    for (const pl of this.plugins) if (typeof pl.afterBuildings === 'function') this._runHook(pl, 'afterBuildings');
    P(0.82, 'Colliders');
    for (const [outer, holes, mh, h, base = 0] of builder.colliders) {
      // ground-standing buildings reach 1 m below their lowest footprint point (see Buildings.js)
      const y0 = base + (mh > 0 ? mh : -1);
      for (const bx of polygonToBoxes(outer, holes)) this.colliders.add(bx[0], y0, bx[2], base + h, bx[1], bx[3], 1);
    }
    this.buildingSkipped = builder.skipped;
  }

  // ================================================================== buildings, streamed per tile
  /** BuildingBuilder options (shared by the full build and the tile builders). */
  _buildingOpts() {
    const o = this.opts, BO = o.buildings || {}, raster = this.raster;
    return {
      groundAt: (x, z) => this._groundAt(x, z),
      // street-facing wall test: road / sidewalk just outside the wall
      streetAt: raster ? (x, z) => raster.any(x, z, 1.6, RBIT.ROAD | RBIT.SIDEWALK) : null,
      offset: this.offset,
      baseGrid: BO.baseGrid || o.baseGrid || [7, 7], detailGrid: BO.detailGrid || o.detailGrid || [8, 8], lodDist: BO.lodDist || o.buildingLod || 300,
      detailFloors: BO.detailFloors ?? 4, maxWindows: BO.maxWindows ?? 120000, style: BO.style || null,
    };
  }

  /**
   * Streaming: footprints + colliders of every building now (colliders stay global), geometry per tile on demand
   * (TileStreamer): district tiles = the base chunks of the full build (same buildings, same order, same per-building
   * seeds, so a built tile is identical to its chunk), filler-ring tiles = fillerCell squares: shells + far-LOD shells.
   * Detail tiles (one per detail cell of a district tile) add that cell's near-LOD detail (real openings + window units)
   * within lodDist * 1.05 + detailMargin of the cell's LOD point. Only the tiles within spawnRadius (and pinned ones) are
   * built here; plugins' afterBuildings hooks see those.
   */
  async _buildBuildingsStreamed(P) {
    const SO = this.streamOpts, b = this.bounds;
    for (const pl of this.plugins) if (typeof pl.beforeBuildings === 'function') this._runHook(pl, 'beforeBuildings');
    const bopts = this._bopts = this._buildingOpts();
    const probe = new BuildingBuilder(b, { ...bopts, grid: false });
    const list = this.data.buildings, fps = this._footprints = new Array(list.length);
    const groups = new Map(), FC = SO.fillerCell, pin = typeof SO.pin === 'function' ? SO.pin : null;
    let n = 0, skipped = 0;
    for (let i = 0; i < list.length; i++) {
      const bd = list[i];
      if (!bd) continue;
      if (!bd.lod && this.landPolys.length && Array.isArray(bd.p) && bd.p.length > 2) {
        let sx = 0, sz = 0; for (const q of bd.p) { sx += q[0]; sz += q[1]; }
        if (!this.isLand(sx / bd.p.length, sz / bd.p.length)) bd.water = 1; // bridge towers etc.
      }
      if (bd._lm) continue;
      let fp = null;
      try { fp = probe.footprint(bd); } catch { fp = null; }
      if (!fp) { skipped++; continue; }
      fps[i] = fp; n++;
      const [cx, cz] = fp.c, district = probe.inRegion(cx, cz);
      let gi, gj;
      if (district) { const k = probe.baseIndex(cx, cz); gi = k % probe.bnx; gj = Math.floor(k / probe.bnx); } else { gi = Math.floor(cx / FC); gj = Math.floor(cz / FC); }
      const key = (district ? 'd' : 'f') + gi + ',' + gj;
      let g = groups.get(key);
      if (!g) groups.set(key, (g = { district, i: gi, j: gj, ids: [], minX: Infinity, maxX: -Infinity, minZ: Infinity, maxZ: -Infinity, maxH: 0, cells: new Set(), pinned: false }));
      g.ids.push(i);
      for (const q of fp.outer) { if (q[0] < g.minX) g.minX = q[0]; if (q[0] > g.maxX) g.maxX = q[0]; if (q[1] < g.minZ) g.minZ = q[1]; if (q[1] > g.maxZ) g.maxZ = q[1]; }
      if (fp.h > g.maxH) g.maxH = fp.h;
      if (district && !bd.lod) g.cells.add(probe.cellIndex(cx, cz));
      if (pin && !g.pinned) { try { g.pinned = !!pin(bd, this.data); } catch { /* ignore */ } }
      if (i % 2000 === 1999) { P(0.55 + 0.05 * (i / list.length), 'Buildings'); await tick(); }
    }
    this.buildingCount = n - (this.fillerCount || 0);
    this.buildingSkipped = skipped;
    this.buildingSamples = [];
    // ---- the streamer: building tiles (level 1: shells + far-LOD shells) and, per detail cell of a district tile, a
    // child tile with the near-LOD detail (real openings + window units) that attaches to its parent's THREE.LOD
    const S = this.streamer = new TileStreamer({
      budgetMs: SO.budgetMs, maxConcurrent: SO.maxConcurrent, uploadBytesPerFrame: SO.uploadBytesPerFrame,
      lookAhead: SO.lookAhead, urgentRadius: SO.urgentRadius, altitudeScale: SO.altitudeScale,
    });
    const lodDist = bopts.lodDist, detailR = lodDist * 1.05 + SO.detailMargin;
    // THREE.LOD's metric: camera distance to the detail cell's reference point (cell centre, y = 40)
    const cellDist = (t, x, y, z) => Math.hypot(x - t.cx, y - 40, z - t.cz);
    const LD = S.addLayer({ name: 'district', radius: SO.radius, keep: SO.keep });
    const LF = S.addLayer({ name: 'filler', cell: FC, radius: SO.fillerRadius, keep: SO.fillerKeep });
    const LN = S.addLayer({
      name: 'detail', radius: detailR, keep: SO.detailKeep,
      want: (t, view, _s, px, pz) => {
        const m = Math.min(cellDist(t, view.x, view.y, view.z), cellDist(t, px, view.y, pz));
        t.dd = m; t.d = TileStreamer.rectDist(t, view.x, view.z);
        t.prio = m - lodDist; // < 0: the LOD already shows the near level (late)
        return m <= detailR || (t.level >= 1 && m <= detailR + SO.detailKeep) ? 1 : 0;
      },
    });
    const W2 = (b.maxX - b.minX) / probe.dnx / 2, H2 = (b.maxZ - b.minZ) / probe.dnz / 2;
    for (const g of groups.values()) {
      const t = {
        minX: g.minX, maxX: g.maxX, minZ: g.minZ, maxZ: g.maxZ, reach: Math.min(SO.maxReach, SO.heightReach * g.maxH), pinned: g.pinned,
        ids: g.ids, maxH: g.maxH, objs: null, lods: null, detailedByCell: null,
      };
      t.build = () => this._tileBuild(t);
      t.dispose = () => this._tileDispose(t);
      S.addTile(g.district ? LD : LF, g.i, g.j, t);
      if (!g.district) continue;
      for (const key of [...g.cells].sort((a, c) => a - c)) {
        const [cx, cz] = probe.cellCentre(key);
        const c = { parent: t, cellKey: key, cx, cz, minX: cx - W2, maxX: cx + W2, minZ: cz - H2, maxZ: cz + H2, near: null };
        c.build = () => this._detailBuild(c);
        c.dispose = () => this._detailDispose(c);
        S.addTile(LN, key, g.i * 1000 + g.j, c);
      }
    }
    // pop-in instrumentation: how far from the camera tiles appear / near detail arrives
    const log = this.streamLog = { commits: 0, minDist: Infinity, detailCommits: 0, lateDetail: 0 };
    S.onCommit = (t, level, view, from) => {
      if (!view || !this.loaded) return;
      log.commits++;
      if (t.parent) { log.detailCommits++; if (cellDist(t, view.x, view.y, view.z) < lodDist) log.lateDetail++; }
      else if (from === 0) log.minDist = Math.min(log.minDist, TileStreamer.rectDist(t, view.x, view.z));
    };
    // ---- build the spawn neighbourhood now (the loading screen waits only for these)
    const sp = this._spawn.position, gy = this.getGroundHeight(sp.x, sp.z);
    const t0 = performance.now();
    const eager = await S.prewarm({ x: sp.x, y: gy + 3, z: sp.z, h: 3 }, { radius: SO.spawnRadius, onProgress: (f) => P(0.6 + 0.2 * f, 'Buildings') });
    this.streamLoad = { eagerTiles: eager, tiles: S.tiles.length, ms: Math.round(performance.now() - t0) };
    console.info('[MapWorld] buildings streamed:', this.streamLoad, `(${n} buildings, ${S.tiles.filter((t) => t.pinned).length} pinned tiles)`);
    // plugins that build on the merged buildings (their colliders are registered before the buildings')
    for (const pl of this.plugins) if (typeof pl.afterBuildings === 'function') this._runHook(pl, 'afterBuildings');
    P(0.82, 'Colliders');
    for (let i = 0; i < fps.length; i++) {
      const fp = fps[i]; if (!fp) continue;
      // ground-standing buildings reach 1 m below their lowest footprint point (see Buildings.js)
      const y0 = fp.base + (fp.mh > 0 ? fp.mh : -1);
      for (const bx of polygonToBoxes(fp.outer, fp.holes)) this.colliders.add(bx[0], y0, bx[2], fp.base + fp.h, bx[1], bx[3], 1);
    }
  }

  /** Building tile build (level 1): generator, one building / one geometry per slice; nothing touches the scene. */
  * _tileBuild(t) {
    const bld = new TileBuildingBuilder(this.bounds, this._bopts, 'low');
    const list = this.data.buildings, fps = this._footprints, byCell = new Map();
    for (const i of t.ids) {
      const bd = list[i];
      const seedR = mulberry32(((hashStr(String(bd.n || '')) * 1e9) ^ (i * 2654435761)) >>> 0); // as the full build
      try {
        if (bld.add(bd, seedR, fps[i]) && bld.lastDetailed) {
          const k = bld.cellIndex(fps[i].c[0], fps[i].c[1]);
          let a = byCell.get(k); if (!a) byCell.set(k, (a = [])); a.push(i);
        }
      } catch (e) { bld.skipped++; }
      yield;
    }
    const out = yield* bld.meshes(this.mat.facadeSet);
    t.stats = { detailed: bld.detailed, tris: out.tris };
    return { bytes: out.bytes, attach: () => this._tileAttach(t, out, byCell), discard: () => this._discardOut(out) };
  }

  _discardOut(out) {
    if (out.base) disposeStreamed(out.base);
    for (const e of out.cells) { if (e.far) disposeStreamed(e.far); if (e.near) disposeStreamed(e.near); }
  }

  _tileAttach(t, out, byCell) {
    const lodDist = this._bopts.lodDist;
    t.objs = []; t.lods = new Map(); t.detailedByCell = byCell;
    // tiles attached after load get their local origins here (load-time tiles: _chunkStatic, after the plugin hooks)
    if (this.loaded) { if (out.base) this._localizeBuildings(out.base); for (const e of out.cells) if (e.far) this._localizeBuildings(e.far); }
    if (out.base) { out.base.userData.streamed = true; this.group.add(out.base); t.objs.push(out.base); }
    for (const e of out.cells) {
      // as BuildingBuilder.build: a THREE.LOD at the detail cell centre; far level now, near level from the detail tile
      const lod = new THREE.LOD();
      lod.name = 'buildings-detail'; lod.userData.streamed = true;
      lod.position.set(e.cx, 40, e.cz);
      const far = e.far || new THREE.Object3D();
      lod.addLevel(far, lodDist, 0.05);
      lod.updateMatrix();
      this.group.add(lod); t.objs.push(lod);
      t.lods.set(e.key, { lod, far });
    }
  }

  _tileDispose(t) {
    if (t.objs) for (const o of t.objs) { this.group.remove(o); disposeStreamed(o); }
    t.objs = null; t.lods = null; t.detailedByCell = null;
  }

  /** Detail tile build (near-LOD walls + window units of one detail cell of a built district tile). */
  * _detailBuild(c) {
    const p = c.parent, ids = (p.detailedByCell && p.detailedByCell.get(c.cellKey)) || [];
    const bld = new TileBuildingBuilder(this.bounds, this._bopts, 'near');
    const list = this.data.buildings, fps = this._footprints;
    for (const i of ids) {
      const bd = list[i];
      const seedR = mulberry32(((hashStr(String(bd.n || '')) * 1e9) ^ (i * 2654435761)) >>> 0);
      try { bld.add(bd, seedR, fps[i]); } catch (e) { bld.skipped++; }
      yield;
    }
    const out = yield* bld.meshes(this.mat.facadeSet);
    const e = out.cells.find((q) => q.key === c.cellKey && q.near) || null;
    for (const q of out.cells) if (q !== e && q.near) disposeStreamed(q.near);
    return {
      bytes: out.bytes,
      attach: () => {
        const L = p.lods && p.lods.get(c.cellKey);
        if (!e) return;
        if (!L) { disposeStreamed(e.near); return; }
        this._localizeBuildings(e.near);
        L.lod.addLevel(e.near, 0); c.near = e.near; c.lod = L;
      },
      discard: () => { if (e) disposeStreamed(e.near); },
    };
  }

  /** Detail tile -> 0: take the near level out of the parent's LOD (far shells stay) and free it. */
  _detailDispose(c) {
    if (!c.near) return;
    const L = c.lod, lv = L.lod.levels, i = lv.findIndex((l) => l.object === c.near);
    if (i >= 0) lv.splice(i, 1);
    L.lod.remove(c.near);
    disposeStreamed(c.near);
    L.far.visible = true; L.lod._currentLevel = 0;
    c.near = null; c.lod = null;
  }

  /** Building stats of the resident tiles (streaming) or of the full build. */
  get buildingTiles() {
    if (!this.streamer) return null;
    const S = this.streamer, by = { district: [0, 0], filler: [0, 0], detail: [0, 0] };
    for (const t of S.tiles) by[t.layer.name][t.level ? 1 : 0]++;
    return { district: { built: by.district[1], of: by.district[0] + by.district[1] }, detail: { built: by.detail[1], of: by.detail[0] + by.detail[1] }, filler: { built: by.filler[1], of: by.filler[0] + by.filler[1] }, residentMB: +(S.stats.residentBytes / 1e6).toFixed(1) };
  }

  // ================================================================== context ring (low detail)
  /** Bake-frame circles kept free of filler lots: o.clearZones + every plugin's clearZones(). */
  _clearZones() {
    const zones = [...(this.opts.clearZones || [])];
    for (const pl of this.plugins) if (typeof pl.clearZones === 'function') { try { zones.push(...(pl.clearZones() || [])); } catch (e) { console.warn('[MapWorld] clearZones failed', e); } }
    return zones;
  }

  _makeFiller() {
    const d = this.data, b = this.bounds;
    const cx = (b.minX + b.maxX) / 2, cz = (b.minZ + b.maxZ) / 2;
    const halfW = Math.max(b.maxX - b.minX, b.maxZ - b.minZ) / 2;
    const Rout = halfW + 1000;
    // dominant street-grid angle per quadrant from the real roads
    const acc = [[0, 0], [0, 0], [0, 0], [0, 0]];
    const quad = (x, z) => (x >= cx ? (z >= cz ? 0 : 1) : (z >= cz ? 3 : 2));
    for (const r of d.roads) {
      if (!r || !r.p || /^(footway|path|steps|cycleway|service|pedestrian|track)$/.test(r.c)) continue;
      for (let i = 0; i + 1 < r.p.length; i++) {
        const a = r.p[i], c = r.p[i + 1], dx = c[0] - a[0], dz = c[1] - a[1], L = Math.hypot(dx, dz);
        if (L < 5) continue;
        const th = Math.atan2(dz, dx) * 4, q = quad((a[0] + c[0]) / 2, (a[1] + c[1]) / 2);
        acc[q][0] += Math.cos(th) * L; acc[q][1] += Math.sin(th) * L;
      }
    }
    const angles = acc.map(([c, s]) => Math.atan2(s, c) / 4);
    const rnd = mulberry32(777);
    const BW = 128, BH = 92, street = 9.5;
    const seen = new Set();
    let count = 0;
    const inCity = (x, z) => x > b.minX - 25 && x < b.maxX + 25 && z > b.minZ - 25 && z < b.maxZ + 25;
    // keep lots off clear zones (plugin landmark sites etc.); zones are in the bake frame
    const off = this.offset || { x: 0, z: 0 };
    const clear = this._clearZones().map((c) => ({ x: c.x - off.x, z: c.z - off.z, r: c.r }));
    const blocked = (x, z) => clear.some((c) => Math.hypot(x - c.x, z - c.z) < c.r);
    const roofCols = ['#e6e6e2', '#d8d6d0', '#7d7d7b', '#5b5b5c', '#9a968e', '#b9b4aa', '#6e7466'];
    for (let q = 0; q < 4; q++) {
      const th = angles[q], ux = Math.cos(th), uz = Math.sin(th), vx = -uz, vz = ux;
      const N = Math.ceil(Rout / Math.min(BW, BH)) + 1;
      for (let i = -N; i <= N; i++) for (let k = -N; k <= N; k++) {
        const bx = cx + ux * i * BW + vx * k * BH, bz = cz + uz * i * BW + vz * k * BH;
        if (Math.hypot(bx - cx, bz - cz) > Rout) continue;
        const C = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([a, c]) => [bx + ux * a * BW / 2 + vx * c * BH / 2, bz + uz * a * BW / 2 + vz * c * BH / 2]);
        if (C.some((p) => quad(p[0], p[1]) !== q)) continue;
        if (C.some((p) => inCity(p[0], p[1])) || inCity(bx, bz)) continue;
        if (!C.every((p) => this.isLand(p[0], p[1])) || !this.isLand(bx, bz)) continue;
        // streets around the block (deduplicated)
        for (let e = 0; e < 4; e++) {
          const p0 = C[e], p1 = C[(e + 1) % 4];
          const kk = `${Math.round((p0[0] + p1[0]) / 4)},${Math.round((p0[1] + p1[1]) / 4)}`;
          if (seen.has(kk)) continue; seen.add(kk);
          d.roads.push({ p: [[p0[0], p0[1]], [p1[0], p1[1]]], c: 'residential', l: 2, filler: 1 });
        }
        // lots
        const dist = Math.hypot(bx - cx, bz - cz);
        const t = clamp((dist - halfW) / 1000, 0, 1);
        const hBase = lerp(30, 11, t);
        const iw = BW - 2 * street - 2, ih = BH - 2 * street - 2;
        const nx = 3 + Math.floor(rnd() * 3), nz = 2;
        for (let a = 0; a < nx; a++) for (let c = 0; c < nz; c++) {
          if (rnd() < 0.06) continue;
          const l0 = -iw / 2 + (iw * a) / nx + 0.4, l1 = -iw / 2 + (iw * (a + 1)) / nx - 0.4;
          const m0 = -ih / 2 + (ih * c) / nz + 0.4, m1 = -ih / 2 + (ih * (c + 1)) / nz - 0.4;
          const ring = [[l0, m0], [l1, m0], [l1, m1], [l0, m1]].map(([p, s]) => [bx + ux * p + vx * s, bz + uz * p + vz * s]);
          let h = hBase * (0.45 + rnd() * 1.05);
          if (rnd() < 0.04 * (1 - t)) h += 30 + rnd() * 70;
          else if (rnd() < 0.08) h *= 1.6; // a few mid-rises break up the skyline
          const lc = ring.reduce((a, q) => [a[0] + q[0] / 4, a[1] + q[1] / 4], [0, 0]);
          if (blocked(lc[0], lc[1])) continue;
          const fb = { p: ring, h: Math.round(h * 10) / 10, t: rnd() < 0.55 ? 'apartments' : rnd() < 0.5 ? 'commercial' : 'yes', lod: 1 };
          if (rnd() < 0.6) fb.rc = roofCols[Math.floor(rnd() * roofCols.length)];
          if (h < 13 && rnd() < 0.22) { fb.r = rnd() < 0.6 ? 'gabled' : 'hipped'; fb.rc = rnd() < 0.5 ? '#6a5d55' : '#8c5444'; }
          d.buildings.push(fb);
          count++;
        }
      }
    }
    this.fillerCount = count;
    this._fillerGrid = { cx, cz, Rout, angles, BW, BH, street };
    this._makeOuterLots(angles, cx, cz, Rout, BW, BH, street, inCity, blocked);
  }

  /**
   * Outer land (beyond the filler ring, to the map edge): low context blocks on the same street grid,
   * drawn as one instanced mesh by the Waterfront (Shore.js) instead of full filler buildings.
   */
  _makeOuterLots(angles, cx, cz, Rout, BW, BH, street, inCity, blocked) {
    const OB = this.outerBox, rnd = mulberry32(4243);
    const lots = this._outerLots = [];
    const quad = (x, z) => (x >= cx ? (z >= cz ? 0 : 1) : (z >= cz ? 3 : 2));
    const maxD = Math.max(...[[OB.minX, OB.minZ], [OB.maxX, OB.minZ], [OB.minX, OB.maxZ], [OB.maxX, OB.maxZ]].map(([x, z]) => Math.hypot(x - cx, z - cz)));
    const inBox = (x, z) => x > OB.minX + 30 && x < OB.maxX - 30 && z > OB.minZ + 30 && z < OB.maxZ - 30;
    // mainland only (the islands get their own treatment in Shore.js)
    let main = null, best = 0;
    for (const L of this.landPolys) { const a = Math.abs(signedArea(L.ring)); if (a > best) { best = a; main = L; } }
    if (!main) return;
    const onMain = (x, z) => (main.idx ? main.idx.contains(x, z) : pointInRing(x, z, main.ring));
    for (let q = 0; q < 4; q++) {
      const th = angles[q], ux = Math.cos(th), uz = Math.sin(th), vx = -uz, vz = ux;
      const N = Math.ceil(maxD / Math.min(BW, BH)) + 1;
      for (let i = -N; i <= N; i++) for (let k = -N; k <= N; k++) {
        const bx = cx + ux * i * BW + vx * k * BH, bz = cz + uz * i * BW + vz * k * BH;
        const dist = Math.hypot(bx - cx, bz - cz);
        if (dist <= Rout || !inBox(bx, bz) || quad(bx, bz) !== q || inCity(bx, bz)) continue;
        const C = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([a, c]) => [bx + ux * a * BW / 2 + vx * c * BH / 2, bz + uz * a * BW / 2 + vz * c * BH / 2]);
        if (C.some((p) => quad(p[0], p[1]) !== q || !inBox(p[0], p[1]))) continue;
        if (!C.every((p) => onMain(p[0], p[1])) || !onMain(bx, bz) || blocked(bx, bz)) continue;
        const hc = this.terrain ? this.terrain.rawHeightAt(bx, bz) : 0; // headless (dev/traffic-sim.mjs): flat
        if (hc > 150) continue; // the high hills stay natural
        const iw = BW - 2 * street - 4, ih = BH - 2 * street - 4;
        const far = dist > 3300;
        const nx = far ? 1 + Math.floor(rnd() * 2) : 2 + Math.floor(rnd() * 2), nz = far ? 1 : 2;
        for (let a = 0; a < nx; a++) for (let c = 0; c < nz; c++) {
          if (rnd() < 0.08) continue;
          const l0 = -iw / 2 + (iw * a) / nx + 1.2, l1 = -iw / 2 + (iw * (a + 1)) / nx - 1.2;
          const m0 = -ih / 2 + (ih * c) / nz + 1.2, m1 = -ih / 2 + (ih * (c + 1)) / nz - 1.2;
          const lcx = (l0 + l1) / 2, lcz = (m0 + m1) / 2;
          let h = 7 + rnd() * 7;
          if (rnd() < 0.06) h += 10 + rnd() * 22;
          lots.push([bx + ux * lcx + vx * lcz, bz + uz * lcx + vz * lcz, l1 - l0, m1 - m0, th, h, Math.floor(rnd() * 7)]);
        }
      }
    }
  }

  // ================================================================== culling
  /**
   * Split the big merged static meshes into tiles (MeshChunker) so the main view and both shadow cascades cull
   * them per tile instead of shading the whole city in every pass. Runs last: plugins edit the merged building /
   * structure geometry in place while they build; plugin chunk() hooks add their own big meshes.
   */
  _chunkStatic() {
    const t0 = performance.now();
    const o = this.opts, C = o.chunk || {}, b = this.bounds;
    const cx = (b.minX + b.maxX) / 2, cz = (b.minZ + b.maxZ) / 2;
    const half = Math.max(b.maxX - b.minX, b.maxZ - b.minZ) / 2;
    const ring = (inner, outer) => (x, z) => { const d = Math.max(Math.abs(x - cx), Math.abs(z - cz)); return d < half + 60 ? inner : d < half + 1100 ? outer : outer * 4; };
    // tile size (m) inside the detailed district; x2 in the filler ring, x8 beyond (0 = keep merged). Measured at
    // 1024x768: 320 m building / structure tiles take ~1.4 M triangles out of the shadow passes for ~+50 draw calls
    // in the main pass; the ground (main pass only, fragment-bound) gains little, so its tiles are large.
    const plan = {
      buildings: C.buildings ?? o.chunkBuildings ?? 320,
      'sidewalks+structures': C.structures ?? o.chunkStructures ?? 320,
      ground: C.ground ?? o.chunkGround ?? 640,
    };
    let tiles = 0;
    for (const m of this.group.children.slice()) {
      const c = m.isMesh && !m.userData.streamed ? plan[m.name] : 0; // streamed building tiles are tile-sized already
      if (!c || (m.geometry.boundingSphere && m.geometry.boundingSphere.radius < c)) continue; // already tile-sized
      const r = replaceWithChunks(m, { cellAt: ring(c, c * 2), minTris: 64 });
      tiles += r === m ? 1 : r.children.length;
    }
    // plugin meshes (e.g. a long bridge's steel: one 3 km mesh -> 400 m pieces, only the near ones reach the shadow maps)
    for (const pl of this.plugins) {
      if (typeof pl.chunk !== 'function' || this._failed.has(pl) || (pl.name && !this[pl.name])) continue;
      for (const e of pl.chunk(this) || []) if (e && e.mesh && e.cell) tiles += replaceWithChunks(e.mesh, { cell: e.cell }).children.length || 1;
    }
    // building shells (base chunks / tiles and the LOD far + near walls) move to their own XZ centre: the renderer's
    // front-to-back sort of opaque objects then draws the nearest shells first instead of in creation order (every
    // merged mesh used to sit at the origin); plugin hooks that edit merged geometry by world coordinates ran above
    this._localizeBuildings(this.group);
    this._chunkTiles = tiles;
    this._chunkMs = Math.round(performance.now() - t0);
  }

  /** Give the building shell meshes under root a local origin (MeshChunker.localizeMeshOrigin). */
  _localizeBuildings(root) {
    if (!root) return;
    root.traverse((m) => { if (m.isMesh && BUILDING_SHELL_NAMES.has(m.name)) localizeMeshOrigin(m); });
  }

  // ================================================================== runtime
  _applyTime() {
    const a = this.atmo;
    a.maxBuildingHeight = Math.min(330, this.maxBuildingHeight || 200);
    const u = this.mat.facade.userData.uniforms;
    u.uNight.value = a.windowGlow;
    u.uLitFrac.value = 0.2 + 0.14 * a.darkness;
    this._timeVersion = a.version;
  }

  /** Change time of day at runtime (hours 0..24 or preset name). */
  setTimeOfDay(t) {
    if (!this.atmo) return;
    const h = typeof t === 'number' ? t : (this.atmo.timePresets || TIME_PRESETS)[t];
    if (!Number.isFinite(h)) return;
    this.atmo.setTime(h);
    this._applyTime();
  }
  get timeOfDay() { return this.atmo ? this.atmo.hours : null; }

  update(dt, cameraPos) {
    if (!this.loaded) return;
    dt = Number.isFinite(dt) ? clamp(dt, 0, 0.25) : 0;
    this.time += dt;
    if (this.streamer && cameraPos) {
      const v = this._streamView || (this._streamView = { x: 0, y: 0, z: 0, h: 0 });
      v.x = cameraPos.x; v.y = cameraPos.y; v.z = cameraPos.z; v.h = cameraPos.y - this.getGroundHeight(cameraPos.x, cameraPos.z);
      v.extra = this.isLand(cameraPos.x, cameraPos.z) ? 0 : this.streamOpts.openRadius;
      const S = this.streamer;
      S.update(v, dt);
      // tile builds reuse pooled scratch buffers; give them back to the GC after 5 s without streaming work
      this._streamIdle = S.stats.pending || S.stats.building ? 0 : (this._streamIdle || 0) + dt;
      if (this._streamIdle > 5 && !this._poolFreed) { releaseTileBuffers(); this._poolFreed = true; } else if (!this._streamIdle) this._poolFreed = false;
    }
    if (this.atmo.version !== this._timeVersion) this._applyTime();
    if (cameraPos) this.atmo.update(cameraPos);
    if (this.waterfront) this.waterfront.update(dt, cameraPos, this.time, this.atmo);
    else this.mat.water.userData.u.uTime.value = this.time;
    this.props.update(dt, cameraPos, this.atmo.darkness, this.time, this._viewInfo(cameraPos));
    for (const pl of this.plugins) {
      if (typeof pl.update !== 'function' || this._failed.has(pl) || (pl.name && !this[pl.name])) continue;
      pl.update(dt, cameraPos, this.atmo.darkness, this);
    }
    if (this.traffic) this.traffic.update(dt, cameraPos, this.atmo.darkness);
  }

  /**
   * Engine system adapter: engine.addSystem(world.system({ camera })) loads the world with the engine and updates it
   * in lateUpdate (after the camera has moved). Sets atmo.camera after load (shadow cascades lead the view).
   */
  system({ camera = null, loadWeight = 1, onLoad = null } = {}) {
    const pos = new THREE.Vector3();
    return {
      loadWeight,
      load: async (progress) => { await this.load(progress); if (camera && this.atmo) this.atmo.camera = camera; if (onLoad) await onLoad(this); },
      lateUpdate: (dt) => this.update(dt, camera ? pos.copy(camera.position) : null),
    };
  }

  /** PhysicsWorld view of the world (static collider grid + ground query + traffic boxes). */
  createPhysics(o = {}) {
    const tr = this.traffic;
    this.physics = PhysicsWorld.fromWorld(this, { dynamic: tr && tr.queryColliders ? [(x, z, r, out) => tr.queryColliders(x, z, r, out)] : [], deckStep: 0.35, ...o });
    return this.physics;
  }

  /**
   * This frame's view frustum + direction of the camera the Atmosphere follows (null until the game sets
   * atmo.camera). Built from the camera's current position / quaternion: its matrixWorld is only refreshed by the
   * renderer, i.e. one frame late at update time (the camera state itself is left untouched).
   */
  _viewInfo(cameraPos) {
    const cam = this.atmo && this.atmo.camera;
    if (!cam || !cam.projectionMatrix || !cameraPos) return null;
    const v = this._view || (this._view = { position: new THREE.Vector3(), frustum: new THREE.Frustum(), direction: new THREE.Vector3(), _w: new THREE.Matrix4(), _m: new THREE.Matrix4(), frame: 0 });
    v._w.compose(cam.position, cam.quaternion, cam.scale);
    if (cam.parent) v._w.premultiply(cam.parent.matrixWorld);
    v._m.copy(v._w).invert().premultiply(cam.projectionMatrix);
    v.frustum.setFromProjectionMatrix(v._m);
    v.direction.set(-v._w.elements[8], -v._w.elements[9], -v._w.elements[10]).normalize();
    v.position.copy(cameraPos);
    v.frame++;
    return v;
  }

  /**
   * Height of the drivable surface at (x,z): the terrain ground surface on land (MSL metres; +0.05 on
   * asphalt), the water level over the water, or the deck height of an elevated road. Optional `y`
   * (current height of the querying object): decks more than `step` m above `y` (default 1.6, the traffic lane
   * bake walks lanes with it) are ignored, so cars can drive UNDER bridges; and where the terrain is far above
   * `y` but a deck is below it (a deck running into a hill, e.g. a tunnel portal) the deck wins. Without `y` the
   * highest surface wins. The vehicle physics passes a wheel's lowest contact height and step 0.35
   * (PhysicsWorld.fromWorld deckStep): a deck a car cannot climb must not become its ground (it used to
   * teleport 1-1.6 m up onto low ramps).
   */
  getGroundHeight(x, z, y, step = 1.6) {
    if (!Number.isFinite(x) || !Number.isFinite(z)) return 0;
    let g = this.waterLevel;
    if (this.isLand(x, z)) {
      g = this.terrain ? this.terrain.heightAt(x, z) : 0;
      g += this.pavingLift(x, z);
    }
    if (this.deckHash) {
      const lim = Number.isFinite(y) ? y + step : Infinity;
      let deck = -Infinity;
      this.deckHash.forEachNear(x, z, 0, (s, t) => {
        const h = s.ha + (s.hb - s.ha) * t;
        if (h > deck && h <= lim) deck = h;
      });
      if (deck > g || (deck > -Infinity && g > lim)) g = deck;
    }
    return g;
  }

  /**
   * Height of the paving above the ground mesh at (x,z), matching what RoadNetwork draws: asphalt ribbons sit
   * ROAD_LIFT above the surface (read from the exact ribbon geometry, net.segHash, with a short ramp outside the
   * edge instead of the 1 m raster pixels, which made single wheels hop 5 cm along a lane), sidewalk slabs and
   * kerb tops SIDEWALK_LIFT (raster), bare ground 0.
   */
  pavingLift(x, z) {
    const w = this.carriagewayAt(x, z);
    if (w >= 1) return ROAD_LIFT;
    if (this.raster) {
      const bits = this.raster.get(x, z);
      if ((bits & RBIT.SIDEWALK) && !(bits & (RBIT.ROAD | RBIT.SERVICE))) return SIDEWALK_LIFT;
    }
    return w * ROAD_LIFT;
  }

  /** 1 inside a ground-level carriageway (exact ribbon geometry), fading to 0 over ROAD_EDGE_RAMP outside its edge. */
  carriagewayAt(x, z) {
    let w = 0;
    const net = this.net;
    if (net && net.segHash) {
      net.segHash.forEachNear(x, z, ROAD_EDGE_RAMP, (sg, t, lat) => {
        const e = sg.hw - Math.abs(lat);                       // > 0 inside the carriageway
        const k = e >= 0 ? 1 : 1 + e / ROAD_EDGE_RAMP;         // fades to 0 over the ramp outside the edge
        if (k > w) w = k;
      });
    } else if (this.raster && (this.raster.get(x, z) & (RBIT.ROAD | RBIT.SERVICE))) w = 1;
    return w;
  }

  /**
   * Tyre surface at (x,z) for the vehicle physics (PhysicsWorld.setSurfaceProvider contract): grip and rolling-
   * resistance multipliers + tag. Asphalt 1/1, sidewalk paving, park grass, bare ground, and the bay: water is
   * not a road (almost no grip, enormous drag), so a car that drives into it bogs down and needs its thrusters.
   */
  surfaceAt(x, z, out) {
    out.mu = 1; out.crr = 1; out.tag = 'asphalt';
    if (!Number.isFinite(x) || !Number.isFinite(z)) return out;
    if (!this.isLand(x, z)) { out.mu = SURFACES.water.mu; out.crr = SURFACES.water.crr; out.tag = 'water'; return out; }
    if (this.carriagewayAt(x, z) >= 1) return out;
    const bits = this.raster ? this.raster.get(x, z) : 0;
    if (bits & (RBIT.ROAD | RBIT.SERVICE)) return out;       // junction fills the ribbon test misses
    const S = (bits & RBIT.SIDEWALK) ? SURFACES.paving : (bits & RBIT.PARK) ? SURFACES.grass : SURFACES.ground;
    out.mu = S.mu; out.crr = S.crr; out.tag = S.tag;
    return out;
  }

  /** Static AABB colliders near (x,z): [{minX,minY,maxX,maxY,minZ,maxZ,kind}] (shared objects - read only). */
  queryColliders(x, z, radius) {
    const out = this.colliders.query(x, z, radius, []);
    if (this.traffic) this.traffic.queryColliders(x, z, radius, out);
    return out;
  }

  /** Spawn pose on a wide real road near the district centre. headingRad: rotation.y such that local -Z points along the road. */
  getSpawn() {
    return { position: this._spawn.position.clone(), headingRad: this._spawn.headingRad };
  }

  getMinimapData() {
    const net = this.net;
    const parks = (this.data?.parks || []).map((p) => p.p);
    return {
      roads: net ? net.minimap : [],
      roadClasses: net ? net.minimapClass : [],
      bounds: { ...this.bounds },
      buildings: (this.data?.buildings || []).filter((b) => !b.lod && b.p).map((b) => b.p),
      land: this.landPolys.map((l) => l.ring),
      parks,
      spawn: [0, 0],
    };
  }

  get sunDirection() {
    if (!this.atmo) return this._sunOut.set(0, 1, 0);
    return this._sunOut.copy(this.atmo.sunDir);
  }
}
