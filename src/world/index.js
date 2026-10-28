// World layer: data-driven big static worlds. MapWorld composes everything from a city.json + terrain.json (+ farland.json)
// with game options and plugins; the systems are usable on their own too. Nothing here imports src/ (game code).
export { MapWorld, DEFAULT_WATER_LEVEL, DEFAULT_STREAMING, ROAD_LIFT, SIDEWALK_LIFT, SURFACES } from './MapWorld.js';
export { TileStreamer } from './TileStreamer.js';
export { MAP_KEYS, assetBase, validateMapData, loadMapData, prepareMapData, findSpawn, generateProceduralCity } from './MapData.js';
export { Terrain, GroundSurface, sampleLevel, drapePolygon, clipConvex, latticeSplit } from './Terrain.js';
export { RoadNetwork, RBIT, roadStyle, createRoadMaterial, createTactileMaterial, SolidBuf, samplePolyline, quadUp, DEFAULT_DECK_HOLD } from './RoadNetwork.js';
export {
  BuildingBuilder, TileBuildingBuilder, NullBuf, GeoBuf, createFacadeMaterials, MODE, enc, tintBytes, tintOf, insetRing, offsetRing, longAxis, UNIT_WINDOW_TRIS, DEFAULT_BUILDING_STYLE,
} from './Buildings.js';
export { FACADE, WIN_RECTS, makeFacadeTextures, PBR, PBR_SETS, GROUND_SETS, loadPbrTextures } from './FacadeTextures.js';
export { Props } from './Props.js';
export {
  FM, FURNITURE_META, FURNITURE_TYPES, makeFurnitureGeometry, makeShelterGlassGeometry, makePanelGeometry, makeDecalGeometry,
  createFurnitureMaterial, createPanelMaterial, createShelterGlassMaterial, formatStreetName, buildSignAtlas, DEFAULT_SIGN_STYLE,
} from './Furniture.js';
export { PartBuilder, VM, VEHICLE_SPECS, VEHICLE_TYPES, DEFAULT_BUS_LIVERY, makeVehicleGeometry, pickVehicleColor, createVehicleMaterial } from './Vehicles.js';
export { Traffic } from './Traffic.js';
export { TrafficSim, TRAFFIC_TYPES, POSE_STRIDE, buildSignalInfos, playerInfoFrom, typeCounts } from './TrafficSim.js';
export { buildTreeMeshes, bakeVegImpostors, IMP_VIEWS, impostorView } from './Vegetation.js';
export { SPECIES, getVegModels } from './VegModels.js';
export { makeVegAtlas, vegUV, VEG_ATLAS_SIZE, VEG_REGIONS } from './VegTextures.js';
export { createVegMaterial, createImpostorMaterial as createVegImpostorMaterial } from './VegMaterials.js';
export {
  Water, WAVES, NWAVES, waveHeight, waveSlope, waveUniforms, WAVE_GLSL, makeWaterNormalMap, makeNoiseTexture, shoreSegments, buildShoreField, createWaterMaterial,
} from './Water.js';
export { Waterfront, createGroundMaterial, bakeLandClass, buildFarLand, shoreTextures } from './Shore.js';
export { Boats, ferryGeometry, shipGeometry, sailboatGeometry, fishingBoatGeometry, launchGeometry, createBoatMaterial } from './Boats.js';
export { hashStr, signedArea, pointInRing, pointInPoly, ringBounds, centroid, sanitizeRing, polylineLength, RingIndex } from './geom.js';
// helpers for big static worlds (tiled culling, per-pass shadow casters, real-light pools)
export { chunkGeometry, chunkMesh, replaceWithChunks, chunkStats, localizeMeshOrigin } from './MeshChunker.js';
export { shadowCount, shadowDrawRange, skipShadowPasses, shadowFrustum, NEAR_SHADOW_LAYER } from './ShadowPasses.js';
export { NearestLights } from './NearestLights.js';
export { clamp } from './geom.js';
