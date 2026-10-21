// Physics layer: pure JS, deterministic, Node-safe (no DOM, THREE math only).
export { PhysicsWorld, LAYER, layerOf } from './PhysicsWorld.js';
export { VehicleController, ROADSTER_SPEC, resolveVehicleSpec, G, WHEEL_LOCAL, THRUSTER } from './VehicleController.js';
export { obbVsAabb, obbFromPose, sweepToContact, bounceVelocity, rayAabb, sphereCastAabb, closestPointOnAabb, aabbOverlap } from './collision.js';
export { ColliderGrid, polygonToBoxes, SegmentHash, Raster, MinHeap } from './spatial.js';
