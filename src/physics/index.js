// Physics layer: pure JS, deterministic, Node-safe (no DOM, THREE math only). PhysicsWorld is the query hub (static
// AABB grid, heightfield, dynamic providers, casts) the vehicle and character controllers drive on; DynamicsWorld
// (dynamics/) is the unified XPBD rigid-body solver that plugs into it through attachToPhysicsWorld (0.6.0).
export { PhysicsWorld, LAYER, layerOf, namedLayers, setNamedLayers } from './PhysicsWorld.js';
export { VehicleController, ROADSTER_SPEC, resolveVehicleSpec, G, WHEEL_LOCAL, THRUSTER } from './VehicleController.js';
export { obbVsAabb, obbFromPose, sweepToContact, bounceVelocity, rayAabb, sphereCastAabb, closestPointOnAabb, aabbOverlap } from './collision.js';
export { ColliderGrid, polygonToBoxes, SegmentHash, Raster, MinHeap } from './spatial.js';
export { CharacterController } from './CharacterController.js';
// ---- dynamics (0.6.0): unified XPBD rigid bodies, joints, queries, worker ----------------------------------------
export { DynamicsWorld, describeShape, shapeFromDescriptor } from './dynamics/DynamicsWorld.js';
export { DynamicsClient, dynamicsWorkerAvailable, POSE_STRIDE as DYNAMICS_POSE_STRIDE, CONTACT_STRIDE as DYNAMICS_CONTACT_STRIDE } from './DynamicsClient.js';
export { BodyStore, ContactBuffer, BODY, FLAG, SHAPE, WORLD_BODY, MAX_BODIES, DYN_RECORD, pairKey, pairA, pairB } from './dynamics/state.js';
export * as Shapes from './dynamics/shapes.js';
export { sphere as sphereShape, capsule as capsuleShape, box as boxShape, convex as convexShape, trimesh as trimeshShape, compound as compoundShape, SHAPE_NAMES } from './dynamics/shapes.js';
export { Broadphase, radixSortU32 } from './dynamics/broadphase.js';
export { Narrowphase } from './dynamics/narrowphase.js';
export { JOINT, JointStore, solveJointsPositions, solveJointsVelocities, jointAngle, jointSwing, jointDistance } from './dynamics/joints.js';
export * as XPBD from './dynamics/xpbd.js';
export { Islands } from './dynamics/islands.js';
export { raycastBodies, shapeCastBodies, overlapBodies, pointInBody, meshBounds } from './dynamics/queries.js';
export { attachToPhysicsWorld, detachFromPhysicsWorld, syncKinematicFromObject } from './dynamics/bridge.js';
export { writeDebugLines, DEBUG_LINES } from './dynamics/debug.js';
// ---- unified XPBD (0.9.0): particles / cloth / rope, soft bodies, particle-rigid coupling + buoyancy, fracture --------
export { ParticleStore, ParticleConstraints, PFLAG, PCON, Cloth, Rope } from './dynamics/particles.js';
export { SoftBody, SoftSkin, VolumeConstraints, tetrahedralize } from './dynamics/softbody.js';
export { ParticleCollider, Buoyancy, closestPointOnShape, sampleShape } from './dynamics/coupling.js';
export { FracturedBody, fracturePieces, splitConvex, FRACTURE_BREAK_FORCE, FRACTURE_COMPLIANCE } from './dynamics/fracture.js';
