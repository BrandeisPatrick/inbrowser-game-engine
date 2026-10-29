// Browser game engine extracted from the Roadster flying-car game. See ../../ENGINE.md.
// Each layer folder is independently importable (engine/src/<layer>/index.js); lower layers never import higher ones.
// core: Engine, Profiler, Assets, Events, Clock, Entity / EntitySystem, deterministic math, bench harness
export * from './core/index.js';
// physics (pure JS, Node-safe): PhysicsWorld, VehicleController, collision helpers, spatial structures
export * from './physics/index.js';
// input / camera
export * from './input/index.js';
export * from './camera/index.js';
export * from './render/index.js';
// anim: animation state machine / blend trees over AnimationMixer, IK
export * from './anim/index.js';
export * from './audio/index.js';
export * from './ui/index.js';
export * from './world/index.js';
