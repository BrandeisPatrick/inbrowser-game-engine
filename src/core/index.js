export { Engine } from './Engine.js';
export { Profiler, ProfilerOverlay } from './Profiler.js';
export { Assets } from './Assets.js';
export { Events } from './Events.js';
export { Clock } from './Clock.js';
export { Entity, EntitySystem } from './Entity.js';
// deterministic math for simulation code (IEEE-exact: + - * / sqrt only); mulberry32 / clamp live under MathDet
// to avoid clashing with the render and world layers' exports
export { hashU32, hashF64, hashF64Array, detAtan, detAtan2, detAsin, detAcos, detSqrt } from './mathDet.js';
export * as MathDet from './mathDet.js';
// browser measurement harness (window.__sample): fixed poses, GPU-synced bench, captures, diffs
export { createBenchHarness, readBenchParams } from './Bench.js';
