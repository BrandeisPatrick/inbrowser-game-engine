// Anim layer: animation state machine (states, 1D blend trees, crossfades, root motion, marker / loop / end events)
// over THREE.AnimationMixer, analytic two-bone IK and a name-based CCD IK wrapper. Imports core (Events) and three.
export { AnimationController } from './AnimationController.js';
export { TwoBoneIK, CCDIK, makeIKChain } from './IK.js';
// 0.9.0: a ragdoll built from a skeleton on the dynamics world (bodies + joints from the bone chains, bones driven back)
export { Ragdoll, DEFAULT_LIMITS as RAGDOLL_LIMITS, RAGDOLL_SEGMENTS, mapBonesAuto } from './Ragdoll.js';
