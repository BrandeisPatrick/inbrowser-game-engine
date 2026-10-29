// Anim layer: animation state machine (states, 1D blend trees, crossfades, root motion, marker / loop / end events)
// over THREE.AnimationMixer, analytic two-bone IK and a name-based CCD IK wrapper. Imports core (Events) and three.
export { AnimationController } from './AnimationController.js';
export { TwoBoneIK, CCDIK, makeIKChain } from './IK.js';
