// Camera layer: rig base + collision utilities, vehicle chase rig, third- / first-person character rigs, orbit and
// free-fly debug rigs.
export { CameraRig, CameraCollision, treeCanopy } from './CameraRig.js';
export { ChaseRig, ChaseRig as ChaseCamera, defaultChaseModes } from './ChaseRig.js';
export { OrbitRig } from './OrbitRig.js';
export { FreeFlyRig, freeFlyActions } from './FreeFlyRig.js';
export { ThirdPersonRig } from './ThirdPersonRig.js';
export { FirstPersonRig } from './FirstPersonRig.js';
import { clamp, smooth01, damp, wrapPi, springStep } from './CameraRig.js';
/** Small math helpers used by the rigs (namespaced to avoid clashing with other layers). */
export const CameraMath = Object.freeze({ clamp, smooth01, damp, wrapPi, springStep });
