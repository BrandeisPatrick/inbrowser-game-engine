// Input layer: action maps (keyboard / gamepad / pointer / virtual), touch controls, vehicle preset.
export { InputMap, approach as inputApproach, ramp as inputRamp, normalizeBinding } from './InputMap.js';
export { GamepadState, PAD_BUTTONS, PAD_AXES, applyDeadzone } from './Gamepad.js';
export { TouchControls, isTouchDevice } from './TouchControls.js';
export { VehicleInput, VehicleInput as Input, VEHICLE_ACTIONS, VEHICLE_TOUCH_LAYOUT } from './VehicleInput.js';
