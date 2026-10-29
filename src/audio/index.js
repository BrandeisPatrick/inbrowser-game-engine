// Audio layer: context/mixer bus, synth voices, positional emitters, ambience, vehicle synth.
export { AudioBus, DEFAULT_BUSES } from './AudioBus.js';
export { noiseBuffer, loopSource, makeFilter, makeGain, makeShaper, softClipCurve, chain as audioChain, setParam, NoiseVoice, OscStack, LFO, Envelope, playTone, playNoiseBurst } from './Synth.js';
export { PositionalAudio, AudioListener3D, Emitter as AudioEmitter } from './Positional.js';
export { CityAmbience } from './Ambience.js';
export { VehicleAudio } from './VehicleAudio.js';
export { registerAudioLoader, SamplePlayer } from './Samples.js';
