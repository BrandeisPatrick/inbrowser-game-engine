// Render layer: lighting / sky / post-processing, texture sets, impostors, material patches and the
// allocation-free instancing / culling / effects utilities. No game knowledge; nothing here imports src/.
export { Atmosphere, TIME_PRESETS, DEFAULT_SUN, DEFAULT_LOOKS, DEFAULT_FLARE, sunPosition } from './Atmosphere.js';
export { PostFX, DEFAULT_GRADE, gradeAt } from './PostFX.js';
export { AOPass } from './passes/AOPass.js';
export { BloomPass } from './passes/BloomPass.js';
export { ExposurePass } from './passes/ExposurePass.js';
export { installCascadeShadowChunk } from './shadowCascade.js';
// opt-in 'fast' directional shadow kernel (5 hardware-compare taps; Atmosphere.setShadowKernel wires it for the sun)
export {
  SHADOW_KERNELS, FAST_SHADOW_DEFINE, installFastShadowChunk, createShadowDepthTexture, attachShadowDepth, setFastShadowMaps,
  setMaterialShadowKernel, applyShadowKernel,
} from './shadowKernel.js';
export { makeCloudTexture, sampleCloud, makeFlareTextures } from './skyTextures.js';
export {
  textureBase, basisPath, loadTextureManifest, loadTextureSet, makeTextureArray, proceduralPbr, TextureSets, ktx2Support,
  makeAsphaltTexture, makePavingTexture, makeConcreteTexture, makeGroundTexture, makeGrassTexture, makeWaterNormalTexture,
  makeCanvas, paintCanvas, finishTexture, makeDataArrayTexture, freeAfterUpload,
} from './TextureSets.js';
export { buildMipChain, dataTextureFromMips, fillTransparent, dilateRGB } from './MipChain.js';
// asset loaders for the core Assets registry (glTF + KTX2 / Draco / meshopt, textures, HDR, .cube LUTs)
export { registerRenderLoaders, parseCubeLUT, disposeInstance } from './loaders.js';
// debug lines for a DynamicsWorld / DynamicsClient (shapes, contacts, joints)
export { PhysicsDebug } from './PhysicsDebug.js';
export { DEFAULT_RINGS, impostorViews, bakeImpostors, createImpostorMaterial, ImpostorBatch } from './Impostors.js';
export { createTimeUniforms, updateTimeUniforms, windSway, ditherFade, applyPatches, makeDepthMaterial, GLSL_IGN, GLSL_HASH } from './MaterialPatches.js';
export { InstancedPool } from './InstancedPool.js';
export { LODGroup } from './LODGroup.js';
export { ChunkCuller } from './Culling.js';
export { DecalTrail } from './Decals.js';
export { ParticleSystem, ParticleEmitter, softDotTexture, atmosphereSunFactor } from './Particles.js';
// procedural noise: seeded simplex 2D / 3D, RNG, hash, tileable value noise; the whole module (incl. the
// clamp / lerp / smoothstep easing helpers) is also available namespaced as Noise
export { Simplex, simplex2, simplex3, mulberry32, hash2, pnoise, pfbm } from './noise.js';
export * as Noise from './noise.js';
