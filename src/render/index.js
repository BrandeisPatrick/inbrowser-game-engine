// Render layer: lighting / sky / post-processing, texture sets, impostors, material patches and the
// allocation-free instancing / culling / effects utilities. No game knowledge; nothing here imports src/.
export { Atmosphere, TIME_PRESETS, DEFAULT_SUN, DEFAULT_LOOKS, DEFAULT_FLARE, sunPosition } from './Atmosphere.js';
export { PostFX, DEFAULT_GRADE, gradeAt, resolveAA, AA_MODES } from './PostFX.js';
// G-buffer path (0.7.0): TAA over the MRT scene pass, camera jitter / history matrices shared with later passes
export { GBuffer, GBUFFER_UNIFORMS, GBUFFER_DEFINES, GBUFFER_VERTEX_TAIL, GBUFFER_CHUNK_TAILS, GBUFFER_GLSL_DECODE, installGBufferChunks, createGBufferTarget, updateGBufferView } from './GBuffer.js';
export { TAAPass, taaFilterWeights, TAA_CLAMPS, TAA_FILTERS, TAA_HISTORY_FORMATS } from './passes/TAAPass.js';
export { CameraHistory, jitterPattern } from './CameraHistory.js';
export { AOPass, AO_NORMALS } from './passes/AOPass.js';
// screen-space / volumetric / lens passes (0.8.0), composed by PostFX ({ ssr, volumetric, motionBlur, dof, lut })
export { SSRPass, SSR_DEFAULTS, SSR_SOURCES, SSR_COMPOSITE_GLSL } from './passes/SSRPass.js';
export { VolumetricPass, VOLUMETRIC_DEFAULTS } from './passes/VolumetricPass.js';
export { MotionBlurPass, MOTION_BLUR_DEFAULTS } from './passes/MotionBlurPass.js';
export { DoFPass, DOF_DEFAULTS, circleOfConfusion, vogelDisc } from './passes/DoFPass.js';
export { LUT_GLSL, LUT_DEFAULT_SIZE, identityLut, lutFromFunction, lutLookup, disposeLut } from './Lut.js';
// the opaque scene (colour + depth) copied mid-frame before a mesh draws, for refractive materials (world Water)
export { SceneCopy } from './SceneCopy.js';
export { BloomPass } from './passes/BloomPass.js';
export { ExposurePass } from './passes/ExposurePass.js';
export { installCascadeShadowChunk, cascadeShadowChunkText, installedCascades, MAX_CASCADES } from './shadowCascade.js';
// directional shadow kernels: 'pcf' (three's), 'fast' (5 hardware-compare taps), 'pcss' (blocker search + penumbra disc);
// Atmosphere.setShadowKernel wires them for the sun cascades (2-4, shadowCascade.js)
export {
  SHADOW_KERNELS, FAST_SHADOW_DEFINE, PCSS_SHADOW_DEFINE, PCSS_SHADOW_GLSL, installFastShadowChunk, installPcssShadowChunk,
  createShadowDepthTexture, attachShadowDepth, setFastShadowMaps, setPcssShadowParams, setMaterialShadowKernel, applyShadowKernel,
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
// clustered forward lights (0.7.0): hundreds of small point / spot lights through a per-material define
export {
  ClusteredLights, installClusteredChunks, applyClustered, syncClustered, setMaterialClustered, clusteredUniforms,
  CLUSTERED_DEFINE, MAX_PER_CELL,
} from './ClusteredLights.js';
// GPU-simulated particles (0.7.0): the CPU ParticleSystem's API over GPUComputationRenderer ping-pong textures
export { GpuParticleSystem, GpuParticleStaging } from './GpuParticles.js';
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
