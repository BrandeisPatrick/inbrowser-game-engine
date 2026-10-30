// Render asset types for core/Assets: glTF models, textures (incl. KTX2), Radiance HDR environments, .cube 3D LUTs.
//
//   const handle = registerRenderLoaders(assets, engine.renderer, { dracoPath: '/draco/' });
//   assets.add('car', { type: 'gltf', url: 'models/car.glb', weight: 4 })
//         .add('grass', { type: 'texture', url: 'tex/grass.jpg', wrap: 'repeat' })
//         .add('sky', { type: 'hdr', url: 'env/sky.hdr', pmrem: true })
//         .add('grade', { type: 'lut', url: 'luts/warm.cube' });
//   await assets.load();
//   const car = assets.get('car').instantiate();  // skinning-safe clone; geometry / materials / textures are shared
//   scene.add(car); ... scene.remove(car); disposeInstance(car);   // frees the clone's own skeletons (bone textures)
//   scene.environment = assets.get('sky');
//   const { texture, size, domainMin, domainMax } = assets.get('grade');
//   handle.dispose();                             // frees the DRACO workers (values are freed by assets.release)
//
// Types (descriptor options after `type` / `url` / `weight`):
//   gltf     -> { scene, animations, cameras, asset, userData, gltf, instantiate() } (gltf = three's result with
//            `parser` nulled, so the GLB binary and the parser caches are not kept alive). The file is fetched with
//            assets.fetch and parsed by one shared GLTFLoader (relative buffers / images resolve against the file's
//            folder). Decoders are attached
//            only when the file's JSON names the extension, so nothing extra downloads otherwise:
//            KHR_texture_basisu -> the shared KTX2Loader of render/TextureSets.js (ktx2Support(renderer): transcoder
//            from basisPath()), when `ktx2` and a renderer; EXT_meshopt_compression -> three's MeshoptDecoder
//            (imported on first use), when `meshopt`; KHR_draco_mesh_compression -> DRACOLoader, only with `dracoPath`.
//            instantiate() = SkeletonUtils.clone(scene) (bones rebound, so skinned clones animate independently); a
//            clone owns only its Skeletons (bone textures): disposeInstance(clone) frees them, never anything shared.
//            dispose: what the file created, snapshotted at load: geometries, materials, the parser's textures
//            (ImageBitmaps closed; never `envMap` or textures assigned later) and skeletons of every scene. Clones
//            share geometries / materials / textures: release a model only once its clones are gone.
//   texture  { srgb = true, wrap = 'repeat' | 'clamp' | 'mirror', flipY = true, anisotropy = min(8, GPU max) }
//            -> THREE.Texture. Images through TextureLoader; `.ktx2` through the shared KTX2Loader (needs a renderer
//            with a compressed format and the transcoder; rejects otherwise). KTX2 keeps its file's colour space
//            unless `srgb` is given, and ignores flipY (compressed data cannot be flipped on upload).
//   hdr      { pmrem = false } -> equirect HDR DataTexture (EquirectangularReflectionMapping, half float); with
//            `pmrem` and a renderer the PMREM environment texture instead (scene.environment / material.envMap).
//   lut      { float = false } -> { size, texture: Data3DTexture RGBA (half float, or float with `float: true`;
//            linear filtering, clamp), title, domainMin, domainMax } from an Adobe .cube 3D LUT (parseCubeLUT).
//            Texel (r, g, b) at x = r, y = g, z = b (red fastest, as in the file); half floats filter linearly on
//            every WebGL2 GPU, float needs OES_texture_float_linear.
// registerRenderLoaders returns { gltfLoader, dracoLoader, dispose() }. disposeInstance(root) disposes the skeletons
// of the skinned meshes under root (an instantiate() clone). Importing this module has no side effects
// (Node-safe); WebGL is only touched when an asset loads.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
import { RGBELoader } from 'three/addons/loaders/RGBELoader.js';
import { clone as cloneSkinned } from 'three/addons/utils/SkeletonUtils.js';
import { ktx2Support } from './TextureSets.js';

const WRAP = { repeat: THREE.RepeatWrapping, clamp: THREE.ClampToEdgeWrapping, mirror: THREE.MirroredRepeatWrapping };
// three r180 renamed RGBELoader to HDRLoader and made RGBELoader a subclass that warns on construction: use its base
// class there, RGBELoader itself on older three (where it extends DataTextureLoader directly)
const HDRBase = Object.getPrototypeOf(RGBELoader);
const HDRLoaderClass = HDRBase === THREE.DataTextureLoader ? RGBELoader : HDRBase;
let _meshopt = null;
const meshoptDecoder = () => _meshopt || (_meshopt = import('three/addons/libs/meshopt_decoder.module.js').then((m) => m.MeshoptDecoder));

/** The glTF JSON as text: the JSON chunk of a .glb, the whole file of a .gltf. */
function gltfJson(buf) {
  const u = new Uint8Array(buf);
  const glb = u.length >= 20 && u[0] === 0x67 && u[1] === 0x6c && u[2] === 0x54 && u[3] === 0x46; // 'glTF'
  return new TextDecoder().decode(glb ? u.subarray(20, 20 + new DataView(buf).getUint32(12, true)) : u);
}

/**
 * What a parsed glTF owns, snapshotted before anything else can be attached: geometries, materials, skeletons of every
 * scene, and the material textures the parser made (its `associations`, plus clones sharing their source: texCoord /
 * KHR_texture_transform variants). `envMap` is never taken.
 */
function ownedBy(gltf) {
  const sources = new Set(), own = { geometries: new Set(), materials: new Set(), textures: new Set(), skeletons: new Set() };
  for (const k of gltf.parser.associations.keys()) if (k && k.isTexture) sources.add(k.source);
  for (const root of gltf.scenes && gltf.scenes.length ? gltf.scenes : [gltf.scene]) root.traverse((o) => {
    if (o.geometry) own.geometries.add(o.geometry);
    if (o.isSkinnedMesh && o.skeleton) own.skeletons.add(o.skeleton);
    for (const m of Array.isArray(o.material) ? o.material : o.material ? [o.material] : []) {
      if (own.materials.has(m)) continue;
      own.materials.add(m);
      for (const k of Object.keys(m)) { const t = m[k]; if (k !== 'envMap' && t && t.isTexture && sources.has(t.source)) own.textures.add(t); }
    }
  });
  return own;
}
function disposeOwned(own) {
  for (const g of own.geometries) g.dispose();
  for (const s of own.skeletons) s.dispose();
  for (const t of own.textures) { if (t.image && typeof t.image.close === 'function') t.image.close(); t.dispose(); } // close: ImageBitmap
  for (const m of own.materials) m.dispose();
}

/**
 * Free what an instantiate() clone owns by itself: the Skeletons of its skinned meshes (bone textures). Geometries,
 * materials and textures are shared with the asset and are left alone (assets.release frees them).
 * @param {THREE.Object3D} root
 */
export function disposeInstance(root) {
  root.traverse((o) => { if (o.isSkinnedMesh && o.skeleton) o.skeleton.dispose(); });
}

/**
 * Register the gltf / texture / hdr / lut types on `assets` (see header).
 * @param {import('../core/Assets.js').Assets} assets
 * @param {THREE.WebGLRenderer|null} renderer  needed for KTX2, PMREM and the default anisotropy
 * @param {object} [o]
 * @param {string|null} [o.dracoPath=null]  DRACO decoder folder (draco_decoder.js / .wasm); null = no DRACO
 * @param {boolean} [o.ktx2=true]  KTX2 textures (glTF KHR_texture_basisu and `.ktx2` texture assets)
 * @param {boolean} [o.meshopt=true]  EXT_meshopt_compression
 * @returns {{gltfLoader: GLTFLoader, dracoLoader: DRACOLoader|null, dispose(): void}}
 */
export function registerRenderLoaders(assets, renderer, { dracoPath = null, ktx2 = true, meshopt = true } = {}) {
  const gltfLoader = new GLTFLoader();
  const dracoLoader = dracoPath ? new DRACOLoader().setDecoderPath(dracoPath) : null;
  if (dracoLoader) gltfLoader.setDRACOLoader(dracoLoader);
  const owned = new WeakMap(); // gltf value -> ownedBy() snapshot
  const kxFor = () => (ktx2 && renderer ? ktx2Support(renderer) : null);
  const anisotropy = (d) => d.anisotropy ?? (renderer ? Math.min(8, renderer.capabilities.getMaxAnisotropy()) : 1);

  assets.register('gltf', async (d, a) => {
    const buf = await (await a.fetchOk(d.url, d.init)).arrayBuffer();
    const json = gltfJson(buf);
    if (json.includes('"KHR_texture_basisu"') && !gltfLoader.ktx2Loader) {
      const kx = kxFor();
      // no KTX2 (option off, no renderer / compressed format, transcoder missing): GLTFLoader uses the extension's
      // fallback image when the file has one and rejects ("setKTX2Loader must be called ...") when it is required
      if (kx && (await kx.ready)) gltfLoader.setKTX2Loader(kx.loader);
    }
    if (meshopt && json.includes('"EXT_meshopt_compression"')) gltfLoader.setMeshoptDecoder(await meshoptDecoder());
    if (!dracoLoader && json.includes('"KHR_draco_mesh_compression"')) throw new Error('uses KHR_draco_mesh_compression: pass { dracoPath } to registerRenderLoaders');
    const gltf = await gltfLoader.parseAsync(buf, THREE.LoaderUtils.extractUrlBase(d.url));
    const own = ownedBy(gltf);
    gltf.parser = null; // holds the GLB body and the parser caches
    const scene = gltf.scene || gltf.scenes[0];
    const v = { scene, animations: gltf.animations || [], cameras: gltf.cameras || [], asset: gltf.asset, userData: gltf.userData, gltf, instantiate: () => cloneSkinned(scene) };
    owned.set(v, own);
    return v;
  }, { dispose: (v) => { const own = owned.get(v); if (own) disposeOwned(own); owned.delete(v); } });

  assets.register('texture', async (d, a) => {
    let tex;
    if (/\.ktx2(?:$|[?#])/i.test(d.url)) {
      const kx = kxFor();
      const bufP = a.fetchOk(d.url, d.init).then((r) => r.arrayBuffer()); // download alongside the transcoder
      if (!kx || !(await kx.ready)) {
        bufP.catch(() => {});
        throw new Error(`KTX2 unavailable (${!ktx2 ? 'ktx2: false' : !renderer ? 'no renderer' : !kx ? 'no compressed texture format' : 'transcoder missing'})`);
      }
      const buf = await bufP;
      tex = await new Promise((res, rej) => kx.loader.parse(buf, res, rej));
      if (d.srgb !== undefined) tex.colorSpace = d.srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
    } else {
      tex = await new THREE.TextureLoader().loadAsync(d.url);
      tex.colorSpace = d.srgb === false ? THREE.NoColorSpace : THREE.SRGBColorSpace;
      tex.flipY = d.flipY ?? true;
    }
    tex.wrapS = tex.wrapT = WRAP[d.wrap] ?? THREE.RepeatWrapping;
    tex.anisotropy = anisotropy(d);
    tex.needsUpdate = true;
    return tex;
  }, { dispose: (t) => t.dispose() });

  let hdrLoader = null;
  const targets = new WeakMap(); // PMREM texture -> its render target (disposed together)
  assets.register('hdr', async (d) => {
    const tex = await (hdrLoader || (hdrLoader = new HDRLoaderClass())).loadAsync(d.url);
    tex.mapping = THREE.EquirectangularReflectionMapping;
    if (!d.pmrem || !renderer) return tex;
    const gen = new THREE.PMREMGenerator(renderer);
    const rt = gen.fromEquirectangular(tex);
    gen.dispose(); tex.dispose();
    targets.set(rt.texture, rt);
    return rt.texture;
  }, { dispose: (t) => (targets.get(t) || t).dispose() });

  assets.register('lut', async (d, a) => parseCubeLUT(await (await a.fetchOk(d.url, d.init)).text(), { float: !!d.float }),
    { dispose: (v) => v.texture.dispose() });

  return { gltfLoader, dracoLoader, dispose() { if (dracoLoader) dracoLoader.dispose(); } };
}

/**
 * Parse an Adobe .cube 3D LUT (TITLE, LUT_3D_SIZE 2..256, DOMAIN_MIN / DOMAIN_MAX, LUT_3D_INPUT_RANGE, '#' comments,
 * then size^3 "r g b" rows, red fastest). Throws with the line number on malformed input, 1D LUTs, a missing or
 * invalid LUT_3D_SIZE, or a row count other than size^3.
 * @param {string} text
 * @param {{float?: boolean}} [o]  float: Float32 texels (default half float)
 * @returns {{size: number, texture: THREE.Data3DTexture, title: string, domainMin: number[], domainMax: number[]}}
 */
export function parseCubeLUT(text, { float = false } = {}) {
  let size = 0, n = 0, data = null, title = '';
  let min = [0, 0, 0], max = [1, 1, 1];
  const lines = String(text).split(/\r\n|\n|\r/);
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line || line[0] === '#') continue;
    const fail = (m) => { throw new Error(`LUT .cube line ${i + 1}: ${m}`); };
    const nums = (k) => { const v = line.split(/\s+/).slice(1).map(Number); if (v.length !== k || !v.every(Number.isFinite)) fail(`expected ${k} numbers`); return v; };
    if (/^[-+.\d]/.test(line)) { // data row
      if (!data) fail('data before LUT_3D_SIZE');
      if (n >= size * size * size) fail(`more than ${size}^3 rows`);
      const p = line.split(/\s+/);
      if (p.length !== 3) fail(`expected "r g b", got "${line}"`);
      for (let k = 0; k < 3; k++) { const v = +p[k]; if (!Number.isFinite(v)) fail(`bad number "${p[k]}"`); data[n * 4 + k] = v; }
      data[n * 4 + 3] = 1; n++;
      continue;
    }
    const kw = line.split(/\s/, 1)[0];
    if (kw === 'TITLE') title = line.slice(5).trim().replace(/^"(.*)"$/, '$1');
    else if (kw === 'LUT_3D_SIZE') {
      if (data) fail('LUT_3D_SIZE given twice');
      const [s] = nums(1);
      if (!Number.isInteger(s) || s < 2 || s > 256) fail(`LUT_3D_SIZE must be an integer in 2..256, got ${s}`);
      size = s; data = new Float32Array(s * s * s * 4);
    } else if (kw === 'LUT_1D_SIZE') fail('1D LUTs are not supported');
    else if (kw === 'DOMAIN_MIN') min = nums(3);
    else if (kw === 'DOMAIN_MAX') max = nums(3);
    else if (kw === 'LUT_3D_INPUT_RANGE') { const [a, b] = nums(2); min = [a, a, a]; max = [b, b, b]; }
    // other keywords (vendor extensions) are ignored
  }
  if (!size) throw new Error('LUT .cube: missing LUT_3D_SIZE');
  if (n !== size * size * size) throw new Error(`LUT .cube: ${n} rows, expected ${size * size * size} (LUT_3D_SIZE ${size})`);
  if (!(min[0] < max[0] && min[1] < max[1] && min[2] < max[2])) throw new Error('LUT .cube: DOMAIN_MIN must be below DOMAIN_MAX');
  let texels = data;
  if (!float) { texels = new Uint16Array(data.length); for (let i = 0; i < data.length; i++) texels[i] = THREE.DataUtils.toHalfFloat(data[i]); }
  const texture = new THREE.Data3DTexture(texels, size, size, size);
  texture.format = THREE.RGBAFormat;
  texture.type = float ? THREE.FloatType : THREE.HalfFloatType;
  texture.minFilter = texture.magFilter = THREE.LinearFilter;
  texture.wrapS = texture.wrapT = texture.wrapR = THREE.ClampToEdgeWrapping;
  texture.generateMipmaps = false;
  texture.unpackAlignment = 1;
  texture.needsUpdate = true;
  return { size, texture, title, domainMin: min, domainMax: max };
}
