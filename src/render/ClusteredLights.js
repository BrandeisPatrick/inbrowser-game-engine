// Clustered forward lights: hundreds of small point / spot lights (street lamps, windows, headlights) for the lit
// built-in materials, at a per-fragment cost that depends on how many lights reach the fragment's cluster instead of
// on how many lights the scene has. three's forward loop pays every real light on every lit fragment (Apple / ANGLE:
// steeply past ~8, see NearestLights.js), so real lights stay few (and keep their shadows) and the rest go here.
//
// CPU, per frame (update(camera), allocation-free, deterministic): the view frustum is cut into gridX x gridY screen
// tiles x gridZ exponential depth slices (slice = floor(log(z / zNear) / log(zFar / zNear) * gridZ), fragments nearer
// than zNear in slice 0, beyond zFar in the last). Each light's bounding sphere (a spot's: the smallest sphere around
// its cone) goes to the slices it spans and, per slice, to the screen tiles of a conservative bound: the exact
// perspective tangent bound of the sphere intersected with the projected box of the sphere-slab slice, 1 px and
// 1e-3 slice of slack for GPU rounding and TAA jitter. A counting sort turns the (cell, light) pairs into one index
// list per cell, lowest light index first, capped at MAX_PER_CELL per cell (the GLSL loop bound) and maxIndices in
// total (whole cells are cut from the far end; `stats.overflow` counts the dropped pairs).
//
// GPU: like the cascade patch (shadowCascade.js) and the fast kernel (shadowKernel.js), a global ShaderChunk patch
// that is inert unless a material defines ENGINE_CLUSTERED (applyClustered / syncClustered): lights_pars_begin gets
// the samplers, uniforms, EngineClLight and engineClLightInfo (three's getPointLightInfo / getSpotLightInfo math on
// the same getDistanceAttenuation / getSpotAttenuation, so a clustered light matches a THREE.PointLight with the same
// distance / decay / colour * intensity), and lights_fragment_begin gets a loop inserted before the spot-light block
// (the directional block, which the cascade patch rewrites, is not touched) that calls RE_Direct per light of the
// fragment's cluster. No shadows. A define is the only program-cache-key switch (chunk text is not in three's key):
// without it a material compiles exactly the stock code. The five uniforms are added to every lit ShaderLib template,
// ShaderLib.shadow included (harmless there: ShadowMaterial never gets the define, so they stay unbound).
//
// Textures (shared by every material; one ClusteredLights instance is bound to them at a time, the last one
// constructed or updated):
//   engineClLights  RGBA32F, min(maxLights, 1024) x 3 rows per band of 1024 lights (one band up to 1024 lights);
//                   light i at x = i & 1023, y = (i >> 10) * 3 + row:
//                   row 0  view-space position xyz, w = range (> 0 point, < 0 spot)          rewritten every frame
//                   row 1  linear colour * intensity rgb, w = decay                          on setLight / setSpot
//                   row 2  spot only: view-space direction towards the light * k, w = -cosOuter * k with
//                          k = 1 / (cosInner - cosOuter), so k = length(xyz), cosOuter = -w / k and
//                          cosInner = cosOuter + 1 / k (exact, one fetch)                     every frame
//   engineClCells   RGBA32UI, 1024 texels (4096 uint32) per row; cells and index lists packed in ONE texture (+2
//                   texture units, not +3). uint32 element e at texel e >> 2, component e & 3:
//                   [0, cells)          cell c = (slice * gridY + tileY) * gridX + tileX:
//                                       word = start | count << 24 (start = element of its first light index)
//                   [base, base + maxIndices)   index lists (base = cells rounded up to 4); each list starts on a
//                                       texel boundary, so the loop fetches one uvec4 per 4 lights
//   engineClParams  vec4 (gridX, gridY, gridZ, sliceScale)       } Float32Array(4) values (shared by reference:
//   engineClParams2 vec4 (sliceBias, zNear, zFar, lightCount)    } three clones Vector4 uniform values per material)
//   engineClTiles   vec4 (tile width px, tile height px, 1 / width, 1 / height)  from setSize()
//
// Upload per frame, through texture.updateRanges (one texSubImage2D per range, each inside one texture row): every
// value the shader can read this frame and nothing else: lights rows 0 and 1 for [0, count) (32 B / light), row 2
// over the span of spot indices; cells the used prefix [0, used) (cell words + every index list). Each request is
// self-sufficient: it never relies on what the GPU storage held before, which may be fresh (three allocates with
// texStorage2D and, on a first upload with ranges, uploads only the ranges), from an older frame (a request three
// never uploaded is replaced by the next one), or re-allocated (dispose, context restore, a second renderer).
// Measured: 320 street lamps at 3024x1964, default grid: ~86 KB / frame (10 KB lights, 14 KB cell words, the rest
// index lists); frames where no light reaches the view upload nothing (the loop is off: lightCount 0).
// Per-frame allocation: none here; three's clearUpdateRanges() empties the updateRanges array it refills.
// GPU per lit fragment: 1 uvec4 fetch (cell word); per listed light 1 RGBA32F fetch + a sphere test (+ 1 uvec4 per
// 4 lights); per light that reaches the fragment +1 fetch (+1 for a spot) and RE_Direct.
//
// Tile mapping: gl_FragCoord of the scene render target. setSize(width, height) must get that target's size in
// pixels: the drawing-buffer size (PostFX renders the scene at floor(css * pixelRatio), pixel ratio 1 inside the
// composer). Viewport offsets and other cameras / targets (reflections, impostor bakes) are not supported: set
// `uniforms.engineClParams2.value[3] = 0` around such a pass to switch the loop off. Likewise MeshPhysicalMaterial's
// transmission pass: three renders the opaque scene into a target of renderer.transmissionResolutionScale x the
// viewport, so below 1 the tile mapping is off in that pass only (clustered light in the refracted image).
//
//   installClusteredChunks();
//   const cl = new ClusteredLights({ maxLights: 1024 });
//   cl.setSize(drawingBufferWidth, drawingBufferHeight);
//   cl.setLight(i, x, y, z, range, r, g, b, decay);       // world space, colour * intensity
//   syncClustered(scene, true);                           // ENGINE_CLUSTERED on lit materials, also ones added later
//   cl.update(camera);                                    // per frame, before rendering
import * as THREE from 'three';

export const CLUSTERED_DEFINE = 'ENGINE_CLUSTERED';
/** Lights read per cluster (the GLSL loop bound); the CPU caps every cell's list at this length. */
export const MAX_PER_CELL = 64;

const BAND = 1024;          // lights per band of 3 rows in the lights texture
const ROWS = 3;
const CELL_ROW = 1024;      // texels per row of the cells texture
const ROW_ELEMS = CELL_ROW * 4;
const ADDR_BITS = 24;       // cell word: element address in the low 24 bits, count above
const MARGIN_PX = 1;        // screen-space slack of the tile bounds (GPU rounding, TAA jitter up to +-0.5 px)
const SLICE_PAD = 1e-3;     // depth-slice slack (in slices)
const MIN_PENUMBRA = 1e-4;  // cosInner - cosOuter floor (three's smoothstep with equal edges is undefined)
const ACTIVE = 1, SPOT = 2, BLACK = 4;

// cloneUniforms (run for each built-in material when its program is built) clones Texture and Vector values: the
// textures return themselves so every material keeps the shared ones (vec4s are Float32Arrays, kept by reference).
class SharedDataTexture extends THREE.DataTexture {
  clone() { return this; }
}

let SH = null; // shared textures, uniforms and the bound instance (created on first use)
function shared() {
  if (SH) return SH;
  const empty = (data, format, type) => {
    const t = new SharedDataTexture(data, 1, 1, format, type);
    t.minFilter = t.magFilter = THREE.NearestFilter; // float32 / integer textures: anything else is incomplete
    t.generateMipmaps = false;
    t.needsUpdate = true; // always valid (sampler type checks) even before an instance is bound
    return t;
  };
  const lights = empty(new Float32Array(4), THREE.RGBAFormat, THREE.FloatType);
  const cells = empty(new Uint32Array(4), THREE.RGBAIntegerFormat, THREE.UnsignedIntType);
  lights.name = 'engineClLights'; cells.name = 'engineClCells';
  const params = new Float32Array(4), params2 = new Float32Array(4), tiles = new Float32Array(4);
  SH = {
    lights, cells, params, params2, tiles,
    emptyLights: lights.image, emptyCells: cells.image,
    uniforms: {
      engineClLights: { value: lights },
      engineClCells: { value: cells },
      engineClParams: { value: params },
      engineClParams2: { value: params2 },
      engineClTiles: { value: tiles },
    },
    textures: Object.freeze({ lights, cells }),
    bound: null,
  };
  return SH;
}

/** The shared uniform objects ({ engineClLights, engineClCells, engineClParams, engineClParams2, engineClTiles }). */
export function clusteredUniforms() { return shared().uniforms; }

const RE_DIRECT_CALL = 'RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );';
const SPOT_ANCHOR = '#if ( NUM_SPOT_LIGHTS > 0 ) && defined( RE_Direct )';

// Both blocks are whole lines (three's build strips the chunks' trailing newlines): with the define absent the
// preprocessed chunks equal three's text.
const PARS = /* glsl */`
#if defined( ${CLUSTERED_DEFINE} )
	uniform highp sampler2D engineClLights;
	uniform highp usampler2D engineClCells;
	uniform vec4 engineClParams;
	uniform vec4 engineClParams2;
	uniform vec4 engineClTiles;
	struct EngineClLight {
		vec3 position;
		vec3 color;
		float distance;
		float decay;
		vec3 direction;
		float coneCos;
		float penumbraCos;
		bool spot;
	};
	highp ivec2 engineClLightTexel( const in highp int li, const in int row ) {
		return ivec2( li & ${BAND - 1}, ( li >> 10 ) * ${ROWS} + row );
	}
	highp ivec2 engineClCellTexel( const in highp int texel ) {
		return ivec2( texel & ${CELL_ROW - 1}, texel >> 10 );
	}
	highp int engineClCellIndex( const in vec3 viewPos ) {
		ivec3 g = ivec3( engineClParams.xyz );
		ivec2 t = clamp( ivec2( gl_FragCoord.xy * engineClTiles.zw * engineClParams.xy ), ivec2( 0 ), g.xy - 1 );
		int s = clamp( int( floor( log( max( - viewPos.z, engineClParams2.y ) ) * engineClParams.w + engineClParams2.x ) ), 0, g.z - 1 );
		return ( s * g.y + t.y ) * g.x + t.x;
	}
	EngineClLight engineClFetch( const in highp int li, const in vec4 row0 ) {
		EngineClLight l;
		vec4 row1 = texelFetch( engineClLights, engineClLightTexel( li, 1 ), 0 );
		l.position = row0.xyz;
		l.distance = abs( row0.w );
		l.spot = row0.w < 0.0;
		l.color = row1.rgb;
		l.decay = row1.a;
		l.direction = vec3( 0.0, 0.0, 1.0 );
		l.coneCos = - 1.0;
		l.penumbraCos = 1.0;
		if ( l.spot ) {
			vec4 row2 = texelFetch( engineClLights, engineClLightTexel( li, 2 ), 0 );
			float k = length( row2.xyz );
			l.direction = row2.xyz / k;
			l.coneCos = - row2.w / k;
			l.penumbraCos = l.coneCos + 1.0 / k;
		}
		return l;
	}
	void engineClLightInfo( const in EngineClLight l, const in vec3 geometryPosition, out IncidentLight light ) {
		vec3 lVector = l.position - geometryPosition;
		light.direction = normalize( lVector );
		float lightDistance = length( lVector );
		if ( l.spot ) {
			float spotAttenuation = getSpotAttenuation( l.coneCos, l.penumbraCos, dot( light.direction, l.direction ) );
			light.color = l.color * spotAttenuation;
		} else {
			light.color = l.color;
		}
		light.color *= getDistanceAttenuation( lightDistance, l.distance, l.decay );
		light.visible = ( light.color != vec3( 0.0 ) );
	}
#endif`;

const LOOP = /* glsl */`#if defined( ${CLUSTERED_DEFINE} ) && defined( RE_Direct )
	if ( engineClParams2.w > 0.0 ) {
		highp int engineClCell = engineClCellIndex( geometryPosition );
		highp uvec4 engineClQuad = texelFetch( engineClCells, engineClCellTexel( engineClCell >> 2 ), 0 );
		highp uint engineClWord = engineClQuad[ engineClCell & 3 ];
		int engineClCount = int( engineClWord >> ${ADDR_BITS}u );
		highp int engineClStart = int( engineClWord & ${(1 << ADDR_BITS) - 1}u );
		for ( int engineClK = 0; engineClK < ${MAX_PER_CELL}; engineClK ++ ) {
			if ( engineClK >= engineClCount ) break;
			if ( ( engineClK & 3 ) == 0 ) engineClQuad = texelFetch( engineClCells, engineClCellTexel( ( engineClStart + engineClK ) >> 2 ), 0 );
			highp int engineClLi = int( engineClQuad[ engineClK & 3 ] );
			vec4 engineClRow0 = texelFetch( engineClLights, engineClLightTexel( engineClLi, 0 ), 0 );
			vec3 engineClV = engineClRow0.xyz - geometryPosition;
			if ( dot( engineClV, engineClV ) >= engineClRow0.w * engineClRow0.w ) continue; // attenuation is 0 there
			engineClLightInfo( engineClFetch( engineClLi, engineClRow0 ), geometryPosition, directLight );
			if ( directLight.visible ) {
				${RE_DIRECT_CALL}
			}
		}
	}
#endif
`;

let installed = null;

/**
 * Install the chunk + template patch (idempotent). Inert for every material without ENGINE_CLUSTERED. false (with a
 * warning) when three's light chunks changed.
 */
export function installClusteredChunks() {
  if (installed !== null) return installed;
  const C = THREE.ShaderChunk;
  const need = [
    [C.common, 'struct IncidentLight'],
    [C.lights_pars_begin, 'float getDistanceAttenuation( const in float lightDistance, const in float cutoffDistance, const in float decayExponent )'],
    [C.lights_pars_begin, 'float getSpotAttenuation( const in float coneCosine, const in float penumbraCosine, const in float angleCosine )'],
    [C.lights_fragment_begin, 'vec3 geometryPosition = - vViewPosition;'],
    [C.lights_fragment_begin, 'IncidentLight directLight;'],
    [C.lights_fragment_begin, SPOT_ANCHOR],
    [C.lights_fragment_begin, RE_DIRECT_CALL],
  ];
  const f = C.lights_fragment_begin;
  if (!need.every(([chunk, s]) => typeof chunk === 'string' && chunk.includes(s))
    || f.indexOf(SPOT_ANCHOR) !== f.lastIndexOf(SPOT_ANCHOR) || f.indexOf(SPOT_ANCHOR) < f.indexOf('IncidentLight directLight;')) {
    console.warn('[render] clustered lights: three.js light chunks changed, clustered lights disabled');
    installed = false;
    return false;
  }
  C.lights_pars_begin += PARS;
  C.lights_fragment_begin = f.replace(SPOT_ANCHOR, LOOP + SPOT_ANCHOR);
  const U = shared().uniforms;
  for (const k in THREE.ShaderLib) {
    const u = THREE.ShaderLib[k].uniforms;
    if (u && u.pointLights && !u.engineClLights) Object.assign(u, U);
  }
  installed = true;
  return true;
}

// ---------------------------------------------------------------------------------------------------------------
// CPU side

// exact x/depth (or y/depth) range of a sphere seen from the origin: tangents of the circle (a, d, r) in the
// axis-depth plane; +-Infinity where the tangent wedge reaches depth <= 0
const _tb = new Float64Array(2);
function tangentBounds(a, d, r) {
  const L2 = a * a + d * d, r2 = r * r;
  if (L2 <= r2) { _tb[0] = -Infinity; _tb[1] = Infinity; return; }
  const t = Math.sqrt(L2 - r2);
  const dl = d * t + a * r, dh = d * t - a * r;
  _tb[0] = dl > 0 ? (a * t - d * r) / dl : -Infinity;
  _tb[1] = dh > 0 ? (a * t + d * r) / dh : Infinity;
}

function makeRanges(n) {
  const a = new Array(n);
  for (let i = 0; i < n; i++) a[i] = { start: 0, count: 0 };
  return a;
}

export class ClusteredLights {
  /**
   * @param {object} [o]
   * @param {number} [o.maxLights=1024]
   * @param {number[]} [o.grid=[16, 9, 24]]  screen tiles x, y and depth slices
   * @param {number} [o.zNear=0.5] @param {number} [o.zFar=400]  slice range (m); lights wholly beyond zFar are dropped
   * @param {number} [o.maxIndices=65536]  index-list capacity (light-cell pairs incl. the per-cell padding to 4)
   * @param {boolean} [o.timing=false]  measure update() into stats.ms
   */
  constructor({ maxLights = 1024, grid = [16, 9, 24], zNear = 0.5, zFar = 400, maxIndices = 65536, timing = false } = {}) {
    const gx = grid[0] | 0, gy = grid[1] | 0, gz = grid[2] | 0;
    if (!(gx > 0 && gy > 0 && gz > 0 && gx <= 4096 && gy <= 4096 && gz <= 4096)) throw new RangeError('ClusteredLights: bad grid');
    if (!(maxLights >= 1 && maxLights <= 1 << 20)) throw new RangeError('ClusteredLights: maxLights must be 1 .. 2^20');
    if (!(zNear > 0 && zFar > zNear)) throw new RangeError('ClusteredLights: need 0 < zNear < zFar');
    this.maxLights = maxLights | 0;
    this.grid = Object.freeze([gx, gy, gz]);
    this.zNear = zNear; this.zFar = zFar;
    this.timing = !!timing;
    this.enabled = true;
    const nCells = gx * gy * gz;
    this.cellCount = nCells;
    this.cellBase = (nCells + 3) & ~3;                     // first element of the index lists
    this.maxIndices = Math.max(4, (Math.ceil(maxIndices) + 3) & ~3);
    if (this.cellBase + this.maxIndices >= 2 ** ADDR_BITS) throw new RangeError('ClusteredLights: cells + maxIndices must stay below 2^24');
    this.sliceScale = gz / Math.log(zFar / zNear);
    this.sliceBias = -Math.log(zNear) * this.sliceScale;

    // world-space light data (CPU only)
    const N = this.maxLights;
    this._wp = new Float64Array(N * 4);    // position xyz, range
    this._wb = new Float64Array(N * 2);    // bounding sphere: centre offset along a spot's axis (0: point), radius
    this._wd = new Float64Array(N * 4);    // spots: direction towards the light xyz, k = 1 / (cosInner - cosOuter)
    this._wc = new Float64Array(N);        // spots: cosOuter
    this._flags = new Uint8Array(N);
    this._count = 0;
    this._spotN = 0; this._spotLo = N; this._spotHi = 0; // spot indices lie in [spotLo, spotHi) (row 2 uploads)

    // texture images
    this.lightsWidth = Math.min(N, BAND);
    this._bands = Math.ceil(N / BAND);
    this.lightData = new Float32Array(this.lightsWidth * 4 * ROWS * this._bands);
    this._limg = { data: this.lightData, width: this.lightsWidth, height: ROWS * this._bands };
    this.cellsHeight = Math.ceil((this.cellBase + this.maxIndices) / ROW_ELEMS);
    this.cellData = new Uint32Array(ROW_ELEMS * this.cellsHeight);
    this._cimg = { data: this.cellData, width: CELL_ROW, height: this.cellsHeight };

    // per-frame scratch
    this._cnt = new Uint32Array(nCells);
    this._start = new Uint32Array(nCells);
    this._n = new Uint8Array(nCells);
    this._fill = new Uint8Array(nCells);
    this._vis = new Uint32Array(N);
    this._vs0 = new Uint16Array(N); this._vs1 = new Uint16Array(N);
    this._rect = new Uint16Array(N * gz * 4);             // per visible light and slice: tx0, tx1, ty0, ty1
    this._rp = new Uint32Array(N);
    // padded depth range of each slice (slice 0 reaches the camera, the last one infinity)
    const zp = Math.exp(SLICE_PAD / this.sliceScale);
    this._slabLo = new Float64Array(gz); this._slabHi = new Float64Array(gz);
    for (let s = 0; s < gz; s++) {
      this._slabLo[s] = s === 0 ? -Infinity : this.sliceDepth(s) / zp;
      this._slabHi[s] = s === gz - 1 ? Infinity : this.sliceDepth(s + 1) * zp;
    }

    // upload bookkeeping
    this._lr = makeRanges(ROWS * this._bands);
    this._cr = makeRanges(this.cellsHeight);

    this.width = 1920; this.height = 1080; this._mx = MARGIN_PX / 1920; this._my = MARGIN_PX / 1080;
    this.stats = { lights: 0, cells: 0, indices: 0, overflow: 0, ms: 0, maxPerCell: 0, bytes: 0 };
    this._disposed = false;
    this._bind();
  }

  /** The shared uniform objects (also in every lit THREE.ShaderLib template after installClusteredChunks()). */
  get uniforms() { return shared().uniforms; }
  /** { lights, cells }: the shared DataTextures. */
  get textures() { return shared().textures; }

  /** Highest light index used + 1. Setting it lower clears the lights at and above it. */
  get count() { return this._count; }
  set count(n) {
    n = Math.max(0, Math.min(this.maxLights, n | 0));
    for (let i = n; i < this._count; i++) this._setFlags(i, 0);
    this._count = n;
  }

  /** Drawing-buffer size of the scene render target (pixels): the tile mapping of gl_FragCoord. */
  setSize(width, height) {
    this.width = Math.max(1, width | 0); this.height = Math.max(1, height | 0);
    this._mx = MARGIN_PX / this.width; this._my = MARGIN_PX / this.height;
    if (shared().bound === this) this._writeTiles();
  }

  /**
   * Point light i. World position (m), range (> 0: three's `distance` cutoff; 0 / infinite ranges cannot be
   * clustered and leave the light off), linear colour * intensity (as three's PointLight uniform), decay.
   */
  setLight(i, x, y, z, range, r, g, b, decay = 2) {
    this._check(i);
    const o = i * 4, wp = this._wp;
    wp[o] = x; wp[o + 1] = y; wp[o + 2] = z; wp[o + 3] = range;
    this._wb[i * 2] = 0; this._wb[i * 2 + 1] = range;
    this._row1(i, r, g, b, decay);
    this._setFlags(i, range > 0 && range < Infinity ? ACTIVE | (r === 0 && g === 0 && b === 0 ? BLACK : 0) : 0);
  }

  /**
   * Spot light i: as setLight plus the direction it points in (world, any length) and the cosines of three's
   * penumbraCos (cosInner = cos(angle * (1 - penumbra))) and coneCos (cosOuter = cos(angle)). cosInner is floored at
   * cosOuter + 1e-4, so a penumbra = 0 spot gets a 1e-4-wide smoothstep edge where three's smoothstep with equal
   * edges is undefined (driver-dependent, a hard step in practice): its rim can differ from a real SpotLight's.
   */
  setSpot(i, x, y, z, range, r, g, b, decay, dx, dy, dz, cosInner, cosOuter) {
    this._check(i);
    const len = Math.hypot(dx, dy, dz);
    if (!(len > 0)) { this.setLight(i, x, y, z, range, r, g, b, decay); return; }
    dx /= len; dy /= len; dz /= len;
    const co = Math.max(-1, Math.min(1, cosOuter));
    const ci = Math.max(cosInner, co + MIN_PENUMBRA);
    const o = i * 4, wp = this._wp, wd = this._wd;
    wp[o] = x; wp[o + 1] = y; wp[o + 2] = z; wp[o + 3] = range;
    wd[o] = -dx; wd[o + 1] = -dy; wd[o + 2] = -dz; wd[o + 3] = 1 / (ci - co);
    this._wc[i] = co;
    // smallest sphere around the lit cone (apex, rim at `range`, spherical cap): centre apex + axis * c. Kept as
    // (c, radius) and placed in view space each frame, where the shader measures `range` (a scaled camera parent)
    let c = 0, rad = range;
    if (co >= Math.SQRT1_2) { c = range / (2 * co); rad = c; } else if (co > 0) { c = range * co; rad = range * Math.sqrt(1 - co * co); }
    this._wb[i * 2] = c; this._wb[i * 2 + 1] = rad;
    this._row1(i, r, g, b, decay);
    this._setFlags(i, range > 0 && range < Infinity ? ACTIVE | SPOT | (r === 0 && g === 0 && b === 0 ? BLACK : 0) : 0);
  }

  clearLight(i) {
    this._check(i);
    this._setFlags(i, 0);
    if (i === this._count - 1) { let n = i; while (n > 0 && !this._flags[n - 1]) n--; this._count = n; }
  }

  /** Depth-slice of a view depth (-z, m): floor(log(z / zNear) / log(zFar / zNear) * gridZ), clamped. */
  sliceOf(depth) {
    if (!(depth > 0)) return 0;
    const s = Math.floor(Math.log(depth) * this.sliceScale + this.sliceBias);
    return s < 0 ? 0 : s >= this.grid[2] ? this.grid[2] - 1 : s;
  }

  /** Depth (m) where slice s begins (s = 0 .. gridZ). */
  sliceDepth(s) { return this.zNear * Math.pow(this.zFar / this.zNear, s / this.grid[2]); }

  /** The cell the shader picks for a fragment at gl_FragCoord (fx, fy) and view depth (JS copy of engineClCellIndex). */
  cellOf(fx, fy, depth) {
    const [gx, gy, gz] = this.grid;
    const tx = Math.min(gx - 1, Math.max(0, Math.floor(fx / this.width * gx)));
    const ty = Math.min(gy - 1, Math.max(0, Math.floor(fy / this.height * gy)));
    const s = Math.min(gz - 1, Math.max(0, Math.floor(Math.log(Math.max(depth, this.zNear)) * this.sliceScale + this.sliceBias)));
    return (s * gy + ty) * gx + tx;
  }

  /** Per frame: cluster the lights for `camera` and queue the texture uploads. Allocation-free. */
  update(camera) {
    const t0 = this.timing ? performance.now() : 0;
    const st = this.stats;
    if (this._disposed) return;
    this._bind();
    const S = SH;
    if (!this.enabled || !camera) {
      S.params2[3] = 0;
      st.lights = st.cells = st.indices = st.overflow = st.maxPerCell = st.bytes = 0;
      st.ms = this.timing ? performance.now() - t0 : 0;
      return;
    }
    camera.updateWorldMatrix(true, false);
    const V = camera.matrixWorldInverse.elements, P = camera.projectionMatrix.elements;
    const persp = P[11] !== 0;
    const gx = this.grid[0], gy = this.grid[1], gz = this.grid[2];
    const scale = this.sliceScale, bias = this.sliceBias, zFar = this.zFar;
    const mx = this._mx, my = this._my;
    const flags = this._flags, wp = this._wp, wb = this._wb, wd = this._wd, wc = this._wc;
    const L = this.lightData, LW = this.lightsWidth;
    const cnt = this._cnt, rect = this._rect, vis = this._vis, vs0 = this._vs0, vs1 = this._vs1, rp = this._rp;
    const slabLo = this._slabLo, slabHi = this._slabHi;
    cnt.fill(0);
    let nVis = 0, rpos = 0;
    const n = this._count;

    // pass 1: view-space light data, cluster bounds, counts per cell
    for (let i = 0; i < n; i++) {
      const f = flags[i];
      if (!(f & ACTIVE)) continue;
      const o = i * 4;
      const x = wp[o], y = wp[o + 1], z = wp[o + 2], range = wp[o + 3];
      const t0o = (((i >> 10) * ROWS) * LW + (i & (BAND - 1))) * 4;
      let cx = V[0] * x + V[4] * y + V[8] * z + V[12];
      let cy = V[1] * x + V[5] * y + V[9] * z + V[13];
      let cz = V[2] * x + V[6] * y + V[10] * z + V[14];
      L[t0o] = cx; L[t0o + 1] = cy; L[t0o + 2] = cz;
      L[t0o + 3] = f & SPOT ? -range : range;
      if (f & SPOT) {
        const k = wd[o + 3], ax = wd[o], ay = wd[o + 1], az = wd[o + 2];
        // towards the light in view space, renormalised like three's transformDirection (V may carry a scale)
        let ux = V[0] * ax + V[4] * ay + V[8] * az, uy = V[1] * ax + V[5] * ay + V[9] * az, uz = V[2] * ax + V[6] * ay + V[10] * az;
        const il = 1 / Math.sqrt(ux * ux + uy * uy + uz * uz);
        ux *= il; uy *= il; uz *= il;
        const t2o = t0o + 2 * LW * 4;
        L[t2o] = ux * k; L[t2o + 1] = uy * k; L[t2o + 2] = uz * k;
        L[t2o + 3] = -wc[i] * k;
        const c = wb[i * 2]; // bounding-sphere centre: apex + axis * c (the axis points away from the light)
        cx -= ux * c; cy -= uy * c; cz -= uz * c;
      }
      if (f & BLACK) continue; // adds nothing: kept out of the clusters
      const cr = wb[i * 2 + 1], depth = -cz;
      const zlo = depth - cr, zhi = depth + cr;
      if ((persp && zhi <= 0) || zlo > zFar) continue; // behind the camera / beyond the clustered range
      // whole-sphere screen bound (fractions of the target, 0 .. 1)
      let gu0, gu1, gv0, gv1;
      if (persp) {
        tangentBounds(cx, depth, cr); gu0 = _tb[0]; gu1 = _tb[1];
        tangentBounds(cy, depth, cr); gv0 = _tb[0]; gv1 = _tb[1];
      } else { gu0 = cx - cr; gu1 = cx + cr; gv0 = cy - cr; gv1 = cy + cr; }
      if (this._tiles(persp, P, gu0, gu1, gv0, gv1, mx, my, gx, gy) === 0) continue; // off screen
      const s0 = zlo > 0 ? Math.min(gz - 1, Math.max(0, Math.floor(Math.log(zlo) * scale + bias - SLICE_PAD))) : 0;
      const s1 = zhi > 0 ? Math.min(gz - 1, Math.max(0, Math.floor(Math.log(zhi) * scale + bias + SLICE_PAD))) : 0;
      vis[nVis] = i; vs0[nVis] = s0; vs1[nVis] = s1; rp[nVis] = rpos; nVis++;
      const r2 = cr * cr;
      for (let s = s0; s <= s1; s++) {
        // the sphere inside this slice's (padded) depth range: a box of half-width rs over depths a .. b
        const a = Math.max(slabLo[s], zlo), b = Math.min(slabHi[s], zhi);
        let tx0 = 1, tx1 = 0, ty0 = 1, ty1 = 0;
        if (a <= b) {
          const dz = depth < a ? a - depth : depth > b ? depth - b : 0;
          const rs = dz > 0 ? Math.sqrt(Math.max(0, r2 - dz * dz)) : cr;
          let u0 = cx - rs, u1 = cx + rs, v0 = cy - rs, v1 = cy + rs;
          if (persp) {
            if (a > 0) {
              u0 = u0 <= 0 ? u0 / a : u0 / b; u1 = u1 >= 0 ? u1 / a : u1 / b;
              v0 = v0 <= 0 ? v0 / a : v0 / b; v1 = v1 >= 0 ? v1 / a : v1 / b;
              if (gu0 > u0) u0 = gu0; if (gu1 < u1) u1 = gu1;
              if (gv0 > v0) v0 = gv0; if (gv1 < v1) v1 = gv1;
            } else { u0 = gu0; u1 = gu1; v0 = gv0; v1 = gv1; }
          }
          if (this._tiles(persp, P, u0, u1, v0, v1, mx, my, gx, gy) !== 0) {
            tx0 = _tr[0]; tx1 = _tr[1]; ty0 = _tr[2]; ty1 = _tr[3];
            for (let ty = ty0; ty <= ty1; ty++) {
              const row = (s * gy + ty) * gx;
              for (let tx = tx0; tx <= tx1; tx++) cnt[row + tx]++;
            }
          } else { tx0 = 1; tx1 = 0; }
        }
        rect[rpos] = tx0; rect[rpos + 1] = tx1; rect[rpos + 2] = ty0; rect[rpos + 3] = ty1; rpos += 4;
      }
    }

    if (nVis === 0) {
      // nothing reaches the view (daytime): the shader skips the loop on lightCount 0 and reads neither texture, so
      // nothing is uploaded (a request still queued from an earlier frame is self-sufficient and may stay)
      S.params2[3] = 0;
      st.lights = st.cells = st.indices = st.overflow = st.maxPerCell = st.bytes = 0;
      st.ms = this.timing ? performance.now() - t0 : 0;
      return;
    }

    // prefix sums -> cell words; each list padded to whole texels, capped per cell and in total
    const C = this.cellData, start = this._start, cn = this._n, fill = this._fill;
    const nCells = this.cellCount, limit = this.cellBase + this.maxIndices;
    let run = this.cellBase, overflow = 0, maxPer = 0, nonEmpty = 0, stored = 0;
    for (let c = 0; c < nCells; c++) {
      const k = cnt[c];
      fill[c] = 0;
      if (k === 0) { cn[c] = 0; C[c] = 0; continue; }
      if (k > maxPer) maxPer = k;
      let m = k < MAX_PER_CELL ? k : MAX_PER_CELL;
      let need = (m + 3) & ~3;
      if (run + need > limit) { need = limit - run; m = need; }
      overflow += k - m;
      cn[c] = m; start[c] = run;
      C[c] = m > 0 ? (run | (m << ADDR_BITS)) >>> 0 : 0;
      run += need; stored += m;
      if (m > 0) nonEmpty++;
    }
    // pass 2: light indices, ascending per cell
    for (let v = 0; v < nVis; v++) {
      const i = vis[v];
      let q = rp[v];
      for (let s = vs0[v], s1 = vs1[v]; s <= s1; s++, q += 4) {
        const tx0 = rect[q], tx1 = rect[q + 1], ty0 = rect[q + 2], ty1 = rect[q + 3];
        for (let ty = ty0; ty <= ty1; ty++) {
          const row = (s * gy + ty) * gx;
          for (let tx = tx0; tx <= tx1; tx++) {
            const c = row + tx;
            if (fill[c] < cn[c]) C[start[c] + fill[c]++] = i;
          }
        }
      }
    }

    S.params[0] = gx; S.params[1] = gy; S.params[2] = gz; S.params[3] = scale;
    S.params2[0] = bias; S.params2[1] = this.zNear; S.params2[2] = zFar; S.params2[3] = nVis;
    st.bytes = this._uploadLights(n) + this._uploadCells(run);
    st.lights = nVis; st.cells = nonEmpty; st.indices = stored; st.overflow = overflow; st.maxPerCell = maxPer;
    st.ms = this.timing ? performance.now() - t0 : 0;
  }

  /** Release the shared textures' GPU storage if this instance is bound; the shader loop is switched off. */
  dispose() {
    if (this._disposed) return;
    this._disposed = true;
    const S = shared();
    if (S.bound !== this) return;
    S.bound = null;
    swapImage(S.lights, S.emptyLights); swapImage(S.cells, S.emptyCells);
    S.params2[3] = 0;
  }

  // -------------------------------------------------------------------------------------------------------------

  _check(i) {
    if (!(i >= 0 && i < this.maxLights) || (i | 0) !== i) throw new RangeError(`ClusteredLights: light index ${i} outside 0 .. ${this.maxLights - 1}`);
  }

  _setFlags(i, f) {
    const was = this._flags[i];
    if ((was & SPOT) && !(f & SPOT)) { if (--this._spotN === 0) { this._spotLo = this.maxLights; this._spotHi = 0; } }
    if ((f & SPOT) && !(was & SPOT)) this._spotN++;
    if (f & SPOT) { if (i < this._spotLo) this._spotLo = i; if (i + 1 > this._spotHi) this._spotHi = i + 1; }
    this._flags[i] = f;
    if (f && i >= this._count) this._count = i + 1;
  }

  _row1(i, r, g, b, decay) {
    const o = ((((i >> 10) * ROWS) + 1) * this.lightsWidth + (i & (BAND - 1))) * 4, L = this.lightData;
    L[o] = r; L[o + 1] = g; L[o + 2] = b; L[o + 3] = decay;
  }

  // screen bound in view units (x/depth for perspective, x for orthographic) -> tile range in _tr; 0 when off screen
  _tiles(persp, P, u0, u1, v0, v1, mx, my, gx, gy) {
    let a, b, c, d;
    if (persp) { a = P[0] * u0 - P[8]; b = P[0] * u1 - P[8]; c = P[5] * v0 - P[9]; d = P[5] * v1 - P[9]; }
    else { a = P[0] * u0 + P[12]; b = P[0] * u1 + P[12]; c = P[5] * v0 + P[13]; d = P[5] * v1 + P[13]; }
    if (a > b) { const t = a; a = b; b = t; }
    if (c > d) { const t = c; c = d; d = t; }
    let tx0 = Math.floor((a * 0.5 + 0.5 - mx) * gx), tx1 = Math.floor((b * 0.5 + 0.5 + mx) * gx);
    let ty0 = Math.floor((c * 0.5 + 0.5 - my) * gy), ty1 = Math.floor((d * 0.5 + 0.5 + my) * gy);
    if (!(tx1 >= 0 && tx0 <= gx - 1 && ty1 >= 0 && ty0 <= gy - 1)) return 0; // also NaN
    if (!(tx0 >= 0)) tx0 = 0; if (tx1 > gx - 1) tx1 = gx - 1;
    if (!(ty0 >= 0)) ty0 = 0; if (ty1 > gy - 1) ty1 = gy - 1;
    _tr[0] = tx0; _tr[1] = tx1; _tr[2] = ty0; _tr[3] = ty1;
    return 1;
  }

  _bind() {
    const S = shared();
    if (S.bound === this) return;
    S.bound = this;
    swapImage(S.lights, this._limg); swapImage(S.cells, this._cimg);
    S.params[0] = this.grid[0]; S.params[1] = this.grid[1]; S.params[2] = this.grid[2]; S.params[3] = this.sliceScale;
    S.params2[0] = this.sliceBias; S.params2[1] = this.zNear; S.params2[2] = this.zFar; S.params2[3] = 0;
    this._writeTiles();
  }

  _writeTiles() {
    const t = SH.tiles;
    t[0] = this.width / this.grid[0]; t[1] = this.height / this.grid[1]; t[2] = 1 / this.width; t[3] = 1 / this.height;
  }

  // queue rows 0 and 1 of [0, n) and row 2 of the spot span: every light value the shader can read (listed lights
  // are < n, row 2 is read for spots only); returns the bytes requested
  _uploadLights(n) {
    const tex = SH.lights, R = this._lr;
    tex.clearUpdateRanges();
    let k = 0, bytes = 0;
    for (let band = 0; band < this._bands; band++) {
      const lo = band * BAND, hi = Math.min(lo + BAND, n);
      if (hi <= lo) break;
      k = this._range(tex, R, k, band, 0, lo, hi);
      k = this._range(tex, R, k, band, 1, lo, hi);
      if (this._spotN > 0) k = this._range(tex, R, k, band, 2, Math.max(lo, this._spotLo), Math.min(hi, this._spotHi));
    }
    if (k === 0) return 0; // (needsUpdate without ranges would upload everything)
    for (let j = 0; j < k; j++) bytes += R[j].count * 4;
    tex.needsUpdate = true;
    return bytes;
  }

  _range(tex, R, k, band, row, i0, i1) {
    if (i1 <= i0) return k;
    const r = R[k];
    r.start = ((band * ROWS + row) * this.lightsWidth + (i0 - band * BAND)) * 4;
    r.count = (i1 - i0) * 4;
    tex.updateRanges.push(r);
    return k + 1;
  }

  // queue the used prefix [0, used) of the cells texture (every cell word and index list), one range per row
  _uploadCells(used) {
    const tex = SH.cells, R = this._cr;
    tex.clearUpdateRanges();
    const rows = Math.ceil(used / ROW_ELEMS);
    for (let j = 0; j < rows; j++) {
      const r = R[j];
      r.start = j * ROW_ELEMS;
      r.count = Math.min(ROW_ELEMS, used - r.start);
      tex.updateRanges.push(r);
    }
    tex.needsUpdate = true;
    return used * 4;
  }
}

const _tr = new Int32Array(4);

// point a shared texture at another image; a different size needs new GPU storage (three sizes it at the first upload)
function swapImage(tex, img) {
  const old = tex.image;
  if (!old || old.width !== img.width || old.height !== img.height) tex.dispose();
  tex.image = img;
  tex.clearUpdateRanges();
  tex.needsUpdate = true;
}

// ---------------------------------------------------------------------------------------------------------------
// material selection

/**
 * Lit materials whose program runs lights_fragment_begin with the shared uniforms: built-ins (patched templates) and
 * lights: true ShaderMaterials (given the shared uniform objects when switched on).
 */
function takesClustered(m) {
  return !!(m.isMeshStandardMaterial || m.isMeshLambertMaterial || m.isMeshPhongMaterial || m.isMeshToonMaterial
    || (m.isShaderMaterial && m.lights === true));
}

/**
 * Switch one material through its ENGINE_CLUSTERED define (needsUpdate when it changes; three keeps both programs).
 * Turning on installs the chunks first. Returns true when the material changed.
 */
export function setMaterialClustered(m, on) {
  if (!m || !takesClustered(m)) return false;
  if (on && installed === null) installClusteredChunks();
  const want = !!on && installed === true;
  if (want === !!(m.defines && CLUSTERED_DEFINE in m.defines)) return false;
  if (want) {
    if (m.isShaderMaterial) {
      const U = shared().uniforms, u = m.uniforms || (m.uniforms = {});
      for (const k in U) if (!u[k]) u[k] = U[k];
    }
    (m.defines || (m.defines = {}))[CLUSTERED_DEFINE] = '';
  } else delete m.defines[CLUSTERED_DEFINE]; // an emptied defines object keys and compiles like none
  m.needsUpdate = true;
  return true;
}

const _apply = { on: false, n: 0 };
function applyTo(o) {
  const m = o.material;
  if (!m) return;
  if (Array.isArray(m)) { for (let i = 0; i < m.length; i++) if (setMaterialClustered(m[i], _apply.on)) _apply.n++; }
  else if (setMaterialClustered(m, _apply.on)) _apply.n++;
}

/** setMaterialClustered on every material under root. Returns the count changed. */
export function applyClustered(root, on) {
  const prevOn = _apply.on, prevN = _apply.n; // re-entrant (a 'childadded' during a traversal)
  _apply.on = !!on; _apply.n = 0;
  if (root) root.traverse(applyTo);
  const n = _apply.n;
  _apply.on = prevOn; _apply.n = prevN;
  return n;
}

function onChildAdded(e) {
  const c = e.child;
  if (!c) return;
  applyClustered(c, true);
  c.traverse(listen);
}
// a subtree leaving a synced tree stops being synced (keeps its define, loses the listeners): nothing outside the
// tree keeps re-applying the define, and no registry holds removed (streamed-out) objects alive
function onChildRemoved(e) { if (e.child) e.child.traverse(unlisten); }
function listen(o) {
  if (!o.hasEventListener('childadded', onChildAdded)) o.addEventListener('childadded', onChildAdded);
  if (!o.hasEventListener('childremoved', onChildRemoved)) o.addEventListener('childremoved', onChildRemoved);
}
function unlisten(o) {
  o.removeEventListener('childadded', onChildAdded);
  o.removeEventListener('childremoved', onChildRemoved);
}

/**
 * applyClustered(scene, on), and while on, keep it on for objects added anywhere under scene later ('childadded' /
 * 'childremoved' listeners on every object of the tree: subtrees added are switched on and listened, subtrees
 * removed are unlistened, so off later covers everything that is still in the tree and nothing removed reacts).
 * Material swaps on existing meshes are not seen: call applyClustered for those. Returns the count changed now.
 */
export function syncClustered(scene, on) {
  const n = applyClustered(scene, on);
  if (scene) scene.traverse(on ? listen : unlisten);
  return n;
}
