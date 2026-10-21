// Moving traffic, main-thread side: the presenter for the simulation core (TrafficSim.js), which runs in a Web Worker
// (traffic.worker.js, default in browsers with module workers; `traffic: { worker: false }` or ?noworker=1 = in-thread,
// as do Node / headless builds) or on the main thread.
//
// Worker mode: the main thread builds the same TrafficSim (lane graph + baked ground heights + initial spawn; also the
// fallback and what dev tools read), hands the worker the road data + baked heights, and from then on posts ONE small
// message per frame (dt, camera position + view-projection matrix, player footprint, the 169 signal phases the bulbs
// show, the pose buffer it is done with). The worker steps the AI and answers with a transferable Float32Array of
// poses (POSE_STRIDE floats per vehicle: x, y, z, yaw, pitch, brake, speed, yaw rate, flags), double-buffered.
// Respawn / recycling decisions stay in the worker. The presenter draws the latest snapshot extrapolated to the
// current frame (speed + yaw rate; the snapshot is one frame old), and queryColliders answers from that snapshot
// (the same oriented AABB slices as before). If the worker fails to start, the in-thread sim takes over (logged once).
//
// Rendering: one InstancedMesh per vehicle type with a small shader patch (Vehicles.js: paint mask, clearcoat,
// head/tail/brake lamps scaled by darkness). Instances are re-packed every frame (_present): near-cascade casters
// first, then the rest of the view, then far-cascade-only casters, so the main pass and each shadow cascade draw only
// a prefix (engine ShadowPasses.shadowCount) instead of all vehicles in every pass. Headlight beams at night.
import * as THREE from 'three';
import { makeVehicleGeometry, VEHICLE_SPECS, createVehicleMaterial } from './Vehicles.js';
import { shadowCount, shadowFrustum } from './ShadowPasses.js';
import { TrafficSim, TRAFFIC_TYPES, POSE_STRIDE, buildSignalInfos, playerInfoFrom, vehicleSlices } from './TrafficSim.js';

const TYPES = TRAFFIC_TYPES;
const MAX_SLICES = 10;
const READY_TIMEOUT = 5000; // ms for the worker to build its sim before the in-thread sim takes over
const _m = new THREE.Matrix4(), _q = new THREE.Quaternion(), _p = new THREE.Vector3(), _e = new THREE.Euler(0, 0, 0, 'YXZ');
const _one = new THREE.Vector3(1, 1, 1), _sph = new THREE.Sphere(), _pm = new THREE.Matrix4();
const _bm = new THREE.Matrix4(), _bm2 = new THREE.Matrix4(), _bp = new THREE.Vector3(), _bq = new THREE.Quaternion(), _bs = new THREE.Vector3();
const clamp01h = (h) => Math.min(1.05, Math.max(0.62, h));
let _warned = false;
const _sx = new THREE.Vector3(), _sy = new THREE.Vector3(), _sz = new THREE.Vector3();

/** Bounding sphere of a directional light's orthographic shadow box (same placement as ShadowPasses.shadowFrustum). */
function shadowBoxSphere(light, out) {
  const sc = light.shadow.camera;
  _sx.setFromMatrixPosition(light.matrixWorld);
  _sz.setFromMatrixPosition(light.target ? light.target.matrixWorld : light.matrixWorld).sub(_sx);
  const L = _sz.length();
  if (L > 1e-6) _sz.multiplyScalar(1 / L); else _sz.set(0, -1, 0);
  const w = sc.right - sc.left, h = sc.top - sc.bottom, d = sc.far - sc.near;
  out.center.copy(_sx).addScaledVector(_sz, (sc.near + sc.far) / 2);
  if (sc.left + sc.right !== 0 || sc.top + sc.bottom !== 0) out.radius = Infinity; // off-centre box: no reject
  else out.radius = 0.5 * Math.sqrt(w * w + h * h + d * d);
  return out;
}
const nearSphere = (a, x, y, z, r) => { const c = a.center, R = a.radius + r, dx = c.x - x, dy = c.y - y, dz = c.z - z; return dx * dx + dy * dy + dz * dz <= R * R; };
function loadPlanes(f, out, o) {
  for (let k = 0; k < 6; k++) { const p = f.planes[k], n = p.normal, j = o + k * 4; out[j] = n.x; out[j + 1] = n.y; out[j + 2] = n.z; out[j + 3] = p.constant; }
}
/** THREE.Frustum.intersectsSphere on planes flattened by loadPlanes. */
function inPlanes(F, o, x, y, z, r) {
  for (let j = o; j < o + 24; j += 4) if (F[j] * x + F[j + 1] * y + F[j + 2] * z + F[j + 3] < -r) return false;
  return true;
}

/** Default for `traffic.worker`: browsers with Worker support, unless ?noworker=1. */
export function trafficWorkerDefault() {
  if (typeof Worker === 'undefined' || typeof window === 'undefined') return false;
  try { const q = new URLSearchParams(window.location.search); if (q.has('noworker') && q.get('noworker') !== '0') return false; } catch { /* no location */ }
  return true;
}

/**
 * Traffic paint / lamp material tweaks (vehicles.js builds the base): brighter brake lamps, police light bars
 * flashing red / blue alternately, and glass that reflects the sky environment more strongly.
 */
function patchTrafficMaterial(m) {
  const tu = { uTimeT: { value: 0 } };
  const prev = m.onBeforeCompile;
  m.onBeforeCompile = (sh, r) => {
    prev(sh, r);
    Object.assign(sh.uniforms, tu);
    sh.vertexShader = sh.vertexShader
      .replace('uniform float uDark;', 'uniform float uDark; uniform float uTimeT; varying float vGlassK;')
      .replace('vec3( 1.0, 0.0, 0.0 ) * ( 0.03 + uDark * 0.45 + brk * ( 0.85 + uDark * 0.3 ) )', 'vec3( 1.0, 0.012, 0.008 ) * ( 0.03 + uDark * 0.5 + brk * ( 1.7 + uDark * 1.9 ) )')
      .replace('vec3( 1.0, 0.04, 0.04 ) * ( uDark * 0.5 )', 'vec3( 1.0, 0.04, 0.03 ) * smoothstep( 0.15, 0.6, sin( uTimeT * 3.6 ) ) * ( 0.8 + uDark * 3.2 )')
      .replace('vec3( 0.08, 0.2, 1.0 ) * ( uDark * 0.5 )', 'vec3( 0.06, 0.2, 1.0 ) * smoothstep( 0.15, 0.6, -sin( uTimeT * 3.6 ) ) * ( 0.8 + uDark * 3.2 )')
      .replace('vSurf = aSurf;', 'vSurf = aSurf;\nvGlassK = ( aSurf.x < 0.045 && aSurf.y < 0.5 && aLamp < 0.5 ) ? 1.0 : 0.0;');
    sh.fragmentShader = sh.fragmentShader
      .replace('varying vec3 vLampE; varying vec3 vSurf;', 'varying vec3 vLampE; varying vec3 vSurf; varying float vGlassK;')
      .replace('#include <aomap_fragment>', '#include <aomap_fragment>\nreflectedLight.indirectSpecular *= 1.0 + 1.6 * vGlassK;');
  };
  m.customProgramCacheKey = () => 'roadster-vehicle-traffic-1';
  m.userData.tu = tu;
  return m;
}

// ------------------------------------------------------------------ Traffic
export class Traffic {
  /**
   * @param {object} o
   * @param {World} o.world  needs getGroundHeight(x,z,y) (height bake) + getSpawn(); optional world.player / world.camera
   * @param {object} o.data  city.json (signals)
   * @param {RoadNetwork} o.net
   * @param {Props} o.props  traffic signal phases (props.signals[].state)
   * @param {number} [o.seed]
   * @param {number} [o.count] vehicles (default world.opts.traffic ?? 250 - a number, or { count }; 0/false disables)
   * @param {THREE.Camera} [o.camera] for out-of-view respawns (also: setCamera, world.camera)
   * @param {{lower?: object, stripe?: object}} [o.busLivery] bus colours (Vehicles.js DEFAULT_BUS_LIVERY)
   * @param {boolean} [o.worker] simulate in a Web Worker (default trafficWorkerDefault())
   */
  constructor({ world, data, net, props, seed = 7, count, camera, busLivery = null, worker } = {}) {
    this.world = world; this.data = data; this.net = net; this.props = props; this.seed = seed;
    this.busLivery = busLivery;
    const wo = world && world.opts ? world.opts.traffic : undefined;
    const opt = count ?? (wo && typeof wo === 'object' ? wo.count : wo);
    this.target = opt === false ? 0 : Number.isFinite(opt) ? Math.max(0, Math.floor(opt)) : 250;
    this.wantWorker = worker ?? trafficWorkerDefault();
    this.count = 0;
    this.camera = camera || null;
    this._player = null; this._plInfo = {};
    this.sim = null;
    this.mode = 'main';          // 'worker' once the worker answered, else the in-thread sim
    this.ready = Promise.resolve();
    this.meshes = [];
    this._cam = new THREE.Vector3(0, 30, 0);
    this._frustum = new THREE.Frustum(); this._hasFrustum = false;
    this.stats = {};
    // measurements (dev pages / profiling): main-thread ms per update (EMA), worker step ms, message bytes
    this.perf = { mainMs: 0, workerMs: 0, postBytes: 0, snapBytes: 0, latency: 0, snapshots: 0, late: 0 };
  }

  /** Optional: () => AABB {minX..maxZ} | THREE.Vector3 | Object3D | {object3d}. Vehicles brake for it. */
  setPlayer(getter) { this._player = getter || null; }
  /** Optional: camera used to keep respawns out of view. */
  setCamera(camera) { this.camera = camera || null; }

  /** The core's lane graph / vehicles (static data in worker mode: use pose(i) for where a vehicle is drawn). */
  get lanes() { return this.sim ? this.sim.lanes : []; }
  get vehicles() { return this.sim ? this.sim.vehicles : []; }
  get typeNames() { return TYPES.map((t) => t.name); }
  /** Signal colour (0 red, 1 yellow, 2 green) of a lane's signal info for phase group g, as the AI sees it. */
  _light(info, g) { return this.sim ? this.sim._light(info, g) : 0; }

  build() {
    if (!this.net || !this.net.list || !this.target) return null;
    const t0 = performance.now();
    const W = this.world;
    const sim = this.sim = new TrafficSim({
      roads: TrafficSim.roadsFrom(this.net),
      signals: buildSignalInfos({ data: this.data, net: this.net, props: this.props }),
      groundAt: W && W.getGroundHeight ? (x, z, y) => W.getGroundHeight(x, z, y) : null,
      spawn: W && W.getSpawn ? W.getSpawn().position : { x: 0, z: 0 },
      seed: this.seed, count: this.target,
    });
    if (!sim.build()) { this.sim = null; return null; }
    this.count = sim.count;
    this.stats = sim.stats;
    const N = sim.vehicles.length;
    this.poses = sim.writePoses(new Float32Array(N * POSE_STRIDE));
    this.pp = new Float32Array(N * 6);     // presented pose per vehicle: x, y, z, yaw, pitch, brake
    this.act = new Uint8Array(N);
    this._buildMeshes();
    this._buildBeams();
    this._present(this.poses, 0);
    if (this.wantWorker) this.ready = this._startWorker();
    this.stats.buildMs = Math.round(performance.now() - t0);
    console.info('[Traffic]', this.stats);
    return this.group;
  }

  // ================================================================ worker
  _startWorker() {
    const sim = this.sim, N = sim.vehicles.length;
    let worker;
    try {
      worker = new Worker(new URL('./traffic.worker.js', import.meta.url), { type: 'module' });
    } catch (e) { this._fallback(e); return Promise.resolve(); }
    this.worker = worker;
    this._msg = { type: 'step', dt: 0, seq: 0, view: new Float32Array(19), hasPv: false, player: new Float32Array(9), hasPlayer: false, sig: sim.sigState, bufs: null };
    this._spares = []; this._postedT = 0; this._snapT = 0; this._seq = 0; this.snap = null;
    this._cvs = sim.vehicles.map((v) => ({
      id: v.id, L: v.L, W: v.W, Hc: v.Hc, WB: v.WB, nb: 0, gen: -1,
      box: { minX: 0, minY: -1e4, maxX: 0, maxY: -1e4, minZ: 0, maxZ: 0 },
      boxes: Array.from({ length: MAX_SLICES }, () => ({ minX: 0, minY: -1e4, maxX: 0, maxY: -1e4, minZ: 0, maxZ: 0, kind: 'vehicle', id: v.id, vx: 0, vz: 0 })),
    }));
    this._snapGen = 0;
    return new Promise((resolve) => {
      let done = false;
      const finish = () => { if (!done) { done = true; clearTimeout(timer); resolve(); } };
      const timer = setTimeout(() => { this._fallback(new Error(`no answer within ${READY_TIMEOUT} ms`)); finish(); }, READY_TIMEOUT);
      worker.onmessage = (ev) => {
        const m = ev.data;
        if (this.worker !== worker) return; // fell back meanwhile
        if (m.type === 'poses') { this._onPoses(m); return; }
        if (m.type === 'ready') {
          const a = new Float32Array(m.buf);
          let dmax = 0; for (let i = 0; i < a.length; i++) dmax = Math.max(dmax, Math.abs(a[i] - this.poses[i]));
          if (dmax > 1e-3) console.warn('[Traffic] worker sim differs from the main-thread build (max pose diff', dmax, ')');
          this.snap = a; this._snapT = m.t; this._postedT = m.t; this._snapGen++;
          this.mode = 'worker';
          this.stats.worker = { buildMs: m.buildMs, lanes: m.lanes, poseDiff: dmax };
          console.info('[Traffic] simulating in a Web Worker', this.stats.worker);
          finish();
        } else if (m.type === 'error') { this._fallback(new Error(m.message)); finish(); }
      };
      worker.onerror = (ev) => { ev.preventDefault && ev.preventDefault(); this._fallback(new Error(ev.message || 'worker error')); finish(); };
      worker.onmessageerror = () => { this._fallback(new Error('message error')); finish(); };
      const bk = sim.baked;
      // the worker gets its own copy of the baked heights (the main-thread sim keeps its graph as the fallback)
      const heights = bk.heights.slice(), index = bk.index.slice();
      worker.postMessage({ type: 'init', roads: sim.roads, signals: sim.sigInfos, heights, index, spawn: { x: sim.spawn.x, z: sim.spawn.z }, seed: sim.seed, count: sim.target, n: N },
        [heights.buffer, index.buffer]);
    });
  }

  _fallback(err) {
    if (this.worker) { try { this.worker.terminate(); } catch { /* gone */ } this.worker = null; }
    const was = this.mode;
    this.mode = 'main';
    if (!_warned) { _warned = true; console.warn(`[Traffic] Web Worker unavailable${was === 'worker' ? ' (failed while running)' : ''}, simulating on the main thread:`, err && err.message ? err.message : err); }
  }

  _onPoses(m) {
    const buf = new Float32Array(m.buf);
    if (this.snap) this._spares.push(this.snap);
    this.snap = buf; this._snapT = m.t; this._snapGen++;
    const P = this.perf;
    P.workerMs = P.workerMs * 0.9 + m.stepMs * 0.1; P.snapBytes = buf.byteLength; P.snapshots++;
    P.latency = this._postedT - m.t; // sim seconds the newest snapshot lags the main thread (one frame when on time)
    this._summary = m.summary;
  }

  _post(dt, pv, info) {
    const m = this._msg, S = this.sim;
    m.dt = dt; m.seq = ++this._seq;
    m.view[0] = this._cam.x; m.view[1] = this._cam.y; m.view[2] = this._cam.z;
    m.hasPv = !!pv; if (pv) m.view.set(pv, 3);
    m.hasPlayer = !!info;
    if (info) { const p = m.player; p[0] = info.x; p[1] = info.y; p[2] = info.z; p[3] = info.fx; p[4] = info.fz; p[5] = info.hl; p[6] = info.hw; p[7] = info.rx; p[8] = info.rz; }
    m.sig = S.sigState;
    const sp = this._spares, tr = [];
    m.bufs = sp.length ? sp.map((b) => b.buffer) : null;
    for (const b of sp) tr.push(b.buffer);
    sp.length = 0;
    this.worker.postMessage(m, tr);
    m.bufs = null; // transferred (detached)
    this._postedT += dt;
    this.perf.postBytes = 19 * 4 + 9 * 4 + m.sig.byteLength + 16;
  }

  // ================================================================ per frame
  update(dt, cameraPos, darkness = 0) {
    const S = this.sim;
    if (!S || !S.vehicles.length) return;
    const t0 = performance.now();
    dt = Number.isFinite(dt) ? Math.min(Math.max(dt, 0), 0.1) : 0;
    if (cameraPos) this._cam.copy(cameraPos);
    const cam = this.camera || (this.world && this.world.camera) || null;
    let pv = null;
    if (cam && cam.projectionMatrix) {
      _pm.multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse);
      this._frustum.setFromProjectionMatrix(_pm); this._hasFrustum = true;
      pv = _pm.elements;
      if (!cameraPos) this._cam.copy(cam.position);
    }
    this.material.userData.u.uDark.value = Math.min(1, Math.max(0, (darkness - 0.12) / 0.45));
    this._lightT = (this._lightT || 0) + dt; this.material.userData.tu.uTimeT.value = this._lightT;
    S.readSignals(this.props && this.props.signals); // the phase the bulbs show this frame
    const src = this._player || (this.world && this.world.player) || null;
    const info = src ? playerInfoFrom(src, this._plInfo) : null;
    if (this.mode === 'worker' && this.worker) {
      this._post(dt, pv, info);
      const late = this._postedT - this._snapT - dt; // > 0: the last snapshot missed its frame
      if (late > 1e-4) this.perf.late++;
      this._present(this.snap, Math.min(0.1, Math.max(0, this._postedT - this._snapT)));
    } else {
      S.setView(this._cam.x, this._cam.y, this._cam.z, pv);
      S.setPlayer(info);
      S.update(dt);
      S.writePoses(this.poses);
      this._present(this.poses, 0);
    }
    this._updateBeams(darkness);
    const ms = performance.now() - t0;
    this.perf.mainMs = this.perf.mainMs * 0.95 + ms * 0.05;
    this.perf.lastMs = ms;
  }

  /** Presented pose of vehicle i -> out { x, y, z, yaw, pitch, brake, active } (where it is drawn this frame). */
  pose(i, out = {}) {
    const q = i * 6, P = this.pp;
    out.x = P[q]; out.y = P[q + 1]; out.z = P[q + 2]; out.yaw = P[q + 3]; out.pitch = P[q + 4]; out.brake = P[q + 5];
    out.active = !!this.act[i];
    return out;
  }

  // ================================================================ meshes
  _buildMeshes() {
    const counts = this.sim.typeCounts;
    this.group = new THREE.Group();
    this.group.name = 'traffic';
    this.material = patchTrafficMaterial(createVehicleMaterial({ brake: true }));
    this._tierIdx = []; this._tierN = new Int32Array(TYPES.length * 3);
    TYPES.forEach((T, ti) => {
      const n = counts[ti];
      this._tierIdx.push(new Int32Array(Math.max(1, n) * 3));
      if (!n) { this.meshes.push(null); return; }
      const geo = makeVehicleGeometry(T.name, T.name === 'bus' && this.busLivery ? { livery: this.busLivery } : undefined);
      const brake = new THREE.InstancedBufferAttribute(new Float32Array(n), 1);
      brake.setUsage(THREE.DynamicDrawUsage);
      geo.setAttribute('aBrake', brake);
      geo.computeBoundingSphere();
      const mesh = new THREE.InstancedMesh(geo, this.material, n);
      mesh.name = `traffic-${T.name}`;
      mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      mesh.frustumCulled = false;
      mesh.castShadow = true; mesh.receiveShadow = true;
      const col = new THREE.Color();
      for (const v of this.sim.vehicles) if (v.type === ti) { col.setRGB(v.rgb[0], v.rgb[1], v.rgb[2]); mesh.setColorAt(v.ii, col); }
      for (let i = 0; i < n; i++) { _m.makeScale(0, 0, 0); mesh.setMatrixAt(i, _m); }
      if (mesh.instanceColor) { mesh.instanceColor.setUsage(THREE.DynamicDrawUsage); mesh.instanceColor.needsUpdate = true; }
      mesh.userData.nNear = n; mesh.userData.nFar = n;
      shadowCount(mesh, (cam) => (cam === this._nearShadowCam ? mesh.userData.nNear : mesh.userData.nFar));
      this.meshes.push(mesh);
      this.group.add(mesh);
    });
  }

  // ================================================================ headlight beams (night, near the camera)
  /** Two faint additive light cones + one road light pool per vehicle, instanced; filled near the camera only. */
  _buildBeams() {
    const MAX = 96; // vehicles with beams (nearest within BEAM_R)
    this._beamMax = MAX;
    const cone = new THREE.CylinderGeometry(0.09, 2.3, 15, 14, 1, true);
    cone.translate(0, -7.5, 0).rotateX(Math.PI / 2 - 0.075); // apex at the origin, opening along -Z, dipped ~4 deg
    const beamMat = new THREE.ShaderMaterial({
      uniforms: { uK: { value: 0 } },
      vertexShader: `varying float vT; varying vec3 vN; varying vec3 vV;
        #include <common>
        #include <fog_pars_vertex>
        void main() { vT = clamp(-position.z / 15.0, 0.0, 1.0);
          vec4 mvPosition = modelViewMatrix * instanceMatrix * vec4(position, 1.0);
          vN = normalize(normalMatrix * mat3(instanceMatrix) * normal); vV = normalize(-mvPosition.xyz);
          gl_Position = projectionMatrix * mvPosition;
          #include <fog_vertex>
        }`,
      fragmentShader: `uniform float uK; varying float vT; varying vec3 vN; varying vec3 vV;
        #include <common>
        #include <fog_pars_fragment>
        void main() { float edge = pow(abs(dot(normalize(vN), normalize(vV))), 1.3);
          float a = uK * (0.3 + 0.7 * edge) * (1.0 - vT) * (1.0 - vT) * smoothstep(0.0, 0.05, vT);
          gl_FragColor = vec4(vec3(1.0, 0.93, 0.8) * a, 1.0);
          #include <fog_fragment>
        }`,
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide, fog: true,
    });
    beamMat.uniforms = THREE.UniformsUtils.merge([THREE.UniformsLib.fog, beamMat.uniforms]);
    this._beamU = beamMat.uniforms;
    const beams = new THREE.InstancedMesh(cone, beamMat, MAX * 2);
    beams.name = 'traffic-headlight-beams'; beams.frustumCulled = false; beams.count = 0; beams.renderOrder = 3;
    beams.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    const pool = new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2);
    const poolMat = new THREE.ShaderMaterial({
      uniforms: THREE.UniformsUtils.merge([THREE.UniformsLib.fog, { uK: { value: 0 } }]),
      vertexShader: `varying vec2 vU;
        #include <common>
        #include <fog_pars_vertex>
        void main() { vU = uv; vec4 mvPosition = modelViewMatrix * instanceMatrix * vec4(position, 1.0); gl_Position = projectionMatrix * mvPosition;
          #include <fog_vertex>
        }`,
      fragmentShader: `uniform float uK; varying vec2 vU;
        #include <common>
        #include <fog_pars_fragment>
        void main() { vec2 q = (vU - 0.5) * 2.0; float r = dot(q, q);
          float a = uK * (1.0 - smoothstep(0.15, 1.0, r)) * (0.55 + 0.45 * smoothstep(-1.0, 0.6, q.y));
          gl_FragColor = vec4(vec3(1.0, 0.92, 0.78) * a, 1.0);
          #include <fog_fragment>
        }`,
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, fog: true,
      polygonOffset: true, polygonOffsetFactor: -6, polygonOffsetUnits: -12,
    });
    this._poolU = poolMat.uniforms;
    const pools = new THREE.InstancedMesh(pool, poolMat, MAX);
    pools.name = 'traffic-headlight-pools'; pools.frustumCulled = false; pools.count = 0; pools.renderOrder = 2;
    pools.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.beams = beams; this.pools = pools;
    this.group.add(beams); this.group.add(pools);
    // per type: headlamp offsets in the vehicle frame (-Z forward)
    this._lampOff = TYPES.map((T) => { const sp = VEHICLE_SPECS[T.name] || {}; return { x: Math.max(0.45, (T.W || 1.8) / 2 - 0.33), y: clamp01h(0.45 * (sp.H || 1.5)), z: -(T.L || 4.6) / 2 + 0.08 }; });
    this._beamCand = null;
  }

  _updateBeams(darkness) {
    if (!this.beams) return;
    const k = Math.min(1, Math.max(0, (darkness - 0.3) / 0.4));
    if (k <= 0.001) { this.beams.count = 0; this.pools.count = 0; return; }
    this._beamU.uK.value = 0.05 * k; this._poolU.uK.value = 0.22 * k;
    const cam = this._cam, R2 = 120 * 120, P = this.pp, A = this.act, V = this.sim.vehicles;
    const C = this._beamCand || (this._beamCand = { d: new Float64Array(V.length), id: new Int32Array(V.length), ord: new Int32Array(V.length) });
    let n = 0;
    for (let i = 0; i < V.length; i++) {
      if (!A[i]) continue;
      const q = i * 6, d2 = (P[q] - cam.x) ** 2 + (P[q + 2] - cam.z) ** 2 + (P[q + 1] - cam.y) ** 2;
      if (d2 < R2) { C.d[n] = d2; C.id[n] = i; n++; }
    }
    if (n > this._beamMax) { // nearest first
      for (let i = 0; i < n; i++) C.ord[i] = i;
      const ord = C.ord.subarray(0, n).sort((a, b) => C.d[a] - C.d[b]);
      for (let i = 0; i < this._beamMax; i++) ord[i] = C.id[ord[i]];
      for (let i = 0; i < this._beamMax; i++) C.id[i] = ord[i];
      n = this._beamMax;
    }
    let bi = 0;
    for (let i = 0; i < n; i++) {
      const id = C.id[i], v = V[id], o = this._lampOff[v.type], q = id * 6;
      _e.set(P[q + 4], P[q + 3], 0, 'YXZ'); _q.setFromEuler(_e);
      _m.compose(_p.set(P[q], P[q + 1], P[q + 2]), _q, _one);
      _bm.makeTranslation(-o.x, o.y, o.z); _bm2.multiplyMatrices(_m, _bm); this.beams.setMatrixAt(bi * 2, _bm2);
      _bm.makeTranslation(o.x, o.y, o.z); _bm2.multiplyMatrices(_m, _bm); this.beams.setMatrixAt(bi * 2 + 1, _bm2);
      _bm.compose(_bp.set(0, 0.07, o.z - 7.5), _bq.identity(), _bs.set((o.x * 2 + 3.2), 1, 11)); _bm2.multiplyMatrices(_m, _bm); this.pools.setMatrixAt(bi, _bm2);
      bi++;
    }
    this.beams.count = bi * 2; this.pools.count = bi;
    this.beams.instanceMatrix.needsUpdate = true; this.pools.instanceMatrix.needsUpdate = true;
  }

  /**
   * Draw the poses in `src` (POSE_STRIDE floats per vehicle), extrapolated by `ext` seconds (worker snapshots are one
   * frame old), packed into their type's instance slots per type in three tiers:
   *   [ near-cascade casters | other vehicles in view | off-screen casters whose shadow can reach the view ]
   * The main pass draws tiers 1+2 (mesh.count; tier 1 may hold a few off-screen near casters), the near cascade
   * tier 1, the far cascade all three. Membership is a sphere test against the camera frustum and the two shadow
   * cameras' frustums (Atmosphere sun / sunFar). Without a camera every active vehicle is drawn in every pass.
   * Matrices are written directly (rotation YXZ: yaw, then pitch); no per-frame allocation.
   */
  _present(src, ext) {
    if (!src) return;
    const W = this.world, A = W && W.atmo;
    const near = A && A.sun && A.sun.castShadow ? A.sun : null, far = A && A.sunFar && A.sunFar.castShadow ? A.sunFar : null;
    const fN = near ? shadowFrustum(near, this._fN || (this._fN = new THREE.Frustum())) : null;
    const fF = far ? shadowFrustum(far, this._fF || (this._fF = new THREE.Frustum())) : null;
    // bounding spheres of the two shadow boxes: a cheap reject before the 6-plane tests
    const sN = fN ? shadowBoxSphere(near, this._sN || (this._sN = new THREE.Sphere())) : null;
    const sF = fF ? shadowBoxSphere(far, this._sF || (this._sF = new THREE.Sphere())) : null;
    this._nearShadowCam = near ? near.shadow.camera : null;
    // view frustum for drawing: World's per-frame view (this frame's camera pose) when available, else the camera's
    const wv = W && W._view;
    const fV = wv && wv.frustum ? wv.frustum : this._hasFrustum ? this._frustum : null;
    // an off-screen vehicle matters to the far cascade only if its shadow can reach the view: test a sphere that
    // covers the vehicle and its shadow (cast away from the sun, length <= 4 m / tan(sun elevation), capped)
    const L = A && A.lightDir, Ly = L ? Math.max(0.05, L.y) : 1, sl = Math.min(90, 4 / Ly) * 0.5;
    const V = this.sim.vehicles, P = this.pp, act = this.act, TN = this._tierN, TI = this._tierIdx;
    TN.fill(0);
    // frustum planes as flat arrays (0: near cascade, 24: view, 48: far cascade): inlined sphere tests
    const FP = this._fp || (this._fp = new Float64Array(72));
    if (fN) loadPlanes(fN, FP, 0);
    if (fV) loadPlanes(fV, FP, 24);
    if (fF) loadPlanes(fF, FP, 48);
    for (let i = 0; i < V.length; i++) {
      const o = i * POSE_STRIDE;
      if (!(src[o + 8] & 1)) { act[i] = 0; continue; }
      const v = V[i];
      let x = src[o], y = src[o + 1], z = src[o + 2], yaw = src[o + 3];
      const pitch = src[o + 4];
      if (ext > 0) {
        const sp = src[o + 6], yr = src[o + 7], d = sp * ext, ym = yaw + yr * ext * 0.5;
        x -= Math.sin(ym) * d; z -= Math.cos(ym) * d; y += Math.tan(pitch) * d; yaw += yr * ext;
      }
      const q = i * 6;
      P[q] = x; P[q + 1] = y; P[q + 2] = z; P[q + 3] = yaw; P[q + 4] = pitch; P[q + 5] = src[o + 5];
      act[i] = 1;
      if (!this.meshes[v.type]) continue;
      let t;
      if (!fV) t = 0; // no camera: every vehicle in every pass
      else {
        const cy = y + v.Hc * 0.5, r = v.L * 0.5 + 0.6;
        if (fN && nearSphere(sN, x, cy, z, r) && inPlanes(FP, 0, x, cy, z, r)) t = 0;
        else if (inPlanes(FP, 24, x, cy, z, r)) t = 1;
        else if (!fF || (nearSphere(sF, x, cy, z, r) && inPlanes(FP, 48, x, cy, z, r))) {
          t = !L || inPlanes(FP, 24, x - L.x * sl, cy - L.y * sl, z - L.z * sl, r + sl) ? 2 : -1;
        } else t = -1;
      }
      if (t < 0) continue;
      const ti = v.type, n = TN[ti * 3 + t]++;
      TI[ti][t * (TI[ti].length / 3) + n] = i;
    }
    for (let ti = 0; ti < this.meshes.length; ti++) {
      const mesh = this.meshes[ti]; if (!mesh) continue;
      const col = mesh.instanceColor ? mesh.instanceColor.array : null, br = mesh.geometry.attributes.aBrake.array, E = mesh.instanceMatrix.array;
      const idx = TI[ti], stride = idx.length / 3;
      let k = 0;
      for (let g = 0; g < 3; g++) {
        const cnt = TN[ti * 3 + g];
        for (let j = 0; j < cnt; j++) {
          const i = idx[g * stride + j], q = i * 6, v = V[i];
          const yaw = P[q + 3], pitch = P[q + 4];
          const ca = Math.cos(pitch), sa = Math.sin(pitch), cy = Math.cos(yaw), sy = Math.sin(yaw), e = k * 16;
          E[e] = cy; E[e + 1] = 0; E[e + 2] = -sy; E[e + 3] = 0;
          E[e + 4] = sy * sa; E[e + 5] = ca; E[e + 6] = cy * sa; E[e + 7] = 0;
          E[e + 8] = ca * sy; E[e + 9] = -sa; E[e + 10] = ca * cy; E[e + 11] = 0;
          E[e + 12] = P[q]; E[e + 13] = P[q + 1]; E[e + 14] = P[q + 2]; E[e + 15] = 1;
          br[k] = P[q + 5];
          if (col) { col[k * 3] = v.rgb[0]; col[k * 3 + 1] = v.rgb[1]; col[k * 3 + 2] = v.rgb[2]; }
          k++;
        }
        if (g === 0) mesh.userData.nNear = k;
        else if (g === 1) mesh.count = k;
      }
      mesh.userData.nFar = k;
      // shadow-less setups (no Atmosphere shadows) still draw tier 1 + 2 in every pass
      if (!fN) mesh.userData.nNear = mesh.count;
      const n = Math.max(1, k);
      mesh.instanceMatrix.clearUpdateRanges(); mesh.instanceMatrix.addUpdateRange(0, n * 16); mesh.instanceMatrix.needsUpdate = true;
      const ab = mesh.geometry.attributes.aBrake; ab.clearUpdateRanges(); ab.addUpdateRange(0, n); ab.needsUpdate = true;
      if (col) { mesh.instanceColor.clearUpdateRanges(); mesh.instanceColor.addUpdateRange(0, n * 3); mesh.instanceColor.needsUpdate = true; }
    }
  }

  // ================================================================ colliders
  /** Append vehicle AABBs {minX,minY,maxX,maxY,minZ,maxZ,kind:'vehicle'} within `radius` of (x,z). Shared objects: read only. */
  queryColliders(x, z, radius, out = []) {
    if (!this.sim) return out;
    if (this.mode !== 'worker' || !this.snap) return this.sim.queryColliders(x, z, radius, out);
    return this._snapQuery(x, z, radius, out);
  }

  /**
   * Worker mode: the same oriented AABB slices as the sim, from the latest snapshot. 250 vehicles are scanned linearly
   * (cheaper than maintaining a grid for a few queries per frame); a vehicle's slices are computed from its snapshot
   * pose the first time a query reaches it (once per snapshot).
   */
  _snapQuery(x, z, radius, out) {
    if (!Number.isFinite(x) || !Number.isFinite(z)) return out;
    const r = Math.max(0, radius || 0), rr = r + 6.5, r2 = r * r;
    if ((2 * rr / 32 + 1) ** 2 > 1024) return out; // the sim's grid query refuses such radii too
    const S = this.snap, C = this._cvs, gen = this._snapGen;
    for (let i = 0; i < C.length; i++) {
      const o = i * POSE_STRIDE;
      if (!(S[o + 8] & 1)) continue;
      const ex = S[o] - x, ez = S[o + 2] - z;
      if (ex > rr || ex < -rr || ez > rr || ez < -rr) continue;
      const c = C[i];
      if (c.gen !== gen) {
        const y = S[o + 1], yaw = S[o + 3], dh = Math.abs(c.WB * Math.tan(S[o + 4])) * 0.5; // axle heights = y -+ dh
        vehicleSlices(c, S[o], S[o + 2], -Math.sin(yaw), -Math.cos(yaw), y - dh, y + c.Hc, S[o + 6]);
        c.gen = gen;
      }
      let b = c.box;
      let dx = x < b.minX ? b.minX - x : x > b.maxX ? x - b.maxX : 0;
      let dz = z < b.minZ ? b.minZ - z : z > b.maxZ ? z - b.maxZ : 0;
      if (dx * dx + dz * dz > r2) continue;
      for (let q = 0; q < c.nb; q++) {
        b = c.boxes[q];
        dx = x < b.minX ? b.minX - x : x > b.maxX ? x - b.maxX : 0;
        dz = z < b.minZ ? b.minZ - z : z > b.maxZ ? z - b.maxZ : 0;
        if (dx * dx + dz * dz <= r2) out.push(b);
      }
    }
    return out;
  }

  /** Debug summary (worker mode: as of the latest snapshot). */
  summary() {
    if (this.mode === 'worker' && this._summary) return this._summary;
    return this.sim ? this.sim.summary() : { count: 0, moving: 0, stopped: 0, onConn: 0, avgSpeed: 0 };
  }

  dispose() {
    if (this.worker) { this.worker.terminate(); this.worker = null; }
    this.mode = 'main';
    for (const m of this.meshes) if (m) { m.geometry.dispose(); m.dispose && m.dispose(); }
    if (this.beams) { this.beams.geometry.dispose(); this.beams.material.dispose(); this.pools.geometry.dispose(); this.pools.material.dispose(); }
    if (this.material) this.material.dispose();
    if (this.group && this.group.parent) this.group.parent.remove(this.group);
  }
}
