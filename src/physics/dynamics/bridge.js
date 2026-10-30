// Bridge between a DynamicsWorld and the engine's PhysicsWorld query hub (pure JS, Node-safe, allocation-free after
// warm-up). The PhysicsWorld is how VehicleController / CharacterController / game raycasts see the world, so
// attaching a dynamics world publishes its moving bodies there as dynamic AABB colliders:
//
//   const physics = new PhysicsWorld();  const dyn = new DynamicsWorld({ physics });
//   attachToPhysicsWorld(dyn, physics);          // crates now block the car, rays from physics.raycast hit them
//   per frame: syncKinematicFromObject(dyn, proxyId, carObject3d); dyn.step(dt);
//
// What the coupling does (one way, deliberately):
// - The provider appends one record { minX, minY, minZ, maxX, maxY, maxZ, kind, id, vx, vz } per alive NON-static
//   body whose world AABB footprint (XZ) is within r of the query point (the ColliderGrid's circle-vs-rectangle rule),
//   ascending body id. `kind` is the body's kind option (dyn.kindOf(id); 'rigid' when the world has no kindOf), and
//   every record carries rec[DYN_RECORD] === true (state.js), the marker solvers use to skip dynamics records.
//   Records are pooled per body id and refreshed from `bodies.aabb` / `vel` at call time: the AABB is whatever the
//   last dynamics step wrote. Pooled identity survives removeBody: a record cached elsewhere (e.g. a PhysicsWorld
//   hit's `box`) silently becomes the next body that reuses that id. Static dynamics bodies are not published (a
//   static trimesh's AABB would be one huge box; put level geometry in the PhysicsWorld instead). Once
//   `dyn.disposed` is true the provider appends nothing.
// - Attach EITHER with attachToPhysicsWorld OR with physics.addDynamicProvider(dyn.colliderProvider), never both:
//   duplicate records make the car's push-out count every body twice.
// - VehicleController collides with these records as boxes: it pushes itself out and bounces (an AABB, so a tilted
//   crate is its bounding box). A kind other than 'vehicle' means a crate whose top is within the car's step height
//   counts as a kerb (driven onto) rather than a wall, and the record's vx / vz make it respond in the crate's frame
//   (VehicleController reads vx / vz on any moving box). The car never moves the crate.
// - The reverse coupling (crates pushed by the car) is the game's job: a KINEMATIC body that follows the car
//   (syncKinematicFromObject each frame) collides with the crates inside the dynamics step. Exclude that proxy from
//   the provider (opts.filter / opts.mask), otherwise the car collides with its own proxy.
// - The DynamicsWorld must not read these records back as static geometry: it collects statics with
//   physics.queryStatic, skips providers named 'dynamicsProvider' (this one) and records marked DYN_RECORD, and
//   filters them out of its physics.raycast.
import { BODY, DYN_RECORD } from './state.js';

const _attached = new WeakMap(); // dyn -> Map<PhysicsWorld, provider>

/**
 * Publish dyn's moving bodies to `physicsWorld` as dynamic colliders; sets `dyn.physics` when it is unset.
 * Attaching the same pair again replaces the previous provider (e.g. to change the options).
 * @param {object} dyn            DynamicsWorld (anything with a BodyStore `bodies`)
 * @param {PhysicsWorld} physicsWorld
 * @param {object} [opts]
 * @param {number} [opts.mask=-1]     only bodies with (bodies.layer & mask) !== 0
 * @param {Function} [opts.filter]    filter(id) => false leaves a body out (e.g. the car's kinematic proxy)
 * @returns {Function} the provider fn(x, z, r, out) (also what detachFromPhysicsWorld removes)
 */
export function attachToPhysicsWorld(dyn, physicsWorld, opts = {}) {
  if (!dyn || !physicsWorld || typeof physicsWorld.addDynamicProvider !== 'function') throw new TypeError('attachToPhysicsWorld(dyn, physicsWorld)');
  let map = _attached.get(dyn);
  if (!map) { map = new Map(); _attached.set(dyn, map); }
  const old = map.get(physicsWorld);
  if (old) physicsWorld.removeDynamicProvider(old);
  const provider = makeProvider(dyn, opts.mask ?? -1, typeof opts.filter === 'function' ? opts.filter : null);
  map.set(physicsWorld, provider);
  provider[DYN_RECORD] = true; // marks the provider itself, like its records
  physicsWorld.addDynamicProvider(provider);
  if (!dyn.physics) dyn.physics = physicsWorld;
  return provider;
}

/** Remove the provider attachToPhysicsWorld registered (dyn.physics is left as it is). Returns true if one was. */
export function detachFromPhysicsWorld(dyn, physicsWorld) {
  const map = _attached.get(dyn);
  const provider = map && map.get(physicsWorld);
  if (!provider) return false;
  physicsWorld.removeDynamicProvider(provider);
  map.delete(physicsWorld);
  return true;
}

function makeProvider(dyn, mask, filter) {
  const pool = [];
  // keep the name: DynamicsWorld skips providers named 'dynamicsProvider' when it collects static boxes
  const kindOf = typeof dyn.kindOf === 'function' ? dyn.kindOf.bind(dyn) : null;
  return function dynamicsProvider(x, z, r, out) {
    if (dyn.disposed) return; // a disposed world must not keep publishing a frozen mirror
    const B = dyn.bodies, A = B.aabb, V = B.vel, n = B.count;
    const rr = r > 0 ? r * r : 0;
    for (let id = 0; id < n; id++) {
      if (!B.alive[id] || B.type[id] === BODY.STATIC) continue;
      if (mask !== -1 && (B.layer[id] & mask) === 0) continue;
      const o = id * 6;
      const minX = A[o], minZ = A[o + 2], maxX = A[o + 3], maxZ = A[o + 5];
      const dx = x < minX ? minX - x : x > maxX ? x - maxX : 0;
      const dz = z < minZ ? minZ - z : z > maxZ ? z - maxZ : 0;
      if (dx * dx + dz * dz > rr) continue;
      if (filter && !filter(id)) continue;
      let rec = pool[id];
      if (rec === undefined) {
        while (pool.length < id) pool.push(undefined); // keep the array packed (ids are dense)
        // 0.5 seeds double field representations (no map transition when the first fractional value lands)
        rec = pool[id] = { minX: 0.5, minY: 0.5, maxX: 0.5, maxY: 0.5, minZ: 0.5, maxZ: 0.5, kind: 'rigid', id, vx: 0.5, vz: 0.5 };
        rec[DYN_RECORD] = true;
      }
      rec.kind = kindOf ? kindOf(id) ?? 'rigid' : 'rigid';
      rec.minX = minX; rec.minY = A[o + 1]; rec.minZ = minZ; rec.maxX = maxX; rec.maxY = A[o + 4]; rec.maxZ = maxZ;
      rec.vx = V[id * 3]; rec.vz = V[id * 3 + 2];
      out.push(rec);
    }
  };
}

/**
 * Drive kinematic body `id` from an Object3D's WORLD transform: setKinematicTarget(id, x, y, z, qx, qy, qz, qw)
 * for the next step (the solver moves it there and derives its velocity, so it pushes dynamic bodies). Scale is
 * removed from the rotation. Updates the object's world matrix (parents included) first. Allocation-free.
 */
export function syncKinematicFromObject(dyn, id, object3d) {
  if (typeof object3d.updateWorldMatrix === 'function') object3d.updateWorldMatrix(true, false);
  const e = object3d.matrixWorld.elements;
  // normalised rotation columns
  let ax = e[0], ay = e[1], az = e[2], bx = e[4], by = e[5], bz = e[6], cx = e[8], cy = e[9], cz = e[10];
  let l = Math.sqrt(ax * ax + ay * ay + az * az); if (l > 0) { ax /= l; ay /= l; az /= l; }
  l = Math.sqrt(bx * bx + by * by + bz * bz); if (l > 0) { bx /= l; by /= l; bz /= l; }
  l = Math.sqrt(cx * cx + cy * cy + cz * cz); if (l > 0) { cx /= l; cy /= l; cz /= l; }
  // rotation matrix -> quaternion (Shepperd, largest diagonal term first). Column-major: m_rc with columns
  // a = (m00, m10, m20), b = (m01, m11, m21), c = (m02, m12, m22).
  const tr = ax + by + cz;
  let qx, qy, qz, qw;
  if (tr > 0) {
    const s = 0.5 / Math.sqrt(tr + 1);
    qw = 0.25 / s; qx = (bz - cy) * s; qy = (cx - az) * s; qz = (ay - bx) * s;
  } else if (ax > by && ax > cz) {
    const s = 2 * Math.sqrt(1 + ax - by - cz);
    qw = (bz - cy) / s; qx = 0.25 * s; qy = (bx + ay) / s; qz = (cx + az) / s;
  } else if (by > cz) {
    const s = 2 * Math.sqrt(1 + by - ax - cz);
    qw = (cx - az) / s; qx = (bx + ay) / s; qy = 0.25 * s; qz = (cy + bz) / s;
  } else {
    const s = 2 * Math.sqrt(1 + cz - ax - by);
    qw = (ay - bx) / s; qx = (cx + az) / s; qy = (cy + bz) / s; qz = 0.25 * s;
  }
  dyn.setKinematicTarget(id, e[12], e[13], e[14], qx, qy, qz, qw);
}
