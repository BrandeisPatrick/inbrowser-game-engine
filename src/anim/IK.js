// IK: analytic two-bone IK (limbs) and a name-based wrapper over three's CCDIKSolver (chains, tails, spines).
//
//   const leg = new TwoBoneIK(thigh, shin, foot, { pole: kneeHint });   // pole: world Vector3 or an Object3D
//   leg.solve(footTargetWorld, weight);                                 // after the mixer, before rendering
//   leg.enabled = false; leg.weight = 0.5; leg.lengthA / leg.lengthB;
//
//   const tail = new CCDIK(skinnedMeshOrSkeleton, [
//     { target: aimObject3D, effector: 'Tail5', links: ['Tail4', 'Tail3', 'Tail2'], iteration: 10 },
//     { target: 'HandTarget', effector: 'Hand', links: 2 },   // links: n = the n parents above the effector
//   ]);
//   tail.solve(weight); tail.enabled; tail.chains (CCDIKSolver `iks` entries); tail.solver
//   makeIKChain(skinnedMesh, { target, effector, links, iteration }) -> an `iks` entry for CCDIKSolver itself
//
// TwoBoneIK works on any three Object3Ds where lower descends from upper and end from lower (normally Bones). It
// only rotates upper and lower: (1) bend the middle joint about the current bend-plane normal so the upper->end
// distance is the clamped upper->target distance (law of cosines on lengthA / lengthB), (2) swing upper so the end
// lies on the upper->target line, (3) with a pole, twist about that line so the middle joint points at the pole.
// A target out of reach leaves the limb fully extended towards it (at most lengthA + lengthB, at least
// |lengthA - lengthB|). A straight limb with no pole bends about the lower bone's local X. weight < 1 slerps both
// local rotations from their pre-solve values. The end bone's own local rotation is untouched (it follows the
// lower bone); world matrices of upper..end are current after solve(), deeper descendants refresh with the
// scene. Lengths are world distances measured at construction (pass them if the rig is rescaled). solve() reads
// the chain's ancestors fresh (updateWorldMatrix) and allocates nothing.
//
// CCDIK: CCDIKSolver wants a SkinnedMesh and bone INDICES, and its target must be a skeleton bone. The wrapper
// takes a SkinnedMesh or a bare Skeleton, bone names / indices / Bones, and any Object3D as target (appended to a
// private copy of the bone list the solver sees), and refreshes the world matrices it reads before solving
// (updateWorldMatrix(true, false) on each target and effector). A target outside the skeleton's subtree is read from
// its own hierarchy: if it (or an ancestor) has matrixAutoUpdate / matrixWorldAutoUpdate off, its world matrix is
// whatever the caller last made it, so update it yourself before solve().
// Chain fields besides target / effector / links (iteration, minAngle, maxAngle, blendFactor) and per-link fields
// (limitation, rotationMin, rotationMax, enabled; give the link as { bone, ... }) pass through to CCDIKSolver.
import { Vector3, Quaternion } from 'three';
import { CCDIKSolver } from 'three/addons/animation/CCDIKSolver.js';

const EPS = 1e-12;
const _a = new Vector3(), _b = new Vector3(), _c = new Vector3(), _t = new Vector3(), _p = new Vector3();
const _ab = new Vector3(), _cb = new Vector3(), _ac = new Vector3(), _at = new Vector3(), _n = new Vector3();
const _k = new Vector3(), _sp = new Vector3(), _ss = new Vector3();   // decompose() position / scale sinks
const _q = new Quaternion(), _qp = new Quaternion(), _qx = new Quaternion();

/** Rotate obj in world space: its world rotation becomes q * (world rotation). Parent's matrixWorld must be current. */
function rotateWorld(obj, q) {
  if (obj.parent) {
    obj.parent.matrixWorld.decompose(_sp, _qp, _ss);
    _qx.copy(_qp).invert().multiply(q).multiply(_qp);
    obj.quaternion.premultiply(_qx);
  } else obj.quaternion.premultiply(q);
}
/** Update world matrices from `top` down to `obj` (top's parent assumed current). */
function refresh(top, obj) {
  if (obj !== top && obj.parent) refresh(top, obj.parent);
  obj.updateWorldMatrix(false, false);
}

export class TwoBoneIK {
  /**
   * @param {THREE.Object3D} upper  e.g. thigh / upper arm
   * @param {THREE.Object3D} lower  e.g. shin / forearm (descendant of upper)
   * @param {THREE.Object3D} end    e.g. foot / hand (descendant of lower)
   * @param {object} [o]
   * @param {THREE.Vector3|THREE.Object3D} [o.pole=null]  world point the middle joint bends towards
   * @param {number} [o.lengthA]  upper->lower length (default: measured now, world units)
   * @param {number} [o.lengthB]  lower->end length (default: measured now)
   */
  constructor(upper, lower, end, { pole = null, lengthA = null, lengthB = null } = {}) {
    this.upper = upper; this.lower = lower; this.end = end;
    this.pole = pole;
    this.enabled = true;
    this.weight = 1;
    end.updateWorldMatrix(true, false);
    _a.setFromMatrixPosition(upper.matrixWorld); _b.setFromMatrixPosition(lower.matrixWorld); _c.setFromMatrixPosition(end.matrixWorld);
    this.lengthA = lengthA ?? _a.distanceTo(_b);
    this.lengthB = lengthB ?? _b.distanceTo(_c);
    this._q0 = new Quaternion(); this._q1 = new Quaternion();
  }

  /**
   * Rotate upper and lower so end reaches the target (world position), or points at it fully extended.
   * @param {THREE.Vector3} target  world position
   * @param {number} [weight=1]  blend with the pre-solve pose (multiplied by this.weight)
   */
  solve(target, weight = 1) {
    const w = weight * this.weight;
    if (!this.enabled || !(w > 0) || !(this.lengthA > 0 && this.lengthB > 0)) return this;
    const A = this.upper, B = this.lower, C = this.end, la = this.lengthA, lb = this.lengthB;
    C.updateWorldMatrix(true, false);
    if (w < 1) { this._q0.copy(A.quaternion); this._q1.copy(B.quaternion); }
    _a.setFromMatrixPosition(A.matrixWorld); _b.setFromMatrixPosition(B.matrixWorld); _c.setFromMatrixPosition(C.matrixWorld);
    _t.copy(target);
    _at.subVectors(_t, _a);
    const dist = _at.length();
    const lat = Math.min(Math.max(dist, Math.abs(la - lb)), la + lb);

    // 1. middle joint: interior angle for |upper->end| = lat
    _ab.subVectors(_a, _b); _cb.subVectors(_c, _b);
    const beta0 = _ab.angleTo(_cb);
    const cosB = (la * la + lb * lb - lat * lat) / (2 * la * lb);
    const beta1 = Math.acos(Math.min(1, Math.max(-1, cosB)));
    // EPS tests are relative (sin^2 of the angle between the vectors, or the share of a length left after projection)
    const abSq = _ab.lengthSq(), tiny = EPS * (la + lb) * (la + lb);
    _n.crossVectors(_ab, _cb);
    if (_n.lengthSq() < EPS * abSq * _cb.lengthSq()) {     // straight (or folded): pick a bend axis
      let ok = false;
      if (this.pole) {
        _p.copy(this.pole.isVector3 ? this.pole : this.pole.getWorldPosition(_p)).sub(_b);
        _n.crossVectors(_ab, _p);
        ok = _n.lengthSq() >= EPS * abSq * _p.lengthSq();
      }
      if (!ok) {
        _n.setFromMatrixColumn(B.matrixWorld, 0);
        const nSq = _n.lengthSq();
        _n.addScaledVector(_ab, -_n.dot(_ab) / (abSq || 1));
        if (!(_n.lengthSq() >= EPS * nSq)) {   // any perpendicular: cross with the axis least aligned of X / Y
          const xs = Math.abs(_ab.x) < Math.abs(_ab.y);
          _n.set(xs ? 1 : 0, xs ? 0 : 1, 0).cross(_ab);
        }
      }
    }
    _n.normalize();
    if (Math.abs(beta1 - beta0) > 1e-15) { rotateWorld(B, _q.setFromAxisAngle(_n, beta1 - beta0)); refresh(B, C); }

    // 2. swing upper so the end lies on the upper->target line
    if (dist * dist > tiny) {
      _c.setFromMatrixPosition(C.matrixWorld);
      _ac.subVectors(_c, _a);
      if (_ac.lengthSq() > tiny) {
        _at.divideScalar(dist);
        rotateWorld(A, _q.setFromUnitVectors(_ac.normalize(), _at));
        refresh(A, C);
      }

      // 3. pole: twist about the upper->target axis so the middle joint faces the pole
      if (this.pole) {
        _p.copy(this.pole.isVector3 ? this.pole : this.pole.getWorldPosition(_p)).sub(_a);
        _b.setFromMatrixPosition(B.matrixWorld);
        _k.subVectors(_b, _a);
        const kSq = _k.lengthSq(), pSq = _p.lengthSq();
        _k.addScaledVector(_at, -_k.dot(_at));
        _p.addScaledVector(_at, -_p.dot(_at));
        if (_k.lengthSq() > EPS * kSq && _p.lengthSq() > EPS * pSq) {
          const ang = Math.atan2(_n.crossVectors(_k, _p).dot(_at), _k.dot(_p));
          if (Math.abs(ang) > 1e-15) { rotateWorld(A, _q.setFromAxisAngle(_at, ang)); refresh(A, C); }
        }
      }
    }

    if (w < 1) {
      _q.copy(A.quaternion); A.quaternion.copy(this._q0).slerp(_q, w);
      _q.copy(B.quaternion); B.quaternion.copy(this._q1).slerp(_q, w);
      refresh(A, C);
    }
    return this;
  }
}

/** Resolve a bone given as name / index / Object3D to its index in `bones` (appending when `extend`). */
function boneIndex(bones, ref, extend, what) {
  if (typeof ref === 'number') { if (bones[ref]) return ref; }
  else if (typeof ref === 'string') { for (let i = 0; i < bones.length; i++) if (bones[i].name === ref) return i; }
  else if (ref && ref.isObject3D) {
    const i = bones.indexOf(ref);
    if (i >= 0) return i;
    if (extend) { bones.push(ref); return bones.length - 1; }
  }
  throw new Error(`IK: ${what} '${ref && ref.isObject3D ? ref.name || ref.uuid : ref}' is not in the skeleton`);
}

function buildChain(bones, spec, extend) {
  const { target, effector, links, ...rest } = spec;
  const eff = boneIndex(bones, effector, false, 'effector');
  const out = { ...rest, target: boneIndex(bones, target, extend, 'target'), effector: eff, links: [] };
  if (out.iteration === undefined) out.iteration = 10;
  if (typeof links === 'number') {
    for (let o = bones[eff].parent, n = 0; o && n < links; o = o.parent, n++) out.links.push({ index: boneIndex(bones, o, false, 'link') });
  } else {
    for (const l of links) {
      if (l && typeof l === 'object' && !l.isObject3D) {
        const { bone, name, index, ...lr } = l;
        out.links.push({ ...lr, index: boneIndex(bones, bone ?? name ?? index, false, 'link') });
      } else out.links.push({ index: boneIndex(bones, l, false, 'link') });
    }
  }
  return out;
}

/**
 * Build a CCDIKSolver `iks` entry from bone names (or indices / Bones). The target must be a skeleton bone here;
 * use CCDIK for arbitrary Object3D targets.
 * @param {THREE.SkinnedMesh|THREE.Skeleton} meshOrSkeleton
 * @param {object} chain  { target, effector, links: [names] | n, iteration = 10, minAngle, maxAngle, blendFactor }
 */
export function makeIKChain(meshOrSkeleton, chain) {
  const sk = meshOrSkeleton.isSkinnedMesh ? meshOrSkeleton.skeleton : meshOrSkeleton;
  return buildChain(sk.bones, chain, false);
}

export class CCDIK {
  /**
   * @param {THREE.SkinnedMesh|THREE.Skeleton} meshOrSkeleton
   * @param {object[]} chains  [{ target (name | index | Object3D), effector, links: [names | { bone, ...limits }] | n,
   *                           iteration = 10, minAngle, maxAngle, blendFactor }]
   */
  constructor(meshOrSkeleton, chains = []) {
    this.skeleton = meshOrSkeleton.isSkinnedMesh ? meshOrSkeleton.skeleton : meshOrSkeleton;
    this.enabled = true;
    this.weight = 1;
    this._bones = this.skeleton.bones.slice();   // + Object3D targets outside the skeleton
    /** CCDIKSolver `iks` entries (indices into the solver's bone list). */
    this.chains = chains.map((c) => buildChain(this._bones, c, true));
    this.solver = new CCDIKSolver({ skeleton: { bones: this._bones } }, this.chains);
  }

  /** Solve every chain (call after the mixer). weight blends with the pre-solve pose (x this.weight). */
  solve(weight = 1) {
    const w = weight * this.weight;
    if (!this.enabled || !(w > 0)) return this;
    const bones = this._bones, ch = this.chains;
    for (let i = 0; i < ch.length; i++) { bones[ch[i].target].updateWorldMatrix(true, false); bones[ch[i].effector].updateWorldMatrix(true, false); }
    this.solver.update(w);
    return this;
  }
}
