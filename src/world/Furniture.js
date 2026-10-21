// Procedural street furniture (North-American city style): geometry library (one merged geometry per prop type,
// vertex-coloured, per-vertex roughness / metalness / night emission / paint mask), a sign atlas
// (CanvasTexture: regulatory signs, green street-name blades for every road name, shelter posters,
// manhole and storm-drain decals; the city-specific texts / posters / blade colours are a `style` option of
// buildSignAtlas, see DEFAULT_SIGN_STYLE) and the materials that read them. Placement lives in Props.js.
// Lamp styles: cobra (arterial), cobraRes (residential), pathOfGold (ornate trident standard), doubleGlobe
// (twin-globe promenade lamp), acorn (plaza) - which streets get which is the game's choice (Props options).
//
// Local frames (origin on the ground / sidewalk top, +Y up):
//   lamps: arm reaches along +X  ·  signal heads & ped heads face -Z (origin: hanger / pole centre)
//   mast arm: unit length along +X (scale X)  ·  signPost: unit height (scale Y)
//   bench / shelter / meter / newsBox / mailbox: front faces -Z  ·  utility pole: crossarm along X
import * as THREE from 'three';
import { PartBuilder } from './Vehicles.js';

// ------------------------------------------------------------------ vertex materials
export const FM = {
  galv: { c: [0.46, 0.48, 0.49], rough: 0.42, metal: 0.85 },
  greyPole: { c: [0.3, 0.31, 0.32], rough: 0.5, metal: 0.6 },
  iron: { c: [0.022, 0.024, 0.022], rough: 0.42, metal: 0.65 },
  bronze: { c: [0.055, 0.06, 0.045], rough: 0.36, metal: 0.75 },
  bronzeHi: { c: [0.16, 0.13, 0.06], rough: 0.3, metal: 0.85 },
  goldGlass: { c: [1.0, 0.56, 0.2], rough: 0.15, metal: 0, emit: 0.42 },
  ledLens: { c: [1.0, 0.84, 0.62], rough: 0.2, metal: 0, emit: 0.8 },
  globe: { c: [0.95, 0.88, 0.74], rough: 0.25, metal: 0, emit: 0.55 },
  housing: { c: [0.026, 0.026, 0.028], rough: 0.55, metal: 0.25 },
  yellow: { c: [0.78, 0.56, 0.02], rough: 0.45, metal: 0.1 },
  concrete: { c: [0.46, 0.45, 0.42], rough: 0.92, metal: 0 },
  wood: { c: [0.17, 0.12, 0.08], rough: 0.92, metal: 0 },
  slat: { c: [0.3, 0.19, 0.1], rough: 0.8, metal: 0 },
  white: { c: [0.8, 0.8, 0.78], rough: 0.5, metal: 0.1 },
  paint: { c: [1, 1, 1], rough: 0.45, metal: 0.25, paint: 1 },
  paintGloss: { c: [1, 1, 1], rough: 0.3, metal: 0.3, paint: 1 },
  uspsBlue: { c: [0.025, 0.09, 0.3], rough: 0.4, metal: 0.35 },
  meterGrey: { c: [0.26, 0.27, 0.29], rough: 0.45, metal: 0.6 },
  screen: { c: [0.3, 0.6, 0.42], rough: 0.1, metal: 0, emit: 0.25 },
  glassDark: { c: [0.04, 0.05, 0.06], rough: 0.08, metal: 0.3 },
  alu: { c: [0.5, 0.51, 0.53], rough: 0.32, metal: 0.85 },
  shrub: { c: [0.42, 0.6, 0.32], rough: 0.9, metal: 0, paint: 1 },
  soil: { c: [0.06, 0.045, 0.03], rough: 1, metal: 0 },
  ceramic: { c: [0.5, 0.5, 0.46], rough: 0.4, metal: 0 },
  black: { c: [0.018, 0.018, 0.018], rough: 0.6, metal: 0.1 },
  reflect: { c: [0.85, 0.85, 0.8], rough: 0.3, metal: 0.2 },
  lightStrip: { c: [0.95, 0.97, 1.0], rough: 0.3, metal: 0, emit: 0.9 },
  pushBtn: { c: [0.55, 0.5, 0.08], rough: 0.4, metal: 0.4 },
  chalk: { c: [0.03, 0.045, 0.04], rough: 0.85, metal: 0 },
  chalkText: { c: [0.55, 0.56, 0.52], rough: 0.8, metal: 0 },
  potGrey: { c: [0.13, 0.13, 0.135], rough: 0.55, metal: 0.05 },
  teak: { c: [0.32, 0.2, 0.1], rough: 0.7, metal: 0 },
};

const M4 = (x = 0, y = 0, z = 0, rx = 0, ry = 0, rz = 0, sx = 1, sy = 1, sz = 1) =>
  new THREE.Matrix4().compose(new THREE.Vector3(x, y, z), new THREE.Quaternion().setFromEuler(new THREE.Euler(rx, ry, rz, 'YXZ')), new THREE.Vector3(sx, sy, sz));
const cyl = (B, x, y0, z, r0, r1, h, seg, m) => B.lathe([[r0, 0], [r1, h]], seg, m, M4(x, y0, z));
const disc = (B, x, y, z, r, seg, m, down = false) => B.lathe(down ? [[0, 0], [r, 0]] : [[r, 0], [0, 0]], seg, m, M4(x, y, z));
const sphere = (B, x, y, z, r, w, h, m, sx = 1, sy = 1, sz = 1) => B.geo(new THREE.SphereGeometry(r, w, h), m, M4(x, y, z, 0, 0, 0, sx, sy, sz));
/** Visor shell: half tube over the top of a lens, axis along -Z, both faces. */
function visor(B, y, zFront, r, len, m) {
  const g = new THREE.CylinderGeometry(r, r, len, 6, 1, true, Math.PI / 2, Math.PI).rotateX(Math.PI / 2).translate(0, y, zFront - len / 2);
  B.geo(g, m);
  const inner = g.clone(); const n = inner.attributes.normal;
  for (let i = 0; i < n.count; i++) n.setXYZ(i, -n.getX(i), -n.getY(i), -n.getZ(i));
  B.geo(inner, m);
}

// ------------------------------------------------------------------ geometry builders
const BUILDERS = {
  /** Arterial cobra-head street light: tapered steel pole, upswept arm, LED cobra head. */
  cobra: (B) => cobraLight(B, 9.0, 2.5),
  cobraRes: (B) => cobraLight(B, 7.6, 1.8),
  /** Ornate trident standard (San Francisco "Path of Gold" style): ornate base, fluted shaft, trident of gold lanterns. */
  pathOfGold: (B) => {
    B.lathe([[0.36, -0.3], [0.36, 0.05], [0.31, 0.12], [0.3, 0.5], [0.25, 0.62], [0.24, 0.95], [0.17, 1.15], [0.13, 1.3]], 10, FM.bronze);
    B.lathe([[0.31, 0.3], [0.335, 0.34], [0.31, 0.38]], 10, FM.bronzeHi);
    B.lathe([[0.115, 1.3], [0.09, 6.55]], 12, (j, i) => (i % 2 ? FM.bronze : FM.bronzeHi));
    B.lathe([[0.09, 6.5], [0.16, 6.6], [0.16, 6.72], [0.1, 6.82], [0.1, 6.92]], 10, FM.bronze);
    lantern(B, 0, 6.92, 0, 1.0);
    for (const s of [-1, 1]) {
      B.tube([[0.06 * s, 6.72, 0], [0.42 * s, 6.74, 0], [0.68 * s, 6.86, 0], [0.8 * s, 7.06, 0]], 0.035, 5, FM.bronze);
      lantern(B, 0.8 * s, 7.06, 0, 0.8);
    }
  },
  /** Twin-globe promenade lamp (crossarm runs along the kerb: FURNITURE_META.alongKerb). */
  doubleGlobe: (B) => {
    B.lathe([[0.24, -0.3], [0.24, 0.1], [0.2, 0.18], [0.19, 0.55], [0.13, 0.75], [0.1, 0.9]], 10, FM.iron);
    cyl(B, 0, 0.9, 0, 0.085, 0.065, 3.4, 8, FM.iron);
    B.lathe([[0.065, 4.25], [0.11, 4.32], [0.11, 4.4], [0.06, 4.46]], 8, FM.iron);
    B.tube([[-0.62, 4.62, 0], [-0.5, 4.42, 0], [-0.25, 4.36, 0], [0, 4.4, 0], [0.25, 4.36, 0], [0.5, 4.42, 0], [0.62, 4.62, 0]], 0.032, 5, FM.iron);
    for (const s of [-1, 1]) {
      B.lathe([[0.04, 4.6], [0.09, 4.66], [0.06, 4.7]], 8, FM.iron, M4(0.62 * s, 0, 0));
      sphere(B, 0.62 * s, 4.92, 0, 0.23, 10, 7, FM.globe);
      B.lathe([[0.07, 5.13], [0.03, 5.2], [0, 5.24]], 6, FM.iron, M4(0.62 * s, 0, 0));
    }
    B.lathe([[0.05, 4.46], [0.03, 4.75], [0.06, 4.8], [0, 4.9]], 6, FM.iron);
  },
  /** Plaza / park acorn lamp (OSM street_lamp nodes away from roads). */
  acorn: (B) => {
    B.lathe([[0.17, -0.3], [0.17, 0.08], [0.13, 0.16], [0.12, 0.5], [0.08, 0.65]], 8, FM.iron);
    cyl(B, 0, 0.65, 0, 0.06, 0.05, 3.15, 7, FM.iron);
    B.lathe([[0.05, 3.8], [0.12, 3.88], [0.1, 3.94]], 8, FM.iron);
    B.lathe([[0.1, 3.94], [0.19, 4.12], [0.17, 4.42], [0.07, 4.6], [0, 4.62]], 8, FM.globe);
    B.lathe([[0.05, 4.6], [0.03, 4.72], [0, 4.78]], 6, FM.iron);
  },
  /** Signal mast pole (7 m) with base cover and arm plate; the arm itself is `mastArm`. */
  mastPole: (B) => {
    B.lathe([[0.34, -0.3], [0.34, 0.02], [0.3, 0.08], [0.22, 0.12], [0.2, 0.42]], 10, FM.greyPole);
    cyl(B, 0, 0.42, 0, 0.19, 0.13, 6.6, 10, FM.galv);
    disc(B, 0, 7.02, 0, 0.13, 10, FM.galv);
    B.box(0.12, 6.3, 0, 0.06, 0.42, 0.36, FM.galv);
  },
  mastArm: (B) => {
    B.geo(new THREE.CylinderGeometry(0.065, 0.115, 1, 8, 1, true).rotateZ(-Math.PI / 2).translate(0.5, 0, 0), FM.galv);
    B.lathe([[0.065, 0], [0, 0]], 8, FM.galv, M4(1, 0, 0, 0, 0, -Math.PI / 2));
  },
  /** Pedestal signal pole (4.6 m). */
  pedPole: (B) => {
    B.lathe([[0.2, -0.3], [0.2, 0.02], [0.15, 0.08], [0.11, 0.12], [0.1, 0.32]], 8, FM.greyPole);
    cyl(B, 0, 0.32, 0, 0.09, 0.075, 4.3, 8, FM.galv);
    disc(B, 0, 4.62, 0, 0.075, 8, FM.galv);
  },
  /** 3-section 12" vehicle signal head with tunnel visors and a yellow-bordered backplate. */
  sigHead: (B) => {
    B.box(0, -0.03, 0, 0.07, 0.1, 0.07, FM.housing);
    B.box(0, -0.6, 0, 0.36, 1.06, 0.25, FM.housing);
    for (let k = 0; k < 3; k++) visor(B, -0.6 + 0.335 * (1 - k), -0.125, 0.165, 0.24, FM.housing);
    B.box(0, -0.6, 0.135, 0.64, 1.34, 0.02, FM.black);
    const z = 0.124, w = 0.045, X = 0.32, Y0 = -1.27, Y1 = 0.07;
    B.quad([-X, Y0, z], [X, Y0, z], [X, Y0 + w, z], [-X, Y0 + w, z], FM.yellow, [0, 0, -1]);
    B.quad([-X, Y1 - w, z], [X, Y1 - w, z], [X, Y1, z], [-X, Y1, z], FM.yellow, [0, 0, -1]);
    B.quad([-X, Y0, z], [-X + w, Y0, z], [-X + w, Y1, z], [-X, Y1, z], FM.yellow, [0, 0, -1]);
    B.quad([X - w, Y0, z], [X, Y0, z], [X, Y1, z], [X - w, Y1, z], FM.yellow, [0, 0, -1]);
  },
  /** Pedestrian signal head on a bracket + push button (origin: pole centre, faces -Z). */
  pedHead: (B) => {
    B.box(0, 2.75, -0.2, 0.06, 0.06, 0.24, FM.housing);
    B.box(0, 2.75, -0.42, 0.4, 0.44, 0.2, FM.housing);
    B.box(0, 2.975, -0.6, 0.42, 0.015, 0.17, FM.housing);
    for (const s of [-1, 1]) B.box(0.205 * s, 2.86, -0.6, 0.015, 0.22, 0.17, FM.housing);
    B.box(0, 1.05, -0.14, 0.05, 0.05, 0.14, FM.housing);
    B.box(0, 1.05, -0.24, 0.11, 0.17, 0.07, FM.pushBtn);
    B.box(0, 1.05, -0.276, 0.05, 0.05, 0.012, FM.black);
    B.box(0, 1.3, -0.215, 0.15, 0.22, 0.012, FM.white);
  },
  signPost: (B) => cyl(B, 0, 0, 0, 0.032, 0.03, 1, 6, FM.galv),
  /** Fire hydrant: white barrel, painted bonnet (instance colour), two hose + one pumper nozzle. */
  hydrant: (B) => {
    const prof = [[0.16, -0.08], [0.16, 0.06], [0.12, 0.09], [0.11, 0.13], [0.12, 0.52], [0.15, 0.55], [0.15, 0.61], [0.12, 0.64], [0.11, 0.7], [0.075, 0.79], [0.03, 0.83], [0, 0.84]];
    B.lathe(prof, 8, (j) => (j >= 7 ? FM.paintGloss : FM.white));
    B.box(0, 0.86, 0, 0.05, 0.05, 0.05, FM.paintGloss);
    for (const s of [-1, 1]) B.lathe([[0.045, 0], [0.045, 0.12], [0.032, 0.14]], 6, FM.white, M4(0.1 * s, 0.42, 0, 0, 0, -s * Math.PI / 2));
    B.lathe([[0.07, 0], [0.07, 0.12], [0.05, 0.15]], 7, FM.white, M4(0, 0.38, -0.1, -Math.PI / 2, 0, 0));
  },
  /** Steel bollard with a reflective band. */
  bollard: (B) => B.lathe([[0.1, -0.1], [0.1, 0.6], [0.1, 0.68], [0.1, 0.84], [0.075, 0.92], [0, 0.95]], 8, (j) => (j === 1 ? FM.reflect : FM.greyPole)),
  /** "Renaissance" style litter receptacle (instance colour: green / black). */
  trashCan: (B) => B.lathe([[0.3, -0.02], [0.31, 0.06], [0.3, 0.12], [0.3, 0.78], [0.32, 0.82], [0.3, 0.87], [0.22, 0.97], [0.12, 1.02], [0, 1.04]], 10,
    (j) => (j === 4 ? FM.black : FM.paint)),
  /** Slatted bench with cast-iron ends (seat faces -Z). */
  bench: (B) => {
    for (let k = 0; k < 3; k++) B.box(0, 0.45, -0.18 + k * 0.13, 1.8, 0.035, 0.1, FM.slat);
    for (let k = 0; k < 2; k++) B.box(0, 0.62 + k * 0.15, 0.13 + k * 0.035, 1.8, 0.1, 0.03, FM.slat, { pitch: -0.25 });
    for (const x of [-0.78, 0.78]) {
      B.box(x, 0.22, -0.2, 0.06, 0.46, 0.05, FM.iron);
      B.box(x, 0.42, 0.14, 0.06, 0.84, 0.05, FM.iron, { pitch: -0.2 });
      B.box(x, 0.62, -0.05, 0.06, 0.04, 0.42, FM.iron);
    }
  },
  /** Circle bike rack. */
  bikeRack: (B) => {
    B.geo(new THREE.TorusGeometry(0.3, 0.025, 4, 12), FM.galv, M4(0, 0.78, 0));
    cyl(B, 0, -0.1, 0, 0.03, 0.03, 0.58, 6, FM.galv);
  },
  /** Newspaper box (instance colour). */
  newsBox: (B) => {
    B.box(0, 0.18, 0, 0.36, 0.36, 0.3, FM.black);
    B.box(0, 0.72, 0, 0.5, 0.74, 0.42, FM.paint);
    B.box(0, 1.1, 0, 0.52, 0.04, 0.44, FM.paint);
    B.box(0, 0.8, -0.212, 0.38, 0.4, 0.01, FM.glassDark);
    B.box(0.16, 1.03, -0.22, 0.12, 0.08, 0.03, FM.meterGrey);
  },
  /** USPS blue collection box. */
  mailbox: (B) => {
    const sh = new THREE.Shape();
    sh.moveTo(-0.27, 0); sh.lineTo(0.27, 0); sh.lineTo(0.27, 0.72); sh.absarc(0, 0.72, 0.27, 0, Math.PI, false); sh.lineTo(-0.27, 0);
    const g = new THREE.ExtrudeGeometry(sh, { depth: 0.48, bevelEnabled: false, curveSegments: 6 }).translate(0, 0.28, -0.24);
    B.geo(g, FM.uspsBlue);
    for (const x of [-0.25, 0.25]) B.box(x, 0.14, 0, 0.04, 0.32, 0.46, FM.uspsBlue);
    B.box(0, 0.95, -0.245, 0.3, 0.05, 0.02, FM.black);
  },
  /** Single-space smart parking meter (screen faces -Z). */
  meter: (B) => {
    cyl(B, 0, -0.1, 0, 0.04, 0.04, 1.2, 6, FM.meterGrey);
    B.box(0, 1.24, 0, 0.17, 0.3, 0.14, FM.meterGrey);
    B.geo(new THREE.CylinderGeometry(0.085, 0.085, 0.14, 6, 1, false, 0, Math.PI).rotateX(Math.PI / 2).rotateZ(Math.PI / 2).translate(0, 1.39, 0), FM.meterGrey);
    B.box(0, 1.28, -0.072, 0.11, 0.07, 0.01, FM.screen);
    B.box(0, 1.17, -0.072, 0.07, 0.05, 0.01, FM.black);
    B.box(0, 1.5, 0.01, 0.16, 0.012, 0.12, FM.glassDark, { pitch: 0.35 });
  },
  /** Bus shelter frame, bench, ad box and roof light (the glass is a separate transparent mesh). */
  shelter: (B) => {
    for (const x of [-1.95, 0, 1.95]) B.box(x, 1.15, 0.68, 0.08, 2.9, 0.08, FM.alu);
    for (const z of [-0.7, 0.7]) B.box(0, 2.62, z, 4.06, 0.09, 0.07, FM.alu);
    for (const x of [-1.95, 0, 1.95]) B.box(x, 2.6, 0, 0.07, 0.07, 1.45, FM.alu);
    B.box(0, 2.555, -0.05, 3.5, 0.02, 0.1, FM.lightStrip);
    B.box(2.08, 1.25, 0.05, 0.17, 1.95, 1.3, FM.alu);
    B.box(0.2, 0.46, 0.42, 1.8, 0.04, 0.4, FM.alu);
    for (const x of [-0.6, 1.0]) B.box(x, 0.22, 0.5, 0.05, 0.46, 0.05, FM.alu);
    B.box(0.2, 0.86, 0.64, 1.8, 0.07, 0.04, FM.alu);
    B.box(-1.2, 1.55, 0.655, 0.62, 0.9, 0.04, FM.glassDark);
  },
  /** Wooden utility pole with crossarm, insulators and a comms cable bracket. */
  utilityPole: (B) => utilityPole(B, false),
  utilityPoleTx: (B) => utilityPole(B, true),
  /** Sidewalk A-frame (sandwich) board: two chalkboards leaning together; frame painted (instance colour). Boards face +-Z. */
  aframe: (B) => {
    for (const sg of [-1, 1]) {
      const o = { pitch: sg * 0.2 };
      B.box(0, 0.47, sg * 0.1, 0.62, 0.94, 0.028, FM.paint, o);
      B.box(0, 0.5, sg * 0.117, 0.5, 0.66, 0.006, FM.chalk, o);
      for (let k = 0; k < 4; k++) B.box(-0.04 + (k % 2) * 0.06, 0.66 - k * 0.12, sg * 0.122, 0.34 - (k % 3) * 0.06, 0.035, 0.004, FM.chalkText, o);
    }
    B.box(0, 0.93, 0, 0.6, 0.03, 0.06, FM.paint);
  },
  /** Bistro set: round metal table + two chairs facing it along X (chair frames painted per instance). */
  cafeSet: (B) => {
    B.lathe([[0.22, 0], [0.2, 0.02], [0.04, 0.05], [0.03, 0.72]], 8, FM.iron);
    B.lathe([[0.34, 0.72], [0.35, 0.74], [0.34, 0.76], [0, 0.765]], 12, FM.alu);
    for (const sg of [-1, 1]) {
      const x = sg * 0.62;
      for (const [lx, lz] of [[-0.17, -0.17], [0.17, -0.17], [0.17, 0.17], [-0.17, 0.17]]) B.box(x + lx, 0.22, lz, 0.025, 0.44, 0.025, FM.paint);
      B.box(x, 0.45, 0, 0.42, 0.03, 0.42, FM.teak);
      B.box(x + sg * 0.2, 0.68, 0, 0.025, 0.46, 0.4, FM.paint, { roll: sg * 0.12 });
      for (let k = 0; k < 3; k++) B.box(x + sg * 0.215 + sg * k * 0.004, 0.62 + k * 0.12, 0, 0.02, 0.07, 0.38, FM.teak, { roll: sg * 0.12 });
    }
  },
  /** Tall shop-front pot with a clipped shrub (shrub tinted per instance). */
  pot: (B) => {
    B.lathe([[0.2, 0], [0.26, 0.02], [0.29, 0.62], [0.31, 0.64], [0.31, 0.68], [0.27, 0.68]], 10, FM.potGrey);
    B.lathe([[0.27, 0.66], [0, 0.66]], 10, FM.soil);
    const g = new THREE.IcosahedronGeometry(1, 1);
    const p = g.attributes.position;
    for (let i = 0; i < p.count; i++) { const x = p.getX(i), y = p.getY(i), z = p.getZ(i); const f = 0.88 + 0.2 * Math.abs(Math.sin(x * 6.1 + y * 4.3 + z * 5.2)); p.setXYZ(i, x * f, y * f, z * f); }
    g.computeVertexNormals();
    B.geo(g, FM.shrub, M4(0, 1.02, 0, 0, 0, 0, 0.34, 0.42, 0.34));
  },
  /** Concrete planter with a shrub (shrub tinted per instance). */
  planter: (B) => {
    B.box(0, 0.3, 0, 1.4, 0.6, 0.6, FM.concrete, { skipBottom: true });
    B.box(0, 0.61, 0, 1.3, 0.02, 0.5, FM.soil);
    const g = new THREE.IcosahedronGeometry(1, 1);
    const p = g.attributes.position;
    for (let i = 0; i < p.count; i++) { const x = p.getX(i), y = p.getY(i), z = p.getZ(i); const f = 0.85 + 0.25 * Math.abs(Math.sin(x * 7.1 + y * 3.3 + z * 5.7)); p.setXYZ(i, x * f, y * f, z * f); }
    g.computeVertexNormals();
    B.geo(g, FM.shrub, M4(0, 0.85, 0, 0, 0, 0, 0.62, 0.38, 0.27));
  },
};

function cobraLight(B, H, A) {
  B.lathe([[0.2, -0.3], [0.2, 0.42], [0.15, 0.52], [0.13, 0.58]], 8, FM.galv);
  cyl(B, 0, 0.58, 0, 0.12, 0.068, H - 0.58, 8, FM.galv);
  disc(B, 0, H, 0, 0.068, 8, FM.galv);
  B.tube([[0.02, H - 0.62, 0], [0.4, H - 0.3, 0], [0.95, H - 0.06, 0], [A * 0.7, H + 0.04, 0], [A, H + 0.06, 0]], [0.055, 0.05, 0.045, 0.042, 0.04], 6, FM.galv);
  // cobra head: rounded upper shell, flat lens underneath
  B.geo(new THREE.SphereGeometry(1, 10, 4, 0, Math.PI * 2, 0, Math.PI / 2), FM.greyPole, M4(A + 0.28, H, 0, 0, 0, 0, 0.42, 0.15, 0.2));
  B.lathe([[0, 0], [1, 0]], 10, FM.ledLens, M4(A + 0.3, H - 0.005, 0, 0, 0, 0, 0.34, 1, 0.15));
}
function lantern(B, x, y, z, s) {
  B.lathe([[0.05 * s, 0], [0.12 * s, 0.05 * s], [0.06 * s, 0.1 * s]], 8, FM.bronze, M4(x, y, z));
  B.lathe([[0.07 * s, 0.1 * s], [0.2 * s, 0.32 * s], [0.21 * s, 0.5 * s], [0.14 * s, 0.68 * s], [0.06 * s, 0.76 * s]], 8, FM.goldGlass, M4(x, y, z));
  B.lathe([[0.08 * s, 0.75 * s], [0.13 * s, 0.8 * s], [0.04 * s, 0.9 * s], [0, 1.0 * s]], 8, FM.bronze, M4(x, y, z));
}
function utilityPole(B, tx) {
  B.lathe([[0.16, -0.5], [0.115, 11.0]], 7, FM.wood);
  disc(B, 0, 11.0, 0, 0.115, 7, FM.wood);
  B.box(0, 10.2, 0, 2.4, 0.1, 0.12, FM.wood);
  for (const s of [-1, 1]) B.box(0.38 * s, 9.85, 0, 0.05, 0.75, 0.04, FM.galv, { roll: s * 0.75 });
  for (const x of [-0.95, 0, 0.95]) B.lathe([[0.05, 0], [0.06, 0.06], [0.04, 0.1], [0.05, 0.14], [0, 0.17]], 5, FM.ceramic, M4(x, 10.25, 0));
  B.box(0.18, 7.6, 0, 0.28, 0.06, 0.06, FM.galv);
  if (tx) {
    B.geo(new THREE.CylinderGeometry(0.28, 0.28, 0.85, 8, 1), FM.meterGrey, M4(0, 8.6, 0.42));
    B.box(0, 8.75, 0.2, 0.1, 0.5, 0.12, FM.galv);
  }
}

/** Local light / wire attachment points. */
export const FURNITURE_META = {
  cobra: { light: [2.5 + 0.3, 9.0 - 0.08, 0] },
  cobraRes: { light: [1.8 + 0.3, 7.6 - 0.08, 0] },
  pathOfGold: { light: [0, 7.3, 0] },
  doubleGlobe: { light: [0, 4.9, 0], alongKerb: true },
  acorn: { light: [0, 4.25, 0] },
  utilityPole: { wires: [[-0.95, 10.4, 0], [0, 10.4, 0], [0.95, 10.4, 0], [0.32, 7.6, 0]] },
};

const _geo = new Map();
/** Furniture geometry (furniture attribute layout), cached. */
export function makeFurnitureGeometry(name) {
  let g = _geo.get(name);
  if (!g) {
    const B = new PartBuilder();
    (BUILDERS[name] || BUILDERS.bollard)(B);
    g = B.build('furniture');
    g.userData.name = name;
    _geo.set(name, g);
  }
  return g;
}
export const FURNITURE_TYPES = Object.keys(BUILDERS);

/** Bus-shelter glazing (roof, back wall, one end), for the transparent glass mesh. */
export function makeShelterGlassGeometry() {
  const parts = [
    new THREE.PlaneGeometry(4.05, 1.48).rotateX(-Math.PI / 2 + 0.04).translate(0, 2.68, 0),
    new THREE.PlaneGeometry(3.8, 1.9).translate(0, 1.3, 0.7),
    new THREE.PlaneGeometry(1.3, 1.9).rotateY(Math.PI / 2).translate(-2.0, 1.3, 0.02),
  ];
  const B = new PartBuilder();
  for (const p of parts) B.geo(p, FM.white);
  const g = B.build('furniture');
  for (const k of ['color', 'aSurf']) g.deleteAttribute(k);
  return g;
}

/** Unit sign panel: front quad (normal +Z) + back quad (normal -Z, u mirrored); attribute aBack. */
export function makePanelGeometry() {
  const P = [], N = [], U = [], K = [];
  const q = (z, nz, back) => {
    const v = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]];
    const order = back ? [0, 2, 1, 3, 5, 4] : [0, 1, 2, 3, 4, 5];
    for (const i of order) { const [x, y] = v[i]; P.push(x, y, z); N.push(0, 0, nz); U.push(back ? 0.5 - x : x + 0.5, y + 0.5); K.push(back ? 1 : 0); }
  };
  q(0.004, 1, false); q(-0.004, -1, true);
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(P, 3));
  g.setAttribute('normal', new THREE.Float32BufferAttribute(N, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(U, 2));
  g.setAttribute('aBack', new THREE.Float32BufferAttribute(K, 1));
  return g;
}
/** Unit flat decal (normal +Y). */
export function makeDecalGeometry() {
  const g = new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2);
  g.setAttribute('aBack', new THREE.Float32BufferAttribute(new Float32Array(g.attributes.position.count), 1));
  return g;
}

// ------------------------------------------------------------------ materials
/** Furniture: vertex colour x instance colour (paint mask), per-vertex roughness/metalness, night emission. */
export function createFurnitureMaterial() {
  const m = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.6, metalness: 0.3 });
  const u = { uNight: { value: 0 } };
  m.onBeforeCompile = (sh) => {
    sh.uniforms.uNight = u.uNight;
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', '#include <common>\nattribute vec4 aSurf; uniform float uNight; varying vec4 vSurf; varying vec3 vEmit;')
      .replace('#include <color_vertex>', `
vColor = vec3( 1.0 );
#ifdef USE_COLOR
vColor *= color;
#endif
vec3 tintC = vec3( 1.0 );
#ifdef USE_INSTANCING_COLOR
tintC = instanceColor.xyz;
#endif
#ifdef USE_BATCHING_COLOR
tintC = getBatchingColor( getIndirectIndex( gl_DrawID ) );
#endif
vColor = mix( vColor, vColor * tintC, aSurf.w );
vSurf = aSurf;
vEmit = color * aSurf.z * uNight;`);
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', '#include <common>\nvarying vec4 vSurf; varying vec3 vEmit;')
      .replace('#include <roughnessmap_fragment>', 'float roughnessFactor = vSurf.x;')
      .replace('#include <metalnessmap_fragment>', 'float metalnessFactor = vSurf.y;')
      .replace('#include <emissivemap_fragment>', '#include <emissivemap_fragment>\ntotalEmissiveRadiance += vEmit;');
  };
  m.customProgramCacheKey = () => 'roadster-furniture-1';
  m.userData.u = u;
  return m;
}
/**
 * Sign panels / decals from the atlas: per-instance aUvA (front rect), aUvB (back rect), aGlow (night
 * emission); vertex-shader distance cull (uCull). decal: polygon offset for road decals.
 */
export function createPanelMaterial(atlas, { decal = false, cull = 400 } = {}) {
  const m = new THREE.MeshStandardMaterial({
    map: atlas || null, color: atlas ? 0xffffff : 0x8a8d90, alphaTest: atlas ? 0.5 : 0, roughness: decal ? 0.8 : 0.55, metalness: decal ? 0.35 : 0.05,
    polygonOffset: decal, polygonOffsetFactor: decal ? -4 : 0, polygonOffsetUnits: decal ? -8 : 0,
  });
  const u = { uNight: { value: 0 }, uCull: { value: cull } };
  m.onBeforeCompile = (sh) => {
    sh.uniforms.uNight = u.uNight; sh.uniforms.uCull = u.uCull;
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', '#include <common>\nattribute vec4 aUvA; attribute vec4 aUvB; attribute float aGlow; attribute float aBack;\nuniform float uCull; varying float vGlow;')
      .replace('#include <uv_vertex>', `#include <uv_vertex>
#ifdef USE_MAP
{ vec4 r = aBack > 0.5 ? aUvB : aUvA; vMapUv = r.xy + uv * r.zw; }
#endif
vGlow = aGlow;`)
      .replace('#include <begin_vertex>', `#include <begin_vertex>
#ifdef USE_INSTANCING
{ vec3 ip = ( modelMatrix * instanceMatrix * vec4( 0.0, 0.0, 0.0, 1.0 ) ).xyz; if ( distance( ip, cameraPosition ) > uCull ) transformed = vec3( 0.0 ); }
#endif`);
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', '#include <common>\nuniform float uNight; varying float vGlow;')
      .replace('#include <emissivemap_fragment>', '#include <emissivemap_fragment>\ntotalEmissiveRadiance += diffuseColor.rgb * vGlow * uNight;');
  };
  m.customProgramCacheKey = () => `roadster-panel-1-${decal ? 1 : 0}`;
  m.userData.u = u;
  return m;
}
export function createShelterGlassMaterial() {
  return new THREE.MeshStandardMaterial({ color: 0xa8c4cc, roughness: 0.06, metalness: 0.1, transparent: true, opacity: 0.2, depthWrite: false, side: THREE.DoubleSide, forceSinglePass: true });
}

// ------------------------------------------------------------------ sign atlas
const AW = 2048, AH = 2048, CELL_W = 256, CELL_H = 48, NAME_Y0 = 512;
const SUFFIX = { street: 'ST', avenue: 'AVE', boulevard: 'BLVD', place: 'PL', alley: 'ALY', lane: 'LN', court: 'CT', terrace: 'TER', way: 'WAY', plaza: 'PLZ', road: 'RD', drive: 'DR', square: 'SQ', walk: 'WALK', highway: 'HWY' };
/** "Main Street" -> ["MAIN", "ST"]. */
export function formatStreetName(n) {
  const w = String(n || '').trim().split(/\s+/);
  if (w.length > 1) { const s = SUFFIX[w[w.length - 1].toLowerCase()]; if (s) return [w.slice(0, -1).join(' ').toUpperCase(), s]; }
  return [w.join(' ').toUpperCase(), ''];
}
const ICON = {
  stop: [0, 0, 240, 240], allway: [256, 0, 240, 80], speed25: [256, 96, 120, 150], speed30: [384, 96, 120, 150],
  noPark: [512, 0, 160, 240], clean: [688, 0, 160, 240], oneR: [864, 0, 300, 100], oneL: [864, 112, 300, 100],
  bus: [1184, 0, 140, 240], back: [1340, 8, 56, 56], manhole: [1420, 0, 240, 240], drain: [1680, 0, 300, 140],
  poster0: [0, 256, 168, 248], poster1: [176, 256, 168, 248], poster2: [352, 256, 168, 248], poster3: [528, 256, 168, 248],
  poster4: [704, 256, 168, 248], poster5: [880, 256, 168, 248],
};
const toUV = ([x, y, w, h], inset = 1) => [(x + inset) / AW, 1 - (y + h - inset) / AH, (w - 2 * inset) / AW, (h - 2 * inset) / AH];

/**
 * City-specific atlas content. posters: 6 x [gradient top, gradient bottom, line 1, line 2, art ('circle' | 'sun' |
 * 'hills' | 'cup')]; manholeText: cast lettering on the manhole covers; blade: street-name blade colours.
 */
export const DEFAULT_SIGN_STYLE = {
  posters: [
    ['#f2b134', '#c2410c', 'CITY FILM', 'FESTIVAL', 'circle'], ['#0e7490', '#082f49', 'RIDE THE', 'STREETCAR', 'hills'],
    ['#fde68a', '#16a34a', 'PARK', 'CONCERTS', 'sun'], ['#e0e7ff', '#4338ca', 'HARBOR BREW', 'COFFEE CO.', 'cup'],
    ['#fecdd3', '#be123c', 'SUMMER', 'SALE  -40%', 'circle'], ['#111827', '#f59e0b', 'SYMPHONY', 'TONIGHT', 'sun'],
  ],
  manholeText: 'SEWER',
  blade: { bg: '#0d5f37', border: '#e8efe9', text: '#f4f7f4' },
};

/**
 * Builds the atlas. Returns { texture (null without a DOM canvas), uv(key) -> [u0, v0, du, dv],
 * nameUV(roadName) -> rect (allocated on demand, up to 256 names) }.
 * @param {string[]} names road names for the street-name blades
 * @param {object} [style] overrides of DEFAULT_SIGN_STYLE ({ posters, manholeText, blade })
 */
export function buildSignAtlas(names = [], style = {}) {
  const ST = { ...DEFAULT_SIGN_STYLE, ...(style || {}) };
  ST.blade = { ...DEFAULT_SIGN_STYLE.blade, ...((style && style.blade) || {}) };
  const slots = new Map();
  const uniq = [...new Set(names.filter(Boolean))].slice(0, (AH - NAME_Y0) / CELL_H * (AW / CELL_W));
  uniq.forEach((n, i) => slots.set(n, [(i % 8) * CELL_W, NAME_Y0 + Math.floor(i / 8) * CELL_H, CELL_W, CELL_H]));
  const out = {
    texture: null,
    uv: (k) => toUV(ICON[k] || ICON.back, k === 'stop' || k === 'manhole' ? 0 : 2),
    nameUV: (n) => (slots.has(n) ? toUV(slots.get(n), 2) : toUV(ICON.back)),
    hasName: (n) => slots.has(n),
  };
  let canvas = null;
  try {
    if (typeof document !== 'undefined') { canvas = document.createElement('canvas'); canvas.width = AW; canvas.height = AH; }
    else if (typeof OffscreenCanvas !== 'undefined') canvas = new OffscreenCanvas(AW, AH);
  } catch { canvas = null; }
  const ctx = canvas && canvas.getContext('2d');
  if (!ctx) return out;
  drawIcons(ctx, ST);
  for (const [n, r] of slots) drawBlade(ctx, n, r, ST.blade);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;
  tex.generateMipmaps = true;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  out.texture = tex;
  return out;
}

const FONT = '"Helvetica Neue", Helvetica, Arial, sans-serif';
function fitText(ctx, text, x, y, maxW, size, weight = 'bold') {
  ctx.font = `${weight} ${size}px ${FONT}`;
  const w = ctx.measureText(text).width;
  if (w > maxW) { ctx.save(); ctx.translate(x, y); ctx.scale(maxW / w, 1); ctx.fillText(text, 0, 0); ctx.restore(); }
  else ctx.fillText(text, x, y);
}
function drawBlade(ctx, name, [x, y, w, h], C = DEFAULT_SIGN_STYLE.blade) {
  ctx.fillStyle = C.bg; ctx.fillRect(x, y, w, h);
  ctx.strokeStyle = C.border; ctx.lineWidth = 2; ctx.strokeRect(x + 4, y + 4, w - 8, h - 8);
  const [base, suf] = formatStreetName(name);
  ctx.fillStyle = C.text; ctx.textBaseline = 'middle'; ctx.textAlign = 'center';
  const sufW = suf ? 34 : 0;
  fitText(ctx, base, x + (w - sufW) / 2 + 2, y + h / 2 + 1, w - 22 - sufW, 30);
  if (suf) { ctx.font = `bold 15px ${FONT}`; ctx.textAlign = 'left'; ctx.fillText(suf, x + w - 12 - sufW + 4, y + h / 2 - 5); }
}
function drawIcons(ctx, ST = DEFAULT_SIGN_STYLE) {
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  // STOP (octagon, transparent corners)
  {
    const [x, y, w] = ICON.stop, cx = x + w / 2, cy = y + w / 2;
    const oct = (r) => { ctx.beginPath(); for (let k = 0; k < 8; k++) { const a = Math.PI / 8 + (k * Math.PI) / 4; ctx.lineTo(cx + Math.cos(a) * r, cy + Math.sin(a) * r); } ctx.closePath(); };
    ctx.fillStyle = '#f2f2f2'; oct(w / 2 / Math.cos(Math.PI / 8) - 1); ctx.fill();
    ctx.fillStyle = '#b3121b'; oct(w / 2 / Math.cos(Math.PI / 8) - 9); ctx.fill();
    ctx.fillStyle = '#ffffff'; fitText(ctx, 'STOP', cx, cy + 3, w * 0.8, 74);
  }
  const rect = (r, bg, border, bw = 6) => { const [x, y, w, h] = r; ctx.fillStyle = border; ctx.fillRect(x, y, w, h); ctx.fillStyle = bg; ctx.fillRect(x + bw, y + bw, w - 2 * bw, h - 2 * bw); };
  { const r = ICON.allway; rect(r, '#b3121b', '#f2f2f2', 5); ctx.fillStyle = '#fff'; fitText(ctx, 'ALL WAY', r[0] + r[2] / 2, r[1] + r[3] / 2 + 2, r[2] - 30, 46); }
  for (const [k, n] of [['speed25', '25'], ['speed30', '30']]) {
    const r = ICON[k]; rect(r, '#f4f4f0', '#111', 5); ctx.fillStyle = '#111';
    fitText(ctx, 'SPEED', r[0] + r[2] / 2, r[1] + 26, r[2] - 20, 22); fitText(ctx, 'LIMIT', r[0] + r[2] / 2, r[1] + 50, r[2] - 20, 22);
    fitText(ctx, n, r[0] + r[2] / 2, r[1] + 102, r[2] - 16, 70);
  }
  {
    const r = ICON.noPark; rect(r, '#f4f4f0', '#c21d24', 5); const cx = r[0] + r[2] / 2;
    ctx.strokeStyle = '#c21d24'; ctx.lineWidth = 7; ctx.beginPath(); ctx.arc(cx, r[1] + 62, 38, 0, Math.PI * 2); ctx.stroke();
    ctx.fillStyle = '#111'; fitText(ctx, 'P', cx, r[1] + 64, 60, 52);
    ctx.beginPath(); ctx.moveTo(cx - 27, r[1] + 35); ctx.lineTo(cx + 27, r[1] + 89); ctx.stroke();
    ctx.fillStyle = '#c21d24'; fitText(ctx, 'NO', cx, r[1] + 126, r[2] - 20, 28); fitText(ctx, 'PARKING', cx, r[1] + 156, r[2] - 18, 26);
    fitText(ctx, 'ANY TIME', cx, r[1] + 192, r[2] - 18, 22); fitText(ctx, 'TOW-AWAY', cx, r[1] + 218, r[2] - 18, 18);
  }
  {
    const r = ICON.clean; rect(r, '#f4f4f0', '#1d6b3a', 5); const cx = r[0] + r[2] / 2;
    ctx.fillStyle = '#c21d24'; fitText(ctx, 'NO PARKING', cx, r[1] + 32, r[2] - 18, 22);
    ctx.fillStyle = '#1d6b3a'; fitText(ctx, '8 AM - 10 AM', cx, r[1] + 70, r[2] - 18, 20); fitText(ctx, 'TUESDAY', cx, r[1] + 100, r[2] - 18, 22);
    ctx.fillStyle = '#111'; fitText(ctx, 'STREET', cx, r[1] + 150, r[2] - 18, 24); fitText(ctx, 'CLEANING', cx, r[1] + 180, r[2] - 18, 24);
    ctx.fillStyle = '#c21d24'; fitText(ctx, 'TOW-AWAY', cx, r[1] + 216, r[2] - 18, 16);
  }
  for (const [k, dir] of [['oneR', 1], ['oneL', -1]]) {
    const [x, y, w, h] = ICON[k]; ctx.fillStyle = '#111'; ctx.fillRect(x, y, w, h);
    const cy = y + h / 2, x0 = x + 18, x1 = x + w - 18, hb = 20, hh = 38, head = 58;
    ctx.fillStyle = '#f4f4f0'; ctx.beginPath();
    if (dir > 0) { ctx.moveTo(x0, cy - hb); ctx.lineTo(x1 - head, cy - hb); ctx.lineTo(x1 - head, cy - hh); ctx.lineTo(x1, cy); ctx.lineTo(x1 - head, cy + hh); ctx.lineTo(x1 - head, cy + hb); ctx.lineTo(x0, cy + hb); }
    else { ctx.moveTo(x1, cy - hb); ctx.lineTo(x0 + head, cy - hb); ctx.lineTo(x0 + head, cy - hh); ctx.lineTo(x0, cy); ctx.lineTo(x0 + head, cy + hh); ctx.lineTo(x0 + head, cy + hb); ctx.lineTo(x1, cy + hb); }
    ctx.closePath(); ctx.fill();
    ctx.fillStyle = '#111'; fitText(ctx, 'ONE WAY', x + w / 2 - dir * 22, cy + 2, 170, 30);
  }
  {
    const r = ICON.bus; rect(r, '#f4f4f0', '#b3121b', 6); const cx = r[0] + r[2] / 2;
    ctx.fillStyle = '#b3121b'; ctx.fillRect(r[0] + 6, r[1] + 6, r[2] - 12, 64);
    ctx.fillStyle = '#fff'; fitText(ctx, 'BUS', cx, r[1] + 40, r[2] - 24, 40);
    ctx.fillStyle = '#111'; ctx.fillRect(cx - 34, r[1] + 92, 68, 52); ctx.fillStyle = '#f4f4f0'; ctx.fillRect(cx - 28, r[1] + 98, 56, 22);
    ctx.beginPath(); ctx.arc(cx - 20, r[1] + 148, 8, 0, 7); ctx.arc(cx + 20, r[1] + 148, 8, 0, 7); ctx.fillStyle = '#111'; ctx.fill();
    fitText(ctx, 'STOP', cx, r[1] + 192, r[2] - 24, 34); ctx.font = `bold 15px ${FONT}`; ctx.fillText('NO PARKING', cx, r[1] + 222);
  }
  { const [x, y, w, h] = ICON.back; const g = ctx.createLinearGradient(x, y, x + w, y + h); g.addColorStop(0, '#9a9da1'); g.addColorStop(1, '#7c7f83'); ctx.fillStyle = g; ctx.fillRect(x, y, w, h); }
  {
    const [x, y, w] = ICON.manhole, cx = x + w / 2, cy = y + w / 2, R = w / 2 - 2;
    ctx.fillStyle = '#3a3936'; ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = '#1f1e1c'; ctx.lineWidth = 6; ctx.beginPath(); ctx.arc(cx, cy, R - 8, 0, Math.PI * 2); ctx.stroke();
    ctx.lineWidth = 3;
    for (let r2 = 20; r2 < R - 20; r2 += 18) { ctx.beginPath(); ctx.arc(cx, cy, r2, 0, Math.PI * 2); ctx.stroke(); }
    for (let k = 0; k < 16; k++) { const a = (k / 16) * Math.PI * 2; ctx.beginPath(); ctx.moveTo(cx + Math.cos(a) * 20, cy + Math.sin(a) * 20); ctx.lineTo(cx + Math.cos(a) * (R - 18), cy + Math.sin(a) * (R - 18)); ctx.stroke(); }
    ctx.fillStyle = '#4a4844'; ctx.fillRect(cx - 50, cy - 16, 100, 32); ctx.fillStyle = '#26251f'; fitText(ctx, ST.manholeText, cx, cy + 1, 92, 18);
  }
  {
    const [x, y, w, h] = ICON.drain; ctx.fillStyle = '#6d6a64'; ctx.fillRect(x, y, w, h);
    ctx.fillStyle = '#15140f'; ctx.fillRect(x + 16, y + 16, w - 32, h - 32);
    ctx.fillStyle = '#4b4943'; for (let k = 0; k < 13; k++) ctx.fillRect(x + 24 + k * 20, y + 16, 7, h - 32);
    ctx.fillRect(x + 16, y + h / 2 - 4, w - 32, 8);
  }
  const posters = ST.posters || DEFAULT_SIGN_STYLE.posters;
  posters.slice(0, 6).forEach(([c0, c1, t0, t1, art], i) => {
    const [x, y, w, h] = ICON[`poster${i}`];
    const g = ctx.createLinearGradient(x, y, x, y + h); g.addColorStop(0, c0); g.addColorStop(1, c1); ctx.fillStyle = g; ctx.fillRect(x, y, w, h);
    ctx.fillStyle = 'rgba(255,255,255,0.75)';
    if (art === 'circle') { ctx.beginPath(); ctx.arc(x + w * 0.5, y + h * 0.36, w * 0.28, 0, 7); ctx.fill(); }
    if (art === 'sun') { ctx.beginPath(); ctx.arc(x + w * 0.62, y + h * 0.3, w * 0.18, 0, 7); ctx.fill(); ctx.fillRect(x + 10, y + h * 0.5, w - 20, 4); }
    if (art === 'hills') { ctx.beginPath(); ctx.moveTo(x, y + h * 0.6); ctx.quadraticCurveTo(x + w * 0.3, y + h * 0.2, x + w * 0.6, y + h * 0.45); ctx.quadraticCurveTo(x + w * 0.8, y + h * 0.3, x + w, y + h * 0.5); ctx.lineTo(x + w, y + h * 0.62); ctx.lineTo(x, y + h * 0.62); ctx.fill(); }
    if (art === 'cup') { ctx.fillRect(x + w * 0.32, y + h * 0.2, w * 0.36, h * 0.3); ctx.beginPath(); ctx.arc(x + w * 0.7, y + h * 0.33, 14, 0, 7); ctx.fill(); }
    ctx.fillStyle = '#ffffff'; fitText(ctx, t0, x + w / 2, y + h * 0.74, w - 16, 24); fitText(ctx, t1, x + w / 2, y + h * 0.86, w - 16, 18, '600');
  });
}
