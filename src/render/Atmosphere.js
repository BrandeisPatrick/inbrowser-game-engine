// Atmosphere: sky (THREE.Sky, patched: bigger sun disc + aureole, blue-hour / night model, stars, moon, drifting
// procedural clouds), sun / moon light with a two-cascade camera-following shadow setup, hemisphere light
// with ground bounce, PMREM environment from the sky, lens flare (occluded by the depth buffer) and fog.
// Fog: `scene` mode = FogExp2 matched to the tone-mapped horizon (no post-processing); `post` mode = no scene
// fog, PostFX does height + distance fog coloured by `skyCube` (sky radiance without sun disc / clouds).
// Every look parameter is keyed on the sun elevation (`looks` keyframes) so any hour works, not only presets.
//
// API
//   const atmo = new Atmosphere({ scene, renderer, camera?, timeOfDay: 'golden' | hours,
//                                 shadows: { cascades: 2, near: 45, far: [260, 420, 650], kernel: 'pcf' }, ...tunings });
//   (legacy: new Atmosphere(scene, renderer, opts) followed by atmo.build())
//   atmo.update(cameraPos, dt?)    per frame: sky follows the camera, clouds drift, cascades + flare follow
//   atmo.setTimeOfDay(hoursOrPreset) / atmo.setTime(hours)
//   atmo.darkness (0 day .. 1 night), atmo.windowGlow (night emissive factor), atmo.sunDir, atmo.sunColor,
//   atmo.fogColor, atmo.fogParams (post fog), atmo.skyCube, atmo.version (bumps on every time change)
//   atmo.setFogMode('scene' | 'post'), atmo.renderFlare(renderer, camera), atmo.setFlareScale(bufferHeight)
//   atmo.setShadowKernel('pcf' | 'fast', root?)  sun shadow filter, switchable at runtime; atmo.shadowKernel
// Tunings (constructor options; defaults are the Roadster values): sun (solar model), looks (keyframes),
// skyGain, sunIntensity, moon, cloudWind, fog, maxCasterHeight, flare, timePresets.
import * as THREE from 'three';
import { Sky } from 'three/addons/objects/Sky.js';
import { Lensflare, LensflareElement } from 'three/addons/objects/Lensflare.js';
import { clamp, smoothstep } from './noise.js';
import { makeCloudTexture, sampleCloud, makeFlareTextures } from './skyTextures.js';
import { installCascadeShadowChunk } from './shadowCascade.js';
import { installFastShadowChunk, createShadowDepthTexture, attachShadowDepth, setFastShadowMaps, applyShadowKernel } from './shadowKernel.js';

const DEG = Math.PI / 180;
export const TIME_PRESETS = { dawn: 7.25, morning: 9.5, noon: 13, afternoon: 15.5, golden: 17.55, sunset: 18.45, dusk: 18.95, night: 22 };

const CASCADES = installCascadeShadowChunk(); // global chunk patch, inert unless the far cascade light exists

/**
 * Default solar model: San Francisco in October. rise / set in hours, maxElevation (deg) at solar noon,
 * nightRate (deg per hour below the horizon, capped at nightDepth), azimuth from north, clockwise, sweeping
 * azimuthStart .. azimuthStart + azimuthSpan between sunrise and sunset.
 */
export const DEFAULT_SUN = { rise: 7.15, set: 18.6, maxElevation: 46, nightDepth: 35, nightRate: 13, azimuthStart: 102, azimuthSpan: 156 };

/** Sun elevation / azimuth (radians, azimuth from north, clockwise) for an hour of the day. */
export function sunPosition(hours, model = DEFAULT_SUN) {
  const rise = model.rise, set = model.set, maxEl = model.maxElevation * DEG;
  const f = (hours - rise) / (set - rise);
  let el;
  if (f >= 0 && f <= 1) el = maxEl * Math.sin(Math.PI * f);
  else el = -Math.min(model.nightDepth * DEG, (f > 1 ? hours - set : rise - hours < 0 ? rise - hours + 24 : rise - hours) * model.nightRate * DEG);
  const az = (model.azimuthStart + model.azimuthSpan * clamp(f, -0.15, 1.15)) * DEG;
  return { el, az };
}

function acesFilmic(c, exposure) { // mirror of three's ACESFilmicToneMapping (linear in -> linear out)
  let r = c.r * exposure / 0.6, g = c.g * exposure / 0.6, b = c.b * exposure / 0.6;
  const ir = 0.59719 * r + 0.35458 * g + 0.04823 * b;
  const ig = 0.07600 * r + 0.90834 * g + 0.01566 * b;
  const ib = 0.02840 * r + 0.13383 * g + 0.83777 * b;
  const fit = (v) => (v * (v + 0.0245786) - 0.000090537) / (v * (0.983729 * v + 0.4329510) + 0.238081);
  const fr = fit(ir), fg = fit(ig), fb = fit(ib);
  return new THREE.Color(
    clamp(1.60475 * fr - 0.53108 * fg - 0.07367 * fb, 0, 1),
    clamp(-0.10208 * fr + 1.10813 * fg - 0.00605 * fb, 0, 1),
    clamp(-0.00327 * fr - 0.07276 * fg + 1.07602 * fb, 0, 1));
}

// ------------------------------------------------------------------ look keyframes (by sun elevation, degrees)
// turb/ray/mie/g: Preetham sky. tw: blue-hour/night sky replaces Preetham (0..1). night: night sky + stars.
// glowA: afterglow band at the sunset point. sun: sun-light gain. env/hemi: ambient. haze/mist: post fog
// densities (1/m at sea level; haze falls off over ~700 m, mist over ~40 m). cover/cirrus/cop: clouds.
// cs/ca: cloud sun / ambient colours (linear HDR). aur: sun aureole gain.
export const DEFAULT_LOOKS = [
  { el: -40, turb: 2.4, ray: 1.4, mie: 0.0030, g: 0.78, tw: 1, night: 1, glowA: 0, sun: 1, env: 0.85, hemi: 0.2, haze: 0.00018, mist: 0.00008, cover: 0.36, cirrus: 0.22, cop: 0.8, cs: [0.012, 0.014, 0.020], ca: [0.032, 0.023, 0.019], aur: 0 },
  { el: -12, turb: 2.4, ray: 1.4, mie: 0.0030, g: 0.78, tw: 1, night: 1, glowA: 0.05, sun: 1, env: 0.85, hemi: 0.2, haze: 0.00018, mist: 0.00008, cover: 0.36, cirrus: 0.22, cop: 0.8, cs: [0.014, 0.015, 0.022], ca: [0.032, 0.024, 0.021], aur: 0 },
  { el: -6, turb: 2.6, ray: 1.6, mie: 0.0030, g: 0.78, tw: 1, night: 0.3, glowA: 0.55, sun: 1, env: 1.1, hemi: 0.26, haze: 0.00017, mist: 0.00010, cover: 0.34, cirrus: 0.26, cop: 0.8, cs: [0.09, 0.045, 0.055], ca: [0.026, 0.026, 0.038], aur: 0 },
  { el: -3, turb: 3.0, ray: 2.0, mie: 0.0035, g: 0.80, tw: 1, night: 0, glowA: 1.0, sun: 1, env: 1.2, hemi: 0.28, haze: 0.00017, mist: 0.00012, cover: 0.33, cirrus: 0.3, cop: 0.85, cs: [0.36, 0.13, 0.10], ca: [0.04, 0.036, 0.052], aur: 0 },
  { el: 0, turb: 3.2, ray: 2.3, mie: 0.0040, g: 0.80, tw: 0.55, night: 0, glowA: 0.9, sun: 1, env: 1.05, hemi: 0.26, haze: 0.00012, mist: 0.00013, cover: 0.32, cirrus: 0.32, cop: 0.85, cs: [0.65, 0.26, 0.13], ca: [0.065, 0.055, 0.065], aur: 0.5 },
  { el: 3, turb: 3.0, ray: 2.2, mie: 0.0038, g: 0.80, tw: 0.12, night: 0, glowA: 0.4, sun: 1, env: 0.95, hemi: 0.24, haze: 0.00012, mist: 0.00011, cover: 0.32, cirrus: 0.32, cop: 0.85, cs: [0.9, 0.45, 0.24], ca: [0.10, 0.095, 0.11], aur: 0.55 },
  { el: 9, turb: 2.3, ray: 1.75, mie: 0.0033, g: 0.80, tw: 0, night: 0, glowA: 0, sun: 1, env: 0.9, hemi: 0.22, haze: 0.00011, mist: 0.00008, cover: 0.30, cirrus: 0.3, cop: 0.85, cs: [1.05, 0.75, 0.5], ca: [0.15, 0.16, 0.19], aur: 0.5 },
  { el: 18, turb: 2.2, ray: 1.6, mie: 0.0031, g: 0.79, tw: 0, night: 0, glowA: 0, sun: 1, env: 0.88, hemi: 0.2, haze: 0.00010, mist: 0.00005, cover: 0.28, cirrus: 0.3, cop: 0.85, cs: [1.2, 1.08, 0.95], ca: [0.21, 0.23, 0.28], aur: 0.4 },
  { el: 32, turb: 2.1, ray: 1.5, mie: 0.0029, g: 0.78, tw: 0, night: 0, glowA: 0, sun: 1, env: 0.86, hemi: 0.18, haze: 0.00008, mist: 0.00004, cover: 0.26, cirrus: 0.28, cop: 0.85, cs: [1.3, 1.25, 1.18], ca: [0.25, 0.28, 0.33], aur: 0.3 },
  { el: 50, turb: 2.0, ray: 1.4, mie: 0.0027, g: 0.78, tw: 0, night: 0, glowA: 0, sun: 1, env: 0.85, hemi: 0.18, haze: 0.00008, mist: 0.00003, cover: 0.25, cirrus: 0.25, cop: 0.85, cs: [1.35, 1.32, 1.28], ca: [0.27, 0.3, 0.36], aur: 0.3 },
  // high sun (tropics / desert noon, up to the zenith): the 50 deg look held, with slightly clearer air (less haze
  // and mist, a touch less ambient) and brighter cloud tops; elevations below 50 deg are unaffected
  { el: 90, turb: 1.95, ray: 1.35, mie: 0.0026, g: 0.78, tw: 0, night: 0, glowA: 0, sun: 1, env: 0.83, hemi: 0.17, haze: 0.00007, mist: 0.000025, cover: 0.25, cirrus: 0.24, cop: 0.85, cs: [1.4, 1.37, 1.33], ca: [0.28, 0.31, 0.37], aur: 0.25 },
];

function lookAt(keys, elDeg) {
  let i = 0;
  while (i < keys.length - 2 && elDeg > keys[i + 1].el) i++;
  const a = keys[i], b = keys[i + 1];
  const t = clamp((elDeg - a.el) / (b.el - a.el), 0, 1);
  const s = t * t * (3 - 2 * t);
  const o = {};
  for (const k in a) {
    if (k === 'el') continue;
    o[k] = Array.isArray(a[k]) ? a[k].map((v, j) => v + (b[k][j] - v) * s) : a[k] + (b[k] - a[k]) * s;
  }
  return o;
}

/** Colour of direct sunlight after the atmosphere (Kasten-Young air mass, per-channel optical depth). */
function sunlightColor(el) {
  const e = Math.max(el / DEG, -1);
  const m = 1 / (Math.sin(Math.max(e, 0) * DEG) + 0.50572 * Math.pow(e + 6.07995, -1.6364));
  const tau = [0.045, 0.085, 0.19];
  const T = tau.map((t) => Math.exp(-m * t));
  const mx = Math.max(...T);
  return { color: new THREE.Color(T[0] / mx, T[1] / mx, T[2] / mx), trans: mx };
}

// ------------------------------------------------------------------ sky shader patch
const SKY_PARS = /* glsl */`
uniform float skyGain;
uniform float uDisc, uDiscCos;
uniform vec3 uGlowCol;
uniform float uTwilight, uNight, uAfterglow;
uniform vec3 uTwZen, uTwHor, uTwGlow, uNightZen, uNightHor, uPollution;
uniform float uClouds, uCover, uCirrus, uCloudOpacity, uCloudScale, uCloudH;
uniform vec2 uCloudOffset;
uniform sampler2D tCloud;
uniform vec3 uCloudSun, uCloudAmb;
uniform float uStars, uTime;
uniform vec3 uMoonDir, uMoonCol;
uniform vec3 uHorizonTint;
uniform float uSkySat;
float rsHash13( vec3 p ) { p = fract( p * 0.1031 ); p += dot( p, p.zyx + 31.32 ); return fract( ( p.x + p.y ) * p.z ); }
vec3 rsHash33( vec3 p ) { p = fract( p * vec3( 0.1031, 0.1030, 0.0973 ) ); p += dot( p, p.yxz + 33.33 ); return fract( ( p.xxy + p.yxx ) * p.zyx ); }
float rsCloud( vec2 uv ) { return texture2D( tCloud, uv ).r * 0.65 + texture2D( tCloud, uv * 2.9 + vec2( 0.37, 0.71 ) ).r * 0.35; }
`;

const SKY_MAIN = /* glsl */`
	vec3 col = retColor * skyGain;
	col = max( mix( vec3( dot( col, vec3( 0.2126, 0.7152, 0.0722 ) ) ), col, uSkySat ), 0.0 ); // Preetham is pale
	float rsCs = max( cosTheta, 0.0 );
	{ // Preetham's horizon is milky: cool it to a blue-grey away from the sun (the sun side stays warm)
		float toSun = dot( normalize( direction.xz + vec2( 1e-5 ) ), normalize( vSunDirection.xz + vec2( 1e-5 ) ) ) * 0.5 + 0.5;
		float hz = exp( -max( direction.y, 0.0 ) * 9.0 );
		col *= mix( vec3( 1.0 ), uHorizonTint, hz * ( 1.0 - toSun * toSun * toSun ) );
	}
	col += uGlowCol * uDisc * ( 0.35 * pow( rsCs, 900.0 ) + 0.05 * pow( rsCs, 48.0 ) );
	// blue hour / night: replaces the Preetham model once the sun is below the horizon
	if ( uTwilight > 0.0 ) {
		float h = max( direction.y, 0.0 );
		vec2 sd = normalize( vSunDirection.xz + vec2( 1e-5 ) );
		vec2 dd = normalize( direction.xz + vec2( 1e-5 ) );
		// clamped: normalize() can leave |dot| a hair above 1, and pow( negative, y ) is NaN (ANGLE/Metal), which
		// PMREM would smear over the whole environment map (black PBR surfaces at some twilight hours)
		float towards = clamp( dot( dd, sd ) * 0.5 + 0.5, 0.0, 1.0 );
		vec3 tw = mix( uTwHor, uTwZen, pow( h, 0.5 ) );
		tw += uTwGlow * uAfterglow * exp( -h * 7.0 ) * pow( towards, 2.5 );
		tw += uTwGlow * vec3( 0.25, 0.45, 0.9 ) * uAfterglow * 0.35 * exp( -abs( h - 0.1 ) * 12.0 ) * pow( 1.0 - towards, 2.0 ); // belt of Venus
		vec3 ng = mix( uNightHor, uNightZen, pow( h, 0.4 ) ) + uPollution * exp( -h * 6.0 );
		tw = mix( tw, ng, uNight );
		col = mix( col, tw, uTwilight );
	}
	// stars + moon (night)
	if ( uStars > 0.0 && direction.y > 0.0 ) {
		vec3 q = direction * 360.0;
		vec3 id = floor( q );
		float hs = rsHash13( id );
		if ( hs > 0.982 ) {
			vec3 o = rsHash33( id + 7.0 );
			float d = length( fract( q ) - ( 0.25 + 0.5 * o ) );
			float b = ( hs - 0.982 ) / 0.018; b = b * b * b;
			float tw = 0.7 + 0.3 * sin( uTime * ( 1.3 + 3.0 * o.x ) + hs * 91.0 );
			col += mix( vec3( 1.0, 0.85, 0.7 ), vec3( 0.75, 0.85, 1.0 ), o.y ) * smoothstep( 0.24, 0.0, d ) * ( 0.06 + 1.6 * b ) * tw * uStars * smoothstep( 0.04, 0.35, direction.y );
		}
	}
	if ( uStars > 0.0 ) {
		float cm = dot( direction, uMoonDir );
		if ( cm > 0.9995 ) {
			vec3 mx = normalize( cross( uMoonDir, vec3( 0.0, 1.0, 0.0 ) ) );
			vec3 my = cross( mx, uMoonDir );
			vec2 p = vec2( dot( direction, mx ), dot( direction, my ) ) / 0.0105;
			float disc = smoothstep( 1.0, 0.93, length( p ) );
			vec3 n = vec3( p, sqrt( max( 0.0, 1.0 - dot( p, p ) ) ) );
			float lit = smoothstep( -0.2, 0.3, dot( n, normalize( vec3( 0.6, 0.2, 0.75 ) ) ) );
			float maria = texture2D( tCloud, p * 0.16 + 0.37 ).b;
			col = mix( col, col * 0.4 + uMoonCol * ( 0.55 + 0.5 * maria ) * ( 0.06 + 0.94 * lit ), disc * uStars );
		}
		float cm2 = max( cm, 0.0 );
		col += uMoonCol * uStars * ( 0.006 * pow( cm2, 1500.0 ) + 0.0025 * pow( cm2, 90.0 ) );
	}
	// clouds: cumulus layer at uCloudH + streaky cirrus at 2.6x, lit by the sun colour, fading into the horizon
	if ( uClouds > 0.0 && direction.y > 0.0 ) {
		float dy = max( direction.y, 0.012 );
		float t = max( uCloudH - cameraPosition.y, 60.0 ) / dy;
		vec2 uv = ( cameraPosition.xz + direction.xz * t ) * uCloudScale + uCloudOffset;
		float n = rsCloud( uv ) + ( texture2D( tCloud, uv * 9.0 ).b - 0.5 ) * 0.12;
		float thr = 1.0 - uCover;
		float d = smoothstep( thr, thr + 0.2, n );
		vec2 sdir = normalize( vSunDirection.xz + vec2( 1e-4 ) ) * 0.012;
		float dS = smoothstep( thr, thr + 0.2, rsCloud( uv + sdir ) );
		float lit = clamp( 1.0 - 0.7 * dS + 0.2 * ( 1.0 - d ), 0.0, 1.0 );
		float fwd = pow( rsCs, 5.0 );
		vec3 cc = uCloudAmb * ( 0.7 + 0.3 * ( 1.0 - d ) ) + uCloudSun * lit * ( 0.45 + 1.5 * fwd * ( 1.0 - 0.6 * d ) );
		float t2 = max( uCloudH * 2.6 - cameraPosition.y, 60.0 ) / dy;
		vec2 w2 = cameraPosition.xz + direction.xz * t2;
		vec2 uv2 = vec2( w2.x * 0.8 + w2.y * 0.6, -w2.x * 0.6 + w2.y * 0.8 ) * uCloudScale * vec2( 0.45, 1.5 ) + uCloudOffset * 1.7;
		float ci = smoothstep( 0.5, 0.95, texture2D( tCloud, uv2 ).g * 0.8 + texture2D( tCloud, uv2 * 3.1 + 0.3 ).g * 0.2 );
		ci *= smoothstep( 0.35, 0.75, texture2D( tCloud, uv2 * 0.21 + vec2( 0.5, 0.2 ) ).r ) * uCirrus; // patchy, not uniform streaks
		vec3 cic = uCloudAmb * 0.9 + uCloudSun * ( 0.8 + 2.2 * fwd );
		float fade = smoothstep( 0.0, 0.05, direction.y );
		float a1 = d * uCloudOpacity * fade;
		float a2 = ci * 0.5 * fade;
		vec3 sky0 = col;
		col = mix( col, mix( cic, sky0, ( 1.0 - exp( -t2 * 0.00002 ) ) * 0.8 ), a2 );
		col = mix( col, mix( cc, sky0, 1.0 - exp( -t * 0.00004 ) ), a1 );
	}
	gl_FragColor = vec4( col, 1.0 );`;

function makeSkyUniforms(cloudTex) {
  return {
    skyGain: { value: 0.6 },
    uDisc: { value: 1 }, uDiscCos: { value: Math.cos(0.0075) },
    uGlowCol: { value: new THREE.Color(0, 0, 0) },
    uTwilight: { value: 0 }, uNight: { value: 0 }, uAfterglow: { value: 0 },
    uTwZen: { value: new THREE.Color(0.018, 0.040, 0.115) },
    uTwHor: { value: new THREE.Color(0.10, 0.11, 0.15) },
    uTwGlow: { value: new THREE.Color(0.75, 0.28, 0.07) },
    uNightZen: { value: new THREE.Color(0.0035, 0.0075, 0.024) },
    uNightHor: { value: new THREE.Color(0.012, 0.015, 0.026) },
    uPollution: { value: new THREE.Color(0.036, 0.027, 0.024) },
    uClouds: { value: 1 }, uCover: { value: 0.3 }, uCirrus: { value: 0.4 }, uCloudOpacity: { value: 0.85 },
    uCloudScale: { value: 1 / 14000 }, uCloudH: { value: 1700 },
    uCloudOffset: { value: new THREE.Vector2(0.13, 0.41) },
    tCloud: { value: cloudTex },
    uCloudSun: { value: new THREE.Color(1, 1, 1) }, uCloudAmb: { value: new THREE.Color(0.3, 0.3, 0.35) },
    uStars: { value: 0 }, uTime: { value: 0 },
    uHorizonTint: { value: new THREE.Color(0.8, 0.93, 1.15) }, uSkySat: { value: 1.3 },
    uMoonDir: { value: new THREE.Vector3(0, 1, 0) }, uMoonCol: { value: new THREE.Color(1.6, 1.55, 1.45) },
  };
}

function patchSky(sky, cloudTex) {
  const m = sky.material;
  Object.assign(m.uniforms, makeSkyUniforms(cloudTex));
  let f = m.fragmentShader;
  const need = ['uniform vec3 up;', 'float sundisk = smoothstep( sunAngularDiameterCos, sunAngularDiameterCos + 0.00002, cosTheta );', 'gl_FragColor = vec4( retColor, 1.0 );'];
  for (const n of need) if (!f.includes(n)) console.warn('[sky] Sky shader patch point missing:', n);
  f = f.replace('uniform vec3 up;', 'uniform vec3 up;\n' + SKY_PARS)
    .replace(need[1], 'float sundisk = uDisc * smoothstep( uDiscCos, uDiscCos + 0.00002, cosTheta );')
    .replace(need[2], SKY_MAIN);
  m.fragmentShader = f;
  m.needsUpdate = true;
  return sky;
}

/** Lens-flare elements: [texture key, size (px at 1300 px buffer height), distance along the axis, rgb, gain]. */
export const DEFAULT_FLARE = [
  ['glow', 380, 0.0, [1.0, 0.92, 0.8], 0.25],
  ['dot', 46, 0.32, [1.0, 0.75, 0.45], 0.16],
  ['hex', 74, 0.52, [0.55, 0.85, 1.0], 0.10],
  ['hex', 118, 0.7, [1.0, 0.7, 0.4], 0.075],
  ['ring', 300, 0.92, [1.0, 1.0, 1.0], 0.07],
  ['hex', 170, 1.12, [0.55, 1.0, 0.7], 0.055],
  ['dot', 90, 1.38, [0.75, 0.6, 1.0], 0.07],
];

const DEFAULT_SHADOWS = {
  cascades: 2,            // 2 = near + far cascade (needs the chunk patch), 1 = a single shadow map
  near: 45,               // near cascade half size (m)
  far: [260, 420, 650],   // far cascade half size (m) by camera height step
  farUp: [70, 200],       // camera heights that step the far box up ...
  farDown: [50, 160],     // ... and back down (hysteresis)
  single: 130,            // half size of the single map when cascades are unavailable / off
  nearMapSize: 2048, farMapSize: 4096, singleMapSize: 4096,
  nearRadius: 2.0, farRadius: 1.4,
  lead: 0.45, farLead: 0.55, // boxes lead the camera's view direction by this fraction of their half size
  farMaxY: 40,            // far box centre height cap (m)
  kernel: 'pcf',          // 'fast': 5 hardware-compare taps per lookup instead of three's 17 (setShadowKernel)
};

export class Atmosphere {
  /**
   * @param {object|THREE.Scene} o  options object ({ scene, renderer, ... }), or legacy (scene, renderer, opts)
   * @param {THREE.Scene} o.scene
   * @param {THREE.WebGLRenderer|null} o.renderer  (null: headless - lights, sun direction and darkness only; no
   *        environment map, sky cube, horizon sampling or lens flare)
   * @param {THREE.Camera} [o.camera]  cascades lead its view direction; lens flare
   * @param {number|string} [o.timeOfDay='golden']  hours 0..24 or a preset name (timePresets)
   * @param {object} [o.shadows]  { cascades: 2|1, near (m), far (m or [m by height step]), kernel: 'pcf'|'fast', ... }
   *        see DEFAULT_SHADOWS
   * @param {boolean} [o.build]  build immediately (default true for the options form, false for the legacy form)
   */
  constructor(o = {}, legacyRenderer, legacyOpts) {
    const legacy = !!(o && o.isScene);
    const opts = legacy ? { ...(legacyOpts || {}), scene: o, renderer: legacyRenderer } : o;
    this.scene = opts.scene; this.renderer = opts.renderer;
    this.timePresets = opts.timePresets || TIME_PRESETS;
    this.sunModel = { ...DEFAULT_SUN, ...(opts.sun || {}) };
    this.looks = opts.looks || DEFAULT_LOOKS;
    const sh = this.shadowOpts = { ...DEFAULT_SHADOWS, ...(opts.shadows || {}) };
    if (opts.shadowNearHalf != null) sh.near = opts.shadowNearHalf;
    this.cascaded = sh.cascades > 1 && CASCADES;
    this.shadowKernel = 'pcf'; // kernel in effect; build() selects sh.kernel through setShadowKernel
    const t = opts.timeOfDay ?? 'golden';
    this.hours = typeof t === 'number' ? t : this.timePresets[t] ?? this.timePresets.golden ?? TIME_PRESETS.golden;
    this.skyGain = opts.skyGain ?? 0.38;
    this.skyRadius = opts.skyRadius ?? 4000;
    this.sunIntensity = opts.sunIntensity ?? 3.9;
    this.moonIntensity = opts.moonIntensity ?? 0.3;
    this.moonColor = new THREE.Color(...(opts.moonColor || [0.62, 0.72, 1.0]));
    this.fogDensity = opts.fogDensity ?? [0.00030, 0.00015]; // scene fog: day density, extra at night
    this.flareSpec = opts.flare === false ? null : opts.flare || DEFAULT_FLARE;
    this.nearHalf = sh.near;                                   // near cascade half size (m), 2048 map
    this.farHalfSteps = Array.isArray(sh.far) ? sh.far.slice() : [sh.far]; // far cascade half size by camera height, 4096 map
    this.shadowHalf = this.nearHalf;
    this.sunDir = new THREE.Vector3(0, 1, 0);
    this.lightDir = new THREE.Vector3(0, 1, 0);
    const moon = opts.moon || { azimuth: 150, elevation: 34 }; // SE, 34 deg up
    this.moonDir = new THREE.Vector3();
    this.moonDir.set(Math.sin(moon.azimuth * DEG) * Math.cos(moon.elevation * DEG), Math.sin(moon.elevation * DEG), -Math.cos(moon.azimuth * DEG) * Math.cos(moon.elevation * DEG)).normalize();
    this.darkness = 0;
    this.windowGlow = 0;
    this.maxBuildingHeight = opts.maxCasterHeight ?? 200; // tallest shadow caster (m): shadow camera distance
    this.version = 0;
    this.fogMode = 'scene';
    this.camera = opts.camera || null;  // optional THREE.Camera: shadow cascades lead the view direction, lens flare
    this.sunColor = new THREE.Color(1, 1, 1);
    this.fogParams = { haze: 0.0002, hazeK: 1 / 700, hazeBase: 0, mist: 0.0001, mistK: 1 / 40, sunScatter: 0.0 };
    this.look = lookAt(this.looks, 10);
    this.cloudWind = new THREE.Vector2(...(opts.cloudWind || [6.5, 2.2])); // m/s at cloud height
    this._frame = 0;
    this._farHalf = this.farHalfSteps[0];
    this._farDirty = true;
    this._lastT = 0;
    this._clock = 0;
    this._fastDepth = null; this._fastMaps = null; this._onChildAdded = null; // 'fast' shadow kernel state
    this.flareScale = 1;
    if (opts.build ?? !legacy) this.build();
  }

  /** Alias of maxBuildingHeight (tallest shadow caster, m). */
  get maxCasterHeight() { return this.maxBuildingHeight; }
  set maxCasterHeight(v) { this.maxBuildingHeight = v; }

  build() {
    const { scene, renderer } = this;
    const sh = this.shadowOpts;
    this.cloudAtlas = makeCloudTexture(512);
    const ct = this.cloudAtlas.texture;
    this.sky = patchSky(new Sky(), ct);
    this.sky.scale.setScalar(this.skyRadius);
    this.sky.frustumCulled = false;
    this.sky.renderOrder = -1;
    this.sky.name = 'sky';
    this.sky.material.uniforms.uStars.value = 0;
    scene.add(this.sky);

    const maxTex = renderer?.capabilities?.maxTextureSize || 4096;
    // near cascade (sharp, around the camera) - this is also the single light when cascades are unavailable
    this.sun = new THREE.DirectionalLight(0xffffff, 3);
    this.sun.name = 'sun';
    this.sun.castShadow = true;
    const nearMap = this.cascaded ? Math.min(sh.nearMapSize, maxTex) : Math.min(sh.singleMapSize, maxTex);
    if (!this.cascaded) this.nearHalf = sh.single;
    this.sun.shadow.mapSize.set(nearMap, nearMap);
    this._setBox(this.sun, this.nearHalf);
    this.sun.shadow.radius = sh.nearRadius;
    scene.add(this.sun, this.sun.target);
    // far cascade: shadow-only (black) light that the patched lighting chunk reads for distant receivers
    if (this.cascaded) {
      this.sunFar = new THREE.DirectionalLight(0x000000, 0);
      this.sunFar.name = 'sunFarCascade';
      this.sunFar.castShadow = true;
      const fm = Math.min(sh.farMapSize, maxTex);
      this.sunFar.shadow.mapSize.set(fm, fm);
      this._setBox(this.sunFar, this._farHalf);
      this.sunFar.shadow.radius = sh.farRadius;
      this.sunFar.shadow.autoUpdate = false;
      this.sunFar.shadow.needsUpdate = true;
      scene.add(this.sunFar, this.sunFar.target);
    }

    this.hemi = new THREE.HemisphereLight(0xbfd4ff, 0x4a4036, 0.3);
    scene.add(this.hemi);

    this.fog = new THREE.FogExp2(0xc0c8d0, 0.00042);
    this.fogPost = new THREE.FogExp2(0x8090a0, 0);
    scene.fog = this.fogMode === 'scene' ? this.fog : this.fogPost;

    this.pmrem = renderer ? new THREE.PMREMGenerator(renderer) : null; // no renderer (headless): no environment map
    this.envScene = new THREE.Scene();
    this.envSky = patchSky(new Sky(), ct);
    this.envSky.scale.setScalar(500);
    this.envScene.add(this.envSky);
    // sky radiance cube for the post fog (no sun disc, no clouds, no stars)
    this.skyCubeRT = new THREE.WebGLCubeRenderTarget(32, { type: THREE.HalfFloatType, generateMipmaps: false, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter });
    this.skyCubeRT.texture.name = 'skyCube';
    this.cubeCam = new THREE.CubeCamera(0.1, 1000, this.skyCubeRT);
    this.envScene.add(this.cubeCam);

    if (this.flareSpec && renderer && typeof document !== 'undefined') this._buildFlare();
    if (sh.kernel && sh.kernel !== 'pcf') this.setShadowKernel(sh.kernel); // the world added later follows ('childadded')
    this.setTime(this.hours);
    return this;
  }

  get skyCube() { return this.skyCubeRT ? this.skyCubeRT.texture : null; }

  /** Colour distant geometry fades to: the scene FogExp2 colour (scene mode) or the linear horizon (post mode). */
  get fogColor() { return this.fogMode === 'scene' ? (this.fog ? this.fog.color : null) : (this.fogPost ? this.fogPost.color : null); }

  /**
   * 'scene': FogExp2 in every material (no post-processing). 'post': PostFX applies height fog from the depth
   * buffer; the scene keeps a zero-density FogExp2 whose colour is the linear horizon, so custom shaders that
   * read fogColor / fogDensity (e.g. a far-land haze) stay consistent without fogging twice.
   */
  setFogMode(mode) {
    this.fogMode = mode === 'post' ? 'post' : 'scene';
    if (!this.fogPost) this.fogPost = new THREE.FogExp2(0x8090a0, 0);
    if (this.horizon) this.fogPost.color.copy(this.horizon);
    if (this.scene) this.scene.fog = this.fogMode === 'scene' ? this.fog : this.fogPost;
  }

  /**
   * Sun shadow filter: 'pcf' (three's 17-tap PCF, the default) or 'fast' (5 hardware-compare taps on a depth texture
   * under each cascade map, shadowKernel.js). Switches every lit material under `root` (default: this scene) through
   * its ENGINE_SHADOW_FAST define, so it also works at runtime for A/B: the first switch compiles the other program of
   * each material, later switches reuse both. While 'fast' is on, objects added to the scene follow ('childadded');
   * a mesh added under an object already in the scene, or a material swapped onto a mesh, keeps its kernel until the
   * next call. 'fast' needs a renderer, PCFShadowMap and the sun (+ far cascade) as the scene's only shadow-casting
   * directional lights; otherwise it warns and selects 'pcf'. Returns the number of materials switched.
   */
  setShadowKernel(kernel, root = this.scene) {
    if (kernel !== 'pcf' && kernel !== 'fast') { console.warn(`[Atmosphere] unknown shadow kernel '${kernel}' ('pcf' | 'fast')`); return 0; }
    if (kernel === 'fast' && !this._fastShadowsOk()) kernel = 'pcf';
    this.shadowKernel = kernel;
    const s = this.scene;
    if (kernel === 'fast') {
      this._syncFastShadows(); // depth textures on the maps + the sampler array, before any material compiles 'fast'
      if (!this._onChildAdded && s) {
        this._onChildAdded = (e) => applyShadowKernel(e.child, this.shadowKernel);
        s.addEventListener('childadded', this._onChildAdded);
      }
    } else if (this._onChildAdded) {
      s.removeEventListener('childadded', this._onChildAdded);
      this._onChildAdded = null;
    }
    return applyShadowKernel(root, kernel);
  }

  _fastShadowsOk() {
    const r = this.renderer;
    let why = '';
    if (!r) why = 'no renderer';
    else if (!this.sun) why = 'not built yet';
    else if (r.shadowMap.type !== THREE.PCFShadowMap) why = 'it replaces PCFShadowMap only';
    else if (!installFastShadowChunk()) why = 'three.js shadow chunks changed';
    else {
      // the sampler array mirrors three's directional shadow order, which this class only knows for its own lights
      this.scene.traverse((o) => {
        if (o.isDirectionalLight && o.castShadow && o !== this.sun && o !== this.sunFar) why = `another shadow-casting directional light '${o.name}'`;
      });
    }
    if (why) console.warn(`[Atmosphere] fast shadow kernel unavailable (${why}), using 'pcf'`);
    return !why;
  }

  // While 'fast' (per frame from update): each shadow-casting cascade keeps its depth texture (three re-creates a map
  // without one after `shadow.map = null`) and the kernel's sampler array lists them in three's order, i.e. shadow
  // casters in scene order: sun, then far cascade. Allocation-free unless a map has to be re-created.
  _syncFastShadows() {
    if (this.renderer.shadowMap.type !== THREE.PCFShadowMap) return; // VSM / basic maps are three's alone
    if (!this._fastDepth) {
      const rev = this.renderer.state?.buffers?.depth?.getReversed?.() === true;
      this._fastDepth = [createShadowDepthTexture(rev), createShadowDepthTexture(rev)];
      this._fastDepth[0].name = 'sun.shadowDepth'; this._fastDepth[1].name = 'sunFarCascade.shadowDepth';
      this._fastMaps = [null, null];
    }
    let n = 0;
    for (let i = 0; i < 2; i++) {
      const light = i === 0 ? this.sun : this.sunFar;
      if (!light || !light.castShadow) continue;
      attachShadowDepth(light, this._fastDepth[i]);
      this._fastMaps[n++] = this._fastDepth[i];
    }
    setFastShadowMaps(this._fastMaps, n);
  }

  _setBox(light, half) {
    const c = light.shadow.camera;
    Object.assign(c, { left: -half, right: half, top: half, bottom: -half, near: 1, far: 2500 });
    c.updateProjectionMatrix();
    light.userData.half = half;
    const texel = (2 * half) / light.shadow.mapSize.x;
    // acne: normal offset ~1.5 texels; small constant depth bias (orthographic depth is linear over near..far)
    light.shadow.normalBias = texel * 1.4;
    light.shadow.bias = -0.00003;
  }

  _buildFlare() {
    const ft = makeFlareTextures();
    this._flareTex = ft;
    this.flare = new Lensflare();
    this.flare.frustumCulled = false;
    this.flareEls = this.flareSpec.map(([tex, size, dist, c, k]) => {
      const e = new LensflareElement(typeof tex === 'string' ? ft[tex] : tex, size, dist, new THREE.Color(c[0] * k, c[1] * k, c[2] * k));
      e.baseSize = size; e.base = new THREE.Color(c[0] * k, c[1] * k, c[2] * k);
      this.flare.addElement(e);
      return e;
    });
    this.flareScene = new THREE.Scene();
    this.flareScene.name = 'flare';
    this.flareScene.add(this.flare);
    this.flare.visible = false;
  }

  /** Lens flare overlay; call after the frame is on screen with the scene depth in the default depth buffer. */
  renderFlare(renderer, camera) {
    if (!this.flare || !this.flare.visible) return;
    const ac = renderer.autoClear;
    renderer.autoClear = false;
    renderer.setRenderTarget(null);
    renderer.render(this.flareScene, camera);
    renderer.autoClear = ac;
  }

  /** Flare element sizes are in drawing-buffer pixels: scale with the canvas height. */
  setFlareScale(drawingBufferHeight) {
    this.flareScale = clamp(drawingBufferHeight / 1300, 0.3, 3);
    for (const e of this.flareEls || []) e.size = e.baseSize * this.flareScale;
  }

  /** Hours 0..24 or a preset name (dawn, morning, noon, afternoon, golden, sunset, dusk, night). */
  setTimeOfDay(t) {
    const h = typeof t === 'number' ? t : this.timePresets[t];
    if (Number.isFinite(h)) this.setTime(h);
    return this;
  }
  get timeOfDay() { return this.hours; }

  setTime(hours) {
    this.hours = ((hours % 24) + 24) % 24;
    const { el, az } = sunPosition(this.hours, this.sunModel);
    this.sunEl = el;
    const elDeg = el / DEG;
    const d = new THREE.Vector3(Math.sin(az) * Math.cos(el), Math.sin(el), -Math.cos(az) * Math.cos(el)).normalize();
    this.sunDir.copy(d);
    this.darkness = smoothstep(6 * DEG, -9 * DEG, el);
    this.windowGlow = clamp(Math.pow(smoothstep(14 * DEG, -6 * DEG, el), 1.5), 0.02, 1);
    const L = this.look = lookAt(this.looks, elDeg);

    // light: sun by day, moon by night
    const day = smoothstep(-2 * DEG, 3 * DEG, el);
    const sl = sunlightColor(el);
    this.sunColor.copy(sl.color);
    const moonCol = this.moonColor;
    this.lightDir.copy(day > 0.02 ? d : this.moonDir);
    this.sun.color.copy(day > 0.02 ? sl.color : moonCol);
    this.sun.intensity = day > 0.02 ? this.sunIntensity * day * Math.pow(sl.trans, 0.8) * L.sun : this.moonIntensity;
    this._farDirty = true;

    // sky uniforms (both skies)
    const glow = this.sunColor.clone().multiplyScalar(L.aur * day);
    for (const s of [this.sky, this.envSky]) {
      const u = s.material.uniforms;
      u.turbidity.value = L.turb;
      u.rayleigh.value = L.ray;
      u.mieCoefficient.value = L.mie;
      u.mieDirectionalG.value = L.g;
      u.sunPosition.value.copy(d);
      u.skyGain.value = this.skyGain;
      u.uGlowCol.value.copy(glow);
      u.uTwilight.value = L.tw;
      u.uNight.value = L.night;
      u.uAfterglow.value = L.glowA;
      u.uCover.value = L.cover;
      u.uCirrus.value = L.cirrus;
      u.uCloudOpacity.value = L.cop;
      u.uCloudSun.value.setRGB(L.cs[0], L.cs[1], L.cs[2]);
      u.uCloudAmb.value.setRGB(L.ca[0], L.ca[1], L.ca[2]);
      u.uMoonDir.value.copy(this.moonDir);
      u.uHorizonTint.value.setRGB(0.8, 0.93, 1.15).lerp(_white, L.tw);
    }
    this.sky.material.uniforms.uStars.value = L.night;
    this.envSky.material.uniforms.uStars.value = 0;

    // environment from the sky (with clouds, without stars)
    if (this.pmrem) {
      if (this.envRT) this.envRT.dispose();
      this.envRT = this.pmrem.fromScene(this.envScene, 0, 0.1, 1000);
      this.scene.environment = this.envRT.texture;
      this.scene.environmentIntensity = L.env;
    }

    // clean sky radiance (no disc, no clouds) for fog colour + horizon sampling
    const eu = this.envSky.material.uniforms;
    eu.uDisc.value = 0; eu.uClouds.value = 0;
    if (this.renderer) this.cubeCam.update(this.renderer, this.envScene);
    const hor = this.sampleHorizon();
    eu.uDisc.value = 1; eu.uClouds.value = 1;
    this.horizon = hor;

    // scene fog (no-post path): tone-mapped horizon colour (three applies fog after tone mapping)
    const exp = (this.renderer && this.renderer.toneMappingExposure) ?? 1;
    const tm = this.renderer && this.renderer.toneMapping === THREE.NoToneMapping ? new THREE.Color(clamp(hor.r * exp, 0, 1), clamp(hor.g * exp, 0, 1), clamp(hor.b * exp, 0, 1)) : acesFilmic(hor, exp);
    this.fog.color.copy(tm);
    this.fog.density = this.fogDensity[0] + this.fogDensity[1] * this.darkness;
    this.fogPost.color.copy(hor);
    // post fog parameters
    const fp = this.fogParams;
    fp.haze = L.haze; fp.mist = L.mist;
    fp.sunScatter = 0.05 * day * (1 - smoothstep(25 * DEG, 45 * DEG, el));

    // hemisphere: sky dome colour from the horizon, ground = warm bounce of the sun (city lights at night)
    const hn = hor.clone().multiplyScalar(1 / Math.max(0.03, Math.max(hor.r, hor.g, hor.b)));
    this.hemi.color.copy(hn).lerp(new THREE.Color(0.5, 0.66, 1.0), 0.5 + 0.3 * this.darkness); // shade reads cool/blue
    const bounce = new THREE.Color(0.42, 0.37, 0.31).lerp(this.sunColor, 0.45 * day);
    this.hemi.groundColor.copy(bounce).lerp(new THREE.Color(0.55, 0.36, 0.2), this.darkness * 0.8);
    this.hemi.intensity = L.hemi;

    // lens flare tint
    if (this.flareEls) this.flareEls.forEach((e, i) => e.color.copy(e.base).multiply(i === 0 ? this.sunColor : _white));
    this.version++;
  }

  sampleHorizon() {
    const r = this.renderer;
    const out = new THREE.Color(0.6, 0.65, 0.7);
    if (!r) return out; // headless: a neutral horizon
    try {
      const S = 16;
      const rt = new THREE.WebGLRenderTarget(S, S, { type: THREE.FloatType, depthBuffer: false });
      const cam = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
      const buf = new Float32Array(S * S * 4);
      const prevRT = r.getRenderTarget();
      let sr = 0, sg = 0, sb = 0, n = 0;
      for (let k = 0; k < 8; k++) {
        const a = (k / 8) * Math.PI * 2;
        cam.position.set(0, 0, 0);
        cam.lookAt(Math.cos(a), -0.02, Math.sin(a));
        r.setRenderTarget(rt);
        r.render(this.envScene, cam);
        r.readRenderTargetPixels(rt, 0, 0, S, S, buf);
        for (let i = 0; i < S * S; i++) {
          const R = buf[i * 4], G = buf[i * 4 + 1], B = buf[i * 4 + 2];
          if (isFinite(R) && isFinite(G) && isFinite(B)) { sr += R; sg += G; sb += B; n++; }
        }
      }
      r.setRenderTarget(prevRT);
      rt.dispose();
      if (n) out.setRGB(sr / n, sg / n, sb / n);
    } catch (e) {
      console.warn('[World] horizon sampling failed', e);
    }
    return out;
  }

  /** Cloud density (0..1) seen in a direction from a position: mirrors the sky shader (used to dim the flare). */
  cloudDensityToward(pos, dir) {
    if (!this.cloudAtlas || dir.y <= 0.01) return 0;
    const u = this.sky.material.uniforms;
    const t = Math.max(u.uCloudH.value - pos.y, 60) / Math.max(dir.y, 0.012);
    const s = u.uCloudScale.value, o = u.uCloudOffset.value;
    const x = (pos.x + dir.x * t) * s + o.x, y = (pos.z + dir.z * t) * s + o.y;
    // texture v runs with +z (DataTexture rows), matching the shader's uv.y = world z
    const n = sampleCloud(this.cloudAtlas, x, y) * 0.65 + sampleCloud(this.cloudAtlas, x * 2.9 + 0.37, y * 2.9 + 0.71) * 0.35;
    const thr = 1 - u.uCover.value;
    return smoothstep(thr, thr + 0.2, n) * u.uCloudOpacity.value * smoothstep(0, 0.05, dir.y);
  }

  /**
   * Per frame. cam: camera WORLD POSITION (Vector3). dt (s): optional; when omitted the wall clock is used.
   * Allocation-free.
   */
  update(cam, dt) {
    if (!cam) return;
    let now;
    if (Number.isFinite(dt)) {
      dt = Math.min(0.1, Math.max(0, dt));
      this._clock += dt;
      now = this._clock * 1000;
    } else {
      now = performance.now();
      dt = this._lastT ? Math.min(0.1, (now - this._lastT) / 1000) : 0;
      this._lastT = now;
    }
    this.sky.position.copy(cam);
    const su = this.sky.material.uniforms;
    su.uTime.value = (now / 1000) % 3600;
    su.uCloudOffset.value.addScaledVector(this.cloudWind, dt * su.uCloudScale.value);

    // horizontal view direction: cascades lead the camera
    let fx = 0, fz = 0;
    if (this.camera) {
      this.camera.getWorldDirection(_v);
      const l = Math.hypot(_v.x, _v.z);
      if (l > 1e-3) { fx = _v.x / l; fz = _v.z / l; }
    }
    const sh = this.shadowOpts;
    const nh = this.nearHalf;
    this._fitCascade(this.sun, cam.x + fx * nh * sh.lead, cam.y, cam.z + fz * nh * sh.lead, nh);

    this._frame++;
    if (this.sunFar) {
      // far cascade box grows with altitude (quantised with hysteresis so texels stay stable)
      const steps = this.farHalfSteps;
      let idx = steps.indexOf(this._farHalf);
      const up = sh.farUp, down = sh.farDown;
      if (idx < steps.length - 1 && cam.y > up[idx]) idx++;
      else if (idx > 0 && cam.y < down[idx - 1]) idx--;
      if (steps[idx] !== this._farHalf) { this._farHalf = steps[idx]; this._setBox(this.sunFar, this._farHalf); this._farDirty = true; }
      // re-render the far map every other frame (it is consistent with its own matrix in between)
      if (this._farDirty || (this._frame & 1) === 0) {
        const fh = this._farHalf, off = fh * sh.farLead;
        this._fitCascade(this.sunFar, cam.x + fx * off, Math.min(cam.y, sh.farMaxY), cam.z + fz * off, fh);
        this.sunFar.shadow.needsUpdate = true;
        this._farDirty = false;
      }
    }
    if (this.shadowKernel === 'fast') this._syncFastShadows();
    this._updateFlare(cam);
  }

  _fitCascade(light, x, y, z, half) {
    // shadow box follows the focus point, snapped to shadow texels to avoid shimmering
    const L = this.lightDir;
    const texel = (2 * half) / light.shadow.mapSize.x;
    const fwd = _f.copy(L).negate();
    const right = _r.crossVectors(fwd, Math.abs(fwd.y) > 0.99 ? _x : _up).normalize();
    const up = _u.crossVectors(right, fwd).normalize();
    let px = x * right.x + y * right.y + z * right.z;
    let py = x * up.x + y * up.y + z * up.z;
    const pz = x * fwd.x + y * fwd.y + z * fwd.z;
    px = Math.round(px / texel) * texel; py = Math.round(py / texel) * texel;
    const c = _c.set(0, 0, 0).addScaledVector(right, px).addScaledVector(up, py).addScaledVector(fwd, pz);
    // distance toward the light so tall casters far up-sun still cast into the box
    const el = Math.max(0.05, Math.asin(clamp(L.y, -1, 1)));
    const dist = clamp(this.maxBuildingHeight / Math.tan(el) + 200, 400, 2200);
    light.target.position.copy(c);
    light.position.copy(c).addScaledVector(L, dist);
    const sc = light.shadow.camera;
    const far = dist + 500 + half;
    if (Math.abs(sc.far - far) > 50) { sc.far = far; sc.updateProjectionMatrix(); }
    light.target.updateMatrixWorld();
    light.updateMatrixWorld();
  }

  _updateFlare(cam) {
    if (!this.flare) return;
    const day = smoothstep(-0.8 * DEG, 1.5 * DEG, this.sunEl);
    if (day <= 0.001) { this.flare.visible = false; return; }
    this.flare.visible = true;
    this.flare.position.copy(cam).addScaledVector(this.sunDir, 5000);
    this.flare.updateMatrixWorld();
    const k = day * (1 - 0.92 * this.cloudDensityToward(cam, this.sunDir));
    const els = this.flareEls;
    for (let i = 0; i < els.length; i++) {
      const e = els[i];
      e.color.copy(e.base).multiplyScalar(k);
      if (i === 0) e.color.multiply(this.sunColor);
    }
  }

  dispose() {
    const s = this.scene;
    for (const o of [this.sky, this.sun, this.sun && this.sun.target, this.sunFar, this.sunFar && this.sunFar.target, this.hemi]) if (o && s) s.remove(o);
    if (s && (s.fog === this.fog || s.fog === this.fogPost)) s.fog = null;
    if (s && this._onChildAdded) { s.removeEventListener('childadded', this._onChildAdded); this._onChildAdded = null; } // materials keep their define
    if (s && this.envRT && s.environment === this.envRT.texture) s.environment = null;
    for (const o of [this.sky, this.envSky]) if (o) o.material.dispose();
    for (const l of [this.sun, this.sunFar]) if (l) l.dispose();
    if (this.envRT) this.envRT.dispose();
    if (this.skyCubeRT) this.skyCubeRT.dispose();
    if (this.pmrem) this.pmrem.dispose();
    if (this.cloudAtlas) this.cloudAtlas.texture.dispose();
    if (this._flareTex) for (const t of Object.values(this._flareTex)) t.dispose();
  }
}
const _f = new THREE.Vector3(), _r = new THREE.Vector3(), _u = new THREE.Vector3(), _c = new THREE.Vector3(), _v = new THREE.Vector3();
const _up = new THREE.Vector3(0, 1, 0), _x = new THREE.Vector3(1, 0, 0);
const _white = new THREE.Color(1, 1, 1);
