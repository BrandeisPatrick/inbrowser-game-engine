// Consumer check of the emitted declarations (npm run types:check): every subpath resolves through `paths` the way
// the package `exports` map it, the public constructors accept their documented options, and the .d.ts files
// themselves pass a strict compile (skipLibCheck off). Type-level only: nothing here runs.
import { Engine, PostFX, DynamicsWorld, Cloth, CharacterController, Atmosphere } from '@vibe/engine';
import { PhysicsWorld, FracturedBody, boxShape, Buoyancy, Rope, SoftBody, ParticleCollider } from '@vibe/engine/physics';
import { Events, Clock, createBenchHarness, Assets } from '@vibe/engine/core';
import { ClusteredLights, GBuffer, GpuParticleSystem } from '@vibe/engine/render';
import { AnimationController, Ragdoll, TwoBoneIK } from '@vibe/engine/anim';
import { InputMap } from '@vibe/engine/input';
import { ThirdPersonRig } from '@vibe/engine/camera';
import { AudioBus } from '@vibe/engine/audio';
import { Minimap } from '@vibe/engine/ui';
import { MapWorld } from '@vibe/engine/world';

const engine = new Engine({ container: document.body });
const physics = new PhysicsWorld({});
const dyn = new DynamicsWorld({ physics, maxParticles: 1024, substeps: 4 });
const id: number = dyn.addBody({ shape: boxShape(0.5, 0.5, 0.5), position: [0, 2, 0], mass: 10 });
dyn.setVelocity(id, 0, 0, 0, 0, 0, 0);
dyn.step(1 / 120);
Cloth.grid(dyn, { nx: 8, ny: 8, spacing: 0.1, position: [0, 1, 0], pins: ['top-left'] });
Rope.chain(dyn, { from: [0, 2, 0], to: [0, 0, 0], segments: 10 });
SoftBody.lattice(dyn, { shape: 'box', size: [0.5, 0.5, 0.5], resolution: 3, position: [1, 1, 0], mass: 5 });
dyn.addParticleSolver(new ParticleCollider(dyn, {}));
dyn.addParticleSolver(new Buoyancy(dyn, { fluid: { heightAt: (x: number, z: number, t: number) => 1 + 0.1 * Math.sin(x + z + t), density: 1000 } }));
const fb = new FracturedBody(dyn, { shape: boxShape(1, 1, 0.2), position: [0, 1, 0], pieces: 10, seed: 1, density: 600 });
fb.update();
const cc = new CharacterController({ radius: 0.35, height: 1.8 });
cc.attachDynamics(dyn, { pushMass: 120 });
const ev = new Events(); ev.on('x', () => {}); ev.emit('x');
const clock = new Clock(); clock.timeScale = 1;
const assets = new Assets({ base: '/' }); void assets;
const atmo: Atmosphere | null = null;
const fx = new PostFX(engine.renderer, engine.scene, engine.camera, { atmosphere: atmo, gbuffer: true, aa: 'taa', ssr: true });
fx.set('ssr.steps', 16);
const lights = new ClusteredLights({ maxLights: 256 });
lights.update(engine.camera);
const rig = new ThirdPersonRig(engine.camera, null, { distance: 4 });
rig.update(0.016);
const harness = createBenchHarness(engine, { name: 'x', poses: {}, settings: {} });
void harness; void GBuffer; void GpuParticleSystem; void AnimationController; void Ragdoll; void TwoBoneIK; void InputMap; void AudioBus; void Minimap; void MapWorld;
