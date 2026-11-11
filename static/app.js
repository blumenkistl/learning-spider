// The world: a z-up three.js scene whose camera orbits the spider horizontally. Everything
// drawn comes from the server's latest moment; the page does no kinematics and no physics.

import * as THREE from 'three';
import { OrbitControls } from '/static/vendor/three/jsm/controls/OrbitControls.js';

const BODY = [0.15, 0.10, 0.06], HEAD_RADIUS = 0.06, HEAD_X = 0.19;
const HIP_X = [0.12, 0.04, -0.04, -0.12], HIP_Y = 0.10;
const FEMUR_R = 0.014, TIBIA_R = 0.010;

// --- scene ---------------------------------------------------------------------------------
const canvas = document.getElementById('world');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0e1116);
scene.fog = new THREE.Fog(0x0e1116, 4, 14);

const camera = new THREE.PerspectiveCamera(45, 1, 0.05, 50);
camera.up.set(0, 0, 1);                               // z is up, before the controls read it
camera.position.set(-1.2, -1.0, 0.75);
const controls = new OrbitControls(camera, canvas);
controls.enablePan = false;
controls.enableDamping = true;
controls.minPolarAngle = 1.15;                        // horizontal rotation only
controls.maxPolarAngle = 1.15;
controls.minDistance = 0.6;
controls.maxDistance = 4;

scene.add(new THREE.HemisphereLight(0xd9e6ff, 0x202830, 0.9));
const sun = new THREE.DirectionalLight(0xffffff, 1.4);
sun.position.set(2, -3, 5);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.camera.left = sun.shadow.camera.bottom = -3;
sun.shadow.camera.right = sun.shadow.camera.top = 3;
scene.add(sun);

const ground = new THREE.Mesh(new THREE.CircleGeometry(5.5, 96), new THREE.MeshStandardMaterial({ color: 0x1b2230, roughness: 1 }));
ground.receiveShadow = true;
scene.add(ground);
const grid = new THREE.GridHelper(10, 40, 0x3a4458, 0x262e3c);
grid.rotation.x = Math.PI / 2;                        // GridHelper is y-up by default
grid.position.z = 0.001;
scene.add(grid);
const rim = new THREE.Mesh(new THREE.RingGeometry(4.98, 5.02, 128), new THREE.MeshBasicMaterial({ color: 0x5ad1ff, side: THREE.DoubleSide }));
rim.position.z = 0.002;
scene.add(rim);

// --- the spider ----------------------------------------------------------------------------
const spider = new THREE.Group();
scene.add(spider);
const skin = new THREE.MeshStandardMaterial({ color: 0x8a6a3c, roughness: 0.7 });
const legSkin = new THREE.MeshStandardMaterial({ color: 0x5a4328, roughness: 0.8 });
const body = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 24), skin);
body.scale.set(...BODY);
body.castShadow = true;
spider.add(body);
const head = new THREE.Mesh(new THREE.SphereGeometry(HEAD_RADIUS, 24, 18), skin);
head.position.set(HEAD_X, 0, 0.005);
head.castShadow = true;
spider.add(head);
for (const s of [-1, 1]) {                           // eyes
  const eye = new THREE.Mesh(new THREE.SphereGeometry(0.012, 10, 8), new THREE.MeshStandardMaterial({ color: 0x101010 }));
  eye.position.set(HEAD_X + 0.05, s * 0.025, 0.025);
  spider.add(eye);
}
const unit = new THREE.CylinderGeometry(1, 1, 1, 10);
const legs = [];
for (let i = 0; i < 8; i++) {
  const femur = new THREE.Mesh(unit, legSkin), tibia = new THREE.Mesh(unit, legSkin);
  femur.castShadow = tibia.castShadow = true;
  const knee = new THREE.Mesh(new THREE.SphereGeometry(FEMUR_R, 10, 8), legSkin);
  const foot = new THREE.Mesh(new THREE.CircleGeometry(0.02, 16), new THREE.MeshBasicMaterial({ color: 0x5ad1ff, transparent: true, opacity: 0.8 }));
  spider.add(femur, tibia, knee);
  scene.add(foot);                                   // planted discs lie in the world, on the ground
  foot.visible = false;
  legs.push({ femur, tibia, knee, foot, hip: new THREE.Vector3(HIP_X[i % 4], (i < 4 ? 1 : -1) * HIP_Y, 0) });
}

const _a = new THREE.Vector3(), _b = new THREE.Vector3(), _d = new THREE.Vector3(), _yUp = new THREE.Vector3(0, 1, 0);
function placeCylinder(mesh, a, b, radius) {
  _d.subVectors(b, a);
  const length = _d.length();
  mesh.position.copy(a).addScaledVector(_d, 0.5);
  mesh.scale.set(radius, length, radius);
  mesh.quaternion.setFromUnitVectors(_yUp, _d.normalize());
}

const target = new THREE.Vector3();
let firstFrame = true;
function draw(pose) {
  const z = pose.z;
  spider.position.set(pose.x, pose.y, z);
  spider.rotation.set(0, 0, pose.heading);
  const c = Math.cos(pose.heading), s = Math.sin(pose.heading);
  for (let i = 0; i < 8; i++) {
    const [kx, ky, kz, fx, fy, fz] = pose.legs[i];
    const leg = legs[i];
    _a.set(kx, ky, kz);                              // body frame: origin at the body centre
    _b.set(fx, fy, fz);
    placeCylinder(leg.femur, leg.hip, _a, FEMUR_R);
    placeCylinder(leg.tibia, _a, _b, TIBIA_R);
    leg.knee.position.copy(_a);
    leg.foot.visible = pose.planted[i];
    leg.foot.position.set(pose.x + c * fx - s * fy, pose.y + s * fx + c * fy, 0.003);
  }
  const next = new THREE.Vector3(pose.x, pose.y, 0.08);
  if (firstFrame || next.distanceTo(target) > 1.0) {   // a recentre: jump with the spider
    camera.position.sub(target).add(next);
    target.copy(next);
    firstFrame = false;
  } else {
    const before = target.clone();
    target.lerp(next, 0.15);
    camera.position.add(target).sub(before);
  }
  controls.target.copy(target);
}

function resize() {
  const w = window.innerWidth, h = window.innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
resize();
(function frame() {
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
})();

// --- HUD -----------------------------------------------------------------------------------
const $ = id => document.getElementById(id);
const fmt = (v, d = 3) => (v === null || v === undefined) ? '—' : Number(v).toFixed(d);
let described = null;
fetch('/api/describe').then(r => r.json()).then(d => {
  described = d;
  const b = d.brain;
  $('brainline').textContent = b
    ? `${b.neurons} neurons · ${b.parameters} parameters · ${b.placement.backend}${b.placement.device ? ':' + b.placement.device : ''} ${b.placement.precision} · seed ${d.seed}`
    : `control: ${d.control && d.control.policy}`;
}).catch(() => {});

function update(m) {
  draw(m.pose);
  $('policy').textContent = m.policy;
  $('moments').textContent = m.moments;
  const mode = $('mode');
  mode.textContent = m.refused ? 'refused' : m.mode;
  mode.className = m.refused ? 'refused' : m.mode;
  $('forward').textContent = fmt(m.distance_forward) + ' m';
  $('net').textContent = fmt(m.distance_net) + ' m' + (m.recentres ? ` · recentred ${m.recentres}×` : '');
  $('speed').textContent = fmt(m.speed) + ' m/s';
  $('reward').textContent = fmt(m.reward, 3) + (m.owed !== null && m.owed !== undefined ? ` (owed ${fmt(m.owed, 3)})` : '');
  $('arousal').textContent = `${fmt(m.level, 2)} / ${fmt(m.want, 2)}`;
  $('sweeps').textContent = `${m.sweeps} (${m.learning_sweeps} learning)`;
  $('ms').textContent = fmt(m.ms, 1);
  $('pace').textContent = m.pace > 0 ? `${m.pace}/s` : 'max';
  $('stance').textContent = m.pose.planted.filter(Boolean).length + ' / 8';
  $('turned').textContent = `${fmt(m.turned, 1)} rad (${fmt(m.omega, 2)} rad/s)`;
  $('refusals').textContent = m.refusals;
  $('pause').textContent = m.paused ? 'Resume' : 'Pause';
  $('pause').dataset.op = m.paused ? 'resume' : 'pause';
  for (const b of document.querySelectorAll('button[data-op=policy]')) b.classList.toggle('on', b.dataset.value === m.policy);
  for (const b of document.querySelectorAll('button[data-op=pace]')) b.classList.toggle('on', Number(b.dataset.value) === m.pace);
  const blocks = m.blocks.slice(-4);
  $('blocks').textContent = blocks.length
    ? 'last blocks (100 moments): ' + blocks.map(b => `${b.forward_m >= 0 ? '+' : ''}${b.forward_m.toFixed(3)} m, ${b.reward_per_moment.toFixed(3)} r`).join(' · ')
    : '';
  $('notes').textContent = m.notes.join('\n');
  strip(m);
}

// --- the brain strip: sensory, association and motor activity of the moment ----------------
const brainCanvas = $('brain'), ctx = brainCanvas.getContext('2d');
function strip(m) {
  const W = brainCanvas.width, H = brainCanvas.height;
  ctx.clearRect(0, 0, W, H);
  const a = m.activity;
  if (!a) {
    ctx.fillStyle = '#8b94a7';
    ctx.font = '11px system-ui';
    ctx.fillText(m.policy === 'random' ? 'random policy: no brain' : 'no settled state', 8, 20);
    return;
  }
  row(a.sensory, 4, 26, W - 8, 1);
  row(a.association, 36, 20, W - 8, 1);
  const motor = a.motor, cell = (W - 8) / 16, h = 24;
  for (let j = 0; j < 16; j++) {
    for (let k = 0; k < 3; k++) {
      const v = motor[3 * j + k];
      ctx.fillStyle = heat(v);
      ctx.fillRect(4 + j * cell, 64 + k * (h / 3), cell - 1, h / 3 - 1);
    }
    if (m.commands) {
      ctx.strokeStyle = '#5ad1ff';
      ctx.strokeRect(4 + j * cell + 0.5, 64 + m.commands[j] * (h / 3) + 0.5, cell - 2, h / 3 - 2);
    }
  }
  ctx.fillStyle = '#8b94a7';
  ctx.font = '9px system-ui';
  ctx.fillText('sensory', 6, 11); ctx.fillText('association', 6, 45); ctx.fillText('motor ↓ hold ↑ per joint', 6, 91);
}
function row(values, y, h, w, pad) {
  const cell = w / values.length;
  for (let i = 0; i < values.length; i++) {
    ctx.fillStyle = heat(values[i]);
    ctx.fillRect(4 + i * cell, y + 8, Math.max(1, cell - pad), h - 8);
  }
}
function heat(v) {
  const t = Math.max(0, Math.min(1, v));
  return `rgb(${Math.round(20 + 235 * t)}, ${Math.round(40 + 160 * t)}, ${Math.round(70 + 40 * (1 - t))})`;
}

// --- controls and the socket ---------------------------------------------------------------
document.querySelectorAll('button[data-op]').forEach(b => b.addEventListener('click', async () => {
  const body = { op: b.dataset.op };
  if (b.dataset.value !== undefined) body.value = isNaN(b.dataset.value) ? b.dataset.value : Number(b.dataset.value);
  const r = await fetch('/api/control', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!r.ok) $('notes').textContent = await r.text();
}));

function connect() {
  const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  ws.onopen = () => { $('link').textContent = 'live'; };
  ws.onmessage = e => update(JSON.parse(e.data));
  ws.onclose = () => { $('link').textContent = 'reconnecting…'; setTimeout(connect, 1000); };
  ws.onerror = () => ws.close();
}
connect();
