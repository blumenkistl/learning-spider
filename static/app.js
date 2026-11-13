// The world: a z-up three.js scene whose camera orbits the spider horizontally. Everything
// drawn comes from the server's latest moment; the page does no kinematics and no physics.

import * as THREE from 'three';
import { OrbitControls } from '/static/vendor/three/jsm/controls/OrbitControls.js';
import { BrainView } from '/static/vendor/brain_view.js';

const PROSOMA = [0.15, 0.10, 0.06], PROSOMA_X = 0.01;       // the front part, which carries the legs
const ABDOMEN = [0.14, 0.115, 0.10], ABDOMEN_X = -0.26, ABDOMEN_Z = 0.03;
const FEMUR_R = 0.014, TIBIA_R = 0.011, COXA_R = 0.017;
const ROOT_Y = 0.03;                                  // where a leg leaves the body: well inside the ellipsoid

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

const GROUND_COLOUR = '#1b2230';                      // the page's own; the view panel changes it, for this browser only
const ground = new THREE.Mesh(new THREE.CircleGeometry(5.5, 96), new THREE.MeshStandardMaterial({ color: GROUND_COLOUR, roughness: 1 }));
ground.receiveShadow = true;
scene.add(ground);
const GRID_COLOUR = '#262e3c';                        // the page's own; the two axes through the origin a shade lighter
let grid = null;
function makeGrid(hex) {                              // GridHelper bakes its colours into the geometry: recolouring is rebuilding
  if (grid) { scene.remove(grid); grid.geometry.dispose(); grid.material.dispose(); }
  const axes = new THREE.Color(hex).lerp(new THREE.Color('#ffffff'), 0.12);
  grid = new THREE.GridHelper(10, 40, axes, hex);
  grid.rotation.x = Math.PI / 2;                      // GridHelper is y-up by default
  grid.position.z = 0.001;
  scene.add(grid);
}
makeGrid(GRID_COLOUR);
const rim = new THREE.Mesh(new THREE.RingGeometry(4.98, 5.02, 128), new THREE.MeshBasicMaterial({ color: 0x5ad1ff, side: THREE.DoubleSide }));
rim.position.z = 0.002;
scene.add(rim);

const _a = new THREE.Vector3(), _b = new THREE.Vector3(), _d = new THREE.Vector3(), _yUp = new THREE.Vector3(0, 1, 0);
function placeCylinder(mesh, a, b, radius) {
  _d.subVectors(b, a);
  const length = _d.length();
  mesh.position.copy(a).addScaledVector(_d, 0.5);
  mesh.scale.set(radius, length, radius);
  mesh.quaternion.setFromUnitVectors(_yUp, _d.normalize());
}

// --- the spider ----------------------------------------------------------------------------
const spider = new THREE.Group();
scene.add(spider);
const skin = new THREE.MeshStandardMaterial({ color: 0x1b2744, roughness: 0.75, metalness: 0.1 });   // cobalt-blue tarantula
const legSkin = new THREE.MeshStandardMaterial({ color: 0x17357a, roughness: 0.55, metalness: 0.25 });
const hairSkin = new THREE.LineBasicMaterial({ color: 0x05070c, transparent: true, opacity: 0.8 });
let hairSeed = 7;
const hairRand = () => (hairSeed = (hairSeed * 1664525 + 1013904223) % 4294967296) / 4294967296;
// Bristles in a mesh's own unit space: they inherit its scale, so a scaled cylinder or sphere stays hairy.
function hairs(count, onSphere, lengthMin, lengthMax) {
  const pts = [];
  for (let i = 0; i < count; i++) {
    const len = lengthMin + (lengthMax - lengthMin) * hairRand();
    if (onSphere) {
      const u = 2 * hairRand() - 1, t = 2 * Math.PI * hairRand(), r = Math.sqrt(1 - u * u);
      const x = r * Math.cos(t), y = u, z = r * Math.sin(t);
      pts.push(x, y, z, x * (1 + len), y * (1 + len), z * (1 + len));
    } else {
      const t = 2 * Math.PI * hairRand(), y = hairRand() - 0.5, tilt = 0.06 * (hairRand() - 0.5);
      const x = Math.cos(t), z = Math.sin(t);
      pts.push(x, y, z, x * (1 + len), y + tilt, z * (1 + len));
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
  return new THREE.LineSegments(g, hairSkin);
}
const prosoma = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 24), skin);
prosoma.scale.set(...PROSOMA);
prosoma.position.x = PROSOMA_X;
prosoma.castShadow = true;
prosoma.add(hairs(500, true, 0.08, 0.18));
spider.add(prosoma);
const abdomen = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 24), skin);
abdomen.scale.set(...ABDOMEN);
abdomen.position.set(ABDOMEN_X, 0, ABDOMEN_Z);
abdomen.castShadow = true;
abdomen.add(hairs(900, true, 0.08, 0.2));
spider.add(abdomen);
const waist = new THREE.Mesh(new THREE.CylinderGeometry(0.028, 0.028, 0.06, 12), legSkin);   // the pedicel
waist.rotation.z = Math.PI / 2;
waist.position.set(-0.14, 0, 0.012);
spider.add(waist);
const eyeSkin = new THREE.MeshStandardMaterial({ color: 0x5ad1ff, emissive: 0x2a7fb0, roughness: 0.3 });
for (const [x, y, z, r] of [[0.150, 0.024, 0.030, 0.011], [0.150, -0.024, 0.030, 0.011]]) {   // two eyes
  const eye = new THREE.Mesh(new THREE.SphereGeometry(r, 10, 8), eyeSkin);
  eye.position.set(x, y, z);
  spider.add(eye);
}
for (const sgn of [-1, 1]) {                           // chelicerae, hanging from the front
  const fang = new THREE.Mesh(new THREE.CylinderGeometry(0.006, 0.003, 0.035, 8), legSkin);
  fang.position.set(0.15, sgn * 0.014, -0.035);
  fang.rotation.x = sgn * 0.25;
  spider.add(fang);
}
const unit = new THREE.CylinderGeometry(1, 1, 1, 12);
const femurShape = new THREE.CylinderGeometry(1, 0.85, 1, 12);   // +y end goes to the knee
const tibiaShape = new THREE.CylinderGeometry(0.55, 1, 1, 12);   // tapers to the foot
const legs = [];                                     // built from /api/describe: the world says how many legs and where their hips are
function buildLegs(hips) {
  for (const leg of legs) { spider.remove(leg.coxa, leg.hipJoint, leg.femur, leg.tibia, leg.knee); scene.remove(leg.foot); }
  legs.length = 0;
  for (const [hx, hy] of hips) {
  const femur = new THREE.Mesh(femurShape, legSkin), tibia = new THREE.Mesh(tibiaShape, legSkin);
  femur.castShadow = tibia.castShadow = true;
  femur.add(hairs(160, false, 0.3, 0.7));
  tibia.add(hairs(200, false, 0.3, 0.8));
  femur.visible = tibia.visible = false;             // unit cylinders until the first pose sizes them
  const knee = new THREE.Mesh(new THREE.SphereGeometry(FEMUR_R * 1.15, 12, 10), legSkin);   // the patella
  knee.visible = false;
  const root = new THREE.Vector3(hx * 0.85, Math.sign(hy) * ROOT_Y, -0.004);
  const hip = new THREE.Vector3(hx, hy, 0);
  const coxa = new THREE.Mesh(unit, legSkin);         // body to hip: the physics hip sits outside the ellipsoid
  coxa.castShadow = true;
  coxa.add(hairs(60, false, 0.2, 0.5));
  placeCylinder(coxa, root, hip, COXA_R);
  const hipJoint = new THREE.Mesh(new THREE.SphereGeometry(COXA_R, 12, 10), legSkin);
  hipJoint.position.copy(hip);
  hipJoint.castShadow = true;
  const foot = new THREE.Mesh(new THREE.CircleGeometry(0.02, 16), new THREE.MeshBasicMaterial({ color: 0x5ad1ff, transparent: true, opacity: 0.8 }));
  spider.add(coxa, hipJoint, femur, tibia, knee);
  scene.add(foot);                                   // planted discs lie in the world, on the ground
  foot.visible = false;
  legs.push({ coxa, hipJoint, femur, tibia, knee, foot, hip });
  }
}


const target = new THREE.Vector3();
let firstFrame = true;
let lastPose = null;
function draw(pose) {
  lastPose = pose;
  const z = pose.z;
  spider.position.set(pose.x, pose.y, z);
  spider.rotation.set(0, 0, pose.heading);
  const c = Math.cos(pose.heading), s = Math.sin(pose.heading);
  for (let i = 0; i < legs.length && i < pose.legs.length; i++) {
    const [kx, ky, kz, fx, fy, fz] = pose.legs[i];
    const leg = legs[i];
    _a.set(kx, ky, kz);                              // body frame: origin at the body centre
    _b.set(fx, fy, fz);
    placeCylinder(leg.femur, leg.hip, _a, FEMUR_R);
    placeCylinder(leg.tibia, _a, _b, TIBIA_R);
    leg.knee.position.copy(_a);
    leg.femur.visible = leg.tibia.visible = leg.knee.visible = true;
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
  describedBy(d);
  buildLegs(d.hips);
  if (lastPose) draw(lastPose);                      // a frame may have arrived first (a paused server sends one)
  const b = d.brain;
  $('brainline').textContent = b
    ? `${b.neurons} neurons · ${b.parameters} parameters · ${b.placement.backend}${b.placement.device ? ':' + b.placement.device : ''} ${b.placement.precision} · seed ${d.seed}`
    : `control: ${d.control && d.control.policy}`;
  if (b) openBrainView();
  else $('brainpanel-note').textContent = '· no brain in this life';
}).catch(() => {});
// What the description settles on the page: the law line, the motor line and the settings' fields.
// Called again after a control that changes the description (a law change, a load).
function describedBy(d) {
  described = d;
  if (lastState) $('law').textContent = law(lastState);
  $('motors').textContent = (d.motors || 'rate') + (d.layout === 'leg' ? ', a slot per leg' : '') + (d.setpoints ? ', setpoints sensed' : '') + (d.compass ? ', a compass' : '') + (d.ground ? ', the ground sense' : '');
  const r = d.reward || {};
  for (const k of LAW_KEYS) {
    const field = $('law-' + k);
    if (field.type === 'checkbox') field.checked = r[k] !== null && r[k] !== undefined && r[k] !== false; else field.value = r[k];
  }
  $('drift').textContent = r.drift ? `drift ${r.drift} m/s` : 'drift off';
  $('drift').classList.toggle('on', !!r.drift);
  $('birth-seed').value = d.seed;
  $('birth-motors').value = d.motors || 'target';
  $('birth-layout').value = d.layout || 'joint';
  $('birth-setpoints').checked = !!d.setpoints;
  $('birth-compass').checked = !!d.compass;
  $('birth-ground').checked = !!d.ground;
  const genes = { ...(d.genes || {}) };
  $('birth-hold').value = genes.hold == null ? '' : genes.hold;
  $('birth-entropy').value = genes.entropy == null ? 0 : genes.entropy;
  $('birth-maturity').checked = !!genes.maturity;
  $('birth-code').value = genes.code == null ? '' : genes.code;
  $('birth-learned-code').value = genes.learned_code == null ? '' : genes.learned_code;
  for (const k of ['hold', 'entropy', 'maturity', 'code', 'learned_code']) delete genes[k];
  $('birth-genes').value = geneWords(genes);
  $('genesline').textContent = 'live brain: ' + (geneWords(d.genes) || 'founder genes') + ' · seed ' + d.seed;
}
const LAW_KEYS = ['stance_pay', 'yaw_pay', 'window', 'high_water', 'old_ground', 'pay_per_metre', 'backward_pay', 'course', 'drift'];
const DRIFT_DOSE = 0.002;   // m/s: what the drift button switches on when the settings field names no dose
// The nursery's words for a brain's gene overrides: {gamma: 0.99, arousal: {need: 0}} → "gamma=0.99 arousal.need=0".
function geneWords(genes, prefix = '') {
  return Object.entries(genes || {}).map(([k, v]) =>
    v && typeof v === 'object' && !Array.isArray(v) ? geneWords(v, prefix + k + '.') : `${prefix}${k}=${Array.isArray(v) ? v.join(',') : v}`
  ).join(' ');
}
async function refreshDescription() {
  try { describedBy(await (await fetch('/api/describe')).json()); } catch (e) { /* the socket shows the state anyway */ }
}

// The reward law, from the settings the server described and the mark of the moment: "progress"
// is the plain law (each moment's progress); a window pays the mean of the last N moments; the
// high-water mark pays only ground beyond the farthest point reached, so the mark is shown with it.
function law(m) {
  const r = described && described.reward;
  if (!r) return '—';
  const parts = [r.window > 1 ? `window ${r.window}` : 'progress'];
  if (r.course !== null && r.course !== undefined) parts.push(`course ${fmt(r.course, 2)} rad · along it ${fmt(m.distance_course)} m`);
  if (r.pay_per_metre !== 40) parts.push(`${r.pay_per_metre}/m`);
  if (r.backward_pay) parts.push(`backward ${r.backward_pay}`);
  if (r.high_water) parts.push(`high-water${r.old_ground ? ' leaking ' + r.old_ground : ''} · mark ${fmt(m.mark)} m`);
  if (r.stance_pay) parts.push(`stance ${r.stance_pay}`);
  if (r.yaw_pay) parts.push(`yaw ${r.yaw_pay}`);
  if (r.drift) parts.push(`drift ${r.drift} m/s`);
  const changes = described.law_changes ? described.law_changes.length : 0;
  if (changes) parts.push(`a schedule: changed ${changes}×`);
  return parts.join(' · ');
}
let lastState = null;

// --- the brain view: the whole connectome in three dimensions, on the right -----------------
// The view is the connectome demo's (static/vendor/brain_view.js); the payload is this brain's
// connectome laid out as an ellipsoid, senses at the back, motor cortex at the front, each
// neuron's lines its incoming synapses. Every frame paints the settled state green and, on a learning
// moment, flashes gold where incoming synapses changed; the flash fades over the next frames.
let brainView = null, flash = null;
async function openBrainView() {
  try {
    const payload = await (await fetch('/api/brainview')).json();
    if (!payload || brainView) return;
    brainView = new BrainView($('view-host'), payload, {
      // some forty fibres cross every pixel of the ball, so each one is faint and the tone
      // map's exposure low; the demo's colours would burn the ball white
      spin: true, controls: 'legend', scale: 'auto', gamma: 0.8, floor: 0.05, exposure: 0.09,
      rest: [0.0015, 0.01, 0.003], lit: [0.03, 0.2, 0.06], flash: [0.6, 0.22, 0.01],
      somaRadius: 0.012, pointRadius: 0.8, pointDim: 0.35, fitScale: 0.95,
    });
    flash = new Float32Array(payload.order.length);
    brainView.start();
    $('brainpanel-note').textContent = `· ${payload.counts.neurons} neurons, ${payload.counts.synapses} synapses`;
    window.addEventListener('resize', () => brainView.resize());
    $('view-fit').addEventListener('click', () => brainView.fit());
    $('view-spin').addEventListener('click', () => {
      brainView.options.spin = !brainView.options.spin;
      $('view-spin').classList.toggle('on', brainView.options.spin);
    });
  } catch (e) {
    $('brainpanel-note').textContent = '· view unavailable';
  }
}
const FLASH_FADE = 0.96;                              // per frame received, as the demo's page
function paintBrain(m) {
  if (!brainView) return;
  const a = m.activity;
  if (!a || !a.all) return;                           // the random control: the last brain state stays
  const repair = a.repair;
  for (let i = 0; i < flash.length; i++) {
    const f = flash[i] * FLASH_FADE;
    flash[i] = repair && repair[i] > f ? repair[i] : f < 0.01 ? 0 : f;
  }
  brainView.setActivity(a.all, flash);
}

function update(m) {
  draw(m.pose);
  $('policy').textContent = m.policy;
  $('moments').textContent = m.moments;
  const mode = $('mode');
  mode.textContent = m.refused ? 'refused' : m.mode;
  mode.className = m.refused ? 'refused' : m.mode;
  $('forward').textContent = fmt(m.distance_forward) + ' m';
  $('net').textContent = fmt(m.displacement_m) + ' m' + (m.recentres ? ` · recentred ${m.recentres}×` : '');
  $('speed').textContent = fmt(m.speed) + ' m/s';
  $('reward').textContent = fmt(m.reward, 3) + (m.owed !== null && m.owed !== undefined ? ` (owed ${fmt(m.owed, 3)})` : '');
  lastState = m;
  $('law').textContent = law(m);
  $('arousal').textContent = `${fmt(m.level, 2)} / ${fmt(m.want, 2)}`;
  $('sweeps').textContent = `${m.sweeps} (${m.learning_sweeps} learning)`;
  $('ms').textContent = fmt(m.ms, 1);
  $('pace').textContent = (m.pace > 0 ? `${m.pace}/s` : 'max') + (m.rate > 0 ? ` · ${fmt(m.rate, 0)}/s made` : '');
  $('stance').textContent = m.pose.planted.filter(Boolean).length + ' / ' + m.pose.planted.length;
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
  learning(m);
  strip(m);
  paintBrain(m);
}

// --- the learning bar: 0 "Brain Stall" to 100 "Learning" ------------------------------------
// The brain learns only while aroused (routine moments are greedy and change nothing), so the
// bar is the aroused share of the brain's last LEARNING_BLOCKS blocks, 0-100. Before the first
// block closes it follows the mode of the moment. Random and frozen never learn: the bar stays
// empty and the note says why. It is a reading of the arousal law, not a measure of a gait.
const LEARNING_BLOCKS = 10;
function learning(m) {
  const pct = $('learning-pct'), cover = $('learning-cover'), note = $('learning-note');
  let score = null, text = '';
  if (m.policy !== 'brain') {
    score = 0;
    text = m.policy === 'random' ? 'the random control never learns' : 'the frozen copy never learns';
  } else {
    const blocks = m.blocks.filter(b => b.policy === 'brain').slice(-LEARNING_BLOCKS);
    if (blocks.length) {
      const share = blocks.reduce((s, b) => s + b.aroused_share, 0) / blocks.length;
      const sweeps = blocks.reduce((s, b) => s + b.learning_sweeps, 0);
      score = Math.round(100 * share);
      text = `aroused in ${score}% of the last ${blocks.length * 100} moments · ${sweeps} learning sweeps`;
    } else {
      score = m.mode === 'aroused' ? 100 : 0;
      text = 'no block closed yet · following the mode of the moment';
    }
  }
  pct.textContent = score;
  cover.style.width = (100 - score) + '%';
  note.textContent = text;
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
  ctx.fillText('sensory', 6, 11); ctx.fillText('association', 6, 45); ctx.fillText(described && described.motors === 'target' ? 'motor: to low end · hold setpoint · to high end, per joint' : 'motor ↓ hold ↑ per joint', 6, 91);
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
// Reset gives birth to what the settings name; genes are composed once, so this is the one
// place they are set. The page reloads on the new life: a reset is a new life, not a new frame.
function birth() {
  // the reward learner's three options and the two codes have fields of their own; the rest is the text, in the nursery's words
  const words = [$('birth-genes').value.trim()];
  if ($('birth-hold').value !== '') words.push('hold=' + Number($('birth-hold').value));
  if (Number($('birth-entropy').value) > 0) words.push('entropy=' + Number($('birth-entropy').value));
  if ($('birth-maturity').checked) words.push('maturity=true');
  if (Number($('birth-code').value) > 0) words.push('code=' + Math.round(Number($('birth-code').value)));
  if (Number($('birth-learned-code').value) > 0) words.push('learned_code=' + Math.round(Number($('birth-learned-code').value)));
  return { seed: Number($('birth-seed').value), motors: $('birth-motors').value, layout: $('birth-layout').value,
           setpoints: $('birth-setpoints').checked, compass: $('birth-compass').checked, ground: $('birth-ground').checked,
           genes: words.filter(Boolean).join(' ') };
}
function resetQuestion(b) {
  return 'Reset everything?\n\nA newborn brain replaces the live one: seed ' + b.seed + ', ' + b.motors + ' motors' + (b.layout === 'leg' ? ', a slot per leg' : '')
    + (b.setpoints ? ', setpoints sensed' : '') + (b.compass ? ', a compass' : '') + (b.ground ? ', the ground sense' : '') + ', ' + (b.genes ? 'genes ' + b.genes : 'founder genes') + '.\n'
    + 'The world returns to the origin, every counter goes to zero and the saved life in runs/ is deleted.\n\n'
    + 'The brain that has lived until now is gone for good.';
}
async function control(body) {
  const r = await fetch('/api/control', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!r.ok) { $('notes').textContent = await r.text(); return null; }
  return r.json();
}
document.querySelectorAll('button[data-op]').forEach(b => b.addEventListener('click', async () => {
  const body = { op: b.dataset.op };
  if (b.dataset.op === 'reset') {
    body.value = birth();
    if (!window.confirm(resetQuestion(body.value))) return;
  } else if (b.dataset.value !== undefined) body.value = isNaN(b.dataset.value) ? b.dataset.value : Number(b.dataset.value);
  const result = await control(body);
  if (result && b.dataset.op === 'reset') location.reload();
  else if (result && b.dataset.op === 'load') refreshDescription();
}));
// The drift button: off → on at the dose the settings field names (or the measured 0.002 m/s), on → off.
$('drift').addEventListener('click', async () => {
  const on = !!(described && described.reward && described.reward.drift);
  const dose = Number($('law-drift').value) > 0 ? Number($('law-drift').value) : DRIFT_DOSE;
  if (await control({ op: 'law', value: { drift: on ? 0 : dose } })) refreshDescription();
});
$('apply-law').addEventListener('click', async () => {
  const value = {};
  for (const k of LAW_KEYS) {
    const field = $('law-' + k);
    value[k] = field.type === 'checkbox' ? field.checked : Number(field.value);
  }
  // the course box: ticked on a life with a course keeps that heading; ticked on one without, the
  // server takes the heading the body holds now (true); unticked is off (null)
  const had = described && described.reward ? described.reward.course : null;
  value.course = value.course ? (had === null || had === undefined ? true : had) : null;
  if (await control({ op: 'law', value })) refreshDescription();
});

function connect() {
  const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  ws.onopen = () => { $('link').textContent = 'live'; if (described) refreshDescription(); };   // a restarted server may hold another life
  ws.onmessage = e => update(JSON.parse(e.data));
  ws.onclose = () => { $('link').textContent = 'reconnecting…'; setTimeout(connect, 1000); };
  ws.onerror = () => ws.close();
}
connect();

// --- the world panel: the colours of the ground and the grid, how the world is drawn, not its law
// (the brain senses joints, feet and motion, never a colour; the choices live in this browser only)
const VIEW = {
  ground: { own: GROUND_COLOUR, paint: hex => ground.material.color.set(hex) },
  grid: { own: GRID_COLOUR, paint: hex => makeGrid(hex) },
};
function paintView(name, hex) {
  VIEW[name].paint(hex);
  $(name + '-colour').value = hex;
}
function rememberView(name, hex) {
  const key = name + '-colour';
  try { if (hex === VIEW[name].own) localStorage.removeItem(key); else localStorage.setItem(key, hex); } catch (e) { /* no storage: the colour lasts the page */ }
}
for (const name of Object.keys(VIEW)) {
  try { const kept = localStorage.getItem(name + '-colour'); if (kept) paintView(name, kept); } catch (e) { /* no storage */ }
  $(name + '-colour').addEventListener('input', (e) => { paintView(name, e.target.value); rememberView(name, e.target.value); });
}
$('view-default').addEventListener('click', () => {
  for (const name of Object.keys(VIEW)) { paintView(name, VIEW[name].own); rememberView(name, VIEW[name].own); }
});
