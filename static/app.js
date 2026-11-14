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
controls.maxDistance = 12;                            // far enough to see the whole arena, its rim 5 m from the centre
// The fog begins beyond the spider at every distance, so zooming out shows the arena and not a haze.
function fogFor(r) { scene.fog.near = r + 3; scene.fog.far = r + 11; }
fogFor(camera.position.length());
// The wheel zooms by the distance scrolled, in the page's own handler: OrbitControls r160
// divides the wheel's delta by the device pixel ratio truncated to an integer, which is zero
// under a browser zoom below 100% (and on any display reporting a ratio under 1), so that one
// notch jumped to the nearest or the farthest distance and the zoom had two steps.
controls.enableZoom = false;
canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  const px = e.deltaY * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 288 : 1);   // lines and pages as pixels
  const k = Math.exp(px * 0.0015);                      // a 100 px notch: 16% nearer or farther
  const offset = camera.position.clone().sub(controls.target);
  const r = Math.min(controls.maxDistance, Math.max(controls.minDistance, offset.length() * k));
  camera.position.copy(controls.target).add(offset.setLength(r));
  fogFor(r);
}, { passive: false });

const sky = new THREE.HemisphereLight(0xd9e6ff, 0x202830, 0.9);
scene.add(sky);
const sun = new THREE.DirectionalLight(0xffffff, 1.4);
const LIGHT = { sky: sky.intensity, sun: sun.intensity };   // the page's own; the world panel scales both
sun.position.set(2, -3, 5);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.camera.left = sun.shadow.camera.bottom = -3;
sun.shadow.camera.right = sun.shadow.camera.top = 3;
scene.add(sun);
scene.add(sun.target);

const GROUND_COLOUR = '#1b2230';                      // the page's own; the view panel changes it, for this browser only
// The ground's height, the body's ground_height: 0 on the flat arena, the sum of the server's
// waves (wavelength, direction, amplitude, phase) on the hills and the steep hills. The page draws the server's terrain;
// it never has one of its own.
let waves = [];
function setArena(name, terrains) {
  waves = ((terrains || {})[name] || []).map(([w, d, a, ph]) => [2 * Math.PI / w * Math.cos(d), 2 * Math.PI / w * Math.sin(d), a, ph]);
}
function groundHeight(x, y) {
  let h = 0;
  for (const [kx, ky, a, ph] of waves) h += a * Math.cos(kx * x + ky * y + ph);
  return h;
}
function lay(geometry, lift) {                        // lifts every vertex of a z-up geometry onto the ground
  const p = geometry.attributes.position;
  for (let i = 0; i < p.count; i++) p.setZ(i, groundHeight(p.getX(i), p.getY(i)) + lift);
  p.needsUpdate = true;
  geometry.computeVertexNormals();
  geometry.computeBoundingSphere();
}
const ground = new THREE.Mesh(new THREE.RingGeometry(0.001, 5.5, 160, 110), new THREE.MeshStandardMaterial({ color: GROUND_COLOUR, roughness: 1 }));
ground.receiveShadow = true;
scene.add(ground);
const GRID_COLOUR = '#262e3c';                        // the page's own; the two axes through the origin a shade lighter
let grid = null, gridHex = GRID_COLOUR;
function makeGrid(hex) {                              // the lines follow the ground: built from short segments, rebuilt to recolour
  gridHex = hex;
  if (grid) { scene.remove(grid); grid.geometry.dispose(); grid.material.dispose(); }
  const line = new THREE.Color(hex), axes = new THREE.Color(hex).lerp(new THREE.Color('#ffffff'), 0.12);
  const pos = [], col = [], STEP = 0.05;
  for (let i = -22; i <= 22; i++) {
    const c = i === 0 ? axes : line, u = i * 0.25;
    for (let t = -5.5; t < 5.5 - 1e-9; t += STEP) {
      for (const [x0, y0, x1, y1] of [[t, u, t + STEP, u], [u, t, u, t + STEP]]) {
        if (Math.hypot(x0, y0) > 5.5 || Math.hypot(x1, y1) > 5.5) continue;   // the grid ends with the ground disc
        pos.push(x0, y0, groundHeight(x0, y0) + 0.001, x1, y1, groundHeight(x1, y1) + 0.001);
        col.push(c.r, c.g, c.b, c.r, c.g, c.b);
      }
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  grid = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ vertexColors: true }));
  scene.add(grid);
}
const rim = new THREE.Mesh(new THREE.RingGeometry(4.98, 5.02, 256), new THREE.MeshBasicMaterial({ color: 0x5ad1ff, side: THREE.DoubleSide }));
scene.add(rim);
let arenaDrawn = null;
function drawArena(name, terrains) {                     // the ground, its grid and the rim, laid on the arena's terrain
  if (name === arenaDrawn) return;
  arenaDrawn = name;
  setArena(name, terrains);
  lay(ground.geometry, 0);
  lay(rim.geometry, 0.002);
  makeGrid(gridHex);
}
drawArena('flat');

// --- the surface: a meadow, gravel, or both, painted on the ground. How the world is drawn, not
// its law: the body's contact and the reward are the same on every surface, the brain senses no
// colour, and the choice lives in this browser only. Drawn on canvases here, nothing is loaded.
const SURFACE_TILE = 0.5;                             // metres one tile covers
function seeded(seed) {                               // a small deterministic generator, so a surface looks the same on every load
  let s = seed >>> 0;
  return () => { s = (s + 0x6d2b79f5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
function tileCanvas(size) { const c = document.createElement('canvas'); c.width = c.height = size; return c; }
function wrapped(size, x, y, r, draw) {               // draws at a point and at its copies across the edges, so the tile repeats seamlessly
  for (const dx of [-size, 0, size]) for (const dy of [-size, 0, size]) {
    if (x + dx < -r || x + dx > size + r || y + dy < -r || y + dy > size + r) continue;
    draw(x + dx, y + dy);
  }
}
function meadowTile(size = 512) {
  const colour = tileCanvas(size), height = tileCanvas(size), rnd = seeded(7);
  const c = colour.getContext('2d'), h = height.getContext('2d');
  c.fillStyle = '#2f4a1f'; c.fillRect(0, 0, size, size);
  h.fillStyle = '#404040'; h.fillRect(0, 0, size, size);
  for (let i = 0; i < 260; i++) {                     // soft patches of lighter and darker soil under the grass
    const x = rnd() * size, y = rnd() * size, r = 10 + rnd() * 40, g = rnd() < 0.5 ? 'rgba(70,95,35,0.25)' : 'rgba(25,40,15,0.3)';
    wrapped(size, x, y, r, (px, py) => { c.fillStyle = g; c.beginPath(); c.arc(px, py, r, 0, 2 * Math.PI); c.fill(); });
  }
  c.lineCap = h.lineCap = 'round';
  for (let i = 0; i < 9000; i++) {                    // blades: short strokes leaning one way, lighter at the tip
    const x = rnd() * size, y = rnd() * size, len = 4 + rnd() * 10, lean = -0.6 + rnd() * 1.2 - 1.57;
    const ex = Math.cos(lean) * len, ey = Math.sin(lean) * len;
    const l = 22 + rnd() * 22, hue = 78 + rnd() * 34, w = 0.8 + rnd() * 1.1;
    wrapped(size, x, y, len, (px, py) => {
      c.strokeStyle = `hsl(${hue},${45 + rnd() * 20}%,${l}%)`; c.lineWidth = w;
      c.beginPath(); c.moveTo(px, py); c.lineTo(px + ex, py + ey); c.stroke();
      const v = 90 + Math.round(rnd() * 120); h.strokeStyle = `rgb(${v},${v},${v})`; h.lineWidth = w;
      h.beginPath(); h.moveTo(px, py); h.lineTo(px + ex, py + ey); h.stroke();
    });
  }
  for (let i = 0; i < 40; i++) {                      // a few small flowers, white and yellow
    const x = rnd() * size, y = rnd() * size, col = rnd() < 0.6 ? '#e8e4d0' : '#e2c43c';
    wrapped(size, x, y, 3, (px, py) => { c.fillStyle = col; c.beginPath(); c.arc(px, py, 1.6, 0, 2 * Math.PI); c.fill(); });
  }
  return { colour, height };
}
function gravelTile(size = 512) {
  const colour = tileCanvas(size), height = tileCanvas(size), rnd = seeded(11);
  const c = colour.getContext('2d'), h = height.getContext('2d');
  c.fillStyle = '#3a352e'; c.fillRect(0, 0, size, size);
  h.fillStyle = '#202020'; h.fillRect(0, 0, size, size);
  for (let i = 0; i < 2600; i++) {                    // stones: rounded, of greys and browns, lit from one side
    const x = rnd() * size, y = rnd() * size, rx = 3 + rnd() * 9, ry = rx * (0.55 + rnd() * 0.45), rot = rnd() * Math.PI;
    const base = 70 + rnd() * 90, warm = rnd() * 18;
    const fill = `rgb(${base + warm},${base + warm * 0.6},${base})`, lit = `rgba(255,250,240,${0.12 + rnd() * 0.12})`;
    const hv = 120 + Math.round(rnd() * 135);
    wrapped(size, x, y, rx + 2, (px, py) => {
      c.fillStyle = 'rgba(0,0,0,0.35)'; c.beginPath(); c.ellipse(px + 1.2, py + 1.6, rx, ry, rot, 0, 2 * Math.PI); c.fill();
      c.fillStyle = fill; c.beginPath(); c.ellipse(px, py, rx, ry, rot, 0, 2 * Math.PI); c.fill();
      c.fillStyle = lit; c.beginPath(); c.ellipse(px - rx * 0.25, py - ry * 0.3, rx * 0.5, ry * 0.45, rot, 0, 2 * Math.PI); c.fill();
      const g = h.createRadialGradient(px, py, 0, px, py, rx);
      g.addColorStop(0, `rgb(${hv},${hv},${hv})`); g.addColorStop(1, 'rgb(40,40,40)');
      h.fillStyle = g; h.beginPath(); h.ellipse(px, py, rx, ry, rot, 0, 2 * Math.PI); h.fill();
    });
  }
  return { colour, height };
}
// Meadow and gravel together: the whole disc on one canvas, gravel under a meadow cut by a
// smooth mask of the same waves wherever it is drawn, so paths of gravel wind through the grass.
function mixedDisc(meadow, gravel, per = 186) {
  const span = 11, size = per * (span / SURFACE_TILE);   // whole tiles, so the disc's canvas repeats seamlessly every 11 m
  const out = { colour: tileCanvas(size), height: tileCanvas(size) };
  const M = 256, mask = tileCanvas(M), m = mask.getContext('2d'), img = m.createImageData(M, M);
  for (let j = 0; j < M; j++) for (let i = 0; i < M; i++) {
    const u = 2 * Math.PI * i / M, w = 2 * Math.PI * j / M;   // whole waves over the 11 m, so the mask repeats with the canvas
    const v = Math.sin(2 * u + 0.4) + Math.sin(2 * w - 1.1) + 0.8 * Math.sin(u + w + 2.0) + 0.6 * Math.sin(4 * u - 3 * w);
    const a = Math.max(0, Math.min(1, (v + 0.35) / 0.5));   // meadow where the sum is high, a soft edge of a few centimetres
    img.data[(j * M + i) * 4 + 3] = Math.round(255 * a);
  }
  m.putImageData(img, 0, 0);
  for (const key of ['colour', 'height']) {
    const ctx = out[key].getContext('2d'), layer = tileCanvas(size), l = layer.getContext('2d');
    const paint = (target, src) => { const p = target.createPattern(src[key], 'repeat'); p.setTransform(new DOMMatrix().scale(per / src[key].width)); target.fillStyle = p; target.fillRect(0, 0, size, size); };
    paint(ctx, gravel);
    paint(l, meadow);
    l.globalCompositeOperation = 'destination-in'; l.imageSmoothingEnabled = true; l.drawImage(mask, 0, 0, size, size);
    ctx.drawImage(layer, 0, 0);
  }
  return out;
}
const surfaces = {};                                   // built on first use, then kept
function surfaceMaps(name) {
  if (surfaces[name]) return surfaces[name];
  const made = name === 'meadow' ? meadowTile() : name === 'gravel' ? gravelTile()
    : mixedDisc(surfaceMaps('meadow').source, surfaceMaps('gravel').source);
  const repeat = name === 'mixed' ? 1 : 11 / SURFACE_TILE;   // the ring's uv spans its 11 m diameter once
  const texture = (canvas, colour) => {
    const t = new THREE.CanvasTexture(canvas);
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    t.repeat.set(repeat, repeat);
    t.anisotropy = renderer.capabilities.getMaxAnisotropy();
    if (colour) t.colorSpace = THREE.SRGBColorSpace;
    return t;
  };
  return (surfaces[name] = { source: made, map: texture(made.colour, true), bump: texture(made.height, false) });
}
let surfaceName = 'plain', groundHex = GROUND_COLOUR;
function setSurface(name) {
  surfaceName = name;
  const mat = ground.material;
  if (name === 'plain') { mat.map = null; mat.bumpMap = null; mat.color.set(groundHex); }
  else {
    const s = surfaceMaps(name);
    mat.map = s.map; mat.bumpMap = s.bump; mat.bumpScale = name === 'meadow' ? 1.2 : 2.5;
    mat.color.set('#ffffff');
  }
  mat.needsUpdate = true;
}
let viewFollows = true;                             // the view panel's choice: the camera follows the spider, or stays
function setView(value) { viewFollows = value !== 'fixed'; }
function setGroundColour(hex) { groundHex = hex; if (surfaceName === 'plain') ground.material.color.set(hex); }

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
const eyeSkin = new THREE.MeshStandardMaterial({ color: 0x7a0f14, emissive: 0x8a0c10, roughness: 0.3 });   // dark red, glowing
for (const [x, y, z, r] of [[0.150, 0.024, 0.030, 0.011], [0.150, -0.024, 0.030, 0.011]]) {   // two eyes
  const eye = new THREE.Mesh(new THREE.SphereGeometry(r, 10, 8), eyeSkin);
  eye.position.set(x, y, z);
  spider.add(eye);
}
// Chelicerae: two stout basal segments reaching forward and down from under the eyes, each
// ending in a fang that curves down and inward, the way a tarantula's hang.
const fangSkin = new THREE.MeshStandardMaterial({ color: 0x0b0d12, roughness: 0.3, metalness: 0.45 });
for (const sgn of [-1, 1]) {
  const root = new THREE.Vector3(0.135, sgn * 0.024, -0.008), joint = new THREE.Vector3(0.19, sgn * 0.028, -0.034);
  const base = new THREE.Mesh(new THREE.CylinderGeometry(1, 1, 1, 12), legSkin);
  placeCylinder(base, root, joint, 0.013);
  base.castShadow = true;
  base.add(hairs(40, false, 0.3, 0.6));
  spider.add(base);
  const knuckle = new THREE.Mesh(new THREE.SphereGeometry(0.013, 10, 8), legSkin);   // rounds the joint
  knuckle.position.copy(joint);
  spider.add(knuckle);
  const curve = new THREE.QuadraticBezierCurve3(joint, new THREE.Vector3(0.215, sgn * 0.024, -0.062),
                                                new THREE.Vector3(0.196, sgn * 0.008, -0.085));
  const fang = new THREE.Mesh(new THREE.TubeGeometry(curve, 10, 0.0045, 8, false), fangSkin);
  fang.castShadow = true;
  spider.add(fang);
  const tip = new THREE.Mesh(new THREE.ConeGeometry(0.0045, 0.014, 8), fangSkin);   // the point, along the curve's end
  const end = curve.getPoint(1), tangent = curve.getTangent(1);
  tip.position.copy(end).addScaledVector(tangent, 0.006);
  tip.quaternion.setFromUnitVectors(_yUp, tangent);
  spider.add(tip);
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
    leg.foot.position.set(pose.x + c * fx - s * fy, pose.y + s * fx + c * fy, (pose.ground ? pose.ground[i] : 0) + 0.003);
  }
  const next = new THREE.Vector3(pose.x, pose.y, groundHeight(pose.x, pose.y) + 0.08);
  if (!viewFollows && !firstFrame) next.copy(target);   // a fixed view: the camera stays where it is, the spider may walk out of it
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
  sun.position.set(target.x + 2, target.y - 3, target.z + 5);   // the shadow's box goes with the spider
  sun.target.position.copy(target);
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
// A number field keeps its own range: a value past min or max is set back to that end when the
// field is left, and when it is read (40 typed into backward pay, a share 0 to 1, becomes 1).
function clamp(field) {
  if (field.type !== 'number' || field.value === '') return;
  let v = Number(field.value);
  if (!Number.isFinite(v)) { field.value = ''; return; }
  if (field.min !== '' && v < Number(field.min)) v = Number(field.min);
  if (field.max !== '' && v > Number(field.max)) v = Number(field.max);
  if (field.step === '1') v = Math.round(v);
  if (String(v) !== field.value) field.value = v;
}
const num = field => { clamp(field); return Number(field.value); };
for (const f of document.querySelectorAll('input[type=number]')) f.addEventListener('change', () => clamp(f));
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
// What the description settles on the page: the law line, the muscle line and the settings' fields.
// Called again after a control that changes the description (a law change, a load).
let lawFilled = null;                                  // the law the fields were last filled from
function describedBy(d) {
  described = d;
  if (lastState) $('law').textContent = law(lastState);
  $('muscles').textContent = (d.muscles || 'rate') + (d.layout === 'leg' ? ', a slot per leg' : '') + (d.setpoints ? ', setpoints sensed' : '') + (d.compass ? ', a compass' : '') + (d.ground ? ', the ground sense' : '');
  const r = d.reward || {};
  drawArena(r.arena || 'flat', d.terrains || { hills: d.hills });   // a server from before the steep hills sent its one terrain as hills
  $('arena').value = r.arena || 'flat';
  // The fields are filled from the server's law only when that law is new to the page: a
  // description re-read for another reason (the socket reconnecting) leaves a half-made edit alone.
  const lawNow = JSON.stringify(LAW_KEYS.map(k => r[k]));
  if (lawNow !== lawFilled) {
    for (const k of LAW_KEYS) {
      const field = $('law-' + k);
      if (field.type === 'checkbox') field.checked = r[k] !== null && r[k] !== undefined && r[k] !== false; else field.value = r[k];
    }
    lawFilled = lawNow;
  }
  $('drift').textContent = r.drift ? `drift ${r.drift} m/s` : 'drift off';
  $('drift').classList.toggle('on', !!r.drift);
  $('slope_pull').textContent = r.slope_pull ? `slope pull ${r.slope_pull}` + ((r.arena || 'flat') === 'flat' ? ' (flat: none)' : '') : 'slope pull off';
  $('slope_pull').classList.toggle('on', !!r.slope_pull);
  $('birth-seed').value = d.seed;
  $('birth-muscles').value = d.muscles || 'target';
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
const LAW_KEYS = ['stance_pay', 'yaw_pay', 'window', 'high_water', 'old_ground', 'pay_per_metre', 'backward_pay', 'course', 'drift', 'strict', 'muscle_rate', 'slope_pull'];
const DRIFT_DOSE = 0.002;   // m/s: what the drift button switches on when the settings field names no dose
const SLOPE_DOSE = 0.5;     // m/s at a slope of 1: what the slope pull button switches on when the field names none (unmeasured)
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
  if (r.course !== null && r.course !== undefined) parts.push(`course ${fmt(r.course, 2)} rad${r.strict ? ', strict' : ''} · along it ${fmt(m.distance_course)} m`);
  if (r.pay_per_metre !== 40) parts.push(`${r.pay_per_metre}/m`);
  if (r.backward_pay) parts.push(`backward ${r.backward_pay}`);
  if (r.high_water) parts.push(`high-water${r.old_ground ? ' leaking ' + r.old_ground : ''} · mark ${fmt(m.mark)} m`);
  if (r.stance_pay) parts.push(`stance ${r.stance_pay}`);
  if (r.yaw_pay) parts.push(`yaw ${r.yaw_pay}`);
  if (r.drift) parts.push(`drift ${r.drift} m/s`);
  if (r.muscle_rate !== undefined && r.muscle_rate !== 1) parts.push(`muscles ×${r.muscle_rate}`);
  if (r.arena && r.arena !== 'flat') parts.push(r.arena);
  if (r.slope_pull) parts.push(`slope pull ${r.slope_pull}`);
  const changes = described.law_changes ? described.law_changes.length : 0;
  if (changes) parts.push(`a schedule: changed ${changes}×`);
  return parts.join(' · ');
}
let lastState = null;

// The speed as a bar under the reading. The moment's speed is a sample of the stride, swinging
// each way within one step, so the bar shows means: the fill is the mean forward speed of the
// last 1000 moments (the learning bar's horizon, 50 s of the body's clock), the tick the last 100.
// The scale is 0.05 m/s each way: the fastest gait measured (1.55 m per 1000 moments) fills 60%.
const SPEED_SCALE = 0.05;
const SPEED_BLOCKS = 10;
function speedbar(m) {
  const dt = (described && described.dt) || 0.05;
  const mean = blocks => blocks.length ? blocks.reduce((a, b) => a + b.forward_m, 0) / (blocks.length * 100 * dt) : 0;
  const share = Math.max(-1, Math.min(1, mean(m.blocks.slice(-SPEED_BLOCKS)) / SPEED_SCALE)) * 50;
  const fill = $('speed-fill');
  fill.classList.toggle('back', share < 0);
  fill.style.left = (share < 0 ? 50 + share : 50) + '%';
  fill.style.width = Math.abs(share) + '%';
  $('speed-mean').style.left = (50 + Math.max(-1, Math.min(1, mean(m.blocks.slice(-1)) / SPEED_SCALE)) * 50) + '%';
  $('speed-label').textContent = fmt(mean(m.blocks.slice(-SPEED_BLOCKS))) + ' m/s';
}

// --- the brain view: the whole connectome in three dimensions, on the right -----------------
// The view is the connectome demo's (static/vendor/brain_view.js); the payload is this brain's
// connectome laid out as an ellipsoid, senses at the back, motor cortex at the front, each
// neuron's lines its incoming synapses. Every frame paints the settled state blue and, on a learning
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
      rest: [0.002, 0.005, 0.012], lit: [0.05, 0.13, 0.32], flash: [0.6, 0.22, 0.01],   // blue for the state, gold for learning
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

// The wait until the spider moves: the page opens with it shown (the server may still be
// composing or loading the brain), a Reset and a lost connection show it again, and it goes
// once the moments count on, or at once on a paused life, which will not move until resumed.
let initFrom = null;
function initializing(on) {
  $('init').classList.toggle('show', on);
  if (on) initFrom = null;
}
function settled(m) {
  if (!$('init').classList.contains('show')) return;
  if (m.paused) { initializing(false); return; }
  if (initFrom === null) initFrom = m.moments;
  else if (m.moments > initFrom) initializing(false);
}

function update(m) {
  settled(m);
  draw(m.pose);
  $('policy').textContent = m.policy;
  $('moments').textContent = m.moments;
  const mode = $('mode');
  mode.textContent = m.refused ? 'refused' : m.mode;
  mode.className = m.refused ? 'refused' : m.mode;
  $('forward').textContent = fmt(m.distance_forward) + ' m';
  $('net').textContent = fmt(m.displacement_m) + ' m' + (m.recentres ? ` · recentred ${m.recentres}×` : '');
  $('speed').textContent = fmt(m.speed) + ' m/s';
  speedbar(m);
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
  const notes = $('notes');
  const pinned = notes.scrollHeight - notes.scrollTop - notes.clientHeight < 4;   // at the end, or not yet scrolling
  notes.textContent = m.notes.join('\n');
  if (pinned) notes.scrollTop = notes.scrollHeight;
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
  ctx.fillText('sensory', 6, 11); ctx.fillText('association', 6, 45); ctx.fillText(described && described.muscles === 'target' ? 'motor: to low end · hold setpoint · to high end, per joint' : 'motor ↓ hold ↑ per joint', 6, 91);
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
  if ($('birth-hold').value !== '') words.push('hold=' + num($('birth-hold')));
  if (num($('birth-entropy')) > 0) words.push('entropy=' + Number($('birth-entropy').value));
  if ($('birth-maturity').checked) words.push('maturity=true');
  if (num($('birth-code')) > 0) words.push('code=' + Math.round(Number($('birth-code').value)));
  if (num($('birth-learned-code')) > 0) words.push('learned_code=' + Math.round(Number($('birth-learned-code').value)));
  return { seed: num($('birth-seed')), muscles: $('birth-muscles').value, layout: $('birth-layout').value,
           setpoints: $('birth-setpoints').checked, compass: $('birth-compass').checked, ground: $('birth-ground').checked,
           genes: words.filter(Boolean).join(' ') };
}
function resetQuestion(b) {
  return 'Reset everything?\n\nA newborn brain replaces the live one: seed ' + b.seed + ', ' + b.muscles + ' muscles' + (b.layout === 'leg' ? ', a slot per leg' : '')
    + (b.setpoints ? ', setpoints sensed' : '') + (b.compass ? ', a compass' : '') + (b.ground ? ', the ground sense' : '') + ', ' + (b.genes ? 'genes ' + b.genes : 'founder genes') + '.\n'
    + 'The world returns to the origin, every counter goes to zero and the saved life in runs/ is deleted.\n\n'
    + 'The brain that has lived until now is gone for good.';
}
// A word that hovers over a button for a moment and fades: the page's answer to a click
// whose result shows nowhere else (a save changes nothing on the panel).
function toast(anchor, text) {
  const t = document.createElement('div');
  t.className = 'toast';
  t.textContent = text;
  document.body.appendChild(t);
  const r = anchor.getBoundingClientRect();
  t.style.left = (r.left + r.width / 2) + 'px';
  t.style.top = r.top + 'px';
  requestAnimationFrame(() => t.classList.add('show'));
  setTimeout(() => { t.classList.remove('show'); setTimeout(() => t.remove(), 300); }, 1100);
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
    initializing(true);
  } else if (b.dataset.value !== undefined) body.value = isNaN(b.dataset.value) ? b.dataset.value : Number(b.dataset.value);
  const result = await control(body);
  if (!result && b.dataset.op === 'reset') initializing(false);   // refused: the old life goes on
  if (result && b.dataset.op === 'reset') location.reload();
  else if (result && b.dataset.op === 'load') refreshDescription();
  else if (result && b.dataset.op === 'save') toast(b, result.saved ? 'saved' : (result.why || 'not saved'));
}));
// The drift button: off → on at the dose the settings field names (or the measured 0.002 m/s), on → off.
$('drift').addEventListener('click', async () => {
  const on = !!(described && described.reward && described.reward.drift);
  const dose = num($('law-drift')) > 0 ? Number($('law-drift').value) : DRIFT_DOSE;
  if (await control({ op: 'law', value: { drift: on ? 0 : dose } })) refreshDescription();
});
// The arena select: the ground is the world's law, unlike the colours below it, so a change is a
// law change on the living brain and makes the life a schedule.
$('arena').addEventListener('change', async (e) => {
  const was = (described && described.reward && described.reward.arena) || 'flat';
  if (await control({ op: 'law', value: { arena: e.target.value } })) refreshDescription();
  else e.target.value = was;
});
// The slope pull button: off → on at the dose the settings field names (or 0.5), on → off.
$('slope_pull').addEventListener('click', async () => {
  const on = !!(described && described.reward && described.reward.slope_pull);
  const dose = num($('law-slope_pull')) > 0 ? Number($('law-slope_pull').value) : SLOPE_DOSE;
  if (await control({ op: 'law', value: { slope_pull: on ? 0 : dose } })) refreshDescription();
});
$('apply-law').addEventListener('click', async () => {
  const value = {};
  for (const k of LAW_KEYS) {
    const field = $('law-' + k);
    value[k] = field.type === 'checkbox' ? field.checked : num(field);
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
  ws.onclose = () => { $('link').textContent = 'reconnecting…'; initializing(true); setTimeout(connect, 1000); };
  ws.onerror = () => ws.close();
}
connect();

// --- the world panel: the surface and the colours of the ground and the grid, how the world is drawn, not its law
// (the brain senses joints, feet and motion, never a colour; the choices live in this browser only)
const VIEW = {
  ground: { own: GROUND_COLOUR, field: 'ground-colour', paint: hex => setGroundColour(hex) },
  surface: { own: 'plain', field: 'surface', paint: name => setSurface(name) },
  follow: { own: 'follows', field: 'follow', paint: value => setView(value) },
  grid: { own: GRID_COLOUR, field: 'grid-colour', paint: hex => makeGrid(hex) },
  light: { own: '1', field: 'light', paint: v => { sky.intensity = LIGHT.sky * Number(v); sun.intensity = LIGHT.sun * Number(v); } },
};
function paintView(name, value) {
  VIEW[name].paint(value);
  $(VIEW[name].field).value = value;
  if (name === 'light') $('light-value').textContent = Number(value).toFixed(1) + '×';
}
function rememberView(name, value) {
  const key = 'world-' + name;
  try { if (String(value) === VIEW[name].own) localStorage.removeItem(key); else localStorage.setItem(key, value); } catch (e) { /* no storage: the choice lasts the page */ }
}
for (const name of Object.keys(VIEW)) {
  paintView(name, VIEW[name].own);
  try { const kept = localStorage.getItem('world-' + name); if (kept) paintView(name, kept); } catch (e) { /* no storage */ }
  $(VIEW[name].field).addEventListener('input', (e) => { paintView(name, e.target.value); rememberView(name, e.target.value); });
}
$('view-default').addEventListener('click', () => {
  for (const name of Object.keys(VIEW)) { paintView(name, VIEW[name].own); rememberView(name, VIEW[name].own); }
});
