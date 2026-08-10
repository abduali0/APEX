import './style.css';
import { FilesetResolver, HandLandmarker, type HandLandmarkerResult } from '@mediapipe/tasks-vision';
import * as THREE from 'three';

type Landmark = { x: number; y: number; z: number };
type HandLandmarks = Landmark[];

const video = document.getElementById('webcam') as HTMLVideoElement;
const overlay = document.getElementById('overlay') as HTMLCanvasElement;
const overlayCtx = overlay.getContext('2d')!;
const sceneCanvas = document.getElementById('scene') as HTMLCanvasElement;
const statusEl = document.getElementById('status')!;

const HAND_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

const MAX_HANDS = 2;
const HAND_COLORS = [0x66ffff, 0xff66d9];
const GRAB_COLOR = 0xffcc33;

// ---------- Three.js hologram setup ----------
const renderer = new THREE.WebGLRenderer({ canvas: sceneCanvas, alpha: true, antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);

const scene3 = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 100);
camera.position.set(0, 0, 6);

function resize() {
  const w = window.innerWidth;
  const h = window.innerHeight;
  renderer.setSize(w, h, false);
  overlay.width = w;
  overlay.height = h;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);

// ---------- Per-hand hologram state ----------
interface HandHologram {
  group: THREE.Group;
  material: THREE.MeshBasicMaterial;
  color: number;

  present: boolean;
  grabbing: boolean;
  homePos: THREE.Vector3;

  // Hover (not grabbing): position/scale drift smoothly toward the hand.
  targetPos: THREE.Vector3;
  targetScale: number;

  // Grab (pinching): rotation and position are bound directly to the hand.
  grabStartHandQuat: THREE.Quaternion | null;
  grabStartObjQuat: THREE.Quaternion | null;
  prevHandQuat: THREE.Quaternion | null;
  prevPos: THREE.Vector3 | null;

  // Momentum carried after release, decays over time (the "flick"/"spin" feel).
  angularVelocity: THREE.Vector3; // axis * radians/sec
  linearVelocity: THREE.Vector3;
}

function createHologram(color: number, homeX: number): HandHologram {
  const group = new THREE.Group();
  const homePos = new THREE.Vector3(homeX, 0, 0);
  group.position.copy(homePos);
  scene3.add(group);

  const geometry = new THREE.IcosahedronGeometry(1, 1);
  const material = new THREE.MeshBasicMaterial({
    color,
    wireframe: true,
    transparent: true,
    opacity: 0.25,
  });
  group.add(new THREE.Mesh(geometry, material));

  const coreGeometry = new THREE.IcosahedronGeometry(1, 1);
  const coreMaterial = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.08 });
  group.add(new THREE.Mesh(coreGeometry, coreMaterial));

  return {
    group,
    material,
    color,
    present: false,
    grabbing: false,
    homePos,
    targetPos: homePos.clone(),
    targetScale: 1,
    grabStartHandQuat: null,
    grabStartObjQuat: null,
    prevHandQuat: null,
    prevPos: null,
    angularVelocity: new THREE.Vector3(),
    linearVelocity: new THREE.Vector3(),
  };
}

const holograms: HandHologram[] = [
  createHologram(HAND_COLORS[0], -1.8),
  createHologram(HAND_COLORS[1], 1.8),
];

const PINCH_ON = 0.045; // normalized thumb-index distance below which a grab starts
const PINCH_OFF = 0.065; // above which a grab releases (hysteresis avoids flicker at the edge)
const AMBIENT_SPIN = 0.15; // rad/s idle rotation when nothing is grabbing
const LINEAR_DAMPING = 2.2; // higher = fling stops sooner
const ANGULAR_DAMPING = 2.6; // higher = spin momentum stops sooner
const VELOCITY_EPSILON = 0.02;

// A vector from landmark `a` to landmark `b`, converted into the mirrored,
// Y-up world space the scene uses (video is mirrored and image Y points down).
function worldDelta(a: Landmark, b: Landmark): THREE.Vector3 {
  return new THREE.Vector3(-(b.x - a.x), -(b.y - a.y), -(b.z - a.z));
}

// Builds an orientation quaternion for the hand from its palm landmarks, so
// "grabbing" the hologram and turning your wrist turns the hologram the same way.
function handOrientation(lm: HandLandmarks): THREE.Quaternion {
  const wrist = lm[0];
  const indexMcp = lm[5];
  const middleMcp = lm[9];
  const pinkyMcp = lm[17];

  const right = worldDelta(pinkyMcp, indexMcp).normalize();
  let up = worldDelta(wrist, middleMcp).normalize();
  const forward = new THREE.Vector3().crossVectors(right, up).normalize();
  up = new THREE.Vector3().crossVectors(forward, right).normalize();

  const m = new THREE.Matrix4().makeBasis(right, up, forward);
  return new THREE.Quaternion().setFromRotationMatrix(m);
}

function updateHologramFromHand(h: HandHologram, lm: HandLandmarks, dt: number) {
  h.present = true;
  const wrist = lm[0];
  const thumbTip = lm[4];
  const indexTip = lm[8];
  const indexMcp = lm[5];
  const pinkyMcp = lm[17];

  // Hand position in world space, used both for hover-follow and as the grab anchor.
  const nx = (1 - wrist.x) * 2 - 1;
  const ny = -(wrist.y * 2 - 1);
  const handPos = new THREE.Vector3(nx * 3, ny * 2, 0);
  h.targetPos.copy(handPos);

  // Palm width is a stable proxy for hand distance from the camera (bigger = closer),
  // so reaching toward the screen zooms in, like inspecting a prototype up close.
  const palmWidth = Math.hypot(indexMcp.x - pinkyMcp.x, indexMcp.y - pinkyMcp.y, indexMcp.z - pinkyMcp.z);
  h.targetScale = THREE.MathUtils.clamp(THREE.MathUtils.mapLinear(palmWidth, 0.07, 0.22, 0.5, 2.6), 0.5, 2.6);

  const pinchDist = Math.hypot(thumbTip.x - indexTip.x, thumbTip.y - indexTip.y, thumbTip.z - indexTip.z);
  const threshold = h.grabbing ? PINCH_OFF : PINCH_ON;
  const shouldGrab = pinchDist < threshold;

  if (shouldGrab && !h.grabbing) {
    // Pinch just started: lock in the reference orientation/position to grab from.
    h.grabbing = true;
    h.grabStartHandQuat = handOrientation(lm);
    h.grabStartObjQuat = h.group.quaternion.clone();
    h.prevHandQuat = h.grabStartHandQuat.clone();
    h.prevPos = h.group.position.clone();
    h.angularVelocity.set(0, 0, 0);
    h.linearVelocity.set(0, 0, 0);
  } else if (!shouldGrab && h.grabbing) {
    // Released: whatever rotational/linear velocity we had carries on and decays.
    h.grabbing = false;
    h.grabStartHandQuat = null;
    h.grabStartObjQuat = null;
    h.prevHandQuat = null;
    h.prevPos = null;
  }

  if (h.grabbing && h.grabStartHandQuat && h.grabStartObjQuat) {
    const currentHandQuat = handOrientation(lm);

    // Rotation follows the hand 1:1 relative to where the grab started.
    const deltaQuat = currentHandQuat.clone().multiply(h.grabStartHandQuat.clone().invert());
    h.group.quaternion.copy(deltaQuat.multiply(h.grabStartObjQuat));

    // Position is dragged directly by the hand while grabbing.
    h.group.position.lerp(handPos, 0.6);

    if (dt > 0 && h.prevHandQuat) {
      // Track angular velocity (axis-angle/sec) from frame-to-frame rotation, so a
      // fast twist right before release keeps spinning ("flicking") the hologram.
      const frameDelta = currentHandQuat.clone().multiply(h.prevHandQuat.clone().invert());
      const angle = 2 * Math.acos(THREE.MathUtils.clamp(frameDelta.w, -1, 1));
      if (angle > 1e-4) {
        const s = Math.sqrt(1 - frameDelta.w * frameDelta.w) || 1;
        const axis = new THREE.Vector3(frameDelta.x / s, frameDelta.y / s, frameDelta.z / s);
        h.angularVelocity.copy(axis.multiplyScalar(angle / dt));
      } else {
        h.angularVelocity.set(0, 0, 0);
      }
    }
    if (dt > 0 && h.prevPos) {
      h.linearVelocity.copy(h.group.position.clone().sub(h.prevPos).divideScalar(dt));
    }
    h.prevHandQuat = currentHandQuat;
    h.prevPos = h.group.position.clone();
  }
}

function drawSkeleton(result: HandLandmarkerResult, grabbing: boolean[]) {
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  result.landmarks.forEach((lm, i) => {
    const color = grabbing[i] ? '255, 204, 51' : i === 0 ? '0, 229, 255' : '255, 90, 210';
    overlayCtx.strokeStyle = `rgba(${color}, 0.9)`;
    overlayCtx.lineWidth = 2;
    overlayCtx.beginPath();
    for (const [a, b] of HAND_CONNECTIONS) {
      const pa = lm[a];
      const pb = lm[b];
      overlayCtx.moveTo(pa.x * overlay.width, pa.y * overlay.height);
      overlayCtx.lineTo(pb.x * overlay.width, pb.y * overlay.height);
    }
    overlayCtx.stroke();

    overlayCtx.fillStyle = `rgba(${color}, 1)`;
    for (const p of lm) {
      overlayCtx.beginPath();
      overlayCtx.arc(p.x * overlay.width, p.y * overlay.height, 3, 0, Math.PI * 2);
      overlayCtx.fill();
    }
  });
}

// ---------- Render / animation loop ----------
const clock = new THREE.Clock();
function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), 0.1);

  for (const h of holograms) {
    if (h.grabbing) {
      // Rotation/position already applied directly in updateHologramFromHand.
    } else if (h.present) {
      // Hovering: drift smoothly toward the hand and idle-spin.
      h.group.position.lerp(h.targetPos, 0.15);
      h.group.rotateY(AMBIENT_SPIN * dt);
    } else {
      // No hand tracked: apply leftover momentum (the "flick"), then settle home.
      const speed = h.linearVelocity.length() + h.angularVelocity.length();
      if (speed > VELOCITY_EPSILON) {
        h.group.position.addScaledVector(h.linearVelocity, dt);
        const angle = h.angularVelocity.length() * dt;
        if (angle > 1e-5) {
          const axis = h.angularVelocity.clone().normalize();
          h.group.quaternion.premultiply(new THREE.Quaternion().setFromAxisAngle(axis, angle));
        }
        const linearDecay = Math.exp(-LINEAR_DAMPING * dt);
        const angularDecay = Math.exp(-ANGULAR_DAMPING * dt);
        h.linearVelocity.multiplyScalar(linearDecay);
        h.angularVelocity.multiplyScalar(angularDecay);
      } else {
        h.linearVelocity.set(0, 0, 0);
        h.angularVelocity.set(0, 0, 0);
        h.group.position.lerp(h.homePos, 0.05);
        h.group.rotateY(AMBIENT_SPIN * dt);
        h.targetScale = 1;
      }
    }

    const nextScale = THREE.MathUtils.lerp(h.group.scale.x, h.targetScale, 0.15);
    h.group.scale.setScalar(nextScale);

    const targetColor = h.grabbing ? GRAB_COLOR : h.color;
    h.material.color.lerp(new THREE.Color(targetColor), 0.2);
    const targetOpacity = h.present || h.linearVelocity.length() + h.angularVelocity.length() > VELOCITY_EPSILON ? 0.85 : 0.2;
    h.material.opacity = THREE.MathUtils.lerp(h.material.opacity, targetOpacity, 0.1);
  }

  renderer.render(scene3, camera);
}

// ---------- MediaPipe HandLandmarker setup ----------
async function setupHandTracking() {
  statusEl.textContent = 'Loading model…';
  const fileset = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
  );
  const handLandmarker = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath:
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
      delegate: 'GPU',
    },
    runningMode: 'VIDEO',
    numHands: MAX_HANDS,
  });

  statusEl.textContent = 'Requesting camera…';
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 1280, height: 720, facingMode: 'user' },
  });
  video.srcObject = stream;
  await new Promise<void>((resolve) => {
    video.onloadedmetadata = () => resolve();
  });
  video.play();
  resize();
  animate();

  statusEl.textContent = 'Pinch to grab & turn, drag to move, release to flick, reach in to zoom';

  let lastTime = -1;
  let lastDetectMs = performance.now();
  function detect() {
    if (video.currentTime !== lastTime) {
      lastTime = video.currentTime;
      const now = performance.now();
      const dt = Math.min((now - lastDetectMs) / 1000, 0.1);
      lastDetectMs = now;
      const result = handLandmarker.detectForVideo(video, now);

      const grabbing: boolean[] = [];
      for (let i = 0; i < holograms.length; i++) {
        const lm = result.landmarks[i];
        if (lm) {
          updateHologramFromHand(holograms[i], lm as HandLandmarks, dt);
        } else {
          holograms[i].present = false;
          if (holograms[i].grabbing) {
            holograms[i].grabbing = false;
            holograms[i].grabStartHandQuat = null;
            holograms[i].grabStartObjQuat = null;
            holograms[i].prevHandQuat = null;
            holograms[i].prevPos = null;
          }
        }
        grabbing.push(holograms[i].grabbing);
      }
      drawSkeleton(result, grabbing);
    }
    requestAnimationFrame(detect);
  }
  detect();
}

setupHandTracking().catch((err) => {
  console.error(err);
  statusEl.textContent = `Error: ${err.message ?? err}`;
});
