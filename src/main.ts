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
const debugEl = document.getElementById('debug')!;

const HAND_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

const MAX_HANDS = 2;
const OBJECT_COLORS = [0x66ffff, 0xff66d9, 0x8cff66, 0xffaa33];
const GRAB_COLOR = 0xffcc33;
const HOVER_HINT_COLOR = 0xffffff;

// Pinch is measured as thumb-to-index distance DIVIDED BY palm width, not as a raw
// absolute distance — a raw threshold only works at one specific distance from the
// camera, since the same physical pinch produces smaller landmark distances the
// farther your hand is from the lens. The ratio is roughly distance-invariant.
const PINCH_ON_RATIO = 0.55; // pinch starts once thumb+index closer than this fraction of palm width
const PINCH_OFF_RATIO = 0.75; // releases above this fraction (hysteresis avoids flicker at the edge)
// Two separate radii: claiming a fresh idle object needs to be tight so neighboring
// grid slots don't overlap (half the grid spacing is ~1.1-1.9 units). Joining an
// object your OTHER hand already holds can be far more generous — that's checked
// and preferred first, so bringing your second hand in to combine doesn't
// accidentally land in an adjacent object's claim zone instead.
const FREE_GRAB_RADIUS = 1.0; // world units to claim an idle object
const COMBINE_RADIUS = 2.5; // world units to join an object your other hand holds
const AMBIENT_SPIN = 0.15; // rad/s idle rotation when nothing holds the object
const LINEAR_DAMPING = 2.2; // higher = fling stops sooner
const ANGULAR_DAMPING = 2.6; // higher = spin momentum stops sooner
const VELOCITY_EPSILON = 0.02;
// Safety clamp on release velocity — even a deliberate flick shouldn't send an
// object rocketing off screen, and this also caps damage from any stray spike.
const MAX_LINEAR_VELOCITY = 6; // world units/sec
const MAX_ANGULAR_VELOCITY = 12; // rad/sec

// A hand disappearing from tracking briefly (motion blur, going just out of frame,
// a momentary detection miss) shouldn't drop its hold outright — that's what was
// causing objects to fly off or lose their spot. Only release after it's been
// missing longer than this, and treat that as a lost grip (no momentum), not a throw.
const LOST_TRACKING_GRACE = 0.35; // seconds

// Exponential-smoothing rates (higher = snappier/less lag, lower = smoother/more lag).
// Using a rate + dt (rather than a flat lerp factor) keeps the feel consistent
// even if the camera's frame rate varies.
const LANDMARK_SMOOTH_RATE = 14; // filters raw hand-tracking jitter
const GRAB_POS_RATE = 16; // how tightly a held object follows its hand(s)
const SCALE_RATE = 10; // zoom smoothing

function smoothFactor(rate: number, dt: number): number {
  return 1 - Math.exp(-rate * dt);
}

// ---------- Three.js scene ----------
const renderer = new THREE.WebGLRenderer({ canvas: sceneCanvas, alpha: true, antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);

const scene3 = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 100);
camera.position.set(0, 0, 7);

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

// Converts a landmark's normalized position within the raw video frame into a
// world-space point on the z=0 plane, using the camera's real projection (not a
// hardcoded scale) — this is what makes a hologram visually line up with your hand.
const posRaycaster = new THREE.Raycaster();
const zPlane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);

function landmarkToWorldPos(lm: Landmark): THREE.Vector3 {
  // `object-fit: cover` scales the video to fill the window and crops whichever
  // axis overflows, so normalized landmark coords (relative to the FULL video
  // buffer) must first be remapped to the VISIBLE (post-crop) area.
  const videoAspect = (video.videoWidth || 16) / (video.videoHeight || 9);
  const containerAspect = window.innerWidth / window.innerHeight;
  let vx = lm.x;
  let vy = lm.y;
  if (videoAspect > containerAspect) {
    const visibleFrac = containerAspect / videoAspect;
    const offset = (1 - visibleFrac) / 2;
    vx = (lm.x - offset) / visibleFrac;
  } else {
    const visibleFrac = videoAspect / containerAspect;
    const offset = (1 - visibleFrac) / 2;
    vy = (lm.y - offset) / visibleFrac;
  }

  // Mirror X (the video is displayed mirrored) and convert to NDC (-1..1).
  const ndcX = (1 - vx) * 2 - 1;
  const ndcY = -(vy * 2 - 1);

  posRaycaster.setFromCamera(new THREE.Vector2(ndcX, ndcY), camera);
  const hit = new THREE.Vector3();
  posRaycaster.ray.intersectPlane(zPlane, hit);
  return hit;
}

// ---------- Grid of hologram "parts" ----------
interface HologramObject {
  id: number;
  group: THREE.Group;
  material: THREE.MeshBasicMaterial;
  color: number;
  homePos: THREE.Vector3;

  holders: number[]; // hand-track ids currently holding this object (0, 1, or 2 of them)
  targetScale: number;

  // Solo-hold reference frame (used while holders.length === 1).
  soloGrabHandQuat: THREE.Quaternion | null;
  soloGrabObjQuat: THREE.Quaternion | null;
  prevHandQuat: THREE.Quaternion | null;
  prevPos: THREE.Vector3 | null;

  // Joint-hold reference frame (used while holders.length === 2, for the "combine" gesture).
  jointStartCombinedQuat: THREE.Quaternion | null;
  jointStartObjQuat: THREE.Quaternion | null;
  jointStartHandDist: number | null;
  jointStartObjScale: number | null;

  // Momentum carried after a solo release, decays over time (the "flick"/"spin" feel).
  angularVelocity: THREE.Vector3;
  linearVelocity: THREE.Vector3;
}

function createHologramObject(id: number, color: number, homePos: THREE.Vector3): HologramObject {
  const group = new THREE.Group();
  group.position.copy(homePos);
  scene3.add(group);

  const geometry = new THREE.IcosahedronGeometry(0.85, 1);
  const material = new THREE.MeshBasicMaterial({ color, wireframe: true, transparent: true, opacity: 0.35 });
  group.add(new THREE.Mesh(geometry, material));

  const coreGeometry = new THREE.IcosahedronGeometry(0.85, 1);
  const coreMaterial = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.08 });
  group.add(new THREE.Mesh(coreGeometry, coreMaterial));

  return {
    id,
    group,
    material,
    color,
    homePos: homePos.clone(),
    holders: [],
    targetScale: 1,
    soloGrabHandQuat: null,
    soloGrabObjQuat: null,
    prevHandQuat: null,
    prevPos: null,
    jointStartCombinedQuat: null,
    jointStartObjQuat: null,
    jointStartHandDist: null,
    jointStartObjScale: null,
    angularVelocity: new THREE.Vector3(),
    linearVelocity: new THREE.Vector3(),
  };
}

// 2x2 grid of independent objects laid out on the "field" in front of the camera.
const GRID_POSITIONS = [
  new THREE.Vector3(-1.9, 1.1, 0),
  new THREE.Vector3(1.9, 1.1, 0),
  new THREE.Vector3(-1.9, -1.1, 0),
  new THREE.Vector3(1.9, -1.1, 0),
];
const objects: HologramObject[] = GRID_POSITIONS.map((pos, i) =>
  createHologramObject(i, OBJECT_COLORS[i % OBJECT_COLORS.length], pos)
);

// ---------- Per-hand tracking ----------
interface HandTrack {
  id: number;
  present: boolean;
  pinching: boolean;
  heldObjectId: number | null;
  smoothedHandPos: THREE.Vector3 | null;
  smoothedPalmWidth: number | null;
  smoothedQuat: THREE.Quaternion | null;
  lastPinchRatio: number | null; // for the debug HUD
  missingTime: number; // seconds since this track was last actually seen
}

function createHandTrack(id: number): HandTrack {
  return {
    id,
    present: false,
    pinching: false,
    heldObjectId: null,
    smoothedHandPos: null,
    smoothedPalmWidth: null,
    smoothedQuat: null,
    lastPinchRatio: null,
    missingTime: 0,
  };
}

const handTracks: HandTrack[] = [createHandTrack(0), createHandTrack(1)];

// A vector from landmark `a` to landmark `b`, converted into the mirrored,
// Y-up world space the scene uses (video is mirrored and image Y points down).
function worldDelta(a: Landmark, b: Landmark): THREE.Vector3 {
  return new THREE.Vector3(-(b.x - a.x), -(b.y - a.y), -(b.z - a.z));
}

function rawWristWorldPos(lm: HandLandmarks): THREE.Vector3 {
  return landmarkToWorldPos(lm[0]);
}

// Builds an orientation quaternion for the hand from its palm landmarks, so
// "grabbing" an object and turning your wrist turns the object the same way.
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

// Matches this frame's raw detections to the persistent hand tracks by nearest
// wrist position, so a track's identity (and whatever it's holding) stays stable
// even though MediaPipe's own array ordering can flip between hands frame to frame.
function assignHandTracks(rawHands: HandLandmarks[]): (HandLandmarks | null)[] {
  const assignment: (HandLandmarks | null)[] = [null, null];
  if (rawHands.length === 0) return assignment;

  const rawPos = rawHands.map(rawWristWorldPos);

  if (rawHands.length === 1) {
    let best = 0;
    let bestDist = Infinity;
    for (const track of handTracks) {
      const d = track.smoothedHandPos ? track.smoothedHandPos.distanceTo(rawPos[0]) : Infinity;
      if (d < bestDist) {
        bestDist = d;
        best = track.id;
      }
    }
    assignment[best] = rawHands[0];
    return assignment;
  }

  // Two raw hands, two tracks: pick whichever pairing minimizes total movement.
  const d00 = handTracks[0].smoothedHandPos?.distanceTo(rawPos[0]) ?? 999;
  const d11 = handTracks[1].smoothedHandPos?.distanceTo(rawPos[1]) ?? 999;
  const d01 = handTracks[0].smoothedHandPos?.distanceTo(rawPos[1]) ?? 999;
  const d10 = handTracks[1].smoothedHandPos?.distanceTo(rawPos[0]) ?? 999;

  if (d00 + d11 <= d01 + d10) {
    assignment[0] = rawHands[0];
    assignment[1] = rawHands[1];
  } else {
    assignment[0] = rawHands[1];
    assignment[1] = rawHands[0];
  }
  return assignment;
}

function updateHandTrack(track: HandTrack, lm: HandLandmarks, dt: number) {
  track.present = true;
  const thumbTip = lm[4];
  const indexTip = lm[8];
  const indexMcp = lm[5];
  const pinkyMcp = lm[17];

  const rawHandPos = rawWristWorldPos(lm);
  const rawPalmWidth = Math.hypot(indexMcp.x - pinkyMcp.x, indexMcp.y - pinkyMcp.y, indexMcp.z - pinkyMcp.z);
  const rawQuat = handOrientation(lm);

  const lmFactor = smoothFactor(LANDMARK_SMOOTH_RATE, dt);
  if (!track.smoothedHandPos) track.smoothedHandPos = rawHandPos.clone();
  else track.smoothedHandPos.lerp(rawHandPos, lmFactor);

  track.smoothedPalmWidth =
    track.smoothedPalmWidth === null ? rawPalmWidth : THREE.MathUtils.lerp(track.smoothedPalmWidth, rawPalmWidth, lmFactor);

  if (!track.smoothedQuat) track.smoothedQuat = rawQuat.clone();
  else track.smoothedQuat.slerp(rawQuat, lmFactor);

  const pinchDist = Math.hypot(thumbTip.x - indexTip.x, thumbTip.y - indexTip.y, thumbTip.z - indexTip.z);
  const pinchRatio = pinchDist / (track.smoothedPalmWidth || rawPalmWidth || 1e-4);
  track.lastPinchRatio = pinchRatio;
  const threshold = track.pinching ? PINCH_OFF_RATIO : PINCH_ON_RATIO;
  track.pinching = pinchRatio < threshold;
}

// Finds the object this (already-pinching) hand should claim or join. Each object
// gets its own eligible radius (joining an object your other hand already holds
// can reach a bit further than claiming a fresh idle one), but candidates are
// compared by ACTUAL distance across both categories — whichever object is
// genuinely closest wins. Previously "joinable" objects were checked first as a
// category, so an idle object you were clearly hovering over could lose out to a
// held object that was merely "in range" but farther away.
function findClaimableObject(track: HandTrack): HologramObject | null {
  if (!track.smoothedHandPos) return null;

  let best: HologramObject | null = null;
  let bestDist = Infinity;
  for (const obj of objects) {
    const isJoinable = obj.holders.length === 1 && !obj.holders.includes(track.id);
    const isFree = obj.holders.length === 0;
    if (!isJoinable && !isFree) continue;

    const radius = isJoinable ? COMBINE_RADIUS : FREE_GRAB_RADIUS;
    const d = obj.group.position.distanceTo(track.smoothedHandPos);
    if (d < radius && d < bestDist) {
      bestDist = d;
      best = obj;
    }
  }
  return best;
}

function releaseHold(track: HandTrack, keepMomentum: boolean) {
  if (track.heldObjectId === null) return;
  const obj = objects.find((o) => o.id === track.heldObjectId);
  track.heldObjectId = null;
  if (!obj) return;
  obj.holders = obj.holders.filter((id) => id !== track.id);

  if (!keepMomentum && obj.holders.length === 0) {
    // Lost tracking, not a deliberate throw: freeze in place instead of flinging.
    obj.linearVelocity.set(0, 0, 0);
    obj.angularVelocity.set(0, 0, 0);
  }

  if (obj.holders.length === 1) {
    // Was joint, now back to solo: start a fresh solo reference frame so there's no pop.
    const remaining = handTracks.find((t) => t.id === obj.holders[0]);
    if (remaining && remaining.smoothedQuat) {
      obj.soloGrabHandQuat = remaining.smoothedQuat.clone();
      obj.soloGrabObjQuat = obj.group.quaternion.clone();
      obj.prevHandQuat = obj.soloGrabHandQuat.clone();
      obj.prevPos = obj.group.position.clone();
    }
    obj.jointStartCombinedQuat = null;
    obj.jointStartObjQuat = null;
    obj.jointStartHandDist = null;
    obj.jointStartObjScale = null;
  } else if (obj.holders.length === 0) {
    // Fully released: momentum (captured just before this) carries it, then it settles in place.
    obj.soloGrabHandQuat = null;
    obj.soloGrabObjQuat = null;
    obj.prevHandQuat = null;
    obj.prevPos = null;
  }
}

function claimHold(track: HandTrack, obj: HologramObject) {
  track.heldObjectId = obj.id;
  obj.holders.push(track.id);
  obj.angularVelocity.set(0, 0, 0);
  obj.linearVelocity.set(0, 0, 0);

  if (obj.holders.length === 1 && track.smoothedQuat) {
    obj.soloGrabHandQuat = track.smoothedQuat.clone();
    obj.soloGrabObjQuat = obj.group.quaternion.clone();
    obj.prevHandQuat = obj.soloGrabHandQuat.clone();
    obj.prevPos = obj.group.position.clone();
  } else if (obj.holders.length === 2) {
    const [aId, bId] = obj.holders;
    const a = handTracks.find((t) => t.id === aId);
    const b = handTracks.find((t) => t.id === bId);
    if (a?.smoothedQuat && b?.smoothedQuat && a.smoothedHandPos && b.smoothedHandPos) {
      obj.jointStartCombinedQuat = a.smoothedQuat.clone().slerp(b.smoothedQuat, 0.5);
      obj.jointStartObjQuat = obj.group.quaternion.clone();
      obj.jointStartHandDist = a.smoothedHandPos.distanceTo(b.smoothedHandPos);
      obj.jointStartObjScale = obj.group.scale.x;
    }
  }
}

function updateGrabsAndHolds(dt: number) {
  // Resolve grab/release transitions first.
  for (const track of handTracks) {
    if (!track.present) {
      track.missingTime += dt;
      // Brief tracking loss (motion blur, a frame the detector missed) doesn't drop
      // the hold — the object just stays exactly where it was. Only a hand that's
      // truly gone for a while releases, and without any fling (it wasn't a throw).
      if (track.heldObjectId !== null && track.missingTime > LOST_TRACKING_GRACE) {
        releaseHold(track, false);
      }
      continue;
    }
    track.missingTime = 0;
    const wantsHold = track.pinching;
    if (wantsHold && track.heldObjectId === null) {
      const target = findClaimableObject(track);
      if (target) claimHold(track, target);
    } else if (!wantsHold && track.heldObjectId !== null) {
      releaseHold(track, true);
    }
  }

  // Apply transforms for currently-held objects.
  for (const obj of objects) {
    if (obj.holders.length === 1) {
      const track = handTracks.find((t) => t.id === obj.holders[0]);
      if (!track?.smoothedQuat || !track.smoothedHandPos || !obj.soloGrabHandQuat || !obj.soloGrabObjQuat) continue;

      const currentHandQuat = track.smoothedQuat;
      const deltaQuat = currentHandQuat.clone().multiply(obj.soloGrabHandQuat.clone().invert());
      obj.group.quaternion.copy(deltaQuat.multiply(obj.soloGrabObjQuat));
      obj.group.position.lerp(track.smoothedHandPos, smoothFactor(GRAB_POS_RATE, dt));
      obj.targetScale = THREE.MathUtils.clamp(
        THREE.MathUtils.mapLinear(track.smoothedPalmWidth ?? 0.14, 0.07, 0.22, 0.5, 2.6),
        0.5,
        2.6
      );

      if (dt > 0 && obj.prevHandQuat) {
        const frameDelta = currentHandQuat.clone().multiply(obj.prevHandQuat.clone().invert());
        const angle = 2 * Math.acos(THREE.MathUtils.clamp(frameDelta.w, -1, 1));
        if (angle > 1e-4) {
          const s = Math.sqrt(1 - frameDelta.w * frameDelta.w) || 1;
          const axis = new THREE.Vector3(frameDelta.x / s, frameDelta.y / s, frameDelta.z / s);
          const speed = Math.min(angle / dt, MAX_ANGULAR_VELOCITY);
          obj.angularVelocity.copy(axis.multiplyScalar(speed));
        } else {
          obj.angularVelocity.set(0, 0, 0);
        }
      }
      if (dt > 0 && obj.prevPos) {
        const vel = obj.group.position.clone().sub(obj.prevPos).divideScalar(dt);
        if (vel.length() > MAX_LINEAR_VELOCITY) vel.setLength(MAX_LINEAR_VELOCITY);
        obj.linearVelocity.copy(vel);
      }
      obj.prevHandQuat = currentHandQuat.clone();
      obj.prevPos = obj.group.position.clone();
    } else if (obj.holders.length === 2) {
      const [aId, bId] = obj.holders;
      const a = handTracks.find((t) => t.id === aId);
      const b = handTracks.find((t) => t.id === bId);
      if (!a?.smoothedQuat || !b?.smoothedQuat || !a.smoothedHandPos || !b.smoothedHandPos) continue;
      if (!obj.jointStartCombinedQuat || !obj.jointStartObjQuat || !obj.jointStartHandDist || !obj.jointStartObjScale) continue;

      // Rotation: the average of both hands' orientation, relative to where the joint grab started.
      const currentCombinedQuat = a.smoothedQuat.clone().slerp(b.smoothedQuat, 0.5);
      const deltaQuat = currentCombinedQuat.clone().multiply(obj.jointStartCombinedQuat.clone().invert());
      obj.group.quaternion.copy(deltaQuat.multiply(obj.jointStartObjQuat));

      // Position: the midpoint between both hands.
      const midpoint = a.smoothedHandPos.clone().add(b.smoothedHandPos).multiplyScalar(0.5);
      obj.group.position.lerp(midpoint, smoothFactor(GRAB_POS_RATE, dt));

      // Scale: pull your hands apart to stretch it bigger, like inspecting a part up close.
      const currentDist = a.smoothedHandPos.distanceTo(b.smoothedHandPos);
      obj.targetScale = THREE.MathUtils.clamp(
        obj.jointStartObjScale * (currentDist / obj.jointStartHandDist),
        0.4,
        3.5
      );
    }
  }
}

// Live readout of what the tracker actually sees, since pinch/grab tuning is
// impossible to get right blind — this makes the raw numbers visible so the
// PINCH_ON_RATIO / GRAB_RADIUS constants above can be tuned against reality.
function updateDebugHud() {
  const lines = handTracks.map((track) => {
    if (!track.present) return `hand ${track.id}: not detected`;
    const ratio = track.lastPinchRatio !== null ? track.lastPinchRatio.toFixed(2) : '?';
    let nearestId = '-';
    let nearestDist = Infinity;
    if (track.smoothedHandPos) {
      for (const obj of objects) {
        const d = obj.group.position.distanceTo(track.smoothedHandPos);
        if (d < nearestDist) {
          nearestDist = d;
          nearestId = String(obj.id);
        }
      }
    }
    const held = track.heldObjectId !== null ? `HOLDING #${track.heldObjectId}` : track.pinching ? 'pinching, nothing in range' : 'open';
    return `hand ${track.id}: pinch ${ratio} (grab<${PINCH_ON_RATIO}) | nearest #${nearestId} @ ${nearestDist.toFixed(2)} (free<${FREE_GRAB_RADIUS} join<${COMBINE_RADIUS}) | ${held}`;
  });
  debugEl.textContent = lines.join('\n');
}

function drawSkeleton(rawHands: HandLandmarks[], trackByRawIndex: (HandTrack | null)[]) {
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  rawHands.forEach((lm, i) => {
    const track = trackByRawIndex[i];
    const held = track && track.heldObjectId !== null;
    const color = held ? '255, 204, 51' : track?.id === 1 ? '255, 90, 210' : '0, 229, 255';
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

  for (const obj of objects) {
    if (obj.holders.length > 0) {
      // Transform already applied in updateGrabsAndHolds.
    } else {
      // Not held: apply leftover momentum (the "flick"), then rest in place.
      const speed = obj.linearVelocity.length() + obj.angularVelocity.length();
      if (speed > VELOCITY_EPSILON) {
        obj.group.position.addScaledVector(obj.linearVelocity, dt);
        const angle = obj.angularVelocity.length() * dt;
        if (angle > 1e-5) {
          const axis = obj.angularVelocity.clone().normalize();
          obj.group.quaternion.premultiply(new THREE.Quaternion().setFromAxisAngle(axis, angle));
        }
        obj.linearVelocity.multiplyScalar(Math.exp(-LINEAR_DAMPING * dt));
        obj.angularVelocity.multiplyScalar(Math.exp(-ANGULAR_DAMPING * dt));
      } else {
        obj.linearVelocity.set(0, 0, 0);
        obj.angularVelocity.set(0, 0, 0);
        obj.group.rotateY(AMBIENT_SPIN * dt);
        // targetScale is left as-is: a released object keeps whatever size you left it at.
      }
    }

    const nextScale = THREE.MathUtils.lerp(obj.group.scale.x, obj.targetScale, smoothFactor(SCALE_RATE, dt));
    obj.group.scale.setScalar(nextScale);

    // Nearest claimable object to any unheld, pinching-adjacent hand gets a hover hint.
    const isHeld = obj.holders.length > 0;
    let isHoverHint = false;
    if (!isHeld) {
      for (const track of handTracks) {
        if (track.present && track.smoothedHandPos && track.heldObjectId === null) {
          if (obj.group.position.distanceTo(track.smoothedHandPos) < FREE_GRAB_RADIUS) isHoverHint = true;
        }
      }
    }

    const targetColor = isHeld ? GRAB_COLOR : isHoverHint ? HOVER_HINT_COLOR : obj.color;
    obj.material.color.lerp(new THREE.Color(targetColor), 0.2);
    const targetOpacity = isHeld ? 0.9 : isHoverHint ? 0.55 : 0.3;
    obj.material.opacity = THREE.MathUtils.lerp(obj.material.opacity, targetOpacity, 0.1);
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
    // Detection confidence (finding a hand-shaped region in the first place) needs
    // to stay strict, or faces/beards/other skin-toned blobs get misread as hands.
    // Presence/tracking confidence (staying locked onto a hand already found) can
    // stay lower — that's what was needed for pinch occlusion, and doesn't cause
    // false positives since it only applies to a region already confirmed as a hand.
    minHandDetectionConfidence: 0.6,
    minHandPresenceConfidence: 0.4,
    minTrackingConfidence: 0.4,
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

  statusEl.textContent = 'Pinch near a part to grab it — both hands on the same part to combine it';

  let lastTime = -1;
  let lastDetectMs = performance.now();
  function detect() {
    if (video.currentTime !== lastTime) {
      lastTime = video.currentTime;
      const now = performance.now();
      const dt = Math.min((now - lastDetectMs) / 1000, 0.1);
      lastDetectMs = now;
      const result: HandLandmarkerResult = handLandmarker.detectForVideo(video, now);
      const rawHands = result.landmarks as HandLandmarks[];

      const assigned = assignHandTracks(rawHands);
      const trackByRawIndex: (HandTrack | null)[] = rawHands.map(() => null);

      for (const track of handTracks) {
        const lm = assigned[track.id];
        if (lm) {
          updateHandTrack(track, lm, dt);
          const rawIndex = rawHands.indexOf(lm);
          if (rawIndex >= 0) trackByRawIndex[rawIndex] = track;
        } else {
          track.present = false;
          track.pinching = false;
        }
      }

      updateGrabsAndHolds(dt);
      drawSkeleton(rawHands, trackByRawIndex);
      updateDebugHud();
    }
    requestAnimationFrame(detect);
  }
  detect();
}

setupHandTracking().catch((err) => {
  console.error(err);
  statusEl.textContent = `Error: ${err.message ?? err}`;
});
