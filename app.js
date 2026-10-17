import {
  HandLandmarker,
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// ---- 手势 -> 猫咪图映射 -------------------------------------------------
const GESTURE_MEMES = {
  rockstar: ["memes/cat.jpg"],
  default: ["memes/pokercat.jpg"],
  oneFingerUp: ["memes/profcat.jpg", "memes/professorcat.jpg"],
  fist: ["memes/punchcat.jpg"],
  shhh: ["memes/shhcat.jpg"],
  twoFingersTogether: [
    "memes/uwucat.jpg",
    "memes/uwucatt.jpg",
    "memes/fingers together muehehe .jpg",
  ],
  handCoverFace: ["memes/hand cover face .jpg"],
  crashOutCat: ["memes/crashout cat .jpg"],
  twoHandsOnHead: ["memes/two hands on head .jpg"],
  handStretchedOut: ["memes/hand stretched out, palm facing up .jpg"],
  sideEyeCat: ["memes/side eye cat.jpg"],
};

// 手势名中文显示
const GESTURE_LABEL_ZH = {
  rockstar: "摇滚 🤘",
  default: "待机 🐱",
  oneFingerUp: "一指向上 ☝️",
  fist: "拳头 ✊",
  shhh: "嘘 🤫",
  twoFingersTogether: "双指并拢 🤏",
  handCoverFace: "捂脸 🤲",
  crashOutCat: "崩溃猫 😱",
  twoHandsOnHead: "抱头 🙆",
  handStretchedOut: "摊手 🖐️",
  sideEyeCat: "斜眼 👀",
};

// 防抖：连续多少帧命中同一手势才切换
const STABLE_FRAMES_REQUIRED = 5;
// 多久没看到手就回到默认
const DEFAULT_FALLBACK_MS = 600;
// 人脸丢失后多久内还信任旧的人脸框
const FACE_STALE_MS = 1200;
// 转头多少度算斜眼
const SIDE_EYE_YAW_DEG = 15.0;
// 手靠近脸的判定距离
const HAND_COVER_FACE_DIST_FACE_LOST = 1.3;
const HAND_COVER_FACE_DIST_FACE_SEEN = 0.7;

const video = document.getElementById("video");
const memeImg = document.getElementById("memeImg");
const debugHud = document.getElementById("debugHud");

let handLandmarker, faceLandmarker;
let lastVideoTime = -1;
let currentGesture = "default";
let candidateGesture = "default";
let candidateStreak = 0;
let lastNonDefaultAt = performance.now();
let lastFace = null;
let lastFaceSeenThisFrame = false;
let lastYawDebug = 0;

async function init() {
  const fileset = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  handLandmarker = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 2,
  });

  faceLandmarker = await FaceLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numFaces: 1,
    outputFacialTransformationMatrixes: true,
  });

  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480 },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();

  requestAnimationFrame(loop);
}

// ---- 几何工具 ---------------------------------------------------------
function vec(a, b) {
  return { x: b.x - a.x, y: b.y - a.y, z: (b.z || 0) - (a.z || 0) };
}
function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y, (a.z || 0) - (b.z || 0));
}
function angleDeg(v1, v2) {
  const dot = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
  const m1 = Math.hypot(v1.x, v1.y, v1.z);
  const m2 = Math.hypot(v2.x, v2.y, v2.z);
  if (m1 < 1e-9 || m2 < 1e-9) return 180;
  return (Math.acos(Math.min(1, Math.max(-1, dot / (m1 * m2)))) * 180) / Math.PI;
}

function fingerExtended(lm, mcp, pip, tip) {
  const angle = angleDeg(vec(lm[mcp], lm[pip]), vec(lm[pip], lm[tip]));
  return angle < 45;
}

function yawFromTransformMatrix(matrixData) {
  const r00 = matrixData[0];
  const r10 = matrixData[4];
  const r20 = matrixData[8];
  const sy = Math.hypot(r00, r10);
  if (sy < 1e-6) return 0;
  return (Math.atan2(-r20, sy) * 180) / Math.PI;
}

function classifyHand(lm) {
  const handScale = dist(lm[0], lm[9]) || 1e-6;
  const indexUp = fingerExtended(lm, 5, 6, 8);
  const middleUp = fingerExtended(lm, 9, 10, 12);
  const ringUp = fingerExtended(lm, 13, 14, 16);
  const pinkyUp = fingerExtended(lm, 17, 18, 20);
  const thumbPinkySpread = dist(lm[4], lm[17]) / handScale;
  const thumbOut = thumbPinkySpread > 1.05;
  const curledCount = [indexUp, middleUp, ringUp, pinkyUp].filter((v) => !v).length;
  return {
    indexUp, middleUp, ringUp, pinkyUp, thumbOut, curledCount, handScale,
    indexTip: lm[8], wrist: lm[0], palmCenter: lm[9],
  };
}

function updateFace(faceResult) {
  const now = performance.now();
  const sawFace = !!(faceResult.faceLandmarks && faceResult.faceLandmarks.length > 0);
  if (sawFace) {
    const f = faceResult.faceLandmarks[0];
    const upperLip = f[13];
    const lowerLip = f[14];
    const rightCheek = f[234];
    const leftCheek = f[454];
    const mouthCenter = {
      x: (upperLip.x + lowerLip.x) / 2,
      y: (upperLip.y + lowerLip.y) / 2,
      z: ((upperLip.z || 0) + (lowerLip.z || 0)) / 2,
    };
    const faceWidth = dist(rightCheek, leftCheek);
    const mouthOpen = dist(upperLip, lowerLip) / faceWidth;
    let yawDeg = 0;
    if (faceResult.facialTransformationMatrixes && faceResult.facialTransformationMatrixes.length > 0) {
      yawDeg = yawFromTransformMatrix(faceResult.facialTransformationMatrixes[0].data);
    }
    lastFace = { mouthCenter, faceWidth, mouthOpen, yawDeg, t: now };
    lastYawDebug = yawDeg;
  }
  lastFaceSeenThisFrame = sawFace;
}

function isPointing(h) {
  return h.indexUp && !h.middleUp && !h.ringUp && !h.pinkyUp;
}

function decideGesture(handResult) {
  const now = performance.now();
  const faceIsFresh = !!lastFace && now - lastFace.t < FACE_STALE_MS;

  if (!handResult.landmarks || handResult.landmarks.length === 0) {
    if (faceIsFresh && Math.abs(lastFace.yawDeg) > SIDE_EYE_YAW_DEG) {
      return "sideEyeCat";
    }
    return "default";
  }

  const hands = handResult.landmarks.map(classifyHand);

  if (hands.length === 2) {
    if (isPointing(hands[0]) && isPointing(hands[1])) {
      const avgScale = (hands[0].handScale + hands[1].handScale) / 2;
      const tipGap = dist(hands[0].indexTip, hands[1].indexTip) / avgScale;
      if (tipGap < 1.4) return "twoFingersTogether";
    }
    if (faceIsFresh) {
      const { mouthCenter, faceWidth } = lastFace;
      const nearFace = hands.every((h) => dist(h.palmCenter, mouthCenter) / faceWidth < 2.2);
      if (nearFace) {
        const headTopY = mouthCenter.y - faceWidth * 1.1;
        const bothAboveHead = hands.every((h) => h.palmCenter.y < headTopY);
        if (bothAboveHead) return "twoHandsOnHead";
        return "crashOutCat";
      }
    }
  }

  const h = hands[0];
  if (h.curledCount === 4) return "fist";
  if (h.thumbOut && h.pinkyUp && !h.indexUp && !h.middleUp && !h.ringUp) return "rockstar";

  if (h.indexUp && !h.middleUp && !h.ringUp && !h.pinkyUp) {
    if (faceIsFresh) {
      const d = dist(h.indexTip, lastFace.mouthCenter) / lastFace.faceWidth;
      if (d < 0.55) return "shhh";
    }
    return "oneFingerUp";
  }

  if (faceIsFresh) {
    const d = dist(h.palmCenter, lastFace.mouthCenter) / lastFace.faceWidth;
    const threshold = lastFaceSeenThisFrame ? HAND_COVER_FACE_DIST_FACE_SEEN : HAND_COVER_FACE_DIST_FACE_LOST;
    if (d < threshold) return "handCoverFace";
  }

  if (h.curledCount === 0) return "handStretchedOut";

  if (faceIsFresh && Math.abs(lastFace.yawDeg) > SIDE_EYE_YAW_DEG) {
    return "sideEyeCat";
  }

  return "default";
}

function pickImage(gesture) {
  const images = GESTURE_MEMES[gesture];
  return images[Math.floor(Math.random() * images.length)];
}

function applyGesture(gesture) {
  if (gesture === currentGesture) return;
  currentGesture = gesture;
  memeImg.src = pickImage(gesture);
}

function loop() {
  const now = performance.now();
  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const ts = performance.now();
    const handResult = handLandmarker.detectForVideo(video, ts);
    const faceResult = faceLandmarker.detectForVideo(video, ts);
    updateFace(faceResult);
    const gesture = decideGesture(handResult);

    if (gesture === candidateGesture) candidateStreak++;
    else { candidateGesture = gesture; candidateStreak = 1; }
    if (candidateStreak >= STABLE_FRAMES_REQUIRED) applyGesture(gesture);

    if (gesture !== "default") lastNonDefaultAt = now;
    if (now - lastNonDefaultAt > DEFAULT_FALLBACK_MS && currentGesture !== "default") {
      applyGesture("default");
    }
    updateDebugHud();
  }
  requestAnimationFrame(loop);
}

function updateDebugHud() {
  if (!debugHud) return;
  const label = GESTURE_LABEL_ZH[currentGesture] || currentGesture;
  debugHud.textContent =
    `手势: ${label}\n` +
    `转头: ${lastYawDebug >= 0 ? "+" : ""}${lastYawDebug.toFixed(1)} 度  (斜视阈值 ±${SIDE_EYE_YAW_DEG.toFixed(1)})`;
}

init().catch((err) => console.error(err));
