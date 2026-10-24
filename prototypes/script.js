// ISL Avatar - JSON clip renderer
// ------------------------------------------------------------
// This file loads a JSON "animation clip" produced by extract.py
// and renders pose + hand keypoints as moving dots at ~30 FPS.
//
// Why Canvas2D?
// - It's the simplest possible renderer.
// - Once this pipeline works end-to-end, upgrading to lines/bones
//   or a full avatar is "just engineering".

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

// Hardcoded clip for the milestone (one gloss: APPLE)
const CLIP_URL = "clips/apple.json";

// Target playback FPS (independent of screen refresh rate)
const FPS = 30;
const FRAME_INTERVAL_MS = 1000 / FPS;

let frames = [];
let currentFrame = 0;
let lastTime = 0;

// --- Stick-figure connections ("bones") ---
// Each pair [a, b] means "draw a line from landmark a to landmark b".
// This is the standard way to visualize MediaPipe landmarks.

// Pose has 33 landmarks (0..32)
const POSE_CONNECTIONS = [
  // Face-ish (helps orientation)
  [0, 1], [1, 2], [2, 3], [3, 7],
  [0, 4], [4, 5], [5, 6], [6, 8],
  [9, 10],

  // Torso
  [11, 12],
  [11, 23], [12, 24], [23, 24],

  // Left arm
  [11, 13], [13, 15],
  [15, 17], [15, 19], [15, 21],
  [17, 19],

  // Right arm
  [12, 14], [14, 16],
  [16, 18], [16, 20], [16, 22],
  [18, 20],

  // Left leg
  [23, 25], [25, 27],
  [27, 29], [29, 31],
  [27, 31],

  // Right leg
  [24, 26], [26, 28],
  [28, 30], [30, 32],
  [28, 32],
];

// Hand has 21 landmarks (0..20)
const HAND_CONNECTIONS = [
  // Thumb
  [0, 1], [1, 2], [2, 3], [3, 4],
  // Index
  [0, 5], [5, 6], [6, 7], [7, 8],
  // Middle
  [0, 9], [9, 10], [10, 11], [11, 12],
  // Ring
  [0, 13], [13, 14], [14, 15], [15, 16],
  // Pinky
  [0, 17], [17, 18], [18, 19], [19, 20],
  // Palm links
  [5, 9], [9, 13], [13, 17],
];

// --- Small rendering helpers ---

function setupHiDPI() {
  // Optimization/quality: scale the backing store for sharp points on HiDPI screens.
  // This doesn't change the UX; it just avoids blurry rendering.
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const cssWidth = canvas.clientWidth;
  const cssHeight = canvas.clientHeight;

  canvas.width = Math.round(cssWidth * dpr);
  canvas.height = Math.round(cssHeight * dpr);

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  // After setTransform, draw in CSS pixel coordinates.
}

function clear() {
  ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
}

function drawPointPx(x, y, radius = 3) {
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fillStyle = "lime";
  ctx.fill();
}

function getXY(p) {
  // Our JSON stores landmarks as either:
  // - object: {x, y, z, v}
  // - array : [x, y, z] or [x, y, z, v]
  const x = Array.isArray(p) ? p[0] : p.x;
  const y = Array.isArray(p) ? p[1] : p.y;

  if (typeof x !== "number" || typeof y !== "number") return null;
  return { x, y };
}

function drawPointsNormalized(points, radius = 3) {
  // MediaPipe landmarks are normalized (x,y) in [0,1] relative to the image.
  // We map them into our canvas.
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;

  for (const p of points) {
    const xy = getXY(p);
    if (!xy) continue;
    drawPointPx(xy.x * w, xy.y * h, radius);
  }
}

function drawConnectionsNormalized(points, connections, lineWidth = 2) {
  // Draw line segments between landmark indices to form a stick figure.
  if (!points || points.length === 0) return;

  const w = canvas.clientWidth;
  const h = canvas.clientHeight;

  ctx.beginPath();

  for (const [a, b] of connections) {
    const pa = points[a];
    const pb = points[b];
    if (!pa || !pb) continue;

    const axy = getXY(pa);
    const bxy = getXY(pb);
    if (!axy || !bxy) continue;

    ctx.moveTo(axy.x * w, axy.y * h);
    ctx.lineTo(bxy.x * w, bxy.y * h);
  }

  ctx.strokeStyle = "lime";
  ctx.lineWidth = lineWidth;
  ctx.stroke();
}

function renderFrame(frame) {
  clear();

  // frame.pose / left_hand / right_hand may be [] if MediaPipe failed that frame.
  // Draw bones first, then points on top.

  if (frame.pose && frame.pose.length) {
    drawConnectionsNormalized(frame.pose, POSE_CONNECTIONS, 2);
    drawPointsNormalized(frame.pose, 2.5);
  }

  if (frame.left_hand && frame.left_hand.length) {
    // If we have the full 21-point hand, draw bones.
    // If we exported "tips" only (6 points), draw just the points.
    if (frame.left_hand.length >= 21) {
      drawConnectionsNormalized(frame.left_hand, HAND_CONNECTIONS, 2);
    }
    drawPointsNormalized(frame.left_hand, 3);
  }

  if (frame.right_hand && frame.right_hand.length) {
    if (frame.right_hand.length >= 21) {
      drawConnectionsNormalized(frame.right_hand, HAND_CONNECTIONS, 2);
    }
    drawPointsNormalized(frame.right_hand, 3);
  }

  // Face (468 points) — draw as tiny dots so it looks like a dense face mesh.
  // This is intentionally minimal: points only (no extra UI).
  if (frame.face && frame.face.length) {
    // Slightly dimmer so it doesn't overpower the hands.
    // Dense mesh: tiny points. Minimal pose-face cues: slightly larger.
    const prevAlpha = ctx.globalAlpha;
    ctx.globalAlpha = 0.75;
    const r = frame.face.length > 50 ? 1.2 : 2.4;
    drawPointsNormalized(frame.face, r);
    ctx.globalAlpha = prevAlpha;
  }
}

function animate(timestamp) {
  if (!frames.length) return;

  if (timestamp - lastTime >= FRAME_INTERVAL_MS) {
    renderFrame(frames[currentFrame]);

    currentFrame++;
    if (currentFrame >= frames.length) currentFrame = 0;

    lastTime = timestamp;
  }

  requestAnimationFrame(animate);
}

async function loadClip() {
  setupHiDPI();

  try {
    const res = await fetch(CLIP_URL);
    if (!res.ok) throw new Error(`HTTP ${res.status} loading ${CLIP_URL}`);

    const data = await res.json();

    // extract.py outputs { meta: {...}, frames: [...] }
    // but we also support the "array only" format.
    frames = Array.isArray(data) ? data : data.frames;

    if (!Array.isArray(frames) || frames.length === 0) {
      throw new Error("Clip JSON loaded but contains no frames");
    }

    currentFrame = 0;
    lastTime = 0;
    requestAnimationFrame(animate);
  } catch (err) {
    clear();
    ctx.fillStyle = "white";
    ctx.font = "16px system-ui";
    ctx.fillText("Failed to load clip. Open DevTools Console.", 20, 40);
    // eslint-disable-next-line no-console
    console.error(err);
  }
}

window.addEventListener("resize", () => {
  setupHiDPI();
});

loadClip();
