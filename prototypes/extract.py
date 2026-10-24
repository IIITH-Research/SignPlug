"""extract.py

Video → MediaPipe keypoints → JSON animation clip.

This is the *foundation* step for the whole avatar system:
- Read a sign-language video file
- Detect pose + hands landmarks for each (sampled) frame
- Serialize those landmarks as a time series JSON "clip"
- Later: load the clip in the browser and render/skin an avatar

This extractor supports two practical modes:

1) **Dense debug mode** (default): saves pose + hands + full face mesh
    (typically 478 face points on this model). Great for verifying tracking.

2) **Avatar2D minimal mode** (recommended for instant web rendering):
    saves pose + hands, and uses ONLY the *pose face landmarks* (nose/eyes/ears/mouth)
    for head/face cues. This is *far* smaller than a full face mesh.

Face options (size controls):
- `--face-mode mesh` (full face mesh)
- `--face-mode pose` (11 pose face points; best for 2D avatars)
- `--face-mode none` (no face)
- `--face-every-n` (when using mesh: keep every Nth point)

Hand options (size controls):
- `--hand-mode full` (21 landmarks; best for sign language accuracy)
- `--hand-mode tips` (6 landmarks: wrist + fingertips; best for 2D avatar performance)

Important Windows/Python note (beginner-friendly):
- On Windows + Python 3.13, the older `mp.solutions.*` API is often not available.
- This project uses the newer **MediaPipe Tasks** API (`mp.tasks.*`), which works
    on your current Python 3.13 install.

The first run will download a small `.task` model file into `models/`.

Usage examples:
  python extract.py
  python extract.py --input dataset/Apple.mp4 --output clips/apple.json
  python extract.py --target-fps 30 --precision 5

Notes for beginners:
- MediaPipe returns *normalized* coordinates (x,y) in [0,1] relative to the image.
- Some frames will have missing hands/pose (occlusion, blur, low light). That's normal.

"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from urllib.request import urlretrieve

import cv2
import mediapipe as mp


@dataclass
class ClipMeta:
    source: str
    source_fps: float
    target_fps: float
    frame_count_source: int
    frame_count_output: int
    pose_landmarks: int = 33
    hand_landmarks: int = 21
    # Face point count depends on face_mode:
    # - mesh: usually 478 on this model (468 + iris)
    # - pose: 11 (nose/eyes/ears/mouth) from the pose landmarks
    # - none: 0
    face_landmarks: int = 0
    preset: str = "dense"
    face_mode: str = "mesh"
    face_every_n: int = 1
    hand_mode: str = "full"
    format: str = "isl_clip_v1"


# --- Model download (one-time) ---

# Official model hosted by Google (used by MediaPipe sample apps).
HOLISTIC_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/1/holistic_landmarker.task"
)


def _ensure_model(model_path: str, url: str) -> None:
    """Ensure a model file exists locally; download if missing."""

    if os.path.exists(model_path):
        return

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    print(f"Model not found. Downloading to: {model_path}")
    print(f"From: {url}")

    # Simple download. If your network blocks this, download manually and re-run.
    urlretrieve(url, model_path)
    print("Model download complete.")


def _round_float(x: float, precision: int) -> float:
    # Optimization: rounding reduces JSON size significantly with negligible visual impact.
    return float(round(float(x), precision))


def _extract_pose_landmarks(pose_landmarks: List[Any], precision: int) -> List[List[float]]:
    """Pose landmarks: 33 points. We store [x, y, z, visibility]."""
    if not pose_landmarks:
        return []

    out: List[List[float]] = []
    for lm in pose_landmarks:
        # Tasks API provides visibility for pose; if not present, default to 1.0.
        v = getattr(lm, "visibility", 1.0)
        out.append(
            [
                _round_float(lm.x, precision),
                _round_float(lm.y, precision),
                _round_float(lm.z, precision),
                _round_float(v, precision),
            ]
        )
    return out


def _extract_hand_landmarks(hand_landmarks: List[Any], precision: int) -> List[List[float]]:
    """Hand landmarks: 21 points. We store [x, y, z]."""
    if not hand_landmarks:
        return []

    out: List[List[float]] = []
    for lm in hand_landmarks:
        out.append(
            [
                _round_float(lm.x, precision),
                _round_float(lm.y, precision),
                _round_float(lm.z, precision),
            ]
        )
    return out


def _extract_hand_tips(hand_landmarks: List[Any], precision: int) -> List[List[float]]:
    """Reduced hand representation for 2D avatar performance.

    Keeps 6 points:
    - 0: wrist
    - 4, 8, 12, 16, 20: fingertips (thumb/index/middle/ring/pinky)
    """

    if not hand_landmarks:
        return []

    keep = [0, 4, 8, 12, 16, 20]
    out: List[List[float]] = []
    for i in keep:
        if i >= len(hand_landmarks):
            continue
        lm = hand_landmarks[i]
        out.append(
            [
                _round_float(lm.x, precision),
                _round_float(lm.y, precision),
                _round_float(lm.z, precision),
            ]
        )
    return out


def _extract_face_landmarks(
    face_landmarks: List[Any],
    precision: int,
    every_n: int,
) -> List[List[float]]:
    """Face landmarks: 468 points.

    Stored as [x, y, z].
    Optimization: keep only every Nth point via `every_n`.
    """

    if not face_landmarks:
        return []

    if every_n < 1:
        every_n = 1

    out: List[List[float]] = []
    for i, lm in enumerate(face_landmarks):
        if i % every_n != 0:
            continue
        out.append(
            [
                _round_float(lm.x, precision),
                _round_float(lm.y, precision),
                _round_float(lm.z, precision),
            ]
        )
    return out


def _extract_face_from_pose(pose_landmarks: List[Any], precision: int) -> List[List[float]]:
    """Minimal face cues derived from pose landmarks.

    Pose indices (11 points total):
    0 nose
    1-6 eyes (inner/center/outer)
    7-8 ears
    9-10 mouth corners

    This is usually enough to:
    - position the head
    - estimate head tilt/rotation in 2D (using eyes/ears)
    without storing the full 478-point face mesh.
    """

    if not pose_landmarks or len(pose_landmarks) < 11:
        return []

    idxs = list(range(0, 11))
    out: List[List[float]] = []
    for i in idxs:
        lm = pose_landmarks[i]
        out.append(
            [
                _round_float(lm.x, precision),
                _round_float(lm.y, precision),
                _round_float(lm.z, precision),
            ]
        )
    return out


def _make_timestamps(
    cap: cv2.VideoCapture,
    frame_index_1based: int,
    source_fps: float,
    fallback_step_msec: float,
    last_t_msec: float,
) -> float:
    """Get a robust timestamp for the current frame.

    MediaPipe VIDEO mode requires monotonically increasing timestamps.
    Some OpenCV builds return 0 for CAP_PROP_POS_MSEC, so we fall back.
    """

    t_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
    if t_msec <= 0.0:
        if source_fps > 0:
            t_msec = (float(frame_index_1based - 1) / source_fps) * 1000.0
        else:
            t_msec = last_t_msec + fallback_step_msec

    # Ensure strictly increasing.
    if t_msec <= last_t_msec:
        t_msec = last_t_msec + 1.0

    return t_msec


def extract_clip(
    input_path: str,
    output_path: str,
    target_fps: float,
    precision: int,
    preset: str,
    face_mode: str,
    face_precision: int,
    face_every_n: int,
    hand_mode: str,
    model_path: str,
    max_frames: Optional[int] = None,
) -> None:
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Input video not found: {input_path}. Put your mp4 in the repo or pass --input."
        )

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(
            "OpenCV could not open the video. If you downloaded it, ensure it's a valid mp4."
        )

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count_source = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # We sample frames by timestamp to get close to a stable target fps.
    # This is more robust than "take every Nth frame".
    next_sample_msec = 0.0
    sample_interval_msec = 1000.0 / float(target_fps)

    # Download the model if missing.
    _ensure_model(model_path, HOLISTIC_MODEL_URL)

    # MediaPipe Tasks API setup (works on Python 3.13).
    BaseOptions = mp.tasks.BaseOptions
    HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
    HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        output_face_blendshapes=False,
        output_segmentation_mask=False,
        # Defaults are generally fine; you can expose these as CLI args later.
    )

    landmarker = HolisticLandmarker.create_from_options(options)

    frames: List[Dict[str, Any]] = []
    frames_read = 0
    last_t_msec = -1.0

    try:
        while cap.isOpened():
            success, frame_bgr = cap.read()
            if not success:
                break

            frames_read += 1

            # Timestamp of *current* frame in milliseconds.
            t_msec = _make_timestamps(
                cap=cap,
                frame_index_1based=frames_read,
                source_fps=source_fps,
                fallback_step_msec=sample_interval_msec,
                last_t_msec=last_t_msec,
            )
            last_t_msec = t_msec

            # Skip frames until it's time for the next sample.
            if t_msec + 1e-6 < next_sample_msec:
                continue

            # Convert BGR → RGB (MediaPipe expects SRGB)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Convert numpy array → MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # Run holistic for this video frame.
            # We save pose + hands always.
            # Face can be dense (mesh), minimal (from pose), or disabled.
            results = landmarker.detect_for_video(mp_image, int(round(t_msec)))

            pose_pts = _extract_pose_landmarks(results.pose_landmarks, precision)

            if hand_mode == "tips":
                left_hand_pts = _extract_hand_tips(results.left_hand_landmarks, precision)
                right_hand_pts = _extract_hand_tips(results.right_hand_landmarks, precision)
            else:
                left_hand_pts = _extract_hand_landmarks(results.left_hand_landmarks, precision)
                right_hand_pts = _extract_hand_landmarks(results.right_hand_landmarks, precision)

            if face_mode == "none":
                face_pts: List[List[float]] = []
            elif face_mode == "pose":
                # Derive minimal face from the first 11 pose landmarks (0..10)
                face_pts = _extract_face_from_pose(results.pose_landmarks, face_precision)
            else:
                # Dense face mesh
                face_pts = _extract_face_landmarks(results.face_landmarks, face_precision, face_every_n)

            frame_landmarks = {
                "pose": pose_pts,
                "left_hand": left_hand_pts,
                "right_hand": right_hand_pts,
                "face": face_pts,
            }

            frames.append(frame_landmarks)
            next_sample_msec += sample_interval_msec

            if max_frames is not None and len(frames) >= max_frames:
                break

            # Lightweight progress print every ~100 output frames.
            if len(frames) % 100 == 0:
                print(f"Processed {len(frames)} sampled frames...")

    finally:
        cap.release()
        landmarker.close()

    # Declare expected face point count for the chosen mode.
    # (The actual per-frame list may be empty when tracking fails.)
    if face_mode == "mesh":
        face_count = 478  # this model outputs 478 (face mesh + iris)
    elif face_mode == "pose":
        face_count = 11
    else:
        face_count = 0

    hand_count = 6 if hand_mode == "tips" else 21

    meta = ClipMeta(
        source=input_path.replace("\\", "/"),
        source_fps=source_fps,
        target_fps=float(target_fps),
        frame_count_source=frame_count_source,
        frame_count_output=len(frames),
        hand_landmarks=hand_count,
        face_landmarks=face_count,
        preset=str(preset),
        face_mode=str(face_mode),
        face_every_n=int(face_every_n) if face_mode == "mesh" else 0,
        hand_mode=str(hand_mode),
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Optimization: compact JSON (no whitespace) to keep file smaller.
    payload = {"meta": asdict(meta), "frames": frames}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))

    print(f"Saved clip: {output_path}")
    print(f"Output frames: {len(frames)} @ target_fps={target_fps}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract pose + hand keypoints into a JSON clip")
    parser.add_argument(
        "--input",
        default="dataset/Apple.mp4",
        help="Path to input video (default: dataset/Apple.mp4)",
    )
    parser.add_argument(
        "--output",
        default="clips/apple.json",
        help="Output JSON clip path (default: clips/apple.json)",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=30.0,
        help="Sample the video to this FPS (default: 30)",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=5,
        help="Decimal precision for saved floats (default: 5). Lower = smaller JSON.",
    )
    parser.add_argument(
        "--preset",
        choices=["dense", "avatar2d"],
        default="dense",
        help=(
            "Export preset: dense = full debug (pose+hands+face mesh), "
            "avatar2d = minimal for fast web skinning"
        ),
    )
    parser.add_argument(
        "--face-mode",
        choices=["mesh", "pose", "none"],
        default="mesh",
        help=(
            "Face landmark mode: mesh (dense), pose (11 face cues from pose), none (disable)"
        ),
    )
    parser.add_argument(
        "--face-precision",
        type=int,
        default=4,
        help="Decimal precision for face floats (default: 4). Lower = smaller JSON.",
    )
    parser.add_argument(
        "--face-every-n",
        type=int,
        default=1,
        help="(mesh mode only) Keep only every Nth face point (default: 1 = keep all)",
    )
    parser.add_argument(
        "--hand-mode",
        choices=["full", "tips"],
        default="full",
        help="Hand mode: full (21 points) or tips (6 points: wrist+fingertips)",
    )
    parser.add_argument(
        "--model",
        default="models/holistic_landmarker.task",
        help="Path to the MediaPipe HolisticLandmarker .task model (default: models/holistic_landmarker.task)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Debug option: stop after N sampled frames",
    )

    args = parser.parse_args()

    # Apply preset defaults (can still be overridden by explicit flags).
    preset = args.preset
    face_mode = args.face_mode
    hand_mode = args.hand_mode

    if preset == "avatar2d":
        # Minimal + fast defaults for a skinned 2D avatar.
        # - Face: use pose face cues (11 points)
        # - Hands: tips only (6 points)
        if args.face_mode == "mesh":
            face_mode = "pose"
        if args.hand_mode == "full":
            hand_mode = "tips"

    extract_clip(
        input_path=args.input,
        output_path=args.output,
        target_fps=args.target_fps,
        precision=args.precision,
        preset=preset,
        face_mode=face_mode,
        face_precision=args.face_precision,
        face_every_n=args.face_every_n,
        hand_mode=hand_mode,
        model_path=args.model,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
