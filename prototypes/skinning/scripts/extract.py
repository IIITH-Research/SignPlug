import os
import sys
from pathlib import Path
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
import numpy as np

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.environ["XDG_CACHE_HOME"] = "/tmp"
Path("/tmp/matplotlib").mkdir(parents=True, exist_ok=True)

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "src" / "videos"
OUTPUT_DIR = ROOT_DIR / "src" / "output"
MODELS_DIR = ROOT_DIR / "src" / "models"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

UPPER_BODY_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
N_BODY = len(UPPER_BODY_INDICES)
VIS_THRESHOLD = 0.6

# Require tasks
hand_model_path = str(MODELS_DIR / 'hand_landmarker.task')
pose_model_path = str(MODELS_DIR / 'pose_landmarker.task')
if not os.path.exists(hand_model_path) or not os.path.exists(pose_model_path):
    print("Error: Model task files not found. Run `python setup.py` first.")
    sys.exit(1)

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def process_media(media_path, output_path, is_video=True):
    print(f"\n--- Processing: {media_path.name} ---")
    cap = cv2.VideoCapture(str(media_path))
    if not cap.isOpened():
        print(f"Error: Could not open {media_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) if is_video else 30.0
    if fps <= 0 or fps != fps: fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_video else 1

    mode = VisionRunningMode.VIDEO if is_video else VisionRunningMode.IMAGE
    
    # We add the confidences to closely match the user's 0.5 strict tracking parameters
    hand_options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=hand_model_path),
        running_mode=mode,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5)

    pose_options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=pose_model_path),
        running_mode=mode,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False)

    all_body_world, all_body_vis, all_body_flag = [], [], []
    all_left_hand, all_right_hand = [], []
    all_left_flag, all_right_flag = [], []
    
    with vision.HandLandmarker.create_from_options(hand_options) as hand_landmarker, \
         vision.PoseLandmarker.create_from_options(pose_options) as pose_landmarker:

        frame_idx = 0
        last_ms = -1
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC)) if is_video else 0
            if timestamp_ms <= last_ms:
                timestamp_ms = last_ms + 1
            last_ms = timestamp_ms

            if is_video:
                pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
                hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
            else:
                pose_result = pose_landmarker.detect(mp_image)
                hand_result = hand_landmarker.detect(mp_image)

            # Body
            body_detected = False
            if pose_result.pose_world_landmarks:
                all_lm = pose_result.pose_world_landmarks[0]
                body_world = np.array(
                    [[all_lm[i].x, all_lm[i].y, all_lm[i].z] for i in UPPER_BODY_INDICES],
                    dtype=np.float32,
                )
                
                # Check for explicit visibility attribute if missing default to 1.0
                body_vis = np.array(
                    [(getattr(all_lm[i], 'visibility', 1.0) or 1.0) for i in UPPER_BODY_INDICES],
                    dtype=np.float32,
                )
                low_vis_mask = body_vis < VIS_THRESHOLD
                body_world[low_vis_mask] = 0.0
                body_detected = True
            else:
                body_world = np.zeros((N_BODY, 3), dtype=np.float32)
                body_vis = np.zeros((N_BODY,), dtype=np.float32)

            all_body_world.append(body_world)
            all_body_vis.append(body_vis)
            all_body_flag.append(body_detected) 

            # Hands
            left_hand_wld = np.zeros((21, 3), dtype=np.float32)
            right_hand_wld = np.zeros((21, 3), dtype=np.float32)
            left_detected, right_detected = False, False

            if hand_result.hand_world_landmarks:
                for idx, hand in enumerate(hand_result.handedness):
                    label = hand[0].category_name
                    score = hand[0].score
                    if score < 0.5: continue
                    
                    landmarks = hand_result.hand_world_landmarks[idx]
                    hand_world = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

                    # ALIGN HANDS TO BODY WRISTS to fix the disjoint/floating origin issue:
                    # MediaPipe hand world landmarks are origin-centered (0,0,0 at wrist)
                    # We translate them to snap onto the Pose world wrists.
                    if label == "Left":
                        if body_detected and body_vis[15] >= VIS_THRESHOLD:
                            wrist_pose = body_world[15]
                            hand_world += (wrist_pose - hand_world[0])
                        left_hand_wld = hand_world
                        left_detected = True
                    elif label == "Right":
                        if body_detected and body_vis[16] >= VIS_THRESHOLD:
                            wrist_pose = body_world[16]
                            hand_world += (wrist_pose - hand_world[0])
                        right_hand_wld = hand_world
                        right_detected = True

            all_left_hand.append(left_hand_wld)
            all_right_hand.append(right_hand_wld)
            all_left_flag.append(left_detected)
            all_right_flag.append(right_detected)

            frame_idx += 1
            if is_video and frame_idx % 50 == 0:
                print(f"Processed frame {frame_idx}/{total_frames}")
                
            if not is_video: break
            
    cap.release()

    if frame_idx == 0:
        print("No frames extracted.")
        return None

    body_world_seq = np.stack(all_body_world, axis=0)
    body_vis_seq = np.stack(all_body_vis, axis=0)
    body_flag_seq = np.array(all_body_flag, dtype=bool)
    left_hand_seq = np.stack(all_left_hand, axis=0)
    right_hand_seq = np.stack(all_right_hand, axis=0)
    left_flag_seq = np.array(all_left_flag, dtype=bool)
    right_flag_seq = np.array(all_right_flag, dtype=bool)
    body_joint_valid = body_vis_seq >= VIS_THRESHOLD

    data = {
        "body_world": body_world_seq,
        "body_vis": body_vis_seq,
        "body_flag": body_flag_seq,
        "body_joint_valid": body_joint_valid,
        "left_hand": left_hand_seq,
        "right_hand": right_hand_seq,
        "left_flag": left_flag_seq,
        "right_flag": right_flag_seq,
        "upper_body_indices": np.array(UPPER_BODY_INDICES),
        "is_video": np.array([is_video])
    }

    np.savez(output_path, **data)
    print(f"Saved to {output_path.name} (Frames: {frame_idx})")
    return data

def main():
    video_exts = {".mp4", ".avi", ".mov", ".mkv"}
    image_exts = {".jpg", ".jpeg", ".png"}

    target_file = sys.argv[1] if len(sys.argv) > 1 else None

    search_dirs = [DATA_DIR, ROOT_DIR / "src" / "intepolation_code" / "videos"]
    found_files = []
    
    for s_dir in search_dirs:
        if s_dir.exists():
            for file_path in s_dir.iterdir():
                if file_path.is_file():
                    if target_file and file_path.name.lower() != target_file.lower() and file_path.stem.lower() != target_file.lower():
                        continue
                    ext = file_path.suffix.lower()
                    if ext in video_exts:
                        found_files.append((file_path, True))
                    elif ext in image_exts:
                        found_files.append((file_path, False))
            
            # If we found the target file in this directory, we don't need to search the other directories
            if found_files and target_file:
                break

    if not found_files:
        print(f"No media found matching '{target_file or ''}' in {[str(d.relative_to(ROOT_DIR)) for d in search_dirs]}. Please add videos or images.")
        return

    for media_path, is_video in found_files:
        out_file = OUTPUT_DIR / f"{media_path.stem}_raw.npz"
        process_media(media_path, out_file, is_video=is_video)

if __name__ == "__main__":
    main()