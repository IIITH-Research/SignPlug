import os
from pathlib import Path

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.environ["XDG_CACHE_HOME"] = "/tmp"
Path("/tmp/matplotlib").mkdir(parents=True, exist_ok=True)

import cv2
import mediapipe as mp
import numpy as np


UPPER_BODY_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
N_BODY = len(UPPER_BODY_INDICES)
VIS_THRESHOLD = 0.6


def extract_keypoints(video_path, output_path=None, save=True):
    """Extract upper-body and hand world landmarks from a video."""
    video_path = Path(video_path)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    mp_holistic = mp.solutions.holistic
    mp_hands = mp.solutions.hands

    holistic_model = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        refine_face_landmarks=False,
    )

    hands_model = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        holistic_model.close()
        hands_model.close()
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {width}x{height} at {fps} FPS, {total_frames} frames")

    all_body_world = []
    all_body_vis = []
    all_body_flag = []
    all_left_hand = []
    all_right_hand = []
    all_left_flag = []
    all_right_flag = []

    frame_idx = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video reached.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            holistic_result = holistic_model.process(rgb_frame)
            hands_result = hands_model.process(rgb_frame)

            body_detected = False
            if holistic_result.pose_world_landmarks:
                all_lm = holistic_result.pose_world_landmarks.landmark

                body_world = np.array(
                    [[all_lm[i].x, all_lm[i].y, all_lm[i].z] for i in UPPER_BODY_INDICES],
                    dtype=np.float32,
                )

                body_vis = np.array(
                    [all_lm[i].visibility for i in UPPER_BODY_INDICES],
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

            left_hand_wld = np.zeros((21, 3), dtype=np.float32)
            right_hand_wld = np.zeros((21, 3), dtype=np.float32)
            left_detected = False
            right_detected = False

            if hands_result.multi_hand_world_landmarks and hands_result.multi_handedness:
                for hand_idx in range(len(hands_result.multi_hand_world_landmarks)):
                    handedness = hands_result.multi_handedness[hand_idx].classification[0]
                    label = handedness.label
                    confidence = handedness.score

                    if confidence < 0.5:
                        continue

                    hand_world = np.array(
                        [
                            [lm.x, lm.y, lm.z]
                            for lm in hands_result.multi_hand_world_landmarks[hand_idx].landmark
                        ],
                        dtype=np.float32,
                    )

                    if label == "Left":
                        right_hand_wld = hand_world
                        right_detected = True
                    elif label == "Right":
                        left_hand_wld = hand_world
                        left_detected = True

            all_left_hand.append(left_hand_wld)
            all_right_hand.append(right_hand_wld)
            all_left_flag.append(left_detected)
            all_right_flag.append(right_detected)

            frame_idx += 1
            if frame_idx % 50 == 0:
                print(f"Processed frame {frame_idx}/{total_frames}")
    finally:
        cap.release()
        holistic_model.close()
        hands_model.close()

    body_world_seq = np.stack(all_body_world, axis=0)
    body_vis_seq = np.stack(all_body_vis, axis=0)
    body_flag_seq = np.array(all_body_flag, dtype=bool)
    left_hand_seq = np.stack(all_left_hand, axis=0)
    right_hand_seq = np.stack(all_right_hand, axis=0)
    left_flag_seq = np.array(all_left_flag, dtype=bool)
    right_flag_seq = np.array(all_right_flag, dtype=bool)
    body_joint_valid = body_vis_seq >= VIS_THRESHOLD

    print(f"\nTotal frames        : {frame_idx}")
    print(f"Body detected       : {body_flag_seq.sum()}")
    print(f"Left hand detected  : {left_flag_seq.sum()}")
    print(f"Right hand detected : {right_flag_seq.sum()}")

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
    }

    if save and output_path is not None:
        np.savez(output_path, **data)
        print(f"\nSaved to {output_path}")

    return data


if __name__ == "__main__":
    extract_keypoints("Apple.mp4", "apple_raw_landmarks.npz")
