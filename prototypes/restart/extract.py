import cv2
import json
import urllib.request
import os
import glob

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- Directory Configuration ---
MODELS_DIR = "models"
VIDEOS_DIR = "videos"
OUTPUT_DIR = "output"

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_file(url, filename):
    filepath = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(filepath):
        print(f"Downloading {filename} to {MODELS_DIR}/...")
        urllib.request.urlretrieve(url, filepath)
        print(f"Downloaded {filename}")
    return filepath

# Download/locate the required task files inside the models directory
face_model_path = download_file('https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task', 'face_landmarker.task')
hand_model_path = download_file('https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task', 'hand_landmarker.task')
pose_model_path = download_file('https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task', 'pose_landmarker.task')

# Create landmarker options
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

face_options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=face_model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1)

hand_options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=hand_model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2)

pose_options = vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=pose_model_path),
    running_mode=VisionRunningMode.VIDEO,
    output_segmentation_masks=False)

def extract_features(video_path, output_data_path):
    print(f"\n--- Processing: {os.path.basename(video_path)} ---")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps: # Handle NaN or 0 FPS
        fps = 30.0
        
    all_frames_data = []

    print("Initializing MediaPipe models...")
    with vision.FaceLandmarker.create_from_options(face_options) as face_landmarker, \
         vision.HandLandmarker.create_from_options(hand_options) as hand_landmarker, \
         vision.PoseLandmarker.create_from_options(pose_options) as pose_landmarker:
        
        frame_idx = 0
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
                
            # Convert BGR to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
            # Calculate timestamp in ms
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            if timestamp_ms < 0:
                timestamp_ms = int(frame_idx * 1000 / fps)
                
            # Ensure timestamps are strictly monotonically increasing
            if frame_idx > 0 and timestamp_ms <= all_frames_data[-1].get("timestamp_ms", -1):
                timestamp_ms = all_frames_data[-1]["timestamp_ms"] + 1

            # Process frame using the 3 models
            face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
            hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
            pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            
            frame_data = {
                "frame": frame_idx,
                "timestamp_ms": timestamp_ms,
                "face_landmarks": [],
                "left_hand_landmarks": [],
                "right_hand_landmarks": [],
                "pose_landmarks": []
            }
            
            # 1. Face
            if face_result.face_landmarks:
                for lm in face_result.face_landmarks[0]:
                    frame_data["face_landmarks"].append({"x": lm.x, "y": lm.y, "z": lm.z})
                    
            # 2. Hands
            if hand_result.hand_landmarks:
                for idx, hand in enumerate(hand_result.handedness):
                    hand_label = hand[0].category_name
                    landmarks = hand_result.hand_landmarks[idx]
                    target_list = frame_data["left_hand_landmarks"] if hand_label == "Left" else frame_data["right_hand_landmarks"]
                    for lm in landmarks:
                        target_list.append({"x": lm.x, "y": lm.y, "z": lm.z})

            # 3. Pose
            if pose_result.pose_landmarks:
                for lm in pose_result.pose_landmarks[0]:
                    frame_data["pose_landmarks"].append({
                        "x": lm.x, 
                        "y": lm.y, 
                        "z": lm.z, 
                        "visibility": getattr(lm, 'visibility', 0.0),
                        "presence": getattr(lm, 'presence', 0.0)
                    })
            
            all_frames_data.append(frame_data)
            
            if frame_idx % 30 == 0:
                print(f"Processed frame {frame_idx}")
                
            frame_idx += 1

    cap.release()
    
    print(f"Saving data to {output_data_path}...")
    with open(output_data_path, 'w') as f:
        json.dump(all_frames_data, f)
        
    print(f"Finished extracting features for {os.path.basename(video_path)}")

def main():
    # Find all videos in the videos directory
    video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(VIDEOS_DIR, ext)))
        
    if not video_files:
        print(f"No video files found in the '{VIDEOS_DIR}' directory.")
        print("Please add your videos there and run the script again.")
        return
        
    print(f"Found {len(video_files)} video(s) to process.")
    
    for video_path in video_files:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_file = os.path.join(OUTPUT_DIR, f"features_{base_name}.json")
        extract_features(video_path, output_file)
        
    print("\nAll videos processed successfully! Check the 'output' folder.")

if __name__ == "__main__":
    main()
