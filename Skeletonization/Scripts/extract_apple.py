from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[2]
SKELETONIZATION_DIR = PROJECT_DIR / "Skeletonization"
SRC_DIR = SKELETONIZATION_DIR / "Src"
OUTPUT_DIR = SKELETONIZATION_DIR / "Scripts" / "Outputs"
INPUT_DIR = SKELETONIZATION_DIR / "Scripts" / "Inputs"

sys.path.insert(0, str(SRC_DIR))

from extract_keypoints import extract_keypoints


if __name__ == "__main__":
    extract_keypoints(
        INPUT_DIR / "Apple.mp4",
        OUTPUT_DIR / "apple_raw_landmarks.npz",
    )
