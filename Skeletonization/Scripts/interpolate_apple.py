from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[2]
SKELETONIZATION_DIR = PROJECT_DIR / "Skeletonization"
SRC_DIR = SKELETONIZATION_DIR / "Src"
OUTPUT_DIR = SKELETONIZATION_DIR / "Scripts" / "Outputs"

sys.path.insert(0, str(SRC_DIR))

from interpolate_keypoints import interpolate_keypoints_file


if __name__ == "__main__":
    interpolate_keypoints_file(
        OUTPUT_DIR / "apple_raw_landmarks.npz",
        OUTPUT_DIR / "apple_signer_landmarks.npz",
    )
