from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[2]
SKELETONIZATION_DIR = PROJECT_DIR / "Skeletonization"
SRC_DIR = SKELETONIZATION_DIR / "Src"
OUTPUT_DIR = SKELETONIZATION_DIR / "Scripts" / "Outputs"

sys.path.insert(0, str(SRC_DIR))

from visualize_skeleton import visualize_3views


if __name__ == "__main__":
    visualize_3views(
        OUTPUT_DIR / "apple_signer_landmarks.npz",
        OUTPUT_DIR / "apple_skeleton_3views.mp4",
        preview_path=OUTPUT_DIR / "apple_skeleton_preview.png",
        title="Apple Signer",
    )
