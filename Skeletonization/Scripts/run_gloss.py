from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[2]
SKELETONIZATION_DIR = PROJECT_DIR / "Skeletonization"
SRC_DIR = SKELETONIZATION_DIR / "Src"

sys.path.insert(0, str(SRC_DIR))

from GlossToSkeleton import GlossToSkeleton


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python3 Skeletonization/Scripts/run_gloss.py <gloss>")

    gloss = sys.argv[1]
    skeleton = GlossToSkeleton(gloss)
    skeleton.run()
