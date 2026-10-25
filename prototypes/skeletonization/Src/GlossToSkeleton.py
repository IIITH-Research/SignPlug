from pathlib import Path

from Skeletonization.Src.extract_keypoints import extract_keypoints
from interpolate_keypoints import interpolate_keypoints_file
from visualize_skeleton import visualize_3views


class GlossToSkeleton:
    """Pipeline class for one gloss/video."""

    VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv")

    def __init__(self, gloss, video_path=None, input_dir=None, output_dir=None):
        self.gloss = gloss
        self.name = self._clean_name(gloss)

        skeletonization_dir = Path(__file__).resolve().parents[1]
        self.input_dir = Path(input_dir) if input_dir is not None else skeletonization_dir / "Scripts" / "Inputs"
        self.output_dir = Path(output_dir) if output_dir is not None else skeletonization_dir / "Scripts" / "Outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.video_path = Path(video_path) if video_path is not None else self._find_video()

        self.raw_npz_path = self.output_dir / f"{self.name}_raw_landmarks.npz"
        self.landmarks_path = self.output_dir / f"{self.name}_signer_landmarks.npz"
        self.preview_path = self.output_dir / f"{self.name}_skeleton_preview.png"
        self.video_3views_path = self.output_dir / f"{self.name}_skeleton_3views.mp4"

    def _clean_name(self, gloss):
        return str(gloss).strip().lower().replace(" ", "_")

    def _find_video(self):
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input folder not found: {self.input_dir}")

        for path in self.input_dir.iterdir():
            if path.is_file() and path.suffix.lower() in self.VIDEO_EXTENSIONS:
                if path.stem.lower().replace(" ", "_") == self.name:
                    return path

        available = sorted(path.name for path in self.input_dir.iterdir() if path.is_file())
        raise FileNotFoundError(
            f"No video found for gloss '{self.gloss}' in {self.input_dir}. "
            f"Available files: {available}"
        )

    def extract(self):
        return extract_keypoints(self.video_path, self.raw_npz_path)

    def interpolate(self):
        return interpolate_keypoints_file(self.raw_npz_path, self.landmarks_path)

    def visualize(self):
        return visualize_3views(
            self.landmarks_path,
            self.video_3views_path,
            preview_path=self.preview_path,
            title=f"{self.gloss.title()} Signer",
        )

    def run(self):
        self.extract()
        self.interpolate()
        return self.visualize()