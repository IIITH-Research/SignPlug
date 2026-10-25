# Skinning Pipeline

## Installation

We now provide a dedicated setup script that handles downloading all dependencies and required machine-learning models. Run this once:

```bash
python setup.py
```

It will create a `.venv/` virtual environment, install the packages from `requirements.txt`, and download `hand_landmarker.task` and `pose_landmarker.task` directly into `src/models/`.

After setup completes, activate the environment:
- Linux/Mac: `source .venv/bin/activate`
- Windows: `.venv\Scripts\activate`

## Suggested pipeline

1. Put videos in `src/videos/`
2. Run `python src/extract.py`
3. Run `python src/smoother.py`
4. Run `python prototype/visualize.py --backend QtAgg`


## Interactive visualizer

`prototype/visualize.py` is an interactive 3D player:
- Drag to rotate/inspect from all sides
- Mouse wheel to zoom
- Play/Pause button and frame slider
- Space toggles play, Left/Right steps one frame

