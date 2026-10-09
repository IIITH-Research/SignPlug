# ISL-AVATAR-PROJECT

This repo contains the **core pipeline** for an ISL avatar system:

Video → MediaPipe keypoints → JSON animation clip → Browser renderer

> **Annotating keyframes?** See the annotator app guide:
> [skinning/src/intepolation_code/README.md](skinning/src/intepolation_code/README.md)

## 1) Install dependencies

If you already ran `pip install mediapipe opencv-python numpy`, you can skip this.

```bash
pip install -r requirements.txt
```

## 2) Extract a JSON clip from the video

Input video is already included here:

- `dataset/Apple.mp4`

Run the extractor:

```bash
python extract.py
```

First run note: `extract.py` downloads a MediaPipe model into `models/` (one-time).

This produces:

- `clips/apple.json`

### Useful options

```bash
python extract.py --input dataset/Apple.mp4 --output clips/apple.json --target-fps 30 --precision 5
```

### Face landmarks (optional but enabled by default)

Face landmarks (often 468 or 478 points depending on the model) make the JSON bigger. You can control it:

- Disable face entirely:

```bash
python extract.py --no-face
```

- Keep face but reduce size (recommended when clips get long):

```bash
python extract.py --face-precision 3 --face-every-n 2
```

## 2D Avatar skinning (minimal datapoints)

If your goal is a **fast 2D skinned avatar** (instant web rendering), you usually do **not** need the full face mesh.

Use the `avatar2d` preset:

```bash
python extract.py --preset avatar2d
```

What it does (minimal + efficient):
- `face_mode = pose`: face is only 11 pose points (nose/eyes/ears/mouth)
- `hand_mode = tips`: hands are 6 points (wrist + fingertips)

You can still override:

```bash
python extract.py --preset avatar2d --hand-mode full
```

- `--target-fps`: resamples the clip to ~30 fps (good for browser playback)
- `--precision`: lower = smaller JSON (e.g., 4) but slightly less accurate
- `--model-complexity`: 0 faster, 2 more accurate (default 1)

## 3) Render it in the browser

Browsers block `fetch()` from local files, so run a local server:

```bash
python -m http.server 8000
```

Then open:

- http://localhost:8000

You should see **moving green dots** (pose + hands), looping the APPLE sign.