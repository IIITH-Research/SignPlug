# UniSign Feature Extraction

Standalone feature extractor using the [Uni-Sign](https://github.com/ZechengLi19/Uni-Sign) model.
Supports **pose-only** and **RGB+Pose** modes. No WLASL embeddings or metadata required.

## Folder Structure

```
unisign_feature_extraction/
├── extract_unisign_embeddings.py   # main extraction script
├── checkpoints/
│   ├── wlasl_pose_only_islr.pth    # pose-only checkpoint (~1.2G)
│   └── wlasl_rgb_pose_islr.pth     # RGB+Pose checkpoint (~1.2G)
└── Uni-Sign/                       # model source code
```

## Requirements

Install dependencies from the Uni-Sign repo:
```bash
pip install -r Uni-Sign/requirements.txt
```

`deepspeed` is **not** required — it is mocked automatically by the script.

## Input Format

- **Pose PKL files**: each `.pkl` file is a dict with keys:
  - `keypoints`: list of per-frame numpy arrays (shape: `[N_people, N_joints, 2]`, normalized 0–1)
  - `scores`: list of per-frame confidence arrays (shape: `[N_people, N_joints]`)
- **Video files** (RGB+Pose only): standard `.mp4` files, named matching the pose PKL (same stem).

## Usage

### Pose-only
```bash
python extract_unisign_embeddings.py \
    --checkpoint checkpoints/wlasl_pose_only_islr.pth \
    --pose-root /path/to/pose_pkls \
    --output embeddings/pose_only.pt
```

### RGB+Pose
```bash
python extract_unisign_embeddings.py \
    --checkpoint checkpoints/wlasl_rgb_pose_islr.pth \
    --pose-root /path/to/pose_pkls \
    --video-root /path/to/videos \
    --rgb-support \
    --output embeddings/rgb_pose.pt
```

### All arguments

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | required | Path to `.pth` checkpoint |
| `--pose-root` | required | Dir containing per-sample `.pkl` pose files |
| `--video-root` | `None` | Dir with per-sample video files (required for `--rgb-support`) |
| `--video-ext` | `.mp4` | Video file extension |
| `--pose-ext` | `.pkl` | Pose file extension |
| `--rgb-support` | off | Enable RGB+Pose mode |
| `--max-length` | `64` | Max frames (longer clips are uniformly subsampled) |
| `--output` | required | Output `.pt` file path |
| `--device` | `cuda` | Device (`cuda` or `cpu`) |
| `--metadata-jsonl` | `None` | Optional path to write per-sample metadata as JSONL |

## Output

A `.pt` file loadable with `torch.load()`, containing:
```python
{
    "embeddings": torch.Tensor,  # shape [N, D], L2-normalized
    "metadata":   list[dict],    # [{"id": "sample_stem"}, ...]
}
```

## Notes

- Samples whose pose PKL fails to load are skipped with a warning.
- In RGB+Pose mode, samples with no matching video are also skipped.
- Pose files are matched to videos by filename stem (`00001.pkl` → `00001.mp4`).
- The script changes `cwd` to `Uni-Sign/` temporarily during model init (required for relative config imports), then restores it.
