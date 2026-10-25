# 1. SMPL-X pose estimation

Recovers frame-wise SMPL-X body, hand and jaw parameters from each ISLRTC dictionary video with
[SMPLer-X](https://github.com/caizhongang/SMPLer-X) (`smpler_x_h32` checkpoint), the first step of the
Widget Dictionary construction (paper §3.1).

- `frame_extractor.ipynb`: extracts the video frames (videos are preprocessed to 25 FPS).
- `inference.py`: our version of SMPLer-X's `main/inference.py`; writes one `.npz` per frame with
  `global_orient`, `body_pose`, `left_hand_pose`, `right_hand_pose`, `jaw_pose`, `betas`, `transl`.
- `human_models.py`: our version of SMPLer-X's `common/utils/human_models.py`.

Setup: clone SMPLer-X, follow its installation, download the SMPL-X model files and the
`smpler_x_h32` checkpoint, then copy these two `.py` files over the originals.
