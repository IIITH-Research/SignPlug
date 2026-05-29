from pathlib import Path

import numpy as np


def interpolate_missing(seq, valid_mask, name="stream"):
    T, N, _ = seq.shape
    filled = seq.copy()

    if valid_mask.ndim == 1:
        valid_mask = np.repeat(valid_mask[:, None], N, axis=1)

    for j in range(N):
        valid_frames = np.where(valid_mask[:, j])[0]

        if len(valid_frames) == 0:
            print(f"  {name} joint {j}: never reliable - dropped")
            continue

        t_all = np.arange(T)
        for coord in range(3):
            v_valid = seq[valid_frames, j, coord]
            filled[:, j, coord] = np.interp(t_all, valid_frames, v_valid)

    return filled


def interpolate_keypoints(data):
    """Apply the same interpolation used in the notebook."""
    interpolated = dict(data)
    interpolated["body_world"] = interpolate_missing(
        interpolated["body_world"],
        interpolated["body_joint_valid"],
        "body",
    )
    interpolated["left_hand"] = interpolate_missing(
        interpolated["left_hand"],
        interpolated["left_flag"],
        "left hand",
    )
    interpolated["right_hand"] = interpolate_missing(
        interpolated["right_hand"],
        interpolated["right_flag"],
        "right hand",
    )
    return interpolated


def interpolate_keypoints_file(input_path, output_path=None):
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with np.load(input_path) as loaded:
        data = {key: loaded[key] for key in loaded.files}

    interpolated = interpolate_keypoints(data)
    np.savez(output_path, **interpolated)
    print(f"Saved interpolated landmarks to {output_path}")
    return output_path
