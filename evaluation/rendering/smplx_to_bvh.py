#!/usr/bin/env python3
"""
Convert SMPL-X .npz motion frames to a single BVH file.
Uses only the 22 body joints (no hands/face) for standard BVH compatibility.

Run with:
  /Volumes/SEAGATE/environments/smplx_env/bin/python smplx_to_bvh.py
"""

import numpy as np
import torch
import smplx
import glob, os
from scipy.spatial.transform import Rotation

FRAMES_DIR   = "/Volumes/SEAGATE/sharif_icvgip/Happy_smplx_interp"
MODEL_FOLDER = "/Volumes/SEAGATE/sharif_icvgip/smplx_models"
OUT_BVH      = "/Volumes/SEAGATE/sharif_icvgip/happy.bvh"
FPS          = 25
SCALE        = 100.0  # meters → centimeters (BVH convention)

# 22 SMPL-X body joints → BVH-compatible names (Mixamo-style)
JOINT_NAMES = [
    'Hips',         # 0  pelvis
    'LeftUpLeg',    # 1  left_hip
    'RightUpLeg',   # 2  right_hip
    'Spine',        # 3  spine1
    'LeftLeg',      # 4  left_knee
    'RightLeg',     # 5  right_knee
    'Spine1',       # 6  spine2
    'LeftFoot',     # 7  left_ankle
    'RightFoot',    # 8  right_ankle
    'Spine2',       # 9  spine3
    'LeftToeBase',  # 10 left_foot
    'RightToeBase', # 11 right_foot
    'Neck',         # 12 neck
    'LeftShoulder', # 13 left_collar
    'RightShoulder',# 14 right_collar
    'Head',         # 15 head
    'LeftArm',      # 16 left_shoulder
    'RightArm',     # 17 right_shoulder
    'LeftForeArm',  # 18 left_elbow
    'RightForeArm', # 19 right_elbow
    'LeftHand',     # 20 left_wrist
    'RightHand',    # 21 right_wrist
]
N_JOINTS = len(JOINT_NAMES)

# Parent index for each joint (-1 = root)
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]

# Build children list
CHILDREN = [[] for _ in range(N_JOINTS)]
for i, p in enumerate(PARENTS):
    if p >= 0:
        CHILDREN[p].append(i)

LEAVES = [i for i in range(N_JOINTS) if len(CHILDREN[i]) == 0]


def create_model():
    return smplx.create(
        MODEL_FOLDER,
        model_type="smplx",
        gender="neutral",
        num_betas=10,
        num_expression_coeffs=10,
        use_pca=False,
        flat_hand_mean=False,
    )


def get_tpose_joints(model):
    """Run SMPL-X with zero pose to get canonical T-pose joint positions (22 body joints)."""
    with torch.no_grad():
        output = model(
            global_orient=torch.zeros(1, 3),
            body_pose=torch.zeros(1, 63),
            betas=torch.zeros(1, 10),
        )
    # joints shape: (1, 127, 3) — first 22 are body joints
    return output.joints.detach().cpu().numpy()[0, :N_JOINTS]  # (22, 3)


def axis_angle_to_euler_zxy(rotvec):
    """Axis-angle → Euler [Z, X, Y] degrees (BVH intrinsic ZXY convention)."""
    r = Rotation.from_rotvec(rotvec)
    return r.as_euler('ZXY', degrees=True)  # [Z, X, Y]


def compute_end_offsets(tpose_joints):
    """Small offset vectors for BVH End Sites (leaf joints)."""
    end_off = {}
    for leaf in LEAVES:
        p = PARENTS[leaf]
        pp = PARENTS[p] if p >= 0 else -1
        if pp >= 0:
            limb = (tpose_joints[p] - tpose_joints[pp]) * SCALE
        else:
            limb = np.array([0.0, 5.0, 0.0])
        norm = np.linalg.norm(limb)
        end_off[leaf] = limb / norm * max(3.0, norm * 0.25) if norm > 0 else np.array([0.0, 3.0, 0.0])
    return end_off


def write_hierarchy(lines, tpose_joints, end_off, idx, indent):
    pad = "    " * indent
    tag = "ROOT" if PARENTS[idx] < 0 else "JOINT"
    lines.append(f"{pad}{tag} {JOINT_NAMES[idx]}")
    lines.append(f"{pad}{{")

    if PARENTS[idx] < 0:
        ox, oy, oz = tpose_joints[idx] * SCALE
        lines.append(f"{pad}    OFFSET {ox:.4f} {oy:.4f} {oz:.4f}")
        lines.append(f"{pad}    CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation")
    else:
        p = PARENTS[idx]
        offset = (tpose_joints[idx] - tpose_joints[p]) * SCALE
        ox, oy, oz = offset
        lines.append(f"{pad}    OFFSET {ox:.4f} {oy:.4f} {oz:.4f}")
        lines.append(f"{pad}    CHANNELS 3 Zrotation Xrotation Yrotation")

    if idx in LEAVES:
        ex, ey, ez = end_off[idx]
        lines.append(f"{pad}    End Site")
        lines.append(f"{pad}    {{")
        lines.append(f"{pad}        OFFSET {ex:.4f} {ey:.4f} {ez:.4f}")
        lines.append(f"{pad}    }}")
    else:
        for child in CHILDREN[idx]:
            write_hierarchy(lines, tpose_joints, end_off, child, indent + 1)

    lines.append(f"{pad}}}")


def write_bvh(tpose_joints, all_rotations, all_translations):
    end_off = compute_end_offsets(tpose_joints)

    lines = ["HIERARCHY"]
    write_hierarchy(lines, tpose_joints, end_off, 0, 0)

    n_frames = len(all_rotations)
    lines.append("MOTION")
    lines.append(f"Frames: {n_frames}")
    lines.append(f"Frame Time: {1.0 / FPS:.6f}")

    for f in range(n_frames):
        row = []
        # Root: XYZ position (scaled)
        tx, ty, tz = all_translations[f] * SCALE
        row.extend([tx, ty, tz])
        # All joints: ZXY Euler in hierarchy order
        for j in range(N_JOINTS):
            euler = axis_angle_to_euler_zxy(all_rotations[f, j])
            row.extend(euler.tolist())  # [Z, X, Y]
        lines.append("  ".join(f"{v:.4f}" for v in row))

    with open(OUT_BVH, 'w') as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"BVH saved: {OUT_BVH}  ({n_frames} frames, {N_JOINTS} joints)")


def main():
    print("Loading SMPL-X model...")
    model = create_model()
    model.eval()

    print("Getting T-pose joint positions...")
    tpose_joints = get_tpose_joints(model)
    print(f"  Pelvis Y (height): {tpose_joints[0, 1]*100:.1f} cm")

    frame_paths = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.npz")))
    print(f"Processing {len(frame_paths)} frames...")

    all_rotations    = []
    all_translations = []

    for i, path in enumerate(frame_paths):
        data = np.load(path, allow_pickle=True)

        global_orient = data['global_orient'].reshape(3)
        body_pose     = data['body_pose'].reshape(21, 3)   # 21 joints
        transl        = data['transl'].reshape(3)

        # Joint 0 = global_orient, joints 1-21 = body_pose
        frame_rots = np.vstack([global_orient[np.newaxis], body_pose])  # (22, 3)
        all_rotations.append(frame_rots)
        all_translations.append(transl)
        print(f"  [{i+1}/{len(frame_paths)}] {os.path.basename(path)}")

    all_rotations    = np.array(all_rotations)    # (N, 22, 3)
    all_translations = np.array(all_translations) # (N, 3)

    print("Writing BVH...")
    write_bvh(tpose_joints, all_rotations, all_translations)
    print("Done.")


if __name__ == "__main__":
    main()
