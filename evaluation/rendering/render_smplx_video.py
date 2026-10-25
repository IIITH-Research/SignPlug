"""
Renders SMPL-X .npz parameter frames into a video.
Uses matplotlib for macOS-compatible headless rendering.

Usage:
    python render_smplx_video.py \
        --frames_dir /Volumes/SEAGATE/sharif_icvgip/Happy_smplx_interp \
        --model_path /Volumes/SEAGATE/sharif_icvgip/smplx_models/SMPLX_NEUTRAL.npz \
        --output /Volumes/SEAGATE/sharif_icvgip/happy_output.mp4 \
        --fps 12
"""

import argparse
import os
import glob
import numpy as np
import smplx
import torch
import cv2
import matplotlib
matplotlib.use("Agg")  # non-interactive, no display needed
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def load_frame(path):
    data = np.load(path, allow_pickle=True)
    return {k: torch.tensor(data[k]).float() for k in data.files}


def render_frame_matplotlib(vertices, faces, img_size, elev=10, azim=180):
    fig = plt.figure(figsize=(img_size / 100, img_size / 100), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#e8e8e8")
    fig.patch.set_facecolor("#e8e8e8")

    # downsample faces for speed (every 4th face)
    faces_sub = faces[::4]
    tris = vertices[faces_sub]

    mesh = Poly3DCollection(tris, alpha=0.85, linewidth=0)
    mesh.set_facecolor([0.78, 0.70, 0.62])
    mesh.set_edgecolor("none")
    ax.add_collection3d(mesh)

    # fit axes to body
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    cx, cy, cz = x.mean(), y.mean(), z.mean()
    r = max(x.max() - x.min(), y.max() - y.min(), z.max() - z.min()) * 0.6

    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy - r, cy + r)
    ax.set_zlim(cz - r, cz + r)

    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)

    fig.tight_layout(pad=0)
    fig.canvas.draw()

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img = buf.reshape(h, w, 4)[:, :, :3]  # drop alpha
    plt.close(fig)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_dir", default="/Volumes/SEAGATE/sharif_icvgip/Happy_smplx_interp")
    parser.add_argument("--model_path", default="/Volumes/SEAGATE/sharif_icvgip/smplx_models/SMPLX_NEUTRAL.npz")
    parser.add_argument("--output", default="/Volumes/SEAGATE/sharif_icvgip/happy_output.mp4")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--img_size", type=int, default=720)
    parser.add_argument("--elev", type=float, default=10)
    parser.add_argument("--azim", type=float, default=180)
    args = parser.parse_args()

    frame_paths = sorted(glob.glob(os.path.join(args.frames_dir, "*.npz")))
    print(f"Found {len(frame_paths)} frames")

    model_folder = os.path.dirname(args.model_path)
    body_model = smplx.create(
        model_folder,
        model_type="smplx",
        gender="neutral",
        num_betas=10,
        num_expression_coeffs=10,
        use_pca=False,
        flat_hand_mean=True,
    )
    body_model.eval()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, args.fps, (args.img_size, args.img_size))

    with torch.no_grad():
        for i, path in enumerate(frame_paths):
            print(f"  Rendering frame {i+1}/{len(frame_paths)}: {os.path.basename(path)}")
            params = load_frame(path)

            output = body_model(
                global_orient=params["global_orient"],
                body_pose=params["body_pose"].reshape(1, -1),
                left_hand_pose=params["left_hand_pose"].reshape(1, -1),
                right_hand_pose=params["right_hand_pose"].reshape(1, -1),
                jaw_pose=params["jaw_pose"],
                betas=params["betas"],
                expression=params["expression"],
                transl=params["transl"],
                return_verts=True,
            )

            vertices = output.vertices.detach().cpu().numpy()[0]
            faces = body_model.faces

            img = render_frame_matplotlib(vertices, faces, args.img_size, args.elev, args.azim)
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # resize to exact img_size x img_size if needed
            if bgr.shape[:2] != (args.img_size, args.img_size):
                bgr = cv2.resize(bgr, (args.img_size, args.img_size))

            writer.write(bgr)

    writer.release()
    print(f"\nVideo saved to: {args.output}")


if __name__ == "__main__":
    main()
