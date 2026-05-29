import os
from pathlib import Path

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.environ["XDG_CACHE_HOME"] = "/tmp"
Path("/tmp/matplotlib").mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


BODY_CONNECTIONS = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 11),
    (0, 12),
    (0, 7),
    (0, 8),
    (7, 3),
    (8, 6),
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def visualize_3views(landmarks_path, output_path, preview_path=None, title="Apple Signer", fps_video=25):
    landmarks_path = Path(landmarks_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if preview_path is not None:
        preview_path = Path(preview_path)
        preview_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(landmarks_path)
    body_world_seq = data["body_world"]
    left_hand_seq = data["left_hand"]
    right_hand_seq = data["right_hand"]
    left_flag_seq = data["left_flag"]
    right_flag_seq = data["right_flag"]

    T = body_world_seq.shape[0]
    print(f"Loaded {T} frames")
    print(f"Left hand detected  : {left_flag_seq.sum()}")
    print(f"Right hand detected : {right_flag_seq.sum()}")

    body_flipped = body_world_seq.copy()
    body_flipped[:, :, 1] = -body_flipped[:, :, 1]

    lhand_translated = left_hand_seq.copy()
    rhand_translated = right_hand_seq.copy()
    lhand_translated[:, :, 1] = -lhand_translated[:, :, 1]
    rhand_translated[:, :, 1] = -rhand_translated[:, :, 1]

    for f in range(T):
        if left_flag_seq[f]:
            lhand_translated[f] = lhand_translated[f] + body_flipped[f, 15]
        if right_flag_seq[f]:
            rhand_translated[f] = rhand_translated[f] + body_flipped[f, 16]

    all_pts = np.concatenate(
        [
            body_flipped.reshape(-1, 3),
            lhand_translated.reshape(-1, 3),
            rhand_translated.reshape(-1, 3),
        ],
        axis=0,
    )

    pad = 0.15
    xmin, xmax = all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad
    ymin, ymax = all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad
    zmin, zmax = all_pts[:, 2].min() - pad, all_pts[:, 2].max() + pad

    print("\nAxis limits (metres):")
    print(f"  X: [{xmin:.2f}, {xmax:.2f}]")
    print(f"  Y: [{ymin:.2f}, {ymax:.2f}]")
    print(f"  Z: [{zmin:.2f}, {zmax:.2f}]")

    def draw_frame(frame_idx, ax_front, ax_side, ax_top):
        body = body_flipped[frame_idx]
        lhand = lhand_translated[frame_idx]
        rhand = rhand_translated[frame_idx]
        l_ok = left_flag_seq[frame_idx]
        r_ok = right_flag_seq[frame_idx]

        for ax, elev, azim, view_title in [
            (ax_front, 10, -90, "Front view"),
            (ax_side, 10, 0, "Side view"),
            (ax_top, 90, -90, "Top view"),
        ]:
            ax.cla()
            ax.set_title(view_title, fontsize=9, pad=2, color="white")
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_zlim(zmin, zmax)
            ax.set_xlabel("X (m)", fontsize=7, color="gray")
            ax.set_ylabel("Y (m)", fontsize=7, color="gray")
            ax.set_zlabel("Z (m)", fontsize=7, color="gray")
            ax.tick_params(labelsize=6, colors="gray")
            ax.view_init(elev=elev, azim=azim)
            ax.set_facecolor("#1a1a1a")
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor("#333333")
            ax.yaxis.pane.set_edgecolor("#333333")
            ax.zaxis.pane.set_edgecolor("#333333")
            ax.grid(True, color="#2a2a2a", linewidth=0.5)

            ax.scatter(body[:, 0], body[:, 1], body[:, 2], c="white", s=18, zorder=5, depthshade=False)

            for i, j in BODY_CONNECTIONS:
                ax.plot(
                    [body[i, 0], body[j, 0]],
                    [body[i, 1], body[j, 1]],
                    [body[i, 2], body[j, 2]],
                    c="deepskyblue",
                    linewidth=1.2,
                )

            if l_ok:
                ax.scatter(lhand[:, 0], lhand[:, 1], lhand[:, 2], c="lime", s=14, zorder=5, depthshade=False)
                for i, j in HAND_CONNECTIONS:
                    ax.plot(
                        [lhand[i, 0], lhand[j, 0]],
                        [lhand[i, 1], lhand[j, 1]],
                        [lhand[i, 2], lhand[j, 2]],
                        c="lime",
                        linewidth=0.9,
                    )

            if r_ok:
                ax.scatter(rhand[:, 0], rhand[:, 1], rhand[:, 2], c="orange", s=14, zorder=5, depthshade=False)
                for i, j in HAND_CONNECTIONS:
                    ax.plot(
                        [rhand[i, 0], rhand[j, 0]],
                        [rhand[i, 1], rhand[j, 1]],
                        [rhand[i, 2], rhand[j, 2]],
                        c="orange",
                        linewidth=0.9,
                    )

    preview_frame = T // 2

    if preview_path is not None:
        fig = plt.figure(figsize=(15, 5), facecolor="#0d0d0d")
        fig.suptitle(
            f"{title} Skeleton - Frame {preview_frame}  "
            f"| R-hand: {'detected' if right_flag_seq[preview_frame] else 'missing'}"
            f"  L-hand: {'detected' if left_flag_seq[preview_frame] else 'missing'}",
            color="white",
            fontsize=10,
        )
        ax1 = fig.add_subplot(131, projection="3d")
        ax2 = fig.add_subplot(132, projection="3d")
        ax3 = fig.add_subplot(133, projection="3d")
        draw_frame(preview_frame, ax1, ax2, ax3)
        plt.tight_layout()
        plt.savefig(preview_path, dpi=120, bbox_inches="tight", facecolor="#0d0d0d")
        plt.close(fig)
        print(f"Preview saved to {preview_path}")

    fig_anim = plt.figure(figsize=(15, 5), facecolor="#0d0d0d")
    ax1 = fig_anim.add_subplot(131, projection="3d")
    ax2 = fig_anim.add_subplot(132, projection="3d")
    ax3 = fig_anim.add_subplot(133, projection="3d")
    title_obj = fig_anim.suptitle("", color="white", fontsize=10)

    def update(frame_idx):
        r_status = "R detected" if right_flag_seq[frame_idx] else "R missing"
        l_status = "L detected" if left_flag_seq[frame_idx] else "L missing"
        title_obj.set_text(f"{title} - Frame {frame_idx + 1}/{T}  |  {r_status}  {l_status}")
        draw_frame(frame_idx, ax1, ax2, ax3)
        return []

    anim = animation.FuncAnimation(
        fig_anim,
        update,
        frames=T,
        interval=1000 / fps_video,
        blit=False,
    )

    writer = animation.FFMpegWriter(
        fps=fps_video,
        metadata={"title": f"{title} 3 Views"},
        bitrate=2000,
    )

    print(f"\nSaving {T} frames to {output_path} ...")
    anim.save(output_path, writer=writer, dpi=100, savefig_kwargs={"facecolor": "#0d0d0d"})
    print("Saved.")
    plt.close(fig_anim)
    return output_path
