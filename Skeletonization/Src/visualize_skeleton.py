import os
import tempfile
from pathlib import Path

_tmp = Path(tempfile.gettempdir())
os.environ.setdefault("MPLCONFIGDIR", str(_tmp / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_tmp))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


LIGHT_BG = "#f6f7f9"
PANEL_BG = "#ffffff"
TEXT = "#111827"
MUTED = "#4b5563"
BODY_COLOR = "#111827"
LEFT_COLOR = "#047857"
RIGHT_COLOR = "#b91c1c"
JOINT_COLOR = "#111827"

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

    with np.load(landmarks_path) as data:
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

    def style_2d_axis(ax, view_title, xlabel, ylabel, xlim, ylim):
        ax.set_facecolor(PANEL_BG)
        ax.set_title(view_title, fontsize=9, color=TEXT, pad=5, weight="bold")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel(xlabel, fontsize=7, color=MUTED, labelpad=1)
        ax.set_ylabel(ylabel, fontsize=7, color=MUTED, labelpad=1)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(False)
        ax.tick_params(labelsize=6, colors=MUTED, length=2)
        for spine in ax.spines.values():
            spine.set_color("#d1d5db")
            spine.set_linewidth(0.8)

    def draw_points_2d(ax, points, connections, dims, color, joint_size=12, line_width=1.3, alpha=1.0):
        x_dim, y_dim = dims
        ax.scatter(
            points[:, x_dim],
            points[:, y_dim],
            c=color,
            s=joint_size,
            zorder=5,
            alpha=alpha,
            edgecolors="white",
            linewidths=0.4,
        )
        for i, j in connections:
            ax.plot(
                [points[i, x_dim], points[j, x_dim]],
                [points[i, y_dim], points[j, y_dim]],
                color=color,
                linewidth=line_width,
                alpha=alpha,
                solid_capstyle="round",
            )

    def style_3d_axis(ax, view_title):
        ax.set_facecolor(PANEL_BG)
        ax.set_title(view_title, fontsize=9, color=TEXT, pad=5, weight="bold")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)
        ax.set_xlabel("X horizontal (m)", fontsize=7, color=MUTED, labelpad=0)
        ax.set_ylabel("Y height (m)", fontsize=7, color=MUTED, labelpad=0)
        ax.set_zlabel("Z depth (m)", fontsize=7, color=MUTED, labelpad=0)
        ax.tick_params(labelsize=6, colors=MUTED, length=2)
        ax.view_init(elev=18, azim=-45)
        ax.set_box_aspect((xmax - xmin, ymax - ymin, zmax - zmin))
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("#d1d5db")
        ax.yaxis.pane.set_edgecolor("#d1d5db")
        ax.zaxis.pane.set_edgecolor("#d1d5db")

    def draw_points_3d(ax, points, connections, color, joint_size=12, line_width=1.2, alpha=1.0):
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=color,
            s=joint_size,
            zorder=5,
            depthshade=False,
            alpha=alpha,
            edgecolors="white",
            linewidths=0.4,
        )
        for i, j in connections:
            ax.plot(
                [points[i, 0], points[j, 0]],
                [points[i, 1], points[j, 1]],
                [points[i, 2], points[j, 2]],
                color=color,
                linewidth=line_width,
                alpha=alpha,
                solid_capstyle="round",
            )

    def draw_hand_focus(ax, view_title, dims, xlabel, ylabel, body, lhand, rhand, l_ok, r_ok):
        style_2d_axis(ax, view_title, xlabel, ylabel, (xmin, xmax) if dims[0] == 0 else (zmin, zmax), (ymin, ymax))
        draw_points_2d(ax, body, BODY_CONNECTIONS, dims, BODY_COLOR, joint_size=8, line_width=0.9, alpha=0.25)

        focus_points = []
        if l_ok:
            draw_points_2d(ax, lhand, HAND_CONNECTIONS, dims, LEFT_COLOR, joint_size=14, line_width=1.4)
            focus_points.append(lhand[:, dims])
        if r_ok:
            draw_points_2d(ax, rhand, HAND_CONNECTIONS, dims, RIGHT_COLOR, joint_size=14, line_width=1.4)
            focus_points.append(rhand[:, dims])

        if focus_points:
            pts = np.concatenate(focus_points, axis=0)
            x0, y0 = pts.min(axis=0)
            x1, y1 = pts.max(axis=0)
            pad_x = max(0.08, (x1 - x0) * 0.35)
            pad_y = max(0.08, (y1 - y0) * 0.35)
            ax.set_xlim(x0 - pad_x, x1 + pad_x)
            ax.set_ylim(y0 - pad_y, y1 + pad_y)
        else:
            ax.text(0.5, 0.5, "No hand detected", ha="center", va="center", transform=ax.transAxes, color=MUTED)

    def draw_frame(frame_idx, axes):
        body = body_flipped[frame_idx]
        lhand = lhand_translated[frame_idx]
        rhand = rhand_translated[frame_idx]
        l_ok = left_flag_seq[frame_idx]
        r_ok = right_flag_seq[frame_idx]

        ax_front, ax_side, ax_top, ax_persp, ax_hands_front, ax_hands_depth = axes
        for ax in axes:
            ax.cla()

        style_2d_axis(ax_front, "FRONT camera view: X horizontal, Y height", "X horizontal (m)", "Y height (m)", (xmin, xmax), (ymin, ymax))
        style_2d_axis(ax_side, "SIDE depth view: Z depth, Y height", "Z depth (m)", "Y height (m)", (zmin, zmax), (ymin, ymax))
        style_2d_axis(ax_top, "TOP floor view: X horizontal, Z depth", "X horizontal (m)", "Z depth (m)", (xmin, xmax), (zmin, zmax))
        style_3d_axis(ax_persp, "45 degree 3D view: X/Y/Z together")

        for ax, dims in [(ax_front, (0, 1)), (ax_side, (2, 1)), (ax_top, (0, 2))]:
            draw_points_2d(ax, body, BODY_CONNECTIONS, dims, BODY_COLOR, joint_size=12, line_width=1.5)
            if l_ok:
                draw_points_2d(ax, lhand, HAND_CONNECTIONS, dims, LEFT_COLOR, joint_size=11, line_width=1.2)
            if r_ok:
                draw_points_2d(ax, rhand, HAND_CONNECTIONS, dims, RIGHT_COLOR, joint_size=11, line_width=1.2)

        draw_points_3d(ax_persp, body, BODY_CONNECTIONS, BODY_COLOR, joint_size=12, line_width=1.3)
        if l_ok:
            draw_points_3d(ax_persp, lhand, HAND_CONNECTIONS, LEFT_COLOR, joint_size=11, line_width=1.1)
        if r_ok:
            draw_points_3d(ax_persp, rhand, HAND_CONNECTIONS, RIGHT_COLOR, joint_size=11, line_width=1.1)

        draw_hand_focus(
            ax_hands_front,
            "HAND close-up front: X/Y overlap check",
            (0, 1),
            "X horizontal (m)",
            "Y height (m)",
            body,
            lhand,
            rhand,
            l_ok,
            r_ok,
        )
        draw_hand_focus(
            ax_hands_depth,
            "HAND close-up depth: Z/Y crossing check",
            (2, 1),
            "Z depth (m)",
            "Y height (m)",
            body,
            lhand,
            rhand,
            l_ok,
            r_ok,
        )

        ax_front.text(0.02, 0.02, "Body", color=BODY_COLOR, transform=ax_front.transAxes, fontsize=7, weight="bold")
        ax_front.text(0.18, 0.02, "Left hand", color=LEFT_COLOR, transform=ax_front.transAxes, fontsize=7, weight="bold")
        ax_front.text(0.42, 0.02, "Right hand", color=RIGHT_COLOR, transform=ax_front.transAxes, fontsize=7, weight="bold")

    preview_frame = T // 2

    if preview_path is not None:
        fig = plt.figure(figsize=(18, 10), facecolor=LIGHT_BG)
        fig.suptitle(
            f"{title} Skeleton - Frame {preview_frame}  "
            f"| R-hand: {'detected' if right_flag_seq[preview_frame] else 'missing'}"
            f"  L-hand: {'detected' if left_flag_seq[preview_frame] else 'missing'}",
            color=TEXT,
            fontsize=11,
            weight="bold",
        )
        axes = [
            fig.add_subplot(2, 3, 1),
            fig.add_subplot(2, 3, 2),
            fig.add_subplot(2, 3, 3),
            fig.add_subplot(2, 3, 4, projection="3d"),
            fig.add_subplot(2, 3, 5),
            fig.add_subplot(2, 3, 6),
        ]
        draw_frame(preview_frame, axes)
        plt.tight_layout()
        plt.savefig(preview_path, dpi=120, bbox_inches="tight", facecolor=LIGHT_BG)
        plt.close(fig)
        print(f"Preview saved to {preview_path}")

    fig_anim = plt.figure(figsize=(18, 10), facecolor=LIGHT_BG)
    axes_anim = [
        fig_anim.add_subplot(2, 3, 1),
        fig_anim.add_subplot(2, 3, 2),
        fig_anim.add_subplot(2, 3, 3),
        fig_anim.add_subplot(2, 3, 4, projection="3d"),
        fig_anim.add_subplot(2, 3, 5),
        fig_anim.add_subplot(2, 3, 6),
    ]
    title_obj = fig_anim.suptitle("", color=TEXT, fontsize=11, weight="bold")

    def update(frame_idx):
        r_status = "R detected" if right_flag_seq[frame_idx] else "R missing"
        l_status = "L detected" if left_flag_seq[frame_idx] else "L missing"
        title_obj.set_text(f"{title} - Frame {frame_idx}/{T - 1}  |  {r_status}  {l_status}")
        draw_frame(frame_idx, axes_anim)
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
    anim.save(output_path, writer=writer, dpi=100, savefig_kwargs={"facecolor": LIGHT_BG})
    print("Saved.")
    plt.close(fig_anim)
    return output_path
