import numpy as np
import os
import json
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "src" / "output"

import matplotlib
from matplotlib.backends import backend_registry, BackendFilter

def configure_interactive_backend(preferred_backend=None):
    current = matplotlib.get_backend().lower()
    interactive_builtins = {
        backend.lower()
        for backend in backend_registry.list_builtin(BackendFilter.INTERACTIVE)
    }
    if current in interactive_builtins:
        return

    candidates = []
    if preferred_backend:
        candidates.append(preferred_backend)
    candidates.extend(["TkAgg", "QtAgg", "Qt5Agg", "GTK3Agg", "WXAgg"])

    for backend in candidates:
        try:
            matplotlib.use(backend, force=True)
            return
        except Exception:
            continue

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default=None)
    parser.add_argument("--interval", type=int, default=33)
    parser.add_argument("--file", default=None, help="Specific .json file to visualize")
    return parser.parse_args()

ARGS = parse_args()
configure_interactive_backend(ARGS.backend)

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, RadioButtons

def get_latest_file():
    ready_files = list(OUTPUT_DIR.glob("*_ready.json"))
    if ARGS.file:
        path = Path(ARGS.file)
        if path.exists(): return path
        path = OUTPUT_DIR / ARGS.file
        if path.exists(): return path
    if not ready_files:
        return None
    return max(ready_files, key=os.path.getmtime)

FILE_PATH = get_latest_file()
if not FILE_PATH:
    print(f"Error: Could not find any *_ready.json files in {OUTPUT_DIR}")
    print("Make sure you run npz_to_json.py first!")
    raise SystemExit(1)

print(f"Visualizing JSON: {FILE_PATH.name}")
with open(FILE_PATH, 'r') as f:
    data = json.load(f)

is_video = data["is_video"]
num_frames = data["num_frames"]

body_seq  = np.array([frame["body"]       for frame in data["frames"]])
left_seq  = np.array([frame["left_hand"]  for frame in data["frames"]])
right_seq = np.array([frame["right_hand"] for frame in data["frames"]])

interactive_backends = {
    backend.lower()
    for backend in backend_registry.list_builtin(BackendFilter.INTERACTIVE)
}
if plt.get_backend().lower() not in interactive_backends:
    print("No interactive Matplotlib backend is available. Check README.")
    raise SystemExit(1)

# ── figure layout ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 10))
ax  = fig.add_axes([0.07, 0.28, 0.88, 0.69], projection="3d")

# ── shared mutable state ───────────────────────────────────────────────────────
state = {
    "frame":    0,
    "playing":  is_video and num_frames > 1,
    # zoom_scale: the half-width of the view cube (smaller = more zoomed in)
    # starts at 1.0 (full body), can go down to 0.01 (single fingertip)
    "zoom":     1.0,
    # pan centre — we shift the view window around this point
    "cx": 0.0, "cy": 0.0, "cz": 0.0,
    # saved camera angles so clear() doesn't reset them
    "elev": 20.0, "azim": -60.0,
    # focus mode: "body" | "left" | "right" | "both_hands"
    "focus": "body",
}

ZOOM_MIN   = 0.01   # can zoom in to 1 cm detail
ZOOM_MAX   = 2.0    # max zoom-out
ZOOM_STEP  = 0.08   # per scroll tick (multiplicative)

# ── widget axes ────────────────────────────────────────────────────────────────
if is_video and num_frames > 1:
    slider_ax  = fig.add_axes([0.12, 0.16, 0.65, 0.03])
    zoom_ax    = fig.add_axes([0.12, 0.11, 0.65, 0.02])
    button_ax  = fig.add_axes([0.80, 0.145, 0.08, 0.045])
    focus_ax   = fig.add_axes([0.80, 0.04,  0.17, 0.10])
    info_ax    = fig.add_axes([0.12, 0.04,  0.65, 0.04])
    info_ax.axis("off")
    info_text  = info_ax.text(0.0, 0.5, "", va="center", fontsize=8,
                               family="monospace", transform=info_ax.transAxes)

    frame_slider = Slider(slider_ax, "Frame", 0, num_frames - 1,
                          valinit=0, valstep=1)
    zoom_slider  = Slider(zoom_ax,   "Zoom",  ZOOM_MIN, ZOOM_MAX,
                          valinit=1.0, valstep=0.001)
    play_button  = Button(button_ax, "Pause")
    radio        = RadioButtons(focus_ax, ("Body", "Left", "Right", "Hands"),
                                active=0)
else:
    # static frame — just a zoom slider + focus radio
    zoom_ax   = fig.add_axes([0.12, 0.16, 0.65, 0.02])
    focus_ax  = fig.add_axes([0.80, 0.04,  0.17, 0.10])
    info_ax   = fig.add_axes([0.12, 0.04,  0.65, 0.06])
    info_ax.axis("off")
    info_text = info_ax.text(0.0, 0.5, "", va="center", fontsize=8,
                              family="monospace", transform=info_ax.transAxes)

    zoom_slider = Slider(zoom_ax,  "Zoom", ZOOM_MIN, ZOOM_MAX,
                         valinit=1.0, valstep=0.001)
    radio       = RadioButtons(focus_ax, ("Body", "Left", "Right", "Hands"),
                               active=0)

# ── skeleton definitions ───────────────────────────────────────────────────────
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
]

FINGER_TIP_INDICES = [4, 8, 12, 16, 20]   # thumb … pinky tips
FINGER_NAMES       = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

BODY_LABELS = {
    0: "Nose", 7: "L Ear", 8: "R Ear",
    11: "L Shoulder", 12: "R Shoulder",
    13: "L Elbow",    14: "R Elbow",
    15: "L Wrist",    16: "R Wrist",
}

# ── auto-centre helper ─────────────────────────────────────────────────────────
def compute_focus_centre(body, left, right, focus):
    """Return (cx, cy, cz) for the chosen focus region."""
    if focus == "left":
        pts = left[np.any(left != 0, axis=1)]
    elif focus == "right":
        pts = right[np.any(right != 0, axis=1)]
    elif focus == "both_hands":
        both = np.vstack([left, right])
        pts  = both[np.any(both != 0, axis=1)]
    else:  # body
        pts = body[np.any(body != 0, axis=1)]

    if len(pts) == 0:
        return 0.0, 0.0, 0.0
    return pts[:,0].mean(), pts[:,1].mean(), pts[:,2].mean()

# ── main draw ------------------------------------------------------------------
def draw_frame(frame_index):
    body  = body_seq[frame_index]
    left  = left_seq[frame_index]
    right = right_seq[frame_index]

    # auto-centre whenever user pressed a focus button
    if state.get("recentre", False):
        cx, cy, cz = compute_focus_centre(body, left, right, state["focus"])
        state["cx"], state["cy"], state["cz"] = cx, cy, cz
        state["recentre"] = False

    cx, cy, cz = state["cx"], state["cy"], state["cz"]
    s = state["zoom"]   # half-width of view cube

    # ── clear & restore camera ─────────────────────────────────────────────────
    ax.clear()
    ax.view_init(elev=state["elev"], azim=state["azim"])

    ax.set_xlim([cx - s, cx + s])
    ax.set_ylim([cy + s, cy - s])   # Y is flipped in image coords
    ax.set_zlim([cz - s, cz + s])

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z (depth)")
    ax.set_title(f"{FILE_PATH.name}  —  frame {frame_index}/{num_frames-1}"
                 f"   zoom={1.0/s:.1f}×   focus={state['focus']}")

    body_valid  = np.any(body  != 0, axis=1)
    left_valid  = np.any(left  != 0, axis=1)
    right_valid = np.any(right != 0, axis=1)

    # ── body (red) ─────────────────────────────────────────────────────────────
    ax.scatter(body[body_valid,0], body[body_valid,1], body[body_valid,2],
               c="red", s=50, label="Body", zorder=5)
    for a, b in POSE_CONNECTIONS:
        if body_valid[a] and body_valid[b]:
            ax.plot([body[a,0], body[b,0]],
                    [body[a,1], body[b,1]],
                    [body[a,2], body[b,2]], "r-", lw=2)

    # ── left hand (green) ──────────────────────────────────────────────────────
    ax.scatter(left[left_valid,0], left[left_valid,1], left[left_valid,2],
               c="limegreen", s=25, label="Left Hand", zorder=5)
    for a, b in HAND_CONNECTIONS:
        if left_valid[a] and left_valid[b]:
            ax.plot([left[a,0], left[b,0]],
                    [left[a,1], left[b,1]],
                    [left[a,2], left[b,2]], color="limegreen", lw=1.5)
    # fingertip labels (left)
    for tip_idx, name in zip(FINGER_TIP_INDICES, FINGER_NAMES):
        if tip_idx < len(left) and left_valid[tip_idx]:
            ax.scatter(*left[tip_idx], c="darkgreen", s=60, zorder=6)
            ax.text(left[tip_idx,0], left[tip_idx,1], left[tip_idx,2],
                    f"L-{name}", fontsize=7, color="darkgreen")

    # ── right hand (blue) ──────────────────────────────────────────────────────
    ax.scatter(right[right_valid,0], right[right_valid,1], right[right_valid,2],
               c="dodgerblue", s=25, label="Right Hand", zorder=5)
    for a, b in HAND_CONNECTIONS:
        if right_valid[a] and right_valid[b]:
            ax.plot([right[a,0], right[b,0]],
                    [right[a,1], right[b,1]],
                    [right[a,2], right[b,2]], color="dodgerblue", lw=1.5)
    # fingertip labels (right)
    for tip_idx, name in zip(FINGER_TIP_INDICES, FINGER_NAMES):
        if tip_idx < len(right) and right_valid[tip_idx]:
            ax.scatter(*right[tip_idx], c="darkblue", s=60, zorder=6)
            ax.text(right[tip_idx,0], right[tip_idx,1], right[tip_idx,2],
                    f"R-{name}", fontsize=7, color="darkblue")

    # ── body landmark labels ───────────────────────────────────────────────────
    for idx, label in BODY_LABELS.items():
        if idx < len(body) and body_valid[idx]:
            ax.text(body[idx,0], body[idx,1], body[idx,2],
                    label, fontsize=8, color="darkred")

    ax.legend(loc="upper right", fontsize=8)

    # ── info bar ───────────────────────────────────────────────────────────────
    tips_info = []
    for tip_idx, name in zip(FINGER_TIP_INDICES, FINGER_NAMES):
        lx = f"({left[tip_idx,0]:.3f},{left[tip_idx,1]:.3f},{left[tip_idx,2]:.3f})" \
             if tip_idx < len(left) and left_valid[tip_idx] else "—"
        rx = f"({right[tip_idx,0]:.3f},{right[tip_idx,1]:.3f},{right[tip_idx,2]:.3f})" \
             if tip_idx < len(right) and right_valid[tip_idx] else "—"
        tips_info.append(f"{name[0]}: L{lx} R{rx}")
    info_text.set_text("  |  ".join(tips_info))

    fig.canvas.draw_idle()

# ── camera-angle tracking ──────────────────────────────────────────────────────
def on_mouse_release(event):
    """Capture camera angles after user rotates the 3-D view."""
    if event.inaxes is ax:
        state["elev"] = ax.elev
        state["azim"] = ax.azim

fig.canvas.mpl_connect("button_release_event", on_mouse_release)

# ── mouse wheel zoom ───────────────────────────────────────────────────────────
def on_scroll(event):
    if event.inaxes is not ax:
        return
    if event.button == "up":
        new_zoom = max(ZOOM_MIN, state["zoom"] * (1.0 - ZOOM_STEP))
    else:
        new_zoom = min(ZOOM_MAX, state["zoom"] * (1.0 + ZOOM_STEP))
    state["zoom"] = new_zoom
    zoom_slider.set_val(new_zoom)   # keeps slider in sync
    draw_frame(state["frame"])

fig.canvas.mpl_connect("scroll_event", on_scroll)

# ── zoom slider ────────────────────────────────────────────────────────────────
def on_zoom_change(val):
    state["zoom"] = float(val)
    draw_frame(state["frame"])

zoom_slider.on_changed(on_zoom_change)

# ── focus radio ───────────────────────────────────────────────────────────────
FOCUS_MAP = {"Body": "body", "Left": "left", "Right": "right", "Hands": "both_hands"}

def on_focus_change(label):
    state["focus"] = FOCUS_MAP[label]
    state["recentre"] = True       # snap view centre to this region
    draw_frame(state["frame"])

radio.on_clicked(on_focus_change)

# ── video-only controls ────────────────────────────────────────────────────────
if is_video and num_frames > 1:
    def on_slider_change(val):
        state["frame"] = int(val)
        draw_frame(state["frame"])
    frame_slider.on_changed(on_slider_change)

    def on_toggle_play(_event):
        state["playing"] = not state["playing"]
        play_button.label.set_text("Pause" if state["playing"] else "Play")
    play_button.on_clicked(on_toggle_play)

    def on_key_press(event):
        if event.key == " ":
            on_toggle_play(None)
        elif event.key == "right":
            frame_slider.set_val(min(state["frame"] + 1, num_frames - 1))
        elif event.key == "left":
            frame_slider.set_val(max(state["frame"] - 1, 0))
        elif event.key == "+":
            zoom_slider.set_val(max(ZOOM_MIN, state["zoom"] * (1.0 - ZOOM_STEP * 2)))
        elif event.key == "-":
            zoom_slider.set_val(min(ZOOM_MAX, state["zoom"] * (1.0 + ZOOM_STEP * 2)))
    fig.canvas.mpl_connect("key_press_event", on_key_press)

    def animate(_):
        if state["playing"]:
            state["frame"] = (state["frame"] + 1) % num_frames
            frame_slider.set_val(state["frame"])

    draw_frame(0)
    ani = FuncAnimation(fig, animate, interval=max(1, ARGS.interval),
                        cache_frame_data=False)
else:
    def on_key_press(event):
        if event.key == "+":
            zoom_slider.set_val(max(ZOOM_MIN, state["zoom"] * (1.0 - ZOOM_STEP * 2)))
        elif event.key == "-":
            zoom_slider.set_val(min(ZOOM_MAX, state["zoom"] * (1.0 + ZOOM_STEP * 2)))
    fig.canvas.mpl_connect("key_press_event", on_key_press)
    draw_frame(0)

fig.text(0.01, 0.01,
         "Scroll wheel = zoom  |  +/- keys = zoom  |  Focus buttons = auto-centre  |  Drag = rotate",
         fontsize=7, color="gray")

try:
    plt.show()
except KeyboardInterrupt:
    pass