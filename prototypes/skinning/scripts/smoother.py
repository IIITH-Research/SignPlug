import os
import glob
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "src" / "output"

class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.01, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * np.pi * cutoff * t_e
        return r / (r + 1)

    def __call__(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 0.0: return self.x_prev
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

    def reset(self, t, x):
        """Reset filter state to a new value — used when tracking is re-acquired
        after a gap so the filter doesn't try to smooth across the discontinuity."""
        self.x_prev = float(x)
        self.dx_prev = 0.0
        self.t_prev = float(t)

def smooth_array(data, fps=30.0, min_cutoff=1.0, beta=0.01, hold_on_missing=True,
                 decay_factor=0.7):
    """Smooth a (T, N, D) array of landmarks using per-coordinate OneEuro filters.

    Args:
        data:            Raw landmark array, shape (T, N, D).
        fps:             Frame rate used to compute time steps.
        min_cutoff:      OneEuro min cutoff frequency.
        beta:            OneEuro speed coefficient.
        hold_on_missing: If True, hold the last smoothed value when tracking is lost
                         (good for body landmarks). If False, exponentially decay
                         toward zero (good for hands — lets JS detect tracking loss).
        decay_factor:    Multiplier applied each frame when decaying (only used when
                         hold_on_missing=False). 0.7 → reaches ~17% after 5 frames.
    """
    T, N, D = data.shape
    smoothed = np.zeros_like(data)

    filters = [[OneEuroFilter(0.0, data[0, i, d], min_cutoff=min_cutoff, beta=beta)
                for d in range(D)] for i in range(N)]

    # Track whether each landmark was missing on the previous frame so we know
    # when tracking is RE-ACQUIRED and we need to reset the filter state.
    was_missing = [np.all(data[0, i] == 0.0) for i in range(N)]

    smoothed[0] = data[0]
    for t in range(1, T):
        t_sec = t / fps
        for i in range(N):
            # Check the RAW input for missing tracking (all zeros)
            currently_missing = np.all(data[t, i] == 0.0)

            if currently_missing:
                if hold_on_missing:
                    # Body behaviour: hold last smoothed value
                    smoothed[t, i] = smoothed[t-1, i]
                else:
                    # Hand behaviour: exponentially decay toward zero
                    smoothed[t, i] = smoothed[t-1, i] * decay_factor
                was_missing[i] = True
            else:
                # Tracking is present this frame
                if was_missing[i]:
                    # Just re-acquired after a gap — reset filter to the new
                    # raw value so it doesn't try to smooth across the gap
                    for d in range(D):
                        filters[i][d].reset(t_sec, data[t, i, d])
                    smoothed[t, i] = data[t, i]
                else:
                    # Normal case: apply OneEuro filter
                    for d in range(D):
                        smoothed[t, i, d] = filters[i][d](t_sec, data[t, i, d])
                was_missing[i] = False
    return smoothed

def process_data():
    import sys
    target_file = sys.argv[1] if len(sys.argv) > 1 else None

    raw_files = []
    for f in OUTPUT_DIR.glob("*_raw.npz"):
        if target_file and target_file.lower() not in f.name.lower():
            continue
        raw_files.append(f)

    if not raw_files:
        print("No raw npz files found to smooth.")
        return

    for npz_path in raw_files:
        print(f"\nSmoothing: {npz_path.name}")
        data = np.load(npz_path)

        is_video = data["is_video"][0] if "is_video" in data else True

        if not is_video or data["body_world"].shape[0] <= 1:
            smoothed_dict = {k: data[k] for k in data.files}
        else:
            body = data["body_world"]
            left = data["left_hand"]
            right = data["right_hand"]

            # Body: hold last value on missing (reliable tracking, brief dropouts)
            smoothed_body = smooth_array(body, min_cutoff=1.0, beta=0.01,
                                         hold_on_missing=True)
            # Hands: decay to zero on missing so JS can detect tracking loss
            smoothed_left = smooth_array(left, min_cutoff=3.0, beta=0.005,
                                         hold_on_missing=False, decay_factor=0.7)
            smoothed_right = smooth_array(right, min_cutoff=3.0, beta=0.005,
                                          hold_on_missing=False, decay_factor=0.7)

            smoothed_dict = {k: data[k] for k in data.files}
            smoothed_dict["body_world"] = smoothed_body
            smoothed_dict["left_hand"] = smoothed_left
            smoothed_dict["right_hand"] = smoothed_right

        out_name = npz_path.name.replace("_raw.npz", "_ready.npz")
        out_path = OUTPUT_DIR / out_name
        np.savez(out_path, **smoothed_dict)
        print(f"Saved smoothed data to: {out_path.name}")

if __name__ == "__main__":
    process_data()