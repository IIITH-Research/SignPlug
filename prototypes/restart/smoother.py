import json
import math
import os
import glob

class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.01, d_cutoff=1.0):
        """
        Initialize the one euro filter.
        :param t0: start time
        :param x0: start value
        :param dx0: initial derivative
        :param min_cutoff: minimum cutoff frequency (Hz)
        :param beta: velocity coefficient
        :param d_cutoff: derivative cutoff frequency (Hz)
        """
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def __call__(self, t, x):
        """
        Compute the filtered value.
        :param t: timestamp
        :param x: new value
        :return: filtered value
        """
        t_e = t - self.t_prev
        
        # If no time has elapsed, return previous state
        if t_e <= 0.0:
            return self.x_prev
            
        # The filtered derivative of the signal.
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = a * x + (1.0 - a) * self.x_prev

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat

def process_smoothing():
    OUTPUT_DIR = "output"
    
    # Find all JSON files in output directory that haven't been smoothed yet
    files = glob.glob(os.path.join(OUTPUT_DIR, "*.json"))
    files = [f for f in files if not f.endswith("_smooth.json")]
    
    if not files:
        print("No raw JSON files found in output/ to smooth.")
        return

    for file_path in files:
        print(f"Reading {file_path}...")
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        print(f"Applying One-Euro Smoothing Filter...")
        
        # Dictionary to store a filter for every single coordinate
        # Key: (group_name, landmark_index, 'x'|'y'|'z') -> Value: OneEuroFilter instance
        filters = {}
        
        smoothed_data = []
        for frame_data in data:
            # We convert the milliseconds timestamp to seconds for the mathematical filter
            t_sec = frame_data["timestamp_ms"] / 1000.0
            
            new_frame_data = {
                "frame": frame_data["frame"],
                "timestamp_ms": frame_data["timestamp_ms"],
            }
            
            for group in ["face_landmarks", "left_hand_landmarks", "right_hand_landmarks", "pose_landmarks"]:
                landmarks = frame_data.get(group, [])
                new_landmarks = []
                
                for idx, lm in enumerate(landmarks):
                    new_lm = {}
                    for axis in ['x', 'y', 'z']:
                        val = lm[axis]
                        filter_key = (group, idx, axis)
                        
                        if filter_key not in filters:
                            # Initialize filter if this coordinate hasn't been seen yet
                            # min_cutoff: higher = less lag but more jitter, lower = more lag but less jitter
                            # beta: higher = less lag during fast movements
                            filters[filter_key] = OneEuroFilter(t_sec, val, min_cutoff=1.0, beta=0.01)
                            new_val = val
                        else:
                            # Apply the mathematical filter to the coordinate
                            new_val = filters[filter_key](t_sec, val)
                            
                        new_lm[axis] = new_val
                        
                    # Copy over visibility/presence if they exist
                    for k, v in lm.items():
                        if k not in ['x', 'y', 'z']:
                            new_lm[k] = v
                            
                    new_landmarks.append(new_lm)
                new_frame_data[group] = new_landmarks
            
            smoothed_data.append(new_frame_data)
            
        # Save output
        out_path = file_path.replace(".json", "_smooth.json")
        with open(out_path, 'w') as f:
            json.dump(smoothed_data, f)
            
        print(f"Saved highly-smoothed JSON to {out_path}\n")

if __name__ == "__main__":
    process_smoothing()
