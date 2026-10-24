import numpy as np
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "src" / "output"

def convert_all():
    import sys
    target_file = sys.argv[1] if len(sys.argv) > 1 else None

    # Find npz files matching the filter
    ready_files = []
    for f in OUTPUT_DIR.glob("*.npz"):
        if target_file:
            if target_file.lower() not in f.name.lower():
                continue
        else:
            # By default, skip _raw files unless specifically requested
            if "_raw" in f.name:
                continue
        ready_files.append(f)

    if not ready_files:
        print(f"No .npz files found in {OUTPUT_DIR}")
        return

    for npz_path in ready_files:
        print(f"Processing: {npz_path.name}...")
        data = np.load(npz_path)

        is_video = bool(data["is_video"][0]) if "is_video" in data else True

        # .tolist() converts numpy arrays into standard Python lists
        body_seq = data["body_world"].tolist()
        left_seq = data["left_hand"].tolist()
        right_seq = data["right_hand"].tolist()

        # Load detection flags if available (backward compatible with old npz files)
        has_flags = "left_flag" in data.files and "right_flag" in data.files
        if has_flags:
            left_flag_seq = data["left_flag"]
            right_flag_seq = data["right_flag"]
        else:
            # Fallback: assume detected if any landmark is non-zero
            left_flag_seq = [not np.all(np.array(f) == 0.0) for f in left_seq]
            right_flag_seq = [not np.all(np.array(f) == 0.0) for f in right_seq]

        # Structure the JSON so it's easy to read in Javascript frame-by-frame
        json_data = {
            "is_video": is_video,
            "num_frames": len(body_seq),
            "frames": []
        }

        for i in range(len(body_seq)):
            json_data["frames"].append({
                "body": body_seq[i],
                "left_hand": left_seq[i],
                "right_hand": right_seq[i],
                "left_detected": bool(left_flag_seq[i]),
                "right_detected": bool(right_flag_seq[i]),
            })

        # Save it as _ready.json
        json_filename = npz_path.stem + ".json"
        json_path = OUTPUT_DIR / json_filename

        # We don't use indent=4 here because video files have thousands of points
        # and indentation makes the file size unnecessarily huge.
        with open(json_path, 'w') as f:
            json.dump(json_data, f)

        print(f"Saved JSON: {json_filename}")

if __name__ == "__main__":
    convert_all()