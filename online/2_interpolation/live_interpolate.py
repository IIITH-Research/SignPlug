# live_interpolator.py — Keyframe Delta-Preserving SLERP Interpolator & Three.js Exporter
#
# FIXED VERSION — changes from your original marked with "# FIX:" comments.
#
import os
import glob
import time
import json                                              # FIX: was missing, crashes on REFERENCE_JSON load
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# flat_hand_mean=False means left_hand_pose/right_hand_pose in the npz files are
# DELTAS relative to a natural relaxed-hand mean pose, not absolute rotations.
# Add the mean back so total rotation matches what actually produced the mesh.
HAND_MEANS_PATH = os.path.join(SCRIPT_DIR, "hand_means.npz")
if os.path.exists(HAND_MEANS_PATH):
    _hm = np.load(HAND_MEANS_PATH)
    LEFT_HAND_MEAN = _hm['left_hand_mean']
    RIGHT_HAND_MEAN = _hm['right_hand_mean']
else:
    print("WARNING: hand_means.npz not found — run export_smplx_avatar.py first. "
          "Hand poses will be wrong (missing mean-pose offset).")
    LEFT_HAND_MEAN = np.zeros((15, 3))
    RIGHT_HAND_MEAN = np.zeros((15, 3))

# Bone naming shared with export_smplx_avatar.py — must match EXACTLY.
# Order matters: this is the true SMPL-X kinematic order (parents array),
# NOT an arbitrary list — index i here corresponds to joint i in the model.
SMPLX_BODY_BONES = [
    'Pelvis',                                             # joint 0  -> from global_orient, NOT body_pose
    'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2',
    'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot', 'Neck',
    'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder',
    'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist'            # joints 1..21 -> from body_pose[0..20]
]

# FIX: reordered to match SMPL-X/MANO's actual left_hand_pose / right_hand_pose
# joint order: index, middle, pinky, ring, thumb (NOT thumb-first).
# Also renamed for consistency with export_smplx_avatar.py bone names.
SMPLX_LHAND_BONES = [
    'L_Index1',  'L_Index2',  'L_Index3',
    'L_Middle1', 'L_Middle2', 'L_Middle3',
    'L_Pinky1',  'L_Pinky2',  'L_Pinky3',
    'L_Ring1',   'L_Ring2',   'L_Ring3',
    'L_Thumb1',  'L_Thumb2',  'L_Thumb3',
]

SMPLX_RHAND_BONES = [
    'R_Index1',  'R_Index2',  'R_Index3',
    'R_Middle1', 'R_Middle2', 'R_Middle3',
    'R_Pinky1',  'R_Pinky2',  'R_Pinky3',
    'R_Ring1',   'R_Ring2',   'R_Ring3',
    'R_Thumb1',  'R_Thumb2',  'R_Thumb3',
]

# NOTE: if your SMPLer-X/AI4Bharat inference pipeline is confirmed to emit hand
# joints in a DIFFERENT order than standard MANO, tell me and I'll swap this back.
# Easiest sanity check: render one hand-heavy word (e.g. a fingerspelled letter)
# with this ordering and visually confirm fingers curl correctly.


def load_word_keyframes(word_name):
    """
    Loads all keyframe .npz files for a given word directory
    and parses exact frame index numbers from filenames (e.g. 00020_0.npz -> 20).
    """
    word_dir = os.path.join(OUTPUT_DIR, word_name, "corrected")
    if not os.path.exists(word_dir):
        possible_dirs = [
            os.path.join(OUTPUT_DIR, word_name.capitalize(), "corrected"),
            os.path.join(OUTPUT_DIR, word_name.lower(), "corrected"),
            os.path.join(OUTPUT_DIR, word_name.title(), "corrected"),   # NEW: handles "Thank_You" style
            os.path.join(OUTPUT_DIR, word_name.upper(), "corrected"),   # NEW: harmless safety net

        ]
        found = False
        for p in possible_dirs:
            if os.path.exists(p):
                word_dir = p
                found = True
                break
        if not found:
            return None

    npz_files = sorted(glob.glob(os.path.join(word_dir, "*.npz")))
    if not npz_files:
        return None

    orient, pose, lhand, rhand, jaw, frame_indices = [], [], [], [], [], []
    for fpath in npz_files:
        fname = os.path.basename(fpath)
        try:
            f_idx = int(fname.split('_')[0])
        except ValueError:
            f_idx = len(frame_indices)

        npz = dict(np.load(fpath, allow_pickle=True))
        orient.append(np.array(npz['global_orient']).reshape(3))
        pose.append(np.array(npz['body_pose']).reshape(21, 3))
        # FIX: add hand mean back — raw values are deltas under flat_hand_mean=False
        lhand.append(np.array(npz['left_hand_pose']).reshape(15, 3) + LEFT_HAND_MEAN)
        rhand.append(np.array(npz['right_hand_pose']).reshape(15, 3) + RIGHT_HAND_MEAN)
        jaw.append(np.array(npz['jaw_pose']).reshape(3))
        frame_indices.append(f_idx)

    return {
        'word': word_name,
        'num_keyframes': len(npz_files),
        'frame_indices': frame_indices,
        'global_orient': np.stack(orient),
        'body_pose': np.stack(pose),
        'left_hand_pose': np.stack(lhand),
        'right_hand_pose': np.stack(rhand),
        'jaw_pose': np.stack(jaw)
    }


def slerp_array(keyframe_rotvecs, key_times, query_times):
    """Performs SLERP across joints for rotation vectors."""
    K, J, _ = keyframe_rotvecs.shape
    out = np.zeros((len(query_times), J, 3))
    for j in range(J):
        rot = R.from_rotvec(keyframe_rotvecs[:, j, :])
        out[:, j, :] = Slerp(key_times, rot)(query_times).as_rotvec()
    return out


def rotvec_to_quat_dict(rotvec_array, bone_names):
    """Converts (N_joints, 3) rotvec array to a dict of bone -> quaternion [x, y, z, w]."""
    out = {}
    for i, bname in enumerate(bone_names):
        r = R.from_rotvec(rotvec_array[i])
        q = r.as_quat()
        out[bname] = [round(float(val), 6) for val in q]
    return out


REFERENCE_JSON = os.path.join(SCRIPT_DIR, "SMPLer-X", "main", "reference_stance.json")
if os.path.exists(REFERENCE_JSON):
    try:
        ref_data = json.load(open(REFERENCE_JSON))
        REF_ORIENT = np.array(ref_data['global_orient'], dtype=np.float32)
    except Exception:
        REF_ORIENT = np.array([-3.1205168, 0.019462075, -0.00084271753], dtype=np.float32)
else:
    REF_ORIENT = np.array([-3.1205168, 0.019462075, -0.00084271753], dtype=np.float32)


def interpolate_sentence(word_list, word_gap=10, target_fps=30):
    """
    On-The-Fly Keyframe Interpolator:
    Preserves exact frame spacing from .npz filenames (e.g. 20, 24, 33 -> deltas 0, 4, 13)
    with a 10-frame transition gap between words.
    Uses a FIXED reference-stance global_orient (Option A — same as your offline renderer),
    so the interpolated original signer's global_orient is computed but intentionally unused.
    """
    start_t = time.time()
    valid_words = []
    accum_params = {
        'global_orient': [],
        'body_pose': [],
        'left_hand_pose': [],
        'right_hand_pose': [],
        'jaw_pose': []
    }

    key_times = []
    current_time_offset = 0.0

    for w in word_list:
        data = load_word_keyframes(w)
        if data is not None:
            valid_words.append(w)
            raw_indices = data['frame_indices']
            first_idx = raw_indices[0]
            for idx in raw_indices:
                rel_t = current_time_offset + (idx - first_idx)
                key_times.append(rel_t)

            word_duration = raw_indices[-1] - first_idx
            current_time_offset += word_duration + word_gap

            accum_params['global_orient'].extend(data['global_orient'])
            accum_params['body_pose'].extend(data['body_pose'])
            accum_params['left_hand_pose'].extend(data['left_hand_pose'])
            accum_params['right_hand_pose'].extend(data['right_hand_pose'])
            accum_params['jaw_pose'].extend(data['jaw_pose'])

    if not valid_words or len(key_times) == 0:
        return {
            "error": f"No valid keyframe data found for words: {word_list}",
            "available_words": [d for d in os.listdir(OUTPUT_DIR) if os.path.isdir(os.path.join(OUTPUT_DIR, d, 'corrected'))]
        }

    for k in accum_params:
        accum_params[k] = np.stack(accum_params[k])

    key_times = np.array(key_times, dtype=float)
    total_frames = int(np.ceil(key_times[-1])) + 1
    query_times = np.arange(total_frames, dtype=float)

    num_keyframes = len(key_times)
    if num_keyframes == 1:
        interp_body   = accum_params['body_pose'].reshape(1, 21, 3).repeat(total_frames, axis=0)
        interp_lhand  = accum_params['left_hand_pose'].reshape(1, 15, 3).repeat(total_frames, axis=0)
        interp_rhand  = accum_params['right_hand_pose'].reshape(1, 15, 3).repeat(total_frames, axis=0)
        interp_jaw    = accum_params['jaw_pose'].reshape(1, 1, 3).repeat(total_frames, axis=0)
    else:
        kf_body   = accum_params['body_pose']
        kf_lhand  = accum_params['left_hand_pose']
        kf_rhand  = accum_params['right_hand_pose']
        kf_jaw    = accum_params['jaw_pose'].reshape(num_keyframes, 1, 3)

        interp_body   = slerp_array(kf_body, key_times, query_times)
        interp_lhand  = slerp_array(kf_lhand, key_times, query_times)
        interp_rhand  = slerp_array(kf_rhand, key_times, query_times)
        interp_jaw    = slerp_array(kf_jaw, key_times, query_times)

    frames = []
    # NOTE: REF_ORIENT / ref_orient_quat is intentionally NOT applied to Pelvis here.
    # It was tuned to stand the mesh upright for the pyrender/OpenCV camera convention
    # (Y-down) used by your offline image renderer. In the three.js viewer (Y-up,
    # OpenGL-style), applying it rotates the whole body ~180 degrees on its side.
    # The exported GLB's native rest pose already stands upright and faces forward
    # in three.js, so we simply leave Pelvis at identity (don't set it at all) and
    # let the bone stay at its rest orientation.

    for i in range(total_frames):
        frame_bones = {}

        # FIX: SMPLX_BODY_BONES[1:] (skip Pelvis), 21 entries matching body_pose[0..20]
        body_dict = rotvec_to_quat_dict(interp_body[i], SMPLX_BODY_BONES[1:])
        frame_bones.update(body_dict)

        frame_bones.update(rotvec_to_quat_dict(interp_lhand[i], SMPLX_LHAND_BONES))
        frame_bones.update(rotvec_to_quat_dict(interp_rhand[i], SMPLX_RHAND_BONES))
        frame_bones['Jaw'] = [round(float(v), 6) for v in R.from_rotvec(interp_jaw[i, 0]).as_quat()]

        frames.append(frame_bones)                                # FIX: was duplicated (appended twice)

    latency_ms = round((time.time() - start_t) * 1000, 2)

    return {
        "success": True,
        "words": valid_words,
        "sentence": " ".join(valid_words),
        "key_times": key_times.tolist(),
        "total_keyframes": num_keyframes,
        "total_frames": total_frames,
        "fps": target_fps,
        "latency_ms": latency_ms,
        "frames": frames,
    }


if __name__ == "__main__":
    test_sentence = ["He", "Home", "come"]
    res = interpolate_sentence(test_sentence)
    if res.get("success"):
        print(f"Interpolation Test: {res['sentence']}")
        print(f"Keytimes: {res.get('key_times')}")
        print(f"Generated {res.get('total_frames')} frames in {res.get('latency_ms')} ms!")
    else:
        print("ERROR:", res.get("error"))
        print("Available words:", res.get("available_words"))