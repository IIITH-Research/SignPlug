"""Prefix every fingerspelling letter sign with the neutral pose, 15 frames before the letter.

Writes into sign_annotation/corrected/<letter>_corrected/. A backup of each folder is made first.
Idempotent: a folder that already holds the neutral keyframe is left alone.
"""
import filecmp, os, re, shutil, sys, time

SIGN_DIR = os.environ.get("SIGN_ANNOTATION_DIR", os.path.expanduser("~/Research/sign_annotation"))
CORRECTED = os.path.join(SIGN_DIR, "corrected")
NEUTRAL = os.path.join(CORRECTED, "Neutral_pose_corrected", "00001_0.npz")
LETTERS = list("abcdefg") + ["H", "I_sign1"] + list("jklmnopqrstuvwx") + ["y", "z"]
MOVE = 15
backup = os.path.expanduser(time.strftime(SIGN_DIR.rstrip("/") + "_backups/letters_before_neutral_%Y%m%d_%H%M%S"))

def frames(d):
    return sorted((int(re.match(r"(\d+)_", f).group(1)), f) for f in os.listdir(d) if f.endswith(".npz"))

for letter in LETTERS:
    d = os.path.join(CORRECTED, f"{letter}_corrected")
    kf = frames(d)
    if any(filecmp.cmp(os.path.join(d, f), NEUTRAL, shallow=False) for _, f in kf):
        print(f"{letter:8s} already has neutral: {[f for _, f in kf]}")
        continue
    shutil.copytree(d, os.path.join(backup, os.path.basename(d)))
    first = kf[0][0]
    shift = 0 if first >= MOVE else MOVE        # make room before the letter if needed
    for idx, f in sorted(kf, reverse=True):     # rename last-first so names never collide
        if shift:
            os.rename(os.path.join(d, f), os.path.join(d, f"{idx + shift:05d}_0.npz"))
    neutral_idx = first + shift - MOVE
    shutil.copy2(NEUTRAL, os.path.join(d, f"{neutral_idx:05d}_0.npz"))
    print(f"{letter:8s} {[f for _, f in kf]} -> {[f for _, f in frames(d)]}")
print("backup:", backup)
