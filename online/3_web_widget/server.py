"""Live demo server: English sentence -> ISL glosses -> keyframes -> live SLERP -> three.js avatar.

The translator (../1_gloss_translation/isl) and ../2_interpolation/live_interpolate.py are used unchanged:
  1. isl.translate_detailed() turns the sentence into glosses and sign folders.
  2. Each gloss with corrected keyframes becomes a word for the interpolator; every other gloss
     is fingerspelled with the alphabet signs (a..z).
  3. live_interpolate.interpolate_sentence() interpolates the keyframes into per-frame bone
     quaternions, which the page plays on the avatar.

live_interpolate.py reads output/<word>/corrected/*.npz (next to itself), while the Gizmo tool saves
keyframes in sign_annotation/corrected/<word>_corrected/. That output/ is kept as a tree of symlinks into
the latter, refreshed on every request, so newly corrected signs show up without a restart.

Set SIGN_ANNOTATION_DIR to the folder holding raw/ and corrected/ (default ~/Research/sign_annotation).
Usage: python server.py [--port 8080]
"""
import argparse
import os
import string
import sys
import threading
import time

from flask import Flask, jsonify, request, send_from_directory

HERE = os.path.dirname(os.path.abspath(__file__))
SIGN_DIR = os.environ.get("SIGN_ANNOTATION_DIR", os.path.expanduser("~/Research/sign_annotation"))
RAW_DIR = os.path.join(SIGN_DIR, "raw")
CORRECTED_DIR = os.path.join(SIGN_DIR, "corrected")
STATIC_DIR = os.path.join(HERE, "static")
MAX_CHARS = 300
WORD_GAP = 15                                         # frames between glosses (interpolator default is 10)
NEUTRAL = "Neutral_pose"                              # rest pose: opens and closes every sentence
LETTER_SIGNS = {"i": "I_sign1"}                       # letter signs whose folder is not the bare letter

ONLINE_DIR = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ONLINE_DIR, "2_interpolation"))
sys.path.insert(0, os.path.join(ONLINE_DIR, "1_gloss_translation"))
import live_interpolate                               # noqa: E402  (unchanged)
from isl.lexicon import Lexicon                       # noqa: E402
from isl.translate import translate_detailed          # noqa: E402

OUTPUT_DIR = live_interpolate.OUTPUT_DIR              # online/2_interpolation/output

app = Flask(__name__, static_folder=None)
TRANSLATE_LOCK = threading.Lock()                     # Stanza pipelines are not thread-safe
LEXICON = Lexicon(sorted(d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))))


# ---------------------------------------------------------------- keyframes
def sync_keyframe_links():
    """Mirror sign_annotation/corrected/<w>_corrected as output/<w>/corrected. Returns {word: n_keyframes}."""
    available = {}
    for entry in os.listdir(CORRECTED_DIR):
        src = os.path.join(CORRECTED_DIR, entry)
        if not entry.endswith("_corrected") or not os.path.isdir(src):
            continue
        word = entry[: -len("_corrected")]
        n = sum(f.endswith(".npz") for f in os.listdir(src))
        if not n:
            continue
        link = os.path.join(OUTPUT_DIR, word, "corrected")
        if not os.path.islink(link):
            os.makedirs(os.path.dirname(link), exist_ok=True)
            os.symlink(src, link)
        available[word] = n
    return available


def letter_folders(available):
    """Alphabet sign for each letter. The 'I' folder is the pronoun (label I / me); the letter is I_sign1."""
    out = {}
    for ch in string.ascii_lowercase:
        for name in (LETTER_SIGNS.get(ch), ch, ch.upper()):
            if name in available and name != "I":
                out[ch] = name
                break
    if "2_Two" in available:
        out["2"] = "2_Two"
    return out


# ---------------------------------------------------------------- planning
def plan_signing(stages, available):
    """Turn translator output into (items for the page, word list for the interpolator)."""
    letters = letter_folders(available)
    items, words = [], []
    for stage in stages:
        for gloss, folder in stage["signs"]:
            if folder and folder in available:
                items.append({"gloss": gloss, "kind": "sign", "folder": folder, "words": [folder]})
                words.append(folder)
                continue
            text = gloss[3:] if gloss.startswith("fs-") else gloss
            spelled, missing = [], []
            for ch in text.lower():
                if ch in letters:
                    spelled.append(letters[ch])
                elif ch.isalnum():
                    missing.append(ch)
            items.append({"gloss": text, "kind": "fingerspell", "folder": folder,
                          "letters": [l[0].upper() if l != "2_Two" else "2" for l in spelled],   # I_sign1 -> I
                          "missing": missing, "words": spelled})
            words.extend(spelled)
    return items, words


def attach_segments(items, available, key_times, first_keyframe=0):
    """Frame range of each item, from the per-word keyframe counts (the interpolator keeps them in order)."""
    k = first_keyframe
    for item in items:
        n = sum(available[w] for w in item["words"])
        if n:
            item["start"], item["end"] = key_times[k], key_times[k + n - 1]
        k += n


# ---------------------------------------------------------------- routes
@app.post("/api/translate")
def api_translate():
    sentence = ((request.get_json(silent=True) or {}).get("sentence") or "").strip()
    if not sentence:
        return jsonify(error="Please type a sentence."), 400
    if len(sentence) > MAX_CHARS:
        return jsonify(error=f"Please keep it under {MAX_CHARS} characters."), 400

    t0 = time.time()
    with TRANSLATE_LOCK:
        stages = translate_detailed(sentence, LEXICON)
    t1 = time.time()

    available = sync_keyframe_links()
    items, words = plan_signing(stages, available)
    glosses = " ".join(item["gloss"] for item in items)
    if not words:
        return jsonify(error="None of the glosses has a sign yet.", gloss=glosses, items=items), 422

    # Open and close with the neutral rest pose; WORD_GAP frames separate it from the glosses.
    neutral = [NEUTRAL] if NEUTRAL in available else []
    result = live_interpolate.interpolate_sentence(neutral + words + neutral, word_gap=WORD_GAP)
    if not result.get("success"):
        return jsonify(error=result.get("error", "Interpolation failed."), gloss=glosses, items=items), 500
    attach_segments(items, available, result["key_times"], first_keyframe=available.get(NEUTRAL, 0) if neutral else 0)

    result.update(english=sentence, gloss=glosses, items=items,
                  timing={"translate_ms": round((t1 - t0) * 1000, 1),
                          "interpolate_ms": result["latency_ms"],
                          "total_ms": round((time.time() - t0) * 1000, 1)})
    return jsonify(result)


@app.get("/api/rest_pose")
def api_rest_pose():
    """One frame of the neutral pose, shown while the avatar waits for a sentence."""
    if NEUTRAL not in sync_keyframe_links():
        return jsonify(error=f"{NEUTRAL} has no corrected keyframes."), 404
    return jsonify(live_interpolate.interpolate_sentence([NEUTRAL], word_gap=WORD_GAP))


@app.get("/api/health")
def api_health():
    available = sync_keyframe_links()
    return jsonify(ok=True, signs=len(available), letters=sorted(letter_folders(available)))


@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/<path:path>")
def static_files(path):
    return send_from_directory(STATIC_DIR, path)


def warm_up():
    """Load Stanza and the WordNet lexicon before the first visitor arrives."""
    with TRANSLATE_LOCK:
        translate_detailed("Hello, I am happy.", LEXICON)
    sync_keyframe_links()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    warm_up()
    print(f"ISL demo ready on http://{args.host}:{args.port}", flush=True)
    app.run(host=args.host, port=args.port, threaded=True)
