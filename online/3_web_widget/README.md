# 3. Web widget

A Flask server and a page where a visitor types an English sentence (or picks one of the example
sentences) and the avatar signs it.

`POST /api/translate {"sentence": "..."}` runs the demo translator (`../1_gloss_translation`), plans the
signs (glosses without a sign are fingerspelled letter by letter), and calls `interpolate_sentence()`
(`../2_interpolation`). Every sentence starts and ends with the neutral pose. The page plays the returned
frames on the avatar with three.js r158 (`static/assets/vendor`, MIT licence).

```bash
SIGN_ANNOTATION_DIR=/path/to/sign_annotation python server.py --port 8080
# or in the background: PYTHON=/path/to/python ./start.sh 8080   (stop with ./stop.sh)
```

Before the first run, put the avatar GLB at `static/assets/finalclothedmale.glb` and `hand_means.npz` in
`../2_interpolation/` (both from `../../offline/4_avatar_export`).

- `build_sentences.py`: rebuilds the example sentences in `static/sentences.json`, keeping only sentences
  whose glosses have signs (at most one fingerspelled word).
- `tools/add_letter_neutral.py`: adds a neutral keypose before each fingerspelling letter sign, so every
  letter starts from rest. Backs up the letter folders first.
