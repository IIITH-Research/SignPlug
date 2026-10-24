# SignPlug - ISL plugin for the web

Code for **SignPlug**, a lightweight, real-time framework that synthesizes Indian Sign Language (ISL)
avatar motion from English sentences directly in the browser (ICVGIP 2026).

**Paper:** [OpenReview](https://openreview.net/forum?id=nFcPxtNxsX) ·
**Project page:** [sharif-commits.github.io/SignPlug](https://sharif-commits.github.io/SignPlug/)

## Pipeline

The code follows the two stages of the paper.

| Stage | Step | Folder |
|---|---|---|
| **Offline: Widget Dictionary construction** (paper §3.1) | 1. SMPL-X pose estimation from ISLRTC dictionary videos (SMPLer-X) | [`offline/1_pose_estimation`](offline/1_pose_estimation) |
| | 2. Sparse keyframe selection (start, end and keyframes per sign) | [`offline/2_keyframe_selection`](offline/2_keyframe_selection) |
| | 3. Manual pose correction with the Gizmo tool | [`offline/3_gizmo_correction`](offline/3_gizmo_correction) |
| | 4. Canonical SMPL-X avatar export for three.js | [`offline/4_avatar_export`](offline/4_avatar_export) |
| **Online: sentence-level synthesis** (paper §3.2) | 1. English sentence → ISL glosses (demo translator) | [`online/1_gloss_translation`](online/1_gloss_translation) |
| | 2. Keypose retrieval + per-bone SLERP interpolation | [`online/2_interpolation`](online/2_interpolation) |
| | 3. Web widget: server + three.js avatar playback | [`online/3_web_widget`](online/3_web_widget) |
| **Evaluation** (paper §4) | Gloss retrieval with Uni-Sign embeddings, avatar rendering | [`evaluation`](evaluation) |

```
offline/
  1_pose_estimation/      SMPLer-X inference script + frame extraction notebook
  2_keyframe_selection/   desktop keyframe annotator (app.py)
  3_gizmo_correction/     Gizmo: browser tool for correcting SMPL-X keyposes
  4_avatar_export/        export_smplx_avatar.py: rigged canonical avatar (GLB) + hand mean poses
online/
  1_gloss_translation/    rule-based demo translator (isl/), tests and test cases
  2_interpolation/        live_interpolate.py
  3_web_widget/           Flask server + page with the signing avatar
evaluation/
  retrieval/              Uni-Sign pose extraction, embeddings and Top-k retrieval
  rendering/              render SMPL-X keyposes to video
prototypes/               early MediaPipe skeletonization and skinning experiments
```

## What is not included

- **SMPL-X / SMPLer-X model files.** They are licensed separately; download them from
  [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de/) and the
  [SMPLer-X repository](https://github.com/caizhongang/SMPLer-X). The avatar GLB and `hand_means.npz` are
  generated from them by `offline/4_avatar_export/export_smplx_avatar.py`.
- **Sign videos and the Widget Dictionary keyposes.** The source videos belong to the ISLRTC dictionary,
  and the dictionary is not released with this code.

## Running the web widget

```bash
cd online/1_gloss_translation && pip install -r requirements.txt
python -c "import stanza, nltk; stanza.download('en'); nltk.download('wordnet'); nltk.download('omw-1.4')"
pip install flask numpy scipy
# put hand_means.npz in online/2_interpolation/ and the avatar GLB in online/3_web_widget/static/assets/
SIGN_ANNOTATION_DIR=/path/to/sign_annotation python online/3_web_widget/server.py --port 8080
```

`SIGN_ANNOTATION_DIR` must contain `raw/<gloss>/` and `corrected/<gloss>_corrected/*.npz`, the layout
written by the keyframe and Gizmo tools. See each folder's README for details.

## Development history

Early work (MediaPipe skeletonization, skinning, axis conventions) is kept in [`prototypes/`](prototypes)
and in the handover branches (`skeleton-code`, `skin-code`, `rotations-handover`,
`interpolations-handover`, `app-handover`). The [issues](https://github.com/sharif-commits/ISL-AVATAR-PROJECT/issues)
record the problems we ran into along the way.

## Acknowledgements

This work is supported by MeitY, Government of India, through the NLTM-Bhashini project.
The demo translator implements the rules of Ghosh and Mamidi,
*English To Indian Sign Language: Rule-Based Translation System Along With Multi-Word Expressions and
Synonym Substitution*, ICON 2022. We build on [SMPL-X](https://smpl-x.is.tue.mpg.de/),
[SMPLer-X](https://github.com/caizhongang/SMPLer-X), [three.js](https://threejs.org/),
[Stanza](https://stanfordnlp.github.io/stanza/) and [Uni-Sign](https://github.com/ZechengLi19/Uni-Sign).

## Citation

The BibTeX entry will be added once the ICVGIP 2026 proceedings are published.
