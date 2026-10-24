# 3. Gizmo: manual pose correction

Browser tool for correcting the sparse SMPL-X keyposes (paper §3.1, Figure 4). The reference sign video is
shown next to the SMPL-X avatar; the annotator selects a bone, rotates it about its local X/Y/Z axes with
the on-screen gizmo, can transfer finger poses to other frames or revert a frame, and saves the corrected
keypose.

```bash
python server.py        # serves http://localhost:8000
```

`server.py` expects a `sign_annotation/` folder next to this folder's parent, with:

- `raw/<gloss>/<frame>_0.npz`: SMPL-X keyposes from steps 1–2
- `pics/<gloss>/<frame>.png`: the matching video frames
- `corrected/<gloss>_corrected/`: where corrected keyposes are saved

`app.js` loads the avatar from `assets/smplx_male.glb`; create it with
`../4_avatar_export/export_smplx_avatar.py` (not included, since it is built from SMPL-X).
