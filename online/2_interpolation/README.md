# 2. Keypose retrieval and SLERP interpolation

`live_interpolate.py` turns a list of glosses into an animation (paper §3.2):

1. loads each gloss's corrected keyposes from `output/<gloss>/corrected/*.npz`, keeping their original
   frame spacing (the frame number is in the file name);
2. places the glosses one after another with a fixed gap between them (`word_gap`);
3. interpolates every SMPL-X joint with SLERP and returns per-frame bone quaternions for three.js.

```python
from live_interpolate import interpolate_sentence
result = interpolate_sentence(["Neutral_pose", "He", "Home", "come", "Neutral_pose"], word_gap=15)
```

Needs `hand_means.npz` in this folder (from `../../offline/4_avatar_export`). The web widget fills
`output/` with links into the Gizmo tool's `corrected/` folder.
