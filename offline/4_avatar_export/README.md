# 4. Canonical avatar export

`export_smplx_avatar.py` builds the canonical SMPL-X avatar used everywhere else: a rigged GLB for
three.js with identity bind-pose rotations, and `hand_means.npz`, the SMPL-X mean hand pose that
`../../online/2_interpolation/live_interpolate.py` adds back to the stored hand-pose deltas.

Run it in the SMPLer-X environment (it needs `smplx`, `torch`, `pygltflib` and the licensed SMPL-X model
files). Bone names match `SMPLX_BODY_BONES` / `SMPLX_LHAND_BONES` / `SMPLX_RHAND_BONES` in
`live_interpolate.py`; rename both together if you change one.
