# Evaluation

## Gloss retrieval (paper §4.2)

Does manual correction make the avatar sign closer to the real signer? Each gloss's real-signer video and
its two avatar videos (raw and corrected keyposes) are embedded with
[Uni-Sign](https://github.com/ZechengLi19/Uni-Sign); each avatar video is ranked against all real-signer
videos by cosine similarity and Top-k accuracy is reported.

- `retrieval/pose_extract_all.py`: pose extraction for Uni-Sign's input format
- `retrieval/extract_unisign_embeddings.py`, `retrieval/embed.py`: Uni-Sign embeddings
  (see `retrieval/README_unisign_features.md`)
- `retrieval/retrieval.py`: Top-1/3/5/10 retrieval
- `retrieval/setup_and_run.sh`, `retrieval/run_full600.sh`: end-to-end runs

## Rendering

- `rendering/render_smplx_video.py`: renders SMPL-X keyposes to an avatar video
- `rendering/smplx_to_bvh.py`: exports SMPL-X motion as BVH

Uni-Sign, SMPL-X and the sign videos are not included; set their paths at the top of each script.
