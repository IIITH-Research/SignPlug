#!/bin/bash
# Full 600-gloss Uni-Sign retrieval run (SignPlug rebuttal). Resumable: rerun to continue.
#   1. RTMPose whole-body poses for RGB / Not_Corrected / Generated videos  -> poses/
#   2. Uni-Sign WLASL RGB+pose embeddings (as the earlier 6-gloss run)      -> ../embeddings/full600_rgb_pose/
#   3. Retrieval tables, queries = Generated & Not_Corrected, gallery = RGB -> results_rgb_pose/
#   4. Same with the pose-only checkpoint (cheap extra, poses already done) -> results_pose_only/
# Start: nohup bash run_full600.sh > run.log 2>&1 &
cd ~/Research/RecognitionTasks/unisign_run
PY=~/Research/RecognitionTasks/unisign_venv/bin/python
R=~/Research/RecognitionTasks
CK=$R/unisign_feature_extraction/checkpoints
FOLDERS="RGB_Videos Not_Corrected_videos Generated_Videos"
export OMP_NUM_THREADS=6 TORCH_THREADS=6
echo "START $(date)"

nice -n 10 $PY pose_extract_all.py --root $R --out poses || { echo "POSE STAGE FAILED"; exit 1; }

embed_all() {  # $1 = tag, $2 = checkpoint, $3.. = extra args
  local tag=$1 ck=$2; shift 2
  local out=$R/embeddings/full600_$tag
  mkdir -p $out
  for d in $FOLDERS; do
    [ -e $out/$d.pt ] && { echo "skip $tag $d (exists)"; continue; }
    echo "== embed $tag $d $(date)"
    nice -n 10 $PY embed.py --checkpoint $ck --pose-root poses/$d --video-root $R/$d \
        --output $out/$d.pt --metadata-jsonl $out/$d.jsonl --device cpu "$@" || { echo "EMBED FAILED $tag $d"; exit 1; }
  done
  $PY retrieval.py --emb-dir $out --out results_$tag
}

embed_all rgb_pose $CK/wlasl_rgb_pose_islr.pth --rgb-support
echo "RGB_POSE_DONE $(date)"
embed_all pose_only $CK/wlasl_pose_only_islr.pth
echo "ALL_DONE $(date)"
