#!/bin/bash
# One nohup chain: build venv -> smoke test (2 videos per folder) -> full 600 run.
# Start: nohup bash setup_and_run.sh > setup_and_run.log 2>&1 &
cd ~/Research/RecognitionTasks/unisign_run
R=~/Research/RecognitionTasks
V=~/Research/RecognitionTasks/unisign_venv
echo "SETUP START $(date)"
if [ ! -x $V/bin/python ]; then
  ~/miniconda3/bin/python -m venv $V || { echo "SETUP FAILED venv"; exit 1; }
fi
$V/bin/pip install -q --upgrade pip
$V/bin/pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu || { echo "SETUP FAILED torch"; exit 1; }
$V/bin/pip install -q numpy transformers sentencepiece einops decord tqdm pillow onnxruntime opencv-python-headless scipy || { echo "SETUP FAILED deps"; exit 1; }
PYTHONPATH=$R/unisign_feature_extraction/Uni-Sign/demo/rtmlib-main $V/bin/python -c "import torch, torchvision, transformers, decord, onnxruntime, rtmlib, cv2, einops, numpy; print('ENV_OK torch', torch.__version__, 'tv', torchvision.__version__, 'tf', transformers.__version__, 'np', numpy.__version__)" || { echo "SETUP FAILED import"; exit 1; }

# Smoke test: 2 glosses per folder through pose + RGB+pose embedding.
echo "SMOKE START $(date)"
S=smoke; rm -rf $S; mkdir -p $S/videos
for d in RGB_Videos Not_Corrected_videos Generated_Videos; do
  mkdir -p $S/videos/$d; cp $R/$d/Bandit.mp4 $R/$d/airport.mp4 $S/videos/$d/
done
export OMP_NUM_THREADS=6 TORCH_THREADS=6
$V/bin/python pose_extract_all.py --root $S/videos --out $S/poses || { echo "SMOKE FAILED pose"; exit 1; }
for d in RGB_Videos Generated_Videos; do
  $V/bin/python embed.py --checkpoint $R/unisign_feature_extraction/checkpoints/wlasl_rgb_pose_islr.pth \
     --pose-root $S/poses/$d --video-root $S/videos/$d --rgb-support --device cpu --output $S/$d.pt \
     || { echo "SMOKE FAILED embed $d"; exit 1; }
done
$V/bin/python -c "
import torch; a=torch.load('$S/RGB_Videos.pt'); b=torch.load('$S/Generated_Videos.pt')
print('SMOKE ids', [m['id'] for m in a['metadata']], 'shape', tuple(a['embeddings'].shape)); print('SMOKE cos gen x rgb', (b['embeddings']@a['embeddings'].T).numpy().round(3).tolist())"
echo "SMOKE_OK $(date)"

bash run_full600.sh
