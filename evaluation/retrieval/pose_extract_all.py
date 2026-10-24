# pose_extract_all.py — RTMPose whole-body keypoints for every video in the three
# RecognitionTasks folders, using Uni-Sign's own demo/pose_extraction.py:process_video
# (same Wholebody model, mode and normalisation). Adds per-video try/except and a
# progress TSV so one bad video can't kill a multi-hour CPU run, and reruns resume.
#
# Usage: python pose_extract_all.py --root ~/Research/RecognitionTasks --out poses
import argparse, os, sys, time, traceback

ap = argparse.ArgumentParser()
ap.add_argument('--root', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--folders', nargs='+',
                default=['Generated_Videos', 'Not_Corrected_videos', 'RGB_Videos'])
ap.add_argument('--mode', default='lightweight')   # pose_extraction.py default
ap.add_argument('--max-workers', type=int, default=2)
args = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(HERE, '..', 'unisign_feature_extraction', 'Uni-Sign', 'demo')
sys.path.insert(0, DEMO)
sys.path.insert(0, os.path.join(DEMO, 'rtmlib-main'))   # pure-Python; its setup.py won't build
from pose_extraction import process_video   # noqa: E402
from rtmlib import Wholebody                 # noqa: E402

wholebody = Wholebody(to_openpose=False, mode=args.mode, backend='onnxruntime', device='cpu')

for folder in args.folders:
    src = os.path.join(args.root, folder)
    tgt = os.path.join(args.out, folder)
    os.makedirs(tgt, exist_ok=True)
    videos = sorted(f for f in os.listdir(src) if f.endswith('.mp4') and not f.endswith('.part.mp4'))
    log = open(os.path.join(args.out, f'{folder}_progress.tsv'), 'a')
    print(f'== {folder}: {len(videos)} videos', flush=True)
    for i, f in enumerate(videos, 1):
        out = os.path.join(tgt, f[:-4] + '.pkl')
        if os.path.exists(out):
            continue
        t0 = time.time()
        try:
            process_video(os.path.join(src, f), tgt, wholebody, max_workers=args.max_workers)
            status = 'ok' if os.path.exists(out) else 'no_output (could not open video?)'
        except Exception:
            status = 'error: ' + traceback.format_exc().strip().splitlines()[-1]
        dt = time.time() - t0
        log.write(f'{f}\t{status}\t{dt:.1f}\n'); log.flush()
        print(f'[{folder} {i}/{len(videos)}] {f}: {status} ({dt:.1f}s)', flush=True)
    log.close()
print('POSE_DONE', flush=True)
