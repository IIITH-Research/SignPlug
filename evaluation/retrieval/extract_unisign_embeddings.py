"""
UniSign feature extractor — pose-only and RGB+pose modes.
Outputs a .pt file per run with keys: embeddings, metadata.

Usage:
  # Pose-only
  python extract_unisign_embeddings.py \
      --checkpoint checkpoints/wlasl_pose_only_islr.pth \
      --pose-root /path/to/pose_pkls \
      --output embeddings/pose_only.pt

  # RGB+Pose
  python extract_unisign_embeddings.py \
      --checkpoint checkpoints/wlasl_rgb_pose_islr.pth \
      --pose-root /path/to/pose_pkls \
      --video-root /path/to/videos \
      --rgb-support \
      --output embeddings/rgb_pose.pt
"""

import os
import sys
import argparse
import pickle
import json
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

# ── locate Uni-Sign repo ──────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UNISIGN_DIR = os.path.join(SCRIPT_DIR, "Uni-Sign")
if not os.path.exists(UNISIGN_DIR):
    raise RuntimeError(f"Uni-Sign repo not found at {UNISIGN_DIR}")
sys.path.insert(0, UNISIGN_DIR)

# mock deepspeed so we don't need it installed
import types, importlib.machinery
def _mock(name):
    m = types.ModuleType(name)
    m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    return m
_ds = _mock("deepspeed")
_ds_comm = _mock("deepspeed.comm")
_ds_acc  = _mock("deepspeed.accelerator")
class _DummyAcc:
    def device_count(self): return 1
_ds_acc.get_accelerator = _DummyAcc
sys.modules.update({"deepspeed": _ds, "deepspeed.comm": _ds_comm,
                    "deepspeed.accelerator": _ds_acc})

from datasets import load_part_kp, load_support_rgb_dict
from models import Uni_Sign


# ── args ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    p.add_argument("--pose-root",  required=True,
                   help="Dir with per-sample .pkl pose files")
    p.add_argument("--video-root", default=None,
                   help="Dir with per-sample video files (required for --rgb-support)")
    p.add_argument("--video-ext",  default=".mp4")
    p.add_argument("--pose-ext",   default=".pkl")
    p.add_argument("--rgb-support", action="store_true",
                   help="Use RGB+Pose mode (requires --video-root)")
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--output",     required=True, help="Output .pt path")
    p.add_argument("--device",     default="cuda")
    p.add_argument("--metadata-jsonl", default=None,
                   help="Optional: path to write per-sample metadata JSONL")
    return p.parse_args()


# ── helpers ───────────────────────────────────────────────────────────────────
class DummyArgs:
    def __init__(self, **kw):
        for k, v in kw.items(): setattr(self, k, v)


def load_model(checkpoint_path, rgb_support, max_length, device):
    model_args = DummyArgs(
        dataset="WLASL",
        max_length=max_length,
        rgb_support=rgb_support,
        hidden_dim=256,
        label_smoothing=0.1,
    )
    original_cwd = os.getcwd()
    os.chdir(UNISIGN_DIR)
    try:
        model = Uni_Sign(args=model_args).to(device)
    finally:
        os.chdir(original_cwd)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    ret = model.load_state_dict(state, strict=True)
    if ret.missing_keys:
        print(f"[warn] Missing keys: {ret.missing_keys}")
    if ret.unexpected_keys:
        print(f"[warn] Unexpected keys: {ret.unexpected_keys}")
    model.eval()
    return model


def load_pose(pkl_path, max_length):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    skeletons = data["keypoints"]
    confs     = data["scores"]
    n = len(confs)
    if n > max_length:
        indices = np.linspace(0, n - 1, max_length, dtype=int).tolist()
        skeletons = [skeletons[i] for i in indices]
        confs     = [confs[i]     for i in indices]
    return load_part_kp(skeletons, confs, force_ok=True)


def collate(pose_sample, support_rgb_dict, rgb_support):
    src = {}
    for key, tensor in pose_sample.items():
        t = tensor.float()
        src[key] = t.unsqueeze(0)
        if "attention_mask" not in src:
            L = len(t)
            mask = pad_sequence([torch.ones(L) + 7], padding_value=0, batch_first=True)
            src["attention_mask"]    = (mask != 0).long()
            src["name_batch"]        = ["sample"]
            src["src_length_batch"]  = torch.LongTensor([L])

    if rgb_support:
        for part in ["left", "right"]:
            for suf in ["sampled_indices", "skeletons_norm", "hands"]:
                k = f"{part}_{suf}"
                if k in support_rgb_dict:
                    v = support_rgb_dict[k]
                    if isinstance(v, torch.Tensor) and v.is_floating_point():
                        v = v.float()
                    src[k] = v
            src[f"{part}_rgb_len"] = [len(support_rgb_dict[f"{part}_sampled_indices"])]
    return src


@torch.no_grad()
def encode(model, src, device):
    src = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
           for k, v in src.items()}
    rgb_support_dict = {}
    if model.args.rgb_support:
        for part in ["left", "right"]:
            feat = model.rgb_support_backbone(src[f"{part}_hands"].to(device))
            rgb_support_dict[f"{part}_sampled_indices"] = src[f"{part}_sampled_indices"].to(device)
            rgb_support_dict[f"{part}_hands"]           = feat

    features = []
    body_feat = None
    for part in model.modes:
        proj = model.proj_linear[part](src[part].to(device)).permute(0, 3, 1, 2)
        gcn  = model.gcn_modules[part](proj)
        if part == "body":
            body_feat = gcn
        else:
            if part == "left":
                if model.args.rgb_support:
                    gcn = model.gather_feat_pose_rgb(
                        gcn, rgb_support_dict["left_hands"],
                        rgb_support_dict["left_sampled_indices"],
                        src["left_rgb_len"],
                        src["left_skeletons_norm"].to(device))
                gcn = gcn + body_feat[..., -2][..., None].detach()
            elif part == "right":
                if model.args.rgb_support:
                    gcn = model.gather_feat_pose_rgb(
                        gcn, rgb_support_dict["right_hands"],
                        rgb_support_dict["right_sampled_indices"],
                        src["right_rgb_len"],
                        src["right_skeletons_norm"].to(device))
                gcn = gcn + body_feat[..., -1][..., None].detach()
            elif part == "face_all":
                gcn = gcn + body_feat[..., 0][..., None].detach()
        gcn  = model.fusion_gcn_modules[part](gcn)
        pool = gcn.mean(-1).transpose(1, 2)
        features.append(pool)

    tokens  = torch.cat(features, dim=-1) + model.part_para
    tokens  = model.pose_proj(tokens)
    mask    = src["attention_mask"].unsqueeze(-1).to(tokens.dtype)
    pooled  = (tokens * mask).sum(1) / mask.sum(1).clamp_min(1.0)
    return F.normalize(pooled.float(), p=2, dim=-1).squeeze(0).cpu()


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  RGB+Pose: {args.rgb_support}")

    if args.rgb_support and not args.video_root:
        raise ValueError("--video-root is required when using --rgb-support")

    model = load_model(args.checkpoint, args.rgb_support, args.max_length, device)

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # collect all pose files
    pose_files = sorted([
        f for f in os.listdir(args.pose_root) if f.endswith(args.pose_ext)
    ])
    print(f"Found {len(pose_files)} pose files in {args.pose_root}")

    all_embeddings = []
    all_metadata   = []

    for fname in tqdm(pose_files, desc="Extracting"):
        sample_id = fname[: -len(args.pose_ext)]
        pkl_path  = os.path.join(args.pose_root, fname)

        try:
            pose_sample = load_pose(pkl_path, args.max_length)
        except Exception as e:
            print(f"[skip] {fname}: pose load failed — {e}")
            continue

        support_rgb_dict = {}
        if args.rgb_support:
            video_path = os.path.join(args.video_root, sample_id + args.video_ext)
            if not os.path.exists(video_path):
                print(f"[skip] {fname}: video not found at {video_path}")
                continue
            with open(pkl_path, "rb") as f:
                raw = pickle.load(f)
            n = len(raw["scores"])
            indices = np.linspace(0, n - 1,
                                  min(n, args.max_length), dtype=int).tolist()
            support_rgb_dict = load_support_rgb_dict(
                np.array(indices),
                [raw["keypoints"][i] for i in indices],
                [raw["scores"][i]    for i in indices],
                video_path,
                data_transform,
            )

        src = collate(pose_sample, support_rgb_dict, args.rgb_support)
        emb = encode(model, src, device)

        all_embeddings.append(emb)
        all_metadata.append({"id": sample_id})

    if not all_embeddings:
        print("No embeddings extracted.")
        return

    embeddings_tensor = torch.stack(all_embeddings)   # [N, D]
    print(f"Extracted {len(embeddings_tensor)} embeddings  shape={tuple(embeddings_tensor.shape)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save({"embeddings": embeddings_tensor, "metadata": all_metadata}, args.output)
    print(f"Saved → {args.output}")

    if args.metadata_jsonl:
        with open(args.metadata_jsonl, "w") as f:
            for item in all_metadata:
                f.write(json.dumps(item) + "\n")
        print(f"Metadata → {args.metadata_jsonl}")


if __name__ == "__main__":
    main()
