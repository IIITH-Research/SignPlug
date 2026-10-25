# embed.py — runs unisign_feature_extraction/extract_unisign_embeddings.py unchanged
# (same pose loading, RGB hand crops and encode()), with three CPU-box workarounds:
#   1. mT5-base is never built. Uni_Sign.__init__ loads it from ./pretrained_weight
#      (not shipped, ~2.3 GB, ~5 GB RAM), but encode() never touches it.
#   2. EfficientNet-B0 is built without downloading ImageNet weights; the checkpoint
#      overwrites them anyway.
#   3. The checkpoint is loaded strictly for every non-mT5 key: anything missing or
#      unexpected outside mt5_model.* aborts the run.
# All CLI arguments are passed straight through to the original script.
import os, sys
import torch, torchvision, transformers

HERE = os.path.dirname(os.path.abspath(__file__))
EXT_DIR = os.path.join(HERE, '..', 'unisign_feature_extraction')
sys.path.insert(0, EXT_DIR)


class _NoMT5(torch.nn.Module):
    pass


transformers.MT5ForConditionalGeneration.from_pretrained = classmethod(lambda cls, *a, **k: _NoMT5())
transformers.T5Tokenizer.from_pretrained = classmethod(lambda cls, *a, **k: None)
_effnet = torchvision.models.efficientnet_b0
torchvision.models.efficientnet_b0 = lambda pretrained=False, **k: _effnet(weights=None)

import extract_unisign_embeddings as ext  # noqa: E402


def load_model(checkpoint_path, rgb_support, max_length, device):
    model_args = ext.DummyArgs(dataset='WLASL', max_length=max_length, rgb_support=rgb_support,
                               hidden_dim=256, label_smoothing=0.1)
    cwd = os.getcwd()
    os.chdir(ext.UNISIGN_DIR)
    try:
        model = ext.Uni_Sign(args=model_args)
    finally:
        os.chdir(cwd)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state = ckpt['model'] if 'model' in ckpt else ckpt
    n_mt5 = sum(k.startswith('mt5_model.') for k in state)
    state = {k: v for k, v in state.items() if not k.startswith('mt5_model.')}
    model.load_state_dict(state, strict=True)   # raises on any non-mT5 mismatch
    print(f'Loaded {len(state)} tensors strictly (skipped {n_mt5} unused mT5 tensors)', flush=True)
    del ckpt, state
    return model.to(device).eval()


ext.load_model = load_model
torch.set_num_threads(int(os.environ.get('TORCH_THREADS', os.cpu_count())))

if __name__ == '__main__':
    ext.main()
