"""
Precompute DINOv2 ViT-S/14 CLS features for every frame under frames/{split}/.

The encoder is frozen and has no input augmentation, so features are
deterministic per frame — caching once turns each subsequent inverse-dynamics
training run into a pure MLP-head problem (seconds per epoch).

Output layout:
    {features_root}/{split}/{clip_id}.pt  — dict with:
        "timestamps": LongTensor  (N,)      microsecond timestamps, sorted
        "features":   FloatTensor (N, 384)  fp32 CLS tokens

Usage:
    python cache_features.py --frames_root frames --features_root features
    python cache_features.py --splits train --batch_size 128
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

# DINOv2 uses bicubic pos-embed interpolation, which MPS does not implement.
# Fall back to CPU for that single op. No-op on CPU/CUDA backends.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from PIL import Image
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# Public: callers (inference, tests) reuse this exact transform so their
# CLS tokens are byte-identical to the cached ones.
PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_encoder(device: torch.device) -> torch.nn.Module:
    encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder.to(device)


@torch.no_grad()
def cache_split(
    frames_root: Path,
    features_root: Path,
    split: str,
    encoder: torch.nn.Module,
    device: torch.device,
    batch_size: int = 64,
    overwrite: bool = False,
) -> None:
    split_in  = frames_root / split
    split_out = features_root / split
    split_out.mkdir(parents=True, exist_ok=True)

    clip_dirs = sorted(d for d in split_in.iterdir() if d.is_dir())
    if not clip_dirs:
        print(f"skipping {split}: no clip dirs in {split_in}")
        return

    total = len(clip_dirs)
    t0 = time.time()
    for i, clip_dir in enumerate(clip_dirs, start=1):
        out_path = split_out / f"{clip_dir.name}.pt"
        if out_path.exists() and not overwrite:
            continue

        jpgs = sorted(clip_dir.glob("*.jpg"), key=lambda p: int(p.stem))
        if not jpgs:
            continue

        timestamps = torch.tensor([int(p.stem) for p in jpgs], dtype=torch.long)

        feats: list[torch.Tensor] = []
        for start in range(0, len(jpgs), batch_size):
            batch_paths = jpgs[start:start + batch_size]
            x = torch.stack([PREPROCESS(Image.open(p).convert("RGB")) for p in batch_paths])
            x = x.to(device, non_blocking=True)
            z = encoder.forward_features(x)["x_norm_clstoken"]
            feats.append(z.float().cpu())

        torch.save(
            {"timestamps": timestamps, "features": torch.cat(feats, dim=0)},
            out_path,
        )

        if i % 20 == 0 or i == total:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0.0
            eta  = (total - i) / rate if rate > 0 else 0.0
            print(f"  [{split}] {i}/{total}  {rate:.1f} clips/s  eta {eta/60:.1f} min")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_root",   default="frames")
    parser.add_argument("--features_root", default="features")
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default=None,
                        help="torch device. Defaults to cuda > mps > cpu.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-encode clips whose .pt files already exist.")
    args = parser.parse_args()

    if args.device is None:
        args.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    device = torch.device(args.device)
    print(f"device: {device}")

    encoder = load_encoder(device)

    frames_root   = Path(args.frames_root)
    features_root = Path(args.features_root)
    for split in args.splits:
        if not (frames_root / split).exists():
            print(f"skipping {split}: {frames_root / split} does not exist")
            continue
        print(f"\ncaching {split}...")
        cache_split(
            frames_root, features_root, split, encoder, device,
            batch_size=args.batch_size, overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
