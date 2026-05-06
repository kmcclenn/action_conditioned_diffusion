"""
Precompute frozen ViT features for every frame under frames/{split}/.

Currently supported encoders:
  - "dinov2": DINOv2 ViT-S/14, CLS token, shape (N, 384). torch.hub.
  - "croco":  CroCo V2 ViT-Base encoder only, all patch tokens, shape
              (N, 196, 768). Uses CroCoNet._encode_image(do_mask=False),
              which runs only patch_embed -> enc_blocks (attn + MLP, RoPE
              inside attn) -> enc_norm. Decoder, cross-attention, mask
              generator, and prediction head are never called. Requires
              the naver/croco repo vendored at third_party/croco/ and a
              CroCo V2 .pth checkpoint passed via --croco_ckpt.

The encoder is frozen and has no input augmentation, so features are
deterministic per frame — caching once turns each subsequent inverse-dynamics
training run into a pure MLP-head problem (seconds per epoch).

Output layout:
    {features_root}/{split}/{clip_id}.pt  — dict with:
        "timestamps": LongTensor   (N,)     microsecond timestamps, sorted
        "features":   FloatTensor            fp32; shape depends on encoder
                                             ("dinov2" -> (N, 384),
                                              "croco"  -> (N, 196, 768))
        "encoder":    str                    "dinov2" | "croco"
        "feat_dim":   int                    last-axis size of features

Usage:
    python cache_features.py --frames_root frames --features_root features
    python cache_features.py --encoder croco \\
        --features_root features_croco --batch_size 64
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Callable

# DINOv2 uses bicubic pos-embed interpolation, which MPS does not implement.
# Fall back to CPU for that single op. No-op on CPU/CUDA backends.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from PIL import Image
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# Public: callers (inference, tests) reuse this exact transform so their
# tokens are byte-identical to the cached ones. Both DINOv2 ViT-S/14 and
# (eventually) CroCo V2 ViT-Base are pretrained at 224x224 with ImageNet
# normalization, so a single transform serves both.
PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# Forward functions take a preprocessed batch (B, 3, 224, 224) and return a
# tensor whose first dim is B. Per-frame shape is encoder-specific:
#   dinov2 -> (B, 384)
#   croco  -> (B, 196, 768)
EncoderForward = Callable[[torch.Tensor], torch.Tensor]


# Vendored CroCo lives at <repo>/third_party/croco. Imported lazily so the
# DINOv2 path doesn't require the submodule to be present.
_CROCO_PATH = Path(__file__).resolve().parent / "third_party" / "croco"


def _load_dinov2(device: torch.device) -> tuple[EncoderForward, int]:
    encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    encoder.eval().to(device)
    for p in encoder.parameters():
        p.requires_grad = False

    def forward(x: torch.Tensor) -> torch.Tensor:
        return encoder.forward_features(x)["x_norm_clstoken"]

    return forward, 384


def _load_croco(device: torch.device, ckpt: Path) -> tuple[EncoderForward, int]:
    if not _CROCO_PATH.exists():
        raise FileNotFoundError(
            f"CroCo submodule not found at {_CROCO_PATH}. Run "
            f"`git submodule add https://github.com/naver/croco third_party/croco` "
            f"and `git submodule update --init`."
        )
    if not ckpt.exists():
        raise FileNotFoundError(f"CroCo checkpoint not found: {ckpt}")
    if str(_CROCO_PATH) not in sys.path:
        sys.path.insert(0, str(_CROCO_PATH))
    from models.croco import CroCoNet  # type: ignore

    state = torch.load(ckpt, map_location="cpu")
    model = CroCoNet(**state.get("croco_kwargs", {}))
    msg = model.load_state_dict(state["model"], strict=True)
    print(f"[croco] load_state_dict: {msg}")
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False

    # Probe feat_dim via a dummy forward through the encoder-only path. This
    # also validates that _encode_image works on this checkpoint before we
    # start a long caching run.
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224, device=device)
        feat, _, _ = model._encode_image(dummy, do_mask=False)
    feat_dim = int(feat.shape[-1])
    print(f"[croco] encoder probe: feat shape {tuple(feat.shape)}")

    def forward(x: torch.Tensor) -> torch.Tensor:
        # _encode_image runs ONLY: patch_embed -> enc_blocks (attn + MLP,
        # RoPE inside attn) -> enc_norm. Returns (feat, pos, masks); we
        # drop pos/masks and keep all patch tokens (no pooling).
        feat, _, _ = model._encode_image(x, do_mask=False)
        return feat  # (B, N_patches, D)

    return forward, feat_dim


def load_encoder(
    name: str,
    device: torch.device,
    croco_ckpt: Path | None = None,
) -> tuple[EncoderForward, int]:
    """Return (forward_fn, feat_dim).

    forward_fn maps preprocessed (B, 3, 224, 224) -> tensor with leading dim B.
    feat_dim is the last-axis size (384 for dinov2; 768 for croco).
    """
    if name == "dinov2":
        return _load_dinov2(device)
    if name == "croco":
        if croco_ckpt is None:
            raise ValueError("--croco_ckpt is required for --encoder croco")
        return _load_croco(device, croco_ckpt)
    raise ValueError(f"unknown encoder {name!r} (expected 'dinov2' or 'croco')")


@torch.no_grad()
def cache_split(
    frames_root: Path,
    features_root: Path,
    split: str,
    forward_fn: EncoderForward,
    feat_dim: int,
    encoder_name: str,
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
            z = forward_fn(x)
            feats.append(z.float().cpu())

        torch.save(
            {
                "timestamps": timestamps,
                "features":   torch.cat(feats, dim=0),
                "encoder":    encoder_name,
                "feat_dim":   feat_dim,
            },
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
    parser.add_argument("--encoder", choices=["dinov2", "croco"], default="dinov2")
    parser.add_argument("--croco_ckpt",
                        default="checkpoints/croco/CroCo_V2_ViTBase_SmallDecoder.pth",
                        help="Path to CroCo V2 checkpoint (.pth). "
                             "Used only when --encoder croco.")
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
    print(f"device: {device}  encoder: {args.encoder}")

    forward_fn, feat_dim = load_encoder(
        args.encoder, device,
        croco_ckpt=Path(args.croco_ckpt) if args.croco_ckpt else None,
    )
    print(f"feat_dim: {feat_dim}")

    frames_root   = Path(args.frames_root)
    features_root = Path(args.features_root)
    for split in args.splits:
        if not (frames_root / split).exists():
            print(f"skipping {split}: {frames_root / split} does not exist")
            continue
        print(f"\ncaching {split}...")
        cache_split(
            frames_root, features_root, split,
            forward_fn, feat_dim, args.encoder, device,
            batch_size=args.batch_size, overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
