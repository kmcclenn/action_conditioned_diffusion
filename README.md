# Action-Conditioned Diffusion

## Environment

The project uses a conda environment defined in [`environment.yml`](environment.yml). Create it once:

```bash
conda env create -f environment.yml
conda activate action-diffusion
```

What's included:
- `pytorch`, `torchvision` — model + data transforms
- `numpy`, `scipy` — numerics
- `pillow` — JPEG I/O used by the frame pipeline
- `ffmpeg`, `yt-dlp` — external binaries that `download_frames.py` shells out to

Update the env after `environment.yml` changes:

```bash
conda env update -f environment.yml --prune
```

Remove it:

```bash
conda env remove -n action-diffusion
```

**GPU note.** The default `pytorch` package resolves to a CPU/MPS build on macOS and to a CUDA build on Linux (CUDA runtime bundled). To pin a specific CUDA version on Linux, add e.g. `pytorch-cuda=12.1` alongside `pytorch`.

## Setup

Download the [RealEstate10K dataset](https://google.github.io/realestate10k/download.html) and place it in the top directory (it will be gitignored). The resulting layout should be:

```
RealEstate10K/
  train/*.txt
  test/*.txt
```

Each `.txt` has a YouTube URL on line 0 and one row per frame afterward (timestamp, intrinsics, pose).

## Building a unified dataset

`download_frames.py` and `dataset.py` are designed to pair. The downloader materializes JPEGs to disk keyed by the same `timestamps` that `parse_clip` returns, and `RealEstate10KDataset` picks them up when you pass `frames_root`.

### 1. Materialize frames to disk

```bash
python download_frames.py RealEstate10K/ frames/ --split train --workers 8 --img_size 256x256
python download_frames.py RealEstate10K/ frames/ --split test  --workers 8 --img_size 256x256
```

Output layout: `frames/{split}/{clip_id}/{timestamp_us}.jpg`.

**Shared subset.** The full RealEstate10K training set is ~72k clips, which is far more data (and disk) than we need. The project fixes a 2,958-clip subset (~6.5 GB) in [`clip_lists/train.txt`](clip_lists/train.txt); `download_frames.py` auto-filters to that list when it exists, so everyone on the team downloads the same clips without any extra flags. To change the subset, regenerate `clip_lists/train.txt` and commit it.

Flags:
- `--workers N` — parallel `yt-dlp` + `ffmpeg` workers
- `--limit N` — process only the first N clips (useful for smoke tests)
- `--img_size WxH` — resize frames during extraction
- `--quality Q` — JPEG quality (`ffmpeg -q:v`, 1-31, lower is better)
- `--no_skip` — re-download frames that already exist

Requires `yt-dlp` and `ffmpeg` on your `PATH`. The downloader is resumable — rerun after failures without re-downloading completed clips.

### 2. Load frames + poses together

```python
from torch.utils.data import DataLoader
from dataset import RealEstate10KDataset

ds = RealEstate10KDataset(
    root="RealEstate10K",
    split="train",
    seq_len=16,
    stride=1,
    return_actions=True,
    img_size=(256, 256),     # also scales K into pixel coords
    frames_root="frames",    # triggers image loading
    frames_only=True,        # drop windows whose JPEGs aren't on disk
)

loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=4)
batch = next(iter(loader))
# batch["images"]:  (B, seq_len,   3, H, W)   float32 in [0, 1]
# batch["K"]:       (B, seq_len,   3, 3)      pixel-coord intrinsics
# batch["P"]:       (B, seq_len,   3, 4)      camera extrinsics [R|t]
# batch["actions"]: (B, seq_len-1, 6)         SE(3) twist (v, omega) per step
```

### Gotchas

- Keep `img_size` consistent between the two steps. The on-disk resolution then matches what the loader returns, and `K` is scaled to those pixel units.
- `frames_only=True` is important. Without it, windows with any missing JPEG silently return samples without an `"images"` key, which breaks default collation.
- `parse_clip` is shared between both files, so the timestamps used to save JPEGs are exactly the ones the loader asks for — no alignment drift.

## Quick sanity check

```bash
python dataset.py RealEstate10K/ --split train --seq_len 16 --frames_root frames/
```

Prints the sample count and the shape/dtype of every tensor in `ds[0]`.

## Inverse dynamics model

A supervised model that maps a pair of frames `(I_i, I_{i+1})` to the 6D SE(3)
twist `a = (v, ω)` between them. Training is split into two stages:

1. **`cache_features.py`** — runs frozen DINOv2 ViT-S/14 over every frame once
   and dumps the CLS tokens to disk.
2. **`inverse_dynamics.py`** — trains a small MLP head on those cached features.

See [docs/inverse_dynamics.md](docs/inverse_dynamics.md) for the architecture
and [docs/action_extraction.md](docs/action_extraction.md) for how the target
is derived from the raw poses.

### 1. Cache DINOv2 features

```bash
# One-time encoder pass. Resumable — rerun after failures to fill gaps.
python cache_features.py --frames_root frames --features_root features
```

Output: `features/{split}/{clip_id}.pt` — a dict with `timestamps` (LongTensor,
N) and `features` (FloatTensor, N×384). Storage is ~680 MB total at fp32 (~617
MB train + ~62 MB test).

### 2. Train the head

```bash
python inverse_dynamics.py \
  --root RealEstate10K \
  --features_root features \
  --batch_size 64 \
  --num_epochs 50
```

What the entry point does:
1. Builds `CachedPairDataset` for `train` and `test` — loads all `.pt` files
   into one in-memory feature tensor and pairs consecutive frames by timestamp.
2. Runs `compute_action_stats(train_ds)` (reads the stacked actions tensor
   directly, no I/O) and registers the `(mean, std)` buffers on the model.
3. Trains with Adam (`lr=1e-4`), linear warmup over 1k steps, grad-clip 1.0,
   MSE on normalized targets. Best checkpoint saved to `inverse_dynamics.pt`;
   early-stops after 5 epochs without improvement.

Each epoch is well under a minute on MPS or CUDA since the feature encoder is
out of the loop.

### Inference

```python
import torch
from inverse_dynamics import InverseDynamicsModel
from cache_features import load_encoder, PREPROCESS

model = InverseDynamicsModel()
ckpt = torch.load("inverse_dynamics.pt", map_location="cpu")
model.load_state_dict(ckpt["model"])
model.eval()

# Encode a pair of raw PIL images to CLS features, then run the head.
encoder = load_encoder(torch.device("cpu"))
x = torch.stack([PREPROCESS(img_i), PREPROCESS(img_next)]).unsqueeze(0)  # (1,2,3,224,224)
with torch.no_grad():
    z = encoder.forward_features(x.view(-1, 3, 224, 224))["x_norm_clstoken"].view(1, 2, -1)
    action = model.denormalize_action(model(z[:, 0], z[:, 1]))  # (1, 6)
```

### Gotchas

- `CachedPairDataset` fails loudly if no feature pairs are found on disk —
  run `cache_features.py` first.
- `cache_features.py` is resumable by default (skips clips whose `.pt` already
  exists). Pass `--overwrite` to re-encode.
- Default `--num_workers 0`: features are already in RAM, so worker processes
  add no benefit and cost memory via fork duplication.
