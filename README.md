# Action-Conditioned Diffusion

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
