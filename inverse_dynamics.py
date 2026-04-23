"""
Inverse dynamics head: predict the 6D SE(3) twist a=(v, omega) between two
consecutive frames from their precomputed DINOv2 CLS features.

  - Input:  z_i, z_{i+1} in R^384 (from cache_features.py)
  - Fusion: concat([z_i, z_{i+1}, z_{i+1} - z_i])  ->  R^1152
  - MLP:    1152 -> 512 -> 512 -> 6
  - Targets are normalized via (mean, std) buffers on the model

Run cache_features.py first to populate {features_root}/{split}/{clip_id}.pt.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from dataset import parse_clip, relative_pose, se3_log


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

class InverseDynamicsModel(nn.Module):
    """MLP head mapping a pair of DINOv2 CLS features to a 6D action.

    The encoder that produces those features lives in cache_features.py and
    is run once offline — see PREPROCESS + load_encoder there for inference.
    """

    def __init__(self, feat_dim: int = 384, dropout: float = 0.1):
        super().__init__()
        self.feat_dim = feat_dim

        in_dim = 3 * feat_dim
        self.head = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 6),
        )

        self.register_buffer("action_mean", torch.zeros(6))
        self.register_buffer("action_std",  torch.ones(6))

    def set_action_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.action_mean.copy_(mean.to(self.action_mean))
        self.action_std.copy_(std.to(self.action_std).clamp_min(1e-8))

    def normalize_action(self, a: torch.Tensor) -> torch.Tensor:
        return (a - self.action_mean) / self.action_std

    def denormalize_action(self, a_norm: torch.Tensor) -> torch.Tensor:
        return a_norm * self.action_std + self.action_mean

    def forward(self, z_i: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
        """Predict the normalized 6D action between two frames.

        Args:
            z_i, z_next: (B, feat_dim) float32 — DINOv2 CLS tokens for the
                two frames, in order.
        Returns:
            (B, 6) float32 — normalized (v, omega). Apply
            ``denormalize_action`` to recover physical units.
        """
        h = torch.cat([z_i, z_next, z_next - z_i], dim=-1)
        return self.head(h)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

class CachedPairDataset(Dataset):
    """Consecutive-frame pairs served from precomputed DINOv2 CLS features.

    Layout expectations (match cache_features.py + dataset.py):
        root          = RealEstate10K/
        features_root = features/
        Feature .pt   at features/{split}/{clip_id}.pt  — dict with
                         "timestamps" (LongTensor, N) and "features" (N, 384)
        Clip .txt     at RealEstate10K/{split}/{clip_id}.txt

    Internally everything is concatenated across clips into three flat
    tensors so __getitem__ is two index lookups and a no-copy slice:
        self._features:  (Nframes, 384)  float32  — all clips' CLS tokens
        self._actions:   (Npairs,  6)    float32  — all valid-pair actions
        self._pair_idx:  (Npairs,  2)    int64    — (i, next) into _features

    Each __getitem__ returns (z_i, z_next, action):
        z_i, z_next: (384,) float32
        action:      (6,)   float32  — se3_log of E_{i+1} @ E_i^{-1}

    Pose timestamps and feature timestamps are matched by value, not
    position: a pose row is dropped if its timestamp isn't in the cache
    (e.g. the JPEG failed to download), so the two streams can drift.
    """

    def __init__(
        self,
        root: str | Path,
        features_root: str | Path,
        split: str = "train",
    ):
        self.root = Path(root)
        self.features_root = Path(features_root)
        self.split = split

        clip_dir = self.root / split
        clip_paths = sorted(clip_dir.glob("*.txt"))
        if not clip_paths:
            raise FileNotFoundError(f"No .txt files found in {clip_dir}")

        feats_per_clip:   list[torch.Tensor] = []
        actions_per_clip: list[torch.Tensor] = []
        pair_idx_chunks:  list[torch.Tensor] = []

        # Running row count: lets us rebase per-clip indices into the
        # concatenated self._features tensor we build at the end.
        feature_offset = 0
        for clip_path in clip_paths:
            feat_path = self.features_root / split / f"{clip_path.stem}.pt"
            if not feat_path.exists():
                continue

            clip = parse_clip(clip_path)
            poses = torch.from_numpy(clip["P"])              # (N, 3, 4)
            clip_actions = se3_log(relative_pose(poses))     # (N-1, 6)
            pose_timestamps = [int(t) for t in clip["timestamps"]]

            cached = torch.load(feat_path, map_location="cpu", weights_only=True)
            feats = cached["features"]                       # already float32
            ts_to_row = {int(t): row for row, t in enumerate(cached["timestamps"].tolist())}

            valid_pairs:   list[tuple[int, int]] = []
            valid_actions: list[torch.Tensor]    = []
            for i in range(len(pose_timestamps) - 1):
                row_i    = ts_to_row.get(pose_timestamps[i])
                row_next = ts_to_row.get(pose_timestamps[i + 1])
                if row_i is None or row_next is None:
                    continue
                valid_pairs.append((feature_offset + row_i, feature_offset + row_next))
                valid_actions.append(clip_actions[i])

            if valid_pairs:
                pair_idx_chunks.append(torch.tensor(valid_pairs, dtype=torch.long))
                actions_per_clip.append(torch.stack(valid_actions))

            feats_per_clip.append(feats)
            feature_offset += feats.shape[0]

        if not pair_idx_chunks:
            raise RuntimeError(
                f"No valid feature pairs found. Run cache_features.py to populate "
                f"{self.features_root}/{split}/."
            )

        self._features = torch.cat(feats_per_clip,   dim=0)   # (Nframes, 384)
        self._actions  = torch.cat(actions_per_clip, dim=0)   # (Npairs,  6)
        self._pair_idx = torch.cat(pair_idx_chunks,  dim=0)   # (Npairs,  2)

    def __len__(self) -> int:
        return self._pair_idx.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row_i, row_next = self._pair_idx[idx].tolist()
        return self._features[row_i], self._features[row_next], self._actions[idx]

    def all_actions(self) -> torch.Tensor:
        """(Npairs, 6) tensor of all action targets — fast path for stats."""
        return self._actions


def collate_pairs(batch):
    z_i    = torch.stack([b[0] for b in batch])
    z_next = torch.stack([b[1] for b in batch])
    action = torch.stack([b[2] for b in batch])
    return z_i, z_next, action


# --------------------------------------------------------------------------- #
# Action statistics (one pass — no I/O when source is a CachedPairDataset)
# --------------------------------------------------------------------------- #

@torch.no_grad()
def compute_action_stats(
    source: DataLoader | Dataset,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-dim mean/std of the action target.

    Fast path: if the source exposes ``all_actions()`` (as
    CachedPairDataset does), pull the stacked target tensor directly and
    skip the iteration. Otherwise stream the source one batch at a time
    with a numerically stable two-pass-equivalent (sum / sum-of-squares).
    """
    if hasattr(source, "all_actions"):
        a = source.all_actions().to(torch.float64)
        return a.mean(dim=0).float(), a.std(dim=0).clamp_min(1e-8).float()

    s  = torch.zeros(6, dtype=torch.float64)
    s2 = torch.zeros(6, dtype=torch.float64)
    n  = 0
    for _, _, a in source:
        a = a.to(torch.float64)
        s  += a.sum(dim=0)
        s2 += (a * a).sum(dim=0)
        n  += a.shape[0]
    mean = s / max(n, 1)
    var  = s2 / max(n, 1) - mean**2
    std  = var.clamp_min(1e-12).sqrt()
    return mean.float(), std.float()


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

@torch.no_grad()
def evaluate(model: InverseDynamicsModel, loader: DataLoader, device) -> float:
    model.eval()
    total_sq = 0.0
    n = 0
    for z_i, z_next, action in loader:
        z_i    = z_i.to(device, non_blocking=True)
        z_next = z_next.to(device, non_blocking=True)
        action = action.to(device, non_blocking=True)
        pred   = model(z_i, z_next)
        target = model.normalize_action(action)
        total_sq += F.mse_loss(pred, target, reduction="sum").item()
        n += z_i.size(0)
    return total_sq / max(n * 6, 1)


def train(
    model: InverseDynamicsModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    lr: float = 1e-4,
    warmup_steps: int = 1000,
    grad_clip: float = 1.0,
    device: str | torch.device = "cuda",
    log_every: int = 100,
    ckpt_path: str | Path | None = "inverse_dynamics.pt",
    early_stop_patience: int = 5,
) -> None:
    device = torch.device(device)
    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def lr_lambda(step: int) -> float:
        return min(1.0, (step + 1) / max(warmup_steps, 1))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    best_val = float("inf")
    patience = 0
    step = 0
    for epoch in range(num_epochs):
        model.train()
        run_sq = 0.0
        n = 0
        for z_i, z_next, action in train_loader:
            z_i    = z_i.to(device, non_blocking=True)
            z_next = z_next.to(device, non_blocking=True)
            action = action.to(device, non_blocking=True)

            pred   = model(z_i, z_next)
            target = model.normalize_action(action)
            loss   = F.mse_loss(pred, target)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            sched.step()

            run_sq += loss.item() * z_i.size(0)
            n += z_i.size(0)
            step += 1
            if step % log_every == 0:
                print(f"epoch {epoch}  step {step}  "
                      f"lr {sched.get_last_lr()[0]:.2e}  "
                      f"train_mse {loss.item():.4f}")

        val = evaluate(model, val_loader, device)
        print(f"[epoch {epoch}] train_mse {run_sq/max(n,1):.4f}  val_mse {val:.4f}")

        if val < best_val - 1e-5:
            best_val = val
            patience = 0
            if ckpt_path:
                torch.save(
                    {"model": model.state_dict(),
                     "epoch": epoch,
                     "val_mse": best_val},
                    ckpt_path,
                )
        else:
            patience += 1
            if patience >= early_stop_patience:
                print(f"early stop at epoch {epoch} (best val_mse {best_val:.4f})")
                break


# --------------------------------------------------------------------------- #
# Entry
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root",          default="RealEstate10K")
    parser.add_argument("--features_root", default="features")
    parser.add_argument("--batch_size",    type=int, default=64)
    parser.add_argument("--num_workers",   type=int, default=0)
    parser.add_argument("--num_epochs",    type=int, default=50)
    parser.add_argument("--device", default=None,
                        help="torch device. Defaults to cuda > mps > cpu.")
    args = parser.parse_args()

    if args.device is None:
        args.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

    train_ds = CachedPairDataset(args.root, args.features_root, split="train")
    val_ds   = CachedPairDataset(args.root, args.features_root, split="test")
    print(f"train pairs: {len(train_ds):,}   test pairs: {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_pairs, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_pairs, pin_memory=True,
    )

    model = InverseDynamicsModel()
    mean, std = compute_action_stats(train_ds)
    model.set_action_stats(mean, std)
    print(f"action mean: {mean.tolist()}")
    print(f"action std:  {std.tolist()}")

    train(model, train_loader, val_loader,
          num_epochs=args.num_epochs, device=args.device)
