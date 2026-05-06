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
import wandb
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


class CrocoMeanPoolIDM(InverseDynamicsModel):
    """CroCo patch tokens (B, N, 768) -> mean-pooled (B, 768) -> DINOv2-style MLP."""

    def __init__(self, feat_dim: int = 768, dropout: float = 0.1):
        super().__init__(feat_dim=feat_dim, dropout=dropout)

    def forward(self, z_i: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
        return super().forward(z_i.mean(dim=1), z_next.mean(dim=1))


class CrocoTransformerIDM(nn.Module):
    """Self-attention head over [CLS, f1, f2] for inverse dynamics.

    Inputs are precomputed CroCo patch tokens f1, f2 of shape (B, N, D). A
    learnable 2D pos-embed is added to each, plus a learnable image-identity
    embedding (one vector per image), then a learnable CLS is prepended. After
    `n_blocks` pre-norm transformer blocks, the CLS output predicts the 6D
    twist via two linear heads (v in R^3, omega in R^3).
    """

    def __init__(
        self,
        feat_dim: int = 768,
        n_tokens: int = 196,
        n_heads: int = 12,
        n_blocks: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.n_tokens = n_tokens

        self.cls       = nn.Parameter(torch.zeros(1, 1, feat_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, feat_dim))
        self.img_embed = nn.Parameter(torch.zeros(2, 1, feat_dim))
        nn.init.trunc_normal_(self.cls,       std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.img_embed, std=0.02)

        block = nn.TransformerEncoderLayer(
            d_model=feat_dim,
            nhead=n_heads,
            dim_feedforward=int(feat_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(block, num_layers=n_blocks)

        self.norm       = nn.LayerNorm(feat_dim)
        self.v_head     = nn.Linear(feat_dim, 3)
        self.omega_head = nn.Linear(feat_dim, 3)

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
        B = z_i.shape[0]
        z_i    = z_i    + self.pos_embed + self.img_embed[0:1]
        z_next = z_next + self.pos_embed + self.img_embed[1:2]
        x = torch.cat([self.cls.expand(B, -1, -1), z_i, z_next], dim=1)
        x = self.blocks(x)
        cls_out = self.norm(x[:, 0])
        return torch.cat([self.v_head(cls_out), self.omega_head(cls_out)], dim=-1)


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

    Per-clip features are kept as mmap-backed tensors (one per clip), so the
    OS pages in only the rows __getitem__ touches. Resident memory stays
    proportional to the working set, not the full cache — the previous
    "concat everything into one tensor" path needed ~135 GB for the CroCo
    cache (196 x 768 per frame x ~450k frames).

        self._clip_features: list[Tensor]    — feats[c] is (n_frames_c, ...)
        self._actions:       (Npairs, 6)     float32  — all valid-pair actions
        self._pair_idx:      (Npairs, 3)     int64    — (clip, row, row_next)

    Each __getitem__ returns (z_i, z_next, action):
        z_i, z_next: feature_shape  float32
        action:      (6,)            float32  — se3_log of E_{i+1} @ E_i^{-1}

    Pose timestamps and feature timestamps are matched by value, not
    position: a pose row is dropped if its timestamp isn't in the cache
    (e.g. the JPEG failed to download), so the two streams can drift.
    """

    def __init__(
        self,
        root: str | Path,
        features_root: str | Path,
        split: str = "train",
        max_abs_action: float = 0.5,
    ):
        self.root = Path(root)
        self.features_root = Path(features_root)
        self.split = split

        clip_dir = self.root / split
        clip_paths = sorted(clip_dir.glob("*.txt"))
        if not clip_paths:
            raise FileNotFoundError(f"No .txt files found in {clip_dir}")

        self._clip_features: list[torch.Tensor] = []
        actions_per_clip: list[torch.Tensor] = []
        pair_idx_chunks:  list[torch.Tensor] = []

        # se3_log is pathological near theta=pi — pairs can produce either
        # non-finite actions or finite-but-huge outliers (adjacent-frame
        # rotations > ~20deg are pose-track failures, not real motion). Both
        # poison action_stats and blow up MSE via Adam's second moment.
        n_dropped_nonfinite = 0
        n_dropped_outlier   = 0

        for clip_path in clip_paths:
            feat_path = self.features_root / split / f"{clip_path.stem}.pt"
            if not feat_path.exists():
                continue

            clip = parse_clip(clip_path)
            poses = torch.from_numpy(clip["P"])              # (N, 3, 4)
            clip_actions = se3_log(relative_pose(poses))     # (N-1, 6)
            clip_finite  = torch.isfinite(clip_actions).all(dim=-1)       # (N-1,)
            clip_inlier  = (clip_actions.abs() <= max_abs_action).all(-1) # (N-1,)
            pose_timestamps = [int(t) for t in clip["timestamps"]]

            # mmap=True keeps `feats` storage backed by the on-disk file —
            # no full read into RAM at init. Requires PyTorch >= 2.1.
            cached = torch.load(feat_path, map_location="cpu",
                                weights_only=True, mmap=True)
            feats = cached["features"]
            ts_to_row = {int(t): row for row, t in enumerate(cached["timestamps"].tolist())}

            clip_idx = len(self._clip_features)
            valid_pairs:   list[tuple[int, int, int]] = []
            valid_actions: list[torch.Tensor]         = []
            for i in range(len(pose_timestamps) - 1):
                row_i    = ts_to_row.get(pose_timestamps[i])
                row_next = ts_to_row.get(pose_timestamps[i + 1])
                if row_i is None or row_next is None:
                    continue
                if not clip_finite[i]:
                    n_dropped_nonfinite += 1
                    continue
                if not clip_inlier[i]:
                    n_dropped_outlier += 1
                    continue
                valid_pairs.append((clip_idx, row_i, row_next))
                valid_actions.append(clip_actions[i])

            if valid_pairs:
                pair_idx_chunks.append(torch.tensor(valid_pairs, dtype=torch.long))
                actions_per_clip.append(torch.stack(valid_actions))
                self._clip_features.append(feats)

        if not pair_idx_chunks:
            raise RuntimeError(
                f"No valid feature pairs found. Run cache_features.py to populate "
                f"{self.features_root}/{split}/."
            )

        self._actions  = torch.cat(actions_per_clip, dim=0)   # (Npairs, 6)
        self._pair_idx = torch.cat(pair_idx_chunks,  dim=0)   # (Npairs, 3)

        if n_dropped_nonfinite or n_dropped_outlier:
            print(f"[{split}] dropped {n_dropped_nonfinite} non-finite + "
                  f"{n_dropped_outlier} outlier pairs (|a| > {max_abs_action}).")

    def __len__(self) -> int:
        return self._pair_idx.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ci, row_i, row_next = self._pair_idx[idx].tolist()
        feats = self._clip_features[ci]
        # Clone the slice so the returned tensor doesn't keep the mmap page
        # pinned beyond the batch — DataLoader workers + pin_memory expect
        # plain owned CPU tensors.
        return feats[row_i].clone(), feats[row_next].clone(), self._actions[idx]

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
    total = 0.0
    n = 0
    for z_i, z_next, action in loader:
        z_i    = z_i.to(device)
        z_next = z_next.to(device)
        action = action.to(device)
        pred   = model(z_i, z_next)
        target = model.normalize_action(action)
        total += F.smooth_l1_loss(pred, target, reduction="sum").item()
        n += z_i.size(0)
    return total / max(n * 6, 1)


@torch.no_grad()
def mean_baseline_loss(model: InverseDynamicsModel, loader: DataLoader, device) -> float:
    """Smooth-L1 loss of the constant predictor (train action mean) on the loader.

    In normalized space this is `smooth_l1_loss(0, normalize_action(action))`.
    """
    total = 0.0
    n = 0
    for _, _, action in loader:
        action = action.to(device)
        target = model.normalize_action(action)
        pred   = torch.zeros_like(target)
        total += F.smooth_l1_loss(pred, target, reduction="sum").item()
        n += action.size(0)
    return total / max(n * 6, 1)


def train(
    model: InverseDynamicsModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 100,
    lr: float = 1e-4,
    warmup_steps: int = 1000,
    grad_clip: float = 1.0,
    device: str | torch.device = "cuda",
    log_every: int = 1000,
    ckpt_dir: str | Path | None = "checkpoints/inverse_dynamics",
    ckpt_every: int = 10,
    use_wandb: bool = False,
) -> None:
    device = torch.device(device)
    model = model.to(device)

    if ckpt_dir is not None:
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    val_mean_baseline = mean_baseline_loss(model, val_loader, device)
    print(f"val mean-baseline loss: {val_mean_baseline:.4f}")
    if use_wandb:
        wandb.run.summary["val/mean_baseline"] = val_mean_baseline

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def lr_lambda(step: int) -> float:
        return min(1.0, (step + 1) / max(warmup_steps, 1))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    best_val = float("inf")
    step = 0
    for epoch in range(num_epochs):
        model.train()
        run_loss = 0.0
        n = 0
        for z_i, z_next, action in train_loader:
            # non_blocking=True on MPS with pinned memory silently returns
            # uninitialized memory for the first batch — keep blocking transfers.
            z_i    = z_i.to(device)
            z_next = z_next.to(device)
            action = action.to(device)

            pred   = model(z_i, z_next)
            target = model.normalize_action(action)
            loss   = F.smooth_l1_loss(pred, target)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            sched.step()

            run_loss += loss.item() * z_i.size(0)
            n += z_i.size(0)
            step += 1
            if use_wandb:
                wandb.log({
                    "train/loss_step": loss.item(),
                    "train/lr": sched.get_last_lr()[0],
                    "epoch": epoch,
                }, step=step)
            if step % log_every == 0:
                print(f"epoch {epoch}  step {step}  "
                      f"lr {sched.get_last_lr()[0]:.2e}  "
                      f"train_loss {loss.item():.4f}")

        val = evaluate(model, val_loader, device)
        train_epoch_loss = run_loss / max(n, 1)
        print(f"[epoch {epoch}] train_loss {train_epoch_loss:.4f}  val_loss {val:.4f}")
        if use_wandb:
            wandb.log({
                "train/loss_epoch": train_epoch_loss,
                "val/loss": val,
                "val/mean_baseline": val_mean_baseline,
                "epoch": epoch,
            }, step=step)

        if ckpt_dir is not None:
            payload = {"model": model.state_dict(), "epoch": epoch, "val_loss": val}

            if val < best_val - 1e-5:
                best_val = val
                torch.save({**payload, "val_loss": best_val}, ckpt_dir / "best.pt")
                print(f"  -> new best val_loss {best_val:.4f} (saved best.pt)")

            # Save every Nth epoch and the final epoch.
            if (epoch + 1) % ckpt_every == 0 or epoch == num_epochs - 1:
                ckpt_path = ckpt_dir / f"epoch_{epoch + 1:03d}.pt"
                torch.save(payload, ckpt_path)
                print(f"  -> saved {ckpt_path.name}")


# --------------------------------------------------------------------------- #
# Entry
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root",          default="RealEstate10K")
    parser.add_argument("--features_root", default="features")
    parser.add_argument("--head", choices=["dino_mlp", "croco_meanpool", "croco_transformer"],
                        default="dino_mlp",
                        help="dino_mlp: 384-D CLS + MLP. "
                             "croco_meanpool: mean-pool 196x768 patches + same MLP. "
                             "croco_transformer: self-attn over [CLS, f1, f2].")
    parser.add_argument("--batch_size",    type=int, default=64)
    parser.add_argument("--num_workers",   type=int, default=0)
    parser.add_argument("--num_epochs",    type=int, default=100)
    parser.add_argument("--ckpt_dir",      default="checkpoints/inverse_dynamics",
                        help="directory for best.pt and periodic epoch_NNN.pt files.")
    parser.add_argument("--ckpt_every",    type=int, default=10,
                        help="save a snapshot every N epochs (in addition to best.pt).")
    parser.add_argument("--device", default=None,
                        help="torch device. Defaults to cuda > mps > cpu.")
    parser.add_argument("--wandb",         action="store_true",
                        help="log train/val loss to Weights & Biases.")
    parser.add_argument("--wandb_project", default="action-conditioned-diffusion")
    parser.add_argument("--wandb_run",     default=None)
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

    if args.head == "dino_mlp":
        model = InverseDynamicsModel()
    elif args.head == "croco_meanpool":
        model = CrocoMeanPoolIDM()
    elif args.head == "croco_transformer":
        sample_feat = train_ds._clip_features[0][0]
        model = CrocoTransformerIDM(feat_dim=sample_feat.shape[-1],
                                    n_tokens=sample_feat.shape[-2])
    else:
        raise ValueError(args.head)
    mean, std = compute_action_stats(train_ds)
    model.set_action_stats(mean, std)
    print(f"action mean: {mean.tolist()}")
    print(f"action std:  {std.tolist()}")

    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run,
            settings=wandb.Settings(init_timeout=300),
            config={
                "head":        args.head,
                "batch_size":  args.batch_size,
                "num_epochs":  args.num_epochs,
                "device":      args.device,
                "train_pairs": len(train_ds),
                "val_pairs":   len(val_ds),
                "action_mean": mean.tolist(),
                "action_std":  std.tolist(),
            },
        )

    try:
        train(model, train_loader, val_loader,
              num_epochs=args.num_epochs, device=args.device,
              ckpt_dir=args.ckpt_dir, ckpt_every=args.ckpt_every,
              use_wandb=args.wandb)
    finally:
        if args.wandb:
            wandb.finish()
