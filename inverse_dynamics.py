"""
Inverse dynamics head: predict the 6D SE(3) twist a=(v, omega) covering a
fixed time interval (default: 0.25 s = 1/action_hz at action_hz=4) between
two frames from their precomputed image features.

  - Input:  z_i, z_{i+0.25s} in R^384 (from cache_features.py)
  - Fusion: concat([z_i, z_next, z_next - z_i])  ->  R^1152
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

from dataset import parse_clip, relative_pose, se3_log, snap_to_time_grid


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
    """Pairs of frames `1/action_hz` seconds apart, served from precomputed
    image features.

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

    Pairs are built on a uniform `1/action_hz`-second grid (anchored at the
    first jointly-valid timestamp), with each grid slot snapped to the
    nearest jointly-valid frame within half the inter-action interval.
    Clips shorter than `min_clip_seconds` are dropped to keep the train set
    aligned with the diffusion-side window length.

    Each __getitem__ returns (z_i, z_next, action):
        z_i, z_next: feature_shape  float32
        action:      (6,)            float32  — se3_log of E_{i+1} @ E_i^{-1},
                                                covering 1/action_hz seconds.
    """

    def __init__(
        self,
        root: str | Path,
        features_root: str | Path,
        split: str = "train",
        action_hz: float = 4.0,
        min_clip_seconds: float = 2.0,
    ):
        self.root = Path(root)
        self.features_root = Path(features_root)
        self.split = split
        self.action_hz = action_hz
        self.min_clip_seconds = min_clip_seconds

        step_us = int(round(1e6 / action_hz))
        max_offset_us = step_us // 2
        min_grid_len = int(round(min_clip_seconds * action_hz)) + 1

        clip_dir = self.root / split
        clip_paths = sorted(clip_dir.glob("*.txt"))
        if not clip_paths:
            raise FileNotFoundError(f"No .txt files found in {clip_dir}")

        self._clip_features: list[torch.Tensor] = []
        actions_per_clip: list[torch.Tensor] = []
        pair_idx_chunks:  list[torch.Tensor] = []

        # se3_log is pathological near theta=pi — pairs can produce non-finite
        # actions (drop those). The previous |a| <= 0.5 outlier filter is
        # removed: 1/action_hz-second deltas are larger than inter-frame
        # deltas and that box would clip real motion.
        n_dropped_short = 0
        n_dropped_nonfinite = 0

        for clip_path in clip_paths:
            feat_path = self.features_root / split / f"{clip_path.stem}.pt"
            if not feat_path.exists():
                continue

            clip = parse_clip(clip_path)
            poses = torch.from_numpy(clip["P"])              # (N, 3, 4)
            pose_ts = [int(t) for t in clip["timestamps"]]

            # mmap=True keeps `feats` storage backed by the on-disk file —
            # no full read into RAM at init. Requires PyTorch >= 2.1.
            cached = torch.load(feat_path, map_location="cpu",
                                weights_only=True, mmap=True)
            feats = cached["features"]
            feat_ts = cached["timestamps"].tolist()

            grid = snap_to_time_grid(pose_ts, feat_ts, step_us, max_offset_us)
            if len(grid) < min_grid_len:
                n_dropped_short += 1
                continue

            clip_idx = len(self._clip_features)
            valid_pairs:   list[tuple[int, int, int]] = []
            valid_actions: list[torch.Tensor]         = []

            # Walk runs of contiguous non-None grid slots and compute
            # consecutive deltas in one batched call per run.
            i = 0
            while i < len(grid):
                if grid[i] is None:
                    i += 1
                    continue
                j = i
                while j < len(grid) and grid[j] is not None:
                    j += 1
                if j - i >= 2:
                    pose_indices = [grid[k][0] for k in range(i, j)]
                    feat_rows    = [grid[k][1] for k in range(i, j)]
                    rel = relative_pose(poses[pose_indices])      # (M-1, 3, 4)
                    actions = se3_log(rel)                        # (M-1, 6)
                    finite = torch.isfinite(actions).all(dim=-1)  # (M-1,)
                    for m in range(j - i - 1):
                        if not finite[m]:
                            n_dropped_nonfinite += 1
                            continue
                        valid_pairs.append((clip_idx, feat_rows[m], feat_rows[m + 1]))
                        valid_actions.append(actions[m])
                i = j

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

        print(f"[{split}] {len(self._clip_features)} clips, "
              f"{self._pair_idx.shape[0]} pairs ({1.0 / action_hz:.3f}s gap; "
              f"dropped {n_dropped_short} short clips (<{min_clip_seconds}s), "
              f"{n_dropped_nonfinite} non-finite pairs).")

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
    ckpt_every: int = 1,
    use_wandb: bool = False,
    init_from: str | Path | None = None,
) -> None:
    device = torch.device(device)
    model = model.to(device)

    start_epoch = 0
    best_val = float("inf")
    if init_from is not None:
        # Warm-start: load weights (and buffers — action stats are buffers) from
        # a prior best.pt. Optimizer/scheduler state is not stored, so callers
        # who want to skip warmup should pass warmup_steps=0.
        ckpt = torch.load(init_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_val = float(ckpt.get("val_loss", float("inf")))
        print(f"resumed from {init_from}: start_epoch={start_epoch}  "
              f"prev_val_loss={best_val:.4f}")

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

    # Align step with cumulative gradient-step count so wandb fork_from / resume
    # picks up exactly where the prior run left off. On a resumed wandb run,
    # also clamp past wandb.run.step (the prior run's max step) so subsequent
    # logs are strictly increasing.
    step = start_epoch * len(train_loader)
    if use_wandb and getattr(wandb.run, "resumed", False):
        step = max(step, wandb.run.step)
    for epoch in range(start_epoch, start_epoch + num_epochs):
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
            payload = {"model": model.state_dict(), "epoch": epoch,
                       "step": step, "val_loss": val}

            if val < best_val - 1e-5:
                best_val = val
                torch.save({**payload, "val_loss": best_val}, ckpt_dir / "best.pt")
                print(f"  -> new best val_loss {best_val:.4f} (saved best.pt)")

            # Save every Nth epoch and the final epoch.
            if (epoch + 1) % ckpt_every == 0 or epoch == start_epoch + num_epochs - 1:
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
    parser.add_argument("--action_hz",     type=float, default=4.0,
                        help="Action rate in Hz. Pairs span 1/action_hz seconds.")
    parser.add_argument("--min_clip_seconds", type=float, default=2.0,
                        help="Discard clips shorter than this (matches the "
                             "diffusion window length).")
    parser.add_argument("--batch_size",    type=int, default=64)
    parser.add_argument("--num_workers",   type=int, default=0)
    parser.add_argument("--num_epochs",    type=int, default=100)
    parser.add_argument("--ckpt_dir",      default="checkpoints/inverse_dynamics",
                        help="directory for best.pt and periodic epoch_NNN.pt files.")
    parser.add_argument("--ckpt_every",    type=int, default=1,
                        help="save a snapshot every N epochs (in addition to best.pt).")
    parser.add_argument("--init_from",     default=None,
                        help="path to a prior checkpoint (e.g. best.pt) to warm-start "
                             "weights from. Optimizer state is not restored.")
    parser.add_argument("--warmup_steps",  type=int, default=1000,
                        help="linear LR warmup steps. Pass 0 when --init_from is set "
                             "to avoid re-warming up an already-trained model.")
    parser.add_argument("--device", default=None,
                        help="torch device. Defaults to cuda > mps > cpu.")
    parser.add_argument("--wandb",         action="store_true",
                        help="log train/val loss to Weights & Biases.")
    parser.add_argument("--wandb_project", default="action-conditioned-diffusion")
    parser.add_argument("--wandb_run",     default=None)
    parser.add_argument("--wandb_run_id",  default=None,
                        help="W&B run id to fork from. Combined with the step "
                             "recorded in --init_from (or --wandb_fork_step), creates "
                             "a new forked run that branches off cleanly at that step.")
    parser.add_argument("--wandb_fork_step", type=int, default=None,
                        help="explicit step to fork from. Required when --init_from "
                             "points at a checkpoint that does not contain 'step' "
                             "(e.g. best.pt files saved before this field was added).")
    args = parser.parse_args()

    if args.device is None:
        args.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

    train_ds = CachedPairDataset(args.root, args.features_root, split="train",
                                 action_hz=args.action_hz,
                                 min_clip_seconds=args.min_clip_seconds)
    val_ds   = CachedPairDataset(args.root, args.features_root, split="test",
                                 action_hz=args.action_hz,
                                 min_clip_seconds=args.min_clip_seconds)
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

    # `fork_from` is gated behind a private preview at W&B, so we use the
    # public `resume="allow"` path: the new process re-attaches to the existing
    # run id and appends to it. Step collisions are avoided by clamping the
    # local step counter to wandb.run.step in train() below. --wandb_fork_step
    # is accepted for compatibility but unused on this path.
    if args.wandb:
        init_kwargs = dict(
            project=args.wandb_project,
            name=args.wandb_run if args.wandb_run_id is None else None,
            settings=wandb.Settings(init_timeout=300),
            config={
                "head":             args.head,
                "action_hz":        args.action_hz,
                "min_clip_seconds": args.min_clip_seconds,
                "batch_size":  args.batch_size,
                "num_epochs":  args.num_epochs,
                "device":      args.device,
                "train_pairs": len(train_ds),
                "val_pairs":   len(val_ds),
                "action_mean": mean.tolist(),
                "action_std":  std.tolist(),
            },
        )
        if args.wandb_run_id is not None:
            init_kwargs["id"] = args.wandb_run_id
            init_kwargs["resume"] = "allow"
            print(f"resuming W&B run {args.wandb_run_id} (resume=allow)")
        wandb.init(**init_kwargs)

    try:
        train(model, train_loader, val_loader,
              num_epochs=args.num_epochs, device=args.device,
              warmup_steps=args.warmup_steps,
              ckpt_dir=args.ckpt_dir, ckpt_every=args.ckpt_every,
              init_from=args.init_from,
              use_wandb=args.wandb)
    finally:
        if args.wandb:
            wandb.finish()
