"""
Inverse dynamics model: predict the 6D SE(3) twist a=(v, omega) between two
consecutive frames directly from raw RGB images.

  - Input:  two RGB frames, each resized to (img_size * img_size)
  - Encoder (per frame, weights shared):
        3*3 conv stem  ->  3 * ResBlock(3*3 conv + skip)  ->  spatial mean-pool
  - Fusion: concat([flat_i, flat_next])  ->  (B, 2*hidden)
  - MLP:    2*hidden -> 128 -> 6

No DINOv2, no feature cache. Images are loaded on-the-fly from the JPEG
files already present in the RealEstate10K image directory.

Architecture follows UniPi Appendix A.2:
    "3*3 conv layer, 3 layers of 3*3 convolutions with residual connection,
     mean-pooling layer across all pixel locations, MLP (128, out_dim)."
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.checkpoint import checkpoint


from tqdm import tqdm

try:
    import wandb
except ModuleNotFoundError:  # pragma: no cover
    wandb = None

from dataset import parse_clip, relative_pose, se3_log


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #

class ResBlock(nn.Module):
    """Two 3*3 convs with a residual skip, BN + ReLU."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + checkpoint(self.block, x, use_reentrant=False))



class FrameEncoder(nn.Module):
    """RGB frame -> flat vector via conv stem + 3 residual blocks + mean-pool.

    Args:
        hidden: channel width throughout (stem output and residual blocks).
    """

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W)  ->  (B, hidden)."""
        return self.body(self.stem(x)).mean(dim=(-2, -1))


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

class InverseDynamicsModel(nn.Module):
    """UniPi-style convolutional inverse dynamics model over raw RGB pairs.

    Args:
        hidden:  conv encoder channel width.
        out_dim: action dimensionality (6 for SE3 log-map).
        dropout: dropout in the MLP head.
    """

    def __init__(
        self,
        hidden:  int   = 64,
        out_dim: int   = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = FrameEncoder(hidden=hidden)

        self.head = nn.Sequential(
            nn.Linear(2 * hidden, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim),
        )

        self.register_buffer("action_mean", torch.zeros(out_dim))
        self.register_buffer("action_std",  torch.ones(out_dim))

    def set_action_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.action_mean.copy_(mean.to(self.action_mean))
        self.action_std.copy_(std.to(self.action_std).clamp_min(1e-8))

    def normalize_action(self, a: torch.Tensor) -> torch.Tensor:
        return (a - self.action_mean) / self.action_std

    def denormalize_action(self, a_norm: torch.Tensor) -> torch.Tensor:
        return a_norm * self.action_std + self.action_mean

    def forward(
        self,
        frame_i:    torch.Tensor,
        frame_next: torch.Tensor,
    ) -> torch.Tensor:
        """(B,3,H,W), (B,3,H,W)  ->  (B, out_dim) normalised action."""
        z_i    = self.encoder(frame_i)
        z_next = self.encoder(frame_next)
        return self.head(torch.cat([z_i, z_next], dim=-1))


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

class RGBPairDataset(Dataset):
    """Consecutive RGB frame pairs loaded directly from disk.

    Expected layout:
        RealEstate10K/
            {split}/
                {clip_id}.txt              <- pose file (parsed by dataset.py)
        frames/
            {split}/
                {clip_id}/
                    {timestamp}.jpg    <- one JPEG per frame

    Each __getitem__ loads two JPEGs, applies the transform, and returns
    (frame_i, frame_next, action) where action is the SE3 log-map of the
    relative pose between the two frames.

    Args:
        root:            path to RealEstate10K/
        frames_root:     path to the extracted frames root
        split:           "train" or "test"
        img_size:        spatial resolution fed to the model (default 224)
        max_abs_action:  outlier threshold — pairs with any action component
                         exceeding this are dropped (pose-track failures)
    """

    # Also add support for cached encodings, see CachedEncodingPairDataset below

    MEAN = (0.485, 0.456, 0.406)
    STD  = (0.229, 0.224, 0.225)

    def __init__(
        self,
        root:           str | Path,
        frames_root:    str | Path = "frames",
        split:          str   = "train",
        img_size:       int   = 224,
        max_abs_action: float = 0.5,
    ):
        self.root        = Path(root)
        self.frames_root = Path(frames_root)
        self.split       = split

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD),
        ])

        clip_dir   = self.root / split
        frames_dir = self.frames_root / split
        clip_paths = sorted(clip_dir.glob("*.txt"))
        if not clip_paths:
            raise FileNotFoundError(f"No .txt files found in {clip_dir}")

        # Each entry: (path_i, path_next, action_tensor)
        self._pairs: list[tuple[Path, Path, torch.Tensor]] = []

        n_dropped_nonfinite = 0
        n_dropped_missing   = 0
        n_dropped_outlier   = 0

        for clip_path in clip_paths:
            clip_frames_dir = frames_dir / clip_path.stem
            if not clip_frames_dir.exists():
                continue

            clip         = parse_clip(clip_path)
            poses        = torch.from_numpy(clip["P"])           # (N, 3, 4)
            clip_actions = se3_log(relative_pose(poses))         # (N-1, 6)
            clip_finite  = torch.isfinite(clip_actions).all(-1)  # (N-1,)
            clip_inlier  = (clip_actions.abs() <= max_abs_action).all(-1)
            timestamps   = [int(t) for t in clip["timestamps"]]

            for i in range(len(timestamps) - 1):
                if not clip_finite[i]:
                    n_dropped_nonfinite += 1
                    continue
                if not clip_inlier[i]:
                    n_dropped_outlier += 1
                    continue

                path_i    = clip_frames_dir / f"{timestamps[i]}.jpg"
                path_next = clip_frames_dir / f"{timestamps[i + 1]}.jpg"
                if not path_i.exists() or not path_next.exists():
                    n_dropped_missing += 1
                    continue

                self._pairs.append((path_i, path_next, clip_actions[i]))

        if not self._pairs:
            raise RuntimeError(
                f"No valid image pairs found under {frames_dir}. "
                "Ensure frames are stored as "
                "{frames_dir}/{clip_id}/{timestamp}.jpg"
            )

        if n_dropped_nonfinite or n_dropped_outlier or n_dropped_missing:
            print(
                f"[{split}] dropped  {n_dropped_nonfinite} non-finite"
                f"  | {n_dropped_outlier} outlier (|a|>{max_abs_action})"
                f"  | {n_dropped_missing} missing-image  pairs."
            )

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int):
        path_i, path_next, action = self._pairs[idx]
        try:
            # Explicitly close images after converting to avoid leaking file handles
            with Image.open(path_i) as img:
                frame_i = self.transform(img.convert("RGB"))
            with Image.open(path_next) as img:
                frame_next = self.transform(img.convert("RGB"))
        except Exception:
            # corrupt JPEG — return a random valid sample instead
            return self[torch.randint(len(self), (1,)).item()]
        return frame_i, frame_next, action

    def all_actions(self) -> torch.Tensor:
        """(Npairs, 6) — returns only pre-computed action tensors, no image I/O."""
        return torch.stack([a for _, _, a in self._pairs], dim=0)


# --------------------------------------------------------------------------- #
# Collation
# --------------------------------------------------------------------------- #

def collate_pairs(batch):
    fi     = torch.stack([b[0] for b in batch])
    fnext  = torch.stack([b[1] for b in batch])
    action = torch.stack([b[2] for b in batch])
    return fi, fnext, action


# --------------------------------------------------------------------------- #
# Action statistics
# --------------------------------------------------------------------------- #

@torch.no_grad()
def compute_action_stats(
    source: DataLoader | Dataset,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-dim mean / std of the action target.

    Fast path: if source exposes ``all_actions()`` pull the full tensor
    directly and skip iteration.  Otherwise stream one batch at a time.
    """
    if hasattr(source, "all_actions"):
        a = source.all_actions().to(torch.float64)
        return a.mean(dim=0).float(), a.std(dim=0).clamp_min(1e-8).float()

    s  = torch.zeros(6, dtype=torch.float64)
    s2 = torch.zeros(6, dtype=torch.float64)
    n  = 0
    for _, _, a in tqdm(source, desc="action_stats", leave=False):
        a   = a.to(torch.float64)
        s  += a.sum(dim=0)
        s2 += (a * a).sum(dim=0)
        n  += a.shape[0]
    mean = s / max(n, 1)
    var  = s2 / max(n, 1) - mean ** 2
    return mean.float(), var.clamp_min(1e-12).sqrt().float()


# --------------------------------------------------------------------------- #
# Checkpoint helpers
# --------------------------------------------------------------------------- #

def save_checkpoint(
    path:       str | Path,
    model:      InverseDynamicsModel,
    optimizer:  torch.optim.Optimizer,
    scheduler:  torch.optim.lr_scheduler._LRScheduler,
    epoch:      int,
    step:       int,
    best_val:   float,
) -> None:
    torch.save(
        {
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch":     epoch,
            "step":      step,
            "best_val":  best_val,
            # save normalisation stats so the checkpoint is self-contained
            "action_mean": model.action_mean.cpu(),
            "action_std":  model.action_std.cpu(),
        },
        path,
    )


def load_checkpoint(
    path:      str | Path,
    model:     InverseDynamicsModel,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
) -> tuple[int, int, float]:
    """Load checkpoint in-place.  Returns (start_epoch, step, best_val)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.set_action_stats(ckpt["action_mean"], ckpt["action_std"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch = ckpt.get("epoch", 0) + 1   # resume from *next* epoch
    step        = ckpt.get("step",  0)
    best_val    = ckpt.get("best_val", float("inf"))
    print(
        f"Resumed from {path}  "
        f"(epoch {ckpt['epoch']}, step {step}, best_val {best_val:.4f})"
    )
    return start_epoch, step, best_val


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

@torch.no_grad()
def evaluate(model: InverseDynamicsModel, loader: DataLoader, device) -> float:
    model.eval()
    total = 0.0
    n = 0
    for fi, fnext, action in tqdm(loader, desc="eval", leave=False):
        fi     = fi.to(device)
        fnext  = fnext.to(device)
        action = action.to(device)
        pred   = model(fi, fnext)
        target = model.normalize_action(action)
        total += F.mse_loss(pred, target, reduction="sum").item()
        n     += fi.size(0)
    return total / max(n * model.action_mean.numel(), 1)


def train(
    model:               InverseDynamicsModel,
    train_loader:        DataLoader,
    val_loader:          DataLoader,
    num_epochs:          int              = 50,
    lr:                  float            = 1e-4,
    warmup_steps:        int              = 10_000,
    grad_clip:           float            = 1.0,
    device:              str | torch.device = "cuda",
    log_every:           int              = 10,
    # ---- checkpoint args -------------------------------------------------- #
    ckpt_dir:            str | Path       = "checkpoints",
    ckpt_every_n_epochs: int              = 5,   # periodic save interval
    resume_from:         str | Path | None = None,  # path to resume checkpoint
    # ----------------------------------------------------------------------- #
    early_stop_patience: int              = 10,
    mixed_precision:     bool             = True,  # use autocast + GradScaler
    wandb_project:       str              = None,
    wandb_run:           str | None       = None,
) -> None:
    """Training loop matching UniPi A.2:
        Adam, lr=1e-4, grad-norm clip=1, linear warmup over 10k steps, MSE loss.
        Optionally uses mixed precision (autocast + GradScaler) for faster training.

    Checkpoints
    -----------
    - ``ckpt_dir/best.pt``       — saved whenever val loss improves.
    - ``ckpt_dir/epoch_{N}.pt``  — saved every ``ckpt_every_n_epochs`` epochs.
    - ``ckpt_dir/last.pt``       — saved at the end of every epoch.
    Pass ``resume_from=<path>`` to continue from any of the above.
    """
    print(f"training on device: {device}")
    device   = torch.device(device)
    model    = model.to(device)
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    scaler = GradScaler('cuda', enabled=mixed_precision)
    print(f"mixed precision: {mixed_precision}")

    if wandb is not None and wandb_project:
        wandb.init(
            project=wandb_project,
            name=wandb_run,
            config={
                "num_epochs":          num_epochs,
                "batch_size":          train_loader.batch_size,
                "learning_rate":       lr,
                "warmup_steps":        warmup_steps,
                "grad_clip":           grad_clip,
                "ckpt_every_n_epochs": ckpt_every_n_epochs,
                "mixed_precision":     mixed_precision,
                "device":              str(device),
            },
        )
    elif wandb_project:
        print("[warn] wandb not installed; skipping W&B logging.")

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def lr_lambda(step: int) -> float:
        return min(1.0, (step + 1) / max(warmup_steps, 1))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # ---- optional resume -------------------------------------------------- #
    start_epoch = 0
    step        = 0
    best_val    = float("inf")
    patience    = 0

    if resume_from is not None:
        start_epoch, step, best_val = load_checkpoint(
            resume_from, model, opt, sched
        )

    # ----------------------------------------------------------------------- #

    for epoch in range(start_epoch, num_epochs):
        model.train()
        run_loss = 0.0
        n = 0

        for fi, fnext, action in tqdm(train_loader, desc=f"train epoch {epoch}", leave=False):
            fi     = fi.to(device, non_blocking=True)
            fnext  = fnext.to(device, non_blocking=True)
            action = action.to(device, non_blocking=True)
            device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
            with autocast(device_type=device_type, enabled=mixed_precision, dtype=torch.float16):       
                pred   = model(fi, fnext)
                target = model.normalize_action(action)
                loss   = F.mse_loss(pred, target)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()

            run_loss += loss.item() * fi.size(0)
            n        += fi.size(0)
            step     += 1

            if step % log_every == 0:
                current_lr = sched.get_last_lr()[0]
                print(f"epoch {epoch}  step {step}  "
                      f"lr {current_lr:.2e}  "
                      f"loss {loss.item():.4f}")
                if wandb is not None and wandb_project:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/lr":   current_lr,
                        "step":       step,
                    })

        val        = evaluate(model, val_loader, device)
        train_loss = run_loss / max(n, 1)
        print(f"[epoch {epoch}]  train {train_loss:.4f}  val {val:.4f}")

        if wandb is not None and wandb_project:
            wandb.log({
                "epoch":            epoch,
                "train/epoch_loss": train_loss,
                "val/loss":         val,
            })

        # ---- always save "last" so you can resume after any interruption -- #
        save_checkpoint(
            ckpt_dir / "last.pt", model, opt, sched, epoch, step, best_val
        )

        # ---- periodic checkpoint ----------------------------------------- #
        if ckpt_every_n_epochs > 0 and (epoch + 1) % ckpt_every_n_epochs == 0:
            save_checkpoint(
                ckpt_dir / f"epoch_{epoch:04d}.pt",
                model, opt, sched, epoch, step, best_val,
            )
            print(f"  saved periodic checkpoint  epoch_{epoch:04d}.pt")

        # ---- best-val checkpoint + early stopping ------------------------- #
        if val < best_val - 1e-5:
            best_val = val
            patience = 0
            save_checkpoint(
                ckpt_dir / "best.pt", model, opt, sched, epoch, step, best_val
            )
            print(f"  saved best checkpoint  (val {best_val:.4f})")
        else:
            patience += 1
            if patience >= early_stop_patience:
                print(f"early stop at epoch {epoch}  "
                      f"(best val {best_val:.4f})")
                if wandb is not None and wandb_project:
                    wandb.log({
                        "early_stop_epoch": epoch,
                        "best_val_loss":    best_val,
                    })
                break

    if wandb is not None and wandb_project:
        wandb.finish()


# --------------------------------------------------------------------------- #
# Entry
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train UniPi-style RGB inverse dynamics model."
    )
    parser.add_argument("--root",        default="RealEstate10K",
                        help="Root containing {split}/*.txt.")
    parser.add_argument("--img_size",    type=int, default=224)
    parser.add_argument("--hidden",      type=int, default=64,
                        help="Conv encoder channel width.")
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_epochs",  type=int, default=50)
    parser.add_argument("--frames_root", default="frames",
                        help="Path to extracted frames root (contains train/ and test/).")
    parser.add_argument("--device",      default=None,
                        help="Torch device. Defaults to cuda > mps > cpu.")
    # ---- checkpoint args -------------------------------------------------- #
    parser.add_argument("--ckpt_dir",            default="checkpoints",
                        help="Directory to save checkpoints.")
    parser.add_argument("--ckpt_every_n_epochs", type=int, default=5,
                        help="Save a periodic checkpoint every N epochs (0=off).")
    parser.add_argument("--resume_from",         default=None,
                        help="Path to a checkpoint to resume training from.")
    # ----------------------------------------------------------------------- #
    parser.add_argument("--wandb_project",   default="idm-conv",
                        help="W&B project name.")
    parser.add_argument("--wandb_run",       default=None,
                        help="W&B run name.")
    parser.add_argument("--mixed_precision", action="store_true",
                        help="Enable mixed precision training (autocast + GradScaler).")
    args = parser.parse_args()

    if args.device is None:
        args.device = (
            "cuda" if torch.cuda.is_available()
            else "mps"  if torch.backends.mps.is_available()
            else "cpu"
        )

    train_ds = RGBPairDataset(
        args.root,
        frames_root=args.frames_root,
        split="train",
        img_size=args.img_size,
    )
    val_ds = RGBPairDataset(
        args.root,
        frames_root=args.frames_root,
        split="test",
        img_size=args.img_size,
    )
    print(f"train pairs: {len(train_ds):,}   val pairs: {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_pairs,
        pin_memory=(args.device != "cpu"),
        # persistent_workers removed — leaks memory across epochs with large datasets
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_pairs,
        pin_memory=(args.device != "cpu"),
    )

    model = InverseDynamicsModel(hidden=args.hidden, out_dim=6)

    # Only compute action stats from scratch if we are NOT resuming
    # (stats are baked into the checkpoint when resuming)
    if args.resume_from is None:
        mean, std = compute_action_stats(train_ds)
        model.set_action_stats(mean, std)
        print(f"action mean: {mean.tolist()}")
        print(f"action std:  {std.tolist()}")

    train(
        model, train_loader, val_loader,
        num_epochs=args.num_epochs,
        device=args.device,
        ckpt_dir=args.ckpt_dir,
        ckpt_every_n_epochs=args.ckpt_every_n_epochs,
        resume_from=args.resume_from,
        mixed_precision=args.mixed_precision,
        wandb_project=args.wandb_project,
        wandb_run=args.wandb_run,
    )