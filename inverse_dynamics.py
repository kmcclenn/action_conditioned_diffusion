"""
Inverse dynamics model: predict the 6D SE(3) twist a=(v, omega) between two
consecutive frames (I_i, I_{i+1}).

  - Frozen DINOv2 ViT-S/14 encoder  -> CLS tokens z_i, z_{i+1} in R^384
  - Fusion: concat([z_i, z_{i+1}, z_{i+1} - z_i])  ->  R^1152
  - MLP head: 1152 -> 512 -> 512 -> 6
  - Target normalization via registered (mean, std) buffers

Supervised target comes from dataset.se3_log(relative_pose(P)).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from dataset import parse_clip, relative_pose, se3_log


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

class InverseDynamicsModel(nn.Module):
    """Predicts a 6D action (v, omega) from a pair of frames."""

    def __init__(self, dropout: float = 0.1):
        super().__init__()

        encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False
        self.encoder = encoder
        self.feat_dim = 384   # ViT-S/14 CLS token

        in_dim = 3 * self.feat_dim
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

    def train(self, mode: bool = True):
        # Keep the encoder in eval mode regardless of the wrapper's mode
        super().train(mode)
        self.encoder.eval()
        return self

    def set_action_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.action_mean.copy_(mean.to(self.action_mean))
        self.action_std.copy_(std.to(self.action_std).clamp_min(1e-8))

    def normalize_action(self, a: torch.Tensor) -> torch.Tensor:
        return (a - self.action_mean) / self.action_std

    def denormalize_action(self, a_norm: torch.Tensor) -> torch.Tensor:
        return a_norm * self.action_std + self.action_mean

    @torch.no_grad()
    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 224, 224) ImageNet-normalized -> (B, 384) CLS token."""
        feats = self.encoder.forward_features(x)
        return feats["x_norm_clstoken"]

    def forward(self, img_i: torch.Tensor, img_next: torch.Tensor) -> torch.Tensor:
        """Return normalized action prediction, shape (B, 6)."""
        z_i = self._encode(img_i)
        z_n = self._encode(img_next)
        h = torch.cat([z_i, z_n, z_n - z_i], dim=-1)
        return self.head(h)

    @torch.no_grad()
    def predict(self, img_i: torch.Tensor, img_next: torch.Tensor) -> torch.Tensor:
        """Return un-normalized action, shape (B, 6). Expects preprocessed input."""
        self.eval()
        return self.denormalize_action(self.forward(img_i, img_next))

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return (p for p in self.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# Preprocessing — apply AFTER any per-pair augmentation
# --------------------------------------------------------------------------- #

_PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def preprocess_pair(
    img_i: torch.Tensor, img_next: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize to 224x224 and ImageNet-normalize. Expects tensors in [0, 1]."""
    return _PREPROCESS(img_i), _PREPROCESS(img_next)


# --------------------------------------------------------------------------- #
# Dataset stub — fill in with your own pair-indexing logic
# --------------------------------------------------------------------------- #

class FramePairDataset(Dataset):
    """Consecutive-frame pairs from RealEstate10K with 6D action targets.

    Layout expectations (match download_frames.py + dataset.py):
        root        = RealEstate10K/
        frames_root = frames/
        JPEGs at    frames/{split}/{clip_id}/{timestamp_us}.jpg
        Clip .txt   at RealEstate10K/{split}/{clip_id}.txt

    Args:
        root:        path to RealEstate10K/ (contains train/ and test/)
        frames_root: path to materialized-frame directory
        split:       "train" or "test"
        img_size:    (W, H) resize target. Default (224, 224) matches DINOv2.
                     Set None to keep native resolution (requires all frames
                     to share a size or you'll get collate errors).

    Each __getitem__ returns (img_i, img_next, action):
        img_i:    (3, H, W) float32 in [0, 1]
        img_next: (3, H, W) float32 in [0, 1]
        action:   (6,)      float32  — se3_log of E_{i+1} @ E_i^{-1}
    """

    def __init__(
        self,
        root: str | Path,
        frames_root: str | Path,
        split: str = "train",
        img_size: Optional[tuple[int, int]] = (224, 224),
    ):
        self.root = Path(root)
        self.frames_root = Path(frames_root)
        self.split = split
        self.img_size = img_size

        if img_size is not None:
            W, H = img_size
            self._transform = transforms.Compose([
                transforms.Resize((H, W), antialias=True),
                transforms.ToTensor(),
            ])
        else:
            self._transform = transforms.ToTensor()

        clip_dir = self.root / split
        clip_paths = sorted(clip_dir.glob("*.txt"))
        if not clip_paths:
            raise FileNotFoundError(f"No .txt files found in {clip_dir}")

        self._pairs: list[tuple[Path, Path, torch.Tensor]] = []
        for clip_path in clip_paths:
            clip = parse_clip(clip_path)
            clip_id = clip_path.stem
            frame_dir = self.frames_root / split / clip_id
            if not frame_dir.exists():
                continue

            P = torch.from_numpy(clip["P"])                  # (N, 3, 4)
            actions = se3_log(relative_pose(P))              # (N-1, 6)
            ts = clip["timestamps"]

            for i in range(len(ts) - 1):
                p_i = frame_dir / f"{ts[i]}.jpg"
                p_n = frame_dir / f"{ts[i + 1]}.jpg"
                if p_i.exists() and p_n.exists():
                    self._pairs.append((p_i, p_n, actions[i].clone()))

        if not self._pairs:
            raise RuntimeError(
                f"No valid frame pairs found. Check that {self.frames_root}/{split}/ "
                f"contains JPEGs named by the timestamps in {clip_dir}/*.txt."
            )

    def __len__(self) -> int:
        return len(self._pairs)

    def _load(self, path: Path) -> torch.Tensor:
        return self._transform(Image.open(path).convert("RGB"))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p_i, p_n, action = self._pairs[idx]
        return self._load(p_i), self._load(p_n), action

    def all_actions(self) -> torch.Tensor:
        """(N, 6) tensor of all action targets — fast path for stats."""
        return torch.stack([p[2] for p in self._pairs])


def collate_pairs(batch):
    img_i    = torch.stack([b[0] for b in batch])
    img_next = torch.stack([b[1] for b in batch])
    action   = torch.stack([b[2] for b in batch])
    return img_i, img_next, action


# --------------------------------------------------------------------------- #
# Action statistics (one pass over the train loader)
# --------------------------------------------------------------------------- #

@torch.no_grad()
def compute_action_stats(
    source: DataLoader | FramePairDataset,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-dim mean/std of the action target. Uses the fast path if given a
    FramePairDataset (avoids loading images)."""
    if isinstance(source, FramePairDataset):
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
    for img_i, img_next, action in loader:
        img_i    = img_i.to(device, non_blocking=True)
        img_next = img_next.to(device, non_blocking=True)
        action   = action.to(device, non_blocking=True)
        img_i, img_next = preprocess_pair(img_i, img_next)
        target = model.normalize_action(action)
        pred = model(img_i, img_next)
        total_sq += F.mse_loss(pred, target, reduction="sum").item()
        n += img_i.size(0)
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

    opt = torch.optim.Adam(model.trainable_parameters(), lr=lr)

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
        for img_i, img_next, action in train_loader:
            img_i    = img_i.to(device, non_blocking=True)
            img_next = img_next.to(device, non_blocking=True)
            action   = action.to(device, non_blocking=True)

            img_i, img_next = preprocess_pair(img_i, img_next)
            target = model.normalize_action(action)

            pred = model(img_i, img_next)
            loss = F.mse_loss(pred, target)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.trainable_parameters(), grad_clip)
            opt.step()
            sched.step()

            run_sq += loss.item() * img_i.size(0)
            n += img_i.size(0)
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
    parser.add_argument("--root",        default="RealEstate10K")
    parser.add_argument("--frames_root", default="frames")
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_epochs",  type=int, default=50)
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train_ds = FramePairDataset(args.root, args.frames_root, split="train")
    val_ds   = FramePairDataset(args.root, args.frames_root, split="test")
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
