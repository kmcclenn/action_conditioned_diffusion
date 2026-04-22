"""
RealEstate10K dataset loader.

Each clip is a .txt file:
  Line 0: YouTube URL
  Lines 1+: timestamp fx fy cx cy | R(3x3) t(3x1)  (row-major, 19 cols total)

Returns sequences of camera intrinsics K (3x3), poses P=[R|t] (3x4),
and optionally 6D SE(3) twists (v, omega) between consecutive frames as "actions".
If frames_root is given and frames exist on disk, also returns image tensors.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


def parse_clip(path: str | Path) -> dict[str, np.ndarray]:
    """Parse a single RealEstate10K .txt file.

    Returns:
        url: str
        timestamps: (N,) int64, microseconds
        K: (N, 3, 3) float32, intrinsics (normalized coords)
        P: (N, 3, 4) float32, extrinsic [R|t]
    """
    with open(path) as f:
        lines = f.read().splitlines()

    url = lines[0]
    rows = np.array([line.split() for line in lines[1:] if line], dtype=np.float64)

    timestamps = rows[:, 0].astype(np.int64)
    # cols 1-4: fx fy cx cy; cols 5-6: undocumented (zero skew params in practice)
    fx, fy, cx, cy = rows[:, 1], rows[:, 2], rows[:, 3], rows[:, 4]

    N = len(rows)
    K = np.zeros((N, 3, 3), dtype=np.float32)
    K[:, 0, 0] = fx
    K[:, 1, 1] = fy
    K[:, 0, 2] = cx
    K[:, 1, 2] = cy
    K[:, 2, 2] = 1.0

    P = rows[:, 7:].astype(np.float32).reshape(N, 3, 4)

    return {"url": url, "timestamps": timestamps, "K": K, "P": P}


def relative_pose(P: torch.Tensor) -> torch.Tensor:
    """Compute relative pose E_rel = E_{t+1} @ E_t^{-1} for consecutive frames.

    E_t = [[R_t, t_t], [0, 1]]  (4x4 homogeneous)
    E_t^{-1} = [[R_t^T, -R_t^T t_t], [0, 1]]

    Args:
        P: (N, 3, 4) extrinsic matrices [R|t]

    Returns:
        (N-1, 3, 4) upper 3 rows of E_rel
    """
    N = P.shape[0]
    R = P[:, :, :3]   # (N, 3, 3)
    t = P[:, :, 3:]   # (N, 3, 1)

    # Build E (N, 4, 4)
    E = torch.zeros(N, 4, 4, dtype=P.dtype, device=P.device)
    E[:, :3, :3] = R
    E[:, :3, 3:] = t
    E[:, 3, 3]   = 1.0

    # Build E_inv (N, 4, 4)  — analytic inverse for SE(3)
    E_inv = torch.zeros(N, 4, 4, dtype=P.dtype, device=P.device)
    Rt = R.transpose(-1, -2)
    E_inv[:, :3, :3] = Rt
    E_inv[:, :3, 3:] = -Rt @ t
    E_inv[:, 3, 3]   = 1.0

    E_rel = E[1:] @ E_inv[:-1]   # (N-1, 4, 4)
    return E_rel[:, :3, :]        # (N-1, 3, 4)


def _skew(w: torch.Tensor) -> torch.Tensor:
    """(..., 3) -> (..., 3, 3) skew-symmetric matrix [w]_x."""
    zero = torch.zeros_like(w[..., 0])
    row0 = torch.stack([zero, -w[..., 2],  w[..., 1]], dim=-1)
    row1 = torch.stack([ w[..., 2], zero, -w[..., 0]], dim=-1)
    row2 = torch.stack([-w[..., 1],  w[..., 0], zero], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def se3_log(T: torch.Tensor) -> torch.Tensor:
    """SE(3) log map: (..., 3, 4) or (..., 4, 4) transforms -> (..., 6) twists (v, omega).

    Returns xi = (v, omega) such that exp(xi) = T. Uses Taylor series near theta=0
    to avoid singularities. Note: pathological at theta = pi (rotation by 180 deg).
    """
    if T.shape[-2] == 4:
        T = T[..., :3, :]
    R = T[..., :3, :3]
    t = T[..., :3, 3]

    cos_theta = (R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5
    cos_theta = cos_theta.clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)
    small = theta < 1e-4

    sin_theta = torch.sin(theta)
    a = torch.where(small, 0.5 + theta**2 / 12.0, theta / (2.0 * sin_theta + 1e-20))
    skew_R = a[..., None, None] * (R - R.transpose(-1, -2))
    omega = torch.stack([skew_R[..., 2, 1], skew_R[..., 0, 2], skew_R[..., 1, 0]], dim=-1)

    # V^{-1} = I - (1/2)[w]_x + c2 [w]_x^2, with c2 = (1 - (theta/2) cot(theta/2)) / theta^2
    th2 = theta / 2.0
    c2 = torch.where(
        small,
        1.0 / 12.0 + theta**2 / 720.0,
        (1.0 - th2 * torch.cos(th2) / (torch.sin(th2) + 1e-20)) / (theta**2 + 1e-20),
    )

    W = _skew(omega)
    W2 = W @ W
    I3 = torch.eye(3, dtype=T.dtype, device=T.device)
    V_inv = I3 - 0.5 * W + c2[..., None, None] * W2
    v = (V_inv @ t.unsqueeze(-1)).squeeze(-1)

    return torch.cat([v, omega], dim=-1)


class RealEstate10KDataset(Dataset):
    """PyTorch Dataset for RealEstate10K.

    Each item is a dict with tensors for a fixed-length window of frames
    sampled from a clip.

    Args:
        root: path to the RealEstate10K directory (contains train/ and test/)
        split: "train" or "test"
        seq_len: number of consecutive frames per sample
        stride: step between frames (1 = every frame, 2 = every other, ...)
        return_actions: if True, include relative_poses (N-1, 3, 4) as "actions"
        img_size: (W, H) — scales K to pixel coords; also used as resize target
                  when loading images. If None, K stays in normalized coords.
        frames_root: directory produced by download_frames.py; if provided and
                     frames exist on disk, each item includes an "images" tensor
                     of shape (seq_len, 3, H, W) with values in [0, 1].
        frames_only: if True, skip samples where any frame is missing from disk
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        seq_len: int = 16,
        stride: int = 1,
        return_actions: bool = True,
        img_size: Optional[tuple[int, int]] = None,
        frames_root: Optional[str | Path] = None,
        frames_only: bool = False,
    ):
        self.root = Path(root)
        self.split = split
        self.seq_len = seq_len
        self.stride = stride
        self.return_actions = return_actions
        self.img_size = img_size
        self.frames_root = Path(frames_root) if frames_root else None
        self.frames_only = frames_only

        if img_size:
            self._img_transform = transforms.Compose([
                transforms.Resize(img_size[::-1]),  # Resize takes (H, W)
                transforms.ToTensor(),
            ])
        else:
            self._img_transform = transforms.ToTensor()

        clip_dir = self.root / split
        self.clip_paths = sorted(clip_dir.glob("*.txt"))
        if not self.clip_paths:
            raise FileNotFoundError(f"No .txt files found in {clip_dir}")

        # Pre-compute (clip_idx, start_frame) index over all clips
        self._index: list[tuple[int, int]] = []
        self._clips: list[dict] = []

        window = (seq_len - 1) * stride + 1
        for ci, path in enumerate(self.clip_paths):
            clip = parse_clip(path)
            self._clips.append(clip)
            N = len(clip["timestamps"])
            clip_id = path.stem

            for start in range(0, N - window + 1):
                if frames_only and self.frames_root is not None:
                    frame_dir = self.frames_root / split / clip_id
                    frame_ids = range(start, start + seq_len * stride, stride)
                    ts_list = clip["timestamps"][list(frame_ids)]
                    if not all((frame_dir / f"{ts}.jpg").exists() for ts in ts_list):
                        continue
                self._index.append((ci, start))

    def __len__(self) -> int:
        return len(self._index)

    def _load_images(self, clip_id: str, timestamps: np.ndarray) -> Optional[torch.Tensor]:
        """Load frames from disk. Returns (seq_len, 3, H, W) or None if any missing."""
        if self.frames_root is None:
            return None
        frame_dir = self.frames_root / self.split / clip_id
        imgs = []
        for ts in timestamps:
            p = frame_dir / f"{ts}.jpg"
            if not p.exists():
                return None
            imgs.append(self._img_transform(Image.open(p).convert("RGB")))
        return torch.stack(imgs)  # (seq_len, 3, H, W)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ci, start = self._index[idx]
        clip = self._clips[ci]
        clip_id = self.clip_paths[ci].stem

        frame_ids = list(range(start, start + self.seq_len * self.stride, self.stride))
        ts  = torch.from_numpy(clip["timestamps"][frame_ids])
        K   = torch.from_numpy(clip["K"][frame_ids])
        P   = torch.from_numpy(clip["P"][frame_ids])

        if self.img_size is not None:
            W, H = self.img_size
            scale = torch.tensor([W, H, 1.0], dtype=K.dtype)
            K = K * scale[None, :, None]

        out: dict = {"timestamps": ts, "K": K, "P": P, "url": clip["url"]}
        if self.return_actions:
            out["actions"] = se3_log(relative_pose(P))   # (seq_len-1, 6)

        images = self._load_images(clip_id, clip["timestamps"][frame_ids])
        if images is not None:
            out["images"] = images   # (seq_len, 3, H, W), float32 in [0, 1]

        return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Path to RealEstate10K/")
    parser.add_argument("--split", default="train")
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--frames_root", default=None, help="Path to downloaded frames directory")
    args = parser.parse_args()

    ds = RealEstate10KDataset(
        args.root, split=args.split, seq_len=args.seq_len, stride=args.stride,
        frames_root=args.frames_root,
    )
    print(f"{args.split}: {len(ds)} samples across {len(ds.clip_paths)} clips")

    sample = ds[0]
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {tuple(v.shape)} {v.dtype}")
        else:
            print(f"  {k}: {v!r}")
