"""
Action-conditioned DDPM over fixed-length 6D SE(3) twist sequences.

Given precomputed CroCoV2 patch tokens for a *start* and *finish* image
2 s apart, generate the n=8 relative twists (v, omega) at 4 Hz that take
the agent from start to finish (default: 8 / 4 Hz; configurable). Each
twist is the SE(3) log of the cumulative pose delta over 0.25 s.
ε-prediction MSE training + classifier-free guidance (CFG).

Pieces:
  - WindowDataset:           per-clip 2 s windows snapped to a 4 Hz grid
                             (9 frames, 8 actions), 0.5 s sliding stride.
                             One window sampled per clip per __getitem__.
  - Diffusion:               cosine β-schedule DDPM with K=100 steps. Plain
                             object (not nn.Module); buffers move via .to().
  - MLPActionDenoiser:       mean-pool tokens -> MLP cond -> residual MLP
                             over flattened (n=8, 6) action sequence + time.
  - TransformerActionDenoiser:
                             keep tokens; learned start/finish image-of-origin
                             embeddings; cross-attention from action tokens
                             onto the (2*N_tok, d) image memory.
  - SE(3) helpers:           exp_se3 / compose_twists / pose_error for the
                             integrated final-pose evaluation metric.

Run:
    # Default CroCo features at features_croco/{train,test}/{clip}.pt:
    python action_diffusion.py --head mlp         --num_epochs 1000
    python action_diffusion.py --head transformer --num_epochs 1000

NOT in MVP (mark as future ablations):
  - DDIM / accelerated sampling (vanilla DDPM only).
  - EMA on weights.
  - Variable n.
  - Causal mask in the transformer self-attention (full self-attn today).
  - CFG scale sweep at eval (single --guidance_scale today; rerun for sweep).
  - Mixed-precision (bf16) training (fp32 only today).
  - Per-slot null embeddings (single shared null per design doc).
  - Sliding-window enumeration; we sample one random window per clip per
    __getitem__ to match the doc's 'one example per clip' implication while
    still giving epoch-level data variety.
  - WD-exempt param groups (LayerNorm/biases get full weight decay today).
  - DINOv2 features as conditioning (CroCo only — feat_dim=768, 196 tokens).
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader, Dataset

from dataset import parse_clip, relative_pose, se3_log, snap_to_time_grid
from inverse_dynamics import CachedPairDataset, compute_action_stats


# --------------------------------------------------------------------------- #
# SE(3) exponential map and pose error (for integrated final-pose metric)
# --------------------------------------------------------------------------- #

def _skew(w: torch.Tensor) -> torch.Tensor:
    """(..., 3) -> (..., 3, 3) skew-symmetric matrix [w]_x."""
    zero = torch.zeros_like(w[..., 0])
    row0 = torch.stack([zero,       -w[..., 2],  w[..., 1]], dim=-1)
    row1 = torch.stack([ w[..., 2],  zero,      -w[..., 0]], dim=-1)
    row2 = torch.stack([-w[..., 1],  w[..., 0],  zero      ], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def exp_se3(xi: torch.Tensor) -> torch.Tensor:
    """SE(3) exp map: (..., 6) twist (v, omega) -> (..., 4, 4) homogeneous T.

    Closed form with Taylor expansion near theta=0 to avoid singularities.
    """
    v = xi[..., :3]
    w = xi[..., 3:]
    theta = w.norm(dim=-1)
    small = theta < 1e-4
    th_safe = theta.clamp_min(1e-20)

    W  = _skew(w)
    W2 = W @ W
    I3 = torch.eye(3, dtype=xi.dtype, device=xi.device)

    a = torch.where(small, 1.0 - theta**2 / 6.0,         torch.sin(theta) / th_safe)
    b = torch.where(small, 0.5 - theta**2 / 24.0,        (1.0 - torch.cos(theta)) / th_safe**2)
    c = torch.where(small, 1.0 / 6.0 - theta**2 / 120.0, (theta - torch.sin(theta)) / th_safe**3)

    R = I3 + a[..., None, None] * W + b[..., None, None] * W2
    V = I3 + b[..., None, None] * W + c[..., None, None] * W2
    t = (V @ v.unsqueeze(-1)).squeeze(-1)

    T = torch.zeros(*xi.shape[:-1], 4, 4, dtype=xi.dtype, device=xi.device)
    T[..., :3, :3] = R
    T[..., :3,  3] = t
    T[...,  3,  3] = 1.0
    return T


def compose_twists(actions: torch.Tensor) -> torch.Tensor:
    """(..., n, 6) twists -> (..., 4, 4) integrated final-pose transform.

    Right-to-left composition matches dataset.py's `relative_pose` convention:
    each xi_t = log(E_{t+1} @ E_t^{-1}), so iterating gives
        E_n @ E_0^{-1} = exp(xi_{n-1}) @ exp(xi_{n-2}) @ ... @ exp(xi_0).
    Returned T is the SE(3) transform that takes the start camera into the
    finish camera in world coordinates.
    """
    Ts = exp_se3(actions)            # (..., n, 4, 4)
    n  = actions.shape[-2]
    out = Ts[..., -1, :, :]
    for t in range(n - 2, -1, -1):
        out = out @ Ts[..., t, :, :]
    return out


def pose_error(T_pred: torch.Tensor, T_gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Translation L2 and rotation angle (deg) between two (..., 4, 4) transforms."""
    t_err  = (T_pred[..., :3, 3] - T_gt[..., :3, 3]).norm(dim=-1)
    R_diff = T_pred[..., :3, :3] @ T_gt[..., :3, :3].transpose(-1, -2)
    cos_t  = ((R_diff.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
    r_err  = torch.acos(cos_t) * (180.0 / math.pi)
    return t_err, r_err


# --------------------------------------------------------------------------- #
# DDPM (cosine schedule, ancestral sampling with CFG)
# --------------------------------------------------------------------------- #

class Diffusion:
    """Cosine-schedule DDPM (Nichol & Dhariwal, 2021). K=100 steps default.

    Plain Python object — not an nn.Module — because it owns no learnable
    parameters. Tensors live on whichever device .to(device) was last called
    with. Pass it into train()/sample() with the same device as the model.
    """

    def __init__(self, num_steps: int = 100, s: float = 0.008,
                 device: torch.device | str = "cpu"):
        self.T = num_steps
        steps = num_steps + 1
        ts = torch.linspace(0, num_steps, steps) / num_steps
        f_t = torch.cos((ts + s) / (1 + s) * math.pi / 2.0) ** 2
        alphas_cumprod = f_t / f_t[0]
        betas = (1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]).clamp(max=0.999)

        self.betas          = betas.to(device)
        self.alphas         = (1.0 - self.betas)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod          = self.alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1.0 - self.alphas_cumprod).sqrt()

    def to(self, device: torch.device | str) -> "Diffusion":
        for k in ("betas", "alphas", "alphas_cumprod",
                  "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod"):
            setattr(self, k, getattr(self, k).to(device))
        return self

    def q_sample(self, x0: torch.Tensor, k: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """x_k = sqrt(α_bar_k) x_0 + sqrt(1 - α_bar_k) ε. k: (B,) long."""
        view = (-1,) + (1,) * (x0.dim() - 1)
        return (self.sqrt_alphas_cumprod[k].view(view) * x0
                + self.sqrt_one_minus_alphas_cumprod[k].view(view) * eps)


# --------------------------------------------------------------------------- #
# Time embedding + residual MLP block
# --------------------------------------------------------------------------- #

class SinusoidalTimeEmbedding(nn.Module):
    """Standard transformer-style sinusoidal embedding of an integer step k."""

    def __init__(self, dim: int):
        super().__init__()
        if dim % 2:
            raise ValueError("SinusoidalTimeEmbedding dim must be even.")
        self.dim = dim

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0)
                          * torch.arange(half, dtype=torch.float32, device=k.device) / half)
        args = k.float()[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResidualMLPBlock(nn.Module):
    """Pre-norm residual MLP block: x + MLP(LN(x))."""

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


# --------------------------------------------------------------------------- #
# Action stat buffers + CFG helper (shared by both denoiser heads)
# --------------------------------------------------------------------------- #

class _ActionDenoiserBase(nn.Module):
    """Shared (de)normalization buffers and forward signature for both heads.

    Subclasses implement `forward(x_k, k, z_i, z_f, uncond_mask=None)` ->
    epsilon prediction in normalized action space, shape (B, n, 6).
    """

    def __init__(self, n_actions: int, action_dim: int = 6):
        super().__init__()
        self.n_actions = n_actions
        self.action_dim = action_dim
        self.register_buffer("action_mean", torch.zeros(action_dim))
        self.register_buffer("action_std",  torch.ones(action_dim))

    def set_action_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.action_mean.copy_(mean.to(self.action_mean))
        self.action_std.copy_(std.to(self.action_std).clamp_min(1e-8))

    def normalize_action(self, a: torch.Tensor) -> torch.Tensor:
        return (a - self.action_mean) / self.action_std

    def denormalize_action(self, a_norm: torch.Tensor) -> torch.Tensor:
        return a_norm * self.action_std + self.action_mean


# --------------------------------------------------------------------------- #
# Head A: MLP denoiser
# --------------------------------------------------------------------------- #

class MLPActionDenoiser(_ActionDenoiserBase):
    """ε̂ = f(x_k, k, mean_pool(z_i), mean_pool(z_f)).

    Conditioning:
        c_in = [mean_pool(z_i); mean_pool(z_f)]    in R^(2*feat_dim)
        c    = MLP(c_in)                           in R^(d_cond)

    Backbone:
        h0 = Linear([flat(x_k); c; t_emb])         in R^d
        h  = ResidualMLPBlock^L(h0)
        eps = LinearOut(LN(h))                     reshape to (B, n, 6)

    CFG null:
        single learned vector of size (2*feat_dim,) substituted for c_in
        when uncond_mask[b] is True.
    """

    def __init__(
        self,
        n_actions: int = 8,
        action_dim: int = 6,
        feat_dim: int = 768,
        d: int = 512,
        d_cond: int = 512,
        n_blocks: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__(n_actions=n_actions, action_dim=action_dim)
        self.feat_dim = feat_dim
        self.d = d

        self.cond_mlp = nn.Sequential(
            nn.Linear(2 * feat_dim, 2 * d_cond),
            nn.GELU(),
            nn.Linear(2 * d_cond, d_cond),
        )
        self.null_cond = nn.Parameter(torch.zeros(2 * feat_dim))
        nn.init.trunc_normal_(self.null_cond, std=0.02)

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(d_cond),
            nn.Linear(d_cond, d_cond),
            nn.GELU(),
            nn.Linear(d_cond, d_cond),
        )

        self.in_proj = nn.Linear(n_actions * action_dim + d_cond + d_cond, d)
        self.blocks  = nn.ModuleList([ResidualMLPBlock(d, dropout=dropout) for _ in range(n_blocks)])
        self.out_norm = nn.LayerNorm(d)
        self.out_proj = nn.Linear(d, n_actions * action_dim)

    def forward(
        self,
        x_k: torch.Tensor,                    # (B, n, 6)
        k:   torch.Tensor,                    # (B,) long
        z_i: torch.Tensor,                    # (B, N_tok, feat_dim)
        z_f: torch.Tensor,                    # (B, N_tok, feat_dim)
        uncond_mask: torch.Tensor | None = None,   # (B,) bool
    ) -> torch.Tensor:
        B = x_k.shape[0]
        cond_in = torch.cat([z_i.mean(dim=1), z_f.mean(dim=1)], dim=-1)  # (B, 2*F)
        if uncond_mask is not None:
            null = self.null_cond[None, :].expand(B, -1)
            cond_in = torch.where(uncond_mask[:, None], null, cond_in)
        c = self.cond_mlp(cond_in)                                       # (B, d_cond)
        t = self.time_mlp(k)                                             # (B, d_cond)

        h = self.in_proj(torch.cat([x_k.reshape(B, -1), c, t], dim=-1))  # (B, d)
        for block in self.blocks:
            h = block(h)
        eps = self.out_proj(self.out_norm(h)).reshape(B, self.n_actions, self.action_dim)
        return eps


# --------------------------------------------------------------------------- #
# Head B: Transformer denoiser (cross-attention over image tokens)
# --------------------------------------------------------------------------- #

class TransformerActionDenoiser(_ActionDenoiserBase):
    """Cross-attn decoder. Action tokens attend to (2*N_tok, d) image memory.

    Memory is `[proj(z_i) + role_start; proj(z_f) + role_finish] + img_pos`.
    CFG null is one learned d-dim token broadcast across all 2*N_tok positions
    (cheaper alternative from the design doc; used here as the default).
    Time embedding is added to every action token before the first block.
    Self-attention is full (not causal) per the design doc default.
    """

    def __init__(
        self,
        n_actions: int = 8,
        action_dim: int = 6,
        feat_dim: int = 768,
        n_tokens_per_img: int = 196,
        d: int = 384,
        n_heads: int = 8,
        n_blocks: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__(n_actions=n_actions, action_dim=action_dim)
        self.feat_dim = feat_dim
        self.n_tokens = n_tokens_per_img
        self.d = d

        self.img_proj = nn.Linear(feat_dim, d) if feat_dim != d else nn.Identity()
        self.img_pos  = nn.Parameter(torch.zeros(1, 2 * n_tokens_per_img, d))
        self.img_role = nn.Parameter(torch.zeros(2, 1, d))   # [start, finish]
        self.null_tok = nn.Parameter(torch.zeros(1, 1, d))   # CFG broadcast null
        nn.init.trunc_normal_(self.img_pos,  std=0.02)
        nn.init.trunc_normal_(self.img_role, std=0.02)
        nn.init.trunc_normal_(self.null_tok, std=0.02)

        self.act_in  = nn.Linear(action_dim, d)
        self.act_pos = nn.Parameter(torch.zeros(1, n_actions, d))
        nn.init.trunc_normal_(self.act_pos, std=0.02)

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, d),
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d,
            nhead=n_heads,
            dim_feedforward=int(d * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_blocks)

        self.out_norm = nn.LayerNorm(d)
        self.out      = nn.Linear(d, action_dim)

    def forward(
        self,
        x_k: torch.Tensor,                    # (B, n, 6)
        k:   torch.Tensor,                    # (B,) long
        z_i: torch.Tensor,                    # (B, N_tok, feat_dim)
        z_f: torch.Tensor,                    # (B, N_tok, feat_dim)
        uncond_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = x_k.shape[0]

        z_i_p  = self.img_proj(z_i) + self.img_role[0:1]   # (B, N, d)
        z_f_p  = self.img_proj(z_f) + self.img_role[1:2]
        memory = torch.cat([z_i_p, z_f_p], dim=1) + self.img_pos    # (B, 2N, d)
        if uncond_mask is not None:
            null = self.null_tok.expand(B, memory.shape[1], -1)
            memory = torch.where(uncond_mask[:, None, None], null, memory)

        h = self.act_in(x_k) + self.act_pos                # (B, n, d)
        t_emb = self.time_mlp(k)                            # (B, d)
        h = h + t_emb[:, None, :]

        h = self.decoder(tgt=h, memory=memory)              # (B, n, d)
        return self.out(self.out_norm(h))                   # (B, n, 6)


# --------------------------------------------------------------------------- #
# Dataset: 9-frame windows backed by precomputed CroCo features
# --------------------------------------------------------------------------- #

class WindowDataset(Dataset):
    """Per-clip fixed-duration windows over CroCo cached features.

    Window covers `n_actions / action_hz` seconds (default: 8 / 4 = 2.0 s),
    sliding by `window_stride_seconds` (default: 0.5 s). The 9 frames inside
    each window are snapped to a uniform `1/action_hz`-second grid; the 8
    actions are SE(3) log-deltas between consecutive snapped frames.

    Layout (mirrors CachedPairDataset in inverse_dynamics.py):
        root          = RealEstate10K/
        features_root = features_croco/
        Feature .pt   at features_croco/{split}/{clip}.pt — dict with
                         "timestamps" (LongTensor, N_f) and
                         "features"   (N_f, N_tok, feat_dim)
        Clip .txt     at RealEstate10K/{split}/{clip}.txt

    Behavior:
      - On init, build the per-clip `1/action_hz` snap grid and enumerate
        windows whose 9 grid slots are all jointly valid (pose + feature).
        Clips shorter than the window duration are dropped automatically.
      - Non-finite windows (rare; from se3_log near theta = pi) are dropped.
      - No outlier filter: 0.25 s deltas are larger than inter-frame deltas
        and the previous |a| <= 0.5 box would clip real motion.
      - len(self) = number of clips with at least one valid window.
      - __getitem__ behavior depends on `deterministic`:
          False (default, train): samples a *random* window per call →
              epoch-level data variety, len ≈ 'one example per clip' which
              matches the design doc's batching math (~12 batches at bs=256).
          True (eval): always returns windows[0] for a fixed val/test set.
              Required so per-epoch val loss and per-seed sampling eval
              compare predictions on the SAME inputs.

    Returns (z_start, z_finish, action_seq):
        z_start, z_finish: (N_tok, feat_dim)
        action_seq:        (n_actions, 6)  raw twists, NOT normalized.
    """

    def __init__(
        self,
        root: str | Path,
        features_root: str | Path,
        split: str = "train",
        n_actions: int = 8,
        action_hz: float = 4.0,
        window_stride_seconds: float = 0.5,
        deterministic: bool = False,
    ):
        self.root = Path(root)
        self.features_root = Path(features_root)
        self.split = split
        self.n_actions = n_actions
        self.action_hz = action_hz
        self.window_stride_seconds = window_stride_seconds
        self.deterministic = deterministic

        step_us = int(round(1e6 / action_hz))
        max_offset_us = step_us // 2
        stride_grid = int(round(window_stride_seconds * action_hz))
        if stride_grid < 1:
            raise ValueError(
                f"window_stride_seconds ({window_stride_seconds}) must be "
                f">= 1/action_hz ({1.0 / action_hz}).")
        n_window_slots = n_actions + 1

        clip_dir = self.root / split
        clip_paths = sorted(clip_dir.glob("*.txt"))
        if not clip_paths:
            raise FileNotFoundError(f"No .txt files found in {clip_dir}")

        # self._clip_features[i]: (N_f, N_tok, feat_dim) mmap-backed tensor
        # self._windows[i]:       list[(start_row, finish_row, action_seq)]
        self._clip_features: list[torch.Tensor] = []
        self._windows: list[list[tuple[int, int, torch.Tensor]]] = []

        n_drop_short = 0
        n_drop_nonfinite = 0

        for clip_path in clip_paths:
            feat_path = self.features_root / split / f"{clip_path.stem}.pt"
            if not feat_path.exists():
                continue

            clip = parse_clip(clip_path)
            poses = torch.from_numpy(clip["P"])
            pose_ts = [int(t) for t in clip["timestamps"]]

            cached = torch.load(feat_path, map_location="cpu",
                                weights_only=True, mmap=True)
            feats = cached["features"]
            feat_ts = cached["timestamps"].tolist()

            grid = snap_to_time_grid(pose_ts, feat_ts, step_us, max_offset_us)
            if len(grid) < n_window_slots:
                n_drop_short += 1
                continue

            windows: list[tuple[int, int, torch.Tensor]] = []
            for k in range(0, len(grid) - n_window_slots + 1, stride_grid):
                slots = grid[k : k + n_window_slots]
                if any(s is None for s in slots):
                    continue
                pose_indices = [s[0] for s in slots]
                feat_rows    = [s[1] for s in slots]
                rel = relative_pose(poses[pose_indices])    # (n_actions, 3, 4)
                actions = se3_log(rel)                      # (n_actions, 6)
                if not torch.isfinite(actions).all():
                    n_drop_nonfinite += 1
                    continue
                windows.append((feat_rows[0], feat_rows[-1], actions))

            if windows:
                self._clip_features.append(feats)
                self._windows.append(windows)

        if not self._windows:
            raise RuntimeError(
                f"No valid windows in {self.features_root}/{split}/. "
                f"Run cache_features.py --encoder croco first.")

        n_total = sum(len(w) for w in self._windows)
        window_seconds = n_actions / action_hz
        print(f"[{split}] {len(self._windows)} clips, {n_total} valid windows "
              f"({window_seconds:.2f}s @ {action_hz}Hz, stride {window_stride_seconds}s; "
              f"dropped {n_drop_short} short clips, {n_drop_nonfinite} nonfinite).")

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        windows = self._windows[idx]
        if self.deterministic:
            i = 0
        else:
            # randint uses the global generator; DataLoader workers each have
            # a distinct seed so multi-worker runs still get diverse windows.
            i = int(torch.randint(0, len(windows), (1,)).item())
        feats = self._clip_features[idx]
        s_row, f_row, action_seq = windows[i]
        return feats[s_row].clone(), feats[f_row].clone(), action_seq


def collate_windows(batch):
    z_i = torch.stack([b[0] for b in batch])
    z_f = torch.stack([b[1] for b in batch])
    a   = torch.stack([b[2] for b in batch])
    return z_i, z_f, a


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #

@torch.no_grad()
def sample(
    model: _ActionDenoiserBase,
    diffusion: Diffusion,
    z_i: torch.Tensor,
    z_f: torch.Tensor,
    guidance_scale: float = 1.0,
) -> torch.Tensor:
    """Vanilla ancestral DDPM with CFG. Returns NORMALIZED actions (B, n, 6).

    Caller is responsible for `model.denormalize_action(...)`.
    """
    model.eval()
    diffusion.to(z_i.device)   # defensive no-op when already on the right device
    B = z_i.shape[0]
    device = z_i.device
    n, ad = model.n_actions, model.action_dim

    x = torch.randn(B, n, ad, device=device)
    for k in reversed(range(diffusion.T)):
        kt = torch.full((B,), k, dtype=torch.long, device=device)
        eps_cond = model(x, kt, z_i, z_f, uncond_mask=None)
        if guidance_scale != 0.0:
            uncond_mask = torch.ones(B, dtype=torch.bool, device=device)
            eps_un = model(x, kt, z_i, z_f, uncond_mask=uncond_mask)
            eps = (1.0 + guidance_scale) * eps_cond - guidance_scale * eps_un
        else:
            eps = eps_cond

        ab_k    = diffusion.alphas_cumprod[k]
        ab_prev = diffusion.alphas_cumprod[k - 1] if k > 0 else torch.ones_like(ab_k)
        beta_k  = diffusion.betas[k]
        alpha_k = diffusion.alphas[k]

        x0_pred = (x - (1.0 - ab_k).sqrt() * eps) / ab_k.sqrt()
        mean = ((ab_prev.sqrt() * beta_k)        / (1.0 - ab_k)) * x0_pred \
             + ((alpha_k.sqrt() * (1.0 - ab_prev)) / (1.0 - ab_k)) * x
        if k > 0:
            var = beta_k * (1.0 - ab_prev) / (1.0 - ab_k)
            x = mean + var.sqrt() * torch.randn(B, n, ad, device=device)
        else:
            x = mean
    return x


# --------------------------------------------------------------------------- #
# Eval
# --------------------------------------------------------------------------- #

@torch.no_grad()
def evaluate_loss(model, diffusion, loader, device) -> float:
    """Per-element ε-MSE on the loader. No sampling — fast per-epoch metric.

    Conditioning is never dropped here (we evaluate the *conditional* model).
    """
    model.eval()
    total = 0.0
    n_elem = 0
    for z_i, z_f, action in loader:
        z_i = z_i.to(device); z_f = z_f.to(device); action = action.to(device)
        a0  = model.normalize_action(action)
        B   = a0.shape[0]
        k   = torch.randint(0, diffusion.T, (B,), device=device)
        eps = torch.randn_like(a0)
        a_k = diffusion.q_sample(a0, k, eps)
        eps_pred = model(a_k, k, z_i, z_f, uncond_mask=None)
        total += F.mse_loss(eps_pred, eps, reduction="sum").item()
        n_elem += a0.numel()
    return total / max(n_elem, 1)


@torch.no_grad()
def evaluate_sampling(
    model: _ActionDenoiserBase,
    diffusion: Diffusion,
    loader: DataLoader,
    device: torch.device,
    guidance_scale: float = 1.0,
    n_seeds: int = 8,
) -> dict[str, float]:
    """Action MSE + integrated SE(3) pose error, averaged over n_seeds samples.

    Reports mean and std across seeds. Std reflects sampler stochasticity, not
    test-set variance — i.e. how multimodal our predictions are per example.
    """
    model.eval()
    seed_action_mse: list[float] = []
    seed_t_err:      list[float] = []
    seed_r_err:      list[float] = []

    for s in range(n_seeds):
        torch.manual_seed(1234 + s)
        a_mse_sum = 0.0
        t_sum = 0.0
        r_sum = 0.0
        n_examples = 0
        n_elem = 0
        for z_i, z_f, action_gt in loader:

            # get inputs
            z_i = z_i.to(device)
            z_f = z_f.to(device)
            action_gt = action_gt.to(device)

            # sample actions and denormalize
            a_norm = sample(model, diffusion, z_i, z_f, guidance_scale=guidance_scale)
            a_pred = model.denormalize_action(a_norm)

            # mse loss between action sequences
            a_mse_sum += F.mse_loss(a_pred, action_gt, reduction="sum").item()

            # get final pose error
            T_pred = compose_twists(a_pred)
            T_gt = compose_twists(action_gt)
            t_err, r_err = pose_error(T_pred, T_gt)
            t_sum += t_err.sum().item()
            r_sum += r_err.sum().item()

            n_examples += action_gt.shape[0]
            n_elem  += action_gt.numel()
        seed_action_mse.append(a_mse_sum / max(n_elem, 1))
        seed_t_err.append(t_sum / max(n_examples, 1))
        seed_r_err.append(r_sum / max(n_examples, 1))

    def _mu_sd(xs: list[float]) -> tuple[float, float]:
        t = torch.tensor(xs, dtype=torch.float64)
        return float(t.mean().item()), float(t.std(unbiased=False).item())

    a_m, a_s = _mu_sd(seed_action_mse)
    t_m, t_s = _mu_sd(seed_t_err)
    r_m, r_s = _mu_sd(seed_r_err)
    return {
        "action_mse_mean":  a_m, "action_mse_std":  a_s,
        "trans_err_mean":   t_m, "trans_err_std":   t_s,
        "rot_err_deg_mean": r_m, "rot_err_deg_std": r_s,
        "guidance_scale":   guidance_scale,
        "n_seeds":          float(n_seeds),
    }


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def train(
    model: _ActionDenoiserBase,
    diffusion: Diffusion,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 1000,
    lr: float = 1e-4,
    weight_decay: float = 1e-6,
    p_uncond: float = 0.1,
    warmup_steps: int = 500,
    grad_clip: float = 1.0,
    device: str | torch.device = "cuda",
    log_every: int = 100,
    ckpt_dir: str | Path | None = "checkpoints/action_diffusion",
    ckpt_every: int = 50,
    use_wandb: bool = False,
    init_from: str | Path | None = None,
    guidance_scale: float = 1.0,
    sampling_eval_every: int = 10,
    sampling_eval_seeds: int = 1,
) -> None:
    device = torch.device(device)
    model = model.to(device)
    diffusion.to(device)

    start_epoch = 0
    best_val = float("inf")
    if init_from is not None:
        ckpt = torch.load(init_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_val = float(ckpt.get("val_loss", float("inf")))
        print(f"resumed from {init_from}: start_epoch={start_epoch}  "
              f"prev_val_loss={best_val:.4f}")

    if ckpt_dir is not None:
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = num_epochs * max(len(train_loader), 1)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    step = start_epoch * len(train_loader)
    if use_wandb and getattr(wandb.run, "resumed", False):
        step = max(step, wandb.run.step)

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        run_loss = 0.0
        n_examples = 0
        for z_i, z_f, action in train_loader:
            z_i = z_i.to(device); z_f = z_f.to(device); action = action.to(device)
            a0  = model.normalize_action(action)
            B   = a0.shape[0]
            k   = torch.randint(0, diffusion.T, (B,), device=device)
            eps = torch.randn_like(a0)
            a_k = diffusion.q_sample(a0, k, eps)

            uncond_mask = (torch.rand(B, device=device) < p_uncond)
            eps_pred = model(a_k, k, z_i, z_f, uncond_mask=uncond_mask)
            loss = F.mse_loss(eps_pred, eps)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            sched.step()

            run_loss   += loss.item() * B
            n_examples += B
            step       += 1
            if use_wandb:
                wandb.log({
                    "train/loss_step": loss.item(),
                    "train/lr":        sched.get_last_lr()[0],
                    "epoch":           epoch,
                }, step=step)
            if step % log_every == 0:
                print(f"epoch {epoch}  step {step}  "
                      f"lr {sched.get_last_lr()[0]:.2e}  "
                      f"train_loss {loss.item():.4f}")

        val = evaluate_loss(model, diffusion, val_loader, device)
        train_epoch_loss = run_loss / max(n_examples, 1)
        print(f"[epoch {epoch}] train_loss {train_epoch_loss:.4f}  val_loss {val:.4f}")
        if use_wandb:
            wandb.log({
                "train/loss_epoch": train_epoch_loss,
                "val/loss":         val,
                "epoch":            epoch,
            }, step=step)

        if sampling_eval_every > 0 and (epoch + 1) % sampling_eval_every == 0:
            samp = evaluate_sampling(model, diffusion, val_loader, device,
                                     guidance_scale=guidance_scale,
                                     n_seeds=sampling_eval_seeds)
            print(f"  -> sampling eval (n_seeds={sampling_eval_seeds}, "
                  f"w={guidance_scale}): action_mse {samp['action_mse_mean']:.4f}  "
                  f"trans_err {samp['trans_err_mean']:.4f}  "
                  f"rot_err {samp['rot_err_deg_mean']:.2f}deg")
            if use_wandb:
                wandb.log({f"val/sampling/{k}": v for k, v in samp.items()},
                          step=step)

        if ckpt_dir is not None:
            payload = {"model": model.state_dict(), "epoch": epoch,
                       "step": step, "val_loss": val}
            if val < best_val - 1e-5:
                best_val = val
                torch.save({**payload, "val_loss": best_val}, ckpt_dir / "best.pt")
                print(f"  -> new best val_loss {best_val:.4f} (saved best.pt)")
            if (epoch + 1) % ckpt_every == 0 or epoch == start_epoch + num_epochs - 1:
                ckpt_path = ckpt_dir / f"epoch_{epoch + 1:03d}.pt"
                torch.save(payload, ckpt_path)
                print(f"  -> saved {ckpt_path.name}")


# --------------------------------------------------------------------------- #
# Entry
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--root",          default="RealEstate10K")
    p.add_argument("--features_root", default="features_croco",
                   help="CroCo cache (features (N, 196, 768)). DINOv2 not supported.")
    p.add_argument("--head", choices=["mlp", "transformer"], default="mlp")

    p.add_argument("--n_actions",    type=int,   default=8,
                   help="Fixed sequence length. Variable n is out of MVP scope.")
    p.add_argument("--action_hz",    type=float, default=4.0,
                   help="Actions per second. Window covers n_actions/action_hz "
                        "seconds (default: 8 / 4 = 2.0 s).")
    p.add_argument("--window_stride_seconds", type=float, default=0.5,
                   help="Sliding-window stride in seconds. Must be a multiple "
                        "of 1/action_hz.")
    p.add_argument("--min_clip_seconds", type=float, default=2.0,
                   help="Discard clips shorter than this when computing "
                        "action stats (matches the diffusion window length).")
    p.add_argument("--num_steps",    type=int,   default=100,
                   help="DDPM diffusion steps K.")
    p.add_argument("--p_uncond",     type=float, default=0.1,
                   help="CFG dropout probability per example, per step.")
    p.add_argument("--guidance_scale", type=float, default=1.0,
                   help="CFG weight w at sampling/eval. To sweep, rerun "
                        "--eval_only with different values.")

    p.add_argument("--batch_size",   type=int, default=256)
    p.add_argument("--num_workers",  type=int, default=0)
    p.add_argument("--num_epochs",   type=int, default=1000)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=None,
                   help="Default: 1e-6 (mlp) / 1e-3 (transformer) per design doc.")
    p.add_argument("--warmup_steps", type=int, default=None,
                   help="Default: 500 (mlp) / 1000 (transformer).")
    p.add_argument("--grad_clip",    type=float, default=1.0)

    p.add_argument("--ckpt_dir",   default="checkpoints/action_diffusion")
    p.add_argument("--ckpt_every", type=int, default=50)
    p.add_argument("--init_from",  default=None,
                   help="Warm-start weights from a prior best.pt / epoch_NNN.pt.")

    p.add_argument("--eval_seeds", type=int, default=8)
    p.add_argument("--eval_only",  action="store_true",
                   help="Skip training; load --init_from then sampling-eval.")
    p.add_argument("--skip_final_eval", action="store_true",
                   help="Skip the post-training sampling eval (saves time on long runs).")
    p.add_argument("--sampling_eval_every", type=int, default=10,
                   help="Run a sampling-based eval (action MSE + integrated "
                        "pose error) every N epochs during training. 0 disables.")
    p.add_argument("--sampling_eval_seeds", type=int, default=1,
                   help="Number of seeds per per-epoch sampling eval. "
                        "Defaults to 1 for speed; --eval_seeds is the larger "
                        "count used for the post-training final eval.")

    p.add_argument("--device", default=None,
                   help="torch device. Defaults to cuda > mps > cpu.")
    p.add_argument("--wandb",         action="store_true")
    p.add_argument("--wandb_project", default="action-conditioned-diffusion")
    p.add_argument("--wandb_run",     default=None)
    p.add_argument("--wandb_run_id",  default=None,
                   help="Existing W&B run id to resume.")
    args = p.parse_args()

    if args.device is None:
        args.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

    train_ds = WindowDataset(args.root, args.features_root, split="train",
                             n_actions=args.n_actions,
                             action_hz=args.action_hz,
                             window_stride_seconds=args.window_stride_seconds,
                             deterministic=False)
    val_ds   = WindowDataset(args.root, args.features_root, split="test",
                             n_actions=args.n_actions,
                             action_hz=args.action_hz,
                             window_stride_seconds=args.window_stride_seconds,
                             deterministic=True)
    print(f"train clips: {len(train_ds):,}   test clips: {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              collate_fn=collate_windows)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True,
                              collate_fn=collate_windows)

    sample_feat = train_ds._clip_features[0][0]   # (N_tok, feat_dim)
    if sample_feat.dim() != 2:
        raise ValueError(
            f"Expected CroCo-style features with shape (N_tok, feat_dim); "
            f"got per-frame shape {tuple(sample_feat.shape)}. "
            f"DINOv2 CLS features are not supported in this MVP.")

    if args.head == "mlp":
        model = MLPActionDenoiser(n_actions=args.n_actions,
                                  feat_dim=sample_feat.shape[-1])
        wd     = 1e-6 if args.weight_decay is None else args.weight_decay
        warmup = 500  if args.warmup_steps is None else args.warmup_steps
    else:
        model = TransformerActionDenoiser(n_actions=args.n_actions,
                                          feat_dim=sample_feat.shape[-1],
                                          n_tokens_per_img=sample_feat.shape[-2])
        wd     = 1e-3 if args.weight_decay is None else args.weight_decay
        warmup = 1000 if args.warmup_steps is None else args.warmup_steps

    # Per-dim mean/std over all 0.25 s-spaced (1/action_hz) transitions in
    # the train split. We borrow CachedPairDataset because its grid pairs
    # are exactly the per-action transitions inside a diffusion window —
    # one entry per unique 0.25 s step, no per-window double-counting.
    stats_ds = CachedPairDataset(args.root, args.features_root, split="train",
                                 action_hz=args.action_hz,
                                 min_clip_seconds=args.min_clip_seconds)
    mean, std = compute_action_stats(stats_ds)
    model.set_action_stats(mean, std)
    print(f"action mean: {mean.tolist()}")
    print(f"action std:  {std.tolist()}")

    diffusion = Diffusion(num_steps=args.num_steps)

    if args.wandb:
        init_kwargs = dict(
            project=args.wandb_project,
            name=args.wandb_run if args.wandb_run_id is None else None,
            settings=wandb.Settings(init_timeout=300),
            config={
                "head":                  args.head,
                "n_actions":             args.n_actions,
                "action_hz":             args.action_hz,
                "window_stride_seconds": args.window_stride_seconds,
                "num_steps":      args.num_steps,
                "p_uncond":       args.p_uncond,
                "guidance_scale": args.guidance_scale,
                "batch_size":     args.batch_size,
                "num_epochs":     args.num_epochs,
                "lr":             args.lr,
                "weight_decay":   wd,
                "warmup_steps":   warmup,
                "device":         args.device,
                "train_clips":    len(train_ds),
                "val_clips":      len(val_ds),
                "action_mean":    mean.tolist(),
                "action_std":     std.tolist(),
            },
        )
        if args.wandb_run_id is not None:
            init_kwargs["id"] = args.wandb_run_id
            init_kwargs["resume"] = "allow"
            print(f"resuming W&B run {args.wandb_run_id} (resume=allow)")
        wandb.init(**init_kwargs)

    try:
        if not args.eval_only:
            train(model, diffusion, train_loader, val_loader,
                  num_epochs=args.num_epochs, device=args.device,
                  lr=args.lr, weight_decay=wd, p_uncond=args.p_uncond,
                  warmup_steps=warmup, grad_clip=args.grad_clip,
                  ckpt_dir=args.ckpt_dir, ckpt_every=args.ckpt_every,
                  init_from=args.init_from, use_wandb=args.wandb,
                  guidance_scale=args.guidance_scale,
                  sampling_eval_every=args.sampling_eval_every,
                  sampling_eval_seeds=args.sampling_eval_seeds)
        elif args.init_from is not None:
            ckpt = torch.load(args.init_from, map_location=args.device,
                              weights_only=False)
            model.load_state_dict(ckpt["model"])
            print(f"loaded weights from {args.init_from} for eval-only run")

        if not args.skip_final_eval:
            model = model.to(args.device); diffusion.to(args.device)
            results = evaluate_sampling(model, diffusion, val_loader,
                                        torch.device(args.device),
                                        guidance_scale=args.guidance_scale,
                                        n_seeds=args.eval_seeds)
            print("\nFinal eval (sampling):")
            for k, v in results.items():
                print(f"  {k}: {v}")
            if args.wandb:
                for kk, vv in results.items():
                    wandb.run.summary[f"eval/{kk}"] = vv
    finally:
        if args.wandb:
            wandb.finish()
