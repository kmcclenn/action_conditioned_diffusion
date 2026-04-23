# Inverse Dynamics Model

## Task

Given two consecutive RGB frames from a RealEstate10K clip, predict the 6D
SE(3) twist `a = (v, ω) ∈ R^6` describing the camera motion between them.
This is the *inverse dynamics* problem: observations → action.

Targets come from `dataset.se3_log(dataset.relative_pose(P))` — the SE(3)
log of the relative pose matrix. See
[action_extraction.md](action_extraction.md) for the math; this doc covers
the learned model.

## Architecture

Training is split into two stages: a one-time encoder pass that dumps CLS
tokens to disk (`cache_features.py`), and an MLP head that learns from those
cached tokens (`inverse_dynamics.py`). At inference, the two stages run
back-to-back on raw image pairs.

```
I_i   ──►┐                                                  (cache_features.py)
         │   frozen DINOv2           [N, 384]
         │   ViT-S/14  ─────────►  features/{split}/{clip_id}.pt
I_{i+1} ─┘
                  │
                  ▼ load cached pair                       (inverse_dynamics.py)
              z_i, z_{i+1}    [B, 384] each
                  │
                  ▼
        concat([ z_i, z_{i+1}, z_{i+1} - z_i ])   [B, 1152]
                  │
                  ▼
              Linear(1152, 512)
              LayerNorm + GELU + Dropout(0.1)
              Linear(512, 512)
              LayerNorm + GELU + Dropout(0.1)
              Linear(512, 6)
                  │
                  ▼
              (v, ω)    [B, 6]     (normalized)
```

Parameter counts (ViT-S/14 + head): ~21.7M backbone frozen, ~860K
trainable in the head.

### Encoder — frozen DINOv2 ViT-S/14 (cached out-of-loop)

- Loaded via `torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")`.
- Input: ImageNet-normalized `224×224` (divisible by patch size 14).
- Output: CLS token from `forward_features(x)["x_norm_clstoken"]` with
  dimension 384, saved to `features/{split}/{clip_id}.pt` as fp32.
- Run once via `cache_features.py`. Because the encoder is frozen and there
  is no input augmentation, the per-frame feature is deterministic — the
  cache is canonical, not an approximation.
- The model class no longer holds the encoder, so there is no need to
  override `.train()` or wrap forward in `no_grad`.

### Fusion — concat with difference

The fusion vector is `[z_i, z_{i+1}, z_{i+1} − z_i]`. The raw concatenation
gives the head full access to both views; the explicit difference makes the
motion signal a linear feature rather than something the MLP has to
discover. Dimensionality: `3 × 384 = 1152`.

### Head — 2-hidden-layer MLP

Two `Linear → LayerNorm → GELU → Dropout(0.1)` blocks, then a final
`Linear(512, 6)`. LayerNorm (rather than BatchNorm) is chosen because
batch statistics across video-pair samples are unreliable — consecutive
frames are highly correlated. Dropout is modest (0.1); with a frozen
feature extractor the head can't overfit very hard, but some regularization
helps when clip coverage is uneven.

## Target normalization

Action components live on wildly different scales — translations are
typically a few centimeters per frame, rotation axis-angle is <0.05 rad.
We normalize per-dimension:

- `action_mean`, `action_std` are registered as buffers on the model and
  persist in the state dict.
- `compute_action_stats(train_ds)` uses the fast path when the source
  exposes `all_actions()` (as `CachedPairDataset` does) — reads stored
  targets directly with no I/O — and returns `(mean, std)` of shape `(6,)`.
- Training minimizes `MSE(pred, normalize_action(target))`.
- At inference, callers apply `model.denormalize_action(...)` to map the
  network output back to physical units.

This matters for optimization (equalizing loss scales across dimensions)
and for checkpoint portability (stats travel with the weights).

## Training

| Knob              | Value                                   |
|-------------------|-----------------------------------------|
| optimizer         | Adam                                    |
| base lr           | 1e-4                                    |
| warmup            | linear over 1000 steps                  |
| post-warmup sched | constant                                |
| grad clip (norm)  | 1.0                                     |
| loss              | MSE on normalized targets               |
| batch size        | 64 (CLI default)                        |
| early stopping    | patience 5 epochs, tol 1e-5             |

The warmup protects the randomly-initialized head from large updates in
the first few hundred steps, especially with an occasional outlier action
near θ ≈ π. Gradient clipping plays the same role as a soft safety net
rather than a major regularizer.

## Data pipeline — `CachedPairDataset`

- Reuses `dataset.parse_clip` so its pose timestamps line up with the
  feature timestamps written by `cache_features.py`.
- At init, walks every clip in the split, loads its `.pt` feature file,
  and pairs consecutive frames by timestamp. All clips' features are
  concatenated into a single in-memory tensor (`self._features`); pairs
  store integer indices into that tensor plus the precomputed action.
- Drops any `(i, i+1)` pair where either timestamp is missing from the
  cache. Clips with no `.pt` file at all are skipped silently — re-run
  `cache_features.py` to fill the gap.
- `__getitem__` is two index lookups and a stack — no decoding, no I/O,
  no resize.

Why drop the on-the-fly image pipeline: the encoder is frozen and has no
augmentation, so the per-frame CLS token is fully determined by the
input frame. Recomputing it every epoch was the bottleneck. With the
cache, an epoch is dominated by the MLP forward/backward, which fits
in seconds on MPS or CUDA at the project's clip-list scale.

## Evaluation

`evaluate` returns per-dim MSE averaged over all samples on the
normalized target. Because normalization makes each dim unit-variance in
training statistics, a val MSE of 1.0 corresponds to "predicting the
mean" — anything substantially below that is learning something.

## Known limitations

- Predicts a point estimate, not a distribution. Near-static frames
  (tiny θ, tiny t) create an identifiability issue where many actions
  produce visually identical image pairs; MSE on those collapses to the
  mean. A heteroscedastic or probabilistic head would be a natural
  extension.
- Pathological at θ = π (180° rotation between adjacent frames). In
  practice this doesn't happen at 30 fps with handheld camera motion.
- No temporal context — single pair only. If sequences help, a small
  transformer over a short window of CLS tokens would be a drop-in
  replacement for the fusion step.
