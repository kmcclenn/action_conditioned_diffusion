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

```
I_i   ──►┐
         │   frozen DINOv2           [B, 384]
         │   ViT-S/14  ─────────►  z_i
I_{i+1} ─┘                         z_{i+1}
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

### Encoder — frozen DINOv2 ViT-S/14

- Loaded via `torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")`.
- Input: ImageNet-normalized `224×224` (divisible by patch size 14).
- Output: CLS token from `forward_features(x)["x_norm_clstoken"]` with
  dimension 384.
- Kept in `eval()` mode and wrapped in `torch.no_grad()` during encoding —
  no gradient graph is built through the backbone, which cuts activation
  memory roughly in half.
- `InverseDynamicsModel.train()` is overridden to recursively set the
  wrapper's mode but force `self.encoder.eval()`. Without this, a
  top-level `model.train()` would flip BatchNorm / dropout submodules of
  the encoder into train mode.

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
- `compute_action_stats(train_ds)` uses the fast path on `FramePairDataset`
  (reads `self._pairs` without loading images) and returns `(mean, std)`
  of shape `(6,)`.
- Training minimizes `MSE(pred, normalize_action(target))`.
- At inference, `model.predict(...)` applies `denormalize_action` so
  callers see physical units.

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

## Data pipeline — `FramePairDataset`

- Reuses `dataset.parse_clip` so the timestamps match what
  `download_frames.py` writes.
- Precomputes per-clip `se3_log(relative_pose(P))` up front — one pass
  over the poses at init, then `__getitem__` only does image I/O.
- Drops any `(i, i+1)` pair whose two JPEGs aren't both present on disk.
  This mirrors `frames_only=True` in the sister dataset.
- Default `img_size=(224, 224)` matches DINOv2's expected input so
  `preprocess_pair` is a near-no-op.

The split between dataset (returns raw `[0, 1]` tensors) and
`preprocess_pair` (does resize + ImageNet normalize) is deliberate: it
lets you drop pair-consistent augmentations (same crop, same flip) in
between.

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
