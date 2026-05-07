# Action Diffusion for Real-Estate Video — Design Doc (MVP)

## Goal

Train a conditional diffusion model that, given a start image and a finish image (each represented by a precomputed CrocoV2 embedding), generates the sequence of 6DoF ego-motion actions that takes the actor from the start view to the finish view.

Formally:

```
p(a_1, ..., a_n | z_i, z_f)
```

where `a_t ∈ ℝ^6` is a twist in 𝔰𝔢(3) (linear `v ∈ ℝ³` + angular `ω ∈ ℝ³`, axis-angle), and `z_i, z_f ∈ ℝ^{196 × 768}` are CrocoV2 patch token embeddings.

This is loosely based on Diffusion Policy (Chi et al. 2023), with two key simplifications:

- **No receding-horizon loop.** We diffuse the *entire* action sequence in a single shot, conditioned only on the start and finish embeddings. There is no re-observation step.
- **No per-timestep visual observation.** The conditioning is a fixed pair `(z_i, z_f)`, not a stream of `O_t`.

## Scope (MVP)

In scope:

- Vanilla DDPM training and sampling.
- Two head architectures: an **MLP head** (mean-pools tokens to a single vector per image) and a **Transformer head** (cross-attends over CrocoV2 tokens).
- Classifier-free guidance (CFG) for conditioning.
- Per-dimension z-score normalization of actions.
- Train/eval loop with wandb logging.
- Two evaluation metrics: per-step action MSE and integrated final-pose error.

Explicitly out of scope for v1 (revisit later):

- DDIM / accelerated sampling (start with vanilla DDPM).
- Square-cosine / iDDPM noise schedules (start with the standard linear or cosine schedule from `diffusers`, whichever is the library default).
- Variable sequence lengths (`n` is fixed at 8).
- CrocoV2 finetuning (frozen, embeddings already cached).
- EMA on model weights, exotic regularization, augmentation.
- 1D temporal CNN head — skipped because the conditioning is a fixed pair, not a stream, so the CNN's locality bias buys us less.

## Data

**Cached CrocoV2 embeddings** live at `/home/kmcclenn/orcd/pool/features_croco`. Each image embedding has shape `(196, 768)` (14×14 patches, ViT-B-ish width). Embeddings are frozen and precomputed.

**Dataset size:** 2957 training pairs. A train/test split is already defined in the repo — reuse it.

**Action sequences:**

- Fixed length `n = 8`.
- Each action is a 6D vector `(v_x, v_y, v_z, ω_x, ω_y, ω_z)`, where `ω` is axis-angle.
- Actions are *relative* twists: `T_{t+1} = T_t · exp(â_t)`.

**Normalization:**

- Compute per-dimension mean and std over the training set, separately for each of the 6 action dimensions. Use the same normalization as in the inverse dynamics model.


## Model

### Shared diffusion backbone

A standard conditional DDPM over the action sequence `a ∈ ℝ^{n × 6}` (n=8).

- **Forward process:** `a^k = √(ᾱ_k) · a^0 + √(1 - ᾱ_k) · ε`, with `ε ∼ N(0, I)`.
- **Noise schedule:** Default to whatever the chosen library (e.g., HuggingFace `diffusers.DDPMScheduler`) ships with — cosine or scaled linear is fine. Make this a config arg.
- **Training objective:** ε-prediction MSE,

  ```
  L = E[ ||ε - ε_θ(a^k, k, z_i, z_f)||² ]
  ```

- **Diffusion steps:** `K = 100` for both training and sampling in the MVP.
- **Time embedding:** Sinusoidal embedding of `k`, projected through a small MLP to the model's hidden dimension. Add to action tokens (transformer head) or concatenate to the input (MLP head).

### Head A — MLP head (default starting point)

Conditioning pathway:

1. Mean-pool each CrocoV2 token sequence over the patch dimension: `z_i, z_f ∈ ℝ^{196 × 768} → ℝ^{768}`.
2. Concatenate: `c = [z_i_pool ; z_f_pool] ∈ ℝ^{1536}`.
3. Project `c` through a 2-layer MLP to a conditioning vector of size `d_cond` (e.g., 512).

Denoising network:

1. Flatten the noised action sequence: `a^k ∈ ℝ^{n × 6} → ℝ^{n·6} = ℝ^{48}`.
2. Concatenate with the conditioning vector and the time embedding.
3. Pass through a stack of residual MLP blocks (e.g., 4–6 blocks, hidden dim 512, GELU, LayerNorm).
4. Project back to `ℝ^{n·6}`, reshape to `(n, 6)` — this is the predicted noise `ε_θ`.

This head is fast, simple, and a strong sanity baseline.

### Head B — Transformer head

Conditioning pathway:

1. Keep CrocoV2 tokens unpooled. Concatenate the two token sequences along the sequence dimension to form a memory of shape `(2·196, 768) = (392, 768)`.
2. Optionally add a learned "image-of-origin" embedding (start vs finish) so the model can distinguish them — this is important and cheap.
3. Project to the head's hidden dim if different from 768.

Denoising network (transformer decoder, following the paper's transformer head):

1. Treat each noised action `a^k_t ∈ ℝ^6` as one token; project to the hidden dim with a linear layer. Result: `n` action tokens.
2. Add a positional embedding over the `n` action positions.
3. Prepend or add the time embedding (sinusoidal-then-MLP) as a conditioning token / FiLM-style modulation. Pick one and stick with it — the simpler choice is to add a time-embedding vector to every action token before the first block.
4. Stack `L` decoder blocks (e.g., L=6, 8 heads, hidden dim 384 or 512). Each block has:
   - Self-attention over action tokens (causal mask is **not** required here — we're predicting the whole sequence at once, so use full self-attention. The paper uses a causal mask but their formulation is for streaming/receding-horizon use; ours isn't. Default to full self-attention. Worth flipping to causal as an ablation later.)
   - Cross-attention from action tokens to the image memory (the 392 image tokens).
   - Feed-forward.
5. Final linear projection from hidden dim back to 6 — predicted noise per timestep.

### Classifier-free guidance (CFG)

During training, with probability `p_uncond = 0.1`, replace `(z_i, z_f)` with a learned null embedding (one per slot, or a single shared null — pick one; one shared null is simpler).

For the MLP head, the null is a learned vector of shape `(2 × 768,)` substituted in place of the concatenated pooled embeddings.

For the transformer head, the null is a learned tensor of shape `(2·196, 768)` — same shape as the image memory — used in place of it. (Cheaper alternative: a single learned token broadcast across the sequence. Start with the broadcast version.)

At sampling time, with guidance scale `w`:

```
ε̂ = (1 + w) · ε_θ(a^k, k, c) - w · ε_θ(a^k, k, ∅)
```

Default `w = 1.0` (mild guidance). Sweep `w ∈ {0.0, 0.5, 1.0, 2.0, 3.0}` as an eval ablation.

## Training

- **Optimizer:** AdamW, lr `1e-4`, weight decay `1e-6` for MLP head, `1e-3` for transformer head (matches the paper's split).
- **LR schedule:** Cosine decay with linear warmup (500 steps for MLP, 1000 for transformer).
- **Batch size:** 256 (small enough that 2957 examples / batch = ~12 batches per epoch; tune down if memory is tight).
- **Epochs:** Start with 1000 epochs, save checkpoint every 50 epochs.
- **Gradient clipping:** norm 1.0.
- **Mixed precision:** bf16 if supported, otherwise fp32.

CFG dropout (`p_uncond = 0.1`) is applied per-example, per-batch.

## Sampling

Vanilla DDPM ancestral sampling, K=100 steps. At each step, run the model twice (conditional + unconditional) and combine via the CFG formula above. With small action dim (`n × 6 = 48`) this is cheap.

Output is a tensor of shape `(batch, 8, 6)` in normalized space. Denormalize before computing pose errors.

## Evaluation

Compute on the held-out test split. Two metrics:

1. **Action-sequence MSE.** Per-dimension MSE between the predicted (denormalized) action sequence and the ground-truth sequence, averaged over the test set. Report overall and split by linear / angular components.

2. **Integrated final-pose error.** Compose the predicted twists via the SE(3) exponential map starting from the identity:

   ```
   T_pred = ∏_{t=1}^{n} exp(â_t_pred)
   T_gt   = ∏_{t=1}^{n} exp(â_t_gt)
   ```

   Report:
   - Translation error: `||t_pred - t_gt||₂` (meters or whatever the dataset's unit is).
   - Rotation error: angle of `R_pred · R_gt^T` (degrees).

   Implement the SE(3) exp map as a small utility. Don't introduce a new dependency for this — `scipy.spatial.transform.Rotation` plus a hand-rolled translation composition is fine, or roll the matrix exponential by hand for the 4×4 form. Most cleanly: write `exp_se3(twist: Tensor[6]) -> Tensor[4, 4]` once and reuse it.

For diffusion sampling, evaluation is stochastic. Report metrics averaged over **8 sampling seeds per test example**, plus the standard deviation across seeds. This lets us see whether multimodality is helping or whether the model is just collapsing to the mean.

## Logging (wandb)

Log per training step: train loss.
Log per epoch: validation action MSE (cheap, no sampling — just one denoising step on noised val data, like the training loss).
Final evaluation: sweep over CFG scales, report all metrics.


