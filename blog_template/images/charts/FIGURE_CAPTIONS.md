# Figure captions

Drop-in descriptions for the charts in `evals/charts/`. All charts are produced
by `evals/make_charts.py`.

Shared conventions used throughout:
- **Action diffusion** model = DDPM over the 6D SE(3) twist sequence between
  consecutive frames, conditioned on DINOv2/CroCo image features. Two denoiser
  heads are compared: an **MLP head** and a **Transformer head**.
- **Open conditioning** = the diffusion model sees only the *start* frame.
  **Closed conditioning** = it sees *both* the start and end frames.
- **IDM baseline** = the supervised inverse-dynamics model (frame pair → twist),
  applied frame-by-frame. The "interp" variant integrates the IDM's
  consecutive-frame predictions; the plain variant predicts at the evaluation
  frame rate directly. For the head-to-head comparisons the IDM-standard pairs
  with open conditioning and the IDM-interp pairs with closed conditioning.
- `w` = classifier-free-guidance scale; `x0_clip` = the clamp applied to the
  predicted clean sample `x_0` during sampling.
- Window lengths 2 s / 4 s / 8 s correspond to 8 / 16 / 32 sampled frames at the
  evaluation frame rate. Action MSE is the per-element MSE on the denormalized
  6D twists; "final rotation/translation error" is the rotation angle (deg) /
  translation L2 (m) of the integrated start→finish pose. Test-clip counts vary
  with window length (≈250 / 142 / 54 valid clips for 2 / 4 / 8 s).

---

### `sweep_action_mse.png` / `sweep_rot_err.png` / `sweep_trans_err.png`

**Caption.** Effect of the sampling clamp `x0_clip` on the action-diffusion model.
Each panel is one (conditioning, window) configuration — rows: open vs. closed
conditioning; columns: 2 s / 4 s / 8 s windows. Within a panel, the six curves
cover the two denoiser heads (orange = MLP, teal = Transformer) crossed with
three guidance scales (solid `w=0`, dashed `w=1`, dotted `w=2`); the horizontal
axis is `x0_clip` and the vertical axis is, respectively, action MSE, final
rotation error (deg), and final translation error (m). Lower is better.
Tighter clamping (`x0_clip≈1`) and no guidance (`w=0`) are consistently best;
larger clamps and larger guidance scales inflate error, and the two heads track
each other closely.

### `per_dim_mse.png`

**Caption.** Per-component action MSE of the diffusion model at the best
sampling setting (`w=0`, `x0_clip=1`). Layout matches the sweep figures (rows:
open/closed conditioning; columns: 2 s / 4 s / 8 s). Within each panel the six
groups are the twist components — linear velocity (vx, vy, vz) and angular
velocity (ωx, ωy, ωz) — with MLP (orange) and Transformer (teal) bars side by
side on a log scale. Error is dominated by the forward/backward translation
component vz and is roughly an order of magnitude smaller on the rotation
components; the two heads are nearly indistinguishable per component.

### `compare_action_mse.png` / `compare_rot_err.png` / `compare_trans_err.png`

**Caption.** Action-diffusion heads vs. the IDM baseline heads, per task
(`w=0`, `x0_clip=1`). Rows: open conditioning (top) and closed conditioning
(bottom); columns: 2 s / 4 s / 8 s windows. Each panel shows four bars —
Diff-MLP, Diff-Transformer, and the two matching IDM baseline heads
(IDM-{MLP,Transformer} for the open row, IDM-{MLP,Transformer} (interp) for the
closed row) — with the metric value annotated above each bar. Metrics are
action MSE, final rotation error (deg), and final translation error (m); lower
is better. At 2 s the supervised IDM has the edge on integrated-pose error,
while at 4–8 s the gap closes and the methods are comparable; the MLP and
Transformer heads perform similarly within each model.

### `rot_err_bars.png`

**Caption.** Final rotation error vs. trajectory length. Left panel: open
conditioning (diffusion sees only the start frame) with the IDM-standard
baseline; right panel: closed conditioning (diffusion sees both frames) with the
IDM-interp baseline. Within each panel, grouped bars give Diff-MLP,
Diff-Transformer and the two IDM heads at 2 s / 4 s / 8 s windows; the diffusion
model uses `w=0`, `x0_clip=1`. Rotation error grows with horizon for every
model; the supervised IDM is strongest at short horizons and the methods
converge by 8 s (where the 8 s differences are within test-set noise, n≈54).

---

*Note for the 8 s column / right end of `rot_err_bars`: only ~54 test clips are
long enough, and diffusion samples are stochastic, so small differences there
(e.g. open vs. closed rotation error) are not significant.*
