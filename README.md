# Action-Conditioned Diffusion

MIT 6.8300 project: action-conditioned diffusion over RealEstate10K video,
where actions are 6D SE(3) twists between consecutive frames.

See [`docs/`](docs/) for architecture details:
- [`docs/dataset.md`](docs/dataset.md) — dataset format and loader.
- [`docs/action_extraction.md`](docs/action_extraction.md) — how `(v, ω)` is
  derived from raw poses.
- [`docs/inverse_dynamics.md`](docs/inverse_dynamics.md) — the inverse
  dynamics model.

## Setup

```bash
conda env create -f environment.yml
conda activate action-diffusion
```

Download [RealEstate10K](https://google.github.io/realestate10k/download.html)
and place it at the repo root:

```
RealEstate10K/
  train/*.txt
  test/*.txt
```

## Pipeline

```bash
# 1. Materialize frames (resumable; auto-filters to clip_lists/train.txt).
python download_frames.py RealEstate10K/ frames/ --split train --workers 8 --img_size 256x256
python download_frames.py RealEstate10K/ frames/ --split test  --workers 8 --img_size 256x256

# 2. Cache encoder features once (resumable). Pick one:
#    DINOv2 CLS (default):
python cache_features.py --frames_root frames --features_root features
#    CroCo v2 patch tokens (requires third_party/croco + a CroCo v2 .pth):
python cache_features.py --encoder croco --croco_ckpt path/to/CroCo_V2.pth \
  --frames_root frames --features_root features_croco

# 3. Train the inverse dynamics head.
python inverse_dynamics.py \
  --root RealEstate10K \
  --features_root features \
  --head dino_mlp \
  --num_epochs 100 \
  --ckpt_dir checkpoints/inverse_dynamics \
  --ckpt_every 10
```

`--head` selects which IDM to train against the cached features:
- `dino_mlp` (default) — DINOv2 CLS + MLP. Use with `--features_root features`.
- `croco_meanpool` — CroCo patch tokens, mean-pooled, then MLP.
- `croco_transformer` — self-attention over `[CLS, f1, f2]` CroCo tokens.

The two CroCo heads expect `--features_root features_croco`.

Add `--wandb` to log to Weights & Biases (run `wandb login` once first).
Checkpoints: `best.pt` (best val loss) plus `epoch_NNN.pt` every
`--ckpt_every` epochs (default 1). Each checkpoint stores `model`, `epoch`,
`step`, and `val_loss`.

## Train on MIT Engaging

```bash
ssh <kerb>@orcd-login.mit.edu.mit.edu
cd /path/to/action_conditioned_diffusion

# One-time
module load miniforge/24.3.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env create -f environment.yml
conda activate action-diffusion
wandb login

# Submit (must be from repo root)
sbatch scripts/train_inverse_dynamics.sbatch        # DINOv2 + MLP
sbatch scripts/train_idm_croco_meanpool.sbatch      # CroCo + mean-pool + MLP
sbatch scripts/train_idm_croco_transformer.sbatch   # CroCo + self-attention
squeue -u $USER
tail -f logs/inv_dyn_<jobid>.out                    # or idm_croco_{mp,tf}_<jobid>.out
```

The DINOv2 script defaults to `mit_normal` (CPU); the two CroCo scripts
default to `mit_normal_gpu` and stage `features_croco/` from pool storage to
node-local NVMe before training. Each writes to
`checkpoints/{inv_dyn,idm_croco_mp,idm_croco_tf}_<jobid>/`.

`RealEstate10K/` and the appropriate features dir (`features/` for DINOv2,
`features_croco/` for CroCo) must exist in the working directory — `rsync`
from your laptop or rerun the cache step on the cluster first. The CroCo
scripts expect `features_croco/` at `~/orcd/pool/features_croco`; submit
`scripts/cache_features_croco.sbatch` once if it isn't there.

### Resuming a run

Both CroCo sbatch scripts accept env vars to warm-start from a prior
checkpoint and continue the original W&B run:

```bash
sbatch --export=ALL,\
RESUME_FROM=checkpoints/idm_croco_mp_<oldjob>/best.pt,\
FORK_RUN_ID=<old_wandb_run_id> \
  scripts/train_idm_croco_meanpool.sbatch
```

`RESUME_FROM` loads weights and skips LR warmup. `FORK_RUN_ID` re-attaches to
the existing W&B run via `resume="allow"` so plots vs `_step` continue
cleanly. If `best.pt` was saved before the `step` field was added, also pass
`FORK_STEP=<step>` (compute as `(epoch + 1) * ceil(train_pairs / batch_size)`).
