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

# 2. Cache DINOv2 CLS features once (resumable).
python cache_features.py --frames_root frames --features_root features

# 3. Train the inverse dynamics head.
python inverse_dynamics.py \
  --root RealEstate10K \
  --features_root features \
  --num_epochs 100 \
  --ckpt_dir checkpoints/inverse_dynamics \
  --ckpt_every 10
```

Add `--wandb` to log to Weights & Biases (run `wandb login` once first).
Checkpoints: `best.pt` (best val loss) plus `epoch_NNN.pt` every
`--ckpt_every` epochs.

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
sbatch scripts/train_inverse_dynamics.sbatch
squeue -u $USER
tail -f logs/inv_dyn_<jobid>.out
```

Defaults: `mit_normal` partition, 8 CPUs, 32 GB, 12 h, no GPU. Output goes to
`checkpoints/inv_dyn_<jobid>/` and `logs/inv_dyn_<jobid>.{out,err}`. For GPU,
swap `-p mit_normal` for a GPU partition like `-p mit_normal_gpu` and add `#SBATCH --gres=gpu:1`.

`RealEstate10K/` and `features/` must already exist in the working directory
— `rsync` from your laptop or rerun steps 1–2 on the login node first.
