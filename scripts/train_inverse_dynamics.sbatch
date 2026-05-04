#!/bin/bash
#SBATCH -J inv_dyn
#SBATCH -p mit_normal
#SBATCH -o logs/inv_dyn_%j.out
#SBATCH -e logs/inv_dyn_%j.err
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

# Run from the repo root regardless of where sbatch is invoked.
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

# --- conda env -------------------------------------------------------------
# Engaging exposes miniforge / anaconda via module load. Either picks up
# the `action-diffusion` env created from environment.yml.
module purge
module load miniforge/24.3.0-0 2>/dev/null || module load anaconda/2023a
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate action-diffusion

# --- wandb -----------------------------------------------------------------
# Expect WANDB_API_KEY to be exported in your shell rc or via `sbatch --export`.
# wandb falls back to offline mode if the key is missing — runs sync later
# with `wandb sync wandb/offline-run-*`.
export WANDB_DIR="${SLURM_SUBMIT_DIR}/wandb"
mkdir -p "${WANDB_DIR}"
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "[warn] WANDB_API_KEY not set — running wandb in offline mode."
    export WANDB_MODE=offline
fi

# --- run -------------------------------------------------------------------
echo "node:        $(hostname)"
echo "submit dir:  ${SLURM_SUBMIT_DIR}"
echo "job id:      ${SLURM_JOB_ID}"
echo "cpus:        ${SLURM_CPUS_PER_TASK}"
nvidia-smi 2>/dev/null || echo "(no GPU on this partition)"

srun python -u inverse_dynamics.py \
    --root          RealEstate10K \
    --features_root features \
    --batch_size    64 \
    --num_workers   "${SLURM_CPUS_PER_TASK}" \
    --num_epochs    50 \
    --wandb \
    --wandb_project action-conditioned-diffusion \
    --wandb_run     "inv_dyn_${SLURM_JOB_ID}"
