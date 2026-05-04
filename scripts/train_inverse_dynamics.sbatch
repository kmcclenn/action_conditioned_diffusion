#!/bin/bash
#SBATCH -J inv_dyn
#SBATCH -p mit_normal
#SBATCH -o logs/inv_dyn_%j.out
#SBATCH -e logs/inv_dyn_%j.err
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -eo pipefail
# `set -u` (nounset) breaks conda's MKL activate hooks, which reference
# MKL_INTERFACE_LAYER before assigning it. Leave nounset off.

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
# `wandb login` writes credentials to ~/.netrc; that's the preferred path on
# the cluster (no key in env / shell history). Only fall back to offline if
# neither WANDB_API_KEY nor a wandb entry in ~/.netrc is available.
if [[ -z "${WANDB_API_KEY:-}" ]] && ! grep -q "api.wandb.ai" "${HOME}/.netrc" 2>/dev/null; then
    echo "[warn] no wandb creds (env or ~/.netrc) — running wandb in offline mode."
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
    --num_epochs    100 \
    --ckpt_dir      "checkpoints/inv_dyn_${SLURM_JOB_ID}" \
    --ckpt_every    10 \
    --wandb \
    --wandb_project action-conditioned-diffusion \
    --wandb_run     "inv_dyn_${SLURM_JOB_ID}"
