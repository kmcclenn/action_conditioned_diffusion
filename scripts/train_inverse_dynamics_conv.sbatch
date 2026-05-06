#!/bin/bash
#SBATCH -J inv_dyn_conv
#SBATCH -p mit_normal_gpu
#SBATCH -o logs/inv_dyn_conv_%j.out
#SBATCH -e logs/inv_dyn_conv_%j.err
#SBATCH -t 6:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1


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
conda activate ${SLURM_SUBMIT_DIR}/envs/action-diffusion
pip install wandb

echo "Conda environment: $(conda env list | grep '*' || echo '(none)')"
# --- wandb -----------------------------------------------------------------
# Expect WANDB_API_KEY to be exported in your shell rc or via `sbatch --export`.
# wandb falls back to offline mode if the key is missing — runs sync later
# with `wandb sync wandb/offline-run-*`.
export WANDB_DIR="${SLURM_SUBMIT_DIR}/wandb_logs"
mkdir -p "${WANDB_DIR}"
# `wandb login` writes credentials to ~/.netrc; that's the preferred path on
# the cluster (no key in env / shell history). Only fall back to offline if
# neither WANDB_API_KEY nor a wandb entry in ~/.netrc is available.
if [[ -z "${WANDB_API_KEY:-}" ]] && ! grep -q "api.wandb.ai" "${HOME}/.netrc" 2>/dev/null; then
    echo "[warn] no wandb creds (env or ~/.netrc) — running wandb in offline mode."
    export WANDB_MODE=offline
fi

# Stage data to local scratch
echo "Staging data to local scratch..."
SCRATCH="/scratch/${USER}/${SLURM_JOB_ID}"
mkdir -p "${SCRATCH}"

# Copy frames (takes ~1 hour)
time rsync -av --delete frames/ "${SCRATCH}/frames/" || exit 1
time rsync -av --delete RealEstate10K/ "${SCRATCH}/RealEstate10K/" || exit 1

echo "Data staged to ${SCRATCH}"
echo "Local disk speed:"
time dd if=/dev/zero of="${SCRATCH}/test.tmp" bs=1M count=1000

# --- run -------------------------------------------------------------------
echo "node:        $(hostname)"
echo "submit dir:  ${SLURM_SUBMIT_DIR}"
echo "job id:      ${SLURM_JOB_ID}"
echo "cpus:        ${SLURM_CPUS_PER_TASK}"
nvidia-smi 2>/dev/null || echo "(no GPU on this partition)"

srun python -u inverse_dynamics_conv.py \
    --root              "${SCRATCH}/RealEstate10K" \
    --frames_root       "${SCRATCH}/frames" \
    --batch_size        64 \
    --num_workers       2 \
    --num_epochs        50 \
    --img_size          224 \
    --hidden            64 \
    --device            cuda \
    --mixed_precision \
    --wandb_project     action-conditioned-diffusion \
    --wandb_run         "inv_dyn_conv_${SLURM_JOB_ID}"


