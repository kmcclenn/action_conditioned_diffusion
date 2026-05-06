#!/bin/bash
#SBATCH -J inv_dyn_conv_resume
#SBATCH -p mit_normal_gpu
#SBATCH -o logs/inv_dyn_conv_resume_%j.out
#SBATCH -e logs/inv_dyn_conv_resume_%j.err
#SBATCH -t 2:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -eo pipefail
# `set -u` (nounset) breaks conda's MKL activate hooks, which reference
# MKL_INTERFACE_LAYER before assigning it. Leave nounset off.

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

module purge
module load miniforge/24.3.0-0 2>/dev/null || module load anaconda/2023a
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/nattang/miniconda3/envs/action-diffusion

# Optional: set WANDB_MODE=offline if you do not have wandb installed.
export WANDB_DIR="${SLURM_SUBMIT_DIR}/wandb"
mkdir -p "${WANDB_DIR}"

if [[ -z "${WANDB_API_KEY:-}" ]] && ! grep -q "api.wandb.ai" "${HOME}/.netrc" 2>/dev/null; then
    echo "[warn] no wandb creds detected; W&B logging will be skipped if wandb is unavailable."
    export WANDB_MODE=offline
fi

echo "node:        $(hostname)"
echo "submit dir:  ${SLURM_SUBMIT_DIR}"
echo "job id:      ${SLURM_JOB_ID}"
echo "cpus:        ${SLURM_CPUS_PER_TASK}"
nvidia-smi 2>/dev/null || echo "(no GPU on this partition)"

srun python -u inverse_dynamics_conv.py \
    --root              RealEstate10K \
    --frames_root       frames \
    --batch_size        128 \
    --num_workers       "${SLURM_CPUS_PER_TASK}" \
    --num_epochs        100 \
    --img_size          224 \
    --hidden            64 \
    --device            cuda \
    --resume_from       checkpoints/best.pt \
    --mixed_precision \
    --wandb_project     action-conditioned-diffusion \
    --wandb_run         "inv_dyn_conv_resume_${SLURM_JOB_ID}"
