#!/bin/bash
#SBATCH -J idm_croco_tf
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -o logs/idm_croco_tf_%j.out
#SBATCH -e logs/idm_croco_tf_%j.err
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -eo pipefail
# `set -u` (nounset) breaks conda's MKL activate hooks, which reference
# MKL_INTERFACE_LAYER before assigning it. Leave nounset off.

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

# --- conda env -------------------------------------------------------------
module purge
module load miniforge/24.3.0-0 2>/dev/null || module load anaconda/2023a
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate action-diffusion

# --- wandb -----------------------------------------------------------------
export WANDB_DIR="${SLURM_SUBMIT_DIR}/wandb"
mkdir -p "${WANDB_DIR}"
if [[ -z "${WANDB_API_KEY:-}" ]] && ! grep -q "api.wandb.ai" "${HOME}/.netrc" 2>/dev/null; then
    echo "[warn] no wandb creds (env or ~/.netrc) — running wandb in offline mode."
    export WANDB_MODE=offline
fi

# --- preflight -------------------------------------------------------------
POOL_FEATURES="/home/kmcclenn/orcd/pool/features_croco"
if [[ ! -d "${POOL_FEATURES}" ]]; then
    echo "[error] missing CroCo feature cache at ${POOL_FEATURES}"
    echo "        run scripts/cache_features_croco.sbatch first."
    exit 1
fi

# --- stage cache to node-local scratch -------------------------------------
# Pool storage gives ~130 MB/s under random reads, so an epoch over the full
# cache takes ~1 h. Node-local scratch (NVMe, microsecond latency) drops that
# to a few minutes after a one-time bulk copy.
FEATURES_ROOT="${TMPDIR:-/tmp}/features_croco"
echo "staging features from ${POOL_FEATURES} -> ${FEATURES_ROOT}..."
mkdir -p "${FEATURES_ROOT}"
t_stage=$(date +%s)
rsync -a "${POOL_FEATURES}/" "${FEATURES_ROOT}/"
echo "stage done in $(( $(date +%s) - t_stage ))s.  size: $(du -sh "${FEATURES_ROOT}" | cut -f1)"

# --- run -------------------------------------------------------------------
echo "node:        $(hostname)"
echo "submit dir:  ${SLURM_SUBMIT_DIR}"
echo "job id:      ${SLURM_JOB_ID}"
echo "cpus:        ${SLURM_CPUS_PER_TASK}"
nvidia-smi || { echo "[error] no GPU visible"; exit 1; }

# Smaller batch than the mean-pool head: transformer attends over
# (1 + 2*196) = 393 tokens at dim 768, so per-sample memory is ~30x higher.
srun python -u inverse_dynamics.py \
    --root          RealEstate10K \
    --features_root "${FEATURES_ROOT}" \
    --head          croco_transformer \
    --batch_size    32 \
    --num_workers   "${SLURM_CPUS_PER_TASK}" \
    --num_epochs    100 \
    --ckpt_dir      "checkpoints/idm_croco_tf_${SLURM_JOB_ID}" \
    --ckpt_every    10 \
    --wandb \
    --wandb_project action-conditioned-diffusion \
    --wandb_run     "idm_croco_tf_${SLURM_JOB_ID}"
