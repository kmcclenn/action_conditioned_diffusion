#!/bin/bash
#SBATCH -J action_diff
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -o logs/action_diff_%j.out
#SBATCH -e logs/action_diff_%j.err
#SBATCH -t 12:00:00
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
# Pool storage gives ~130 MB/s under random reads; node-local NVMe is ~10x
# faster for the random-access patterns the dataset's mmap path produces.
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

# --- dispatch knobs --------------------------------------------------------
# HEAD: mlp (default) | transformer
HEAD="${HEAD:-mlp}"

# --- optional warm-start ---------------------------------------------------
# sbatch --export=ALL,HEAD=transformer,RESUME_FROM=...,FORK_RUN_ID=... \
#        scripts/train_action_diffusion.sbatch
RESUME_ARGS=()
if [[ -n "${RESUME_FROM:-}" ]]; then
    echo "resuming from ${RESUME_FROM}"
    RESUME_ARGS+=(--init_from "${RESUME_FROM}")
fi
unset WANDB_RUN_ID
if [[ -n "${FORK_RUN_ID:-}" ]]; then
    echo "forking W&B run ${FORK_RUN_ID}"
    RESUME_ARGS+=(--wandb_run_id "${FORK_RUN_ID}")
fi

srun env -u WANDB_RUN_ID python -u action_diffusion.py \
    --root          RealEstate10K \
    --features_root "${FEATURES_ROOT}" \
    --head          "${HEAD}" \
    --batch_size    256 \
    --num_workers   "${SLURM_CPUS_PER_TASK}" \
    --num_epochs    1000 \
    --ckpt_dir      "checkpoints/action_diff_${HEAD}_${SLURM_JOB_ID}" \
    --ckpt_every    50 \
    "${RESUME_ARGS[@]}" \
    --wandb \
    --wandb_project action-conditioned-diffusion \
    --wandb_run     "action_diff_${HEAD}_${SLURM_JOB_ID}"
