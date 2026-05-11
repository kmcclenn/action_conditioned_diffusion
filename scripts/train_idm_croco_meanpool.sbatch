#!/bin/bash
#SBATCH -J idm_croco_mp
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -o logs/idm_croco_mp_%j.out
#SBATCH -e logs/idm_croco_mp_%j.err
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
# conda activate action-diffusion
conda activate ${SLURM_SUBMIT_DIR}/envs/action-diffusion


# --- wandb -----------------------------------------------------------------
export WANDB_DIR="${SLURM_SUBMIT_DIR}/wandb"
mkdir -p "${WANDB_DIR}"
if [[ -z "${WANDB_API_KEY:-}" ]] && ! grep -q "api.wandb.ai" "${HOME}/.netrc" 2>/dev/null; then
    echo "[warn] no wandb creds (env or ~/.netrc) — running wandb in offline mode."
    export WANDB_MODE=offline
fi

# --- preflight -------------------------------------------------------------
POOL_FEATURES="/home/nattang/orcd/pool/features_croco"
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

# --- optional warm-start ---------------------------------------------------
# Submit with:
#   sbatch --export=ALL,RESUME_FROM=checkpoints/idm_croco_mp_<job>/best.pt,\
#                    FORK_RUN_ID=<old_wandb_run_id>[,FORK_STEP=<step>] \
#          scripts/train_idm_croco_meanpool.sbatch
# - RESUME_FROM: path to prior checkpoint to warm-start weights from.
# - FORK_RUN_ID: (optional) old W&B run id to fork from for clean plots.
#                NOTE: do NOT use WANDB_RUN_ID — wandb reads that as the *new*
#                run's id and SLURM re-injects it past any local unset.
# - FORK_STEP:   required only when RESUME_FROM is an old checkpoint that
#                pre-dates the 'step' field. Compute as
#                (epoch + 1) * ceil(train_pairs / 64).
RESUME_ARGS=()
if [[ -n "${RESUME_FROM:-}" ]]; then
    echo "resuming from ${RESUME_FROM}"
    RESUME_ARGS+=(--init_from "${RESUME_FROM}" --warmup_steps 0)
fi
# Defensive: if the user passed WANDB_RUN_ID despite the warning, scrub it so
# wandb auto-generates a fresh id for the forked run.
unset WANDB_RUN_ID
if [[ -n "${FORK_RUN_ID:-}" ]]; then
    echo "forking W&B run ${FORK_RUN_ID}"
    RESUME_ARGS+=(--wandb_run_id "${FORK_RUN_ID}")
    if [[ -n "${FORK_STEP:-}" ]]; then
        RESUME_ARGS+=(--wandb_fork_step "${FORK_STEP}")
    fi
fi

srun env -u WANDB_RUN_ID python -u inverse_dynamics.py \
    --root          RealEstate10K \
    --features_root "${FEATURES_ROOT}" \
    --head          croco_meanpool \
    --batch_size    64 \
    --num_workers   "${SLURM_CPUS_PER_TASK}" \
    --num_epochs    100 \
    --ckpt_dir      "checkpoints/idm_croco_mp_${SLURM_JOB_ID}" \
    --ckpt_every    10 \
    "${RESUME_ARGS[@]}" \
    --wandb \
    --wandb_project action-conditioned-diffusion \
    --wandb_run     "idm_croco_mp_${SLURM_JOB_ID}"
