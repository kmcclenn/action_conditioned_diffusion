#!/bin/bash
#SBATCH -J eval_action_diff
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -o logs/eval_action_diff_%j.out
#SBATCH -e logs/eval_action_diff_%j.err
#SBATCH -t 03:00:00
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

# --- preflight -------------------------------------------------------------
POOL_FEATURES="/home/kmcclenn/orcd/pool/features_croco"
if [[ ! -d "${POOL_FEATURES}" ]]; then
    echo "[error] missing CroCo feature cache at ${POOL_FEATURES}"
    exit 1
fi

# --- stage cache to node-local scratch -------------------------------------
# Eval only touches test/, but pool reads under random access are slow enough
# that staging is still a win. Stage only test/ to keep this fast.
FEATURES_ROOT="${TMPDIR:-/tmp}/features_croco"
echo "staging features (test split only) from ${POOL_FEATURES} -> ${FEATURES_ROOT}..."
mkdir -p "${FEATURES_ROOT}/test"
t_stage=$(date +%s)
rsync -a "${POOL_FEATURES}/test/" "${FEATURES_ROOT}/test/"
echo "stage done in $(( $(date +%s) - t_stage ))s.  size: $(du -sh "${FEATURES_ROOT}" | cut -f1)"

# --- run -------------------------------------------------------------------
echo "node:        $(hostname)"
echo "submit dir:  ${SLURM_SUBMIT_DIR}"
echo "job id:      ${SLURM_JOB_ID}"
echo "cpus:        ${SLURM_CPUS_PER_TASK}"
nvidia-smi || { echo "[error] no GPU visible"; exit 1; }

# --- knobs (override via --export=ALL,MLP_CKPT=...,TF_CKPT=...,N_SEEDS=...) -
MLP_CKPT="${MLP_CKPT:-checkpoints/action_diff_4hz_mlp_13508495/epoch_990.pt}"
TF_CKPT="${TF_CKPT:-checkpoints/action_diff_4hz_transformer_13508496/epoch_990.pt}"
GUIDANCE="${GUIDANCE:-0.0 0.5 1.0 2.0 3.0}"
N_SEEDS="${N_SEEDS:-16}"
OUT_DIR="${OUT_DIR:-eval/action_diffusion}"

for ck in "${MLP_CKPT}" "${TF_CKPT}"; do
    if [[ ! -f "${ck}" ]]; then
        echo "[error] missing checkpoint: ${ck}"
        exit 1
    fi
done

srun python -u scripts/eval_action_diffusion.py \
    --root            RealEstate10K \
    --features_root   "${FEATURES_ROOT}" \
    --mlp_ckpt        "${MLP_CKPT}" \
    --tf_ckpt         "${TF_CKPT}" \
    --guidance_scales ${GUIDANCE} \
    --n_seeds         "${N_SEEDS}" \
    --batch_size      256 \
    --num_workers     "${SLURM_CPUS_PER_TASK}" \
    --out_dir         "${OUT_DIR}"
