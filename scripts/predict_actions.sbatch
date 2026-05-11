#!/bin/bash
#SBATCH -J predict_actions
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -o logs/predict_actions_%j.out
#SBATCH -e logs/predict_actions_%j.err
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -eo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

# --- conda env -------------------------------------------------------------
module purge
module load miniforge/24.3.0-0 2>/dev/null || module load anaconda/2023a
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate action-diffusion

# --- run info --------------------------------------------------------------
echo "node:        $(hostname)"
echo "submit dir:  ${SLURM_SUBMIT_DIR}"
echo "job id:      ${SLURM_JOB_ID}"
nvidia-smi || { echo "[error] no GPU visible"; exit 1; }

# --- knobs (override via --export=ALL,MODEL=tf,GIF_DIR=...) ----------------
MODEL="${MODEL:-mp}"         # 'mp' or 'tf'
EVERY_N="${EVERY_N:-1}"

# Input: set exactly one of VIDEO, GIF_DIR, or INTERP_DIR
VIDEO="${VIDEO:-}"
GIF_DIR="${GIF_DIR:-}"
INTERP_DIR="${INTERP_DIR:-}"

# Optional overrides
OUTPUT="${OUTPUT:-}"
IDM_CKPT="${IDM_CKPT:-}"
CROCO_CKPT="${CROCO_CKPT:-pretrained_models/CroCo.pth}"

# --- build input arg -------------------------------------------------------
if [[ -n "${VIDEO}" ]]; then
    INPUT_ARG="--video ${VIDEO}"
elif [[ -n "${GIF_DIR}" ]]; then
    INPUT_ARG="--gif_dir ${GIF_DIR}"
elif [[ -n "${INTERP_DIR}" ]]; then
    INPUT_ARG="--interp_dir ${INTERP_DIR}"
else
    echo "[error] set VIDEO, GIF_DIR, or INTERP_DIR via --export"
    exit 1
fi

EXTRA_ARGS=""
[[ -n "${OUTPUT}"   ]] && EXTRA_ARGS="${EXTRA_ARGS} --output ${OUTPUT}"
[[ -n "${IDM_CKPT}" ]] && EXTRA_ARGS="${EXTRA_ARGS} --idm_ckpt ${IDM_CKPT}"

# --- run -------------------------------------------------------------------
srun python -u scripts/predict_actions.py \
    --model      "${MODEL}" \
    --croco_ckpt "${CROCO_CKPT}" \
    --every_n    "${EVERY_N}" \
    ${INPUT_ARG} \
    ${EXTRA_ARGS}
