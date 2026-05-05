#!/bin/bash
#SBATCH -J croco_cache
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -o logs/croco_cache_%j.out
#SBATCH -e logs/croco_cache_%j.err
#SBATCH -t 04:00:00
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
CKPT="checkpoints/croco/CroCo_V2_ViTBase_SmallDecoder.pth"
if [[ ! -f "${CKPT}" ]]; then
    echo "[error] missing CroCo checkpoint at ${CKPT}"
    echo "        download with:"
    echo "          mkdir -p checkpoints/croco"
    echo "          wget -P checkpoints/croco https://download.europe.naverlabs.com/ComputerVision/CroCo/CroCo_V2_ViTBase_SmallDecoder.pth"
    exit 1
fi

if [[ ! -d "third_party/croco" ]]; then
    echo "[error] CroCo submodule missing. Run: git submodule update --init --recursive"
    exit 1
fi

# --- run -------------------------------------------------------------------
echo "node:        $(hostname)"
echo "submit dir:  ${SLURM_SUBMIT_DIR}"
echo "job id:      ${SLURM_JOB_ID}"
echo "cpus:        ${SLURM_CPUS_PER_TASK}"
nvidia-smi || { echo "[error] no GPU visible"; exit 1; }

srun python -u cache_features.py \
    --encoder       croco \
    --croco_ckpt    "${CKPT}" \
    --frames_root   frames \
    --features_root features_croco \
    --batch_size    64
