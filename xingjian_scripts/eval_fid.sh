#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" || "${CONDA_PREFIX}" != "/mnt/localssd/miniconda3/envs/rae" ]]; then
  echo "[Error] Please activate the 'rae' conda env before running (current: ${CONDA_PREFIX:-none})." >&2
  exit 1
fi

# Usage: ./scripts/eval_fid.sh [SAMPLE_DIR] [STAGE2_SAMPLE_CONFIG]
SAMPLE_DIR=${1:-results/fid_eval}
STAGE2_SAMPLE_CONFIG=${2:-configs/stage2/sampling/ImageNet256/DiTDHXL-DINOv2-B.yaml}
mkdir -p "${SAMPLE_DIR}"/recon "${SAMPLE_DIR}"/gen

echo "[Info] Sampling reconstructions for FID (5k images)."
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  src/stage1_sample_ddp.py \
  --config configs/stage1/training/MAE-B_dec_sample.yaml \
  --data-path /mnt/localssd/data/imagenet/val/ \
  --num-samples 5000 \
  --sample-dir "${SAMPLE_DIR}"/recon \
  --precision bf16

echo "[Info] Sampling generated images for FID (5k images)."
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  src/sample_ddp.py \
  --config "${STAGE2_SAMPLE_CONFIG}" \
  --sample-dir "${SAMPLE_DIR}"/gen \
  --precision bf16 \
  --label-sampling equal \
  --num-samples 5000

echo "[Info] Computing reconstruction FID."
python ../guided-diffusion/evaluation/evaluator.py \
  ../guided-diffusion/evaluation/VIRTUAL_imagenet256_labeled.npz \
  "${SAMPLE_DIR}"/recon/samples.npz

echo "[Info] Computing generative FID."
python ../guided-diffusion/evaluation/evaluator.py \
  ../guided-diffusion/evaluation/VIRTUAL_imagenet256_labeled.npz \
  "${SAMPLE_DIR}"/gen/samples.npz
