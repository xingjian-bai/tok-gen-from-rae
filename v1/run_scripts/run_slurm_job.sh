#!/bin/bash

#SBATCH -p vision-torralba
#SBATCH -q vision-torralba-main
#SBATCH --account=vision-torralba
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=30GB
#SBATCH --gres=gpu:8
#SBATCH --job-name=no_quant_no_post_trunk_norm_with_dit_final_layer_learned_sigma_ip_sample_128_tokens
#SBATCH --output=/data/vision/torralba/selfmanaged/torralba/projects/sduggal/research/tok_gen/v1/run_scripts/slurm_logs/%x_%j.out
#SBATCH --error=/data/vision/torralba/selfmanaged/torralba/projects/sduggal/research/tok_gen/v1/run_scripts/slurm_logs/%x_%j.err
#SBATCH --nodelist=torralba-h100-1

source /data/vision/torralba/sduggal/.bash_profile
mamba activate DiT
echo "Using Python from: $(which python)"
cd /data/vision/torralba/selfmanaged/torralba/projects/sduggal/research/tok_gen/v1/
export TRAIN_DATA_DIR="/data/vision/torralba/datasets/imagenet100"
bash run_scripts/latent_distillation_pretrain.sh