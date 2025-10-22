export WANDB_KEY="7a6fb11753fddfe239fd51545c1664ab843f9e5d"
export WANDB_ENTITY="xingjianb"
export WANDB_PROJECT="rae-stage1-mae"   # or any name you pick
WANDB_BASE_URL=https://adobesensei.wandb.io wandb login
export WANDB_BASE_URL=https://adobesensei.wandb.io

conda activate /mnt/localssd/miniconda3/envs/rae
cd /mnt/localssd/DiffAE/RAE

# stage 1
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 \
  src/train_stage1.py \
  --config configs/stage1/training/MAE-B_dec.yaml \
  --data-path /mnt/localssd/data/imagenet100/train/ \
  --results-dir results/stage1_mae \
  --batch-size 16 \
  --image-size 256 \
  --precision bf16 \
  --log-interval 10 \
  --image-log-interval 500 \
  --save-interval 1000 \
  --run-name stage1-mae-UNFREEZE \
  --wandb

# stage 3
# CUDA_VISIBLE_DEVICES=4,5,6,7 \
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  src/train_unified_v2.py \
  --stage1-config configs/stage1/training/MAE-B_dec_sample.yaml \
  --stage2-config configs/stage2/training/ImageNet256/DiTDH-S_MAE-B.yaml \
  --data-path /mnt/localssd/data/imagenet100/train/ \
  --output-dir results/unified_phase3_first_run \
  --batch-size 8 \
  --epochs 20 \
  --log-interval 10 \
  --image-log-interval 500 \
  --save-interval 1000 \
  --wandb \
  --run-name phase3-mae-diffusion \
  --resume

python run_scripts/create_imagenet100.py --imagenet_dir /mnt/localssd/data/imagenet/ --imagenet100_dir /mnt/localssd/data/imagenet100/

# continue 
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  src/train_unified_v2.py \
  --stage1-config configs/stage1/training/MAE-B_dec_sample.yaml \
  --stage2-config configs/stage2/training/ImageNet256/DiTDH-S_MAE-B.yaml \
  --data-path /mnt/localssd/data/imagenet100/train/ \
  --output-dir results/unified_phase3_first_run \
  --batch-size 8 \
  --epochs 20 \
  --log-interval 10 \
  --image-log-interval 500 \
  --save-interval 1000 \
  --wandb \
  --run-name phase3-mae-diffusion \
  --resume


torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  src/train_unified_v2.py \
  --stage1-config configs/stage1/training/MAE-B_dec_sample.yaml \
  --stage2-config configs/stage2/training/ImageNet256/DiTDH-S_MAE-B.yaml \
  --data-path /mnt/localssd/data/imagenet100/train/ \
  --output-dir results/unified_phase3_first_run \
  --epochs 100 \
  --batch-size 16 \
  --workers 8 \
  --lr-encoder 1e-4 \
  --lr-decoder 1e-4 \
  --lr-diffusion 2e-4 \
  --recon-weight 1.0 \
  --lpips-weight 1.0 \
  --diffusion-weight 1.0 \
  --noise-augment-std 0.05 \
  --log-interval 50 \
  --image-log-interval 1000 \
  --save-interval 1000 \
  --wandb \
  --run-name "phase3-gan-timeshift" \
  --time-shift-alpha 2.0 \
  --diffusion-loss-gamma 1.0 \
  --schedule-rho 1.5 \
  --sample-steps 250
