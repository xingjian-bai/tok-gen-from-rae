TRAIN_DATA_DIR="/data/vision/torralba/datasets/imagenet100"

CUDA_VISIBLE_DEVICES=4 TORCH_DISTRIBUTED_DEBUG=INFO torchrun --standalone --nproc_per_node=1 \
  main_pretrain.py \
  --config configs/reconstruction.yaml \
  --data-path $TRAIN_DATA_DIR \
  --results-dir outputs \
  --experiment-name tok-gen-diffusion-shared-noise-no-ipsample-single-image-overfit-better-lr-no-gen-recon-loss \
  --image-size 256 --precision bf16 \
  # --ckpt outputs/tok-gen-diffusion-shared-noise-no-ipsample-single-image-overfit/checkpoints/0005000.pt
  # --ckpt outputs/tok-gen-diffusion/checkpoints/0015000.pt