TRAIN_DATA_DIR="/data/vision/torralba/datasets/imagenet100"

TORCH_DISTRIBUTED_DEBUG=INFO torchrun --standalone --nproc_per_node=4 \
  main_pretrain.py \
  --config configs/reconstruction.yaml \
  --data-path $TRAIN_DATA_DIR \
  --results-dir outputs \
  --experiment-name tok-gen-diffusion-shared-noise \
  --image-size 256 --precision bf16 \
  # --ckpt outputs/tok-gen-diffusion/checkpoints/0015000.pt