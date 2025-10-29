## Single node run comamnd
CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node=1 \
    --master_port=12346 main_pretrain.py \
    --batch_size 128 \
    --num_workers 10 \
    --model karl_small \
    --base_tokenizer vqgan_adaptive \
    --factorize_latent \
    --epochs 400 \
    --warmup_epochs 20 \
    --blr 1.e-4 --weight_decay 0.05 \
    --output_dir ./output_dir/latent_distillation_pretrain/no_quant_no_post_trunk_norm_with_dit_final_layer_learned_sigma_ip_sample_adaptive_test/ \
    --resume ./output_dir/latent_distillation_pretrain/no_quant_no_post_trunk_norm_with_dit_final_layer_learned_sigma_ip_sample_adaptive/checkpoint-260.pth \
    --data_path $TRAIN_DATA_DIR

    # --resume ./output_dir/latent_distillation_pretrain/tok_gen_256_batch_norm_with_diffusion_learnable_pos_embed_plus_noise_large_bs128/checkpoint-last.pth \
    # --resume ./output_dir/latent_distillation_pretrain/tok_gen_256_batch_norm/checkpoint-last.pth \
    # --resume ./output_dir/latent_distillation_pretrain/tok_gen_256_batch_norm_with_diffusion_learnable_pos_embed_plus_noise_large_bs128_diffusion_vis_redo_layernorm_first_sep_linear_project_fixed_small_var/checkpoint-last.pth \
    # --quantize_latent \
