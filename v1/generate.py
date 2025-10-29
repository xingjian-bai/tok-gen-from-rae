import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
import sys
sys.path.append("/data/vision/torralba/selfmanaged/torralba/projects/sduggal/research/tok_gen/v1/")
sys.path.append("/data/vision/torralba/selfmanaged/torralba/projects/sduggal/research/tok_gen/v1/base_tokenizers/")
import kolmogorov_tokenizers
from utils import misc
from tqdm import tqdm
import os


# set arguments accordingly
args = {
    'device': 'cuda:0',
    'input_size': 256,
    'model': 'karl_small',
    'base_tokenizer': 'vqgan',
    'ckpt': '/data/vision/torralba/selfmanaged/torralba/projects/sduggal/research/tok_gen/v1/output_dir/latent_distillation_pretrain/no_quant_no_post_trunk_norm_with_dit_final_layer_learned_sigma_ip_sample_128_tokens/checkpoint-last.pth',
    # 'ckpt': '/data/vision/torralba/selfmanaged/torralba/projects/sduggal/research/tok_gen/v1/output_dir/latent_distillation_pretrain/linear_project_actnom_outside_trunk_no_quant_no_post_trunk_norm_with_dit_final_layer/checkpoint-last.pth',
    # 'ckpt': '/data/vision/torralba/selfmanaged/torralba/projects/sduggal/research/tok_gen/v1/output_dir/latent_distillation_pretrain/no_quant_no_post_trunk_norm_with_dit_final_layer_semi_large/checkpoint-last.pth',
    # 'ckpt': '/data/vision/torralba/selfmanaged/torralba/projects/sduggal/research/tok_gen/v1/output_dir/latent_distillation_pretrain/no_quant_no_post_trunk_norm_with_dit_final_layer_large_sigma/checkpoint-last.pth',
    # 'ckpt': '/data/vision/torralba/selfmanaged/torralba/projects/sduggal/research/tok_gen/v1/output_dir/latent_distillation_pretrain/no_quant_no_post_trunk_norm_with_dit_final_layer_learned_sigma/checkpoint-last.pth',
    # 'ckpt': '/data/vision/torralba/selfmanaged/torralba/projects/sduggal/research/tok_gen/v1/output_dir/latent_distillation_pretrain/no_quant_no_post_trunk_norm_with_dit_final_layer_learned_sigma_ip_sample/checkpoint-last.pth',
    'factorize_latent': True,
    'quantize_latent': False,
    'cfg_scale': 4,
}
args = misc.Args(**args)


def main():
    args.save_path = "/".join(args.ckpt.split("/")[:-1]) + "/gen_cfg_{}_250_steps".format(args.cfg_scale)
    if not os.path.exists(args.save_path):
        os.system("mkdir -p " + args.save_path)
    base_tokenizer_args = {
    "id": args.base_tokenizer,
    "is_requires_grad": False
    }
    model = kolmogorov_tokenizers.__dict__[args.model](
        base_tokenizer_args=base_tokenizer_args, 
        quantize_latent=args.quantize_latent, factorize_latent=args.factorize_latent,
        train_stage="latent_distillation_pretrain"
    )
    model.to(args.device)
    checkpoint = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(checkpoint['ema'], strict=True)
    model.eval()

    num_tokens = model.num_latent_token_count
    latent_size = 1024
    num_visuals = 50
    num_labels = 100
    total = 0
    
    for label in tqdm(range(0, num_labels)):
        class_labels = [label for _ in range(num_visuals)]
        n = len(class_labels)
        z = torch.randn(n, num_tokens, latent_size, device=args.device)
        y = torch.tensor(class_labels, device=args.device)

        # Setup classifier-free guidance:
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([100] * n, device=args.device)
        y = torch.cat([y, y_null], 0)
        num_latent_tokens = z.shape[1]
        model_kwargs = dict(diffusion_label=y, num_latent_tokens=num_latent_tokens, cfg_scale=args.cfg_scale, pos_embed=model.latent_tokens[:,:num_latent_tokens].repeat(z.shape[0], 1, 1), is_training=False)

        # Sample images:
        # z = z.permute([0,2,1])        
        samples = model.diffusion.p_sample_loop(
            model.encoder.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=args.device
        )
        # samples = samples.permute([0,2,1])
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        # samples = self.encoder.post_trunk_normalization.to_standard(samples, is_training=False)
        x_quantized, quant_result_dict = model.quantization(samples)
        # print(x_quantized.shape, x_quantized.min(), x_quantized.min(), "x_quantized.shape, x_quantized.min(), x_quantized.min()")
        
        pos_embed_indices = model.decoder_latent_tokens_timestep_embed[:num_tokens]
        masked_2d_tokens = model.decoder_mask_token.repeat(num_visuals, 256, 1).to(pos_embed_indices.dtype)
        sampled_positional_embeddings = F.interpolate(model.decoder_positional_embedding[None], size=(16,16)).reshape(1, model.decoder_positional_embedding.shape[0], -1).permute(0,2,1)
        masked_2d_tokens = masked_2d_tokens + sampled_positional_embeddings
        decoded_logits, decoded_code = model.decoding(x_quantized, masked_2d_tokens, pos_embed_indices)
        reconstructed_imgs = model.reconstruct_images(decoded_logits, decoded_code, (16,16))
        # iter_logs_dict.update({
        #     "reconstructed_imgs_{}_{}".format(num_latent_tokens, is_latent_halting*1): reconstructed_imgs,
        # })

        # print(reconstructed_imgs.shape, "reconstructed_imgs shape check...")

        # Save samples to disk as individual .png files
        reconstructed_imgs = np.clip(reconstructed_imgs.cpu().numpy().transpose([0, 2, 3, 1]) * 255, 0, 255)
        reconstructed_imgs = reconstructed_imgs.astype(np.uint8)
        for i, sample in enumerate(reconstructed_imgs):
            index = i + total
            print(sample.min(), sample.max(), sample.shape)
            Image.fromarray(sample).save(f"{args.save_path}/{str(label).zfill(6)}_{index:06d}.png")

        total += reconstructed_imgs.shape[0]
        
main()