
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.latent_distillation_modules import Encoder, Decoder
from diffusion import create_diffusion
import timm
from torch.cuda.amp import autocast
import numpy as np
from kolmogorov_tokenizers.utils import make_stratified_timesteps, group_images_by_timestep, get_1d_sincos_pos_embed_torch

class KARLTokenizer(nn.Module):

    def __init__(self, 
            base_tokenizer,
            encoder_width, encoder_num_layers, encoder_num_heads,
            decoder_width, decoder_num_layers, decoder_num_heads,
            quantize_latent=True, factorize_latent=True, vq_codebook_size=4096, vq_token_dim=12, vq_commitment_cost=0.25, vq_use_l2_norm = True,
            num_init_latent_tokens=32, patch_size=16, max_rollout_iters=8,
            dynamic_halting=True, dynamic_halting_threshold=0.55, max_grid_tokens=64,
            train_stage="latent_distillation_pretrain"
        ):
        
        super().__init__()
        
        self.min_tokens = 16
        self.max_tokens = 512
        self.train_stage = train_stage
        self.quantize_latent = quantize_latent
        if quantize_latent is True: factorize_latent=True
        self.factorize_latent = factorize_latent
        self.dynamic_halting = dynamic_halting
        self.dynamic_halting_threshold = dynamic_halting_threshold
        scale = encoder_width ** -0.5

        self.encoder = Encoder(encoder_width, encoder_num_layers, encoder_num_heads, joint_diffusion=True)
        self.encoder_positional_embedding = nn.Parameter(scale * torch.randn(encoder_width, max_grid_tokens, max_grid_tokens))

        # self.encoder_ln_pre = nn.LayerNorm(encoder_width)
        self.encoder_ln_post_halt = nn.LayerNorm(encoder_width)
        self.encoder_ln_post = nn.LayerNorm(encoder_width)
        self.pre_quantizer_mlp = nn.Linear(encoder_width, vq_token_dim, bias=True)
        self.halting_mlp = nn.Sequential(nn.Linear(encoder_width, 512, bias=True), nn.Tanh(), nn.Linear(512, 1, bias=True))

        self.decoder = Decoder(decoder_width, decoder_num_layers, decoder_num_heads, factorize_latent=self.factorize_latent, output_dim=base_tokenizer.codebook_size, joint_diffusion=False)
        self.decoder_positional_embedding = nn.Parameter(scale * torch.randn(decoder_width, max_grid_tokens, max_grid_tokens))
        self.decoder_mask_token  = nn.Parameter(scale * torch.randn(1, 1, decoder_width))
        self.decoder_embed = nn.Linear(vq_token_dim, decoder_width, bias=True)
        self.decoder_latent_tokens_timestep_embed = nn.Parameter(scale * torch.randn(512, decoder_width))
        
        self.latent_tokens = nn.Parameter(scale * torch.randn(512, encoder_width))
        
        self.patch_embed = nn.Conv2d(
            in_channels=3, out_channels=encoder_width-base_tokenizer.embed_dim,
            kernel_size=patch_size, stride=patch_size, bias=True)

        # self.patch_embed = nn.Conv2d(
        #     in_channels=3, out_channels=encoder_width,
        #     kernel_size=patch_size, stride=patch_size, bias=True)

        # self.all_token_counts = torch.arange(0, 256+32, 32)[1:]
        # self.all_token_counts = torch.cat((torch.arange(1, 16, 1), torch.arange(0, 256+128, 16)[1:]), dim=0)
        self.all_token_counts = torch.arange(0, 256+128, 16)[1:] ## original
        # print("all_token_counts: ", self.all_token_counts)
        
        # we discretize the reconstruction loss used for conditioning KARL encoder.
        # TODO: might not be important to discretize.
        self.rec_losses = torch.tensor([0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.09, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.24, 0.28, 0.32, 0.38])
        self.rec_loss_embeddings = nn.Parameter(self.rec_losses[:, None].repeat(1, encoder_width), requires_grad=True)

        self.logged_min_loss_values_prob = torch.zeros_like(self.rec_losses)
        self.logged_min_loss_values_prob[(self.rec_losses <= 0.04)] = 1.0
        self.logged_min_loss_values_prob /= self.logged_min_loss_values_prob.sum()


        self.apply(self._init_weights)
        
        if self.quantize_latent:
            from modules.vector_quantizer import VectorQuantizer
            # Intialization for Quantizer is done inside VectorQuantizer
            self.quantize = VectorQuantizer(
                codebook_size=vq_codebook_size,
                token_size=vq_token_dim,
                commitment_cost=vq_commitment_cost,
                use_l2_norm=vq_use_l2_norm)
        
        self.base_tokenizer = base_tokenizer

        if self.train_stage=="full_finetuning":
            # TODO: Ablate the requirement of different discriminators for different recurrent rollout iterations.
            # Intuition is at different rollout iteration .....
            from modules.losses.vqperceptual import VQLPIPSWithDiscriminator
            self.gan_losses = nn.ModuleList([VQLPIPSWithDiscriminator(
                disc_conditional= False, disc_in_channels= 3, 
                disc_start= 0, disc_weight= 0.2, codebook_weight= 1.0, # perceptual_weight=0.0
            ) for _ in range(max_rollout_iters)])
        
        if self.train_stage=="latent_distillation_pretrain":
            from modules.losses.nll import LabelSmoothingCrossEntropy
            self.criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

        self.num_latent_token_count = 256
        # self.diffusion = create_diffusion(timestep_respacing="250", predict_xstart=True, learn_sigma=True)  # default: 1000 steps, linear noise schedule
        # self.diffusion = create_diffusion(timestep_respacing="250", predict_xstart=True, learn_sigma=False, sigma_small=True)  # default: 1000 steps, linear noise schedule
        # self.diffusion = create_diffusion(timestep_respacing="", predict_xstart=True, learn_sigma=False, sigma_small=False)  # default: 1000 steps, linear noise schedule
        # self.diffusion = create_diffusion(timestep_respacing="", predict_xstart=True, learn_sigma=False, sigma_small=True)  # default: 1000 steps, linear noise schedule
        self.diffusion = create_diffusion(timestep_respacing="", predict_xstart=True, learn_sigma=True)  # default: 1000 steps, linear noise schedule
        latent_tokens_initialization = get_1d_sincos_pos_embed_torch(K=self.num_latent_token_count, D=encoder_width)
        self.latent_tokens = nn.Parameter(latent_tokens_initialization, requires_grad=True)

        '''
        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.encoder.transformer:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.encoder.transformer:
            print("adaln_zero_initialization...")
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            

        # Zero-out output layers:
        nn.init.constant_(self.encoder.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.encoder.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.encoder.final_layer.linear.weight, 0)
        nn.init.constant_(self.encoder.final_layer.linear.bias, 0)
        '''
        

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm) and module.weight is not None:
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def preprocess_encoder(self, grid_shape_2d):
        sampled_positional_embeddings = F.interpolate(self.encoder_positional_embedding[None], size=grid_shape_2d).reshape(1, self.encoder_positional_embedding.shape[0], -1).permute(0,2,1)
        return sampled_positional_embeddings
    
    def preprocess_decoder(self, img_tokens, grid_shape_2d):
        mask_tokens = self.decoder_mask_token.repeat(img_tokens.shape[0], img_tokens.shape[1], 1).to(img_tokens.dtype)
        sampled_positional_embeddings = F.interpolate(self.decoder_positional_embedding[None], size=grid_shape_2d).reshape(1, self.decoder_positional_embedding.shape[0], -1).permute(0,2,1)
        mask_tokens = mask_tokens + sampled_positional_embeddings
        return mask_tokens


    def reconstruct_images(self, logits, code, reconstruction_shape):
        # code = code.reshape(code.shape[0], reconstruction_shape[0], reconstruction_shape[1], code.shape[-1]).permute([0,3,1,2])
        # return self.base_tokenizer.vqgan.decode(code)
        if self.train_stage=="latent_distillation_pretrain":
            # decode using logits.
            logits = logits[:, :, :self.base_tokenizer.codebook_size]
            sample_dist = torch.distributions.categorical.Categorical(logits=logits)
            sampled_ids = sample_dist.sample()
            bsz = sampled_ids.shape[0]
            z_q = self.base_tokenizer.vqgan.quantize.get_codebook_entry(sampled_ids, shape=(bsz, reconstruction_shape[0], reconstruction_shape[1], self.base_tokenizer.codebook_emb_dim))
            return self.base_tokenizer.vqgan.decode(z_q)
        elif self.train_stage=="full_finetuning":
            # decode using code directly.
            code = code.reshape(code.shape[0], reconstruction_shape[0], reconstruction_shape[1], code.shape[-1]).permute([0,3,1,2])
            return self.base_tokenizer.vqgan.decode(code)

    def get_2d_tokens(self, imgs):
        vqgan_tokens, gt_indices = self.base_tokenizer.get_img_tokens(imgs)
        img_tokens = self.patch_embed(imgs)
        grid_shape_2d = img_tokens.shape[-2:]
        img_tokens = img_tokens.reshape(img_tokens.shape[0], img_tokens.shape[1], -1).permute([0,2,1])
        vqgan_tokens = vqgan_tokens.reshape(img_tokens.shape[0], img_tokens.shape[1], -1).permute([0,2,1])
        img_tokens = torch.cat((img_tokens, vqgan_tokens), dim=2)
    
        ## img_tokens = F.normalize(img_tokens, dim=-1)
        return img_tokens, gt_indices, grid_shape_2d
    

    def get_positional_embeddings(self, img_tokens, grid_shape_2d):
        assert(grid_shape_2d==(16,16) or grid_shape_2d==(16,32))
        masked_2d_tokens = self.preprocess_decoder(img_tokens, grid_shape_2d)
        encoder_sampled_positional_embeddings = self.preprocess_encoder(grid_shape_2d)
        return masked_2d_tokens, encoder_sampled_positional_embeddings

    def sample_token_counts(self, batch_size, sampled_token_counts=None, sampled_code_losses=None):
        token_counts = [self.num_latent_token_count]
        code_losses = None
        return token_counts, code_losses

    def forward(self, imgs, labels, learned_latent_tokens_ema=None, epoch=None, gan_optimizer_idx=None, gan_loss_weight=None, is_training=True):
        self.epoch = epoch
        self.gan_optimizer_idx = gan_optimizer_idx
        self.gan_loss_weight = gan_loss_weight
        
        ## sample 1D tokens -- representation learning phase
        total_loss, all_logs_dict, learned_latent_tokens = self.forward_representation_learning(imgs)
        print(learned_latent_tokens.shape, "learned_latent_tokens shape check new")
        
        # print(learned_latent_tokens.shape, learned_latent_tokens.min(), learned_latent_tokens.max(), "learned_latent_tokens.shape, learned_latent_tokens.min(), learned_latent_tokens.max()")
        if not is_training:
            num_images_to_log = 16 #(from utils/log_visuals)
            learned_latent_tokens = learned_latent_tokens[:num_images_to_log].repeat(16, 1, 1)
            labels = labels[:num_images_to_log].repeat(16)
            # diffusion_t = torch.randint(0, self.diffusion.num_timesteps, (learned_latent_tokens.shape[0],), device=learned_latent_tokens.get_device())
            diffusion_t = make_stratified_timesteps(learned_latent_tokens.shape[0], self.diffusion.num_timesteps, device=learned_latent_tokens.get_device())
            print(labels.shape, learned_latent_tokens.shape, "labels.shape, learned_latent_tokens_shape_check....")
            print(diffusion_t, "diffusion_t")
        else:
            diffusion_t = torch.randint(0, self.diffusion.num_timesteps, (learned_latent_tokens.shape[0],), device=learned_latent_tokens.get_device())
        num_latent_tokens=learned_latent_tokens.shape[1]
        diffusion_model_kwargs = dict(diffusion_label=labels, num_latent_tokens=num_latent_tokens, pos_embed=self.latent_tokens[:,:num_latent_tokens].repeat(learned_latent_tokens.shape[0], 1, 1), is_training=is_training)
        
        if learned_latent_tokens_ema is None: 
            learned_latent_tokens_ema = learned_latent_tokens
        else:
            assert(False)
        diffusion_loss_dict = self.diffusion.training_losses(self.encoder, learned_latent_tokens_ema.detach(), diffusion_t, diffusion_model_kwargs)
        # diffusion_loss_dict = self.diffusion.training_losses(self.encoder, learned_latent_tokens.detach(), diffusion_t, diffusion_model_kwargs)
        # diffusion_loss_dict = self.diffusion.training_losses(self.encoder, learned_latent_tokens, diffusion_t, diffusion_model_kwargs)
        
        diffusion_loss = diffusion_loss_dict["loss"].mean()
        if not is_training:
            
            ## single step denoising visualizations...
            samples = diffusion_loss_dict["model_output"]
            # samples = self.encoder.post_trunk_normalization.to_standard(samples, is_training)
            x_quantized, quant_result_dict = self.quantization(samples)
            # print(x_quantized.shape, x_quantized.min(), x_quantized.min(), "x_quantized.shape, x_quantized.min(), x_quantized.min()")
            
            pos_embed_indices = self.decoder_latent_tokens_timestep_embed[:self.num_latent_token_count]
            masked_2d_tokens = self.decoder_mask_token.repeat(x_quantized.shape[0], 256, 1).to(pos_embed_indices.dtype)
            sampled_positional_embeddings = F.interpolate(self.decoder_positional_embedding[None], size=(16,16)).reshape(1, self.decoder_positional_embedding.shape[0], -1).permute(0,2,1)
            masked_2d_tokens = masked_2d_tokens + sampled_positional_embeddings
            decoded_logits, decoded_code = self.decoding(x_quantized, masked_2d_tokens, pos_embed_indices)
            denoised_reconstructed_imgs = self.reconstruct_images(decoded_logits, decoded_code, (16,16))
            groups = group_images_by_timestep(denoised_reconstructed_imgs, diffusion_t, name_tag="single_step_denoised_reconstructed_imgs")
            all_logs_dict.update(groups)

            '''
            ### multi-step denoising visualizations...
            noise = torch.randn_like(learned_latent_tokens)
            noisy_learned_latent_tokens = self.diffusion.q_sample(learned_latent_tokens, diffusion_t, noise=noise)
            all_samples = []
            start_t_idx = -1
            end_t_idx = -1
            curr_t=-1
            cfg_scale = 4.0
            for t_idx in range(diffusion_t.shape[0]):
                if diffusion_t[t_idx]!=curr_t: 
                    curr_t = diffusion_t[t_idx]
                    if start_t_idx==-1: 
                        start_t_idx = t_idx
                    else:
                        end_t_idx = t_idx
                        y = labels[start_t_idx:end_t_idx]
                        y_null = torch.tensor([100] * y.shape[0], device=y.get_device())
                        y = torch.cat([y, y_null], 0)
                        z = noisy_learned_latent_tokens[start_t_idx:end_t_idx]
                        z = torch.cat([z, z], 0)
                        
                        diffusion_model_kwargs = dict(diffusion_label=y, num_latent_tokens=num_latent_tokens, pos_embed=self.latent_tokens[:,:num_latent_tokens].repeat(z.shape[0], 1, 1), is_training=is_training, cfg_scale=cfg_scale)
                        
                        samples = self.diffusion.p_sample_loop(
                            self.encoder.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=diffusion_model_kwargs, progress=True, device=noisy_learned_latent_tokens.get_device(), starting_timestamp=diffusion_t[start_t_idx]
                        )
                        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
                        all_samples.append(samples)
                        print(samples.shape, "samples shape check....")
                        start_t_idx = t_idx
            
            end_t_idx = diffusion_t.shape[0]
            y = labels[start_t_idx:end_t_idx]
            y_null = torch.tensor([100] * y.shape[0], device=y.get_device())
            y = torch.cat([y, y_null], 0)
            z = noisy_learned_latent_tokens[start_t_idx:end_t_idx]
            z = torch.cat([z, z], 0)            
            diffusion_model_kwargs = dict(diffusion_label=y, num_latent_tokens=num_latent_tokens, pos_embed=self.latent_tokens[:,:num_latent_tokens].repeat(z.shape[0], 1, 1), is_training=is_training, cfg_scale=cfg_scale)
            samples = self.diffusion.p_sample_loop(
                self.encoder.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=diffusion_model_kwargs, progress=True, device=noisy_learned_latent_tokens.get_device(), starting_timestamp=diffusion_t[start_t_idx]
            )
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
            print(samples.shape, "samples shape check....")
            all_samples.append(samples)
            samples = torch.cat((all_samples), dim=0)
            print(samples.shape, "all_samples shape check....")

            x_quantized, quant_result_dict = self.quantization(samples)
            # print(x_quantized.shape, x_quantized.min(), x_quantized.min(), "x_quantized.shape, x_quantized.min(), x_quantized.min()")
            
            pos_embed_indices = self.decoder_latent_tokens_timestep_embed[:self.num_latent_token_count]
            masked_2d_tokens = self.decoder_mask_token.repeat(x_quantized.shape[0], 256, 1).to(pos_embed_indices.dtype)
            sampled_positional_embeddings = F.interpolate(self.decoder_positional_embedding[None], size=(16,16)).reshape(1, self.decoder_positional_embedding.shape[0], -1).permute(0,2,1)
            masked_2d_tokens = masked_2d_tokens + sampled_positional_embeddings
            decoded_logits, decoded_code = self.decoding(x_quantized, masked_2d_tokens, pos_embed_indices)
            denoised_reconstructed_imgs = self.reconstruct_images(decoded_logits, decoded_code, (16,16))
            groups = group_images_by_timestep(denoised_reconstructed_imgs, diffusion_t, name_tag="multi_step_denoised_reconstructed_imgs")
            all_logs_dict.update(groups)
            '''
            

        all_logs_dict.update({
            "diffusion_loss": diffusion_loss.item()
        })
        total_loss = total_loss + diffusion_loss

        # denoise 1D tokens (detach 1D tokens)
        all_logs = [all_logs_dict]
        return total_loss, all_logs

    def init_latent_tokens(self, latent_tokens, is_denoising):
        noise = torch.randn_like(latent_tokens)
        if is_denoising:
            # add noise to latent_tokens
            assert(False)
            pass
        else:
            # two ablations: just noise or learned latent tokens like KARL/ALIT.
            # noisy_latents = noise
            # print("adding noise to latents")
            noisy_latents = noise + latent_tokens

        return noisy_latents, noise

    def diffusion_encoding(self, img_tokens=None, latent_tokens=None, is_denoising=False):
        num_latent_tokens = latent_tokens.shape[1]
        x, x_noise = self.init_latent_tokens(latent_tokens, is_denoising)
        # x = latent_tokens
        # print(x[0].min(), x[0].max(), x[0].mean(), x[0].std(), "pre rep learning min max mean")
        # print("================ rep learning =========================")
        if img_tokens is not None:
            x = torch.cat([x, img_tokens], dim=1)
            diffusion_t = torch.zeros_like(x[:,0,0]) - 1
            diffusion_label = torch.zeros_like(x[:,0,0]) - 1
            diffusion_label = diffusion_label.long()
        else:
            diffusion_t = None
            diffusion_label = None
        
        ## see if we need layer normnalization pre diffusion encoding
        ## probably se do not need, since encoder has ln.
        # x = self.encoder_ln_pre(x)
        x = self.encoder(x, num_latent_tokens=num_latent_tokens, attn_mask=None, diffusion_t=diffusion_t, diffusion_label=diffusion_label)
        # print("rep learning done....")
        return x

    
    def minimum_sufficient_statistics(self, x, num_img_tokens, is_latent_halting):
        pass
    
    def quantization(self, x_latents):
        latent_tokens_factorized = self.pre_quantizer_mlp(self.encoder_ln_post(x_latents))
        iter_logs_dict = {}
        if self.quantize_latent:
            assert(False)
            latent_tokens_quantized, quant_result_dict = self.quantize(latent_tokens_factorized, is_quantize=True)
        else:
            latent_tokens_quantized = latent_tokens_factorized
            quant_result_dict = None
        
        return latent_tokens_quantized, quant_result_dict


    def decoding(self, latent_tokens_quantized, masked_2d_tokens, pos_embed_indices, decoder_attn_mask=None):
        decoded_latent_1D_tokens = self.decoder_embed(latent_tokens_quantized)
        decoded_logits = self.decoder(decoded_latent_1D_tokens, masked_2d_tokens, pos_embed_indices, attn_mask=decoder_attn_mask) 
        decoded_logits_softmax = torch.nn.functional.softmax(decoded_logits, dim=-1)
        decoded_code = torch.einsum('nlc,cd->nld', decoded_logits_softmax, self.base_tokenizer.vqgan.quantize.embedding.weight.data)            
        return decoded_logits, decoded_code
    
    def forward_representation_learning(self, imgs, is_latent_halting=False, return_latent_only=False):
        
        sampled_token_counts, sampled_code_losses = self.sample_token_counts(batch_size=imgs.shape[0])
        # sampled_code_losses, sampled_code_loss_embeddings = self.get_discretized_loss_and_embedding(sampled_code_losses[...,0])
        num_latent_tokens = sampled_token_counts[0]

        img_tokens, gt_indices, grid_shape_2d = self.get_2d_tokens(imgs)
        masked_2d_tokens, encoder_sampled_positional_embeddings = self.get_positional_embeddings(img_tokens, grid_shape_2d)
        x = img_tokens + encoder_sampled_positional_embeddings
        
        
        # with autocast(enabled=False):
        # latent_tokens = self.latent_tokens[:num_latent_tokens][None].repeat(x.shape[0], 1, 1)
        # latent_tokens = get_1d_sincos_pos_embed_torch(num_latent_tokens, x.shape[2], device=x.get_device()).repeat(x.shape[0],1,1)
        latent_tokens = self.latent_tokens[:,:num_latent_tokens].repeat(x.shape[0], 1, 1)
        # print(latent_tokens.shape, "latent_tokens shape")
        # learned_latent_tokens_for_diffusion, learned_latent_tokens = self.diffusion_encoding(img_tokens=x, latent_tokens=latent_tokens)      
        learned_latent_tokens_for_diffusion = self.diffusion_encoding(img_tokens=x, latent_tokens=latent_tokens)      
        learned_latent_tokens = learned_latent_tokens_for_diffusion
        if return_latent_only: return learned_latent_tokens_for_diffusion

        # learned_latent_tokens = x_encoded[:,:num_latent_tokens]

        print(learned_latent_tokens.shape, learned_latent_tokens.min(), learned_latent_tokens.max(), learned_latent_tokens.mean(), learned_latent_tokens.std(), "learned_latent_tokens output of forward rep")
        x_quantized, quant_result_dict = self.quantization(learned_latent_tokens)
        
        pos_embed_indices = self.decoder_latent_tokens_timestep_embed[:sampled_token_counts[0]]
        # print(x_quantized.shape, x_quantized.min(), x_quantized.min(), "x_quantized.shape, x_quantized.min(), x_quantized.min()")
        decoded_logits, decoded_code = self.decoding(x_quantized, masked_2d_tokens, pos_embed_indices)

        # print(learned_latent_tokens_for_diffusion.min(), learned_latent_tokens_for_diffusion.max(), learned_latent_tokens_for_diffusion.mean(), "learned_latent_tokens_for_diffusion min max mean")
        # print(learned_latent_tokens.min(), learned_latent_tokens.max(), learned_latent_tokens.mean(), "learned_latent_tokens min max mean")
        # print(x_quantized.min(), x_quantized.max(), x_quantized.mean(), "x_quantized min max mean")
        # print(decoded_logits.min(), decoded_logits.max(), decoded_logits.mean(), "decoded_logits min max mean")
        # print("==============================================")

        total_loss = 0.
        iter_logs_dict = {}
        ## Loss Computation -- Reconstruction Losses
        if self.train_stage == "latent_distillation_pretrain":
            iter_nll_loss, iter_code_loss = self.forward_loss(gt_indices, decoded_logits, decoded_code)
            
            total_loss = total_loss + 2. * (iter_nll_loss + 1. * iter_code_loss)
            iter_logs_dict.update({
                "nll_loss_{}_{}".format(num_latent_tokens, is_latent_halting*1): iter_nll_loss.item(),
                "code_loss_{}_{}".format(num_latent_tokens, is_latent_halting*1): iter_code_loss.item(),
            })

            # if not is_latent_halting:
            with torch.no_grad():
                reconstructed_imgs = self.reconstruct_images(decoded_logits, decoded_code, grid_shape_2d)
                iter_rec_loss = torch.abs(imgs.contiguous() - reconstructed_imgs.contiguous()).reshape(decoded_code.shape[0], -1).mean(dim=-1)
                iter_logs_dict.update({
                    "reconstructed_imgs_{}_{}".format(num_latent_tokens, is_latent_halting*1): reconstructed_imgs,
                })

        elif self.train_stage == "full_finetuning":
            assert(False)
            reconstructed_imgs = self.reconstruct_images(decoded_logits, decoded_code, grid_shape_2d)
            gan_loss, logs_dict, iter_rec_loss = self.forward_gan_losses(
                imgs, reconstructed_imgs, is_latent_halting, 
                optimizer_idx=self.gan_optimizer_idx, latent_token_count=max(0,256-masked_token_count), 
                discriminator_loss_weight=self.gan_loss_weight
            )

            total_loss = total_loss + gan_loss
            iter_logs_dict.update(logs_dict)
            iter_logs_dict.update({
                "reconstructed_imgs_{}_{}".format(num_latent_tokens, is_latent_halting*1): reconstructed_imgs,
            })

        ## Loss Computation -- Quantization Loss
        if self.quantize_latent: 
            total_loss = total_loss + (1. * quant_result_dict['quantizer_loss'])
            iter_logs_dict.update({
                "quantization_loss_{}".format(num_latent_tokens): quant_result_dict['quantizer_loss'].item(),
            })
        
        return total_loss, iter_logs_dict, learned_latent_tokens_for_diffusion



    def forward_loss(self, gt_indices, decoded_logits, decoded_code):
        bsz, seq_len = gt_indices.size()
        assert(bsz==decoded_code.shape[0])
        assert(seq_len==decoded_code.shape[1])
        
        nll_loss_pixel, _ = self.criterion(decoded_logits[:, :, :self.base_tokenizer.codebook_size].reshape(bsz*seq_len, -1), gt_indices.reshape(bsz*seq_len))
        nll_loss_pixel = nll_loss_pixel.reshape(bsz, seq_len)

        vqgan_embedding_shape = self.base_tokenizer.vqgan.quantize.embedding.weight.data.shape[-1]
        gt_code = torch.gather(self.base_tokenizer.vqgan.quantize.embedding.weight.data, dim=0, index=gt_indices.reshape(bsz*seq_len)[...,None].repeat(1, vqgan_embedding_shape))
        gt_code = gt_code.reshape(bsz, seq_len, vqgan_embedding_shape)
        assert(gt_code.shape == decoded_code.shape)
        code_loss = (gt_code - decoded_code)**2
        
        return nll_loss_pixel.mean(), code_loss.mean()


    def get_last_layer(self):
        return self.base_tokenizer.vqgan.decoder.conv_out.weight

    def forward_gan_losses(self, imgs, reconstructed_imgs, is_latent_halting, optimizer_idx, latent_token_count, discriminator_loss_weight):
        assert(optimizer_idx is not None)
        # iter_idx = 0 # min(4, (latent_token_count // 64))
        iter_idx = max(0,min(8, (latent_token_count // 32))-1)
        if discriminator_loss_weight==0:
            global_step=-torch.inf
            self.gan_losses[iter_idx].discriminator_weight = 0.2
        else:
            global_step=torch.inf
            self.gan_losses[iter_idx].discriminator_weight = discriminator_loss_weight
        if optimizer_idx == 0:
            aeloss, log_dict_ae, iter_rec_loss = self.gan_losses[iter_idx](
                imgs, reconstructed_imgs, optimizer_idx, global_step=global_step,
                last_layer=self.get_last_layer(), split="train")

            iter_log_dict_ae = {}
            for key in log_dict_ae.keys():
                iter_log_dict_ae["{}_{}_{}".format(key, latent_token_count, is_latent_halting*1)] = log_dict_ae[key]

            return aeloss, iter_log_dict_ae, iter_rec_loss

        if optimizer_idx == 1:
            discloss, log_dict_disc, iter_rec_loss = self.gan_losses[iter_idx](
                imgs, reconstructed_imgs, optimizer_idx, global_step=global_step,
                last_layer=self.get_last_layer(), split="train")
            
            iter_log_dict_disc = {}
            for key in log_dict_disc.keys():
                iter_log_dict_disc["{}_{}".format(key, iter_idx)] = log_dict_disc[key]
            
            return discloss, iter_log_dict_disc, iter_rec_loss


    def diffusion_generate(self, cfg_scale, device):
        num_tokens = self.num_latent_token_count
        latent_size = 1024
        num_visuals = 16
        
        class_labels = [label for label in range(num_visuals)]
        n = len(class_labels)
        z = torch.randn(n, num_tokens, latent_size, device=device)
        y = torch.tensor(class_labels, device=device)

        # Setup classifier-free guidance:
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([100] * n, device=device)
        y = torch.cat([y, y_null], 0)
        num_latent_tokens = z.shape[1]
        model_kwargs = dict(diffusion_label=y, num_latent_tokens=num_latent_tokens, cfg_scale=cfg_scale, pos_embed=self.latent_tokens[:,:num_latent_tokens].repeat(z.shape[0], 1, 1), is_training=False)

        # Sample images:
        # z = z.permute([0,2,1])        
        samples = self.diffusion.p_sample_loop(
            self.encoder.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )
        # samples = samples.permute([0,2,1])
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        # samples = self.encoder.post_trunk_normalization.to_standard(samples, is_training=False)
        x_quantized, quant_result_dict = self.quantization(samples)
        # print(x_quantized.shape, x_quantized.min(), x_quantized.min(), "x_quantized.shape, x_quantized.min(), x_quantized.min()")
        
        pos_embed_indices = self.decoder_latent_tokens_timestep_embed[:num_tokens]
        masked_2d_tokens = self.decoder_mask_token.repeat(num_visuals, 256, 1).to(pos_embed_indices.dtype)
        sampled_positional_embeddings = F.interpolate(self.decoder_positional_embedding[None], size=(16,16)).reshape(1, self.decoder_positional_embedding.shape[0], -1).permute(0,2,1)
        masked_2d_tokens = masked_2d_tokens + sampled_positional_embeddings
        decoded_logits, decoded_code = self.decoding(x_quantized, masked_2d_tokens, pos_embed_indices)
        reconstructed_imgs = self.reconstruct_images(decoded_logits, decoded_code, (16,16))
        # iter_logs_dict.update({
        #     "reconstructed_imgs_{}_{}".format(num_latent_tokens, is_latent_halting*1): reconstructed_imgs,
        # })

        # print(reconstructed_imgs.shape, "reconstructed_imgs shape check...")

        return reconstructed_imgs, num_tokens