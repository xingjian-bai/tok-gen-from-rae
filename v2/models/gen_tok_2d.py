import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from transport import create_transport
from modules.encoders.patch_embedding import ViTMAEEmbeddings
from modules.encoders.lightningDIT import LightningDiT
from modules.disc import build_discriminator, select_gan_losses, LPIPS, calculate_adaptive_weight
from modules.decoders import GeneralDecoder
from transformers import AutoConfig
from modules.diffusion import create_diffusion
from models.utils import make_stratified_timesteps, group_images_by_timestep


def get_1d_sincos_pos_embed_torch(K: int, D: int, device=None, dtype=None) -> torch.Tensor:
    """
    Fixed 1D sin/cos positional embedding.
    Returns [1, K, D] (broadcastable across batch).
    """
    # support odd D by padding then truncating
    De = D if D % 2 == 0 else D + 1
    half = De // 2

    pos = torch.arange(K, device=device, dtype=torch.float32)             # [K]
    omega = torch.arange(half, device=device, dtype=torch.float32) / half # [half]
    omega = 1.0 / (10000 ** omega)                                        # [half]

    angles = pos[:, None] * omega[None, :]                                # [K, half]
    emb = torch.cat([angles.sin(), angles.cos()], dim=1)                  # [K, De]
    emb = emb[:, :D]                                                      # [K, D]
    return emb.unsqueeze(0).to(dtype=dtype)                               # [1, K, D]

class GenTok2D(nn.Module):

    def __init__(self, gan_cfg, device, decoder_config_path='configs/decoder/ViTB/config.json'):
        super().__init__()
        self.num_latent_tokens = 256
        self.gan_cfg = gan_cfg

        self.patch_embed = ViTMAEEmbeddings(image_size=256, patch_size=16, num_channels=3, hidden_size=768) # same configuration as DIT-B/2
        
        self.init_encoder()
        self.init_decoder(decoder_config_path)
        self.init_gan(gan_cfg, device)
        
        latent_tokens_initialization = get_1d_sincos_pos_embed_torch(K=self.num_latent_tokens, D=768)
        self.latent_tokens = nn.Parameter(latent_tokens_initialization, requires_grad=True)
        
        self.diffusion = create_diffusion(timestep_respacing="", predict_xstart=True, learn_sigma=False, sigma_small=True)  # default: 1000 steps, linear noise schedule
        # self.transport = create_transport()

    def get_learnable_params(self, no_disc=False, only_disc=False):
        only_disc_params = []
        no_disc_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad: continue
            if "discriminator" in name:
                only_disc_params.append(param)
            else:
                no_disc_params.append(param)
        if only_disc: return only_disc_params
        if no_disc: return no_disc_params

    def init_encoder(self):
        self.reshape_to_2d = True
        self.encoder = LightningDiT(
            input_size = 16,
            patch_size = 1,
            in_channels = 768,
            hidden_size = 768,
            depth = 12,
            num_heads = 12,
            mlp_ratio = 4.0,
            class_dropout_prob = 0.1,
            num_classes = 100,
            learn_sigma=False,
            use_qknorm = False,
            use_swiglu = True,
            use_rope = True,
            use_rmsnorm = True,
            wo_shift = False,
        )

    def init_gan(self, gan_cfg, device):
        disc_cfg = gan_cfg.get("disc", {})
        if not disc_cfg:
            raise ValueError("gan.disc configuration is required for stage-1 training.")
        loss_cfg = gan_cfg.get("loss", {})
        self.perceptual_weight = float(loss_cfg.get("perceptual_weight", 0.0))
        self.disc_weight = float(loss_cfg.get("disc_weight", 0.0))
        
        gan_start_epoch = int(loss_cfg.get("disc_start", 0))
        disc_update_epoch = int(loss_cfg.get("disc_upd_start", gan_start_epoch))
        lpips_start_epoch = int(loss_cfg.get("lpips_start", 0))
        disc_updates = int(loss_cfg.get("disc_updates", 1))

        self.max_d_weight = float(loss_cfg.get("max_d_weight", 1e4))
        disc_loss_type = loss_cfg.get("disc_loss", "hinge")
        gen_loss_type = loss_cfg.get("gen_loss", "vanilla")
        self.discriminator, self.disc_aug = build_discriminator(disc_cfg, device)
        self.disc_loss_fn, self.gen_loss_fn = select_gan_losses(disc_loss_type, gen_loss_type)

        self.lpips = LPIPS()
        self.lpips.eval()

        self.last_layer = self.decoder.decoder_pred.weight


    def init_decoder(self, decoder_config_path, decoder_patch_size=16):
        decoder_config = AutoConfig.from_pretrained(decoder_config_path)
        decoder_config.hidden_size = 768 # set the hidden size of the decoder to be the same as the encoder's output
        decoder_config.patch_size = decoder_patch_size
        # decoder_config.image_size = int(decoder_patch_size * sqrt(self.base_patches)) 
        self.decoder = GeneralDecoder(decoder_config, num_patches=256)



    def forward(self, imgs, labels, use_gan, use_lpips, optimizer_idx, only_reconstruction=False):
        assert(labels==None) # let's focus on unconditional generation first.
        self.use_lpips = use_lpips
        self.use_gan = use_gan
        self.optimizer_idx = optimizer_idx

        total_loss, output_dict, x_noise  = self.forward_representation_learning(imgs)
        
        if not only_reconstruction and self.optimizer_idx=="generator":
            learned_latent_tokens = output_dict['learned_latent_tokens']
            learned_latent_tokens = learned_latent_tokens.reshape(learned_latent_tokens.shape[0], learned_latent_tokens.shape[1], -1).permute([0,2,1])
            generation_dict = self.forward_generative_modeling(imgs, learned_latent_tokens, x_noise)
            total_loss = total_loss + generation_dict["total_loss"]
            for key in generation_dict.keys():
                output_dict[key] = generation_dict[key]
            output_dict["total_loss"]=total_loss
        return total_loss, output_dict


    def get_img_patch_embeddings(self, imgs):
        patch_embed = self.patch_embed(imgs)
        return patch_embed

    def encode(self, imgs):
        img_patch_embed = self.get_img_patch_embeddings(imgs)
        latent_tokens = self.latent_tokens[:,:self.num_latent_tokens].repeat(img_patch_embed.shape[0], 1, 1)

        noise = torch.randn_like(latent_tokens)
        noisy_latent_tokens = noise + latent_tokens

        x = torch.cat([noisy_latent_tokens, img_patch_embed], dim=1)
        diffusion_t = torch.zeros_like(x[:,0,0]) - 1

        ## no class conditioning for now.
        diffusion_label = None # torch.zeros_like(x[:,0,0]).long() #- 1
        # diffusion_label = diffusion_label.long()

        z = self.encoder(x, t=diffusion_t, y=diffusion_label, pos_embed=None)

        # if self.training and self.noise_tau > 0:
        #     z = self.noising(z)
        
        # if self.reshape_to_2d:
        #     b, n, c = z.shape
        #     h = w = int(math.sqrt(self.num_latent_tokens))
        #     z = z.transpose(1, 2).view(b, c, h, w)
        
        # if self.do_normalization:
        #     latent_mean = self.latent_mean.to(z.device) if self.latent_mean is not None else 0
        #     latent_var = self.latent_var.to(z.device) if self.latent_var is not None else 1
        #     z = (z - latent_mean) / torch.sqrt(latent_var + self.eps)

        return z, noise
    
    def decode(self, z, reshape_to_2d=True):
        # if self.do_normalization:
        #     latent_mean = self.latent_mean.to(z.device) if self.latent_mean is not None else 0
        #     latent_var = self.latent_var.to(z.device) if self.latent_var is not None else 1
        #     z = z * torch.sqrt(latent_var + self.eps) + latent_mean
        if reshape_to_2d:
            b, c, h, w = z.shape
            n = h * w
            z = z.view(b, c, n).transpose(1, 2)
        output = self.decoder(z, drop_cls_token=False).logits
        x_rec = self.decoder.unpatchify(output)
        # x_rec = x_rec * self.encoder_std.to(x_rec.device) + self.encoder_mean.to(x_rec.device)
        
        return x_rec
    

    def forward_representation_learning(self, imgs):
        x, x_noise = self.encode(imgs)
        recon = self.decode(x)
        reconstruction_loss, reconstruction_dict = self.forward_reconstruction_loss(recon, imgs, self.optimizer_idx)
        reconstruction_dict.update(learned_latent_tokens=x)
        return reconstruction_loss, reconstruction_dict, x_noise


    def forward_generative_modeling(self, imgs, z, z_noise, y=None):
        # model_kwargs = dict(y=y)
        # loss_dict = self.transport.training_losses(self.encoder, z, model_kwargs)

        if not self.training:
            num_images_to_log = 16 # (from utils/log_visuals)
            z = z[:num_images_to_log].repeat(16, 1, 1)
            z_noise = z_noise[:num_images_to_log].repeat(16, 1, 1)
            if y is not None: y = y[:num_images_to_log].repeat(16)
            diffusion_t = make_stratified_timesteps(z.shape[0], self.diffusion.num_timesteps, device=z.get_device())
        else:
            diffusion_t = torch.randint(0, self.diffusion.num_timesteps, (z.shape[0],), device=z.get_device())
        
        diffusion_model_kwargs = dict(y=y, pos_embed=self.latent_tokens[:,:z.shape[1]].repeat(z.shape[0], 1, 1), is_unpatchify=False)
        diffusion_loss_dict = self.diffusion.training_losses(self.encoder, z.detach(), diffusion_t, diffusion_model_kwargs, noise=z_noise)
        diffusion_loss = diffusion_loss_dict["loss"].mean()
        total_loss = diffusion_loss
        output_dict = {}

        if self.training and self.optimizer_idx == "generator":
            # _prev_req = [p.requires_grad for p in self.decoder.parameters()]
            # _prev_mode = self.decoder.training
            # for p in self.decoder.parameters():
            #     p.requires_grad_(False)
            # self.decoder.eval()  # optional: avoid BN/dr dropout drift

            # denoised_reconstructed_imgs = self.decode(diffusion_loss_dict["model_output"], reshape_to_2d=False)
            # total_loss, output_dict = self.forward_reconstruction_loss(denoised_reconstructed_imgs, imgs, self.optimizer_idx, branch_name="diffusion/", total_loss=total_loss)
            
            # for p, rg in zip(self.decoder.parameters(), _prev_req):
            #     p.requires_grad_(rg)
            # self.decoder.train(_prev_mode)

            output_dict.update({"diffusion/diffusion_loss": diffusion_loss})
            output_dict.update(total_loss=total_loss)
        

        if not self.training:
            output_dict = {}
            output_dict.update({"diffusion/diffusion_loss": diffusion_loss})
            output_dict.update(total_loss=total_loss)    

            ## single step denoising visualizations...
            denoised_reconstructed_imgs = self.decode(diffusion_loss_dict["model_output"], reshape_to_2d=False)
            groups = group_images_by_timestep(denoised_reconstructed_imgs, diffusion_t, name_tag="single_step_denoised_reconstructed_imgs")
            output_dict.update(groups)

        return output_dict
        

    def forward_reconstruction_loss(self, recon, imgs, optimizer_idx, branch_name="", total_loss=0.):

        real_normed = imgs * 2.0 - 1.0
        recon_normed = recon * 2.0 - 1.0
        output_dict = {}

        if self.training and optimizer_idx=="generator":
            self.discriminator.eval()
            rec_loss = F.l1_loss(recon, imgs)
            if self.use_lpips:
                lpips_loss = self.lpips(imgs, recon)
            else:
                lpips_loss = rec_loss.new_zeros(())
            if "diffusion" in branch_name:
                recon_total = self.perceptual_weight * lpips_loss
            else:
                recon_total = rec_loss + self.perceptual_weight * lpips_loss

            if self.use_gan:
                fake_aug = self.disc_aug.aug(recon_normed)
                logits_fake, _ = self.discriminator(fake_aug, None)
                gan_loss = self.gen_loss_fn(logits_fake)
                adaptive_weight = calculate_adaptive_weight(
                    recon_total, gan_loss, self.last_layer, self.max_d_weight
                )
                total_loss = total_loss + recon_total + self.disc_weight * adaptive_weight * gan_loss
            else:
                gan_loss = torch.zeros_like(recon_total)
                adaptive_weight = torch.zeros_like(recon_total)
                total_loss = total_loss + recon_total

            output_dict = {
                '{}rec_loss'.format(branch_name): rec_loss,
                '{}recon_total'.format(branch_name): recon_total,
                '{}lpips_loss'.format(branch_name): lpips_loss,
                '{}gan_loss'.format(branch_name): gan_loss,
                '{}adaptive_weight'.format(branch_name): adaptive_weight,
            }

        elif self.training and optimizer_idx=="discriminator":
            self.discriminator.train()
            fake_detached = recon_normed.detach()
            # discretize
            fake_detached = fake_detached.clamp(-1.0, 1.0)
            fake_detached = torch.round((fake_detached + 1.0) * 127.5) / 127.5 - 1.0
            fake_input = self.disc_aug.aug(fake_detached)
            real_input = self.disc_aug.aug(real_normed)
            logits_fake, logits_real = self.discriminator(fake_input, real_input)
            d_loss = self.disc_loss_fn(logits_real, logits_fake)
            total_loss = total_loss + d_loss

            output_dict = {
                '{}disc_loss'.format(branch_name): d_loss.detach(),
                '{}logits_fake'.format(branch_name): logits_fake.detach().mean(),
                '{}logits_real'.format(branch_name): logits_real.detach().mean()
            }
        
        output_dict.update({
            f"{branch_name}recon": recon,
            f"total_loss": total_loss,
        })

        return total_loss, output_dict
    

    def diffusion_generate(self, device):
        num_tokens = 256
        latent_size = 768
        num_visuals = 16
        n = num_visuals
        z = torch.randn(n, num_tokens, latent_size, device=device)
        y = None

        model_kwargs = dict(y=y, pos_embed=self.latent_tokens[:,:num_tokens].repeat(z.shape[0], 1, 1), is_unpatchify=False)
        samples = self.diffusion.p_sample_loop(
            self.encoder.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )
        reconstructed_imgs = self.decode(samples, reshape_to_2d=False)
        return reconstructed_imgs

        
        
    


