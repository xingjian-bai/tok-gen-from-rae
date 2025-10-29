import torch
import torch.nn as nn
from modules.base import ResidualAttentionBlock, FinalLayer
import math




#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

# class TokenBatchNorm1d(nn.Module):
#     def __init__(self, d_feat, eps=1e-5, momentum=0.01, sync=False, affine=False):
#         super().__init__()
#         bn_cls = nn.SyncBatchNorm if sync else nn.BatchNorm1d
#         self.bn = bn_cls(d_feat, eps=eps, momentum=momentum,
#                          affine=affine, track_running_stats=True)

#     def forward(self, x):              # x: [B, T, D]
#         B, T, D = x.shape
#         y = x.transpose(1, 2)          # [B, D, T]
#         y = self.bn(y)                 # normalize over D, stats across B,T
#         return y.transpose(1, 2)       # back to [B, T, D]

class TokenActNorm1d(nn.Module):
    def __init__(self, d, eps=1e-6, momentum=0.01):
        super().__init__()
        # Keep master EMA stats in float32
        self.register_buffer("mean", torch.zeros(d, dtype=torch.float32))
        self.register_buffer("var",  torch.ones(d,  dtype=torch.float32))
        self.eps = eps
        self.momentum = momentum

    @torch.no_grad()
    def _update(self, x):  # x: [B, T, D]
        # Compute stats in fp32 regardless of autocast
        with torch.cuda.amp.autocast(enabled=False):
            x32 = x.detach().to(torch.float32)
            m = x32.mean(dim=(0, 1))
            v = x32.var(dim=(0, 1), unbiased=False)
        # Lerp EMA buffers (also fp32)
        self.mean.lerp_(m, self.momentum)
        self.var.lerp_(v, self.momentum)

    def to_standard(self, x, is_training=True, update_stats=True):
        if is_training and update_stats:
            self._update(x)

        # Use fp32 master stats, but cast to x.dtype for arithmetic
        mean = self.mean.to(dtype=x.dtype, device=x.device)
        var  = self.var.to(dtype=x.dtype, device=x.device)
        # keep everything in x.dtype
        return (x - mean) / (var + self.eps).sqrt()

    def from_standard(self, z):
        mean = self.mean.to(dtype=z.dtype, device=z.device)
        var  = self.var.to(dtype=z.dtype, device=z.device)
        return z * (var + self.eps).sqrt() + mean


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.hidden_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.special_time_emb = nn.Parameter(torch.zeros(hidden_size))
        nn.init.normal_(self.special_time_emb, std=0.02)

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        # build sinusoid in fp32 for stability
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb  # fp32

    def forward(self, t: torch.Tensor):
        if t.dim() == 0:
            t = t[None]

        # ensure same device as the module
        dev = next(self.parameters()).device
        t = t.to(device=dev, dtype=torch.long)

        mask_special = (t < 0)
        mask_normal  = ~mask_special

        normal_out = None
        if mask_normal.any():
            t_norm = t[mask_normal].to(dtype=torch.float32)
            t_freq = self.timestep_embedding(t_norm, self.frequency_embedding_size)  # fp32
            # under autocast, Linear will return fp16/bf16 automatically
            normal_out = self.mlp(t_freq)  # dtype = autocast dtype or param dtype
            # assert(False)

        special_out = None
        if mask_special.any():
            special_out = self.special_time_emb.unsqueeze(0).expand(mask_special.sum(), -1)  # param dtype

        # choose target dtype from whatever we computed
        if normal_out is not None:
            target_dtype = normal_out.dtype
        else:
            target_dtype = special_out.dtype

        # allocate out with the *final* dtype and correct device
        out = torch.empty(t.shape[0], self.hidden_size, device=dev, dtype=target_dtype)

        if normal_out is not None:
            out[mask_normal] = normal_out
        if special_out is not None:
            if special_out.dtype != target_dtype:
                special_out = special_out.to(target_dtype)
            out[mask_special] = special_out

        return out



class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations.
    Supports:
      - classifier-free guidance dropout
      - labels == -1 as a special 'rep-learning' token with its own embedding
    """
    def __init__(self, num_classes, hidden_size, dropout_prob: float):
        super().__init__()
        use_cfg_embedding = 1 if dropout_prob > 0 else 0
        # +1 extra slot for special (-1) label
        self.special_label_index = num_classes + use_cfg_embedding
        self.null_label_index    = num_classes if use_cfg_embedding else None

        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding + 1, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        nn.init.normal_(self.embedding_table.weight, std=0.02)

    def token_drop(self, labels: torch.Tensor, force_drop_ids: torch.Tensor = None):
        """
        Drops *valid* labels (0..num_classes-1) to enable classifier-free guidance.
        Special labels (-1) are not dropped.
        """
        if self.null_label_index is None:
            return labels

        valid_mask = (labels >= 0)
        if force_drop_ids is None:
            drop_mask = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_mask = (force_drop_ids == 1)

        # Only drop where labels are valid class ids
        drop_mask = drop_mask & valid_mask
        out = labels.clone()
        # print(drop_mask.sum(), self.null_label_index, "drop_mask.sum(), self.null_label_index, checks...")
        out[drop_mask] = self.null_label_index
        return out

    def forward(self, labels: torch.Tensor, train: bool, force_drop_ids: torch.Tensor = None):
        """
        labels: int64 tensor [N]; use -1 for the special 'rep-learning' condition.
        """
        if labels.dim() == 0:
            labels = labels[None]

        # Apply CFG dropout on valid labels (keeps -1 untouched)
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        # else:
        #     print("not dropping labels....")

        # Map special label -1 → dedicated embedding index
        special_mask = (labels < 0)
        if special_mask.any():
            labels = labels.clone()
            labels[special_mask] = self.special_label_index

        embeddings = self.embedding_table(labels)
        return embeddings



class Encoder(nn.Module):
    def __init__(self, width, num_layers, num_heads, mlp_ratio=4.0, class_dropout_prob=0.1, num_classes=100, joint_diffusion=False):
        super().__init__()
        
        self.width = width
        self.num_layers = num_layers
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(
                ResidualAttentionBlock(width, num_heads, mlp_ratio=mlp_ratio, joint_diffusion=joint_diffusion)
            )

        self.t_embedder = TimestepEmbedder(width)
        self.y_embedder = LabelEmbedder(num_classes, width, class_dropout_prob)
        self.final_layer = FinalLayer(width, out_size=width*2)
        # self.final_layer = FinalLayer(width)
        # self.final_layer_eps = FinalLayer(width)
        # self.final_layer_xstart = FinalLayer(width)

        # self.rep_projection = nn.Linear(width, width, bias=True)
        # self.diff_projection = nn.Linear(width, 2*width, bias=True)
        # self.diff_projection = nn.Linear(width, width, bias=True)

        # self.latent_batch_norm = TokenBatchNorm1d(width)
        # self.latent_batch_norm = TokenBatchNorm1d(2*width)

        # self.latent_batch_norm = TokenBatchNorm1d(width)
        # self.post_trunk_normalization = TokenActNorm1d(width)

        # self.latent_layer_norm = nn.LayerNorm(width)

        # self.x0_projection = nn.Linear(width, width, bias=True)
        # self.xvar_projection = nn.Linear(width, width, bias=True)

        self.rec_losses = torch.tensor([0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.09, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.24, 0.28, 0.32, 0.38])
        self.rec_loss_embeddings = nn.Parameter(self.rec_losses[:, None].repeat(1, width), requires_grad=True)


    def get_discretized_loss_and_embedding(self, rec_loss_tensor):
        def discretize(loss_tensor, bin_values, bin_embeddings):
            bin_values = bin_values.to(loss_tensor.device)
            bin_embeddings = bin_embeddings.to(loss_tensor.device)

            loss_tensor_exp = loss_tensor.unsqueeze(-1)         # [B, S, 1]
            bin_values_exp = bin_values.view(1, 1, -1)           # [1, 1, N]

            valid_bins_mask = bin_values_exp >= loss_tensor_exp
            bin_diffs = bin_values_exp - loss_tensor_exp
            bin_diffs[~valid_bins_mask] = float('inf')

            bin_indices = bin_diffs.argmin(dim=-1)              # [B, S]

            binned_loss = bin_values[bin_indices]               # [B, S]
            binned_embed = bin_embeddings[bin_indices]          # [B, S, D]

            return binned_loss, binned_embed

        binned_rec_loss, binned_rec_embed = discretize(
            rec_loss_tensor, self.rec_losses, self.rec_loss_embeddings
        )
        return binned_rec_loss[:,0], binned_rec_embed[:,0]





    def forward(self, x, diffusion_t=None, num_latent_tokens=None, diffusion_label=None, attn_mask=None, pos_embed=None, is_training=True, rec_loss=None):
        # print(num_latent_tokens, "num_latent_tokens check....")
        diffusion_visualization = False
        if x.shape[2]!=self.width: 
            assert(False)
            diffusion_visualization=True
            x = x.permute([0,2,1])
        # if x.shape[2]!=self.width: 
        #     assert(False)
        #     x = x.permute([0,2,1])

        is_training = (self.training and is_training)
        if diffusion_t is not None and diffusion_label is not None:
            if pos_embed is not None: x = x + pos_embed
            # diffusion_t = torch.zeros_like(diffusion_t) - 1
            # diffusion_label = torch.zeros_like(diffusion_label) - 1
            t = self.t_embedder(diffusion_t) 
            y = self.y_embedder(diffusion_label, is_training)
            diffusion_c = t + y
        else: 
            assert(False)
            diffusion_c = None

        if rec_loss is None:
            rec_loss = torch.zeros_like(diffusion_label).float()
        rec_loss = rec_loss[:,None]

        rec_loss_discrete, rec_loss_embed = self.get_discretized_loss_and_embedding(rec_loss)
        print(rec_loss[:10], "rec_loss_first_10")
        print(rec_loss_discrete[:10], "rec_loss_discrete_first_10")

        diffusion_c = diffusion_c + rec_loss_embed

        x = x.permute(1, 0, 2)
        for i in range(self.num_layers):
            x = self.transformer[i](x, attn_mask, diffusion_c)
        # x = x.permute(1, 0, 2)
        # print(x.shape, x.min(), x.max(), "pre batch norm x check...")

        # x = self.diff_projection(x)
        # x = self.latent_batch_norm(x[:,:num_latent_tokens])
        # x = self.latent_layer_norm(x[:,:num_latent_tokens])
        
        # if diffusion_t[0]==-1:
        #     assert(diffusion_visualization==False)
        #     # rep_x = self.rep_projection(x)
        #     rep_x = self.diff_projection(x)
        #     # print(x.shape, x.min(), x.max(), "x learned_latents_for_diffusion")
        #     # print(rep_x.shape, rep_x.min(), rep_x.max(), "rep_x learned_latents")
        #     return x, rep_x[:,:,:1024]
        # else:
        #     # assert(False)
        #     assert(pos_embed is not None)
        #     x = self.diff_projection(x)
        #     if diffusion_visualization: 
        #         assert(False)
        #         x = x.permute([0,2,1])
        #     # x = x.permute([0,2,1])
            
        #     return x
        
        # x = torch.cat((self.latent_batch_norm(x[:,:num_latent_tokens]), x[:,num_latent_tokens:]), dim=1)
        # if diffusion_t[0]==-1: return x[:,:,:1024]

        '''
        x = x[:,:num_latent_tokens]
        x0 = self.x0_projection(x)
        if diffusion_t[0]==-1: 
            x0 = self.post_trunk_normalization.to_standard(x0, is_training)
            return x0
        else: 
            return x0
            # return torch.cat((x0, self.xvar_projection(x)), dim=-1)
        return x 
        '''


        x = x[:num_latent_tokens]
        x = self.final_layer(x, diffusion_c)
        x = x.permute(1, 0, 2)

        if diffusion_t[0]==-1: return x[:,:,:1024]

        '''
        x = x[:num_latent_tokens]
        if diffusion_t[0]==-1: 
            x = self.final_layer_xstart(x, diffusion_c)
        else:
            x = self.final_layer_eps(x, diffusion_c)
        x = x.permute(1, 0, 2)
        '''


        '''
        if diffusion_t[0]==-1: 
            x = self.post_trunk_normalization.to_standard(x, is_training)
            return x
        else: 
            return x
        '''
        
        return x 
        


    def forward_with_cfg(self, x, diffusion_t, num_latent_tokens, diffusion_label, pos_embed, cfg_scale, is_training=True, rec_loss=None):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, diffusion_t, num_latent_tokens, diffusion_label, attn_mask=None, pos_embed=pos_embed, is_training=is_training, rec_loss=rec_loss)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        eps, rest = model_out[:, :, :1024], model_out[:, :, 1024:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        # return torch.cat([eps, rest], dim=1)
        return torch.cat([eps, rest], dim=2)
    

class Decoder(nn.Module):
    def __init__(self, width, num_layers, num_heads, factorize_latent, output_dim, mlp_ratio=4.0, joint_diffusion=False):
        super().__init__()
        
        self.num_layers = num_layers
        self.factorize_latent = factorize_latent

        self.ln_pre = nn.LayerNorm(width)
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(ResidualAttentionBlock(width, num_heads, mlp_ratio=mlp_ratio, joint_diffusion=joint_diffusion))
        self.ln_post = nn.LayerNorm(width)

        self.ffn = nn.Sequential(
            nn.Linear(width, 2*width, bias=True), nn.Tanh(),
            nn.Linear(2*width, output_dim)
        )


    def forward(self, latent_1D_tokens, masked_2D_tokens, pos_embed_indices, attn_mask=None):
        latent_1D_tokens = latent_1D_tokens + pos_embed_indices[None]
        x = torch.cat([masked_2D_tokens, latent_1D_tokens], dim=1)
        
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        all_attn_weights = []
        for i in range(self.num_layers):
            x = self.transformer[i](x, attn_mask=attn_mask)
        x = x.permute(1, 0, 2)

        reconstructed_2D_tokens = x[:, :masked_2D_tokens.shape[1]]
        reconstructed_2D_tokens = self.ln_post(reconstructed_2D_tokens)
        reconstructed_2D_tokens = self.ffn(reconstructed_2D_tokens)

        return reconstructed_2D_tokens