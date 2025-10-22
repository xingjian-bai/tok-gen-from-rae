"""
Dual-mode MAE encoder that supports both image encoding and latent denoising.

This module extends the pre-trained ViT-MAE backbone with:
  * Additional token embeddings for latent inputs.
  * Learned positional embeddings for latent tokens.
  * Simple time embedding that conditions every token #! (t=1 --> reconstruction, t<1 --> diffusion).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init
from transformers import ViTMAEForPreTraining


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal + MLP time embedding.
    """

    def __init__(self, embed_dim: int, hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        half_dim = embed_dim // 2
        self.register_buffer(
            "frequency",
            torch.exp(-math.log(10000) * torch.arange(half_dim) / half_dim),
            persistent=False,
        )
        hidden_dim = hidden_dim or embed_dim * 2
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.dim() == 0:
            timesteps = timesteps.unsqueeze(0)
        timesteps = timesteps.view(-1, 1)
        freqs = self.frequency.unsqueeze(0).to(timesteps.device) * timesteps
        sin = torch.sin(freqs)
        cos = torch.cos(freqs)
        emb = torch.cat([sin, cos], dim=-1)
        if emb.shape[-1] < self.embed_dim:
            emb = F.pad(emb, (0, self.embed_dim - emb.shape[-1]))
        return self.mlp(emb)


def _resize_positional_embedding(pos_embed: torch.Tensor, new_tokens: int) -> torch.Tensor:
    """
    Resize 2D sine-cosine positional embeddings using bicubic interpolation.
    """
    old_tokens = pos_embed.shape[1]
    if old_tokens == new_tokens:
        return pos_embed
    side_old = int(math.sqrt(old_tokens))
    side_new = int(math.sqrt(new_tokens))
    if side_new * side_new != new_tokens:
        raise ValueError("Positional embedding requires square number of tokens.")

    pos = pos_embed[0].reshape(side_old, side_old, -1).permute(2, 0, 1).unsqueeze(0)
    pos = F.interpolate(pos, size=(side_new, side_new), mode="bicubic", align_corners=False)
    pos = pos.squeeze(0).permute(1, 2, 0).reshape(1, side_new * side_new, -1)
    return pos


class DualMAEEncoder(nn.Module):
    """
    Dual-input MAE encoder with shared transformer weights for reconstruction & diffusion.
    """

    def __init__(
        self,
        model_name: str = "facebook/vit-mae-base",
        image_size: int = 256,
        latent_tokens: Optional[int] = None,
        latent_dim: Optional[int] = None,
        patch_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        mae = ViTMAEForPreTraining.from_pretrained(model_name)
        self.mae = mae
        mae.config.image_size = image_size
        mae.vit.embeddings.patch_embeddings.image_size = (image_size, image_size)
        self.embed_dim = mae.config.hidden_size
        self.patch_size = patch_size or mae.config.patch_size
        self.image_tokens = (image_size // self.patch_size) ** 2
        self.latent_tokens = latent_tokens or self.image_tokens
        self.latent_grid = int(math.sqrt(self.latent_tokens))
        if self.latent_grid * self.latent_grid != self.latent_tokens:
            raise ValueError("latent_tokens must form a square grid.")

        self.latent_dim = latent_dim or self.embed_dim
        self.latent_proj = nn.Linear(self.latent_dim, self.embed_dim)
        nn_init.eye_(self.latent_proj.weight)
        nn_init.zeros_(self.latent_proj.bias)

        # Positional embeddings
        image_pos = mae.vit.embeddings.position_embeddings[:, 1:, :]
        image_pos = _resize_positional_embedding(image_pos, self.image_tokens)
        self.image_pos_embed = nn.Parameter(image_pos)
        self.latent_pos_embed = nn.Parameter(torch.zeros(1, self.latent_tokens, self.embed_dim))
        nn_init.trunc_normal_(self.latent_pos_embed, std=0.02)

        self.image_placeholder = nn.Parameter(torch.zeros(1, self.image_tokens, self.embed_dim))
        self.latent_placeholder = nn.Parameter(torch.zeros(1, self.latent_tokens, self.embed_dim))
        nn_init.trunc_normal_(self.image_placeholder, std=0.02)
        nn_init.trunc_normal_(self.latent_placeholder, std=0.02)

        self.time_embed = SinusoidalTimeEmbedding(self.embed_dim)

    def load_from_stage1(self, state_dict: dict) -> None:
        mapped = {}
        for key, value in state_dict.items():
            if key.startswith("encoder."):
                mapped_key = key.replace("encoder.", "mae.")
                mapped[mapped_key] = value
        self.load_state_dict(mapped, strict=False)

    def _patch_embed(self, images: torch.Tensor) -> torch.Tensor:
        x = self.mae.vit.embeddings.patch_embeddings(images)
        if x.dim() == 3:
            return x
        x = x.flatten(2).transpose(1, 2)
        return x

    def encode_image(self, images: torch.Tensor, timesteps: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.forward(images=images, latents=None, timesteps=timesteps)

    def encode_latent(self, latents: torch.Tensor, timesteps: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.forward(images=None, latents=latents, timesteps=timesteps)

    def forward(
        self,
        images: Optional[torch.Tensor],
        latents: Optional[torch.Tensor],
        timesteps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        batch = None

        if images is not None:
            image_tokens = self._patch_embed(images)
            batch = image_tokens.size(0)
        else:
            if latents is None:
                raise ValueError("At least one of images or latents must be provided.")
            batch = latents.size(0) if latents.dim() > 1 else latents.size(0)
            image_tokens = self.image_placeholder.expand(batch, -1, -1)

        if latents is not None:
            if latents.dim() == 4:
                latents = latents.flatten(2).transpose(1, 2)
            elif latents.dim() == 3:
                pass
            else:
                raise ValueError("Latents must have shape (B,C,H,W) or (B,N,D).")
            latents = self.latent_proj(latents.to(device))
        else:
            latents = self.latent_placeholder.expand(batch, -1, -1)

        image_tokens = image_tokens.to(device) + self.image_pos_embed
        latent_tokens = latents + self.latent_pos_embed

        tokens = torch.cat([image_tokens, latent_tokens], dim=1)

        if timesteps is None:
            timesteps = torch.ones(batch, device=device)
        else:
            timesteps = timesteps.to(device)

        time_condition = self.time_embed(timesteps).unsqueeze(1)
        tokens = tokens + time_condition

        hidden_states = self.mae.vit.encoder(tokens).last_hidden_state
        hidden_states = self.mae.vit.layernorm(hidden_states)

        image_states = hidden_states[:, : self.image_tokens, :]
        latent_states = hidden_states[:, self.image_tokens :, :]

        def to_grid(seq: torch.Tensor) -> torch.Tensor:
            grid = seq.transpose(1, 2).reshape(seq.size(0), self.embed_dim, self.latent_grid, self.latent_grid)
            return grid

        if images is not None and latents is None:
            return to_grid(image_states)

        return to_grid(latent_states)


__all__ = ["DualMAEEncoder"]
