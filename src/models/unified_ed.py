"""
Utilities for combining the dual-mode MAE encoder with the Stage 1 decoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .dual_mae_encoder import DualMAEEncoder


@dataclass
class Stage1Stats:
    mean: torch.Tensor
    std: torch.Tensor
    input_size: int
    patch_size: int
    reshape_to_2d: bool


class UnifiedEncoderDecoder(nn.Module):
    """
    Wrapper that bundles the dual-mode encoder and Stage 1 decoder/normalisation stats.
    """

    def __init__(
        self,
        encoder: DualMAEEncoder,
        decoder: nn.Module,
        stats: Stage1Stats,
        latent_dim: int,
        latent_grid: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.register_buffer("encoder_mean", stats.mean, persistent=False)
        self.register_buffer("encoder_std", stats.std, persistent=False)
        self.input_size = stats.input_size
        self.patch_size = stats.patch_size
        self.reshape_to_2d = stats.reshape_to_2d
        self.latent_dim = latent_dim
        self.latent_grid = latent_grid

    def _normalise(self, images: torch.Tensor) -> torch.Tensor:
        return (images - self.encoder_mean) / self.encoder_std

    def encode_image(self, images: torch.Tensor, timesteps: Optional[torch.Tensor] = None) -> torch.Tensor:
        images = nn.functional.interpolate(images, size=self.input_size, mode="bicubic", align_corners=False)
        normed = self._normalise(images)
        return self.encoder(images=normed, latents=None, timesteps=timesteps)

    def encode_latent(self, latents: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return self.encoder(images=None, latents=latents, timesteps=timesteps)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        z = latents
        if self.reshape_to_2d:
            b, c, h, w = z.shape
            z = z.view(b, c, h * w).transpose(1, 2)
        recon = self.decoder(z, drop_cls_token=False).logits
        recon = self.decoder.unpatchify(recon)
        recon = recon * self.encoder_std + self.encoder_mean
        return recon.clamp(0.0, 1.0)

    def forward_reconstruction(self, images: torch.Tensor, timesteps: Optional[torch.Tensor] = None) -> torch.Tensor:
        latents = self.encode_image(images, timesteps=timesteps)
        return self.decode(latents), latents

    def forward_diffusion(self, latents: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return self.encode_latent(latents, timesteps)


def build_unified_model(stage1, dual_encoder: DualMAEEncoder) -> UnifiedEncoderDecoder:
    stats = Stage1Stats(
        mean=stage1.encoder_mean,
        std=stage1.encoder_std,
        input_size=stage1.encoder_input_size,
        patch_size=stage1.encoder_patch_size,
        reshape_to_2d=stage1.reshape_to_2d,
    )
    latent_grid = int(stage1.base_patches ** 0.5)
    return UnifiedEncoderDecoder(
        encoder=dual_encoder,
        decoder=stage1.decoder,
        stats=stats,
        latent_dim=stage1.latent_dim,
        latent_grid=latent_grid,
    )


__all__ = ["UnifiedEncoderDecoder", "build_unified_model"]
