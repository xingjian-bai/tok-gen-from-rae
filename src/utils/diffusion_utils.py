"""
Helper functions for continuous-time diffusion losses on latents.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch


def cosine_alpha_sigma(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Cosine schedule mapping t in [0, 1] -> (alpha, sigma).

    alpha(t) = cos(pi/2 * t)
    sigma(t) = sin(pi/2 * t)
    """
    t = torch.clamp(t, 0.0, 1.0)
    alpha = torch.cos(t * math.pi / 2)
    sigma = torch.sin(t * math.pi / 2)
    return alpha, sigma


def velocity_target(z: torch.Tensor, noise: torch.Tensor, alpha: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """
    Compute velocity target v = alpha * noise - sigma * z.
    """
    reshape = (-1,) + (1,) * (z.dim() - 1)
    alpha = alpha.view(*reshape)
    sigma = sigma.view(*reshape)
    return alpha * noise - sigma * z


__all__ = ["cosine_alpha_sigma", "velocity_target"]
