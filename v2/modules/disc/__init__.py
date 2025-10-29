import torch

from .diffaug import DiffAug
from .discriminator import DinoDiscriminator
from .gan_loss import hinge_d_loss, vanilla_d_loss, vanilla_g_loss
from .lpips import LPIPS


def calculate_adaptive_weight(
    recon_loss: torch.Tensor,
    gan_loss: torch.Tensor,
    layer: torch.nn.Parameter,
    max_d_weight: float = 1e4,
) -> torch.Tensor:
    recon_grads = torch.autograd.grad(recon_loss, layer, retain_graph=True)[0]
    gan_grads = torch.autograd.grad(gan_loss, layer, retain_graph=True)[0]
    d_weight = torch.norm(recon_grads) / (torch.norm(gan_grads) + 1e-6)
    d_weight = torch.clamp(d_weight, 0.0, max_d_weight)
    return d_weight.detach()


def select_gan_losses(disc_kind: str, gen_kind: str):
    if disc_kind == "hinge":
        disc_loss_fn = hinge_d_loss
    elif disc_kind == "vanilla":
        disc_loss_fn = vanilla_d_loss
    else:
        raise ValueError(f"Unsupported discriminator loss '{disc_kind}'")

    if gen_kind == "vanilla":
        gen_loss_fn = vanilla_g_loss
    else:
        raise ValueError(f"Unsupported generator loss '{gen_kind}'")
    return disc_loss_fn, gen_loss_fn


def build_discriminator(
    config: dict,
    device: torch.device,
) -> tuple[DinoDiscriminator, DiffAug]:
    """Instantiate Dino-based discriminator and its augmentation policy."""
    arch_cfg = config.get("arch", {})
    ckpt_path = arch_cfg.get("dino_ckpt_path")
    if not ckpt_path:
        raise ValueError("DINO discriminator requires 'dino_ckpt_path' in gan.disc.arch.")
    disc = DinoDiscriminator(
        device=device,
        dino_ckpt_path=ckpt_path,
        ks=int(arch_cfg.get("ks", 3)),
        key_depths=tuple(arch_cfg.get("key_depths", (2, 5, 8, 11))),
        norm_type=arch_cfg.get("norm_type", "bn"),
        using_spec_norm=bool(arch_cfg.get("using_spec_norm", True)),
        norm_eps=float(arch_cfg.get("norm_eps", 1e-6)),
        recipe=arch_cfg.get("recipe", "S_8"),
    ).to(device)

    aug_cfg = config.get("augment", {})
    augment = DiffAug(prob=float(aug_cfg.get("prob", 1.0)), cutout=float(aug_cfg.get("cutout", 0.0)))
    return disc, augment


__all__ = [
    "LPIPS",
    "DiffAug",
    "DinoDiscriminator",
    "hinge_d_loss",
    "vanilla_d_loss",
    "vanilla_g_loss",
    "build_discriminator",
]
