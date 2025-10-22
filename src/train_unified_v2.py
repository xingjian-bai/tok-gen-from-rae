#!/usr/bin/env python3
"""
Unified reconstruction + diffusion training with a dual-mode encoder (Phase 3).

This script alternates reconstruction and diffusion objectives per batch using a shared
encoder/decoder. It supports Weights & Biases logging, optional FakeData sanity runs, and
image visualisations (≥8 samples) at configurable intervals.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
from utils import wandb_utils

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms, utils as tv_utils
import math
import matplotlib.pyplot as plt

from omegaconf import OmegaConf

from disc.lpips import LPIPS  # type: ignore
from models.dual_mae_encoder import DualMAEEncoder
from models.unified_ed import UnifiedEncoderDecoder, build_unified_model
from utils.train_utils import parse_configs
from utils.model_utils import instantiate_from_config
from utils.diffusion_utils import cosine_alpha_sigma, velocity_target

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


def init_wandb(args, config: Dict, is_main: bool) -> None:
    if not args.wandb:
        return
    if wandb is None:
        raise ImportError("wandb is required for logging but is not installed.")
    if not is_main:
        return
    entity = os.environ.get("WANDB_ENTITY") or os.environ.get("ENTITY")
    project = os.environ.get("WANDB_PROJECT") or os.environ.get("PROJECT")
    key = os.environ.get("WANDB_KEY")
    missing = [name for name, val in [("WANDB_KEY", key), ("WANDB_ENTITY or ENTITY", entity), ("WANDB_PROJECT or PROJECT", project)] if val is None]
    if missing:
        raise EnvironmentError(f"Missing WandB environment variables: {', '.join(missing)}")
    wandb.login(key=key)
    wandb.init(entity=entity, project=project, config=config, name=args.run_name, resume="allow")


def build_dataloader(
    path: Optional[Path],
    batch_size: int,
    workers: int,
    image_size: int,
    fake_data: bool,
    distributed: bool,
    world_size: int,
    rank: int,
) -> Tuple[DataLoader, Optional[DistributedSampler]]:
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    if fake_data:
        dataset = datasets.FakeData(size=batch_size * 10, image_size=(3, image_size, image_size), num_classes=1000, transform=transform)
    else:
        if path is None:
            raise ValueError("data-path must be provided unless --fake-data is set.")
        dataset = datasets.ImageFolder(str(path), transform=transform)
    sampler: Optional[DistributedSampler]
    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        sampler = None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader, sampler


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified encoder-decoder training with alternating objectives.")
    parser.add_argument("--stage1-config", type=Path, required=True, help="Path to Stage 1 config (must include ckpt).")
    parser.add_argument("--stage2-config", type=Path, required=True, help="Path to Stage 2 config (for diffusion params).")
    parser.add_argument("--data-path", type=Path, help="ImageFolder root (ImageNet).")
    parser.add_argument("--output-dir", type=Path, default=Path("results/unified_phase3"), help="Directory for checkpoints/logs.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr-encoder", type=float, default=1e-4)
    parser.add_argument("--lr-decoder", type=float, default=1e-4)
    parser.add_argument("--lr-diffusion", type=float, default=2e-4)
    parser.add_argument("--recon-weight", type=float, default=1.0)
    parser.add_argument("--lpips-weight", type=float, default=None)
    parser.add_argument("--diffusion-weight", type=float, default=1.0)
    parser.add_argument("--noise-augment-std", type=float, default=0.05, help="Gaussian std for decoder input noise.")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--image-log-interval", type=int, default=None, help="Steps between image logs (defaults to 20 * log_interval).")
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in output dir.")
    parser.add_argument("--fake-data", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        rank = 0
        world_size = 1
        device = torch.device(args.device)

    is_main = rank == 0

    if is_main:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    os.chdir(Path(__file__).resolve().parent.parent)  # ensure relative paths resolved from repo root

    stage1_cfg, *_ = parse_configs(str(args.stage1_config))
    assert "ckpt" in stage1_cfg, "Stage1 config must contain a ckpt field pointing to the trained weights."
    stage1_dict = OmegaConf.to_container(stage1_cfg, resolve=True)
    stage1_cfg = OmegaConf.create(stage1_dict)
    stage1: nn.Module = instantiate_from_config(stage1_cfg).eval()
    if is_main:
        print(f"[Info] Stage1 checkpoint loaded: {stage1_dict['ckpt']}")
        print(f"[Info] Stage1 encoder trainable flag will be enabled in Phase 3.")

    dual_encoder = DualMAEEncoder(
        model_name=stage1_cfg["params"]["encoder_params"]["model_name"],
        image_size=stage1_cfg["params"]["encoder_input_size"],
        latent_tokens=stage1.base_patches,
        latent_dim=stage1.latent_dim,
        patch_size=stage1.encoder_patch_size,
    )
    dual_encoder.load_from_stage1(stage1.encoder.state_dict())
    if is_main:
        print("[Info] Dual-mode encoder initialised from Stage1 encoder weights.")

    base_model = build_unified_model(stage1, dual_encoder).to(device)
    base_model.encoder.train()
    base_model.decoder.train()
    if is_main:
        print(f"[Info] Encoder parameters trainable: {any(p.requires_grad for p in base_model.encoder.parameters())}")
        print("[Info] Decoder weights inherited from Stage1 decoder.")

   
    lpips_loss = LPIPS().to(device).eval()

    if distributed:
        ddp_model = torch.nn.parallel.DistributedDataParallel(
            base_model,
            device_ids=[device.index],
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    else:
        ddp_model = base_model

    model = ddp_model.module if distributed else ddp_model

    recon_opt = optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": args.lr_encoder},
            {"params": model.decoder.parameters(), "lr": args.lr_decoder},
        ],
        betas=(0.9, 0.95),
    )
    diff_opt = optim.AdamW(model.encoder.parameters(), lr=args.lr_diffusion, betas=(0.9, 0.95))

    loader, sampler = build_dataloader(
        args.data_path,
        args.batch_size,
        args.workers,
        model.input_size,
        args.fake_data,
        distributed,
        world_size,
        rank,
    )
    stage2_cfg, *_ = parse_configs(str(args.stage2_config))
    if stage2_cfg is not None and "misc" in stage2_cfg:
        misc_cfg = OmegaConf.to_container(stage2_cfg["misc"], resolve=True)
    else:
        misc_cfg = {}
    if is_main:
        print(f"[Info] Latent size: {model.latent_dim} dim, grid {model.latent_grid}x{model.latent_grid}")
        print("[Info] Using cosine alpha/sigma schedule for diffusion timesteps.")

    stage1_params = stage1_dict.get("params", {})
    stage1_noise_tau = float(stage1_params.get("noise_tau", 0.0))
    loss_cfg = stage1_dict.get("gan", {}).get("loss", {})
    lpips_start_epoch = int(loss_cfg.get("lpips_start", 0))
    perceptual_weight = float(args.lpips_weight if args.lpips_weight is not None else loss_cfg.get("perceptual_weight", 1.0))
    recon_weight = float(args.recon_weight if args.recon_weight is not None else 1.0)
    noise_std = stage1_noise_tau if args.noise_augment_std is None else args.noise_augment_std

    image_log_interval = args.image_log_interval if args.image_log_interval is not None else args.log_interval * 20

    config_for_wandb = {
        "lr_encoder": args.lr_encoder,
        "lr_decoder": args.lr_decoder,
        "lr_diffusion": args.lr_diffusion,
        "recon_weight": recon_weight,
        "lpips_weight": perceptual_weight,
        "diffusion_weight": args.diffusion_weight,
        "noise_augment_std": noise_std,
        "lpips_start_epoch": lpips_start_epoch,
        "image_log_interval": image_log_interval,
        "latent_dim": model.latent_dim,
        "latent_grid": model.latent_grid,
        "data_path": str(args.data_path) if args.data_path else "fake",
    }
    if args.run_name is None:
        args.run_name = f"phase3-{Path(args.stage1_config).stem}"
    init_wandb(args, config_for_wandb, is_main)

    start_epoch = 0
    step = 0
    latest_path = ""
    if args.resume:
        if is_main:
            ckpts = sorted(args.output_dir.glob("phase3_step_*.pt"))
            latest_path = str(ckpts[-1]) if ckpts else ""
        if distributed:
            obj = [latest_path]
            dist.broadcast_object_list(obj, src=0)
            latest_path = obj[0]
        if latest_path:
            ckpt = torch.load(latest_path, map_location="cpu")
            model.load_state_dict(ckpt["model"])
            recon_opt.load_state_dict(ckpt["recon_opt"])
            diff_opt.load_state_dict(ckpt["diff_opt"])
            step = ckpt.get("step", 0)
            start_epoch = ckpt.get("epoch", 0)
            if is_main:
                print(f"[Info] Resumed from {latest_path} (epoch={start_epoch}, step={step}).")
    if distributed:
        dist.barrier()

    total_batches = len(loader)
    if is_main:
        print(f"[Info] Batches per epoch: {total_batches}")

    lpips_start_step = lpips_start_epoch * total_batches

    if is_main:
        t_vals = torch.linspace(0.0, 0.999, steps=1000)
        alpha_vals, sigma_vals = cosine_alpha_sigma(t_vals)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(t_vals.cpu().numpy(), alpha_vals.cpu().numpy(), label="alpha(t)")
        ax.plot(t_vals.cpu().numpy(), sigma_vals.cpu().numpy(), label="sigma(t)")
        ax.set_xlabel("t")
        ax.set_ylabel("value")
        ax.set_title("Cosine Alpha/Sigma Schedule")
        ax.legend()
        plot_path = args.output_dir / "alpha_sigma_curve.png"
        fig.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)
        if args.wandb and wandb is not None:
            wandb.log({"plots/alpha_sigma": wandb.Image(str(plot_path))}, step=0)

    starting_step = step
    for epoch in range(start_epoch, args.epochs):
        epoch_recon_sum = 0.0
        epoch_lpips_sum = 0.0
        epoch_total_sum = 0.0
        epoch_diff_sum = 0.0
        epoch_batches = 0
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch_images, batch_labels in loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            # Reconstruction step
            recon_opt.zero_grad(set_to_none=True)
            latents_clean = model.encode_image(batch_images, timesteps=torch.ones(batch_images.size(0), device=device))
            if step == 0:
                assert latents_clean.shape[1:] == (model.latent_dim, model.latent_grid, model.latent_grid), \
                    f"Unexpected latent shape {latents_clean.shape}"
                if is_main:
                    print(f"[Debug] Latent shape confirmed: {latents_clean.shape}")
            latents_base = latents_clean
            if noise_std > 0:
                sigma = noise_std * torch.rand(latents_base.size(0), device=device).view(-1, 1, 1, 1)
                latents_noisy = latents_base + sigma * torch.randn_like(latents_base)
            else:
                latents_noisy = latents_base
            recon_image = model.decode(latents_noisy)
            recon_l1 = F.l1_loss(recon_image, batch_images) * recon_weight
            use_lpips = step >= lpips_start_step and perceptual_weight > 0
            lpips_val = lpips_loss(recon_image, batch_images).mean() if use_lpips else torch.zeros((), device=device)
            total_recon = recon_l1 + perceptual_weight * lpips_val
            total_recon.backward()
            recon_opt.step()

            # Diffusion step
            diff_opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                latents_detached = latents_base.detach()
            t = torch.rand(batch_images.size(0), device=device).clamp(0.001, 0.999)
            alpha, sigma = cosine_alpha_sigma(t)
            noise = torch.randn_like(latents_detached)
            z_noisy = alpha.view(-1, 1, 1, 1) * latents_detached + sigma.view(-1, 1, 1, 1) * noise
            pred_velocity = model.forward_diffusion(z_noisy, t)
            target_velocity = velocity_target(latents_detached, noise, alpha, sigma)
            diff_loss = F.mse_loss(pred_velocity, target_velocity) * args.diffusion_weight
            diff_loss.backward()
            diff_opt.step()

            step += 1
            epoch_batches += 1
            epoch_recon_sum += recon_l1.item()
            epoch_lpips_sum += lpips_val.item()
            epoch_total_sum += total_recon.item()
            epoch_diff_sum += diff_loss.item()
            if (step == starting_step + 1 or step % args.log_interval == 0 or (args.max_steps and step >= args.max_steps)) and is_main:
                msg = (
                    f"[Epoch {epoch} Step {step}] "
                    f"loss/total={total_recon.item():.4f} "
                    f"loss/recon={recon_l1.item():.4f} loss/lpips={lpips_val.item():.4f} "
                    f"loss/diffusion={diff_loss.item():.4f} "
                    f"p_weight={perceptual_weight}"
                )
                print(msg)
                if args.wandb and wandb is not None:
                    enc_grad = math.sqrt(
                        sum(
                            (p.grad.detach().pow(2).sum().item() if p.grad is not None else 0.0)
                            for p in model.encoder.parameters()
                        )
                    )
                    wandb.log(
                        {
                            "loss/total": total_recon.item(),
                            "loss/recon": recon_l1.item(),
                            "loss/lpips": lpips_val.item(),
                            "loss/diffusion": diff_loss.item(),
                            "grad/encoder": enc_grad,
                            "perceptual_weight": perceptual_weight,
                            "metrics/lpips_active": float(use_lpips),
                            "metrics/image_placeholder_norm": base_model.encoder.image_placeholder.norm().item(),
                            "metrics/latent_placeholder_norm": base_model.encoder.latent_placeholder.norm().item(),
                            "step": step,
                            "epoch": epoch,
                        },
                        step=step,
                    )

            if (step == starting_step + 1 or step % image_log_interval == 0) and is_main:
                diffusion_preds: Dict[float, Tuple[torch.Tensor, torch.Tensor]] = {}
                with torch.no_grad():
                    diffusion_levels = [
                        0.0,
                        0.05,
                        0.1,
                        0.2,
                        0.3,
                        0.4,
                        0.5,
                        0.6,
                        0.7,
                        0.8,
                        0.9,
                        1.0,
                    ]
                    for t_val in diffusion_levels:
                        t_level = torch.full((latents_base.size(0),), t_val, device=device)
                        alpha_level, sigma_level = cosine_alpha_sigma(t_level)
                        noise_level = torch.randn_like(latents_base)
                        z_noisy_level = alpha_level.view(-1, 1, 1, 1) * latents_base + sigma_level.view(-1, 1, 1, 1) * noise_level
                        pred_velocity_level = model.forward_diffusion(z_noisy_level, t_level)
                        pred_latents_level = (
                            alpha_level.view(-1, 1, 1, 1) * z_noisy_level
                            - sigma_level.view(-1, 1, 1, 1) * pred_velocity_level
                        )
                        direct_decode = model.decode(z_noisy_level)
                        pred_decode = model.decode(pred_latents_level)
                        diffusion_preds[t_val] = (direct_decode, pred_decode)
                k_vis = min(8, batch_images.size(0))
                recon_tensor = torch.cat([batch_images[:k_vis].detach().cpu(), recon_image[:k_vis].detach().cpu()], dim=0)
                wandb_utils.log_image(recon_tensor, step=step, nrow=k_vis, name_suffix="_recon")
                gt_cpu = batch_images[:k_vis].detach().cpu()
                recon_cpu = recon_image[:k_vis].detach().cpu()
                for t_val, (direct_img, pred_img) in diffusion_preds.items():
                    direct_cpu = direct_img[:k_vis].detach().cpu()
                    pred_cpu = pred_img[:k_vis].detach().cpu()
                    vis_tensor = torch.cat(
                        [
                            gt_cpu,
                            recon_cpu,
                            direct_cpu,
                            pred_cpu,
                        ],
                        dim=0,
                    )
                    wandb_utils.log_image(
                        vis_tensor,
                        step=step,
                        nrow=k_vis,
                        name_suffix=f"_diffusion_{t_val:.2f}",
                    )

            if step % args.save_interval == 0 and is_main:
                ckpt = {
                    "model": model.state_dict(),
                    "recon_opt": recon_opt.state_dict(),
                    "diff_opt": diff_opt.state_dict(),
                    "step": step,
                    "epoch": epoch,
                }
                torch.save(ckpt, args.output_dir / f"phase3_step_{step:07d}.pt")

            if args.max_steps and step >= args.max_steps:
                if is_main:
                    print("[Info] Max steps reached; exiting training loop.")
                break
        if args.max_steps and step >= args.max_steps:
            break
        if is_main and epoch_batches > 0:
            avg_recon = epoch_recon_sum / epoch_batches
            avg_lpips = epoch_lpips_sum / epoch_batches
            avg_total = epoch_total_sum / epoch_batches
            avg_diff = epoch_diff_sum / epoch_batches
            print(
                f"[Epoch {epoch}] "
                f"epoch/loss_total={avg_total:.4f} epoch/loss_recon={avg_recon:.4f} "
                f"epoch/loss_lpips={avg_lpips:.4f} epoch/loss_diffusion={avg_diff:.4f}"
            )
            if args.wandb and wandb is not None:
                wandb.log(
                    {
                        "epoch/loss_total": avg_total,
                        "epoch/loss_recon": avg_recon,
                        "epoch/loss_lpips": avg_lpips,
                        "epoch/loss_diffusion": avg_diff,
                        "metrics/image_placeholder_norm": base_model.encoder.image_placeholder.norm().item(),
                        "metrics/latent_placeholder_norm": base_model.encoder.latent_placeholder.norm().item(),
                        "epoch": epoch,
                    },
                    step=step,
                )

    if args.wandb and wandb is not None and is_main:
        wandb.finish()

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
