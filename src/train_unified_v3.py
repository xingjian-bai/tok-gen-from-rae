#!/usr/bin/env python3
"""
Unified reconstruction + diffusion training with a dual-mode encoder (Phase 3).

This script alternates reconstruction and diffusion objectives per batch using a shared
encoder/decoder. It supports Weights & Biases logging, optional FakeData sanity runs, and
image visualisations (≥8 samples) at configurable intervals.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
from utils import wandb_utils

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

from omegaconf import OmegaConf

from disc import (  # type: ignore
    DiffAug,
    build_discriminator,
    hinge_d_loss,
    vanilla_d_loss,
    vanilla_g_loss,
)
from disc.lpips import LPIPS  # type: ignore
from models.dual_mae_encoder import DualMAEEncoder
from models.unified_ed import UnifiedEncoderDecoder, build_unified_model
from utils.train_utils import parse_configs
from utils.model_utils import instantiate_from_config
from utils.optim_utils import build_optimizer, build_scheduler
from stage2.transport import create_transport, Sampler

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


def generate_samples(
    transport_sampler: Sampler,
    sampler_mode: str,
    sampler_defaults: Dict[str, Any],
    velocity_model,
    model: UnifiedEncoderDecoder,
    batch_size: int,
    latent_dim: int,
    latent_grid: int,
    device: torch.device,
    num_steps: int,
    bn_mean: Optional[torch.Tensor] = None,
    bn_std: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run Stage-2 style sampling for a given number of integration steps and decode
    both the final latent and a stack of intermediate latents for visualisation.
    """
    with torch.inference_mode():
        init = torch.randn(batch_size, latent_dim, latent_grid, latent_grid, device=device)
        if sampler_mode == "SDE":
            params = dict(sampler_defaults)
            params["num_steps"] = num_steps
            sample_fn = transport_sampler.sample_sde(**params)
        else:  # default to ODE sampling
            params = dict(sampler_defaults)
            params["num_steps"] = num_steps
            sample_fn = transport_sampler.sample_ode(**params)

        trajectory = sample_fn(init, velocity_model)
        if isinstance(trajectory, torch.Tensor):
            # torchdiffeq returns tensor shaped (steps, B, C, H, W)
            traj_list = [trajectory[i] for i in range(trajectory.shape[0])]
        else:
            traj_list = list(trajectory)

        final_latent = traj_list[-1]

        if bn_mean is None:
            bn_mean = torch.zeros(1, latent_dim, 1, 1, device=device)
        if bn_std is None:
            bn_std = torch.ones(1, latent_dim, 1, 1, device=device)
        bn_std = bn_std.clamp(min=1e-6)

        def _denorm(latent: torch.Tensor) -> torch.Tensor:
            return latent * bn_std + bn_mean

        decoded_final = model.decode(_denorm(final_latent))

        total_steps = len(traj_list)
        if total_steps <= 1:
            progress_latents = [final_latent]
        else:
            idx = torch.linspace(0, total_steps - 1, steps=min(10, total_steps), device=device)
            progress_latents = [traj_list[int(i.item())] for i in idx]

        progress_decoded = [model.decode(_denorm(lat)) for lat in progress_latents]
        if len(progress_decoded) == 1:
            progress_stack = progress_decoded[0].unsqueeze(0)
        else:
            progress_stack = torch.stack(progress_decoded, dim=0)

    return decoded_final, progress_stack


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified encoder-decoder training with alternating objectives.")
    parser.add_argument("--stage1-config", type=Path, required=True, help="Path to Stage 1 config (must include ckpt).")
    parser.add_argument("--stage2-config", type=Path, required=True, help="Path to Stage 2 config (for diffusion params).")
    parser.add_argument("--data-path", type=Path, help="ImageFolder root (ImageNet).")
    parser.add_argument("--output-dir", type=Path, default=Path("results/unified_phase3"), help="Directory for checkpoints/logs.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr-encoder", type=float, default=1e-4)
    parser.add_argument("--lr-decoder", type=float, default=1e-4)
    parser.add_argument("--lr-diffusion", type=float, default=2e-4)
    parser.add_argument("--recon-weight", type=float, default=1.0)
    parser.add_argument("--lpips-weight", type=float, default=None)
    parser.add_argument("--diffusion-weight", type=float, default=1.0)
    parser.add_argument("--diffusion-start-epoch", type=int, default=10, help="Delay diffusion training until this epoch (>=0).")
    parser.add_argument("--noise-augment-std", type=float, default=0.05, help="Gaussian std for decoder input noise.")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--image-log-interval", type=int, default=None, help="Steps between image logs (defaults to 20 * log_interval).")
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--sample-steps",
        type=int,
        nargs="*",
        default=None,
        help="Number of integration steps for unconditional sampling visualisations (Stage-II defaults are used when omitted).",
    )
    # EMA of batch normalization statistics for latents (used for offline sampling)
    parser.add_argument("--bn-ema-decay", type=float, default=0.99, help="EMA decay for per-channel latent mean/var tracking.")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in output dir.")
    parser.add_argument("--fake-data", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--freeze-decoder", action="store_true", help="Freeze decoder parameters and keep it in eval mode during training.")
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
    stage1_dict = OmegaConf.to_container(stage1_cfg, resolve=True)
    stage1_cfg = OmegaConf.create(stage1_dict)
    ckpt_path = stage1_dict.get("ckpt", None)
    stage1: nn.Module = instantiate_from_config(stage1_cfg).eval()
    if is_main:
        if ckpt_path:
            print(f"[Info] Stage1 checkpoint loaded: {ckpt_path}")
        else:
            print("[Info] Stage1 ckpt not provided; will use pretrained MAE weights for DualMAEEncoder.")
        print(f"[Info] Stage1 encoder trainable flag will be enabled in Phase 3.")

    dual_encoder = DualMAEEncoder(
        model_name=stage1_cfg["params"]["encoder_params"]["model_name"],
        image_size=stage1_cfg["params"]["encoder_input_size"],
        latent_tokens=stage1.base_patches,
        latent_dim=stage1.latent_dim,
        patch_size=stage1.encoder_patch_size,
    )
    # If Stage-1 checkpoint is provided, initialise from its encoder; otherwise keep pretrained MAE weights.
    if ckpt_path:
        dual_encoder.load_from_stage1(stage1.encoder.state_dict())
        if is_main:
            print("[Info] Dual-mode encoder initialised from Stage1 encoder weights.")
    else:
        if is_main:
            print("[Info] Dual-mode encoder initialised from HuggingFace MAE pretrained weights.")

    base_model = build_unified_model(stage1, dual_encoder).to(device)
    base_model.encoder.train()
    if args.freeze_decoder:
        base_model.decoder.eval()
        for p in base_model.decoder.parameters():
            p.requires_grad = False
    else:
        base_model.decoder.train()
    if is_main:
        print(f"[Info] Encoder parameters trainable: {any(p.requires_grad for p in base_model.encoder.parameters())}")
        print("[Info] Decoder weights inherited from Stage1 decoder.")
        print(f"[Info] Decoder frozen: {bool(args.freeze_decoder)}")

   
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
    decoder_last_layer = model.decoder.decoder_pred.weight

    # parameter counts (after potential freezing)
    enc_total_params = sum(p.numel() for p in model.encoder.parameters())
    dec_total_params = sum(p.numel() for p in model.decoder.parameters())
    enc_trainable_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    dec_trainable_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    if is_main:
        print(f"[Info] Encoder params: {enc_total_params/1e6:.2f}M (trainable {enc_trainable_params/1e6:.2f}M)")
        print(f"[Info] Decoder params: {dec_total_params/1e6:.2f}M (trainable {dec_trainable_params/1e6:.2f}M)")

    recon_param_groups = [
        {"params": model.encoder.parameters(), "lr": args.lr_encoder},
    ]
    if any(p.requires_grad for p in model.decoder.parameters()):
        recon_param_groups.append({"params": model.decoder.parameters(), "lr": args.lr_decoder})
    recon_opt = optim.AdamW(recon_param_groups, betas=(0.9, 0.95))
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
    if stage2_cfg is not None:
        stage2_dict = OmegaConf.to_container(stage2_cfg, resolve=True)
    else:
        stage2_dict = {}
    misc_cfg = stage2_dict.get("misc", {}) or {}
    transport_cfg = stage2_dict.get("transport", {}) or {}
    sampler_cfg = stage2_dict.get("sampler", {}) or {}
    guidance_cfg = stage2_dict.get("guidance", {}) or {}
    transport_params = dict(transport_cfg.get("params", {}))
    transport_params.pop("time_dist_shift", None)  # handled explicitly below
    default_shift_dim = int(misc_cfg.get("time_dist_shift_dim", model.latent_dim * model.latent_grid * model.latent_grid))
    default_shift_base = int(misc_cfg.get("time_dist_shift_base", 4096))
    time_dist_shift = math.sqrt(default_shift_dim / default_shift_base) if default_shift_base > 0 else 1.0
    transport = create_transport(**transport_params, time_dist_shift=time_dist_shift)
    transport_sampler = Sampler(transport)
    def velocity_model(x_t: torch.Tensor, t: torch.Tensor, **_kwargs: Any) -> torch.Tensor:
        # Prefer channel-first end-to-end. If transport provides (B,L,C), reshape to CF; else pass through.
        if x_t.dim() == 3:
            b, l, c = x_t.shape
            g = model.latent_grid
            assert l == g * g, "Token length must equal grid*grid"
            x_cf = x_t.transpose(1, 2).reshape(b, c, g, g)
            v_cf = model.forward_diffusion(x_cf, t)
            return v_cf.flatten(2).transpose(1, 2)
        else:
            return model.forward_diffusion(x_t, t)
    sampler_mode = str(sampler_cfg.get("mode", "ODE")).upper()
    sampler_params = dict(sampler_cfg.get("params", {}))
    if sampler_mode == "SDE":
        sampler_defaults: Dict[str, Any] = {
            "sampling_method": sampler_params.get("sampling_method", "euler"),
            "diffusion_form": sampler_params.get("diffusion_form", "SBDM"),
            "diffusion_norm": float(sampler_params.get("diffusion_norm", 1.0)),
            "last_step": sampler_params.get("last_step", "Mean"),
            "last_step_size": float(sampler_params.get("last_step_size", 0.04)),
        }
        sampler_base_steps = int(sampler_params.get("num_steps", 250))
    else:
        sampler_defaults = {
            "sampling_method": sampler_params.get("sampling_method", "dopri5"),
            "atol": float(sampler_params.get("atol", 1e-6)),
            "rtol": float(sampler_params.get("rtol", 1e-3)),
            "reverse": bool(sampler_params.get("reverse", False)),
        }
        sampler_base_steps = int(sampler_params.get("num_steps", 50))
    # Normalise sample steps now that Stage-II defaults are known.
    raw_sample_steps = args.sample_steps if args.sample_steps is not None else [sampler_base_steps]
    if isinstance(raw_sample_steps, (int, float)):
        raw_sample_steps = [raw_sample_steps]
    else:
        raw_sample_steps = list(raw_sample_steps)
    normalised_steps: List[int] = []
    for value in raw_sample_steps:
        try:
            step_int = int(value)
        except (TypeError, ValueError):
            continue
        if step_int > 0:
            normalised_steps.append(step_int)
    if not normalised_steps:
        normalised_steps = [sampler_base_steps]
    seen_steps: set[int] = set()
    ordered_sample_steps: List[int] = []
    for step_int in normalised_steps:
        if step_int not in seen_steps:
            seen_steps.add(step_int)
            ordered_sample_steps.append(step_int)
    args.sample_steps = ordered_sample_steps
    if is_main:
        print(f"[Info] Latent size: {model.latent_dim} dim, grid {model.latent_grid}x{model.latent_grid}")
        print(f"[Info] Stage-II transport: path={transport_params.get('path_type', 'Linear')}, prediction={transport_params.get('prediction', 'velocity')}, time_dist_shift={time_dist_shift:.4f}")

    stage1_params = stage1_dict.get("params", {})
    stage1_noise_tau = float(stage1_params.get("noise_tau", 0.0))
    # --- Read GAN config from FULL YAML (like Stage-1 does), fallback to Stage-2 if needed ---
    full_stage1_yaml = OmegaConf.load(str(args.stage1_config))
    stage1_gan_section = full_stage1_yaml.get("gan", None)
    if stage1_gan_section is None or not stage1_gan_section:
        try:
            full_stage2_yaml = OmegaConf.load(str(args.stage2_config))
            stage2_gan_section = full_stage2_yaml.get("gan", None)
        except Exception:
            stage2_gan_section = None
        gan_cfg = OmegaConf.to_container(stage2_gan_section, resolve=True) if stage2_gan_section is not None else {}
    else:
        gan_cfg = OmegaConf.to_container(stage1_gan_section, resolve=True)
    if not isinstance(gan_cfg, dict):
        gan_cfg = {}
    loss_cfg = gan_cfg.get("loss", {})
    lpips_start_epoch = int(loss_cfg.get("lpips_start", 0))
    perceptual_weight = float(args.lpips_weight if args.lpips_weight is not None else loss_cfg.get("perceptual_weight", 1.0))
    recon_weight = float(args.recon_weight if args.recon_weight is not None else 1.0)
    noise_std = stage1_noise_tau if args.noise_augment_std is None else args.noise_augment_std
    disc_cfg = gan_cfg.get("disc", {})
    disc_weight = float(loss_cfg.get("disc_weight", 0.0))
    gan_start_epoch = int(loss_cfg.get("disc_start", 0))
    disc_update_epoch = int(loss_cfg.get("disc_upd_start", gan_start_epoch))
    disc_updates = int(loss_cfg.get("disc_updates", 1))
    max_d_weight = float(loss_cfg.get("max_d_weight", 1e4))
    disc_loss_type = loss_cfg.get("disc_loss", "hinge")
    gen_loss_type = loss_cfg.get("gen_loss", "vanilla")

    image_log_interval = args.image_log_interval if args.image_log_interval is not None else args.log_interval * 20

    config_for_wandb = {
        "lr_encoder": args.lr_encoder,
        "lr_decoder": args.lr_decoder,
        "lr_diffusion": args.lr_diffusion,
        "recon_weight": recon_weight,
        "lpips_weight": perceptual_weight,
        "diffusion_weight": args.diffusion_weight,
        "diffusion_start_epoch": args.diffusion_start_epoch,
        "noise_augment_std": noise_std,
        "lpips_start_epoch": lpips_start_epoch,
        "image_log_interval": image_log_interval,
        "gan_weight": disc_weight,
        "latent_dim": model.latent_dim,
        "latent_grid": model.latent_grid,
        "data_path": str(args.data_path) if args.data_path else "fake",
        "sample_steps": args.sample_steps,
        "sampler_mode": sampler_mode,
        "time_dist_shift": time_dist_shift,
        "transport_path": transport_params.get("path_type", "Linear"),
        "transport_prediction": transport_params.get("prediction", "velocity"),
        "sampler_defaults": sampler_defaults,
        "freeze_decoder": bool(args.freeze_decoder),
        "params_encoder_total": int(enc_total_params),
        "params_decoder_total": int(dec_total_params),
        "params_encoder_trainable": int(enc_trainable_params),
        "params_decoder_trainable": int(dec_trainable_params),
    }
    # print(config_for_wandb)  # verbose debug disabled
    if args.run_name is None:
        args.run_name = f"phase3-{Path(args.stage1_config).stem}"
    init_wandb(args, config_for_wandb, is_main)

    start_epoch = 0
    step = 0
    latest_path = ""
    bn_resume_mean = None
    bn_resume_var = None
    resume_ckpt: Optional[Dict[str, Any]] = None  # stash checkpoint for GAN components
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
            try:
                recon_opt.load_state_dict(ckpt["recon_opt"])
            except (ValueError, RuntimeError) as e:
                if is_main:
                    print(f"[Warn] Failed to load recon optimizer state (likely param-group mismatch due to freeze flag): {e}")
            try:
                diff_opt.load_state_dict(ckpt["diff_opt"])
            except (ValueError, RuntimeError) as e:
                if is_main:
                    print(f"[Warn] Failed to load diffusion optimizer state: {e}")
            step = ckpt.get("step", 0)
            start_epoch = ckpt.get("epoch", 0)
            if "latent_bn_ema_mean" in ckpt:
                bn_resume_mean = ckpt["latent_bn_ema_mean"].to(device)
            if "latent_bn_ema_var" in ckpt:
                bn_resume_var = ckpt["latent_bn_ema_var"].to(device)
            resume_ckpt = ckpt  # keep full checkpoint so GAN states can be restored after discriminator is constructed
            if is_main:
                print(f"[Info] Resumed from {latest_path} (epoch={start_epoch}, step={step}).")
    if distributed:
        dist.barrier()

    total_batches = len(loader)
    if is_main:
        print(f"[Info] Batches per epoch: {total_batches}")
        print(f"[Info] Diffusion path warm-up: disabled until epoch >= {args.diffusion_start_epoch} (step >= {max(0, args.diffusion_start_epoch * total_batches)}).")

    lpips_start_step = lpips_start_epoch * total_batches
    gan_start_step = gan_start_epoch * total_batches
    disc_update_step = disc_update_epoch * total_batches
    diffusion_start_step = max(0, args.diffusion_start_epoch * total_batches)
    diffusion_enabled_logged = step >= diffusion_start_step
    diffusion_warmup_logged = False
    if diffusion_enabled_logged and is_main:
        print(f"[Info] Diffusion path already active on resume (current step={step}, start_step={diffusion_start_step}).")

    if disc_weight > 0:
        # Validate GAN discriminator config, provide helpful error if missing
        arch_cfg = disc_cfg.get("arch", {}) if isinstance(disc_cfg, dict) else {}
        if not arch_cfg or not arch_cfg.get("dino_ckpt_path"):
            raise ValueError(
                "GAN is enabled (disc_weight>0) but 'gan.disc.arch.dino_ckpt_path' is missing. "
                "Please set this in your YAML (top-level 'gan' section) or disable GAN by setting disc_weight=0."
            )
        discriminator, disc_aug = build_discriminator(disc_cfg, device)
        disc_params = [p for p in discriminator.parameters() if p.requires_grad]
        disc_optimizer, disc_optim_msg = build_optimizer(disc_params, disc_cfg)
        disc_scheduler = None
        if disc_cfg.get("scheduler"):
            disc_scheduler, disc_sched_msg = build_scheduler(disc_optimizer, total_batches, disc_cfg)
        else:
            disc_sched_msg = None
        disc_loss_fn, gen_loss_fn = (
            hinge_d_loss if disc_loss_type == "hinge" else vanilla_d_loss,
            vanilla_g_loss if gen_loss_type == "vanilla" else vanilla_g_loss,
        )
        if is_main:
            print(f"[Info] GAN enabled with weight {disc_weight}")
            print(f"[Info] Discriminator parameters: {sum(p.numel() for p in discriminator.parameters())/1e6:.2f}M")
            if disc_optim_msg:
                print(disc_optim_msg)
            if disc_sched_msg:
                print(disc_sched_msg)
        if resume_ckpt is not None:
            if "discriminator" in resume_ckpt:
                missing, unexpected = discriminator.load_state_dict(resume_ckpt["discriminator"], strict=False)
                if is_main and (missing or unexpected):
                    print(f"[Warn] Discriminator state mismatch on resume (missing={missing}, unexpected={unexpected}).")
            elif is_main:
                print("[Warn] Resume checkpoint lacks discriminator weights; GAN will warm start.")
            if "disc_optimizer" in resume_ckpt and disc_optimizer is not None:
                disc_optimizer.load_state_dict(resume_ckpt["disc_optimizer"])
            elif is_main and disc_optimizer is not None:
                print("[Warn] Resume checkpoint lacks discriminator optimizer state; optimizer reinitialised.")
            if "disc_scheduler" in resume_ckpt and disc_scheduler is not None:
                disc_scheduler.load_state_dict(resume_ckpt["disc_scheduler"])
            elif is_main and disc_scheduler is not None:
                print("[Warn] Resume checkpoint lacks discriminator scheduler state; scheduler reinitialised.")
            # free checkpoint dict to reduce host memory pressure
            resume_ckpt = None
    else:
        discriminator = None
        disc_optimizer = None
        disc_scheduler = None
        disc_loss_fn = None
        gen_loss_fn = None
        disc_aug = None
        if is_main:
            print("[Info] GAN disabled (disc_weight <= 0).")

    # if is_main:
    #     t_vals = torch.linspace(0.0, 0.999, steps=1000)
    #     alpha_vals, sigma_vals = cosine_alpha_sigma(t_vals)
    #     fig, ax = plt.subplots(figsize=(6, 4))
    #     ax.plot(t_vals.cpu().numpy(), alpha_vals.cpu().numpy(), label="alpha(t)")
    #     ax.plot(t_vals.cpu().numpy(), sigma_vals.cpu().numpy(), label="sigma(t)")
    #     ax.set_xlabel("t")
    #     ax.set_ylabel("value")
    #     ax.set_title("Cosine Alpha/Sigma Schedule")
    #     ax.legend()
    #     plot_path = args.output_dir / "alpha_sigma_curve.png"
    #     fig.savefig(plot_path, bbox_inches="tight")
    #     plt.close(fig)
    #     if args.wandb and wandb is not None:
    #         wandb.log({"plots/alpha_sigma": wandb.Image(str(plot_path))}, step=0)

    starting_step = step
    # interval aggregators for reduced-variance logging
    interval_sums: Dict[str, float] = {}
    interval_counts: Dict[str, int] = {}
    # persistent EMA stats for latent BN across training, saved in checkpoints
    bn_ema_mean = torch.zeros(1, model.latent_dim, 1, 1, device=device)
    bn_ema_var = torch.ones(1, model.latent_dim, 1, 1, device=device)
    if bn_resume_mean is not None:
        bn_ema_mean.copy_(bn_resume_mean)
    if bn_resume_var is not None:
        bn_ema_var.copy_(bn_resume_var)
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        epoch_recon_sum = 0.0
        epoch_lpips_sum = 0.0
        epoch_total_sum = 0.0
        epoch_diff_sum = 0.0
        epoch_gan_sum = 0.0
        epoch_batches = 0
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch_images, batch_labels in loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            real_normed = batch_images * 2.0 - 1.0

            # Reconstruction step
            recon_opt.zero_grad(set_to_none=True)
            latents_clean = model.encode_image(batch_images, timesteps=torch.ones(batch_images.size(0), device=device))
            if step == 0:
                assert latents_clean.shape[1:] == (model.latent_dim, model.latent_grid, model.latent_grid), \
                    f"Unexpected latent shape {latents_clean.shape}"
                if is_main:
                    print(f"[Debug] Latent shape confirmed: {latents_clean.shape}")
            latents_base = latents_clean

            # assert noise_std == 0, "Noise augmentation is not supported for native diffusion"
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
            recon_normed = recon_image * 2.0 - 1.0
            use_gan = disc_weight > 0 and step >= gan_start_step
            gan_loss_val = torch.zeros_like(total_recon)
            adaptive_weight = torch.zeros_like(total_recon)
            if use_gan:
                fake_aug = disc_aug.aug(recon_normed)
                logits_fake, _ = discriminator(fake_aug, None)
                gan_loss_val = gen_loss_fn(logits_fake)
                if args.freeze_decoder or not decoder_last_layer.requires_grad:
                    adaptive_weight = torch.ones_like(total_recon)
                else:
                    adaptive_weight = calculate_adaptive_weight(total_recon, gan_loss_val, decoder_last_layer, max_d_weight)
                total_loss = total_recon + disc_weight * adaptive_weight * gan_loss_val
            else:
                total_loss = total_recon
            total_loss.backward()
            total_loss_value = total_loss.detach().item()
            recon_opt.step()

            # Track latent statistics with an EMA so that sampling/visualisation can denormalise consistently later on.

            # Diffusion step (optional warm-up) using Stage-II transport
            with torch.no_grad():
                latents_detached = latents_base.detach()
            bn_mean_use = bn_ema_mean
            bn_std_use = bn_ema_var.sqrt().clamp(min=1e-6)
            train_diffusion = step >= diffusion_start_step
            diff_loss_value = 0.0
            if is_main and not train_diffusion and not diffusion_warmup_logged:
                print(
                    f"[Info] Diffusion branch still warming up (current step={step}, will enable at step {diffusion_start_step})."
                )
                diffusion_warmup_logged = True
            if train_diffusion:
                diff_opt.zero_grad(set_to_none=True)
                # Unconditional: pass channel-first latents directly (transport can handle CF)
                terms = transport.training_losses(velocity_model, latents_base, {})
                diff_loss = terms["loss"].mean() * args.diffusion_weight
                diff_loss.backward()
                diff_opt.step()
                diff_loss_value = diff_loss.detach().item()
                if is_main and not diffusion_enabled_logged:
                    print(f"[Info] Diffusion branch enabled at epoch {epoch}, step {step} (threshold step={diffusion_start_step}).")
                    diffusion_enabled_logged = True
            combined_loss_value = total_loss_value + diff_loss_value

            train_disc = disc_weight > 0 and step >= disc_update_step
            disc_metrics = None
            if train_disc:
                discriminator.train()
                for _ in range(disc_updates):
                    disc_optimizer.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        fake_detached = recon_normed.detach().clamp(-1.0, 1.0)
                        fake_detached = torch.round((fake_detached + 1.0) * 127.5) / 127.5 - 1.0
                        fake_input = disc_aug.aug(fake_detached)
                        real_input = disc_aug.aug(real_normed)
                    logits_fake, logits_real = discriminator(fake_input, real_input)
                    d_loss = disc_loss_fn(logits_real, logits_fake)
                    d_loss.backward()
                    disc_optimizer.step()
                    disc_metrics = {
                        "loss/disc": d_loss.detach(),
                        "disc/logits_real": logits_real.detach().mean(),
                        "disc/logits_fake": logits_fake.detach().mean(),
                    }
                discriminator.eval()
                if disc_scheduler is not None:
                    disc_scheduler.step()

            # accumulate per-step metrics
            def _acc(name: str, val: float) -> None:
                interval_sums[name] = interval_sums.get(name, 0.0) + float(val)
                interval_counts[name] = interval_counts.get(name, 0) + 1

            _acc("loss/total", combined_loss_value)
            _acc("loss/recon", recon_l1.item())
            _acc("loss/lpips", lpips_val.item())
            _acc("loss/diffusion", diff_loss_value)
            _acc("loss/gan", gan_loss_val.item())
            _acc("gan/weight", (disc_weight * adaptive_weight.item()) if use_gan else 0.0)
            enc_grad_val = math.sqrt(
                sum(
                    (p.grad.detach().pow(2).sum().item() if p.grad is not None else 0.0)
                    for p in model.encoder.parameters()
                )
            )
            _acc("grad/encoder", enc_grad_val)
            _acc("latent/bn_mean", bn_mean_use.mean().item())  # EMA mean (per-channel mean averaged)
            _acc("latent/bn_std", bn_std_use.mean().item())
            _acc("latent/bn_var", (bn_std_use * bn_std_use).mean().item())  # EMA var
            if disc_metrics is not None:
                _acc("loss/disc", disc_metrics["loss/disc"].item())
                _acc("disc/logits_real", disc_metrics["disc/logits_real"].item())
                _acc("disc/logits_fake", disc_metrics["disc/logits_fake"].item())

            step += 1
            epoch_batches += 1
            epoch_recon_sum += recon_l1.item()
            epoch_lpips_sum += lpips_val.item()
            epoch_total_sum += combined_loss_value
            epoch_diff_sum += diff_loss_value
            if disc_weight > 0:
                epoch_gan_sum += gan_loss_val.item()
            # update EMA BN stats every step (multi-GPU aggregated)
            with torch.no_grad():
                # compute global mean and variance via mean of squares
                m1 = latents_detached.mean(dim=(0, 2, 3), keepdim=True)
                m2 = (latents_detached * latents_detached).mean(dim=(0, 2, 3), keepdim=True)
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(m1, op=dist.ReduceOp.SUM)
                    dist.all_reduce(m2, op=dist.ReduceOp.SUM)
                    m1 /= world_size
                    m2 /= world_size
                v = (m2 - m1 * m1).clamp(min=0.0)
                decay = float(args.bn_ema_decay)
                bn_ema_mean.mul_(decay).add_(m1, alpha=1.0 - decay)
                bn_ema_var.mul_(decay).add_(v, alpha=1.0 - decay)
                # log batch stats alongside EMA stats
                _acc("latent/batch_mean", m1.mean().item())
                _acc("latent/batch_var", v.mean().item())

            should_log = (step == starting_step + 1) or (step % args.log_interval == 0) or (args.max_steps and step >= args.max_steps)
            if should_log:
                # compute distributed averages
                averaged: Dict[str, float] = {}
                for k, s in list(interval_sums.items()):
                    sum_tensor = torch.tensor(s, device=device, dtype=torch.float32)
                    cnt_tensor = torch.tensor(interval_counts.get(k, 0), device=device, dtype=torch.float32)
                    if dist.is_available() and dist.is_initialized():
                        dist.all_reduce(sum_tensor, op=dist.ReduceOp.SUM)
                        dist.all_reduce(cnt_tensor, op=dist.ReduceOp.SUM)
                    denom = max(cnt_tensor.item(), 1.0)
                    averaged[k] = (sum_tensor.item() / denom)

                if is_main:
                    msg = (
                        f"[Epoch {epoch} Step {step}] "
                        f"loss/total={averaged.get('loss/total', 0.0):.4f} "
                        f"loss/recon={averaged.get('loss/recon', 0.0):.4f} "
                        f"loss/lpips={averaged.get('loss/lpips', 0.0):.4f} "
                        f"loss/diffusion={averaged.get('loss/diffusion', 0.0):.4f} "
                        f"loss/gan={averaged.get('loss/gan', 0.0):.4f} "
                        f"p_weight={perceptual_weight} "
                        f"ema_mu={averaged.get('latent/bn_mean', 0.0):.4f} "
                        f"ema_var={averaged.get('latent/bn_var', 0.0):.4f} "
                        f"batch_mu={averaged.get('latent/batch_mean', 0.0):.4f} "
                        f"batch_var={averaged.get('latent/batch_var', 0.0):.4f}"
                    )
                    print(msg)
                    if args.wandb and wandb is not None:
                        payload = {
                            **averaged,
                            "perceptual_weight": perceptual_weight,
                            "metrics/lpips_active": float(use_lpips),
                            "metrics/diffusion_active": float(train_diffusion),
                            "metrics/image_placeholder_norm": base_model.encoder.image_placeholder.norm().item(),
                            "metrics/latent_placeholder_norm": base_model.encoder.latent_placeholder.norm().item(),
                            "latent/ema_mean": bn_ema_mean.mean().item(),
                            "latent/ema_std": bn_ema_var.sqrt().mean().item(),
                            "latent/ema_var": bn_ema_var.mean().item(),
                            "step": step,
                            "epoch": epoch,
                        }
                        wandb.log(payload, step=step)
                # reset interval
                interval_sums.clear()
                interval_counts.clear()

            if (step == starting_step + 1 or (train_diffusion and step % image_log_interval == 0)) and is_main:
                diffusion_preds: Dict[float, Tuple[torch.Tensor, torch.Tensor]] = {}
                with torch.no_grad():
                    # switch to eval() to avoid dropout randomness during visualization
                    was_training = model.training
                    model.eval()
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
                        base_t = torch.full((latents_base.size(0),), float(t_val), device=device, dtype=latents_base.dtype)
                        if time_dist_shift > 1.0:
                            shifted_t = time_dist_shift * base_t / (1.0 + (time_dist_shift - 1.0) * base_t)
                        else:
                            shifted_t = base_t
                        # Build x_t via transport path in channel-first
                        noise_cf = torch.randn_like(latents_base)
                        _, x_t_cf, _ = transport.path_sampler.plan(shifted_t, noise_cf, latents_base)
                        # Predict velocity in CF
                        pred_flow_cf = velocity_model(x_t_cf, shifted_t)
                        # Linear path clean latent: x1 = x_t + (1 - t) * v (only valid for Linear path)
                        clean_latents_cf = x_t_cf + (1.0 - shifted_t).view(-1, 1, 1, 1) * pred_flow_cf
                        # Decode direct noisy and denoised
                        direct_decode = model.decode(x_t_cf)
                        pred_decode = model.decode(clean_latents_cf)
                        diffusion_preds[t_val] = (direct_decode, pred_decode)
                        del noise_cf, x_t_cf, pred_flow_cf, clean_latents_cf
                    # restore mode
                    if was_training:
                        model.train()
                        if args.freeze_decoder:
                            model.decoder.eval()
                k_vis = min(8, batch_images.size(0))
                recon_tensor = torch.cat([batch_images[:k_vis].detach().cpu(), recon_image[:k_vis].detach().cpu()], dim=0)
                wandb_utils.log_image(
                    recon_tensor,
                    step=step,
                    nrow=k_vis,
                    name_suffix="_recon",
                    caption="Row1: GT, Row2: Reconstruction",
                )
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
                        caption="Rows: GT, Recon, Noisy decode, Denoised decode",
                    )
                    del vis_tensor, direct_cpu, pred_cpu
                del recon_tensor, gt_cpu, recon_cpu
                diffusion_preds.clear()

                for sample_step_count in args.sample_steps:
                    step_count = max(1, int(sample_step_count))
                    samples, sample_progress = generate_samples(
                        transport_sampler=transport_sampler,
                        sampler_mode=sampler_mode,
                        sampler_defaults=sampler_defaults,
                        velocity_model=velocity_model,
                        model=model,
                        batch_size=k_vis,
                        latent_dim=model.latent_dim,
                        latent_grid=model.latent_grid,
                        device=device,
                        num_steps=step_count,
                        bn_mean=bn_ema_mean,
                        bn_std=bn_ema_var.sqrt().clamp(min=1e-6),
                    )
                    samples_cpu = samples[:k_vis].detach().cpu()
                    progress_cpu = sample_progress[:, :k_vis].detach().cpu()
                    progress_rows = [row for row in progress_cpu]
                    progress_rows.append(samples_cpu)
                    stacked = torch.cat(progress_rows, dim=0)
                    wandb_utils.log_image(
                        stacked,
                        step=step,
                        nrow=k_vis,
                        name_suffix=f"_samples_steps{step_count:04d}",
                        caption=f"Steps={step_count}: rows show predicted clean latents from high-noise → low-noise (10 levels) plus final decode.",
                    )
                    del stacked, progress_rows, progress_cpu, samples_cpu, samples, sample_progress
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

            if step % args.save_interval == 0 and is_main:
                ckpt = {
                    "model": model.state_dict(),
                    "recon_opt": recon_opt.state_dict(),
                    "diff_opt": diff_opt.state_dict(),
                    "step": step,
                    "epoch": epoch,
                    # persist EMA latent BN stats for future offline sampling
                    "latent_bn_ema_mean": bn_ema_mean.detach().cpu(),
                    "latent_bn_ema_var": bn_ema_var.detach().cpu(),
                }
                if disc_weight > 0 and discriminator is not None and disc_optimizer is not None:
                    ckpt["discriminator"] = discriminator.state_dict()
                    ckpt["disc_optimizer"] = disc_optimizer.state_dict()
                    if disc_scheduler is not None:
                        ckpt["disc_scheduler"] = disc_scheduler.state_dict()
                torch.save(ckpt, args.output_dir / f"phase3_step_{step:07d}.pt")

            if args.max_steps and step >= args.max_steps:
                if is_main:
                    print("[Info] Max steps reached; exiting training loop.")
                break
        if args.max_steps and step >= args.max_steps:
            break
        if is_main and epoch_batches > 0:
            epoch_time = time.time() - epoch_start_time
            avg_recon = epoch_recon_sum / epoch_batches
            avg_lpips = epoch_lpips_sum / epoch_batches
            avg_total = epoch_total_sum / epoch_batches
            avg_diff = epoch_diff_sum / epoch_batches
            avg_gan = epoch_gan_sum / epoch_batches if disc_weight > 0 else 0.0
            print(
                f"[Epoch {epoch}] "
                f"epoch/loss_total={avg_total:.4f} epoch/loss_recon={avg_recon:.4f} "
                f"epoch/loss_lpips={avg_lpips:.4f} epoch/loss_diffusion={avg_diff:.4f} "
                f"epoch/loss_gan={avg_gan:.4f} epoch/time_sec={epoch_time:.1f}"
            )
            if args.wandb and wandb is not None:
                wandb.log(
                    {
                        "epoch/loss_total": avg_total,
                        "epoch/loss_recon": avg_recon,
                        "epoch/loss_lpips": avg_lpips,
                        "epoch/loss_diffusion": avg_diff,
                        "epoch/loss_gan": avg_gan,
                        "epoch/time_sec": epoch_time,
                        "metrics/diffusion_active": float(step >= diffusion_start_step),
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
