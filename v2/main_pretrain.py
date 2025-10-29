# Copyright (c) Meta Platforms.
# Licensed under the MIT license.
"""
Stage-1 RAE training script with reconstruction, LPIPS, and GAN losses.

This script adapts the training logic from the Kakao Brain VQGAN trainer while
targeting the RAE autoencoder architecture used in this repository.
"""

from __future__ import annotations

import argparse
import math
import os
import wandb
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP
from glob import glob

from omegaconf import OmegaConf
from utils import wandb_utils
from utils.train_utils import setup_distributed, load_checkpoint, prepare_dataloader, create_logger, cleanup_distributed
from utils.optim_utils import build_optimizer, build_scheduler, all_learnable_params
from models.gen_tok_2d import GenTok2D
from engines.pretrain import train_one_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage-1 RAE with GAN and LPIPS losses.")
    parser.add_argument("--config", type=str, required=True, help="YAML config containing a stage_1 section.")
    parser.add_argument("--data-path", type=Path, required=True, help="Directory with ImageFolder structure.")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to store training outputs.")
    parser.add_argument("--experiment-name", type=str, required=True, default=None, help="Directory to store training outputs.")
    parser.add_argument("--image-size", type=int, default=256, help="Image resolution (assumes square images).")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--global-seed", type=int, default=None, help="Override training.global_seed from the config.")    
    parser.add_argument("--ckpt", type=str, default=None, help="Optional checkpoint path to resume training.")
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases for logging if set.')
    return parser.parse_args()


def main():
    args = parse_args()
    rank, world_size, device = setup_distributed()

    full_cfg = OmegaConf.load(args.config)
    training_section = full_cfg.get("training", None)
    training_cfg = OmegaConf.to_container(training_section, resolve=True) if training_section is not None else {}
    training_cfg = dict(training_cfg) if isinstance(training_cfg, dict) else {}

    batch_size = int(training_cfg.get("batch_size", 16))
    num_workers = int(training_cfg.get("num_workers", 4))
    clip_grad_val = training_cfg.get("clip_grad", 1.0)
    clip_grad = float(clip_grad_val) if clip_grad_val is not None else None
    if clip_grad is not None and clip_grad <= 0:
        clip_grad = None
    log_interval = int(training_cfg.get("log_interval", 100))
    checkpoint_interval = int(training_cfg.get("checkpoint_interval", 1000))
    visualization_interval = int(training_cfg.get("visualization_interval", 1000))
    wandb_interval = int(training_cfg.get("wandb_interval", 1000))
    ema_decay = float(training_cfg.get("ema_decay", 0.9999))
    num_epochs = int(training_cfg.get("epochs", 200))
    default_seed = int(training_cfg.get("global_seed", 0))
    global_seed = args.global_seed if args.global_seed is not None else default_seed
    seed = global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_dir = os.path.join(args.results_dir, args.experiment_name)
        checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        # wandb.init(project="tok-gen-2D", group="imagenet100", name=args.experiment_name, config=args)
        wandb.login(host="https://api.wandb.ai", key="217ca03d641f6d7855af1710e3a107c01a1641f0")
        wandb.init(project="tok-gen", entity="latent-mage", group="imagenet100", name=args.experiment_name, config=args)

    else:
        experiment_dir = None
        checkpoint_dir = None
        logger = create_logger(None)

    
    gan_section = full_cfg.get("gan", None)
    gan_cfg = OmegaConf.to_container(gan_section, resolve=True) if gan_section is not None else {}
    if not gan_cfg:
        raise ValueError("Config must define a top-level 'gan' section for stage-1 training.")
    
    ddp_model = GenTok2D(gan_cfg, device).to(device)
    # Build the full ignore list of param & buffer names under "discriminator"
    ignore = [f"discriminator.{n}" for n, _ in ddp_model.discriminator.named_parameters()]
    ignore += [f"discriminator.{n}" for n, _ in ddp_model.discriminator.named_buffers()]
    ddp_model._ddp_params_and_buffers_to_ignore = ignore

    ema_model = deepcopy(ddp_model).to(device).eval()
    ema_model.requires_grad_(False)
    ddp_model = DDP(ddp_model, device_ids=[device.index], broadcast_buffers=False, find_unused_parameters=True)  # type: ignore[arg-type]
    model_woddp = ddp_model.module

    ## weight decay is 0 always, I always kept weight decay 0 only for bias and layernorm terms.
    optimizer, optim_msg = build_optimizer(model_woddp.get_learnable_params(no_disc=True), training_cfg)
    scheduler: LambdaLR | None = None
    sched_msg: Optional[str] = None
    
    disc_cfg = gan_cfg.get("disc", {})
    disc_optimizer, disc_optim_msg = build_optimizer(model_woddp.get_learnable_params(only_disc=True), disc_cfg)
    disc_scheduler: LambdaLR | None = None
    disc_sched_msg: Optional[str] = None

    scaler: GradScaler | None
    if args.precision == "fp16":
        scaler = GradScaler()
        autocast_kwargs = dict(enabled=True, dtype=torch.float16)
    elif args.precision == "bf16":
        scaler = None
        autocast_kwargs = dict(enabled=True, dtype=torch.bfloat16)
    else:
        scaler = None
        autocast_kwargs = dict(enabled=False)

    loader, sampler = prepare_dataloader(
        args.data_path, args.image_size, batch_size, num_workers, rank, world_size
    )
    steps_per_epoch = len(loader)
    if steps_per_epoch == 0:
        raise RuntimeError("Dataloader returned zero batches. Check dataset and batch size settings.")

    if training_cfg.get("scheduler"):
        scheduler, sched_msg = build_scheduler(optimizer, steps_per_epoch, training_cfg)

    if disc_cfg.get("scheduler"):
        disc_scheduler, disc_sched_msg = build_scheduler(disc_optimizer, steps_per_epoch, disc_cfg)

    start_epoch = 0
    global_step = 0
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
        if ckpt_path.is_file():
            start_epoch, global_step = load_checkpoint(
                ckpt_path,
                ddp_model,
                ema_model,
                optimizer,
                scheduler,
                disc_optimizer,
                disc_scheduler,
            )
            logger.info(f"[Rank {rank}] Resumed from {ckpt_path} (epoch={start_epoch}, step={global_step}).")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    if rank == 0:
        num_params = sum(p.numel() for p in ddp_model.parameters() if p.requires_grad)
        logger.info(f"Stage-1 RAE trainable parameters: {num_params/1e6:.2f}M")
        # logger.info(f"Discriminator architecture:\n{discriminator}")
        # num_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
        # logger.info(f"Discriminator trainable parameters: {num_params/1e6:.2f}M")
        # logger.info(f"Using {disc_loss_type} discriminator loss and {gen_loss_type} generator loss.")
        # logger.info(f"Perceptual (LPIPS) weight: {perceptual_weight:.6f}, GAN weight: {disc_weight:.6f}")
        # logger.info(f"GAN training starts at epoch {gan_start_epoch}, discriminator updates start at epoch {disc_update_epoch}, LPIPS loss starts at epoch {lpips_start_epoch}.")
        # if disc_aug is not None:
        #     logger.info(f"Using DiffAug with policies: {disc_aug}")
        # else:
        #     logger.info("Not using DiffAug.")
        if clip_grad is not None:
            logger.info(f"Clipping gradients to max norm {clip_grad}.")
        else:
            logger.info("Not clipping gradients.")
        # print optim and schel
        logger.info(optim_msg)
        print(sched_msg if sched_msg else "No LR scheduler for generator.")
        logger.info(disc_optim_msg)
        print(disc_sched_msg if disc_sched_msg else "No LR scheduler for discriminator.")
        logger.info(f"Training for {num_epochs} epochs, batch size {batch_size} per GPU.")
        logger.info(f"Dataset contains {len(loader.dataset)} samples, {steps_per_epoch} steps per epoch.")
        logger.info(f"Running with world size {world_size}, starting from epoch {start_epoch} to {num_epochs}.")

    for epoch in range(start_epoch, num_epochs):
        sampler.set_epoch(epoch)

        global_step, epoch_metrics = train_one_epoch(
            ddp_model, ema_model, ema_decay,
            loader, optimizer, disc_optimizer, scheduler, disc_scheduler,
            scaler, clip_grad, autocast_kwargs,
            epoch, global_step, device,
            logger, log_interval, 
            checkpoint_dir, checkpoint_interval, visualization_interval, wandb_interval
        )
        
        if rank == 0:
            num_batches = epoch_metrics["num_batches"]
            avg_recon = (epoch_metrics["recon"] / num_batches).item()
            avg_lpips = (epoch_metrics["lpips"] / num_batches).item()
            avg_gan = (epoch_metrics["gan"] / num_batches).item()
            avg_total = (epoch_metrics["total"] / num_batches).item()
            epoch_stats = {
                "epoch/loss_total": avg_total,
                "epoch/loss_recon": avg_recon,
                "epoch/loss_lpips": avg_lpips,
                "epoch/loss_gan": avg_gan,
            }
            logger.info(
                f"[Epoch {epoch}] "
                + ", ".join(f"{k}: {v:.4f}" for k, v in epoch_stats.items())
            )
            wandb_utils.log(epoch_stats, step=global_step)
    
    cleanup_distributed()


if __name__ == "__main__":
    main()
