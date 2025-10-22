#!/usr/bin/env python3
"""
Phase 5 training script: alternate reconstruction & diffusion updates in one loop.

This script wires the Phase 4 unified model into a simple single-GPU trainer that:
  * Runs an MSE reconstruction loss on the decoder every other batch.
  * Runs the velocity-matching diffusion loss (via the existing transport utilities) on the
    diffusion transformer on alternating batches.
  * Keeps the MAE encoder frozen while updating decoder/diffusion with separate optimisers.

The training command expects the Stage 2 config YAML (it bundles both stage_1 and stage_2
sections). Example usage on a tiny fake dataset:

  python src/train_unified.py \\
      --config configs/stage2/training/ImageNet256/DiTDH-S_MAE-B.yaml \\
      --fake-data --epochs 1 --max-steps 10

For real training, omit ``--fake-data`` and provide ``--data-path`` pointing to an ImageFolder.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Tuple

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from omegaconf import OmegaConf

from stage2.transport import create_transport
from unified_model import UnifiedRAEModel
from utils.train_utils import parse_configs


def build_dataloader(
    data_path: Path | None,
    image_size: int,
    batch_size: int,
    num_workers: int,
    use_fake: bool,
) -> Tuple[DataLoader, int]:
    """
    Construct a DataLoader for ImageNet-style folder or a synthetic FakeData fallback.
    """
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    if use_fake:
        dataset = datasets.FakeData(
            size=batch_size * 10,
            image_size=(3, image_size, image_size),
            num_classes=1000,
            transform=transform,
        )
    else:
        if data_path is None or not data_path.exists():
            raise FileNotFoundError("data_path must point to an ImageFolder when not using --fake-data.")
        dataset = datasets.ImageFolder(str(data_path), transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader, len(dataset.classes) if hasattr(dataset, "classes") else dataset.num_classes


def should_stop(step: int, max_steps: int | None) -> bool:
    return max_steps is not None and step >= max_steps


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified reconstruction + diffusion training loop.")
    parser.add_argument("--config", required=True, help="YAML config containing stage_1 and stage_2 sections.")
    parser.add_argument("--data-path", type=Path, help="ImageFolder root for ImageNet 256x256 training images.")
    parser.add_argument("--fake-data", action="store_true", help="Use torchvision FakeData for sanity checks.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to iterate.")
    parser.add_argument("--max-steps", type=int, default=None, help="Early-stop after this many total steps.")
    parser.add_argument("--batch-size", type=int, default=16, help="Per-iteration batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="Data loader worker processes.")
    parser.add_argument("--image-size", type=int, default=256, help="Input resolution fed to the encoder.")
    parser.add_argument("--lr-recon", type=float, default=2e-4, help="Learning rate for the decoder optimiser.")
    parser.add_argument("--lr-diff", type=float, default=2e-4, help="Learning rate for the diffusion optimiser.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Override training device.")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("results/unified"), help="Directory to save checkpoints.")
    parser.add_argument("--save-every", type=int, default=1000, help="Checkpoint frequency in steps.")
    args = parser.parse_args()

    device = torch.device(args.device)

    (
        stage1_cfg,
        stage2_cfg,
        transport_cfg,
        _sampler_cfg,
        _guidance_cfg,
        misc_cfg,
        _training_cfg,
    ) = parse_configs(args.config)

    repo_root = Path(args.config).resolve().parent.parent.parent.parent
    os.chdir(repo_root)  # decoder config paths (e.g. configs/decoder/ViTXL) resolve relative to repo root.

    model = UnifiedRAEModel.from_configs(stage1_cfg, stage2_cfg).to(device)
    model.decoder.train()
    model.diffusion.train()

    decoder_opt = optim.Adam(model.decoder.parameters(), lr=args.lr_recon, betas=(0.5, 0.9))
    diffusion_opt = optim.Adam(model.diffusion.parameters(), lr=args.lr_diff, betas=(0.9, 0.95))

    transport_params = OmegaConf.to_container(
        transport_cfg.get("params", {}), resolve=True
    ) if transport_cfg is not None else {}
    misc_params = OmegaConf.to_container(misc_cfg, resolve=True) if misc_cfg is not None else {}
    shift_dim = misc_params.get("time_dist_shift_dim", 196_608)
    shift_base = misc_params.get("time_dist_shift_base", 4096)
    time_dist_shift = (shift_dim / shift_base) ** 0.5
    transport = create_transport(
        **transport_params,
        time_dist_shift=time_dist_shift,
    )

    loader, num_classes = build_dataloader(
        data_path=args.data_path,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_fake=args.fake_data,
    )

    model.decoder.requires_grad_(True)
    model.diffusion.requires_grad_(True)

    step = 0
    reconstruction_turn = True
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if reconstruction_turn:
                decoder_opt.zero_grad(set_to_none=True)
                recon = model.forward_reconstruct(images)
                recon_loss = F.mse_loss(recon, images)
                recon_loss.backward()
                decoder_opt.step()
                diffusion_loss = None
                current_loss = recon_loss.item()
                loss_name = "recon"
            else:
                diffusion_opt.zero_grad(set_to_none=True)
                with torch.no_grad():
                    latents = model.encode(images)
                diff_loss = transport.training_losses(
                    model.diffusion,
                    latents,
                    model_kwargs={"y": labels},
                )["loss"].mean()
                diff_loss.backward()
                diffusion_opt.step()
                recon_loss = None
                current_loss = diff_loss.item()
                loss_name = "diff"

            reconstruction_turn = not reconstruction_turn
            step += 1

            if step % 50 == 0 or should_stop(step, args.max_steps):
                print(
                    f"[Epoch {epoch} | Step {step}] "
                    f"{loss_name}_loss={current_loss:.4f} "
                    f"(decoder_lr={decoder_opt.param_groups[0]['lr']:.2e}, "
                    f"diff_lr={diffusion_opt.param_groups[0]['lr']:.2e})"
                )

            if step % args.save_every == 0:
                ckpt = {
                    "decoder": model.decoder.state_dict(),
                    "diffusion": model.diffusion.state_dict(),
                    "decoder_opt": decoder_opt.state_dict(),
                    "diffusion_opt": diffusion_opt.state_dict(),
                    "step": step,
                    "epoch": epoch,
                }
                torch.save(ckpt, args.checkpoint_dir / f"unified_step_{step:07d}.pt")

            if should_stop(step, args.max_steps):
                break
        if should_stop(step, args.max_steps):
            break


if __name__ == "__main__":
    main()
