import torch
import wandb
from einops import rearrange
from typing import Dict
from torch.cuda.amp import autocast
from collections import defaultdict
from utils import wandb_utils
from utils.train_utils import save_checkpoint, MetricLogger, SmoothedValue

@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, current_model: torch.nn.Module, decay: float) -> None:
    ema_params = dict(ema_model.named_parameters())
    model_params = dict(current_model.named_parameters())
    for name, param in model_params.items():
        if name in ema_params:
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def train_one_epoch(
        model, ema, ema_decay,
        loader, optimizer, disc_optimizer, scheduler, disc_scheduler,
        scaler, clip_grad, autocast_kwargs,
        epoch, global_step, device,
        logger, log_interval, 
        checkpoint_dir, checkpoint_interval, visualization_interval, wandb_interval
    ):
    
    model.train()
    ema.eval()
    
    # metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # header = 'Epoch: [{}]'.format(epoch)
    # print_freq = 20
    
    epoch_metrics: Dict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(1, device=device))
    num_batches = 0
    steps_per_epoch = len(loader)

    gan_cfg = model.module.gan_cfg
    loss_cfg = gan_cfg.get("loss", {})
    perceptual_weight = float(loss_cfg.get("perceptual_weight", 0.0))
    disc_weight = float(loss_cfg.get("disc_weight", 0.0))
    gan_start_epoch = int(loss_cfg.get("disc_start", 0))
    disc_update_epoch = int(loss_cfg.get("disc_upd_start", gan_start_epoch))
    lpips_start_epoch = int(loss_cfg.get("lpips_start", 0))
    disc_updates = int(loss_cfg.get("disc_updates", 1))

    gan_start_step = gan_start_epoch * steps_per_epoch
    disc_update_step = disc_update_epoch * steps_per_epoch
    lpips_start_step = lpips_start_epoch * steps_per_epoch

    for step, (images, _) in enumerate(loader): # metric_logger.log_every(loader, print_freq, header)
        use_gan = global_step >= gan_start_step and disc_weight > 0.0
        train_disc = global_step >= disc_update_step and disc_weight > 0.0
        use_lpips = global_step >= lpips_start_step and perceptual_weight > 0.0
        
        images = images.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        optimizer_idx = "generator"
        
        with autocast(**autocast_kwargs):
            total_loss, output_dict = \
                model(
                    images, labels=None, 
                    use_gan=use_gan, use_lpips=use_lpips, 
                    optimizer_idx=optimizer_idx, only_reconstruction=False
                )

        if scaler:
            scaler.scale(total_loss).backward()
            if clip_grad is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        update_ema(ema, model.module, ema_decay)

        epoch_metrics["recon"] += output_dict["rec_loss"].detach()
        epoch_metrics["lpips"] += output_dict["lpips_loss"].detach()
        epoch_metrics["gan"] += output_dict["gan_loss"].detach()
        epoch_metrics["total"] += total_loss.detach()
        epoch_metrics["diffusion"] += output_dict["diffusion_loss"].detach()
        num_batches += 1
        epoch_metrics["num_batches"] = num_batches
        
        ## not doing metric logging...
        # metric_log = {
        #     "recon": output_dict["rec_loss"].item(),
        #     "lpips": output_dict["lpips_loss"].item(),
        #     "total": total_loss.item(),
        # }
        # if "gan_loss" in output_dict: metric_log["gan"] = output_dict["gan_loss"].item()
        # metric_logger.update(**metric_log)
        # lr = optimizer.param_groups[0]["lr"]
        # metric_logger.update(lr=lr)

        disc_metrics: Dict[str, torch.Tensor] = {}
        if train_disc:
            optimizer_idx = "discriminator"
            for _ in range(disc_updates):
                disc_optimizer.zero_grad(set_to_none=True)
                with autocast(**autocast_kwargs):
                    disc_loss, disc_metrics = \
                        model(
                            images, labels=None, 
                            use_gan=use_gan, use_lpips=use_lpips, 
                            optimizer_idx=optimizer_idx, only_reconstruction=True
                        )
                    if scaler:
                        scaler.scale(disc_loss).backward()
                        scaler.step(disc_optimizer)
                        scaler.update()
                    else:
                        disc_loss.backward()
                        disc_optimizer.step()
                    
                    if disc_scheduler is not None:
                        disc_scheduler.step()
            
        if log_interval > 0 and global_step % log_interval == 0 and wandb_utils.is_main_process():
            stats = {
                "loss/total": total_loss.detach().item(),
                "loss/recon": output_dict["rec_loss"].detach().item(),
                "loss/lpips": output_dict["lpips_loss"].detach().item(),
                "loss/gan": output_dict["gan_loss"].detach().item(),
                "loss/diffusion": output_dict["diffusion_loss"].detach().item(),
                "gan/weight": output_dict["adaptive_weight"].detach().item(),
                "lr/generator": optimizer.param_groups[0]["lr"],
            }
            if disc_metrics:
                stats.update(
                    {
                        "loss/disc": disc_metrics["disc_loss"].detach().item(),
                        "disc/logits_real": disc_metrics["logits_real"].detach().item(),
                        "disc/logits_fake": disc_metrics["logits_fake"].detach().item(),
                        "lr/discriminator": disc_optimizer.param_groups[0]["lr"],
                    }
                )
            logger.info(
                f"[Epoch {epoch} | Step {global_step}] "
                + ", ".join(f"{k}: {v:.4f}" for k, v in stats.items())
            )

            if wandb_interval > 0 and global_step % wandb_interval == 0 and wandb_utils.is_main_process():
                wandb_utils.log(stats, step=global_step)

        if checkpoint_interval > 0 and global_step % checkpoint_interval == 0 and wandb_utils.is_main_process():
            ckpt_path = f"{checkpoint_dir}/{global_step:07d}.pt"
            save_checkpoint(
                ckpt_path,
                global_step,
                epoch,
                model,
                ema,
                optimizer,
                scheduler,
                disc_optimizer,
                disc_scheduler,
            )
        
        if visualization_interval > 0 and global_step % visualization_interval == 0 and wandb_utils.is_main_process():
            with torch.no_grad():
                optimizer_idx = "generator"
                total_loss, output_dict = \
                    ema(
                        images, labels=None, 
                        use_gan=use_gan, use_lpips=use_lpips, 
                        optimizer_idx=optimizer_idx, only_reconstruction=False
                    )
                wandb_utils.log_visuals(images, output_dict["recon"], global_step, wandb_log_key='train-input-reconstruction')
                for key in output_dict.keys():
                    if "reconstructed_imgs" in key:
                        wandb_utils.log_visuals(images, output_dict[key], global_step, wandb_log_key=key)

                ## Diffusion generation
                samples = ema.diffusion_generate(device=device)
                rows, cols = 4, 4
                vis_img = rearrange(samples[:rows*cols], '(r c) ch h w -> ch (r h) (c w)', r=rows, c=cols)
                vis_img = ((vis_img.clamp(-1, 1) + 1) / 2).cpu().detach()
                wandb_images = [wandb.Image(vis_img, caption="4x4 grid")]
                wandb.log({'train-input-generations': wandb_images}, step=global_step)


        global_step+=1

    return global_step, epoch_metrics