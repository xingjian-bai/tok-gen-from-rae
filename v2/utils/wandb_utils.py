import wandb
import torch
from torchvision.utils import make_grid
import torch.distributed as dist
from PIL import Image
import os
import argparse
import hashlib
import math
from einops import rearrange


def is_main_process():
    return dist.get_rank() == 0

def namespace_to_dict(namespace):
    return {
        k: namespace_to_dict(v) if isinstance(v, argparse.Namespace) else v
        for k, v in vars(namespace).items()
    }


def generate_run_id(exp_name):
    # https://stackoverflow.com/questions/16008670/how-to-hash-a-string-into-8-digits
    return str(int(hashlib.sha256(exp_name.encode('utf-8')).hexdigest(), 16) % 10 ** 8)


def initialize(args, entity, exp_name, project_name):
    config_dict = namespace_to_dict(args)
    wandb.login(key=os.environ["WANDB_KEY"])
    wandb.init(
        entity=entity,
        project=project_name,
        name=exp_name,
        config=config_dict,
        id=generate_run_id(exp_name),
        resume="allow",
    )


def log(stats, step=None):
    if is_main_process():
        wandb.log({k: v for k, v in stats.items()}, step=step)


def log_image(sample, step=None):
    if is_main_process():
        sample = array2grid(sample)
        wandb.log({f"samples": wandb.Image(sample), "train_step": step})


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x, nrow=nrow, normalize=True, value_range=(0,1))
    x = x.clamp(0, 1).mul(255).permute(1,2,0).to('cpu', torch.uint8).numpy()
    return x

def log_visuals(input_samples, output_samples, global_step, wandb_log_key, num_images_to_log=16):
    # Make sure num_images_to_log is factor of 4.
    input_imgs = input_samples[:num_images_to_log]
    reconstructed_imgs = output_samples[:num_images_to_log]
    reconstructed_imgs = reconstructed_imgs.clamp(0.0, 1.0)
    
    vis_img = torch.cat([input_imgs.detach().cpu(), reconstructed_imgs.detach().cpu()], dim=0)
    vis_img = rearrange(vis_img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=2)
    
    wandb_images = []
    wandb_images.append(wandb.Image(vis_img, caption="input; prediction;")) # (vis_img + 1) / 2
    wandb.log({wandb_log_key: wandb_images}, step=global_step)
