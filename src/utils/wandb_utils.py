try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None  # type: ignore
import torch
from torchvision.utils import make_grid
import torch.distributed as dist
from PIL import Image
import os
import argparse
import hashlib
import math


def is_main_process():
    """Return True if this is rank 0 or if distributed is not initialized."""
    if not dist.is_available():
        return True
    if not dist.is_initialized():
        return True
    try:
        return dist.get_rank() == 0
    except Exception:
        return True

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
    if wandb is None:
        return
    if is_main_process():
        wandb.log({k: v for k, v in stats.items()}, step=step)


def log_image(sample, step=None, nrow=None, name_suffix="", caption=None):
    if wandb is None:
        return
    if is_main_process():
        sample = array2grid(sample, nrow=nrow)
        wandb.log({f"samples{name_suffix}": wandb.Image(sample, caption=caption), "train_step": step})


def array2grid(x, nrow=None):
    if nrow is None:
        nrow = round(math.sqrt(x.size(0)))
        nrow = max(1, nrow)
    x = make_grid(x, nrow=nrow, normalize=True, value_range=(0,1))
    x = x.clamp(0, 1).mul(255).permute(1,2,0).to('cpu', torch.uint8).numpy()
    return x
