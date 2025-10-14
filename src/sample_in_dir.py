# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained SiT.
"""
import torch.nn as nn
import math
from time import time
import argparse
from utils.model_utils import instantiate_from_config
from stage2.transport import create_transport, Sampler
from utils.train_utils import parse_configs
from stage1 import RAE
from torchvision.utils import save_image
import torch
import sys
import os
from stage2.models import Stage2ModelProtocol
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rae_config, model_config, transport_config, sampler_config, guidance_config, misc, _ = parse_configs(args.config)
    rae: RAE = instantiate_from_config(rae_config).to(device)
    model: Stage2ModelProtocol = instantiate_from_config(model_config).to(device)
    model.eval()  # important!
    rae.eval()
    shift_dim = misc.get("time_dist_shift_dim", 768 * 16 * 16)
    shift_base = misc.get("time_dist_shift_base", 4096)
    time_dist_shift = math.sqrt(
        shift_dim / shift_base)
    print(
        f"Using time_dist_shift={time_dist_shift:.4f} = sqrt({shift_dim}/{shift_base}).")
    transport = create_transport(
        **transport_config['params'],
        time_dist_shift=time_dist_shift
    )
    sampler = Sampler(transport)
    mode, sampler_params = sampler_config['mode'], sampler_config['params']
    if mode == "ODE":
        sample_fn = sampler.sample_ode(
            **sampler_params
        )
    elif mode == "SDE":
        sample_fn = sampler.sample_sde(
            **sampler_params,
            # sampling_method=args.sampling_method,
            # diffusion_form=args.diffusion_form,
            # diffusion_norm=args.diffusion_norm,
            # last_step=args.last_step,
            # last_step_size=args.last_step_size,
            # num_steps=args.num_sampling_steps,
        )
    else:
        raise NotImplementedError(f"Invalid sampling mode {mode}.")
    
    num_classes = int(misc.get("num_classes", 1000))
    latent_size = tuple(int(dim) for dim in misc.get("latent_size", (768, 16, 16)))
    if args.class_id < 0 or args.class_id >= num_classes:
        raise ValueError(f"class_id {args.class_id} is outside the valid range [0, {num_classes - 1}].")
    if args.num_samples <= 0:
        raise ValueError("--num_samples must be a positive integer.")

    # Labels to condition the model with:
    class_labels = [args.class_id] * args.num_samples

    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, *latent_size, device=device)
    y = torch.tensor(class_labels, device=device).long()
    
    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    null_label = int(misc.get("null_label", num_classes))
    y_null = torch.tensor([null_label] * n, device=device).long()
    y = torch.cat([y, y_null], 0)
    
    # set guidance setup
    guidance_scale = guidance_config.get("scale", 1.0)
    if guidance_scale > 1.0:
        t_min, t_max = guidance_config.get("t_min", 0.0), guidance_config.get("t_max", 1.0)
        model_kwargs = dict(y=y, cfg_scale=guidance_scale,
                            cfg_interval=(t_min, t_max))
        guidance_method = guidance_config.get("method", "cfg")
        if guidance_method == "autoguidance":
            guid_model_config = guidance_config.get("guidance_model", None)
            assert guid_model_config is not None, "Please provide a guidance model config when using autoguidance."
            guid_model: Stage2ModelProtocol = instantiate_from_config(guid_model_config).to(device)
            guid_model.eval()  # important!
            guid_fwd = guid_model.forward
            model_kwargs['additional_model_forward'] = guid_fwd
            model_fwd = model.forward_with_autoguidance
        else:
            model_fwd = model.forward_with_cfg
    else:
        model_kwargs = dict(y=y)
        model_fwd = model.forward
    # Sample images:
    start_time = time()
    samples:torch.Tensor = sample_fn(z, model_fwd, **model_kwargs)[-1]
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    # samples = vae.decode(samples / 0.18215).sample
    samples = rae.decode(samples).clamp_(0.0, 1.0)
    print(f"Sampling took {time() - start_time:.2f} seconds.")

    # Save and display images:
    image_size = samples.shape[-1]
    print(f"Saving {samples.shape[0]} samples of size {image_size}x{image_size}.")
    class_dir = os.path.join(
        args.save_dir,
        f"class_{args.class_id:03d}_i{image_size}"
    )
    os.makedirs(class_dir, exist_ok=True)
    for idx, sample in enumerate(samples):
        output_path = os.path.join(class_dir, f"{idx:05d}.png")
        save_image(sample, output_path, normalize=True, value_range=(0, 1))
    print(f"Saved {len(samples)} sample(s) to {class_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the config file.")
    # if len(sys.argv) < 2:
    #     print("Usage: program.py <mode> [options]")
    #     sys.exit(1)

    # mode = sys.argv[1]

    # assert mode[:2] != "--", "Usage: program.py <mode> [options]"
    # assert mode in ["ODE", "SDE"], "Invalid mode. Please choose 'ODE' or 'SDE'"

    # parser.add_argument("--model", type=str, default="DDTXL")
    # parser.add_argument("--guid-model", type=str, default=None)
    # parser.add_argument("--guid-model-ckpt", type=str, default=None)
    # parser.add_argument("--image-size", type=int,
    #                     choices=[256, 512], default=256)
    # parser.add_argument("--num-classes", type=int, default=1000)
    # parser.add_argument("--cfg-scale", type=float, default=4.0)
    # parser.add_argument("--cfg-t-min", type=float, default=0.0)
    # parser.add_argument("--cfg-t-max", type=float, default=1.0)
    # parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="samples",
                        help="Directory to save the sampled images.")
    parser.add_argument("--class_id", type=int, required=True,
                        help="Class id to condition the sampler on.")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of samples to generate for the provided class id.")
    # parser.add_argument("--ckpt", type=str, default=None,
    #                     help="Optional path to a SiT checkpoint (default: auto-download a pre-trained SiT-XL/2 model).")

    args = parser.parse_known_args()[0]
    main(args)
