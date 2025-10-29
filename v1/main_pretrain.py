
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import wandb
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm
from copy import deepcopy

# assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
import kolmogorov_tokenizers
from engines.latent_distillation_pretrain import train_one_epoch
torch.autograd.set_detect_anomaly(True)

def get_args_parser():
    parser = argparse.ArgumentParser('ALIT Training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N', help='epochs to warmup LR')

    parser.add_argument('--model', default='alit_small', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--input_size', default=256, type=int, help='images input size')
    parser.add_argument('--grad_clip', type=float, default=3.0, help='Gradient clip')

    # ALIT arguments
    parser.add_argument('--base_tokenizer', default="vqgan", type=str, help='Base 2D Tokenizer. Current options: VQGAN | VAE')
    parser.add_argument('--quantize_latent', action='store_true', help='Quantization of 1D latent tokens (before passing to decoder)')
    parser.add_argument('--factorize_latent', action='store_true', help='Factorization of 1D latent tokens (before passing to decoder)')
    parser.set_defaults(pin_mem=False)

    # Dataset parameters
    parser.add_argument('--data_path', default=None, type=str, help='dataset path')
    parser.add_argument('--output_dir', default='./output_dir', help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir', help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser
    

def main(args):
    
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # simple augmentation
    transform_train = transforms.Compose([
            # transforms.CenterCrop(args.input_size),
            # transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0)),
            transforms.RandomResizedCrop(args.input_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True, # persistent_workers=True,
    )

    base_tokenizer_args = {
        "id": args.base_tokenizer,
        "is_requires_grad": False
    }
    model = kolmogorov_tokenizers.__dict__[args.model](
        base_tokenizer_args=base_tokenizer_args, 
        quantize_latent=args.quantize_latent, factorize_latent=args.factorize_latent,
        train_stage="latent_distillation_pretrain"
    )
    model.to(device)
    print(sum(p.numel() for p in model.parameters()), "num params")

    model_without_ddp = model
    ema = deepcopy(model_without_ddp).to(device)

    '''
    ## Comment previous ema creation and uncomment this, if the above one fails. Latest version of transformers and peft doesn;t work with simple 
    def init_ema(model):
        """Create an EMA copy of the model with the same constructor args and weights."""
        ema = adaptive_tokenizers.__dict__[args.model](
            base_tokenizer_args=base_tokenizer_args, 
            quantize_latent=args.quantize_latent, 
            factorize_latent=args.factorize_latent,
            train_stage="latent_distillation_pretrain"
        )
        ema.load_state_dict(copy.deepcopy(model.state_dict()))
        for param in ema.parameters():
            param.requires_grad = False
        ema.to(device)
        return ema

    ema = init_ema(model_without_ddp)
    '''

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
        # args.lr = args.blr #* eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    # param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    # misc.load_model(args=args, model_without_ddp=model_without_ddp, ema=ema)
    misc.resume_model(args=args, model_without_ddp=model_without_ddp, ema=ema, optimizer=optimizer, loss_scaler=loss_scaler)

    if global_rank==0:
        wandb.init(project="tok-gen", group="imagenet100", name="no_quant_no_post_trunk_norm_with_dit_final_layer_learned_sigma_ip_sample_adaptive_test", config=args)


    ## debugging for nan checks...
    def scan_model(model):
        bad = []
        with torch.no_grad():
            for n,p in model.named_parameters():
                if p.numel() and not torch.isfinite(p).all():
                    bad.append(("param", n))
            for n,b in model.named_buffers():
                if b.numel() and not torch.isfinite(b).all():
                    bad.append(("buffer", n))
        return bad

    def scan_optimizer(opt):
        bad = []
        for i, st in opt.state.items():
            for k,v in st.items():
                if torch.is_tensor(v) and v.numel() and not torch.isfinite(v).all():
                    bad.append((i, k))
        return bad

    print("bad_model:", scan_model(model)[:10])
    print("bad_opt:", scan_optimizer(optimizer)[:10])  # if loading optimizer state


    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_one_epoch(
            model, ema, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model_without_ddp=model_without_ddp, ema=ema, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        misc.save_model_last(
            args=args, model_without_ddp=model_without_ddp, ema=ema, optimizer=optimizer,
            loss_scaler=loss_scaler, epoch=epoch)

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))



if __name__ == '__main__':
    torch.cuda.empty_cache()
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.log_dir = args.output_dir
    main(args)