import os
import cv2 
import time
import random 
import datetime
import argparse
import numpy as np
from tqdm import tqdm
from piq import ssim,psnr
from itertools import cycle

import torch
import torch.nn as nn
from torch.utils import data
import torch.distributed as dist 
from torch.utils.data.distributed import DistributedSampler 
from torch.nn.parallel import DistributedDataParallel as DDP 
from torch.utils.tensorboard import SummaryWriter
import torch.amp as amp  # Update GradScaler import

from utils import dict2string,mkdir,get_lr,torch2cvimg,second2hours
from loaders import docres_loader
from models import restormer_arch

# Set PyTorch memory allocation configuration
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32,expandable_segments:True'

def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) 
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) 
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
    #torch.use_deterministic_algorithms(True)
# seed_torch()


def getBasecoord(h,w):
    base_coord0 = np.tile(np.arange(h).reshape(h,1),(1,w)).astype(np.float32)
    base_coord1 = np.tile(np.arange(w).reshape(1,w),(h,1)).astype(np.float32)
    base_coord = np.concatenate((np.expand_dims(base_coord1,-1),np.expand_dims(base_coord0,-1)),-1)
    return base_coord

def train(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create log directory
    mkdir(args.logdir)
    mkdir(os.path.join(args.logdir, args.experiment_name))
    log_file_path = os.path.join(args.logdir, args.experiment_name, 'log.txt')
    log_file = open(log_file_path, 'a')
    log_file.write('\n---------------  ' + args.experiment_name + '  ---------------\n')
    log_file.close()

    # Setup tensorboard
    if args.tboard:
        writer = SummaryWriter(os.path.join(args.logdir, args.experiment_name, 'runs'))

    # Setup Dataset
    dataset_setting = {
        'task': 'appearance',
        'ratio': 1,
        'im_path': 'data/train/appearance/',
        'json_paths': ['data/train_appearance.json']
    }

    # Create datasets and dataloaders
    train_dataset = docres_loader.DocResTrainDataset(dataset=dataset_setting, img_size=args.im_size)
    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # Setup Model with reduced size
    model = restormer_arch.Restormer(
        inp_channels=6,
        out_channels=3,
        dim=24,  # Further reduced from 32
        num_blocks=[1,1,2,2],  # Further reduced from [1,2,2,3]
        num_refinement_blocks=2,  # Further reduced from 3
        heads=[1,2,2,4],  # Reduced head sizes
        ffn_expansion_factor=1.5,  # Further reduced from 2.0
        bias=False,
        LayerNorm_type='WithBias',
        dual_pixel_task=True
    )
    model = model.to(device)

    # Enable gradient checkpointing
    model.train()
    if hasattr(model, 'encoder_level1'):
        model.encoder_level1.gradient_checkpointing = True
    if hasattr(model, 'encoder_level2'):
        model.encoder_level2.gradient_checkpointing = True
    if hasattr(model, 'encoder_level3'):
        model.encoder_level3.gradient_checkpointing = True
    if hasattr(model, 'latent'):
        model.latent.gradient_checkpointing = True

    # Enable memory efficient attention
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    
    # Empty CUDA cache
    torch.cuda.empty_cache()

    # Optimizer with gradient clipping
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.l_rate, weight_decay=5e-4)
    grad_scaler = amp.GradScaler('cuda')

    # LR Scheduler
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.total_iter, eta_min=1e-6)

    # Load checkpoint if resuming
    iter_start = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        iter_start = checkpoint['iter']
        print(f"Loaded checkpoint '{args.resume}' (iter {iter_start})")

    # Training loop
    l1_loss = nn.L1Loss()
    train_iter = iter(train_loader)
    
    for iters in range(iter_start, args.total_iter):
        start_time = time.time()
        
        try:
            in_im, gt_im = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            in_im, gt_im = next(train_iter)
            
        in_im = in_im.float().to(device)
        gt_im = gt_im.float().to(device)

        # Forward pass with mixed precision
        with amp.autocast(device_type='cuda'):
            pred_im = model(in_im, 'appearance')
            loss = l1_loss(pred_im, gt_im)

        # Backward pass with gradient scaling
        optimizer.zero_grad()
        grad_scaler.scale(loss).backward()
        
        # Gradient clipping
        grad_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        grad_scaler.step(optimizer)
        grad_scaler.update()

        # Log progress
        if (iters + 1) % 10 == 0:
            duration = time.time() - start_time
            log_str = f'iter [{iters+1}/{args.total_iter}] -- loss: {loss.item():.4f} -- lr: {get_lr(optimizer):.6f} -- time remaining: {second2hours(duration*(args.total_iter-iters))}'
            print(log_str)
            
            if args.tboard:
                writer.add_scalar('Train Loss/Iterations', loss.item(), iters)
                if (iters + 1) % 50 == 0:  # Save images every 50 iterations
                    writer.add_images('Input', in_im[:, :3], iters)
                    writer.add_images('Ground Truth', gt_im, iters)
                    writer.add_images('Prediction', pred_im, iters)
            
            with open(log_file_path, 'a') as f:
                f.write(log_str + '\n')

        # Save checkpoint
        if (iters + 1) % 5000 == 0:
            state = {
                'iter': iters + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
            }
            torch.save(state, os.path.join(args.logdir, args.experiment_name, f"checkpoint_{iters+1}.pkl"))

        sched.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training settings')
    parser.add_argument('--experiment_name', type=str, default='appearance_enhancement')
    parser.add_argument('--logdir', type=str, default='logs')
    parser.add_argument('--im_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=2)  # Further reduced from 4
    parser.add_argument('--l_rate', type=float, default=2e-4)
    parser.add_argument('--total_iter', type=int, default=300000)
    parser.add_argument('--tboard', type=bool, default=True)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--profile_memory', action='store_true', help='Enable CUDA memory profiling')
    
    args = parser.parse_args()
    seed_torch()
    train(args)