"""
Based on Official Implementation of [NeurIPS 2024] Expanding Sparse Tuning for Low Memory Usage 
https://github.com/ssfgunner/SNELL
"""
import argparse
import os 
import sys
import random
import torch
import numpy as np

from pathlib import Path
from timm.loss import LabelSmoothingCrossEntropy
from timm.scheduler import create_scheduler
from timm.utils import accuracy
from utils.dataset import build_dataset

import model as models
from timm.models import load_checkpoint
import time

from utils.gps import gps_masking_params, gps_masking_params_from_npz


def seed_everything(seed):
    """
    Fixing all seeds for reproducibillity
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--model_name', default=None, type=str)
    parser.add_argument('--pretrained', default=True, type=bool)
    parser.add_argument('--resume', default=None, type=str, help='path to the pre-trained model')
    parser.add_argument('--val-interval', default=10, type=int)
    parser.add_argument('--exp_name', default='', type=str)
    parser.add_argument('--test', action='store_true', help='using test-split or validation split')
    
    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    
    # AutoFormer config
    parser.add_argument('--input-size', type=int, default=224)
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--lr-power', type=float, default=1.0,
                        help='power of the polynomial lr scheduler')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    
    # Augmentation parameters
    parser.add_argument('--no-aug', action='store_true')
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic (default: bicubic))')
    
    # Dataset parameters
    parser.add_argument('--data-path', default='./data/imagenet', type=str,
                        help='dataset path')
    parser.add_argument('--output_dir', default='./output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--data-set', default='IMNET', type=str, help='Image Net dataset path')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU')
    parser.add_argument('--no-pin-mem', action='store_true', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    
    # GPS Parameters
    parser.add_argument('--pre-idx', default=None, type=str,
                        help='Path of predetermined indices of max gradient values per output neuron')
    parser.add_argument('--topk', default=1, type=int, help='Selecting Top-k parameters')
    parser.add_argument('--low-rank-dim', default=1, type=int, help='The rank of LoRA')
    parser.add_argument('--no_drop_out', action='store_true')
    parser.add_argument('--no_drop_path', action='store_true')
    
    return parser


def main(args):
    
    seed_everything(args.seed)
    
    # 모든 출력을 log 파일에 기록
    log_file = open(f'{args.output_dir}/log_everything.log', 'w')
    sys.stdout = log_file
    
    print(args)
    
    device = torch.device(args.device)

    train_dataset, args.n_classes = build_dataset(is_train=True, args=args)
    eval_dataset, _ = build_dataset(is_train=False, args=args)
    
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    eval_sampler = torch.utils.data.SequentialSampler(eval_dataset)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=train_sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=int(2*args.batch_size),
        sampler=eval_sampler, 
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    
    print(f'{args.data_set} dataset, train: {len(train_dataset)}, test: {len(eval_dataset)}')
    
    model = models.__dict__[args.model_name](
        pretrained=args.pretrained,
        img_size=args.input_size,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        low_rank_dim=args.low_rank_dim,
        num_classes=args.n_classes,
        no_drop_out=args.no_drop_out,
        no_drop_path=args.no_drop_path,
        topk=args.topk)
    
    if args.resume is not None:
        load_checkpoint(model, args.resume)
        print(f'load from {args.resume}')
    
    model.to(device)
    
    if args.pre_idx is not None:
        gps_masking_params_from_npz(model, args)
    else:
        grad_dataset, args.n_classes = build_dataset(is_train=True, args=args, multiview=True)
        grad_loader = torch.utils.data.DataLoader(
            grad_dataset, sampler=train_sampler,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
        gps_masking_params(model, grad_loader, args)
    
    if args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(
        model.parameters(),  
        lr=args.lr,          
        weight_decay=args.weight_decay
    )
    
    args.warmup_steps=[0.0002]
    args.lr_noise_std=1.0
    args.lr_k_decay=1.0
    args.lr_noise_pct=0.67
    args.cycle_decay=0.5
    
    lr_scheduler, _ = create_scheduler(args, optimizer)
    
    output_dir = Path(args.output_dir)
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    
    print('Start training')
    
    log_freq = 5
    max_accuracy = 0.0
    
    for epoch in range(args.start_epoch, args.epochs):
        torch.cuda.reset_peak_memory_stats()
        
        train_engine(model, criterion, train_loader, optimizer, device, epoch)
        lr_scheduler.step(epoch+1)
        
        if (epoch+1) % log_freq == 0:
           max_accuracy = eval_engine(model, eval_loader, device, epoch, max_accuracy)

    print(f'Training is done. Max Accuracy {max_accuracy}')
    log_file.close()
    
    
def train_engine(model, criterion, loader, optimizer, device, epoch):
    
    model.train()
    criterion.train()
    if epoch == 0:
        print(f'Model init GPU mem: {torch.cuda.memory_allocated()/1024**2.:2f}')
    
    total_loss = 0
    total_images = 0
    total_time = 0
    
    for idx, (samples, targets) in enumerate(loader):
        torch.cuda.synchronize()
        start_time = time.time()
        
        for p in model.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
                    
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        if epoch ==  0 and idx == 0:
            print(f'Data input GPU mem: {torch.cuda.memory_allocated()/1024**2.:2f}')
        
        outputs = model(samples)
        loss = criterion(outputs, targets)
        
        if epoch ==  0 and idx == 0:
            print(f'Feed Forward GPU mem: {torch.cuda.max_memory_allocated()/1024**2.:2f}')

        total_loss += loss.item()
        
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
    
        if epoch == 0 and idx == 0:
            print(f'Backward Propagation GPU mem: {torch.cuda.memory_allocated()/1024**2}')
        
        torch.cuda.synchronize()
        batch_time = time.time() - start_time
        batch_size = samples.shape[0]
        total_images += batch_size
        total_time += batch_time
    
    print(f'Epoch[{epoch+1}] Train loss {total_loss / total_images:.4f} | Average processing time per image: {total_time / total_images:.4f}(s / it)')
    
    
@torch.no_grad()
def eval_engine(model, loader, device, epoch, max_accuracy):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    
    total_acc1 = 0
    total_acc5 = 0
    total_images = 0
    total_loss = 0
    
    for images, targets in loader:
        
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast():
            output = model(images)            
            loss = criterion(output, targets)
        
        total_loss += loss.item()
        
        batch_size = images.shape[0]
        
        try:
            acc1, acc5 = accuracy(output, targets, topk=(1, 5))
            total_acc1 += acc1 * batch_size
            total_acc5 += acc5 * batch_size
            total_images += batch_size
        
        except RuntimeError:
            # class num <= 5
            acc1 = accuracy(output, targets, topk=(1,))
            total_acc1 += acc1 * batch_size
            total_acc5 += acc5 * batch_size
            total_images = batch_size
    
    avg_acc1 = total_acc1 / total_images
    avg_acc5 = total_acc5 / total_images
    
    avg_loss = total_loss / total_images
    
    max_accuracy = max(max_accuracy, avg_acc1)
            
    print(f'Epoch[{epoch+1}] Test loss {avg_loss:.4f} Acc@1 {avg_acc1} Acc@5 {avg_acc5} Max Acc {max_accuracy}')
    
    return max_accuracy

    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'GPS + LoRA training and evaluation scripts', parents=[get_args_parse()], add_help=False)
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)