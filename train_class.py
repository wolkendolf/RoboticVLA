"""
Training script for classification on 8-dimensional vectors with 256 bins, using Distributed Data Parallel (DDP).
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from model_class import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Default config values
out_dir = 'out'
eval_interval = 100
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = False
init_from = 'scratch'
# Wandb logging
wandb_log = True
wandb_project = 'robot-trajectory-classification'
wandb_run_name = 'nanoGPT-classification'
# Data
dataset = "/data/kazachkovda/trajectories/"
gradient_accumulation_steps = 6
batch_size = 12
block_size = 64
# Model
n_layer = 12  # Уменьшен для экономии памяти
n_head = 12   # Уменьшен для экономии памяти
n_embd = 768  # Уменьшен для экономии памяти
dropout = 0.1
bias = False
vector_dim = 8
num_bins = 1024
# AdamW optimizer
learning_rate = 6e-4
max_iters = 10000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
# Learning rate decay settings
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5
# DDP settings
backend = 'nccl'
# System
device = 'cuda:1'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False  # Отключено для экономии памяти
# -----------------------------------------------------------------------------

config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

# DDP setup
ddp = int(os.environ.get('RANK', -1)) != -1  # Проверяем, запущено ли обучение с DDP
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # Только процесс с рангом 0 будет логировать и сохранять чекпоинты
    seed_offset = ddp_rank
    # Корректируем gradient_accumulation_steps для DDP
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # Если DDP не используется, работаем на одной GPU или CPU
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    ddp_rank = 0
    ddp_local_rank = 0

vectors_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
if master_process:
    print(f"vectors per iteration will be: {vectors_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Custom Dataset class with discretization
class TrajectoryDataset(Dataset):
    def __init__(self, data_file, block_size, num_bins=256, min_val=-1, max_val=1):
        self.data = np.load(data_file)  # Shape: (num_steps, vector_dim)
        self.block_size = block_size
        self.vector_dim = self.data.shape[-1]
        self.num_bins = num_bins
        self.min_val = min_val
        self.max_val = max_val

        # Дискретизация данных
        self.data = self.discretize(self.data)

    def discretize(self, vectors):
        # vectors: (..., vector_dim), значения в [min_val, max_val]
        # Нормализуем в [0, 1]
        normalized = (vectors - self.min_val) / (self.max_val - self.min_val)  # [0, 1]
        # Преобразуем в индексы бинов [0, num_bins-1]
        bin_indices = np.floor(normalized * self.num_bins).astype(np.int64)
        bin_indices = np.clip(bin_indices, 0, self.num_bins - 1)
        return bin_indices

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]  # (block_size, vector_dim)
        y = self.data[idx + 1:idx + 1 + self.block_size]  # (block_size, vector_dim)
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# Data loader setup with DistributedSampler for DDP
data_dir = os.path.join('data', dataset)
train_dataset = TrajectoryDataset(os.path.join(data_dir, 'train.npy'), block_size, num_bins=num_bins)
val_dataset = TrajectoryDataset(os.path.join(data_dir, 'val.npy'), block_size, num_bins=num_bins)

# Используем DistributedSampler для распределения данных между GPU
train_sampler = DistributedSampler(train_dataset, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True)
val_sampler = DistributedSampler(val_dataset, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=False)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    sampler=train_sampler,
    num_workers=0,  # Уменьшено для стабильности
    pin_memory=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    sampler=val_sampler,
    num_workers=0,
    pin_memory=True
)

# Итераторы для загрузки данных
train_iter = iter(train_loader)
val_iter = iter(val_loader)

def get_batch(split):
    data_loader = train_loader if split == 'train' else val_loader
    data_iter = train_iter if split == 'train' else val_iter
    try:
        x, y = next(data_iter)
    except StopIteration:
        # При использовании DDP нужно обновить sampler на каждой эпохе
        if split == 'train':
            train_loader.sampler.set_epoch(ddp_rank)
        data_iter = iter(data_loader)
        if split == 'train':
            globals()['train_iter'] = data_iter
        else:
            globals()['val_iter'] = data_iter
        x, y = next(data_iter)
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    return x, y

# Init these up here, can override if init_from='resume'
iter_num = 0
best_val_loss = 1e9

# Model init
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    dropout=dropout,
    vector_dim=vector_dim,
    num_bins=num_bins
)
if init_from == 'scratch':
    if master_process:
        print("Initializing a new model from scratch")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    if master_process:
        print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vector_dim', 'num_bins']:
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
else:
    raise ValueError("init_from must be 'scratch' or 'resume'")

if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size

model.to(device)

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None

# Оборачиваем модель в DDP
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

if compile:
    if master_process:
        print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        data_loader = train_loader if split == 'train' else val_loader
        data_iter = iter(data_loader)
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            try:
                X, Y = next(data_iter)
            except StopIteration:
                if split == 'train':
                    train_loader.sampler.set_epoch(ddp_rank)
                data_iter = iter(data_loader)
                X, Y = next(data_iter)
            X, Y = X.to(device), Y.to(device)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        # Собираем потери со всех GPU
        if ddp:
            torch.distributed.all_reduce(losses, op=torch.distributed.ReduceOp.SUM)
            losses /= ddp_world_size
        out[split] = losses.mean().item()
    model.train()
    return out

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0

while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100,
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # В DDP синхронизация градиентов происходит только на последнем шаге накопления
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            _, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch('train')
        scaler.scale(loss).backward()
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()