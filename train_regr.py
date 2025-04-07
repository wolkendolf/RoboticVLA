"""
This training script can be run both on a single GPU in debug mode,
and also in a larger training run with distributed data parallel (DDP).
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
from torch.utils.data import Dataset, DataLoader, TensorDataset

from model_regr import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Default config values designed to train a GPT model on 8-dimensional vector sequences
# I/O
out_dir = "out"
eval_interval = 100
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = False
init_from = "scratch"

# Wandb logging
wandb_log = True
wandb_project = "robot-trajectory"
wandb_run_name = "nanoGPT-regression"

# Data
data_dir = "/data/kazachkovda/trajectories/"  # path to folder with raw trajectories
gradient_accumulation_steps = 5 * 8
batch_size = 12
block_size = 64

# Model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.1
bias = False
vector_dim = 8

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
backend = "nccl"

# System
device = "cuda:1"
# dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
dtype = "float16"
compile = True
# -----------------------------------------------------------------------------

config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("configurator.py").read())
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

# Various inits, derived attributes, I/O setup
ddp = int(os.environ.get("RANK", -1)) != -1  # Is this a DDP run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # This process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # Each process gets a different seed
    # World_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # If not DDP, we are running on a single GPU, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

# We no longer calculate tokens_per_iter since we're not working with tokens
vectors_per_iter = (
    gradient_accumulation_steps * ddp_world_size * batch_size * block_size
)
print(f"vectors per iteration will be: {vectors_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # Allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # Allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # For later use in torch.autocast
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)


# Custom Dataset class for loading trajectories
class TrajectoryDataset(Dataset):
    def __init__(self, data_file, block_size):
        # Load the preprocessed data (assumed to be a NumPy array or PyTorch tensor)
        self.data = np.load(
            data_file, allow_pickle=True
        )  # Shape: (num_steps, vector_dim)
        self.block_size = block_size
        self.vector_dim = self.data.shape[-1]

    def __len__(self):
        # Number of possible sequences
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # Get a sequence of length block_size and its target (next sequence)
        x = self.data[idx : idx + self.block_size]  # (block_size, vector_dim)
        y = self.data[idx + 1 : idx + 1 + self.block_size]  # (block_size, vector_dim)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(
            y, dtype=torch.float32
        )


# Data loader setup
train_dataset = TrajectoryDataset(os.path.join(data_dir, "train.npy"), block_size)
val_dataset = TrajectoryDataset(os.path.join(data_dir, "val.npy"), block_size)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
)

# Create iterators for the data loaders
train_iter = iter(train_loader)
val_iter = iter(val_loader)


def get_batch(split):
    # Fetch a batch from the appropriate data loader
    data_loader = train_loader if split == "train" else val_loader
    data_iter = train_iter if split == "train" else val_iter
    try:
        x, y = next(data_iter)
    except StopIteration:
        # Reset the iterator if we reach the end
        data_iter = iter(data_loader)
        if split == "train":
            globals()["train_iter"] = data_iter
        else:
            globals()["val_iter"] = data_iter
        x, y = next(data_iter)
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    return x, y


# Init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e3

# Model init
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    dropout=dropout,
    vector_dim=vector_dim,
)  # Start with model_args from command line
if init_from == "scratch":
    print("Initializing a new model from scratch")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == "resume":
    print(f"Resuming training from {out_dir}")
    # Resume training from a checkpoint
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    # Force these config attributes to be equal otherwise we can't resume training
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vector_dim"]:
        model_args[k] = checkpoint_model_args[k]
    # Create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    # Fix the keys of the state dictionary
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]
else:
    raise ValueError("init_from must be 'scratch' or 'resume' for regression tasks")

# Crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args["block_size"] = (
        block_size  # So that the checkpoint will have the right value
    )
model.to(device)

# Initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

# Optimizer
optimizer = model.configure_optimizers(
    weight_decay, learning_rate, (beta1, beta2), device_type
)
if init_from == "resume":
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None  # Free up memory

# Compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # Requires PyTorch 2.0

# Wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


# Helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        data_loader = train_loader if split == "train" else val_loader
        data_iter = iter(data_loader)
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            try:
                X, Y = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                X, Y = next(data_iter)
            X, Y = X.to(device), Y.to(device)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# Learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) Linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) If it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) In between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# Logging
if wandb_log and master_process:
    import wandb

    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

try:
    # Training loop
    X, Y = get_batch("train")  # Fetch the very first batch
    t0 = time.time()
    local_iter_num = 0  # Number of iterations in the lifetime of this process
    raw_model = model.module if ddp else model  # Unwrap DDP container if needed
    running_mfu = -1.0
    while True:

        # Determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0 and master_process:
            losses = estimate_loss()
            print(
                f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
            if wandb_log:
                wandb.log(
                    {
                        "iter": iter_num,
                        "train/loss": losses["train"],
                        "val/loss": losses["val"],
                        "lr": lr,
                        "mfu": running_mfu * 100,  # Convert to percentage
                    }
                )
            if losses["val"] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses["val"]
                if iter_num > 0:
                    checkpoint = {
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_args": model_args,
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        "config": config,
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
        if iter_num == 0 and eval_only:
            break

        # Forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                model.require_backward_grad_sync = (
                    micro_step == gradient_accumulation_steps - 1
                )
            with ctx:
                _, loss = model(X, Y)
                loss = (
                    loss / gradient_accumulation_steps
                )  # Scale the loss to account for gradient accumulation
            # Immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch("train")
            # Backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        # Clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # Step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # Flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # Timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0 and master_process:
            lossf = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5:  # Let the training loop settle a bit
                mfu = raw_model.estimate_mfu(
                    batch_size * gradient_accumulation_steps, dt
                )
                running_mfu = (
                    mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                )
            print(
                f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%"
            )
        iter_num += 1
        local_iter_num += 1

        # Termination conditions
        if iter_num > max_iters:
            break
    if ddp:
        destroy_process_group()

except KeyboardInterrupt:
    if ddp:
        destroy_process_group()
    print("Обучение остановлено пользователем.")
    wandb.run.finish()  # Альтернативный метод завершения
