"""
Full definition of a GPT Model adapted for classification on 8-dimensional vectors with 256 bins.
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    vector_dim: int = 8  # Размерность вектора (8 в данном случае)
    num_bins: int = 256  # Количество бинов для дискретизации

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            # Вместо input_projection используем nn.Embedding для индексов бинов
            wte = nn.Embedding(config.num_bins, config.n_embd),  # Эмбеддинги для бинов
            wpe = nn.Embedding(config.block_size, config.n_embd),  # Позиционные эмбеддинги
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        # Выходной слой: предсказываем распределение по num_bins для каждой компоненты
        self.head = nn.Linear(config.n_embd, config.vector_dim * config.num_bins, bias=config.bias)

        # Инициализация весов
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        device = x.device
        b, t, d = x.size()  # x: (batch_size, sequence_length, vector_dim)
        assert d == self.config.vector_dim, f"Input vector dimension must be {self.config.vector_dim}, got {d}"
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        assert x.dtype == torch.long, "Input must be indices (torch.long), not continuous values"

        pos = torch.arange(0, t, dtype=torch.long, device=device)  # (t)

        # Преобразуем индексы бинов в эмбеддинги
        # x: (b, t, d) -> (b, t, d, n_embd) после эмбеддинга
        x = self.transformer.wte(x)  # (b, t, d, n_embd)
        # Суммируем эмбеддинги по векторным компонентам (d), чтобы получить (b, t, n_embd)
        x = x.sum(dim=2)  # (b, t, n_embd)

        # Добавляем позиционные эмбеддинги
        pos_emb = self.transformer.wpe(pos)  # (t, n_embd)
        x = self.transformer.drop(x + pos_emb)

        # Пропускаем через трансформер
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        # Предсказываем распределение по бинам для каждой компоненты
        logits = self.head(x)  # (b, t, vector_dim * num_bins)
        logits = logits.view(b, t, self.config.vector_dim, self.config.num_bins)  # (b, t, vector_dim, num_bins)

        if targets is not None:
            # targets: (b, t, vector_dim), значения — индексы бинов (0-255)
            # Кросс-энтропийная потеря ожидает (b, num_classes, ...) и (b, ...)
            # Поэтому преобразуем logits и targets
            loss = F.cross_entropy(logits.view(-1, self.config.num_bins), targets.view(-1))
        else:
            # Во время инференса возвращаем предсказания для последнего шага
            logits = logits[:, -1, :, :]  # (b, vector_dim, num_bins)
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        raise NotImplementedError("from_pretrained is not supported for this task")

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt)
        flops_promised = 312e12
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, x, max_new_vectors, temperature=1.0):
        """
        x: (b, t, vector_dim) — индексы бинов (torch.long)
        Returns: (b, t + max_new_vectors, vector_dim) — индексы бинов
        """
        for _ in range(max_new_vectors):
            x_cond = x if x.size(1) <= self.config.block_size else x[:, -self.config.block_size:, :]
            logits, _ = self(x_cond)  # logits: (b, vector_dim, num_bins)
            # Применяем температуру и выбираем наиболее вероятный бин
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)  # (b, vector_dim, num_bins)
            next_bins = torch.argmax(probs, dim=-1)  # (b, vector_dim)
            x = torch.cat((x, next_bins.unsqueeze(1)), dim=1)  # (b, t+1, vector_dim)
        return x

    @staticmethod
    def decode_bins(bin_indices, num_bins=256, min_val=-1, max_val=1):
        """
        Декодирование индексов бинов обратно в непрерывные значения.
        bin_indices: (...,) — индексы бинов (0-255)
        Returns: (...,) — непрерывные значения в диапазоне [min_val, max_val]
        """
        bin_width = (max_val - min_val) / num_bins
        bin_centers = min_val + (bin_indices + 0.5) * bin_width
        return bin_centers