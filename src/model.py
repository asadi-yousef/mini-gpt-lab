import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


class AttentionHead(nn.Module):
    def __init__(self, d_model , head_size , block_size):
        super().__init__()
        self.query = nn.Linear(d_model,head_size,bias=False)
        self.key = nn.Linear(d_model,head_size,bias=False)
        self.value = nn.Linear(d_model,head_size,bias=False)
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
            )
    def forward(self, x):
        B,T,C = x.shape

        q = self.query(x)  # (B, T, head_size)
        k = self.key(x)    # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        # compute QK^t
        scores = q @ k.transpose(-2,-1)  # (B, T, T)

        # scale
        scores = scores / (k.shape[-1] ** 0.5)

        # masking
        scores = scores.masked_fill(
            self.tril[:T,:T] == 0 ,
            float("-inf")
        )

        # attention probabilities
        weights = F.softmax(scores,dim=-1)  # (B, T, T)

        # weighted sum of values
        out = weights @ v           # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, block_size):
        super().__init__()

        assert d_model % n_heads == 0

        self.head_size = d_model // n_heads
        self.attn = nn.ModuleList()

        self.o = nn.Linear(d_model, d_model, bias=False)

        # gathering all the heads into a list
        for i in range(n_heads):
            head = AttentionHead(
                d_model,
                self.head_size,
                block_size
            )
            self.attn.append(head)

    def forward(self, x):
        outputs = []
        # computing outputs for each head
        for head in self.attn:  
            outputs.append(head(x))

        # concatenate all outputs
        out = torch.cat(outputs,dim=-1)

        out = self.o(out)
        return out

class MLP(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.c_fc    = nn.Linear(d_model, 4 * d_model, bias=True)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * d_model, d_model, bias=True)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, d_model, n_heads, block_size):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            block_size=block_size
        )

        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model)

    def forward(self, x):
        # Self-attention + residual connection
        x = x + self.attn(self.ln1(x))

        # Feed-forward network + residual connection
        x = x + self.mlp(self.ln2(x))

        return x

class GPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        n_heads,
        n_layers,
        block_size
    ):
        super().__init__()

        self.block_size = block_size

        # token identity
        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model
        )

        # token position
        self.position_embedding = nn.Embedding(
            block_size,
            d_model
        )

        # stack of transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                d_model=d_model,
                n_heads=n_heads,
                block_size=block_size
            )
            for _ in range(n_layers)
        ])

        # final normalization
        self.ln_f = nn.LayerNorm(d_model)

        # project each contextualized token to vocabulary logits
        self.lm_head = nn.Linear(
            d_model,
            vocab_size,
            bias=False
        )

    def forward(self, idx, targets=None):
        B, T = idx.shape

        assert T <= self.block_size

        # (B, T) -> (B, T, d_model)
        token_emb = self.token_embedding(idx)

        # positions: [0, 1, ..., T-1]
        positions = torch.arange(
            T,
            device=idx.device
        )

        # (T,) -> (T, d_model)
        pos_emb = self.position_embedding(positions)

        # broadcasting:
        # (B,T,d_model) + (T,d_model)
        x = token_emb + pos_emb

        # transformer stack
        for block in self.blocks:
            x = block(x)

        # final layer norm
        x = self.ln_f(x)

        # (B,T,d_model) -> (B,T,vocab_size)
        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            B, T, C = logits.shape

            logits_flat = logits.reshape(B * T, C)
            targets_flat = targets.reshape(B * T)

            loss = F.cross_entropy(
                logits_flat,
                targets_flat
            )

        return logits, loss
    