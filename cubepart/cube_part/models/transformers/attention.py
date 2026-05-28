import math
from functools import partial

import torch
import torch.nn as nn

from .norm import LayerNorm, RMSNorm


def init_linear(module, embed_dim: int):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=math.sqrt(1.0 / embed_dim))
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)


def init_tfixup(module: nn.Module, num_layers: int):
    """Special initialization from https://www.cs.toronto.edu/~mvolkovs/ICML2020_tfixup.pdf

    Args:
        module (nn.Module): decoder/encoder module
        num_layers (int): number of layers in the module
    """
    with torch.no_grad():
        for pn, p in module.named_parameters():
            if (
                pn.endswith("c_proj.weight")
                or pn.endswith("up_proj.weight")
                or pn.endswith("down_proj.weight")
            ):
                p *= (4 * num_layers) ** (-0.25)
            elif pn.endswith("c_v.weight"):
                p *= (4 * num_layers) ** (-0.25) * math.sqrt(2)


class MLP(nn.Module):
    def __init__(self, embed_dim, hidden_dim, bias=True, approximate="none"):
        super().__init__()
        self.up_proj = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, embed_dim, bias=bias)
        self.act_fn = nn.GELU(approximate=approximate)

        init_linear(self.up_proj, embed_dim)
        init_linear(self.down_proj, hidden_dim)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.up_proj(x)))


class SelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        # key, query, value projections for all heads, but in a batch
        self.c_qk = nn.Linear(embed_dim, 2 * embed_dim, bias=bias)
        self.c_v = nn.Linear(embed_dim, embed_dim, bias=bias)
        # output projection
        self.c_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # regularization
        self.dropout = dropout

        head_dim = embed_dim // num_heads
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

        self.apply(partial(init_linear, embed_dim=embed_dim))

    def forward(self, x, attn_mask=None, is_causal: bool = False):
        # batch size, sequence length, embedding dimensionality (n_embd)
        b, l, d = x.shape

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k = self.c_qk(x).chunk(2, dim=-1)
        v = self.c_v(x)

        q = q.view(b, l, self.num_heads, -1).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(b, l, self.num_heads, -1).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(b, l, self.num_heads, -1).transpose(1, 2)  # (B, nh, T, hs)

        q = self.q_norm(q)
        k = self.k_norm(k)

        is_causal = is_causal and attn_mask is None
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # efficient attention using Flash Attention CUDA kernels
        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        y = (
            y.transpose(1, 2).contiguous().view(b, l, d)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y


class CrossAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        q_dim=None,
        kv_dim=None,
        dropout: float = 0.0,
        bias: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0

        q_dim = q_dim or embed_dim
        kv_dim = kv_dim or embed_dim

        # key, query, value projections for all heads, but in a batch
        self.c_q = nn.Linear(q_dim, embed_dim, bias=bias)
        self.c_k = nn.Linear(kv_dim, embed_dim, bias=bias)
        self.c_v = nn.Linear(kv_dim, embed_dim, bias=bias)
        # output projection
        self.c_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.num_heads = num_heads
        # regularization
        self.dropout = dropout

        self.apply(partial(init_linear, embed_dim=embed_dim))

    def forward(self, x, c, attn_mask=None, is_causal: bool = False):
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k = self.c_q(x), self.c_k(c)
        v = self.c_v(c)

        # batch size, sequence length, embedding dimensionality (n_embd)
        b, l, d = q.shape
        s = k.shape[1]

        q = q.view(b, l, self.num_heads, -1).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(b, s, self.num_heads, -1).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(b, s, self.num_heads, -1).transpose(1, 2)  # (B, nh, T, hs)

        # self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # efficient attention using Flash Attention CUDA kernels
        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=(attn_mask is not None) and is_causal,
        )

        y = (
            y.transpose(1, 2).contiguous().view(b, l, d)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y


class EncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.ln_1 = LayerNorm(embed_dim, elementwise_affine=False, eps=eps)
        self.attn = SelfAttention(
            embed_dim, num_heads, dropout=dropout, bias=bias, eps=eps
        )
        self.ln_2 = LayerNorm(embed_dim, elementwise_affine=False, eps=eps)
        self.mlp = MLP(embed_dim=embed_dim, hidden_dim=embed_dim * 4, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, is_causal: bool = False):
        x = x + self.dropout(
            self.attn(self.ln_1(x), attn_mask=attn_mask, is_causal=is_causal)
        )
        x = x + self.dropout(self.mlp(self.ln_2(x)))
        return x


class EncoderCrossAttentionLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        q_dim=None,
        kv_dim=None,
        dropout: float = 0.0,
        bias: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        q_dim = q_dim or embed_dim
        kv_dim = kv_dim or embed_dim

        self.attn = CrossAttention(
            embed_dim,
            num_heads,
            q_dim=q_dim,
            kv_dim=kv_dim,
            dropout=dropout,
            bias=bias,
            eps=eps,
        )

        self.ln_1 = LayerNorm(q_dim, elementwise_affine=False, eps=eps)
        self.ln_2 = LayerNorm(kv_dim, elementwise_affine=False, eps=eps)

        self.ln_f = LayerNorm(embed_dim, elementwise_affine=False, eps=eps)
        self.mlp = MLP(embed_dim=embed_dim, hidden_dim=embed_dim * 4, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, c, attn_mask=None, is_causal: bool = False):
        x = x + self.dropout(
            self.attn(
                self.ln_1(x), self.ln_2(c), attn_mask=attn_mask, is_causal=is_causal
            )
        )
        x = x + self.dropout(self.mlp(self.ln_f(x)))
        return x
