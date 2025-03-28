from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cube3d.model.transformers.cache import Cache
from cube3d.model.transformers.norm import LayerNorm, RMSNorm
from cube3d.model.transformers.rope import scaled_dot_product_attention_with_rotary_emb


class SwiGLUMLP(nn.Module):
    def __init__(self, embed_dim, hidden_dim, bias=True, **kwargs):
        """
        A PyTorch implementation of the SwiGLU (Swish-Gated Linear Unit) MLP layer.
        This module consists of three linear projections: `gate_proj`, `up_proj`, and `down_proj`.
        It applies the SwiGLU activation function, which combines the Swish activation with a gating mechanism,
        followed by a projection back to the original embedding dimension.
        Args:
            embed_dim (int): The dimensionality of the input embeddings.
            hidden_dim (int): The dimensionality of the hidden layer.
            bias (bool, optional): Whether to include bias terms in the linear layers. Defaults to True.
            **kwargs: Additional keyword arguments (currently unused).
        """
        super().__init__()
        self.gate_proj = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, embed_dim, bias=bias)

    # Ignore copy
    def forward(self, x):
        """
        Applies a forward pass.
        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: The output tensor after applying the forward pass.
        """

        down_proj = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class SelfAttentionWithRotaryEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        eps: float = 1e-6,
    ):
        """
        A PyTorch module implementing self-attention with rotary embeddings.

        Args:
            embed_dim (int): The dimensionality of the input embeddings.
            num_heads (int): The number of attention heads.
            bias (bool, optional): Whether to include bias terms in the linear projections. Defaults to True.
            eps (float, optional): A small value added for numerical stability in normalization. Defaults to 1e-6.
        """
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        # key, query, value projections for all heads, but in a batch
        self.c_qk = nn.Linear(embed_dim, 2 * embed_dim, bias=False)
        self.c_v = nn.Linear(embed_dim, embed_dim, bias=bias)
        # output projection
        self.c_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        head_dim = embed_dim // num_heads
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

    def forward(
        self,
        x,
        freqs_cis: torch.Tensor,
        attn_mask=None,
        is_causal: bool = False,
        kv_cache: Optional[Cache] = None,
        curr_pos_id: Optional[torch.Tensor] = None,
        decode: bool = False,
    ):
        """
        Forward pass for the SelfAttentionWithRotaryEmbedding instance.
        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed rotary positional embeddings.
            attn_mask (Optional[torch.Tensor], optional): Attention mask to apply during self-attention. Defaults to None.
            is_causal (bool, optional): Whether to apply causal masking for autoregressive decoding. Defaults to False.
            kv_cache (Optional[Cache], optional): Cache object for storing key and value states for decoding. Defaults to None.
            curr_pos_id (Optional[torch.Tensor], optional): Current position indices for decoding. Required if `decode` is True. Defaults to None.
            decode (bool, optional): Whether the model is in decoding mode. Defaults to False.
        Returns:
            torch.Tensor: Output tensor after applying self-attention and projection.
        """
        # batch size, sequence length, embedding dim
        b, l, d = x.shape

        # compute q, k, v and then split per q, k, v
        q, k = self.c_qk(x).chunk(2, dim=-1)
        v = self.c_v(x)

        # split per head
        q = q.view(b, l, self.num_heads, -1).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(b, l, self.num_heads, -1).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(b, l, self.num_heads, -1).transpose(1, 2)  # (B, nh, T, hs)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if kv_cache is not None:
            if not decode:
                kv_cache.key_states[:, :, : k.shape[2], :].copy_(k)
                kv_cache.value_states[:, :, : k.shape[2], :].copy_(v)
            else:
                assert curr_pos_id is not None
                kv_cache.update(curr_pos_id, k, v)
            k = kv_cache.key_states
            v = kv_cache.value_states

        # self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # efficient attention using Flash Attention CUDA kernels
        y = scaled_dot_product_attention_with_rotary_emb(
            q,
            k,
            v,
            freqs_cis=freqs_cis,
            attn_mask=attn_mask,
            curr_pos_id=curr_pos_id if decode else None,
            is_causal=is_causal,
        )

        y = (
            y.transpose(1, 2).contiguous().view(b, l, d)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y


class DecoderLayerWithRotaryEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        eps: float = 1e-6,
    ) -> None:
        """
        Initializes the transformer model with rotary embeddings.
        Args:
            embed_dim (int): The dimensionality of the embedding space.
            num_heads (int): The number of attention heads.
            bias (bool, optional): Whether to include bias terms in the layers. Defaults to True.
            eps (float, optional): A small value added for numerical stability in layer normalization. Defaults to 1e-6.
        """
        super().__init__()

        self.ln_1 = LayerNorm(embed_dim, elementwise_affine=False, eps=eps)
        self.attn = SelfAttentionWithRotaryEmbedding(
            embed_dim, num_heads=num_heads, bias=bias, eps=eps
        )
        self.ln_2 = LayerNorm(embed_dim, elementwise_affine=False, eps=eps)
        self.mlp = SwiGLUMLP(embed_dim, embed_dim * 4, bias=bias)

    @classmethod
    def from_config(cls, cfg):
        """
        Create an instance of the class using the provided configuration.
        Args:
            cfg: A configuration object containing the following attributes:
                - n_embd (int): The size of the embedding dimension.
                - n_head (int): The number of attention heads.
                - bias (bool): Whether to include a bias term.
                - eps (float): A small value added for numerical stability.
        Returns:
            An instance of the class initialized with the specified configuration.
        """

        return cls(
            cfg.n_embd,
            num_heads=cfg.n_head,
            bias=cfg.bias,
            eps=cfg.eps,
        )

    def forward(
        self,
        x,
        freqs_cis: torch.Tensor,
        attn_mask=None,
        is_causal: bool = True,
        kv_cache: Optional[Cache] = None,
        curr_pos_id: Optional[torch.Tensor] = None,
        decode: bool = False,
    ):
        """
        Forward pass for the transformer model.
        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed sinusoidal positional encodings.
            attn_mask (Optional[torch.Tensor], optional): Attention mask to apply during self-attention.
                Defaults to None.
            is_causal (bool, optional): Whether to apply causal masking for autoregressive decoding.
                Defaults to True.
            kv_cache (Optional[Cache], optional): Key-value cache for efficient decoding.
                Defaults to None.
            curr_pos_id (Optional[torch.Tensor], optional): Current position IDs for decoding.
                Defaults to None.
            decode (bool, optional): Whether the model is in decoding mode.
                Defaults to False.
        Returns:
            torch.Tensor: Output tensor.
        """
        out = self.attn(
            self.ln_1(x),
            freqs_cis=freqs_cis,
            attn_mask=attn_mask,
            is_causal=is_causal,
            kv_cache=kv_cache,
            curr_pos_id=curr_pos_id,
            decode=decode,
        )
        x = x + out
        x = x + self.mlp(self.ln_2(x))
        return x
