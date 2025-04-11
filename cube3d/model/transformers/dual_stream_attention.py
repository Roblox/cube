from typing import Optional, Tuple

import torch
import torch.nn as nn

from cube3d.model.transformers.cache import Cache
from cube3d.model.transformers.norm import LayerNorm, RMSNorm
from cube3d.model.transformers.roformer import SwiGLUMLP
from cube3d.model.transformers.rope import scaled_dot_product_attention_with_rotary_emb


class DismantledPreAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        query: bool = True,
        bias: bool = True,
    ) -> None:
        """
        Initializes the DismantledPreAttention module.
        Args:
            embed_dim (int): The dimensionality of the embedding space.
            num_heads (int): The number of attention heads.
            query (bool, optional): Whether to include query-key projection. Defaults to True.
            bias (bool, optional): Whether to include bias in linear layers. Defaults to True.
        Raises:
            AssertionError: If `embed_dim` is not divisible by `num_heads`.
        """
        super().__init__()
        assert embed_dim % num_heads == 0
        self.query = query

        head_dim = embed_dim // num_heads
        # key, query, value projections for all heads, but in a batch
        if query:
            self.c_qk = nn.Linear(embed_dim, 2 * embed_dim, bias=False)
            self.q_norm = RMSNorm(head_dim)
        else:
            self.c_k = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_norm = RMSNorm(head_dim)
        self.c_v = nn.Linear(embed_dim, embed_dim, bias=bias)

        # (B, T, C) -> (B, nh, T, hs)
        self.to_mha = lambda x: x.view(*x.shape[:2], num_heads, -1).transpose(1, 2)

    def forward(self, x):
        """
        Forward pass for the dismantled pre-attention mechanism.
        Args:
            x (torch.Tensor): Input tensor of shape (..., input_dim).
        Returns:
            tuple: A tuple containing:
                - q (torch.Tensor or None): Query tensor after normalization and transformation,
                  or None if `self.query` is False.
                - k (torch.Tensor): Key tensor after normalization and transformation.
                - v (torch.Tensor): Value tensor after transformation.
        """

        if self.query:
            q, k = self.c_qk(x).chunk(2, dim=-1)
            q = self.q_norm(self.to_mha(q))
        else:
            q = None
            k = self.c_k(x)

        k = self.k_norm(self.to_mha(k))
        v = self.to_mha(self.c_v(x))

        return (q, k, v)


class DismantledPostAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        bias: bool = True,
        eps: float = 1e-6,
    ) -> None:
        """
        Initializes the DismantledPostAttention module.
        Args:
            embed_dim (int): The dimensionality of the embedding space.
            bias (bool, optional): Whether to include a bias term in the linear projection. Defaults to True.
            eps (float, optional): A small value added to the denominator for numerical stability in layer normalization. Defaults to 1e-6.
        """
        super().__init__()
        self.c_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.ln_3 = LayerNorm(embed_dim, elementwise_affine=False, eps=eps)
        self.mlp = SwiGLUMLP(embed_dim, embed_dim * 4, bias=bias)

    def forward(self, x, a):
        """
        Forward pass of the dual stream attention mechanism.
        Args:
            x (torch.Tensor): The input tensor to the model.
            a (torch.Tensor): The attention tensor to be combined with the input.
        Returns:
            torch.Tensor: The output tensor after applying the projection,
                          layer normalization, and MLP transformations.
        """

        x = x + self.c_proj(a)
        x = x + self.mlp(self.ln_3(x))
        return x


class DualStreamAttentionWithRotaryEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        cond_pre_only: bool = False,
        bias: bool = True,
    ):
        """
        Initializes the DualStreamAttention module.
        Args:
            embed_dim (int): The dimensionality of the embedding space.
            num_heads (int): The number of attention heads.
            cond_pre_only (bool, optional): If True, the conditional pre-attention
                will only process the key and value, not the query. Defaults to False.
            bias (bool, optional): Whether to include a bias term in the attention layers.
                Defaults to True.
        """
        super().__init__()

        self.cond_pre_only = cond_pre_only

        self.pre_x = DismantledPreAttention(
            embed_dim=embed_dim, num_heads=num_heads, query=True, bias=bias
        )

        self.pre_c = DismantledPreAttention(
            embed_dim=embed_dim, num_heads=num_heads, query=not cond_pre_only, bias=bias
        )

    def forward(
        self,
        x,
        c: Optional[torch.Tensor],
        freqs_cis,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        kv_cache: Optional[Cache] = None,
        curr_pos_id: Optional[torch.Tensor] = None,
        decode: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for dual stream Multi-Head Attention.

        Efficient single weight matrix multiplication with results split into query, key, value.

        Parameters
        ----------
        x : torch.Tensor
            Hidden states [B, L, D]
        c : torch.Tensor
            Condition [B, S, D]
        freqs_cis: torch.Tensor
            Precomputed RoPE matrix from precompute_freqs_cis [B, S+L, Hd]
        attn_mask : torch.Tensor, optional
            Attention mask [B, S+L, S+L], by default None
        kv_cache: None | Tensor
            key-value cache, but only if not None; if None - it means that it's disabled
            contains cache for keys and value from all previous steps
        kv_cache_cond: None | Tensor
            key-value cache, but only if not None; if None - it means that it's disabled
            contains cache for keys and value from all previous steps for the text conditioning.

        Returns
        -------
        torch.Tensor
            Hidden state output [B, L, D]
        """
        if kv_cache is None or not decode:
            # Either training or prefill
            qkv_c = self.pre_c(c)
            qkv_x = self.pre_x(x)
            # prepend condition stream
            # (B, nh, Tc, hs) + (B, nh, Tx, hs) -> (B, nh, Tc+Tx, hs)
            if self.cond_pre_only:
                q = qkv_x[0]
            else:
                q = torch.cat([qkv_c[0], qkv_x[0]], dim=2)
            k = torch.cat([qkv_c[1], qkv_x[1]], dim=2)
            v = torch.cat([qkv_c[2], qkv_x[2]], dim=2)

        else:
            # if using kv cache, query would only be the last token in the sequence, hence is_causal is False
            assert x.shape[1] == 1
            is_causal = False
            q, k, v = self.pre_x(x)

        if kv_cache is not None:
            if not decode:
                kv_cache.key_states[:, :, : k.shape[2], :].copy_(k)
                kv_cache.value_states[:, :, : k.shape[2], :].copy_(v)
            else:
                assert curr_pos_id is not None
                kv_cache.update(curr_pos_id, k, v)
            k = kv_cache.key_states
            v = kv_cache.value_states

        if attn_mask is not None:
            # trim attention mask to length
            if decode:
                assert curr_pos_id is not None
                attn_mask = attn_mask[..., curr_pos_id, :]
            else:
                attn_mask = attn_mask[..., -q.shape[2] :, :]

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
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

        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(x.shape[0], -1, x.shape[2])

        if y.shape[1] == x.shape[1]:
            y_c = None
            y_x = y
        else:
            assert c is not None, "Conditioning is required for dual stream attention"
            y_c, y_x = torch.split(y, [c.shape[1], x.shape[1]], dim=1)
        return y_x, y_c


class DualStreamDecoderLayerWithRotaryEmbedding(nn.Module):
    """Nicely wrapped decoder layer block for dual stream GPT model"""

    def __init__(
        self,
        embed_dim,
        num_heads: int,
        cond_pre_only: bool = False,
        bias: bool = True,
        eps: float = 1.0e-6,
    ) -> None:
        """
        Initializes the DualStreamDecoderLayerWithRotaryEmbedding module with optional conditional pre-only mode.
        Args:
            embed_dim (int): The dimensionality of the embedding space.
            num_heads (int): The number of attention heads.
            cond_pre_only (bool, optional): If True, applies conditional processing only before attention. Defaults to False.
            bias (bool, optional): If True, includes bias terms in the attention and post-attention layers. Defaults to True.
            eps (float, optional): A small value added for numerical stability in layer normalization. Defaults to 1.0e-6.
        """
        super().__init__()

        self.ln_1 = LayerNorm(embed_dim, elementwise_affine=False, eps=eps)
        self.ln_2 = LayerNorm(embed_dim, elementwise_affine=False, eps=eps)

        self.attn = DualStreamAttentionWithRotaryEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            cond_pre_only=cond_pre_only,
            bias=bias,
        )

        self.post_1 = DismantledPostAttention(embed_dim, bias=bias, eps=eps)
        if not cond_pre_only:
            self.post_2 = DismantledPostAttention(embed_dim, bias=bias, eps=eps)

    @classmethod
    def from_config(cls, cfg, cond_pre_only: bool = False):
        """
        Create an instance of the class using the provided configuration.
        Args:
            cfg: A configuration object containing the necessary parameters:
                - n_embd (int): The size of the embedding dimension.
                - n_head (int): The number of attention heads.
                - bias (bool): Whether to include a bias term.
                - eps (float): A small value added for numerical stability.
            cond_pre_only (bool, optional): If True, applies conditioning only in the pre-processing step.
                Defaults to False.
        Returns:
            An instance of the class initialized with the specified configuration.
        """

        return cls(
            cfg.n_embd,
            num_heads=cfg.n_head,
            cond_pre_only=cond_pre_only,
            bias=cfg.bias,
            eps=cfg.eps,
        )

    def forward(
        self,
        x,
        c,
        freqs_cis: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        kv_cache: Optional[Cache] = None,
        curr_pos_id: Optional[torch.Tensor] = None,
        decode: bool = False,
    ):
        """
        Forward pass for DualStreamDecoderLayerWithRotaryEmbedding.

        Parameters
        ----------
        x : torch.Tensor
            Hidden states [B, L, D]
        c : torch.Tensor
            Condition [B, S, D]
        freqs_cis: torch.Tensor
            Postional embedding from RoPE [B, S+L, hd]
        attn_mask : torch.Tensor, optional
            Attention mask [B, S+L, S+L], by default None
        kv_vache : torch.Tensor, optional
            kv_cache by default None

        Returns
        -------
        torch.Tensor
            Hidden state output [B, L, D]
        torch.Tensor
            kv_cache output [1, L, D]
        """
        a_x, a_c = self.attn(
            self.ln_1(x),
            # NOTE condition could be none if using kv cache
            self.ln_2(c) if c is not None else None,
            freqs_cis=freqs_cis,
            attn_mask=attn_mask,
            is_causal=is_causal,
            kv_cache=kv_cache,
            curr_pos_id=curr_pos_id,
            decode=decode,
        )
        x = self.post_1(x, a_x)
        if a_c is not None:
            c = self.post_2(c, a_c)
        else:
            c = None
        return x, c
