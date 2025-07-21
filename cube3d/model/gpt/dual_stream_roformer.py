from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from cube3d.model.transformers.cache import Cache
from cube3d.model.transformers.dual_stream_attention import (
    DualStreamDecoderLayerWithRotaryEmbedding,
)
from cube3d.model.transformers.norm import LayerNorm
from cube3d.model.transformers.roformer import DecoderLayerWithRotaryEmbedding
from cube3d.model.transformers.rope import precompute_freqs_cis


class DualStreamRoformer(nn.Module):
    @dataclass
    class Config:
        checkpoint_path: str = ""
        n_layer: int = 12
        n_single_layer: int = 0
        rope_theta: float = 1000

        n_head: int = 16
        n_embd: int = 2048
        bias: bool = False  # bias in Linears and LayerNorms
        eps: float = 1e-6  # Norm eps

        shape_model_vocab_size: int = 4096
        shape_model_embed_dim: int = 16

        text_model_embed_dim: int = 512
        use_pooled_text_embed: bool = False

        encoder_with_cls_token: bool = True

        use_bbox: bool = False

    def __init__(self, cfg: Config) -> None:
        """
        Initializes the DualStreamRoFormer model.
        Args:
            cfg (Config): Configuration object containing model parameters.
        Attributes:
            cfg (Config): Stores the configuration object.
            text_proj (nn.Linear): Linear layer to project text model embeddings to the desired embedding dimension.
            shape_proj (nn.Linear, optional): Linear layer to project shape model embeddings to the desired embedding
                dimension
            vocab_size (int): Vocabulary size for the shape model, including special tokens.
            shape_bos_id (int): Token ID for the beginning-of-sequence (BOS) token for the shape model.
            shape_eos_id (int): Token ID for the end-of-sequence (EOS) token for the shape model.
            padding_id (int): Token ID for the padding token.
            transformer (nn.ModuleDict): Dictionary containing the following components:
                - wte (nn.Embedding): Embedding layer for the vocabulary.
                - dual_blocks (nn.ModuleList): List of dual-stream decoder layers with rotary embeddings.
                - single_blocks (nn.ModuleList): List of single-stream decoder layers with rotary embeddings.
                - ln_f (LayerNorm): Layer normalization applied to the final output.
            lm_head (nn.Linear): Linear layer mapping the final embeddings to the vocabulary size for language modeling.
        """

        super().__init__()

        self.cfg = cfg

        self.text_proj = nn.Linear(
            in_features=self.cfg.text_model_embed_dim,
            out_features=self.cfg.n_embd,
            bias=self.cfg.bias,
        )

        self.shape_proj = nn.Linear(self.cfg.shape_model_embed_dim, self.cfg.n_embd)

        self.vocab_size = self.cfg.shape_model_vocab_size

        def add_special_token():
            token_id = self.vocab_size
            self.vocab_size += 1
            return token_id

        self.shape_bos_id = add_special_token()
        self.shape_eos_id = add_special_token()
        self.padding_id = add_special_token()

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(
                    self.vocab_size,
                    self.cfg.n_embd,
                    padding_idx=self.padding_id,
                ),
                dual_blocks=nn.ModuleList(
                    [
                        DualStreamDecoderLayerWithRotaryEmbedding.from_config(
                            self.cfg, cond_pre_only=(i == self.cfg.n_layer - 1)
                        )
                        for i in range(self.cfg.n_layer)
                    ]
                ),
                single_blocks=nn.ModuleList(
                    [
                        DecoderLayerWithRotaryEmbedding.from_config(self.cfg)
                        for _ in range(self.cfg.n_single_layer)
                    ]
                ),
                ln_f=LayerNorm(
                    self.cfg.n_embd, elementwise_affine=False, eps=self.cfg.eps
                ),
            )
        )

        self.lm_head = nn.Linear(self.cfg.n_embd, self.vocab_size, bias=False)

        if self.cfg.use_bbox:
            self.bbox_proj = nn.Linear(3, self.cfg.n_embd)

    def encode_text(self, text_embed):
        """
        Encodes the given text embeddings by projecting them through a linear transformation.
        Args:
            text_embed (torch.Tensor): A tensor representing the text embeddings to be encoded.
        Returns:
            torch.Tensor: The projected text embeddings after applying the linear transformation.
        """

        return self.text_proj(text_embed)

    def encode_token(self, tokens):
        """
        Encodes the input tokens using the word token embedding layer of the transformer model.
        Args:
            tokens (torch.Tensor): A tensor containing the input tokens to be encoded.
        Returns:
            torch.Tensor: A tensor containing the encoded token embeddings.
        """

        return self.transformer.wte(tokens)

    def init_kv_cache(
        self,
        batch_size: int,
        cond_len: int,
        max_shape_tokens: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> list[Cache]:
        """
        Initializes the key-value cache for the transformer model.
        This method creates a list of `Cache` objects to store the key and value
        states for both dual-stream and single-stream transformer blocks. The
        cache is pre-allocated with zeros and is used to optimize the computation
        of attention mechanisms during model inference.
        Args:
            batch_size (int): The batch size for the input data.
            cond_len (int): The length of the conditioning sequence.
            max_shape_tokens (int): The maximum number of tokens in the shape sequence.
            dtype (torch.dtype): The data type for the tensors (e.g., torch.float32).
            device (torch.device): The device on which the tensors will be allocated
                (e.g., torch.device('cuda') or torch.device('cpu')).
        Returns:
            list[Cache]: A list of `Cache` objects containing pre-allocated key and
            value states for each transformer block.
        """
        num_heads = self.cfg.n_head
        max_all_tokens = cond_len + max_shape_tokens
        per_head_dim = self.cfg.n_embd // num_heads

        kv_cache = [
            Cache(
                key_states=torch.zeros(
                    (batch_size, num_heads, max_all_tokens, per_head_dim),
                    dtype=dtype,
                    device=device,
                ),
                value_states=torch.zeros(
                    (batch_size, num_heads, max_all_tokens, per_head_dim),
                    dtype=dtype,
                    device=device,
                ),
            )
            for _ in range(len(self.transformer.dual_blocks))
        ]
        kv_cache += [
            Cache(
                key_states=torch.zeros(
                    (batch_size, num_heads, max_shape_tokens, per_head_dim),
                    dtype=dtype,
                    device=device,
                ),
                value_states=torch.zeros(
                    (batch_size, num_heads, max_shape_tokens, per_head_dim),
                    dtype=dtype,
                    device=device,
                ),
            )
            for _ in range(len(self.transformer.single_blocks))
        ]
        return kv_cache

    def forward(
        self,
        embed: torch.Tensor,
        cond: torch.Tensor,
        kv_cache: Optional[list[Cache]] = None,
        curr_pos_id: Optional[torch.Tensor] = None,
        decode: bool = False,
    ):
        """
        Forward pass for the dual-stream RoFormer model.
        Args:
            embed (torch.Tensor): The input embedding tensor.
            cond (torch.Tensor): The conditioning tensor.
            kv_cache (Optional[list[Cache]]): A list of key-value caches for each layer, used for decoding. Default is None.
            curr_pos_id (Optional[torch.Tensor]): The current position ID tensor of shape (batch_size,). Required if `decode` is True. Default is None.
            decode (bool): Whether the model is in decoding mode. Default is False.
        Returns:
            torch.Tensor: The output logits tensor.
        """
        b, l = embed.shape[:2]
        s = cond.shape[1]
        device = embed.device

        attn_mask = torch.tril(
            torch.ones(s + l, s + l, dtype=torch.bool, device=device)
        )

        position_ids = torch.arange(l, dtype=torch.long, device=device)  # shape (t)
        position_ids = position_ids.unsqueeze_(0).expand(b, -1)

        s_freqs_cis = precompute_freqs_cis(
            dim=self.cfg.n_embd // self.cfg.n_head,
            t=position_ids,
            theta=self.cfg.rope_theta,
        )

        position_ids = torch.cat(
            [
                torch.zeros([b, s], dtype=torch.long, device=position_ids.device),
                position_ids,
            ],
            dim=1,
        )
        d_freqs_cis = precompute_freqs_cis(
            dim=self.cfg.n_embd // self.cfg.n_head,
            t=position_ids,
            theta=self.cfg.rope_theta,
        )

        if kv_cache is not None and decode:
            assert curr_pos_id is not None
            embed = embed[:, curr_pos_id, :]

        h = embed
        c = cond

        layer_idx = 0
        for block in self.transformer.dual_blocks:
            h, c = block(
                h,
                c=c,
                freqs_cis=d_freqs_cis,
                attn_mask=attn_mask,
                is_causal=True,
                kv_cache=kv_cache[layer_idx] if kv_cache is not None else None,
                curr_pos_id=curr_pos_id + s if curr_pos_id is not None else None,
                decode=decode,
            )
            layer_idx += 1
        for block in self.transformer.single_blocks:
            h = block(
                h,
                freqs_cis=s_freqs_cis,
                attn_mask=None,
                is_causal=True,
                kv_cache=kv_cache[layer_idx] if kv_cache is not None else None,
                curr_pos_id=curr_pos_id,
                decode=decode,
            )
            layer_idx += 1

        # Normalization
        h = self.transformer.ln_f(h)
        logits = self.lm_head(h)

        return logits
