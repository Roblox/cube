import logging
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from safetensors.torch import load_file

from cube_part.models.transformers.attention import (
    EncoderCrossAttentionLayer,
    EncoderLayer,
    init_linear,
    init_tfixup,
)
from cube_part.models.transformers.norm import LayerNorm
from cube_part.utils.base import BaseModule

from ...utils.config import dict_to_namespace
from .bottlenecks.dcae import DeepCompressionBottleNeck
from .utils import AutoEncoder, get_embedder

logger = logging.getLogger(__name__)


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int, bias: bool = True):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, embed_dim, bias=bias)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.apply(partial(init_linear, embed_dim=embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class OneDGridEncoder(nn.Module):
    def __init__(
        self,
        embedder,
        num_latents: int,
        point_feats: int,
        embed_point_feats: bool,
        width: int,
        num_heads: int,
        num_layers: int,
        grid_size: int = 16,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.embedder = embedder

        # generate grid points
        ax = torch.linspace(-1, 1, grid_size)
        query = torch.meshgrid(ax, ax, ax, indexing="ij")
        query = self.embedder(torch.stack(query, dim=-1).view(-1, 3))
        # register grid query points
        self.register_buffer("grid_query", query, persistent=False)

        # register pad query
        self.query = nn.Parameter(torch.empty([num_latents, width]).uniform_(-1, 1))

        self.num_latents = num_latents
        self.embed_point_feats = embed_point_feats
        in_dim = (
            self.embedder.out_dim * 2
            if self.embed_point_feats
            else self.embedder.out_dim + point_feats
        )

        self.feat_in = nn.Linear(in_dim, width)
        self.query_in = nn.Linear(self.embedder.out_dim, width)

        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.blocks.append(
                    EncoderCrossAttentionLayer(
                        embed_dim=width,
                        num_heads=num_heads,
                        dropout=dropout,
                        eps=eps,
                    )
                )
            else:
                self.blocks.append(
                    EncoderLayer(
                        embed_dim=width, num_heads=num_heads, dropout=dropout, eps=eps
                    )
                )
        self.ln_f = LayerNorm(width, eps=eps)

        init_tfixup(self, num_layers)

    def _embed(self, pts, feats):
        data = self.embedder(pts)
        if feats is not None:
            if self.embed_point_feats:
                feats = self.embedder(feats)
            data = torch.cat([data, feats], dim=-1)
        return data

    @torch.compile(fullgraph=True)
    def _forward(self, x):
        batch_size = x.shape[0]
        grid_hidden_states = (
            self.query_in(self.grid_query).unsqueeze(0).expand(batch_size, -1, -1)
        )
        hidden_states = self.feat_in(x)

        for i, block in enumerate(self.blocks):
            if i == 0:
                hidden_states = self.blocks[0](grid_hidden_states, hidden_states)
                hidden_states = torch.cat(
                    [hidden_states, self.query.unsqueeze(0).expand(batch_size, -1, -1)],
                    dim=1,
                )
            else:
                hidden_states = block(hidden_states)

        hidden_states = self.ln_f(hidden_states)
        return hidden_states

    def forward(self, pts: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            pts (torch.Tensor): [B, N, 3]
            feats (torch.Tensor): [B, N, C]
        """

        # prepare data
        x = self._embed(pts, feats)

        hidden_states = self._forward(x)
        hidden_states = torch.split(
            hidden_states, [self.grid_query.shape[0], self.query.shape[0]], dim=1
        )
        return hidden_states[1]


class OneDGridDecoder(nn.Module):
    def __init__(
        self,
        num_latents: int,
        width: int,
        num_heads: int,
        num_layers: int,
        embedder,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.query_in = nn.Linear(embedder.out_dim, width)

        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                EncoderLayer(
                    embed_dim=width, num_heads=num_heads, dropout=dropout, eps=eps
                )
                for _ in range(num_layers)
            ]
        )

        init_tfixup(self, num_layers)

    @torch.compile(fullgraph=True)
    def _forward(self, h):
        h = self.drop(h)
        for block in self.blocks:
            h = block(h)
        return h

    def forward(self, z, query):
        batch_size = z.shape[0]
        hidden_states = self.query_in(query).unsqueeze(0).expand(batch_size, -1, -1)
        hidden_states = torch.cat([hidden_states, z], dim=1)

        hidden_states = self._forward(hidden_states)
        hidden_states = torch.split(hidden_states, [query.shape[0], z.shape[1]], dim=1)
        return hidden_states[0]


class OneDBottleNeck(nn.Module):
    """Thin wrapper around the inner bottleneck block so checkpoint keys
    line up with ``bottleneck.block.*``."""

    def __init__(self, block=None) -> None:
        super().__init__()
        self.block = block

    def forward(self, h: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, dict]:
        if self.block is None:
            return h, {}
        return self.block(h, **kwargs)


class OneDOccupancyDecoder(nn.Module):
    def __init__(
        self, embedder, out_features: int, width: int, num_heads: int, eps=1e-6
    ) -> None:
        super().__init__()

        self.embedder = embedder
        self.query_in = MLPEmbedder(self.embedder.out_dim, width)

        self.attn_out = EncoderCrossAttentionLayer(embed_dim=width, num_heads=num_heads)
        self.ln_f = LayerNorm(width, eps=eps)
        self.c_head = nn.Linear(width, out_features)

    def query(self, queries: torch.Tensor):
        return self.query_in(self.embedder(queries))

    def forward(
        self,
        queries: torch.Tensor,
        latents: torch.Tensor,
        skip_query_transform: bool = False,
    ):
        if not skip_query_transform:
            queries = self.query(queries)
        x = self.attn_out(queries, latents)
        x = self.c_head(self.ln_f(x))
        return x


class OneDGridAutoEncoder(AutoEncoder):
    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_path: Optional[str] = None

        # point features embedding
        embed_type: str = "fourier"
        num_freqs: int = 8
        include_pi: bool = False
        point_feats: int = 0
        embed_point_feats: bool = False

        # network params
        num_encoder_latents: int = 256
        embed_dim: int = 12
        width: int = 768
        num_heads: int = 12
        out_dim: int = 1
        dropout: float = 0.0
        eps: float = 1e-6

        num_encoder_layers: int = 1
        num_decoder_layers: int = 23

        adaptive_embed_dim: List[int] = field(default_factory=list)

    cfg: Config

    @torch.compiler.disable(recursive=True)
    def configure(self) -> None:
        super().configure()
        self.cfg = OmegaConf.to_container(self.cfg)
        self.cfg = dict_to_namespace(self.cfg)

        self.embedder = get_embedder(
            embed_type=self.cfg.embed_type,
            num_freqs=self.cfg.num_freqs,
            include_pi=self.cfg.include_pi,
        )

        self.encoder = OneDGridEncoder(
            embedder=self.embedder,
            num_latents=self.cfg.num_encoder_latents,
            point_feats=self.cfg.point_feats,
            embed_point_feats=self.cfg.embed_point_feats,
            width=self.cfg.width,
            num_heads=self.cfg.num_heads,
            num_layers=self.cfg.num_encoder_layers,
            dropout=self.cfg.dropout,
            eps=self.cfg.eps,
        )

        block = DeepCompressionBottleNeck(
            width=self.cfg.width,
            embed_dim=self.cfg.embed_dim,
            adaptive_embed_dim=getattr(self.cfg, "adaptive_embed_dim", None),
        )
        self.bottleneck = OneDBottleNeck(block=block)

        self.decoder = OneDGridDecoder(
            num_latents=self.cfg.num_encoder_latents,
            embedder=self.embedder,
            width=self.cfg.width,
            num_heads=self.cfg.num_heads,
            num_layers=self.cfg.num_decoder_layers,
            dropout=0.0,
            eps=self.cfg.eps,
        )

        self.occupancy_decoder = OneDOccupancyDecoder(
            embedder=self.embedder,
            out_features=self.cfg.out_dim,
            width=self.cfg.width,
            num_heads=self.cfg.num_heads,
            eps=self.cfg.eps,
        )

        if self.cfg.pretrained_model_path is not None:
            path = self.cfg.pretrained_model_path
            if path.endswith(".safetensors"):
                state_dict = load_file(path)
            else:
                obj = torch.load(path, map_location="cpu")
                state_dict = (
                    obj.get("state_dict", obj) if isinstance(obj, dict) else obj
                )
            m, u = self.load_state_dict(state_dict, strict=False)
            if len(m) > 0:
                logger.warning(f"==> Missing parameters from checkpoint: {m}")
            if len(u) > 0:
                logger.warning(f"==> Unknown parameters from checkpoint: {u}")

    def encode(self, x: torch.Tensor, **kwargs):
        if self.cfg.point_feats > 3:  # VAE has sharp bit
            if isinstance(x, dict):
                num_coarse_points = x["coarse"].shape[1]
                if "sharp" in x:
                    x = torch.cat([x["coarse"], x["sharp"]], dim=1)
                else:
                    x = x["coarse"]
            else:
                num_coarse_points = x.shape[1]

            sharp_mask = torch.zeros([*x.shape[:-1], 1], device=x.device, dtype=x.dtype)
            sharp_mask[:, num_coarse_points:, :] = 1
            x = torch.cat([x, sharp_mask], dim=-1)

        elif isinstance(x, dict):
            x = x["coarse"]

        pts, feats = x[..., :3], x[..., 3:]
        z_e = self.encoder(pts, feats)

        # quantize or kl
        z, d = self.bottleneck(z_e, **kwargs)

        return z_e, z, None, d

    def decode(self, z: torch.Tensor):
        h = self.decoder(z, self.encoder.grid_query)
        return h

    def query(
        self,
        queries: torch.Tensor,
        latents: torch.Tensor,
        skip_query_transform: bool = False,
    ):
        logits = self.occupancy_decoder(queries, latents, skip_query_transform).squeeze(
            -1
        )
        return logits
