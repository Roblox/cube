from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import DiagonalGaussianDistribution


class DeepCompressionBottleNeck(nn.Module):
    def __init__(
        self,
        width: int,
        embed_dim: Optional[int] = None,
        adaptive_embed_dim: Optional[List[int]] = None,
    ):
        super().__init__()

        embed_dim = embed_dim or width
        self.adaptive_embed_dim = adaptive_embed_dim
        self.embed_dim = embed_dim

        # ensure adaptive embed dim has enough length
        if adaptive_embed_dim is not None and len(adaptive_embed_dim) > 0:
            max_embed_dim = max(embed_dim, max(adaptive_embed_dim))
        else:
            max_embed_dim = embed_dim

        self.c_in = nn.Linear(width, max_embed_dim * 2)
        self.c_out = nn.Linear(embed_dim, width)

    def kl_embed(self, moments: torch.Tensor, sample_posterior: bool = False):
        posterior = None

        posterior = DiagonalGaussianDistribution(moments, feat_dim=-1)
        if sample_posterior:
            kl_embed = posterior.sample()
        else:
            kl_embed = posterior.mode()

        return kl_embed, {"posterior": posterior}

    def forward(self, z: torch.Tensor, **kwargs):
        # project
        z_e = self.c_in(z)
        z_e, ret_dict = self.kl_embed(z_e)

        if self.adaptive_embed_dim is not None:
            # if the model was trained with adaptive embed dim, the checkpoint
            # can carry more dimensions than the configured ``embed_dim``.
            z_e = z_e[..., : self.embed_dim]

        ret_dict["z"] = z_e
        z = F.linear(z_e, self.c_out.weight[:, : z_e.shape[-1]], self.c_out.bias)
        return z, ret_dict
