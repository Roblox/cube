from typing import Union

import torch
import torch.nn as nn


@torch.compile(fullgraph=True)
def fused_rms_norm(x: torch.Tensor, weight: Union[float, nn.Parameter], eps: float):
    x = x.float()
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True).add_(eps))) * weight


class LayerNorm(nn.LayerNorm):
    def forward(self, input: torch.Tensor, *args, **kwargs):
        y = super().forward(input.float())
        return y.type_as(input)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else 1.0

    def forward(self, x):
        # NOTE for normalization layers, we force them to run in full precision, same as layer norm
        return fused_rms_norm(x, weight=self.weight, eps=self.eps).type_as(x)
