import torch
import torch.nn as nn

import diffusers

from cube_part.models.transformers.norm import LayerNorm, RMSNorm


def replace_norm_with_fp32(model):
    for name, module in model.named_children():
        if isinstance(module, nn.LayerNorm):
            # Create the new layer with appropriate arguments
            # This example assumes new_layer_type can take the same config/shape
            # as the old layer. The kwargs might need adjustment based on the
            # specific new layer (e.g., for GroupNorm, you need num_groups).
            setattr(
                model,
                name,
                LayerNorm(
                    module.normalized_shape,
                    eps=module.eps,
                    elementwise_affine=module.elementwise_affine,
                ),
            )
        elif isinstance(module, diffusers.models.normalization.RMSNorm):
            setattr(
                model,
                name,
                RMSNorm(
                    module.dim[0],
                    eps=module.eps,
                    elementwise_affine=module.elementwise_affine,
                ),
            )
        else:
            replace_norm_with_fp32(module)

    return model


def _basic_init(module):  # common transformer init
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _embed_init(module, std=0.02):  # init for projection layers
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _zero_init(module):  # adaln-zero init
    if isinstance(module, nn.Linear):
        nn.init.zeros_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
