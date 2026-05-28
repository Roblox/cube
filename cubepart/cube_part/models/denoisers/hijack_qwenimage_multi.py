"""
Modify the forward method of Qwen-Image https://github.com/QwenLM/Qwen-Image
See Apache 2.0 license: https://github.com/QwenLM/Qwen-Image/blob/main/LICENSE
"""

import types
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from diffusers import AutoModel, QwenImageTransformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from cube_part.models.transformers.gated_attention import QwenGatedTransformerBlock

from .utils import _basic_init, _embed_init, _zero_init, replace_norm_with_fp32


# This function wraps the transformer blocks to enable torch.compile.
@torch.compile(fullgraph=True)
def _hijack_forward_compiled(
    model,
    hidden_states,
    encoder_hidden_states,
    encoder_hidden_states_mask,
    temb,
    image_rotary_emb,
    attention_kwargs,
):
    multi_index_block = 0
    trunc_image_rotary_emb = (
        image_rotary_emb[0][: hidden_states.shape[1]],
        image_rotary_emb[1],
    )
    num_multi = image_rotary_emb[0].shape[0] // hidden_states.shape[1]
    multi_temb = temb.unflatten(0, (-1, num_multi))[:, 0]

    for index_block, block in enumerate(model.transformer_blocks):
        if index_block in model.multi_attention_layer_index:
            multi_hidden_states = model.multi_transformer_blocks[multi_index_block](
                hidden_states=hidden_states.reshape(
                    -1, num_multi * hidden_states.shape[1], *hidden_states.shape[2:]
                ),
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                temb=multi_temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=attention_kwargs,
            )
            multi_hidden_states = multi_hidden_states.reshape(*hidden_states.shape)
            multi_index_block += 1

        if torch.is_grad_enabled() and model.gradient_checkpointing:
            encoder_hidden_states, hidden_states = model._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                encoder_hidden_states_mask,
                temb,
                trunc_image_rotary_emb,
            )

        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                temb=temb,
                image_rotary_emb=trunc_image_rotary_emb,
                joint_attention_kwargs={},
            )

        if index_block in model.multi_attention_layer_index:
            hidden_states = 0.5 * (hidden_states + multi_hidden_states)

    # Use only the image part (hidden_states) from the dual-stream blocks
    hidden_states = model.norm_out(hidden_states, temb)
    output = model.proj_out(hidden_states)

    # FIXME this to avoid unused parameters
    output = output + encoder_hidden_states.mean() * 0.0

    return output


# See the original implementation: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_qwenimage.py#L832
def hijack_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    encoder_hidden_states_mask: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_shapes: Optional[List[Tuple[int, int, int]]] = None,
    txt_seq_lens: Optional[List[int]] = None,
    guidance: torch.Tensor = None,  # TODO: this should probably be removed
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
) -> Union[torch.Tensor, Transformer2DModelOutput]:
    """
    The [`QwenTransformer2DModel`] forward method.

    Args:
        hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
            Input `hidden_states`.
        encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
            Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
        encoder_hidden_states_mask (`torch.Tensor` of shape `(batch_size, text_sequence_length)`):
            Mask of the input conditions.
        timestep ( `torch.LongTensor`):
            Used to indicate denoising step.
        attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
            tuple.

    Returns:
        If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
        `tuple` where the first element is the sample tensor.
    """
    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()

    hidden_states = self.img_in(hidden_states)

    encoder_hidden_states = self.txt_norm(encoder_hidden_states)
    encoder_hidden_states = self.txt_in(encoder_hidden_states)

    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000

    with torch.autocast(hidden_states.device.type, enabled=False):
        temb = (
            self.time_text_embed(timestep, timestep)
            if guidance is None
            else self.time_text_embed(timestep, guidance, timestep)
        )
        temb = temb.to(hidden_states.dtype)

    if self.pos_embed is not None:
        image_rotary_emb = self.pos_embed(
            img_shapes, txt_seq_lens, device=hidden_states.device
        )

    else:
        image_rotary_emb = None

    output = _hijack_forward_compiled(
        self,
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        encoder_hidden_states_mask=encoder_hidden_states_mask,
        temb=temb,
        image_rotary_emb=image_rotary_emb,
        attention_kwargs=attention_kwargs,
    )

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)


def init_weights(model):
    # init all layers
    model.apply(_basic_init)

    # embedding layers
    model.img_in.apply(_embed_init)
    model.txt_in.apply(_embed_init)
    model.time_text_embed.apply(_embed_init)

    # zero init layers
    for block in model.transformer_blocks:
        block.img_mod.apply(_zero_init)
        block.txt_mod.apply(_zero_init)

    for block in model.multi_transformer_blocks:
        block.img_mod.apply(_zero_init)

    model.norm_out.apply(_zero_init)
    model.proj_out.apply(_zero_init)


def build_qwenimage_multi_model(
    model_type: str,
    in_channels: int,
    condition_channels: int,
    num_layers: int,
    enable_mrope: bool = False,
    multi_attention_layer_index: Optional[List[int]] = None,
    **kwargs,
):
    model_config = AutoModel.load_config(model_type, subfolder="transformer")

    # set input/output channels
    model_config["in_channels"] = in_channels
    if "out_channels" in model_config:
        model_config["out_channels"] = in_channels

    # set condition channels
    for key in ["condition_channels", "cross_attention_dim", "joint_attention_dim"]:
        if key in model_config:
            model_config[key] = condition_channels

    # set model params
    model_config["patch_size"] = 1
    model_config["num_attention_heads"] = 12

    # disable the multimodal pos emb from qwenimage
    if not enable_mrope:
        model_config["axes_dims_rope"] = [model_config["attention_head_dim"], 0, 0]
    model_config["num_layers"] = num_layers

    model = QwenImageTransformer2DModel.from_config(model_config)

    if multi_attention_layer_index is not None:
        multi_transformer_blocks = nn.ModuleList(
            [
                QwenGatedTransformerBlock(
                    dim=model.inner_dim,
                    num_attention_heads=model_config["num_attention_heads"],
                    attention_head_dim=model_config["attention_head_dim"],
                )
                for _ in multi_attention_layer_index
            ]
        )
        model.add_module("multi_transformer_blocks", multi_transformer_blocks)
        setattr(model, "multi_attention_layer_index", multi_attention_layer_index)

    model = model.to(torch.float32)

    # replace all normalizations with fp32 versions
    model = replace_norm_with_fp32(model)
    init_weights(model)

    # set all parameters to be trainable
    model.train().requires_grad_(True)

    model.forward = types.MethodType(hijack_forward, model)

    return model
