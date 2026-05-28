"""Inference-only shape diffusion system.

Keeps only the components needed by ``cube_part.pipelines.shape_denoiser``
to run inference end-to-end:

* ``configure()``: build the text encoder, the shape autoencoder, and the
  diffusion transformer, then optionally load a pretrained checkpoint.
* ``_forward_diffusion_model()``: the core forward pass used inside the
  diffusion sampling loop.
* ``apply_part_text_template()``: prompt templating used by the
  ``PartShapeDenoiserPipeline``.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from safetensors.torch import load_file

from cube_part.models.autoencoders.one_d_grid_autoencoder import OneDGridAutoEncoder
from cube_part.models.denoisers.condition_processors import build_condition_processor
from cube_part.models.denoisers.hijack_qwenimage_multi import (
    build_qwenimage_multi_model,
)
from cube_part.utils.base import BaseSystem
from cube_part.utils.config import dict_to_namespace


class ShapeDenoiserSystem(BaseSystem):
    @dataclass
    class Config(BaseSystem.Config):
        shape_model: dict = field(default_factory=dict)
        per_channel_shape_model_normalization: bool = True

        # text-encoder ("base model") config
        base_model_type: str = "Qwen/Qwen3-VL-4B-Instruct"
        base_model_path: Optional[str] = None
        base_tokenizer_max_length: int = 128
        base_padding_mode: Optional[str] = None

        # diffusion model config
        diffusion_model_type: str = "Qwen/Qwen-Image"
        diffusion_model_config_path: Optional[str] = None

        diffusion_num_layers: int = 23
        diffusion_enable_mrope: bool = False
        diffusion_multi_attention_layer_index: List[int] = field(
            default_factory=lambda: [0, 4, 8, 16]
        )

        diffusion_enable_shape_rope: bool = True
        diffusion_enable_shape_guidance: bool = False

        # only ``timestep_eps`` is used at inference time
        timestep_shift: Optional[float] = None
        timestep_shift_base: int = 4096
        timestep_eps: float = 5.0e-2

        attn_implementation: str = "sdpa"
        gradient_checkpointing: bool = False

    cfg: Config

    @torch.compiler.disable(recursive=True)
    def configure(self) -> None:
        shape_model = OneDGridAutoEncoder(self.cfg.shape_model)
        self.shape_model = shape_model.requires_grad_(False).eval()

        shape_model_normalization_dim = (
            self.shape_model.cfg.embed_dim
            if self.cfg.per_channel_shape_model_normalization
            else 1
        )
        self.shape_model_shift = nn.Parameter(
            torch.zeros(shape_model_normalization_dim, dtype=torch.float32),
            requires_grad=False,
        )
        self.shape_model_scale = nn.Parameter(
            torch.ones(shape_model_normalization_dim, dtype=torch.float32),
            requires_grad=False,
        )

        # text encoder
        self.base_model = build_condition_processor(
            self.cfg.base_model_type,
            tokenizer_max_length=self.cfg.base_tokenizer_max_length,
            padding_mode=self.cfg.base_padding_mode,
            model_path=self.cfg.base_model_path,
            attn_implementation=self.cfg.attn_implementation,
        )
        default_negative_prompt = ""
        self.default_negative_prompt = self.base_model.prompt_template_encode.format(
            default_negative_prompt
        )

        # diffusion noise scheduler
        timestep_shift = self.cfg.timestep_shift
        if timestep_shift is None:
            timestep_shift = math.sqrt(
                self.shape_model.cfg.num_encoder_latents
                * self.shape_model.cfg.embed_dim
                / self.cfg.timestep_shift_base
            )
        self.diffusion_noise_scheduler = FlowMatchEulerDiscreteScheduler(
            shift=timestep_shift
        )

        # diffusion transformer
        self.diffusion_model = build_qwenimage_multi_model(
            self.cfg.diffusion_model_type,
            config_path=self.cfg.diffusion_model_config_path,
            in_channels=self.shape_model.cfg.embed_dim,
            condition_channels=self.base_model.hidden_size,
            num_layers=self.cfg.diffusion_num_layers,
            enable_mrope=self.cfg.diffusion_enable_mrope,
            multi_attention_layer_index=list(
                self.cfg.diffusion_multi_attention_layer_index
            ),
            attn_implementation=self.cfg.attn_implementation,
        )

        # convert structured config to a SimpleNamespace so existing attribute
        # access patterns (e.g. ``self.cfg.timestep_eps``) keep working
        self.cfg = OmegaConf.to_container(self.cfg)
        self.cfg = dict_to_namespace(self.cfg)

        if self.cfg.gradient_checkpointing:
            self.diffusion_model.enable_gradient_checkpointing()

        if self.cfg.pretrained_model_path is not None:
            path = self.cfg.pretrained_model_path
            if path.endswith(".safetensors"):
                state_dict = load_file(path)
            else:
                obj = torch.load(path, map_location="cpu")
                state_dict = (
                    obj.get("state_dict", obj) if isinstance(obj, dict) else obj
                )
            self.load_state_dict(state_dict, strict=False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_model.eval()
        self.shape_model.eval()
        return self

    def _normalize_vae_latents(self, latents):
        return latents * self.shape_model_scale + self.shape_model_shift

    def _unnormalize_vae_latents(self, latents):
        return (latents - self.shape_model_shift) / self.shape_model_scale

    def apply_part_text_template(self, batch_part_text: List[List[str]]):
        """Apply the chat template used by the parts diffusion pipeline."""
        batch_text = []
        for part_text in batch_part_text:
            base_prompt = "A image contains following parts: "
            format_parts = [f"Part {i + 1}: {e}" for i, e in enumerate(part_text) if e]
            base_prompt += ", ".join(format_parts)

            for e in part_text:
                prompt = base_prompt + f"Target to segment: {e}."
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "Describe the key features of the input image (color, shape, "
                            "size, texture, objects, background), then explain how the "
                            "user's text instruction should alter or modify the image. "
                            "Generate a new image that meets the user's requirements while "
                            "maintaining consistency with the original input where "
                            "appropriate."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ]
                batch_text.append(
                    self.base_model.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                )
            batch_text.append("")  # placeholder for the full mesh
        return batch_text

    def _forward_diffusion_model(
        self,
        noisy_model_input,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask,
        img_shapes=None,
        txt_seq_lens=None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if img_shapes is None:
            if self.cfg.diffusion_enable_shape_rope:
                if self.cfg.diffusion_enable_mrope:
                    height = int(math.sqrt(noisy_model_input.shape[1]))
                    width = noisy_model_input.shape[1] // height
                    img_shapes = [[(1, height, width)]]
                else:
                    img_shapes = [[(noisy_model_input.shape[1], 1, 1)]]
            else:
                img_shapes = [[(1, 1, 1)]]

        if txt_seq_lens is None:
            txt_seq_lens = [encoder_hidden_states.shape[1]]

        return self.diffusion_model(
            hidden_states=noisy_model_input,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_attention_mask,
            timestep=timestep
            / self.diffusion_noise_scheduler.config.num_train_timesteps,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            attention_kwargs=attention_kwargs,
            return_dict=False,
        )[0]
