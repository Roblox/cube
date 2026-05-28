import logging
import math
from typing import Optional, Union

import torch

import numpy as np

from diffusers import (
    DPMSolverMultistepScheduler,
    FlowMatchEulerDiscreteScheduler,
    FlowMatchHeunDiscreteScheduler,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    retrieve_timesteps,
)
from tqdm import tqdm

from cube_part.pipelines.base import ShapeInput
from cube_part.systems.shape_denoiser import ShapeDenoiserSystem
from cube_part.utils.config import load_config
from cube_part.utils.runtime import Benchmarker

logger = logging.getLogger(__name__)
FORMAT = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)


class ShapeDenoiserPipeline:  # to be closer to diffusers
    def __init__(
        self,
        config_path: str,
        checkpoint_path: Optional[str] = None,
        vae_checkpoint_path: Optional[str] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        extract_geometry_fn_name: str = "extract_geometry_naive",
    ) -> None:
        """
        Initialize the ShapeDenoiserPipeline.

        Args:
            config_path (str): Path to the config file.
            checkpoint_path (Optional[str]): Path to the diffusion-system
                checkpoint. Overrides ``system.pretrained_model_path``
                in the YAML config.
            vae_checkpoint_path (Optional[str]): Path to the shape VAE
                ``.safetensors`` checkpoint. Overrides
                ``system.shape_model.pretrained_model_path`` in the YAML
                config.
            device (Optional[Union[torch.device, str, int]]): Device to use for inference.
            extract_geometry_fn_name (str): Name of the function to use for geometry extraction.
        """

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        config = load_config(config_path)
        if checkpoint_path is not None:
            logging.info(f"Using the specified checkpoint: {checkpoint_path}")
            config.system.pretrained_model_path = checkpoint_path
        else:
            logging.warning("No checkpoint provided, using config default!")

        if vae_checkpoint_path is not None:
            logging.info(f"Using the specified VAE checkpoint: {vae_checkpoint_path}")
            config.system.shape_model.pretrained_model_path = vae_checkpoint_path

        # hard coded options for inference only
        config.system.attn_implementation = "sdpa"
        config.system.gradient_checkpointing = False

        system = ShapeDenoiserSystem(config.system).eval().to(device)

        torch.set_grad_enabled(False)

        self.system = system
        self.device = torch.device(device)
        self.cfg = config
        self.extract_geometry_fn_name = extract_geometry_fn_name

    @torch.no_grad()
    def encode_shape(self, surface: torch.Tensor, return_mesh: bool = False):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, z, _, result_dict = self.system.shape_model.encode(surface)
            if return_mesh:
                latents = self.system.shape_model.decode(z)
                mesh_v_f, _ = self.system.shape_model.extract_geometry(
                    latents, chunk_size=100_000, use_warp=True
                )
            else:
                mesh_v_f = None

            return result_dict["z"], mesh_v_f

    @torch.no_grad()
    def decode_shape(
        self, shape_ids, resolution_base: float = 8.0, chunk_size: int = 100_000
    ):
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            # vq-vae
            latents = self.system.shape_model.decode(
                self.system.shape_model.bottleneck.block.c_out(shape_ids)
            )
            bounds = 1.0 + 1.0 / (2 * 2**resolution_base)
            meshes, _ = self.system.shape_model.extract_geometry(
                latents,
                resolution_base=resolution_base,
                chunk_size=chunk_size,
                use_warp=True,
                bounds=bounds,
                fn_name=self.extract_geometry_fn_name,
            )
        return meshes

    def prepare_latents(self, batch_size: int, num_latents: int, seed=None):
        # seed random generator
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        # get init noise
        latents = torch.randn(
            [batch_size, num_latents, self.system.shape_model.cfg.embed_dim],
            device=self.device,
            dtype=torch.float32,
            generator=generator,
        )

        return latents

    def prepare_noise_scheduler(self, scheduler_type, timeshift: float = 1.0):
        SCHEDULER_TYPES = {
            "dpm_solver": DPMSolverMultistepScheduler,
            "euler": FlowMatchEulerDiscreteScheduler,
            "heun": FlowMatchHeunDiscreteScheduler,
        }

        if scheduler_type not in SCHEDULER_TYPES:
            raise ValueError(f"Scheduler type {scheduler_type} is not supported")
        elif scheduler_type == "dpm_solver":
            noise_scheduler = DPMSolverMultistepScheduler(
                use_flow_sigmas=True,
                flow_shift=timeshift,
                prediction_type="flow_prediction",
            )
        else:
            noise_scheduler = SCHEDULER_TYPES[scheduler_type](shift=timeshift)

        return noise_scheduler

    def prepare_sigmas(self, scheduler_type: str, num_inference_steps: int):
        if scheduler_type != "dpm_solver":
            # NOTE do not add timeshift here to avoid shifting twice
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        else:
            sigmas = None
        return sigmas


class PartShapeDenoiserPipeline(ShapeDenoiserPipeline):
    def check_inputs(self, shape_input):
        if shape_input.prompt is None:
            raise ValueError(f"Provide prompt input for {self.__class__}.")
        if shape_input.latents is None:
            raise ValueError(f"Provide latent input for {self.__class__}.")

    @torch.no_grad()
    def input_to_part_shape(
        self,
        shape_input: ShapeInput,
        guidance_scale: float = 7.5,
        resolution_base: float = 9.0,
        chunk_size: int = 100_000,
        timer: Benchmarker = Benchmarker(enabled=False),
        seed: Optional[int] = 0,
        scheduler_type: str = "dpm_solver",
        timeshift: float = 1.0,
        num_inference_steps: int = 50,
        output_mesh: bool = True,
        output_hidden_states: bool = False,
        **kwargs,
    ):
        for arg, value in locals().items():
            if arg != "self":  # Exclude `self`
                logging.info(f"{arg}:{value}")

        self.check_inputs(shape_input)

        prompts = (
            shape_input.prompt
            if isinstance(shape_input.prompt[0], list)
            else [shape_input.prompt]
        )

        batch_size = len(prompts)
        num_parts = 8
        num_latents = self.system.shape_model.cfg.num_encoder_latents

        sample_mask = torch.zeros(
            [batch_size, num_parts + 1], dtype=torch.bool, device=self.device
        )
        sample_mask[:, -1] = True

        for i, prompt in enumerate(prompts):
            sample_mask[i, : len(prompt)] = True
            if len(prompt) < num_parts:
                prompts[i] = prompt + [""] * (num_parts - len(prompt))

        prompts = self.system.apply_part_text_template(prompts)
        attention_mask = torch.repeat_interleave(sample_mask, num_latents, dim=-1).view(
            batch_size, 1, 1, -1
        )
        attention_kwargs = {"attention_mask": attention_mask}

        latents = self.prepare_latents(batch_size * num_parts, num_latents, seed=seed)
        latents = latents.unflatten(0, (batch_size, num_parts))  # [b, f, l, d]
        input_latents = self.system._normalize_vae_latents(
            shape_input.latents
        ).unsqueeze_(1)
        img_shapes = [
            [(1, int(math.sqrt(num_latents)), int(math.sqrt(num_latents)))]
            * (num_parts + 1)
        ]

        # prepare for cfg
        if guidance_scale > 0.0:
            prompts = prompts + [self.system.default_negative_prompt] * len(prompts)
            latents = torch.cat([latents, latents], dim=0)
            if self.system.cfg.diffusion_enable_shape_guidance:
                uncond_input_latents = torch.zeros_like(input_latents)
            else:
                uncond_input_latents = input_latents
            input_latents = torch.cat([input_latents, uncond_input_latents], dim=0)

        # get text embed
        encoder_hidden_states, encoder_attention_mask = self.system.base_model(prompts)

        # prepare timesteps
        noise_scheduler = self.prepare_noise_scheduler(
            scheduler_type, timeshift=timeshift
        )
        sigmas = self.prepare_sigmas(scheduler_type, num_inference_steps)
        timesteps, num_inference_steps = retrieve_timesteps(
            noise_scheduler, num_inference_steps, device=self.device, sigmas=sigmas
        )

        # generation loop
        noise_scheduler.set_begin_index(0)
        for _, t in enumerate(tqdm(timesteps)):
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(encoder_hidden_states.shape[0]).to(latents.dtype)

            with torch.autocast(self.device.type, dtype=torch.bfloat16):
                model_pred = self.system._forward_diffusion_model(
                    torch.cat([latents, input_latents], dim=1).flatten(
                        0, 1
                    ),  # [b, f+1, l, d] -> [b*(f+1), l, d]
                    timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    img_shapes=img_shapes,
                    attention_kwargs=attention_kwargs,
                ).to(latents.dtype)
            # throw away conditions
            model_pred = model_pred.unflatten(0, (latents.shape[0], -1))[:, :-1]

            # convert to velocity
            model_pred = (latents - model_pred) / (
                t / noise_scheduler.config.num_train_timesteps
            ).clamp_min(self.system.cfg.timestep_eps)

            if guidance_scale > 0.0:
                # gamma = guidance_scale * guidance_schedule(
                #     t / noise_scheduler.config.num_train_timesteps
                # )
                gamma = guidance_scale

                # do cfg
                latents, _ = latents.chunk(2, dim=0)
                cond_pred, uncond_pred = model_pred.chunk(2, dim=0)
                comb_pred = uncond_pred + gamma * (cond_pred - uncond_pred)

                # naive path
                model_pred = comb_pred

                latents = noise_scheduler.step(
                    model_pred.flatten(0, 1),
                    t,
                    latents.flatten(0, 1),
                    return_dict=False,
                )[0].unflatten(0, (-1, num_parts))
                latents = torch.cat([latents, latents], dim=0)
            else:
                latents = noise_scheduler.step(
                    model_pred.flatten(0, 1),
                    t,
                    latents.flatten(0, 1),
                    return_dict=False,
                )[0].unflatten(0, (-1, num_parts))

        if guidance_scale > 0.0:
            latents, _ = latents.chunk(2, dim=0)

        # resume original scaling
        latents = self.system._unnormalize_vae_latents(
            latents.view(-1, *latents.shape[-2:])
        )
        sample_mask = sample_mask[:, :-1].flatten().view(-1, 1, 1).expand_as(latents)
        latents = latents[sample_mask].reshape(-1, *latents.shape[-2:])

        if not output_mesh:
            return latents

        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            logging.info("shape decoding: start")
            with timer.benchmark("vq_decode"):
                mesh = self.decode_shape(
                    latents.float(),
                    resolution_base=resolution_base,
                    chunk_size=chunk_size,
                )
                logging.info("shape decoding: done")

        if output_hidden_states:
            return mesh, latents
        return mesh
