import torch
from tqdm import tqdm
from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast

from cube3d.inference.logits_postprocesses import process_logits
from cube3d.inference.utils import load_config, load_model_weights, parse_structured
from cube3d.model.autoencoder.one_d_autoencoder import OneDAutoEncoder
from cube3d.model.gpt.dual_stream_roformer import DualStreamRoformer
from cube3d.model.transformers.cache import Cache


class Engine:
    def __init__(
        self,
        config_path: str,
        gpt_ckpt_path: str,
        shape_ckpt_path: str,
        device: torch.device,
    ):
        """
        Initializes the inference engine with the given configuration and checkpoint paths.
        Args:
            config_path (str): Path to the configuration file.
            gpt_ckpt_path (str): Path to the GPT model checkpoint file.
            shape_ckpt_path (str): Path to the shape model checkpoint file.
            device (torch.device): The device to run the models on (e.g., 'cpu' or 'cuda').
        Attributes:
            cfg (dict): Loaded configuration from the config file.
            device (torch.device): The device to run the models on.
            gpt_model (DualStreamRoformer): The GPT model initialized and loaded with weights.
            shape_model (OneDAutoEncoder): The shape model initialized and loaded with weights.
            text_model (CLIPTextModelWithProjection): The text model initialized from a pretrained model.
            text_tokenizer (CLIPTokenizerFast): The tokenizer for the text model.
            max_new_tokens (int): Maximum number of new tokens for the shape model.
            min_id (int): Minimum ID for the shape model codes.
            max_id (int): Maximum ID for the shape model codes.
        """

        self.cfg = load_config(config_path)
        self.device = device

        self.gpt_model = DualStreamRoformer(
            parse_structured(DualStreamRoformer.Config, self.cfg.gpt_model)
        )
        load_model_weights(
            self.gpt_model,
            gpt_ckpt_path,
        )
        self.gpt_model = self.gpt_model.eval().to(self.device)

        self.shape_model = OneDAutoEncoder(
            parse_structured(OneDAutoEncoder.Config, self.cfg.shape_model)
        )
        load_model_weights(
            self.shape_model,
            shape_ckpt_path,
        )
        self.shape_model = self.shape_model.eval().to(self.device)

        # copy vq codebook to gpt
        with torch.no_grad():
            codebook = self.shape_model.bottleneck.block.get_codebook()
            codebook = self.gpt_model.shape_proj(codebook).detach()
        self.gpt_model.transformer.wte.weight.data[: codebook.shape[0]] = codebook

        self.text_model = CLIPTextModelWithProjection.from_pretrained(
            self.cfg.text_model_pretrained_model_name_or_path,
            force_download=False,
            device_map=self.device,
        ).eval()
        self.text_tokenizer = CLIPTokenizerFast.from_pretrained(
            self.cfg.text_model_pretrained_model_name_or_path
        )

        self.max_new_tokens = self.shape_model.cfg.num_encoder_latents
        self.min_id = 0
        self.max_id = self.shape_model.cfg.num_codes

    @torch.inference_mode()
    def prepare_inputs(self, prompts: list[str], guidance_scale: float):
        """
        Prepares the input embeddings for the model based on the provided prompts and guidance scale.
        Args:
            prompts (list[str]): A list of prompt strings to be encoded.
            guidance_scale (float): A scaling factor for guidance. If greater than 0.0, additional processing is applied.
        Returns:
            tuple: A tuple containing:
                - embed (torch.Tensor): The encoded input embeddings.
                - cond (torch.Tensor): The condition embeddings, which may include unconditional embeddings if guidance_scale is greater than 0.0.
        """

        prompt_embeds = self.run_clip(prompts)

        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            embed = self.encode_input(prompt_embeds, self.gpt_model.shape_bos_id)

        cond = prompt_embeds
        if guidance_scale > 0.0:
            embed = torch.cat([embed, embed], dim=0)
            uncond_embeds = self.run_clip([""] * len(prompts))
            cond = torch.cat([prompt_embeds, uncond_embeds], dim=0)

        return embed, cond

    @torch.inference_mode()
    def run_clip(self, text_inputs):
        """
        Processes the given text inputs using a text tokenizer and a text model, and returns the encoded text embeddings.
        Args:
            text_inputs (str or List[str]): The input text or list of texts to be processed.
        Returns:
            torch.Tensor: The encoded text embeddings.
        """

        text_inputs = self.text_tokenizer(
            text_inputs,
            max_length=self.text_tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
            # use full precision for text encoder
            with torch.autocast(device_type=self.device.type, enabled=False):
                encoded = self.text_model(**text_inputs)
            if self.gpt_model.cfg.use_pooled_text_embed:
                embed = encoded.text_embeds.unsqueeze(1)  # [bs, 1, 512]
            else:
                embed = encoded.last_hidden_state  # [bs, 77, 512]
        embed = self.gpt_model.encode_text(embed)

        return embed

    @torch.inference_mode()
    def encode_input(self, inputs: torch.Tensor, bos: int):
        """
        Encodes the beginning of sequence (BOS) token for the given input tensor.
        Args:
            inputs (torch.Tensor): The input tensor containing sequences.
            bos (int): The beginning of sequence token ID.
        Returns:
            torch.Tensor: The encoded BOS token embeddings.
        """

        b = inputs.shape[0]
        bos_embed = self.gpt_model.encode_token(
            torch.full(
                (b, 1),
                fill_value=bos,
                dtype=torch.long,
                device=self.device,
            )
        )
        return bos_embed

    @torch.inference_mode()
    def run_gpt(
        self,
        prompts: list[str],
        use_kv_cache: bool,
        guidance_scale: float = 3.0,
        top_p: float = None,
    ):
        """
        Generates text using a GPT model based on the provided prompts.
        Args:
            prompts (list[str]): A list of input prompts to generate text from.
            use_kv_cache (bool): Whether to use key-value caching for faster generation.
            guidance_scale (float, optional): The scale for guidance during generation. Default is 3.0.
            top_p (float, optional): The cumulative probability threshold for nucleus sampling.
            If None, argmax selection is performed (deterministic generation). Otherwise, smallest set of tokens with cumulative probability ≥ top_p are kept (stochastic generation).
        Returns:
            torch.Tensor: A tensor containing the generated token IDs.
        """
        embed, cond = self.prepare_inputs(prompts, guidance_scale)

        output_ids = []

        batch_size, input_seq_len, dim = embed.shape
        max_seq_len = input_seq_len + self.max_new_tokens
        embed_buffer = torch.zeros(
            (batch_size, max_seq_len, dim), dtype=embed.dtype, device=embed.device
        )
        embed_buffer[:, :input_seq_len, :].copy_(embed)
        cond_len = cond.shape[1]
        kv_cache = None
        if use_kv_cache:
            kv_cache = self.gpt_model.init_kv_cache(
                batch_size,
                cond_len,
                self.max_new_tokens + 1,  # +1 for the BOS token
                torch.bfloat16,
                embed.device,
            )
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            for i in tqdm(range(self.max_new_tokens), desc=f"generating"):
                curr_pos_id = torch.tensor([i], dtype=torch.long, device=embed.device)
                logits = self.gpt_model(
                    embed_buffer,
                    cond,
                    kv_cache=kv_cache,
                    curr_pos_id=curr_pos_id if use_kv_cache else None,
                    decode=(i > 0) if use_kv_cache else False,
                )
                if use_kv_cache:
                    logits = logits[:, 0, ...]
                else:
                    logits = logits[:, i, ...]

                logits = logits[..., self.min_id : self.max_id]

                if guidance_scale > 0.0:
                    logits, uncond_logits = logits.float().chunk(2, dim=0)
                    gamma = (
                        guidance_scale * (self.max_new_tokens - i) / self.max_new_tokens
                    )
                    logits = (1 + gamma) * logits - gamma * uncond_logits
                next_id = process_logits(
                    logits,
                    top_p=top_p,
                )
                output_ids.append(next_id)
                next_embed = self.gpt_model.encode_token(next_id)
                if guidance_scale > 0.0:
                    next_embed = torch.cat([next_embed, next_embed], dim=0)
                embed_buffer[:, i + input_seq_len, :].copy_(next_embed.squeeze(1))

        return torch.cat(output_ids, dim=1)

    @torch.inference_mode()
    def run_shape_decode(
        self,
        output_ids: torch.Tensor,
        resolution_base: float = 8.0,
        chunk_size: int = 100_000,
    ):
        """
        Decodes the shape from the given output IDs and extracts the geometry.
        Args:
            output_ids (torch.Tensor): The tensor containing the output IDs.
            resolution_base (float, optional): The base resolution for geometry extraction. Defaults to 8.43.
            chunk_size (int, optional): The chunk size for processing. Defaults to 100,000.
        Returns:
            tuple: A tuple containing the vertices and faces of the mesh.
        """
        shape_ids = (
            output_ids[:, : self.shape_model.cfg.num_encoder_latents, ...]
            .clamp_(0, self.shape_model.cfg.num_codes - 1)
            .view(-1, self.shape_model.cfg.num_encoder_latents)
        )
        latents = self.shape_model.decode_indices(shape_ids)
        mesh_v_f, _ = self.shape_model.extract_geometry(
            latents,
            resolution_base=resolution_base,
            chunk_size=chunk_size,
            use_warp=True,
        )
        return mesh_v_f

    @torch.inference_mode()
    def t2s(
        self,
        prompts: list[str],
        use_kv_cache: bool,
        guidance_scale: float = 3.0,
        resolution_base: float = 8.0,
        chunk_size: int = 100_000,
        top_p: float = None,
    ):
        """
        Generates a 3D mesh from text prompts using a GPT model and shape decoder.
        Args:
            prompts (list[str]): A list of text prompts to guide the generation.
            use_kv_cache (bool): Whether to use key-value caching for the GPT model.
            guidance_scale (float, optional): The scale of guidance for the GPT model. Default is 3.0.
            resolution_base (float, optional): The base resolution for the shape decoder. Default is 8.0.
            chunk_size (int, optional): The chunk size for processing the shape decoding. Default is 100,000.
            top_p (float, optional): The cumulative probability threshold for nucleus sampling. 
                                    If None, argmax selection is performed (deterministic generation). Otherwise, smallest set of tokens with cumulative probability ≥ top_p are kept (stochastic generation).
        Returns:
            mesh_v_f: The generated 3D mesh vertices and faces.
        """
        output_ids = self.run_gpt(prompts, use_kv_cache, guidance_scale, top_p)
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            mesh_v_f = self.run_shape_decode(output_ids, resolution_base, chunk_size)
        return mesh_v_f


class EngineFast(Engine):
    def __init__(
        self,
        config_path: str,
        gpt_ckpt_path: str,
        shape_ckpt_path: str,
        device: torch.device,
    ):
        """
        Initializes the inference engine with the given configuration and checkpoint paths.
        Args:
            config_path (str): Path to the configuration file.
            gpt_ckpt_path (str): Path to the GPT checkpoint file.
            shape_ckpt_path (str): Path to the shape checkpoint file.
            device (torch.device): The device to run the inference on (e.g., CPU or CUDA).
        """

        assert (
            device.type == "cuda"
        ), "EngineFast is only supported on cuda devices, please use Engine on non-cuda devices"

        super().__init__(config_path, gpt_ckpt_path, shape_ckpt_path, device)

        # CUDA Graph params
        self.graph = torch.cuda.CUDAGraph()
        self.embed_buffer = torch.Tensor()
        self.cond_buffer = torch.Tensor()
        self.logits_buffer = torch.Tensor()
        self.curr_pos_id = torch.tensor([0], dtype=torch.long, device=self.device)
        self.kv_cache: list[Cache] = []

        self._warmup_and_capture_graph()

    def _warmup_and_capture_graph(self):
        """
        Warms up the model by running a series of forward passes and captures the CUDA graph for efficient execution.
        This method performs the following steps:
        1. Prepares the input embeddings and conditions using a warmup prompt.
        2. Initializes buffers for embeddings and conditions.
        3. Initializes the key-value cache for the GPT model.
        4. Runs a series of warmup passes to prefill the model and generate logits.
        5. Captures the CUDA graph for the model's forward pass to optimize future executions.
        """

        warmup_prompt = "A cube"
        embed, cond = self.prepare_inputs([warmup_prompt], guidance_scale=3.0)

        batch_size, input_seq_len, dim = embed.shape
        max_seq_len = input_seq_len + self.max_new_tokens
        self.embed_buffer = torch.zeros(
            (batch_size, max_seq_len, dim), dtype=embed.dtype, device=self.device
        )
        self.embed_buffer[:, :input_seq_len, :].copy_(embed)

        self.cond_buffer = torch.empty_like(cond)
        self.cond_buffer.copy_(cond)
        cond_len = self.cond_buffer.shape[1]

        # Initialize kv_cache for the first time
        self.kv_cache = self.gpt_model.init_kv_cache(
            batch_size,
            cond_len,
            self.max_new_tokens + 1,  # +1 for the BOS token
            torch.bfloat16,
            self.device,
        )

        num_warmup_passes = 10

        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            self._set_curr_pos_id(0)
            _ = self._prefill_and_return_logits()

            for x in range(1, num_warmup_passes):
                self._set_curr_pos_id(x)
                self.logits_buffer = self.gpt_model(
                    embed=self.embed_buffer,
                    cond=self.cond_buffer,
                    kv_cache=self.kv_cache,
                    curr_pos_id=self.curr_pos_id,
                    decode=True,
                )

        side_stream = torch.cuda.Stream(device=self.device)
        with torch.cuda.graph(self.graph, stream=side_stream):
            with torch.autocast(self.device.type, dtype=torch.bfloat16):
                self.logits_buffer = self.gpt_model(
                    embed=self.embed_buffer,
                    cond=self.cond_buffer,
                    kv_cache=self.kv_cache,
                    curr_pos_id=self.curr_pos_id,
                    decode=True,
                )

    def _reset_kv_cache(self):
        """
        Resets the key-value cache by setting all key and value states to zero.
        This method iterates through each cache in the `kv_cache` attribute and
        calls the `zero_()` method on both `key_states` and `value_states` to
        reset them to their initial state.
        """

        for cache in self.kv_cache:
            cache.key_states.zero_()
            cache.value_states.zero_()

    def _prefill_and_return_logits(self) -> torch.Tensor:
        """
        Prefills the model's key-value cache and returns the logits.
        This method resets the key-value cache and then performs a forward pass
        through the GPT model in eager mode to prefill the logits.
        Returns:
            torch.Tensor: The prefilled logits tensor with the first dimension removed.
        """

        self._reset_kv_cache()

        # Prefill is always eager
        prefill_logits = self.gpt_model(
            embed=self.embed_buffer,
            cond=self.cond_buffer,
            kv_cache=self.kv_cache,
            curr_pos_id=self.curr_pos_id,
            decode=False,
        )

        return prefill_logits[:, 0, ...]

    def _set_curr_pos_id(self, pos: int):
        """
        Set the current position ID.
        This method updates the `curr_pos_id` attribute with the given position.
        Args:
            pos (int): The position ID to set.
        """

        self.curr_pos_id.copy_(
            torch.tensor([pos], dtype=torch.long, device=self.device)
        )

    def run_gpt(
        self, 
        prompts: list[str], 
        use_kv_cache: bool, 
        guidance_scale: float = 3.0,
        top_p: float = None
    ):
        """
        Runs the GPT model to generate text based on the provided prompts.
        Args:
            prompts (list[str]): A list of input prompts for the GPT model. Only a single prompt is supported.
            use_kv_cache (bool): Flag indicating whether to use key-value caching. (Currently not used)
            guidance_scale (float, optional): The scale factor for guidance. Default is 3.0.
            top_p (float, optional): The cumulative probability threshold for nucleus sampling.
            If None, argmax selection is performed. Otherwise, smallest set of tokens with cumulative probability ≥ top_p are kept.
        Returns:
            torch.Tensor: A tensor containing the generated output token IDs.
        Raises:
            AssertionError: If the batch size is greater than 1.
        """

        embed, cond = self.prepare_inputs(prompts, guidance_scale)
        assert len(prompts) == 1, "batch size > 1 not support for EngineFast"

        batch_size, input_seq_len, _ = embed.shape
        self.embed_buffer.zero_()
        self.embed_buffer[:, :input_seq_len, :].copy_(embed)

        assert self.cond_buffer.shape == cond.shape
        self.cond_buffer.copy_(cond)

        output_ids = torch.zeros(
            (batch_size // 2, self.max_new_tokens), dtype=torch.int, device=self.device
        )

        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            self._set_curr_pos_id(0)

            logits = self._prefill_and_return_logits()

            logits = logits[..., self.min_id : self.max_id]
            if guidance_scale > 0.0:
                logits, uncond_logits = logits.float().chunk(2, dim=0)
                gamma = guidance_scale
                logits = (1 + gamma) * logits - gamma * uncond_logits
            next_id = process_logits(logits, top_p=top_p)

            output_ids[:, 0] = next_id.squeeze()
            next_embed = self.gpt_model.encode_token(next_id)
            next_embed = next_embed.repeat(2, 1, 1)
            self.embed_buffer[:, input_seq_len, :].copy_(next_embed.squeeze(1))

            for i in tqdm(
                range(1, self.max_new_tokens), desc=f"generating"
            ):
                self._set_curr_pos_id(i)
                self.graph.replay()

                logits = self.logits_buffer[:, 0, ...]

                logits = logits[..., self.min_id : self.max_id]
                if guidance_scale > 0.0:
                    logits, uncond_logits = logits.float().chunk(2, dim=0)
                    gamma = (
                        guidance_scale * (self.max_new_tokens - i) / self.max_new_tokens
                    )
                    logits = (1 + gamma) * logits - gamma * uncond_logits
                next_id = process_logits(logits, top_p=top_p)

                output_ids[:, i] = next_id.squeeze()
                next_embed = self.gpt_model.encode_token(next_id)
                next_embed = next_embed.repeat(2, 1, 1)
                self.embed_buffer[:, i + input_seq_len, :].copy_(next_embed.squeeze(1))

        return output_ids
