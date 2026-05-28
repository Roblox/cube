from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def build_condition_processor(model_type: str, **kwargs):
    if model_type.startswith("Qwen/Qwen3-VL"):
        return Qwen3VLProcessor(model_type, **kwargs)
    raise NotImplementedError(f"Unsupported condition model: {model_type!r}")


class BaseConditionProcessor(nn.Module):
    @property
    def device(self):
        return self.text_encoder.device


class Qwen3VLProcessor(BaseConditionProcessor):
    def __init__(
        self,
        model_type: str,
        tokenizer_max_length: int,
        padding_mode: Optional[str] = None,
        model_path: Optional[str] = None,
        attn_implementation: str = "sdpa",
    ):
        super().__init__()

        if model_path is not None:
            self.processor = AutoProcessor.from_pretrained(
                model_path, local_files_only=True
            )
            text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                dtype=torch.float16,
                attn_implementation=attn_implementation,
                local_files_only=True,
            )
        else:
            self.processor = AutoProcessor.from_pretrained(model_type)
            text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
                model_type, dtype=torch.float16, attn_implementation=attn_implementation
            )
        self.text_encoder = text_encoder.requires_grad_(False).eval()

        # follow qwen image
        self.prompt_template_encode = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        self.prompt_template_encode_start_idx = 34

        self.tokenizer_max_length = tokenizer_max_length
        self.padding_mode = padding_mode

    @property
    def hidden_size(self):
        return self.text_encoder.config.text_config.hidden_size

    def _post_process_encoder_hidden_states(
        self,
        encoder_hidden_states,
        encoder_attention_mask,
        template_encode_start_idx: int,
    ):
        encoder_hidden_states = encoder_hidden_states[:, template_encode_start_idx:]
        encoder_attention_mask = encoder_attention_mask[:, template_encode_start_idx:]
        if self.padding_mode == "zeros":
            # zero out padding tokens
            encoder_hidden_states = (
                encoder_hidden_states
                * encoder_attention_mask.unsqueeze(-1).to(encoder_hidden_states.device)
            )
        return encoder_hidden_states.contiguous(), encoder_attention_mask.contiguous()

    def _process_text(self, text):
        inputs = self.processor(
            text=text,
            max_length=self.tokenizer_max_length
            + self.prompt_template_encode_start_idx,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with torch.autocast(self.text_encoder.device.type, dtype=torch.float16):
            self.text_encoder.base_model.rope_deltas = None
            encoder_hidden_states = self.text_encoder(
                input_ids=inputs["input_ids"].to(self.text_encoder.device),
                attention_mask=inputs["attention_mask"].to(self.text_encoder.device),
                output_hidden_states=True,
            )
        return self._post_process_encoder_hidden_states(
            encoder_hidden_states.hidden_states[-1],
            inputs["attention_mask"],
            template_encode_start_idx=self.prompt_template_encode_start_idx,
        )

    @torch.no_grad()
    def __call__(self, text):
        encoder_hidden_states, encoder_attention_mask = self._process_text(text)
        return encoder_hidden_states.detach(), encoder_attention_mask.detach()
