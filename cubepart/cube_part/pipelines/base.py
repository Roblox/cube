from dataclasses import dataclass
from typing import List, Optional, Union

import torch


@dataclass
class ShapeInput:
    prompt: Optional[Union[str, List[str], List[List[str]]]] = None
    latents: Optional[torch.Tensor] = (
        None  # pre-encoded shape latents (e.g. from ShapeDenoiserPipeline.encode_shape)
    )
