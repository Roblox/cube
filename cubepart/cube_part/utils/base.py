"""Minimal base classes used by the inference modules."""

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn

from omegaconf import DictConfig

from cube_part.utils.config import parse_structured
from cube_part.utils.runtime import get_device


class BaseModule(nn.Module):
    """Base class for configurable modules.

    Inference-oriented base with no dependency on PyTorch Lightning,
    distributed loaders, or any training utilities.
    """

    @dataclass
    class Config:
        pass

    cfg: Config

    def __init__(
        self,
        cfg: Optional[Union[dict, DictConfig]] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.cfg = parse_structured(self.Config, cfg)
        self.device = get_device()
        self.configure(*args, **kwargs)
        self.register_buffer("_dummy", torch.zeros(0).float(), persistent=False)

    def configure(self, *args, **kwargs) -> None:
        pass


class BaseSystem(nn.Module):
    """Inference-only stand-in for ``pytorch_lightning.LightningModule``.

    This class takes a config dict and instantiates the model components in
    ``configure()``. It does not implement training/validation/optimizer
    hooks.
    """

    @dataclass
    class Config:
        pretrained_model_path: Optional[str] = None

    cfg: Config

    def __init__(self, cfg, resumed: bool = False) -> None:
        super().__init__()
        self.cfg = parse_structured(self.Config, cfg)
        self.configure()

    def configure(self) -> None:
        pass
