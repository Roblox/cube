from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_model


def load_config(cfg_path: str) -> Any:
    """
    Load and resolve a configuration file.
    Args:
        cfg_path (str): The path to the configuration file.
    Returns:
        Any: The loaded and resolved configuration object.
    Raises:
        AssertionError: If the loaded configuration is not an instance of DictConfig.
    """

    cfg = OmegaConf.load(cfg_path)
    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)
    return cfg


def parse_structured(cfg_type: Any, cfg: DictConfig) -> Any:
    """
    Parses a configuration dictionary into a structured configuration object.
    Args:
        cfg_type (Any): The type of the structured configuration object.
        cfg (DictConfig): The configuration dictionary to be parsed.
    Returns:
        Any: The structured configuration object created from the dictionary.
    """

    scfg = OmegaConf.structured(cfg_type(**cfg))
    return scfg


def load_model_weights(model: torch.nn.Module, ckpt_path: str) -> None:
    """
    Load a safetensors checkpoint into a PyTorch model.
    The model is updated in place.

    Args:
        model: PyTorch model to load weights into
        ckpt_path: Path to the safetensors checkpoint file

    Returns:
        None
    """
    assert ckpt_path.endswith(".safetensors"), (
        f"Checkpoint path '{ckpt_path}' is not a safetensors file"
    )

    load_model(model, ckpt_path)

def select_device() -> Any:
    """
    Selects the appropriate PyTorch device for tensor allocation.
    
    Returns:
        Any: The `torch.device` object.
    """
    return torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )