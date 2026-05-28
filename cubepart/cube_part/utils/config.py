"""Configuration helpers backed by OmegaConf, plus a dict→namespace bridge.

Two related concerns live here:

* :func:`load_config` / :func:`parse_structured` are the entry points used
  by the pipeline and the ``BaseModule``/``BaseSystem`` constructors to
  load YAML configs and turn them into structured ``DictConfig`` objects.
* :func:`dict_to_namespace` converts a plain dict (typically the output
  of ``OmegaConf.to_container``) into a ``SimpleNamespace`` so that the
  model classes can keep using attribute access.
"""

from types import SimpleNamespace
from typing import Any, Optional, Union

from omegaconf import DictConfig, OmegaConf


def load_config(
    *yamls: str,
    cli_args: Optional[list] = None,
    from_string: bool = False,
    **kwargs,
) -> DictConfig:
    """Load one or more YAML configs (or strings) and merge with CLI overrides."""
    cli_args = cli_args or []
    if from_string:
        confs = [OmegaConf.create(s) for s in yamls]
    else:
        confs = [OmegaConf.load(f) for f in yamls]
    cli_conf = OmegaConf.from_cli(cli_args)
    cfg = OmegaConf.merge(*confs, cli_conf, kwargs)
    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)
    return cfg


def parse_structured(fields: Any, cfg: Optional[Union[dict, DictConfig]] = None) -> Any:
    cfg = cfg or {}
    return OmegaConf.structured(fields(**cfg))


def dict_to_namespace(d: Any) -> Any:
    """Recursively convert a dict (or list of dicts) to ``SimpleNamespace``."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    if isinstance(d, list):
        return [dict_to_namespace(item) for item in d]
    return d
