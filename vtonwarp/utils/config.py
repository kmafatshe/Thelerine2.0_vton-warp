"""Minimal YAML config with attribute access and CLI overrides."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import yaml


class Config(dict):
    """A dict whose keys are also attributes, applied recursively."""

    def __init__(self, mapping: dict | None = None):
        super().__init__()
        for key, value in (mapping or {}).items():
            self[key] = Config(value) if isinstance(value, dict) else value

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name, value):
        self[name] = Config(value) if isinstance(value, dict) else value

    def merge(self, overrides: list[str]) -> "Config":
        """Apply `a.b=value` CLI overrides, parsing values as YAML scalars."""
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"override must be key=value, got {item!r}")
            path, raw = item.split("=", 1)
            node = self
            *parents, leaf = path.split(".")
            for part in parents:
                node = node.setdefault(part, Config())
            node[leaf] = yaml.safe_load(raw)
        return self


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    config = Config(yaml.safe_load(Path(path).read_text()))
    return config.merge(overrides or [])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
