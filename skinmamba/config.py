from __future__ import annotations

import copy
from pathlib import Path
import yaml


def merge(base, update):
    out = copy.deepcopy(base)
    for key, value in update.items():
        out[key] = merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else copy.deepcopy(value)
    return out


def load_config(path, overrides=()):
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    parent = config.pop("extends", None)
    if parent:
        config = merge(load_config(path.parent / parent), config)
    for override in overrides:
        key, sep, value = override.partition("=")
        if not sep:
            raise ValueError(f"Override must be key=value: {override}")
        node = config
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)
    validate_config(config)
    return config


def validate_config(config):
    if config["data"]["source"] not in ("isic2017", "isic2018"):
        raise ValueError("Only ISIC2017/2018 may be training sources. PH2 is test-only.")
    size = config["data"]["image_size"]
    if not isinstance(size, int) or size < 32 or size % 32:
        raise ValueError("image_size must be an integer multiple of 32")
    if config["training"]["precision"] not in ("fp32", "amp_fp16"):
        raise ValueError("precision must be fp32 or amp_fp16")
    if config["training"]["selection"] not in ("val_loss", "val_dice"):
        raise ValueError("selection must use source validation only")
    if not 0 < config["evaluation"]["threshold"] < 1:
        raise ValueError("threshold must be between zero and one")
    for key in ("epochs", "batch_size", "t_max"):
        if config["training"][key] < 1:
            raise ValueError(f"training.{key} must be positive")
    if config["training"]["workers"] < 0 or config["training"]["lr"] <= 0:
        raise ValueError("workers must be nonnegative and lr must be positive")
