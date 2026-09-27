from __future__ import annotations

import copy
from pathlib import Path
import yaml
from .experiments import RECONSTRUCTION_DEFAULTS, RECONSTRUCTION_PRESETS, reconstruction_options

ABLATIONS = (
    "unfused_control", "sampling_only", "constant_scale", "geometry",
    "uniform_identity", "no_bridge", "wider", "bce_only", "dice_only",
    "uniform_sampling", "adaptive_sampling", "adaptive_geometry",
    "adaptive_coverage", "cclas",
) + tuple(RECONSTRUCTION_PRESETS)


def merge(base, update):
    out = copy.deepcopy(base)
    for key, value in update.items():
        out[key] = merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else copy.deepcopy(value)
    return out


def apply_overrides(config, overrides):
    config = copy.deepcopy(config)
    for override in overrides:
        key, sep, value = override.partition("=")
        if not sep:
            raise ValueError(f"Override must be key=value: {override}")
        node = config
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)
    return config


def load_config(path, overrides=()):
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    parent = config.pop("extends", None)
    if parent:
        config = merge(load_config(path.parent / parent), config)
    config = apply_overrides(config, overrides)
    validate_config(config)
    return config


def select_experiment(config, experiment=None, ablation=None):
    """Overlay experiment controls without replacing the source/training config.

    The explicit flag selects a model variant; other config settings remain
    unless named by the selected ablation (e.g. bridge or loss weights).
    """
    if experiment is None and ablation is None:
        return copy.deepcopy(config)
    if ablation is not None and ablation not in ABLATIONS:
        raise ValueError(f"Unknown ablation: {ablation}")
    name = ablation or ("reconstruction" if experiment == "main" else "baseline")
    if name in RECONSTRUCTION_PRESETS:
        return merge(config, {"name": name, "model": reconstruction_options(name)})
    config = copy.deepcopy(config)
    # Do not let irrelevant new-family settings split baseline aggregation into
    # separate groups depending on which YAML the user selected it from.
    for key in RECONSTRUCTION_DEFAULTS:
        config["model"].pop(key, None)
    variant = {"geometry": "sampled_geometry", "sampling_only": "sampled_index",
               "constant_scale": "sampled_constant", "uniform_identity": "sampled_geometry",
               "unfused_control": "unfused_control", "uniform_sampling": "sampled_index",
               "adaptive_sampling": "adaptive_index", "adaptive_geometry": "adaptive_geometry",
               "adaptive_coverage": "adaptive_coverage", "cclas": "cclas"}.get(name, "baseline")
    patch = {"name": experiment or f"ablation_{name}", "model": {"family": "ultralight", "variant": variant}}
    if name == "uniform_identity":
        patch["model"].update(sample_ratio=1.0, sampling_power=1.0)
    elif name == "uniform_sampling":
        patch["model"]["sampling_power"] = 1.0
    elif name == "no_bridge":
        patch["model"]["bridge"] = False
    elif name == "wider":
        patch["model"]["channels"] = [12, 24, 36, 48, 72, 96]
    elif name == "bce_only":
        patch["loss"] = {"bce_weight": 1.0, "dice_weight": 0.0}
    elif name == "dice_only":
        patch["loss"] = {"bce_weight": 0.0, "dice_weight": 1.0}
    return merge(config, patch)


def validate_config(config):
    if not isinstance(config["seed"], int) or isinstance(config["seed"], bool) or not 0 <= config["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
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
        if not isinstance(config["training"][key], int) or isinstance(config["training"][key], bool) or config["training"][key] < 1:
            raise ValueError(f"training.{key} must be positive")
    if config["training"]["workers"] < 0 or config["training"]["lr"] <= 0:
        raise ValueError("workers must be nonnegative and lr must be positive")
