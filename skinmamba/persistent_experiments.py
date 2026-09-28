"""Presets for the source-only persistent-memory experiment (PSM v1)."""
from __future__ import annotations

import copy

DOMAIN_DEFAULTS = dict(
    two_view=True, weight=.05, warmup_epochs=5, location="states",
    pair_weight=1., mean_weight=1., covariance_weight=1., variance_weight=.1,
    std_floor=.05, brightness=.10, contrast=.20, color=.15, gamma=.20,
)

PSM_PRESETS = {
    "psm_baseline": dict(baseline=True, two_view=False, weight=0.),
    "psm_baseline_aug": dict(baseline=True, weight=0.),
    "psm_memory": dict(two_view=False, weight=0.),
    "psm_memory_aug": dict(weight=0.),
    "psm_independent_aug": dict(memory_mode="independent", weight=0.),
    "psm_independent_dom": dict(memory_mode="independent"),
    "psm_main": {},
    "psm_feature_dom": dict(location="queries"),
    "psm_moments_only": dict(pair_weight=0.),
    "psm_no_variance": dict(variance_weight=0.),
}


def persistent_experiment(config, name):
    result = copy.deepcopy(config)
    preset = dict(PSM_PRESETS[name])
    baseline = preset.pop("baseline", False)
    mode = preset.pop("memory_mode", "persistent")
    # Reuse the unchanged author backbone; no historical reconstruction settings.
    backbone_keys = ("backend", "channels", "groups", "bridge", "split_att", "d_state", "d_conv",
                     "expand", "num_classes", "input_channels")
    model = {k: v for k, v in result["model"].items() if k in backbone_keys}
    model.update(family="ultralight" if baseline else "persistent",
                 variant="baseline" if baseline else name)
    if not baseline:
        model.update(memory_mode=mode, memory_dim=result["model"].get("memory_dim", 16),
                     memory_strength=result["model"].get("memory_strength", .1))
    result.update(name=name, model=model, experiment_suite="persistent_v1")
    # Presets reset all switches; explicit CLI --set applies afterwards.
    result["domain"] = {**DOMAIN_DEFAULTS, **preset}
    return result
