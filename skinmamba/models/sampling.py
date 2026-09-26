"""Fixed, monotone row-wise sampling for controlled delta ablations."""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def make_sampling_plan(height, width, sample_ratio, sampling_power, *, device=None):
    if not (0 < sample_ratio <= 1) or not math.isfinite(sample_ratio):
        raise ValueError("sample_ratio must be finite and in (0, 1].")
    if sampling_power <= 0 or not math.isfinite(sampling_power):
        raise ValueError("sampling_power must be finite and positive.")
    if height < 1 or width < 1:
        raise ValueError("Feature map dimensions must be positive.")
    count = min(width, max(2, round(width * sample_ratio)))
    x = torch.linspace(0, 1, count, device=device, dtype=torch.float32).pow(sampling_power) * (width - 1)
    # One unit is one pixel of the original stage grid, regardless of density.
    local_steps = torch.ones_like(x)
    if count > 1:
        local_steps[1:] = x[1:] - x[:-1]
    geometry = local_steps.repeat(height)
    constant = torch.full_like(geometry, (width - 1) / (count - 1) if count > 1 else 1.0)
    row_starts = torch.arange(height, device=device) * count
    geometry[row_starts] = 1.0
    constant[row_starts] = 1.0
    yn = torch.linspace(-1, 1, height, device=device) if height > 1 else torch.zeros(1, device=device)
    xn = 2 * x / (width - 1) - 1 if width > 1 else torch.zeros_like(x)
    yy, xx = torch.meshgrid(yn, xn, indexing="ij")
    grid = torch.stack([xx, yy], dim=-1).unsqueeze(0)
    # Locations for linear interpolation on the *nonuniform* sample coordinates.
    targets = torch.arange(width, device=device, dtype=torch.float32)
    if count > 1:
        right = torch.searchsorted(x.contiguous(), targets).clamp(1, count - 1)
        left = right - 1
        weight = (targets - x[left]) / (x[right] - x[left])
    else:
        left = right = torch.zeros(width, device=device, dtype=torch.long)
        weight = torch.zeros_like(targets)
    return {
        "x": x, "grid": grid, "geometry": geometry, "constant": constant,
        "index": torch.ones_like(geometry), "left": left, "right": right,
        "weight": weight, "row_starts": row_starts,
        "height": height, "width": width, "sample_count": count,
        "row_transition_count": max(0, height - 1),
    }


def sample_features(features, plan):
    grid = plan["grid"].to(dtype=features.dtype).expand(features.shape[0], -1, -1, -1)
    return F.grid_sample(features, grid, mode="bilinear", padding_mode="border", align_corners=True)


def restore_features(sampled, plan):
    """Interpolate horizontal coordinates back, preserving the original row order."""
    weight = plan["weight"].to(dtype=sampled.dtype)[None, None, None, :]
    left = sampled.index_select(-1, plan["left"])
    right = sampled.index_select(-1, plan["right"])
    return left + weight * (right - left)
