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


def _project_steps(steps, total, minimum, maximum):
    """Shrink each row toward uniform spacing just enough to satisfy bounds."""
    mean = total / steps.shape[-1]
    if minimum < 0 or not minimum < mean < maximum:
        raise ValueError("coverage spacing bounds must straddle the mean spacing")
    count = steps.shape[-1]
    if minimum * count > total + 1e-6 or maximum * count < total - 1e-6:
        raise ValueError("coverage spacing bounds cannot cover the requested width")
    deviation = steps - mean
    upper = (maximum - mean) / deviation.clamp_min(1e-8)
    lower = (mean - minimum) / (-deviation).clamp_min(1e-8)
    strength = torch.minimum(upper, lower).amin(dim=-1, keepdim=True).clamp(max=1.0)
    return mean + strength * deviation


def make_adaptive_sampling_plan(
    scores,
    sample_ratio,
    *,
    adaptive_lambda=1.0,
    coverage=False,
    min_spacing=0.5,
    max_spacing=2.0,
    delta_bounds=None,
    eps=1e-4,
):
    """Build a differentiable, fixed-budget horizontal sampling plan.

    ``scores`` is a positive Bx1xHxW importance map.  Larger scores create
    denser samples by assigning smaller inverse-density intervals.  Endpoints
    are always retained, so the sampler cannot discard a lesion touching an
    image border.  All tensors after ``scores`` remain on its device.
    """
    if scores.ndim != 4 or scores.shape[1] != 1:
        raise ValueError("scores must have shape [batch, 1, height, width]")
    batch, _, height, width = scores.shape
    if height < 1 or width < 1:
        raise ValueError("Feature map dimensions must be positive")
    if not (0 < sample_ratio <= 1) or not math.isfinite(sample_ratio):
        raise ValueError("sample_ratio must be finite and in (0, 1].")
    if not 0 <= adaptive_lambda <= 1 or not math.isfinite(adaptive_lambda):
        raise ValueError("adaptive_lambda must be finite and in [0, 1].")
    if width == 1:
        x = scores.new_zeros(batch, height, 1, dtype=torch.float32)
        yn = torch.linspace(-1, 1, height, device=scores.device).view(1, height, 1).expand(batch, -1, -1)
        return {
            "x": x, "grid": torch.stack((x, yn), dim=-1),
            "left": torch.zeros(batch, height, 1, device=scores.device, dtype=torch.long),
            "right": torch.zeros(batch, height, 1, device=scores.device, dtype=torch.long),
            "weight": x, "geometry": torch.ones(batch, height, device=scores.device),
            "height": height, "width": width, "sample_count": 1,
            "row_transition_count": max(0, height - 1), "adaptive": True,
        }
    count = min(width, max(2, round(width * sample_ratio)))
    score = scores.float().squeeze(1).clamp_min(eps)
    # A uniform component prevents an initially uncertain score head from
    # collapsing all samples into one narrow lesion region.
    score = score / score.mean(dim=-1, keepdim=True).clamp_min(eps)
    density = (1.0 - adaptive_lambda) + adaptive_lambda * score
    # The W-1 feature-grid edges define a discrete density. Inverting its CDF
    # at K evenly spaced quantiles yields monotone, subpixel sample locations.
    edge_density = (density[..., :-1] + density[..., 1:]) * 0.5
    edge_prob = edge_density / edge_density.sum(dim=-1, keepdim=True).clamp_min(eps)
    cdf = torch.cat((torch.zeros_like(edge_prob[..., :1]), edge_prob.cumsum(dim=-1)), dim=-1)
    quantiles = torch.linspace(0, 1, count, device=scores.device).view(1, 1, count).expand(batch, height, -1)
    right_cdf = torch.searchsorted(cdf.contiguous(), quantiles.contiguous(), right=False).clamp(1, width - 1)
    left_cdf = right_cdf - 1
    low = cdf.gather(-1, left_cdf)
    high = cdf.gather(-1, right_cdf)
    x_raw = left_cdf + (quantiles - low) / (high - low).clamp_min(eps)
    steps = x_raw[..., 1:] - x_raw[..., :-1]
    if coverage:
        steps = _project_steps(steps, float(width - 1), min_spacing, max_spacing)
    x_inner = steps.cumsum(dim=-1)[..., :-1]
    endpoint = edge_prob.new_zeros(batch, height, 1)
    x = torch.cat((endpoint, x_inner,
                   torch.full_like(endpoint, float(width - 1))), dim=-1)
    # Uniform row coordinates and per-sample horizontal coordinates.
    yn = torch.linspace(-1, 1, height, device=scores.device, dtype=x.dtype)
    xn = 2 * x / (width - 1) - 1
    yy = yn.view(1, height, 1).expand(batch, height, count)
    grid = torch.stack([xn, yy], dim=-1)

    targets = torch.arange(width, device=scores.device, dtype=x.dtype).view(1, 1, width)
    left = (targets.unsqueeze(-1) >= x.unsqueeze(-2)).sum(dim=-1).sub(1).clamp(0, count - 2)
    right = left + 1
    left_x = x.gather(-1, left)
    right_x = x.gather(-1, right)
    weight = (targets - left_x) / (right_x - left_x).clamp_min(eps)
    row_steps = torch.cat((torch.ones(batch, height, 1, device=scores.device, dtype=x.dtype), steps), dim=-1)
    mean_spacing = (width - 1) / (count - 1)
    geometry = row_steps / mean_spacing
    if delta_bounds is not None:
        delta_min, delta_max = delta_bounds
        if not 0 < delta_min < delta_max or not all(math.isfinite(v) for v in delta_bounds):
            raise ValueError("delta_bounds must contain finite 0 < min < max")
        geometry = geometry.clamp(delta_min, delta_max)
    # The first token in every row starts a new spatial scan.
    geometry = torch.cat((torch.ones_like(geometry[..., :1]), geometry[..., 1:]), dim=-1)
    return {
        "x": x, "grid": grid, "left": left, "right": right, "weight": weight,
        "geometry": geometry.flatten(1), "height": height, "width": width,
        "sample_count": count, "row_transition_count": max(0, height - 1),
        "adaptive": True,
    }


def sample_adaptive_features(features, plan):
    grid = plan["grid"].to(dtype=features.dtype)
    return F.grid_sample(features, grid, mode="bilinear", padding_mode="border", align_corners=True)


def restore_adaptive_features(sampled, plan):
    weight = plan["weight"].to(dtype=sampled.dtype).unsqueeze(1)
    left = sampled.gather(-1, plan["left"].unsqueeze(1).expand(-1, sampled.shape[1], -1, -1))
    right = sampled.gather(-1, plan["right"].unsqueeze(1).expand(-1, sampled.shape[1], -1, -1))
    return left + weight * (right - left)
