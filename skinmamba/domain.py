"""Source-only paired appearance augmentation and latent consistency.

No target loader or fitted target statistics. CORAL-style moments, paired
consistency and a VICReg-inspired variance floor are explicit separate terms;
this implementation is an experimental combination, not an author reproduction.
"""
from __future__ import annotations

import hashlib
import torch
from torch.nn import functional as F


def style_view(image, normalization, options, generator=None):
    """Geometry-preserving RGB transforms after dataset normalization.

    Input/output convention is kept (official_minmax255 is [0,255], unit [0,1]).
    No independent renormalization after the transform, which would cancel some
    appearance perturbations. Bounds are deliberately mild and configurable.
    """
    if normalization not in {"unit", "official_minmax255"}:
        raise ValueError("Unknown input normalization")
    scale = 255. if normalization == "official_minmax255" else 1.
    x = image.float() / scale

    def draw(key, channels=1, center=1.):
        shape = (len(x), channels, 1, 1)
        return center + (2 * torch.rand(shape, device=x.device, generator=generator) - 1) * options[key]

    contrast, brightness, color, gamma = draw("contrast"), draw("brightness", center=0.), draw("color", 3), draw("gamma")
    center = x.mean((2, 3), keepdim=True)
    x = ((x - center) * contrast + center) * color + brightness
    return x.clamp(0, 1).pow(gamma) * scale


def fixed_style_view(image, ids, normalization, options, seed):
    """Deterministic per-image pseudo-domain, independent of loader batch size."""
    output = []
    for index, key in enumerate(ids):
        digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
        generator = torch.Generator(device=image.device).manual_seed(int.from_bytes(digest[:8], "little") % (2**63 - 1))
        output.append(style_view(image[index:index + 1], normalization, options, generator))
    return torch.cat(output)


def state_consistency(first, second, options):
    if len(first) != len(second) or not first:
        raise ValueError("State stages must match and be nonempty")
    terms = {k: [] for k in ("pair", "mean", "covariance", "variance")}
    for a, b in zip(first, second):
        a, b = a.float(), b.float()
        if a.ndim != 2 or a.shape != b.shape:
            raise ValueError("Paired states must have equal [images, channels] shape")
        terms["pair"].append(F.mse_loss(a, b))
        terms["mean"].append(F.mse_loss(a.mean(0), b.mean(0)))
        ac, bc = a - a.mean(0), b - b.mean(0)
        if len(a) > 1:
            ca, cb = ac.T @ ac / (len(a) - 1), bc.T @ bc / (len(a) - 1)
            # Frobenius squared / (4*d*d), the Deep CORAL convention.
            terms["covariance"].append((ca - cb).square().mean() / 4)
            sa, sb = torch.sqrt(ca.diag() + 1e-4), torch.sqrt(cb.diag() + 1e-4)
            terms["variance"].append((F.relu(options["std_floor"] - sa).mean()
                                       + F.relu(options["std_floor"] - sb).mean()) / 2)
        else:
            # A singleton tail batch has no sample covariance; do not invent one.
            zero = (a.sum() + b.sum()) * 0
            terms["covariance"].append(zero)
            terms["variance"].append(zero)
    reduced = {k: torch.stack(v).mean() for k, v in terms.items()}
    total = sum(options[f"{k}_weight"] * v for k, v in reduced.items())
    return total, reduced


def domain_weight(options, epoch):
    warmup = options.get("warmup_epochs", 0)
    return options["weight"] * min(1., (epoch + 1) / warmup) if warmup else options["weight"]


def training_objective(model, image, target, criterion, config, epoch, style_generator=None):
    """Called inside autocast; all loss/statistic arithmetic is FP32."""
    options = config.get("domain", {})
    if not options.get("two_view", False):
        loss = criterion(model(image), target)
        return loss, {"seg": loss.detach()}
    use_state_loss = options["weight"] > 0
    styled = style_view(image, config["data"]["normalization"], options, style_generator)
    if use_state_loss:
        a, aux_a = model(image, return_states=True)
        b, aux_b = model(styled, return_states=True)
    else:
        a, b = model(image), model(styled)
    seg = (criterion(a, target) + criterion(b, target)) / 2
    if not use_state_loss:
        return seg, {"seg": seg.detach()}
    location = options["location"]
    with torch.autocast(device_type=image.device.type, enabled=False):
        dom, terms = state_consistency(aux_a[location], aux_b[location], options)
    weight = domain_weight(options, epoch)
    return seg + weight * dom, {"seg": seg.detach(), "dom": dom.detach(),
                               **{k: v.detach() for k, v in terms.items()}}
