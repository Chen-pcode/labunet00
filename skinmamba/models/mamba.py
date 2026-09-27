"""Mamba-1 reference equations and optional official CUDA execution.

The Python recurrence is for small correctness tests, not performance claims.
Parameter names follow state-spaces/mamba's mamba_simple.Mamba.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def selective_scan_reference(u, delta, A, B, C, D=None, z=None):
    """Mamba-1 recurrence using its approximate input discretization delta * B.

    u/delta: [batch, inner, length]; B/C: [batch, state, length].
    This intentionally does not claim exact zero-order-hold discretization.
    """
    original_dtype = u.dtype
    u, delta, A, B, C = (v.float() for v in (u, delta, A, B, C))
    state = u.new_zeros(u.shape[0], u.shape[1], A.shape[1])
    outputs = []
    for t in range(u.shape[-1]):
        dt = delta[:, :, t]
        state = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0)) * state
        state = state + dt.unsqueeze(-1) * B[:, None, :, t] * u[:, :, t, None]
        outputs.append((state * C[:, None, :, t]).sum(dim=-1))
    y = torch.stack(outputs, dim=-1)
    if D is not None:
        y = y + D.float()[None, :, None] * u
    if z is not None:
        y = y * F.silu(z.float())
    return y.to(original_dtype)


def mamba_forward(module, hidden_states, delta_factors=None, backend="reference"):
    """Full-sequence Mamba-1 forward with shared or per-image delta factors."""
    if hidden_states.ndim != 3 or hidden_states.shape[-1] != module.d_model:
        raise ValueError("Mamba input must have shape [batch, length, d_model].")
    if backend == "cuda" and not hidden_states.is_cuda:
        raise RuntimeError("backend='cuda' requires CUDA input; no CPU fallback is allowed.")
    length = hidden_states.shape[1]
    xz = module.in_proj(hidden_states).transpose(1, 2)
    x, z = xz.chunk(2, dim=1)
    x = F.silu(module.conv1d(x)[..., :length])
    projected = module.x_proj(x.transpose(1, 2))
    dt, B, C = torch.split(projected, [module.dt_rank, module.d_state, module.d_state], dim=-1)
    # Add the learned bias exactly once, before softplus and geometric scaling.
    raw_delta = F.linear(dt, module.dt_proj.weight, bias=None).transpose(1, 2)
    delta = F.softplus(raw_delta.float() + module.dt_proj.bias.float()[None, :, None])
    if delta_factors is not None:
        if delta_factors.ndim == 1:
            if delta_factors.numel() != length:
                raise ValueError("delta_factors must contain one factor per sequence token.")
            factors = delta_factors[None, None, :]
        elif delta_factors.ndim == 2:
            if delta_factors.shape != (hidden_states.shape[0], length):
                raise ValueError("Batched delta_factors must have shape [batch, length].")
            factors = delta_factors[:, None, :]
        else:
            raise ValueError("delta_factors must have shape [length] or [batch, length].")
        delta = delta * factors.to(device=delta.device, dtype=delta.dtype)
    B, C = B.transpose(1, 2).contiguous(), C.transpose(1, 2).contiguous()
    A = -torch.exp(module.A_log.float())
    if backend == "cuda":
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

        y = selective_scan_fn(
            x, delta.to(dtype=x.dtype), A, B, C, module.D.float(), z=z,
            delta_bias=None, delta_softplus=False,
        )
    elif backend == "reference":
        y = selective_scan_reference(x, delta, A, B, C, module.D, z)
    else:
        raise ValueError(f"Unknown Mamba backend: {backend}")
    return module.out_proj(y.transpose(1, 2))


class ReferenceMamba(nn.Module):
    """Pure PyTorch Mamba-1 with official parameter names and initialization."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, d_conv, groups=self.d_inner,
            padding=d_conv - 1, bias=True,
        )
        self.activation = "silu"
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(min=1e-4)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.dt_proj.bias._no_reinit = True
        self.A_log = nn.Parameter(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1).log())
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, hidden_states):
        return mamba_forward(self, hidden_states, backend="reference")


def create_mamba(d_model, d_state, d_conv, expand, backend):
    if backend == "reference":
        return ReferenceMamba(d_model, d_state, d_conv, expand)
    if backend != "cuda":
        raise ValueError("backend must be 'cuda' or 'reference'.")
    if not torch.cuda.is_available():
        raise RuntimeError("backend='cuda' requested, but CUDA is unavailable. Select 'reference' explicitly for small CPU tests.")
    try:
        from mamba_ssm import Mamba
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn  # noqa: F401
    except (ImportError, OSError) as exc:
        raise RuntimeError("The official mamba_ssm CUDA package and selective_scan kernel are required; no fallback is used.") from exc
    return Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
