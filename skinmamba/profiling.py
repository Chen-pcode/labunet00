"""Measured forward latency and transparent core-arithmetic FLOP estimates.

``flops`` counts Conv/Linear (one MAC = two FLOPs) and Mamba-1 projections,
depthwise convolution, and selective recurrence. It is NOT a complete model
FLOP total: normalization, general elementwise operations, interpolation,
sampling, and memory movement are outside this stated scope. ``flops_total``
is therefore None and ``flops_complete`` is False. Unknown computational
modules make ``flops`` None rather than silently omitting a possible SSM;
``flops_known_subtotal`` remains available for diagnosis.

PVM wrappers can expose ``profiling_scan_spec(inputs, output)`` with batch,
length, channels (= d_inner), state_size, dt_rank, groups, d_model and d_conv.
Such a wrapper owns its ``mamba`` subtree: that core is counted analytically,
not by its child hooks, which also handles official fused CUDA execution.
"""
from __future__ import annotations

import contextlib
import math
import statistics
import time
from collections import Counter
from typing import Any

import torch
from torch import nn


_FLOP_SCOPE = (
    "Core arithmetic estimate per input batch; Conv/Linear use 2 FLOPs/MAC; "
    "includes Mamba-1 projections, depthwise convolution and selective scan. "
    "Excludes normalization, general activations/elementwise operations, "
    "interpolation/grid sampling, reductions outside the scan, and memory movement."
)


def _mamba_core_flops(spec: dict[str, Any]) -> dict[str, int]:
    """Estimate Mamba-1 arithmetic, not executed kernel instruction counts.

    Approximate-discretization scan per (batch, token, inner channel):
    delta*A, exp, state multiply, delta*B*u, state add, and C readout sum
    cost 8*N-1 scalar operations. D*u + y costs 2; SiLU(z)*y is estimated
    at 5 (negation, exp, addition, division, multiplication). Thus 8*N+6
    with the default D/z. Each exp/division counts as one scalar operation,
    NOT one hardware instruction. No backward cost is included.
    """
    required = ("batch", "length", "channels", "state_size", "dt_rank", "groups", "d_model", "d_conv")
    values = {key: int(spec[key]) for key in required}
    if any(value <= 0 for value in values.values()):
        raise ValueError("Mamba profiling dimensions must all be positive.")
    b, length, inner, state, rank, groups, width, kernel = (values[key] for key in required)
    tokens = b * length * groups
    linear = 2 * tokens * (
        width * (2 * inner) + inner * (rank + 2 * state)
        + rank * inner + inner * width
    )
    conv = 2 * tokens * inner * kernel
    per_channel = 8 * state - 1
    if spec.get("has_skip", True):
        per_channel += 2
    if spec.get("has_gate", True):
        per_channel += 5
    scan = tokens * inner * per_channel
    return {"mamba_projections": linear, "mamba_depthwise_conv": conv, "selective_scan": scan}


def _is_standard_mamba(module: nn.Module) -> bool:
    # Recognize only the Mamba-1 shape/layout used by the official/reference
    # implementations; an unrelated module named Mamba is not sufficient.
    return all(hasattr(module, attr) for attr in (
        "d_model", "d_inner", "d_state", "dt_rank", "d_conv", "in_proj",
        "x_proj", "dt_proj", "out_proj", "conv1d", "A_log", "D",
    ))


def _count_core_flops(model: nn.Module, inputs: torch.Tensor) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    unsupported: set[str] = set()
    omitted: set[str] = set()
    handles = []
    named = list(model.named_modules())
    wrappers = [(name, module) for name, module in named if callable(getattr(module, "profiling_scan_spec", None))]
    owned: set[int] = set()
    for _, wrapper in wrappers:
        core = getattr(wrapper, "mamba", None)
        if core is None:
            raise ValueError("profiling_scan_spec wrapper must expose its owned core as .mamba.")
        owned.update(id(child) for child in core.modules())

    standards = [(name, module) for name, module in named if id(module) not in owned and _is_standard_mamba(module)]
    for _, core in standards:
        owned.update(id(child) for child in core.modules() if child is not core)
    recognized_core_ids = {id(module) for _, module in wrappers + standards}

    def wrapper_hook(module, args, output):
        spec = module.profiling_scan_spec(args, output)
        counts.update(_mamba_core_flops(spec))
        if spec.get("sampled", False):
            omitted.add("sampling and reconstruction")
        if spec.get("delta_scaled", False):
            omitted.add("geometric delta multiplication")

    def standard_hook(module, args, output):
        x = args[0]
        if x.ndim != 3:
            raise ValueError("Mamba-1 profiling expects [batch,length,d_model].")
        counts.update(_mamba_core_flops({
            "batch": x.shape[0], "length": x.shape[1], "channels": module.d_inner,
            "state_size": module.d_state, "dt_rank": module.dt_rank, "groups": 1,
            "d_model": module.d_model, "d_conv": module.d_conv,
            "has_skip": module.D is not None, "has_gate": True,
        }))

    def conv_hook(module, args, output):
        counts["conv"] += int(2 * output.numel() * (module.in_channels // module.groups) * math.prod(module.kernel_size))

    def linear_hook(module, args, output):
        counts["linear"] += int(2 * output.numel() * module.in_features)

    # Operations deliberately excluded from the declared common core metric.
    excluded = (
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm,
        nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
        nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.GELU, nn.SiLU, nn.Sigmoid, nn.Tanh,
        nn.Softmax, nn.Softplus, nn.ELU, nn.PReLU,
        nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d,
        nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d,
        nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d,
        nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d,
        nn.Upsample,
    )
    zero_cost = (nn.Identity, nn.Flatten, nn.Unflatten, nn.Dropout, nn.Dropout2d, nn.Dropout3d)
    try:
        for name, module in named:
            label = f"{name or '<root>'}: {type(module).__name__}"
            if id(module) in owned:
                continue
            if callable(getattr(module, "profiling_scan_spec", None)):
                handles.append(module.register_forward_hook(wrapper_hook))
            elif _is_standard_mamba(module):
                handles.append(module.register_forward_hook(standard_hook))
            elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                handles.append(module.register_forward_hook(conv_hook))
            elif isinstance(module, nn.Linear):
                handles.append(module.register_forward_hook(linear_hook))
            elif isinstance(module, excluded):
                omitted.add(label)
            elif isinstance(module, zero_cost):
                continue
            elif not list(module.children()):
                unsupported.add(label)
            elif any(word in type(module).__name__.lower() for word in ("mamba", "selectivescan", "ssm")):
                if not any(id(child) in recognized_core_ids for child in module.modules()):
                    unsupported.add(label + " (unrecognized state-space computation)")
        model(inputs)
    finally:
        for handle in handles:
            handle.remove()

    subtotal = int(sum(counts.values()))
    return {
        "flops": subtotal if not unsupported else None,
        "flops_known_subtotal": subtotal,
        "flops_total": None,
        "flops_complete": False,
        "flops_is_estimate": True,
        "flops_scope": _FLOP_SCOPE,
        "flops_breakdown": dict(counts),
        "flops_unsupported": sorted(unsupported),
        "flops_excluded_modules": sorted(omitted),
        "flops_scan_formula": "B*L*d_inner*groups*(8*d_state-1+2[D]+5[z]); exp/div count as 1 scalar op",
        "flops_notes": (
            "Mamba depthwise convolution counts L valid causal outputs; extra padded "
            "intermediates in a reference implementation are not included. Hooks cannot "
            "discover arbitrary functional tensor operations; this is not a full graph trace."
        ),
    }


def profile_model(
    model: nn.Module,
    input_shape: tuple[int, ...] = (1, 3, 256, 256),
    device: str = "cuda",
    precision: str = "fp32",
    warmup: int = 30,
    iterations: int = 100,
) -> dict[str, Any]:
    """Measure synchronous resident-input forward inference, never extrapolate FPS.

    Runs in inference/eval mode on zero-valued inputs; includes model forward
    only, not data loading, preprocessing, host transfer or sigmoid/threshold
    outside the model. CUDA is synchronized before/after every timed forward.
    ``fps`` is input batch size / measured mean seconds; latency values are ms
    PER BATCH. CPU measurements are explicitly labelled and are not GPU claims.

    Precision is fp32 or autocast fp16/bf16 (amp_fp16/amp_bf16 aliases accepted);
    CPU fp16 is rejected. The function
    preserves every module's training flag and returns a single-device model
    to its original device even on failure. Parameter dtypes are not changed.
    Mixed-device models are rejected to avoid ambiguous restoration.

    ``size_mb`` means FP32 parameter tensor bytes / 2**20 (MiB), excluding
    buffers, optimizer state and serialization overhead. CUDA peak memory is
    allocated memory, including resident model/input; CPU peak memory is None.
    FLOP values follow this module's explicitly limited core-arithmetic scope.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module.")
    if not input_shape or any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in input_shape):
        raise ValueError("input_shape must contain positive integer dimensions.")
    if not isinstance(warmup, int) or isinstance(warmup, bool) or warmup < 0:
        raise ValueError("warmup must be a nonnegative integer.")
    if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 1:
        raise ValueError("iterations must be a positive integer.")
    precision = {"amp_fp16": "fp16", "amp_bf16": "bf16"}.get(precision, precision)
    if precision not in ("fp32", "fp16", "bf16"):
        raise ValueError("precision must be fp32, fp16, or bf16.")
    target = torch.device(device)
    if target.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU and CUDA profiling are supported.")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no CPU fallback is performed.")
    if target.type == "cpu" and precision == "fp16":
        raise ValueError("CPU fp16 profiling is unsupported; use fp32 or bf16.")

    tensors = list(model.parameters()) + list(model.buffers())
    devices = {tensor.device for tensor in tensors}
    if len(devices) > 1:
        raise ValueError("Profiling requires a model on a single original device.")
    original_device = next(iter(devices), torch.device("cpu"))
    if original_device.type == "meta":
        raise ValueError("A meta-device model has no values to profile.")
    if any(tensor.is_floating_point() and tensor.dtype != torch.float32 for tensor in tensors):
        raise ValueError("Use FP32 model weights; precision selects autocast without changing weights.")
    modes = [(module, module.training) for module in model.modules()]
    params = sum(parameter.numel() for parameter in model.parameters())
    autocast_dtype = torch.float16 if precision == "fp16" else torch.bfloat16

    def autocast_context():
        return contextlib.nullcontext() if precision == "fp32" else torch.autocast(target.type, dtype=autocast_dtype)

    def sync():
        if target.type == "cuda":
            torch.cuda.synchronize(target)

    try:
        model.to(target)
        model.eval()
        inputs = torch.zeros(input_shape, dtype=torch.float32, device=target)
        with torch.inference_mode(), autocast_context():
            flop_result = _count_core_flops(model, inputs)
            for _ in range(warmup):
                model(inputs)
            sync()
            if target.type == "cuda":
                torch.cuda.reset_peak_memory_stats(target)
                initial_memory = torch.cuda.memory_allocated(target)
            timings = []
            for _ in range(iterations):
                sync()
                start = time.perf_counter()
                model(inputs)
                sync()
                timings.append((time.perf_counter() - start) * 1000.0)
            if target.type == "cuda":
                peak_bytes = torch.cuda.max_memory_allocated(target)
                peak_memory = peak_bytes / 2**20
                extra_memory = max(0, peak_bytes - initial_memory) / 2**20
            else:
                peak_memory, extra_memory = None, None

        mean_ms = statistics.fmean(timings)
        sorted_ms = sorted(timings)

        def percentile(q):
            position = (len(sorted_ms) - 1) * q
            low = int(position)
            high = min(low + 1, len(sorted_ms) - 1)
            return sorted_ms[low] + (sorted_ms[high] - sorted_ms[low]) * (position - low)

        return {
            "params": int(params),
            "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "size_mb": float(params * 4 / 2**20),
            "size_definition": "FP32 parameter tensor storage in MiB; excludes buffers and serialization overhead",
            "fps": float(input_shape[0] * 1000.0 / mean_ms),
            "latency_mean_ms": float(mean_ms),
            "latency_p50_ms": float(percentile(0.50)),
            "latency_p95_ms": float(percentile(0.95)),
            "peak_memory_mb": peak_memory,
            "peak_memory_increment_mb": extra_memory,
            "memory_definition": "CUDA peak allocated MiB incl. resident model/input; CPU not measured",
            "device": str(target),
            "device_name": torch.cuda.get_device_name(target) if target.type == "cuda" else "CPU",
            "precision": precision,
            "backend": getattr(model, "backend", "unspecified"),
            "input_shape": list(input_shape),
            "warmup": warmup,
            "iterations": iterations,
            "timing_scope": "Measured synchronized model forward per batch; resident zero input; excludes data pipeline",
            "torch_version": str(torch.__version__),
            "cpu_threads": torch.get_num_threads(),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            **flop_result,
        }
    finally:
        if original_device != target:
            model.to(original_device)
        for module, was_training in modes:
            module.training = was_training
