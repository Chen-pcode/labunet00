"""Fail-closed, real-GPU Mamba numerical and network verification.

No CUDA check is skipped or replaced by a CPU result. Run on the intended
Kaggle GPU after installing the official scan packages.
"""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def metadata_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def tolerance(precision):
    return {"fp32": {"forward_atol": 1e-4, "forward_rtol": 2e-3, "gradient_atol": 5e-4, "gradient_rtol": 1e-2},
            "amp_fp16": {"forward_atol": 3e-3, "forward_rtol": 3e-2, "gradient_atol": 5e-3, "gradient_rtol": 8e-2},
            "amp_bf16": {"forward_atol": 2e-2, "forward_rtol": 8e-2, "gradient_atol": 3e-2, "gradient_rtol": 1.5e-1}}[precision]


def autocast(torch, precision):
    return torch.autocast("cuda", enabled=precision != "fp32",
                          dtype=torch.bfloat16 if precision == "amp_bf16" else torch.float16)


def compare(torch, actual, expected, *, atol, rtol):
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise AssertionError("Non-finite values in the CUDA/reference comparison.")
    torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol)
    return float((actual.detach().float() - expected.detach().float()).abs().max().cpu())


def short_parity(torch, precision, mode):
    from mamba_ssm import Mamba
    from skinmamba.models.mamba import ReferenceMamba, mamba_forward

    torch.manual_seed(812)
    official = Mamba(d_model=8, d_state=8, d_conv=4, expand=2).cuda()
    reference = ReferenceMamba(8, 8, 4, 2).cuda()
    reference.load_state_dict(official.state_dict(), strict=True)
    values = torch.randn(2, 19, 8, device="cuda")
    x = values.clone().requires_grad_(True)
    xr = values.clone().requires_grad_(True)
    factors = torch.linspace(.25, 2.5, 19, device="cuda")
    factors[0] = 1
    limits = tolerance(precision)
    with autocast(torch, precision):
        if mode == "official_baseline":
            y, yr = official(x), reference(xr)
        else:
            ds = factors if mode == "geometry" else None
            y = mamba_forward(official, x, ds, "cuda")
            yr = mamba_forward(reference, xr, ds, "reference")
    max_output = compare(torch, y, yr, atol=limits["forward_atol"], rtol=limits["forward_rtol"])
    # Fixed float32 probe gives both implementations identical upstream gradients.
    probe = torch.randn_like(y, dtype=torch.float32)
    (y.float() * probe).mean().backward()
    (yr.float() * probe).mean().backward()
    max_input = compare(torch, x.grad, xr.grad, atol=limits["gradient_atol"], rtol=limits["gradient_rtol"])
    maxima = {}
    for name, parameter in official.named_parameters():
        ref_parameter = dict(reference.named_parameters())[name]
        if parameter.grad is None or ref_parameter.grad is None:
            raise AssertionError(f"Missing gradient: {name}")
        maxima[name] = compare(torch, parameter.grad, ref_parameter.grad,
                               atol=limits["gradient_atol"], rtol=limits["gradient_rtol"])
    cross_precision = None
    if precision != "fp32":
        # Also compare AMP to a float32 reference on identical weights/input.
        # Matching CUDA and reference under the same autocast is insufficient.
        reference.zero_grad(set_to_none=True)
        xf = values.clone().requires_grad_(True)
        with torch.autocast("cuda", enabled=False):
            yf = (reference(xf) if mode == "official_baseline" else
                  mamba_forward(reference, xf, factors if mode == "geometry" else None, "reference"))
        error = compare(torch, y, yf, atol=limits["forward_atol"], rtol=limits["forward_rtol"])
        (yf.float() * probe).mean().backward()
        input_error = compare(torch, x.grad, xf.grad,
                              atol=limits["gradient_atol"], rtol=limits["gradient_rtol"])
        parameter_errors = {}
        for name, parameter in official.named_parameters():
            parameter_errors[name] = compare(torch, parameter.grad, dict(reference.named_parameters())[name].grad,
                                              atol=limits["gradient_atol"], rtol=limits["gradient_rtol"])
        cross_precision = {"reference_precision": "fp32", "max_forward_abs_error": error,
                           "max_input_gradient_abs_error": input_error,
                           "parameter_gradient_max_abs_errors": parameter_errors}
    if mode == "unfused_control":
        # Direct fused/manual comparison using exactly the official parameters.
        with torch.no_grad(), autocast(torch, precision):
            fused = official(values)
            manual = mamba_forward(official, values, backend="cuda")
        compare(torch, manual, fused, atol=limits["forward_atol"], rtol=limits["forward_rtol"])
    return {"precision": precision, "mode": mode, "tolerances": limits,
            "max_forward_abs_error": max_output, "max_input_gradient_abs_error": max_input,
            "parameter_gradient_max_abs_errors": maxima, "amp_vs_fp32": cross_precision}


def network_check(torch, precision, variant):
    from skinmamba.losses import BCEDiceLoss
    from skinmamba.models import build_model

    torch.manual_seed(912)
    model = build_model({"backend": "cuda", "variant": variant, "sample_ratio": .5,
                         "sampling_power": 1.5, "geometry_stage": "encoder4"}).cuda().train()
    optimizer = torch.optim.SGD(model.parameters(), lr=.001)
    image = torch.randn(1, 3, 256, 256, device="cuda")
    yy, xx = torch.meshgrid(torch.arange(256, device="cuda"), torch.arange(256, device="cuda"), indexing="ij")
    target = (((xx - 123) ** 2 + (yy - 132) ** 2) < 60 ** 2).float()[None, None]
    torch.cuda.reset_peak_memory_stats()
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "amp_fp16", init_scale=1024.0)
    with autocast(torch, precision):
        logits = model(image)
        loss = BCEDiceLoss()(logits, target)
    if logits.shape != target.shape or not torch.isfinite(logits).all() or not torch.isfinite(loss):
        raise AssertionError("Invalid full-network output or loss.")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradient_norms = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise AssertionError(f"Missing full-network gradient: {name}")
        if not torch.isfinite(parameter.grad).all():
            raise AssertionError(f"Non-finite full-network gradient: {name}")
        gradient_norms[name] = float(parameter.grad.detach().float().norm().cpu())
    if not any(norm > 0 for norm in gradient_norms.values()):
        raise AssertionError("All full-network gradients are zero.")
    for name in ("encoder4.0.mamba.A_log", "encoder4.0.mamba.dt_proj.weight", "encoder4.0.mamba.x_proj.weight"):
        if gradient_norms[name] == 0:
            raise AssertionError(f"Selected-stage recurrence gradient is zero: {name}")
    torch.cuda.synchronize()
    return {"precision": precision, "variant": variant, "input_shape": [1, 3, 256, 256],
            "loss": float(loss.detach().cpu()), "parameters": sum(p.numel() for p in model.parameters()),
            "gradient_scale": scaler.get_scale(),
            "sampling": model.sampling_report(), "gradient_norms": gradient_norms,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", choices=["fp32", "amp_fp16", "amp_bf16", "all"], default="all")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "cuda_verification.json")
    args = parser.parse_args()
    report = {"status": "failed", "checks": [], "environment": {"python": sys.version}}
    try:
        import torch

        report["environment"].update({"torch": torch.__version__, "torch_cuda": torch.version.cuda,
                                      "mamba_ssm": metadata_version("mamba-ssm"),
                                      "causal_conv1d": metadata_version("causal-conv1d")})
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this verification; CPU checks are not a substitute.")
        import mamba_ssm  # noqa: F401
        import causal_conv1d  # noqa: F401
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn  # noqa: F401

        properties = torch.cuda.get_device_properties(0)
        report["environment"].update({"device": properties.name, "capability": list(torch.cuda.get_device_capability(0)),
                                      "total_memory_bytes": properties.total_memory})
        if args.precision == "amp_bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("The requested BF16 precision is not supported by this GPU.")
        # Disable TF32 to make fp32 reference comparisons meaningful.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        precisions = ["fp32", "amp_fp16"] if args.precision == "all" else [args.precision]
        report["requested_precisions"] = precisions
        jobs = []
        for precision in precisions:
            jobs.extend(("short_parity", precision, mode) for mode in ("official_baseline", "unfused_control", "geometry"))
            jobs.extend(("network", precision, variant) for variant in
                        ("baseline", "unfused_control", "sampled_index", "sampled_constant", "sampled_geometry"))
        for kind, precision, mode in jobs:
            label = f"{kind}/{precision}/{mode}"
            print(f"Checking {label}", flush=True)
            record = {"check": label, "status": "failed"}
            try:
                record.update(short_parity(torch, precision, mode) if kind == "short_parity"
                              else network_check(torch, precision, mode))
                record["status"] = "passed"
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                record["traceback"] = traceback.format_exc()
                print(record["error"], file=sys.stderr, flush=True)
            report["checks"].append(record)
            gc.collect()
            torch.cuda.empty_cache()
        report["status"] = "passed" if all(check["status"] == "passed" for check in report["checks"]) else "failed"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        print(report["error"], file=sys.stderr, flush=True)
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"CUDA verification {report['status']}: {args.output}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
