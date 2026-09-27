"""Verify new presets on the selected device; random-data checks, not a benchmark."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from skinmamba.config import load_config, select_experiment
from skinmamba.engine import get_device, amp_context
from skinmamba.experiments import RECONSTRUCTION_PRESETS
from skinmamba.losses import BCEDiceLoss
from skinmamba.models import build_model
from skinmamba.profiling import profile_model
from skinmamba.utils import environment, seed_everything, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["fp32", "amp_fp16", "all"], default="fp32")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--variants", nargs="+", default=list(RECONSTRUCTION_PRESETS), choices=list(RECONSTRUCTION_PRESETS))
    parser.add_argument("--output", default="reports/reconstruction_device_check.json")
    args = parser.parse_args()
    device = get_device(args.device)
    torch.set_num_threads(1)
    precisions = ["fp32", "amp_fp16"] if args.precision == "all" else [args.precision]
    report = {"purpose": "Random-input software verification only; not trained accuracy or reliable timing",
              "environment": environment(), "checks": [], "status": "running"}
    try:
        for preset in args.variants:
            for precision in precisions:
                seed_everything(42)
                config = select_experiment(load_config(ROOT / "configs/main_isic2018.yaml",
                    [f"data.image_size={args.image_size}"]), ablation=preset)
                model = build_model(config).to(device).train()
                image = torch.randn(2, 3, args.image_size, args.image_size, device=device)
                target = torch.randint(0, 2, (2, 1, args.image_size, args.image_size), device=device).float()
                optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
                scaler = torch.amp.GradScaler("cuda", enabled=precision == "amp_fp16")
                with amp_context(device, precision):
                    logits = model(image)
                loss = BCEDiceLoss()(logits, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite {preset}/{precision} loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError(f"Missing/nonfinite {preset}/{precision} gradients")
                scaler.step(optimizer)
                scaler.update()
                model.eval()
                with torch.inference_mode(), amp_context(device, precision):
                    if not torch.isfinite(model(image)).all():
                        raise FloatingPointError(f"Nonfinite {preset}/{precision} post-update logits")
                profile = profile_model(model, (1, 3, args.image_size, args.image_size), str(device),
                                        precision, warmup=1, iterations=2)
                if profile["flops"] is None:
                    raise RuntimeError(f"Uncounted operators: {profile['flops_unsupported']}")
                report["checks"].append({"variant": preset, "precision": precision,
                    "finite_forward_backward_update": True, "params": profile["params"],
                    "flops": profile["flops"], "flops_breakdown": profile["flops_breakdown"]})
                print(f"PASS {preset}/{precision}: params={profile['params']}, core_flops={profile['flops']}", flush=True)
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(args.output, report)


if __name__ == "__main__":
    main()
