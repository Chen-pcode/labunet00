"""Random-input PSM forward/backward/optimizer/profile checks on actual device."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from skinmamba.config import load_config, select_experiment, validate_config
from skinmamba.domain import training_objective
from skinmamba.engine import get_device, amp_context
from skinmamba.models import build_model
from skinmamba.losses import BCEDiceLoss
from skinmamba.persistent_experiments import PSM_PRESETS
from skinmamba.profiling import profile_model
from skinmamba.utils import write_json, seed_everything, environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["fp32", "amp_fp16", "all"], default="fp32")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--variants", nargs="+", choices=PSM_PRESETS, default=list(PSM_PRESETS))
    parser.add_argument("--output", default="reports/psm_device_check.json")
    args = parser.parse_args()
    device = get_device(args.device)
    torch.set_num_threads(1)
    report = dict(status="running", environment=environment(), checks=[],
                  purpose="Synthetic software/device check, not trained accuracy or reliable performance comparison")
    try:
        for variant in args.variants:
            for precision in ["fp32", "amp_fp16"] if args.precision == "all" else [args.precision]:
                seed_everything(42)
                c = select_experiment(load_config(ROOT / "configs/psm/isic2018.yaml"), ablation=variant)
                c["data"]["image_size"] = args.image_size
                c["training"]["precision"] = precision
                if device.type == "cpu":
                    c["model"]["backend"] = "reference"
                validate_config(c)
                model = build_model(c).to(device).train()
                optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
                scaler = torch.amp.GradScaler("cuda", enabled=precision == "amp_fp16")
                x = torch.rand(args.batch_size, 3, args.image_size, args.image_size, device=device) * 255
                y = torch.randint(0, 2, (args.batch_size, 1, args.image_size, args.image_size), device=device).float()
                with amp_context(device, precision):
                    loss, _ = training_objective(model, x, y, BCEDiceLoss(), c, epoch=0)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError("Missing or nonfinite gradients")
                scaler.step(optimizer)
                scaler.update()
                model.eval()
                with torch.inference_mode(), amp_context(device, precision):
                    if not torch.isfinite(model(x)).all():
                        raise FloatingPointError("Nonfinite post-update output")
                profile = profile_model(model, (1, 3, args.image_size, args.image_size), str(device),
                                        precision, warmup=1, iterations=2)
                if profile["flops"] is None:
                    raise RuntimeError(f"Unknown profiling operators: {profile['flops_unsupported']}")
                report["checks"].append(dict(variant=variant, precision=precision, loss=float(loss.detach()),
                    params=profile["params"], flops=profile["flops"], finite_forward_backward_update=True))
                print(f"PASS {variant}/{precision}", flush=True)
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(args.output, report)


if __name__ == "__main__":
    main()
