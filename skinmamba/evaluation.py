"""One evaluator for all models and test domains.

External nn.Modules may call evaluate_model with output_kind='logits' or
'probabilities'. Return a single BCHW tensor or provide an adapter explicitly.
"""
from __future__ import annotations

from pathlib import Path
import copy
import numpy as np
from PIL import Image
import torch

from .data import SkinDataset, make_loader, make_manifest, audit_manifest
from .engine import get_device, amp_context
from .metrics import segmentation_metrics, aggregate_metrics
from .models import build_model
from .profiling import profile_model
from .utils import load_checkpoint, write_json, write_csv, file_sha256, environment, seed_everything


@torch.inference_mode()
def evaluate_model(model, loader, device="cuda", threshold=.5, output_kind="logits", precision="fp32", save_dir=None):
    if output_kind not in ("logits", "probabilities"):
        raise ValueError("output_kind must explicitly name logits or probabilities")
    device = get_device(device)
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    rows = []
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    try:
        for batch in loader:
            with amp_context(device, precision):
                output = model(batch["image"].to(device, non_blocking=True))
            if not isinstance(output, torch.Tensor) or output.shape != batch["mask"].shape:
                raise ValueError("Evaluator expects a single Bx1xHxW output matching masks; use a model adapter")
            prob = output.float().sigmoid() if output_kind == "logits" else output.float()
            if not torch.isfinite(prob).all() or prob.min() < 0 or prob.max() > 1:
                raise ValueError("Nonfinite/out-of-range model probabilities")
            predictions = (prob >= threshold).cpu().numpy()[:, 0]
            truth = batch["mask"].numpy()[:, 0] >= .5
            for key, pred, target in zip(batch["id"], predictions, truth):
                rows.append({"id": key, **segmentation_metrics(pred, target)})
                if save_dir:
                    Image.fromarray(pred.astype(np.uint8) * 255).save(Path(save_dir) / f"{key}.png")
    finally:
        for module, was_training in modes:
            module.training = was_training
    return rows, aggregate_metrics(rows)


def evaluate_checkpoint(checkpoint_path, data_root=None, output_dir=None, device="cuda", backend=None,
                        save_predictions=False, run_profile=True, profile_iterations=None):
    checkpoint = load_checkpoint(checkpoint_path)
    config = copy.deepcopy(checkpoint["config"])
    if data_root:
        config["data"]["root"] = str(data_root)
    if backend:
        config["model"]["backend"] = backend
    seed_everything(config["seed"], config["training"].get("strict_determinism", False))
    device = get_device(device)
    output_dir = Path(output_dir or Path(checkpoint_path).parent / "evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    current = make_manifest(config["data"]["root"])
    if current["fingerprint"] != checkpoint["manifest"]["fingerprint"]:
        raise ValueError("Evaluation dataset/splits differ from training audit. Refusing silently changed test data.")
    audit = audit_manifest(current, config["data"]["source"])
    write_json(output_dir / "data_audit.json", audit)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    complexity = {}
    if run_profile:
        options = config.get("profiling", {})
        size = config["data"]["image_size"]
        complexity = profile_model(model, input_shape=(1, 3, size, size), device=str(device),
            precision=config["training"]["precision"], warmup=options.get("warmup", 30),
            iterations=profile_iterations or options.get("iterations", 100))
        write_json(output_dir / "profile.json", complexity)
    results = []
    for domain in ("isic2017", "isic2018", "ph2"):
        records = [r for r in current["records"] if r["domain"] == domain and r["split"] == "test"]
        ds = SkinDataset(config["data"]["root"], records, config["data"]["image_size"], config["data"]["normalization"])
        loader = make_loader(ds, config["evaluation"]["batch_size"], config["training"]["workers"])
        rows, scores = evaluate_model(model, loader, str(device), config["evaluation"]["threshold"],
            precision=config["training"]["precision"], save_dir=output_dir / "predictions" / domain if save_predictions else None)
        write_csv(output_dir / f"{domain}_per_image.csv", rows)
        row = {"source": config["data"]["source"], "target": domain, "subset": "full",
               "seed": config["seed"], "variant": config["model"]["variant"], "n": len(rows),
               "external": domain != config["data"]["source"],
               **scores, **{k: complexity.get(k) for k in ("params", "flops", "size_mb", "fps",
                   "flops_scope", "flops_complete", "device_name", "precision", "latency_p50_ms",
                   "latency_p95_ms", "peak_memory_mb")}}
        results.append(row)
        print(f"{domain}/full: n={len(rows)}, dice={scores.get('dice')}, iou={scores.get('iou')}, hd95={scores.get('hd95')}", flush=True)
    report = {"checkpoint": str(Path(checkpoint_path).resolve()), "checkpoint_sha256": file_sha256(checkpoint_path),
        "selected_epoch": checkpoint["epoch"] + 1, "config": config, "environment": environment(),
        "profile": complexity, "audit": audit, "results": results,
        "sampling": model.sampling_report() if hasattr(model, "sampling_report") else None}
    write_json(output_dir / "results.json", report)
    write_csv(output_dir / "summary.csv", results)
    return report
