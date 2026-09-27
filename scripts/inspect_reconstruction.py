"""Export source-validation feature differences from a trained checkpoint.

Ground-truth masks are used only for saved diagnostics, never model routing.
Map brightness is feature magnitude, not a calibrated uncertainty probability.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch
from scipy.ndimage import binary_erosion, binary_dilation
from torch.nn import functional as F

from skinmamba.data import SkinDataset, make_manifest, make_loader, audit_manifest
from skinmamba.engine import get_device, amp_context
from skinmamba.models import build_model
from skinmamba.utils import load_checkpoint, write_json, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-images", type=int, default=8)
    args = parser.parse_args()
    if args.max_images < 1:
        parser.error("--max-images must be positive")
    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint["config"]
    root = args.data_root or config["data"]["root"]
    manifest = make_manifest(root)
    if manifest["fingerprint"] != checkpoint["manifest"]["fingerprint"]:
        raise ValueError("Dataset changed since training")
    audit = audit_manifest(manifest, config["data"]["source"])
    excluded = set(audit["source_train_val_overlap"])
    records = [r for r in manifest["records"] if r["domain"] == config["data"]["source"]
               and r["split"] == "val" and r["id"] not in excluded][:args.max_images]
    dataset = SkinDataset(root, records, config["data"]["image_size"], config["data"]["normalization"])
    device = get_device(args.device)
    model = build_model(config).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    if not hasattr(model, "set_diagnostics"):
        raise ValueError("This command requires a reconstruction-family checkpoint")
    model.set_diagnostics(True)
    output = Path(args.output_dir)
    rows = []
    with torch.inference_mode():
        for batch in make_loader(dataset, 1, 0):
            with amp_context(device, config["training"]["precision"]):
                logits = model(batch["image"].to(device))
            prob = logits.float().sigmoid()[0, 0].cpu().numpy()
            truth = batch["mask"][0, 0].numpy() >= .5
            pred = prob >= config["evaluation"]["threshold"]
            boundary = binary_dilation(truth, iterations=2) ^ binary_erosion(truth, iterations=2)
            error = pred != truth
            folder = output / batch["id"][0]
            folder.mkdir(parents=True, exist_ok=True)
            maps = {"probability": prob, "target": truth, "prediction": pred,
                    "error": error, "target_boundary_band": boundary}
            for name, feature in model.diagnostic_maps().items():
                feature = F.interpolate(feature.float()[:, None], size=truth.shape, mode="bilinear", align_corners=False)[0, 0]
                array = feature.cpu().numpy()
                maps[name.replace("/", "_")] = array
                rows.append({"id": batch["id"][0], "feature": name,
                             "mean_error_pixels": float(array[error].mean()) if error.any() else None,
                             "mean_correct_pixels": float(array[~error].mean()) if (~error).any() else None,
                             "mean_boundary_band": float(array[boundary].mean()) if boundary.any() else None,
                             "mean_other_pixels": float(array[~boundary].mean()) if (~boundary).any() else None})
            np.savez_compressed(folder / "maps.npz", **maps)
            for name, array in maps.items():
                gray = array.astype(np.float32)
                if name not in {"probability", "target", "prediction", "error", "target_boundary_band"}:
                    low, high = np.percentile(gray, [1, 99])
                    gray = (gray - low) / max(float(high - low), 1e-8)
                Image.fromarray((np.clip(gray, 0, 1) * 255).astype(np.uint8)).save(folder / f"{name}.png")
    if rows:
        write_csv(output / "feature_region_means.csv", rows)
    write_json(output / "protocol.json", {"source": config["data"]["source"], "split": "val",
        "n": len(records), "ids": [r["id"] for r in records], "checkpoint": str(args.checkpoint),
        "interpretation": "Descriptive diagnostic only; feature difference is not a proven boundary or error estimator",
        "png_scaling": "Feature maps use independent 1st/99th percentile display scales; quantitative raw maps in npz",
        "gt_in_model_forward": False})
    print(f"Saved source-validation diagnostics for {len(records)} images to {output}")


if __name__ == "__main__":
    main()
