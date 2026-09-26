"""Small real-file CPU integration check; never a segmentation benchmark."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from skinmamba.config import load_config
from skinmamba.data import make_manifest, SkinDataset, make_loader, audit_manifest
from skinmamba.engine import validate
from skinmamba.evaluation import evaluate_model
from skinmamba.losses import BCEDiceLoss
from skinmamba.models import build_model
from skinmamba.utils import seed_everything, write_json, environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="../data")
    parser.add_argument("--manifest")
    parser.add_argument("--output", default="reports/real_data_smoke.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8")) if args.manifest else make_manifest(args.data_root)
    reports = []
    for source in ("isic2017", "isic2018"):
        audit = audit_manifest(manifest, source)
        for variant in ("baseline", "unfused_control", "sampled_index", "sampled_constant", "sampled_geometry"):
            seed_everything(42)
            config = load_config(ROOT / "configs/base.yaml", ["model.backend=reference", f"model.variant={variant}"])
            model = build_model(config)
            criterion = BCEDiceLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
            def loader(domain, split, exclude=()):
                records = [r for r in manifest["records"] if r["domain"] == domain and r["split"] == split and r["id"] not in exclude][:2]
                ds = SkinDataset(args.data_root, records, 64, augmentation="official" if split == "train" else "none")
                return make_loader(ds, 2)
            batch = next(iter(loader(source, "train")))
            loss = criterion(model(batch["image"]), batch["mask"])
            loss.backward()
            assert torch.isfinite(loss)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            optimizer.step()
            scores = validate(model, loader(source, "val", audit["source_train_val_overlap"]), criterion, torch.device("cpu"), "fp32", .5)
            assert all(torch.isfinite(torch.tensor(v)) for v in scores.values())
            domains = {}
            for domain in ("isic2017", "isic2018", "ph2"):
                rows, metrics = evaluate_model(model, loader(domain, "test"), device="cpu")
                assert len(rows) == 2 and 0 <= metrics["dice"] <= 1
                domains[domain] = {"images_checked": len(rows), "metric_schema_valid": True}
            reports.append({"source": source, "variant": variant, "image_size": 64,
                            "train_images": 2, "validation_images": 2, "finite_loss_and_gradients": True,
                            "test_domains": domains, "sampling": model.sampling_report()})
            print(f"passed {source}/{variant}: train backward + source validation + three test loaders", flush=True)
    write_json(args.output, {"purpose": "One-step software integration using 2 real files per split; not accuracy or speed results",
                            "environment": environment(), "checks": reports, "status": "passed"})


if __name__ == "__main__":
    main()
