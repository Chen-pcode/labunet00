from __future__ import annotations

import copy
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch

from skinmamba.config import load_config
from skinmamba.data import (SkinDataset, make_manifest, audit_manifest, paired_files,
                            validate_source_split)
from skinmamba.engine import train, resume_signature
from skinmamba.evaluation import evaluate_checkpoint
from skinmamba.utils import load_checkpoint
from skinmamba.cli import aggregate_runs


@pytest.fixture
def fixture_data(tmp_path):
    root = tmp_path / "data"
    rng = np.random.default_rng(23)
    for domain in ("isic2017", "isic2018", "ph2"):
        for split in (("test",) if domain == "ph2" else ("train", "val", "test")):
            for folder in ("images", "masks"):
                (root / domain / split / folder).mkdir(parents=True)
            for i in range(2):
                name = f"{domain}_{split}_{i}"
                image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
                mask = np.zeros((32, 32), dtype=np.uint8)
                mask[8:24, 8 + i:24 + i] = 255
                Image.fromarray(image).save(root / domain / split / "images" / f"{name}.png")
                Image.fromarray(mask).save(root / domain / split / "masks" / f"{name}_segmentation.png")
    return root


def test_dataset_pairing_and_hash_leakage(fixture_data):
    source = fixture_data / "isic2018" / "train" / "images" / "isic2018_train_0.png"
    target = fixture_data / "isic2017" / "test" / "images" / "isic2017_test_0.png"
    target.write_bytes(source.read_bytes())
    manifest = make_manifest(fixture_data)
    report = audit_manifest(manifest, "isic2018")
    assert report["tests"]["isic2017"]["overlap_ids"] == ["isic2017_test_0"]
    assert report["tests"]["isic2017"]["clean_count"] == 1
    assert report["tests"]["ph2"]["total"] == 2
    target_mask = fixture_data / "isic2017" / "test" / "masks" / "isic2017_test_0_segmentation.png"
    target_mask.rename(target_mask.with_name("unmatched.png"))
    with pytest.raises(ValueError, match="Unpaired"):
        paired_files(fixture_data, "isic2017", "test")


def test_source_validation_exclusion_and_deterministic_augmentation(fixture_data):
    train_image = fixture_data / "isic2017/train/images/isic2017_train_0.png"
    val_image = fixture_data / "isic2017/val/images/isic2017_val_0.png"
    val_image.write_bytes(train_image.read_bytes())
    manifest = make_manifest(fixture_data)
    report = audit_manifest(manifest, "isic2017")
    assert report["source_train_val_overlap"] == ["isic2017_val_0"]
    validate_source_split(report, "exclude")
    with pytest.raises(ValueError, match="leakage"):
        validate_source_split(report, "error")
    records = [r for r in manifest["records"] if r["split"] == "train"]
    ds = SkinDataset(fixture_data, records, 32, augmentation="official")
    for field in ("image", "mask"):
        torch.testing.assert_close(ds[0][field], ds[0][field], rtol=0, atol=0)
    assert set(ds[0]["mask"].unique().tolist()).issubset({0, 1})


def test_complete_train_resume_and_three_domain_evaluation(fixture_data, tmp_path):
    torch.set_num_threads(1)
    config = load_config(Path(__file__).resolve().parents[1] / "configs/smoke_cpu.yaml")
    config["data"]["root"] = str(fixture_data)
    config["model"]["d_state"] = 2
    config["model"]["variant"] = "sampled_geometry"
    continuous, resumed = tmp_path / "continuous", tmp_path / "resumed"
    train(config, continuous, device="cpu")
    train(config, resumed, device="cpu", stop_after_epoch=1)
    train(config, resumed, device="cpu", resume=resumed / "last.pt")
    one, two = load_checkpoint(continuous / "last.pt"), load_checkpoint(resumed / "last.pt")
    for key in one["model"]:
        torch.testing.assert_close(one["model"][key], two["model"][key], rtol=0, atol=0)
    assert one["best_epoch"] == two["best_epoch"]
    report = evaluate_checkpoint(resumed / "best.pt", device="cpu", profile_iterations=2)
    assert {row["target"] for row in report["results"]} == {"isic2017", "isic2018", "ph2"}
    for row in report["results"]:
        for key in ("params", "flops", "size_mb", "fps", "dice", "iou", "miou", "accuracy", "sensitivity", "specificity", "f1", "hd95"):
            assert key in row
        assert row["dice"] == row["f1"]
    assert report["profile"]["device"] == "cpu"
    assert (resumed / "evaluation/summary.csv").is_file()
    bad = copy.deepcopy(config)
    bad["evaluation"]["threshold"] = .7
    assert resume_signature(bad) != resume_signature(config)
    with pytest.raises(ValueError, match="configuration changed"):
        train(bad, resumed, device="cpu", resume=resumed / "last.pt")


def test_config_override_source_guard():
    config_path = Path(__file__).resolve().parents[1] / "configs/base.yaml"
    with pytest.raises(ValueError, match="test-only"):
        load_config(config_path, ["data.source=ph2"])


def test_aggregation_separates_data_threshold_and_hardware(tmp_path):
    config = load_config(Path(__file__).resolve().parents[1] / "configs/base.yaml")
    row = dict(source="isic2018", target="ph2", subset="full", seed=42,
               dice=.8, hd95=4.0, hd95_failed_count=1, hd95_finite_count=199)
    base = {"config": config, "results": [row], "checkpoint_sha256": "one-checkpoint",
            "audit": {"fingerprint": "data-v1"}, "profile": {"device_name": "T4", "precision": "fp32"}}
    variants = [base]
    for key, value in (("device", "P100"), ("threshold", .7), ("fingerprint", "data-v2")):
        report = copy.deepcopy(base)
        if key == "device":
            report["profile"]["device_name"] = value
        elif key == "threshold":
            report["config"]["evaluation"]["threshold"] = value
        else:
            report["audit"]["fingerprint"] = value
        variants.append(report)
    variants.append(copy.deepcopy(base))  # Identical reevaluation is deduplicated.
    for i, report in enumerate(variants):
        folder = tmp_path / f"eval{i}"
        folder.mkdir()
        (folder / "results.json").write_text(json.dumps(report), encoding="utf-8")
    result = aggregate_runs(tmp_path, tmp_path / "aggregate.csv")
    assert len(result) == 4
    assert all(r["n_runs"] == 1 and r["hd95_failed_count_sum"] == 1 for r in result)
