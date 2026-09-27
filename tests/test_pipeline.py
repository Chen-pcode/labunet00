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
from skinmamba.cli import aggregate_runs, main


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


def test_complete_train_resume_and_source_ph2_evaluation(fixture_data, tmp_path):
    torch.set_num_threads(1)
    # An overlapping test image must still be evaluated as part of the full set.
    source = fixture_data / "isic2018/train/images/isic2018_train_0.png"
    target = fixture_data / "isic2017/test/images/isic2017_test_0.png"
    target.write_bytes(source.read_bytes())
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
    assert {row["target"] for row in report["results"]} == {"isic2018", "ph2"}
    assert len(report["results"]) == 2
    assert all(row["subset"] == "full" and row["n"] == 2 for row in report["results"])
    assert report["audit"]["tests"]["isic2017"]["overlap_ids"] == ["isic2017_test_0"]
    for row in report["results"]:
        for key in ("params", "flops", "size_mb", "fps", "dice", "iou", "miou", "accuracy", "sensitivity", "specificity", "f1", "hd95"):
            assert key in row
        assert row["dice"] == row["f1"]
    assert report["profile"]["device"] == "cpu"
    assert (resumed / "evaluation/summary.csv").is_file()
    assert {row["target"] for row in evaluate_checkpoint(resumed / "best.pt", device="cpu",
        profile_iterations=1, include_other_isic=True)["results"]} == {"isic2017", "isic2018", "ph2"}
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
    legacy_clean = {**row, "subset": "clean", "dice": .99}
    base = {"config": config, "results": [row, legacy_clean], "checkpoint_sha256": "one-checkpoint",
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
    assert all(r["subset"] == "full" and r["dice_mean"] == .8 for r in result)


@pytest.mark.parametrize("flag,variant", [
    (["--baseline"], "baseline"), (["--main"], "reconstruction"),
    (["--main-experiment"], "reconstruction"),
    (["--ablation", "sampling_only"], "sampled_index"),
    (["--ablation", "constant_scale"], "sampled_constant"),
    (["--ablation", "uniform_sampling"], "sampled_index"),
    (["--ablation", "adaptive_sampling"], "adaptive_index"),
    (["--ablation", "adaptive_geometry"], "adaptive_geometry"),
    (["--ablation", "adaptive_coverage"], "adaptive_coverage"),
    (["--ablation", "no_bridge"], "baseline"),
    (["--ablation", "bce_only"], "baseline"),
])
def test_cli_experiment_epoch_seed_reach_training(monkeypatch, tmp_path, flag, variant):
    captured = {}
    def record(config, run_dir, device, resume, stop):
        captured.update(config)
        return tmp_path / "best.pt"
    monkeypatch.setattr("skinmamba.engine.train", record)
    config_path = Path(__file__).resolve().parents[1] / "configs/baseline_isic2017.yaml"
    main(["train", *flag, "--epoch", "1", "--seed", "2026", "--config", str(config_path),
          "--run-dir", str(tmp_path / "run"), "--set", "seed=42", "--set", "training.epochs=20"])
    assert captured["model"]["variant"] == variant
    assert captured["data"]["source"] == "isic2017"
    assert captured["training"]["epochs"] == 1 and captured["seed"] == 2026
    assert captured["training"]["t_max"] == 50
    if flag[-1] == "no_bridge":
        assert captured["model"]["bridge"] is False
    if flag[-1] == "bce_only":
        assert captured["loss"]["dice_weight"] == 0


@pytest.mark.parametrize("flags", [["--epoch", "0"], ["--seed", "-1"],
    ["--baseline", "--main"], ["--main", "--ablation", "sampling_only"],
    ["--ablation", "unknown"]])
def test_cli_rejects_invalid_experiment_settings(tmp_path, flags):
    with pytest.raises(SystemExit) as exc:
        main(["train", "--run-dir", str(tmp_path / "run"), *flags])
    assert exc.value.code == 2


def test_cli_runs_selected_baseline_for_one_epoch_and_seed(fixture_data, tmp_path):
    torch.set_num_threads(1)
    config_path = Path(__file__).resolve().parents[1] / "configs/smoke_cpu.yaml"
    run_dir = tmp_path / "cli_one_epoch"
    main(["run", "--baseline", "--epochs", "1", "--seed", "2026", "--config", str(config_path),
          "--data-root", str(fixture_data), "--run-dir", str(run_dir), "--device", "cpu",
          "--set", "model.d_state=2"])
    checkpoint = load_checkpoint(run_dir / "last.pt")
    assert checkpoint["epoch"] == 0 and len(checkpoint["history"]) == 1
    assert checkpoint["config"]["seed"] == 2026
    assert checkpoint["config"]["model"]["variant"] == "baseline"
    report = json.loads((run_dir / "evaluation/results.json").read_text(encoding="utf-8"))
    assert len(report["results"]) == 2
    assert all(row["subset"] == "full" and row["seed"] == 2026 for row in report["results"])


def test_cli_runs_cclas_and_profiles_one_epoch(fixture_data, tmp_path):
    torch.set_num_threads(1)
    config_path = Path(__file__).resolve().parents[1] / "configs/smoke_cpu.yaml"
    run_dir = tmp_path / "cclas_one_epoch"
    main(["run", "--ablation", "cclas", "--epoch", "1", "--seed", "2026", "--config", str(config_path),
          "--data-root", str(fixture_data), "--run-dir", str(run_dir), "--device", "cpu",
          "--set", "model.d_state=2"])
    checkpoint = load_checkpoint(run_dir / "best.pt")
    assert checkpoint["config"]["model"]["variant"] == "cclas"
    report = json.loads((run_dir / "evaluation/results.json").read_text(encoding="utf-8"))
    assert report["sampling"]["coverage"] and report["sampling"]["bounded_delta"]
    assert all("empty_prediction_count" in row for row in report["results"])
    assert report["profile"]["flops"] is not None
    assert any("score_head" in key for key in checkpoint["model"])
    assert report["profile"]["params"] > 0


@pytest.mark.parametrize("source", ["isic2017", "isic2018"])
def test_reconstruction_resume_and_full_evaluation(fixture_data, tmp_path, source):
    from skinmamba.config import select_experiment
    torch.set_num_threads(1)
    config = select_experiment(load_config(Path(__file__).resolve().parents[1] / "configs/smoke_cpu.yaml"), experiment="main")
    config["data"].update(root=str(fixture_data), source=source)
    config["model"]["state_dim"] = 4
    continuous, resumed = tmp_path / "continuous", tmp_path / "resumed"
    train(config, continuous, device="cpu")
    train(config, resumed, device="cpu", stop_after_epoch=1)
    train(config, resumed, device="cpu", resume=resumed / "last.pt")
    one, two = load_checkpoint(continuous / "last.pt"), load_checkpoint(resumed / "last.pt")
    for key in one["model"]:
        torch.testing.assert_close(one["model"][key], two["model"][key], rtol=0, atol=0)
    report = evaluate_checkpoint(resumed / "best.pt", device="cpu", profile_iterations=1, save_predictions=True)
    assert {r["target"] for r in report["results"]} == {source, "ph2"}
    assert report["model_family"] == "reconstruction"
    assert report["profile"]["flops"] > 0
    assert report["profile"]["flops_unsupported"] == []
    assert report["sampling"] is None
    assert len(list((resumed / "evaluation/predictions").rglob("*.png"))) == 4
