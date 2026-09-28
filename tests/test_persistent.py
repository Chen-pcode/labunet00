from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

from skinmamba.config import load_config, select_experiment, validate_config
from skinmamba.domain import fixed_style_view, style_view, state_consistency, training_objective
from skinmamba.models import build_model
from skinmamba.persistent_experiments import PSM_PRESETS, DOMAIN_DEFAULTS
from skinmamba.profiling import profile_model
from skinmamba.state_diagnostics import compare_distributions, mmd2, diagnose_states, _matched_control
from skinmamba.losses import BCEDiceLoss
from skinmamba.utils import load_checkpoint
from skinmamba.engine import train
from skinmamba.evaluation import evaluate_checkpoint
from skinmamba.cli import main
from test_pipeline import fixture_data

ROOT = Path(__file__).resolve().parents[1]


def config_for(name="psm_main"):
    c = select_experiment(load_config(ROOT / "configs/psm/isic2018.yaml"), ablation=name)
    c["model"].update(backend="reference", d_state=2, expand=1)
    c["data"]["image_size"] = 32
    c["training"].update(epochs=2, t_max=2, workers=0, batch_size=2)
    c["profiling"].update(warmup=0, iterations=1)
    return c


@pytest.mark.parametrize("name", list(PSM_PRESETS))
def test_all_presets_forward_backward_profile(name):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    config = config_for(name)
    validate_config(config)
    model = build_model(config)
    image, mask = torch.rand(2, 3, 32, 32) * 255, torch.randint(0, 2, (2, 1, 32, 32)).float()
    loss, components = training_objective(model, image, mask, BCEDiceLoss(), config, 0)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    if config["model"]["family"] == "persistent":
        assert model.writers[0].query.weight.grad.abs().sum() > 0
        assert model.readers[0].affine.weight.grad.abs().sum() > 0
    profile = profile_model(model, (1, 3, 32, 32), "cpu", warmup=0, iterations=1)
    assert profile["flops"] > 0 and profile["flops_unsupported"] == []


def test_no_read_equals_backbone_and_state_resets():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    baseline = build_model(config_for("psm_baseline"))
    torch.manual_seed(7)
    model = build_model(config_for())
    for k, v in baseline.state_dict().items():
        torch.testing.assert_close(v, model.state_dict()[k], rtol=0, atol=0)
    image = torch.rand(2, 3, 32, 32) * 255
    with torch.no_grad():
        torch.testing.assert_close(baseline(image), model(image, intervention="no_read"), rtol=0, atol=0)
        first, aux = model(image, return_states=True)
        model(image.flip(0))
        second, again = model(image, return_states=True)
        separate = torch.cat([model(x[None]) for x in image])
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(first, separate, atol=3e-5, rtol=1e-4)
    for a, b in zip(aux["states"], again["states"]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_carry_control_same_capacity_and_actual_cross_scale_gradient():
    torch.set_num_threads(1)
    persistent = build_model(config_for("psm_main"))
    independent = build_model(config_for("psm_independent_dom"))
    independent.load_state_dict(persistent.state_dict(), strict=True)
    assert sum(p.numel() for p in persistent.parameters()) == sum(p.numel() for p in independent.parameters())
    image = torch.rand(2, 3, 32, 32) * 255
    _, aux = persistent(image, return_states=True)
    aux["queries"][0].retain_grad()
    aux["states"][-1].sum().backward()
    assert aux["queries"][0].grad.abs().sum() > 0
    _, separate = independent(image, return_states=True)
    separate["queries"][0].retain_grad()
    separate["states"][-1].sum().backward()
    assert separate["queries"][0].grad is None
    torch.testing.assert_close(persistent(image, intervention="no_carry"), independent(image))


def test_style_is_reproducible_and_batch_independent():
    image = torch.rand(3, 3, 12, 12)
    keys = ["one", "two", "three"]
    together = fixed_style_view(image, keys, "unit", DOMAIN_DEFAULTS, 8)
    separate = torch.cat([fixed_style_view(image[i:i + 1], [key], "unit", DOMAIN_DEFAULTS, 8)
                          for i, key in enumerate(keys)])
    torch.testing.assert_close(together, separate, rtol=0, atol=0)
    scaled = fixed_style_view(image * 255, keys, "official_minmax255", DOMAIN_DEFAULTS, 8)
    torch.testing.assert_close(together * 255, scaled, atol=1e-4, rtol=1e-5)
    assert together.min() >= 0 and together.max() <= 1
    assert not torch.allclose(together, image)
    zero = {**DOMAIN_DEFAULTS, **dict.fromkeys(("brightness", "contrast", "color", "gamma"), 0.)}
    torch.testing.assert_close(style_view(image, "unit", zero), image)
    generator = torch.Generator().manual_seed(19)
    first = style_view(image, "unit", DOMAIN_DEFAULTS, generator)
    torch.randn(1000)  # An architecture consuming more global RNG must not alter styles.
    second = style_view(image, "unit", DOMAIN_DEFAULTS, torch.Generator().manual_seed(19))
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_moment_loss_hand_calculation_singleton_and_pair_information():
    opts = {**DOMAIN_DEFAULTS, "variance_weight": 0.}
    a = torch.tensor([[0., 0.], [2., 4.]], requires_grad=True)
    b = a.detach() + torch.tensor([1., -1.])
    total, terms = state_consistency([a], [b], opts)
    assert terms["pair"].item() == pytest.approx(1.)
    assert terms["mean"].item() == pytest.approx(1.)
    assert terms["covariance"].item() == pytest.approx(0.)
    total.backward()
    assert a.grad.abs().sum() > 0
    _, shuffled = state_consistency([a], [a.flip(0)], opts)
    assert shuffled["mean"] == 0 and shuffled["covariance"] == 0 and shuffled["pair"] > 0
    single, terms = state_consistency([a[:1]], [b[:1]], opts)
    assert torch.isfinite(single) and terms["covariance"] == 0 and terms["variance"] == 0


def test_distances_known_shift_and_scale_collapse_check():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(30, 4))
    y, ref = x + np.array([1, 0, 0, 0]), rng.normal(size=(40, 4))
    result, draws = compare_distributions(x, y, ref, paired=True, bootstrap=8)
    assert result["mu_l2"] == pytest.approx(1.)
    assert result["cov_fro"] == pytest.approx(0., abs=1e-12)
    smaller, _ = compare_distributions(.1 * x, .1 * y, .1 * ref, paired=True, bootstrap=8)
    assert smaller["mu_l2"] == pytest.approx(.1)
    assert smaller["mu_l2_normalized"] == pytest.approx(result["mu_l2_normalized"])
    assert smaller["mmd2"] == pytest.approx(result["mmd2"])
    assert mmd2(x, x, 1., paired=True) == pytest.approx(0., abs=1e-12)
    assert draws.shape == (8, 4)


def test_control_guard_and_baseline_cli_schedule(monkeypatch, tmp_path):
    a, b = config_for(), config_for("psm_memory_aug")
    _matched_control(a, b)
    b["domain"]["two_view"] = False
    with pytest.raises(ValueError, match="two-view"):
        _matched_control(a, b)
    captured = {}
    monkeypatch.setattr("skinmamba.engine.train", lambda c, *args: captured.update(c))
    main(["train", "--baseline", "--config", str(ROOT / "configs/psm/isic2018.yaml"),
          "--epoch", "3", "--seed", "2026", "--run-dir", str(tmp_path)])
    assert captured["training"]["t_max"] == 3
    assert not captured["domain"]["two_view"] and captured["domain"]["weight"] == 0
    main(["train", "--psm-main", "--config", str(ROOT / "configs/psm/isic2018.yaml"),
          "--epoch", "3", "--set", "training.t_max=7", "--run-dir", str(tmp_path)])
    assert captured["training"]["t_max"] == 7 and captured["model"]["variant"] == "psm_main"


def test_training_resume_evaluation_and_frozen_diagnostic(fixture_data, tmp_path):
    torch.set_num_threads(1)
    config = config_for()
    config["data"]["root"] = str(fixture_data)
    full, partial = tmp_path / "full", tmp_path / "partial"
    train(config, full, "cpu")
    train(config, partial, "cpu", stop_after_epoch=1)
    train(config, partial, "cpu", resume=partial / "last.pt")
    a, b = load_checkpoint(full / "last.pt"), load_checkpoint(partial / "last.pt")
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
    assert "train_covariance" in a["history"][0]
    report = evaluate_checkpoint(full / "best.pt", device="cpu", profile_iterations=1)
    assert {r["target"] for r in report["results"]} == {"isic2018", "ph2"}
    for row in report["results"]:
        assert all(k in row for k in ("params", "flops", "size_mb", "fps", "dice", "iou", "miou",
                                      "accuracy", "sensitivity", "specificity", "f1", "hd95", "empty_prediction_count"))
    control = select_experiment(config, ablation="psm_memory_aug")
    train(control, tmp_path / "control", "cpu")
    # Training objective must never be invoked by frozen diagnostics.
    with mock.patch("skinmamba.engine.training_objective", side_effect=AssertionError("Unexpected training")):
        diagnostic = diagnose_states(full / "best.pt", tmp_path / "control/best.pt", device="cpu", bootstrap=4,
                                     interventions=True)
        assert {r["target"] for r in diagnostic["distances"]} == {"pseudo"}
        assert {r["domain"] for r in diagnostic["segmentation"]} == {"source", "pseudo"}
        final = diagnose_states(full / "best.pt", tmp_path / "control/best.pt", device="cpu", bootstrap=4,
                                final=True, expected_ph2=2, interventions=True)
    assert {r["target"] for r in final["changes"]} == {"pseudo", "ph2"}
    assert final["protocol"]["target_fitting"] is False
    assert len(final["changes"]) == 6
    assert (full / "state_diagnostics_final/main/ph2_states.npz").exists()
    from scripts.summarize_psm import summarize
    summary = summarize(tmp_path, tmp_path / "summary")
    assert len(summary) == 3 and all(r["complete"] for r in summary)
    assert (tmp_path / "summary/state_changes_all.csv").exists()
    selected = next(r for r in summary if r["run"] == str(full.resolve()))
    assert selected["val_dice"] == a["history"][a["best_epoch"]]["val_dice"]
    with pytest.raises(ValueError, match="all 200"):
        diagnose_states(full / "best.pt", device="cpu", final=True, bootstrap=0)
