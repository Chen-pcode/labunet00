"""Correctness checks for Mamba math and the controlled sampling ablations."""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest import mock

import pytest
import torch
from torch.nn import functional as F

from skinmamba.models import build_model
from skinmamba.models.mamba import ReferenceMamba, mamba_forward, selective_scan_reference
from skinmamba.models.sampling import (make_sampling_plan, restore_features, sample_features,
                                       make_adaptive_sampling_plan, sample_adaptive_features,
                                       restore_adaptive_features)
from skinmamba.models.ultralight import PVMLayer


def settings(**updates):
    return {"backend": "reference", "channels": [8, 16, 24, 32, 48, 64], **updates}


def test_mamba_recurrence_uses_delta_for_decay_and_input():
    # Hand calculation: h1=2*.4=.8, h2=exp(-.3)*.8+.3*3.
    u = torch.tensor([[[2.0, 3.0]]], requires_grad=True)
    delta = torch.tensor([[[0.4, 0.3]]], requires_grad=True)
    y = selective_scan_reference(u, delta, torch.tensor([[-1.0]]),
                                 torch.ones(1, 1, 2), torch.ones(1, 1, 2))
    expected = torch.tensor([[[0.8, 0.8 * torch.exp(torch.tensor(-0.3)) + 0.9]]])
    torch.testing.assert_close(y, expected)
    y.sum().backward()
    assert u.grad is not None and delta.grad is not None
    assert torch.isfinite(u.grad).all() and torch.isfinite(delta.grad).all()


def test_delta_bias_softplus_scaling_order():
    module = ReferenceMamba(2, d_state=2, d_conv=2, expand=1)
    with torch.no_grad():
        module.dt_proj.bias.fill_(0.7)
    x = torch.randn(1, 4, 2)
    factors = torch.tensor([1.0, 0.5, 2.0, 1.0])
    captured = {}

    def capture(u, delta, *args, **kwargs):
        captured["delta"] = delta.detach()
        return selective_scan_reference(u, delta, *args, **kwargs)

    with mock.patch("skinmamba.models.mamba.selective_scan_reference", side_effect=capture):
        mamba_forward(module, x, factors)
    xz = module.in_proj(x).transpose(1, 2)
    values = F.silu(module.conv1d(xz.chunk(2, dim=1)[0])[..., :4])
    dt = module.x_proj(values.transpose(1, 2))[..., :module.dt_rank]
    raw = F.linear(dt, module.dt_proj.weight).transpose(1, 2)
    expected = F.softplus(raw + 0.7) * factors[None, None, :]
    torch.testing.assert_close(captured["delta"], expected)


def test_batched_delta_factors_scale_each_image_independently():
    module = ReferenceMamba(2, d_state=2, d_conv=2, expand=1)
    x = torch.randn(2, 4, 2)
    factors = torch.tensor([[1.0, .5, 1.5, 1.0], [1.0, 1.4, .6, 1.0]])
    together = mamba_forward(module, x, factors)
    separate = torch.cat([mamba_forward(module, x[i:i + 1], factors[i]) for i in range(2)])
    torch.testing.assert_close(together, separate)
    with pytest.raises(ValueError, match="Batched delta_factors"):
        mamba_forward(module, x, factors[:1])


def test_fixed_distance_units_and_row_seams():
    plan = make_sampling_plan(3, 9, 0.5, 1.5)
    x = plan["x"]
    assert torch.all(x[1:] > x[:-1])
    steps = plan["geometry"].reshape(3, -1)
    torch.testing.assert_close(steps[:, 1:].sum(dim=1), torch.full((3,), 8.0))
    torch.testing.assert_close(steps[:, 0], torch.ones(3))
    torch.testing.assert_close(plan["constant"].reshape(3, -1)[:, 1:],
                               torch.full((3, x.numel() - 1), 8 / (x.numel() - 1)))
    assert plan["row_transition_count"] == 2
    assert not torch.allclose(plan["geometry"], plan["constant"])
    # Units do not change with sampling density: row distance still equals W-1.
    dense = make_sampling_plan(3, 9, 0.75, 1.5)
    assert dense["geometry"].reshape(3, -1)[0, 1:].sum().item() == pytest.approx(8.0)


def test_nonuniform_interpolation_recovers_linear_spatial_signal():
    plan = make_sampling_plan(3, 9, 0.5, 1.5)
    image = (torch.arange(9)[None, :] + 10 * torch.arange(3)[:, None]).float()[None, None]
    torch.testing.assert_close(restore_features(sample_features(image, plan), plan), image,
                               rtol=1e-5, atol=2e-6)


def test_adaptive_plan_covers_borders_and_backpropagates():
    scores = torch.ones(1, 1, 2, 32, requires_grad=True)
    with torch.no_grad():
        scores[..., 10:15] = 20
    plan = make_adaptive_sampling_plan(scores, .75, adaptive_lambda=.95,
        coverage=True, min_spacing=.25 * 31 / 23, max_spacing=2.5 * 31 / 23,
        delta_bounds=(.5, 1.5))
    x = plan["x"]
    assert x.shape == (1, 2, 24)
    torch.testing.assert_close(x[..., 0], torch.zeros(1, 2))
    torch.testing.assert_close(x[..., -1], torch.full((1, 2), 31.0))
    steps = x[..., 1:] - x[..., :-1]
    assert steps.min() >= .25 * 31 / 23 - 1e-5
    assert steps.max() <= 2.5 * 31 / 23 + 1e-5
    assert plan["geometry"].min() >= .5 - 1e-6
    assert plan["geometry"].max() <= 1.5 + 1e-6
    features = torch.randn(1, 3, 2, 32, requires_grad=True)
    restored = restore_adaptive_features(sample_adaptive_features(features, plan), plan)
    restored.square().mean().backward()
    assert scores.grad is not None and scores.grad.abs().sum() > 0
    assert features.grad is not None and torch.isfinite(features.grad).all()


def test_adaptive_sampling_has_uniform_limit_and_distinct_constraints():
    scores = torch.rand(1, 1, 2, 32)
    uniform = make_adaptive_sampling_plan(scores, .75, adaptive_lambda=0)
    torch.testing.assert_close(uniform["x"][0, 0], torch.linspace(0, 31, 24), atol=1e-5, rtol=1e-5)
    concentrated = torch.ones(1, 1, 2, 32)
    concentrated[..., 10:15] = 1000
    raw = make_adaptive_sampling_plan(concentrated, .75, adaptive_lambda=.95)
    covered = make_adaptive_sampling_plan(concentrated, .75, adaptive_lambda=.95,
        coverage=True, min_spacing=.25 * 31 / 23, max_spacing=2.5 * 31 / 23)
    bounded = make_adaptive_sampling_plan(concentrated, .75, adaptive_lambda=.95,
        coverage=True, min_spacing=.25 * 31 / 23, max_spacing=2.5 * 31 / 23,
        delta_bounds=(.5, 1.5))
    assert not torch.allclose(raw["x"], covered["x"])
    torch.testing.assert_close(covered["x"], bounded["x"])
    assert not torch.allclose(covered["geometry"], bounded["geometry"])


def test_adaptive_width_one_is_identity_sampling():
    image = torch.randn(1, 3, 1, 1)
    plan = make_adaptive_sampling_plan(torch.ones(1, 1, 1, 1), .75, coverage=True)
    torch.testing.assert_close(restore_adaptive_features(sample_adaptive_features(image, plan), plan), image)
    assert plan["sample_count"] == 1


def test_adaptive_layer_score_head_and_ssm_receive_gradients():
    layer = PVMLayer(8, 8, d_state=2, d_conv=2, backend="reference",
                     variant="cclas", sample_ratio=.75)
    output = layer(torch.randn(1, 8, 3, 8))
    output.square().mean().backward()
    assert layer.score_head[0].weight.grad is not None
    assert layer.score_head[0].weight.grad.abs().sum() > 0
    assert layer.mamba.A_log.grad is not None
    report = layer.sampling_report()
    assert report["samples_per_row"] == 6
    assert report["coverage"] and report["bounded_delta"]


@pytest.mark.parametrize("variant", ["sampled_index", "sampled_constant", "sampled_geometry"])
def test_full_uniform_sampling_matches_baseline(variant):
    torch.manual_seed(11)
    baseline = PVMLayer(8, 8, d_state=3, d_conv=2, backend="reference")
    sampled = PVMLayer(8, 8, d_state=3, d_conv=2, backend="reference",
                       variant=variant, sample_ratio=1, sampling_power=1)
    sampled.load_state_dict(baseline.state_dict(), strict=True)
    x = torch.randn(2, 8, 3, 5)
    torch.testing.assert_close(sampled(x), baseline(x), atol=2e-6, rtol=2e-5)


def test_sampling_variants_keep_parameters_coordinates_and_have_gradients():
    torch.manual_seed(17)
    variants = [PVMLayer(8, 8, d_state=3, d_conv=2, backend="reference",
                         variant=v, sample_ratio=0.5, sampling_power=1.5)
                for v in ("sampled_index", "sampled_constant", "sampled_geometry")]
    for module in variants[1:]:
        module.load_state_dict(variants[0].state_dict(), strict=True)
    x = torch.randn(1, 8, 3, 9, requires_grad=True)
    outputs = [m(x) for m in variants]
    for module, output in zip(variants, outputs):
        output.square().mean().backward(retain_graph=True)
        for name in ("A_log", "x_proj.weight", "dt_proj.weight", "conv1d.weight"):
            gradient = dict(module.mamba.named_parameters())[name].grad
            assert gradient is not None and torch.isfinite(gradient).all()
            assert gradient.abs().sum() > 0
        assert module.sampling_report()["sample_x"] == variants[0].sampling_report()["sample_x"]
    assert torch.isfinite(x.grad).all()
    assert not torch.allclose(outputs[0], outputs[1], atol=1e-7, rtol=1e-6)
    assert not torch.allclose(outputs[1], outputs[2], atol=1e-7, rtol=1e-6)


def test_preserves_unsampled_residual():
    layer = PVMLayer(8, 8, d_state=2, backend="reference", variant="sampled_geometry", sample_ratio=0.5)
    x = torch.randn(1, 8, 3, 7)
    with mock.patch("skinmamba.models.ultralight.mamba_forward", side_effect=lambda module, values, *args: torch.zeros_like(values)):
        actual = layer(x)
    flat = x.flatten(2).transpose(1, 2)
    expected = layer.proj(layer.norm(layer.skip_scale * layer.norm(flat))).transpose(1, 2).reshape_as(x)
    torch.testing.assert_close(actual, expected)


def test_unfused_control_matches_full_grid_baseline():
    torch.manual_seed(19)
    baseline = PVMLayer(8, 8, d_state=3, backend="reference")
    control = PVMLayer(8, 8, d_state=3, backend="reference", variant="unfused_control")
    control.load_state_dict(baseline.state_dict(), strict=True)
    x = torch.randn(1, 8, 3, 5)
    torch.testing.assert_close(control(x), baseline(x), rtol=0, atol=0)
    assert control.sampling_report() is None
    assert control.profiling_scan_spec((x,))["length"] == 15


def test_baseline_returns_logits_and_network_backward():
    model = build_model({"model": settings(d_state=2, variant="sampled_geometry", sample_ratio=0.5)})
    x = torch.randn(1, 3, 32, 32)
    output = model(x)
    assert output.shape == (1, 1, 32, 32)
    F.binary_cross_entropy_with_logits(output, torch.zeros_like(output)).backward()
    assert torch.isfinite(model.encoder4[0].mamba.A_log.grad).all()
    # Setting a large final bias demonstrates the API does not apply sigmoid.
    with torch.no_grad():
        model.final.weight.zero_()
        model.final.bias.fill_(4)
        torch.testing.assert_close(model(x), torch.full_like(output, 4))


def test_reference_baseline_matches_original_author_topology_and_initialization():
    original = Path(__file__).resolve().parents[1] / "third_party" / "UltraLight_VM_UNet.original.py"
    if not original.exists():
        pytest.skip("Optional author source comparison requires the adjacent original repository.")
    # Load unchanged author source while replacing only its optional imports.
    timm = types.ModuleType("timm")
    timm_models = types.ModuleType("timm.models")
    layers = types.ModuleType("timm.models.layers")
    layers.trunc_normal_ = torch.nn.init.trunc_normal_
    mamba_package = types.ModuleType("mamba_ssm")
    mamba_package.Mamba = ReferenceMamba
    with mock.patch.dict(sys.modules, {"timm": timm, "timm.models": timm_models,
                                      "timm.models.layers": layers, "mamba_ssm": mamba_package}):
        spec = importlib.util.spec_from_file_location("original_ultralight_for_test", original)
        source = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(source)
        torch.manual_seed(31)
        original_model = source.UltraLight_VM_UNet()
    torch.manual_seed(31)
    adapted = build_model(settings())
    old_state, new_state = original_model.state_dict(), adapted.state_dict()
    assert list(old_state) == list(new_state)
    for key in old_state:
        torch.testing.assert_close(old_state[key], new_state[key], rtol=0, atol=0)
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(original_model(x), adapted(x).sigmoid(), rtol=1e-5, atol=1e-6)


def test_cuda_does_not_silently_fall_back():
    with mock.patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError, match="CUDA is unavailable"):
            build_model(settings(backend="cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA parity requires a CUDA GPU.")
def test_official_cuda_reference_parity():
    mamba_ssm = pytest.importorskip("mamba_ssm")
    torch.manual_seed(44)
    cuda = mamba_ssm.Mamba(d_model=4, d_state=4, d_conv=3, expand=2).cuda()
    reference = ReferenceMamba(4, 4, 3, 2).cuda()
    reference.load_state_dict(cuda.state_dict(), strict=True)
    x = torch.randn(2, 7, 4, device="cuda")
    factors = torch.tensor([1, .5, 1.5, 1, .7, .8, 1.5], device="cuda")
    torch.testing.assert_close(cuda(x), reference(x), atol=3e-5, rtol=3e-4)
    torch.testing.assert_close(mamba_forward(cuda, x, factors, "cuda"),
                               mamba_forward(reference, x, factors), atol=3e-5, rtol=3e-4)
