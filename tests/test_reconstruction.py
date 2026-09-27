"""Source parity, mechanism controls, checkpoint compatibility and core counts."""
from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from skinmamba.config import load_config, select_experiment
from skinmamba.experiments import RECONSTRUCTION_PRESETS
from skinmamba.models import build_model
from skinmamba.models.reconstruction import HSMSSD, LocalAttender, ReadbackLayer, ChannelInteraction
from skinmamba.profiling import profile_model

ROOT = Path(__file__).resolve().parents[1]


def author_definitions(path, names, namespace):
    """Execute only selected, unchanged AST definitions from pinned snapshots.

Avoid unrelated backbone imports, pretrained downloads and timm/fvcore deps.
"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    assert {node.name for node in selected} == set(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)


def test_hsm_core_output_and_gradient_match_pinned_author_code():
    torch.set_num_threads(1)
    scope = dict(torch=torch, nn=nn, F=F, math=math)
    folder = ROOT / "third_party/EfficientViM"
    author_definitions(folder / "utils.original.py", ["ConvLayer1D", "ConvLayer2D"], scope)
    author_definitions(folder / "EfficientViM.original.py", ["HSMSSD"], scope)
    original = scope["HSMSSD"](d_model=8, state_dim=4)
    ours = HSMSSD(8, state_dim=4, readback=True)
    ours.load_state_dict(original.state_dict(), strict=True)
    x = torch.randn(2, 8, 4, 4, requires_grad=True)
    ref_input = x.detach().flatten(2).requires_grad_()
    expected, _ = original(ref_input)
    actual, readback = ours(x)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad.flatten(2), ref_input.grad, rtol=1e-5, atol=1e-7)
    for name, parameter in ours.named_parameters():
        torch.testing.assert_close(parameter.grad, dict(original.named_parameters())[name].grad, rtol=2e-5, atol=1e-6)
    assert readback.shape == x.shape


@pytest.mark.parametrize("shape,scale", [((2, 5, 3, 4), 1), ((1, 3, 2, 3), 2)])
def test_attender_output_and_gradients_match_upstream(shape, scale):
    torch.set_num_threads(1)
    scope = dict(torch=torch, nn=nn, F=F, np=np)
    author_definitions(ROOT / "third_party/UPLiFT/uplift.original.py", ["LocalAttender", "residual_auto"], scope)
    original = scope["LocalAttender"](in_channels=7, num_connected=9, conv_res=False)
    ours = LocalAttender(7)
    ours.load_state_dict(original.state_dict(), strict=True)
    value = torch.randn(shape, requires_grad=True)
    guide = torch.randn(shape[0], 7, shape[2] * scale, shape[3] * scale, requires_grad=True)
    value_ref, guide_ref = value.detach().requires_grad_(), guide.detach().requires_grad_()
    actual, expected = ours(guide, value), original(guide_ref, value_ref)
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(value.grad, value_ref.grad)
    torch.testing.assert_close(guide.grad, guide_ref.grad, atol=2e-6, rtol=2e-5)


def test_attender_preserves_constant_values_and_checks_scale():
    module = LocalAttender(3)
    values = torch.full((1, 4, 2, 3), 2.5)
    torch.testing.assert_close(module(torch.randn(1, 3, 4, 6), values), torch.full((1, 4, 4, 6), 2.5))
    with pytest.raises(ValueError, match="integer"):
        module(torch.randn(1, 3, 5, 6), values)


@pytest.mark.parametrize("preset", list(RECONSTRUCTION_PRESETS))
def test_every_preset_rectangular_forward_backward_profile_and_reload(preset):
    torch.set_num_threads(1)
    config = select_experiment(load_config(ROOT / "configs/smoke_cpu.yaml"), ablation=preset)
    config["model"]["state_dim"] = 4
    model = build_model(config)
    x = torch.randn(2, 3, 32, 64)
    logits = model(x)
    assert logits.shape == (2, 1, 32, 64)
    F.binary_cross_entropy_with_logits(logits, torch.rand_like(logits)).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    restored = build_model(config)
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(restored(x), logits)
    profile = profile_model(model, (1, 3, 32, 64), "cpu", warmup=0, iterations=1)
    assert profile["flops"] > 0 and profile["flops_unsupported"] == []
    assert profile["flops_breakdown"]["state_matmul"] > 0
    if config["model"]["spatial_bridge"] == "guided":
        assert profile["flops_breakdown"]["local_attender_aggregation"] > 0
    if config["model"]["channel_bridge"] == "cross":
        assert profile["flops_breakdown"]["channel_attention_matmul"] > 0


def test_full_difference_highpass_controls_match_capacity_and_change_computation():
    layers = [ReadbackLayer(8, 12, 4, mode) for mode in ("full", "difference", "highpass")]
    for layer in layers[1:]:
        layer.load_state_dict(layers[0].state_dict())
    assert len({sum(p.numel() for p in layer.parameters()) for layer in layers}) == 1
    x = torch.randn(2, 8, 4, 6)
    outputs = [layer(x) for layer in layers]
    assert not torch.allclose(outputs[0], outputs[1])
    assert not torch.allclose(outputs[1], outputs[2])
    for layer in layers:
        layer.beta.data.zero_()
    torch.testing.assert_close(layers[0](x), layers[1](x))
    torch.testing.assert_close(layers[1](x), layers[2](x))


def test_legacy_bridge_shell_preserves_original_bridge_equations():
    config = select_experiment(load_config(ROOT / "configs/smoke_cpu.yaml"), ablation="readback")
    model = build_model(config).eval()
    from skinmamba.models.ultralight import SC_Att_Bridge
    original = SC_Att_Bridge(config["model"]["channels"])
    original.satt.load_state_dict(model.satt.state_dict())
    original.catt.load_state_dict(model.catt.state_dict())
    features = [torch.randn(2, c, 32 // 2**i, 32 // 2**i)
                for i, c in enumerate(config["model"]["channels"][:5])]
    spatial = [a * e for a, e in zip(model.satt(*features), features)]
    residual = [e + s for e, s in zip(features, spatial)]
    weights = model.catt(*residual)
    for got, expected in zip([c * r + s for c, r, s in zip(weights, residual, spatial)], original(*features)):
        torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_custom_functional_flops_are_counted_once_with_conv_children():
    core = HSMSSD(4, state_dim=3, readback=True)
    x = torch.randn(2, 4, 2, 3)
    assert core.profiling_extra_flops((x,), None) == {"state_matmul": 3 * 2 * 2 * 4 * 6 * 3}
    channel = ChannelInteraction(4, reduction=2)
    assert channel.profiling_extra_flops((x, x), None) == {"channel_attention_matmul": 4 * 2 * 2**2 * 6}
    profile = profile_model(core, tuple(x.shape), "cpu", warmup=0, iterations=1)
    expected_conv = 2 * 2 * (6 * 4 * 9 + 6 * 9 * 9 + 3 * 4 * 8 + 3 * 4 * 4)
    assert profile["flops"] == expected_conv + 3 * 2 * 2 * 4 * 6 * 3


def test_main_switch_and_old_cclas_are_explicit():
    config = load_config(ROOT / "configs/main_isic2018.yaml")
    assert config["model"]["family"] == "reconstruction"
    assert select_experiment(config, experiment="main")["model"]["variant"] == "reconstruction"
    baseline = select_experiment(config, experiment="baseline")
    assert baseline["model"]["family"] == "ultralight"
    assert select_experiment(config, ablation="cclas")["model"]["variant"] == "cclas"


def test_diagnostics_do_not_change_logits_or_keep_autograd_graphs():
    model = build_model(load_config(ROOT / "configs/main_isic2018.yaml")).eval()
    x = torch.randn(1, 3, 32, 32)
    expected = model(x)
    model.set_diagnostics(True)
    actual = model(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert model.diagnostic_maps()
    assert all(not value.requires_grad for value in model.diagnostic_maps().values())
    model.set_diagnostics(False)
    assert model.diagnostic_maps() == {}
