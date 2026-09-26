import pytest
import torch
from torch import nn

from skinmamba.profiling import profile_model


def _profile(model, shape=(1, 3, 4, 4), **kwargs):
    return profile_model(model, input_shape=shape, device="cpu", warmup=0, iterations=2, **kwargs)


def test_conv_linear_count_uses_two_flops_per_mac_and_measures_cpu_only():
    model = nn.Sequential(nn.Conv2d(3, 2, 1, bias=False), nn.Flatten(), nn.Linear(32, 4, bias=False))
    result = _profile(model)
    assert result["flops"] == 2 * (4 * 4 * 2 * 3 + 4 * 32)
    assert result["params"] == 3 * 2 + 32 * 4
    assert result["size_mb"] == result["params"] * 4 / 2**20
    assert result["flops_total"] is None and result["flops_complete"] is False
    assert result["fps"] > 0 and result["latency_mean_ms"] > 0
    assert result["latency_p95_ms"] >= result["latency_p50_ms"]
    assert result["device"] == "cpu" and result["peak_memory_mb"] is None


def test_profiling_preserves_individual_training_flags_and_dtype():
    model = nn.Sequential(nn.Conv2d(3, 2, 1), nn.BatchNorm2d(2), nn.Dropout())
    model.train()
    model[1].eval()
    modes = [module.training for module in model.modules()]
    running_mean = model[1].running_mean.clone()
    _profile(model)
    assert [module.training for module in model.modules()] == modes
    assert torch.equal(model[1].running_mean, running_mean)
    assert all(p.dtype == torch.float32 and p.device.type == "cpu" for p in model.parameters())


def test_unknown_state_space_layer_never_reports_an_incomplete_total_as_flops():
    class UnknownMamba(nn.Module):
        def forward(self, x):
            return torch.cumsum(x, dim=-1)

    result = _profile(nn.Sequential(nn.Conv2d(3, 3, 1), UnknownMamba()))
    assert result["flops"] is None
    assert result["flops_known_subtotal"] == 2 * 1 * 3 * 4 * 4 * 3
    assert any("UnknownMamba" in item for item in result["flops_unsupported"])


def test_scan_protocol_counts_functional_core_and_does_not_double_count_children():
    class ScanWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.mamba = nn.Linear(3, 3, bias=False)

        def forward(self, x):
            return self.mamba(x)

        def profiling_scan_spec(self, inputs, output):
            return {"batch": inputs[0].shape[0], "length": inputs[0].shape[1],
                    "channels": 6, "state_size": 2, "dt_rank": 1, "groups": 1,
                    "d_model": 3, "d_conv": 4, "delta_scaled": True}

    result = _profile(ScanWrapper(), (1, 5, 3))
    linear = 2 * 5 * (3 * 12 + 6 * (1 + 4) + 1 * 6 + 6 * 3)
    conv = 2 * 5 * 6 * 4
    scan = 5 * 6 * (8 * 2 + 6)
    assert result["flops"] == linear + conv + scan
    assert result["flops_breakdown"]["selective_scan"] == scan
    assert "linear" not in result["flops_breakdown"]
    assert result["flops_excluded_modules"]


def test_failed_forward_removes_hooks_and_restores_modes():
    class Broken(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 3, 1)

        def forward(self, x):
            self.conv(x)
            raise RuntimeError("deliberate test failure")

    model = Broken().train()
    with pytest.raises(RuntimeError, match="deliberate"):
        _profile(model)
    assert model.training and model.conv.training
    assert not model.conv._forward_hooks


@pytest.mark.parametrize("kwargs", [{"iterations": 0}, {"warmup": -1}, {"precision": "int8"}, {"precision": "fp16"}, {"input_shape": (1, 0, 4, 4)}])
def test_invalid_requests_are_rejected(kwargs):
    defaults = {"device": "cpu", "warmup": 0, "iterations": 1}
    defaults.update(kwargs)
    with pytest.raises(ValueError):
        profile_model(nn.Conv2d(3, 2, 1), **defaults)


def test_cuda_request_never_silently_falls_back_to_cpu():
    if torch.cuda.is_available():
        pytest.skip("This check targets hosts without CUDA.")
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        profile_model(nn.Conv2d(3, 2, 1), warmup=0, iterations=1)


def test_amp_alias_uses_autocast_without_mutating_fp32_parameters():
    model = nn.Conv2d(3, 2, 1)
    result = _profile(model, precision="amp_bf16")
    assert result["precision"] == "bf16"
    assert model.weight.dtype == torch.float32
