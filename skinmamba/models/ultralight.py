"""UltraLight VM-UNet adaptation; see REFERENCES_MODEL.md for provenance.

Baseline module names, topology and outer initialization follow the MIT source.
The only baseline output change is returning logits rather than sigmoid values.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .mamba import create_mamba, mamba_forward
from .sampling import (make_sampling_plan, restore_features, sample_features,
                       make_adaptive_sampling_plan, sample_adaptive_features,
                       restore_adaptive_features)


ADAPTIVE_VARIANTS = {"adaptive_index", "adaptive_geometry", "adaptive_coverage", "cclas"}


class PVMLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2,
                 groups=4, backend="cuda", variant="baseline", sample_ratio=1.0,
                 sampling_power=1.5, adaptive_lambda=0.75,
                 coverage_min_factor=0.25, coverage_max_factor=2.5,
                 delta_min=0.5, delta_max=1.5):
        super().__init__()
        if input_dim % groups:
            raise ValueError("PVM input channels must be divisible by groups.")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = create_mamba(input_dim // groups, d_state, d_conv, expand, backend)
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))
        self.groups = groups
        self.backend = backend
        self.variant = variant
        self.sample_ratio = sample_ratio
        self.sampling_power = sampling_power
        self.adaptive_lambda = adaptive_lambda
        self.coverage_min_factor = coverage_min_factor
        self.coverage_max_factor = coverage_max_factor
        self.delta_min = delta_min
        self.delta_max = delta_max
        if variant in ADAPTIVE_VARIANTS:
            self.score_head = nn.Sequential(
                nn.Conv2d(input_dim, input_dim, 3, padding=1, groups=input_dim),
                nn.GELU(), nn.Conv2d(input_dim, 1, 1), nn.Sigmoid(),
            )
        self._sampling_plans = {}
        self._last_plan = None

    def _plan(self, height, width, device):
        key = (height, width, str(device))
        if key not in self._sampling_plans:
            self._sampling_plans[key] = make_sampling_plan(
                height, width, self.sample_ratio, self.sampling_power, device=device,
            )
        self._last_plan = self._sampling_plans[key]
        return self._last_plan

    def sampling_report(self):
        if self._last_plan is None:
            return None
        plan = self._last_plan
        if plan.get("adaptive"):
            steps = plan["x"][..., 1:] - plan["x"][..., :-1]
            sample_x = plan["x"][0, 0].detach().cpu().tolist()
        else:
            steps = plan["geometry"].reshape(plan["height"], plan["sample_count"])[0, 1:]
            sample_x = plan["x"].detach().cpu().tolist()
        return {
            "variant": self.variant, "height": plan["height"], "width": plan["width"],
            "samples_per_row": plan["sample_count"],
            "original_tokens": plan["height"] * plan["width"],
            "sampled_tokens": plan["height"] * plan["sample_count"],
            "distance_unit": "one pixel of the original stage feature grid",
            "row_transition_count": plan["row_transition_count"],
            "row_transition_delta_factor": 1.0, "initial_token_delta_factor": 1.0,
            "min_within_row_step": float(steps.min().detach().cpu()) if steps.numel() else None,
            "max_within_row_step": float(steps.max().detach().cpu()) if steps.numel() else None,
            "sample_x": sample_x,
            "sample_x_scope": "first image, first row" if plan.get("adaptive") else "all rows",
            "adaptive_lambda": self.adaptive_lambda if plan.get("adaptive") else None,
            "coverage": self.variant in {"adaptive_coverage", "cclas"},
            "bounded_delta": self.variant == "cclas",
        }

    def profiling_scan_spec(self, inputs, output=None):
        """Describe all shared-core calls; profiler must skip the .mamba subtree.

        Official CUDA fast paths bypass child module hooks, so count the complete
        Mamba core analytically from this specification instead of mixing hooks.
        """
        x = inputs[0] if isinstance(inputs, tuple) else inputs
        height, width = x.shape[-2:]
        count = width if self.variant in {"baseline", "unfused_control"} else min(width, max(2, round(width * self.sample_ratio)))
        return {
            "batch": x.shape[0], "length": height * count,
            "channels": self.mamba.d_inner, "state_size": self.mamba.d_state,
            "dt_rank": self.mamba.dt_rank, "groups": self.groups,
            "d_model": self.mamba.d_model, "d_conv": self.mamba.d_conv,
            "sampled": self.variant not in {"baseline", "unfused_control"},
            "delta_scaled": self.variant not in {"baseline", "unfused_control", "sampled_index", "adaptive_index"},
        }

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.float()
        batch, channels, height, width = x.shape
        if channels != self.input_dim:
            raise ValueError("Unexpected PVM input channel count.")
        x_flat = x.reshape(batch, channels, height * width).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        if self.variant in {"baseline", "unfused_control"}:
            outputs = [
                (self.mamba(part) if self.variant == "baseline" else
                 mamba_forward(self.mamba, part, backend=self.backend)) + self.skip_scale * part
                for part in torch.chunk(x_norm, self.groups, dim=2)
            ]
            x_mamba = torch.cat(outputs, dim=2)
        else:
            normalized_image = x_norm.transpose(1, 2).reshape(batch, channels, height, width)
            if self.variant in ADAPTIVE_VARIANTS:
                count = min(width, max(2, round(width * self.sample_ratio)))
                average = (width - 1) / (count - 1) if count > 1 else 1.0
                plan = make_adaptive_sampling_plan(
                    self.score_head(normalized_image), self.sample_ratio,
                    adaptive_lambda=self.adaptive_lambda,
                    coverage=self.variant in {"adaptive_coverage", "cclas"},
                    min_spacing=self.coverage_min_factor * average,
                    max_spacing=self.coverage_max_factor * average,
                    delta_bounds=(self.delta_min, self.delta_max) if self.variant == "cclas" else None,
                )
                sampled = sample_adaptive_features(normalized_image, plan)
                self._last_plan = {key: value.detach() if isinstance(value, torch.Tensor) else value
                                   for key, value in plan.items()}
            else:
                plan = self._plan(height, width, x.device)
                sampled = sample_features(normalized_image, plan)
            sequence = sampled.flatten(2).transpose(1, 2)
            factor_key = {"sampled_index": "index", "sampled_constant": "constant",
                          "sampled_geometry": "geometry", "adaptive_index": None,
                          "adaptive_geometry": "geometry", "adaptive_coverage": "geometry",
                          "cclas": "geometry"}[self.variant]
            # All sampled variants reuse the same core projections and kernels.
            factors = None if factor_key is None else plan[factor_key]
            outputs = [mamba_forward(self.mamba, part, factors, self.backend)
                       for part in torch.chunk(sequence, self.groups, dim=2)]
            processed = torch.cat(outputs, dim=2).transpose(1, 2).reshape(
                batch, channels, height, plan["sample_count"],
            )
            restore = restore_adaptive_features if self.variant in ADAPTIVE_VARIANTS else restore_features
            restored = restore(processed, plan).flatten(2).transpose(1, 2)
            # Keep the original full-grid residual; do not interpolate the skip path.
            x_mamba = restored + self.skip_scale * x_norm
        x_mamba = self.proj(self.norm(x_mamba))
        return x_mamba.transpose(-1, -2).reshape(batch, self.output_dim, height, width)


class Channel_Att_Bridge(nn.Module):
    def __init__(self, c_list, split_att="fc"):
        super().__init__()
        c_list_sum = sum(c_list) - c_list[-1]
        self.split_att = split_att
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.get_all_att = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        for i in range(5):
            setattr(self, f"att{i + 1}", nn.Linear(c_list_sum, c_list[i]) if split_att == "fc"
                    else nn.Conv1d(c_list_sum, c_list[i], 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, t1, t2, t3, t4, t5):
        tensors = (t1, t2, t3, t4, t5)
        att = torch.cat([self.avgpool(t) for t in tensors], dim=1)
        att = self.get_all_att(att.squeeze(-1).transpose(-1, -2))
        if self.split_att != "fc":
            att = att.transpose(-1, -2)
        outputs = []
        for i, t in enumerate(tensors, 1):
            value = self.sigmoid(getattr(self, f"att{i}")(att))
            if self.split_att == "fc":
                value = value.transpose(-1, -2)
            outputs.append(value.unsqueeze(-1).expand_as(t))
        return tuple(outputs)


class Spatial_Att_Bridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_conv2d = nn.Sequential(
            nn.Conv2d(2, 1, 7, stride=1, padding=9, dilation=3), nn.Sigmoid(),
        )

    def forward(self, t1, t2, t3, t4, t5):
        return tuple(self.shared_conv2d(torch.cat([t.mean(dim=1, keepdim=True),
                                                t.max(dim=1, keepdim=True).values], dim=1))
                     for t in (t1, t2, t3, t4, t5))


class SC_Att_Bridge(nn.Module):
    def __init__(self, c_list, split_att="fc"):
        super().__init__()
        self.catt = Channel_Att_Bridge(c_list, split_att)
        self.satt = Spatial_Att_Bridge()

    def forward(self, t1, t2, t3, t4, t5):
        inputs = (t1, t2, t3, t4, t5)
        spatial = tuple(a * t for a, t in zip(self.satt(*inputs), inputs))
        residual = tuple(s + t for s, t in zip(spatial, inputs))
        channel = self.catt(*residual)
        return tuple(a * r + s for a, r, s in zip(channel, residual, spatial))


class UltraLight_VM_UNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, c_list=None,
                 split_att="fc", bridge=True, groups=4, backend="cuda",
                 variant="baseline", sample_ratio=1.0, sampling_power=1.5,
                 geometry_stage="encoder4", d_state=16, d_conv=4, expand=2,
                 adaptive_lambda=0.75, coverage_min_factor=0.25,
                 coverage_max_factor=2.5, delta_min=0.5, delta_max=1.5):
        super().__init__()
        c_list = list(c_list or [8, 16, 24, 32, 48, 64])
        if groups <= 0 or d_state <= 0 or d_conv <= 0 or expand <= 0:
            raise ValueError("groups, d_state, d_conv and expand must be positive.")
        if len(c_list) != 6 or any(c <= 0 or c % 4 or c % groups for c in c_list):
            raise ValueError("channels must contain six positive multiples of 4 and groups.")
        if variant not in {"baseline", "unfused_control", "sampled_index", "sampled_constant", "sampled_geometry", *ADAPTIVE_VARIANTS}:
            raise ValueError(f"Unknown variant: {variant}")
        stages = {"encoder4", "encoder5", "encoder6", "decoder1", "decoder2", "decoder3"}
        if geometry_stage not in stages:
            raise ValueError(f"geometry_stage must be one of {sorted(stages)}")
        if not (0 < sample_ratio <= 1) or sampling_power <= 0 or not math.isfinite(sampling_power):
            raise ValueError("Invalid sampling ratio or power.")
        if not 0 <= adaptive_lambda <= 1 or not math.isfinite(adaptive_lambda):
            raise ValueError("adaptive_lambda must be finite and in [0, 1].")
        if not 0 < coverage_min_factor < 1 < coverage_max_factor:
            raise ValueError("coverage factors must satisfy 0 < min < 1 < max.")
        if not 0 < delta_min < 1 < delta_max:
            raise ValueError("delta bounds must satisfy 0 < min < 1 < max.")
        self.bridge = bridge
        self.backend = backend
        self.variant = variant
        self.geometry_stage = geometry_stage

        def pvm(input_dim, output_dim, stage):
            return PVMLayer(input_dim, output_dim, d_state, d_conv, expand, groups, backend,
                            variant if stage == geometry_stage else "baseline",
                            sample_ratio, sampling_power, adaptive_lambda,
                            coverage_min_factor, coverage_max_factor, delta_min, delta_max)

        self.encoder1 = nn.Sequential(nn.Conv2d(input_channels, c_list[0], 3, padding=1))
        self.encoder2 = nn.Sequential(nn.Conv2d(c_list[0], c_list[1], 3, padding=1))
        self.encoder3 = nn.Sequential(nn.Conv2d(c_list[1], c_list[2], 3, padding=1))
        self.encoder4 = nn.Sequential(pvm(c_list[2], c_list[3], "encoder4"))
        self.encoder5 = nn.Sequential(pvm(c_list[3], c_list[4], "encoder5"))
        self.encoder6 = nn.Sequential(pvm(c_list[4], c_list[5], "encoder6"))
        if bridge:
            self.scab = SC_Att_Bridge(c_list, split_att)
        self.decoder1 = nn.Sequential(pvm(c_list[5], c_list[4], "decoder1"))
        self.decoder2 = nn.Sequential(pvm(c_list[4], c_list[3], "decoder2"))
        self.decoder3 = nn.Sequential(pvm(c_list[3], c_list[2], "decoder3"))
        self.decoder4 = nn.Sequential(nn.Conv2d(c_list[2], c_list[1], 3, padding=1))
        self.decoder5 = nn.Sequential(nn.Conv2d(c_list[1], c_list[0], 3, padding=1))
        for i in range(5):
            setattr(self, f"ebn{i + 1}", nn.GroupNorm(4, c_list[i]))
        for i, channels in enumerate(reversed(c_list[:5]), 1):
            setattr(self, f"dbn{i}", nn.GroupNorm(4, channels))
        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv1d):
            n = module.kernel_size[0] * module.out_channels
            module.weight.data.normal_(0, math.sqrt(2.0 / n))
        elif isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels // module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()

    def sampling_report(self):
        report = getattr(self, self.geometry_stage)[0].sampling_report()
        return {"stage": self.geometry_stage, **report} if report else None

    def forward(self, x):
        if x.ndim != 4 or min(x.shape[-2:]) < 32 or any(size % 32 for size in x.shape[-2:]):
            raise ValueError("Input must be BCHW with height and width positive multiples of 32.")
        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out
        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out
        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out
        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2))
        t4 = out
        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))
        t5 = out
        if self.bridge:
            t1, t2, t3, t4, t5 = self.scab(t1, t2, t3, t4, t5)
        out = F.gelu(self.encoder6(out))
        out5 = F.gelu(self.dbn1(self.decoder1(out))) + t5
        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)), scale_factor=2, mode="bilinear", align_corners=True)) + t4
        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)), scale_factor=2, mode="bilinear", align_corners=True)) + t3
        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)), scale_factor=2, mode="bilinear", align_corners=True)) + t2
        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)), scale_factor=2, mode="bilinear", align_corners=True)) + t1
        return F.interpolate(self.final(out1), scale_factor=2, mode="bilinear", align_corners=True)
