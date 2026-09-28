"""Explicit per-image latent memory over baseline PVM outputs.

This is NOT the internal selective-scan hidden state. The author PVM/SAB/CAB
remain intact; persistent means across encoder scales within ONE forward.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .ultralight import UltraLight_VM_UNet


class StateUpdate(nn.Module):
    def __init__(self, channels, dim):
        super().__init__()
        self.query = nn.Linear(channels, dim)
        self.gate = nn.Linear(2 * dim, dim)

    def forward(self, feature, previous):
        query = torch.tanh(self.query(feature.mean((2, 3))))
        gate = torch.sigmoid(self.gate(torch.cat((query, previous), dim=-1)))
        state = gate * previous + (1 - gate) * query
        return state, query

    def profiling_extra_flops(self, inputs, output):
        feature, previous = inputs
        # Pooling, tanh and sigmoid are excluded like other baseline activations.
        return {"memory_update": int(4 * previous.numel())}


class StateRead(nn.Module):
    def __init__(self, dim, channels, strength):
        super().__init__()
        self.affine = nn.Linear(dim, 2 * channels)
        self.strength = strength
        nn.init.normal_(self.affine.weight, std=.01)
        nn.init.zeros_(self.affine.bias)

    def forward(self, feature, state):
        gamma, beta = (self.strength * torch.tanh(self.affine(state))).chunk(2, -1)
        return feature * (1 + gamma[..., None, None]) + beta[..., None, None]

    def profiling_extra_flops(self, inputs, output):
        feature, state = inputs
        return {"memory_read": int(2 * feature.numel() + 3 * feature.shape[0] * feature.shape[1])}


class PersistentUNet(UltraLight_VM_UNet):
    def __init__(self, options):
        channels = options.get("channels", [8, 16, 24, 32, 48, 64])
        super().__init__(
            num_classes=options.get("num_classes", 1), input_channels=options.get("input_channels", 3),
            c_list=channels, split_att=options.get("split_att", "fc"), bridge=options.get("bridge", True),
            groups=options.get("groups", 4), backend=options.get("backend", "cuda"), variant="baseline",
            d_state=options.get("d_state", 16), d_conv=options.get("d_conv", 4), expand=options.get("expand", 2),
        )
        self.variant = options.get("variant", "psm_main")
        self.memory_mode = options.get("memory_mode", "persistent")
        self.memory_dim = options.get("memory_dim", 16)
        if self.memory_mode not in {"persistent", "independent"} or self.memory_dim < 2:
            raise ValueError("memory_mode must be persistent/independent and memory_dim >= 2")
        self.writers = nn.ModuleList(StateUpdate(c, self.memory_dim) for c in channels[3:6])
        self.readers = nn.ModuleList(StateRead(self.memory_dim, c, options.get("memory_strength", .1))
                                     for c in reversed(channels[2:5]))

    def forward(self, x, return_states=False, intervention="none"):
        if x.ndim != 4 or min(x.shape[-2:]) < 32 or any(s % 32 for s in x.shape[-2:]):
            raise ValueError("Input must be BCHW with height/width multiples of 32")
        if intervention not in {"none", "no_carry", "no_read"}:
            raise ValueError("intervention must be none/no_carry/no_read")
        t1 = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t2 = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(t1)), 2, 2))
        t3 = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(t2)), 2, 2))
        t4 = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(t3)), 2, 2))
        t5 = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(t4)), 2, 2))
        out = F.gelu(self.encoder6(t5))
        # Local variable: never carried between images, calls, views, or batches.
        previous = out.new_zeros((x.shape[0], self.memory_dim))
        states, queries = [], []
        for writer, feature in zip(self.writers, (t4, t5, out)):
            if self.memory_mode == "independent" or intervention == "no_carry":
                previous = torch.zeros_like(previous)
            previous, query = writer(feature, previous)
            states.append(previous)
            queries.append(query)
        if self.bridge:
            t1, t2, t3, t4, t5 = self.scab(t1, t2, t3, t4, t5)

        def read(index, feature):
            return feature if intervention == "no_read" else self.readers[index](feature, states[2 - index])

        out5 = read(0, F.gelu(self.dbn1(self.decoder1(out)))) + t5
        out4 = read(1, F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)), scale_factor=2,
                                           mode="bilinear", align_corners=True))) + t4
        out3 = read(2, F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)), scale_factor=2,
                                           mode="bilinear", align_corners=True))) + t3
        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)), scale_factor=2,
                                    mode="bilinear", align_corners=True)) + t2
        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)), scale_factor=2,
                                    mode="bilinear", align_corners=True)) + t1
        logits = F.interpolate(self.final(out1), scale_factor=2, mode="bilinear", align_corners=True)
        return (logits, {"states": tuple(states), "queries": tuple(queries)}) if return_states else logits
