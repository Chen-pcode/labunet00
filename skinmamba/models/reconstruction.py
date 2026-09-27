"""State readback and decoder reconstruction experiments: unvalidated hypothesis.

HSMSSD adapted from EfficientViM, MIT, Copyright (c) 2024 MLVLab.
LocalAttender adapted from UPLiFT, MIT, Copyright (c) 2026 mwalmer-umd.
Pinned originals/licenses: third_party/{EfficientViM,UPLiFT}/.
See RECONSTRUCTION_EXPERIMENTS.md for attribution and proposed changes.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F
from .ultralight import UltraLight_VM_UNet, Spatial_Att_Bridge, Channel_Att_Bridge


class ChannelNorm(nn.Module):
    """Per-pixel LayerNorm over channels, with no spatial statistics mixing."""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.movedim(1, -1)).movedim(-1, 1)


class Conv1D(nn.Module):
    # Preserve upstream's .conv keys for exact core weight loading.
    def __init__(self, incoming, outgoing):
        super().__init__()
        self.conv = nn.Conv1d(incoming, outgoing, 1, bias=False)

    def forward(self, x):
        return self.conv(x)


class Conv2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)

    def forward(self, x):
        return self.conv(x)


class HSMSSD(nn.Module):
    """Author HSM-SSD equations; BCHW input and optional pre-mixer readback.

No scan extension or pretrained weights. A retains the author's spatially
constant offset inside a spatial softmax; it is not a Mamba-1 forgetting rate.
"""
    def __init__(self, d_model, state_dim=16, ssd_expand=1, readback=False):
        super().__init__()
        self.state_dim = state_dim
        self.d_inner = int(ssd_expand * d_model)
        self.readback = readback
        self.BCdt_proj = Conv1D(d_model, 3 * state_dim)
        self.dw = Conv2D(3 * state_dim)
        self.hz_proj = Conv1D(d_model, 2 * self.d_inner)
        self.out_proj = Conv1D(self.d_inner, d_model)
        self.A = nn.Parameter(torch.empty(state_dim).uniform_(1, 16))
        self.D = nn.Parameter(torch.ones(1))
        self.act = nn.SiLU()

    def forward(self, x):
        batch, channels, height, width = x.shape
        flat = x.flatten(2)
        bc_dt = self.dw(self.BCdt_proj(flat).reshape(batch, -1, height, width)).flatten(2)
        b, c, dt = bc_dt.chunk(3, dim=1)
        a = (dt + self.A[None, :, None]).softmax(-1)
        state = flat @ (a * b).transpose(-1, -2)
        h, z = self.hz_proj(state).chunk(2, dim=1)
        mixed = self.out_proj(h * self.act(z) + h * self.D)
        global_feature = (mixed @ c).reshape(batch, channels, height, width)
        readback = (state @ c).reshape_as(x) if self.readback else None
        return global_feature, readback

    def profiling_extra_flops(self, args, output):
        b, channels, h, w = args[0].shape
        # Conv children count separately: state write, mixed read, optional raw read.
        return {"state_matmul": 2 * b * channels * h * w * self.state_dim * (2 + int(self.readback))}


class LocalRefine(nn.Module):
    """Same-capacity conditional DWConv in full/difference/highpass controls."""
    def __init__(self, channels):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.context = nn.Conv2d(channels, channels, 1)
        self.project = nn.Conv2d(channels, channels, 1)

    def forward(self, value, context):
        return self.project(F.gelu(self.dw(value)) * self.context(context).sigmoid())


class ReadbackLayer(nn.Module):
    def __init__(self, incoming, outgoing, state_dim=16, local_input="difference", residual_init=.1):
        super().__init__()
        if local_input not in {"none", "full", "difference", "highpass"}:
            raise ValueError(f"Unknown core_local: {local_input}")
        self.local_input = local_input
        self.norm = ChannelNorm(incoming)
        self.mixer = HSMSSD(incoming, state_dim, readback=local_input != "none")
        if local_input != "none":
            self.read_norm = ChannelNorm(incoming)
            self.local = LocalRefine(incoming)
            self.beta = nn.Parameter(torch.full((1, incoming, 1, 1), residual_init))
        self.out_norm = ChannelNorm(incoming)
        self.proj = nn.Conv2d(incoming, outgoing, 1)
        self.capture_diagnostics = False
        self.diagnostics = {}

    def forward(self, x):
        u = self.norm(x)
        global_feature, readback = self.mixer(u)
        if self.local_input != "none":
            reconstructed = self.read_norm(readback)
            difference = u - reconstructed
            if self.local_input == "full":
                value = u
            elif self.local_input == "highpass":
                value = u - F.avg_pool2d(F.pad(u, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
            else:
                value = difference
            correction = self.beta * self.local(value, reconstructed)
            global_feature = global_feature + correction
            if self.capture_diagnostics:
                self.diagnostics = {"readback_difference": difference.detach().square().mean(1).sqrt(),
                                    "correction": correction.detach().square().mean(1).sqrt()}
        return self.proj(self.out_norm(u + global_feature))


class LocalAttender(nn.Module):
    """UPLiFT fixed 3x3 LocalAttender with conv_res=False and BCHW input.

Offset order, replication padding and neighborhood softmax match upstream.
Looped accumulation avoids materializing B*C*9*H_out*W_out. Supports rectangular
images and integer guide/value scale factors independently for H and W.
"""
    def __init__(self, guide_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(guide_channels, 9, 1)
        self.offsets = [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)]

    def forward(self, guide, value):
        b, channels, h, w = value.shape
        gh, gw = guide.shape[-2:]
        if guide.shape[0] != b or gh % h or gw % w or gh < h or gw < w:
            raise ValueError("LocalAttender guide must have an integer spatial multiple of value")
        sh, sw = gh // h, gw // w
        weights = self.conv1(guide).softmax(1).reshape(b, 9, h, sh, w, sw)
        padded = F.pad(value, (1, 1, 1, 1), mode="replicate")
        result = 0
        for k, (di, dj) in enumerate(self.offsets):
            neighbor = padded[:, :, 1 + di:1 + di + h, 1 + dj:1 + dj + w]
            result = result + neighbor[:, :, :, None, :, None] * weights[:, None, k]
        return result.reshape(b, channels, gh, gw)

    def profiling_extra_flops(self, args, output):
        return {"local_attender_aggregation": output.numel() * 17}  # 9 multiplies + 8 adds


class ChannelInteraction(nn.Module):
    """Decoder-conditioned CxC attention, inspired by DCA; not full DCA."""
    def __init__(self, channels, reduction=1):
        super().__init__()
        self.width = max(1, channels // reduction)
        self.query = nn.Conv2d(channels, self.width, 1, bias=False)
        self.key = nn.Conv2d(channels, self.width, 1, bias=False)
        self.value = nn.Conv2d(channels, self.width, 1, bias=False)
        self.project = nn.Conv2d(self.width, channels, 1, bias=False)
        self.log_temperature = nn.Parameter(torch.zeros(()))

    def forward(self, context, value):
        b, _, h, w = value.shape
        q, k, v = self.query(context).flatten(2), self.key(value).flatten(2), self.value(value).flatten(2)
        scale = (h * w) ** .25  # scale before accumulation to reduce fp16 overflow
        temperature = self.log_temperature.clamp(-4, 4).exp()
        relation = ((q / scale) @ (k / scale).transpose(-1, -2) / temperature).softmax(-1)
        return self.project((relation @ v).reshape(b, self.width, h, w))

    def profiling_extra_flops(self, args, output):
        b, _, h, w = args[0].shape
        return {"channel_attention_matmul": 4 * b * self.width ** 2 * h * w}


class ReconstructionBridge(nn.Module):
    # The legacy/legacy instance has no child modules; it only interpolates,
    # activates and adds, which are outside the common core FLOP scope.
    profiling_container_only = True
    def __init__(self, channels, spatial="guided", channel="cross", value_input="difference",
                 local_compensation=True, residual_init=.1, channel_reduction=1):
        super().__init__()
        self.spatial, self.channel, self.value_input = spatial, channel, value_input
        self.local_compensation = local_compensation and spatial == "guided"
        if spatial == "guided":
            self.guide = nn.Conv2d(2 * channels, channels, 1)
            self.attender = LocalAttender(channels)
        if spatial == "guided" or channel == "cross":
            self.align = ChannelNorm(channels)  # shared coordinates for subtraction
        if self.local_compensation:
            self.local = LocalRefine(channels)
            self.beta = nn.Parameter(torch.full((1, channels, 1, 1), residual_init))
        if channel == "cross":
            self.interaction = ChannelInteraction(channels, channel_reduction)
            self.gamma = nn.Parameter(torch.full((1, channels, 1, 1), residual_init))
        self.capture_diagnostics = False
        self.diagnostics = {}

    def forward(self, encoder, decoder, skip):
        # decoder is normalized but not activated, matching the baseline order.
        up = F.interpolate(decoder, size=encoder.shape[-2:], mode="bilinear", align_corners=True)
        if self.spatial == "guided":
            guide = F.gelu(self.guide(torch.cat((encoder, up), 1)))
            reconstructed = F.gelu(self.attender(guide, decoder))
        else:
            reconstructed = F.gelu(up)
        if self.spatial == "guided" or self.channel == "cross":
            context = self.align(reconstructed)
            difference = self.align(encoder) - context
            value = difference if self.value_input == "difference" else self.align(encoder)
        output = reconstructed + skip
        if self.local_compensation:
            output = output + self.beta * self.local(value, context)
        if self.channel == "cross":
            output = output + self.gamma * self.interaction(context, value)
        if self.capture_diagnostics and (self.spatial == "guided" or self.channel == "cross"):
            self.diagnostics = {"cross_layer_difference": difference.detach().square().mean(1).sqrt(),
                                "reconstruction": reconstructed.detach().square().mean(1).sqrt()}
        return output


class ReconstructionUNet(UltraLight_VM_UNet):
    """Same six-stage U-Net scaffold; independent core/bridge switches."""
    def __init__(self, options):
        channels = options.get("channels", [8, 16, 24, 32, 48, 64])
        local_input = options.get("core_local", "difference")
        state_dim = options.get("state_dim", 16)
        residual_init = options.get("residual_init", .1)
        spatial = options.get("spatial_bridge", "guided")
        channel = options.get("channel_bridge", "cross")
        value_input = options.get("bridge_input", "difference")
        if spatial not in {"legacy", "guided"} or channel not in {"legacy", "cross"}:
            raise ValueError("spatial_bridge/channel_bridge must be legacy|guided / legacy|cross")
        if value_input not in {"difference", "full"}:
            raise ValueError("bridge_input must be difference or full")
        if not isinstance(state_dim, int) or isinstance(state_dim, bool) or state_dim < 1:
            raise ValueError("state_dim must be a positive integer")
        if not math.isfinite(residual_init) or residual_init < 0:
            raise ValueError("residual_init must be finite and nonnegative")
        reduction = options.get("channel_reduction", 1)
        if not isinstance(reduction, int) or isinstance(reduction, bool) or reduction < 1:
            raise ValueError("channel_reduction must be a positive integer")

        def factory(incoming, outgoing, stage):
            return ReadbackLayer(incoming, outgoing, state_dim, local_input, residual_init)

        super().__init__(num_classes=options.get("num_classes", 1),
                         input_channels=options.get("input_channels", 3), c_list=channels,
                         bridge=False, backend="torch", pvm_factory=factory)
        self.spatial_mode, self.channel_mode = spatial, channel
        self.satt = Spatial_Att_Bridge() if spatial == "legacy" else None
        self.catt = Channel_Att_Bridge(channels, options.get("split_att", "fc")) if channel == "legacy" else None
        self.bridges = nn.ModuleList([
            ReconstructionBridge(c, spatial, channel, value_input,
                options.get("bridge_local", True), residual_init, reduction) for c in reversed(channels[:5])
        ])
        self.bridges.apply(self._init_weights)
        if self.satt is not None:
            self.satt.apply(self._init_weights)
        if self.catt is not None:
            self.catt.apply(self._init_weights)

    def sampling_report(self):
        return None

    def set_diagnostics(self, enabled=True):
        for module in self.modules():
            if isinstance(module, (ReadbackLayer, ReconstructionBridge)):
                module.capture_diagnostics = enabled
                module.diagnostics = {}

    def diagnostic_maps(self):
        return {f"{name}/{key}": value for name, module in self.named_modules()
                if isinstance(module, (ReadbackLayer, ReconstructionBridge))
                for key, value in module.diagnostics.items()}

    def forward(self, x):
        if x.ndim != 4 or min(x.shape[-2:]) < 32 or any(n % 32 for n in x.shape[-2:]):
            raise ValueError("Input must be BCHW with height and width positive multiples of 32")
        features = []
        out = x
        for i in range(1, 6):
            out = F.gelu(F.max_pool2d(getattr(self, f"ebn{i}")(getattr(self, f"encoder{i}")(out)), 2))
            features.append(out)
        spatial = ([a * e for a, e in zip(self.satt(*features), features)]
                   if self.satt is not None else [0] * 5)
        residual = [e + s for e, s in zip(features, spatial)]
        weights = self.catt(*residual) if self.catt is not None else [1] * 5
        skips = [c * r + s for c, r, s in zip(weights, residual, spatial)]
        out = F.gelu(self.encoder6(out))
        for i, (encoder, skip, bridge) in enumerate(zip(reversed(features), reversed(skips), self.bridges), 1):
            decoder = getattr(self, f"dbn{i}")(getattr(self, f"decoder{i}")(out))
            out = bridge(encoder, decoder, skip)
        return F.interpolate(self.final(out), size=x.shape[-2:], mode="bilinear", align_corners=True)
