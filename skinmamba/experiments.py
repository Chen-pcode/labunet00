"""Versioned reconstruction presets. Settings, not performance claims."""

RECONSTRUCTION_DEFAULTS = {
    "family": "reconstruction", "state_dim": 16, "core_local": "difference",
    "spatial_bridge": "guided", "channel_bridge": "cross",
    "bridge_input": "difference", "bridge_local": True,
    "residual_init": .1, "channel_reduction": 1,
}

RECONSTRUCTION_PRESETS = {
    "hsm_only": dict(core_local="none", spatial_bridge="legacy", channel_bridge="legacy"),
    "hsm_local": dict(core_local="full", spatial_bridge="legacy", channel_bridge="legacy"),
    "hsm_highpass": dict(core_local="highpass", spatial_bridge="legacy", channel_bridge="legacy"),
    "readback": dict(spatial_bridge="legacy", channel_bridge="legacy"),
    "readback_spatial": dict(channel_bridge="legacy"),
    "readback_channel": dict(spatial_bridge="legacy"),
    "reconstruction": {},
    "bridge_full": dict(bridge_input="full"),
    "all_full": dict(core_local="full", bridge_input="full"),
    "direct_transfer": dict(core_local="none", bridge_input="full", bridge_local=False),
    "no_core_compensation": dict(core_local="none"),
    "no_bridge_local": dict(bridge_local=False),
}


def reconstruction_options(name):
    return {**RECONSTRUCTION_DEFAULTS, **RECONSTRUCTION_PRESETS[name], "variant": name}
