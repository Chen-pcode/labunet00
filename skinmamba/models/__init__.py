"""Model factory for explicit CUDA production and CPU reference backends."""
from .ultralight import UltraLight_VM_UNet


def build_model(config: dict):
    options = config.get("model", config)
    return UltraLight_VM_UNet(
        num_classes=options.get("num_classes", 1),
        input_channels=options.get("input_channels", 3),
        c_list=options.get("channels", [8, 16, 24, 32, 48, 64]),
        split_att=options.get("split_att", "fc"),
        bridge=options.get("bridge", True), groups=options.get("groups", 4),
        backend=options.get("backend", "cuda"),
        variant=options.get("variant", "baseline"),
        sample_ratio=options.get("sample_ratio", 1.0),
        sampling_power=options.get("sampling_power", 1.5),
        geometry_stage=options.get("geometry_stage", "encoder4"),
        d_state=options.get("d_state", 16), d_conv=options.get("d_conv", 4),
        expand=options.get("expand", 2),
        adaptive_lambda=options.get("adaptive_lambda", 0.75),
        coverage_min_factor=options.get("coverage_min_factor", 0.25),
        coverage_max_factor=options.get("coverage_max_factor", 2.5),
        delta_min=options.get("delta_min", 0.5),
        delta_max=options.get("delta_max", 1.5),
    )


__all__ = ["build_model", "UltraLight_VM_UNet"]
