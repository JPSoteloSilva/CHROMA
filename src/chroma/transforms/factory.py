from __future__ import annotations

from typing import Callable, Dict, Sequence

import kornia.color as Kcolor
from torch import Tensor, nn

from chroma.transforms.gradient_magnitude import GradientMagnitude
from chroma.transforms.inter_channel_correlation import InterChannelCorrelation
from chroma.transforms.noise_quant_restore import NoiseQuantRestore

TransformConfig = Dict[str, object]


class _KorniaWrapper(nn.Module):
    """Wrapper to make Kornia functions compatible with nn.Sequential."""

    def __init__(self, func: Callable[[Tensor], Tensor]) -> None:
        super().__init__()
        self.func = func

    def forward(self, x: Tensor) -> Tensor:
        return self.func(x)


# Registry mapping transform names to factory functions
_REGISTRY: dict[str, Callable[..., nn.Module]] = {
    # Direct Kornia functions wrapped for nn.Sequential
    "rgb_to_yuv": lambda: _KorniaWrapper(Kcolor.rgb_to_yuv),
    "rgb_to_hsv": lambda: _KorniaWrapper(Kcolor.rgb_to_hsv),
    "rgb_to_lab": lambda: _KorniaWrapper(Kcolor.rgb_to_lab),
    # Custom modules with unique logic
    "gradient_magnitude": GradientMagnitude,
    "inter_channel_correlation": InterChannelCorrelation,
    "noise_quant_restore": NoiseQuantRestore,
}


def build_transform_pipeline(configs: Sequence[TransformConfig]) -> nn.Sequential:
    """
    Build a nn.Sequential pipeline from configuration dictionaries.

    Follows Kornia best practices by using nn.Sequential for deterministic transforms.
    All transforms expect batched input of shape (B, C, H, W).

    Args:
        configs: List of transform configuration dictionaries, each with a "type" field
                and optional parameters for that transform.

    Returns:
        nn.Sequential pipeline of transforms

    Example:
        configs = [
            {"type": "rgb_to_yuv"},
            {"type": "gradient_magnitude", "operator": "Sobel", "normalization": "zscore"},
            {"type": "inter_channel_correlation", "window": 7}
        ]
        pipeline = build_transform_pipeline(configs)
        output = pipeline(input_tensor)  # input must be (B, C, H, W)
    """
    modules: list[nn.Module] = []
    for cfg in configs:
        if "type" not in cfg:
            raise ValueError("Each transform config must include a 'type' field.")
        name = str(cfg["type"]).lower()
        if name not in _REGISTRY:
            valid = ", ".join(sorted(_REGISTRY.keys()))
            raise ValueError(f"Unknown transform type '{name}'. Valid types: {valid}")

        factory = _REGISTRY[name]
        kwargs = {k: v for k, v in cfg.items() if k != "type"}
        modules.append(factory(**kwargs))

    return nn.Sequential(*modules)
