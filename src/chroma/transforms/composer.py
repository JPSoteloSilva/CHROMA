from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import torch
from torch import Tensor, nn

from chroma.transforms.factory import build_transform_pipeline

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class PipelineSpec:
    """Configuration description for a feature pipeline."""

    name: str
    transforms: List[Dict[str, Any]]
    standardize: Optional[str] = None  # e.g. "zscore"


class BatchFeatureComposer(nn.Module):
    """
    Compose batched feature volumes from configurable transform pipelines.

    Parameters
    ----------
    include_rgb:
        If ``True`` (default), the original RGB tensor is included as the first
        chunk in the output volume.
    rgb_standardize:
        Optional mode applied to the RGB chunk when included. Supported values:
        ``None`` (raw [0,1]), ``"imagenet"`` (mean/std normalization) or
        ``"zscore"`` (per-sample z-score).
    pipelines:
        Iterable of pipeline specifications. Each pipeline is built using the
        existing transform factory and applied independently to the RGB batch.
    """

    def __init__(
        self,
        *,
        include_rgb: bool = True,
        rgb_standardize: Optional[str] = None,
        pipelines: Optional[Iterable[Dict[str, Any]]] = None,
    ) -> None:
        super().__init__()
        self.include_rgb = include_rgb

        if rgb_standardize is not None:
            norm_mode = str(rgb_standardize).lower()
            if norm_mode not in {"imagenet", "zscore", "none"}:
                raise ValueError("rgb_standardize must be one of {'imagenet', 'zscore', 'none'}")
            self.rgb_standardize = None if norm_mode == "none" else norm_mode
        else:
            self.rgb_standardize = None

        parsed_specs = [self._parse_pipeline_spec(cfg) for cfg in (pipelines or [])]
        self._pipeline_order: List[str] = [spec.name for spec in parsed_specs]
        modules = {spec.name: build_transform_pipeline(spec.transforms) for spec in parsed_specs}
        self.pipelines = nn.ModuleDict(modules)
        self._pipeline_specs: Dict[str, PipelineSpec] = {spec.name: spec for spec in parsed_specs}

        if self.include_rgb and self.rgb_standardize == "imagenet":
            mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
            std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
            self.register_buffer("_rgb_mean", mean, persistent=False)
            self.register_buffer("_rgb_std", std, persistent=False)
        else:
            self.register_buffer("_rgb_mean", None, persistent=False)
            self.register_buffer("_rgb_std", None, persistent=False)

        self._channel_slices: Dict[str, slice] = {}
        self._output_channels: Optional[int] = None

    @staticmethod
    def _parse_pipeline_spec(cfg: Dict[str, Any]) -> PipelineSpec:
        if "name" not in cfg:
            raise ValueError("Each feature pipeline config must include a 'name' field")
        if "transforms" not in cfg:
            raise ValueError(f"Pipeline '{cfg['name']}' is missing 'transforms' list")
        transforms = cfg["transforms"]
        if not isinstance(transforms, list) or not transforms:
            raise ValueError(f"Pipeline '{cfg['name']}' must provide a non-empty transforms list")
        standardize = cfg.get("standardize")
        if standardize is not None:
            standardize = str(standardize).lower()
            if standardize not in {"none", "zscore"}:
                raise ValueError(
                    f"Unsupported standardize='{standardize}' for pipeline '{cfg['name']}'. "
                    "Supported: 'none', 'zscore'"
                )
            if standardize == "none":
                standardize = None
        return PipelineSpec(name=str(cfg["name"]), transforms=transforms, standardize=standardize)

    @property
    def output_channels(self) -> Optional[int]:
        """Return the number of channels produced on the last forward pass."""

        return self._output_channels

    @property
    def channel_slices(self) -> Dict[str, slice]:
        """Return the channel slice mapping for the last forward pass."""

        return dict(self._channel_slices)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(
                f"BatchFeatureComposer expects input of shape (B, 3, H, W); "
                f"received {tuple(x.shape)}"
            )

        chunks: List[Tensor] = []
        slices: Dict[str, slice] = {}
        offset = 0

        if self.include_rgb:
            rgb_chunk = x
            if self.rgb_standardize == "imagenet":
                mean = self._rgb_mean.to(device=x.device, dtype=x.dtype)
                std = self._rgb_std.to(device=x.device, dtype=x.dtype)
                rgb_chunk = (x - mean) / std
            elif self.rgb_standardize == "zscore":
                rgb_chunk = self._zscore(x)
            chunks.append(rgb_chunk)
            slices["rgb"] = slice(offset, offset + rgb_chunk.shape[1])
            offset += rgb_chunk.shape[1]

        for name in self._pipeline_order:
            module = self.pipelines[name]
            y = module(x)
            if y.ndim != 4:
                raise ValueError(
                    f"Pipeline '{name}' must return a batched tensor of shape "
                    f"(B, C, H, W); got {tuple(y.shape)}"
                )
            if self._pipeline_specs[name].standardize == "zscore":
                y = self._zscore(y)
            chunks.append(y)
            slices[name] = slice(offset, offset + y.shape[1])
            offset += y.shape[1]

        if not chunks:
            raise ValueError(
                "BatchFeatureComposer has no outputs configured "
                "(include_rgb=False and no pipelines)"
            )

        volume = torch.cat(chunks, dim=1)

        self._channel_slices = slices
        self._output_channels = volume.shape[1]
        return volume

    @staticmethod
    def _zscore(t: Tensor, eps: float = 1e-3) -> Tensor:
        mean = t.mean(dim=(2, 3), keepdim=True)
        std = t.std(dim=(2, 3), keepdim=True).clamp_min(eps)
        return (t - mean) / std

    def infer_output_channels(
        self, height: int, width: int, device: Optional[torch.device] = None
    ) -> int:
        """Run a dummy forwarding pass to determine output channels."""

        dummy = torch.zeros(1, 3, height, width, device=device or torch.device("cpu"))
        with torch.no_grad():
            out = self.forward(dummy)
        return out.shape[1]
