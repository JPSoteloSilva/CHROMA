from __future__ import annotations

from typing import Literal

import kornia.filters as Kfilters
import torch
from torch import Tensor, nn

Operator = Literal["Sobel", "Scharr"]
Normalization = Literal["none", "zscore"]


class GradientMagnitude(nn.Module):
    """
    Compute gradient magnitudes for each channel independently, returning a
    three-channel tensor.

    Expects batched input of shape (B, C, H, W) following Kornia conventions.

    Args:
        operator: Kernel to use for spatial gradients ("Sobel" or "Scharr").
        normalization: Optional per-channel z-score normalization ("none" or "zscore").
        eps: Numerical stability term.
    """

    def __init__(
        self,
        operator: Operator = "Sobel",
        normalization: Normalization = "zscore",
        eps: float = 1e-3,
    ) -> None:
        super().__init__()
        kernel = "scharr" if operator.lower() == "scharr" else "sobel"
        self.gradient = Kfilters.SpatialGradient(mode=kernel, order=1, normalized=False)
        self.normalization = normalization
        self.eps = eps

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            Gradient magnitudes of shape (B, 3, H, W)
        """
        grads = self.gradient(x)  # (B, 3, 2, H, W)
        dx = grads[:, :, 0]
        dy = grads[:, :, 1]
        out = torch.sqrt(dx * dx + dy * dy + self.eps)

        if self.normalization == "zscore":
            mean = out.mean(dim=(2, 3), keepdim=True)
            std = out.std(dim=(2, 3), keepdim=True).clamp_min(self.eps)
            out = (out - mean) / std

        return out
