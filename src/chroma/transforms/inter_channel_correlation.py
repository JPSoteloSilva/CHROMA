from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

PairOrder = Literal["sequential", "upper"]


class InterChannelCorrelation(nn.Module):
    """
    Compute pairwise Pearson correlation between the three input channels over a
    sliding window. The output consists of three channels ordered as:

        0: correlation(channel_0, channel_1)
        1: correlation(channel_1, channel_2)
        2: correlation(channel_0, channel_2)

    Expects batched input of shape (B, C, H, W) following Kornia conventions.

    Args:
        window: Odd kernel size for local correlation.
        stride: Optional stride before upsampling back to the input resolution.
        eps: Numerical stability term for variance/standard deviation.
    """

    def __init__(self, window: int = 7, stride: int = 1, eps: float = 1e-3) -> None:
        super().__init__()
        if window % 2 == 0:
            raise ValueError("window must be odd")
        self.window = int(window)
        self.stride = int(stride)
        self.eps = eps

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            Correlation maps of shape (B, 3, H, W)
        """
        b, _, h, w = x.shape
        k = self.window
        s = max(1, self.stride)

        # Box blur using avg_pool2d with reflect padding
        def blur(t: Tensor) -> Tensor:
            pad = k // 2
            t_padded = F.pad(t, (pad, pad, pad, pad), mode="reflect")
            return F.avg_pool2d(t_padded, kernel_size=k, stride=1)

        pairs = [(0, 1), (1, 2), (0, 2)]
        outputs: list[Tensor] = []

        for i, j in pairs:
            a = x[:, i : i + 1]
            b_ = x[:, j : j + 1]
            mu_a = blur(a)
            mu_b = blur(b_)
            mu_ab = blur(a * b_)

            var_a = blur(a * a) - mu_a * mu_a
            var_b = blur(b_ * b_) - mu_b * mu_b
            cov_ab = mu_ab - mu_a * mu_b

            denom = (
                torch.sqrt(var_a.clamp_min(self.eps)) * torch.sqrt(var_b.clamp_min(self.eps))
                + self.eps
            )
            corr = cov_ab / denom
            outputs.append(corr.clamp(-1.0, 1.0))

        out = torch.cat(outputs, dim=1)

        if s != 1:
            out = F.avg_pool2d(out, kernel_size=s, stride=s)
            out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)

        return out
