from __future__ import annotations

import torch
from torch import Tensor, nn


class NoiseQuantRestore(nn.Module):
    """
    K-fold noise-quantization-restoration pipeline to reveal compression artifacts.

    This matches the approach in secret_lies_in_color.py's color_quant_restore():
    1. Generate K noisy+quantized copies of the image
    2. Average them to get a "restored" version
    3. Compute difference: |original - restored|

    The averaging acts as denoising, and the difference reveals artifacts
    that are more pronounced in generated images.

    Expects batched input of shape (B, 3, H, W) in [0,1] following Kornia conventions.
    Returns difference maps of shape (B, 3, H, W).
    """

    def __init__(
        self,
        K: int = 8,
        sigma: float = 0.01,
        num_levels: int = 16,
        seed: int | None = None,
    ) -> None:
        """
        Args:
            K: Number of noisy+quantized copies to generate and average
            sigma: Standard deviation of Gaussian noise in [0, 1] range
            num_levels: Number of quantization levels (e.g., 16)
            seed: Optional fixed seed for deterministic noise (useful for inference)
        """
        super().__init__()
        if K < 1:
            raise ValueError("K must be >= 1")
        if num_levels < 2:
            raise ValueError("num_levels must be >= 2")

        self.K = int(K)
        self.sigma = float(sigma)
        self.num_levels = int(num_levels)
        self.seed = seed

    def _quantize(self, x: Tensor) -> Tensor:
        """Uniform quantization: round(x * (L-1)) / (L-1)"""
        L = self.num_levels
        return torch.round(x * (L - 1)) / (L - 1)

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Image in [0,1], shape (B, 3, H, W)

        Returns:
            Difference maps of shape (B, 3, H, W)
        """
        b, c, h, w = x.shape
        device, dtype = x.device, x.dtype

        # Generate K noisy + quantized copies (vectorized for speed)
        # Shape: (b, K, c, h, w)
        quantized_clean = self._quantize(x)

        if self.seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.seed)
            noise = torch.randn(b, self.K, c, h, w, device=device, dtype=dtype, generator=generator)
        else:
            noise = torch.randn(b, self.K, c, h, w, device=device, dtype=dtype)
        noise = noise * self.sigma
        x_expanded = x.unsqueeze(1)  # (b, 1, c, h, w)

        # Add noise and clamp
        noisy = (x_expanded + noise).clamp(0.0, 1.0)

        # Quantize each copy
        quantized = self._quantize(noisy)  # (b, K, c, h, w)
        quantized_all = torch.cat([quantized_clean.unsqueeze(1), quantized], dim=1)

        # Restore by averaging over K dimension
        x_restored = quantized_all.mean(dim=1)  # (b, c, h, w)

        diff = torch.abs(x - x_restored)
        return diff
