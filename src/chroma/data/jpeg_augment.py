from __future__ import annotations

import io
import random
from typing import Tuple, Union

import torch.nn as nn
from PIL import Image
from torch import Tensor
from torchvision.transforms import functional as TF


class RandomJPEGCompression(nn.Module):
    """Torchvision-compatible module that applies random JPEG compression.

    This transform can be inserted into torchvision pipelines (before
    ``ToTensor()``). It accepts both PIL images and tensors, converts them to
    RGB PIL images for compression, and restores the original type after
    compression. When the `.probability` check fails, the original input is
    returned unchanged.
    """

    def __init__(
        self,
        quality_min: int = 70,
        quality_max: int = 95,
        probability: float = 0.5,
    ) -> None:
        super().__init__()
        if not (1 <= quality_min <= 100 and 1 <= quality_max <= 100):
            raise ValueError(
                f"JPEG quality range must be within [1, 100], got "
                f"quality_min={quality_min}, quality_max={quality_max}"
            )
        if quality_min > quality_max:
            raise ValueError(
                f"quality_min must be <= quality_max, got "
                f"quality_min={quality_min}, quality_max={quality_max}"
            )
        if not (0.0 <= probability <= 1.0):
            raise ValueError(f"probability must be in [0, 1], got {probability}")

        self.quality_range: Tuple[int, int] = (int(quality_min), int(quality_max))
        self.probability: float = float(probability)

    def forward(self, img: Union[Image.Image, Tensor]) -> Union[Image.Image, Tensor]:
        """Apply random JPEG compression to the image or tensor."""
        if self.probability < 1.0 and random.random() > self.probability:
            return img

        if isinstance(img, Tensor):
            pil_img = TF.to_pil_image(img.clamp(0.0, 1.0))
            was_tensor = True
        else:
            pil_img = img
            was_tensor = False

        quality = random.randint(self.quality_range[0], self.quality_range[1])

        buffer = io.BytesIO()
        img_rgb = pil_img.convert("RGB")
        img_rgb.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)

        compressed = Image.open(buffer).convert("RGB")
        buffer.close()

        if was_tensor:
            return TF.to_tensor(compressed)
        return compressed

    def __repr__(self) -> str:
        qmin, qmax = self.quality_range
        return (
            f"{self.__class__.__name__}(quality_min={qmin}, "
            f"quality_max={qmax}, probability={self.probability})"
        )
