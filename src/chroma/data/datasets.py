from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

# Common image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def _collect_images_from_directory(directory: Union[str, Path]) -> List[str]:
    """Recursively collect all image paths from a directory."""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {directory}")

    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(str(Path(root) / file))

    return sorted(image_paths)


def collect_image_paths(path_or_paths: Union[str, List[str]]) -> List[str]:
    """
    Collect image paths from one or more directories (recursively).

    Args:
        path_or_paths: A single directory path (str) or list of directory paths

    Returns:
        List of absolute image paths
    """
    if isinstance(path_or_paths, str):
        path_or_paths = [path_or_paths]

    all_paths = []
    for path in path_or_paths:
        path_obj = Path(path)

        if not path_obj.exists():
            raise FileNotFoundError(f"Path does not exist: {path_obj}")

        if path_obj.is_dir():
            # Collect all images from directory
            all_paths.extend(_collect_images_from_directory(path_obj))
        else:
            raise ValueError(f"Non-directory path: {path_obj}. Only directories are supported.")

    return all_paths


class ImagePathsDataset(Dataset):
    """
    Dataset that reads images from a list of (path, label) and returns
    (rgb: Tensor, label, path).

    This dataset expects a composed torchvision transform to be passed
    externally (e.g., via the datamodule) to apply cropping/augmentation.
    """

    def __init__(
        self,
        items: Sequence[Tuple[str, int]],
        transform: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> None:
        super().__init__()
        self.items = list(items)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def _load_image(self, path: str) -> Tensor:
        with Image.open(path) as im:
            im = im.convert("RGB")
        if self.transform is not None:
            return self.transform(im)
        raise RuntimeError(
            "ImagePathsDataset requires a transform (e.g., including ToTensor); none was provided."
        )

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        rgb = self._load_image(path)
        return rgb, int(label), path


def build_items_from_lists(train_real: List[str], train_gen: List[str]) -> List[Tuple[str, int]]:
    items: List[Tuple[str, int]] = []
    items.extend([(p, 0) for p in train_real])
    items.extend([(p, 1) for p in train_gen])
    return items
