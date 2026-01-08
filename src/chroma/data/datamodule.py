from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from lightning.pytorch import LightningDataModule
from loguru import logger
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.transforms import InterpolationMode

from chroma.data.datasets import ImagePathsDataset, build_items_from_lists, collect_image_paths
from chroma.data.jpeg_augment import RandomJPEGCompression


class RandomScaleResize(nn.Module):
    """Downscale an image, then resize it back to the target crop size."""

    def __init__(
        self,
        crop_size: int,
        scale_min: float = 0.8,
        scale_max: float = 1.0,
        probability: float = 1.0,
        interpolation: str | InterpolationMode = "bilinear",
    ) -> None:
        super().__init__()
        if crop_size <= 0:
            raise ValueError("crop_size must be positive")
        self.crop_size = int(crop_size)

        low = float(scale_min)
        high = float(scale_max)
        if low <= 0 or high <= 0 or low > high or high > 1.0:
            raise ValueError("scale_min/scale_max must satisfy 0 < scale_min <= scale_max <= 1")
        self.scale_min = low
        self.scale_max = high

        if not (0.0 <= probability <= 1.0):
            raise ValueError("probability must be between 0 and 1")
        self.probability = float(probability)

        if isinstance(interpolation, InterpolationMode):
            self.interpolation = interpolation
        else:
            key = str(interpolation).replace("-", "_").upper()
            try:
                self.interpolation = InterpolationMode[key]
            except KeyError:
                self.interpolation = InterpolationMode.BILINEAR

    def forward(self, img: Image.Image) -> Image.Image:
        if self.probability < 1.0 and random.random() > self.probability:
            return img

        scale = random.uniform(self.scale_min, self.scale_max)
        if scale >= 0.999 or self.scale_min == self.scale_max == 1.0:
            return img

        down_size = max(1, int(round(self.crop_size * scale)))
        if down_size == self.crop_size:
            return img

        resized = TF.resize(img, down_size, interpolation=self.interpolation)
        return TF.resize(resized, self.crop_size, interpolation=self.interpolation)

    def __repr__(self) -> str:
        interp_str = (
            self.interpolation.name.lower()
            if isinstance(self.interpolation, InterpolationMode)
            else str(self.interpolation)
        )
        return (
            f"{self.__class__.__name__}(crop_size={self.crop_size}, "
            f"scale_min={self.scale_min}, scale_max={self.scale_max}, "
            f"probability={self.probability}, interpolation={interp_str})"
        )


class ImageListsDataModule(LightningDataModule):
    _train_ds: Optional[ImagePathsDataset]
    _val_ds: Optional[ImagePathsDataset]
    _sampler: Optional[WeightedRandomSampler]

    def __init__(
        self,
        train_real_paths: Union[str, List[str]],
        train_gen_paths: Union[str, List[str]],
        val_real_paths: Union[str, List[str]],
        val_gen_paths: Union[str, List[str]],
        heldout_real_paths: Optional[Union[str, List[str]]] = None,
        heldout_gen_paths: Optional[Union[str, List[str]]] = None,
        test_real_paths: Optional[Union[str, List[str]]] = None,
        test_gen_paths: Optional[Union[str, List[str]]] = None,
        batch_size: int = 64,
        num_workers: int = 8,
        prefetch_factor: Optional[int] = None,
        drop_last: bool = True,
        use_weighted_sampler: bool = True,
        sampling_strategy: str = "class",
        # image
        crop_size: int = 224,
        pad_mode: str = "reflect",
        augment: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.train_real_paths = train_real_paths
        self.train_gen_paths = train_gen_paths
        self.val_real_paths = val_real_paths
        self.val_gen_paths = val_gen_paths
        self.heldout_real_paths = heldout_real_paths
        self.heldout_gen_paths = heldout_gen_paths
        self.test_real_paths = test_real_paths
        self.test_gen_paths = test_gen_paths
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.drop_last = drop_last
        self.use_weighted_sampler = use_weighted_sampler
        allowed_strategies = {"none", "class", "source", "class_source"}
        requested_strategy = str(sampling_strategy).lower()
        if requested_strategy not in allowed_strategies:
            logger.warning(
                "Unknown sampling_strategy '%s'. Falling back to 'class'.", requested_strategy
            )
            requested_strategy = "class"
        self.sampling_strategy = requested_strategy if use_weighted_sampler else "none"
        self.crop_size = crop_size
        self.pad_mode = pad_mode
        self.augment = augment or {}

        self._train_ds: Optional[ImagePathsDataset] = None
        self._val_ds: Optional[ImagePathsDataset] = None
        self._heldout_ds: Optional[ImagePathsDataset] = None
        self._test_ds: Optional[ImagePathsDataset] = None
        self._sampler: Optional[WeightedRandomSampler] = None
        self._train_sources: Optional[List[str]] = None

    def _collect_paths_with_sources(
        self, path_or_paths: Union[str, List[str]], prefix: str
    ) -> tuple[List[str], List[str]]:
        if isinstance(path_or_paths, str):
            path_list = [path_or_paths]
        else:
            path_list = list(path_or_paths)

        all_paths: List[str] = []
        source_ids: List[str] = []

        for idx, entry in enumerate(path_list):
            images = collect_image_paths(entry)
            source_name = f"{prefix}:{Path(entry).name}:{idx}"
            all_paths.extend(images)
            source_ids.extend([source_name] * len(images))

        return all_paths, source_ids

    def _maybe_build_sampler(self, train_items: List[tuple[str, int]]) -> None:
        """Configure a WeightedRandomSampler according to the sampling strategy."""
        if self.sampling_strategy == "none" or not train_items:
            self._sampler = None
            return

        labels = torch.tensor([lbl for _, lbl in train_items], dtype=torch.long)
        weights = torch.ones(len(train_items), dtype=torch.float)

        if self.sampling_strategy in {"class", "class_source"}:
            class_counts = torch.bincount(labels, minlength=2).float()
            class_counts[class_counts == 0] = 1.0
            class_weights = 1.0 / class_counts
            weights *= class_weights[labels]

        if self.sampling_strategy in {"source", "class_source"}:
            if not self._train_sources or len(self._train_sources) != len(train_items):
                raise ValueError(
                    "sampling_strategy='source' requires source metadata for each training item."
                )
            source_counts = Counter(self._train_sources)
            source_weight_vec = torch.tensor(
                [1.0 / source_counts[src] for src in self._train_sources], dtype=torch.float
            )
            weights *= source_weight_vec

        self._sampler = WeightedRandomSampler(
            weights=weights.tolist(),
            num_samples=len(train_items),
            replacement=True,
        )

    def prepare_data(self) -> None:
        # Validation will happen in setup() when we actually collect paths
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        # Build torchvision transforms
        # Note: We keep images in [0,1] range for forensic feature extraction.
        # RGB normalization happens in the model's FeatureComposer via rgb_standardize.

        # RandomCrop with pad_if_needed handles small images automatically
        # Keep transforms on PIL images until the final ToTensor() call.
        train_tfms: List[T.Transform] = [
            T.RandomCrop(self.crop_size, pad_if_needed=True, padding_mode=self.pad_mode),
        ]
        hflip_p = float(self.augment.get("random_hflip_p", 0.0))
        if hflip_p > 0.0:
            train_tfms.append(T.RandomHorizontalFlip(p=hflip_p))
        rot_deg = self.augment.get("random_rotation_degrees")
        if rot_deg is not None and float(rot_deg) > 0.0:
            train_tfms.append(T.RandomRotation(float(rot_deg)))
        cj = self.augment.get("color_jitter")
        if isinstance(cj, dict) and cj:
            train_tfms.append(T.ColorJitter(**cj))

        blur_cfg = self.augment.get("blur")
        if isinstance(blur_cfg, dict):
            kernel_size = int(blur_cfg.get("kernel_size", 3))
            if kernel_size % 2 == 0:
                kernel_size += 1
            sigma = blur_cfg.get("sigma")
            probability = float(blur_cfg.get("probability", 0.5))
            logger.info(
                f"Enabling blur augmentation: kernel_size={kernel_size}, sigma={sigma}, "
                f"probability={probability}"
            )
            blur_transform = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
            train_tfms.append(T.RandomApply([blur_transform], p=probability))

        resize_cfg = self.augment.get("resize")
        if isinstance(resize_cfg, dict):
            scale_min = float(resize_cfg.get("scale_min", 0.8))
            scale_max = float(resize_cfg.get("scale_max", 1.0))
            probability = float(resize_cfg.get("probability", 1.0))
            interpolation = resize_cfg.get("interpolation", "bilinear")
            logger.info(
                f"Enabling scale-resize augmentation: "
                f"scale_min={scale_min}, scale_max={scale_max}, probability={probability}"
            )
            train_tfms.append(
                RandomScaleResize(
                    crop_size=self.crop_size,
                    scale_min=scale_min,
                    scale_max=scale_max,
                    probability=probability,
                    interpolation=interpolation,
                )
            )

        # Optional JPEG compression augmentation (training only)
        jpeg_cfg = self.augment.get("jpeg")
        if isinstance(jpeg_cfg, dict):
            quality_min = int(jpeg_cfg.get("quality_min", 70))
            quality_max = int(jpeg_cfg.get("quality_max", 95))
            probability = float(jpeg_cfg.get("probability", 0.5))
            logger.info(
                f"Enabling JPEG augmentation: "
                f"quality_min={quality_min}, quality_max={quality_max}, "
                f"probability={probability}"
            )
            train_tfms.append(
                RandomJPEGCompression(
                    quality_min=quality_min,
                    quality_max=quality_max,
                    probability=probability,
                )
            )

        # ToTensor must be last so that all previous transforms operate on PIL images.
        train_tfms.append(T.ToTensor())
        train_transform = T.Compose(train_tfms)

        val_tfms: List[T.Transform] = [
            T.ToTensor(),
            T.CenterCrop(self.crop_size),
        ]
        val_transform = T.Compose(val_tfms)

        logger.info(f"Train transform: {train_transform}")
        logger.info(f"Val transform: {val_transform}")

        # Setup training and validation datasets for 'fit' stage
        if stage == "fit" or stage is None:
            # Collect image paths from directories/files/lists
            logger.info("📂 Collecting training images...")
            self._train_sources = None
            train_real, real_sources = self._collect_paths_with_sources(
                self.train_real_paths, prefix="real"
            )
            train_gen, gen_sources = self._collect_paths_with_sources(
                self.train_gen_paths, prefix="gen"
            )
            logger.info("📂 Collecting validation images...")
            val_real = collect_image_paths(self.val_real_paths)
            val_gen = collect_image_paths(self.val_gen_paths)

            train_items = build_items_from_lists(train_real, train_gen)
            val_items = build_items_from_lists(val_real, val_gen)

            if len(train_items) == 0:
                raise ValueError("Training set is empty. Check your list files.")
            if len(val_items) == 0:
                raise ValueError("Validation set is empty. Check your list files.")

            logger.info(
                f"Training: {len(train_real)} real + {len(train_gen)} generated = "
                f"{len(train_items)} total"
            )
            logger.info(
                f"Validation: {len(val_real)} real + {len(val_gen)} generated = "
                f"{len(val_items)} total"
            )

            self._train_ds = ImagePathsDataset(train_items, transform=train_transform)
            self._train_sources = real_sources + gen_sources
            self._val_ds = ImagePathsDataset(val_items, transform=val_transform)

            self._maybe_build_sampler(train_items)

        # Setup heldout dataset (second validation loader) for fit/validate stages
        if stage in ("fit", "validate") or stage is None:
            if self.heldout_real_paths and self.heldout_gen_paths:
                logger.info("📂 Collecting heldout images...")
                hold_real = collect_image_paths(self.heldout_real_paths)
                hold_gen = collect_image_paths(self.heldout_gen_paths)
                hold_items = build_items_from_lists(hold_real, hold_gen)
                logger.info(
                    f"Heldout: {len(hold_real)} real + {len(hold_gen)} generated = "
                    f"{len(hold_items)} total"
                )
                self._heldout_ds = ImagePathsDataset(hold_items, transform=val_transform)
            else:
                self._heldout_ds = None

        # Setup test dataset only for test/validate stages (not during fit)
        if stage in ("test", "validate") or stage is None:
            if self.test_real_paths and self.test_gen_paths:
                logger.info("📂 Collecting test images...")
                test_real = collect_image_paths(self.test_real_paths)
                test_gen = collect_image_paths(self.test_gen_paths)
                test_items = build_items_from_lists(test_real, test_gen)
                logger.info(
                    f"Test: {len(test_real)} real + {len(test_gen)} generated = "
                    f"{len(test_items)} total"
                )

                self._test_ds = ImagePathsDataset(test_items, transform=val_transform)
            else:
                self._test_ds = None

    def train_dataloader(self) -> DataLoader:
        assert self._train_ds is not None, "setup() must be called before train_dataloader()"
        sampler = self._sampler
        kwargs = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": True,
            "sampler": sampler,
            "shuffle": sampler is None,
            "drop_last": self.drop_last,
        }
        if self.num_workers > 0 and self.prefetch_factor is not None:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(self._train_ds, **kwargs)

    def val_dataloader(self) -> DataLoader | List[DataLoader]:
        assert self._val_ds is not None, "setup() must be called before val_dataloader()"
        kwargs = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": True,
            "shuffle": False,
            "drop_last": False,
        }
        if self.num_workers > 0 and self.prefetch_factor is not None:
            kwargs["prefetch_factor"] = self.prefetch_factor
        val_loader = DataLoader(self._val_ds, **kwargs)
        if self._heldout_ds is not None:
            heldout_loader = DataLoader(self._heldout_ds, **kwargs)
            return [val_loader, heldout_loader]
        return val_loader

    def test_dataloader(self) -> Optional[DataLoader]:
        if self._test_ds is None:
            return None
        kwargs = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": True,
            "shuffle": False,
            "drop_last": False,
        }
        if self.num_workers > 0 and self.prefetch_factor is not None:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(self._test_ds, **kwargs)
