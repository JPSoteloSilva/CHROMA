from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BackboneTrainConfig(BaseModel):
    """Configuration for training backbone layer bands.

    `first_n` and `last_n` describe how many of the earliest and latest
    parameter scopes (in the timm backbone) should remain trainable.
    """

    first_n: int
    last_n: int


class LRGroupsConfig(BaseModel):
    """Configuration for two learning rate groups: backbone vs head."""

    backbone_lr_mult: float
    head_lr_mult: float


class OptimizerConfig(BaseModel):
    """Configuration for AdamW optimizer."""

    init_args: Dict[str, Any] = Field(default_factory=dict)


class SchedulerConfig(BaseModel):
    """Configuration for ReduceLROnPlateau scheduler."""

    init_args: Dict[str, Any] = Field(default_factory=dict)
    monitor: Optional[str] = None  # Metric to monitor (defaults to "val/auroc")


class ModelConfig(BaseModel):
    """Configuration for the model."""

    num_classes: int = 2
    backbone_name: str = "resnet50"
    pretrained: bool = True
    feature_composer: FeatureComposerConfig
    # If null/omitted, all backbone/head scopes are trainable.
    # If provided, exactly the first_n and last_n scopes are trainable;
    # middle scopes are frozen.
    train_layers: Optional[BackboneTrainConfig] = None
    lr_groups: LRGroupsConfig
    log_score_histograms: bool = False
    # Directory where per-epoch heldout prediction CSVs are written.
    heldout_csv_path: Optional[str] = None
    # Optional metadata CSV for heldout AVG AUC computation.
    # Must contain at least columns: 'filename' and 'typ'.
    heldout_meta_csv_path: Optional[str] = None
    optimizer: Optional[OptimizerConfig] = None
    scheduler: Optional[SchedulerConfig] = None
    class_weights: Optional[List[float]] = None  # alternative to weighted sampler


class JPEGAugmentConfigModel(BaseModel):
    """Configuration for random JPEG compression augmentation."""

    quality_min: int = 70
    quality_max: int = 95
    probability: float = 0.5


class BlurAugmentConfigModel(BaseModel):
    """Configuration for Gaussian blur augmentation."""

    kernel_size: int = Field(3, ge=1)
    sigma: Optional[Union[Tuple[float, float], float]] = None
    probability: float = Field(0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def ensure_odd_kernel_size(self):
        kernel = int(self.kernel_size)
        if kernel % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.kernel_size = kernel
        return self


class ResizeAugmentConfigModel(BaseModel):
    """Configuration for random downscale-then-resize augmentation."""

    scale_min: float = Field(0.8, gt=0.0, le=1.0)
    scale_max: float = Field(1.0, gt=0.0, le=1.0)
    probability: float = Field(1.0, ge=0.0, le=1.0)
    interpolation: str = "bilinear"

    @model_validator(mode="after")
    def validate_scale_range(self):
        if self.scale_min > self.scale_max:
            raise ValueError("scale_min must be <= scale_max")
        return self


class AugmentConfigModel(BaseModel):
    """Configuration for data augmentation."""

    random_hflip_p: float = 0.5
    random_rotation_degrees: float | None = 0.0
    color_jitter: Optional[Dict[str, Any]] = None
    jpeg: Optional[JPEGAugmentConfigModel] = None
    blur: Optional[BlurAugmentConfigModel] = None
    resize: Optional[ResizeAugmentConfigModel] = None


class TransformConfig(BaseModel):
    """Configuration for a single transform in a pipeline."""

    type: str
    # Allow any additional fields
    model_config = ConfigDict(extra="allow")


class PipelineConfig(BaseModel):
    """Configuration for a feature pipeline."""

    name: str
    standardize: Optional[str] = None  # e.g., "zscore"
    transforms: List[TransformConfig]


class FeatureComposerConfig(BaseModel):
    """
    Configuration for the feature composer (channels-only, single config).

    Always required in model config. Supports RGB-only via include_rgb=True with empty pipelines.
    """

    include_rgb: bool = True
    rgb_standardize: Optional[str] = None  # e.g., "imagenet", "zscore"
    pipelines: Optional[List[PipelineConfig]] = None
    model_config = ConfigDict(extra="allow")


class DataConfig(BaseModel):
    """Configuration for data module."""

    train_real_paths: Union[str, List[str]]
    train_gen_paths: Union[str, List[str]]
    val_real_paths: Union[str, List[str]]
    val_gen_paths: Union[str, List[str]]
    # Optional heldout split used as a second validation dataloader during fit
    heldout_real_paths: Optional[Union[str, List[str]]] = None
    heldout_gen_paths: Optional[Union[str, List[str]]] = None
    test_real_paths: Optional[Union[str, List[str]]] = None
    test_gen_paths: Optional[Union[str, List[str]]] = None
    batch_size: int = 64
    num_workers: int = 8
    prefetch_factor: Optional[int] = None  # None = default (2), or set manually
    drop_last: bool = True
    use_weighted_sampler: bool = True
    crop_size: int = 224
    pad_mode: str = "reflect"
    augment: Optional[AugmentConfigModel] = None
    sampling_strategy: Literal["none", "class", "source", "class_source"] = "class"


class CallbackConfig(BaseModel):
    """Configuration for a callback."""

    type: str
    init_args: Dict[str, Any] = Field(default_factory=dict)
    # Allow class_path as alternative to type
    class_path: Optional[str] = None


class LoggerConfig(BaseModel):
    """Configuration for logger."""

    type: str = "tensorboard"  # "wandb", "tensorboard", or null
    init_args: Dict[str, Any] = Field(default_factory=dict)


class TrainerConfig(BaseModel):
    """Configuration for PyTorch Lightning trainer."""

    max_epochs: int
    precision: Union[str, int] = "32"
    accelerator: str = "gpu"
    devices: Union[int, List[int], str] = 1
    log_every_n_steps: int = 50
    limit_train_batches: Optional[Union[int, float]] = None
    limit_val_batches: Optional[Union[int, float]] = None
    overfit_batches: Union[int, float] = 0
    callbacks: List[CallbackConfig] = Field(default_factory=list)
    logger: Optional[LoggerConfig] = None
    # Allow any additional trainer arguments
    model_config = ConfigDict(extra="allow")


class TrainingConfig(BaseModel):
    """Root configuration for training."""

    seed_everything: Optional[int] = None
    model: ModelConfig
    data: DataConfig
    trainer: TrainerConfig

    @classmethod
    def from_yaml(cls, yaml_path: str) -> TrainingConfig:
        """Load configuration from a YAML file."""
        from pathlib import Path

        import yaml

        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def model_dump_dict(self) -> Dict[str, Any]:
        """Convert config to plain dictionary."""
        return self.model_dump(mode="python", exclude_none=False)
