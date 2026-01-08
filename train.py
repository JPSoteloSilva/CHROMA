from __future__ import annotations

import argparse
from typing import Any, List, Optional

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from loguru import logger

from chroma.config import (
    CallbackConfig,
    LoggerConfig,
    TrainingConfig,
)
from chroma.data.datamodule import ImageListsDataModule
from chroma.models.module import ChromaModel


def instantiate_callbacks(callbacks_cfg: List[CallbackConfig]) -> List[Any]:
    """Create callback instances from config."""
    callbacks: List[Any] = []

    for cb_cfg in callbacks_cfg:
        cb_type = cb_cfg.class_path or cb_cfg.type

        # Extract init_args
        init_args = cb_cfg.init_args.copy()

        # Handle common callback types
        if "ModelCheckpoint" in cb_type or cb_type == "ModelCheckpoint":
            callbacks.append(ModelCheckpoint(**init_args))
        elif "LearningRateMonitor" in cb_type or cb_type == "LearningRateMonitor":
            callbacks.append(LearningRateMonitor(**init_args))
        else:
            # Try to dynamically import
            if "." in cb_type:
                module_path, class_name = cb_type.rsplit(".", 1)
                module = __import__(module_path, fromlist=[class_name])
                callback_class = getattr(module, class_name)
                callbacks.append(callback_class(**init_args))
            else:
                logger.warning(f"Unknown callback type '{cb_type}', skipping")

    return callbacks


def instantiate_logger(logger_cfg: Optional[LoggerConfig]) -> Optional[Any]:
    """Create logger instance from config."""
    if not logger_cfg:
        return None

    logger_type = logger_cfg.type

    if logger_type == "wandb":
        try:
            from lightning.pytorch.loggers import WandbLogger

            return WandbLogger(**logger_cfg.init_args)
        except ImportError:
            logger.warning("wandb not installed, falling back to default logger")
            return None
    elif logger_type == "tensorboard":
        from lightning.pytorch.loggers import TensorBoardLogger

        return TensorBoardLogger(**logger_cfg.init_args)
    else:
        return None


def main():
    parser = argparse.ArgumentParser(description="Train AI-generated image detector")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument(
        "--ckpt_path", type=str, default=None, help="Path to checkpoint to resume from"
    )
    args = parser.parse_args()

    # Load config
    cfg = TrainingConfig.from_yaml(args.config)
    logger.info(f"Loaded config from: {args.config}")

    # Seed everything
    if cfg.seed_everything is not None:
        seed_everything(cfg.seed_everything, workers=True)
        logger.info(f"Set seed: {cfg.seed_everything}")

    # Create datamodule
    data_dict = cfg.data.model_dump(mode="python", exclude_none=False)
    dm = ImageListsDataModule(**data_dict)
    logger.info("Created datamodule")

    # Create model
    model_dict = cfg.model.model_dump(mode="python", exclude_none=False)

    # Convert optimizer/scheduler configs to dict format expected by model
    if model_dict.get("optimizer"):
        opt = model_dict["optimizer"]
        if hasattr(opt, "model_dump"):
            model_dict["optimizer"] = opt.model_dump(mode="python")
    if model_dict.get("scheduler"):
        sch = model_dict["scheduler"]
        if hasattr(sch, "model_dump"):
            model_dict["scheduler"] = sch.model_dump(mode="python")

    model = ChromaModel(**model_dict)
    logger.info(f"Created model: {model_dict.get('backbone_name', 'resnet50')}")

    # Create callbacks
    callbacks = instantiate_callbacks(cfg.trainer.callbacks)
    logger.info(f"Created {len(callbacks)} callbacks")

    # Create logger
    pl_logger = instantiate_logger(cfg.trainer.logger)
    if pl_logger:
        logger.info(f"Using logger: {type(pl_logger).__name__}")

    # Create trainer
    trainer_dict = cfg.trainer.model_dump(mode="python", exclude_none=False)
    trainer_dict.pop("callbacks", None)
    trainer_dict.pop("logger", None)

    trainer = Trainer(**trainer_dict, callbacks=callbacks, logger=pl_logger)
    logger.info(f"Created trainer (max_epochs={trainer_dict.get('max_epochs')})")

    # Train
    logger.info("\nStarting training...")
    trainer.fit(model, datamodule=dm, ckpt_path=args.ckpt_path)

    # Test evaluation (if test set exists)
    test_loader = dm.test_dataloader()
    if test_loader is not None:
        logger.info("\nRunning final test evaluation...")
        trainer.test(model, dataloaders=test_loader)

    logger.info("\n✓ Training complete!")


if __name__ == "__main__":
    main()
