from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn.functional as F
import wandb
from lightning.pytorch import LightningModule
from torch import Tensor, nn
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryF1Score,
)

from chroma.config import BackboneTrainConfig, LRGroupsConfig
from chroma.transforms.composer import BatchFeatureComposer


class ChromaModel(LightningModule):
    def __init__(
        self,
        feature_composer: Dict[str, Any],
        num_classes: int = 2,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        train_layers: Union[BackboneTrainConfig, Dict[str, Any], None] = None,
        lr_groups: Union[LRGroupsConfig, Dict[str, Any]] = None,
        heldout_meta_csv_path: Optional[str] = None,
        optimizer: Optional[Dict[str, Any]] = None,
        scheduler: Optional[Dict[str, Any]] = None,
        log_score_histograms: bool = False,
        heldout_csv_path: Optional[str] = None,
        class_weights: Optional[list[float]] = None,
    ) -> None:
        super().__init__()
        # Composer is always required
        if not feature_composer:
            raise ValueError("feature_composer config is required in the model configuration.")
        self.save_hyperparameters(ignore=["class_weights"])

        # Validate and store training bands / LR group configuration
        if isinstance(train_layers, dict):
            self._train_cfg: Optional[BackboneTrainConfig] = BackboneTrainConfig(**train_layers)
        else:
            self._train_cfg = train_layers

        if self._train_cfg is not None:
            if self._train_cfg.first_n <= 0 and self._train_cfg.last_n <= 0:
                raise ValueError(
                    "train_layers configuration requires first_n > 0 or last_n > 0 "
                    "(or both). Use null to train all layers."
                )

        if lr_groups is None:
            raise ValueError("Model configuration must provide an 'lr_groups' section.")

        if isinstance(lr_groups, dict):
            self._lr_groups_cfg = LRGroupsConfig(**lr_groups)
        else:
            self._lr_groups_cfg = lr_groups

        # Build the single composer (channels-only) from config
        self._composer = BatchFeatureComposer(
            include_rgb=feature_composer.get("include_rgb", True),
            rgb_standardize=feature_composer.get("rgb_standardize"),
            pipelines=feature_composer.get("pipelines"),
        )

        # Infer input channels from composer
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 16, 16)
            inferred_in_chans = int(self._composer(dummy).shape[1])

        self.model = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=inferred_in_chans,
            num_classes=num_classes,
            global_pool="avg",
        )

        # Apply generic first/last-N training bands to backbone and head scopes
        self._apply_train_bands()

        weight_tensor = None
        if class_weights is not None:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
        self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)

        self.train_acc = BinaryAccuracy()
        self.train_auroc = BinaryAUROC()
        self.train_f1 = BinaryF1Score()
        self.train_ap = BinaryAveragePrecision()
        self.val_acc = BinaryAccuracy()
        self.val_auroc = BinaryAUROC()
        self.val_f1 = BinaryF1Score()
        self.val_ap = BinaryAveragePrecision()
        # Heldout (second validation dataloader) metrics
        self.hold_acc = BinaryAccuracy()
        self.hold_auroc = BinaryAUROC()
        self.hold_f1 = BinaryF1Score()
        self.hold_ap = BinaryAveragePrecision()
        self.test_acc = BinaryAccuracy()
        self.test_auroc = BinaryAUROC()
        self.test_f1 = BinaryF1Score()
        self.test_ap = BinaryAveragePrecision()

        self._optimizer_cfg = optimizer or {
            "init_args": {"lr": 3e-4, "weight_decay": 0.01},
        }
        self._scheduler_cfg = scheduler

        self._log_score_hist: bool = bool(log_score_histograms)
        self._heldout_meta_csv_path: Optional[str] = heldout_meta_csv_path
        # Optional CSV path for logging heldout predictions
        self._heldout_csv_path: Optional[str] = heldout_csv_path
        self._heldout_seen: bool = False
        # Score caches for histogram visualization per epoch
        self._scores_train: Dict[str, List[float]] = {"real": [], "fake": []}
        self._scores_val: Dict[str, List[float]] = {"real": [], "fake": []}
        self._scores_heldout: Dict[str, List[float]] = {"real": [], "fake": []}
        self._scores_test: Dict[str, List[float]] = {"real": [], "fake": []}
        # Accumulated (relative_path, score) pairs for heldout CSV logging
        self._heldout_records: List[tuple[str, float]] = []

    # ------------------------------------------------------------------
    # Generic layer training-band helpers
    # ------------------------------------------------------------------

    def _parameter_scopes(self) -> List[str]:
        """
        Build an ordered list of parameter scopes from the backbone model.

        Each scope corresponds to the module path prefix before the final
        parameter name, e.g. \"blocks.0.attn.qkv.weight\" -> \"blocks.0.attn.qkv\".
        """

        scopes: List[str] = []
        seen = set()
        for name, _ in self.model.named_parameters():
            scope = name.rsplit(".", 1)[0] if "." in name else name
            if scope not in seen:
                seen.add(scope)
                scopes.append(scope)
        return scopes

    def _apply_train_bands(self) -> None:
        """
        Train only the first `first_n` and last `last_n` parameter scopes.

        Scopes are determined by `_parameter_scopes()` and are shared across
        backbone and head so that training can target early/late parts of the
        entire timm model in a model-agnostic way.
        """

        if self._train_cfg is None:
            # Train all backbone/head parameters.
            return

        scopes = self._parameter_scopes()
        if not scopes:
            return

        total = len(scopes)
        first_n = max(0, int(self._train_cfg.first_n))
        last_n = max(0, int(self._train_cfg.last_n))

        first_n = min(first_n, total)
        last_n = min(last_n, total)

        # At this point, validation has ensured that not both are zero.
        first_indices = set(range(first_n))
        last_start = max(total - last_n, 0)
        last_indices = set(range(last_start, total))

        selected_indices = first_indices | last_indices
        train_scopes = {scopes[i] for i in selected_indices}

        # First freeze all backbone parameters, then unfreeze only the selected scopes.
        for _, param in self.model.named_parameters():
            param.requires_grad = False

        for name, param in self.model.named_parameters():
            scope = name.rsplit(".", 1)[0] if "." in name else name
            if scope in train_scopes:
                param.requires_grad = True

    def forward(self, rgb: Tensor) -> Tensor:
        # Always apply composer (Lightning handles device placement)
        self._composer.train(self.training)
        volume = self._composer(rgb)
        return self.model(volume)

    def _shared_step(self, batch, stage: str) -> Tensor:
        # Batch may be (rgb, y) or (rgb, y, paths)
        if isinstance(batch, (tuple, list)) and len(batch) == 3:
            rgb, y, paths = batch
        else:
            rgb, y = batch
            paths = None
        rgb = rgb.to(self.device)
        logits = self(rgb)
        loss = self.criterion(logits, y)

        probs = F.softmax(logits, dim=1)[:, 1]

        if stage == "train":
            self.train_acc.update(probs, y)
            self.train_auroc.update(probs, y)
            self.train_f1.update(probs, y)
            self.train_ap.update(probs, y)
            # collect scores for histograms
            if self._log_score_hist:
                p = probs.detach().cpu()
                yy = y.detach().cpu()
                self._scores_train["real"].extend(p[yy == 0].tolist())
                self._scores_train["fake"].extend(p[yy == 1].tolist())
        elif stage == "val":
            self.val_acc.update(probs, y)
            self.val_auroc.update(probs, y)
            self.val_f1.update(probs, y)
            self.val_ap.update(probs, y)
            # collect scores for histograms
            if self._log_score_hist:
                p = probs.detach().cpu()
                yy = y.detach().cpu()
                self._scores_val["real"].extend(p[yy == 0].tolist())
                self._scores_val["fake"].extend(p[yy == 1].tolist())
        elif stage == "test":
            self.test_acc.update(probs, y)
            self.test_auroc.update(probs, y)
            self.test_f1.update(probs, y)
            self.test_ap.update(probs, y)
            if self._log_score_hist:
                p = probs.detach().cpu()
                yy = y.detach().cpu()
                self._scores_test["real"].extend(p[yy == 0].tolist())
                self._scores_test["fake"].extend(p[yy == 1].tolist())
        elif stage == "heldout":
            self.hold_acc.update(probs, y)
            self.hold_auroc.update(probs, y)
            self.hold_f1.update(probs, y)
            self.hold_ap.update(probs, y)
            self._heldout_seen = True
            if self._log_score_hist:
                p = probs.detach().cpu()
                yy = y.detach().cpu()
                self._scores_heldout["real"].extend(p[yy == 0].tolist())
                self._scores_heldout["fake"].extend(p[yy == 1].tolist())
            # Record per-sample predictions for CSV logging if configured
            if self._heldout_csv_path and paths is not None:
                p = probs.detach().cpu().tolist()
                # paths is a list/sequence of strings (one per sample in batch)
                for path_str, score in zip(paths, p):
                    rel = self._relative_heldout_path(path_str)
                    self._heldout_records.append((rel, float(score)))

        self.log(
            f"{stage}/loss",
            loss,
            prog_bar=True,
            on_step=(stage == "train"),
            on_epoch=True,
        )
        return loss

    def training_step(self, batch, _batch_idx: int):
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, _batch_idx: int, dataloader_idx: int = 0):
        stage = "val" if dataloader_idx == 0 else "heldout"
        return self._shared_step(batch, stage=stage)

    def test_step(self, batch, _batch_idx: int):
        return self._shared_step(batch, stage="test")

    def on_train_epoch_end(self) -> None:
        acc = self.train_acc.compute()
        auroc = self.train_auroc.compute()
        f1 = self.train_f1.compute()
        ap = self.train_ap.compute()
        self.log("train/acc", acc, prog_bar=True)
        self.log("train/auroc", auroc, prog_bar=True)
        self.log("train/f1", f1, prog_bar=True)
        self.log("train/ap", ap, prog_bar=False)
        # Log train histogram
        if self._log_score_hist and self.logger is not None:
            try:
                self._log_histogram(
                    "train/score_hist",
                    self._scores_train["real"],
                    self._scores_train["fake"],
                    "Train score distribution",
                )
            except Exception:
                pass
        self.train_acc.reset()
        self.train_auroc.reset()
        self.train_f1.reset()
        self.train_ap.reset()
        self._scores_train = {"real": [], "fake": []}

    def on_validation_epoch_end(self) -> None:
        acc = self.val_acc.compute()
        auroc = self.val_auroc.compute()
        f1 = self.val_f1.compute()
        ap = self.val_ap.compute()
        self.log("val/acc", acc, prog_bar=True)
        self.log("val/auroc", auroc, prog_bar=True)
        self.log("val/f1", f1, prog_bar=True)
        self.log("val/ap", ap, prog_bar=False)
        # Log histograms of prediction scores for val
        if self._log_score_hist and self.logger is not None:
            try:
                self._log_histogram(
                    "val/score_hist",
                    self._scores_val["real"],
                    self._scores_val["fake"],
                    "Validation score distribution",
                )
            except Exception:
                pass
        self.val_acc.reset()
        self.val_auroc.reset()
        self.val_f1.reset()
        self.val_ap.reset()
        self._scores_val = {"real": [], "fake": []}
        # Heldout metrics if a second validation dataloader ran
        if self._heldout_seen:
            h_acc = self.hold_acc.compute()
            h_auroc = self.hold_auroc.compute()
            h_f1 = self.hold_f1.compute()
            h_ap = self.hold_ap.compute()
            self.log("heldout/acc", h_acc, prog_bar=True)
            self.log("heldout/auroc", h_auroc, prog_bar=True)
            self.log("heldout/f1", h_f1, prog_bar=True)
            self.log("heldout/ap", h_ap, prog_bar=False)
            # Log heldout histogram
            if self._log_score_hist and self.logger is not None:
                try:
                    self._log_histogram(
                        "heldout/score_hist",
                        self._scores_heldout["real"],
                        self._scores_heldout["fake"],
                        "Heldout score distribution",
                    )
                except Exception:
                    pass
            # Compute generator-aware AVG AUC on heldout if metadata is provided
            if self._heldout_meta_csv_path and self._heldout_records:
                try:
                    import pandas as pd

                    from chroma.compute_metrics import compute_metrics, dict_metrics

                    pred_df = pd.DataFrame(self._heldout_records, columns=["filename", "chroma"])
                    auc_df = compute_metrics(
                        input_csv=self._heldout_meta_csv_path,
                        table=pred_df,
                        metrics_fun=dict_metrics["auc"],
                    )
                    if "chroma" in auc_df.index and "AVG" in auc_df.columns:
                        avg_auc = float(auc_df.loc["chroma", "AVG"])
                        self.log(
                            "heldout/avg_auc",
                            avg_auc,
                            prog_bar=True,
                            on_step=False,
                            on_epoch=True,
                        )
                except Exception:
                    # Do not crash training if metrics computation fails
                    pass
            # Optionally write heldout CSV with per-image predictions
            if self._heldout_csv_path and self._heldout_records:
                self._write_heldout_csv()
            self.hold_acc.reset()
            self.hold_auroc.reset()
            self.hold_f1.reset()
            self.hold_ap.reset()
            self._heldout_seen = False
            self._scores_heldout = {"real": [], "fake": []}

    def on_test_epoch_end(self) -> None:
        acc = self.test_acc.compute()
        auroc = self.test_auroc.compute()
        f1 = self.test_f1.compute()
        ap = self.test_ap.compute()
        self.log("test/acc", acc, prog_bar=True)
        self.log("test/auroc", auroc, prog_bar=True)
        self.log("test/f1", f1, prog_bar=True)
        self.log("test/ap", ap, prog_bar=False)
        # Log test histogram once
        if self._log_score_hist and self.logger is not None:
            try:
                self._log_histogram(
                    "test/score_hist",
                    self._scores_test["real"],
                    self._scores_test["fake"],
                    "Test score distribution",
                )
            except Exception:
                pass
        self.test_acc.reset()
        self.test_auroc.reset()
        self.test_f1.reset()
        self.test_ap.reset()
        self._scores_test = {"real": [], "fake": []}

    def configure_optimizers(self):
        from torch.optim import AdamW

        opt_cfg = self._optimizer_cfg
        init_args: Dict[str, Any] = opt_cfg.get("init_args", {})
        if "lr" not in init_args:
            raise ValueError("Optimizer init_args must include an 'lr' value.")

        base_lr = float(init_args.pop("lr"))

        backbone_mult = float(self._lr_groups_cfg.backbone_lr_mult)
        head_mult = float(self._lr_groups_cfg.head_lr_mult)

        # Identify head parameters via timm classifier/head helpers when possible
        head_param_ids = self._get_head_param_ids()

        backbone_params: List[nn.Parameter] = []
        head_params: List[nn.Parameter] = []

        for p in self.parameters():
            if not p.requires_grad:
                continue
            if id(p) in head_param_ids:
                head_params.append(p)
            else:
                backbone_params.append(p)

        param_groups = [
            {"params": backbone_params, "lr": base_lr * backbone_mult},
            {"params": head_params, "lr": base_lr * head_mult},
        ]

        optimizer = AdamW(param_groups, lr=base_lr, **init_args)

        if self._scheduler_cfg:
            from torch.optim.lr_scheduler import ReduceLROnPlateau

            sch_args: Dict[str, Any] = self._scheduler_cfg.get("init_args", {})
            monitor: Optional[str] = self._scheduler_cfg.get("monitor")

            scheduler = ReduceLROnPlateau(optimizer, **sch_args)

            # Default to val/auroc if monitor not specified
            monitor_metric = monitor if monitor is not None else "val/auroc"
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": monitor_metric,
                },
            }

        return optimizer

    # -------------------------------------------------------------------------
    # Head parameter identification for LR groups
    # -------------------------------------------------------------------------

    def _get_head_param_ids(self) -> set[int]:
        """
        Identify classifier/head parameters for timm models.

        Uses timm's get_classifier() when available, falling back to common
        attribute names (\"head\", \"fc\", \"classifier\") if needed.
        """

        head_ids: set[int] = set()
        model = self.model

        modules: List[nn.Module] = []

        # Preferred: timm-provided classifier accessor
        if hasattr(model, "get_classifier"):
            try:
                cls = model.get_classifier()
            except TypeError:
                cls = None
            if isinstance(cls, nn.Module):
                modules.append(cls)
            elif isinstance(cls, str) and hasattr(model, cls):
                attr = getattr(model, cls)
                if isinstance(attr, nn.Module):
                    modules.append(attr)

        # Fallback: common attribute names
        if not modules:
            for attr_name in ("head", "fc", "classifier"):
                if hasattr(model, attr_name):
                    attr = getattr(model, attr_name)
                    if isinstance(attr, nn.Module):
                        modules.append(attr)
                        break

        for m in modules:
            for p in m.parameters(recurse=True):
                head_ids.add(id(p))

        return head_ids

    # -------------------------------------------------------------------------
    # Heldout CSV utilities
    # -------------------------------------------------------------------------

    def _relative_heldout_path(self, full_path: str) -> str:
        """
        Convert an absolute heldout image path into a relative path for CSV.

        By default, this strips everything up to and including the first
        'test_set' component if present, so:
            /scratch/.../test_set/biggan_256/img.png -> biggan_256/img.png
        Otherwise, it falls back to the basename.
        """
        p = Path(full_path)
        parts = p.parts
        if "test_set" in parts:
            idx = parts.index("test_set")
            rel = Path(*parts[idx + 1 :])
            return str(rel)
        return p.name

    def _write_heldout_csv(self) -> None:
        """Write accumulated heldout predictions to CSV at the configured path."""
        # Only write from global rank 0 in distributed settings
        if hasattr(self.trainer, "is_global_zero") and not self.trainer.is_global_zero:
            return
        if not self._heldout_records or not self._heldout_csv_path:
            return

        # Treat heldout_csv_path as a directory and write one file per epoch
        out_dir = Path(self._heldout_csv_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        epoch_idx = int(getattr(self, "current_epoch", 0))
        out_path = out_dir / f"epoch={epoch_idx}.csv"

        with out_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "chroma"])
            for rel_path, score in self._heldout_records:
                writer.writerow([rel_path, f"{score:.6f}"])

        # Clear records after writing so each epoch overwrites with fresh data
        self._heldout_records = []

    def _log_histogram(
        self, key: str, real_scores: List[float], fake_scores: List[float], title: str
    ):
        """Helper to plot and log histogram to W&B."""
        if not (real_scores or fake_scores):
            return
        fig = self._plot_score_histogram(real_scores, fake_scores, title)
        if hasattr(self.logger, "experiment"):
            # Use trainer.logger_connector to get the correct step for epoch-level logs
            # This avoids conflicts with the global_step which increments during training
            self.logger.experiment.log({key: wandb.Image(fig)}, commit=False)
        plt.close(fig)

    @staticmethod
    def _plot_score_histogram(real_scores: List[float], fake_scores: List[float], title: str = ""):
        bins = np.linspace(0.0, 1.0, 51)
        fig, ax = plt.subplots(figsize=(7, 4))
        if real_scores:
            ax.hist(
                real_scores, bins=bins, color="tab:blue", alpha=0.55, label="real", density=False
            )
        if fake_scores:
            ax.hist(
                fake_scores, bins=bins, color="tab:orange", alpha=0.55, label="fake", density=False
            )
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("p(fake)")
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.2)
        ax.legend()
        if title:
            ax.set_title(title)
        fig.tight_layout()
        return fig

    # removed numpy and helper logging; we log figures directly to W&B
