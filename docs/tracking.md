# Experiment Tracking & Metrics

CHROMA integrates with **Weights & Biases (W&B)** and **TensorBoard** for experiment tracking. It logs scalar metrics, loss curves, and distribution histograms of prediction scores.

## Configuring the Logger

In your YAML configuration (`trainer` section):

### Weights & Biases (Recommended)
```yaml
trainer:
  logger:
    type: wandb
    init_args:
      project: "chroma-deepfake-detection"
      name: "experiment-name"  # Optional
      save_dir: "wandb_logs"
      tags: ["a100", "resnet50", "lab-corr"]
```

### TensorBoard
```yaml
trainer:
  logger:
    type: tensorboard
    init_args:
      save_dir: "tb_logs"
      name: "my_experiment"
```

## Logged Metrics

The model automatically logs the following metrics for `train`, `val`, `test`, and `heldout` stages:

*   **Accuracy** (`acc`): Binary classification accuracy.
*   **AUROC** (`auroc`): Area Under the Receiver Operating Characteristic curve.
*   **F1 Score** (`f1`): F1 score at threshold 0.5.
*   **Average Precision** (`ap`): Area under the Precision-Recall curve.
*   **Loss** (`loss`): Binary Cross Entropy loss.

### Score Histograms
If `log_score_histograms: true` is set in the `model` config, the trainer generates histograms of the predicted `p(fake)` scores for real vs. fake images at the end of every epoch. This is crucial for visualizing the separation between classes.

*   **Blue**: Real images
*   **Orange**: Generated images

## Heldout Evaluation

CHROMA supports a "heldout" validation set (defined via `heldout_real_paths` / `heldout_gen_paths`) to monitor generalization to unseen generators during training.

### Automatic CSV Logging
If `heldout_csv_path` is provided in the model config, the model writes a CSV file at the end of every epoch containing the predictions for the heldout set:

```csv
filename,chroma
biggan_256/0001.png,0.98231
real_coco/0042.jpg,0.01234
...
```

This allows for offline analysis of per-generator performance.

### Metadata-Aware Metrics
If `heldout_meta_csv_path` is provided, the model can compute the "Average AUC" across different generator families (if the metadata file follows the expected format).


