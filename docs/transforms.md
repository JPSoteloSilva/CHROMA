# Feature Transforms & Configuration

CHROMA uses a flexible **feature composer** system to augment standard RGB images with forensic cues extracted in various color spaces. This system is configured entirely through the YAML config file.

## The Feature Composer

The `BatchFeatureComposer` takes an RGB batch `(B, 3, H, W)` and produces a multi-channel feature volume `(B, C_out, H, W)`. It works by:

1.  Optionally including the original RGB image (standardized if requested).
2.  Running a list of independent **pipelines**.
3.  Concatenating all outputs along the channel dimension.

The model's first convolution layer is automatically adapted to accept `C_out` input channels.

## Configuration Structure

In your config file (e.g., `configs/a100_resnet50.yaml`), the `model` section contains a `feature_composer` block:

```yaml
model:
  feature_composer:
    include_rgb: true             # Whether to include the original RGB image
    rgb_standardize: imagenet     # Normalization for RGB: "imagenet", "zscore", or null
    pipelines:
      - name: rgb_corr            # Unique name for this pipeline
        transforms:               # List of transforms to apply sequentially
          - type: inter_channel_correlation
            window: 7
```

## Available Transforms

All transforms expect input tensors of shape `(B, 3, H, W)` and return tensors of the same spatial resolution.

### 1. Inter-Channel Correlation
Computes local Pearson correlation between channel pairs (R-G, G-B, R-B) over a sliding window.

*   **Type**: `inter_channel_correlation`
*   **Parameters**:
    *   `window` (int, default: 7): Size of the sliding window (must be odd).
    *   `stride` (int, default: 1): Stride for the sliding window.
    *   `eps` (float, default: 1e-3): Stability term for variance calculation.

### 2. Color Space Conversions
Wrappers around Kornia color conversions. Useful as preprocessing steps before other transforms.

*   **Types**: 
    *   `rgb_to_lab`
    *   `rgb_to_yuv`
    *   `rgb_to_hsv`
*   **Parameters**: None.

### 3. Gradient Magnitude
Computes the magnitude of spatial gradients for each channel.

*   **Type**: `gradient_magnitude`
*   **Parameters**:
    *   `operator` (str, default: "Sobel"): Gradient operator, either "Sobel" or "Scharr".
    *   `normalization` (str, default: "zscore"): Per-channel normalization, "zscore" or "none".
    *   `eps` (float, default: 1e-3): Stability term.

### 4. Noise Quantization Restore
Implements a "forensic noise" feature by averaging multiple quantized, noisy copies of the image and computing the residual difference from the original. This highlights compression and generation artifacts.

*   **Type**: `noise_quant_restore`
*   **Parameters**:
    *   `K` (int, default: 8): Number of noisy copies to average.
    *   `sigma` (float, default: 0.01): Noise standard deviation.
    *   `num_levels` (int, default: 16): Number of quantization levels.
    *   `seed` (int, optional): Random seed for reproducibility.

## Example Configuration

Here is a configuration that stacks **RGB**, **Lab Correlations**, and **Gradient Magnitudes**:

```yaml
model:
  feature_composer:
    include_rgb: true
    rgb_standardize: imagenet
    pipelines:
      # Pipeline 1: Lab Correlations
      - name: lab_correlation
        transforms:
          - type: rgb_to_lab
          - type: inter_channel_correlation
            window: 7
            
      # Pipeline 2: Gradient Magnitudes (on RGB)
      - name: gradients
        transforms:
          - type: gradient_magnitude
            operator: Sobel
            normalization: zscore
```

**Resulting Input Volume:**
*   3 channels (RGB)
*   + 3 channels (Lab Correlation)
*   + 3 channels (Gradient Magnitude)
*   **Total**: 9 input channels.


