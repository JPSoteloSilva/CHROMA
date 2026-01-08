import argparse
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
import yaml
from loguru import logger
from PIL import Image

from chroma.transforms.composer import BatchFeatureComposer


def normalize_for_display(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize a tensor to [0, 1] for saving as an image."""
    # Simple min-max normalization per channel for visualization
    # Avoid division by zero
    min_val = tensor.flatten(2).min(2, keepdim=True)[0].unsqueeze(3)
    max_val = tensor.flatten(2).max(2, keepdim=True)[0].unsqueeze(3)
    return (tensor - min_val) / (max_val - min_val + 1e-6)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize feature transformations on a single image."
    )
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML (uses model.feature_composer)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="output_viz", help="Directory to save output images"
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")

    args = parser.parse_args()

    # 1. Load Config
    try:
        with open(args.config, "r") as f:
            cfg_dict = yaml.safe_load(f)

        # Extract feature_composer config
        if "model" in cfg_dict and "feature_composer" in cfg_dict["model"]:
            fc_config = cfg_dict["model"]["feature_composer"]
        elif "feature_composer" in cfg_dict:
            fc_config = cfg_dict["feature_composer"]
        else:
            fc_config = cfg_dict

        logger.info(f"Loaded feature configuration from {args.config}")

    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return

    # 2. Setup Device
    device = torch.device(args.device)

    # 3. Initialize Composer
    try:
        composer = BatchFeatureComposer(
            include_rgb=fc_config.get("include_rgb", True),
            rgb_standardize=fc_config.get("rgb_standardize"),
            pipelines=fc_config.get("pipelines"),
        )
        composer.to(device)
        composer.eval()
    except Exception as e:
        logger.error(f"Failed to initialize BatchFeatureComposer: {e}")
        return

    # 4. Load Image
    img_path = Path(args.image)
    if not img_path.exists():
        logger.error(f"Image not found: {img_path}")
        return

    try:
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            # Convert to tensor (C, H, W) in [0, 1]
            x = TF.to_tensor(im).unsqueeze(0).to(device)  # (1, C, H, W)
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        return

    # 5. Run Composer
    with torch.no_grad():
        # Run forward to populate channel_slices
        volume = composer(x)
        slices = composer.channel_slices

    # 6. Save Outputs
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving outputs to {out_dir}")

    # Iterate over the slices to extract and save each feature block
    for name, sl in slices.items():
        # Extract chunk: (1, C_chunk, H, W)
        chunk = volume[:, sl, :, :]

        # Normalize for visualization
        chunk_vis = normalize_for_display(chunk)

        if chunk.shape[1] == 3:
            # Save as RGB image
            save_path = out_dir / f"{name}.png"
            TF.to_pil_image(chunk_vis.squeeze(0).cpu()).save(save_path)
            logger.info(f"Saved {name} ({chunk.shape[1]} ch) to {save_path}")
        else:
            # Save individual channels
            for c in range(chunk.shape[1]):
                save_path = out_dir / f"{name}_ch{c}.png"
                TF.to_pil_image(chunk_vis[:, c, :, :].squeeze(0).cpu()).save(save_path)
                logger.info(f"Saved {name} channel {c} to {save_path}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
