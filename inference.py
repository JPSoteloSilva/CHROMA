from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from loguru import logger
from PIL import Image
from torchvision.transforms import functional as TF
from tqdm import tqdm

from chroma.config import TrainingConfig
from chroma.data.datasets import collect_image_paths
from chroma.models.module import ChromaModel


def load_model_and_config(checkpoint_path: str, config_path: str, device: str = "cpu"):
    """Load model from checkpoint and config (for crop size only)."""
    # Load config
    cfg = TrainingConfig.from_yaml(config_path)

    # Load model
    model = ChromaModel.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    model.to(device)

    return model, None, cfg


def preprocess_image(image_path: str, crop_size: int = 224) -> torch.Tensor:
    """Load and preprocess a single image."""
    with Image.open(image_path) as im:
        im = im.convert("RGB")

    x = TF.to_tensor(im)  # [0,1], (3, H, W)

    # Pad if needed
    _, h, w = x.shape
    if min(h, w) < crop_size:
        pad_h = max(0, crop_size - h)
        pad_w = max(0, crop_size - w)
        padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
        x = TF.pad(x, padding, padding_mode="reflect")

    # Center crop
    x = TF.center_crop(x, crop_size)

    return x


def predict_on_folder(
    model,
    input_folder: str,
    output_csv: str,
    batch_size: int = 32,
    device: str = "cpu",
    crop_size: int = 224,
):
    """Predict on all images in folder and save results to CSV."""
    # Collect all image paths
    logger.info(f"Collecting images from {input_folder}...")
    image_paths = collect_image_paths(input_folder)
    logger.info(f"Found {len(image_paths)} images")

    if not image_paths:
        logger.warning("No images found!")
        return

    # Prepare CSV
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "probability_fake"])

        # Process in batches
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing"):
            batch_paths = image_paths[i : i + batch_size]

            # Load and preprocess batch
            batch_tensors = []
            valid_paths = []
            for path in batch_paths:
                try:
                    tensor = preprocess_image(path, crop_size)
                    batch_tensors.append(tensor)
                    valid_paths.append(path)
                except Exception as e:
                    logger.error(f"Error loading {path}: {e}")

            if not batch_tensors:
                continue

            # Stack to batch
            batch = torch.stack(batch_tensors, dim=0).to(device)

            # Predict
            with torch.no_grad():
                logits = model(batch)
                probs = torch.softmax(logits, dim=1)[:, 1]  # Probability of class 1 (fake)

            # Write results
            for path, prob in zip(valid_paths, probs.cpu().numpy()):
                writer.writerow([path, float(prob)])

    logger.info(f"✓ Results saved to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Run inference on folder of images")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint (.ckpt)"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to training config (.yaml)")
    parser.add_argument("--input-folder", type=str, required=True, help="Folder containing images")
    parser.add_argument("--output-csv", type=str, required=True, help="Output CSV path")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument(
        "--crop-size", type=int, default=224, help="Crop size (should match training)"
    )

    args = parser.parse_args()

    # Load model and config
    logger.info("Loading model and config...")
    model, composer, cfg = load_model_and_config(args.checkpoint, args.config, args.device)

    # Get crop size from config if not specified
    crop_size = cfg.data.crop_size if cfg.data.crop_size else args.crop_size

    # Run inference
    predict_on_folder(
        model=model,
        input_folder=args.input_folder,
        output_csv=args.output_csv,
        batch_size=args.batch_size,
        device=args.device,
        crop_size=crop_size,
    )


if __name__ == "__main__":
    main()
