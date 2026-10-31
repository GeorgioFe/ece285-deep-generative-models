import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import ignite
import torchvision

from ignite.metrics import FID, InceptionScore

from src.dataset import make_test_loader
from src.utils import (
    get_device,
    load_config,
    load_trained_model,
    set_seed,
)


def prepare_for_inception(images):
    """Convert grayscale images into 299x299 RGB tensors."""

    # [B, 1, H, W] -> [B, 3, H, W]
    images = images.repeat(1, 3, 1, 1)

    images = F.interpolate(
        images,
        size=(299, 299),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )

    return images.clamp(0.0, 1.0)


@torch.no_grad()
def evaluate(args):
    config = load_config(args.config)

    device = get_device(args.device)
    set_seed(args.seed)

    model, checkpoint = load_trained_model(
        config,
        args.checkpoint,
        device,
    )

    test_loader = make_test_loader(
        config,
        device,
        batch_size=args.batch_size or config["eval_batch_size"],
    )

    print(f"Using device: {device}")
    print(f"Evaluating checkpoint epoch: {checkpoint['epoch']}")

    # Ignite downloads pretrained Inception weights
    # the first time these metrics are initialized.
    fid = FID(device=device)
    inception_score = InceptionScore(device=device)

    fid.reset()
    inception_score.reset()

    # Fixed random generator for reproducible evaluation.
    generator = torch.Generator().manual_seed(args.seed)

    num_samples = 0

    for batch_idx, (real, _) in enumerate(test_loader):

        real = real.to(device)

        batch_size = real.size(0)

        # Generate fresh random samples from N(0, I).
        z = torch.randn(
            batch_size,
            config["latent_dim"],
            generator=generator,
        ).to(device)

        fake = model.decode(z)

        # Match channels and resolution for Inception.
        real_rgb = prepare_for_inception(real)
        fake_rgb = prepare_for_inception(fake)

        # Update the metrics batch by batch.
        fid.update((fake_rgb, real_rgb))
        inception_score.update(fake_rgb)

        num_samples += batch_size

        print(
            f"Processed {num_samples}/{len(test_loader.dataset)} images",
            end="\r",
        )

    fid_value = float(fid.compute())
    is_value = float(inception_score.compute())

    results = {
        "fid": fid_value,
        "inception_score": is_value,
        "num_real_images": num_samples,
        "num_generated_images": num_samples,
        "checkpoint_epoch": checkpoint["epoch"],
        "seed": args.seed,
        "image_size": config["image_size"],
        "inception_input_size": 299,
        "fid_feature_type": "Ignite default InceptionV3 logits",
        "fid_num_features": 1000,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "ignite_version": ignite.__version__,
    }

    output_path = Path(
        config["output_dir"]
    ) / "metrics.json"

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(results, f, indent=4)

    print("\n")
    print(f"FID: {fid_value:.4f}")
    print(f"Inception Score: {is_value:.4f}")
    print(f"Saved metrics to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate VAE using FID and IS."
    )

    parser.add_argument(
        "--config", default="config.yaml"
    )

    parser.add_argument(
        "--checkpoint",
        default="outputs/baseline/checkpoints/best.pt",
    )

    parser.add_argument(
        "--device", default="auto"
    )

    parser.add_argument(
        "--batch-size", type=int, default=None
    )

    parser.add_argument(
        "--seed", type=int, default=123
    )

    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
