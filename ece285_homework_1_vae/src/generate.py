import argparse
import math
from pathlib import Path

import torch
from torchvision.utils import save_image

from src.utils import (
    get_device,
    load_config,
    load_trained_model,
    set_seed,
)


@torch.no_grad()
def generate(args):
    config = load_config(args.config)

    device = get_device(args.device)
    set_seed(args.seed)

    model, checkpoint = load_trained_model(
        config,
        args.checkpoint,
        device,
    )

    output_dir = Path(
        args.output_dir
        or Path(config["output_dir"]) / "figures" / "generated"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    # Generate on CPU for a fixed seeded latent sequence,
    # then transfer the latent vectors to the selected device.
    generator = torch.Generator().manual_seed(args.seed)

    z = torch.randn(
        args.num_samples,
        config["latent_dim"],
        generator=generator,
    ).to(device)

    samples = model.decode(z).cpu()

    # Save a grid.
    nrow = math.ceil(math.sqrt(args.num_samples))

    save_image(
        samples,
        output_dir / "generated_grid.png",
        nrow=nrow,
        padding=2,
    )

    # Save each sample separately.
    for i, image in enumerate(samples):
        save_image(
            image,
            output_dir / f"sample_{i:03d}.png",
        )

    print(f"Generated {args.num_samples} images.")
    print(f"Saved to: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic X-rays from a VAE."
    )

    parser.add_argument(
        "--config", default="config.yaml"
    )

    parser.add_argument(
        "--checkpoint",
        default="outputs/baseline/checkpoints/best.pt",
    )

    parser.add_argument(
        "--num-samples", type=int, default=25
    )

    parser.add_argument(
        "--seed", type=int, default=123
    )

    parser.add_argument(
        "--device", default="auto"
    )

    parser.add_argument(
        "--output-dir", default=None
    )

    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
