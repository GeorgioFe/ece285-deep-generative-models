import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image

from src.dataset import make_train_val_loaders
from src.utils import (
    get_device,
    load_config,
    load_trained_model,
    set_seed,
)


def plot_loss_curves(history_path, output_path):
    """Plot training and validation loss curves."""

    with open(
        history_path,
        "r",
        newline="",
        encoding="utf-8",
    ) as f:
        history = list(csv.DictReader(f))

    epochs = [int(r["epoch"]) for r in history]

    metrics = [
        ("total", "Total VAE Loss"),
        ("recon", "Reconstruction Loss"),
        ("kl", "KL Divergence"),
    ]

    fig, axes = plt.subplots(
        1, 3, figsize=(13, 3.5)
    )

    for ax, (key, title) in zip(axes, metrics):
        train_values = [
            float(r[f"train_{key}"]) for r in history
        ]

        val_values = [
            float(r[f"val_{key}"]) for r in history
        ]

        ax.plot(
            epochs,
            train_values,
            label="Train",
        )

        ax.plot(
            epochs,
            val_values,
            label="Validation",
        )

        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


@torch.no_grad()
def plot_reconstructions(
    model,
    val_loader,
    device,
    output_path,
):
    """Save original and reconstructed validation X-rays."""

    model.eval()

    images, _ = next(iter(val_loader))

    images = images[:8].to(device)

    reconstructions = model.reconstruct(images)

    # First row: originals
    # Second row: reconstructions
    comparison = torch.cat(
        [images.cpu(), reconstructions.cpu()],
        dim=0,
    )

    save_image(
        comparison,
        output_path,
        nrow=images.size(0),
        padding=3,
    )


def main(args):
    config = load_config(args.config)

    device = get_device(args.device)
    set_seed(config["seed"])

    output_dir = Path(config["output_dir"])
    figures_dir = output_dir / "figures"

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_loss_curves(
        output_dir / "history.csv",
        figures_dir / "loss_curves.png",
    )

    model, _ = load_trained_model(
        config,
        args.checkpoint,
        device,
    )

    _, val_loader = make_train_val_loaders(
        config,
        device,
    )

    plot_reconstructions(
        model,
        val_loader,
        device,
        figures_dir / "reconstruction_comparison.png",
    )

    print("Report figures saved to:", figures_dir)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate VAE report figures."
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

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
