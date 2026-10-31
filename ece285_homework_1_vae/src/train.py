import argparse
import csv
import math
from pathlib import Path
from tqdm.auto import tqdm

import torch
from torch.optim import Adam
from torchvision.utils import save_image

from src.dataset import make_train_val_loaders
from src.model import ConvVAE, vae_loss
from src.utils import (
    get_device,
    load_config,
    save_config,
    set_seed,
)


def run_epoch(
    model,
    loader,
    device,
    beta,
    optimizer=None,
    max_batches=None,
):
    """Run one training or validation epoch."""

    training = optimizer is not None
    model.train(training)

    totals = {
        "total": 0.0,
        "recon": 0.0,
        "kl": 0.0,
    }

    num_images = 0

    progress_bar = tqdm(
        enumerate(loader),
        total=min(len(loader), max_batches)
        if max_batches is not None else len(loader),
        desc="Train" if training else "Validation",
        leave=False,
    )

    for batch_idx, (images, _) in progress_bar:
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(
            device,
            non_blocking=(device.type == "cuda"),
        )

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            recon, mu, logvar = model(images)

            losses = vae_loss(
                recon, images, mu, logvar, beta
            )

            if training:
                losses["total"].backward()
                optimizer.step()

        batch_size = images.size(0)
        num_images += batch_size

        for name in totals:
            totals[name] += (
                losses[name].item() * batch_size
            )

        progress_bar.set_postfix(
            loss=f"{totals['total'] / num_images:.4f}",
            recon=f"{totals['recon'] / num_images:.4f}",
            kl=f"{totals['kl'] / num_images:.4f}",
        )

    if num_images == 0:
        raise RuntimeError("No images processed in this epoch.")

    return {
        name: value / num_images
        for name, value in totals.items()
    }


@torch.no_grad()
def save_progress_images(
    model,
    val_loader,
    fixed_z,
    device,
    output_dir,
    epoch,
):
    """Save comparable images across training epochs."""

    model.eval()

    figure_dir = Path(output_dir) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    # Same latent vectors at each saved epoch.
    generated = model.decode(fixed_z.to(device))

    save_image(
        generated.cpu(),
        figure_dir / f"samples_epoch_{epoch:03d}.png",
        nrow=4,
    )

    # Same validation images at each saved epoch.
    real, _ = next(iter(val_loader))
    real = real[:8].to(device)

    recon = model.reconstruct(real)

    comparison = torch.cat(
        [real.cpu(), recon.cpu()],
        dim=0,
    )

    save_image(
        comparison,
        figure_dir / f"recon_epoch_{epoch:03d}.png",
        nrow=real.size(0),
    )


def write_history(history, path):
    """Save epoch-level losses as CSV."""

    fieldnames = [
        "epoch",
        "train_total",
        "train_recon",
        "train_kl",
        "val_total",
        "val_recon",
        "val_kl",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def train(args):
    # ------------------------------------------
    # Configuration
    # ------------------------------------------
    config = load_config(args.config)

    if args.device is not None:
        config["device"] = args.device

    if args.epochs is not None:
        config["epochs"] = args.epochs

    if args.batch_size is not None:
        config["batch_size"] = args.batch_size

    if args.data_dir is not None:
        config["data_dir"] = args.data_dir

    if args.output_dir is not None:
        config["output_dir"] = args.output_dir

    if config["image_size"] != 128:
        raise ValueError(
            "This baseline architecture requires image_size=128."
        )

    device = get_device(config["device"])
    set_seed(config["seed"])

    output_dir = Path(config["output_dir"])
    checkpoint_dir = output_dir / "checkpoints"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    save_config(
        config,
        output_dir / "config_used.yaml",
    )

    print(f"Using device: {device}")

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # ------------------------------------------
    # Dataset
    # ------------------------------------------
    train_loader, val_loader = make_train_val_loaders(
        config,
        device,
    )

    # ------------------------------------------
    # Model and optimizer
    # ------------------------------------------
    model = ConvVAE(
        latent_dim=config["latent_dim"]
    ).to(device)

    optimizer = Adam(
        model.parameters(),
        lr=config["learning_rate"],
    )

    # ------------------------------------------
    # Checkpoint / resume
    # ------------------------------------------
    start_epoch = 1
    best_val = math.inf
    history = []

    history_path = output_dir / "history.csv"

    if args.resume is not None:
        checkpoint = torch.load(
            args.resume,
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(
            checkpoint["model_state"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state"]
        )

        start_epoch = checkpoint["epoch"] + 1
        best_val = checkpoint["best_val"]

        if history_path.exists():
            with open(
                history_path,
                "r",
                newline="",
                encoding="utf-8",
            ) as f:
                history = list(csv.DictReader(f))

            history = [
                row for row in history
                if int(row["epoch"]) < start_epoch
            ]

        print(f"Resuming at epoch {start_epoch}")

    # Same latent vectors for progress images.
    generator = torch.Generator().manual_seed(
        config["seed"]
    )

    fixed_z = torch.randn(
        16,
        config["latent_dim"],
        generator=generator,
    )

    # ------------------------------------------
    # Training loop
    # ------------------------------------------
    for epoch in range(start_epoch, config["epochs"] + 1):

        train_stats = run_epoch(
            model,
            train_loader,
            device,
            config["beta"],
            optimizer=optimizer,
            max_batches=args.max_train_batches,
        )

        val_stats = run_epoch(
            model,
            val_loader,
            device,
            config["beta"],
            max_batches=args.max_val_batches,
        )

        row = {
            "epoch": epoch,
            "train_total": train_stats["total"],
            "train_recon": train_stats["recon"],
            "train_kl": train_stats["kl"],
            "val_total": val_stats["total"],
            "val_recon": val_stats["recon"],
            "val_kl": val_stats["kl"],
        }

        history.append(row)
        write_history(history, history_path)

        improved = val_stats["total"] < best_val

        if improved:
            best_val = val_stats["total"]

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val": best_val,
            "config": config,
        }

        # Latest checkpoint for resuming.
        torch.save(
            checkpoint,
            checkpoint_dir / "last.pt",
        )

        # Best checkpoint for final generation.
        if improved:
            torch.save(
                checkpoint,
                checkpoint_dir / "best.pt",
            )

        print(
            f"Epoch {epoch:03d}/{config['epochs']:03d} | "
            f"Train: {train_stats['total']:.4f} | "
            f"Val: {val_stats['total']:.4f} | "
            f"Recon: {val_stats['recon']:.4f} | "
            f"KL: {val_stats['kl']:.4f}"
        )

        if (
            epoch % config["sample_every"] == 0
            or epoch == config["epochs"]
        ):
            save_progress_images(
                model,
                val_loader,
                fixed_z,
                device,
                output_dir,
                epoch,
            )

    print("\nTraining complete.")
    print(f"Best validation loss: {best_val:.4f}")
    print(f"Results saved in: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a convolutional VAE."
    )

    parser.add_argument(
        "--config", default="config.yaml"
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", default=None)

    # Optional fast smoke-test limits.
    parser.add_argument(
        "--max-train-batches", type=int, default=None
    )
    parser.add_argument(
        "--max-val-batches", type=int, default=None
    )

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
