import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(config_path):
    """Read experiment settings from YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config, path):
    """Save the configuration used in an experiment."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def get_device(requested="auto"):
    """Choose CUDA, Apple MPS, or CPU."""

    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            requested = "mps"
        else:
            requested = "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable.")

    return torch.device(requested)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_trained_model(config, checkpoint_path, device):
    """Load a trained VAE from a trusted local checkpoint."""
    from src.model import ConvVAE

    model = ConvVAE(
        latent_dim=config["latent_dim"]
    ).to(device)

    # Only load checkpoints created by this project
    # or another source you trust.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, checkpoint
