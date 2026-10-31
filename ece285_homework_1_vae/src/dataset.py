from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


class ResizeWithPadding:
    """Resize an image without distorting its aspect ratio."""

    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        image = image.convert("L")

        return ImageOps.pad(
            image,
            (self.size, self.size),
            method=Image.Resampling.BICUBIC,
            color=0,
        )


def get_transform(image_size):
    """Return the preprocessing pipeline."""

    return transforms.Compose([
        ResizeWithPadding(image_size),
        transforms.ToTensor(),
    ])


def load_image_folder(folder, image_size):
    """Load images from NORMAL and PNEUMONIA subfolders."""

    folder = Path(folder)

    if not folder.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {folder}\n"
            "Check data_dir in config.yaml."
        )

    return datasets.ImageFolder(
        root=str(folder),
        transform=get_transform(image_size),
    )


def split_train_validation(dataset, val_fraction, seed):
    """Create reproducible, class-stratified subsets."""

    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1.")

    generator = torch.Generator().manual_seed(seed)

    train_indices = []
    val_indices = []

    for class_id in sorted(set(dataset.targets)):
        indices = [
            i for i, label in enumerate(dataset.targets)
            if label == class_id
        ]

        perm = torch.randperm(
            len(indices),
            generator=generator,
        ).tolist()

        shuffled = [indices[i] for i in perm]

        n_val = max(1, round(len(shuffled) * val_fraction))

        val_indices.extend(shuffled[:n_val])
        train_indices.extend(shuffled[n_val:])

    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
    )


def make_train_val_loaders(config, device):
    """Build training and validation DataLoaders."""

    data_dir = Path(config["data_dir"])

    full_train = load_image_folder(
        data_dir / "train",
        config["image_size"],
    )

    train_dataset, val_dataset = split_train_validation(
        full_train,
        config["val_fraction"],
        config["seed"],
    )

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=pin_memory,
        generator=torch.Generator().manual_seed(config["seed"]),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=pin_memory,
    )

    print(f"Training images:   {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")

    return train_loader, val_loader


def make_test_loader(config, device, batch_size=None):
    """Load the untouched test dataset."""

    test_dataset = load_image_folder(
        Path(config["data_dir"]) / "test",
        config["image_size"],
    )

    loader = DataLoader(
        test_dataset,
        batch_size=batch_size or config["eval_batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    print(f"Test images: {len(test_dataset)}")

    return loader
