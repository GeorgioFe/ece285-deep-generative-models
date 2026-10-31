# Chest X-Ray Generation Using a Variational Autoencoder

ECE 285 — Homework 1

## Overview

This project implements a convolutional Variational Autoencoder
(VAE) in PyTorch to generate synthetic chest X-ray images.

The model is trained on the Chest X-Ray Pneumonia dataset.
It learns an unconditional generative distribution from both
NORMAL and PNEUMONIA images.

## Dataset

Source:
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Expected structure:

data/chest_xray/
    train/
        NORMAL/
        PNEUMONIA/
    val/
        NORMAL/
        PNEUMONIA/
    test/
        NORMAL/
        PNEUMONIA/

The model uses a stratified 85/15 split of the supplied
training folder for training and validation.

The supplied test folder is reserved for final evaluation.

## Model

- Architecture: Convolutional VAE
- Input: 128x128 grayscale images
- Latent dimension: 128
- Prior: Standard normal N(0, I)
- Reconstruction: Fixed-variance Gaussian observation model
- Objective: Reconstruction loss + beta * KL divergence

## Environment

Recommended: Python 3.11

On Windows with an NVIDIA GPU:

1. Create a virtual environment:

   py -3.11 -m venv .venv

2. Activate it in PowerShell:

   .\.venv\Scripts\Activate.ps1

3. Install CUDA-enabled PyTorch:

   python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

4. Install other dependencies:

   python -m pip install -r requirements.txt

## Commands

Run all commands from the project root.

### Model sanity check

python -m src.model

### Training

python -m src.train --config config.yaml --device auto

### Resume training

python -m src.train --config config.yaml --resume outputs/baseline/checkpoints/last.pt

### Generate images

python -m src.generate --config config.yaml --checkpoint outputs/baseline/checkpoints/best.pt --num-samples 25

### Evaluate FID and Inception Score

python -m src.evaluate --config config.yaml --checkpoint outputs/baseline/checkpoints/best.pt

### Create figures

python -m scripts.make_figures --config config.yaml --checkpoint outputs/baseline/checkpoints/best.pt

## Outputs

Training saves:

- checkpoints/best.pt
- checkpoints/last.pt
- history.csv
- config_used.yaml
- figures/

Evaluation saves:

- metrics.json

## Evaluation

We use PyTorch-Ignite's default InceptionV3-based
FID and Inception Score implementations.

FID compares independent generated samples against
the supplied held-out test images.

These metrics are limited for medical images because
the underlying Inception model is trained on ImageNet.

## Reproducibility

Random seeds and experimental parameters are
recorded in the configuration.

The same Python code supports CUDA, MPS, and CPU,
although platform-specific PyTorch installations
may be required.

## References

Kingma and Welling, Auto-Encoding Variational Bayes (2013).

Kermany et al., Identifying Medical Diagnoses and Treatable
Diseases by Image-Based Deep Learning (2018).

PyTorch-Ignite:
https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/
