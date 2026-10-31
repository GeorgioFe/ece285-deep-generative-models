
import torch
import torch.nn as nn


class ConvVAE(nn.Module):
    """Convolutional VAE for 128x128 grayscale images."""

    def __init__(self, latent_dim=128):
        super().__init__()

        self.latent_dim = latent_dim

        # ------------------------------------------
        # Encoder: image -> feature representation
        # ------------------------------------------
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Flatten(),
        )

        # Encoder output: 256 x 8 x 8
        self.feature_dim = 256 * 8 * 8

        # Predict distribution parameters
        self.fc_mu = nn.Linear(
            self.feature_dim, latent_dim
        )

        self.fc_logvar = nn.Linear(
            self.feature_dim, latent_dim
        )

        # ------------------------------------------
        # Decoder: latent vector -> image
        # ------------------------------------------
        self.decoder_input = nn.Linear(
            latent_dim, self.feature_dim
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        features = self.encoder(x)

        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)

        x = x.view(-1, 256, 8, 8)

        return self.decoder(x)

    def forward(self, x):
        mu, logvar = self.encode(x)

        z = self.reparameterize(mu, logvar)

        recon = self.decode(z)

        return recon, mu, logvar

    def reconstruct(self, x):
        """Deterministic reconstruction using latent mean."""
        mu, _ = self.encode(x)

        return self.decode(mu)

    def sample(self, num_samples, device):
        """Generate images by sampling the prior."""
        z = torch.randn(
            num_samples,
            self.latent_dim,
            device=device,
        )

        return self.decode(z)


def vae_loss(recon, x, mu, logvar, beta=1.0):
    """Negative ELBO with a fixed-variance Gaussian likelihood."""

    # Reconstruction error, summed across image pixels,
    # then averaged across the batch.
    recon_loss = (
        0.5 * (recon - x).pow(2)
    ).flatten(1).sum(dim=1).mean()

    # KL divergence against N(0, I).
    # Sum across latent dimensions, average across batch.
    kl_loss = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp(),
        dim=1,
    ).mean()

    total_loss = recon_loss + beta * kl_loss

    return {
        "total": total_loss,
        "recon": recon_loss,
        "kl": kl_loss,
    }


if __name__ == "__main__":
    # Quick architecture sanity check.
    model = ConvVAE(latent_dim=128)

    x = torch.randn(2, 1, 128, 128)

    recon, mu, logvar = model(x)

    assert recon.shape == x.shape
    assert mu.shape == (2, 128)
    assert logvar.shape == (2, 128)

    losses = vae_loss(recon, x, mu, logvar)
    losses["total"].backward()

    print("Model sanity check passed.")
    print("Input:", tuple(x.shape))
    print("Reconstruction:", tuple(recon.shape))
    print("Latent mean:", tuple(mu.shape))
    print("Total parameters:",
          sum(p.numel() for p in model.parameters()))
