import torch
import torch.nn as nn

class PRISMDecoder(nn.Module):
    """
    Decodes a latent vector z back into a 1D signal of length seq_len.

    Used twice:
        stellar_decoder(z_s) → S(t)   shape (B, 1, 1024)
        transit_decoder(z_t) → T(t)   shape (B, 1, 1024)

    Bottleneck channels reduced to 128 (from 256) to match the
    smaller encoder and prevent overfitting on small datasets.
    """

    def __init__(self, latent_dim=64, out_channels=1, seq_len=1024):
        super().__init__()

        self.bottleneck_channels = 128         # must match upsampler input
        self.bottleneck_len      = seq_len // 16   # 1024 // 16 = 64

        # Linear expansion from latent space to bottleneck
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.bottleneck_channels * self.bottleneck_len),
            nn.ReLU(),
        )

        # Transposed conv upsampling — mirrors the smaller encoder
        # (B, 128, 64) → (B, 64, 128) → (B, 32, 256) → (B, 16, 512) → (B, 1, 1024)
        self.upsampler = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.ConvTranspose1d(16, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        """
        z      : (batch, latent_dim)
        returns: signal (batch, 1, seq_len)
        """
        x = self.fc(z)
        x = x.view(x.size(0),
                   self.bottleneck_channels,
                   self.bottleneck_len)
        x = self.upsampler(x)
        return x


def build_decoders(latent_dim=64, seq_len=1024):
    """Returns stellar and transit decoders — identical architecture, separate weights."""
    stellar_decoder = PRISMDecoder(latent_dim=latent_dim, seq_len=seq_len)
    transit_decoder = PRISMDecoder(latent_dim=latent_dim, seq_len=seq_len)
    return stellar_decoder, transit_decoder


if __name__ == "__main__":
    stellar_dec, transit_dec = build_decoders(latent_dim=64)

    z_s = torch.randn(8, 64)
    z_t = torch.randn(8, 64)

    S_t = stellar_dec(z_s)
    T_t = transit_dec(z_t)

    print(f"z_s  : {z_s.shape}")
    print(f"z_t  : {z_t.shape}")
    print(f"S(t) : {S_t.shape}")
    print(f"T(t) : {T_t.shape}")
    assert S_t.shape == (8, 1, 1024), f"Wrong shape: {S_t.shape}"
    assert T_t.shape == (8, 1, 1024), f"Wrong shape: {T_t.shape}"
    print("decoder.py smoke test passed.")