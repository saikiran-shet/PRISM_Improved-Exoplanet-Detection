import torch
import torch.nn as nn

class PRISMDecoder(nn.Module):
    """
    Decodes a latent vector z back into a 1D signal of length seq_len.

    Used twice:
        stellar_decoder(z_s) → S(t)   shape (B, 1, 1024)
        transit_decoder(z_t) → T(t)   shape (B, 1, 1024)

    Architecture:
        Linear expansion: latent_dim → 256 → 256*64 (match encoder bottleneck)
        Reshape to (B, 256, 64)
        4 ConvTranspose1d blocks upsample back to (B, 1, 1024)

    Why ConvTranspose1d?
        It's the natural inverse of Conv1d+MaxPool. It learns how to
        "spread" a compressed representation back into a longer sequence.
        The upsampling mirrors the encoder's downsampling exactly.

    Why separate decoders instead of one shared decoder?
        S(t) should be smooth and slow-varying.
        T(t) should be sparse with sharp dips.
        Separate decoders let each one specialize — the physics losses
        then enforce these different structural properties independently.

    Output activation: Sigmoid
        Forces output to [0, 1] — same range as our normalized input X(t).
        Without this, reconstructed values could be arbitrary floats and
        the reconstruction loss would be dominated by scale errors.
    """

    def __init__(self, latent_dim=64, out_channels=1, seq_len=1024):
        super().__init__()

        # The encoder bottleneck is (256 channels, seq_len/16 time steps)
        self.bottleneck_channels = 256
        self.bottleneck_len      = seq_len // 16   # 1024 // 16 = 64

        # ── Linear expansion from latent space to bottleneck ─────────────────
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.bottleneck_channels * self.bottleneck_len),
            nn.ReLU(),
        )

        # ── Transposed conv upsampling blocks ────────────────────────────────
        # Each block doubles sequence length, halves channels
        # (B, 256, 64) → (B, 128, 128) → (B, 64, 256) → (B, 32, 512) → (B, 1, 1024)
        self.upsampler = nn.Sequential(
            # Block 1: 64 → 128
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            # Block 2: 128 → 256
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            # Block 3: 256 → 512
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            # Block 4: 512 → 1024, collapse to 1 output channel
            nn.ConvTranspose1d(32, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),   # output in [0, 1]
        )

    def forward(self, z):
        """
        z      : (batch, latent_dim)
        returns: signal (batch, 1, seq_len)
        """
        x = self.fc(z)                                         # (B, 256*64)
        x = x.view(x.size(0),
                   self.bottleneck_channels,
                   self.bottleneck_len)                        # (B, 256, 64)
        x = self.upsampler(x)                                  # (B, 1, 1024)
        return x


def build_decoders(latent_dim=64, seq_len=1024):
    """
    Convenience function — returns both decoders as a named tuple.
    Both have identical architecture but separate weights.
    """
    stellar_decoder = PRISMDecoder(latent_dim=latent_dim, seq_len=seq_len)
    transit_decoder = PRISMDecoder(latent_dim=latent_dim, seq_len=seq_len)
    return stellar_decoder, transit_decoder


if __name__ == "__main__":
    stellar_dec, transit_dec = build_decoders(latent_dim=64)

    z_s = torch.randn(8, 64)
    z_t = torch.randn(8, 64)

    S_t = stellar_dec(z_s)
    T_t = transit_dec(z_t)

    print(f"z_s    : {z_s.shape}")
    print(f"z_t    : {z_t.shape}")
    print(f"S(t)   : {S_t.shape}")          # (8, 1, 1024)
    print(f"T(t)   : {T_t.shape}")          # (8, 1, 1024)
    print(f"S+T range: [{(S_t+T_t).min():.3f}, {(S_t+T_t).max():.3f}]")

    p_s = sum(p.numel() for p in stellar_dec.parameters())
    p_t = sum(p.numel() for p in transit_dec.parameters())
    print(f"Params (each decoder): {p_s:,}")
    print("decoder.py smoke test passed.")