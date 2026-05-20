import torch
import torch.nn as nn

class PRISMEncoder(nn.Module):
    """
    Shared 1D CNN encoder with dual latent heads.

    Input  : (batch, 1, 1024)  — 1 channel, 1024 time steps
    Output : z_s (batch, latent_dim)  — stellar activity summary
             z_t (batch, latent_dim)  — transit signal summary

    Architecture:
        4 Conv1d blocks (conv → batchnorm → ReLU → maxpool)
        Each block halves the sequence length and doubles channels:
        (1, 1024) → (32, 512) → (64, 256) → (128, 128) → (256, 64)
        Flatten → two independent linear heads → z_s, z_t

    Why Conv1d and not Transformer?
        Conv1d trains in seconds on CPU, easy to debug, strong inductive bias
        for local patterns (exactly what transits are — localized dips).
        Transformer can be swapped in later once this baseline works.

    Why BatchNorm?
        Light curves from different stars have different noise levels.
        BatchNorm normalizes activations per-batch so the network doesn't
        have to learn to ignore scale differences.

    Why two separate heads instead of splitting one vector?
        Separate linear layers mean the gradient from L_MI can push z_s
        and z_t apart independently without fighting over shared weights.
    """

    def __init__(self, in_channels=1, latent_dim=64, seq_len=1024):
        super().__init__()

        # ── Shared convolutional backbone ────────────────────────────────────
        # Each block: Conv1d → BatchNorm1d → ReLU → MaxPool1d(2)
        # kernel_size=7 captures ~7 cadence (~2hr) local patterns
        # padding=3 keeps length stable before pooling
        self.backbone = nn.Sequential(
            # Block 1: (B, 1,   1024) → (B, 32,  512)
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 2: (B, 32,  512) → (B, 64,  256)
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 3: (B, 64,  256) → (B, 128, 128)
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 4: (B, 128, 128) → (B, 256, 64)
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # Compute flattened size dynamically — robust to seq_len changes
        flat_size = self._get_flat_size(in_channels, seq_len)

        # ── Dual latent heads ────────────────────────────────────────────────
        # z_s head: stellar activity — learns slow, smooth variability patterns
        self.stellar_head = nn.Sequential(
            nn.Linear(flat_size, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

        # z_t head: transit signal — learns sharp, localized dip patterns
        self.transit_head = nn.Sequential(
            nn.Linear(flat_size, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

    def _get_flat_size(self, in_channels, seq_len):
        """Dry-run a dummy tensor to get the flattened backbone output size."""
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, seq_len)
            out   = self.backbone(dummy)
            return out.view(1, -1).shape[1]

    def forward(self, x):
        """
        x      : (batch, 1, seq_len)
        returns: z_s (batch, latent_dim), z_t (batch, latent_dim)
        """
        features = self.backbone(x)          # (B, 256, 64)
        features = features.view(features.size(0), -1)   # (B, 16384) flatten

        z_s = self.stellar_head(features)    # (B, latent_dim)
        z_t = self.transit_head(features)    # (B, latent_dim)

        return z_s, z_t


if __name__ == "__main__":
    model = PRISMEncoder(latent_dim=64)
    x     = torch.randn(8, 1, 1024)          # batch of 8 light curves
    z_s, z_t = model(x)
    print(f"Input  : {x.shape}")
    print(f"z_s    : {z_s.shape}")           # (8, 64)
    print(f"z_t    : {z_t.shape}")           # (8, 64)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Params : {total_params:,}")
    print("encoder.py smoke test passed.")