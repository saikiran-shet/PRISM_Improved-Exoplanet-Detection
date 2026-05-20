import torch
import torch.nn as nn

class PRISMClassifier(nn.Module):
    """
    Maps z_t → planet probability with uncertainty estimation via MC Dropout.

    Input  : z_t (batch, latent_dim)
    Output : probability scalar (batch, 1)  in [0, 1]

    Architecture:
        3 fully-connected layers with dropout between each.
        Dropout rate 0.3 — aggressive enough to give meaningful uncertainty
        estimates, not so aggressive it prevents learning.

    MC Dropout — how it works:
        Normal inference: dropout OFF → single deterministic output
        MC Dropout:       dropout ON  → run N times → mean ± std

        PyTorch's nn.Dropout is OFF during model.eval() by default.
        To enable MC Dropout at test time, we call model.train() only on
        the dropout layers, or use the enable_dropout() helper below.

    Why only z_t and not z_s?
        This is the core architectural claim of PRISM.
        By the time z_t reaches the classifier, the MINE loss has forced
        stellar activity OUT of it. The classifier sees only transit information.
        Using z_s too would re-introduce the stellar noise we worked to remove.
    """

    def __init__(self, latent_dim=64, hidden_dim=128, dropout_p=0.3):
        super().__init__()

        self.classifier = nn.Sequential(
            # Layer 1: latent_dim → hidden_dim
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),

            # Layer 2: hidden_dim → hidden_dim/2
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),

            # Layer 3: hidden_dim/2 → 1 probability
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, z_t):
        """
        z_t    : (batch, latent_dim)
        returns: (batch, 1) probability in [0, 1]
        """
        return self.classifier(z_t)

    def predict_with_uncertainty(self, z_t, n_passes=20):
        """
        MC Dropout inference — keeps dropout active across n_passes forward passes.

        Steps:
        1. Enable dropout layers (set them to train mode)
        2. Run the same z_t through the classifier n_passes times
        3. Stack results → compute mean (prediction) and std (uncertainty)

        Returns:
            mean : (batch, 1) — the actual planet probability prediction
            std  : (batch, 1) — uncertainty; high std = model is unsure

        Interpretation:
            mean=0.89, std=0.02 → confident planet
            mean=0.72, std=0.18 → uncertain → flag for human review
            mean=0.11, std=0.03 → confident non-planet
        """
        # Enable dropout for uncertainty estimation
        self._enable_dropout()

        preds = []
        with torch.no_grad():
            for _ in range(n_passes):
                pred = self.classifier(z_t)   # (B, 1)
                preds.append(pred)

        preds = torch.stack(preds, dim=0)     # (n_passes, B, 1)
        mean  = preds.mean(dim=0)             # (B, 1)
        std   = preds.std(dim=0)              # (B, 1)

        # Restore eval mode
        self.eval()

        return mean, std

    def _enable_dropout(self):
        """Sets only Dropout layers to train mode, leaves everything else in eval."""
        self.eval()
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()


if __name__ == "__main__":
    clf = PRISMClassifier(latent_dim=64)
    clf.eval()

    z_t = torch.randn(8, 64)

    # Standard prediction
    prob = clf(z_t)
    print(f"z_t shape       : {z_t.shape}")
    print(f"Prob shape      : {prob.shape}")         # (8, 1)
    print(f"Prob range      : [{prob.min():.3f}, {prob.max():.3f}]")

    # MC Dropout uncertainty
    mean, std = clf.predict_with_uncertainty(z_t, n_passes=20)
    print(f"MC mean shape   : {mean.shape}")         # (8, 1)
    print(f"MC std shape    : {std.shape}")           # (8, 1)
    print(f"Sample mean/std : {mean[0].item():.3f} ± {std[0].item():.3f}")

    p = sum(p.numel() for p in clf.parameters())
    print(f"Params          : {p:,}")
    print("classifier.py smoke test passed.")