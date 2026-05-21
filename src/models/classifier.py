import torch
import torch.nn as nn


class PRISMClassifier(nn.Module):
    """
    Maps z_t → planet probability with uncertainty via MC Dropout.

    Input  : z_t (batch, latent_dim)
    Output : probability (batch, 1) in [0, 1]

    Final Sigmoid ensures output is always a valid probability.
    ClassifierLoss uses raw probabilities (not logits) so Sigmoid
    must stay here — do NOT remove it.
    """

    def __init__(self, latent_dim=64, hidden_dim=128, dropout_p=0.3):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),

            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()             # ← must be here — output in [0, 1]
        )

    def forward(self, z_t):
        """
        z_t    : (batch, latent_dim)
        returns: (batch, 1) in [0, 1]
        """
        return self.classifier(z_t)

    def predict_with_uncertainty(self, z_t, n_passes=20):
        """
        MC Dropout inference.
        Keeps dropout active across n_passes forward passes.

        Returns:
            mean : (batch, 1) — planet probability
            std  : (batch, 1) — uncertainty estimate
        """
        self._enable_dropout()

        preds = []
        with torch.no_grad():
            for _ in range(n_passes):
                preds.append(self.classifier(z_t))

        preds = torch.stack(preds, dim=0)   # (n_passes, B, 1)
        mean  = preds.mean(dim=0)           # (B, 1)
        std   = preds.std(dim=0)            # (B, 1)

        self.eval()
        return mean, std

    def _enable_dropout(self):
        """Sets only Dropout layers to train mode for MC Dropout."""
        self.eval()
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()


if __name__ == "__main__":
    clf = PRISMClassifier(latent_dim=64)
    clf.eval()

    z_t  = torch.randn(8, 64)
    prob = clf(z_t)

    print(f"z_t shape  : {z_t.shape}")
    print(f"prob shape : {prob.shape}")
    print(f"prob range : [{prob.min():.4f}, {prob.max():.4f}]  ← must be [0,1]")

    assert prob.min() >= 0.0 and prob.max() <= 1.0, "Sigmoid missing!"

    mean, std = clf.predict_with_uncertainty(z_t, n_passes=20)
    print(f"MC mean    : {mean[0].item():.4f} ± {std[0].item():.4f}")
    print("classifier.py smoke test passed.")