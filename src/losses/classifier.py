import torch
import torch.nn as nn


class ClassifierLoss(nn.Module):
    """
    Weighted BCE for planet/no-planet classification.

    Expects prob in [0,1] — Sigmoid applied inside PRISMClassifier.
    pos_weight = n_negative / n_positive — computed from training data.
    Label smoothing prevents overconfidence on small datasets.
    """

    def __init__(self, pos_weight=1.0, smoothing=0.1):
        super().__init__()
        self.pos_weight = pos_weight
        self.smoothing  = smoothing

    def forward(self, prob, labels):
        """
        prob   : (B, 1) — sigmoid output, values in [0, 1]
        labels : (B,)   — float 0.0 or 1.0
        """
        labels = labels.view(-1, 1).float()
        labels_smooth = labels * (1 - self.smoothing) + 0.5 * self.smoothing
        prob   = prob.clamp(1e-7, 1 - 1e-7)

        loss = -(
            self.pos_weight * labels_smooth       * torch.log(prob)
            +                 (1 - labels_smooth) * torch.log(1 - prob)
        )
        return loss.mean()


if __name__ == "__main__":
    loss_fn = ClassifierLoss(pos_weight=0.27)

    prob_perfect   = torch.tensor([[0.99], [0.01], [0.98], [0.02]])
    labels_perfect = torch.tensor([1.0, 0.0, 1.0, 0.0])
    print(f"Perfect: {loss_fn(prob_perfect, labels_perfect).item():.4f}")

    prob_wrong   = torch.tensor([[0.01], [0.99]])
    labels_wrong = torch.tensor([1.0, 0.0])
    print(f"Wrong  : {loss_fn(prob_wrong, labels_wrong).item():.4f}")
    print("classifier.py smoke test passed.")
