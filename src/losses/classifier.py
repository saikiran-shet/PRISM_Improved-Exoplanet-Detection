import torch
import torch.nn as nn


class ClassifierLoss(nn.Module):
    """
    Weighted Binary Cross Entropy for planet / no-planet classification.

    pos_weight = n_negative / n_positive
        Computed from actual training data in train.py — never hardcoded.
        Down-weights the majority planet class so the classifier
        cannot cheat by always predicting planet.

    Example:
        150 non-planet, 550 planet → pos_weight = 150/550 = 0.27
        A wrong planet prediction is penalized 0.27× less than
        a wrong non-planet prediction — balancing the gradient signal.

    Label smoothing (0.1):
        Targets become {0.05, 0.95} instead of {0, 1}.
        Prevents the model from becoming overconfident on small datasets.
        Reduces overfitting on easy majority-class examples.

    prob   : (B, 1) — sigmoid output from PRISMClassifier
    labels : (B,)   — float 0.0 or 1.0
    """

    def __init__(self, pos_weight=1.0, smoothing=0.1):
        super().__init__()
        self.pos_weight = pos_weight
        self.smoothing  = smoothing

    def forward(self, prob, labels):
        labels = labels.view(-1, 1).float()

        # Label smoothing
        labels_smooth = labels * (1 - self.smoothing) + 0.5 * self.smoothing

        # Clamp to avoid log(0)
        prob = prob.clamp(1e-7, 1 - 1e-7)

        loss = -(
            self.pos_weight * labels_smooth       * torch.log(prob)
            +                 (1 - labels_smooth) * torch.log(1 - prob)
        )
        return loss.mean()


if __name__ == "__main__":
    loss_fn = ClassifierLoss(pos_weight=0.27, smoothing=0.1)

    # Perfect predictions
    prob_perfect   = torch.tensor([[0.99], [0.01], [0.98], [0.02]])
    labels_perfect = torch.tensor([1.0, 0.0, 1.0, 0.0])
    l_perfect = loss_fn(prob_perfect, labels_perfect)
    print(f"L_classify (perfect) : {l_perfect.item():.4f}  ← expect near 0")

    # Random predictions
    prob_random   = torch.rand(8, 1)
    labels_random = torch.randint(0, 2, (8,)).float()
    l_random = loss_fn(prob_random, labels_random)
    print(f"L_classify (random)  : {l_random.item():.4f}  ← expect ~0.5-0.8")

    # All wrong
    prob_wrong   = torch.tensor([[0.01], [0.99]])
    labels_wrong = torch.tensor([1.0, 0.0])
    l_wrong = loss_fn(prob_wrong, labels_wrong)
    print(f"L_classify (wrong)   : {l_wrong.item():.4f}  ← expect high")
    print("classifier.py smoke test passed.")