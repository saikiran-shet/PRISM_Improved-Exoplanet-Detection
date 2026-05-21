import torch
import torch.nn as nn

class ClassifierLoss(nn.Module):
    """
    L_classify = Binary Cross Entropy between predicted probability and true label.

    BCE(p, y) = -[y * log(p) + (1-y) * log(1-p)]

    y=1 (planet)    → pushes p toward 1.0
    y=0 (no planet) → pushes p toward 0.0

    pos_weight handles class imbalance:
        If we have 550 planet curves (50 real + 500 injected) and 50 non-planet,
        the model could get 92% accuracy by always predicting "planet".
        pos_weight=1.0 treats both classes equally — adjust if imbalance is severe.

    Why BCE and not focal loss?
        BCE is the standard baseline. Once you have results, if the model
        is over-confident on easy examples you can switch to focal loss.
        Keep it simple for the first working version.

    prob   : (B, 1) — sigmoid output from classifier
    labels : (B,)   — float 0.0 or 1.0
    """

    def __init__(self, pos_weight=1.0):
        super().__init__()
        # pos_weight < 1 down-weights the planet class
        # 98 non-planet / 542 planet = 0.18
        pw = torch.tensor([pos_weight])
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)

    def forward(self, prob, labels):
        labels = labels.view(-1, 1)
        return self.bce(prob, labels)


if __name__ == "__main__":
    loss_fn = ClassifierLoss(pos_weight=1.0)

    # Perfect predictions
    prob_perfect   = torch.tensor([[0.99], [0.01], [0.98], [0.02]])
    labels_perfect = torch.tensor([1.0, 0.0, 1.0, 0.0])
    l_perfect = loss_fn(prob_perfect, labels_perfect)
    print(f"L_classify (perfect) : {l_perfect.item():.4f}  ← expect near 0")

    # Random predictions
    prob_random   = torch.rand(8, 1)
    labels_random = torch.randint(0, 2, (8,)).float()
    l_random = loss_fn(prob_random, labels_random)
    print(f"L_classify (random)  : {l_random.item():.4f}  ← expect ~0.6-0.8")

    # Wrong predictions
    prob_wrong   = torch.tensor([[0.01], [0.99]])
    labels_wrong = torch.tensor([1.0, 0.0])
    l_wrong = loss_fn(prob_wrong, labels_wrong)
    print(f"L_classify (wrong)   : {l_wrong.item():.4f}  ← expect high")
    print("classifier.py smoke test passed.")