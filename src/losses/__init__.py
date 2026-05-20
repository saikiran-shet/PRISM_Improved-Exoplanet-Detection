from src.losses.reconstruction import ReconstructionLoss
from src.losses.mine           import MINELoss, MINENetwork
from src.losses.physics        import PhysicsLoss, MandelAgolLoss, TotalVariationLoss
from src.losses.classifier     import ClassifierLoss

def build_losses(latent_dim=64, seq_len=1024,
                 lambda_tv=0.1, pos_weight=1.0):
    """
    Instantiates all four losses.
    Returns a dict used directly in the training loop.

    Lambda weights (how much each loss contributes to total):
        Set in train.py — not here. These are just the loss objects.
    """
    return dict(
        recon   = ReconstructionLoss(),
        mine    = MINELoss(latent_dim=latent_dim),
        physics = PhysicsLoss(seq_len=seq_len, lambda_tv=lambda_tv),
        classify = ClassifierLoss(pos_weight=pos_weight),
    )