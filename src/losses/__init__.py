from src.losses.reconstruction import ReconstructionLoss
from src.losses.mine           import MINELoss, MINENetwork
from src.losses.physics        import PhysicsLoss, MandelAgolLoss, TotalVariationLoss
from src.losses.classifier     import ClassifierLoss

def build_losses(latent_dim=64, seq_len=1024,
                 lambda_tv=0.1, pos_weight=1.0):
    """
    Instantiates all four losses.
    pos_weight passed to ClassifierLoss to handle class imbalance.
    Computed from actual training data in train.py — never hardcoded.
    """
    return dict(
        recon    = ReconstructionLoss(),
        mine     = MINELoss(latent_dim=latent_dim),
        physics  = PhysicsLoss(seq_len=seq_len, lambda_tv=lambda_tv),
        classify = ClassifierLoss(pos_weight=pos_weight),
    )