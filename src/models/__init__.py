from src.losses.reconstruction import ReconstructionLoss
from src.losses.mine           import MINELoss
from src.losses.physics        import PhysicsLoss
from src.losses.classifier     import WeightedBCELoss


def build_losses(latent_dim=64,
                 seq_len=1024,
                 lambda_tv=0.1,
                 pos_weight=1.0):

    return dict(

        recon = ReconstructionLoss(),

        mine = MINELoss(
            latent_dim=latent_dim
        ),

        physics = PhysicsLoss(
            seq_len=seq_len,
            lambda_tv=lambda_tv
        ),

        classify = WeightedBCELoss(
            pos_weight=pos_weight,
            smoothing=0.1
        ),
    )