from src.models.encoder    import PRISMEncoder
from src.models.decoder    import PRISMDecoder, build_decoders
from src.models.classifier import PRISMClassifier

def build_prism(latent_dim=64, seq_len=1024, dropout_p=0.3):
    """
    Instantiates the full PRISM model — all four components.
    Returns a dict for easy access in the training loop.
    """
    encoder             = PRISMEncoder(latent_dim=latent_dim, seq_len=seq_len)
    stellar_dec, \
    transit_dec         = build_decoders(latent_dim=latent_dim, seq_len=seq_len)
    classifier          = PRISMClassifier(latent_dim=latent_dim,
                                          dropout_p=dropout_p)
    return dict(
        encoder         = encoder,
        stellar_decoder = stellar_dec,
        transit_decoder = transit_dec,
        classifier      = classifier,
    )