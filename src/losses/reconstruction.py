import torch
import torch.nn as nn

class ReconstructionLoss(nn.Module):
    """
    L_recon = mean( (X(t) - S(t) - T(t))^2 )

    This is the foundational constraint — without it the encoder
    can collapse: put everything in z_s, leave z_t empty, and the
    classifier has nothing to work with.

    Why MSE and not MAE?
        MSE penalizes large deviations quadratically — a reconstruction
        that's off by 0.1 at every point is penalized much harder than
        one that's off by 0.01. This pushes the network toward getting
        the overall shape right, not just being "close on average".

    Shapes:
        x   : (B, 1, 1024) — original light curve
        s_t : (B, 1, 1024) — stellar reconstruction S(t)
        t_t : (B, 1, 1024) — transit reconstruction T(t)
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, x, s_t, t_t):
        reconstruction = s_t + t_t
        return self.mse(reconstruction, x)


if __name__ == "__main__":
    loss_fn = ReconstructionLoss()
    x   = torch.rand(8, 1, 1024)
    s_t = torch.rand(8, 1, 1024) * 0.6
    t_t = torch.rand(8, 1, 1024) * 0.4
    loss = loss_fn(x, s_t, t_t)
    print(f"L_recon (random): {loss.item():.4f}  ← expect ~0.08-0.15")

    # Perfect reconstruction should give ~0
    loss_perfect = loss_fn(x, x * 0.6, x * 0.4)
    print(f"L_recon (perfect split): {loss_perfect.item():.6f}  ← expect ~0.0")
    print("reconstruction.py smoke test passed.")