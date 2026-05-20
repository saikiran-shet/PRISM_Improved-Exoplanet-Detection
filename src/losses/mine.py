import torch
import torch.nn as nn

class MINENetwork(nn.Module):
    """
    Auxiliary network for MINE (Mutual Information Neural Estimation).
    Donsker-Varadhan representation of mutual information.

    Takes a (z_s, z_t) pair — either real (from same curve) or shuffled
    (z_t from a different curve) — and outputs a scalar score.

    High score for real pairs, low score for shuffled = detected dependence.
    We train PRISM to make this network score real and shuffled pairs equally.

    Architecture: simple 3-layer MLP on concatenated [z_s; z_t]
    Input:  (batch, latent_dim * 2)  — z_s and z_t concatenated
    Output: (batch, 1)               — scalar score
    """

    def __init__(self, latent_dim=64, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
            # No activation — raw logits needed for DV bound
        )

    def forward(self, z_s, z_t):
        """Concatenate and score the pair."""
        pair = torch.cat([z_s, z_t], dim=-1)   # (B, latent_dim*2)
        return self.net(pair)                    # (B, 1)


class MINELoss(nn.Module):
    """
    Computes the MINE lower bound on mutual information I(z_s; z_t).

    Donsker-Varadhan (DV) bound:
        I(z_s; z_t) >= E[f(z_s, z_t)] - log(E[e^f(z_s, z_t')])

    where:
        f          = the MINE network
        (z_s, z_t) = real pairs      (from same light curve)
        z_t'       = shuffled z_t    (from a different curve in the batch)

    The DV bound is always <= true MI. We maximize it to get a tight estimate,
    then minimize it from PRISM's perspective to reduce MI.

    Two separate optimizers:
        mine_optimizer : maximizes the bound (trains MINENetwork to detect MI)
        prism_optimizer: minimizes the bound (trains encoder to reduce MI)

    This adversarial dynamic is what makes MINE principled rather than heuristic.

    How shuffling works:
        z_t' = z_t[torch.randperm(B)]
        This creates "marginal" samples — z_s and z_t' are from different
        curves so any score correlation must be due to learned dependence,
        not just shared input.

    Returns:
        mi_estimate : scalar — the MI lower bound (minimize this in PRISM)
    """

    def __init__(self, latent_dim=64, hidden_dim=128):
        super().__init__()
        self.mine_net = MINENetwork(latent_dim=latent_dim, hidden_dim=hidden_dim)

    def forward(self, z_s, z_t):
        """
        z_s : (B, latent_dim) — stellar latent vectors
        z_t : (B, latent_dim) — transit latent vectors
        Both from the SAME batch of light curves.
        """
        B = z_s.size(0)

        # ── Real pairs: z_s and z_t from the same curve ──────────────────────
        score_real = self.mine_net(z_s, z_t)            # (B, 1)

        # ── Shuffled pairs: z_t from a DIFFERENT curve in the batch ──────────
        perm     = torch.randperm(B, device=z_t.device)
        z_t_shuf = z_t[perm]
        score_shuf = self.mine_net(z_s, z_t_shuf)       # (B, 1)

        # ── DV bound: E[f(real)] - log(E[e^f(shuffled)]) ─────────────────────
        # E[f(real)]           → mean over batch
        # log(E[e^f(shuffled)])→ log-sum-exp trick for numerical stability
        t_real  = score_real.mean()
        t_shuf  = torch.logsumexp(score_shuf, dim=0) - torch.log(
                      torch.tensor(float(B), device=z_t.device)
                  )

        mi_estimate = t_real - t_shuf

        # PRISM wants to MINIMIZE MI (make z_s ⊥ z_t).
        # The DV bound is a LOWER bound — maximizing it tightens the estimate.
        # We return it as-is:
        #   - MINENetwork optimizer: maximizes this  (gradient ascent)
        #   - PRISM encoder optimizer: minimizes this (gradient descent)
        return mi_estimate

    def get_mine_parameters(self):
        """Returns only the MINE network params — for the separate MINE optimizer."""
        return self.mine_net.parameters()


if __name__ == "__main__":
    mine = MINELoss(latent_dim=64)

    # Correlated z_s, z_t — MI should be high
    z_s_corr = torch.randn(32, 64)
    z_t_corr = z_s_corr + 0.1 * torch.randn(32, 64)   # almost identical
    mi_high = mine(z_s_corr, z_t_corr)
    print(f"MI (correlated)   : {mi_high.item():.4f}  ← expect positive / high")

    # Independent z_s, z_t — MI should be near 0
    z_s_ind = torch.randn(32, 64)
    z_t_ind = torch.randn(32, 64)
    mi_low = mine(z_s_ind, z_t_ind)
    print(f"MI (independent)  : {mi_low.item():.4f}  ← expect near 0 or negative")

    params = sum(p.numel() for p in mine.parameters())
    print(f"MINE params       : {params:,}")
    print("mine.py smoke test passed.")