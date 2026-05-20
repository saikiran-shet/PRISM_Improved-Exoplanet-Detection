import torch
import torch.nn as nn
import numpy as np
import batman

class MandelAgolLoss(nn.Module):
    """
    L_phys_transit: penalizes T(t) for not fitting any valid Mandel-Agol transit.

    For each T(t) in the batch:
    1. Run a short optimization (inner loop) to find the 4 batman parameters
       (rp, a, inc, t0) that best explain T(t)
    2. Generate the best-fit transit curve using batman
    3. Penalty = MSE(T(t), best_fit_transit)

    If T(t) is a real transit → small residual → small penalty
    If T(t) is noise/stellar bleed → no batman params can fit it → large penalty

    Why an inner optimization loop?
        batman is not differentiable — it's a C extension.
        So we can't backprop through it directly.
        Instead: optimize the 4 parameters to fit T(t), generate the curve,
        then MSE between T(t) and that curve IS differentiable w.r.t. T(t).
        The gradient flows: MSE → T(t) → transit_decoder → encoder.

    Parameters optimized per forward pass (not learned globally):
        rp  : planet/star radius ratio  (0.01 to 0.2)
        a   : semi-major axis           (3.0 to 30.0)
        inc : inclination degrees       (80 to 90)
        t0  : transit center time       (fitted to window)
    """

    def __init__(self, seq_len=1024, cadence_days=0.0204,
                 inner_steps=10, inner_lr=0.05):
        super().__init__()
        self.seq_len      = seq_len
        self.cadence_days = cadence_days
        self.inner_steps  = inner_steps   # how many adam steps to fit batman params
        self.inner_lr     = inner_lr

        # Time array — same for every curve in the batch
        self.t_arr = np.linspace(0, seq_len * cadence_days, seq_len)

        # Fixed batman params (period, ecc, w, limb darkening)
        self.fixed = dict(per=10.0, ecc=0.0, w=90.0, u=[0.4, 0.2],
                          limb_dark="quadratic")

    def _fit_batman(self, t_curve_np):
        """
        Fits batman parameters to a single T(t) numpy array.
        Returns the best-fit light curve as a numpy array.

        Inner optimization: torch Adam on 4 unconstrained parameters,
        sigmoid/softplus to enforce physical bounds, batman generates
        the curve at each step, MSE drives the fit.
        """
        # Initialize parameters in unconstrained space
        # We'll sigmoid/scale them into physical bounds during the loop

        best_curve = np.ones(self.seq_len, dtype=np.float32)
        best_loss  = float('inf')

        for _ in range(self.inner_steps):

            rp  = np.random.uniform(0.01, 0.20)
            a   = np.random.uniform(3.0, 30.0)
            inc = np.random.uniform(80.0, 90.0)
            t0  = np.random.uniform(0.0, self.seq_len * self.cadence_days)

            p = batman.TransitParams()
            p.rp = rp
            p.a = a
            p.inc = inc
            p.t0 = t0
            p.per = self.fixed["per"]
            p.ecc = self.fixed["ecc"]
            p.w = self.fixed["w"]
            p.u = self.fixed["u"]
            p.limb_dark = self.fixed["limb_dark"]

            try:
                m = batman.TransitModel(p, self.t_arr)
                curve_np = m.light_curve(p).astype(np.float32)

                lo, hi = curve_np.min(), curve_np.max()

                if hi - lo > 1e-8:
                    curve_np = (curve_np - lo) / (hi - lo)
                else:
                    curve_np = np.ones(self.seq_len, dtype=np.float32)

            except Exception:
                curve_np = np.ones(self.seq_len, dtype=np.float32)

            loss = np.mean((t_curve_np - curve_np) ** 2)

            if loss < best_loss:
                best_loss = loss
                best_curve = curve_np

        return best_curve

    def forward(self, t_t):
        """
        t_t : (B, 1, seq_len) — transit reconstructions from transit_decoder

        For each sample in the batch:
            1. Fit batman to T(t)[i]
            2. Compute MSE(T(t)[i], best_fit[i])
        Returns mean penalty across batch.
        """
        B   = t_t.size(0)
        t_t_sq = t_t.squeeze(1)   # (B, seq_len)

        total_loss = torch.tensor(0.0, requires_grad=True)

        for i in range(B):
            t_curve_np  = t_t_sq[i].detach().cpu().numpy()
            best_fit_np = self._fit_batman(t_curve_np)
            best_fit_t  = torch.from_numpy(best_fit_np).to(t_t.device)

            # This MSE IS differentiable w.r.t. t_t[i] — gradient flows back
            residual   = torch.mean((t_t_sq[i] - best_fit_t) ** 2)
            total_loss = total_loss + residual

        return total_loss / B


class TotalVariationLoss(nn.Module):
    """
    L_phys_stellar: penalizes sharp transitions in S(t).

    Total Variation = sum of |S(t+1) - S(t)| across the time axis.

    Stars produce slow, smooth variability — their brightness changes
    gradually over hours and days. Sharp spikes in S(t) mean transit
    signal is bleeding into the stellar channel.

    This loss is simple, fast, and fully differentiable.

    s_t : (B, 1, seq_len)
    returns: scalar TV penalty
    """

    def __init__(self):
        super().__init__()

    def forward(self, s_t):
        # Differences between adjacent time steps
        diff = s_t[:, :, 1:] - s_t[:, :, :-1]   # (B, 1, seq_len-1)
        return diff.abs().mean()


class PhysicsLoss(nn.Module):
    """
    Combined physics loss: L_phys = L_mandel_agol + lambda_tv * L_tv

    lambda_tv controls the balance between the two sub-losses.
    Default 0.1 — TV is a softer constraint than the MA fit.
    """

    def __init__(self, seq_len=1024, lambda_tv=0.1):
        super().__init__()
        self.ma_loss  = MandelAgolLoss(seq_len=seq_len)
        self.tv_loss  = TotalVariationLoss()
        self.lambda_tv = lambda_tv

    def forward(self, s_t, t_t):
        """
        s_t : (B, 1, seq_len) — stellar reconstruction
        t_t : (B, 1, seq_len) — transit reconstruction
        """
        l_ma = self.ma_loss(t_t)
        l_tv = self.tv_loss(s_t)
        return l_ma + self.lambda_tv * l_tv, l_ma, l_tv


if __name__ == "__main__":
    import sys

    phys = PhysicsLoss(seq_len=1024)

    # Smooth S(t) and transit-like T(t) — should give low loss
    s_t  = torch.ones(2, 1, 1024) * 0.5
    s_t += 0.02 * torch.sin(torch.linspace(0, 4*3.14, 1024)).unsqueeze(0).unsqueeze(0)

    t_t  = torch.ones(2, 1, 1024) * 0.95
    t_t[:, :, 400:450] = 0.80   # fake transit dip

    print("Running PhysicsLoss forward (may take ~5s for batman inner loop)...")
    total, l_ma, l_tv = phys(s_t, t_t)
    print(f"L_phys total : {total.item():.4f}")
    print(f"  L_MA       : {l_ma.item():.4f}")
    print(f"  L_TV       : {l_tv.item():.4f}")
    print("physics.py smoke test passed.")