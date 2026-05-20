import torch

from src.models import build_prism
from src.losses import build_losses


# ── Build models and losses ───────────────────────────────────────────────
models = build_prism(latent_dim=64)
losses = build_losses(latent_dim=64)

enc   = models['encoder']
s_dec = models['stellar_decoder']
t_dec = models['transit_decoder']
clf   = models['classifier']


# ── Fake batch for smoke testing ──────────────────────────────────────────
x = torch.rand(4, 1, 1024)   # (batch, channel, seq_len)

labels = torch.tensor([
    1.0,
    0.0,
    1.0,
    0.0
])


# ── Full forward pass ─────────────────────────────────────────────────────
z_s, z_t = enc(x)

S_t = s_dec(z_s)
T_t = t_dec(z_t)

prob = clf(z_t)


# ── Compute all losses ────────────────────────────────────────────────────
l_recon = losses['recon'](x, S_t, T_t)

l_mi = losses['mine'](z_s, z_t)

l_phys, l_ma, l_tv = losses['physics'](S_t, T_t)

l_cls = losses['classify'](prob, labels)


# ── Combined PRISM objective ──────────────────────────────────────────────
total = (
    l_recon
    + 0.1 * l_mi
    + 0.1 * l_phys
    + l_cls
)


# ── Print diagnostics ─────────────────────────────────────────────────────
print(f"Input shape : {x.shape}")

print(f"z_s shape   : {z_s.shape}")
print(f"z_t shape   : {z_t.shape}")

print(f"S(t) shape  : {S_t.shape}")
print(f"T(t) shape  : {T_t.shape}")

print(f"Prob shape  : {prob.shape}")

print("\n── Individual Losses ──")
print(f"L_recon     : {l_recon.item():.4f}")
print(f"L_MI        : {l_mi.item():.4f}")
print(f"L_phys      : {l_phys.item():.4f}")
print(f"  L_MA      : {l_ma.item():.4f}")
print(f"  L_TV      : {l_tv.item():.4f}")
print(f"L_classify  : {l_cls.item():.4f}")

print("\n── Total Loss ──")
print(f"L_total     : {total.item():.4f}")

print("\nAll losses integrated OK.")