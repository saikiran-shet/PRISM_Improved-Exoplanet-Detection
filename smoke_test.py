import torch
from src.models import build_prism

models = build_prism(latent_dim=64)

enc   = models['encoder']
s_dec = models['stellar_decoder']
t_dec = models['transit_decoder']
clf   = models['classifier']

x = torch.randn(4, 1, 1024)

z_s, z_t = enc(x)

S_t = s_dec(z_s)
T_t = t_dec(z_t)

prob  = clf(z_t)
recon = S_t + T_t

print(f"X(t)   : {x.shape}")
print(f"z_s    : {z_s.shape}")
print(f"z_t    : {z_t.shape}")
print(f"S(t)   : {S_t.shape}")
print(f"T(t)   : {T_t.shape}")
print(f"S+T    : {recon.shape}")
print(f"prob   : {prob.shape}")

print("Full forward pass OK.")