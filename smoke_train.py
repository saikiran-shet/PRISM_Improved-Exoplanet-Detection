import torch
import numpy as np
import os
import sys

print("=" * 60)
print("PRISM SMOKE TRAIN — step by step diagnostic")
print("=" * 60)

# ── Step 1: Check processed data exists ──────────────────────────────────────
print("\n[1/7] Checking processed data...")
proc_dir = "data/processed"
if not os.path.exists(proc_dir):
    print("  FAIL: data/processed/ does not exist. Run preprocess.py first.")
    sys.exit(1)

files     = os.listdir(proc_dir)
n_planet  = sum(1 for f in files if "label1" in f)
n_noplant = sum(1 for f in files if "label0" in f)

if len(files) == 0:
    print("  FAIL: No files in data/processed/. Run preprocess.py first.")
    sys.exit(1)

print(f"  OK — {len(files)} files found")
print(f"       Planet   (label1): {n_planet}")
print(f"       No-planet(label0): {n_noplant}")

if n_planet == 0:
    print("  WARN: Zero planet curves. Check download.py ran correctly.")
if n_noplant == 0:
    print("  WARN: Zero non-planet curves. Check download.py ran correctly.")

# ── Step 2: Check a sample file is valid ─────────────────────────────────────
print("\n[2/7] Validating a sample file...")
sample_path = os.path.join(proc_dir, files[0])
sample      = np.load(sample_path)

assert sample.shape == (1024,), \
    f"  FAIL: Expected shape (1024,), got {sample.shape}"
assert sample.dtype == np.float32, \
    f"  FAIL: Expected float32, got {sample.dtype}"
assert np.isnan(sample).sum() == 0, \
    f"  FAIL: {np.isnan(sample).sum()} NaNs found — rerun preprocess.py"
assert sample.min() >= 0.0 and sample.max() <= 1.0, \
    f"  FAIL: Values outside [0,1] — min={sample.min():.4f} max={sample.max():.4f}"

print(f"  OK — shape={sample.shape}, dtype={sample.dtype}, "
      f"min={sample.min():.4f}, max={sample.max():.4f}, NaNs=0")

# ── Step 3: Check dataset loads and injects correctly ────────────────────────
print("\n[3/7] Building dataset with 10 injections...")
from src.data.dataset import get_dataloaders

try:
    train_loader, val_loader, test_loader = get_dataloaders(
        proc_dir     = proc_dir,
        n_injections = 10,
        batch_size   = 4,
        val_split    = 0.15,
        test_split   = 0.15,
    )
except Exception as e:
    print(f"  FAIL: {e}")
    sys.exit(1)

n_train = len(train_loader.dataset)
n_val   = len(val_loader.dataset)
n_test  = len(test_loader.dataset)
print(f"  OK — train={n_train}, val={n_val}, test={n_test}")

# Check one batch shape
x_batch, y_batch = next(iter(train_loader))
assert x_batch.shape[1] == 1 and x_batch.shape[2] == 1024, \
    f"  FAIL: Expected (B,1,1024), got {x_batch.shape}"
assert set(y_batch.numpy().astype(int)).issubset({0, 1}), \
    f"  FAIL: Labels contain values other than 0 and 1"

print(f"  OK — batch shape={x_batch.shape}, "
      f"labels={y_batch.numpy().astype(int).tolist()}")

# ── Step 4: Check model builds and forward pass works ────────────────────────
print("\n[4/7] Building PRISM model and running forward pass...")
from src.models import build_prism

try:
    models = build_prism(latent_dim=64, seq_len=1024, dropout_p=0.3)
except Exception as e:
    print(f"  FAIL: build_prism() crashed: {e}")
    sys.exit(1)

enc   = models['encoder']
s_dec = models['stellar_decoder']
t_dec = models['transit_decoder']
clf   = models['classifier']

x_test = torch.randn(4, 1, 1024)
try:
    z_s, z_t = enc(x_test)
    S_t      = s_dec(z_s)
    T_t      = t_dec(z_t)
    prob     = clf(z_t)
except Exception as e:
    print(f"  FAIL: Forward pass crashed: {e}")
    sys.exit(1)

assert z_s.shape  == (4, 64),      f"  FAIL: z_s shape {z_s.shape}"
assert z_t.shape  == (4, 64),      f"  FAIL: z_t shape {z_t.shape}"
assert S_t.shape  == (4, 1, 1024), f"  FAIL: S_t shape {S_t.shape}"
assert T_t.shape  == (4, 1, 1024), f"  FAIL: T_t shape {T_t.shape}"
assert prob.shape == (4, 1),       f"  FAIL: prob shape {prob.shape}"
assert prob.min() >= 0.0 and prob.max() <= 1.0, \
    f"  FAIL: prob outside [0,1] — {prob.min():.4f} to {prob.max():.4f}"

total_params = sum(p.numel() for m in models.values()
                   for p in m.parameters())
print(f"  OK — z_s={z_s.shape}, z_t={z_t.shape}, "
      f"S_t={S_t.shape}, prob={prob.shape}")
print(f"  OK — Total parameters: {total_params:,}")

# ── Step 5: Check all four losses compute without error ──────────────────────
print("\n[5/7] Testing all four losses...")
from src.losses import build_losses

# Subset wraps PRISMDataset — access underlying dataset via .dataset
full_train_dataset = train_loader.dataset.dataset
all_samples        = full_train_dataset.samples

n_pos      = sum(1 for _, l in all_samples if l == 1.0)
n_neg      = sum(1 for _, l in all_samples if l == 0.0)
pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

try:
    losses = build_losses(latent_dim=64, seq_len=1024,
                          pos_weight=pos_weight)
except Exception as e:
    print(f"  FAIL: build_losses() crashed: {e}")
    sys.exit(1)

x_l      = torch.rand(4, 1, 1024)
labels_l = torch.tensor([1.0, 0.0, 1.0, 0.0])
z_s_l, z_t_l = enc(x_l)
S_t_l        = s_dec(z_s_l)
T_t_l        = t_dec(z_t_l)
prob_l       = clf(z_t_l)

try:
    l_recon             = losses['recon'](x_l, S_t_l, T_t_l)
    l_mi                = losses['mine'](z_s_l, z_t_l)
    l_phys, l_ma, l_tv  = losses['physics'](S_t_l, T_t_l)
    l_cls               = losses['classify'](prob_l, labels_l)
except Exception as e:
    print(f"  FAIL: Loss computation crashed: {e}")
    sys.exit(1)

assert np.isfinite(l_recon.item()), "FAIL: L_recon is NaN/Inf"
assert np.isfinite(l_mi.item()),    "FAIL: L_MI is NaN/Inf"
assert np.isfinite(l_phys.item()),  "FAIL: L_phys is NaN/Inf"
assert np.isfinite(l_cls.item()),   "FAIL: L_classify is NaN/Inf"

print(f"  OK — L_recon   : {l_recon.item():.4f}")
print(f"  OK — L_MI      : {l_mi.item():.4f}")
print(f"  OK — L_phys    : {l_phys.item():.4f}  "
      f"(MA={l_ma.item():.4f}, TV={l_tv.item():.4f})")
print(f"  OK — L_classify: {l_cls.item():.4f}")
print(f"  OK — pos_weight: {pos_weight:.4f}  "
      f"(n_pos={n_pos}, n_neg={n_neg})")

# ── Step 6: Run 2 actual training epochs ─────────────────────────────────────
print("\n[6/7] Running 2 training epochs...")
from src.train import train

cfg = dict(
    latent_dim   = 64,
    seq_len      = 1024,
    dropout_p    = 0.3,
    n_injections = 10,
    batch_size   = 4,
    val_split    = 0.15,
    test_split   = 0.15,
    epochs       = 2,
    patience     = 8,
    lr_prism     = 1e-4,
    lr_mine      = 1e-4,
    lambda_recon = 1.0,
    lambda_mi    = 0.1,
    lambda_phys  = 0.1,
    lambda_cls   = 1.0,
    mine_steps   = 1,
    ckpt_dir     = "outputs/checkpoints",
    proc_dir     = "data/processed",
)

try:
    _, _, history = train(cfg)
except Exception as e:
    print(f"  FAIL: Training crashed: {e}")
    sys.exit(1)

assert len(history) == 2, \
    f"  FAIL: Expected 2 epochs in history, got {len(history)}"

e1 = history[0]
e2 = history[1]

print(f"  OK — Epoch 1: train={e1['train_total']:.4f} "
      f"val={e1['val_total']:.4f} acc={e1['val_accuracy']:.3f}")
print(f"  OK — Epoch 2: train={e2['train_total']:.4f} "
      f"val={e2['val_total']:.4f} acc={e2['val_accuracy']:.3f}")

for key in ['train_recon', 'train_mi', 'train_phys', 'train_cls']:
    val = e1[key]
    assert np.isfinite(val), \
        f"  FAIL: {key}={val} is not finite (NaN/Inf)"
print("  OK — All loss values are finite (no NaN/Inf)")

# ── Step 7: Check checkpoint was saved ───────────────────────────────────────
print("\n[7/7] Checking checkpoint saved correctly...")
ckpt_path = "outputs/checkpoints/prism_best.pt"
if not os.path.exists(ckpt_path):
    print(f"  FAIL: Checkpoint not found at {ckpt_path}")
    sys.exit(1)

ckpt = torch.load(ckpt_path, map_location='cpu')
required_keys = ['encoder', 'stellar_decoder',
                 'transit_decoder', 'classifier', 'mine', 'cfg']
for k in required_keys:
    assert k in ckpt, f"  FAIL: Missing key '{k}' in checkpoint"

print(f"  OK — Checkpoint found: {ckpt_path}")
print(f"  OK — Keys: {list(ckpt.keys())}")
print(f"  OK — Saved at epoch {ckpt['epoch']}, "
      f"val_loss={ckpt['val_loss']:.4f}")

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ALL 7 CHECKS PASSED — PRISM is ready for full training")
print("=" * 60)
print("\nRun full training with:")
print("  python -m src.train")