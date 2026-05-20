import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from src.data.inject import batch_inject
from src.data.preprocess import get_label
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

class PRISMDataset(Dataset):
    """
    Loads all light curves from data/processed/ and augments with injected transits.

    Each item returned is:
        flux   : float32 tensor of shape (1, seq_len)  ← the "1" is the channel dim for Conv1d
        label  : float32 scalar — 1.0 = planet, 0.0 = no planet

    The (1, seq_len) shape matters — PyTorch Conv1d expects (batch, channels, length).
    We have 1 channel (brightness over time), unlike images which have 3 (RGB).
    """

    def __init__(self, proc_dir="data/processed", n_injections=500, seed=42):
        self.samples = []   # list of (flux_array, label) tuples

        # ── Load real downloaded curves ──────────────────────────────────────
        files = sorted([f for f in os.listdir(proc_dir) if f.endswith(".npy")])
        if not files:
            raise RuntimeError(f"No .npy files found in {proc_dir}. Run Phase 1 first.")

        real_planet   = []
        real_noplant  = []

        for fname in files:
            flux  = np.load(os.path.join(proc_dir, fname)).astype(np.float32)
            label = get_label(fname)
            self.samples.append((flux, float(label)))
            if label == 0:
                real_noplant.append(flux)
            else:
                real_planet.append(flux)

        log.info(f"Loaded {len(real_planet)} real planet curves, "
                 f"{len(real_noplant)} non-planet curves.")

        # ── Inject synthetic transits into non-planet curves ─────────────────
        # Why? We only have ~50 real planet curves. Injected ones give us
        # hundreds more with known ground truth AND known physical parameters.
        if n_injections > 0 and real_noplant:
            injected = batch_inject(real_noplant, n_injections=n_injections, seed=seed)
            for flux_inj, _ in injected:
                self.samples.append((flux_inj, 1.0))   # label=1, it has a transit
            log.info(f"Added {len(injected)} injected planet curves.")

        log.info(f"Total dataset size: {len(self.samples)} curves.")
        n_pos = sum(1 for _, l in self.samples if l == 1.0)
        n_neg = len(self.samples) - n_pos
        log.info(f"Class balance — planet: {n_pos}, no-planet: {n_neg}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        flux, label = self.samples[idx]
        # Add channel dimension: (seq_len,) → (1, seq_len)
        flux_tensor  = torch.from_numpy(flux).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return flux_tensor, label_tensor


def get_dataloaders(proc_dir="data/processed", n_injections=500,
                    batch_size=32, val_split=0.15, test_split=0.15, seed=42):
    """
    Builds train / val / test DataLoaders with stratified-ish random splits.

    Split sizes (defaults):
        train : 70%
        val   : 15%  ← used for early stopping / hyperparameter tuning
        test  : 15%  ← held out completely until final evaluation

    Returns: train_loader, val_loader, test_loader
    """
    dataset = PRISMDataset(proc_dir=proc_dir, n_injections=n_injections, seed=seed)
    n       = len(dataset)
    n_test  = int(n * test_split)
    n_val   = int(n * val_split)
    n_train = n - n_val - n_test

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)

    log.info(f"Splits — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(n_injections=50)
    batch_flux, batch_labels = next(iter(train_loader))
    print(f"Batch flux shape  : {batch_flux.shape}")   # (32, 1, 1024)
    print(f"Batch label shape : {batch_labels.shape}") # (32,)
    print(f"Label values      : {batch_labels[:8]}")
    print(f"Flux min/max      : {batch_flux.min():.4f} / {batch_flux.max():.4f}")
    print("dataset.py smoke test passed.")