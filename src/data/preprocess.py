import numpy as np
import os
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

def fill_gaps(flux):
    """
    Replaces NaN / Inf values using linear interpolation.
    Kepler sometimes drops cadences — NaNs break every downstream operation.
    Strategy: find indices of valid points, interpolate missing ones between them.
    """
    flux = flux.copy()
    bad = ~np.isfinite(flux)
    if bad.any():
        idx = np.arange(len(flux))
        flux[bad] = np.interp(idx[bad], idx[~bad], flux[~bad])
    return flux

def remove_spikes(flux, sigma=5.0):
    """
    Clips instrument artifacts — cosmic ray hits, detector glitches.
    Planet transits cause ~1% dips (well within 5-sigma).
    Instrument spikes are often 10-50x the noise floor — safely removed.
    np.clip() is applied symmetrically around the median.
    """
    flux = flux.copy()
    med = np.median(flux)
    std = np.std(flux)
    return np.clip(flux, med - sigma * std, med + sigma * std)

def normalize(flux):
    """
    Rescales flux to [0, 1] range: (x - min) / (max - min).
    Removes absolute brightness differences between stars.
    The model only needs to see the shape of variability, not raw electron counts.
    Guard against flat/dead curves with the 1e-8 check.
    """
    lo, hi = flux.min(), flux.max()
    if (hi - lo) < 1e-8:
        return np.zeros_like(flux)
    return (flux - lo) / (hi - lo)

def preprocess_one(flux, sigma=5.0):
    """Full pipeline for a single curve. Always in this order:
    1. fill gaps    — NaNs must go first or std/median are corrupted
    2. remove spikes — clip before normalize or spikes pull the [0,1] range
    3. normalize    — last, so range is clean
    """
    flux = fill_gaps(flux)
    flux = remove_spikes(flux, sigma)
    flux = normalize(flux)
    return flux.astype(np.float32)

def get_label(fname):
    """Parses label from filename like 'Kepler-10_label1.npy' → 1"""
    try:
        return int(fname.split("label")[1].replace(".npy", ""))
    except Exception:
        raise ValueError(f"Cannot parse label from: {fname}")

def preprocess_all(raw_dir="data/raw", out_dir="data/processed"):
    os.makedirs(out_dir, exist_ok=True)
    files = [f for f in os.listdir(raw_dir) if f.endswith(".npy")]
    if not files:
        log.error(f"No .npy files in {raw_dir}. Run download.py first.")
        return
    log.info(f"Preprocessing {len(files)} files...")
    for fname in tqdm(files):
        flux = np.load(os.path.join(raw_dir, fname))
        clean = preprocess_one(flux)
        np.save(os.path.join(out_dir, fname), clean)
    log.info(f"Done → {out_dir}/")

if __name__ == "__main__":
    preprocess_all()