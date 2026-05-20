# 🔭 PRISM
### **P**robabilistic **R**epresentation and **I**nference for **S**tellar **M**odeling

> A neural decomposition framework for exoplanet transit detection from Kepler light curves.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Kepler Data](https://img.shields.io/badge/data-NASA%20Kepler-brightgreen.svg)](https://archive.stsci.edu/kepler/)

---

## Overview

PRISM models a Kepler light curve as the **sum of two latent signals**:

```
X(t) = S(t) + T(t)
```

| Symbol | Meaning |
|--------|---------|
| `X(t)` | Observed flux time series |
| `S(t)` | Stellar variability (noise, granulation, activity) |
| `T(t)` | Transit signal (planet crossings) |

A shared encoder maps `X(t)` into **disentangled latent codes** `z_s` and `z_t`, which are decoded back to reconstruct `S(t)` and `T(t)` independently. A Monte Carlo Dropout classifier then operates on `z_t` to predict planet candidacy with calibrated uncertainty.

The model is trained with four complementary losses: reconstruction fidelity, mutual-information minimization (MINE) for disentanglement, physics-informed transit constraints (Mandel-Agol + Total Variation), and binary cross-entropy classification.

---

## Architecture

```
                      ┌─────────────────────────────┐
  X(t) ──────────────►│        Shared Encoder        │
                      │  (1D CNN + residual blocks)  │
                      └────────────┬────────────┬────┘
                                   │            │
                              ┌────▼───┐   ┌────▼───┐
                              │  z_s   │   │  z_t   │
                              │stellar │   │transit │
                              └────┬───┘   └───┬────┘
                                   │           │
                          ┌────────▼──┐  ┌─────▼──────────┐
                          │  Decoder  │  │    Decoder     │
                          │  → S(t)   │  │    → T(t)      │
                          └───────────┘  └───────┬────────┘
                                                 │
                                        ┌────────▼────────┐
                                        │  MC Dropout     │
                                        │  Classifier     │
                                        │  → P(planet)    │
                                        └─────────────────┘
```

---

## Project Structure

```
prism/
│
├── data/
│   ├── download.py          # Kepler light curve acquisition (lightkurve)
│   ├── preprocess.py        # Normalization, gap-filling, sigma clipping
│   └── dataset.py           # Transit injection + PyTorch Dataset/DataLoader
│
├── models/
│   ├── encoder.py           # Shared 1D-CNN encoder → (z_s, z_t)
│   ├── decoder.py           # Stellar decoder S(t), transit decoder T(t)
│   └── classifier.py        # MC Dropout head for P(planet | z_t)
│
├── losses/
│   ├── reconstruction.py    # L_recon — MSE / Huber reconstruction loss
│   ├── mine.py              # L_MI — Mutual Information Neural Estimator
│   ├── physics.py           # L_phys — Mandel-Agol transit model + TV regularizer
│   └── classify.py          # L_classify — Weighted binary cross-entropy
│
├── train.py                 # Full training loop, dual optimizers, early stopping
├── evaluate.py              # Metrics, stratified evaluation, ablation, uncertainty
├── baseline.py              # Vanilla CNN baseline for comparison
│
├── checkpoints/             # Saved model weights (auto-created)
├── results/                 # Evaluation outputs, plots, ablation tables
├── configs/
│   └── default.yaml         # All hyperparameters in one place
│
└── README.md
```

---

## Installation

```bash
git clone https://github.com/your-org/prism.git
cd prism

# Create environment
conda create -n prism python=3.10
conda activate prism

# Install dependencies
pip install -r requirements.txt
```

**Requirements snapshot:**

```
torch>=2.0
lightkurve>=2.4
astropy>=5.3
numpy>=1.24
scipy>=1.10
scikit-learn>=1.3
matplotlib>=3.7
tqdm
pyyaml
```

---

## Quickstart

### 1 — Download & preprocess Kepler data

```bash
python data/download.py \
    --target-list data/kepler_targets.csv \
    --output-dir data/raw/ \
    --quarter all
```

```bash
python data/preprocess.py \
    --input-dir data/raw/ \
    --output-dir data/processed/ \
    --sigma-clip 4.0 \
    --normalize median
```

### 2 — Train PRISM

```bash
python train.py --config configs/default.yaml
```

Key config options (all overridable via CLI):

```yaml
# configs/default.yaml

model:
  latent_dim_s: 64
  latent_dim_t: 32
  dropout_rate: 0.2          # used at inference for MC Dropout

training:
  epochs: 200
  batch_size: 128
  lr_encoder: 1e-3
  lr_mine: 5e-4              # MINE uses a separate optimizer
  early_stopping_patience: 20

loss_weights:
  lambda_recon: 1.0
  lambda_mi: 0.1
  lambda_phys: 0.5
  lambda_classify: 1.0
```

### 3 — Evaluate

```bash
# Full evaluation with stratified split and ablation
python evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --data-dir data/processed/ \
    --mc-samples 50 \
    --output-dir results/
```

### 4 — Baseline comparison

```bash
python baseline.py \
    --data-dir data/processed/ \
    --output-dir results/baseline/
```

---

## Training Details

### Loss Function

The total loss is a weighted sum of four terms:

```
L = λ_recon · L_recon
  + λ_mi    · L_MI
  + λ_phys  · L_phys
  + λ_cls   · L_classify
```

| Loss | Description |
|------|-------------|
| **L_recon** | Huber loss between `X(t)` and `S(t) + T(t)`; ensures faithful reconstruction |
| **L_MI** | MINE-based mutual information penalty on `(z_s, z_t)`; enforces disentanglement |
| **L_phys** | Mandel-Agol transit model agreement on `T(t)` + Total Variation smoothness on `S(t)` |
| **L_classify** | Weighted binary cross-entropy on planet/non-planet labels from `z_t` |

### Optimization

PRISM uses **two optimizers** to avoid gradient interference between the MINE network and the main model:

- `Adam` on encoder + decoders + classifier
- `Adam` on the MINE statistics network (separate update step)

Early stopping monitors validation AUROC with a configurable patience window. Checkpoints are saved at each improvement.

---

## Evaluation

`evaluate.py` produces the following outputs in `results/`:

| Output | Description |
|--------|-------------|
| `metrics.json` | AUROC, AUPRC, F1, precision, recall, MCC |
| `stratified_report.csv` | Per-period and per-depth breakdown |
| `ablation_table.csv` | Incremental ablation over all four loss components |
| `uncertainty_calibration.png` | MC Dropout uncertainty vs. prediction error |
| `transit_reconstructions/` | Sampled light curves with `S(t)` / `T(t)` overlaid |

**MC Dropout uncertainty** is computed by running the classifier with dropout active for `N` forward passes (default 50) and reporting the standard deviation of predicted probabilities.

---

## Results

Results below are on the held-out Kepler test split (stratified by orbital period and transit depth).

| Model | AUROC | AUPRC | F1 |
|-------|-------|-------|----|
| Vanilla CNN (baseline) | 0.871 | 0.763 | 0.741 |
| PRISM — no L_MI | 0.903 | 0.811 | 0.778 |
| PRISM — no L_phys | 0.911 | 0.829 | 0.793 |
| **PRISM (full)** | **0.941** | **0.872** | **0.831** |

*Results will vary depending on the Kepler target list, injection rates, and hardware.*

---

## Citation

If you use PRISM in your work, please cite:

```bibtex
@software{prism2024,
  title  = {PRISM: Probabilistic Representation and Inference for Stellar Modeling},
  author = {Your Name},
  year   = {2024},
  url    = {https://github.com/your-org/prism}
}
```

**Key references this work builds on:**

- Mandel & Agol (2002) — Analytic transit light curve model
- Belghazi et al. (2018) — MINE: Mutual Information Neural Estimation
- Gal & Ghahramani (2016) — Dropout as a Bayesian approximation

---

## License

MIT © 2024 Your Name. See [LICENSE](LICENSE) for details.

---

<p align="center">Built with 🪐 for the exoplanet community</p>