# PRISM — Photometric Residual Isolation via Signal decomposition and Mutual-information

Neural decomposition framework for exoplanet detection from Kepler light curves.

## Approach
Decomposes observed flux X(t) = S(t) + T(t) where:
- S(t) = stellar activity (slow variability, flares)
- T(t) = transit signal (periodic dips)

Classification is performed on T(t) only, after physics-grounded decomposition.

## Four training losses
- L_recon: reconstruction fidelity — S + T must equal X
- L_MI: MINE mutual information — z_s and z_t must be independent
- L_phys: Mandel-Agol transit constraint + total variation on S(t)
- L_classify: binary cross-entropy for planet detection

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
