import numpy as np
import batman
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

def make_transit_model(seq_len=1024, cadence_days=0.0204):
    """
    Builds a batman transit model on a time array matching our light curve length.
    cadence_days=0.0204 matches Kepler long-cadence (29.4 min = 0.0204 days).
    Returns the params object and time array — reuse these across many injections.
    """
    t = np.linspace(0, seq_len * cadence_days, seq_len)
    params = batman.TransitParams()
    params.limb_dark = "quadratic"
    params.u = [0.4, 0.2]          # typical solar-like limb darkening coefficients
    return params, t

def sample_transit_params(rng):
    """
    Randomly samples physically plausible transit parameters.
    Ranges chosen to cover small-to-medium planets on short-to-medium orbits.

    rp   : planet/star radius ratio → controls dip depth (depth ≈ rp²)
           0.05 = tiny (Mercury-like), 0.15 = large (hot Jupiter)
    a    : semi-major axis in stellar radii → controls duration
    inc  : orbital inclination (degrees) → near 90° = edge-on = visible transit
    t0   : transit center time — placed in first half so it's always in the window
    per  : orbital period (days) — short enough that at least one transit appears
    ecc  : eccentricity — zero (circular orbit, simplest case)
    w    : longitude of periastron — irrelevant for circular but batman needs it
    """
    rp  = rng.uniform(0.05, 0.15)
    a   = rng.uniform(5.0, 20.0)
    inc = rng.uniform(85.0, 90.0)
    t0  = rng.uniform(1.0, 5.0)
    per = rng.uniform(3.0, 15.0)
    return dict(rp=rp, a=a, inc=inc, t0=t0, per=per, ecc=0.0, w=90.0)

def inject_transit(flux, rng=None, seq_len=1024, cadence_days=0.0204):
    """
    Takes a clean normalized light curve and injects one synthetic transit.

    Steps:
    1. Sample random transit parameters
    2. Build batman model → get fractional flux drop at each timestep
    3. Multiply the light curve by the transit model
       (batman outputs 1.0 everywhere except during transit where it dips)
    4. Re-normalize to [0,1] so the output matches preprocessed format

    Returns:
        injected_flux : float32 array of length seq_len
        params_dict   : the batman parameters used (needed for physics loss later)
    """
    if rng is None:
        rng = np.random.default_rng()

    params, t = make_transit_model(seq_len, cadence_days)
    p = sample_transit_params(rng)

    params.t0      = p["t0"]
    params.per     = p["per"]
    params.rp      = p["rp"]
    params.a       = p["a"]
    params.inc     = p["inc"]
    params.ecc     = p["ecc"]
    params.w       = p["w"]

    try:
        m = batman.TransitModel(params, t)
        transit_curve = m.light_curve(params).astype(np.float32)
    except Exception as e:
        log.warning(f"batman failed with params {p}: {e}. Returning unmodified flux.")
        return flux.copy(), p

    injected = flux * transit_curve

    # re-normalize — multiplication shifts the range slightly
    lo, hi = injected.min(), injected.max()
    if (hi - lo) > 1e-8:
        injected = (injected - lo) / (hi - lo)

    return injected.astype(np.float32), p

def batch_inject(raw_curves, n_injections=500, seed=42):
    """
    Takes a list of clean (label=0) light curves, generates n_injections
    synthetic planet curves by randomly picking a base curve and injecting.

    Returns:
        injected_list : list of (flux_array, params_dict) tuples
    """
    rng = np.random.default_rng(seed)
    injected_list = []
    for _ in range(n_injections):
        base = raw_curves[rng.integers(len(raw_curves))]
        flux_inj, params = inject_transit(base, rng=rng)
        injected_list.append((flux_inj, params))
    log.info(f"Generated {len(injected_list)} injected transit curves.")
    return injected_list

if __name__ == "__main__":
    # quick smoke test
    import os
    proc_dir = "data/processed"
    files = [f for f in os.listdir(proc_dir) if "label0" in f]
    curves = [np.load(os.path.join(proc_dir, f)) for f in files[:5]]
    results = batch_inject(curves, n_injections=10)
    flux, p = results[0]
    print(f"Injected shape : {flux.shape}")
    print(f"Min/Max        : {flux.min():.4f} / {flux.max():.4f}")
    print(f"Params used    : rp={p['rp']:.3f}, per={p['per']:.1f}d, inc={p['inc']:.1f}°")
    print("inject.py smoke test passed.")