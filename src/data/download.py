import lightkurve as lk
import numpy as np
import os
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_kic_list(filepath, max_stars=None):
    """Loads KIC IDs from a text file, one per line."""
    with open(filepath) as f:
        kics = [int(line.strip()) for line in f if line.strip()]
    if max_stars:
        kics = kics[:max_stars]
    return kics


def extract_flux(lc, seq_len):
    """
    Converts lightkurve MaskedNDArray → plain float32 numpy array.
    Masked positions become NaN — handled by preprocess.fill_gaps().
    """
    raw = lc.flux.value
    if isinstance(raw, np.ma.MaskedArray):
        raw = raw.filled(np.nan)
    flux = np.array(raw, dtype=np.float32)
    if len(flux) < seq_len:
        return None
    return flux[:seq_len]


def download_one(args):
    """
    Downloads one quarter for one KIC star.
    Args tuple: (kic_id, label, out_dir, seq_len)
    Returns (kic_id, True/False).
    """
    kic_id, label, out_dir, seq_len = args
    star_name = f"KIC {kic_id}"
    safe_name = f"KIC_{kic_id}"
    out_path  = os.path.join(out_dir, f"{safe_name}_label{label}.npy")

    if os.path.exists(out_path):
        return kic_id, True

    try:
        results = lk.search_lightcurve(
            star_name, mission="Kepler", author="Kepler"
        )
        if len(results) == 0:
            return kic_id, False

        lc   = results[0].download()
        lc   = lc.remove_nans().normalize()
        flux = extract_flux(lc, seq_len)

        if flux is None:
            return kic_id, False

        np.save(out_path, flux)
        return kic_id, True

    except Exception as e:
        log.warning(f"KIC {kic_id}: {e}")
        return kic_id, False


def download_all(
    confirmed_file   = "confirmed_kics.txt",
    false_pos_file   = "false_positive_kics.txt",
    out_dir          = "data/raw",
    seq_len          = 1024,
    max_confirmed    = 500,    # how many planet stars to download
    max_false_pos    = 500,    # how many non-planet stars to download
    max_workers      = 6,      # parallel threads — don't exceed 8
):
    """
    Downloads light curves from NASA-vetted Kepler KOI catalog.

    Balanced download: equal numbers of confirmed planets
    and false positives → no class imbalance from the start.

    max_confirmed and max_false_pos control how many you download.
    Start with 500 each (1000 total) for a meaningful dataset.
    Increase to 1000 each once you confirm the pipeline works.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Load KIC lists
    if not os.path.exists(confirmed_file):
        log.error(f"Missing {confirmed_file}. Run fetch_catalog.py first.")
        return
    if not os.path.exists(false_pos_file):
        log.error(f"Missing {false_pos_file}. Run fetch_catalog.py first.")
        return

    confirmed_kics = load_kic_list(confirmed_file,  max_stars=max_confirmed)
    false_pos_kics  = load_kic_list(false_pos_file,  max_stars=max_false_pos)

    log.info(f"Confirmed planets to download  : {len(confirmed_kics)}")
    log.info(f"False positives to download    : {len(false_pos_kics)}")
    log.info(f"Parallel workers               : {max_workers}")

    # Build task list
    tasks = (
        [(kic, 1, out_dir, seq_len) for kic in confirmed_kics] +
        [(kic, 0, out_dir, seq_len) for kic in false_pos_kics]
    )

    ok = fail = skipped = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_one, t): t[0]
            for t in tasks
        }
        for future in tqdm(as_completed(futures),
                           total=len(futures),
                           desc="Downloading"):
            kic_id = futures[future]
            try:
                _, success = future.result()
                if success:
                    ok += 1
                else:
                    fail += 1
            except Exception as e:
                fail += 1
                log.error(f"KIC {kic_id}: {e}")

    log.info(f"Done — {ok} saved, {fail} failed.")

    # Summary
    all_files = os.listdir(out_dir)
    n_planet  = sum(1 for f in all_files if "label1" in f)
    n_noplant = sum(1 for f in all_files if "label0" in f)
    log.info(f"Summary — planet: {n_planet}, "
             f"non-planet: {n_noplant}, "
             f"total: {len(all_files)}")


if __name__ == "__main__":
    download_all(
        max_confirmed = 500,
        max_false_pos = 500,
        max_workers   = 6,
    )