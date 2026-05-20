import lightkurve as lk
import numpy as np
import os
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# 50 confirmed planet hosts — label 1
CONFIRMED_PLANETS = [
    "Kepler-10", "Kepler-22", "Kepler-62", "Kepler-69", "Kepler-442",
    "Kepler-452", "Kepler-186", "Kepler-438", "Kepler-296", "Kepler-174",
    "Kepler-61",  "Kepler-283", "Kepler-395", "Kepler-430", "Kepler-444",
    "Kepler-20",  "Kepler-42",  "Kepler-55",  "Kepler-80",  "Kepler-90",
    "Kepler-11",  "Kepler-18",  "Kepler-25",  "Kepler-30",  "Kepler-37",
    "Kepler-48",  "Kepler-50",  "Kepler-51",  "Kepler-56",  "Kepler-60",
    "Kepler-65",  "Kepler-68",  "Kepler-79",  "Kepler-84",  "Kepler-88",
    "Kepler-93",  "Kepler-94",  "Kepler-95",  "Kepler-96",  "Kepler-97",
    "Kepler-98",  "Kepler-99",  "Kepler-100", "Kepler-101", "Kepler-102",
    "Kepler-103", "Kepler-106", "Kepler-109", "Kepler-113", "Kepler-131",
]

# 50 stars with no detected planets — label 0
NON_PLANET_STARS = [
    "KIC 3733346",  "KIC 4349452",  "KIC 5098444",  "KIC 6196457",
    "KIC 7103006",  "KIC 8077137",  "KIC 9410930",  "KIC 10074700",
    "KIC 11295426", "KIC 12066749", "KIC 3239945",  "KIC 4045571",
    "KIC 5128972",  "KIC 6289240",  "KIC 7206837",  "KIC 8292840",
    "KIC 9475552",  "KIC 10453521", "KIC 11401755", "KIC 12255108",
    "KIC 3114667",  "KIC 4243274",  "KIC 5301750",  "KIC 6425957",
    "KIC 7341231",  "KIC 8394721",  "KIC 9595827",  "KIC 10514430",
    "KIC 11497958", "KIC 12417799", "KIC 3326428",  "KIC 4453041",
    "KIC 5450764",  "KIC 6587001",  "KIC 7433192",  "KIC 8493142",
    "KIC 9656919",  "KIC 10592770", "KIC 11557439", "KIC 12507577",
    "KIC 3441784",  "KIC 4570949",  "KIC 5612566",  "KIC 6706048",
    "KIC 7519888",  "KIC 8586756",  "KIC 9706819",  "KIC 10660165",
    "KIC 11617378", "KIC 12644822",
    "KIC 2166460",  "KIC 2694810",  "KIC 3231341",  "KIC 3543333",
    "KIC 3642335",  "KIC 3831454",  "KIC 4049084",  "KIC 4077526",
    "KIC 4169400",  "KIC 4243461",  "KIC 4349452",  "KIC 4570949",
    "KIC 4758368",  "KIC 4913852",  "KIC 5080652",  "KIC 5164255",
    "KIC 5358241",  "KIC 5444504",  "KIC 5613330",  "KIC 5653126",
    "KIC 5735762",  "KIC 5951458",  "KIC 6062088",  "KIC 6268648",
    "KIC 6381562",  "KIC 6508221",  "KIC 6665695",  "KIC 6862328",
    "KIC 7009496",  "KIC 7199397",  "KIC 7300184",  "KIC 7419318",
    "KIC 7582608",  "KIC 7668648",  "KIC 7771531",  "KIC 7943602",
    "KIC 8077137",  "KIC 8211096",  "KIC 8349582",  "KIC 8494142",
    "KIC 8631743",  "KIC 8738735",  "KIC 8866102",  "KIC 9002278",
    "KIC 9150827",  "KIC 9305831",  "KIC 9410930",  "KIC 9573350",
    "KIC 9704149",  "KIC 9836441",  "KIC 10019708", "KIC 10124866",
    "KIC 10264660", "KIC 10386984", "KIC 10514430", "KIC 10656508",
    "KIC 10789273", "KIC 10925104", "KIC 11074933", "KIC 11189959",
    "KIC 11304458", "KIC 11403216", "KIC 11551692", "KIC 11709832",
    "KIC 11874676", "KIC 12066749", "KIC 12252424", "KIC 12417799",
    "KIC 12554589", "KIC 12644822",
]

def download_one(star_name, label, out_dir, seq_len=1024):
    """
    Downloads one star's light curve, trims to seq_len points, saves as .npy.
    Returns True on success, False on any failure.
    Skips if file already exists — safe to re-run after partial failures.
    """
    safe_name = star_name.replace(" ", "_")
    out_path = os.path.join(out_dir, f"{safe_name}_label{label}.npy")

    if os.path.exists(out_path):           # already downloaded — skip
        return True

    try:
        results = lk.search_lightcurve(star_name, mission="Kepler", author="Kepler")
        if len(results) == 0:
            log.warning(f"No Kepler data found for {star_name}")
            return False

        lc = results[0].download()         # grab the first available quarter
        lc = lc.remove_nans().normalize()
        flux = np.array(lc.flux.filled(np.nan), dtype=np.float32)
        flux = flux[~np.isnan(flux)]

        if len(flux) < seq_len:
            log.warning(f"{star_name}: only {len(flux)} points, need {seq_len}. Skipping.")
            return False

        flux = flux[:seq_len]              # take first seq_len cadences
        np.save(out_path, flux)
        log.info(f"Saved → {out_path}")
        return True

    except Exception as e:
        log.error(f"Failed {star_name}: {e}")
        return False

def download_all(out_dir="data/raw", seq_len=1024):
    os.makedirs(out_dir, exist_ok=True)
    ok = fail = 0

    log.info("── Planet hosts (label=1) ──")
    for star in tqdm(CONFIRMED_PLANETS, desc="Planets"):
        result = download_one(star, 1, out_dir, seq_len)
        ok += result; fail += not result

    log.info("── Non-planet stars (label=0) ──")
    for star in tqdm(NON_PLANET_STARS, desc="Non-planets"):
        result = download_one(star, 0, out_dir, seq_len)
        ok += result; fail += not result

    log.info(f"Done — {ok} saved, {fail} failed.")

if __name__ == "__main__":
    download_all()