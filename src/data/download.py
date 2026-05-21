import lightkurve as lk
import numpy as np
import os
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

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

NON_PLANET_STARS = [
    "KIC 1026032",  "KIC 1161620",  "KIC 1164678",  "KIC 1296413",
    "KIC 1431060",  "KIC 1571511",  "KIC 1618690",  "KIC 1720554",
    "KIC 1873174",  "KIC 1995732",  "KIC 2010607",  "KIC 2010809",
    "KIC 2021174",  "KIC 2141150",  "KIC 2159783",  "KIC 2166460",
    "KIC 2281485",  "KIC 2302548",  "KIC 2304168",  "KIC 2438113",
    "KIC 2444412",  "KIC 2557168",  "KIC 2571238",  "KIC 2694810",
    "KIC 2714077",  "KIC 2720563",  "KIC 2836292",  "KIC 2853038",
    "KIC 2968789",  "KIC 2987027",  "KIC 3100219",  "KIC 3102916",
    "KIC 3114667",  "KIC 3230491",  "KIC 3231341",  "KIC 3239945",
    "KIC 3326428",  "KIC 3340573",  "KIC 3352751",  "KIC 3441784",
    "KIC 3442055",  "KIC 3456181",  "KIC 3543333",  "KIC 3559650",
    "KIC 3641726",  "KIC 3642335",  "KIC 3731676",  "KIC 3733346",
    "KIC 3831454",  "KIC 3832474",  "KIC 3934529",  "KIC 3936357",
    "KIC 4043190",  "KIC 4045571",  "KIC 4049084",  "KIC 4077526",
    "KIC 4144800",  "KIC 4169400",  "KIC 4243274",  "KIC 4243461",
    "KIC 4244082",  "KIC 4349452",  "KIC 4350554",  "KIC 4453041",
    "KIC 4455365",  "KIC 4472818",  "KIC 4570949",  "KIC 4574987",
    "KIC 4663622",  "KIC 4664609",  "KIC 4758368",  "KIC 4760618",
    "KIC 4851217",  "KIC 4853438",  "KIC 4913852",  "KIC 4914923",
    "KIC 5008189",  "KIC 5009475",  "KIC 5080652",  "KIC 5097598",
    "KIC 5098444",  "KIC 5128972",  "KIC 5164255",  "KIC 5165605",
    "KIC 5213466",  "KIC 5214740",  "KIC 5301750",  "KIC 5358241",
    "KIC 5359613",  "KIC 5444504",  "KIC 5445790",  "KIC 5450764",
    "KIC 5536555",  "KIC 5537861",  "KIC 5612566",  "KIC 5613330",
    "KIC 5614616",  "KIC 5653126",  "KIC 5735762",  "KIC 5736988",
    "KIC 5812701",  "KIC 5813977",  "KIC 5866724",  "KIC 5951458",
    "KIC 6028116",  "KIC 6062088",  "KIC 6106415",  "KIC 6196457",
    "KIC 6268648",  "KIC 6289240",  "KIC 6381562",  "KIC 6425957",
    "KIC 6508221",  "KIC 6587001",  "KIC 6665695",  "KIC 6706048",
    "KIC 6862328",  "KIC 6923953",  "KIC 7009496",  "KIC 7103006",
    "KIC 7199397",  "KIC 7206837",  "KIC 7300184",  "KIC 7341231",
    "KIC 7419318",  "KIC 7433192",  "KIC 7519888",  "KIC 7582608",
    "KIC 7668648",  "KIC 7700622",  "KIC 7771531",  "KIC 7848385",
    "KIC 7943602",  "KIC 8011334",  "KIC 8077137",  "KIC 8150320",
    "KIC 8211096",  "KIC 8292840",  "KIC 8349582",  "KIC 8394721",
    "KIC 8460208",  "KIC 8493142",  "KIC 8494142",  "KIC 8586756",
    "KIC 8631743",  "KIC 8738735",  "KIC 8804455",  "KIC 8866102",
    "KIC 8936174",  "KIC 9002278",  "KIC 9016693",  "KIC 9082860",
    "KIC 9150827",  "KIC 9210192",  "KIC 9305831",  "KIC 9410930",
    "KIC 9475552",  "KIC 9532126",  "KIC 9573350",  "KIC 9595827",
    "KIC 9656919",  "KIC 9704149",  "KIC 9706819",  "KIC 9757613",
    "KIC 9836441",  "KIC 9941859",  "KIC 10019708", "KIC 10074700",
    "KIC 10124866", "KIC 10196240", "KIC 10264660", "KIC 10319385",
    "KIC 10386984", "KIC 10453521", "KIC 10514430", "KIC 10592770",
    "KIC 10656508", "KIC 10660165", "KIC 10723994", "KIC 10789273",
    "KIC 10858716", "KIC 10925104", "KIC 10990886", "KIC 11074933",
    "KIC 11144556", "KIC 11189959", "KIC 11240260", "KIC 11295426",
    "KIC 11304458", "KIC 11401755", "KIC 11403216", "KIC 11453592",
    "KIC 11497958", "KIC 11551692", "KIC 11557439", "KIC 11617378",
    "KIC 11661210", "KIC 11709832", "KIC 11769022", "KIC 11820830",
    "KIC 11874676", "KIC 11918099", "KIC 11969129", "KIC 12066749",
    "KIC 12110942", "KIC 12156048", "KIC 12202140", "KIC 12252424",
    "KIC 12255108", "KIC 12303549", "KIC 12368972", "KIC 12417799",
    "KIC 12470053", "KIC 12507577", "KIC 12554589", "KIC 12599700",
    "KIC 12644822",
]

def download_one(star_name, label, out_dir, seq_len=1024):
    """
    Downloads ALL available Kepler quarters for a star.
    Each quarter saved as a separate .npy file with quarter index in name.
    Skips files that already exist — safe to re-run after partial failures.
    Returns count of files saved.
    """
    safe_name = star_name.replace(" ", "_")

    try:
        results = lk.search_lightcurve(
            star_name, mission="Kepler", author="Kepler"
        )
        if len(results) == 0:
            log.warning(f"No Kepler data found for {star_name}")
            return 0

        saved = 0
        for i, result in enumerate(results):
            out_path = os.path.join(
                out_dir, f"{safe_name}_q{i}_label{label}.npy"
            )
            if os.path.exists(out_path):
                saved += 1
                continue
            try:
                lc   = result.download()
                lc   = lc.remove_nans().normalize()
                flux = lc.flux.value.astype(np.float32)
                if len(flux) < seq_len:
                    log.warning(f"{star_name} Q{i}: only {len(flux)} points, skipping")
                    continue
                flux = flux[:seq_len]
                np.save(out_path, flux)
                saved += 1
            except Exception as e:
                log.warning(f"{star_name} Q{i}: {e}")
                continue

        if saved > 0:
            log.info(f"{star_name}: saved {saved} quarters")
        return saved

    except Exception as e:
        log.error(f"Failed {star_name}: {e}")
        return 0


def download_all(out_dir="data/raw", seq_len=1024):
    os.makedirs(out_dir, exist_ok=True)

    # Deduplicate lists at runtime — safety net
    planets    = list(dict.fromkeys(CONFIRMED_PLANETS))
    nonplanets = list(dict.fromkeys(NON_PLANET_STARS))

    log.info(f"Planet hosts    : {len(planets)} unique stars")
    log.info(f"Non-planet stars: {len(nonplanets)} unique stars")

    total_saved = total_failed = 0

    log.info("── Downloading planet hosts (label=1) ──")
    for star in tqdm(planets, desc="Planets"):
        n = download_one(star, 1, out_dir, seq_len)
        if n > 0:
            total_saved += n
        else:
            total_failed += 1

    log.info("── Downloading non-planet stars (label=0) ──")
    for star in tqdm(nonplanets, desc="Non-planets"):
        n = download_one(star, 0, out_dir, seq_len)
        if n > 0:
            total_saved += n
        else:
            total_failed += 1

    log.info(f"Done — {total_saved} quarter files saved, "
             f"{total_failed} stars failed entirely.")

    # Final count
    all_files  = os.listdir(out_dir)
    n_planet   = sum(1 for f in all_files if "label1" in f)
    n_noplant  = sum(1 for f in all_files if "label0" in f)
    log.info(f"data/raw summary — planet: {n_planet}, non-planet: {n_noplant}, "
             f"total: {len(all_files)}")


if __name__ == "__main__":
    download_all()