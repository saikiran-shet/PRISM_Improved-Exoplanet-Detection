"""
Fetches confirmed planet hosts and false positives
from NASA Exoplanet Archive using the updated astroquery API.
"""
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
import pandas as pd

print("Fetching confirmed planets from NASA Exoplanet Archive...")

# Fetch CONFIRMED planets
confirmed_table = NasaExoplanetArchive.query_criteria(
    table="cumulative",
    select="kepid,koi_disposition",
    where="koi_disposition='CONFIRMED'"
)
confirmed_df = confirmed_table.to_pandas()
print(f"Confirmed planets  : {len(confirmed_df)}")

# Fetch FALSE POSITIVES
print("Fetching false positives...")
false_pos_table = NasaExoplanetArchive.query_criteria(
    table="cumulative",
    select="kepid,koi_disposition",
    where="koi_disposition='FALSE POSITIVE'"
)
false_pos_df = false_pos_table.to_pandas()
print(f"False positives    : {len(false_pos_df)}")

# Get unique KIC IDs
confirmed_kics = confirmed_df['kepid'].astype(int).unique().tolist()
false_pos_kics  = false_pos_df['kepid'].astype(int).unique().tolist()

print(f"\nUnique confirmed KICs : {len(confirmed_kics)}")
print(f"Unique false pos KICs : {len(false_pos_kics)}")

# Save to files
with open('confirmed_kics.txt', 'w') as f:
    for kic in confirmed_kics:
        f.write(f"{kic}\n")

with open('false_positive_kics.txt', 'w') as f:
    for kic in false_pos_kics:
        f.write(f"{kic}\n")

print(f"\nSaved → confirmed_kics.txt")
print(f"Saved → false_positive_kics.txt")
print(f"\nSample confirmed : {confirmed_kics[:5]}")
print(f"Sample false pos : {false_pos_kics[:5]}")