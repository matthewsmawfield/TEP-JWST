
import pandas as pd
from astropy.io import fits
import os

def inspect_csv(path):
    print(f"\n--- Inspecting {os.path.basename(path)} ---")
    try:
        df = pd.read_csv(path, nrows=5)
        print("Columns:", list(df.columns))
    except Exception as e:
        print(f"Error reading CSV: {e}")

def inspect_fits(path):
    print(f"\n--- Inspecting {os.path.basename(path)} ---")
    try:
        with fits.open(path) as hdul:
            hdul.info()
            for i, hdu in enumerate(hdul):
                if hasattr(hdu, 'columns'):
                    print(f"HDU {i} Columns: {hdu.columns.names}")
    except Exception as e:
        print(f"Error reading FITS: {e}")

# CSVs
inspect_csv("/Users/matthewsmawfield/www/TEP-JWST/data/interim/combined_spectroscopic_catalog.csv")
inspect_csv("/Users/matthewsmawfield/www/TEP-JWST/data/interim/jades_highz_physical.csv")
inspect_csv("/Users/matthewsmawfield/www/TEP-JWST/data/interim/uncover_highz_sed_properties.csv")

# FITS
inspect_fits("/Users/matthewsmawfield/www/TEP-JWST/data/raw/kokorev_lrd_catalog_v1.1.fits")
