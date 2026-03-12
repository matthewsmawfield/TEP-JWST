
from astropy.io import fits
import pandas as pd
import numpy as np

def scan_fits_for_columns(path, keywords):
    print(f"\nScanning {path} for keywords: {keywords}")
    try:
        with fits.open(path) as hdul:
            for i, hdu in enumerate(hdul):
                if hasattr(hdu, 'columns'):
                    cols = hdu.columns.names
                    found = [c for c in cols if any(k.lower() in c.lower() for k in keywords)]
                    if found:
                        print(f"HDU {i} Matches: {found}")
                    else:
                        print(f"HDU {i}: No matches.")
    except Exception as e:
        print(f"Error: {e}")

def scan_csv_for_columns(path, keywords):
    print(f"\nScanning {path} for keywords: {keywords}")
    try:
        df = pd.read_csv(path, nrows=5)
        cols = list(df.columns)
        found = [c for c in cols if any(k.lower() in c.lower() for k in keywords)]
        if found:
            print(f"Matches: {found}")
        else:
            print("No matches.")
    except Exception as e:
        print(f"Error: {e}")

keywords = ['sigma', 'vel', 'disp', 'width', 'fwhm', 'rad', 'size', 're_', 'r_eff', 'morph']

print("--- UNCOVER ---")
scan_fits_for_columns("/Users/matthewsmawfield/www/TEP-JWST/data/raw/uncover/UNCOVER_DR4_SPS_catalog.fits", keywords)
scan_fits_for_columns("/Users/matthewsmawfield/www/TEP-JWST/data/raw/uncover/UNCOVER_DR4_SPS_zspec_catalog.fits", keywords)

print("\n--- LRD (Kokorev) ---")
scan_fits_for_columns("/Users/matthewsmawfield/www/TEP-JWST/data/raw/kokorev_lrd_catalog_v1.1.fits", keywords)

print("\n--- JADES ---")
scan_fits_for_columns("/Users/matthewsmawfield/www/TEP-JWST/data/raw/JADES_z_gt_8_Candidates_Hainline_et_al.fits", keywords)
