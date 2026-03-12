#!/usr/bin/env python3
# Estimated runtime from last full canonical run (2026-03-09 15:52 UTC; full pipeline 32m18s): 0.4s.
"""
TEP-JWST Step 31: CEERS Data Download and Processing

Downloads the CEERS DR1 photometric catalog from the Texas Advanced Computing
Center and processes it for TEP analysis.

Data source: Cox et al. (2025) - The CEERS Photometric and Physical Parameter Catalog
URL: https://web.corral.tacc.utexas.edu/ceersdata/DR1/Catalog/ceers_cat_v1.0.fits.gz

Author: Matthew L. Smawfield
Date: January 2026
"""

import json
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.downloader import smart_download

STEP_NUM = "031"
STEP_NAME = "ceers_download"

DATA_PATH = PROJECT_ROOT / "data" / "raw"
INTERIM_PATH = PROJECT_ROOT / "data" / "interim"
LOGS_PATH = PROJECT_ROOT / "logs"
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"

for p in [DATA_PATH, INTERIM_PATH, LOGS_PATH]:
    p.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")
set_step_logger(logger)

CEERS_CATALOG_URL = "https://web.corral.tacc.utexas.edu/ceersdata/DR1/Catalog/ceers_cat_v1.0.fits.gz"
CEERS_CATALOG_FILE = DATA_PATH / "ceers_cat_v1.0.fits.gz"


def download_ceers_catalog():
    """Download CEERS DR1 catalog if not already present."""
    return smart_download(
        url=CEERS_CATALOG_URL,
        dest=CEERS_CATALOG_FILE,
        min_size_mb=100,
        logger=logger,
    )


def process_ceers_catalog(force=False):
    """Process CEERS catalog and extract z>8 sample.
    
    Args:
        force: If True, reprocess even if output files exist.
    """
    from astropy.table import Table
    import pandas as pd
    import numpy as np
    
    # Check if outputs already exist (backfill)
    z8_file = INTERIM_PATH / "ceers_z8_sample.csv"
    highz_file = INTERIM_PATH / "ceers_highz_sample.csv"
    
    if not force and z8_file.exists() and highz_file.exists():
        # Verify files have content
        try:
            z8_df = pd.read_csv(z8_file)
            highz_df = pd.read_csv(highz_file)
            if len(z8_df) > 0 and len(highz_df) > 0:
                print_status(f"\nCEERS processed data already exists:", "INFO")
                print_status(f"  High-z sample (z > 4): N = {len(highz_df)}", "INFO")
                print_status(f"  z > 8 sample: N = {len(z8_df)}", "INFO")
                return len(z8_df)
        except Exception:
            pass  # Fall through to reprocess
    
    print_status("\nProcessing CEERS catalog...", "INFO")
    
    # Load catalog
    tab = Table.read(CEERS_CATALOG_FILE)
    print_status(f"Total sources: {len(tab)}", "INFO")
    
    # Filter to 1D columns only (avoid multidimensional aperture columns)
    names = [name for name in tab.colnames if len(tab[name].shape) <= 1]
    tab_1d = tab[names]
    
    # Convert to pandas
    df = tab_1d.to_pandas()
    
    # Rename columns for consistency with UNCOVER
    df = df.rename(columns={
        'NUMBER': 'id',
        'RA': 'ra',
        'DEC': 'dec',
        'LP_Z_BEST': 'z_phot',
        'LP_Z_MED': 'z_med',
        'LP_MASS_BEST': 'log_Mstar',
        'LP_MASS_MED': 'log_Mstar_med',
        'LP_SFR_BEST': 'sfr',
        'LP_EBV_BEST': 'ebv',
        'DB_AV': 'dust',
        'DB_MASS': 'log_Mstar_db',
        'DB_SFR': 'sfr_db',
    })
    
    print_status(f"\nRedshift distribution:", "INFO")
    print_status(f"  z < 4: {(df['z_phot'] < 4).sum()}", "INFO")
    print_status(f"  4 <= z < 8: {((df['z_phot'] >= 4) & (df['z_phot'] < 8)).sum()}", "INFO")
    print_status(f"  z >= 8: {(df['z_phot'] >= 8).sum()}", "INFO")
    
    # Save full high-z sample (z > 4)
    highz = df[(df['z_phot'] >= 4) & (df['z_phot'] <= 12)].copy()
    highz = highz.dropna(subset=['log_Mstar'])
    highz = highz[highz['log_Mstar'] > 6]
    
    highz_file = INTERIM_PATH / "ceers_highz_sample.csv"
    highz[['id', 'ra', 'dec', 'z_phot', 'log_Mstar', 'sfr', 'dust']].to_csv(highz_file, index=False)
    print_status(f"\nSaved high-z sample (z > 4): N = {len(highz)} to {highz_file.name}", "INFO")
    
    # Save z > 8 sample for TEP analysis
    z8 = df[(df['z_phot'] >= 8) & (df['z_phot'] <= 12)].copy()
    z8 = z8.dropna(subset=['log_Mstar', 'dust'])
    z8 = z8[z8['log_Mstar'] > 6]
    z8 = z8[z8['dust'] >= 0]
    
    z8_file = INTERIM_PATH / "ceers_z8_sample.csv"
    z8[['id', 'ra', 'dec', 'z_phot', 'log_Mstar', 'sfr', 'dust']].to_csv(z8_file, index=False)
    print_status(f"Saved z > 8 sample: N = {len(z8)} to {z8_file.name}", "INFO")
    
    if len(z8) > 0:
        print_status(f"\nz > 8 sample statistics:", "INFO")
        print_status(f"  z range: {z8['z_phot'].min():.2f} - {z8['z_phot'].max():.2f}", "INFO")
        print_status(f"  log(M*) range: {z8['log_Mstar'].min():.2f} - {z8['log_Mstar'].max():.2f}", "INFO")
        print_status(f"  A_V range: {z8['dust'].min():.2f} - {z8['dust'].max():.2f}", "INFO")
    
    return len(z8)


def main():
    print_status("=" * 70, "INFO")
    print_status("STEP 34: CEERS Data Download and Processing", "INFO")
    print_status("=" * 70, "INFO")
    print_status("", "INFO")
    
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    # Download catalog
    success = download_ceers_catalog()
    if not success:
        print_status("\nERROR: Failed to download CEERS catalog", "INFO")
        status = {"status": "skipped", "reason": "CEERS catalog download failed or not available"}
        with open(OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json", "w") as _f:
            json.dump(status, _f, indent=2)
        return
    
    # Process catalog
    n_z8 = process_ceers_catalog()
    
    print_status("", "INFO")
    print_status(f"Step {STEP_NUM} complete.", "INFO")
    print_status(f"CEERS z > 8 sample ready for analysis: N = {n_z8}", "INFO")
    status = {"status": "complete", "n_z8": n_z8, "catalog": "CEERS DR1"}
    with open(OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json", "w") as _f:
        json.dump(status, _f, indent=2)


if __name__ == "__main__":
    main()
