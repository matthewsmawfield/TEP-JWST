#!/usr/bin/env python3
# Estimated runtime from last full canonical run (2026-03-09 15:52 UTC; full pipeline 32m18s): 0.4s.
"""
TEP-JWST Step 33: COSMOS-Web Data Download and Processing

Downloads the COSMOS-Web DR1 LePhare catalog with physical parameters
and processes it for TEP analysis.

Data source: Shuntov et al. (2025) - COSMOS2025 catalog
URL: https://cosmos2025.iap.fr/

Author: Matthew L. Smawfield
Date: January 2026
"""

import json
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging (severity levels: DEBUG/INFO/WARNING/ERROR/SUCCESS)
from scripts.utils.p_value_utils import format_p_value  # Safe p-value formatting (prevents floating-point underflow at p < 1e-300)
from scripts.utils.downloader import smart_download  # Robust HTTP download utility with integrity checking, resume capability, and authentication support

STEP_NUM = "033"  # Pipeline step number (sequential 001-176)
STEP_NAME = "cosmosweb_download"  # COSMOS-Web download: retrieves DR1 LePhare catalog (Shuntov et al. 2025) from IAP server

DATA_PATH = PROJECT_ROOT / "data" / "raw"  # Raw catalogue directory (external datasets from Zenodo/MAST/TACC/IAP)
INTERIM_PATH = PROJECT_ROOT / "data" / "interim"  # Processed intermediate products (CSV format for downstream steps)
LOGS_PATH = PROJECT_ROOT / "logs"  # Log directory (one plain-text log per step for debugging traceability)
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"  # JSON output directory (machine-readable statistical results)

for p in [DATA_PATH, INTERIM_PATH, LOGS_PATH]:
    p.mkdir(parents=True, exist_ok=True)  # Create directory tree if missing; exist_ok=True allows safe re-runs

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")  # Step-specific logger (isolated per-step logging for traceability)
set_step_logger(logger)  # Register as global step logger so print_status() routes to this step's log

# COSMOS-Web DR1 catalog (Shuntov et al. 2025 / COSMOS2025) – requires authentication
COSMOSWEB_BASE_URL = "https://cosmos2025.iap.fr/data/catalog"
COSMOSWEB_LEPHARE_FILE = DATA_PATH / "COSMOSWeb_mastercatalog_v1_lephare.fits"  # LePhare SED-fit catalog
COSMOSWEB_CIGALE_FILE = DATA_PATH / "COSMOSWeb_mastercatalog_v1_cigale.fits"  # CIGALE SED-fit catalog


def download_cosmosweb_catalog():
    """Download COSMOS-Web LePhare catalog if not already present."""
    if COSMOSWEB_LEPHARE_FILE.exists() and COSMOSWEB_LEPHARE_FILE.stat().st_size / 1e6 >= 200:
        print_status(f"Using existing local catalog: {COSMOSWEB_LEPHARE_FILE.name}", "INFO")
        return True

    username = os.environ.get('COSMOS_USER')
    password = os.environ.get('COSMOS_PASS')

    if not username or not password:
        print_status("COSMOS-Web download requires credentials.", "INFO")
        print_status("Set environment variables COSMOS_USER and COSMOS_PASS, or download manually.", "INFO")
        print_status(f"  Manual: curl -u USER:PASS -L -o {COSMOSWEB_LEPHARE_FILE} {COSMOSWEB_BASE_URL}/COSMOSWeb_mastercatalog_v1_lephare.fits", "INFO")
        return False

    url = f"{COSMOSWEB_BASE_URL}/COSMOSWeb_mastercatalog_v1_lephare.fits"
    return smart_download(
        url=url,
        dest=COSMOSWEB_LEPHARE_FILE,
        min_size_mb=200,
        auth=(username, password),
        logger=logger,
    )


def process_cosmosweb_catalog(force=False):
    """Process COSMOS-Web catalog and extract z>8 sample."""
    from astropy.table import Table
    import pandas as pd
    import numpy as np
    
    # Check if outputs already exist (backfill)
    z8_file = INTERIM_PATH / "cosmosweb_z8_sample.csv"
    highz_file = INTERIM_PATH / "cosmosweb_highz_sample.csv"
    
    if not force and z8_file.exists() and highz_file.exists():
        try:
            z8_df = pd.read_csv(z8_file)
            highz_df = pd.read_csv(highz_file)
            if len(z8_df) > 0 and len(highz_df) > 0:
                print_status(f"\nCOSMOS-Web processed data already exists:", "INFO")
                print_status(f"  High-z sample (z > 4): N = {len(highz_df)}", "INFO")
                print_status(f"  z > 8 sample: N = {len(z8_df)}", "INFO")
                return len(z8_df)
        except Exception as e:
            print_status(f"WARNING: Could not read cached COSMOS-Web data: {e}", "INFO")
    
    if not COSMOSWEB_LEPHARE_FILE.exists():
        print_status("ERROR: COSMOS-Web catalog not found. Run download first.", "INFO")
        return 0
    
    print_status("\nProcessing COSMOS-Web catalog...", "INFO")
    
    # Load catalog
    tab = Table.read(COSMOSWEB_LEPHARE_FILE)
    print_status(f"Total sources: {len(tab)}", "INFO")
    print_status(f"Columns: {tab.colnames[:20]}...", "INFO")
    
    # Find relevant columns
    z_cols = [c for c in tab.colnames if 'z_' in c.lower() or 'redshift' in c.lower() or c.lower() == 'z']
    mass_cols = [c for c in tab.colnames if 'mass' in c.lower() or 'mstar' in c.lower()]
    dust_cols = [c for c in tab.colnames if 'dust' in c.lower() or 'av' in c.lower() or 'ebv' in c.lower()]
    
    print_status(f"\nRedshift columns: {z_cols[:5]}", "INFO")
    print_status(f"Mass columns: {mass_cols[:5]}", "INFO")
    print_status(f"Dust columns: {dust_cols[:5]}", "INFO")
    
    # Filter to 1D columns
    names = [name for name in tab.colnames if len(tab[name].shape) <= 1]
    tab_1d = tab[names]
    df = tab_1d.to_pandas()
    
    # Identify best columns for z, mass, dust
    # COSMOS-Web uses: zfinal, mass_med, ebv_minchi2
    z_col = None
    for c in ['zfinal', 'zpdf_med', 'zPDF', 'z_phot', 'ZPHOT', 'z_best', 'Z_BEST', 'redshift']:
        if c in df.columns:
            z_col = c
            break
    
    mass_col = None
    for c in ['mass_med', 'mass_minchi2', 'MASS_MED', 'MASS_BEST', 'mass', 'log_mass', 'MASS']:
        if c in df.columns:
            mass_col = c
            break
    
    dust_col = None
    for c in ['ebv_minchi2', 'EBV_BEST', 'EBV_MED', 'AV_BEST', 'AV', 'ebv', 'dust']:
        if c in df.columns:
            dust_col = c
            break
    
    print_status(f"\nUsing columns: z={z_col}, mass={mass_col}, dust={dust_col}", "INFO")
    
    if z_col is None or mass_col is None:
        print_status("ERROR: Could not identify required columns", "INFO")
        print_status(f"Available columns: {list(df.columns)[:30]}", "INFO")
        return 0
    
    # Rename for consistency
    df = df.rename(columns={
        z_col: 'z_phot',
        mass_col: 'log_Mstar',
    })
    if dust_col:
        df = df.rename(columns={dust_col: 'dust'})
    
    # Check if mass is in log or linear
    if df['log_Mstar'].median() > 100:
        df['log_Mstar'] = np.log10(df['log_Mstar'])
    
    print_status(f"\nRedshift distribution:", "INFO")
    print_status(f"  z < 4: {(df['z_phot'] < 4).sum()}", "INFO")
    print_status(f"  4 <= z < 8: {((df['z_phot'] >= 4) & (df['z_phot'] < 8)).sum()}", "INFO")
    print_status(f"  z >= 8: {(df['z_phot'] >= 8).sum()}", "INFO")
    
    # Save high-z sample
    highz = df[(df['z_phot'] >= 4) & (df['z_phot'] <= 12)].copy()
    highz = highz.dropna(subset=['log_Mstar'])
    highz = highz[highz['log_Mstar'] > 6]
    
    cols_to_save = ['z_phot', 'log_Mstar']
    if 'dust' in highz.columns:
        cols_to_save.append('dust')
    
    highz[cols_to_save].to_csv(highz_file, index=False)
    print_status(f"\nSaved high-z sample (z > 4): N = {len(highz)} to {highz_file.name}", "INFO")
    
    # Save z > 8 sample
    z8 = df[(df['z_phot'] >= 8) & (df['z_phot'] <= 12)].copy()
    z8 = z8.dropna(subset=['log_Mstar'])
    z8 = z8[z8['log_Mstar'] > 6]
    if 'dust' in z8.columns:
        z8 = z8[z8['dust'] >= 0]
    
    z8[cols_to_save].to_csv(z8_file, index=False)
    print_status(f"Saved z > 8 sample: N = {len(z8)} to {z8_file.name}", "INFO")
    
    if len(z8) > 0:
        print_status(f"\nz > 8 sample statistics:", "INFO")
        print_status(f"  z range: {z8['z_phot'].min():.2f} - {z8['z_phot'].max():.2f}", "INFO")
        print_status(f"  log(M*) range: {z8['log_Mstar'].min():.2f} - {z8['log_Mstar'].max():.2f}", "INFO")
        if 'dust' in z8.columns:
            print_status(f"  Dust range: {z8['dust'].min():.2f} - {z8['dust'].max():.2f}", "INFO")
    
    return len(z8)


def main():
    print_status("=" * 70, "INFO")
    print_status("STEP 33: COSMOS-Web Data Download and Processing", "INFO")
    print_status("=" * 70, "INFO")
    print_status("", "INFO")
    
    # Download catalog
    success = download_cosmosweb_catalog()
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    if not success:
        print_status("\nWARNING: Could not download COSMOS-Web catalog", "INFO")
        print_status("Skipping COSMOS-Web analysis.", "INFO")
        status = {"status": "skipped", "reason": "COSMOS-Web catalog download failed or not available"}
        with open(OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json", "w") as _f:
            json.dump(status, _f, indent=2)
        return
    
    # Process catalog
    n_z8 = process_cosmosweb_catalog()
    
    print_status("", "INFO")
    print_status(f"Step {STEP_NUM} complete.", "INFO")
    if n_z8 > 0:
        print_status(f"COSMOS-Web z > 8 sample ready for analysis: N = {n_z8}", "INFO")
    else:
        print_status("No z > 8 sample extracted.", "INFO")
    status = {"status": "SUCCESS", "n_z8": n_z8, "catalog": "COSMOS-Web DR1 LePhare"}
    with open(OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json", "w") as _f:
        json.dump(status, _f, indent=2)


if __name__ == "__main__":
    main()
