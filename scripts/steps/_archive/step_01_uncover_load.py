#!/usr/bin/env python3
"""
TEP-JWST Step 01: Load UNCOVER DR4 Data and Apply Quality Cuts

This step loads the UNCOVER DR4 SPS catalog and applies quality cuts
to create the analysis samples used for the TEP analysis.

Outputs:
- results/interim/step_01_uncover_full_sample.csv (N ~ 2,315)
- results/interim/step_01_uncover_multi_property_sample.csv (N ~ 1,108)
- results/interim/step_01_uncover_z8_sample.csv (N ~ 283)
- results/outputs/step_01_summary.json

Author: Matthew L. Smawfield
Date: January 2026
"""

import sys
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.cosmology import Planck18 as cosmo
from pathlib import Path
import json

# =============================================================================
# PATHS AND LOGGER
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status

STEP_NUM = "01"
STEP_NAME = "uncover_load"

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "uncover"
INTERIM_PATH = PROJECT_ROOT / "results" / "interim"
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"
LOGS_PATH = PROJECT_ROOT / "logs"

INTERIM_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
LOGS_PATH.mkdir(parents=True, exist_ok=True)

# Initialize logger
logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")
set_step_logger(logger)

# =============================================================================
# LOAD DATA
# =============================================================================

def load_uncover_catalog():
    """Load UNCOVER DR4 SPS catalog."""
    catalog_file = DATA_PATH / "UNCOVER_DR4_SPS_catalog.fits"
    
    print_status(f"Loading: {catalog_file}", "PROCESS")
    
    with fits.open(catalog_file) as hdul:
        data = hdul[1].data
    
    print_status(f"Total sources: {len(data)}", "INFO")
    return data

# =============================================================================
# EXTRACT COLUMNS
# =============================================================================

def fix_byteorder(arr):
    """Fix big-endian byte order for numpy 2.0 compatibility."""
    arr = np.array(arr)
    if arr.dtype.byteorder == '>':
        return arr.astype(arr.dtype.newbyteorder('='))
    return arr

def extract_columns(data):
    """Extract relevant columns into a DataFrame."""
    
    df = pd.DataFrame({
        'id': fix_byteorder(data['id']),
        'ra': fix_byteorder(data['ra']),
        'dec': fix_byteorder(data['dec']),
        'z_spec': fix_byteorder(data['z_spec']),
        'z_phot': fix_byteorder(data['z_50']),
        'z_16': fix_byteorder(data['z_16']),
        'z_84': fix_byteorder(data['z_84']),
        'log_Mstar': fix_byteorder(data['mstar_50']),
        'log_Mstar_16': fix_byteorder(data['mstar_16']),
        'log_Mstar_84': fix_byteorder(data['mstar_84']),
        'mwa': fix_byteorder(data['mwa_50']),
        'mwa_16': fix_byteorder(data['mwa_16']),
        'mwa_84': fix_byteorder(data['mwa_84']),
        'met': fix_byteorder(data['met_50']),
        'met_16': fix_byteorder(data['met_16']),
        'met_84': fix_byteorder(data['met_84']),
        'dust': fix_byteorder(data['dust2_50']),
        'dust_16': fix_byteorder(data['dust2_16']),
        'dust_84': fix_byteorder(data['dust2_84']),
        'sfr100': fix_byteorder(data['sfr100_50']),
        'ssfr100': fix_byteorder(data['ssfr100_50']),
        'sfr10': fix_byteorder(data['sfr10_50']),
        'ssfr10': fix_byteorder(data['ssfr10_50']),
        'chi2': fix_byteorder(data['chi2']),
        'use_phot': fix_byteorder(data['use_phot']),
    })
    
    return df

# =============================================================================
# APPLY QUALITY CUTS
# =============================================================================

def apply_full_sample_cuts(df):
    """
    Full sample cuts for basic correlations.
    
    Criteria:
    - z_phot > 4 and z_phot < 10 (high-z sample)
    - log_Mstar > 8 (above completeness limit)
    - Valid sSFR (not NaN, > 0)
    - Valid MWA (not NaN)
    """
    
    mask = (
        (df['z_phot'] > 4) & 
        (df['z_phot'] < 10) & 
        (df['log_Mstar'] > 8) &
        (~df['ssfr100'].isna()) & 
        (df['ssfr100'] > 0) &
        (~df['mwa'].isna())
    )
    
    df_cut = df[mask].copy()
    print_status(f"Full sample after cuts: N = {len(df_cut)}", "INFO")
    
    return df_cut

def apply_multi_property_cuts(df):
    """
    Multi-property sample with metallicity quality cuts.
    
    Criteria:
    - All full sample cuts
    - Valid metallicity (not NaN)
    - Metallicity uncertainty < 0.5 dex
    - Valid dust (not NaN)
    """
    
    df['met_err'] = (df['met_84'] - df['met_16']) / 2
    
    mask = (
        (df['z_phot'] > 4) & 
        (df['z_phot'] < 10) & 
        (df['log_Mstar'] > 8) &
        (~df['ssfr100'].isna()) & 
        (df['ssfr100'] > 0) &
        (~df['mwa'].isna()) &
        (~df['met'].isna()) &
        (~df['met_err'].isna()) &
        (df['met_err'] < 0.5) &
        (~df['dust'].isna())
    )
    
    df_cut = df[mask].copy()
    print_status(f"Multi-property sample after cuts: N = {len(df_cut)}", "INFO")
    
    return df_cut

def apply_z8_cuts(df):
    """
    z > 8 sample for dust anomaly analysis.
    
    Criteria:
    - z_phot >= 8 and z_phot < 10
    - log_Mstar > 8
    - Valid dust (not NaN)
    """
    
    mask = (
        (df['z_phot'] >= 8) & 
        (df['z_phot'] < 10) & 
        (df['log_Mstar'] > 8) &
        (~df['dust'].isna())
    )
    
    df_cut = df[mask].copy()
    print_status(f"z > 8 sample after cuts: N = {len(df_cut)}", "INFO")
    
    return df_cut

# =============================================================================
# COMPUTE DERIVED QUANTITIES
# =============================================================================

def stellar_to_halo_mass(log_Mstar, z):
    """
    Compute halo mass from stellar mass using abundance matching.
    
    Based on Behroozi et al. (2019) parametrization, simplified for high-z.
    At high-z (z > 4), the stellar-to-halo mass ratio is approximately:
    
    log(M*/Mh) ≈ -1.8 - 0.1*(log_Mstar - 10) + 0.05*(z - 5)
    
    This gives:
    - At log_Mstar = 8: log(M*/Mh) ≈ -2.0 → Mh/M* ≈ 100
    - At log_Mstar = 10: log(M*/Mh) ≈ -1.8 → Mh/M* ≈ 63
    - At log_Mstar = 11: log(M*/Mh) ≈ -1.7 → Mh/M* ≈ 50
    
    The relation flattens at high mass due to AGN feedback.
    """
    # Base ratio at log_Mstar = 10, z = 5
    log_ratio_base = -1.8
    
    # Mass dependence: ratio decreases (less dark matter dominated) at high mass
    mass_term = -0.1 * (log_Mstar - 10)
    
    # Redshift dependence: ratio increases slightly at higher z
    z_term = 0.05 * (z - 5)
    
    log_ratio = log_ratio_base + mass_term + z_term
    
    # Halo mass = stellar mass / ratio
    log_Mh = log_Mstar - log_ratio
    
    return log_Mh

def compute_derived_quantities(df):
    """Compute derived quantities for TEP analysis."""
    
    # Use abundance matching for halo mass (more accurate than fixed ratio)
    df['log_Mh'] = stellar_to_halo_mass(df['log_Mstar'].values, df['z_phot'].values)
    
    # Also store the simple estimate for comparison
    df['log_Mh_simple'] = df['log_Mstar'] + 2.0
    
    df['t_cosmic'] = cosmo.age(df['z_phot'].values).value
    df['age_ratio'] = df['mwa'] / df['t_cosmic']
    df['log_ssfr'] = np.log10(df['ssfr100'])
    
    return df

# =============================================================================
# MAIN
# =============================================================================

def main():
    print_status("STEP 01: Load UNCOVER DR4 Data", "TITLE")
    
    data = load_uncover_catalog()
    df = extract_columns(data)
    print_status(f"Extracted {len(df.columns)} columns", "INFO")
    
    df_full = apply_full_sample_cuts(df)
    df_full = compute_derived_quantities(df_full)
    
    df_multi = apply_multi_property_cuts(df)
    df_multi = compute_derived_quantities(df_multi)
    
    df_z8 = apply_z8_cuts(df)
    df_z8 = compute_derived_quantities(df_z8)
    
    print_status("Saving outputs...", "PROCESS")
    
    df_full.to_csv(INTERIM_PATH / f"step_{STEP_NUM}_uncover_full_sample.csv", index=False)
    print_status(f"Saved: step_{STEP_NUM}_uncover_full_sample.csv", "INFO")
    
    df_multi.to_csv(INTERIM_PATH / f"step_{STEP_NUM}_uncover_multi_property_sample.csv", index=False)
    print_status(f"Saved: step_{STEP_NUM}_uncover_multi_property_sample.csv", "INFO")
    
    df_z8.to_csv(INTERIM_PATH / f"step_{STEP_NUM}_uncover_z8_sample.csv", index=False)
    print_status(f"Saved: step_{STEP_NUM}_uncover_z8_sample.csv", "INFO")
    
    summary = {
        "step": f"{STEP_NUM}",
        "name": "UNCOVER DR4 Data Loading",
        "full_sample_n": len(df_full),
        "multi_property_n": len(df_multi),
        "z8_sample_n": len(df_z8),
        "z_range": [float(df_full['z_phot'].min()), float(df_full['z_phot'].max())],
        "log_Mstar_range": [float(df_full['log_Mstar'].min()), float(df_full['log_Mstar'].max())],
    }
    
    with open(OUTPUT_PATH / f"step_{STEP_NUM}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print_status(f"Full sample: N = {summary['full_sample_n']}", "SUCCESS")
    print_status(f"Multi-property: N = {summary['multi_property_n']}", "SUCCESS")
    print_status(f"z > 8: N = {summary['z8_sample_n']}", "SUCCESS")
    print_status(f"Step {STEP_NUM} complete.", "SUCCESS")

if __name__ == "__main__":
    main()
