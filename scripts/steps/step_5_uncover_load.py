#!/usr/bin/env python3
"""
TEP-JWST Step 5: Load UNCOVER DR4 Data and Apply Quality Cuts

This step loads the UNCOVER DR4 SPS catalog and applies quality cuts
to create the analysis samples used for the seven-thread TEP analysis.

Outputs:
- results/interim/uncover_full_sample.csv (N ~ 2,315)
- results/interim/uncover_multi_property_sample.csv (N ~ 1,108)
- results/interim/uncover_z8_sample.csv (N ~ 283)
- results/interim/step_5_summary.json
"""

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.cosmology import Planck18 as cosmo
from pathlib import Path
import json

# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "uncover"
OUTPUT_PATH = PROJECT_ROOT / "results" / "interim"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# =============================================================================
# LOAD DATA
# =============================================================================

def load_uncover_catalog():
    """Load UNCOVER DR4 SPS catalog."""
    catalog_file = DATA_PATH / "UNCOVER_DR4_SPS_catalog.fits"
    
    print(f"Loading: {catalog_file}")
    
    with fits.open(catalog_file) as hdul:
        data = hdul[1].data
    
    print(f"Total sources: {len(data)}")
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
    print(f"Full sample after cuts: N = {len(df_cut)}")
    
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
    print(f"Multi-property sample after cuts: N = {len(df_cut)}")
    
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
    print(f"z > 8 sample after cuts: N = {len(df_cut)}")
    
    return df_cut

# =============================================================================
# COMPUTE DERIVED QUANTITIES
# =============================================================================

def compute_derived_quantities(df):
    """Compute derived quantities for TEP analysis."""
    
    df['log_Mh'] = df['log_Mstar'] + 2.0
    df['t_cosmic'] = cosmo.age(df['z_phot'].values).value
    df['age_ratio'] = df['mwa'] / df['t_cosmic']
    df['log_ssfr'] = np.log10(df['ssfr100'])
    
    return df

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("STEP 5: Load UNCOVER DR4 Data")
    print("=" * 60)
    print()
    
    data = load_uncover_catalog()
    df = extract_columns(data)
    print(f"Extracted {len(df.columns)} columns")
    print()
    
    df_full = apply_full_sample_cuts(df)
    df_full = compute_derived_quantities(df_full)
    
    df_multi = apply_multi_property_cuts(df)
    df_multi = compute_derived_quantities(df_multi)
    
    df_z8 = apply_z8_cuts(df)
    df_z8 = compute_derived_quantities(df_z8)
    
    print()
    print("Saving outputs...")
    
    df_full.to_csv(OUTPUT_PATH / "uncover_full_sample.csv", index=False)
    print(f"  -> {OUTPUT_PATH / 'uncover_full_sample.csv'}")
    
    df_multi.to_csv(OUTPUT_PATH / "uncover_multi_property_sample.csv", index=False)
    print(f"  -> {OUTPUT_PATH / 'uncover_multi_property_sample.csv'}")
    
    df_z8.to_csv(OUTPUT_PATH / "uncover_z8_sample.csv", index=False)
    print(f"  -> {OUTPUT_PATH / 'uncover_z8_sample.csv'}")
    
    summary = {
        "full_sample_n": len(df_full),
        "multi_property_n": len(df_multi),
        "z8_sample_n": len(df_z8),
        "z_range": [float(df_full['z_phot'].min()), float(df_full['z_phot'].max())],
        "log_Mstar_range": [float(df_full['log_Mstar'].min()), float(df_full['log_Mstar'].max())],
    }
    
    with open(OUTPUT_PATH / "step_5_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print()
    print("Summary:")
    print(f"  Full sample: N = {summary['full_sample_n']}")
    print(f"  Multi-property: N = {summary['multi_property_n']}")
    print(f"  z > 8: N = {summary['z8_sample_n']}")
    print()
    print("Step 5 complete.")

if __name__ == "__main__":
    main()
