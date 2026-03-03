#!/usr/bin/env python3
"""
TEP-JWST Step 39: Environmental Screening Analysis

Tests the TEP prediction that galaxies in high-density environments (deep group/cluster potentials)
should be "screened" from the TEP effect, appearing more "standard" (younger/bluer) than
field galaxies of the same mass.

Methodology:
1. Load UNCOVER catalog (RA, Dec, z).
2. Compute Local Density (\\Sigma_5) using Nth nearest neighbor.
   - Performed in redshift slices to isolate physical associations.
3. Test Correlation: Density vs Age Ratio.
   - TEP Prediction: High Density -> Screened -> Lower \\Gamma_t -> Younger Appearance.
   - Expected: Negative correlation between Density and Age Ratio.
4. Control for Stellar Mass (as Mass correlates with Density).

Author: Matthew L. Smawfield
Date: January 2026
"""

import sys
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.tep_model import compute_gamma_t

STEP_NUM = "39"
STEP_NAME = "environment_screening"
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"

def load_catalog():
    """Load UNCOVER catalog."""
    # Use the processed CSV from Step 01 or similar
    path = PROJECT_ROOT / "results" / "interim" / "step_02_uncover_full_sample_tep.csv"
    if not path.exists():
        print_status("Catalog not found.", "ERROR")
        return None
    df = pd.read_csv(path)
    if 'z' not in df.columns and 'z_phot' in df.columns:
        df['z'] = pd.to_numeric(df['z_phot'], errors='coerce')
    
    # Ensure required columns
    if 'age_ratio' not in df.columns:
        # Calculate if missing
        # Check for direct columns first
        if 'mwa_Gyr' in df.columns and 't_cosmic_Gyr' in df.columns:
             df['age_ratio'] = df['mwa_Gyr'] / df['t_cosmic_Gyr']
        elif 'mwa' in df.columns and 't_cosmic' in df.columns:
            df['age_ratio'] = pd.to_numeric(df['mwa'], errors='coerce') / pd.to_numeric(df['t_cosmic'], errors='coerce')
        elif 'mwa' in df.columns and 'z' in df.columns:
            from astropy.cosmology import Planck18 as cosmo
            t_cosmic = cosmo.age(df['z'].values).value
            df['age_ratio'] = df['mwa'] / t_cosmic
        else:
            print_status("Missing columns to calculate age_ratio", "WARN")
            print_status(f"Available columns: {list(df.columns)}", "INFO")
            
    if 'gamma_t' not in df.columns:
        z_col = 'z' if 'z' in df.columns else 'z_phot'
        if 'log_Mh' in df.columns and z_col in df.columns:
            df['gamma_t'] = compute_gamma_t(df['log_Mh'], df[z_col])
        elif 'log_Mstar' in df.columns and z_col in df.columns:
            log_Mh = df['log_Mstar'] + 2.0
            df['gamma_t'] = compute_gamma_t(log_Mh, df[z_col])
    
    return df

def compute_density(df, n_neighbor=5, dz_slice=0.2):
    """Compute surface density Sigma_N."""
    
    # Initialize density column
    df['sigma_5'] = np.nan
    df['log_sigma_5'] = np.nan
    
    # Iterate through redshift slices
    z_min = df['z'].min()
    z_max = df['z'].max()
    
    # Sliding window
    z_centers = np.arange(z_min, z_max, dz_slice/2)
    
    densities = np.zeros(len(df)) * np.nan
    counts = np.zeros(len(df))
    
    print_status(f"Computing densities in z-slices (dz={dz_slice})...", "INFO")
    
    for z_c in z_centers:
        z_lo = z_c - dz_slice/2
        z_hi = z_c + dz_slice/2
        
        mask = (df['z'] >= z_lo) & (df['z'] < z_hi)
        if mask.sum() <= n_neighbor + 1:
            continue
            
        sub = df[mask]
        
        # Coordinates (RA/Dec approx flat for small field)
        # Proper distance would be better, but angular separation is fine for relative ranking
        coords = np.column_stack([sub['ra'], sub['dec']])
        
        tree = cKDTree(coords)
        
        # Query for N+1 neighbors (0 is self)
        dists, _ = tree.query(coords, k=n_neighbor+1)
        
        # Distance to Nth neighbor (degrees)
        d_nth = dists[:, n_neighbor]
        
        # Sigma ~ N / (pi * d_nth^2)
        # Units: number per sq degree
        sigma = n_neighbor / (np.pi * d_nth**2)
        
        # Store (average if multiple overlaps)
        # Actually just assign to valid indices. 
        # Overlaps might occur, we can just take the value from the 'best' centered slice
        # But sliding window is complex. Let's do non-overlapping bins or just accept overwrite.
        # Better: Assign to subset.
        
        # To avoid overwrite issues, we'll just do non-overlapping bins for simplicity first, 
        # or just overwrite and let the last write win (fine for dense grid).
        
        # Map back to original indices
        indices = sub.index
        densities[indices] = sigma
    
    df['sigma_5'] = densities
    df['log_sigma_5'] = np.log10(df['sigma_5'])
    
    # Filter undefined
    df_valid = df.dropna(subset=['log_sigma_5'])
    print_status(f"Computed densities for {len(df_valid)} galaxies.", "INFO")
    
    return df_valid

def analyze_screening(df):
    results = {}
    
    # 1. Density vs Age Ratio
    # TEP Prediction: Negative (High Density -> Screened -> Young)
    
    # Clean data
    sub = df.dropna(subset=['age_ratio', 'log_sigma_5', 'log_Mstar'])
    
    rho, p = stats.spearmanr(sub['log_sigma_5'], sub['age_ratio'])
    print_status(f"rho(Density, Age Ratio) = {rho:+.3f}, p = {p:.2e}", "INFO")
    results['rho_dens_age'] = rho
    results['p_dens_age'] = p
    
    # 2. Control for Mass
    # Mass correlates with Density (massive galaxies in dense regions)
    # Mass correlates with Age Ratio (TEP)
    # If Density -> Mass -> Age, we expect Positive correlation.
    # TEP Screening predicts Negative.
    
    # Partial Correlation: Density vs Age | Mass
    # Residuals
    slope_m, intercept_m, _, _, _ = stats.linregress(sub['log_Mstar'], sub['log_sigma_5'])
    resid_dens = sub['log_sigma_5'] - (slope_m * sub['log_Mstar'] + intercept_m)
    
    slope_a, intercept_a, _, _, _ = stats.linregress(sub['log_Mstar'], sub['age_ratio'])
    resid_age = sub['age_ratio'] - (slope_a * sub['log_Mstar'] + intercept_a)
    
    rho_part, p_part = stats.spearmanr(resid_dens, resid_age)
    print_status(f"rho(Density, Age | Mass) = {rho_part:+.3f}, p = {p_part:.2e}", "INFO")
    results['rho_partial'] = rho_part
    results['p_partial'] = p_part
    
    # 3. Density vs Gamma_t (Check consistency)
    # Gamma_t depends on Mass.
    # Is there an environmental dependence of Gamma_t? (Should be none by definition, unless we added environment term)
    # The pipeline definition of Gamma_t is Mass-based.
    # So this just checks Mass-Density correlation.
    rho_g, p_g = stats.spearmanr(sub['log_sigma_5'], sub['gamma_t'])
    print_status(f"rho(Density, Gamma_t) = {rho_g:+.3f} (Mass-Density link)", "INFO")
    
    # 4. Regime Comparison (Field vs Dense)
    threshold = sub['log_sigma_5'].median()
    field = sub[sub['log_sigma_5'] < threshold]
    dense = sub[sub['log_sigma_5'] >= threshold]
    
    age_field = field['age_ratio'].mean()
    age_dense = dense['age_ratio'].mean()
    
    print_status(f"Mean Age Ratio: Field = {age_field:.3f}, Dense = {age_dense:.3f}", "INFO")
    results['age_field'] = age_field
    results['age_dense'] = age_dense
    
    return results

def main():
    logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=PROJECT_ROOT / "logs" / f"step_{STEP_NUM}_{STEP_NAME}.log")
    set_step_logger(logger)
    
    print_status("Loading Catalog...", "INFO")
    df = load_catalog()
    if df is None: return
    
    df = compute_density(df)
    results = analyze_screening(df)
    
    # Interpretation
    if results['rho_partial'] < 0 and results['p_partial'] < 0.05:
        print_status("\nCONCLUSION: Screening Signature Detected.", "INFO")
        print_status("Denser environments show younger ages at fixed mass, consistent with TEP screening.", "INFO")
        results['conclusion'] = 'Screening Detected'
    else:
        print_status("\nCONCLUSION: No Screening Detected.", "INFO")
        results['conclusion'] = 'No Screening'
        
    # Save
    out_file = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print_status(f"Saved to {out_file}", "INFO")

if __name__ == "__main__":
    main()
