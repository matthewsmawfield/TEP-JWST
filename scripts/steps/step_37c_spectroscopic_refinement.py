#!/usr/bin/env python3
"""
TEP-JWST Step 37c: Spectroscopic Refinement (Simpson's Paradox Check)

Investigates the discrepancy between the full sample negative correlation
and the strong positive correlation in the z=4-6 bin.

Author: Matthew L. Smawfield
Date: January 2026
"""

import sys
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import json
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.tep_model import compute_gamma_t as tep_gamma, stellar_to_halo_mass_behroozi_like

STEP_NUM = "37c"
STEP_NAME = "spectroscopic_refinement"
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"

def load_data():
    path = DATA_INTERIM / "combined_spectroscopic_catalog.csv"
    df = pd.read_csv(path)
    
    # Recalculate TEP quantities to be sure
    # Using the parameters from Step 37
    
    # Filter valid
    df = df.dropna(subset=['log_Mstar', 'mwa', 'z_spec'])
    df = df[df['log_Mstar'] > 6]
    
    # Calculate quantities
    from astropy.cosmology import Planck18 as cosmo
    
    # gamma_t
    log_mh = stellar_to_halo_mass_behroozi_like(df['log_Mstar'].values, df['z_spec'].values)
    df['gamma_t'] = tep_gamma(log_mh, df['z_spec'].values)
    
    df['t_cosmic'] = cosmo.age(df['z_spec'].values).value
    df['age_ratio'] = df['mwa'] / df['t_cosmic']
    
    # Filter outliers
    df = df[df['age_ratio'] < 5.0]
    
    return df

def analyze_bins(df):
    results = {}
    
    # Define bins
    bins = {
        'z4_6': (4, 6),
        'z6_8': (6, 8),
        'z8_plus': (8, 20),
        'full': (4, 20)
    }
    
    print_status(f"{'Bin':<10} | {'N':<5} | {'Rho':<8} | {'P-value':<10} | {'Mean AgeRatio':<15} | {'Mean Gamma':<15}", "INFO")
    print_status("-" * 80, "INFO")
    
    for name, (z_min, z_max) in bins.items():
        sub = df[(df['z_spec'] >= z_min) & (df['z_spec'] < z_max)]
        if len(sub) < 5:
            continue
            
        rho, p = stats.spearmanr(sub['gamma_t'], sub['age_ratio'])
        mean_age = sub['age_ratio'].mean()
        mean_gamma = sub['gamma_t'].mean()
        
        results[name] = {
            'n': len(sub),
            'rho': rho,
            'p': p,
            'mean_age_ratio': mean_age,
            'mean_gamma': mean_gamma
        }
        
        print_status(f"{name:<10} | {len(sub):<5} | {rho:+.3f}    | {p:.2e}     | {mean_age:.3f}           | {mean_gamma:.3f}", "INFO")

    return results

def check_simpsons_paradox(df):
    """
    Check if normalizing by redshift bin resolves the negative global trend.
    Normalize age_ratio and gamma_t within each redshift bin (z-score), then combine.
    """
    print_status("\n--- Checking Simpson's Paradox (Z-score Normalization) ---", "INFO")
    
    df['z_bin'] = pd.cut(df['z_spec'], bins=[4, 6, 8, 20], labels=['z4_6', 'z6_8', 'z8_plus'])
    
    # Calculate z-scores within bins
    df['age_ratio_norm'] = df.groupby('z_bin')['age_ratio'].transform(lambda x: (x - x.mean()) / x.std())
    df['gamma_t_norm'] = df.groupby('z_bin')['gamma_t'].transform(lambda x: (x - x.mean()) / x.std())
    
    # Drop NaNs (bins with 0 or 1 item might result in NaN std)
    df_norm = df.dropna(subset=['age_ratio_norm', 'gamma_t_norm'])
    
    rho, p = stats.spearmanr(df_norm['gamma_t_norm'], df_norm['age_ratio_norm'])
    
    print_status(f"Global correlation (Raw): rho = {stats.spearmanr(df['gamma_t'], df['age_ratio'])[0]:.3f}", "INFO")
    print_status(f"Global correlation (Bin-Normalized): rho = {rho:.3f}, p = {p:.2e}", "INFO")
    
    return {'rho_norm': rho, 'p_norm': p, 'n_norm': len(df_norm)}

def main():
    logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=PROJECT_ROOT / "logs" / f"step_{STEP_NUM}_{STEP_NAME}.log")
    set_step_logger(logger)
    
    print_status("Loading Data...", "INFO")
    df = load_data()
    print_status(f"Loaded {len(df)} sources.", "INFO")
    
    results = {}
    results['bin_analysis'] = analyze_bins(df)
    results['simpsons_check'] = check_simpsons_paradox(df)
    
    # Determine conclusion
    if results['bin_analysis']['z4_6']['p'] < 0.05 and results['simpsons_check']['rho_norm'] > 0:
        print_status("\nCONCLUSION: Simpson's Paradox Confirmed.", "INFO")
        print_status("The negative global trend is an artifact of combining redshift bins.", "INFO")
        print_status("Within bins (especially z=4-6), the TEP signal is POSITIVE and SIGNIFICANT.", "INFO")
    else:
        print_status("\nCONCLUSION: Mixed Signals.", "INFO")
    
    # Save
    out_file = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
        
    print_status(f"Saved to {out_file}", "INFO")

if __name__ == "__main__":
    main()
