#!/usr/bin/env python3
"""
TEP-JWST Step 6: TEP Model and Gamma_t Calculation

This step applies the TEP model to compute chronological enhancement
factors (Gamma_t) for all galaxies in the sample.

TEP Model:
    Gamma_t = 1 + alpha(z) * (2/3) * (log_Mh - log_Mh_ref) * z_factor
    
    where:
    - alpha(z) = alpha_0 * sqrt(1 + z)
    - alpha_0 = 0.58 ± 0.16 (from Cepheid calibration, Paper 12)
    - log_Mh_ref = 12.0 (reference halo mass)
    - z_factor = (1 + z) / (1 + z_ref)
    - z_ref = 5.5 (reference redshift)

Inputs:
- results/interim/uncover_full_sample.csv
- results/interim/uncover_multi_property_sample.csv

Outputs:
- results/interim/uncover_full_sample_tep.csv (with Gamma_t)
- results/interim/uncover_multi_property_sample_tep.csv (with Gamma_t)
- results/interim/step_6_summary.json
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json

# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "results" / "interim"
OUTPUT_PATH = PROJECT_ROOT / "results" / "interim"

# =============================================================================
# TEP MODEL CONSTANTS
# =============================================================================

ALPHA_0 = 0.58
ALPHA_UNCERTAINTY = 0.16
LOG_MH_REF = 12.0
Z_REF = 5.5

# =============================================================================
# TEP MODEL FUNCTIONS
# =============================================================================

def tep_alpha(z, alpha_0=ALPHA_0):
    """
    Redshift-dependent TEP coupling.
    
    alpha(z) = alpha_0 * sqrt(1 + z)
    
    This scaling comes from the theoretical expectation that the
    TEP coupling increases with the background field strength,
    which scales as sqrt(1 + z) in the early universe.
    """
    return alpha_0 * np.sqrt(1 + z)

def tep_gamma(log_Mh, z, alpha_0=ALPHA_0):
    """
    TEP chronological enhancement factor.
    
    Gamma_t = 1 + alpha(z) * (2/3) * delta_log_Mh * z_factor
    
    where:
    - delta_log_Mh = log_Mh - log_Mh_ref
    - z_factor = (1 + z) / (1 + z_ref)
    
    For Gamma_t > 1: Enhanced proper time (deeper potential)
    For Gamma_t < 1: Reduced proper time (shallower potential)
    For Gamma_t = 1: Reference potential (log_Mh = 12)
    """
    alpha_z = tep_alpha(z, alpha_0)
    delta_log_Mh = log_Mh - LOG_MH_REF
    z_factor = (1 + z) / (1 + Z_REF)
    
    return 1.0 + alpha_z * (2/3) * delta_log_Mh * z_factor

def isochrony_mass_bias(gamma_t):
    """
    Mass-to-light ratio bias from isochrony assumption.
    
    M/L_apparent / M/L_true = Gamma_t^0.7
    
    This comes from the stellar population aging: older populations
    have higher M/L, and the M/L scales approximately as t^0.7.
    """
    return np.where(gamma_t > 0, np.power(np.maximum(gamma_t, 0.01), 0.7), 1.0)

# =============================================================================
# APPLY TEP MODEL
# =============================================================================

def apply_tep_model(df):
    """Apply TEP model to compute Gamma_t and derived quantities."""
    
    df = df.copy()
    
    df['alpha_z'] = tep_alpha(df['z_phot'].values)
    df['gamma_t'] = tep_gamma(df['log_Mh'].values, df['z_phot'].values)
    df['ml_bias'] = isochrony_mass_bias(df['gamma_t'].values)
    df['log_Mstar_true'] = df['log_Mstar'] - np.log10(df['ml_bias'])
    
    return df

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("STEP 6: TEP Model and Gamma_t Calculation")
    print("=" * 60)
    print()
    
    print("TEP Model Parameters:")
    print(f"  alpha_0 = {ALPHA_0} ± {ALPHA_UNCERTAINTY}")
    print(f"  log_Mh_ref = {LOG_MH_REF}")
    print(f"  z_ref = {Z_REF}")
    print()
    
    df_full = pd.read_csv(INPUT_PATH / "uncover_full_sample.csv")
    print(f"Loaded full sample: N = {len(df_full)}")
    
    df_multi = pd.read_csv(INPUT_PATH / "uncover_multi_property_sample.csv")
    print(f"Loaded multi-property sample: N = {len(df_multi)}")
    print()
    
    print("Applying TEP model...")
    df_full = apply_tep_model(df_full)
    df_multi = apply_tep_model(df_multi)
    
    print()
    print("Gamma_t Statistics (Full Sample):")
    print(f"  Min: {df_full['gamma_t'].min():.3f}")
    print(f"  Max: {df_full['gamma_t'].max():.3f}")
    print(f"  Median: {df_full['gamma_t'].median():.3f}")
    print(f"  N with Gamma_t > 1: {(df_full['gamma_t'] > 1).sum()}")
    print(f"  N with Gamma_t > 1.5: {(df_full['gamma_t'] > 1.5).sum()}")
    
    print()
    print("Saving outputs...")
    
    df_full.to_csv(OUTPUT_PATH / "uncover_full_sample_tep.csv", index=False)
    print(f"  -> {OUTPUT_PATH / 'uncover_full_sample_tep.csv'}")
    
    df_multi.to_csv(OUTPUT_PATH / "uncover_multi_property_sample_tep.csv", index=False)
    print(f"  -> {OUTPUT_PATH / 'uncover_multi_property_sample_tep.csv'}")
    
    summary = {
        "alpha_0": ALPHA_0,
        "alpha_uncertainty": ALPHA_UNCERTAINTY,
        "log_Mh_ref": LOG_MH_REF,
        "z_ref": Z_REF,
        "gamma_t_stats_full": {
            "min": float(df_full['gamma_t'].min()),
            "max": float(df_full['gamma_t'].max()),
            "median": float(df_full['gamma_t'].median()),
            "n_gt_1": int((df_full['gamma_t'] > 1).sum()),
            "n_gt_1p5": int((df_full['gamma_t'] > 1.5).sum()),
        },
        "gamma_t_stats_multi": {
            "min": float(df_multi['gamma_t'].min()),
            "max": float(df_multi['gamma_t'].max()),
            "median": float(df_multi['gamma_t'].median()),
        },
    }
    
    with open(OUTPUT_PATH / "step_6_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print()
    print("Step 6 complete.")

if __name__ == "__main__":
    main()
