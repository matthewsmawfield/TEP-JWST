#!/usr/bin/env python3
# Estimated runtime from last full canonical run (2026-03-09 15:52 UTC; full pipeline 32m18s): 0.7s.
"""
TEP-JWST Step 2: TEP Model and Gamma_t Calculation

This step applies the TEP model to compute chronological enhancement
factors (Gamma_t) for all galaxies in the sample.

TEP Model (Exponential Form from Paper 1):
    Gamma_t = exp[alpha(z) * (2/3) * (log_Mh - log_Mh_ref) * z_factor]
    
    where:
    - alpha(z) = alpha_0 * sqrt(1 + z)
    - alpha_0 = 0.58 ± 0.16 (from Cepheid calibration, Paper 12)
    - log_Mh_ref = 12.0 (reference halo mass)
    - z_factor = (1 + z) / (1 + z_ref)
    - z_ref = 5.5 (reference redshift)
    
    The exponential form ensures Gamma_t > 0 always:
    - Gamma_t > 1: Enhanced proper time (deeper potential)
    - Gamma_t < 1: Suppressed proper time (shallower potential)
    - Gamma_t = 1: Reference potential (log_Mh = 12)

Inputs:
- results/interim/uncover_full_sample.csv
- results/interim/uncover_multi_property_sample.csv

Outputs:
- results/interim/uncover_full_sample_tep.csv (with Gamma_t)
- results/interim/uncover_multi_property_sample_tep.csv (with Gamma_t)
- results/interim/step_6_summary.json
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import json


# =============================================================================
# PATHS AND LOGGER
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.p_value_utils import format_p_value, safe_json_default

STEP_NUM = "002"
STEP_NAME = "tep_model"

DATA_PATH = PROJECT_ROOT / "data"
INTERIM_PATH = PROJECT_ROOT / "results" / "interim"
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"
LOGS_PATH = PROJECT_ROOT / "logs"

for p in [INTERIM_PATH, OUTPUT_PATH, LOGS_PATH]:
    p.mkdir(parents=True, exist_ok=True)

# Initialize logger
logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")
set_step_logger(logger)

# =============================================================================
# TEP MODEL FUNCTIONS (Imported from Shared Utils)
# =============================================================================

from scripts.utils.tep_model import (
    ALPHA_0, ALPHA_UNCERTAINTY, LOG_MH_REF, Z_REF,
    tep_alpha, compute_gamma_t as tep_gamma, isochrony_mass_bias
)

# =============================================================================
# APPLY TEP MODEL
# =============================================================================

def apply_tep_model(df):
    """Apply TEP model to compute Gamma_t and derived quantities."""
    
    df = df.copy()
    
    # Core TEP quantities
    df['alpha_z'] = tep_alpha(df['z_phot'].values)
    df['gamma_t'] = tep_gamma(df['log_Mh'].values, df['z_phot'].values)
    
    # Effective time: t_eff = t_cosmic × Γ_t
    # This is the proper time experienced by stellar populations
    # With exponential form, Gamma_t is always positive
    df['t_eff'] = df['t_cosmic'] * df['gamma_t']
    
    # Isochrony bias: M/L_apparent / M/L_true = Γ_t^n
    # Redshift-dependent n from Step 44 forward-modeling validation:
    #   n ≈ 0.9 at z = 4–6  (moderate metallicity)
    #   n ≈ 0.5 at z > 6    (low metallicity; primary high-z analysis)
    #   n = 0.7 default for intermediate cases
    z = df['z_phot'].values
    n_ml = np.where(z > 6, 0.5, np.where(z > 4, 0.9, 0.7))
    df['n_ml'] = n_ml
    df['ml_bias'] = np.power(np.maximum(df['gamma_t'].values, 0.01), n_ml)
    df['log_Mstar_true'] = df['log_Mstar'] - np.log10(df['ml_bias'])
    
    return df

# =============================================================================
# MAIN
# =============================================================================

def main():
    print_status("=" * 60, "INFO")
    print_status("STEP 2: TEP Model and Gamma_t Calculation", "INFO")
    print_status("=" * 60, "INFO")
    print_status("", "INFO")
    
    print_status("TEP Model Parameters:", "INFO")
    print_status(f"  alpha_0 = {ALPHA_0} ± {ALPHA_UNCERTAINTY}", "INFO")
    print_status(f"  log_Mh_ref = {LOG_MH_REF}", "INFO")
    print_status(f"  z_ref = {Z_REF}", "INFO")
    print_status("", "INFO")
    
    full_path = INTERIM_PATH / f"step_001_uncover_full_sample.csv"
    multi_path = INTERIM_PATH / f"step_001_uncover_multi_property_sample.csv"
    
    if not full_path.exists() or not multi_path.exists():
        print_status("ERROR: Input files from step 001 not found.", "ERROR")
        return

    df_full = pd.read_csv(full_path)
    print_status(f"Loaded full sample: N = {len(df_full)}", "INFO")
    
    df_multi = pd.read_csv(multi_path)
    print_status(f"Loaded multi-property sample: N = {len(df_multi)}", "INFO")
    print_status("", "INFO")
    
    if len(df_full) == 0 or len(df_multi) == 0:
        print_status("ERROR: Input dataframes are empty.", "ERROR")
        return

    print_status("Applying TEP model...", "INFO")
    df_full = apply_tep_model(df_full)
    df_multi = apply_tep_model(df_multi)
    
    print_status("", "INFO")
    print_status("Gamma_t Statistics (Full Sample):", "INFO")
    print_status(f"  Min: {df_full['gamma_t'].min():.3f}", "INFO")
    print_status(f"  Max: {df_full['gamma_t'].max():.3f}", "INFO")
    print_status(f"  Median: {df_full['gamma_t'].median():.3f}", "INFO")
    print_status(f"  N with Gamma_t > 1: {(df_full['gamma_t'] > 1).sum()}", "INFO")
    print_status(f"  N with Gamma_t > 1.5: {(df_full['gamma_t'] > 1.5).sum()}", "INFO")
    
    print_status("", "INFO")
    print_status("Saving outputs...", "INFO")
    
    df_full.to_csv(INTERIM_PATH / f"step_{STEP_NUM}_uncover_full_sample_tep.csv", index=False)
    print_status(f"Saved: step_{STEP_NUM}_uncover_full_sample_tep.csv", "INFO")
    
    df_multi.to_csv(INTERIM_PATH / f"step_{STEP_NUM}_uncover_multi_property_sample_tep.csv", index=False)
    print_status(f"Saved: step_{STEP_NUM}_uncover_multi_property_sample_tep.csv", "INFO")
    
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
    
    with open(OUTPUT_PATH / f"step_{STEP_NUM}_tep_model.json", "w") as f:
        json.dump(summary, f, indent=2, default=safe_json_default)
    
    print_status("", "INFO")
    print_status(f"Step {STEP_NUM} complete.", "INFO")

if __name__ == "__main__":
    main()
