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

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging for step-level tracking
from scripts.utils.p_value_utils import format_p_value, safe_json_default  # Safe p-value formatting (prevents underflow) & JSON serialiser for numpy types
from scripts.utils.downloader import smart_download  # Robust HTTP download utility with integrity checking

STEP_NUM = "002"  # Pipeline step number (sequential, 001-176)
STEP_NAME = "tep_model"  # Used in log / output filenames for traceability

DATA_PATH = PROJECT_ROOT / "data"  # Top-level data directory for raw/external catalogs
INTERIM_PATH = PROJECT_ROOT / "results" / "interim"  # Pre-processed intermediate products (CSV intermediates between steps)
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"  # JSON output directory (machine-readable results)
LOGS_PATH = PROJECT_ROOT / "logs"  # Log directory (one log file per step for debugging)

for p in [INTERIM_PATH, OUTPUT_PATH, LOGS_PATH]:
    p.mkdir(parents=True, exist_ok=True)  # Create directory tree if missing; exist_ok prevents race conditions in parallel runs

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")  # Step-specific logger (isolated logs per step for debugging)
set_step_logger(logger)  # Register as global step logger so print_status() routes here

# =============================================================================
# TEP MODEL FUNCTIONS (Imported from Shared Utils)
# =============================================================================
#
# Mathematical constants imported from scripts/utils/tep_model.py:
#
# ALPHA_0 = 0.58  (dimensionless)
#   - Coupling strength from Cepheid calibration (Paper 12)
#   - Represents the fractional strength of the scalar field coupling to the metric
#   - Theoretical basis: alpha_0 arises from the conformal factor A(phi) = exp(alpha_0 * phi/M_pl)
#
# ALPHA_UNCERTAINTY = 0.16  (dimensionless)
#   - 1-sigma uncertainty from Cepheid distance ladder analysis
#   - Propagates to ~27% uncertainty in predicted Gamma_t at typical masses
#
# LOG_MH_REF = 12.0  (log10(M_halo/Msun))
#   - Reference halo mass where Gamma_t = 1 by definition
#   - Chosen near the knee of the SMHM relation where M* is well-constrained
#   - Mathematically: Gamma_t(M_h = M_ref, z = z_ref) = 1 exactly
#
# Z_REF = 5.5  (dimensionless)
#   - Reference redshift where the coupling strength is calibrated
#   - At z != z_ref, alpha(z) = alpha_0 * sqrt((1+z)/(1+z_ref))
#
# tep_alpha(z)  ->  alpha_0 * sqrt((1+z)/(1+z_ref))
#   - Redshift-dependent coupling from TEP theory (Paper 1, Eq. 3)
#   - Derivation: scalar field kinetic term scaling with cosmic density
#
# compute_gamma_t(log_Mh, z)  ->  exp[alpha(z) * (2/3) * (log_Mh - log_Mh_ref) * ((1+z)/(1+z_ref))]
#   - Exponential form ensures Gamma_t > 0 always
#   - The (2/3) factor comes from spherical collapse: t_dyn ~ 1/sqrt(G*rho) ~ M_h^(-1/2) * R^(3/2) ~ M_h^(2/3)
#   - The z_factor accounts for higher coupling strength at early times
#
# isochrony_mass_bias(gamma_t, n_ML=0.7)  ->  n_ML * log10(gamma_t)
#   - Predicted stellar mass bias from enhanced proper time
#   - n_ML ~ 0.7 from M/L ~ t^0.7 (mass-to-light evolves with age)
#   - Bias in dex: M*_obs / M*_true = Gamma_t^(n_ML)

from scripts.utils.tep_model import (
    ALPHA_0, ALPHA_UNCERTAINTY, LOG_MH_REF, Z_REF,
    tep_alpha, compute_gamma_t as tep_gamma, isochrony_mass_bias
)

# =============================================================================
# APPLY TEP MODEL
# =============================================================================

def apply_tep_model(df):
    """Apply TEP model to compute Gamma_t and derived quantities.

    For each galaxy, this function computes:

    1. alpha(z) = alpha_0 * sqrt(1+z)
       The redshift-dependent TEP coupling strength. The sqrt(1+z) factor
       arises because the scalar field gradient scales with the expansion
       rate, which increases at earlier epochs.

    2. Gamma_t = exp[ alpha(z) * (2/3) * (log_Mh - log_Mh_ref) * z_factor ]
       The chronological enhancement factor. Gamma_t encodes how much
       faster (>1) or slower (<1) proper time accumulates in a halo of
       mass M_h relative to the reference mass M_h_ref = 10^12 Msun.
       The (2/3) exponent maps halo mass to potential depth via the
       virial relation Phi ~ M^(2/3). The z_factor = (1+z)/(1+z_ref)
       accounts for the evolving NFW concentration at earlier epochs.

    3. t_eff = t_cosmic * Gamma_t
       The effective proper time experienced by stellar populations.
       SED fitting interprets this elapsed proper time as the apparent
       stellar age, so galaxies with Gamma_t > 1 appear older than the
       cosmic age at their redshift.

    4. ml_bias = Gamma_t^n_ml
       The isochrony mass-to-light bias. When SED fitting assumes
       standard time flow, a stellar population that has evolved for
       longer proper time is assigned a higher M/L ratio, leading to
       an overestimated stellar mass:
         M*_apparent / M*_true = Gamma_t^n_ml
       The exponent n_ml depends on the age-metallicity degeneracy
       and varies with redshift (calibrated in step_044 forward modeling):
         n_ml ~ 0.9 at z = 4-6  (moderate metallicity, strong M/L-age slope)
         n_ml ~ 0.5 at z > 6    (low metallicity, flatter M/L-age slope)
         n_ml = 0.7 default      (intermediate fallback)

    5. log_Mstar_true = log_Mstar - log10(ml_bias)
       The TEP-corrected (de-biased) stellar mass, removing the
       isochrony-induced overestimate.
    """
    
    df = df.copy()
    
    # alpha(z) = alpha_0 * sqrt(1+z): redshift-dependent coupling strength
    df['alpha_z'] = tep_alpha(df['z_phot'].values)
    
    # Gamma_t: chronological enhancement factor from halo mass and redshift
    df['gamma_t'] = tep_gamma(df['log_Mh'].values, df['z_phot'].values)
    
    # Effective proper time experienced by stellar populations [Gyr]
    # Gamma_t is always positive (exponential form), so t_eff > 0 always
    df['t_eff'] = df['t_cosmic'] * df['gamma_t']
    
    # Isochrony mass-to-light bias: M/L_apparent = M/L_true * Gamma_t^n_ml
    # n_ml is redshift-dependent because the slope of the M/L-age relation
    # changes with metallicity (which evolves with redshift)
    z = df['z_phot'].values
    n_ml = np.where(z > 6, 0.5, np.where(z > 4, 0.9, 0.7))
    df['n_ml'] = n_ml
    
    # Floor Gamma_t at 0.01 to prevent log(0) in edge cases
    df['ml_bias'] = np.power(np.maximum(df['gamma_t'].values, 0.01), n_ml)
    
    # TEP-corrected stellar mass: remove the isochrony overestimate
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
