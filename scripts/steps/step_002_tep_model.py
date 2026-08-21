#!/usr/bin/env python3
# Estimated runtime from last full canonical run (2026-03-09 15:52 UTC; full pipeline 32m18s): 0.7s.
"""
TEP-JWST Step 2: Potential-Depth and Observable-Response Calculation

This step computes the positive halo potential-depth proxy and the empirical
mass-to-light inference response for every galaxy in the sample.

TEP Observable Response (Potential-Linear Form):
    R_ML = exp[ K * (Psi - Psi_ref) * sqrt(1+z) ]

    where:
    - K is the magnitude-sector observable response coefficient
    - Psi = |Phi|/c^2 is positive potential depth
    - Psi_ref is the reference potential depth
    - sqrt(1+z) is the adopted response-evolution factor

R_ML is positive and is not the conformal factor A, a local proper-time ratio,
or the microscopic scalar coupling.

Inputs:
- results/interim/uncover_full_sample.csv
- results/interim/uncover_multi_property_sample.csv

Outputs:
- results/interim/uncover_full_sample_tep.csv (with R_ML)
- results/interim/uncover_multi_property_sample_tep.csv (with R_ML)
- results/outputs/step_002_tep_model.json
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

from scripts.utils.logger import (
    TEPLogger, set_step_logger, print_status,
    log_subsection, log_data, log_dict, log_timing,
)  # Centralised logging for step-level tracking
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
# KAPPA_GAL = 9.6e5 mag (Observable Response Coefficient)
#   - Calibrated from the Paper 11 environmental ladder response
#   - Distinct from beta_A, A(phi), and a local proper-time ratio
#
# KAPPA_GAL_UNCERTAINTY = 4.0e5 mag
#   - 1-sigma uncertainty from the distance-ladder response analysis
#
# LOG_MH_REF = 12.0  (log10(M_halo/Msun))
#   - Reference halo mass where R_ML = 1 by definition
#
# Z_REF = 5.5  (dimensionless)
#   - Reference redshift for response calculations
#
# compute_ml_response(log_Mh, z)
#   - Uses positive Psi = |Phi|/c^2 as the environmental predictor
#   - Returns R_ML > 0 without identifying it with a local clock rate
#
# ml_inference_bias(response, n_ML=0.7)
#   - Bias in inferred mass: M*_obs / M*_true = R_ML^(n_ML)

from scripts.utils.tep_model import (
    KAPPA_GAL, KAPPA_GAL_UNCERTAINTY, LOG_MH_REF, Z_REF,
    tep_alpha, compute_ml_response, compute_ml_response_self_consistent,
    ml_inference_bias, stellar_to_halo_mass_behroozi_like
)

# =============================================================================
# APPLY TEP MODEL
# =============================================================================

def apply_tep_model(df):
    """Compute the observable M/L response and derived inference quantities.

    The output retains legacy ``gamma_t`` and ``t_eff`` columns for file-format
    compatibility. They are aliases of ``ml_response`` and
    ``t_inferred_proxy`` and must not be interpreted as local clock rates or
    accumulated matter-frame proper time.

    Uses the self-consistent R_ML (damped fixed-point iteration) rather than
    the single-pass value.  The single-pass computation evaluates R_ML from
    the observed (biased) mass via log_Mh, creating a mass circularity that
    overcorrects high-mass galaxies.  The self-consistent solution iterates
    M*→Mh→R_ML→M*_true to a stable fixed point.
    """
    df = df.copy()
    z = df['z_phot'].values
    n_ml = np.where(z > 6, 0.5, np.where(z > 4, 0.9, 0.7))
    df['n_ml'] = n_ml

    # Self-consistent R_ML: iterate M*→Mh→R_ML→M*_true to convergence
    # The redshift-dependent n_ml array is passed through to the iteration
    df['response_z'] = tep_alpha(z)
    ml_response, log_mstar_true = compute_ml_response_self_consistent(
        df['log_Mstar'].values, z, n=n_ml
    )
    df['ml_response'] = ml_response
    df['gamma_t'] = df['ml_response']
    df['t_inferred_proxy'] = df['t_cosmic'] * df['ml_response']
    df['t_eff'] = df['t_inferred_proxy']

    # Update log_Mh from the corrected (true) stellar mass
    df['log_Mh'] = stellar_to_halo_mass_behroozi_like(log_mstar_true, z)

    df['ml_bias'] = ml_inference_bias(df['ml_response'].values, n_ml)
    df['log_Mstar_true'] = log_mstar_true

    return df

# =============================================================================
# MAIN
# =============================================================================

def main():
    print_status("STEP 002: TEP Model and R_ML Calculation", "TITLE")
    print_status("Computing the potential-depth environmental predictor and M/L inference response.", "INFO")
    print_status("")

    # ------------------------------------------------------------------
    # Stage 1: Display model parameters
    # ------------------------------------------------------------------
    log_subsection("Stage 1: TEP Model Parameters")

    print_status("Potential-linear R_ML formula:", "INFO")
    print_status("  R_ML = exp[ K * (Psi - Psi_ref) * sqrt(1+z) ]", "INFO")
    print_status("")
    log_data("kappa_gal (response coefficient)", f"{KAPPA_GAL} ± {KAPPA_GAL_UNCERTAINTY} mag")
    log_data("kappa_gal source", "Cepheid calibration (Paper 11), transferred via K_gal")
    log_data("log_Mh_ref (reference halo mass)", LOG_MH_REF)
    log_data("z_ref (reference redshift)", Z_REF)
    log_data("Redshift scaling", "alpha(z) = kappa_gal * sqrt(1+z)")
    print_status("")

    # ------------------------------------------------------------------
    # Stage 2: Load input data from step 001
    # ------------------------------------------------------------------
    log_subsection("Stage 2: Loading Input Samples")

    full_path = INTERIM_PATH / "step_001_uncover_full_sample.csv"
    multi_path = INTERIM_PATH / "step_001_uncover_multi_property_sample.csv"

    if not full_path.exists() or not multi_path.exists():
        print_status("ERROR: Input files from step 001 not found.", "ERROR")
        print_status(f"  Expected: {full_path.name}", "ERROR")
        print_status(f"  Expected: {multi_path.name}", "ERROR")
        return

    with log_timing("Loading full sample"):
        df_full = pd.read_csv(full_path)
    print_status(f"Full sample: N = {len(df_full)}", "INFO")

    with log_timing("Loading multi-property sample"):
        df_multi = pd.read_csv(multi_path)
    print_status(f"Multi-property sample: N = {len(df_multi)}", "INFO")
    print_status("")

    if len(df_full) == 0 or len(df_multi) == 0:
        print_status("ERROR: Input dataframes are empty.", "ERROR")
        return

    # ------------------------------------------------------------------
    # Stage 3: Apply TEP model
    # ------------------------------------------------------------------
    log_subsection("Stage 3: Computing R_ML and Derived Quantities")

    print_status("Computing per-galaxy:", "INFO")
    print_status("  (a) response_z = kappa_gal * sqrt(1+z) — response evolution", "INFO")
    print_status("  (b) R_ML = exp[...]                      — observable inference response", "INFO")
    print_status("  (c) t_inferred_proxy = t_cosmic * R_ML  — observer-side proxy [Gyr]", "INFO")
    print_status("  (d) ml_bias = R_ML^n_ml                  — mass-to-light inference bias", "INFO")
    print_status("  (e) log_Mstar_true = log_Mstar - log10(ml_bias) — corrected mass", "INFO")
    print_status("")

    with log_timing("Applying TEP model to full sample"):
        df_full = apply_tep_model(df_full)

    with log_timing("Applying TEP model to multi-property sample"):
        df_multi = apply_tep_model(df_multi)

    # ------------------------------------------------------------------
    # Stage 4: Report R_ML distribution statistics
    # ------------------------------------------------------------------
    log_subsection("Stage 4: R_ML Distribution Statistics")

    print_status("Full sample:", "INFO")
    log_data("N", len(df_full), indent=4)
    log_data("R_ML min", df_full['ml_response'].min(), indent=4)
    log_data("R_ML max", df_full['ml_response'].max(), indent=4)
    log_data("R_ML median", df_full['ml_response'].median(), indent=4)
    log_data("R_ML mean", df_full['ml_response'].mean(), indent=4)
    log_data("N (R_ML > 1)", int((df_full['ml_response'] > 1).sum()), indent=4)
    log_data("N (R_ML > 1.5)", int((df_full['ml_response'] > 1.5).sum()), indent=4)
    log_data("N (R_ML > 2.0)", int((df_full['ml_response'] > 2.0).sum()), indent=4)
    log_data("Inferred-time proxy range [Gyr]", f"[{df_full['t_inferred_proxy'].min():.3f}, {df_full['t_inferred_proxy'].max():.3f}]", indent=4)

    print_status("Multi-property sample:", "INFO")
    log_data("N", len(df_multi), indent=4)
    log_data("R_ML median", df_multi['ml_response'].median(), indent=4)
    log_data("Mass bias median (dex)", df_multi['ml_bias'].apply(np.log10).median(), indent=4)

    # Per-redshift breakdown
    print_status("Per-redshift R_ML (full sample):", "DEBUG")
    for z_lo, z_hi in [(4, 6), (6, 7), (7, 8), (8, 9), (9, 10)]:
        sub = df_full[(df_full['z_phot'] >= z_lo) & (df_full['z_phot'] < z_hi)]
        if len(sub) > 0:
            print_status(f"  z=[{z_lo},{z_hi}): N={len(sub):>5}, "
                         f"median R_ML={sub['ml_response'].median():.3f}, "
                         f"median inferred-time proxy={sub['t_inferred_proxy'].median():.3f} Gyr", "DEBUG")

    print_status("")

    # ------------------------------------------------------------------
    # Stage 5: Save outputs
    # ------------------------------------------------------------------
    log_subsection("Stage 5: Saving Outputs")

    with log_timing("Writing TEP-enriched CSVs and summary JSON"):
        df_full.to_csv(INTERIM_PATH / f"step_{STEP_NUM}_uncover_full_sample_tep.csv", index=False)
        print_status(f"Saved: step_{STEP_NUM}_uncover_full_sample_tep.csv ({len(df_full)} rows)", "INFO")

        df_multi.to_csv(INTERIM_PATH / f"step_{STEP_NUM}_uncover_multi_property_sample_tep.csv", index=False)
        print_status(f"Saved: step_{STEP_NUM}_uncover_multi_property_sample_tep.csv ({len(df_multi)} rows)", "INFO")

        summary = {
            "kappa_gal": KAPPA_GAL,
            "kappa_gal_uncertainty": KAPPA_GAL_UNCERTAINTY,
            "log_Mh_ref": LOG_MH_REF,
            "z_ref": Z_REF,
            "quantity_dictionary": {
                "potential_depth": "Psi = |Phi|/c^2 >= 0",
                "ml_response": "R_ML > 0; observable inference response",
                "local_clock_offset": "Delta ln A < 0 in a deeper well; not computed from KAPPA_GAL",
                "legacy_gamma_t_column": "Alias of ml_response for compatibility",
            },
            "ml_response_stats_full": {
                "min": float(df_full['ml_response'].min()),
                "max": float(df_full['ml_response'].max()),
                "median": float(df_full['ml_response'].median()),
                "n_gt_1": int((df_full['ml_response'] > 1).sum()),
                "n_gt_1p5": int((df_full['ml_response'] > 1.5).sum()),
            },
            "ml_response_stats_multi": {
                "min": float(df_multi['ml_response'].min()),
                "max": float(df_multi['ml_response'].max()),
                "median": float(df_multi['ml_response'].median()),
            },
            "gamma_t_stats_full": {
                "legacy_alias_for": "ml_response_stats_full",
                "min": float(df_full['ml_response'].min()),
                "max": float(df_full['ml_response'].max()),
                "median": float(df_full['ml_response'].median()),
                "n_gt_1": int((df_full['ml_response'] > 1).sum()),
                "n_gt_1p5": int((df_full['ml_response'] > 1.5).sum()),
            },
            "gamma_t_stats_multi": {
                "legacy_alias_for": "ml_response_stats_multi",
                "min": float(df_multi['ml_response'].min()),
                "max": float(df_multi['ml_response'].max()),
                "median": float(df_multi['ml_response'].median()),
            },
        }

        with open(OUTPUT_PATH / f"step_{STEP_NUM}_tep_model.json", "w") as f:
            json.dump(summary, f, indent=2, default=safe_json_default)
        print_status(f"Saved: step_{STEP_NUM}_tep_model.json", "INFO")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print_status("")
    print_status("Step 002 complete — TEP model applied:", "SUCCESS")
    print_status(f"  Full sample:       N = {len(df_full):>6}, median R_ML = {df_full['ml_response'].median():.3f}", "SUCCESS")
    print_status(f"  Multi-property:    N = {len(df_multi):>6}, median R_ML = {df_multi['ml_response'].median():.3f}", "SUCCESS")
    print_status(f"  R_ML range: [{df_full['ml_response'].min():.3f}, {df_full['ml_response'].max():.3f}]", "SUCCESS")
    print_status(f"  N(R_ML > 1): {int((df_full['ml_response'] > 1).sum())} ({100 * (df_full['ml_response'] > 1).mean():.1f}%)", "SUCCESS")

if __name__ == "__main__":
    main()
