#!/usr/bin/env python3
"""
TEP-JWST Step 8: Threads 2-4 - Partial Correlations

This step tests threads 2-4 of TEP evidence using partial correlations
(controlling for redshift):

Thread 2: Γ_t vs Age Ratio
Thread 3: Γ_t vs Metallicity  
Thread 4: Γ_t vs Dust

TEP Prediction:
    Galaxies with higher predicted Γ_t should show:
    - Higher age ratios (older-appearing populations)
    - Higher metallicities (more enrichment time)
    - Higher dust content (more processing time)

Method:
    Partial correlation controlling for redshift via residualization.
    This removes the confounding effect of mass-z selection.

Inputs:
- results/interim/uncover_full_sample_tep.csv
- results/interim/uncover_multi_property_sample_tep.csv

Outputs:
- results/outputs/thread_2_gamma_age.json
- results/outputs/thread_3_gamma_metallicity.json
- results/outputs/thread_4_gamma_dust.json
"""

import sys
import numpy as np
np.random.seed(42)
import pandas as pd
from scipy.stats import spearmanr, linregress
from pathlib import Path
import json


# =============================================================================
# PATHS AND LOGGER
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.p_value_utils import format_p_value, safe_json_default

STEP_NUM = "04"
STEP_NAME = "thread2_4_partial_correlations"

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
# PARTIAL CORRELATION
# =============================================================================

def partial_correlation(x, y, z_control):
    """
    Compute partial correlation between x and y, controlling for z.
    
    Method: Residualize both x and y against z, then correlate residuals.
    """
    slope_x, int_x, _, _, _ = linregress(z_control, x)
    x_resid = x - (slope_x * z_control + int_x)
    
    slope_y, int_y, _, _, _ = linregress(z_control, y)
    y_resid = y - (slope_y * z_control + int_y)
    
    return spearmanr(x_resid, y_resid)

def partial_correlation_double(x, y, z_control, mass_control, use_log_x=False):
    """
    Compute partial correlation between x and y, controlling for BOTH z AND mass.
    
    This is more robust because Γ_t is derived from mass, so controlling for
    mass isolates the z-dependent component of TEP.
    
    Method: Multiple regression residualization.
    
    IMPORTANT: When x is gamma_t (exponential), set use_log_x=True to properly
    residualize log(gamma_t) instead of gamma_t. This is because gamma_t is
    exponential in z and mass, so linear residualization of gamma_t leaves
    nonlinear structure that creates spurious correlations.
    """
    # If x is exponential (like gamma_t), residualize log(x) instead
    if use_log_x:
        x_for_resid = np.log(np.maximum(x, 1e-10))
    else:
        x_for_resid = x
    
    # Residualize x against both z and mass
    X_controls = np.column_stack([z_control, mass_control, np.ones(len(z_control))])
    coeffs_x = np.linalg.lstsq(X_controls, x_for_resid, rcond=None)[0]
    x_resid = x_for_resid - X_controls @ coeffs_x
    
    # Residualize y against both z and mass
    coeffs_y = np.linalg.lstsq(X_controls, y, rcond=None)[0]
    y_resid = y - X_controls @ coeffs_y
    
    return spearmanr(x_resid, y_resid)

def bootstrap_partial_ci(x, y, z_control, n_boot=1000):
    """Bootstrap CI for partial correlation."""
    n = len(x)
    rhos = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        rho, _ = partial_correlation(x[idx], y[idx], z_control[idx])
        rhos.append(rho)
    return np.percentile(rhos, [2.5, 97.5])

def bootstrap_partial_ci_double(x, y, z_control, mass_control, n_boot=1000, use_log_x=False):
    """Bootstrap CI for double-control partial correlation."""
    n = len(x)
    rhos = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        rho, _ = partial_correlation_double(x[idx], y[idx], z_control[idx], mass_control[idx], use_log_x=use_log_x)
        rhos.append(rho)
    return np.percentile(rhos, [2.5, 97.5])

# =============================================================================
# THREAD 2: Γ_t vs Age Ratio
# =============================================================================

def test_thread_2(df):
    """Thread 2: Γ_t vs Age Ratio (partial correlation)."""
    print_status("\n" + "=" * 50, "INFO")
    print_status("THREAD 2: Γ_t vs Age Ratio", "INFO")
    print_status("=" * 50, "INFO")
    
    if len(df) < 10:
        print_status(f"ERROR: Insufficient sample size (N={len(df)}) for partial correlation analysis.", "ERROR")
        return {
            "test": "Thread 2: Γ_t vs Age Ratio",
            "n": int(len(df)),
            "significant": False,
            "significant_double": False,
            "interpretation": "Insufficient Data"
        }

    gamma_t = df['gamma_t'].values
    age_ratio = df['age_ratio'].values
    z = df['z_phot'].values
    mass = df['log_Mstar'].values
    
    rho_raw, p_raw = spearmanr(gamma_t, age_ratio)
    print_status(f"Raw correlation: rho = {rho_raw:.3f}, p = {p_raw:.2e}", "INFO")
    
    # Single control (z only)
    rho_partial, p_partial = partial_correlation(gamma_t, age_ratio, z)
    ci = bootstrap_partial_ci(gamma_t, age_ratio, z)
    
    # Double control (z AND mass) - use log(gamma_t) for proper residualization
    rho_double, p_double = partial_correlation_double(gamma_t, age_ratio, z, mass, use_log_x=True)
    ci_double = bootstrap_partial_ci_double(gamma_t, age_ratio, z, mass, use_log_x=True)
    
    significant = bool(p_partial < 0.001)
    significant_double = bool(p_double < 0.05)
    
    print_status(f"  N = {len(df)}", "INFO")
    print_status(f"  ρ(Γ_t, age_ratio | z) = {rho_partial:.3f}, p = {p_partial:.2e}", "INFO")
    print_status(f"  ρ(Γ_t, age_ratio | z, M*) = {rho_double:.3f}, p = {p_double:.2e}", "INFO")
    print_status(f"  ★ SIGNIFICANT (z-control): {significant}", "INFO")
    print_status(f"  ★ SIGNIFICANT (z+M* control): {significant_double}", "INFO")
    
    return {
        "test": "Thread 2: Γ_t vs Age Ratio",
        "n": int(len(df)),
        "raw_rho": float(rho_raw),
        "raw_p": format_p_value(p_raw),
        "partial_rho": float(rho_partial),
        "partial_p": format_p_value(p_partial),
        "ci_95": [float(ci[0]), float(ci[1])],
        "partial_rho_double": float(rho_double),
        "partial_p_double": format_p_value(p_double),
        "ci_95_double": [float(ci_double[0]), float(ci_double[1])],
        "significant": significant,
        "significant_double": significant_double,
        "interpretation": "Enhanced time → older appearance" if significant else "Inconclusive",
    }

# =============================================================================
# THREAD 3: Γ_t vs Metallicity
# =============================================================================

def test_thread_3(df):
    """Thread 3: Γ_t vs Metallicity (partial correlation)."""
    print_status("\n" + "=" * 50, "INFO")
    print_status("THREAD 3: Γ_t vs Metallicity", "INFO")
    print_status("=" * 50, "INFO")
    
    gamma_t = df['gamma_t'].values
    met = df['met'].values
    z = df['z_phot'].values
    mass = df['log_Mstar'].values
    
    rho_raw, p_raw = spearmanr(gamma_t, met)
    print_status(f"Raw correlation: rho = {rho_raw:.3f}, p = {p_raw:.2e}", "INFO")
    
    # Single control (z only)
    rho_partial, p_partial = partial_correlation(gamma_t, met, z)
    ci = bootstrap_partial_ci(gamma_t, met, z)
    
    # Double control (z AND mass) - more robust
    rho_double, p_double = partial_correlation_double(gamma_t, met, z, mass, use_log_x=True)
    ci_double = bootstrap_partial_ci_double(gamma_t, met, z, mass, use_log_x=True)
    
    significant = bool(p_partial < 0.001)
    significant_double = bool(p_double < 0.05)
    
    print_status(f"  N = {len(df)}", "INFO")
    print_status(f"  ρ(Γ_t, metallicity | z) = {rho_partial:.3f}, p = {p_partial:.2e}", "INFO")
    print_status(f"  ρ(Γ_t, metallicity | z, M*) = {rho_double:.3f}, p = {p_double:.2e}", "INFO")
    print_status(f"  ★ SIGNIFICANT (z-control): {significant}", "INFO")
    print_status(f"  ★ SIGNIFICANT (z+M* control): {significant_double}", "INFO")
    
    return {
        "test": "Thread 3: Γ_t vs Metallicity",
        "n": int(len(df)),
        "raw_rho": float(rho_raw),
        "raw_p": format_p_value(p_raw),
        "partial_rho": float(rho_partial),
        "partial_p": format_p_value(p_partial),
        "ci_95": [float(ci[0]), float(ci[1])],
        "partial_rho_double": float(rho_double),
        "partial_p_double": format_p_value(p_double),
        "ci_95_double": [float(ci_double[0]), float(ci_double[1])],
        "significant": significant,
        "significant_double": significant_double,
        "interpretation": "Enhanced time → more enrichment" if significant else "Inconclusive",
    }

# =============================================================================
# THREAD 4: Γ_t vs Dust
# =============================================================================

def test_thread_4(df):
    """Thread 4: Γ_t vs Dust (partial correlation)."""
    print_status("\n" + "=" * 50, "INFO")
    print_status("THREAD 4: Γ_t vs Dust", "INFO")
    print_status("=" * 50, "INFO")
    
    if len(df) < 10:
        print_status(f"ERROR: Insufficient sample size (N={len(df)}) for partial correlation analysis.", "ERROR")
        return {
            "test": "Thread 4: Γ_t vs Dust",
            "n": int(len(df)),
            "significant": False,
            "significant_double": False,
            "interpretation": "Insufficient Data"
        }

    gamma_t = df['gamma_t'].values
    dust = df['dust'].values
    z = df['z_phot'].values
    mass = df['log_Mstar'].values
    
    rho_raw, p_raw = spearmanr(gamma_t, dust)
    print_status(f"Raw correlation: rho = {rho_raw:.3f}, p = {p_raw:.2e}", "INFO")
    
    # Single control (z only)
    rho_partial, p_partial = partial_correlation(gamma_t, dust, z)
    ci = bootstrap_partial_ci(gamma_t, dust, z)
    
    # Double control (z AND mass) - more robust
    rho_double, p_double = partial_correlation_double(gamma_t, dust, z, mass, use_log_x=True)
    ci_double = bootstrap_partial_ci_double(gamma_t, dust, z, mass, use_log_x=True)
    
    significant = bool(p_partial < 0.001)
    significant_double = bool(p_double < 0.05)
    
    print_status(f"  N = {len(df)}", "INFO")
    print_status(f"  ρ(Γ_t, dust | z) = {rho_partial:.3f}, p = {p_partial:.2e}", "INFO")
    print_status(f"  ρ(Γ_t, dust | z, M*) = {rho_double:.3f}, p = {p_double:.2e}", "INFO")
    print_status(f"  ★ SIGNIFICANT (z-control): {significant}", "INFO")
    print_status(f"  ★ SIGNIFICANT (z+M* control): {significant_double}", "INFO")
    
    return {
        "test": "Thread 4: Γ_t vs Dust",
        "n": int(len(df)),
        "raw_rho": float(rho_raw),
        "raw_p": format_p_value(p_raw),
        "partial_rho": float(rho_partial),
        "partial_p": format_p_value(p_partial),
        "ci_95": [float(ci[0]), float(ci[1])],
        "partial_rho_double": float(rho_double),
        "partial_p_double": format_p_value(p_double),
        "ci_95_double": [float(ci_double[0]), float(ci_double[1])],
        "significant": significant,
        "significant_double": significant_double,
        "interpretation": "Enhanced time → more dust processing" if significant else "Inconclusive",
    }

# =============================================================================
# MAIN
# =============================================================================

def main():
    print_status("=" * 60, "INFO")
    print_status("STEP 8: Threads 2-4 - Partial Correlations", "INFO")
    print_status("=" * 60, "INFO")
    
    full_path = INTERIM_PATH / "step_002_uncover_full_sample_tep.csv"
    multi_path = INTERIM_PATH / "step_002_uncover_multi_property_sample_tep.csv"
    
    if not full_path.exists() or not multi_path.exists():
        print_status("ERROR: Input files from step 002 not found.", "ERROR")
        return

    df_full = pd.read_csv(full_path)
    print_status(f"Loaded full sample: N = {len(df_full)}", "INFO")
    
    df_multi = pd.read_csv(multi_path)
    print_status(f"Loaded multi-property sample: N = {len(df_multi)}", "INFO")
    
    result_2 = test_thread_2(df_full)
    with open(OUTPUT_PATH / f"step_{STEP_NUM}_thread2_gamma_age.json", "w") as f:
        json.dump(result_2, f, indent=2, default=safe_json_default)
    
    result_3 = test_thread_3(df_multi)
    with open(OUTPUT_PATH / f"step_{STEP_NUM}_thread3_gamma_metallicity.json", "w") as f:
        json.dump(result_3, f, indent=2, default=safe_json_default)
    
    result_4 = test_thread_4(df_multi)
    with open(OUTPUT_PATH / f"step_{STEP_NUM}_thread4_gamma_dust.json", "w") as f:
        json.dump(result_4, f, indent=2, default=safe_json_default)
    
    print_status("\n" + "=" * 50, "INFO")
    print_status("SUMMARY", "INFO")
    print_status("=" * 50, "INFO")
    print_status(f"Thread 2 (Γ_t vs Age):        {'✓' if result_2['significant'] else '✗'}", "INFO")
    print_status(f"Thread 3 (Γ_t vs Metallicity): {'✓' if result_3['significant'] else '✗'}", "INFO")
    print_status(f"Thread 4 (Γ_t vs Dust):        {'✓' if result_4['significant'] else '✗'}", "INFO")
    print_status("", "INFO")
    print_status(f"Step {STEP_NUM} complete.", "INFO")

if __name__ == "__main__":
    main()
