#!/usr/bin/env python3
# Estimated runtime from last full canonical run (2026-03-09 15:52 UTC; full pipeline 32m18s): 4.4s.
"""
TEP-JWST Step 5: Threads 2-4 - Partial Correlations

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging (severity levels: DEBUG/INFO/WARNING/ERROR/SUCCESS)
from scripts.utils.p_value_utils import format_p_value, safe_json_default  # Safe p-value formatting (prevents underflow at p < 1e-300) & JSON serialiser
from scripts.utils.rank_stats import partial_rank_correlation, bootstrap_partial_rank_ci  # Partial Spearman: residualization method + bootstrap CI for partial rho

STEP_NUM = "005"  # Pipeline step number (005 out of 176 sequential steps)
STEP_NAME = "thread2_4_partial_correlations"  # Threads 2-4: Gamma_t correlations with age ratio, metallicity, dust (controlling for z)

DATA_PATH = PROJECT_ROOT / "data"  # Top-level data directory (raw external catalogs: UNCOVER, CEERS, COSMOS-Web, JADES)
INTERIM_PATH = PROJECT_ROOT / "results" / "interim"  # Pre-processed intermediate products (CSV format, human-readable for debugging)
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"  # JSON output directory (machine-readable statistical summaries)
LOGS_PATH = PROJECT_ROOT / "logs"  # Log directory (one plain-text file per step for execution tracing)

for p in [INTERIM_PATH, OUTPUT_PATH, LOGS_PATH]:
    p.mkdir(parents=True, exist_ok=True)  # Create directory tree if missing; parents=True ensures all intermediate directories exist

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")  # Step-specific logger (isolated logging per step for debugging traceability)
set_step_logger(logger)  # Register as global step logger so print_status() writes to this step's log file

# =============================================================================
# PARTIAL CORRELATION
# =============================================================================

def partial_correlation(x, y, z_control):
    """Spearman partial correlation between x and y controlling for z_control.

    Method (residualization):
      1. Regress ranks of x on ranks of z_control -> residuals r_x
      2. Regress ranks of y on ranks of z_control -> residuals r_y
      3. Compute Spearman rho between r_x and r_y

    This removes the confounding influence of z_control (e.g. redshift)
    so the remaining correlation reflects the direct x-y association.
    """
    rho, p, _ = partial_rank_correlation(x, y, z_control)
    return rho, p

def partial_correlation_double(x, y, z_control, mass_control, use_log_x=False):
    """Spearman partial correlation controlling for both redshift and stellar mass.

    This is a stricter test than single-variable control: any residual
    correlation between Gamma_t and a stellar-population property after
    removing both z and M* dependence cannot be attributed to the
    well-known mass-metallicity or mass-age scaling relations.
    """
    controls = np.column_stack([z_control, mass_control])
    rho, p, _ = partial_rank_correlation(x, y, controls)
    return rho, p

def bootstrap_partial_ci(x, y, z_control, n_boot=1000):
    """Bootstrap 95% CI for the single-control partial rank correlation.

    Resamples (x, y, z_control) rows with replacement n_boot times,
    recomputes the partial rho each time, and returns the 2.5th/97.5th
    percentiles of the bootstrap distribution.
    """
    return bootstrap_partial_rank_ci(x, y, z_control, n_boot=n_boot, seed=42)

def bootstrap_partial_ci_double(x, y, z_control, mass_control, n_boot=1000, use_log_x=False):
    """Bootstrap 95% CI for the double-control partial rank correlation."""
    controls = np.column_stack([z_control, mass_control])
    return bootstrap_partial_rank_ci(x, y, controls, n_boot=n_boot, seed=42)

# =============================================================================
# THREAD 2: Γ_t vs Age Ratio
# =============================================================================

def test_thread_2(df):
    """Thread 2: Gamma_t vs Age Ratio (partial correlation).

    TEP prediction: rho(Gamma_t, age_ratio | z) > 0.
    Galaxies in deeper potentials (higher Gamma_t) accumulate more
    proper time, so their SED-fitted mass-weighted ages appear closer
    to or exceeding the cosmic age. The age_ratio = mwa / t_cosmic
    should therefore correlate positively with Gamma_t after removing
    the trivial redshift dependence (younger universe -> lower t_cosmic).

    Two controls are applied:
      - Single control (z only): the primary test.
      - Double control (z + M*): rules out residual mass-age correlations
        that could mimic a TEP signal via the mass -> halo mass -> Gamma_t
        chain.
    """
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
    """Thread 3: Gamma_t vs Metallicity (partial correlation).

    TEP prediction: rho(Gamma_t, Z | z) > 0.
    If stellar populations in high-Gamma_t halos experience more proper
    time, they undergo more stellar generations and therefore produce
    more metals. The SED-fitted metallicity log(Z/Zsun) should correlate
    positively with Gamma_t after controlling for redshift.

    The double-control (z + M*) test checks whether the signal survives
    removal of the well-known mass-metallicity relation (MZR).
    """
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
    """Thread 4: Gamma_t vs Dust (partial correlation).

    TEP prediction: rho(Gamma_t, A_V | z) > 0.
    Dust production requires time: AGB stars need ~100-300 Myr of
    stellar evolution to reach the thermally-pulsing phase that
    produces carbonaceous and silicate grains. Galaxies with higher
    Gamma_t have more effective time for dust production, so their
    SED-fitted dust attenuation A_V should be higher at fixed redshift.

    The double-control (z + M*) test removes the mass-dust correlation
    that arises simply because more massive galaxies have more ISM.
    """
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
    
    summary = {
        'step': STEP_NUM, 'name': STEP_NAME,
        'thread2_gamma_age': result_2,
        'thread3_gamma_metallicity': result_3,
        'thread4_gamma_dust': result_4,
    }
    with open(OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json", "w") as f:
        json.dump(summary, f, indent=2, default=safe_json_default)
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
