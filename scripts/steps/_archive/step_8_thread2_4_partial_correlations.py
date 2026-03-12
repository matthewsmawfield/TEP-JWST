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

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, linregress
from pathlib import Path
import json

# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "results" / "interim"
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

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

def bootstrap_partial_ci(x, y, z_control, n_boot=1000):
    """Bootstrap CI for partial correlation."""
    n = len(x)
    rhos = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        rho, _ = partial_correlation(x[idx], y[idx], z_control[idx])
        rhos.append(rho)
    return np.percentile(rhos, [2.5, 97.5])

# =============================================================================
# THREAD 2: Γ_t vs Age Ratio
# =============================================================================

def test_thread_2(df):
    """Thread 2: Γ_t vs Age Ratio (partial correlation)."""
    print("\n" + "=" * 50)
    print("THREAD 2: Γ_t vs Age Ratio")
    print("=" * 50)
    
    gamma_t = df['gamma_t'].values
    age_ratio = df['age_ratio'].values
    z = df['z_phot'].values
    
    rho_raw, p_raw = spearmanr(gamma_t, age_ratio)
    print(f"Raw correlation: rho = {rho_raw:.3f}, p = {p_raw:.2e}")
    
    rho_partial, p_partial = partial_correlation(gamma_t, age_ratio, z)
    ci = bootstrap_partial_ci(gamma_t, age_ratio, z)
    
    significant = bool(p_partial < 0.001)
    
    print(f"  N = {len(df)}")
    print(f"  ρ(Γ_t, age_ratio | z) = {rho_partial:.3f}")
    print(f"  p = {p_partial:.2e}")
    print(f"  ★ SIGNIFICANT: {significant}")
    
    return {
        "test": "Thread 2: Γ_t vs Age Ratio",
        "n": int(len(df)),
        "raw_rho": float(rho_raw),
        "raw_p": float(p_raw),
        "partial_rho": float(rho_partial),
        "partial_p": float(p_partial),
        "ci_95": [float(ci[0]), float(ci[1])],
        "significant": significant,
        "interpretation": "Enhanced time → older appearance" if significant else "Inconclusive",
    }

# =============================================================================
# THREAD 3: Γ_t vs Metallicity
# =============================================================================

def test_thread_3(df):
    """Thread 3: Γ_t vs Metallicity (partial correlation)."""
    print("\n" + "=" * 50)
    print("THREAD 3: Γ_t vs Metallicity")
    print("=" * 50)
    
    gamma_t = df['gamma_t'].values
    met = df['met'].values
    z = df['z_phot'].values
    
    rho_raw, p_raw = spearmanr(gamma_t, met)
    print(f"Raw correlation: rho = {rho_raw:.3f}, p = {p_raw:.2e}")
    
    rho_partial, p_partial = partial_correlation(gamma_t, met, z)
    ci = bootstrap_partial_ci(gamma_t, met, z)
    
    significant = bool(p_partial < 0.001)
    
    print(f"  N = {len(df)}")
    print(f"  ρ(Γ_t, metallicity | z) = {rho_partial:.3f}")
    print(f"  p = {p_partial:.2e}")
    print(f"  ★ SIGNIFICANT: {significant}")
    
    return {
        "test": "Thread 3: Γ_t vs Metallicity",
        "n": int(len(df)),
        "raw_rho": float(rho_raw),
        "raw_p": float(p_raw),
        "partial_rho": float(rho_partial),
        "partial_p": float(p_partial),
        "ci_95": [float(ci[0]), float(ci[1])],
        "significant": significant,
        "interpretation": "Enhanced time → more enrichment" if significant else "Inconclusive",
    }

# =============================================================================
# THREAD 4: Γ_t vs Dust
# =============================================================================

def test_thread_4(df):
    """Thread 4: Γ_t vs Dust (partial correlation)."""
    print("\n" + "=" * 50)
    print("THREAD 4: Γ_t vs Dust")
    print("=" * 50)
    
    gamma_t = df['gamma_t'].values
    dust = df['dust'].values
    z = df['z_phot'].values
    
    rho_raw, p_raw = spearmanr(gamma_t, dust)
    print(f"Raw correlation: rho = {rho_raw:.3f}, p = {p_raw:.2e}")
    
    rho_partial, p_partial = partial_correlation(gamma_t, dust, z)
    ci = bootstrap_partial_ci(gamma_t, dust, z)
    
    significant = bool(p_partial < 0.001)
    
    print(f"  N = {len(df)}")
    print(f"  ρ(Γ_t, dust | z) = {rho_partial:.3f}")
    print(f"  p = {p_partial:.2e}")
    print(f"  ★ SIGNIFICANT: {significant}")
    
    return {
        "test": "Thread 4: Γ_t vs Dust",
        "n": int(len(df)),
        "raw_rho": float(rho_raw),
        "raw_p": float(p_raw),
        "partial_rho": float(rho_partial),
        "partial_p": float(p_partial),
        "ci_95": [float(ci[0]), float(ci[1])],
        "significant": significant,
        "interpretation": "Enhanced time → more dust processing" if significant else "Inconclusive",
    }

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("STEP 8: Threads 2-4 - Partial Correlations")
    print("=" * 60)
    
    df_full = pd.read_csv(INPUT_PATH / "uncover_full_sample_tep.csv")
    print(f"Loaded full sample: N = {len(df_full)}")
    
    df_multi = pd.read_csv(INPUT_PATH / "uncover_multi_property_sample_tep.csv")
    print(f"Loaded multi-property sample: N = {len(df_multi)}")
    
    result_2 = test_thread_2(df_full)
    with open(OUTPUT_PATH / "thread_2_gamma_age.json", "w") as f:
        json.dump(result_2, f, indent=2)
    
    result_3 = test_thread_3(df_multi)
    with open(OUTPUT_PATH / "thread_3_gamma_metallicity.json", "w") as f:
        json.dump(result_3, f, indent=2)
    
    result_4 = test_thread_4(df_multi)
    with open(OUTPUT_PATH / "thread_4_gamma_dust.json", "w") as f:
        json.dump(result_4, f, indent=2)
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Thread 2 (Γ_t vs Age):        {'✓' if result_2['significant'] else '✗'}")
    print(f"Thread 3 (Γ_t vs Metallicity): {'✓' if result_3['significant'] else '✗'}")
    print(f"Thread 4 (Γ_t vs Dust):        {'✓' if result_4['significant'] else '✗'}")
    print()
    print("Step 8 complete.")

if __name__ == "__main__":
    main()
