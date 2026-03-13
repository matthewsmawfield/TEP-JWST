#!/usr/bin/env python3
# Estimated runtime from last full canonical run (2026-03-09 15:52 UTC; full pipeline 32m18s): 0.7s.
"""
TEP-JWST Step 030: z>8 Dust Anomaly - Quantitative Prediction Test

The z>8 dust anomaly is the paper's strongest evidence. This step provides
a rigorous quantitative test:

1. Standard physics predicts: At z>8, t_cosmic < 600 Myr is insufficient
   for significant dust production via AGB stars (requires ~300 Myr)
   
2. TEP predicts: Effective time t_eff = t_cosmic × Γ_t can exceed
   the AGB threshold even when t_cosmic does not

3. Test: Do galaxies with t_eff > 300 Myr have significantly more dust
   than those with t_eff < 300 Myr, even when t_cosmic is similar?

Author: Matthew L. Smawfield
Date: January 2026
"""

import sys
import numpy as np
import pandas as pd
from scipy import stats  # Hypothesis tests (Mann-Whitney, Spearman)
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging (severity levels: DEBUG/INFO/WARNING/ERROR/SUCCESS)
from scripts.utils.p_value_utils import format_p_value, safe_json_default  # Safe p-value formatting (prevents floating-point underflow at p < 1e-300) & JSON serialiser
from scripts.utils.rank_stats import partial_rank_correlation  # Partial Spearman: residualization method to control for confounders

STEP_NUM = "030"  # Pipeline step number (sequential 001-176)
STEP_NAME = "z8_dust_prediction"  # z>8 dust prediction: tests AGB threshold (300 Myr) via t_eff = t_cosmic × Gamma_t

INTERIM_PATH = PROJECT_ROOT / "results" / "interim"  # Pre-processed intermediate products (CSV format for step-to-step data flow)
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"  # JSON output directory (machine-readable statistical results)
LOGS_PATH = PROJECT_ROOT / "logs"  # Log directory (one plain-text log per step for debugging traceability)

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)  # Create results/outputs/ if missing; parents=True ensures full path tree exists
LOGS_PATH.mkdir(parents=True, exist_ok=True)  # Create logs/ if missing

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")  # Step-specific logger (isolated per-step logging for traceability)
set_step_logger(logger)  # Register as global step logger so print_status() routes to this step's log

# AGB dust production threshold: stellar evolution models require ~300 Myr for
# significant AGB-driven dust yields (e.g. Valiante+2009, Schneider+2014).
T_AGB_THRESHOLD = 300  # Myr


def load_z8_data():
    """Load z>8 sample with TEP calculations."""
    data_path = INTERIM_PATH / "step_002_uncover_full_sample_tep.csv"
    
    if not data_path.exists():
        print_status(f"ERROR: Input file not found: {data_path}", "ERROR")
        return None
        
    df = pd.read_csv(data_path)
    
    # Filter for z > 8
    df = df[df['z_phot'] >= 8].copy()
    df = df.dropna(subset=['dust', 'log_Mstar', 'gamma_t', 't_cosmic'])
    
    # Compute effective time: t_eff = t_cosmic × Γ_t (TEP-enhanced timescale)
    df['t_cosmic_Myr'] = df['t_cosmic'] * 1000  # Convert Gyr → Myr
    df['t_eff_Myr'] = df['t_cosmic_Myr'] * df['gamma_t']  # TEP-corrected effective age
    
    return df


def test_agb_threshold():
    """Test whether t_eff > 300 Myr predicts higher dust content."""
    print_status("=" * 70, "INFO")
    print_status("TEST 1: AGB Threshold Prediction", "INFO")
    print_status("=" * 70, "INFO")
    print_status("", "INFO")
    
    df = load_z8_data()
    if df is None:
        return None
        
    print_status(f"z > 8 sample: N = {len(df)}", "INFO")
    print_status(f"Mean z: {df['z_phot'].mean():.2f}", "INFO")
    print_status(f"Mean t_cosmic: {df['t_cosmic_Myr'].mean():.0f} Myr", "INFO")
    print_status("", "INFO")
    
    # Split by t_eff threshold
    above_threshold = df[df['t_eff_Myr'] >= T_AGB_THRESHOLD]
    below_threshold = df[df['t_eff_Myr'] < T_AGB_THRESHOLD]
    
    print_status(f"Galaxies with t_eff >= {T_AGB_THRESHOLD} Myr: N = {len(above_threshold)}", "INFO")
    print_status(f"Galaxies with t_eff < {T_AGB_THRESHOLD} Myr: N = {len(below_threshold)}", "INFO")
    print_status("", "INFO")
    
    if len(above_threshold) < 10 or len(below_threshold) < 10:
        print_status("⚠ Insufficient sample size for robust test", "INFO")
        return None
    
    # Compare dust content
    dust_above = above_threshold['dust'].values
    dust_below = below_threshold['dust'].values
    
    mean_above = np.mean(dust_above)
    mean_below = np.mean(dust_below)
    
    # Mann-Whitney U test (non-parametric)
    stat, p_value = stats.mannwhitneyu(dust_above, dust_below, alternative='greater')
    
    # Effect size (Cohen's d): standardised mean difference between groups
    pooled_std = np.sqrt((np.var(dust_above) + np.var(dust_below)) / 2)
    cohens_d = (mean_above - mean_below) / pooled_std if pooled_std > 0 else 0
    
    print_status("RESULTS:", "INFO")
    print_status(f"  Mean dust (t_eff >= {T_AGB_THRESHOLD} Myr): {mean_above:.3f}", "INFO")
    print_status(f"  Mean dust (t_eff < {T_AGB_THRESHOLD} Myr): {mean_below:.3f}", "INFO")
    print_status(f"  Ratio: {mean_above/mean_below:.2f}×", "INFO")
    print_status(f"  Mann-Whitney p-value: {p_value:.2e}", "INFO")
    print_status(f"  Cohen's d: {cohens_d:.2f}", "INFO")
    print_status("", "INFO")
    
    if p_value < 0.05 and mean_above > mean_below:
        print_status("✓ TEP PREDICTION CONFIRMED: t_eff > 300 Myr → more dust", "INFO")
        status = "confirmed"
    else:
        print_status("⚠ TEP prediction not confirmed", "INFO")
        status = "not_confirmed"
    
    return {
        'n_above': len(above_threshold),
        'n_below': len(below_threshold),
        'mean_dust_above': float(mean_above),
        'mean_dust_below': float(mean_below),
        'ratio': float(mean_above / mean_below) if mean_below > 0 else None,
        'p_value': format_p_value(p_value),
        'cohens_d': float(cohens_d),
        'status': status
    }


def test_cosmic_time_control():
    """Test that t_eff predicts dust BEYOND t_cosmic."""
    print_status("", "INFO")
    print_status("=" * 70, "INFO")
    print_status("TEST 2: Controlling for Cosmic Time", "INFO")
    print_status("=" * 70, "INFO")
    print_status("", "INFO")
    
    df = load_z8_data()
    if df is None:
        return None

    # Partial correlation: t_eff vs dust, controlling for t_cosmic
    rho_partial, p_partial, _ = partial_rank_correlation(
        df['t_eff_Myr'].values,
        df['dust'].values,
        df['t_cosmic_Myr'].values,
    )
    
    # Raw correlation for comparison
    rho_raw, p_raw = stats.spearmanr(df['t_eff_Myr'], df['dust'])
    
    print_status("RESULTS:", "INFO")
    print_status(f"  Raw correlation (t_eff vs dust): ρ = {rho_raw:.3f}, p = {p_raw:.2e}", "INFO")
    print_status(f"  Partial correlation (t_eff vs dust | t_cosmic): ρ = {rho_partial:.3f}, p = {p_partial:.2e}", "INFO")
    print_status("", "INFO")
    
    if p_partial < 0.05 and rho_partial > 0:
        print_status("✓ t_eff predicts dust BEYOND what t_cosmic alone explains", "INFO")
        status = "confirmed"
    else:
        print_status("⚠ t_eff does not add predictive power beyond t_cosmic", "INFO")
        status = "not_confirmed"
    
    return {
        'rho_raw': float(rho_raw),
        'p_raw': format_p_value(p_raw),
        'rho_partial': float(rho_partial),
        'p_partial': format_p_value(p_partial),
        'status': status
    }


def test_mass_independence():
    """Test that the z>8 dust-mass correlation is not just a mass effect."""
    print_status("", "INFO")
    print_status("=" * 70, "INFO")
    print_status("TEST 3: Mass Independence at z>8", "INFO")
    print_status("=" * 70, "INFO")
    print_status("", "INFO")
    
    df = load_z8_data()
    if df is None:
        return None
    
    # Split into mass bins
    mass_median = df['log_Mstar'].median()
    low_mass = df[df['log_Mstar'] < mass_median]
    high_mass = df[df['log_Mstar'] >= mass_median]
    
    print_status(f"Mass median: log(M*) = {mass_median:.2f}", "INFO")
    print_status(f"Low-mass sample: N = {len(low_mass)}", "INFO")
    print_status(f"High-mass sample: N = {len(high_mass)}", "INFO")
    print_status("", "INFO")
    
    # Test t_eff vs dust correlation in each mass bin
    rho_low, p_low = stats.spearmanr(low_mass['t_eff_Myr'], low_mass['dust'])
    rho_high, p_high = stats.spearmanr(high_mass['t_eff_Myr'], high_mass['dust'])
    
    print_status("RESULTS:", "INFO")
    print_status(f"  Low-mass: ρ(t_eff, dust) = {rho_low:.3f}, p = {p_low:.2e}", "INFO")
    print_status(f"  High-mass: ρ(t_eff, dust) = {rho_high:.3f}, p = {p_high:.2e}", "INFO")
    print_status("", "INFO")
    
    # Both should be positive if TEP is real
    both_positive = rho_low > 0 and rho_high > 0
    any_significant = p_low < 0.05 or p_high < 0.05
    
    if both_positive:
        print_status("✓ t_eff-dust correlation is positive in BOTH mass bins", "INFO")
        status = "confirmed"
    else:
        print_status("⚠ t_eff-dust correlation is not consistent across mass bins", "INFO")
        status = "not_confirmed"
    
    return {
        'mass_median': float(mass_median),
        'low_mass': {'n': len(low_mass), 'rho': float(rho_low), 'p': format_p_value(p_low)},
        'high_mass': {'n': len(high_mass), 'rho': float(rho_high), 'p': format_p_value(p_high)},
        'both_positive': bool(both_positive),
        'status': status
    }


def test_redshift_gradient():
    """Test that the dust anomaly strengthens with redshift (as TEP predicts)."""
    print_status("", "INFO")
    print_status("=" * 70, "INFO")
    print_status("TEST 4: Redshift Gradient", "INFO")
    print_status("=" * 70, "INFO")
    print_status("", "INFO")
    
    df = load_z8_data()
    
    # Split by redshift
    z_bins = [(8, 8.5), (8.5, 9), (9, 10)]
    
    results = []
    for z_lo, z_hi in z_bins:
        bin_df = df[(df['z_phot'] >= z_lo) & (df['z_phot'] < z_hi)]
        if len(bin_df) >= 10:
            rho, p = stats.spearmanr(bin_df['log_Mstar'], bin_df['dust'])
            results.append({
                'z_range': f"[{z_lo}, {z_hi})",
                'n': len(bin_df),
                'rho': rho,
                'p': format_p_value(p)
            })
            print_status(f"  z = [{z_lo}, {z_hi}): N = {len(bin_df)}, ρ = {rho:.3f}, p = {p:.2e}", "INFO")
    
    print_status("", "INFO")
    
    # Check if correlation strengthens with z
    if len(results) >= 2:
        rhos = [r['rho'] for r in results]
        if rhos[-1] > rhos[0]:
            print_status("✓ Mass-dust correlation STRENGTHENS with redshift", "INFO")
            status = "confirmed"
        else:
            print_status("⚠ Mass-dust correlation does not strengthen with redshift", "INFO")
            status = "not_confirmed"
    else:
        status = "insufficient_data"
    
    return {
        'bins': results,
        'status': status
    }


def main():
    print_status("=" * 70, "INFO")
    print_status("STEP 32: z>8 Dust Anomaly - Quantitative Prediction Test", "INFO")
    print_status("=" * 70, "INFO")
    print_status("", "INFO")
    print_status("The z>8 dust anomaly is the paper's strongest evidence.", "INFO")
    print_status("This step provides rigorous quantitative tests.", "INFO")
    print_status("", "INFO")
    
    results = {}
    
    # Test 1: AGB threshold
    results['agb_threshold'] = test_agb_threshold()
    
    # Test 2: Cosmic time control
    results['cosmic_time_control'] = test_cosmic_time_control()
    
    # Test 3: Mass independence
    results['mass_independence'] = test_mass_independence()
    
    # Test 4: Redshift gradient
    results['redshift_gradient'] = test_redshift_gradient()
    
    # Summary
    print_status("", "INFO")
    print_status("=" * 70, "INFO")
    print_status("SUMMARY", "INFO")
    print_status("=" * 70, "INFO")
    
    tests_passed = 0
    total_tests = 0
    
    for test_name, test_result in results.items():
        if test_result is not None:
            total_tests += 1
            if test_result.get('status') == 'confirmed':
                tests_passed += 1
                print_status(f"  ✓ {test_name}: CONFIRMED", "INFO")
            else:
                print_status(f"  ⚠ {test_name}: {test_result.get('status', 'unknown')}", "INFO")
    
    print_status("", "INFO")
    print_status(f"Tests passed: {tests_passed}/{total_tests}", "INFO")
    
    results['summary'] = {
        'tests_passed': tests_passed,
        'total_tests': total_tests,
        'overall_status': 'strong' if tests_passed >= 3 else 'moderate' if tests_passed >= 2 else 'weak'
    }
    
    # Save results
    output_file = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=safe_json_default)
    
    print_status(f"\nResults saved to: {output_file}", "INFO")
    print_status(f"Step {STEP_NUM} complete.", "INFO")


if __name__ == "__main__":
    main()
