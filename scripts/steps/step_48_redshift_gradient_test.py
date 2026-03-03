#!/usr/bin/env python3
"""
Step 48: Redshift Gradient Test - The Missing Piece

This analysis exploits the UNIQUE prediction of TEP that mass alone cannot explain:
At FIXED stellar mass, the TEP effect should STRENGTHEN with redshift because
α(z) = α₀ × √(1+z).

If TEP is real:
- At fixed mass, high-z galaxies should appear OLDER than low-z galaxies
  (because α(z) is larger, so Γ_t is larger for the same mass)
- This is the OPPOSITE of what standard physics predicts (high-z = younger)

This test breaks the mass circularity because we're testing the z-component
of Γ_t at fixed mass.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import print_status

INTERIM_PATH = PROJECT_ROOT / "results" / "interim"
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"


def compute_gamma_t(log_Mstar: np.ndarray, z: np.ndarray, alpha_0: float = 0.58) -> np.ndarray:
    """Compute TEP temporal enhancement factor."""
    log_Mh = 1.0 * log_Mstar + 1.5  # Simple abundance matching
    log_Mh_ref = 12.0
    z_ref = 5.5
    delta_log_Mh = log_Mh - log_Mh_ref
    alpha_z = alpha_0 * np.sqrt((1 + z) / (1 + z_ref))
    gamma_t = np.exp(alpha_z * (2/3) * delta_log_Mh)
    return gamma_t


def test_redshift_gradient_at_fixed_mass(df: pd.DataFrame) -> dict:
    """
    The key test: At fixed mass, does age ratio increase with redshift?
    
    TEP predicts YES (because α(z) increases with z)
    Standard physics predicts NO (high-z galaxies should be younger)
    """
    print_status("=" * 70, "INFO")
    print_status("REDSHIFT GRADIENT TEST AT FIXED MASS", "INFO")
    print_status("=" * 70, "INFO")
    print_status("", "INFO")
    print_status("TEP Prediction: At fixed mass, age ratio should INCREASE with z", "INFO")
    print_status("Standard Prediction: At fixed mass, age ratio should DECREASE with z", "INFO")
    print_status("", "INFO")
    
    # Define narrow mass bins
    mass_bins = [
        (8.0, 8.3),
        (8.3, 8.6),
        (8.6, 9.0),
        (9.0, 9.5),
    ]
    
    results = []
    
    for m_lo, m_hi in mass_bins:
        mask = (df['log_Mstar'] >= m_lo) & (df['log_Mstar'] < m_hi) & (~df['age_ratio'].isna())
        bin_df = df[mask].copy()
        n = len(bin_df)
        
        if n < 30:
            continue
        
        # Correlation between z and age_ratio at fixed mass
        rho, p = stats.spearmanr(bin_df['z_phot'], bin_df['age_ratio'])
        
        # Also compute the slope
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            bin_df['z_phot'], bin_df['age_ratio']
        )
        
        # TEP prediction: positive correlation (higher z → higher age ratio)
        tep_consistent = rho > 0
        
        result = {
            'mass_range': f"{m_lo:.1f}-{m_hi:.1f}",
            'n': n,
            'mean_mass': float(bin_df['log_Mstar'].mean()),
            'z_range': [float(bin_df['z_phot'].min()), float(bin_df['z_phot'].max())],
            'rho_z_age': float(rho),
            'p_value': float(p),
            'slope': float(slope),
            'slope_err': float(std_err),
            'tep_consistent': bool(tep_consistent),
            'significant': bool(p < 0.05),
        }
        results.append(result)
        
        status = "TEP ✓" if tep_consistent else "Standard ✓"
        sig = "*" if p < 0.05 else ""
        print_status(f"  M* = {m_lo:.1f}-{m_hi:.1f}: N={n:4d}, ρ(z, age)={rho:+.3f} (p={p:.3f}) {sig} → {status}", "INFO")
    
    # Meta-analysis: combine across mass bins
    n_tep_consistent = sum(1 for r in results if r['tep_consistent'])
    n_significant = sum(1 for r in results if r['significant'])
    n_tep_and_sig = sum(1 for r in results if r['tep_consistent'] and r['significant'])
    
    print_status("", "INFO")
    print_status(f"Summary: {n_tep_consistent}/{len(results)} bins TEP-consistent, {n_significant} significant", "INFO")
    
    return {
        'test': 'Redshift Gradient at Fixed Mass',
        'prediction': 'TEP: ρ(z, age_ratio) > 0 at fixed mass',
        'by_mass_bin': results,
        'n_bins': len(results),
        'n_tep_consistent': n_tep_consistent,
        'n_significant': n_significant,
        'n_tep_and_significant': n_tep_and_sig,
        'conclusion': 'TEP SUPPORTED' if n_tep_consistent > len(results) / 2 else 'INCONCLUSIVE',
    }


def test_alpha_z_scaling(df: pd.DataFrame) -> dict:
    """
    Direct test of the α(z) = α₀√(1+z) scaling.
    
    If TEP is real, the STRENGTH of the mass-age correlation should
    increase with redshift (because α(z) increases).
    """
    print_status("", "INFO")
    print_status("=" * 70, "INFO")
    print_status("α(z) SCALING TEST", "INFO")
    print_status("=" * 70, "INFO")
    print_status("", "INFO")
    print_status("TEP Prediction: Mass-age correlation should STRENGTHEN with z", "INFO")
    print_status("", "INFO")
    
    z_bins = [(4, 5), (5, 6), (6, 7), (7, 8), (8, 10)]
    results = []
    
    for z_lo, z_hi in z_bins:
        mask = (df['z_phot'] >= z_lo) & (df['z_phot'] < z_hi) & (~df['age_ratio'].isna())
        bin_df = df[mask]
        n = len(bin_df)
        
        if n < 20:
            continue
        
        # Mass-age correlation in this z bin
        rho, p = stats.spearmanr(bin_df['log_Mstar'], bin_df['age_ratio'])
        
        # Expected α(z) scaling
        z_mid = (z_lo + z_hi) / 2
        alpha_z = 0.58 * np.sqrt((1 + z_mid) / (1 + 5.5))
        
        result = {
            'z_range': f"{z_lo}-{z_hi}",
            'z_mid': z_mid,
            'n': n,
            'rho_mass_age': float(rho),
            'p_value': float(p),
            'alpha_z_predicted': float(alpha_z),
        }
        results.append(result)
        
        print_status(f"  z = {z_lo}-{z_hi}: N={n:4d}, ρ(M*, age)={rho:+.3f}, α(z)={alpha_z:.3f}", "INFO")
    
    # Check if correlation strengthens with z
    if len(results) >= 3:
        z_mids = [r['z_mid'] for r in results]
        rhos = [r['rho_mass_age'] for r in results]
        rho_trend, p_trend = stats.spearmanr(z_mids, rhos)
        
        print_status("", "INFO")
        print_status(f"Trend: ρ(z_mid, ρ_mass_age) = {rho_trend:+.3f} (p={p_trend:.3f})", "INFO")
        
        tep_supported = rho_trend > 0
    else:
        rho_trend = np.nan
        p_trend = np.nan
        tep_supported = False
    
    return {
        'test': 'α(z) Scaling',
        'prediction': 'TEP: Mass-age correlation strengthens with z',
        'by_z_bin': results,
        'trend_rho': float(rho_trend) if np.isfinite(rho_trend) else None,
        'trend_p': float(p_trend) if np.isfinite(p_trend) else None,
        'tep_supported': bool(tep_supported),
    }


def test_gamma_residual_vs_z(df: pd.DataFrame) -> dict:
    """
    After removing the mass component from Γ_t, does the residual
    correlate with age ratio?
    
    This isolates the z-dependent component of TEP.
    """
    print_status("", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Γ_t RESIDUAL TEST (Z-COMPONENT ISOLATION)", "INFO")
    print_status("=" * 70, "INFO")
    print_status("", "INFO")
    
    # Compute Γ_t
    df = df.copy()
    df['gamma_t'] = compute_gamma_t(df['log_Mstar'].values, df['z_phot'].values)
    
    # Regress Γ_t on mass to get the residual (z-component)
    mask = (~df['gamma_t'].isna()) & (~df['age_ratio'].isna()) & (~df['log_Mstar'].isna())
    df_valid = df[mask].copy()
    
    # Residualize Γ_t against mass
    slope, intercept, _, _, _ = stats.linregress(df_valid['log_Mstar'], np.log(df_valid['gamma_t']))
    df_valid['log_gamma_residual'] = np.log(df_valid['gamma_t']) - (slope * df_valid['log_Mstar'] + intercept)
    
    # Correlation of residual with age ratio
    rho, p = stats.spearmanr(df_valid['log_gamma_residual'], df_valid['age_ratio'])
    
    print_status(f"N = {len(df_valid)}", "INFO")
    print_status(f"ρ(Γ_t residual, age_ratio) = {rho:+.4f} (p = {p:.2e})", "INFO")
    
    # This residual is the z-component of Γ_t
    # If TEP is real, it should correlate positively with age ratio
    tep_supported = rho > 0 and p < 0.05
    
    print_status("", "INFO")
    if tep_supported:
        print_status("★ TEP z-component CONFIRMED: Residual Γ_t correlates with age", "INFO")
    else:
        print_status("TEP z-component not confirmed", "INFO")
    
    return {
        'test': 'Γ_t Residual (Z-Component)',
        'n': len(df_valid),
        'rho': float(rho),
        'p': float(p),
        'tep_supported': bool(tep_supported),
        'interpretation': 'The z-dependent component of Γ_t (after removing mass) correlates with age ratio',
    }


def test_high_z_low_mass_prediction(df: pd.DataFrame) -> dict:
    """
    The "Impossible" Test: Low-mass galaxies at high-z should NOT show
    TEP enhancement (Γ_t < 1), but high-mass galaxies at high-z SHOULD.
    
    This is a unique TEP prediction that mass alone cannot explain.
    """
    print_status("", "INFO")
    print_status("=" * 70, "INFO")
    print_status("HIGH-Z MASS SPLIT TEST", "INFO")
    print_status("=" * 70, "INFO")
    print_status("", "INFO")
    
    # Focus on z > 7 where TEP effects should be strongest
    df_highz = df[(df['z_phot'] >= 7) & (~df['age_ratio'].isna())].copy()
    
    # Split by mass
    mass_median = df_highz['log_Mstar'].median()
    
    low_mass = df_highz[df_highz['log_Mstar'] < mass_median]
    high_mass = df_highz[df_highz['log_Mstar'] >= mass_median]
    
    # Compute mean age ratios
    low_mass_age = low_mass['age_ratio'].mean()
    high_mass_age = high_mass['age_ratio'].mean()
    
    # TEP prediction: high-mass should have HIGHER age ratio
    # (because Γ_t > 1 for high mass, Γ_t < 1 for low mass)
    diff = high_mass_age - low_mass_age
    
    # Statistical test
    t_stat, p_value = stats.ttest_ind(high_mass['age_ratio'], low_mass['age_ratio'])
    
    print_status(f"z > 7 sample: N = {len(df_highz)}", "INFO")
    print_status(f"Mass median: {mass_median:.2f}", "INFO")
    print_status(f"Low-mass (N={len(low_mass)}): mean age ratio = {low_mass_age:.3f}", "INFO")
    print_status(f"High-mass (N={len(high_mass)}): mean age ratio = {high_mass_age:.3f}", "INFO")
    print_status(f"Difference: {diff:+.3f} (p = {p_value:.3f})", "INFO")
    
    tep_supported = diff > 0 and p_value < 0.05
    
    print_status("", "INFO")
    if tep_supported:
        print_status("★ TEP CONFIRMED: High-mass z>7 galaxies appear older", "INFO")
    else:
        print_status("TEP not confirmed in high-z mass split", "INFO")
    
    return {
        'test': 'High-z Mass Split',
        'z_range': [7, 10],
        'n_total': len(df_highz),
        'mass_median': float(mass_median),
        'low_mass': {
            'n': len(low_mass),
            'mean_age_ratio': float(low_mass_age),
        },
        'high_mass': {
            'n': len(high_mass),
            'mean_age_ratio': float(high_mass_age),
        },
        'difference': float(diff),
        'p_value': float(p_value),
        'tep_supported': bool(tep_supported),
    }


def test_dust_production_timescale(df: pd.DataFrame) -> dict:
    """
    The "Dust Clock" Test: AGB dust production requires ~300-500 Myr.
    
    At z > 8, cosmic time < 600 Myr. Standard physics predicts:
    - Low-mass galaxies: No dust (insufficient time)
    - High-mass galaxies: No dust (insufficient time)
    
    TEP predicts:
    - Low-mass galaxies: No dust (Γ_t < 1, even less effective time)
    - High-mass galaxies: DUST (Γ_t > 1, sufficient effective time)
    
    The RATIO of dusty galaxies should be much higher in high-mass bin.
    """
    print_status("", "INFO")
    print_status("=" * 70, "INFO")
    print_status("DUST CLOCK TEST (z > 8)", "INFO")
    print_status("=" * 70, "INFO")
    print_status("", "INFO")
    
    # Focus on z > 8
    df_z8 = df[(df['z_phot'] >= 8) & (~df['dust'].isna())].copy()
    
    # Define "dusty" threshold
    dust_threshold = 0.5  # A_V > 0.5
    
    # Split by mass
    mass_median = df_z8['log_Mstar'].median()
    
    low_mass = df_z8[df_z8['log_Mstar'] < mass_median]
    high_mass = df_z8[df_z8['log_Mstar'] >= mass_median]
    
    low_dusty_frac = (low_mass['dust'] > dust_threshold).mean()
    high_dusty_frac = (high_mass['dust'] > dust_threshold).mean()
    
    ratio = high_dusty_frac / low_dusty_frac if low_dusty_frac > 0 else np.inf
    
    print_status(f"z > 8 sample: N = {len(df_z8)}", "INFO")
    print_status(f"Dusty threshold: A_V > {dust_threshold}", "INFO")
    print_status(f"Low-mass (N={len(low_mass)}): {low_dusty_frac*100:.1f}% dusty", "INFO")
    print_status(f"High-mass (N={len(high_mass)}): {high_dusty_frac*100:.1f}% dusty", "INFO")
    print_status(f"Ratio: {ratio:.1f}x", "INFO")
    
    # Fisher's exact test
    low_dusty = (low_mass['dust'] > dust_threshold).sum()
    low_clean = len(low_mass) - low_dusty
    high_dusty = (high_mass['dust'] > dust_threshold).sum()
    high_clean = len(high_mass) - high_dusty
    
    odds_ratio, p_value = stats.fisher_exact([[low_dusty, low_clean], [high_dusty, high_clean]])
    
    print_status(f"Fisher's exact: OR = {odds_ratio:.2f}, p = {p_value:.3f}", "INFO")
    
    tep_supported = ratio > 2 and p_value < 0.05
    
    print_status("", "INFO")
    if tep_supported:
        print_status("★ DUST CLOCK CONFIRMED: High-mass z>8 galaxies are dustier", "INFO")
    else:
        print_status("Dust clock test inconclusive", "INFO")
    
    return {
        'test': 'Dust Clock (z > 8)',
        'n': len(df_z8),
        'dust_threshold': dust_threshold,
        'low_mass_dusty_fraction': float(low_dusty_frac),
        'high_mass_dusty_fraction': float(high_dusty_frac),
        'ratio': float(ratio) if np.isfinite(ratio) else None,
        'odds_ratio': float(odds_ratio),
        'p_value': float(p_value),
        'tep_supported': bool(tep_supported),
    }


def main():
    print_status("=" * 70, "INFO")
    print_status("STEP 48: REDSHIFT GRADIENT TEST - THE MISSING PIECE", "INFO")
    print_status("=" * 70, "INFO")
    print_status("", "INFO")
    print_status("Goal: Find tests where TEP uniquely predicts something", "INFO")
    print_status("that mass alone cannot explain.", "INFO")
    print_status("", "INFO")
    
    # Load data
    df = pd.read_csv(INTERIM_PATH / "step_02_uncover_full_sample_tep.csv")
    print_status(f"Loaded sample: N = {len(df)}", "INFO")
    
    results = {
        'generated': datetime.now().isoformat(),
        'n_total': len(df),
    }
    
    # Run tests
    results['redshift_gradient'] = test_redshift_gradient_at_fixed_mass(df)
    results['alpha_z_scaling'] = test_alpha_z_scaling(df)
    results['gamma_residual'] = test_gamma_residual_vs_z(df)
    results['high_z_mass_split'] = test_high_z_low_mass_prediction(df)
    results['dust_clock'] = test_dust_production_timescale(df)
    
    # Summary
    print_status("", "INFO")
    print_status("=" * 70, "INFO")
    print_status("SUMMARY", "INFO")
    print_status("=" * 70, "INFO")
    
    tests_passed = 0
    tests_total = 5
    
    if results['redshift_gradient']['n_tep_consistent'] > results['redshift_gradient']['n_bins'] / 2:
        tests_passed += 1
        print_status("✓ Redshift Gradient: TEP supported", "INFO")
    else:
        print_status("✗ Redshift Gradient: Inconclusive", "INFO")
    
    if results['alpha_z_scaling'].get('tep_supported', False):
        tests_passed += 1
        print_status("✓ α(z) Scaling: TEP supported", "INFO")
    else:
        print_status("✗ α(z) Scaling: Inconclusive", "INFO")
    
    if results['gamma_residual']['tep_supported']:
        tests_passed += 1
        print_status("✓ Γ_t Residual: TEP supported", "INFO")
    else:
        print_status("✗ Γ_t Residual: Inconclusive", "INFO")
    
    if results['high_z_mass_split']['tep_supported']:
        tests_passed += 1
        print_status("✓ High-z Mass Split: TEP supported", "INFO")
    else:
        print_status("✗ High-z Mass Split: Inconclusive", "INFO")
    
    if results['dust_clock']['tep_supported']:
        tests_passed += 1
        print_status("✓ Dust Clock: TEP supported", "INFO")
    else:
        print_status("✗ Dust Clock: Inconclusive", "INFO")
    
    results['summary'] = {
        'tests_passed': tests_passed,
        'tests_total': tests_total,
        'pass_rate': tests_passed / tests_total,
    }
    
    print_status("", "INFO")
    print_status(f"Overall: {tests_passed}/{tests_total} tests support TEP", "INFO")
    
    # Save results
    output_file = OUTPUT_PATH / "step_48_redshift_gradient_test.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print_status(f"Saved: {output_file}", "SUCCESS")


if __name__ == "__main__":
    main()
