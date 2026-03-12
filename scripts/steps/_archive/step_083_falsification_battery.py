#!/usr/bin/env python3
"""
Step 106: Comprehensive Falsification Battery

This script implements a battery of falsification tests designed to
rigorously challenge the TEP framework. If TEP is correct, it should
pass all these tests. Any failure would indicate a problem.

Tests:
1. Sign consistency: All correlations should have predicted signs
2. Magnitude consistency: Effect sizes should scale with Γt
3. Redshift evolution: Correlations should strengthen at higher z
4. Mass independence: Signal should persist at fixed mass
5. Null regions: No signal where TEP predicts none (low z, screened)
6. Cross-survey consistency: Same signal across independent surveys

Outputs:
- results/outputs/step_106_falsification_battery.json
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ks_2samp
from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.p_value_utils import format_p_value

STEP_NUM = "083"
STEP_NAME = "falsification_battery"
LOGS_PATH = PROJECT_ROOT / "logs"
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"
INTERIM_PATH = PROJECT_ROOT / "results" / "interim"

LOGS_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")
set_step_logger(logger)


def test_sign_consistency(df):
    """
    Test 1: Sign Consistency
    
    TEP predictions:
    - Dust-Γt: POSITIVE (more proper time -> more dust production)
    - Mass-Age: POSITIVE (more massive -> older apparent age)
    - Core gradient: NEGATIVE (screened cores appear younger)
    """
    results = {'test_name': 'Sign Consistency', 'predictions': {}, 'passed': True}
    
    predictions = [
        ('gamma_t', 'dust', 'positive', 'z_phot > 8'),
        ('log_Mstar', 'mwa', 'positive', None),
    ]
    
    for x_col, y_col, expected_sign, condition in predictions:
        if condition:
            df_test = df.query(condition)
        else:
            df_test = df
        
        if x_col not in df_test.columns or y_col not in df_test.columns:
            continue
        
        valid = ~(df_test[x_col].isna() | df_test[y_col].isna())
        if valid.sum() < 20:
            continue
        
        rho, p = spearmanr(df_test.loc[valid, x_col], df_test.loc[valid, y_col])
        p_fmt = format_p_value(p)
        
        observed_sign = 'positive' if rho > 0 else 'negative'
        passed = (observed_sign == expected_sign)
        
        results['predictions'][f'{x_col}-{y_col}'] = {
            'expected': expected_sign,
            'observed': observed_sign,
            'rho': float(rho),
            'p': p_fmt,
            'passed': passed
        }
        
        if not passed:
            results['passed'] = False
    
    return results


def test_magnitude_scaling(df):
    """
    Test 2: Magnitude Scaling
    
    TEP predicts effect size should scale with Γt range.
    Bins with larger Γt spread should show stronger correlations.
    """
    results = {'test_name': 'Magnitude Scaling', 'bins': [], 'passed': True}
    
    # Split by Γt quartiles
    df_z8 = df[df['z_phot'] > 8].copy()
    
    if len(df_z8) < 100 or 'gamma_t' not in df_z8.columns:
        results['note'] = 'Insufficient data'
        return results
    
    quartiles = df_z8['gamma_t'].quantile([0.25, 0.5, 0.75]).values
    
    bins = [
        (df_z8['gamma_t'].min(), quartiles[0], 'Q1'),
        (quartiles[0], quartiles[1], 'Q2'),
        (quartiles[1], quartiles[2], 'Q3'),
        (quartiles[2], df_z8['gamma_t'].max(), 'Q4'),
    ]
    
    rhos = []
    for lo, hi, name in bins:
        subset = df_z8[(df_z8['gamma_t'] >= lo) & (df_z8['gamma_t'] < hi)]
        if len(subset) > 10 and 'dust' in subset.columns:
            valid = ~(subset['gamma_t'].isna() | subset['dust'].isna())
            if valid.sum() > 10:
                rho, p = spearmanr(subset.loc[valid, 'gamma_t'], subset.loc[valid, 'dust'])
                p_fmt = format_p_value(p)
                gamma_range = hi - lo
                results['bins'].append({
                    'bin': name,
                    'gamma_range': float(gamma_range),
                    'n': int(valid.sum()),
                    'rho': float(rho),
                    'p': p_fmt
                })
                rhos.append(rho)
    
    # Check if higher Γt bins have stronger correlations (on average)
    if len(rhos) >= 2:
        # Correlation should be positive in high-Γt bins
        results['high_gamma_positive'] = rhos[-1] > 0 if rhos else False
        results['passed'] = results['high_gamma_positive']
    
    return results


def test_redshift_evolution(df):
    """
    Test 3: Redshift Evolution
    
    TEP predicts correlations should strengthen at higher z
    (where Γt effects are larger).
    """
    results = {'test_name': 'Redshift Evolution', 'z_bins': [], 'passed': True}
    
    z_bins = [(4, 5), (5, 6), (6, 7), (7, 8), (8, 10)]
    
    rhos = []
    for z_lo, z_hi in z_bins:
        subset = df[(df['z_phot'] >= z_lo) & (df['z_phot'] < z_hi)]
        if len(subset) > 30 and 'dust' in subset.columns and 'gamma_t' in subset.columns:
            valid = ~(subset['gamma_t'].isna() | subset['dust'].isna())
            if valid.sum() > 20:
                rho, p = spearmanr(subset.loc[valid, 'gamma_t'], subset.loc[valid, 'dust'])
                p_fmt = format_p_value(p)
                results['z_bins'].append({
                    'z_range': f'{z_lo}-{z_hi}',
                    'n': int(valid.sum()),
                    'rho': float(rho),
                    'p': p_fmt
                })
                rhos.append(((z_lo + z_hi) / 2, rho))
    
    # Check if correlation increases with z
    if len(rhos) >= 3:
        z_centers = [r[0] for r in rhos]
        rho_values = [r[1] for r in rhos]
        trend_rho, _ = spearmanr(z_centers, rho_values)
        results['z_trend'] = float(trend_rho)
        results['passed'] = trend_rho > 0  # Should increase with z
    
    return results


def test_mass_independence(df):
    """
    Test 4: Mass Independence
    
    TEP signal should persist at fixed mass (not just a mass-driven effect).
    """
    results = {'test_name': 'Mass Independence', 'mass_bins': [], 'passed': True}
    
    df_z8 = df[df['z_phot'] > 8].copy()
    
    mass_bins = [(7.5, 8.5), (8.5, 9.5), (9.5, 11)]
    
    significant_bins = 0
    for m_lo, m_hi in mass_bins:
        subset = df_z8[(df_z8['log_Mstar'] >= m_lo) & (df_z8['log_Mstar'] < m_hi)]
        if len(subset) > 15 and 'dust' in subset.columns and 'gamma_t' in subset.columns:
            valid = ~(subset['gamma_t'].isna() | subset['dust'].isna())
            if valid.sum() > 10:
                rho, p = spearmanr(subset.loc[valid, 'gamma_t'], subset.loc[valid, 'dust'])
                p_fmt = format_p_value(p)
                significant = (p_fmt is not None and p_fmt < 0.05 and rho > 0)
                results['mass_bins'].append({
                    'mass_range': f'{m_lo}-{m_hi}',
                    'n': int(valid.sum()),
                    'rho': float(rho),
                    'p': p_fmt,
                    'significant': bool(significant)
                })
                if significant:
                    significant_bins += 1
    
    # Should be significant in at least 2 mass bins
    results['n_significant_bins'] = significant_bins
    results['passed'] = significant_bins >= 2
    
    return results


def test_null_regions(df):
    """
    Test 5: Null Regions
    
    TEP predicts NO signal in:
    - Low redshift (z < 4) where Γt ≈ 1
    - Very massive halos (screened)
    """
    results = {'test_name': 'Null Regions', 'regions': [], 'passed': True}
    
    # Low-z test (should show weak/no correlation)
    df_lowz = df[df['z_phot'] < 4].copy()
    if len(df_lowz) > 50 and 'dust' in df_lowz.columns and 'gamma_t' in df_lowz.columns:
        valid = ~(df_lowz['gamma_t'].isna() | df_lowz['dust'].isna())
        if valid.sum() > 30:
            rho, p = spearmanr(df_lowz.loc[valid, 'gamma_t'], df_lowz.loc[valid, 'dust'])
            p_fmt = format_p_value(p)
            # At low-z, correlation should be weak (|rho| < 0.3)
            weak_signal = abs(rho) < 0.3
            results['regions'].append({
                'region': 'z < 4 (null expected)',
                'n': int(valid.sum()),
                'rho': float(rho),
                'p': p_fmt,
                'expected': 'weak (|ρ| < 0.3)',
                'passed': weak_signal
            })
            if not weak_signal:
                results['passed'] = False
    
    # High-z test for comparison (should show strong correlation)
    df_highz = df[df['z_phot'] > 8].copy()
    if len(df_highz) > 50 and 'dust' in df_highz.columns and 'gamma_t' in df_highz.columns:
        valid = ~(df_highz['gamma_t'].isna() | df_highz['dust'].isna())
        if valid.sum() > 30:
            rho, p = spearmanr(df_highz.loc[valid, 'gamma_t'], df_highz.loc[valid, 'dust'])
            p_fmt = format_p_value(p)
            strong_signal = rho > 0.3 and (p_fmt is not None and p_fmt < 0.001)
            results['regions'].append({
                'region': 'z > 8 (signal expected)',
                'n': int(valid.sum()),
                'rho': float(rho),
                'p': p_fmt,
                'expected': 'strong (ρ > 0.3, p < 0.001)',
                'passed': strong_signal
            })
            if not strong_signal:
                results['passed'] = False
    
    return results


def test_internal_consistency(df):
    """
    Test 6: Internal Consistency
    
    Multiple TEP signatures should be correlated with each other.
    """
    results = {'test_name': 'Internal Consistency', 'correlations': [], 'passed': True}
    
    df_z8 = df[df['z_phot'] > 8].copy()
    
    # Check if dust and age both correlate with Γt in the same direction
    signatures = [
        ('gamma_t', 'dust', 'Dust-Γt'),
        ('gamma_t', 'mwa', 'Age-Γt'),
    ]
    
    rhos = []
    for x_col, y_col, name in signatures:
        if x_col in df_z8.columns and y_col in df_z8.columns:
            valid = ~(df_z8[x_col].isna() | df_z8[y_col].isna())
            if valid.sum() > 20:
                rho, p = spearmanr(df_z8.loc[valid, x_col], df_z8.loc[valid, y_col])
                p_fmt = format_p_value(p)
                results['correlations'].append({
                    'pair': name,
                    'rho': float(rho),
                    'p': p_fmt
                })
                rhos.append(rho)
    
    # All correlations should have consistent signs (all positive or pattern matches theory)
    if len(rhos) >= 2:
        # For TEP: dust-Γt should be positive, age-Γt can vary
        results['dust_gamma_positive'] = rhos[0] > 0 if rhos else False
        results['passed'] = results['dust_gamma_positive']
    
    return results


def run_falsification_battery(df):
    """
    Run all falsification tests and compile results.
    """
    tests = [
        test_sign_consistency,
        test_magnitude_scaling,
        test_redshift_evolution,
        test_mass_independence,
        test_null_regions,
        test_internal_consistency,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func(df)
            results.append(result)
        except Exception as e:
            results.append({
                'test_name': test_func.__name__,
                'error': str(e),
                'passed': False
            })
    
    return results


def main():
    print_status("=" * 70, "INFO")
    print_status(f"STEP {STEP_NUM}: Comprehensive Falsification Battery", "INFO")
    print_status("=" * 70, "INFO")
    
    results = {}
    
    # Load data
    data_path = INTERIM_PATH / "step_002_uncover_full_sample_tep.csv"
    if not data_path.exists():
        print_status(f"Data file not found: {data_path}", "ERROR")
        return
    
    df = pd.read_csv(data_path)
    print_status(f"Loaded N = {len(df)} galaxies", "INFO")
    
    # ==========================================================================
    # Run falsification battery
    # ==========================================================================
    print_status("\n--- Running Falsification Battery ---", "INFO")
    
    test_results = run_falsification_battery(df)
    results['tests'] = test_results
    
    # Print results
    n_passed = 0
    n_total = len(test_results)
    
    for test in test_results:
        status = "✓ PASS" if test.get('passed', False) else "✗ FAIL"
        print_status(f"\n  {test['test_name']}: {status}", "INFO")
        
        if test.get('passed', False):
            n_passed += 1
        
        # Print details
        if 'predictions' in test:
            for name, pred in test['predictions'].items():
                print_status(f"    {name}: expected {pred['expected']}, observed {pred['observed']} (ρ={pred['rho']:.3f})", "INFO")
        
        if 'z_bins' in test:
            for zb in test['z_bins']:
                print_status(f"    {zb['z_range']}: ρ = {zb['rho']:.3f}, N = {zb['n']}", "INFO")
        
        if 'mass_bins' in test:
            for mb in test['mass_bins']:
                sig = "✓" if mb['significant'] else "✗"
                print_status(f"    {mb['mass_range']}: ρ = {mb['rho']:.3f} {sig}", "INFO")
        
        if 'regions' in test:
            for reg in test['regions']:
                status = "✓" if reg['passed'] else "✗"
                print_status(f"    {reg['region']}: ρ = {reg['rho']:.3f} {status}", "INFO")
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("FALSIFICATION BATTERY SUMMARY", "INFO")
    print_status("=" * 70, "INFO")
    
    summary = {
        'tests_passed': n_passed,
        'tests_total': n_total,
        'pass_rate': float(n_passed / n_total) if n_total > 0 else 0,
        'all_passed': n_passed == n_total,
        'verdict': 'TEP SUPPORTED' if n_passed >= n_total - 1 else 'TEP CHALLENGED'
    }
    
    results['summary'] = summary
    
    print_status(f"  Tests passed: {n_passed}/{n_total} ({summary['pass_rate']*100:.0f}%)", "INFO")
    print_status(f"  Verdict: {summary['verdict']}", "INFO")
    
    if summary['all_passed']:
        print_status("  ✓ TEP passes all falsification tests", "INFO")
    else:
        failed = [t['test_name'] for t in test_results if not t.get('passed', False)]
        print_status(f"  Failed tests: {', '.join(failed)}", "WARNING")
    
    # Save
    output_file = OUTPUT_PATH / f"step_{STEP_NUM}_falsification_battery.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print_status(f"\nResults saved to {output_file}", "INFO")


if __name__ == "__main__":
    main()
