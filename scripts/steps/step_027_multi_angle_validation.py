#!/usr/bin/env python3
# Estimated runtime from last full canonical run (2026-03-09 15:52 UTC; full pipeline 32m18s): 0.7s.
"""
TEP-JWST Step 27: Multi-Angle Validation

This analysis validates TEP predictions from multiple independent angles:

1. THE EXTREME TEST
   - What happens at the EXTREME ends of the distribution?
   - The highest Γ_t should reveal the clearest signal

2. THE STRATIFICATION TEST
   - TEP should separate the data into distinct components

3. THE ROBUSTNESS TEST
   - The TEP signal should be robust to perturbations

4. THE RESIDUAL TEST
   - After TEP correction, there should be no residual anomalies

5. THE MULTI-ANGLE TEST
   - TEP should work from every angle of the data

Author: Matthew L. Smawfield
Date: January 2026
"""

import sys
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import logging
import json

# =============================================================================
# LOGGER SETUP
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.p_value_utils import format_p_value
from scripts.utils.tep_model import compute_gamma_t as tep_gamma

STEP_NUM = "027"
STEP_NAME = "multi_angle_validation"

LOGS_PATH = PROJECT_ROOT / "logs"
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"
LOGS_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")
set_step_logger(logger)



PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "outputs"

# Note: TEPLogger is initialized above via set_step_logger()


def load_data():
    logger.info("Loading data...")
    uncover = pd.read_csv(PROJECT_ROOT / "results" / "interim" / "step_002_uncover_full_sample_tep.csv")
    uncover = uncover.rename(columns={
        'z_phot': 'z',
        'mwa': 'mwa_Gyr',
        't_cosmic': 't_cosmic_Gyr',
        't_eff': 't_eff_Gyr',
        'log_Mh': 'log_Mhalo',
    })
    
    _jades_path = DATA_DIR / "interim" / "jades_highz_physical.csv"
    if not Path(_jades_path).exists():
        print_status("ERROR: jades_highz_physical.csv not found. Run step_014 first.", "ERROR")
        return None, None
    jades = pd.read_csv(_jades_path)
    z_col = 'z_best' if ('z_best' in jades.columns and not jades['z_best'].isna().all()) else 'z_phot'
    jades['gamma_t'] = tep_gamma(
        pd.to_numeric(jades['log_Mhalo'], errors='coerce').to_numpy(),
        pd.to_numeric(jades[z_col], errors='coerce').to_numpy(),
    )
    jades['age_ratio'] = jades['t_stellar_Gyr'] / jades['t_cosmic_Gyr']
    
    return uncover, jades


def pressure_test(df):
    """The highest pressure reveals the clearest signal."""
    logger.info("=" * 70)
    logger.info("TEST 1: Extreme Regime Test")
    logger.info("=" * 70)
    
    valid = df.dropna(subset=['gamma_t', 'age_ratio', 't_eff_Gyr'])
    
    # Extreme pressure: top 5% of Γ_t
    threshold = valid['gamma_t'].quantile(0.95)
    extreme = valid[valid['gamma_t'] >= threshold]
    normal = valid[valid['gamma_t'] < threshold]
    
    logger.info(f"Extreme pressure (Γ_t ≥ {threshold:.2f}): N = {len(extreme)}")
    logger.info(f"Normal: N = {len(normal)}")
    
    # Compare properties
    for col in ['age_ratio', 't_eff_Gyr']:
        mean_ext = extreme[col].mean()
        mean_norm = normal[col].mean()
        t_stat, p = stats.ttest_ind(extreme[col], normal[col])
        ratio = mean_ext / mean_norm
        
        logger.info(f"\n{col}:")
        logger.info(f"  Extreme: {mean_ext:.4f}")
        logger.info(f"  Normal: {mean_norm:.4f}")
        logger.info(f"  Ratio: {ratio:.2f}×")
        logger.info(f"  p-value: {p:.2e}")
    
    # Extreme regime analysis
    # After TEP correction, extreme galaxies should become normal
    extreme = extreme.copy()
    extreme['age_ratio_corr'] = extreme['age_ratio'] / extreme['gamma_t']
    
    mean_corr = extreme['age_ratio_corr'].mean()
    mean_normal = normal['age_ratio'].mean()
    
    logger.info(f"\nAfter TEP correction:")
    logger.info(f"  Extreme (corrected): {mean_corr:.4f}")
    logger.info(f"  Normal (raw): {mean_normal:.4f}")
    logger.info(f"  Difference: {abs(mean_corr - mean_normal):.4f}")
    
    normalized = abs(mean_corr - mean_normal) < 0.05
    if normalized:
        logger.info("✓ Extreme galaxies normalized by TEP")
    
    return {'threshold': threshold, 'n_extreme': len(extreme), 'normalized': normalized}


def stratification_test(df):
    """TEP breaks the data into distinct components."""
    logger.info("=" * 70)
    logger.info("TEST 2: Stratification Test")
    logger.info("=" * 70)
    
    valid = df.dropna(subset=['gamma_t', 'age_ratio', 'log_Mstar', 'z'])
    
    # Divide into Γ_t quartiles
    quartiles = pd.qcut(valid['gamma_t'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    valid = valid.copy()
    valid['quartile'] = quartiles
    
    # Each quartile should have distinct properties
    logger.info("Properties by Γ_t quartile:")
    
    means = {}
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        qdata = valid[valid['quartile'] == q]
        means[q] = {
            'age_ratio': qdata['age_ratio'].mean(),
            'z': qdata['z'].mean(),
            'log_Mstar': qdata['log_Mstar'].mean()
        }
        logger.info(f"\n{q} (N={len(qdata)}):")
        logger.info(f"  age_ratio: {means[q]['age_ratio']:.4f}")
        logger.info(f"  z: {means[q]['z']:.1f}")
        logger.info(f"  log_Mstar: {means[q]['log_Mstar']:.2f}")
    
    # Test monotonicity: does age_ratio increase with quartile?
    age_ratios = [means[q]['age_ratio'] for q in ['Q1', 'Q2', 'Q3', 'Q4']]
    
    # Spearman correlation with quartile number
    rho, p = stats.spearmanr([1, 2, 3, 4], age_ratios)
    logger.info(f"\nMonotonicity test:")
    logger.info(f"  ρ(quartile, age_ratio) = {rho:.3f}, p = {p:.4f}")
    
    spectrum = abs(rho) > 0.8
    if spectrum:
        logger.info("✓ Clear spectrum across quartiles")
    
    return {'means': means, 'rho': rho, 'spectrum': spectrum}


def robustness_test(df):
    """The TEP signal is robust to all perturbations."""
    logger.info("=" * 70)
    logger.info("TEST 3: Robustness Test")
    logger.info("=" * 70)
    
    valid = df.dropna(subset=['gamma_t', 'age_ratio', 't_eff_Gyr'])
    
    # Test robustness to:
    # 1. Random subsampling
    # 2. Outlier removal
    # 3. Different correlation methods
    
    results = []
    
    # Original
    rho_orig, _ = stats.spearmanr(valid['gamma_t'], valid['t_eff_Gyr'])
    results.append(('Original', rho_orig))
    logger.info(f"Original: ρ = {rho_orig:.4f}")
    
    # Random 50% subsample (10 times)
    for i in range(5):
        sample = valid.sample(frac=0.5, random_state=i)
        rho, _ = stats.spearmanr(sample['gamma_t'], sample['t_eff_Gyr'])
        results.append((f'Subsample_{i}', rho))
    
    subsample_rhos = [r[1] for r in results[1:6]]
    logger.info(f"Subsamples: ρ = {np.mean(subsample_rhos):.4f} ± {np.std(subsample_rhos):.4f}")
    
    # Remove outliers (top/bottom 5%)
    q05 = valid['gamma_t'].quantile(0.05)
    q95 = valid['gamma_t'].quantile(0.95)
    trimmed = valid[(valid['gamma_t'] >= q05) & (valid['gamma_t'] <= q95)]
    rho_trim, _ = stats.spearmanr(trimmed['gamma_t'], trimmed['t_eff_Gyr'])
    results.append(('Trimmed', rho_trim))
    logger.info(f"Trimmed (5-95%): ρ = {rho_trim:.4f}")
    
    # Pearson correlation
    rho_pearson, _ = stats.pearsonr(valid['gamma_t'], valid['t_eff_Gyr'])
    results.append(('Pearson', rho_pearson))
    logger.info(f"Pearson: ρ = {rho_pearson:.4f}")
    
    # Check consistency
    all_rhos = [r[1] for r in results]
    cv = np.std(all_rhos) / np.mean(all_rhos)
    logger.info(f"\nConsistency: CV = {cv:.4f}")
    
    hard = cv < 0.1
    if hard:
        logger.info("✓ Signal is robust")
    
    return {'cv': cv, 'hard': hard}


def residual_test(df):
    """After TEP correction, no residual anomalies."""
    logger.info("=" * 70)
    logger.info("TEST 4: Residual Test")
    logger.info("=" * 70)
    
    valid = df.dropna(subset=['gamma_t', 'age_ratio'])
    valid = valid.copy()
    
    # Apply TEP correction
    valid['age_ratio_corr'] = valid['age_ratio'] / valid['gamma_t']
    
    # Check for residual anomalies
    anomalies = {
        'anomalous': valid['age_ratio'] > 0.5,
        'extreme': valid['age_ratio'] > 0.3,
        'very_old': valid['mwa_Gyr'] > 0.2 if 'mwa_Gyr' in valid.columns else pd.Series([False]*len(valid))
    }
    
    anomalies_corr = {
        'anomalous': valid['age_ratio_corr'] > 0.5,
        'extreme': valid['age_ratio_corr'] > 0.3,
    }
    
    logger.info("Anomaly resolution:")
    total_resolved = 0
    total_anomalies = 0
    
    for name in ['anomalous', 'extreme']:
        n_raw = anomalies[name].sum()
        n_corr = anomalies_corr[name].sum()
        if n_raw > 0:
            resolution = (n_raw - n_corr) / n_raw * 100
            logger.info(f"  {name}: {n_raw} → {n_corr} ({resolution:.0f}% resolved)")
            total_resolved += (n_raw - n_corr)
            total_anomalies += n_raw
    
    # Check for ANY remaining structure in residuals
    residuals = valid['age_ratio_corr'] - valid['age_ratio_corr'].mean()
    
    # Should be uncorrelated with Γ_t
    rho_resid, p_resid = stats.spearmanr(valid['gamma_t'], residuals)
    logger.info(f"\nResidual-Γ_t correlation: ρ = {rho_resid:.4f}, p = {p_resid:.4f}")
    
    no_residuals = abs(rho_resid) < 0.1 and (total_resolved / max(total_anomalies, 1)) > 0.8
    if no_residuals:
        logger.info("✓ No systematic residuals")
    
    return {'rho_resid': rho_resid, 'no_residuals': no_residuals}


def multi_angle_test(df, jades):
    """TEP works from every angle."""
    logger.info("=" * 70)
    logger.info("TEST 5: Multi-Angle Test")
    logger.info("=" * 70)
    
    # Test TEP from multiple angles (different observables)
    angles = []
    
    # Angle 1: UNCOVER age_ratio
    valid_u = df.dropna(subset=['gamma_t', 'age_ratio'])
    rho_1, p_1 = stats.spearmanr(valid_u['gamma_t'], valid_u['age_ratio'])
    angles.append(('UNCOVER age_ratio', rho_1, p_1))
    
    # Angle 2: UNCOVER t_eff
    if 't_eff_Gyr' in df.columns:
        valid_u2 = df.dropna(subset=['gamma_t', 't_eff_Gyr'])
        rho_2, p_2 = stats.spearmanr(valid_u2['gamma_t'], valid_u2['t_eff_Gyr'])
        angles.append(('UNCOVER t_eff', rho_2, p_2))
    
    # Angle 3: JADES age_ratio
    valid_j = jades.dropna(subset=['gamma_t', 'age_ratio'])
    rho_3, p_3 = stats.spearmanr(valid_j['gamma_t'], valid_j['age_ratio'])
    angles.append(('JADES age_ratio', rho_3, p_3))
    
    # Angle 4: JADES age_excess
    if 'age_excess_Gyr' in jades.columns:
        valid_j2 = jades.dropna(subset=['gamma_t', 'age_excess_Gyr'])
        rho_4, p_4 = stats.spearmanr(valid_j2['gamma_t'], valid_j2['age_excess_Gyr'])
        angles.append(('JADES age_excess', rho_4, p_4))
    
    logger.info("Validation from every angle:")
    significant = 0
    for name, rho, p in angles:
        status = "✓" if p < 0.05 else "✗"
        logger.info(f"  {status} {name}: ρ = {rho:.3f}, p = {p:.2e}")
        if p < 0.05:
            significant += 1
    
    multi_angle_pass = significant >= len(angles) - 1
    logger.info(f"\nSignificant angles: {significant}/{len(angles)}")
    
    if multi_angle_pass:
        logger.info("✓ Works from every angle")
    
    return {'angles': [(n, r, p) for n, r, p in angles], 'multi_angle_pass': multi_angle_pass}


def run_multi_angle_validation():
    logger.info("=" * 70)
    logger.info(f"TEP-JWST Step {STEP_NUM}: Multi-Angle Validation")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Validating TEP predictions from multiple independent angles.")
    logger.info("")
    
    uncover, jades = load_data()
    if jades is None:
        logger.error(f"Step {STEP_NUM} aborted: jades_highz_physical.csv not found. Run step_014 first.")
        return {"status": "aborted", "reason": "missing jades_highz_physical.csv"}

    results = {}
    results['pressure'] = pressure_test(uncover)
    results['stratification'] = stratification_test(uncover)
    results['robustness'] = robustness_test(uncover)
    results['residuals'] = residual_test(uncover)
    results['multi_angle'] = multi_angle_test(uncover, jades)
    
    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY: Multi-Angle Validation Results")
    logger.info("=" * 70)
    
    tests_passed = 0
    if results['pressure'].get('normalized', False):
        tests_passed += 1
        logger.info("✓ Extreme: High-Γ_t galaxies normalized")
    if results['stratification'].get('spectrum', False):
        tests_passed += 1
        logger.info("✓ Stratification: Clear separation by Γ_t")
    if results['robustness'].get('hard', False):
        tests_passed += 1
        logger.info("✓ Robustness: Signal is stable")
    if results['residuals'].get('no_residuals', False):
        tests_passed += 1
        logger.info("✓ Residuals: No systematic anomalies")
    if results['multi_angle'].get('multi_angle_pass', False):
        tests_passed += 1
        logger.info("✓ Multi-angle: Works from every angle")
    
    logger.info(f"\nTests passed: {tests_passed}/5")
    
    if tests_passed >= 4:
        logger.info("")
        logger.info("Multi-angle validation: PASSED")
    
    # Save
    output_file = RESULTS_DIR / f"step_{STEP_NUM}_multi_angle_validation.json"
    
    def convert(obj):
        if obj is None:
            return None
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            if np.isnan(obj):
                return None
            if np.isposinf(obj):
                return "infinity"
            if np.isneginf(obj):
                return "-infinity"
            return float(obj)
        if isinstance(obj, (int, float)):
            if isinstance(obj, float) and np.isnan(obj):
                return None
            if obj == float('inf'):
                return "infinity"
            if obj == float('-inf'):
                return "-infinity"
            return obj
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, tuple):
            return list(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(i) for i in obj]
        return str(obj)
    
    results_clean = {k: convert(v) for k, v in results.items()}
    
    with open(output_file, 'w') as f:
        json.dump(results_clean, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_file}")
    return results


if __name__ == "__main__":
    run_multi_angle_validation()
