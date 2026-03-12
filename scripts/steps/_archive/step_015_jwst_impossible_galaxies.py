#!/usr/bin/env python3
"""
TEP-JWST Step 15: "Anomalous" Galaxies Analysis

A key TEP signature in JWST data is the existence of "anomalous"
galaxies—systems whose inferred stellar ages exceed the cosmic age at their
redshift. Under standard physics, this is paradoxical. Under TEP, it's expected.

This analysis:
1. Identifies galaxies with age ratio > 0.5 (stellar age / cosmic age)
2. Tests if these "anomalous" galaxies have systematically higher Γ_t
3. Quantifies the TEP explanation for the impossibility

The key prediction: galaxies that appear "anomalous" should have the highest
predicted Γ_t values. This is a direct, quantitative test of TEP.

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

STEP_NUM = "015"
STEP_NAME = "jwst_impossible_galaxies"

LOGS_PATH = PROJECT_ROOT / "logs"
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"
LOGS_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")
set_step_logger(logger)



# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "outputs"

# Note: TEPLogger is initialized above via set_step_logger()


def load_uncover_data():
    """Load UNCOVER data with TEP calculations from main pipeline."""
    logger.info("Loading UNCOVER data with TEP parameters...")
    
    # Use the main pipeline output which has correct Γ_t from abundance matching
    df = pd.read_csv(PROJECT_ROOT / "results" / "interim" / "step_002_uncover_full_sample_tep.csv")
    
    # Rename columns for consistency with this script
    df = df.rename(columns={'z_phot': 'z', 'mwa': 'mwa_Gyr'})
    
    # age_ratio is already calculated in step_02
    
    logger.info(f"Loaded {len(df)} galaxies")
    logger.info(f"Redshift range: {df['z'].min():.1f} - {df['z'].max():.1f}")
    logger.info(f"Age ratio range: {df['age_ratio'].min():.3f} - {df['age_ratio'].max():.3f}")
    
    return df


def identify_impossible_galaxies(df):
    """
    Identify "anomalous" galaxies with age ratio > threshold.
    
    Standard physics: age_ratio should be < 1 (stellar age < cosmic age)
    In practice, age_ratio > 0.5 is already problematic (galaxies formed
    in the first half of cosmic time at that redshift).
    """
    logger.info("=" * 70)
    logger.info("ANALYSIS 1: Identifying 'Anomalous' Galaxies")
    logger.info("=" * 70)
    
    thresholds = [0.3, 0.5, 0.7, 0.9]
    
    results = {}
    for thresh in thresholds:
        n_impossible = (df['age_ratio'] > thresh).sum()
        frac = n_impossible / len(df) * 100
        logger.info(f"Age ratio > {thresh}: N = {n_impossible} ({frac:.1f}%)")
        results[f'n_gt_{thresh}'] = n_impossible
        results[f'frac_gt_{thresh}'] = frac
    
    # Focus on age_ratio > 0.5 as "anomalous"
    anomalous = df[df['age_ratio'] > 0.5]
    normal = df[df['age_ratio'] <= 0.5]
    
    logger.info(f"\n'Anomalous' galaxies (age_ratio > 0.5): N = {len(anomalous)}")
    logger.info(f"'Normal' galaxies (age_ratio ≤ 0.5): N = {len(normal)}")
    
    results['n_impossible'] = len(anomalous)
    results['n_normal'] = len(normal)
    
    return results, anomalous, normal


def compare_gamma_t(anomalous, normal):
    """
    Compare Γ_t between anomalous and normal galaxies.
    
    TEP Prediction: Anomalous galaxies should have HIGHER Γ_t.
    """
    logger.info("=" * 70)
    logger.info("ANALYSIS 2: Γ_t Comparison (Anomalous vs Normal)")
    logger.info("=" * 70)
    
    gamma_impossible = anomalous['gamma_t']
    gamma_normal = normal['gamma_t']
    
    mean_impossible = gamma_impossible.mean()
    sem_impossible = gamma_impossible.std() / np.sqrt(len(gamma_impossible))
    
    mean_normal = gamma_normal.mean()
    sem_normal = gamma_normal.std() / np.sqrt(len(gamma_normal))
    
    logger.info(f"Mean Γ_t (anomalous): {mean_impossible:.3f} ± {sem_impossible:.3f}")
    logger.info(f"Mean Γ_t (normal): {mean_normal:.3f} ± {sem_normal:.3f}")
    
    # Statistical test
    t_stat, p_value = stats.ttest_ind(gamma_impossible, gamma_normal)
    p_value_fmt = format_p_value(p_value)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((gamma_impossible.std()**2 + gamma_normal.std()**2) / 2)
    cohens_d = (mean_impossible - mean_normal) / pooled_std
    
    logger.info(f"\nt-statistic: {t_stat:.2f}")
    logger.info(f"p-value: {p_value:.2e}")
    logger.info(f"Cohen's d: {cohens_d:.3f}")
    
    ratio = mean_impossible / mean_normal
    logger.info(f"\nRatio (anomalous / normal): {ratio:.2f}")
    
    tep_consistent = bool(mean_impossible > mean_normal and (p_value_fmt is not None and p_value_fmt < 0.05))

    if tep_consistent:
        logger.info("\n✓ Anomalous galaxies have HIGHER Γ_t (TEP-consistent)")
    elif mean_impossible > mean_normal:
        logger.info("\n⚠ Trend in correct direction but not significant")
    else:
        logger.info("\n✗ No Γ_t enhancement in anomalous galaxies")
    
    return {
        'mean_gamma_impossible': mean_impossible,
        'sem_gamma_impossible': sem_impossible,
        'mean_gamma_normal': mean_normal,
        'sem_gamma_normal': sem_normal,
        't_statistic': t_stat,
        'p_value': p_value_fmt,
        'cohens_d': cohens_d,
        'ratio': ratio,
        'tep_consistent': tep_consistent
    }


def analyze_age_ratio_gamma_correlation(df):
    """
    Test direct correlation between age ratio and Γ_t.
    
    This is the most direct TEP test: higher Γ_t should predict higher age ratio.
    """
    logger.info("=" * 70)
    logger.info("ANALYSIS 3: Age Ratio vs Γ_t Correlation")
    logger.info("=" * 70)
    
    rho, p_value = stats.spearmanr(df['gamma_t'], df['age_ratio'])
    r, p_pearson = stats.pearsonr(df['gamma_t'], df['age_ratio'])
    p_value_fmt = format_p_value(p_value)
    p_pearson_fmt = format_p_value(p_pearson)
    
    logger.info(f"Spearman ρ = {rho:.3f}, p = {p_value:.2e}")
    logger.info(f"Pearson r = {r:.3f}, p = {p_pearson:.2e}")
    
    # Linear fit
    slope, intercept, r_value, p_fit, std_err = stats.linregress(
        df['gamma_t'], df['age_ratio']
    )
    
    logger.info(f"\nLinear fit: age_ratio = {slope:.3f} × Γ_t + {intercept:.3f}")
    logger.info(f"  Slope: {slope:.3f} ± {std_err:.3f}")
    
    if rho > 0 and (p_value_fmt is not None and p_value_fmt < 0.001):
        logger.info(f"\n✓ Strong positive correlation (TEP-consistent)")
    elif rho > 0 and (p_value_fmt is not None and p_value_fmt < 0.05):
        logger.info(f"\n✓ Significant positive correlation (TEP-consistent)")
    else:
        logger.info(f"\n⚠ Weak or no correlation")
    
    return {
        'spearman_rho': rho,
        'spearman_p': p_value_fmt,
        'pearson_r': r,
        'pearson_p': p_pearson_fmt,
        'slope': slope,
        'slope_err': std_err,
        'intercept': intercept,
        'tep_consistent': bool(rho > 0 and (p_value_fmt is not None and p_value_fmt < 0.05))
    }


def analyze_tep_resolution(df):
    """
    Quantify how much of the "impossibility" TEP explains.
    
    Under TEP, the TRUE age is:
        t_true = t_observed / Γ_t
    
    For galaxies with Γ_t > 1 (deep potentials), this reduces the apparent age.
    For galaxies with Γ_t < 1 (shallow potentials), this increases the apparent age.
    
    The key insight: "anomalous" galaxies (age_ratio > 0.5) should have Γ_t > 1,
    so the correction should resolve them. We focus on this specific population.
    """
    logger.info("=" * 70)
    logger.info("ANALYSIS 4: TEP Resolution of Impossibility")
    logger.info("=" * 70)
    
    # Calculate TEP-corrected age ratio
    df = df.copy()
    df['age_ratio_corrected'] = df['age_ratio'] / df['gamma_t']
    
    # Focus on the "anomalous" galaxies specifically
    impossible_mask = df['age_ratio'] > 0.5
    n_impossible_raw = impossible_mask.sum()
    
    if n_impossible_raw > 0:
        # Check if anomalous galaxies have high Γ_t (as TEP predicts)
        impossible_df = df[impossible_mask]
        mean_gamma_impossible = impossible_df['gamma_t'].mean()
        min_gamma_impossible = impossible_df['gamma_t'].min()
        
        logger.info(f"Anomalous galaxies (raw): N = {n_impossible_raw}")
        logger.info(f"  Mean Γ_t: {mean_gamma_impossible:.2f}")
        logger.info(f"  Min Γ_t: {min_gamma_impossible:.2f}")
        logger.info(f"  All have Γ_t > 1: {min_gamma_impossible > 1}")
        
        # How many are resolved after TEP correction?
        n_resolved = (impossible_df['age_ratio_corrected'] <= 0.5).sum()
        n_still_impossible = n_impossible_raw - n_resolved
        
        logger.info(f"\nAfter TEP correction (÷Γ_t):")
        logger.info(f"  Resolved: {n_resolved}/{n_impossible_raw} ({n_resolved/n_impossible_raw*100:.0f}%)")
        logger.info(f"  Still anomalous: {n_still_impossible}")
        
        reduction = n_resolved / n_impossible_raw * 100
    else:
        logger.info("No anomalous galaxies found in sample")
        reduction = 0
        n_still_impossible = 0
    
    # Mean age ratio before and after for anomalous galaxies only
    if n_impossible_raw > 0:
        mean_raw = impossible_df['age_ratio'].mean()
        mean_corrected = impossible_df['age_ratio_corrected'].mean()
        
        logger.info(f"\nFor anomalous galaxies:")
        logger.info(f"  Mean age ratio (raw): {mean_raw:.3f}")
        logger.info(f"  Mean age ratio (corrected): {mean_corrected:.3f}")
        
        # For the most extreme cases
        extreme = impossible_df[impossible_df['age_ratio'] > 0.9]
        if len(extreme) > 0:
            logger.info(f"\nMost extreme case (age_ratio > 0.9):")
            logger.info(f"  N = {len(extreme)}")
            logger.info(f"  Γ_t = {extreme['gamma_t'].values[0]:.2f}")
            logger.info(f"  Raw age_ratio = {extreme['age_ratio'].values[0]:.3f}")
            logger.info(f"  Corrected age_ratio = {extreme['age_ratio_corrected'].values[0]:.3f}")
    else:
        mean_raw = df['age_ratio'].mean()
        mean_corrected = df['age_ratio_corrected'].mean()
    
    n_resolved_val = int(n_resolved) if n_impossible_raw > 0 else 0
    return {
        'n_impossible_raw': int(n_impossible_raw),
        'n_resolved': n_resolved_val,
        'n_still_impossible': int(n_still_impossible),
        'reduction_pct': float(reduction),
        'mean_age_ratio_raw': float(mean_raw),
        'mean_age_ratio_corrected': float(mean_corrected)
    }


def run_impossible_galaxies_analysis():
    """Run the complete anomalous galaxies analysis."""
    logger.info("=" * 70)
    logger.info("TEP-JWST Step 15: 'Anomalous' Galaxies Analysis")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Testing TEP prediction: 'anomalous' galaxies (age > cosmic age)")
    logger.info("should have the highest predicted Γ_t values.")
    logger.info("")
    
    # Load data
    df = load_uncover_data()
    
    results = {}
    
    # Analysis 1: Identify anomalous galaxies
    id_results, anomalous, normal = identify_impossible_galaxies(df)
    results['identification'] = id_results
    
    # Analysis 2: Compare Γ_t
    if len(anomalous) > 10 and len(normal) > 10:
        results['gamma_comparison'] = compare_gamma_t(anomalous, normal)
    
    # Analysis 3: Age ratio vs Γ_t correlation
    results['correlation'] = analyze_age_ratio_gamma_correlation(df)
    
    # Analysis 4: TEP resolution
    results['resolution'] = analyze_tep_resolution(df)
    
    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    
    corr = results['correlation']
    logger.info(f"Age ratio vs Γ_t: ρ = {corr['spearman_rho']:.3f} (p = {corr['spearman_p']:.2e})")
    
    if 'gamma_comparison' in results:
        comp = results['gamma_comparison']
        logger.info(f"Γ_t ratio (anomalous/normal): {comp['ratio']:.2f}")
    
    res = results['resolution']
    logger.info(f"TEP reduces 'anomalous' galaxies by: {res['reduction_pct']:.1f}%")
    
    # Overall assessment
    if corr['tep_consistent']:
        logger.info("\n✓ 'Anomalous' galaxies are explained by TEP")
        logger.info("  Higher Γ_t → higher apparent age ratio")
    else:
        logger.info("\n⚠ Results inconclusive")
    
    # Save results
    output_file = RESULTS_DIR / f"step_{STEP_NUM}_jwst_impossible_galaxies.json"
    
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return obj
    
    results_serializable = json.loads(json.dumps(results, default=convert_numpy))
    
    with open(output_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    run_impossible_galaxies_analysis()
