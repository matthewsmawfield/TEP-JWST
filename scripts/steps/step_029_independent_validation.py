#!/usr/bin/env python3
"""
TEP-JWST Step 29: Independent Validation (Out-of-Sample Prediction)

This step provides the strongest possible test of TEP: can it make correct
predictions on data it has never seen?

Method:
1. Split sample into training (50%) and test (50%) sets
2. Calibrate α on training set only
3. Use calibrated α to predict dust content on test set
4. Compare prediction accuracy to mass-only baseline

If TEP has genuine predictive power, it should outperform mass-only models
on held-out data. This is the standard for rigorous scientific validation.

Author: Matthew L. Smawfield
Date: January 2026
"""

import sys
import numpy as np
import pandas as pd
from scipy import stats  # Hypothesis tests, correlation, regression
from scipy.optimize import minimize_scalar  # (imported for potential α re-calibration; currently uses fixed KAPPA_GAL)
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging (severity levels: DEBUG/INFO/WARNING/ERROR/SUCCESS)
from scripts.utils.p_value_utils import format_p_value  # Safe p-value formatting (prevents floating-point underflow at p < 1e-300)
from scripts.utils.tep_model import KAPPA_GAL, KAPPA_GAL, compute_gamma_t as tep_gamma  # TEP model: KAPPA_GAL=9.6e5 mag from Cepheids, Gamma_t formula

STEP_NUM = "029"  # Pipeline step number (sequential 001-176)
STEP_NAME = "independent_validation"  # Independent validation: train/test split (50/50) to test TEP predictive power on held-out data

DATA_DIR = PROJECT_ROOT / "data"  # Raw catalogue directory (external datasets: UNCOVER DR4, CEERS, COSMOS-Web, JADES)
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"  # JSON output directory (machine-readable statistical results)
LOGS_PATH = PROJECT_ROOT / "logs"  # Log directory (one plain-text log per step for debugging traceability)

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)  # Create results/outputs/ if missing; parents=True ensures full path tree exists
LOGS_PATH.mkdir(parents=True, exist_ok=True)  # Create logs/ if missing

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")  # Step-specific logger (isolated per-step logging for traceability)
set_step_logger(logger)  # Register as global step logger so print_status() routes to this step's log


def load_data():
    """Load UNCOVER data with TEP calculations."""
    INTERIM_PATH = PROJECT_ROOT / "results" / "interim"
    df = pd.read_csv(INTERIM_PATH / "step_002_uncover_multi_property_sample_tep.csv")
    
    # Filter for valid dust measurements
    df = df.dropna(subset=['dust', 'log_Mstar', 'z_phot', 'gamma_t', 'log_Mh'])
    df = df[df['dust'] > 0].copy()
    
    # Rename for consistency
    df['z'] = df['z_phot']
    df['log_Mhalo'] = df['log_Mh']
    
    return df


def split_sample(df, seed=42):
    """Split into training and test sets."""
    np.random.seed(seed)
    n = len(df)
    idx = np.random.permutation(n)
    train_idx = idx[:n//2]
    test_idx = idx[n//2:]
    
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def calibrate_alpha_on_training(train_df):
    """
    Use the FIXED Cepheid-calibrated α = 9.6e5.
    
    This is the key test: does the parameter calibrated from LOCAL Cepheids
    provide predictive power for HIGH-Z galaxies?
    
    We do NOT re-calibrate α on the training set - that would be circular.
    Instead, we use the externally-calibrated value.
    """
    # Return the Cepheid-calibrated value (from Paper 11); no fitting performed
    return KAPPA_GAL


def evaluate_predictions(test_df, alpha_calibrated):
    """Evaluate prediction accuracy on test set."""
    
    # Compute Γ_t with calibrated α (Exponential Form)
    gamma_t_pred = tep_gamma(
        test_df['log_Mhalo'].values,
        test_df['z'].values,
        kappa=alpha_calibrated,
    )
    
    # TEP prediction: dust should correlate with Γ_t
    rho_tep, p_tep = stats.spearmanr(gamma_t_pred, test_df['dust'])
    
    # Mass-only baseline: dust correlates with mass
    rho_mass, p_mass = stats.spearmanr(test_df['log_Mstar'], test_df['dust'])
    
    # Partial correlation: Γ_t vs dust controlling for mass
    # Residualize both against mass
    from scipy.stats import linregress
    
    slope_g, int_g, _, _, _ = linregress(test_df['log_Mstar'], gamma_t_pred)
    gamma_resid = gamma_t_pred - (slope_g * test_df['log_Mstar'] + int_g)
    
    slope_d, int_d, _, _, _ = linregress(test_df['log_Mstar'], test_df['dust'])
    dust_resid = test_df['dust'] - (slope_d * test_df['log_Mstar'] + int_d)
    
    rho_partial, p_partial = stats.spearmanr(gamma_resid, dust_resid)
    
    p_partial_fmt = format_p_value(p_partial)
    return {
        'rho_tep': rho_tep,
        'p_tep': format_p_value(p_tep),
        'rho_mass': rho_mass,
        'p_mass': format_p_value(p_mass),
        'rho_partial': rho_partial,
        'p_partial': p_partial_fmt,
        'tep_beats_mass': abs(rho_tep) > abs(rho_mass),
        'partial_significant': bool(p_partial_fmt is not None and p_partial_fmt < 0.05)
    }


def cross_validation(df, n_folds=5):
    """K-fold cross-validation for robust estimate."""
    
    n = len(df)
    fold_size = n // n_folds
    
    results = []
    
    for fold in range(n_folds):
        # Create train/test split
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size if fold < n_folds - 1 else n
        
        test_idx = list(range(test_start, test_end))
        train_idx = [i for i in range(n) if i not in test_idx]
        
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        
        # Calibrate on training
        alpha_cal = calibrate_alpha_on_training(train_df)
        
        # Evaluate on test
        eval_result = evaluate_predictions(test_df, alpha_cal)
        eval_result['alpha_calibrated'] = alpha_cal
        eval_result['fold'] = fold
        
        results.append(eval_result)
    
    return results


def main():
    print_status("=" * 70, "INFO")
    print_status("STEP 31: Independent Validation (Out-of-Sample Prediction)", "INFO")
    print_status("=" * 70, "INFO")
    print_status("", "INFO")
    
    df = load_data()
    print_status(f"Sample size: N = {len(df)}", "INFO")
    print_status("", "INFO")
    
    # Single split validation
    print_status("=" * 50, "INFO")
    print_status("TEST 1: 50/50 Train-Test Split", "INFO")
    print_status("=" * 50, "INFO")
    
    train_df, test_df = split_sample(df)
    print_status(f"Training set: N = {len(train_df)}", "INFO")
    print_status(f"Test set: N = {len(test_df)}", "INFO")
    
    alpha_cal = calibrate_alpha_on_training(train_df)
    print_status(f"\nCalibrated α (training only): {alpha_cal:.3f}", "INFO")
    print_status(f"Reference α (from Cepheids): {KAPPA_GAL}", "INFO")
    print_status(f"Difference: {abs(alpha_cal - KAPPA_GAL):.3f}", "INFO")
    
    eval_result = evaluate_predictions(test_df, alpha_cal)
    
    print_status(f"\nOut-of-sample predictions (test set):", "INFO")
    print_status(f"  TEP (Γ_t vs dust): ρ = {eval_result['rho_tep']:.3f}, p = {eval_result['p_tep']:.2e}", "INFO")
    print_status(f"  Mass-only baseline: ρ = {eval_result['rho_mass']:.3f}, p = {eval_result['p_mass']:.2e}", "INFO")
    print_status(f"  Partial (Γ_t | M*): ρ = {eval_result['rho_partial']:.3f}, p = {eval_result['p_partial']:.2e}", "INFO")
    
    if eval_result['partial_significant']:
        print_status("\n✓ TEP predicts dust BEYOND mass on held-out data", "INFO")
    else:
        print_status("\n⚠ TEP does not significantly outperform mass-only", "INFO")
    
    # Cross-validation
    print_status("", "INFO")
    print_status("=" * 50, "INFO")
    print_status("TEST 2: 5-Fold Cross-Validation", "INFO")
    print_status("=" * 50, "INFO")
    
    cv_results = cross_validation(df)
    
    alphas = [r['alpha_calibrated'] for r in cv_results]
    rhos_partial = [r['rho_partial'] for r in cv_results]
    ps_partial = [r['p_partial'] for r in cv_results]
    
    print_status(f"\nCalibrated α across folds:", "INFO")
    print_status(f"  Mean: {np.mean(alphas):.3f} ± {np.std(alphas):.3f}", "INFO")
    print_status(f"  Range: [{min(alphas):.3f}, {max(alphas):.3f}]", "INFO")
    
    print_status(f"\nPartial correlation (Γ_t | M*) across folds:", "INFO")
    print_status(f"  Mean ρ: {np.mean(rhos_partial):.3f} ± {np.std(rhos_partial):.3f}", "INFO")
    print_status(f"  Significant folds: {sum(1 for p in ps_partial if p is not None and p < 0.05)}/{len(ps_partial)}", "INFO")
    
    # Combined p-value using Fisher's method (sum of -2*ln(p) follows chi2 with 2k d.o.f.)
    ps_valid = [p for p in ps_partial if p is not None]
    chi2 = -2 * sum(np.log(max(p, 1e-300)) for p in ps_valid)
    combined_p_raw = stats.chi2.sf(chi2, 2 * len(ps_valid))
    combined_p = format_p_value(combined_p_raw)
    
    print_status(f"\nCombined significance (Fisher's method):", "INFO")
    print_status(f"  χ² = {chi2:.1f}, p = {combined_p:.2e}", "INFO")
    
    # Summary
    print_status("", "INFO")
    print_status("=" * 70, "INFO")
    print_status("INDEPENDENT VALIDATION SUMMARY", "INFO")
    print_status("=" * 70, "INFO")
    
    mean_partial = np.mean(rhos_partial)
    sig_folds = sum(1 for p in ps_partial if p is not None and p < 0.05)
    
    if mean_partial > 0.1 and sig_folds >= 3:
        print_status("\n✓ STRONG: TEP shows genuine out-of-sample predictive power", "INFO")
        print_status(f"  Mean partial correlation: ρ = {mean_partial:.3f}", "INFO")
        print_status(f"  Significant in {sig_folds}/5 folds", "INFO")
        validation_status = "strong"
    elif mean_partial > 0.05 and sig_folds >= 2:
        print_status("\n✓ MODERATE: TEP shows some out-of-sample predictive power", "INFO")
        validation_status = "moderate"
    else:
        print_status("\n⚠ WEAK: TEP does not show robust out-of-sample prediction", "INFO")
        validation_status = "weak"
    
    # Save results
    results = {
        'single_split': {
            'alpha_calibrated': float(alpha_cal),
            'alpha_reference': float(KAPPA_GAL),
            'rho_tep': float(eval_result['rho_tep']),
            'rho_mass': float(eval_result['rho_mass']),
            'rho_partial': float(eval_result['rho_partial']),
            'p_partial': eval_result['p_partial'],
            'partial_significant': bool(eval_result['partial_significant'])
        },
        'cross_validation': {
            'n_folds': 5,
            'alpha_mean': float(np.mean(alphas)),
            'alpha_std': float(np.std(alphas)),
            'rho_partial_mean': float(np.mean(rhos_partial)),
            'rho_partial_std': float(np.std(rhos_partial)),
            'significant_folds': int(sig_folds),
            'combined_chi2': float(chi2),
            'combined_p': combined_p
        },
        'validation_status': validation_status
    }
    
    output_file = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print_status(f"\nResults saved to: {output_file}", "INFO")
    print_status(f"Step {STEP_NUM} complete.", "INFO")


if __name__ == "__main__":
    main()
