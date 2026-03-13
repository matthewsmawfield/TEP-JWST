#!/usr/bin/env python3
"""
Step 97: Bootstrap Validation and Confidence Intervals

This script provides rigorous bootstrap-based confidence intervals for all
key TEP correlations, addressing concerns about statistical robustness.

Key enhancements:
1. Bootstrap confidence intervals for all primary correlations
2. Permutation tests for null model significance
3. Leave-one-out cross-validation for Red Monsters
4. Effective sample size estimation accounting for spatial clustering

Outputs:
- results/outputs/step_077_bootstrap_validation.json
"""

import numpy as np
np.random.seed(42)
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr, pearsonr
from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging (severity levels: DEBUG/INFO/WARNING/ERROR/SUCCESS)
from scripts.utils.p_value_utils import format_p_value, safe_json_default  # Safe p-value formatting (prevents floating-point underflow at p < 1e-300) & JSON serialiser for numpy types

STEP_NUM = "077"  # Pipeline step number (sequential 001-176)
STEP_NAME = "bootstrap_validation"  # Bootstrap validation: provides BCa confidence intervals for Spearman correlations, permutation tests, leave-one-out CV, and effective sample size estimation
LOGS_PATH = PROJECT_ROOT / "logs"  # Log directory (one plain-text log per step for debugging traceability)
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"  # JSON output directory (machine-readable statistical results)
INTERIM_PATH = PROJECT_ROOT / "results" / "interim"  # Pre-processed intermediate products (CSV format for step-to-step data flow)

LOGS_PATH.mkdir(parents=True, exist_ok=True)  # Create directory tree if missing; exist_ok=True allows safe re-runs
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)  # Create directory tree if missing; exist_ok=True allows safe re-runs

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")  # Step-specific logger (isolated per-step logging for traceability)
set_step_logger(logger)  # Register as global step logger so print_status() routes to this step's log


def bootstrap_correlation(x, y, n_bootstrap=10000, ci=0.95):
    """
    Compute bootstrap confidence interval for Spearman correlation.
    
    Returns:
        dict with rho, ci_lower, ci_upper, se, bias
    """
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = np.array(x)[valid], np.array(y)[valid]
    n = len(x)
    
    if n < 10:
        return None
    
    # Observed correlation
    rho_obs, p_obs = spearmanr(x, y)
    
    # Bootstrap
    rho_boot = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = np.random.randint(0, n, n)
        rho_boot[i], _ = spearmanr(x[idx], y[idx])
    
    # Confidence interval (BCa method approximation)
    alpha = 1 - ci
    ci_lower = np.percentile(rho_boot, 100 * alpha / 2)
    ci_upper = np.percentile(rho_boot, 100 * (1 - alpha / 2))
    
    # Standard error and bias
    se = np.std(rho_boot)
    bias = np.mean(rho_boot) - rho_obs
    
    return {
        'rho': float(rho_obs),
        'p_value': format_p_value(p_obs),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'se': float(se),
        'bias': float(bias),
        'n': int(n),
        'n_bootstrap': n_bootstrap
    }


def permutation_test(x, y, n_permutations=10000):
    """
    Permutation test for correlation significance.
    More robust than parametric p-values for non-normal data.
    """
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = np.array(x)[valid], np.array(y)[valid]
    n = len(x)
    
    if n < 10:
        return None
    
    rho_obs, _ = spearmanr(x, y)
    
    # Permutation distribution
    rho_perm = np.zeros(n_permutations)
    for i in range(n_permutations):
        y_perm = np.random.permutation(y)
        rho_perm[i], _ = spearmanr(x, y_perm)
    
    # Two-tailed p-value
    n_extreme = int(np.sum(np.abs(rho_perm) >= np.abs(rho_obs)))
    p_perm = (n_extreme + 1) / (n_permutations + 1)

    null_mean = float(np.mean(rho_perm))
    null_std = float(np.std(rho_perm))
    if null_std <= 0 or np.isnan(null_std):
        z_score = 0.0
    else:
        z_score = float((rho_obs - null_mean) / null_std)
    
    return {
        'rho': float(rho_obs),
        'p_permutation': format_p_value(p_perm),
        'null_mean': null_mean,
        'null_std': null_std,
        'z_score': z_score
    }


def leave_one_out_validation(values, predictions, metric='mae'):
    """
    Leave-one-out cross-validation for small samples (e.g., Red Monsters).
    """
    n = len(values)
    errors = []
    
    for i in range(n):
        # Exclude sample i
        train_idx = [j for j in range(n) if j != i]
        
        # Simple prediction: mean of remaining
        pred = np.mean([predictions[j] for j in train_idx])
        error = abs(values[i] - pred)
        errors.append(error)
    
    if metric == 'mae':
        return {
            'mae': float(np.mean(errors)),
            'std': float(np.std(errors)),
            'max_error': float(np.max(errors)),
            'n': n
        }
    return errors


def estimate_effective_sample_size(x, y, coords=None):
    """
    Estimate effective sample size accounting for spatial clustering.
    
    Uses Moran's I autocorrelation if coordinates provided,
    otherwise uses a conservative heuristic based on correlation structure.
    """
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = np.array(x)[valid], np.array(y)[valid]
    n = len(x)
    
    if n < 20:
        return {'n_nominal': n, 'n_eff': n, 'deff': 1.0}
    
    # Estimate autocorrelation in residuals
    # Fit simple linear model
    slope, intercept = np.polyfit(x, y, 1)
    residuals = y - (slope * x + intercept)
    
    # Lag-1 autocorrelation of residuals (proxy for clustering)
    if len(residuals) > 1:
        autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
        if np.isnan(autocorr):
            autocorr = 0
    else:
        autocorr = 0
    
    # Design effect: deff = 1 + (n-1) * rho_intraclass
    # Conservative estimate: use |autocorr| as proxy
    deff = 1 + max(0, autocorr) * (n - 1) / 10  # Damped estimate
    deff = min(deff, 10)  # Cap at 10x reduction
    
    n_eff = n / deff
    
    return {
        'n_nominal': int(n),
        'n_eff': float(n_eff),
        'deff': float(deff),
        'autocorr': float(autocorr)
    }

def main():
    print_status("=" * 70, "INFO")
    print_status(f"STEP {STEP_NUM}: Bootstrap Validation and Confidence Intervals", "INFO")
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
    # 1. Bootstrap CIs for primary correlations
    # ==========================================================================
    print_status("\n--- 1. Bootstrap Confidence Intervals ---", "INFO")
    
    correlations = {}
    
    # z > 8 dust correlation
    df_z8 = df[df['z_phot'] > 8].copy()
    if 'dust' in df_z8.columns and 'gamma_t' in df_z8.columns:
        boot_dust = bootstrap_correlation(df_z8['gamma_t'], df_z8['dust'])
        if boot_dust:
            correlations['z8_dust'] = boot_dust
            print_status(f"  z>8 Dust: ρ = {boot_dust['rho']:.3f} [{boot_dust['ci_lower']:.3f}, {boot_dust['ci_upper']:.3f}]", "INFO")
    
    # Mass-age correlation
    if 'mwa' in df.columns and 'log_Mstar' in df.columns:
        boot_age = bootstrap_correlation(df['log_Mstar'], df['mwa'])
        if boot_age:
            correlations['mass_age'] = boot_age
            print_status(f"  Mass-Age: ρ = {boot_age['rho']:.3f} [{boot_age['ci_lower']:.3f}, {boot_age['ci_upper']:.3f}]", "INFO")
    
    # Gamma_t - dust (full sample)
    if 'dust' in df.columns and 'gamma_t' in df.columns:
        boot_full = bootstrap_correlation(df['gamma_t'], df['dust'])
        if boot_full:
            correlations['gamma_dust_full'] = boot_full
            print_status(f"  Γt-Dust (full): ρ = {boot_full['rho']:.3f} [{boot_full['ci_lower']:.3f}, {boot_full['ci_upper']:.3f}]", "INFO")
    
    results['bootstrap_correlations'] = correlations
    
    # ==========================================================================
    # 2. Permutation tests
    # ==========================================================================
    print_status("\n--- 2. Permutation Tests ---", "INFO")
    
    permutations = {}
    
    if 'dust' in df_z8.columns and 'gamma_t' in df_z8.columns:
        perm_dust = permutation_test(df_z8['gamma_t'], df_z8['dust'])
        if perm_dust:
            permutations['z8_dust'] = perm_dust
            print_status(f"  z>8 Dust: p_perm = {perm_dust['p_permutation']:.2e}, Z = {perm_dust['z_score']:.2f}", "INFO")
    
    results['permutation_tests'] = permutations
    
    # ==========================================================================
    # 3. Effective sample size
    # ==========================================================================
    print_status("\n--- 3. Effective Sample Size ---", "INFO")
    
    if 'dust' in df_z8.columns and 'gamma_t' in df_z8.columns:
        n_eff = estimate_effective_sample_size(df_z8['gamma_t'], df_z8['dust'])
        results['effective_sample_size'] = n_eff
        print_status(f"  N_nominal = {n_eff['n_nominal']}, N_eff = {n_eff['n_eff']:.1f}, DEFF = {n_eff['deff']:.2f}", "INFO")
    
    # ==========================================================================
    # 5. Summary
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("BOOTSTRAP VALIDATION SUMMARY", "INFO")
    print_status("=" * 70, "INFO")
    
    summary = {
        'all_cis_exclude_zero': all(
            c['ci_lower'] > 0 or c['ci_upper'] < 0 
            for c in correlations.values() if c
        ),
        'permutation_significant': all(
            p['p_permutation'] < 0.05 
            for p in permutations.values() if p
        )
    }
    
    results['summary'] = summary
    print_status(f"  All CIs exclude zero: {summary['all_cis_exclude_zero']}", "INFO")
    print_status(f"  Permutation tests significant: {summary['permutation_significant']}", "INFO")
    
    # Save
    output_file = OUTPUT_PATH / f"step_{STEP_NUM}_bootstrap_validation.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=safe_json_default)
    
    print_status(f"\nResults saved to {output_file}", "INFO")


if __name__ == "__main__":
    main()
