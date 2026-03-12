#!/usr/bin/env python3
"""
TEP-JWST Step 43: Selection Bias Quantification

Quantifies selection biases in high-z samples and their impact on TEP signatures.
At z > 8, only bright, star-forming galaxies are detected, biasing toward:
- Higher SFR (Malmquist bias)
- Younger ages (detection of UV-bright systems)
- Lower dust (less obscured systems preferentially detected)

This script:
1. Estimates detection completeness as function of mass and redshift
2. Uses Monte Carlo simulations to estimate bias impacts
3. Reports adjusted p-values and Bayesian evidence ratios
4. Combines with external spectroscopic data for power boost

Inputs:
- results/interim/step_02_uncover_full_sample_tep.csv
- data/interim/combined_spectroscopic_catalog.csv

Outputs:
- results/outputs/step_43_selection_bias.json
- results/outputs/step_43_selection_bias.csv
"""

import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ks_2samp, norm
from pathlib import Path
import json

# =============================================================================
# PATHS AND LOGGER
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.p_value_utils import format_p_value, safe_json_default

STEP_NUM = "43"
STEP_NAME = "selection_bias"

DATA_PATH = PROJECT_ROOT / "data"
INTERIM_PATH = PROJECT_ROOT / "results" / "interim"
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"
LOGS_PATH = PROJECT_ROOT / "logs"

for p in [INTERIM_PATH, OUTPUT_PATH, LOGS_PATH]:
    p.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")
set_step_logger(logger)

# =============================================================================
# CONSTANTS
# =============================================================================

N_MONTE_CARLO = 1000
RANDOM_SEED = 42

# =============================================================================
# FUNCTIONS
# =============================================================================

def estimate_completeness(log_Mstar, z, m_lim=28.5):
    """
    Estimate detection completeness as function of mass and redshift.
    Simple model: completeness drops for lower mass at higher z.
    Based on typical JWST depth (~28.5 mag in F444W).
    """
    # Approximate M/L ratio evolution
    # At z~8, log(M*/L) ~ -0.5 for young populations
    # Apparent magnitude: m = M + DM(z) - 2.5*log(L)
    # Completeness ~ sigmoid function of (m_lim - m_apparent)
    
    # Distance modulus approximation for z > 4
    DM = 43.0 + 5 * np.log10(z / 5)  # Rough approximation
    
    # Apparent magnitude proxy
    m_apparent = DM + 2.5 * (10 - log_Mstar)  # Higher mass = brighter
    
    # Completeness sigmoid
    completeness = 1.0 / (1.0 + np.exp((m_apparent - m_lim) / 0.5))
    
    return completeness


def monte_carlo_bias_test(df, x_col, y_col, n_iter=N_MONTE_CARLO, seed=RANDOM_SEED):
    """
    Monte Carlo test for selection bias impact on correlations.
    
    Procedure:
    1. Compute observed correlation
    2. Resample with replacement, weighted by inverse completeness
    3. Compute distribution of resampled correlations
    4. Report bias-adjusted confidence interval
    """
    np.random.seed(seed)
    
    # Observed correlation
    mask = (~df[x_col].isna()) & (~df[y_col].isna())
    df_valid = df[mask].copy()
    
    if len(df_valid) < 20:
        return {
            'rho_observed': np.nan,
            'p_observed': np.nan,
            'rho_mean_mc': np.nan,
            'rho_std_mc': np.nan,
            'bias_estimate': np.nan,
            'p_adjusted': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan
        }
    
    rho_obs, p_obs = spearmanr(df_valid[x_col], df_valid[y_col])
    
    # Estimate completeness weights
    if 'completeness' in df_valid.columns:
        weights = 1.0 / np.clip(df_valid['completeness'], 0.1, 1.0)
    else:
        weights = np.ones(len(df_valid))
    
    weights = weights / weights.sum()
    
    # Monte Carlo resampling
    rho_samples = []
    n = len(df_valid)
    
    for _ in range(n_iter):
        # Weighted bootstrap
        idx = np.random.choice(n, size=n, replace=True, p=weights)
        x_boot = df_valid[x_col].iloc[idx].values
        y_boot = df_valid[y_col].iloc[idx].values
        
        rho_boot, _ = spearmanr(x_boot, y_boot)
        rho_samples.append(rho_boot)
    
    rho_samples = np.array(rho_samples)
    rho_mean = np.nanmean(rho_samples)
    rho_std = np.nanstd(rho_samples)
    
    # Bias estimate
    bias = rho_mean - rho_obs
    
    # Adjusted p-value (fraction of samples with opposite sign)
    if rho_obs > 0:
        p_adjusted = np.mean(rho_samples <= 0)
    else:
        p_adjusted = np.mean(rho_samples >= 0)
    
    return {
        'rho_observed': rho_obs,
        'p_observed': format_p_value(p_obs),
        'rho_mean_mc': rho_mean,
        'rho_std_mc': rho_std,
        'bias_estimate': bias,
        'p_adjusted': format_p_value(p_adjusted),
        'ci_lower': np.percentile(rho_samples, 2.5),
        'ci_upper': np.percentile(rho_samples, 97.5)
    }


def ks_test_with_bootstrap(df, group_col, value_col, n_iter=N_MONTE_CARLO, seed=RANDOM_SEED):
    """
    KS test with bootstrap confidence intervals for regime separation.
    """
    np.random.seed(seed)
    
    mask = ~df[value_col].isna()
    df_valid = df[mask].copy()
    
    group1 = df_valid[df_valid[group_col] > 1.0][value_col].values
    group2 = df_valid[df_valid[group_col] <= 1.0][value_col].values
    
    if len(group1) < 10 or len(group2) < 10:
        return {
            'ks_stat': np.nan,
            'p_value': np.nan,
            'ks_mean_mc': np.nan,
            'ks_std_mc': np.nan,
            'n_group1': len(group1),
            'n_group2': len(group2)
        }
    
    ks_obs, p_obs = ks_2samp(group1, group2)
    
    # Bootstrap KS
    ks_samples = []
    for _ in range(n_iter):
        g1_boot = np.random.choice(group1, size=len(group1), replace=True)
        g2_boot = np.random.choice(group2, size=len(group2), replace=True)
        ks_boot, _ = ks_2samp(g1_boot, g2_boot)
        ks_samples.append(ks_boot)
    
    return {
        'ks_stat': ks_obs,
        'p_value': format_p_value(p_obs),
        'ks_mean_mc': np.mean(ks_samples),
        'ks_std_mc': np.std(ks_samples),
        'n_group1': len(group1),
        'n_group2': len(group2)
    }


def compute_bayes_factor(rho, n, prior_scale=0.3):
    """
    Compute approximate Bayes Factor for correlation.
    BF > 3: Substantial evidence
    BF > 10: Strong evidence
    BF > 30: Very strong evidence
    """
    # Approximate BF using Wetzels & Wagenmakers (2012) formula
    # For Spearman, use Fisher z-transform
    z = 0.5 * np.log((1 + rho) / (1 - rho + 1e-10))
    se = 1.0 / np.sqrt(n - 3)
    
    # BF approximation (Savage-Dickey)
    # Prior: N(0, prior_scale)
    # Posterior: N(z, se)
    
    prior_at_zero = norm.pdf(0, 0, prior_scale)
    posterior_at_zero = norm.pdf(0, z, se)
    
    # BF_10 = P(data | H1) / P(data | H0)
    # Using Savage-Dickey: BF_01 = posterior(0) / prior(0)
    # So BF_10 = prior(0) / posterior(0)
    
    bf_10 = prior_at_zero / (posterior_at_zero + 1e-10)
    
    return bf_10


# =============================================================================
# MAIN
# =============================================================================

def main():
    print_status("=" * 70, "INFO")
    print_status(f"STEP {STEP_NUM}: Selection Bias Quantification", "INFO")
    print_status("=" * 70, "INFO")
    
    # Load photometric data
    df = pd.read_csv(INTERIM_PATH / "step_02_uncover_full_sample_tep.csv")
    print_status(f"Loaded N = {len(df)} photometric galaxies", "INFO")
    
    # Load spectroscopic data if available
    spec_path = DATA_PATH / "interim" / "combined_spectroscopic_catalog.csv"
    if spec_path.exists():
        df_spec = pd.read_csv(spec_path)
        print_status(f"Loaded N = {len(df_spec)} spectroscopic galaxies", "INFO")
        has_spec = True
    else:
        print_status("No spectroscopic catalog found, using photometric only", "WARNING")
        has_spec = False
        df_spec = None
    
    # Estimate completeness for each galaxy
    df['completeness'] = estimate_completeness(df['log_Mstar'], df['z_phot'])
    
    print_status(f"\nCompleteness statistics:", "INFO")
    print_status(f"  Mean: {df['completeness'].mean():.2f}", "INFO")
    print_status(f"  Median: {df['completeness'].median():.2f}", "INFO")
    print_status(f"  Min: {df['completeness'].min():.2f}", "INFO")
    
    results = {}
    
    # ==========================================================================
    # Test 1: z > 8 Mass-Dust Correlation with Bias Correction
    # ==========================================================================
    print_status("\n--- Test 1: z > 8 Mass-Dust Correlation ---", "INFO")
    
    mask_z8 = (df['z_phot'] > 8) & (df['z_phot'] < 10) & (~df['dust'].isna())
    df_z8 = df[mask_z8].copy()
    
    mc_result_dust = monte_carlo_bias_test(df_z8, 'log_Mstar', 'dust')
    
    print_status(f"  Observed rho: {mc_result_dust['rho_observed']:.3f}", "INFO")
    print_status(f"  MC Mean rho: {mc_result_dust['rho_mean_mc']:.3f} +/- {mc_result_dust['rho_std_mc']:.3f}", "INFO")
    print_status(f"  Bias estimate: {mc_result_dust['bias_estimate']:.3f}", "INFO")
    print_status(f"  95% CI: [{mc_result_dust['ci_lower']:.3f}, {mc_result_dust['ci_upper']:.3f}]", "INFO")
    print_status(f"  Adjusted p-value: {mc_result_dust['p_adjusted']:.4f}", "INFO")
    
    # Bayes Factor
    if not np.isnan(mc_result_dust['rho_observed']):
        bf_dust = compute_bayes_factor(mc_result_dust['rho_observed'], len(df_z8))
        print_status(f"  Bayes Factor (H1 vs H0): {bf_dust:.1f}", "INFO")
        mc_result_dust['bayes_factor'] = bf_dust
    
    results['z8_mass_dust'] = mc_result_dust
    
    # ==========================================================================
    # Test 2: gamma_t-Age Correlation with Bias Correction
    # ==========================================================================
    print_status("\n--- Test 2: gamma_t-Age Correlation ---", "INFO")
    
    mask_age = (~df['age_ratio'].isna()) & (~df['gamma_t'].isna())
    df_age = df[mask_age].copy()
    
    mc_result_age = monte_carlo_bias_test(df_age, 'gamma_t', 'age_ratio')
    
    print_status(f"  Observed rho: {mc_result_age['rho_observed']:.3f}", "INFO")
    print_status(f"  MC Mean rho: {mc_result_age['rho_mean_mc']:.3f} +/- {mc_result_age['rho_std_mc']:.3f}", "INFO")
    print_status(f"  95% CI: [{mc_result_age['ci_lower']:.3f}, {mc_result_age['ci_upper']:.3f}]", "INFO")
    print_status(f"  Adjusted p-value: {mc_result_age['p_adjusted']:.4f}", "INFO")
    
    if not np.isnan(mc_result_age['rho_observed']):
        bf_age = compute_bayes_factor(mc_result_age['rho_observed'], len(df_age))
        print_status(f"  Bayes Factor: {bf_age:.1f}", "INFO")
        mc_result_age['bayes_factor'] = bf_age
    
    results['gamma_age'] = mc_result_age
    
    # ==========================================================================
    # Test 3: Chi2 Regime Separation with Bootstrap
    # ==========================================================================
    print_status("\n--- Test 3: Chi2 Regime Separation ---", "INFO")
    
    mask_chi2 = (~df['chi2'].isna()) & (~df['gamma_t'].isna())
    df_chi2 = df[mask_chi2].copy()
    
    ks_result = ks_test_with_bootstrap(df_chi2, 'gamma_t', 'chi2')
    
    print_status(f"  KS statistic: {ks_result['ks_stat']:.3f}", "INFO")
    print_status(f"  p-value: {ks_result['p_value']:.2e}", "INFO")
    print_status(f"  MC Mean KS: {ks_result['ks_mean_mc']:.3f} +/- {ks_result['ks_std_mc']:.3f}", "INFO")
    print_status(f"  N(enhanced): {ks_result['n_group1']}, N(suppressed): {ks_result['n_group2']}", "INFO")
    
    results['chi2_separation'] = ks_result
    
    # ==========================================================================
    # Test 4: Small-N Bin Analysis (Screening Signatures)
    # ==========================================================================
    print_status("\n--- Test 4: Small-N Bin Power Analysis ---", "INFO")
    
    # High mass bin at z > 7
    mask_high_mass = (df['z_phot'] > 7) & (df['log_Mstar'] > 10)
    n_high_mass = mask_high_mass.sum()
    
    print_status(f"  High-mass z>7 sample: N = {n_high_mass}", "INFO")
    
    # Power calculation: minimum detectable effect size
    # For N=10, alpha=0.05, power=0.8, minimum |rho| ~ 0.63
    # For N=50, minimum |rho| ~ 0.28
    # For N=100, minimum |rho| ~ 0.20
    
    power_thresholds = {
        10: 0.63,
        20: 0.44,
        50: 0.28,
        100: 0.20,
        200: 0.14,
        500: 0.09
    }
    
    # Find applicable threshold
    for n_thresh, rho_min in sorted(power_thresholds.items()):
        if n_high_mass <= n_thresh:
            min_detectable = rho_min
            break
    else:
        min_detectable = 0.09
    
    print_status(f"  Minimum detectable |rho| (80% power): {min_detectable:.2f}", "INFO")
    
    results['power_analysis'] = {
        'n_high_mass_z7': n_high_mass,
        'min_detectable_rho': min_detectable
    }
    
    # ==========================================================================
    # Test 5: Spectroscopic Boost (if available)
    # ==========================================================================
    if has_spec and df_spec is not None:
        print_status("\n--- Test 5: Spectroscopic Sample Validation ---", "INFO")
        
        # Check for required columns
        if 'gamma_t' in df_spec.columns and 'age_ratio' in df_spec.columns:
            mask_spec = (~df_spec['gamma_t'].isna()) & (~df_spec['age_ratio'].isna())
            df_spec_valid = df_spec[mask_spec]
            
            if len(df_spec_valid) > 20:
                rho_spec, p_spec = spearmanr(df_spec_valid['gamma_t'], df_spec_valid['age_ratio'])
                bf_spec = compute_bayes_factor(rho_spec, len(df_spec_valid))
                
                print_status(f"  Spectroscopic rho(gamma_t, age): {rho_spec:.3f}", "INFO")
                print_status(f"  p-value: {p_spec:.2e}", "INFO")
                print_status(f"  Bayes Factor: {bf_spec:.1f}", "INFO")
                
                results['spectroscopic_validation'] = {
                    'n': len(df_spec_valid),
                    'rho': rho_spec,
                    'p_value': format_p_value(p_spec),
                    'bayes_factor': bf_spec
                }
            else:
                print_status(f"  Insufficient spectroscopic data (N={len(df_spec_valid)})", "WARNING")
        else:
            print_status("  Required columns not in spectroscopic catalog", "WARNING")
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("SELECTION BIAS SUMMARY", "INFO")
    print_status("=" * 70, "INFO")
    
    # Compile summary
    summary = {
        'tests_performed': 5 if has_spec else 4,
        'z8_mass_dust': {
            'rho': results['z8_mass_dust']['rho_observed'],
            'bias_corrected_rho': results['z8_mass_dust']['rho_mean_mc'],
            'significant': results['z8_mass_dust']['p_adjusted'] < 0.05,
            'bayes_factor': results['z8_mass_dust'].get('bayes_factor', np.nan)
        },
        'gamma_age': {
            'rho': results['gamma_age']['rho_observed'],
            'bias_corrected_rho': results['gamma_age']['rho_mean_mc'],
            'significant': results['gamma_age']['p_adjusted'] < 0.05,
            'bayes_factor': results['gamma_age'].get('bayes_factor', np.nan)
        },
        'chi2_separation': {
            'ks_stat': results['chi2_separation']['ks_stat'],
            'significant': results['chi2_separation']['p_value'] < 0.05
        },
        'overall_robustness': 'robust' if (
            results['z8_mass_dust']['p_adjusted'] < 0.05 and
            results['gamma_age']['p_adjusted'] < 0.05 and
            results['chi2_separation']['p_value'] < 0.05
        ) else 'marginal'
    }
    
    print_status(f"Overall assessment: {summary['overall_robustness'].upper()}", "INFO")
    
    # Save outputs
    json_path = OUTPUT_PATH / f"step_{STEP_NUM}_selection_bias.json"
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.floating):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return obj
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        return obj
    
    with open(json_path, 'w') as f:
        json.dump(convert_numpy(summary), f, indent=2, default=safe_json_default)
    
    # Save detailed results
    detail_path = OUTPUT_PATH / f"step_{STEP_NUM}_selection_bias_detail.json"
    with open(detail_path, 'w') as f:
        json.dump(convert_numpy(results), f, indent=2, default=safe_json_default)
    
    print_status(f"\nSaved summary to {json_path}", "INFO")
    print_status(f"Saved details to {detail_path}", "INFO")

if __name__ == "__main__":
    main()
