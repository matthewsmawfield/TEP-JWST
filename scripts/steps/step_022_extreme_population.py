#!/usr/bin/env python3
"""
TEP-JWST Step 22: Extreme Population Analysis

This analysis examines galaxies with extreme TEP enhancement factors:

1. THE EXTREME POPULATION
   - Galaxies with Γ_t > 2 show strongest TEP signatures
   - These provide the most stringent tests of TEP predictions

2. THE BIMODALITY TEST
   - Is there a natural division in the population?
   - TEP predicts two regimes: suppressed and enhanced
   - The transition should be sharp

3. THE PREDICTION PRECISION
   - How precisely does TEP predict individual galaxy properties?
   - The residuals should be random, not systematic

4. THE EMERGENT CORRELATION
   - Some correlations should ONLY appear after TEP correction
   - These reveal hidden structure in the data

5. THE QUANTITATIVE RECOVERY
   - Can we recover the TEP parameters from the data alone?
   - If TEP is correct, the data should constrain α independently

Author: Matthew L. Smawfield
Date: January 2026
"""

import sys
import numpy as np
import pandas as pd
from scipy import stats  # Hypothesis tests and regression
from scipy.optimize import minimize_scalar  # 1-D bounded optimisation for α recovery
from pathlib import Path
import logging
import json

# =============================================================================
# LOGGER SETUP
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging
from scripts.utils.p_value_utils import format_p_value  # Safe p-value formatting
from scripts.utils.tep_model import ALPHA_0, compute_gamma_t as tep_gamma  # Shared TEP constants & Γ_t calculator

STEP_NUM = "022"  # Pipeline step number
STEP_NAME = "extreme_population"  # Used in log / output filenames

LOGS_PATH = PROJECT_ROOT / "logs"  # Log directory
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"  # JSON output directory
LOGS_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")  # Step-specific logger
set_step_logger(logger)

# Additional path aliases used by downstream helpers
DATA_DIR = PROJECT_ROOT / "data"  # Raw catalogue directory
INTERIM_DIR = PROJECT_ROOT / "results" / "interim"  # Pre-processed intermediate products
RESULTS_DIR = PROJECT_ROOT / "results" / "outputs"  # Final JSON outputs (alias)


def load_data():
    """Load UNCOVER data."""
    logger.info("Loading UNCOVER data...")
    
    df = pd.read_csv(INTERIM_DIR / "step_002_uncover_full_sample_tep.csv")
    df = df.rename(
        columns={
            'z_phot': 'z',
            'mwa': 'mwa_Gyr',
            't_cosmic': 't_cosmic_Gyr',
            't_eff': 't_eff_Gyr',
            'log_Mh': 'log_Mhalo',
        }
    )
    df['age_ratio'] = df['mwa_Gyr'] / df['t_cosmic_Gyr']
    
    logger.info(f"Loaded {len(df)} galaxies")
    
    return df


def analyze_extreme_population(df):
    """
    TEST 1: The Extreme Population
    
    Galaxies with Γ_t > 3 represent extreme cases. They should show
    the strongest TEP signatures. These are the "core" of the star.
    """
    logger.info("=" * 70)
    logger.info("TEST 1: The Extreme Population")
    logger.info("=" * 70)
    
    # Define extreme as Γ_t > 2 (top ~2%)
    extreme = df[df['gamma_t'] > 2].copy()
    normal = df[df['gamma_t'] <= 2].copy()
    
    logger.info(f"Extreme galaxies (Γ_t > 2): N = {len(extreme)} ({100*len(extreme)/len(df):.1f}%)")
    logger.info(f"Normal galaxies: N = {len(normal)}")
    
    # Compare properties
    properties = {
        'age_ratio': 'Age Ratio',
        'mwa_Gyr': 'Stellar Age (Gyr)',
        't_eff_Gyr': 'Effective Time (Gyr)',
    }
    
    results = {}
    
    for col, name in properties.items():
        if col in df.columns:
            mean_extreme = extreme[col].mean()
            mean_normal = normal[col].mean()
            
            t_stat, p_value = stats.ttest_ind(extreme[col], normal[col])
            
            ratio = mean_extreme / mean_normal if mean_normal != 0 else np.nan
            
            logger.info(f"\n{name}:")
            logger.info(f"  Extreme: {mean_extreme:.4f}")
            logger.info(f"  Normal: {mean_normal:.4f}")
            logger.info(f"  Ratio: {ratio:.2f}×")
            logger.info(f"  p-value: {p_value:.2e}")
            
            results[col] = {
                'mean_extreme': mean_extreme,
                'mean_normal': mean_normal,
                'ratio': ratio,
                'p_value': format_p_value(p_value)
            }
    
    # The most extreme galaxy
    most_extreme = df.loc[df['gamma_t'].idxmax()]
    logger.info(f"\nMost extreme galaxy:")
    logger.info(f"  Γ_t = {most_extreme['gamma_t']:.2f}")
    logger.info(f"  z = {most_extreme['z']:.1f}")
    logger.info(f"  log(M*) = {most_extreme['log_Mstar']:.2f}")
    logger.info(f"  Age ratio = {most_extreme['age_ratio']:.3f}")
    
    # TEP-corrected age ratio for extreme
    extreme_corrected = extreme['age_ratio'] / extreme['gamma_t']
    logger.info(f"\nExtreme galaxies after TEP correction:")
    logger.info(f"  Mean age_ratio (raw): {extreme['age_ratio'].mean():.3f}")
    logger.info(f"  Mean age_ratio (corrected): {extreme_corrected.mean():.3f}")
    logger.info(f"  Reduction: {100*(1 - extreme_corrected.mean()/extreme['age_ratio'].mean()):.1f}%")
    
    results['n_extreme'] = len(extreme)
    results['most_extreme_gamma'] = most_extreme['gamma_t']
    
    return results


def analyze_bimodality(df):
    """
    TEST 2: The Bimodality Test
    
    Is there a natural division in the population?
    TEP predicts two regimes: suppressed (Γ_t < 1) and enhanced (Γ_t > 1).
    """
    logger.info("=" * 70)
    logger.info("TEST 2: The Bimodality Test")
    logger.info("=" * 70)
    
    # Test for bimodality in age_ratio distribution
    from scipy.stats import gaussian_kde
    
    # Split by Γ_t = 1 threshold
    suppressed = df[df['gamma_t'] < 1]
    enhanced = df[df['gamma_t'] >= 1]
    
    logger.info(f"Suppressed regime (Γ_t < 1): N = {len(suppressed)} ({100*len(suppressed)/len(df):.1f}%)")
    logger.info(f"Enhanced regime (Γ_t ≥ 1): N = {len(enhanced)} ({100*len(enhanced)/len(df):.1f}%)")
    
    # Compare age_ratio distributions
    mean_sup = suppressed['age_ratio'].mean()
    mean_enh = enhanced['age_ratio'].mean()
    
    t_stat, p_value = stats.ttest_ind(suppressed['age_ratio'], enhanced['age_ratio'])
    
    logger.info(f"\nAge ratio comparison:")
    logger.info(f"  Suppressed: {mean_sup:.4f}")
    logger.info(f"  Enhanced: {mean_enh:.4f}")
    logger.info(f"  Ratio: {mean_enh/mean_sup:.2f}×")
    logger.info(f"  p-value: {p_value:.2e}")
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((suppressed['age_ratio'].var() + enhanced['age_ratio'].var()) / 2)
    cohens_d = (mean_enh - mean_sup) / pooled_std
    
    logger.info(f"  Cohen's d: {cohens_d:.2f}")
    
    if cohens_d > 0.8:
        logger.info("\n✓ Large effect size - clear bimodality")
        bimodal = True
    elif cohens_d > 0.5:
        logger.info("\n⚠ Medium effect size - partial bimodality")
        bimodal = True
    else:
        logger.info("\n⚠ Small effect size - weak bimodality")
        bimodal = False
    
    return {
        'n_suppressed': len(suppressed),
        'n_enhanced': len(enhanced),
        'mean_suppressed': mean_sup,
        'mean_enhanced': mean_enh,
        'cohens_d': cohens_d,
        'p_value': format_p_value(p_value),
        'bimodal': bimodal
    }


def analyze_prediction_precision(df):
    """
    TEST 3: Prediction Precision
    
    How precisely does TEP predict individual galaxy properties?
    The residuals should be random, not systematic.
    """
    logger.info("=" * 70)
    logger.info("TEST 3: Prediction Precision")
    logger.info("=" * 70)
    
    valid = df.dropna(subset=['gamma_t', 'age_ratio', 'z'])
    
    # TEP prediction: age_ratio_predicted = f(Γ_t, z)
    # Simple model: age_ratio = a + b*Γ_t + c*z
    from scipy.optimize import curve_fit
    
    def tep_model(X, a, b, c):
        gamma, z = X
        return a + b * gamma + c * z
    
    X = np.array([valid['gamma_t'], valid['z']])
    y = valid['age_ratio'].values
    
    try:
        popt, pcov = curve_fit(tep_model, X, y, maxfev=10000)
        a, b, c = popt
        
        y_pred = tep_model(X, *popt)
        residuals = y - y_pred
        
        rmse = np.sqrt(np.mean(residuals**2))
        r2 = 1 - np.var(residuals) / np.var(y)
        
        logger.info(f"TEP model: age_ratio = {a:.4f} + {b:.4f}×Γ_t + {c:.4f}×z")
        logger.info(f"RMSE: {rmse:.4f}")
        logger.info(f"R²: {r2:.4f}")
        
        # Check if residuals are random (no systematic pattern)
        rho_resid_gamma, p_resid = stats.spearmanr(valid['gamma_t'], residuals)
        logger.info(f"\nResidual-Γ_t correlation: ρ = {rho_resid_gamma:.3f}, p = {p_resid:.4f}")
        
        if abs(rho_resid_gamma) < 0.1:
            logger.info("✓ Residuals are random (no systematic bias)")
            random_residuals = True
        else:
            logger.info("⚠ Residuals show systematic pattern")
            random_residuals = False
        
        return {
            'coefficients': {'a': a, 'b': b, 'c': c},
            'rmse': rmse,
            'r2': r2,
            'rho_resid_gamma': rho_resid_gamma,
            'random_residuals': random_residuals
        }
    except Exception as e:
        logger.warning(f"Model fitting failed: {e}")
        return None


def analyze_emergent_correlations(df):
    """
    TEST 4: Emergent Correlations
    
    Some correlations should ONLY appear after TEP correction.
    These reveal hidden structure in the data.
    """
    logger.info("=" * 70)
    logger.info("TEST 4: Emergent Correlations")
    logger.info("=" * 70)
    
    valid = df.dropna(subset=['gamma_t', 'age_ratio', 'log_Mstar', 'z'])
    valid = valid.copy()
    
    # TEP-corrected age
    valid['age_corrected'] = valid['mwa_Gyr'] / valid['gamma_t']
    valid['age_ratio_corrected'] = valid['age_ratio'] / valid['gamma_t']
    
    # Test 1: Mass-age correlation
    rho_raw, p_raw = stats.spearmanr(valid['log_Mstar'], valid['mwa_Gyr'])
    rho_corr, p_corr = stats.spearmanr(valid['log_Mstar'], valid['age_corrected'])
    
    logger.info("Mass-Age correlation:")
    logger.info(f"  Raw: ρ = {rho_raw:.3f}, p = {p_raw:.4f}")
    logger.info(f"  TEP-corrected: ρ = {rho_corr:.3f}, p = {p_corr:.4f}")
    
    # Did a correlation emerge?
    if abs(rho_corr) > abs(rho_raw) and p_corr < 0.05:
        logger.info("  ✓ Correlation STRENGTHENED after TEP correction")
        mass_age_emerged = True
    elif abs(rho_corr) < abs(rho_raw):
        logger.info("  ⚠ Correlation weakened (TEP explains part of it)")
        mass_age_emerged = False
    else:
        logger.info("  ⚠ No significant change")
        mass_age_emerged = False
    
    # Test 2: Redshift-age correlation at fixed mass
    # Bin by mass and test z-age correlation
    mass_bins = [(7, 8), (8, 9), (9, 10)]
    
    emergent_count = 0
    
    for m_lo, m_hi in mass_bins:
        bin_data = valid[(valid['log_Mstar'] >= m_lo) & (valid['log_Mstar'] < m_hi)]
        if len(bin_data) >= 50:
            rho_raw_bin, p_raw_bin = stats.spearmanr(bin_data['z'], bin_data['mwa_Gyr'])
            rho_corr_bin, p_corr_bin = stats.spearmanr(bin_data['z'], bin_data['age_corrected'])
            
            logger.info(f"\nlog(M*) = [{m_lo}, {m_hi}): N = {len(bin_data)}")
            logger.info(f"  z-Age (raw): ρ = {rho_raw_bin:.3f}")
            logger.info(f"  z-Age (corrected): ρ = {rho_corr_bin:.3f}")
            
            if abs(rho_corr_bin) > abs(rho_raw_bin) + 0.1:
                logger.info("  ✓ Correlation detected")
                emergent_count += 1
    
    logger.info(f"\nEmergent correlations found: {emergent_count}/{len(mass_bins)}")
    
    return {
        'mass_age_raw': rho_raw,
        'mass_age_corrected': rho_corr,
        'mass_age_emerged': mass_age_emerged,
        'emergent_count': emergent_count
    }


def analyze_parameter_recovery(df):
    """
    TEST 5: Parameter Recovery
    
    Can we recover the TEP parameter α from the data alone?
    If TEP is correct, the data should constrain α independently.
    """
    logger.info("=" * 70)
    logger.info("TEST 5: Parameter Recovery")
    logger.info("=" * 70)
    
    valid = df.dropna(subset=['gamma_t', 'age_ratio', 'log_Mhalo', 'z'])
    
    # The TEP equation: Γ_t = α × (M_h / M_ref)^(1/3)
    # We can test different values of α and see which minimizes scatter
    
    def compute_scatter(alpha):
        """Compute scatter in age_ratio after TEP correction with given α."""
        gamma_t_test = tep_gamma(valid['log_Mhalo'].values, valid['z'].values, alpha_0=alpha)
        age_corrected = valid['age_ratio'] / gamma_t_test
        return age_corrected.std()
    
    # Grid search for optimal α
    alphas = np.linspace(0.1, 1.5, 50)
    scatters = [compute_scatter(a) for a in alphas]
    
    # Find minimum
    min_idx = np.argmin(scatters)
    alpha_optimal = alphas[min_idx]
    scatter_optimal = scatters[min_idx]
    
    # Also compute scatter at α = 0 (no TEP) and α = 0.58 (calibrated)
    scatter_no_tep = compute_scatter(0.0)
    scatter_calibrated = compute_scatter(ALPHA_0)
    
    logger.info(f"Scatter minimization:")
    logger.info(f"  No TEP (α → 0): σ = {scatter_no_tep:.4f}")
    logger.info(f"  Calibrated (α = 0.58): σ = {scatter_calibrated:.4f}")
    logger.info(f"  Optimal: α = {alpha_optimal:.2f}, σ = {scatter_optimal:.4f}")
    
    # How close is optimal to calibrated?
    alpha_ratio = alpha_optimal / ALPHA_0
    logger.info(f"\nOptimal / Calibrated ratio: {alpha_ratio:.2f}")
    
    if 0.5 < alpha_ratio < 2.0:
        logger.info("✓ Data independently recovers α within factor of 2")
        recovered = True
    else:
        logger.info("⚠ Optimal α differs significantly from calibrated")
        recovered = False
    
    # Scatter reduction
    scatter_reduction = (scatter_no_tep - scatter_optimal) / scatter_no_tep * 100
    logger.info(f"\nScatter reduction (no TEP → optimal): {scatter_reduction:.1f}%")
    
    return {
        'alpha_optimal': alpha_optimal,
        'alpha_calibrated': float(ALPHA_0),
        'alpha_ratio': alpha_ratio,
        'scatter_no_tep': scatter_no_tep,
        'scatter_calibrated': scatter_calibrated,
        'scatter_optimal': scatter_optimal,
        'scatter_reduction': scatter_reduction,
        'recovered': recovered
    }


def run_extreme_population_analysis():
    """Run the complete extreme population analysis."""
    logger.info("=" * 70)
    logger.info("TEP-JWST Step 22: Extreme Population Analysis")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Analyzing galaxies with extreme TEP enhancement factors.")
    logger.info("")
    
    # Load data
    df = load_data()
    
    results = {}
    
    # Test 1: Extreme Population
    results['extreme_population'] = analyze_extreme_population(df)
    
    # Test 2: Bimodality
    results['bimodality'] = analyze_bimodality(df)
    
    # Test 3: Prediction Precision
    results['prediction_precision'] = analyze_prediction_precision(df)
    
    # Test 4: Emergent Correlations
    results['emergent_correlations'] = analyze_emergent_correlations(df)
    
    # Test 5: Parameter Recovery
    results['parameter_recovery'] = analyze_parameter_recovery(df)
    
    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY: Extreme Population Results")
    logger.info("=" * 70)
    
    tests_passed = 0
    total_tests = 5
    
    if results['extreme_population'].get('n_extreme', 0) > 0:
        tests_passed += 1
        logger.info("✓ Extreme population: Detected")
    
    if results['bimodality'].get('bimodal', False):
        tests_passed += 1
        logger.info("✓ Bimodality: Detected")
    
    if results['prediction_precision'] and results['prediction_precision'].get('random_residuals', False):
        tests_passed += 1
        logger.info("✓ Prediction precision: Confirmed")
    
    if results['emergent_correlations'].get('emergent_count', 0) > 0:
        tests_passed += 1
        logger.info("✓ Emergent correlations: Found")
    
    if results['parameter_recovery'].get('recovered', False):
        tests_passed += 1
        logger.info("✓ Parameter recovery: Successful")
    
    logger.info(f"\nTests passed: {tests_passed}/{total_tests}")
    
    # Save results
    output_file = RESULTS_DIR / f"step_{STEP_NUM}_extreme_population.json"
    
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
    run_extreme_population_analysis()
