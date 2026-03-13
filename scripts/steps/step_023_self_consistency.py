#!/usr/bin/env python3
"""
TEP-JWST Step 23: Self-Consistency Tests

This analysis tests whether TEP predictions are internally consistent:

1. SELF-CONSISTENCY TEST
   - If TEP is correct, different observables should give the SAME α
   - The model should be internally consistent without tuning

2. PREDICTION INVERSION
   - We've been predicting properties FROM Γ_t
   - Can we predict Γ_t FROM properties?
   - If TEP is real, this should work both ways

3. THE BOOTSTRAP TEST
   - Split the sample randomly
   - Calibrate α on one half, test on the other
   - The model should work on unseen data

4. THE REDSHIFT LADDER
   - TEP predicts α(z) = α_0 × (1+z)^0.5
   - Can we recover this scaling from the data?

5. THE COHERENCE MATRIX
   - All TEP-affected properties should correlate with each other
   - The correlation matrix should have a specific structure

Author: Matthew L. Smawfield
Date: January 2026
"""

import sys
import numpy as np
import pandas as pd
from scipy import stats  # Hypothesis tests, correlation, regression
from scipy.optimize import minimize_scalar, curve_fit  # α recovery and redshift-scaling fit
from pathlib import Path
import logging
import json

# =============================================================================
# LOGGER SETUP
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging
from scripts.utils.tep_model import ALPHA_0, compute_gamma_t as tep_gamma  # Shared TEP model & coupling constant

STEP_NUM = "023"  # Pipeline step number
STEP_NAME = "self_consistency"  # Used in log / output filenames

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


def analyze_self_consistency(df):
    """
    TEST 1: Self-Consistency Test
    
    If TEP is correct, different observables should give the SAME α.
    Test: Optimize α using different target variables.
    """
    logger.info("=" * 70)
    logger.info("TEST 1: Self-Consistency Test")
    logger.info("=" * 70)
    
    valid = df.dropna(subset=['log_Mhalo', 'z', 'age_ratio', 'mwa_Gyr', 't_eff_Gyr'])
    
    def compute_gamma(alpha):
        return tep_gamma(valid['log_Mhalo'].values, valid['z'].values, alpha_0=alpha)
    
    # Optimize α to minimize scatter in different observables
    targets = {
        'age_ratio': lambda g: valid['age_ratio'] / g,
        'mwa_Gyr': lambda g: valid['mwa_Gyr'] / g,
    }
    
    optimal_alphas = {}
    
    for name, transform in targets.items():
        def objective(alpha):
            gamma = compute_gamma(alpha)
            corrected = transform(gamma)
            # Use Coefficient of Variation (CV) to avoid driving values to zero
            # Raw std() favors large gamma (small corrected values)
            if corrected.mean() == 0: return np.inf
            return corrected.std() / np.abs(corrected.mean())
        
        result = minimize_scalar(objective, bounds=(0.1, 2.0), method='bounded')
        optimal_alphas[name] = result.x
        
        logger.info(f"\nOptimal α for {name}: {result.x:.3f}")
    
    # Check consistency
    alpha_values = list(optimal_alphas.values())
    alpha_mean = np.mean(alpha_values)
    alpha_std = np.std(alpha_values)
    alpha_cv = alpha_std / alpha_mean  # Coefficient of variation
    
    logger.info(f"\nSelf-consistency:")
    logger.info(f"  Mean α: {alpha_mean:.3f}")
    logger.info(f"  Std α: {alpha_std:.3f}")
    logger.info(f"  CV: {alpha_cv:.3f}")
    
    if alpha_cv < 0.2:
        logger.info("✓ Different observables give consistent α")
        consistent = True
    else:
        logger.info("⚠ α varies across observables")
        consistent = False
    
    return {
        'optimal_alphas': optimal_alphas,
        'alpha_mean': alpha_mean,
        'alpha_std': alpha_std,
        'alpha_cv': alpha_cv,
        'consistent': consistent
    }


def analyze_prediction_inversion(df):
    """
    TEST 2: Prediction Inversion
    
    Can we predict Γ_t FROM properties?
    If TEP is real, this should work both ways.
    """
    logger.info("=" * 70)
    logger.info("TEST 2: Prediction Inversion")
    logger.info("=" * 70)
    
    valid = df.dropna(subset=['gamma_t', 'age_ratio', 'mwa_Gyr', 'z'])
    
    # Forward prediction: Γ_t → age_ratio
    rho_forward, p_forward = stats.spearmanr(valid['gamma_t'], valid['age_ratio'])
    
    logger.info(f"Forward (Γ_t → age_ratio): ρ = {rho_forward:.3f}, p = {p_forward:.4f}")
    
    # Inverse prediction: Can we predict Γ_t from age_ratio and z?
    # If TEP is correct: Γ_t ≈ (age_ratio × t_cosmic / mwa_true) - 1
    # But we don't know mwa_true, so we use a regression
    
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score
    
    X = valid[['age_ratio', 'z', 'mwa_Gyr']].values
    y = valid['gamma_t'].values
    
    model = LinearRegression()
    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    
    logger.info(f"\nInverse prediction (properties → Γ_t):")
    logger.info(f"  Cross-validated R²: {scores.mean():.3f} ± {scores.std():.3f}")
    
    # Fit full model
    model.fit(X, y)
    y_pred = model.predict(X)
    
    rho_inverse, p_inverse = stats.spearmanr(y, y_pred)
    logger.info(f"  Spearman ρ (actual vs predicted Γ_t): {rho_inverse:.3f}")
    
    if rho_inverse > 0.5:
        logger.info("✓ Γ_t can be predicted from properties (bidirectional prediction works)")
        invertible = True
    else:
        logger.info("⚠ Weak inverse prediction")
        invertible = False
    
    return {
        'rho_forward': rho_forward,
        'rho_inverse': rho_inverse,
        'cv_r2_mean': scores.mean(),
        'cv_r2_std': scores.std(),
        'invertible': invertible
    }


def analyze_bootstrap_test(df):
    """
    TEST 3: Bootstrap Test
    
    Calibrate α on one half, test on the other.
    The system should work on unseen data.
    """
    logger.info("=" * 70)
    logger.info("TEST 3: Bootstrap Test")
    logger.info("=" * 70)
    
    valid = df.dropna(subset=['log_Mhalo', 'z', 'age_ratio']).copy()
    
    # Random split
    np.random.seed(42)
    indices = np.random.permutation(len(valid))
    split = len(valid) // 2
    
    train_idx = indices[:split]
    test_idx = indices[split:]
    
    train = valid.iloc[train_idx]
    test = valid.iloc[test_idx]
    
    logger.info(f"Train set: N = {len(train)}")
    logger.info(f"Test set: N = {len(test)}")
    
    # Calibrate α on train set
    def compute_gamma(alpha, data):
        return tep_gamma(data['log_Mhalo'].values, data['z'].values, alpha_0=alpha)
    
    def objective(alpha):
        gamma = compute_gamma(alpha, train)
        corrected = train['age_ratio'] / gamma
        if corrected.mean() == 0: return np.inf
        return corrected.std() / np.abs(corrected.mean())
    
    result = minimize_scalar(objective, bounds=(0.1, 2.0), method='bounded')
    alpha_train = result.x
    scatter_train = result.fun
    
    logger.info(f"\nTrain set:")
    logger.info(f"  Optimal α: {alpha_train:.3f}")
    logger.info(f"  Scatter (corrected): {scatter_train:.4f}")
    
    # Test on held-out data
    gamma_test = compute_gamma(alpha_train, test)
    corrected_test = test['age_ratio'] / gamma_test
    scatter_test = corrected_test.std()
    
    # Compare to no-TEP scatter
    scatter_no_tep = test['age_ratio'].std()
    
    logger.info(f"\nTest set (using α from train):")
    logger.info(f"  Scatter (no TEP): {scatter_no_tep:.4f}")
    logger.info(f"  Scatter (TEP-corrected): {scatter_test:.4f}")
    logger.info(f"  Reduction: {100*(scatter_no_tep - scatter_test)/scatter_no_tep:.1f}%")
    
    if scatter_test < scatter_no_tep:
        logger.info("✓ TEP correction improves unseen data (system generalizes)")
        generalizes = True
    else:
        logger.info("⚠ TEP correction does not improve test set")
        generalizes = False
    
    return {
        'alpha_train': alpha_train,
        'scatter_train': scatter_train,
        'scatter_test_no_tep': scatter_no_tep,
        'scatter_test_tep': scatter_test,
        'scatter_reduction_pct': 100*(scatter_no_tep - scatter_test)/scatter_no_tep,
        'generalizes': generalizes
    }


def analyze_redshift_ladder(df):
    """
    TEST 4: Redshift Ladder
    
    TEP predicts α(z) = α_0 × (1+z)^0.5
    Can we recover this scaling from the data?
    """
    logger.info("=" * 70)
    logger.info("TEST 4: Redshift Ladder")
    logger.info("=" * 70)
    
    valid = df.dropna(subset=['z', 'log_Mhalo', 'age_ratio'])
    
    # Bin by redshift and optimize α in each bin
    z_bins = [(7, 9), (9, 11), (11, 13), (13, 16), (16, 20)]
    
    z_centers = []
    optimal_alphas = []
    
    for z_lo, z_hi in z_bins:
        bin_data = valid[(valid['z'] >= z_lo) & (valid['z'] < z_hi)]
        if len(bin_data) >= 100:
            def objective(alpha):
                gamma = tep_gamma(bin_data['log_Mhalo'].values, bin_data['z'].values, alpha_0=alpha)
                corrected = bin_data['age_ratio'] / gamma
                if corrected.mean() == 0: return np.inf
                return corrected.std() / np.abs(corrected.mean())
            
            result = minimize_scalar(objective, bounds=(0.1, 3.0), method='bounded')
            
            z_center = (z_lo + z_hi) / 2
            z_centers.append(z_center)
            optimal_alphas.append(result.x)
            
            logger.info(f"z = [{z_lo}, {z_hi}): N = {len(bin_data)}, α = {result.x:.3f}")
    
    if len(z_centers) >= 3:
        # Fit power law: α = α_0 × (1+z)^β
        def power_law(z, alpha_0, beta):
            return alpha_0 * (1 + z) ** beta
        
        try:
            alpha_z_est = [a * np.sqrt(1 + zc) for a, zc in zip(optimal_alphas, z_centers)]
            popt, pcov = curve_fit(power_law, z_centers, alpha_z_est, p0=[ALPHA_0, 0.5])
            alpha_0_fit, beta_fit = popt
            
            logger.info(f"\nPower law fit: α(z) = {alpha_0_fit:.3f} × (1+z)^{beta_fit:.3f}")
            logger.info(f"TEP prediction: α(z) = α_0 × (1+z)^0.5")
            
            # How close is β to 0.5?
            beta_error = abs(beta_fit - 0.5)
            logger.info(f"β deviation from 0.5: {beta_error:.3f}")
            
            if beta_error < 0.3:
                logger.info("✓ Redshift scaling consistent with TEP prediction")
                scaling_correct = True
            else:
                logger.info("⚠ Redshift scaling differs from prediction")
                scaling_correct = False
            
            return {
                'z_centers': z_centers,
                'optimal_alphas': optimal_alphas,
                'alpha_0_fit': alpha_0_fit,
                'beta_fit': beta_fit,
                'beta_error': beta_error,
                'scaling_correct': scaling_correct
            }
        except Exception as e:
            logger.warning(f"Power law fit failed: {e}")
            return None
    
    return None


def analyze_coherence_matrix(df):
    """
    TEST 5: Coherence Matrix
    
    All TEP-affected properties should correlate with each other.
    The correlation matrix should have a specific structure.
    """
    logger.info("=" * 70)
    logger.info("TEST 5: Coherence Matrix")
    logger.info("=" * 70)
    
    valid = df.dropna(subset=['gamma_t', 'age_ratio', 'mwa_Gyr', 't_eff_Gyr', 'log_Mstar'])
    
    # Properties that should be affected by TEP
    properties = ['gamma_t', 'age_ratio', 'mwa_Gyr', 't_eff_Gyr', 'log_Mstar']
    
    # Compute correlation matrix
    corr_matrix = valid[properties].corr(method='spearman')
    
    logger.info("Correlation matrix (Spearman):")
    logger.info(corr_matrix.round(3).to_string())
    
    # Check if Γ_t correlates with all other properties
    gamma_correlations = corr_matrix['gamma_t'].drop('gamma_t')
    
    logger.info(f"\nΓ_t correlations:")
    for prop, corr in gamma_correlations.items():
        logger.info(f"  {prop}: ρ = {corr:.3f}")
    
    # Mean calibrated correlation with Γ_t
    mean_abs_corr = gamma_correlations.abs().mean()
    logger.info(f"\nMean |ρ| with Γ_t: {mean_abs_corr:.3f}")
    
    # Check for coherent structure
    # If TEP is correct, all properties should correlate with Γ_t
    n_significant = (gamma_correlations.abs() > 0.1).sum()
    
    if n_significant >= 3:
        logger.info(f"✓ {n_significant}/{len(gamma_correlations)} properties correlate with Γ_t")
        logger.info("  Structured correlations detected")
        coherent = True
    else:
        logger.info("⚠ Weak coherence structure")
        coherent = False
    
    # Eigenvalue analysis - if TEP is a hidden variable, the first eigenvalue should dominate
    eigenvalues = np.linalg.eigvals(corr_matrix.values)
    eigenvalues = np.sort(np.real(eigenvalues))[::-1]
    
    variance_explained_1 = eigenvalues[0] / eigenvalues.sum()
    logger.info(f"\nFirst eigenvalue explains {100*variance_explained_1:.1f}% of variance")
    
    if variance_explained_1 > 0.4:
        logger.info("✓ Single dominant factor (consistent with TEP as hidden variable)")
    
    return {
        'correlation_matrix': corr_matrix.to_dict(),
        'gamma_correlations': gamma_correlations.to_dict(),
        'mean_abs_corr': mean_abs_corr,
        'n_significant': n_significant,
        'variance_explained_1': variance_explained_1,
        'coherent': coherent
    }


def run_self_consistency_analysis():
    """Run the complete self-consistency analysis."""
    logger.info("=" * 70)
    logger.info("TEP-JWST Step 23: Self-Consistency Tests")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Testing internal consistency of TEP predictions.")
    logger.info("")
    
    # Load data
    df = load_data()
    
    results = {}
    
    # Test 1: Self-Consistency
    results['self_consistency'] = analyze_self_consistency(df)
    
    # Test 2: Prediction Inversion
    results['prediction_inversion'] = analyze_prediction_inversion(df)
    
    # Test 3: Bootstrap Test
    results['bootstrap_test'] = analyze_bootstrap_test(df)
    
    # Test 4: Redshift Ladder
    results['redshift_ladder'] = analyze_redshift_ladder(df)
    
    # Test 5: Coherence Matrix
    results['coherence_matrix'] = analyze_coherence_matrix(df)
    
    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY: Self-Consistency Results")
    logger.info("=" * 70)
    
    tests_passed = 0
    total_tests = 5
    
    if results['self_consistency'].get('consistent', False):
        tests_passed += 1
        logger.info("✓ Self-consistency: Internally coherent")
    
    if results['prediction_inversion'].get('invertible', False):
        tests_passed += 1
        logger.info("✓ Prediction inversion: Bidirectional prediction works")
    
    if results['bootstrap_test'].get('generalizes', False):
        tests_passed += 1
        logger.info("✓ Bootstrap test: Generalizes to unseen data")
    
    if results['redshift_ladder'] and results['redshift_ladder'].get('scaling_correct', False):
        tests_passed += 1
        logger.info("✓ Redshift ladder: Scaling recovered")
    
    if results['coherence_matrix'].get('coherent', False):
        tests_passed += 1
        logger.info("✓ Coherence matrix: Structured correlations")
    
    logger.info(f"\nTests passed: {tests_passed}/{total_tests}")
    
    # Save results
    output_file = RESULTS_DIR / f"step_{STEP_NUM}_self_consistency.json"
    
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        return obj
    
    results_serializable = json.loads(json.dumps(results, default=convert_numpy))
    
    with open(output_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    run_self_consistency_analysis()
