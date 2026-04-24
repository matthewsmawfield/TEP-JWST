#!/usr/bin/env python3
# Estimated runtime from last full canonical run (2026-03-09 15:52 UTC; full pipeline 32m18s): 0.8s.
"""
TEP-JWST Step 26: Prediction-Observation Alignment

This step tests whether TEP predictions align with observations using
independent quantities (not tautological comparisons).

Key tests:
1. Convergence: Do multiple methods yield consistent α values?
2. Statistical significance: Is the combined evidence significant?
3. Prediction alignment: Does TEP correction reduce mass dependence?

Author: Matthew L. Smawfield
Date: January 2026
"""

import sys
import numpy as np
import pandas as pd
from scipy import stats  # Hypothesis tests and correlation
from scipy.optimize import minimize_scalar  # 1-D bounded optimisation for α convergence test
from pathlib import Path
import logging
import json

# =============================================================================
# LOGGER SETUP
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging (severity levels: DEBUG/INFO/WARNING/ERROR/SUCCESS)
from scripts.utils.p_value_utils import format_p_value  # Safe p-value formatting (prevents floating-point underflow at p < 1e-300)
from scripts.utils.tep_model import ALPHA_0, ALPHA_CLOCK_EFF, compute_gamma_t as tep_gamma  # TEP model: alpha_eff=9.6e5 mag from Cepheids (alpha_0=0.58 legacy), Gamma_t formula

STEP_NUM = "026"  # Pipeline step number (sequential 001-176)
STEP_NAME = "prediction_alignment"  # Tests TEP prediction-observation alignment via convergence, significance, and correction tests

LOGS_PATH = PROJECT_ROOT / "logs"  # Log directory (one plain-text file per step for debugging traceability)
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"  # JSON output directory (machine-readable statistical results)
LOGS_PATH.mkdir(parents=True, exist_ok=True)  # Create logs/ if missing; parents=True ensures full path tree exists
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)  # Create results/outputs/ if missing

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")  # Step-specific logger (isolated per-step logging for traceability)
set_step_logger(logger)  # Register as global step logger so print_status() routes to this step's log

# Additional path aliases used by downstream helpers
DATA_DIR = PROJECT_ROOT / "data"  # Raw catalogue directory (external datasets: UNCOVER, CEERS, COSMOS-Web, JADES)
INTERIM_DIR = PROJECT_ROOT / "results" / "interim"  # Pre-processed intermediate products (CSV format for step-to-step data flow)
RESULTS_DIR = PROJECT_ROOT / "results" / "outputs"  # Final JSON outputs (machine-readable statistical results, alias for OUTPUT_PATH)


def load_data():
    logger.info("Loading data...")
    uncover = pd.read_csv(INTERIM_DIR / "step_002_uncover_full_sample_tep.csv")
    uncover = uncover.rename(
        columns={
            'z_phot': 'z',
            'mwa': 'mwa_Gyr',
            't_cosmic': 't_cosmic_Gyr',
            't_eff': 't_eff_Gyr',
            'log_Mh': 'log_Mhalo',
        }
    )

    if 't_assembly_Myr' not in uncover.columns:
        legacy_path = DATA_DIR / "interim" / "uncover_highz_sed_properties.csv"
        if legacy_path.exists():
            legacy = pd.read_csv(legacy_path)
            legacy_asm = legacy[['id', 't_assembly_Myr']].drop_duplicates(subset=['id'])
            uncover = uncover.merge(legacy_asm, on='id', how='left')

    uncover['age_ratio'] = uncover['mwa_Gyr'] / uncover['t_cosmic_Gyr']
    
    _jades_path = DATA_DIR / "interim" / "jades_highz_physical.csv"
    if not Path(_jades_path).exists():
        print_status("ERROR: jades_highz_physical.csv not found. Run step_014 first.", "ERROR")
        return None, None
    jades = pd.read_csv(_jades_path)
    jades['gamma_t'] = tep_gamma(jades['log_Mhalo'].values, jades['z_best'].values, alpha_0=ALPHA_0)
    jades['age_ratio'] = jades['t_stellar_Gyr'] / jades['t_cosmic_Gyr']
    
    return uncover, jades


def convergence_test(df):
    """Multiple methods converge on the same α."""
    logger.info("=" * 70)
    logger.info("TEST 1: Convergence Test")
    logger.info("=" * 70)
    
    valid = df.dropna(subset=['gamma_t', 'age_ratio', 'log_Mhalo', 'z', 'log_Mstar'])
    
    # Method 1: Scatter minimization
    def scatter_obj(alpha):
        gamma = tep_gamma(valid['log_Mhalo'].values, valid['z'].values, alpha_0=alpha)
        return (valid['age_ratio'] / gamma).std()
    
    r1 = minimize_scalar(scatter_obj, bounds=(0.1, 3.0), method='bounded')
    alpha_1 = r1.x
    
    # Method 2: Correlation maximization
    def corr_obj(alpha):
        gamma = tep_gamma(valid['log_Mhalo'].values, valid['z'].values, alpha_0=alpha)
        corrected = valid['age_ratio'] / gamma
        rho, _ = stats.spearmanr(valid['log_Mstar'], corrected)
        return -abs(rho)
    
    r2 = minimize_scalar(corr_obj, bounds=(0.1, 3.0), method='bounded')
    alpha_2 = r2.x
    
    logger.info(f"Method 1 (scatter min): α = {alpha_1:.3f}")
    logger.info(f"Method 2 (corr max): α = {alpha_2:.3f}")
    logger.info(f"Calibrated: α = {ALPHA_0:.3f}")
    
    # Check convergence
    alphas = [alpha_1, alpha_2, ALPHA_0]
    mean_alpha = np.mean(alphas)
    std_alpha = np.std(alphas)
    cv = std_alpha / mean_alpha
    
    logger.info(f"\nConvergence: CV = {cv:.3f}")
    
    converged = cv < 0.5
    if converged:
        logger.info("✓ Methods converge")
    
    return {'alpha_1': alpha_1, 'alpha_2': alpha_2, 'cv': cv, 'converged': converged}


def null_rejection_test(df):
    """Test the null model."""
    logger.info("=" * 70)
    logger.info("TEST 2: null model Test")
    logger.info("=" * 70)
    
    subset_cols = ['gamma_t', 'age_ratio', 't_eff_Gyr']
    if 't_assembly_Myr' in df.columns:
        subset_cols.append('t_assembly_Myr')
    valid = df.dropna(subset=subset_cols)

    # Collect p-values from multiple tests
    p_values = []

    # Test 1: Γ_t vs t_eff
    _, p1 = stats.spearmanr(valid['gamma_t'], valid['t_eff_Gyr'])
    p_values.append(('t_eff', p1))

    # Test 2: Γ_t vs t_assembly (if available)
    if 't_assembly_Myr' in valid.columns:
        _, p2 = stats.spearmanr(valid['gamma_t'], valid['t_assembly_Myr'])
        p_values.append(('t_assembly', p2))
    
    # Test 3: Scatter reduction
    scatter_raw = valid['age_ratio'].std()
    scatter_corr = (valid['age_ratio'] / valid['gamma_t']).std()
    # Bootstrap for p-value
    n_better = 0
    for _ in range(1000):
        perm = np.random.permutation(valid['gamma_t'].values)
        scatter_perm = (valid['age_ratio'] / perm).std()
        if scatter_perm <= scatter_corr:
            n_better += 1
    p3 = n_better / 1000
    p_values.append(('scatter', max(p3, 1e-10)))
    
    # Test 4: Anomalous galaxies
    anomalous = valid[valid['age_ratio'] > 0.5]
    normal = valid[valid['age_ratio'] <= 0.5]
    if len(anomalous) > 3 and len(normal) > 3:
        _, p4 = stats.ttest_ind(anomalous['gamma_t'], normal['gamma_t'])
        p_values.append(('anomalous', p4))
    
    logger.info("Individual p-values:")
    for name, p in p_values:
        logger.info(f"  {name}: p = {p:.2e}")
    
    # Fisher's combined test
    chi2 = -2 * sum(np.log(max(p, 1e-300)) for _, p in p_values)
    df_fisher = 2 * len(p_values)
    combined_p_raw = stats.chi2.sf(chi2, df_fisher)
    combined_p = format_p_value(combined_p_raw)
    
    if combined_p is None:
        logger.info(f"\nFisher's combined: χ² = {chi2:.1f}, p = N/A")
    else:
        logger.info(f"\nFisher's combined: χ² = {chi2:.1f}, p = {combined_p:.2e}")
    
    # Convert to sigma
    sigma = None
    if combined_p is not None and 0 < combined_p < 1:
        sigma = float(stats.norm.isf(combined_p / 2))
    
    if sigma is None:
        logger.info("Equivalent: N/A")
    else:
        logger.info(f"Equivalent: {sigma:.1f}σ")
    
    disrupted = bool(sigma is not None and sigma > 5)
    if disrupted:
        logger.info("✓ null model disrupted")
    
    return {'chi2': chi2, 'combined_p': combined_p, 'sigma': sigma, 'disrupted': disrupted}


def totality_test(df):
    """
    Test prediction-observation alignment using INDEPENDENT quantities.
    
    TEP predicts: age_ratio_corrected = age_ratio / Γ_t should be
    INDEPENDENT of mass (removing the TEP-induced mass dependence).
    
    We test this by comparing:
    - Predicted: The mass-dependence should vanish after TEP correction
    - Observed: The actual correlation between corrected age ratio and mass
    """
    logger.info("=" * 70)
    logger.info("TEST 3: Prediction-Observation Alignment")
    logger.info("=" * 70)
    
    valid = df.dropna(subset=['gamma_t', 'age_ratio', 'log_Mstar'])
    valid = valid.copy()
    
    # TEP prediction: After correction, age_ratio should be INDEPENDENT of mass
    # Raw correlation (before correction)
    rho_raw, p_raw = stats.spearmanr(valid['log_Mstar'], valid['age_ratio'])
    logger.info(f"Raw: ρ(M*, age_ratio) = {rho_raw:.4f}, p = {p_raw:.2e}")
    
    # Corrected correlation (after TEP)
    valid['age_ratio_corrected'] = valid['age_ratio'] / valid['gamma_t']
    rho_corr, p_corr = stats.spearmanr(valid['log_Mstar'], valid['age_ratio_corrected'])
    logger.info(f"Corrected: ρ(M*, age_ratio_corr) = {rho_corr:.4f}, p = {p_corr:.2e}")
    
    # The prediction: |rho_corr| < |rho_raw| (TEP removes mass dependence)
    improvement = abs(rho_raw) - abs(rho_corr)
    logger.info(f"\nImprovement: Δ|ρ| = {improvement:.4f}")
    
    # Scatter reduction
    scatter_raw = valid['age_ratio'].std()
    scatter_corr = valid['age_ratio_corrected'].std()
    scatter_reduction = (scatter_raw - scatter_corr) / scatter_raw * 100
    logger.info(f"Scatter reduction: {scatter_reduction:.1f}%")
    
    # Test by mass bin: Does correction flatten the mass dependence?
    mass_bins = [(8, 9), (9, 10), (10, 11)]
    
    raw_means = []
    corr_means = []
    
    for m_lo, m_hi in mass_bins:
        bin_data = valid[(valid['log_Mstar'] >= m_lo) & (valid['log_Mstar'] < m_hi)]
        if len(bin_data) >= 30:
            raw_means.append(bin_data['age_ratio'].mean())
            corr_means.append(bin_data['age_ratio_corrected'].mean())
            logger.info(f"log(M*) = [{m_lo}, {m_hi}): raw = {raw_means[-1]:.4f}, corr = {corr_means[-1]:.4f}")
    
    # Measure flattening: std of bin means should decrease
    if len(raw_means) >= 3:
        spread_raw = np.std(raw_means)
        spread_corr = np.std(corr_means)
        flattening = (spread_raw - spread_corr) / spread_raw * 100
        logger.info(f"\nBin spread reduction: {flattening:.1f}%")
        
        # Success if correlation weakens and spread reduces
        success = improvement > 0 and flattening > 0
    else:
        flattening = 0
        success = False
    
    if success:
        logger.info("\n✓ TEP correction reduces mass dependence as predicted")
    else:
        logger.info("\n⚠ TEP correction does not fully remove mass dependence")
    
    return {
        'rho_raw': float(rho_raw),
        'rho_corrected': float(rho_corr),
        'improvement': float(improvement),
        'scatter_reduction_pct': float(scatter_reduction),
        'flattening_pct': float(flattening),
        'success': bool(success)
    }


def run_totality_analysis():
    logger.info("=" * 70)
    logger.info(f"TEP-JWST Step {STEP_NUM}: Prediction-Observation Alignment")
    logger.info("=" * 70)
    logger.info("")
    
    uncover, jades = load_data()
    if jades is None:
        logger.error(f"Step {STEP_NUM} aborted: jades_highz_physical.csv not found. Run step_014 first.")
        return {"status": "aborted", "reason": "missing jades_highz_physical.csv"}

    results = {}
    results['convergence'] = convergence_test(uncover)
    results['null_rejection'] = null_rejection_test(uncover)
    results['totality'] = totality_test(uncover)
    
    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    
    tests_passed = 0
    if results['convergence'].get('converged', False):
        tests_passed += 1
        logger.info("✓ Convergence: Multiple methods yield consistent α")
    if results['null_rejection'].get('disrupted', False):
        tests_passed += 1
        logger.info("✓ Statistical significance: Combined p < 0.05")
    if results['totality'].get('success', False):
        tests_passed += 1
        logger.info("✓ Prediction alignment: TEP reduces mass dependence")
    
    logger.info(f"\nTests passed: {tests_passed}/3")
    
    # Save
    output_file = RESULTS_DIR / f"step_{STEP_NUM}_prediction_alignment.json"
    
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
        return str(obj)
    
    results_clean = {}
    for k, v in results.items():
        results_clean[k] = {kk: convert(vv) for kk, vv in v.items()}
    
    with open(output_file, 'w') as f:
        json.dump(results_clean, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_file}")
    return results


if __name__ == "__main__":
    run_totality_analysis()
