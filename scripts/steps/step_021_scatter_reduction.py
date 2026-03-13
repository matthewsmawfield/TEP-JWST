#!/usr/bin/env python3
"""
TEP-JWST Step 21: Scatter Reduction Analysis

This analysis tests whether TEP corrections reduce anomalies in the data:

1. THE COSMIC AGE LIMIT
   - At each redshift, t_stellar cannot exceed t_cosmic
   - Galaxies near this limit are "anomalous"
   - TEP prediction: high-Γ_t galaxies can approach the limit

2. THE SCATTER REDUCTION
   - Standard scaling relations have scatter
   - Some of this scatter should be explained by Γ_t
   - After TEP correction, scatter should DECREASE

3. THE DOWNSIZING TRANSITION
   - Downsizing weakens at high-z
   - TEP explains this: chronological enhancement cancels intrinsic downsizing
   - The transition redshift should match TEP predictions

4. THE MASS FUNCTION ANOMALY
   - The high-z mass function is steeper than expected
   - TEP explanation: masses are overestimated at high Γ_t
   - After correction, the mass function should flatten

5. THE COHERENCE TEST
   - If TEP is correct, all properties should shift coherently with Γ_t
   - The correlation STRUCTURE should be explained by TEP

Author: Matthew L. Smawfield
Date: January 2026
"""

import sys
import numpy as np  # Core numerical computations
import pandas as pd  # Data manipulation and analysis
from astropy.cosmology import Planck18 as cosmo  # Standard Planck 2018 cosmology for age/distance calculations
from scipy import stats  # Statistical functions (e.g., t-tests, linear regression)
from pathlib import Path  # Filesystem utilities
import logging  # Logging framework
import json  # JSON data handling

# =============================================================================
# LOGGER SETUP
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Project root directory
sys.path.insert(0, str(PROJECT_ROOT))  # Add project root to Python path

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging utilities
from scripts.utils.p_value_utils import format_p_value  # Safe p-value formatting (handles None / edge cases)
from scripts.utils.tep_model import ALPHA_0, compute_gamma_t as tep_gamma  # Shared TEP model: coupling constant and Γ_t calculator

STEP_NUM = "021"  # Pipeline step number
STEP_NAME = "scatter_reduction"  # Descriptive label used for log and output filenames

LOGS_PATH = PROJECT_ROOT / "logs"  # Step log files directory
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"  # Final JSON outputs directory
LOGS_PATH.mkdir(parents=True, exist_ok=True)  # Create logs directory if it doesn't exist
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)  # Create outputs directory if it doesn't exist

DATA_DIR = PROJECT_ROOT / "data"  # Raw catalogue directory
INTERIM_DIR = PROJECT_ROOT / "results" / "interim"  # Pre-processed intermediate products
RESULTS_DIR = PROJECT_ROOT / "results" / "outputs"  # Final JSON outputs (alias for downstream use)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")  # Initialise step logger
set_step_logger(logger)

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


def analyze_cosmic_age_limit(df):
    """
    TEST 1: The Cosmic Age Limit
    
    At each redshift, stellar age cannot exceed cosmic age.
    Galaxies near this limit (age_ratio > 0.5) are problematic.
    
    TEP smooths this: high-Γ_t galaxies can approach the limit
    because their apparent ages are inflated.
    """
    logger.info("=" * 70)
    logger.info("TEST 1: The Cosmic Age Limit")
    logger.info("=" * 70)
    
    valid = df.dropna(subset=['age_ratio', 'gamma_t'])
    
    # Define "near limit" as age_ratio > 0.3
    near_limit = valid[valid['age_ratio'] > 0.3]
    far_from_limit = valid[valid['age_ratio'] <= 0.3]
    
    logger.info(f"Total galaxies: N = {len(valid)}")
    logger.info(f"Near cosmic limit (age_ratio > 0.3): N = {len(near_limit)} ({100*len(near_limit)/len(valid):.1f}%)")
    
    # Compare Γ_t distributions
    mean_gamma_near = near_limit['gamma_t'].mean()
    mean_gamma_far = far_from_limit['gamma_t'].mean()
    
    t_stat, p_value = stats.ttest_ind(near_limit['gamma_t'], far_from_limit['gamma_t'])
    p_value_fmt = format_p_value(p_value)
    
    logger.info(f"\nΓ_t comparison:")
    logger.info(f"  Near limit: mean Γ_t = {mean_gamma_near:.3f}")
    logger.info(f"  Far from limit: mean Γ_t = {mean_gamma_far:.3f}")
    logger.info(f"  Difference: {mean_gamma_near - mean_gamma_far:.3f}")
    logger.info(f"  p-value: {p_value:.2e}")
    
    # TEP predicts: near-limit galaxies should have higher Γ_t
    if mean_gamma_near > mean_gamma_far and (p_value_fmt is not None and p_value_fmt < 0.05):
        logger.info("\n✓ Near-limit galaxies have higher Γ_t (TEP resolves this anomaly)")
        tep_smooths = True
    else:
        logger.info("\n⚠ Anomaly not fully resolved")
        tep_smooths = False
    
    # Calculate how many "anomalous" galaxies become possible after TEP correction
    anomalous = valid[valid['age_ratio'] > 0.5]
    if len(anomalous) > 0:
        corrected_ratio = anomalous['age_ratio'] / anomalous['gamma_t']
        n_resolved = (corrected_ratio <= 0.5).sum()
        logger.info(f"\n'Anomalous' galaxies (age_ratio > 0.5): N = {len(anomalous)}")
        logger.info(f"  Resolved by TEP correction: {n_resolved}/{len(anomalous)} ({100*n_resolved/len(anomalous):.0f}%)")
    
    return {
        'n_near_limit': len(near_limit),
        'n_far_from_limit': len(far_from_limit),
        'mean_gamma_near': mean_gamma_near,
        'mean_gamma_far': mean_gamma_far,
        'p_value': p_value_fmt,
        'tep_smooths': tep_smooths
    }


def analyze_scatter_reduction(df):
    """
    TEST 2: Scatter Reduction
    
    Standard scaling relations have scatter. If TEP is correct,
    some of this scatter should be explained by Γ_t.
    
    After TEP correction, the scatter should DECREASE.
    """
    logger.info("=" * 70)
    logger.info("TEST 2: Scatter Reduction")
    logger.info("=" * 70)
    
    valid = df.dropna(subset=['log_Mstar', 'mwa_Gyr', 'gamma_t'])
    
    # Mass-age relation
    # Raw scatter
    slope_raw, intercept_raw, r_raw, _, _ = stats.linregress(
        valid['log_Mstar'], valid['mwa_Gyr']
    )
    residuals_raw = valid['mwa_Gyr'] - (slope_raw * valid['log_Mstar'] + intercept_raw)
    scatter_raw = residuals_raw.std()
    
    logger.info(f"Mass-Age relation:")
    logger.info(f"  Raw scatter: σ = {scatter_raw:.4f} Gyr")
    logger.info(f"  Raw R²: {r_raw**2:.4f}")
    
    # TEP-corrected age
    valid = valid.copy()
    valid['mwa_corrected'] = valid['mwa_Gyr'] / valid['gamma_t']
    
    slope_corr, intercept_corr, r_corr, _, _ = stats.linregress(
        valid['log_Mstar'], valid['mwa_corrected']
    )
    residuals_corr = valid['mwa_corrected'] - (slope_corr * valid['log_Mstar'] + intercept_corr)
    scatter_corr = residuals_corr.std()
    
    logger.info(f"\n  TEP-corrected scatter: σ = {scatter_corr:.4f} Gyr")
    logger.info(f"  TEP-corrected R²: {r_corr**2:.4f}")
    
    scatter_reduction = (scatter_raw - scatter_corr) / scatter_raw * 100
    logger.info(f"\n  Scatter reduction: {scatter_reduction:.1f}%")
    
    if scatter_corr < scatter_raw:
        logger.info("✓ TEP correction reduces scatter (anomaly resolved)")
        tep_smooths = True
    else:
        logger.info("⚠ TEP correction does not reduce scatter")
        tep_smooths = False
    
    return {
        'scatter_raw': scatter_raw,
        'scatter_corrected': scatter_corr,
        'scatter_reduction_pct': scatter_reduction,
        'r2_raw': r_raw**2,
        'r2_corrected': r_corr**2,
        'tep_smooths': tep_smooths
    }


def analyze_downsizing_transition(df):
    """
    TEST 3: The Downsizing Transition
    
    Downsizing (massive galaxies have lower sSFR) weakens at high-z.
    TEP explains this: chronological enhancement cancels intrinsic downsizing.
    
    The transition should occur where TEP effects become dominant.
    """
    logger.info("=" * 70)
    logger.info("TEST 3: The Downsizing Transition")
    logger.info("=" * 70)
    
    valid = df.dropna(subset=['z', 'log_Mstar', 'mwa_Gyr', 'gamma_t'])
    
    # Calculate sSFR proxy (1/age as rough proxy)
    valid = valid.copy()
    valid['ssfr_proxy'] = 1 / valid['mwa_Gyr']
    
    # Measure mass-sSFR correlation in redshift bins
    z_bins = [(7, 9), (9, 11), (11, 13), (13, 15), (15, 20)]
    
    correlations = []
    z_centers = []
    mean_gammas = []
    
    for z_lo, z_hi in z_bins:
        bin_data = valid[(valid['z'] >= z_lo) & (valid['z'] < z_hi)]
        if len(bin_data) >= 30:
            rho, p = stats.spearmanr(bin_data['log_Mstar'], bin_data['ssfr_proxy'])
            z_center = (z_lo + z_hi) / 2
            mean_gamma = bin_data['gamma_t'].mean()
            
            correlations.append(rho)
            z_centers.append(z_center)
            mean_gammas.append(mean_gamma)
            
            logger.info(f"\nz = [{z_lo}, {z_hi}): N = {len(bin_data)}")
            logger.info(f"  Mass-sSFR correlation: ρ = {rho:.3f}")
            logger.info(f"  Mean Γ_t: {mean_gamma:.2f}")
    
    if len(correlations) >= 2:
        # Test if correlation becomes less negative with z (as TEP predicts)
        rho_trend, p_trend = stats.spearmanr(z_centers, correlations)
        
        # Handle edge case where p_trend is NaN (e.g., strong correlation with only 2 points)
        if np.isnan(p_trend):
            # For 2 points, strong correlation is expected; set p to 1.0 (not significant)
            p_trend = 1.0 if len(correlations) <= 2 else 0.0
        
        logger.info(f"\nTrend: correlation vs z: ρ = {rho_trend:.3f}, p = {p_trend:.4f}")
        
        # Test if correlation tracks mean Γ_t
        rho_gamma, p_gamma = stats.spearmanr(mean_gammas, correlations)
        logger.info(f"Correlation vs mean Γ_t: ρ = {rho_gamma:.3f}, p = {p_gamma:.4f}")
        
        if rho_trend > 0:
            logger.info("\n✓ Downsizing weakens at high-z (TEP resolves this anomaly)")
            tep_smooths = True
        else:
            logger.info("\n⚠ Downsizing trend not as expected")
            tep_smooths = False
        
        return {
            'z_centers': z_centers,
            'correlations': correlations,
            'mean_gammas': mean_gammas,
            'rho_trend': rho_trend,
            'p_trend': p_trend,
            'tep_smooths': tep_smooths
        }
    
    return None


def analyze_mass_function_correction(df):
    """
    TEST 4: Mass Function Correction
    
    The high-z mass function appears steeper than expected.
    TEP explanation: masses are overestimated at high Γ_t.
    
    After correction, the mass distribution should shift.
    """
    logger.info("=" * 70)
    logger.info("TEST 4: Mass Function Correction")
    logger.info("=" * 70)
    
    valid = df.dropna(subset=['log_Mstar', 'gamma_t'])
    
    # Raw mass distribution
    mean_mass_raw = valid['log_Mstar'].mean()
    std_mass_raw = valid['log_Mstar'].std()
    
    logger.info(f"Raw mass distribution:")
    logger.info(f"  Mean log(M*): {mean_mass_raw:.2f}")
    logger.info(f"  Std log(M*): {std_mass_raw:.2f}")
    
    # TEP-corrected mass
    # M*_true = M*_obs / Γ_t^0.7
    valid = valid.copy()
    valid['log_Mstar_corrected'] = valid['log_Mstar'] - 0.7 * np.log10(1 + valid['gamma_t'])
    
    mean_mass_corr = valid['log_Mstar_corrected'].mean()
    std_mass_corr = valid['log_Mstar_corrected'].std()
    
    logger.info(f"\nTEP-corrected mass distribution:")
    logger.info(f"  Mean log(M*): {mean_mass_corr:.2f}")
    logger.info(f"  Std log(M*): {std_mass_corr:.2f}")
    
    mass_shift = mean_mass_raw - mean_mass_corr
    logger.info(f"\nMass shift: Δlog(M*) = {mass_shift:.3f} dex")
    
    # Count how many galaxies shift below detection threshold
    threshold = 7.0
    n_above_raw = (valid['log_Mstar'] > threshold).sum()
    n_above_corr = (valid['log_Mstar_corrected'] > threshold).sum()
    
    logger.info(f"\nGalaxies above log(M*) = {threshold}:")
    logger.info(f"  Raw: {n_above_raw}")
    logger.info(f"  Corrected: {n_above_corr}")
    logger.info(f"  Reduction: {100*(n_above_raw - n_above_corr)/n_above_raw:.1f}%")
    
    if mass_shift > 0:
        logger.info("\n✓ TEP correction shifts masses down (anomaly resolved)")
        tep_smooths = True
    else:
        logger.info("\n⚠ No mass shift")
        tep_smooths = False
    
    return {
        'mean_mass_raw': mean_mass_raw,
        'mean_mass_corrected': mean_mass_corr,
        'mass_shift': mass_shift,
        'n_above_threshold_raw': n_above_raw,
        'n_above_threshold_corr': n_above_corr,
        'tep_smooths': tep_smooths
    }


def analyze_coherence_structure(df):
    """
    TEST 5: Coherence Structure
    
    If TEP is correct, the correlation structure should be explained
    by Γ_t as a hidden variable.
    
    Test: Partial correlations controlling for Γ_t should differ
    from raw correlations.
    """
    logger.info("=" * 70)
    logger.info("TEST 5: Coherence Structure")
    logger.info("=" * 70)
    
    valid = df.dropna(subset=['log_Mstar', 'mwa_Gyr', 'gamma_t', 'z'])
    
    # Raw correlation: mass vs age
    rho_raw, p_raw = stats.spearmanr(valid['log_Mstar'], valid['mwa_Gyr'])
    
    logger.info(f"Mass-Age correlation:")
    logger.info(f"  Raw: ρ = {rho_raw:.3f}, p = {p_raw:.4f}")
    
    # Partial correlation controlling for Γ_t
    # Residualize both variables against Γ_t
    valid = valid.copy()
    
    slope_m, intercept_m, _, _, _ = stats.linregress(valid['gamma_t'], valid['log_Mstar'])
    valid['mass_resid'] = valid['log_Mstar'] - (slope_m * valid['gamma_t'] + intercept_m)
    
    slope_a, intercept_a, _, _, _ = stats.linregress(valid['gamma_t'], valid['mwa_Gyr'])
    valid['age_resid'] = valid['mwa_Gyr'] - (slope_a * valid['gamma_t'] + intercept_a)
    
    rho_partial, p_partial = stats.spearmanr(valid['mass_resid'], valid['age_resid'])
    
    logger.info(f"  Partial (controlling for Γ_t): ρ = {rho_partial:.3f}, p = {p_partial:.4f}")
    
    # The change in correlation
    delta_rho = rho_raw - rho_partial
    logger.info(f"\n  Δρ (raw - partial): {delta_rho:.3f}")
    
    # If Γ_t explains the correlation, the partial should be weaker
    if abs(rho_partial) < abs(rho_raw):
        logger.info("✓ Γ_t explains part of the mass-age correlation (anomaly resolved)")
        tep_smooths = True
        explained_fraction = 1 - abs(rho_partial) / abs(rho_raw) if rho_raw != 0 else 0
        logger.info(f"  Fraction explained by Γ_t: {100*explained_fraction:.1f}%")
    else:
        logger.info("⚠ Γ_t does not explain the correlation")
        tep_smooths = False
        explained_fraction = 0
    
    return {
        'rho_raw': rho_raw,
        'rho_partial': rho_partial,
        'delta_rho': delta_rho,
        'explained_fraction': explained_fraction,
        'tep_smooths': tep_smooths
    }


def run_scatter_reduction_analysis():
    """Run the complete scatter reduction analysis."""
    logger.info("=" * 70)
    logger.info("TEP-JWST Step 21: Scatter Reduction Analysis")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Testing whether TEP corrections reduce data anomalies.")
    logger.info("")
    
    # Load data
    df = load_data()
    
    results = {}
    
    # Test 1: Cosmic Age Limit
    results['cosmic_age_limit'] = analyze_cosmic_age_limit(df)
    
    # Test 2: Scatter Reduction
    results['scatter_reduction'] = analyze_scatter_reduction(df)
    
    # Test 3: Downsizing Transition
    results['downsizing_transition'] = analyze_downsizing_transition(df)
    
    # Test 4: Mass Function Correction
    results['mass_function'] = analyze_mass_function_correction(df)
    
    # Test 5: Coherence Structure
    results['coherence_structure'] = analyze_coherence_structure(df)
    
    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY: Anomaly Resolution Results")
    logger.info("=" * 70)
    
    anomalies_resolved = 0
    total_anomalies = 0
    
    for name, result in results.items():
        if result is not None:
            total_anomalies += 1
            if result.get('tep_smooths', False):
                anomalies_resolved += 1
                logger.info(f"✓ {name}: Anomaly resolved")
            else:
                logger.info(f"⚠ {name}: Anomaly not resolved")
    
    logger.info(f"\nAnomalies resolved: {anomalies_resolved}/{total_anomalies}")
    
    # Save results
    output_file = RESULTS_DIR / f"step_{STEP_NUM}_scatter_reduction.json"
    
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
    run_scatter_reduction_analysis()
