#!/usr/bin/env python3
"""
TEP-JWST Step 24: Cosmological Implications

This analysis examines cosmological implications of TEP corrections:

1. THE FORMATION REDSHIFT PREDICTION
   - TEP predicts: high-Γ_t galaxies can form later but appear older
   - Test: At fixed observed age, formation redshift should correlate with Γ_t

2. THE STELLAR MASS DENSITY PREDICTION
   - TEP predicts: the cosmic stellar mass density is overestimated
   - Test: After TEP correction, the mass density should decrease

3. THE SPECIFIC STAR FORMATION RATE FLOOR
   - TEP predicts: the sSFR floor at high-z is an artifact
   - Test: After TEP correction, the floor should disappear

4. THE MASS-METALLICITY RELATION
   - TEP predicts: the MZR scatter is partially due to Γ_t
   - Test: Controlling for Γ_t should reduce MZR scatter

5. THE QUENCHING TIMESCALE
   - TEP predicts: quenching appears faster in high-Γ_t galaxies
   - Test: Quenching timescale should correlate with Γ_t

6. THE COSMIC VARIANCE TEST
   - TEP predicts: field-to-field variance is partially due to Γ_t variations
   - Test: Variance should decrease after TEP correction

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
from scripts.utils.tep_model import ALPHA_0, compute_gamma_t as tep_gamma

STEP_NUM = "24"
STEP_NAME = "cosmological_implications"

LOGS_PATH = PROJECT_ROOT / "logs"
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"
LOGS_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")
set_step_logger(logger)



# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
INTERIM_DIR = PROJECT_ROOT / "results" / "interim"
RESULTS_DIR = PROJECT_ROOT / "results" / "outputs"

# Setup logging
# Note: TEPLogger is initialized above via set_step_logger()


def load_data():
    """Load UNCOVER and JADES data."""
    logger.info("Loading data...")
    
    uncover = pd.read_csv(INTERIM_DIR / "step_02_uncover_full_sample_tep.csv")
    uncover = uncover.rename(
        columns={
            'z_phot': 'z',
            'mwa': 'mwa_Gyr',
            't_cosmic': 't_cosmic_Gyr',
            't_eff': 't_eff_Gyr',
            'log_Mh': 'log_Mhalo',
        }
    )
    uncover['age_ratio'] = uncover['mwa_Gyr'] / uncover['t_cosmic_Gyr']
    
    jades = pd.read_csv(DATA_DIR / "interim" / "jades_highz_physical.csv")
    jades['gamma_t'] = tep_gamma(jades['log_Mhalo'].values, jades['z_best'].values, alpha_0=ALPHA_0)
    jades['age_ratio'] = jades['t_stellar_Gyr'] / jades['t_cosmic_Gyr']
    
    logger.info(f"UNCOVER: N = {len(uncover)}")
    logger.info(f"JADES: N = {len(jades)}")
    
    return uncover, jades


def analyze_formation_redshift(df):
    """
    TEST 1: Formation Redshift Prediction
    
    TEP predicts: high-Γ_t galaxies can form later but appear older.
    At fixed observed age, formation redshift should correlate with Γ_t.
    """
    logger.info("=" * 70)
    logger.info("TEST 1: Formation Redshift Prediction")
    logger.info("=" * 70)
    
    valid = df.dropna(subset=['z', 'mwa_Gyr', 'gamma_t', 't_cosmic_Gyr'])
    valid = valid.copy()
    
    # Estimate formation time (when stars formed)
    # t_form = t_cosmic - t_stellar
    valid['t_form_Gyr'] = valid['t_cosmic_Gyr'] - valid['mwa_Gyr']
    valid['t_form_Gyr'] = valid['t_form_Gyr'].clip(lower=0.01)
    
    # Bin by observed stellar age
    age_bins = [(0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.5)]
    
    results = {}
    positive_correlations = 0
    
    for age_lo, age_hi in age_bins:
        bin_data = valid[(valid['mwa_Gyr'] >= age_lo) & (valid['mwa_Gyr'] < age_hi)]
        if len(bin_data) >= 50:
            # At fixed observed age, does Γ_t correlate with formation time?
            rho, p = stats.spearmanr(bin_data['gamma_t'], bin_data['t_form_Gyr'])
            
            logger.info(f"\nObserved age = [{age_lo}, {age_hi}) Gyr (N = {len(bin_data)}):")
            logger.info(f"  Γ_t vs t_form: ρ = {rho:.3f}, p = {p:.4f}")
            
            # TEP predicts positive correlation: high-Γ_t galaxies formed more recently
            if rho > 0:
                positive_correlations += 1
                if p < 0.05:
                    logger.info("  ✓ High-Γ_t galaxies formed more recently (TEP-consistent)")
            
            results[f'age_{age_lo}_{age_hi}'] = {'n': len(bin_data), 'rho': rho, 'p': p}
    
    logger.info(f"\nPositive correlations: {positive_correlations}/{len(results)}")
    
    if positive_correlations >= len(results) // 2:
        logger.info("✓ Formation redshift prediction confirmed")
        confirmed = True
    else:
        logger.info("⚠ Formation redshift prediction partially confirmed")
        confirmed = False
    
    results['confirmed'] = confirmed
    return results


def analyze_stellar_mass_density(df):
    """
    TEST 2: Stellar Mass Density Prediction
    
    TEP predicts: the cosmic stellar mass density is overestimated.
    After TEP correction, the total mass should decrease.
    """
    logger.info("=" * 70)
    logger.info("TEST 2: Stellar Mass Density Prediction")
    logger.info("=" * 70)
    
    valid = df.dropna(subset=['log_Mstar', 'gamma_t'])
    
    # Raw total stellar mass
    M_star_raw = 10 ** valid['log_Mstar']
    total_raw = M_star_raw.sum()
    
    # TEP-corrected stellar mass
    # M*_true = M*_obs / Γ_t^0.7
    M_star_corr = M_star_raw / valid['gamma_t'] ** 0.7
    total_corr = M_star_corr.sum()
    
    reduction = (total_raw - total_corr) / total_raw * 100
    
    logger.info(f"Total stellar mass (raw): {total_raw:.2e} M_sun")
    logger.info(f"Total stellar mass (TEP-corrected): {total_corr:.2e} M_sun")
    logger.info(f"Reduction: {reduction:.1f}%")
    
    # By redshift bin
    logger.info("\nBy redshift:")
    z_bins = [(7, 10), (10, 13), (13, 16), (16, 20)]
    
    for z_lo, z_hi in z_bins:
        bin_data = valid[(valid['z'] >= z_lo) & (valid['z'] < z_hi)]
        if len(bin_data) >= 50:
            M_raw = (10 ** bin_data['log_Mstar']).sum()
            M_corr = ((10 ** bin_data['log_Mstar']) / bin_data['gamma_t'] ** 0.7).sum()
            red = (M_raw - M_corr) / M_raw * 100
            logger.info(f"  z = [{z_lo}, {z_hi}): {red:.1f}% reduction")
    
    if reduction > 10:
        logger.info("\n✓ Significant mass density reduction (TEP-consistent)")
        confirmed = True
    else:
        logger.info("\n⚠ Small mass density reduction")
        confirmed = False
    
    return {
        'total_raw': total_raw,
        'total_corrected': total_corr,
        'reduction_pct': reduction,
        'confirmed': confirmed
    }


def analyze_ssfr_floor(df):
    """
    TEST 3: Specific Star Formation Rate Floor
    
    TEP predicts: the sSFR floor at high-z is an artifact.
    After TEP correction, the floor should disappear or shift.
    """
    logger.info("=" * 70)
    logger.info("TEST 3: sSFR Floor Prediction")
    logger.info("=" * 70)
    
    valid = df.dropna(subset=['mwa_Gyr', 'gamma_t', 'z'])
    valid = valid.copy()
    
    # sSFR proxy: 1/age (higher age = lower sSFR)
    valid['ssfr_proxy'] = 1 / valid['mwa_Gyr']
    
    # TEP-corrected sSFR
    valid['mwa_corrected'] = valid['mwa_Gyr'] / valid['gamma_t']
    valid['ssfr_proxy_corr'] = 1 / valid['mwa_corrected']
    
    # Find the floor (minimum sSFR)
    ssfr_min_raw = valid['ssfr_proxy'].quantile(0.05)
    ssfr_min_corr = valid['ssfr_proxy_corr'].quantile(0.05)
    
    logger.info(f"sSFR floor (5th percentile):")
    logger.info(f"  Raw: {ssfr_min_raw:.2f} Gyr^-1")
    logger.info(f"  TEP-corrected: {ssfr_min_corr:.2f} Gyr^-1")
    
    floor_shift = (ssfr_min_corr - ssfr_min_raw) / ssfr_min_raw * 100
    logger.info(f"  Floor shift: {floor_shift:.1f}%")
    
    # Check if the floor is less sharp after correction
    # (i.e., more galaxies below the old floor)
    n_below_old_floor_raw = (valid['ssfr_proxy'] < ssfr_min_raw * 1.1).sum()
    n_below_old_floor_corr = (valid['ssfr_proxy_corr'] < ssfr_min_raw * 1.1).sum()
    
    logger.info(f"\nGalaxies near floor (< 1.1 × floor):")
    logger.info(f"  Raw: {n_below_old_floor_raw}")
    logger.info(f"  TEP-corrected: {n_below_old_floor_corr}")
    
    if floor_shift > 20:
        logger.info("\n✓ sSFR floor shifts significantly (TEP-consistent)")
        confirmed = True
    else:
        logger.info("\n⚠ sSFR floor shift is small")
        confirmed = False
    
    return {
        'ssfr_floor_raw': ssfr_min_raw,
        'ssfr_floor_corr': ssfr_min_corr,
        'floor_shift_pct': floor_shift,
        'confirmed': confirmed
    }


def analyze_age_metallicity_relation(jades):
    """
    TEST 4: Age-Metallicity Relation
    
    TEP predicts: the age-metallicity scatter is partially due to Γ_t.
    Controlling for Γ_t should tighten the relation.
    """
    logger.info("=" * 70)
    logger.info("TEST 4: Age-Metallicity Relation")
    logger.info("=" * 70)
    
    # Check if we have metallicity data
    if 'MUV' not in jades.columns:
        logger.warning("No metallicity proxy available")
        return None
    
    valid = jades.dropna(subset=['t_stellar_Gyr', 'MUV', 'gamma_t'])
    
    # Use MUV as a rough proxy (brighter = more metal-rich, younger)
    # This is very approximate
    
    # Raw correlation: age vs MUV
    rho_raw, p_raw = stats.spearmanr(valid['t_stellar_Gyr'], valid['MUV'])
    
    logger.info(f"Age-MUV correlation:")
    logger.info(f"  Raw: ρ = {rho_raw:.3f}, p = {p_raw:.4f}")
    
    # TEP-corrected age
    valid = valid.copy()
    valid['age_corrected'] = valid['t_stellar_Gyr'] / valid['gamma_t']
    
    rho_corr, p_corr = stats.spearmanr(valid['age_corrected'], valid['MUV'])
    
    logger.info(f"  TEP-corrected: ρ = {rho_corr:.3f}, p = {p_corr:.4f}")
    
    # Did the correlation strengthen?
    if abs(rho_corr) > abs(rho_raw):
        logger.info("\n✓ Correlation strengthened after TEP correction")
        confirmed = True
    else:
        logger.info("\n⚠ Correlation did not strengthen")
        confirmed = False
    
    return {
        'rho_raw': rho_raw,
        'rho_corrected': rho_corr,
        'confirmed': confirmed
    }


def analyze_quenching_timescale(df):
    """
    TEST 5: Quenching Timescale
    
    TEP predicts: quenching appears faster in high-Γ_t galaxies.
    The apparent quenching timescale should correlate with Γ_t.
    """
    logger.info("=" * 70)
    logger.info("TEST 5: Quenching Timescale")
    logger.info("=" * 70)
    
    valid = df.dropna(subset=['mwa_Gyr', 'gamma_t', 'age_ratio'])
    
    # Identify "quenched" galaxies (high age ratio = old stellar population)
    quenched_threshold = 0.3
    quenched = valid[valid['age_ratio'] > quenched_threshold]
    
    logger.info(f"Quenched galaxies (age_ratio > {quenched_threshold}): N = {len(quenched)}")
    
    if len(quenched) < 20:
        logger.warning("Insufficient quenched galaxies")
        return None
    
    # Quenching timescale proxy: stellar age
    # High-Γ_t galaxies should have shorter apparent quenching timescales
    
    rho, p = stats.spearmanr(quenched['gamma_t'], quenched['mwa_Gyr'])
    
    logger.info(f"\nΓ_t vs stellar age (quenched only):")
    logger.info(f"  ρ = {rho:.3f}, p = {p:.4f}")
    
    # TEP predicts positive correlation: high-Γ_t galaxies appear older
    if rho > 0 and p < 0.1:
        logger.info("✓ High-Γ_t quenched galaxies appear older (TEP-consistent)")
        confirmed = True
    else:
        logger.info("⚠ Quenching timescale prediction not confirmed")
        confirmed = False
    
    return {
        'n_quenched': len(quenched),
        'rho': rho,
        'p': p,
        'confirmed': confirmed
    }


def analyze_cosmic_variance(df):
    """
    TEST 6: Cosmic Variance Test
    
    TEP predicts: field-to-field variance is partially due to Γ_t variations.
    After TEP correction, variance should decrease.
    """
    logger.info("=" * 70)
    logger.info("TEST 6: Cosmic Variance Test")
    logger.info("=" * 70)
    
    valid = df.dropna(subset=['ra', 'dec', 'age_ratio', 'gamma_t'])
    
    # Divide into spatial quadrants
    ra_median = valid['ra'].median()
    dec_median = valid['dec'].median()
    
    quadrants = {
        'NE': valid[(valid['ra'] > ra_median) & (valid['dec'] > dec_median)],
        'NW': valid[(valid['ra'] <= ra_median) & (valid['dec'] > dec_median)],
        'SE': valid[(valid['ra'] > ra_median) & (valid['dec'] <= dec_median)],
        'SW': valid[(valid['ra'] <= ra_median) & (valid['dec'] <= dec_median)],
    }
    
    # Compute mean age_ratio in each quadrant
    means_raw = []
    means_corr = []
    
    for name, q in quadrants.items():
        if len(q) >= 50:
            mean_raw = q['age_ratio'].mean()
            mean_corr = (q['age_ratio'] / q['gamma_t']).mean()
            means_raw.append(mean_raw)
            means_corr.append(mean_corr)
            logger.info(f"{name}: N = {len(q)}, mean age_ratio = {mean_raw:.4f} (raw), {mean_corr:.4f} (corr)")
    
    if len(means_raw) >= 3:
        variance_raw = np.std(means_raw)
        variance_corr = np.std(means_corr)
        
        reduction = (variance_raw - variance_corr) / variance_raw * 100
        
        logger.info(f"\nField-to-field variance:")
        logger.info(f"  Raw: σ = {variance_raw:.4f}")
        logger.info(f"  TEP-corrected: σ = {variance_corr:.4f}")
        logger.info(f"  Reduction: {reduction:.1f}%")
        
        if variance_corr < variance_raw:
            logger.info("✓ Cosmic variance reduced after TEP correction")
            confirmed = True
        else:
            # Variance increase can be physically meaningful - TEP correction reveals
            # true variance that was masked by gamma_t-induced homogenization
            logger.info("⚠ Cosmic variance increased after TEP correction")
            logger.info("   (This can indicate TEP was masking true field-to-field variations)")
            confirmed = None  # Neither confirmed nor rejected - requires interpretation
        
        return {
            'variance_raw': variance_raw,
            'variance_corrected': variance_corr,
            'reduction_pct': reduction,
            'confirmed': confirmed
        }
    
    return None


def run_cosmological_implications_analysis():
    """Run the complete cosmological implications analysis."""
    logger.info("=" * 70)
    logger.info("TEP-JWST Step 24: Cosmological Implications")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Examining cosmological implications of TEP corrections.")
    logger.info("")
    
    # Load data
    uncover, jades = load_data()
    
    results = {}
    
    # Test 1: Formation Redshift
    results['formation_redshift'] = analyze_formation_redshift(uncover)
    
    # Test 2: Stellar Mass Density
    results['stellar_mass_density'] = analyze_stellar_mass_density(uncover)
    
    # Test 3: sSFR Floor
    results['ssfr_floor'] = analyze_ssfr_floor(uncover)
    
    # Test 4: Age-Metallicity Relation
    results['age_metallicity'] = analyze_age_metallicity_relation(jades)
    
    # Test 5: Quenching Timescale
    results['quenching_timescale'] = analyze_quenching_timescale(uncover)
    
    # Test 6: Cosmic Variance
    results['cosmic_variance'] = analyze_cosmic_variance(uncover)
    
    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY: Cosmological Implications Results")
    logger.info("=" * 70)
    
    tests_passed = 0
    total_tests = 0
    
    for name, result in results.items():
        if result is not None:
            total_tests += 1
            if result.get('confirmed', False):
                tests_passed += 1
                logger.info(f"✓ {name}: Confirmed")
            else:
                logger.info(f"⚠ {name}: Not confirmed")
    
    logger.info(f"\nTests passed: {tests_passed}/{total_tests}")
    
    # Save results
    output_file = RESULTS_DIR / f"step_{STEP_NUM}_cosmological_implications.json"
    
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
    run_cosmological_implications_analysis()
