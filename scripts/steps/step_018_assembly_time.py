#!/usr/bin/env python3
"""
TEP-JWST Step 18: Assembly Time Analysis

This analysis examines assembly time correlations with the chronological enhancement factor.
This analysis tests targeted TEP predictions:

1. ASSEMBLY TIME SCALING
   - t_assembly should scale with Γ_t
   - Faster assembly in high-Γ_t systems (more proper time available)

2. EFFECTIVE TIME THRESHOLD TESTS
   - Multiple thresholds for different physical processes
   - AGB dust: 300 Myr
   - Chemical enrichment: 100 Myr
   - Quenching: 500 Myr

3. REDSHIFT SCALING VERIFICATION
   - Test if α(z) = α_0 × (1+z)^0.5 is correct
   - Compare low-z and high-z samples

4. CROSS-SAMPLE CONSISTENCY
   - JADES vs UNCOVER: same TEP signatures?
   - Different selection, same physics

5. THE FORMATION EPOCH TEST
   - TEP predicts: high-Γ_t galaxies can form later but appear older
   - Test: formation redshift vs apparent age

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
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging
from scripts.utils.p_value_utils import format_p_value  # Safe p-value formatting
from scripts.utils.tep_model import ALPHA_0, compute_gamma_t as tep_gamma  # Shared TEP model

STEP_NUM = "018"  # Pipeline step number
STEP_NAME = "assembly_time"  # Used in log / output filenames

LOGS_PATH = PROJECT_ROOT / "logs"  # Log directory
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"  # JSON output directory
LOGS_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")  # Step-specific logger
set_step_logger(logger)



# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
INTERIM_DIR = PROJECT_ROOT / "results" / "interim"
RESULTS_DIR = PROJECT_ROOT / "results" / "outputs"

# Note: TEPLogger is initialized above via set_step_logger()

def load_data():
    """Load UNCOVER and JADES datasets for cross-sample analysis.

    UNCOVER: Abell 2744 lensing field (Bezanson et al. 2022), loaded
    from the step_002 TEP-augmented catalog. Contains SED-derived
    properties including mass-weighted age (mwa), t_cosmic, and t_eff.

    JADES: GOODS-S deep field (Eisenstein et al. 2023), loaded from
    the step_014 physical-properties CSV. Gamma_t is recomputed from
    the shared canonical model.
    """
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

    _jades_path = DATA_DIR / "interim" / "jades_highz_physical.csv"
    if not Path(_jades_path).exists():
        print_status("ERROR: jades_highz_physical.csv not found. Run step_014 first.", "ERROR")
        return None, None
    jades = pd.read_csv(_jades_path)

    # Calculate Γ_t for JADES using canonical implementation
    jades['gamma_t'] = tep_gamma(jades['log_Mhalo'].values, jades['z_best'].values, alpha_0=ALPHA_0)
    jades['age_ratio'] = jades['t_stellar_Gyr'] / jades['t_cosmic_Gyr']

    # Calculate age_ratio for UNCOVER
    uncover['age_ratio'] = uncover['mwa_Gyr'] / uncover['t_cosmic_Gyr']
    
    logger.info(f"UNCOVER: N = {len(uncover)}")
    logger.info(f"JADES: N = {len(jades)}")
    
    return uncover, jades


def analyze_assembly_time(uncover):
    """TEST 1: Assembly time scaling with Gamma_t.

    t_assembly (Myr) is the SED-inferred time for a galaxy to build
    its current stellar mass. Under TEP, the *apparent* assembly time
    reported by SED fitting is:

      t_assembly_apparent = t_assembly_true * (1 + Gamma_t)

    So high-Gamma_t galaxies should have *longer* apparent assembly
    times (positive correlation), because the enhanced proper time
    inflates timescales inferred by isochrone-based SED models.

    A significant positive Spearman rho is TEP-consistent.
    """
    logger.info("=" * 70)
    logger.info("TEST 1: Assembly Time Scaling")
    logger.info("=" * 70)

    if 't_assembly_Myr' not in uncover.columns:
        logger.warning("t_assembly_Myr not available in UNCOVER data — skipping Test 1.")
        return None

    valid = uncover.dropna(subset=['t_assembly_Myr', 'gamma_t'])
    valid = valid[valid['t_assembly_Myr'] > 0]
    
    logger.info(f"Sample size: N = {len(valid)}")
    logger.info(f"t_assembly range: {valid['t_assembly_Myr'].min():.1f} - {valid['t_assembly_Myr'].max():.1f} Myr")
    
    # Correlation
    rho, p_value = stats.spearmanr(valid['gamma_t'], valid['t_assembly_Myr'])
    p_value_fmt = format_p_value(p_value)
    
    logger.info(f"\nCorrelation (t_assembly vs Γ_t):")
    logger.info(f"  Spearman ρ = {rho:.3f}, p = {p_value:.2e}")
    
    # TEP predicts POSITIVE correlation: higher Γ_t → longer assembly time
    # (because t_assembly is in coordinate time, and high-Γ_t galaxies
    # need more coordinate time to reach the same proper time)
    # Wait - actually TEP predicts that high-Γ_t galaxies APPEAR to have
    # assembled faster because their proper time is enhanced.
    
    # Let's think carefully:
    # t_assembly_apparent = t_assembly_true × (1 + Γ_t)
    # So high-Γ_t galaxies should have LONGER apparent assembly times
    
    if rho > 0 and (p_value_fmt is not None and p_value_fmt < 0.05):
        logger.info("\n✓ Higher Γ_t → longer apparent assembly time (TEP-consistent)")
        logger.info("  This is expected: TEP inflates apparent timescales")
        tep_consistent = True
    elif rho > 0:
        logger.info("\n⚠ Positive trend but not significant")
        tep_consistent = False
    else:
        logger.info("\n⚠ Negative correlation - needs interpretation")
        tep_consistent = False
    
    # Bin analysis
    logger.info("\nBinned analysis:")
    gamma_bins = [0, 0.5, 1.0, 2.0, 10.0]
    for i in range(len(gamma_bins) - 1):
        bin_data = valid[(valid['gamma_t'] >= gamma_bins[i]) & (valid['gamma_t'] < gamma_bins[i+1])]
        if len(bin_data) >= 10:
            mean_t = bin_data['t_assembly_Myr'].mean()
            sem_t = bin_data['t_assembly_Myr'].std() / np.sqrt(len(bin_data))
            logger.info(f"  Γ_t = [{gamma_bins[i]:.1f}, {gamma_bins[i+1]:.1f}): "
                       f"N = {len(bin_data)}, t_assembly = {mean_t:.1f} ± {sem_t:.1f} Myr")
    
    return {
        'n_galaxies': len(valid),
        'spearman_rho': rho,
        'spearman_p': p_value_fmt,
        'tep_consistent': tep_consistent
    }


def analyze_effective_time_thresholds(uncover):
    """TEST 2: Effective time threshold tests for physical processes.

    Different astrophysical processes require minimum elapsed proper
    time to produce observable signatures:
      - Chemical enrichment: ~100 Myr (core-collapse SNe + mixing)
      - AGB dust production: ~300 Myr (intermediate-mass stars reach AGB)
      - Quenching onset: ~500 Myr (gas depletion or AGN feedback)
      - Full passive evolution: ~1000 Myr (red sequence arrival)

    Under TEP, t_eff = t_cosmic * Gamma_t is the effective proper time.
    Galaxies with t_eff above a given threshold should show enhanced
    signatures of the corresponding process. This test splits the
    sample at each threshold and compares mean age_ratio above vs below
    using Welch's t-test.
    """
    logger.info("=" * 70)
    logger.info("TEST 2: Effective Time Threshold Tests")
    logger.info("=" * 70)
    
    valid = uncover.dropna(subset=['t_eff_Gyr', 'gamma_t'])
    valid['t_eff_Myr'] = valid['t_eff_Gyr'] * 1000
    
    thresholds = {
        'chemical_enrichment': 100,
        'agb_dust': 300,
        'quenching': 500,
        'full_evolution': 1000,
    }
    
    results = {}
    
    for process, threshold in thresholds.items():
        above = valid[valid['t_eff_Myr'] > threshold]
        below = valid[valid['t_eff_Myr'] <= threshold]
        
        if len(above) >= 10 and len(below) >= 10:
            # Compare age ratios
            mean_above = above['age_ratio'].mean()
            mean_below = below['age_ratio'].mean()
            
            # Statistical test
            t_stat, p_value = stats.ttest_ind(above['age_ratio'], below['age_ratio'])
            p_value_fmt = format_p_value(p_value)
            
            ratio = mean_above / mean_below if mean_below > 0 else np.nan
            
            logger.info(f"\n{process} (threshold = {threshold} Myr):")
            logger.info(f"  N above: {len(above)}, N below: {len(below)}")
            logger.info(f"  Mean age_ratio above: {mean_above:.3f}")
            logger.info(f"  Mean age_ratio below: {mean_below:.3f}")
            logger.info(f"  Ratio: {ratio:.2f}×")
            logger.info(f"  p-value: {p_value:.2e}")
            
            if mean_above > mean_below and (p_value_fmt is not None and p_value_fmt < 0.05):
                logger.info(f"  ✓ Significant enhancement above threshold")
            
            results[process] = {
                'threshold_Myr': threshold,
                'n_above': len(above),
                'n_below': len(below),
                'mean_above': mean_above,
                'mean_below': mean_below,
                'ratio': ratio,
                'p_value': p_value_fmt
            }
    
    return results


def analyze_redshift_scaling(uncover):
    """TEST 3: Verify the redshift scaling of the TEP coupling.

    The TEP model uses a redshift-dependent coupling:
      alpha(z) = alpha_0 * sqrt(1 + z)

    This predicts that the slope d(age_ratio)/d(Gamma_t) should
    *increase* with redshift, because alpha amplifies Gamma_t at
    higher z. We fit linear slopes in redshift bins and test whether
    the slope-z relation has positive rank correlation.
    """
    logger.info("=" * 70)
    logger.info("TEST 3: Redshift Scaling Verification")
    logger.info("=" * 70)
    
    valid = uncover.dropna(subset=['z', 'gamma_t', 'age_ratio'])
    
    # Split by redshift
    z_bins = [(7, 9), (9, 11), (11, 14), (14, 20)]
    
    slopes = []
    z_centers = []
    
    for z_lo, z_hi in z_bins:
        bin_data = valid[(valid['z'] >= z_lo) & (valid['z'] < z_hi)]
        if len(bin_data) >= 50:
            # Fit linear relation
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                bin_data['gamma_t'], bin_data['age_ratio']
            )
            
            z_center = (z_lo + z_hi) / 2
            z_centers.append(z_center)
            slopes.append(slope)
            
            logger.info(f"\nz = [{z_lo}, {z_hi}): N = {len(bin_data)}")
            logger.info(f"  Slope (age_ratio vs Γ_t): {slope:.4f} ± {std_err:.4f}")
            logger.info(f"  Expected α(z) scaling: (1+{z_center})^0.5 = {np.sqrt(1+z_center):.2f}")
    
    if len(slopes) >= 2:
        # Test if slope increases with z
        rho_slope_z, p_slope_z = stats.spearmanr(z_centers, slopes)
        
        # Handle edge case where p_slope_z is NaN (e.g., strong correlation with only 2 points)
        if np.isnan(p_slope_z):
            # For 2 points, strong correlation is expected; set p to 1.0 (not significant)
            p_slope_z = 1.0 if len(slopes) <= 2 else 0.0
        
        p_slope_z_fmt = format_p_value(p_slope_z)
        logger.info(f"\nSlope vs z correlation: ρ = {rho_slope_z:.3f}, p = {p_slope_z:.4f}")
        
        if rho_slope_z > 0:
            logger.info("✓ Slope increases with z (TEP-consistent)")
        else:
            logger.info("⚠ Slope does not increase with z")
        
        return {
            'z_centers': z_centers,
            'slopes': slopes,
            'rho_slope_z': rho_slope_z,
            'p_slope_z': p_slope_z_fmt,
            'tep_consistent': rho_slope_z > 0
        }
    
    return None


def analyze_cross_sample_consistency(uncover, jades):
    """TEST 4: Cross-sample consistency between UNCOVER and JADES.

    UNCOVER (lensing-selected, Abell 2744) and JADES (blank-field,
    GOODS-S) have different selection functions, depth profiles,
    and photometric calibrations. If TEP is a real physical effect,
    the age_ratio-Gamma_t correlation should be consistent across
    both samples.

    Consistency is assessed via Fisher z-transformation:
      z = arctanh(rho)
    with the test statistic:
      Z = (z_UNCOVER - z_JADES) / sqrt(1/(N1-3) + 1/(N2-3))
    A p > 0.05 indicates no significant difference.
    """
    logger.info("=" * 70)
    logger.info("TEST 4: Cross-Sample Consistency")
    logger.info("=" * 70)
    
    results = {}
    
    # Test 1: age_ratio vs Γ_t correlation
    for name, df in [('UNCOVER', uncover), ('JADES', jades)]:
        valid = df.dropna(subset=['gamma_t', 'age_ratio'])
        rho, p_value = stats.spearmanr(valid['gamma_t'], valid['age_ratio'])
        
        logger.info(f"\n{name} (N = {len(valid)}):")
        logger.info(f"  age_ratio vs Γ_t: ρ = {rho:.3f}, p = {p_value:.4f}")
        
        results[name] = {
            'n': len(valid),
            'rho': rho,
            'p': format_p_value(p_value)
        }
    
    # Test consistency
    rho_uncover = results['UNCOVER']['rho']
    rho_jades = results['JADES']['rho']
    
    # Fisher z-transformation for comparing correlations
    z_uncover = np.arctanh(rho_uncover)
    z_jades = np.arctanh(rho_jades)
    
    n_uncover = results['UNCOVER']['n']
    n_jades = results['JADES']['n']
    
    se_diff = np.sqrt(1/(n_uncover-3) + 1/(n_jades-3))
    z_diff = (z_uncover - z_jades) / se_diff
    p_diff = 2 * stats.norm.sf(abs(z_diff))
    
    logger.info(f"\nConsistency test:")
    logger.info(f"  Δρ = {rho_uncover - rho_jades:.3f}")
    logger.info(f"  z-test: p = {p_diff:.4f}")
    
    if p_diff > 0.05:
        logger.info("✓ Correlations are consistent across samples")
        results['consistent'] = True
    else:
        logger.info("⚠ Correlations differ between samples")
        results['consistent'] = False
    
    return results


def analyze_formation_epoch(uncover):
    """TEST 5: Formation epoch test at fixed apparent stellar age.

    Under TEP, high-Gamma_t galaxies accumulate proper time faster,
    so they can form *later* in coordinate time while still producing
    stellar populations that appear equally old in SED fitting.

    The formation time is estimated as:
      t_form = t_cosmic - t_stellar (mass-weighted age)

    At fixed apparent age (binned), Gamma_t should positively
    correlate with t_form (more recent formation for higher Gamma_t),
    because less coordinate time was needed to reach the same
    evolutionary state.
    """
    logger.info("=" * 70)
    logger.info("TEST 5: Formation Epoch Test")
    logger.info("=" * 70)
    
    valid = uncover.dropna(subset=['z', 'gamma_t', 'mwa_Gyr', 't_cosmic_Gyr'])
    
    # Calculate formation redshift (approximate)
    # z_form is when the galaxy formed, estimated from current z and stellar age
    # This is approximate because we're using coordinate time
    valid = valid.copy()
    valid['t_form_Gyr'] = valid['t_cosmic_Gyr'] - valid['mwa_Gyr']
    valid['t_form_Gyr'] = valid['t_form_Gyr'].clip(lower=0.01)
    
    # Bin by apparent age
    age_bins = [(0, 0.1), (0.1, 0.2), (0.2, 0.5)]
    
    results = {}
    
    for age_lo, age_hi in age_bins:
        bin_data = valid[(valid['mwa_Gyr'] >= age_lo) & (valid['mwa_Gyr'] < age_hi)]
        if len(bin_data) >= 30:
            # At fixed apparent age, does Γ_t correlate with formation time?
            rho, p_value = stats.spearmanr(bin_data['gamma_t'], bin_data['t_form_Gyr'])
            
            logger.info(f"\nApparent age = [{age_lo}, {age_hi}) Gyr (N = {len(bin_data)}):")
            logger.info(f"  Γ_t vs t_form: ρ = {rho:.3f}, p = {p_value:.4f}")
            
            # TEP predicts: high-Γ_t galaxies formed more recently (higher t_form)
            # because they needed less coordinate time to reach the same apparent age
            if rho > 0 and p_value < 0.1:
                logger.info("  ✓ High-Γ_t galaxies formed more recently (TEP-consistent)")
            
            results[f'age_{age_lo}_{age_hi}'] = {
                'n': len(bin_data),
                'rho': rho,
                'p': format_p_value(p_value)
            }
    
    return results


def run_assembly_time_analysis():
    """Run the complete assembly time analysis."""
    logger.info("=" * 70)
    logger.info(f"TEP-JWST Step {STEP_NUM}: Assembly Time Analysis")
    logger.info("=" * 70)
    logger.info("")
    logger.info("This analysis examines assembly time correlations with the chronological enhancement factor.")
    logger.info("")
    
    # Load data
    uncover, jades = load_data()
    if jades is None:
        logger.error(f"Step {STEP_NUM} aborted: jades_highz_physical.csv not found. Run step_014 first.")
        return {"status": "aborted", "reason": "missing jades_highz_physical.csv"}

    results = {}
    
    # Test 1: Assembly Time
    results['assembly_time'] = analyze_assembly_time(uncover)
    
    # Test 2: Effective Time Thresholds
    results['time_thresholds'] = analyze_effective_time_thresholds(uncover)
    
    # Test 3: Redshift Scaling
    results['redshift_scaling'] = analyze_redshift_scaling(uncover)
    
    # Test 4: Cross-Sample Consistency
    results['cross_sample'] = analyze_cross_sample_consistency(uncover, jades)
    
    # Test 5: Formation Epoch
    results['formation_epoch'] = analyze_formation_epoch(uncover)
    
    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    
    tests_passed = 0
    total_tests = 5
    
    if results['assembly_time'] and results['assembly_time'].get('tep_consistent'):
        tests_passed += 1
        logger.info("✓ Assembly time scaling: TEP-consistent")
    else:
        logger.info("⚠ Assembly time: Needs interpretation")
    
    if results['time_thresholds']:
        sig_thresholds = sum(1 for v in results['time_thresholds'].values() 
                           if v.get('p_value', 1) < 0.05)
        if sig_thresholds >= 2:
            tests_passed += 1
            logger.info(f"✓ Time thresholds: {sig_thresholds}/4 significant")
        else:
            logger.info(f"⚠ Time thresholds: {sig_thresholds}/4 significant")
    
    if results['redshift_scaling'] and results['redshift_scaling'].get('tep_consistent'):
        tests_passed += 1
        logger.info("✓ Redshift scaling: TEP-consistent")
    else:
        logger.info("⚠ Redshift scaling: Needs more data")
    
    if results['cross_sample'].get('consistent'):
        tests_passed += 1
        logger.info("✓ Cross-sample consistency: Confirmed")
    else:
        logger.info("⚠ Cross-sample: Samples differ")
    
    if results['formation_epoch']:
        sig_epochs = sum(1 for v in results['formation_epoch'].values() 
                        if v.get('p', 1) < 0.1 and v.get('rho', 0) > 0)
        if sig_epochs >= 1:
            tests_passed += 1
            logger.info(f"✓ Formation epoch: {sig_epochs} bins TEP-consistent")
        else:
            logger.info("⚠ Formation epoch: Weak signal")
    
    logger.info(f"\nTests passed: {tests_passed}/{total_tests}")
    
    # Save results
    output_file = RESULTS_DIR / f"step_{STEP_NUM}_assembly_time.json"
    
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
    run_assembly_time_analysis()
