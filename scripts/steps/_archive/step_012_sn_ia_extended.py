#!/usr/bin/env python3
"""
TEP-JWST Step 12: Extended SN Ia Analysis

Building on the mass step analysis (Step 13), we explore additional TEP signatures
in the Pantheon+ data:

1. REDSHIFT EVOLUTION OF MASS STEP
   - If TEP is correct, the mass step should evolve with redshift
   - At higher z, galaxies are denser → stronger TEP effect
   - Prediction: mass step should INCREASE with redshift

2. COLOR AND STRETCH RESIDUALS
   - SN Ia standardization uses color (c) and stretch (x1)
   - If TEP affects the underlying physics, residuals after standardization
     should correlate with host mass/environment
   - This is the "second-order" TEP signature

3. HUBBLE RESIDUAL VS HOST PROPERTIES
   - Beyond mass, test correlation with other host properties
   - Peculiar velocity (VPEC) as proxy for local density
   - Galactic extinction (MWEBV) as control

4. CALIBRATOR VS HUBBLE FLOW COMPARISON
   - Calibrators (IS_CALIBRATOR=1) are in special environments
   - TEP predicts systematic offset between calibrators and Hubble flow SNe
   - This is the core of the Hubble Tension

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
from scripts.utils.p_value_utils import format_p_value, safe_json_default

STEP_NUM = "012"
STEP_NAME = "sn_ia_extended"

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
TEP_H0_DATA = PROJECT_ROOT.parent / "TEP-H0" / "data" / "raw" / "Pantheon+SH0ES.dat"

# Note: TEPLogger is initialized above via set_step_logger()

# =============================================================================
# TEP PARAMETERS (from TEP-H0)
# =============================================================================
ALPHA_TEP = 0.58
SIGMA_REF = 75.25  # km/s
MASS_THRESHOLD = 10.0  # log(M*/Msun)


def load_pantheon_data():
    """Load Pantheon+ data."""
    logger.info(f"Loading Pantheon+ data from {TEP_H0_DATA}")
    df = pd.read_csv(TEP_H0_DATA, sep=r'\s+')
    logger.info(f"Loaded {len(df)} SN Ia observations")
    return df


def calculate_hubble_residuals(df):
    """
    Calculate Hubble residuals from corrected magnitudes.
    
    Hubble residual = m_b_corr - μ_model(z)
    
    where μ_model is the distance modulus from a fiducial cosmology.
    For Pantheon+, we use: μ = 5*log10(d_L/10pc)
    with d_L from flat ΛCDM (H0=70, Ωm=0.3).
    """
    from astropy.cosmology import FlatLambdaCDM
    import astropy.units as u
    
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    
    df = df.copy()
    
    # Calculate expected distance modulus for each SN
    # Use zHD (Hubble diagram redshift, corrected for peculiar velocity)
    valid_z = df['zHD'] > 0.001
    
    df.loc[valid_z, 'mu_model'] = cosmo.distmod(df.loc[valid_z, 'zHD']).value
    
    # Hubble residual = observed - expected
    # Positive residual = fainter than expected = farther than expected
    df['hubble_residual'] = df['m_b_corr'] - df['mu_model']
    
    # Center residuals (remove any global offset)
    valid = df['hubble_residual'].notna()
    df.loc[valid, 'hubble_residual'] -= df.loc[valid, 'hubble_residual'].mean()
    
    return df


def analyze_mass_step_redshift_evolution(df):
    """
    Test if the mass step evolves with redshift.
    
    TEP Prediction: Higher-z galaxies are denser → stronger TEP effect
    → mass step should INCREASE with redshift
    """
    logger.info("=" * 70)
    logger.info("ANALYSIS 1: Mass Step Redshift Evolution")
    logger.info("=" * 70)
    
    # Filter valid data
    valid = df[(df['HOST_LOGMASS'] > 0) & (df['zHD'] > 0.01)].copy()
    
    # Define redshift bins
    z_bins = [0.01, 0.03, 0.05, 0.08, 0.15, 0.3, 1.0]
    z_labels = ['0.01-0.03', '0.03-0.05', '0.05-0.08', '0.08-0.15', '0.15-0.30', '0.30-1.00']
    
    valid['z_bin'] = pd.cut(valid['zHD'], bins=z_bins, labels=z_labels)
    
    results = []
    for z_label in z_labels:
        bin_data = valid[valid['z_bin'] == z_label]
        if len(bin_data) < 20:
            continue
        
        low_mass = bin_data[bin_data['HOST_LOGMASS'] < MASS_THRESHOLD]
        high_mass = bin_data[bin_data['HOST_LOGMASS'] >= MASS_THRESHOLD]
        
        if len(low_mass) < 5 or len(high_mass) < 5:
            continue
        
        mass_step = high_mass['hubble_residual'].mean() - low_mass['hubble_residual'].mean()
        mass_step_err = np.sqrt(
            (high_mass['hubble_residual'].std() / np.sqrt(len(high_mass)))**2 +
            (low_mass['hubble_residual'].std() / np.sqrt(len(low_mass)))**2
        )
        
        z_median = bin_data['zHD'].median()
        
        results.append({
            'z_bin': z_label,
            'z_median': z_median,
            'n_total': len(bin_data),
            'n_low': len(low_mass),
            'n_high': len(high_mass),
            'mass_step': mass_step,
            'mass_step_err': mass_step_err
        })
        
        logger.info(f"z = {z_label}: N = {len(bin_data)}, "
                   f"mass step = {mass_step:.4f} ± {mass_step_err:.4f} mag")
    
    if len(results) < 3:
        logger.warning("Insufficient data for redshift evolution analysis")
        return {'evolution_detected': False, 'bins': results}
    
    # Test for evolution: correlate mass step with redshift
    z_values = [r['z_median'] for r in results]
    step_values = [r['mass_step'] for r in results]
    
    rho, p_value = stats.spearmanr(z_values, step_values)
    p_value_fmt = format_p_value(p_value)
    
    logger.info(f"\nMass step vs redshift correlation:")
    logger.info(f"  Spearman ρ = {rho:.3f}, p = {p_value:.4f}")
    
    if rho > 0 and (p_value_fmt is not None and p_value_fmt < 0.1):
        logger.info("\n✓ Mass step INCREASES with redshift (TEP-consistent)")
    elif rho < 0 and (p_value_fmt is not None and p_value_fmt < 0.1):
        logger.info("\n✗ Mass step DECREASES with redshift (opposite to TEP)")
    else:
        logger.info("\n⚠ No significant redshift evolution detected")
    
    return {
        'bins': results,
        'spearman_rho': rho,
        'spearman_p': p_value_fmt,
        'evolution_detected': rho > 0 and (p_value_fmt is not None and p_value_fmt < 0.1)
    }


def analyze_calibrator_offset(df):
    """
    Compare calibrator SNe to Hubble flow SNe.
    
    TEP Prediction: Calibrators are in special environments (anchors like LMC,
    NGC 4258, M31) which are screened. Hubble flow SNe are in field environments
    which are unscreened. This creates a systematic offset.
    """
    logger.info("=" * 70)
    logger.info("ANALYSIS 2: Calibrator vs Hubble Flow Offset")
    logger.info("=" * 70)
    
    # Identify calibrators
    calibrators = df[df['IS_CALIBRATOR'] == 1]
    hubble_flow = df[(df['IS_CALIBRATOR'] == 0) & (df['zHD'] > 0.01)]
    
    logger.info(f"Calibrators: N = {len(calibrators)}")
    logger.info(f"Hubble flow (z > 0.01): N = {len(hubble_flow)}")
    
    if len(calibrators) < 3:
        logger.warning("Insufficient calibrators for analysis")
        return None
    
    # Compare mean residuals
    cal_mean = calibrators['hubble_residual'].mean()
    cal_err = calibrators['hubble_residual'].std() / np.sqrt(len(calibrators))
    
    hf_mean = hubble_flow['hubble_residual'].mean()
    hf_err = hubble_flow['hubble_residual'].std() / np.sqrt(len(hubble_flow))
    
    offset = cal_mean - hf_mean
    offset_err = np.sqrt(cal_err**2 + hf_err**2)
    
    logger.info(f"\nMean residuals:")
    logger.info(f"  Calibrators: {cal_mean:.4f} ± {cal_err:.4f} mag")
    logger.info(f"  Hubble flow: {hf_mean:.4f} ± {hf_err:.4f} mag")
    logger.info(f"  Offset (Cal - HF): {offset:.4f} ± {offset_err:.4f} mag")
    
    # Compare host masses
    cal_mass = calibrators[calibrators['HOST_LOGMASS'] > 0]['HOST_LOGMASS'].mean()
    hf_mass = hubble_flow[hubble_flow['HOST_LOGMASS'] > 0]['HOST_LOGMASS'].mean()
    
    logger.info(f"\nMean host mass:")
    logger.info(f"  Calibrators: log(M*) = {cal_mass:.2f}")
    logger.info(f"  Hubble flow: log(M*) = {hf_mass:.2f}")
    
    # TEP interpretation
    logger.info(f"\nTEP Interpretation:")
    logger.info(f"  Calibrators are in screened environments (group halos)")
    logger.info(f"  Hubble flow SNe are in mixed environments")
    logger.info(f"  The offset reflects the differential TEP bias")
    
    return {
        'n_calibrators': len(calibrators),
        'n_hubble_flow': len(hubble_flow),
        'cal_mean_residual': cal_mean,
        'cal_mean_err': cal_err,
        'hf_mean_residual': hf_mean,
        'hf_mean_err': hf_err,
        'offset': offset,
        'offset_err': offset_err,
        'cal_mean_mass': cal_mass,
        'hf_mean_mass': hf_mass
    }


def analyze_residual_correlations(df):
    """
    Test for correlations between Hubble residuals and various host properties.
    """
    logger.info("=" * 70)
    logger.info("ANALYSIS 3: Hubble Residual Correlations")
    logger.info("=" * 70)
    
    valid = df[(df['HOST_LOGMASS'] > 0) & (df['zHD'] > 0.01)].copy()
    
    correlations = {}
    
    # Test various correlations
    properties = [
        ('HOST_LOGMASS', 'Host stellar mass'),
        ('c', 'SN color'),
        ('x1', 'SN stretch'),
        ('zHD', 'Redshift'),
    ]
    
    for col, name in properties:
        if col in valid.columns:
            valid_subset = valid[valid[col].notna()]
            rho, p = stats.spearmanr(valid_subset[col], valid_subset['hubble_residual'])
            correlations[col] = {'name': name, 'rho': rho, 'p': format_p_value(p)}
            logger.info(f"{name}: ρ = {rho:.3f}, p = {p:.4f}")
    
    return correlations


def analyze_high_z_mass_step(df):
    """
    Focus on high-z SNe where TEP effects should be strongest.
    """
    logger.info("=" * 70)
    logger.info("ANALYSIS 4: High-z Mass Step (z > 0.1)")
    logger.info("=" * 70)
    
    high_z = df[(df['zHD'] > 0.1) & (df['HOST_LOGMASS'] > 0)].copy()
    
    logger.info(f"High-z sample (z > 0.1): N = {len(high_z)}")
    logger.info(f"Redshift range: {high_z['zHD'].min():.3f} - {high_z['zHD'].max():.3f}")
    
    low_mass = high_z[high_z['HOST_LOGMASS'] < MASS_THRESHOLD]
    high_mass = high_z[high_z['HOST_LOGMASS'] >= MASS_THRESHOLD]
    
    logger.info(f"Low-mass hosts: N = {len(low_mass)}")
    logger.info(f"High-mass hosts: N = {len(high_mass)}")
    
    if len(low_mass) < 10 or len(high_mass) < 10:
        logger.warning("Insufficient data for high-z mass step")
        return None
    
    mass_step = high_mass['hubble_residual'].mean() - low_mass['hubble_residual'].mean()
    mass_step_err = np.sqrt(
        (high_mass['hubble_residual'].std() / np.sqrt(len(high_mass)))**2 +
        (low_mass['hubble_residual'].std() / np.sqrt(len(low_mass)))**2
    )
    
    t_stat, p_value = stats.ttest_ind(high_mass['hubble_residual'], low_mass['hubble_residual'])
    p_value_fmt = format_p_value(p_value)
    
    logger.info(f"\nHigh-z mass step: {mass_step:.4f} ± {mass_step_err:.4f} mag")
    logger.info(f"t-statistic: {t_stat:.2f}, p = {p_value:.4f}")
    
    # Compare to full sample
    full_low = df[(df['HOST_LOGMASS'] > 0) & (df['HOST_LOGMASS'] < MASS_THRESHOLD)]
    full_high = df[(df['HOST_LOGMASS'] >= MASS_THRESHOLD)]
    full_step = full_high['hubble_residual'].mean() - full_low['hubble_residual'].mean()
    
    logger.info(f"\nComparison:")
    logger.info(f"  Full sample mass step: {full_step:.4f} mag")
    logger.info(f"  High-z mass step: {mass_step:.4f} mag")
    logger.info(f"  Ratio (high-z / full): {mass_step / full_step:.2f}")
    
    if mass_step > full_step:
        logger.info("\n✓ High-z mass step is LARGER (TEP-consistent)")
    else:
        logger.info("\n⚠ High-z mass step is not larger than full sample")
    
    return {
        'n_high_z': len(high_z),
        'n_low_mass': len(low_mass),
        'n_high_mass': len(high_mass),
        'mass_step': mass_step,
        'mass_step_err': mass_step_err,
        't_statistic': t_stat,
        'p_value': p_value_fmt,
        'full_sample_step': full_step,
        'ratio': mass_step / full_step if full_step != 0 else None
    }


def run_extended_sn_analysis():
    """Run the complete extended SN Ia analysis."""
    logger.info("=" * 70)
    logger.info("TEP-JWST Step 12: Extended SN Ia Analysis")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Exploring additional TEP signatures in Pantheon+ data")
    logger.info("")
    
    # Load data
    df = load_pantheon_data()
    df = calculate_hubble_residuals(df)
    
    results = {}
    
    # Analysis 1: Mass step redshift evolution
    results['z_evolution'] = analyze_mass_step_redshift_evolution(df)
    
    # Analysis 2: Calibrator offset
    results['calibrator_offset'] = analyze_calibrator_offset(df)
    
    # Analysis 3: Residual correlations
    results['correlations'] = analyze_residual_correlations(df)
    
    # Analysis 4: High-z mass step
    results['high_z_step'] = analyze_high_z_mass_step(df)
    
    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    
    if results['z_evolution']['evolution_detected']:
        logger.info("✓ Mass step increases with redshift (TEP-consistent)")
    else:
        logger.info("⚠ No significant mass step evolution with redshift")
    
    if results['high_z_step'] and results['high_z_step']['ratio'] and results['high_z_step']['ratio'] > 1.0:
        logger.info("✓ High-z mass step is enhanced (TEP-consistent)")
    else:
        logger.info("⚠ High-z mass step not significantly enhanced")
    
    # Save results
    output_file = RESULTS_DIR / f"step_{STEP_NUM}_sn_ia_extended.json"
    
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
        json.dump(results_serializable, f, indent=2, default=safe_json_default)
    
    logger.info(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    run_extended_sn_analysis()
