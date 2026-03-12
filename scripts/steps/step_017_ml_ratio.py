#!/usr/bin/env python3
# Estimated runtime from last full canonical run (2026-03-09 15:52 UTC; full pipeline 32m18s): 2.7s.
"""
TEP-JWST Step 17: Mass-to-Light Ratio Analysis

This analysis examines mass-to-light ratio correlations with the chronological enhancement factor.

This analysis identifies natural gaps in the data where TEP predictions can be
tested without forcing the theory into the data.

TESTS TO PERFORM:

1. AGE EXCESS DISTRIBUTION
   - JADES provides age_excess = t_stellar - t_cosmic
   - Negative values are "anomalous" (stellar age > cosmic age)
   - TEP predicts: age_excess should correlate with Γ_t

2. PHOTOMETRIC REDSHIFT SCATTER
   - Photo-z errors should be LARGER for high-Γ_t galaxies
   - Because TEP affects the SED shape, confusing the photo-z fit
   - This is a PREDICTION, not a post-hoc explanation

3. SED FIT QUALITY (χ²)
   - Standard SED models assume isochrony
   - High-Γ_t galaxies should have WORSE fits (higher χ²)
   - Because the isochrony assumption is violated for them

4. MASS-TO-LIGHT RATIO ANOMALY
   - M/L should be systematically overestimated for high-Γ_t galaxies
   - This manifests as: at fixed MUV, high-Γ_t galaxies have higher M*
   - We can test this directly

5. SPECTROSCOPIC vs PHOTOMETRIC REDSHIFT
   - For galaxies with both z_spec and z_phot
   - The difference should correlate with Γ_t
   - Because TEP affects the photometric SED but not emission lines

Author: Matthew L. Smawfield
Date: January 2026
"""

import sys
import numpy as np
import pandas as pd
from scipy import stats
from astropy.io import fits
from pathlib import Path
import logging
import json

# =============================================================================
# LOGGER SETUP
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.p_value_utils import format_p_value
from scripts.utils.rank_stats import partial_rank_correlation
from scripts.utils.tep_model import compute_gamma_t

STEP_NUM = "017"
STEP_NAME = "ml_ratio"

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

# Note: TEPLogger is initialized above via set_step_logger()


def load_jades_data():
    """Load JADES data with age excess."""
    logger.info("Loading JADES data...")

    _path = DATA_DIR / "interim" / "jades_highz_physical.csv"
    if not _path.exists():
        logger.error("jades_highz_physical.csv not found. Run step_014 first.")
        return None
    df = pd.read_csv(_path)
    
    df['z_best'] = np.where(df['z_spec'] > 0, df['z_spec'], df['z_phot'])
    df['gamma_t'] = compute_gamma_t(df['log_Mhalo'], df['z_best'])
    
    logger.info(f"Loaded {len(df)} JADES galaxies")
    
    return df


def load_jades_raw():
    """Load raw JADES catalog with photo-z errors and χ²."""
    logger.info("Loading raw JADES catalog...")

    fits_path = DATA_DIR / "raw" / "JADES_z_gt_8_Candidates_Hainline_et_al.fits"
    if not fits_path.exists():
        logger.warning("Raw JADES FITS not found — raw-catalog tests will be skipped.")
        return None

    with fits.open(fits_path) as hdu:
        def to_native(arr):
            arr = np.array(arr)
            if hasattr(arr.dtype, 'byteorder') and arr.dtype.byteorder == '>':
                return arr.astype(arr.dtype.newbyteorder('='))
            return arr

        kron   = hdu['KRON'].data
        photoz = hdu['PHOTOZ'].data

        df_k = pd.DataFrame({
            'id':     to_native(kron['ID']),
            'F150W':  to_native(kron['F150W_KRON']),
        })
        df_p = pd.DataFrame({
            'id':         to_native(photoz['ID']),
            'z_phot':     to_native(photoz['EAZY_z_a']),
            'z_phot_lo':  to_native(photoz['EAZY_l68']),
            'z_phot_hi':  to_native(photoz['EAZY_u68']),
            'z_l95':      to_native(photoz['EAZY_l95']),
            'chi2_best':  to_native(photoz['EAZY_chisq_min']),
        })
        df = df_k.merge(df_p, on='id', how='inner')

    df['z_phot_err'] = (df['z_phot_hi'] - df['z_phot_lo']) / 2
    df['z_spec'] = np.nan  # DR2 photometry catalog has no spec-z
    df['z_best'] = df['z_phot']
    df['P_z_gt_7'] = (df['z_l95'] > 7).astype(float)
    df['chi2_lowz'] = np.nan   # not available in DR2
    df['delta_chi2'] = np.nan  # not available in DR2

    # Compute MUV from F150W
    from astropy.cosmology import Planck18 as _cosmo
    d_L_pc = np.array([_cosmo.luminosity_distance(zi).to('pc').value
                        for zi in df['z_best']])
    f_nJy = df['F150W'].values.astype(float)
    f_nJy = np.where(f_nJy > 0, f_nJy, np.nan)
    df['MUV'] = (-2.5 * np.log10(f_nJy * 1e-9 / 3631)
                 - 5.0 * np.log10(d_L_pc / 10.0)
                 + 2.5 * np.log10(1.0 + df['z_best'].values))

    # Use best redshift
    df['z_best'] = df['z_phot']
    
    # Estimate Γ_t from MUV (proxy for mass)
    # MUV ~ -2.5 log(L) ~ -2.5 log(M*) + const
    # Rough: log(M*) ~ -MUV/2.5 + 4
    df['log_Mstar_approx'] = -df['MUV'] / 2.5 + 4
    df['log_Mhalo_approx'] = df['log_Mstar_approx'] + 2
    df['gamma_t'] = compute_gamma_t(df['log_Mhalo_approx'], df['z_best'])
    
    logger.info(f"Loaded {len(df)} raw JADES candidates")
    
    return df


def analyze_age_excess(df):
    """
    TEST 1: Age Excess Distribution
    
    age_excess = t_stellar - t_cosmic
    Negative values mean stellar age > cosmic age (anomalous)
    
    TEP predicts: age_excess should be MORE NEGATIVE for high-Γ_t galaxies
    (because their stellar ages are inflated by TEP)
    """
    logger.info("=" * 70)
    logger.info("TEST 1: Age Excess Distribution")
    logger.info("=" * 70)
    
    valid = df.dropna(subset=['age_excess_Gyr', 'gamma_t'])
    
    logger.info(f"Sample size: N = {len(valid)}")
    logger.info(f"Age excess range: {valid['age_excess_Gyr'].min():.3f} to {valid['age_excess_Gyr'].max():.3f} Gyr")
    
    # Correlation
    rho, p_value = stats.spearmanr(valid['gamma_t'], valid['age_excess_Gyr'])
    p_value_fmt = format_p_value(p_value)
    
    logger.info(f"\nCorrelation (age_excess vs Γ_t):")
    logger.info(f"  Spearman ρ = {rho:.3f}, p = {p_value:.4f}")
    
    # TEP predicts negative correlation (higher Γ_t → more negative age_excess)
    if rho < 0 and (p_value_fmt is not None and p_value_fmt < 0.05):
        logger.info("\n✓ Higher Γ_t → more negative age_excess (TEP-consistent)")
        logger.info("  Correlation confirmed")
        tep_consistent = True
    elif rho < 0:
        logger.info("\n⚠ Trend in correct direction but not significant")
        tep_consistent = False
    else:
        logger.info("\n✗ No negative correlation detected")
        tep_consistent = False
    
    # Fraction of "anomalous" galaxies by Γ_t bin
    logger.info("\nFraction with age_excess < 0 by Γ_t bin:")
    gamma_bins = [0, 0.3, 0.5, 0.7, 1.0, 5.0]
    for i in range(len(gamma_bins) - 1):
        bin_data = valid[(valid['gamma_t'] >= gamma_bins[i]) & (valid['gamma_t'] < gamma_bins[i+1])]
        if len(bin_data) >= 5:
            frac_impossible = (bin_data['age_excess_Gyr'] < 0).mean() * 100
            logger.info(f"  Γ_t = [{gamma_bins[i]:.1f}, {gamma_bins[i+1]:.1f}): "
                       f"N = {len(bin_data)}, anomalous = {frac_impossible:.1f}%")
    
    return {
        'n_galaxies': len(valid),
        'spearman_rho': rho,
        'spearman_p': p_value_fmt,
        'tep_consistent': tep_consistent
    }


def analyze_photoz_scatter(df):
    """
    TEST 2: Photo-z Scatter
    
    TEP affects the SED shape, which should confuse photo-z fitting.
    High-Γ_t galaxies should have LARGER photo-z errors.
    """
    logger.info("=" * 70)
    logger.info("TEST 2: Photo-z Scatter")
    logger.info("=" * 70)
    
    valid = df.dropna(subset=['z_phot_err', 'gamma_t'])
    valid = valid[valid['z_phot_err'] > 0]
    
    logger.info(f"Sample size: N = {len(valid)}")
    
    # Correlation
    rho, p_value = stats.spearmanr(valid['gamma_t'], valid['z_phot_err'])
    p_value_fmt = format_p_value(p_value)
    
    logger.info(f"\nCorrelation (z_phot_err vs Γ_t):")
    logger.info(f"  Spearman ρ = {rho:.3f}, p = {p_value:.4f}")
    
    if rho > 0 and (p_value_fmt is not None and p_value_fmt < 0.05):
        logger.info("\n✓ Higher Γ_t → larger photo-z error (TEP-consistent)")
        logger.info("  TEP confuses the SED fitting")
        tep_consistent = True
    else:
        logger.info("\n⚠ No significant correlation")
        tep_consistent = False
    
    return {
        'n_galaxies': len(valid),
        'spearman_rho': rho,
        'spearman_p': p_value_fmt,
        'tep_consistent': tep_consistent
    }


def analyze_sed_fit_quality(df):
    """
    TEST 3: SED Fit Quality (χ²)
    
    Standard SED models assume isochrony. High-Γ_t galaxies should have
    WORSE fits (higher χ²) because the isochrony assumption is violated for them.
    """
    logger.info("=" * 70)
    logger.info("TEST 3: SED Fit Quality (χ²)")
    logger.info("=" * 70)
    
    valid = df.dropna(subset=['chi2_best', 'gamma_t'])
    valid = valid[valid['chi2_best'] > 0]
    
    logger.info(f"Sample size: N = {len(valid)}")
    logger.info(f"χ² range: {valid['chi2_best'].min():.1f} to {valid['chi2_best'].max():.1f}")
    
    # Correlation
    rho, p_value = stats.spearmanr(valid['gamma_t'], valid['chi2_best'])
    p_value_fmt = format_p_value(p_value)
    
    logger.info(f"\nCorrelation (χ² vs Γ_t):")
    logger.info(f"  Spearman ρ = {rho:.3f}, p = {p_value:.4f}")
    
    if rho > 0 and (p_value_fmt is not None and p_value_fmt < 0.05):
        logger.info("\n✓ Higher Γ_t → worse SED fit (TEP-consistent)")
        logger.info("  Standard models fail for TEP-affected galaxies")
        tep_consistent = True
    elif rho > 0:
        logger.info("\n⚠ Trend in correct direction but not significant")
        tep_consistent = False
    else:
        logger.info("\n✗ No positive correlation detected")
        tep_consistent = False
    
    return {
        'n_galaxies': len(valid),
        'spearman_rho': rho,
        'spearman_p': p_value_fmt,
        'tep_consistent': tep_consistent
    }


def analyze_mass_to_light(df):
    """
    TEST 4: Mass-to-Light Ratio Anomaly
    
    At fixed MUV (luminosity), high-Γ_t galaxies should have higher M*
    because TEP inflates the M/L ratio.
    """
    logger.info("=" * 70)
    logger.info("TEST 4: Mass-to-Light Ratio Anomaly")
    logger.info("=" * 70)
    
    valid = df.dropna(subset=['log_Mstar', 'MUV', 'gamma_t'])
    
    logger.info(f"Sample size: N = {len(valid)}")
    
    # Calculate M/L proxy: log(M*) - (-MUV/2.5)
    # Higher values = higher M/L
    valid = valid.copy()
    valid['ML_proxy'] = valid['log_Mstar'] + valid['MUV'] / 2.5
    
    # Correlation
    rho, p_value = stats.spearmanr(valid['gamma_t'], valid['ML_proxy'])
    p_value_fmt = format_p_value(p_value)
    
    logger.info(f"\nCorrelation (M/L proxy vs Γ_t):")
    logger.info(f"  Spearman ρ = {rho:.3f}, p = {p_value:.4f}")
    
    if rho > 0 and (p_value_fmt is not None and p_value_fmt < 0.05):
        logger.info("\n✓ Higher Γ_t → higher M/L (TEP-consistent)")
        logger.info("  TEP inflates mass-to-light ratios")
        tep_consistent = True
    else:
        logger.info("\n⚠ No significant correlation")
        tep_consistent = False
    
    # Partial correlation controlling for redshift
    rho_partial, p_partial, _ = partial_rank_correlation(
        valid['gamma_t'].values,
        valid['ML_proxy'].values,
        valid['z_best'].values,
    )
    logger.info(f"\nPartial correlation (controlling for z):")
    logger.info(f"  Spearman ρ = {rho_partial:.3f}, p = {p_partial:.4f}")
    
    return {
        'n_galaxies': len(valid),
        'spearman_rho': rho,
        'spearman_p': p_value_fmt,
        'rho_partial': rho_partial,
        'p_partial': format_p_value(p_partial),
        'tep_consistent': tep_consistent or (rho_partial > 0 and (format_p_value(p_partial) is not None and format_p_value(p_partial) < 0.05))
    }


def analyze_spec_phot_offset(df):
    """
    TEST 5: Spectroscopic vs Photometric Redshift
    
    For galaxies with both z_spec and z_phot, the difference should
    correlate with Γ_t because TEP affects the photometric SED but
    not emission lines.
    """
    logger.info("=" * 70)
    logger.info("TEST 5: Spec-z vs Photo-z Offset")
    logger.info("=" * 70)
    
    # Filter for galaxies with both z_spec and z_phot
    valid = df[(df['z_spec'] > 0) & (df['z_phot'] > 0)].copy()
    valid = valid.dropna(subset=['gamma_t'])
    
    logger.info(f"Galaxies with both z_spec and z_phot: N = {len(valid)}")
    
    if len(valid) < 10:
        logger.warning("Insufficient sample for spec-phot analysis")
        return None
    
    # Calculate offset
    valid['z_offset'] = valid['z_phot'] - valid['z_spec']
    valid['z_offset_abs'] = np.abs(valid['z_offset'])
    
    logger.info(f"Mean |z_offset|: {valid['z_offset_abs'].mean():.3f}")
    
    # Correlation with Γ_t
    rho, p_value = stats.spearmanr(valid['gamma_t'], valid['z_offset_abs'])
    p_value_fmt = format_p_value(p_value)
    
    logger.info(f"\nCorrelation (|z_offset| vs Γ_t):")
    logger.info(f"  Spearman ρ = {rho:.3f}, p = {p_value:.4f}")
    
    if rho > 0 and (p_value_fmt is not None and p_value_fmt < 0.1):
        logger.info("\n✓ Higher Γ_t → larger photo-z offset (TEP-consistent)")
        tep_consistent = True
    else:
        logger.info("\n⚠ No significant correlation")
        tep_consistent = False
    
    return {
        'n_galaxies': len(valid),
        'mean_offset': valid['z_offset_abs'].mean(),
        'spearman_rho': rho,
        'spearman_p': p_value_fmt,
        'tep_consistent': tep_consistent
    }


def run_ml_ratio_analysis():
    """Run the complete mass-to-light ratio analysis."""
    logger.info("=" * 70)
    logger.info(f"TEP-JWST Step {STEP_NUM}: Mass-to-Light Ratio Analysis")
    logger.info("=" * 70)
    logger.info("")

    # Load data
    df_jades = load_jades_data()
    df_raw   = load_jades_raw()

    if df_jades is None:
        logger.error(f"Step {STEP_NUM} aborted: jades_highz_physical.csv not available (run step_014 first).")
        return {"status": "aborted", "reason": "missing jades_highz_physical.csv"}

    results = {}
    
    # Test 1: Age Excess
    results['age_excess'] = analyze_age_excess(df_jades)
    
    # Test 2: Photo-z Scatter
    results['photoz_scatter'] = analyze_photoz_scatter(df_raw)
    
    # Test 3: SED Fit Quality
    results['sed_quality'] = analyze_sed_fit_quality(df_raw)
    
    # Test 4: Mass-to-Light Ratio
    results['mass_to_light'] = analyze_mass_to_light(df_jades)
    
    # Test 5: Spec-Phot Offset
    results['spec_phot'] = analyze_spec_phot_offset(df_raw)
    
    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    
    tests_passed = 0
    total_tests = 0
    
    for name, result in results.items():
        if result is not None:
            total_tests += 1
            if result.get('tep_consistent', False):
                tests_passed += 1
                logger.info(f"✓ {name}: TEP-consistent")
            else:
                logger.info(f"⚠ {name}: Not significant")
    
    logger.info(f"\nTests passed: {tests_passed}/{total_tests}")
    
    # Save results
    output_file = RESULTS_DIR / f"step_{STEP_NUM}_ml_ratio.json"
    
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
    run_ml_ratio_analysis()
