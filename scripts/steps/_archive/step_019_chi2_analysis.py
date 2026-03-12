#!/usr/bin/env python3
"""
TEP-JWST Step 19: Chi-Squared Correlation Analysis

This analysis examines chi-squared correlations as a TEP diagnostic.

This analysis explores additional datasets and predictions:

1. GALAXY SIZE-MASS RELATION
   - TEP predicts: high-Γ_t galaxies should appear more compact
   - Because M/L is inflated, the inferred size at fixed mass is smaller
   - Test: size residuals should correlate with Γ_t

2. PHOTO-Z UNCERTAINTY SCALING
   - TEP distorts the SED shape
   - High-Γ_t galaxies should have larger photo-z uncertainties
   - Test: σ(z) should correlate with Γ_t

3. SPECTROSCOPIC vs PHOTOMETRIC SAMPLES
   - Spec-z galaxies have independent redshift confirmation
   - Photo-z only galaxies may have TEP-induced biases
   - Test: Compare TEP signatures between samples

4. REDSHIFT DISTRIBUTION ANOMALY
   - If TEP affects photo-z, the redshift distribution should be distorted
   - High-Γ_t galaxies may be systematically placed at incorrect redshifts
   - Test: Compare z_spec - z_phot vs Γ_t

5. THE COMPACTNESS PARADOX
   - High-z massive galaxies are surprisingly compact
   - TEP explanation: their masses are overestimated, so they appear compact
   - Test: Compactness should correlate with Γ_t

Author: Matthew L. Smawfield
Date: January 2026
"""

import sys
import json
import urllib.request
import numpy as np
import pandas as pd
from scipy import stats
from astropy.io import fits
from pathlib import Path
import logging

# =============================================================================
# LOGGER SETUP
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.p_value_utils import format_p_value
from scripts.utils.tep_model import compute_gamma_t

STEP_NUM = "019"
STEP_NAME = "chi2_analysis"

LOGS_PATH = PROJECT_ROOT / "logs"
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"
LOGS_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")
set_step_logger(logger)



DATA_DIR    = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "outputs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# JADES DR2 GOODS-S Deep photometry v2.0 (45k+ sources)
# Ref: Rieke et al. 2023; Robertson et al. 2023
JADES_GOODSS_URL  = (
    "https://archive.stsci.edu/hlsps/jades/dr2/goods-s/catalogs/"
    "hlsp_jades_jwst_nircam_goods-s-deep_photometry_v2.0_catalog.fits"
)
JADES_GOODSS_FILE = DATA_DIR / "raw" / "hlsp_jades_jwst_nircam_goods-s-deep_photometry_v2.0_catalog.fits"
JADES_GOODSS_SIZE_MB_MIN = 200  # ~673 MB; truncated file <200 MB triggers re-download


def download_jades_goodss():
    """Download JADES GOODS-S Deep photometry catalog from MAST HLSP."""
    JADES_GOODSS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if JADES_GOODSS_FILE.exists():
        size_mb = JADES_GOODSS_FILE.stat().st_size / 1e6
        if size_mb >= JADES_GOODSS_SIZE_MB_MIN:
            logger.info(f"JADES GOODS-S catalog present ({size_mb:.0f} MB)")
            return True
        logger.info(f"Existing file too small ({size_mb:.1f} MB) — re-downloading.")
        JADES_GOODSS_FILE.unlink()
    logger.info("Downloading JADES GOODS-S Deep photometry v2.0 from MAST (~673 MB)...")
    logger.info(f"  URL: {JADES_GOODSS_URL}")
    try:
        def _prog(b, bs, tot):
            if tot > 0:
                print(f"\r  {min(100, b*bs*100//tot)}%", end="", flush=True)
        urllib.request.urlretrieve(JADES_GOODSS_URL, JADES_GOODSS_FILE, reporthook=_prog)
        print()
        logger.info(f"Download complete: {JADES_GOODSS_FILE.stat().st_size/1e6:.0f} MB")
        return True
    except Exception as exc:
        logger.error(f"Download failed: {exc}")
        logger.error("Manual: https://archive.stsci.edu/hlsps/jades/")
        if JADES_GOODSS_FILE.exists():
            JADES_GOODSS_FILE.unlink()
        return False


def load_jades_full_catalog():
    """Download (if needed) and load the full JADES GOODS-S catalog."""
    logger.info("Loading JADES GOODS-S Deep catalog...")

    if not JADES_GOODSS_FILE.exists() or JADES_GOODSS_FILE.stat().st_size / 1e6 < JADES_GOODSS_SIZE_MB_MIN:
        if not download_jades_goodss():
            logger.error("Cannot proceed without JADES GOODS-S catalog.")
            return None

    with fits.open(JADES_GOODSS_FILE) as hdu:
        # Get IDs and positions from FLAG extension
        flag_data = hdu[2].data
        
        # Get sizes from SIZE extension
        size_data = hdu[3].data
        
        # Get photo-z from PHOTOZ extension
        photoz_data = hdu[9].data
        
        # Get photometry from KRON extension
        kron_data = hdu[7].data
        
        def to_native(arr):
            arr = np.array(arr)
            if arr.dtype.byteorder == '>':
                return arr.astype(arr.dtype.newbyteorder('='))
            return arr
        
        df = pd.DataFrame({
            'id': to_native(flag_data['ID']),
            'ra': to_native(flag_data['RA']),
            'dec': to_native(flag_data['DEC']),
            'z_phot': to_native(photoz_data['EAZY_z_a']),
            'z_phot_lo': to_native(photoz_data['EAZY_l68']),
            'z_phot_hi': to_native(photoz_data['EAZY_u68']),
            'chi2_phot': to_native(photoz_data['EAZY_chisq_min']),
            'r_kron': to_native(size_data['R_KRON']),
            'npix': to_native(size_data['NPIX_DET']),
        })
        
        # Get F277W flux for magnitude estimate
        if 'F277W_FLUX' in [c.name for c in kron_data.columns]:
            df['f277w'] = to_native(kron_data['F277W_FLUX'])
        elif 'F277W' in [c.name for c in kron_data.columns]:
            df['f277w'] = to_native(kron_data['F277W'])
    
    # Calculate photo-z uncertainty
    df['z_phot_err'] = (df['z_phot_hi'] - df['z_phot_lo']) / 2
    
    # Filter for high-z candidates
    df = df[(df['z_phot'] > 4) & (df['z_phot'] < 15)].copy()
    df = df[df['r_kron'] > 0].copy()
    
    # Estimate stellar mass from photo-z and flux (rough approximation)
    # This is very approximate - just for testing
    if 'f277w' in df.columns:
        df = df[df['f277w'] > 0].copy()
        # Rough M* estimate: log(M*) ~ -0.4 * (m - 25) + 9 at z~7
        df['mag_f277w'] = -2.5 * np.log10(df['f277w']) + 28.9  # AB mag
        df['log_Mstar_approx'] = -0.4 * (df['mag_f277w'] - 25) + 9
        df['log_Mstar_approx'] = df['log_Mstar_approx'].clip(7, 12)
    else:
        # Use size as proxy for mass
        df['log_Mstar_approx'] = np.log10(df['r_kron'] * 100) + 8
        df['log_Mstar_approx'] = df['log_Mstar_approx'].clip(7, 12)
     
    # Calculate Γ_t
    df['log_Mhalo'] = df['log_Mstar_approx'] + 2
    df['gamma_t'] = compute_gamma_t(df['log_Mhalo'], df['z_phot'])
     
    logger.info(f"Loaded {len(df)} high-z candidates from JADES GOODS-S")
     
    return df


JADES_HAINLINE_FILE = DATA_DIR / "raw" / "JADES_z_gt_8_Candidates_Hainline_et_al.fits"


def load_jades_z8_candidates():
    """Download (if needed) and load JADES z>8 candidates."""
    logger.info("Loading JADES z>8 candidates (Hainline+23)...")

    if not JADES_HAINLINE_FILE.exists() or JADES_HAINLINE_FILE.stat().st_size < 1e6:
        if not download_jades_goodss():
            logger.error("Cannot proceed without JADES catalog.")
            return None

    with fits.open(JADES_HAINLINE_FILE) as hdu:
        def to_native(arr):
            arr = np.array(arr)
            if hasattr(arr.dtype, 'byteorder') and arr.dtype.byteorder == '>':
                return arr.astype(arr.dtype.newbyteorder('='))
            return arr

        kron   = hdu['KRON'].data
        photoz = hdu['PHOTOZ'].data

        df_k = pd.DataFrame({
            'id':  to_native(kron['ID']),
            'ra':  to_native(kron['RA']),
            'dec': to_native(kron['DEC']),
            'F150W': to_native(kron['F150W_KRON']),
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
    df['z_spec']     = np.nan
    df['z_best']     = df['z_phot']
    df['has_spec']   = False
    df['P_z_gt_7']   = (df['z_l95'] > 7).astype(float)
    df['delta_chi2'] = np.nan

    # Compute MUV from F150W
    from astropy.cosmology import Planck18 as _cosmo
    d_L_pc = np.array([_cosmo.luminosity_distance(zi).to('pc').value
                        for zi in df['z_best']])
    f_nJy = df['F150W'].values.astype(float)
    f_nJy = np.where(f_nJy > 0, f_nJy, np.nan)
    df['MUV'] = (-2.5 * np.log10(f_nJy * 1e-9 / 3631)
                 - 5.0 * np.log10(d_L_pc / 10.0)
                 + 2.5 * np.log10(1.0 + df['z_best'].values))
    
    # Estimate mass from MUV
    df['log_Mstar_approx'] = -df['MUV'] / 2.5 + 4
    df['log_Mhalo'] = df['log_Mstar_approx'] + 2
    df['gamma_t'] = compute_gamma_t(df['log_Mhalo'], df['z_best'])
    
    logger.info(f"Loaded {len(df)} z>8 candidates")
    logger.info(f"  With spec-z: {df['has_spec'].sum()}")
    
    return df


def analyze_photoz_uncertainty(df):
    """
    TEST 1: Photo-z Uncertainty Scaling
    
    TEP distorts the SED shape, which should confuse photo-z fitting.
    High-Γ_t galaxies should have larger photo-z uncertainties.
    """
    logger.info("=" * 70)
    logger.info("TEST 1: Photo-z Uncertainty Scaling")
    logger.info("=" * 70)
    
    valid = df.dropna(subset=['z_phot_err', 'gamma_t'])
    valid = valid[valid['z_phot_err'] > 0]
    
    logger.info(f"Sample size: N = {len(valid)}")
    
    # Correlation
    rho, p_value = stats.spearmanr(valid['gamma_t'], valid['z_phot_err'])
    p_value_fmt = format_p_value(p_value)
    
    logger.info(f"\nCorrelation (z_phot_err vs Γ_t):")
    logger.info(f"  Spearman ρ = {rho:.3f}, p = {p_value:.4f}")
    
    # TEP predicts positive correlation
    if rho > 0 and (p_value_fmt is not None and p_value_fmt < 0.05):
        logger.info("\n✓ Higher Γ_t → larger photo-z uncertainty (TEP-consistent)")
        tep_consistent = True
    elif rho > 0:
        logger.info("\n⚠ Positive trend but not significant")
        tep_consistent = False
    else:
        logger.info("\n⚠ No positive correlation")
        tep_consistent = False
    
    return {
        'n_galaxies': len(valid),
        'spearman_rho': rho,
        'spearman_p': p_value_fmt,
        'tep_consistent': tep_consistent
    }


def analyze_spec_phot_comparison(df):
    """
    TEST 2: Spectroscopic vs Photometric Comparison
    
    For galaxies with both z_spec and z_phot, the offset should
    correlate with Γ_t because TEP affects the photometric SED.
    """
    logger.info("=" * 70)
    logger.info("TEST 2: Spec-z vs Photo-z Offset")
    logger.info("=" * 70)
    
    # Filter for galaxies with spec-z
    spec = df[df['has_spec']].copy()
    
    logger.info(f"Galaxies with spec-z: N = {len(spec)}")
    
    if len(spec) < 10:
        logger.warning("Insufficient spec-z sample")
        return None
    
    # Calculate offset
    spec['z_offset'] = spec['z_phot'] - spec['z_spec']
    spec['z_offset_abs'] = np.abs(spec['z_offset'])
    
    logger.info(f"Mean z_offset: {spec['z_offset'].mean():.3f}")
    logger.info(f"Mean |z_offset|: {spec['z_offset_abs'].mean():.3f}")
    
    # Correlation with Γ_t
    rho, p_value = stats.spearmanr(spec['gamma_t'], spec['z_offset_abs'])
    p_value_fmt = format_p_value(p_value)
    
    logger.info(f"\nCorrelation (|z_offset| vs Γ_t):")
    logger.info(f"  Spearman ρ = {rho:.3f}, p = {p_value:.4f}")
    
    # Also test signed offset
    rho_signed, p_signed = stats.spearmanr(spec['gamma_t'], spec['z_offset'])
    p_signed_fmt = format_p_value(p_signed)
    logger.info(f"\nSigned correlation (z_offset vs Γ_t):")
    logger.info(f"  Spearman ρ = {rho_signed:.3f}, p = {p_signed:.4f}")
    
    # Compare TEP signatures between spec and phot-only samples
    phot_only = df[~df['has_spec']].copy()
    
    if len(phot_only) >= 30:
        # Compare mean Γ_t
        mean_gamma_spec = spec['gamma_t'].mean()
        mean_gamma_phot = phot_only['gamma_t'].mean()
        
        t_stat, p_diff = stats.ttest_ind(spec['gamma_t'], phot_only['gamma_t'])
        
        logger.info(f"\nSpec vs Phot-only comparison:")
        logger.info(f"  Mean Γ_t (spec): {mean_gamma_spec:.3f}")
        logger.info(f"  Mean Γ_t (phot-only): {mean_gamma_phot:.3f}")
        logger.info(f"  Difference p-value: {p_diff:.4f}")
    
    return {
        'n_spec': len(spec),
        'mean_z_offset': spec['z_offset'].mean(),
        'mean_z_offset_abs': spec['z_offset_abs'].mean(),
        'rho_abs': rho,
        'p_abs': p_value_fmt,
        'rho_signed': rho_signed,
        'p_signed': p_signed_fmt
    }


def analyze_chi2_scaling(df):
    """
    TEST 3: χ² Scaling with Γ_t
    
    Standard SED models assume isochrony. High-Γ_t galaxies should
    have worse fits (higher χ²) because the isochrony assumption is violated for them.
    """
    logger.info("=" * 70)
    logger.info("TEST 3: χ² Scaling with Γ_t")
    logger.info("=" * 70)
    
    chi2_col = 'chi2_best' if 'chi2_best' in df.columns else 'chi2_phot'
    
    valid = df.dropna(subset=[chi2_col, 'gamma_t'])
    valid = valid[valid[chi2_col] > 0]
    
    logger.info(f"Sample size: N = {len(valid)}")
    
    # Correlation
    rho, p_value = stats.spearmanr(valid['gamma_t'], valid[chi2_col])
    p_value_fmt = format_p_value(p_value)
    
    logger.info(f"\nCorrelation (χ² vs Γ_t):")
    logger.info(f"  Spearman ρ = {rho:.3f}, p = {p_value:.4f}")
    
    # Bin analysis
    logger.info("\nBinned analysis:")
    gamma_bins = [0, 0.3, 0.5, 0.7, 1.0, 5.0]
    for i in range(len(gamma_bins) - 1):
        bin_data = valid[(valid['gamma_t'] >= gamma_bins[i]) & (valid['gamma_t'] < gamma_bins[i+1])]
        if len(bin_data) >= 10:
            mean_chi2 = bin_data[chi2_col].mean()
            median_chi2 = bin_data[chi2_col].median()
            logger.info(f"  Γ_t = [{gamma_bins[i]:.1f}, {gamma_bins[i+1]:.1f}): "
                       f"N = {len(bin_data)}, mean χ² = {mean_chi2:.1f}, median = {median_chi2:.1f}")
    
    if rho > 0 and (p_value_fmt is not None and p_value_fmt < 0.05):
        logger.info("\n✓ Higher Γ_t → worse SED fit (TEP-consistent)")
        tep_consistent = True
    else:
        logger.info("\n⚠ No significant positive correlation")
        tep_consistent = False
    
    return {
        'n_galaxies': len(valid),
        'spearman_rho': rho,
        'spearman_p': p_value_fmt,
        'tep_consistent': tep_consistent
    }


def analyze_size_mass_relation(df):
    """
    TEST 4: Size-Mass Relation Anomaly
    
    TEP predicts that high-Γ_t galaxies should appear more compact
    because their masses are overestimated.
    
    At fixed apparent mass, high-Γ_t galaxies should have smaller sizes.
    """
    logger.info("=" * 70)
    logger.info("TEST 4: Size-Mass Relation")
    logger.info("=" * 70)
    
    if 'r_kron' not in df.columns:
        logger.warning("No size data available")
        return None
    
    valid = df.dropna(subset=['r_kron', 'log_Mstar_approx', 'gamma_t'])
    valid = valid[valid['r_kron'] > 0]
    
    logger.info(f"Sample size: N = {len(valid)}")
    
    # Calculate size residuals from mass-size relation
    # Fit log(r) = a × log(M*) + b
    slope, intercept, _, _, _ = stats.linregress(
        valid['log_Mstar_approx'], np.log10(valid['r_kron'])
    )
    
    valid = valid.copy()
    valid['log_r_expected'] = slope * valid['log_Mstar_approx'] + intercept
    valid['size_residual'] = np.log10(valid['r_kron']) - valid['log_r_expected']
    
    logger.info(f"Size-mass relation: log(r) = {slope:.3f} × log(M*) + {intercept:.3f}")
    
    # Correlation between size residual and Γ_t
    rho, p_value = stats.spearmanr(valid['gamma_t'], valid['size_residual'])
    p_value_fmt = format_p_value(p_value)
    
    logger.info(f"\nCorrelation (size_residual vs Γ_t):")
    logger.info(f"  Spearman ρ = {rho:.3f}, p = {p_value:.4f}")
    
    # TEP predicts NEGATIVE correlation (high Γ_t → smaller than expected)
    if rho < 0 and (p_value_fmt is not None and p_value_fmt < 0.05):
        logger.info("\n✓ Higher Γ_t → smaller than expected (TEP-consistent)")
        logger.info("  This is the 'compactness paradox' explained by TEP")
        tep_consistent = True
    elif rho < 0:
        logger.info("\n⚠ Negative trend but not significant")
        tep_consistent = False
    else:
        logger.info("\n⚠ No negative correlation")
        tep_consistent = False
    
    return {
        'n_galaxies': len(valid),
        'size_mass_slope': slope,
        'size_mass_intercept': intercept,
        'spearman_rho': rho,
        'spearman_p': p_value_fmt,
        'tep_consistent': tep_consistent
    }


def analyze_delta_chi2_diagnostic(df):
    """
    TEST 5: Δχ² as TEP Diagnostic
    
    Δχ² = χ²(z<7) - χ²(z_best) measures how much better the high-z
    solution is compared to low-z. Under TEP, high-Γ_t galaxies should
    have LARGER Δχ² because the TEP-distorted SED is more distinct.
    """
    logger.info("=" * 70)
    logger.info("TEST 5: Δχ² Diagnostic")
    logger.info("=" * 70)
    
    if 'delta_chi2' not in df.columns:
        logger.warning("No Δχ² data available")
        return None
    
    valid = df.dropna(subset=['delta_chi2', 'gamma_t'])
    valid = valid[valid['delta_chi2'] > 0]

    logger.info(f"Sample size: N = {len(valid)}")

    if len(valid) < 10:
        logger.warning("Insufficient valid Δχ² data — skipping Test 5.")
        return None

    # Correlation
    rho, p_value = stats.spearmanr(valid['gamma_t'], valid['delta_chi2'])
    p_value_fmt = format_p_value(p_value)
    
    logger.info(f"\nCorrelation (Δχ² vs Γ_t):")
    logger.info(f"  Spearman ρ = {rho:.3f}, p = {p_value:.4f}")
    
    # Bin analysis
    logger.info("\nBinned analysis:")
    gamma_bins = [0, 0.3, 0.5, 0.7, 1.0, 5.0]
    for i in range(len(gamma_bins) - 1):
        bin_data = valid[(valid['gamma_t'] >= gamma_bins[i]) & (valid['gamma_t'] < gamma_bins[i+1])]
        if len(bin_data) >= 5:
            mean_dchi2 = bin_data['delta_chi2'].mean()
            logger.info(f"  Γ_t = [{gamma_bins[i]:.1f}, {gamma_bins[i+1]:.1f}): "
                       f"N = {len(bin_data)}, mean Δχ² = {mean_dchi2:.1f}")
    
    if rho > 0 and (p_value_fmt is not None and p_value_fmt < 0.05):
        logger.info("\n✓ Higher Γ_t → larger Δχ² (TEP-consistent)")
        tep_consistent = True
    else:
        logger.info("\n⚠ No significant positive correlation")
        tep_consistent = False
    
    return {
        'n_galaxies': len(valid),
        'spearman_rho': rho,
        'spearman_p': p_value_fmt,
        'tep_consistent': tep_consistent
    }


def run_chi2_analysis():
    """Run the complete chi-squared correlation analysis."""
    logger.info("=" * 70)
    logger.info(f"TEP-JWST Step {STEP_NUM}: Chi-Squared Correlation Analysis")
    logger.info("=" * 70)

    results = {}

    # Load JADES z>8 candidates (has more detailed properties)
    df_z8 = load_jades_z8_candidates()
    if df_z8 is None:
        logger.error(f"Step {STEP_NUM} aborted: JADES data unavailable.")
        return {"status": "aborted", "reason": "JADES catalog unavailable"}

    # Test 1: Photo-z uncertainty
    results['photoz_uncertainty'] = analyze_photoz_uncertainty(df_z8)
    
    # Test 2: Spec vs Phot comparison
    results['spec_phot'] = analyze_spec_phot_comparison(df_z8)
    
    # Test 3: χ² scaling
    results['chi2_scaling'] = analyze_chi2_scaling(df_z8)
    
    # Test 5: Δχ² diagnostic
    results['delta_chi2'] = analyze_delta_chi2_diagnostic(df_z8)
    
    # Try to load full catalog for size analysis
    try:
        df_full = load_jades_full_catalog()
        results['size_mass'] = analyze_size_mass_relation(df_full)
    except Exception as e:
        logger.warning(f"Could not load full catalog: {e}")
        results['size_mass'] = None
    
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
    output_file = RESULTS_DIR / f"step_{STEP_NUM}_chi2_analysis.json"
    
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
    run_chi2_analysis()
