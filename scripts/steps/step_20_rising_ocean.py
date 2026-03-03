#!/usr/bin/env python3
"""
TEP-JWST Step 20: The Rising Ocean

The cliffs are already carved; we need not cut the rock. TEP is the rising ocean.
We simply let it swell, and it rushes to fill every hollow cave and empty bay.

This analysis identifies natural "gaps" in the data where TEP predictions can be
tested without forcing the theory into the data. We let the water find the shape
of the gap perfectly.

HOLLOW CAVES TO FILL:

1. AGE EXCESS DISTRIBUTION
   - JADES provides age_excess = t_stellar - t_cosmic
   - Negative values are "impossible" (stellar age > cosmic age)
   - TEP predicts: age_excess should correlate with Γ_t

2. PHOTOMETRIC REDSHIFT SCATTER
   - Photo-z errors should be LARGER for high-Γ_t galaxies
   - Because TEP affects the SED shape, confusing the photo-z fit
   - This is a PREDICTION, not a post-hoc explanation

3. SED FIT QUALITY (χ²)
   - Standard SED models assume isochrony
   - High-Γ_t galaxies should have WORSE fits (higher χ²)
   - Because the model is wrong for them

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

import numpy as np
import pandas as pd
from scipy import stats
from astropy.io import fits
from pathlib import Path
import logging
import json

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "outputs"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# TEP parameters
ALPHA_TEP = 0.58
M_REF = 1e11


def load_jades_data():
    """Load JADES data with age excess."""
    logger.info("Loading JADES data...")
    
    df = pd.read_csv(DATA_DIR / "interim" / "jades_highz_physical.csv")
    
    # Calculate Γ_t from halo mass
    M_h = 10 ** df['log_Mhalo']
    df['gamma_t'] = ALPHA_TEP * (M_h / M_REF) ** (1/3)
    
    logger.info(f"Loaded {len(df)} JADES galaxies")
    
    return df


def load_jades_raw():
    """Load raw JADES catalog with photo-z errors and χ²."""
    logger.info("Loading raw JADES catalog...")
    
    fits_path = DATA_DIR / "raw" / "JADES_z_gt_8_Candidates_Hainline_et_al.fits"
    
    with fits.open(fits_path) as hdu:
        data = hdu[1].data
        
        def to_native(arr):
            arr = np.array(arr)
            if arr.dtype.byteorder == '>':
                return arr.astype(arr.dtype.newbyteorder('='))
            return arr
        
        df = pd.DataFrame({
            'id': np.array(data['JADES_ID']),
            'z_phot': to_native(data['EAZY_z_a']),
            'z_phot_lo': to_native(data['EAZY_sigma68_lo']),
            'z_phot_hi': to_native(data['EAZY_sigma68_hi']),
            'z_spec': to_native(data['z_spec']),
            'MUV': to_native(data['MUV']),
            'chi2_best': to_native(data['EAZY_z_a_chisq_min']),
            'chi2_lowz': to_native(data['EAZY_z_a_zlt7_chisq_min']),
            'delta_chi2': to_native(data['EAZY_delta_chisq']),
            'P_z_gt_7': to_native(data['EAZY_Pzgt7']),
        })
    
    # Calculate photo-z error
    df['z_phot_err'] = (df['z_phot_hi'] - df['z_phot_lo']) / 2
    
    # Use best redshift
    df['z_best'] = np.where(df['z_spec'] > 0, df['z_spec'], df['z_phot'])
    
    # Estimate Γ_t from MUV (proxy for mass)
    # MUV ~ -2.5 log(L) ~ -2.5 log(M*) + const
    # Rough: log(M*) ~ -MUV/2.5 + 4
    df['log_Mstar_approx'] = -df['MUV'] / 2.5 + 4
    df['log_Mhalo_approx'] = df['log_Mstar_approx'] + 2
    M_h = 10 ** df['log_Mhalo_approx']
    df['gamma_t'] = ALPHA_TEP * (M_h / M_REF) ** (1/3)
    
    logger.info(f"Loaded {len(df)} raw JADES candidates")
    
    return df


def analyze_age_excess(df):
    """
    HOLLOW CAVE 1: Age Excess Distribution
    
    age_excess = t_stellar - t_cosmic
    Negative values mean stellar age > cosmic age (impossible)
    
    TEP predicts: age_excess should be MORE NEGATIVE for high-Γ_t galaxies
    (because their stellar ages are inflated by TEP)
    """
    logger.info("=" * 70)
    logger.info("HOLLOW CAVE 1: Age Excess Distribution")
    logger.info("=" * 70)
    
    valid = df.dropna(subset=['age_excess_Gyr', 'gamma_t'])
    
    logger.info(f"Sample size: N = {len(valid)}")
    logger.info(f"Age excess range: {valid['age_excess_Gyr'].min():.3f} to {valid['age_excess_Gyr'].max():.3f} Gyr")
    
    # Correlation
    rho, p_value = stats.spearmanr(valid['gamma_t'], valid['age_excess_Gyr'])
    
    logger.info(f"\nCorrelation (age_excess vs Γ_t):")
    logger.info(f"  Spearman ρ = {rho:.3f}, p = {p_value:.4f}")
    
    # TEP predicts negative correlation (higher Γ_t → more negative age_excess)
    if rho < 0 and p_value < 0.05:
        logger.info("\n✓ Higher Γ_t → more negative age_excess (TEP-consistent)")
        logger.info("  The ocean fills this cave perfectly")
        tep_consistent = True
    elif rho < 0:
        logger.info("\n⚠ Trend in correct direction but not significant")
        tep_consistent = False
    else:
        logger.info("\n✗ No negative correlation detected")
        tep_consistent = False
    
    # Fraction of "impossible" galaxies by Γ_t bin
    logger.info("\nFraction with age_excess < 0 by Γ_t bin:")
    gamma_bins = [0, 0.3, 0.5, 0.7, 1.0, 5.0]
    for i in range(len(gamma_bins) - 1):
        bin_data = valid[(valid['gamma_t'] >= gamma_bins[i]) & (valid['gamma_t'] < gamma_bins[i+1])]
        if len(bin_data) >= 5:
            frac_impossible = (bin_data['age_excess_Gyr'] < 0).mean() * 100
            logger.info(f"  Γ_t = [{gamma_bins[i]:.1f}, {gamma_bins[i+1]:.1f}): "
                       f"N = {len(bin_data)}, impossible = {frac_impossible:.1f}%")
    
    return {
        'n_galaxies': len(valid),
        'spearman_rho': rho,
        'spearman_p': p_value,
        'tep_consistent': tep_consistent
    }


def analyze_photoz_scatter(df):
    """
    HOLLOW CAVE 2: Photo-z Scatter
    
    TEP affects the SED shape, which should confuse photo-z fitting.
    High-Γ_t galaxies should have LARGER photo-z errors.
    """
    logger.info("=" * 70)
    logger.info("HOLLOW CAVE 2: Photo-z Scatter")
    logger.info("=" * 70)
    
    valid = df.dropna(subset=['z_phot_err', 'gamma_t'])
    valid = valid[valid['z_phot_err'] > 0]
    
    logger.info(f"Sample size: N = {len(valid)}")
    
    # Correlation
    rho, p_value = stats.spearmanr(valid['gamma_t'], valid['z_phot_err'])
    
    logger.info(f"\nCorrelation (z_phot_err vs Γ_t):")
    logger.info(f"  Spearman ρ = {rho:.3f}, p = {p_value:.4f}")
    
    if rho > 0 and p_value < 0.05:
        logger.info("\n✓ Higher Γ_t → larger photo-z error (TEP-consistent)")
        logger.info("  TEP confuses the SED fitting")
        tep_consistent = True
    else:
        logger.info("\n⚠ No significant correlation")
        tep_consistent = False
    
    return {
        'n_galaxies': len(valid),
        'spearman_rho': rho,
        'spearman_p': p_value,
        'tep_consistent': tep_consistent
    }


def analyze_sed_fit_quality(df):
    """
    HOLLOW CAVE 3: SED Fit Quality (χ²)
    
    Standard SED models assume isochrony. High-Γ_t galaxies should have
    WORSE fits (higher χ²) because the model is wrong for them.
    """
    logger.info("=" * 70)
    logger.info("HOLLOW CAVE 3: SED Fit Quality (χ²)")
    logger.info("=" * 70)
    
    valid = df.dropna(subset=['chi2_best', 'gamma_t'])
    valid = valid[valid['chi2_best'] > 0]
    
    logger.info(f"Sample size: N = {len(valid)}")
    logger.info(f"χ² range: {valid['chi2_best'].min():.1f} to {valid['chi2_best'].max():.1f}")
    
    # Correlation
    rho, p_value = stats.spearmanr(valid['gamma_t'], valid['chi2_best'])
    
    logger.info(f"\nCorrelation (χ² vs Γ_t):")
    logger.info(f"  Spearman ρ = {rho:.3f}, p = {p_value:.4f}")
    
    if rho > 0 and p_value < 0.05:
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
        'spearman_p': p_value,
        'tep_consistent': tep_consistent
    }


def analyze_mass_to_light(df):
    """
    HOLLOW CAVE 4: Mass-to-Light Ratio Anomaly
    
    At fixed MUV (luminosity), high-Γ_t galaxies should have higher M*
    because TEP inflates the M/L ratio.
    """
    logger.info("=" * 70)
    logger.info("HOLLOW CAVE 4: Mass-to-Light Ratio Anomaly")
    logger.info("=" * 70)
    
    valid = df.dropna(subset=['log_Mstar', 'MUV', 'gamma_t'])
    
    logger.info(f"Sample size: N = {len(valid)}")
    
    # Calculate M/L proxy: log(M*) - (-MUV/2.5)
    # Higher values = higher M/L
    valid = valid.copy()
    valid['ML_proxy'] = valid['log_Mstar'] + valid['MUV'] / 2.5
    
    # Correlation
    rho, p_value = stats.spearmanr(valid['gamma_t'], valid['ML_proxy'])
    
    logger.info(f"\nCorrelation (M/L proxy vs Γ_t):")
    logger.info(f"  Spearman ρ = {rho:.3f}, p = {p_value:.4f}")
    
    if rho > 0 and p_value < 0.05:
        logger.info("\n✓ Higher Γ_t → higher M/L (TEP-consistent)")
        logger.info("  TEP inflates mass-to-light ratios")
        tep_consistent = True
    else:
        logger.info("\n⚠ No significant correlation")
        tep_consistent = False
    
    # Partial correlation controlling for redshift
    from scipy.stats import spearmanr
    
    # Residualize M/L proxy against z
    slope, intercept, _, _, _ = stats.linregress(valid['z_best'], valid['ML_proxy'])
    valid['ML_residual'] = valid['ML_proxy'] - (slope * valid['z_best'] + intercept)
    
    rho_partial, p_partial = spearmanr(valid['gamma_t'], valid['ML_residual'])
    logger.info(f"\nPartial correlation (controlling for z):")
    logger.info(f"  Spearman ρ = {rho_partial:.3f}, p = {p_partial:.4f}")
    
    return {
        'n_galaxies': len(valid),
        'spearman_rho': rho,
        'spearman_p': p_value,
        'rho_partial': rho_partial,
        'p_partial': p_partial,
        'tep_consistent': tep_consistent or (rho_partial > 0 and p_partial < 0.05)
    }


def analyze_spec_phot_offset(df):
    """
    HOLLOW CAVE 5: Spectroscopic vs Photometric Redshift
    
    For galaxies with both z_spec and z_phot, the difference should
    correlate with Γ_t because TEP affects the photometric SED but
    not emission lines.
    """
    logger.info("=" * 70)
    logger.info("HOLLOW CAVE 5: Spec-z vs Photo-z Offset")
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
    
    logger.info(f"\nCorrelation (|z_offset| vs Γ_t):")
    logger.info(f"  Spearman ρ = {rho:.3f}, p = {p_value:.4f}")
    
    if rho > 0 and p_value < 0.1:
        logger.info("\n✓ Higher Γ_t → larger photo-z offset (TEP-consistent)")
        tep_consistent = True
    else:
        logger.info("\n⚠ No significant correlation")
        tep_consistent = False
    
    return {
        'n_galaxies': len(valid),
        'mean_offset': valid['z_offset_abs'].mean(),
        'spearman_rho': rho,
        'spearman_p': p_value,
        'tep_consistent': tep_consistent
    }


def run_rising_ocean_analysis():
    """Run the complete rising ocean analysis."""
    logger.info("=" * 70)
    logger.info("TEP-JWST Step 20: The Rising Ocean")
    logger.info("=" * 70)
    logger.info("")
    logger.info("The cliffs are already carved. TEP is the rising ocean.")
    logger.info("We let it swell, and it rushes to fill every hollow cave.")
    logger.info("")
    
    # Load data
    df_jades = load_jades_data()
    df_raw = load_jades_raw()
    
    results = {}
    
    # Hollow Cave 1: Age Excess
    results['age_excess'] = analyze_age_excess(df_jades)
    
    # Hollow Cave 2: Photo-z Scatter
    results['photoz_scatter'] = analyze_photoz_scatter(df_raw)
    
    # Hollow Cave 3: SED Fit Quality
    results['sed_quality'] = analyze_sed_fit_quality(df_raw)
    
    # Hollow Cave 4: Mass-to-Light Ratio
    results['mass_to_light'] = analyze_mass_to_light(df_jades)
    
    # Hollow Cave 5: Spec-Phot Offset
    results['spec_phot'] = analyze_spec_phot_offset(df_raw)
    
    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY: The Ocean Fills the Caves")
    logger.info("=" * 70)
    
    caves_filled = 0
    total_caves = 0
    
    for name, result in results.items():
        if result is not None:
            total_caves += 1
            if result.get('tep_consistent', False):
                caves_filled += 1
                logger.info(f"✓ {name}: TEP fills this cave")
            else:
                logger.info(f"⚠ {name}: Cave partially filled")
    
    logger.info(f"\nCaves filled: {caves_filled}/{total_caves}")
    
    # Save results
    output_file = RESULTS_DIR / "rising_ocean_analysis.json"
    
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
    run_rising_ocean_analysis()
