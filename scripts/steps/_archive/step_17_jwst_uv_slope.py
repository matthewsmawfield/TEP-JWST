#!/usr/bin/env python3
"""
TEP-JWST Step 17: UV Slope (β) Analysis

The UV spectral slope β is defined as f_λ ∝ λ^β, typically measured between
rest-frame 1500-2500 Å. It correlates with:
- Dust attenuation (redder = more dust)
- Stellar age (older = redder)
- Metallicity (higher Z = redder)

TEP Prediction:
At fixed dust content, massive galaxies should have REDDER UV slopes because
their stellar populations appear older due to enhanced proper time.

The UV slope can be measured from JWST photometry by fitting a power law to
rest-frame UV bands.

Author: Matthew L. Smawfield
Date: January 2026
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
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

# =============================================================================
# TEP PARAMETERS
# =============================================================================
ALPHA_TEP = 0.58
M_REF = 1e11  # Solar masses


def load_jades_photometry():
    """Load JADES z>8 candidates with photometry."""
    logger.info("Loading JADES z>8 candidates...")
    
    fits_path = DATA_DIR / "raw" / "JADES_z_gt_8_Candidates_Hainline_et_al.fits"
    
    with fits.open(fits_path) as hdu:
        data = hdu[1].data
        
        # Helper to convert big-endian to native byte order
        def to_native(arr):
            arr = np.array(arr)
            if arr.dtype.byteorder == '>':
                return arr.astype(arr.dtype.newbyteorder('='))
            return arr
        
        # Extract relevant columns (convert to native byte order)
        df = pd.DataFrame({
            'id': np.array(data['JADES_ID']),
            'ra': to_native(data['RA']),
            'dec': to_native(data['DEC']),
            'z_phot': to_native(data['EAZY_z_a']),
            'z_spec': to_native(data['z_spec']),
            'MUV': to_native(data['MUV']),
            'P_z_gt_7': to_native(data['EAZY_Pzgt7']),
            # NIRCam fluxes (in nJy)
            'F115W': to_native(data['NRC_F115W_flux']),
            'F115W_err': to_native(data['NRC_F115W_flux_err']),
            'F150W': to_native(data['NRC_F150W_flux']),
            'F150W_err': to_native(data['NRC_F150W_flux_err']),
            'F200W': to_native(data['NRC_F200W_flux']),
            'F200W_err': to_native(data['NRC_F200W_flux_err']),
            'F277W': to_native(data['NRC_F277W_flux']),
            'F277W_err': to_native(data['NRC_F277W_flux_err']),
            'F356W': to_native(data['NRC_F356W_flux']),
            'F356W_err': to_native(data['NRC_F356W_flux_err']),
            'F444W': to_native(data['NRC_F444W_flux']),
            'F444W_err': to_native(data['NRC_F444W_flux_err']),
        })
    
    # Use spec-z if available, else phot-z
    df['z_best'] = np.where(df['z_spec'] > 0, df['z_spec'], df['z_phot'])
    
    # Filter for high-confidence z > 8
    df = df[(df['z_best'] > 8) & (df['P_z_gt_7'] > 0.5)].copy()
    
    logger.info(f"Loaded {len(df)} high-confidence z > 8 candidates")
    
    return df


def calculate_uv_slope(df):
    """
    Calculate UV slope β from photometry.
    
    For z ~ 8-12, rest-frame UV (1500-2500 Å) falls in:
    - z=8: observed 1.35-2.25 μm (F150W, F200W)
    - z=10: observed 1.65-2.75 μm (F200W, F277W)
    - z=12: observed 1.95-3.25 μm (F200W, F277W, F356W)
    
    We use a simple two-band slope: β = (log(f1/f2)) / (log(λ1/λ2)) - 2
    """
    logger.info("Calculating UV slopes...")
    
    # Central wavelengths in μm
    wavelengths = {
        'F115W': 1.154,
        'F150W': 1.501,
        'F200W': 1.989,
        'F277W': 2.762,
        'F356W': 3.568,
        'F444W': 4.421,
    }
    
    beta_values = []
    beta_errors = []
    
    for _, row in df.iterrows():
        z = row['z_best']
        
        # Select bands that probe rest-frame UV (1500-2500 Å)
        # Rest-frame λ = observed λ / (1+z)
        rest_uv_min = 0.15  # μm (1500 Å)
        rest_uv_max = 0.30  # μm (3000 Å)
        
        # Find bands in rest-frame UV
        uv_bands = []
        for band, obs_wave in wavelengths.items():
            rest_wave = obs_wave / (1 + z)
            if rest_uv_min < rest_wave < rest_uv_max:
                flux = row[band]
                flux_err = row[f'{band}_err']
                if flux > 0 and flux_err > 0 and flux / flux_err > 2:
                    uv_bands.append((band, obs_wave, flux, flux_err))
        
        if len(uv_bands) >= 2:
            # Use first and last UV bands for slope
            band1, wave1, flux1, err1 = uv_bands[0]
            band2, wave2, flux2, err2 = uv_bands[-1]
            
            # β = d(log f_λ) / d(log λ)
            # For f_ν (which is what we have), f_λ ∝ f_ν * λ^2
            # So β_λ = β_ν + 2
            
            log_flux_ratio = np.log10(flux2 / flux1)
            log_wave_ratio = np.log10(wave2 / wave1)
            
            beta = log_flux_ratio / log_wave_ratio - 2
            
            # Error propagation
            rel_err1 = err1 / flux1
            rel_err2 = err2 / flux2
            log_flux_err = np.sqrt(rel_err1**2 + rel_err2**2) / np.log(10)
            beta_err = log_flux_err / abs(log_wave_ratio)
            
            beta_values.append(beta)
            beta_errors.append(beta_err)
        else:
            beta_values.append(np.nan)
            beta_errors.append(np.nan)
    
    df['beta'] = beta_values
    df['beta_err'] = beta_errors
    
    valid = df['beta'].notna().sum()
    logger.info(f"Calculated UV slopes for {valid} galaxies")
    
    return df


def calculate_tep_parameters(df):
    """Calculate TEP parameters for each galaxy."""
    logger.info("Calculating TEP parameters...")
    
    # Estimate halo mass from stellar mass (abundance matching)
    # log(M_h) ≈ log(M*) + 2 at high-z
    df['log_Mhalo'] = df['MUV'] / (-2.5) + 4  # Rough M* from MUV
    df['log_Mhalo'] = np.clip(df['log_Mhalo'] + 2, 10, 14)
    
    # Calculate Gamma_t
    M_h = 10 ** df['log_Mhalo']
    df['gamma_t'] = ALPHA_TEP * (M_h / M_REF) ** (1/3)
    
    return df


def analyze_beta_mass_correlation(df):
    """
    Test for correlation between UV slope and mass.
    
    Standard physics: β correlates with dust (more massive = more dust = redder)
    TEP adds: β also correlates with age (more massive = older = redder)
    
    The TEP contribution should be visible as EXCESS reddening at high mass.
    """
    logger.info("=" * 70)
    logger.info("ANALYSIS 1: UV Slope vs Mass Correlation")
    logger.info("=" * 70)
    
    valid = df.dropna(subset=['beta', 'MUV'])
    
    logger.info(f"Sample size: N = {len(valid)}")
    logger.info(f"β range: {valid['beta'].min():.2f} to {valid['beta'].max():.2f}")
    logger.info(f"MUV range: {valid['MUV'].min():.2f} to {valid['MUV'].max():.2f}")
    
    # Correlation with MUV (brighter = more massive)
    # Note: MUV is negative, so more negative = brighter = more massive
    rho, p_value = stats.spearmanr(valid['MUV'], valid['beta'])
    
    logger.info(f"\nCorrelation (β vs MUV):")
    logger.info(f"  Spearman ρ = {rho:.3f}, p = {p_value:.4f}")
    
    # MUV is negative, so negative correlation means brighter (more massive) = redder
    if rho < 0:
        logger.info(f"  → Brighter (more massive) galaxies have REDDER UV slopes")
        logger.info(f"  → Consistent with TEP: enhanced age in massive systems")
    else:
        logger.info(f"  → No mass-dependent reddening detected")
    
    # Bin by MUV
    logger.info(f"\nBinned analysis:")
    muv_bins = [-22, -20, -19, -18, -17]
    for i in range(len(muv_bins) - 1):
        bin_data = valid[(valid['MUV'] >= muv_bins[i]) & (valid['MUV'] < muv_bins[i+1])]
        if len(bin_data) >= 3:
            mean_beta = bin_data['beta'].mean()
            sem_beta = bin_data['beta'].std() / np.sqrt(len(bin_data))
            logger.info(f"  MUV = [{muv_bins[i]}, {muv_bins[i+1]}): "
                       f"N = {len(bin_data)}, β = {mean_beta:.2f} ± {sem_beta:.2f}")
    
    return {
        'n_galaxies': len(valid),
        'spearman_rho': rho,
        'spearman_p': p_value,
        'tep_consistent': rho < 0
    }


def analyze_beta_redshift_evolution(df):
    """
    Test for redshift evolution of the β-mass relation.
    
    TEP Prediction: At higher z, the TEP effect is stronger (denser galaxies),
    so the β-mass correlation should be STRONGER at higher z.
    """
    logger.info("=" * 70)
    logger.info("ANALYSIS 2: β-Mass Relation Redshift Evolution")
    logger.info("=" * 70)
    
    valid = df.dropna(subset=['beta', 'MUV', 'z_best'])
    
    # Split by redshift
    z_low = valid[valid['z_best'] < 10]
    z_high = valid[valid['z_best'] >= 10]
    
    logger.info(f"Low-z (z < 10): N = {len(z_low)}")
    logger.info(f"High-z (z ≥ 10): N = {len(z_high)}")
    
    results = {}
    
    if len(z_low) >= 10:
        rho_low, p_low = stats.spearmanr(z_low['MUV'], z_low['beta'])
        logger.info(f"\nLow-z (z < 10): ρ = {rho_low:.3f}, p = {p_low:.4f}")
        results['rho_low_z'] = rho_low
        results['p_low_z'] = p_low
    
    if len(z_high) >= 10:
        rho_high, p_high = stats.spearmanr(z_high['MUV'], z_high['beta'])
        logger.info(f"High-z (z ≥ 10): ρ = {rho_high:.3f}, p = {p_high:.4f}")
        results['rho_high_z'] = rho_high
        results['p_high_z'] = p_high
    
    if 'rho_low_z' in results and 'rho_high_z' in results:
        delta_rho = results['rho_high_z'] - results['rho_low_z']
        logger.info(f"\nΔρ (high-z - low-z) = {delta_rho:.3f}")
        
        if delta_rho < 0:
            logger.info("✓ β-mass correlation is STRONGER at high-z (TEP-consistent)")
        else:
            logger.info("⚠ β-mass correlation is not stronger at high-z")
        
        results['delta_rho'] = delta_rho
        results['tep_consistent'] = delta_rho < 0
    
    return results


def analyze_beta_gamma_correlation(df):
    """
    Direct test: does β correlate with Γ_t?
    
    This is the most direct TEP test: galaxies with higher predicted Γ_t
    should have redder UV slopes at fixed other properties.
    """
    logger.info("=" * 70)
    logger.info("ANALYSIS 3: UV Slope vs Γ_t Correlation")
    logger.info("=" * 70)
    
    valid = df.dropna(subset=['beta', 'gamma_t'])
    
    logger.info(f"Sample size: N = {len(valid)}")
    
    rho, p_value = stats.spearmanr(valid['gamma_t'], valid['beta'])
    
    logger.info(f"Correlation (β vs Γ_t):")
    logger.info(f"  Spearman ρ = {rho:.3f}, p = {p_value:.4f}")
    
    if rho > 0 and p_value < 0.05:
        logger.info(f"\n✓ Higher Γ_t → REDDER UV slopes (TEP-consistent)")
    elif rho > 0:
        logger.info(f"\n⚠ Positive trend but not significant")
    else:
        logger.info(f"\n✗ No positive correlation detected")
    
    return {
        'n_galaxies': len(valid),
        'spearman_rho': rho,
        'spearman_p': p_value,
        'tep_consistent': rho > 0 and p_value < 0.05
    }


def run_uv_slope_analysis():
    """Run the complete UV slope analysis."""
    logger.info("=" * 70)
    logger.info("TEP-JWST Step 17: UV Slope (β) Analysis")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Testing TEP prediction: massive galaxies should have redder")
    logger.info("UV slopes due to enhanced stellar ages.")
    logger.info("")
    
    # Load and process data
    df = load_jades_photometry()
    df = calculate_uv_slope(df)
    df = calculate_tep_parameters(df)
    
    results = {}
    
    # Analysis 1: β-mass correlation
    results['beta_mass'] = analyze_beta_mass_correlation(df)
    
    # Analysis 2: Redshift evolution
    results['z_evolution'] = analyze_beta_redshift_evolution(df)
    
    # Analysis 3: β-Γ_t correlation
    results['beta_gamma'] = analyze_beta_gamma_correlation(df)
    
    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    
    if results['beta_mass']['tep_consistent']:
        logger.info("✓ UV slope correlates with mass (TEP-consistent)")
    else:
        logger.info("⚠ No significant β-mass correlation")
    
    if results.get('beta_gamma', {}).get('tep_consistent'):
        logger.info("✓ UV slope correlates with Γ_t (TEP-consistent)")
    else:
        logger.info("⚠ No significant β-Γ_t correlation")
    
    # Save results
    output_file = RESULTS_DIR / "jwst_uv_slope_analysis.json"
    
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
    run_uv_slope_analysis()
