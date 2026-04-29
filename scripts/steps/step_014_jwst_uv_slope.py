#!/usr/bin/env python3
# Estimated runtime from last full canonical run (2026-03-09 15:52 UTC; full pipeline 32m18s): 1.3s.
"""
TEP-JWST Step 014: JADES Data Ingestion + UV Slope (β) Analysis

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

import sys
import json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from astropy.io import fits
from astropy.cosmology import Planck18
from pathlib import Path
import logging

# =============================================================================
# LOGGER SETUP
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging (severity levels: DEBUG/INFO/WARNING/ERROR/SUCCESS)
from scripts.utils.p_value_utils import format_p_value  # Safe p-value formatting (prevents floating-point underflow at p < 1e-300)
from scripts.utils.tep_model import KAPPA_GAL, KAPPA_GAL, compute_gamma_t as tep_gamma  # TEP model: KAPPA_GAL=9.6e5 mag from Cepheids, Gamma_t formula
from scripts.utils.downloader import smart_download  # Robust HTTP download utility with integrity checking

STEP_NUM = "014"  # Pipeline step number (sequential 001-176)
STEP_NAME = "jwst_uv_slope"  # JADES UV slope: tests TEP prediction that massive galaxies have redder UV slopes at fixed dust

LOGS_PATH = PROJECT_ROOT / "logs"  # Log directory (one plain-text file per step for debugging traceability)
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"  # JSON output directory (machine-readable statistical results)
LOGS_PATH.mkdir(parents=True, exist_ok=True)  # Create logs/ if missing; parents=True ensures full path tree exists
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)  # Create results/outputs/ if missing

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")  # Step-specific logger (isolated per-step logging for traceability)
set_step_logger(logger)  # Register as global step logger so print_status() routes to this step's log



DATA_DIR    = PROJECT_ROOT / "data"
INTERIM_DIR = DATA_DIR / "interim"
RESULTS_DIR = PROJECT_ROOT / "results" / "outputs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# JADES DR1 GOODS-S photometry catalog (Hainline et al. 2024, ApJS 297, 78)
# MAST HLSP: https://archive.stsci.edu/hlsps/jades/
# Multiple candidate URLs — the first one that responds is used.
JADES_HAINLINE_URLS = [
    # DR2 v2.0 GOODS-S photometry (confirmed live on MAST HLSP)
    "https://archive.stsci.edu/hlsps/jades/dr2/goods-s/catalogs/hlsp_jades_jwst_nircam_goods-s-deep_photometry_v2.0_catalog.fits",
    # DR3 fallback
    "https://archive.stsci.edu/hlsps/jades/dr3/goods-s/catalogs/hlsp_jades_jwst_nircam_goods-s-deep_photometry_v3.0_catalog.fits",
]
JADES_HAINLINE_FILE = DATA_DIR / "raw" / "JADES_z_gt_8_Candidates_Hainline_et_al.fits"
JADES_HAINLINE_SIZE_MB_MIN = 100  # full catalog is ~673 MB

# =============================================================================
# DOWNLOAD
# =============================================================================

def download_jades_hainline():
    """Download the JADES DR2 photometry catalog from MAST HLSP."""
    return smart_download(
        url=JADES_HAINLINE_URLS[0],
        dest=JADES_HAINLINE_FILE,
        min_size_mb=JADES_HAINLINE_SIZE_MB_MIN,
        fallback_urls=JADES_HAINLINE_URLS[1:],
        logger=logger,
    )


# =============================================================================
# PHYSICAL PROPERTY DERIVATION  (from archived step_3_process_real_data.py)
# =============================================================================

def _muv_to_stellar_mass(MUV, z):
    """Convert absolute UV magnitude to stellar mass using empirical scaling.

    Averages two independent calibrations:
      - Song et al. (2016): log_M* = (9.0 - 0.05*(z-8)) - 0.50*(MUV + 21)
      - Stefanon et al. (2021): log_M* = (8.8 - 0.03*(z-8)) - 0.45*(MUV + 21)

    Both are empirically calibrated at z = 4-10 from UV-luminosity-selected
    samples with SED-fitted stellar masses. The average reduces systematic
    bias from either calibration alone. The combined uncertainty is ~0.43 dex,
    dominated by the intrinsic scatter in the MUV-M* relation at high-z.
    """
    MUV = np.atleast_1d(MUV)
    z   = np.atleast_1d(z)
    # Song et al. 2016
    a_s = 9.0 - 0.05 * (z - 8);  b_s = -0.50
    lm_song = a_s + b_s * (MUV + 21)
    # Stefanon et al. 2021
    a_st = 8.8 - 0.03 * (z - 8); b_st = -0.45
    lm_stef = a_st + b_st * (MUV + 21)
    log_Mstar     = (lm_song + lm_stef) / 2
    log_Mstar_err = 0.43  # combined systematic + statistical
    return log_Mstar, log_Mstar_err


def _estimate_stellar_age(MUV, z):
    """Empirical UV-luminosity to stellar age estimate.

    Uses a simple parametric model:
      t_stellar = t_cosmic * f_age(MUV)
    where f_age is a luminosity-dependent age fraction clipped to [0.1, 0.9].
    Brighter galaxies (more negative MUV) get higher f_age because they
    are more massive and form earlier. This is a rough proxy; proper ages
    require full SED fitting (available in UNCOVER but not in JADES DR2
    photometry-only catalogs).
    """
    MUV      = np.atleast_1d(MUV)
    z        = np.atleast_1d(z)
    t_cosmic = np.array([Planck18.age(zi).value for zi in z])
    f_age    = np.clip(0.5 + 0.1 * (-(MUV - (-21))), 0.1, 0.9)
    t_stellar = t_cosmic * f_age
    return t_stellar


def _derive_and_save_physical_properties(df):
    """Compute log_Mstar, log_Mhalo, t_stellar_Gyr, age_ratio and save
    data/interim/jades_highz_physical.csv for downstream steps."""
    z  = df["z_best"].values
    muv = df["MUV"].values

    log_Mstar, _   = _muv_to_stellar_mass(muv, z)
    log_ratio      = 2.0 + 0.1 * (z - 8)          # SHMR at high-z
    log_Mhalo      = log_Mstar + log_ratio
    t_cosmic       = np.array([Planck18.age(zi).value for zi in z])
    t_stellar      = _estimate_stellar_age(muv, z)
    age_excess     = t_stellar - t_cosmic
    age_ratio      = np.where(t_cosmic > 0, t_stellar / t_cosmic, np.nan)

    id_col = "ID" if "ID" in df.columns else "id"
    keep = [id_col, "z_phot", "z_best", "MUV"]
    phys = df[keep].copy().rename(columns={id_col: "ID"})
    phys["z_spec"] = np.nan  # DR2 photometry catalog has no spec-z
    phys["log_Mstar"]    = log_Mstar
    phys["log_Mhalo"]    = log_Mhalo
    phys["t_cosmic_Gyr"] = t_cosmic
    phys["t_stellar_Gyr"]= t_stellar
    phys["age_excess_Gyr"]= age_excess
    phys["age_ratio"]    = age_ratio
    phys["source"]       = "JADES-GOODS-S"

    out = INTERIM_DIR / "jades_highz_physical.csv"
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    phys.to_csv(out, index=False)
    logger.info(f"Saved jades_highz_physical.csv: N={len(phys)} galaxies → {out}")
    return phys


# =============================================================================
# TEP PARAMETERS
# =============================================================================


def load_jades_photometry():
    """Download (if needed) and load JADES z>8 candidates with photometry."""
    logger.info("Loading JADES z>8 candidates (Hainline+23)...")

    if not JADES_HAINLINE_FILE.exists() or JADES_HAINLINE_FILE.stat().st_size / 1e6 < JADES_HAINLINE_SIZE_MB_MIN:
        if not download_jades_hainline():
            logger.error("Cannot proceed without JADES Hainline catalog.")
            return None

    with fits.open(JADES_HAINLINE_FILE) as hdu:
        def to_native(arr):
            arr = np.array(arr)
            if hasattr(arr.dtype, 'byteorder') and arr.dtype.byteorder == '>':
                return arr.astype(arr.dtype.newbyteorder('='))
            return arr

        # DR2 catalog: join KRON (fluxes) + PHOTOZ (photo-z) on ID
        kron  = hdu['KRON'].data
        photoz = hdu['PHOTOZ'].data

        df_kron = pd.DataFrame({
            'id':        to_native(kron['ID']),
            'ra':        to_native(kron['RA']),
            'dec':       to_native(kron['DEC']),
            'F115W':     to_native(kron['F115W_KRON']),
            'F115W_err': to_native(kron['F115W_KRON_S']),
            'F150W':     to_native(kron['F150W_KRON']),
            'F150W_err': to_native(kron['F150W_KRON_S']),
            'F200W':     to_native(kron['F200W_KRON']),
            'F200W_err': to_native(kron['F200W_KRON_S']),
            'F277W':     to_native(kron['F277W_KRON']),
            'F277W_err': to_native(kron['F277W_KRON_S']),
            'F356W':     to_native(kron['F356W_KRON']),
            'F356W_err': to_native(kron['F356W_KRON_S']),
            'F444W':     to_native(kron['F444W_KRON']),
            'F444W_err': to_native(kron['F444W_KRON_S']),
        })
        df_pz = pd.DataFrame({
            'id':     to_native(photoz['ID']),
            'z_phot': to_native(photoz['EAZY_z_a']),
            'z_l95':  to_native(photoz['EAZY_l95']),
            'z_u95':  to_native(photoz['EAZY_u95']),
        })
        df = df_kron.merge(df_pz, on='id', how='inner')

    # DR2 has no spec-z column — use photo-z throughout
    df['z_best'] = df['z_phot']
    # Use P_z_gt_7 proxy: 1 if 95th-percentile lower bound > 7, else 0
    df['P_z_gt_7'] = (df['z_l95'] > 7).astype(float)

    # Filter for high-confidence z > 8
    df = df[(df['z_best'] > 8) & (df['P_z_gt_7'] > 0.5)].copy()

    # Compute MUV from F150W flux (proxy for rest-frame ~1500-1700 Å at z~8-12)
    d_L_pc = np.array([Planck18.luminosity_distance(zi).to('pc').value for zi in df['z_best']])
    f_nJy = df['F150W'].values.copy().astype(float)
    f_nJy = np.where(f_nJy > 0, f_nJy, np.nan)
    m_ab = -2.5 * np.log10(f_nJy * 1e-9 / 3631)
    dm   = 5.0 * np.log10(d_L_pc / 10.0)
    k_corr = 2.5 * np.log10(1.0 + df['z_best'].values)
    df['MUV'] = m_ab - dm + k_corr

    logger.info(f"Loaded {len(df)} high-confidence z > 8 candidates")
    
    return df


def calculate_uv_slope(df):
    """Calculate UV spectral slope beta from broadband photometry.

    The UV slope is defined as f_lambda ~ lambda^beta, where beta = -2
    for a flat f_nu spectrum. Star-forming galaxies typically have
    beta = -2.5 to -1.5; dust reddening and age push beta toward less
    negative (redder) values.

    For z ~ 8-12, rest-frame UV (1500-2500 Angstrom) falls in:
      z=8:  observed 1.35-2.25 um (F150W, F200W)
      z=10: observed 1.65-2.75 um (F200W, F277W)
      z=12: observed 1.95-3.25 um (F200W, F277W, F356W)

    Method:
      For each galaxy, identify which NIRCam bands probe rest-frame UV,
      require S/N > 2, then compute the power-law slope from the first
      and last qualifying bands:
        beta = d(log f_nu) / d(log lambda) - 2
      The -2 converts from f_nu to f_lambda convention.

    Error propagation uses standard first-order formula for the ratio
    of two flux measurements.
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
    df['gamma_t'] = tep_gamma(df['log_Mhalo'].values, df['z_best'].values, kappa=KAPPA_GAL)
     
    return df


def analyze_beta_mass_correlation(df):
    """Test for correlation between UV slope beta and galaxy mass.

    Under standard physics, beta correlates with stellar mass primarily
    through the mass-dust relation: more massive galaxies have more ISM
    and hence more dust attenuation, producing redder UV slopes.

    TEP adds a second channel: more massive galaxies have higher Gamma_t,
    so their stellar populations appear older, shifting the intrinsic
    (dust-free) UV slope redward. The TEP contribution manifests as
    EXCESS reddening at high mass beyond what dust alone predicts.

    MUV is used as a mass proxy (more negative = brighter = more massive),
    so a negative rho(MUV, beta) indicates brighter galaxies are redder.
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
    p_value_fmt = format_p_value(p_value)
    
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
        'spearman_p': p_value_fmt,
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
        results['p_low_z'] = format_p_value(p_low)
    
    if len(z_high) >= 10:
        rho_high, p_high = stats.spearmanr(z_high['MUV'], z_high['beta'])
        logger.info(f"High-z (z ≥ 10): ρ = {rho_high:.3f}, p = {p_high:.4f}")
        results['rho_high_z'] = rho_high
        results['p_high_z'] = format_p_value(p_high)
    
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
    """Direct test: does UV slope beta correlate with Gamma_t?

    This is the most direct TEP test for this dataset. Unlike the
    beta-mass correlation (which could arise from dust alone), a
    positive correlation between beta and Gamma_t would indicate that
    the TEP-predicted chronological enhancement contributes to the
    UV reddening beyond the standard dust pathway.

    A positive rho(Gamma_t, beta) with p < 0.05 is TEP-consistent.
    """
    logger.info("=" * 70)
    logger.info("ANALYSIS 3: UV Slope vs Γ_t Correlation")
    logger.info("=" * 70)
    
    valid = df.dropna(subset=['beta', 'gamma_t'])
    
    logger.info(f"Sample size: N = {len(valid)}")
    
    rho, p_value = stats.spearmanr(valid['gamma_t'], valid['beta'])
    p_value_fmt = format_p_value(p_value)
    
    logger.info(f"Correlation (β vs Γ_t):")
    logger.info(f"  Spearman ρ = {rho:.3f}, p = {p_value:.4f}")
    
    if rho > 0 and (p_value_fmt is not None and p_value_fmt < 0.05):
        logger.info(f"\n✓ Higher Γ_t → REDDER UV slopes (TEP-consistent)")
    elif rho > 0:
        logger.info(f"\n⚠ Positive trend but not significant")
    else:
        logger.info(f"\n✗ No positive correlation detected")
    
    return {
        'n_galaxies': len(valid),
        'spearman_rho': rho,
        'spearman_p': p_value_fmt,
        'tep_consistent': rho > 0 and (p_value_fmt is not None and p_value_fmt < 0.05)
    }


def run_uv_slope_analysis():
    """Run the complete UV slope analysis."""
    logger.info("=" * 70)
    logger.info(f"TEP-JWST Step {STEP_NUM}: JADES Data Ingestion + UV Slope (β) Analysis")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Testing TEP prediction: massive galaxies should have redder")
    logger.info("UV slopes due to enhanced stellar ages.")
    logger.info("")

    # Load data (downloads if missing)
    df = load_jades_photometry()
    if df is None:
        logger.error(f"Step {STEP_NUM} aborted: JADES data unavailable.")
        return {"status": "aborted", "reason": "JADES catalog unavailable"}

    # Persist physical properties for downstream steps (017-027)
    _derive_and_save_physical_properties(df)

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
    output_file = RESULTS_DIR / f"step_{STEP_NUM}_{STEP_NAME}.json"
    
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
    logger.info(f"Step {STEP_NUM} complete.")
    return results


if __name__ == "__main__":
    run_uv_slope_analysis()
