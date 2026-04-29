#!/usr/bin/env python3
# Estimated runtime from last full canonical run (2026-03-09 15:52 UTC; full pipeline 32m18s): 2.9s.
"""
TEP-JWST Step 001: Load UNCOVER DR4 Data and Apply Quality Cuts

This step loads the UNCOVER DR4 SPS catalog and applies quality cuts
to create the analysis samples used for the TEP analysis.

Outputs:
- results/interim/step_001_uncover_full_sample.csv (N ~ 2,315)
- results/interim/step_001_uncover_multi_property_sample.csv (N ~ 1,108)
- results/interim/step_001_uncover_z8_sample.csv (N ~ 283)
- results/outputs/step_001_summary.json

Author: Matthew L. Smawfield
Date: January 2026
"""

import json
import sys
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.cosmology import Planck18 as cosmo
from pathlib import Path

# =============================================================================
# PATHS AND LOGGER
# =============================================================================
# Resolve the project root by navigating two directories up from this script:
# scripts/steps/step_001_... -> scripts/ -> PROJECT_ROOT/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.p_value_utils import format_p_value, safe_json_default
from scripts.utils.downloader import smart_download

STEP_NUM = "001"
STEP_NAME = "uncover_load"

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "uncover"

# ---------------------------------------------------------------------------
# UNCOVER DR4 SPS catalog (Wang et al. 2024, ApJS 270, 12)
# DOI: 10.5281/zenodo.14281664
#
# This is the primary photometric catalog for the Abell 2744 lensing field,
# containing ~50,000 sources with SED-fitted stellar population parameters
# (stellar mass, SFR, dust, metallicity, mass-weighted age) from Prospector.
# ---------------------------------------------------------------------------
UNCOVER_DR4_URL = (
    "https://zenodo.org/api/records/14281664/files/"
    "UNCOVER_DR4_SPS_catalog.fits/content"
)
UNCOVER_DR4_FILE = DATA_PATH / "UNCOVER_DR4_SPS_catalog.fits"
UNCOVER_DR4_SIZE_MB_MIN = 60  # valid file is ~62.8 MB
INTERIM_PATH = PROJECT_ROOT / "results" / "interim"
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"
LOGS_PATH = PROJECT_ROOT / "logs"

INTERIM_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
LOGS_PATH.mkdir(parents=True, exist_ok=True)

# Initialize logger
logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")
set_step_logger(logger)

# =============================================================================
# LOAD DATA
# =============================================================================

def download_uncover_catalog():
    """Download UNCOVER DR4 SPS catalog from Zenodo if not already present."""
    return smart_download(
        url=UNCOVER_DR4_URL,
        dest=UNCOVER_DR4_FILE,
        min_size_mb=UNCOVER_DR4_SIZE_MB_MIN,
        logger=logger,
    )


def load_uncover_catalog():
    """Download (if needed) and load UNCOVER DR4 SPS catalog."""
    if not UNCOVER_DR4_FILE.exists() or UNCOVER_DR4_FILE.stat().st_size / 1e6 < UNCOVER_DR4_SIZE_MB_MIN:
        success = download_uncover_catalog()
        if not success:
            print_status("ERROR: Could not obtain UNCOVER DR4 catalog. Aborting.", "ERROR")
            return None

    print_status(f"Loading: {UNCOVER_DR4_FILE}", "PROCESS")

    with fits.open(UNCOVER_DR4_FILE) as hdul:
        data = hdul[1].data

    print_status(f"Total sources: {len(data)}", "INFO")
    return data

# =============================================================================
# EXTRACT COLUMNS
# =============================================================================

def fix_byteorder(arr):
    """Fix big-endian byte order for numpy 2.0 compatibility.

    FITS files store data in big-endian ('>') format. NumPy 2.0+ raises
    errors when mixing big-endian arrays with native-endian operations.
    This converts to the system's native byte order ('=') to avoid those
    errors while preserving the numerical values.
    """
    arr = np.array(arr)
    if arr.dtype.byteorder == '>':
        return arr.astype(arr.dtype.newbyteorder('='))
    return arr

def extract_columns(data):
    """Extract relevant columns into a DataFrame.

    Column mapping (UNCOVER DR4 Prospector names -> analysis names):
      z_50         -> z_phot       : median photometric redshift from EAZY
      mstar_50     -> log_Mstar    : log10(M*/Msun), median SED-fitted stellar mass
      mwa_50       -> mwa          : mass-weighted age [Gyr] from Prospector
      met_50       -> met          : log(Z/Zsun), stellar metallicity
      dust2_50     -> dust         : A_V [mag], diffuse dust attenuation (Calzetti)
      sfr100_50    -> sfr100       : SFR averaged over last 100 Myr [Msun/yr]
      ssfr100_50   -> ssfr100      : specific SFR over last 100 Myr [1/yr]
      sfr10_50     -> sfr10        : SFR averaged over last 10 Myr [Msun/yr]
      ssfr10_50    -> ssfr10       : specific SFR over last 10 Myr [1/yr]

    The _16 and _84 suffixes denote the 16th and 84th posterior percentiles,
    providing approximate 1-sigma uncertainties on each parameter.
    """
    
    df = pd.DataFrame({
        'id': fix_byteorder(data['id']),
        'ra': fix_byteorder(data['ra']),
        'dec': fix_byteorder(data['dec']),
        'z_spec': fix_byteorder(data['z_spec']),
        'z_phot': fix_byteorder(data['z_50']),         # median photo-z
        'z_16': fix_byteorder(data['z_16']),            # photo-z 16th percentile
        'z_84': fix_byteorder(data['z_84']),            # photo-z 84th percentile
        'log_Mstar': fix_byteorder(data['mstar_50']),   # log10(M*/Msun)
        'log_Mstar_16': fix_byteorder(data['mstar_16']),
        'log_Mstar_84': fix_byteorder(data['mstar_84']),
        'mwa': fix_byteorder(data['mwa_50']),           # mass-weighted age [Gyr]
        'mwa_16': fix_byteorder(data['mwa_16']),
        'mwa_84': fix_byteorder(data['mwa_84']),
        'met': fix_byteorder(data['met_50']),           # log(Z/Zsun)
        'met_16': fix_byteorder(data['met_16']),
        'met_84': fix_byteorder(data['met_84']),
        'dust': fix_byteorder(data['dust2_50']),        # A_V [mag]
        'dust_16': fix_byteorder(data['dust2_16']),
        'dust_84': fix_byteorder(data['dust2_84']),
        'sfr100': fix_byteorder(data['sfr100_50']),     # SFR_100Myr [Msun/yr]
        'ssfr100': fix_byteorder(data['ssfr100_50']),   # sSFR_100Myr [1/yr]
        'sfr10': fix_byteorder(data['sfr10_50']),       # SFR_10Myr [Msun/yr]
        'ssfr10': fix_byteorder(data['ssfr10_50']),     # sSFR_10Myr [1/yr]
        'chi2': fix_byteorder(data['chi2']),            # best-fit chi-squared
        'use_phot': fix_byteorder(data['use_phot']),    # photometry quality flag
    })
    
    return df

# =============================================================================
# APPLY QUALITY CUTS
# =============================================================================

def apply_full_sample_cuts(df):
    """
    Full sample cuts for basic correlations.
    
    Criteria:
    - z_phot > 4 and z_phot < 10 (high-z sample)
    - log_Mstar > 8 (above completeness limit)
    - Valid sSFR (not NaN, > 0)
    - Valid MWA (not NaN)

    Rationale:
      z > 4  : TEP effects scale as alpha(z) = kappa_gal * sqrt(1+z), so they
               become observationally significant only above z ~ 4.
      z < 10 : Upper bound of reliable UNCOVER photometric redshifts.
      M* > 10^8 Msun : Below this, the catalog becomes severely incomplete
               at z > 6 due to surface-brightness limits.
      Valid sSFR/MWA : Needed for the downstream age-ratio and sSFR
               correlation analyses (Threads 1-4).
    """
    
    mask = (
        (df['z_phot'] > 4) & 
        (df['z_phot'] < 10) & 
        (df['log_Mstar'] > 8) &
        (~df['ssfr100'].isna()) & 
        (df['ssfr100'] > 0) &
        (~df['mwa'].isna())
    )
    
    df_cut = df[mask].copy()
    print_status(f"Full sample after cuts: N = {len(df_cut)}", "INFO")
    
    return df_cut

def apply_multi_property_cuts(df):
    """
    Multi-property sample with metallicity quality cuts.
    
    Criteria:
    - All full sample cuts
    - Valid metallicity (not NaN)
    - Metallicity uncertainty < 0.5 dex
    - Valid dust (not NaN)

    Rationale:
      This stricter sample is used for Threads 3 (Gamma_t vs metallicity)
      and Thread 4 (Gamma_t vs dust), which require well-constrained
      metallicity and dust posteriors. The 0.5 dex metallicity uncertainty
      threshold removes objects whose broad posteriors would dilute any
      genuine TEP-driven correlation.
    """
    
    df['met_err'] = (df['met_84'] - df['met_16']) / 2
    
    mask = (
        (df['z_phot'] > 4) & 
        (df['z_phot'] < 10) & 
        (df['log_Mstar'] > 8) &
        (~df['ssfr100'].isna()) & 
        (df['ssfr100'] > 0) &
        (~df['mwa'].isna()) &
        (~df['met'].isna()) &
        (~df['met_err'].isna()) &
        (df['met_err'] < 0.5) &
        (~df['dust'].isna())
    )
    
    df_cut = df[mask].copy()
    print_status(f"Multi-property sample after cuts: N = {len(df_cut)}", "INFO")
    
    return df_cut

def apply_z8_cuts(df):
    """
    z > 8 sample for dust anomaly analysis.
    
    Criteria:
    - z_phot >= 8 and z_phot < 10
    - log_Mstar > 8
    - Valid dust (not NaN)

    Rationale:
      At z > 8 the universe is < 600 Myr old. Standard AGB dust production
      requires 100-300 Myr of stellar evolution, so under standard physics
      the most massive galaxies should not yet be heavily dust-obscured.
      TEP predicts that enhanced proper time (Gamma_t > 1) in massive halos
      allows sufficient time for AGB dust production, creating an anomalously
      strong mass-dust correlation at these redshifts. This sample isolates
      the regime where the TEP dust signature is strongest.
    """
    
    mask = (
        (df['z_phot'] >= 8) & 
        (df['z_phot'] < 10) & 
        (df['log_Mstar'] > 8) &
        (~df['dust'].isna())
    )
    
    df_cut = df[mask].copy()
    print_status(f"z > 8 sample after cuts: N = {len(df_cut)}", "INFO")
    
    return df_cut

# =============================================================================
# COMPUTE DERIVED QUANTITIES
# =============================================================================

def stellar_to_halo_mass(log_Mstar, z):
    """
    Compute halo mass from stellar mass using abundance matching.
    
    Based on Behroozi et al. (2019) parametrization, simplified for high-z.
    At high-z (z > 4), the stellar-to-halo mass ratio is approximately:
    
        log(M*/Mh) ≈ -1.8 - 0.1*(log_Mstar - 10) + 0.05*(z - 5)
    
    This gives:
    - At log_Mstar = 8:  log(M*/Mh) ≈ -2.0  ->  Mh/M* ≈ 100
    - At log_Mstar = 10: log(M*/Mh) ≈ -1.8  ->  Mh/M* ≈ 63
    - At log_Mstar = 11: log(M*/Mh) ≈ -1.7  ->  Mh/M* ≈ 50
    
    The relation flattens at high mass due to AGN feedback.

    Why halo mass matters for TEP:
      The TEP chronological enhancement factor Gamma_t depends on the
      depth of the gravitational potential, which is set primarily by
      the dark-matter halo mass M_h rather than the stellar mass M*.
      This function provides the M* -> M_h bridge needed to compute
      Gamma_t from the SED-fitted stellar masses in the catalog.
    """
    # Base ratio at log_Mstar = 10, z = 5
    log_ratio_base = -1.8
    
    # Mass dependence: ratio decreases (less dark matter dominated) at high mass
    mass_term = -0.1 * (log_Mstar - 10)
    
    # Redshift dependence: ratio increases slightly at higher z
    z_term = 0.05 * (z - 5)
    
    log_ratio = log_ratio_base + mass_term + z_term
    
    # Halo mass = stellar mass / ratio
    log_Mh = log_Mstar - log_ratio
    
    return log_Mh

def compute_derived_quantities(df):
    """Compute derived quantities for TEP analysis.

    Derived columns added:
      log_Mh       : log10(M_halo/Msun) via Behroozi-like abundance matching.
      log_Mh_simple: log10(M_halo/Msun) using a fixed M_h/M* = 100 ratio
                     (retained for cross-check only; not used in main analysis).
      t_cosmic     : Age of the universe at the galaxy's redshift [Gyr],
                     computed from Planck18 cosmology.
      age_ratio    : mwa / t_cosmic. Under standard physics this should be
                     <= 1; values approaching or exceeding 1 indicate that
                     the SED-fitted stellar age is comparable to the cosmic
                     age, which TEP explains via enhanced proper time.
      log_ssfr     : log10(sSFR_100Myr) for correlation analyses.
    """
    
    # Use abundance matching for halo mass (more accurate than fixed ratio)
    df['log_Mh'] = stellar_to_halo_mass(df['log_Mstar'].values, df['z_phot'].values)
    
    # Also store the simple estimate for comparison
    df['log_Mh_simple'] = df['log_Mstar'] + 2.0
    
    # Cosmic age at each galaxy's redshift from Planck18 LCDM
    df['t_cosmic'] = cosmo.age(df['z_phot'].values).value  # [Gyr]
    
    # Age ratio: how close the SED-fitted stellar age is to the cosmic age
    df['age_ratio'] = df['mwa'] / df['t_cosmic']
    
    # log specific SFR for Thread 1 (mass-sSFR inversion test)
    df['log_ssfr'] = np.log10(df['ssfr100'])
    
    return df

# =============================================================================
# MAIN
# =============================================================================

def main():
    print_status("STEP 01: Load UNCOVER DR4 Data", "TITLE")
    
    data = load_uncover_catalog()
    if data is None:
        print_status("Aborting step 001 due to missing data.", "ERROR")
        return

    df = extract_columns(data)
    print_status(f"Extracted {len(df.columns)} columns", "INFO")
    
    df_full = apply_full_sample_cuts(df)
    df_full = compute_derived_quantities(df_full)
    
    df_multi = apply_multi_property_cuts(df)
    df_multi = compute_derived_quantities(df_multi)
    
    df_z8 = apply_z8_cuts(df)
    df_z8 = compute_derived_quantities(df_z8)
    
    print_status("Saving outputs...", "PROCESS")
    
    df_full.to_csv(INTERIM_PATH / f"step_{STEP_NUM}_uncover_full_sample.csv", index=False)
    print_status(f"Saved: step_{STEP_NUM}_uncover_full_sample.csv", "INFO")
    
    df_multi.to_csv(INTERIM_PATH / f"step_{STEP_NUM}_uncover_multi_property_sample.csv", index=False)
    print_status(f"Saved: step_{STEP_NUM}_uncover_multi_property_sample.csv", "INFO")
    
    df_z8.to_csv(INTERIM_PATH / f"step_{STEP_NUM}_uncover_z8_sample.csv", index=False)
    print_status(f"Saved: step_{STEP_NUM}_uncover_z8_sample.csv", "INFO")
    
    summary = {
        "step": f"{STEP_NUM}",
        "name": "UNCOVER DR4 Data Loading",
        "full_sample_n": len(df_full),
        "multi_property_n": len(df_multi),
        "z8_sample_n": len(df_z8),
        "z_range": [float(df_full['z_phot'].min()), float(df_full['z_phot'].max())],
        "log_Mstar_range": [float(df_full['log_Mstar'].min()), float(df_full['log_Mstar'].max())],
    }
    
    with open(OUTPUT_PATH / f"step_{STEP_NUM}_uncover_load.json", "w") as f:
        json.dump(summary, f, indent=2, default=safe_json_default)
    
    print_status(f"Full sample: N = {summary['full_sample_n']}", "SUCCESS")
    print_status(f"Multi-property: N = {summary['multi_property_n']}", "SUCCESS")
    print_status(f"z > 8: N = {summary['z8_sample_n']}", "SUCCESS")
    print_status(f"Step {STEP_NUM} complete.", "SUCCESS")

if __name__ == "__main__":
    main()
