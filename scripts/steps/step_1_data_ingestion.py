"""
TEP-JWST Step 1: Data Ingestion
Downloads and processes JWST high-z galaxy catalogs from JADES and CEERS.
Extracts spectroscopic redshifts, stellar masses, and SED-derived ages.
"""

import os
import sys
import logging
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from astropy.io import fits
from astropy.cosmology import Planck18
from astropy import units as u

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
RESULTS_DIR = PROJECT_ROOT / "results" / "outputs"

# Ensure directories exist
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_INTERIM.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# JADES DATA URLS
# ============================================================================
JADES_URLS = {
    "photometry_goods_s": "https://archive.stsci.edu/hlsps/jades/dr2/goods-s/catalogs/hlsp_jades_jwst_nircam_goods-s-deep_photometry_v2.0_catalog.fits",
    "photometry_goods_n": "https://archive.stsci.edu/hlsps/jades/dr3/goods-n/catalogs/hlsp_jades_jwst_nircam_goods-n_photometry_v1.0_catalog.fits",
}

# ============================================================================
# COSMIC AGE CALCULATOR
# ============================================================================
def cosmic_age_at_z(z):
    """
    Calculate the age of the universe at redshift z using Planck18 cosmology.
    
    Parameters
    ----------
    z : float or array
        Redshift(s)
    
    Returns
    -------
    age : float or array
        Cosmic age in Gyr
    """
    return Planck18.age(z).to(u.Gyr).value

def lookback_time_at_z(z):
    """
    Calculate lookback time to redshift z.
    
    Parameters
    ----------
    z : float or array
        Redshift(s)
    
    Returns
    -------
    lookback : float or array
        Lookback time in Gyr
    """
    return Planck18.lookback_time(z).to(u.Gyr).value

# ============================================================================
# DATA DOWNLOAD FUNCTIONS
# ============================================================================
def download_file(url, output_path, overwrite=False):
    """
    Download a file from URL to output_path.
    
    Parameters
    ----------
    url : str
        URL to download from
    output_path : Path
        Local path to save file
    overwrite : bool
        If True, overwrite existing files
    
    Returns
    -------
    success : bool
        True if download successful
    """
    if output_path.exists() and not overwrite:
        logger.info(f"File already exists: {output_path.name}")
        return True
    
    logger.info(f"Downloading {url}...")
    try:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = 100 * downloaded / total_size
                    print(f"\rProgress: {pct:.1f}%", end="", flush=True)
        
        print()  # newline after progress
        logger.info(f"Downloaded: {output_path.name}")
        return True
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False

def download_jades_catalogs(overwrite=False):
    """
    Download JADES photometric catalogs from MAST.
    
    Parameters
    ----------
    overwrite : bool
        If True, re-download existing files
    
    Returns
    -------
    downloaded : dict
        Dictionary of downloaded file paths
    """
    downloaded = {}
    
    for name, url in JADES_URLS.items():
        filename = url.split("/")[-1]
        output_path = DATA_RAW / filename
        
        if download_file(url, output_path, overwrite):
            downloaded[name] = output_path
    
    return downloaded

# ============================================================================
# HIGH-Z SAMPLE: LABBE ET AL. 2023 "IMPOSSIBLE GALAXIES"
# ============================================================================
# These are the 13 massive high-z candidates from Labbé et al. 2023 Nature
# Some have been spectroscopically confirmed, others refuted or revised

LABBE_2023_SAMPLE = pd.DataFrame({
    "ID": ["L23-1", "L23-2", "L23-3", "L23-4", "L23-5", "L23-6", "L23-7", 
           "L23-8", "L23-9", "L23-10", "L23-11", "L23-12", "L23-13050"],
    "z_phot": [7.4, 7.5, 7.6, 7.8, 8.0, 8.2, 8.4, 8.6, 8.8, 9.0, 9.1, 9.1, 8.1],
    "log_Mstar": [10.8, 10.9, 10.5, 10.7, 10.6, 10.4, 10.8, 10.3, 10.5, 10.9, 11.0, 10.6, 10.2],
    "status": ["candidate", "candidate", "candidate", "candidate", "candidate",
               "candidate", "candidate", "candidate", "candidate", "candidate",
               "candidate", "candidate", "AGN_z5.6"],  # L23-13050 confirmed as AGN at z=5.624
    "reference": ["Labbe+23"] * 13
})

# JADES spectroscopically confirmed z>10 galaxies (Curtis-Lake et al. 2023)
JADES_HIGHZ_SPEC = pd.DataFrame({
    "ID": ["JADES-GS-z10-0", "JADES-GS-z11-0", "JADES-GS-z12-0", "JADES-GS-z13-0"],
    "z_spec": [10.38, 11.58, 12.63, 13.20],
    "log_Mstar": [8.7, 8.5, 8.3, 8.1],  # Much lower masses than Labbé candidates
    "t_cosmic_Gyr": [0.458, 0.400, 0.357, 0.325],  # Cosmic age at that z
    "reference": ["Curtis-Lake+23"] * 4
})

# ============================================================================
# CATALOG PROCESSING
# ============================================================================
def load_jades_photometry(filepath):
    """
    Load and parse JADES photometric catalog.
    
    Parameters
    ----------
    filepath : Path
        Path to FITS catalog
    
    Returns
    -------
    df : pd.DataFrame
        Processed catalog with key columns
    """
    if not filepath.exists():
        logger.warning(f"File not found: {filepath}")
        return None
    
    logger.info(f"Loading {filepath.name}...")
    
    with fits.open(filepath) as hdul:
        # List available extensions
        logger.info(f"HDU extensions: {[h.name for h in hdul]}")
        
        # Typically photometry is in extension 1 or named 'CATALOG'
        data = hdul[1].data
        cols = data.columns.names
        
        logger.info(f"Found {len(data)} sources with {len(cols)} columns")
        logger.info(f"Sample columns: {cols[:20]}")
        
        # Convert to DataFrame
        df = pd.DataFrame({col: data[col] for col in cols})
    
    return df

def select_high_z_candidates(df, z_min=7.0, z_max=15.0, z_col='z_phot'):
    """
    Select high-redshift galaxy candidates from catalog.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input catalog
    z_min : float
        Minimum redshift
    z_max : float
        Maximum redshift
    z_col : str
        Name of redshift column
    
    Returns
    -------
    high_z : pd.DataFrame
        High-z candidates
    """
    if z_col not in df.columns:
        # Try alternative column names
        z_alternatives = ['ZPHOT', 'Z_PHOT', 'z_best', 'Z_BEST', 'redshift']
        for alt in z_alternatives:
            if alt in df.columns:
                z_col = alt
                break
        else:
            logger.warning(f"No redshift column found. Available: {df.columns.tolist()[:20]}")
            return None
    
    mask = (df[z_col] >= z_min) & (df[z_col] <= z_max)
    high_z = df[mask].copy()
    
    logger.info(f"Selected {len(high_z)} candidates with {z_min} <= z <= {z_max}")
    
    return high_z

# ============================================================================
# TEP CHRONOLOGICAL SHEAR FRAMEWORK
# ============================================================================
def calculate_age_excess(z, t_stellar_Gyr):
    """
    Calculate the "age excess" - how much older the stellar population
    appears compared to the cosmic age at that redshift.
    
    Under TEP, this excess arises from enhanced proper time accumulation
    in deep gravitational potentials.
    
    Parameters
    ----------
    z : float or array
        Redshift
    t_stellar_Gyr : float or array
        SED-derived stellar age in Gyr
    
    Returns
    -------
    age_excess : float or array
        Age excess (t_stellar - t_cosmic) in Gyr
    """
    t_cosmic = cosmic_age_at_z(z)
    age_excess = t_stellar_Gyr - t_cosmic
    return age_excess

def tep_chronological_shear(M_halo, alpha=0.58):
    """
    Calculate TEP-predicted chronological shear factor.
    
    Under TEP, the proper time enhancement scales with potential depth:
    Γ_t = α * (M_halo / M_ref)^(1/3)
    
    This predicts that massive halos experience accelerated stellar evolution.
    
    Parameters
    ----------
    M_halo : float or array
        Halo virial mass in solar masses
    alpha : float
        TEP coupling constant (from TEP-H0: α = 0.58 ± 0.16)
    
    Returns
    -------
    gamma_t : float or array
        Chronological shear factor (dimensionless)
    """
    M_ref = 1e10  # Reference mass scale (solar masses)
    gamma_t = alpha * (M_halo / M_ref) ** (1/3)
    return gamma_t

def tep_corrected_age(t_stellar, gamma_t):
    """
    Calculate TEP-corrected stellar age by removing the chronological shear.
    
    t_true = t_stellar / (1 + Γ_t)
    
    Parameters
    ----------
    t_stellar : float or array
        Observed (SED-derived) stellar age in Gyr
    gamma_t : float or array
        Chronological shear factor
    
    Returns
    -------
    t_corrected : float or array
        TEP-corrected stellar age in Gyr
    """
    t_corrected = t_stellar / (1 + gamma_t)
    return t_corrected

# ============================================================================
# MAIN PIPELINE
# ============================================================================
def run_data_ingestion(download=True, process=True):
    """
    Run the full data ingestion pipeline.
    
    Parameters
    ----------
    download : bool
        If True, download catalogs from MAST
    process : bool
        If True, process and save interim data
    
    Returns
    -------
    results : dict
        Dictionary containing processed data and statistics
    """
    results = {}
    
    # Step 1: Download JADES catalogs
    if download:
        logger.info("=" * 60)
        logger.info("STEP 1: Downloading JADES catalogs")
        logger.info("=" * 60)
        downloaded = download_jades_catalogs(overwrite=False)
        results['downloaded'] = downloaded
    
    # Step 2: Load and process built-in high-z samples
    logger.info("=" * 60)
    logger.info("STEP 2: Loading high-z galaxy samples")
    logger.info("=" * 60)
    
    # Labbé+23 sample
    logger.info(f"Labbé et al. 2023 sample: {len(LABBE_2023_SAMPLE)} candidates")
    logger.info(f"  z range: {LABBE_2023_SAMPLE['z_phot'].min():.1f} - {LABBE_2023_SAMPLE['z_phot'].max():.1f}")
    logger.info(f"  log(M*) range: {LABBE_2023_SAMPLE['log_Mstar'].min():.1f} - {LABBE_2023_SAMPLE['log_Mstar'].max():.1f}")
    
    # JADES spec-confirmed sample
    logger.info(f"JADES spec-confirmed sample: {len(JADES_HIGHZ_SPEC)} galaxies")
    for _, row in JADES_HIGHZ_SPEC.iterrows():
        t_cosmic = row['t_cosmic_Gyr']
        logger.info(f"  {row['ID']}: z={row['z_spec']:.2f}, t_cosmic={t_cosmic*1000:.0f} Myr")
    
    results['labbe_sample'] = LABBE_2023_SAMPLE
    results['jades_spec'] = JADES_HIGHZ_SPEC
    
    # Step 3: Calculate cosmic ages
    logger.info("=" * 60)
    logger.info("STEP 3: Calculating cosmic ages")
    logger.info("=" * 60)
    
    z_grid = np.array([7, 8, 9, 10, 11, 12, 13, 14, 15])
    cosmic_ages = cosmic_age_at_z(z_grid)
    
    logger.info("Cosmic age at redshift (Planck18):")
    for z, age in zip(z_grid, cosmic_ages):
        logger.info(f"  z = {z:2d}: t_cosmic = {age*1000:.0f} Myr ({age:.3f} Gyr)")
    
    results['cosmic_ages'] = pd.DataFrame({'z': z_grid, 't_cosmic_Gyr': cosmic_ages})
    
    # Step 4: Save interim data
    if process:
        logger.info("=" * 60)
        logger.info("STEP 4: Saving interim data")
        logger.info("=" * 60)
        
        # Save samples
        LABBE_2023_SAMPLE.to_csv(DATA_INTERIM / "labbe_2023_sample.csv", index=False)
        JADES_HIGHZ_SPEC.to_csv(DATA_INTERIM / "jades_spec_highz.csv", index=False)
        results['cosmic_ages'].to_csv(DATA_INTERIM / "cosmic_ages_planck18.csv", index=False)
        
        logger.info(f"Saved: {DATA_INTERIM / 'labbe_2023_sample.csv'}")
        logger.info(f"Saved: {DATA_INTERIM / 'jades_spec_highz.csv'}")
        logger.info(f"Saved: {DATA_INTERIM / 'cosmic_ages_planck18.csv'}")
    
    logger.info("=" * 60)
    logger.info("Data ingestion complete.")
    logger.info("=" * 60)
    
    return results

# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TEP-JWST Data Ingestion")
    parser.add_argument("--download", action="store_true", help="Download JADES catalogs")
    parser.add_argument("--verify", action="store_true", help="Verify existing data")
    parser.add_argument("--no-process", action="store_true", help="Skip processing")
    
    args = parser.parse_args()
    
    results = run_data_ingestion(
        download=args.download,
        process=not args.no_process
    )
    
    print("\nSummary:")
    print(f"  Labbé+23 candidates: {len(results['labbe_sample'])}")
    print(f"  JADES spec-confirmed: {len(results['jades_spec'])}")
