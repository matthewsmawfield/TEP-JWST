"""
TEP-JWST Step 3: Process Real JWST Data
Extracts high-z galaxy sample from Hainline+23 JADES catalog and derives
stellar masses from MUV using standard mass-to-light relations.
"""

import os
import sys
import logging
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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# COSMOLOGICAL FUNCTIONS
# ============================================================================
def cosmic_age_at_z(z):
    """Calculate cosmic age at redshift z in Gyr."""
    return Planck18.age(z).to(u.Gyr).value

def luminosity_distance(z):
    """Calculate luminosity distance in Mpc."""
    return Planck18.luminosity_distance(z).to(u.Mpc).value

# ============================================================================
# MUV TO STELLAR MASS CONVERSION
# ============================================================================
def muv_to_stellar_mass(MUV, z, method='song2016'):
    """
    Convert absolute UV magnitude to stellar mass using literature relations.
    
    At high-z, the UV luminosity traces recent star formation, and stellar mass
    can be estimated using mass-to-light ratios calibrated from SED fitting.
    
    Parameters
    ----------
    MUV : float or array
        Absolute UV magnitude (rest-frame 1500 Å)
    z : float or array
        Redshift
    method : str
        Conversion method:
        - 'song2016': Song et al. 2016 relation (default)
        - 'duncan2014': Duncan et al. 2014 relation
        - 'stefanon2021': Stefanon et al. 2021 relation
    
    Returns
    -------
    log_Mstar : float or array
        Log10 stellar mass in solar masses
    log_Mstar_err : float or array
        Uncertainty in log stellar mass (typical ~0.3-0.5 dex)
    
    References
    ----------
    Song et al. 2016, ApJ, 825, 5: M* = 10^(a + b * MUV) with z-evolution
    Duncan et al. 2014, MNRAS, 444, 2960
    Stefanon et al. 2021, ApJ, 922, 29
    """
    MUV = np.atleast_1d(MUV)
    z = np.atleast_1d(z)
    
    if method == 'song2016':
        # Song et al. 2016: log(M*) = a + b * (MUV + 21)
        # Calibrated at z~4-8, with mild z-evolution
        # At z~8: a ~ 9.0, b ~ -0.5
        a = 9.0 - 0.05 * (z - 8)  # Mild evolution
        b = -0.50
        log_Mstar = a + b * (MUV + 21)
        log_Mstar_err = 0.4 * np.ones_like(log_Mstar)  # Typical scatter
        
    elif method == 'duncan2014':
        # Duncan et al. 2014 relation
        # log(M*) = -0.4 * (MUV + 21) + 9.5 at z~7
        a = 9.5 - 0.1 * (z - 7)
        b = -0.4
        log_Mstar = a + b * (MUV + 21)
        log_Mstar_err = 0.5 * np.ones_like(log_Mstar)
        
    elif method == 'stefanon2021':
        # Stefanon et al. 2021 - calibrated specifically for z>7
        # More robust for the high-z regime
        a = 8.8 - 0.03 * (z - 8)
        b = -0.45
        log_Mstar = a + b * (MUV + 21)
        log_Mstar_err = 0.35 * np.ones_like(log_Mstar)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return log_Mstar, log_Mstar_err

# ============================================================================
# STELLAR AGE ESTIMATION
# ============================================================================
def estimate_stellar_age(MUV, z, beta_UV=None):
    """
    Estimate stellar population age from UV properties.
    
    For high-z galaxies, stellar ages can be constrained from:
    1. UV slope (beta): Bluer = younger
    2. UV luminosity relative to cosmic time
    3. Balmer break strength (if available)
    
    Without detailed SED fitting, we estimate ages using empirical relations
    between UV luminosity and age, assuming continuous star formation.
    
    Parameters
    ----------
    MUV : float or array
        Absolute UV magnitude
    z : float or array
        Redshift
    beta_UV : float or array, optional
        UV spectral slope (typically -2.5 to -1.5 for high-z galaxies)
    
    Returns
    -------
    t_stellar : float or array
        Estimated stellar age in Gyr
    t_stellar_err : float or array
        Uncertainty in stellar age (large, ~50%)
    """
    MUV = np.atleast_1d(MUV)
    z = np.atleast_1d(z)
    
    # Get cosmic age at this redshift
    t_cosmic = np.array([cosmic_age_at_z(zi) for zi in z])
    
    # Empirical relation: brighter galaxies tend to be more evolved
    # This is based on the idea that more massive galaxies formed earlier
    # and have had more time to build up stellar mass
    
    # UV luminosity relative to characteristic luminosity L*
    # M*_UV ~ -21 at z~8
    delta_MUV = MUV - (-21)
    
    # Brighter galaxies (negative delta_MUV) have had more time to form stars
    # We parameterize this as a fraction of cosmic age
    # f_age = 0.5 for L* galaxies, scaling with luminosity
    f_age = 0.5 + 0.1 * (-delta_MUV)  # Brighter = older fraction
    f_age = np.clip(f_age, 0.1, 0.9)  # Physical bounds
    
    # If UV slope available, use it to refine age
    if beta_UV is not None:
        # Bluer (more negative beta) = younger
        # beta ~ -2.5 for very young (<50 Myr), -1.5 for older (>200 Myr)
        beta_UV = np.atleast_1d(beta_UV)
        age_factor = 1.0 + 0.2 * (beta_UV + 2.0)  # Redder = older
        f_age = f_age * np.clip(age_factor, 0.5, 1.5)
    
    t_stellar = t_cosmic * f_age
    t_stellar_err = t_stellar * 0.5  # 50% uncertainty (large)
    
    return t_stellar, t_stellar_err

# ============================================================================
# LOAD AND PROCESS HAINLINE+23 CATALOG
# ============================================================================
def load_hainline_catalog():
    """
    Load and process the Hainline et al. 2023 JADES z>8 catalog.
    
    Returns
    -------
    df : pd.DataFrame
        Processed catalog with derived stellar masses
    """
    catalog_path = DATA_RAW / "JADES_z_gt_8_Candidates_Hainline_et_al.fits"
    
    if not catalog_path.exists():
        logger.error(f"Catalog not found: {catalog_path}")
        logger.error("Run step_1_data_ingestion.py --download first")
        return None
    
    logger.info(f"Loading {catalog_path.name}...")
    
    with fits.open(catalog_path) as hdul:
        # Use PRIMARY_SAMPLE extension
        data = hdul['PRIMARY_SAMPLE'].data
        
        df = pd.DataFrame({
            'ID': data['JADES_ID'],
            'RA': data['RA'],
            'DEC': data['DEC'],
            'z_phot': data['EAZY_z_a'],
            'z_phot_lo': data['EAZY_z_a'] - data['EAZY_sigma68_lo'],
            'z_phot_hi': data['EAZY_z_a'] + data['EAZY_sigma68_hi'],
            'z_spec': data['z_spec'],
            'MUV': data['MUV'],
            'mag_F277W': data['m_F277W_Kron'],
            'P_z_gt_7': data['EAZY_Pzgt7'],
            'delta_chisq': data['EAZY_delta_chisq'],
        })
    
    # Replace -9999 with NaN for z_spec
    df.loc[df['z_spec'] < 0, 'z_spec'] = np.nan
    
    # Use spec-z where available, otherwise photo-z
    df['z_best'] = df['z_spec'].fillna(df['z_phot'])
    
    logger.info(f"Loaded {len(df)} sources")
    logger.info(f"  z_phot range: {df['z_phot'].min():.1f} - {df['z_phot'].max():.1f}")
    logger.info(f"  With spec-z: {df['z_spec'].notna().sum()}")
    
    return df

def derive_physical_properties(df):
    """
    Derive stellar masses and ages from the catalog.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input catalog with MUV and redshift
    
    Returns
    -------
    df : pd.DataFrame
        Catalog with added physical properties
    """
    logger.info("Deriving stellar masses from MUV...")
    
    # Derive stellar masses using multiple methods for robustness
    log_Mstar_song, err_song = muv_to_stellar_mass(
        df['MUV'].values, df['z_best'].values, method='song2016'
    )
    log_Mstar_stef, err_stef = muv_to_stellar_mass(
        df['MUV'].values, df['z_best'].values, method='stefanon2021'
    )
    
    # Average of methods
    df['log_Mstar'] = (log_Mstar_song + log_Mstar_stef) / 2
    df['log_Mstar_err'] = np.sqrt(err_song**2 + err_stef**2) / 2 + 0.15  # Add systematic
    
    # Convert to halo mass using SHMR
    # At z~8-10, M_halo/M_star ~ 100-200
    log_ratio = 2.0 + 0.1 * (df['z_best'] - 8)  # Mild z-evolution
    df['log_Mhalo'] = df['log_Mstar'] + log_ratio
    
    # Estimate stellar ages
    logger.info("Estimating stellar ages...")
    t_stellar, t_stellar_err = estimate_stellar_age(
        df['MUV'].values, df['z_best'].values
    )
    df['t_stellar_Gyr'] = t_stellar
    df['t_stellar_err_Gyr'] = t_stellar_err
    
    # Calculate cosmic age at each redshift
    df['t_cosmic_Gyr'] = df['z_best'].apply(cosmic_age_at_z)
    
    # Calculate age excess (TEP signal)
    df['age_excess_Gyr'] = df['t_stellar_Gyr'] - df['t_cosmic_Gyr']
    
    logger.info(f"Stellar mass range: log(M*) = {df['log_Mstar'].min():.1f} - {df['log_Mstar'].max():.1f}")
    logger.info(f"Halo mass range: log(M_h) = {df['log_Mhalo'].min():.1f} - {df['log_Mhalo'].max():.1f}")
    
    return df

# ============================================================================
# QUALITY CUTS
# ============================================================================
def apply_quality_cuts(df, min_pz_gt7=0.5, min_delta_chisq=4.0, z_range=(8.0, 14.0)):
    """
    Apply quality cuts to select robust high-z candidates.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input catalog
    min_pz_gt7 : float
        Minimum P(z>7) probability
    min_delta_chisq : float
        Minimum delta chi-squared between high-z and low-z solutions
    z_range : tuple
        (z_min, z_max) for redshift selection
    
    Returns
    -------
    df_clean : pd.DataFrame
        Quality-filtered catalog
    """
    logger.info("Applying quality cuts...")
    
    n_initial = len(df)
    
    # Redshift range
    mask = (df['z_best'] >= z_range[0]) & (df['z_best'] <= z_range[1])
    logger.info(f"  z in [{z_range[0]}, {z_range[1]}]: {mask.sum()} / {n_initial}")
    
    # P(z>7) cut
    mask &= df['P_z_gt_7'] >= min_pz_gt7
    logger.info(f"  + P(z>7) >= {min_pz_gt7}: {mask.sum()}")
    
    # Delta chi-squared cut (distinguishes high-z from low-z degeneracy)
    mask &= df['delta_chisq'] >= min_delta_chisq
    logger.info(f"  + delta_chisq >= {min_delta_chisq}: {mask.sum()}")
    
    # Valid MUV
    mask &= df['MUV'].notna() & (df['MUV'] < -15) & (df['MUV'] > -25)
    logger.info(f"  + Valid MUV: {mask.sum()}")
    
    df_clean = df[mask].copy().reset_index(drop=True)
    
    logger.info(f"Quality cuts: {n_initial} -> {len(df_clean)} sources")
    
    return df_clean

# ============================================================================
# MAIN PROCESSING
# ============================================================================
def process_real_data():
    """
    Process real JWST data for TEP analysis.
    
    Returns
    -------
    df : pd.DataFrame
        Processed high-z galaxy sample with derived properties
    """
    logger.info("=" * 60)
    logger.info("TEP-JWST: Processing Real JWST Data")
    logger.info("=" * 60)
    
    # Load Hainline+23 catalog
    df = load_hainline_catalog()
    if df is None:
        return None
    
    # Derive physical properties
    df = derive_physical_properties(df)
    
    # Apply quality cuts
    df_clean = apply_quality_cuts(
        df, 
        min_pz_gt7=0.7,      # High confidence
        min_delta_chisq=4.0,  # Strong preference for high-z
        z_range=(8.0, 14.0)   # Focus on z>8
    )
    
    # Save processed data
    output_path = DATA_INTERIM / "jades_highz_physical.csv"
    df_clean.to_csv(output_path, index=False)
    logger.info(f"Saved: {output_path}")
    
    # Summary statistics
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Final sample: N = {len(df_clean)}")
    logger.info(f"Redshift: z = {df_clean['z_best'].median():.1f} (median)")
    logger.info(f"Stellar mass: log(M*) = {df_clean['log_Mstar'].median():.2f} (median)")
    logger.info(f"Halo mass: log(M_h) = {df_clean['log_Mhalo'].median():.2f} (median)")
    logger.info(f"Age excess: {df_clean['age_excess_Gyr'].median()*1000:.0f} Myr (median)")
    
    return df_clean

# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    df = process_real_data()
    
    if df is not None:
        print(f"\nProcessed {len(df)} high-z galaxies")
        print(df[['ID', 'z_best', 'MUV', 'log_Mstar', 'log_Mhalo', 't_cosmic_Gyr', 't_stellar_Gyr', 'age_excess_Gyr']].head(10))
