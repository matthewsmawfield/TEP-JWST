"""
TEP-JWST Step 4: Labbé+23 "Impossible Galaxy" Analysis
Analyzes the original Labbé et al. 2023 sample of massive high-z galaxies
to test the TEP Chronological Shear prediction.

These galaxies are "impossible" because their high stellar masses require
star formation histories that would exceed the cosmic age at their redshifts.
Under TEP, this is explained by enhanced proper time accumulation in deep
gravitational potentials (Chronological Shear).
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from astropy.cosmology import Planck18
from astropy import units as u
from astropy.io import ascii
import json

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

# ============================================================================
# STELLAR MASS TO FORMATION TIME
# ============================================================================
def mass_to_formation_time(log_Mstar, sfr_mode='continuous'):
    """
    Estimate the minimum star formation timescale required to build up
    the observed stellar mass.
    
    For a galaxy with stellar mass M*, assuming continuous star formation
    at a rate SFR, the formation time is:
    
    t_form = M* / SFR
    
    At high-z, typical SFRs are 10-100 Msun/yr for massive galaxies.
    
    Parameters
    ----------
    log_Mstar : float or array
        Log10 stellar mass in solar masses
    sfr_mode : str
        Star formation history assumption:
        - 'continuous': constant SFR
        - 'burst': short burst (100 Myr)
        - 'rising': rising SFR (common at high-z)
    
    Returns
    -------
    t_form : float or array
        Minimum formation timescale in Gyr
    """
    Mstar = 10**log_Mstar
    
    if sfr_mode == 'continuous':
        # Typical SFR for massive galaxies: 10-100 Msun/yr
        # Use mass-dependent SFR from main sequence relation
        # log(SFR) ~ 0.8 * log(M*) - 6.5 at z~8
        log_sfr = 0.8 * log_Mstar - 6.5
        sfr = 10**log_sfr  # Msun/yr
        t_form = (Mstar / sfr) / 1e9  # Convert yr to Gyr
        
    elif sfr_mode == 'burst':
        # Assume 100 Myr burst duration (typical for high-z)
        t_form = 0.1 * np.ones_like(log_Mstar)
        
    elif sfr_mode == 'rising':
        # Rising SFH: SFR(t) ~ t, integrated mass ~ SFR_final * t^2 / 2
        # Solve for t given final mass
        # More realistic for high-z galaxies
        sfr_final = 10**(0.8 * log_Mstar - 6.0)  # Higher normalization
        t_form = np.sqrt(2 * Mstar / sfr_final) / 1e9
    
    return t_form

def estimate_stellar_age_from_mass(log_Mstar, z):
    """
    Estimate the minimum stellar population age required to explain
    the observed stellar mass, assuming reasonable SFR limits.
    
    This is what makes these galaxies "impossible" - the required
    ages often exceed the cosmic age at the observed redshift.
    
    Parameters
    ----------
    log_Mstar : float or array
        Log10 stellar mass
    z : float or array
        Redshift
    
    Returns
    -------
    t_stellar : float or array
        Estimated minimum stellar age in Gyr
    t_stellar_err : float or array
        Uncertainty (factor of ~2)
    """
    # Use continuous SFR as baseline
    t_continuous = mass_to_formation_time(log_Mstar, 'continuous')
    t_rising = mass_to_formation_time(log_Mstar, 'rising')
    
    # Average of models with uncertainty
    t_stellar = (t_continuous + t_rising) / 2
    t_stellar_err = np.abs(t_continuous - t_rising) / 2 + 0.1  # Add systematic floor
    
    # Apply SED fitting correction factor
    # SED fits typically give older ages due to outshining bias
    # Correction factor ~1.5 based on simulations
    sed_factor = 1.5
    t_stellar = t_stellar * sed_factor
    t_stellar_err = t_stellar_err * sed_factor
    
    return t_stellar, t_stellar_err

# ============================================================================
# LOAD LABBÉ+23 DATA
# ============================================================================
def load_labbe_2023():
    """
    Load the Labbé et al. 2023 "impossible galaxy" sample from their
    published data release.
    
    Returns
    -------
    df : pd.DataFrame
        The Labbé+23 sample with derived properties
    """
    data_path = DATA_RAW / "red-massive-candidates-main" / "sample_revision3_2207.12446.ecsv"
    
    if not data_path.exists():
        logger.error(f"Labbé+23 data not found: {data_path}")
        return None
    
    logger.info(f"Loading Labbé+23 sample from {data_path.name}")
    
    # Read ECSV format
    table = ascii.read(data_path, format='ecsv')
    df = table.to_pandas()
    
    # Rename columns for clarity
    df = df.rename(columns={
        'id': 'ID',
        'ra': 'RA',
        'dec': 'DEC',
        'z': 'z_phot',
        'mass': 'log_Mstar',
        'masslo': 'log_Mstar_lo',
        'masshi': 'log_Mstar_hi'
    })
    
    logger.info(f"Loaded {len(df)} galaxies")
    logger.info(f"  Redshift range: z = {df['z_phot'].min():.2f} - {df['z_phot'].max():.2f}")
    logger.info(f"  Stellar mass range: log(M*) = {df['log_Mstar'].min():.2f} - {df['log_Mstar'].max():.2f}")
    
    return df

# ============================================================================
# TEP ANALYSIS
# ============================================================================
def derive_tep_properties(df):
    """
    Derive TEP-relevant properties for the Labbé+23 sample.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input catalog
    
    Returns
    -------
    df : pd.DataFrame
        Catalog with TEP properties added
    """
    logger.info("Deriving TEP properties...")
    
    # Calculate cosmic age at each redshift
    df['t_cosmic_Gyr'] = df['z_phot'].apply(cosmic_age_at_z)
    df['t_cosmic_Myr'] = df['t_cosmic_Gyr'] * 1000
    
    # Estimate stellar ages from mass
    t_stellar, t_stellar_err = estimate_stellar_age_from_mass(
        df['log_Mstar'].values, 
        df['z_phot'].values
    )
    df['t_stellar_Gyr'] = t_stellar
    df['t_stellar_err_Gyr'] = t_stellar_err
    
    # Calculate age excess (the "impossible" signal)
    df['age_excess_Gyr'] = df['t_stellar_Gyr'] - df['t_cosmic_Gyr']
    df['age_excess_Myr'] = df['age_excess_Gyr'] * 1000
    
    # Flag "impossible" galaxies (t_stellar > t_cosmic)
    df['is_impossible'] = df['age_excess_Gyr'] > 0
    n_impossible = df['is_impossible'].sum()
    logger.info(f"  'Impossible' galaxies (t_stellar > t_cosmic): {n_impossible}/{len(df)}")
    
    # Convert stellar mass to halo mass using SHMR
    # At z~8, M_halo/M_star ~ 100-200
    log_ratio = 2.0 + 0.1 * (df['z_phot'] - 8)
    df['log_Mhalo'] = df['log_Mstar'] + log_ratio
    
    # Calculate TEP chronological shear prediction
    # Γ_t = α * (M_halo / M_ref)^(1/3)
    alpha = 0.58  # From TEP-H0
    M_ref = 1e10
    df['M_halo'] = 10**df['log_Mhalo']
    df['gamma_t_predicted'] = alpha * (df['M_halo'] / M_ref)**(1/3)
    
    # TEP-corrected cosmic age (what it would appear to be with TEP)
    df['t_tep_predicted_Gyr'] = df['t_cosmic_Gyr'] * (1 + df['gamma_t_predicted'])
    
    logger.info(f"  Halo mass range: log(M_h) = {df['log_Mhalo'].min():.2f} - {df['log_Mhalo'].max():.2f}")
    logger.info(f"  Predicted Γ_t range: {df['gamma_t_predicted'].min():.2f} - {df['gamma_t_predicted'].max():.2f}")
    
    return df

def fit_mass_age_relation(df):
    """
    Fit the mass-age excess relation to test the TEP M^(1/3) prediction.
    
    Parameters
    ----------
    df : pd.DataFrame
        Catalog with age_excess_Gyr and log_Mhalo
    
    Returns
    -------
    results : dict
        Fit results
    """
    logger.info("Fitting mass-age relation...")
    
    # Use only galaxies with positive age excess (the "impossible" ones)
    mask = df['age_excess_Gyr'] > 0
    df_fit = df[mask].copy()
    
    if len(df_fit) < 3:
        logger.warning(f"Only {len(df_fit)} galaxies with positive age excess")
        # Use all galaxies for correlation analysis
        df_fit = df.copy()
    
    # Log-log fit: log(age_excess) vs log(M_halo)
    log_M = np.log10(df_fit['M_halo'].values)
    
    # For age excess, handle negative values by using absolute value
    # and tracking sign
    age_excess = df_fit['age_excess_Gyr'].values
    
    # Spearman correlation (rank-based, handles any distribution)
    rho, p_spearman = stats.spearmanr(df_fit['log_Mhalo'], df_fit['t_stellar_Gyr'])
    logger.info(f"  Spearman ρ (M_halo vs t_stellar): {rho:.3f}, p = {p_spearman:.4f}")
    
    # Pearson correlation
    r, p_pearson = stats.pearsonr(df_fit['log_Mhalo'], df_fit['t_stellar_Gyr'])
    logger.info(f"  Pearson r (M_halo vs t_stellar): {r:.3f}, p = {p_pearson:.4f}")
    
    # Linear regression in log-log space for positive age excess
    positive_mask = age_excess > 0
    if positive_mask.sum() >= 3:
        log_age = np.log10(age_excess[positive_mask])
        log_M_pos = log_M[positive_mask]
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_M_pos, log_age)
        
        logger.info(f"  Power-law fit (positive age excess only, N={positive_mask.sum()}):")
        logger.info(f"    Slope: {slope:.3f} ± {std_err:.3f}")
        logger.info(f"    TEP prediction: 0.333")
        logger.info(f"    Tension: {abs(slope - 1/3) / std_err:.1f}σ")
    else:
        slope, intercept, r_value, std_err = np.nan, np.nan, np.nan, np.nan
        logger.warning("  Insufficient positive age excess values for power-law fit")
    
    results = {
        "n_total": len(df),
        "n_impossible": df['is_impossible'].sum(),
        "n_fit": len(df_fit),
        "spearman_rho": rho,
        "spearman_p": p_spearman,
        "pearson_r": r,
        "pearson_p": p_pearson,
        "slope": slope,
        "slope_err": std_err,
        "intercept": intercept,
        "r_squared": r_value**2 if not np.isnan(r_value) else np.nan,
        "tep_prediction": 1/3,
        "slope_tension_sigma": abs(slope - 1/3) / std_err if not np.isnan(std_err) and std_err > 0 else np.nan
    }
    
    return results

def bootstrap_analysis(df, n_bootstrap=1000):
    """
    Bootstrap uncertainty estimation for the mass-age correlation.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input catalog
    n_bootstrap : int
        Number of bootstrap iterations
    
    Returns
    -------
    results : dict
        Bootstrap statistics
    """
    logger.info(f"Running bootstrap analysis (N={n_bootstrap})...")
    
    n = len(df)
    slopes = []
    rhos = []
    
    for i in range(n_bootstrap):
        # Resample with replacement
        idx = np.random.choice(n, size=n, replace=True)
        df_boot = df.iloc[idx]
        
        # Calculate correlation
        rho, _ = stats.spearmanr(df_boot['log_Mhalo'], df_boot['t_stellar_Gyr'])
        rhos.append(rho)
        
        # Try power-law fit on positive age excess
        mask = df_boot['age_excess_Gyr'] > 0
        if mask.sum() >= 3:
            log_M = np.log10(df_boot.loc[mask, 'M_halo'])
            log_age = np.log10(df_boot.loc[mask, 'age_excess_Gyr'])
            try:
                slope, _, _, _, _ = stats.linregress(log_M, log_age)
                slopes.append(slope)
            except:
                pass
    
    slopes = np.array(slopes)
    rhos = np.array(rhos)
    
    results = {
        "rho_median": np.median(rhos),
        "rho_16": np.percentile(rhos, 16),
        "rho_84": np.percentile(rhos, 84),
        "slope_median": np.median(slopes) if len(slopes) > 0 else np.nan,
        "slope_16": np.percentile(slopes, 16) if len(slopes) > 0 else np.nan,
        "slope_84": np.percentile(slopes, 84) if len(slopes) > 0 else np.nan,
        "slope_std": np.std(slopes) if len(slopes) > 0 else np.nan,
        "n_successful_slope": len(slopes)
    }
    
    logger.info(f"  Spearman ρ: {results['rho_median']:.3f} [{results['rho_16']:.3f}, {results['rho_84']:.3f}]")
    if len(slopes) > 0:
        logger.info(f"  Slope: {results['slope_median']:.3f} [{results['slope_16']:.3f}, {results['slope_84']:.3f}]")
    
    return results

# ============================================================================
# MAIN ANALYSIS
# ============================================================================
def run_labbe_analysis():
    """
    Run the full Labbé+23 analysis to test TEP predictions.
    
    Returns
    -------
    results : dict
        Complete analysis results
    """
    logger.info("=" * 60)
    logger.info("TEP-JWST: Labbé+23 'Impossible Galaxy' Analysis")
    logger.info("=" * 60)
    
    # Load data
    df = load_labbe_2023()
    if df is None:
        return None
    
    # Derive TEP properties
    df = derive_tep_properties(df)
    
    # Fit mass-age relation
    fit_results = fit_mass_age_relation(df)
    
    # Bootstrap analysis
    boot_results = bootstrap_analysis(df, n_bootstrap=1000)
    
    # Summary statistics
    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    
    logger.info(f"Sample: N = {len(df)} Labbé+23 galaxies")
    logger.info(f"Redshift: z = {df['z_phot'].median():.2f} (median)")
    logger.info(f"Stellar mass: log(M*) = {df['log_Mstar'].median():.2f} (median)")
    logger.info(f"Cosmic age: {df['t_cosmic_Myr'].median():.0f} Myr (median)")
    logger.info(f"Estimated stellar age: {df['t_stellar_Gyr'].median()*1000:.0f} Myr (median)")
    logger.info(f"Age excess: {df['age_excess_Myr'].median():.0f} Myr (median)")
    logger.info(f"'Impossible' galaxies: {fit_results['n_impossible']}/{len(df)}")
    
    logger.info("-" * 40)
    logger.info("TEP Test:")
    logger.info(f"  Mass-age correlation (Spearman): ρ = {fit_results['spearman_rho']:.3f}, p = {fit_results['spearman_p']:.4f}")
    if not np.isnan(fit_results['slope']):
        logger.info(f"  Power-law slope: {fit_results['slope']:.3f} ± {fit_results['slope_err']:.3f}")
        logger.info(f"  TEP prediction: 0.333 (M^(1/3))")
        if not np.isnan(fit_results['slope_tension_sigma']):
            consistency = "CONSISTENT" if fit_results['slope_tension_sigma'] < 2 else "INCONSISTENT"
            logger.info(f"  Tension: {fit_results['slope_tension_sigma']:.1f}σ → {consistency}")
    
    # Save results
    df.to_csv(DATA_INTERIM / "labbe_2023_tep_analysis.csv", index=False)
    logger.info(f"\nSaved: {DATA_INTERIM / 'labbe_2023_tep_analysis.csv'}")
    
    # Save results JSON
    all_results = {
        "sample_size": len(df),
        "z_median": float(df['z_phot'].median()),
        "log_Mstar_median": float(df['log_Mstar'].median()),
        "t_cosmic_median_Myr": float(df['t_cosmic_Myr'].median()),
        "t_stellar_median_Myr": float(df['t_stellar_Gyr'].median() * 1000),
        "age_excess_median_Myr": float(df['age_excess_Myr'].median()),
        "n_impossible": int(fit_results['n_impossible']),
        "spearman_rho": float(fit_results['spearman_rho']),
        "spearman_p": float(fit_results['spearman_p']),
        "pearson_r": float(fit_results['pearson_r']),
        "pearson_p": float(fit_results['pearson_p']),
        "slope": float(fit_results['slope']) if not np.isnan(fit_results['slope']) else None,
        "slope_err": float(fit_results['slope_err']) if not np.isnan(fit_results['slope_err']) else None,
        "tep_prediction": 0.333,
        "slope_tension_sigma": float(fit_results['slope_tension_sigma']) if not np.isnan(fit_results['slope_tension_sigma']) else None,
        "bootstrap_rho_median": float(boot_results['rho_median']),
        "bootstrap_slope_median": float(boot_results['slope_median']) if not np.isnan(boot_results['slope_median']) else None,
        "bootstrap_slope_std": float(boot_results['slope_std']) if not np.isnan(boot_results['slope_std']) else None,
        "data_source": "Labbe et al. 2023, Nature 616, 266",
        "reference": "arXiv:2207.12446"
    }
    
    with open(RESULTS_DIR / "labbe_2023_tep_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved: {RESULTS_DIR / 'labbe_2023_tep_results.json'}")
    
    return all_results, df

# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    results, df = run_labbe_analysis()
    
    if results is not None:
        print("\n" + "=" * 60)
        print("Labbé+23 Sample Summary Table")
        print("=" * 60)
        print(df[['ID', 'z_phot', 'log_Mstar', 't_cosmic_Myr', 't_stellar_Gyr', 'age_excess_Myr', 'is_impossible']].to_string(index=False))
