"""
TEP-JWST Step 2: Mass-Age Analysis
Tests the TEP prediction that stellar age excess correlates with halo mass
following the M^(1/3) scaling from the Universal Critical Density framework.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.optimize import curve_fit
import json

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
RESULTS_DIR = PROJECT_ROOT / "results" / "outputs"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

# Ensure directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# COSMOLOGICAL FUNCTIONS
# ============================================================================
from astropy.cosmology import Planck18
from astropy import units as u

def cosmic_age_at_z(z):
    """Calculate cosmic age at redshift z in Gyr."""
    return Planck18.age(z).to(u.Gyr).value

# ============================================================================
# STELLAR-TO-HALO MASS RELATION (SHMR)
# ============================================================================
def stellar_to_halo_mass(log_Mstar, z=8.0):
    """
    Convert stellar mass to halo virial mass using the SHMR.
    
    At high-z, we use the abundance matching relation from Behroozi+19.
    
    Parameters
    ----------
    log_Mstar : float or array
        Log10 stellar mass in solar masses
    z : float
        Redshift (affects SHMR normalization)
    
    Returns
    -------
    log_Mhalo : float or array
        Log10 halo virial mass in solar masses
    """
    # Simplified SHMR: M_halo / M_star ~ 100 at characteristic mass
    # At high-z, this ratio increases
    z_factor = 1 + 0.1 * (z - 8)  # Evolution with redshift
    log_ratio = 2.0 + 0.2 * z_factor  # ~100-200 at z=8-12
    
    log_Mhalo = log_Mstar + log_ratio
    return log_Mhalo

def halo_virial_velocity(log_Mhalo, z=8.0):
    """
    Calculate halo virial velocity from mass.
    
    V_vir = (G * M_halo / R_vir)^0.5
    
    Using V_vir ~ 200 km/s * (M_halo / 10^12 Msun)^(1/3) * (1+z)^0.5
    
    Parameters
    ----------
    log_Mhalo : float or array
        Log10 halo mass in solar masses
    z : float
        Redshift
    
    Returns
    -------
    V_vir : float or array
        Virial velocity in km/s
    """
    M_ref = 1e12  # Reference mass
    M_halo = 10**log_Mhalo
    
    V_vir = 200 * (M_halo / M_ref)**(1/3) * (1 + z)**0.5
    return V_vir

# ============================================================================
# TEP CHRONOLOGICAL SHEAR MODEL
# ============================================================================
def tep_age_enhancement(log_Mhalo, alpha=0.58, beta=1/3):
    """
    Calculate TEP-predicted age enhancement factor.
    
    The TEP framework predicts that proper time accumulates faster in deep
    gravitational potentials. For high-z halos:
    
    Γ_t = α * (M_halo / M_ref)^β
    
    where β = 1/3 from the Universal Scaling Law (TEP-UCD Paper 7).
    
    Parameters
    ----------
    log_Mhalo : float or array
        Log10 halo mass in solar masses
    alpha : float
        TEP coupling constant (calibrated from TEP-H0: α = 0.58 ± 0.16)
    beta : float
        Scaling exponent (predicted: 1/3 from TEP-UCD)
    
    Returns
    -------
    gamma_t : float or array
        Age enhancement factor (dimensionless)
    """
    M_ref = 1e10  # Reference mass scale
    M_halo = 10**log_Mhalo
    
    gamma_t = alpha * (M_halo / M_ref)**beta
    return gamma_t

def tep_predicted_age(t_cosmic, gamma_t):
    """
    Calculate TEP-predicted observed stellar age.
    
    t_observed = t_cosmic * (1 + Γ_t)
    
    Stars in deep potentials experience more proper time cycles,
    so they appear "older" than the cosmic age would allow.
    
    Parameters
    ----------
    t_cosmic : float or array
        Cosmic age at the galaxy's redshift in Gyr
    gamma_t : float or array
        Age enhancement factor
    
    Returns
    -------
    t_predicted : float or array
        TEP-predicted stellar age in Gyr
    """
    return t_cosmic * (1 + gamma_t)

# ============================================================================
# POWER-LAW FITTING
# ============================================================================
def power_law(x, a, b):
    """Power law: y = a * x^b"""
    return a * x**b

def fit_mass_age_relation(log_Mhalo, age_excess):
    """
    Fit the mass-age excess relation to test M^(1/3) scaling.
    
    Parameters
    ----------
    log_Mhalo : array
        Log10 halo masses
    age_excess : array
        Age excess in Gyr (t_stellar - t_cosmic)
    
    Returns
    -------
    results : dict
        Fit results including slope, intercept, and statistics
    """
    # Convert to linear for power-law fit
    M_halo = 10**log_Mhalo
    
    # Ensure positive age excess for log fit
    valid = age_excess > 0
    if valid.sum() < 3:
        logger.warning("Insufficient positive age excess values for fit")
        return None
    
    M_valid = M_halo[valid]
    age_valid = age_excess[valid]
    
    # Log-log fit for power law
    log_M = np.log10(M_valid)
    log_age = np.log10(age_valid)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_M, log_age)
    
    results = {
        "slope": slope,
        "slope_err": std_err,
        "intercept": intercept,
        "r_squared": r_value**2,
        "p_value": p_value,
        "n_points": len(M_valid),
        "tep_prediction": 1/3,  # TEP predicts slope = 1/3
        "slope_tension_sigma": abs(slope - 1/3) / std_err if std_err > 0 else np.inf
    }
    
    return results

# ============================================================================
# STATISTICAL TESTS
# ============================================================================
def bootstrap_slope(log_Mhalo, age_excess, n_bootstrap=1000):
    """
    Bootstrap estimation of slope uncertainty.
    
    Parameters
    ----------
    log_Mhalo : array
        Log10 halo masses
    age_excess : array
        Age excess values
    n_bootstrap : int
        Number of bootstrap iterations
    
    Returns
    -------
    results : dict
        Bootstrap statistics
    """
    n = len(log_Mhalo)
    slopes = []
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        M_boot = log_Mhalo[idx]
        age_boot = age_excess[idx]
        
        # Simple linear fit in log-log space
        valid = age_boot > 0
        if valid.sum() < 3:
            continue
        
        log_M = np.log10(10**M_boot[valid])
        log_age = np.log10(age_boot[valid])
        
        slope, _, _, _, _ = stats.linregress(log_M, log_age)
        slopes.append(slope)
    
    slopes = np.array(slopes)
    
    return {
        "slope_median": np.median(slopes),
        "slope_mean": np.mean(slopes),
        "slope_std": np.std(slopes),
        "slope_16": np.percentile(slopes, 16),
        "slope_84": np.percentile(slopes, 84),
        "n_successful": len(slopes)
    }

def permutation_test(log_Mhalo, age_excess, n_permutations=10000):
    """
    Permutation test for significance of mass-age correlation.
    
    Parameters
    ----------
    log_Mhalo : array
        Log10 halo masses
    age_excess : array
        Age excess values
    n_permutations : int
        Number of permutations
    
    Returns
    -------
    p_value : float
        Permutation p-value
    """
    # Observed correlation
    r_obs, _ = stats.spearmanr(log_Mhalo, age_excess)
    
    # Permutation distribution
    r_perm = np.zeros(n_permutations)
    for i in range(n_permutations):
        age_shuffled = np.random.permutation(age_excess)
        r_perm[i], _ = stats.spearmanr(log_Mhalo, age_shuffled)
    
    # Two-sided p-value
    p_value = np.mean(np.abs(r_perm) >= np.abs(r_obs))
    
    return p_value, r_obs

# ============================================================================
# SYNTHETIC DATA FOR DEMONSTRATION
# ============================================================================
def generate_synthetic_highz_sample(n=50, seed=42):
    """
    Generate synthetic high-z galaxy sample for pipeline development.
    
    This creates realistic distributions based on observed JWST samples.
    
    Parameters
    ----------
    n : int
        Number of galaxies
    seed : int
        Random seed for reproducibility
    
    Returns
    -------
    df : pd.DataFrame
        Synthetic galaxy sample
    """
    np.random.seed(seed)
    
    # Redshift distribution (peaked at z~9)
    z = np.random.normal(9.0, 1.5, n)
    z = np.clip(z, 7.0, 14.0)
    
    # Stellar mass distribution (log-normal)
    log_Mstar = np.random.normal(9.5, 0.8, n)
    log_Mstar = np.clip(log_Mstar, 8.0, 11.5)
    
    # Calculate halo masses
    log_Mhalo = stellar_to_halo_mass(log_Mstar, z=z)
    
    # Cosmic age at each redshift
    t_cosmic = np.array([cosmic_age_at_z(zi) for zi in z])
    
    # TEP-enhanced stellar ages
    gamma_t = tep_age_enhancement(log_Mhalo, alpha=0.58, beta=1/3)
    t_stellar_tep = t_cosmic * (1 + gamma_t)
    
    # Add observational scatter (SED fitting uncertainty ~30%)
    t_stellar_obs = t_stellar_tep * np.random.lognormal(0, 0.3, n)
    
    # Calculate age excess
    age_excess = t_stellar_obs - t_cosmic
    
    df = pd.DataFrame({
        "ID": [f"SYN-{i:03d}" for i in range(n)],
        "z": z,
        "log_Mstar": log_Mstar,
        "log_Mhalo": log_Mhalo,
        "t_cosmic_Gyr": t_cosmic,
        "t_stellar_Gyr": t_stellar_obs,
        "age_excess_Gyr": age_excess,
        "gamma_t_true": gamma_t,
        "V_vir_km_s": halo_virial_velocity(log_Mhalo, z)
    })
    
    return df

# ============================================================================
# MAIN ANALYSIS
# ============================================================================
def run_mass_age_analysis(use_synthetic=True):
    """
    Run the mass-age correlation analysis.
    
    Parameters
    ----------
    use_synthetic : bool
        If True, use synthetic data for pipeline development
    
    Returns
    -------
    results : dict
        Analysis results
    """
    logger.info("=" * 60)
    logger.info("TEP-JWST: Mass-Age Correlation Analysis")
    logger.info("=" * 60)
    
    results = {}
    
    # Step 1: Load or generate data
    if use_synthetic:
        logger.info("Using synthetic data for pipeline development")
        logger.warning("NOTE: Replace with real JWST data for publication!")
        df = generate_synthetic_highz_sample(n=50, seed=42)
    else:
        # Load real data from interim
        labbe_path = DATA_INTERIM / "labbe_2023_sample.csv"
        jades_path = DATA_INTERIM / "jades_spec_highz.csv"
        
        if not labbe_path.exists():
            logger.error("Data files not found. Run step_1_data_ingestion.py first.")
            return None
        
        df_labbe = pd.read_csv(labbe_path)
        df_jades = pd.read_csv(jades_path)
        
        # Combine samples (requires additional processing)
        logger.info("Loading observed samples...")
        df = df_labbe  # Placeholder
    
    logger.info(f"Sample size: N = {len(df)}")
    logger.info(f"Redshift range: z = {df['z'].min():.1f} - {df['z'].max():.1f}")
    logger.info(f"Stellar mass range: log(M*) = {df['log_Mstar'].min():.1f} - {df['log_Mstar'].max():.1f}")
    
    results["sample_size"] = len(df)
    results["z_range"] = [df['z'].min(), df['z'].max()]
    results["log_Mstar_range"] = [df['log_Mstar'].min(), df['log_Mstar'].max()]
    
    # Step 2: Fit mass-age relation
    logger.info("-" * 40)
    logger.info("Fitting Mass-Age Relation")
    logger.info("-" * 40)
    
    fit_results = fit_mass_age_relation(
        df['log_Mhalo'].values,
        df['age_excess_Gyr'].values
    )
    
    if fit_results:
        logger.info(f"Slope: {fit_results['slope']:.3f} ± {fit_results['slope_err']:.3f}")
        logger.info(f"TEP Prediction: {fit_results['tep_prediction']:.3f}")
        logger.info(f"Tension with TEP: {fit_results['slope_tension_sigma']:.1f}σ")
        logger.info(f"R²: {fit_results['r_squared']:.3f}")
        logger.info(f"p-value: {fit_results['p_value']:.4f}")
        
        results["fit"] = fit_results
    
    # Step 3: Bootstrap uncertainty
    logger.info("-" * 40)
    logger.info("Bootstrap Analysis (N=1000)")
    logger.info("-" * 40)
    
    boot_results = bootstrap_slope(
        df['log_Mhalo'].values,
        df['age_excess_Gyr'].values,
        n_bootstrap=1000
    )
    
    logger.info(f"Slope (bootstrap): {boot_results['slope_median']:.3f} [{boot_results['slope_16']:.3f}, {boot_results['slope_84']:.3f}]")
    results["bootstrap"] = boot_results
    
    # Step 4: Permutation test
    logger.info("-" * 40)
    logger.info("Permutation Test (N=10000)")
    logger.info("-" * 40)
    
    p_perm, r_obs = permutation_test(
        df['log_Mhalo'].values,
        df['age_excess_Gyr'].values,
        n_permutations=10000
    )
    
    logger.info(f"Spearman ρ: {r_obs:.3f}")
    logger.info(f"Permutation p-value: {p_perm:.4f}")
    
    results["permutation"] = {
        "spearman_rho": r_obs,
        "p_value": p_perm
    }
    
    # Step 5: Save results
    logger.info("-" * 40)
    logger.info("Saving Results")
    logger.info("-" * 40)
    
    # Save processed data
    df.to_csv(DATA_INTERIM / "highz_sample_processed.csv", index=False)
    logger.info(f"Saved: {DATA_INTERIM / 'highz_sample_processed.csv'}")
    
    # Save results JSON
    results_json = {
        "sample_size": results["sample_size"],
        "z_range": results["z_range"],
        "log_Mstar_range": results["log_Mstar_range"],
        "fit_slope": fit_results["slope"] if fit_results else None,
        "fit_slope_err": fit_results["slope_err"] if fit_results else None,
        "tep_prediction": 1/3,
        "slope_tension_sigma": fit_results["slope_tension_sigma"] if fit_results else None,
        "r_squared": fit_results["r_squared"] if fit_results else None,
        "bootstrap_slope": boot_results["slope_median"],
        "bootstrap_err": boot_results["slope_std"],
        "spearman_rho": r_obs,
        "permutation_p": p_perm,
        "synthetic_data": use_synthetic
    }
    
    with open(RESULTS_DIR / "mass_age_analysis.json", 'w') as f:
        json.dump(results_json, f, indent=2)
    logger.info(f"Saved: {RESULTS_DIR / 'mass_age_analysis.json'}")
    
    # Summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"N = {results['sample_size']} high-z galaxies")
    logger.info(f"Mass-Age Slope: {boot_results['slope_median']:.2f} ± {boot_results['slope_std']:.2f}")
    logger.info(f"TEP Prediction: 0.33 (M^(1/3) scaling)")
    if fit_results:
        consistency = "CONSISTENT" if fit_results["slope_tension_sigma"] < 2 else "INCONSISTENT"
        logger.info(f"Result: {consistency} with TEP prediction")
    
    return results

# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TEP-JWST Mass-Age Analysis")
    parser.add_argument("--real", action="store_true", help="Use real data (requires data download)")
    
    args = parser.parse_args()
    
    results = run_mass_age_analysis(use_synthetic=not args.real)
