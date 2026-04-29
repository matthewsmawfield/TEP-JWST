#!/usr/bin/env python3
"""
TEP-JWST Step 132: LRD Population Differential Temporal Topology Analysis

Expands the single-LRD simulation (Step 41) to a population-level analysis
using the Kokorev et al. (2024) catalog of 260 photometrically selected
Little Red Dots at 4 < z < 9.

This addresses the feedback that the differential temporal topology mechanism
must be shown to be universal, not anecdotal.

Data Source:
- Kokorev et al. 2024, arXiv:2401.09981
- GitHub: https://github.com/VasilyKokorev/lrd_phot

Inputs:
- Downloaded LRD catalog (FITS)

Outputs:
- results/outputs/step_132_lrd_validation.json
- results/outputs/step_132_lrd_population.csv
- results/figures/lrd_population_time_bubble.png
"""

import sys
import numpy as np
import pandas as pd
from astropy.cosmology import Planck18 as cosmo
from astropy.io import fits
from pathlib import Path
import json
import urllib.request
import matplotlib.pyplot as plt

# =============================================================================
# PATHS AND LOGGER
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging
from scripts.utils.tep_model import (
    KAPPA_GAL, KAPPA_GAL_UNCERTAINTY, 
    LOG_MH_REF, Z_REF, PHI_REF_0,
    get_phi_from_log_mh, compute_gamma_t_from_phi
)

STEP_NUM = "132"  # Pipeline step number
STEP_NAME = "lrd_validation"  # Used in log / output filenames

OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"  # JSON output directory
FIGURES_PATH = PROJECT_ROOT / "results" / "figures"  # Publication figures directory
DATA_PATH = PROJECT_ROOT / "data" / "raw"  # Raw external catalogues
LOGS_PATH = PROJECT_ROOT / "logs"  # Log directory

for p in [OUTPUT_PATH, FIGURES_PATH, DATA_PATH, LOGS_PATH]:
    p.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")  # Step-specific logger
set_step_logger(logger)

# =============================================================================
# CONSTANTS & PARAMETERS
# KAPPA_GAL, Z_REF, LOG_MH_REF imported from scripts.utils.tep_model
# =============================================================================

G = 4.301e-6  # kpc (km/s)^2 / M_sun
PHI_REF_VIR = PHI_REF_0  # Reference potential depth from tep_model.py
T_SALPETER = 0.045  # Gyr (Eddington e-folding time)

# LRD-specific parameters
CONCENTRATION_FACTOR = 10.0  # Ratio of central to virial potential (compact cores)
R_E_TYPICAL_PC = 150  # Typical LRD effective radius in parsecs

# =============================================================================
# DATA DOWNLOAD
# =============================================================================

LRD_CATALOG_URL = "https://github.com/VasilyKokorev/lrd_phot/raw/master/lrd_table_v1.1.fits"
LRD_CATALOG_PATH = DATA_PATH / "kokorev_lrd_catalog_v1.1.fits"


def download_lrd_catalog():
    """Download the Kokorev et al. 2024 LRD catalog if not present."""
    if LRD_CATALOG_PATH.exists():
        print_status(f"LRD catalog already exists: {LRD_CATALOG_PATH}", "INFO")
        return True
    
    print_status(f"Downloading LRD catalog from {LRD_CATALOG_URL}...", "INFO")
    try:
        urllib.request.urlretrieve(LRD_CATALOG_URL, LRD_CATALOG_PATH)
        print_status(f"Downloaded to {LRD_CATALOG_PATH}", "INFO")
        return True
    except Exception as e:
        print_status(f"Failed to download: {e}", "ERROR")
        return False


def load_lrd_catalog():
    """Load the LRD catalog and extract relevant columns."""
    with fits.open(LRD_CATALOG_PATH) as hdul:
        data = hdul[1].data
        
        # Convert to pandas DataFrame directly using astropy's Table
        from astropy.table import Table
        table = Table.read(LRD_CATALOG_PATH)
        df = table.to_pandas()
    
    # Rename columns to standard names
    col_map = {
        'z_phot': 'z',
        'lbol': 'log_Lbol',
        'av': 'Av',
        'r_eff_50_phys': 'Re_pc',
        'muv': 'Muv',
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    
    # Filter to valid entries with physical redshifts (LRDs are at z > 4)
    df = df[pd.notna(df['z']) & (df['z'] > 4) & (df['z'] < 10)].copy()
    
    print_status(f"Loaded {len(df)} LRDs from catalog", "INFO")
    print_status(f"Redshift range: {df['z'].min():.2f} - {df['z'].max():.2f}", "INFO")
    print_status(f"Median z: {df['z'].median():.2f}", "INFO")
    return df

# =============================================================================
# TEP CALCULATIONS
# =============================================================================

MUV_REF = -19.5
LOG_MSTAR_REF = 8.7
ML_SLOPE = 0.4


def muv_to_log_mstar(muv):
    """Conservative UV-luminosity stellar-mass proxy for Kokorev LRDs."""
    return -ML_SLOPE * (np.asarray(muv) - MUV_REF) + LOG_MSTAR_REF

def estimate_halo_mass(log_Mstar, z):
    """
    Estimate halo mass from stellar mass using abundance matching.
    Uses Behroozi+19 relation at high-z.
    """
    # LRD-specific Behroozi+19-like relation with z-evolution
    # NOTE: differs from shared stellar_to_halo_mass (+2.0 fixed offset)
    # by using a z-dependent offset (+1.5 at z=5, +1.7 at z=7)
    log_Mh = log_Mstar + 1.5 + 0.1 * (z - 5)
    return np.clip(log_Mh, 9.0, 14.0)


def estimate_sigma_from_size(log_Mstar, Re_pc):
    """
    Estimate velocity dispersion from stellar mass and size.
    sigma^2 ~ G M / (5 R_e) (Wolf+10 estimator)
    """
    M_star = 10**log_Mstar
    Re_kpc = Re_pc / 1000.0
    sigma_sq = G * M_star / (5 * Re_kpc)
    return np.sqrt(np.maximum(sigma_sq, 1.0))


def get_tep_gamma_potential(phi_local, phi_ref, z):
    """
    Wrapper for backward compatibility, using the harmonized kernel.
    The phi_ref argument is kept for API compatibility but is not used
    (reference potential is built into compute_gamma_t_from_phi).
    """
    return compute_gamma_t_from_phi(phi_local, z)


def calculate_differential_topology(z, log_Mh, concentration=CONCENTRATION_FACTOR):
    """
    Calculate the differential temporal topology between
    the galactic center (BH) and the stellar halo.
    
    Returns:
        gamma_halo: Enhancement factor at virial radius
        gamma_cen: Enhancement factor at center
        boost_factor: Differential BH growth boost
    """
    M_h = 10**log_Mh
    
    # Virial radius scales with mass and redshift
    R_vir = 30.0 * (M_h / 1e11)**(1/3) * (10.0 / (1 + z))  # kpc
    
    # Potentials (proportional to V^2)
    Phi_vir = get_phi_from_log_mh(log_Mh)
    Phi_cen = Phi_vir * concentration
    
    # TEP factors
    gamma_halo = get_tep_gamma_potential(Phi_vir, PHI_REF_VIR, z)
    gamma_cen = get_tep_gamma_potential(Phi_cen, PHI_REF_VIR, z)
    
    # Differential growth
    t_cosmic = cosmo.age(z).value  # Gyr
    delta_gamma = gamma_cen - gamma_halo
    extra_efolds = delta_gamma * t_cosmic / T_SALPETER
    boost_factor = np.exp(np.clip(extra_efolds, -50, 50))  # Prevent overflow
    
    return gamma_halo, gamma_cen, boost_factor, t_cosmic


def analyze_lrd_population(df):
    """
    Apply Differential Temporal Topology analysis to the full LRD population.
    """
    results = []
    
    for _, row in df.iterrows():
        z = row['z']
        
        # Estimate halo mass from object-level stellar-mass information when
        # available. The Kokorev catalog does not ship stellar masses, so the
        # canonical fallback is the same conservative MUV proxy used by step 142.
        log_Mstar = row.get('log_Mstar', np.nan)
        mass_source = 'catalog'
        if not pd.notna(log_Mstar) and pd.notna(row.get('Muv')) and -30 < row.get('Muv') < 0:
            log_Mstar = float(muv_to_log_mstar(row['Muv']))
            mass_source = 'MUV_proxy'
        if pd.notna(log_Mstar):
            log_Mh = estimate_halo_mass(log_Mstar, z)
        else:
            log_Mh = 11.0  # Default for LRDs
            mass_source = 'default_logMh'
        
        # Estimate concentration from compactness
        # LRDs are very compact, so concentration is high
        if pd.notna(row.get('Re_pc')) and row['Re_pc'] > 0:
            Re_pc = row['Re_pc']  # FITS unit is pc.
            # Smaller radius = higher concentration
            concentration = np.clip(500 / Re_pc, 5, 50)
        else:
            Re_pc = R_E_TYPICAL_PC
            concentration = CONCENTRATION_FACTOR
        
        # Calculate differential temporal topology
        gamma_halo, gamma_cen, boost, t_cosmic = calculate_differential_topology(
            z, log_Mh, concentration
        )
        
        results.append({
            'id': row['id'],
            'field': row.get('field', 'unknown'),
            'z': z,
            'Muv': row.get('Muv', np.nan),
            'log_Mstar': log_Mstar,
            'mass_source': mass_source,
            'log_Mh': log_Mh,
            'Re_pc': Re_pc,
            'concentration': concentration,
            't_cosmic_Gyr': t_cosmic,
            'gamma_halo': gamma_halo,
            'gamma_cen': gamma_cen,
            'delta_gamma': gamma_cen - gamma_halo,
            'boost_factor': boost,
            'log_boost': np.log10(boost) if boost > 0 else 0,
        })
    
    return pd.DataFrame(results)


def create_population_figure(df_results):
    """Create visualization of the LRD population Differential Temporal Topology analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Boost factor vs redshift
    ax1 = axes[0, 0]
    valid = df_results['log_boost'] > 0
    sc = ax1.scatter(df_results.loc[valid, 'z'], 
                     df_results.loc[valid, 'log_boost'],
                     c=df_results.loc[valid, 'log_Mh'],
                     cmap='viridis', alpha=0.7, s=30)
    ax1.set_xlabel('Redshift')
    ax1.set_ylabel(r'$\log_{10}$(Boost Factor)')
    ax1.set_title('BH Growth Boost vs Redshift')
    plt.colorbar(sc, ax=ax1, label=r'$\log M_h$')
    
    # 2. Delta Gamma distribution
    ax2 = axes[0, 1]
    ax2.hist(df_results['delta_gamma'], bins=30, edgecolor='black', alpha=0.7)
    ax2.axvline(df_results['delta_gamma'].median(), color='red', linestyle='--',
                label=f"Median = {df_results['delta_gamma'].median():.2f}")
    ax2.set_xlabel(r'$\Delta\Gamma_t$ (Center - Halo)')
    ax2.set_ylabel('Count')
    ax2.set_title('Differential Temporal Enhancement')
    ax2.legend()
    
    # 3. Gamma_cen vs Gamma_halo
    ax3 = axes[1, 0]
    ax3.scatter(df_results['gamma_halo'], df_results['gamma_cen'],
                c=df_results['z'], cmap='plasma', alpha=0.7, s=30)
    ax3.plot([0, df_results['gamma_cen'].max()], 
             [0, df_results['gamma_cen'].max()], 
             'k--', alpha=0.5, label='1:1')
    ax3.set_xlabel(r'$\Gamma_t$ (Halo)')
    ax3.set_ylabel(r'$\Gamma_t$ (Center)')
    ax3.set_title('Core vs Halo Enhancement')
    ax3.legend()
    
    # 4. Summary statistics by redshift bin
    ax4 = axes[1, 1]
    z_bins = [4, 5, 6, 7, 8, 9, 10]
    z_centers = [4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
    median_boosts = []
    for i in range(len(z_bins) - 1):
        mask = (df_results['z'] >= z_bins[i]) & (df_results['z'] < z_bins[i+1])
        if mask.sum() > 0:
            median_boosts.append(df_results.loc[mask, 'log_boost'].median())
        else:
            median_boosts.append(np.nan)
    
    ax4.bar(z_centers, median_boosts, width=0.8, edgecolor='black', alpha=0.7)
    ax4.set_xlabel('Redshift Bin')
    ax4.set_ylabel(r'Median $\log_{10}$(Boost)')
    ax4.set_title('Boost Factor by Redshift')
    
    plt.tight_layout()
    fig_path = FIGURES_PATH / "lrd_population_time_bubble.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print_status(f"Saved figure to {fig_path}", "INFO")


def run_analysis():
    """Main analysis function."""
    print_status("=" * 70, "INFO")
    print_status(f"STEP {STEP_NUM}: LRD Population Differential Temporal Topology Analysis", "INFO")
    print_status("=" * 70, "INFO")
    
    # Download catalog
    if not download_lrd_catalog():
        print_status("Cannot proceed without LRD catalog", "ERROR")
        return
    
    # Load data
    df = load_lrd_catalog()
    
    if len(df) == 0:
        print_status("No valid LRDs in catalog", "ERROR")
        return
    
    # Analyze population
    print_status(f"\nAnalyzing {len(df)} LRDs...", "INFO")
    df_results = analyze_lrd_population(df)
    
    # Summary statistics
    print_status("\n" + "=" * 50, "INFO")
    print_status("POPULATION SUMMARY", "INFO")
    print_status("=" * 50, "INFO")
    print_status(f"Total LRDs analyzed: {len(df_results)}", "INFO")
    print_status(f"Redshift range: {df_results['z'].min():.2f} - {df_results['z'].max():.2f}", "INFO")
    print_status(f"Median z: {df_results['z'].median():.2f}", "INFO")
    print_status(f"\nTemporal Enhancement:", "INFO")
    print_status(f"  Median Gamma_halo: {df_results['gamma_halo'].median():.2f}", "INFO")
    print_status(f"  Median Gamma_cen: {df_results['gamma_cen'].median():.2f}", "INFO")
    print_status(f"  Median Delta_Gamma: {df_results['delta_gamma'].median():.2f}", "INFO")
    print_status(f"\nBH Growth Boost:", "INFO")
    print_status(f"  Median log(Boost): {df_results['log_boost'].median():.1f}", "INFO")
    print_status(f"  Mean log(Boost): {df_results['log_boost'].mean():.1f}", "INFO")
    
    # Fraction with significant boost
    significant = (df_results['log_boost'] > 3).sum()  # > 1000x boost
    print_status(f"  LRDs with >1000x boost: {significant}/{len(df_results)} ({100*significant/len(df_results):.1f}%)", "INFO")
    
    # Save results
    csv_path = OUTPUT_PATH / f"step_{STEP_NUM}_lrd_population.csv"
    df_results.to_csv(csv_path, index=False)
    print_status(f"\nSaved CSV to {csv_path}", "INFO")
    
    # Create figure
    create_population_figure(df_results)
    
    # JSON summary
    summary = {
        "test": f"Step {STEP_NUM}: LRD Population Differential Temporal Topology Analysis",
        "data_source": "Kokorev et al. 2024 (arXiv:2401.09981)",
        "sample_size": len(df_results),
        "parameters": {
            "kappa_gal": KAPPA_GAL,
            "kappa_gal_uncertainty": KAPPA_GAL_UNCERTAINTY,
            "z_ref": Z_REF,
            "t_salpeter_Gyr": T_SALPETER,
            "stellar_mass_proxy": "MUV_proxy_if_catalog_log_Mstar_missing",
        },
        "mass_sources": df_results["mass_source"].value_counts().to_dict(),
        "redshift_range": {
            "min": float(df_results['z'].min()),
            "max": float(df_results['z'].max()),
            "median": float(df_results['z'].median()),
        },
        "temporal_enhancement": {
            "median_gamma_halo": float(df_results['gamma_halo'].median()),
            "median_gamma_cen": float(df_results['gamma_cen'].median()),
            "median_delta_gamma": float(df_results['delta_gamma'].median()),
        },
        "bh_growth_boost": {
            "median_log_boost": float(df_results['log_boost'].median()),
            "mean_log_boost": float(df_results['log_boost'].mean()),
            "fraction_gt_1000x": float(significant / len(df_results)),
        },
        "conclusion": (
            "After correcting the Kokorev radius units and using an object-level "
            "MUV stellar-mass proxy where catalog stellar masses are absent, "
            f"{100*significant/len(df_results):.0f}% of {len(df_results)} LRDs show >1000x "
            "differential BH growth boost. This is a mechanism stress test, not "
            "a calibrated resolution of the observed M_BH/M_* anomaly."
        ),
    }
    
    json_path = OUTPUT_PATH / f"step_{STEP_NUM}_lrd_validation.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print_status(f"Saved JSON to {json_path}", "INFO")
    
    print_status(f"\nStep {STEP_NUM} complete.", "INFO")
    return df_results


if __name__ == "__main__":
    run_analysis()
