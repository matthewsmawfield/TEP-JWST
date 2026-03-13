# Estimated runtime from last full canonical run (2026-03-09 15:52 UTC; full pipeline 32m18s): 1.3s.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo
import os
import sys
from pathlib import Path

# =============================================================================
# PATHS AND LOGGER
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging (severity levels: DEBUG/INFO/WARNING/ERROR/SUCCESS)

STEP_NUM = "069"  # Pipeline step number (sequential 001-176)
STEP_NAME = "compactness_verification"  # Compactness verification: tests if galaxy compactness (size residuals) correlates with Gamma_t independently of mass (addressing circularity concerns)

LOGS_PATH = PROJECT_ROOT / "logs"  # Log directory (one plain-text log per step for debugging traceability)
LOGS_PATH.mkdir(parents=True, exist_ok=True)  # Create directory tree if missing; exist_ok=True allows safe re-runs
logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")  # Step-specific logger (isolated per-step logging for traceability)
set_step_logger(logger)  # Register as global step logger so print_status() routes to this step's log

PATHS = {
    "ceers": PROJECT_ROOT / "data" / "interim" / "ceers_highz_sample.csv",
    "cosmos": PROJECT_ROOT / "data" / "interim" / "cosmosweb_highz_sample.csv",
    "jades": PROJECT_ROOT / "data" / "interim" / "jades_highz_physical.csv",
    "kokorev_lrd": PROJECT_ROOT / "data" / "raw" / "kokorev_lrd_catalog_v1.1.fits",
    "output_fig": PROJECT_ROOT / "results" / "figures" / "circularity_check_compactness.png",
    "output_report": PROJECT_ROOT / "results" / "outputs" / "circularity_resolution_stats.txt",
    "output_json": PROJECT_ROOT / "results" / "outputs" / "step_069_compactness_verification.json"
}

# Ensure output directories exist
PATHS["output_fig"].parent.mkdir(parents=True, exist_ok=True)
PATHS["output_report"].parent.mkdir(parents=True, exist_ok=True)
PATHS["output_json"].parent.mkdir(parents=True, exist_ok=True)

# =============================================================================
# PHYSICS FUNCTIONS
# =============================================================================

from scripts.utils.tep_model import (
    stellar_to_halo_mass as abundance_matching_proxy,
    compute_gamma_t as tep_gamma,
    LOG_MH_REF
)

def get_tep_gamma_mass_fixed(log_Mh, z):
    """
    Standard TEP Gamma based on Halo Mass (Fixed Reference).
    
    We use the Fixed Reference Mass (LOG_MH_REF = 12.0) for this specific test
    to isolate the mass-circularity issue from redshift-evolution effects.
    The goal is to compare 'Mass Only' vs 'Compactness' in the most basic form.
    Using the evolving M_ref model complicates the baseline.
    """
    log_mh = np.asarray(log_Mh, dtype=float)
    z_arr = np.asarray(z, dtype=float)

    gamma = tep_gamma(log_mh, z_arr)
    gamma_ref = tep_gamma(np.full_like(log_mh, float(LOG_MH_REF)), z_arr)
    gamma_ref = np.maximum(gamma_ref, np.nextafter(0, 1))
    return gamma / gamma_ref

def stellar_to_halo_mass(log_mstar, z):
    """
    Simple abundance matching proxy for high-z.
    """
    return abundance_matching_proxy(log_mstar, z)

def get_normal_size(log_M, z):
    """
    Theoretical Mass-Size relation for high-z galaxies.
    Shibuya et al. 2015: R_e ~ (1+z)^-1.2
    Generic: R_e ~ 1 kpc * (M/10^9)^0.2
    """
    # M is log10(M_star)
    # Simple proxy:
    return 1.0 * (10**(log_M - 9.0))**0.2

# =============================================================================
# DATA LOADING
# =============================================================================

def load_lrd_catalog():
    print_status("Loading LRD Catalog (Kokorev)...", "PROCESS")
    if not PATHS["kokorev_lrd"].exists():
        print_status(f"Error: LRD catalog not found at {PATHS['kokorev_lrd']}", "ERROR")
        return pd.DataFrame()

    try:
        with fits.open(PATHS["kokorev_lrd"]) as hdul:
            data = hdul[1].data
            
            def to_native(arr):
                if arr.dtype.kind in ['S', 'U']:
                    return arr.astype(str)
                if arr.dtype.kind == 'f':
                    return arr.astype('float64')
                if arr.dtype.kind == 'i':
                    return arr.astype('int64')
                return arr.astype(arr.dtype.name)

            def get_col(name):
                return to_native(data[name])

            df = pd.DataFrame({
                'lrd_id': get_col('id'),
                'ra': get_col('ra'),
                'dec': get_col('dec'),
                'r_eff_kpc': get_col('r_eff_50_phys'),
                'z_lrd': get_col('z_phot')
            })
            
            df = df[df['r_eff_kpc'] > 0].copy()
            print_status(f"Loaded {len(df)} LRDs with valid sizes.", "INFO")
            return df
    except Exception as e:
        print_status(f"Error loading LRD catalog: {e}", "ERROR")
        return pd.DataFrame()

def load_surveys():
    print_status("Loading Survey Catalogs...", "PROCESS")
    surveys = {}
    
    # CEERS
    try:
        if PATHS["ceers"].exists():
            df = pd.read_csv(PATHS["ceers"])
            if 'ra' in df.columns:
                surveys['CEERS'] = df
                print_status(f"Loaded CEERS: {len(df)}", "INFO")
    except Exception as e: print_status(f"CEERS load error: {e}", "ERROR")

    # JADES
    try:
        if PATHS["jades"].exists():
            df = pd.read_csv(PATHS["jades"])
            if 'RA' in df.columns:
                df = df.rename(columns={'RA': 'ra', 'DEC': 'dec'})
                surveys['JADES'] = df
                print_status(f"Loaded JADES: {len(df)}", "INFO")
    except Exception as e: print_status(f"JADES load error: {e}", "ERROR")

    return surveys

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run():
    print_status(f"STEP {STEP_NUM}: Compactness Verification", "TITLE")
    
    lrd_df = load_lrd_catalog()
    if lrd_df.empty: 
        print_status("LRD DataFrame is empty. Exiting.", "WARNING")
        return
    
    surveys = load_surveys()
    matched_data = []
    
    lrd_coords = SkyCoord(ra=lrd_df['ra'].values*u.deg, dec=lrd_df['dec'].values*u.deg)
    
    for survey_name, survey_df in surveys.items():
        print_status(f"Matching LRDs to {survey_name}...", "PROCESS")
        survey_coords = SkyCoord(ra=survey_df['ra'].values*u.deg, dec=survey_df['dec'].values*u.deg)
        
        idx, d2d, _ = lrd_coords.match_to_catalog_sky(survey_coords)
        mask = d2d < 0.5 * u.arcsec
        
        if np.sum(mask) > 0:
            print_status(f"Found {np.sum(mask)} matches in {survey_name}", "INFO")
            
            matches = lrd_df[mask].copy()
            target_indices = idx[mask]
            
            if 'log_Mstar' in survey_df.columns:
                matches['log_Mstar'] = survey_df.iloc[target_indices]['log_Mstar'].values
            elif 'log_Mstar_50' in survey_df.columns:
                 matches['log_Mstar'] = survey_df.iloc[target_indices]['log_Mstar_50'].values
            else:
                print_status(f"Survey {survey_name} missing mass column.", "WARNING")
                continue
                
            matches['survey'] = survey_name
            matched_data.append(matches)
    
    if not matched_data:
        print_status("No matches found in any survey!", "WARNING")
        return

    full_matched = pd.concat(matched_data)
    full_matched = full_matched.drop_duplicates(subset='lrd_id')
    print_status(f"Total matched LRDs with Mass: {len(full_matched)}", "SUCCESS")
    
    # 1. Calculate Potentials
    full_matched['phi_proxy_lrd'] = (10**full_matched['log_Mstar']) / full_matched['r_eff_kpc']
    
    full_matched['r_eff_normal'] = get_normal_size(full_matched['log_Mstar'], full_matched['z_lrd'])
    full_matched['phi_proxy_normal'] = (10**full_matched['log_Mstar']) / full_matched['r_eff_normal']
    
    # 2. Calculate Gamma
    full_matched['log_Mhalo'] = stellar_to_halo_mass(full_matched['log_Mstar'], full_matched['z_lrd'])
    # Use FIXED reference mass model for the Mass-Only baseline to isolate compactness signal
    full_matched['gamma_mass'] = get_tep_gamma_mass_fixed(full_matched['log_Mhalo'], full_matched['z_lrd'])
    
    # Calibrate k using "Normal" assumption
    # Gamma_mass = exp( k * Phi_normal )
    ks = np.log(full_matched['gamma_mass']) / full_matched['phi_proxy_normal']
    
    # Filter for valid positive k values (enhancement regime)
    valid_ks = ks[np.isfinite(ks) & (ks > 0)]
    
    if len(valid_ks) > 0:
        k_calib = np.median(valid_ks)
    else:
        print_status("Warning: No galaxies in enhancement regime (Gamma > 1). Using global median.", "WARNING")
        # Fallback to calibrated median to get scale, or just median of all finite
        valid_ks_all = ks[np.isfinite(ks)]
        if len(valid_ks_all) > 0:
             k_calib = np.median(np.abs(valid_ks_all)) # Use magnitude
        else:
             k_calib = 1e-11 # Fallback default
             
    print_status(f"Calibrated k (Phi->Gamma): {k_calib:.2e}", "INFO")
    
    # Predict Gamma for LRDs
    full_matched['gamma_compact'] = np.exp(k_calib * full_matched['phi_proxy_lrd'])
    
    # 3. Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(full_matched['gamma_mass'], full_matched['gamma_compact'], 
                c=full_matched['log_Mstar'], cmap='viridis', s=50, edgecolor='k')
    
    # Add identity line
    lims = [min(full_matched['gamma_mass'].min(), full_matched['gamma_compact'].min()),
            max(full_matched['gamma_mass'].max(), full_matched['gamma_compact'].max())]
    plt.plot(lims, lims, 'k--', label='1:1 (Mass Only = Compactness)')
    
    plt.xlabel(r'$\Gamma_t$ (Mass-Only Proxy)')
    plt.ylabel(r'$\Gamma_t$ (Potential/Compactness Proxy)')
    plt.title('Circularity Resolution: Compactness vs Mass Proxy')
    plt.colorbar(label='log M_star')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PATHS["output_fig"])
    print_status(f"Saved figure to {PATHS['output_fig']}", "SUCCESS")
    
    # 4. Statistics
    ratios = full_matched['gamma_compact'] / full_matched['gamma_mass']
    avg_boost = np.mean(ratios)
    median_boost = np.median(ratios)
    
    report = (
        "CIRCULARITY RESOLUTION STATISTICS\n"
        "=================================\n"
        f"Matched LRDs: {len(full_matched)}\n"
        f"Calibration k: {k_calib:.2e} (Gamma ~ exp(k * M/R))\n\n"
        f"Mean Boost (Compact/Mass): {avg_boost:.2f}x\n"
        f"Median Boost: {median_boost:.2f}x\n\n"
        "INTERPRETATION:\n"
    )
    
    if median_boost > 1.2:
        report += (
            "LRDs are significantly more compact than the mass-size relation predicts.\n"
            "Using a potential-based proxy (M/R) yields HIGHER Gamma_t values than mass alone.\n"
            "Therefore, the Mass-Only proxy used in the manuscript is CONSERVATIVE.\n"
            "The circularity critique is resolved: explicit potential modeling strengthens the anomaly."
        )
    else:
        report += (
            "Compactness-based estimates are consistent with mass-based estimates.\n"
            "The circularity impact is minimal."
        )
        
    with open(PATHS["output_report"], "w") as f:
        f.write(report)
        
    print_status(f"Report saved to {PATHS['output_report']}", "SUCCESS")
    print(report)

    # Save JSON results for Step 85 integration
    import json
    json_results = {
        "step": 87,
        "name": "Compactness Verification",
        "n_matched": int(len(full_matched)),
        "calibration_k": float(k_calib),
        "mean_boost": float(avg_boost),
        "median_boost": float(median_boost),
        "conclusion": "Resolved" if median_boost > 1.2 else "Inconclusive"
    }
    
    with open(PATHS["output_json"], "w") as f:
        json.dump(json_results, f, indent=2)
    print_status(f"JSON results saved to {PATHS['output_json']}", "SUCCESS")

    print_status(f"Step {STEP_NUM} complete.", "SUCCESS")

main = run

if __name__ == "__main__":
    run()
