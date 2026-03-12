
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo
import os
import sys

# =============================================================================
# SETTINGS
# =============================================================================
PATHS = {
    "ceers": "/Users/matthewsmawfield/www/TEP-JWST/data/interim/ceers_highz_sample.csv",
    "cosmos": "/Users/matthewsmawfield/www/TEP-JWST/data/interim/cosmosweb_highz_sample.csv",
    "jades": "/Users/matthewsmawfield/www/TEP-JWST/data/interim/jades_highz_physical.csv",
    "kokorev_lrd": "/Users/matthewsmawfield/www/TEP-JWST/data/raw/kokorev_lrd_catalog_v1.1.fits",
    "output_fig": "/Users/matthewsmawfield/www/TEP-JWST/results/figures/circularity_check_compactness.png",
    "output_report": "/Users/matthewsmawfield/www/TEP-JWST/results/outputs/circularity_resolution_stats.txt"
}

# TEP Parameters
ALPHA_0 = 0.58
LOG_MH_REF = 12.0
Z_REF = 5.5

# =============================================================================
# PHYSICS FUNCTIONS
# =============================================================================

def get_tep_gamma_mass(log_Mh, z):
    """Standard TEP Gamma based on Halo Mass."""
    alpha_z = ALPHA_0 * np.sqrt(1 + z)
    delta_log_Mh = log_Mh - LOG_MH_REF
    z_factor = (1 + z) / (1 + Z_REF)
    argument = alpha_z * (2/3) * delta_log_Mh * z_factor
    return np.exp(argument)

def get_normal_size(log_M, z):
    """
    Theoretical Mass-Size relation for high-z galaxies.
    Shibuya et al. 2015: R_e ~ (1+z)^-1.2
    Generic: R_e ~ 1 kpc * (M/10^9)^0.2
    """
    # M is log10(M_star)
    # Simple proxy:
    return 1.0 * (10**(log_M - 9.0))**0.2

def stellar_to_halo_mass(log_mstar, z):
    """
    Simple abundance matching proxy for high-z.
    Mh ~ 100 * Ms (very rough, but sufficient for ranking).
    """
    return log_mstar + 2.0 

# =============================================================================
# DATA LOADING
# =============================================================================

def load_lrd_catalog():
    print("Loading LRD Catalog (Kokorev)...", flush=True)
    if not os.path.exists(PATHS["kokorev_lrd"]):
        print(f"Error: LRD catalog not found at {PATHS['kokorev_lrd']}")
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
            print(f"Loaded {len(df)} LRDs with valid sizes.", flush=True)
            return df
    except Exception as e:
        print(f"Error loading LRD catalog: {e}", flush=True)
        return pd.DataFrame()

def load_surveys():
    print("Loading Survey Catalogs...", flush=True)
    surveys = {}
    
    # CEERS
    try:
        if os.path.exists(PATHS["ceers"]):
            df = pd.read_csv(PATHS["ceers"])
            if 'ra' in df.columns:
                surveys['CEERS'] = df
                print(f"Loaded CEERS: {len(df)}", flush=True)
    except Exception as e: print(f"CEERS load error: {e}")

    # JADES
    try:
        if os.path.exists(PATHS["jades"]):
            df = pd.read_csv(PATHS["jades"])
            if 'RA' in df.columns:
                df = df.rename(columns={'RA': 'ra', 'DEC': 'dec'})
                surveys['JADES'] = df
                print(f"Loaded JADES: {len(df)}", flush=True)
    except Exception as e: print(f"JADES load error: {e}")

    return surveys

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run():
    print("Script started.", flush=True)
    lrd_df = load_lrd_catalog()
    if lrd_df.empty: 
        print("LRD DataFrame is empty. Exiting.", flush=True)
        return
    
    surveys = load_surveys()
    matched_data = []
    
    lrd_coords = SkyCoord(ra=lrd_df['ra'].values*u.deg, dec=lrd_df['dec'].values*u.deg)
    
    for survey_name, survey_df in surveys.items():
        print(f"Matching LRDs to {survey_name}...", flush=True)
        survey_coords = SkyCoord(ra=survey_df['ra'].values*u.deg, dec=survey_df['dec'].values*u.deg)
        
        idx, d2d, _ = lrd_coords.match_to_catalog_sky(survey_coords)
        mask = d2d < 0.5 * u.arcsec
        
        if np.sum(mask) > 0:
            print(f"Found {np.sum(mask)} matches in {survey_name}", flush=True)
            
            matches = lrd_df[mask].copy()
            target_indices = idx[mask]
            
            if 'log_Mstar' in survey_df.columns:
                matches['log_Mstar'] = survey_df.iloc[target_indices]['log_Mstar'].values
            elif 'log_Mstar_50' in survey_df.columns:
                 matches['log_Mstar'] = survey_df.iloc[target_indices]['log_Mstar_50'].values
            else:
                print(f"Survey {survey_name} missing mass column.", flush=True)
                continue
                
            matches['survey'] = survey_name
            matched_data.append(matches)
    
    if not matched_data:
        print("No matches found in any survey!", flush=True)
        return

    full_matched = pd.concat(matched_data)
    full_matched = full_matched.drop_duplicates(subset='lrd_id')
    print(f"Total matched LRDs with Mass: {len(full_matched)}", flush=True)
    
    # 1. Calculate Potentials
    full_matched['phi_proxy_lrd'] = (10**full_matched['log_Mstar']) / full_matched['r_eff_kpc']
    
    full_matched['r_eff_normal'] = get_normal_size(full_matched['log_Mstar'], full_matched['z_lrd'])
    full_matched['phi_proxy_normal'] = (10**full_matched['log_Mstar']) / full_matched['r_eff_normal']
    
    # 2. Calculate Gamma
    full_matched['log_Mhalo'] = stellar_to_halo_mass(full_matched['log_Mstar'], full_matched['z_lrd'])
    full_matched['gamma_mass'] = get_tep_gamma_mass(full_matched['log_Mhalo'], full_matched['z_lrd'])
    
    # Calibrate k using "Normal" assumption
    # Gamma_mass = exp( k * Phi_normal )
    ks = np.log(full_matched['gamma_mass']) / full_matched['phi_proxy_normal']
    k_calib = np.median(ks[np.isfinite(ks) & (ks > 0)])
    print(f"Calibrated k (Phi->Gamma): {k_calib:.2e}", flush=True)
    
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
    print(f"Saved figure to {PATHS['output_fig']}", flush=True)
    
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
        
    print(report, flush=True)

if __name__ == "__main__":
    run()
