#!/usr/bin/env python3
"""
TEP-JWST Step 037: Resolved Stellar Population Analysis

Tests the TEP prediction that inner regions of galaxies (deeper potential)
should appear older (redder) than outer regions, after controlling for
standard inside-out growth.

Methodology:
1. Load JADES photometry with CIRC_CONV (PSF-matched) apertures.
2. Define Inner (CIRC1, ~0.15") and Outer (CIRC4, ~0.50") apertures.
3. Calculate Color Gradient: Delta(F115W - F444W) = (Inner - Outer).
   Note: Positive gradient means Inner is Redder (Older).
4. Correlate Gradient with Total Stellar Mass.
   TEP Prediction: More massive galaxies (deeper potential) should show
   stronger positive gradients (older cores) or resist the standard blue-core trend.

Author: Matthew L. Smawfield
Date: January 2026
"""

import sys
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
import json
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.p_value_utils import format_p_value, safe_json_default

STEP_NUM = "38"
STEP_NAME = "resolved_gradients"
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"

def load_jades_photometry():
    """Load JADES PSF-matched photometry."""
    # We prioritize GOODS-S Deep because it has the most extensive CIRC_CONV data
    files = [
        DATA_RAW / "hlsp_jades_jwst_nircam_goods-s-deep_photometry_v2.0_catalog.fits",
        # GOODS-N often has different format, stick to GOODS-S for consistency if possible
        # DATA_RAW / "hlsp_jades_jwst_nircam_goods-n_photometry_v1.0_catalog.fits"
    ]
    
    dfs = []
    for f in files:
        if not f.exists():
            print_status(f"File not found: {f}", "WARN")
            continue
            
        print_status(f"Loading {f.name}...", "INFO")
        with fits.open(f) as hdul:
            # We want CIRC_CONV extension for PSF-matched apertures
            if 'CIRC_CONV' in hdul:
                data = hdul['CIRC_CONV'].data
                
                # Convert FITS data to DataFrame
                # We only need specific columns to save memory
                cols_needed = ['ID', 'RA', 'DEC']
                bands = ['F115W', 'F444W']
                circs = ['CIRC1', 'CIRC4'] # CIRC1=0.15", CIRC4=0.50" (Outer)
                
                # Check available columns
                available_cols = data.columns.names
                
                selected_data = {}
                
                # Helper to safely extract and convert to native endian
                def safe_extract(col_name):
                    arr = data[col_name]
                    if arr.dtype.byteorder == '>':
                        # NumPy 2.0 fix: use view to change byteorder interpretation, then byteswap to native
                        arr = arr.view(arr.dtype.newbyteorder('>')).byteswap().view(arr.dtype.newbyteorder('='))
                    return arr

                for c in cols_needed:
                    if c in available_cols:
                        selected_data[c] = safe_extract(c)
                
                found_bands = True
                for b in bands:
                    for c in circs:
                        col_name = f"{b}_{c}"
                        if col_name in available_cols:
                            selected_data[col_name] = safe_extract(col_name)
                        else:
                            print_status(f"Missing column: {col_name}", "WARN")
                            found_bands = False
                
                if found_bands:
                    df = pd.DataFrame(selected_data)
                    dfs.append(df)
            else:
                print_status(f"No CIRC_CONV extension in {f.name}", "WARN")
                
    if not dfs:
        return None
        
    combined = pd.concat(dfs, ignore_index=True)
    return combined

def load_physical_catalog():
    """Load JADES physical properties."""
    path = DATA_INTERIM / "jades_highz_physical.csv"
    if not path.exists():
        print_status(f"Physical catalog not found at {path}", "ERROR")
        return None
    return pd.read_csv(path)

def calculate_gradients(df_phot, df_phys):
    """Calculate color gradients and match with physical props."""
    
    print_status("Matching catalogs by coordinates...", "INFO")
    
    # Coordinate matching
    sc_phot = SkyCoord(ra=df_phot['RA'].values*u.deg, dec=df_phot['DEC'].values*u.deg)
    sc_phys = SkyCoord(ra=df_phys['RA'].values*u.deg, dec=df_phys['DEC'].values*u.deg)
    
    idx, d2d, _ = sc_phys.match_to_catalog_sky(sc_phot)
    
    # Strict matching (0.1 arcsec)
    mask = d2d < 0.1*u.arcsec
    
    matched_phys = df_phys[mask].copy()
    matched_phot = df_phot.iloc[idx[mask]].copy()
    
    # Merge
    merged = matched_phot.copy()
    merged['log_Mstar'] = matched_phys['log_Mstar'].values
    merged['z_best'] = matched_phys['z_best'].values
    merged['phys_id'] = matched_phys['ID'].values
    
    print_status(f"Matched {len(merged)} sources.", "INFO")
    
    # Define Bands and Apertures
    # Inner: CIRC1 (0.15") - Probes the core
    # Outer: CIRC4 (0.50") - Probes the disk/outskirts
    # Color: F115W - F444W (Rest-UV - Rest-Optical/NIR at high z)
    
    # Fluxes
    f_blue_in = merged['F115W_CIRC1']
    f_blue_out = merged['F115W_CIRC4']
    f_red_in = merged['F444W_CIRC1']
    f_red_out = merged['F444W_CIRC4']
    
    # Filter for valid detection in all bands (> 0 and finite)
    valid = (f_blue_in > 0) & (f_blue_out > 0) & (f_red_in > 0) & (f_red_out > 0)
    merged = merged[valid].copy()
    
    print_status(f"Sources with valid fluxes: {len(merged)}", "INFO")
    
    f_blue_in = merged['F115W_CIRC1']
    f_blue_out = merged['F115W_CIRC4']
    f_red_in = merged['F444W_CIRC1']
    f_red_out = merged['F444W_CIRC4']
    
    # Calculate Colors (AB Mag)
    # Mag = -2.5 * log10(Flux) + ZP (ZP cancels in gradient)
    # Color = Mag_Blue - Mag_Red
    #       = -2.5 log(Fb) - (-2.5 log(Fr))
    #       = -2.5 log(Fb/Fr)
    
    # Gradient = Color_In - Color_Out
    #          = [-2.5 log(Fb_in/Fr_in)] - [-2.5 log(Fb_out/Fr_out)]
    #          = -2.5 * [ log(Fb_in/Fr_in) - log(Fb_out/Fr_out) ]
    #          = -2.5 * log( (Fb_in * Fr_out) / (Fr_in * Fb_out) )
    
    ratio = (f_blue_in * f_red_out) / (f_red_in * f_blue_out)
    merged['color_gradient'] = -2.5 * np.log10(ratio)
    
    # Compactness (Flux Ratio: Inner/Outer in Red band)
    merged['compactness'] = f_red_in / f_red_out
    
    # Filter out unphysical gradients (artifacts)
    # |Gradient| > 2 mag is likely bad photometry or contamination
    merged = merged[(merged['color_gradient'] > -2.0) & (merged['color_gradient'] < 2.0)]
    
    # Filter for high-z (z > 4) to ensure F115W is UV and F444W is Optical
    merged = merged[merged['z_best'] > 4.0]
    
    return merged

def analyze_correlations(df):
    """Analyze correlations with Mass."""
    print_status("\n--- Gradient Correlations ---", "INFO")
    
    n = len(df)
    print_status(f"N = {n} (final sample z>4)", "INFO")
    
    if n < 10:
        print_status("Not enough sources for correlation.", "WARN")
        return None
        
    # 1. Gradient vs Mass
    rho, p = stats.spearmanr(df['log_Mstar'], df['color_gradient'])
    print_status(f"rho(Mass, Gradient) = {rho:+.3f}, p = {p:.2e}", "INFO")
    
    # 2. Gradient vs Redshift
    rho_z, p_z = stats.spearmanr(df['z_best'], df['color_gradient'])
    print_status(f"rho(z, Gradient) = {rho_z:+.3f}, p = {p_z:.2e}", "INFO")
    
    # 3. Compactness vs Mass
    rho_c, p_c = stats.spearmanr(df['log_Mstar'], df['compactness'])
    print_status(f"rho(Mass, Compactness) = {rho_c:+.3f}, p = {p_c:.2e}", "INFO")
    
    # Bin Analysis
    mass_bins = [(7, 8.5), (8.5, 9.5), (9.5, 11)]
    binned_results = []
    
    for m_lo, m_hi in mass_bins:
        sub = df[(df['log_Mstar'] >= m_lo) & (df['log_Mstar'] < m_hi)]
        if len(sub) > 5:
            mean_grad = sub['color_gradient'].mean()
            sem_grad = sub['color_gradient'].sem()
            binned_results.append({
                "mass_range": [float(m_lo), float(m_hi)],
                "n": int(len(sub)),
                "mean_gradient": float(mean_grad),
                "sem_gradient": float(sem_grad)
            })
            print_status(f"Mass {m_lo}-{m_hi}: N={len(sub)}, <Grad>={mean_grad:+.3f} +/- {sem_grad:.3f}", "INFO")
            
    return {
        'n': n,
        'rho_mass_grad': float(rho),
        'p_mass_grad': format_p_value(p),
        'rho_z_grad': float(rho_z),
        'p_z_grad': format_p_value(p_z),
        'binned_results': binned_results
    }

def main():
    logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=PROJECT_ROOT / "logs" / f"step_{STEP_NUM}_{STEP_NAME}.log")
    set_step_logger(logger)
    
    print_status("Loading Data...", "INFO")
    df_phot = load_jades_photometry()
    df_phys = load_physical_catalog()
    
    if df_phot is None or df_phys is None:
        return
    
    df_grad = calculate_gradients(df_phot, df_phys)
    
    if df_grad is not None and not df_grad.empty:
        results = analyze_correlations(df_grad)
        
        if results:
            # Save results
            out_file = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
            with open(out_file, 'w') as f:
                json.dump(results, f, indent=2, default=safe_json_default)
            print_status(f"Saved to {out_file}", "INFO")
    else:
        print_status("No valid gradient data derived.", "WARN")

if __name__ == "__main__":
    main()
