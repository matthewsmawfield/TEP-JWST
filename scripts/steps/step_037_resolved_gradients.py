#!/usr/bin/env python3
# Estimated runtime from last full canonical run (2026-03-09 15:52 UTC; full pipeline 32m18s): 0.9s.
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
from scipy import stats  # Hypothesis tests and correlation
from pathlib import Path
from astropy.io import fits  # FITS catalogue I/O
from astropy.coordinates import SkyCoord  # Sky coordinate matching between catalogues
from astropy import units as u  # Astropy unit system
import json
import warnings

warnings.filterwarnings('ignore')  # Suppress astropy / numpy runtime warnings

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging (severity levels: DEBUG/INFO/WARNING/ERROR/SUCCESS)
from scripts.utils.p_value_utils import format_p_value, safe_json_default  # Safe p-value formatting (prevents floating-point underflow at p < 1e-300) & JSON serialiser for numpy types

STEP_NUM = "037"  # Pipeline step number (sequential 001-176)
STEP_NAME = "resolved_gradients"  # Resolved gradients: tests TEP prediction that inner galaxy regions (deeper potential) appear older/redder
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"  # JSON output directory (machine-readable statistical results)
DATA_RAW = PROJECT_ROOT / "data" / "raw"  # Raw catalogue directory (FITS files from MAST/ESA)
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"  # Processed intermediate products (CSV format for step-to-step data flow)


def _native_array(arr):
    """Convert big-endian FITS arrays to native byte order for scipy/pandas compatibility."""
    out = np.array(arr)
    if out.dtype.byteorder == '>':
        out = out.byteswap().view(out.dtype.newbyteorder('='))
    return out


def _load_photometry_extension(file_path):
    with fits.open(file_path, memmap=False) as hdul:
        ext_name = 'CIRC_CONV' if 'CIRC_CONV' in hdul else ('CIRC' if 'CIRC' in hdul else None)
        if ext_name is None:
            print_status(f"No circular-aperture extension in {file_path.name}", "WARN")
            return None

        data = hdul[ext_name].data
        available_cols = data.columns.names
        selected_data = {}
        cols_needed = ['ID', 'RA', 'DEC']
        bands = ['F115W', 'F444W']
        circs = ['CIRC1', 'CIRC4']

        for c in cols_needed:
            if c in available_cols:
                selected_data[c] = _native_array(data[c])

        found_bands = True
        for band in bands:
            for circ in circs:
                col_name = f"{band}_{circ}"
                if col_name in available_cols:
                    selected_data[col_name] = _native_array(data[col_name])
                else:
                    print_status(f"Missing column: {col_name}", "WARN")
                    found_bands = False

        if not found_bands:
            return None

        df = pd.DataFrame(selected_data)
        df['ID'] = pd.to_numeric(df['ID'], errors='coerce')
        df = df.dropna(subset=['ID'])
        print_status(f"Loaded {len(df)} rows from {file_path.name} [{ext_name}]", "INFO")
        return df

def load_jades_photometry():
    """Load JADES PSF-matched photometry."""
    files = [
        DATA_RAW / "JADES_z_gt_8_Candidates_Hainline_et_al.fits",
        DATA_RAW / "hlsp_jades_jwst_nircam_goods-s-deep_photometry_v2.0_catalog.fits",
    ]

    for f in files:
        if not f.exists():
            print_status(f"File not found: {f}", "WARN")
            continue

        print_status(f"Loading {f.name}...", "INFO")
        df = _load_photometry_extension(f)
        if df is not None and not df.empty:
            return df

    return None

def load_physical_catalog():
    """Load JADES physical properties."""
    path = DATA_INTERIM / "jades_highz_physical.csv"
    if not path.exists():
        print_status(f"Physical catalog not found at {path}", "ERROR")
        return None
    df = pd.read_csv(path)
    df['ID'] = pd.to_numeric(df['ID'], errors='coerce')
    return df.dropna(subset=['ID'])

def calculate_gradients(df_phot, df_phys):
    """Calculate color gradients and match with physical props."""
    print_status("Matching catalogs by JADES ID...", "INFO")

    merged = df_phot.merge(
        df_phys[['ID', 'log_Mstar', 'z_best']],
        on='ID',
        how='inner',
        suffixes=('', '_phys'),
    )
    merged['phys_id'] = merged['ID']

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

    merged = merged.dropna(subset=['log_Mstar', 'z_best', 'color_gradient', 'compactness'])

    return merged

def analyze_correlations(df):
    """Analyze correlations with Mass."""
    print_status("\n--- Gradient Correlations ---", "INFO")
    
    n = len(df)
    print_status(f"N = {n} (final sample z>4)", "INFO")
    
    if n < 10:
        print_status("Not enough sources for correlation.", "WARN")
        return None

    def safe_spearman(x, y):
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 3:
            return None, None
        x_valid = np.asarray(x)[mask]
        y_valid = np.asarray(y)[mask]
        if np.unique(x_valid).size < 2 or np.unique(y_valid).size < 2:
            return None, None
        rho, p = stats.spearmanr(x_valid, y_valid)
        if not np.isfinite(rho) or not np.isfinite(p):
            return None, None
        return float(rho), float(p)
        
    # 1. Gradient vs Mass
    rho, p = safe_spearman(df['log_Mstar'].values, df['color_gradient'].values)
    if rho is None:
        print_status("rho(Mass, Gradient) unavailable after finite-value filtering", "WARN")
    else:
        print_status(f"rho(Mass, Gradient) = {rho:+.3f}, p = {p:.2e}", "INFO")
    
    # 2. Gradient vs Redshift
    rho_z, p_z = safe_spearman(df['z_best'].values, df['color_gradient'].values)
    if rho_z is None:
        print_status("rho(z, Gradient) unavailable after finite-value filtering", "WARN")
    else:
        print_status(f"rho(z, Gradient) = {rho_z:+.3f}, p = {p_z:.2e}", "INFO")
    
    # 3. Compactness vs Mass
    rho_c, p_c = safe_spearman(df['log_Mstar'].values, df['compactness'].values)
    if rho_c is None:
        print_status("rho(Mass, Compactness) unavailable after finite-value filtering", "WARN")
    else:
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
        'rho_mass_grad': rho,
        'p_mass_grad': format_p_value(p),
        'rho_z_grad': rho_z,
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
        status = {"status": "skipped", "reason": "JADES photometry not available (no CIRC_CONV apertures)"}
        with open(OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json", "w") as _f:
            json.dump(status, _f, indent=2)
        return
    
    df_grad = calculate_gradients(df_phot, df_phys)
    
    out_file = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    if df_grad is not None and not df_grad.empty:
        interim_file = DATA_INTERIM / "jades_resolved_gradients.csv"
        df_grad.to_csv(interim_file, index=False)
        print_status(f"Saved resolved gradients to {interim_file}", "INFO")
        results = analyze_correlations(df_grad)
        if results:
            with open(out_file, 'w') as f:
                json.dump(results, f, indent=2, default=safe_json_default)
            print_status(f"Saved to {out_file}", "INFO")
        else:
            status = {"status": "skipped", "reason": "correlation analysis returned no results"}
            with open(out_file, 'w') as f:
                json.dump(status, f, indent=2)
    else:
        print_status("No valid gradient data derived.", "WARN")
        status = {"status": "skipped", "reason": "no CIRC_CONV aperture data in JADES catalog"}
        with open(out_file, 'w') as f:
            json.dump(status, f, indent=2)

if __name__ == "__main__":
    main()
