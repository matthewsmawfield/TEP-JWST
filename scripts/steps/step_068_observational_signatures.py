#!/usr/bin/env python3
"""
TEP-JWST Step 68: Observational Signatures (Raw Photometry)

Purpose:
    To address concerns about SED fitting circularity, this step analyzes
    trends in *raw observational space* (Color-Magnitude Diagrams).
    If TEP effects are real, they must manifest as model-independent 
    correlations in the photometry.

Methodology:
    1. Load JADES PSF-matched photometry (CIRC_CONV).
    2. Construct observed Colors (F115W - F444W) and Magnitudes (F444W).
    3. Select high-z candidates using z_phot from physical catalog (to isolate epoch).
    4. Test correlations:
       - Magnitude vs Color (Mass proxy vs Age/Dust proxy)
       - Surface Brightness vs Color (Potential depth proxy vs Age/Dust proxy)
    
    TEP Prediction:
       - Deeper potentials (Brighter/More Compact) -> Higher Gamma_t -> Older/Dustier -> Redder Color.
       - Correlation should be stronger than standard "Downsizing" alone would predict at z > 8.

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging (severity levels: DEBUG/INFO/WARNING/ERROR/SUCCESS)
from scripts.utils.p_value_utils import format_p_value  # Safe p-value formatting (prevents floating-point underflow at p < 1e-300)

STEP_NUM = "068"  # Pipeline step number (sequential 001-176)
STEP_NAME = "observational_signatures"  # Observational signatures: model-independent raw photometry analysis (Color-Magnitude Diagrams) avoiding SED fitting circularity
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"  # JSON output directory (machine-readable statistical results)
DATA_RAW = PROJECT_ROOT / "data" / "raw"  # Raw catalogue directory (FITS files from MAST/ESA)
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"  # Processed intermediate products (CSV format for step-to-step data flow)


def _native_array(arr):
    out = np.array(arr)
    if out.dtype.byteorder == '>':
        out = out.byteswap().view(out.dtype.newbyteorder('='))
    return out

def load_jades_data():
    """Load JADES photometry and physical catalog."""
    phot_files = [
        DATA_RAW / "JADES_z_gt_8_Candidates_Hainline_et_al.fits",
        DATA_RAW / "hlsp_jades_jwst_nircam_goods-s-deep_photometry_v2.0_catalog.fits",
    ]

    df_phot = None
    for phot_file in phot_files:
        if not phot_file.exists():
            continue

        print_status(f"Loading photometry: {phot_file.name}", "INFO")
        with fits.open(phot_file, memmap=False) as hdul:
            ext_name = 'CIRC_CONV' if 'CIRC_CONV' in hdul else ('CIRC' if 'CIRC' in hdul else None)
            if ext_name is None:
                continue

            data = hdul[ext_name].data
            cols = data.columns.names
            needed = ['ID', 'RA', 'DEC', 'F115W_CIRC4', 'F444W_CIRC4', 'F115W_CIRC1', 'F444W_CIRC1']

            if not all(c in cols for c in needed):
                continue

            df_phot = pd.DataFrame({c: _native_array(data[c]) for c in needed})
            df_phot['ID'] = pd.to_numeric(df_phot['ID'], errors='coerce')
            df_phot = df_phot.dropna(subset=['ID'])
            break

    if df_phot is None:
        print_status("No usable JADES circular-aperture photometry found.", "ERROR")
        return None, None

    # 2. Load Physical Catalog (for redshift selection)
    phys_file = DATA_INTERIM / "jades_highz_physical.csv"
    if not phys_file.exists():
        print_status(f"Physical catalog not found: {phys_file}", "ERROR")
        return None, None

    df_phys = pd.read_csv(phys_file)
    df_phys['ID'] = pd.to_numeric(df_phys['ID'], errors='coerce')
    df_phys = df_phys.dropna(subset=['ID'])

    return df_phot, df_phys

def match_and_analyze(df_phot, df_phys):
    """Match catalogs and analyze raw photometric trends."""

    print_status("Matching catalogs by JADES ID...", "INFO")

    matched = df_phot.merge(
        df_phys[['ID', 'z_best', 'log_Mstar']],
        on='ID',
        how='inner',
    )

    print_status(f"Matched {len(matched)} sources.", "INFO")
    
    # Filter for High-z
    # z > 5.5 to ensure F115W is UV and F444W is Optical (Balmer break enters F444W at z~10? No)
    # F444W is 4.4um. At z=8, 4.4um / 9 = 0.48um (Rest-frame V-band/optical)
    # F115W is 1.15um. At z=8, 1.15um / 9 = 0.12um (Rest-frame UV)
    # So F115W - F444W is UV-Optical color.
    
    highz = matched[matched['z_best'] > 7.0].copy()
    print_status(f"Analyzing z > 7 sample: N = {len(highz)}", "INFO")
    
    # Calculate Observables
    # 1. Magnitude (AB) = -2.5 * log10(Flux_nJy) + 31.4
    # But we work in relative space for correlations.
    # Fluxes are in nJy (usually in JADES cats).
    
    # Filter valid fluxes
    valid_mask = (highz['F115W_CIRC4'] > 0) & (highz['F444W_CIRC4'] > 0)
    data = highz[valid_mask].copy()
    
    # Magnitudes (F444W) - Proxy for Stellar Mass
    data['mag_F444W'] = -2.5 * np.log10(data['F444W_CIRC4']) + 31.4
    
    # Color (F115W - F444W) - Proxy for Age/Dust
    # Color = -2.5 log(F115) - (-2.5 log(F444)) = -2.5 * log(F115/F444)
    data['color_UV_Opt'] = -2.5 * np.log10(data['F115W_CIRC4'] / data['F444W_CIRC4'])
    
    # Compactness Proxy (Surface Brightness / Concentration)
    # Ratio of Inner (CIRC1) to Outer (CIRC4) flux in F444W (Mass map)
    # Higher Ratio = More Compact = Deeper Potential at fixed mass
    data['compactness'] = data['F444W_CIRC1'] / data['F444W_CIRC4']
    
    # --- ANALYSIS ---
    
    results = {}
    
    # 1. Magnitude vs Color
    # TEP: Brighter (Massive) -> Redder (Older)
    # Correlation should be Negative (Mag decreases as Flux increases) vs Positive (Redder is higher Color)?
    # Brighter: Mag is LOWER. Redder: Color is HIGHER.
    # So we expect NEGATIVE correlation between Mag and Color.
    # Or POSITIVE correlation between Flux and Color.
    
    rho_mag, p_mag = stats.spearmanr(data['mag_F444W'], data['color_UV_Opt'])
    print_status(f"Correlation: Mag(F444W) vs Color(UV-Opt)", "INFO")
    print_status(f"rho = {rho_mag:.3f}, p = {p_mag:.2e}", "INFO")
    # Expected: Negative (Bright=Low Mag -> Red=High Color)
    
    results['mag_color'] = {'rho': float(rho_mag), 'p': format_p_value(p_mag), 'n': len(data)}
    
    # 2. Compactness vs Color (at fixed Magnitude?)
    # Partial correlation
    # TEP: More Compact -> Deeper Potential -> Redder
    
    # Calculate residuals of Color vs Mag to remove mass trend
    slope, intercept, _, _, _ = stats.linregress(data['mag_F444W'], data['color_UV_Opt'])
    data['color_resid'] = data['color_UV_Opt'] - (slope * data['mag_F444W'] + intercept)
    
    rho_comp, p_comp = stats.spearmanr(data['compactness'], data['color_resid'])
    print_status(f"Correlation: Compactness vs Color Residuals (fixed Mag)", "INFO")
    print_status(f"rho = {rho_comp:.3f}, p = {p_comp:.2e}", "INFO")
    # Expected: Positive (More compact -> Redder)
    
    results['compactness_color_resid'] = {'rho': float(rho_comp), 'p': format_p_value(p_comp), 'n': len(data)}
    
    # 3. Z > 8 Subset check
    z8 = data[data['z_best'] > 8.0]
    if len(z8) > 10:
        rho_z8, p_z8 = stats.spearmanr(z8['mag_F444W'], z8['color_UV_Opt'])
        print_status(f"z > 8 Subset (N={len(z8)}): Mag vs Color", "INFO")
        print_status(f"rho = {rho_z8:.3f}, p = {p_z8:.2e}", "INFO")
        results['z8_mag_color'] = {'rho': float(rho_z8), 'p': format_p_value(p_z8), 'n': len(z8)}
        
    return results

def main():
    logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=PROJECT_ROOT / "logs" / f"step_{STEP_NUM}_{STEP_NAME}.log")
    set_step_logger(logger)
    
    print_status("Starting Observational Signatures Analysis...", "INFO")
    
    df_phot, df_phys = load_jades_data()
    
    if df_phot is not None and df_phys is not None:
        results = match_and_analyze(df_phot, df_phys)
        
        with open(OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json", "w") as f:
            json.dump(results, f, indent=2)
            
        print_status(f"Results saved to {OUTPUT_PATH / f'step_{STEP_NUM}_{STEP_NAME}.json'}", "SUCCESS")
    else:
        print_status("Data loading failed.", "ERROR")
        status = {"status": "skipped", "reason": "JADES photometry/physical catalog not available"}
        with open(OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json", "w") as _f:
            json.dump(status, _f, indent=2)

if __name__ == "__main__":
    main()
