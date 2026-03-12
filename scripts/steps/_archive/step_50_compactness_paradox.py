#!/usr/bin/env python3
"""
TEP-JWST Step 50: Compactness Paradox Test

This step tests a unique TEP prediction: high-Gamma_t galaxies should appear
MORE COMPACT at fixed apparent stellar mass.

Physical Rationale:
- TEP inflates the apparent stellar mass (M* overestimated)
- The true mass is lower than inferred
- At fixed apparent mass, high-Gamma_t galaxies have lower TRUE mass
- Lower true mass → smaller expected size
- Result: High-Gamma_t galaxies appear "too compact" for their apparent mass

This explains the "impossibly compact" galaxies at high-z without invoking
exotic physics. The compactness is an artifact of mass overestimation.

Prediction:
- At fixed apparent M*, high-Gamma_t galaxies should have SMALLER sizes
- The size residual (observed - expected) should correlate NEGATIVELY with Gamma_t
"""

import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, linregress
from astropy.io import fits
from pathlib import Path
import json

# =============================================================================
# PATHS AND LOGGER
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status

STEP_NUM = "50"
STEP_NAME = "compactness_paradox"

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "uncover"
INTERIM_PATH = PROJECT_ROOT / "results" / "interim"
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"
LOGS_PATH = PROJECT_ROOT / "logs"

for p in [OUTPUT_PATH, LOGS_PATH]:
    p.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")
set_step_logger(logger)

# =============================================================================
# TEP PARAMETERS
# =============================================================================

ALPHA_0 = 0.58
LOG_MH_REF = 12.0
Z_REF = 5.5

def compute_gamma_t(log_Mh, z):
    alpha_z = ALPHA_0 * np.sqrt(1 + z)
    log_mh_ref_z = LOG_MH_REF - 1.5 * np.log10(1 + z)
    delta_log_Mh = log_Mh - log_mh_ref_z
    z_factor = (1 + z) / (1 + Z_REF)
    argument = alpha_z * (2/3) * delta_log_Mh * z_factor
    return np.exp(argument)

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    print_status("=" * 70, "INFO")
    print_status(f"STEP {STEP_NUM}: Compactness Paradox Test", "INFO")
    print_status("=" * 70, "INFO")
    print_status("", "INFO")
    print_status("TEP Prediction: High-Γ_t galaxies appear TOO COMPACT", "INFO")
    print_status("Reason: Their apparent mass is inflated, making them seem small", "INFO")
    
    # Load UNCOVER catalog with size data
    try:
        hdu = fits.open(DATA_PATH / "UNCOVER_DR4_SPS_catalog.fits")
        data = hdu[1].data
    except FileNotFoundError:
        print_status("UNCOVER catalog not found", "ERROR")
        return
    
    # Extract columns
    z = np.array(data['z_50'])
    mstar = np.array(data['mstar_50'])
    
    # Try different size columns
    size_col = None
    for col in ['re_kron', 'r_kron', 'kron_radius', 'flux_radius', 'a_image']:
        if col in data.names:
            size_col = col
            break
    
    if size_col is None:
        # Use flux_aper columns as proxy
        print_status("No direct size column found, checking for aperture data...", "INFO")
        # Check for any size-related columns
        size_cols = [c for c in data.names if 'radius' in c.lower() or 'size' in c.lower() or 'kron' in c.lower()]
        print_status(f"Available size-related columns: {size_cols}", "INFO")
        
        if len(size_cols) == 0:
            print_status("No size data available in catalog", "WARNING")
            # Fall back to using chi2 as a proxy for SED mismatch
            print_status("Using chi2 as alternative diagnostic...", "INFO")
            return analyze_chi2_compactness(data, z, mstar)
    
    size = np.array(data[size_col])
    print_status(f"Using size column: {size_col}", "INFO")
    
    # Quality cuts
    valid = (z > 4) & (z < 10) & (mstar > 8) & (size > 0) & ~np.isnan(size)
    
    z = z[valid]
    mstar = mstar[valid]
    size = size[valid]
    
    # Compute Gamma_t
    log_Mh = mstar + 2.0
    gamma_t = compute_gamma_t(log_Mh, z)
    
    print_status(f"\nSample size: N = {len(z)}", "INFO")
    
    results = {
        'n_total': int(len(z)),
        'size_column': size_col,
        'tests': {}
    }
    
    # ==========================================================================
    # TEST 1: Size-Mass Relation
    # ==========================================================================
    print_status("\n" + "=" * 50, "INFO")
    print_status("TEST 1: Size-Mass Relation", "INFO")
    print_status("=" * 50, "INFO")
    
    # Fit size-mass relation
    log_size = np.log10(size)
    slope, intercept, r, p, se = linregress(mstar, log_size)
    
    print_status(f"\nSize-Mass Relation: log(r) = {slope:.3f} × log(M*) + {intercept:.3f}", "INFO")
    print_status(f"  R² = {r**2:.3f}", "INFO")
    
    # Calculate size residuals
    log_size_expected = slope * mstar + intercept
    size_residual = log_size - log_size_expected
    
    results['tests']['size_mass_relation'] = {
        'slope': float(slope),
        'intercept': float(intercept),
        'r_squared': float(r**2)
    }
    
    # ==========================================================================
    # TEST 2: Size Residual vs Gamma_t
    # ==========================================================================
    print_status("\n" + "=" * 50, "INFO")
    print_status("TEST 2: Size Residual vs Γ_t", "INFO")
    print_status("=" * 50, "INFO")
    
    rho, p_val = spearmanr(gamma_t, size_residual)
    
    print_status(f"\nρ(Γ_t, size_residual) = {rho:.3f} (p = {p_val:.2e})", "INFO")
    
    # TEP predicts NEGATIVE correlation
    if rho < 0 and p_val < 0.05:
        print_status("✓ COMPACTNESS PARADOX CONFIRMED", "INFO")
        print_status("  High-Γ_t galaxies are smaller than expected at fixed mass", "INFO")
        compactness_confirmed = True
    elif rho < 0:
        print_status("⚠ Negative trend but not significant", "INFO")
        compactness_confirmed = False
    else:
        print_status("✗ No compactness paradox detected", "INFO")
        compactness_confirmed = False
    
    results['tests']['size_residual'] = {
        'rho': float(rho),
        'p': float(p_val),
        'compactness_confirmed': compactness_confirmed
    }
    
    # ==========================================================================
    # TEST 3: Regime Comparison
    # ==========================================================================
    print_status("\n" + "=" * 50, "INFO")
    print_status("TEST 3: Size by Γ_t Regime", "INFO")
    print_status("=" * 50, "INFO")
    
    enhanced = gamma_t > 1
    suppressed = gamma_t < 0.5
    
    if enhanced.sum() > 10 and suppressed.sum() > 10:
        mean_resid_enh = size_residual[enhanced].mean()
        mean_resid_sup = size_residual[suppressed].mean()
        
        print_status(f"\nMean size residual:", "INFO")
        print_status(f"  Enhanced (Γ_t > 1): {mean_resid_enh:.3f} dex", "INFO")
        print_status(f"  Suppressed (Γ_t < 0.5): {mean_resid_sup:.3f} dex", "INFO")
        print_status(f"  Difference: {mean_resid_enh - mean_resid_sup:.3f} dex", "INFO")
        
        # Convert to linear factor
        linear_factor = 10**(mean_resid_enh - mean_resid_sup)
        print_status(f"  Enhanced are {linear_factor:.2f}× the size of suppressed at fixed mass", "INFO")
        
        results['tests']['regime_comparison'] = {
            'n_enhanced': int(enhanced.sum()),
            'n_suppressed': int(suppressed.sum()),
            'mean_resid_enhanced': float(mean_resid_enh),
            'mean_resid_suppressed': float(mean_resid_sup),
            'difference': float(mean_resid_enh - mean_resid_sup),
            'linear_factor': float(linear_factor)
        }
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("SUMMARY", "INFO")
    print_status("=" * 70, "INFO")
    
    if compactness_confirmed:
        print_status("\n✓ COMPACTNESS PARADOX CONFIRMED", "INFO")
        print_status("  High-Γ_t galaxies appear too compact for their apparent mass.", "INFO")
        print_status("  TEP explanation: Their true mass is lower (M* is inflated).", "INFO")
        overall = "CONFIRMED"
    else:
        print_status("\n⚠ Compactness paradox not clearly detected", "INFO")
        overall = "NOT_CONFIRMED"
    
    results['summary'] = {'overall': overall}
    
    # Save
    json_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print_status(f"\nSaved to {json_path}", "INFO")


def analyze_chi2_compactness(data, z, mstar):
    """
    Alternative: Use chi2 as proxy for SED mismatch.
    High-Gamma_t galaxies should have worse fits AND appear compact.
    """
    print_status("\n" + "=" * 50, "INFO")
    print_status("ALTERNATIVE: χ² Analysis", "INFO")
    print_status("=" * 50, "INFO")
    
    chi2 = np.array(data['chi2'])
    
    valid = (z > 4) & (z < 10) & (mstar > 8) & ~np.isnan(chi2)
    z = z[valid]
    mstar = mstar[valid]
    chi2 = chi2[valid]
    
    log_Mh = mstar + 2.0
    gamma_t = compute_gamma_t(log_Mh, z)
    
    print_status(f"Sample: N = {len(z)}", "INFO")
    
    # Correlation
    rho, p = spearmanr(gamma_t, chi2)
    print_status(f"\nρ(Γ_t, χ²) = {rho:.3f} (p = {p:.2e})", "INFO")
    
    if rho > 0 and p < 0.001:
        print_status("✓ High-Γ_t galaxies have worse SED fits", "INFO")
        print_status("  This supports TEP: isochrony assumption fails", "INFO")
    
    results = {
        'test': 'chi2_alternative',
        'n': int(len(z)),
        'rho': float(rho),
        'p': float(p)
    }
    
    json_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
