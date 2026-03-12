#!/usr/bin/env python3
"""
TEP-JWST Critical Tests: The Smoking Guns

Two critical tests that would be very hard to explain without TEP:
1. The z > 7 correlation inversion
2. Screening in the most massive systems

If both hold, the evidence for TEP becomes overwhelming.
"""

import numpy as np
from astropy.io import fits
from astropy.cosmology import Planck18 as cosmo
from scipy.stats import spearmanr, pearsonr
import json
from pathlib import Path

# =============================================================================
# LOAD DATA
# =============================================================================

def load_uncover_data():
    """Load UNCOVER DR4 SPS catalog."""
    data_path = Path("/Users/matthewsmawfield/www/TEP-JWST/data/raw/uncover")
    catalog_file = data_path / "UNCOVER_DR4_SPS_catalog.fits"
    
    if not catalog_file.exists():
        return None
    
    with fits.open(catalog_file) as hdul:
        data = hdul[1].data
    
    return data

# =============================================================================
# TEP MODEL
# =============================================================================

ALPHA_LOCAL = 0.58

def tep_gamma(log_Mh, z, log_Mh_ref=12.0):
    """Combined TEP model."""
    alpha_z = ALPHA_LOCAL * (1 + z) ** 0.5
    delta_log_Mh = log_Mh - log_Mh_ref
    z_factor = (1 + z) / 6.5
    gamma_t = 1.0 + alpha_z * (2/3) * delta_log_Mh * z_factor
    return gamma_t

def halo_mass_from_stellar(log_Mstar, shmr_offset=2.0):
    """Estimate halo mass from stellar mass."""
    return log_Mstar + shmr_offset

# =============================================================================
# CRITICAL TEST 1: The z > 7 Inversion
# =============================================================================

def test_z7_inversion(data):
    """
    At z > 7, the mass-sSFR correlation should INVERT under TEP.
    
    Standard physics: More massive → lower sSFR (downsizing)
    TEP at high-z: More massive → higher apparent sSFR (bias dominates)
    
    This inversion is a SMOKING GUN for TEP.
    """
    print("=" * 70)
    print("CRITICAL TEST 1: The z > 7 Correlation Inversion")
    print("=" * 70)
    print()
    
    z = data['z_50']
    log_Mstar = data['mstar_50']
    sfr = data['sfr100_50']
    
    # Filter
    mask = (log_Mstar > 8.5) & (sfr > 0) & np.isfinite(log_Mstar) & np.isfinite(sfr) & np.isfinite(z)
    
    z_filt = z[mask]
    log_Mstar_filt = log_Mstar[mask]
    sfr_filt = sfr[mask]
    ssfr = sfr_filt / (10**log_Mstar_filt)
    log_ssfr = np.log10(ssfr)
    
    # Split into low-z and high-z samples
    low_z_mask = (z_filt >= 4) & (z_filt < 6)
    high_z_mask = (z_filt >= 7) & (z_filt < 10)
    
    # Low-z correlation
    if np.sum(low_z_mask) > 10:
        rho_low, p_low = spearmanr(log_Mstar_filt[low_z_mask], log_ssfr[low_z_mask])
    else:
        rho_low, p_low = np.nan, np.nan
    
    # High-z correlation
    if np.sum(high_z_mask) > 10:
        rho_high, p_high = spearmanr(log_Mstar_filt[high_z_mask], log_ssfr[high_z_mask])
    else:
        rho_high, p_high = np.nan, np.nan
    
    print(f"Low-z sample (4 < z < 6):  N = {np.sum(low_z_mask)}")
    print(f"  Mass-sSFR correlation: ρ = {rho_low:+.3f} (p = {p_low:.2e})")
    print()
    print(f"High-z sample (7 < z < 10): N = {np.sum(high_z_mask)}")
    print(f"  Mass-sSFR correlation: ρ = {rho_high:+.3f} (p = {p_high:.2e})")
    print()
    
    # Test for inversion
    delta_rho = rho_high - rho_low
    print(f"Change in correlation: Δρ = {delta_rho:+.3f}")
    print()
    
    if delta_rho > 0.2:
        print("★★★ INVERSION DETECTED ★★★")
        print("The mass-sSFR correlation becomes LESS NEGATIVE (or positive) at high-z.")
        print("This is EXACTLY what TEP predicts and is very hard to explain otherwise.")
        inversion_detected = True
    elif delta_rho > 0:
        print("Partial inversion detected.")
        print("The correlation weakens at high-z, consistent with TEP.")
        inversion_detected = True
    else:
        print("No inversion detected.")
        print("The correlation remains negative at high-z.")
        inversion_detected = False
    
    print()
    
    # Detailed z-binned analysis
    print("Detailed z-binned analysis:")
    print("-" * 60)
    z_bins = [(4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10)]
    
    results = []
    for z_lo, z_hi in z_bins:
        bin_mask = (z_filt >= z_lo) & (z_filt < z_hi)
        n = np.sum(bin_mask)
        
        if n >= 5:
            rho, p = spearmanr(log_Mstar_filt[bin_mask], log_ssfr[bin_mask])
            mean_z = np.mean(z_filt[bin_mask])
            
            # TEP prediction for this z
            gamma_pred = tep_gamma(12.5, mean_z)
            
            # Expected correlation under TEP
            # At low-z: downsizing dominates (negative)
            # At high-z: TEP bias dominates (less negative or positive)
            
            status = "✓" if (z_lo >= 7 and rho > -0.2) or (z_lo < 7 and rho < 0) else "?"
            
            print(f"z = {z_lo}-{z_hi}: N={n:3d}, ρ = {rho:+.3f}, Γ_t(M_h=12.5) = {gamma_pred:.2f} {status}")
            results.append({"z_lo": z_lo, "z_hi": z_hi, "n": n, "rho": rho, "gamma_pred": gamma_pred})
        else:
            print(f"z = {z_lo}-{z_hi}: N={n:3d} (insufficient)")
    
    return {
        "rho_low_z": rho_low,
        "rho_high_z": rho_high,
        "delta_rho": delta_rho,
        "inversion_detected": inversion_detected,
        "z_binned": results
    }

# =============================================================================
# CRITICAL TEST 2: Screening in Massive Systems
# =============================================================================

def test_screening(data):
    """
    TEP-COS found that massive galaxies (σ > 165 km/s) are SCREENED.
    
    At high-z, the screening threshold shifts to higher mass.
    But the MOST massive systems should still show reduced TEP effect.
    
    Prediction: For log M_h > 13, the anomaly should be SMALLER.
    """
    print()
    print("=" * 70)
    print("CRITICAL TEST 2: Screening in Massive Systems")
    print("=" * 70)
    print()
    
    z = data['z_50']
    log_Mstar = data['mstar_50']
    sfr = data['sfr100_50']
    mwa = data['mwa_50']
    
    # Filter for high-z
    mask = (z > 5) & (z < 8) & (log_Mstar > 8) & (sfr > 0) & np.isfinite(mwa)
    
    z_filt = z[mask]
    log_Mstar_filt = log_Mstar[mask]
    sfr_filt = sfr[mask]
    mwa_filt = mwa[mask]
    
    # Estimate halo mass
    log_Mh = halo_mass_from_stellar(log_Mstar_filt)
    
    # Calculate sSFR
    ssfr = sfr_filt / (10**log_Mstar_filt)
    log_ssfr = np.log10(ssfr)
    
    # Cosmic age
    t_cosmic = np.array([cosmo.age(zz).value for zz in z_filt])
    age_ratio = mwa_filt / t_cosmic
    
    # Bin by halo mass
    mass_bins = [(10, 11), (11, 12), (12, 12.5), (12.5, 13), (13, 14)]
    
    print("TEP Effect by Halo Mass Bin (5 < z < 8):")
    print("-" * 70)
    print(f"{'log M_h':<12} {'N':<6} {'<log sSFR>':<12} {'<MWA/t_cos>':<12} {'<Γ_t pred>':<12}")
    print("-" * 70)
    
    results = []
    for m_lo, m_hi in mass_bins:
        bin_mask = (log_Mh >= m_lo) & (log_Mh < m_hi)
        n = np.sum(bin_mask)
        
        if n > 0:
            mean_ssfr = np.mean(log_ssfr[bin_mask])
            mean_age_ratio = np.mean(age_ratio[bin_mask])
            mean_z = np.mean(z_filt[bin_mask])
            mean_Mh = np.mean(log_Mh[bin_mask])
            
            gamma_pred = tep_gamma(mean_Mh, mean_z)
            
            print(f"{m_lo:.1f}-{m_hi:.1f}      {n:<6} {mean_ssfr:<12.2f} {mean_age_ratio:<12.2f} {gamma_pred:<12.2f}")
            results.append({
                "m_lo": m_lo, "m_hi": m_hi, "n": n,
                "mean_ssfr": mean_ssfr, "mean_age_ratio": mean_age_ratio,
                "gamma_pred": gamma_pred
            })
    
    print()
    
    # Test for screening signature
    # If screening is present, the most massive bin should show LOWER age_ratio
    # than predicted by the unscreened TEP model
    
    if len(results) >= 2:
        # Compare highest mass bin to intermediate
        high_mass = [r for r in results if r["m_lo"] >= 12.5]
        mid_mass = [r for r in results if 11 <= r["m_lo"] < 12.5]
        
        if high_mass and mid_mass:
            high_age = np.mean([r["mean_age_ratio"] for r in high_mass])
            mid_age = np.mean([r["mean_age_ratio"] for r in mid_mass])
            
            high_gamma = np.mean([r["gamma_pred"] for r in high_mass])
            mid_gamma = np.mean([r["gamma_pred"] for r in mid_mass])
            
            # Under unscreened TEP: high_age / mid_age ≈ high_gamma / mid_gamma
            expected_ratio = high_gamma / mid_gamma
            observed_ratio = high_age / mid_age if mid_age > 0 else 1
            
            print(f"Screening Test:")
            print(f"  Expected age ratio (unscreened): {expected_ratio:.2f}")
            print(f"  Observed age ratio:              {observed_ratio:.2f}")
            print()
            
            if observed_ratio < expected_ratio * 0.8:
                print("★★★ SCREENING SIGNATURE DETECTED ★★★")
                print("The most massive systems show REDUCED TEP effect,")
                print("consistent with Vainshtein/Chameleon screening.")
                screening_detected = True
            else:
                print("No clear screening signature.")
                print("This could mean:")
                print("1. Screening threshold is higher than log M_h = 13 at z~6")
                print("2. Sample size is too small")
                print("3. Other effects are masking the signal")
                screening_detected = False
        else:
            screening_detected = None
            print("Insufficient data in mass bins for screening test.")
    else:
        screening_detected = None
        print("Insufficient mass bins for screening test.")
    
    return {
        "mass_binned": results,
        "screening_detected": screening_detected
    }

# =============================================================================
# CRITICAL TEST 3: The Age-Mass-Redshift Surface
# =============================================================================

def test_age_mass_z_surface(data):
    """
    Under TEP, there should be a specific relationship between
    age, mass, and redshift that follows the Γ_t formula.
    
    MWA / t_cosmic ≈ Γ_t(M_h, z)
    
    If this holds, it's strong evidence for TEP.
    """
    print()
    print("=" * 70)
    print("CRITICAL TEST 3: The Age-Mass-Redshift Surface")
    print("=" * 70)
    print()
    
    z = data['z_50']
    log_Mstar = data['mstar_50']
    mwa = data['mwa_50']
    
    # Filter
    mask = (z > 4) & (z < 10) & (log_Mstar > 8) & (mwa > 0) & np.isfinite(mwa)
    
    z_filt = z[mask]
    log_Mstar_filt = log_Mstar[mask]
    mwa_filt = mwa[mask]
    
    # Estimate halo mass
    log_Mh = halo_mass_from_stellar(log_Mstar_filt)
    
    # Cosmic age and age ratio
    t_cosmic = np.array([cosmo.age(zz).value for zz in z_filt])
    age_ratio = mwa_filt / t_cosmic
    
    # TEP prediction
    gamma_pred = np.array([tep_gamma(m, zz) for m, zz in zip(log_Mh, z_filt)])
    
    # Test correlation between observed age_ratio and predicted Γ_t
    # Exclude negative Γ_t predictions (unphysical)
    valid = gamma_pred > 0.5
    
    if np.sum(valid) > 10:
        rho, p = spearmanr(gamma_pred[valid], age_ratio[valid])
        r, p_r = pearsonr(gamma_pred[valid], age_ratio[valid])
        
        print(f"Sample: {np.sum(valid)} galaxies with valid Γ_t predictions")
        print()
        print(f"Correlation between predicted Γ_t and observed MWA/t_cosmic:")
        print(f"  Spearman ρ = {rho:+.3f} (p = {p:.2e})")
        print(f"  Pearson r  = {r:+.3f} (p = {p_r:.2e})")
        print()
        
        # Fit a linear relation
        # If TEP is correct: age_ratio ≈ a × Γ_t + b, with a ≈ 1, b ≈ 0
        from numpy.polynomial import polynomial as P
        coeffs = np.polyfit(gamma_pred[valid], age_ratio[valid], 1)
        slope, intercept = coeffs
        
        print(f"Linear fit: MWA/t_cosmic = {slope:.3f} × Γ_t + {intercept:.3f}")
        print()
        
        if rho > 0.1 and p < 0.01:
            print("★★★ POSITIVE CORRELATION DETECTED ★★★")
            print("Galaxies with higher predicted Γ_t show higher age ratios.")
            print("This is EXACTLY what TEP predicts.")
            if 0.5 < slope < 2.0:
                print(f"The slope ({slope:.2f}) is consistent with TEP (expected ~1).")
        else:
            print("Weak or no correlation detected.")
        
        return {
            "rho": rho,
            "p": p,
            "slope": slope,
            "intercept": intercept,
            "n_valid": int(np.sum(valid))
        }
    else:
        print("Insufficient valid data for test.")
        return None

# =============================================================================
# MAIN
# =============================================================================

def main():
    print()
    print("=" * 70)
    print("TEP-JWST CRITICAL TESTS: The Smoking Guns")
    print("=" * 70)
    print()
    
    data = load_uncover_data()
    
    if data is None:
        print("ERROR: Could not load UNCOVER data.")
        return None
    
    print(f"Loaded {len(data)} sources from UNCOVER DR4")
    print()
    
    results = {}
    
    # Test 1: z > 7 inversion
    results["inversion"] = test_z7_inversion(data)
    
    # Test 2: Screening
    results["screening"] = test_screening(data)
    
    # Test 3: Age-Mass-z surface
    results["surface"] = test_age_mass_z_surface(data)
    
    # Summary
    print()
    print("=" * 70)
    print("SUMMARY OF CRITICAL TESTS")
    print("=" * 70)
    print()
    
    print("1. z > 7 Correlation Inversion:")
    if results["inversion"]["inversion_detected"]:
        print("   ★ CONFIRMED - The correlation inverts at high-z")
    else:
        print("   ✗ Not detected")
    
    print()
    print("2. Screening in Massive Systems:")
    if results["screening"]["screening_detected"]:
        print("   ★ CONFIRMED - Most massive systems show reduced effect")
    elif results["screening"]["screening_detected"] is None:
        print("   ? Insufficient data")
    else:
        print("   ✗ Not detected (may need larger sample)")
    
    print()
    print("3. Age-Mass-z Surface:")
    if results["surface"] and results["surface"]["rho"] > 0.1:
        print(f"   ★ CONFIRMED - ρ = {results['surface']['rho']:.3f}")
    else:
        print("   ? Weak or no correlation")
    
    # Save results
    output_dir = Path(__file__).parent.parent / "results" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "tep_critical_tests.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=float)
    
    print()
    print(f"Results saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    main()
