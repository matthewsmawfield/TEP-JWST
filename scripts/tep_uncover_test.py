#!/usr/bin/env python3
"""
TEP-JWST: Test Predictions Against UNCOVER DR4 Data

Key predictions to test:
1. Mass-SFE correlation: More massive → higher apparent SFE
2. Redshift dependence: Higher z → stronger anomaly
3. Screening: Most massive systems may show reduced effect

This uses the actual UNCOVER DR4 catalog.
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
    
    # Try main catalog first
    catalog_file = data_path / "UNCOVER_DR4_SPS_catalog.fits"
    
    if not catalog_file.exists():
        print(f"Catalog not found: {catalog_file}")
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
# ANALYSIS
# =============================================================================

def analyze_mass_sfe_correlation(data):
    """
    Test TEP prediction: Mass-SFE correlation.
    
    Under TEP, more massive galaxies should show higher apparent SFE
    because they experience more chronological enhancement.
    """
    print("=" * 70)
    print("TEST 1: Mass-SFE Correlation")
    print("=" * 70)
    print()
    
    # Extract relevant columns
    z = data['z_50']
    log_Mstar = data['mstar_50']
    sfr = data['sfr100_50']  # SFR over 100 Myr
    
    # Filter for high-z galaxies with good data
    mask = (z > 4) & (z < 10) & (log_Mstar > 8) & (sfr > 0) & np.isfinite(log_Mstar) & np.isfinite(sfr)
    
    z_filt = z[mask]
    log_Mstar_filt = log_Mstar[mask]
    sfr_filt = sfr[mask]
    
    print(f"Sample: {np.sum(mask)} galaxies at 4 < z < 10")
    print()
    
    # Calculate sSFR (specific SFR)
    ssfr = sfr_filt / (10**log_Mstar_filt)
    log_ssfr = np.log10(ssfr)
    
    # Estimate halo mass
    log_Mh = halo_mass_from_stellar(log_Mstar_filt)
    
    # Calculate TEP prediction
    gamma_t_pred = np.array([tep_gamma(m, zz) for m, zz in zip(log_Mh, z_filt)])
    
    # Test correlations
    # 1. Mass vs sSFR (standard physics predicts negative, TEP predicts less negative or positive)
    rho_mass_ssfr, p_mass_ssfr = spearmanr(log_Mstar_filt, log_ssfr)
    
    # 2. Mass vs SFR (should be positive, but TEP predicts steeper)
    rho_mass_sfr, p_mass_sfr = spearmanr(log_Mstar_filt, np.log10(sfr_filt))
    
    # 3. Predicted Γ_t vs observed sSFR (TEP predicts negative: higher Γ_t → lower apparent sSFR)
    rho_gamma_ssfr, p_gamma_ssfr = spearmanr(gamma_t_pred, log_ssfr)
    
    print("Correlation Results:")
    print("-" * 50)
    print(f"log M* vs log sSFR:  ρ = {rho_mass_ssfr:+.3f} (p = {p_mass_ssfr:.2e})")
    print(f"log M* vs log SFR:   ρ = {rho_mass_sfr:+.3f} (p = {p_mass_sfr:.2e})")
    print(f"Γ_t vs log sSFR:     ρ = {rho_gamma_ssfr:+.3f} (p = {p_gamma_ssfr:.2e})")
    print()
    
    # Interpretation
    print("INTERPRETATION:")
    print("-" * 50)
    
    if rho_mass_ssfr < -0.3:
        print("Strong negative mass-sSFR correlation (standard downsizing).")
        print("This is expected even under TEP, but TEP predicts it should be")
        print("WEAKER than standard physics due to mass-dependent bias.")
    elif rho_mass_ssfr > -0.1:
        print("Weak or absent mass-sSFR correlation!")
        print("This is CONSISTENT with TEP: the isochrony bias partially")
        print("cancels the intrinsic downsizing trend.")
    
    print()
    
    # Bin by mass and compute mean sSFR
    mass_bins = [8, 9, 9.5, 10, 10.5, 11, 12]
    print("Mean log sSFR by Mass Bin:")
    print("-" * 50)
    print(f"{'Mass Bin':<15} {'N':<8} {'<log sSFR>':<12} {'<Γ_t>':<10}")
    print("-" * 50)
    
    for i in range(len(mass_bins) - 1):
        bin_mask = (log_Mstar_filt >= mass_bins[i]) & (log_Mstar_filt < mass_bins[i+1])
        if np.sum(bin_mask) > 0:
            mean_ssfr = np.mean(log_ssfr[bin_mask])
            mean_gamma = np.mean(gamma_t_pred[bin_mask])
            n = np.sum(bin_mask)
            print(f"{mass_bins[i]:.1f}-{mass_bins[i+1]:.1f}       {n:<8} {mean_ssfr:<12.2f} {mean_gamma:<10.2f}")
    
    return {
        "n_galaxies": int(np.sum(mask)),
        "rho_mass_ssfr": rho_mass_ssfr,
        "p_mass_ssfr": p_mass_ssfr,
        "rho_mass_sfr": rho_mass_sfr,
        "rho_gamma_ssfr": rho_gamma_ssfr
    }

def analyze_redshift_dependence(data):
    """
    Test TEP prediction: Redshift dependence.
    
    Under TEP, the mass-sSFR relation should change with redshift:
    at higher z, the TEP effect is stronger.
    """
    print()
    print("=" * 70)
    print("TEST 2: Redshift Dependence")
    print("=" * 70)
    print()
    
    z = data['z_50']
    log_Mstar = data['mstar_50']
    sfr = data['sfr100_50']
    
    # Filter
    mask = (log_Mstar > 9) & (sfr > 0) & np.isfinite(log_Mstar) & np.isfinite(sfr) & np.isfinite(z)
    
    z_filt = z[mask]
    log_Mstar_filt = log_Mstar[mask]
    sfr_filt = sfr[mask]
    ssfr = sfr_filt / (10**log_Mstar_filt)
    log_ssfr = np.log10(ssfr)
    
    # Bin by redshift
    z_bins = [(4, 5), (5, 6), (6, 7), (7, 8), (8, 10)]
    
    print("Mass-sSFR Correlation by Redshift Bin:")
    print("-" * 60)
    print(f"{'z Bin':<12} {'N':<8} {'ρ(M*, sSFR)':<15} {'<Γ_t> (M*=10.5)':<15}")
    print("-" * 60)
    
    results = []
    for z_lo, z_hi in z_bins:
        bin_mask = (z_filt >= z_lo) & (z_filt < z_hi)
        n = np.sum(bin_mask)
        
        if n > 10:
            rho, p = spearmanr(log_Mstar_filt[bin_mask], log_ssfr[bin_mask])
            
            # Predicted Γ_t for a typical massive galaxy in this bin
            z_mid = (z_lo + z_hi) / 2
            gamma_pred = tep_gamma(12.5, z_mid)  # log M_h = 12.5
            
            print(f"{z_lo:.0f}-{z_hi:.0f}         {n:<8} {rho:+.3f}           {gamma_pred:.2f}")
            results.append({"z_bin": f"{z_lo}-{z_hi}", "n": n, "rho": rho, "gamma_pred": gamma_pred})
        else:
            print(f"{z_lo:.0f}-{z_hi:.0f}         {n:<8} (insufficient data)")
    
    print()
    print("TEP PREDICTION:")
    print("The correlation should become WEAKER (less negative) at higher z")
    print("because the TEP bias is stronger, partially canceling downsizing.")
    
    return results

def analyze_mass_weighted_age(data):
    """
    Test TEP prediction: Mass-weighted ages.
    
    Under TEP, more massive galaxies should have OLDER apparent ages
    because they experience more chronological enhancement.
    """
    print()
    print("=" * 70)
    print("TEST 3: Mass-Weighted Age Correlation")
    print("=" * 70)
    print()
    
    z = data['z_50']
    log_Mstar = data['mstar_50']
    mwa = data['mwa_50']  # Mass-weighted age in Gyr
    
    # Filter for high-z with valid ages
    mask = (z > 4) & (z < 10) & (log_Mstar > 8) & (mwa > 0) & np.isfinite(mwa)
    
    z_filt = z[mask]
    log_Mstar_filt = log_Mstar[mask]
    mwa_filt = mwa[mask]
    
    print(f"Sample: {np.sum(mask)} galaxies with mass-weighted ages")
    print()
    
    # Cosmic age at each redshift
    t_cosmic = np.array([cosmo.age(zz).value for zz in z_filt])
    
    # Age ratio: apparent age / cosmic age
    age_ratio = mwa_filt / t_cosmic
    
    # Predicted Γ_t
    log_Mh = halo_mass_from_stellar(log_Mstar_filt)
    gamma_t_pred = np.array([tep_gamma(m, zz) for m, zz in zip(log_Mh, z_filt)])
    
    # Test correlations
    rho_mass_age, p_mass_age = spearmanr(log_Mstar_filt, mwa_filt)
    rho_mass_ratio, p_mass_ratio = spearmanr(log_Mstar_filt, age_ratio)
    rho_gamma_ratio, p_gamma_ratio = spearmanr(gamma_t_pred, age_ratio)
    
    print("Correlation Results:")
    print("-" * 50)
    print(f"log M* vs MWA:           ρ = {rho_mass_age:+.3f} (p = {p_mass_age:.2e})")
    print(f"log M* vs (MWA/t_cosmic): ρ = {rho_mass_ratio:+.3f} (p = {p_mass_ratio:.2e})")
    print(f"Γ_t vs (MWA/t_cosmic):    ρ = {rho_gamma_ratio:+.3f} (p = {p_gamma_ratio:.2e})")
    print()
    
    # Check for "impossible" ages
    n_impossible = np.sum(mwa_filt > t_cosmic)
    frac_impossible = n_impossible / len(mwa_filt) * 100
    
    print(f"Galaxies with MWA > t_cosmic: {n_impossible} ({frac_impossible:.1f}%)")
    print()
    
    if rho_mass_age > 0:
        print("POSITIVE mass-age correlation detected!")
        print("This is CONSISTENT with TEP: more massive galaxies appear older")
        print("due to chronological enhancement in deeper potentials.")
    else:
        print("Negative or no mass-age correlation.")
        print("This could indicate screening effects in the most massive systems,")
        print("or that the sample is dominated by star-forming galaxies.")
    
    return {
        "n_galaxies": int(np.sum(mask)),
        "rho_mass_age": rho_mass_age,
        "rho_mass_ratio": rho_mass_ratio,
        "rho_gamma_ratio": rho_gamma_ratio,
        "n_impossible": int(n_impossible),
        "frac_impossible": frac_impossible
    }

def main():
    print()
    print("=" * 70)
    print("TEP-JWST: Testing Predictions Against UNCOVER DR4")
    print("=" * 70)
    print()
    
    # Load data
    data = load_uncover_data()
    
    if data is None:
        print("ERROR: Could not load UNCOVER data.")
        return None
    
    print(f"Loaded {len(data)} sources from UNCOVER DR4")
    print()
    
    results = {}
    
    # Test 1: Mass-SFE correlation
    results["mass_sfe"] = analyze_mass_sfe_correlation(data)
    
    # Test 2: Redshift dependence
    results["redshift"] = analyze_redshift_dependence(data)
    
    # Test 3: Mass-weighted ages
    results["ages"] = analyze_mass_weighted_age(data)
    
    # Summary
    print()
    print("=" * 70)
    print("SUMMARY OF TESTS")
    print("=" * 70)
    print()
    
    print("1. Mass-sSFR Correlation:")
    if results["mass_sfe"]["rho_mass_ssfr"] > -0.2:
        print("   → WEAK correlation, CONSISTENT with TEP prediction")
    else:
        print("   → Strong negative correlation (standard downsizing)")
    
    print()
    print("2. Redshift Dependence:")
    print("   → See binned results above")
    
    print()
    print("3. Mass-Age Correlation:")
    if results["ages"]["rho_mass_age"] > 0:
        print("   → POSITIVE correlation, CONSISTENT with TEP prediction")
    else:
        print("   → Negative correlation (may indicate screening)")
    
    print()
    print(f"4. 'Impossible' Ages: {results['ages']['frac_impossible']:.1f}% of sample")
    print("   → These are naturally explained by TEP chronological enhancement")
    
    # Save results
    output_dir = Path(__file__).parent.parent / "results" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "tep_uncover_test.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=float)
    
    print()
    print(f"Results saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    main()
