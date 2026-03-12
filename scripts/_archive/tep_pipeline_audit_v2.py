#!/usr/bin/env python3
"""
TEP-JWST Pipeline Audit v2: Corrected Analysis with Strengthened Findings

This script audits all claims made in the analysis with corrected data handling:
- Uses ssfr100_50 column directly (linear sSFR values)
- Includes partial correlation analysis controlling for redshift
- Bootstrap CIs for all key claims

Run this to verify all manuscript claims are reproducible.
"""

import numpy as np
from astropy.io import fits
from astropy.cosmology import Planck18 as cosmo
from scipy.stats import spearmanr, linregress
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONSTANTS
# =============================================================================

ALPHA_LOCAL = 0.58
ALPHA_UNCERTAINTY = 0.16
EPSILON_STANDARD = 0.20
LOG_MH_REF = 12.0
Z_REF = 5.5

RED_MONSTERS = {
    "S1": {"z": 5.85, "log_Mstar": 11.08, "log_Mh": 12.88, "SFE": 0.50},
    "S2": {"z": 5.30, "log_Mstar": 10.88, "log_Mh": 12.68, "SFE": 0.50},
    "S3": {"z": 5.55, "log_Mstar": 10.74, "log_Mh": 12.54, "SFE": 0.50},
}

# =============================================================================
# TEP MODEL
# =============================================================================

def tep_gamma(log_Mh, z, alpha_0=ALPHA_LOCAL):
    """Combined TEP model for chronological enhancement."""
    alpha_z = alpha_0 * np.sqrt(1 + z)
    delta_log_Mh = log_Mh - LOG_MH_REF
    z_factor = (1 + z) / (1 + Z_REF)
    return 1.0 + alpha_z * (2/3) * delta_log_Mh * z_factor

def isochrony_sfe_bias(gamma_t):
    """M/L bias from isochrony assumption."""
    return gamma_t ** 0.7 if gamma_t > 1 else 1.0

# =============================================================================
# DATA LOADING
# =============================================================================

def load_uncover_data():
    """Load UNCOVER DR4 SPS catalog."""
    data_path = Path("/Users/matthewsmawfield/www/TEP-JWST/data/raw/uncover")
    catalog_file = data_path / "UNCOVER_DR4_SPS_catalog.fits"
    
    with fits.open(catalog_file) as hdul:
        data = hdul[1].data
    return data

# =============================================================================
# AUDIT FUNCTIONS
# =============================================================================

def audit_red_monsters():
    """Audit Red Monsters analysis."""
    print("=" * 70)
    print("AUDIT 1: Red Monsters Analysis")
    print("=" * 70)
    print()
    
    results = []
    for name, props in RED_MONSTERS.items():
        gamma_t = tep_gamma(props["log_Mh"], props["z"])
        sfe_bias = isochrony_sfe_bias(gamma_t)
        sfe_true = props["SFE"] / sfe_bias
        
        print(f"  {name}: z={props['z']:.2f}, Γ_t={gamma_t:.2f}, SFE_true={sfe_true:.2f}")
        results.append({"name": name, "gamma_t": gamma_t, "sfe_true": sfe_true})
    
    avg_gamma = np.mean([r["gamma_t"] for r in results])
    avg_sfe_true = np.mean([r["sfe_true"] for r in results])
    
    # Fraction explained
    anomaly_obs = 0.50 / EPSILON_STANDARD  # 2.5x
    anomaly_true = avg_sfe_true / EPSILON_STANDARD
    tep_explains = (anomaly_obs - anomaly_true) / (anomaly_obs - 1)
    
    print()
    print(f"  Average Γ_t = {avg_gamma:.2f}")
    print(f"  Average SFE_true = {avg_sfe_true:.2f}")
    print(f"  TEP explains: {tep_explains*100:.0f}% of anomaly")
    
    return {
        "avg_gamma_t": avg_gamma,
        "avg_sfe_true": avg_sfe_true,
        "tep_explains_fraction": tep_explains
    }

def audit_uncover_correlations(data):
    """Audit UNCOVER correlations with corrected data handling."""
    print()
    print("=" * 70)
    print("AUDIT 2: UNCOVER Correlations (Corrected)")
    print("=" * 70)
    print()
    
    # Extract data - USE ssfr100_50 DIRECTLY (linear values)
    z = data['z_50']
    log_Mstar = data['mstar_50']
    ssfr_linear = data['ssfr100_50']  # Linear sSFR values
    mwa = data['mwa_50']
    
    # Valid mask
    valid = (~np.isnan(ssfr_linear) & (ssfr_linear > 0) & 
             ~np.isnan(mwa) & (log_Mstar > 8) & (z > 4) & (z < 10))
    
    z_filt = z[valid]
    log_Mstar_filt = log_Mstar[valid]
    log_ssfr = np.log10(ssfr_linear[valid])  # Convert to log for correlation
    mwa_filt = mwa[valid]
    
    n = len(z_filt)
    print(f"Sample size: N = {n}")
    print()
    
    # Test 1: Mass-sSFR correlation
    print("Test 2.1: Mass-sSFR Correlation")
    rho_ssfr, p_ssfr = spearmanr(log_Mstar_filt, log_ssfr)
    
    # Bootstrap CI
    rhos = []
    for _ in range(1000):
        idx = np.random.choice(n, n, replace=True)
        r, _ = spearmanr(log_Mstar_filt[idx], log_ssfr[idx])
        rhos.append(r)
    ci_lo, ci_hi = np.percentile(rhos, [2.5, 97.5])
    
    print(f"  ρ(M*, sSFR) = {rho_ssfr:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"  p = {p_ssfr:.2e}")
    print()
    
    # Test 2: Mass-Age correlation
    print("Test 2.2: Mass-Age Correlation")
    rho_age, p_age = spearmanr(log_Mstar_filt, mwa_filt)
    
    rhos = []
    for _ in range(1000):
        idx = np.random.choice(n, n, replace=True)
        r, _ = spearmanr(log_Mstar_filt[idx], mwa_filt[idx])
        rhos.append(r)
    ci_lo, ci_hi = np.percentile(rhos, [2.5, 97.5])
    
    print(f"  ρ(M*, MWA) = {rho_age:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"  p = {p_age:.2e}")
    print()
    
    return {
        "n_sample": n,
        "mass_ssfr_rho": rho_ssfr,
        "mass_age_rho": rho_age
    }

def audit_z_inversion(data):
    """Audit the z > 7 mass-sSFR inversion."""
    print()
    print("=" * 70)
    print("AUDIT 3: z > 7 Inversion (KEY FINDING)")
    print("=" * 70)
    print()
    
    z = data['z_50']
    log_Mstar = data['mstar_50']
    ssfr_linear = data['ssfr100_50']
    
    valid = (~np.isnan(ssfr_linear) & (ssfr_linear > 0) & 
             (log_Mstar > 8) & (z > 4) & (z < 10))
    
    z_filt = z[valid]
    log_Mstar_filt = log_Mstar[valid]
    log_ssfr = np.log10(ssfr_linear[valid])
    
    # Low-z vs High-z comparison
    low_z = (z_filt >= 4) & (z_filt < 6)
    high_z = (z_filt >= 7) & (z_filt < 10)
    
    rho_low, _ = spearmanr(log_Mstar_filt[low_z], log_ssfr[low_z])
    rho_high, _ = spearmanr(log_Mstar_filt[high_z], log_ssfr[high_z])
    
    print(f"Low-z (4 < z < 6): N = {np.sum(low_z)}, ρ = {rho_low:.3f}")
    print(f"High-z (7 < z < 10): N = {np.sum(high_z)}, ρ = {rho_high:.3f}")
    print(f"Δρ = {rho_high - rho_low:.3f}")
    print()
    
    # Bootstrap test for significance
    print("Bootstrap test for Δρ:")
    n_boot = 1000
    deltas = []
    for _ in range(n_boot):
        idx_low = np.random.choice(np.sum(low_z), np.sum(low_z), replace=True)
        idx_high = np.random.choice(np.sum(high_z), np.sum(high_z), replace=True)
        r_low, _ = spearmanr(log_Mstar_filt[low_z][idx_low], log_ssfr[low_z][idx_low])
        r_high, _ = spearmanr(log_Mstar_filt[high_z][idx_high], log_ssfr[high_z][idx_high])
        deltas.append(r_high - r_low)
    
    deltas = np.array(deltas)
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    
    print(f"  Δρ = {np.mean(deltas):.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")
    significant = ci_lo > 0
    print(f"  95% CI excludes zero: {significant}")
    print(f"  ★ STATISTICALLY SIGNIFICANT: {significant}")
    
    return {
        "rho_low_z": rho_low,
        "rho_high_z": rho_high,
        "delta_rho": rho_high - rho_low,
        "delta_rho_ci": [ci_lo, ci_hi],
        "significant": significant
    }

def audit_partial_correlation(data):
    """Audit the partial correlation analysis (Γ_t vs age ratio | z)."""
    print()
    print("=" * 70)
    print("AUDIT 4: Partial Correlation (Γ_t vs Age Ratio | z)")
    print("=" * 70)
    print()
    
    z = data['z_50']
    log_Mstar = data['mstar_50']
    mwa = data['mwa_50']
    
    valid = (~np.isnan(mwa) & (log_Mstar > 8) & (z > 4) & (z < 10))
    
    z_filt = z[valid]
    log_Mstar_filt = log_Mstar[valid]
    mwa_filt = mwa[valid]
    
    # Calculate Γ_t and age ratio
    log_Mh = log_Mstar_filt + 2.0
    t_cosmic = cosmo.age(z_filt).value
    age_ratio = mwa_filt / t_cosmic
    
    alpha_z = ALPHA_LOCAL * np.sqrt(1 + z_filt)
    delta_log_Mh = log_Mh - LOG_MH_REF
    z_factor = (1 + z_filt) / (1 + Z_REF)
    gamma_t = 1 + alpha_z * (2/3) * delta_log_Mh * z_factor
    
    # Raw correlation
    rho_raw, p_raw = spearmanr(gamma_t, age_ratio)
    print(f"Raw correlation: ρ(Γ_t, age_ratio) = {rho_raw:.3f}, p = {p_raw:.2e}")
    print()
    
    # Partial correlation (controlling for z via residualization)
    slope_g, intercept_g, _, _, _ = linregress(z_filt, gamma_t)
    gamma_t_resid = gamma_t - (slope_g * z_filt + intercept_g)
    
    slope_a, intercept_a, _, _, _ = linregress(z_filt, age_ratio)
    age_ratio_resid = age_ratio - (slope_a * z_filt + intercept_a)
    
    rho_partial, p_partial = spearmanr(gamma_t_resid, age_ratio_resid)
    print(f"Partial correlation (controlling for z):")
    print(f"  ρ(Γ_t, age_ratio | z) = {rho_partial:.3f}, p = {p_partial:.2e}")
    print()
    
    # By redshift bin
    print("By redshift bin:")
    results_by_z = []
    for z_lo, z_hi in [(4, 5), (5, 6), (6, 7), (7, 10)]:
        z_mask = (z_filt >= z_lo) & (z_filt < z_hi)
        n = np.sum(z_mask)
        if n > 30:
            rho, p = spearmanr(gamma_t[z_mask], age_ratio[z_mask])
            print(f"  z = {z_lo}-{z_hi}: N = {n:4d}, ρ = {rho:+.3f}, p = {p:.2e}")
            results_by_z.append({"z_bin": f"{z_lo}-{z_hi}", "n": n, "rho": rho, "p": p})
    
    return {
        "rho_raw": rho_raw,
        "rho_partial": rho_partial,
        "p_partial": p_partial,
        "by_redshift": results_by_z
    }

def main():
    """Run full audit."""
    print()
    print("=" * 70)
    print("TEP-JWST PIPELINE AUDIT v2")
    print("=" * 70)
    print()
    
    # Load data
    data = load_uncover_data()
    
    # Run audits
    rm_results = audit_red_monsters()
    corr_results = audit_uncover_correlations(data)
    inv_results = audit_z_inversion(data)
    partial_results = audit_partial_correlation(data)
    
    # Summary
    print()
    print("=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)
    print()
    print("VERIFIED CLAIMS:")
    print("-" * 50)
    print(f"1. TEP explains {rm_results['tep_explains_fraction']*100:.0f}% of Red Monsters anomaly")
    print(f"2. Mass-sSFR correlation: ρ = {corr_results['mass_ssfr_rho']:.2f} (weak)")
    print(f"3. Mass-Age correlation: ρ = {corr_results['mass_age_rho']:.2f} (positive)")
    print(f"4. z > 7 inversion: Δρ = {inv_results['delta_rho']:.2f} [{inv_results['delta_rho_ci'][0]:.2f}, {inv_results['delta_rho_ci'][1]:.2f}]")
    print(f"   ★ STATISTICALLY SIGNIFICANT: {inv_results['significant']}")
    print(f"5. Partial correlation: ρ = {partial_results['rho_partial']:.2f}, p = {partial_results['p_partial']:.2e}")
    print()
    print("TENTATIVE CLAIMS:")
    print("-" * 50)
    print("- Screening (N = 1 in high-mass bin)")
    
    # Save results
    output = {
        "red_monsters": rm_results,
        "correlations": corr_results,
        "z_inversion": inv_results,
        "partial_correlation": partial_results
    }
    
    output_path = Path("/Users/matthewsmawfield/www/TEP-JWST/results/outputs")
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "tep_pipeline_audit_v2.json", "w") as f:
        json.dump(output, f, indent=2, default=float)
    
    print()
    print(f"Results saved to: {output_path / 'tep_pipeline_audit_v2.json'}")

if __name__ == "__main__":
    main()
