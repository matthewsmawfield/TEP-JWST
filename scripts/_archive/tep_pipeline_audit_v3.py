#!/usr/bin/env python3
"""
TEP-JWST Pipeline Audit v3: Complete Seven-Thread Analysis

This script audits all claims made in the manuscript with the full
seven-thread evidence pattern:

1. z > 7 Mass-sSFR Inversion
2. Γ_t vs Age Ratio (partial correlation)
3. Γ_t vs Metallicity (partial correlation)
4. Γ_t vs Dust (partial correlation)
5. z > 8 Dust Anomaly
6. Age-Metallicity Coherence
7. Multi-Property Split Test

Run this to verify all manuscript claims are reproducible.
"""

import numpy as np
from astropy.io import fits
from astropy.cosmology import Planck18 as cosmo
from scipy.stats import spearmanr, linregress, mannwhitneyu
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

def partial_corr(x, y, z_control):
    """Partial correlation controlling for z."""
    slope_x, int_x, _, _, _ = linregress(z_control, x)
    x_resid = x - (slope_x * z_control + int_x)
    
    slope_y, int_y, _, _, _ = linregress(z_control, y)
    y_resid = y - (slope_y * z_control + int_y)
    
    return spearmanr(x_resid, y_resid)

def bootstrap_ci(x, y, n_boot=1000, func=spearmanr):
    """Bootstrap confidence interval for correlation."""
    n = len(x)
    rhos = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        r, _ = func(x[idx], y[idx])
        rhos.append(r)
    return np.percentile(rhos, [2.5, 97.5])

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
# THREAD TESTS
# =============================================================================

def test_thread_1_z7_inversion(data):
    """Thread 1: z > 7 Mass-sSFR Inversion."""
    print("\n" + "=" * 60)
    print("THREAD 1: z > 7 Mass-sSFR Inversion")
    print("=" * 60)
    
    z = data['z_50']
    mstar = data['mstar_50']
    ssfr = data['ssfr100_50']
    
    valid = (~np.isnan(ssfr) & (ssfr > 0) & (mstar > 8) & (z > 4) & (z < 10))
    z_filt = z[valid]
    mstar_filt = mstar[valid]
    log_ssfr = np.log10(ssfr[valid])
    
    low_z = (z_filt >= 4) & (z_filt < 6)
    high_z = (z_filt >= 7) & (z_filt < 10)
    
    rho_low, _ = spearmanr(mstar_filt[low_z], log_ssfr[low_z])
    rho_high, _ = spearmanr(mstar_filt[high_z], log_ssfr[high_z])
    delta_rho = rho_high - rho_low
    
    # Bootstrap for delta
    n_boot = 1000
    deltas = []
    for _ in range(n_boot):
        idx_low = np.random.choice(np.sum(low_z), np.sum(low_z), replace=True)
        idx_high = np.random.choice(np.sum(high_z), np.sum(high_z), replace=True)
        r_low, _ = spearmanr(mstar_filt[low_z][idx_low], log_ssfr[low_z][idx_low])
        r_high, _ = spearmanr(mstar_filt[high_z][idx_high], log_ssfr[high_z][idx_high])
        deltas.append(r_high - r_low)
    ci = np.percentile(deltas, [2.5, 97.5])
    
    significant = ci[0] > 0
    
    print(f"  Low-z (4-6): N = {np.sum(low_z)}, ρ = {rho_low:.3f}")
    print(f"  High-z (7-10): N = {np.sum(high_z)}, ρ = {rho_high:.3f}")
    print(f"  Δρ = {delta_rho:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
    print(f"  ★ SIGNIFICANT: {significant}")
    
    return {"delta_rho": delta_rho, "ci": list(ci), "significant": significant}

def test_thread_2_gamma_age(data):
    """Thread 2: Γ_t vs Age Ratio (partial correlation)."""
    print("\n" + "=" * 60)
    print("THREAD 2: Γ_t vs Age Ratio (partial)")
    print("=" * 60)
    
    z = data['z_50']
    mstar = data['mstar_50']
    mwa = data['mwa_50']
    
    valid = (~np.isnan(mwa) & (mstar > 8) & (z > 4) & (z < 10))
    z_filt = z[valid]
    mstar_filt = mstar[valid]
    mwa_filt = mwa[valid]
    
    log_Mh = mstar_filt + 2.0
    t_cosmic = cosmo.age(z_filt).value
    age_ratio = mwa_filt / t_cosmic
    gamma_t = tep_gamma(log_Mh, z_filt)
    
    rho, p = partial_corr(gamma_t, age_ratio, z_filt)
    
    print(f"  N = {len(z_filt)}")
    print(f"  ρ(Γ_t, age_ratio | z) = {rho:.3f}")
    print(f"  p = {p:.2e}")
    print(f"  ★ SIGNIFICANT: {p < 0.001}")
    
    return {"rho": rho, "p": float(p), "significant": p < 0.001}

def test_thread_3_gamma_metallicity(data):
    """Thread 3: Γ_t vs Metallicity (partial correlation)."""
    print("\n" + "=" * 60)
    print("THREAD 3: Γ_t vs Metallicity (partial)")
    print("=" * 60)
    
    z = data['z_50']
    mstar = data['mstar_50']
    met = data['met_50']
    met_16 = data['met_16']
    met_84 = data['met_84']
    
    met_err = (met_84 - met_16) / 2
    valid = (~np.isnan(met) & ~np.isnan(met_err) & (met_err < 0.5) &
             (mstar > 8) & (z > 4) & (z < 10))
    
    z_filt = z[valid]
    mstar_filt = mstar[valid]
    met_filt = met[valid]
    
    log_Mh = mstar_filt + 2.0
    gamma_t = tep_gamma(log_Mh, z_filt)
    
    rho, p = partial_corr(gamma_t, met_filt, z_filt)
    
    print(f"  N = {len(z_filt)}")
    print(f"  ρ(Γ_t, metallicity | z) = {rho:.3f}")
    print(f"  p = {p:.2e}")
    print(f"  ★ SIGNIFICANT: {p < 0.001}")
    
    return {"rho": rho, "p": float(p), "significant": p < 0.001}

def test_thread_4_gamma_dust(data):
    """Thread 4: Γ_t vs Dust (partial correlation) - multi-property sample."""
    print("\n" + "=" * 60)
    print("THREAD 4: Γ_t vs Dust (partial, multi-property sample)")
    print("=" * 60)
    
    z = data['z_50']
    mstar = data['mstar_50']
    mwa = data['mwa_50']
    met = data['met_50']
    met_16 = data['met_16']
    met_84 = data['met_84']
    dust = data['dust2_50']
    ssfr = data['ssfr100_50']
    
    # Use multi-property sample with quality cuts (same as coherence analysis)
    met_err = (met_84 - met_16) / 2
    valid = (~np.isnan(mwa) & ~np.isnan(met) & ~np.isnan(dust) & 
             ~np.isnan(ssfr) & (ssfr > 0) & ~np.isnan(met_err) & (met_err < 0.5) &
             (mstar > 8) & (z > 4) & (z < 10))
    
    z_filt = z[valid]
    mstar_filt = mstar[valid]
    dust_filt = dust[valid]
    
    log_Mh = mstar_filt + 2.0
    gamma_t = tep_gamma(log_Mh, z_filt)
    
    rho, p = partial_corr(gamma_t, dust_filt, z_filt)
    
    print(f"  N = {len(z_filt)}")
    print(f"  ρ(Γ_t, dust | z) = {rho:.3f}")
    print(f"  p = {p:.2e}")
    print(f"  ★ SIGNIFICANT: {p < 0.001}")
    
    return {"rho": rho, "p": float(p), "significant": p < 0.001}

def test_thread_5_z8_dust(data):
    """Thread 5: z > 8 Dust Anomaly."""
    print("\n" + "=" * 60)
    print("THREAD 5: z > 8 Dust Anomaly")
    print("=" * 60)
    
    z = data['z_50']
    mstar = data['mstar_50']
    dust = data['dust2_50']
    
    valid = (~np.isnan(dust) & (mstar > 8) & (z >= 8) & (z < 10))
    mstar_filt = mstar[valid]
    dust_filt = dust[valid]
    
    rho, p = spearmanr(mstar_filt, dust_filt)
    ci = bootstrap_ci(mstar_filt, dust_filt)
    
    print(f"  N = {len(mstar_filt)}")
    print(f"  ρ(M*, dust) at z > 8 = {rho:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
    print(f"  p = {p:.2e}")
    print(f"  ★ SIGNIFICANT: {ci[0] > 0}")
    
    # Mean dust by mass
    print("\n  Mean A_V by mass:")
    for m_lo, m_hi in [(8, 8.5), (8.5, 9), (9, 10), (10, 12)]:
        m_mask = (mstar_filt >= m_lo) & (mstar_filt < m_hi)
        n = np.sum(m_mask)
        if n > 3:
            mean_dust = np.mean(dust_filt[m_mask])
            print(f"    log M* = {m_lo}-{m_hi}: N = {n:3d}, <A_V> = {mean_dust:.2f}")
    
    return {"rho": rho, "ci": list(ci), "significant": ci[0] > 0}

def test_thread_6_age_metallicity(data):
    """Thread 6: Age-Metallicity Coherence."""
    print("\n" + "=" * 60)
    print("THREAD 6: Age-Metallicity Coherence")
    print("=" * 60)
    
    z = data['z_50']
    mstar = data['mstar_50']
    mwa = data['mwa_50']
    met = data['met_50']
    met_16 = data['met_16']
    met_84 = data['met_84']
    
    met_err = (met_84 - met_16) / 2
    valid = (~np.isnan(mwa) & ~np.isnan(met) & ~np.isnan(met_err) & 
             (met_err < 0.3) & (mstar > 8) & (z > 4) & (z < 10))
    
    z_filt = z[valid]
    mwa_filt = mwa[valid]
    met_filt = met[valid]
    
    t_cosmic = cosmo.age(z_filt).value
    age_ratio = mwa_filt / t_cosmic
    
    rho, p = spearmanr(age_ratio, met_filt)
    ci = bootstrap_ci(age_ratio, met_filt)
    
    print(f"  N = {len(z_filt)}")
    print(f"  ρ(age_ratio, metallicity) = {rho:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
    print(f"  p = {p:.2e}")
    print(f"  ★ SIGNIFICANT: {ci[0] > 0}")
    
    return {"rho": rho, "ci": list(ci), "significant": ci[0] > 0}

def test_thread_7_multi_property(data):
    """Thread 7: Multi-Property Split Test."""
    print("\n" + "=" * 60)
    print("THREAD 7: Multi-Property Split Test")
    print("=" * 60)
    
    z = data['z_50']
    mstar = data['mstar_50']
    mwa = data['mwa_50']
    met = data['met_50']
    met_16 = data['met_16']
    met_84 = data['met_84']
    dust = data['dust2_50']
    ssfr = data['ssfr100_50']
    
    met_err = (met_84 - met_16) / 2
    valid = (~np.isnan(mwa) & ~np.isnan(met) & ~np.isnan(dust) & 
             ~np.isnan(ssfr) & (ssfr > 0) & ~np.isnan(met_err) & (met_err < 0.5) &
             (mstar > 8) & (z > 4) & (z < 10))
    
    z_filt = z[valid]
    mstar_filt = mstar[valid]
    mwa_filt = mwa[valid]
    met_filt = met[valid]
    dust_filt = dust[valid]
    
    log_Mh = mstar_filt + 2.0
    t_cosmic = cosmo.age(z_filt).value
    age_ratio = mwa_filt / t_cosmic
    gamma_t = tep_gamma(log_Mh, z_filt)
    
    # Split by median Gamma_t
    gamma_median = np.median(gamma_t)
    high_gamma = gamma_t > gamma_median
    low_gamma = ~high_gamma
    
    print(f"  N = {len(z_filt)}")
    print(f"  Γ_t median = {gamma_median:.3f}")
    print()
    
    results = {}
    for name, q in [('age_ratio', age_ratio), ('metallicity', met_filt), ('dust', dust_filt)]:
        mean_low = np.mean(q[low_gamma])
        mean_high = np.mean(q[high_gamma])
        diff = mean_high - mean_low
        stat, p = mannwhitneyu(q[high_gamma], q[low_gamma], alternative='greater')
        
        print(f"  {name}:")
        print(f"    Low Γ_t: {mean_low:.3f}")
        print(f"    High Γ_t: {mean_high:.3f}")
        print(f"    Diff: {diff:+.3f}")
        print(f"    p = {p:.2e}")
        print(f"    ★ SIGNIFICANT: {p < 0.001}")
        
        results[name] = {"diff": diff, "p": float(p), "significant": p < 0.001}
    
    return results

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run complete seven-thread audit."""
    print()
    print("=" * 70)
    print("TEP-JWST PIPELINE AUDIT v3: THE SEVEN THREADS")
    print("=" * 70)
    
    data = load_uncover_data()
    
    results = {
        "thread_1_z7_inversion": test_thread_1_z7_inversion(data),
        "thread_2_gamma_age": test_thread_2_gamma_age(data),
        "thread_3_gamma_metallicity": test_thread_3_gamma_metallicity(data),
        "thread_4_gamma_dust": test_thread_4_gamma_dust(data),
        "thread_5_z8_dust": test_thread_5_z8_dust(data),
        "thread_6_age_metallicity": test_thread_6_age_metallicity(data),
        "thread_7_multi_property": test_thread_7_multi_property(data),
    }
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: THE TAPESTRY")
    print("=" * 70)
    
    threads_significant = sum([
        results["thread_1_z7_inversion"]["significant"],
        results["thread_2_gamma_age"]["significant"],
        results["thread_3_gamma_metallicity"]["significant"],
        results["thread_4_gamma_dust"]["significant"],
        results["thread_5_z8_dust"]["significant"],
        results["thread_6_age_metallicity"]["significant"],
        all(v["significant"] for v in results["thread_7_multi_property"].values()),
    ])
    
    print(f"\n  Threads significant: {threads_significant}/7")
    print()
    print("  Thread 1 (z > 7 Inversion):     ", "✓" if results["thread_1_z7_inversion"]["significant"] else "✗")
    print("  Thread 2 (Γ_t vs Age):          ", "✓" if results["thread_2_gamma_age"]["significant"] else "✗")
    print("  Thread 3 (Γ_t vs Metallicity):  ", "✓" if results["thread_3_gamma_metallicity"]["significant"] else "✗")
    print("  Thread 4 (Γ_t vs Dust):         ", "✓" if results["thread_4_gamma_dust"]["significant"] else "✗")
    print("  Thread 5 (z > 8 Dust):          ", "✓" if results["thread_5_z8_dust"]["significant"] else "✗")
    print("  Thread 6 (Age-Metallicity):     ", "✓" if results["thread_6_age_metallicity"]["significant"] else "✗")
    print("  Thread 7 (Multi-Property):      ", "✓" if all(v["significant"] for v in results["thread_7_multi_property"].values()) else "✗")
    
    # Save results
    output_path = Path("/Users/matthewsmawfield/www/TEP-JWST/results/outputs")
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "tep_pipeline_audit_v3.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    
    print(f"\n  Results saved to: {output_path / 'tep_pipeline_audit_v3.json'}")

if __name__ == "__main__":
    main()
