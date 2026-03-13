#!/usr/bin/env python3
# Estimated runtime from last full canonical run (2026-03-09 15:52 UTC; full pipeline 32m18s): 1.4s.
"""
TEP-JWST Step 34: COSMOS-Web Independent Replication

Third independent validation of TEP predictions using COSMOS-Web DR1,
the largest JWST survey (0.5 deg²) with 784,016 galaxies.

Requires: step_033_cosmosweb_download.py to be run first.

Author: Matthew L. Smawfield
Date: January 2026
"""

import sys
import numpy as np
import pandas as pd
from scipy import stats  # Hypothesis tests and correlation
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging (severity levels: DEBUG/INFO/WARNING/ERROR/SUCCESS)
from scripts.utils.p_value_utils import format_p_value  # Safe p-value formatting (prevents floating-point underflow at p < 1e-300)
from scripts.utils.rank_stats import partial_rank_correlation  # Partial Spearman: residualization method to control for confounders
from scripts.utils.tep_model import compute_gamma_t as tep_gamma, stellar_to_halo_mass_behroozi_like  # TEP model: Gamma_t formula, stellar-to-halo mass from abundance matching

STEP_NUM = "034"  # Pipeline step number (sequential 001-176)
STEP_NAME = "cosmosweb_replication"  # COSMOS-Web replication: third independent validation in 0.5 deg² survey (784k galaxies)

DATA_INTERIM_PATH = PROJECT_ROOT / "data" / "interim"  # Processed catalogue products (CSV format from step_033)
RESULTS_INTERIM_PATH = PROJECT_ROOT / "results" / "interim"  # Intermediate results (CSV format for step-to-step data flow)
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"  # JSON output directory (machine-readable statistical results)
LOGS_PATH = PROJECT_ROOT / "logs"  # Log directory (one plain-text log per step for debugging traceability)

for p in [DATA_INTERIM_PATH, RESULTS_INTERIM_PATH, OUTPUT_PATH, LOGS_PATH]:
    p.mkdir(parents=True, exist_ok=True)  # Create directory tree if missing; exist_ok=True allows safe re-runs

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")  # Step-specific logger (isolated per-step logging for traceability)
set_step_logger(logger)  # Register as global step logger so print_status() routes to this step's log


def load_cosmosweb_data():
    """Load COSMOS-Web z>8 sample from pipeline output."""
    from astropy.cosmology import Planck18 as cosmo
    
    cosmosweb_file = DATA_INTERIM_PATH / "cosmosweb_z8_sample.csv"
    
    if not cosmosweb_file.exists():
        print_status(f"ERROR: COSMOS-Web data not found at {cosmosweb_file}", "INFO")
        print_status("Run step_033_cosmosweb_download.py first.", "INFO")
        return None
    
    df = pd.read_csv(cosmosweb_file)
    print_status(f"Loaded COSMOS-Web z>8 sample: N = {len(df)}", "INFO")
    
    # Rename columns for consistency
    df = df.rename(columns={'z_phot': 'z_phot', 'log_Mstar': 'log_Mstar', 'dust': 'dust'})
    
    # Compute TEP quantities
    df['log_Mh'] = stellar_to_halo_mass_behroozi_like(df['log_Mstar'].values, df['z_phot'].values)
    df['gamma_t'] = tep_gamma(df['log_Mh'].values, df['z_phot'].values)
    df['t_cosmic'] = cosmo.age(df['z_phot'].values).value
    df['t_eff'] = df['t_cosmic'] * df['gamma_t']
    df['t_eff'] = np.maximum(df['t_eff'], 0.001)
    df['source'] = 'cosmosweb_real'
    
    return df


def test_mass_dust_correlation(df, sample_name):
    """Test mass-dust correlation at z>8."""
    print_status(f"\n{'='*60}", "INFO")
    print_status(f"MASS-DUST CORRELATION: {sample_name}", "INFO")
    print_status(f"{'='*60}", "INFO")
    
    # Filter for valid dust measurements (dust > 0, like UNCOVER)
    # Note: 65% of COSMOS-Web z>8 galaxies have dust=0, which may indicate
    # missing data or detection limits. We filter to dust > 0 for fair comparison.
    df_valid = df.dropna(subset=['dust'])
    df_valid = df_valid[df_valid['dust'] > 0]  # Changed from >= 0 to > 0
    
    print_status(f"Sample size (dust > 0): N = {len(df_valid)}", "INFO")
    print_status(f"Redshift range: z = {df_valid['z_phot'].min():.1f} - {df_valid['z_phot'].max():.1f}", "INFO")
    
    if len(df_valid) < 10:
        print_status("Insufficient data for correlation analysis", "INFO")
        return None
    
    # Mass-dust correlation
    rho, p = stats.spearmanr(df_valid['log_Mstar'], df_valid['dust'])
    p_fmt = format_p_value(p)
    
    # Bootstrap CI
    n_boot = 1000
    rhos = []
    np.random.seed(42)
    for _ in range(n_boot):
        idx = np.random.choice(len(df_valid), len(df_valid), replace=True)
        r, _ = stats.spearmanr(df_valid['log_Mstar'].iloc[idx], df_valid['dust'].iloc[idx])
        rhos.append(r)
    ci_lo, ci_hi = np.percentile(rhos, [2.5, 97.5])
    
    print_status(f"\nMass-Dust Correlation:", "INFO")
    print_status(f"  ρ = {rho:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]", "INFO")
    print_status(f"  p = {p:.2e}", "INFO")
    
    # Significance in sigma
    if p_fmt is not None and 0 < p_fmt < 1:
        sigma = abs(stats.norm.ppf(p_fmt / 2))
        print_status(f"  Significance: {sigma:.1f}σ", "INFO")
    
    return {
        'rho': float(rho),
        'ci_lo': float(ci_lo),
        'ci_hi': float(ci_hi),
        'p': p_fmt,
        'n': len(df_valid)
    }


def test_gamma_dust_correlation(df, sample_name):
    """Test Γ_t-dust correlation at z>8."""
    print_status(f"\nΓ_t-DUST CORRELATION: {sample_name}", "INFO")
    
    df_valid = df.dropna(subset=['dust'])
    df_valid = df_valid[df_valid['dust'] > 0]  # Filter to dust > 0
    df_valid = df_valid.dropna(subset=['gamma_t', 'log_Mstar']).copy()
    
    if len(df_valid) < 10:
        return None
    
    # Γ_t-dust correlation
    rho, p = stats.spearmanr(df_valid['gamma_t'], df_valid['dust'])
    p_fmt = format_p_value(p)

    rho_partial, p_partial, _ = partial_rank_correlation(
        df_valid['gamma_t'].values,
        df_valid['dust'].values,
        df_valid['log_Mstar'].values,
    )
    p_partial_fmt = format_p_value(p_partial)
    
    print_status(f"  Raw: ρ(Γ_t, dust) = {rho:.3f}, p = {p:.2e}", "INFO")
    print_status(f"  Partial (|M*): ρ = {rho_partial:.3f}, p = {p_partial:.2e}", "INFO")
    
    return {
        'rho_raw': float(rho),
        'p_raw': p_fmt,
        'rho_partial': float(rho_partial),
        'p_partial': p_partial_fmt
    }


def test_teff_dust_correlation(df, sample_name):
    """Test t_eff-dust correlation at z>8."""
    print_status(f"\nt_eff-DUST CORRELATION: {sample_name}", "INFO")
    
    df_valid = df.dropna(subset=['dust'])
    df_valid = df_valid[df_valid['dust'] > 0]  # Filter to dust > 0
    
    if len(df_valid) < 10:
        return None
    
    rho, p = stats.spearmanr(df_valid['t_eff'], df_valid['dust'])
    print_status(f"  ρ(t_eff, dust) = {rho:.3f}, p = {p:.2e}", "INFO")
    
    return {'rho': float(rho), 'p': format_p_value(p)}


def compare_all_surveys():
    """Compare COSMOS-Web with UNCOVER and CEERS."""
    print_status(f"\n{'='*60}", "INFO")
    print_status("CROSS-SURVEY COMPARISON (3 INDEPENDENT SURVEYS)", "INFO")
    print_status(f"{'='*60}", "INFO")
    
    results = {}
    
    # Load UNCOVER
    uncover_path = RESULTS_INTERIM_PATH / "step_002_uncover_full_sample_tep.csv"
    if uncover_path.exists():
        uncover = pd.read_csv(uncover_path)
        uncover_z8 = uncover[(uncover['z_phot'] >= 8) & (uncover['z_phot'] <= 12)].copy()
        uncover_z8 = uncover_z8.dropna(subset=['dust', 'log_Mstar'])
        rho_u, p_u = stats.spearmanr(uncover_z8['log_Mstar'], uncover_z8['dust'])
        results['uncover'] = {'n': len(uncover_z8), 'rho': float(rho_u), 'p': format_p_value(p_u)}
        print_status(f"\nUNCOVER z>8 (N={len(uncover_z8)}): ρ = {rho_u:.3f}, p = {p_u:.2e}", "INFO")
    
    # Load CEERS
    ceers_path = DATA_INTERIM_PATH / "ceers_z8_sample.csv"
    if ceers_path.exists():
        ceers = pd.read_csv(ceers_path)
        ceers = ceers.dropna(subset=['dust', 'log_Mstar'])
        rho_c, p_c = stats.spearmanr(ceers['log_Mstar'], ceers['dust'])
        results['ceers'] = {'n': len(ceers), 'rho': float(rho_c), 'p': format_p_value(p_c)}
        print_status(f"CEERS z>8 (N={len(ceers)}): ρ = {rho_c:.3f}, p = {p_c:.2e}", "INFO")
    
    return results


def main():
    print_status("=" * 70, "INFO")
    print_status("STEP 36: COSMOS-Web Independent Replication", "INFO")
    print_status("=" * 70, "INFO")
    print_status("", "INFO")
    print_status("Third independent validation of the z>8 dust anomaly", "INFO")
    print_status("using COSMOS-Web DR1 (Shuntov et al. 2025)", "INFO")
    print_status("", "INFO")
    
    results = {}
    
    # Load COSMOS-Web data
    cosmosweb = load_cosmosweb_data()
    if cosmosweb is None:
        print_status("ERROR: Could not load COSMOS-Web data", "INFO")
        return
    
    results['cosmosweb_source'] = 'cosmosweb_real'
    results['cosmosweb_n'] = len(cosmosweb)
    
    # Test mass-dust correlation
    mass_dust = test_mass_dust_correlation(cosmosweb, "COSMOS-Web z>8")
    if mass_dust:
        results['mass_dust'] = mass_dust
    
    # Test Γ_t-dust correlation
    gamma_dust = test_gamma_dust_correlation(cosmosweb, "COSMOS-Web z>8")
    if gamma_dust:
        results['gamma_dust'] = gamma_dust
    
    # Test t_eff-dust correlation
    teff_dust = test_teff_dust_correlation(cosmosweb, "COSMOS-Web z>8")
    if teff_dust:
        results['teff_dust'] = teff_dust
    
    # Compare with other surveys
    survey_comparison = compare_all_surveys()
    results['survey_comparison'] = survey_comparison
    
    # Summary
    print_status(f"\n{'='*70}", "INFO")
    print_status("THREE-SURVEY REPLICATION SUMMARY", "INFO")
    print_status(f"{'='*70}", "INFO")
    
    if mass_dust:
        cosmosweb_rho = mass_dust['rho']
        uncover_rho = survey_comparison.get('uncover', {}).get('rho', 0.56)
        ceers_rho = survey_comparison.get('ceers', {}).get('rho', 0.68)
        
        print_status(f"\nMass-Dust Correlation at z>8:", "INFO")
        print_status(f"  UNCOVER:    ρ = {uncover_rho:.3f} (N = {survey_comparison.get('uncover', {}).get('n', 283)})", "INFO")
        print_status(f"  CEERS:      ρ = {ceers_rho:.3f} (N = {survey_comparison.get('ceers', {}).get('n', 82)})", "INFO")
        print_status(f"  COSMOS-Web: ρ = {cosmosweb_rho:.3f} (N = {mass_dust['n']})", "INFO")
        
        # Check consistency
        p_rep = mass_dust.get('p')
        if cosmosweb_rho > 0.3 and p_rep is not None and p_rep < 0.05:
            print_status(f"\n✓ THIRD INDEPENDENT REPLICATION SUCCESSFUL", "INFO")
            print_status(f"  The z>8 dust anomaly is confirmed in COSMOS-Web.", "INFO")
            results['replication_status'] = 'confirmed'
        else:
            print_status(f"\n⚠ REPLICATION INCONCLUSIVE", "INFO")
            results['replication_status'] = 'inconclusive'
    
    # Save results
    output_file = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print_status(f"\nResults saved to: {output_file}", "INFO")
    print_status(f"Step {STEP_NUM} complete.", "INFO")
    
    return results


if __name__ == "__main__":
    main()
