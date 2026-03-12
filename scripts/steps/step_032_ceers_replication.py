#!/usr/bin/env python3
# Estimated runtime from last full canonical run (2026-03-09 15:52 UTC; full pipeline 32m18s): 1.3s.
"""
TEP-JWST Step 32: Independent Replication with CEERS Data

This step provides independent validation of TEP predictions using the
CEERS (Cosmic Evolution Early Release Science) survey DR1 catalog,
which is completely independent of the UNCOVER data used in the primary analysis.

Key test: Does the z>8 dust anomaly replicate in CEERS?

Requires: step_031_ceers_download.py to be run first to download the catalog.

Author: Matthew L. Smawfield
Date: January 2026
"""

import sys
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.p_value_utils import format_p_value
from scripts.utils.rank_stats import partial_rank_correlation
from scripts.utils.tep_model import compute_gamma_t as tep_gamma, stellar_to_halo_mass_behroozi_like

STEP_NUM = "032"
STEP_NAME = "ceers_replication"

DATA_INTERIM_PATH = PROJECT_ROOT / "data" / "interim"
RESULTS_INTERIM_PATH = PROJECT_ROOT / "results" / "interim"
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"
LOGS_PATH = PROJECT_ROOT / "logs"

for p in [DATA_INTERIM_PATH, RESULTS_INTERIM_PATH, OUTPUT_PATH, LOGS_PATH]:
    p.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")
set_step_logger(logger)

BOOTSTRAP_SEED = 42


def load_ceers_data():
    """Load CEERS z>8 sample from pipeline output."""
    from astropy.cosmology import Planck18 as cosmo
    
    ceers_file = DATA_INTERIM_PATH / "ceers_z8_sample.csv"
    
    if not ceers_file.exists():
        print_status(f"ERROR: CEERS data not found at {ceers_file}", "INFO")
        print_status("Run step_031_ceers_download.py first to download the catalog.", "INFO")
        return None
    
    df = pd.read_csv(ceers_file)
    print_status(f"Loaded CEERS z>8 sample: N = {len(df)}", "INFO")
    
    # Compute TEP quantities
    df['log_Mh'] = stellar_to_halo_mass_behroozi_like(df['log_Mstar'].values, df['z_phot'].values)
    df['gamma_t'] = tep_gamma(df['log_Mh'].values, df['z_phot'].values)
    df['t_cosmic'] = cosmo.age(df['z_phot'].values).value
    df['t_eff'] = df['t_cosmic'] * df['gamma_t']
    df['t_eff'] = np.maximum(df['t_eff'], 0.001)
    df['source'] = 'ceers_real'
    
    return df


def test_mass_dust_correlation(df, sample_name):
    """Test mass-dust correlation at z>8."""
    print_status(f"\n{'='*60}", "INFO")
    print_status(f"MASS-DUST CORRELATION: {sample_name}", "INFO")
    print_status(f"{'='*60}", "INFO")
    
    print_status(f"Sample size: N = {len(df)}", "INFO")
    print_status(f"Redshift range: z = {df['z_phot'].min():.1f} - {df['z_phot'].max():.1f}", "INFO")
    
    # Mass-dust correlation
    rho, p = stats.spearmanr(df['log_Mstar'], df['dust'])
    
    # Bootstrap CI
    n_boot = 1000
    rhos = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for _ in range(n_boot):
        idx = rng.choice(len(df), len(df), replace=True)
        r, _ = stats.spearmanr(df['log_Mstar'].iloc[idx], df['dust'].iloc[idx])
        rhos.append(r)
    ci_lo, ci_hi = np.percentile(rhos, [2.5, 97.5])
    
    print_status(f"\nMass-Dust Correlation:", "INFO")
    print_status(f"  ρ = {rho:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]", "INFO")
    print_status(f"  p = {p:.2e}", "INFO")
    
    return {
        'rho': float(rho),
        'ci_lo': float(ci_lo),
        'ci_hi': float(ci_hi),
        'p': format_p_value(p),
        'n': len(df)
    }


def test_gamma_dust_correlation(df, sample_name):
    """Test Γ_t-dust correlation at z>8."""
    print_status(f"\nΓ_t-DUST CORRELATION: {sample_name}", "INFO")

    df = df.dropna(subset=['gamma_t', 'dust', 'log_Mstar']).copy()
    
    # Γ_t-dust correlation
    rho, p = stats.spearmanr(df['gamma_t'], df['dust'])

    rho_partial, p_partial, _ = partial_rank_correlation(
        df['gamma_t'].values,
        df['dust'].values,
        df['log_Mstar'].values,
    )
    
    print_status(f"  Raw: ρ(Γ_t, dust) = {rho:.3f}, p = {p:.2e}", "INFO")
    print_status(f"  Partial (|M*): ρ = {rho_partial:.3f}, p = {p_partial:.2e}", "INFO")
    
    return {
        'rho_raw': float(rho),
        'p_raw': format_p_value(p),
        'rho_partial': float(rho_partial),
        'p_partial': format_p_value(p_partial)
    }


def compare_with_uncover():
    """Compare CEERS results with UNCOVER results."""
    print_status(f"\n{'='*60}", "INFO")
    print_status("CROSS-SURVEY COMPARISON", "INFO")
    print_status(f"{'='*60}", "INFO")
    
    # Load UNCOVER z>8 results
    uncover_path = RESULTS_INTERIM_PATH / "step_002_uncover_full_sample_tep.csv"
    if uncover_path.exists():
        uncover = pd.read_csv(uncover_path)
        uncover_z8 = uncover[(uncover['z_phot'] >= 8) & (uncover['z_phot'] <= 10)].copy()
        uncover_z8 = uncover_z8.dropna(subset=['dust', 'log_Mstar', 'gamma_t'])
        
        rho_uncover, p_uncover = stats.spearmanr(uncover_z8['log_Mstar'], uncover_z8['dust'])
        print_status(f"\nUNCOVER z>8 (N={len(uncover_z8)}):", "INFO")
        print_status(f"  ρ(M*, dust) = {rho_uncover:.3f}, p = {p_uncover:.2e}", "INFO")
        
        return {
            'uncover_n': len(uncover_z8),
            'uncover_rho': float(rho_uncover),
            'uncover_p': format_p_value(p_uncover)
        }
    
    return None


def main():
    print_status("=" * 70, "INFO")
    print_status("STEP 32: Independent Replication with CEERS Data", "INFO")
    print_status("=" * 70, "INFO")
    print_status("", "INFO")
    print_status("Testing whether the z>8 dust anomaly replicates in CEERS,", "INFO")
    print_status("an independent JWST survey.", "INFO")
    print_status("", "INFO")
    
    results = {}
    
    # Load CEERS data
    ceers = load_ceers_data()
    if ceers is None:
        print_status("CEERS data unavailable — run step_031 to download. Aborting.", "ERROR")
        return {"status": "aborted", "reason": "missing CEERS data"}
    results['ceers_source'] = ceers['source'].iloc[0]
    results['ceers_n'] = len(ceers)
    
    # Test mass-dust correlation
    results['mass_dust'] = test_mass_dust_correlation(ceers, "CEERS z>8")
    
    # Test Γ_t-dust correlation
    results['gamma_dust'] = test_gamma_dust_correlation(ceers, "CEERS z>8")
    
    # Compare with UNCOVER
    uncover_results = compare_with_uncover()
    if uncover_results:
        results['uncover'] = uncover_results
    
    # Summary
    print_status(f"\n{'='*70}", "INFO")
    print_status("REPLICATION SUMMARY", "INFO")
    print_status(f"{'='*70}", "INFO")
    
    ceers_rho = results['mass_dust']['rho']
    uncover_rho = results.get('uncover', {}).get('uncover_rho')

    print_status(f"\nMass-Dust Correlation at z>8:", "INFO")
    if uncover_rho is not None:
        print_status(f"  UNCOVER: ρ = {uncover_rho:.3f}", "INFO")
    else:
        print_status("  UNCOVER: unavailable in current comparison summary", "INFO")
    print_status(f"  CEERS:   ρ = {ceers_rho:.3f}", "INFO")
    
    # Check if results are consistent
    p_rep = results.get('mass_dust', {}).get('p')
    if ceers_rho > 0.3 and p_rep is not None and p_rep < 0.05:
        print_status(f"\n✓ REPLICATION SUCCESSFUL", "INFO")
        print_status(f"  The z>8 dust anomaly is confirmed in independent CEERS data.", "INFO")
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
