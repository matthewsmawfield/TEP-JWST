#!/usr/bin/env python3
"""
TEP-JWST Step 37: Spectroscopic Validation

Validates TEP predictions using spectroscopically confirmed galaxies
from the UNCOVER DR4 zspec catalog.

Author: Matthew L. Smawfield
Date: January 2026
"""

import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.p_value_utils import format_p_value, safe_json_default
from scripts.utils.tep_model import compute_gamma_t as tep_gamma, stellar_to_halo_mass_behroozi_like

STEP_NUM = "37"
STEP_NAME = "spectroscopic_validation"

DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "uncover"
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"
LOGS_PATH = PROJECT_ROOT / "logs"

for p in [OUTPUT_PATH, LOGS_PATH]:
    p.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")
set_step_logger(logger)


def load_spectroscopic_catalog():
    """Load combined spectroscopic catalog."""
    from astropy.cosmology import Planck18 as cosmo
    
    spec_file = PROJECT_ROOT / "data" / "interim" / "combined_spectroscopic_catalog.csv"
    
    if not spec_file.exists():
        print_status(f"ERROR: Combined spectroscopic catalog not found at {spec_file}", "INFO")
        return None
    
    print_status(f"Loading spectroscopic catalog: {spec_file.name}", "INFO")
    df = pd.read_csv(spec_file)
    
    print_status(f"Total spectroscopic sources: {len(df)}", "INFO")
    
    # Rename columns if needed (already standardized in step 37b)
    # columns: id, ra, dec, z_spec, log_Mstar, mwa, dust, met, source_catalog
    
    # Filter valid data
    df = df.dropna(subset=['log_Mstar', 'mwa'])
    df = df[df['log_Mstar'] > 6]
    
    # Compute TEP quantities
    log_mh = stellar_to_halo_mass_behroozi_like(df['log_Mstar'].values, df['z_spec'].values)
    df['gamma_t'] = tep_gamma(log_mh, df['z_spec'].values)
    df['t_cosmic'] = cosmo.age(df['z_spec'].values).value
    df['age_ratio'] = df['mwa'] / df['t_cosmic']
    
    # Filter outliers in age_ratio
    df = df[df['age_ratio'] < 5.0] # Remove extreme outliers
    
    print_status(f"Valid sources for analysis: N = {len(df)}", "INFO")
    
    return df


def test_correlations(df, sample_name):
    """Test TEP correlations in spectroscopic sample."""
    print_status(f"\n{'='*60}", "INFO")
    print_status(f"SPECTROSCOPIC CORRELATIONS: {sample_name}", "INFO")
    print_status(f"{'='*60}", "INFO")
    
    results = {'sample_name': sample_name, 'n': len(df)}
    
    if len(df) < 10:
        print_status(f"Insufficient sample size (N = {len(df)})", "INFO")
        return results
    
    print_status(f"Sample size: N = {len(df)}", "INFO")
    print_status(f"Redshift range: z = {df['z_spec'].min():.2f} - {df['z_spec'].max():.2f}", "INFO")
    
    # Gamma_t vs age_ratio
    rho, p = stats.spearmanr(df['gamma_t'], df['age_ratio'])
    # Bootstrap 95% CI
    np.random.seed(42)
    boot_rhos = []
    for _ in range(2000):
        idx = np.random.choice(len(df), size=len(df), replace=True)
        br, _ = stats.spearmanr(df['gamma_t'].iloc[idx], df['age_ratio'].iloc[idx])
        boot_rhos.append(br)
    ci_lo, ci_hi = np.percentile(boot_rhos, [2.5, 97.5])
    print_status(f"\nrho(Gamma_t, age_ratio) = {rho:.3f} [{ci_lo:.3f}, {ci_hi:.3f}], p = {p:.2e}", "INFO")
    results['rho_gamma_age'] = float(rho)
    results['rho_gamma_age_ci'] = [float(ci_lo), float(ci_hi)]
    results['p_gamma_age'] = format_p_value(p)
    
    # Mass-dust correlation
    df_dust = df[df['dust'] > 0]
    if len(df_dust) > 10:
        rho_md, p_md = stats.spearmanr(df_dust['log_Mstar'], df_dust['dust'])
        print_status(f"rho(M*, dust) = {rho_md:.3f}, p = {p_md:.2e}, N = {len(df_dust)}", "INFO")
        results['rho_mass_dust'] = float(rho_md)
        results['p_mass_dust'] = format_p_value(p_md)
        results['n_dust'] = len(df_dust)
        
        # Gamma_t vs dust
        rho_gd, p_gd = stats.spearmanr(df_dust['gamma_t'], df_dust['dust'])
        print_status(f"rho(Gamma_t, dust) = {rho_gd:.3f}, p = {p_gd:.2e}", "INFO")
        results['rho_gamma_dust'] = float(rho_gd)
        results['p_gamma_dust'] = format_p_value(p_gd)
    
    return results


def analyze_by_redshift(df):
    """Analyze correlations by redshift bin."""
    print_status(f"\n{'='*60}", "INFO")
    print_status("CORRELATIONS BY REDSHIFT BIN", "INFO")
    print_status(f"{'='*60}", "INFO")
    
    bins = [(4, 6), (6, 8), (8, 12)]
    bin_results = []
    
    for lo, hi in bins:
        mask = (df['z_spec'] >= lo) & (df['z_spec'] < hi)
        df_bin = df[mask]
        
        if len(df_bin) > 10:
            rho, p = stats.spearmanr(df_bin['gamma_t'], df_bin['age_ratio'])
            print_status(f"z = {lo}-{hi}: N = {len(df_bin):3d}, rho(Gamma_t, age_ratio) = {rho:+.3f}, p = {p:.4f}", "INFO")
            bin_results.append({
                'z_lo': lo,
                'z_hi': hi,
                'n': len(df_bin),
                'rho': float(rho),
                'p': format_p_value(p)
            })
        else:
            print_status(f"z = {lo}-{hi}: N = {len(df_bin):3d} (insufficient)", "INFO")
    
    return bin_results


def main():
    print_status("=" * 70, "INFO")
    print_status("STEP 37: Spectroscopic Validation", "INFO")
    print_status("=" * 70, "INFO")
    print_status("", "INFO")
    print_status("Validating TEP predictions using spectroscopically confirmed galaxies", "INFO")
    print_status("", "INFO")
    
    results = {}
    
    # Load spectroscopic catalog
    df = load_spectroscopic_catalog()
    if df is None:
        print_status("ERROR: Could not load spectroscopic data", "INFO")
        return
    
    results['total_spec'] = len(df)
    
    # Full sample analysis
    full_results = test_correlations(df, "Full Sample (z > 4)")
    results['full_sample'] = full_results
    
    # z > 8 sample
    df_z8 = df[df['z_spec'] >= 8].copy()
    z8_results = test_correlations(df_z8, "z > 8 Sample")
    results['z8_sample'] = z8_results
    
    # By redshift bin
    bin_results = analyze_by_redshift(df)
    results['redshift_bins'] = bin_results
    
    # Summary
    print_status(f"\n{'='*70}", "INFO")
    print_status("SPECTROSCOPIC VALIDATION SUMMARY", "INFO")
    print_status(f"{'='*70}", "INFO")
    
    if full_results.get('rho_gamma_age') is not None:
        rho = full_results['rho_gamma_age']
        p = full_results['p_gamma_age']
        
        print_status(f"\nFull sample (N = {full_results['n']}):", "INFO")
        print_status(f"  rho(Gamma_t, age_ratio) = {rho:+.3f}, p = {p:.4f}", "INFO")
        
        if p is not None and p < 0.05:
            print_status(f"\n✓ SPECTROSCOPIC VALIDATION SUCCESSFUL", "INFO")
            print_status(f"  TEP prediction confirmed in spectroscopically validated sample", "INFO")
            results['validation_status'] = 'confirmed'
        else:
            print_status(f"\n⚠ SPECTROSCOPIC VALIDATION INCONCLUSIVE", "INFO")
            print_status(f"  Correlation not significant at p < 0.05", "INFO")
            results['validation_status'] = 'inconclusive'
    
    # Save results
    output_file = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=safe_json_default)
    
    print_status(f"\nResults saved to: {output_file}", "INFO")
    print_status(f"Step {STEP_NUM} complete.", "INFO")
    
    return results


if __name__ == "__main__":
    main()
