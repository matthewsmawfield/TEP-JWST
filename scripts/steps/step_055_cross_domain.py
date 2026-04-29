#!/usr/bin/env python3
"""
TEP-JWST Step 55: Cross-Domain Consistency

This step tests the most powerful evidence for TEP: consistency across
completely independent domains. If TEP is real, the SAME alpha parameter
should work across:

1. JWST high-z galaxies (this paper)
2. SN Ia host galaxies (TEP-H0, kappa = 9.6e5 mag)
3. Globular cluster pulsars (TEP-COS)

This is the ultimate test: a single parameter explaining phenomena
across 15 orders of magnitude in mass.
"""

import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, linregress  # Rank correlation and OLS regression
from scipy.optimize import minimize_scalar  # 1-D optimisation for cross-domain α fit
from pathlib import Path
import json

# =============================================================================
# PATHS AND LOGGER
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging (severity levels: DEBUG/INFO/WARNING/ERROR/SUCCESS)
from scripts.utils.p_value_utils import format_p_value  # Safe p-value formatting (prevents floating-point underflow at p < 1e-300)
from scripts.utils.tep_model import KAPPA_GAL, compute_gamma_t as tep_gamma  # TEP model: KAPPA_GAL=9.6e5 mag from Cepheids, Gamma_t formula

STEP_NUM = "055"  # Pipeline step number (sequential 001-176)
STEP_NAME = "cross_domain"  # Cross-domain consistency: tests if kappa=9.6e5 mag works across JWST, SN Ia, and globular clusters (15 orders of magnitude)

INTERIM_PATH = PROJECT_ROOT / "results" / "interim"  # Pre-processed intermediate products (CSV format for step-to-step data flow)
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"  # JSON output directory (machine-readable statistical results)
LOGS_PATH = PROJECT_ROOT / "logs"  # Log directory (one plain-text log per step for debugging traceability)

for p in [OUTPUT_PATH, LOGS_PATH]:
    p.mkdir(parents=True, exist_ok=True)  # Create directory tree if missing; exist_ok=True allows safe re-runs

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")  # Step-specific logger (isolated per-step logging for traceability)
set_step_logger(logger)  # Register as global step logger so print_status() routes to this step's log

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def gamma_t_with_alpha(log_Mh, z, alpha):
    """Compute Gamma_t with variable alpha."""
    return tep_gamma(log_Mh, z, kappa=alpha)

def main():
    print_status("=" * 70, "INFO")
    print_status(f"STEP {STEP_NUM}: Cross-Domain Consistency", "INFO")
    print_status("=" * 70, "INFO")
    print_status("\nTesting if the SAME alpha works across all domains...\n", "INFO")
    
    # Load data
    df = pd.read_csv(INTERIM_PATH / "step_002_uncover_full_sample_tep.csv")
    print_status(f"Loaded N = {len(df)} galaxies", "INFO")
    
    results = {
        'n_total': len(df),
        'cross_domain': {}
    }
    
    # ==========================================================================
    # TEST 1: Optimal Alpha from JWST Data
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 1: Optimal Alpha from JWST Data", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Evaluating κ_gal transfer; Spearman rank tests cannot recover κ amplitude.\n", "INFO")
    
    high_z = df[df['z_phot'] > 8].dropna(subset=['log_Mh', 'dust', 'z_phot'])
    
    if len(high_z) > 50:
        def neg_correlation(alpha):
            gamma_t = gamma_t_with_alpha(high_z['log_Mh'].values, high_z['z_phot'].values, alpha)
            rho, _ = spearmanr(gamma_t, high_z['dust'])
            return -rho  # Negative because we minimize
        
        # Search in the correct κ units for diagnostic purposes only.
        # Spearman correlations are rank based, and Gamma_t is monotonic in κ
        # for this sample, so this objective cannot identify the κ amplitude.
        result = minimize_scalar(neg_correlation, bounds=(1.0e5, 2.0e6), method='bounded')
        optimal_alpha = result.x
        optimal_rho = -result.fun
        
        print_status(f"Diagnostic κ at optimizer boundary/interior: {optimal_alpha:.3e}", "INFO")
        print_status(f"Maximum correlation: ρ = {optimal_rho:.3f}", "INFO")
        print_status("κ amplitude recovery is marked reference-only because the rank objective is nearly invariant under monotonic rescaling.", "WARNING")
        
        # Compare to TEP-H0 alpha
        tep_h0_alpha = KAPPA_GAL  # Legacy 9.6e5 maps to kappa = 9.6e5 mag
        gamma_h0 = gamma_t_with_alpha(high_z['log_Mh'].values, high_z['z_phot'].values, tep_h0_alpha)
        rho_h0, p_h0 = spearmanr(gamma_h0, high_z['dust'])
        
        print_status(f"\nWith Cepheid prior κ_gal ({tep_h0_alpha:.3e} mag): ρ = {rho_h0:.3f}", "INFO")
        
        results['cross_domain']['alpha_optimization'] = {
            'diagnostic_kappa_optimizer': float(optimal_alpha),
            'optimal_rho': float(optimal_rho),
            'tep_h0_kappa_gal': tep_h0_alpha,
            'tep_h0_rho': float(rho_h0),
            'valid_kappa_recovery': False,
            'note': 'Reference-only: Spearman rank objective is insensitive to monotonic κ rescaling; use step_101 for scale-aware κ recovery.'
        }
    
    # ==========================================================================
    # TEST 2: Alpha Sensitivity
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 2: Alpha Sensitivity", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Testing correlation strength across alpha values.\n", "INFO")
    
    if len(high_z) > 50:
        alphas = KAPPA_GAL * np.linspace(0.5, 1.5, 17)
        alpha_results = []
        
        for alpha in alphas:
            gamma_t = gamma_t_with_alpha(high_z['log_Mh'].values, high_z['z_phot'].values, alpha)
            rho, p = spearmanr(gamma_t, high_z['dust'])
            alpha_results.append({
                'alpha': float(alpha),
                'rho': float(rho),
                'p': format_p_value(p)
            })
            marker = "◆" if abs(alpha - KAPPA_GAL) / KAPPA_GAL < 0.01 else ""
            print_status(f"κ = {alpha:.2e}: ρ = {rho:.3f} {marker}", "INFO")
        
        results['cross_domain']['alpha_sensitivity'] = alpha_results
    
    # ==========================================================================
    # TEST 3: Redshift-Dependent Alpha
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 3: Redshift-Dependent Alpha", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Testing if optimal alpha varies with redshift.\n", "INFO")
    
    valid = df.dropna(subset=['log_Mh', 'dust', 'z_phot'])
    
    z_bins = [(6, 7), (7, 8), (8, 9), (9, 12)]
    z_alpha_results = []
    
    for z_lo, z_hi in z_bins:
        bin_data = valid[(valid['z_phot'] >= z_lo) & (valid['z_phot'] < z_hi)]
        if len(bin_data) > 30:
            def neg_corr(alpha):
                gamma_t = gamma_t_with_alpha(bin_data['log_Mh'].values, bin_data['z_phot'].values, alpha)
                rho, _ = spearmanr(gamma_t, bin_data['dust'])
                return -rho
            
            result = minimize_scalar(neg_corr, bounds=(1.0e5, 2.0e6), method='bounded')
            opt_alpha = result.x
            opt_rho = -result.fun
            
            z_alpha_results.append({
                'z_range': f'{z_lo}-{z_hi}',
                'n': int(len(bin_data)),
                'optimal_alpha': float(opt_alpha),
                'optimal_rho': float(opt_rho)
            })
            print_status(f"z = {z_lo}-{z_hi}: diagnostic κ = {opt_alpha:.2e}, ρ = {opt_rho:.3f}", "INFO")
    
    # Diagnostic spread only; do not interpret as κ consistency because the
    # rank objective is insensitive to the amplitude.
    if len(z_alpha_results) >= 3:
        alphas = [r['optimal_alpha'] for r in z_alpha_results]
        alpha_std = np.std(alphas)
        alpha_mean = np.mean(alphas)
        
        print_status(f"\nDiagnostic κ spread: {alpha_mean:.2e} ± {alpha_std:.2e} (reference-only)", "INFO")
    
    results['cross_domain']['z_dependent_alpha'] = z_alpha_results
    
    # ==========================================================================
    # TEST 4: Mass Scale Invariance
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 4: Mass Scale Invariance", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Testing if the effect works across different mass scales.\n", "INFO")
    
    high_z = df[df['z_phot'] > 8].dropna(subset=['log_Mh', 'dust', 'gamma_t'])
    
    if len(high_z) > 50:
        # Split by halo mass
        mass_bins = [(9, 10), (10, 11), (11, 13)]
        mass_results = []
        
        for m_lo, m_hi in mass_bins:
            bin_data = high_z[(high_z['log_Mh'] >= m_lo) & (high_z['log_Mh'] < m_hi)]
            if len(bin_data) > 10:
                rho, p = spearmanr(bin_data['gamma_t'], bin_data['dust'])
                mass_results.append({
                    'mass_range': f'{m_lo}-{m_hi}',
                    'n': int(len(bin_data)),
                    'rho': float(rho),
                    'p': format_p_value(p)
                })
                sig = "✓" if rho > 0.2 and p < 0.05 else ""
                print_status(f"log M_h = {m_lo}-{m_hi}: ρ = {rho:.3f} (p = {p:.2e}) {sig}", "INFO")
        
        results['cross_domain']['mass_scale'] = mass_results
    
    # ==========================================================================
    # TEST 5: Cross-Survey Alpha Consistency
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 5: Cross-Survey Rank Diagnostic", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Testing if CEERS gives the same optimal alpha.\n", "INFO")
    
    ceers_path = PROJECT_ROOT / "data" / "interim" / "ceers_z8_sample.csv"
    if ceers_path.exists():
        ceers = pd.read_csv(ceers_path)
        ceers_valid = ceers.dropna(subset=['log_Mstar', 'dust'])
        
        if len(ceers_valid) > 20:
            # Use M* as proxy for M_h (with offset)
            ceers_valid = ceers_valid.copy()
            ceers_valid['log_Mh_proxy'] = ceers_valid['log_Mstar'] + 1.5
            
            # Assume z ~ 9 for CEERS z8 sample
            z_ceers = 9.0
            
            def neg_corr_ceers(alpha):
                gamma_t = gamma_t_with_alpha(ceers_valid['log_Mh_proxy'].values, 
                                          np.full(len(ceers_valid), z_ceers), alpha)
                rho, _ = spearmanr(gamma_t, ceers_valid['dust'])
                return -rho
            
            result = minimize_scalar(neg_corr_ceers, bounds=(1.0e5, 2.0e6), method='bounded')
            ceers_alpha = result.x
            ceers_rho = -result.fun
            
            print_status(f"CEERS diagnostic κ: {ceers_alpha:.3e}", "INFO")
            print_status(f"CEERS correlation: ρ = {ceers_rho:.3f}", "INFO")
            print_status(f"UNCOVER diagnostic κ: {optimal_alpha:.3e}", "INFO")
            print_status("Difference is not interpreted as κ recovery because the objective is rank-invariant.", "INFO")
            
            results['cross_domain']['ceers_alpha'] = {
                'ceers_diagnostic_kappa': float(ceers_alpha),
                'ceers_rho': float(ceers_rho),
                'uncover_diagnostic_kappa': float(optimal_alpha),
                'valid_kappa_recovery': False
            }
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("SUMMARY: Cross-Domain Consistency", "INFO")
    print_status("=" * 70, "INFO")
    
    print_status(f"\nKey findings:", "INFO")
    print_status(f"  • Cepheid prior κ_gal: {KAPPA_GAL:.3e} mag", "INFO")
    print_status(f"  • JWST rank diagnostic κ: {optimal_alpha:.3e} (reference-only)", "INFO")
    print_status("  • Cross-domain κ amplitude recovery is not claimed from this rank test.", "INFO")
    
    results['summary'] = {
        'jwst_diagnostic_kappa': float(optimal_alpha),
        'tep_h0_kappa_gal': float(KAPPA_GAL),
        'valid_kappa_recovery': False,
        'consistent': None,
        'assessment': 'reference_only_rank_invariant; use step_101 for scale-aware κ recovery'
    }
    
    # Save
    json_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print_status(f"\nSaved to {json_path}", "INFO")

if __name__ == "__main__":
    main()
