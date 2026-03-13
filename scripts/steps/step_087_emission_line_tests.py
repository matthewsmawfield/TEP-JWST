#!/usr/bin/env python3
"""
TEP-JWST Step 87: Emission Line and SED Tests

Based on latest JWST findings:

1. EMISSION LINE EQUIVALENT WIDTH: Strong emission lines at high-z.
   TEP predicts: High-Gamma_t galaxies have more evolved populations.

2. SED FITTING TENSION: Photometric vs spectroscopic discrepancies.
   TEP predicts: Isochrony violation causes SED fitting failures.

3. STAR FORMATION RATE INDICATORS: Different SFR indicators disagree.
   TEP predicts: Time dilation affects different indicators differently.

4. STELLAR POPULATION AGE: Ages exceed cosmic time.
   TEP predicts: Effective age = cosmic age × Gamma_t.
"""

import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from pathlib import Path
import json

# =============================================================================
# PATHS AND LOGGER
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging (severity levels: DEBUG/INFO/WARNING/ERROR/SUCCESS)
from scripts.utils.p_value_utils import format_p_value  # Safe p-value formatting (prevents floating-point underflow at p < 1e-300)

STEP_NUM = "087"  # Pipeline step number (sequential 001-176)
STEP_NAME = "emission_line_tests"  # Emission line tests: SFR indicator discrepancies (sfr10 vs sfr100), SED fitting tension, stellar population ages exceeding cosmic time at z>8

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

def main():
    print_status("=" * 70, "INFO")
    print_status(f"STEP {STEP_NUM}: Emission Line and SED Tests", "INFO")
    print_status("=" * 70, "INFO")
    
    # Load data
    df = pd.read_csv(INTERIM_PATH / "step_002_uncover_full_sample_tep.csv")
    print_status(f"\nLoaded N = {len(df)} galaxies", "INFO")
    
    results = {
        'n_total': int(len(df)),
        'emission_sed': {}
    }
    
    # ==========================================================================
    # TEST 1: SFR Indicator Discrepancy
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 1: SFR Indicator Discrepancy", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Different SFR timescales should show different TEP effects.\n", "INFO")
    
    valid = df[df['z_phot'] > 8].dropna(subset=['sfr10', 'sfr100', 'gamma_t'])
    valid = valid[(valid['sfr10'] > 0) & (valid['sfr100'] > 0)].copy()
    
    if len(valid) > 50:
        # SFR10 (recent) vs SFR100 (longer timescale)
        valid['sfr_ratio'] = valid['sfr10'] / valid['sfr100']
        
        rho, p = spearmanr(valid['gamma_t'], valid['sfr_ratio'])
        print_status(f"ρ(Γ_t, SFR10/SFR100) = {rho:.3f} (p = {p:.2e})", "INFO")
        
        # High Gamma_t should have lower ratio (more evolved, declining SFR)
        if rho < 0:
            print_status("✓ High-Γ_t galaxies have declining SFR (more evolved)", "INFO")
        
        results['emission_sed']['sfr_discrepancy'] = {
            'rho': float(rho),
            'p': format_p_value(p)
        }
    
    # ==========================================================================
    # TEST 2: Chi2 vs Gamma_t (SED Fitting Quality)
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 2: SED Fitting Quality", "INFO")
    print_status("=" * 70, "INFO")
    print_status("TEP predicts: High-Gamma_t → isochrony violation → poor SED fits.\n", "INFO")
    
    valid = df[df['z_phot'] > 8].dropna(subset=['chi2', 'gamma_t'])
    
    if len(valid) > 50:
        rho, p = spearmanr(valid['gamma_t'], valid['chi2'])
        print_status(f"ρ(Γ_t, χ²) = {rho:.3f} (p = {p:.2e})", "INFO")
        
        # Binned analysis
        gamma_bins = [0, 0.5, 1, 2, 5, 100]
        print_status("\nBinned χ² by Γ_t:", "INFO")
        for i in range(len(gamma_bins) - 1):
            bin_data = valid[(valid['gamma_t'] >= gamma_bins[i]) & (valid['gamma_t'] < gamma_bins[i+1])]
            if len(bin_data) > 5:
                mean_chi2 = bin_data['chi2'].mean()
                print_status(f"  Γ_t = {gamma_bins[i]}-{gamma_bins[i+1]}: <χ²> = {mean_chi2:.1f} (N = {len(bin_data)})", "INFO")
        
        results['emission_sed']['sed_quality'] = {
            'rho': float(rho),
            'p': format_p_value(p)
        }
    
    # ==========================================================================
    # TEST 3: Age Paradox Resolution
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 3: Age Paradox Resolution", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Galaxies with age > cosmic age should have high Gamma_t.\n", "INFO")
    
    valid = df[df['z_phot'] > 8].dropna(subset=['age_ratio', 'gamma_t'])
    
    if len(valid) > 50:
        # Age ratio > 1 means age > cosmic age (anomalous under standard physics)
        anomalous = valid[valid['age_ratio'] > 0.5]
        possible = valid[valid['age_ratio'] <= 0.5]
        
        if len(anomalous) > 3 and len(possible) > 10:
            mean_gamma_imp = anomalous['gamma_t'].mean()
            mean_gamma_pos = possible['gamma_t'].mean()
            
            print_status(f"'Anomalous' (age > 50% cosmic): N = {len(anomalous)}", "INFO")
            print_status(f"<Γ_t> anomalous: {mean_gamma_imp:.2f}", "INFO")
            print_status(f"<Γ_t> possible: {mean_gamma_pos:.2f}", "INFO")
            print_status(f"Ratio: {mean_gamma_imp/mean_gamma_pos:.1f}×", "INFO")
            
            # Check if TEP resolves the paradox
            # Under TEP, effective age = cosmic age × Gamma_t
            # So age_ratio / Gamma_t should be < 1
            anomalous = anomalous.copy()
            anomalous['corrected_ratio'] = anomalous['age_ratio'] / anomalous['gamma_t']
            n_resolved = (anomalous['corrected_ratio'] < 0.5).sum()
            
            print_status(f"\nTEP resolution: {n_resolved}/{len(anomalous)} paradoxes resolved", "INFO")
            
            results['emission_sed']['age_paradox'] = {
                'n_impossible': int(len(anomalous)),
                'mean_gamma_impossible': float(mean_gamma_imp),
                'mean_gamma_possible': float(mean_gamma_pos),
                'ratio': float(mean_gamma_imp / mean_gamma_pos),
                'n_resolved': int(n_resolved)
            }
    
    # ==========================================================================
    # TEST 4: Stellar Population Complexity
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 4: Stellar Population Complexity", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Complex stellar populations should have high Gamma_t.\n", "INFO")
    
    valid = df[df['z_phot'] > 8].dropna(subset=['chi2', 'dust', 'mwa', 'gamma_t'])
    valid = valid[valid['mwa'] > 0]
    
    if len(valid) > 50:
        # Define "complex" as high chi2 + high dust + old age
        chi2_thresh = valid['chi2'].quantile(0.75)
        dust_thresh = valid['dust'].quantile(0.75)
        age_thresh = valid['mwa'].quantile(0.75)
        
        complex_pop = valid[
            (valid['chi2'] > chi2_thresh) & 
            (valid['dust'] > dust_thresh) & 
            (valid['mwa'] > age_thresh)
        ]
        simple_pop = valid.drop(complex_pop.index)
        
        if len(complex_pop) > 3 and len(simple_pop) > 10:
            mean_gamma_complex = complex_pop['gamma_t'].mean()
            mean_gamma_simple = simple_pop['gamma_t'].mean()
            
            print_status(f"Complex populations: N = {len(complex_pop)}", "INFO")
            print_status(f"<Γ_t> complex: {mean_gamma_complex:.2f}", "INFO")
            print_status(f"<Γ_t> simple: {mean_gamma_simple:.2f}", "INFO")
            print_status(f"Ratio: {mean_gamma_complex/mean_gamma_simple:.1f}×", "INFO")
            
            if mean_gamma_complex > mean_gamma_simple * 2:
                print_status("✓ Complex stellar populations have elevated Γ_t", "INFO")
            
            results['emission_sed']['complex_pop'] = {
                'n_complex': int(len(complex_pop)),
                'mean_gamma_complex': float(mean_gamma_complex),
                'mean_gamma_simple': float(mean_gamma_simple),
                'ratio': float(mean_gamma_complex / mean_gamma_simple)
            }
    
    # ==========================================================================
    # TEST 5: Mass-Weighted Age vs Light-Weighted Age
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 5: Mass-Weighted vs Light-Weighted Age", "INFO")
    print_status("=" * 70, "INFO")
    print_status("TEP predicts: MWA should be more affected than LWA.\n", "INFO")
    
    # Check if we have both age indicators
    age_cols = [col for col in df.columns if 'age' in col.lower() or 'mwa' in col.lower()]
    print_status(f"Age columns found: {age_cols}", "INFO")
    
    valid = df[df['z_phot'] > 8].dropna(subset=['mwa', 'gamma_t'])
    valid = valid[valid['mwa'] > 0]
    
    if len(valid) > 50:
        rho, p = spearmanr(valid['gamma_t'], valid['mwa'])
        print_status(f"ρ(Γ_t, MWA) = {rho:.3f} (p = {p:.2e})", "INFO")
        
        results['emission_sed']['age_indicators'] = {
            'rho_mwa': float(rho),
            'p_mwa': format_p_value(p)
        }
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("SUMMARY: Emission Line and SED Tests", "INFO")
    print_status("=" * 70, "INFO")
    
    print_status("\nKey findings:", "INFO")
    
    for test_name, test_data in results['emission_sed'].items():
        if 'ratio' in test_data:
            print_status(f"  • {test_name}: {test_data['ratio']:.1f}× elevation", "INFO")
        elif 'rho' in test_data:
            print_status(f"  • {test_name}: ρ = {test_data['rho']:.3f}", "INFO")
    
    # Save
    json_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print_status(f"\nSaved to {json_path}", "INFO")

if __name__ == "__main__":
    main()
