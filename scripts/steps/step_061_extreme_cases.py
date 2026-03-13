#!/usr/bin/env python3
"""
TEP-JWST Step 61: Extreme Cases Analysis

Focus on the MOST EXTREME cases - these are the hardest to explain
with standard physics and should show the strongest TEP signatures.

1. THE OLDEST GALAXIES: Galaxies with age > 0.5 × t_cosmic
2. THE DUSTIEST GALAXIES: Galaxies with A_V > 1 at z > 9
3. THE MOST MASSIVE: Galaxies with log M* > 10 at z > 8
4. THE HIGHEST CHI2: Galaxies with worst SED fits
5. THE MULTI-EXTREME: Galaxies extreme in multiple properties
"""

import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
from pathlib import Path
import json

# =============================================================================
# PATHS AND LOGGER
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging (severity levels: DEBUG/INFO/WARNING/ERROR/SUCCESS)
from scripts.utils.p_value_utils import format_p_value  # Safe p-value formatting (prevents floating-point underflow at p < 1e-300)

STEP_NUM = "061"  # Pipeline step number (sequential 001-176)
STEP_NAME = "extreme_cases"  # Extreme cases: tests 5 anomalous galaxy populations (oldest, dustiest, most massive, highest chi², multi-extreme) for strongest TEP signatures

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
    print_status(f"STEP {STEP_NUM}: Extreme Cases Analysis", "INFO")
    print_status("=" * 70, "INFO")
    
    # Load data
    df = pd.read_csv(INTERIM_PATH / "step_002_uncover_full_sample_tep.csv")
    print_status(f"\nLoaded N = {len(df)} galaxies", "INFO")
    
    results = {
        'n_total': int(len(df)),
        'extreme_cases': {}
    }
    
    # ==========================================================================
    # EXTREME CASE 1: The Oldest Galaxies
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("EXTREME CASE 1: The Oldest Galaxies", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Galaxies with age_ratio > 0.5 (age > 50% of cosmic age)\n", "INFO")
    
    valid = df.dropna(subset=['age_ratio', 'gamma_t'])
    
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        extreme = valid[valid['age_ratio'] > thresh]
        normal = valid[valid['age_ratio'] <= thresh]
        
        if len(extreme) > 3:
            mean_gamma_ext = extreme['gamma_t'].mean()
            mean_gamma_norm = normal['gamma_t'].mean()
            ratio = mean_gamma_ext / mean_gamma_norm if mean_gamma_norm > 0 else 0
            
            print_status(f"age_ratio > {thresh}: N = {len(extreme)}, <Γ_t> = {mean_gamma_ext:.2f}, ratio = {ratio:.1f}×", "INFO")
    
    # The most extreme
    most_extreme = valid.nlargest(10, 'age_ratio')
    print_status(f"\nTop 10 oldest galaxies:", "INFO")
    for idx, row in most_extreme.iterrows():
        print_status(f"  ID {idx}: age_ratio = {row['age_ratio']:.3f}, Γ_t = {row['gamma_t']:.2f}, z = {row['z_phot']:.2f}", "INFO")
    
    results['extreme_cases']['oldest'] = {
        'n_extreme': int(len(valid[valid['age_ratio'] > 0.5])),
        'mean_gamma_extreme': float(valid[valid['age_ratio'] > 0.5]['gamma_t'].mean()) if len(valid[valid['age_ratio'] > 0.5]) > 0 else 0
    }
    
    # ==========================================================================
    # EXTREME CASE 2: The Dustiest Galaxies at z > 9
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("EXTREME CASE 2: The Dustiest Galaxies at z > 9", "INFO")
    print_status("=" * 70, "INFO")
    print_status("At z > 9, t_cosmic < 550 Myr. High dust is 'anomalous'.\n", "INFO")
    
    z9 = df[df['z_phot'] > 9].dropna(subset=['dust', 'gamma_t'])
    
    if len(z9) > 10:
        # Dusty galaxies
        dusty = z9[z9['dust'] > 0.5]
        not_dusty = z9[z9['dust'] <= 0.5]
        
        print_status(f"z > 9 sample: N = {len(z9)}", "INFO")
        print_status(f"Dusty (A_V > 0.5): N = {len(dusty)}", "INFO")
        
        if len(dusty) > 3 and len(not_dusty) > 3:
            mean_gamma_dusty = dusty['gamma_t'].mean()
            mean_gamma_not = not_dusty['gamma_t'].mean()
            
            print_status(f"<Γ_t> dusty: {mean_gamma_dusty:.2f}", "INFO")
            print_status(f"<Γ_t> not dusty: {mean_gamma_not:.2f}", "INFO")
            print_status(f"Ratio: {mean_gamma_dusty/mean_gamma_not:.1f}×", "INFO")
            
            results['extreme_cases']['dusty_z9'] = {
                'n_dusty': int(len(dusty)),
                'mean_gamma_dusty': float(mean_gamma_dusty),
                'mean_gamma_not': float(mean_gamma_not),
                'ratio': float(mean_gamma_dusty / mean_gamma_not)
            }
    
    # ==========================================================================
    # EXTREME CASE 3: The Most Massive at z > 8
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("EXTREME CASE 3: The Most Massive at z > 8", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Massive galaxies at high-z are 'anomalous' under standard physics.\n", "INFO")
    
    z8 = df[df['z_phot'] > 8].dropna(subset=['log_Mstar', 'gamma_t', 'dust'])
    
    if len(z8) > 20:
        # Most massive
        massive = z8[z8['log_Mstar'] > 9.5]
        not_massive = z8[z8['log_Mstar'] <= 9.5]
        
        print_status(f"z > 8 sample: N = {len(z8)}", "INFO")
        print_status(f"Massive (log M* > 9.5): N = {len(massive)}", "INFO")
        
        if len(massive) > 3:
            mean_gamma_massive = massive['gamma_t'].mean()
            mean_gamma_not = not_massive['gamma_t'].mean()
            
            print_status(f"<Γ_t> massive: {mean_gamma_massive:.2f}", "INFO")
            print_status(f"<Γ_t> not massive: {mean_gamma_not:.2f}", "INFO")
            print_status(f"Ratio: {mean_gamma_massive/mean_gamma_not:.1f}×", "INFO")
            
            # Also check dust
            mean_dust_massive = massive['dust'].mean()
            mean_dust_not = not_massive['dust'].mean()
            
            print_status(f"\n<Dust> massive: {mean_dust_massive:.3f}", "INFO")
            print_status(f"<Dust> not massive: {mean_dust_not:.3f}", "INFO")
            
            results['extreme_cases']['massive_z8'] = {
                'n_massive': int(len(massive)),
                'mean_gamma_massive': float(mean_gamma_massive),
                'mean_gamma_not': float(mean_gamma_not),
                'ratio': float(mean_gamma_massive / mean_gamma_not)
            }
    
    # ==========================================================================
    # EXTREME CASE 4: The Worst SED Fits
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("EXTREME CASE 4: The Worst SED Fits", "INFO")
    print_status("=" * 70, "INFO")
    print_status("High chi2 indicates isochrony violation (TEP signature).\n", "INFO")
    
    z8 = df[df['z_phot'] > 8].dropna(subset=['chi2', 'gamma_t'])
    
    if len(z8) > 20:
        # Worst fits (top 10%)
        chi2_thresh = z8['chi2'].quantile(0.9)
        worst_fits = z8[z8['chi2'] > chi2_thresh]
        good_fits = z8[z8['chi2'] <= chi2_thresh]
        
        print_status(f"z > 8 sample: N = {len(z8)}", "INFO")
        print_status(f"Worst fits (top 10%): N = {len(worst_fits)}", "INFO")
        
        if len(worst_fits) > 3:
            mean_gamma_worst = worst_fits['gamma_t'].mean()
            mean_gamma_good = good_fits['gamma_t'].mean()
            
            print_status(f"<Γ_t> worst fits: {mean_gamma_worst:.2f}", "INFO")
            print_status(f"<Γ_t> good fits: {mean_gamma_good:.2f}", "INFO")
            print_status(f"Ratio: {mean_gamma_worst/mean_gamma_good:.1f}×", "INFO")
            
            results['extreme_cases']['worst_fits'] = {
                'n_worst': int(len(worst_fits)),
                'mean_gamma_worst': float(mean_gamma_worst),
                'mean_gamma_good': float(mean_gamma_good),
                'ratio': float(mean_gamma_worst / mean_gamma_good)
            }
    
    # ==========================================================================
    # EXTREME CASE 5: Multi-Extreme Galaxies
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("EXTREME CASE 5: Multi-Extreme Galaxies", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Galaxies extreme in MULTIPLE properties simultaneously.\n", "INFO")
    
    valid = df[df['z_phot'] > 8].dropna(subset=['dust', 'mwa', 'chi2', 'log_Mstar', 'gamma_t'])
    
    if len(valid) > 50:
        # Define extreme thresholds (top 25%)
        dust_thresh = valid['dust'].quantile(0.75)
        age_thresh = valid['mwa'].quantile(0.75)
        chi2_thresh = valid['chi2'].quantile(0.75)
        mass_thresh = valid['log_Mstar'].quantile(0.75)
        
        valid = valid.copy()
        valid['n_extreme'] = (
            (valid['dust'] > dust_thresh).astype(int) +
            (valid['mwa'] > age_thresh).astype(int) +
            (valid['chi2'] > chi2_thresh).astype(int) +
            (valid['log_Mstar'] > mass_thresh).astype(int)
        )
        
        print_status("Mean Γ_t by number of extreme properties:", "INFO")
        for n in range(5):
            subset = valid[valid['n_extreme'] == n]
            if len(subset) > 0:
                mean_gamma = subset['gamma_t'].mean()
                print_status(f"  N_extreme = {n}: <Γ_t> = {mean_gamma:.2f} (N = {len(subset)})", "INFO")
        
        # Correlation
        rho, p = spearmanr(valid['n_extreme'], valid['gamma_t'])
        print_status(f"\nρ(N_extreme, Γ_t) = {rho:.3f} (p = {p:.2e})", "INFO")
        
        results['extreme_cases']['multi_extreme'] = {
            'rho': float(rho),
            'p': format_p_value(p)
        }
    
    # ==========================================================================
    # EXTREME CASE 6: The "Anomalous" Population
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("EXTREME CASE 6: The 'Anomalous' Population", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Galaxies that violate multiple standard physics constraints.\n", "INFO")
    
    valid = df[df['z_phot'] > 9].dropna(subset=['age_ratio', 'dust', 'gamma_t'])
    
    if len(valid) > 10:
        # "Anomalous" = age_ratio > 0.4 AND dust > 0.3
        anomalous = valid[(valid['age_ratio'] > 0.4) & (valid['dust'] > 0.3)]
        possible = valid.drop(anomalous.index)
        
        print_status(f"z > 9 sample: N = {len(valid)}", "INFO")
        print_status(f"'Anomalous' (age_ratio > 0.4 AND dust > 0.3): N = {len(anomalous)}", "INFO")
        
        if len(anomalous) > 0 and len(possible) > 0:
            mean_gamma_imp = anomalous['gamma_t'].mean()
            mean_gamma_pos = possible['gamma_t'].mean()
            
            print_status(f"<Γ_t> anomalous: {mean_gamma_imp:.2f}", "INFO")
            print_status(f"<Γ_t> possible: {mean_gamma_pos:.2f}", "INFO")
            print_status(f"Ratio: {mean_gamma_imp/mean_gamma_pos:.1f}×", "INFO")
            
            results['extreme_cases']['anomalous'] = {
                'n_impossible': int(len(anomalous)),
                'mean_gamma_impossible': float(mean_gamma_imp),
                'mean_gamma_possible': float(mean_gamma_pos),
                'ratio': float(mean_gamma_imp / mean_gamma_pos)
            }
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("SUMMARY: Extreme Cases Analysis", "INFO")
    print_status("=" * 70, "INFO")
    
    print_status("\nKey findings:", "INFO")
    
    for case_name, case_data in results['extreme_cases'].items():
        if 'ratio' in case_data:
            print_status(f"  • {case_name}: {case_data['ratio']:.1f}× elevation in Γ_t", "INFO")
    
    # Save
    json_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print_status(f"\nSaved to {json_path}", "INFO")

if __name__ == "__main__":
    main()
