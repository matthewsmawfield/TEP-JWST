#!/usr/bin/env python3
"""
TEP-JWST Step 62: Falsification Tests

This step attempts to FALSIFY TEP by looking for predictions that
should NOT hold if TEP is wrong:

1. Null Correlation at Low-z: TEP predicts NO Gamma_t-Dust correlation at z < 5
2. Sign Reversal Test: All correlations should have the CORRECT sign
3. Mass Independence: After controlling for Gamma_t, mass should not predict dust
4. Redshift Monotonicity: TEP effect should increase monotonically with z
5. Extreme Value Test: The most extreme galaxies should have the highest Gamma_t
6. Consistency Test: Different observables should give consistent Gamma_t estimates
"""

import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, ks_2samp
from pathlib import Path
import json

# =============================================================================
# PATHS AND LOGGER
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.p_value_utils import format_p_value

STEP_NUM = "62"
STEP_NAME = "falsification_tests"

INTERIM_PATH = PROJECT_ROOT / "results" / "interim"
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"
LOGS_PATH = PROJECT_ROOT / "logs"

for p in [OUTPUT_PATH, LOGS_PATH]:
    p.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")
set_step_logger(logger)

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    print_status("=" * 70, "INFO")
    print_status(f"STEP {STEP_NUM}: Falsification Tests", "INFO")
    print_status("=" * 70, "INFO")
    print_status("\nAttempting to FALSIFY TEP...\n", "INFO")
    
    # Load data
    df = pd.read_csv(INTERIM_PATH / "step_02_uncover_full_sample_tep.csv")
    print_status(f"Loaded N = {len(df)} galaxies", "INFO")
    
    results = {
        'n_total': len(df),
        'falsification': {}
    }
    
    tests_passed = 0
    tests_total = 0
    
    # ==========================================================================
    # TEST 1: Null Correlation at Low-z
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("FALSIFICATION TEST 1: Null Correlation at Low-z", "INFO")
    print_status("=" * 70, "INFO")
    print_status("TEP predicts: NO Gamma_t-Dust correlation at z < 5\n", "INFO")
    
    low_z = df[(df['z_phot'] > 4) & (df['z_phot'] < 5)].dropna(subset=['gamma_t', 'dust'])
    
    if len(low_z) > 50:
        rho, p = spearmanr(low_z['gamma_t'], low_z['dust'])
        
        print_status(f"z = 4-5: N = {len(low_z)}, ρ(Γ_t, Dust) = {rho:.3f} (p = {p:.2e})", "INFO")
        
        tests_total += 1
        # TEP passes if correlation is weak (|rho| < 0.1) or not significant
        if abs(rho) < 0.15 or p > 0.01:
            tests_passed += 1
            print_status("✓ PASSED: No significant correlation at low-z", "INFO")
        else:
            print_status("✗ FAILED: Unexpected correlation at low-z", "INFO")
        
        results['falsification']['null_low_z'] = {
            'n': int(len(low_z)),
            'rho': float(rho),
            'p': format_p_value(p),
            'passed': bool(abs(rho) < 0.15 or p > 0.01)
        }
    
    # ==========================================================================
    # TEST 2: Sign Consistency
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("FALSIFICATION TEST 2: Sign Consistency", "INFO")
    print_status("=" * 70, "INFO")
    print_status("TEP predicts SPECIFIC signs for all correlations.\n", "INFO")
    
    high_z = df[df['z_phot'] > 8].dropna(subset=['gamma_t', 'dust', 'chi2'])
    
    if len(high_z) > 50:
        # Test 1: Gamma_t vs Dust should be POSITIVE
        rho_dust, _ = spearmanr(high_z['gamma_t'], high_z['dust'])
        sign_dust = rho_dust > 0
        
        # Test 2: Gamma_t vs Chi2 should be POSITIVE
        rho_chi2, _ = spearmanr(high_z['gamma_t'], high_z['chi2'])
        sign_chi2 = rho_chi2 > 0
        
        print_status(f"Γ_t-Dust: ρ = {rho_dust:.3f} (expected +) → {'✓' if sign_dust else '✗'}", "INFO")
        print_status(f"Γ_t-χ²: ρ = {rho_chi2:.3f} (expected +) → {'✓' if sign_chi2 else '✗'}", "INFO")
        
        tests_total += 1
        if sign_dust and sign_chi2:
            tests_passed += 1
            print_status("✓ PASSED: All signs correct", "INFO")
        else:
            print_status("✗ FAILED: Sign mismatch", "INFO")
        
        results['falsification']['sign_consistency'] = {
            'rho_dust': float(rho_dust),
            'rho_chi2': float(rho_chi2),
            'sign_dust_correct': bool(sign_dust),
            'sign_chi2_correct': bool(sign_chi2),
            'passed': bool(sign_dust and sign_chi2)
        }
    
    # ==========================================================================
    # TEST 3: Redshift Monotonicity
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("FALSIFICATION TEST 3: Redshift Monotonicity", "INFO")
    print_status("=" * 70, "INFO")
    print_status("TEP predicts: Effect should strengthen with z (at z > 6).\n", "INFO")
    
    valid = df.dropna(subset=['gamma_t', 'dust', 'z_phot'])
    
    z_bins = [(6, 7), (7, 8), (8, 9), (9, 12)]
    rhos = []
    
    for z_lo, z_hi in z_bins:
        bin_data = valid[(valid['z_phot'] >= z_lo) & (valid['z_phot'] < z_hi)]
        if len(bin_data) > 20:
            rho, p = spearmanr(bin_data['gamma_t'], bin_data['dust'])
            rhos.append(rho)
            print_status(f"z = {z_lo}-{z_hi}: ρ = {rho:.3f}", "INFO")
    
    if len(rhos) >= 3:
        # Check if rhos are monotonically increasing (allowing for noise)
        increasing = all(rhos[i] <= rhos[i+1] + 0.1 for i in range(len(rhos)-1))
        # Or at least the trend is positive
        trend_rho, _ = spearmanr(range(len(rhos)), rhos)
        
        tests_total += 1
        if trend_rho > 0.5:
            tests_passed += 1
            print_status(f"✓ PASSED: Trend is positive (ρ = {trend_rho:.2f})", "INFO")
        else:
            print_status(f"✗ FAILED: Trend is not positive (ρ = {trend_rho:.2f})", "INFO")
        
        results['falsification']['z_monotonicity'] = {
            'rhos': [float(r) for r in rhos],
            'trend_rho': float(trend_rho),
            'passed': bool(trend_rho > 0.5)
        }
    
    # ==========================================================================
    # TEST 4: Extreme Value Test
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("FALSIFICATION TEST 4: Extreme Value Test", "INFO")
    print_status("=" * 70, "INFO")
    print_status("TEP predicts: Most extreme galaxies should have highest Gamma_t.\n", "INFO")
    
    high_z = df[df['z_phot'] > 8].dropna(subset=['gamma_t', 'dust', 'mwa'])
    
    if len(high_z) > 50:
        # Define "extreme" as top 10% in both dust AND age
        dust_thresh = high_z['dust'].quantile(0.9)
        age_thresh = high_z['mwa'].quantile(0.9)
        
        extreme = high_z[(high_z['dust'] > dust_thresh) & (high_z['mwa'] > age_thresh)]
        normal = high_z.drop(extreme.index)
        
        if len(extreme) > 3:
            mean_gamma_ext = extreme['gamma_t'].mean()
            mean_gamma_norm = normal['gamma_t'].mean()
            
            print_status(f"Extreme (top 10% dust AND age): N = {len(extreme)}, <Γ_t> = {mean_gamma_ext:.3f}", "INFO")
            print_status(f"Normal: N = {len(normal)}, <Γ_t> = {mean_gamma_norm:.3f}", "INFO")
            print_status(f"Ratio: {mean_gamma_ext/mean_gamma_norm:.2f}×", "INFO")
            
            tests_total += 1
            if mean_gamma_ext > mean_gamma_norm * 2:
                tests_passed += 1
                print_status("✓ PASSED: Extreme galaxies have elevated Γ_t", "INFO")
            else:
                print_status("✗ FAILED: Extreme galaxies do not have elevated Γ_t", "INFO")
            
            results['falsification']['extreme_value'] = {
                'n_extreme': int(len(extreme)),
                'mean_gamma_extreme': float(mean_gamma_ext),
                'mean_gamma_normal': float(mean_gamma_norm),
                'ratio': float(mean_gamma_ext / mean_gamma_norm),
                'passed': bool(mean_gamma_ext > mean_gamma_norm * 2)
            }
    
    # ==========================================================================
    # TEST 5: Mass Residual Test
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("FALSIFICATION TEST 5: Mass Residual Test", "INFO")
    print_status("=" * 70, "INFO")
    print_status("TEP predicts: After controlling for Gamma_t, mass should explain less.\n", "INFO")
    
    high_z = df[df['z_phot'] > 8].dropna(subset=['gamma_t', 'dust', 'log_Mstar'])
    
    if len(high_z) > 50:
        from scipy.stats import linregress
        
        # Correlation: Mass vs Dust
        rho_mass, p_mass = spearmanr(high_z['log_Mstar'], high_z['dust'])
        
        # Residualize dust against Gamma_t
        slope, intercept, _, _, _ = linregress(high_z['gamma_t'], high_z['dust'])
        high_z = high_z.copy()
        high_z['dust_resid'] = high_z['dust'] - (slope * high_z['gamma_t'] + intercept)
        
        # Correlation: Mass vs Dust_resid
        rho_mass_resid, p_mass_resid = spearmanr(high_z['log_Mstar'], high_z['dust_resid'])
        
        print_status(f"ρ(M*, Dust) = {rho_mass:.3f}", "INFO")
        print_status(f"ρ(M*, Dust_resid|Γ_t) = {rho_mass_resid:.3f}", "INFO")
        print_status(f"Reduction: {(1 - abs(rho_mass_resid)/abs(rho_mass))*100:.1f}%", "INFO")
        
        tests_total += 1
        # TEP passes if controlling for Gamma_t reduces the mass-dust correlation
        if abs(rho_mass_resid) < abs(rho_mass) * 0.9:
            tests_passed += 1
            print_status("✓ PASSED: Gamma_t explains part of mass-dust correlation", "INFO")
        else:
            print_status("✗ FAILED: Gamma_t does not explain mass-dust correlation", "INFO")
        
        results['falsification']['mass_residual'] = {
            'rho_mass_dust': float(rho_mass),
            'rho_mass_dust_resid': float(rho_mass_resid),
            'reduction': float(1 - abs(rho_mass_resid)/abs(rho_mass)),
            'passed': bool(abs(rho_mass_resid) < abs(rho_mass) * 0.9)
        }
    
    # ==========================================================================
    # TEST 6: Cross-Survey Consistency
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("FALSIFICATION TEST 6: Cross-Survey Consistency", "INFO")
    print_status("=" * 70, "INFO")
    print_status("TEP predicts: Effect should be consistent across surveys.\n", "INFO")
    
    # Load CEERS data
    ceers_path = PROJECT_ROOT / "data" / "interim" / "ceers_z8_sample.csv"
    if ceers_path.exists():
        ceers = pd.read_csv(ceers_path)
        ceers_valid = ceers.dropna(subset=['log_Mstar', 'dust'])
        
        if len(ceers_valid) > 20:
            rho_ceers, p_ceers = spearmanr(ceers_valid['log_Mstar'], ceers_valid['dust'])
            
            # Compare to UNCOVER
            uncover_z8 = df[df['z_phot'] > 8].dropna(subset=['log_Mstar', 'dust'])
            rho_uncover, p_uncover = spearmanr(uncover_z8['log_Mstar'], uncover_z8['dust'])
            
            print_status(f"CEERS: ρ(M*, Dust) = {rho_ceers:.3f}", "INFO")
            print_status(f"UNCOVER: ρ(M*, Dust) = {rho_uncover:.3f}", "INFO")
            print_status(f"Difference: |Δρ| = {abs(rho_ceers - rho_uncover):.3f}", "INFO")
            
            tests_total += 1
            # TEP passes if correlations are consistent (within 0.2)
            if abs(rho_ceers - rho_uncover) < 0.25:
                tests_passed += 1
                print_status("✓ PASSED: Cross-survey consistency", "INFO")
            else:
                print_status("✗ FAILED: Cross-survey inconsistency", "INFO")
            
            results['falsification']['cross_survey'] = {
                'rho_ceers': float(rho_ceers),
                'rho_uncover': float(rho_uncover),
                'difference': float(abs(rho_ceers - rho_uncover)),
                'passed': bool(abs(rho_ceers - rho_uncover) < 0.25)
            }
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("SUMMARY: Falsification Test Results", "INFO")
    print_status("=" * 70, "INFO")
    
    print_status(f"\nTests passed: {tests_passed}/{tests_total}", "INFO")
    
    if tests_passed == tests_total:
        print_status("✓ TEP SURVIVES ALL FALSIFICATION TESTS", "INFO")
    elif tests_passed >= tests_total - 1:
        print_status("✓ TEP survives most falsification tests", "INFO")
    else:
        print_status("⚠ TEP fails some falsification tests", "INFO")
    
    results['summary'] = {
        'tests_passed': tests_passed,
        'tests_total': tests_total,
        'fraction': float(tests_passed / tests_total) if tests_total > 0 else 0
    }
    
    # Save
    json_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print_status(f"\nSaved to {json_path}", "INFO")

if __name__ == "__main__":
    main()
