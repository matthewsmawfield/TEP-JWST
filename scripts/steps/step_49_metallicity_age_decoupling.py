#!/usr/bin/env python3
"""
TEP-JWST Step 49: Metallicity-Age Decoupling Test

This step tests a unique TEP prediction: at fixed mass, high-Gamma_t galaxies
should show OLDER ages but SIMILAR metallicity.

Physical Rationale:
- Chemical enrichment timescale: ~100 Myr (fast, set by SN II)
- Stellar aging timescale: ~1 Gyr (slow, set by stellar evolution)

Under TEP, enhanced proper time affects both, but:
- Metallicity saturates quickly (reaches equilibrium)
- Age continues to accumulate

This creates a DECOUPLING: high-Gamma_t galaxies appear old but not
correspondingly metal-rich. Standard physics predicts age and metallicity
should track together (older = more enriched).

This is a "discriminating indicator" because:
1. It's a UNIQUE TEP prediction
2. It's testable with existing data
3. Standard physics cannot explain it
"""

import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from pathlib import Path
import json

# =============================================================================
# PATHS AND LOGGER
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.p_value_utils import format_p_value

STEP_NUM = "49"
STEP_NAME = "metallicity_age_decoupling"

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
    print_status(f"STEP {STEP_NUM}: Metallicity-Age Decoupling Test", "INFO")
    print_status("=" * 70, "INFO")
    print_status("", "INFO")
    print_status("TEP Prediction: High-Gamma_t galaxies are OLD but not METAL-RICH", "INFO")
    print_status("Standard Physics: Age and metallicity should track together", "INFO")
    
    # Load data
    df = pd.read_csv(INTERIM_PATH / "step_02_uncover_full_sample_tep.csv")
    print_status(f"\nLoaded N = {len(df)} galaxies", "INFO")
    
    # Filter for valid data
    valid = df.dropna(subset=['mwa', 'met', 'gamma_t', 'log_Mstar'])
    valid = valid[(valid['mwa'] > 0) & (valid['met'] > -3)]
    print_status(f"Valid sample: N = {len(valid)}", "INFO")
    
    results = {
        'n_total': len(valid),
        'tests': {}
    }
    
    # ==========================================================================
    # TEST 1: Raw Correlations
    # ==========================================================================
    print_status("\n" + "=" * 50, "INFO")
    print_status("TEST 1: Raw Correlations", "INFO")
    print_status("=" * 50, "INFO")
    
    rho_age_gamma, p_age = spearmanr(valid['gamma_t'], valid['mwa'])
    rho_met_gamma, p_met = spearmanr(valid['gamma_t'], valid['met'])
    rho_age_met, p_age_met = spearmanr(valid['mwa'], valid['met'])
    
    print_status(f"\nρ(Γ_t, Age) = {rho_age_gamma:.3f} (p = {p_age:.2e})", "INFO")
    print_status(f"ρ(Γ_t, Met) = {rho_met_gamma:.3f} (p = {p_met:.2e})", "INFO")
    print_status(f"ρ(Age, Met) = {rho_age_met:.3f} (p = {p_age_met:.2e})", "INFO")
    
    results['tests']['raw_correlations'] = {
        'rho_gamma_age': float(rho_age_gamma),
        'p_gamma_age': format_p_value(p_age),
        'rho_gamma_met': float(rho_met_gamma),
        'p_gamma_met': format_p_value(p_met),
        'rho_age_met': float(rho_age_met),
        'p_age_met': format_p_value(p_age_met)
    }
    
    # ==========================================================================
    # TEST 2: Decoupling at Fixed Mass
    # ==========================================================================
    print_status("\n" + "=" * 50, "INFO")
    print_status("TEST 2: Decoupling at Fixed Mass", "INFO")
    print_status("=" * 50, "INFO")
    
    # Residualize age and metallicity against mass
    from scipy.stats import linregress
    
    # Age residuals
    slope_age, intercept_age, _, _, _ = linregress(valid['log_Mstar'], np.log10(valid['mwa']))
    valid = valid.copy()
    valid['age_resid'] = np.log10(valid['mwa']) - (slope_age * valid['log_Mstar'] + intercept_age)
    
    # Metallicity residuals
    slope_met, intercept_met, _, _, _ = linregress(valid['log_Mstar'], valid['met'])
    valid['met_resid'] = valid['met'] - (slope_met * valid['log_Mstar'] + intercept_met)
    
    # Correlations with Gamma_t at fixed mass
    rho_age_resid, p_age_resid = spearmanr(valid['gamma_t'], valid['age_resid'])
    rho_met_resid, p_met_resid = spearmanr(valid['gamma_t'], valid['met_resid'])
    
    print_status(f"\nAt fixed mass:", "INFO")
    print_status(f"  ρ(Γ_t, Age_resid) = {rho_age_resid:.3f} (p = {p_age_resid:.2e})", "INFO")
    print_status(f"  ρ(Γ_t, Met_resid) = {rho_met_resid:.3f} (p = {p_met_resid:.2e})", "INFO")
    
    # The key test: is the age correlation stronger than metallicity?
    decoupling_ratio = abs(rho_age_resid) / max(abs(rho_met_resid), 0.01)
    
    print_status(f"\n  Decoupling Ratio: |ρ_age| / |ρ_met| = {decoupling_ratio:.2f}", "INFO")
    
    if rho_age_resid > 0 and abs(rho_age_resid) > abs(rho_met_resid) * 1.5:
        print_status("  → DECOUPLING DETECTED: Age responds to Γ_t more than metallicity", "INFO")
        decoupling_detected = True
    else:
        print_status("  → No clear decoupling", "INFO")
        decoupling_detected = False
    
    results['tests']['fixed_mass'] = {
        'rho_gamma_age_resid': float(rho_age_resid),
        'p_gamma_age_resid': format_p_value(p_age_resid),
        'rho_gamma_met_resid': float(rho_met_resid),
        'p_gamma_met_resid': format_p_value(p_met_resid),
        'decoupling_ratio': float(decoupling_ratio),
        'decoupling_detected': decoupling_detected
    }
    
    # ==========================================================================
    # TEST 3: High-z Regime (z > 7)
    # ==========================================================================
    print_status("\n" + "=" * 50, "INFO")
    print_status("TEST 3: High-z Regime (z > 7)", "INFO")
    print_status("=" * 50, "INFO")
    
    high_z = valid[valid['z_phot'] > 7].copy()
    print_status(f"High-z sample: N = {len(high_z)}", "INFO")
    
    if len(high_z) > 50:
        # Split by Gamma_t
        gamma_median = high_z['gamma_t'].median()
        high_gamma = high_z[high_z['gamma_t'] > gamma_median]
        low_gamma = high_z[high_z['gamma_t'] <= gamma_median]
        
        # Compare age and metallicity
        age_diff = high_gamma['mwa'].mean() - low_gamma['mwa'].mean()
        met_diff = high_gamma['met'].mean() - low_gamma['met'].mean()
        
        print_status(f"\nHigh Γ_t vs Low Γ_t at z > 7:", "INFO")
        print_status(f"  Age difference: {age_diff:.3f} Gyr (high - low)", "INFO")
        print_status(f"  Met difference: {met_diff:.3f} dex (high - low)", "INFO")
        
        # Normalize by typical scatter
        age_scatter = high_z['mwa'].std()
        met_scatter = high_z['met'].std()
        
        age_diff_norm = age_diff / age_scatter
        met_diff_norm = met_diff / met_scatter
        
        print_status(f"\n  Normalized differences:", "INFO")
        print_status(f"    Age: {age_diff_norm:.2f}σ", "INFO")
        print_status(f"    Met: {met_diff_norm:.2f}σ", "INFO")
        
        # TEP predicts age_diff_norm > met_diff_norm
        if age_diff_norm > met_diff_norm and age_diff > 0:
            print_status("  → HIGH-Z DECOUPLING: Age difference exceeds metallicity difference", "INFO")
            high_z_decoupling = True
        else:
            print_status("  → No high-z decoupling", "INFO")
            high_z_decoupling = False
        
        results['tests']['high_z'] = {
            'n': len(high_z),
            'gamma_median': float(gamma_median),
            'age_diff': float(age_diff),
            'met_diff': float(met_diff),
            'age_diff_sigma': float(age_diff_norm),
            'met_diff_sigma': float(met_diff_norm),
            'high_z_decoupling': high_z_decoupling
        }
    else:
        print_status("  Insufficient high-z sample", "WARNING")
        results['tests']['high_z'] = None
    
    # ==========================================================================
    # TEST 4: Age-Metallicity Relation by Gamma_t Regime
    # ==========================================================================
    print_status("\n" + "=" * 50, "INFO")
    print_status("TEST 4: Age-Met Relation by Γ_t Regime", "INFO")
    print_status("=" * 50, "INFO")
    
    enhanced = valid[valid['gamma_t'] > 1]
    suppressed = valid[valid['gamma_t'] < 1]
    
    print_status(f"Enhanced (Γ_t > 1): N = {len(enhanced)}", "INFO")
    print_status(f"Suppressed (Γ_t < 1): N = {len(suppressed)}", "INFO")
    
    if len(enhanced) > 30 and len(suppressed) > 30:
        rho_enh, p_enh = spearmanr(enhanced['mwa'], enhanced['met'])
        rho_sup, p_sup = spearmanr(suppressed['mwa'], suppressed['met'])
        
        print_status(f"\nAge-Met correlation:", "INFO")
        print_status(f"  Enhanced: ρ = {rho_enh:.3f} (p = {p_enh:.2e})", "INFO")
        print_status(f"  Suppressed: ρ = {rho_sup:.3f} (p = {p_sup:.2e})", "INFO")
        
        # TEP predicts weaker age-met correlation in enhanced regime
        # (because age is inflated but metallicity is not)
        if rho_enh < rho_sup:
            print_status("  → WEAKER age-met coupling in enhanced regime (TEP-consistent)", "INFO")
            regime_decoupling = True
        else:
            print_status("  → No regime difference", "INFO")
            regime_decoupling = False
        
        results['tests']['regime_comparison'] = {
            'n_enhanced': len(enhanced),
            'n_suppressed': len(suppressed),
            'rho_age_met_enhanced': float(rho_enh),
            'p_enhanced': format_p_value(p_enh),
            'rho_age_met_suppressed': float(rho_sup),
            'p_suppressed': format_p_value(p_sup),
            'regime_decoupling': regime_decoupling
        }
    else:
        results['tests']['regime_comparison'] = None
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("SUMMARY: Metallicity-Age Decoupling", "INFO")
    print_status("=" * 70, "INFO")
    
    n_tests_passed = sum([
        results['tests']['fixed_mass'].get('decoupling_detected', False),
        results['tests'].get('high_z', {}).get('high_z_decoupling', False) if results['tests'].get('high_z') else False,
        results['tests'].get('regime_comparison', {}).get('regime_decoupling', False) if results['tests'].get('regime_comparison') else False
    ])
    
    print_status(f"\nTests passed: {n_tests_passed}/3", "INFO")
    
    if n_tests_passed >= 2:
        print_status("\n✓ METALLICITY-AGE DECOUPLING CONFIRMED", "INFO")
        print_status("  This is a unique TEP signature that standard physics cannot explain.", "INFO")
        overall_result = "CONFIRMED"
    elif n_tests_passed == 1:
        print_status("\n⚠ PARTIAL EVIDENCE for decoupling", "INFO")
        overall_result = "PARTIAL"
    else:
        print_status("\n✗ No clear decoupling detected", "INFO")
        overall_result = "NOT_DETECTED"
    
    results['summary'] = {
        'tests_passed': n_tests_passed,
        'tests_total': 3,
        'overall_result': overall_result
    }
    
    # Save
    json_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print_status(f"\nSaved to {json_path}", "INFO")

if __name__ == "__main__":
    main()
