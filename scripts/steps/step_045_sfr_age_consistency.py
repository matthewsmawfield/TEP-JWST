#!/usr/bin/env python3
# Estimated runtime from last full canonical run (2026-03-09 15:52 UTC; full pipeline 32m18s): 0.7s.
"""
TEP-JWST Step 45: SFR-Age Consistency Test

This step tests a unique TEP prediction: the relationship between
star formation rate (SFR) and stellar age should BREAK DOWN for
high-Gamma_t galaxies.

Physical Rationale:
- Standard physics: SFR and age are tightly coupled
  - High SFR → young population (short timescale)
  - Low SFR → old population (long timescale)
  
- Under TEP: Age is inflated but SFR is not
  - High-Gamma_t galaxies appear OLD despite having HIGH SFR
  - This breaks the standard SFR-age anticorrelation

This is a "discriminating indicator" because:
1. It's a UNIQUE TEP prediction
2. SFR is measured independently of age (UV/IR vs SED fitting)
3. Standard physics cannot explain old galaxies with high SFR
"""

import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr  # Rank correlation
from pathlib import Path
import json

# =============================================================================
# PATHS AND LOGGER
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging (severity levels: DEBUG/INFO/WARNING/ERROR/SUCCESS)
from scripts.utils.p_value_utils import format_p_value  # Safe p-value formatting (prevents floating-point underflow at p < 1e-300)

STEP_NUM = "045"  # Pipeline step number (sequential 001-176)
STEP_NAME = "sfr_age_consistency"  # SFR-age consistency: tests TEP prediction that high-Gamma_t galaxies break standard SFR-age anticorrelation (discriminating indicator)

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
    print_status(f"STEP {STEP_NUM}: SFR-Age Consistency Test", "INFO")
    print_status("=" * 70, "INFO")
    print_status("", "INFO")
    print_status("Standard Physics: High SFR → Young age (anticorrelation)", "INFO")
    print_status("TEP Prediction: High-Γ_t galaxies break this relationship", "INFO")
    
    # Load data
    df = pd.read_csv(INTERIM_PATH / "step_002_uncover_full_sample_tep.csv")
    print_status(f"\nLoaded N = {len(df)} galaxies", "INFO")
    
    # Filter for valid data
    valid = df.dropna(subset=['mwa', 'ssfr100', 'gamma_t', 'log_Mstar'])
    valid = valid[(valid['mwa'] > 0) & (valid['ssfr100'] > 0)]
    print_status(f"Valid sample: N = {len(valid)}", "INFO")
    
    results = {
        'n_total': len(valid),
        'tests': {}
    }
    
    # ==========================================================================
    # TEST 1: Overall SFR-Age Correlation
    # ==========================================================================
    print_status("\n" + "=" * 50, "INFO")
    print_status("TEST 1: Overall SFR-Age Correlation", "INFO")
    print_status("=" * 50, "INFO")
    
    # Use log(sSFR) for better scaling
    valid = valid.copy()
    valid['log_ssfr'] = np.log10(valid['ssfr100'])
    valid['log_age'] = np.log10(valid['mwa'])
    
    rho_all, p_all = spearmanr(valid['log_ssfr'], valid['log_age'])
    
    print_status(f"\nρ(log sSFR, log Age) = {rho_all:.3f} (p = {p_all:.2e})", "INFO")
    print_status("  Expected: Strong negative (high sSFR → young)", "INFO")
    
    results['tests']['overall'] = {
        'rho': float(rho_all),
        'p': format_p_value(p_all)
    }
    
    # ==========================================================================
    # TEST 2: SFR-Age by Gamma_t Regime
    # ==========================================================================
    print_status("\n" + "=" * 50, "INFO")
    print_status("TEST 2: SFR-Age by Γ_t Regime", "INFO")
    print_status("=" * 50, "INFO")
    
    enhanced = valid[valid['gamma_t'] > 1]
    suppressed = valid[valid['gamma_t'] < 0.5]
    
    print_status(f"\nEnhanced (Γ_t > 1): N = {len(enhanced)}", "INFO")
    print_status(f"Suppressed (Γ_t < 0.5): N = {len(suppressed)}", "INFO")
    
    if len(enhanced) > 30 and len(suppressed) > 30:
        rho_enh, p_enh = spearmanr(enhanced['log_ssfr'], enhanced['log_age'])
        rho_sup, p_sup = spearmanr(suppressed['log_ssfr'], suppressed['log_age'])
        
        print_status(f"\nSFR-Age correlation:", "INFO")
        print_status(f"  Enhanced: ρ = {rho_enh:.3f} (p = {p_enh:.2e})", "INFO")
        print_status(f"  Suppressed: ρ = {rho_sup:.3f} (p = {p_sup:.2e})", "INFO")
        
        # TEP predicts WEAKER anticorrelation in enhanced regime
        if abs(rho_enh) < abs(rho_sup):
            print_status("\n✓ WEAKER SFR-Age coupling in enhanced regime", "INFO")
            print_status("  TEP explanation: Age is inflated independently of SFR", "INFO")
            regime_decoupling = True
        else:
            print_status("\n⚠ No regime difference detected", "INFO")
            regime_decoupling = False
        
        results['tests']['regime_comparison'] = {
            'n_enhanced': len(enhanced),
            'n_suppressed': len(suppressed),
            'rho_enhanced': float(rho_enh),
            'p_enhanced': format_p_value(p_enh),
            'rho_suppressed': float(rho_sup),
            'p_suppressed': format_p_value(p_sup),
            'regime_decoupling': regime_decoupling
        }
    else:
        regime_decoupling = False
        results['tests']['regime_comparison'] = None
    
    # ==========================================================================
    # TEST 3: "Old but Star-Forming" Population
    # ==========================================================================
    print_status("\n" + "=" * 50, "INFO")
    print_status("TEST 3: 'Old but Star-Forming' Population", "INFO")
    print_status("=" * 50, "INFO")
    
    # Define thresholds
    age_threshold = valid['mwa'].quantile(0.75)  # Top 25% in age
    ssfr_threshold = valid['log_ssfr'].quantile(0.5)  # Above median sSFR
    
    # "Old but star-forming" galaxies
    old_sf = valid[(valid['mwa'] > age_threshold) & (valid['log_ssfr'] > ssfr_threshold)]
    
    print_status(f"\nAge threshold (75th %ile): {age_threshold:.3f} Gyr", "INFO")
    print_status(f"sSFR threshold (median): {10**ssfr_threshold:.2e} yr⁻¹", "INFO")
    print_status(f"'Old but Star-Forming' galaxies: N = {len(old_sf)}", "INFO")
    
    if len(old_sf) > 10:
        # Compare Gamma_t of this population vs rest
        mean_gamma_old_sf = old_sf['gamma_t'].mean()
        mean_gamma_rest = valid[~valid.index.isin(old_sf.index)]['gamma_t'].mean()
        
        print_status(f"\nMean Γ_t:", "INFO")
        print_status(f"  'Old but Star-Forming': {mean_gamma_old_sf:.3f}", "INFO")
        print_status(f"  Rest of sample: {mean_gamma_rest:.3f}", "INFO")
        print_status(f"  Ratio: {mean_gamma_old_sf/mean_gamma_rest:.2f}×", "INFO")
        
        # TEP predicts this population has HIGH Gamma_t
        if mean_gamma_old_sf > mean_gamma_rest * 1.2:
            print_status("\n✓ 'Old but Star-Forming' galaxies have elevated Γ_t", "INFO")
            print_status("  TEP explanation: Their ages are inflated by temporal enhancement", "INFO")
            old_sf_elevated = True
        else:
            print_status("\n⚠ No Γ_t elevation detected", "INFO")
            old_sf_elevated = False
        
        results['tests']['old_but_sf'] = {
            'n': len(old_sf),
            'age_threshold': float(age_threshold),
            'ssfr_threshold': float(ssfr_threshold),
            'mean_gamma_old_sf': float(mean_gamma_old_sf),
            'mean_gamma_rest': float(mean_gamma_rest),
            'ratio': float(mean_gamma_old_sf / mean_gamma_rest),
            'old_sf_elevated': old_sf_elevated
        }
    else:
        old_sf_elevated = False
        results['tests']['old_but_sf'] = None
    
    # ==========================================================================
    # TEST 4: Residual Age vs Gamma_t at Fixed sSFR
    # ==========================================================================
    print_status("\n" + "=" * 50, "INFO")
    print_status("TEST 4: Age Residual at Fixed sSFR", "INFO")
    print_status("=" * 50, "INFO")
    
    from scipy.stats import linregress
    
    # Fit age-sSFR relation
    slope, intercept, _, _, _ = linregress(valid['log_ssfr'], valid['log_age'])
    valid['age_resid_ssfr'] = valid['log_age'] - (slope * valid['log_ssfr'] + intercept)
    
    print_status(f"\nAge-sSFR relation: log(Age) = {slope:.3f} × log(sSFR) + {intercept:.3f}", "INFO")
    
    # Correlation of residual with Gamma_t
    rho_resid, p_resid = spearmanr(valid['gamma_t'], valid['age_resid_ssfr'])
    
    print_status(f"\nρ(Γ_t, Age_resid|sSFR) = {rho_resid:.3f} (p = {p_resid:.2e})", "INFO")
    
    # TEP predicts POSITIVE correlation (high Gamma_t → older than expected at fixed sSFR)
    if rho_resid > 0 and p_resid < 0.001:
        print_status("\n✓ STRONG EVIDENCE: High-Γ_t galaxies are older than expected at fixed sSFR", "INFO")
        print_status("  This is a unique TEP signature", "INFO")
        age_resid_positive = True
    elif rho_resid > 0 and p_resid < 0.05:
        print_status("\n⚠ Moderate evidence for age inflation at fixed sSFR", "INFO")
        age_resid_positive = True
    else:
        print_status("\n✗ No age inflation detected at fixed sSFR", "INFO")
        age_resid_positive = False
    
    results['tests']['age_resid_ssfr'] = {
        'slope': float(slope),
        'intercept': float(intercept),
        'rho': float(rho_resid),
        'p': format_p_value(p_resid),
        'age_resid_positive': age_resid_positive
    }
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("SUMMARY: SFR-Age Consistency", "INFO")
    print_status("=" * 70, "INFO")
    
    n_tests_passed = sum([
        regime_decoupling,
        old_sf_elevated,
        age_resid_positive
    ])
    
    print_status(f"\nTests passed: {n_tests_passed}/3", "INFO")
    
    if n_tests_passed >= 2:
        print_status("\n✓ SFR-AGE DECOUPLING CONFIRMED", "INFO")
        print_status("  High-Γ_t galaxies appear older than their SFR would suggest.", "INFO")
        print_status("  This is a unique TEP signature.", "INFO")
        overall = "CONFIRMED"
    elif n_tests_passed == 1:
        print_status("\n⚠ PARTIAL EVIDENCE for SFR-Age decoupling", "INFO")
        overall = "PARTIAL"
    else:
        print_status("\n✗ No clear SFR-Age decoupling detected", "INFO")
        overall = "NOT_DETECTED"
    
    results['summary'] = {
        'tests_passed': n_tests_passed,
        'tests_total': 3,
        'overall': overall
    }
    
    # Save
    json_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print_status(f"\nSaved to {json_path}", "INFO")

if __name__ == "__main__":
    main()
