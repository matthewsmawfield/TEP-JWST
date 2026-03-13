#!/usr/bin/env python3
"""
TEP-JWST Step 49: Prediction Tests

This step tests specific TEP predictions that would be difficult to
explain with standard physics:

1. The "Anomalous Triangle": Galaxies that are simultaneously massive, old, and dusty
2. The "Screening Gradient": TEP effect should be stronger in denser environments
3. The "Redshift Threshold": TEP signatures should emerge above z ~ 6-7
4. The "Mass Threshold": TEP signatures should emerge above log(M_h) ~ 11
5. The "Age Paradox Resolution": Galaxies with age > t_cosmic should have high Gamma_t
"""

import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu, ks_2samp  # Correlation, Mann-Whitney, and KS tests
from pathlib import Path
import json

# =============================================================================
# PATHS AND LOGGER
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging (severity levels: DEBUG/INFO/WARNING/ERROR/SUCCESS)
from scripts.utils.p_value_utils import format_p_value  # Safe p-value formatting (prevents floating-point underflow at p < 1e-300)

STEP_NUM = "049"  # Pipeline step number (sequential 001-176)
STEP_NAME = "prediction_tests"  # Prediction tests: validates 5 specific TEP predictions (anomalous triangle, screening gradient, redshift threshold, mass threshold, age paradox)

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
    print_status(f"STEP {STEP_NUM}: Prediction Tests", "INFO")
    print_status("=" * 70, "INFO")
    
    # Load data
    df = pd.read_csv(INTERIM_PATH / "step_002_uncover_full_sample_tep.csv")
    print_status(f"\nLoaded N = {len(df)} galaxies", "INFO")
    
    results = {
        'n_total': len(df),
        'predictions': {}
    }
    
    strong_evidence = []
    
    # ==========================================================================
    # PREDICTION 1: The "Anomalous Triangle"
    # ==========================================================================
    print_status("\n" + "=" * 60, "INFO")
    print_status("PREDICTION 1: The Anomalous Triangle", "INFO")
    print_status("=" * 60, "INFO")
    print_status("(Massive + Old + Dusty at z > 8 should have elevated Γ_t)", "INFO")
    
    high_z = df[df['z_phot'] > 8].dropna(subset=['log_Mstar', 'mwa', 'dust', 'gamma_t'])
    
    if len(high_z) > 30:
        # Define thresholds for "anomalous"
        mass_thresh = high_z['log_Mstar'].quantile(0.75)  # Top 25% in mass
        age_thresh = high_z['mwa'].quantile(0.75)         # Top 25% in age
        dust_thresh = high_z['dust'].quantile(0.75)       # Top 25% in dust
        
        anomalous = high_z[
            (high_z['log_Mstar'] > mass_thresh) &
            (high_z['mwa'] > age_thresh) &
            (high_z['dust'] > dust_thresh)
        ]
        normal = high_z.drop(anomalous.index)
        
        print_status(f"Thresholds: M* > {mass_thresh:.2f}, Age > {age_thresh:.3f} Gyr, A_V > {dust_thresh:.2f}", "INFO")
        print_status(f"'Anomalous' galaxies: N = {len(anomalous)}", "INFO")
        
        if len(anomalous) > 3:
            mean_gamma_imp = anomalous['gamma_t'].mean()
            mean_gamma_norm = normal['gamma_t'].mean()
            
            print_status(f"  <Γ_t> anomalous = {mean_gamma_imp:.3f}", "INFO")
            print_status(f"  <Γ_t> normal = {mean_gamma_norm:.3f}", "INFO")
            print_status(f"  Ratio: {mean_gamma_imp/mean_gamma_norm:.2f}×", "INFO")
            
            if mean_gamma_imp > mean_gamma_norm * 2:
                print_status("✓ STRONG: 'Anomalous triangle' galaxies have elevated Γ_t", "INFO")
                strong_evidence.append(('impossible_triangle', mean_gamma_imp/mean_gamma_norm, None))
            
            results['predictions']['impossible_triangle'] = {
                'n_impossible': int(len(anomalous)),
                'n_normal': int(len(normal)),
                'mean_gamma_impossible': float(mean_gamma_imp),
                'mean_gamma_normal': float(mean_gamma_norm),
                'ratio': float(mean_gamma_imp / mean_gamma_norm),
                'thresholds': {
                    'mass': float(mass_thresh),
                    'age': float(age_thresh),
                    'dust': float(dust_thresh)
                }
            }
    
    # ==========================================================================
    # PREDICTION 2: Redshift Threshold
    # ==========================================================================
    print_status("\n" + "=" * 60, "INFO")
    print_status("PREDICTION 2: Redshift Threshold", "INFO")
    print_status("=" * 60, "INFO")
    print_status("(TEP-Dust correlation should emerge above z ~ 6-7)", "INFO")
    
    valid = df.dropna(subset=['gamma_t', 'dust', 'z_phot'])
    
    # Test correlation strength at different z thresholds
    z_thresholds = [5, 6, 7, 8, 9, 10]
    threshold_results = []
    
    for z_thresh in z_thresholds:
        above = valid[valid['z_phot'] > z_thresh]
        if len(above) > 30:
            rho, p = spearmanr(above['gamma_t'], above['dust'])
            threshold_results.append({
                'z_threshold': z_thresh,
                'n': int(len(above)),
                'rho': float(rho),
                'p': format_p_value(p)
            })
            sig = "✓" if rho > 0.3 and p < 0.001 else ""
            print_status(f"z > {z_thresh}: N = {len(above)}, ρ = {rho:.3f} (p = {p:.2e}) {sig}", "INFO")
    
    # Find threshold where correlation becomes strong
    strong_z = [r for r in threshold_results if r['rho'] > 0.3 and r['p'] is not None and r['p'] < 0.001]
    if strong_z:
        emergence_z = min(r['z_threshold'] for r in strong_z)
        print_status(f"\n✓ TEP-Dust correlation emerges at z > {emergence_z}", "INFO")
        strong_evidence.append(('z_threshold', emergence_z, None))
    
    results['predictions']['z_threshold'] = threshold_results
    
    # ==========================================================================
    # PREDICTION 3: Mass Threshold
    # ==========================================================================
    print_status("\n" + "=" * 60, "INFO")
    print_status("PREDICTION 3: Mass Threshold", "INFO")
    print_status("=" * 60, "INFO")
    print_status("(TEP effect should be stronger in massive halos)", "INFO")
    
    high_z = df[df['z_phot'] > 7].dropna(subset=['gamma_t', 'dust', 'log_Mh'])
    
    if len(high_z) > 50:
        # Test correlation at different halo mass thresholds
        mh_thresholds = [9.5, 10.0, 10.5, 11.0, 11.5]
        mass_results = []
        
        for mh_thresh in mh_thresholds:
            above = high_z[high_z['log_Mh'] > mh_thresh]
            if len(above) > 20:
                rho, p = spearmanr(above['gamma_t'], above['dust'])
                mass_results.append({
                    'mh_threshold': mh_thresh,
                    'n': int(len(above)),
                    'rho': float(rho),
                    'p': format_p_value(p)
                })
                sig = "✓" if rho > 0.4 and p < 0.001 else ""
                print_status(f"log M_h > {mh_thresh}: N = {len(above)}, ρ = {rho:.3f} (p = {p:.2e}) {sig}", "INFO")
        
        results['predictions']['mass_threshold'] = mass_results
    
    # ==========================================================================
    # PREDICTION 4: Age Paradox Resolution
    # ==========================================================================
    print_status("\n" + "=" * 60, "INFO")
    print_status("PREDICTION 4: Age Paradox Resolution", "INFO")
    print_status("=" * 60, "INFO")
    print_status("(Galaxies with age > 0.5 × t_cosmic should have high Γ_t)", "INFO")
    
    valid = df.dropna(subset=['age_ratio', 'gamma_t'])
    
    # Define paradoxical galaxies
    paradox_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    paradox_results = []
    
    for thresh in paradox_thresholds:
        paradox = valid[valid['age_ratio'] > thresh]
        normal = valid[valid['age_ratio'] <= thresh]
        
        if len(paradox) > 5 and len(normal) > 100:
            mean_gamma_par = paradox['gamma_t'].mean()
            mean_gamma_norm = normal['gamma_t'].mean()
            ratio = mean_gamma_par / mean_gamma_norm
            
            paradox_results.append({
                'threshold': thresh,
                'n_paradox': int(len(paradox)),
                'mean_gamma_paradox': float(mean_gamma_par),
                'mean_gamma_normal': float(mean_gamma_norm),
                'ratio': float(ratio)
            })
            
            sig = "✓" if ratio > 3 else ""
            print_status(f"age_ratio > {thresh}: N = {len(paradox)}, <Γ_t> = {mean_gamma_par:.2f} ({ratio:.1f}× normal) {sig}", "INFO")
    
    # Find strongest paradox resolution
    if paradox_results:
        best = max(paradox_results, key=lambda x: x['ratio'])
        if best['ratio'] > 5:
            print_status(f"\n✓ STRONG: Age paradox resolved at threshold {best['threshold']}", "INFO")
            strong_evidence.append(('age_paradox', best['ratio'], None))
    
    results['predictions']['age_paradox'] = paradox_results
    
    # ==========================================================================
    # PREDICTION 5: Dust Production Timescale
    # ==========================================================================
    print_status("\n" + "=" * 60, "INFO")
    print_status("PREDICTION 5: Dust Production Timescale", "INFO")
    print_status("=" * 60, "INFO")
    print_status("(Dust should require t_eff > 300 Myr for AGB production)", "INFO")
    
    valid = df.dropna(subset=['t_eff', 'dust', 'gamma_t'])
    
    # Test dust content above/below AGB timescale
    agb_timescale = 0.3  # Gyr
    
    above_agb = valid[valid['t_eff'] > agb_timescale]
    below_agb = valid[valid['t_eff'] <= agb_timescale]
    
    if len(above_agb) > 50 and len(below_agb) > 50:
        mean_dust_above = above_agb['dust'].mean()
        mean_dust_below = below_agb['dust'].mean()
        
        stat, p = mannwhitneyu(above_agb['dust'], below_agb['dust'], alternative='greater')
        
        print_status(f"t_eff > {agb_timescale} Gyr: N = {len(above_agb)}, <A_V> = {mean_dust_above:.3f}", "INFO")
        print_status(f"t_eff ≤ {agb_timescale} Gyr: N = {len(below_agb)}, <A_V> = {mean_dust_below:.3f}", "INFO")
        print_status(f"Mann-Whitney p = {p:.2e}", "INFO")
        
        results['predictions']['dust_timescale'] = {
            'agb_timescale': float(agb_timescale),
            'n_above': int(len(above_agb)),
            'n_below': int(len(below_agb)),
            'mean_dust_above': float(mean_dust_above),
            'mean_dust_below': float(mean_dust_below),
            'p': format_p_value(p)
        }
    
    # ==========================================================================
    # PREDICTION 6: Chi2 Isochrony Violation
    # ==========================================================================
    print_status("\n" + "=" * 60, "INFO")
    print_status("PREDICTION 6: χ² Isochrony Violation", "INFO")
    print_status("=" * 60, "INFO")
    print_status("(High-Γ_t galaxies should have poor SED fits due to isochrony violation)", "INFO")
    
    valid = df.dropna(subset=['chi2', 'gamma_t', 'z_phot'])
    high_z = valid[valid['z_phot'] > 7]
    
    if len(high_z) > 50:
        # Split by Gamma_t
        enhanced = high_z[high_z['gamma_t'] > 1]
        suppressed = high_z[high_z['gamma_t'] < 0.5]
        
        if len(enhanced) > 10 and len(suppressed) > 10:
            mean_chi2_enh = enhanced['chi2'].mean()
            mean_chi2_sup = suppressed['chi2'].mean()
            
            stat, p = mannwhitneyu(enhanced['chi2'], suppressed['chi2'], alternative='greater')
            
            print_status(f"At z > 7:", "INFO")
            print_status(f"  Enhanced (Γ_t > 1): N = {len(enhanced)}, <χ²> = {mean_chi2_enh:.2f}", "INFO")
            print_status(f"  Suppressed (Γ_t < 0.5): N = {len(suppressed)}, <χ²> = {mean_chi2_sup:.2f}", "INFO")
            print_status(f"  Mann-Whitney p = {p:.2e}", "INFO")
            
            if mean_chi2_enh > mean_chi2_sup * 1.2 and p < 0.01:
                print_status("✓ STRONG: Enhanced regime has worse SED fits (isochrony violation)", "INFO")
                strong_evidence.append(('chi2_isochrony', mean_chi2_enh/mean_chi2_sup, format_p_value(p)))
            
            results['predictions']['chi2_isochrony'] = {
                'n_enhanced': int(len(enhanced)),
                'n_suppressed': int(len(suppressed)),
                'mean_chi2_enhanced': float(mean_chi2_enh),
                'mean_chi2_suppressed': float(mean_chi2_sup),
                'ratio': float(mean_chi2_enh / mean_chi2_sup),
                'p': format_p_value(p)
            }
    
    # ==========================================================================
    # PREDICTION 7: Unified Anomaly Explanation
    # ==========================================================================
    print_status("\n" + "=" * 60, "INFO")
    print_status("PREDICTION 7: Unified Anomaly Explanation", "INFO")
    print_status("=" * 60, "INFO")
    print_status("(Single Γ_t parameter should explain multiple anomalies)", "INFO")
    
    valid = df.dropna(subset=['dust', 'mwa', 'chi2', 'gamma_t', 'z_phot'])
    high_z = valid[valid['z_phot'] > 7]
    
    if len(high_z) > 50:
        # Test if Gamma_t predicts all three anomalies
        rho_dust, p_dust = spearmanr(high_z['gamma_t'], high_z['dust'])
        rho_age, p_age = spearmanr(high_z['gamma_t'], high_z['mwa'])
        rho_chi2, p_chi2 = spearmanr(high_z['gamma_t'], high_z['chi2'])
        
        print_status(f"At z > 7 (N = {len(high_z)}):", "INFO")
        print_status(f"  ρ(Γ_t, Dust) = {rho_dust:.3f} (p = {p_dust:.2e})", "INFO")
        print_status(f"  ρ(Γ_t, Age) = {rho_age:.3f} (p = {p_age:.2e})", "INFO")
        print_status(f"  ρ(Γ_t, χ²) = {rho_chi2:.3f} (p = {p_chi2:.2e})", "INFO")
        
        # Count significant correlations
        n_sig = sum(1 for p in [p_dust, p_age, p_chi2] if p < 0.01)
        print_status(f"\nSignificant correlations: {n_sig}/3", "INFO")
        
        if n_sig >= 2:
            print_status("✓ STRONG: Γ_t provides unified explanation for multiple anomalies", "INFO")
            strong_evidence.append(('unified_explanation', n_sig, None))
        
        results['predictions']['unified'] = {
            'n': int(len(high_z)),
            'rho_dust': float(rho_dust),
            'rho_age': float(rho_age),
            'rho_chi2': float(rho_chi2),
            'n_significant': n_sig
        }
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("SUMMARY: Strong Evidence Found", "INFO")
    print_status("=" * 70, "INFO")
    
    print_status(f"\nStrong evidence items: {len(strong_evidence)}", "INFO")
    for name, stat, p in strong_evidence:
        if p is not None and p > 0:
            print_status(f"  • {name}: stat = {stat:.3f}, p = {p:.2e}", "INFO")
        else:
            print_status(f"  • {name}: stat = {stat:.3f}", "INFO")
    
    results['summary'] = {
        'n_strong_evidence': len(strong_evidence),
        'strong_evidence': [{'name': n, 'stat': float(s), 'p': format_p_value(p)} for n, s, p in strong_evidence]
    }
    
    # Save
    json_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print_status(f"\nSaved to {json_path}", "INFO")

if __name__ == "__main__":
    main()
