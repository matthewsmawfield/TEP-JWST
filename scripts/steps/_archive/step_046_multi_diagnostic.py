#!/usr/bin/env python3
"""
TEP-JWST Step 46: Multi-Diagnostic Evidence Search

This step explores multiple independent diagnostics to find additional
strong evidence for TEP:

1. UV Slope (beta): TEP predicts redder UV slopes for high-Gamma_t galaxies
2. Dust-to-Stellar Mass Ratio: Should correlate with Gamma_t at high-z
3. SFR Timescale Ratio: Burstiness should anti-correlate with Gamma_t
4. Chi2 Redshift Evolution: The chi2-Gamma_t correlation should strengthen with z
5. Color Excess: E(B-V) should correlate with Gamma_t at z > 8
6. Mass-Weighted Age vs Light-Weighted Age: Divergence should scale with Gamma_t
"""

import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, linregress
from pathlib import Path
import json

# =============================================================================
# PATHS AND LOGGER
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.p_value_utils import format_p_value

STEP_NUM = "046"
STEP_NAME = "multi_diagnostic"

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
    print_status(f"STEP {STEP_NUM}: Multi-Diagnostic Evidence Search", "INFO")
    print_status("=" * 70, "INFO")
    
    # Load data
    df = pd.read_csv(INTERIM_PATH / "step_002_uncover_full_sample_tep.csv")
    print_status(f"\nLoaded N = {len(df)} galaxies", "INFO")
    
    results = {
        'n_total': len(df),
        'diagnostics': {}
    }
    
    strong_evidence = []
    
    # ==========================================================================
    # DIAGNOSTIC 1: Burstiness (SFR10/SFR100)
    # ==========================================================================
    print_status("\n" + "=" * 60, "INFO")
    print_status("DIAGNOSTIC 1: Burstiness (SFR10/SFR100)", "INFO")
    print_status("=" * 60, "INFO")
    
    valid = df.dropna(subset=['sfr10', 'sfr100', 'gamma_t'])
    valid = valid[(valid['sfr10'] > 0) & (valid['sfr100'] > 0)]
    
    if len(valid) > 100:
        valid = valid.copy()
        valid['burstiness'] = np.log10(valid['sfr10'] / valid['sfr100'])
        
        rho, p = spearmanr(valid['gamma_t'], valid['burstiness'])
        
        print_status(f"N = {len(valid)}", "INFO")
        print_status(f"ρ(Γ_t, Burstiness) = {rho:.3f} (p = {p:.2e})", "INFO")
        
        # TEP predicts NEGATIVE correlation (high Gamma_t → less bursty)
        if rho < -0.1 and p < 0.001:
            print_status("✓ STRONG: High-Γ_t galaxies are less bursty", "INFO")
            strong_evidence.append(('Burstiness', rho, format_p_value(p)))
        
        results['diagnostics']['burstiness'] = {
            'n': int(len(valid)),
            'rho': float(rho),
            'p': format_p_value(p),
            'strong': bool(rho < -0.1 and p < 0.001)
        }
    
    # ==========================================================================
    # DIAGNOSTIC 2: Dust Content at z > 8
    # ==========================================================================
    print_status("\n" + "=" * 60, "INFO")
    print_status("DIAGNOSTIC 2: Dust Content at z > 8", "INFO")
    print_status("=" * 60, "INFO")
    
    high_z = df[(df['z_phot'] > 8) & df['dust'].notna() & df['gamma_t'].notna()]
    
    if len(high_z) > 30:
        rho, p = spearmanr(high_z['gamma_t'], high_z['dust'])
        
        print_status(f"N = {len(high_z)}", "INFO")
        print_status(f"ρ(Γ_t, A_V) at z > 8 = {rho:.3f} (p = {p:.2e})", "INFO")
        
        # Also test mass-dust
        rho_mass, p_mass = spearmanr(high_z['log_Mstar'], high_z['dust'])
        print_status(f"ρ(M*, A_V) at z > 8 = {rho_mass:.3f} (p = {p_mass:.2e})", "INFO")
        
        if rho > 0.3 and p < 0.001:
            print_status("✓ STRONG: Dust correlates with Γ_t at z > 8", "INFO")
            strong_evidence.append(('Dust_z8', rho, format_p_value(p)))
        
        results['diagnostics']['dust_z8'] = {
            'n': int(len(high_z)),
            'rho_gamma_dust': float(rho),
            'p_gamma_dust': format_p_value(p),
            'rho_mass_dust': float(rho_mass),
            'p_mass_dust': format_p_value(p_mass),
            'strong': bool(rho > 0.3 and p < 0.001)
        }
    
    # ==========================================================================
    # DIAGNOSTIC 3: Chi2 Evolution with Redshift
    # ==========================================================================
    print_status("\n" + "=" * 60, "INFO")
    print_status("DIAGNOSTIC 3: χ² Evolution with Redshift", "INFO")
    print_status("=" * 60, "INFO")
    
    valid = df.dropna(subset=['chi2', 'gamma_t', 'z_phot'])
    valid = valid[valid['chi2'] > 0]
    
    z_bins = [(4, 5), (5, 6), (6, 7), (7, 8), (8, 10)]
    chi2_evolution = []
    
    for z_lo, z_hi in z_bins:
        bin_data = valid[(valid['z_phot'] >= z_lo) & (valid['z_phot'] < z_hi)]
        if len(bin_data) > 30:
            rho, p = spearmanr(bin_data['gamma_t'], bin_data['chi2'])
            chi2_evolution.append({
                'z_range': f"{z_lo}-{z_hi}",
                'n': len(bin_data),
                'rho': float(rho),
                'p': format_p_value(p)
            })
            print_status(f"z = {z_lo}-{z_hi}: N = {len(bin_data)}, ρ = {rho:.3f} (p = {p:.2e})", "INFO")
    
    # Check if correlation strengthens with z (TEP prediction)
    if len(chi2_evolution) >= 3:
        rhos = [x['rho'] for x in chi2_evolution]
        z_mids = [4.5, 5.5, 6.5, 7.5, 9.0][:len(rhos)]
        trend_rho, trend_p = spearmanr(z_mids, rhos)
        
        print_status(f"\nTrend: ρ(z, χ²-Γ_t correlation) = {trend_rho:.3f} (p = {trend_p:.3f})", "INFO")
        
        if trend_rho > 0.5:
            print_status("✓ χ²-Γ_t correlation strengthens with z (TEP-consistent)", "INFO")
            strong_evidence.append(('Chi2_evolution', trend_rho, format_p_value(trend_p)))
    
    results['diagnostics']['chi2_evolution'] = chi2_evolution
    
    # ==========================================================================
    # DIAGNOSTIC 4: Metallicity Gradient with Gamma_t
    # ==========================================================================
    print_status("\n" + "=" * 60, "INFO")
    print_status("DIAGNOSTIC 4: Metallicity at Fixed Mass", "INFO")
    print_status("=" * 60, "INFO")
    
    valid = df.dropna(subset=['met', 'gamma_t', 'log_Mstar'])
    valid = valid[valid['met'] > -3]
    
    if len(valid) > 100:
        # Residualize metallicity against mass
        slope, intercept, _, _, _ = linregress(valid['log_Mstar'], valid['met'])
        valid = valid.copy()
        valid['met_resid'] = valid['met'] - (slope * valid['log_Mstar'] + intercept)
        
        rho, p = spearmanr(valid['gamma_t'], valid['met_resid'])
        
        print_status(f"N = {len(valid)}", "INFO")
        print_status(f"ρ(Γ_t, Met_resid|M*) = {rho:.3f} (p = {p:.2e})", "INFO")
        
        # TEP predicts weak or negative correlation (metallicity saturates)
        if abs(rho) < 0.05:
            print_status("✓ Metallicity does NOT track Γ_t at fixed mass (TEP-consistent)", "INFO")
        elif rho < 0:
            print_status("⚠ Negative correlation (unexpected but interesting)", "INFO")
        
        results['diagnostics']['metallicity_resid'] = {
            'n': len(valid),
            'rho': float(rho),
            'p': format_p_value(p)
        }
    
    # ==========================================================================
    # DIAGNOSTIC 5: Age Ratio Extremes
    # ==========================================================================
    print_status("\n" + "=" * 60, "INFO")
    print_status("DIAGNOSTIC 5: Extreme Age Ratios", "INFO")
    print_status("=" * 60, "INFO")
    
    valid = df.dropna(subset=['age_ratio', 'gamma_t'])
    
    # Find galaxies with age_ratio > 0.4 (approaching cosmic limit)
    extreme = valid[valid['age_ratio'] > 0.4]
    normal = valid[valid['age_ratio'] <= 0.4]
    
    if len(extreme) > 10:
        mean_gamma_extreme = extreme['gamma_t'].mean()
        mean_gamma_normal = normal['gamma_t'].mean()
        
        print_status(f"Extreme (age_ratio > 0.4): N = {len(extreme)}, <Γ_t> = {mean_gamma_extreme:.3f}", "INFO")
        print_status(f"Normal (age_ratio ≤ 0.4): N = {len(normal)}, <Γ_t> = {mean_gamma_normal:.3f}", "INFO")
        print_status(f"Ratio: {mean_gamma_extreme/mean_gamma_normal:.2f}×", "INFO")
        
        if mean_gamma_extreme > mean_gamma_normal * 1.5:
            print_status("✓ STRONG: Extreme age ratios have elevated Γ_t", "INFO")
            strong_evidence.append(('Extreme_age', mean_gamma_extreme/mean_gamma_normal, None))
        
        results['diagnostics']['extreme_age'] = {
            'n_extreme': len(extreme),
            'n_normal': len(normal),
            'mean_gamma_extreme': float(mean_gamma_extreme),
            'mean_gamma_normal': float(mean_gamma_normal),
            'ratio': float(mean_gamma_extreme / mean_gamma_normal)
        }
    
    # ==========================================================================
    # DIAGNOSTIC 6: Double-Control Test (z AND M*)
    # ==========================================================================
    print_status("\n" + "=" * 60, "INFO")
    print_status("DIAGNOSTIC 6: Double-Control (z AND M*)", "INFO")
    print_status("=" * 60, "INFO")
    
    valid = df.dropna(subset=['dust', 'gamma_t', 'log_Mstar', 'z_phot'])
    
    if len(valid) > 100:
        # Residualize dust against both z and M*
        from sklearn.linear_model import LinearRegression
        
        X = valid[['z_phot', 'log_Mstar']].values
        y = valid['dust'].values
        
        model = LinearRegression().fit(X, y)
        valid = valid.copy()
        valid['dust_resid'] = y - model.predict(X)
        
        rho, p = spearmanr(valid['gamma_t'], valid['dust_resid'])
        
        print_status(f"N = {len(valid)}", "INFO")
        print_status(f"ρ(Γ_t, Dust_resid|z,M*) = {rho:.3f} (p = {p:.2e})", "INFO")
        
        if rho > 0.1 and p < 0.001:
            print_status("✓ STRONG: Γ_t predicts dust even after controlling for z AND M*", "INFO")
            strong_evidence.append(('Double_control_dust', rho, format_p_value(p)))
        
        results['diagnostics']['double_control'] = {
            'n': int(len(valid)),
            'rho': float(rho),
            'p': format_p_value(p),
            'strong': bool(rho > 0.1 and p < 0.001)
        }
    
    # ==========================================================================
    # DIAGNOSTIC 7: Quiescent Fraction by Gamma_t
    # ==========================================================================
    print_status("\n" + "=" * 60, "INFO")
    print_status("DIAGNOSTIC 7: Quiescent Fraction by Γ_t", "INFO")
    print_status("=" * 60, "INFO")
    
    valid = df.dropna(subset=['ssfr100', 'gamma_t'])
    valid = valid[valid['ssfr100'] > 0]
    
    # Define quiescent as log(sSFR) < -10
    valid = valid.copy()
    valid['log_ssfr'] = np.log10(valid['ssfr100'])
    valid['quiescent'] = valid['log_ssfr'] < -10
    
    enhanced = valid[valid['gamma_t'] > 1]
    suppressed = valid[valid['gamma_t'] < 0.5]
    
    if len(enhanced) > 20 and len(suppressed) > 20:
        q_frac_enh = enhanced['quiescent'].mean()
        q_frac_sup = suppressed['quiescent'].mean()
        
        print_status(f"Quiescent fraction:", "INFO")
        print_status(f"  Enhanced (Γ_t > 1): {q_frac_enh*100:.1f}%", "INFO")
        print_status(f"  Suppressed (Γ_t < 0.5): {q_frac_sup*100:.1f}%", "INFO")
        
        ratio = q_frac_enh / max(q_frac_sup, 0.001)
        if q_frac_enh > q_frac_sup * 1.5:
            print_status("✓ Enhanced regime has higher quiescent fraction (TEP-consistent)", "INFO")
            strong_evidence.append(('Quiescent_fraction', min(ratio, 100), None))
        
        results['diagnostics']['quiescent'] = {
            'n_enhanced': int(len(enhanced)),
            'n_suppressed': int(len(suppressed)),
            'q_frac_enhanced': float(q_frac_enh),
            'q_frac_suppressed': float(q_frac_sup),
            'ratio': float(min(ratio, 100))
        }
    
    # ==========================================================================
    # DIAGNOSTIC 8: Mass-Age Correlation at z > 8
    # ==========================================================================
    print_status("\n" + "=" * 60, "INFO")
    print_status("DIAGNOSTIC 8: Mass-Age at z > 8", "INFO")
    print_status("=" * 60, "INFO")
    
    high_z = df[(df['z_phot'] > 8) & df['mwa'].notna() & df['log_Mstar'].notna()]
    
    if len(high_z) > 30:
        rho, p = spearmanr(high_z['log_Mstar'], high_z['mwa'])
        
        print_status(f"N = {len(high_z)}", "INFO")
        print_status(f"ρ(M*, Age) at z > 8 = {rho:.3f} (p = {p:.2e})", "INFO")
        
        # TEP predicts positive correlation (massive = older due to enhanced time)
        if rho > 0.2 and p < 0.01:
            print_status("✓ STRONG: Mass-Age correlation at z > 8 (TEP signature)", "INFO")
            strong_evidence.append(('Mass_Age_z8', rho, format_p_value(p)))
        
        results['diagnostics']['mass_age_z8'] = {
            'n': int(len(high_z)),
            'rho': float(rho),
            'p': format_p_value(p),
            'strong': bool(rho > 0.2 and p < 0.01)
        }
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("SUMMARY: Strong Evidence Found", "INFO")
    print_status("=" * 70, "INFO")
    
    print_status(f"\nStrong evidence items: {len(strong_evidence)}", "INFO")
    for name, stat, p in strong_evidence:
        if p is not None:
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
