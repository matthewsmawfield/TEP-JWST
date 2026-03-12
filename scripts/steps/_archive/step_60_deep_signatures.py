#!/usr/bin/env python3
"""
TEP-JWST Step 60: Deep Signatures

This step explores deeper, more subtle TEP signatures:

1. Quenching Timescale: High-Gamma_t galaxies should quench faster (more effective time)
2. Chemical Evolution: Metallicity enrichment rate should differ by regime
3. Color Gradient: Internal color gradients should correlate with Gamma_t
4. SFH Complexity: High-Gamma_t galaxies should have more complex SFHs
5. Dust-to-Gas Ratio Proxy: Should scale with effective time
6. Main Sequence Offset: Position on SF main sequence should correlate with Gamma_t
"""

import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, linregress, ks_2samp
from pathlib import Path
import json

# =============================================================================
# PATHS AND LOGGER
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.p_value_utils import format_p_value

STEP_NUM = "60"
STEP_NAME = "deep_signatures"

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
    print_status(f"STEP {STEP_NUM}: Deep Signatures", "INFO")
    print_status("=" * 70, "INFO")
    
    # Load data
    df = pd.read_csv(INTERIM_PATH / "step_02_uncover_full_sample_tep.csv")
    print_status(f"\nLoaded N = {len(df)} galaxies", "INFO")
    
    results = {
        'n_total': len(df),
        'tests': {}
    }
    
    evidence = []
    
    # ==========================================================================
    # TEST 1: Quenching Fraction by Regime
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 1: Quenching Fraction by Regime", "INFO")
    print_status("=" * 70, "INFO")
    print_status("TEP predicts faster quenching in enhanced regime (more effective time).\n", "INFO")
    
    valid = df.dropna(subset=['ssfr100', 'gamma_t', 'z_phot'])
    valid = valid[valid['ssfr100'] > 0]
    high_z = valid[valid['z_phot'] > 6].copy()
    
    if len(high_z) > 100:
        high_z['log_ssfr'] = np.log10(high_z['ssfr100'])
        
        # Define quiescent as log(sSFR) < -10
        high_z['quiescent'] = high_z['log_ssfr'] < -10
        
        # Compare by regime
        enhanced = high_z[high_z['gamma_t'] > 1]
        intermediate = high_z[(high_z['gamma_t'] >= 0.5) & (high_z['gamma_t'] <= 1)]
        suppressed = high_z[high_z['gamma_t'] < 0.5]
        
        q_frac_enh = enhanced['quiescent'].mean() if len(enhanced) > 10 else np.nan
        q_frac_int = intermediate['quiescent'].mean() if len(intermediate) > 10 else np.nan
        q_frac_sup = suppressed['quiescent'].mean() if len(suppressed) > 10 else np.nan
        
        print_status(f"Quiescent fraction at z > 6:", "INFO")
        print_status(f"  Enhanced (Γ_t > 1): {q_frac_enh*100:.1f}% (N = {len(enhanced)})", "INFO")
        print_status(f"  Intermediate: {q_frac_int*100:.1f}% (N = {len(intermediate)})", "INFO")
        print_status(f"  Suppressed (Γ_t < 0.5): {q_frac_sup*100:.1f}% (N = {len(suppressed)})", "INFO")
        
        if q_frac_enh > q_frac_sup * 1.5:
            print_status("✓ Enhanced regime has higher quiescent fraction", "INFO")
            evidence.append(('quenching_fraction', q_frac_enh/max(q_frac_sup, 0.001), None))
        
        results['tests']['quenching'] = {
            'q_frac_enhanced': float(q_frac_enh) if not np.isnan(q_frac_enh) else None,
            'q_frac_intermediate': float(q_frac_int) if not np.isnan(q_frac_int) else None,
            'q_frac_suppressed': float(q_frac_sup) if not np.isnan(q_frac_sup) else None
        }
    
    # ==========================================================================
    # TEST 2: Main Sequence Offset
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 2: Star-Forming Main Sequence Offset", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Position relative to SF main sequence should correlate with Gamma_t.\n", "INFO")
    
    valid = df.dropna(subset=['log_Mstar', 'sfr100', 'gamma_t', 'z_phot'])
    valid = valid[valid['sfr100'] > 0]
    high_z = valid[valid['z_phot'] > 7].copy()
    
    if len(high_z) > 50:
        high_z['log_sfr'] = np.log10(high_z['sfr100'])
        
        # Fit main sequence
        slope, intercept, _, _, _ = linregress(high_z['log_Mstar'], high_z['log_sfr'])
        high_z['ms_offset'] = high_z['log_sfr'] - (slope * high_z['log_Mstar'] + intercept)
        
        rho, p = spearmanr(high_z['gamma_t'], high_z['ms_offset'])
        
        print_status(f"N = {len(high_z)}", "INFO")
        print_status(f"ρ(Γ_t, MS_offset) = {rho:.3f} (p = {p:.2e})", "INFO")
        
        # TEP predicts NEGATIVE correlation (high Gamma_t → below MS due to inflated M*)
        if rho < -0.1 and p < 0.01:
            print_status("✓ High-Γ_t galaxies are below the main sequence", "INFO")
            evidence.append(('ms_offset', rho, format_p_value(p)))
        
        results['tests']['ms_offset'] = {
            'n': int(len(high_z)),
            'rho': float(rho),
            'p': format_p_value(p)
        }
    
    # ==========================================================================
    # TEST 3: Metallicity Enrichment Rate
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 3: Metallicity Enrichment Rate", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Metallicity per unit stellar mass should differ by regime.\n", "INFO")
    
    valid = df.dropna(subset=['met', 'log_Mstar', 'gamma_t', 'z_phot'])
    high_z = valid[valid['z_phot'] > 7].copy()
    
    if len(high_z) > 50:
        # Metallicity enrichment efficiency = Z / M*
        high_z['met_eff'] = high_z['met'] - high_z['log_Mstar']
        
        rho, p = spearmanr(high_z['gamma_t'], high_z['met_eff'])
        
        print_status(f"N = {len(high_z)}", "INFO")
        print_status(f"ρ(Γ_t, Z/M*) = {rho:.3f} (p = {p:.2e})", "INFO")
        
        results['tests']['met_enrichment'] = {
            'n': int(len(high_z)),
            'rho': float(rho),
            'p': format_p_value(p)
        }
    
    # ==========================================================================
    # TEST 4: SFH Complexity (Burstiness Variance)
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 4: SFH Complexity", "INFO")
    print_status("=" * 70, "INFO")
    print_status("High-Gamma_t galaxies should have smoother SFHs (less bursty).\n", "INFO")
    
    valid = df.dropna(subset=['sfr10', 'sfr100', 'gamma_t', 'z_phot'])
    valid = valid[(valid['sfr10'] > 0) & (valid['sfr100'] > 0)]
    high_z = valid[valid['z_phot'] > 7].copy()
    
    if len(high_z) > 50:
        high_z['burstiness'] = np.log10(high_z['sfr10'] / high_z['sfr100'])
        
        # Compare burstiness distributions
        enhanced = high_z[high_z['gamma_t'] > 1]
        suppressed = high_z[high_z['gamma_t'] < 0.5]
        
        if len(enhanced) > 10 and len(suppressed) > 10:
            mean_burst_enh = enhanced['burstiness'].mean()
            mean_burst_sup = suppressed['burstiness'].mean()
            std_burst_enh = enhanced['burstiness'].std()
            std_burst_sup = suppressed['burstiness'].std()
            
            print_status(f"Burstiness (log SFR10/SFR100):", "INFO")
            print_status(f"  Enhanced: {mean_burst_enh:.3f} ± {std_burst_enh:.3f}", "INFO")
            print_status(f"  Suppressed: {mean_burst_sup:.3f} ± {std_burst_sup:.3f}", "INFO")
            
            # TEP predicts lower burstiness (smoother SFH) in enhanced regime
            if mean_burst_enh < mean_burst_sup:
                print_status("✓ Enhanced regime has smoother SFHs (less bursty)", "INFO")
                evidence.append(('sfh_smoothness', mean_burst_sup - mean_burst_enh, None))
            
            results['tests']['sfh_complexity'] = {
                'mean_burst_enhanced': float(mean_burst_enh),
                'mean_burst_suppressed': float(mean_burst_sup),
                'std_burst_enhanced': float(std_burst_enh),
                'std_burst_suppressed': float(std_burst_sup)
            }
    
    # ==========================================================================
    # TEST 5: Dust Production Rate
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 5: Dust Production Rate", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Dust per unit SFR should be higher in enhanced regime (more AGB time).\n", "INFO")
    
    valid = df.dropna(subset=['dust', 'sfr100', 'gamma_t', 'z_phot'])
    valid = valid[(valid['dust'] > 0) & (valid['sfr100'] > 0)]
    high_z = valid[valid['z_phot'] > 7].copy()
    
    if len(high_z) > 50:
        # Dust production efficiency = A_V / SFR
        high_z['dust_prod'] = np.log10(high_z['dust'] / high_z['sfr100'])
        
        rho, p = spearmanr(high_z['gamma_t'], high_z['dust_prod'])
        
        print_status(f"N = {len(high_z)}", "INFO")
        print_status(f"ρ(Γ_t, A_V/SFR) = {rho:.3f} (p = {p:.2e})", "INFO")
        
        # TEP predicts POSITIVE correlation (more time → more dust per unit SF)
        if rho > 0.1 and p < 0.01:
            print_status("✓ High-Γ_t galaxies have higher dust production efficiency", "INFO")
            evidence.append(('dust_production', rho, format_p_value(p)))
        
        results['tests']['dust_production'] = {
            'n': int(len(high_z)),
            'rho': float(rho),
            'p': format_p_value(p)
        }
    
    # ==========================================================================
    # TEST 6: Age-Metallicity Relation
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 6: Age-Metallicity Relation by Regime", "INFO")
    print_status("=" * 70, "INFO")
    print_status("The age-metallicity relation should differ between regimes.\n", "INFO")
    
    valid = df.dropna(subset=['mwa', 'met', 'gamma_t', 'z_phot'])
    valid = valid[valid['mwa'] > 0]
    high_z = valid[valid['z_phot'] > 7].copy()
    
    if len(high_z) > 50:
        enhanced = high_z[high_z['gamma_t'] > 1]
        suppressed = high_z[high_z['gamma_t'] < 0.5]
        
        if len(enhanced) > 10 and len(suppressed) > 10:
            rho_enh, p_enh = spearmanr(enhanced['mwa'], enhanced['met'])
            rho_sup, p_sup = spearmanr(suppressed['mwa'], suppressed['met'])
            
            print_status(f"Age-Metallicity correlation:", "INFO")
            print_status(f"  Enhanced: ρ = {rho_enh:.3f} (p = {p_enh:.2e})", "INFO")
            print_status(f"  Suppressed: ρ = {rho_sup:.3f} (p = {p_sup:.2e})", "INFO")
            
            results['tests']['age_met_relation'] = {
                'rho_enhanced': float(rho_enh),
                'rho_suppressed': float(rho_sup),
                'difference': float(rho_enh - rho_sup)
            }
    
    # ==========================================================================
    # TEST 7: Stellar Mass Growth Rate
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 7: Stellar Mass Growth Rate", "INFO")
    print_status("=" * 70, "INFO")
    print_status("M* / SFR (mass doubling time) should differ by regime.\n", "INFO")
    
    valid = df.dropna(subset=['log_Mstar', 'sfr100', 'gamma_t', 'z_phot'])
    valid = valid[valid['sfr100'] > 0]
    high_z = valid[valid['z_phot'] > 7].copy()
    
    if len(high_z) > 50:
        # Mass doubling time = M* / SFR
        high_z['t_double'] = 10**high_z['log_Mstar'] / high_z['sfr100'] / 1e9  # in Gyr
        high_z['log_t_double'] = np.log10(high_z['t_double'])
        
        rho, p = spearmanr(high_z['gamma_t'], high_z['log_t_double'])
        
        print_status(f"N = {len(high_z)}", "INFO")
        print_status(f"ρ(Γ_t, log t_double) = {rho:.3f} (p = {p:.2e})", "INFO")
        
        # TEP predicts POSITIVE correlation (high Gamma_t → longer doubling time due to inflated M*)
        if rho > 0.2 and p < 0.001:
            print_status("✓ High-Γ_t galaxies have longer mass doubling times", "INFO")
            evidence.append(('mass_doubling', rho, format_p_value(p)))
        
        results['tests']['mass_doubling'] = {
            'n': int(len(high_z)),
            'rho': float(rho),
            'p': format_p_value(p)
        }
    
    # ==========================================================================
    # TEST 8: The "Downsizing" Signature
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 8: The 'Downsizing' Signature", "INFO")
    print_status("=" * 70, "INFO")
    print_status("More massive galaxies should have older populations (downsizing).\n", "INFO")
    print_status("TEP predicts this is ENHANCED at high-z.\n", "INFO")
    
    valid = df.dropna(subset=['log_Mstar', 'mwa', 'z_phot'])
    valid = valid[valid['mwa'] > 0]
    
    # Compare downsizing at different z
    z_bins = [(4, 6), (6, 8), (8, 12)]
    downsizing_results = []
    
    for z_lo, z_hi in z_bins:
        bin_data = valid[(valid['z_phot'] >= z_lo) & (valid['z_phot'] < z_hi)]
        if len(bin_data) > 30:
            rho, p = spearmanr(bin_data['log_Mstar'], bin_data['mwa'])
            downsizing_results.append({
                'z_range': f"{z_lo}-{z_hi}",
                'n': int(len(bin_data)),
                'rho': float(rho),
                'p': format_p_value(p)
            })
            print_status(f"z = {z_lo}-{z_hi}: ρ(M*, Age) = {rho:.3f} (p = {p:.2e})", "INFO")
    
    results['tests']['downsizing'] = downsizing_results
    
    # ==========================================================================
    # TEST 9: Redshift Evolution of TEP Strength
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 9: Redshift Evolution of TEP Strength", "INFO")
    print_status("=" * 70, "INFO")
    print_status("The Gamma_t-Dust correlation should strengthen with z.\n", "INFO")
    
    valid = df.dropna(subset=['gamma_t', 'dust', 'z_phot'])
    
    z_bins = [(4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 12)]
    z_evolution = []
    
    for z_lo, z_hi in z_bins:
        bin_data = valid[(valid['z_phot'] >= z_lo) & (valid['z_phot'] < z_hi)]
        if len(bin_data) > 20:
            rho, p = spearmanr(bin_data['gamma_t'], bin_data['dust'])
            z_evolution.append({
                'z_range': f"{z_lo}-{z_hi}",
                'z_mid': (z_lo + z_hi) / 2,
                'n': int(len(bin_data)),
                'rho': float(rho),
                'p': format_p_value(p)
            })
            sig = "✓" if rho > 0.3 and p < 0.01 else ""
            print_status(f"z = {z_lo}-{z_hi}: ρ = {rho:.3f} (p = {p:.2e}) {sig}", "INFO")
    
    # Check trend
    if len(z_evolution) >= 4:
        z_mids = [r['z_mid'] for r in z_evolution]
        rhos = [r['rho'] for r in z_evolution]
        trend_rho, trend_p = spearmanr(z_mids, rhos)
        
        print_status(f"\nTrend: ρ(z, TEP strength) = {trend_rho:.3f} (p = {trend_p:.3f})", "INFO")
        
        if trend_rho > 0.6:
            print_status("✓ TEP effect strengthens with redshift", "INFO")
            evidence.append(('z_evolution', trend_rho, format_p_value(trend_p)))
    
    results['tests']['z_evolution'] = z_evolution
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("SUMMARY: Deep Signatures Found", "INFO")
    print_status("=" * 70, "INFO")
    
    print_status(f"\nEvidence items: {len(evidence)}", "INFO")
    for name, stat, p in evidence:
        if p is not None and p > 0:
            print_status(f"  • {name}: stat = {stat:.3f}, p = {p:.2e}", "INFO")
        else:
            print_status(f"  • {name}: stat = {stat:.3f}", "INFO")
    
    results['summary'] = {
        'n_evidence': len(evidence),
        'evidence': [{'name': n, 'stat': float(s), 'p': format_p_value(p)} for n, s, p in evidence]
    }
    
    # Save
    json_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print_status(f"\nSaved to {json_path}", "INFO")

if __name__ == "__main__":
    main()
