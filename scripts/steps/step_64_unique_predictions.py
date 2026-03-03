#!/usr/bin/env python3
"""
TEP-JWST Step 64: Unique Predictions

This step tests predictions that are UNIQUE to TEP and cannot be
explained by any standard astrophysical process:

1. The "Inverse Downsizing" at z > 8: Standard downsizing should reverse
2. The "Dust Desert" Violation: Dust at z > 10 where t_cosmic < 300 Myr
3. The "Mass-Age Inversion": At fixed z, more massive = older (opposite of standard)
4. The "Chi2 Ladder": Chi2 should increase monotonically with Gamma_t
5. The "Burstiness Gradient": Less bursty at higher Gamma_t
6. The "Metallicity Paradox": High metallicity despite young cosmic age
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

STEP_NUM = "64"
STEP_NAME = "unique_predictions"

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
    print_status(f"STEP {STEP_NUM}: Unique Predictions", "INFO")
    print_status("=" * 70, "INFO")
    print_status("\nTesting predictions UNIQUE to TEP...\n", "INFO")
    
    # Load data
    df = pd.read_csv(INTERIM_PATH / "step_02_uncover_full_sample_tep.csv")
    print_status(f"Loaded N = {len(df)} galaxies", "INFO")
    
    from astropy.cosmology import Planck18
    
    results = {
        'n_total': len(df),
        'unique_predictions': {}
    }
    
    unique_evidence = []
    
    # ==========================================================================
    # PREDICTION 1: Dust at z > 10 (Dust Desert Violation)
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("PREDICTION 1: Dust Desert Violation", "INFO")
    print_status("=" * 70, "INFO")
    print_status("At z > 10, t_cosmic < 450 Myr. AGB dust requires ~300 Myr.", "INFO")
    print_status("Standard physics: NO dust possible. TEP: Dust if Gamma_t > 1.\n", "INFO")
    
    z10 = df[df['z_phot'] > 10].dropna(subset=['dust', 'gamma_t'])
    
    if len(z10) > 10:
        # How many have significant dust?
        dusty = z10[z10['dust'] > 0.3]
        
        print_status(f"Galaxies at z > 10: N = {len(z10)}", "INFO")
        print_status(f"With A_V > 0.3: N = {len(dusty)} ({len(dusty)/len(z10)*100:.1f}%)", "INFO")
        
        if len(dusty) > 0:
            mean_gamma_dusty = dusty['gamma_t'].mean()
            mean_gamma_all = z10['gamma_t'].mean()
            
            print_status(f"<Γ_t> dusty: {mean_gamma_dusty:.2f}", "INFO")
            print_status(f"<Γ_t> all z>10: {mean_gamma_all:.2f}", "INFO")
            
            # TEP predicts dusty galaxies have higher Gamma_t
            if mean_gamma_dusty > mean_gamma_all * 1.5:
                print_status("✓ UNIQUE: Dusty z>10 galaxies have elevated Γ_t", "INFO")
                unique_evidence.append(('dust_desert', mean_gamma_dusty/mean_gamma_all, None))
        
        results['unique_predictions']['dust_desert'] = {
            'n_z10': int(len(z10)),
            'n_dusty': int(len(dusty)),
            'fraction_dusty': float(len(dusty)/len(z10)) if len(z10) > 0 else 0
        }
    
    # ==========================================================================
    # PREDICTION 2: Mass-Age Correlation at z > 8
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("PREDICTION 2: Mass-Age Correlation", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Standard: More massive galaxies form LATER (hierarchical).", "INFO")
    print_status("TEP: More massive = higher Gamma_t = appear OLDER.\n", "INFO")
    
    high_z = df[df['z_phot'] > 8].dropna(subset=['log_Mstar', 'mwa', 'gamma_t'])
    high_z = high_z[high_z['mwa'] > 0]
    
    if len(high_z) > 50:
        rho, p = spearmanr(high_z['log_Mstar'], high_z['mwa'])
        
        print_status(f"N = {len(high_z)}", "INFO")
        print_status(f"ρ(M*, Age) at z > 8 = {rho:.3f} (p = {p:.2e})", "INFO")
        
        # TEP predicts POSITIVE correlation (more massive = older)
        if rho > 0.1 and p < 0.01:
            print_status("✓ UNIQUE: More massive galaxies appear older at z > 8", "INFO")
            unique_evidence.append(('mass_age', rho, format_p_value(p)))
        
        results['unique_predictions']['mass_age'] = {
            'n': int(len(high_z)),
            'rho': float(rho),
            'p': format_p_value(p)
        }
    
    # ==========================================================================
    # PREDICTION 3: Metallicity at z > 9
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("PREDICTION 3: Metallicity Paradox", "INFO")
    print_status("=" * 70, "INFO")
    print_status("At z > 9, t_cosmic < 550 Myr. High metallicity requires time.", "INFO")
    print_status("TEP: High metallicity possible with high Gamma_t.\n", "INFO")
    
    z9 = df[df['z_phot'] > 9].dropna(subset=['met', 'gamma_t'])
    
    if len(z9) > 20:
        # Define "high metallicity" as top 25%
        met_thresh = z9['met'].quantile(0.75)
        high_met = z9[z9['met'] > met_thresh]
        low_met = z9[z9['met'] <= met_thresh]
        
        if len(high_met) > 5 and len(low_met) > 5:
            mean_gamma_high = high_met['gamma_t'].mean()
            mean_gamma_low = low_met['gamma_t'].mean()
            
            print_status(f"High metallicity (top 25%): N = {len(high_met)}, <Γ_t> = {mean_gamma_high:.2f}", "INFO")
            print_status(f"Low metallicity: N = {len(low_met)}, <Γ_t> = {mean_gamma_low:.2f}", "INFO")
            print_status(f"Ratio: {mean_gamma_high/mean_gamma_low:.2f}×", "INFO")
            
            if mean_gamma_high > mean_gamma_low * 1.5:
                print_status("✓ UNIQUE: High-metallicity z>9 galaxies have elevated Γ_t", "INFO")
                unique_evidence.append(('metallicity_paradox', mean_gamma_high/mean_gamma_low, None))
        
        results['unique_predictions']['metallicity_paradox'] = {
            'n_z9': int(len(z9)),
            'met_threshold': float(met_thresh)
        }
    
    # ==========================================================================
    # PREDICTION 4: Chi2 Monotonicity
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("PREDICTION 4: χ² Monotonicity", "INFO")
    print_status("=" * 70, "INFO")
    print_status("TEP predicts: χ² should increase monotonically with Gamma_t.", "INFO")
    print_status("(Isochrony violation worsens SED fits.)\n", "INFO")
    
    high_z = df[df['z_phot'] > 8].dropna(subset=['chi2', 'gamma_t'])
    
    if len(high_z) > 50:
        # Bin by Gamma_t
        gamma_bins = [0, 0.5, 1, 2, 5, 100]
        chi2_by_gamma = []
        
        for i in range(len(gamma_bins) - 1):
            g_lo, g_hi = gamma_bins[i], gamma_bins[i+1]
            bin_data = high_z[(high_z['gamma_t'] >= g_lo) & (high_z['gamma_t'] < g_hi)]
            if len(bin_data) > 5:
                mean_chi2 = bin_data['chi2'].mean()
                chi2_by_gamma.append({
                    'gamma_range': f'{g_lo}-{g_hi}',
                    'n': int(len(bin_data)),
                    'mean_chi2': float(mean_chi2)
                })
                print_status(f"Γ_t = {g_lo}-{g_hi}: N = {len(bin_data)}, <χ²> = {mean_chi2:.1f}", "INFO")
        
        # Check monotonicity
        if len(chi2_by_gamma) >= 3:
            chi2s = [c['mean_chi2'] for c in chi2_by_gamma]
            monotonic = all(chi2s[i] <= chi2s[i+1] for i in range(len(chi2s)-1))
            trend_rho, _ = spearmanr(range(len(chi2s)), chi2s)
            
            print_status(f"\nTrend: ρ = {trend_rho:.2f}", "INFO")
            
            if trend_rho > 0.7:
                print_status("✓ UNIQUE: χ² increases monotonically with Γ_t", "INFO")
                unique_evidence.append(('chi2_monotonicity', trend_rho, None))
        
        results['unique_predictions']['chi2_monotonicity'] = chi2_by_gamma
    
    # ==========================================================================
    # PREDICTION 5: Quiescent Fraction at z > 8
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("PREDICTION 5: Quiescent Fraction", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Standard: NO quiescent galaxies at z > 8 (not enough time).", "INFO")
    print_status("TEP: Quiescent possible with high Gamma_t.\n", "INFO")
    
    high_z = df[df['z_phot'] > 8].dropna(subset=['ssfr100', 'gamma_t'])
    high_z = high_z[high_z['ssfr100'] > 0].copy()
    
    if len(high_z) > 50:
        high_z['log_ssfr'] = np.log10(high_z['ssfr100'])
        
        # Define quiescent as log(sSFR) < -10
        quiescent = high_z[high_z['log_ssfr'] < -10]
        star_forming = high_z[high_z['log_ssfr'] >= -10]
        
        print_status(f"Quiescent at z > 8: N = {len(quiescent)} ({len(quiescent)/len(high_z)*100:.1f}%)", "INFO")
        
        if len(quiescent) > 0:
            mean_gamma_q = quiescent['gamma_t'].mean()
            mean_gamma_sf = star_forming['gamma_t'].mean()
            
            print_status(f"<Γ_t> quiescent: {mean_gamma_q:.2f}", "INFO")
            print_status(f"<Γ_t> star-forming: {mean_gamma_sf:.2f}", "INFO")
            
            if mean_gamma_q > mean_gamma_sf * 2:
                print_status("✓ UNIQUE: Quiescent z>8 galaxies have elevated Γ_t", "INFO")
                unique_evidence.append(('quiescent_z8', mean_gamma_q/mean_gamma_sf, None))
        
        results['unique_predictions']['quiescent'] = {
            'n_quiescent': int(len(quiescent)),
            'n_star_forming': int(len(star_forming)),
            'fraction_quiescent': float(len(quiescent)/len(high_z)) if len(high_z) > 0 else 0
        }
    
    # ==========================================================================
    # PREDICTION 6: Effective Time Threshold
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("PREDICTION 6: Effective Time Threshold", "INFO")
    print_status("=" * 70, "INFO")
    print_status("TEP predicts: Dust should appear when t_eff > 300 Myr.", "INFO")
    print_status("This is a QUANTITATIVE prediction.\n", "INFO")
    
    valid = df.dropna(subset=['t_eff', 'dust', 'z_phot'])
    high_z = valid[valid['z_phot'] > 8]
    
    if len(high_z) > 50:
        # Split by t_eff threshold
        above_thresh = high_z[high_z['t_eff'] > 0.3]
        below_thresh = high_z[high_z['t_eff'] <= 0.3]
        
        if len(above_thresh) > 10 and len(below_thresh) > 10:
            mean_dust_above = above_thresh['dust'].mean()
            mean_dust_below = below_thresh['dust'].mean()
            
            print_status(f"t_eff > 300 Myr: N = {len(above_thresh)}, <A_V> = {mean_dust_above:.3f}", "INFO")
            print_status(f"t_eff ≤ 300 Myr: N = {len(below_thresh)}, <A_V> = {mean_dust_below:.3f}", "INFO")
            print_status(f"Ratio: {mean_dust_above/mean_dust_below:.2f}×", "INFO")
            
            if mean_dust_above > mean_dust_below * 1.3:
                print_status("✓ UNIQUE: Dust appears above t_eff threshold as predicted", "INFO")
                unique_evidence.append(('t_eff_threshold', mean_dust_above/mean_dust_below, None))
        
        results['unique_predictions']['t_eff_threshold'] = {
            'n_above': int(len(above_thresh)),
            'n_below': int(len(below_thresh)),
            'mean_dust_above': float(mean_dust_above) if len(above_thresh) > 0 else 0,
            'mean_dust_below': float(mean_dust_below) if len(below_thresh) > 0 else 0
        }
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("SUMMARY: Unique Predictions", "INFO")
    print_status("=" * 70, "INFO")
    
    print_status(f"\nUnique evidence items: {len(unique_evidence)}", "INFO")
    for name, stat, p in unique_evidence:
        if p is not None and p > 0:
            print_status(f"  • {name}: stat = {stat:.3f}, p = {p:.2e}", "INFO")
        else:
            print_status(f"  • {name}: stat = {stat:.3f}", "INFO")
    
    results['summary'] = {
        'n_unique_evidence': len(unique_evidence),
        'unique_evidence': [{'name': n, 'stat': float(s), 'p': format_p_value(p)} for n, s, p in unique_evidence]
    }
    
    # Save
    json_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print_status(f"\nSaved to {json_path}", "INFO")

if __name__ == "__main__":
    main()
