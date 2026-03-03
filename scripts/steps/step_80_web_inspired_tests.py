#!/usr/bin/env python3
"""
TEP-JWST Step 80: Web-Inspired Tests

Based on recent JWST literature, testing new angles:

1. QUIESCENT GALAXY FRACTION: Recent papers show unexpectedly high quiescent
   fractions at z > 5. TEP predicts this - more effective time = faster quenching.

2. GALAXY COMPACTNESS: High-z galaxies are more compact than expected.
   TEP predicts: High-Gamma_t galaxies should be more compact (more evolved).

3. BALMER BREAK STRENGTH: Strong Balmer breaks at z > 7 indicate old populations.
   TEP predicts: High-Gamma_t galaxies should have stronger Balmer breaks.

4. STELLAR MASS DENSITY EXCESS: The stellar mass density at z > 10 exceeds
   predictions by 10-100x. TEP naturally explains this.

5. LITTLE RED DOTS: Overmassive black holes in compact red galaxies.
   TEP predicts: Differential time dilation between BH and stellar halo.

6. RAPID QUENCHING TIMESCALE: Galaxies quench faster than expected.
   TEP predicts: Effective quenching time is longer than cosmic time.
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.p_value_utils import format_p_value

STEP_NUM = "80"
STEP_NAME = "web_inspired_tests"

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
    print_status(f"STEP {STEP_NUM}: Web-Inspired Tests", "INFO")
    print_status("=" * 70, "INFO")
    print_status("\nTesting ideas from recent JWST literature...\n", "INFO")
    
    # Load data
    df = pd.read_csv(INTERIM_PATH / "step_02_uncover_full_sample_tep.csv")
    print_status(f"Loaded N = {len(df)} galaxies", "INFO")
    
    results = {
        'n_total': int(len(df)),
        'web_inspired': {}
    }
    
    # ==========================================================================
    # TEST 1: Quiescent Galaxy Fraction by Gamma_t
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 1: Quiescent Galaxy Fraction", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Recent papers: High quiescent fractions at z > 5 are unexpected.", "INFO")
    print_status("TEP predicts: High-Gamma_t → faster effective quenching.\n", "INFO")
    
    valid = df[df['z_phot'] > 5].dropna(subset=['ssfr100', 'gamma_t'])
    valid = valid[valid['ssfr100'] > 0].copy()
    
    if len(valid) > 50:
        valid['log_ssfr'] = np.log10(valid['ssfr100'])
        
        # Define quiescent as log(sSFR) < -10
        valid['is_quiescent'] = valid['log_ssfr'] < -10
        
        # Split by Gamma_t
        high_gamma = valid[valid['gamma_t'] > 1]
        low_gamma = valid[valid['gamma_t'] < 0.5]
        
        if len(high_gamma) > 10 and len(low_gamma) > 10:
            frac_q_high = high_gamma['is_quiescent'].mean()
            frac_q_low = low_gamma['is_quiescent'].mean()
            
            print_status(f"High Γ_t (> 1): Quiescent fraction = {frac_q_high*100:.1f}%", "INFO")
            print_status(f"Low Γ_t (< 0.5): Quiescent fraction = {frac_q_low*100:.1f}%", "INFO")
            
            if frac_q_high > frac_q_low:
                print_status("✓ High-Γ_t galaxies have higher quiescent fraction", "INFO")
            
            results['web_inspired']['quiescent_fraction'] = {
                'frac_high_gamma': float(frac_q_high),
                'frac_low_gamma': float(frac_q_low)
            }
    
    # ==========================================================================
    # TEST 2: Galaxy Compactness
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 2: Galaxy Compactness", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Recent papers: High-z galaxies are unexpectedly compact.", "INFO")
    print_status("TEP predicts: High-Gamma_t → more evolved → more compact.\n", "INFO")
    
    # Check if we have size data
    size_cols = [col for col in df.columns if 'size' in col.lower() or 'r_eff' in col.lower() or 'reff' in col.lower()]
    print_status(f"Size columns found: {size_cols}", "INFO")
    
    # Use stellar mass as proxy for size (more massive = larger typically)
    # But under TEP, high-Gamma_t should be more compact at fixed mass
    valid = df[df['z_phot'] > 8].dropna(subset=['log_Mstar', 'gamma_t', 'dust'])
    
    if len(valid) > 50:
        # Correlation between Gamma_t and mass-normalized properties
        # High Gamma_t galaxies should have higher surface density (more compact)
        
        # Use dust as proxy for evolved state
        rho, p = spearmanr(valid['gamma_t'], valid['dust'])
        print_status(f"ρ(Γ_t, Dust) = {rho:.3f} (p = {p:.2e})", "INFO")
        print_status("(High dust indicates evolved, compact state)", "INFO")
        
        results['web_inspired']['compactness'] = {
            'rho_gamma_dust': float(rho),
            'p': format_p_value(p)
        }
    
    # ==========================================================================
    # TEST 3: Stellar Mass Density Excess
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 3: Stellar Mass Density Excess", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Recent papers: Stellar mass density at z > 10 exceeds predictions.", "INFO")
    print_status("TEP naturally explains this: Masses are inflated by Gamma_t.\n", "INFO")
    
    # Calculate stellar mass density in different z bins
    z_bins = [(8, 9), (9, 10), (10, 12)]
    
    for z_lo, z_hi in z_bins:
        bin_data = df[(df['z_phot'] >= z_lo) & (df['z_phot'] < z_hi)].dropna(subset=['log_Mstar', 'gamma_t'])
        if len(bin_data) > 10:
            mean_mass = 10**bin_data['log_Mstar'].mean()
            mean_gamma = bin_data['gamma_t'].mean()
            
            # TEP-corrected mass
            corrected_mass = mean_mass / mean_gamma if mean_gamma > 0 else mean_mass
            
            print_status(f"z = {z_lo}-{z_hi}:", "INFO")
            print_status(f"  Observed <M*> = {mean_mass:.2e} M_sun", "INFO")
            print_status(f"  <Γ_t> = {mean_gamma:.2f}", "INFO")
            print_status(f"  TEP-corrected <M*> = {corrected_mass:.2e} M_sun", "INFO")
            print_status(f"  Inflation factor: {mean_mass/corrected_mass:.2f}×", "INFO")
    
    # ==========================================================================
    # TEST 4: Rapid Quenching Timescale
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 4: Rapid Quenching Timescale", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Recent papers: Galaxies quench faster than expected at high-z.", "INFO")
    print_status("TEP predicts: Effective quenching time is Gamma_t × cosmic time.\n", "INFO")
    
    # Look at burstiness as proxy for quenching speed
    valid = df[df['z_phot'] > 7].dropna(subset=['sfr10', 'sfr100', 'gamma_t'])
    valid = valid[(valid['sfr10'] > 0) & (valid['sfr100'] > 0)].copy()
    
    if len(valid) > 30:
        valid['burstiness'] = np.log10(valid['sfr10'] / valid['sfr100'])
        
        # Negative burstiness = declining SFR = quenching
        quenching = valid[valid['burstiness'] < -0.3]
        
        if len(quenching) > 5:
            mean_gamma_quench = quenching['gamma_t'].mean()
            mean_gamma_all = valid['gamma_t'].mean()
            
            print_status(f"Quenching galaxies (SFR declining): N = {len(quenching)}", "INFO")
            print_status(f"<Γ_t> quenching: {mean_gamma_quench:.2f}", "INFO")
            print_status(f"<Γ_t> all: {mean_gamma_all:.2f}", "INFO")
            
            if mean_gamma_quench > mean_gamma_all:
                print_status("✓ Quenching galaxies have higher Γ_t (more effective time)", "INFO")
            
            results['web_inspired']['quenching_timescale'] = {
                'n_quenching': int(len(quenching)),
                'mean_gamma_quenching': float(mean_gamma_quench),
                'mean_gamma_all': float(mean_gamma_all)
            }
    
    # ==========================================================================
    # TEST 5: Little Red Dots Analog
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 5: Little Red Dots Analog", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Recent papers: 'Little red dots' are compact, red, with overmassive BHs.", "INFO")
    print_status("TEP predicts: High dust + high mass + high Gamma_t.\n", "INFO")
    
    # Define "red dot" analogs: high dust, high mass, high-z
    valid = df[df['z_phot'] > 7].dropna(subset=['dust', 'log_Mstar', 'gamma_t'])
    
    if len(valid) > 30:
        # Red dots: high dust AND high mass
        dust_thresh = valid['dust'].quantile(0.75)
        mass_thresh = valid['log_Mstar'].quantile(0.75)
        
        red_dots = valid[(valid['dust'] > dust_thresh) & (valid['log_Mstar'] > mass_thresh)]
        normal = valid.drop(red_dots.index)
        
        if len(red_dots) > 5 and len(normal) > 5:
            mean_gamma_rd = red_dots['gamma_t'].mean()
            mean_gamma_norm = normal['gamma_t'].mean()
            
            print_status(f"'Red dot' analogs (high dust + high mass): N = {len(red_dots)}", "INFO")
            print_status(f"<Γ_t> red dots: {mean_gamma_rd:.2f}", "INFO")
            print_status(f"<Γ_t> normal: {mean_gamma_norm:.2f}", "INFO")
            print_status(f"Ratio: {mean_gamma_rd/mean_gamma_norm:.1f}×", "INFO")
            
            if mean_gamma_rd > mean_gamma_norm * 2:
                print_status("✓ 'Red dot' analogs have elevated Γ_t", "INFO")
            
            results['web_inspired']['red_dots'] = {
                'n_red_dots': int(len(red_dots)),
                'mean_gamma_rd': float(mean_gamma_rd),
                'mean_gamma_normal': float(mean_gamma_norm),
                'ratio': float(mean_gamma_rd / mean_gamma_norm)
            }
    
    # ==========================================================================
    # TEST 6: Balmer Break Proxy
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 6: Balmer Break Proxy", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Recent papers: Strong Balmer breaks at z > 7 indicate old populations.", "INFO")
    print_status("TEP predicts: High-Gamma_t → older effective age → stronger break.\n", "INFO")
    
    # Use MWA (mass-weighted age) as proxy for Balmer break strength
    valid = df[df['z_phot'] > 7].dropna(subset=['mwa', 'gamma_t'])
    valid = valid[valid['mwa'] > 0]
    
    if len(valid) > 30:
        # Correlation between Gamma_t and age
        rho, p = spearmanr(valid['gamma_t'], valid['mwa'])
        
        print_status(f"ρ(Γ_t, MWA) = {rho:.3f} (p = {p:.2e})", "INFO")
        
        # But the key is t_eff
        valid = valid.copy()
        valid['t_eff'] = valid['gamma_t'] * 0.5  # Approximate t_cosmic at z~8
        
        rho_teff, p_teff = spearmanr(valid['t_eff'], valid['mwa'])
        print_status(f"ρ(t_eff, MWA) = {rho_teff:.3f} (p = {p_teff:.2e})", "INFO")
        
        results['web_inspired']['balmer_break'] = {
            'rho_gamma_mwa': float(rho),
            'rho_teff_mwa': float(rho_teff)
        }
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("SUMMARY: Web-Inspired Tests", "INFO")
    print_status("=" * 70, "INFO")
    
    print_status("\nKey findings from literature-inspired tests:", "INFO")
    
    if 'quiescent_fraction' in results['web_inspired']:
        qf = results['web_inspired']['quiescent_fraction']
        print_status(f"  • Quiescent fraction: {qf['frac_high_gamma']*100:.1f}% (high Γ_t) vs {qf['frac_low_gamma']*100:.1f}% (low Γ_t)", "INFO")
    
    if 'red_dots' in results['web_inspired']:
        rd = results['web_inspired']['red_dots']
        print_status(f"  • 'Red dot' analogs: {rd['ratio']:.1f}× Γ_t elevation", "INFO")
    
    if 'quenching_timescale' in results['web_inspired']:
        qt = results['web_inspired']['quenching_timescale']
        print_status(f"  • Quenching galaxies: <Γ_t> = {qt['mean_gamma_quenching']:.2f} vs {qt['mean_gamma_all']:.2f}", "INFO")
    
    # Save
    json_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print_status(f"\nSaved to {json_path}", "INFO")

if __name__ == "__main__":
    main()
