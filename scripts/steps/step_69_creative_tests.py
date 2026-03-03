#!/usr/bin/env python3
"""
TEP-JWST Step 69: Creative Tests

This step explores creative, unconventional angles:

1. THE "TWIN" TEST: Galaxies with similar mass but different Gamma_t should differ
2. THE "GRADIENT" TEST: Does the effect have spatial structure?
3. THE "PHASE SPACE" TEST: Where in property space does TEP work best?
4. THE "PREDICTION" TEST: Can we predict NEW observables from Gamma_t?
5. THE "CONSISTENCY" TEST: Do different age indicators agree?
6. THE "THRESHOLD" TEST: Is there a critical Gamma_t where effects appear?
"""

import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ks_2samp, mannwhitneyu
from pathlib import Path
import json

# =============================================================================
# PATHS AND LOGGER
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.p_value_utils import format_p_value

STEP_NUM = "69"
STEP_NAME = "creative_tests"

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
    print_status(f"STEP {STEP_NUM}: Creative Tests", "INFO")
    print_status("=" * 70, "INFO")
    print_status("\nExploring creative, unconventional angles...\n", "INFO")
    
    # Load data
    df = pd.read_csv(INTERIM_PATH / "step_02_uncover_full_sample_tep.csv")
    print_status(f"Loaded N = {len(df)} galaxies", "INFO")
    
    results = {
        'n_total': int(len(df)),
        'creative': {}
    }
    
    high_z = df[df['z_phot'] > 8].dropna(subset=['log_Mstar', 'dust', 'gamma_t', 'z_phot', 'mwa', 'chi2'])
    
    # ==========================================================================
    # TEST 1: The "Twin" Test
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("CREATIVE TEST 1: The 'Twin' Test", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Galaxies with SIMILAR mass but DIFFERENT Gamma_t should differ in dust.\n", "INFO")
    
    if len(high_z) > 50:
        # Find "twins" - galaxies with similar mass (within 0.2 dex)
        high_z = high_z.copy()
        high_z['mass_bin'] = (high_z['log_Mstar'] / 0.2).astype(int)
        
        twin_results = []
        
        for mass_bin in high_z['mass_bin'].unique():
            twins = high_z[high_z['mass_bin'] == mass_bin]
            if len(twins) >= 10:
                # Split by Gamma_t
                gamma_median = twins['gamma_t'].median()
                high_gamma = twins[twins['gamma_t'] > gamma_median]
                low_gamma = twins[twins['gamma_t'] <= gamma_median]
                
                if len(high_gamma) >= 3 and len(low_gamma) >= 3:
                    dust_high = high_gamma['dust'].mean()
                    dust_low = low_gamma['dust'].mean()
                    
                    twin_results.append({
                        'mass_bin': float(mass_bin * 0.2),
                        'n_twins': int(len(twins)),
                        'dust_high_gamma': float(dust_high),
                        'dust_low_gamma': float(dust_low),
                        'ratio': float(dust_high / dust_low) if dust_low > 0 else 0
                    })
        
        if twin_results:
            avg_ratio = np.mean([r['ratio'] for r in twin_results if r['ratio'] > 0])
            print_status(f"Average dust ratio (high Γ_t / low Γ_t) at fixed mass: {avg_ratio:.2f}×", "INFO")
            
            if avg_ratio > 1.2:
                print_status("\n✓ Twins with higher Γ_t have more dust → Mass is not the only factor", "INFO")
        
        results['creative']['twin_test'] = twin_results
    
    # ==========================================================================
    # TEST 2: The "Threshold" Test
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("CREATIVE TEST 2: The 'Threshold' Test", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Is there a critical Gamma_t where TEP effects become significant?\n", "INFO")
    
    if len(high_z) > 50:
        # Test different Gamma_t thresholds
        thresholds = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
        threshold_results = []
        
        for thresh in thresholds:
            above = high_z[high_z['gamma_t'] > thresh]
            below = high_z[high_z['gamma_t'] <= thresh]
            
            if len(above) >= 5 and len(below) >= 5:
                dust_above = above['dust'].mean()
                dust_below = below['dust'].mean()
                
                # Mann-Whitney test
                stat, p = mannwhitneyu(above['dust'], below['dust'], alternative='greater')
                
                threshold_results.append({
                    'threshold': float(thresh),
                    'n_above': int(len(above)),
                    'n_below': int(len(below)),
                    'dust_above': float(dust_above),
                    'dust_below': float(dust_below),
                    'ratio': float(dust_above / dust_below) if dust_below > 0 else 0,
                    'p': format_p_value(p)
                })
                
                sig = "✓" if p < 0.01 else ""
                print_status(f"Γ_t > {thresh}: N = {len(above)}, <Dust> = {dust_above:.3f} vs {dust_below:.3f} (p = {p:.2e}) {sig}", "INFO")
        
        results['creative']['threshold_test'] = threshold_results
    
    # ==========================================================================
    # TEST 3: The "Phase Space" Test
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("CREATIVE TEST 3: The 'Phase Space' Test", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Where in property space does TEP work best?\n", "INFO")
    
    if len(high_z) > 50:
        # Test correlation in different regions of phase space
        regions = [
            ('High mass', high_z['log_Mstar'] > high_z['log_Mstar'].median()),
            ('Low mass', high_z['log_Mstar'] <= high_z['log_Mstar'].median()),
            ('High z', high_z['z_phot'] > high_z['z_phot'].median()),
            ('Low z', high_z['z_phot'] <= high_z['z_phot'].median()),
        ]
        
        phase_results = []
        
        for name, mask in regions:
            subset = high_z[mask]
            if len(subset) > 20:
                rho, p = spearmanr(subset['gamma_t'], subset['dust'])
                phase_results.append({
                    'region': name,
                    'n': int(len(subset)),
                    'rho': float(rho),
                    'p': format_p_value(p)
                })
                sig = "✓" if rho > 0.3 and p < 0.01 else ""
                print_status(f"{name}: ρ = {rho:.3f} (p = {p:.2e}) {sig}", "INFO")
        
        results['creative']['phase_space'] = phase_results
    
    # ==========================================================================
    # TEST 4: The "Multi-Observable" Test
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("CREATIVE TEST 4: The 'Multi-Observable' Test", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Does Gamma_t predict MULTIPLE observables simultaneously?\n", "INFO")
    
    if len(high_z) > 50:
        observables = ['dust', 'mwa', 'chi2']
        multi_results = []
        
        for obs in observables:
            rho, p = spearmanr(high_z['gamma_t'], high_z[obs])
            multi_results.append({
                'observable': obs,
                'rho': float(rho),
                'p': format_p_value(p),
                'significant': bool(p < 0.05)
            })
            sig = "✓" if p < 0.05 else ""
            print_status(f"ρ(Γ_t, {obs}) = {rho:.3f} (p = {p:.2e}) {sig}", "INFO")
        
        n_significant = sum(1 for r in multi_results if r['significant'])
        print_status(f"\nSignificant correlations: {n_significant}/{len(observables)}", "INFO")
        
        if n_significant >= 2:
            print_status("✓ Gamma_t predicts multiple observables → Unified explanation", "INFO")
        
        results['creative']['multi_observable'] = multi_results
    
    # ==========================================================================
    # TEST 5: The "Rank Preservation" Test
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("CREATIVE TEST 5: The 'Rank Preservation' Test", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Do galaxies maintain their rank across different observables?\n", "INFO")
    
    if len(high_z) > 50:
        # Rank galaxies by different properties
        high_z = high_z.copy()
        high_z['gamma_rank'] = high_z['gamma_t'].rank(pct=True)
        high_z['dust_rank'] = high_z['dust'].rank(pct=True)
        high_z['age_rank'] = high_z['mwa'].rank(pct=True)
        high_z['chi2_rank'] = high_z['chi2'].rank(pct=True)
        
        # Correlation between ranks
        rho_dust, _ = spearmanr(high_z['gamma_rank'], high_z['dust_rank'])
        rho_age, _ = spearmanr(high_z['gamma_rank'], high_z['age_rank'])
        rho_chi2, _ = spearmanr(high_z['gamma_rank'], high_z['chi2_rank'])
        
        print_status(f"Rank correlation (Γ_t vs Dust): {rho_dust:.3f}", "INFO")
        print_status(f"Rank correlation (Γ_t vs Age): {rho_age:.3f}", "INFO")
        print_status(f"Rank correlation (Γ_t vs χ²): {rho_chi2:.3f}", "INFO")
        
        results['creative']['rank_preservation'] = {
            'rho_dust': float(rho_dust),
            'rho_age': float(rho_age),
            'rho_chi2': float(rho_chi2)
        }
    
    # ==========================================================================
    # TEST 6: The "Extreme Concordance" Test
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("CREATIVE TEST 6: The 'Extreme Concordance' Test", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Do the TOP galaxies in Gamma_t also top in dust, age, chi2?\n", "INFO")
    
    if len(high_z) > 50:
        # Get top 10% in each property
        gamma_top = set(high_z.nlargest(int(len(high_z) * 0.1), 'gamma_t').index)
        dust_top = set(high_z.nlargest(int(len(high_z) * 0.1), 'dust').index)
        age_top = set(high_z.nlargest(int(len(high_z) * 0.1), 'mwa').index)
        chi2_top = set(high_z.nlargest(int(len(high_z) * 0.1), 'chi2').index)
        
        # Overlap
        overlap_dust = len(gamma_top & dust_top) / len(gamma_top)
        overlap_age = len(gamma_top & age_top) / len(gamma_top)
        overlap_chi2 = len(gamma_top & chi2_top) / len(gamma_top)
        
        print_status(f"Overlap (top 10% Γ_t ∩ top 10% Dust): {overlap_dust*100:.1f}%", "INFO")
        print_status(f"Overlap (top 10% Γ_t ∩ top 10% Age): {overlap_age*100:.1f}%", "INFO")
        print_status(f"Overlap (top 10% Γ_t ∩ top 10% χ²): {overlap_chi2*100:.1f}%", "INFO")
        
        # Expected by chance: 10%
        if overlap_dust > 0.3:
            print_status("\n✓ High concordance between Γ_t and dust extremes", "INFO")
        
        results['creative']['extreme_concordance'] = {
            'overlap_dust': float(overlap_dust),
            'overlap_age': float(overlap_age),
            'overlap_chi2': float(overlap_chi2)
        }
    
    # ==========================================================================
    # TEST 7: The "Residual Pattern" Test
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("CREATIVE TEST 7: The 'Residual Pattern' Test", "INFO")
    print_status("=" * 70, "INFO")
    print_status("After removing Gamma_t effect, are residuals random?\n", "INFO")
    
    if len(high_z) > 50:
        from scipy.stats import linregress
        
        # Fit dust vs Gamma_t
        slope, intercept, _, _, _ = linregress(high_z['gamma_t'], high_z['dust'])
        high_z = high_z.copy()
        high_z['dust_resid'] = high_z['dust'] - (slope * high_z['gamma_t'] + intercept)
        
        # Check if residuals correlate with anything
        rho_mass, p_mass = spearmanr(high_z['log_Mstar'], high_z['dust_resid'])
        rho_z, p_z = spearmanr(high_z['z_phot'], high_z['dust_resid'])
        
        print_status(f"Residual correlation with M*: ρ = {rho_mass:.3f} (p = {p_mass:.2e})", "INFO")
        print_status(f"Residual correlation with z: ρ = {rho_z:.3f} (p = {p_z:.2e})", "INFO")
        
        # If residuals are random, correlations should be weak
        if abs(rho_mass) < 0.2 and abs(rho_z) < 0.2:
            print_status("\n✓ Residuals are approximately random → Gamma_t captures the signal", "INFO")
        
        results['creative']['residual_pattern'] = {
            'rho_mass': float(rho_mass),
            'rho_z': float(rho_z)
        }
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("SUMMARY: Creative Tests", "INFO")
    print_status("=" * 70, "INFO")
    
    tests_passed = 0
    tests_total = 7
    
    # Count passed tests
    twin = results['creative'].get('twin_test', [])
    if twin and np.mean([r['ratio'] for r in twin if r['ratio'] > 0]) > 1.2:
        tests_passed += 1
    
    thresh = results['creative'].get('threshold_test', [])
    if thresh and any(r.get('p') is not None and r['p'] < 0.01 for r in thresh):
        tests_passed += 1
    
    phase = results['creative'].get('phase_space', [])
    if phase and sum(1 for r in phase if r['rho'] > 0.3) >= 2:
        tests_passed += 1
    
    multi = results['creative'].get('multi_observable', [])
    if multi and sum(1 for r in multi if r['significant']) >= 2:
        tests_passed += 1
    
    rank = results['creative'].get('rank_preservation', {})
    if rank.get('rho_dust', 0) > 0.3:
        tests_passed += 1
    
    extreme = results['creative'].get('extreme_concordance', {})
    if extreme.get('overlap_dust', 0) > 0.3:
        tests_passed += 1
    
    resid = results['creative'].get('residual_pattern', {})
    if abs(resid.get('rho_mass', 1)) < 0.3:
        tests_passed += 1
    
    print_status(f"\nCreative tests passed: {tests_passed}/{tests_total}", "INFO")
    
    results['summary'] = {
        'tests_passed': tests_passed,
        'tests_total': tests_total
    }
    
    # Save
    json_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print_status(f"\nSaved to {json_path}", "INFO")

if __name__ == "__main__":
    main()
