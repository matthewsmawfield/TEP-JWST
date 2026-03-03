#!/usr/bin/env python3
"""
TEP-JWST Step 72: The Ultimate Missing Piece

What would make TEP IRREFUTABLE? Let's think deeply:

1. THE SPECTROSCOPIC TEST: Do galaxies with SPECTROSCOPIC redshifts show the same effect?
   - Photo-z errors could create spurious correlations
   - Spec-z removes this systematic entirely
   
2. THE INTERNAL GRADIENT TEST: Do galaxy CORES differ from OUTSKIRTS?
   - TEP predicts: Core (high density) = screened = young
   - Outskirts (low density) = unscreened = old
   - This is a UNIQUE prediction that standard physics cannot mimic
   
3. THE BALMER BREAK TEST: Does the Balmer break strength correlate with Gamma_t?
   - Balmer break is a direct age indicator independent of SED fitting
   - If TEP is real, high-Gamma_t galaxies should have stronger Balmer breaks
   
4. THE EMISSION LINE TEST: Do emission line ratios correlate with Gamma_t?
   - [OIII]/Hβ, [NII]/Hα are metallicity/age indicators
   - Independent of photometric SED fitting
   
5. THE DYNAMICAL MASS TEST: Does M_dyn differ from M_* in a TEP-predicted way?
   - TEP predicts: M_*(SED) is inflated relative to M_dyn
   - This is a QUANTITATIVE prediction
   
6. THE COSMIC VARIANCE TEST: Is the effect consistent across different fields?
   - If TEP is real, it should work in UNCOVER, CEERS, COSMOS-Web, JADES
   
7. THE REDSHIFT DESERT TEST: Is there a "desert" at z ~ 6-7 where TEP is weak?
   - TEP predicts a transition zone where screening changes
"""

import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu, ks_2samp
from pathlib import Path
import json

# =============================================================================
# PATHS AND LOGGER
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.p_value_utils import format_p_value

STEP_NUM = "72"
STEP_NAME = "ultimate_missing_piece"

INTERIM_PATH = PROJECT_ROOT / "results" / "interim"
DATA_PATH = PROJECT_ROOT / "data" / "interim"
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
    print_status(f"STEP {STEP_NUM}: The Ultimate Missing Piece", "INFO")
    print_status("=" * 70, "INFO")
    print_status("\nSearching for the evidence that would make TEP IRREFUTABLE...\n", "INFO")
    
    # Load data
    df = pd.read_csv(INTERIM_PATH / "step_02_uncover_full_sample_tep.csv")
    print_status(f"Loaded N = {len(df)} galaxies", "INFO")
    
    results = {
        'n_total': int(len(df)),
        'ultimate': {}
    }
    
    # ==========================================================================
    # MISSING PIECE 1: Spectroscopic Confirmation
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("MISSING PIECE 1: Spectroscopic Confirmation", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Do galaxies with SPECTROSCOPIC redshifts show the same effect?\n", "INFO")
    
    # Check if we have spec-z data
    spec_cols = [col for col in df.columns if 'spec' in col.lower() or 'zspec' in col.lower()]
    print_status(f"Spectroscopic columns found: {spec_cols}", "INFO")
    
    # Load combined spectroscopic catalog if available
    spec_path = DATA_PATH / "combined_spectroscopic_catalog.csv"
    if spec_path.exists():
        spec_df = pd.read_csv(spec_path)
        print_status(f"Loaded spectroscopic catalog: N = {len(spec_df)}", "INFO")
        
        # Check for high-z spec-z galaxies
        if 'z_spec' in spec_df.columns:
            high_z_spec = spec_df[spec_df['z_spec'] > 7]
            print_status(f"Spectroscopic z > 7: N = {len(high_z_spec)}", "INFO")
            
            if len(high_z_spec) > 10:
                # Merge with main catalog to get Gamma_t
                # This would require matching by ID or coordinates
                print_status("Spectroscopic sample available for validation", "INFO")
    else:
        print_status("No combined spectroscopic catalog found", "INFO")
    
    # Check UNCOVER for spec-z flag
    if 'use_zspec' in df.columns:
        spec_sample = df[df['use_zspec'] == 1]
        print_status(f"UNCOVER spec-z sample: N = {len(spec_sample)}", "INFO")
        
        if len(spec_sample) > 20:
            spec_high_z = spec_sample[spec_sample['z_phot'] > 7].dropna(subset=['gamma_t', 'dust'])
            if len(spec_high_z) > 10:
                rho, p = spearmanr(spec_high_z['gamma_t'], spec_high_z['dust'])
                print_status(f"Spec-z sample (z > 7): ρ(Γ_t, Dust) = {rho:.3f} (p = {p:.2e})", "INFO")
                
                if rho > 0.3 and p < 0.05:
                    print_status("✓ SPECTROSCOPIC CONFIRMATION: Effect holds in spec-z sample", "INFO")
                
                results['ultimate']['spectroscopic'] = {
                    'n': int(len(spec_high_z)),
                    'rho': float(rho),
                    'p': format_p_value(p)
                }
    
    # ==========================================================================
    # MISSING PIECE 2: The Redshift Transition
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("MISSING PIECE 2: The Redshift Transition", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Is there a sharp transition in TEP strength at some redshift?\n", "INFO")
    
    valid = df.dropna(subset=['gamma_t', 'dust', 'z_phot'])
    
    # Fine-grained z bins
    z_bins = np.arange(4, 12, 0.5)
    z_evolution = []
    
    for i in range(len(z_bins) - 1):
        z_lo, z_hi = z_bins[i], z_bins[i+1]
        bin_data = valid[(valid['z_phot'] >= z_lo) & (valid['z_phot'] < z_hi)]
        if len(bin_data) > 15:
            rho, p = spearmanr(bin_data['gamma_t'], bin_data['dust'])
            p_fmt = format_p_value(p)
            z_evolution.append({
                'z_mid': float((z_lo + z_hi) / 2),
                'n': int(len(bin_data)),
                'rho': float(rho),
                'p': p_fmt
            })
    
    if z_evolution:
        print_status("Redshift evolution of Γ_t-Dust correlation:", "INFO")
        for r in z_evolution:
            marker = "✓" if r['rho'] > 0.3 else "○" if r['rho'] > 0 else "✗"
            print_status(f"  z = {r['z_mid']:.1f}: ρ = {r['rho']:+.3f} (N = {r['n']}) {marker}", "INFO")
        
        # Find transition point
        rhos = [r['rho'] for r in z_evolution]
        z_mids = [r['z_mid'] for r in z_evolution]
        
        # Look for where correlation becomes significant
        transition_z = None
        for i, r in enumerate(z_evolution):
            if r['rho'] > 0.3 and r.get('p') is not None and r['p'] < 0.05:
                transition_z = r['z_mid']
                break
        
        if transition_z:
            print_status(f"\n✓ TRANSITION DETECTED: TEP becomes significant at z ≈ {transition_z}", "INFO")
        
        results['ultimate']['z_transition'] = z_evolution
    
    # ==========================================================================
    # MISSING PIECE 3: The Mass-Independent Signal
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("MISSING PIECE 3: The Mass-Independent Signal", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Does TEP predict dust INDEPENDENTLY of mass?\n", "INFO")
    
    high_z = df[df['z_phot'] > 8].dropna(subset=['gamma_t', 'dust', 'log_Mstar'])
    
    if len(high_z) > 50:
        from scipy.stats import linregress
        
        # Residualize dust against mass
        slope, intercept, _, _, _ = linregress(high_z['log_Mstar'], high_z['dust'])
        high_z = high_z.copy()
        high_z['dust_resid'] = high_z['dust'] - (slope * high_z['log_Mstar'] + intercept)
        
        # Does Gamma_t predict the RESIDUAL dust?
        rho_resid, p_resid = spearmanr(high_z['gamma_t'], high_z['dust_resid'])
        
        # Also residualize Gamma_t against mass
        slope_g, intercept_g, _, _, _ = linregress(high_z['log_Mstar'], high_z['gamma_t'])
        high_z['gamma_resid'] = high_z['gamma_t'] - (slope_g * high_z['log_Mstar'] + intercept_g)
        
        # Partial correlation
        rho_partial, p_partial = spearmanr(high_z['gamma_resid'], high_z['dust_resid'])
        
        print_status(f"ρ(Γ_t, Dust_resid|M*) = {rho_resid:.3f} (p = {p_resid:.2e})", "INFO")
        print_status(f"ρ(Γ_t_resid, Dust_resid) = {rho_partial:.3f} (p = {p_partial:.2e})", "INFO")
        
        if rho_partial > 0.1 and p_partial < 0.05:
            print_status("\n✓ MASS-INDEPENDENT SIGNAL: Gamma_t predicts dust beyond mass", "INFO")
        
        results['ultimate']['mass_independent'] = {
            'rho_resid': float(rho_resid),
            'rho_partial': float(rho_partial),
            'p_partial': format_p_value(p_partial)
        }
    
    # ==========================================================================
    # MISSING PIECE 4: The Impossible Population
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("MISSING PIECE 4: The Impossible Population", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Are there galaxies that are IMPOSSIBLE under standard physics?\n", "INFO")
    
    high_z = df[df['z_phot'] > 9].dropna(subset=['gamma_t', 'dust', 'mwa', 'age_ratio'])
    
    if len(high_z) > 20:
        # Define "impossible" criteria
        # 1. Age ratio > 0.5 (apparent age > 50% of cosmic age)
        # 2. High dust (A_V > 0.5) at z > 9 where t_cosmic < 550 Myr
        # 3. High metallicity at z > 9
        
        impossible = high_z[
            (high_z['age_ratio'] > 0.5) | 
            (high_z['dust'] > 0.5)
        ]
        
        print_status(f"'Impossible' galaxies at z > 9: N = {len(impossible)}", "INFO")
        
        if len(impossible) > 3:
            mean_gamma_imp = impossible['gamma_t'].mean()
            mean_gamma_norm = high_z.drop(impossible.index)['gamma_t'].mean() if len(high_z) > len(impossible) else 0
            
            print_status(f"  <Γ_t> impossible: {mean_gamma_imp:.2f}", "INFO")
            print_status(f"  <Γ_t> normal: {mean_gamma_norm:.2f}", "INFO")
            
            if mean_gamma_norm > 0:
                ratio = mean_gamma_imp / mean_gamma_norm
                print_status(f"  Ratio: {ratio:.1f}×", "INFO")
                
                if ratio > 5:
                    print_status("\n✓ IMPOSSIBLE POPULATION: Extreme galaxies have extreme Γ_t", "INFO")
                
                results['ultimate']['impossible_population'] = {
                    'n_impossible': int(len(impossible)),
                    'mean_gamma_impossible': float(mean_gamma_imp),
                    'mean_gamma_normal': float(mean_gamma_norm),
                    'ratio': float(ratio)
                }
    
    # ==========================================================================
    # MISSING PIECE 5: The Quantitative Prediction
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("MISSING PIECE 5: The Quantitative Prediction", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Does t_eff = t_cosmic × Γ_t QUANTITATIVELY predict dust?\n", "INFO")
    
    high_z = df[df['z_phot'] > 8].dropna(subset=['t_eff', 'dust'])
    
    if len(high_z) > 50:
        # AGB dust requires ~300 Myr
        # Prediction: Dust should appear when t_eff > 0.3 Gyr
        
        # Bin by t_eff and check dust
        t_eff_bins = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 1.0), (1.0, 5.0)]
        quant_results = []
        
        for t_lo, t_hi in t_eff_bins:
            bin_data = high_z[(high_z['t_eff'] >= t_lo) & (high_z['t_eff'] < t_hi)]
            if len(bin_data) >= 5:
                mean_dust = bin_data['dust'].mean()
                quant_results.append({
                    't_eff_range': f'{t_lo}-{t_hi}',
                    't_eff_mid': (t_lo + t_hi) / 2,
                    'n': int(len(bin_data)),
                    'mean_dust': float(mean_dust)
                })
                
                # Dust should be low below 0.3 Gyr, high above
                expected = "low" if t_hi <= 0.3 else "high"
                actual = "low" if mean_dust < 0.5 else "high"
                match = "✓" if expected == actual else "✗"
                
                print_status(f"  t_eff = {t_lo}-{t_hi} Gyr: <Dust> = {mean_dust:.3f} (expected: {expected}) {match}", "INFO")
        
        # Check if dust increases with t_eff
        if len(quant_results) >= 4:
            t_effs = [r['t_eff_mid'] for r in quant_results]
            dusts = [r['mean_dust'] for r in quant_results]
            rho, p = spearmanr(t_effs, dusts)
            
            print_status(f"\nTrend: ρ(t_eff, Dust) = {rho:.3f} (p = {p:.2e})", "INFO")
            
            if rho > 0.8:
                print_status("✓ QUANTITATIVE MATCH: Dust increases with t_eff as predicted", "INFO")
            
            results['ultimate']['quantitative_prediction'] = {
                'bins': quant_results,
                'trend_rho': float(rho)
            }
    
    # ==========================================================================
    # MISSING PIECE 6: The Cross-Survey Validation
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("MISSING PIECE 6: Cross-Survey Validation", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Does the effect hold across DIFFERENT surveys?\n", "INFO")
    
    # Load CEERS data
    ceers_path = DATA_PATH / "ceers_z8_sample.csv"
    if ceers_path.exists():
        ceers = pd.read_csv(ceers_path)
        ceers_valid = ceers.dropna(subset=['log_Mstar', 'dust'])
        
        if len(ceers_valid) > 20:
            rho_ceers, p_ceers = spearmanr(ceers_valid['log_Mstar'], ceers_valid['dust'])
            
            # Compare to UNCOVER
            uncover_z8 = df[df['z_phot'] > 8].dropna(subset=['log_Mstar', 'dust'])
            rho_uncover, p_uncover = spearmanr(uncover_z8['log_Mstar'], uncover_z8['dust'])
            
            print_status(f"CEERS (z > 8): ρ(M*, Dust) = {rho_ceers:.3f} (N = {len(ceers_valid)})", "INFO")
            print_status(f"UNCOVER (z > 8): ρ(M*, Dust) = {rho_uncover:.3f} (N = {len(uncover_z8)})", "INFO")
            
            # Check consistency
            if abs(rho_ceers - rho_uncover) < 0.2:
                print_status("\n✓ CROSS-SURVEY VALIDATION: Consistent across surveys", "INFO")
            
            results['ultimate']['cross_survey'] = {
                'rho_ceers': float(rho_ceers),
                'rho_uncover': float(rho_uncover),
                'difference': float(abs(rho_ceers - rho_uncover))
            }
    
    # ==========================================================================
    # MISSING PIECE 7: The Ultimate Smoking Gun
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("MISSING PIECE 7: The Ultimate Smoking Gun", "INFO")
    print_status("=" * 70, "INFO")
    print_status("What is the ONE observation that would be IMPOSSIBLE without TEP?\n", "INFO")
    
    # The ultimate smoking gun: A galaxy at z > 10 with:
    # - High dust (A_V > 1)
    # - Old stellar population (MWA > 300 Myr)
    # - High metallicity
    # At z > 10, t_cosmic < 450 Myr, so this is IMPOSSIBLE under standard physics
    
    z10 = df[df['z_phot'] > 10].dropna(subset=['dust', 'mwa', 'gamma_t'])
    
    if len(z10) > 5:
        # Find the most extreme cases
        extreme = z10[(z10['dust'] > 0.5) | (z10['mwa'] > 200)]
        
        print_status(f"Galaxies at z > 10: N = {len(z10)}", "INFO")
        print_status(f"Extreme (dust > 0.5 OR age > 200 Myr): N = {len(extreme)}", "INFO")
        
        if len(extreme) > 0:
            print_status("\nExtreme z > 10 galaxies:", "INFO")
            for idx, row in extreme.head(5).iterrows():
                print_status(f"  ID {idx}: z = {row['z_phot']:.2f}, Dust = {row['dust']:.2f}, Γ_t = {row['gamma_t']:.2f}", "INFO")
            
            mean_gamma = extreme['gamma_t'].mean()
            print_status(f"\n<Γ_t> of extreme z>10 galaxies: {mean_gamma:.2f}", "INFO")
            
            if mean_gamma > 5:
                print_status("\n✓ SMOKING GUN: Extreme z>10 galaxies have extreme Γ_t", "INFO")
            
            results['ultimate']['smoking_gun'] = {
                'n_z10': int(len(z10)),
                'n_extreme': int(len(extreme)),
                'mean_gamma_extreme': float(mean_gamma)
            }
    
    # ==========================================================================
    # SYNTHESIS: What is the ULTIMATE missing piece?
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("SYNTHESIS: The Ultimate Missing Piece", "INFO")
    print_status("=" * 70, "INFO")
    
    print_status("\nBased on this analysis, the ULTIMATE missing pieces are:", "INFO")
    print_status("", "INFO")
    print_status("1. SPECTROSCOPIC CONFIRMATION", "INFO")
    print_status("   - Need more spec-z galaxies at z > 8 to rule out photo-z systematics", "INFO")
    print_status("   - Current spec-z sample is limited", "INFO")
    print_status("", "INFO")
    print_status("2. DYNAMICAL MASS COMPARISON", "INFO")
    print_status("   - TEP predicts M_*(SED) > M_dyn for high-Gamma_t galaxies", "INFO")
    print_status("   - This would be a QUANTITATIVE, UNIQUE prediction", "INFO")
    print_status("   - Requires IFU spectroscopy (e.g., NIRSpec)", "INFO")
    print_status("", "INFO")
    print_status("3. INTERNAL GRADIENT MEASUREMENT", "INFO")
    print_status("   - TEP predicts: Core = young, Outskirts = old", "INFO")
    print_status("   - Opposite of standard inside-out formation", "INFO")
    print_status("   - Requires resolved photometry/spectroscopy", "INFO")
    print_status("", "INFO")
    print_status("4. EMISSION LINE RATIOS", "INFO")
    print_status("   - Independent age/metallicity indicators", "INFO")
    print_status("   - Should correlate with Gamma_t", "INFO")
    print_status("   - Requires deep spectroscopy", "INFO")
    
    results['ultimate']['synthesis'] = {
        'missing_pieces': [
            'Spectroscopic confirmation at z > 8',
            'Dynamical mass comparison (M_dyn vs M_*)',
            'Internal gradient measurement (core vs outskirts)',
            'Emission line ratio correlations'
        ]
    }
    
    # Save
    json_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print_status(f"\nSaved to {json_path}", "INFO")

if __name__ == "__main__":
    main()
