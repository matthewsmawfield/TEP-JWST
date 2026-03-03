#!/usr/bin/env python3
"""
TEP-JWST Step 52: Independent Tests for Strong Evidence

This step explores additional independent tests that could provide
strong evidence for TEP:

1. Photometric Uncertainty Scaling: High-Gamma_t galaxies should have larger uncertainties
2. Redshift Uncertainty Scaling: Photo-z errors should correlate with Gamma_t
3. Mass Uncertainty Scaling: Stellar mass errors should correlate with Gamma_t
4. Age Uncertainty Scaling: Age errors should correlate with Gamma_t
5. Spectroscopic vs Photometric Offset: z_spec - z_phot should correlate with Gamma_t
6. Chi2 at Fixed Properties: After controlling for all observables, chi2 should still correlate
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

STEP_NUM = "52"
STEP_NAME = "independent_tests"

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
    print_status(f"STEP {STEP_NUM}: Independent Tests for Strong Evidence", "INFO")
    print_status("=" * 70, "INFO")
    
    # Load data
    df = pd.read_csv(INTERIM_PATH / "step_02_uncover_full_sample_tep.csv")
    print_status(f"\nLoaded N = {len(df)} galaxies", "INFO")
    
    results = {
        'n_total': len(df),
        'tests': {}
    }
    
    strong_evidence = []
    
    # ==========================================================================
    # TEST 1: Redshift Uncertainty
    # ==========================================================================
    print_status("\n" + "=" * 60, "INFO")
    print_status("TEST 1: Redshift Uncertainty", "INFO")
    print_status("=" * 60, "INFO")
    
    # Calculate z uncertainty from z_16 and z_84
    if 'z_16' in df.columns and 'z_84' in df.columns:
        valid = df.dropna(subset=['z_16', 'z_84', 'gamma_t', 'z_phot'])
        valid = valid.copy()
        valid['z_err'] = (valid['z_84'] - valid['z_16']) / 2
        valid = valid[valid['z_err'] > 0]
        
        if len(valid) > 100:
            # Normalize by z to get fractional error
            valid['z_err_frac'] = valid['z_err'] / valid['z_phot']
            
            rho, p = spearmanr(valid['gamma_t'], valid['z_err_frac'])
            
            print_status(f"N = {len(valid)}", "INFO")
            print_status(f"ρ(Γ_t, z_err/z) = {rho:.3f} (p = {p:.2e})", "INFO")
            
            # TEP predicts POSITIVE correlation (distorted SED → uncertain z)
            if rho > 0.1 and p < 0.001:
                print_status("✓ STRONG: High-Γ_t galaxies have larger z uncertainties", "INFO")
                strong_evidence.append(('z_uncertainty', rho, format_p_value(p)))
            
            results['tests']['z_uncertainty'] = {
                'n': int(len(valid)),
                'rho': float(rho),
                'p': format_p_value(p),
                'strong': bool(rho > 0.1 and p < 0.001)
            }
    
    # ==========================================================================
    # TEST 2: Mass Uncertainty
    # ==========================================================================
    print_status("\n" + "=" * 60, "INFO")
    print_status("TEST 2: Mass Uncertainty", "INFO")
    print_status("=" * 60, "INFO")
    
    if 'log_Mstar_16' in df.columns and 'log_Mstar_84' in df.columns:
        valid = df.dropna(subset=['log_Mstar_16', 'log_Mstar_84', 'gamma_t'])
        valid = valid.copy()
        valid['mstar_err'] = (valid['log_Mstar_84'] - valid['log_Mstar_16']) / 2
        valid = valid[valid['mstar_err'] > 0]
        
        if len(valid) > 100:
            rho, p = spearmanr(valid['gamma_t'], valid['mstar_err'])
            
            print_status(f"N = {len(valid)}", "INFO")
            print_status(f"ρ(Γ_t, M*_err) = {rho:.3f} (p = {p:.2e})", "INFO")
            
            if rho > 0.1 and p < 0.001:
                print_status("✓ STRONG: High-Γ_t galaxies have larger mass uncertainties", "INFO")
                strong_evidence.append(('mass_uncertainty', rho, format_p_value(p)))
            
            results['tests']['mass_uncertainty'] = {
                'n': int(len(valid)),
                'rho': float(rho),
                'p': format_p_value(p),
                'strong': bool(rho > 0.1 and p < 0.001)
            }
    
    # ==========================================================================
    # TEST 3: Age Uncertainty
    # ==========================================================================
    print_status("\n" + "=" * 60, "INFO")
    print_status("TEST 3: Age Uncertainty", "INFO")
    print_status("=" * 60, "INFO")
    
    if 'mwa_16' in df.columns and 'mwa_84' in df.columns:
        valid = df.dropna(subset=['mwa_16', 'mwa_84', 'gamma_t', 'mwa'])
        valid = valid.copy()
        valid['age_err'] = (valid['mwa_84'] - valid['mwa_16']) / 2
        valid = valid[(valid['age_err'] > 0) & (valid['mwa'] > 0)]
        valid['age_err_frac'] = valid['age_err'] / valid['mwa']
        
        if len(valid) > 100:
            rho, p = spearmanr(valid['gamma_t'], valid['age_err_frac'])
            
            print_status(f"N = {len(valid)}", "INFO")
            print_status(f"ρ(Γ_t, Age_err/Age) = {rho:.3f} (p = {p:.2e})", "INFO")
            
            if rho > 0.1 and p < 0.001:
                print_status("✓ STRONG: High-Γ_t galaxies have larger age uncertainties", "INFO")
                strong_evidence.append(('age_uncertainty', rho, format_p_value(p)))
            
            results['tests']['age_uncertainty'] = {
                'n': int(len(valid)),
                'rho': float(rho),
                'p': format_p_value(p),
                'strong': bool(rho > 0.1 and p < 0.001)
            }
    
    # ==========================================================================
    # TEST 4: Spectroscopic Offset
    # ==========================================================================
    print_status("\n" + "=" * 60, "INFO")
    print_status("TEST 4: Spectroscopic vs Photometric Offset", "INFO")
    print_status("=" * 60, "INFO")
    
    spec = df.dropna(subset=['z_spec', 'z_phot', 'gamma_t'])
    
    if len(spec) > 30:
        spec = spec.copy()
        spec['z_offset'] = np.abs(spec['z_phot'] - spec['z_spec'])
        
        rho, p = spearmanr(spec['gamma_t'], spec['z_offset'])
        
        print_status(f"N = {len(spec)} (with spec-z)", "INFO")
        print_status(f"ρ(Γ_t, |z_phot - z_spec|) = {rho:.3f} (p = {p:.2e})", "INFO")
        
        if rho > 0.2 and p < 0.05:
            print_status("✓ High-Γ_t galaxies have larger photo-z offsets", "INFO")
            strong_evidence.append(('spec_offset', rho, format_p_value(p)))
        
        results['tests']['spec_offset'] = {
            'n': int(len(spec)),
            'rho': float(rho),
            'p': format_p_value(p),
            'strong': bool(rho > 0.2 and p < 0.05)
        }
    
    # ==========================================================================
    # TEST 5: Chi2 at High-z Only
    # ==========================================================================
    print_status("\n" + "=" * 60, "INFO")
    print_status("TEST 5: χ² Correlation at z > 7", "INFO")
    print_status("=" * 60, "INFO")
    
    high_z = df[(df['z_phot'] > 7) & df['chi2'].notna() & df['gamma_t'].notna()]
    
    if len(high_z) > 50:
        rho, p = spearmanr(high_z['gamma_t'], high_z['chi2'])
        
        print_status(f"N = {len(high_z)}", "INFO")
        print_status(f"ρ(Γ_t, χ²) at z > 7 = {rho:.3f} (p = {p:.2e})", "INFO")
        
        if rho > 0.1 and p < 0.01:
            print_status("✓ STRONG: χ² correlates with Γ_t at z > 7", "INFO")
            strong_evidence.append(('chi2_highz', rho, format_p_value(p)))
        
        results['tests']['chi2_highz'] = {
            'n': int(len(high_z)),
            'rho': float(rho),
            'p': format_p_value(p),
            'strong': bool(rho > 0.1 and p < 0.01)
        }
    
    # ==========================================================================
    # TEST 6: Dust-Metallicity Decoupling
    # ==========================================================================
    print_status("\n" + "=" * 60, "INFO")
    print_status("TEST 6: Dust-Metallicity Decoupling", "INFO")
    print_status("=" * 60, "INFO")
    
    valid = df.dropna(subset=['dust', 'met', 'gamma_t'])
    
    if len(valid) > 100:
        # Standard physics: dust and metallicity should correlate
        # TEP: high-Gamma_t galaxies have more dust but not more metals
        
        enhanced = valid[valid['gamma_t'] > 1]
        suppressed = valid[valid['gamma_t'] < 0.5]
        
        if len(enhanced) > 20 and len(suppressed) > 20:
            rho_enh, p_enh = spearmanr(enhanced['dust'], enhanced['met'])
            rho_sup, p_sup = spearmanr(suppressed['dust'], suppressed['met'])
            
            print_status(f"Dust-Metallicity correlation:", "INFO")
            print_status(f"  Enhanced (Γ_t > 1): ρ = {rho_enh:.3f} (p = {p_enh:.2e})", "INFO")
            print_status(f"  Suppressed (Γ_t < 0.5): ρ = {rho_sup:.3f} (p = {p_sup:.2e})", "INFO")
            
            # TEP predicts weaker dust-met correlation in enhanced regime
            if rho_enh < rho_sup:
                print_status("✓ Weaker dust-met coupling in enhanced regime (TEP-consistent)", "INFO")
                if abs(rho_enh - rho_sup) > 0.1:
                    strong_evidence.append(('dust_met_decoupling', rho_sup - rho_enh, None))
            
            results['tests']['dust_met_decoupling'] = {
                'n_enhanced': int(len(enhanced)),
                'n_suppressed': int(len(suppressed)),
                'rho_enhanced': float(rho_enh),
                'rho_suppressed': float(rho_sup),
                'difference': float(rho_sup - rho_enh)
            }
    
    # ==========================================================================
    # TEST 7: Triple Correlation (Gamma_t, Age, Dust)
    # ==========================================================================
    print_status("\n" + "=" * 60, "INFO")
    print_status("TEST 7: Triple Correlation Test", "INFO")
    print_status("=" * 60, "INFO")
    
    valid = df.dropna(subset=['mwa', 'dust', 'gamma_t'])
    valid = valid[(valid['mwa'] > 0) & (valid['dust'] > 0)]
    
    if len(valid) > 100:
        # Create combined score: galaxies that are both old AND dusty
        valid = valid.copy()
        valid['age_rank'] = valid['mwa'].rank(pct=True)
        valid['dust_rank'] = valid['dust'].rank(pct=True)
        valid['combined_score'] = valid['age_rank'] + valid['dust_rank']
        
        rho, p = spearmanr(valid['gamma_t'], valid['combined_score'])
        
        print_status(f"N = {len(valid)}", "INFO")
        print_status(f"ρ(Γ_t, Age_rank + Dust_rank) = {rho:.3f} (p = {p:.2e})", "INFO")
        
        if rho > 0.2 and p < 0.001:
            print_status("✓ STRONG: Γ_t predicts combined old+dusty signature", "INFO")
            strong_evidence.append(('triple_correlation', rho, format_p_value(p)))
        
        results['tests']['triple_correlation'] = {
            'n': int(len(valid)),
            'rho': float(rho),
            'p': format_p_value(p),
            'strong': bool(rho > 0.2 and p < 0.001)
        }
    
    # ==========================================================================
    # TEST 8: Effective Time Threshold
    # ==========================================================================
    print_status("\n" + "=" * 60, "INFO")
    print_status("TEST 8: Effective Time Threshold", "INFO")
    print_status("=" * 60, "INFO")
    
    valid = df.dropna(subset=['t_eff', 'dust', 'gamma_t'])
    
    if len(valid) > 100:
        # TEP predicts dust production requires t_eff > 300 Myr (AGB timescale)
        threshold = 0.3  # Gyr
        
        above = valid[valid['t_eff'] > threshold]
        below = valid[valid['t_eff'] <= threshold]
        
        if len(above) > 20 and len(below) > 20:
            mean_dust_above = above['dust'].mean()
            mean_dust_below = below['dust'].mean()
            
            stat, p = mannwhitneyu(above['dust'], below['dust'], alternative='greater')
            
            print_status(f"Dust content by t_eff threshold ({threshold} Gyr):", "INFO")
            print_status(f"  t_eff > {threshold}: N = {len(above)}, <A_V> = {mean_dust_above:.3f}", "INFO")
            print_status(f"  t_eff ≤ {threshold}: N = {len(below)}, <A_V> = {mean_dust_below:.3f}", "INFO")
            print_status(f"  Mann-Whitney p = {p:.2e}", "INFO")
            
            if mean_dust_above > mean_dust_below * 1.5 and p < 0.001:
                print_status("✓ STRONG: Dust production requires sufficient effective time", "INFO")
                strong_evidence.append(('t_eff_threshold', mean_dust_above/mean_dust_below, format_p_value(p)))
            
            results['tests']['t_eff_threshold'] = {
                'threshold': float(threshold),
                'n_above': int(len(above)),
                'n_below': int(len(below)),
                'mean_dust_above': float(mean_dust_above),
                'mean_dust_below': float(mean_dust_below),
                'ratio': float(mean_dust_above / max(mean_dust_below, 0.001)),
                'p': format_p_value(p)
            }
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("SUMMARY: Strong Evidence Found", "INFO")
    print_status("=" * 70, "INFO")
    
    print_status(f"\nStrong evidence items: {len(strong_evidence)}", "INFO")
    for name, stat, p in strong_evidence:
        if p is None:
            print_status(f"  • {name}: stat = {stat:.3f}", "INFO")
        else:
            print_status(f"  • {name}: stat = {stat:.3f}, p = {p:.2e}", "INFO")
    
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
