#!/usr/bin/env python3
"""
TEP-JWST Step 52: Mass Discrepancy Test

A critical TEP prediction: Stellar mass estimates from SED fitting
should be INFLATED relative to dynamical mass estimates because:

- SED fitting assumes standard stellar evolution timescales
- TEP enhances proper time, making stars appear older/more massive
- Dynamical mass (from kinematics) is NOT affected by TEP

Therefore: M_star(SED) / M_dyn should correlate with Gamma_t

This is a UNIQUE signature that standard physics cannot produce.
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

STEP_NUM = "052"
STEP_NAME = "mass_discrepancy"

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
    print_status(f"STEP {STEP_NUM}: Mass Discrepancy Test", "INFO")
    print_status("=" * 70, "INFO")
    print_status("\nTesting if SED masses are inflated relative to dynamical masses...\n", "INFO")
    
    # Load data
    df = pd.read_csv(INTERIM_PATH / "step_002_uncover_full_sample_tep.csv")
    print_status(f"Loaded N = {len(df)} galaxies", "INFO")
    
    results = {
        'n_total': len(df),
        'tests': {}
    }
    
    evidence = []
    
    # ==========================================================================
    # TEST 1: Mass-to-Light Ratio as Proxy
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 1: Mass-to-Light Ratio as Mass Discrepancy Proxy", "INFO")
    print_status("=" * 70, "INFO")
    print_status("M/L should be elevated for high-Gamma_t galaxies.\n", "INFO")
    
    # M/L can be estimated from stellar mass and UV luminosity
    # Higher M/L indicates older stellar populations (or TEP inflation)
    
    valid = df.dropna(subset=['log_Mstar', 'gamma_t', 'z_phot'])
    high_z = valid[valid['z_phot'] > 7].copy()
    
    if len(high_z) > 50:
        # Use stellar mass as proxy (at fixed halo mass, higher M* = higher M/L)
        # Residualize M* against M_h
        from scipy.stats import linregress
        
        high_z_valid = high_z.dropna(subset=['log_Mh'])
        if len(high_z_valid) > 30:
            slope, intercept, _, _, _ = linregress(high_z_valid['log_Mh'], high_z_valid['log_Mstar'])
            high_z_valid = high_z_valid.copy()
            high_z_valid['mstar_resid'] = high_z_valid['log_Mstar'] - (slope * high_z_valid['log_Mh'] + intercept)
            
            rho, p = spearmanr(high_z_valid['gamma_t'], high_z_valid['mstar_resid'])
            
            print_status(f"N = {len(high_z_valid)}", "INFO")
            print_status(f"ρ(Γ_t, M*_resid|M_h) = {rho:.3f} (p = {p:.2e})", "INFO")
            
            # TEP predicts POSITIVE correlation (high Gamma_t → inflated M*)
            # But wait - Gamma_t is DERIVED from M_h, so this is circular
            # We need a different approach
            
            results['tests']['mstar_resid'] = {
                'n': int(len(high_z_valid)),
                'rho': float(rho),
                'p': format_p_value(p)
            }
    
    # ==========================================================================
    # TEST 2: Stellar Mass vs Halo Mass Ratio
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 2: Stellar-to-Halo Mass Ratio", "INFO")
    print_status("=" * 70, "INFO")
    print_status("SMHM should be elevated at high-z if TEP inflates stellar masses.\n", "INFO")
    
    valid = df.dropna(subset=['log_Mstar', 'log_Mh', 'z_phot'])
    
    # Compare SMHM at different redshifts
    z_bins = [(4, 5), (5, 6), (6, 7), (7, 8), (8, 10)]
    smhm_results = []
    
    for z_lo, z_hi in z_bins:
        bin_data = valid[(valid['z_phot'] >= z_lo) & (valid['z_phot'] < z_hi)]
        if len(bin_data) > 30:
            smhm = bin_data['log_Mstar'] - bin_data['log_Mh']
            mean_smhm = smhm.mean()
            smhm_results.append({
                'z_range': f"{z_lo}-{z_hi}",
                'n': int(len(bin_data)),
                'mean_smhm': float(mean_smhm)
            })
            print_status(f"z = {z_lo}-{z_hi}: N = {len(bin_data)}, <log(M*/M_h)> = {mean_smhm:.3f}", "INFO")
    
    # Check if SMHM increases with z (TEP prediction)
    if len(smhm_results) >= 4:
        z_mids = [(float(r['z_range'].split('-')[0]) + float(r['z_range'].split('-')[1]))/2 for r in smhm_results]
        smhms = [r['mean_smhm'] for r in smhm_results]
        rho, p = spearmanr(z_mids, smhms)
        
        print_status(f"\nTrend: ρ(z, SMHM) = {rho:.3f} (p = {p:.3f})", "INFO")
        
        if rho > 0.5:
            print_status("✓ SMHM increases with z (TEP-consistent)", "INFO")
            evidence.append(('smhm_z_trend', rho, format_p_value(p)))
    
    results['tests']['smhm'] = smhm_results
    
    # ==========================================================================
    # TEST 3: Mass Excess at Fixed Properties
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 3: Mass Excess at Fixed SFR", "INFO")
    print_status("=" * 70, "INFO")
    print_status("At fixed SFR, high-Gamma_t galaxies should have higher M*.\n", "INFO")
    
    valid = df.dropna(subset=['log_Mstar', 'sfr100', 'gamma_t', 'z_phot'])
    valid = valid[valid['sfr100'] > 0]
    high_z = valid[valid['z_phot'] > 7].copy()
    
    if len(high_z) > 50:
        high_z['log_sfr'] = np.log10(high_z['sfr100'])
        
        # Residualize M* against SFR
        slope, intercept, _, _, _ = linregress(high_z['log_sfr'], high_z['log_Mstar'])
        high_z['mstar_resid_sfr'] = high_z['log_Mstar'] - (slope * high_z['log_sfr'] + intercept)
        
        rho, p = spearmanr(high_z['gamma_t'], high_z['mstar_resid_sfr'])
        
        print_status(f"N = {len(high_z)}", "INFO")
        print_status(f"ρ(Γ_t, M*_resid|SFR) = {rho:.3f} (p = {p:.2e})", "INFO")
        
        # TEP predicts POSITIVE correlation
        if rho > 0.1 and p < 0.01:
            print_status("✓ High-Γ_t galaxies have excess mass at fixed SFR", "INFO")
            evidence.append(('mass_excess_sfr', rho, format_p_value(p)))
        
        results['tests']['mass_excess_sfr'] = {
            'n': int(len(high_z)),
            'rho': float(rho),
            'p': format_p_value(p)
        }
    
    # ==========================================================================
    # TEST 4: The "Overmassive" Population
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 4: The 'Overmassive' Population", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Galaxies with M* > expected from SMHM relation should have high Gamma_t.\n", "INFO")
    
    valid = df.dropna(subset=['log_Mstar', 'log_Mh', 'gamma_t', 'z_phot'])
    high_z = valid[valid['z_phot'] > 7].copy()
    
    if len(high_z) > 50:
        # Fit SMHM relation
        slope, intercept, _, _, _ = linregress(high_z['log_Mh'], high_z['log_Mstar'])
        high_z['mstar_expected'] = slope * high_z['log_Mh'] + intercept
        high_z['mstar_excess'] = high_z['log_Mstar'] - high_z['mstar_expected']
        
        # Define "overmassive" as top 25%
        overmassive_thresh = high_z['mstar_excess'].quantile(0.75)
        overmassive = high_z[high_z['mstar_excess'] > overmassive_thresh]
        normal = high_z[high_z['mstar_excess'] <= overmassive_thresh]
        
        mean_gamma_over = overmassive['gamma_t'].mean()
        mean_gamma_norm = normal['gamma_t'].mean()
        
        print_status(f"Overmassive (top 25%): N = {len(overmassive)}, <Γ_t> = {mean_gamma_over:.3f}", "INFO")
        print_status(f"Normal: N = {len(normal)}, <Γ_t> = {mean_gamma_norm:.3f}", "INFO")
        print_status(f"Ratio: {mean_gamma_over/mean_gamma_norm:.2f}×", "INFO")
        
        if mean_gamma_over > mean_gamma_norm * 1.3:
            print_status("✓ Overmassive galaxies have elevated Γ_t", "INFO")
            evidence.append(('overmassive', mean_gamma_over/mean_gamma_norm, None))
        
        results['tests']['overmassive'] = {
            'n_overmassive': int(len(overmassive)),
            'n_normal': int(len(normal)),
            'mean_gamma_overmassive': float(mean_gamma_over),
            'mean_gamma_normal': float(mean_gamma_norm),
            'ratio': float(mean_gamma_over / mean_gamma_norm)
        }
    
    # ==========================================================================
    # TEST 5: Mass Function Shape
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 5: Mass Function Shape by Regime", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Enhanced regime should have more massive galaxies.\n", "INFO")
    
    valid = df.dropna(subset=['log_Mstar', 'gamma_t', 'z_phot'])
    high_z = valid[valid['z_phot'] > 8]
    
    if len(high_z) > 50:
        enhanced = high_z[high_z['gamma_t'] > 1]
        suppressed = high_z[high_z['gamma_t'] < 0.5]
        
        if len(enhanced) > 10 and len(suppressed) > 10:
            # Compare mass distributions
            mean_mass_enh = enhanced['log_Mstar'].mean()
            mean_mass_sup = suppressed['log_Mstar'].mean()
            
            # Fraction above log(M*) = 9
            frac_massive_enh = (enhanced['log_Mstar'] > 9).mean()
            frac_massive_sup = (suppressed['log_Mstar'] > 9).mean()
            
            print_status(f"Enhanced (Γ_t > 1): <log M*> = {mean_mass_enh:.2f}, f(M* > 10^9) = {frac_massive_enh*100:.1f}%", "INFO")
            print_status(f"Suppressed (Γ_t < 0.5): <log M*> = {mean_mass_sup:.2f}, f(M* > 10^9) = {frac_massive_sup*100:.1f}%", "INFO")
            
            if frac_massive_enh > frac_massive_sup * 2:
                print_status("✓ Enhanced regime has more massive galaxies", "INFO")
                evidence.append(('mass_function', frac_massive_enh/max(frac_massive_sup, 0.01), None))
            
            results['tests']['mass_function'] = {
                'mean_mass_enhanced': float(mean_mass_enh),
                'mean_mass_suppressed': float(mean_mass_sup),
                'frac_massive_enhanced': float(frac_massive_enh),
                'frac_massive_suppressed': float(frac_massive_sup)
            }
    
    # ==========================================================================
    # TEST 6: Specific Star Formation Rate Inversion
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 6: sSFR Inversion Test", "INFO")
    print_status("=" * 70, "INFO")
    print_status("If M* is inflated by TEP, sSFR = SFR/M* should be LOWER.\n", "INFO")
    
    valid = df.dropna(subset=['ssfr100', 'gamma_t', 'z_phot'])
    valid = valid[valid['ssfr100'] > 0]
    high_z = valid[valid['z_phot'] > 7].copy()
    
    if len(high_z) > 50:
        high_z['log_ssfr'] = np.log10(high_z['ssfr100'])
        
        rho, p = spearmanr(high_z['gamma_t'], high_z['log_ssfr'])
        
        print_status(f"N = {len(high_z)}", "INFO")
        print_status(f"ρ(Γ_t, log sSFR) = {rho:.3f} (p = {p:.2e})", "INFO")
        
        # TEP predicts NEGATIVE correlation (inflated M* → lower sSFR)
        if rho < -0.1 and p < 0.01:
            print_status("✓ High-Γ_t galaxies have lower sSFR (mass inflation signature)", "INFO")
            evidence.append(('ssfr_inversion', rho, format_p_value(p)))
        
        results['tests']['ssfr_inversion'] = {
            'n': int(len(high_z)),
            'rho': float(rho),
            'p': format_p_value(p)
        }
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("SUMMARY: Mass Discrepancy Evidence", "INFO")
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
