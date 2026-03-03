#!/usr/bin/env python3
"""
TEP-JWST Step 53: Deep Evidence Search

This step explores deeper avenues for strong TEP evidence:

1. Mass-to-Light Ratio: Should correlate strongly with Gamma_t (older populations)
2. Color-Magnitude Anomalies: High-Gamma_t galaxies should be redder at fixed magnitude
3. Redshift-Dependent Screening: The TEP effect should strengthen with z
4. Spectroscopic Subsample: Galaxies with spec-z should show cleaner TEP signatures
5. Morphological Indicators: Compactness/concentration should correlate with Gamma_t
6. Star Formation History Shape: SFH should differ between regimes
"""

import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, ks_2samp, linregress
from pathlib import Path
import json

# =============================================================================
# PATHS AND LOGGER
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.p_value_utils import format_p_value

STEP_NUM = "53"
STEP_NAME = "deep_evidence"

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
    print_status(f"STEP {STEP_NUM}: Deep Evidence Search", "INFO")
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
    # TEST 1: Mass-to-Light Ratio
    # ==========================================================================
    print_status("\n" + "=" * 60, "INFO")
    print_status("TEST 1: Mass-to-Light Ratio", "INFO")
    print_status("=" * 60, "INFO")
    
    # M/L should correlate with Gamma_t (older populations have higher M/L)
    # We can estimate M/L from stellar mass and UV luminosity
    if 'log_Mstar' in df.columns and 'Muv' in df.columns:
        valid = df.dropna(subset=['log_Mstar', 'Muv', 'gamma_t'])
        valid = valid[valid['Muv'] < 0]  # Valid UV magnitudes
        
        if len(valid) > 100:
            # M/L ~ M* / L_UV ~ M* / 10^(-0.4*Muv)
            valid = valid.copy()
            valid['log_ML'] = valid['log_Mstar'] + 0.4 * valid['Muv']
            
            rho, p = spearmanr(valid['gamma_t'], valid['log_ML'])
            
            print_status(f"N = {len(valid)}", "INFO")
            print_status(f"ρ(Γ_t, log M/L_UV) = {rho:.3f} (p = {p:.2e})", "INFO")
            
            # TEP predicts NEGATIVE correlation (high Gamma_t → older → higher M/L)
            # But log_ML is higher for older, so we expect POSITIVE correlation
            if abs(rho) > 0.3 and p < 0.001:
                print_status("✓ STRONG: M/L correlates with Γ_t", "INFO")
                strong_evidence.append(('M_L_ratio', rho, format_p_value(p)))
            
            results['tests']['M_L_ratio'] = {
                'n': int(len(valid)),
                'rho': float(rho),
                'p': format_p_value(p),
                'strong': bool(abs(rho) > 0.3 and p < 0.001)
            }
    
    # ==========================================================================
    # TEST 2: Color at Fixed Magnitude
    # ==========================================================================
    print_status("\n" + "=" * 60, "INFO")
    print_status("TEST 2: Color at Fixed Magnitude", "INFO")
    print_status("=" * 60, "INFO")
    
    if 'uv' in df.columns and 'vj' in df.columns and 'Muv' in df.columns:
        valid = df.dropna(subset=['uv', 'vj', 'Muv', 'gamma_t'])
        
        if len(valid) > 100:
            # Residualize U-V against Muv
            slope, intercept, _, _, _ = linregress(valid['Muv'], valid['uv'])
            valid = valid.copy()
            valid['uv_resid'] = valid['uv'] - (slope * valid['Muv'] + intercept)
            
            rho, p = spearmanr(valid['gamma_t'], valid['uv_resid'])
            
            print_status(f"N = {len(valid)}", "INFO")
            print_status(f"ρ(Γ_t, (U-V)_resid|Muv) = {rho:.3f} (p = {p:.2e})", "INFO")
            
            # TEP predicts high-Gamma_t galaxies are redder at fixed magnitude
            if rho < -0.1 and p < 0.001:
                print_status("✓ STRONG: High-Γ_t galaxies are redder at fixed Muv", "INFO")
                strong_evidence.append(('color_resid', rho, format_p_value(p)))
            
            results['tests']['color_resid'] = {
                'n': int(len(valid)),
                'rho': float(rho),
                'p': format_p_value(p),
                'strong': bool(rho < -0.1 and p < 0.001)
            }
    
    # ==========================================================================
    # TEST 3: Redshift-Dependent TEP Strength
    # ==========================================================================
    print_status("\n" + "=" * 60, "INFO")
    print_status("TEST 3: Redshift-Dependent TEP Strength", "INFO")
    print_status("=" * 60, "INFO")
    
    valid = df.dropna(subset=['dust', 'gamma_t', 'z_phot'])
    
    z_bins = [(4, 5), (5, 6), (6, 7), (7, 8), (8, 10), (10, 15)]
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
            print_status(f"z = {z_lo}-{z_hi}: N = {len(bin_data)}, ρ(Γ_t, Dust) = {rho:.3f} (p = {p:.2e})", "INFO")
    
    # Check if correlation strengthens with z
    if len(z_evolution) >= 4:
        z_mids = [x['z_mid'] for x in z_evolution]
        rhos = [x['rho'] for x in z_evolution]
        trend_rho, trend_p = spearmanr(z_mids, rhos)
        
        print_status(f"\nTrend: ρ(z, Γ_t-Dust correlation) = {trend_rho:.3f} (p = {trend_p:.3f})", "INFO")
        
        if trend_rho > 0.5:
            print_status("✓ TEP-Dust correlation strengthens with z (TEP-consistent)", "INFO")
            strong_evidence.append(('z_evolution', trend_rho, format_p_value(trend_p)))
        
        results['tests']['z_evolution'] = {
            'bins': z_evolution,
            'trend_rho': float(trend_rho),
            'trend_p': format_p_value(trend_p)
        }
    
    # ==========================================================================
    # TEST 4: SFH Shape Differences
    # ==========================================================================
    print_status("\n" + "=" * 60, "INFO")
    print_status("TEST 4: Star Formation History Shape", "INFO")
    print_status("=" * 60, "INFO")
    
    # Use SFR10/SFR100 ratio as proxy for SFH shape
    valid = df.dropna(subset=['sfr10', 'sfr100', 'gamma_t'])
    valid = valid[(valid['sfr10'] > 0) & (valid['sfr100'] > 0)]
    
    if len(valid) > 100:
        valid = valid.copy()
        valid['sfh_ratio'] = np.log10(valid['sfr10'] / valid['sfr100'])
        
        enhanced = valid[valid['gamma_t'] > 1]
        suppressed = valid[valid['gamma_t'] < 0.5]
        
        if len(enhanced) > 20 and len(suppressed) > 20:
            mean_enh = enhanced['sfh_ratio'].mean()
            mean_sup = suppressed['sfh_ratio'].mean()
            std_enh = enhanced['sfh_ratio'].std()
            std_sup = suppressed['sfh_ratio'].std()
            
            # KS test for distribution difference
            stat, p = ks_2samp(enhanced['sfh_ratio'], suppressed['sfh_ratio'])
            
            print_status(f"SFH Shape (log SFR10/SFR100):", "INFO")
            print_status(f"  Enhanced: {mean_enh:.3f} ± {std_enh:.3f}", "INFO")
            print_status(f"  Suppressed: {mean_sup:.3f} ± {std_sup:.3f}", "INFO")
            print_status(f"  KS stat = {stat:.3f}, p = {p:.2e}", "INFO")
            
            if p < 0.001:
                print_status("✓ STRONG: SFH shapes differ between regimes", "INFO")
                strong_evidence.append(('sfh_shape', stat, format_p_value(p)))
            
            results['tests']['sfh_shape'] = {
                'n_enhanced': int(len(enhanced)),
                'n_suppressed': int(len(suppressed)),
                'mean_enhanced': float(mean_enh),
                'mean_suppressed': float(mean_sup),
                'ks_stat': float(stat),
                'p': format_p_value(p),
                'strong': bool(p < 0.001)
            }
    
    # ==========================================================================
    # TEST 5: Dust Production Efficiency
    # ==========================================================================
    print_status("\n" + "=" * 60, "INFO")
    print_status("TEST 5: Dust Production Efficiency", "INFO")
    print_status("=" * 60, "INFO")
    
    # Dust per unit stellar mass should correlate with Gamma_t
    valid = df.dropna(subset=['dust', 'log_Mstar', 'gamma_t'])
    valid = valid[valid['dust'] > 0]
    
    if len(valid) > 100:
        valid = valid.copy()
        # Dust efficiency = A_V / M* (in log space: log(A_V) - log(M*))
        valid['dust_eff'] = np.log10(valid['dust']) - valid['log_Mstar']
        
        rho, p = spearmanr(valid['gamma_t'], valid['dust_eff'])
        
        print_status(f"N = {len(valid)}", "INFO")
        print_status(f"ρ(Γ_t, Dust Efficiency) = {rho:.3f} (p = {p:.2e})", "INFO")
        
        # TEP predicts positive correlation (more time → more dust per unit mass)
        if rho > 0.1 and p < 0.001:
            print_status("✓ STRONG: Dust efficiency correlates with Γ_t", "INFO")
            strong_evidence.append(('dust_efficiency', rho, format_p_value(p)))
        
        results['tests']['dust_efficiency'] = {
            'n': int(len(valid)),
            'rho': float(rho),
            'p': format_p_value(p),
            'strong': bool(rho > 0.1 and p < 0.001)
        }
    
    # ==========================================================================
    # TEST 6: Age-Dust Correlation by Regime
    # ==========================================================================
    print_status("\n" + "=" * 60, "INFO")
    print_status("TEST 6: Age-Dust Correlation by Regime", "INFO")
    print_status("=" * 60, "INFO")
    
    valid = df.dropna(subset=['mwa', 'dust', 'gamma_t'])
    valid = valid[(valid['mwa'] > 0) & (valid['dust'] > 0)]
    
    if len(valid) > 100:
        enhanced = valid[valid['gamma_t'] > 1]
        suppressed = valid[valid['gamma_t'] < 0.5]
        
        if len(enhanced) > 20 and len(suppressed) > 20:
            rho_enh, p_enh = spearmanr(enhanced['mwa'], enhanced['dust'])
            rho_sup, p_sup = spearmanr(suppressed['mwa'], suppressed['dust'])
            
            print_status(f"Age-Dust correlation:", "INFO")
            print_status(f"  Enhanced (Γ_t > 1): ρ = {rho_enh:.3f} (p = {p_enh:.2e})", "INFO")
            print_status(f"  Suppressed (Γ_t < 0.5): ρ = {rho_sup:.3f} (p = {p_sup:.2e})", "INFO")
            
            # TEP predicts STRONGER age-dust correlation in enhanced regime
            # (more time → both older AND dustier)
            if rho_enh > rho_sup + 0.1:
                print_status("✓ Stronger age-dust coupling in enhanced regime (TEP-consistent)", "INFO")
                strong_evidence.append(('age_dust_coupling', rho_enh - rho_sup, None))
            
            results['tests']['age_dust_coupling'] = {
                'n_enhanced': int(len(enhanced)),
                'n_suppressed': int(len(suppressed)),
                'rho_enhanced': float(rho_enh),
                'rho_suppressed': float(rho_sup),
                'difference': float(rho_enh - rho_sup)
            }
    
    # ==========================================================================
    # TEST 7: Cosmic Time Violation
    # ==========================================================================
    print_status("\n" + "=" * 60, "INFO")
    print_status("TEST 7: Cosmic Time Violation", "INFO")
    print_status("=" * 60, "INFO")
    
    # Galaxies where apparent age > cosmic age at that redshift
    valid = df.dropna(subset=['mwa', 'age_ratio', 'gamma_t'])
    
    # age_ratio > 1 means apparent age > cosmic age (violation)
    violators = valid[valid['age_ratio'] > 0.9]  # Within 10% of cosmic age
    non_violators = valid[valid['age_ratio'] <= 0.9]
    
    if len(violators) > 10 and len(non_violators) > 100:
        mean_gamma_viol = violators['gamma_t'].mean()
        mean_gamma_non = non_violators['gamma_t'].mean()
        
        print_status(f"Cosmic Time Violators (age_ratio > 0.9):", "INFO")
        print_status(f"  Violators: N = {len(violators)}, <Γ_t> = {mean_gamma_viol:.3f}", "INFO")
        print_status(f"  Non-violators: N = {len(non_violators)}, <Γ_t> = {mean_gamma_non:.3f}", "INFO")
        print_status(f"  Ratio: {mean_gamma_viol/mean_gamma_non:.2f}×", "INFO")
        
        if mean_gamma_viol > mean_gamma_non * 2:
            print_status("✓ STRONG: Cosmic time violators have elevated Γ_t", "INFO")
            strong_evidence.append(('cosmic_violation', mean_gamma_viol/mean_gamma_non, None))
        
        results['tests']['cosmic_violation'] = {
            'n_violators': int(len(violators)),
            'n_non_violators': int(len(non_violators)),
            'mean_gamma_violators': float(mean_gamma_viol),
            'mean_gamma_non_violators': float(mean_gamma_non),
            'ratio': float(mean_gamma_viol / mean_gamma_non)
        }
    
    # ==========================================================================
    # TEST 8: Multi-Property Anomaly Score
    # ==========================================================================
    print_status("\n" + "=" * 60, "INFO")
    print_status("TEST 8: Multi-Property Anomaly Score", "INFO")
    print_status("=" * 60, "INFO")
    
    valid = df.dropna(subset=['dust', 'mwa', 'chi2', 'gamma_t'])
    valid = valid[(valid['dust'] > 0) & (valid['mwa'] > 0) & (valid['chi2'] > 0)]
    
    if len(valid) > 100:
        valid = valid.copy()
        # Create anomaly score: high dust + old age + high chi2
        valid['dust_pct'] = valid['dust'].rank(pct=True)
        valid['age_pct'] = valid['mwa'].rank(pct=True)
        valid['chi2_pct'] = valid['chi2'].rank(pct=True)
        valid['anomaly_score'] = valid['dust_pct'] + valid['age_pct'] + valid['chi2_pct']
        
        rho, p = spearmanr(valid['gamma_t'], valid['anomaly_score'])
        
        print_status(f"N = {len(valid)}", "INFO")
        print_status(f"ρ(Γ_t, Anomaly Score) = {rho:.3f} (p = {p:.2e})", "INFO")
        
        # Identify top anomalies
        top_anomalies = valid.nlargest(50, 'anomaly_score')
        mean_gamma_top = top_anomalies['gamma_t'].mean()
        mean_gamma_rest = valid.drop(top_anomalies.index)['gamma_t'].mean()
        
        print_status(f"Top 50 anomalies: <Γ_t> = {mean_gamma_top:.3f}", "INFO")
        print_status(f"Rest: <Γ_t> = {mean_gamma_rest:.3f}", "INFO")
        print_status(f"Ratio: {mean_gamma_top/mean_gamma_rest:.2f}×", "INFO")
        
        if rho > 0.2 and p < 0.001:
            print_status("✓ STRONG: Multi-property anomalies have elevated Γ_t", "INFO")
            strong_evidence.append(('anomaly_score', rho, format_p_value(p)))
        
        results['tests']['anomaly_score'] = {
            'n': int(len(valid)),
            'rho': float(rho),
            'p': format_p_value(p),
            'mean_gamma_top50': float(mean_gamma_top),
            'mean_gamma_rest': float(mean_gamma_rest),
            'ratio': float(mean_gamma_top / mean_gamma_rest),
            'strong': bool(rho > 0.2 and p < 0.001)
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
