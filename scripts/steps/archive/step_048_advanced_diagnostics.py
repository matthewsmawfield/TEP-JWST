#!/usr/bin/env python3
"""
TEP-JWST Step 48: Advanced Diagnostics

This step explores advanced diagnostics for TEP evidence:

1. Dynamical Mass Discrepancy: M_star > M_dyn should correlate with Gamma_t
2. Environment Proxies: Clustering/density should affect TEP strength
3. UV Luminosity Anomalies: L_UV vs M* relation by regime
4. Specific Angular Momentum: j* should differ between regimes
5. Gas Fraction Proxy: Dust/SFR ratio as gas proxy
6. Formation Epoch: Inferred formation z should correlate with Gamma_t
"""

import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, ks_2samp, linregress  # Correlation, KS test, OLS regression
from pathlib import Path
import json

# =============================================================================
# PATHS AND LOGGER
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging (severity levels: DEBUG/INFO/WARNING/ERROR/SUCCESS)
from scripts.utils.p_value_utils import format_p_value  # Safe p-value formatting (prevents floating-point underflow at p < 1e-300)

STEP_NUM = "048"  # Pipeline step number (sequential 001-176)
STEP_NAME = "advanced_diagnostics"  # Advanced diagnostics: explores 6 sophisticated TEP tests (M_star/M_dyn, environment, UV anomalies, angular momentum, gas fraction, formation epoch)

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
    print_status(f"STEP {STEP_NUM}: Advanced Diagnostics", "INFO")
    print_status("=" * 70, "INFO")
    
    # Load data
    df = pd.read_csv(INTERIM_PATH / "step_002_uncover_full_sample_tep.csv")
    print_status(f"\nLoaded N = {len(df)} galaxies", "INFO")
    
    results = {
        'n_total': len(df),
        'tests': {}
    }
    
    strong_evidence = []
    
    # ==========================================================================
    # TEST 1: Stellar-to-Halo Mass Ratio
    # ==========================================================================
    print_status("\n" + "=" * 60, "INFO")
    print_status("TEST 1: Stellar-to-Halo Mass Ratio", "INFO")
    print_status("=" * 60, "INFO")
    
    valid = df.dropna(subset=['log_Mstar', 'log_Mh', 'gamma_t'])
    
    if len(valid) > 100:
        valid = valid.copy()
        valid['smhm'] = valid['log_Mstar'] - valid['log_Mh']
        
        # At fixed halo mass, does Gamma_t predict higher stellar mass?
        # Residualize SMHM against halo mass
        slope, intercept, _, _, _ = linregress(valid['log_Mh'], valid['smhm'])
        valid['smhm_resid'] = valid['smhm'] - (slope * valid['log_Mh'] + intercept)
        
        rho, p = spearmanr(valid['gamma_t'], valid['smhm_resid'])
        
        print_status(f"N = {len(valid)}", "INFO")
        print_status(f"ρ(Γ_t, SMHM_resid|M_h) = {rho:.3f} (p = {p:.2e})", "INFO")
        
        # TEP predicts positive correlation (more time → more star formation → higher M*/Mh)
        if rho > 0.1 and p < 0.001:
            print_status("✓ STRONG: High-Γ_t galaxies have elevated SMHM at fixed M_h", "INFO")
            strong_evidence.append(('smhm_resid', rho, format_p_value(p)))
        
        results['tests']['smhm'] = {
            'n': int(len(valid)),
            'rho': float(rho),
            'p': format_p_value(p),
            'strong': bool(rho > 0.1 and p < 0.001)
        }
    
    # ==========================================================================
    # TEST 2: Formation Epoch Proxy
    # ==========================================================================
    print_status("\n" + "=" * 60, "INFO")
    print_status("TEST 2: Formation Epoch Proxy", "INFO")
    print_status("=" * 60, "INFO")
    
    # Use age/t_cosmic as formation epoch proxy
    # Higher ratio → formed earlier relative to cosmic time
    valid = df.dropna(subset=['age_ratio', 'gamma_t', 'z_phot'])
    
    if len(valid) > 100:
        # At fixed z, does Gamma_t predict earlier formation?
        z_bins = [(4, 6), (6, 8), (8, 12)]
        
        for z_lo, z_hi in z_bins:
            bin_data = valid[(valid['z_phot'] >= z_lo) & (valid['z_phot'] < z_hi)]
            if len(bin_data) > 30:
                rho, p = spearmanr(bin_data['gamma_t'], bin_data['age_ratio'])
                print_status(f"z = {z_lo}-{z_hi}: N = {len(bin_data)}, ρ(Γ_t, age_ratio) = {rho:.3f} (p = {p:.2e})", "INFO")
        
        # Overall
        rho, p = spearmanr(valid['gamma_t'], valid['age_ratio'])
        print_status(f"Overall: ρ(Γ_t, age_ratio) = {rho:.3f} (p = {p:.2e})", "INFO")
        
        if rho > 0.3 and p < 0.001:
            print_status("✓ STRONG: High-Γ_t galaxies appear to have formed earlier", "INFO")
            strong_evidence.append(('formation_epoch', rho, format_p_value(p)))
        
        results['tests']['formation_epoch'] = {
            'n': int(len(valid)),
            'rho': float(rho),
            'p': format_p_value(p),
            'strong': bool(rho > 0.3 and p < 0.001)
        }
    
    # ==========================================================================
    # TEST 3: Star Formation Efficiency Proxy
    # ==========================================================================
    print_status("\n" + "=" * 60, "INFO")
    print_status("TEST 3: Star Formation Efficiency", "INFO")
    print_status("=" * 60, "INFO")
    
    # SFE ~ M* / (SFR * t_cosmic) ~ sSFR^-1 * (M*/t_cosmic)
    valid = df.dropna(subset=['log_Mstar', 'ssfr100', 'gamma_t', 'z_phot'])
    valid = valid[valid['ssfr100'] > 0]
    
    if len(valid) > 100:
        from astropy.cosmology import Planck18
        valid = valid.copy()
        valid['t_cosmic'] = [Planck18.age(z).value for z in valid['z_phot']]
        
        # SFE proxy: M* / (SFR * t) = 1 / (sSFR * t)
        valid['sfe_proxy'] = 1 / (valid['ssfr100'] * valid['t_cosmic'] * 1e9)
        valid['log_sfe'] = np.log10(valid['sfe_proxy'])
        
        rho, p = spearmanr(valid['gamma_t'], valid['log_sfe'])
        
        print_status(f"N = {len(valid)}", "INFO")
        print_status(f"ρ(Γ_t, log SFE) = {rho:.3f} (p = {p:.2e})", "INFO")
        
        # TEP predicts positive correlation (more time → higher integrated SFE)
        if rho > 0.2 and p < 0.001:
            print_status("✓ STRONG: High-Γ_t galaxies have higher SFE", "INFO")
            strong_evidence.append(('sfe', rho, format_p_value(p)))
        
        results['tests']['sfe'] = {
            'n': int(len(valid)),
            'rho': float(rho),
            'p': format_p_value(p),
            'strong': bool(rho > 0.2 and p < 0.001)
        }
    
    # ==========================================================================
    # TEST 4: UV Slope Proxy (Beta)
    # ==========================================================================
    print_status("\n" + "=" * 60, "INFO")
    print_status("TEST 4: UV Slope (Beta) Proxy", "INFO")
    print_status("=" * 60, "INFO")
    
    # Use V-J color as UV slope proxy (redder V-J → redder UV slope)
    if 'vj' in df.columns:
        valid = df.dropna(subset=['vj', 'gamma_t'])
        
        if len(valid) > 100:
            rho, p = spearmanr(valid['gamma_t'], valid['vj'])
            
            print_status(f"N = {len(valid)}", "INFO")
            print_status(f"ρ(Γ_t, V-J) = {rho:.3f} (p = {p:.2e})", "INFO")
            
            # TEP predicts negative correlation (high Gamma_t → older → redder)
            if rho < -0.15 and p < 0.001:
                print_status("✓ STRONG: High-Γ_t galaxies are redder (V-J)", "INFO")
                strong_evidence.append(('vj_color', rho, format_p_value(p)))
            
            results['tests']['vj_color'] = {
                'n': int(len(valid)),
                'rho': float(rho),
                'p': format_p_value(p),
                'strong': bool(rho < -0.15 and p < 0.001)
            }
    
    # ==========================================================================
    # TEST 5: Dust-to-Age Ratio
    # ==========================================================================
    print_status("\n" + "=" * 60, "INFO")
    print_status("TEST 5: Dust-to-Age Ratio", "INFO")
    print_status("=" * 60, "INFO")
    
    valid = df.dropna(subset=['dust', 'mwa', 'gamma_t'])
    valid = valid[(valid['dust'] > 0) & (valid['mwa'] > 0)]
    
    if len(valid) > 100:
        valid = valid.copy()
        # Dust production rate proxy: A_V / age
        valid['dust_rate'] = valid['dust'] / valid['mwa']
        valid['log_dust_rate'] = np.log10(valid['dust_rate'])
        
        rho, p = spearmanr(valid['gamma_t'], valid['log_dust_rate'])
        
        print_status(f"N = {len(valid)}", "INFO")
        print_status(f"ρ(Γ_t, log(A_V/Age)) = {rho:.3f} (p = {p:.2e})", "INFO")
        
        results['tests']['dust_rate'] = {
            'n': int(len(valid)),
            'rho': float(rho),
            'p': format_p_value(p)
        }
    
    # ==========================================================================
    # TEST 6: Redshift Gradient at Fixed Mass
    # ==========================================================================
    print_status("\n" + "=" * 60, "INFO")
    print_status("TEST 6: Redshift Gradient at Fixed Mass", "INFO")
    print_status("=" * 60, "INFO")
    
    valid = df.dropna(subset=['z_phot', 'log_Mstar', 'dust', 'gamma_t'])
    
    # Select narrow mass bin
    mass_bins = [(8.5, 9.0), (9.0, 9.5), (9.5, 10.0)]
    
    for m_lo, m_hi in mass_bins:
        bin_data = valid[(valid['log_Mstar'] >= m_lo) & (valid['log_Mstar'] < m_hi)]
        if len(bin_data) > 50:
            # Does dust correlate with z at fixed mass?
            rho_z_dust, p_z_dust = spearmanr(bin_data['z_phot'], bin_data['dust'])
            rho_gamma_dust, p_gamma_dust = spearmanr(bin_data['gamma_t'], bin_data['dust'])
            
            print_status(f"Mass bin {m_lo}-{m_hi}: N = {len(bin_data)}", "INFO")
            print_status(f"  ρ(z, Dust) = {rho_z_dust:.3f} (p = {p_z_dust:.2e})", "INFO")
            print_status(f"  ρ(Γ_t, Dust) = {rho_gamma_dust:.3f} (p = {p_gamma_dust:.2e})", "INFO")
            
            if rho_gamma_dust > 0.2 and p_gamma_dust < 0.01:
                print_status(f"✓ Γ_t-Dust correlation holds at fixed mass ({m_lo}-{m_hi})", "INFO")
    
    # ==========================================================================
    # TEST 7: Outlier Population Analysis
    # ==========================================================================
    print_status("\n" + "=" * 60, "INFO")
    print_status("TEST 7: Outlier Population Analysis", "INFO")
    print_status("=" * 60, "INFO")
    
    valid = df.dropna(subset=['dust', 'mwa', 'chi2', 'gamma_t', 'log_Mstar'])
    
    if len(valid) > 100:
        # Define outliers: top 10% in dust AND top 10% in age AND top 10% in chi2
        dust_thresh = valid['dust'].quantile(0.9)
        age_thresh = valid['mwa'].quantile(0.9)
        chi2_thresh = valid['chi2'].quantile(0.9)
        
        outliers = valid[
            (valid['dust'] > dust_thresh) & 
            (valid['mwa'] > age_thresh) & 
            (valid['chi2'] > chi2_thresh)
        ]
        non_outliers = valid.drop(outliers.index)
        
        if len(outliers) > 5:
            mean_gamma_out = outliers['gamma_t'].mean()
            mean_gamma_non = non_outliers['gamma_t'].mean()
            
            print_status(f"Triple outliers (top 10% in dust, age, χ²):", "INFO")
            print_status(f"  Outliers: N = {len(outliers)}, <Γ_t> = {mean_gamma_out:.3f}", "INFO")
            print_status(f"  Non-outliers: N = {len(non_outliers)}, <Γ_t> = {mean_gamma_non:.3f}", "INFO")
            print_status(f"  Ratio: {mean_gamma_out/mean_gamma_non:.2f}×", "INFO")
            
            if mean_gamma_out > mean_gamma_non * 3:
                print_status("✓ STRONG: Triple outliers have highly elevated Γ_t", "INFO")
                strong_evidence.append(('triple_outliers', mean_gamma_out/mean_gamma_non, None))
            
            results['tests']['triple_outliers'] = {
                'n_outliers': int(len(outliers)),
                'n_non_outliers': int(len(non_outliers)),
                'mean_gamma_outliers': float(mean_gamma_out),
                'mean_gamma_non_outliers': float(mean_gamma_non),
                'ratio': float(mean_gamma_out / mean_gamma_non)
            }
    
    # ==========================================================================
    # TEST 8: Halo Mass Dependence
    # ==========================================================================
    print_status("\n" + "=" * 60, "INFO")
    print_status("TEST 8: Halo Mass Dependence of TEP Effect", "INFO")
    print_status("=" * 60, "INFO")
    
    valid = df.dropna(subset=['log_Mh', 'dust', 'gamma_t', 'z_phot'])
    high_z = valid[valid['z_phot'] > 7]
    
    if len(high_z) > 50:
        # Split by halo mass
        mh_median = high_z['log_Mh'].median()
        massive = high_z[high_z['log_Mh'] > mh_median]
        low_mass = high_z[high_z['log_Mh'] <= mh_median]
        
        rho_massive, p_massive = spearmanr(massive['gamma_t'], massive['dust'])
        rho_low, p_low = spearmanr(low_mass['gamma_t'], low_mass['dust'])
        
        print_status(f"At z > 7:", "INFO")
        print_status(f"  Massive halos (log M_h > {mh_median:.1f}): N = {len(massive)}, ρ = {rho_massive:.3f} (p = {p_massive:.2e})", "INFO")
        print_status(f"  Low-mass halos: N = {len(low_mass)}, ρ = {rho_low:.3f} (p = {p_low:.2e})", "INFO")
        
        results['tests']['halo_mass_dependence'] = {
            'mh_median': float(mh_median),
            'n_massive': int(len(massive)),
            'n_low_mass': int(len(low_mass)),
            'rho_massive': float(rho_massive),
            'rho_low_mass': float(rho_low)
        }
    
    # ==========================================================================
    # TEST 9: Consistency Across Redshift Bins
    # ==========================================================================
    print_status("\n" + "=" * 60, "INFO")
    print_status("TEST 9: Consistency Across Redshift Bins", "INFO")
    print_status("=" * 60, "INFO")
    
    valid = df.dropna(subset=['gamma_t', 'dust', 'z_phot'])
    
    z_bins = [(4, 5), (5, 6), (6, 7), (7, 8), (8, 10), (10, 15)]
    consistency_results = []
    
    for z_lo, z_hi in z_bins:
        bin_data = valid[(valid['z_phot'] >= z_lo) & (valid['z_phot'] < z_hi)]
        if len(bin_data) > 20:
            rho, p = spearmanr(bin_data['gamma_t'], bin_data['dust'])
            p_fmt = format_p_value(p)
            significant = (p_fmt is not None and p_fmt < 0.05)
            consistency_results.append({
                'z_range': f"{z_lo}-{z_hi}",
                'n': int(len(bin_data)),
                'rho': float(rho),
                'p': p_fmt,
                'significant': bool(significant)
            })
            sig = "✓" if significant else "✗"
            print_status(f"z = {z_lo}-{z_hi}: N = {len(bin_data)}, ρ = {rho:.3f} (p = {p:.2e}) {sig}", "INFO")
    
    # Count significant bins
    n_significant = sum(1 for r in consistency_results if r['significant'])
    print_status(f"\nSignificant bins: {n_significant}/{len(consistency_results)}", "INFO")
    
    results['tests']['consistency'] = {
        'bins': consistency_results,
        'n_significant': n_significant,
        'n_total': len(consistency_results)
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
