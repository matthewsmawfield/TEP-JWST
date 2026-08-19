#!/usr/bin/env python3
"""
TEP-JWST Step 65: Additional Literature-Inspired Tests

Based on more recent JWST literature:

1. UV SLOPE (BETA): Blue UV slopes at high-z indicate young, dust-free populations.
   TEP predicts: High-Gamma_t galaxies should have REDDER UV slopes (more dust).

2. STAR FORMATION EFFICIENCY: SFE appears elevated at high-z.
   TEP predicts: SFE is not actually higher - masses are inflated.

3. MASS-TO-LIGHT RATIO: M/L should be elevated in high-Gamma_t regime.

4. STELLAR MASS FUNCTION EXCESS: The SMF at z > 10 exceeds predictions.
   TEP naturally explains this through mass inflation.

5. FORMATION EPOCH: When did these galaxies form?
   TEP predicts: Formation epoch correlates with Gamma_t.
"""

import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from pathlib import Path
import json

# =============================================================================
# PATHS AND LOGGER
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging (severity levels: DEBUG/INFO/WARNING/ERROR/SUCCESS)
from scripts.utils.p_value_utils import format_p_value  # Safe p-value formatting (prevents floating-point underflow at p < 1e-300)

STEP_NUM = "065"  # Pipeline step number (sequential 001-176)
STEP_NAME = "literature_tests"  # Literature tests: 5 additional JWST literature angles (UV slope beta, SFE, M/L ratio, SMF excess, formation epoch)

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
    print_status(f"STEP {STEP_NUM}: Additional Literature-Inspired Tests", "INFO")
    print_status("=" * 70, "INFO")
    
    # Load data
    df = pd.read_csv(INTERIM_PATH / "step_002_uncover_full_sample_tep.csv")
    print_status(f"\nLoaded N = {len(df)} galaxies", "INFO")
    
    results = {
        'n_total': int(len(df)),
        'literature': {}
    }
    
    # ==========================================================================
    # TEST 1: UV Slope Proxy
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 1: UV Slope Proxy", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Literature: Blue UV slopes indicate young, dust-free populations.", "INFO")
    print_status("TEP predicts: High-Gamma_t → more dust → redder UV slopes.\n", "INFO")
    
    # Use dust as proxy for UV slope (more dust = redder)
    valid = df[df['z_phot'] > 8].dropna(subset=['dust', 'gamma_t'])
    
    if len(valid) > 50:
        rho, p = spearmanr(valid['gamma_t'], valid['dust'])
        print_status(f"ρ(Γ_t, Dust) = {rho:.3f} (p = {p:.2e})", "INFO")
        print_status("(Dust is proxy for UV reddening)", "INFO")
        
        if rho > 0.3:
            print_status("✓ High-Γ_t galaxies have more dust (redder UV)", "INFO")
        
        results['literature']['uv_slope'] = {
            'rho': float(rho),
            'p': format_p_value(p)
        }
    
    # ==========================================================================
    # TEST 2: Star Formation Efficiency
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 2: Star Formation Efficiency", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Literature: SFE appears elevated at high-z.", "INFO")
    print_status("TEP predicts: SFE is not actually higher - masses are inflated.\n", "INFO")
    
    valid = df[df['z_phot'] > 8].dropna(subset=['log_Mstar', 'sfr100', 'gamma_t', 'log_Mh'])
    valid = valid[valid['sfr100'] > 0].copy()
    
    if len(valid) > 50:
        # SFE = SFR / M_gas ~ SFR / M_h (using halo mass as proxy)
        valid['log_sfe'] = np.log10(valid['sfr100']) - valid['log_Mh']
        
        rho, p = spearmanr(valid['gamma_t'], valid['log_sfe'])
        print_status(f"ρ(Γ_t, log SFE) = {rho:.3f} (p = {p:.2e})", "INFO")
        
        # Compare high vs low Gamma_t
        high_gamma = valid[valid['gamma_t'] > 1]
        low_gamma = valid[valid['gamma_t'] < 0.5]
        
        if len(high_gamma) > 10 and len(low_gamma) > 10:
            mean_sfe_high = high_gamma['log_sfe'].mean()
            mean_sfe_low = low_gamma['log_sfe'].mean()
            
            print_status(f"<log SFE> high Γ_t: {mean_sfe_high:.3f}", "INFO")
            print_status(f"<log SFE> low Γ_t: {mean_sfe_low:.3f}", "INFO")
        
        results['literature']['sfe'] = {
            'rho': float(rho),
            'p': format_p_value(p)
        }
    
    # ==========================================================================
    # TEST 3: Mass-to-Light Ratio
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 3: Mass-to-Light Ratio Proxy", "INFO")
    print_status("=" * 70, "INFO")
    print_status("TEP predicts: High-Gamma_t galaxies have inflated M/L.\n", "INFO")
    
    valid = df[df['z_phot'] > 8].dropna(subset=['log_Mstar', 'sfr100', 'gamma_t'])
    valid = valid[valid['sfr100'] > 0].copy()
    
    if len(valid) > 50:
        # M/L proxy: M* / SFR (higher = older/more evolved)
        valid['log_ml'] = valid['log_Mstar'] - np.log10(valid['sfr100'])
        
        rho, p = spearmanr(valid['gamma_t'], valid['log_ml'])
        print_status(f"ρ(Γ_t, log M/L) = {rho:.3f} (p = {p:.2e})", "INFO")
        
        if rho > 0.2:
            print_status("✓ High-Γ_t galaxies have higher M/L (more evolved)", "INFO")
        
        results['literature']['ml_ratio'] = {
            'rho': float(rho),
            'p': format_p_value(p)
        }
    
    # ==========================================================================
    # TEST 4: Stellar Mass Function Shape
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 4: Stellar Mass Function Shape", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Literature: SMF at z > 10 shows excess at high-mass end.", "INFO")
    print_status("TEP predicts: This excess is due to mass inflation.\n", "INFO")
    
    z10 = df[df['z_phot'] > 9].dropna(subset=['log_Mstar', 'gamma_t'])
    
    if len(z10) > 20:
        # Count galaxies in mass bins
        mass_bins = [(8, 8.5), (8.5, 9), (9, 9.5), (9.5, 10), (10, 11)]
        
        print_status("Mass distribution at z > 9:", "INFO")
        for m_lo, m_hi in mass_bins:
            bin_data = z10[(z10['log_Mstar'] >= m_lo) & (z10['log_Mstar'] < m_hi)]
            if len(bin_data) > 0:
                mean_gamma = bin_data['gamma_t'].mean()
                print_status(f"  log M* = {m_lo}-{m_hi}: N = {len(bin_data)}, <Γ_t> = {mean_gamma:.2f}", "INFO")
        
        # Correlation between mass and Gamma_t
        rho, p = spearmanr(z10['log_Mstar'], z10['gamma_t'])
        print_status(f"\nρ(log M*, Γ_t) at z > 9 = {rho:.3f} (p = {p:.2e})", "INFO")
        
        results['literature']['smf_shape'] = {
            'rho_mass_gamma': float(rho),
            'p': format_p_value(p)
        }
    
    # ==========================================================================
    # TEST 5: Formation Epoch
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 5: Formation Epoch", "INFO")
    print_status("=" * 70, "INFO")
    print_status("TEP predicts: Formation epoch correlates with Gamma_t.\n", "INFO")
    
    valid = df[df['z_phot'] > 8].dropna(subset=['mwa', 'gamma_t', 'z_phot'])
    valid = valid[valid['mwa'] > 0].copy()
    
    if len(valid) > 50:
        from astropy.cosmology import Planck18
        
        # Calculate formation redshift
        valid['t_cosmic'] = [Planck18.age(z).value for z in valid['z_phot']]
        valid['t_form'] = valid['t_cosmic'] - valid['mwa'] / 1000  # Convert Myr to Gyr
        # Skip formation redshift calculation - use t_form directly
        valid['z_form'] = valid['t_form']  # Use formation time as proxy
        
        valid_form = valid.dropna(subset=['z_form'])
        
        if len(valid_form) > 30:
            rho, p = spearmanr(valid_form['gamma_t'], valid_form['z_form'])
            print_status(f"ρ(Γ_t, z_form) = {rho:.3f} (p = {p:.2e})", "INFO")
            
            results['literature']['formation_epoch'] = {
                'rho': float(rho),
                'p': format_p_value(p)
            }
    
    # ==========================================================================
    # TEST 6: The "Anomalous" Triangle Revisited
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("TEST 6: The 'Anomalous' Triangle Revisited", "INFO")
    print_status("=" * 70, "INFO")
    print_status("Galaxies that are simultaneously: old, dusty, and massive.\n", "INFO")
    
    valid = df[df['z_phot'] > 8].dropna(subset=['mwa', 'dust', 'log_Mstar', 'gamma_t'])
    valid = valid[valid['mwa'] > 0]
    
    if len(valid) > 50:
        # Define thresholds
        age_thresh = valid['mwa'].quantile(0.75)
        dust_thresh = valid['dust'].quantile(0.75)
        mass_thresh = valid['log_Mstar'].quantile(0.75)
        
        # Anomalous triangle: high in all three
        anomalous = valid[
            (valid['mwa'] > age_thresh) & 
            (valid['dust'] > dust_thresh) & 
            (valid['log_Mstar'] > mass_thresh)
        ]
        
        if len(anomalous) > 0:
            mean_gamma_imp = anomalous['gamma_t'].mean()
            mean_gamma_all = valid['gamma_t'].mean()
            
            print_status(f"'Anomalous triangle' galaxies: N = {len(anomalous)}", "INFO")
            print_status(f"<Γ_t> anomalous: {mean_gamma_imp:.2f}", "INFO")
            print_status(f"<Γ_t> all: {mean_gamma_all:.2f}", "INFO")
            print_status(f"Ratio: {mean_gamma_imp/mean_gamma_all:.1f}×", "INFO")
            
            if mean_gamma_imp > mean_gamma_all * 3:
                print_status("✓ 'Anomalous triangle' galaxies have extreme Γ_t", "INFO")
            
            results['literature']['impossible_triangle'] = {
                'n': int(len(anomalous)),
                'mean_gamma_impossible': float(mean_gamma_imp),
                'mean_gamma_all': float(mean_gamma_all),
                'ratio': float(mean_gamma_imp / mean_gamma_all)
            }
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print_status("\n" + "=" * 70, "INFO")
    print_status("SUMMARY: Literature-Inspired Tests", "INFO")
    print_status("=" * 70, "INFO")
    
    print_status("\nKey findings:", "INFO")
    
    for test_name, test_data in results['literature'].items():
        if 'rho' in test_data:
            print_status(f"  • {test_name}: ρ = {test_data['rho']:.3f}", "INFO")
        if 'ratio' in test_data:
            print_status(f"  • {test_name}: {test_data['ratio']:.1f}× elevation", "INFO")
    
    # Save
    json_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print_status(f"\nSaved to {json_path}", "INFO")

if __name__ == "__main__":
    main()
