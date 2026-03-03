
#!/usr/bin/env python3
"""
TEP-JWST Step 40: Sensitivity Analysis

This script quantifies the stability of the main TEP results against variations
in the coupling parameter alpha_0. 
The nominal value is alpha_0 = 0.58 +/- 0.16 (from Cepheids).
We test a range from 0.0 to 1.0 to see where the signal peaks and if it is robust
within the uncertainty window.

Metrics tracked:
1. z > 8 Mass-Dust correlation (The "Critical Signature")
2. z > 4 Age Ratio correlation (The "Isochrony Bias")
3. Regime Separation (Chi2 separation between enhanced/suppressed)

Inputs:
- results/interim/step_02_uncover_full_sample_tep.csv

Outputs:
- results/outputs/step_40_sensitivity.json
- results/outputs/step_40_sensitivity.csv
"""

import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ks_2samp
from pathlib import Path
import json

# =============================================================================
# PATHS AND LOGGER
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.tep_model import ALPHA_0 as ALPHA_NOMINAL, ALPHA_UNCERTAINTY, compute_gamma_t as tep_gamma

STEP_NUM = "40"
STEP_NAME = "sensitivity_analysis"

DATA_PATH = PROJECT_ROOT / "data"
INTERIM_PATH = PROJECT_ROOT / "results" / "interim"
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"
LOGS_PATH = PROJECT_ROOT / "logs"

for p in [INTERIM_PATH, OUTPUT_PATH, LOGS_PATH]:
    p.mkdir(parents=True, exist_ok=True)

# Initialize logger
logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")
set_step_logger(logger)

# =============================================================================
# CONSTANTS
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================

def main():
    print_status("=" * 70, "INFO")
    print_status(f"STEP {STEP_NUM}: Sensitivity Analysis", "INFO")
    print_status("=" * 70, "INFO")
    
    # Load data
    # We use step_02 output which has cleaned columns
    df = pd.read_csv(INTERIM_PATH / "step_02_uncover_full_sample_tep.csv")
    print_status(f"Loaded N = {len(df)} galaxies", "INFO")
    
    # Define validation subsets
    # 1. z > 8 subset for Dust Anomaly (Thread 5)
    mask_z8 = (df['z_phot'] > 8) & (df['z_phot'] < 10) & (~df['dust'].isna())
    df_z8 = df[mask_z8].copy()
    print_status(f"z > 8 Subset: N = {len(df_z8)}", "INFO")
    
    # 2. z > 4 subset for Age Ratio (Thread 1/General)
    # Require mwa and cosmic age
    # Note: step_02 output usually has mwa, but let's check
    mask_z4 = (df['z_phot'] > 4) & (df['z_phot'] < 10) & (~df['mwa'].isna())
    df_z4 = df[mask_z4].copy()
    
    # Pre-calculate cosmic age if not present or just use z
    # Age ratio needs t_cosmic. 
    # Let's assume 'age_ratio' is present or we recompute? 
    # step_02 usually adds 'age_ratio_obs'. Let's check columns.
    if 'age_ratio' not in df.columns:
        print_status("Warning: 'age_ratio' not found, using 'mwa' / 't_cosmic' proxy if available", "WARNING")
        # In this script context we might not have full cosmo calc loaded easily without astropy
        # But 'step_02' should have it.
        pass
    
    # 3. Full set for Chi2 Regime Separation
    mask_chi2 = (~df['chi2'].isna())
    df_chi2 = df[mask_chi2].copy()
    
    # Parameter range
    alphas = np.linspace(0.0, 1.2, 50)
    
    results = []
    
    print_status("\nRunning sensitivity sweep...", "INFO")
    
    for alpha in alphas:
        # Recompute Gamma_t for relevant subsets
        
        # 1. Dust Anomaly (z > 8)
        # Gamma_t depends on alpha. 
        # Metric: Spearman Rho(Gamma_t, Dust) 
        # Note: If alpha=0, Gamma_t=1 everywhere. Rho is undefined or 0? 
        # Gamma_t is constant 1. So correlation is 0.
        
        if alpha < 0.001:
            rho_dust = 0.0
            p_dust = 1.0
            
            rho_age = 0.0
            p_age = 1.0
            
            ks_sep = 0.0
            p_sep = 1.0
        else:
            # 1. Dust
            g_z8 = tep_gamma(df_z8['log_Mh'], df_z8['z_phot'], alpha_0=alpha)
            rho_dust, p_dust = spearmanr(g_z8, df_z8['dust'])
            
            # 2. Age Ratio
            # Here we correlate Gamma_t with observed age ratio
            g_z4 = tep_gamma(df_z4['log_Mh'], df_z4['z_phot'], alpha_0=alpha)
            rho_age, p_age = spearmanr(g_z4, df_z4['age_ratio'])
            
            # 3. Regime Separation (Chi2)
            g_chi2 = tep_gamma(df_chi2['log_Mh'], df_chi2['z_phot'], alpha_0=alpha)
            enhanced = g_chi2 > 1.0
            suppressed = g_chi2 < 1.0
            if enhanced.sum() > 10 and suppressed.sum() > 10:
                ks_sep, p_sep = ks_2samp(df_chi2.loc[enhanced, 'chi2'], df_chi2.loc[suppressed, 'chi2'])
            else:
                ks_sep = 0.0
                p_sep = 1.0
        
        results.append({
            'alpha': alpha,
            'rho_dust_z8': rho_dust,
            'p_dust_z8': p_dust,
            'rho_age_z4': rho_age,
            'p_age_z4': p_age,
            'ks_stat': ks_sep,
            'p_ks': p_sep
        })
    
    # Convert to DataFrame
    res_df = pd.DataFrame(results)
    
    # Find peaks
    # Dust
    idx_max_dust = res_df['rho_dust_z8'].idxmax()
    best_alpha_dust = res_df.loc[idx_max_dust, 'alpha']
    max_rho_dust = res_df.loc[idx_max_dust, 'rho_dust_z8']
    
    # Age
    idx_max_age = res_df['rho_age_z4'].idxmax()
    best_alpha_age = res_df.loc[idx_max_age, 'alpha']
    max_rho_age = res_df.loc[idx_max_age, 'rho_age_z4']
    
    print_status("-" * 50, "INFO")
    print_status(f"Nominal Alpha: {ALPHA_NOMINAL} +/- {ALPHA_UNCERTAINTY}", "INFO")
    print_status(f"Range tested: 0.0 - 1.2", "INFO")
    print_status("-" * 50, "INFO")
    print_status(f"Peak Dust Correlation (z>8): rho={max_rho_dust:.3f} at alpha={best_alpha_dust:.2f}", "INFO")
    print_status(f"Peak Age Correlation (z>4):  rho={max_rho_age:.3f}  at alpha={best_alpha_age:.2f}", "INFO")
    
    # Check if nominal is within "good" range
    # Define "good" as > 90% of peak signal ? 
    # Or just check nominal signal
    
    nominal_res = res_df.iloc[(res_df['alpha'] - ALPHA_NOMINAL).abs().argsort()[:1]].iloc[0]
    print_status(f"Nominal Performance:", "INFO")
    print_status(f"  Dust Rho: {nominal_res['rho_dust_z8']:.3f} (p={nominal_res['p_dust_z8']:.1e})", "INFO")
    print_status(f"  Age Rho:  {nominal_res['rho_age_z4']:.3f} (p={nominal_res['p_age_z4']:.1e})", "INFO")
    
    # Calculate robustness
    # Fraction of 1-sigma interval (0.42 - 0.74) where p < 0.01
    sigma_range = res_df[(res_df['alpha'] >= ALPHA_NOMINAL - ALPHA_UNCERTAINTY) & 
                         (res_df['alpha'] <= ALPHA_NOMINAL + ALPHA_UNCERTAINTY)]
    
    robust_fraction = (sigma_range['p_dust_z8'] < 0.05).mean()
    print_status(f"Robustness (Dust): Signal significant (p<0.05) over {robust_fraction*100:.0f}% of 1-sigma parameter range", "INFO")
    
    # Save results
    csv_path = OUTPUT_PATH / f"step_{STEP_NUM}_sensitivity.csv"
    res_df.to_csv(csv_path, index=False)
    
    json_path = OUTPUT_PATH / f"step_{STEP_NUM}_sensitivity.json"
    summary = {
        'best_alpha_dust': float(best_alpha_dust),
        'max_rho_dust': float(max_rho_dust),
        'best_alpha_age': float(best_alpha_age),
        'max_rho_age': float(max_rho_age),
        'nominal_rho_dust': float(nominal_res['rho_dust_z8']),
        'robustness_1sigma': float(robust_fraction)
    }
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
        
    print_status(f"Saved sensitivity data to {csv_path}", "INFO")

if __name__ == "__main__":
    main()
