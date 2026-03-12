#!/usr/bin/env python3
"""
TEP-JWST Step 40: Mass Sensitivity Analysis

Tests robustness of TEP signatures against systematic mass overestimation.
Recent MIRI studies (MACROSS 2024) show NIRCam-only masses can be overestimated
by 0.5-1 dex due to age-attenuation degeneracy and emission-line contamination.

This script:
1. Applies mass reductions of 0.0, 0.3, 0.5, 0.7, 1.0 dex
2. Recomputes halo masses and Gamma_t
3. Re-runs key correlation tests (dust, age, chi2)
4. Reports which signatures survive mass corrections

Inputs:
- results/interim/step_002_uncover_full_sample_tep.csv

Outputs:
- results/outputs/step_040_mass_sensitivity.json
- results/outputs/step_040_mass_sensitivity.csv
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
from scripts.utils.tep_model import ALPHA_0, compute_gamma_t as tep_gamma, stellar_to_halo_mass_behroozi_like

STEP_NUM = "040"
STEP_NAME = "mass_sensitivity"

DATA_PATH = PROJECT_ROOT / "data"
INTERIM_PATH = PROJECT_ROOT / "results" / "interim"
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"
LOGS_PATH = PROJECT_ROOT / "logs"

for p in [INTERIM_PATH, OUTPUT_PATH, LOGS_PATH]:
    p.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")
set_step_logger(logger)

# =============================================================================
# CONSTANTS
# =============================================================================

# Abundance matching relation (Behroozi+ 2019 simplified)
# log(M_h) = log(M_*) + offset(z)
# At z~6-8, offset ~ 1.5-2.0
def stellar_to_halo_mass(log_Mstar, z):
    return stellar_to_halo_mass_behroozi_like(log_Mstar, z)

def gamma_t_with_alpha(log_Mh, z, alpha0=ALPHA_0):
    """Compute Gamma_t for given halo mass and redshift."""
    return tep_gamma(log_Mh, z, alpha_0=alpha0)

# =============================================================================
# MAIN
# =============================================================================

def main():
    print_status("=" * 70, "INFO")
    print_status(f"STEP {STEP_NUM}: Mass Sensitivity Analysis", "INFO")
    print_status("Testing robustness against MIRI-indicated mass overestimation", "INFO")
    print_status("=" * 70, "INFO")
    
    # Load data
    df = pd.read_csv(INTERIM_PATH / "step_002_uncover_full_sample_tep.csv")
    print_status(f"Loaded N = {len(df)} galaxies", "INFO")
    
    # Mass reduction scenarios (in dex)
    mass_reductions = [0.0, 0.3, 0.5, 0.7, 1.0]
    
    results = []
    
    for delta_M in mass_reductions:
        print_status(f"\n--- Mass Reduction: {delta_M} dex ---", "INFO")
        
        # Apply mass reduction
        df_test = df.copy()
        df_test['log_Mstar_corrected'] = df_test['log_Mstar'] - delta_M
        
        # Recompute halo mass
        df_test['log_Mh_corrected'] = stellar_to_halo_mass(
            df_test['log_Mstar_corrected'], 
            df_test['z_phot']
        )
        
        # Recompute Gamma_t
        df_test['Gamma_t_corrected'] = gamma_t_with_alpha(
            df_test['log_Mh_corrected'],
            df_test['z_phot']
        )
        
        # Test 1: z > 8 Mass-Dust Correlation
        mask_z8 = (df_test['z_phot'] > 8) & (df_test['z_phot'] < 10) & (~df_test['dust'].isna())
        df_z8 = df_test[mask_z8]
        
        if len(df_z8) > 20:
            rho_mass_dust, p_mass_dust = spearmanr(df_z8['log_Mstar_corrected'], df_z8['dust'])
            rho_gamma_dust, p_gamma_dust = spearmanr(df_z8['Gamma_t_corrected'], df_z8['dust'])
        else:
            rho_mass_dust, p_mass_dust = np.nan, np.nan
            rho_gamma_dust, p_gamma_dust = np.nan, np.nan
        
        # Test 2: Age Ratio Correlation (z > 4)
        mask_z4 = (df_test['z_phot'] > 4) & (~df_test['age_ratio'].isna())
        df_z4 = df_test[mask_z4]
        
        if len(df_z4) > 50:
            rho_gamma_age, p_gamma_age = spearmanr(df_z4['Gamma_t_corrected'], df_z4['age_ratio'])
        else:
            rho_gamma_age, p_gamma_age = np.nan, np.nan
        
        # Test 3: Chi2 Regime Separation
        mask_chi2 = ~df_test['chi2'].isna()
        df_chi2 = df_test[mask_chi2]
        
        enhanced = df_chi2['Gamma_t_corrected'] > 1.0
        suppressed = df_chi2['Gamma_t_corrected'] < 1.0
        
        if enhanced.sum() > 10 and suppressed.sum() > 10:
            ks_stat, p_ks = ks_2samp(
                df_chi2.loc[enhanced, 'chi2'],
                df_chi2.loc[suppressed, 'chi2']
            )
            mean_chi2_enhanced = df_chi2.loc[enhanced, 'chi2'].mean()
            mean_chi2_suppressed = df_chi2.loc[suppressed, 'chi2'].mean()
        else:
            ks_stat, p_ks = np.nan, np.nan
            mean_chi2_enhanced, mean_chi2_suppressed = np.nan, np.nan
        
        # Test 4: Multi-property coherence (dust, age, metallicity)
        mask_multi = (
            (~df_test['dust'].isna()) & 
            (~df_test['age_ratio'].isna()) & 
            (~df_test['met'].isna())
        )
        df_multi = df_test[mask_multi]
        
        if len(df_multi) > 100:
            # Split by median Gamma_t
            median_gamma = df_multi['Gamma_t_corrected'].median()
            high_gamma = df_multi['Gamma_t_corrected'] > median_gamma
            
            dust_diff = df_multi.loc[high_gamma, 'dust'].mean() - df_multi.loc[~high_gamma, 'dust'].mean()
            age_diff = df_multi.loc[high_gamma, 'age_ratio'].mean() - df_multi.loc[~high_gamma, 'age_ratio'].mean()
            met_diff = df_multi.loc[high_gamma, 'met'].mean() - df_multi.loc[~high_gamma, 'met'].mean()
        else:
            dust_diff, age_diff, met_diff = np.nan, np.nan, np.nan
        
        # Store results
        result = {
            'mass_reduction_dex': delta_M,
            'n_z8': len(df_z8),
            'rho_mass_dust_z8': rho_mass_dust,
            'p_mass_dust_z8': p_mass_dust,
            'rho_gamma_dust_z8': rho_gamma_dust,
            'p_gamma_dust_z8': p_gamma_dust,
            'rho_gamma_age_z4': rho_gamma_age,
            'p_gamma_age_z4': p_gamma_age,
            'ks_chi2_separation': ks_stat,
            'p_ks_chi2': p_ks,
            'mean_chi2_enhanced': mean_chi2_enhanced,
            'mean_chi2_suppressed': mean_chi2_suppressed,
            'dust_diff_high_low_gamma': dust_diff,
            'age_diff_high_low_gamma': age_diff,
            'met_diff_high_low_gamma': met_diff,
            'n_enhanced': int(enhanced.sum()) if not np.isnan(ks_stat) else 0,
            'n_suppressed': int(suppressed.sum()) if not np.isnan(ks_stat) else 0
        }
        results.append(result)
        
        # Print summary
        print_status(f"  Mass-Dust (z>8): rho={rho_mass_dust:.3f}, p={p_mass_dust:.2e}", "INFO")
        print_status(f"  Gamma-Dust (z>8): rho={rho_gamma_dust:.3f}, p={p_gamma_dust:.2e}", "INFO")
        print_status(f"  Gamma-Age (z>4): rho={rho_gamma_age:.3f}, p={p_gamma_age:.2e}", "INFO")
        print_status(f"  Chi2 Separation: KS={ks_stat:.3f}, p={p_ks:.2e}", "INFO")
    
    # Convert to DataFrame
    res_df = pd.DataFrame(results)
    
    # Summary analysis
    print_status("\n" + "=" * 70, "INFO")
    print_status("SENSITIVITY SUMMARY", "INFO")
    print_status("=" * 70, "INFO")
    
    # Check which signatures survive 0.5 dex reduction
    row_05 = res_df[res_df['mass_reduction_dex'] == 0.5].iloc[0]
    row_10 = res_df[res_df['mass_reduction_dex'] == 1.0].iloc[0]
    
    signatures_05 = {
        'mass_dust_z8': row_05['p_mass_dust_z8'] < 0.05 if not np.isnan(row_05['p_mass_dust_z8']) else False,
        'gamma_dust_z8': row_05['p_gamma_dust_z8'] < 0.05 if not np.isnan(row_05['p_gamma_dust_z8']) else False,
        'gamma_age_z4': row_05['p_gamma_age_z4'] < 0.05 if not np.isnan(row_05['p_gamma_age_z4']) else False,
        'chi2_separation': row_05['p_ks_chi2'] < 0.05 if not np.isnan(row_05['p_ks_chi2']) else False
    }
    
    signatures_10 = {
        'mass_dust_z8': row_10['p_mass_dust_z8'] < 0.05 if not np.isnan(row_10['p_mass_dust_z8']) else False,
        'gamma_dust_z8': row_10['p_gamma_dust_z8'] < 0.05 if not np.isnan(row_10['p_gamma_dust_z8']) else False,
        'gamma_age_z4': row_10['p_gamma_age_z4'] < 0.05 if not np.isnan(row_10['p_gamma_age_z4']) else False,
        'chi2_separation': row_10['p_ks_chi2'] < 0.05 if not np.isnan(row_10['p_ks_chi2']) else False
    }
    
    print_status(f"Signatures surviving 0.5 dex reduction: {sum(signatures_05.values())}/4", "INFO")
    for k, v in signatures_05.items():
        status = "PASS" if v else "FAIL"
        print_status(f"  {k}: {status}", "INFO")
    
    print_status(f"\nSignatures surviving 1.0 dex reduction: {sum(signatures_10.values())}/4", "INFO")
    for k, v in signatures_10.items():
        status = "PASS" if v else "FAIL"
        print_status(f"  {k}: {status}", "INFO")
    
    # Save outputs
    csv_path = OUTPUT_PATH / f"step_{STEP_NUM}_mass_sensitivity.csv"
    res_df.to_csv(csv_path, index=False)
    
    summary = {
        'mass_reductions_tested': mass_reductions,
        'signatures_surviving_05dex': signatures_05,
        'signatures_surviving_10dex': signatures_10,
        'n_surviving_05dex': sum(signatures_05.values()),
        'n_surviving_10dex': sum(signatures_10.values()),
        'conclusion': 'robust' if sum(signatures_05.values()) >= 3 else 'sensitive'
    }
    
    json_path = OUTPUT_PATH / f"step_{STEP_NUM}_mass_sensitivity.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print_status(f"\nSaved results to {csv_path}", "INFO")
    print_status(f"Saved summary to {json_path}", "INFO")

if __name__ == "__main__":
    main()
