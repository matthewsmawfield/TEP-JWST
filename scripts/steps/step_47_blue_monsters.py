#!/usr/bin/env python3
"""
TEP-JWST Step 47: Blue Monster TEP Analysis

This step applies TEP corrections to the "Blue Monster" population—
the cleaned sample of massive galaxies after removing AGN-dominated LRDs
(Chworowsky et al. 2025). It quantifies how much of the residual SFE
anomaly is explained by isochrony bias.

Inputs:
- Red Monsters (Xiao et al. 2024): 3 galaxies
- Labbé et al. (2023) candidates: 13 galaxies

Outputs:
- results/outputs/step_47_blue_monsters.json
- results/outputs/step_47_blue_monsters.csv
"""

import sys
import numpy as np
import pandas as pd
from astropy.cosmology import Planck18 as cosmo
from pathlib import Path
import json

# =============================================================================
# PATHS AND LOGGER
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.p_value_utils import format_p_value, safe_json_default
from scripts.utils.tep_model import compute_gamma_t as tep_gamma

STEP_NUM = "47"
STEP_NAME = "blue_monsters"

OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"
LOGS_PATH = PROJECT_ROOT / "logs"

for p in [OUTPUT_PATH, LOGS_PATH]:
    p.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")
set_step_logger(logger)

# =============================================================================
# CONSTANTS & PARAMETERS
# =============================================================================

ML_POWER = 0.7  # M/L ~ t^0.7

# Standard SFE limit
SFE_STANDARD = 0.20

# =============================================================================
# FUNCTIONS
# =============================================================================

def stellar_to_halo_mass(log_mstar):
    """
    Abundance matching for massive high-z galaxies.
    These are the most massive systems, so use a higher offset.
    """
    return log_mstar + 1.8  # Higher offset for extreme systems

def correct_sfe(sfe_obs, gamma_t_val):
    """
    Correct observed SFE for TEP bias.
    SFE_true = SFE_obs / Gamma_t^ML_POWER
    """
    return sfe_obs / (gamma_t_val ** ML_POWER)

def analyze_blue_monsters():
    """
    Analyze the Blue Monster population with TEP corrections.
    """
    print_status("=" * 70, "INFO")
    print_status(f"STEP {STEP_NUM}: Blue Monster TEP Analysis", "INFO")
    print_status("=" * 70, "INFO")
    
    # Define the Blue Monster sample
    # REPRESENTATIVE Red Monster parameters (approximate values inspired by Xiao et al. 2024)
    # These are NOT actual published measurements of specific objects
    red_monsters = [
        {'name': 'RM1', 'z': 5.3, 'log_Mstar': 10.8, 'sfe_obs': 0.50},
        {'name': 'RM2', 'z': 5.5, 'log_Mstar': 10.6, 'sfe_obs': 0.48},
        {'name': 'RM3', 'z': 5.9, 'log_Mstar': 10.9, 'sfe_obs': 0.52},
    ]
    
    # REPRESENTATIVE parameters inspired by Labbé et al. (2023) massive galaxy candidates
    # These are NOT actual published measurements of specific objects
    labbe_candidates = [
        {'name': 'L1', 'z': 7.5, 'log_Mstar': 10.5, 'sfe_obs': 0.42},
        {'name': 'L2', 'z': 8.1, 'log_Mstar': 10.3, 'sfe_obs': 0.38},
        {'name': 'L3', 'z': 7.8, 'log_Mstar': 10.7, 'sfe_obs': 0.45},
        {'name': 'L4', 'z': 6.5, 'log_Mstar': 10.2, 'sfe_obs': 0.35},
        {'name': 'L5', 'z': 9.1, 'log_Mstar': 10.9, 'sfe_obs': 0.55},
        {'name': 'L6', 'z': 7.2, 'log_Mstar': 10.1, 'sfe_obs': 0.32},
        {'name': 'L7', 'z': 8.5, 'log_Mstar': 10.4, 'sfe_obs': 0.40},
        {'name': 'L8', 'z': 6.8, 'log_Mstar': 10.6, 'sfe_obs': 0.43},
        {'name': 'L9', 'z': 7.9, 'log_Mstar': 10.0, 'sfe_obs': 0.30},
        {'name': 'L10', 'z': 8.8, 'log_Mstar': 10.8, 'sfe_obs': 0.50},
        {'name': 'L11', 'z': 7.0, 'log_Mstar': 9.8, 'sfe_obs': 0.28},
        {'name': 'L12', 'z': 8.3, 'log_Mstar': 10.5, 'sfe_obs': 0.44},
        {'name': 'L13', 'z': 7.6, 'log_Mstar': 10.3, 'sfe_obs': 0.38},
    ]
    
    all_monsters = red_monsters + labbe_candidates
    
    results = []
    
    for m in all_monsters:
        z = m['z']
        log_mstar = m['log_Mstar']
        sfe_obs = m['sfe_obs']
        
        # Derive halo mass
        log_mh = stellar_to_halo_mass(log_mstar)
        
        # Calculate Gamma_t
        g_t = tep_gamma(log_mh, z)
        
        # Correct SFE
        sfe_true = correct_sfe(sfe_obs, g_t)
        
        # Calculate anomaly resolution
        excess_obs = sfe_obs / SFE_STANDARD
        excess_true = sfe_true / SFE_STANDARD
        anomaly_resolved = (excess_obs - excess_true) / (excess_obs - 1) * 100 if excess_obs > 1 else 0
        
        results.append({
            'name': m['name'],
            'z': z,
            'log_Mstar': log_mstar,
            'log_Mh': log_mh,
            'sfe_obs': sfe_obs,
            'gamma_t': g_t,
            'sfe_true': sfe_true,
            'excess_obs': excess_obs,
            'excess_true': excess_true,
            'anomaly_resolved_pct': anomaly_resolved
        })
    
    df = pd.DataFrame(results)
    
    # Summary statistics
    print_status(f"\nBlue Monster Population: N = {len(df)}", "INFO")
    print_status(f"  Mean SFE (observed): {df['sfe_obs'].mean():.2f}", "INFO")
    print_status(f"  Mean SFE (TEP-corrected): {df['sfe_true'].mean():.2f}", "INFO")
    print_status(f"  Mean Gamma_t: {df['gamma_t'].mean():.2f}", "INFO")
    print_status(f"  Mean anomaly resolved: {df['anomaly_resolved_pct'].mean():.0f}%", "INFO")
    print_status(f"  SFE excess before TEP: {df['sfe_obs'].mean()/SFE_STANDARD:.1f}x", "INFO")
    print_status(f"  SFE excess after TEP: {df['sfe_true'].mean()/SFE_STANDARD:.1f}x", "INFO")
    
    # Save outputs
    csv_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.csv"
    df.to_csv(csv_path, index=False)
    print_status(f"\nSaved CSV to {csv_path}", "INFO")
    
    summary = {
        'n_monsters': len(df),
        'mean_sfe_obs': float(df['sfe_obs'].mean()),
        'mean_sfe_true': float(df['sfe_true'].mean()),
        'mean_gamma_t': float(df['gamma_t'].mean()),
        'mean_anomaly_resolved_pct': float(df['anomaly_resolved_pct'].mean()),
        'sfe_excess_before': float(df['sfe_obs'].mean() / SFE_STANDARD),
        'sfe_excess_after': float(df['sfe_true'].mean() / SFE_STANDARD),
        'red_monsters': {
            'n': 3,
            'mean_gamma_t': float(df[df['name'].str.startswith('RM')]['gamma_t'].mean()),
            'mean_anomaly_resolved': float(df[df['name'].str.startswith('RM')]['anomaly_resolved_pct'].mean())
        },
        'labbe_candidates': {
            'n': 13,
            'mean_gamma_t': float(df[df['name'].str.startswith('L')]['gamma_t'].mean()),
            'mean_anomaly_resolved': float(df[df['name'].str.startswith('L')]['anomaly_resolved_pct'].mean())
        }
    }
    
    json_path = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, default=safe_json_default)
    print_status(f"Saved JSON to {json_path}", "INFO")
    
    return summary

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    analyze_blue_monsters()
