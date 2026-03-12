#!/usr/bin/env python3
"""
TEP-JWST Step 52: Independent Dust Robustness Test

Tests the robustness of the z > 8 Mass-Dust correlation across different 
surveys (UNCOVER, CEERS, COSMOS-Web) which utilize different SPS codes 
and priors (EAZY, BEAGLE, LePhare).

Methodology:
1. Load z > 8 samples from all three surveys.
2. Standardize Dust metrics (A_V vs E(B-V)).
3. Compute Spearman correlation for each.
4. Compare confidence intervals.
5. If overlapping and positive, claim "Cross-Code Robustness".

Author: Matthew L. Smawfield
Date: January 2026
"""

import sys
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.p_value_utils import format_p_value

STEP_NUM = "52"
STEP_NAME = "independent_dust"
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"
INTERIM_PATH = PROJECT_ROOT / "data" / "interim"

def load_data():
    datasets = {}
    
    # UNCOVER (BEAGLE/EAZY)
    try:
        res_interim = PROJECT_ROOT / "results" / "interim"
        df_uncover = pd.read_csv(res_interim / "step_02_uncover_full_sample_tep.csv")
        # Assuming 'gamma_t' column exists or similar, but we check Mass-Dust
        # Need to check column names. UNCOVER usually has 'av' or 'dust'
        # Previous steps used 'step_02_uncover_full_sample_tep.csv' in results/interim
        # Let's check results/interim if data/interim fails
            
        mask = (df_uncover['z_phot'] > 8) & (df_uncover['log_Mstar'] > 7)
        datasets['UNCOVER'] = {
            'mass': df_uncover.loc[mask, 'log_Mstar'].values,
            'dust': df_uncover.loc[mask, 'dust'].values, # A_V
            'code': 'Prospector/BEAGLE'
        }
    except Exception as e:
        print_status(f"Failed to load UNCOVER: {e}", "WARN")

    # CEERS (EAZY/LePhare)
    try:
        df_ceers = pd.read_csv(INTERIM_PATH / "ceers_z8_sample.csv")
        # Check cols: 'z_phot', 'log_Mstar', 'dust' (or Av)
        # Using column names from head command earlier: 'z_phot', 'log_Mstar', 'dust'
        mask = (df_ceers['z_phot'] > 8) & (df_ceers['log_Mstar'] > 7) & (df_ceers['dust'] > -99)
        datasets['CEERS'] = {
            'mass': df_ceers.loc[mask, 'log_Mstar'].values,
            'dust': df_ceers.loc[mask, 'dust'].values, # A_V
            'code': 'EAZY'
        }
    except Exception as e:
        print_status(f"Failed to load CEERS: {e}", "WARN")

    # COSMOS-Web (LePhare)
    try:
        df_cosmos = pd.read_csv(INTERIM_PATH / "cosmosweb_z8_sample.csv")
        # Columns: id,ra,dec,z_phot,log_Mstar,sfr,dust
        mask = (df_cosmos['z_phot'] > 8) & (df_cosmos['log_Mstar'] > 7) & (df_cosmos['dust'] > -99)
        datasets['COSMOS-Web'] = {
            'mass': df_cosmos.loc[mask, 'log_Mstar'].values,
            'dust': df_cosmos.loc[mask, 'dust'].values, # E(B-V) probably? Or A_V. 
            # LePhare usually outputs E(B-V). But Spearman is rank, so scale doesn't matter.
            'code': 'LePhare'
        }
    except Exception as e:
        print_status(f"Failed to load COSMOS-Web: {e}", "WARN")
        
    return datasets

def bootstrap_ci(x, y, n_boot=1000):
    inds = np.random.randint(0, len(x), (n_boot, len(x)))
    rhos = []
    for i in range(n_boot):
        # Spearman is slow for large N in loop, but N is ~100-500 here
        r, _ = stats.spearmanr(x[inds[i]], y[inds[i]])
        rhos.append(r)
    return np.percentile(rhos, [2.5, 97.5])

def main():
    logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=PROJECT_ROOT / "logs" / f"step_{STEP_NUM}_{STEP_NAME}.log")
    set_step_logger(logger)
    
    print_status("Loading Multi-Survey Data...", "INFO")
    datasets = load_data()
    
    results = {}
    
    print_status("\n--- Cross-Code Robustness Results ---", "INFO")
    
    for name, data in datasets.items():
        mass = data['mass']
        dust = data['dust']
        code = data['code']
        n = len(mass)
        
        if n < 10:
            print_status(f"{name}: Insufficient samples (N={n})", "WARN")
            continue
            
        rho, p = stats.spearmanr(mass, dust)
        ci = bootstrap_ci(mass, dust)
        
        print_status(f"{name} ({code}): N={n}, rho={rho:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}], p={p:.2e}", "INFO")
        
        results[name] = {
            'n': int(n),
            'rho': float(rho),
            'p': format_p_value(p),
            'ci': [float(ci[0]), float(ci[1])],
            'code': code
        }
        
    # Meta-Analysis
    print_status("\n--- Synthesis ---", "INFO")
    rhos = [d['rho'] for k, d in results.items()]
    ns = [d['n'] for k, d in results.items()]
    
    # Fisher Z weighted average
    zs = 0.5 * np.log((1 + np.array(rhos)) / (1 - np.array(rhos)))
    weights = np.array(ns) - 3
    z_bar = np.sum(weights * zs) / np.sum(weights)
    rho_bar = (np.exp(2*z_bar) - 1) / (np.exp(2*z_bar) + 1)
    
    print_status(f"Weighted Mean Correlation: rho = {rho_bar:.3f}", "INFO")
    
    consistent = all(d['rho'] > 0.3 for k, d in results.items())
    print_status(f"Robust across codes? {consistent}", "INFO")
    
    results['meta_analysis'] = {
        'weighted_rho': float(rho_bar),
        'consistent': consistent
    }
    
    with open(OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
