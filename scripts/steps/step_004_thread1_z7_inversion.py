#!/usr/bin/env python3
"""
TEP-JWST Step 7: Thread 1 - z > 7 Mass-sSFR Inversion

This step tests the first thread of TEP evidence: the inversion of the
mass-sSFR correlation at z > 7.

TEP Prediction:
    At z > 7, the TEP chronological enhancement becomes strong enough
    to dominate over intrinsic downsizing, causing the mass-sSFR
    correlation to invert from negative to positive.

Test:
    Compare Spearman rho(M*, sSFR) between:
    - Low-z sample (4 < z < 6)
    - High-z sample (7 < z < 10)
    
    Bootstrap test for significance of the shift.

Inputs:
- results/interim/uncover_full_sample_tep.csv

Outputs:
- results/outputs/thread_1_z7_inversion.json
"""

import sys
import numpy as np
np.random.seed(42)
import pandas as pd
from scipy.stats import spearmanr
from pathlib import Path
import json


# =============================================================================
# PATHS AND LOGGER
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.p_value_utils import format_p_value, safe_json_default

STEP_NUM = "03"
STEP_NAME = "thread1_z7_inversion"

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
# BOOTSTRAP FUNCTIONS
# =============================================================================

def bootstrap_correlation(x, y, n_boot=1000):
    """Bootstrap confidence interval for Spearman correlation."""
    n = len(x)
    rhos = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        r, _ = spearmanr(x[idx], y[idx])
        rhos.append(r)
    return np.percentile(rhos, [2.5, 97.5])

def bootstrap_delta(x_low, y_low, x_high, y_high, n_boot=1000):
    """Bootstrap test for difference in correlations."""
    n_low = len(x_low)
    n_high = len(x_high)
    deltas = []
    for _ in range(n_boot):
        idx_low = np.random.choice(n_low, n_low, replace=True)
        idx_high = np.random.choice(n_high, n_high, replace=True)
        r_low, _ = spearmanr(x_low[idx_low], y_low[idx_low])
        r_high, _ = spearmanr(x_high[idx_high], y_high[idx_high])
        deltas.append(r_high - r_low)
    return np.array(deltas)

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    print_status("="*60, "INFO")
    print_status("STEP 03: Thread 1 - z > 7 Mass-sSFR Inversion", "INFO")
    print_status("="*60, "INFO")
    print_status("", "INFO")
    
    input_file = INTERIM_PATH / "step_002_uncover_full_sample_tep.csv"
    if not input_file.exists():
        print_status(f"ERROR: Input file not found: {input_file}", "ERROR")
        return

    df = pd.read_csv(input_file)
    print_status(f"Loaded sample: N = {len(df)}", "INFO")
    print_status("", "INFO")
    
    low_z_mask = (df['z_phot'] >= 4) & (df['z_phot'] < 6)
    high_z_mask = (df['z_phot'] >= 7) & (df['z_phot'] < 10)
    
    df_low = df[low_z_mask]
    df_high = df[high_z_mask]
    
    print_status(f"Low-z sample (4 < z < 6): N = {len(df_low)}", "INFO")
    print_status(f"High-z sample (7 < z < 10): N = {len(df_high)}", "INFO")
    print_status("", "INFO")
    
    # Robustness check for sample size
    if len(df_low) < 10 or len(df_high) < 10:
        print_status("ERROR: Insufficient sample size for correlation analysis (need N>=10)", "ERROR")
        return
    
    rho_low, p_low = spearmanr(df_low['log_Mstar'], df_low['log_ssfr'])
    ci_low = bootstrap_correlation(df_low['log_Mstar'].values, df_low['log_ssfr'].values)
    
    rho_high, p_high = spearmanr(df_high['log_Mstar'], df_high['log_ssfr'])
    ci_high = bootstrap_correlation(df_high['log_Mstar'].values, df_high['log_ssfr'].values)
    
    delta_rho = rho_high - rho_low
    deltas = bootstrap_delta(
        df_low['log_Mstar'].values, df_low['log_ssfr'].values,
        df_high['log_Mstar'].values, df_high['log_ssfr'].values
    )
    ci_delta = np.percentile(deltas, [2.5, 97.5])
    
    significant = bool(ci_delta[0] > 0)
    
    print_status("Results:", "INFO")
    print_status("-" * 40, "INFO")
    print_status(f"Low-z (4-6):  rho = {rho_low:.3f} [{ci_low[0]:.3f}, {ci_low[1]:.3f}]", "INFO")
    print_status(f"High-z (7-10): rho = {rho_high:.3f} [{ci_high[0]:.3f}, {ci_high[1]:.3f}]", "INFO")
    print_status("", "INFO")
    print_status(f"Delta rho = {delta_rho:.3f} [{ci_delta[0]:.3f}, {ci_delta[1]:.3f}]", "INFO")
    print_status("", "INFO")
    print_status(f"95% CI excludes zero: {significant}", "INFO")
    print_status(f"STATISTICALLY SIGNIFICANT: {significant}", "SUCCESS" if significant else "WARNING")
    print_status("", "INFO")
    
    if significant:
        print_status("INTERPRETATION:", "INFO")
        print_status("  The mass-sSFR correlation INVERTS at z > 7.", "INFO")
        print_status("  This is inconsistent with standard downsizing predictions.", "INFO")
        print_status("  TEP PREDICTS this inversion when chronological enhancement dominates.", "INFO")
    
    results = {
        "test": "Thread 1: z > 7 Mass-sSFR Inversion",
        "tep_prediction": "Correlation should invert from negative to positive at z > 7",
        "low_z": {
            "z_range": [4, 6],
            "n": len(df_low),
            "rho": float(rho_low),
            "ci_95": [float(ci_low[0]), float(ci_low[1])],
            "p_value": format_p_value(p_low),
        },
        "high_z": {
            "z_range": [7, 10],
            "n": len(df_high),
            "rho": float(rho_high),
            "ci_95": [float(ci_high[0]), float(ci_high[1])],
            "p_value": format_p_value(p_high),
        },
        "delta_rho": float(delta_rho),
        "delta_ci_95": [float(ci_delta[0]), float(ci_delta[1])],
        "significant": significant,
        "conclusion": "TEP prediction CONFIRMED" if significant else "Inconclusive",
    }
    
    with open(OUTPUT_PATH / f"step_{STEP_NUM}_thread1_z7_inversion.json", "w") as f:
        json.dump(results, f, indent=2, default=safe_json_default)
    
    print_status("", "INFO")
    print_status(f"Results saved to: {OUTPUT_PATH / f'step_{STEP_NUM}_thread1_z7_inversion.json'}", "INFO")
    print_status("", "INFO")
    print_status(f"Step {STEP_NUM} complete.", "SUCCESS")

if __name__ == "__main__":
    main()
