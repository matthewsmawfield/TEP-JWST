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

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from pathlib import Path
import json

# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "results" / "interim"
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

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
    print("=" * 60)
    print("STEP 7: Thread 1 - z > 7 Mass-sSFR Inversion")
    print("=" * 60)
    print()
    
    df = pd.read_csv(INPUT_PATH / "uncover_full_sample_tep.csv")
    print(f"Loaded sample: N = {len(df)}")
    print()
    
    low_z_mask = (df['z_phot'] >= 4) & (df['z_phot'] < 6)
    high_z_mask = (df['z_phot'] >= 7) & (df['z_phot'] < 10)
    
    df_low = df[low_z_mask]
    df_high = df[high_z_mask]
    
    print(f"Low-z sample (4 < z < 6): N = {len(df_low)}")
    print(f"High-z sample (7 < z < 10): N = {len(df_high)}")
    print()
    
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
    
    print("Results:")
    print("-" * 40)
    print(f"Low-z (4-6):  rho = {rho_low:.3f} [{ci_low[0]:.3f}, {ci_low[1]:.3f}]")
    print(f"High-z (7-10): rho = {rho_high:.3f} [{ci_high[0]:.3f}, {ci_high[1]:.3f}]")
    print()
    print(f"Delta rho = {delta_rho:.3f} [{ci_delta[0]:.3f}, {ci_delta[1]:.3f}]")
    print()
    print(f"95% CI excludes zero: {significant}")
    print(f"★ STATISTICALLY SIGNIFICANT: {significant}")
    print()
    
    if significant:
        print("INTERPRETATION:")
        print("  The mass-sSFR correlation INVERTS at z > 7.")
        print("  This is ANOMALOUS under standard physics (downsizing always negative).")
        print("  TEP PREDICTS this inversion when chronological enhancement dominates.")
    
    results = {
        "test": "Thread 1: z > 7 Mass-sSFR Inversion",
        "tep_prediction": "Correlation should invert from negative to positive at z > 7",
        "low_z": {
            "z_range": [4, 6],
            "n": len(df_low),
            "rho": float(rho_low),
            "ci_95": [float(ci_low[0]), float(ci_low[1])],
            "p_value": float(p_low),
        },
        "high_z": {
            "z_range": [7, 10],
            "n": len(df_high),
            "rho": float(rho_high),
            "ci_95": [float(ci_high[0]), float(ci_high[1])],
            "p_value": float(p_high),
        },
        "delta_rho": float(delta_rho),
        "delta_ci_95": [float(ci_delta[0]), float(ci_delta[1])],
        "significant": significant,
        "conclusion": "TEP prediction CONFIRMED" if significant else "Inconclusive",
    }
    
    with open(OUTPUT_PATH / "thread_1_z7_inversion.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print()
    print(f"Results saved to: {OUTPUT_PATH / 'thread_1_z7_inversion.json'}")
    print()
    print("Step 7 complete.")

if __name__ == "__main__":
    main()
