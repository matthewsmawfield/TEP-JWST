#!/usr/bin/env python3
"""
TEP-JWST Step 10: Threads 6-7 - Coherence Tests

This step tests threads 6-7 of TEP evidence:

Thread 6: Age-Metallicity Coherence
    Older-appearing galaxies should be more metal-enriched.

Thread 7: Multi-Property Split Test
    Galaxies with high Γ_t should show higher age ratio, metallicity,
    and dust content compared to low Γ_t galaxies.

TEP Prediction:
    If TEP is real, all stellar population properties should shift
    coherently with predicted Γ_t. This is the "fingerprint" of TEP.

Inputs:
- results/interim/uncover_multi_property_sample_tep.csv

Outputs:
- results/outputs/thread_6_age_metallicity.json
- results/outputs/thread_7_multi_property.json
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
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
# BOOTSTRAP
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

# =============================================================================
# THREAD 6: Age-Metallicity Coherence
# =============================================================================

def test_thread_6(df):
    """Thread 6: Age-Metallicity Coherence."""
    print("\n" + "=" * 50)
    print("THREAD 6: Age-Metallicity Coherence")
    print("=" * 50)
    print()
    
    age_ratio = df['age_ratio'].values
    met = df['met'].values
    
    rho, p = spearmanr(age_ratio, met)
    ci = bootstrap_correlation(age_ratio, met)
    
    significant = bool(ci[0] > 0)
    
    print(f"N = {len(df)}")
    print(f"rho(age_ratio, metallicity) = {rho:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
    print(f"p = {p:.2e}")
    print()
    print(f"★ STATISTICALLY SIGNIFICANT: {significant}")
    
    if significant:
        print()
        print("INTERPRETATION:")
        print("  Older-appearing galaxies are more metal-enriched.")
        print("  This is consistent with TEP: enhanced time → more enrichment.")
    
    return {
        "test": "Thread 6: Age-Metallicity Coherence",
        "n": int(len(df)),
        "rho": float(rho),
        "p": float(p),
        "ci_95": [float(ci[0]), float(ci[1])],
        "significant": significant,
        "interpretation": "Older → more enriched (TEP consistent)" if significant else "Inconclusive",
    }

# =============================================================================
# THREAD 7: Multi-Property Split Test
# =============================================================================

def test_thread_7(df):
    """Thread 7: Multi-Property Split Test."""
    print("\n" + "=" * 50)
    print("THREAD 7: Multi-Property Split Test")
    print("=" * 50)
    print()
    
    gamma_t = df['gamma_t'].values
    gamma_median = np.median(gamma_t)
    
    high_gamma = gamma_t > gamma_median
    low_gamma = ~high_gamma
    
    print(f"N = {len(df)}")
    print(f"Γ_t median = {gamma_median:.3f}")
    print()
    
    properties = [
        ('age_ratio', df['age_ratio'].values),
        ('metallicity', df['met'].values),
        ('dust', df['dust'].values),
    ]
    
    results = {}
    all_significant = True
    
    for name, values in properties:
        mean_low = np.mean(values[low_gamma])
        mean_high = np.mean(values[high_gamma])
        diff = mean_high - mean_low
        
        stat, p = mannwhitneyu(values[high_gamma], values[low_gamma], alternative='greater')
        significant = bool(p < 0.001)
        
        if not significant:
            all_significant = False
        
        print(f"{name}:")
        print(f"  Low Γ_t:  {mean_low:.3f}")
        print(f"  High Γ_t: {mean_high:.3f}")
        print(f"  Diff:     {diff:+.3f}")
        print(f"  p = {p:.2e}")
        print(f"  ★ SIGNIFICANT: {significant}")
        print()
        
        results[name] = {
            "mean_low_gamma": float(mean_low),
            "mean_high_gamma": float(mean_high),
            "diff": float(diff),
            "p": float(p),
            "significant": significant,
        }
    
    print("=" * 50)
    print(f"ALL PROPERTIES SIGNIFICANT: {all_significant}")
    
    if all_significant:
        print()
        print("INTERPRETATION:")
        print("  All three properties (age, metallicity, dust) shift")
        print("  in the direction predicted by TEP.")
        print("  This is the TEP 'fingerprint' - coherent multi-property shift.")
    
    return {
        "test": "Thread 7: Multi-Property Split Test",
        "n": int(len(df)),
        "gamma_median": float(gamma_median),
        "properties": results,
        "all_significant": all_significant,
        "interpretation": "TEP fingerprint confirmed" if all_significant else "Partial or no confirmation",
    }

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("STEP 10: Threads 6-7 - Coherence Tests")
    print("=" * 60)
    
    df = pd.read_csv(INPUT_PATH / "uncover_multi_property_sample_tep.csv")
    print(f"Loaded multi-property sample: N = {len(df)}")
    
    result_6 = test_thread_6(df)
    with open(OUTPUT_PATH / "thread_6_age_metallicity.json", "w") as f:
        json.dump(result_6, f, indent=2)
    
    result_7 = test_thread_7(df)
    with open(OUTPUT_PATH / "thread_7_multi_property.json", "w") as f:
        json.dump(result_7, f, indent=2)
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Thread 6 (Age-Metallicity): {'✓' if result_6['significant'] else '✗'}")
    print(f"Thread 7 (Multi-Property):  {'✓' if result_7['all_significant'] else '✗'}")
    print()
    print("Step 10 complete.")

if __name__ == "__main__":
    main()
