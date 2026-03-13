#!/usr/bin/env python3
# Estimated runtime from last full canonical run (2026-03-09 15:52 UTC; full pipeline 32m18s): 0.9s.
"""
TEP-JWST Step 7: Threads 6-7 - Coherence Tests

This step tests threads 6-7 of TEP evidence:

Thread 6: Age-Metallicity Coherence
    Older-appearing galaxies should be more metal-enriched.

Thread 7: Multi-Property Split Test
    Galaxies with high Γ_t should show higher age ratio, metallicity,
    and dust content compared to low Γ_t galaxies.

TEP Prediction:
    If TEP is real, all stellar population properties should shift
    coherently with predicted Γ_t. This is the characteristic signature of TEP.

Inputs:
- results/interim/uncover_multi_property_sample_tep.csv

Outputs:
- results/outputs/thread_6_age_metallicity.json
- results/outputs/thread_7_multi_property.json
"""

import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
from pathlib import Path
import json


# =============================================================================
# PATHS AND LOGGER
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status  # Centralised logging (severity levels: DEBUG/INFO/WARNING/ERROR/SUCCESS)
from scripts.utils.p_value_utils import format_p_value, safe_json_default  # Safe p-value formatting (prevents floating-point underflow at extreme significance) & JSON serialiser

STEP_NUM = "007"  # Pipeline step number (sequential 001-176)
STEP_NAME = "thread6_7_coherence"  # Threads 6-7: Age-metallicity coherence & multi-property split (TEP coherence tests)

DATA_PATH = PROJECT_ROOT / "data"  # Top-level data directory (raw external catalogs: UNCOVER DR4, CEERS, COSMOS-Web, JADES)
INTERIM_PATH = PROJECT_ROOT / "results" / "interim"  # Pre-processed intermediate products (CSV format for step-to-step data flow)
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"  # JSON output directory (machine-readable statistical results)
LOGS_PATH = PROJECT_ROOT / "logs"  # Log directory (one plain-text log per step for debugging traceability)

for p in [INTERIM_PATH, OUTPUT_PATH, LOGS_PATH]:
    p.mkdir(parents=True, exist_ok=True)  # Create directory tree if missing; exist_ok=True allows safe re-runs

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")  # Step-specific logger (isolated per-step logging for traceability)
set_step_logger(logger)  # Register as global step logger so print_status() routes here

BOOTSTRAP_SEED = 42  # Fixed RNG seed for reproducible bootstrap confidence intervals

# =============================================================================
# BOOTSTRAP
# =============================================================================

def bootstrap_correlation(x, y, n_boot=1000, seed=BOOTSTRAP_SEED):
    """Bootstrap 95% CI for Spearman rho via case resampling.

    Uses numpy's default_rng for reproducibility. Returns the [2.5, 97.5]
    percentile interval from n_boot bootstrap resamples.
    """
    n = len(x)
    rhos = []
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        r, _ = spearmanr(x[idx], y[idx])
        rhos.append(r)
    return np.percentile(rhos, [2.5, 97.5])

# =============================================================================
# THREAD 6: Age-Metallicity Coherence
# =============================================================================

def test_thread_6(df):
    """Thread 6: Age-Metallicity Coherence.

    TEP prediction: rho(age_ratio, metallicity) > 0.
    If TEP causes galaxies in deeper potentials to accumulate more
    proper time, those galaxies will appear both older (higher age_ratio)
    AND more metal-enriched (higher metallicity). This creates a positive
    correlation between age_ratio and metallicity that is a hallmark of
    a single underlying temporal mechanism.

    Under standard physics, the mass-age and mass-metallicity relations
    could produce a spurious correlation, but the multi-property sample
    already controls for metallicity quality (uncertainty < 0.5 dex).

    Significance criterion: 95% bootstrap CI on rho excludes zero.
    """
    print_status("\n" + "=" * 50, "INFO")
    print_status("THREAD 6: Age-Metallicity Coherence", "INFO")
    print_status("=" * 50, "INFO")
    print_status("", "INFO")
    
    age_ratio = df['age_ratio'].values
    met = df['met'].values
    
    # Raw Spearman correlation between age_ratio and metallicity
    rho, p = spearmanr(age_ratio, met)
    ci = bootstrap_correlation(age_ratio, met)
    
    # Significant if the entire 95% CI is above zero (positive correlation)
    significant = bool(ci[0] > 0)
    
    print_status(f"N = {len(df)}", "INFO")
    print_status(f"rho(age_ratio, metallicity) = {rho:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]", "INFO")
    print_status(f"p = {p:.2e}", "INFO")
    print_status("", "INFO")
    print_status(f"★ STATISTICALLY SIGNIFICANT: {significant}", "INFO")
    
    if significant:
        print_status("", "INFO")
        print_status("INTERPRETATION:", "INFO")
        print_status("  Older-appearing galaxies are more metal-enriched.", "INFO")
        print_status("  This is consistent with TEP: enhanced time → more enrichment.", "INFO")
    
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
    """Thread 7: Multi-Property Split Test.

    This is the strongest coherence test for TEP. The sample is split
    at the median Gamma_t into two groups:
      - Low Gamma_t:  halos below the median -> less enhanced proper time
      - High Gamma_t: halos above the median -> more enhanced proper time

    TEP predicts that the high-Gamma_t group should have:
      - Higher age_ratio  (older-appearing stellar populations)
      - Higher metallicity (more chemical enrichment time)
      - Higher dust content (more dust production time)

    All three shifts must be in the TEP-predicted direction simultaneously.
    A coherent multi-property shift is extremely unlikely to arise from
    a single confounding variable (e.g. mass alone could drive one or two
    properties but not all three in the predicted direction with high
    significance).

    Statistical test: one-sided Mann-Whitney U for each property.
    Significance threshold: p < 0.001 for each property.
    Overall: all three must be individually significant.
    """
    print_status("\n" + "=" * 50, "INFO")
    print_status("THREAD 7: Multi-Property Split Test", "INFO")
    print_status("=" * 50, "INFO")
    print_status("", "INFO")
    
    gamma_t = df['gamma_t'].values
    gamma_median = np.median(gamma_t)
    
    # Split at the median Gamma_t to create equal-sized comparison groups
    high_gamma = gamma_t > gamma_median
    low_gamma = ~high_gamma
    
    print_status(f"N = {len(df)}", "INFO")
    print_status(f"Γ_t median = {gamma_median:.3f}", "INFO")
    print_status("", "INFO")
    
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
        
        # One-sided Mann-Whitney U test: H_a is that the high-Gamma_t group
        # has larger values than the low-Gamma_t group (alternative='greater').
        # Mann-Whitney is non-parametric and tests whether one distribution
        # is stochastically greater than the other.
        stat, p = mannwhitneyu(values[high_gamma], values[low_gamma], alternative='greater')
        significant = bool(p < 0.001)
        
        if not significant:
            all_significant = False
        
        print_status(f"{name}:", "INFO")
        print_status(f"  Low Γ_t:  {mean_low:.3f}", "INFO")
        print_status(f"  High Γ_t: {mean_high:.3f}", "INFO")
        print_status(f"  Diff:     {diff:+.3f}", "INFO")
        print_status(f"  p = {p:.2e}", "INFO")
        print_status(f"  ★ SIGNIFICANT: {significant}", "INFO")
        print_status("", "INFO")
        
        results[name] = {
            "mean_low_gamma": float(mean_low),
            "mean_high_gamma": float(mean_high),
            "diff": float(diff),
            "p": float(p),
            "significant": significant,
        }
    
    print_status("=" * 50, "INFO")
    print_status(f"ALL PROPERTIES SIGNIFICANT: {all_significant}", "INFO")
    
    if all_significant:
        print_status("", "INFO")
        print_status("INTERPRETATION:", "INFO")
        print_status("  All three properties (age, metallicity, dust) shift", "INFO")
        print_status("  in the direction predicted by TEP.", "INFO")
        print_status("  This is the TEP signature - coherent multi-property shift.", "INFO")
    
    return {
        "test": "Thread 7: Multi-Property Split Test",
        "n": int(len(df)),
        "gamma_median": float(gamma_median),
        "properties": results,
        "all_significant": all_significant,
        "interpretation": "TEP signature confirmed" if all_significant else "Partial or no confirmation",
    }

# =============================================================================
# MAIN
# =============================================================================

def main():
    print_status("=" * 60, "INFO")
    print_status("STEP 7: Threads 6-7 - Coherence Tests", "INFO")
    print_status("=" * 60, "INFO")
    
    data_path = INTERIM_PATH / "step_002_uncover_multi_property_sample_tep.csv"
    if not data_path.exists():
        print_status("ERROR: Input file not found. Run step_001 and step_002 first.", "ERROR")
        return
    df = pd.read_csv(data_path)
    print_status(f"Loaded multi-property sample: N = {len(df)}", "INFO")
    
    result_6 = test_thread_6(df)
    with open(OUTPUT_PATH / f"step_{STEP_NUM}_thread6_age_metallicity.json", "w") as f:
        json.dump(result_6, f, indent=2)
    
    result_7 = test_thread_7(df)
    with open(OUTPUT_PATH / f"step_{STEP_NUM}_thread7_multi_property.json", "w") as f:
        json.dump(result_7, f, indent=2)
    
    summary = {
        'step': STEP_NUM, 'name': STEP_NAME,
        'thread6_age_metallicity': result_6,
        'thread7_multi_property': result_7,
    }
    with open(OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print_status("\n" + "=" * 50, "INFO")
    print_status("SUMMARY", "INFO")
    print_status("=" * 50, "INFO")
    print_status(f"Thread 6 (Age-Metallicity): {'✓' if result_6['significant'] else '✗'}", "INFO")
    print_status(f"Thread 7 (Multi-Property):  {'✓' if result_7['all_significant'] else '✗'}", "INFO")
    print_status("", "INFO")
    print_status(f"Step {STEP_NUM} complete.", "INFO")

if __name__ == "__main__":
    main()
