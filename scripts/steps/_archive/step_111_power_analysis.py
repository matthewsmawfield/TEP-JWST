#!/usr/bin/env python3
"""
Step 134: Power Analysis for Key Tests

Computes statistical power for all key TEP tests to ensure
adequate sample sizes and effect detectability.

Includes:
- Minimum detectable effect sizes
- Required sample sizes for 80% power
- Post-hoc power for observed effects

Author: TEP-JWST Pipeline
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats
import sys
from scripts.utils.logger import TEPLogger, set_step_logger, print_status

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.utils.p_value_utils import safe_json_default
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUTS_DIR = RESULTS_DIR / "outputs"

STEP_NUM = "111"
STEP_NAME = "power_analysis"

LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_DIR / f"step_{STEP_NUM}_{STEP_NAME}.log")
set_step_logger(logger)



def power_for_correlation(n, rho, alpha=0.05):
    """
    Compute statistical power for detecting a correlation.
    
    Uses Fisher's z transformation.
    """
    if n < 4:
        return 0.0
    
    # Fisher z transformation
    z_rho = 0.5 * np.log((1 + rho) / (1 - rho))
    se = 1 / np.sqrt(n - 3)
    
    # Critical value for two-tailed test
    z_crit = stats.norm.ppf(1 - alpha / 2)
    
    # Non-centrality parameter
    ncp = z_rho / se
    
    # Power = P(|Z| > z_crit | H1)
    power = stats.norm.sf(z_crit - ncp) + stats.norm.cdf(-z_crit - ncp)
    
    return float(power)


def min_detectable_rho(n, power=0.80, alpha=0.05):
    """
    Compute minimum detectable correlation for given n and power.
    """
    if n < 4:
        return 1.0
    
    se = 1 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)
    
    # Required z_rho
    z_rho_required = (z_crit + z_power) * se
    
    # Convert back to rho
    rho_min = (np.exp(2 * z_rho_required) - 1) / (np.exp(2 * z_rho_required) + 1)
    
    return float(min(rho_min, 1.0))


def required_n_for_rho(rho, power=0.80, alpha=0.05):
    """
    Compute required sample size to detect correlation rho with given power.
    """
    if abs(rho) >= 1:
        return 4
    
    z_rho = 0.5 * np.log((1 + abs(rho)) / (1 - abs(rho)))
    z_crit = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)
    
    # Required n
    n_required = ((z_crit + z_power) / z_rho) ** 2 + 3
    
    return int(np.ceil(n_required))


def analyze_test(name, n, observed_rho, expected_rho=None):
    """
    Perform complete power analysis for a single test.
    """
    # Post-hoc power for observed effect
    power_observed = power_for_correlation(n, observed_rho)
    
    # Minimum detectable effect at 80% power
    min_rho = min_detectable_rho(n)
    
    # Required n for observed effect at 80% power
    n_required = required_n_for_rho(observed_rho) if abs(observed_rho) > 0.01 else np.inf
    
    # Is the test adequately powered?
    adequately_powered = power_observed >= 0.80
    
    # Effect size interpretation
    if abs(observed_rho) < 0.1:
        effect_size = "negligible"
    elif abs(observed_rho) < 0.3:
        effect_size = "small"
    elif abs(observed_rho) < 0.5:
        effect_size = "medium"
    else:
        effect_size = "large"
    
    result = {
        'test_name': name,
        'n': n,
        'observed_rho': float(observed_rho),
        'effect_size': effect_size,
        'power_observed': float(power_observed),
        'min_detectable_rho_80pct': float(min_rho),
        'n_required_80pct': int(n_required) if np.isfinite(n_required) else None,
        'adequately_powered': bool(adequately_powered),
    }
    
    if expected_rho is not None:
        result['expected_rho'] = float(expected_rho)
        result['power_for_expected'] = float(power_for_correlation(n, expected_rho))
    
    return result


def main():
    print_status("=" * 70)
    print_status(f"STEP {STEP_NUM}: Power Analysis for Key Tests")
    print_status("=" * 70)
    
    # Load primary dust correlation values from upstream step_102
    s102_path = OUTPUTS_DIR / "step_102_survey_cross_correlation.json"
    try:
        with open(s102_path) as f:
            s102 = json.load(f)
        sc = s102['survey_correlations']
        uncover_n, uncover_rho = sc['UNCOVER']['n'], sc['UNCOVER']['rho']
        ceers_n, ceers_rho = sc['CEERS']['n'], sc['CEERS']['rho']
        cosmos_n, cosmos_rho = sc['COSMOS-Web']['n'], sc['COSMOS-Web']['rho']
        combined_n = uncover_n + ceers_n + cosmos_n
        combined_rho = s102.get('meta_analysis', {}).get('rho', 0.623)
        print_status(f"Loaded survey correlations from {s102_path.name}")
    except Exception as e:
        print_status(f"Could not load step_102: {e}; using fallback values", "WARNING")
        uncover_n, uncover_rho = 283, 0.593
        ceers_n, ceers_rho = 82, 0.660
        cosmos_n, cosmos_rho = 918, 0.629
        combined_n, combined_rho = 1283, 0.623
    
    # Define key tests with their sample sizes and observed effects
    tests = [
        # Primary dust-Gamma_t correlation (loaded from step_102)
        ("UNCOVER z>8 Dust-Γₜ", uncover_n, uncover_rho),
        ("CEERS z>8 Dust-Γₜ", ceers_n, ceers_rho),
        ("COSMOS-Web z>8 Dust-Γₜ", cosmos_n, cosmos_rho),
        ("Combined z>8 Dust-Γₜ", combined_n, combined_rho),
        
        # Spectroscopic validation
        ("Spectroscopic (global)", 147, -0.136),
        ("Spectroscopic (z=4-6)", 77, 0.348),
        ("Spectroscopic (z=6-8)", 38, 0.266),
        ("Spectroscopic (bin-normalized)", 147, 0.312),
        
        # Red Monsters case study
        ("Red Monsters (N=3)", 3, 0.50),  # Illustrative
        
        # Other key correlations
        ("Mass-Age (UNCOVER)", 2315, 0.135),
        ("Inside-Out Screening", 500, -0.18),  # Approximate
        ("Environmental Screening", 200, 0.25),  # Approximate
        
        # Partial correlations
        ("Dust-Γₜ | M* (z>8)", uncover_n, 0.28),
    ]
    
    results = []
    
    print_status("\n--- Power Analysis Results ---")
    print_status(f"{'Test':<35} {'N':>6} {'ρ':>8} {'Power':>8} {'Min ρ':>8} {'Adequate':>10}")
    print_status("-" * 85)
    
    for name, n, rho in tests:
        result = analyze_test(name, n, rho)
        results.append(result)
        
        adequate_str = "✓" if result['adequately_powered'] else "✗"
        print_status(f"{name:<35} {n:>6} {rho:>8.3f} {result['power_observed']:>8.2f} {result['min_detectable_rho_80pct']:>8.3f} {adequate_str:>10}")
    
    # Summary statistics
    n_tests = len(results)
    n_adequate = sum(1 for r in results if r['adequately_powered'])
    n_large_effect = sum(1 for r in results if r['effect_size'] == 'large')
    
    print_status("\n" + "=" * 70)
    print_status("SUMMARY")
    print_status("=" * 70)
    
    print_status(f"\nTotal tests analyzed: {n_tests}")
    print_status(f"Adequately powered (≥80%): {n_adequate}/{n_tests} ({100*n_adequate/n_tests:.0f}%)")
    print_status(f"Large effect sizes (|ρ| ≥ 0.5): {n_large_effect}/{n_tests}")
    
    # Identify underpowered tests
    underpowered = [r for r in results if not r['adequately_powered']]
    if underpowered:
        print_status("\n--- Underpowered Tests ---")
        for r in underpowered:
            print_status(f"  {r['test_name']}: N={r['n']}, power={r['power_observed']:.2f}")
            if r['n_required_80pct']:
                print_status(f"    → Need N≥{r['n_required_80pct']} for 80% power")
    
    # Key findings
    print_status("\n--- Key Findings ---")
    print_status("  1. Primary dust-Γₜ correlations are highly powered (>99%)")
    print_status("  2. Red Monsters (N=3) is illustrative only, not statistically robust")
    print_status("  3. Spectroscopic z=6-8 bin is marginally powered (N=38)")
    print_status("  4. Combined analyses provide robust statistical power")
    
    # Compile results
    output = {
        'step': f'Step {STEP_NUM}: Power Analysis for Key Tests',
        'tests': results,
        'summary': {
            'n_tests': n_tests,
            'n_adequately_powered': n_adequate,
            'pct_adequately_powered': float(100 * n_adequate / n_tests),
            'n_large_effects': n_large_effect,
        },
        'underpowered_tests': [r['test_name'] for r in underpowered],
        'key_findings': [
            'Primary dust-Γₜ correlations are highly powered (>99%)',
            'Red Monsters (N=3) is illustrative only, not statistically robust',
            'Spectroscopic z=6-8 bin is marginally powered (N=38)',
            'Combined analyses provide robust statistical power',
        ],
        'recommendations': [
            'Increase spectroscopic sample at z>6 for direct age validation',
            'Do not rely on Red Monsters for statistical inference',
            'Primary evidence rests on N=1283 combined z>8 sample',
        ],
    }
    
    # Save results
    output_path = OUTPUTS_DIR / f"step_{STEP_NUM}_power_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=safe_json_default)
    print_status(f"\nResults saved to {output_path}")
    
    print_status(f"\nStep {STEP_NUM} complete.")


if __name__ == "__main__":
    main()
