#!/usr/bin/env python3
"""
Step 91: Power Analysis and Sample Size Justification

This script performs statistical power analysis to address reviewer concerns
about the sample size (specifically N=3 Red Monsters and N=32 z>8 spectroscopic).

Objectives:
1. Calculate the statistical power of the Red Monster test (N=3).
   - How likely were we to resolve 50% of the anomaly by chance?
2. Calculate the power of the spectroscopic sample (N=147).
   - What is the minimum detectable correlation strength?
3. Calculate the power of the photometric sample (N=2315).
   - Verify the "p << 10^-50" claim is robust even with conservative assumptions.
4. Estimate effective sample size (N_eff) accounting for cosmic variance.

Output:
- results/outputs/step_91_power_analysis.json
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import json
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.p_value_utils import format_p_value

# Setup logging
STEP_NUM = "91"
STEP_NAME = "power_analysis"
LOGS_PATH = PROJECT_ROOT / "logs"
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"

LOGS_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")
set_step_logger(logger)


MIN_POS_FLOAT = np.nextafter(0, 1)
MIN_LOG_FLOAT = float(np.log(MIN_POS_FLOAT))


def _load_json(path: Path):
    with open(path, 'r') as f:
        return json.load(f)


def _p_value_from_t(t_stat: float, df: float, two_tailed: bool = True):
    log_sf = float(stats.t.logsf(abs(t_stat), df))
    if two_tailed:
        log_p = float(np.log(2.0) + log_sf)
    else:
        log_p = log_sf

    p_val = None
    if log_p >= MIN_LOG_FLOAT:
        p_val = float(np.exp(log_p))

    log10_p = float(log_p / np.log(10.0))

    return {
        'p_value': p_val,
        'log10_p': log10_p,
    }


def _format_p_value(p_val, log10_p, sig_figs: int = 3):
    if p_val is not None:
        if p_val >= 1e-3:
            return f"{p_val:.{sig_figs}g}"
        return f"{p_val:.{sig_figs}e}"
    return f"10^({log10_p:.1f})"

def compute_power_pearson(n, r, alpha=0.05):
    """
    Compute statistical power for Pearson correlation.
    n: sample size
    r: effect size (correlation coefficient)
    alpha: significance level
    """
    # Fisher Z transformation
    z = 0.5 * np.log((1 + r) / (1 - r))
    z_crit = stats.norm.ppf(1 - alpha/2)  # Two-tailed
    
    # Standard error of Z
    se = 1 / np.sqrt(n - 3)
    
    # Power = P(Z > z_crit | r)
    # Z_stat = z / se
    # We want P(Z_stat > z_crit) = P((z - 0)/se > z_crit)
    # Actually, centered at z_true.
    # Power is probability that observed z is in critical region.
    # Critical region: |z_obs| > z_crit * se
    
    z_upper = z_crit * se

    p_upper = stats.norm.sf((z_upper - z) / se)
    p_lower = stats.norm.cdf((-z_upper - z) / se)

    return p_upper + p_lower

def red_monster_power_analysis():
    """
    Analyze the N=3 Red Monster case.
    The anomaly was ~0.50 SFE. TEP reduced it to ~0.34 (54% resolution).
    This is equivalent to a systematic shift, not just a correlation.
    
    Test: One-sample t-test (or paired if we had distribution).
    Hypothesis: The mean SFE reduction is > 0.
    Observed reduction: 0.16.
    Standard deviation of reduction: Need to estimate.
    
    If N=3, how large does the effect need to be to be significant?
    t = mean / (std / sqrt(n))
    """
    print_status("\n--- Red Monster Power Analysis (N=3) ---", "INFO")
    
    blue_monsters_csv = OUTPUT_PATH / "step_47_blue_monsters.csv"
    if not blue_monsters_csv.exists():
        raise FileNotFoundError(f"Required input not found: {blue_monsters_csv}")

    df = pd.read_csv(blue_monsters_csv)
    rm = df[df['name'].astype(str).str.startswith('RM')].copy()

    if len(rm) == 0:
        raise ValueError(f"No Red Monster rows found in {blue_monsters_csv}")

    n = int(len(rm))

    reductions = (rm['sfe_obs'] - rm['sfe_true']).astype(float).values
    mean_reduction = float(np.mean(reductions))
    std_reduction = float(np.std(reductions, ddof=1)) if n > 1 else 0.0
    
    # t-statistic
    t_stat = mean_reduction / (std_reduction / np.sqrt(n))
    p_dict = _p_value_from_t(t_stat, df=n - 1, two_tailed=False)
    p_val = p_dict['p_value']
    
    print_status(f"Effect size (SFE reduction): {mean_reduction}", "INFO")
    print_status(f"Assumed scatter: {std_reduction}", "INFO")
    print_status(f"t-statistic: {t_stat:.2f}", "INFO")
    print_status(
        f"p-value (one-tailed): {_format_p_value(p_val, p_dict['log10_p'])}",
        "INFO",
    )
    
    # Power: What is the probability of detecting this effect at alpha=0.05?
    # Non-centrality parameter delta = mean / (std/sqrt(n))
    # Power = P(t > t_crit | delta)
    t_crit = stats.t.ppf(0.95, df=n-1)
    power = stats.nct.sf(t_crit, df=n-1, nc=t_stat)
    
    print_status(f"Power to detect {mean_reduction:.3f} reduction (alpha=0.05): {power:.2f}", "INFO")
    
    return {
        'n': n,
        'mean_reduction': mean_reduction,
        'std_reduction': std_reduction,
        't_stat': float(t_stat),
        'p_value': format_p_value(p_val),
        'log10_p_value': float(p_dict['log10_p']),
        'power': float(power)
        ,
        'source': {
            'csv': str(blue_monsters_csv),
            'rows_used': [str(x) for x in rm['name'].tolist()],
        }
    }

def spectroscopic_power_analysis():
    """
    N = 32 at z > 8.
    Observed correlation rho ~ 0.5.
    """
    print_status("\n--- Spectroscopic Sample Power (z > 8, N=32) ---", "INFO")
    
    spectro_json = OUTPUT_PATH / "step_37c_spectroscopic_refinement.json"
    if not spectro_json.exists():
        raise FileNotFoundError(f"Required input not found: {spectro_json}")

    spec = _load_json(spectro_json)
    z8 = spec.get('bin_analysis', {}).get('z8_plus', {})
    if not z8:
        raise KeyError("Missing bin_analysis.z8_plus in step_37c_spectroscopic_refinement.json")

    n = int(z8.get('n'))
    r_obs = float(z8.get('rho'))
    
    power = compute_power_pearson(n, r_obs)
    print_status(f"Observed correlation: r = {r_obs}", "INFO")
    print_status(f"Power (alpha=0.05): {power:.2f}", "INFO")
    
    # Minimum detectable effect
    rs = np.linspace(0.1, 0.9, 50)
    powers = [compute_power_pearson(n, r) for r in rs]
    min_detectable = rs[np.where(np.array(powers) > 0.8)[0][0]]
    print_status(f"Minimum detectable effect (80% power): r > {min_detectable:.2f}", "INFO")
    
    return {
        'n': n,
        'r_obs': r_obs,
        'power': float(power),
        'min_detectable_r': float(min_detectable),
        'source': {
            'json': str(spectro_json),
            'field': 'bin_analysis.z8_plus.rho',
        },
    }

def photometric_significance_check():
    """
    N = 2315.
    Claim: p << 10^-50.
    Issue: Independence. Spatial correlation reduces effective N.
    """
    print_status("\n--- Photometric Significance & Independence ---", "INFO")
    
    z8_dust_json = OUTPUT_PATH / "step_05_thread5_z8_dust.json"
    if not z8_dust_json.exists():
        raise FileNotFoundError(f"Required input not found: {z8_dust_json}")

    z8_dust = _load_json(z8_dust_json)
    z8_result = z8_dust.get('z8_result', {})
    if not z8_result:
        raise KeyError("Missing z8_result in step_05_thread5_z8_dust.json")

    n_nominal = int(z8_result.get('n'))
    r_main = float(z8_result.get('rho'))

    bootstrap_json = OUTPUT_PATH / "step_97_bootstrap_validation.json"
    n_eff = None
    n_eff_source = None

    if bootstrap_json.exists():
        boot = _load_json(bootstrap_json)
        eff = boot.get('effective_sample_size', {})
        if eff:
            n_eff = float(eff.get('n_eff'))
            n_eff_source = 'effective_sample_size.n_eff'

    if n_eff is None:
        n_eff = float(n_nominal / 10.0)
        n_eff_source = 'fallback:n_nominal/10'

    t_nom = float(r_main * np.sqrt((n_nominal - 2) / (1 - r_main**2)))
    p_nom_dict = _p_value_from_t(t_nom, df=n_nominal - 2, two_tailed=True)

    t_eff = float(r_main * np.sqrt((n_eff - 2) / (1 - r_main**2)))
    p_eff_dict = _p_value_from_t(t_eff, df=n_eff - 2, two_tailed=True)

    print_status(f"Nominal N: {n_nominal}", "INFO")
    print_status(
        f"  p-value (r={r_main:.3f}, two-tailed): {_format_p_value(p_nom_dict['p_value'], p_nom_dict['log10_p'])}",
        "INFO",
    )

    print_status(f"Effective N_eff: {n_eff:.1f}", "INFO")
    print_status(
        f"  p-value (r={r_main:.3f}, two-tailed): {_format_p_value(p_eff_dict['p_value'], p_eff_dict['log10_p'])}",
        "INFO",
    )

    return {
        'n_nominal': n_nominal,
        'n_eff': float(n_eff),
        'r_obs': r_main,
        't_nominal': t_nom,
        't_effective': t_eff,
        'p_nominal': format_p_value(p_nom_dict['p_value']),
        'p_effective': format_p_value(p_eff_dict['p_value']),
        'log10_p_nominal': float(p_nom_dict['log10_p']),
        'log10_p_effective': float(p_eff_dict['log10_p']),
        'source': {
            'rho_json': str(z8_dust_json),
            'rho_field': 'z8_result.rho',
            'n_field': 'z8_result.n',
            'n_eff_json': str(bootstrap_json) if bootstrap_json.exists() else None,
            'n_eff_field': n_eff_source,
        },
    }

def run_analysis():
    results = {}
    
    results['red_monsters'] = red_monster_power_analysis()
    results['spectroscopic'] = spectroscopic_power_analysis()
    results['photometric'] = photometric_significance_check()
    
    with open(OUTPUT_PATH / "step_91_power_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print_status(f"\nResults saved to {OUTPUT_PATH / 'step_91_power_analysis.json'}", "INFO")

if __name__ == "__main__":
    run_analysis()
