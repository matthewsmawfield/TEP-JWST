#!/usr/bin/env python3
"""
TEP-JWST Step 45: Posterior Predictive Checks (Dust Correlation)

Tests whether the observed mass-dust correlation at z > 8 is a byproduct of
SED fitting priors or likelihood geometry.

Methodology:
1. Generate synthetic galaxy populations drawn from the Prior (Mass, Dust, Z).
   - Prior: Typically flat or log-normal in Mass, Dust often exp-declining or flat.
   - Crucially: Assume Mass and Dust are UNCORRELATED in the prior.
2. Apply observational selection cuts (Depth limits).
3. Measure the resulting "Prior-Induced" correlation.
4. Compare observed correlation (+0.56) to the Prior-Induced distribution.
   - If Prior produces rho ~ 0, and we see +0.56, it's data-driven.
   - If Prior produces rho ~ 0.5, our result is an artifact.

Author: Matthew L. Smawfield
Date: January 2026
"""

import sys
import numpy as np
np.random.seed(42)
import pandas as pd
from scipy import stats
from pathlib import Path
import json
import warnings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.p_value_utils import format_p_value, safe_json_default

STEP_NUM = "45"
STEP_NAME = "posterior_predictive"
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"
INTERIM_PATH = PROJECT_ROOT / "results" / "interim"

def simulate_prior_correlation(n_samples=5000):
    """
    Simulate the correlation induced by standard SED fitting priors and selection effects.
    
    Standard Assumptions for High-z SED Fitting (e.g., Prospector/Bagpipes):
    - Mass: Uniform in log(M) or slightly declining mass function.
    - Dust (Av): Exponential decline P(Av) ~ exp(-Av/tau) or Uniform[0, 4].
    - Correlation: Priors usually assume Mass and Dust are INDEPENDENT.
    
    Selection Effect:
    - Flux limit ~ Mass * exp(-Av).
    - We only detect sources above a flux threshold.
    - High Mass + High Dust = Detectable.
    - Low Mass + High Dust = Undetectable (Faint).
    - Low Mass + Low Dust = Detectable.
    - Result: Selection cuts out the "Low Mass + High Dust" corner.
    - This artificially CREATES a positive correlation!
    """
    
    # 1. Draw from Prior (Independent)
    # Mass Function: Schechter-like (steep faint end)
    # log_M ~ Uniform[7, 11] for simplicity of broad prior coverage
    log_m = np.random.uniform(7, 11, n_samples)
    
    # Dust Prior: Exponential (typical for galaxies)
    # P(Av) ~ exp(-Av / 1.0)
    av = np.random.exponential(scale=1.0, size=n_samples)
    # Clip to physical range [0, 5]
    av = np.clip(av, 0, 5)
    
    # 2. Apply Selection Effects (Flux Limit)
    # Flux (UV) propto Mass * exp(-Av) (roughly)
    # log(Flux) ~ log(M) - 0.4 * Av * k
    # k ~ 1 for approx UV slope
    # Let's say we detect if log(Flux) > Threshold
    
    # Threshold chosen to match observed sample size fraction roughly
    # observed z>8 sample is small fraction of total halos
    
    proxy_mag = -2.5 * log_m + av # Faint is positive mag
    # Actually let's work in Log Flux
    log_flux = log_m - 0.4 * av * 2.5 # approximation
    
    # Cut: detect top 20%
    threshold = np.percentile(log_flux, 80)
    detected = log_flux > threshold
    
    log_m_det = log_m[detected]
    av_det = av[detected]
    
    # 3. Measure Induced Correlation
    rho, p = stats.spearmanr(log_m_det, av_det)
    
    return rho

def main():
    logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=PROJECT_ROOT / "logs" / f"step_{STEP_NUM}_{STEP_NAME}.log")
    set_step_logger(logger)
    
    print_status("Running Posterior Predictive Checks...", "INFO")
    
    # Load observed rho from upstream step_05 if available
    obs_rho = 0.56  # fallback
    s05_path = OUTPUT_PATH / "step_05_thread5_z8_dust.json"
    if s05_path.exists():
        try:
            with open(s05_path) as _f:
                _s05 = json.load(_f)
            _z8r = _s05.get('z8_result', {})
            if _z8r.get('rho') is not None:
                obs_rho = float(_z8r['rho'])
                print_status(f"  Loaded obs_rho={obs_rho:.3f} from step_05", "INFO")
        except Exception as _e:
            print_status(f"  WARNING: Could not load step_05: {_e}", "WARNING")
    print_status(f"Observed Correlation: rho = {obs_rho:.3f}", "INFO")
    
    # Run Simulation
    n_trials = 1000
    prior_rhos = []
    
    print_status(f"Simulating {n_trials} mock surveys...", "INFO")
    for _ in range(n_trials):
        r = simulate_prior_correlation(n_samples=2000)
        prior_rhos.append(r)
        
    prior_rhos = np.array(prior_rhos)
    
    # Statistics
    mean_prior = np.mean(prior_rhos)
    std_prior = np.std(prior_rhos)
    p95_prior = np.percentile(prior_rhos, 95)
    
    print_status("\n--- Results ---", "INFO")
    print_status(f"Prior-Induced Correlation: rho = {mean_prior:.3f} +/- {std_prior:.3f}", "INFO")
    print_status(f"95th Percentile of Prior:  rho = {p95_prior:.3f}", "INFO")
    
    # Comparison
    # Is observed value consistent with Prior?
    z_score = (obs_rho - mean_prior) / std_prior
    
    print_status(f"Z-Score of Observation: {z_score:.1f} sigma", "INFO")
    
    conclusion = ""
    if z_score > 3:
        conclusion = "The observed correlation (+0.56) is significantly stronger than selection effects alone (+{:.2f}). Data drives the signal.".format(mean_prior)
    else:
        conclusion = "The observed correlation is consistent with selection bias."
        
    print_status(conclusion, "INFO")
    
    results = {
        "obs_rho": obs_rho,
        "prior_mean": float(mean_prior),
        "prior_std": float(std_prior),
        "prior_p95": float(p95_prior),
        "z_score": float(z_score),
        "conclusion": conclusion
    }
    
    with open(OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json", "w") as f:
        json.dump(results, f, indent=2, default=safe_json_default)
        
if __name__ == "__main__":
    main()
