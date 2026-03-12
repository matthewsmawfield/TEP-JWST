#!/usr/bin/env python3
"""
Step 116: Spectroscopic Age Tension Prediction Framework

Quantifies the theoretical expected reduction in the "anomalous early
galaxy" age tension using the TEP chronological enhancement factor.

This step does NOT use empirical measurements from literature, but rather
demonstrates the mathematical property of the TEP model:
  t_true = t_apparent / Gamma_t

It quantifies how much tension is relieved for hypothetical massive galaxies
at high redshift.
"""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.p_value_utils import safe_json_default
from scripts.utils.tep_model import compute_gamma_t, stellar_to_halo_mass_behroozi_like

STEP_NUM = "116"
STEP_NAME = "spectroscopic_age_tension_prediction"
LOGS_PATH = PROJECT_ROOT / "logs"
LOGS_PATH.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")
set_step_logger(logger)

RESULTS_DIR = PROJECT_ROOT / "results" / "outputs"


def run():
    print_status("=" * 60, "TITLE")
    print_status(f"STEP {STEP_NUM}: TEP Age Tension Reduction Framework", "TITLE")
    print_status("=" * 60, "TITLE")

    # Define a grid of extreme conditions where age tension is reported
    z_grid = [5.0, 7.0, 9.0, 11.0]
    log_m_grid = [9.0, 10.0, 10.5, 11.0]
    
    # Assume Planck18 cosmic age roughly
    # z=5 ~ 1.15 Gyr, z=7 ~ 0.77 Gyr, z=9 ~ 0.54 Gyr, z=11 ~ 0.42 Gyr
    cosmic_ages = {5.0: 1.15, 7.0: 0.77, 9.0: 0.54, 11.0: 0.42}
    
    results = []
    
    for z in z_grid:
        for log_m in log_m_grid:
            log_mh = stellar_to_halo_mass_behroozi_like(log_m, z)
            gamma_t = compute_gamma_t(log_mh, z)
            
            t_univ = cosmic_ages[z]
            
            # What happens to an apparent age that equals the age of the Universe?
            # i.e. maximally tense case
            apparent_age_max = t_univ 
            true_age_tep = apparent_age_max / gamma_t
            
            results.append({
                "z": z,
                "log_mstar": log_m,
                "log_mh": float(log_mh),
                "t_univ_Gyr": t_univ,
                "gamma_t": float(gamma_t),
                "apparent_age_Gyr": apparent_age_max,
                "true_age_tep_Gyr": float(true_age_tep),
                "age_reduction_pct": float((1 - 1/gamma_t) * 100)
            })
            
            print_status(
                f"z={z}, M*={log_m} -> Γ_t={gamma_t:.2f}. "
                f"Max age {t_univ}Gyr reduced to {true_age_tep:.2f}Gyr", 
                "INFO"
            )

    mean_reduction = np.mean([r["age_reduction_pct"] for r in results if r["gamma_t"] > 1])
    
    print_status(f"\nMean age reduction for massive systems: {mean_reduction:.1f}%", "SUCCESS")

    output = {
        "step": STEP_NUM,
        "description": "Theoretical Spectroscopic Age Tension Reduction",
        "is_model_prediction": True,
        "grid_results": results,
        "summary": {
            "mean_reduction_pct": float(mean_reduction),
            "mechanism": "t_true = t_apparent / Gamma_t"
        }
    }
    
    out_path = RESULTS_DIR / f"step_{STEP_NUM}_literature_spectroscopic_ages.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=safe_json_default)
        
    print_status(f"Saved to {out_path.name}", "SUCCESS")
    return output


if __name__ == "__main__":
    run()
