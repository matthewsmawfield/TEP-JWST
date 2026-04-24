"""
Step 164: UNCOVER z=9-12 Null Branch Audit

This step audits the UNCOVER z=9-12 sample to assess null-branch status
and provide context for the stacked reddening surrogate analysis (step_165).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.p_value_utils import safe_json_default

STEP_NUM = "164"
STEP_NAME = "uncover_z9_null_audit"
INTERIM_PATH = PROJECT_ROOT / "results" / "interim"
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"
LOGS_PATH = PROJECT_ROOT / "logs"
for path in [INTERIM_PATH, OUTPUT_PATH, LOGS_PATH]:
    path.mkdir(parents=True, exist_ok=True)

logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")
set_step_logger(logger)

INPUT_CSV = INTERIM_PATH / "step_152_uncover_dr4_full_sps.csv"


def main():
    print_status("Step 164: UNCOVER z=9-12 Null Branch Audit", "STEP")
    
    # Load UNCOVER DR4 data
    if not INPUT_CSV.exists():
        print_status(f"Input not found: {INPUT_CSV}", "WARN")
        results = {
            "test": "Step 164: UNCOVER z=9-12 Null Branch Audit",
            "status": "skipped",
            "reason": "Input data not available",
            "z9_12_sample": {"n_total": 0, "n_with_dust": 0, "n_with_mass": 0},
            "null_audit_context": None
        }
        with open(OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json", "w") as f:
            json.dump(results, f, indent=2, default=safe_json_default)
        print_status("Step 164 complete (skipped)", "SUCCESS")
        return
    
    df = pd.read_csv(INPUT_CSV)
    print_status(f"Loaded UNCOVER DR4: N={len(df)}", "INFO")
    
    # z=9-12 sample
    z_col = 'z_phot' if 'z_phot' in df.columns else 'z'
    z9_12 = df[(df[z_col] >= 9) & (df[z_col] <= 12)].copy()
    n_z9_12 = len(z9_12)
    print_status(f"z=9-12 sample: N={n_z9_12}", "INFO")
    
    # Assess data availability
    dust_col = 'dust_av' if 'dust_av' in z9_12.columns else ('dust2' if 'dust2' in z9_12.columns else None)
    n_with_dust = int(z9_12[dust_col].notna().sum()) if dust_col else 0
    n_with_mass = int(z9_12['log_Mstar'].notna().sum()) if 'log_Mstar' in z9_12.columns else 0
    n_with_teff = int(z9_12['t_eff'].notna().sum()) if 't_eff' in z9_12.columns else 0
    
    # Null branch assessment
    is_null_branch = n_with_dust < 5
    
    results = {
        "test": "Step 164: UNCOVER z=9-12 Null Branch Audit",
        "status": "complete",
        "z9_12_sample": {
            "n_total": n_z9_12,
            "n_with_dust": n_with_dust,
            "n_with_mass": n_with_mass,
            "n_with_teff": n_with_teff
        },
        "null_audit_context": {
            "is_null_branch": bool(is_null_branch),
            "reason": "Insufficient dust measurements for L1-style correlation" if is_null_branch else "Adequate data for analysis",
            "recommendation": "Use stacked reddening surrogate (step_165)" if is_null_branch else "Direct analysis possible"
        }
    }
    
    output_file = OUTPUT_PATH / f"step_{STEP_NUM}_{STEP_NAME}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=safe_json_default)
    
    print_status(f"Saved: {output_file}", "SUCCESS")
    print_status(f"z=9-12: N={n_z9_12}, with dust={n_with_dust}, null_branch={is_null_branch}", "INFO")
    print_status("Step 164 complete.", "SUCCESS")


if __name__ == "__main__":
    main()
