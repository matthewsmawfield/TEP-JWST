#!/usr/bin/env python3
"""
TEP-JWST: Photometry Anomalies Pipeline

Runs analyses related to dust, UV slopes, mass-to-light ratios, and SFR-age consistency.
Targeted at observers focused on SED fitting and photometric properties.

Usage:
    python scripts/run_photometry_anomalies.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STEPS_DIR = PROJECT_ROOT / "scripts" / "steps"

PHOTOMETRY_STEPS = [
    # Data and model
    "step_001_uncover_load.py",
    "step_002_tep_model.py",
    
    # Dust and SED analysis
    "step_006_thread5_z8_dust.py",
    "step_014_jwst_uv_slope.py",
    "step_017_ml_ratio.py",
    "step_030_z8_dust_prediction.py",
    "step_129_dust_models.py",
    
    # SFR and age consistency
    "step_045_sfr_age_consistency.py",
    "step_044_metallicity_age_decoupling.py",
    
    # Multi-survey replication
    "step_032_ceers_replication.py",
    "step_034_cosmosweb_replication.py",
    "step_153_cosmos2025_sed_analysis.py",
    "step_157_cosmos2025_ssfr_inversion.py",
    
    # Summary
    "step_008_summary.py",
]

def run_step(script_name):
    """Run a single step script."""
    script_path = STEPS_DIR / script_name
    print(f"\n{'='*70}")
    print(f"PHOTOMETRY: {script_name}")
    print(f"{'='*70}\n")
    
    import subprocess
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=False,
    )
    
    return result.returncode == 0

def main():
    print("="*70)
    print("TEP-JWST: PHOTOMETRY ANOMALIES PIPELINE")
    print("="*70)
    print(f"\nRunning {len(PHOTOMETRY_STEPS)} steps")
    print("Dust, UV slopes, M/L ratios, and SFR-age consistency.\n")
    
    success_count = 0
    for step in PHOTOMETRY_STEPS:
        if run_step(step):
            success_count += 1
        else:
            print(f"\nPipeline stopped at {step}")
            break
    
    print("\n" + "="*70)
    print("PHOTOMETRY PIPELINE COMPLETE")
    print("="*70)
    print(f"Steps completed: {success_count}/{len(PHOTOMETRY_STEPS)}")

if __name__ == "__main__":
    main()
