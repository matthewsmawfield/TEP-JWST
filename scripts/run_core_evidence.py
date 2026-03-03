#!/usr/bin/env python3
"""
TEP-JWST: Core Evidence Fast-Path

Runs only the four independent lines of evidence (L1-L4) in ~5 minutes.
This is the recommended entry point for first-time users and reviewers.

Lines of Evidence:
  L1: Dust-Gamma_t correlation (z>8)
  L2: Inside-out core screening (JADES)
  L3: Mass-sSFR inversion at z>7
  L4: Dynamical mass resolution (M*/M_dyn > 1 cases)

Usage:
    python scripts/run_core_evidence.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STEPS_DIR = PROJECT_ROOT / "scripts" / "steps"
OUTPUTS_DIR = PROJECT_ROOT / "results" / "outputs"

CORE_EVIDENCE_STEPS = [
    # Data loading and model
    "step_001_uncover_load.py",
    "step_002_tep_model.py",
    
    # L1: Dust-Gamma_t correlation (primary)
    "step_006_thread5_z8_dust.py",
    "step_030_z8_dust_prediction.py",
    
    # L2: Core screening
    "step_037_resolved_gradients.py",
    "step_139_colour_gradient_steiger.py",
    
    # L3: Mass-sSFR inversion
    "step_004_thread1_z7_inversion.py",
    "step_157_cosmos2025_ssfr_inversion.py",
    
    # L4: Kinematic/Dynamical mass
    "step_117_dynamical_mass_comparison.py",
    "step_143_mass_proxy_breaker.py",
    
    # Summary
    "step_140_evidence_tier_summary.py",
]

def run_step(script_name):
    """Run a single step script."""
    script_path = STEPS_DIR / script_name
    print(f"\n{'='*70}")
    print(f"CORE EVIDENCE: {script_name}")
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
    print("TEP-JWST: CORE EVIDENCE FAST-PATH (L1-L4)")
    print("="*70)
    print(f"\nRunning {len(CORE_EVIDENCE_STEPS)} steps (~5 minutes)")
    print("This validates the four independent lines of evidence.\n")
    
    success_count = 0
    for step in CORE_EVIDENCE_STEPS:
        if run_step(step):
            success_count += 1
        else:
            print(f"\nPipeline stopped at {step}")
            break
    
    print("\n" + "="*70)
    print("CORE EVIDENCE COMPLETE")
    print("="*70)
    print(f"Steps completed: {success_count}/{len(CORE_EVIDENCE_STEPS)}")
    print(f"\nKey outputs:")
    print(f"  - results/outputs/step_006_*.json (L1: Dust correlation)")
    print(f"  - results/outputs/step_037_*.json (L2: Core screening)")
    print(f"  - results/outputs/step_004_*.json (L3: sSFR inversion)")
    print(f"  - results/outputs/step_117_*.json (L4: Dynamical masses)")

if __name__ == "__main__":
    main()
