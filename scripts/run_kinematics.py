#!/usr/bin/env python3
"""
TEP-JWST: Kinematics & Dynamical Mass Pipeline

Runs analyses related to velocity dispersions, dynamical masses, and resolved gradients.
Targeted at dynamicists and spectroscopists. This is the crucial L4 evidence
that breaks the mass-proxy circularity.

Key Outputs:
  - Resolution of 11/11 M*/M_dyn > 1 impossible galaxies
  - Kinematic screening signatures
  - Core-halo mass gradients

Usage:
    python scripts/run_kinematics.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STEPS_DIR = PROJECT_ROOT / "scripts" / "steps"

KINEMATICS_STEPS = [
    # Data and model
    "step_001_uncover_load.py",
    "step_002_tep_model.py",
    
    # Core L4: Dynamical mass comparison
    "step_117_dynamical_mass_comparison.py",
    "step_143_mass_proxy_breaker.py",
    
    # Spectroscopic validation
    "step_035_spectroscopic_validation.py",
    "step_036_spectroscopic_refinement.py",
    "step_158_dja_balmer_decrement.py",
    
    # Resolved kinematics
    "step_037_resolved_gradients.py",
    "step_095_lrd_core_halo_mass.py",
    "step_139_colour_gradient_steiger.py",
    
    # New spectroscopic datasets
    "step_149_jades_dr4_ingestion.py",
    "step_150_dja_nirspec_merged.py",
    "step_151_dja_ceers_crossmatch.py",
    
    # Validation
    "step_119_blind_validation.py",
]

def run_step(script_name):
    """Run a single step script."""
    script_path = STEPS_DIR / script_name
    print(f"\n{'='*70}")
    print(f"KINEMATICS: {script_name}")
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
    print("TEP-JWST: KINEMATICS & DYNAMICAL MASS PIPELINE")
    print("="*70)
    print(f"\nRunning {len(KINEMATICS_STEPS)} steps")
    print("Velocity dispersions, dynamical masses, and spectroscopic validation.\n")
    
    success_count = 0
    for step in KINEMATICS_STEPS:
        if run_step(step):
            success_count += 1
        else:
            print(f"\nPipeline stopped at {step}")
            break
    
    print("\n" + "="*70)
    print("KINEMATICS PIPELINE COMPLETE")
    print("="*70)
    print(f"Steps completed: {success_count}/{len(KINEMATICS_STEPS)}")
    print(f"\nKey outputs:")
    print(f"  - results/outputs/step_117_*.json (M*/M_dyn resolution)")
    print(f"  - results/outputs/step_037_*.json (Resolved gradients)")
    print(f"  - results/outputs/step_149_*.json (JADES DR4 spec-z)")

if __name__ == "__main__":
    main()
