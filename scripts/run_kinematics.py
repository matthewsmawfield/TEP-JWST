#!/usr/bin/env python3
"""
TEP-JWST: Kinematics & Dynamical Mass Pipeline

Runs analyses related to velocity dispersions, dynamical masses, and
resolved colour gradients. Targeted at dynamicists and spectroscopists.
This is the crucial L4 evidence that breaks the mass-proxy circularity.

Key Outputs:
  - Live regime-level resolution of M*/M_dyn anomalies via TEP isochrony correction (L4)
  - Kinematic screening signatures (colour gradients)
  - Optional Balmer-decrement dust follow-up when the merged DJA products are available
  - JADES DR4 and DJA spectroscopic ingestion

Usage:
    python scripts/run_kinematics.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.pipeline_runner import run_pipeline

KINEMATICS_STEPS = [
    # ── Data foundation ───────────────────────────────────────────────────
    "step_001_uncover_load.py",            # UNCOVER DR4 load
    "step_002_tep_model.py",               # Γ_t computation

    # ── L4: Dynamical mass comparison ────────────────────────────────────────
    "step_117_dynamical_mass_comparison.py", # live regime-level kinematic consistency test
    "step_143_mass_proxy_breaker.py",      # 3-test degeneracy break
    "step_159_mass_measurement_bias.py",   # TEP mass bias quantification

    # ── Spectroscopic validation ───────────────────────────────────────────
    "step_035_spectroscopic_validation.py", # Primary spectroscopic test
    "step_036_spectroscopic_refinement.py", # Simpson's paradox check
    "step_158_dja_balmer_decrement.py",    # external-catalog Balmer follow-up if merged products exist

    # ── Resolved kinematics and gradients ──────────────────────────────
    "step_037_resolved_gradients.py",      # Resolved core screening (JADES)
    "step_095_lrd_core_halo_mass.py",      # LRD core-halo mass derivation
    "step_139_colour_gradient_steiger.py", # Steiger Z: t_eff vs M*

    # ── Multi-survey spectroscopic ingestion ───────────────────────────
    "step_149_jades_dr4_ingestion.py",     # JADES DR4 (2,858 spec-z)
    "step_150_dja_nirspec_merged.py",      # DJA NIRSpec Merged v4.4
    "step_151_dja_ceers_crossmatch.py",    # DJA × CEERS+UNCOVER SED (776 z>5)
    "step_152_uncover_dr4_full_sps.py",    # UNCOVER DR4 full SPS for late-z audit
    "step_155_jades_dr5_morphology.py",    # JADES DR5 morphology support
    "step_164_uncover_z9_null_audit.py",   # UNCOVER z=9-12 null-branch audit
    "step_169_dja_sigma_pilot.py",         # DJA pilot sigma extraction from individual public spectra

    # ── Blind validation ───────────────────────────────────────────────────
    "step_119_blind_validation.py",        # Blind validation protocol
]


def main():
    results = run_pipeline(
        pipeline_name="KINEMATICS & DYNAMICAL MASS",
        steps=KINEMATICS_STEPS,
        description="Velocity dispersions, dynamical masses, spectroscopic validation",
    )
    print("\nKey outputs:")
    print("  results/outputs/step_117_*.json  — M*/M_dyn resolution (L4)")
    print("  results/outputs/step_037_*.json  — Resolved colour gradients (L2)")
    print("  results/outputs/step_149_*.json  — JADES DR4 spec-z ingestion")
    print("  results/outputs/step_158_*.json  — Balmer decrement dust test")
    print("  results/outputs/step_169_*.json  — DJA pilot sigma extraction + Balmer-vs-sigma audit")

    n_fail = sum(1 for r in results if r["status"] != "PASS")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
